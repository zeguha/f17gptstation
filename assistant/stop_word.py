"""Stop-word detector for best-effort barge-in.

Design goal
-----------
Allow saying "стоп/отмена/хватит" while TTS is playing to interrupt playback.

Constraints
-----------
- No full AEC: the mic hears our own TTS voice while it's playing, so there's
  no silence gap for Vosk to finalize an utterance on until playback itself
  ends — confirmed empirically (the target word only ever showed up, mixed
  with garbled TTS bleed-through, at the very end of a finalized utterance,
  well after it stopped mattering). So detection is driven by the
  *in-progress* partial hypothesis (checked every frame, matched against the
  tail of the text) rather than waiting for a finalized result. Kept
  extremely conservative (strict thresholds) since partials are less stable
  than final results.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from difflib import SequenceMatcher

from .asr_engines import VoskEngine
from .vad import Vad, VadConfig


log = logging.getLogger("assistant.stop_word")


def _match_ratio_suffix(text: str, phrase: str, *, max_suffix_words: int = 4) -> float:
    """Similarity ratio of `phrase` against the *end* of `text`.

    Mirrors `postprocess.wake_match_ratio_prefix`, but for the tail: an
    in-progress partial hypothesis grows left-to-right, so a word just said
    shows up at the end, however much unrelated text (e.g. our own TTS
    bleeding into the mic) precedes it.
    """

    words = text.split()
    phrase_words = phrase.split()
    if not words or not phrase_words:
        return 0.0

    best = 0.0
    for k in range(max(1, len(phrase_words) - 1), min(len(words), len(phrase_words) + 2) + 1):
        if k > max_suffix_words:
            break
        frag = " ".join(words[-k:])
        r = SequenceMatcher(None, frag, phrase).ratio()
        if r > best:
            best = r
    return best


@dataclass(slots=True)
class StopWordConfig:
    sample_rate: int = 16_000
    frame_ms: int = 30

    # Comma-separated list, e.g. "стоп,хватит,отмена"
    words: list[str] = None  # filled in __post_init__

    # Very strict thresholds to avoid TTS self-trigger.
    match_threshold: float = 0.92
    min_speech_ratio: float = 0.20
    # Recent-history window for the partial (fast) path: this is checked in
    # near-real-time as the word is being said, so it should reflect "was
    # there speech just now", not the whole utterance.
    confirm_window_sec: float = 0.7
    # Safety cap on how much VAD history we accumulate for the finalized
    # (slow, fallback) path — needs to comfortably cover a short word plus
    # the trailing silence Vosk needs to finalize it, not just the word.
    final_window_sec: float = 4.0
    cooldown_sec: float = 1.0

    vad_mode: int = 3
    vad_energy_threshold: float = 0.012

    def __post_init__(self) -> None:
        if self.words is None:
            self.words = ["стоп", "хватит", "отмена"]
        self.words = [w.strip().lower() for w in self.words if w and w.strip()]


class StopWordDetector:
    def __init__(self, *, cfg: StopWordConfig, engine: VoskEngine):
        """Create detector.

        `engine` should decode free-form (no `grammar=`). Grammar-constrained
        decoding (`UpdateGrammarFst`) needs a lexicon that small Vosk models
        don't ship, which makes it silently never match any word regardless
        of the grammar list — confirmed empirically, not a theoretical
        concern. We fuzzy-match the free-form text against `cfg.words`
        ourselves below, the same way wake-word detection does.
        """

        self.cfg = cfg
        self.engine = engine

        self._vad = Vad(
            VadConfig(
                sample_rate=cfg.sample_rate,
                frame_ms=cfg.frame_ms,
                webrtcvad_mode=cfg.vad_mode,
                energy_threshold=cfg.vad_energy_threshold,
            )
        )
        self._last_fire = 0.0
        self._session = self.engine.start_session()
        self._window: list[bool] = []
        self._recent_max = max(1, int(self.cfg.confirm_window_sec * 1000 / self.cfg.frame_ms))
        self._final_max = max(self._recent_max, int(self.cfg.final_window_sec * 1000 / self.cfg.frame_ms))

    @classmethod
    def from_model_path(cls, *, cfg: StopWordConfig, model_path: str) -> "StopWordDetector":
        eng = VoskEngine(model_path=model_path, sample_rate=cfg.sample_rate)
        return cls(cfg=cfg, engine=eng)

    def _speech_ratio(self, flags: list[bool]) -> float:
        if not flags:
            return 0.0
        return sum(1 for x in flags if x) / len(flags)

    def _reset_session(self) -> None:
        try:
            self._session = self.engine.start_session()
        except Exception:
            pass
        self._window.clear()

    def accept_frame(self, pcm16: bytes) -> bool:
        """Feed one PCM16 frame. Returns True if stop word detected."""

        now = time.monotonic()
        if (now - self._last_fire) < self.cfg.cooldown_sec:
            return False

        is_sp = self._vad.is_speech(pcm16)
        self._window.append(bool(is_sp))
        # Capped generously (see final_window_sec) rather than at the short
        # recent-window size: the finalized-result fallback below needs the
        # whole in-progress utterance's history, not just the last moment.
        if len(self._window) > self._final_max:
            self._window = self._window[-self._final_max :]

        self._session.accept_audio(pcm16)

        # Fast path: the in-progress partial hypothesis, checked every frame.
        # This is what actually fires during TTS playback — see module
        # docstring for why waiting on finalization doesn't work here.
        partial_fn = getattr(self._session, "partial_text", None)
        if partial_fn is not None:
            partial = partial_fn()
            if partial:
                recent_ratio = self._speech_ratio(self._window[-self._recent_max :])
                if self._check_match(partial, recent_ratio, source="partial"):
                    return True

        # Slow path / fallback: a finalized result (covers engines without
        # partial support, and utterances that do finalize naturally, e.g.
        # a stop word said into silence rather than over TTS playback).
        consume = getattr(self._session, "_consume_if_accepted", None)
        if consume is None:
            return False
        out = consume()
        if out is None or not out.text:
            return False

        full_ratio = self._speech_ratio(self._window)
        self._window.clear()  # this utterance is fully consumed either way
        return self._check_match(out.text, full_ratio, source="final")

    def _check_match(self, text: str, ratio: float, *, source: str) -> bool:
        txt = text.strip().lower()
        best = 0.0
        best_w = ""
        for w in self.cfg.words:
            # Suffix, not prefix: whether from a still-growing partial or a
            # finalized utterance that merged with TTS bleed-through, the
            # word we care about is whatever was said *last*.
            r = _match_ratio_suffix(txt, w)
            if r > best:
                best = r
                best_w = w

        if best < self.cfg.match_threshold:
            log.debug("stop candidate rejected (%s): text='%s' best=%.3f", source, txt, best)
            return False
        if ratio < self.cfg.min_speech_ratio:
            log.debug(
                "stop candidate rejected by speech_ratio=%.3f (%s) text='%s'", ratio, source, txt
            )
            return False

        self._last_fire = time.monotonic()
        log.info(
            "stop-word detected (%s): '%s' (text='%s', best=%.3f, speech_ratio=%.3f)",
            source,
            best_w,
            txt,
            best,
            ratio,
        )
        self._reset_session()
        return True
