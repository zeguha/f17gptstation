"""Stop-word detector for best-effort barge-in.

Design goal
-----------
Allow saying "стоп/отмена/хватит" while TTS is playing to interrupt playback.

Constraints
-----------
- No full AEC initially.
- Therefore: extremely conservative detection (limited grammar + strict thresholds).

We use a small local Vosk session with grammar restricted to the stop words.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .asr_engines import VoskEngine
from .postprocess import wake_match_ratio_prefix
from .vad import Vad, VadConfig


log = logging.getLogger("assistant.stop_word")


@dataclass(slots=True)
class StopWordConfig:
    sample_rate: int = 16_000
    frame_ms: int = 30

    # Comma-separated list, e.g. "стоп,хватит,отмена"
    words: list[str] = None  # filled in __post_init__

    # Very strict thresholds to avoid TTS self-trigger.
    match_threshold: float = 0.92
    min_speech_ratio: float = 0.20
    confirm_window_sec: float = 0.7
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

        Important: `engine` MUST be constructed with `grammar=cfg.words`.
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
        self._window_max = max(1, int(self.cfg.confirm_window_sec * 1000 / self.cfg.frame_ms))

    @classmethod
    def from_model_path(cls, *, cfg: StopWordConfig, model_path: str) -> "StopWordDetector":
        eng = VoskEngine(model_path=model_path, sample_rate=cfg.sample_rate, grammar=cfg.words)
        return cls(cfg=cfg, engine=eng)

    def _speech_ratio(self, flags: list[bool]) -> float:
        if not flags:
            return 0.0
        return sum(1 for x in flags if x) / len(flags)

    def accept_frame(self, pcm16: bytes) -> bool:
        """Feed one PCM16 frame. Returns True if stop word detected."""

        now = time.monotonic()
        if (now - self._last_fire) < self.cfg.cooldown_sec:
            return False

        is_sp = self._vad.is_speech(pcm16)
        self._window.append(bool(is_sp))
        if len(self._window) > self._window_max:
            self._window = self._window[-self._window_max :]

        self._session.accept_audio(pcm16)

        # Stop words are short: we rely on Vosk's utterance finalization.
        consume = getattr(self._session, "_consume_if_accepted", None)
        if consume is None:
            return False
        out = consume()
        if out is None or not out.text:
            return False

        txt = out.text.strip().lower()
        ratio = self._speech_ratio(self._window)
        best = 0.0
        best_w = ""
        for w in self.cfg.words:
            r = wake_match_ratio_prefix(txt, w)
            if r > best:
                best = r
                best_w = w

        if best < self.cfg.match_threshold:
            log.debug("stop candidate rejected: text='%s' best=%.3f", txt, best)
            return False
        if ratio < self.cfg.min_speech_ratio:
            log.debug("stop candidate rejected by speech_ratio=%.3f text='%s'", ratio, txt)
            return False

        self._last_fire = time.monotonic()
        log.info("stop-word detected: '%s' (text='%s', best=%.3f, speech_ratio=%.3f)", best_w, txt, best, ratio)

        # Reset session so the next stop can be detected cleanly.
        try:
            self._session = self.engine.start_session()
            self._window.clear()
        except Exception:
            pass
        return True
