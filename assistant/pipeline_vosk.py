"""Wake → command recording → ASR pipeline (Vosk-first).

State machine:
1) WAIT_WAKE
2) CONFIRM_WAKE (debounce, cooldown, metrics)
3) RECORD_COMMAND (VAD driven)
4) ASR_COMMAND
5) POSTPROCESS

Key property: wake phrase never leaks into the command.

We ensure this by:
- using a dedicated wake recognizer session (grammar-limited)
- clearing the audio queue and dropping a small tail after wake
- starting a new ASR session for the command
- final text safety strip for wake phrase
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional

from .asr_engines import AsrEngine, AsrResult, VoskEngine
from .audio_stream import AudioStream
from .postprocess import strip_wake_phrase, wake_match_ratio_prefix
from .vad import Vad, VadConfig


log = logging.getLogger("assistant.pipeline")


class PipelineCancelled(Exception):
    """Raised out of a blocking pipeline call when `cancel_event` is set.

    `wait_for_wake()`/`record_command()` otherwise loop with no bound while
    waiting for speech, so this is how a caller running them on a worker
    thread (e.g. via `asyncio.to_thread`) can get them to return promptly on
    shutdown instead of blocking process exit until the wake phrase is heard.
    """


class StopReason(str, Enum):
    SILENCE_TIMEOUT = "silence_timeout"
    MAX_PHRASE = "max_phrase"
    NO_SPEECH_START = "no_speech_start"
    TOO_SHORT = "too_short"
    EMBEDDED_IN_WAKE = "embedded_in_wake"


@dataclass(slots=True)
class PipelineConfig:
    # audio
    sample_rate: int = 16_000
    frame_ms: int = 30

    # wake
    wake_phrase: str = "мышка сосиска"
    wake_min_conf: float = 0.70
    wake_match_threshold: float = 0.78
    wake_cooldown_sec: float = 1.0
    wake_confirm_window_sec: float = 0.8
    # How much of the last `wake_confirm_window_sec` must be classified as speech by VAD.
    # If your VAD is too strict (common on some Macs / quiet mics), lower to 0.0–0.1.
    wake_min_speech_ratio: float = 0.10  # in confirm window
    wake_tail_drop_ms: int = 250

    # command recording
    command_start_timeout_sec: float = 5.0
    max_silence_sec: float = 0.8
    max_phrase_sec: float = 12.0
    min_phrase_sec: float = 0.35
    start_speech_min_frames: int = 3  # debounce speech start

    # safety text filter
    wake_strip_fuzzy_threshold: float = 0.80

    # VAD
    vad_mode: int = 2
    vad_energy_threshold: float = 0.010

    # Cloud mode: record command audio and return it without running local ASR.
    skip_command_asr: bool = False


@dataclass(frozen=True, slots=True)
class WakeMetrics:
    time_to_wake_sec: float
    confidence: Optional[float]
    speech_ratio: float
    wake_text: str = ""
    wake_ratio: float = 0.0
    embedded_command_text: str = ""


@dataclass(frozen=True, slots=True)
class CommandMetrics:
    duration_sec: float
    stop_reason: StopReason
    speech_frames: int
    total_frames: int


@dataclass(frozen=True, slots=True)
class Interaction:
    wake: WakeMetrics
    command: CommandMetrics
    asr: AsrResult
    final_text: str
    command_pcm16: bytes = b""
    command_sample_rate: int = 16_000


class WakeCommandPipeline:
    def __init__(
        self,
        *,
        cfg: PipelineConfig,
        audio: AudioStream,
        wake_engine: VoskEngine,
        command_engine: AsrEngine,
        cancel_event: Optional[threading.Event] = None,
    ):
        self.cfg = cfg
        self.audio = audio
        self.wake_engine = wake_engine
        self.command_engine = command_engine
        self._cancel_event = cancel_event

        self.vad = Vad(
            VadConfig(
                sample_rate=cfg.sample_rate,
                frame_ms=cfg.frame_ms,
                webrtcvad_mode=cfg.vad_mode,
                energy_threshold=cfg.vad_energy_threshold,
            )
        )

        self.frame_samples = int(cfg.sample_rate * cfg.frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2
        if self.audio.frame_bytes != self.frame_bytes:
            raise ValueError(
                "AudioStream frame size mismatch: "
                f"audio.frame_bytes={self.audio.frame_bytes} != expected={self.frame_bytes}"
            )

        self._last_wake_time: float = 0.0
        self.false_wakes: int = 0

    def _drain_for(self, seconds: float) -> None:
        """Actively read and drop audio for the given duration."""
        deadline = time.monotonic() + float(seconds)
        while time.monotonic() < deadline:
            _ = self.audio.read(timeout=0.05)

    def wait_for_wake(self) -> WakeMetrics:
        """Block until wake phrase detected (with debounce & metrics)."""

        start = time.monotonic()
        wake_session = self.wake_engine.start_session()

        # Keep a window of recent VAD decisions to confirm wake happened on speech, not noise.
        window_frames: Deque[bool] = deque(maxlen=max(1, int(self.cfg.wake_confirm_window_sec * 1000 / self.cfg.frame_ms)))

        while True:
            if self._cancel_event is not None and self._cancel_event.is_set():
                raise PipelineCancelled("wait_for_wake: cancelled")

            ch = self.audio.read(timeout=0.5)
            if ch is None:
                continue

            # cooldown
            now = time.monotonic()
            if (now - self._last_wake_time) < self.cfg.wake_cooldown_sec:
                continue

            is_sp = self.vad.is_speech(ch.pcm16)
            window_frames.append(is_sp)
            wake_session.accept_audio(ch.pcm16)

            # For Vosk, we need to check for a final result. The session implementation
            # only returns FinalResult(), so for wake we create an internal recognizer session
            # by using VoskEngine with grammar and checking interim by peeking at recognizer.
            # To keep the abstraction small, wake uses a dedicated method below.
            res = _try_get_vosk_result(wake_session)
            if res is None:
                continue

            text = (res.text or "").strip().lower()
            conf = res.avg_confidence
            if not text:
                continue

            # Wake detection: fuzzy prefix matching.
            # When Vosk finalizes an utterance it often contains only the spoken phrase,
            # so prefix match is usually sufficient.
            w_ratio = wake_match_ratio_prefix(text, self.cfg.wake_phrase)
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "wake probe: text='%s' ratio=%.3f conf=%s vad_speech=%s",
                    text,
                    w_ratio,
                    conf,
                    is_sp,
                )

            if w_ratio < self.cfg.wake_match_threshold:
                if log.isEnabledFor(logging.DEBUG):
                    log.debug(
                        "wake rejected: below_threshold (ratio=%.3f < %.3f) text='%s'",
                        w_ratio,
                        self.cfg.wake_match_threshold,
                        text,
                    )
                continue

            if log.isEnabledFor(logging.INFO):
                log.info("wake candidate: text='%s' ratio=%.3f conf=%s", text, w_ratio, conf)

            if conf is not None and conf < self.cfg.wake_min_conf:
                log.debug("wake rejected: low_confidence (conf=%.3f)", conf)
                continue

            total = len(window_frames)
            speech = sum(1 for x in window_frames if x)
            ratio = (speech / total) if total else 0.0
            if ratio < self.cfg.wake_min_speech_ratio:
                # If ASR matched the wake phrase extremely well but VAD says "no speech",
                # prefer ASR to avoid a "wake does nothing" UX. This happens on some
                # macOS setups where WebRTC VAD is overly strict.
                # But VAD is the only signal that ties the match to something actually
                # said near the mic; without it (or without decent confidence backing
                # the match) a strong text ratio alone is exactly what a short, common
                # wake word (e.g. a real name) picks up from background TV/music/dialogue.
                # So only take this shortcut when confidence also backs the match.
                wake_words = self.cfg.wake_phrase.lower().strip().split()
                conf_ok = conf is None or conf >= self.cfg.wake_min_conf
                if w_ratio >= 0.95 and len(wake_words) <= 2 and conf_ok:
                    log.warning(
                        "wake accepted despite low speech_ratio=%.3f (w_ratio=%.3f, text='%s')",
                        ratio,
                        w_ratio,
                        text,
                    )
                else:
                    log.debug(
                        "wake rejected: low_speech_ratio (ratio=%.3f < %.3f, text='%s', w_ratio=%.3f)",
                        ratio,
                        self.cfg.wake_min_speech_ratio,
                        text,
                        w_ratio,
                    )
                    continue

            # If the user says wake+command in one breath, Vosk will often finalize it as
            # a single utterance: "олег как у тебя дела". In that case we should not
            # throw away the command audio (we don't have it anymore), so we accept the
            # remainder of the recognized text as the command.
            embedded_cmd = strip_wake_phrase(text, self.cfg.wake_phrase, fuzzy_threshold=self.cfg.wake_strip_fuzzy_threshold)

            if embedded_cmd:
                log.info("wake embedded command extracted: '%s'", embedded_cmd)
            else:
                log.info("wake embedded command: none (will record separately)")

            self._last_wake_time = time.monotonic()
            return WakeMetrics(
                time_to_wake_sec=self._last_wake_time - start,
                confidence=conf,
                speech_ratio=ratio,
                wake_text=text,
                wake_ratio=w_ratio,
                embedded_command_text=embedded_cmd,
            )

    def record_command(self) -> tuple[bytes, CommandMetrics]:
        """Record a command utterance after wake using VAD-driven stop conditions."""

        # Critical: clear any pre-roll / wake tail, so wake never leaks into command.
        self.audio.clear()
        self._drain_for(self.cfg.wake_tail_drop_ms / 1000.0)

        start_wait = time.monotonic()
        frames: list[bytes] = []
        speech_frames = 0
        started = False
        start_debounce = 0
        last_speech_t = None

        while True:
            if self._cancel_event is not None and self._cancel_event.is_set():
                raise PipelineCancelled("record_command: cancelled")

            now = time.monotonic()
            if not started and (now - start_wait) > self.cfg.command_start_timeout_sec:
                self.false_wakes += 1
                return b"", CommandMetrics(
                    duration_sec=0.0,
                    stop_reason=StopReason.NO_SPEECH_START,
                    speech_frames=0,
                    total_frames=0,
                )

            ch = self.audio.read(timeout=0.5)
            if ch is None:
                continue

            is_sp = self.vad.is_speech(ch.pcm16)

            if not started:
                if is_sp:
                    start_debounce += 1
                else:
                    start_debounce = 0

                if start_debounce >= self.cfg.start_speech_min_frames:
                    started = True
                    last_speech_t = time.monotonic()
                    frames.append(ch.pcm16)
                    speech_frames += 1
                continue

            # started
            frames.append(ch.pcm16)
            if is_sp:
                speech_frames += 1
                last_speech_t = time.monotonic()

            total_frames = len(frames)
            dur = (total_frames * self.cfg.frame_ms) / 1000.0

            if dur >= self.cfg.max_phrase_sec:
                return b"".join(frames), CommandMetrics(
                    duration_sec=dur,
                    stop_reason=StopReason.MAX_PHRASE,
                    speech_frames=speech_frames,
                    total_frames=total_frames,
                )

            if last_speech_t is not None:
                silence = time.monotonic() - last_speech_t
                if silence >= self.cfg.max_silence_sec:
                    if dur < self.cfg.min_phrase_sec:
                        return b"", CommandMetrics(
                            duration_sec=dur,
                            stop_reason=StopReason.TOO_SHORT,
                            speech_frames=speech_frames,
                            total_frames=total_frames,
                        )
                    return b"".join(frames), CommandMetrics(
                        duration_sec=dur,
                        stop_reason=StopReason.SILENCE_TIMEOUT,
                        speech_frames=speech_frames,
                        total_frames=total_frames,
                    )

    def run_once(self) -> Optional[Interaction]:
        """Run one interaction: wake → record → ASR → postprocess.

        Returns
        -------
        Interaction | None
            None when recording ended without a usable command (e.g. false wake).
        """

        wake = self.wait_for_wake()

        # Fast path: wake phrase already includes the command ("wake + query" without pause).
        if wake.embedded_command_text:
            log.info("run_once: using embedded command text")
            asr = AsrResult(text=wake.embedded_command_text, avg_confidence=None)
            cmd = CommandMetrics(
                duration_sec=0.0,
                stop_reason=StopReason.EMBEDDED_IN_WAKE,
                speech_frames=0,
                total_frames=0,
            )
            return Interaction(
                wake=wake,
                command=cmd,
                asr=asr,
                final_text=wake.embedded_command_text,
                command_pcm16=b"",
                command_sample_rate=self.cfg.sample_rate,
            )

        log.info("run_once: recording command after wake")
        pcm, cmd = self.record_command()
        if not pcm:
            log.info("No command captured after wake (reason=%s)", cmd.stop_reason)
            return None

        if self.cfg.skip_command_asr:
            return Interaction(
                wake=wake,
                command=cmd,
                asr=AsrResult(text="", avg_confidence=None),
                final_text="",
                command_pcm16=pcm,
                command_sample_rate=self.cfg.sample_rate,
            )

        sess = self.command_engine.start_session()
        # Feed in chunked manner (keeps memory stable if you switch to streaming ASR later).
        step = self.frame_bytes * 10
        for i in range(0, len(pcm), step):
            sess.accept_audio(pcm[i : i + step])
        asr = sess.finalize()

        final_text = strip_wake_phrase(
            asr.text,
            self.cfg.wake_phrase,
            fuzzy_threshold=self.cfg.wake_strip_fuzzy_threshold,
        )

        return Interaction(
            wake=wake,
            command=cmd,
            asr=asr,
            final_text=final_text,
            command_pcm16=pcm,
            command_sample_rate=self.cfg.sample_rate,
        )


def _try_get_vosk_result(session) -> Optional[AsrResult]:
    """Hacky helper: wake needs final results from Vosk recognizer.

    Our `VoskSession` only exposes FinalResult(). For wake detection we need to know
    *when* a full utterance ended (AcceptWaveform == True). Vosk exposes this via
    recognizer.AcceptWaveform(...)->bool.

    To keep the public interface clean, we piggy-back on the internal recognizer
    stored inside `VoskSession`.
    """

    # We rely on VoskSession remembering whether AcceptWaveform() accepted.
    fn = getattr(session, "_consume_if_accepted", None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None
