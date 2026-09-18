from __future__ import annotations

import threading
import unittest

from assistant.asr_engines import AsrResult
from assistant.audio_stream import AudioChunk
from assistant.pipeline_vosk import PipelineCancelled, PipelineConfig, WakeCommandPipeline


class _SilentAudio:
    """Yields silent (non-speech) frames forever; can trip a cancel_event after N reads."""

    def __init__(self, frame_bytes: int, *, cancel_event: threading.Event | None = None, cancel_after: int = 20):
        self.frame_bytes = frame_bytes
        self._cancel_event = cancel_event
        self._cancel_after = cancel_after
        self._reads = 0

    def read(self, timeout: float = 0.5, *, q=None):
        self._reads += 1
        if self._cancel_event is not None and self._reads >= self._cancel_after:
            self._cancel_event.set()
        return AudioChunk(pcm16=b"\x00" * self.frame_bytes, t_monotonic=0.0)

    def clear(self) -> None:
        pass


class _ScriptedWakeSession:
    """Returns `results` in order from `_consume_if_accepted`, then None forever."""

    def __init__(self, results: list[AsrResult]):
        self._results = list(results)
        self._idx = 0

    def accept_audio(self, pcm16: bytes) -> None:
        pass

    def _consume_if_accepted(self):
        if self._idx >= len(self._results):
            return None
        r = self._results[self._idx]
        self._idx += 1
        return r


class _ScriptedWakeEngine:
    def __init__(self, results: list[AsrResult]):
        self._results = results

    def start_session(self):
        return _ScriptedWakeSession(self._results)


def _make_pipeline(*, wake_results: list[AsrResult], cancel_event: threading.Event) -> WakeCommandPipeline:
    cfg = PipelineConfig(
        sample_rate=16000,
        frame_ms=30,
        wake_phrase="олег",
        wake_match_threshold=0.60,
        wake_min_speech_ratio=0.5,  # silent frames -> ratio stays 0.0, always "low"
        wake_min_conf=0.70,
        wake_confirm_window_sec=0.3,
    )
    frame_bytes = int(cfg.sample_rate * cfg.frame_ms / 1000) * 2
    audio = _SilentAudio(frame_bytes, cancel_event=cancel_event)
    return WakeCommandPipeline(
        cfg=cfg,
        audio=audio,
        wake_engine=_ScriptedWakeEngine(wake_results),
        command_engine=_ScriptedWakeEngine(wake_results),
        cancel_event=cancel_event,
    )


class TestWakeConfidenceGate(unittest.TestCase):
    def test_low_confidence_match_is_rejected_without_speech(self):
        # Exact text match (ratio=1.0) but low confidence and no VAD speech:
        # must NOT be accepted despite the short-wake-phrase escape hatch.
        cancel_event = threading.Event()
        pipeline = _make_pipeline(
            wake_results=[AsrResult(text="олег", avg_confidence=0.20)],
            cancel_event=cancel_event,
        )
        with self.assertRaises(PipelineCancelled):
            pipeline.wait_for_wake()

    def test_high_confidence_match_is_accepted_without_speech(self):
        # Same weak VAD signal, but high confidence backs the match this time:
        # the escape hatch should still let it through.
        cancel_event = threading.Event()
        pipeline = _make_pipeline(
            wake_results=[AsrResult(text="олег", avg_confidence=0.95)],
            cancel_event=cancel_event,
        )
        wake = pipeline.wait_for_wake()
        self.assertEqual(wake.wake_text, "олег")


if __name__ == "__main__":
    unittest.main()
