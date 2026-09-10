from __future__ import annotations

import threading
import unittest

from assistant.pipeline_vosk import PipelineCancelled, PipelineConfig, WakeCommandPipeline


class _FakeAudio:
    """Minimal AudioStream stand-in: never yields a frame, never blocks."""

    def __init__(self, frame_bytes: int):
        self.frame_bytes = frame_bytes

    def read(self, timeout: float = 0.5, *, q=None):
        return None

    def clear(self) -> None:
        pass


class _FakeEngine:
    def start_session(self):
        return self


class TestPipelineCancellation(unittest.TestCase):
    def _make_pipeline(self, cancel_event: threading.Event | None) -> WakeCommandPipeline:
        cfg = PipelineConfig(sample_rate=16000, frame_ms=30, wake_tail_drop_ms=1)
        frame_bytes = int(cfg.sample_rate * cfg.frame_ms / 1000) * 2
        return WakeCommandPipeline(
            cfg=cfg,
            audio=_FakeAudio(frame_bytes),
            wake_engine=_FakeEngine(),
            command_engine=_FakeEngine(),
            cancel_event=cancel_event,
        )

    def test_wait_for_wake_raises_when_cancelled(self):
        cancel_event = threading.Event()
        cancel_event.set()
        pipeline = self._make_pipeline(cancel_event)
        with self.assertRaises(PipelineCancelled):
            pipeline.wait_for_wake()

    def test_record_command_raises_when_cancelled(self):
        cancel_event = threading.Event()
        cancel_event.set()
        pipeline = self._make_pipeline(cancel_event)
        with self.assertRaises(PipelineCancelled):
            pipeline.record_command()

    def test_no_cancel_event_is_backward_compatible(self):
        # Default (no cancel_event passed) must not require the new parameter.
        pipeline = self._make_pipeline(None)
        self.assertIsNone(pipeline._cancel_event)


if __name__ == "__main__":
    unittest.main()
