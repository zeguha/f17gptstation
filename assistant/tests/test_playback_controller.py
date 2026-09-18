from __future__ import annotations

import asyncio
import threading
import unittest
from unittest.mock import patch

from assistant.audio_playback import PlaybackInfo
from assistant.main_cloud_orchestrated import PlaybackController


class TestPlaybackControllerBargeIn(unittest.TestCase):
    def test_stop_interrupts_in_progress_playback(self):
        """Regression test for two real bugs found on hardware:

        1. `_current` was only published *after* playback finished (the old
           play_wav_bytes blocked internally), so stop() never had anything
           to act on while audio was actually playing.
        2. Even once that was fixed, stop() used to await the *same* lock
           play_blocking() holds for its entire duration, so stop() would
           just block until playback had already finished on its own.

        Both together meant barge-in detection could fire correctly and
        stop() would still have no effect until natural completion.
        """

        stop_signal = threading.Event()
        started_signal = threading.Event()
        stopped_calls: list[PlaybackInfo] = []

        def fake_start(wav_bytes, *, device):
            started_signal.set()
            return PlaybackInfo(backend="fake")

        def fake_wait(info):
            # Simulates a long-running player: only returns once stop_playback
            # (mocked below) is actually called.
            stop_signal.wait(timeout=5)

        def fake_stop(info):
            stopped_calls.append(info)
            stop_signal.set()

        async def scenario() -> None:
            controller = PlaybackController()
            play_task = asyncio.create_task(controller.play_blocking(b"x", device=None))

            await asyncio.to_thread(started_signal.wait, 2)
            self.assertTrue(started_signal.is_set(), "start_playback was never called")
            self.assertTrue(controller.playing, "_current/_playing must be set while still playing")

            # Must complete promptly -- if stop() were still gated by the
            # same lock play_blocking() holds for the whole duration, this
            # would hang until fake_wait's 5s timeout and fail the assertion.
            await asyncio.wait_for(controller.stop(), timeout=2.0)

            await asyncio.wait_for(play_task, timeout=2.0)
            self.assertFalse(controller.playing)

        with patch("assistant.main_cloud_orchestrated.start_playback", side_effect=fake_start), patch(
            "assistant.main_cloud_orchestrated.wait_playback", side_effect=fake_wait
        ), patch("assistant.main_cloud_orchestrated.stop_playback", side_effect=fake_stop):
            asyncio.run(scenario())

        self.assertEqual(len(stopped_calls), 1)

    def test_stop_is_a_noop_when_nothing_is_playing(self):
        async def scenario() -> None:
            controller = PlaybackController()
            await asyncio.wait_for(controller.stop(), timeout=1.0)

        with patch("assistant.main_cloud_orchestrated.stop_playback") as mock_stop:
            asyncio.run(scenario())
            mock_stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
