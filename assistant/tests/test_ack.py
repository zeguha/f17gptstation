from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

from assistant.main_cloud_orchestrated import _worker_stt


@dataclass
class _FakeInteraction:
    final_text: str = ""
    command_pcm16: bytes = b""
    command_sample_rate: int = 16000


@dataclass
class _FakeAudioCfg:
    output_device: int | None = None


@dataclass
class _FakeCfg:
    audio: _FakeAudioCfg = field(default_factory=_FakeAudioCfg)


class TestAckOnUnderstoodCommand(unittest.TestCase):
    def test_ack_played_for_embedded_text(self):
        async def scenario() -> None:
            in_q: asyncio.Queue = asyncio.Queue()
            out_q: asyncio.Queue = asyncio.Queue()
            stop_event = asyncio.Event()
            playback = AsyncMock()
            cfg = _FakeCfg()

            await in_q.put(_FakeInteraction(final_text="какая погода"))

            task = asyncio.create_task(
                _worker_stt(
                    in_q=in_q,
                    out_q=out_q,
                    stt=AsyncMock(),
                    stop_event=stop_event,
                    playback=playback,
                    cfg=cfg,
                    ack_wav=b"ACKBYTES",
                )
            )
            interaction, user_text = await asyncio.wait_for(out_q.get(), timeout=2.0)
            self.assertEqual(user_text, "какая погода")

            # The ack is fire-and-forget (asyncio.create_task inside the
            # worker); give the event loop a tick to run it.
            await asyncio.sleep(0.05)
            playback.play_blocking.assert_awaited_once_with(b"ACKBYTES", device=None)

            stop_event.set()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(scenario())

    def test_no_ack_when_nothing_understood(self):
        async def scenario() -> None:
            in_q: asyncio.Queue = asyncio.Queue()
            out_q: asyncio.Queue = asyncio.Queue()
            stop_event = asyncio.Event()
            playback = AsyncMock()
            cfg = _FakeCfg()

            # No final_text and no command_pcm16 -> user_text stays empty,
            # STT is never even called.
            await in_q.put(_FakeInteraction())

            task = asyncio.create_task(
                _worker_stt(
                    in_q=in_q,
                    out_q=out_q,
                    stt=AsyncMock(),
                    stop_event=stop_event,
                    playback=playback,
                    cfg=cfg,
                    ack_wav=b"ACKBYTES",
                )
            )
            interaction, user_text = await asyncio.wait_for(out_q.get(), timeout=2.0)
            self.assertEqual(user_text, "")

            await asyncio.sleep(0.05)
            playback.play_blocking.assert_not_awaited()

            stop_event.set()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
