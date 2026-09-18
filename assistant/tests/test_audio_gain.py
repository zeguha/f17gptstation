from __future__ import annotations

import unittest

import numpy as np

from assistant.audio_stream import _apply_gain_int16


def _samples(*values: int) -> bytes:
    return np.array(values, dtype=np.int16).tobytes()


def _unpack(pcm16: bytes) -> list[int]:
    return np.frombuffer(pcm16, dtype=np.int16).tolist()


class TestApplyGain(unittest.TestCase):
    def test_gain_1_is_identity(self):
        raw = _samples(0, 100, -100, 32000)
        self.assertEqual(_apply_gain_int16(raw, 1.0), raw)

    def test_gain_doubles_amplitude(self):
        raw = _samples(0, 100, -100, 1000)
        out = _unpack(_apply_gain_int16(raw, 2.0))
        self.assertEqual(out, [0, 200, -200, 2000])

    def test_gain_clips_instead_of_wrapping(self):
        raw = _samples(20000, -20000)
        out = _unpack(_apply_gain_int16(raw, 3.0))
        # Without clipping this would wrap around to a negative/positive
        # value near zero (classic int16 overflow) instead of saturating.
        self.assertEqual(out, [32767, -32768])

    def test_empty_input(self):
        self.assertEqual(_apply_gain_int16(b"", 2.0), b"")


if __name__ == "__main__":
    unittest.main()
