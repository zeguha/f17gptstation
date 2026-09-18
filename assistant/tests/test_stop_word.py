from __future__ import annotations

import unittest

from assistant.stop_word import StopWordConfig, StopWordDetector, _match_ratio_suffix


class TestMatchRatioSuffix(unittest.TestCase):
    def test_exact_word_matches(self):
        self.assertGreaterEqual(_match_ratio_suffix("стоп", "стоп"), 0.99)

    def test_word_buried_in_garbage_prefix_still_matches(self):
        # This is exactly the real bug: TTS bleed-through merged with the
        # actual stop word, which only shows up at the very end.
        r = _match_ratio_suffix("сейчас не об бы девять стоп", "стоп")
        self.assertGreaterEqual(r, 0.99)

    def test_word_only_at_start_does_not_match_as_well(self):
        # Old prefix-based matching would have caught this instead; suffix
        # matching should not, since the word is not what was said *last*.
        r = _match_ratio_suffix("стоп это случайное продолжение фразы", "стоп")
        self.assertLess(r, 0.5)

    def test_empty_text_no_match(self):
        self.assertEqual(_match_ratio_suffix("", "стоп"), 0.0)


class _ScriptedSession:
    """Fake AsrSession: scripted sequence of partial_text() values."""

    def __init__(self, partials: list[str]):
        self._partials = list(partials)
        self._idx = 0

    def accept_audio(self, pcm16: bytes) -> None:
        pass

    def partial_text(self) -> str:
        if self._idx >= len(self._partials):
            return self._partials[-1] if self._partials else ""
        val = self._partials[self._idx]
        self._idx += 1
        return val


class _ScriptedEngine:
    def __init__(self, partials: list[str]):
        self._partials = partials

    def start_session(self):
        return _ScriptedSession(self._partials)


class TestStopWordDetectorPartialPath(unittest.TestCase):
    def _frame_bytes(self, cfg: StopWordConfig) -> int:
        return int(cfg.sample_rate * cfg.frame_ms / 1000) * 2

    def test_fires_once_partial_hypothesis_ends_with_stop_word(self):
        cfg = StopWordConfig(min_speech_ratio=0.0)  # isolate from VAD/webrtcvad specifics
        # Growing partial hypothesis, as if TTS bleed-through preceded the
        # actual "стоп" the user said.
        partials = ["", "сейчас", "сейчас не", "сейчас не об бы девять", "сейчас не об бы девять стоп"]
        engine = _ScriptedEngine(partials)
        detector = StopWordDetector(cfg=cfg, engine=engine)

        frame = b"\x00" * self._frame_bytes(cfg)
        fired_on = None
        for i in range(len(partials)):
            if detector.accept_frame(frame):
                fired_on = i
                break
        self.assertEqual(fired_on, len(partials) - 1)

    def test_does_not_fire_on_unrelated_partial(self):
        cfg = StopWordConfig(min_speech_ratio=0.0)
        partials = ["привет", "как дела", "это не стоп-слово вовсе"]
        engine = _ScriptedEngine(partials)
        detector = StopWordDetector(cfg=cfg, engine=engine)

        frame = b"\x00" * self._frame_bytes(cfg)
        for _ in partials:
            self.assertFalse(detector.accept_frame(frame))

    def test_cooldown_prevents_immediate_refire(self):
        cfg = StopWordConfig(min_speech_ratio=0.0, cooldown_sec=1.0)
        engine = _ScriptedEngine(["стоп"])
        detector = StopWordDetector(cfg=cfg, engine=engine)
        frame = b"\x00" * self._frame_bytes(cfg)

        self.assertTrue(detector.accept_frame(frame))
        # Immediately after firing, still within cooldown -> must not fire again.
        self.assertFalse(detector.accept_frame(frame))


if __name__ == "__main__":
    unittest.main()
