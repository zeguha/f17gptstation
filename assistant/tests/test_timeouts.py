import unittest

from assistant.pipeline_vosk import PipelineConfig


class TestConfigMath(unittest.TestCase):
    def test_frame_samples(self):
        cfg = PipelineConfig(sample_rate=16000, frame_ms=30)
        self.assertEqual(int(cfg.sample_rate * cfg.frame_ms / 1000), 480)


if __name__ == "__main__":
    unittest.main()

