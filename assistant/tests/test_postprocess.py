import unittest

from assistant.postprocess import strip_wake_phrase


class TestPostprocess(unittest.TestCase):
    def test_strip_exact_prefix(self):
        self.assertEqual(strip_wake_phrase("мышка сосиска включи свет", "мышка сосиска"), "включи свет")

    def test_strip_exact_only(self):
        self.assertEqual(strip_wake_phrase("мышка сосиска", "мышка сосиска"), "")

    def test_no_strip_if_not_prefix(self):
        self.assertEqual(strip_wake_phrase("включи мышка сосиска свет", "мышка сосиска"), "включи мышка сосиска свет")


if __name__ == "__main__":
    unittest.main()

