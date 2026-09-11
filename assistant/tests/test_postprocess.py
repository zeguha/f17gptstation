import unittest

from assistant.postprocess import clean_for_speech, strip_wake_phrase


class TestPostprocess(unittest.TestCase):
    def test_strip_exact_prefix(self):
        self.assertEqual(strip_wake_phrase("мышка сосиска включи свет", "мышка сосиска"), "включи свет")

    def test_strip_exact_only(self):
        self.assertEqual(strip_wake_phrase("мышка сосиска", "мышка сосиска"), "")

    def test_no_strip_if_not_prefix(self):
        self.assertEqual(strip_wake_phrase("включи мышка сосиска свет", "мышка сосиска"), "включи мышка сосиска свет")


class TestCleanForSpeech(unittest.TestCase):
    def test_strips_wrapped_citation(self):
        t = clean_for_speech("Новая модель вышла. ([axios.com](https://www.axios.com/foo?utm_source=openai)) Дальше текст.")
        self.assertNotIn("axios.com", t)
        self.assertNotIn("(", t)
        self.assertEqual(t, "Новая модель вышла. Дальше текст.")

    def test_strips_plain_markdown_link_keeps_label(self):
        t = clean_for_speech("Смотри [сайт](https://example.com/page) тут.")
        self.assertEqual(t, "Смотри сайт тут.")

    def test_strips_bare_url(self):
        t = clean_for_speech("Ссылка: https://example.com/page тут.")
        self.assertNotIn("http", t)

    def test_strips_headers_and_emphasis(self):
        t = clean_for_speech("### Заголовок\n**Важно**: это *текст*.")
        self.assertNotIn("#", t)
        self.assertNotIn("*", t)
        self.assertIn("Важно", t)
        self.assertIn("текст", t)

    def test_plain_text_unchanged(self):
        self.assertEqual(clean_for_speech("Обычный ответ без разметки."), "Обычный ответ без разметки.")


if __name__ == "__main__":
    unittest.main()

