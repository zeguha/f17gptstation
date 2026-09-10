import unittest

from assistant.intent_weather import detect_weather_intent


class TestWeatherIntent(unittest.TestCase):
    def test_now_has_no_time_target(self):
        i = detect_weather_intent("какая погода")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.intent, "weather_now")
        self.assertIsNone(i.day_offset)
        self.assertIsNone(i.weekday)
        self.assertIsNone(i.part_of_day)

    def test_tomorrow(self):
        i = detect_weather_intent("какая погода завтра")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.day_offset, 1)
        self.assertIsNone(i.weekday)

    def test_day_after_tomorrow(self):
        i = detect_weather_intent("погода послезавтра")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.day_offset, 2)

    def test_today_explicit(self):
        i = detect_weather_intent("погода сегодня вечером")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.day_offset, 0)
        self.assertEqual(i.part_of_day, "evening")

    def test_weekday(self):
        i = detect_weather_intent("какая погода в пятницу")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.weekday, 4)
        self.assertIsNone(i.day_offset)

    def test_weekday_does_not_leak_into_place(self):
        i = detect_weather_intent("погода в пятницу")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertIsNone(i.place_text)

    def test_place_still_extracted_with_weekday(self):
        i = detect_weather_intent("погода в москве в пятницу")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.place_text, "москве")
        self.assertEqual(i.weekday, 4)

    def test_part_of_day_morning(self):
        i = detect_weather_intent("погода завтра утром")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.day_offset, 1)
        self.assertEqual(i.part_of_day, "morning")

    def test_umbrella_still_detected(self):
        i = detect_weather_intent("нужен ли зонт завтра")
        self.assertIsNotNone(i)
        assert i is not None
        self.assertEqual(i.intent, "weather_umbrella")
        self.assertEqual(i.day_offset, 1)


if __name__ == "__main__":
    unittest.main()
