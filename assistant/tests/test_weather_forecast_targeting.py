from __future__ import annotations

import unittest
from datetime import date

from assistant.intent_weather import WeatherIntent
from assistant.weather_openmeteo import ForecastPoint
from assistant.weather_skill import (
    _closest_point,
    _is_forecast_request,
    _resolve_target_date,
    _when_label_ru,
)


class TestForecastTargeting(unittest.TestCase):
    def test_now_is_not_a_forecast_request(self):
        i = WeatherIntent(intent="weather_now", place_text=None, confidence=0.75)
        self.assertFalse(_is_forecast_request(i))

    def test_tomorrow_is_a_forecast_request(self):
        i = WeatherIntent(intent="weather_now", place_text=None, confidence=0.75, day_offset=1)
        self.assertTrue(_is_forecast_request(i))

    def test_part_of_day_alone_is_a_forecast_request(self):
        i = WeatherIntent(intent="weather_now", place_text=None, confidence=0.75, part_of_day="evening")
        self.assertTrue(_is_forecast_request(i))

    def test_resolve_target_date_tomorrow(self):
        i = WeatherIntent(intent="weather_now", place_text=None, confidence=0.75, day_offset=1)
        today = date(2026, 9, 11)  # Friday
        target, offset = _resolve_target_date(i, today=today)
        self.assertEqual(target, date(2026, 9, 12))
        self.assertEqual(offset, 1)

    def test_resolve_target_date_weekday_wraps_to_next_week(self):
        # 2026-09-11 is a Friday (weekday()==4). Asking for Monday should be 3 days ahead.
        i = WeatherIntent(intent="weather_now", place_text=None, confidence=0.75, weekday=0)
        today = date(2026, 9, 11)
        target, offset = _resolve_target_date(i, today=today)
        self.assertEqual(target, date(2026, 9, 14))
        self.assertEqual(offset, 3)

    def test_when_label_tomorrow_evening(self):
        i = WeatherIntent(
            intent="weather_now", place_text=None, confidence=0.75, day_offset=1, part_of_day="evening"
        )
        label = _when_label_ru(i, offset=1)
        self.assertEqual(label, "завтра вечером")

    def test_when_label_weekday(self):
        i = WeatherIntent(intent="weather_now", place_text=None, confidence=0.75, weekday=4)
        label = _when_label_ru(i, offset=3)
        self.assertEqual(label, "в пятницу")

    def test_closest_point_picks_matching_day_and_hour(self):
        points = [
            ForecastPoint("2026-09-12T09:00", 10.0, 9.0, 3.0, 50.0, 10, 1),
            ForecastPoint("2026-09-12T19:00", 5.0, 3.0, 4.0, 60.0, 20, 3),
            ForecastPoint("2026-09-13T09:00", 12.0, 11.0, 2.0, 40.0, 5, 0),
        ]
        p = _closest_point(points, target_date=date(2026, 9, 12), target_hour=19)
        self.assertIsNotNone(p)
        assert p is not None
        self.assertEqual(p.time_local, "2026-09-12T19:00")

    def test_closest_point_returns_none_when_day_missing(self):
        points = [ForecastPoint("2026-09-12T09:00", 10.0, 9.0, 3.0, 50.0, 10, 1)]
        p = _closest_point(points, target_date=date(2026, 9, 20), target_hour=13)
        self.assertIsNone(p)


if __name__ == "__main__":
    unittest.main()
