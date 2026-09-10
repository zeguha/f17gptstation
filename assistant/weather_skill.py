"""Weather skill orchestrator.

This is the only module entrypoint other parts of the assistant should call.
It:
- detects intent (done outside)
- resolves location (explicit place text -> geocode; else geo)
- fetches weather
- renders RU answer
"""

from __future__ import annotations

import time

import re
from datetime import date, datetime, timedelta

from .config import WeatherConfig
from .geo import GeoUnavailable, get_location
from .intent_weather import WeatherIntent
from .nlg_weather_ru import render_weather_answer_ru
from .weather_openmeteo import (
    ForecastPoint,
    ForecastSeries,
    WeatherNow,
    fetch_weather_forecast,
    fetch_weather_now,
    geocode_place,
    reverse_geocode,
)


class WeatherSkillError(RuntimeError):
    pass


_WEATHER_CACHE: dict[tuple[int, int, str], tuple[float, str]] = {}

_WEEKDAY_LABELS_RU: dict[int, str] = {
    0: "в понедельник",
    1: "во вторник",
    2: "в среду",
    3: "в четверг",
    4: "в пятницу",
    5: "в субботу",
    6: "в воскресенье",
}

_PART_OF_DAY_LABELS_RU: dict[str, str] = {
    "morning": "утром",
    "day": "днём",
    "evening": "вечером",
    "night": "ночью",
}

_PART_OF_DAY_HOUR: dict[str, int] = {
    "morning": 9,
    "day": 13,
    "evening": 19,
    "night": 23,
}


def _resolve_target_date(user_intent: WeatherIntent, *, today: date) -> tuple[date, int]:
    """Return (target_date, day_offset_from_today)."""

    if user_intent.weekday is not None:
        offset = (user_intent.weekday - today.weekday()) % 7
        return today + timedelta(days=offset), offset
    offset = user_intent.day_offset or 0
    return today + timedelta(days=offset), offset


def _day_label_ru(*, offset: int, weekday: int | None) -> str:
    if offset == 0:
        return "сегодня"
    if offset == 1:
        return "завтра"
    if offset == 2 and weekday is None:
        return "послезавтра"
    if weekday is not None:
        return _WEEKDAY_LABELS_RU[weekday]
    return "послезавтра"


def _when_label_ru(user_intent: WeatherIntent, *, offset: int) -> str:
    day_label = _day_label_ru(offset=offset, weekday=user_intent.weekday)
    part_label = _PART_OF_DAY_LABELS_RU.get(user_intent.part_of_day or "")
    if part_label:
        return f"{day_label} {part_label}"
    return day_label


def _is_forecast_request(user_intent: WeatherIntent) -> bool:
    if user_intent.weekday is not None:
        return True
    if user_intent.day_offset is not None and user_intent.day_offset > 0:
        return True
    if user_intent.part_of_day is not None:
        return True
    return False


def _closest_point(points: list[ForecastPoint], *, target_date: date, target_hour: int) -> ForecastPoint | None:
    prefix = target_date.strftime("%Y-%m-%d")
    same_day = [p for p in points if p.time_local.startswith(prefix)]
    if not same_day:
        return None

    def _hour_of(p: ForecastPoint) -> int:
        try:
            return int(p.time_local[11:13])
        except (ValueError, IndexError):
            return 0

    return min(same_day, key=lambda p: abs(_hour_of(p) - target_hour))


def _forecast_point_to_weather_now(p: ForecastPoint, *, timezone: str | None) -> WeatherNow:
    return WeatherNow(
        temperature_c=p.temperature_c,
        apparent_c=p.apparent_c,
        wind_ms=p.wind_ms,
        humidity_pct=p.humidity_pct,
        precipitation_mm=None,
        precipitation_prob_pct=p.precipitation_prob_pct,
        weather_code=p.weather_code,
        timezone=timezone,
    )


_RU_ENDINGS: tuple[str, ...] = (
    # prepositional / genitive / dative / instrumental / plural-ish, very rough
    "ами",
    "ями",
    "ах",
    "ях",
    "ом",
    "ем",
    "ой",
    "ей",
    "ою",
    "ею",
    "ам",
    "ям",
    "у",
    "ю",
    "е",
    "и",
    "а",
    "я",
)


def _place_candidates_ru(place_text: str) -> list[str]:
    """Generate fallback candidates for RU inflected city names.

    Example: "екатеринбурге" -> ["екатеринбурге", "екатеринбург"].

    We intentionally keep it heuristic (no heavy morphology deps).
    """

    p = (place_text or "").strip().lower()
    p = re.sub(r"\s+", " ", p)
    p = p.strip(" ,.!?\t\n")
    if not p:
        return []

    cands: list[str] = [p]

    # Strip common endings (longest first)
    base = p
    for suf in _RU_ENDINGS:
        if len(base) > len(suf) + 2 and base.endswith(suf):
            base2 = base[: -len(suf)]
            if base2 and base2 not in cands:
                cands.append(base2)
            base = base2
            break

    # A couple of additional guesses for feminine forms like "москв" -> "москва".
    # Only add if base ends with a consonant.
    if base and base[-1] not in "аеёиоуыэюяь":
        for suf in ("а", "я", "ь"):
            v = base + suf
            if v not in cands:
                cands.append(v)

    # Keep only reasonable-length candidates
    out: list[str] = []
    for x in cands:
        x2 = x.strip()
        if 2 <= len(x2) <= 64 and x2 not in out:
            out.append(x2)
    return out


def _cache_key(lat: float, lon: float, *, when: str) -> tuple[int, int, str]:
    # Round to ~1km to improve cache hits and reduce privacy risk.
    return (int(round(float(lat) * 100)), int(round(float(lon) * 100)), when)


def _cache_get(lat: float, lon: float, *, when: str, ttl_sec: float) -> str | None:
    k = _cache_key(lat, lon, when=when)
    item = _WEATHER_CACHE.get(k)
    if not item:
        return None
    ts, val = item
    if (time.time() - ts) > float(ttl_sec):
        return None
    return val


def _cache_put(lat: float, lon: float, text: str, *, when: str) -> None:
    k = _cache_key(lat, lon, when=when)
    _WEATHER_CACHE[k] = (time.time(), text)


async def handle_weather_intent(user_intent: WeatherIntent, *, cfg: WeatherConfig) -> str:
    if not cfg.enabled:
        raise WeatherSkillError("Weather skill disabled")

    place_name: str | None = None
    lat: float | None = None
    lon: float | None = None

    # 1) Explicit place in query has priority.
    if user_intent.place_text:
        place = None
        last_err: Exception | None = None
        for cand in _place_candidates_ru(user_intent.place_text):
            try:
                place = await geocode_place(cand, timeout_sec=cfg.geo_timeout_sec)
            except Exception as e:
                last_err = e
                place = None
            if place is not None:
                break
        if place is None:
            if last_err is not None:
                raise WeatherSkillError("Не смог найти этот город. Скажи, пожалуйста, название иначе.")
            raise WeatherSkillError("Не нашёл такой город. Скажи, пожалуйста, ещё раз.")
        place_name = place.name
        lat, lon = place.latitude, place.longitude
    else:
        # 2) Best-effort geo.
        try:
            pt = await get_location(
                geo_mode=cfg.geo_mode,
                fixed_lat=cfg.fixed_lat,
                fixed_lon=cfg.fixed_lon,
                ip_geo_url=cfg.ip_geo_url,
                timeout_sec=cfg.geo_timeout_sec,
                cache_ttl_sec=cfg.geo_cache_ttl_sec,
            )
        except GeoUnavailable:
            raise WeatherSkillError("Не вижу геолокацию. Скажи, в каком ты городе — и я скажу погоду.")

        lat, lon = pt.lat, pt.lon
        # Make the answer nicer with place name.
        if pt.city:
            place_name = pt.city
        else:
            r = await reverse_geocode(lat, lon, timeout_sec=cfg.geo_timeout_sec)
            if r and r.name:
                place_name = r.name

    assert lat is not None and lon is not None

    if not _is_forecast_request(user_intent):
        cached = _cache_get(lat, lon, when="now", ttl_sec=cfg.weather_cache_ttl_sec)
        if cached is not None:
            return cached

        w = await fetch_weather_now(lat, lon, timeout_sec=cfg.weather_timeout_sec)
        text = render_weather_answer_ru(w, place=place_name, mode=user_intent.intent)
        _cache_put(lat, lon, text, when="now")
        return text

    target_date, offset = _resolve_target_date(user_intent, today=datetime.now().date())
    when_label = _when_label_ru(user_intent, offset=offset)
    when_key = f"{target_date.isoformat()}:{user_intent.part_of_day or ''}"

    cached = _cache_get(lat, lon, when=when_key, ttl_sec=cfg.weather_cache_ttl_sec)
    if cached is not None:
        return cached

    series: ForecastSeries = await fetch_weather_forecast(lat, lon, timeout_sec=cfg.weather_timeout_sec)
    target_hour = _PART_OF_DAY_HOUR.get(user_intent.part_of_day or "", 13)
    point = _closest_point(series.points, target_date=target_date, target_hour=target_hour)
    if point is None:
        raise WeatherSkillError("Не нашёл прогноз на этот день. Спроси про ближайшие несколько дней.")

    w = _forecast_point_to_weather_now(point, timezone=series.timezone)
    text = render_weather_answer_ru(w, place=place_name, mode=user_intent.intent, when_label=when_label)
    _cache_put(lat, lon, text, when=when_key)
    return text
