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

from .config import WeatherConfig
from .geo import GeoUnavailable, get_location
from .intent_weather import WeatherIntent
from .nlg_weather_ru import render_weather_answer_ru
from .weather_openmeteo import fetch_weather_now, geocode_place, reverse_geocode


class WeatherSkillError(RuntimeError):
    pass


_WEATHER_CACHE: dict[tuple[int, int], tuple[float, str]] = {}


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


def _cache_key(lat: float, lon: float) -> tuple[int, int]:
    # Round to ~1km to improve cache hits and reduce privacy risk.
    return (int(round(float(lat) * 100)), int(round(float(lon) * 100)))


def _cache_get(lat: float, lon: float, *, ttl_sec: float) -> str | None:
    k = _cache_key(lat, lon)
    item = _WEATHER_CACHE.get(k)
    if not item:
        return None
    ts, val = item
    if (time.time() - ts) > float(ttl_sec):
        return None
    return val


def _cache_put(lat: float, lon: float, text: str) -> None:
    k = _cache_key(lat, lon)
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

    cached = _cache_get(lat, lon, ttl_sec=cfg.weather_cache_ttl_sec)
    if cached is not None:
        return cached

    w = await fetch_weather_now(lat, lon, timeout_sec=cfg.weather_timeout_sec)
    text = render_weather_answer_ru(w, place=place_name, mode=user_intent.intent)
    _cache_put(lat, lon, text)
    return text
