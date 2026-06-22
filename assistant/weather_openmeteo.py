"""Weather provider: Open-Meteo (no API key).

Docs:
- https://open-meteo.com/
- Geocoding: https://open-meteo.com/en/docs/geocoding-api

We only need current conditions + near-hour precipitation probability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import aiohttp

from .net import HttpStatusError, RetryConfig, ssl_context, with_retries


@dataclass(frozen=True, slots=True)
class Place:
    name: str
    country: str | None
    admin1: str | None
    latitude: float
    longitude: float
    timezone: str | None = None


@dataclass(frozen=True, slots=True)
class WeatherNow:
    temperature_c: float
    apparent_c: float
    wind_ms: float
    humidity_pct: float | None
    precipitation_mm: float | None
    precipitation_prob_pct: int | None
    weather_code: int | None
    timezone: str | None


def _retry_cfg() -> RetryConfig:
    # Weather APIs are sometimes flaky; keep retries small.
    return RetryConfig(max_attempts=3, base_delay_sec=0.4, max_delay_sec=4.0)


async def _get_json(url: str, *, timeout_sec: float) -> dict:
    timeout = aiohttp.ClientTimeout(total=float(timeout_sec))
    connector = aiohttp.TCPConnector(ssl=ssl_context())
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True, connector=connector) as s:
        async with s.get(url, headers={"User-Agent": "gpt-station-assistant/1.0"}) as r:
            txt = await r.text()
            if r.status >= 300:
                raise HttpStatusError(r.status, txt, where="open-meteo")
            try:
                return json.loads(txt)
            except Exception:
                # keep original body for easier debugging
                raise RuntimeError(f"open-meteo: invalid json: {txt[:500]}")


async def geocode_place(name: str, *, timeout_sec: float = 4.0, language: str = "ru") -> Place | None:
    q = (name or "").strip()
    if not q:
        return None
    url = (
        "https://geocoding-api.open-meteo.com/v1/search"
        f"?name={aiohttp.helpers.quote(q)}&count=5&language={aiohttp.helpers.quote(language)}&format=json"
    )

    async def call() -> dict:
        return await _get_json(url, timeout_sec=timeout_sec)

    data = await with_retries(call, cfg=_retry_cfg(), what="open-meteo.geocode")
    res = data.get("results")
    if not isinstance(res, list) or not res:
        return None
    top = res[0] or {}
    try:
        return Place(
            name=str(top.get("name") or q),
            country=(str(top.get("country")) if top.get("country") else None),
            admin1=(str(top.get("admin1")) if top.get("admin1") else None),
            latitude=float(top["latitude"]),
            longitude=float(top["longitude"]),
            timezone=(str(top.get("timezone")) if top.get("timezone") else None),
        )
    except Exception:
        return None


async def reverse_geocode(lat: float, lon: float, *, timeout_sec: float = 4.0, language: str = "ru") -> Place | None:
    url = (
        "https://geocoding-api.open-meteo.com/v1/reverse"
        f"?latitude={lat}&longitude={lon}&language={aiohttp.helpers.quote(language)}&format=json"
    )

    async def call() -> dict:
        return await _get_json(url, timeout_sec=timeout_sec)

    data = await with_retries(call, cfg=_retry_cfg(), what="open-meteo.reverse")
    res = data.get("results")
    if not isinstance(res, list) or not res:
        return None
    top = res[0] or {}
    try:
        return Place(
            name=str(top.get("name") or ""),
            country=(str(top.get("country")) if top.get("country") else None),
            admin1=(str(top.get("admin1")) if top.get("admin1") else None),
            latitude=float(top["latitude"]),
            longitude=float(top["longitude"]),
            timezone=(str(top.get("timezone")) if top.get("timezone") else None),
        )
    except Exception:
        return None


async def fetch_weather_now(lat: float, lon: float, *, timeout_sec: float = 6.0) -> WeatherNow:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,relative_humidity_2m"
        "&hourly=precipitation_probability"
        "&forecast_days=1&timezone=auto"
    )

    async def call() -> dict:
        return await _get_json(url, timeout_sec=timeout_sec)

    data = await with_retries(call, cfg=_retry_cfg(), what="open-meteo.weather")
    cur = data.get("current") or {}
    hourly = data.get("hourly") or {}

    prob0 = None
    probs = hourly.get("precipitation_probability")
    if isinstance(probs, list) and probs:
        try:
            prob0 = int(round(float(probs[0])))
        except Exception:
            prob0 = None

    def _f(x):
        return None if x is None else float(x)

    def _i(x):
        return None if x is None else int(x)

    return WeatherNow(
        temperature_c=float(cur.get("temperature_2m")),
        apparent_c=float(cur.get("apparent_temperature")),
        wind_ms=float(cur.get("wind_speed_10m")),
        humidity_pct=_f(cur.get("relative_humidity_2m")),
        precipitation_mm=_f(cur.get("precipitation")),
        precipitation_prob_pct=prob0,
        weather_code=_i(cur.get("weather_code")),
        timezone=(str(data.get("timezone")) if data.get("timezone") else None),
    )
