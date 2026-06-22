"""Geolocation helpers for the weather skill.

This repo targets Raspberry Pi / desktop setups where GPS may be absent.
So we implement:
- fixed coordinates from env (best for a stationary device)
- IP geolocation (coarse city-level)

Mobile/desktop OS location services can be integrated later behind the same interface.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import aiohttp

from .net import HttpStatusError, RetryConfig, ssl_context, with_retries


@dataclass(frozen=True, slots=True)
class GeoPoint:
    lat: float
    lon: float
    source: str  # fixed | ip | manual
    city: str | None = None


class GeoUnavailable(RuntimeError):
    pass


_LAST: tuple[float, GeoPoint] | None = None


def _retry_cfg() -> RetryConfig:
    return RetryConfig(max_attempts=3, base_delay_sec=0.4, max_delay_sec=4.0)


async def _get_json(url: str, *, timeout_sec: float) -> dict:
    timeout = aiohttp.ClientTimeout(total=float(timeout_sec))
    connector = aiohttp.TCPConnector(ssl=ssl_context())
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True, connector=connector) as s:
        async with s.get(url, headers={"User-Agent": "gpt-station-assistant/1.0"}) as r:
            txt = await r.text()
            if r.status >= 300:
                raise HttpStatusError(r.status, txt, where="geo")
            try:
                return json.loads(txt)
            except Exception:
                raise RuntimeError(f"geo: invalid json: {txt[:500]}")


def _valid_lat_lon(lat: float, lon: float) -> bool:
    if lat == 0.0 and lon == 0.0:
        return False
    return -90.0 <= float(lat) <= 90.0 and -180.0 <= float(lon) <= 180.0


def get_last_known(*, ttl_sec: float) -> GeoPoint | None:
    global _LAST
    if _LAST is None:
        return None
    ts, pt = _LAST
    if (time.time() - ts) > float(ttl_sec):
        return None
    return pt


def set_last_known(pt: GeoPoint) -> None:
    global _LAST
    _LAST = (time.time(), pt)


async def get_location(
    *,
    geo_mode: str,
    fixed_lat: float | None,
    fixed_lon: float | None,
    ip_geo_url: str,
    timeout_sec: float,
    cache_ttl_sec: float,
) -> GeoPoint:
    """Return best-effort current location.

    Supported modes:
    - fixed: require fixed_lat/fixed_lon
    - ip: IP geo only
    - auto: fixed -> last_known -> ip
    """

    mode = (geo_mode or "auto").lower().strip()

    if mode in ("fixed", "auto"):
        if fixed_lat is not None and fixed_lon is not None and _valid_lat_lon(fixed_lat, fixed_lon):
            pt = GeoPoint(lat=float(fixed_lat), lon=float(fixed_lon), source="fixed")
            set_last_known(pt)
            return pt
        if mode == "fixed":
            raise GeoUnavailable("GEO_MODE=fixed but GEO_FIXED_LAT/LON not set")

    # last-known cache
    if mode == "auto":
        cached = get_last_known(ttl_sec=cache_ttl_sec)
        if cached is not None:
            return cached

    if mode in ("ip", "auto"):
        url = (ip_geo_url or "").strip()
        if not url:
            raise GeoUnavailable("IP_GEO_URL is empty")

        async def call() -> dict:
            return await _get_json(url, timeout_sec=timeout_sec)

        data = await with_retries(call, cfg=_retry_cfg(), what="geo.ip")

        lat = data.get("latitude")
        lon = data.get("longitude")
        if lat is None:
            lat = data.get("lat")
        if lon is None:
            lon = data.get("lon")

        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except Exception:
            raise GeoUnavailable(f"IP geo didn't return lat/lon: {data}")

        if not _valid_lat_lon(lat_f, lon_f):
            raise GeoUnavailable(f"Invalid lat/lon from IP geo: {lat_f},{lon_f}")

        city = None
        for k in ("city", "town", "region"):
            if data.get(k):
                city = str(data.get(k))
                break

        pt = GeoPoint(lat=lat_f, lon=lon_f, source="ip", city=city)
        set_last_known(pt)
        return pt

    raise GeoUnavailable(f"Unsupported GEO_MODE={mode}")
