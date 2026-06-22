"""Spotify Web API client (async).

Implements:
- token refresh on 401
- retry/backoff for transient errors
- basic rate limit handling (429 + Retry-After)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

import aiohttp

from .net import AsyncLimiter, HttpStatusError, RetryConfig, ssl_context, with_retries
from .spotify_oauth import refresh_access_token
from .spotify_tokens import SpotifyTokenStore, SpotifyTokens
from .spotify_types import RepeatMode, SpotifyDevice, SpotifySearchItem


class SpotifyApiError(RuntimeError):
    pass


class SpotifyAuthRequired(SpotifyApiError):
    pass


class SpotifyNoActiveDevice(SpotifyApiError):
    pass


@dataclass(slots=True)
class SpotifyClientConfig:
    client_id: str
    token_store_path: str
    timeout_sec: float = 10.0
    max_in_flight: int = 2
    max_attempts: int = 4
    base_delay_sec: float = 0.3
    max_delay_sec: float = 4.0


def _parse_devices(js: dict) -> list[SpotifyDevice]:
    out: list[SpotifyDevice] = []
    for d in (js.get("devices") or []):
        if not isinstance(d, dict):
            continue
        did = str(d.get("id") or "")
        name = str(d.get("name") or "")
        if not did or not name:
            continue
        out.append(
            SpotifyDevice(
                id=did,
                name=name,
                is_active=bool(d.get("is_active")),
                type=str(d.get("type") or ""),
                volume_percent=(int(d["volume_percent"]) if d.get("volume_percent") is not None else None),
            )
        )
    return out


def _extract_retry_after(resp: aiohttp.ClientResponse) -> float | None:
    ra = resp.headers.get("Retry-After")
    if not ra:
        return None
    try:
        return max(0.0, float(ra))
    except Exception:
        return None


class SpotifyClient:
    def __init__(self, cfg: SpotifyClientConfig):
        self.cfg = cfg
        self._store = SpotifyTokenStore(path=cfg.token_store_path)
        self._tokens: SpotifyTokens | None = self._store.load()
        self._limiter = AsyncLimiter(max_in_flight=int(cfg.max_in_flight))

    def is_connected(self) -> bool:
        t = self._tokens
        return bool(t and t.refresh_token)

    async def ensure_fresh_token(self) -> SpotifyTokens:
        t = self._tokens
        if t is None or not t.refresh_token:
            raise SpotifyAuthRequired("Spotify не подключён. Скажи: 'подключи Spotify'.")
        if not t.is_expired():
            return t
        new_t = await refresh_access_token(client_id=self.cfg.client_id, refresh_token=t.refresh_token, timeout_sec=20.0)
        self._tokens = new_t
        self._store.save(new_t)
        return new_t

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
        allow_204: bool = True,
        where: str = "spotify",
    ) -> tuple[int, str, dict]:
        """Returns (status, text, headers_dict)."""

        url = "https://api.spotify.com/v1" + path

        async def _call() -> tuple[int, str, dict]:
            tokens = await self.ensure_fresh_token()
            timeout = aiohttp.ClientTimeout(total=float(self.cfg.timeout_sec))
            connector = aiohttp.TCPConnector(ssl=ssl_context())
            headers = {
                "Authorization": f"Bearer {tokens.access_token}",
                "Content-Type": "application/json",
            }
            async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=True) as s:
                async with s.request(method, url, params=params, json=json_body, headers=headers) as resp:
                    if allow_204 and resp.status == 204:
                        return resp.status, "", dict(resp.headers)
                    txt = await resp.text()
                    return resp.status, txt, dict(resp.headers)

        # Transient retry loop that also respects 429 Retry-After.
        retry_cfg = RetryConfig(
            max_attempts=int(self.cfg.max_attempts),
            base_delay_sec=float(self.cfg.base_delay_sec),
            max_delay_sec=float(self.cfg.max_delay_sec),
        )

        last_err: BaseException | None = None
        for attempt in range(1, retry_cfg.max_attempts + 1):
            try:
                async with self._limiter:
                    status, txt, headers = await with_retries(_call, cfg=retry_cfg, what=where)
                if status == 401:
                    # Access token invalid/expired unexpectedly. Force refresh once.
                    t = self._tokens
                    if t is None or not t.refresh_token:
                        raise SpotifyAuthRequired("Spotify не подключён.")
                    new_t = await refresh_access_token(
                        client_id=self.cfg.client_id,
                        refresh_token=t.refresh_token,
                        timeout_sec=20.0,
                    )
                    self._tokens = new_t
                    self._store.save(new_t)
                    # retry immediately
                    continue

                if status == 429:
                    # Respect Retry-After, then retry.
                    ra = None
                    try:
                        ra = float(headers.get("Retry-After") or "")
                    except Exception:
                        ra = None
                    await asyncio.sleep(max(0.5, ra or 1.0))
                    continue

                if status >= 300:
                    raise HttpStatusError(status, txt, where=where)
                return status, txt, headers
            except BaseException as e:  # noqa: BLE001
                last_err = e
                if attempt >= retry_cfg.max_attempts:
                    raise
                await asyncio.sleep(min(retry_cfg.max_delay_sec, retry_cfg.base_delay_sec * (2 ** (attempt - 1))))

        assert last_err is not None
        raise last_err

    async def _player_request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
        allow_204: bool = True,
        where: str,
    ) -> tuple[int, str, dict]:
        """Like _request(), but maps "no active device" errors consistently.

        Spotify playback endpoints may return 404 when there is no active device.
        We convert it to SpotifyNoActiveDevice so the skill can return a friendly RU message.
        """

        try:
            return await self._request(
                method,
                path,
                params=params,
                json_body=json_body,
                allow_204=allow_204,
                where=where,
            )
        except HttpStatusError as e:
            if e.status == 404:
                raise SpotifyNoActiveDevice("Нет активного устройства Spotify") from e
            raise

    async def get_devices(self) -> list[SpotifyDevice]:
        _status, txt, _h = await self._request("GET", "/me/player/devices", where="spotify devices")
        js = json.loads(txt or "{}")
        return _parse_devices(js)

    async def get_my_playlists(self, *, limit: int = 50, max_pages: int = 3) -> list[SpotifySearchItem]:
        out: list[SpotifySearchItem] = []
        offset = 0
        for _ in range(int(max_pages)):
            _status, txt, _h = await self._request(
                "GET",
                "/me/playlists",
                params={"limit": int(limit), "offset": int(offset)},
                where="spotify me/playlists",
            )
            js = json.loads(txt or "{}")
            items = js.get("items") or []
            if not isinstance(items, list) or not items:
                break
            for p in items:
                if not isinstance(p, dict):
                    continue
                uri = str(p.get("uri") or "")
                name = str(p.get("name") or "")
                if uri and name:
                    out.append(SpotifySearchItem(uri=uri, name=name, kind="playlist", description=""))
            if not js.get("next"):
                break
            offset += int(limit)
        return out

    async def get_current_track(self) -> dict:
        pb = await self.get_playback()
        item = pb.get("item")
        if not isinstance(item, dict):
            raise SpotifyApiError("Не вижу текущий трек")
        return item

    async def transfer_playback(self, device_id: str, *, play: bool = True) -> None:
        await self._player_request(
            "PUT",
            "/me/player",
            json_body={"device_ids": [device_id], "play": bool(play)},
            where="spotify transfer",
        )

    async def get_playback(self) -> dict:
        # 204 means no active device.
        try:
            status, txt, _h = await self._request("GET", "/me/player", allow_204=True, where="spotify playback")
        except HttpStatusError as e:
            if e.status == 404:
                raise SpotifyNoActiveDevice("Нет активного устройства Spotify")
            raise
        if status == 204:
            raise SpotifyNoActiveDevice("Нет активного устройства Spotify")
        return json.loads(txt or "{}")

    async def play(self, *, device_id: str | None = None, context_uri: str | None = None, uris: list[str] | None = None) -> None:
        params = {"device_id": device_id} if device_id else None
        body: dict = {}
        if context_uri:
            body["context_uri"] = context_uri
        if uris:
            body["uris"] = uris
        await self._player_request(
            "PUT",
            "/me/player/play",
            params=params,
            json_body=body or None,
            where="spotify play",
        )

    async def pause(self, *, device_id: str | None = None) -> None:
        params = {"device_id": device_id} if device_id else None
        await self._player_request("PUT", "/me/player/pause", params=params, where="spotify pause")

    async def next_track(self) -> None:
        await self._player_request("POST", "/me/player/next", where="spotify next")

    async def prev_track(self) -> None:
        await self._player_request("POST", "/me/player/previous", where="spotify prev")

    async def seek_ms(self, pos_ms: int) -> None:
        await self._player_request("PUT", "/me/player/seek", params={"position_ms": int(pos_ms)}, where="spotify seek")

    async def set_volume(self, volume_percent: int) -> None:
        v = max(0, min(100, int(volume_percent)))
        await self._player_request("PUT", "/me/player/volume", params={"volume_percent": v}, where="spotify volume")

    async def set_shuffle(self, on: bool) -> None:
        await self._player_request(
            "PUT",
            "/me/player/shuffle",
            params={"state": str(bool(on)).lower()},
            where="spotify shuffle",
        )

    async def set_repeat(self, mode: RepeatMode) -> None:
        await self._player_request("PUT", "/me/player/repeat", params={"state": mode}, where="spotify repeat")

    async def add_to_queue(self, uri: str) -> None:
        await self._player_request("POST", "/me/player/queue", params={"uri": uri}, where="spotify queue")

    async def like_track(self, track_id: str) -> None:
        await self._request("PUT", "/me/tracks", params={"ids": track_id}, where="spotify like")

    async def unlike_track(self, track_id: str) -> None:
        await self._request("DELETE", "/me/tracks", params={"ids": track_id}, where="spotify unlike")

    async def playlist_add(self, playlist_id: str, track_uri: str) -> None:
        await self._request(
            "POST",
            f"/playlists/{playlist_id}/tracks",
            json_body={"uris": [track_uri]},
            where="spotify playlist add",
        )

    async def playlist_remove(self, playlist_id: str, track_uri: str) -> None:
        await self._request(
            "DELETE",
            f"/playlists/{playlist_id}/tracks",
            json_body={"tracks": [{"uri": track_uri}]},
            where="spotify playlist remove",
        )

    async def search(
        self,
        q: str,
        *,
        types: list[str],
        limit: int = 5,
        market: str | None = None,
    ) -> list[SpotifySearchItem]:
        # NOTE:
        # - Spotify /search works without `market`.
        # - Some accounts/tokens may return HTTP 403 "Insufficient client scope" when `market=from_token` is passed.
        #   (Observed in this project during voice command "включи nirvana").
        params: dict[str, object] = {
            "q": q,
            "type": ",".join(types),
            "limit": int(limit),
        }
        if market:
            params["market"] = market
        _status, txt, _h = await self._request("GET", "/search", params=params, where="spotify search")
        js = json.loads(txt or "{}")

        # Parse result lists first, then emit in the order requested by `types`.
        # This allows the caller (skill) to express intent preference, e.g. prefer artist for 1-word queries.
        parsed: dict[str, list[SpotifySearchItem]] = {
            "track": [],
            "album": [],
            "playlist": [],
            "artist": [],
        }

        tracks = (((js.get("tracks") or {}) if isinstance(js.get("tracks"), dict) else {}).get("items") or [])
        for t in tracks:
            if not isinstance(t, dict):
                continue
            uri = str(t.get("uri") or "")
            name = str(t.get("name") or "")
            if not uri or not name:
                continue
            artists = []
            for a in (t.get("artists") or []):
                if isinstance(a, dict) and a.get("name"):
                    artists.append(str(a["name"]))
            desc = " — ".join(artists) if artists else ""
            parsed["track"].append(SpotifySearchItem(uri=uri, name=name, kind="track", description=desc))

        albums = (((js.get("albums") or {}) if isinstance(js.get("albums"), dict) else {}).get("items") or [])
        for a in albums:
            if not isinstance(a, dict):
                continue
            uri = str(a.get("uri") or "")
            name = str(a.get("name") or "")
            if not uri or not name:
                continue
            artists = []
            for ar in (a.get("artists") or []):
                if isinstance(ar, dict) and ar.get("name"):
                    artists.append(str(ar["name"]))
            desc = " — ".join(artists) if artists else ""
            parsed["album"].append(SpotifySearchItem(uri=uri, name=name, kind="album", description=desc))

        playlists = (((js.get("playlists") or {}) if isinstance(js.get("playlists"), dict) else {}).get("items") or [])
        for p in playlists:
            if not isinstance(p, dict):
                continue
            uri = str(p.get("uri") or "")
            name = str(p.get("name") or "")
            if not uri or not name:
                continue
            owner = (p.get("owner") or {}) if isinstance(p.get("owner"), dict) else {}
            desc = str(owner.get("display_name") or "")
            parsed["playlist"].append(SpotifySearchItem(uri=uri, name=name, kind="playlist", description=desc))

        artists = (((js.get("artists") or {}) if isinstance(js.get("artists"), dict) else {}).get("items") or [])
        for ar in artists:
            if not isinstance(ar, dict):
                continue
            uri = str(ar.get("uri") or "")
            name = str(ar.get("name") or "")
            if not uri or not name:
                continue
            parsed["artist"].append(SpotifySearchItem(uri=uri, name=name, kind="artist", description=""))

        out: list[SpotifySearchItem] = []
        for kind in types:
            out.extend(parsed.get(kind, []))
        return out
