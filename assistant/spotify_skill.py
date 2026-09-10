"""Spotify deterministic skill orchestrator.

This module is the single entrypoint used by assistant main loops.

Responsibilities
----------------
- interpret a parsed SpotifyIntent
- call Spotify Web API client
- handle ambiguous requests with a short clarification prompt ("скажи номер")
- return short RU feedback strings suitable for TTS
"""

from __future__ import annotations

import asyncio
import logging
import re

import aiohttp

from .intent_spotify import SpotifyIntent
from .net import HttpStatusError
from .spotify_client import SpotifyApiError, SpotifyAuthRequired, SpotifyClient, SpotifyNoActiveDevice
from .spotify_context import clear_pending, get_pending, set_pending
from .spotify_types import SpotifyDevice


log = logging.getLogger("assistant.spotify_skill")


class SpotifySkillError(RuntimeError):
    pass


async def _ensure_active_device(client: SpotifyClient, preferred_device_name: str | None) -> str | None:
    """Try to make some Spotify device active.

    Spotify may return 204 on /me/player even when devices exist, until playback is
    transferred to one of them.

    Returns the chosen device name (or None if already active / no change).
    """

    devs = await client.get_devices()
    if not devs:
        return None
    if any(d.is_active for d in devs):
        return None

    pref = (preferred_device_name or "").strip().lower()

    def _rank(d: SpotifyDevice) -> tuple[int, int]:
        # lower is better
        t = (d.type or "").strip().lower()
        type_prio = {
            "computer": 0,
            "speaker": 1,
            "tv": 2,
            "avr": 3,
            "smartphone": 4,
            "tablet": 5,
        }.get(t, 10)
        name_hit = 0 if (pref and pref in d.name.strip().lower()) else 1
        return (name_hit, type_prio)

    chosen = sorted(devs, key=_rank)[0]
    await client.transfer_playback(chosen.id, play=False)
    return chosen.name


def _friendly_error(e: BaseException) -> str:
    # Auth
    if isinstance(e, SpotifyAuthRequired):
        return str(e)

    # Common playback issue
    if isinstance(e, SpotifyNoActiveDevice):
        return "Нет активного устройства Spotify. Открой Spotify на телефоне/ПК или скажи: 'какие устройства Spotify?'."

    # Spotify API status mapping
    if isinstance(e, HttpStatusError):
        if e.status == 403:
            return "Нет прав на это действие в Spotify. Переподключи Spotify с нужными разрешениями (scopes)."
        if e.status == 429:
            return "Spotify временно ограничил запросы. Повтори через пару секунд."
        if e.status in (500, 502, 503, 504):
            return "Spotify сейчас недоступен. Попробуй чуть позже."
        return "Не смог выполнить команду Spotify."

    # Network / timeouts
    if isinstance(e, (aiohttp.ClientError, asyncio.TimeoutError)):
        return "Не могу связаться со Spotify (нет сети/таймаут). Попробуй чуть позже."

    # Domain errors
    if isinstance(e, SpotifyApiError):
        return str(e)

    return "Не смог выполнить команду Spotify."


_RE_CHOICE = re.compile(r"\b(перв(ый|ое)|втор(ой|ое)|трет(ий|ье)|четв[её]рт(ый|ое)|пят(ый|ое)|\d{1,2})\b")


def _choice_index_ru(t: str) -> int | None:
    m = _RE_CHOICE.search(t or "")
    if not m:
        return None
    w = m.group(1)
    if w.isdigit():
        i = int(w)
        return i - 1 if i > 0 else None
    w = w.lower()
    mapping = {
        "первый": 0,
        "первое": 0,
        "второй": 1,
        "второе": 1,
        "третий": 2,
        "третье": 2,
        "четвёртый": 3,
        "четвертый": 3,
        "четвёртое": 3,
        "четвертое": 3,
        "пятый": 4,
        "пятое": 4,
    }
    return mapping.get(w)


def _short_device_list(devs: list[SpotifyDevice]) -> str:
    if not devs:
        return "Устройств не вижу. Открой Spotify на телефоне или компьютере."
    parts: list[str] = []
    for i, d in enumerate(devs[:5], start=1):
        mark = " (активно)" if d.is_active else ""
        parts.append(f"{i}) {d.name}{mark}")
    return "Доступные устройства: " + "; ".join(parts)


def _spotify_uri_from_any(s: str) -> str | None:
    t = (s or "").strip()
    if t.startswith("spotify:"):
        return t
    m = re.search(r"https?://open\.spotify\.com/(track|album|playlist|artist)/([A-Za-z0-9]+)", t)
    if not m:
        return None
    kind, sid = m.group(1), m.group(2)
    return f"spotify:{kind}:{sid}"


async def handle_spotify_intent(
    user_intent: SpotifyIntent,
    *,
    client: SpotifyClient,
    normalized_text: str,
    preferred_device_name: str | None = None,
) -> tuple[str, str | None]:
    """Execute Spotify intent.

    Returns: (answer_text, new_preferred_device_name)
    """

    # Connect is handled outside (CLI flow), but we return instruction here.
    if user_intent.action == "spotify_connect":
        return (
            "Чтобы подключить Spotify, запусти на устройстве ассистента: 'python3 -m assistant.spotify_cli auth'. "
            "Я скажу ссылку, открой её в браузере и подтверди доступ.",
            preferred_device_name,
        )

    if not client.is_connected():
        raise SpotifyAuthRequired("Spotify не подключён. Скажи: 'подключи Spotify'.")

    try:
        # 0) Clarification follow-up: user says "первый/второй/3".
        p = get_pending()
        if p is not None:
            idx = _choice_index_ru(normalized_text)
            if idx is not None and 0 <= idx < len(p.items):
                chosen = p.items[idx]
                clear_pending()

                if p.purpose == "play":
                    if chosen.uri.startswith("spotify:track:"):
                        await client.play(uris=[chosen.uri])
                        return (f"Включаю: {chosen.name}.", preferred_device_name)
                    await client.play(context_uri=chosen.uri)
                    return (f"Запускаю: {chosen.name}.", preferred_device_name)

                if p.purpose == "queue":
                    await client.add_to_queue(chosen.uri)
                    return (f"Добавил в очередь: {chosen.name}.", preferred_device_name)

                if p.purpose == "like":
                    tid = chosen.uri.split(":")[-1]
                    await client.like_track(tid)
                    return (f"Добавил в «Понравившиеся»: {chosen.name}.", preferred_device_name)

                if p.purpose == "unlike":
                    tid = chosen.uri.split(":")[-1]
                    await client.unlike_track(tid)
                    return (f"Убрал из «Понравившиеся»: {chosen.name}.", preferred_device_name)

        # 1) Devices
        if user_intent.action == "device_list":
            devs = await client.get_devices()
            return (_short_device_list(devs), preferred_device_name)

        if user_intent.action == "device_set":
            devs = await client.get_devices()
            if not devs:
                return ("Не вижу устройств Spotify. Открой Spotify на телефоне/ПК и повтори.", preferred_device_name)
            name = (user_intent.device_name or "").strip().lower()

            best: SpotifyDevice | None = None
            for d in devs:
                dn = d.name.strip().lower()
                if dn == name or name == dn:
                    best = d
                    break
            if best is None:
                for d in devs:
                    dn = d.name.strip().lower()
                    if name and name in dn:
                        best = d
                        break
            if best is None:
                return (_short_device_list(devs) + ". Скажи: 'переключи на <название>'.", preferred_device_name)

            await client.transfer_playback(best.id, play=True)
            return (f"Переключил на {best.name}.", best.name)

        # 2) Playback controls
        if user_intent.action == "play":
            new_name = await _ensure_active_device(client, preferred_device_name)
            if new_name:
                preferred_device_name = new_name
            await client.play()
            return ("Включил.", preferred_device_name)

        if user_intent.action == "pause":
            new_name = await _ensure_active_device(client, preferred_device_name)
            if new_name:
                preferred_device_name = new_name
            await client.pause()
            return ("Пауза.", preferred_device_name)

        if user_intent.action == "stop":
            # Spotify has no true stop; approximate by pause.
            new_name = await _ensure_active_device(client, preferred_device_name)
            if new_name:
                preferred_device_name = new_name
            await client.pause()
            return ("Остановил.", preferred_device_name)

        if user_intent.action == "next":
            new_name = await _ensure_active_device(client, preferred_device_name)
            if new_name:
                preferred_device_name = new_name
            await client.next_track()
            return ("Следующий трек.", preferred_device_name)

        if user_intent.action == "prev":
            new_name = await _ensure_active_device(client, preferred_device_name)
            if new_name:
                preferred_device_name = new_name
            await client.prev_track()
            return ("Предыдущий трек.", preferred_device_name)

        if user_intent.action == "seek_rel":
            delta = int(user_intent.seconds or 0)
            if delta == 0:
                return ("На сколько секунд перемотать?", preferred_device_name)
            new_name = await _ensure_active_device(client, preferred_device_name)
            if new_name:
                preferred_device_name = new_name
            pb = await client.get_playback()
            cur = int(pb.get("progress_ms") or 0)
            item = pb.get("item") or {}
            dur = int((item.get("duration_ms") or 0) if isinstance(item, dict) else 0)
            new_pos = max(0, cur + delta * 1000)
            if dur > 0:
                new_pos = min(dur - 500, new_pos)
            await client.seek_ms(new_pos)
            if delta > 0:
                return (f"Перемотал вперёд на {abs(delta)} секунд.", preferred_device_name)
            return (f"Перемотал назад на {abs(delta)} секунд.", preferred_device_name)

        if user_intent.action == "volume":
            v = user_intent.volume_percent
            if v is None:
                return ("Скажи громкость в процентах, например: 'громкость 40'.", preferred_device_name)
            new_name = await _ensure_active_device(client, preferred_device_name)
            if new_name:
                preferred_device_name = new_name
            await client.set_volume(int(v))
            return (f"Громкость {max(0, min(100, int(v)))} процентов.", preferred_device_name)

        if user_intent.action == "shuffle":
            if user_intent.shuffle_on is None:
                return ("Включить или выключить перемешивание?", preferred_device_name)
            new_name = await _ensure_active_device(client, preferred_device_name)
            if new_name:
                preferred_device_name = new_name
            await client.set_shuffle(bool(user_intent.shuffle_on))
            return (
                ("Перемешивание включено." if user_intent.shuffle_on else "Перемешивание выключено."),
                preferred_device_name,
            )

        if user_intent.action == "repeat":
            if user_intent.repeat_mode is None:
                return ("Повтор: выключить, альбом/плейлист или один трек?", preferred_device_name)
            new_name = await _ensure_active_device(client, preferred_device_name)
            if new_name:
                preferred_device_name = new_name
            await client.set_repeat(user_intent.repeat_mode)
            msg = {
                "off": "Повтор выключен.",
                "context": "Повтор альбома/плейлиста включён.",
                "track": "Повтор трека включён.",
            }[user_intent.repeat_mode]
            return (msg, preferred_device_name)

        # 3) Library ops
        if user_intent.action in ("like_current", "unlike_current"):
            pb = await client.get_playback()
            item = pb.get("item")
            if not isinstance(item, dict):
                return ("Не понял, какой трек сейчас играет.", preferred_device_name)
            tid = str(item.get("id") or "")
            if not tid:
                return ("Не смог получить id текущего трека.", preferred_device_name)
            if user_intent.action == "like_current":
                await client.like_track(tid)
                return ("Добавил в «Понравившиеся».", preferred_device_name)
            await client.unlike_track(tid)
            return ("Убрал из «Понравившихся».", preferred_device_name)

        # 4) play/queue by name or link
        if user_intent.action in ("play_named", "queue_named"):
            q = (user_intent.uri_or_url or user_intent.query or "").strip()
            if not q:
                return ("Что именно включить? Название или ссылку на Spotify.", preferred_device_name)

            uri = _spotify_uri_from_any(q)
            log.info("play/queue_named: content_kind=%s query=%r uri_from_link=%r", user_intent.content_kind, q, uri)

            if uri is None and user_intent.content_kind == "playlist":
                # Spotify's public /search for type=playlist mostly surfaces
                # curated/public playlists, not the user's own — which is what
                # people usually mean by "включи плейлист <название>". Look
                # there first (same lookup used by playlist_add/playlist_remove).
                pls = await client.get_my_playlists(limit=50, max_pages=3)
                name = q.strip().lower()
                cand = [p for p in pls if p.name.strip().lower() == name]
                if not cand:
                    cand = [p for p in pls if name in p.name.strip().lower()]
                log.info(
                    "playlist lookup in my library: query=%r my_playlists=%r matched=%r",
                    name,
                    [p.name for p in pls],
                    [p.name for p in cand],
                )
                if cand:
                    uri = cand[0].uri

            if uri is None:
                # Decide search types priority.
                # For short queries like "нирвана" users often mean artist.
                words = [w for w in re.split(r"\s+", q.strip()) if w]
                if user_intent.content_kind == "artist":
                    types = ["artist"]
                elif user_intent.content_kind == "playlist":
                    types = ["playlist"]
                elif user_intent.content_kind == "auto" and len(words) <= 2:
                    types = ["artist", "track", "album", "playlist"]
                else:
                    types = ["track", "album", "playlist", "artist"]

                # We intentionally do not ask the user to choose for common "включи <запрос>".
                # Spotify returns relevance-ranked results; we pick the best by type priority.
                items = await client.search(q, types=types, limit=5)
                log.info(
                    "spotify search: query=%r types=%s results=%r",
                    q,
                    types,
                    [(it.kind, it.name, it.uri) for it in items],
                )
                if not items:
                    if user_intent.content_kind == "playlist":
                        return ("Не нашёл такой плейлист ни в твоей библиотеке, ни в поиске Spotify.", preferred_device_name)
                    return ("Не нашёл в Spotify. Скажи точнее: название и исполнителя.", preferred_device_name)
                uri = items[0].uri

            if user_intent.action == "queue_named":
                new_name = await _ensure_active_device(client, preferred_device_name)
                if new_name:
                    preferred_device_name = new_name
                await client.add_to_queue(uri)
                return ("Добавил в очередь.", preferred_device_name)

            # play_named
            new_name = await _ensure_active_device(client, preferred_device_name)
            if new_name:
                preferred_device_name = new_name
            if uri.startswith("spotify:track:"):
                await client.play(uris=[uri])
                return ("Включил трек.", preferred_device_name)
            await client.play(context_uri=uri)
            return ("Запустил.", preferred_device_name)

        if user_intent.action in ("like_named", "unlike_named"):
            q = (user_intent.query or "").strip()
            if not q:
                return ("Какой трек? Скажи название и исполнителя.", preferred_device_name)
            items = await client.search(q, types=["track"], limit=5)
            if not items:
                return ("Не нашёл такой трек в Spotify.", preferred_device_name)
            if len(items) > 1:
                set_pending("like" if user_intent.action == "like_named" else "unlike", items[:5])
                opts = []
                for i, it in enumerate(items[:5], start=1):
                    extra = f" — {it.description}" if it.description else ""
                    opts.append(f"{i}) {it.name}{extra}")
                return ("Какой именно? " + "; ".join(opts) + ". Скажи номер.", preferred_device_name)

            tid = items[0].uri.split(":")[-1]
            if user_intent.action == "like_named":
                await client.like_track(tid)
                return ("Добавил в «Понравившиеся».", preferred_device_name)
            await client.unlike_track(tid)
            return ("Убрал из «Понравившихся».", preferred_device_name)

        if user_intent.action in ("playlist_add", "playlist_remove"):
            pl_name = (user_intent.playlist_name or "").strip()
            if not pl_name:
                return ("В какой плейлист? Скажи название.", preferred_device_name)

            pls = await client.get_my_playlists(limit=50, max_pages=3)
            cand = [p for p in pls if p.name.strip().lower() == pl_name.lower()]
            if not cand:
                cand = [p for p in pls if pl_name.lower() in p.name.strip().lower()]
            if not cand:
                return ("Не нашёл такой плейлист в твоей библиотеке Spotify.", preferred_device_name)

            playlist_id = cand[0].uri.split(":")[-1]

            if user_intent.query:
                items = await client.search(user_intent.query, types=["track"], limit=5)
                if not items:
                    return ("Не нашёл такой трек в Spotify.", preferred_device_name)
                track_uri = items[0].uri
            else:
                item = await client.get_current_track()
                track_uri = str(item.get("uri") or "")

            if not track_uri:
                return ("Не смог определить трек.", preferred_device_name)

            if user_intent.action == "playlist_add":
                await client.playlist_add(playlist_id, track_uri)
                return (f"Добавил в плейлист «{cand[0].name}».", preferred_device_name)
            await client.playlist_remove(playlist_id, track_uri)
            return (f"Убрал из плейлиста «{cand[0].name}».", preferred_device_name)

        raise SpotifySkillError(f"Неизвестное действие Spotify: {user_intent.action}")
    except BaseException as e:  # noqa: BLE001
        return (_friendly_error(e), preferred_device_name)
