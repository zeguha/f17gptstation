"""Spotify intent detection (RU, rule-based).

This is intentionally heuristic. Complex/rare phrasing can fall back to LLM.

We also support a "clarification" state: user can say "первый/второй/3" after
assistant proposes a list of options.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


SpotifyAction = Literal[
    "spotify_connect",
    "play",
    "pause",
    "stop",
    "next",
    "prev",
    "seek_rel",
    "volume",
    "shuffle",
    "repeat",
    "device_list",
    "device_set",
    "play_named",
    "queue_named",
    "playlist_add",
    "playlist_remove",
    "like_named",
    "unlike_named",
    "like_current",
    "unlike_current",
]


@dataclass(frozen=True, slots=True)
class SpotifyIntent:
    action: SpotifyAction
    confidence: float = 0.75

    # parameters
    seconds: int | None = None
    volume_percent: int | None = None
    shuffle_on: bool | None = None
    repeat_mode: Literal["off", "context", "track"] | None = None

    # device
    device_name: str | None = None

    # content
    query: str | None = None  # name/artist/genre freeform
    uri_or_url: str | None = None
    content_kind: Literal["track", "album", "playlist", "artist", "auto"] = "auto"

    # playlist ops
    playlist_name: str | None = None


_RE_CONNECT = re.compile(r"\b(подключи|привяжи|авторизуй)\s+(spotify|спотифай)\b")

_RE_PLAY = re.compile(r"\b(играй|воспроизведи|включи|продолжи)\b")
_RE_PAUSE = re.compile(r"\b(пауза|приостанови)\b")
_RE_STOP = re.compile(r"\b(стоп|останови)\b")
_RE_NEXT = re.compile(r"\b(следующ(ий|ую)|дальше|переключи)\b")
_RE_PREV = re.compile(r"\b(предыдущ(ий|ую)|назад\s+трек)\b")

_RE_FWD = re.compile(r"\b(?:перемотай|промотай)\s+впер[её]д\s+на\s+(\d{1,4})\s*(?:секунд(?:у|ы)?|с)\b")
_RE_BACK = re.compile(r"\b(?:перемотай|промотай)\s+назад\s+на\s+(\d{1,4})\s*(?:секунд(?:у|ы)?|с)\b")

_RE_VOL = re.compile(r"\b(громкост[ьи]|звук)\s+(на|до)?\s*(\d{1,3})\s*%?\b")

_RE_SHUFFLE_ON = re.compile(r"\b(перемеш(ай|ивание)|шафл)\s*(вкл(ючи)?|включи|on)\b")
_RE_SHUFFLE_OFF = re.compile(r"\b(перемеш(ай|ивание)|шафл)\s*(выкл(ючи)?|выключи|off)\b")

_RE_REPEAT_OFF = re.compile(r"\b(повтор)\s*(выкл(ючи)?|выключи|off)\b")
_RE_REPEAT_TRACK = re.compile(r"\b(повтор)\s*(трек(а)?|одного|this|track)\b")
_RE_REPEAT_CONTEXT = re.compile(r"\b(повтор)\s*(альбом(а)?|плейлист(а)?|всего|контекст|on)\b")

_RE_DEVICES = re.compile(r"\b(какие\s+устройства|устройства\s+spotify|куда\s+воспроизводит)\b")
_RE_SET_DEVICE = re.compile(r"\b(переключи|перенеси|играй)\s+(на|в)\s+(.{2,64})\b")

_RE_LIKE = re.compile(r"\b(лайк|понрав|добавь\s+в\s+понравивш)\b")
_RE_UNLIKE = re.compile(r"\b(убери\s+из\s+понравивш|дизлайк|не\s+нрав)\b")

_RE_URL = re.compile(r"(spotify:(track|album|playlist|artist):[A-Za-z0-9]+|https?://open\.spotify\.com/(track|album|playlist|artist)/[A-Za-z0-9]+)")

_RE_PLAY_NAMED = re.compile(r"\b(включи|запусти|поставь)\s+(.{2,120})\b")
_RE_QUEUE_NAMED = re.compile(r"\b(поставь\s+в\s+очередь|добавь\s+в\s+очередь)\s+(.{2,120})\b")

_RE_PLAY_ARTIST = re.compile(r"\b(включи|запусти|поставь)\s+(исполнителя|артиста)\s+(.{2,120})\b")
_RE_PLAY_GENRE = re.compile(r"\b(включи|запусти|поставь)\s+жанр\s+(.{2,80})\b")

_RE_PLAYLIST_ADD_CURRENT = re.compile(r"\b(добавь|сохрани)\s+(этот\s+трек|текущ(ий|ую)\s+трек)\s+в\s+плейлист\s+(.{2,80})\b")
_RE_PLAYLIST_REMOVE_CURRENT = re.compile(r"\b(удали|убери)\s+(этот\s+трек|текущ(ий|ую)\s+трек)\s+из\s+плейлиста\s+(.{2,80})\b")

_RE_PLAYLIST_ADD_BY_NAME = re.compile(r"\b(добавь|сохрани)\s+(.{2,120})\s+в\s+плейлист\s+(.{2,80})\b")
_RE_PLAYLIST_REMOVE_BY_NAME = re.compile(r"\b(удали|убери)\s+(.{2,120})\s+из\s+плейлиста\s+(.{2,80})\b")

_RE_LIKE_NAMED = re.compile(r"\b(лайкни|добавь\s+в\s+понравившиеся)\s+(.{2,120})\b")
_RE_UNLIKE_NAMED = re.compile(r"\b(убери\s+из\s+понравившихся)\s+(.{2,120})\b")


def _strip_quotes(s: str) -> str:
    s2 = (s or "").strip()
    if (s2.startswith('"') and s2.endswith('"')) or (s2.startswith("'") and s2.endswith("'")):
        return s2[1:-1].strip()
    return s2


def detect_spotify_intent(normalized_text: str) -> SpotifyIntent | None:
    t = (normalized_text or "").strip()
    if not t:
        return None

    if _RE_CONNECT.search(t):
        return SpotifyIntent(action="spotify_connect", confidence=0.9)

    if _RE_DEVICES.search(t):
        return SpotifyIntent(action="device_list", confidence=0.8)

    # Like/unlike current
    if _RE_LIKE.search(t):
        return SpotifyIntent(action="like_current", confidence=0.75)
    if _RE_UNLIKE.search(t):
        return SpotifyIntent(action="unlike_current", confidence=0.75)

    if _RE_SHUFFLE_ON.search(t):
        return SpotifyIntent(action="shuffle", shuffle_on=True, confidence=0.8)
    if _RE_SHUFFLE_OFF.search(t):
        return SpotifyIntent(action="shuffle", shuffle_on=False, confidence=0.8)

    if _RE_REPEAT_OFF.search(t):
        return SpotifyIntent(action="repeat", repeat_mode="off", confidence=0.8)
    if _RE_REPEAT_TRACK.search(t):
        return SpotifyIntent(action="repeat", repeat_mode="track", confidence=0.8)
    if _RE_REPEAT_CONTEXT.search(t):
        return SpotifyIntent(action="repeat", repeat_mode="context", confidence=0.7)

    m = _RE_FWD.search(t)
    if m:
        return SpotifyIntent(action="seek_rel", seconds=int(m.group(1)), confidence=0.8)
    m = _RE_BACK.search(t)
    if m:
        return SpotifyIntent(action="seek_rel", seconds=-int(m.group(1)), confidence=0.8)

    m = _RE_VOL.search(t)
    if m:
        return SpotifyIntent(action="volume", volume_percent=int(m.group(3)), confidence=0.8)

    if _RE_PAUSE.search(t):
        return SpotifyIntent(action="pause", confidence=0.8)
    if _RE_STOP.search(t):
        return SpotifyIntent(action="stop", confidence=0.75)
    if _RE_NEXT.search(t):
        return SpotifyIntent(action="next", confidence=0.75)
    if _RE_PREV.search(t):
        return SpotifyIntent(action="prev", confidence=0.75)

    # Play by artist/genre (must be checked before generic "play" and generic play_named)
    m = _RE_PLAY_ARTIST.search(t)
    if m:
        q = _strip_quotes(m.group(3))
        urlm = _RE_URL.search(q)
        return SpotifyIntent(
            action="play_named",
            query=q,
            uri_or_url=(urlm.group(1) if urlm else None),
            content_kind="artist",
            confidence=0.75,
        )
    m = _RE_PLAY_GENRE.search(t)
    if m:
        q = _strip_quotes(m.group(2))
        # Use Spotify search query syntax.
        return SpotifyIntent(action="play_named", query=f"genre:{q}", content_kind="track", confidence=0.65)

    # Queue / play named
    m = _RE_QUEUE_NAMED.search(t)
    if m:
        q = _strip_quotes(m.group(2))
        urlm = _RE_URL.search(q)
        return SpotifyIntent(action="queue_named", query=q, uri_or_url=(urlm.group(1) if urlm else None), confidence=0.75)

    m = _RE_PLAY_NAMED.search(t)
    if m and _RE_PLAY.search(t):
        q = _strip_quotes(m.group(2))
        urlm = _RE_URL.search(q)
        return SpotifyIntent(action="play_named", query=q, uri_or_url=(urlm.group(1) if urlm else None), confidence=0.7)

    # Device set by name (heuristic)
    m = _RE_SET_DEVICE.search(t)
    if m:
        dev = _strip_quotes(m.group(3))
        # Avoid catching generic phrases like "на 50" (volume).
        if not re.fullmatch(r"\d{1,3}%?", dev):
            return SpotifyIntent(action="device_set", device_name=dev, confidence=0.65)

    # Playlist ops (current track)
    m = _RE_PLAYLIST_ADD_CURRENT.search(t)
    if m:
        pl = _strip_quotes(m.group(4))
        return SpotifyIntent(action="playlist_add", playlist_name=pl, query=None, confidence=0.8)
    m = _RE_PLAYLIST_REMOVE_CURRENT.search(t)
    if m:
        pl = _strip_quotes(m.group(4))
        return SpotifyIntent(action="playlist_remove", playlist_name=pl, query=None, confidence=0.8)

    # Playlist ops by track name
    m = _RE_PLAYLIST_ADD_BY_NAME.search(t)
    if m and "плейлист" in t:
        q = _strip_quotes(m.group(2))
        pl = _strip_quotes(m.group(3))
        return SpotifyIntent(action="playlist_add", playlist_name=pl, query=q, confidence=0.65)
    m = _RE_PLAYLIST_REMOVE_BY_NAME.search(t)
    if m and "плейлист" in t:
        q = _strip_quotes(m.group(2))
        pl = _strip_quotes(m.group(3))
        return SpotifyIntent(action="playlist_remove", playlist_name=pl, query=q, confidence=0.65)

    # Like/unlike by name
    m = _RE_LIKE_NAMED.search(t)
    if m:
        q = _strip_quotes(m.group(2))
        return SpotifyIntent(action="like_named", query=q, confidence=0.7)
    m = _RE_UNLIKE_NAMED.search(t)
    if m:
        q = _strip_quotes(m.group(2))
        return SpotifyIntent(action="unlike_named", query=q, confidence=0.7)

    # Generic play (keep near the end to avoid shadowing more specific branches)
    if t in ("играй", "воспроизведение", "продолжи") or _RE_PLAY.search(t):
        return SpotifyIntent(action="play", confidence=0.65)

    return None
