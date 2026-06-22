"""Spotify integration types.

Keep these small and dependency-free so other modules (intent parsing, dialog state)
can import them without pulling in aiohttp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RepeatMode = Literal["off", "context", "track"]


@dataclass(frozen=True, slots=True)
class SpotifyDevice:
    id: str
    name: str
    is_active: bool
    type: str
    volume_percent: int | None = None


@dataclass(frozen=True, slots=True)
class SpotifySearchItem:
    """A minimal representation of a playable item."""

    uri: str
    name: str
    kind: Literal["track", "album", "playlist", "artist"]
    description: str = ""

