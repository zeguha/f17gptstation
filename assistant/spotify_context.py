"""In-memory dialog context for Spotify disambiguation.

We keep this module extremely small and process-local.

Rationale
---------
The assistant currently has no generic slot-filling state machine.
To support short clarifications like "первый/второй/3" after presenting options,
we store the last proposed choices in memory with a short TTL.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .spotify_types import SpotifySearchItem


@dataclass(slots=True)
class SpotifyPendingChoice:
    ts: float
    purpose: str  # e.g. play|queue|playlist_add|playlist_remove
    items: list[SpotifySearchItem]


_PENDING: SpotifyPendingChoice | None = None


def set_pending(purpose: str, items: list[SpotifySearchItem]) -> None:
    global _PENDING
    _PENDING = SpotifyPendingChoice(ts=time.time(), purpose=str(purpose), items=list(items))


def clear_pending() -> None:
    global _PENDING
    _PENDING = None


def get_pending(*, max_age_sec: float = 90.0) -> SpotifyPendingChoice | None:
    p = _PENDING
    if p is None:
        return None
    if (time.time() - float(p.ts)) > float(max_age_sec):
        clear_pending()
        return None
    return p

