"""Token storage for Spotify OAuth.

We store refresh_token and short-lived access_token on disk.
No password is ever stored.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass


class SpotifyTokenError(RuntimeError):
    pass


@dataclass(slots=True)
class SpotifyTokens:
    access_token: str
    refresh_token: str
    expires_at: float  # unix seconds

    def is_expired(self, *, skew_sec: float = 30.0) -> bool:
        return time.time() >= float(self.expires_at) - float(skew_sec)


class SpotifyTokenStore:
    def __init__(self, *, path: str):
        self.path = path

    def load(self) -> SpotifyTokens | None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                js = json.load(f)
            return SpotifyTokens(
                access_token=str(js.get("access_token") or ""),
                refresh_token=str(js.get("refresh_token") or ""),
                expires_at=float(js.get("expires_at") or 0.0),
            )
        except FileNotFoundError:
            return None
        except Exception as e:  # noqa: BLE001
            raise SpotifyTokenError(f"Не смог прочитать токены Spotify: {e}")

    def save(self, t: SpotifyTokens) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp_path = self.path + ".tmp"
        data = {
            "access_token": t.access_token,
            "refresh_token": t.refresh_token,
            "expires_at": float(t.expires_at),
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)
        # Best-effort: restrict permissions (works on POSIX).
        try:
            os.chmod(self.path, 0o600)
        except Exception:
            pass

