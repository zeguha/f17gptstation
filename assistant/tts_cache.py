"""Tiny disk cache for TTS WAV output.

Useful on Raspberry Pi 3B to reduce latency and cost for repeated phrases.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _key(parts: list[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


@dataclass(slots=True)
class TtsCache:
    root: Path

    @classmethod
    def default(cls) -> "TtsCache":
        base = os.environ.get("TTS_CACHE_DIR")
        if base:
            return cls(Path(base))
        return cls(Path.home() / ".cache" / "assistant" / "tts")

    def get(self, *, provider: str, model: str, voice: str, text: str) -> Optional[bytes]:
        k = _key([provider, model, voice, text])
        p = self.root / (k + ".wav")
        try:
            return p.read_bytes() if p.exists() else None
        except Exception:
            return None

    def put(self, *, provider: str, model: str, voice: str, text: str, wav_bytes: bytes) -> None:
        if not wav_bytes:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        k = _key([provider, model, voice, text])
        p = self.root / (k + ".wav")
        try:
            p.write_bytes(wav_bytes)
        except Exception:
            pass

