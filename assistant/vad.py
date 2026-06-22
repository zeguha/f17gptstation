"""Voice activity detection helpers.

Uses webrtcvad when available; falls back to simple energy gating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class VadConfig:
    sample_rate: int = 16_000
    frame_ms: int = 30
    webrtcvad_mode: int = 2  # 0..3 (3 is most aggressive)
    energy_threshold: float = 0.010  # mean(|float|) threshold fallback


class Vad:
    """VAD over fixed-size PCM16 mono frames."""

    def __init__(self, cfg: VadConfig):
        self.cfg = cfg
        self._webrtcvad = None
        try:
            # Some versions of `webrtcvad` emit a deprecation warning because they
            # import `pkg_resources` internally. It's harmless at runtime.
            import warnings

            warnings.filterwarnings(
                "ignore",
                message=r"pkg_resources is deprecated as an API.*",
                category=UserWarning,
            )
            import webrtcvad

            self._webrtcvad = webrtcvad.Vad(int(cfg.webrtcvad_mode))
        except Exception:
            self._webrtcvad = None

        self.frame_samples = int(cfg.sample_rate * cfg.frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2

    @property
    def have_webrtcvad(self) -> bool:
        return self._webrtcvad is not None

    def is_speech(self, pcm16: bytes) -> bool:
        if not pcm16:
            return False
        if len(pcm16) != self.frame_bytes:
            # Caller should provide fixed frames; be defensive.
            if len(pcm16) < self.frame_bytes:
                pcm16 = pcm16 + b"\x00" * (self.frame_bytes - len(pcm16))
            else:
                pcm16 = pcm16[: self.frame_bytes]

        if self._webrtcvad is not None:
            try:
                return bool(self._webrtcvad.is_speech(pcm16, int(self.cfg.sample_rate)))
            except Exception:
                # fall through
                pass

        # Energy fallback: compute mean(abs(float32))
        arr = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
        m = float(np.mean(np.abs(arr)) / 32768.0)
        return m >= float(self.cfg.energy_threshold)


@dataclass(frozen=True, slots=True)
class VadStats:
    speech_frames: int
    total_frames: int


def count_speech_frames(vad: Vad, frames: list[bytes]) -> VadStats:
    s = 0
    for f in frames:
        if vad.is_speech(f):
            s += 1
    return VadStats(speech_frames=s, total_frames=len(frames))
