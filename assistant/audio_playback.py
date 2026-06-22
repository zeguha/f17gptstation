"""Audio playback utilities (Linux/Raspberry Pi friendly).

We prefer simple system players to avoid fighting with low-level audio APIs on SBCs:
- Linux: `aplay` if available
- macOS: `afplay` if available

Fallback: sounddevice output.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
import threading
from typing import Optional


@dataclass(slots=True)
class PlaybackInfo:
    backend: str
    path: Optional[str] = None
    proc: Optional[subprocess.Popen] = None


def play_wav_bytes(wav_bytes: bytes, *, device: Optional[int] = None) -> PlaybackInfo:
    """Play WAV bytes synchronously."""

    if not wav_bytes:
        return PlaybackInfo(backend="noop")

    # Use OS-native CLI players when possible.
    if shutil.which("aplay"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            path = f.name
        try:
            cmd = ["aplay", "-q", path]
            if device is not None:
                # ALSA device syntax: -D plughw:<card>,<device> etc. We cannot infer that from index.
                # Keep it simple: allow user to use ALSA via system config.
                pass
            # Use Popen so we can stop playback (barge-in).
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            proc.wait()
            return PlaybackInfo(backend="aplay", path=path, proc=proc)
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    if shutil.which("afplay"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            path = f.name
        try:
            proc = subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            proc.wait()
            return PlaybackInfo(backend="afplay", path=path, proc=proc)
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    # Fallback: sounddevice
    try:
        import numpy as np
        import sounddevice as sd
        import wave
        import io

        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            raw = w.readframes(n)
        pcm = np.frombuffer(raw, dtype=np.int16)
        sd.play(pcm, samplerate=sr, device=device)
        sd.wait()
        return PlaybackInfo(backend="sounddevice")
    except Exception:
        return PlaybackInfo(backend="failed")


def stop_playback(info: PlaybackInfo) -> None:
    """Best-effort stop playback started by play_wav_bytes()."""

    try:
        if info.proc and info.proc.poll() is None:
            try:
                info.proc.terminate()
                info.proc.wait(timeout=1)
            except Exception:
                try:
                    info.proc.kill()
                except Exception:
                    pass
    except Exception:
        pass

    # Fallback for sounddevice.
    try:
        import sounddevice as sd

        sd.stop()
    except Exception:
        pass
