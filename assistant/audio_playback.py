"""Audio playback utilities (Linux/Raspberry Pi friendly).

We prefer simple system players to avoid fighting with low-level audio APIs on SBCs:
- Linux: `aplay` if available
- macOS: `afplay` if available

Fallback: sounddevice output.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
import threading
from typing import Optional


log = logging.getLogger("assistant.audio_playback")


@dataclass(slots=True)
class PlaybackInfo:
    backend: str
    path: Optional[str] = None
    proc: Optional[subprocess.Popen] = None


def start_playback(wav_bytes: bytes, *, device: Optional[int] = None) -> PlaybackInfo:
    """Start playback and return immediately with a handle that can stop it.

    Split from the old play-and-block-until-done shape specifically so a
    caller (`PlaybackController.play_blocking`) can publish the handle
    *before* the audio finishes, not after — otherwise there is nothing for
    barge-in to call `stop_playback()` on until it's already too late to
    matter. Confirmed on real hardware: without this split, stop-word
    detection fired correctly mid-playback but had no running process left
    to interrupt, so audio always ran to its natural end regardless.
    """

    if not wav_bytes:
        return PlaybackInfo(backend="noop")

    # Use OS-native CLI players when possible.
    if shutil.which("aplay"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            path = f.name
        cmd = ["aplay", "-q", path]
        if device is not None:
            # ALSA device syntax: -D plughw:<card>,<device> etc. We cannot infer that from index.
            # Keep it simple: allow user to use ALSA via system config.
            pass
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return PlaybackInfo(backend="aplay", path=path, proc=proc)

    if shutil.which("afplay"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            path = f.name
        proc = subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return PlaybackInfo(backend="afplay", path=path, proc=proc)

    # Fallback: sounddevice. sd.play() itself is already non-blocking.
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
        return PlaybackInfo(backend="sounddevice")
    except Exception:
        log.exception("sounddevice playback failed to start (wav_bytes=%d, device=%s)", len(wav_bytes), device)
        return PlaybackInfo(backend="failed")


def wait_playback(info: PlaybackInfo) -> None:
    """Block until playback started by `start_playback()` finishes (naturally or via stop_playback())."""

    try:
        if info.proc is not None:
            info.proc.wait()
        elif info.backend == "sounddevice":
            import sounddevice as sd

            sd.wait()
    finally:
        if info.path:
            try:
                os.unlink(info.path)
            except Exception:
                pass


def play_wav_bytes(wav_bytes: bytes, *, device: Optional[int] = None) -> PlaybackInfo:
    """Play WAV bytes synchronously (start, then block until done).

    For callers that don't need early-stop; `PlaybackController` in
    main_cloud_orchestrated.py uses `start_playback`/`wait_playback` directly
    instead so it can support barge-in.
    """

    info = start_playback(wav_bytes, device=device)
    wait_playback(info)
    return info


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
