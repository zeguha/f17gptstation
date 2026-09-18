"""Audio capture utilities.

This module provides a non-blocking, callback-based microphone stream that yields
fixed-size PCM16 mono frames via a queue.

Design goals:
- Cross-platform (sounddevice)
- Low latency (small frame size, no heavy work in callback)
- Testable (core logic isolated)
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Optional

import sounddevice as sd
import numpy as np


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """A chunk of PCM audio with an approximate monotonic timestamp."""

    pcm16: bytes
    t_monotonic: float


def _apply_gain_int16(pcm16: bytes, gain: float) -> bytes:
    """Scale PCM16 mono samples by `gain`, clipped to avoid wraparound distortion."""

    if not pcm16:
        return pcm16
    x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    x *= gain
    np.clip(x, -32768, 32767, out=x)
    return x.astype(np.int16).tobytes()


class AudioStream:
    """Non-blocking microphone stream that outputs fixed-size PCM16 mono frames.

    Notes
    -----
    - Uses `sounddevice.RawInputStream` with dtype=int16, channels=1.
    - Callback is minimal: enqueue bytes, drop oldest on overflow.
    - `read()` blocks for up to `timeout` seconds.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        frame_samples: int = 480,
        device: Optional[int] = None,
        queue_max_chunks: int = 200,
        ignore_event=None,
        input_gain: float = 1.0,
    ):
        self.sample_rate = int(sample_rate)
        self.frame_samples = int(frame_samples)
        self.device = device
        self.ignore_event = ignore_event
        # Digital pre-amp applied to every captured frame before it reaches
        # VAD/ASR. Useful when the mic has no hardware capture-level control
        # (confirmed: some USB devices expose only an on/off capture switch,
        # no volume) and turning the speaker down isn't wanted — e.g. so a
        # quiet "стоп" barge-in registers without shouting, without making
        # the assistant's own TTS answers harder to hear.
        self.input_gain = float(input_gain)

        # Output frame subscribers.
        # We broadcast the same normalized frames to multiple independent readers
        # (wake/command pipeline, stop-word listener, debug recorder, etc.).
        self._q: "queue.Queue[AudioChunk]" = queue.Queue(maxsize=int(queue_max_chunks))
        self._subs: list["queue.Queue[AudioChunk]"] = [self._q]
        self._stream: Optional[sd.InputStream] = None

        # Capture side (may differ from `self.sample_rate` on macOS / some devices)
        self.capture_rate: int = self.sample_rate
        self._raw_q: "queue.Queue[tuple[bytes, float]]" = queue.Queue(maxsize=int(queue_max_chunks) * 3)
        self._raw_buf = bytearray()
        self._resample_buf = bytearray()

        # derived
        self.frame_bytes = self.frame_samples * 2  # mono int16

    def __enter__(self) -> "AudioStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self._stream is not None:
            return

        # macOS AUHAL is picky about samplerate/blocksize.
        # Try to open at 16k first; if it fails, try 48k/32k and downsample to 16k.
        candidate_rates: list[int] = [self.sample_rate, 48_000, 32_000]
        try:
            dev_info = sd.query_devices(self.device, kind="input")
            default_rate = int(float(dev_info.get("default_samplerate") or 0) or 0)
            if default_rate and default_rate not in candidate_rates:
                candidate_rates.append(default_rate)
        except Exception:
            default_rate = 0

        def _cb(indata, frames, time_info, status):
            # Never do heavy work here.
            if status:
                pass
            if self.ignore_event is not None and getattr(self.ignore_event, "is_set", lambda: False)():
                return
            try:
                pcm = indata.tobytes()
            except Exception:
                return
            try:
                self._raw_q.put_nowait((pcm, time.monotonic()))
            except queue.Full:
                # Drop to keep bounded latency
                try:
                    _ = self._raw_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._raw_q.put_nowait((pcm, time.monotonic()))
                except queue.Full:
                    pass

        last_err: Optional[Exception] = None
        for sr in candidate_rates:
            try:
                # blocksize=0 lets PortAudio pick, often fixes AUHAL -10851.
                self._stream = sd.InputStream(
                    samplerate=int(sr),
                    blocksize=0,
                    device=self.device,
                    dtype="int16",
                    channels=1,
                    callback=_cb,
                    latency="low",
                )
                self._stream.start()
                self.capture_rate = int(sr)
                break
            except Exception as e:
                last_err = e
                try:
                    if self._stream is not None:
                        self._stream.close()
                except Exception:
                    pass
                self._stream = None

        if self._stream is None:
            raise RuntimeError(f"Failed to open input stream (device={self.device}, rates={candidate_rates}): {last_err}")

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
        finally:
            try:
                self._stream.close()
            finally:
                self._stream = None

    def clear(self) -> None:
        """Drop all queued audio for the default reader.

        Notes
        -----
        This does NOT clear other subscribers, so a stop-word listener can keep running
        independently.
        """
        self.clear_queue(self._q)

    def clear_queue(self, q: "queue.Queue[AudioChunk]") -> None:
        """Drop queued audio for a specific subscriber queue."""
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def subscribe(self, *, queue_max_chunks: int = 200) -> "queue.Queue[AudioChunk]":
        q: "queue.Queue[AudioChunk]" = queue.Queue(maxsize=int(queue_max_chunks))
        self._subs.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue[AudioChunk]") -> None:
        try:
            self._subs.remove(q)
        except ValueError:
            return
        # Drain raw buffers
        while True:
            try:
                self._raw_q.get_nowait()
            except queue.Empty:
                break
        self._raw_buf.clear()
        self._resample_buf.clear()

    def _downsample_int16(self, pcm16: bytes, src_rate: int, dst_rate: int) -> bytes:
        """Downsample PCM16 mono.

        Designed for common integer factors (48k→16k, 32k→16k).
        """

        if src_rate == dst_rate:
            return pcm16
        if not pcm16:
            return b""

        x = np.frombuffer(pcm16, dtype=np.int16)

        if src_rate % dst_rate == 0:
            factor = src_rate // dst_rate
            if factor <= 1:
                return pcm16
            # Simple decimation. For voice + VAD this is usually fine.
            y = x[::factor]
            return y.astype(np.int16, copy=False).tobytes()

        # Fallback: linear resample (slower, but no extra deps)
        x_f = x.astype(np.float32)
        n_dst = int(round(len(x_f) * (dst_rate / src_rate)))
        if n_dst <= 0:
            return b""
        src_idx = np.linspace(0, len(x_f) - 1, num=len(x_f), dtype=np.float32)
        dst_idx = np.linspace(0, len(x_f) - 1, num=n_dst, dtype=np.float32)
        y = np.interp(dst_idx, src_idx, x_f)
        y = np.clip(np.round(y), -32768, 32767).astype(np.int16)
        return y.tobytes()

    def _pump_frames(self, timeout: float) -> None:
        """Move data from raw queue into fixed-size output frames."""

        # Pull at least one raw packet (or timeout).
        try:
            pcm, t0 = self._raw_q.get(timeout=float(timeout))
            self._raw_buf.extend(pcm)
            t_stamp = t0
        except queue.Empty:
            return

        # Drain the rest quickly (reduce overhead)
        while True:
            try:
                pcm2, _t2 = self._raw_q.get_nowait()
                self._raw_buf.extend(pcm2)
            except queue.Empty:
                break

        # Convert to desired sample rate if needed.
        if self.capture_rate != self.sample_rate:
            converted = self._downsample_int16(bytes(self._raw_buf), self.capture_rate, self.sample_rate)
            self._raw_buf.clear()
            self._resample_buf.extend(converted)
            buf = self._resample_buf
        else:
            buf = self._raw_buf

        # Emit fixed frames.
        while len(buf) >= self.frame_bytes:
            frame = bytes(buf[: self.frame_bytes])
            del buf[: self.frame_bytes]
            if self.input_gain != 1.0:
                frame = _apply_gain_int16(frame, self.input_gain)
            chunk = AudioChunk(pcm16=frame, t_monotonic=t_stamp)

            # Broadcast with bounded latency: drop oldest on overflow.
            for sub in list(self._subs):
                try:
                    sub.put_nowait(chunk)
                except queue.Full:
                    try:
                        _ = sub.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        sub.put_nowait(chunk)
                    except queue.Full:
                        pass

    def read(self, timeout: float = 0.5, *, q: "queue.Queue[AudioChunk]" | None = None) -> Optional[AudioChunk]:
        """Read next frame from the default queue or from a subscriber queue."""

        target = q or self._q
        # Ensure output queue has fixed frames.
        if target.empty():
            self._pump_frames(timeout=timeout)
        try:
            return target.get(timeout=float(timeout))
        except queue.Empty:
            return None
