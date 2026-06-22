"""ASR engine abstraction and implementations.

The project supports multiple backends via a small unified interface.

Backends included:
- Vosk (streaming, local)
- whisper.cpp CLI (batch, local; optional)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import soundfile as sf


@dataclass(frozen=True, slots=True)
class AsrResult:
    text: str
    avg_confidence: Optional[float] = None


class AsrSession(Protocol):
    def accept_audio(self, pcm16: bytes) -> None: ...
    def finalize(self) -> AsrResult: ...


class AsrEngine(Protocol):
    def start_session(self) -> AsrSession: ...


def _avg_word_conf(vosk_json: dict) -> Optional[float]:
    words = vosk_json.get("result")
    if not isinstance(words, list) or not words:
        return None
    confs = []
    for w in words:
        if isinstance(w, dict) and isinstance(w.get("conf"), (int, float)):
            confs.append(float(w["conf"]))
    if not confs:
        return None
    return sum(confs) / len(confs)


class VoskSession:
    def __init__(self, recognizer):
        self._rec = recognizer
        self._last_accepted: bool = False

    def accept_audio(self, pcm16: bytes) -> None:
        if pcm16:
            # Vosk returns True when an utterance is finalized.
            try:
                self._last_accepted = bool(self._rec.AcceptWaveform(pcm16))
            except Exception:
                self._last_accepted = False

    def finalize(self) -> AsrResult:
        try:
            data = json.loads(self._rec.FinalResult())
        except Exception:
            data = {}
        text = (data.get("text") or "").strip().lower()
        return AsrResult(text=text, avg_confidence=_avg_word_conf(data))

    def _consume_if_accepted(self) -> Optional[AsrResult]:
        """Return `Result()` only if the last AcceptWaveform() accepted."""
        if not self._last_accepted:
            return None
        self._last_accepted = False
        try:
            data = json.loads(self._rec.Result())
        except Exception:
            data = {}
        text = (data.get("text") or "").strip().lower()
        if not text:
            return None
        return AsrResult(text=text, avg_confidence=_avg_word_conf(data))


class VoskEngine:
    def __init__(
        self,
        *,
        model_path: str | None = None,
        sample_rate: int = 16_000,
        grammar: Optional[list[str]] = None,
        _model=None,
    ):
        """Create Vosk engine.

        Notes
        -----
        - Loading a Vosk `Model(...)` is expensive in RAM/CPU.
        - For multiple recognizers (wake + stop-word) prefer sharing the same model via `_model`.
        """

        from vosk import KaldiRecognizer, Model

        if _model is None:
            if not model_path or not os.path.isdir(model_path):
                raise RuntimeError(f"Vosk model folder not found: {model_path}")
            _model = Model(model_path)
        self._model = _model
        self.model_path = model_path or ""
        self.sample_rate = int(sample_rate)
        self.grammar = grammar
        self._KaldiRecognizer = KaldiRecognizer

    def clone_with(self, *, grammar: Optional[list[str]] = None, sample_rate: int | None = None) -> "VoskEngine":
        """Create a new engine sharing the same loaded Vosk model."""

        return VoskEngine(
            model_path=self.model_path or None,
            sample_rate=int(sample_rate or self.sample_rate),
            grammar=grammar,
            _model=self._model,
        )

    def start_session(self) -> AsrSession:
        if self.grammar is not None:
            import json as _json

            rec = self._KaldiRecognizer(self._model, self.sample_rate, _json.dumps(self.grammar))
        else:
            rec = self._KaldiRecognizer(self._model, self.sample_rate)
        return VoskSession(rec)


class WhisperCliEngine:
    """whisper.cpp backend (batch) via whisper-cli.

    This is not streaming ASR, but it is local and provides good quality.
    """

    def __init__(
        self,
        *,
        whisper_bin: str,
        model_path: str,
        language: str = "ru",
        sample_rate: int = 16_000,
        timeout_sec: float = 60.0,
    ):
        self.whisper_bin = whisper_bin
        self.model_path = model_path
        self.language = language
        self.sample_rate = int(sample_rate)
        self.timeout_sec = float(timeout_sec)

        if not os.path.isfile(self.whisper_bin):
            raise RuntimeError(f"whisper-cli not found: {self.whisper_bin}")
        if not os.path.isfile(self.model_path):
            raise RuntimeError(f"whisper model not found: {self.model_path}")

    def start_session(self) -> AsrSession:
        return _WhisperCliSession(self)


class _WhisperCliSession:
    def __init__(self, engine: WhisperCliEngine):
        self._engine = engine
        self._buf = bytearray()

    def accept_audio(self, pcm16: bytes) -> None:
        if pcm16:
            self._buf.extend(pcm16)

    def finalize(self) -> AsrResult:
        if not self._buf:
            return AsrResult(text="")

        pcm = np.frombuffer(bytes(self._buf), dtype=np.int16)
        tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmpf.close()
        try:
            sf.write(tmpf.name, pcm, self._engine.sample_rate, subtype="PCM_16")
            proc = subprocess.run(
                [
                    self._engine.whisper_bin,
                    "-m",
                    self._engine.model_path,
                    "-f",
                    tmpf.name,
                    "-l",
                    self._engine.language,
                ],
                capture_output=True,
                text=True,
                timeout=self._engine.timeout_sec,
            )
            text = (proc.stdout or "").strip().lower()
            return AsrResult(text=text)
        except subprocess.TimeoutExpired:
            return AsrResult(text="")
        finally:
            try:
                os.unlink(tmpf.name)
            except Exception:
                pass
