"""Provider interfaces (STT/LLM/TTS)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True, slots=True)
class SttResult:
    text: str
    raw: Optional[dict] = None


@dataclass(frozen=True, slots=True)
class LlmResult:
    text: str
    raw: Optional[dict] = None


@dataclass(frozen=True, slots=True)
class TtsResult:
    wav_bytes: bytes
    raw: Optional[dict] = None


class SttClient(Protocol):
    async def transcribe_wav(self, wav_bytes: bytes, *, language: str = "ru") -> SttResult: ...


class LlmClient(Protocol):
    async def chat(self, messages: list[dict]) -> LlmResult: ...


class TtsClient(Protocol):
    async def synthesize(self, text: str, *, language: str = "ru") -> TtsResult: ...

