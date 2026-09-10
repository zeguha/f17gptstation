"""OpenAI implementations for STT/LLM/TTS.

Uses direct HTTPS calls via aiohttp to keep dependencies minimal.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from dataclasses import field
import ssl

import aiohttp

from .cloud_interfaces import LlmClient, LlmResult, SttClient, SttResult, TtsClient, TtsResult
from .config import CloudConfig
from .net import AsyncLimiter, HttpStatusError, RetryConfig, with_retries


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class OpenAIHttp:
    cfg: CloudConfig
    _limiter_obj: AsyncLimiter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._limiter_obj = AsyncLimiter(max_in_flight=int(self.cfg.cloud_max_in_flight))

    def _ssl_context(self) -> ssl.SSLContext | None:
        """Return SSL context.

        On some macOS Python installs (especially from python.org), the system CA bundle
        may be missing which causes:
        `SSLCertVerificationError: unable to get local issuer certificate`.

        We prefer a secure fix via `certifi` if available.
        """

        try:
            import certifi  # type: ignore

            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return None

    def _retry_cfg(self) -> RetryConfig:
        return RetryConfig(
            max_attempts=int(self.cfg.cloud_max_attempts),
            base_delay_sec=float(self.cfg.cloud_base_delay_sec),
            max_delay_sec=float(self.cfg.cloud_max_delay_sec),
        )

    def _limiter(self) -> AsyncLimiter:
        return self._limiter_obj

    def _headers(self) -> dict:
        if not self.cfg.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return {"Authorization": f"Bearer {self.cfg.openai_api_key}"}


class OpenAIStt(OpenAIHttp, SttClient):
    async def transcribe_wav(self, wav_bytes: bytes, *, language: str = "ru") -> SttResult:
        url = self.cfg.openai_base_url.rstrip("/") + "/audio/transcriptions"
        data = aiohttp.FormData()
        data.add_field("model", self.cfg.openai_stt_model)
        data.add_field("language", language)
        if self.cfg.openai_stt_prompt:
            data.add_field("prompt", self.cfg.openai_stt_prompt)
        data.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")

        async def _call() -> SttResult:
            timeout = aiohttp.ClientTimeout(total=self.cfg.openai_timeout_sec)
            connector = aiohttp.TCPConnector(ssl=self._ssl_context())
            async with aiohttp.ClientSession(
                timeout=timeout,
                headers=self._headers(),
                connector=connector,
                trust_env=True,
            ) as s:
                async with s.post(url, data=data) as resp:
                    txt = await resp.text()
                    if resp.status >= 300:
                        raise HttpStatusError(resp.status, txt, where="OpenAI STT")
                    js = json.loads(txt)
                    return SttResult(text=(js.get("text") or "").strip(), raw=js)

        async with self._limiter():
            return await with_retries(_call, cfg=self._retry_cfg(), what="openai_stt")


class OpenAILlm(OpenAIHttp, LlmClient):
    async def chat(self, messages: list[dict]) -> LlmResult:
        url = self.cfg.openai_base_url.rstrip("/") + "/chat/completions"
        body: dict = {
            "model": self.cfg.openai_llm_model,
            "messages": messages,
        }

        # Some models (e.g. certain reasoning/realtime variants) only support default temperature.
        if self.cfg.openai_llm_temperature is not None:
            body["temperature"] = float(self.cfg.openai_llm_temperature)

        async def _call() -> LlmResult:
            timeout = aiohttp.ClientTimeout(total=self.cfg.openai_timeout_sec)
            connector = aiohttp.TCPConnector(ssl=self._ssl_context())
            async with aiohttp.ClientSession(
                timeout=timeout,
                headers={**self._headers(), "Content-Type": "application/json"},
                connector=connector,
                trust_env=True,
            ) as s:
                async with s.post(url, json=body) as resp:
                    js = await resp.json(content_type=None)
                    if resp.status >= 300:
                        raise HttpStatusError(resp.status, json.dumps(js, ensure_ascii=False), where="OpenAI LLM")
                    try:
                        text = js["choices"][0]["message"]["content"]
                    except Exception:
                        text = ""
                    return LlmResult(text=(text or "").strip(), raw=js)

        async with self._limiter():
            return await with_retries(_call, cfg=self._retry_cfg(), what="openai_llm")


class OpenAITts(OpenAIHttp, TtsClient):
    async def synthesize(self, text: str, *, language: str = "ru") -> TtsResult:
        # Uses the modern audio/speech endpoint.
        url = self.cfg.openai_base_url.rstrip("/") + "/audio/speech"
        body = {
            "model": self.cfg.openai_tts_model,
            "voice": self.cfg.openai_tts_voice,
            "input": text,
            "response_format": "wav",
        }

        async def _call() -> TtsResult:
            timeout = aiohttp.ClientTimeout(total=self.cfg.openai_timeout_sec)
            connector = aiohttp.TCPConnector(ssl=self._ssl_context())
            async with aiohttp.ClientSession(
                timeout=timeout,
                headers={**self._headers(), "Content-Type": "application/json"},
                connector=connector,
                trust_env=True,
            ) as s:
                async with s.post(url, json=body) as resp:
                    if resp.status >= 300:
                        t = await resp.text()
                        raise HttpStatusError(resp.status, t, where="OpenAI TTS")
                    audio = await resp.read()
                    return TtsResult(wav_bytes=audio, raw=None)

        async with self._limiter():
            return await with_retries(_call, cfg=self._retry_cfg(), what="openai_tts")
