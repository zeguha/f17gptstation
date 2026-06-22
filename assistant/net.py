"""Small networking helpers (retry/backoff/timeouts).

Goal: keep cloud calls reliable on Raspberry Pi + домашняя сеть.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
import ssl
from typing import Awaitable, Callable, Iterable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryConfig:
    max_attempts: int = 3
    base_delay_sec: float = 0.4
    max_delay_sec: float = 6.0
    jitter: float = 0.25  # 0..1 fraction
    retryable_statuses: tuple[int, ...] = (408, 409, 425, 429, 500, 502, 503, 504)


class HttpStatusError(RuntimeError):
    """HTTP error with status and body preserved for logging/retry logic."""

    def __init__(self, status: int, body: str, *, where: str = ""):
        super().__init__(f"{where} HTTP {status}: {body}")
        self.status = int(status)
        self.body = body
        self.where = where


def _is_retryable_exc(exc: BaseException, *, cfg: RetryConfig) -> bool:
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, HttpStatusError):
        return exc.status in cfg.retryable_statuses
    return False


def _backoff(attempt: int, *, cfg: RetryConfig) -> float:
    # attempt: 1..N
    # exponential backoff with jitter
    d = min(cfg.max_delay_sec, cfg.base_delay_sec * (2 ** max(0, attempt - 1)))
    j = (random.random() * 2 - 1) * cfg.jitter * d
    return max(0.0, d + j)


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    cfg: RetryConfig,
    what: str,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
) -> T:
    last: BaseException | None = None
    for attempt in range(1, int(cfg.max_attempts) + 1):
        try:
            return await fn()
        except BaseException as e:  # noqa: BLE001 - we re-raise below
            last = e
            if attempt >= cfg.max_attempts or not _is_retryable_exc(e, cfg=cfg):
                raise
            delay = _backoff(attempt, cfg=cfg)
            if on_retry is not None:
                on_retry(attempt, delay, e)
            await asyncio.sleep(delay)
    assert last is not None
    raise last


class AsyncLimiter:
    """A tiny async semaphore wrapper to cap in-flight cloud calls."""

    def __init__(self, max_in_flight: int):
        self._sem = asyncio.Semaphore(max(1, int(max_in_flight)))

    async def __aenter__(self):
        await self._sem.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._sem.release()
        return False


_SSL_SENTINEL: object = object()
_SSL_CTX: ssl.SSLContext | None | object = _SSL_SENTINEL


def ssl_context() -> ssl.SSLContext | None:
    """Best-effort SSL context.

    On some macOS Python installs (notably python.org), the system CA bundle can be
    missing, causing `SSLCertVerificationError` for HTTPS.

    If `certifi` is available we use its CA bundle.
    """

    global _SSL_CTX
    if _SSL_CTX is not _SSL_SENTINEL:
        return _SSL_CTX  # type: ignore[return-value]
    try:
        import certifi  # type: ignore

        _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        # Fall back to a system/default context. On some systems this may still fail
        # if CA bundle is missing, in which case install certifi.
        _SSL_CTX = ssl.create_default_context()
    return _SSL_CTX  # type: ignore[return-value]
