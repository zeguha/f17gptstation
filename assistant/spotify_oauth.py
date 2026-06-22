"""Spotify OAuth Authorization Code + PKCE.

Flow:
- Generate auth URL with PKCE code challenge.
- Start a tiny local HTTP server for redirect_uri.
- Exchange code -> tokens.

Notes:
- No client_secret is required for PKCE (public client).
- User must be logged into Spotify in browser.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import secrets
import time
from dataclasses import dataclass
from typing import Callable

import aiohttp
from aiohttp import web

from .net import ssl_context
from .spotify_tokens import SpotifyTokens


class SpotifyOAuthError(RuntimeError):
    pass


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _pkce_verifier() -> str:
    # 43..128 chars, URL-safe
    return _b64url(secrets.token_bytes(64))


def _pkce_challenge(verifier: str) -> str:
    h = hashlib.sha256(verifier.encode("utf-8")).digest()
    return _b64url(h)


@dataclass(frozen=True, slots=True)
class SpotifyOAuthConfig:
    client_id: str
    redirect_uri: str
    scopes: list[str]
    timeout_sec: float = 120.0


def build_authorize_url(cfg: SpotifyOAuthConfig, *, state: str, code_challenge: str) -> str:
    # https://accounts.spotify.com/authorize
    scope = " ".join([s for s in cfg.scopes if s])
    import urllib.parse

    q = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": cfg.client_id,
            "redirect_uri": cfg.redirect_uri,
            "scope": scope,
            "state": state,
            "code_challenge_method": "S256",
            "code_challenge": code_challenge,
        }
    )
    return f"https://accounts.spotify.com/authorize?{q}"


async def exchange_code_for_tokens(
    *,
    client_id: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
    timeout_sec: float,
) -> SpotifyTokens:
    url = "https://accounts.spotify.com/api/token"
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": code_verifier,
    }
    timeout = aiohttp.ClientTimeout(total=float(timeout_sec))
    connector = aiohttp.TCPConnector(ssl=ssl_context())
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=True) as s:
        async with s.post(url, data=data) as r:
            js = await r.json(content_type=None)
            if r.status >= 300:
                raise SpotifyOAuthError(f"OAuth token exchange HTTP {r.status}: {js}")
            access_token = str(js.get("access_token") or "")
            refresh_token = str(js.get("refresh_token") or "")
            expires_in = float(js.get("expires_in") or 0.0)
            if not access_token or not refresh_token or expires_in <= 0:
                raise SpotifyOAuthError(f"Неожиданный ответ токенов: {js}")
            return SpotifyTokens(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_at=time.time() + expires_in,
            )


async def refresh_access_token(
    *,
    client_id: str,
    refresh_token: str,
    timeout_sec: float,
) -> SpotifyTokens:
    url = "https://accounts.spotify.com/api/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    timeout = aiohttp.ClientTimeout(total=float(timeout_sec))
    connector = aiohttp.TCPConnector(ssl=ssl_context())
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=True) as s:
        async with s.post(url, data=data) as r:
            js = await r.json(content_type=None)
            if r.status >= 300:
                raise SpotifyOAuthError(f"OAuth refresh HTTP {r.status}: {js}")
            access_token = str(js.get("access_token") or "")
            expires_in = float(js.get("expires_in") or 0.0)
            # refresh_token может не вернуться при refresh — используем старый.
            new_refresh = str(js.get("refresh_token") or "")
            if not access_token or expires_in <= 0:
                raise SpotifyOAuthError(f"Неожиданный ответ refresh: {js}")
            return SpotifyTokens(
                access_token=access_token,
                refresh_token=(new_refresh or refresh_token),
                expires_at=time.time() + expires_in,
            )


async def run_local_pkce_flow(
    cfg: SpotifyOAuthConfig,
    *,
    on_authorize_url: Callable[[str], None] | None = None,
) -> SpotifyTokens:
    """Run OAuth flow and return tokens.

    If `on_authorize_url` is provided, it is called after the local HTTP server is
    started, before waiting for the callback.
    """

    if not cfg.client_id:
        raise SpotifyOAuthError("SPOTIFY_CLIENT_ID не задан")

    verifier = _pkce_verifier()
    challenge = _pkce_challenge(verifier)
    state = _b64url(secrets.token_bytes(16))
    authorize_url = build_authorize_url(cfg, state=state, code_challenge=challenge)

    got: dict[str, str] = {}
    ev = asyncio.Event()

    async def _handler(request: web.Request) -> web.Response:
        nonlocal got
        params = request.query
        got = {k: v for k, v in params.items()}
        ev.set()
        # Simple HTML page.
        return web.Response(
            text=(
                "<html><body>Spotify авторизация завершена. "
                "Можно закрыть это окно и вернуться к ассистенту.</body></html>"
            ),
            content_type="text/html",
        )

    # Parse host/port/path from redirect_uri.
    import urllib.parse

    u = urllib.parse.urlparse(cfg.redirect_uri)
    host = u.hostname or "127.0.0.1"
    port = int(u.port or 80)
    path = u.path or "/callback"

    app = web.Application()
    app.router.add_get(path, _handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()

    if on_authorize_url is not None:
        on_authorize_url(authorize_url)

    try:
        # Wait for callback or timeout.
        try:
            await asyncio.wait_for(ev.wait(), timeout=float(cfg.timeout_sec))
        except asyncio.TimeoutError as e:
            raise SpotifyOAuthError("Не дождался OAuth callback. Открой ссылку авторизации ещё раз.") from e

        if got.get("state") != state:
            raise SpotifyOAuthError("OAuth state mismatch. Попробуй ещё раз.")

        if "error" in got:
            raise SpotifyOAuthError(f"Spotify OAuth error: {got.get('error')}")
        code = got.get("code") or ""
        if not code:
            raise SpotifyOAuthError("Не получил code из OAuth callback")

        tokens = await exchange_code_for_tokens(
            client_id=cfg.client_id,
            code=code,
            redirect_uri=cfg.redirect_uri,
            code_verifier=verifier,
            timeout_sec=20.0,
        )
        return tokens
    finally:
        with contextlib.suppress(Exception):
            await runner.cleanup()


# Local import to keep top-level imports minimal.
import contextlib
