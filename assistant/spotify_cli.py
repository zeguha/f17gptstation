"""CLI helper to connect Spotify via OAuth+PKCE.

Usage:
  python3 -m assistant.spotify_cli auth
"""

from __future__ import annotations

import asyncio

from .config import AppConfig
from .spotify_oauth import SpotifyOAuthConfig, run_local_pkce_flow
from .spotify_scopes import SPOTIFY_SCOPES
from .spotify_tokens import SpotifyTokenStore


async def _auth() -> int:
    cfg = AppConfig()
    sp = cfg.spotify
    store = SpotifyTokenStore(path=sp.token_path)
    oauth_cfg = SpotifyOAuthConfig(
        client_id=sp.client_id,
        redirect_uri=sp.redirect_uri,
        scopes=list(SPOTIFY_SCOPES),
        timeout_sec=180.0,
    )

    def _print_url(url: str) -> None:
        print("\n=== Spotify OAuth ===")
        print("1) Открой в браузере ссылку:")
        print(url)
        print("\n2) Разреши доступ. После этого страница скажет, что можно закрыть окно.")

    tokens = await run_local_pkce_flow(oauth_cfg, on_authorize_url=_print_url)
    store.save(tokens)
    print("\n✅ Токены сохранены в", sp.token_path)
    return 0


def main() -> None:
    import sys

    cmd = (sys.argv[1:] or [""])[0]
    if cmd != "auth":
        print("Usage: python3 -m assistant.spotify_cli auth")
        raise SystemExit(2)
    raise SystemExit(asyncio.run(_auth()))


if __name__ == "__main__":
    main()
