"""Spotify OAuth scopes required for the supported voice commands."""

from __future__ import annotations


# Full scope set for the requested features.
SPOTIFY_SCOPES: tuple[str, ...] = (
    # Playback control + state
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",

    # Library (Liked Songs)
    "user-library-read",
    "user-library-modify",

    # Playlists editing
    "playlist-read-private",
    "playlist-modify-public",
    "playlist-modify-private",

    # Optional: to improve disambiguation by reading user playlists
    "playlist-read-collaborative",
)
