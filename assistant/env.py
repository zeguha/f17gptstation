"""Minimal `.env` loader.

The project intentionally avoids depending on `python-dotenv` for runtime. This
loader supports the common `KEY=value` format and never overrides variables
already provided by the shell, systemd, Docker, or GitHub Actions.
"""

from __future__ import annotations

import os
from pathlib import Path


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(path: str | os.PathLike[str] | None = None) -> None:
    """Load environment variables from `.env` if it exists.

    Existing environment variables are preserved. Lines beginning with `#`,
    blank lines, and malformed lines without `=` are ignored.
    """

    env_path = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists() or not env_path.is_file():
        return

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _strip_quotes(value)


load_env_file()
