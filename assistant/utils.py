"""Small shared helpers.

Keep this module dependency-free.
"""

from __future__ import annotations

import re


def normalize_text(s: str) -> str:
    """Lowercase + strip punctuation, keep RU/EN letters and digits."""
    s2 = (s or "").lower()
    s2 = re.sub(r"[^а-яёa-z0-9\s]", " ", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2
