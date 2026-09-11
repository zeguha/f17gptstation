"""Post-processing for ASR text.

Main objectives:
- normalize
- ensure wake phrase never leaks into final command
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher


_re_nonword = re.compile(r"[^а-яёa-z0-9\s]", re.IGNORECASE)
_re_ws = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = _re_nonword.sub(" ", s)
    s = _re_ws.sub(" ", s).strip()
    return s


# Web-search-grounded models (e.g. gpt-5-search-api) routinely cite sources as
# "([site.com](https://...))" — readable as text, unusable read aloud by TTS.
_re_wrapped_citation = re.compile(r"\(\[([^\]]+)\]\((?:https?://|www\.)[^()]*\)\)")
_re_md_link = re.compile(r"\[([^\]]+)\]\((?:https?://|www\.)[^()]*\)")
_re_bare_url = re.compile(r"https?://\S+")
_re_md_header = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_re_md_hr = re.compile(r"^\s*[-*_]{3,}\s*$", re.MULTILINE)
_re_md_emphasis = re.compile(r"(\*\*\*|\*\*|\*|__)")


def clean_for_speech(text: str) -> str:
    """Strip markdown links/citations/headers from LLM text before TTS.

    Meant for text that is about to be spoken, not for text kept in dialog
    history — the citations are still useful context for follow-up turns.
    """

    t = text or ""
    t = _re_wrapped_citation.sub("", t)
    t = _re_md_link.sub(r"\1", t)
    t = _re_bare_url.sub("", t)
    t = _re_md_header.sub("", t)
    t = _re_md_hr.sub("", t)
    t = _re_md_emphasis.sub("", t)
    t = _re_ws.sub(" ", t).strip()
    return t

def strip_wake_phrase(
    text: str,
    wake_phrase: str,
    *,
    fuzzy_threshold: float = 0.80,
    max_prefix_words: int = 6,
) -> str:
    """Remove wake phrase if it appears at the beginning of the recognized text.

    This is a *safety net*; primary protection should be pipeline separation.
    """

    t = normalize_text(text)
    w = normalize_text(wake_phrase)
    if not t or not w:
        return t

    if t == w:
        return ""

    if t.startswith(w + " "):
        return t[len(w) :].strip()

    # fuzzy *prefix* check (only from the beginning, not anywhere in the prefix)
    # We intentionally do NOT remove wake phrase in the middle of the command,
    # because that can delete valid user content.
    words = t.split()
    wake_words = w.split()

    best_k = None
    best_ratio = 0.0
    # Try several prefix lengths around wake length.
    for k in range(max(1, len(wake_words) - 1), min(len(words), len(wake_words) + 2) + 1):
        if k > max_prefix_words:
            break
        frag = " ".join(words[:k])
        r = SequenceMatcher(None, frag, w).ratio()
        if r > best_ratio:
            best_ratio = r
            best_k = k

    if best_k is not None and best_ratio >= fuzzy_threshold:
        return " ".join(words[best_k:]).strip()
    return t


def wake_match_ratio_prefix(
    text: str,
    wake_phrase: str,
    *,
    max_prefix_words: int = 6,
) -> float:
    """Similarity ratio of `wake_phrase` against the *start* of `text`.

    Used for wake detection when a Vosk model does not support runtime grammars.
    Returns a value in [0..1].
    """

    t = normalize_text(text)
    w = normalize_text(wake_phrase)
    if not t or not w:
        return 0.0

    words = t.split()
    wake_words = w.split()
    if not words or not wake_words:
        return 0.0

    best = 0.0
    for k in range(max(1, len(wake_words) - 1), min(len(words), len(wake_words) + 2) + 1):
        if k > max_prefix_words:
            break
        frag = " ".join(words[:k])
        r = SequenceMatcher(None, frag, w).ratio()
        if r > best:
            best = r
    return best
