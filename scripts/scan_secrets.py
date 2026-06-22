"""Simple local secret scanner for pre-publication checks."""

from __future__ import annotations

import re
import sys
from pathlib import Path


PATTERNS = {
    "openai_api_key": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"),
    "groq_api_key": re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    "generic_secret_assignment": re.compile(
        r"(?i)(api[_-]?key|secret|password|passwd|client[_-]?secret)\s*[:=]\s*['\"]([^'\"]{20,})['\"]"
    ),
}

EXCLUDED_DIRS = {
    ".git",
    ".secrets",
    ".state",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "assistant/models",
    "models",
    "whisper.cpp",
    "whisper.cpp/build",
}

EXCLUDED_SUFFIXES = {".pyc", ".wav", ".mp3", ".flac", ".ogg", ".bin", ".mdl", ".fst", ".carpa", ".png", ".jpg", ".jpeg"}
PLACEHOLDER_WORDS = {"replace-with", "your-", "example", "placeholder"}


def is_excluded(path: Path) -> bool:
    text = path.as_posix()
    return path.suffix in EXCLUDED_SUFFIXES or any(text == d or text.startswith(d + "/") for d in EXCLUDED_DIRS)


def is_placeholder(value: str) -> bool:
    low = value.lower()
    return any(word in low for word in PLACEHOLDER_WORDS)


def main() -> int:
    findings: list[str] = []
    for path in Path(".").rglob("*"):
        if not path.is_file() or is_excluded(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                raw = match.group(0)
                if is_placeholder(raw):
                    continue
                if "os.environ" in raw or "${{ secrets." in raw:
                    continue
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{path}:{line}: {name}: {raw[:120]}")

    if findings:
        print("Potential secrets found:", file=sys.stderr)
        print("\n".join(findings), file=sys.stderr)
        return 1
    print("No obvious secrets found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
