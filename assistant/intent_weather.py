"""Weather intent detection (lightweight, rule-based).

We keep it deterministic and fast:
- works offline
- avoids LLM for a common skill
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WeatherIntent:
    intent: str  # weather_now | weather_umbrella | weather_clothes
    place_text: str | None
    confidence: float


_WEATHER_RE = re.compile(r"(погод|температур|градус|на\s+улице|за\s+окном|ветер|осадк|дожд|снег)")
_UMBRELLA_RE = re.compile(r"(нужен\s+ли\s+зонт|зонт|дожд|ливень|морос)")
_CLOTHES_RE = re.compile(r"(как\s+одет|как\s+одеться|что\s+надет|в\s+ч[её]м\s+идти|куртк|шапк|перчатк)")

# Very simple place extraction; we validate via geocoding later.
_PLACE_RE = re.compile(r"\bв\s+([а-яё\-\s]{2,40})\b")


def detect_weather_intent(normalized_text: str) -> WeatherIntent | None:
    t = (normalized_text or "").strip()
    if not t:
        return None

    if not (_WEATHER_RE.search(t) or _UMBRELLA_RE.search(t) or _CLOTHES_RE.search(t)):
        return None

    intent = "weather_now"
    if _UMBRELLA_RE.search(t):
        intent = "weather_umbrella"
    elif _CLOTHES_RE.search(t):
        intent = "weather_clothes"

    place = None
    m = _PLACE_RE.search(t)
    if m:
        # Avoid capturing too much trailing text.
        place = m.group(1).strip(" ,.!?\t\n")
        if place and len(place) > 2:
            place = place[:64]
        else:
            place = None

    return WeatherIntent(intent=intent, place_text=place, confidence=0.75)

