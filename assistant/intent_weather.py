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

    # Time targeting (all None/0 => "right now", matching legacy behavior).
    # day_offset: 0=today, 1=tomorrow, 2=day after tomorrow.
    day_offset: int | None = None
    # 0=Monday..6=Sunday (Python's date.weekday()); used when user names a weekday
    # instead of a relative day ("в пятницу").
    weekday: int | None = None
    # "morning" | "day" | "evening" | "night"
    part_of_day: str | None = None


_WEATHER_RE = re.compile(r"(погод|температур|градус|на\s+улице|за\s+окном|ветер|осадк|дожд|снег)")
_UMBRELLA_RE = re.compile(r"(нужен\s+ли\s+зонт|зонт|дожд|ливень|морос)")
_CLOTHES_RE = re.compile(r"(как\s+одет|как\s+одеться|что\s+надет|в\s+ч[её]м\s+идти|куртк|шапк|перчатк)")

_WEEKDAY_RE = re.compile(
    r"\b(понедельник|вторник|сред[ауы]|четверг|пятниц[ауы]|суббот[ауы]|воскресень[ея])\b"
)
_WEEKDAY_TO_NUM: dict[str, int] = {
    "понедельник": 0,
    "вторник": 1,
    "сред": 2,
    "четверг": 3,
    "пятниц": 4,
    "суббот": 5,
    "воскресень": 6,
}

_TOMORROW_RE = re.compile(r"\bзавтра\b")
_DAY_AFTER_TOMORROW_RE = re.compile(r"\bпослезавтра\b")
_TODAY_RE = re.compile(r"\bсегодня\b")

_MORNING_RE = re.compile(r"\bутр(ом|о|а)\b")
_DAYTIME_RE = re.compile(r"\bдн[её]м\b")
_EVENING_RE = re.compile(r"\bвечер(ом|а)?\b")
_NIGHT_RE = re.compile(r"\bноч(ью|и|ь)\b")

# Very simple place extraction; we validate via geocoding later.
_PLACE_RE = re.compile(r"\bв\s+([а-яё\-\s]{2,40})\b")

# Words that must not be swallowed into the extracted place name, e.g.
# "погода в москве в пятницу вечером" -> "москве" (not "москве в пятницу вечером").
# Uses fullmatch against the same patterns as time-word detection so we don't
# accidentally cut real place names that merely start with a similar prefix
# (e.g. "Днепр" must not be treated as "днём").
_PLACE_STOPWORD_RE = re.compile(
    r"^(?:в|на"
    r"|понедельник|вторник|сред[ауы]?|четверг|пятниц[ауы]?|суббот[ауы]?|воскресень[ея]?"
    r"|сегодня|завтра|послезавтра"
    r"|утр(?:ом|о|а)?|дн[её]м|вечер(?:ом|а)?|ноч(?:ью|и|ь)?)$"
)


def _is_place_stopword(word: str) -> bool:
    return bool(_PLACE_STOPWORD_RE.match(word))


def _clean_place(raw: str) -> str | None:
    words = [w for w in raw.strip(" ,.!?\t\n").split() if w]
    kept: list[str] = []
    for w in words:
        if _is_place_stopword(w):
            break
        kept.append(w)
    place = " ".join(kept).strip()
    return place if len(place) >= 2 else None


def _weekday_num(text: str) -> int | None:
    m = _WEEKDAY_RE.search(text)
    if not m:
        return None
    word = m.group(1)
    for stem, num in _WEEKDAY_TO_NUM.items():
        if word.startswith(stem):
            return num
    return None


def _part_of_day(text: str) -> str | None:
    if _MORNING_RE.search(text):
        return "morning"
    if _DAYTIME_RE.search(text):
        return "day"
    if _EVENING_RE.search(text):
        return "evening"
    if _NIGHT_RE.search(text):
        return "night"
    return None


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
        place = _clean_place(m.group(1))
        if place:
            place = place[:64]

    # Time targeting.
    day_offset: int | None = None
    weekday: int | None = None

    wd = _weekday_num(t)
    if wd is not None:
        weekday = wd
    elif _DAY_AFTER_TOMORROW_RE.search(t):
        day_offset = 2
    elif _TOMORROW_RE.search(t):
        day_offset = 1
    elif _TODAY_RE.search(t):
        day_offset = 0

    part_of_day = _part_of_day(t)

    return WeatherIntent(
        intent=intent,
        place_text=place,
        confidence=0.75,
        day_offset=day_offset,
        weekday=weekday,
        part_of_day=part_of_day,
    )
