"""Lights intent detection (RU, rule-based).

Design goals:
- deterministic and fast
- tolerant to casual phrasing
- focuses on common home lighting actions

NLU model:
- intent/action: power, brightness, ct, color, night_mode, scene, query_state, discover
- slots: target_scope (room/group/all/lamp), target_name, params
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


LightAction = Literal[
    "discover",
    "power_on",
    "power_off",
    "brightness_set",
    "brightness_delta",
    "ct_set",
    "ct_delta",
    "color_set",
    "night_mode_on",
    "night_mode_off",
    "scene_apply",
    "state_query",
]


TargetScope = Literal["unspecified", "lamp", "room", "group", "all"]


@dataclass(frozen=True, slots=True)
class LightsIntent:
    action: LightAction
    confidence: float = 0.75

    # target
    target_scope: TargetScope = "unspecified"
    target_name: str | None = None

    # parameters
    brightness_percent: int | None = None
    brightness_delta: int | None = None

    ct_kelvin: int | None = None
    ct_delta: int | None = None
    ct_mode: Literal["warm", "neutral", "cool"] | None = None

    color_rgb: tuple[int, int, int] | None = None
    color_name: str | None = None

    scene_name: str | None = None


# --- keyword primitives ---
_RE_LIGHT_WORD = re.compile(r"\b(свет|ламп(а|ы)?|люстр(а|ы)?|бра|ночник)\b")

_RE_MUSIC_TALK = re.compile(r"\b(громкост\w*|звук\w*|громче|тише|секунд\w*|минут\w*|трек\w*|перемотай\w*|промотай\w*)\b")

_RE_ON = re.compile(r"\b(включи|вруби|зажги|сделай\s+свет|свет\s+вкл|вкл)\b")
_RE_OFF = re.compile(r"\b(выключи|выруби|погаси|свет\s+выкл|выкл)\b")

_RE_DISCOVER = re.compile(r"\b(найди|поиск|обнаруж(ь|ить)|сканируй)\s+(ламп|лампы|устройств(а)?)(\s+gauss)?\b")

_RE_QUERY = re.compile(r"\b(как\s+сейчас|какой\s+сейчас|какая\s+настройка|статус|состояние)\b")

_RE_ALL = re.compile(r"\b(везде|по\s+всему\s+дому|во\s+всех\s+комнатах)\b")

_RE_IN_ROOM = re.compile(r"\b(?:в|на)\s+([а-яё\-\s]{2,40})\b")

# Words that must not be part of the extracted room name.
_ROOM_STOPWORDS: tuple[str, ...] = (
    "включи",
    "включить",
    "выключи",
    "выключить",
    "поставь",
    "сделай",
    "на",
    "до",
    "яркость",
    "свет",
    "лампа",
    "лампы",
    "люстра",
    "бра",
    "ночник",
    "режим",
    "сценарий",
)

_RE_BRIGHTNESS_SET = re.compile(r"\b(?:яркост[ьи]|сделай\s+яркост[ьи]|приглуши\s+до|на)\s*(\d{1,3})\s*%?\b")
_RE_BRIGHTNESS_PLAIN = re.compile(r"\b(\d{1,3})\s*%\b")

# Relative brightness. Verbs only count next to "яркость"/"свет" so that e.g.
# "убавь громкость" (music) never lands here.
_DIM_VERBS = r"(?:убавь|уменьши|понизь|снизь|сбавь|притуши|приглуши)"
_BRIGHT_VERBS = r"(?:прибавь|увеличь|повысь|добавь|подними|усиль)"
_RE_DIM = re.compile(
    r"\b(?:" + _DIM_VERBS + r"\s+(?:немного\s+|чуть\s+)?(?:яркост\w*|свет\w*)"
    r"|приглуши|притуши|потуши\s+чуть|темнее|потемнее|тусклее|потусклее"
    r"|яркост\w*\s+(?:меньше|ниже|поменьше))\b"
)
_RE_BRIGHTER = re.compile(
    r"\b(?:" + _BRIGHT_VERBS + r"\s+(?:немного\s+|чуть\s+)?(?:яркост\w*|свет\w*)"
    r"|ярче|поярче|светлее|посветлее|сильнее"
    r"|яркост\w*\s+(?:больше|выше|побольше))\b"
)
_RE_BRIGHT_MAX = re.compile(
    r"\b(?:максимальн\w*\s+яркост\w*"
    r"|яркост\w*\s+(?:на\s+)?(?:максимум|максимальн\w*|полную)"
    r"|на\s+полную|на\s+максимум)\b"
)
_RE_BRIGHT_MIN = re.compile(
    r"\b(?:минимальн\w*\s+яркост\w*|яркост\w*\s+(?:на\s+)?(?:минимум|минимальн\w*)|на\s+минимум)\b"
)
_RE_DELTA_NA = re.compile(r"\bна\s+([+-]?\d{1,3})\b")
_RE_DELTA_PCT = re.compile(r"([+-]?\d{1,3})\s*%")


def _extract_delta(t: str) -> int | None:
    m = _RE_DELTA_NA.search(t) or _RE_DELTA_PCT.search(t)
    return abs(int(m.group(1))) if m else None

_RE_CT_K = re.compile(r"\b(\d{4,5})\s*k\b")
_RE_WARMER = re.compile(r"\b(теплее|потеплее|желтее)\b")
_RE_COOLER = re.compile(r"\b(холоднее|похолоднее|белее|синее)\b")
_RE_NEUTRAL = re.compile(r"\b(нейтрал(ьн(ый|ее)|ь))\b")

_RE_COLOR_RGB = re.compile(r"\b(?:rgb\s*)?(\d{1,3})\s*[,: ]\s*(\d{1,3})\s*[,: ]\s*(\d{1,3})\b")

_COLOR_WORDS: dict[str, tuple[int, int, int]] = {
    "красный": (255, 0, 0),
    "алый": (255, 40, 40),
    "оранжевый": (255, 140, 0),
    "желтый": (255, 210, 0),
    "жёлтый": (255, 210, 0),
    "зеленый": (0, 200, 0),
    "зелёный": (0, 200, 0),
    "синий": (30, 90, 255),
    "голубой": (80, 160, 255),
    "фиолетовый": (150, 70, 255),
    "розовый": (255, 90, 180),
    "белый": (255, 255, 255),
    "тёплый": (255, 160, 80),
    "теплый": (255, 160, 80),
}

_RE_NIGHT_ON = re.compile(r"\b(ночн(ой|ое)\s+режим|ночник)\s*(включи|вкл|on)?\b")
_RE_NIGHT_OFF = re.compile(r"\b(ночн(ой|ое)\s+режим)\s*(выключи|выкл|off)\b")

_RE_SCENE = re.compile(r"\b(сценарий|сцена|режим)\s+([а-яёa-z0-9\-\s]{2,40})\b")
_RE_SCENE_SHORT = re.compile(r"\b(кино|чтение|сон|relax|романтик(а)?)\b")


def _clamp_pct(x: int) -> int:
    return max(0, min(100, int(x)))


def _extract_target(normalized_text: str) -> tuple[TargetScope, str | None]:
    t = normalized_text
    if _RE_ALL.search(t):
        return ("all", None)
    # best-effort: "в <room>". We'll treat it as room by default.
    m = _RE_IN_ROOM.search(t)
    if m:
        name = m.group(1).strip(" ,.!?\t\n")
        # Cut on stopwords to avoid capturing verbs: "в детской выключи" -> "детской".
        words = [w for w in name.split() if w]
        cut: list[str] = []
        for w in words:
            if w in _ROOM_STOPWORDS:
                break
            cut.append(w)
        name = " ".join(cut).strip()
        # Avoid matching "вкл" etc.
        if name and len(name) >= 2 and name not in ("кл", "выкл"):
            return ("room", name)
    return ("unspecified", None)


def detect_lights_intent(normalized_text: str) -> LightsIntent | None:
    t = (normalized_text or "").strip()
    if not t:
        return None

    # Music/volume/seek talk with a bare number ("громкость на 50", "перемотай на
    # 30 секунд") must not be mistaken for brightness — lights are routed before
    # Spotify, and a bare "на N" is otherwise enough to look like a brightness set.
    if _RE_MUSIC_TALK.search(t) and not (_RE_LIGHT_WORD.search(t) or re.search(r"\bяркост\w*", t)):
        return None

    target_scope, target_name = _extract_target(t)
    has_light_word = bool(_RE_LIGHT_WORD.search(t))
    has_all = bool(_RE_ALL.search(t))
    has_roomish_target = target_scope in ("room", "group", "lamp") and bool(target_name)

    # We route only if user likely talks about lights.
    # Important: users often omit the word "свет" when context is obvious:
    # "в детской выключи", "приглуши до 30", "поставь 3000k", "поставь синий".
    looks_like_lights = False
    if has_light_word or has_all or has_roomish_target:
        looks_like_lights = True

    # IMPORTANT: "включи/выключи" alone is ambiguous (music vs lights).
    # Treat it as lights only when there is extra evidence (light word / room / all).
    if (_RE_ON.search(t) or _RE_OFF.search(t)) and (has_light_word or has_all or has_roomish_target):
        looks_like_lights = True
    if _RE_DISCOVER.search(t):
        looks_like_lights = True
    if _RE_QUERY.search(t) and (has_light_word or has_all or has_roomish_target):
        looks_like_lights = True
    if (_RE_BRIGHTNESS_SET.search(t) or _RE_DIM.search(t) or _RE_BRIGHTER.search(t)
            or _RE_BRIGHT_MAX.search(t) or _RE_BRIGHT_MIN.search(t)):
        looks_like_lights = True
    if _RE_CT_K.search(t) or _RE_WARMER.search(t) or _RE_COOLER.search(t) or _RE_NEUTRAL.search(t):
        looks_like_lights = True
    if _RE_NIGHT_ON.search(t) or _RE_NIGHT_OFF.search(t):
        looks_like_lights = True
    if _RE_SCENE.search(t) or _RE_SCENE_SHORT.search(t):
        looks_like_lights = True
    if _RE_COLOR_RGB.search(t):
        looks_like_lights = True
    if not looks_like_lights:
        # color words may be present without "цвет" token; still treat if "поставь/сделай" appears.
        if "поставь" in t or "сделай" in t:
            for w in _COLOR_WORDS.keys():
                if re.search(rf"\b{re.escape(w)}\b", t):
                    looks_like_lights = True
                    break
    if not looks_like_lights:
        return None


    if _RE_DISCOVER.search(t):
        return LightsIntent(action="discover", target_scope="all", confidence=0.85)

    if _RE_QUERY.search(t) and has_light_word:
        return LightsIntent(action="state_query", target_scope=target_scope, target_name=target_name, confidence=0.75)

    # Night mode
    if _RE_NIGHT_OFF.search(t):
        return LightsIntent(action="night_mode_off", target_scope=target_scope, target_name=target_name, confidence=0.8)
    if _RE_NIGHT_ON.search(t) and ("выкл" not in t):
        return LightsIntent(action="night_mode_on", target_scope=target_scope, target_name=target_name, confidence=0.75)

    # Scenes
    m = _RE_SCENE.search(t)
    if m:
        name = (m.group(2) or "").strip()
        return LightsIntent(action="scene_apply", target_scope=target_scope, target_name=target_name, scene_name=name, confidence=0.75)
    m = _RE_SCENE_SHORT.search(t)
    if m and ("свет" in t or "ламп" in t or "режим" in t or "сцен" in t):
        return LightsIntent(action="scene_apply", target_scope=target_scope, target_name=target_name, scene_name=m.group(1), confidence=0.65)

    # Color (RGB or words)
    m = _RE_COLOR_RGB.search(t)
    if m and ("цвет" in t or "поставь" in t or "сделай" in t):
        rgb = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return LightsIntent(action="color_set", target_scope=target_scope, target_name=target_name, color_rgb=rgb, confidence=0.7)

    for w, rgb in _COLOR_WORDS.items():
        if re.search(rf"\b{re.escape(w)}\b", t) and ("цвет" in t or "поставь" in t or "сделай" in t):
            return LightsIntent(action="color_set", target_scope=target_scope, target_name=target_name, color_rgb=rgb, color_name=w, confidence=0.72)

    # CT Kelvin / modes
    m = _RE_CT_K.search(t)
    if m:
        k = int(m.group(1))
        return LightsIntent(action="ct_set", target_scope=target_scope, target_name=target_name, ct_kelvin=k, confidence=0.75)

    if _RE_WARMER.search(t) and ("свет" in t or "тепле" in t):
        return LightsIntent(action="ct_delta", target_scope=target_scope, target_name=target_name, ct_mode="warm", ct_delta=-350, confidence=0.7)
    if _RE_COOLER.search(t) and ("свет" in t or "холод" in t or "бел" in t):
        return LightsIntent(action="ct_delta", target_scope=target_scope, target_name=target_name, ct_mode="cool", ct_delta=+350, confidence=0.7)
    if _RE_NEUTRAL.search(t) and ("свет" in t or "температур" in t):
        return LightsIntent(action="ct_set", target_scope=target_scope, target_name=target_name, ct_mode="neutral", ct_kelvin=4000, confidence=0.65)

    # Brightness max/min ("на максимум", "минимальная яркость", "на полную").
    if _RE_BRIGHT_MAX.search(t):
        return LightsIntent(action="brightness_set", target_scope=target_scope, target_name=target_name, brightness_percent=100, confidence=0.75)
    if _RE_BRIGHT_MIN.search(t):
        return LightsIntent(action="brightness_set", target_scope=target_scope, target_name=target_name, brightness_percent=10, confidence=0.75)

    # "убавь яркость на 20" is relative, "приглуши до 30" / "яркость на 50" absolute.
    # Relative wording wins for "на N" unless the user said "до N".
    is_relative = bool(_RE_DIM.search(t) or _RE_BRIGHTER.search(t))
    says_absolute_target = bool(re.search(r"\bдо\s+\d", t))

    # Brightness set
    m = _RE_BRIGHTNESS_SET.search(t)
    if m and (says_absolute_target or not is_relative):
        p = _clamp_pct(int(m.group(1)))
        return LightsIntent(action="brightness_set", target_scope=target_scope, target_name=target_name, brightness_percent=p, confidence=0.8)
    m = _RE_BRIGHTNESS_PLAIN.search(t)
    if m and (says_absolute_target or not is_relative) and ("ярк" in t or "приглуш" in t or "на" in t):
        p = _clamp_pct(int(m.group(1)))
        return LightsIntent(action="brightness_set", target_scope=target_scope, target_name=target_name, brightness_percent=p, confidence=0.65)

    # Brightness relative
    if _RE_DIM.search(t):
        d = _extract_delta(t)
        return LightsIntent(action="brightness_delta", target_scope=target_scope, target_name=target_name, brightness_delta=-_clamp_pct(d if d else 20), confidence=0.65 if d else 0.6)
    if _RE_BRIGHTER.search(t):
        d = _extract_delta(t)
        return LightsIntent(action="brightness_delta", target_scope=target_scope, target_name=target_name, brightness_delta=_clamp_pct(d if d else 20), confidence=0.65 if d else 0.6)

    # Power
    if _RE_ON.search(t):
        # If user said "включи на 30" -> treat as brightness set.
        m2 = _RE_BRIGHTNESS_SET.search(t) or _RE_BRIGHTNESS_PLAIN.search(t)
        if m2:
            p = _clamp_pct(int(m2.group(1)))
            return LightsIntent(action="brightness_set", target_scope=target_scope, target_name=target_name, brightness_percent=p, confidence=0.75)
        return LightsIntent(action="power_on", target_scope=target_scope, target_name=target_name, confidence=0.75)
    if _RE_OFF.search(t):
        return LightsIntent(action="power_off", target_scope=target_scope, target_name=target_name, confidence=0.75)

    return None
