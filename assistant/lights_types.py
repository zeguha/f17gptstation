"""Types for smart lights control.

We keep the core domain model adapter-agnostic so we can swap Gauss integration
without changing NLU/skill routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ColorMode = Literal["white", "ct", "rgb", "scene"]
CtMode = Literal["warm", "neutral", "cool"]


@dataclass(slots=True)
class LampCapabilities:
    supports_brightness: bool = True
    supports_ct: bool = True
    supports_rgb: bool = False
    supports_scenes: bool = False
    supports_night_mode: bool = True

    # Typical ranges; adapter may override per device.
    ct_min_k: int = 2700
    ct_max_k: int = 6500


@dataclass(slots=True)
class LampState:
    is_on: bool = False
    brightness_percent: int | None = None  # 0..100
    ct_kelvin: int | None = None
    color_rgb: tuple[int, int, int] | None = None
    color_mode: ColorMode = "white"
    night_mode: bool = False
    scene: str | None = None

    updated_at: float | None = None  # monotonic/epoch; set by manager


@dataclass(slots=True)
class Lamp:
    id: str
    name: str
    room: str | None = None
    groups: list[str] = field(default_factory=list)
    is_online: bool = True

    capabilities: LampCapabilities = field(default_factory=LampCapabilities)
    state: LampState = field(default_factory=LampState)


@dataclass(slots=True)
class Group:
    id: str
    name: str
    room: str | None = None
    lamp_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Scene:
    id: str
    name: str
    # A scene can be adapter-native or assistant-level (expanded into commands).
    kind: Literal["adapter", "assistant"] = "assistant"
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LightCommandResult:
    ok: bool
    text: str
    changed_lamp_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

