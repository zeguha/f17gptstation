"""Gauss lights adapter abstraction.

Gauss устройства могут управляться:
1) по LAN-протоколу (часто UDP/TCP + локальное шифрование/пейринг)
2) через облако производителя (HTTP API + токены)
3) через посредника (Home Assistant / MQTT)

Т.к. точный протокол может отличаться по линейкам, здесь задаём интерфейс.
Реальная интеграция заменит :class:`GaussMockAdapter`.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from typing import Protocol

from .lights_types import Lamp, LampCapabilities, LampState


class GaussAdapterError(RuntimeError):
    pass


class GaussAuthRequired(GaussAdapterError):
    pass


class GaussDeviceOffline(GaussAdapterError):
    pass


class GaussAdapter(Protocol):
    """Adapter interface (async) used by the device manager."""

    async def discover(self) -> list[Lamp]:
        """Discover lamps (best-effort)."""

    async def get_state(self, lamp_id: str) -> LampState:
        """Fetch current state from device/cloud."""

    async def set_power(self, lamp_id: str, *, on: bool) -> None:
        ...

    async def set_brightness(self, lamp_id: str, *, percent: int) -> None:
        ...

    async def set_ct(self, lamp_id: str, *, kelvin: int) -> None:
        ...

    async def set_rgb(self, lamp_id: str, *, rgb: tuple[int, int, int]) -> None:
        ...

    async def set_night_mode(self, lamp_id: str, *, on: bool) -> None:
        ...

    async def apply_scene(self, lamp_id: str, *, scene_name: str) -> None:
        ...

    async def capabilities(self, lamp_id: str) -> LampCapabilities:
        ...


@dataclass(slots=True)
class GaussMockAdapter:
    """Minimal mock adapter.

    - holds state in memory
    - simulates network latency
    - never does real I/O

    This is enough to wire end-to-end routing and state persistence.
    """

    latency_ms: int = 40

    # internal stores
    _lamps: dict[str, Lamp] | None = None

    def _ensure(self) -> None:
        if self._lamps is not None:
            return
        # A small default "home" to make the demo usable out-of-box.
        self._lamps = {
            "gauss-1": Lamp(id="gauss-1", name="люстра", room="гостиная", capabilities=LampCapabilities(supports_rgb=True, supports_ct=True)),
            "gauss-2": Lamp(id="gauss-2", name="бра", room="спальня", capabilities=LampCapabilities(supports_rgb=False, supports_ct=True)),
            "gauss-3": Lamp(id="gauss-3", name="ночник", room="детская", capabilities=LampCapabilities(supports_rgb=True, supports_ct=False, supports_night_mode=True)),
            "gauss-4": Lamp(id="gauss-4", name="кухня", room="кухня", capabilities=LampCapabilities(supports_rgb=False, supports_ct=True)),
        }
        # initialize reasonable defaults
        now = time.time()
        for l in self._lamps.values():
            l.state.is_on = False
            l.state.brightness_percent = 50
            l.state.ct_kelvin = 3500 if l.capabilities.supports_ct else None
            l.state.color_rgb = (255, 255, 255) if l.capabilities.supports_rgb else None
            l.state.updated_at = now

    async def _sleep(self) -> None:
        await asyncio.sleep(max(0.0, float(self.latency_ms) / 1000.0))

    async def discover(self) -> list[Lamp]:
        self._ensure()
        await self._sleep()
        assert self._lamps is not None
        # Return copies to prevent accidental mutation.
        out: list[Lamp] = []
        for l in self._lamps.values():
            out.append(
                Lamp(
                    id=l.id,
                    name=l.name,
                    room=l.room,
                    groups=list(l.groups or []),
                    is_online=bool(l.is_online),
                    capabilities=LampCapabilities(**asdict(l.capabilities)),
                    state=LampState(**asdict(l.state)),
                )
            )
        return out

    async def get_state(self, lamp_id: str) -> LampState:
        self._ensure()
        await self._sleep()
        assert self._lamps is not None
        if lamp_id not in self._lamps:
            raise GaussAdapterError("Unknown device")
        l = self._lamps[lamp_id]
        if not l.is_online:
            raise GaussDeviceOffline("Lamp offline")
        return LampState(**asdict(l.state))

    async def set_power(self, lamp_id: str, *, on: bool) -> None:
        self._ensure()
        await self._sleep()
        assert self._lamps is not None
        l = self._lamps[lamp_id]
        if not l.is_online:
            raise GaussDeviceOffline("Lamp offline")
        l.state.is_on = bool(on)
        l.state.updated_at = time.time()

    async def set_brightness(self, lamp_id: str, *, percent: int) -> None:
        self._ensure()
        await self._sleep()
        assert self._lamps is not None
        l = self._lamps[lamp_id]
        if not l.is_online:
            raise GaussDeviceOffline("Lamp offline")
        l.state.brightness_percent = int(max(0, min(100, percent)))
        l.state.is_on = l.state.brightness_percent > 0
        l.state.updated_at = time.time()

    async def set_ct(self, lamp_id: str, *, kelvin: int) -> None:
        self._ensure()
        await self._sleep()
        assert self._lamps is not None
        l = self._lamps[lamp_id]
        if not l.is_online:
            raise GaussDeviceOffline("Lamp offline")
        if not l.capabilities.supports_ct:
            raise GaussAdapterError("CT not supported")
        k = int(kelvin)
        k = max(l.capabilities.ct_min_k, min(l.capabilities.ct_max_k, k))
        l.state.ct_kelvin = k
        l.state.color_mode = "ct"
        l.state.updated_at = time.time()
        # changing CT implies on
        l.state.is_on = True

    async def set_rgb(self, lamp_id: str, *, rgb: tuple[int, int, int]) -> None:
        self._ensure()
        await self._sleep()
        assert self._lamps is not None
        l = self._lamps[lamp_id]
        if not l.is_online:
            raise GaussDeviceOffline("Lamp offline")
        if not l.capabilities.supports_rgb:
            raise GaussAdapterError("RGB not supported")
        r, g, b = rgb
        l.state.color_rgb = (int(max(0, min(255, r))), int(max(0, min(255, g))), int(max(0, min(255, b))))
        l.state.color_mode = "rgb"
        l.state.updated_at = time.time()
        l.state.is_on = True

    async def set_night_mode(self, lamp_id: str, *, on: bool) -> None:
        self._ensure()
        await self._sleep()
        assert self._lamps is not None
        l = self._lamps[lamp_id]
        if not l.is_online:
            raise GaussDeviceOffline("Lamp offline")
        if not l.capabilities.supports_night_mode:
            raise GaussAdapterError("Night mode not supported")
        l.state.night_mode = bool(on)
        if l.state.night_mode:
            l.state.is_on = True
            if l.capabilities.supports_brightness:
                l.state.brightness_percent = min(15, int(l.state.brightness_percent or 15))
        l.state.updated_at = time.time()

    async def apply_scene(self, lamp_id: str, *, scene_name: str) -> None:
        self._ensure()
        await self._sleep()
        assert self._lamps is not None
        l = self._lamps[lamp_id]
        if not l.is_online:
            raise GaussDeviceOffline("Lamp offline")
        # mock: scene is just a label + some canned settings
        s = (scene_name or "").strip().lower()
        l.state.scene = s
        l.state.color_mode = "scene"
        l.state.is_on = True
        if s in ("кино", "movie"):
            l.state.brightness_percent = 20
            if l.capabilities.supports_ct:
                l.state.ct_kelvin = 2700
        elif s in ("чтение", "reading"):
            l.state.brightness_percent = 75
            if l.capabilities.supports_ct:
                l.state.ct_kelvin = 4000
        elif s in ("сон", "sleep"):
            l.state.brightness_percent = 10
            if l.capabilities.supports_ct:
                l.state.ct_kelvin = 2700
        l.state.updated_at = time.time()

    async def capabilities(self, lamp_id: str) -> LampCapabilities:
        self._ensure()
        await self._sleep()
        assert self._lamps is not None
        l = self._lamps[lamp_id]
        return LampCapabilities(**asdict(l.capabilities))
