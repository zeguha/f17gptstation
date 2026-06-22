"""Device manager for smart lights.

Responsibilities:
- discovery + onboarding (naming/room/group)
- target resolution (lamp/group/room/all)
- caching device state
- best-effort sync on demand (pull) and placeholders for push events
"""

from __future__ import annotations

import time
from dataclasses import dataclass
import re

from .lights_gauss_adapter import GaussAdapter, GaussAdapterError, GaussDeviceOffline
from .lights_store import (
    group_from_json,
    group_to_json,
    lamp_from_json,
    lamp_to_json,
    load_lights_state,
    save_lights_state,
    scene_from_json,
    scene_to_json,
)
from .lights_types import Group, Lamp, LightCommandResult, Scene


@dataclass(slots=True)
class LightsManagerConfig:
    state_path: str
    state_cache_ttl_sec: float = 6.0


class LightsManager:
    def __init__(self, *, adapter: GaussAdapter, cfg: LightsManagerConfig):
        self._adapter = adapter
        self._cfg = cfg
        self._loaded = False
        self._lamps: dict[str, Lamp] = {}
        self._groups: dict[str, Group] = {}
        self._scenes: dict[str, Scene] = {}
        self._last_pull_ts: dict[str, float] = {}

    # -------- persistence --------

    def _load(self) -> None:
        if self._loaded:
            return
        raw = load_lights_state(self._cfg.state_path)
        self._lamps = {k: lamp_from_json(v) for k, v in (raw.get("lamps") or {}).items()}
        self._groups = {k: group_from_json(v) for k, v in (raw.get("groups") or {}).items()}
        self._scenes = {k: scene_from_json(v) for k, v in (raw.get("scenes") or {}).items()}
        self._loaded = True

    def _save(self) -> None:
        raw = {
            "version": 1,
            "lamps": {k: lamp_to_json(v) for k, v in self._lamps.items()},
            "groups": {k: group_to_json(v) for k, v in self._groups.items()},
            "scenes": {k: scene_to_json(v) for k, v in self._scenes.items()},
        }
        save_lights_state(self._cfg.state_path, raw)

    # -------- discovery --------

    async def discover_and_merge(self) -> LightCommandResult:
        self._load()
        lamps = await self._adapter.discover()
        added = 0
        updated = 0
        for l in lamps:
            if l.id in self._lamps:
                # merge some fields, preserve user naming/rooms/groups
                existing = self._lamps[l.id]
                existing.is_online = l.is_online
                existing.capabilities = l.capabilities
                updated += 1
            else:
                self._lamps[l.id] = l
                added += 1
        self._save()
        if added == 0 and updated == 0:
            return LightCommandResult(ok=True, text="Устройств не нашёл.")
        if added > 0:
            return LightCommandResult(ok=True, text=f"Нашёл {added} ламп(ы). Готово.")
        return LightCommandResult(ok=True, text=f"Обновил устройства: {updated}.")

    # -------- resolution helpers --------

    def known_rooms(self) -> list[str]:
        self._load()
        rooms = {l.room for l in self._lamps.values() if l.room}
        rooms |= {g.room for g in self._groups.values() if g.room}
        return sorted({r for r in rooms if r})

    def known_groups(self) -> list[str]:
        self._load()
        return sorted({g.name for g in self._groups.values() if g.name})

    def known_lamps(self) -> list[str]:
        self._load()
        return sorted({l.name for l in self._lamps.values() if l.name})

    def resolve_targets(
        self,
        *,
        target_scope: str,
        target_name: str | None,
        allow_all: bool,
        default_scope: str = "room",
        default_room: str | None = None,
    ) -> tuple[list[Lamp], list[str]]:
        """Return lamps + warnings."""
        self._load()
        warnings: list[str] = []
        scope = (target_scope or "").strip().lower() or "unspecified"
        name = (target_name or "").strip().lower() or None

        # 1) explicit all
        if scope == "all" and allow_all:
            return (list(self._lamps.values()), warnings)

        # 2) room
        if scope == "room" and name:
            lamps = self._resolve_room(name)
            if lamps:
                return (lamps, warnings)
            warnings.append(f"Комнату '{name}' не нашёл.")
            return ([], warnings)

        # 3) group
        if scope == "group" and name:
            gs = [g for g in self._groups.values() if g.name.strip().lower() == name]
            if not gs:
                warnings.append(f"Группу '{name}' не нашёл.")
                return ([], warnings)
            lamp_ids: list[str] = []
            for g in gs:
                lamp_ids.extend(list(g.lamp_ids or []))
            lamps = [self._lamps[x] for x in lamp_ids if x in self._lamps]
            return (lamps, warnings)

        # 4) by lamp name
        if name:
            exact = [l for l in self._lamps.values() if l.name.strip().lower() == name]
            if len(exact) == 1:
                return (exact, warnings)
            if len(exact) > 1:
                warnings.append(f"Нашёл несколько ламп с именем '{name}'.")
                return (exact, warnings)
            partial = [l for l in self._lamps.values() if name in l.name.strip().lower()]
            if len(partial) == 1:
                return (partial, warnings)
            if len(partial) > 1:
                warnings.append(f"Нашёл несколько ламп по '{name}'.")
                return (partial, warnings)

        # 5) unspecified target: all if allowed? No; use safe default.
        if scope in ("unspecified", ""):
            ds = (default_scope or "").strip().lower() or "room"
            dr = (default_room or "").strip().lower() or None
            if ds == "all" and allow_all:
                return (list(self._lamps.values()), warnings)
            if ds == "room" and dr:
                lamps = self._resolve_room(dr)
                if lamps:
                    return (lamps, warnings)
            warnings.append("Не понял, где менять свет: назови комнату или скажи 'везде'.")
            return ([], warnings)

        warnings.append("Не понял цель команды.")
        return ([], warnings)

    # -------- room matching helpers --------

    _RU_ENDINGS: tuple[str, ...] = (
        "ами",
        "ями",
        "ах",
        "ях",
        "ом",
        "ем",
        "ой",
        "ей",
        "ою",
        "ею",
        "ам",
        "ям",
        "у",
        "ю",
        "е",
        "и",
        "а",
        "я",
    )

    def _room_candidates_ru(self, room_text: str) -> list[str]:
        p = (room_text or "").strip().lower()
        p = re.sub(r"\s+", " ", p)
        p = p.strip(" ,.!?\t\n")
        if not p:
            return []
        cands: list[str] = [p]

        # Adjective-like forms: "детской" -> "детская", "большой" -> "большая".
        if p.endswith("ой") and len(p) > 4:
            v = p[:-2] + "ая"
            if v not in cands:
                cands.append(v)
        if p.endswith("ей") and len(p) > 4:
            v = p[:-2] + "яя"
            if v not in cands:
                cands.append(v)
        base = p
        for suf in self._RU_ENDINGS:
            if len(base) > len(suf) + 1 and base.endswith(suf):
                base2 = base[: -len(suf)]
                if base2 and base2 not in cands:
                    cands.append(base2)
                base = base2
                break
        if base and base[-1] not in "аеёиоуыэюяь":
            for suf in ("а", "я", "ь"):
                v = base + suf
                if v not in cands:
                    cands.append(v)
        out: list[str] = []
        for x in cands:
            if 2 <= len(x) <= 64 and x not in out:
                out.append(x)
        return out

    def _resolve_room(self, user_room: str) -> list[Lamp]:
        """Resolve room name with simple RU inflection heuristics."""
        u_cands = self._room_candidates_ru(user_room)
        if not u_cands:
            return []
        out: list[Lamp] = []
        for l in self._lamps.values():
            r = (l.room or "").strip().lower()
            if not r:
                continue
            r_cands = self._room_candidates_ru(r)
            if set(u_cands) & set(r_cands):
                out.append(l)
        return out

    # -------- state sync --------

    async def pull_state_if_stale(self, lamp: Lamp) -> None:
        """Pull state (best-effort) if cache stale."""
        now = time.time()
        last = self._last_pull_ts.get(lamp.id, 0.0)
        if (now - last) < float(self._cfg.state_cache_ttl_sec):
            return
        try:
            st = await self._adapter.get_state(lamp.id)
            lamp.state = st
            lamp.state.updated_at = now
            lamp.is_online = True
            self._last_pull_ts[lamp.id] = now
            self._save()
        except GaussDeviceOffline:
            lamp.is_online = False
            self._save()
        except Exception:
            # Keep old cached state.
            self._last_pull_ts[lamp.id] = now

    # -------- execution wrappers --------

    async def apply_to_lamps(self, lamps: list[Lamp], fn) -> tuple[list[str], list[str]]:
        changed: list[str] = []
        warnings: list[str] = []
        for l in lamps:
            # pull state before action so we can answer queries and validate caps
            await self.pull_state_if_stale(l)
            if not l.is_online:
                warnings.append(f"{l.name}: офлайн")
                continue
            try:
                await fn(l)
                changed.append(l.id)
            except GaussAdapterError as e:
                warnings.append(f"{l.name}: {str(e)}")
            except Exception:
                warnings.append(f"{l.name}: не получилось")
        return (changed, warnings)
