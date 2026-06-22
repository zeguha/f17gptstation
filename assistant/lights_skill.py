"""Lights skill orchestrator.

Entry point used by assistant main loops:
- detect intent outside (deterministic router)
- execute intent via LightsManager + adapter
- return RU text suitable for TTS
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from .intent_lights import LightsIntent
from .lights_device_manager import LightsManager
from .lights_gauss_adapter import GaussAdapterError, GaussDeviceOffline
from .lights_types import LightCommandResult


class LightsSkillError(RuntimeError):
    pass


def _fmt_pct(v: int | None) -> str:
    return "?" if v is None else f"{max(0, min(100, int(v)))}%"


def _ct_to_words(k: int) -> str:
    # heuristic
    if k <= 3200:
        return "тёплый"
    if k >= 5200:
        return "холодный"
    return "нейтральный"


def _kelvin_for_mode(mode: str) -> int:
    m = (mode or "").lower().strip()
    if m == "warm":
        return 2700
    if m == "cool":
        return 6000
    return 4000


@dataclass(slots=True)
class LightsSkillDeps:
    manager: LightsManager
    request_timeout_sec: float = 4.0
    default_scope: str = "room"  # room|all
    default_room: str | None = None


async def handle_lights_intent(user_intent: LightsIntent, *, deps: LightsSkillDeps) -> str:
    """Execute intent and return short RU response."""

    # Safety timeout wrapper.
    async def _do() -> str:
        mi = deps.manager

        if user_intent.action == "discover":
            r = await mi.discover_and_merge()
            return r.text

        lamps, warnings = mi.resolve_targets(
            target_scope=user_intent.target_scope,
            target_name=user_intent.target_name,
            allow_all=True,
            default_scope=deps.default_scope,
            default_room=deps.default_room,
        )
        if not lamps:
            if warnings:
                return warnings[0]
            return "Не вижу, где менять свет."

        changed: list[str] = []
        warn: list[str] = []

        async def apply(fn):
            nonlocal changed, warn
            ch, wn = await mi.apply_to_lamps(lamps, fn)
            changed, warn = ch, (warnings + wn)

        if user_intent.action == "power_on":
            await apply(lambda l: mi._adapter.set_power(l.id, on=True))
            txt = "Включил свет."
        elif user_intent.action == "power_off":
            await apply(lambda l: mi._adapter.set_power(l.id, on=False))
            txt = "Выключил свет."
        elif user_intent.action == "brightness_set":
            p = user_intent.brightness_percent
            if p is None:
                return "На сколько процентов яркость?"

            async def _fn(l):
                if not l.capabilities.supports_brightness:
                    raise GaussAdapterError("яркость не поддерживается")
                await mi._adapter.set_brightness(l.id, percent=int(p))

            await apply(_fn)
            txt = f"Яркость {_fmt_pct(int(p))}."
        elif user_intent.action == "brightness_delta":
            d = int(user_intent.brightness_delta or 0)
            if d == 0:
                return "На сколько процентов изменить яркость?"

            async def _fn(l):
                if not l.capabilities.supports_brightness:
                    raise GaussAdapterError("яркость не поддерживается")
                # pull current to compute new
                await mi.pull_state_if_stale(l)
                cur = int(l.state.brightness_percent or 0)
                await mi._adapter.set_brightness(l.id, percent=max(0, min(100, cur + d)))

            await apply(_fn)
            txt = "Сделал " + ("ярче." if d > 0 else "темнее.")
        elif user_intent.action in ("ct_set", "ct_delta"):
            # If user asked for a qualitative mode, map it.
            k = user_intent.ct_kelvin
            if k is None and user_intent.ct_mode:
                k = _kelvin_for_mode(user_intent.ct_mode)

            if user_intent.action == "ct_delta":
                delta = int(user_intent.ct_delta or 0)

                async def _fn(l):
                    if not l.capabilities.supports_ct:
                        raise GaussAdapterError("температура не поддерживается")
                    await mi.pull_state_if_stale(l)
                    cur = int(l.state.ct_kelvin or 4000)
                    await mi._adapter.set_ct(l.id, kelvin=cur + delta)

                await apply(_fn)
                txt = "Сделал свет " + ("теплее." if delta < 0 else "холоднее.")
            else:
                if k is None:
                    return "Какую цветовую температуру поставить? Например: 3000K."

                async def _fn(l):
                    if not l.capabilities.supports_ct:
                        raise GaussAdapterError("температура не поддерживается")
                    await mi._adapter.set_ct(l.id, kelvin=int(k))

                await apply(_fn)
                txt = f"Температура {int(k)}K ({_ct_to_words(int(k))})."
        elif user_intent.action == "color_set":
            rgb = user_intent.color_rgb
            if rgb is None:
                return "Какой цвет поставить?"

            async def _fn(l):
                if not l.capabilities.supports_rgb:
                    raise GaussAdapterError("цвет не поддерживается")
                await mi._adapter.set_rgb(l.id, rgb=rgb)

            await apply(_fn)
            txt = f"Поставил цвет{(' ' + user_intent.color_name) if user_intent.color_name else ''}."
        elif user_intent.action in ("night_mode_on", "night_mode_off"):
            on = user_intent.action == "night_mode_on"

            async def _fn(l):
                await mi._adapter.set_night_mode(l.id, on=on)

            await apply(_fn)
            txt = "Ночной режим включён." if on else "Ночной режим выключен."
        elif user_intent.action == "scene_apply":
            sn = (user_intent.scene_name or "").strip()
            if not sn:
                return "Какой сценарий? Например: кино или чтение."

            async def _fn(l):
                await mi._adapter.apply_scene(l.id, scene_name=sn)

            await apply(_fn)
            txt = f"Сценарий: {sn}."
        elif user_intent.action == "state_query":
            # show one or aggregated state
            # Pull fresh state for each lamp (ttl makes it cheap)
            for l in lamps:
                await deps.manager.pull_state_if_stale(l)
            if len(lamps) == 1:
                l = lamps[0]
                if not l.is_online:
                    return f"{l.name}: офлайн."
                b = l.state.brightness_percent
                k = l.state.ct_kelvin
                c = l.state.color_rgb
                parts: list[str] = [f"{l.name}: {'включена' if l.state.is_on else 'выключена'}"]
                if b is not None:
                    parts.append(f"яркость {_fmt_pct(b)}")
                if k is not None:
                    parts.append(f"{k}K")
                if c is not None and l.capabilities.supports_rgb:
                    parts.append(f"rgb {c[0]},{c[1]},{c[2]}")
                if l.state.night_mode:
                    parts.append("ночной режим")
                return ", ".join(parts) + "."
            on_cnt = sum(1 for l in lamps if l.is_online and l.state.is_on)
            off_cnt = sum(1 for l in lamps if l.is_online and not l.state.is_on)
            offl = sum(1 for l in lamps if not l.is_online)
            txt = f"Сейчас: включено {on_cnt}, выключено {off_cnt}"
            if offl:
                txt += f", офлайн {offl}"
            return txt + "."
        else:
            return "Не понял команду для света."

        # add warnings tail
        if warn:
            # keep it short for TTS
            tail = "; ".join(warn[:2])
            return f"{txt} ({tail})"
        if not changed:
            return "Не получилось выполнить команду."
        return txt

    try:
        return await asyncio.wait_for(_do(), timeout=float(deps.request_timeout_sec))
    except asyncio.TimeoutError:
        raise LightsSkillError("Не успел связаться с лампами (таймаут).")
    except GaussAdapterError as e:
        # Adapter-level diagnostics should be user-facing (short and actionable).
        raise LightsSkillError(str(e))
    except GaussDeviceOffline:
        raise LightsSkillError("Часть ламп офлайн.")
    except Exception:
        raise LightsSkillError("Не смог выполнить команду света.")
