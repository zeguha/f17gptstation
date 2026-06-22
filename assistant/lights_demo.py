"""Minimal CLI demo for Gauss lights skill.

Usage:
  python3 -m assistant.lights_demo "везде включи на 50%"
"""

from __future__ import annotations

import argparse
import asyncio
import os

from .intent_lights import detect_lights_intent
from .lights_device_manager import LightsManager, LightsManagerConfig
from .lights_gauss_adapter import GaussMockAdapter
from .lights_skill import LightsSkillDeps, LightsSkillError, handle_lights_intent
from .utils import normalize_text


async def _run_once(text: str, *, state_path: str) -> None:
    adapter = GaussMockAdapter()
    mgr = LightsManager(adapter=adapter, cfg=LightsManagerConfig(state_path=state_path))
    # ensure initial discovery for demo
    await mgr.discover_and_merge()

    norm = normalize_text(text)
    i = detect_lights_intent(norm)
    print("IN :", text)
    print("NORM:", norm)
    print("INT:", i)
    if i is None:
        print("OUT:", "(не распознано)")
        return
    try:
        out = await handle_lights_intent(
            i,
            deps=LightsSkillDeps(
                manager=mgr,
                default_scope="all",
                default_room=None,
            ),
        )
        print("OUT:", out)
    except LightsSkillError as e:
        print("OUT:", str(e))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+", help="Команда на русском")
    ap.add_argument(
        "--state",
        default=os.path.join(os.path.dirname(__file__), "..", ".state", "lights.json"),
        help="Путь к JSON состоянию",
    )
    args = ap.parse_args()
    text = " ".join(args.text)
    asyncio.run(_run_once(text, state_path=args.state))


if __name__ == "__main__":
    main()
