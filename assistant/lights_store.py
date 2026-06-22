"""State storage for lights (simple JSON).

We keep this layer dumb and synchronous:
- no network
- no business logic
- atomic-ish writes
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

from .lights_types import Group, Lamp, LampCapabilities, LampState, Scene


def _now() -> float:
    return time.time()


def load_lights_state(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "version": 1,
            "updated_at": _now(),
            "lamps": {},
            "groups": {},
            "scenes": {},
        }


def save_lights_state(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    data2 = dict(data)
    data2["updated_at"] = _now()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data2, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def lamp_to_json(l: Lamp) -> dict:
    d = asdict(l)
    # tuples -> lists for json
    if d.get("state", {}).get("color_rgb") is not None:
        d["state"]["color_rgb"] = list(d["state"]["color_rgb"])
    return d


def lamp_from_json(d: dict) -> Lamp:
    caps = LampCapabilities(**(d.get("capabilities") or {}))
    st = d.get("state") or {}
    rgb = st.get("color_rgb")
    if isinstance(rgb, list) and len(rgb) == 3:
        st = dict(st)
        st["color_rgb"] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    state = LampState(**st)
    return Lamp(
        id=str(d.get("id") or ""),
        name=str(d.get("name") or ""),
        room=(d.get("room") or None),
        groups=list(d.get("groups") or []),
        is_online=bool(d.get("is_online", True)),
        capabilities=caps,
        state=state,
    )


def group_to_json(g: Group) -> dict:
    return asdict(g)


def group_from_json(d: dict) -> Group:
    return Group(
        id=str(d.get("id") or ""),
        name=str(d.get("name") or ""),
        room=(d.get("room") or None),
        lamp_ids=list(d.get("lamp_ids") or []),
    )


def scene_to_json(s: Scene) -> dict:
    return asdict(s)


def scene_from_json(d: dict) -> Scene:
    return Scene(
        id=str(d.get("id") or ""),
        name=str(d.get("name") or ""),
        kind=str(d.get("kind") or "assistant"),
        payload=dict(d.get("payload") or {}),
    )

