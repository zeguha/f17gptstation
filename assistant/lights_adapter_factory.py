"""Adapter factory for smart lights.

The rest of the system should depend on the GaussAdapter protocol, not on a
concrete integration. This module selects an implementation based on config.
"""

from __future__ import annotations

from .lights_gauss_adapter import GaussAdapter, GaussMockAdapter


def build_gauss_adapter(kind: str) -> GaussAdapter:
    k = (kind or "").strip().lower()
    if k in ("mock", "demo", "test", ""):
        return GaussMockAdapter()
    if k in ("wiz", "wiz_lan", "wizlan", "wiz-local"):
        # Lazy import to keep base install minimal.
        from .lights_wiz_lan_adapter import WizLanAdapter

        return WizLanAdapter()
    # Safe fallback: still functional in dev.
    return GaussMockAdapter()

