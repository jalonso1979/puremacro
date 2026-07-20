"""F1 Slice A — each connector declares FALLBACK_POLICY (adopting
the Slice B fallback contract from inception)."""
from __future__ import annotations

import importlib

import pytest


_F1A_CONNECTORS = [
    "bi", "bnm", "bsp", "cbn", "cbe", "cbk",
]


@pytest.mark.parametrize("name", _F1A_CONNECTORS)
def test_connector_declares_fallback_policy(name):
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    assert hasattr(mod, "FALLBACK_POLICY"), (
        f"{name}.py must declare FALLBACK_POLICY (F1 Slice A contract)"
    )
    assert isinstance(mod.FALLBACK_POLICY, tuple), (
        f"{name}.FALLBACK_POLICY must be a tuple"
    )
    assert len(mod.FALLBACK_POLICY) >= 1, (
        f"{name}.FALLBACK_POLICY must have at least one stage"
    )


@pytest.mark.parametrize("name", _F1A_CONNECTORS)
def test_fallback_policy_stages_are_supported(name):
    from puremacro.narrative.sources._fallback import SUPPORTED_STAGES
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    bad = set(mod.FALLBACK_POLICY) - SUPPORTED_STAGES
    assert not bad, (
        f"{name}.FALLBACK_POLICY contains unsupported stages: {sorted(bad)}. "
        f"Supported: {sorted(SUPPORTED_STAGES)}"
    )
