"""F2.4 — the 7 fallback-aware connectors all declare valid FALLBACK_POLICY."""
from __future__ import annotations

import importlib

import pytest


_FALLBACK_CONNECTORS = [
    "eu_eurlex", "eu_parliament", "us_cbo",
    "rba", "bok", "riksbank", "sarb",
]


@pytest.mark.parametrize("name", _FALLBACK_CONNECTORS)
def test_connector_declares_fallback_policy(name):
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    assert hasattr(mod, "FALLBACK_POLICY"), (
        f"{name}.py must declare FALLBACK_POLICY (F2.4 contract)"
    )
    assert isinstance(mod.FALLBACK_POLICY, tuple), (
        f"{name}.FALLBACK_POLICY must be a tuple"
    )
    assert len(mod.FALLBACK_POLICY) >= 1, (
        f"{name}.FALLBACK_POLICY must have at least one stage"
    )


@pytest.mark.parametrize("name", _FALLBACK_CONNECTORS)
def test_fallback_policy_stages_are_supported(name):
    from puremacro.narrative.sources._fallback import SUPPORTED_STAGES
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    bad = set(mod.FALLBACK_POLICY) - SUPPORTED_STAGES
    assert not bad, (
        f"{name}.FALLBACK_POLICY contains unsupported stages: {sorted(bad)}. "
        f"Supported: {sorted(SUPPORTED_STAGES)}"
    )
