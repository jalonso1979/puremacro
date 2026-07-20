"""F2.4 — SUPPORTED_STAGES registry + policy validation."""
from __future__ import annotations

import pytest


def test_supported_stages_has_expected_entries():
    from puremacro.narrative.sources._fallback import SUPPORTED_STAGES
    assert SUPPORTED_STAGES == frozenset({"live", "wayback", "playwright"})


def test_unknown_stage_in_policy_raises():
    from puremacro.narrative.sources._fallback import fetch_with_fallback
    with pytest.raises(ValueError, match="unknown stage"):
        fetch_with_fallback(
            "https://example.com/", policy=("not_a_stage",), source="x",
        )


def test_empty_policy_raises():
    from puremacro.narrative.sources._fallback import fetch_with_fallback
    with pytest.raises(ValueError, match="empty policy"):
        fetch_with_fallback(
            "https://example.com/", policy=(), source="x",
        )


def test_fallback_exhausted_is_runtimeerror():
    from puremacro.narrative.sources._fallback import FallbackExhaustedError
    assert issubclass(FallbackExhaustedError, RuntimeError)


def test_fallback_stage_unavailable_is_runtimeerror():
    from puremacro.narrative.sources._fallback import FallbackStageUnavailable
    assert issubclass(FallbackStageUnavailable, RuntimeError)
