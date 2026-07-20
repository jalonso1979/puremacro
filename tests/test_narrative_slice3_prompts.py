"""Slice 3: smoke tests for macropru / fx / structural LLM prompts."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative.scoring.llm import (
    _PROMPTS, _build_prompt, _validate_event_dict, score_llm,
)


# ---------------------------------------------------------------------------
# _build_prompt — coverage of the three kinds shipped in Slice 1 but not
# previously smoke-tested in their own test
# ---------------------------------------------------------------------------
def test_build_prompt_macropru_contains_target_enum():
    p = _build_prompt(kind="macropru", language="en", country="GBR",
                      date="2022-09-01", text="capital buffer increased")
    low = p.lower()
    assert "capital_buffer" in low
    assert "ltv_dsti" in low
    assert "tightening" in low
    assert "loosening" in low


def test_build_prompt_fx_contains_target_enum():
    p = _build_prompt(kind="fx", language="en", country="JPN",
                      date="2022-10-21", text="MoF intervened to buy JPY")
    low = p.lower()
    assert "intervention" in low
    assert "peg_change" in low


def test_build_prompt_structural_contains_target_enum():
    p = _build_prompt(kind="structural", language="en", country="ITA",
                      date="2018-01-01", text="labor reform passed")
    low = p.lower()
    assert "labor" in low
    assert "product_market" in low
    assert "trade" in low
    assert "tax_admin" in low


# ---------------------------------------------------------------------------
# _validate_event_dict — per-kind acceptance / rejection
# ---------------------------------------------------------------------------
def test_validate_macropru_accepts_well_formed_dict():
    d = {
        "target": "capital_buffer",
        "magnitude_pct": 0.5,
        "subtarget": None,
        "sign": +1,
        "confidence": 0.8,
        "excerpt": "the FPC raised the CCyB",
    }
    assert _validate_event_dict(d, kind="macropru") is True


def test_validate_macropru_rejects_wrong_target():
    d = {
        "target": "investment",   # fiscal target, not macropru
        "magnitude_pct": 0.5,
        "sign": +1,
        "confidence": 0.8,
    }
    assert _validate_event_dict(d, kind="macropru") is False


def test_validate_fx_accepts_well_formed_dict():
    d = {
        "target": "intervention",
        "magnitude_usd_bn": 36.0,
        "sign": +1,
        "confidence": 0.9,
    }
    assert _validate_event_dict(d, kind="fx") is True


def test_validate_structural_accepts_well_formed_dict():
    d = {
        "target": "labor",
        "magnitude_z": 1.5,
        "sign": +1,
        "confidence": 0.7,
    }
    assert _validate_event_dict(d, kind="structural") is True


def test_validate_structural_rejects_wrong_magnitude_key():
    """The structural schema uses magnitude_z, not magnitude_usd_bn."""
    d = {
        "target": "labor",
        "magnitude_usd_bn": 1.5,   # wrong key
        "sign": +1,
        "confidence": 0.7,
    }
    assert _validate_event_dict(d, kind="structural") is False


# ---------------------------------------------------------------------------
# score_llm dry_run for the three kinds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["macropru", "fx", "structural"])
def test_score_llm_dry_run_supports_kind(kind):
    records = [(pd.Timestamp("2020-01-01"), "x", "u")]
    out = score_llm(records, backend=None, kind=kind, dry_run=True)
    assert out == []
