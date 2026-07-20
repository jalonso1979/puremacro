"""Tests for NarrativeEvent kind/language extension (Slice 1)."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative import NarrativeEvent


def _base(**kw):
    """Build a fiscal event with sane defaults; override via kwargs."""
    defaults = dict(
        date=pd.Timestamp("2020-03-15"),
        country="USA",
        magnitude=10.0,
        magnitude_unit="USD_bn",
        target="investment",
        subtarget="defense",
        sign=+1,
        confidence=0.8,
        source_text="test",
        source_url="https://example.test",
        scoring_method="manual",
    )
    defaults.update(kw)
    return defaults


def test_default_kind_is_fiscal():
    e = NarrativeEvent(**_base())
    assert e.kind == "fiscal"
    assert e.language == "en"


def test_explicit_monetary_kind_with_valid_target():
    e = NarrativeEvent(**_base(
        kind="monetary",
        target="policy_rate",
        magnitude_unit="bps",
        magnitude=25.0,
    ))
    assert e.kind == "monetary"
    assert e.target == "policy_rate"


def test_invalid_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        NarrativeEvent(**_base(kind="not_a_kind"))


def test_fiscal_target_rejected_for_monetary_kind():
    with pytest.raises(ValueError, match="target"):
        NarrativeEvent(**_base(kind="monetary", target="investment"))


def test_monetary_target_rejected_for_fiscal_kind():
    with pytest.raises(ValueError, match="target"):
        NarrativeEvent(**_base(kind="fiscal", target="policy_rate"))


@pytest.mark.parametrize("kind,target", [
    ("fiscal",     "investment"),
    ("fiscal",     "consumption"),
    ("fiscal",     "both"),
    ("monetary",   "policy_rate"),
    ("monetary",   "asset_purchase"),
    ("monetary",   "forward_guidance"),
    ("monetary",   "fx_intervention"),
    ("monetary",   "lending_facility"),
    ("macropru",   "capital_buffer"),
    ("macropru",   "ltv_dsti"),
    ("macropru",   "sector_limit"),
    ("macropru",   "reserve_requirement"),
    ("fx",         "intervention"),
    ("fx",         "peg_change"),
    ("structural", "labor"),
    ("structural", "product_market"),
    ("structural", "trade"),
    ("structural", "tax_admin"),
])
def test_all_valid_kind_target_combinations(kind, target):
    e = NarrativeEvent(**_base(kind=kind, target=target))
    assert e.kind == kind
    assert e.target == target


def test_language_default_en():
    e = NarrativeEvent(**_base())
    assert e.language == "en"


def test_explicit_language_passes_through():
    e = NarrativeEvent(**_base(language="es"))
    assert e.language == "es"


def test_to_dict_round_trip_preserves_kind_language():
    e = NarrativeEvent(**_base(kind="monetary", target="policy_rate",
                                language="de", magnitude_unit="bps"))
    d = e.to_dict()
    assert d["kind"] == "monetary"
    assert d["language"] == "de"
    e2 = NarrativeEvent.from_dict(d)
    assert e2.kind == "monetary"
    assert e2.language == "de"
    assert e2.target == "policy_rate"


def test_from_dict_legacy_payload_without_kind_defaults_fiscal():
    """Legacy serialized events must still load (kind absent → 'fiscal')."""
    legacy = dict(
        date="2020-03-15", country="USA", magnitude=10.0,
        magnitude_unit="USD_bn", target="investment", subtarget="defense",
        sign=+1, confidence=0.8, source_text="test",
        source_url="https://example.test", scoring_method="manual",
    )
    e = NarrativeEvent.from_dict(legacy)
    assert e.kind == "fiscal"
    assert e.language == "en"
