"""Tests for kind-aware events_to_quarterly."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative import NarrativeEvent, events_to_quarterly


def _ev(date, kind, target, magnitude, sign, country="USA", magnitude_unit=None):
    if magnitude_unit is None:
        magnitude_unit = {
            "fiscal": "USD_bn",
            "monetary": "bps",
            "macropru": "ratio",
            "fx": "USD_bn",
            "structural": "z",
        }[kind]
    return NarrativeEvent(
        date=pd.Timestamp(date), country=country,
        magnitude=magnitude, magnitude_unit=magnitude_unit,
        target=target, subtarget=None, sign=sign,
        confidence=0.9, source_text="t", source_url="u",
        scoring_method="manual", kind=kind,
    )


def test_pure_fiscal_unchanged_when_kind_filter_none():
    """Backwards compat: no-kind-filter on all-fiscal list works as before."""
    events = [
        _ev("2020-01-15", "fiscal", "investment", 10.0, +1),
        _ev("2020-04-15", "fiscal", "consumption", 5.0, -1),
    ]
    s = events_to_quarterly(events)
    assert len(s) == 2
    assert s.iloc[0] == 10.0
    assert s.iloc[1] == -5.0


def test_mixed_kind_without_filter_raises():
    events = [
        _ev("2020-01-15", "fiscal", "investment", 10.0, +1),
        _ev("2020-04-15", "monetary", "policy_rate", 25.0, +1),
    ]
    with pytest.raises(ValueError, match="multiple kinds"):
        events_to_quarterly(events)


def test_kind_filter_monetary_yields_only_monetary():
    events = [
        _ev("2020-01-15", "fiscal", "investment", 10.0, +1),
        _ev("2020-01-25", "monetary", "policy_rate", 25.0, +1),
        _ev("2020-04-15", "monetary", "policy_rate", 50.0, -1),
    ]
    s = events_to_quarterly(events, kind_filter="monetary")
    assert s.iloc[0] == 25.0
    assert s.iloc[1] == -50.0


def test_kind_filter_macropru_uses_count_aggregation():
    """macropru: signed COUNT of actions per quarter, not magnitude sum."""
    events = [
        _ev("2020-01-10", "macropru", "capital_buffer", 100.0, +1),
        _ev("2020-02-10", "macropru", "capital_buffer", 250.0, +1),
        _ev("2020-02-25", "macropru", "ltv_dsti",       0.05,  -1),
    ]
    s = events_to_quarterly(events, kind_filter="macropru")
    assert s.iloc[0] == 1.0


def test_kind_filter_structural_uses_indicator():
    """structural: presence indicator (any signed event ⇒ ±1)."""
    events = [
        _ev("2020-01-10", "structural", "labor", 0.5, +1),
        _ev("2020-01-30", "structural", "trade", 0.2, +1),
    ]
    s = events_to_quarterly(events, kind_filter="structural")
    assert s.iloc[0] == 1.0


def test_kind_filter_with_unknown_kind_raises():
    events = [_ev("2020-01-15", "fiscal", "investment", 10.0, +1)]
    with pytest.raises(ValueError, match="kind_filter"):
        events_to_quarterly(events, kind_filter="not_a_kind")


def test_empty_after_kind_filter_returns_empty_series():
    events = [_ev("2020-01-15", "fiscal", "investment", 10.0, +1)]
    s = events_to_quarterly(events, kind_filter="monetary")
    assert s.empty
