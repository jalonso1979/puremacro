"""Tests for the profile-driven SurpriseFilter rewrite."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative import NarrativeEvent
from puremacro.narrative.quality.surprise_filter import SurpriseFilter


def _ev(**overrides) -> NarrativeEvent:
    base = dict(
        date=pd.Timestamp("2010-03-15"),
        country="USA",
        magnitude=1.0,
        magnitude_unit="pct_gdp",
        target="investment",
        subtarget=None,
        sign=+1,
        confidence=0.9,
        source_text="...",
        source_url="https://example.org/x",
        scoring_method="manual",
    )
    base.update(overrides)
    return NarrativeEvent(**base)


def test_filter_flags_multi_quarter_profile():
    f = SurpriseFilter(decay_factor=0.5, confidence_floor=0.0)
    e = _ev(
        date=pd.Timestamp("2010-01-15"),
        implementation_profile=[
            (pd.Timestamp("2010-04-01"), 0.5),
            (pd.Timestamp("2010-07-01"), 0.5),
        ],
    )
    out = f.apply([e])
    assert len(out) == 1
    assert out[0].metadata.get("anticipation_flag") is True
    assert out[0].confidence == pytest.approx(0.45)


def test_filter_flags_delayed_single_quarter_profile():
    f = SurpriseFilter(decay_factor=0.5, confidence_floor=0.0)
    e = _ev(
        date=pd.Timestamp("2010-01-15"),
        implementation_profile=[(pd.Timestamp("2010-07-01"), 1.0)],
    )
    out = f.apply([e])
    assert out[0].metadata.get("anticipation_flag") is True


def test_filter_does_not_flag_default_profile_outside_budget_calendar():
    f = SurpriseFilter(decay_factor=0.5, confidence_floor=0.0)
    # Default profile + USA budget month is February → March event with no
    # subtarget should not be flagged via legacy budget path.
    e = _ev(date=pd.Timestamp("2010-03-15"), subtarget=None)
    out = f.apply([e])
    assert out[0].metadata.get("anticipation_flag") is not True
    assert out[0].confidence == pytest.approx(0.9)


def test_filter_legacy_budget_calendar_path_still_works():
    # Default profile, but date in USA budget month (Feb) AND subtarget in
    # _BUDGET_SUBTARGETS → still flagged via legacy fallback.
    f = SurpriseFilter(decay_factor=0.5, confidence_floor=0.0)
    e = _ev(date=pd.Timestamp("2010-02-10"), subtarget="transfers")
    out = f.apply([e])
    assert out[0].metadata.get("anticipation_flag") is True


def test_filter_unanticipated_metadata_preempts_profile_check():
    # Even if profile is multi-quarter, an explicit unanticipated tag wins.
    f = SurpriseFilter(decay_factor=0.5, confidence_floor=0.0)
    e = _ev(
        date=pd.Timestamp("2010-01-15"),
        implementation_profile=[
            (pd.Timestamp("2010-04-01"), 0.5),
            (pd.Timestamp("2010-07-01"), 0.5),
        ],
    )
    e.metadata["unanticipated"] = True
    out = f.apply([e])
    assert out[0].metadata.get("anticipation_flag") is not True
    assert out[0].confidence == pytest.approx(0.9)
