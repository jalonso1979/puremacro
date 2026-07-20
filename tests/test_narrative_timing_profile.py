"""Tests for NarrativeEvent.implementation_profile and the two-series
NarrativeInstrument shape introduced by the announcement-vs-
implementation timing spec."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative import NarrativeEvent


def _make_event(**overrides) -> NarrativeEvent:
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


def test_event_default_profile_is_empty_list():
    e = _make_event()
    assert e.implementation_profile == []


def test_event_effective_profile_falls_back_to_single_quarter():
    e = _make_event()
    prof = e.effective_profile
    assert prof == [(pd.Timestamp("2010-03-15"), 1.0)]


def test_event_delay_quarters_zero_for_default():
    e = _make_event(date=pd.Timestamp("2010-03-15"))
    assert e.delay_quarters == 0


def test_event_delay_quarters_multi():
    e = _make_event(
        date=pd.Timestamp("2010-01-15"),
        implementation_profile=[
            (pd.Timestamp("2010-07-01"), 0.5),
            (pd.Timestamp("2010-10-01"), 0.5),
        ],
    )
    # Announcement quarter 2010Q1, first impl quarter 2010Q3 → delay 2.
    assert e.delay_quarters == 2


def test_event_profile_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="weights"):
        _make_event(implementation_profile=[
            (pd.Timestamp("2010-04-01"), 0.3),
            (pd.Timestamp("2010-07-01"), 0.4),
        ])


def test_event_profile_dates_must_not_predate_announcement():
    with pytest.raises(ValueError, match="predate"):
        _make_event(
            date=pd.Timestamp("2010-06-01"),
            implementation_profile=[(pd.Timestamp("2009-01-01"), 1.0)],
        )


def test_event_profile_same_quarter_as_announcement_ok():
    # Profile date a few days before announcement is allowed (< 7-day slack).
    e = _make_event(
        date=pd.Timestamp("2010-03-31"),
        implementation_profile=[(pd.Timestamp("2010-03-25"), 1.0)],
    )
    assert len(e.implementation_profile) == 1


def test_event_round_trip_with_profile():
    e = _make_event(implementation_profile=[
        (pd.Timestamp("2010-04-01"), 0.5),
        (pd.Timestamp("2010-07-01"), 0.5),
    ])
    d = e.to_dict()
    e2 = NarrativeEvent.from_dict(d)
    assert len(e2.implementation_profile) == 2
    for (d1, w1), (d2, w2) in zip(e.implementation_profile, e2.implementation_profile):
        assert pd.Timestamp(d1) == pd.Timestamp(d2)
        assert w1 == pytest.approx(w2)


def test_event_round_trip_legacy_dict_without_profile_key():
    # A dict serialized before this spec landed has no implementation_profile key.
    d = dict(
        date="2010-03-15", country="USA", magnitude=1.0, magnitude_unit="pct_gdp",
        target="investment", subtarget=None, sign=+1, confidence=0.9,
        source_text="...", source_url="https://example.org/x",
        scoring_method="manual", metadata={},
    )
    e = NarrativeEvent.from_dict(d)
    assert e.implementation_profile == []
    assert e.effective_profile == [(pd.Timestamp("2010-03-15"), 1.0)]


# ---------------------------------------------------------------------------
# Step 2 tests: events_to_quarterly iv_kind switch
# ---------------------------------------------------------------------------

from puremacro.narrative.aggregate import events_to_quarterly


def _ev(date, mag, sign=+1, profile=None):
    return NarrativeEvent(
        date=pd.Timestamp(date), country="USA",
        magnitude=mag, magnitude_unit="pct_gdp",
        target="investment", subtarget=None, sign=sign,
        confidence=1.0, source_text="...",
        source_url="https://example.org/x",
        scoring_method="manual",
        implementation_profile=profile or [],
    )


def test_events_to_quarterly_announcement_kind_matches_legacy_for_default_profile():
    events = [
        _ev("2010-02-15", 1.0),
        _ev("2010-08-01", 2.0, sign=-1),
    ]
    legacy = events_to_quarterly(events, iv_kind="announcement")
    # Mass on announcement quarters: 2010Q1 = +1.0, 2010Q3 = -2.0; Q2 = 0.
    # Q4 is outside the min–max range so is absent from the series (treated as 0).
    assert legacy.loc[pd.Timestamp("2010-01-01")] == pytest.approx(1.0)
    assert legacy.loc[pd.Timestamp("2010-04-01")] == pytest.approx(0.0)
    assert legacy.loc[pd.Timestamp("2010-07-01")] == pytest.approx(-2.0)
    assert legacy.get(pd.Timestamp("2010-10-01"), 0.0) == pytest.approx(0.0)


def test_events_to_quarterly_implementation_distributes_via_profile():
    # Use 2010-01-05 so the profile date 2010-01-01 is within the 7-day slack.
    e = _ev(
        "2010-01-05", mag=1.5, sign=+1,
        profile=[
            (pd.Timestamp("2010-01-01"), 0.4),
            (pd.Timestamp("2010-04-01"), 0.4),
            (pd.Timestamp("2010-07-01"), 0.2),
        ],
    )
    impl = events_to_quarterly([e], iv_kind="implementation")
    assert impl.loc[pd.Timestamp("2010-01-01")] == pytest.approx(0.60)
    assert impl.loc[pd.Timestamp("2010-04-01")] == pytest.approx(0.60)
    assert impl.loc[pd.Timestamp("2010-07-01")] == pytest.approx(0.30)


def test_events_to_quarterly_implementation_default_profile_matches_announcement():
    # Trivial profile (none) → implementation series equals announcement series.
    events = [_ev("2010-02-15", 1.0), _ev("2011-08-01", 2.0, sign=-1)]
    ann = events_to_quarterly(events, iv_kind="announcement")
    impl = events_to_quarterly(events, iv_kind="implementation")
    pd.testing.assert_series_equal(ann, impl)


def test_events_to_quarterly_unknown_iv_kind_raises():
    with pytest.raises(ValueError, match="iv_kind"):
        events_to_quarterly([_ev("2010-02-15", 1.0)], iv_kind="bogus")


# ---------------------------------------------------------------------------
# Step 3 tests: NarrativeInstrument holds two parallel series
# ---------------------------------------------------------------------------

from puremacro.narrative import NarrativeInstrument


def test_instrument_default_quarterly_is_implementation_series():
    e = _ev(
        "2010-01-05", mag=1.5, sign=+1,
        profile=[
            (pd.Timestamp("2010-01-01"), 0.5),
            (pd.Timestamp("2010-04-01"), 0.5),
        ],
    )
    inst = NarrativeInstrument.from_events([e])
    # implementation: 2010Q1 = 0.75, 2010Q2 = 0.75
    assert inst.quarterly is inst.implementation_series
    assert inst.implementation_series.loc[pd.Timestamp("2010-01-01")] == pytest.approx(0.75)
    assert inst.implementation_series.loc[pd.Timestamp("2010-04-01")] == pytest.approx(0.75)
    # announcement: full mass at 2010Q1
    assert inst.announcement_series.loc[pd.Timestamp("2010-01-01")] == pytest.approx(1.5)
    assert inst.announcement_series.loc[pd.Timestamp("2010-04-01")] == pytest.approx(0.0)


def test_instrument_two_series_match_for_default_profile():
    e = _ev("2010-02-01", 2.0, sign=+1)  # no profile
    inst = NarrativeInstrument.from_events([e])
    pd.testing.assert_series_equal(inst.announcement_series, inst.implementation_series)


def test_instrument_diagnostics_includes_profile_stats():
    events = [
        _ev("2010-01-15", 1.0, sign=+1),
        _ev("2011-01-15", 1.0, sign=+1, profile=[
            (pd.Timestamp("2011-04-01"), 0.5),
            (pd.Timestamp("2011-07-01"), 0.5),
        ]),
    ]
    inst = NarrativeInstrument.from_events(events)
    diag = inst.diagnostics()
    assert diag["n_events_with_profile"] == 1
    assert diag["max_profile_span_quarters"] == 2
    assert "announcement_vs_implementation_corr" in diag


def test_instrument_diagnostics_corr_is_one_for_default_profiles():
    events = [_ev("2010-01-15", 1.0), _ev("2010-07-15", 2.0, sign=-1)]
    inst = NarrativeInstrument.from_events(events)
    diag = inst.diagnostics()
    assert diag["announcement_vs_implementation_corr"] == pytest.approx(1.0)
