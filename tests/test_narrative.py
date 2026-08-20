"""Unit tests for the narrative module.

Covers the offline-deterministic surface: ``NarrativeEvent`` validation,
quarterly aggregation, dedup clustering / representative selection,
calendar diagnostics, and benchmark comparison. The replication
loaders are not exercised here — those make network calls and are
covered by the live-replication smoke tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.narrative import (
    NarrativeEvent, NarrativeInstrument,
    events_to_quarterly,
    cluster_events, deduplicate, representative,
    event_density, compare_to,
    # Re-exported replication helpers — verify the alias surface exists.
    dglp_csv_to_events,
    ramey_csv_to_events,
    romer_romer_2010_csv_to_events,
    romer_romer_2017_csv_to_events,
    mertens_ravn_csv_to_events,
    cloyne_csv_to_events,
    devries_csv_to_events,
    guajardo_csv_to_events,
    mitchell_csv_to_events,
)


# ---------------------------------------------------------------------------
# NarrativeEvent: validation contract
# ---------------------------------------------------------------------------


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


def test_narrative_event_clamps_confidence_and_abs_magnitude():
    e = _make_event(magnitude=-3.5, confidence=1.7)
    assert e.magnitude == 3.5
    assert e.confidence == 1.0
    e2 = _make_event(confidence=-0.2)
    assert e2.confidence == 0.0


def test_narrative_event_signed_magnitude():
    pos = _make_event(magnitude=2.0, sign=+1)
    neg = _make_event(magnitude=2.0, sign=-1)
    amb = _make_event(magnitude=2.0, sign=0)
    assert pos.signed_magnitude == 2.0
    assert neg.signed_magnitude == -2.0
    assert amb.signed_magnitude == 0.0


def test_narrative_event_rejects_invalid_target():
    with pytest.raises(ValueError, match="target"):
        _make_event(target="nonsense")


def test_narrative_event_rejects_invalid_sign():
    with pytest.raises(ValueError, match="sign"):
        _make_event(sign=2)


def test_narrative_event_rejects_invalid_scorer():
    with pytest.raises(ValueError, match="scoring_method"):
        _make_event(scoring_method="hand-wave")


def test_narrative_event_roundtrip_through_dict():
    e = _make_event(metadata={"act_id": "HR-1234"})
    d = e.to_dict()
    e2 = NarrativeEvent.from_dict(d)
    assert e.country == e2.country
    assert e.signed_magnitude == e2.signed_magnitude
    assert e2.metadata["act_id"] == "HR-1234"


def test_narrative_event_from_dict_drops_unknown_keys():
    d = _make_event().to_dict()
    d["accidental_field"] = "ignored"
    e = NarrativeEvent.from_dict(d)
    assert e.country == "USA"


# ---------------------------------------------------------------------------
# events_to_quarterly: aggregation rules
# ---------------------------------------------------------------------------


def _two_event_panel():
    return [
        _make_event(date=pd.Timestamp("2010-02-15"), magnitude=1.0, sign=+1),
        _make_event(date=pd.Timestamp("2010-02-25"), magnitude=3.0, sign=-1),
        _make_event(date=pd.Timestamp("2010-08-10"), magnitude=2.0, sign=+1),
    ]


def test_events_to_quarterly_sum_default():
    q = events_to_quarterly(_two_event_panel())
    # Q1 2010: +1 + (-3) = -2; Q2 zero; Q3 2010: +2.
    assert q.loc["2010-01-01"] == -2.0
    assert q.loc["2010-04-01"] == 0.0
    assert q.loc["2010-07-01"] == +2.0


def test_events_to_quarterly_no_sign_weight():
    q = events_to_quarterly(_two_event_panel(), sign_weighted=False)
    # Q1 2010: 1 + 3 = 4; Q3 2010: 2.
    assert q.loc["2010-01-01"] == 4.0
    assert q.loc["2010-07-01"] == 2.0


def test_events_to_quarterly_max_abs():
    q = events_to_quarterly(_two_event_panel(), aggregation="max")
    # Q1 2010: largest |signed| is -3 (sign preserved).
    assert q.loc["2010-01-01"] == -3.0


def test_events_to_quarterly_first():
    q = events_to_quarterly(_two_event_panel(), aggregation="first")
    # Earliest in Q1 2010 is the +1 event on Feb 15.
    assert q.loc["2010-01-01"] == +1.0


def test_events_to_quarterly_target_filter():
    events = [
        _make_event(date=pd.Timestamp("2010-02-15"), target="investment"),
        _make_event(date=pd.Timestamp("2010-02-15"), target="consumption"),
        _make_event(date=pd.Timestamp("2010-02-15"), target="both"),
    ]
    q_inv = events_to_quarterly(events, target_filter="investment")
    # investment + both → 2 events (each magnitude 1, sign +).
    assert q_inv.loc["2010-01-01"] == 2.0
    q_con = events_to_quarterly(events, target_filter="consumption")
    assert q_con.loc["2010-01-01"] == 2.0


def test_events_to_quarterly_confidence_threshold():
    events = [
        _make_event(date=pd.Timestamp("2010-02-15"), confidence=0.4),
        _make_event(date=pd.Timestamp("2010-02-15"), confidence=0.95),
    ]
    q = events_to_quarterly(events, confidence_threshold=0.5)
    assert q.loc["2010-01-01"] == 1.0  # only the high-confidence event


def test_events_to_quarterly_empty_returns_empty_series():
    q = events_to_quarterly([])
    assert q.empty
    assert q.name == "narrative_iv"


def test_events_to_quarterly_unknown_aggregation_raises():
    with pytest.raises(ValueError, match="aggregation"):
        events_to_quarterly(_two_event_panel(), aggregation="median")


# ---------------------------------------------------------------------------
# Deduplication: clustering, representative selection, audit
# ---------------------------------------------------------------------------


def test_cluster_events_groups_by_date_window():
    events = [
        _make_event(date=pd.Timestamp("2010-03-01"), country="USA",
                    magnitude=1.0, sign=+1, scoring_method="llm"),
        _make_event(date=pd.Timestamp("2010-04-15"), country="USA",
                    magnitude=1.2, sign=+1, scoring_method="manual"),
        # Different country — should not cluster.
        _make_event(date=pd.Timestamp("2010-03-05"), country="GBR",
                    magnitude=1.0, sign=+1, scoring_method="llm"),
    ]
    clusters = cluster_events(events, date_tol_days=90)
    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_cluster_events_respects_sign_when_required():
    events = [
        _make_event(date=pd.Timestamp("2010-03-01"), magnitude=1.0, sign=+1,
                    scoring_method="llm"),
        _make_event(date=pd.Timestamp("2010-03-10"), magnitude=1.0, sign=-1,
                    scoring_method="manual"),
    ]
    strict = cluster_events(events, require_same_sign=True)
    assert len(strict) == 2
    loose = cluster_events(events, require_same_sign=False)
    assert len(loose) == 1


def test_cluster_events_rejects_magnitude_outliers():
    events = [
        _make_event(date=pd.Timestamp("2010-03-01"), magnitude=1.0,
                    scoring_method="llm"),
        _make_event(date=pd.Timestamp("2010-03-10"), magnitude=10.0,
                    scoring_method="manual"),  # ratio 10 > tol 4
    ]
    clusters = cluster_events(events, magnitude_ratio_tol=4.0)
    assert len(clusters) == 2


def test_representative_prefers_manual_then_high_confidence():
    cluster = [
        _make_event(scoring_method="llm", confidence=0.9, magnitude=1.0),
        _make_event(scoring_method="keyword", confidence=0.8, magnitude=1.0),
        _make_event(scoring_method="manual", confidence=0.5, magnitude=1.0),
    ]
    rep = representative(cluster)
    assert rep.scoring_method == "manual"


def test_representative_within_tier_uses_confidence_then_magnitude():
    cluster = [
        _make_event(scoring_method="llm", confidence=0.7, magnitude=1.0),
        _make_event(scoring_method="llm", confidence=0.9, magnitude=1.0),
        _make_event(scoring_method="llm", confidence=0.9, magnitude=2.0),
    ]
    rep = representative(cluster)
    assert rep.confidence == 0.9 and rep.magnitude == 2.0


def test_deduplicate_returns_audit_for_dropped_duplicates():
    events = [
        _make_event(date=pd.Timestamp("2010-03-01"), magnitude=1.0,
                    scoring_method="llm",
                    source_url="https://a"),
        _make_event(date=pd.Timestamp("2010-03-10"), magnitude=1.0,
                    scoring_method="manual",
                    source_url="https://b"),
    ]
    reps, audit = deduplicate(events)
    assert len(reps) == 1
    assert reps[0].scoring_method == "manual"
    assert len(audit) == 1
    assert audit.iloc[0]["dup_method"] == "llm"
    assert audit.iloc[0]["dup_url"] == "https://a"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_event_density_handles_empty():
    out = event_density([])
    assert out["n_total"] == 0
    assert out["max_gap_q"] == 0


def test_event_density_counts_zero_gap_quarters():
    events = [
        _make_event(date=pd.Timestamp("2010-01-15")),
        _make_event(date=pd.Timestamp("2011-01-15")),  # 4-quarter gap
    ]
    out = event_density(events)
    assert out["n_total"] == 2
    assert out["max_gap_q"] >= 3


def test_compare_to_correlation_recovers_unit():
    idx = pd.date_range("2000-01-01", periods=20, freq="QS")
    s = pd.Series(np.arange(20.0), index=idx, name="a")
    out = compare_to(s, s)
    assert abs(out["correlation"] - 1.0) < 1e-12
    assert out["n_obs"] == 20


def test_compare_to_handles_disjoint_index():
    a = pd.Series([1.0], index=pd.date_range("2000-01-01", periods=1, freq="QS"))
    b = pd.Series([1.0], index=pd.date_range("2010-01-01", periods=1, freq="QS"))
    out = compare_to(a, b)
    assert out["n_obs"] == 0
    assert np.isnan(out["correlation"])


# ---------------------------------------------------------------------------
# Replication helpers: surface check (no network)
# ---------------------------------------------------------------------------


def test_replication_csv_to_events_helpers_are_callable():
    """The CSV → events helpers must be importable from the public
    namespace so examples don't reach into private modules."""
    helpers = [
        dglp_csv_to_events,
        ramey_csv_to_events,
        romer_romer_2010_csv_to_events,
        romer_romer_2017_csv_to_events,
        mertens_ravn_csv_to_events,
        cloyne_csv_to_events,
        devries_csv_to_events,
        guajardo_csv_to_events,
        mitchell_csv_to_events,
    ]
    for h in helpers:
        assert callable(h), f"{h!r} not callable"


def test_dglp_csv_to_events_round_trips_minimal_csv():
    df = pd.DataFrame({
        "date": [1992, 1993, 2010],
        "country": ["ITA", "ITA", "ESP"],
        "tax_based": [1.4, 0.4, 0.4],
        "spending_based": [0.5, 1.6, 1.4],
    })
    events = dglp_csv_to_events(df, flip_sign=True)
    assert len(events) > 0
    assert all(isinstance(e, NarrativeEvent) for e in events)
    # flip_sign=True turns DGLP consolidations into contractionary
    # (sign = -1) events under the puremacro convention.
    assert all(e.sign == -1 for e in events if e.magnitude > 0)


def test_narrative_instrument_from_events_builds_quarterly_series():
    events = _two_event_panel()
    inst = NarrativeInstrument.from_events(events)
    assert isinstance(inst.quarterly, pd.Series)
    assert inst.quarterly.name == "narrative_iv"
    diag = inst.diagnostics()
    assert diag["n_events"] == 3


def test_new_loaders_importable_from_narrative_namespace():
    from puremacro.narrative import (
        load_devries_2011, devries_csv_to_events,
        load_guajardo_2011, guajardo_csv_to_events,
        load_mitchell_historical, mitchell_csv_to_events,
    )
    for fn in [load_devries_2011, devries_csv_to_events,
               load_guajardo_2011, guajardo_csv_to_events,
               load_mitchell_historical, mitchell_csv_to_events]:
        assert callable(fn)


def test_from_canonical_accepts_new_flags_without_crash():
    """from_canonical with all new loaders disabled still works."""
    from puremacro.narrative import HomogeneousFiscalPanel
    panel = HomogeneousFiscalPanel.from_canonical(
        include_dglp=False,
        include_rr2017=False,
        include_rr2010_us=False,
        include_mr2013_us=False,
        include_cloyne_uk=False,
        include_ramey_us=False,
        include_devries=False,
        include_guajardo=False,
        include_mitchell=False,
    )
    assert panel.events == []
    assert panel.metadata["n_raw_events"] == 0


def test_devries_csv_to_events_long_format():
    """Long-format CSV (one row per event) round-trips to NarrativeEvents."""
    from puremacro.narrative.replication.devries_2011_consolidations import (
        devries_csv_to_events,
    )
    df = pd.DataFrame({
        "country": ["ITA", "ITA", "ESP", "ESP"],
        "year":    [1992,   1993,  2010,  2010],
        "quarter": [1,      2,     1,     3],
        "action_pct_gdp": [1.4, 0.5, 0.4, 1.2],
        "type": ["tax", "expenditure", "tax", "expenditure"],
        "narrative": ["Budget law 1992", "Spending review", "Retiro", "Recorte"],
    })
    events = devries_csv_to_events(df)
    assert len(events) == 4
    assert all(isinstance(e, NarrativeEvent) for e in events)
    assert all(e.magnitude_unit == "pct_gdp" for e in events)
    tax_events = [e for e in events if e.subtarget == "tax"]
    exp_events  = [e for e in events if e.subtarget == "expenditure"]
    assert len(tax_events) == 2
    assert len(exp_events) == 2
    # Consolidations enter as contractionary (-1).
    assert all(e.sign == -1 for e in events)
    # Source text comes from the narrative column.
    assert any("Budget law" in e.source_text for e in events)


def test_devries_csv_to_events_missing_type_defaults_to_general():
    from puremacro.narrative.replication.devries_2011_consolidations import (
        devries_csv_to_events,
    )
    df = pd.DataFrame({
        "country": ["FRA"],
        "year":    [2005],
        "quarter": [2],
        "action_pct_gdp": [0.8],
    })
    events = devries_csv_to_events(df)
    assert len(events) == 1
    assert events[0].subtarget == "general"
    assert events[0].target == "both"


def test_devries_csv_to_events_handles_float_year_column():
    """pd.read_csv returns float64 for integer columns with any NaN — must parse."""
    from puremacro.narrative.replication.devries_2011_consolidations import (
        devries_csv_to_events,
    )
    # Simulate what pd.read_csv produces for a year column with floats.
    df = pd.DataFrame({
        "country": ["DEU", "DEU"],
        "year":    [1978.0, 1980.0],   # floats, not ints
        "quarter": [1.0, 3.0],
        "action_pct_gdp": [0.5, 0.7],
        "type": ["tax", "expenditure"],
    })
    events = devries_csv_to_events(df)
    assert len(events) == 2
    # Grouping sorts by (country, subtarget, year); check both years are parsed.
    years = {e.date.year for e in events}
    assert years == {1978, 1980}


def test_guajardo_csv_to_events_includes_em_countries():
    from puremacro.narrative.replication.guajardo_2011_aeipf import (
        guajardo_csv_to_events,
    )
    df = pd.DataFrame({
        "country": ["BRA", "BRA", "ZAF", "USA"],
        "year":    [2005,   2006,  2008,  2003],
        "quarter": [1,      3,     2,     4],
        "action_pct_gdp": [0.8, 1.1, 0.5, 2.0],
        "category": ["expenditure", "tax", "combined", "tax"],
    })
    events = guajardo_csv_to_events(df)
    countries = {e.country for e in events}
    assert "BRA" in countries
    assert "ZAF" in countries
    assert "USA" in countries
    # "combined" → target="both"
    zaf_events = [e for e in events if e.country == "ZAF"]
    assert zaf_events[0].target == "both"


def test_guajardo_csv_to_events_sign_is_contractionary():
    from puremacro.narrative.replication.guajardo_2011_aeipf import (
        guajardo_csv_to_events,
    )
    df = pd.DataFrame({
        "country": ["MEX"],
        "year":    [2010],
        "quarter": [1],
        "action_pct_gdp": [1.5],
    })
    events = guajardo_csv_to_events(df)
    assert events[0].sign == -1
    assert events[0].metadata["replication"] == "guajardo_2011"


def test_guajardo_csv_to_events_handles_float_year_column():
    from puremacro.narrative.replication.guajardo_2011_aeipf import (
        guajardo_csv_to_events,
    )
    df = pd.DataFrame({
        "country": ["KOR", "POL"],
        "year":    [2000.0, 2005.0],
        "quarter": [2.0, 4.0],
        "action_pct_gdp": [0.6, 0.9],
        "category": ["tax", "expenditure"],
    })
    events = guajardo_csv_to_events(df)
    assert len(events) == 2
    assert events[0].date.year == 2000
    assert events[1].date.year == 2005


def test_mitchell_csv_to_events_distributes_annual_to_quarterly():
    from puremacro.narrative.replication.mitchell_historical import (
        mitchell_csv_to_events,
    )
    df = pd.DataFrame({
        "country": ["GBR", "GBR"],
        "year":    [1960, 1961],
        "gdp_share": [0.20, 0.22],  # government spending as fraction of GDP
    })
    events = mitchell_csv_to_events(df)
    # Annual → 4 quarterly events per year.
    assert len(events) == 8
    # Each quarterly event has magnitude = annual_gdp_share * 100 / 4.
    assert abs(events[0].magnitude - 0.20 * 100 / 4) < 1e-9
    assert events[0].confidence == 0.6
    assert events[0].scoring_method == "manual"
    assert events[0].magnitude_unit == "pct_gdp"
    # All events tagged as "both" (cannot distinguish investment vs consumption).
    assert all(e.target == "both" for e in events)


def test_mitchell_csv_to_events_handles_float_year_column():
    from puremacro.narrative.replication.mitchell_historical import (
        mitchell_csv_to_events,
    )
    df = pd.DataFrame({
        "country": ["FRA", "FRA"],
        "year":    [1960.0, 1961.0],   # floats from pd.read_csv
        "gdp_share": [0.18, 0.19],
    })
    events = mitchell_csv_to_events(df)
    assert len(events) == 8  # 2 years × 4 quarters
    assert events[0].date.year == 1960
    assert events[4].date.year == 1961


# ---------------------------------------------------------------------------
# Cloyne 2013 UK — effective_date column parsing
# ---------------------------------------------------------------------------


def test_cloyne_loader_uses_effective_date_when_present():
    """Cloyne CSV with effective_date column attaches implementation_profile."""
    rows = [
        # Bill announced 2010Q3; effective 2011Q1.
        {"date": "2010Q3", "exog": 0.4, "effective_date": "2011-01-01"},
    ]
    df = pd.DataFrame(rows)
    events = cloyne_csv_to_events(df)
    assert len(events) == 1
    e = events[0]
    assert e.date == pd.Timestamp("2010-07-01")
    assert e.implementation_profile == [(pd.Timestamp("2011-01-01"), 1.0)]
    assert e.delay_quarters == 2


def test_cloyne_loader_default_when_no_effective_date_column():
    rows = [{"date": "2003Q2", "exog": 0.3}]
    df = pd.DataFrame(rows)
    events = cloyne_csv_to_events(df)
    assert len(events) == 1
    assert events[0].implementation_profile == []
    assert events[0].effective_profile == [(pd.Timestamp("2003-04-01"), 1.0)]


def test_cloyne_loader_skips_invalid_effective_date():
    rows = [{"date": "2003Q2", "exog": 0.3, "effective_date": ""}]
    df = pd.DataFrame(rows)
    events = cloyne_csv_to_events(df)
    assert events[0].implementation_profile == []


# ---------------------------------------------------------------------------
# Mertens-Ravn 2013 — impl_q column parsing for kind="anticipated"
# ---------------------------------------------------------------------------


def test_mr2013_loader_anticipated_uses_impl_quarter_when_present():
    """For kind='anticipated', impl_q in CSV → implementation_profile points at it."""
    rows = [
        # Tax act announced 2003Q2; implementation 2004Q1.
        {"date": "2003Q2", "mtr_a": 0.5, "impl_q": "2004Q1"},
    ]
    df = pd.DataFrame(rows)
    events = mertens_ravn_csv_to_events(df, kind="anticipated")
    assert len(events) == 1
    e = events[0]
    assert e.date == pd.Timestamp("2003-04-01")
    assert e.implementation_profile == [(pd.Timestamp("2004-01-01"), 1.0)]
    assert e.delay_quarters == 3


def test_mr2013_loader_unanticipated_default_profile():
    """For kind='unanticipated', no profile is attached (announcement = impl)."""
    rows = [{"date": "2003Q2", "mtr_u": 0.4}]
    df = pd.DataFrame(rows)
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert len(events) == 1
    assert events[0].implementation_profile == []


def test_mr2013_loader_anticipated_no_impl_column_default_profile():
    """If CSV lacks impl_q, anticipated events fall back to default profile."""
    rows = [{"date": "2003Q2", "mtr_a": 0.5}]
    df = pd.DataFrame(rows)
    events = mertens_ravn_csv_to_events(df, kind="anticipated")
    assert len(events) == 1
    assert events[0].implementation_profile == []


# ---------------------------------------------------------------------------
# Ramey 2011 — is_news_shock metadata flag
# ---------------------------------------------------------------------------


def test_ramey_loader_marks_events_as_news_shocks():
    """Ramey 2011 events should carry is_news_shock=True in metadata."""
    rows = [{"date": "1965Q3", "news_q": 0.8}]
    df = pd.DataFrame(rows)
    events = ramey_csv_to_events(df)
    assert len(events) == 1
    assert events[0].metadata.get("is_news_shock") is True
    assert events[0].metadata.get("replication") == "ramey_2011"
    # Default profile — Ramey events deliberately don't backfill timing.
    assert events[0].implementation_profile == []


def test_cluster_events_merges_via_profile_quarter_overlap():
    """DGLP-style tranche event (multi-quarter profile starting Jan 1) and
    a quarterly-announcement event (date in Q3, empty profile) for the
    same fiscal episode should merge via profile-quarter overlap, even
    though their announcement dates are >90 days apart."""
    from puremacro.narrative.dedup import cluster_events
    # DGLP-style: USA 1981 tranche, profile spans 4 quarters of 1981
    dglp = NarrativeEvent(
        date=pd.Timestamp("1981-01-01"),
        country="USA",
        magnitude=0.5,
        magnitude_unit="pct_gdp",
        target="consumption",
        subtarget="tax",
        sign=-1,
        confidence=1.0,
        source_text="DGLP USA 1981",
        source_url="https://example.org/dglp",
        scoring_method="manual",
        metadata={"replication": "dglp_2011"},
        implementation_profile=[
            (pd.Timestamp("1981-01-01"), 0.25),
            (pd.Timestamp("1981-04-01"), 0.25),
            (pd.Timestamp("1981-07-01"), 0.25),
            (pd.Timestamp("1981-10-01"), 0.25),
        ],
    )
    # RR2017-style: USA 1981 announcement Q3, empty profile (so
    # effective_profile fallback = [(1981-07-01, 1.0)]).
    rr2017 = NarrativeEvent(
        date=pd.Timestamp("1981-07-01"),
        country="USA",
        magnitude=0.6,
        magnitude_unit="pct_gdp",
        target="consumption",
        subtarget="tax",
        sign=-1,
        confidence=1.0,
        source_text="RR2017 USA 1981",
        source_url="https://example.org/rr2017",
        scoring_method="manual",
        metadata={"replication": "rr_2017"},
    )
    clusters = cluster_events([dglp, rr2017], date_tol_days=90)
    # Profile overlap: DGLP's {1981Q1..Q4} ∩ RR2017's {1981Q3} = {1981Q3} → merge
    assert len(clusters) == 1, (
        "Expected DGLP tranche event and RR2017 Q3 announcement to merge "
        "via profile-quarter overlap"
    )


def test_cluster_events_does_not_merge_disjoint_profile_quarters():
    """Two events with profiles in different quarters of the same year
    should NOT merge (different fiscal acts within the year)."""
    from puremacro.narrative.dedup import cluster_events
    # Italy Q1 1993 income-tax hike
    e1 = NarrativeEvent(
        date=pd.Timestamp("1993-01-15"),
        country="ITA",
        magnitude=0.4,
        magnitude_unit="pct_gdp",
        target="consumption",
        subtarget="tax",
        sign=-1,
        confidence=1.0,
        source_text="Italy income tax 1993Q1",
        source_url="https://example.org/ita1",
        scoring_method="manual",
    )
    # Italy Q3 1993 VAT change (independent act, 7+ months later)
    e2 = NarrativeEvent(
        date=pd.Timestamp("1993-08-15"),
        country="ITA",
        magnitude=0.5,
        magnitude_unit="pct_gdp",
        target="consumption",
        subtarget="tax",
        sign=-1,
        confidence=1.0,
        source_text="Italy VAT 1993Q3",
        source_url="https://example.org/ita2",
        scoring_method="manual",
    )
    clusters = cluster_events([e1, e2], date_tol_days=90)
    # Date window: 212 days apart → outside 90-day tol.
    # Profiles (both empty → effective = single-quarter): {1993Q1} vs {1993Q3}, disjoint.
    # → 2 separate clusters
    assert len(clusters) == 2


def test_cluster_events_default_profile_unchanged_within_window():
    """Two events with default empty profiles, dates within 90 days, should
    still merge as before — profile-overlap addition is purely additive."""
    from puremacro.narrative.dedup import cluster_events
    e1 = NarrativeEvent(
        date=pd.Timestamp("2010-02-15"),
        country="USA",
        magnitude=0.4,
        magnitude_unit="pct_gdp",
        target="consumption",
        subtarget="tax",
        sign=+1,
        confidence=1.0,
        source_text="...",
        source_url="https://example.org/x",
        scoring_method="manual",
    )
    e2 = NarrativeEvent(
        date=pd.Timestamp("2010-04-10"),
        country="USA",
        magnitude=0.5,
        magnitude_unit="pct_gdp",
        target="consumption",
        subtarget="tax",
        sign=+1,
        confidence=1.0,
        source_text="...",
        source_url="https://example.org/y",
        scoring_method="manual",
    )
    clusters = cluster_events([e1, e2], date_tol_days=90)
    # 54 days apart, within 90-day window → merge (existing behavior)
    assert len(clusters) == 1


def test_from_canonical_adler_2024_supersedes_dglp_in_dedup(tmp_path):
    """When both Adler 2024 and DGLP 2011 cover the same OECD country-year,
    the dedup'd panel surfaces Adler 2024 (priority-1 within canonical)."""
    from puremacro.narrative.panel import HomogeneousFiscalPanel
    adler_csv = tmp_path / "adler.csv"
    adler_csv.write_text(
        "country,year,tax_pct_gdp,exp_pct_gdp\n"
        "USA,1985,0.5,0.0\n", encoding="utf-8")
    dglp_csv = tmp_path / "dglp.csv"
    dglp_csv.write_text(
        "country,year,tax_pct_gdp,exp_pct_gdp\n"
        "USA,1985,0.5,0.0\n", encoding="utf-8")
    panel = HomogeneousFiscalPanel.from_canonical(
        countries=["USA"],
        include_adler_2024=True,
        include_dglp=True,
        include_rr2017=False,
        include_rr2010_us=False,
        include_mr2013_us=False,
        include_cloyne_uk=False,
        include_ramey_us=False,
        include_devries=False,
        include_guajardo=False,
        include_eu_nms_2023=False,
        canonical_csv_paths={
            "adler_2024": str(adler_csv),
            "dglp_2011":  str(dglp_csv),
        },
    )
    usa = [e for e in panel.events if e.country == "USA"]
    assert len(usa) == 1
    assert usa[0].metadata.get("replication") == "adler_2024"


def test_from_canonical_eu_nms_2023_loads_independently(tmp_path):
    """EU NMS 2023 events appear in the panel; they don't overlap with DGLP."""
    from puremacro.narrative.panel import HomogeneousFiscalPanel
    eu_csv = tmp_path / "eu_nms.csv"
    eu_csv.write_text(
        "country,year,category,impact_t,impact_t+1,impact_t+2,"
        "impact_t+3,impact_t+4,impact_t+5,gdp,endogeneity\n"
        "POL,2010,Personal income tax,2.0,1.5,1.0,0.0,0.0,0.0,400,exogenous\n", encoding="utf-8")
    panel = HomogeneousFiscalPanel.from_canonical(
        countries=["POL"],
        include_adler_2024=False,
        include_dglp=False,
        include_rr2017=False,
        include_rr2010_us=False,
        include_mr2013_us=False,
        include_cloyne_uk=False,
        include_ramey_us=False,
        include_devries=False,
        include_guajardo=False,
        include_eu_nms_2023=True,
        canonical_csv_paths={"eu_nms_2023": str(eu_csv)},
    )
    pol = [e for e in panel.events if e.country == "POL"]
    assert len(pol) == 1
    assert pol[0].metadata.get("replication") == "eu_nms_2023"


def test_from_canonical_imf_covid_2022_loads_independently(tmp_path):
    """COVID events appear in the panel; they don't overlap with DGLP/Adler."""
    from puremacro.narrative.panel import HomogeneousFiscalPanel
    covid_csv = tmp_path / "covid.csv"
    covid_csv.write_text(
        "country,date,policy,policy_class,magnitude_pct_gdp,description\n"
        "USA,2020-04-15,unemployment_benefits,fiscal,2.5,CARES Act\n", encoding="utf-8")
    panel = HomogeneousFiscalPanel.from_canonical(
        countries=["USA"],
        include_adler_2024=False,
        include_dglp=False,
        include_rr2017=False,
        include_rr2010_us=False,
        include_mr2013_us=False,
        include_cloyne_uk=False,
        include_ramey_us=False,
        include_devries=False,
        include_guajardo=False,
        include_eu_nms_2023=False,
        include_imf_covid_2022=True,
        canonical_csv_paths={"imf_covid_2022": str(covid_csv)},
    )
    usa = [e for e in panel.events if e.country == "USA"]
    assert len(usa) == 1
    assert usa[0].metadata.get("replication") == "imf_covid_2022"
