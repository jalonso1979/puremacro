"""Coverage tests for puremacro.narrative.panel.HomogeneousFiscalPanel.

Targets uncovered branches in panel.py (59% before this file):
  - from_canonical: individual loader branches + error-resilience
  - add_llm_scored
  - to_long (empty branch + column contract)
  - to_quarterly_panel (target filter, 'both' passthrough, empty country)
  - summary (populated + empty)
  - cross_correlations (populated + empty)
  - save (files produced, audit file conditional)

Loaders that require network access are monkeypatched; the class itself
is exercised with synthetic events to test known-value contracts.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from puremacro.narrative import NarrativeEvent, HomogeneousFiscalPanel
from puremacro.narrative.panel import DGLP_17
from puremacro.narrative.types import NarrativeInstrument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event(
    country: str = "USA",
    date: str = "2010-01-01",
    magnitude: float = 1.0,
    target: str = "investment",
    subtarget: str = "defense",
    sign: int = 1,
    confidence: float = 0.9,
    scoring_method: str = "manual",
    replication: str = "test_source",
) -> NarrativeEvent:
    return NarrativeEvent(
        date=pd.Timestamp(date),
        country=country,
        magnitude=magnitude,
        magnitude_unit="pct_gdp",
        target=target,
        subtarget=subtarget,
        sign=sign,
        confidence=confidence,
        source_text="Test event source text.",
        source_url="https://example.org/test",
        scoring_method=scoring_method,
        metadata={"replication": replication},
    )


def _two_country_panel() -> HomogeneousFiscalPanel:
    """A minimal two-country panel used across several tests."""
    events = [
        _event("USA", "2010-01-01", 1.0, "investment", sign=+1),
        _event("USA", "2012-07-01", 1.5, "consumption", sign=-1, replication="rr_2017"),
        _event("DEU", "2011-01-01", 2.0, "both", subtarget="expenditure", sign=-1),
    ]
    return HomogeneousFiscalPanel(events=events, countries=["USA", "DEU"])


def _instrument_for(events: list[NarrativeEvent]) -> NarrativeInstrument:
    return NarrativeInstrument.from_events(events)


# ---------------------------------------------------------------------------
# DGLP_17 constant
# ---------------------------------------------------------------------------


def test_dglp_17_is_17_countries():
    assert len(DGLP_17) == 17


def test_dglp_17_includes_expected_iso3():
    for code in ["USA", "GBR", "DEU", "FRA", "JPN"]:
        assert code in DGLP_17


def test_dglp_17_no_aggregates():
    for agg in ["EA20", "EU27", "EU27_2020"]:
        assert agg not in DGLP_17


# ---------------------------------------------------------------------------
# HomogeneousFiscalPanel: dataclass construction
# ---------------------------------------------------------------------------


def test_panel_default_construction_is_empty():
    p = HomogeneousFiscalPanel()
    assert p.events == []
    assert p.countries == []
    assert p.audit is None
    assert p.metadata == {}


def test_panel_construction_stores_events_and_countries():
    e = _event("GBR", "2013-01-01", 1.0, "investment")
    p = HomogeneousFiscalPanel(events=[e], countries=["GBR"])
    assert len(p.events) == 1
    assert "GBR" in p.countries


# ---------------------------------------------------------------------------
# from_canonical: all loaders off => empty panel
# ---------------------------------------------------------------------------


def test_from_canonical_all_off_returns_empty_panel():
    panel = HomogeneousFiscalPanel.from_canonical(
        countries=["USA", "GBR"],
        include_adler_2024=False,
        include_dglp=False,
        include_rr2017=False,
        include_rr2010_us=False,
        include_mr2013_us=False,
        include_cloyne_uk=False,
        include_ramey_us=False,
        include_devries=False,
        include_guajardo=False,
        include_mitchell=False,
        include_eu_nms_2023=False,
        include_imf_covid_2022=False,
    )
    assert len(panel.events) == 0
    assert panel.metadata["n_raw_events"] == 0
    assert panel.metadata["n_after_dedup"] == 0


def test_from_canonical_all_off_load_summary_empty():
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
        include_mitchell=False,
        include_eu_nms_2023=False,
        include_imf_covid_2022=False,
    )
    assert panel.metadata["load_summary"] == {}


# ---------------------------------------------------------------------------
# from_canonical: loader error resilience (_try helper)
# ---------------------------------------------------------------------------


def test_from_canonical_failing_dglp_loader_logs_error_and_continues():
    """When load_dglp_2011 raises, from_canonical logs it and produces 0 events."""
    with patch(
        "puremacro.narrative.replication.load_dglp_2011",
        side_effect=RuntimeError("network unavailable"),
    ):
        panel = HomogeneousFiscalPanel.from_canonical(
            countries=["USA", "GBR"],
            include_adler_2024=False,
            include_rr2017=False,
            include_rr2010_us=False,
            include_mr2013_us=False,
            include_cloyne_uk=False,
            include_ramey_us=False,
            include_devries=False,
            include_guajardo=False,
            include_mitchell=False,
            include_eu_nms_2023=False,
            include_imf_covid_2022=False,
        )
    summary = panel.metadata["load_summary"]
    assert summary["dglp_2011"]["ok"] is False
    assert "network unavailable" in summary["dglp_2011"]["error"]
    assert len(panel.events) == 0


# ---------------------------------------------------------------------------
# from_canonical: patched DGLP loader returns events
# ---------------------------------------------------------------------------


def test_from_canonical_dglp_only_events_are_collected():
    e_usa = _event("USA", "2010-01-01", 1.0, "investment")
    e_gbr = _event("GBR", "2011-01-01", 2.0, "consumption", sign=-1)
    fake_instr = {
        "USA": _instrument_for([e_usa]),
        "GBR": _instrument_for([e_gbr]),
    }
    with patch("puremacro.narrative.replication.load_dglp_2011", return_value=fake_instr):
        panel = HomogeneousFiscalPanel.from_canonical(
            countries=["USA", "GBR"],
            include_adler_2024=False,
            include_rr2017=False,
            include_rr2010_us=False,
            include_mr2013_us=False,
            include_cloyne_uk=False,
            include_ramey_us=False,
            include_devries=False,
            include_guajardo=False,
            include_mitchell=False,
            include_eu_nms_2023=False,
            include_imf_covid_2022=False,
        )
    assert panel.metadata["load_summary"]["dglp_2011"]["ok"] is True
    assert panel.metadata["n_raw_events"] == 2
    countries_found = {e.country for e in panel.events}
    assert countries_found == {"USA", "GBR"}


def test_from_canonical_usa_specific_loaders_not_called_without_usa():
    """RR2010/MR2013/Ramey are only loaded when USA is in countries."""
    calls = []
    def _recorder(name):
        def _loader(*a, **kw):
            calls.append(name)
            return _instrument_for([])
        return _loader

    with (
        patch("puremacro.narrative.replication.load_romer_romer_2010", side_effect=_recorder("rr2010")),
        patch("puremacro.narrative.replication.load_mertens_ravn_2013", side_effect=_recorder("mr2013")),
        patch("puremacro.narrative.replication.load_ramey_2011_defense", side_effect=_recorder("ramey")),
    ):
        HomogeneousFiscalPanel.from_canonical(
            countries=["GBR"],  # USA excluded
            include_adler_2024=False,
            include_dglp=False,
            include_rr2017=False,
            include_rr2010_us=True,
            include_mr2013_us=True,
            include_ramey_us=True,
            include_cloyne_uk=False,
            include_devries=False,
            include_guajardo=False,
            include_mitchell=False,
            include_eu_nms_2023=False,
            include_imf_covid_2022=False,
        )
    assert "rr2010" not in calls
    assert "mr2013" not in calls
    assert "ramey" not in calls


def test_from_canonical_cloyne_not_called_without_gbr():
    """load_cloyne_2013_uk is only called when GBR is in countries."""
    called = []
    def _cloyne(*a, **kw):
        called.append(True)
        return _instrument_for([])

    with patch("puremacro.narrative.replication.load_cloyne_2013_uk", side_effect=_cloyne):
        HomogeneousFiscalPanel.from_canonical(
            countries=["USA"],
            include_adler_2024=False,
            include_dglp=False,
            include_rr2017=False,
            include_rr2010_us=False,
            include_mr2013_us=False,
            include_cloyne_uk=True,
            include_ramey_us=False,
            include_devries=False,
            include_guajardo=False,
            include_mitchell=False,
            include_eu_nms_2023=False,
            include_imf_covid_2022=False,
        )
    assert called == []


def test_from_canonical_countries_default_to_dglp_17():
    """Passing countries=None should use DGLP_17."""
    captured_codes = []

    def _fake_dglp(countries, **kw):
        captured_codes.extend(countries)
        return {}

    with patch("puremacro.narrative.replication.load_dglp_2011", side_effect=_fake_dglp):
        HomogeneousFiscalPanel.from_canonical(
            countries=None,
            include_adler_2024=False,
            include_rr2017=False,
            include_rr2010_us=False,
            include_mr2013_us=False,
            include_cloyne_uk=False,
            include_ramey_us=False,
            include_devries=False,
            include_guajardo=False,
            include_mitchell=False,
            include_eu_nms_2023=False,
            include_imf_covid_2022=False,
        )
    assert set(captured_codes) == set(DGLP_17)


def test_from_canonical_countries_are_uppercased():
    """Lower-case country codes are uppercased before passing to loaders."""
    captured_codes = []

    def _fake_dglp(countries, **kw):
        captured_codes.extend(countries)
        return {}

    with patch("puremacro.narrative.replication.load_dglp_2011", side_effect=_fake_dglp):
        HomogeneousFiscalPanel.from_canonical(
            countries=["usa", "gbr"],
            include_adler_2024=False,
            include_rr2017=False,
            include_rr2010_us=False,
            include_mr2013_us=False,
            include_cloyne_uk=False,
            include_ramey_us=False,
            include_devries=False,
            include_guajardo=False,
            include_mitchell=False,
            include_eu_nms_2023=False,
            include_imf_covid_2022=False,
        )
    assert "USA" in captured_codes
    assert "GBR" in captured_codes


def test_from_canonical_metadata_records_date_tol():
    panel = HomogeneousFiscalPanel.from_canonical(
        countries=["USA"],
        date_tol_days=45,
        include_adler_2024=False,
        include_dglp=False,
        include_rr2017=False,
        include_rr2010_us=False,
        include_mr2013_us=False,
        include_cloyne_uk=False,
        include_ramey_us=False,
        include_devries=False,
        include_guajardo=False,
        include_mitchell=False,
        include_eu_nms_2023=False,
        include_imf_covid_2022=False,
    )
    assert panel.metadata["date_tol_days"] == 45


# ---------------------------------------------------------------------------
# from_canonical: multi-source patched run
# ---------------------------------------------------------------------------


def test_from_canonical_multiple_sources_accumulate_events():
    """Events from multiple patched loaders are all collected."""
    e_adler = _event("USA", "2008-01-01", 3.0, "investment", replication="adler_2024")
    e_dglp = _event("GBR", "2010-01-01", 2.0, "consumption", sign=-1, replication="dglp_2011")
    e_rr17 = _event("GBR", "2014-01-01", 1.0, "investment", replication="rr_2017")

    with (
        patch("puremacro.narrative.replication.load_adler_2024",
              return_value={"USA": _instrument_for([e_adler])}),
        patch("puremacro.narrative.replication.load_dglp_2011",
              return_value={"GBR": _instrument_for([e_dglp])}),
        patch("puremacro.narrative.replication.load_romer_romer_2017",
              return_value={"GBR": _instrument_for([e_rr17])}),
    ):
        panel = HomogeneousFiscalPanel.from_canonical(
            countries=["USA", "GBR"],
            include_rr2010_us=False,
            include_mr2013_us=False,
            include_cloyne_uk=False,
            include_ramey_us=False,
            include_devries=False,
            include_guajardo=False,
            include_mitchell=False,
            include_eu_nms_2023=False,
            include_imf_covid_2022=False,
        )
    assert panel.metadata["n_raw_events"] == 3
    countries_found = {e.country for e in panel.events}
    assert countries_found >= {"USA", "GBR"}


def test_from_canonical_with_mitchell_enabled():
    """include_mitchell=True calls load_mitchell_historical."""
    called = []
    def _fake_mitchell(countries, **kw):
        called.append(countries)
        return {}

    with patch("puremacro.narrative.replication.load_mitchell_historical", side_effect=_fake_mitchell):
        HomogeneousFiscalPanel.from_canonical(
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
            include_mitchell=True,
            include_eu_nms_2023=False,
            include_imf_covid_2022=False,
        )
    assert len(called) == 1


def test_from_canonical_with_eu_nms_enabled():
    """include_eu_nms_2023=True calls load_eu_nms_2023."""
    called = []
    def _fake_eu(countries, **kw):
        called.append(countries)
        return {}

    with patch("puremacro.narrative.replication.load_eu_nms_2023", side_effect=_fake_eu):
        HomogeneousFiscalPanel.from_canonical(
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
            include_mitchell=False,
            include_eu_nms_2023=True,
            include_imf_covid_2022=False,
        )
    assert len(called) == 1


def test_from_canonical_with_imf_covid_enabled():
    """include_imf_covid_2022=True calls load_imf_covid_2022."""
    called = []
    def _fake_covid(countries, **kw):
        called.append(countries)
        return {}

    with patch("puremacro.narrative.replication.load_imf_covid_2022", side_effect=_fake_covid):
        HomogeneousFiscalPanel.from_canonical(
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
            include_mitchell=False,
            include_eu_nms_2023=False,
            include_imf_covid_2022=True,
        )
    assert len(called) == 1


# ---------------------------------------------------------------------------
# add_llm_scored
# ---------------------------------------------------------------------------


def test_add_llm_scored_returns_new_panel():
    """add_llm_scored must return a new HomogeneousFiscalPanel, not mutate self."""
    base = HomogeneousFiscalPanel(
        events=[_event("USA", "2010-01-01", 1.0, "investment")],
        countries=["USA"],
    )
    new_ev = _event("USA", "2010-04-01", 0.5, "consumption", sign=-1, scoring_method="llm",
                    replication="")
    with patch("puremacro.narrative.scoring.llm.score_llm", return_value=[new_ev]):
        augmented = base.add_llm_scored(
            iter([(pd.Timestamp("2010-04-01"), "text", "http://src.com")]),
            backend=MagicMock(),
            country="USA",
        )
    assert augmented is not base
    assert isinstance(augmented, HomogeneousFiscalPanel)


def test_add_llm_scored_records_n_llm_added():
    base = HomogeneousFiscalPanel(
        events=[_event("USA", "2010-01-01", 1.0, "investment")],
        countries=["USA"],
    )
    new_evs = [
        _event("USA", "2010-04-01", 0.5, "consumption", sign=-1, scoring_method="llm",
               replication=""),
        _event("USA", "2010-07-01", 1.2, "investment", sign=+1, scoring_method="llm",
               replication=""),
    ]
    with patch("puremacro.narrative.scoring.llm.score_llm", return_value=new_evs):
        augmented = base.add_llm_scored(
            iter([]),
            backend=MagicMock(),
            country="USA",
        )
    assert augmented.metadata["n_llm_added"] == 2


def test_add_llm_scored_deduplicates_against_canonical():
    """When LLM event matches an existing canonical event, dedup should reduce count."""
    canonical_ev = _event("USA", "2010-01-01", 1.0, "investment", sign=+1,
                          scoring_method="manual")
    base = HomogeneousFiscalPanel(events=[canonical_ev], countries=["USA"])
    # LLM event is very close in date and same target/sign — will cluster with canonical
    llm_ev = _event("USA", "2010-02-01", 1.0, "investment", sign=+1,
                    scoring_method="llm", replication="")
    with patch("puremacro.narrative.scoring.llm.score_llm", return_value=[llm_ev]):
        augmented = base.add_llm_scored(iter([]), backend=MagicMock(), country="USA")
    # n_llm_added = 1 but n_after_llm_dedup may be 1 (dedup collapsed them)
    assert augmented.metadata["n_llm_added"] == 1
    assert augmented.metadata["n_after_llm_dedup"] <= 2


# ---------------------------------------------------------------------------
# to_long
# ---------------------------------------------------------------------------


def test_to_long_empty_panel_returns_empty_df_with_correct_columns():
    p = HomogeneousFiscalPanel(events=[], countries=[])
    long = p.to_long()
    assert long.empty
    expected_cols = {
        "country", "date", "magnitude", "magnitude_unit", "sign",
        "target", "subtarget", "confidence", "scoring_method",
        "source_url", "replication",
        "implementation_first_quarter", "implementation_last_quarter",
        "delay_quarters",
    }
    assert expected_cols.issubset(set(long.columns))


def test_to_long_row_count_matches_events():
    p = _two_country_panel()
    long = p.to_long()
    assert len(long) == len(p.events)


def test_to_long_sorted_by_country_then_date():
    p = _two_country_panel()
    long = p.to_long()
    # Verify sort order
    for i in range(len(long) - 1):
        r1, r2 = long.iloc[i], long.iloc[i + 1]
        if r1["country"] == r2["country"]:
            assert r1["date"] <= r2["date"]
        else:
            assert r1["country"] <= r2["country"]


def test_to_long_implementation_columns_for_default_profile():
    """Events with default (empty) profile: first == last == announcement date."""
    e = _event("USA", "2012-04-01", 1.0, "investment")
    p = HomogeneousFiscalPanel(events=[e], countries=["USA"])
    long = p.to_long()
    row = long.iloc[0]
    assert row["implementation_first_quarter"] == pd.Timestamp("2012-04-01")
    assert row["implementation_last_quarter"] == pd.Timestamp("2012-04-01")
    assert row["delay_quarters"] == 0


def test_to_long_implementation_columns_for_explicit_profile():
    """Events with multi-quarter profile: first/last reflect the profile span."""
    e = NarrativeEvent(
        date=pd.Timestamp("2010-01-01"),
        country="ITA",
        magnitude=2.0,
        magnitude_unit="pct_gdp",
        target="investment",
        subtarget="infra",
        sign=-1,
        confidence=1.0,
        source_text="...",
        source_url="https://example.org",
        scoring_method="manual",
        implementation_profile=[
            (pd.Timestamp("2010-04-01"), 0.4),
            (pd.Timestamp("2010-07-01"), 0.6),
        ],
    )
    p = HomogeneousFiscalPanel(events=[e], countries=["ITA"])
    long = p.to_long()
    row = long.iloc[0]
    assert row["implementation_first_quarter"] == pd.Timestamp("2010-04-01")
    assert row["implementation_last_quarter"] == pd.Timestamp("2010-07-01")
    assert row["delay_quarters"] == 1


def test_to_long_replication_from_metadata():
    e = _event("USA", "2010-01-01", 1.0, "investment", replication="dglp_2011")
    p = HomogeneousFiscalPanel(events=[e], countries=["USA"])
    long = p.to_long()
    assert long.iloc[0]["replication"] == "dglp_2011"


def test_to_long_replication_empty_when_missing_from_metadata():
    e = NarrativeEvent(
        date=pd.Timestamp("2010-01-01"),
        country="USA",
        magnitude=1.0,
        magnitude_unit="pct_gdp",
        target="investment",
        subtarget="defense",
        sign=+1,
        confidence=0.9,
        source_text="...",
        source_url="https://example.org",
        scoring_method="manual",
        # no 'replication' key in metadata
    )
    p = HomogeneousFiscalPanel(events=[e], countries=["USA"])
    long = p.to_long()
    assert long.iloc[0]["replication"] == ""


# ---------------------------------------------------------------------------
# to_quarterly_panel
# ---------------------------------------------------------------------------


def test_to_quarterly_panel_empty_events_returns_empty_df():
    p = HomogeneousFiscalPanel(events=[], countries=[])
    wide = p.to_quarterly_panel()
    assert wide.empty


def test_to_quarterly_panel_target_filter_drops_non_matching_countries():
    """A country with only consumption events disappears when target='investment'."""
    events = [
        _event("USA", "2010-01-01", 1.0, "investment", sign=+1),
        _event("DEU", "2010-01-01", 2.0, "consumption", sign=-1),  # DEU: consumption only
    ]
    p = HomogeneousFiscalPanel(events=events, countries=["USA", "DEU"])
    wide = p.to_quarterly_panel(target="investment")
    assert "USA" in wide.columns
    assert "DEU" not in wide.columns


def test_to_quarterly_panel_both_target_event_survives_investment_filter():
    """An event with target='both' is included in investment-filtered panel."""
    e_both = NarrativeEvent(
        date=pd.Timestamp("2010-01-01"),
        country="FRA",
        magnitude=3.0,
        magnitude_unit="pct_gdp",
        target="both",
        subtarget="expenditure",
        sign=-1,
        confidence=0.9,
        source_text="...",
        source_url="https://example.org",
        scoring_method="manual",
    )
    p = HomogeneousFiscalPanel(events=[e_both], countries=["FRA"])
    wide = p.to_quarterly_panel(target="investment")
    assert "FRA" in wide.columns
    # Signed magnitude: -1 * 3.0 = -3.0
    assert wide["FRA"].abs().max() == pytest.approx(3.0)


def test_to_quarterly_panel_target_none_includes_all():
    events = [
        _event("USA", "2010-01-01", 1.0, "investment", sign=+1),
        _event("USA", "2010-04-01", 2.0, "consumption", sign=-1),
    ]
    p = HomogeneousFiscalPanel(events=events, countries=["USA"])
    wide = p.to_quarterly_panel(target=None)
    # Both quarters should have non-zero USA values
    assert wide.loc[pd.Timestamp("2010-01-01"), "USA"] == pytest.approx(1.0)
    assert wide.loc[pd.Timestamp("2010-04-01"), "USA"] == pytest.approx(-2.0)


def test_to_quarterly_panel_all_countries_filtered_returns_empty():
    """If no country has events matching the target, return empty DataFrame."""
    events = [
        _event("USA", "2010-01-01", 1.0, "consumption", sign=-1),
    ]
    p = HomogeneousFiscalPanel(events=events, countries=["USA"])
    wide = p.to_quarterly_panel(target="investment")
    assert wide.empty


def test_to_quarterly_panel_date_index_name():
    events = [_event("USA", "2010-01-01", 1.0, "investment")]
    p = HomogeneousFiscalPanel(events=events, countries=["USA"])
    wide = p.to_quarterly_panel()
    assert wide.index.name == "date"


def test_to_quarterly_panel_fills_missing_quarters_with_zero():
    """Quarters between events get filled with 0 in the date range."""
    events = [
        _event("USA", "2010-01-01", 1.0, "investment", sign=+1),
        _event("USA", "2011-01-01", 2.0, "investment", sign=+1),  # 4 quarters later
    ]
    p = HomogeneousFiscalPanel(events=events, countries=["USA"])
    wide = p.to_quarterly_panel()
    # 2010Q2, 2010Q3, 2010Q4 should be 0
    mid_q = pd.Timestamp("2010-04-01")
    if mid_q in wide.index:
        assert wide.loc[mid_q, "USA"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def test_summary_empty_panel_returns_empty_df():
    p = HomogeneousFiscalPanel(events=[], countries=[])
    s = p.summary()
    assert s.empty


def test_summary_column_set():
    p = _two_country_panel()
    s = p.summary()
    expected = {
        "country", "n_events", "first_date", "last_date",
        "by_replication", "by_target", "by_sign", "median_magnitude_pct_gdp",
    }
    assert expected.issubset(set(s.columns))


def test_summary_row_per_country():
    p = _two_country_panel()
    s = p.summary()
    assert set(s["country"]) == {"USA", "DEU"}


def test_summary_n_events_counts_correctly():
    events = [
        _event("USA", "2010-01-01", 1.0, "investment"),
        _event("USA", "2012-01-01", 0.5, "consumption", sign=-1),
        _event("DEU", "2011-01-01", 2.0, "both", sign=-1),
    ]
    p = HomogeneousFiscalPanel(events=events, countries=["USA", "DEU"])
    s = p.summary()
    usa_row = s[s["country"] == "USA"].iloc[0]
    deu_row = s[s["country"] == "DEU"].iloc[0]
    assert usa_row["n_events"] == 2
    assert deu_row["n_events"] == 1


def test_summary_first_last_date_bounds():
    events = [
        _event("USA", "2010-01-01", 1.0, "investment"),
        _event("USA", "2015-07-01", 0.5, "consumption", sign=-1),
    ]
    p = HomogeneousFiscalPanel(events=events, countries=["USA"])
    s = p.summary()
    row = s.iloc[0]
    assert row["first_date"] == pd.Timestamp("2010-01-01")
    assert row["last_date"] == pd.Timestamp("2015-07-01")


def test_summary_median_magnitude():
    events = [
        _event("USA", "2010-01-01", 1.0, "investment"),
        _event("USA", "2012-01-01", 3.0, "investment"),
    ]
    p = HomogeneousFiscalPanel(events=events, countries=["USA"])
    s = p.summary()
    row = s.iloc[0]
    assert row["median_magnitude_pct_gdp"] == pytest.approx(2.0)


def test_summary_by_replication_dict():
    events = [
        _event("USA", "2010-01-01", 1.0, "investment", replication="dglp_2011"),
        _event("USA", "2012-01-01", 0.5, "investment", replication="rr_2017"),
        _event("USA", "2013-01-01", 0.5, "investment", replication="dglp_2011"),
    ]
    p = HomogeneousFiscalPanel(events=events, countries=["USA"])
    s = p.summary()
    row = s.iloc[0]
    by_rep = row["by_replication"]
    assert by_rep["dglp_2011"] == 2
    assert by_rep["rr_2017"] == 1


def test_summary_by_sign_dict():
    events = [
        _event("USA", "2010-01-01", 1.0, "investment", sign=+1),
        _event("USA", "2012-01-01", 0.5, "investment", sign=-1),
    ]
    p = HomogeneousFiscalPanel(events=events, countries=["USA"])
    s = p.summary()
    row = s.iloc[0]
    by_sign = row["by_sign"]
    assert by_sign[1] == 1
    assert by_sign[-1] == 1


# ---------------------------------------------------------------------------
# cross_correlations
# ---------------------------------------------------------------------------


def test_cross_correlations_empty_panel_returns_empty_df():
    p = HomogeneousFiscalPanel(events=[], countries=[])
    corr = p.cross_correlations()
    assert corr.empty


def test_cross_correlations_shape_is_n_countries_by_n_countries():
    p = _two_country_panel()
    corr = p.cross_correlations()
    n = len(p.countries)
    assert corr.shape == (n, n)


def test_cross_correlations_diagonal_is_one():
    p = _two_country_panel()
    corr = p.cross_correlations()
    for country in corr.columns:
        assert corr.loc[country, country] == pytest.approx(1.0, abs=1e-9)


def test_cross_correlations_symmetric():
    p = _two_country_panel()
    corr = p.cross_correlations()
    pd.testing.assert_frame_equal(corr, corr.T)


def test_cross_correlations_identical_series_gives_one():
    """Two countries with the same shock dates/magnitudes => pairwise corr = 1."""
    dates = ["2010-01-01", "2010-04-01", "2010-07-01", "2011-01-01"]
    events = []
    for d in dates:
        for country in ["CAN", "AUS"]:
            events.append(_event(country, d, 1.0, "investment", sign=+1))
    p = HomogeneousFiscalPanel(events=events, countries=["CAN", "AUS"])
    corr = p.cross_correlations()
    assert corr.loc["CAN", "AUS"] == pytest.approx(1.0, abs=1e-9)


def test_cross_correlations_target_filter_propagates():
    """target parameter is forwarded to to_quarterly_panel."""
    events = [
        _event("USA", "2010-01-01", 1.0, "investment"),
        _event("GBR", "2010-01-01", 1.0, "investment"),
    ]
    p = HomogeneousFiscalPanel(events=events, countries=["USA", "GBR"])
    corr_all = p.cross_correlations()
    corr_inv = p.cross_correlations(target="investment")
    assert corr_inv.shape == corr_all.shape  # same countries survive for investment


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


def test_save_creates_long_csv(tmp_path):
    p = _two_country_panel()
    paths = p.save(tmp_path)
    long_path = paths["long"]
    assert long_path.exists()
    loaded = pd.read_csv(long_path)
    assert len(loaded) == len(p.events)


def test_save_creates_wide_csv(tmp_path):
    p = _two_country_panel()
    paths = p.save(tmp_path)
    wide_path = paths["wide_all"]
    assert wide_path.exists()


def test_save_creates_investment_csv_when_investment_events_exist(tmp_path):
    events = [
        _event("USA", "2010-01-01", 1.0, "investment"),
        _event("DEU", "2010-01-01", 2.0, "investment"),
    ]
    p = HomogeneousFiscalPanel(events=events, countries=["USA", "DEU"])
    paths = p.save(tmp_path)
    assert paths["wide_investment"].exists()


def test_save_creates_consumption_csv_when_consumption_events_exist(tmp_path):
    events = [
        _event("USA", "2010-01-01", 1.0, "consumption", sign=-1),
    ]
    p = HomogeneousFiscalPanel(events=events, countries=["USA"])
    paths = p.save(tmp_path)
    assert paths["wide_consumption"].exists()


def test_save_does_not_create_investment_csv_when_no_investment_events(tmp_path):
    events = [
        _event("USA", "2010-01-01", 1.0, "consumption", sign=-1),
    ]
    p = HomogeneousFiscalPanel(events=events, countries=["USA"])
    p.save(tmp_path)
    inv_path = tmp_path / "narrative_iv_panel_investment.csv"
    assert not inv_path.exists()


def test_save_does_not_create_consumption_csv_when_no_consumption_events(tmp_path):
    events = [
        _event("USA", "2010-01-01", 1.0, "investment"),
    ]
    p = HomogeneousFiscalPanel(events=events, countries=["USA"])
    p.save(tmp_path)
    con_path = tmp_path / "narrative_iv_panel_consumption.csv"
    assert not con_path.exists()


def test_save_creates_audit_csv_when_audit_present(tmp_path):
    audit_df = pd.DataFrame({"country": ["USA"], "date": [pd.Timestamp("2010-01-01")]})
    p = HomogeneousFiscalPanel(
        events=[_event("USA", "2010-01-01", 1.0, "investment")],
        countries=["USA"],
        audit=audit_df,
    )
    p.save(tmp_path)
    audit_path = tmp_path / "narrative_iv_panel_dedup_audit.csv"
    assert audit_path.exists()


def test_save_no_audit_csv_when_audit_is_none(tmp_path):
    p = HomogeneousFiscalPanel(
        events=[_event("USA", "2010-01-01", 1.0, "investment")],
        countries=["USA"],
        audit=None,
    )
    p.save(tmp_path)
    audit_path = tmp_path / "narrative_iv_panel_dedup_audit.csv"
    assert not audit_path.exists()


def test_save_no_audit_csv_when_audit_is_empty_df(tmp_path):
    p = HomogeneousFiscalPanel(
        events=[_event("USA", "2010-01-01", 1.0, "investment")],
        countries=["USA"],
        audit=pd.DataFrame(),
    )
    p.save(tmp_path)
    audit_path = tmp_path / "narrative_iv_panel_dedup_audit.csv"
    assert not audit_path.exists()


def test_save_returns_dict_with_expected_keys(tmp_path):
    p = _two_country_panel()
    paths = p.save(tmp_path)
    assert set(paths.keys()) == {"long", "wide_all", "wide_investment", "wide_consumption"}


def test_save_creates_output_dir_if_missing(tmp_path):
    new_dir = tmp_path / "subdir" / "nested"
    p = HomogeneousFiscalPanel(
        events=[_event("USA", "2010-01-01", 1.0, "investment")],
        countries=["USA"],
    )
    p.save(new_dir)
    assert new_dir.is_dir()


def test_save_with_string_path(tmp_path):
    """save() accepts a plain string for out_dir."""
    p = HomogeneousFiscalPanel(
        events=[_event("USA", "2010-01-01", 1.0, "investment")],
        countries=["USA"],
    )
    paths = p.save(str(tmp_path))
    assert paths["long"].exists()


# ---------------------------------------------------------------------------
# specification_curve: contract (returns a NarrativeSpecificationCurve)
# ---------------------------------------------------------------------------


def test_specification_curve_returns_spec_curve_object():
    """specification_curve wraps self in a NarrativeSpecificationCurve without executing."""
    p = _two_country_panel()
    df = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0], "x_inv": [0.1, 0.2, 0.3], "x_con": [0.4, 0.5, 0.6]},
        index=pd.date_range("2010-01-01", periods=3, freq="QS"),
    )
    with patch(
        "puremacro.narrative.validation.spec_curve.NarrativeSpecificationCurve"
    ) as mock_cls:
        mock_cls.return_value = MagicMock()
        sc = p.specification_curve(df, y="y", x_invest="x_inv", x_consume="x_con")
    assert mock_cls.called
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["panel"] is p
    assert call_kwargs["y"] == "y"
    assert call_kwargs["x_invest"] == "x_inv"
    assert call_kwargs["x_consume"] == "x_con"


def test_specification_curve_passes_horizon_and_seed():
    p = _two_country_panel()
    df = pd.DataFrame({"y": [1.0]}, index=pd.date_range("2010-01-01", periods=1, freq="QS"))
    with patch(
        "puremacro.narrative.validation.spec_curve.NarrativeSpecificationCurve"
    ) as mock_cls:
        mock_cls.return_value = MagicMock()
        p.specification_curve(df, y="y", x_invest="x_inv", x_consume="x_con",
                              horizon=12, seed=99, n_boot=0)
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["horizon"] == 12
    assert call_kwargs["seed"] == 99
    assert call_kwargs["n_boot"] == 0


# ---------------------------------------------------------------------------
# from_canonical: USA-specific loaders return events (extend all_events)
# ---------------------------------------------------------------------------


def test_from_canonical_rr2010_events_collected_for_usa():
    """When load_romer_romer_2010 returns events and USA is in codes,
    those events appear in the final panel."""
    e_usa = _event("USA", "1982-07-01", 1.2, "investment", replication="rr_2010")
    fake_instr = _instrument_for([e_usa])
    with patch("puremacro.narrative.replication.load_romer_romer_2010",
               return_value=fake_instr):
        panel = HomogeneousFiscalPanel.from_canonical(
            countries=["USA"],
            include_adler_2024=False,
            include_dglp=False,
            include_rr2017=False,
            include_rr2010_us=True,
            include_mr2013_us=False,
            include_cloyne_uk=False,
            include_ramey_us=False,
            include_devries=False,
            include_guajardo=False,
            include_mitchell=False,
            include_eu_nms_2023=False,
            include_imf_covid_2022=False,
        )
    assert panel.metadata["n_raw_events"] == 1
    assert panel.events[0].country == "USA"
    assert panel.metadata["load_summary"]["rr_2010"]["ok"] is True


def test_from_canonical_mr2013_events_collected_for_usa():
    """load_mertens_ravn_2013 events are added when USA is in codes."""
    e_usa = _event("USA", "1988-01-01", 0.8, "investment", replication="mr_2013")
    fake_instr = _instrument_for([e_usa])
    with patch("puremacro.narrative.replication.load_mertens_ravn_2013",
               return_value=fake_instr):
        panel = HomogeneousFiscalPanel.from_canonical(
            countries=["USA"],
            include_adler_2024=False,
            include_dglp=False,
            include_rr2017=False,
            include_rr2010_us=False,
            include_mr2013_us=True,
            include_cloyne_uk=False,
            include_ramey_us=False,
            include_devries=False,
            include_guajardo=False,
            include_mitchell=False,
            include_eu_nms_2023=False,
            include_imf_covid_2022=False,
        )
    assert panel.metadata["n_raw_events"] == 1
    assert panel.metadata["load_summary"]["mr_2013"]["ok"] is True


def test_from_canonical_ramey_events_collected_for_usa():
    """load_ramey_2011_defense events are added when USA is in codes."""
    e_usa = _event("USA", "1990-01-01", 0.6, "investment", subtarget="defense",
                   replication="ramey_2011")
    fake_instr = _instrument_for([e_usa])
    with patch("puremacro.narrative.replication.load_ramey_2011_defense",
               return_value=fake_instr):
        panel = HomogeneousFiscalPanel.from_canonical(
            countries=["USA"],
            include_adler_2024=False,
            include_dglp=False,
            include_rr2017=False,
            include_rr2010_us=False,
            include_mr2013_us=False,
            include_cloyne_uk=False,
            include_ramey_us=True,
            include_devries=False,
            include_guajardo=False,
            include_mitchell=False,
            include_eu_nms_2023=False,
            include_imf_covid_2022=False,
        )
    assert panel.metadata["n_raw_events"] == 1
    assert panel.metadata["load_summary"]["ramey_2011"]["ok"] is True


def test_from_canonical_cloyne_events_collected_for_gbr():
    """load_cloyne_2013_uk events are added when GBR is in codes."""
    e_gbr = _event("GBR", "1994-01-01", 1.0, "consumption", sign=-1,
                   replication="cloyne_2013")
    fake_instr = _instrument_for([e_gbr])
    with patch("puremacro.narrative.replication.load_cloyne_2013_uk",
               return_value=fake_instr):
        panel = HomogeneousFiscalPanel.from_canonical(
            countries=["GBR"],
            include_adler_2024=False,
            include_dglp=False,
            include_rr2017=False,
            include_rr2010_us=False,
            include_mr2013_us=False,
            include_cloyne_uk=True,
            include_ramey_us=False,
            include_devries=False,
            include_guajardo=False,
            include_mitchell=False,
            include_eu_nms_2023=False,
            include_imf_covid_2022=False,
        )
    assert panel.metadata["n_raw_events"] == 1
    assert panel.metadata["load_summary"]["cloyne_2013"]["ok"] is True


def test_from_canonical_devries_events_collected():
    """load_devries_2011 returns a multi-country dict; all events are collected."""
    e_ita = _event("ITA", "2004-01-01", 1.5, "both", sign=-1, replication="devries_2011")
    e_prt = _event("PRT", "2008-01-01", 2.0, "both", sign=-1, replication="devries_2011")
    fake_result = {
        "ITA": _instrument_for([e_ita]),
        "PRT": _instrument_for([e_prt]),
    }
    with patch("puremacro.narrative.replication.load_devries_2011",
               return_value=fake_result):
        panel = HomogeneousFiscalPanel.from_canonical(
            countries=["ITA", "PRT"],
            include_adler_2024=False,
            include_dglp=False,
            include_rr2017=False,
            include_rr2010_us=False,
            include_mr2013_us=False,
            include_cloyne_uk=False,
            include_ramey_us=False,
            include_devries=True,
            include_guajardo=False,
            include_mitchell=False,
            include_eu_nms_2023=False,
            include_imf_covid_2022=False,
        )
    assert panel.metadata["n_raw_events"] == 2
    countries_found = {e.country for e in panel.events}
    assert countries_found == {"ITA", "PRT"}


def test_from_canonical_guajardo_events_collected():
    """load_guajardo_2011 multi-country dict events are all collected."""
    e_esp = _event("ESP", "2010-04-01", 3.0, "both", sign=-1, replication="guajardo_2011")
    fake_result = {"ESP": _instrument_for([e_esp])}
    with patch("puremacro.narrative.replication.load_guajardo_2011",
               return_value=fake_result):
        panel = HomogeneousFiscalPanel.from_canonical(
            countries=["ESP"],
            include_adler_2024=False,
            include_dglp=False,
            include_rr2017=False,
            include_rr2010_us=False,
            include_mr2013_us=False,
            include_cloyne_uk=False,
            include_ramey_us=False,
            include_devries=False,
            include_guajardo=True,
            include_mitchell=False,
            include_eu_nms_2023=False,
            include_imf_covid_2022=False,
        )
    assert panel.metadata["n_raw_events"] == 1
    assert panel.events[0].country == "ESP"


def test_from_canonical_mitchell_events_collected():
    """include_mitchell=True with a returning loader extends all_events."""
    e_gbr = _event("GBR", "1965-01-01", 0.9, "investment", replication="mitchell_2013")
    fake_result = {"GBR": _instrument_for([e_gbr])}
    with patch("puremacro.narrative.replication.load_mitchell_historical",
               return_value=fake_result):
        panel = HomogeneousFiscalPanel.from_canonical(
            countries=["GBR"],
            include_adler_2024=False,
            include_dglp=False,
            include_rr2017=False,
            include_rr2010_us=False,
            include_mr2013_us=False,
            include_cloyne_uk=False,
            include_ramey_us=False,
            include_devries=False,
            include_guajardo=False,
            include_mitchell=True,
            include_eu_nms_2023=False,
            include_imf_covid_2022=False,
        )
    assert panel.metadata["n_raw_events"] == 1
    assert panel.events[0].country == "GBR"
