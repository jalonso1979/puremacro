"""Tests for within-year quarterization helper and the loader profile
backfills (DGLP / Devries / Guajardo / RR2010)."""
from __future__ import annotations

import io

import pandas as pd
import pytest

from puremacro.narrative.replication._within_year import (
    annualize_to_quarters, FY_START_MONTH,
)


def test_uniform_within_year_splits_evenly():
    out = annualize_to_quarters(2010, weight=1.0, rule="uniform")
    dates = [d for d, _ in out]
    weights = [w for _, w in out]
    assert dates == [
        pd.Timestamp("2010-01-01"), pd.Timestamp("2010-04-01"),
        pd.Timestamp("2010-07-01"), pd.Timestamp("2010-10-01"),
    ]
    assert weights == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_uniform_within_year_scales_total_weight():
    out = annualize_to_quarters(2010, weight=0.6, rule="uniform")
    weights = [w for _, w in out]
    assert sum(weights) == pytest.approx(0.6)
    assert weights == pytest.approx([0.15, 0.15, 0.15, 0.15])


def test_front_loaded_profile():
    out = annualize_to_quarters(2010, weight=1.0, rule="front_loaded")
    weights = [w for _, w in out]
    assert weights == pytest.approx([0.40, 0.30, 0.20, 0.10])


def test_fiscal_year_uk_starts_in_april():
    out = annualize_to_quarters(2010, weight=1.0, rule="fiscal_year", country="GBR")
    dates = [d for d, _ in out]
    assert dates[0] == pd.Timestamp("2010-04-01")
    assert dates[-1] == pd.Timestamp("2011-01-01")


def test_fiscal_year_us_federal_starts_in_october():
    out = annualize_to_quarters(2010, weight=1.0, rule="fiscal_year", country="USA")
    dates = [d for d, _ in out]
    assert dates[0] == pd.Timestamp("2010-10-01")
    assert dates[-1] == pd.Timestamp("2011-07-01")


def test_fiscal_year_unknown_country_raises():
    with pytest.raises(KeyError, match="ZZZ"):
        annualize_to_quarters(2010, weight=1.0, rule="fiscal_year", country="ZZZ")


def test_unknown_rule_raises():
    with pytest.raises(ValueError, match="rule"):
        annualize_to_quarters(2010, weight=1.0, rule="bogus")


def test_known_fy_start_months_are_sane():
    # Sanity: a few well-known calendars.
    assert FY_START_MONTH["GBR"] == 4    # UK April-March
    assert FY_START_MONTH["USA"] == 10   # US federal Oct-Sep
    assert FY_START_MONTH["JPN"] == 4    # Japan April-March
    assert FY_START_MONTH["DEU"] == 1    # Germany Jan-Dec


# ---------------------------------------------------------------------------
# DGLP loader: consecutive-year grouping
# ---------------------------------------------------------------------------

from puremacro.narrative.replication.dglp_2011_consolidations import (
    dglp_csv_to_events,
)


def _dglp_csv(rows):
    """Build an in-memory DGLP-style wide CSV. ``rows`` is a list of
    (country, year, tax_pct_gdp, exp_pct_gdp) tuples."""
    df = pd.DataFrame(rows, columns=["country", "year", "tax_pct_gdp",
                                     "exp_pct_gdp"])
    return df


def test_dglp_loader_groups_consecutive_years_into_one_event():
    df = _dglp_csv([
        ("ITA", 1992, 0.0, 1.0),
        ("ITA", 1993, 0.0, 0.5),
        ("ITA", 1994, 0.0, 0.3),
    ])
    events = dglp_csv_to_events(df, flip_sign=True, within_year_rule="uniform")
    # Three consecutive expenditure rows → 1 event with 12-quarter profile.
    italy_exp = [e for e in events if e.country == "ITA" and e.subtarget == "expenditure"]
    assert len(italy_exp) == 1
    e = italy_exp[0]
    assert e.date == pd.Timestamp("1992-01-01")
    assert len(e.implementation_profile) == 12  # 3 years × 4 quarters
    weights = [w for _, w in e.implementation_profile]
    assert sum(weights) == pytest.approx(1.0)
    # Total magnitude is the sum of tranches.
    assert e.magnitude == pytest.approx(1.8)


def test_dglp_loader_breaks_group_on_gap_year():
    df = _dglp_csv([
        ("DEU", 2003, 0.5, 0.0),
        ("DEU", 2004, 0.3, 0.0),
        ("DEU", 2007, 0.2, 0.0),  # gap years 2005-2006
    ])
    events = dglp_csv_to_events(df, flip_sign=True, within_year_rule="uniform")
    deu_tax = [e for e in events if e.country == "DEU" and e.subtarget == "tax"]
    # 2003-04 grouped, 2007 separate → 2 events.
    assert len(deu_tax) == 2
    sorted_evs = sorted(deu_tax, key=lambda e: e.date)
    assert sorted_evs[0].date == pd.Timestamp("2003-01-01")
    assert len(sorted_evs[0].implementation_profile) == 8
    assert sorted_evs[1].date == pd.Timestamp("2007-01-01")
    assert len(sorted_evs[1].implementation_profile) == 4


def test_dglp_loader_zero_rows_skipped():
    df = _dglp_csv([
        ("FRA", 1995, 0.0, 0.0),
        ("FRA", 1996, 0.5, 0.0),
    ])
    events = dglp_csv_to_events(df, flip_sign=True, within_year_rule="uniform")
    # Zero row dropped, 1996 standalone tax event remains.
    fra_tax = [e for e in events if e.country == "FRA" and e.subtarget == "tax"]
    assert len(fra_tax) == 1
    assert fra_tax[0].date == pd.Timestamp("1996-01-01")


def test_dglp_loader_front_loaded_rule_applies():
    df = _dglp_csv([("USA", 2010, 1.0, 0.0)])
    events = dglp_csv_to_events(df, flip_sign=True, within_year_rule="front_loaded")
    usa_tax = [e for e in events if e.country == "USA"]
    assert len(usa_tax) == 1
    weights = [w for _, w in usa_tax[0].implementation_profile]
    assert weights == pytest.approx([0.40, 0.30, 0.20, 0.10])


# ---------------------------------------------------------------------------
# Devries loader: consecutive-year grouping
# ---------------------------------------------------------------------------

from puremacro.narrative.replication.devries_2011_consolidations import (
    devries_csv_to_events,
)


def _devries_csv(rows):
    """Build an in-memory long-format Devries-style CSV. ``rows`` is a
    list of (country, year, action_pct_gdp, type) tuples."""
    df = pd.DataFrame(rows, columns=["country", "year", "action_pct_gdp", "type"])
    return df


def test_devries_loader_groups_consecutive_tax_rows():
    df = _devries_csv([
        ("ESP", 2010, 1.0, "tax"),
        ("ESP", 2011, 0.5, "tax"),
        ("ESP", 2012, 0.3, "tax"),
    ])
    events = devries_csv_to_events(df, within_year_rule="uniform")
    esp_tax = [e for e in events if e.country == "ESP" and e.subtarget == "tax"]
    assert len(esp_tax) == 1
    e = esp_tax[0]
    assert e.date == pd.Timestamp("2010-01-01")
    assert len(e.implementation_profile) == 12
    assert e.magnitude == pytest.approx(1.8)
    assert e.sign == -1   # consolidation = contractionary


def test_devries_loader_separates_tax_from_expenditure():
    df = _devries_csv([
        ("FRA", 2011, 0.4, "tax"),
        ("FRA", 2011, 0.6, "expenditure"),
    ])
    events = devries_csv_to_events(df, within_year_rule="uniform")
    fra_evs = [e for e in events if e.country == "FRA"]
    assert len(fra_evs) == 2
    assert {e.subtarget for e in fra_evs} == {"tax", "expenditure"}
    for ev in fra_evs:
        assert len(ev.implementation_profile) == 4


def test_devries_loader_fiscal_year_rule_uk():
    df = _devries_csv([("GBR", 2010, 1.0, "tax")])
    events = devries_csv_to_events(df, within_year_rule="fiscal_year")
    e = [ev for ev in events if ev.country == "GBR"][0]
    first_q = e.implementation_profile[0][0]
    assert first_q == pd.Timestamp("2010-04-01")


# ---------------------------------------------------------------------------
# Guajardo loader: consecutive-year grouping
# ---------------------------------------------------------------------------

from puremacro.narrative.replication.guajardo_2011_aeipf import (
    guajardo_csv_to_events,
)


def _guajardo_csv(rows):
    df = pd.DataFrame(rows, columns=["country", "year", "action_pct_gdp", "type"])
    return df


def test_guajardo_loader_groups_consecutive_years():
    df = _guajardo_csv([
        ("PRT", 2010, 1.0, "tax"),
        ("PRT", 2011, 0.5, "tax"),
    ])
    events = guajardo_csv_to_events(df, within_year_rule="uniform")
    prt_tax = [e for e in events if e.country == "PRT" and e.subtarget == "tax"]
    assert len(prt_tax) == 1
    e = prt_tax[0]
    assert e.date == pd.Timestamp("2010-01-01")
    assert len(e.implementation_profile) == 8
    assert e.magnitude == pytest.approx(1.5)


def test_guajardo_loader_within_year_rule_default_uniform():
    df = _guajardo_csv([("IRL", 2009, 0.4, "expenditure")])
    events = guajardo_csv_to_events(df)  # no within_year_rule kwarg
    irl_evs = [e for e in events if e.country == "IRL"]
    assert len(irl_evs) == 1
    weights = [w for _, w in irl_evs[0].implementation_profile]
    assert weights == pytest.approx([0.25, 0.25, 0.25, 0.25])


# ---------------------------------------------------------------------------
# Romer-Romer 2010 loader: native effective_date column
# ---------------------------------------------------------------------------

from puremacro.narrative.replication.romer_romer_2010 import (
    romer_romer_2010_csv_to_events,
)


def _rr2010_csv(rows, *, with_effective: bool):
    cols = ["date", "rr_exog"] + (["effective_date"] if with_effective else [])
    df = pd.DataFrame(rows, columns=cols)
    return df


def test_rr2010_loader_uses_effective_date_when_present():
    rows = [
        # Bill signed 1981Q3; ERTA-style effective 1981Q4 (initial bracket cut).
        ("1981Q3", 0.6, "1981-10-01"),
    ]
    df = _rr2010_csv(rows, with_effective=True)
    events = romer_romer_2010_csv_to_events(df)
    assert len(events) == 1
    e = events[0]
    # Announcement quarter remains the bill date.
    assert e.date == pd.Timestamp("1981-07-01")
    # Profile points at effective quarter, not signing quarter.
    assert e.implementation_profile == [(pd.Timestamp("1981-10-01"), 1.0)]
    assert e.delay_quarters == 1


def test_rr2010_loader_default_when_no_effective_date_column():
    rows = [("2003Q2", 0.5)]
    df = _rr2010_csv(rows, with_effective=False)
    events = romer_romer_2010_csv_to_events(df)
    assert len(events) == 1
    e = events[0]
    # No profile attached; effective_profile falls back to single quarter.
    assert e.implementation_profile == []
    assert e.effective_profile == [(pd.Timestamp("2003-04-01"), 1.0)]


def test_rr2010_loader_skips_invalid_effective_date():
    rows = [("2003Q2", 0.5, "")]
    df = _rr2010_csv(rows, with_effective=True)
    events = romer_romer_2010_csv_to_events(df)
    # Empty effective_date string falls back to default.
    assert events[0].implementation_profile == []


# ---------------------------------------------------------------------------
# HomogeneousFiscalPanel: iv_kind forwarding and to_long() columns
# ---------------------------------------------------------------------------

from puremacro.narrative.panel import HomogeneousFiscalPanel
from puremacro.narrative import NarrativeEvent


def _make_panel_from_inline_events():
    e1 = NarrativeEvent(
        date=pd.Timestamp("2010-01-01"), country="ITA", magnitude=1.5,
        magnitude_unit="pct_gdp", target="both", subtarget="expenditure",
        sign=-1, confidence=1.0, source_text="...",
        source_url="https://example.org/x", scoring_method="manual",
        implementation_profile=[
            (pd.Timestamp("2010-01-01"), 0.5),
            (pd.Timestamp("2010-04-01"), 0.5),
        ],
    )
    e2 = NarrativeEvent(
        date=pd.Timestamp("2011-01-01"), country="USA", magnitude=0.5,
        magnitude_unit="pct_gdp", target="consumption", subtarget="tax",
        sign=+1, confidence=1.0, source_text="...",
        source_url="https://example.org/x", scoring_method="manual",
    )
    return HomogeneousFiscalPanel(events=[e1, e2], countries=["ITA", "USA"])


def test_panel_to_quarterly_panel_default_is_implementation():
    p = _make_panel_from_inline_events()
    wide = p.to_quarterly_panel()
    # Italy: signed magnitude -1.5, 50/50 across 2010Q1/Q2.
    assert wide.loc[pd.Timestamp("2010-01-01"), "ITA"] == pytest.approx(-0.75)
    assert wide.loc[pd.Timestamp("2010-04-01"), "ITA"] == pytest.approx(-0.75)


def test_panel_to_quarterly_panel_announcement_kind_puts_full_mass_at_announce():
    p = _make_panel_from_inline_events()
    wide = p.to_quarterly_panel(iv_kind="announcement")
    assert wide.loc[pd.Timestamp("2010-01-01"), "ITA"] == pytest.approx(-1.5)
    # 2010Q2 is absent from the announcement-mode panel (no events land there).
    ita = wide["ITA"].reindex(
        pd.date_range("2010-01-01", "2010-04-01", freq="QS"), fill_value=0.0
    )
    assert ita.loc[pd.Timestamp("2010-04-01")] == pytest.approx(0.0)


def test_panel_to_long_includes_implementation_columns():
    p = _make_panel_from_inline_events()
    long = p.to_long()
    assert "implementation_first_quarter" in long.columns
    assert "implementation_last_quarter" in long.columns
    assert "delay_quarters" in long.columns
    # ITA event: profile spans 2010Q1..Q2.
    ita_row = long[long["country"] == "ITA"].iloc[0]
    assert ita_row["implementation_first_quarter"] == pd.Timestamp("2010-01-01")
    assert ita_row["implementation_last_quarter"] == pd.Timestamp("2010-04-01")
    # USA event: default profile → first==last==date.
    usa_row = long[long["country"] == "USA"].iloc[0]
    assert usa_row["implementation_first_quarter"] == pd.Timestamp("2011-01-01")
    assert usa_row["implementation_last_quarter"] == pd.Timestamp("2011-01-01")
    assert usa_row["delay_quarters"] == 0
