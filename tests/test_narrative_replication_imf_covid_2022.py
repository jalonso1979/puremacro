"""Tests for the IMF COVID 2022 (WP/22/114) announcement-level fiscal loader."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


_FIXTURE = (
    Path(__file__).parent / "fixtures" / "narrative"
    / "synthetic_imf_covid_2022.csv"
)


def _read_fixture() -> pd.DataFrame:
    return pd.read_csv(_FIXTURE)


def test_imf_covid_loader_filters_to_fiscal_only():
    """5 fiscal + 2 non-fiscal rows in fixture → 5 fiscal events.
    USA 2020-04-15 + 2020-05-01 unemployment_benefits aggregate to one
    2020Q2 event (within-quarter aggregation), so total = 4 events."""
    from puremacro.narrative.replication.imf_covid_2022_announcements import (
        imf_covid_2022_csv_to_events,
    )
    events = imf_covid_2022_csv_to_events(_read_fixture())
    # USA unemployment_benefits Q2 (aggregated) + USA health_spending Q2
    # + DEU wage_subsidies Q1 + BRA cash_transfers Q2 = 4 events.
    assert len(events) == 4
    countries = {e.country for e in events}
    assert countries == {"USA", "DEU", "BRA"}


def test_imf_covid_loader_include_non_fiscal_param():
    """include_non_fiscal=True → emits monetary + prudential rows too."""
    from puremacro.narrative.replication.imf_covid_2022_announcements import (
        imf_covid_2022_csv_to_events,
    )
    events = imf_covid_2022_csv_to_events(
        _read_fixture(), include_non_fiscal=True,
    )
    # 4 fiscal-aggregated + 1 monetary (USA rate cut) + 1 prudential
    # (DEU capital relief) = 6 events.
    assert len(events) == 6


def test_imf_covid_loader_aggregates_same_quarter_same_category():
    """Two USA unemployment_benefits rows in 2020Q2 (April 15 + May 1) →
    one event with summed magnitude 3.5%."""
    from puremacro.narrative.replication.imf_covid_2022_announcements import (
        imf_covid_2022_csv_to_events,
    )
    events = imf_covid_2022_csv_to_events(_read_fixture())
    usa_ui = [
        e for e in events
        if e.country == "USA" and e.subtarget == "transfers"
    ]
    assert len(usa_ui) == 1
    assert usa_ui[0].magnitude == pytest.approx(3.5)


def test_imf_covid_loader_policy_to_subtarget_mapping():
    """unemployment_benefits → transfers; wage_subsidies → wages;
    health_spending → health; cash_transfers → transfers."""
    from puremacro.narrative.replication.imf_covid_2022_announcements import (
        imf_covid_2022_csv_to_events,
    )
    events = imf_covid_2022_csv_to_events(_read_fixture())
    usa_ui = next(
        e for e in events if e.country == "USA" and e.subtarget == "transfers"
    )
    usa_health = next(
        e for e in events if e.country == "USA" and e.subtarget == "health"
    )
    deu_wage = next(e for e in events if e.country == "DEU")
    bra_cash = next(e for e in events if e.country == "BRA")
    assert usa_ui.subtarget == "transfers"
    assert usa_health.subtarget == "health"
    assert deu_wage.subtarget == "wages"
    assert bra_cash.subtarget == "transfers"


def test_imf_covid_loader_sign_convention_expansionary_default():
    """COVID announcements are expansionary by default → sign=+1."""
    from puremacro.narrative.replication.imf_covid_2022_announcements import (
        imf_covid_2022_csv_to_events,
    )
    events = imf_covid_2022_csv_to_events(_read_fixture())
    assert all(e.sign == +1 for e in events)


def test_imf_covid_loader_metadata_includes_replication_marker():
    """Every event has metadata['replication'] == 'imf_covid_2022'."""
    from puremacro.narrative.replication.imf_covid_2022_announcements import (
        imf_covid_2022_csv_to_events,
    )
    events = imf_covid_2022_csv_to_events(_read_fixture())
    assert all(
        e.metadata.get("replication") == "imf_covid_2022" for e in events
    )


def test_imf_covid_loader_iso2_normalized_to_iso3():
    """ISO2 codes (e.g., 'US') get normalized to ISO3 ('USA')."""
    from puremacro.narrative.replication.imf_covid_2022_announcements import (
        imf_covid_2022_csv_to_events,
    )
    df = pd.DataFrame([
        {"country": "US", "date": "2020-04-15",
         "policy": "unemployment_benefits", "policy_class": "fiscal",
         "magnitude_pct_gdp": 1.0, "description": "test"},
    ])
    events = imf_covid_2022_csv_to_events(df)
    assert len(events) == 1
    assert events[0].country == "USA"


def test_imf_covid_loader_load_with_csv_path_and_country_filter():
    """`load(csv_path=, countries=['USA'])` returns USA-only dict."""
    from puremacro.narrative.replication.imf_covid_2022_announcements import (
        load,
    )
    out = load(csv_path=str(_FIXTURE), countries=["USA"])
    assert set(out.keys()) == {"USA"}


def test_imf_covid_loader_unknown_policy_falls_back_to_general():
    """Unknown fiscal policy_category strings → subtarget='general'."""
    from puremacro.narrative.replication.imf_covid_2022_announcements import (
        imf_covid_2022_csv_to_events,
    )
    df = pd.DataFrame([
        {"country": "USA", "date": "2020-06-01",
         "policy": "obscure_widget_handout", "policy_class": "fiscal",
         "magnitude_pct_gdp": 0.5, "description": "test"},
    ])
    events = imf_covid_2022_csv_to_events(df)
    assert len(events) == 1
    assert events[0].subtarget == "general"
