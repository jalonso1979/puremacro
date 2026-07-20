"""Tests for the Adler 2024 (IMF WP/24/210) action-based fiscal-
consolidation loader."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


_FIXTURE = (
    Path(__file__).parent / "fixtures" / "narrative" / "synthetic_adler_2024.csv"
)


def _read_fixture() -> pd.DataFrame:
    return pd.read_csv(_FIXTURE)


def test_adler_loader_groups_consecutive_oecd():
    """USA 1985+1986 tax → 1 event with 8-quarter profile."""
    from puremacro.narrative.replication.adler_2024_consolidations import (
        adler_csv_to_events,
    )
    events = adler_csv_to_events(
        _read_fixture(), flip_sign=True, within_year_rule="uniform",
    )
    usa_tax = [e for e in events if e.country == "USA" and e.subtarget == "tax"]
    # 1985-86 grouped (consecutive), 2007 separate → 2 events.
    assert len(usa_tax) == 2
    sorted_evs = sorted(usa_tax, key=lambda e: e.date)
    assert sorted_evs[0].date == pd.Timestamp("1985-01-01")
    assert len(sorted_evs[0].implementation_profile) == 8  # 2 years × 4 quarters
    assert sorted_evs[0].magnitude == pytest.approx(0.8)
    weights = [w for _, w in sorted_evs[0].implementation_profile]
    assert sum(weights) == pytest.approx(1.0)


def test_adler_loader_groups_consecutive_lac():
    """BRA 2015+2016 tax → 1 event with 8-quarter profile."""
    from puremacro.narrative.replication.adler_2024_consolidations import (
        adler_csv_to_events,
    )
    events = adler_csv_to_events(
        _read_fixture(), flip_sign=True, within_year_rule="uniform",
    )
    bra_tax = [e for e in events if e.country == "BRA" and e.subtarget == "tax"]
    assert len(bra_tax) == 1
    e = bra_tax[0]
    assert e.date == pd.Timestamp("2015-01-01")
    assert len(e.implementation_profile) == 8
    assert e.magnitude == pytest.approx(1.0)


def test_adler_loader_handles_oecd_and_lac_in_same_csv():
    """Combined fixture → events emitted for both regions."""
    from puremacro.narrative.replication.adler_2024_consolidations import (
        adler_csv_to_events,
    )
    events = adler_csv_to_events(
        _read_fixture(), flip_sign=True, within_year_rule="uniform",
    )
    countries = {e.country for e in events}
    assert "USA" in countries
    assert "BRA" in countries
    assert "DEU" in countries


def test_adler_loader_metadata_includes_replication_marker():
    """Every event has metadata['replication'] == 'adler_2024'."""
    from puremacro.narrative.replication.adler_2024_consolidations import (
        adler_csv_to_events,
    )
    events = adler_csv_to_events(
        _read_fixture(), flip_sign=True, within_year_rule="uniform",
    )
    assert all(e.metadata.get("replication") == "adler_2024" for e in events)


def test_adler_loader_sign_convention_is_consolidation_negative():
    """flip_sign=True (default) → consolidations end up sign=-1 (contractionary)."""
    from puremacro.narrative.replication.adler_2024_consolidations import (
        adler_csv_to_events,
    )
    events = adler_csv_to_events(
        _read_fixture(), flip_sign=True, within_year_rule="uniform",
    )
    # Every event in the fixture is a positive consolidation tranche.
    assert all(e.sign == -1 for e in events)


def test_adler_loader_within_year_rule_front_loaded():
    """Front-loaded rule produces [0.40, 0.30, 0.20, 0.10] weights per year."""
    from puremacro.narrative.replication.adler_2024_consolidations import (
        adler_csv_to_events,
    )
    # Single-year fixture row: DEU 1992 expenditure 1.0
    df = pd.DataFrame([
        {"country": "DEU", "year": 1992, "tax_pct_gdp": 0.0, "exp_pct_gdp": 1.0},
    ])
    events = adler_csv_to_events(
        df, flip_sign=True, within_year_rule="front_loaded",
    )
    deu_evs = [e for e in events if e.country == "DEU"]
    assert len(deu_evs) == 1
    weights = [w for _, w in deu_evs[0].implementation_profile]
    assert weights == pytest.approx([0.40, 0.30, 0.20, 0.10])


def test_adler_loader_load_with_csv_path_and_country_filter():
    """`load(csv_path=, countries=['USA'])` returns NarrativeInstrument dict
    keyed by ISO3, only containing USA."""
    from puremacro.narrative.replication.adler_2024_consolidations import load
    out = load(csv_path=str(_FIXTURE), countries=["USA"])
    assert set(out.keys()) == {"USA"}
    inst = out["USA"]
    assert all(e.country == "USA" for e in inst.events)
