"""Offline unit tests for puremacro.fetch.emissions.

Tests World Bank WDI emissions collector, OECD SDMX air GHG emissions collector,
and the unified emissions panel builder using recorded offline fixtures.
Verifies long-form schema compliance, unit conversions (Mt to kt), indicator
and sectoral mappings, aggregate filtering, frequency harmonization (A -> Q),
and resilience against network and parser errors.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch import emissions as em_mod
from puremacro.fetch.emissions import (
    fetch_emissions_panel,
    fetch_oecd_ghg,
    fetch_wdi_emissions,
)

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "emissions"
_WDI_FIXTURE = _FIXTURE_DIR / "wdi_emissions.json"
_WDI_ERROR_FIXTURE = _FIXTURE_DIR / "wdi_error.json"
_OECD_FIXTURE = _FIXTURE_DIR / "oecd_ghg.csv"

_EXPECTED_COLS = ["code", "date", "variable", "value", "sa_source", "source"]


@pytest.fixture
def mock_wdi(monkeypatch):
    """Monkeypatch _cached_get to serve the frozen WDI fixture."""
    fixture_bytes = _WDI_FIXTURE.read_bytes()

    def _mock_get(url, *, refresh=False, timeout=60):
        return fixture_bytes

    monkeypatch.setattr(em_mod, "_cached_get", _mock_get)
    return fixture_bytes


@pytest.fixture
def mock_oecd(monkeypatch):
    """Monkeypatch _get_oecd_csv to serve the frozen OECD SDMX CSV fixture."""
    df_raw = pd.read_csv(_OECD_FIXTURE)

    def _mock_csv(agency_flow, key, start_period, *, refresh=False):
        return df_raw.copy()

    monkeypatch.setattr(em_mod, "_get_oecd_csv", _mock_csv)
    return df_raw


# ============================================================================
# 1. World Bank WDI Emissions Tests
# ============================================================================


def test_wdi_emissions_schema_and_types(mock_wdi):
    """Test schema, column types, and sorting of fetch_wdi_emissions."""
    df = fetch_wdi_emissions()
    assert not df.empty
    assert list(df.columns) == _EXPECTED_COLS
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_float_dtype(df["value"])
    assert df["value"].notna().all()
    assert (df["sa_source"] == "none").all()
    assert df["source"].str.startswith("WorldBank:WDI:").all()


def test_wdi_emissions_filters_aggregates_and_nulls(mock_wdi):
    """Ensure regional aggregates (e.g. WLD) and null observations are dropped."""
    df = fetch_wdi_emissions()
    # WLD is in the fixture but must be excluded by is_country()
    assert "WLD" not in df["code"].values
    assert set(df["code"].unique()).issubset({"USA", "DEU", "GBR"})
    # USA has a null record for 2021 in the fixture which must not appear
    usa_2021 = df[(df["code"] == "USA") & (df["date"] == pd.Timestamp("2021-01-01"))]
    assert usa_2021.empty


def test_wdi_emissions_country_filtering(mock_wdi):
    """Test explicit country code filtering."""
    df_usa = fetch_wdi_emissions(codes=["USA"])
    assert not df_usa.empty
    assert set(df_usa["code"].unique()) == {"USA"}

    df_deu_gbr = fetch_wdi_emissions(codes=["deu", "gbr"])
    assert set(df_deu_gbr["code"].unique()) == {"DEU", "GBR"}

    df_empty = fetch_wdi_emissions(codes=[])
    assert df_empty.empty
    assert list(df_empty.columns) == _EXPECTED_COLS


def test_wdi_emissions_year_filtering(mock_wdi):
    """Test start_year and end_year bounding."""
    df_2020 = fetch_wdi_emissions(start_year=2020, end_year=2020)
    assert not df_2020.empty
    assert (df_2020["date"] == pd.Timestamp("2020-01-01")).all()

    df_pre_2020 = fetch_wdi_emissions(start_year=2018, end_year=2019)
    assert not df_pre_2020.empty
    assert df_pre_2020["date"].max() <= pd.Timestamp("2019-01-01")


def test_wdi_emissions_unit_scaling_and_variables(mock_wdi):
    """Verify conversion of Mt to kt (*1000) for totals and 1.0 for per-capita."""
    df = fetch_wdi_emissions(codes=["USA"], start_year=2020, end_year=2020)
    var_map = dict(zip(df["variable"], df["value"]))

    # Per capita: 13.68 metric tons
    assert "co2_pc_a" in var_map
    assert var_map["co2_pc_a"] == pytest.approx(13.68)

    # CO2 total: 4535.38 Mt -> 4,535,380.0 kt
    assert "co2_total_kt_a" in var_map
    assert var_map["co2_total_kt_a"] == pytest.approx(4535380.0)

    # GHG total: 5711.23 Mt -> 5,711,230.0 kt
    assert "ghg_total_kt_a" in var_map
    assert var_map["ghg_total_kt_a"] == pytest.approx(5711230.0)

    # Methane: 621.45 Mt -> 621,450.0 kt
    assert "methane_kt_a" in var_map
    assert var_map["methane_kt_a"] == pytest.approx(621450.0)

    # Nitrous oxide: 398.12 Mt -> 398,120.0 kt
    assert "nitrous_oxide_kt_a" in var_map
    assert var_map["nitrous_oxide_kt_a"] == pytest.approx(398120.0)


def test_wdi_emissions_legacy_fallback(mock_wdi):
    """Verify legacy codes in fixture (EN.ATM.CO2E.PC, EN.ATM.CO2E.KT) are properly mapped."""
    df_gbr = fetch_wdi_emissions(codes=["GBR"], start_year=2018, end_year=2018)
    var_map = dict(zip(df_gbr["variable"], df_gbr["value"]))

    assert "co2_pc_a" in var_map
    assert var_map["co2_pc_a"] == pytest.approx(5.41)

    assert "co2_total_kt_a" in var_map
    assert var_map["co2_total_kt_a"] == pytest.approx(361540.0)


def test_wdi_emissions_error_and_empty_handling(monkeypatch):
    """Test response handling on API error message, network failure, and malformed JSON."""
    # World Bank API error code 175
    err_bytes = _WDI_ERROR_FIXTURE.read_bytes()
    monkeypatch.setattr(em_mod, "_cached_get", lambda *a, **k: err_bytes)
    df_err = fetch_wdi_emissions()
    assert df_err.empty
    assert list(df_err.columns) == _EXPECTED_COLS

    # Network failure (cached_get returning empty bytes or raising)
    monkeypatch.setattr(em_mod, "_cached_get", lambda *a, **k: b"")
    df_net = fetch_wdi_emissions()
    assert df_net.empty
    assert list(df_net.columns) == _EXPECTED_COLS

    # Malformed bytes
    monkeypatch.setattr(em_mod, "_cached_get", lambda *a, **k: b"not json")
    df_malformed = fetch_wdi_emissions()
    assert df_malformed.empty
    assert list(df_malformed.columns) == _EXPECTED_COLS


# ============================================================================
# 2. OECD SDMX Air Emissions (DF_AIR_GHG) Tests
# ============================================================================


def test_oecd_ghg_schema_and_types(mock_oecd):
    """Test schema, column types, and sorting of fetch_oecd_ghg."""
    df = fetch_oecd_ghg()
    assert not df.empty
    assert list(df.columns) == _EXPECTED_COLS
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_float_dtype(df["value"])
    assert df["value"].notna().all()
    assert (df["sa_source"] == "none").all()
    assert df["source"].str.startswith("OECD:DSD_AIR_GHG@DF_AIR_GHG:").all()


def test_oecd_ghg_sectoral_mapping_and_scaling(mock_oecd):
    """Verify mapping of 1A1, 1A2, 1A3, 1A4b, and _T to canonical variable names and values in kt."""
    df_usa_2020 = fetch_oecd_ghg(codes=["USA"], start_year=2020)
    var_map = dict(zip(df_usa_2020["variable"], df_usa_2020["value"]))

    # 1A1 Energy industries: 1445100.0 kt
    assert "ghg_energy_industries_kt_a" in var_map
    assert var_map["ghg_energy_industries_kt_a"] == pytest.approx(1445100.0)

    # 1A2 Manufacturing: 770100.0 kt
    assert "ghg_manufacturing_kt_a" in var_map
    assert var_map["ghg_manufacturing_kt_a"] == pytest.approx(770100.0)

    # 1A3 Transport: 1620500.0 kt
    assert "ghg_transport_kt_a" in var_map
    assert var_map["ghg_transport_kt_a"] == pytest.approx(1620500.0)

    # 1A4b Residential: 315800.0 kt
    assert "ghg_residential_kt_a" in var_map
    assert var_map["ghg_residential_kt_a"] == pytest.approx(315800.0)

    # _T Total: 5981400.0 kt
    assert "ghg_total_kt_a" in var_map
    assert var_map["ghg_total_kt_a"] == pytest.approx(5981400.0)


def test_oecd_ghg_filters_aggregates_and_unmapped(mock_oecd):
    """Verify regional aggregate (OECD) and unmapped sector (5) are excluded from default queries."""
    df = fetch_oecd_ghg()
    assert "OECD" not in df["code"].values
    assert set(df["code"].unique()) == {"USA", "DEU"}

    # Sector 5 (waste) is in fixture but not in default 5 IPCC sectors
    assert not any(v.startswith("ghg_5") for v in df["variable"])


def test_oecd_ghg_sector_and_country_filtering(mock_oecd):
    """Test filtering by specific sectors and country codes."""
    df_transport = fetch_oecd_ghg(sectors=["1A3"])
    assert set(df_transport["variable"].unique()) == {"ghg_transport_kt_a"}

    df_deu = fetch_oecd_ghg(codes=["DEU"])
    assert set(df_deu["code"].unique()) == {"DEU"}

    df_empty = fetch_oecd_ghg(codes=[])
    assert df_empty.empty
    assert list(df_empty.columns) == _EXPECTED_COLS


def test_oecd_ghg_year_filtering(mock_oecd):
    """Test start_year and end_year bounding for OECD GHG."""
    df_2020 = fetch_oecd_ghg(codes=["USA"], start_year=2020, end_year=2020)
    assert not df_2020.empty
    assert (df_2020["date"] == pd.Timestamp("2020-01-01")).all()

    df_2019 = fetch_oecd_ghg(codes=["USA"], start_year=2019, end_year=2019)
    assert not df_2019.empty
    assert (df_2019["date"] == pd.Timestamp("2019-01-01")).all()


def test_oecd_ghg_error_handling(monkeypatch):
    """Test graceful degradation when OECD SDMX query fails or returns invalid CSV."""
    monkeypatch.setattr(em_mod, "_get_oecd_csv", lambda *a, **k: pd.DataFrame())
    df_fail = fetch_oecd_ghg()
    assert df_fail.empty
    assert list(df_fail.columns) == _EXPECTED_COLS

    # Missing required columns
    bad_df = pd.DataFrame({"FOO": [1, 2], "BAR": [3, 4]})
    monkeypatch.setattr(em_mod, "_get_oecd_csv", lambda *a, **k: bad_df)
    df_bad = fetch_oecd_ghg()
    assert df_bad.empty
    assert list(df_bad.columns) == _EXPECTED_COLS


# ============================================================================
# 3. Unified Emissions Panel Tests
# ============================================================================


def test_emissions_panel_annual_combination(mock_wdi, mock_oecd):
    """Verify fetch_emissions_panel merges WDI and OECD without duplicate keys."""
    panel = fetch_emissions_panel(codes=["USA", "DEU"], start_year=2019, frequency="A")
    assert not panel.empty
    assert list(panel.columns) == _EXPECTED_COLS

    # Check that both WDI-specific and OECD-specific variables exist
    vars_present = set(panel["variable"].unique())
    assert "co2_pc_a" in vars_present
    assert "co2_total_kt_a" in vars_present
    assert "ghg_total_kt_a" in vars_present
    assert "methane_kt_a" in vars_present
    assert "ghg_energy_industries_kt_a" in vars_present
    assert "ghg_transport_kt_a" in vars_present

    # Assert ZERO duplicate (code, date, variable) triplets
    dups = panel.duplicated(subset=["code", "date", "variable"], keep=False)
    assert not dups.any(), panel[dups]

    # Verify OECD takes priority for ghg_total_kt_a when both exist (e.g. USA 2020)
    usa_total_2020 = panel[
        (panel["code"] == "USA")
        & (panel["date"] == pd.Timestamp("2020-01-01"))
        & (panel["variable"] == "ghg_total_kt_a")
    ]
    assert len(usa_total_2020) == 1
    assert "OECD" in usa_total_2020["source"].iloc[0]


def test_emissions_panel_quarterly_resampling(mock_wdi, mock_oecd):
    """Verify quarterly frequency expands annual data to Q1..Q4 with _q suffix."""
    panel_q = fetch_emissions_panel(codes=["USA"], start_year=2020, frequency="Q")
    assert not panel_q.empty

    # All variables must end in _q
    assert all(v.endswith("_q") for v in panel_q["variable"].unique())

    # All sources must be marked resampled_from_A:
    assert panel_q["source"].str.startswith("resampled_from_A:").all()

    # Dates must include Q1, Q2, Q3, Q4
    months = {d.month for d in panel_q["date"].unique()}
    assert months == {1, 4, 7, 10}

    # Value should be repeated across quarters
    co2_q = panel_q[panel_q["variable"] == "co2_pc_q"]
    assert len(co2_q) == 4
    assert co2_q["value"].nunique() == 1
    assert co2_q["value"].iloc[0] == pytest.approx(13.68)


def test_emissions_panel_invalid_frequency():
    """Verify unsupported frequencies raise clear ValueError."""
    with pytest.raises(ValueError, match="Unsupported frequency"):
        fetch_emissions_panel(frequency="M")

    with pytest.raises(ValueError, match="Unsupported frequency"):
        fetch_emissions_panel(frequency="daily")


def test_emissions_panel_empty_when_both_fail(monkeypatch):
    """Verify empty long-form DataFrame when both fetchers return empty."""
    monkeypatch.setattr(em_mod, "fetch_wdi_emissions", lambda *a, **k: em_mod._EMPTY.copy())
    monkeypatch.setattr(em_mod, "fetch_oecd_ghg", lambda *a, **k: em_mod._EMPTY.copy())
    panel = fetch_emissions_panel()
    assert panel.empty
    assert list(panel.columns) == _EXPECTED_COLS
