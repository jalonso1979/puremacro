"""Comprehensive offline tests for energy transition and commodity benchmark suites.

Tests:
- puremacro.fetch.energy_transition (fetch_energy_transition)
- puremacro.fetch.wb_pink_sheet (fetch_prices, fetch_indices, fetch_commodity_benchmarks)
- puremacro.fetch.commodities (convenience fetchers and re-exports)
"""
from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from puremacro.fetch.energy_transition import (
    VARIABLES as ENERGY_VARIABLES,
    fetch_energy_transition,
)
from puremacro.fetch.wb_pink_sheet import (
    COMMODITY_CATEGORIES,
    fetch_commodity_benchmarks,
    fetch_indices,
    fetch_prices,
)
from puremacro.fetch.commodities import (
    fetch_agriculture_benchmarks,
    fetch_commodity_indices,
    fetch_energy_benchmarks,
    fetch_metal_benchmarks,
)

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "commodities"
_MOCK_XLSX = _FIXTURE_DIR / "cmo_historical_mock.xlsx"
_MOCK_EMBER_CSV = _FIXTURE_DIR / "ember_generation_mock.csv"
_MOCK_ENERGY_FIXTURE = _FIXTURE_DIR / "energy_transition_fixture.csv"

_EXPECTED_SCHEMA = ["code", "date", "variable", "value", "sa_source", "source"]


# =============================================================================
# 1. Commodity Benchmarks Unit Tests
# =============================================================================

def test_fetch_commodity_benchmarks_monthly_schema():
    df = fetch_commodity_benchmarks(
        file_path=_MOCK_XLSX,
        frequency="M",
        start_date="2020-01-01",
        include_indices=True,
    )
    assert not df.empty
    assert list(df.columns) == _EXPECTED_SCHEMA
    assert (df["code"] == "WLD").all()
    assert (df["sa_source"] == "none").all()
    assert df["value"].notna().all()
    assert (df["value"] > 0).all()


def test_fetch_commodity_benchmarks_all_categories_present():
    df = fetch_commodity_benchmarks(
        file_path=_MOCK_XLSX,
        frequency="M",
        include_indices=True,
    )
    vars_found = set(df["variable"].unique())

    # Energy
    expected_energy = {"brent", "wti", "natgas_us", "natgas_eu", "coal_au", "coal_za"}
    assert expected_energy.issubset(vars_found), f"Missing energy: {expected_energy - vars_found}"

    # Industrial Metals
    expected_ind_metals = {"copper", "aluminum", "iron_ore"}
    assert expected_ind_metals.issubset(vars_found), f"Missing industrial metals: {expected_ind_metals - vars_found}"

    # Precious Metals
    expected_prec_metals = {"gold", "silver"}
    assert expected_prec_metals.issubset(vars_found), f"Missing precious metals: {expected_prec_metals - vars_found}"

    # Agriculture & Fertilizers
    expected_agri_fert = {"wheat", "maize", "rice", "soybeans", "phosphate_rock", "dap", "urea"}
    assert expected_agri_fert.issubset(vars_found), f"Missing agri/fert: {expected_agri_fert - vars_found}"

    # Indices
    expected_indices = {"index_total", "index_energy", "index_non_energy", "index_agri", "index_metals", "index_fert"}
    assert expected_indices.issubset(vars_found), f"Missing indices: {expected_indices - vars_found}"


def test_fetch_commodity_benchmarks_include_indices_toggle():
    df_with = fetch_commodity_benchmarks(file_path=_MOCK_XLSX, include_indices=True)
    df_without = fetch_commodity_benchmarks(file_path=_MOCK_XLSX, include_indices=False)

    indices_with = {v for v in df_with["variable"].unique() if v.startswith("index_")}
    indices_without = {v for v in df_without["variable"].unique() if v.startswith("index_")}

    assert len(indices_with) >= 6
    assert len(indices_without) == 0


def test_fetch_commodity_benchmarks_category_filter_energy():
    df = fetch_commodity_benchmarks(file_path=_MOCK_XLSX, categories=["energy"])
    assert set(df["variable"].unique()) == {
        "brent", "wti", "natgas_us", "natgas_eu", "coal_au", "coal_za"
    }


def test_fetch_commodity_benchmarks_category_filter_metals_alias():
    df = fetch_commodity_benchmarks(file_path=_MOCK_XLSX, categories=["metals"])
    assert set(df["variable"].unique()) == {
        "copper", "aluminum", "iron_ore", "gold", "silver"
    }


def test_fetch_commodity_benchmarks_category_filter_agri_and_fert():
    df = fetch_commodity_benchmarks(file_path=_MOCK_XLSX, categories=["agri", "fert"])
    assert set(df["variable"].unique()) == {
        "wheat", "maize", "rice", "soybeans", "phosphate_rock", "dap", "urea"
    }


def test_fetch_commodity_benchmarks_quarterly_rollup_math():
    df_m = fetch_commodity_benchmarks(file_path=_MOCK_XLSX, frequency="M", categories=["energy"])
    df_q = fetch_commodity_benchmarks(file_path=_MOCK_XLSX, frequency="Q", categories=["energy"])

    assert not df_q.empty
    assert list(df_q.columns) == _EXPECTED_SCHEMA
    assert (df_q["code"] == "WLD").all()

    # All quarterly variables end with _q by default
    assert all(v.endswith("_q") for v in df_q["variable"].unique())

    # All sources are marked as resampled
    assert all(s.startswith("resampled_from_M:") for s in df_q["source"].unique())

    # Dates are quarter start (months 1, 4, 7, 10)
    assert set(df_q["date"].dt.month.unique()).issubset({1, 4, 7, 10})
    assert (df_q["date"].dt.day == 1).all()

    # Verify mathematical accuracy: quarterly mean of monthly values for brent in 2020Q1
    brent_m_q1 = df_m[
        (df_m["variable"] == "brent") &
        (df_m["date"] >= "2020-01-01") &
        (df_m["date"] < "2020-04-01")
    ]["value"].mean()

    brent_q_q1 = df_q[
        (df_q["variable"] == "brent_q") &
        (df_q["date"] == "2020-01-01")
    ]["value"].iloc[0]

    assert brent_q_q1 == pytest.approx(brent_m_q1, rel=1e-5)


def test_fetch_commodity_benchmarks_start_date_filter():
    df = fetch_commodity_benchmarks(file_path=_MOCK_XLSX, start_date="2022-06-01")
    assert (df["date"] >= pd.Timestamp("2022-06-01")).all()


def test_fetch_commodity_benchmarks_invalid_frequency_raises():
    with pytest.raises(ValueError, match="frequency"):
        fetch_commodity_benchmarks(file_path=_MOCK_XLSX, frequency="DAILY")


def test_fetch_commodity_benchmarks_download_failure_returns_empty(monkeypatch):
    from puremacro.fetch import wb_pink_sheet as mod
    def _fail(*a, **kw):
        raise OSError("offline simulated")
    monkeypatch.setattr(mod, "cached_get", _fail)
    df = fetch_commodity_benchmarks()
    assert df.empty
    assert list(df.columns) == _EXPECTED_SCHEMA


def test_wb_pink_sheet_fetch_indices_direct():
    bytes_data = _MOCK_XLSX.read_bytes()
    df_idx = fetch_indices(workbook_bytes=bytes_data)
    assert not df_idx.empty
    assert list(df_idx.columns) == _EXPECTED_SCHEMA
    assert (df_idx["source"] == "WorldBank:PinkSheet:MonthlyIndices").all()
    # At least 16 distinct indices
    assert len(df_idx["variable"].unique()) >= 16
    assert "index_total" in df_idx["variable"].values
    assert "index_energy" in df_idx["variable"].values


def test_commodities_convenience_functions():
    bytes_data = _MOCK_XLSX.read_bytes()

    df_nrg = fetch_energy_benchmarks(workbook_bytes=bytes_data)
    assert not df_nrg.empty
    assert set(df_nrg["variable"].unique()) == {
        "brent", "wti", "natgas_us", "natgas_eu", "coal_au", "coal_za"
    }

    df_met = fetch_metal_benchmarks(workbook_bytes=bytes_data)
    assert not df_met.empty
    assert set(df_met["variable"].unique()) == {
        "copper", "aluminum", "iron_ore", "gold", "silver"
    }

    df_agri = fetch_agriculture_benchmarks(workbook_bytes=bytes_data)
    assert not df_agri.empty
    assert set(df_agri["variable"].unique()) == {
        "wheat", "maize", "rice", "soybeans", "phosphate_rock", "dap", "urea"
    }

    df_idx = fetch_commodity_indices(workbook_bytes=bytes_data)
    assert not df_idx.empty
    assert "index_total" in df_idx["variable"].values


# =============================================================================
# 2. Energy Transition Unit Tests
# =============================================================================

def test_fetch_energy_transition_ember_mock_schema_and_variables():
    df = fetch_energy_transition(csv_path=_MOCK_EMBER_CSV)
    assert not df.empty
    assert list(df.columns) == _EXPECTED_SCHEMA
    assert df["value"].notna().all()

    vars_found = set(df["variable"].unique())
    for req_var in ENERGY_VARIABLES:
        assert req_var in vars_found, f"Missing expected variable: {req_var}"


def test_fetch_energy_transition_country_codes_filter():
    df = fetch_energy_transition(
        codes=["USA", "DEU"], csv_path=_MOCK_EMBER_CSV
    )
    assert set(df["code"].unique()) == {"USA", "DEU"}


def test_fetch_energy_transition_drops_aggregates():
    # ember_generation_mock.csv contains 'WLD' (World)
    df = fetch_energy_transition(csv_path=_MOCK_EMBER_CSV)
    codes = set(df["code"].unique())
    assert "WLD" not in codes
    assert "EU" not in codes
    assert "G20" not in codes
    assert "WORLD" not in codes
    assert all(len(c) == 3 and c.isalpha() for c in codes)


def test_fetch_energy_transition_start_year_filter():
    df = fetch_energy_transition(start_year=2020, csv_path=_MOCK_EMBER_CSV)
    assert (df["date"].dt.year >= 2020).all()


def test_fetch_energy_transition_share_ranges():
    df = fetch_energy_transition(csv_path=_MOCK_EMBER_CSV)
    share_vars = [
        "elec_gen_renewable_pct_a",
        "elec_gen_hydro_pct_a",
        "elec_gen_nuclear_pct_a",
        "elec_gen_fossil_pct_a",
    ]
    for sv in share_vars:
        vals = df[df["variable"] == sv]["value"]
        assert (vals >= 0.0).all()
        assert (vals <= 100.0).all()


def test_fetch_energy_transition_fixture_compatibility():
    df = fetch_energy_transition(csv_path=_MOCK_ENERGY_FIXTURE)
    assert not df.empty
    assert list(df.columns) == _EXPECTED_SCHEMA
    assert set(ENERGY_VARIABLES).issubset(set(df["variable"].unique()))


def test_fetch_energy_transition_network_failure_returns_empty(monkeypatch):
    from puremacro.fetch import energy_transition as mod
    def _fail(*a, **kw):
        raise OSError("offline simulated")
    monkeypatch.setattr(mod, "cached_get", _fail)
    df = fetch_energy_transition()
    assert df.empty
    assert list(df.columns) == _EXPECTED_SCHEMA


def test_fetch_energy_transition_via_cached_get_mock(monkeypatch):
    """Verify standard cached_get download path when csv_path is not specified."""
    from puremacro.fetch import energy_transition as mod
    ember_bytes = _MOCK_EMBER_CSV.read_bytes()

    def _mock_get(url, *a, **kw):
        if "ember" in url.lower():
            return ember_bytes
        raise OSError("no other urls in test")

    monkeypatch.setattr(mod, "cached_get", _mock_get)
    df = fetch_energy_transition(codes=["USA", "FRA"])
    assert not df.empty
    assert set(df["code"].unique()) == {"USA", "FRA"}
    assert "elec_gen_total_twh_a" in df["variable"].values
