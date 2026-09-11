"""Unit tests for international financial and macroprudential data collectors.

Verifies:
1. Sovereign debt yield curves (10Y, 2Y) and date normalization.
2. Central bank policy rates with BIS WS_CBPOL and FRED fallbacks, including
   ECB rate mapping to Eurozone member states.
3. BIS macroprudential collectors: Credit-to-GDP gap, total credit to private sector,
   and residential property prices.
4. Financial conditions and stress indices (TED spread, HY OAS, EM OAS, NFCI).
5. Mathematical derivations of term spreads and sovereign spreads vs USA or DEU.
6. Schema compliance, typing invariants, edge case resilience, and 100% offline execution.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

import puremacro.fetch.financial as fin

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "financial"

EXPECTED_COLUMNS = ["code", "date", "variable", "value", "sa_source", "source"]


def mock_cached_get(url: str, *args, **kwargs) -> bytes:
    """Offline dispatcher mapping remote URLs to recorded frozen fixtures."""
    parsed = urlparse(url)
    # 1. FRED CSV request: /graph/fredgraph.csv?id=<series_id>
    if "fredgraph.csv" in parsed.path or "id=" in parsed.query:
        qs = parse_qs(parsed.query)
        series_ids = qs.get("id", [])
        if series_ids:
            s_id = series_ids[0]
            fixture_file = FIXTURE_DIR / f"{s_id}.csv"
            if fixture_file.exists():
                return fixture_file.read_bytes()

    # 2. BIS SDMX-CSV requests: WS_CBPOL, WS_CREDIT_GAP, WS_TC, WS_SPP
    for flow in ["WS_CBPOL", "WS_CREDIT_GAP", "WS_TC", "WS_SPP"]:
        if flow in url:
            fixture_file = FIXTURE_DIR / f"{flow}.csv"
            if fixture_file.exists():
                return fixture_file.read_bytes()

    raise FileNotFoundError(f"Offline mock: no fixture recorded for URL: {url}")


@pytest.fixture(autouse=True)
def offline_mock(monkeypatch):
    """Ensure all tests run 100% offline using recorded test fixtures."""
    from puremacro.fetch import _http
    monkeypatch.setattr(_http, "cached_get", mock_cached_get)
    monkeypatch.setattr(fin, "cached_get", mock_cached_get)


# ---------------------------------------------------------------------------
# 1. Sovereign Yields
# ---------------------------------------------------------------------------

def test_fetch_sovereign_yields_schema_and_values():
    """Verify sovereign yields return schema-conforming long-form DataFrame."""
    df = fin.fetch_sovereign_yields(codes=["USA", "DEU", "GBR", "JPN"])
    assert not df.empty
    assert list(df.columns) == EXPECTED_COLUMNS

    # Check ISO-3 codes
    assert set(df["code"].unique()).issubset({"USA", "DEU", "GBR", "JPN"})
    assert "USA" in df["code"].values
    assert "DEU" in df["code"].values

    # Check variables
    assert set(df["variable"].unique()) == {"yield_10y", "yield_2y"}

    # Check date properties: Timestamp and normalized to month-start
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert (df["date"].dt.day == 1).all()

    # Check value properties
    assert pd.api.types.is_float_dtype(df["value"])
    assert not df["value"].isna().any()

    # Check sources
    us_10y = df[(df["code"] == "USA") & (df["variable"] == "yield_10y")]
    assert (us_10y["source"] == "FRED:GS10").all()
    assert (us_10y["sa_source"] == "none").all()

    de_10y = df[(df["code"] == "DEU") & (df["variable"] == "yield_10y")]
    assert (de_10y["source"] == "FRED:IRLTLT01DEM156N").all()


def test_fetch_sovereign_yields_filtering():
    """Verify maturity, start_date, and country code filtering."""
    # Maturity filtering
    df_10 = fin.fetch_sovereign_yields(codes=["USA"], maturities=["10Y"])
    assert set(df_10["variable"].unique()) == {"yield_10y"}

    df_2 = fin.fetch_sovereign_yields(codes=["USA"], maturities=["2Y"])
    assert set(df_2["variable"].unique()) == {"yield_2y"}

    # Date cutoff
    df_date = fin.fetch_sovereign_yields(codes=["USA"], start_date="2020-04-01")
    assert df_date["date"].min() >= pd.Timestamp("2020-04-01")


def test_fetch_sovereign_yields_unknown_code():
    """Verify unknown country codes are ignored gracefully."""
    df = fin.fetch_sovereign_yields(codes=["NONEXISTENT_XYZ"])
    assert df.empty
    assert list(df.columns) == EXPECTED_COLUMNS


# ---------------------------------------------------------------------------
# 2. Central Bank Policy Rates
# ---------------------------------------------------------------------------

def test_fetch_policy_rates_bis_with_eurozone_mapping():
    """Verify BIS WS_CBPOL fetch and automatic ECB mapping to Eurozone members."""
    df = fin.fetch_policy_rates(codes=["USA", "DEU", "FRA", "GBR", "JPN"])
    assert not df.empty
    assert list(df.columns) == EXPECTED_COLUMNS

    # All requested countries are present
    assert set(df["code"].unique()) == {"USA", "DEU", "FRA", "GBR", "JPN"}
    assert (df["variable"] == "policy_rate").all()
    assert (df["date"].dt.day == 1).all()

    # DEU and FRA received ECB policy rate (-0.50 in 2020-01)
    de_rate = df[(df["code"] == "DEU") & (df["date"] == "2020-01-01")]["value"].iloc[0]
    fr_rate = df[(df["code"] == "FRA") & (df["date"] == "2020-01-01")]["value"].iloc[0]
    assert de_rate == -0.50
    assert fr_rate == -0.50

    # US Federal Funds rate (1.55 in 2020-01)
    us_rate = df[(df["code"] == "USA") & (df["date"] == "2020-01-01")]["value"].iloc[0]
    assert us_rate == 1.55


def test_fetch_policy_rates_fred_fallback(monkeypatch):
    """Verify FRED fallback for policy rates when BIS is unavailable."""
    def mock_no_bis(url, *args, **kwargs):
        if "WS_CBPOL" in url:
            raise RuntimeError("BIS API temporarily down")
        return mock_cached_get(url, *args, **kwargs)

    monkeypatch.setattr(fin, "cached_get", mock_no_bis)

    df = fin.fetch_policy_rates(codes=["USA", "DEU"], source_preference="fred")
    assert not df.empty
    assert set(df["code"].unique()) == {"USA", "DEU"}
    assert (df["variable"] == "policy_rate").all()
    us_row = df[(df["code"] == "USA") & (df["date"] == "2020-01-01")]
    assert us_row["source"].iloc[0] == "FRED:FEDFUNDS"


# ---------------------------------------------------------------------------
# 3. BIS Macroprudential Indicators
# ---------------------------------------------------------------------------

def test_fetch_bis_macroprudential_credit_gap():
    """Verify BIS Credit-to-GDP gap indicators (gap, ratio, trend)."""
    df = fin.fetch_bis_macroprudential(
        codes=["USA", "DEU"],
        indicators=["credit_gap_q", "credit_to_gdp_q", "credit_trend_q"],
    )
    assert not df.empty
    assert list(df.columns) == EXPECTED_COLUMNS
    assert set(df["code"].unique()) == {"USA", "DEU"}

    expected_vars = {"credit_gap_q", "credit_to_gdp_q", "credit_trend_q"}
    assert set(df["variable"].unique()) == expected_vars

    # Quarterly date normalization: Q1 -> Jan 1, Q2 -> Apr 1, Q3 -> Jul 1
    assert set(df["date"].dt.month.unique()).issubset({1, 4, 7, 10})
    assert (df["date"].dt.day == 1).all()

    # Values check for USA Q1 2020
    us_q1_gap = df[(df["code"] == "USA") & (df["date"] == "2020-01-01") & (df["variable"] == "credit_gap_q")]
    assert us_q1_gap["value"].iloc[0] == -5.2


def test_fetch_bis_macroprudential_total_credit():
    """Verify BIS total credit to private non-financial sector (% of GDP and USD)."""
    df = fin.fetch_bis_total_credit(codes=["USA", "DEU"])
    assert not df.empty
    assert list(df.columns) == EXPECTED_COLUMNS
    assert set(df["code"].unique()) == {"USA", "DEU"}
    assert set(df["variable"].unique()) == {"credit_private_pct_gdp_q", "credit_private_usd_q"}

    # Value check for USA USD billions
    us_usd = df[(df["code"] == "USA") & (df["date"] == "2020-01-01") & (df["variable"] == "credit_private_usd_q")]
    assert us_usd["value"].iloc[0] == 32850.5


def test_fetch_bis_macroprudential_property_prices():
    """Verify BIS real and nominal residential property prices."""
    df_both = fin.fetch_bis_property_prices(codes=["USA", "DEU"], price_type="both")
    assert not df_both.empty
    assert set(df_both["variable"].unique()) == {"property_price_real_q", "property_price_nominal_q"}

    df_real = fin.fetch_bis_property_prices(codes=["USA"], price_type="real")
    assert set(df_real["variable"].unique()) == {"property_price_real_q"}

    df_nom = fin.fetch_bis_property_prices(codes=["DEU"], price_type="nominal")
    assert set(df_nom["variable"].unique()) == {"property_price_nominal_q"}


def test_fetch_bis_macroprudential_all_indicators():
    """Verify fetching all 7 macroprudential indicators simultaneously."""
    df = fin.fetch_bis_macroprudential(codes=["USA", "DEU"])
    assert not df.empty
    assert len(df["variable"].unique()) == 7
    assert (df["source"].str.startswith("BIS:WS_")).all()


# ---------------------------------------------------------------------------
# 4. Financial Conditions & Stress Indices
# ---------------------------------------------------------------------------

def test_fetch_financial_conditions():
    """Verify TED rate, US HY OAS, EM OAS, NFCI, and T10Y2Y spread."""
    df = fin.fetch_financial_conditions()
    assert not df.empty
    assert list(df.columns) == EXPECTED_COLUMNS

    expected_vars = {"ted_spread", "hy_spread", "em_spread", "nfci", "term_spread_us"}
    assert set(df["variable"].unique()) == expected_vars

    # Monthly frequency normalization: month-start Timestamps
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert (df["date"].dt.day == 1).all()

    # Geographic codes
    assert "USA" in df["code"].values
    assert "WLD" in df["code"].values  # EM OAS assigned to WLD
    em_df = df[df["variable"] == "em_spread"]
    assert (em_df["code"] == "WLD").all()

    # Sources
    ted_df = df[df["variable"] == "ted_spread"]
    assert (ted_df["source"] == "FRED:TEDRATE").all()


def test_fetch_financial_conditions_single_indicator():
    """Verify selective fetching of financial conditions."""
    df = fin.fetch_financial_conditions(indicators=["nfci"], start_date="2020-03-01")
    assert not df.empty
    assert set(df["variable"].unique()) == {"nfci"}
    assert df["date"].min() >= pd.Timestamp("2020-03-01")


# ---------------------------------------------------------------------------
# 5. Sovereign Spreads Computation
# ---------------------------------------------------------------------------

def test_compute_sovereign_spreads_derivation():
    """Verify 10Y-2Y term spread and sovereign risk spread calculations."""
    yields = fin.fetch_sovereign_yields(codes=["USA", "DEU"])
    spreads = fin.compute_sovereign_spreads(yields, benchmark_code="USA")

    assert not spreads.empty
    assert list(spreads.columns) == EXPECTED_COLUMNS
    assert set(spreads["variable"].unique()) == {"term_spread", "sovereign_spread"}
    assert (spreads["sa_source"] == "derived").all()

    # 1. Term Spread: yield_10y - yield_2y
    # In 2020-01: US 10Y = 1.76, US 2Y = 1.52 -> term spread = 0.24
    us_term = spreads[(spreads["code"] == "USA") & (spreads["date"] == "2020-01-01") & (spreads["variable"] == "term_spread")]
    assert np.isclose(us_term["value"].iloc[0], 0.24, atol=1e-4)

    # In 2020-01: DE 10Y = -0.23, DE 2Y = -0.60 -> term spread = 0.37
    de_term = spreads[(spreads["code"] == "DEU") & (spreads["date"] == "2020-01-01") & (spreads["variable"] == "term_spread")]
    assert np.isclose(de_term["value"].iloc[0], 0.37, atol=1e-4)

    # 2. Sovereign Spread vs USA
    # For USA itself, sovereign spread must be identically 0.0
    us_sov = spreads[(spreads["code"] == "USA") & (spreads["variable"] == "sovereign_spread")]
    assert (us_sov["value"] == 0.0).all()

    # For DEU: DE 10Y (-0.23) - US 10Y (1.76) = -1.99
    de_sov = spreads[(spreads["code"] == "DEU") & (spreads["date"] == "2020-01-01") & (spreads["variable"] == "sovereign_spread")]
    assert np.isclose(de_sov["value"].iloc[0], -1.99, atol=1e-4)


def test_compute_sovereign_spreads_include_yields():
    """Verify include_yields=True appends spreads to yields."""
    yields = fin.fetch_sovereign_yields(codes=["USA"])
    combined = fin.compute_sovereign_spreads(yields, benchmark_code="USA", include_yields=True)
    vars_present = set(combined["variable"].unique())
    assert {"yield_10y", "yield_2y", "term_spread", "sovereign_spread"}.issubset(vars_present)


def test_compute_sovereign_spreads_missing_benchmark():
    """Verify warning and graceful handling when benchmark country is not in yields."""
    # Only DEU yields provided, benchmark is USA
    yields_de = fin.fetch_sovereign_yields(codes=["DEU"])
    with pytest.warns(UserWarning, match="Benchmark code 'USA' not found"):
        spreads = fin.compute_sovereign_spreads(yields_de, benchmark_code="USA")
    # Term spread is still computed!
    assert "term_spread" in spreads["variable"].values
    assert "sovereign_spread" not in spreads["variable"].values


def test_compute_sovereign_spreads_empty():
    """Verify empty DataFrame input returns empty DataFrame."""
    out = fin.compute_sovereign_spreads(pd.DataFrame())
    assert out.empty
    assert list(out.columns) == EXPECTED_COLUMNS


# ---------------------------------------------------------------------------
# 6. Resilience, Error Handling & ISO-3 Mappings
# ---------------------------------------------------------------------------

def test_offline_connection_failure(monkeypatch):
    """Verify fetchers return schema-conforming empty DataFrame on network drops."""
    def mock_broken(url, *args, **kwargs):
        raise ConnectionError("Network unreachable")

    monkeypatch.setattr(fin, "cached_get", mock_broken)

    df_y = fin.fetch_sovereign_yields(codes=["USA"])
    assert df_y.empty
    assert list(df_y.columns) == EXPECTED_COLUMNS

    df_p = fin.fetch_policy_rates(codes=["USA"])
    assert df_p.empty
    assert list(df_p.columns) == EXPECTED_COLUMNS

    df_m = fin.fetch_bis_macroprudential(codes=["USA"])
    assert df_m.empty
    assert list(df_m.columns) == EXPECTED_COLUMNS

    df_f = fin.fetch_financial_conditions()
    assert df_f.empty
    assert list(df_f.columns) == EXPECTED_COLUMNS


def test_mappings_integrity():
    """Verify BIS and ISO-3 mapping bidirectional consistency."""
    assert fin._BIS_TO_ISO3["US"] == "USA"
    assert fin._BIS_TO_ISO3["DE"] == "DEU"
    assert fin._ISO3_TO_BIS["USA"] == "US"
    assert fin._ISO3_TO_BIS["DEU"] == "DE"
    assert "DEU" in fin._EUROZONE_ISO3
    assert "FRA" in fin._EUROZONE_ISO3
    assert "USA" not in fin._EUROZONE_ISO3
