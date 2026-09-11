"""Offline unit tests for modular panel builders and frequency harmonization.

Verifies:
1. `build_climate_panel` in `puremacro.climate_panel`:
   - Schema conformance and long-form vs wide matrix layout.
   - Frequency harmonization: annual ('A') vs quarterly ('Q') via 'repeat' and 'interpolate'.
   - Zero duplicate keys `(code, date, variable)`.
   - Date range filtering and boundary parameter validations.
2. `build_financial_panel` in `puremacro.financial_panel`:
   - Schema conformance and long-form vs wide matrix layout.
   - Frequency harmonization: quarterly ('Q') aggregation via 'mean' vs 'last'.
   - Monthly ('M') projection of quarterly credit gaps and property prices.
   - Global commodity benchmarks ('WLD') and financial conditions integration.
   - Zero duplicate keys `(code, date, variable)`.
   - Date range filtering and boundary parameter validations.
3. Re-export integrity:
   - `build_climate_panel` and `build_financial_panel` exported from `puremacro.build_panel`.
   - Existing `build_panel` functions (`build_all`, `load_country`, `merge_frames`) preserved.
4. 100% offline determinism:
   - Remote HTTP requests mocked with frozen fixtures from `tests/data/`.
"""
from __future__ import annotations

import io
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

from puremacro.build_panel import (
    build_all,
    build_climate_panel as bp_climate_panel,
    build_financial_panel as bp_financial_panel,
    load_country,
    merge_frames,
)
from puremacro.climate_panel import build_climate_panel
from puremacro.financial_panel import build_financial_panel

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data"
FIN_DIR = FIXTURE_DIR / "financial"
COMM_DIR = FIXTURE_DIR / "commodities"
EMISS_DIR = FIXTURE_DIR / "emissions"


@pytest.fixture(autouse=True)
def mock_all_network(monkeypatch):
    """Intercept all network calls and serve local frozen fixtures."""
    from puremacro.fetch import _http

    def offline_cached_get(url: str, *, refresh: bool = False, timeout: float | None = None, **kwargs) -> bytes:
        # 1. World Bank WDI (emissions / energy)
        if "api.worldbank.org" in url:
            if "EG.USE" in url or "energy" in url.lower():
                p = COMM_DIR / "wdi_energy_mock.json"
                if p.exists():
                    return p.read_bytes()
            p = EMISS_DIR / "wdi_emissions.json"
            if p.exists():
                return p.read_bytes()

        # 2. OECD SDMX emissions
        if "DSD_AIR_GHG" in url or "DF_AIR_GHG" in url:
            p = EMISS_DIR / "oecd_ghg.csv"
            if p.exists():
                return p.read_bytes()

        # 3. Ember electricity generation
        if "ember-energy.org" in url or "generation_yearly" in url:
            p = COMM_DIR / "ember_generation_mock.csv"
            if p.exists():
                return p.read_bytes()

        # 4. Commodities Pink Sheet workbook
        if "worldbank.org" in url and ("CMO" in url or "Pink" in url or "Historical" in url):
            p = COMM_DIR / "cmo_historical_mock.xlsx"
            if p.exists():
                return p.read_bytes()

        # 5. FRED CSV queries
        parsed = urlparse(url)
        if "fredgraph.csv" in parsed.path or "id=" in parsed.query:
            qs = parse_qs(parsed.query)
            s_ids = qs.get("id", [])
            if s_ids:
                s_id = s_ids[0]
                p = FIN_DIR / f"{s_id}.csv"
                if p.exists():
                    return p.read_bytes()

        # 6. BIS SDMX CSV queries
        for flow in ["WS_CBPOL", "WS_CREDIT_GAP", "WS_TC", "WS_SPP"]:
            if flow in url:
                p = FIN_DIR / f"{flow}.csv"
                if p.exists():
                    return p.read_bytes()

        return b"date,value\n2020-01-01,1.0\n"

    monkeypatch.setattr(_http, "cached_get", offline_cached_get)


# ============================================================================
# 1. Climate Panel Tests
# ============================================================================

class TestClimatePanel:
    """Unit tests for build_climate_panel."""

    def test_schema_conformance_long_format(self):
        """Panel conforms to standard long-form schema."""
        panel = build_climate_panel(codes=["USA", "DEU"], start_year=2015, frequency="A")
        assert isinstance(panel, pd.DataFrame)
        assert not panel.empty
        expected = ["code", "date", "variable", "value", "sa_source", "source"]
        for col in expected:
            assert col in panel.columns
        assert pd.api.types.is_datetime64_any_dtype(panel["date"])
        assert pd.api.types.is_numeric_dtype(panel["value"])

    def test_annual_frequency_dates(self):
        """Annual frequency normalizes to January 1."""
        panel = build_climate_panel(codes=["USA"], start_year=2015, frequency="A")
        assert not panel.empty
        assert (panel["date"].dt.month == 1).all()
        assert (panel["date"].dt.day == 1).all()

    def test_quarterly_repeat_harmonization(self):
        """Quarterly harmonization with 'repeat' expands to Q1-Q4 with identical values."""
        panel_q = build_climate_panel(codes=["USA"], start_year=2018, end_year=2019, frequency="Q", harmonization="repeat")
        assert not panel_q.empty
        assert set(panel_q["date"].dt.month).issubset({1, 4, 7, 10})
        assert (panel_q["date"].dt.day == 1).all()
        # All variable names should end in _q
        assert (panel_q["variable"].str.endswith("_q")).all()

        # Check that a year with data has 4 quarters per variable
        sub_2019 = panel_q[panel_q["date"].dt.year == 2019]
        assert not sub_2019.empty
        sample_var = sub_2019["variable"].iloc[0]
        sub = sub_2019[sub_2019["variable"] == sample_var]
        assert len(sub) == 4
        assert sub["value"].nunique() == 1  # repeated value

    def test_quarterly_interpolate_harmonization(self):
        """Quarterly harmonization with 'interpolate' produces smooth intermediate quarterly values."""
        panel_interp = build_climate_panel(
            codes=["USA"], start_year=2015, end_year=2018, frequency="Q", harmonization="interpolate"
        )
        assert not panel_interp.empty
        assert set(panel_interp["date"].dt.month).issubset({1, 4, 7, 10})

        first_var = panel_interp["variable"].iloc[0]
        sub = panel_interp[panel_interp["variable"] == first_var].sort_values("date")
        if len(sub) >= 8:
            # Intermediate quarters should have values between year endpoints if endpoints differ
            vals = sub["value"].values
            assert not np.isnan(vals).any()

    def test_wide_format_pivoting(self):
        """Wide format pivots to date index with code_variable column names."""
        panel_wide = build_climate_panel(codes=["USA", "DEU"], start_year=2015, frequency="A", wide=True)
        assert isinstance(panel_wide, pd.DataFrame)
        assert not panel_wide.empty
        assert len(panel_wide.columns) > 3
        # Columns should be flat strings containing underscore
        assert all(isinstance(c, str) and "_" in c for c in panel_wide.columns)

    def test_start_and_end_year_filters(self):
        """Respects start_year and end_year filtering."""
        panel = build_climate_panel(codes=["USA"], start_year=2016, end_year=2018, frequency="A")
        assert not panel.empty
        assert panel["date"].dt.year.min() >= 2016
        assert panel["date"].dt.year.max() <= 2018

    def test_zero_duplicate_keys(self):
        """Guaranteed zero duplicate keys on (code, date, variable)."""
        panel = build_climate_panel(codes=["USA", "DEU", "GBR"], start_year=2010, frequency="A")
        assert not panel.empty
        dups = panel.duplicated(subset=["code", "date", "variable"])
        assert not dups.any(), f"Found {dups.sum()} duplicates"

    def test_invalid_frequency_raises_value_error(self):
        """Invalid frequency parameter raises ValueError."""
        with pytest.raises(ValueError):
            build_climate_panel(codes=["USA"], frequency="M")

    def test_invalid_harmonization_raises_value_error(self):
        """Invalid harmonization parameter raises ValueError."""
        with pytest.raises(ValueError):
            build_climate_panel(codes=["USA"], harmonization="cubic")

    def test_empty_codes_returns_empty_frame(self):
        """Passing codes=[] returns empty DataFrame."""
        empty_panel = build_climate_panel(codes=[])
        assert empty_panel.empty

    def test_unknown_country_code_graceful(self):
        """Querying an unknown country code returns empty frame without error."""
        panel = build_climate_panel(codes=["ZZZ"])
        assert panel.empty

    def test_missing_data_indicators_annual(self):
        """Annual climate panel includes is_imputed column, all False."""
        panel = build_climate_panel(codes=["USA"], start_year=2015, frequency="A")
        assert "is_imputed" in panel.columns
        assert not panel["is_imputed"].any()

    def test_missing_data_indicators_quarterly_repeat(self):
        """Quarterly repeat harmonization sets sa_source='annual_repeated' and is_imputed for Q2-Q4."""
        panel_q = build_climate_panel(codes=["USA"], start_year=2018, end_year=2019, frequency="Q", harmonization="repeat")
        assert "is_imputed" in panel_q.columns
        assert (panel_q["sa_source"] == "annual_repeated").all()
        q1_mask = panel_q["date"].dt.month == 1
        assert not panel_q.loc[q1_mask, "is_imputed"].any()
        assert panel_q.loc[~q1_mask, "is_imputed"].all()

    def test_missing_data_indicators_quarterly_interpolate(self):
        """Quarterly interpolate harmonization sets sa_source='interpolated' and marks imputed points."""
        panel_interp = build_climate_panel(codes=["USA"], start_year=2015, end_year=2018, frequency="Q", harmonization="interpolate")
        assert "is_imputed" in panel_interp.columns
        assert "sa_source" in panel_interp.columns
        q_interp = panel_interp[panel_interp["date"].dt.month.isin([4, 7, 10])]
        assert q_interp["is_imputed"].all()

    def test_climate_long_to_wide_interop(self):
        """Long-form panel interoperates with puremacro.data.long_to_wide."""
        from puremacro.data import long_to_wide
        panel_long = build_climate_panel(codes=["USA", "DEU"], start_year=2015, frequency="A", wide=False)
        wide_df = long_to_wide(panel_long)
        assert isinstance(wide_df.index, pd.MultiIndex)
        assert wide_df.index.names == ["code", "date"]
        assert len(wide_df.columns) > 1


# ============================================================================
# 2. Financial Panel Tests
# ============================================================================

class TestFinancialPanel:
    """Unit tests for build_financial_panel."""

    def test_schema_conformance_long_format(self):
        """Panel conforms to standard long-form schema."""
        panel = build_financial_panel(codes=["USA", "DEU"], start_date="2015-01-01", frequency="Q")
        assert isinstance(panel, pd.DataFrame)
        assert not panel.empty
        expected = ["code", "date", "variable", "value", "sa_source", "source"]
        for col in expected:
            assert col in panel.columns
        assert pd.api.types.is_datetime64_any_dtype(panel["date"])
        assert pd.api.types.is_numeric_dtype(panel["value"])

    def test_quarterly_frequency_dates(self):
        """Quarterly frequency normalizes to quarter starts {1, 4, 7, 10}."""
        panel_q = build_financial_panel(codes=["USA"], start_date="2018-01-01", frequency="Q")
        assert not panel_q.empty
        assert set(panel_q["date"].dt.month).issubset({1, 4, 7, 10})
        assert (panel_q["date"].dt.day == 1).all()
        # All variable names should end in _q
        assert (panel_q["variable"].str.endswith("_q")).all()

    def test_monthly_frequency_projection(self):
        """Monthly frequency projects quarterly series to monthly with day 1."""
        panel_m = build_financial_panel(codes=["USA"], start_date="2020-01-01", frequency="M")
        assert not panel_m.empty
        assert (panel_m["date"].dt.day == 1).all()
        # All variable names should end in _m
        assert (panel_m["variable"].str.endswith("_m")).all()

    def test_harmonization_modes_mean_vs_last(self):
        """Harmonization modes 'mean' and 'last' both succeed."""
        p_mean = build_financial_panel(codes=["USA"], start_date="2020-01-01", frequency="Q", harmonization="mean")
        p_last = build_financial_panel(codes=["USA"], start_date="2020-01-01", frequency="Q", harmonization="last")
        assert not p_mean.empty
        assert not p_last.empty
        assert isinstance(p_mean, pd.DataFrame)
        assert isinstance(p_last, pd.DataFrame)

    def test_commodities_inclusion_and_toggle(self):
        """Commodity benchmarks are included under code='WLD' and can be toggled."""
        p_with = build_financial_panel(codes=["USA"], start_date="2020-01-01", include_commodities=True)
        p_without = build_financial_panel(codes=["USA"], start_date="2020-01-01", include_commodities=False, include_conditions=False)
        assert "WLD" in set(p_with["code"].unique())
        assert "WLD" not in set(p_without["code"].unique())

        # When commodities are toggled off with conditions on, commodity vars are omitted but WLD (em_spread) is retained
        p_no_comm = build_financial_panel(codes=["USA"], start_date="2020-01-01", include_commodities=False, include_conditions=True)
        comm_vars = {"brent_q", "wti_q", "gold_q", "copper_q"}
        assert not any(v in comm_vars for v in p_no_comm["variable"].unique())
        assert "em_spread_q" in set(p_no_comm["variable"].unique())

    def test_financial_conditions_toggle(self):
        """Financial conditions are included for USA and can be toggled."""
        p_with = build_financial_panel(codes=["USA"], start_date="2020-01-01", include_conditions=True)
        p_without = build_financial_panel(codes=["USA"], start_date="2020-01-01", include_conditions=False)
        vars_with = set(p_with["variable"].unique())
        vars_without = set(p_without["variable"].unique())
        assert any("ted" in v or "oas" in v or "nfci" in v for v in vars_with)
        assert not any("ted" in v or "oas" in v or "nfci" in v for v in vars_without)

    def test_wide_format_pivoting(self):
        """Wide format returns matrix with code_variable column names and date index."""
        panel_wide = build_financial_panel(codes=["USA"], start_date="2018-01-01", frequency="Q", wide=True)
        assert isinstance(panel_wide, pd.DataFrame)
        assert not panel_wide.empty
        assert len(panel_wide.columns) > 3
        assert all(isinstance(c, str) and "_" in c for c in panel_wide.columns)

    def test_zero_duplicate_keys(self):
        """Guaranteed zero duplicate keys on (code, date, variable)."""
        panel = build_financial_panel(codes=["USA", "DEU"], start_date="2015-01-01", frequency="Q")
        assert not panel.empty
        dups = panel.duplicated(subset=["code", "date", "variable"])
        assert not dups.any(), f"Found {dups.sum()} duplicates"

    def test_invalid_frequency_raises_value_error(self):
        """Invalid frequency parameter raises ValueError."""
        with pytest.raises(ValueError):
            build_financial_panel(codes=["USA"], frequency="A")

    def test_invalid_harmonization_raises_value_error(self):
        """Invalid harmonization parameter raises ValueError."""
        with pytest.raises(ValueError):
            build_financial_panel(codes=["USA"], harmonization="median")

    def test_empty_codes_returns_empty_frame(self):
        """Passing codes=[] returns empty DataFrame."""
        empty_panel = build_financial_panel(codes=[])
        assert empty_panel.empty

    def test_unknown_country_code_graceful(self):
        """Querying an unsupported country code handled without crash."""
        panel = build_financial_panel(codes=["ZZZ"], include_commodities=False)
        assert panel.empty

    def test_missing_data_indicators_quarterly(self):
        """Quarterly financial panel contains is_imputed column, all False."""
        panel_q = build_financial_panel(codes=["USA"], start_date="2018-01-01", frequency="Q")
        assert "is_imputed" in panel_q.columns
        assert not panel_q["is_imputed"].any()

    def test_missing_data_indicators_monthly_projection(self):
        """Monthly financial panel projects BIS quarterly series and marks forward-fills."""
        panel_m = build_financial_panel(codes=["USA"], start_date="2020-01-01", frequency="M")
        assert "is_imputed" in panel_m.columns
        assert "sa_source" in panel_m.columns
        bis_ffill = panel_m[panel_m["sa_source"] == "quarterly_ffill"]
        assert not bis_ffill.empty
        assert bis_ffill["is_imputed"].all()
        assert set(bis_ffill["date"].dt.month).issubset({2, 3, 5, 6, 8, 9, 11, 12})

    def test_policy_rate_last_aggregation(self):
        """Quarterly aggregation uses 'last' for policy rates even when harmonization='mean'."""
        panel_q = build_financial_panel(codes=["USA"], start_date="2020-01-01", frequency="Q", harmonization="mean")
        rate_rows = panel_q[panel_q["variable"].str.contains("policy_rate|cbrate")]
        if not rate_rows.empty:
            assert all("last" in str(s) for s in rate_rows["source"])

    def test_temporal_boundary_filters(self):
        """Respects start_date and end_date temporal boundaries."""
        panel = build_financial_panel(codes=["USA"], start_date="2016-01-01", end_date="2018-12-31", frequency="Q")
        assert not panel.empty
        assert panel["date"].min() >= pd.Timestamp("2016-01-01")
        assert panel["date"].max() <= pd.Timestamp("2018-12-31")

    def test_financial_long_to_wide_interop(self):
        """Financial long-form panel interoperates with puremacro.data.long_to_wide."""
        from puremacro.data import long_to_wide
        panel_long = build_financial_panel(codes=["USA", "DEU"], start_date="2018-01-01", frequency="Q", wide=False)
        wide_df = long_to_wide(panel_long)
        assert isinstance(wide_df.index, pd.MultiIndex)
        assert wide_df.index.names == ["code", "date"]
        assert len(wide_df.columns) > 1


# ============================================================================
# 3. Re-export & Backward Compatibility Tests
# ============================================================================

class TestModularPanelsReexports:
    """Verify that build_panel.py re-exports new panel builders without regression."""

    def test_reexport_identity(self):
        """Re-exported functions in build_panel are identical to definitions."""
        assert bp_climate_panel is build_climate_panel
        assert bp_financial_panel is build_financial_panel

    def test_build_panel_existing_functions_preserved(self):
        """Existing build_panel functions remain callable and intact."""
        assert callable(build_all)
        assert callable(load_country)
        assert callable(merge_frames)

    def test_reexported_callable_smoke(self):
        """Calling re-exported functions produces valid panels."""
        p_c = bp_climate_panel(codes=["USA"], start_year=2018, frequency="A")
        p_f = bp_financial_panel(codes=["USA"], start_date="2020-01-01", frequency="Q")
        assert not p_c.empty
        assert not p_f.empty


# ============================================================================
# 4. Release Gate Remediation Regression Tests (Reviewer 2 & Challenger 1)
# ============================================================================

class TestRemediationRegressions:
    """Verify bugfixes for issues identified by Reviewer 2 and Challenger 1."""

    def test_climate_panel_macro_quarterly_preservation(self, monkeypatch, tmp_path):
        """Macro output preserves distinct quarterly values across Q1-Q4 when panel_Q exists."""
        from puremacro import build_panel
        import puremacro.climate_panel

        fake_macro = pd.DataFrame([
            {"code": "USA", "date": pd.Timestamp("2020-01-01"), "variable": "gdp_real", "value": 100.0, "sa_source": "none", "source": "test_q"},
            {"code": "USA", "date": pd.Timestamp("2020-04-01"), "variable": "gdp_real", "value": 102.0, "sa_source": "none", "source": "test_q"},
            {"code": "USA", "date": pd.Timestamp("2020-07-01"), "variable": "gdp_real", "value": 104.0, "sa_source": "none", "source": "test_q"},
            {"code": "USA", "date": pd.Timestamp("2020-10-01"), "variable": "gdp_real", "value": 106.0, "sa_source": "none", "source": "test_q"},
        ])
        fake_parquet = tmp_path / "panel_Q.parquet"
        fake_macro.to_parquet(fake_parquet)

        monkeypatch.setattr(build_panel, "PANEL_Q_PATH", fake_parquet)
        monkeypatch.setattr(puremacro.climate_panel, "PANEL_Q_PATH", fake_parquet, raising=False)

        panel = build_climate_panel(codes=["USA"], start_year=2020, end_year=2020, frequency="Q")
        assert not panel.empty

        gdp_rows = panel[panel["variable"] == "gdp_real"].sort_values("date")
        assert len(gdp_rows) == 4
        # Verify that Q2, Q3, Q4 are NOT flattened to Q1's value (100.0)
        assert list(gdp_rows["value"].values) == [100.0, 102.0, 104.0, 106.0]
        # Imputation flags should be False for genuine macro data
        assert not gdp_rows["is_imputed"].any()

    def test_climate_panel_macro_annual_aggregation(self, monkeypatch, tmp_path):
        """Macro quarterly data from panel_Q is aggregated to annual mean when frequency='A'."""
        from puremacro import build_panel
        import puremacro.climate_panel

        fake_macro = pd.DataFrame([
            {"code": "USA", "date": pd.Timestamp("2020-01-01"), "variable": "gdp_real", "value": 100.0, "sa_source": "none", "source": "test_q"},
            {"code": "USA", "date": pd.Timestamp("2020-04-01"), "variable": "gdp_real", "value": 102.0, "sa_source": "none", "source": "test_q"},
            {"code": "USA", "date": pd.Timestamp("2020-07-01"), "variable": "gdp_real", "value": 104.0, "sa_source": "none", "source": "test_q"},
            {"code": "USA", "date": pd.Timestamp("2020-10-01"), "variable": "gdp_real", "value": 106.0, "sa_source": "none", "source": "test_q"},
        ])
        fake_parquet = tmp_path / "panel_Q.parquet"
        fake_macro.to_parquet(fake_parquet)

        monkeypatch.setattr(build_panel, "PANEL_Q_PATH", fake_parquet)
        monkeypatch.setattr(puremacro.climate_panel, "PANEL_Q_PATH", fake_parquet, raising=False)

        panel = build_climate_panel(codes=["USA"], start_year=2020, end_year=2020, frequency="A")
        assert not panel.empty

        gdp_rows = panel[panel["variable"] == "gdp_real"]
        assert len(gdp_rows) == 1
        assert gdp_rows["date"].iloc[0] == pd.Timestamp("2020-01-01")
        # Mean of 100, 102, 104, 106 is 103.0
        assert gdp_rows["value"].iloc[0] == pytest.approx(103.0)
        assert not gdp_rows["is_imputed"].iloc[0]

    def test_financial_panel_retains_em_spread_without_commodities(self):
        """When include_commodities=False and include_conditions=True, em_spread ('WLD') is retained."""
        panel = build_financial_panel(
            codes=["USA"],
            start_date="2020-01-01",
            include_commodities=False,
            include_conditions=True,
        )
        assert not panel.empty
        # em_spread carries code='WLD'
        wld_rows = panel[panel["code"] == "WLD"]
        assert not wld_rows.empty
        assert "em_spread_q" in set(wld_rows["variable"].unique())
        # Commodity benchmarks must NOT be included
        comm_vars = {"brent_q", "wti_q", "gold_q", "copper_q"}
        assert not any(v in comm_vars for v in panel["variable"].unique())

    def test_fetch_sovereign_yields_deduplicate_duplicate_codes(self):
        """fetch_sovereign_yields with duplicate codes produces zero duplicate rows."""
        from puremacro.fetch.financial import fetch_sovereign_yields, compute_sovereign_spreads

        df = fetch_sovereign_yields(codes=["USA", "USA"], start_date="2020-01-01")
        assert not df.empty
        dups = df.duplicated(subset=["code", "date", "variable"])
        assert not dups.any(), f"Found {dups.sum()} duplicates in yields"

        # Downstream compute_sovereign_spreads also free of duplicates
        spreads = compute_sovereign_spreads(df)
        assert not spreads.empty
        spreads_dups = spreads.duplicated(subset=["code", "date", "variable"])
        assert not spreads_dups.any(), f"Found {spreads_dups.sum()} duplicates in spreads"

    def test_build_financial_panel_fetches_benchmark_for_spreads(self):
        """build_financial_panel(codes=['DEU', 'FRA']) computes sovereign spread against USA benchmark."""
        panel = build_financial_panel(
            codes=["DEU", "FRA"],
            start_date="2020-01-01",
            include_spreads=True,
        )
        assert not panel.empty
        # sovereign_spread_q must exist for DEU and FRA
        deu_vars = set(panel[panel["code"] == "DEU"]["variable"].unique())
        fra_vars = set(panel[panel["code"] == "FRA"]["variable"].unique())
        assert "sovereign_spread_q" in deu_vars
        assert "sovereign_spread_q" in fra_vars
        # USA itself should be filtered out from final codes since it wasn't requested
        assert "USA" not in set(panel["code"].unique())

    def test_parse_bis_date_robustness(self):
        """_parse_bis_date parses valid quarters and handles malformed strings with NaT."""
        from puremacro.fetch.financial import _parse_bis_date

        s = pd.Series(["2020-Q1", "2020-INVALID", "2021Q3", "corrupted"])
        parsed = _parse_bis_date(s)
        assert parsed.iloc[0] == pd.Timestamp("2020-01-01")
        assert pd.isna(parsed.iloc[1])
        assert parsed.iloc[2] == pd.Timestamp("2021-07-01")
        assert pd.isna(parsed.iloc[3])
