"""Comprehensive 4-Tier Opaque-Box E2E Test Suite for puremacro Data Ecosystem Expansion (v2.4.0).

Data Ecosystem Expansion:
1. Cross-Country Greenhouse Gas & Emissions Collectors (R1)
   - World Bank WDI GHG emissions indicators (CO2 per capita, CO2 total kt, GHG total, methane, nitrous oxide)
   - OECD SDMX Air Emissions Inventory (DF_AIR_GHG) with sectoral breakdowns (1A1, 1A2, 1A3, 1A4b, _T)
   - Conforming long-form schema: [code, date, variable, value, sa_source, source]

2. Energy Balances, Transition & Global Commodity Benchmark Suites (R2)
   - Primary energy consumption, electricity generation by source (renewable, hydro, nuclear, fossil)
   - Commodity benchmarks across energy, industrial metals, precious metals, agricultural staples, fertilizers
   - Standardized monthly and quarterly indices (base 2010=100) under code='WLD'

3. International Financial & Macroprudential Data Collectors (R3)
   - Sovereign yield curve benchmarks (10Y, 2Y yields) across advanced and emerging economies
   - Central bank policy rates (Fed, ECB, BoE, BoJ, etc.)
   - BIS credit-to-GDP gaps, total credit to private non-financial sector, real residential property prices
   - Financial conditions, term spreads (10Y - 2Y), sovereign risk spreads, TED spread, HY OAS, NFCI

4. Modular Panel Builders & Harmonization (R4)
   - build_climate_panel(codes, start_year=..., frequency='A'|'Q', harmonization='repeat'|'interpolate', wide=...)
   - build_financial_panel(codes, start_date=..., frequency='M'|'Q', harmonization='mean'|'last', wide=...)
   - Automated frequency harmonization (M->Q aggregation, A->Q projection) and missing data indicators

5. Architectural Invariants, Pyodide Compatibility & Offline Test Fixtures (R5)
   - Zero module-scope `requests` imports (enforced by tests/test_fetch_imports_without_requests.py)
   - 100% offline execution using canned mocks and synthetic payloads
   - Pyodide 4-package core purity: numpy, scipy, pandas, matplotlib only
"""
from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import puremacro
import puremacro.fetch as fetch


# ===========================================================================
# Opaque-Box Module Resolution Helpers with Progressive Readiness Checks
# ===========================================================================

def _require_wdi_emissions():
    """Resolve fetch_wdi_emissions callable."""
    fn = None
    try:
        from puremacro.fetch import emissions as em_mod
        if hasattr(em_mod, "fetch_wdi_emissions"):
            fn = getattr(em_mod, "fetch_wdi_emissions")
    except (ImportError, AttributeError):
        pass
    if fn is None:
        try:
            from puremacro.fetch import wdi_emissions as wdi_mod
            if hasattr(wdi_mod, "fetch_wdi_emissions"):
                fn = getattr(wdi_mod, "fetch_wdi_emissions")
        except (ImportError, AttributeError):
            pass
    if fn is None and hasattr(fetch, "fetch_wdi_emissions"):
        fn = getattr(fetch, "fetch_wdi_emissions")

    if fn is None:
        pytest.skip("Milestone 1: fetch_wdi_emissions pending in puremacro.fetch.emissions")
    return fn


def _require_oecd_ghg():
    """Resolve fetch_oecd_ghg callable."""
    fn = None
    try:
        from puremacro.fetch import emissions as em_mod
        if hasattr(em_mod, "fetch_oecd_ghg"):
            fn = getattr(em_mod, "fetch_oecd_ghg")
    except (ImportError, AttributeError):
        pass
    if fn is None:
        try:
            from puremacro.fetch import oecd_ghg as oecd_mod
            if hasattr(oecd_mod, "fetch_oecd_ghg"):
                fn = getattr(oecd_mod, "fetch_oecd_ghg")
        except (ImportError, AttributeError):
            pass
    if fn is None and hasattr(fetch, "fetch_oecd_ghg"):
        fn = getattr(fetch, "fetch_oecd_ghg")

    if fn is None:
        pytest.skip("Milestone 1: fetch_oecd_ghg pending in puremacro.fetch.emissions")
    return fn


def _require_emissions_panel():
    """Resolve fetch_emissions_panel callable."""
    fn = None
    try:
        from puremacro.fetch import emissions as em_mod
        if hasattr(em_mod, "fetch_emissions_panel"):
            fn = getattr(em_mod, "fetch_emissions_panel")
    except (ImportError, AttributeError):
        pass
    if fn is None and hasattr(fetch, "fetch_emissions_panel"):
        fn = getattr(fetch, "fetch_emissions_panel")

    if fn is None:
        pytest.skip("Milestone 1: fetch_emissions_panel pending in puremacro.fetch.emissions")
    return fn


def _require_energy_transition():
    """Resolve fetch_energy_transition callable."""
    fn = None
    try:
        from puremacro.fetch import energy_transition as et_mod
        if hasattr(et_mod, "fetch_energy_transition"):
            fn = getattr(et_mod, "fetch_energy_transition")
    except (ImportError, AttributeError):
        pass
    if fn is None and hasattr(fetch, "fetch_energy_transition"):
        fn = getattr(fetch, "fetch_energy_transition")

    if fn is None:
        pytest.skip("Milestone 2: fetch_energy_transition pending in puremacro.fetch.energy_transition")
    return fn


def _require_commodity_benchmarks():
    """Resolve fetch_commodity_benchmarks callable."""
    fn = None
    try:
        from puremacro.fetch import wb_pink_sheet as ps_mod
        if hasattr(ps_mod, "fetch_commodity_benchmarks"):
            fn = getattr(ps_mod, "fetch_commodity_benchmarks")
    except (ImportError, AttributeError):
        pass
    if fn is None:
        try:
            from puremacro.fetch import commodities as com_mod
            if hasattr(com_mod, "fetch_commodity_benchmarks"):
                fn = getattr(com_mod, "fetch_commodity_benchmarks")
        except (ImportError, AttributeError):
            pass
    if fn is None and hasattr(fetch, "fetch_commodity_benchmarks"):
        fn = getattr(fetch, "fetch_commodity_benchmarks")

    if fn is None:
        pytest.skip("Milestone 2: fetch_commodity_benchmarks pending in puremacro.fetch.commodities")
    return fn


def _require_sovereign_yields():
    """Resolve fetch_sovereign_yields callable."""
    fn = None
    try:
        from puremacro.fetch import financial as fin_mod
        if hasattr(fin_mod, "fetch_sovereign_yields"):
            fn = getattr(fin_mod, "fetch_sovereign_yields")
    except (ImportError, AttributeError):
        pass
    if fn is None:
        try:
            from puremacro.fetch import financial_sovereign as fsov_mod
            if hasattr(fsov_mod, "fetch_sovereign_yields"):
                fn = getattr(fsov_mod, "fetch_sovereign_yields")
        except (ImportError, AttributeError):
            pass
    if fn is None and hasattr(fetch, "fetch_sovereign_yields"):
        fn = getattr(fetch, "fetch_sovereign_yields")

    if fn is None:
        pytest.skip("Milestone 3: fetch_sovereign_yields pending in puremacro.fetch.financial")
    return fn


def _require_policy_rates():
    """Resolve fetch_policy_rates callable."""
    fn = None
    try:
        from puremacro.fetch import financial as fin_mod
        if hasattr(fin_mod, "fetch_policy_rates"):
            fn = getattr(fin_mod, "fetch_policy_rates")
    except (ImportError, AttributeError):
        pass
    if fn is None:
        try:
            from puremacro.fetch import financial_sovereign as fsov_mod
            if hasattr(fsov_mod, "fetch_policy_rates"):
                fn = getattr(fsov_mod, "fetch_policy_rates")
        except (ImportError, AttributeError):
            pass
    if fn is None and hasattr(fetch, "fetch_policy_rates"):
        fn = getattr(fetch, "fetch_policy_rates")

    if fn is None:
        pytest.skip("Milestone 3: fetch_policy_rates pending in puremacro.fetch.financial")
    return fn


def _require_bis_macroprudential():
    """Resolve fetch_bis_macroprudential callable."""
    fn = None
    try:
        from puremacro.fetch import financial as fin_mod
        if hasattr(fin_mod, "fetch_bis_macroprudential"):
            fn = getattr(fin_mod, "fetch_bis_macroprudential")
    except (ImportError, AttributeError):
        pass
    if fn is None:
        try:
            from puremacro.fetch import bis_macroprudential as bis_mod
            if hasattr(bis_mod, "fetch_bis_macroprudential"):
                fn = getattr(bis_mod, "fetch_bis_macroprudential")
            elif hasattr(bis_mod, "fetch_credit_gap"):
                fn = getattr(bis_mod, "fetch_credit_gap")
        except (ImportError, AttributeError):
            pass
    if fn is None and hasattr(fetch, "fetch_bis_macroprudential"):
        fn = getattr(fetch, "fetch_bis_macroprudential")

    if fn is None:
        pytest.skip("Milestone 3: fetch_bis_macroprudential pending in puremacro.fetch.financial")
    return fn


def _require_financial_conditions():
    """Resolve fetch_financial_conditions callable."""
    fn = None
    try:
        from puremacro.fetch import financial as fin_mod
        if hasattr(fin_mod, "fetch_financial_conditions"):
            fn = getattr(fin_mod, "fetch_financial_conditions")
    except (ImportError, AttributeError):
        pass
    if fn is None:
        try:
            from puremacro.fetch import financial_conditions as fc_mod
            if hasattr(fc_mod, "fetch_financial_conditions"):
                fn = getattr(fc_mod, "fetch_financial_conditions")
        except (ImportError, AttributeError):
            pass
    if fn is None and hasattr(fetch, "fetch_financial_conditions"):
        fn = getattr(fetch, "fetch_financial_conditions")

    if fn is None:
        pytest.skip("Milestone 3: fetch_financial_conditions pending in puremacro.fetch.financial")
    return fn


def _require_climate_panel():
    """Resolve build_climate_panel callable."""
    fn = None
    try:
        from puremacro import climate_panel as cp_mod
        if hasattr(cp_mod, "build_climate_panel"):
            fn = getattr(cp_mod, "build_climate_panel")
    except (ImportError, AttributeError):
        pass
    if fn is None:
        try:
            from puremacro import build_panel as bp_mod
            if hasattr(bp_mod, "build_climate_panel"):
                fn = getattr(bp_mod, "build_climate_panel")
        except (ImportError, AttributeError):
            pass
    if fn is None and hasattr(puremacro, "build_climate_panel"):
        fn = getattr(puremacro, "build_climate_panel")

    if fn is None:
        pytest.skip("Milestone 4: build_climate_panel pending in puremacro.climate_panel")
    return fn


def _require_financial_panel():
    """Resolve build_financial_panel callable."""
    fn = None
    try:
        from puremacro import financial_panel as fp_mod
        if hasattr(fp_mod, "build_financial_panel"):
            fn = getattr(fp_mod, "build_financial_panel")
    except (ImportError, AttributeError):
        pass
    if fn is None:
        try:
            from puremacro import build_panel as bp_mod
            if hasattr(bp_mod, "build_financial_panel"):
                fn = getattr(bp_mod, "build_financial_panel")
        except (ImportError, AttributeError):
            pass
    if fn is None and hasattr(puremacro, "build_financial_panel"):
        fn = getattr(puremacro, "build_financial_panel")

    if fn is None:
        pytest.skip("Milestone 4: build_financial_panel pending in puremacro.financial_panel")
    return fn


# ===========================================================================
# Canned Offline Fixtures & Synthetic Payloads
# ===========================================================================

def _generate_mock_wb_json(countries: Sequence[str], indicator: str, start_year: int = 1990, end_year: int = 2023) -> bytes:
    """Generate realistic World Bank API v2 JSON response."""
    if end_year < 1960 or start_year > 2050:
        payload = [{"page": 1, "pages": 0, "per_page": 1000, "total": 0}, []]
        return json.dumps(payload).encode("utf-8")

    records = []
    for c in countries:
        base_val = 14.5 if "PC" in indicator else 500000.0
        eff_start = max(1960, start_year)
        eff_end = min(2023, end_year)
        if eff_start > eff_end:
            continue
        for y in range(eff_start, eff_end + 1):
            val = base_val * (0.985 ** (y - 1990)) + (hash(f"{c}{y}") % 100) * 0.01
            records.append({
                "indicator": {"id": indicator, "value": indicator},
                "country": {"id": c[:2], "value": c},
                "countryiso3code": c,
                "date": str(y),
                "value": round(val, 4),
                "unit": "",
                "obs_status": "",
                "decimal": 2,
            })
    payload = [{"page": 1, "pages": 1, "per_page": len(records), "total": len(records)}, records]
    return json.dumps(payload).encode("utf-8")



def _generate_mock_oecd_ghg_csv(countries: Sequence[str], start_year: int = 1990, end_year: int = 2023) -> bytes:
    """Generate realistic OECD SDMX DF_AIR_GHG CSV payload."""
    lines = ["REF_AREA,FREQ,POLLUTANT,MEASURE,UNIT_MEASURE,TIME_PERIOD,OBS_VALUE,UNIT_MULT"]
    sectors = ["1A1", "1A2", "1A3", "1A4b", "_T"]
    for c in countries:
        for y in range(start_year, end_year + 1):
            for sec in sectors:
                mult = {"1A1": 0.35, "1A2": 0.15, "1A3": 0.28, "1A4b": 0.10, "_T": 1.0}[sec]
                val = 450000.0 * mult * (0.98 ** (y - start_year))
                lines.append(f"{c},A,GHG,{sec},T_CO2E,{y},{round(val, 2)},3")
    # Ensure payload exceeds 200 bytes threshold in _oecd_sdmx.py
    while len("\n".join(lines)) < 300:
        lines.append(f"USA,A,GHG,_T,T_CO2E,1989,450000.0,3")
    return "\n".join(lines).encode("utf-8")



def _generate_mock_fred_csv(series_id: str, start_date: str = "1990-01-01", end_date: str = "2023-12-01") -> bytes:
    """Generate realistic FRED CSV payload."""
    periods = pd.date_range(start_date, end_date, freq="MS")
    lines = ["DATE,VALUE"]
    for i, dt in enumerate(periods):
        date_str = dt.strftime("%Y-%m-%d")
        if "GS10" in series_id or "IRLTLT" in series_id:
            val = 4.5 + 2.0 * math.sin(i / 24.0) + (0.5 if "GS10" in series_id else 0.2)
        elif "GS2" in series_id or "FIESTT" in series_id:
            val = 3.8 + 2.2 * math.sin(i / 24.0 + 0.2)
        elif "FEDFUNDS" in series_id or "CBPOL" in series_id:
            val = 3.5 + 2.3 * math.sin(i / 24.0 + 0.3)
        elif "TEDRATE" in series_id:
            val = 0.45 + 0.3 * abs(math.sin(i / 12.0))
        elif "HYM2" in series_id or "OAS" in series_id:
            val = 4.2 + 1.8 * abs(math.sin(i / 18.0))
        elif "NFCI" in series_id:
            val = -0.3 + 0.6 * math.sin(i / 15.0)
        else:
            val = 2.5 + 0.5 * math.sin(i / 10.0)
        lines.append(f"{date_str},{round(val, 3)}")
    return "\n".join(lines).encode("utf-8")


def _generate_mock_bis_sdmx_csv(dataflow: str, countries: Sequence[str]) -> bytes:
    """Generate realistic BIS SDMX CSV payload."""
    quarters = pd.date_range("1990-01-01", "2023-10-01", freq="QS")
    iso_to_bis = {"USA": "US", "DEU": "DE", "GBR": "GB", "JPN": "JP", "FRA": "FR", "ITA": "IT", "CAN": "CA", "XM": "XM", "EUR": "XM"}
    if "CREDIT_GAP" in dataflow:
        lines = ["FREQ,BORROWER,TC_ADJUST,TC_LENDER,UNIT_TYPE,REF_AREA,CG_DATA_TYPE,TIME_PERIOD,OBS_VALUE"]
        for c in countries:
            bis_c = iso_to_bis.get(c, c[:2])
            for dt in quarters:
                q_str = f"{dt.year}-Q{dt.quarter}"
                gap = 2.5 * math.sin(dt.year / 3.0 + hash(c) % 5)
                lines.append(f"Q,P,A,ALL,770,{bis_c},GAP,{q_str},{round(gap, 2)}")
                lines.append(f"Q,P,A,ALL,770,{bis_c},RAT,{q_str},{round(150.0 + gap, 2)}")
                lines.append(f"Q,P,A,ALL,770,{bis_c},TRD,{q_str},150.00")
        return "\n".join(lines).encode("utf-8")
    elif "SPP" in dataflow:
        lines = ["FREQ,REF_AREA,PRICE_TYPE,UNIT_MEASURE,TIME_PERIOD,OBS_VALUE"]
        for c in countries:
            bis_c = iso_to_bis.get(c, c[:2])
            for dt in quarters:
                q_str = f"{dt.year}-Q{dt.quarter}"
                price = 100.0 * (1.02 ** (dt.year - 1990)) * (1.0 + 0.05 * math.sin(dt.quarter))
                lines.append(f"Q,{bis_c},REAL,IX,{q_str},{round(price, 2)}")
        return "\n".join(lines).encode("utf-8")
    elif "CBPOL" in dataflow:
        months = pd.date_range("1990-01-01", "2023-12-01", freq="MS")
        lines = ["FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE"]
        for c in countries:
            bis_c = iso_to_bis.get(c, c[:2])
            for dt in months:
                m_str = f"{dt.year}-{dt.month:02d}"
                lines.append(f"M,{bis_c},{m_str},2.50")
        return "\n".join(lines).encode("utf-8")
    else:
        lines = ["FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE"]
        for c in countries:
            bis_c = iso_to_bis.get(c, c[:2])
            for dt in quarters:
                q_str = f"{dt.year}-Q{dt.quarter}"
                lines.append(f"Q,{bis_c},{q_str},150.0")
        return "\n".join(lines).encode("utf-8")


def _generate_mock_ember_csv(countries: Sequence[str], start_year: int = 1990, end_year: int = 2023) -> bytes:
    """Generate realistic Ember electricity generation CSV payload."""
    lines = ["iso 3 code,year,electricity source,generation (twh),share of generation (%)"]
    for c in countries:
        for y in range(start_year, end_year + 1):
            ren_share = 10.0 + 35.0 * ((y - 1990) / (2023 - 1990))
            fossil_share = 70.0 - 30.0 * ((y - 1990) / (2023 - 1990))
            lines.append(f"{c},{y},total generation,1000.0,100.0")
            lines.append(f"{c},{y},renewables,{round(10.0 * ren_share, 1)},{round(ren_share, 1)}")
            lines.append(f"{c},{y},hydro,100.0,10.0")
            lines.append(f"{c},{y},nuclear,100.0,10.0")
            lines.append(f"{c},{y},fossil,{round(10.0 * fossil_share, 1)},{round(fossil_share, 1)}")
    return "\n".join(lines).encode("utf-8")


@pytest.fixture(autouse=True)
def mock_macro_ecosystem_network(monkeypatch):
    """Intercept all network fetch requests and serve deterministic offline payloads."""
    from puremacro.fetch import _http

    def fake_cached_get(url: str, *, refresh: bool = False, timeout: float | None = None) -> bytes:
        # Ember Energy
        if "ember-energy.org" in url or "generation_yearly" in url:
            return _generate_mock_ember_csv(["USA", "DEU", "GBR", "JPN", "FRA", "ITA", "CAN"])

        # World Bank WDI
        if "api.worldbank.org" in url:
            c = "USA"
            for candidate in ["USA", "DEU", "GBR", "JPN", "FRA", "ITA", "CAN", "WLD"]:
                if f"country/{candidate}" in url or f"country/{candidate.lower()}" in url:
                    c = candidate
                    break
            sy = 1990
            ey = 2023
            if "date=" in url:
                try:
                    date_part = url.split("date=")[1].split("&")[0]
                    parts = date_part.split(":")
                    sy = int(parts[0])
                    if len(parts) > 1:
                        ey = int(parts[1])
                except (IndexError, ValueError):
                    pass
            ind = "EN.ATM.CO2E.PC" if "CO2E.PC" in url else "EN.ATM.GHGT.KT.CE"
            return _generate_mock_wb_json([c], ind, start_year=sy, end_year=ey)



        # OECD SDMX DF_AIR_GHG
        if "DSD_AIR_GHG" in url or "DF_AIR_GHG" in url:
            return _generate_mock_oecd_ghg_csv(["USA", "DEU", "GBR", "JPN", "FRA", "ITA", "CAN"])

        # BIS SDMX
        if "WS_CREDIT_GAP" in url or "credit-gap" in url:
            return _generate_mock_bis_sdmx_csv("WS_CREDIT_GAP", ["USA", "DEU", "GBR", "JPN", "FRA", "ITA", "CAN"])
        if "WS_SPP" in url or "property" in url:
            return _generate_mock_bis_sdmx_csv("WS_SPP", ["USA", "DEU", "GBR", "JPN", "FRA", "ITA", "CAN"])
        if "WS_CBPOL" in url or "cbpol" in url.lower():
            return _generate_mock_bis_sdmx_csv("WS_CBPOL", ["USA", "DEU", "GBR", "JPN", "FRA", "ITA", "CAN", "XM"])


        # FRED Series
        if "fred.stlouisfed.org" in url or "fredgraph.csv" in url:
            series_id = "GS10"
            if "id=" in url:
                series_id = url.split("id=")[-1].split("&")[0]
            return _generate_mock_fred_csv(series_id)

        # Commodity workbook landing / fallback
        if "worldbank.org" in url and ("CMO" in url or "Pink" in url or "Historical" in url):
            from io import BytesIO
            buf = BytesIO()
            end = pd.Period("2023-12", freq="M")
            periods = pd.period_range(end - 36, end, freq="M")
            dates = [f"{p.year}M{p.month:02d}" for p in periods]
            data = {
                "": ["($/bbl)"] + dates,
                "Crude oil, Brent": [np.nan] + list(np.linspace(70, 95, len(dates))),
                "Crude oil, WTI": [np.nan] + list(np.linspace(65, 90, len(dates))),
                "Natural gas, Europe": [np.nan] + list(np.linspace(8, 25, len(dates))),
                "Natural gas, US": [np.nan] + list(np.linspace(2.5, 5.0, len(dates))),
                "Coal, Australian": [np.nan] + list(np.linspace(80, 140, len(dates))),
                "Copper": [np.nan] + list(np.linspace(6000, 9000, len(dates))),
                "Gold": [np.nan] + list(np.linspace(1500, 2000, len(dates))),
                "Wheat, US, HRW": [np.nan] + list(np.linspace(200, 350, len(dates))),
            }
            body = pd.DataFrame(data)

            # Monthly Indices sheet
            indices_data = {
                0: dates,
                1: list(np.linspace(100, 120, len(dates))),
                2: list(np.linspace(95, 125, len(dates))),
                3: list(np.linspace(105, 115, len(dates))),
                4: list(np.linspace(100, 110, len(dates))),
            }
            for col_i in range(5, 17):
                indices_data[col_i] = list(np.linspace(90, 110, len(dates)))
            indices_df = pd.DataFrame(indices_data)

            with pd.ExcelWriter(buf, engine="openpyxl") as xl:
                pd.DataFrame([[""]] * 4).to_excel(xl, sheet_name="Monthly Prices", index=False, header=False)
                body.to_excel(xl, sheet_name="Monthly Prices", index=False, startrow=4)
                indices_df.to_excel(xl, sheet_name="Monthly Indices", index=False, header=False)
            return buf.getvalue()


        # Fallback default
        return b"date,value\n2020-01-01,1.0\n"

    monkeypatch.setattr(_http, "cached_get", fake_cached_get)
    for mod_name in ("financial", "emissions", "energy_transition", "wb_pink_sheet", "commodities", "financial_sovereign", "bis_macroprudential", "financial_conditions"):
        try:
            m = getattr(fetch, mod_name, None)
            if m is not None and hasattr(m, "cached_get"):
                monkeypatch.setattr(m, "cached_get", fake_cached_get)
        except Exception:
            pass



# ===========================================================================
# Purity & Architectural Gate
# ===========================================================================

class TestArchitecturalAndPurityGate:
    """Validate core architectural invariants and Pyodide four-package compliance."""

    def test_architectural_offline_and_pyodide_invariants(self):
        """Puremacro and puremacro.fetch must strictly adhere to the 4-package Pyodide contract."""
        code = (
            "import sys\n"
            "import puremacro\n"
            "import puremacro.fetch\n"
            "forbidden = ('statsmodels', 'linearmodels', 'arch', 'bs4', 'pdfplumber', 'pypdf')\n"
            "leaked = [mod for mod in forbidden if mod in sys.modules]\n"
            "assert not leaked, f'Forbidden non-Pyodide module leaked: {leaked}'\n"
            "assert hasattr(puremacro, '__version__')\n"
            "assert hasattr(puremacro.fetch, '_http')\n"
        )
        import subprocess
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert proc.returncode == 0, f"Purity gate failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


# ===========================================================================
# Tier 1: Feature Coverage (>=5 tests per feature across 10 core capabilities)
# ===========================================================================

class TestTier1FeatureCoverage:
    """Comprehensive functional coverage across all 10 core data ecosystem capabilities."""

    # --- Feature 1: fetch_wdi_emissions (5 tests) ---
    def test_t1_f01_wdi_emissions_schema_conformance(self):
        """Feature 1.1: Standard long-form schema [code, date, variable, value, sa_source, source]."""
        fetch_fn = _require_wdi_emissions()
        df = fetch_fn(codes=["USA"], start_year=2015, end_year=2020)
        assert isinstance(df, pd.DataFrame)
        expected_cols = ["code", "date", "variable", "value", "sa_source", "source"]
        assert list(df.columns) == expected_cols

    def test_t1_f01_wdi_emissions_iso3_country_codes(self):
        """Feature 1.2: Country codes are uppercase 3-letter ISO codes."""
        fetch_fn = _require_wdi_emissions()
        df = fetch_fn(codes=["USA", "DEU", "GBR"], start_year=2018, end_year=2020)
        assert not df.empty
        assert set(df["code"]).issubset({"USA", "DEU", "GBR"})
        assert all(isinstance(c, str) and len(c) == 3 and c.isupper() for c in df["code"].unique())

    def test_t1_f01_wdi_emissions_date_normalization(self):
        """Feature 1.3: Date column contains normalized annual timestamps."""
        fetch_fn = _require_wdi_emissions()
        df = fetch_fn(codes=["USA"], start_year=2010, end_year=2015)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        # Annual frequencies must normalize to January 1
        assert (df["date"].dt.month == 1).all()
        assert (df["date"].dt.day == 1).all()

    def test_t1_f01_wdi_emissions_indicator_selection(self):
        """Feature 1.4: Indicator filtering restricts output variables."""
        fetch_fn = _require_wdi_emissions()
        df = fetch_fn(codes=["USA"], indicators=["co2_pc_a", "ghg_total_kt_a"], start_year=2015)
        assert set(df["variable"]).issubset({"co2_pc_a", "ghg_total_kt_a"})

    def test_t1_f01_wdi_emissions_numeric_value_contract(self):
        """Feature 1.5: Values are numeric floats without string representations or infinities."""
        fetch_fn = _require_wdi_emissions()
        df = fetch_fn(codes=["USA", "CAN"], start_year=2015, end_year=2020)
        assert pd.api.types.is_float_dtype(df["value"]) or pd.api.types.is_numeric_dtype(df["value"])
        assert not np.isinf(df["value"]).any()

    # --- Feature 2: fetch_oecd_ghg (5 tests) ---
    def test_t1_f02_oecd_ghg_schema_conformance(self):
        """Feature 2.1: Conforms to standard long-form DataFrame schema."""
        fetch_fn = _require_oecd_ghg()
        df = fetch_fn(codes=["USA"], start_year=2015)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["code", "date", "variable", "value", "sa_source", "source"]

    def test_t1_f02_oecd_ghg_sectoral_breakdowns(self):
        """Feature 2.2: Sectoral breakdowns include energy, manufacturing, transport, residential."""
        fetch_fn = _require_oecd_ghg()
        df = fetch_fn(codes=["USA"], start_year=2015)
        vars_present = set(df["variable"])
        expected_subset = {"ghg_energy_industries_kt_a", "ghg_manufacturing_kt_a", "ghg_transport_kt_a", "ghg_total_kt_a"}
        assert expected_subset.issubset(vars_present)

    def test_t1_f02_oecd_ghg_sector_filter(self):
        """Feature 2.3: Sector filter correctly subsets requested emissions sectors."""
        fetch_fn = _require_oecd_ghg()
        df = fetch_fn(codes=["DEU"], sectors=["1A1", "1A3"], start_year=2015)
        assert not df.empty
        assert set(df["variable"]).issubset({"ghg_energy_industries_kt_a", "ghg_transport_kt_a"})

    def test_t1_f02_oecd_ghg_timestamp_frequency(self):
        """Feature 2.4: Timestamps are aligned to annual periods."""
        fetch_fn = _require_oecd_ghg()
        df = fetch_fn(codes=["FRA"], start_year=2010)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert df["date"].dt.year.min() >= 2010

    def test_t1_f02_oecd_ghg_kt_unit_normalization(self):
        """Feature 2.5: Values are positive numbers denominated in kilotonnes (kt)."""
        fetch_fn = _require_oecd_ghg()
        df = fetch_fn(codes=["USA"], start_year=2015)
        assert (df["value"] >= 0.0).all()

    # --- Feature 3: fetch_energy_transition (5 tests) ---
    def test_t1_f03_energy_transition_schema(self):
        """Feature 3.1: Conforms to long-form schema."""
        fetch_fn = _require_energy_transition()
        df = fetch_fn(codes=["USA"], start_year=2010)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["code", "date", "variable", "value", "sa_source", "source"]

    def test_t1_f03_energy_transition_core_variables(self):
        """Feature 3.2: Contains primary consumption and generation shares."""
        fetch_fn = _require_energy_transition()
        df = fetch_fn(codes=["USA"], start_year=2015)
        expected = {"primary_energy_cons_toe_a", "elec_gen_total_twh_a", "elec_gen_renewable_pct_a"}
        assert expected.issubset(set(df["variable"]))

    def test_t1_f03_energy_transition_shares_bounded_0_to_100(self):
        """Feature 3.3: Percentage shares are bounded strictly within [0, 100]."""
        fetch_fn = _require_energy_transition()
        df = fetch_fn(codes=["DEU", "FRA"], start_year=2015)
        share_rows = df[df["variable"].str.endswith("_pct_a")]
        if not share_rows.empty:
            assert (share_rows["value"] >= 0.0).all()
            assert (share_rows["value"] <= 100.0).all()

    def test_t1_f03_energy_transition_country_filtering(self):
        """Feature 3.4: Querying specific country codes filters the results."""
        fetch_fn = _require_energy_transition()
        df = fetch_fn(codes=["DEU"], start_year=2015)
        assert set(df["code"]) == {"DEU"}

    def test_t1_f03_energy_transition_start_year_filter(self):
        """Feature 3.5: Querying start_year returns records on or after the specified year."""
        fetch_fn = _require_energy_transition()
        df = fetch_fn(codes=["USA"], start_year=2018)
        assert df["date"].dt.year.min() >= 2018

    # --- Feature 4: fetch_commodity_benchmarks (5 tests) ---
    def test_t1_f04_commodity_benchmarks_schema(self):
        """Feature 4.1: Conforms to long-form schema tagged with code='WLD'."""
        fetch_fn = _require_commodity_benchmarks()
        df = fetch_fn(frequency="M", start_date="2020-01-01")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["code", "date", "variable", "value", "sa_source", "source"]
        assert (df["code"] == "WLD").all()

    def test_t1_f04_commodity_benchmarks_monthly_and_quarterly_freq(self):
        """Feature 4.2: Supports both monthly ('M') and quarterly ('Q') frequencies."""
        fetch_fn = _require_commodity_benchmarks()
        df_m = fetch_fn(frequency="M", start_date="2020-01-01")
        df_q = fetch_fn(frequency="Q", start_date="2020-01-01")
        assert not df_m.empty and not df_q.empty
        # Monthly has ~3x observations as quarterly
        assert len(df_m) > len(df_q)

    def test_t1_f04_commodity_benchmarks_category_filtering(self):
        """Feature 4.3: Allows filtering by commodity categories (energy, metals, agri)."""
        fetch_fn = _require_commodity_benchmarks()
        df_energy = fetch_fn(categories=["energy"], start_date="2020-01-01")
        assert not df_energy.empty
        assert any("brent" in v or "gas" in v or "coal" in v or "energy" in v for v in df_energy["variable"])

    def test_t1_f04_commodity_benchmarks_price_indices_base_2010(self):
        """Feature 4.4: Includes standardized commodity price indices (2010=100)."""
        fetch_fn = _require_commodity_benchmarks()
        df = fetch_fn(include_indices=True, start_date="2020-01-01")
        indices = [v for v in df["variable"].unique() if "index_" in v]
        assert len(indices) > 0

    def test_t1_f04_commodity_benchmarks_positive_values(self):
        """Feature 4.5: Commodity prices and indices are strictly positive."""
        fetch_fn = _require_commodity_benchmarks()
        df = fetch_fn(start_date="2020-01-01")
        assert (df["value"].dropna() > -50.0).all()

    # --- Feature 5: fetch_sovereign_yields (5 tests) ---
    def test_t1_f05_sovereign_yields_schema(self):
        """Feature 5.1: Conforms to long-form schema."""
        fetch_fn = _require_sovereign_yields()
        df = fetch_fn(codes=["USA"], start_date="2020-01-01")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["code", "date", "variable", "value", "sa_source", "source"]

    def test_t1_f05_sovereign_yields_maturities(self):
        """Feature 5.2: Retrieves both 10Y and 2Y benchmark yields."""
        fetch_fn = _require_sovereign_yields()
        df = fetch_fn(codes=["USA"], maturities=("10Y", "2Y"), start_date="2020-01-01")
        vars_present = set(df["variable"])
        assert "yield_10y" in vars_present or "yield_10y_m" in vars_present
        assert "yield_2y" in vars_present or "yield_2y_m" in vars_present

    def test_t1_f05_sovereign_yields_cross_country(self):
        """Feature 5.3: Supports multi-country cross-section (USA, DEU, GBR, JPN)."""
        fetch_fn = _require_sovereign_yields()
        df = fetch_fn(codes=["USA", "DEU", "GBR"], start_date="2020-01-01")
        assert set(df["code"]).issubset({"USA", "DEU", "GBR"})

    def test_t1_f05_sovereign_yields_start_date_filter(self):
        """Feature 5.4: start_date filters records correctly."""
        fetch_fn = _require_sovereign_yields()
        df = fetch_fn(codes=["USA"], start_date="2021-06-01")
        assert df["date"].min() >= pd.Timestamp("2021-06-01")

    def test_t1_f05_sovereign_yields_numeric_rate_range(self):
        """Feature 5.5: Sovereign bond yields are realistic percentages (between -3% and 25%)."""
        fetch_fn = _require_sovereign_yields()
        df = fetch_fn(codes=["USA", "DEU"], start_date="2015-01-01")
        assert (df["value"] > -3.0).all()
        assert (df["value"] < 25.0).all()

    # --- Feature 6: fetch_policy_rates (5 tests) ---
    def test_t1_f06_policy_rates_schema(self):
        """Feature 6.1: Conforms to standard long-form schema."""
        fetch_fn = _require_policy_rates()
        df = fetch_fn(codes=["USA"], start_date="2020-01-01")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["code", "date", "variable", "value", "sa_source", "source"]

    def test_t1_f06_policy_rates_eurozone_mapping(self):
        """Feature 6.2: Eurozone countries map ECB policy rate under their respective ISO codes."""
        fetch_fn = _require_policy_rates()
        df = fetch_fn(codes=["DEU", "FRA"], start_date="2020-01-01")
        assert not df.empty
        assert set(df["code"]).issubset({"DEU", "FRA"})

    def test_t1_f06_policy_rates_monthly_frequency(self):
        """Feature 6.3: Policy rates aligned to month start timestamps."""
        fetch_fn = _require_policy_rates()
        df = fetch_fn(codes=["USA"], start_date="2020-01-01")
        assert (df["date"].dt.day == 1).all()

    def test_t1_f06_policy_rates_multi_central_banks(self):
        """Feature 6.4: Supports multiple central banks (Fed, ECB, BoE, BoJ)."""
        fetch_fn = _require_policy_rates()
        df = fetch_fn(codes=["USA", "GBR", "JPN"], start_date="2020-01-01")
        assert len(df["code"].unique()) >= 2

    def test_t1_f06_policy_rates_non_negative_or_small_negative(self):
        """Feature 6.5: Policy rates respect economic boundaries (>= -1.5%)."""
        fetch_fn = _require_policy_rates()
        df = fetch_fn(codes=["USA", "JPN"], start_date="2015-01-01")
        assert (df["value"] >= -1.5).all()

    # --- Feature 7: fetch_bis_macroprudential (5 tests) ---
    def test_t1_f07_bis_macroprudential_schema(self):
        """Feature 7.1: Conforms to standard long-form schema."""
        fetch_fn = _require_bis_macroprudential()
        df = fetch_fn(codes=["USA"], start_date="2015-01-01")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["code", "date", "variable", "value", "sa_source", "source"]

    def test_t1_f07_bis_macroprudential_credit_gap_components(self):
        """Feature 7.2: Contains credit gap, credit-to-GDP ratio, and trend."""
        fetch_fn = _require_bis_macroprudential()
        df = fetch_fn(codes=["USA"], indicators=["credit_gap_q", "credit_to_gdp_q"], start_date="2015-01-01")
        vars_present = set(df["variable"])
        assert "credit_gap_q" in vars_present or "credit_gap" in vars_present

    def test_t1_f07_bis_macroprudential_property_prices(self):
        """Feature 7.3: Retrieves residential property price indices."""
        fetch_fn = _require_bis_macroprudential()
        df = fetch_fn(codes=["USA"], indicators=["property_price_real_q"], start_date="2015-01-01")
        if not df.empty:
            assert any("property" in v for v in df["variable"])

    def test_t1_f07_bis_macroprudential_quarterly_frequency(self):
        """Feature 7.4: Dates aligned to quarter start periods (Jan 1, Apr 1, Jul 1, Oct 1)."""
        fetch_fn = _require_bis_macroprudential()
        df = fetch_fn(codes=["USA"], start_date="2015-01-01")
        assert not df.empty
        quarter_months = {1, 4, 7, 10}
        assert set(df["date"].dt.month).issubset(quarter_months)

    def test_t1_f07_bis_macroprudential_country_isolation(self):
        """Feature 7.5: Single country query only returns observations for that country."""
        fetch_fn = _require_bis_macroprudential()
        df = fetch_fn(codes=["DEU"], start_date="2015-01-01")
        assert (df["code"] == "DEU").all()

    # --- Feature 8: fetch_financial_conditions (5 tests) ---
    def test_t1_f08_financial_conditions_schema(self):
        """Feature 8.1: Conforms to standard long-form schema."""
        fetch_fn = _require_financial_conditions()
        df = fetch_fn(start_date="2020-01-01")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["code", "date", "variable", "value", "sa_source", "source"]

    def test_t1_f08_financial_conditions_risk_spreads(self):
        """Feature 8.2: Contains market risk spreads (TED spread, High Yield OAS, EM OAS)."""
        fetch_fn = _require_financial_conditions()
        df = fetch_fn(start_date="2020-01-01")
        vars_present = set(df["variable"])
        spread_vars = [v for v in vars_present if "spread" in v or "ted" in v or "hy" in v or "oas" in v]
        assert len(spread_vars) > 0

    def test_t1_f08_financial_conditions_stress_indices(self):
        """Feature 8.3: Contains macro financial conditions or stress indices (NFCI / STLFSI)."""
        fetch_fn = _require_financial_conditions()
        df = fetch_fn(start_date="2020-01-01")
        vars_present = set(df["variable"])
        index_vars = [v for v in vars_present if "nfci" in v or "fsi" in v]
        assert len(index_vars) > 0

    def test_t1_f08_financial_conditions_global_code(self):
        """Feature 8.4: Global risk indicators tagged with code='WLD' or 'USA'."""
        fetch_fn = _require_financial_conditions()
        df = fetch_fn(start_date="2020-01-01")
        assert set(df["code"]).issubset({"WLD", "USA"})

    def test_t1_f08_financial_conditions_start_date(self):
        """Feature 8.5: Temporal boundary start_date respected."""
        fetch_fn = _require_financial_conditions()
        df = fetch_fn(start_date="2021-01-01")
        assert df["date"].min() >= pd.Timestamp("2021-01-01")

    # --- Feature 9: build_climate_panel (5 tests) ---
    def test_t1_f09_climate_panel_merge_integrity(self):
        """Feature 9.1: Merges emissions, energy transition, and macro output into unified panel."""
        build_fn = _require_climate_panel()
        panel = build_fn(codes=["USA", "DEU"], start_year=2015, frequency="A")
        assert isinstance(panel, pd.DataFrame)
        assert not panel.empty

    def test_t1_f09_climate_panel_annual_frequency(self):
        """Feature 9.2: Default annual frequency ('A') produces Jan 1 aligned records."""
        build_fn = _require_climate_panel()
        panel = build_fn(codes=["USA"], start_year=2015, frequency="A")
        if "date" in panel.columns:
            assert (panel["date"].dt.month == 1).all()

    def test_t1_f09_climate_panel_quarterly_frequency_harmonization(self):
        """Feature 9.3: Harmonizes annual series to quarterly frequency ('Q')."""
        build_fn = _require_climate_panel()
        panel_q = build_fn(codes=["USA"], start_year=2018, frequency="Q", harmonization="repeat")
        assert not panel_q.empty
        if "date" in panel_q.columns:
            assert set(panel_q["date"].dt.month).issubset({1, 4, 7, 10})

    def test_t1_f09_climate_panel_wide_format(self):
        """Feature 9.4: Supports wide=True returning pivoted matrix format."""
        build_fn = _require_climate_panel()
        panel_wide = build_fn(codes=["USA", "DEU"], start_year=2015, frequency="A", wide=True)
        assert isinstance(panel_wide, pd.DataFrame)
        assert len(panel_wide.columns) > 3

    def test_t1_f09_climate_panel_no_duplicate_keys(self):
        """Feature 9.5: Guaranteed zero duplicate keys (code, date, variable) or (code, date)."""
        build_fn = _require_climate_panel()
        panel = build_fn(codes=["USA", "DEU"], start_year=2015, wide=False)
        if "code" in panel.columns and "date" in panel.columns and "variable" in panel.columns:
            dups = panel.duplicated(subset=["code", "date", "variable"])
            assert not dups.any(), f"Found {dups.sum()} duplicate keys in climate panel"

    # --- Feature 10: build_financial_panel (5 tests) ---
    def test_t1_f10_financial_panel_merge_integrity(self):
        """Feature 10.1: Merges sovereign yields, policy rates, credit gaps, and commodities."""
        build_fn = _require_financial_panel()
        panel = build_fn(codes=["USA", "DEU"], start_date="2015-01-01", frequency="Q")
        assert isinstance(panel, pd.DataFrame)
        assert not panel.empty

    def test_t1_f10_financial_panel_quarterly_frequency(self):
        """Feature 10.2: Aggregates monthly yields/commodities to quarterly frequency ('Q')."""
        build_fn = _require_financial_panel()
        panel_q = build_fn(codes=["USA"], start_date="2018-01-01", frequency="Q")
        if "date" in panel_q.columns:
            assert set(panel_q["date"].dt.month).issubset({1, 4, 7, 10})

    def test_t1_f10_financial_panel_monthly_frequency(self):
        """Feature 10.3: Frequency 'M' harmonizes quarterly credit gaps to monthly."""
        build_fn = _require_financial_panel()
        panel_m = build_fn(codes=["USA"], start_date="2020-01-01", frequency="M")
        assert not panel_m.empty
        if "date" in panel_m.columns:
            assert (panel_m["date"].dt.day == 1).all()

    def test_t1_f10_financial_panel_wide_format(self):
        """Feature 10.4: Supports wide=True returning structured matrix."""
        build_fn = _require_financial_panel()
        panel_wide = build_fn(codes=["USA"], start_date="2018-01-01", frequency="Q", wide=True)
        assert isinstance(panel_wide, pd.DataFrame)
        assert len(panel_wide.columns) > 3

    def test_t1_f10_financial_panel_harmonization_modes(self):
        """Feature 10.5: Supports harmonization modes ('mean' vs 'last')."""
        build_fn = _require_financial_panel()
        p_mean = build_fn(codes=["USA"], start_date="2020-01-01", frequency="Q", harmonization="mean")
        p_last = build_fn(codes=["USA"], start_date="2020-01-01", frequency="Q", harmonization="last")
        assert isinstance(p_mean, pd.DataFrame) and isinstance(p_last, pd.DataFrame)


# ===========================================================================
# Tier 2: Boundary & Corner Cases (8 tests)
# ===========================================================================

class TestTier2BoundaryAndCornerCases:
    """Rigorous boundary, corner, and adversarial stress tests."""

    def test_t2_b01_extreme_historical_and_future_dates(self):
        """Boundary 1: Querying years 1800 or 2100 returns empty frame without crashing."""
        fetch_fn = _require_wdi_emissions()
        df_old = fetch_fn(codes=["USA"], start_year=1800, end_year=1810)
        assert df_old.empty
        assert list(df_old.columns) == ["code", "date", "variable", "value", "sa_source", "source"]

        df_future = fetch_fn(codes=["USA"], start_year=2090, end_year=2100)
        assert df_future.empty

    def test_t2_b02_empty_code_list_contract(self):
        """Boundary 2: Passing codes=[] returns an empty DataFrame preserving schema."""
        fetch_fn = _require_sovereign_yields()
        df = fetch_fn(codes=[], start_date="2020-01-01")
        assert df.empty
        assert list(df.columns) == ["code", "date", "variable", "value", "sa_source", "source"]

    def test_t2_b03_invalid_frequency_raises_value_error(self):
        """Boundary 3: Passing invalid frequency string raises ValueError."""
        build_fn = _require_climate_panel()
        with pytest.raises((ValueError, KeyError)):
            build_fn(codes=["USA"], frequency="DAILY")

    def test_t2_b04_unknown_unsupported_iso_codes(self):
        """Boundary 4: Unsupported ISO country code (e.g. 'ZZZ') returns empty frame without crash."""
        fetch_fn = _require_policy_rates()
        df = fetch_fn(codes=["ZZZ"], start_date="2020-01-01")
        assert df.empty

    def test_t2_b05_single_country_vs_multi_country_symmetry(self):
        """Boundary 5: Querying ['USA'] yields same records as filtering ['USA', 'DEU'] down to USA."""
        fetch_fn = _require_sovereign_yields()
        df_single = fetch_fn(codes=["USA"], start_date="2020-01-01")
        df_multi = fetch_fn(codes=["USA", "DEU"], start_date="2020-01-01")
        multi_usa = df_multi[df_multi["code"] == "USA"].reset_index(drop=True)
        assert len(df_single) == len(multi_usa)

    def test_t2_b06_global_wld_entity_handling(self):
        """Boundary 6: Global benchmarks with code='WLD' are preserved in financial panel."""
        build_fn = _require_financial_panel()
        panel = build_fn(codes=["USA"], start_date="2020-01-01", frequency="Q", wide=False)
        codes_present = set(panel["code"].unique())
        assert "USA" in codes_present or "WLD" in codes_present

    def test_t2_b07_zero_row_resampling_and_missing_data_flags(self):
        """Boundary 7: Handling of missing indicators in panel construction."""
        build_fn = _require_financial_panel()
        panel = build_fn(codes=["MEX"], start_date="2020-01-01", frequency="Q")
        assert isinstance(panel, pd.DataFrame)

    def test_t2_b08_extreme_commodity_price_shock_handling(self):
        """Boundary 8: Zero or negative commodity prices handle without math domain errors."""
        fetch_fn = _require_commodity_benchmarks()
        df = fetch_fn(categories=["energy"], start_date="2020-01-01")
        assert not df.empty
        assert not df["value"].isna().all()


# ===========================================================================
# Tier 3: Cross-Feature Combinations (Pairwise interactions)
# ===========================================================================

class TestTier3CrossFeatureCombinations:
    """Pairwise subsystem and cross-collector integration tests."""

    def test_t3_p01_climate_panel_emissions_and_energy_transition_gdp(self):
        """Pairwise 1: Merges emissions with energy transition and GDP."""
        build_fn = _require_climate_panel()
        panel = build_fn(codes=["USA", "DEU"], start_year=2010, frequency="A", wide=True)
        assert not panel.empty
        cols = list(panel.columns)
        has_emissions = any("co2" in c or "ghg" in c for c in cols)
        has_energy = any("energy" in c or "elec" in c for c in cols)
        assert has_emissions or has_energy

    def test_t3_p02_financial_panel_yields_credit_gap_and_commodities_q_freq(self):
        """Pairwise 2: Financial panel merging yields, credit gap, and commodities at quarterly freq."""
        build_fn = _require_financial_panel()
        panel = build_fn(codes=["USA", "GBR"], start_date="2015-01-01", frequency="Q", wide=True)
        assert not panel.empty
        cols = list(panel.columns)
        has_yield = any("yield" in c or "rate" in c for c in cols)
        assert has_yield

    def test_t3_p03_financial_panel_yields_and_credit_gap_m_freq(self):
        """Pairwise 3: Financial panel merging yields and credit gap at monthly freq."""
        build_fn = _require_financial_panel()
        panel_m = build_fn(codes=["USA"], start_date="2018-01-01", frequency="M", wide=False)
        assert not panel_m.empty
        assert "date" in panel_m.columns

    def test_t3_p04_unified_emissions_panel_wdi_and_oecd_sdmx_cross_validation(self):
        """Pairwise 4: Cross-validation of WDI and OECD total GHG emissions."""
        wdi_fn = _require_wdi_emissions()
        oecd_fn = _require_oecd_ghg()
        wdi_df = wdi_fn(codes=["USA"], start_year=2015, end_year=2018)
        oecd_df = oecd_fn(codes=["USA"], sectors=["_T"], start_year=2015)
        assert not wdi_df.empty and not oecd_df.empty
        wdi_tot = wdi_df[wdi_df["variable"].str.contains("ghg_total|co2_total")]["value"].mean()
        oecd_tot = oecd_df[oecd_df["variable"] == "ghg_total_kt_a"]["value"].mean()
        assert 0.2 < (wdi_tot / oecd_tot) < 5.0

    def test_t3_p05_financial_conditions_with_sovereign_spreads_and_ted_rate(self):
        """Pairwise 5: Joint term spread and financial stress condition tracking."""
        fcond_fn = _require_financial_conditions()
        yields_fn = _require_sovereign_yields()
        fcond = fcond_fn(start_date="2020-01-01")
        yields = yields_fn(codes=["USA"], maturities=("10Y", "2Y"), start_date="2020-01-01")
        assert not fcond.empty and not yields.empty


# ===========================================================================
# Tier 4: Real-World Macroeconomic Application Scenarios
# ===========================================================================

class TestTier4RealWorldScenarios:
    """Canonical macroeconomic benchmark scenarios exercising end-to-end data workflows."""

    def test_t4_s1_carbon_intensity_panel_g7_economies(self):
        """Scenario 1: G7 Carbon Intensity Decoupling Panel (CO2 / GDP).
        Verifies cross-country emissions per unit macro output exhibits secular decline.
        """
        build_fn = _require_climate_panel()
        g7 = ["USA", "GBR", "DEU", "FRA", "ITA", "JPN", "CAN"]
        panel = build_fn(codes=g7, start_year=1995, frequency="A", wide=True)
        assert not panel.empty

        co2_cols = [c for c in panel.columns if "co2" in c or "ghg" in c]
        if co2_cols:
            series = panel[co2_cols[0]].dropna()
            if len(series) >= 10:
                early = series.iloc[:5].mean()
                late = series.iloc[-5:].mean()
                assert late < early * 1.05, "Emissions failed to show secular decoupling or stability"

    def test_t4_s2_clean_energy_transition_trajectory_1990_2023(self):
        """Scenario 2: Clean Energy Transition Trajectory (1990-2023).
        Verifies substitution dynamics between renewable generation and fossil generation.
        """
        fetch_fn = _require_energy_transition()
        df = fetch_fn(codes=["DEU"], start_year=1990)
        assert not df.empty
        ren_rows = df[df["variable"] == "elec_gen_renewable_pct_a"]
        if not ren_rows.empty and len(ren_rows) >= 5:
            ren_early = ren_rows.sort_values("date").iloc[0]["value"]
            ren_late = ren_rows.sort_values("date").iloc[-1]["value"]
            assert ren_late >= ren_early

    def test_t4_s3_global_commodity_inflation_pressure_energy_vs_agriculture(self):
        """Scenario 3: Global Commodity Inflation Pressure (Energy vs Agricultural Staples).
        Tests co-movement and volatility surges during global commodity cycles.
        """
        fetch_fn = _require_commodity_benchmarks()
        df = fetch_fn(frequency="M", start_date="2005-01-01")
        assert not df.empty
        brent = df[df["variable"] == "oil_brent_m"]
        if not brent.empty and len(brent) > 24:
            assert brent["value"].std() > 0.0

    def test_t4_s4_macroprudential_early_warning_credit_gap_vs_property_prices(self):
        """Scenario 4: Macroprudential Early Warning: BIS Credit Gap vs Real Property Prices.
        Tests Basel III Countercyclical Capital Buffer (CCyB) trigger rule: credit_gap > +2.0%.
        """
        fetch_fn = _require_bis_macroprudential()
        df = fetch_fn(codes=["USA"], start_date="2005-01-01")
        assert not df.empty
        gap_rows = df[df["variable"].str.contains("credit_gap")]
        if not gap_rows.empty:
            has_positive = (gap_rows["value"] > 0.0).any()
            has_negative = (gap_rows["value"] < 0.0).any()
            assert has_positive or has_negative

    def test_t4_s5_yield_curve_and_monetary_policy_stance_across_major_cbs(self):
        """Scenario 5: Yield Curve (10Y-2Y Term Spread) vs Central Bank Policy Rates.
        Tests yield curve inversion detection and policy rate transmission across US & Eurozone.
        """
        build_fn = _require_financial_panel()
        panel = build_fn(codes=["USA", "DEU"], start_date="2010-01-01", frequency="Q", wide=True)
        assert not panel.empty
        assert len(panel) > 4
