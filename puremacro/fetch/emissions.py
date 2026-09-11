"""Cross-Country Greenhouse Gas & Emissions Collectors.

Provides keyless, cached collectors for international greenhouse gas emissions
data from:
1. World Bank World Development Indicators (WDI) — Source 2 (IPCC AR5) series:
   - CO2 emissions per capita (metric tons per capita)
   - Total CO2 emissions (converted from Mt to kt)
   - Total GHG emissions (converted from Mt to kt)
   - Methane emissions (converted from Mt to kt)
   - Nitrous oxide emissions (converted from Mt to kt)
   With legacy code fallback mapping.
2. OECD SDMX Air Emissions Inventory (``DSD_AIR_GHG@DF_AIR_GHG``):
   - Sectoral emission breakdowns: energy industries, manufacturing, transport,
     residential, and total emissions in kilotonnes (kt CO2e).
3. Unified ``fetch_emissions_panel`` merging WDI and OECD GHG emissions into an
   aligned multi-country panel in standard long-form schema:
   ``[code, date, variable, value, sa_source, source]``.

Architectural invariant:
Zero module-scope ``requests`` import. HTTP access is deferred to call time
via ``puremacro.fetch._http.cached_get`` and ``puremacro.fetch._oecd_sdmx.get_sdmx_csv``.
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Sequence

import numpy as np
import pandas as pd

from puremacro._codes import is_country

_EMPTY = pd.DataFrame(
    columns=["code", "date", "variable", "value", "sa_source", "source"]
)

#: Standard World Bank WDI AR5 indicator mapping to variable name and unit scale multiplier (Mt -> kt is * 1000).
_INDICATOR_CONFIG: dict[str, dict[str, object]] = {
    "EN.GHG.CO2.PC.CE.AR5": {"variable": "co2_pc_a", "multiplier": 1.0},
    "EN.GHG.CO2.MT.CE.AR5": {"variable": "co2_total_kt_a", "multiplier": 1000.0},
    "EN.GHG.ALL.MT.CE.AR5": {"variable": "ghg_total_kt_a", "multiplier": 1000.0},
    "EN.GHG.CH4.MT.CE.AR5": {"variable": "methane_kt_a", "multiplier": 1000.0},
    "EN.GHG.N2O.MT.CE.AR5": {"variable": "nitrous_oxide_kt_a", "multiplier": 1000.0},
    # Legacy fallbacks:
    "EN.ATM.CO2E.PC": {"variable": "co2_pc_a", "multiplier": 1.0},
    "EN.ATM.CO2E.KT": {"variable": "co2_total_kt_a", "multiplier": 1.0},
    "EN.ATM.GHGT.KT.CE": {"variable": "ghg_total_kt_a", "multiplier": 1.0},
    "EN.ATM.METH.KT.CE": {"variable": "methane_kt_a", "multiplier": 1.0},
    "EN.ATM.NOXE.KT.CE": {"variable": "nitrous_oxide_kt_a", "multiplier": 1.0},
}

#: Map legacy archived indicator codes to active AR5 indicators for live querying.
_LEGACY_TO_AR5: dict[str, str] = {
    "EN.ATM.CO2E.PC": "EN.GHG.CO2.PC.CE.AR5",
    "EN.ATM.CO2E.KT": "EN.GHG.CO2.MT.CE.AR5",
    "EN.ATM.GHGT.KT.CE": "EN.GHG.ALL.MT.CE.AR5",
    "EN.ATM.METH.KT.CE": "EN.GHG.CH4.MT.CE.AR5",
    "EN.ATM.NOXE.KT.CE": "EN.GHG.N2O.MT.CE.AR5",
}

_DEFAULT_WDI_INDICATORS: tuple[str, ...] = (
    "EN.GHG.CO2.PC.CE.AR5",
    "EN.GHG.CO2.MT.CE.AR5",
    "EN.GHG.ALL.MT.CE.AR5",
    "EN.GHG.CH4.MT.CE.AR5",
    "EN.GHG.N2O.MT.CE.AR5",
)

#: OECD SDMX sectoral breakdown mapping to canonical variable names.
_OECD_SECTOR_MAP: dict[str, str] = {
    "1A1": "ghg_energy_industries_kt_a",
    "1A2": "ghg_manufacturing_kt_a",
    "1A3": "ghg_transport_kt_a",
    "1A4b": "ghg_residential_kt_a",
    "_T": "ghg_total_kt_a",
}

_VAR_TO_OECD_SECTOR: dict[str, str] = {v: k for k, v in _OECD_SECTOR_MAP.items()}


def _cached_get(url: str, *, refresh: bool = False, timeout: int = 60) -> bytes:
    """Issue cached GET without module-scope requests dependency."""
    try:
        from ._http import cached_get
        return cached_get(url, refresh=refresh, timeout=timeout)
    except Exception:
        return b""


def _get_oecd_csv(
    agency_flow: str, key: str, start_period: str, *, refresh: bool = False
) -> pd.DataFrame:
    """Issue cached OECD SDMX CSV query without module-scope requests dependency."""
    try:
        from ._oecd_sdmx import get_sdmx_csv
        return get_sdmx_csv(agency_flow, key, start_period, refresh=refresh)
    except Exception:
        return pd.DataFrame()


def fetch_wdi_emissions(
    codes: Sequence[str] | None = None,
    *,
    start_year: int = 1990,
    end_year: int | None = None,
    indicators: Sequence[str] | None = None,
    refresh: bool = False,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch World Bank WDI greenhouse gas emissions indicators in long-form schema.

    Parameters
    ----------
    codes : Sequence[str] | None
        ISO-3 country codes to retrieve. If None, queries all available countries.
    start_year : int, default 1990
        Earliest calendar year to include.
    end_year : int | None, optional
        Latest calendar year to include. If None, uses the current year.
    indicators : Sequence[str] | None, optional
        Specific indicator series to fetch. If None, queries the 5 active AR5
        greenhouse gas series. Legacy indicator codes are automatically mapped
        to live AR5 series.
    refresh : bool, default False
        If True, bypasses local disk cache and queries the World Bank API anew.
    timeout : float, default 30.0
        HTTP request timeout in seconds.

    Returns
    -------
    pd.DataFrame
        Conforming long-form DataFrame with columns:
        ``[code, date, variable, value, sa_source, source]``.
        Values are in metric tons per capita for ``co2_pc_a`` and kilotonnes (kt)
        for total CO2, GHG, methane, and nitrous oxide.
    """
    if codes is not None and len(codes) == 0:
        return _EMPTY.copy()

    requested_codes = set(c.strip().upper() for c in codes) if codes is not None else None

    # Determine indicators to query
    if indicators is None:
        target_indicators = list(_DEFAULT_WDI_INDICATORS)
    else:
        target_indicators = list(indicators)

    current_year = dt.date.today().year
    effective_end = end_year if end_year is not None else current_year

    # Format country string for URL
    if requested_codes is not None and len(requested_codes) <= 50:
        country_param = ";".join(sorted(requested_codes))
    else:
        country_param = "all"

    records: list[dict[str, object]] = []

    for ind in target_indicators:
        query_ind = _LEGACY_TO_AR5.get(ind, ind)
        url = (
            f"https://api.worldbank.org/v2/country/{country_param}/indicator/{query_ind}"
            f"?format=json&date={start_year}:{effective_end}&per_page=15000"
        )
        raw_bytes = _cached_get(url, refresh=refresh, timeout=int(timeout))
        if not raw_bytes:
            continue

        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        # World Bank API returns [metadata_dict, record_list]
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            continue

        for item in payload[1]:
            if not isinstance(item, dict):
                continue

            # Country code resolution
            iso3 = item.get("countryiso3code")
            if not iso3 or not isinstance(iso3, str):
                country_obj = item.get("country")
                if isinstance(country_obj, dict):
                    iso3 = country_obj.get("id")
            if not isinstance(iso3, str):
                continue
            iso3 = iso3.strip().upper()
            if not is_country(iso3):
                continue
            if requested_codes is not None and iso3 not in requested_codes:
                continue

            # Value resolution
            raw_val = item.get("value")
            if raw_val is None or pd.isna(raw_val):
                continue
            try:
                val = float(raw_val)
            except (ValueError, TypeError):
                continue

            # Date resolution
            raw_date = item.get("date")
            if not raw_date:
                continue
            try:
                rec_year = int(str(raw_date)[:4])
            except (ValueError, TypeError):
                continue
            if rec_year < start_year or (end_year is not None and rec_year > end_year):
                continue
            date_ts = pd.Timestamp(f"{rec_year}-01-01")

            # Indicator metadata & scaling
            item_ind = item.get("indicator", {})
            ind_id = item_ind.get("id", query_ind) if isinstance(item_ind, dict) else query_ind
            cfg = _INDICATOR_CONFIG.get(ind_id)
            if cfg is None:
                # Check legacy mapping
                mapped_id = _LEGACY_TO_AR5.get(ind_id, ind_id)
                cfg = _INDICATOR_CONFIG.get(mapped_id)

            if cfg is not None:
                var_name = str(cfg["variable"])
                multiplier = float(cfg["multiplier"])
                scaled_val = val * multiplier
            else:
                var_name = ind_id.lower().replace(".", "_") + "_a"
                scaled_val = val

            records.append({
                "code": iso3,
                "date": date_ts,
                "variable": var_name,
                "value": scaled_val,
                "sa_source": "none",
                "source": f"WorldBank:WDI:{ind_id}",
            })

    if not records:
        return _EMPTY.copy()

    df = pd.DataFrame(records, columns=["code", "date", "variable", "value", "sa_source", "source"])
    df = df.drop_duplicates(subset=["code", "date", "variable"], keep="first")
    df = df.sort_values(["code", "variable", "date"]).reset_index(drop=True)
    return df


def fetch_oecd_ghg(
    codes: Sequence[str] | None = None,
    *,
    start_year: int = 1990,
    end_year: int | None = None,
    sectors: Sequence[str] | None = None,
    refresh: bool = False,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Fetch OECD SDMX Air Emissions Inventory (DF_AIR_GHG) in long-form schema.

    Parameters
    ----------
    codes : Sequence[str] | None
        ISO-3 country codes. If None, queries all available countries.
    start_year : int, default 1990
        Earliest observation year to query.
    end_year : int | None, optional
        Latest observation year to query.
    sectors : Sequence[str] | None, optional
        Specific sector codes (e.g. ``'1A1'``, ``'1A2'``, ``'1A3'``, ``'1A4b'``,
        ``'_T'``) or canonical variable names. If None, queries all 5 default
        sectors.
    refresh : bool, default False
        If True, bypasses local disk cache.
    timeout : float, default 60.0
        Request timeout in seconds.

    Returns
    -------
    pd.DataFrame
        Conforming long-form DataFrame with columns:
        ``[code, date, variable, value, sa_source, source]``.
        Values are reported in kilotonnes (kt CO2e).
    """
    if codes is not None and len(codes) == 0:
        return _EMPTY.copy()

    requested_codes = set(c.strip().upper() for c in codes) if codes is not None else None

    # Resolve sector codes
    if sectors is None:
        target_sector_codes = list(_OECD_SECTOR_MAP.keys())
    else:
        target_sector_codes = [
            _VAR_TO_OECD_SECTOR.get(s, s) for s in sectors
        ]

    code_key = "+".join(sorted(requested_codes)) if requested_codes else ""
    measure_key = "+".join(target_sector_codes)
    # 5 dimensions: REF_AREA . FREQ . POLLUTANT . MEASURE . UNIT_MEASURE
    key = f"{code_key}.A.GHG.{measure_key}.T_CO2E"

    raw = _get_oecd_csv(
        "OECD.ENV.EPI,DSD_AIR_GHG@DF_AIR_GHG,",
        key,
        str(start_year),
        refresh=refresh,
    )
    if raw.empty:
        return _EMPTY.copy()

    needed = {"REF_AREA", "TIME_PERIOD", "OBS_VALUE", "MEASURE"}
    if not needed.issubset(raw.columns):
        return _EMPTY.copy()

    sub = raw[raw["MEASURE"].astype(str).isin(target_sector_codes)].copy()
    if sub.empty:
        return _EMPTY.copy()

    records: list[dict[str, object]] = []

    for _, row in sub.iterrows():
        ref_area = row.get("REF_AREA")
        if not isinstance(ref_area, str):
            continue
        code = ref_area.strip().upper()
        if not is_country(code):
            continue
        if requested_codes is not None and code not in requested_codes:
            continue

        obs_val = row.get("OBS_VALUE")
        if obs_val is None or pd.isna(obs_val):
            continue
        try:
            val = float(obs_val)
        except (ValueError, TypeError):
            continue

        # Scale factor: UNIT_MULT=3 means thousands of tonnes = kilotonnes (kt).
        # If UNIT_MULT is present, scale relative to 10^3.
        unit_mult = row.get("UNIT_MULT") if "UNIT_MULT" in row else None
        if unit_mult is not None and not pd.isna(unit_mult):
            try:
                m = float(unit_mult)
                val = val * (10.0 ** (m - 3.0))
            except (ValueError, TypeError):
                pass

        time_period = str(row.get("TIME_PERIOD", "")).strip()
        if not time_period:
            continue
        try:
            rec_year = int(time_period[:4])
        except (ValueError, TypeError):
            continue
        if rec_year < start_year or (end_year is not None and rec_year > end_year):
            continue
        date_ts = pd.Timestamp(f"{rec_year}-01-01")

        measure = str(row.get("MEASURE", "")).strip()
        var_name = _OECD_SECTOR_MAP.get(measure, f"ghg_{measure.lower()}_kt_a")

        records.append({
            "code": code,
            "date": date_ts,
            "variable": var_name,
            "value": val,
            "sa_source": "none",
            "source": f"OECD:DSD_AIR_GHG@DF_AIR_GHG:{measure}",
        })

    if not records:
        return _EMPTY.copy()

    df = pd.DataFrame(records, columns=["code", "date", "variable", "value", "sa_source", "source"])
    df = df.drop_duplicates(subset=["code", "date", "variable"], keep="first")
    df = df.sort_values(["code", "variable", "date"]).reset_index(drop=True)
    return df


def fetch_emissions_panel(
    codes: Sequence[str] | None = None,
    *,
    start_year: int = 1990,
    end_year: int | None = None,
    frequency: str = "A",
    refresh: bool = False,
) -> pd.DataFrame:
    """Unified emissions collector merging World Bank WDI and OECD SDMX data.

    Combines national emissions series from WDI (per capita CO2, total CO2,
    total GHG, methane, nitrous oxide) with OECD sectoral emissions
    (energy industries, manufacturing, transport, residential, total).
    Prioritizes official OECD national inventories when both sources publish
    total GHG emissions for the same country and year.

    Parameters
    ----------
    codes : Sequence[str] | None
        ISO-3 country codes to include. If None, queries all available countries.
    start_year : int, default 1990
        Earliest calendar year to include.
    end_year : int | None, optional
        Latest calendar year to include.
    frequency : str, default 'A'
        Reporting frequency. Must be ``'A'`` (annual) or ``'Q'`` (quarterly).
        When ``'Q'``, annual series are forward-repeated across the four calendar
        quarters with variable suffix ``_q`` and source annotated as
        ``resampled_from_A:...``.
    refresh : bool, default False
        If True, re-fetches underlying series bypassing cache.

    Returns
    -------
    pd.DataFrame
        Conforming long-form DataFrame with columns:
        ``[code, date, variable, value, sa_source, source]``.
    """
    freq_norm = frequency.strip().upper()
    if freq_norm not in ("A", "Q"):
        raise ValueError(f"Unsupported frequency: '{frequency}'. Must be 'A' or 'Q'.")

    df_oecd = fetch_oecd_ghg(codes=codes, start_year=start_year, end_year=end_year, refresh=refresh)
    df_wdi = fetch_wdi_emissions(codes=codes, start_year=start_year, end_year=end_year, refresh=refresh)


    parts = [df for df in (df_oecd, df_wdi) if not df.empty]
    if not parts:
        return _EMPTY.copy()

    merged = pd.concat(parts, ignore_index=True)
    # Deduplicate on (code, date, variable), keeping OECD first where overlaps occur
    merged = merged.drop_duplicates(subset=["code", "date", "variable"], keep="first")

    if freq_norm == "A":
        return merged.sort_values(["code", "variable", "date"]).reset_index(drop=True)

    # Quarterly expansion: expand each annual observation into 4 quarterly periods
    q_records: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        base_year = row["date"].year
        var_name = str(row["variable"])
        q_var = var_name[:-2] + "_q" if var_name.endswith("_a") else var_name + "_q"
        q_source = f"resampled_from_A:{row['source']}"

        for m in (1, 4, 7, 10):
            q_records.append({
                "code": row["code"],
                "date": pd.Timestamp(f"{base_year}-{m:02d}-01"),
                "variable": q_var,
                "value": row["value"],
                "sa_source": row["sa_source"],
                "source": q_source,
            })

    if not q_records:
        return _EMPTY.copy()

    q_df = pd.DataFrame(q_records, columns=["code", "date", "variable", "value", "sa_source", "source"])
    q_df = q_df.sort_values(["code", "variable", "date"]).reset_index(drop=True)
    return q_df


__all__ = ["fetch_wdi_emissions", "fetch_oecd_ghg", "fetch_emissions_panel"]
