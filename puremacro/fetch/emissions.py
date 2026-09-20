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
from collections.abc import Sequence

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
    except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
        return b""


def _get_oecd_csv(
    agency_flow: str, key: str, start_period: str, *, refresh: bool = False
) -> pd.DataFrame:
    """Issue cached OECD SDMX CSV query without module-scope requests dependency."""
    try:
        from ._oecd_sdmx import get_sdmx_csv
        return get_sdmx_csv(agency_flow, key, start_period, refresh=refresh)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
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

    requested_codes = {c.strip().upper() for c in codes} if codes is not None else None

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

    records = []

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

        batch_df = pd.DataFrame.from_records([item for item in payload[1] if isinstance(item, dict)])
        if batch_df.empty:
            continue

        # Country code resolution
        if "countryiso3code" in batch_df.columns:
            iso3_series = batch_df["countryiso3code"]
        else:
            iso3_series = pd.Series([None] * len(batch_df), index=batch_df.index)

        # fill missing with country.id if available
        if "country" in batch_df.columns:
            country_dicts = batch_df["country"]
            mask_missing = iso3_series.isna() | (iso3_series == "")
            if mask_missing.any():
                def extract_country_id(c):
                    return c.get("id") if isinstance(c, dict) else None
                iso3_series.loc[mask_missing] = country_dicts.loc[mask_missing].apply(extract_country_id)

        batch_df["iso3"] = iso3_series.astype(str).str.strip().str.upper()
        # Drop rows where iso3 is essentially missing or just 'NAN'/'NONE' from astype(str)
        batch_df = batch_df[~batch_df["iso3"].isin(["NAN", "NONE", ""])]
        if batch_df.empty:
            continue

        batch_df = batch_df[batch_df["iso3"].apply(is_country)]
        if requested_codes is not None:
            batch_df = batch_df[batch_df["iso3"].isin(requested_codes)]

        if batch_df.empty:
            continue

        # Value resolution
        if "value" not in batch_df.columns:
            continue
        batch_df["value"] = pd.to_numeric(batch_df["value"], errors="coerce")
        batch_df = batch_df.dropna(subset=["value"])
        if batch_df.empty:
            continue

        # Date resolution
        if "date" not in batch_df.columns:
            continue
        batch_df["rec_year"] = batch_df["date"].astype(str).str[:4]
        batch_df["rec_year"] = pd.to_numeric(batch_df["rec_year"], errors="coerce")
        batch_df = batch_df.dropna(subset=["rec_year"])
        batch_df = batch_df[batch_df["rec_year"] >= start_year]
        if end_year is not None:
            batch_df = batch_df[batch_df["rec_year"] <= end_year]
        if batch_df.empty:
            continue

        batch_df["date_ts"] = pd.to_datetime(batch_df["rec_year"].astype(int).astype(str) + "-01-01")

        # Indicator metadata & scaling
        if "indicator" in batch_df.columns:
            # Vectorized indicator processing
            # extract id
            def extract_id(item_ind, def_ind=query_ind):
                return item_ind.get("id", def_ind) if isinstance(item_ind, dict) else def_ind
            ind_ids = batch_df["indicator"].apply(extract_id)

            # Map variables and multipliers
            def map_var(ind_id):
                cfg = _INDICATOR_CONFIG.get(ind_id) or _INDICATOR_CONFIG.get(_LEGACY_TO_AR5.get(ind_id, ind_id))
                return str(cfg["variable"]) if cfg else ind_id.lower().replace(".", "_") + "_a"

            def map_mult(ind_id):
                cfg = _INDICATOR_CONFIG.get(ind_id) or _INDICATOR_CONFIG.get(_LEGACY_TO_AR5.get(ind_id, ind_id))
                return float(cfg["multiplier"]) if cfg else 1.0

            batch_df["variable"] = ind_ids.map(map_var)
            batch_df["value"] = batch_df["value"] * ind_ids.map(map_mult)
            batch_df["source"] = "WorldBank:WDI:" + ind_ids
        else:
            batch_df["variable"] = query_ind.lower().replace(".", "_") + "_a"
            batch_df["source"] = "WorldBank:WDI:" + query_ind

        batch_df["sa_source"] = "none"

        # Select and rename columns
        batch_df["code"] = batch_df["iso3"]
        batch_df["date"] = batch_df["date_ts"]

        records.append(batch_df[["code", "date", "variable", "value", "sa_source", "source"]])

    if not records:
        return _EMPTY.copy()

    df = pd.concat(records, ignore_index=True)
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

    requested_codes = {c.strip().upper() for c in codes} if codes is not None else None

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

    # 1. Filter out invalid areas and types
    sub["REF_AREA"] = sub["REF_AREA"].astype(str).str.strip().str.upper()
    if requested_codes is not None:
        sub = sub[sub["REF_AREA"].isin(requested_codes)]

    sub = sub[sub["REF_AREA"].apply(is_country)]

    if sub.empty:
        return _EMPTY.copy()

    # 2. Filter OBS_VALUE
    sub["OBS_VALUE"] = pd.to_numeric(sub["OBS_VALUE"], errors="coerce")
    sub = sub.dropna(subset=["OBS_VALUE"])

    if sub.empty:
        return _EMPTY.copy()

    # 3. Unit mult
    if "UNIT_MULT" in sub.columns:
        sub["UNIT_MULT"] = pd.to_numeric(sub["UNIT_MULT"], errors="coerce")
        # For non-null UNIT_MULT, val = val * 10 ** (m - 3)
        # For null UNIT_MULT, val = val
        multiplier = 10.0 ** (sub["UNIT_MULT"].fillna(3.0) - 3.0)
        sub["OBS_VALUE"] = sub["OBS_VALUE"] * multiplier

    # 4. time_period
    sub["TIME_PERIOD"] = sub["TIME_PERIOD"].astype(str).str.strip().str[:4]
    sub["TIME_PERIOD"] = pd.to_numeric(sub["TIME_PERIOD"], errors="coerce")
    sub = sub.dropna(subset=["TIME_PERIOD"])
    sub = sub[sub["TIME_PERIOD"] >= start_year]
    if end_year is not None:
        sub = sub[sub["TIME_PERIOD"] <= end_year]

    if sub.empty:
        return _EMPTY.copy()

    # 5. create result cols
    sub["code"] = sub["REF_AREA"]
    sub["date"] = pd.to_datetime(sub["TIME_PERIOD"].astype(int).astype(str) + "-01-01")

    def map_measure(m: str) -> str:
        m = str(m).strip()
        return _OECD_SECTOR_MAP.get(m, f"ghg_{m.lower()}_kt_a")

    sub["variable"] = sub["MEASURE"].apply(map_measure)
    sub["value"] = sub["OBS_VALUE"]
    sub["sa_source"] = "none"
    sub["source"] = "OECD:DSD_AIR_GHG@DF_AIR_GHG:" + sub["MEASURE"].astype(str).str.strip()

    df = sub[["code", "date", "variable", "value", "sa_source", "source"]]

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
    merged["_q_var"] = merged["variable"].astype(str).str.replace(r"_a$", "_q", regex=True)
    mask = ~merged["variable"].astype(str).str.endswith("_a")
    if mask.any():
        merged.loc[mask, "_q_var"] = merged.loc[mask, "variable"].astype(str) + "_q"

    merged["_q_source"] = "resampled_from_A:" + merged["source"].astype(str)

    # Repeat rows 4 times
    idx_repeated = merged.index.repeat(4)
    q_df = merged.loc[idx_repeated].copy()

    # Generate months using np.repeat to ensure alignment with the index repetition
    months = np.tile([1, 4, 7, 10], len(merged))

    # Create the new dates (base_year-month-01)
    base_years = q_df["date"].dt.year
    q_df["date"] = pd.to_datetime(
        base_years.astype(str) + "-" + pd.Series(months, index=q_df.index).astype(str).str.zfill(2) + "-01",
        format="%Y-%m-%d"
    )

    q_df["variable"] = q_df["_q_var"]
    q_df["source"] = q_df["_q_source"]

    q_df = q_df[["code", "date", "variable", "value", "sa_source", "source"]]
    q_df = q_df.sort_values(["code", "variable", "date"]).reset_index(drop=True)
    return q_df


__all__ = ["fetch_emissions_panel", "fetch_oecd_ghg", "fetch_wdi_emissions"]
