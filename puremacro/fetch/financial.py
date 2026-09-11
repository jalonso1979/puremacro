"""International financial and macroprudential data collectors.

Covers:
1. Sovereign debt yields (10Y and 2Y benchmarks) across advanced and emerging
   economies via keyless FRED (FRED CSV endpoint) and OECD MEI series.
2. Central bank policy rates via BIS WS_CBPOL SDMX-CSV and keyless FRED.
3. BIS macroprudential indicators:
   - Credit-to-GDP gap (WS_CREDIT_GAP): gap, credit-to-gdp ratio, and trend.
   - Total credit to private non-financial sector (WS_TC): % of GDP and USD.
   - Selected residential property price indices (WS_SPP): real and nominal.
4. Financial conditions and spreads:
   - Term spreads (10Y - 2Y) and sovereign spreads vs USA or DEU.
   - Systemic liquidity and credit spreads: TED rate, US High Yield OAS,
     Emerging Markets OAS, Chicago Fed NFCI, St. Louis Fed FSI.

Schema
------
All collectors adhere strictly to the standard long-form schema:
``[code, date, variable, value, sa_source, source]``
where:
* ``code``      — ISO-3 country code string (or ``"WLD"`` for global series)
* ``date``      — ``pd.Timestamp`` normalized to period start (monthly or quarterly)
* ``variable``  — canonical variable name string
* ``value``     — numeric float observation
* ``sa_source`` — seasonal adjustment source or ``"none"`` / ``"derived"`` / ``"bis"`` / ``"fred"``
* ``source``    — data provider provenance identifier

Architectural Invariant
-----------------------
Zero module-scope ``requests`` imports. All network requests are routed
through :func:`puremacro.fetch._http.cached_get`.
"""
from __future__ import annotations

import io
import re
import warnings
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import _http

def cached_get(url: str, *, refresh: bool = False, timeout: int = 60, **kwargs) -> bytes:
    """Delegate to _http.cached_get so any runtime monkeypatching is preserved."""
    return _http.cached_get(url, refresh=refresh, timeout=timeout, **kwargs)


_EMPTY = pd.DataFrame({
    "code": pd.Series(dtype="str"),
    "date": pd.Series(dtype="datetime64[ns]"),
    "variable": pd.Series(dtype="str"),
    "value": pd.Series(dtype="float64"),
    "sa_source": pd.Series(dtype="str"),
    "source": pd.Series(dtype="str"),
})

# ---------------------------------------------------------------------------
# Country Code & Dimension Mappings
# ---------------------------------------------------------------------------

_BIS_TO_ISO3: dict[str, str] = {
    "US": "USA", "DE": "DEU", "FR": "FRA", "IT": "ITA", "ES": "ESP",
    "GB": "GBR", "JP": "JPN", "KR": "KOR", "CA": "CAN", "AU": "AUS",
    "NZ": "NZL", "MX": "MEX", "BR": "BRA", "CL": "CHL", "AR": "ARG",
    "TR": "TUR", "IL": "ISR", "ZA": "ZAF", "IN": "IND", "CN": "CHN",
    "CH": "CHE", "SE": "SWE", "NO": "NOR", "DK": "DNK", "FI": "FIN",
    "AT": "AUT", "BE": "BEL", "NL": "NLD", "PT": "PRT", "GR": "GRC",
    "IE": "IRL", "PL": "POL", "CZ": "CZE", "HU": "HUN", "SK": "SVK",
    "SI": "SVN", "EE": "EST", "LV": "LVA", "LT": "LTU", "LU": "LUX",
    "CO": "COL", "RU": "RUS", "ID": "IDN", "SA": "SAU", "SG": "SGP",
    "MY": "MYS", "TH": "THA", "PH": "PHL", "XM": "EUR", "EA": "EUR",
}

_ISO3_TO_BIS: dict[str, str] = {
    iso3: bis for bis, iso3 in _BIS_TO_ISO3.items() if len(bis) == 2 and bis not in ("XM", "EA")
}
_ISO3_TO_BIS.update({
    "USA": "US", "DEU": "DE", "GBR": "GB", "JPN": "JP", "FRA": "FR",
    "ITA": "IT", "ESP": "ES", "CAN": "CA", "AUS": "AU", "CHE": "CH",
    "MEX": "MX", "BRA": "BR", "KOR": "KR", "CHN": "CN", "IND": "IN",
    "ZAF": "ZA", "NLD": "NL", "SWE": "SE", "NOR": "NO", "NZL": "NZ",
    "BEL": "BE", "AUT": "AT", "PRT": "PT", "IRL": "IE", "GRC": "GR",
    "FIN": "FI", "POL": "PL", "DNK": "DK", "TUR": "TR", "CHL": "CL",
    "ARG": "AR", "COL": "CO", "IDN": "ID", "RUS": "RU", "ISR": "IL",
    "CZE": "CZ", "HUN": "HU", "SVK": "SK", "SVN": "SI", "EST": "EE",
    "LVA": "LV", "LTU": "LT", "LUX": "LU",
})

_EUROZONE_ISO3: set[str] = {
    "DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT", "FIN", "PRT",
    "GRC", "IRL", "SVK", "SVN", "EST", "LVA", "LTU", "LUX", "CYP",
    "MLT", "HRV",
}

# 10Y Sovereign Yields on FRED (OECD MEI pattern: IRLTLT01{ISO2}M156N)
_YIELD_10Y_FRED_MAP: dict[str, str] = {
    "USA": "GS10",
    "DEU": "IRLTLT01DEM156N",
    "GBR": "IRLTLT01GBM156N",
    "JPN": "IRLTLT01JPM156N",
    "FRA": "IRLTLT01FRM156N",
    "ITA": "IRLTLT01ITM156N",
    "ESP": "IRLTLT01ESM156N",
    "CAN": "IRLTLT01CAM156N",
    "AUS": "IRLTLT01AUM156N",
    "CHE": "IRLTLT01CHM156N",
    "MEX": "IRLTLT01MXM156N",
    "BRA": "IRLTLT01BRM156N",
    "KOR": "IRLTLT01KRM156N",
    "NLD": "IRLTLT01NLM156N",
    "SWE": "IRLTLT01SEM156N",
    "NOR": "IRLTLT01NOM156N",
    "NZL": "IRLTLT01NZM156N",
    "BEL": "IRLTLT01BEM156N",
    "AUT": "IRLTLT01ATM156N",
    "PRT": "IRLTLT01PTM156N",
    "IRL": "IRLTLT01IEM156N",
    "GRC": "IRLTLT01GRM156N",
    "FIN": "IRLTLT01FIM156N",
    "POL": "IRLTLT01PLM156N",
    "ZAF": "IRLTLT01ZAM156N",
    "IND": "IRLTLT01INM156N",
    "CHN": "IRLTLT01CNM156N",
    "DNK": "IRLTLT01DKM156N",
    "CZE": "IRLTLT01CZM156N",
    "HUN": "IRLTLT01HUM156N",
    "CHL": "IRLTLT01CLM156N",
    "COL": "IRLTLT01COM156N",
    "ISR": "IRLTLT01ILM156N",
    "RUS": "IRLTLT01RUM156N",
}

# 2Y Sovereign Yields on FRED (OECD MEI pattern: FIESTT01{ISO2}M156N)
_YIELD_2Y_FRED_MAP: dict[str, str] = {
    "USA": "GS2",
    "DEU": "FIESTT01DEM156N",
    "GBR": "FIESTT01GBM156N",
    "JPN": "FIESTT01JPM156N",
    "FRA": "FIESTT01FRM156N",
    "ITA": "FIESTT01ITM156N",
    "ESP": "FIESTT01ESM156N",
    "CAN": "FIESTT01CAM156N",
    "AUS": "FIESTT01AUM156N",
    "CHE": "FIESTT01CHM156N",
    "KOR": "FIESTT01KRM156N",
    "NLD": "FIESTT01NLM156N",
    "SWE": "FIESTT01SEM156N",
    "NOR": "FIESTT01NOM156N",
    "NZL": "FIESTT01NZM156N",
    "BEL": "FIESTT01BEM156N",
    "AUT": "FIESTT01ATM156N",
}

# Policy Rates on FRED
_POLICY_RATE_FRED_MAP: dict[str, str] = {
    "USA": "FEDFUNDS",
    "EUR": "ECBDFR",
    "GBR": "BOERUKM",
    "JPN": "INTDSRJPM193N",
    "CAN": "INTDSRCAM193N",
    "CHE": "INTDSRCHM193N",
    "AUS": "INTDSRAUM193N",
    "SWE": "INTDSRSEM193N",
    "NOR": "INTDSRNOM193N",
    "KOR": "INTDSRKRM193N",
    "BRA": "INTDSRBRM193N",
    "MEX": "INTDSRMXM193N",
    "ZAF": "INTDSRZAM193N",
    "IND": "INTDSRINM193N",
    "CHN": "INTDSRCNM193N",
    "NZL": "INTDSRNZM193N",
    "DNK": "INTDSRDKM193N",
    "CHL": "INTDSRCLM193N",
    "COL": "INTDSRCOM193N",
    "ISR": "INTDSRILM193N",
    "POL": "INTDSRPLM193N",
}

# Financial Conditions & Stress Indices on FRED
_FINANCIAL_CONDITIONS_SERIES: dict[str, tuple[str, str, str]] = {
    "ted_spread": ("TEDRATE", "USA", "3-Month TED Spread"),
    "hy_spread": ("BAMLH0A0HYM2", "USA", "ICE BofA US High Yield Index OAS"),
    "em_spread": ("BAMLEMHBPOAS", "WLD", "ICE BofA Emerging Markets High Yield OAS"),
    "nfci": ("NFCI", "USA", "Chicago Fed National Financial Conditions Index"),
    "stlfsi": ("STLFSI4", "USA", "St. Louis Fed Financial Stress Index"),
    "term_spread_us": ("T10Y2Y", "USA", "10Y minus 2Y US Treasury Spread"),
}

# URL endpoints
_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
_BIS_CBPOL_URL = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/M.ALL?format=csv"
_BIS_CBPOL_SINGLE_URL = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/M.{cc}?format=csv"
_BIS_CREDIT_GAP_URL = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CREDIT_GAP/1.0/Q.ALL.ALL.ALL?format=csv"
_BIS_TC_URL = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_TC/1.0/Q.P.ALL.ALL.ALL.ALL?format=csv"
_BIS_SPP_URL = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_SPP/1.0/Q.ALL.ALL.ALL?format=csv"


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _read_fred_series(
    series_id: str,
    *,
    refresh: bool = False,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch and parse keyless FRED CSV into date and numeric value columns."""
    url = _FRED_CSV_URL.format(series_id=series_id)
    try:
        raw = cached_get(url, refresh=refresh, timeout=int(timeout))
    except Exception:
        return pd.DataFrame(columns=["date", "value"])

    if not raw:
        return pd.DataFrame(columns=["date", "value"])

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        return pd.DataFrame(columns=["date", "value"])

    if df.empty or len(df.columns) < 2:
        return pd.DataFrame(columns=["date", "value"])

    date_col = df.columns[0]
    val_col = df.columns[1]

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["value"] = pd.to_numeric(
        df[val_col].replace({".": np.nan, "ND": np.nan, "": np.nan}),
        errors="coerce",
    )
    df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return df[["date", "value"]]


def _read_fred_monthly(
    series_id: str,
    *,
    refresh: bool = False,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch FRED series and normalize to month-start Timestamp via monthly mean."""
    df = _read_fred_series(series_id, refresh=refresh, timeout=timeout)
    if df.empty:
        return df
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.groupby("date", as_index=False)["value"].mean()
    return df


def _read_bis_csv(
    url: str,
    *,
    refresh: bool = False,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch and parse BIS SDMX-CSV with case-normalized column headers."""
    try:
        raw = cached_get(url, refresh=refresh, timeout=int(timeout))
    except Exception:
        return pd.DataFrame()

    if not raw:
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    cols_map = {c: str(c).strip().upper() for c in df.columns}
    df = df.rename(columns=cols_map)
    return df


def _parse_bis_date(s: pd.Series) -> pd.Series:
    """Parse BIS TIME_PERIOD into period-start Timestamps (Q1->Jan 1, M->1st)."""
    str_s = s.astype(str).str.strip()
    if str_s.str.contains(r"\d{4}[-]?Q[1-4]", case=False, regex=True).any():
        clean_q = str_s.str.replace("-", "", regex=False)
        try:
            return pd.PeriodIndex(clean_q, freq="Q").to_timestamp()
        except Exception:
            def _parse_single_q(val: str) -> pd.Timestamp:
                m = re.match(r"^(\d{4})Q([1-4])$", str(val).strip(), re.IGNORECASE)
                if m:
                    y, q = int(m.group(1)), int(m.group(2))
                    m_start = (q - 1) * 3 + 1
                    return pd.Timestamp(f"{y:04d}-{m_start:02d}-01")
                return pd.NaT

            return clean_q.apply(_parse_single_q)
    elif str_s.str.match(r"^\d{4}-\d{2}$").any():
        return pd.to_datetime(str_s + "-01", errors="coerce")
    else:
        return pd.to_datetime(str_s, errors="coerce")


# ---------------------------------------------------------------------------
# Public Fetchers
# ---------------------------------------------------------------------------

def fetch_sovereign_yields(
    codes: Sequence[str] | None = None,
    *,
    start_date: str = "1990-01-01",
    maturities: Sequence[str] = ("10Y", "2Y"),
    refresh: bool = False,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch 10Y and 2Y sovereign bond yields across countries in long-form schema.

    Parameters
    ----------
    codes : sequence of str, optional
        ISO-3 country codes (e.g. ``["USA", "DEU", "GBR", "JPN"]``).
        Defaults to major advanced and emerging market economies.
    start_date : str, default "1990-01-01"
        Earliest observation date (inclusive).
    maturities : sequence of str, default ("10Y", "2Y")
        Bond maturities to retrieve (subset of ``"10Y"``, ``"2Y"``).
    refresh : bool, default False
        If True, re-download and update cached files.
    timeout : float, default 30.0
        Request timeout in seconds.

    Returns
    -------
    pd.DataFrame
        Conforming to long-form schema:
        ``[code, date, variable, value, sa_source, source]``
        Variables: ``yield_10y``, ``yield_2y``.
    """
    if codes is None:
        target_codes = [
            "USA", "DEU", "GBR", "JPN", "FRA", "ITA", "ESP", "CAN",
            "AUS", "CHE", "MEX", "BRA", "KOR", "NLD", "SWE", "NOR",
        ]
    else:
        target_codes = list(dict.fromkeys(c.upper().strip() for c in codes))

    req_maturities = {m.upper().strip() for m in maturities}
    cutoff = pd.to_datetime(start_date)
    rows: list[pd.DataFrame] = []

    for c in target_codes:
        # 10Y Sovereign Yield
        if "10Y" in req_maturities:
            s_id = _YIELD_10Y_FRED_MAP.get(c)
            if not s_id and c in _ISO3_TO_BIS:
                s_id = f"IRLTLT01{_ISO3_TO_BIS[c]}M156N"
            if s_id:
                df_10 = _read_fred_monthly(s_id, refresh=refresh, timeout=timeout)
                if not df_10.empty:
                    df_10 = df_10[df_10["date"] >= cutoff].copy()
                    if not df_10.empty:
                        df_10["code"] = c
                        df_10["variable"] = "yield_10y"
                        df_10["sa_source"] = "none"
                        df_10["source"] = f"FRED:{s_id}"
                        rows.append(df_10[["code", "date", "variable", "value", "sa_source", "source"]])

        # 2Y Sovereign Yield
        if "2Y" in req_maturities:
            s_id_2 = _YIELD_2Y_FRED_MAP.get(c)
            if not s_id_2 and c in _ISO3_TO_BIS:
                s_id_2 = f"FIESTT01{_ISO3_TO_BIS[c]}M156N"
            if s_id_2:
                df_2 = _read_fred_monthly(s_id_2, refresh=refresh, timeout=timeout)
                if not df_2.empty:
                    df_2 = df_2[df_2["date"] >= cutoff].copy()
                    if not df_2.empty:
                        df_2["code"] = c
                        df_2["variable"] = "yield_2y"
                        df_2["sa_source"] = "none"
                        df_2["source"] = f"FRED:{s_id_2}"
                        rows.append(df_2[["code", "date", "variable", "value", "sa_source", "source"]])

    if not rows:
        return _EMPTY.copy()

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["code", "date", "variable"], keep="first")
    out = out.sort_values(["code", "date", "variable"]).reset_index(drop=True)
    return out


def fetch_policy_rates(
    codes: Sequence[str] | None = None,
    *,
    start_date: str = "1990-01-01",
    source_preference: str = "bis",
    refresh: bool = False,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch central bank policy rates across advanced and emerging economies.

    Queries BIS central bank policy rates (WS_CBPOL) with seamless fallback
    to keyless FRED policy rates (FEDFUNDS, ECBDFR, BOERUKM, etc.). Automatically
    maps the ECB policy rate to Eurozone country ISO-3 codes when requested.

    Parameters
    ----------
    codes : sequence of str, optional
        ISO-3 country codes. Defaults to major central banks.
    start_date : str, default "1990-01-01"
        Earliest observation date (inclusive).
    source_preference : {"bis", "fred"}, default "bis"
        Primary provider to query.
    refresh : bool, default False
        If True, re-downloads and updates cache.
    timeout : float, default 30.0
        HTTP request timeout in seconds.

    Returns
    -------
    pd.DataFrame
        Conforming to long-form schema:
        ``[code, date, variable, value, sa_source, source]``
        Variable: ``policy_rate``.
    """
    if codes is None:
        target_codes = [
            "USA", "DEU", "GBR", "JPN", "FRA", "ITA", "ESP", "CAN",
            "AUS", "CHE", "MEX", "BRA", "KOR", "CHN", "IND", "ZAF",
        ]
    else:
        target_codes = list(dict.fromkeys(c.upper().strip() for c in codes))

    cutoff = pd.to_datetime(start_date)
    found_codes: set[str] = set()
    rows: list[pd.DataFrame] = []

    # 1. Attempt BIS WS_CBPOL if preferred
    if source_preference == "bis":
        df_bis = _read_bis_csv(_BIS_CBPOL_URL, refresh=refresh, timeout=timeout)
        if not df_bis.empty:
            ref_col = "REF_AREA" if "REF_AREA" in df_bis.columns else ("BORROWERS_CTY" if "BORROWERS_CTY" in df_bis.columns else None)
            time_col = "TIME_PERIOD" if "TIME_PERIOD" in df_bis.columns else ("TIME" if "TIME" in df_bis.columns else None)
            val_col = "OBS_VALUE" if "OBS_VALUE" in df_bis.columns else ("VALUE" if "VALUE" in df_bis.columns else None)

            if ref_col and time_col and val_col:
                df_bis["date"] = _parse_bis_date(df_bis[time_col])
                df_bis["value"] = pd.to_numeric(df_bis[val_col], errors="coerce")
                df_bis = df_bis.dropna(subset=["date", "value"])
                df_bis = df_bis[df_bis["date"] >= cutoff]

                # Map BIS codes
                df_bis["code"] = df_bis[ref_col].astype(str).str.strip().map(_BIS_TO_ISO3)

                # Collect country series
                for c in target_codes:
                    sub = df_bis[df_bis["code"] == c].copy()
                    if not sub.empty:
                        sub["variable"] = "policy_rate"
                        sub["sa_source"] = "none"
                        sub["source"] = f"BIS:WS_CBPOL:{c}"
                        rows.append(sub[["code", "date", "variable", "value", "sa_source", "source"]])
                        found_codes.add(c)
                    elif c in _EUROZONE_ISO3:
                        # Map ECB policy rate (code EUR or XM)
                        ecb_sub = df_bis[df_bis["code"] == "EUR"].copy()
                        if ecb_sub.empty and "XM" in df_bis[ref_col].values:
                            ecb_sub = df_bis[df_bis[ref_col] == "XM"].copy()
                        if not ecb_sub.empty:
                            sub_c = ecb_sub.copy()
                            sub_c["code"] = c
                            sub_c["variable"] = "policy_rate"
                            sub_c["sa_source"] = "none"
                            sub_c["source"] = "BIS:WS_CBPOL:XM"
                            rows.append(sub_c[["code", "date", "variable", "value", "sa_source", "source"]])
                            found_codes.add(c)

    # 2. Query FRED for remaining requested countries
    missing_codes = [c for c in target_codes if c not in found_codes]
    if missing_codes:
        # Check if Eurozone ECB rate is needed for any missing Eurozone country
        needed_ez = [c for c in missing_codes if c in _EUROZONE_ISO3]
        if needed_ez:
            ecb_fred = _read_fred_monthly(_POLICY_RATE_FRED_MAP["EUR"], refresh=refresh, timeout=timeout)
            if not ecb_fred.empty:
                ecb_fred = ecb_fred[ecb_fred["date"] >= cutoff].copy()
                if not ecb_fred.empty:
                    for c in needed_ez:
                        sub_ez = ecb_fred.copy()
                        sub_ez["code"] = c
                        sub_ez["variable"] = "policy_rate"
                        sub_ez["sa_source"] = "none"
                        sub_ez["source"] = f"FRED:{_POLICY_RATE_FRED_MAP['EUR']}"
                        rows.append(sub_ez[["code", "date", "variable", "value", "sa_source", "source"]])
                        found_codes.add(c)

        for c in missing_codes:
            if c in found_codes:
                continue
            s_id = _POLICY_RATE_FRED_MAP.get(c)
            if s_id:
                df_f = _read_fred_monthly(s_id, refresh=refresh, timeout=timeout)
                if not df_f.empty:
                    df_f = df_f[df_f["date"] >= cutoff].copy()
                    if not df_f.empty:
                        df_f["code"] = c
                        df_f["variable"] = "policy_rate"
                        df_f["sa_source"] = "none"
                        df_f["source"] = f"FRED:{s_id}"
                        rows.append(df_f[["code", "date", "variable", "value", "sa_source", "source"]])
                        found_codes.add(c)

    if not rows:
        return _EMPTY.copy()

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["code", "date", "variable"], keep="first")
    out = out.sort_values(["code", "date", "variable"]).reset_index(drop=True)
    return out


def fetch_bis_macroprudential(
    codes: Sequence[str] | None = None,
    *,
    start_date: str = "1990-01-01",
    indicators: Sequence[str] | None = None,
    refresh: bool = False,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch BIS credit-to-GDP gap, total credit to private sector, and property prices.

    Variables
    ---------
    * ``credit_gap_q``              — Credit-to-GDP gap (percentage points deviation)
    * ``credit_to_gdp_q``          — Credit-to-GDP ratio (%)
    * ``credit_trend_q``           — Credit-to-GDP trend ratio (%)
    * ``credit_private_pct_gdp_q`` — Total credit to private non-financial sector (% of GDP)
    * ``credit_private_usd_q``     — Total credit to private non-financial sector (USD billions)
    * ``property_price_real_q``    — Real residential property price index (2010=100)
    * ``property_price_nominal_q`` — Nominal residential property price index

    Parameters
    ----------
    codes : sequence of str, optional
        ISO-3 country codes. If None, returns all available countries.
    start_date : str, default "1990-01-01"
        Earliest observation date (inclusive).
    indicators : sequence of str, optional
        Specific indicator variables to retrieve. If None, retrieves all available.
    refresh : bool, default False
        If True, re-downloads and updates cache.
    timeout : float, default 30.0
        Request timeout in seconds.

    Returns
    -------
    pd.DataFrame
        Conforming to long-form schema:
        ``[code, date, variable, value, sa_source, source]``.
    """
    if codes is not None:
        codes = list(dict.fromkeys(codes))
    cutoff = pd.to_datetime(start_date)
    target_codes = {c.upper().strip() for c in codes} if codes is not None else None

    # Canonical indicator sets
    all_vars = {
        "credit_gap_q", "credit_to_gdp_q", "credit_trend_q",
        "credit_private_pct_gdp_q", "credit_private_usd_q",
        "property_price_real_q", "property_price_nominal_q",
    }
    if indicators is None:
        req_vars = all_vars
    else:
        req_vars = set()
        for ind in indicators:
            ind_clean = ind.lower().strip()
            if not ind_clean.endswith("_q") and f"{ind_clean}_q" in all_vars:
                req_vars.add(f"{ind_clean}_q")
            elif ind_clean in all_vars:
                req_vars.add(ind_clean)

    rows: list[pd.DataFrame] = []

    # -----------------------------------------------------------------------
    # 1. BIS Credit-to-GDP Gap (WS_CREDIT_GAP)
    # -----------------------------------------------------------------------
    cg_vars = {"credit_gap_q", "credit_to_gdp_q", "credit_trend_q"} & req_vars
    if cg_vars:
        df_cg = _read_bis_csv(_BIS_CREDIT_GAP_URL, refresh=refresh, timeout=timeout)
        if not df_cg.empty:
            ref_col = "BORROWERS_CTY" if "BORROWERS_CTY" in df_cg.columns else ("REF_AREA" if "REF_AREA" in df_cg.columns else None)
            type_col = "CG_DATA_TYPE" if "CG_DATA_TYPE" in df_cg.columns else ("DATA_TYPE" if "DATA_TYPE" in df_cg.columns else None)
            time_col = "TIME_PERIOD" if "TIME_PERIOD" in df_cg.columns else ("TIME" if "TIME" in df_cg.columns else None)
            val_col = "OBS_VALUE" if "OBS_VALUE" in df_cg.columns else ("VALUE" if "VALUE" in df_cg.columns else None)

            if ref_col and type_col and time_col and val_col:
                df_cg["date"] = _parse_bis_date(df_cg[time_col])
                df_cg["value"] = pd.to_numeric(df_cg[val_col], errors="coerce")
                df_cg = df_cg.dropna(subset=["date", "value"])
                df_cg = df_cg[df_cg["date"] >= cutoff]

                df_cg["code"] = df_cg[ref_col].astype(str).str.strip().map(_BIS_TO_ISO3)
                df_cg = df_cg.dropna(subset=["code"])
                if target_codes is not None:
                    df_cg = df_cg[df_cg["code"].isin(target_codes)]

                type_map = {
                    "GAP": "credit_gap_q",
                    "RAT": "credit_to_gdp_q",
                    "RATIO": "credit_to_gdp_q",
                    "TRD": "credit_trend_q",
                    "TREND": "credit_trend_q",
                }
                df_cg["variable"] = df_cg[type_col].astype(str).str.strip().str.upper().map(type_map)
                df_cg = df_cg[df_cg["variable"].isin(cg_vars)]
                if not df_cg.empty:
                    df_cg["sa_source"] = "none"
                    df_cg["source"] = "BIS:WS_CREDIT_GAP"
                    rows.append(df_cg[["code", "date", "variable", "value", "sa_source", "source"]])

    # -----------------------------------------------------------------------
    # 2. BIS Total Credit to Private Non-Financial Sector (WS_TC)
    # -----------------------------------------------------------------------
    tc_vars = {"credit_private_pct_gdp_q", "credit_private_usd_q"} & req_vars
    if tc_vars:
        df_tc = _read_bis_csv(_BIS_TC_URL, refresh=refresh, timeout=timeout)
        if not df_tc.empty:
            ref_col = "BORROWERS_CTY" if "BORROWERS_CTY" in df_tc.columns else ("REF_AREA" if "REF_AREA" in df_tc.columns else None)
            unit_col = "UNIT_MEASURE" if "UNIT_MEASURE" in df_tc.columns else ("UNIT" if "UNIT" in df_tc.columns else None)
            time_col = "TIME_PERIOD" if "TIME_PERIOD" in df_tc.columns else ("TIME" if "TIME" in df_tc.columns else None)
            val_col = "OBS_VALUE" if "OBS_VALUE" in df_tc.columns else ("VALUE" if "VALUE" in df_tc.columns else None)

            if ref_col and unit_col and time_col and val_col:
                df_tc["date"] = _parse_bis_date(df_tc[time_col])
                df_tc["value"] = pd.to_numeric(df_tc[val_col], errors="coerce")
                df_tc = df_tc.dropna(subset=["date", "value"])
                df_tc = df_tc[df_tc["date"] >= cutoff]

                df_tc["code"] = df_tc[ref_col].astype(str).str.strip().map(_BIS_TO_ISO3)
                df_tc = df_tc.dropna(subset=["code"])
                if target_codes is not None:
                    df_tc = df_tc[df_tc["code"].isin(target_codes)]

                # Filter borrower if column present
                if "TC_BORROWERS" in df_tc.columns:
                    df_tc = df_tc[df_tc["TC_BORROWERS"].astype(str).str.strip().str.upper().isin(["P", "ALL"])]

                def _map_tc_unit(u: str) -> str | None:
                    u_up = str(u).strip().upper()
                    if u_up in ("770", "P") or "GDP" in u_up:
                        return "credit_private_pct_gdp_q"
                    elif "USD" in u_up or "DOLLAR" in u_up:
                        return "credit_private_usd_q"
                    return None

                df_tc["variable"] = df_tc[unit_col].map(_map_tc_unit)
                df_tc = df_tc[df_tc["variable"].isin(tc_vars)]
                if not df_tc.empty:
                    df_tc["sa_source"] = "none"
                    df_tc["source"] = "BIS:WS_TC"
                    rows.append(df_tc[["code", "date", "variable", "value", "sa_source", "source"]])

    # -----------------------------------------------------------------------
    # 3. BIS Selected Property Prices (WS_SPP)
    # -----------------------------------------------------------------------
    spp_vars = {"property_price_real_q", "property_price_nominal_q"} & req_vars
    if spp_vars:
        df_spp = _read_bis_csv(_BIS_SPP_URL, refresh=refresh, timeout=timeout)
        if not df_spp.empty:
            ref_col = "REF_AREA" if "REF_AREA" in df_spp.columns else ("BORROWERS_CTY" if "BORROWERS_CTY" in df_spp.columns else None)
            val_type_col = "VALUE" if "VALUE" in df_spp.columns else ("PRICE_TYPE" if "PRICE_TYPE" in df_spp.columns else None)
            time_col = "TIME_PERIOD" if "TIME_PERIOD" in df_spp.columns else ("TIME" if "TIME" in df_spp.columns else None)
            val_col = "OBS_VALUE" if "OBS_VALUE" in df_spp.columns else ("OBS" if "OBS" in df_spp.columns else None)

            if ref_col and val_type_col and time_col and val_col:
                df_spp["date"] = _parse_bis_date(df_spp[time_col])
                df_spp["value"] = pd.to_numeric(df_spp[val_col], errors="coerce")
                df_spp = df_spp.dropna(subset=["date", "value"])
                df_spp = df_spp[df_spp["date"] >= cutoff]

                df_spp["code"] = df_spp[ref_col].astype(str).str.strip().map(_BIS_TO_ISO3)
                df_spp = df_spp.dropna(subset=["code"])
                if target_codes is not None:
                    df_spp = df_spp[df_spp["code"].isin(target_codes)]

                def _map_spp_val(v: str) -> str | None:
                    v_up = str(v).strip().upper()
                    if v_up in ("R", "REAL") or "REAL" in v_up:
                        return "property_price_real_q"
                    elif v_up in ("N", "NOMINAL") or "NOMINAL" in v_up:
                        return "property_price_nominal_q"
                    return None

                df_spp["variable"] = df_spp[val_type_col].map(_map_spp_val)
                df_spp = df_spp[df_spp["variable"].isin(spp_vars)]
                if not df_spp.empty:
                    df_spp["sa_source"] = "none"
                    df_spp["source"] = "BIS:WS_SPP"
                    rows.append(df_spp[["code", "date", "variable", "value", "sa_source", "source"]])

    if not rows:
        return _EMPTY.copy()

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["code", "date", "variable"], keep="first")
    out = out.sort_values(["code", "date", "variable"]).reset_index(drop=True)
    return out


def fetch_bis_credit_gap(
    codes: Sequence[str] | None = None,
    *,
    start_date: str = "1990-01-01",
    refresh: bool = False,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Convenience helper for BIS credit-to-GDP gap indicators."""
    return fetch_bis_macroprudential(
        codes,
        start_date=start_date,
        indicators=["credit_gap_q", "credit_to_gdp_q", "credit_trend_q"],
        refresh=refresh,
        timeout=timeout,
    )


def fetch_bis_total_credit(
    codes: Sequence[str] | None = None,
    *,
    start_date: str = "1990-01-01",
    refresh: bool = False,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Convenience helper for BIS total credit to private non-financial sector."""
    return fetch_bis_macroprudential(
        codes,
        start_date=start_date,
        indicators=["credit_private_pct_gdp_q", "credit_private_usd_q"],
        refresh=refresh,
        timeout=timeout,
    )


def fetch_bis_property_prices(
    codes: Sequence[str] | None = None,
    *,
    price_type: str = "real",
    start_date: str = "1990-01-01",
    refresh: bool = False,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Convenience helper for BIS residential property price indices.

    Parameters
    ----------
    price_type : {"real", "nominal", "both"}, default "real"
        Which price series to return.
    """
    if price_type == "real":
        inds = ["property_price_real_q"]
    elif price_type == "nominal":
        inds = ["property_price_nominal_q"]
    else:
        inds = ["property_price_real_q", "property_price_nominal_q"]

    return fetch_bis_macroprudential(
        codes,
        start_date=start_date,
        indicators=inds,
        refresh=refresh,
        timeout=timeout,
    )


def fetch_financial_conditions(
    *,
    indicators: Sequence[str] | None = None,
    start_date: str = "1990-01-01",
    frequency: str = "M",
    refresh: bool = False,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch sovereign spreads, term spreads, TED spread, HY OAS, EM OAS, and NFCI.

    Keyless FRED financial stress and credit spreads:
    * ``ted_spread``     — TED rate (TEDRATE, 3M LIBOR minus 3M Treasury Bill)
    * ``hy_spread``      — US High Yield OAS (BAMLH0A0HYM2)
    * ``em_spread``      — Emerging Markets OAS (BAMLEMHBPOAS)
    * ``nfci``           — Chicago Fed National Financial Conditions Index (NFCI)
    * ``stlfsi``         — St. Louis Fed Financial Stress Index (STLFSI4)
    * ``term_spread_us`` — 10Y Treasury minus 2Y Treasury Yield Spread (T10Y2Y)

    Parameters
    ----------
    indicators : sequence of str, optional
        Indicators to retrieve. Defaults to standard financial conditions suite.
    start_date : str, default "1990-01-01"
        Earliest observation date (inclusive).
    frequency : {"M", "D"}, default "M"
        Aggregation frequency. "M" returns month-start Timestamp averages.
    refresh : bool, default False
        If True, re-downloads and updates cache.
    timeout : float, default 30.0
        Request timeout in seconds.

    Returns
    -------
    pd.DataFrame
        Conforming to long-form schema:
        ``[code, date, variable, value, sa_source, source]``.
    """
    if indicators is None:
        req_indicators = ["ted_spread", "hy_spread", "em_spread", "nfci", "term_spread_us"]
    else:
        req_indicators = [ind.lower().strip() for ind in indicators]

    cutoff = pd.to_datetime(start_date)
    rows: list[pd.DataFrame] = []

    for ind in req_indicators:
        meta = _FINANCIAL_CONDITIONS_SERIES.get(ind)
        if not meta:
            continue
        series_id, default_code, _desc = meta
        if frequency == "M":
            df = _read_fred_monthly(series_id, refresh=refresh, timeout=timeout)
        else:
            df = _read_fred_series(series_id, refresh=refresh, timeout=timeout)

        if not df.empty:
            df = df[df["date"] >= cutoff].copy()
            if not df.empty:
                df["code"] = default_code
                df["variable"] = ind
                df["sa_source"] = "none"
                df["source"] = f"FRED:{series_id}"
                rows.append(df[["code", "date", "variable", "value", "sa_source", "source"]])

    if not rows:
        return _EMPTY.copy()

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["code", "date", "variable"]).reset_index(drop=True)
    return out


def compute_sovereign_spreads(
    yields_df: pd.DataFrame,
    benchmark_code: str = "USA",
    *,
    include_yields: bool = False,
) -> pd.DataFrame:
    """Compute 10Y - 2Y term spreads and sovereign spreads vs USA or DEU.

    Parameters
    ----------
    yields_df : pd.DataFrame
        Long-form yields DataFrame with columns ``[code, date, variable, value]``
        (as returned by :func:`fetch_sovereign_yields`).
    benchmark_code : str, default "USA"
        Benchmark sovereign for risk spread calculations (e.g. ``"USA"`` or ``"DEU"``).
    include_yields : bool, default False
        If True, appends the computed spreads to the input yields.

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with:
        * ``term_spread``: ``yield_10y - yield_2y`` per country.
        * ``sovereign_spread``: ``yield_10y - yield_10y[benchmark_code]``.
          Evaluates to 0.0 for the benchmark country itself.
    """
    if yields_df.empty or "variable" not in yields_df.columns or "value" not in yields_df.columns:
        return _EMPTY.copy()

    y10 = yields_df[yields_df["variable"] == "yield_10y"][["code", "date", "value"]].copy()
    y2 = yields_df[yields_df["variable"] == "yield_2y"][["code", "date", "value"]].copy()
    y10 = y10.drop_duplicates(subset=["code", "date"], keep="first")
    y2 = y2.drop_duplicates(subset=["code", "date"], keep="first")

    spread_frames: list[pd.DataFrame] = []

    # 1. Term Spread: 10Y - 2Y
    if not y10.empty and not y2.empty:
        term_m = pd.merge(y10, y2, on=["code", "date"], suffixes=("_10y", "_2y"))
        if not term_m.empty:
            term_m["value"] = term_m["value_10y"] - term_m["value_2y"]
            term_m["variable"] = "term_spread"
            term_m["sa_source"] = "derived"
            term_m["source"] = "derived:yield_10y-yield_2y"
            spread_frames.append(term_m[["code", "date", "variable", "value", "sa_source", "source"]])

    # 2. Sovereign Risk Spread: 10Y minus benchmark 10Y
    b_code = benchmark_code.upper().strip()
    if not y10.empty:
        bench_df = y10[y10["code"] == b_code][["date", "value"]].copy()
        if bench_df.empty:
            warnings.warn(
                f"Benchmark code '{b_code}' not found in 10Y yields; cannot compute sovereign spread.",
                UserWarning,
                stacklevel=2,
            )
        else:
            sov_m = pd.merge(y10, bench_df, on="date", suffixes=("", "_bench"))
            if not sov_m.empty:
                sov_m["value"] = sov_m["value"] - sov_m["value_bench"]
                # For the benchmark country itself, value is identically 0.0
                sov_m.loc[sov_m["code"] == b_code, "value"] = 0.0
                sov_m["variable"] = "sovereign_spread"
                sov_m["sa_source"] = "derived"
                sov_m["source"] = f"derived:yield_10y-benchmark_{b_code}"
                spread_frames.append(sov_m[["code", "date", "variable", "value", "sa_source", "source"]])

    if not spread_frames:
        if include_yields:
            return yields_df.copy()
        return _EMPTY.copy()

    computed_spreads = pd.concat(spread_frames, ignore_index=True)

    if include_yields:
        out = pd.concat([yields_df, computed_spreads], ignore_index=True)
    else:
        out = computed_spreads

    out = out.drop_duplicates(subset=["code", "date", "variable"], keep="first")
    out = out.sort_values(["code", "date", "variable"]).reset_index(drop=True)
    return out


__all__ = [
    "fetch_sovereign_yields",
    "fetch_policy_rates",
    "fetch_bis_macroprudential",
    "fetch_bis_credit_gap",
    "fetch_bis_total_credit",
    "fetch_bis_property_prices",
    "fetch_financial_conditions",
    "compute_sovereign_spreads",
    "_BIS_TO_ISO3",
    "_ISO3_TO_BIS",
    "_EUROZONE_ISO3",
]
