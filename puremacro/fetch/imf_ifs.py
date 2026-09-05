"""IMF International Financial Statistics (IFS) monthly fetcher.

Endpoint: ``https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS/<key>``.

We pull three monthly indicators across ~140 countries since 1948:

- ``cpi_ifs_m``   -- IFS ``PCPI_IX`` (CPI all-items index)
- ``ip_ifs_m``    -- IFS ``AIP_IX`` (industrial production index)
- ``urate_ifs_m`` -- IFS ``LUR_PT`` (unemployment rate, % of labour force)

The fetcher is intended as a *gap-filler* downstream: when a country has
an OECD measurement of the same concept on the same ``(code, date)``
cell, prefer OECD; only IFS rows for cells not already populated should
be kept by the build orchestrator.

ISO-2 -> ISO-3 conversion: IFS uses ISO-2 area codes; we map a curated
subset (the ones that overlap with OECD/WUI/EPU country lists) and
reject the rest. The mapping covers ~150 economies; aggregate codes
(``EU``, ``WO``, etc.) are handled by ``src._codes.is_country`` after
mapping.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .._codes import is_country

_BASE = "https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS"
_TIMEOUT = 120

_EMPTY = pd.DataFrame(
    columns=["code", "date", "variable", "value", "sa_source", "source"]
)

# IMF uses ISO-2; project uses ISO-3. Maintained list — covers the ~80
# countries we care about plus a long EM tail.
ISO2_TO_ISO3: dict[str, str] = {
    # Americas
    "US": "USA", "CA": "CAN", "MX": "MEX", "BR": "BRA", "AR": "ARG",
    "CL": "CHL", "CO": "COL", "PE": "PER", "EC": "ECU", "VE": "VEN",
    "BO": "BOL", "UY": "URY", "PY": "PRY", "DO": "DOM", "CR": "CRI",
    "PA": "PAN", "GT": "GTM", "JM": "JAM", "TT": "TTO",
    # Europe
    "GB": "GBR", "DE": "DEU", "FR": "FRA", "IT": "ITA", "ES": "ESP",
    "NL": "NLD", "BE": "BEL", "AT": "AUT", "CH": "CHE", "SE": "SWE",
    "NO": "NOR", "DK": "DNK", "FI": "FIN", "IE": "IRL", "PT": "PRT",
    "GR": "GRC", "PL": "POL", "CZ": "CZE", "HU": "HUN", "RO": "ROU",
    "BG": "BGR", "HR": "HRV", "SK": "SVK", "SI": "SVN", "EE": "EST",
    "LV": "LVA", "LT": "LTU", "LU": "LUX", "IS": "ISL", "MT": "MLT",
    "CY": "CYP", "TR": "TUR", "RU": "RUS", "UA": "UKR", "RS": "SRB",
    # Asia
    "JP": "JPN", "KR": "KOR", "CN": "CHN", "IN": "IND", "ID": "IDN",
    "MY": "MYS", "TH": "THA", "PH": "PHL", "VN": "VNM", "SG": "SGP",
    "HK": "HKG", "TW": "TWN", "PK": "PAK", "BD": "BGD", "LK": "LKA",
    "NP": "NPL", "KZ": "KAZ", "UZ": "UZB", "AZ": "AZE", "AM": "ARM",
    "GE": "GEO", "MN": "MNG",
    # Oceania
    "AU": "AUS", "NZ": "NZL",
    # Africa & Middle East
    "ZA": "ZAF", "EG": "EGY", "MA": "MAR", "TN": "TUN", "DZ": "DZA",
    "NG": "NGA", "KE": "KEN", "GH": "GHA", "CI": "CIV", "SN": "SEN",
    "TZ": "TZA", "UG": "UGA", "ET": "ETH", "AO": "AGO", "ZM": "ZMB",
    "CM": "CMR", "MU": "MUS", "BW": "BWA",
    "SA": "SAU", "AE": "ARE", "IL": "ISR", "QA": "QAT", "KW": "KWT",
    "BH": "BHR", "OM": "OMN", "JO": "JOR", "LB": "LBN", "IR": "IRN",
    "IQ": "IRQ",
}

_INDICATORS = {
    # output variable -> IFS series code
    "cpi_ifs_m":   "PCPI_IX",
    "ip_ifs_m":    "AIP_IX",
    "urate_ifs_m": "LUR_PT",
}

# Variables emitted in log() — index series (no negative/zero values).
# Rates / percentages (e.g. urate_ifs_m) stay in level form.
_LOG_VARS: frozenset[str] = frozenset({"cpi_ifs_m", "ip_ifs_m"})


def _http_json(indicator: str, codes: Iterable[str] | None, start_period: str) -> dict[str, Any]:
    """Fetch one indicator across countries; returns the parsed JSON.

    IMF SDMX-JSON key format: ``<freq>.<area>.<indicator>``.
    ``area`` is a ``+``-joined list of ISO-2 codes, or empty for "all".
    """
    if codes is None:
        area_key = ""
    else:
        # Map ISO-3 -> ISO-2 for the request; drop anything we don't have a mapping for.
        iso3_to_iso2 = {v: k for k, v in ISO2_TO_ISO3.items()}
        iso2 = [iso3_to_iso2[c] for c in codes if c in iso3_to_iso2]
        if not iso2:
            return {"CompactData": {"DataSet": {}}}
        area_key = "+".join(iso2)

    key = f"M.{area_key}.{indicator}"
    url = f"{_BASE}/{key}?startPeriod={start_period}"
    # Deferred so that importing this module needs no network stack, the same
    # reason `_oecd_sdmx.get_sdmx_csv` defers its own. An absent `requests` is
    # "this source has nothing", which is what a missing indicator already is.
    try:
        import requests
    except ImportError:
        return {"CompactData": {"DataSet": {}}}
    r = requests.get(url, timeout=_TIMEOUT)
    if r.status_code == 404:
        return {"CompactData": {"DataSet": {}}}
    r.raise_for_status()
    return r.json()


def _parse_compact_data(payload: dict[str, Any], indicator: str, out_var: str) -> pd.DataFrame:
    """Parse a CompactData JSON envelope into long-form."""
    series = payload.get("CompactData", {}).get("DataSet", {}).get("Series", [])
    if isinstance(series, dict):
        series = [series]
    if not series:
        return _EMPTY.copy()

    rows = []
    for s in series:
        ref_area = s.get("@REF_AREA")
        iso3 = ISO2_TO_ISO3.get(ref_area)
        if iso3 is None or not is_country(iso3):
            continue
        obs = s.get("Obs", [])
        if isinstance(obs, dict):
            obs = [obs]
        for o in obs:
            v = o.get("@OBS_VALUE")
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            rows.append({
                "code": iso3,
                "date": o.get("@TIME_PERIOD"),
                "value": fv,
            })

    if not rows:
        return _EMPTY.copy()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    if out_var in _LOG_VARS:
        df = df[df["value"] > 0]
        df["value"] = np.log(df["value"])
    df["variable"] = out_var
    df["sa_source"] = "none"  # IFS PCPI/AIP/LUR are NSA — X-13 will pick up
    df["source"] = f"IMF:IFS:{indicator}"
    return df[["code", "date", "variable", "value", "sa_source", "source"]]


def fetch(codes: Iterable[str] | None = None, *, start_period: str = "1990") -> pd.DataFrame:
    """Pull all three IFS monthly indicators for *codes* (None -> all available)."""
    frames: list[pd.DataFrame] = []
    for out_var, indicator in _INDICATORS.items():
        try:
            payload = _http_json(indicator, codes, start_period)
        except Exception as exc:
            print(f"[imf_ifs] {indicator} fetch failed: {exc}")
            continue
        frames.append(_parse_compact_data(payload, indicator, out_var))
    return pd.concat(frames, ignore_index=True) if frames else _EMPTY.copy()


__all__ = ["fetch", "ISO2_TO_ISO3"]
