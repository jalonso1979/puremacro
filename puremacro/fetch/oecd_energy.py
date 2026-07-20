"""OECD CPI — energy sub-component (COICOP CP045 = electricity, gas, fuel).

Pulls from the OECD SDMX dataflow ``DSD_PRICES@DF_PRICES_ALL`` with
EXPENDITURE=CP045, UNIT_MEASURE=IX (index level). NSA only (OECD does
not publish SA for this component); flagged ``sa_source='x13_pending'``
for downstream X-13 enforcement.

Returns long-form schema: ``code, date, variable, value, sa_source, source``.
"""
from __future__ import annotations

import io
from typing import Iterable

import numpy as np
import pandas as pd
import requests

_BASE = "https://sdmx.oecd.org/public/rest/data"
_FMT = "csvfilewithlabels"
_TIMEOUT = 180

_EMPTY = pd.DataFrame(columns=["code", "date", "variable", "value", "sa_source", "source"])


def _get_csv(agency_flow: str, key: str, start_period: str) -> pd.DataFrame:
    """Issue one OECD SDMX REST call via the cached helper (empty on failure)."""
    from ._oecd_sdmx import get_sdmx_csv
    return get_sdmx_csv(agency_flow, key, start_period)
    # Legacy direct path (kept for reference only — never executed):
    url = f"{_BASE}/{agency_flow}/{key}?startPeriod={start_period}&format={_FMT}"
    try:
        r = requests.get(url, timeout=_TIMEOUT)
    except Exception:
        return pd.DataFrame()
    if r.status_code == 404 or not r.ok or len(r.text) < 200:
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(r.text), low_memory=False)
    except Exception:
        return pd.DataFrame()


def fetch_energy_cpi(
    codes: Iterable[str] | None = None,
    *,
    start_period: str = "1995",
) -> pd.DataFrame:
    """Fetch monthly energy CPI (COICOP CP045) for the given country codes.

    Returns an empty long-form frame on error. The quarterly version is
    built downstream in ``build_panel`` via the standard M->Q rollup.
    """
    code_key = "+".join(c.upper() for c in codes) if codes else ""
    # 8 dims for DSD_PRICES@DF_PRICES_ALL:
    # REF_AREA . FREQ . METHODOLOGY . MEASURE . UNIT_MEASURE . EXPENDITURE . ADJUSTMENT . TRANSFORMATION
    key = f"{code_key}.M....CP045.."
    raw = _get_csv("OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,", key, start_period)
    if raw.empty:
        return _EMPTY.copy()

    needed = {"REF_AREA", "TIME_PERIOD", "OBS_VALUE", "UNIT_MEASURE",
              "EXPENDITURE", "ADJUSTMENT"}
    if not needed.issubset(raw.columns):
        return _EMPTY.copy()

    sub = raw[(raw["UNIT_MEASURE"] == "IX") & (raw["EXPENDITURE"] == "CP045")].copy()
    if sub.empty:
        return _EMPTY.copy()

    out_parts: list[pd.DataFrame] = []
    for code, grp in sub.groupby("REF_AREA"):
        # Prefer methodology='N' (national) over HICP if both present.
        if "METHODOLOGY" in grp.columns and "N" in grp["METHODOLOGY"].dropna().values:
            grp = grp[grp["METHODOLOGY"] == "N"]
        # Prefer SA, else NSA.
        adj_vals = grp["ADJUSTMENT"].dropna().unique()
        if "S" in adj_vals:
            grp = grp[grp["ADJUSTMENT"] == "S"]
            sa_src = "oecd"
        elif "N" in adj_vals:
            grp = grp[grp["ADJUSTMENT"] == "N"]
            sa_src = "x13_pending"
        else:
            continue
        grp = grp.drop_duplicates(subset=["TIME_PERIOD"], keep="first")
        grp = grp[pd.to_numeric(grp["OBS_VALUE"], errors="coerce") > 0]
        if grp.empty:
            continue
        try:
            dates = pd.to_datetime(grp["TIME_PERIOD"].astype(str) + "-01")
        except Exception:
            continue
        out = pd.DataFrame({
            "code": code,
            "date": dates.values,
            "variable": "log_cpi_energy_m",
            "value": np.log(pd.to_numeric(grp["OBS_VALUE"], errors="coerce").astype(float).values),
            "sa_source": sa_src,
            "source": "OECD:DSD_PRICES@DF_PRICES_ALL:CP045",
        })
        out_parts.append(out)
    if not out_parts:
        return _EMPTY.copy()
    return pd.concat(out_parts, ignore_index=True)


__all__ = ["fetch_energy_cpi"]
