"""OECD QNA labor fetcher: total-economy employment + hours.

Pulls from OECD SDMX dataflow ``DSD_NAMAIN1@DF_QNA`` with:
  TRANSACTION = EMP, ACTIVITY = _T (total economy), SECTOR = S1
  UNIT_MEASURE = PS (persons) or H (hours)
  ADJUSTMENT  = Y (SA) preferred, falls back to N + flags x13_pending

Returns long-form DataFrame: code, date, variable, value, sa_source, source.

Used by ``src.build_panel`` to fill labor coverage gaps for countries not
present in the curated local Volatility/QNA.xlsx workbook (the workbook
remains the primary source for European countries).
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
    """Fetch one OECD SDMX REST URL via the cached helper.

    Cached on disk via ``_oecd_sdmx.get_sdmx_csv``; returns empty frame on
    any failure so callers can treat missing-data and rate-limited
    symmetrically.
    """
    from ._oecd_sdmx import get_sdmx_csv
    return get_sdmx_csv(agency_flow, key, start_period)
    # Legacy direct path (kept for reference only — never executed):
    url = f"{_BASE}/{agency_flow}/{key}?startPeriod={start_period}&format={_FMT}"
    try:
        r = requests.get(url, timeout=_TIMEOUT)
    except requests.RequestException:
        return pd.DataFrame()
    if not r.ok or len(r.text) < 200:
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(r.text), low_memory=False)
    except Exception:
        return pd.DataFrame()


def _quarter_to_date(s: pd.Series) -> pd.Series:
    return pd.PeriodIndex(s.astype(str), freq="Q").to_timestamp(how="start")


def fetch_qna_labor(codes: Iterable[str] | None = None,
                    *, start_period: str = "1995") -> pd.DataFrame:
    """Fetch QNA total-economy employment + hours for the given country codes.

    Returns a long-form DataFrame with rows for log_emp_qna and log_hours_qna.
    Empty frame returned on any error or for codes with no data.
    """
    if codes is None:
        code_key = ""
    else:
        code_key = "+".join(c.upper() for c in codes)

    agency_flow = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,"
    # 13 dims: FREQ.ADJUSTMENT.REF_AREA.SECTOR.COUNTERPART.TRANSACTION.INSTR.
    #          ACTIVITY.EXPENDITURE.UNIT_MEASURE.PRICE_BASE.TRANSFORMATION.TABLE
    # We hit it as broadly as possible (FREQ=Q only) and filter post-hoc.
    key = f"Q..{code_key}............."
    raw = _get_csv(agency_flow, key, start_period)
    if raw.empty:
        return _EMPTY.copy()

    required_cols = {"TRANSACTION", "ACTIVITY", "SECTOR", "UNIT_MEASURE",
                     "ADJUSTMENT", "REF_AREA", "TIME_PERIOD", "OBS_VALUE"}
    if not required_cols.issubset(set(raw.columns)):
        return _EMPTY.copy()

    # Filter to total-economy labor: TRANSACTION=EMP, ACTIVITY=_T, SECTOR=S1
    base = raw[(raw["TRANSACTION"] == "EMP") & (raw["ACTIVITY"] == "_T")
               & (raw["SECTOR"] == "S1")].copy()
    if base.empty:
        return _EMPTY.copy()

    parts = []

    def _emit(sub: pd.DataFrame, *, variable: str, sa: str, source: str) -> None:
        if sub.empty:
            return
        sub = sub.drop_duplicates(subset=["REF_AREA", "TIME_PERIOD"], keep="first").copy()
        sub["OBS_VALUE"] = pd.to_numeric(sub["OBS_VALUE"], errors="coerce")
        sub = sub.dropna(subset=["OBS_VALUE"])
        sub = sub[sub["OBS_VALUE"] > 0]
        if sub.empty:
            return
        sub["date"] = _quarter_to_date(sub["TIME_PERIOD"])
        sub["value"] = np.log(sub["OBS_VALUE"].astype(float))
        sub["code"] = sub["REF_AREA"]
        sub["variable"] = variable
        sub["sa_source"] = sa
        sub["source"] = source
        parts.append(sub[["code", "date", "variable", "value", "sa_source", "source"]])

    # ----- log_emp_qna (UNIT_MEASURE=PS) -----
    emp = base[base["UNIT_MEASURE"] == "PS"].copy()
    if not emp.empty:
        # Prefer SA (ADJUSTMENT=Y) over NSA per code
        emp_y = emp[emp["ADJUSTMENT"] == "Y"]
        emp_n = emp[emp["ADJUSTMENT"] == "N"]
        codes_with_sa = set(emp_y["REF_AREA"].unique())
        emp_n = emp_n[~emp_n["REF_AREA"].isin(codes_with_sa)]
        _emit(emp_y, variable="log_emp_qna", sa="oecd",
              source="OECD:DSD_NAMAIN1@DF_QNA:EMP/PS")
        _emit(emp_n, variable="log_emp_qna", sa="x13_pending",
              source="OECD:DSD_NAMAIN1@DF_QNA:EMP/PS")

    # ----- log_hours_qna (UNIT_MEASURE=H) -----
    hrs = base[base["UNIT_MEASURE"] == "H"].copy()
    if not hrs.empty:
        hrs_y = hrs[hrs["ADJUSTMENT"] == "Y"]
        hrs_n = hrs[hrs["ADJUSTMENT"] == "N"]
        codes_with_sa = set(hrs_y["REF_AREA"].unique())
        hrs_n = hrs_n[~hrs_n["REF_AREA"].isin(codes_with_sa)]
        _emit(hrs_y, variable="log_hours_qna", sa="oecd",
              source="OECD:DSD_NAMAIN1@DF_QNA:EMP/H")
        _emit(hrs_n, variable="log_hours_qna", sa="x13_pending",
              source="OECD:DSD_NAMAIN1@DF_QNA:EMP/H")

    if not parts:
        return _EMPTY.copy()
    return pd.concat(parts, ignore_index=True)


__all__ = ["fetch_qna_labor"]
