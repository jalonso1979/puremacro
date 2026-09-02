"""OECD QNA expenditure / sectoral data from a local workbook.

The Volatility track curates a workbook (``Volatility/QNA.xlsx``) with
real consumption / government / trade aggregates and sectoral employment
+ hours for ~50 OECD members. This fetcher reads it and emits the
project-standard long-form schema.

Two sheets are read:
- OECD-QNA-EXP  -> real consumption (RCON), government (RGOV), exports
                  (RX), imports (RM). The sheet also has EMP and HOUR
                  columns but they are NOT used here — log_emp_qna and
                  log_hours_qna come exclusively from OECD-SECTOR sums
                  (wider country coverage).
- OECD-SECTOR   -> sectoral RVAB / EMP / HRS, summed across sectors per
                  (code, period) to produce log_emp_qna / log_hours_qna.

When both EXP-sheet EMP and SECTOR-sheet sum exist, the SECTOR sum wins
(it covers more countries -- the EXP-sheet EMP column has gaps).

Path resolution priority:
    1. explicit ``path`` argument
    2. ``OECD_QNA_LOCAL_XLSX`` env var
    3. default ``MAV/Volatility/QNA.xlsx`` on the user's Drive
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

#: Root of the separate project's data tree this module reads when it is
#: present. It was an absolute path naming one person's Google Drive account,
#: so on every other machine it silently could not resolve. `PUREMACRO_MAV_ROOT`
#: is the same variable `puremacro/examples/*.py` and `fetch/wui*.py` read.
_DEFAULT_DRIVE = Path(os.environ.get(
    "PUREMACRO_MAV_ROOT",
    Path.home() / "Library" / "CloudStorage" / "My Drive"))
_DEFAULT_PATH = _DEFAULT_DRIVE / "Volatility" / "QNA.xlsx"

_EMPTY_SCHEMA = pd.DataFrame(
    columns=["code", "date", "variable", "value", "sa_source", "source"]
)

_EXP_VARS = {
    # workbook column -> output variable name
    "RGDP":  "log_gdp_real",
    "RGFCF": "log_gfcf_real",
    "RCON":  "log_con_real",
    "RGOV":  "log_govcon_real",
    "RX":    "log_exports_real",
    "RM":    "log_imports_real",
}


def _resolve_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    if "OECD_QNA_LOCAL_XLSX" in os.environ:
        return Path(os.environ["OECD_QNA_LOCAL_XLSX"])
    return _DEFAULT_PATH


def _quarter_to_date(period: pd.Series) -> pd.Series:
    return pd.PeriodIndex(period.astype(str), freq="Q").to_timestamp(how="start")


def _exp_long(exp: pd.DataFrame) -> pd.DataFrame:
    if exp.empty:
        return _EMPTY_SCHEMA.copy()
    out_parts = []
    exp = exp.copy()
    exp["date"] = _quarter_to_date(exp["period"])
    for col, var in _EXP_VARS.items():
        if col not in exp.columns:
            continue
        sub = exp[["code", "date", col]].dropna(subset=[col])
        sub = sub[sub[col] > 0].copy()
        sub["value"] = np.log(sub[col].astype(float))
        sub["variable"] = var
        sub["sa_source"] = "oecd"
        sub["source"] = f"Volatility/QNA.xlsx OECD-QNA-EXP:{col}"
        out_parts.append(sub[["code", "date", "variable", "value", "sa_source", "source"]])
    return pd.concat(out_parts, ignore_index=True) if out_parts else _EMPTY_SCHEMA.copy()


def _sector_long(sec: pd.DataFrame) -> pd.DataFrame:
    if sec.empty:
        return _EMPTY_SCHEMA.copy()
    sec = sec.copy()
    sec["date"] = _quarter_to_date(sec["period"])
    out_parts = []
    for col, var in (("EMP", "log_emp_qna"), ("HRS", "log_hours_qna")):
        if col not in sec.columns:
            continue
        # Sum across sectors per (code, date)
        tot = (sec[["code", "date", col]]
               .dropna(subset=[col])
               .groupby(["code", "date"], as_index=False)[col].sum())
        tot = tot[tot[col] > 0]
        tot["value"] = np.log(tot[col].astype(float))
        tot["variable"] = var
        tot["sa_source"] = "oecd"
        tot["source"] = f"Volatility/QNA.xlsx OECD-SECTOR:sum_{col}"
        out_parts.append(tot[["code", "date", "variable", "value", "sa_source", "source"]])
    return pd.concat(out_parts, ignore_index=True) if out_parts else _EMPTY_SCHEMA.copy()


def fetch_extended(path: str | Path | None = None) -> pd.DataFrame:
    """Long-form QNA extension: RCON / RGOV / RX / RM + sectoral EMP/HRS sums.

    Returns an empty DataFrame (with the correct columns) when the workbook
    is missing -- the build then continues without these variables instead
    of crashing.
    """
    p = _resolve_path(path)
    if not p.exists():
        return _EMPTY_SCHEMA.copy()

    try:
        exp = pd.read_excel(p, sheet_name="OECD-QNA-EXP")
    except ValueError as exc:
        print(f"[oecd_qna_local] OECD-QNA-EXP sheet not found in {p}: {exc}")
        exp = pd.DataFrame()
    try:
        sec = pd.read_excel(p, sheet_name="OECD-SECTOR")
    except ValueError as exc:
        print(f"[oecd_qna_local] OECD-SECTOR sheet not found in {p}: {exc}")
        sec = pd.DataFrame()
    frames = [f for f in (_exp_long(exp), _sector_long(sec)) if not f.empty]
    if not frames:
        return _EMPTY_SCHEMA.copy()
    return pd.concat(frames, ignore_index=True)


__all__ = ["fetch_extended"]
