"""Unified shock-series registry for the labor-transitions atlas.

Each loader returns a monthly pd.Series of shock values, z-scored against
its own sample. The registry covers the canonical empirical macro shock
series we have locally:

- LTUI (Federal Reserve labor-tech-uncertainty index)
- EPU national (Baker-Bloom-Davis)
- WUI (World Uncertainty Index)
- GPR (Iacoviello geopolitical risk)
- JLN macro and financial uncertainty (Jurado-Ludvigson-Ng)
- Romer-Romer tax shocks (narrative)
- Ramey defense news shocks (narrative)
- Bauer-Swanson FOMC path-factor surprises (monetary)
- TFP utilization-adjusted (Fernald)
- Oil shocks (Kilian) — best-effort, when cached

Each registry entry exposes a `label`, `category` (uncertainty / tax /
monetary / tfp / oil / geopolitical / narrative), and a `loader` returning
a monthly Series.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ShockSpec:
    name: str
    label: str
    category: str
    loader: Callable[[Path], pd.Series]


def _z(s: pd.Series) -> pd.Series:
    s = s.dropna()
    return (s - s.mean()) / s.std(ddof=0)


# -------------------- individual loaders --------------------------

def _load_ltui(root: Path) -> pd.Series:
    df = pd.read_parquet(root / "notebooks" / "data_cache" / "ltui_usa_monthly.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return _z(df.set_index("date")["ltui_z"])


def _load_epu_national(root: Path) -> pd.Series:
    xls = root / "data" / "raw" / "www.policyuncertainty.com" / "media_US_Historical_EPU_data.xlsx"
    df = pd.read_excel(xls)
    # Excel structure varies; auto-detect date and EPU columns
    cols = {c.lower(): c for c in df.columns}
    year_col = cols.get("year")
    month_col = cols.get("month")
    epu_col = next(
        (cols[c] for c in cols if "us" in c.lower() and "epu" in c.lower()), None
    ) or next((cols[c] for c in cols if "news" in c.lower() and "based" in c.lower()), None) \
       or next((cols[c] for c in cols if "epu" in c.lower()), None)
    if year_col is None or epu_col is None:
        raise ValueError(f"Could not parse EPU columns: {list(df.columns)}")
    df = df.dropna(subset=[year_col, epu_col]).copy()
    df["date"] = pd.to_datetime(
        df[year_col].astype(int).astype(str) + "-"
        + df[month_col].astype(int).astype(str).str.zfill(2) + "-01"
    )
    return _z(df.set_index("date")[epu_col].astype(float))


def _load_wui(root: Path) -> pd.Series:
    """World Uncertainty Index, GDP-weighted (Ahir, Bloom, Furceri).
    No US-specific sheet in the local copy; the world average is a strong
    correlate of US uncertainty and is used here as a proxy.
    """
    xls = root / "data" / "raw" / "worlduncertaintyindex.com" / "wp-content_uploads_2024_10_WUI_Data.xlsx"
    df = pd.read_excel(xls, sheet_name="F1", header=2)
    # cols are ['year', 'year2', 'WUI', ...]
    date_col = df.columns[0]
    val_col = next((c for c in df.columns if "WUI" in str(c).upper()), None)
    if val_col is None:
        raise ValueError(f"WUI column not found: {list(df.columns)}")
    df = df.dropna(subset=[date_col, val_col]).copy()
    raw = df[date_col].astype(str).str.lower().str.replace(" ", "")
    valid = raw.str.match(r"^\d{4}q[1-4]$", na=False)
    df = df.loc[valid].copy()
    df["qdate"] = pd.PeriodIndex(
        df[date_col].astype(str).str.replace(" ", ""), freq="Q"
    ).to_timestamp(how="end")
    q = df.set_index("qdate")[val_col].astype(float).dropna()
    m = q.resample("MS").ffill()
    return _z(m)


def _load_gpr(root: Path) -> pd.Series:
    xls = root / "data" / "raw" / "www.matteoiacoviello.com" / "gpr_files_data_gpr_export.xls"
    df = pd.read_excel(xls)
    date_col = next((c for c in df.columns if "month" in str(c).lower() or "date" in str(c).lower()), df.columns[0])
    val_col = next((c for c in df.columns if str(c).upper() == "GPR" or "GPR_RECENT" in str(c).upper()), None)
    if val_col is None:
        val_col = next((c for c in df.columns if "gpr" in str(c).lower()), None)
    if val_col is None:
        raise ValueError(f"GPR column not found in {list(df.columns)}")
    df["date"] = pd.to_datetime(df[date_col])
    return _z(df.set_index("date")[val_col].astype(float).dropna())


def _load_romer_tax(root: Path) -> pd.Series:
    df = pd.read_csv(root / "data" / "raw" / "romer_romer_2010_tax_shocks.csv")
    # Romer-Romer is quarterly; locate date column robustly
    cols_lower = {c.lower(): c for c in df.columns}
    date_col = (cols_lower.get("date") or cols_lower.get("qdate")
                or cols_lower.get("quarter") or cols_lower.get("yyyyq")
                or df.columns[0])
    val_col = (cols_lower.get("exogenous") or cols_lower.get("rr")
               or cols_lower.get("shock") or cols_lower.get("rr_exog"))
    if val_col is None:
        # Fall back: try first numeric column
        for c in df.columns:
            if c == date_col:
                continue
            try:
                df[c].astype(float)
                val_col = c
                break
            except Exception:
                continue
    if val_col is None:
        raise ValueError(f"Could not find RR shock column in {list(df.columns)}")
    df["date"] = pd.to_datetime(df[date_col])
    s = df.set_index("date")[val_col].astype(float).dropna()
    # Quarterly -> monthly: place shock at start of quarter, zero elsewhere
    out = s.resample("MS").asfreq().fillna(0.0)
    return _z(out)


def _load_ramey_defense(root: Path) -> pd.Series:
    df = pd.read_csv(root / "data" / "raw" / "ramey_2011_defense_news.csv")
    cols_lower = {c.lower(): c for c in df.columns}
    date_col = (cols_lower.get("date") or cols_lower.get("qdate")
                or df.columns[0])
    val_col = (cols_lower.get("news") or cols_lower.get("ramey_news")
               or cols_lower.get("ramey_def_news"))
    if val_col is None:
        for c in df.columns:
            if c == date_col:
                continue
            try:
                df[c].astype(float)
                val_col = c
                break
            except Exception:
                continue
    if val_col is None:
        raise ValueError(f"Could not find Ramey news column in {list(df.columns)}")
    df["date"] = pd.to_datetime(df[date_col])
    s = df.set_index("date")[val_col].astype(float).dropna()
    out = s.resample("MS").asfreq().fillna(0.0)
    return _z(out)


def _load_mertens_ravn(root: Path) -> pd.Series:
    df = pd.read_csv(root / "data" / "raw" / "mertens_ravn_2013_unanticipated_tax.csv")
    cols_lower = {c.lower(): c for c in df.columns}
    date_col = cols_lower.get("date", df.columns[0])
    # Pick the strongest numeric column other than date
    val_col = None
    for c in df.columns:
        if c == date_col:
            continue
        try:
            arr = df[c].astype(float)
            val_col = c
            break
        except Exception:
            continue
    if val_col is None:
        raise ValueError("No numeric column in Mertens-Ravn")
    df["date"] = pd.to_datetime(df[date_col])
    s = df.set_index("date")[val_col].astype(float).dropna()
    out = s.resample("MS").asfreq().fillna(0.0)
    return _z(out)


def _load_tpu_us(root: Path) -> pd.Series:
    """USA trade-policy uncertainty from the WUI Excel T8 sheet."""
    xls = root / "data" / "raw" / "worlduncertaintyindex.com" / "wp-content_uploads_2024_10_WUI_Data.xlsx"
    df = pd.read_excel(xls, sheet_name="T8", header=0)
    if "USA" not in df.columns:
        raise ValueError("USA column not in T8")
    date_col = df.columns[0]
    raw = df[date_col].astype(str).str.lower().str.replace(" ", "")
    valid = raw.str.match(r"^\d{4}q[1-4]$", na=False)
    df = df.loc[valid].copy()
    df["qdate"] = pd.PeriodIndex(
        df[date_col].astype(str).str.replace(" ", ""), freq="Q"
    ).to_timestamp(how="end")
    q = df.set_index("qdate")["USA"].astype(float).dropna()
    m = q.resample("MS").ffill()
    return _z(m)


def _load_fred_series(series_id: str) -> pd.Series:
    """Helper: fetch a single FRED series via fredapi. Requires FRED_API_KEY."""
    import os
    from fredapi import Fred
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY must be set in environment")
    s = Fred(api_key=key).get_series(series_id)
    s.index = pd.to_datetime(s.index)
    s.name = series_id
    return s


def _load_wpui_us(root: Path) -> pd.Series:
    """Ahir-Bloom-Furceri pandemic uncertainty for the United States.
    Native quarterly; we forward-fill to monthly so each month within a
    quarter inherits that quarter's pandemic-uncertainty value.
    """
    cache = root / "notebooks" / "data_cache" / "wpui_us.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["wpui"].astype(float)
    else:
        raw = _load_fred_series("WUPIUSA")
        s = raw.dropna().astype(float)
        s.name = "wpui"
        s.to_frame().rename_axis("date").reset_index().to_parquet(cache)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    # Force quarterly -> monthly forward-fill across the full month-span.
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s_monthly = s.reindex(full_idx).ffill()
    return _z(s_monthly)


def _load_pandemic_emv(root: Path) -> pd.Series:
    """BBD Equity Market Volatility: Infectious Disease Tracker (monthly)."""
    cache = root / "notebooks" / "data_cache" / "pandemic_emv.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["emv"].astype(float)
    else:
        raw = _load_fred_series("INFECTDISEMVTRACK")
        s = raw.dropna().astype(float)
        s.name = "emv"
        s.to_frame().rename_axis("date").reset_index().to_parquet(cache)
    return _z(s)


def _load_frbsf_tfp(root: Path) -> pd.Series:
    """Fernald utilization-adjusted TFP growth (quarterly)."""
    xls = root / "data" / "raw" / "www.frbsf.org" / "wp-content_uploads_quarterly_tfp.xlsx"
    df = pd.read_excel(xls, sheet_name="quarterly", skiprows=1)
    date_col = df.columns[0]
    tfp_col = next((c for c in df.columns if "tfp" in str(c).lower() and "util" in str(c).lower()), None)
    if tfp_col is None:
        tfp_col = next((c for c in df.columns if "dtfp_util" in str(c).lower()), None)
    if tfp_col is None:
        raise ValueError(f"Could not find TFP util-adjusted col in {list(df.columns)}")
    df = df.dropna(subset=[date_col, tfp_col]).copy()
    raw = df[date_col].astype(str).str.strip()
    # Keep only rows that look like a real YYYY-Q label (any of
    # 'YYYY:QN', 'YYYYQN', 'YYYY:N'). Drop summary rows like
    # 'FULLSAMPLEMEAN', '2004:4-2019:4', etc.
    valid = raw.str.match(r"^\d{4}[:\s]?Q?[1-4]$", na=False)
    df = df.loc[valid].copy()
    cleaned = (df[date_col].astype(str).str.strip()
               .str.replace(":", "Q").str.replace("QQ", "Q").str.replace(" ", ""))
    df["qdate"] = pd.PeriodIndex(cleaned, freq="Q").to_timestamp(how="end")
    q = df.set_index("qdate")[tfp_col].astype(float)
    m = q.resample("MS").ffill()
    return _z(m)


# -------------------- registry --------------------------

SHOCK_REGISTRY: list[ShockSpec] = [
    ShockSpec("LTUI",        "Fed labor-tech-uncertainty",       "uncertainty",  _load_ltui),
    ShockSpec("EPU_US",      "Baker-Bloom-Davis EPU (national)", "uncertainty",  _load_epu_national),
    ShockSpec("WUI",         "World Uncertainty Index (US)",     "uncertainty",  _load_wui),
    ShockSpec("TPU_US",      "WUI Trade-Policy Uncertainty (US)","trade",        _load_tpu_us),
    ShockSpec("WPUI_US",     "World Pandemic Uncertainty (US)",  "pandemic",     _load_wpui_us),
    ShockSpec("Pandemic_EMV","BBD Infectious-Disease EMV",       "pandemic",     _load_pandemic_emv),
    ShockSpec("GPR",         "Iacoviello geopolitical risk",     "geopolitical", _load_gpr),
    ShockSpec("RR_tax",      "Romer-Romer exogenous tax",        "tax",          _load_romer_tax),
    ShockSpec("Ramey_def",   "Ramey defense-news",               "fiscal",       _load_ramey_defense),
    ShockSpec("MR_tax",      "Mertens-Ravn unanticipated tax",   "tax",          _load_mertens_ravn),
    ShockSpec("Fernald_TFP", "Fernald util-adj TFP growth",      "tfp",          _load_frbsf_tfp),
]


def load_all_shocks(root: Path) -> dict[str, pd.Series]:
    """Attempt to load each registered shock; skip those that fail."""
    out: dict[str, pd.Series] = {}
    for spec in SHOCK_REGISTRY:
        try:
            s = spec.loader(root)
            if len(s.dropna()) > 24:
                out[spec.name] = s
                print(f"  loaded {spec.name:12s} ({spec.category:13s}): "
                      f"n={len(s.dropna()):4d}, "
                      f"{s.dropna().index.min().date()} -> {s.dropna().index.max().date()}")
            else:
                print(f"  skipped {spec.name:12s}: too short ({len(s)} obs)")
        except Exception as e:
            print(f"  failed {spec.name:12s}: {type(e).__name__}: {e}")
    return out


def categorise(name: str) -> str:
    for spec in SHOCK_REGISTRY:
        if spec.name == name:
            return spec.category
    return "other"
