"""External benchmark series for LTUI / LWUI validation (slice b foundation).

Currently exposes:
    fetch_bbd_ai_epu_monthly(cache_dir=None, column="aieu") -> pd.Series

Pulls the Baker-Bloom-Davis AI-related uncertainty data ("AIEU") published
at policyuncertainty.com. The canonical CSV is daily, with three indices:
    - daily_ai_index   : AI coverage frequency
    - daily_aie_index  : AI ∩ economics
    - daily_aieu_index : AI ∩ economics ∩ uncertainty (closest analog to
                         a "labor-tech-uncertainty" benchmark for LTUI)
We default to the AIEU column and resample to monthly mean.

Caches the parsed monthly series to ``<cache_dir>/bbd_aieu_monthly.parquet``
so repeat calls within a session and across runs avoid the network. Set
``PUREMACRO_REFETCH_AI_EPU=1`` in the environment to force re-fetch.
``PUREMACRO_AI_EPU_URL`` overrides the source URL if BBD relocates.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import requests


_DEFAULT_AI_EPU_URL = os.environ.get(
    "PUREMACRO_AI_EPU_URL",
    "https://www.policyuncertainty.com/media/All_Daily_AIEU_Data.csv",
)

_VALID_COLUMNS = {"ai", "aie", "aieu"}


def fetch_bbd_ai_epu_monthly(
    *,
    cache_dir: Path | str | None = None,
    column: str = "aieu",
    url: str | None = None,
    timeout: float = 30.0,
) -> pd.Series:
    """Return BBD AI-related uncertainty (AIEU) monthly series.

    Parameters
    ----------
    cache_dir : optional cache directory. If supplied, the parsed monthly
        series is persisted to ``<cache_dir>/bbd_aieu_monthly.parquet`` and
        re-read on subsequent calls.
    column : ``"ai"`` | ``"aie"`` | ``"aieu"`` (default). Selects which of
        the three published indices to return. ``"aieu"`` (AI ∩ economics
        ∩ uncertainty) is the closest analog to a labor-tech-uncertainty
        benchmark.
    url : optional override of the CSV URL.
    timeout : HTTP timeout in seconds.

    Returns
    -------
    pd.Series indexed by month-start Timestamps. Float64.
    """
    if column not in _VALID_COLUMNS:
        raise ValueError(f"column must be one of {_VALID_COLUMNS}; got {column!r}")
    src_url = url or _DEFAULT_AI_EPU_URL
    cache_path: Path | None = None
    refetch = os.environ.get("PUREMACRO_REFETCH_AI_EPU") == "1"

    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"bbd_{column}_monthly.parquet"
        if cache_path.exists() and not refetch:
            return pd.read_parquet(cache_path)[column]

    resp = requests.get(src_url, timeout=timeout)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text))
    raw.columns = [c.strip().lower() for c in raw.columns]
    required = {"day", "month", "year"}
    if not required.issubset(raw.columns):
        raise ValueError(
            f"AIEU CSV is missing day/month/year columns; got {list(raw.columns)}"
        )
    val_col = f"daily_{column}_index"
    if val_col not in raw.columns:
        candidates = [c for c in raw.columns if c not in required]
        raise ValueError(
            f"AIEU CSV missing expected column {val_col!r}; available: {candidates}"
        )
    raw["date"] = pd.to_datetime(dict(year=raw["year"], month=raw["month"], day=raw["day"]))
    daily = raw.set_index("date")[val_col].astype(float).sort_index()
    monthly = daily.resample("MS").mean()
    monthly.name = column

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        monthly.to_frame(column).to_parquet(cache_path, index=True)

    return monthly


_DEFAULT_GPR_URL = os.environ.get(
    "PUREMACRO_GPR_URL",
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls",
)


def fetch_caldara_iacoviello_gpr_monthly(
    *,
    cache_dir: Path | str | None = None,
    column: str = "GPR",
    url: str | None = None,
    timeout: float = 60.0,
) -> pd.Series:
    """Return Caldara-Iacoviello Geopolitical Risk (GPR) monthly series.

    The canonical XLS at matteoiacoviello.com publishes the global GPR
    plus per-country variants ``GPRC_<ISO3>`` (e.g. ``GPRC_USA``).

    Parameters
    ----------
    cache_dir : optional cache directory. Series is persisted to
        ``<cache_dir>/ci_gpr_{column}_monthly.parquet``.
    column : ``"GPR"`` (global; default), ``"GPRT"`` (threats), ``"GPRA"``
        (acts), or ``"GPRC_<ISO3>"`` country-specific.
    url : optional override.
    timeout : HTTP timeout in seconds.

    Returns
    -------
    pd.Series indexed by month-start Timestamps. Float64.
    """
    src_url = url or _DEFAULT_GPR_URL
    cache_path: Path | None = None
    refetch = os.environ.get("PUREMACRO_REFETCH_GPR") == "1"

    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"ci_gpr_{column.lower()}_monthly.parquet"
        if cache_path.exists() and not refetch:
            return pd.read_parquet(cache_path)[column]

    raw = pd.read_excel(src_url, timeout=timeout) if False else pd.read_excel(src_url)
    if "month" not in raw.columns or column not in raw.columns:
        raise ValueError(
            f"GPR XLS missing expected columns; got {list(raw.columns)[:5]}..."
        )
    raw["month"] = pd.to_datetime(raw["month"])
    ser = raw.set_index("month")[column].astype(float).dropna().sort_index()
    ser.index = ser.index.to_period("M").to_timestamp()
    ser.name = column

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ser.to_frame(column).to_parquet(cache_path, index=True)

    return ser


_DEFAULT_WUI_URL = os.environ.get(
    "PUREMACRO_WUI_URL",
    "https://worlduncertaintyindex.com/wp-content/uploads/2026/04/WUI_Data.xlsx",
)
_DEFAULT_BBD_COUNTRY_URL = os.environ.get(
    "PUREMACRO_BBD_COUNTRY_URL",
    "https://www.policyuncertainty.com/media/All_Country_Data.xlsx",
)


def fetch_ahir_bloom_furceri_wui_panel(
    *,
    cache_dir: Path | str | None = None,
    country: str | None = None,
    url: str | None = None,
    timeout: float = 60.0,
) -> pd.DataFrame | pd.Series:
    """Return Ahir-Bloom-Furceri World Uncertainty Index panel (quarterly).

    Pulls the ``T2`` sheet of the canonical WUI workbook — a 144-country
    quarterly panel back to 1952 (most country series start later).

    Parameters
    ----------
    cache_dir : optional cache dir; persisted as
        ``<cache_dir>/abf_wui_panel_quarterly.parquet``.
    country : optional ISO3 code to extract a single country as a Series.
        ``None`` returns the full DataFrame (columns = ISO3 codes,
        index = quarter-start Timestamps).
    url : optional override of the XLSX URL.

    Returns
    -------
    pd.DataFrame indexed by quarter-start Timestamps, columns = ISO3.
    If ``country`` is set, returns pd.Series for that ISO3.
    """
    src_url = url or _DEFAULT_WUI_URL
    cache_path: Path | None = None
    refetch = os.environ.get("PUREMACRO_REFETCH_WUI") == "1"

    if cache_dir is not None:
        cache_path = Path(cache_dir) / "abf_wui_panel_quarterly.parquet"
        if cache_path.exists() and not refetch:
            df = pd.read_parquet(cache_path)
        else:
            df = None
    else:
        df = None

    if df is None:
        raw = pd.read_excel(src_url, sheet_name="T2")
        raw = raw.rename(columns={"year": "quarter"})
        # 'YYYYqQ' → Timestamp at quarter-start
        raw["date"] = pd.to_datetime(
            raw["quarter"].astype(str).str.replace("q", "Q"), errors="coerce",
        )
        raw = raw.dropna(subset=["date"]).set_index("date")
        df = raw.drop(columns=["quarter"]).astype(float).sort_index()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=True)

    if country is not None:
        if country not in df.columns:
            raise ValueError(
                f"country {country!r} not in WUI panel; have {len(df.columns)} ISO3s"
            )
        return df[country]
    return df


def fetch_bbd_country_epu_monthly(
    *,
    country: str = "US",
    cache_dir: Path | str | None = None,
    url: str | None = None,
    timeout: float = 60.0,
) -> pd.Series:
    """Return Baker-Bloom-Davis country-level EPU monthly series.

    Pulls a single column from ``All_Country_Data.xlsx``. Available
    countries (column names in the workbook): Australia, Brazil, Canada,
    Chile, China, France, Germany, Greece, India, Ireland, Italy, Japan,
    Korea, Pakistan, Russia, Spain, Singapore, UK, US, Sweden, Mexico,
    plus ``GEPU_current`` and ``GEPU_ppp`` global aggregates.

    Parameters
    ----------
    country : column name in the workbook (case-sensitive).
    cache_dir : optional cache dir; the full workbook is cached as
        ``<cache_dir>/bbd_country_epu_panel.parquet`` once and re-sliced
        on subsequent calls.
    """
    src_url = url or _DEFAULT_BBD_COUNTRY_URL
    cache_path: Path | None = None
    refetch = os.environ.get("PUREMACRO_REFETCH_BBD_COUNTRY") == "1"

    if cache_dir is not None:
        cache_path = Path(cache_dir) / "bbd_country_epu_panel.parquet"
        if cache_path.exists() and not refetch:
            df = pd.read_parquet(cache_path)
        else:
            df = None
    else:
        df = None

    if df is None:
        raw = pd.read_excel(src_url)
        # Drop any trailing notes rows where Year is non-numeric.
        raw = raw[pd.to_numeric(raw["Year"], errors="coerce").notna()].copy()
        raw["Year"] = raw["Year"].astype(int)
        raw["Month"] = pd.to_numeric(raw["Month"], errors="coerce").astype("Int64")
        raw = raw.dropna(subset=["Month"]).copy()
        raw["Month"] = raw["Month"].astype(int)
        raw["date"] = pd.to_datetime(
            raw["Year"].astype(str) + "-"
            + raw["Month"].astype(str).str.zfill(2) + "-01"
        )
        df = raw.drop(columns=["Year", "Month"]).set_index("date").sort_index()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=True)

    if country not in df.columns:
        raise ValueError(
            f"country {country!r} not in BBD All_Country_Data; "
            f"available: {sorted(df.columns)}"
        )
    ser = df[country].astype(float).dropna()
    ser.name = f"bbd_epu_{country.lower().replace(' ', '_')}"
    return ser


__all__ = [
    "fetch_bbd_ai_epu_monthly",
    "fetch_caldara_iacoviello_gpr_monthly",
    "fetch_ahir_bloom_furceri_wui_panel",
    "fetch_bbd_country_epu_monthly",
]
