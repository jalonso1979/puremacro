"""Long-history US quarterly loader (1950Q1-present) for the teaching track.

Builds a single DataFrame with DatetimeIndex at quarter-start, containing
FRED macro series and the long-history uncertainty menu (News-EPU historical,
JLN, LMN, WUI, GPR) plus the post-1985 EPU/VIX for continuity. Monthly
upstream series are averaged to quarter-start.

Upstream fetchers emit their own canonical variable names; this loader
maps a subset of them to pedagogical aliases used by the T3 notebook:

    fetcher      upstream variable   loader alias
    -----------  ------------------  ---------------
    jln          jln_macro_1         jln_macro_h1
    lmn          lmn_real            lmn_real_h1
    lmn          lmn_fin             lmn_fin_h1
    epu (3-cmp)  epu_m               epu_3comp
    fred         vix                 vix  (passthrough)
    fred         log_gdp_real, ...   (passthrough, no rename)
    wui          wui_q               wui_q  (via fetch_q())
    gpr          gpr_m               gpr_m
    news-hist    epu_news_hist       epu_news_hist

The FRED block is panel-backed: when ``FRED_API_KEY`` is unset we
read the same series from ``data/processed/panel_Q.parquet`` and
``data/processed/panel_M.parquet`` (built from FRED at panel-build
time). This keeps T3 offline-friendly like the other T-notebooks.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


_PANEL_Q = Path("data/processed/panel_Q.parquet")
_PANEL_M = Path("data/processed/panel_M.parquet")
_FRED_PANEL_Q_VARS = (
    "log_gdp_real", "log_gfcf_real", "log_hours", "log_emp",
    "log_wages_real", "log_income_real", "urate", "short_rate",
    "log_deflator",
)
_FRED_PANEL_M_VARS = ("vix",)


def _fetch_fred_from_panel() -> pd.DataFrame:
    """Build the FRED-equivalent tidy frame from the prebuilt panels.

    Returns the same schema as :func:`puremacro.fetch.fred.fetch_all_us`
    (``code, date, variable, value, sa_source, source``) so the rest of
    ``load_us_long`` is unchanged. Used as the FRED-API-key-less path.
    """
    parts: list[pd.DataFrame] = []
    if _PANEL_Q.exists():
        pq = pd.read_parquet(_PANEL_Q)
        sub_q = pq[(pq["code"] == "USA") & (pq["variable"].isin(_FRED_PANEL_Q_VARS))]
        if not sub_q.empty:
            parts.append(sub_q[["code", "date", "variable", "value", "sa_source", "source"]])
    if _PANEL_M.exists():
        pm = pd.read_parquet(_PANEL_M)
        sub_m = pm[(pm["code"] == "USA") & (pm["variable"].isin(_FRED_PANEL_M_VARS))]
        if not sub_m.empty:
            parts.append(sub_m[["code", "date", "variable", "value", "sa_source", "source"]])
    if not parts:
        return pd.DataFrame(columns=["code", "date", "variable", "value", "sa_source", "source"])
    return pd.concat(parts, ignore_index=True)


# Indirection layer so tests can monkeypatch these names. Imports are
# deferred so this module is importable in environments where optional
# fetch-time dependencies (e.g. fredapi) are not installed.
def _fetch_fred_all() -> pd.DataFrame:
    """Fetch FRED series, with a panel-Q/M fallback when FRED_API_KEY is unset.

    The panel was itself built from FRED at panel-build time, so values
    are identical. The fallback keeps T3 runnable offline (matching
    every other T-notebook in the project, which all read the panels).
    """
    if not os.environ.get("FRED_API_KEY"):
        return _fetch_fred_from_panel()
    from ..fetch import fred as _fred_mod
    df = _fred_mod.fetch_all_us()
    # If the live fetch came back empty (every series threw), still
    # fall back to the panel — empty would otherwise drop the FRED
    # block silently and break downstream key access.
    if df.empty:
        return _fetch_fred_from_panel()
    return df


def _fetch_news_epu_hist() -> pd.DataFrame:
    from ..fetch import epu_news_historical as _epu_hist_mod

    return _epu_hist_mod.fetch()


def _fetch_jln() -> pd.DataFrame:
    from ..fetch import jln as _jln_mod

    return _jln_mod.fetch()


def _fetch_lmn() -> pd.DataFrame:
    from ..fetch import lmn as _lmn_mod

    return _lmn_mod.fetch()


def _fetch_wui() -> pd.DataFrame:
    """Quarterly WUI. `fetch_q()` returns variable='wui_q'."""
    from ..fetch import wui as _wui_mod

    return _wui_mod.fetch_q()


def _fetch_gpr() -> pd.DataFrame:
    from ..fetch import gpr as _gpr_mod

    return _gpr_mod.fetch()


def _fetch_epu_3comp() -> pd.DataFrame:
    """3-component EPU from policyuncertainty.com (variable='epu_m')."""
    from ..fetch import epu as _epu_mod

    return _epu_mod.fetch()


def _remap_variable(tidy: pd.DataFrame, rename: dict[str, str]) -> pd.DataFrame:
    """Rename values in the 'variable' column using the provided map."""
    if not rename:
        return tidy
    out = tidy.copy()
    out["variable"] = out["variable"].replace(rename)
    return out


def _tidy_to_wide_quarterly(tidy: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Pivot a tidy long-form frame (date+variable+value) to wide quarterly.

    Any monthly observations are averaged within the quarter-start period.
    Filters to USA rows only (when the frame has a 'code' column).
    """
    tidy = tidy.copy()
    if "code" in tidy.columns:
        tidy = tidy[tidy["code"] == "USA"]
    tidy = tidy[tidy["variable"].isin(variables)]
    tidy["date"] = pd.to_datetime(tidy["date"])
    tidy["qdate"] = tidy["date"].dt.to_period("Q").dt.to_timestamp(how="start")
    wide = (
        tidy.groupby(["qdate", "variable"])["value"].mean().unstack("variable")
    )
    wide.index.name = "date"
    return wide


def load_us_long(start: str = "1950Q1", end: str = "2024Q4") -> pd.DataFrame:
    """Return a quarterly USA DataFrame covering the long history."""
    start_ts = pd.Period(start, freq="Q").to_timestamp(how="start")
    end_ts = pd.Period(end, freq="Q").to_timestamp(how="start")

    # FRED vars include `vix` (VIXCLS) which is aggregated to quarter by mean.
    fred_vars = [
        "log_gdp_real", "log_gfcf_real", "log_hours", "log_emp",
        "log_wages_real", "log_income_real", "urate", "short_rate",
        "log_deflator", "vix",
    ]

    parts = [
        _tidy_to_wide_quarterly(_fetch_fred_all(), fred_vars),
        _tidy_to_wide_quarterly(_fetch_news_epu_hist(), ["epu_news_hist"]),
        _tidy_to_wide_quarterly(
            _remap_variable(_fetch_jln(), {"jln_macro_1": "jln_macro_h1"}),
            ["jln_macro_h1"],
        ),
        _tidy_to_wide_quarterly(
            _remap_variable(_fetch_lmn(),
                            {"lmn_real": "lmn_real_h1", "lmn_fin": "lmn_fin_h1"}),
            ["lmn_real_h1", "lmn_fin_h1"],
        ),
        _tidy_to_wide_quarterly(_fetch_wui(), ["wui_q"]),
        _tidy_to_wide_quarterly(_fetch_gpr(), ["gpr_m"]),
        _tidy_to_wide_quarterly(
            _remap_variable(_fetch_epu_3comp(), {"epu_m": "epu_3comp"}),
            ["epu_3comp"],
        ),
    ]

    out = pd.concat(parts, axis=1)
    out = out.loc[(out.index >= start_ts) & (out.index <= end_ts)]
    out = out.sort_index()
    # Reindex to guarantee all quarters in the range are present.
    grid = pd.date_range(start_ts, end_ts, freq="QS")
    out = out.reindex(grid)
    out.index.name = "date"
    return out
