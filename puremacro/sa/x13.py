"""Per-unit X-13ARIMA-SEATS seasonal adjustment.

Mirrors the :mod:`src.sa.stl` API so call sites can swap method via
``deseasonalize_x13`` (preferred) and fall back to STL when the X-13
binary is unavailable or rejects a series (too short, too noisy, etc.).

The X-13 binary is located via:
1. The ``X13PATH`` environment variable (directory containing ``x13as``);
2. ``shutil.which('x13as')`` on the user's PATH;
3. ``~/.local/bin`` as a last resort.

If none of those succeed, callers receive a :class:`RuntimeError` and
should catch it to fall back to STL.

Public API:
    deseasonalize_x13(df, value_col, by, date_col, freq='Q') -> pd.Series
    residual_seasonality_F(...)                              -> re-exported
"""
from __future__ import annotations

import logging
import os
import shutil
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .stl import residual_seasonality_F  # re-export
from .stl import _stl_one  # STL fallback used per-unit

logger = logging.getLogger(__name__)

_PERIOD = {"Q": 4, "M": 12}
_FREQ_TS = {"Q": "QS", "M": "MS"}


def _resolve_x13_dir() -> str | None:
    env = os.environ.get("X13PATH", "")
    if env:
        p = Path(env)
        if p.is_file():
            return str(p.parent)
        if p.is_dir() and any((p / n).exists() for n in ("x13as", "x13ashtml")):
            return str(p)
    which = shutil.which("x13as") or shutil.which("x13ashtml")
    if which:
        return str(Path(which).parent)
    home_local = Path.home() / ".local" / "bin"
    if (home_local / "x13as").exists():
        return str(home_local)
    return None


_X13_DIR = _resolve_x13_dir()


def x13_available() -> bool:
    """Return True if the X-13 binary is reachable."""
    return _X13_DIR is not None


def _x13_one(values: np.ndarray, dates: pd.DatetimeIndex, period: int) -> np.ndarray:
    """Run x13 on a single series; fall back to STL on failure.

    Returns the seasonally-adjusted series aligned 1:1 with ``values``.
    NaNs in input are preserved in output.
    """
    s = pd.Series(values, index=dates).copy()
    nan_mask = s.isna()
    if nan_mask.all():
        return values.astype(float)
    s_interp = s.interpolate(limit_direction="both")
    if s_interp.isna().any():
        return values.astype(float)

    if _X13_DIR is None:
        return _stl_one(values, period=period)

    # statsmodels.x13_arima_analysis is fussy: it requires a regular
    # DatetimeIndex with a known frequency, no missing periods, and at
    # least 3 full years. Skip series that don't satisfy this.
    if len(s_interp) < 3 * period:
        return _stl_one(values, period=period)

    period_letter = "Q" if period == 4 else "M"
    s_x = s_interp.copy()
    s_x.index = pd.DatetimeIndex(s_x.index).to_period(period_letter).to_timestamp(how="start")
    if s_x.index.has_duplicates:
        return _stl_one(values, period=period)

    try:
        from statsmodels.tsa.x13 import x13_arima_analysis

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = x13_arima_analysis(
                s_x,
                x12path=_X13_DIR,
                outlier=True,
                trading=False,
                forecast_periods=0,
                print_stdout=False,
            )
        # ``res.seasadj`` aligns with the input index (may be slightly
        # shorter if X-13 dropped leading/trailing NaNs).
        sa = pd.Series(res.seasadj.values, index=res.seasadj.index)
        sa = sa.reindex(s_x.index).to_numpy(dtype=float)
    except Exception as e:  # pragma: no cover - depends on series shape
        logger.debug("X-13 failed (%s); falling back to STL", e)
        return _stl_one(values, period=period)

    sa = np.where(nan_mask.values, np.nan, sa)
    return sa


def deseasonalize_x13(
    df: pd.DataFrame,
    value_col: str,
    *,
    by: str,
    date_col: str = "date",
    freq: Literal["Q", "M"] = "Q",
    min_obs: int = 24,
) -> pd.Series:
    """Return a seasonally-adjusted Series aligned to ``df.index`` (X-13 first).

    Same contract as :func:`src.sa.stl.deseasonalize` but uses X-13 ARIMA
    SEATS per unit; if X-13 is unavailable or fails on a series, the
    function falls back to STL silently. Units below ``min_obs`` are
    returned as NaN.
    """
    if freq not in _PERIOD:
        raise ValueError(f"freq must be 'Q' or 'M', got {freq!r}")
    period = _PERIOD[freq]

    out = pd.Series(np.nan, index=df.index, dtype=float)
    work = df[[by, date_col, value_col]].copy()
    work["_orig_idx"] = df.index
    work = work.sort_values([by, date_col])

    for unit, sub in work.groupby(by, sort=False):
        v = sub[value_col].to_numpy(dtype=float)
        if np.isfinite(v).sum() < min_obs:
            continue
        d = pd.DatetimeIndex(sub[date_col].to_numpy())
        sa = _x13_one(v, d, period=period)
        out.loc[sub["_orig_idx"].values] = sa

    return out


__all__ = [
    "deseasonalize_x13",
    "residual_seasonality_F",
    "x13_available",
]
