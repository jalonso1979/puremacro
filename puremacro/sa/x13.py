"""Per-unit seasonal adjustment with a three-tier engine fallback.

``deseasonalize_x13`` adjusts each unit's series preferring, in order:

1. the external **X-13ARIMA-SEATS binary** (statsmodels wrapper), when
   reachable and the series is long and gap-free enough for it;
2. the native pure-Python **X-11/ARIMA engine** (:mod:`puremacro.sa.x11`)
   --- the Pyodide/browser default, since no binary can run there;
3. **STL** (:mod:`puremacro.sa.stl`), the universal last resort for
   series too short for the airline fit.

This makes the browser default a genuine X-11-class adjuster rather than
STL. Pass ``engines={}`` to learn which tier ran for each unit.

The X-13 binary is located via the ``X13PATH`` environment variable
(directory containing ``x13as``), ``shutil.which('x13as')`` on PATH, or
``~/.local/bin``; when none succeed the native engine takes over
automatically.

Public API:
    deseasonalize_x13(df, value_col, by, date_col, freq='Q', engines=None)
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


def _x11_native_one(values: np.ndarray, dates: pd.DatetimeIndex,
                    period: int) -> np.ndarray | None:
    """Native X-11/ARIMA on one series (:mod:`puremacro.sa.x11`).

    Returns the seasonally-adjusted array (1:1 with ``values``, NaNs
    preserved) or ``None`` if the series cannot be adjusted natively
    (all-NaN, too short for the airline fit, uninterpolable gaps), so the
    caller can fall back to STL. Pure Python — the browser default when
    no X-13 binary is present."""
    s = pd.Series(values, index=dates)
    nan_mask = s.isna()
    if nan_mask.all():
        return None
    s_interp = s.interpolate(limit_direction="both")
    # airline fit needs (1-B)(1-B^s)y plus a few residuals: s+1 lost to
    # differencing, then >= 3s for the MLE.
    if s_interp.isna().any() or len(s_interp) < 4 * period + 1:
        return None
    try:
        from .x11 import x11_arima
        res = x11_arima(s_interp.to_numpy(dtype=float), period=period,
                        mode="auto")
        sa = res.sa
    except Exception as e:  # pragma: no cover - depends on series shape
        logger.debug("native X-11 failed (%s); falling back to STL", e)
        return None
    return np.where(nan_mask.values, np.nan, sa)


def _x13_one(values: np.ndarray, dates: pd.DatetimeIndex, period: int,
             *, engine: list | None = None) -> np.ndarray:
    """Seasonally adjust one series, preferring the X-13 binary, then the
    native X-11 engine, then STL.

    Returns the SA series aligned 1:1 with ``values`` (NaNs preserved).
    If ``engine`` is a list it is appended the tag of the engine that
    actually ran (``'binary'`` / ``'native'`` / ``'stl'``)."""
    def _rec(tag: str) -> None:
        if engine is not None:
            engine.append(tag)

    s = pd.Series(values, index=dates).copy()
    nan_mask = s.isna()
    if nan_mask.all():
        _rec("stl")
        return values.astype(float)
    s_interp = s.interpolate(limit_direction="both")
    if s_interp.isna().any():
        _rec("stl")
        return values.astype(float)

    # 1) The genuine X-13 binary, when reachable. statsmodels'
    # x13_arima_analysis needs a regular gap-free DatetimeIndex with a
    # known frequency and >= 3 full years; series failing that skip to
    # the native engine rather than STL.
    if _X13_DIR is not None and len(s_interp) >= 3 * period:
        period_letter = "Q" if period == 4 else "M"
        s_x = s_interp.copy()
        s_x.index = (pd.DatetimeIndex(s_x.index).to_period(period_letter)
                     .to_timestamp(how="start"))
        if not s_x.index.has_duplicates:
            try:
                from statsmodels.tsa.x13 import x13_arima_analysis
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = x13_arima_analysis(
                        s_x, x12path=_X13_DIR, outlier=True, trading=False,
                        forecast_periods=0, print_stdout=False,
                    )
                sa = pd.Series(res.seasadj.values, index=res.seasadj.index)
                sa = sa.reindex(s_x.index).to_numpy(dtype=float)
                _rec("binary")
                return np.where(nan_mask.values, np.nan, sa)
            except Exception as e:  # pragma: no cover - series-shape dependent
                logger.debug("X-13 binary failed (%s); trying native X-11", e)

    # 2) The native X-11/ARIMA engine (pure Python; the Pyodide default).
    nat = _x11_native_one(values, dates, period)
    if nat is not None:
        _rec("native")
        return nat

    # 3) STL, the universal last resort.
    _rec("stl")
    return _stl_one(values, period=period)


def deseasonalize_x13(
    df: pd.DataFrame,
    value_col: str,
    *,
    by: str,
    date_col: str = "date",
    freq: Literal["Q", "M"] = "Q",
    min_obs: int = 24,
    engines: dict | None = None,
) -> pd.Series:
    """Return a seasonally-adjusted Series aligned to ``df.index``.

    Per unit, adjustment prefers the external X-13ARIMA-SEATS binary,
    falls back to the native pure-Python X-11/ARIMA engine
    (:mod:`puremacro.sa.x11`) when the binary is unavailable or declines,
    and to STL only when neither applies. This makes the browser default
    a genuine X-11-class adjuster rather than STL. Same group contract as
    :func:`puremacro.sa.stl.deseasonalize`; units below ``min_obs`` are
    returned as NaN. Pass ``engines`` as a dict to receive, per unit, the
    tag of the engine that actually ran (``'binary'`` / ``'native'`` /
    ``'stl'``) --- so callers can assert ``'stl'`` never sneaks in."""
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
        tag: list = []
        sa = _x13_one(v, d, period=period, engine=tag)
        if engines is not None and tag:
            engines[unit] = tag[-1]
        out.loc[sub["_orig_idx"].values] = sa

    return out


__all__ = [
    "deseasonalize_x13",
    "residual_seasonality_F",
    "x13_available",
]
