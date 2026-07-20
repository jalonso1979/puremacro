"""Regime helpers for the long-history uncertainty study.

* :data:`CALENDAR_REGIMES` — fixed calendar-period boundaries (Cold War end,
  Great Moderation, GFC era, Pandemic-plus).
* :func:`add_calendar_regime` — assigns each row a regime label.
* :func:`bai_perron_breaks` — endogenous break detection on a univariate
  series. Thin adapter over :func:`puremacro.tests.breaks.bai_perron`.
* :func:`state_indicator_smooth` — Auerbach-Gorodnichenko-style logistic
  state in (0, 1), monotone in the lagged percentile of the input series.
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd

CALENDAR_REGIMES: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "cold_war_end":     (pd.Timestamp("1985-01-01"), pd.Timestamp("1991-12-31")),
    "great_moderation": (pd.Timestamp("1992-01-01"), pd.Timestamp("2007-12-31")),
    "gfc_era":          (pd.Timestamp("2008-01-01"), pd.Timestamp("2019-12-31")),
    "pandemic_plus":    (pd.Timestamp("2020-01-01"), pd.Timestamp("2099-12-31")),
}


class BreakResult(TypedDict):
    date: pd.Timestamp
    index: int
    ci_lo: pd.Timestamp
    ci_hi: pd.Timestamp


def add_calendar_regime(panel: pd.DataFrame, *, date_col: str = "date",
                        out_col: str = "regime") -> pd.DataFrame:
    """Add a categorical regime column based on :data:`CALENDAR_REGIMES`.

    Rows with dates outside any defined regime keep ``NaN`` in *out_col*.
    With the bundled CALENDAR_REGIMES the union spans 1985-01..2099-12 so
    no real-world date should be ``NaN``.
    """
    out = panel.copy()
    labels = pd.Series(np.nan, index=out.index, dtype=object)
    for name, (lo, hi) in CALENDAR_REGIMES.items():
        mask = (out[date_col] >= lo) & (out[date_col] <= hi)
        labels.loc[mask] = name
    out[out_col] = labels
    return out


def bai_perron_breaks(series: pd.Series, *, max_breaks: int = 5,
                      min_segment: int = 24) -> list[BreakResult]:
    """Endogenous break detection on a univariate series.

    Thin adapter over :func:`puremacro.tests.breaks.bai_perron`. Selects
    the partition chosen by the sequential supF test (Bai-Perron 2003)
    and converts it to a list of :class:`BreakResult` dicts so callers
    can read break dates and approximate 90% CIs by date.

    The CI is approximated as ``date ± min_segment // 4`` (in series
    index units) rather than the full Bai (1997) asymptotic — see
    docstring of the original module for rationale.
    """
    # Import via puremacro package when the editable install is active;
    # fall back to a file-relative load so the function works from any cwd
    # (including the repo root, where a namespace-path `puremacro` shadows
    # the editable install and makes `puremacro.tests` resolve to the wrong
    # directory).
    try:
        from puremacro.tests.breaks import bai_perron as _bp_dict
    except ImportError:
        import importlib.util as _ilu
        import pathlib as _pl
        _breaks_path = _pl.Path(__file__).resolve().parents[1] / "tests" / "breaks.py"
        _spec = _ilu.spec_from_file_location("_puremacro_breaks", _breaks_path)
        _mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
        _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
        _bp_dict = _mod.bai_perron

    s = series.dropna().sort_index()
    if len(s) < 2 * min_segment:
        return []
    trim = max(min_segment / len(s), 0.05)
    res = _bp_dict(s.values.astype(float), max_breaks=max_breaks, trim=trim)
    selected = res["selected_m"]
    if selected == 0:
        return []
    idxs = res["breaks"][selected]
    half_ci = max(min_segment // 4, 1)
    out: list[BreakResult] = []
    for i in idxs:
        if i <= 0 or i >= len(s):
            continue
        d = s.index[i]
        lo = s.index[max(0, i - half_ci)]
        hi = s.index[min(len(s) - 1, i + half_ci)]
        out.append({"date": d, "index": int(i), "ci_lo": lo, "ci_hi": hi})
    return out


def state_indicator_smooth(series: pd.Series, *, lag: int = 12,
                           gamma: float = 1.5,
                           threshold_quantile: float = 0.5) -> pd.Series:
    """Logistic smooth-transition state F(z) ∈ (0, 1) à la Auerbach-Gorodnichenko.

    z is the lagged-by-*lag* percentile rank (in [0, 1]) of *series*.
    The transition pivots around *threshold_quantile* and uses *gamma*
    as steepness, scaled by 0.25:

        F(z) = 1 / (1 + exp(-gamma * (z - threshold_quantile) / 0.25))

    Default `threshold_quantile=0.5` reproduces the median-pivot behaviour.
    """
    z = series.shift(lag).rank(pct=True)
    F = 1.0 / (1.0 + np.exp(-gamma * (z - threshold_quantile) / 0.25))
    F.name = "state"
    return F


__all__ = [
    "CALENDAR_REGIMES",
    "BreakResult",
    "add_calendar_regime",
    "bai_perron_breaks",
    "state_indicator_smooth",
]
