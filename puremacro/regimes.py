"""Regime-indicator helpers — glue between break-detection / NBER-style
date dictionaries and downstream regime-conditional models
(``lp_state_dep``, ``rigobon_svar``, ``ms_var_fit``, etc.).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def breaks_to_regimes(break_indices: Iterable[int], T: int) -> np.ndarray:
    """Convert Bai-Perron break-index list to a (T,) regime indicator.

    ``break_indices`` is the list returned by ``bai_perron(...)["breaks"][m]``;
    each break index k means the regime number flips at observation k
    (so observations 0..k-1 are regime 0, k..k'-1 regime 1, etc.).
    """
    breaks = sorted(int(b) for b in break_indices)
    regimes = np.zeros(T, dtype=int)
    for r, b in enumerate(breaks):
        regimes[b:] = r + 1
    return regimes


def dates_to_regimes(
    date_index: pd.DatetimeIndex,
    regime_dates: dict,
    default_label: int = 0,
) -> pd.Series:
    """Map an arbitrary date-keyed regime spec onto a date index.

    ``regime_dates`` is a dict ``{label: [(start, end), ...]}`` giving
    the inclusive [start, end] intervals for each regime label. Periods
    not covered get ``default_label``.

    Example
    -------
    >>> nber = {"recession": [("2008-01-01", "2009-06-30"),
    ...                       ("2020-02-01", "2020-04-30")]}
    >>> dates_to_regimes(idx, {1: nber["recession"]}, default_label=0)
    """
    out = pd.Series(default_label, index=date_index, dtype=int)
    for label, intervals in regime_dates.items():
        for start, end in intervals:
            mask = (date_index >= pd.Timestamp(start)) & (date_index <= pd.Timestamp(end))
            out.loc[mask] = int(label)
    out.name = "regime"
    return out


def regime_dummies(
    regime_indicator: np.ndarray | pd.Series,
    n_regimes: int | None = None,
    drop_first: bool = False,
) -> pd.DataFrame:
    """Return a (T, K) one-hot encoding of regimes.

    Useful for plugging straight into LP / VAR design matrices.
    """
    arr = np.asarray(regime_indicator).ravel().astype(int)
    if n_regimes is None:
        n_regimes = int(arr.max() + 1)
    cols = {}
    rng_start = 1 if drop_first else 0
    for k in range(rng_start, n_regimes):
        cols[f"regime_{k}"] = (arr == k).astype(float)
    if isinstance(regime_indicator, pd.Series):
        return pd.DataFrame(cols, index=regime_indicator.index)
    return pd.DataFrame(cols)


def smoothed_to_indicator(
    smoothed_probs: np.ndarray,
    threshold: float = 0.5,
    target_regime: int | None = None,
) -> np.ndarray:
    """Convert MS-VAR smoothed probabilities to a hard regime indicator.

    ``target_regime`` defaults to ``argmax`` of each row. Use a specific
    column index when you want a binary "in regime k vs not" indicator.
    """
    sm = np.asarray(smoothed_probs)
    if target_regime is None:
        return np.argmax(sm, axis=1)
    return (sm[:, int(target_regime)] >= threshold).astype(int)


__all__ = [
    "breaks_to_regimes",
    "dates_to_regimes",
    "regime_dummies",
    "smoothed_to_indicator",
]
