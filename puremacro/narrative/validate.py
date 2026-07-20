"""Diagnostics and benchmark comparison for a narrative instrument."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .types import NarrativeEvent


def event_density(events: Iterable[NarrativeEvent]) -> dict:
    """Events-per-quarter histogram + simple gap stats.

    Returns
    -------
    dict with
        per_quarter : pd.Series — count of events per quarter
        n_total     : int
        n_zero_qs   : int — quarters with zero events (within the
                            envelope min..max of observed dates)
        max_gap_q   : int — longest run of consecutive zero-event quarters
    """
    events = list(events)
    if not events:
        return {"per_quarter": pd.Series(dtype=int),
                "n_total": 0, "n_zero_qs": 0, "max_gap_q": 0}
    dates = [e.date for e in events]
    s = pd.Series(1, index=pd.DatetimeIndex(dates))
    per_q = s.resample("QS").sum().fillna(0).astype(int)
    n_zero = int((per_q == 0).sum())
    # max consecutive zero-quarter run
    if (per_q == 0).any():
        runs = (per_q != 0).cumsum()
        max_gap = (per_q == 0).groupby(runs).sum().max()
    else:
        max_gap = 0
    return {
        "per_quarter": per_q,
        "n_total": len(events),
        "n_zero_qs": n_zero,
        "max_gap_q": int(max_gap),
    }


def weak_iv_summary(
    z: np.ndarray | pd.Series,
    x_endog: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    *,
    controls: np.ndarray | pd.DataFrame | None = None,
    grid_points: int = 200,
) -> dict:
    """Weak-IV diagnostics on a candidate narrative instrument.

    Wraps the existing ``puremacro.inference.weak_iv`` machinery:
        - first-stage F (Cragg-Donald and Kleibergen-Paap),
        - Anderson-Rubin p-value at 2SLS estimate,
        - Montiel Olea-Stock-Watson identification-robust band.
    """
    from ..inference.weak_iv import (
        cragg_donald_f, kleibergen_paap_f,
        anderson_rubin_test, msw_bands,
    )
    z_arr = np.asarray(z, dtype=float).ravel()
    x_arr = np.asarray(x_endog, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if controls is None:
        ctl = np.ones((len(y_arr), 1))
    else:
        ctl = np.asarray(controls, dtype=float)
        if ctl.ndim == 1:
            ctl = ctl[:, None]
        ctl = np.column_stack([np.ones(len(y_arr)), ctl])

    # Drop any rows with NaN in z, x, y.
    mask = ~(np.isnan(z_arr) | np.isnan(x_arr) | np.isnan(y_arr))
    if ctl.shape[1] > 1:
        mask = mask & ~np.any(np.isnan(ctl), axis=1)
    z_arr, x_arr, y_arr, ctl = z_arr[mask], x_arr[mask], y_arr[mask], ctl[mask]

    cd = cragg_donald_f(y_arr, x_arr, z_arr)
    kp = kleibergen_paap_f(y_arr, x_arr, z_arr,
                            W=ctl[:, 1:] if ctl.shape[1] > 1 else None)

    # 2SLS β̂ for AR test:
    # x_hat = projection of x on (z, controls).
    Zfull = np.column_stack([ctl, z_arr])
    proj = Zfull @ np.linalg.solve(Zfull.T @ Zfull, Zfull.T @ x_arr)
    beta_2sls = float((proj @ y_arr) / (proj @ x_arr)) if (proj @ x_arr) != 0 else float("nan")
    ar = anderson_rubin_test(beta_2sls, y_arr, x_arr, z_arr,
                              controls=ctl[:, 1:] if ctl.shape[1] > 1 else None)
    grid = np.linspace(beta_2sls - 5.0, beta_2sls + 5.0, grid_points)
    msw_lo, msw_hi = msw_bands(grid, y_arr, x_arr, z_arr,
                                 controls=ctl[:, 1:] if ctl.shape[1] > 1 else None)
    return {
        "n_obs": int(len(y_arr)),
        "first_stage_f_cd": float(cd),
        "first_stage_f_kp": float(kp),
        "beta_2sls": beta_2sls,
        "AR_p_value_at_2sls": ar.p_value,
        "MSW_band": (msw_lo, msw_hi),
    }


def compare_to(
    z_user: pd.Series,
    z_benchmark: pd.Series,
    *,
    max_lag: int = 8,
) -> dict:
    """Compare a custom narrative instrument to a benchmark series.

    - correlation at lag 0
    - lead-lag CCF over [-max_lag, +max_lag]
    - event overlap on quarters where both are non-zero
    """
    s = pd.concat([z_user.rename("user"), z_benchmark.rename("bench")], axis=1).dropna()
    if s.empty:
        return {"n_obs": 0, "correlation": float("nan")}
    rho0 = float(s["user"].corr(s["bench"]))
    ccf = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = s["user"][lag:], s["bench"][:len(s) - lag]
        else:
            a, b = s["user"][:lag], s["bench"][-lag:]
        ccf[lag] = float(np.corrcoef(a.values, b.values)[0, 1]) if len(a) > 1 else float("nan")
    nonzero_overlap = int(((s["user"] != 0) & (s["bench"] != 0)).sum())
    return {
        "n_obs": int(len(s)),
        "correlation": rho0,
        "ccf": ccf,
        "nonzero_overlap_quarters": nonzero_overlap,
    }


__all__ = ["event_density", "weak_iv_summary", "compare_to"]
