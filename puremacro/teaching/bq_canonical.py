"""Canonical Blanchard-Quah (1989) identification for teaching.

Thin wrapper over `puremacro.var.identify.bq.bq_svar` that:
  1. First-differences the permanent variable (log GDP), keeps the
     transitory variable (urate) in levels.
  2. Invokes the existing `bq_svar` with `permanent_var_idx=0`.
  3. Applies sign normalization so the supply shock raises GDP on
     impact and the demand shock raises urate on impact.
  4. Returns (point, lower, upper) IRFs with canonical axis order
     `[horizon, response_variable, shock_index]`, with GDP responses
     already cumulated to levels (matching the `bq_svar` convention).

Also provides `bq_employment_epu` — the corrected version of the
current T3 mis-specified system `[Δlog N, Δlog EPU]`, retained as a
teaching moment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from puremacro.var.identify.bq import bq_svar


def _normalize_sign(
    point: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    *, impact_rule: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flip sign of each shock so that (response, shock) at h=0 is positive.

    `impact_rule` is a list of (response_idx, shock_idx) pairs giving,
    for each shock, which response must be positive on impact.

    Arrays are canonical shape (H+1, n, n): axis0=horizon, axis1=response,
    axis2=shock.
    """
    point = point.copy()
    lo = lo.copy()
    hi = hi.copy()
    for resp_idx, shock_idx in impact_rule:
        if point[0, resp_idx, shock_idx] < 0:
            point[:, :, shock_idx] *= -1
            lo_flipped = -hi[:, :, shock_idx]
            hi_flipped = -lo[:, :, shock_idx]
            lo[:, :, shock_idx] = lo_flipped
            hi[:, :, shock_idx] = hi_flipped
    return point, lo, hi


def bq_gdp_urate(
    df: pd.DataFrame,
    *,
    p: int = 4,
    horizon: int = 40,
    n_boot: int = 1000,
    ci: float = 0.9,
    seed: int = 12345,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical BQ on [Δlog GDP, urate].

    Supply shock (column 0) = permanent effect on log GDP.
    Demand shock (column 1) = transitory (long-run effect on GDP = 0).
    """
    sub = df[["log_gdp_real", "urate"]].dropna().copy()
    Y = np.column_stack([
        np.diff(sub["log_gdp_real"].values),
        sub["urate"].values[1:],
    ])
    _bq_res = bq_svar(
        Y, p=p, horizon=horizon,
        permanent_var_idx=0, n_boot=n_boot, ci=ci, seed=seed,
    )
    point = _bq_res.irf_point   # (H+1, n, n)
    lo = _bq_res.irf_lower
    hi = _bq_res.irf_upper
    return _normalize_sign(
        point, lo, hi,
        impact_rule=[(0, 0), (1, 1)],
    )


def bq_employment_epu(
    df: pd.DataFrame,
    *,
    p: int = 4,
    horizon: int = 20,
    n_boot: int = 1000,
    ci: float = 0.9,
    seed: int = 12345,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corrected version of T3's current `[Δlog N, Δlog EPU]` BQ cell.

    **Pedagogical note:** log EPU is bounded/stationary, so Δlog EPU is
    over-differenced and the "uncertainty shock has zero long-run effect
    on log N" restriction is not the canonical Blanchard-Quah (1989)
    identification. Retained for teaching — the notebook flags this
    mis-specification explicitly.
    """
    sub = df[["log_emp", "log_epu_for_bq"]].dropna().copy()
    Y = np.column_stack([
        np.diff(sub["log_emp"].values) * 100,
        np.diff(sub["log_epu_for_bq"].values) * 100,
    ])
    _bq_res = bq_svar(
        Y, p=p, horizon=horizon,
        permanent_var_idx=0, n_boot=n_boot, ci=ci, seed=seed,
    )
    point = _bq_res.irf_point   # (H+1, n, n)
    lo = _bq_res.irf_lower
    hi = _bq_res.irf_upper
    return _normalize_sign(
        point, lo, hi,
        impact_rule=[(0, 0), (1, 1)],
    )
