"""Quantile autoregression (Koenker-Xiao 2006).

For a univariate series ``y``, fits

    Q_τ[y_{t+h} | y_t, …, y_{t-p+1}, controls_t] = α_τ + Σ_l β_{τ,l} y_{t-l+1} + γ_τ' z_t

at each requested quantile τ and horizon h. Bands by stationary
bootstrap (default) or by inverting the rank-test statistic.

Used as the time-series counterpart of :func:`puremacro.lp.lp_quantile`
when no second variable is at hand and the goal is one-step / multi-step
conditional-quantile forecasting (the workhorse of growth-at-risk).

References
----------
Koenker, R. and Xiao, Z. (2006). Quantile autoregression. JASA 101(475).
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..lp.quantile import _qreg


def qar(
    y: np.ndarray | pd.Series,
    *,
    quantiles: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
    horizons: Iterable[int] = (1, 4),
    p: int = 4,
    controls: np.ndarray | pd.DataFrame | None = None,
    n_boot: int = 200,
    alpha: float = 0.10,
    seed: int = 0,
) -> pd.DataFrame:
    """Quantile autoregression with bootstrapped bands.

    Parameters
    ----------
    y : array-like, shape (T,)
        Univariate target series.
    quantiles : sequence of float, default (0.05, 0.25, 0.50, 0.75, 0.95)
        Quantile levels ``τ ∈ (0, 1)`` to fit.
    horizons : iterable of int, default (1, 4)
        Forecast horizons (in periods of ``y``).
    p : int, default 4
        AR lag length.
    controls : (T, k) array-like, optional
        Extra regressors aligned with ``y``.
    n_boot : int, default 200
        Pairs-bootstrap replications for the bands.
    alpha : float, default 0.10
        Two-sided coverage = ``1 − α`` (so 0.10 ⇒ 90 % bands).
    seed : int, default 0
        RNG seed for the bootstrap.

    Returns
    -------
    pandas.DataFrame
        One row per ``(h, tau)`` pair with columns ``h``, ``tau``,
        ``beta0`` … ``betaK`` (intercept + lag and control coefficients),
        ``q_pred`` (predicted quantile evaluated at the last in-sample
        observation), and ``lo`` / ``hi`` bootstrap bands.
    """
    y = np.asarray(y, dtype=float).ravel()
    T = y.shape[0]
    horizons = list(horizons)
    quantiles = list(quantiles)
    rng = np.random.default_rng(seed)
    if controls is not None:
        Z = np.asarray(controls, dtype=float)
        if Z.ndim == 1:
            Z = Z[:, None]
    else:
        Z = None

    rows = []
    for h in horizons:
        # Build (X, y_h) panel with p lags and current controls.
        max_lag = p
        T_eff = T - max_lag - h
        if T_eff <= max_lag + 5:
            continue
        cols = [np.ones(T_eff)]
        for lag in range(1, p + 1):
            cols.append(y[max_lag - lag: T - lag - h])
        if Z is not None:
            for j in range(Z.shape[1]):
                cols.append(Z[max_lag: T - h, j])
        X = np.column_stack(cols)
        y_h = y[max_lag + h:]

        # Last-row predictor for q_pred (the t = T - 1 in-sample regressor).
        last_row = [1.0]
        for lag in range(1, p + 1):
            last_row.append(y[T - lag])
        if Z is not None:
            for j in range(Z.shape[1]):
                last_row.append(Z[T - 1, j])
        x_last = np.array(last_row)

        for tau in quantiles:
            beta_hat = _qreg(y_h, X, tau)
            q_pred = float(x_last @ beta_hat)

            # Pairs bootstrap band on q_pred.
            boot = np.empty(n_boot)
            n_obs = X.shape[0]
            for b in range(n_boot):
                idx = rng.integers(0, n_obs, size=n_obs)
                bb = _qreg(y_h[idx], X[idx], tau)
                boot[b] = float(x_last @ bb)
            lo = float(np.nanpercentile(boot, 100 * alpha / 2))
            hi = float(np.nanpercentile(boot, 100 * (1 - alpha / 2)))
            row = {"h": h, "tau": tau,
                    "q_pred": q_pred, "lo": lo, "hi": hi}
            for i, b in enumerate(beta_hat):
                row[f"beta{i}"] = float(b)
            rows.append(row)

    return pd.DataFrame(rows)


__all__ = ["qar"]
