"""Threshold VECM (Hansen-Seo 2002, simplified).

A two-regime threshold error-correction model with regime-dependent
adjustment speeds:

    Δy_t = (μ^L + α^L (β' y_{t-1}) + Γ^L Δy_{t-1}) 1{w_{t-d} ≤ γ}
         + (μ^H + α^H (β' y_{t-1}) + Γ^H Δy_{t-1}) 1{w_{t-d} >  γ}
         + ε_t

The threshold variable ``w_t`` is by default the cointegrating residual
``β' y_{t-d}``, which makes this a "threshold cointegration" model in
the Balke-Fomby / Hansen-Seo sense — adjustment toward the long-run
equilibrium happens at different speeds in different regimes (e.g.,
fast above a threshold, slow below).

Estimation: profile over (β, γ) on a coarse grid, then OLS per regime.
For pedagogy we treat ``β`` as known (or estimated by Engle-Granger
in a first stage) and only grid-search the threshold ``γ``.

References
----------
Balke, N.S. and Fomby, T.B. (1997). Threshold cointegration. IER 38(3).
Hansen, B.E. and Seo, B. (2002). Testing for two-regime threshold
    cointegration in vector error-correction models. JoE 110(2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TVECMResult:
    beta: np.ndarray              # cointegrating vector
    threshold: float
    delay: int
    alpha_low: np.ndarray         # adjustment speeds in low regime
    alpha_high: np.ndarray
    intercept_low: np.ndarray
    intercept_high: np.ndarray
    Gamma_low: np.ndarray         # short-run dynamics, low regime
    Gamma_high: np.ndarray
    Sigma_low: np.ndarray
    Sigma_high: np.ndarray
    n_low: int
    n_high: int
    rss_total: float


def _engle_granger_beta(Y: np.ndarray) -> np.ndarray:
    """Engle-Granger first-stage: y_0 = β' y_rest + u; return β as a
    cointegrating vector with leading 1."""
    if Y.shape[1] < 2:
        raise ValueError("need at least 2 variables for cointegration")
    y0 = Y[:, 0]
    Y_rest = Y[:, 1:]
    X = np.column_stack([np.ones(len(y0)), Y_rest])
    coefs, *_ = np.linalg.lstsq(X, y0, rcond=None)
    beta = np.r_[1.0, -coefs[1:]]
    # No constant in beta — store separately if needed.
    return beta


def tvecm_fit(
    Y: np.ndarray,
    *,
    p: int = 1,
    delay: int = 1,
    n_threshold_grid: int = 50,
    trim: float = 0.15,
    beta: Optional[np.ndarray] = None,
) -> TVECMResult:
    """Fit a two-regime threshold VECM by grid search.

    Parameters
    ----------
    Y : (T, n) ndarray of I(1) cointegrated variables.
    p : VAR lag order (so ECM uses ``p-1`` lags of differences).
    delay : delay ``d`` for the threshold variable w_{t-d}.
    n_threshold_grid : number of candidate γ values.
    trim : minimum fraction of obs in each regime.
    beta : pre-specified cointegrating vector. Defaults to Engle-Granger.
    """
    Y = np.asarray(Y, dtype=float)
    T, n = Y.shape
    if beta is None:
        beta = _engle_granger_beta(Y)

    ec = Y @ beta             # cointegrating residual w_t
    dY = np.diff(Y, axis=0)   # (T-1, n)

    # Build short-run lag matrix Z (lagged differences) of dim (T-1-?, n*(p-1)).
    k_lags = max(0, p - 1)
    if k_lags == 0:
        Z = np.zeros((len(dY), 0))
    else:
        rows = []
        for t in range(k_lags, len(dY)):
            rows.append(np.concatenate([dY[t - j - 1] for j in range(k_lags)]))
        Z = np.array(rows)

    start = k_lags + delay  # need w_{t-d} and lagged dY
    # align: indep variables at index t use ec at t-delay+1 ... let's be careful:
    # ECM uses ec[t-1] (i.e., lagged level). For row t in dY (which is
    # diff at calendar position t+1), the ec to use is ec[t]. Threshold
    # variable is w_{t-d+1} = ec[t-d+1]. We index everything from start.
    dY_dep = dY[start:]
    ec_lag = ec[start:-1] if False else ec[start:][:len(dY_dep)]
    # Simpler: align by position. Use ec at indices matching dY_dep.
    # dY[t] corresponds to Y[t+1] - Y[t]. Lagged level is Y[t] hence ec[t].
    ec_lag = ec[start:start + len(dY_dep)]
    threshold_var = ec[start - delay:start - delay + len(dY_dep)]
    Z_use = Z[start - k_lags:start - k_lags + len(dY_dep)] if k_lags > 0 else \
            np.zeros((len(dY_dep), 0))

    # Candidate thresholds from quantiles of threshold_var.
    cand = np.quantile(threshold_var, np.linspace(trim, 1 - trim,
                                                   n_threshold_grid))
    best = None
    for gamma in cand:
        mask_lo = threshold_var <= gamma
        n_lo = int(mask_lo.sum())
        n_hi = len(threshold_var) - n_lo
        min_seg = int(trim * len(threshold_var))
        if n_lo < min_seg or n_hi < min_seg:
            continue
        # Per-regime regressors: [const, ec_lag, Z]
        def _ols(mask):
            X = np.column_stack([np.ones(mask.sum()), ec_lag[mask],
                                  Z_use[mask] if k_lags > 0
                                  else np.zeros((mask.sum(), 0))])
            B = np.linalg.lstsq(X, dY_dep[mask], rcond=None)[0]
            r = dY_dep[mask] - X @ B
            return B, r
        B_lo, r_lo = _ols(mask_lo)
        B_hi, r_hi = _ols(~mask_lo)
        rss = float((r_lo ** 2).sum() + (r_hi ** 2).sum())
        if best is None or rss < best["rss"]:
            best = {"rss": rss, "gamma": float(gamma),
                     "mask_lo": mask_lo, "B_lo": B_lo, "r_lo": r_lo,
                     "B_hi": B_hi, "r_hi": r_hi,
                     "n_lo": n_lo, "n_hi": n_hi}
    if best is None:
        raise RuntimeError("No threshold split satisfied the trim.")

    def _unpack(B):
        # B shape: (2 + k_lags*n, n). Rows 0/1 are scalar coefs across
        # equations, rows 2: are coefficients on lagged differences.
        intercept = B[0]
        alpha = B[1]
        Gamma = B[2:].T if k_lags > 0 else np.zeros((n, 0))
        return intercept, alpha, Gamma

    int_lo, alpha_lo, Gamma_lo = _unpack(best["B_lo"])
    int_hi, alpha_hi, Gamma_hi = _unpack(best["B_hi"])
    Sigma_lo = (best["r_lo"].T @ best["r_lo"]) / max(1, best["n_lo"] - 2)
    Sigma_hi = (best["r_hi"].T @ best["r_hi"]) / max(1, best["n_hi"] - 2)

    return TVECMResult(
        beta=beta,
        threshold=best["gamma"],
        delay=delay,
        alpha_low=alpha_lo,
        alpha_high=alpha_hi,
        intercept_low=int_lo,
        intercept_high=int_hi,
        Gamma_low=Gamma_lo,
        Gamma_high=Gamma_hi,
        Sigma_low=Sigma_lo,
        Sigma_high=Sigma_hi,
        n_low=best["n_lo"],
        n_high=best["n_hi"],
        rss_total=best["rss"],
    )


__all__ = ["tvecm_fit", "TVECMResult"]
