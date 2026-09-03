"""Volatility-model diagnostic battery.

  ARCH-LM test (Engle 1982): regress squared residuals on q lags;
      the LM statistic ``T R²`` is χ²(q) under H0 of no ARCH.

  Ljung-Box on squared residuals: a complementary portmanteau test for
      remaining ARCH structure in the residuals of a fitted GARCH-type
      model.

These two together cover the standard "is my volatility model
mis-specified?" battery you see in any GARCH paper.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2, f as f_dist


def arch_lm_test(u: np.ndarray, q: int = 5) -> dict:
    """Engle (1982) Lagrange-multiplier test for ARCH effects.

    Parameters
    ----------
    u : 1-D array
        Residuals of the conditional-mean model (or raw returns if
        you're testing for ARCH effects directly).
    q : int
        Number of lags of squared residuals to include.

    Returns
    -------
    dict with ``lm_stat`` (= T R²), ``p_value`` (χ²(q)), ``f_stat``,
    ``f_p_value`` (Wald-style F variant), ``df``.
    """
    u = np.asarray(u, dtype=float).ravel()
    if u.size <= q + 1:
        raise ValueError(f"need more than {q + 1} observations for q = {q}")
    u2 = u ** 2
    T = u2.shape[0] - q
    y = u2[q:]
    cols = [np.ones(T)]
    for k in range(1, q + 1):
        cols.append(u2[q - k: -k])
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    ss_res = float(resid @ resid)
    ss_tot = float((y - y.mean()) @ (y - y.mean()))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    lm = T * r2
    f = (r2 / q) / max((1 - r2) / (T - q - 1), 1e-12)
    return {
        "lm_stat":     float(lm),
        "p_value":     float(chi2.sf(lm, df=q)),
        "f_stat":      float(f),
        "f_p_value":   float(f_dist.sf(f, q, T - q - 1)),
        "df":          int(q),
        "n_obs":       int(T),
    }


def ljung_box_squared(u: np.ndarray, lags: int = 10) -> dict:
    """Ljung-Box Q-statistic on squared residuals.

    Q = T(T+2) Σ_{k=1..L} ρ²(u²)_k / (T - k)  ~  χ²(lags) under H0.

    A standard mis-specification check applied *after* fitting a
    conditional-volatility model: if the squared standardised
    residuals still show autocorrelation, the model has not absorbed
    all the ARCH structure.
    """
    u = np.asarray(u, dtype=float).ravel()
    if u.size <= lags + 1:
        raise ValueError(f"need more than {lags + 1} observations")
    u2 = u ** 2
    u2 = u2 - u2.mean()
    T = u2.shape[0]
    denom = float(u2 @ u2)
    if denom <= 0:
        return {"q_stat": 0.0, "p_value": 1.0, "df": int(lags)}
    q_stat = 0.0
    for k in range(1, lags + 1):
        rho_k = float(u2[k:] @ u2[:-k]) / denom
        q_stat += (rho_k ** 2) / (T - k)
    q_stat *= T * (T + 2)
    return {
        "q_stat":  float(q_stat),
        "p_value": float(chi2.sf(q_stat, df=lags)),
        "df":      int(lags),
    }


__all__ = ["arch_lm_test", "ljung_box_squared"]
