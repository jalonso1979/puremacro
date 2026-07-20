"""Diebold-Mariano (1995) + Giacomini-White (2006) forecast tests.

DM tests unconditional equal predictive ability of two forecasts.
GW tests conditional predictive ability — i.e., whether the loss
differential is forecastable from a vector of conditioning variables.

Small-sample correction (Harvey-Leybourne-Newbold 1997) applied to DM.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm, t as t_dist


def _loss_diff(e1: np.ndarray, e2: np.ndarray, loss: str, power: float) -> np.ndarray:
    """Per-period loss differential L1 - L2."""
    e1 = np.asarray(e1, dtype=float).ravel()
    e2 = np.asarray(e2, dtype=float).ravel()
    if loss == "mse":
        return e1 ** 2 - e2 ** 2
    if loss == "mae":
        return np.abs(e1) - np.abs(e2)
    if loss == "lp":
        return np.abs(e1) ** power - np.abs(e2) ** power
    raise ValueError(f"unknown loss {loss!r}")


def diebold_mariano(e1, e2, h: int = 1, loss: str = "mse", power: float = 2.0,
                    small_sample: bool = True) -> dict:
    """Diebold-Mariano (1995) test of H_0: E[L(e1) - L(e2)] = 0.

    Long-run variance via Newey-West with bandwidth h-1 (the natural
    choice when forecasts are h-step-ahead).

    With small_sample=True, applies the Harvey-Leybourne-Newbold (1997)
    correction: t* = sqrt((T + 1 - 2h + h(h-1)/T) / T) * t.
    """
    d = _loss_diff(e1, e2, loss, power)
    T = len(d)
    d_bar = float(np.mean(d))
    # Long-run variance via Bartlett-NW
    g0 = float(np.sum((d - d_bar) ** 2) / T)
    s2 = g0
    L = max(0, h - 1)
    for ell in range(1, L + 1):
        if ell >= T:
            break
        cov = float(np.sum((d[ell:] - d_bar) * (d[:-ell] - d_bar)) / T)
        s2 += 2.0 * (1.0 - ell / (L + 1.0)) * cov
    if s2 <= 0:
        return {"stat": np.nan, "p_value": np.nan,
                "n_obs": int(T), "lag_used": int(L)}
    stat = d_bar / np.sqrt(s2 / T)
    if small_sample and T > 0:
        correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
        stat = correction * stat
        # Use t-distribution with T-1 df under HLN correction
        p_value = 2.0 * (1.0 - t_dist.cdf(abs(stat), df=T - 1))
    else:
        p_value = 2.0 * (1.0 - norm.cdf(abs(stat)))
    return {
        "stat": float(stat),
        "p_value": float(p_value),
        "n_obs": int(T),
        "lag_used": int(L),
    }


def giacomini_white(e1, e2, h: int = 1, conditioning_vars=None,
                    loss: str = "mse", power: float = 2.0) -> dict:
    """Giacomini-White (2006) conditional predictive ability test.

    H_0: E[L_diff_t * h_t] = 0 for instrument vector h_t. Test stat is
    a Wald chi-squared with df = q (the number of conditioning variables
    plus one for the constant).

    If `conditioning_vars` is None or has 0 columns, falls back to a
    constant — effectively the unconditional DM-style test using a
    chi-square(1) reference distribution.
    """
    d = _loss_diff(e1, e2, loss, power)
    T = len(d)
    if conditioning_vars is None or (
        hasattr(conditioning_vars, "shape") and conditioning_vars.shape[-1] == 0
    ):
        H = np.ones((T, 1))
    else:
        H = np.asarray(conditioning_vars, dtype=float)
        if H.ndim == 1:
            H = H.reshape(-1, 1)
        # Stack a constant
        H = np.column_stack([np.ones(T), H])
    # Sample moment Z_T = mean(H * d)
    moments = H * d[:, None]
    Z = moments.mean(axis=0)
    # Long-run variance of moments via Bartlett-NW with lags h-1
    L = max(0, h - 1)
    Sigma = (moments.T @ moments) / T
    for ell in range(1, L + 1):
        if ell >= T:
            break
        Gamma = (moments[ell:].T @ moments[:-ell]) / T
        Sigma = Sigma + (1.0 - ell / (L + 1.0)) * (Gamma + Gamma.T)
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        return {"stat": np.nan, "p_value": np.nan, "df": Sigma.shape[0]}
    stat = T * (Z @ Sigma_inv @ Z)
    df = Sigma.shape[0]
    p_value = 1.0 - chi2.cdf(stat, df=df)
    return {
        "stat": float(stat),
        "p_value": float(p_value),
        "df": int(df),
    }


__all__ = ["diebold_mariano", "giacomini_white"]
