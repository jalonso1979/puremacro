"""Swamy (1970) slope homogeneity test for panel VARs / panel LPs.

Reference: Swamy, P. A. V. B. (1970). Efficient inference in a random
coefficient regression model. Econometrica, 38(2), 311-323.
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def swamy_test(
    beta_hat: np.ndarray,
    sigma_hat: np.ndarray,
) -> tuple[float, float, int]:
    """Swamy (1970) slope homogeneity test for panel VARs.

    Parameters
    ----------
    beta_hat : np.ndarray, shape (N, K)
        Per-country OLS coefficient vectors (N countries, K regressors).
    sigma_hat : np.ndarray, shape (N, K, K)
        Per-country OLS covariance matrices.

    Returns
    -------
    (S_stat, p_value, df) : tuple[float, float, int]
        S_stat   : Swamy χ² statistic.
        p_value  : P(χ²(df) ≥ S_stat).
        df       : K * (N - 1) degrees of freedom.
    """
    N, K = beta_hat.shape
    beta_mg = beta_hat.mean(axis=0)  # (K,)
    S = 0.0
    for i in range(N):
        diff = beta_hat[i] - beta_mg  # (K,)
        S_i = diff @ np.linalg.solve(sigma_hat[i], diff)
        S += S_i
    df = K * (N - 1)
    p_val = float(1.0 - stats.chi2.cdf(S, df))
    return float(S), p_val, df
