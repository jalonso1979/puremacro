"""Over-identification: Hansen J test for 2SLS / IV-GMM, plus a small
Stock-Yogo critical-value lookup for weak-instrument decisions.

Hansen J:
    Run 2SLS to get residuals u_hat. Regress u_hat on the instrument
    matrix Z (and any included exogenous controls). The Hansen J
    statistic is T * R^2; under H_0 of valid instruments and correct
    moment conditions, J ~ chi^2(l - k) where l = #instruments and
    k = #endogenous regressors.

Stock-Yogo critical values are tabulated for the maximal IV bias
relative to OLS (Stock-Yogo 2005, Tables 5.2 / 5.3). We expose a
small lookup for the single-endogenous-regressor case at the most
common "max bias" thresholds.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def hansen_j(
    y: np.ndarray,
    X_endog: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray | None = None,
) -> dict:
    """Hansen J statistic for 2SLS over-identification.

    Parameters
    ----------
    y       : (T,) outcome.
    X_endog : (T, k) endogenous regressors.
    Z       : (T, l) instruments (l > k for the test to be meaningful).
    W       : (T, m) included exogenous controls (incl. constant if any).

    Returns
    -------
    dict with stat (J), p_value, df (= l - k), n_obs.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(X_endog if X_endog.ndim > 1 else X_endog[:, None])
    Z = np.atleast_2d(Z if Z.ndim > 1 else Z[:, None])
    T = len(y)
    k = X.shape[1]
    l = Z.shape[1]

    if W is None:
        W = np.ones((T, 1))
    else:
        W = np.atleast_2d(W if W.ndim > 1 else W[:, None])
        # ensure a constant is present
        if not np.allclose(W[:, 0], 1.0):
            W = np.column_stack([np.ones(T), W])

    full_Z = np.column_stack([W, Z])

    # First stage: project X on (W, Z)
    PZ = full_Z @ np.linalg.solve(full_Z.T @ full_Z, full_Z.T)
    X_hat = PZ @ X

    # 2SLS: combine X_hat with W as second-stage regressors
    XW_hat = np.column_stack([X_hat, W])
    XW = np.column_stack([X, W])
    beta = np.linalg.solve(XW_hat.T @ XW, XW_hat.T @ y)
    u_hat = y - XW @ beta

    # Auxiliary regression of u_hat on full_Z; J = T R^2
    proj = full_Z @ np.linalg.solve(full_Z.T @ full_Z, full_Z.T @ u_hat)
    rss = float(np.sum((u_hat - proj) ** 2))
    tss = float(np.sum((u_hat - u_hat.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    J = T * r2
    df = l - k
    p_value = 1.0 - chi2.cdf(J, df=df) if df > 0 else np.nan
    return {
        "stat": float(J),
        "p_value": float(p_value),
        "df": int(df),
        "n_obs": int(T),
    }


# Stock-Yogo (2005) Table 5.2 — single endogenous regressor (k=1),
# 2SLS critical values for "max bias relative to OLS" thresholds.
# Keys: (n_instruments, max_bias_pct) -> first-stage F critical value.
_STOCK_YOGO_BIAS_K1 = {
    # (l, threshold%) -> CV
    (3,  5):  13.91, (3,  10):  9.08, (3,  20): 6.46, (3,  30): 5.39,
    (4,  5):  16.85, (4,  10): 10.27, (4,  20): 6.71, (4,  30): 5.34,
    (5,  5):  18.37, (5,  10): 10.83, (5,  20): 6.77, (5,  30): 5.25,
    (6,  5):  19.28, (6,  10): 11.12, (6,  20): 6.76, (6,  30): 5.15,
    (7,  5):  19.86, (7,  10): 11.29, (7,  20): 6.73, (7,  30): 5.07,
    (8,  5):  20.25, (8,  10): 11.39, (8,  20): 6.69, (8,  30): 4.99,
}


def stock_yogo_cv(n_instruments: int, max_bias_pct: int = 10) -> float | None:
    """Look up the Stock-Yogo critical value for first-stage F.

    Compare your first-stage F-stat against this CV; reject "weak
    instruments" if F exceeds it.

    Parameters
    ----------
    n_instruments : int   — number of excluded instruments l.
    max_bias_pct  : int   — maximal IV bias relative to OLS, in % (5/10/20/30).

    Returns
    -------
    Critical value (float) or None if (l, max_bias_pct) is not tabulated.
    The classic Staiger-Stock rule of thumb of "F > 10" corresponds to
    n_instruments=3, max_bias_pct=10 in this table.
    """
    return _STOCK_YOGO_BIAS_K1.get((int(n_instruments), int(max_bias_pct)))


__all__ = ["hansen_j", "stock_yogo_cv"]
