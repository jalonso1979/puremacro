"""Cointegration tests + Vector Error Correction Model.

- Engle-Granger (1987) two-step cointegration test.
- Johansen (1988, 1991) trace test for cointegrating rank.
- VECM fit via reduced-rank regression.

Critical values: MacKinnon-Haug-Michelis (1999) response-surface 5%
values for the trace statistic, **unrestricted-constant** case (Johansen's
Case III: a linear trend in the data, an intercept in the cointegrating
relation, no trend in the CE) — which is what :func:`johansen` actually
estimates, since its auxiliary regression carries an intercept.
For pedagogical use; for publication use a full critical-value table
(all significance levels and all five deterministic cases).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .._linalg import inv_xtx


# Johansen trace 5% critical values, keyed by n - r (the number of
# remaining common-stochastic-trend dimensions under H_0: rank <= r).
#
# Source: MacKinnon, J.G., Haug, A.A. and Michelis, L. (1999), "Numerical
# distribution functions of likelihood ratio tests for cointegration",
# Journal of Applied Econometrics 14(5), 563-577 — response-surface
# quantiles, as tabulated in ``statsmodels.tsa.coint_tables`` (verified
# against ``c_sjt(n - r, det_order)[1]``, statsmodels 0.14.6).
#
# TWO cases are kept because they are easy to confuse, and mixing them is
# exactly the bug this table used to have (it carried 3.84 from the
# constant case and 12.32, 24.28, ... from the no-deterministic case, so
# johansen() over-rejected H_0: r = 0 by ~3 points of the trace statistic):
#
#   "const"  — unrestricted constant (Johansen Case III, statsmodels
#              det_order=0). THIS IS WHAT :func:`johansen` ESTIMATES: its
#              auxiliary regression on lagged differences includes an
#              intercept (``row = [1.0]``), so the constant is concentrated
#              out unrestrictedly.
#   "none"   — no deterministic term at all (statsmodels det_order=-1).
#              Kept for reference / future use; NOT what johansen() fits.
_JOHANSEN_TRACE_CV_5PCT_BY_DET: dict[str, dict[int, float]] = {
    "const": {
        1: 3.8415,
        2: 15.4943,
        3: 29.7961,
        4: 47.8545,
        5: 69.8189,
        6: 95.7542,
        7: 125.6185,
        8: 159.5290,
        9: 197.3772,
        10: 239.2468,
        11: 285.1402,
        12: 334.9795,
    },
    "none": {
        1: 4.1296,
        2: 12.3212,
        3: 24.2761,
        4: 40.1749,
        5: 60.0627,
        6: 83.9383,
        7: 111.7797,
        8: 143.6691,
        9: 179.5199,
        10: 219.4051,
        11: 263.2603,
        12: 311.1288,
    },
}

# Default table used by johansen(): the deterministic case it truly fits.
_JOHANSEN_TRACE_CV_5PCT = _JOHANSEN_TRACE_CV_5PCT_BY_DET["const"]


def _adf_residual_test(u: np.ndarray, max_lag: int = 4) -> dict:
    """Quick ADF on a residual series, with constant only.

    For Engle-Granger residuals the asymptotic distribution differs from
    the standard ADF (the residuals are estimated, not observed), so the
    p-value here uses MacKinnon (1996) tau distribution coefficients for
    n=2 cointegrated regressors as a rough approximation.
    """
    u = np.asarray(u, dtype=float).ravel()
    T = len(u)
    if T < max_lag + 5:
        return {"adf_t": np.nan, "adf_p": np.nan}
    du = np.diff(u)
    u_lag = u[:-1]
    # Build regressors: const, u_{t-1}, lagged Δu's
    X_cols = [np.ones(T - 1 - max_lag), u_lag[max_lag:]]
    for k in range(1, max_lag + 1):
        X_cols.append(du[max_lag - k: -k] if k < max_lag else du[:-max_lag])
    X = np.column_stack(X_cols)
    y = du[max_lag:]
    if X.shape[0] != y.shape[0]:
        # Trim to common length
        m = min(X.shape[0], y.shape[0])
        X = X[:m]
        y = y[:m]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(np.sum(resid ** 2))
    df = X.shape[0] - X.shape[1]
    s2 = rss / df
    XtX_inv = inv_xtx(X, name="engle_granger_adf")
    se = np.sqrt(s2 * np.diag(XtX_inv))
    adf_t = beta[1] / se[1]
    # Approximate p-value via Hamilton (1994) tau_2 distribution constants
    # for n=2 case (very rough — ok for pedagogical use)
    # Phillips-Ouliaris (1990) Table 2: tau_2 5% ≈ -3.34 at T=infinity
    # Use a rough z-equivalent via asymptotic Dickey-Fuller distribution
    # parameters (mean -1.61, std 0.94 approx). This is only a heuristic.
    p_value = float(norm.cdf((adf_t - (-1.61)) / 0.94))
    return {"adf_t": float(adf_t), "adf_p": p_value}


def engle_granger(y, x) -> dict:
    """Engle-Granger (1987) two-step cointegration test.

    Step 1: y = α + β x + u  (OLS).
    Step 2: ADF test on the residuals.

    Returns
    -------
    dict with keys: cointegrating_beta, alpha_const, residuals, adf_t, adf_p
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    T = len(y)
    X = np.column_stack([np.ones(T), x])
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha = float(coefs[0])
    beta = float(coefs[1])
    resid = y - X @ coefs
    adf = _adf_residual_test(resid, max_lag=4)
    return {
        "cointegrating_beta": beta,
        "alpha_const": alpha,
        "residuals": resid,
        **adf,
    }


def johansen(Y: pd.DataFrame, p: int = 2) -> dict:
    """Johansen (1988, 1991) trace test for cointegrating rank.

    Algorithm:
      1. Build ΔY_t and Y_{t-1}.
      2. Regress both on lagged ΔY (k = p-1 lags) + constant; collect
         residuals R0 (from ΔY) and R1 (from Y_{t-1}).
      3. Form S_ij = R_i' R_j / T and solve generalized eigenvalue
         problem |λ S_11 − S_01' S_00^{-1} S_01| = 0.
      4. Trace statistic at rank r: −T Σ_{i=r+1}^{n} log(1 − λ_i).
      5. Compare against tabulated 5% critical values; report r̂.

    Deterministic specification: the auxiliary regression in step 2
    includes an **unrestricted constant** (Johansen's Case III, i.e.
    ``det_order=0`` in ``statsmodels.tsa.vector_ar.vecm.coint_johansen``),
    so the critical values used are the MacKinnon-Haug-Michelis (1999)
    5% quantiles for that case: 3.84, 15.49, 29.80, 47.85, 69.82, … for
    ``n − r = 1, 2, 3, 4, 5, …``.

    Returns
    -------
    dict with keys:
        eigenvalues : array of n eigenvalues, descending
        trace_stats : array of n trace statistics (one per H_0: rank ≤ r)
        crit_values : array of 5% critical values
        r_hat       : estimated rank
        alpha       : (n, r̂) loadings
        beta        : (n, r̂) cointegrating vectors
    """
    Y_arr = np.asarray(Y, dtype=float)
    T, n = Y_arr.shape
    if p < 2:
        p = 2
    k = p - 1  # number of lagged differences
    if T <= p + k + 5:
        raise ValueError(f"need T > p + k + 5; got T={T}, p={p}")

    dY = np.diff(Y_arr, axis=0)  # (T-1, n)
    Y_lag = Y_arr[:-1]            # (T-1, n)

    # Auxiliary regressors: lagged differences + const
    aux_rows: list = []
    targets_dY_list: list = []
    targets_Ylag_list: list = []
    for t in range(k, T - 1):
        row = [1.0]
        for j in range(1, k + 1):
            row.extend(dY[t - j])
        aux_rows.append(row)
        targets_dY_list.append(dY[t])
        targets_Ylag_list.append(Y_lag[t])
    Z = np.array(aux_rows)         # (T_eff, 1 + n*k)
    targets_dY: np.ndarray = np.array(targets_dY_list)
    targets_Ylag: np.ndarray = np.array(targets_Ylag_list)
    T_eff = len(Z)

    # Residuals R0 (from ΔY) and R1 (from Y_lag), regressed on Z
    # OLS for each column then residuals
    def _residualize(Y_target):
        beta_aux, *_ = np.linalg.lstsq(Z, Y_target, rcond=None)
        return Y_target - Z @ beta_aux

    R0 = _residualize(targets_dY)
    R1 = _residualize(targets_Ylag)

    S00 = R0.T @ R0 / T_eff
    S01 = R0.T @ R1 / T_eff
    S11 = R1.T @ R1 / T_eff

    # Generalized eigenvalue problem: solve |λ S11 − S01' S00^{-1} S01| = 0
    # Equivalent to standard eigvals of S11^{-1/2} S01' S00^{-1} S01 S11^{-1/2}
    # We do it via scipy or just direct inversion.
    A = np.linalg.solve(S11, S01.T @ np.linalg.solve(S00, S01))
    eigvals, eigvecs = np.linalg.eig(A)
    # Sort descending by real part
    order = np.argsort(-eigvals.real)
    eigvals = eigvals[order].real
    eigvecs = eigvecs[:, order].real
    # Clip to [0, 1) to avoid numerical issues in log(1 - λ)
    eigvals = np.clip(eigvals, 0.0, 0.999999)

    # Trace statistic for each rank hypothesis
    trace_stats = np.array([
        -T_eff * np.sum(np.log(1.0 - eigvals[r:]))
        for r in range(n)
    ])

    # Critical values (5%, unrestricted-constant case = what we estimate,
    # since Z carries an intercept above). MacKinnon-Haug-Michelis (1999).
    cv_table = _JOHANSEN_TRACE_CV_5PCT_BY_DET["const"]
    crit = np.array([
        cv_table.get(n - r, 9999.0)
        for r in range(n)
    ])

    # r̂: smallest r where trace_stats[r] < crit[r]
    r_hat = n  # default — full rank if all rejected
    for r in range(n):
        if trace_stats[r] < crit[r]:
            r_hat = r
            break

    # Cointegrating vectors β are the top r̂ eigenvectors
    if r_hat == 0:
        beta_hat = np.zeros((n, 0))
        alpha_hat = np.zeros((n, 0))
    else:
        beta_hat = eigvecs[:, :r_hat]
        # α = S01 β (β' S11 β)^-1
        Sbb = beta_hat.T @ S11 @ beta_hat
        alpha_hat = S01 @ beta_hat @ np.linalg.inv(Sbb)

    return {
        "eigenvalues": eigvals,
        "trace_stats": trace_stats,
        "crit_values": crit,
        "r_hat": int(r_hat),
        "alpha": alpha_hat,
        "beta": beta_hat,
    }


def fit_vecm(Y: pd.DataFrame, p: int = 2, r: int = 1) -> dict:
    """Fit a VECM with cointegrating rank r. Returns α, β, Π = α β'."""
    out = johansen(Y, p=p)
    # Override r_hat with user-supplied r
    eigvals = out["eigenvalues"]
    Y_arr = np.asarray(Y, dtype=float)
    T, n = Y_arr.shape
    k = max(1, p - 1)
    dY = np.diff(Y_arr, axis=0)
    Y_lag = Y_arr[:-1]

    aux_rows_v: list = []
    targets_dY_list_v: list = []
    targets_Ylag_list_v: list = []
    for t in range(k, T - 1):
        row = [1.0]
        for j in range(1, k + 1):
            row.extend(dY[t - j])
        aux_rows_v.append(row)
        targets_dY_list_v.append(dY[t])
        targets_Ylag_list_v.append(Y_lag[t])
    Z = np.array(aux_rows_v)
    targets_dY: np.ndarray = np.array(targets_dY_list_v)
    targets_Ylag: np.ndarray = np.array(targets_Ylag_list_v)
    T_eff = len(Z)

    def _residualize(Y_target):
        beta_aux, *_ = np.linalg.lstsq(Z, Y_target, rcond=None)
        return Y_target - Z @ beta_aux

    R0 = _residualize(targets_dY)
    R1 = _residualize(targets_Ylag)
    S00 = R0.T @ R0 / T_eff
    S01 = R0.T @ R1 / T_eff
    S11 = R1.T @ R1 / T_eff

    A = np.linalg.solve(S11, S01.T @ np.linalg.solve(S00, S01))
    eigvals, eigvecs = np.linalg.eig(A)
    order = np.argsort(-eigvals.real)
    eigvecs = eigvecs[:, order].real

    beta = eigvecs[:, :r]
    Sbb = beta.T @ S11 @ beta
    alpha = S01 @ beta @ np.linalg.inv(Sbb)
    Pi = alpha @ beta.T

    # Residuals: ΔY_t − Π Y_{t-1} − Γ Z_t (project ΔY on [Y_lag, Z])
    full_X = np.column_stack([targets_Ylag, Z])
    full_coefs, *_ = np.linalg.lstsq(full_X, targets_dY, rcond=None)
    residuals = targets_dY - full_X @ full_coefs

    return {
        "alpha": alpha,
        "beta": beta,
        "Pi": Pi,
        "residuals": residuals,
    }


__all__ = ["engle_granger", "johansen", "fit_vecm"]
