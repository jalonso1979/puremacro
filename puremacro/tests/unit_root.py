"""Unit-root and stationarity tests.

- ADF (Said-Dickey 1984): augmented Dickey-Fuller; auto-lag via SIC.
- KPSS (1992): tests stationarity null.
- PP (Phillips-Perron 1988): non-parametric correction to DF.
- Zivot-Andrews (1992): ADF with single endogenous structural break.

Critical values use small simplified tables. For publication use, swap
in MacKinnon (1996) or LLSW tables. P-values are interpolated linearly
between tabulated points.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .._linalg import inv_xtx


# ADF / PP critical values — MacKinnon (1996), constant only, T=infinity
# (1%, 5%, 10%). These are the asymptotic intercepts; ``_mackinnon_cv``
# applies the (1/T, 1/T^2) finite-sample correction from MacKinnon's
# response surface.
_DF_CV = {
    "n": (-2.567, -1.941, -1.616),  # no constant
    "c": (-3.430, -2.861, -2.567),  # constant
    "ct": (-3.958, -3.410, -3.127),  # constant + trend
}

# MacKinnon (1996), J. Appl. Econometrics 11, Table 2: response-surface
# coefficients for the ADF/PP critical values at the (1%, 5%, 10%)
# levels. CV(T) = beta_inf + beta_1 / T + beta_2 / T^2.
# Keys: regression -> ((b_inf_1pct, b1_1pct, b2_1pct), (5%), (10%)).
_DF_CV_RS = {
    "n":  ((-2.5658, -1.960,  -10.04), (-1.9393, -0.398,   0.0),   (-1.6156, -0.181,   0.0)),
    "c":  ((-3.4336, -5.999,  -29.25), (-2.8621, -2.738,  -8.36),  (-2.5671, -1.438,  -4.48)),
    "ct": ((-3.9638, -8.353,  -47.44), (-3.4126, -4.039, -17.83),  (-3.1279, -2.418,  -7.58)),
}


def _mackinnon_cv(T: int, regression: str = "c") -> tuple[float, float, float]:
    """Return MacKinnon (1996) finite-sample CVs (1%, 5%, 10%) for given T.

    Falls back to the asymptotic table in ``_DF_CV`` if the regression
    spec is not in ``_DF_CV_RS``.
    """
    rs = _DF_CV_RS.get(regression)
    if rs is None:
        return _DF_CV.get(regression, _DF_CV["c"])
    out: list[float] = []
    for (b_inf, b1, b2) in rs:
        out.append(b_inf + b1 / T + b2 / (T ** 2))
    return (out[0], out[1], out[2])

# KPSS critical values — Kwiatkowski et al. (1992), Table 1
_KPSS_CV = {
    "c": (0.739, 0.463, 0.347),   # 1%, 5%, 10%
    "ct": (0.216, 0.146, 0.119),
}


def _interpolate_pvalue(stat: float, cv_table: tuple[float, float, float],
                       reject_low: bool = True) -> float:
    """Map a test statistic to an interpolated p-value using a 3-point CV table.

    `cv_table` is (1%, 5%, 10%).
    `reject_low=True`: small / negative stat → reject (e.g. ADF, PP).
    `reject_low=False`: large stat → reject (e.g. KPSS).
    """
    cv1, cv5, cv10 = cv_table
    if reject_low:
        # Stats below cv1 → p < 0.01; between cv1 and cv5 → 0.01-0.05; etc.
        if stat <= cv1:
            return 0.005
        if stat <= cv5:
            return 0.01 + (stat - cv1) / (cv5 - cv1) * 0.04
        if stat <= cv10:
            return 0.05 + (stat - cv5) / (cv10 - cv5) * 0.05
        return min(0.5, 0.10 + (stat - cv10) / abs(cv10) * 0.40)
    else:
        if stat >= cv1:
            return 0.005
        if stat >= cv5:
            return 0.01 + (cv1 - stat) / (cv1 - cv5) * 0.04
        if stat >= cv10:
            return 0.05 + (cv5 - stat) / (cv5 - cv10) * 0.05
        return min(0.5, 0.10 + (cv10 - stat) / abs(cv10) * 0.40)


def _select_lag_sic(y: np.ndarray, max_lag: int, regression: str) -> int:
    """Auto-select ADF lag order by minimum SIC."""
    T = len(y)
    dy = np.diff(y)
    best_lag, best_sic = 0, np.inf
    for k in range(0, max_lag + 1):
        if T - k - 2 < 5:
            break
        # Build ADF regression: Δy_t = α + ρ y_{t-1} + Σ_{j=1}^{k} γ_j Δy_{t-j} + ε
        n = T - k - 1
        rhs = [y[k:T-1]]  # y_{t-1}
        if regression in ("c", "ct"):
            rhs = [np.ones(n)] + rhs
        if regression == "ct":
            rhs.append(np.arange(n))
        for j in range(1, k + 1):
            rhs.append(dy[k-j:T-1-j])
        X = np.column_stack(rhs)
        ydep = dy[k:]
        if X.shape[0] != ydep.shape[0]:
            m = min(X.shape[0], ydep.shape[0])
            X = X[:m]; ydep = ydep[:m]
        try:
            beta, *_ = np.linalg.lstsq(X, ydep, rcond=None)
            resid = ydep - X @ beta
            rss = float(np.sum(resid ** 2))
            sic = np.log(rss / len(ydep)) + np.log(len(ydep)) * X.shape[1] / len(ydep)
            if sic < best_sic:
                best_sic = sic; best_lag = k
        except np.linalg.LinAlgError:
            continue
    return best_lag


def adf_test(y, lags: str | int = "auto", regression: str = "c", max_lag: int = 8) -> dict:
    """Augmented Dickey-Fuller test.

    H_0: y has a unit root.
    Test stat: t-statistic on the lagged level coefficient ρ in
        Δy_t = α + ρ y_{t-1} + Σ γ_j Δy_{t-j} + ε.
    """
    y = np.asarray(y, dtype=float).ravel()
    T = len(y)
    if isinstance(lags, str) and lags == "auto":
        k = _select_lag_sic(y, max_lag=max_lag, regression=regression)
    else:
        k = int(lags)
    dy = np.diff(y)
    n = T - k - 1
    rhs = [y[k:T-1]]
    if regression in ("c", "ct"):
        rhs = [np.ones(n)] + rhs
    if regression == "ct":
        rhs.append(np.arange(n).astype(float))
    for j in range(1, k + 1):
        rhs.append(dy[k-j:T-1-j])
    X = np.column_stack(rhs)
    ydep = dy[k:]
    m = min(X.shape[0], ydep.shape[0])
    X = X[:m]; ydep = ydep[:m]
    beta, *_ = np.linalg.lstsq(X, ydep, rcond=None)
    resid = ydep - X @ beta
    sigma2 = float(np.sum(resid ** 2)) / max(1, len(ydep) - X.shape[1])
    XtX_inv = inv_xtx(X, name="adf")
    se = np.sqrt(sigma2 * np.diag(XtX_inv))
    # Index of lagged level coefficient
    rho_idx = 1 if regression in ("c", "ct") else 0
    if regression == "ct":
        rho_idx = 1  # const at 0, lag y at 1, trend at 2
    t_stat = beta[rho_idx] / se[rho_idx]
    cv = _mackinnon_cv(T, regression=regression)
    p_value = _interpolate_pvalue(t_stat, cv, reject_low=True)
    return {
        "stat": float(t_stat),
        "p_value": float(p_value),
        "lag_used": int(k),
        "crit_values": {"1%": cv[0], "5%": cv[1], "10%": cv[2]},
        "decision": "reject" if t_stat < cv[1] else "accept",
        "regression": regression,
    }


def kpss_test(y, regression: str = "c", lags: int | None = None) -> dict:
    """KPSS (1992) stationarity test.

    H_0: y is (level- or trend-) stationary. Reject → unit root present.
    """
    y = np.asarray(y, dtype=float).ravel()
    T = len(y)
    if regression == "c":
        e = y - np.mean(y)
    elif regression == "ct":
        X = np.column_stack([np.ones(T), np.arange(T)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ beta
    else:
        raise ValueError(f"regression must be 'c' or 'ct'; got {regression}")
    if lags is None:
        lags = int(np.ceil(12 * (T / 100) ** 0.25))
    # Long-run variance via Bartlett kernel
    g0 = np.sum(e ** 2) / T
    s2 = g0
    for ell in range(1, lags + 1):
        if ell >= T:
            break
        cov = np.sum(e[ell:] * e[:-ell]) / T
        s2 += 2.0 * (1.0 - ell / (lags + 1.0)) * cov
    cum_e = np.cumsum(e)
    eta = np.sum(cum_e ** 2) / (T ** 2 * max(s2, 1e-12))
    cv = _KPSS_CV.get(regression, _KPSS_CV["c"])
    p_value = _interpolate_pvalue(eta, cv, reject_low=False)
    return {
        "stat": float(eta),
        "p_value": float(p_value),
        "lag_used": int(lags),
        "crit_values": {"1%": cv[0], "5%": cv[1], "10%": cv[2]},
        "decision": "reject" if eta > cv[1] else "accept",
        "regression": regression,
    }


def pp_test(y, regression: str = "c", lags: int | None = None) -> dict:
    """Phillips-Perron (1988) non-parametric unit-root test.

    Same null as ADF but corrects the t-statistic non-parametrically
    rather than including augmenting lags.
    """
    y = np.asarray(y, dtype=float).ravel()
    T = len(y)
    dy = np.diff(y)
    n = T - 1
    rhs = [y[:-1]]
    if regression in ("c", "ct"):
        rhs = [np.ones(n)] + rhs
    if regression == "ct":
        rhs.append(np.arange(n).astype(float))
    X = np.column_stack(rhs)
    beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
    resid = dy - X @ beta
    sigma2 = float(np.sum(resid ** 2)) / max(1, n - X.shape[1])
    XtX_inv = inv_xtx(X, name="phillips_perron")
    se = np.sqrt(sigma2 * np.diag(XtX_inv))
    rho_idx = 1 if regression in ("c", "ct") else 0
    if regression == "ct":
        rho_idx = 1
    t_dickey = beta[rho_idx] / se[rho_idx]
    if lags is None:
        lags = int(np.ceil(4 * (T / 100) ** (1 / 4)))
    # PP correction: estimate long-run variance s²_L of residuals
    g0 = np.sum(resid ** 2) / n
    s2_L = g0
    for ell in range(1, lags + 1):
        if ell >= n: break
        cov = np.sum(resid[ell:] * resid[:-ell]) / n
        s2_L += 2.0 * (1.0 - ell / (lags + 1.0)) * cov
    s2_L = max(s2_L, 1e-12)
    # Phillips-Perron Z_t statistic
    t_pp = np.sqrt(g0 / s2_L) * t_dickey - 0.5 * (s2_L - g0) * np.sqrt(n) * np.sqrt(np.diag(XtX_inv)[rho_idx]) / s2_L
    cv = _mackinnon_cv(T, regression=regression)
    p_value = _interpolate_pvalue(t_pp, cv, reject_low=True)
    return {
        "stat": float(t_pp),
        "p_value": float(p_value),
        "lag_used": int(lags),
        "crit_values": {"1%": cv[0], "5%": cv[1], "10%": cv[2]},
        "decision": "reject" if t_pp < cv[1] else "accept",
        "regression": regression,
    }


def zivot_andrews_test(y, regression: str = "c", trim: float = 0.15,
                        max_lag: int = 4) -> dict:
    """Zivot-Andrews (1992) unit-root test with one endogenous break.

    Searches for the break date that minimises the t-statistic on the
    lagged level coefficient ρ in
        Δy_t = α + δ DU_t(τ) + ρ y_{t-1} + Σ γ_j Δy_{t-j} + ε
    where DU_t(τ) = 1{t > τ T} (level shift). The test statistic is the
    minimum t. Critical values (5%) for the constant case ≈ -4.80
    (Zivot-Andrews 1992 Table 2 Model A, T=∞).
    """
    y = np.asarray(y, dtype=float).ravel()
    T = len(y)
    start = int(np.floor(trim * T)); end = int(np.floor((1 - trim) * T))
    best_t = np.inf; best_break = start
    dy = np.diff(y)
    for tau in range(start, end + 1):
        DU = (np.arange(T) > tau).astype(float)
        n = T - max_lag - 1
        rhs = [np.ones(n), DU[max_lag + 1:], y[max_lag:T-1]]
        if regression == "ct":
            rhs.append(np.arange(n).astype(float))
        for j in range(1, max_lag + 1):
            rhs.append(dy[max_lag - j: T - 1 - j])
        X = np.column_stack(rhs)
        ydep = dy[max_lag:]
        m = min(X.shape[0], ydep.shape[0])
        X = X[:m]; ydep = ydep[:m]
        try:
            beta, *_ = np.linalg.lstsq(X, ydep, rcond=None)
            resid = ydep - X @ beta
            sigma2 = float(np.sum(resid ** 2)) / max(1, len(ydep) - X.shape[1])
            XtX_inv = inv_xtx(X, name="zivot_andrews")
            se = np.sqrt(sigma2 * np.diag(XtX_inv))
            # Index 2 for lagged y (after const, DU)
            t_rho = beta[2] / se[2]
            if t_rho < best_t:
                best_t = t_rho; best_break = tau
        except np.linalg.LinAlgError:
            continue
    crit_5pct = -4.80  # ZA 1992 Table 2 Model A constant case
    p_value = _interpolate_pvalue(best_t, (-5.34, crit_5pct, -4.58), reject_low=True)
    return {
        "stat": float(best_t),
        "p_value": float(p_value),
        "break_date": int(best_break),
        "lag_used": int(max_lag),
        "crit_values": {"1%": -5.34, "5%": crit_5pct, "10%": -4.58},
        "decision": "reject" if best_t < crit_5pct else "accept",
        "regression": regression,
    }


__all__ = ["adf_test", "kpss_test", "pp_test", "zivot_andrews_test"]
