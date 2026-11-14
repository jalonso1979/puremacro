"""Pyodide-clean core unit-root and stationarity test primitives.

Contains the ADF and KPSS test routines plus MacKinnon (1996) critical-value
approximations using only NumPy and internal linear-algebra helpers so both
:mod:`puremacro.data` (shippable under Pyodide) and
:mod:`puremacro.tests.unit_root` share a single implementation.
"""
from __future__ import annotations

import numpy as np

from ._linalg import inv_xtx

_NEAR_EXACT_VARIANCE_TOL = 1e-24

# ADF / PP critical values — MacKinnon (1996), asymptotic intercepts (1%, 5%, 10%).
_DF_CV: dict[str, tuple[float, float, float]] = {
    "n": (-2.567, -1.941, -1.616),
    "c": (-3.430, -2.861, -2.567),
    "ct": (-3.958, -3.410, -3.127),
}

# MacKinnon (1996), J. Appl. Econometrics 11, Table 2 response-surface coefficients:
# CV(T) = beta_inf + beta_1 / T + beta_2 / T^2.
_DF_CV_RS: dict[str, tuple[tuple[float, float, float], ...]] = {
    "n": ((-2.5658, -1.960, -10.04), (-1.9393, -0.398, 0.0), (-1.6156, -0.181, 0.0)),
    "c": ((-3.4336, -5.999, -29.25), (-2.8621, -2.738, -8.36), (-2.5671, -1.438, -4.48)),
    "ct": ((-3.9638, -8.353, -47.44), (-3.4126, -4.039, -17.83), (-3.1279, -2.418, -7.58)),
}

# KPSS critical values — Kwiatkowski et al. (1992), Table 1 (1%, 5%, 10%).
_KPSS_CV: dict[str, tuple[float, float, float]] = {
    "c": (0.739, 0.463, 0.347),
    "ct": (0.216, 0.146, 0.119),
}


def _mackinnon_cv(T: int, regression: str = "c") -> tuple[float, float, float]:
    """Return MacKinnon (1996) finite-sample CVs (1%, 5%, 10%) for given ``T``."""
    rs = _DF_CV_RS.get(regression)
    if rs is None:
        return _DF_CV.get(regression, _DF_CV["c"])
    out = [b_inf + b1 / T + b2 / (T ** 2) for b_inf, b1, b2 in rs]
    return (out[0], out[1], out[2])


def _interpolate_pvalue(
    stat: float,
    cv_table: tuple[float, float, float],
    reject_low: bool = True,
) -> float:
    """Interpolate a p-value from a 3-point ``(1%, 5%, 10%)`` critical-value tuple."""
    if not np.isfinite(stat):
        return float("nan")
    cv1, cv5, cv10 = cv_table
    if reject_low:
        if stat <= cv1:
            return 0.005
        if stat <= cv5:
            return 0.01 + (stat - cv1) / (cv5 - cv1) * 0.04
        if stat <= cv10:
            return 0.05 + (stat - cv5) / (cv10 - cv5) * 0.05
        return float(min(0.5, 0.10 + (stat - cv10) / abs(cv10) * 0.40))
    if stat >= cv1:
        return 0.005
    if stat >= cv5:
        return 0.01 + (cv1 - stat) / (cv1 - cv5) * 0.04
    if stat >= cv10:
        return 0.05 + (cv5 - stat) / (cv5 - cv10) * 0.05
    return float(min(0.5, 0.10 + (cv10 - stat) / abs(cv10) * 0.40))


def _build_adf_design(
    y: np.ndarray,
    k: int,
    regression: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(X, ydep)`` for an ADF(k) regression."""
    T = len(y)
    dy = np.diff(y)
    n = T - k - 1
    rhs: list[np.ndarray] = [y[k:T - 1]]
    if regression in ("c", "ct"):
        rhs = [np.ones(n)] + rhs
    if regression == "ct":
        rhs.append(np.arange(n, dtype=float))
    for j in range(1, k + 1):
        rhs.append(dy[k - j:T - 1 - j])
    X = np.column_stack(rhs)
    ydep = dy[k:]
    m = min(X.shape[0], ydep.shape[0])
    return X[:m], ydep[:m]


def _select_lag_sic(y: np.ndarray, max_lag: int, regression: str) -> int:
    """Auto-select ADF lag order by minimum Schwarz Information Criterion (SIC)."""
    T = len(y)
    best_lag, best_sic = 0, np.inf
    for k in range(0, max_lag + 1):
        if T - k - 2 < 5:
            break
        X, ydep = _build_adf_design(y, k, regression)
        dof = max(1, len(ydep) - X.shape[1])
        try:
            beta, *_ = np.linalg.lstsq(X, ydep, rcond=None)
            resid = ydep - X @ beta
            rss = float(np.sum(resid ** 2))
            if not np.isfinite(rss) or rss / dof <= _NEAR_EXACT_VARIANCE_TOL:
                continue
            sic = float(np.log(rss / len(ydep)) + np.log(len(ydep)) * X.shape[1] / len(ydep))
            if sic < best_sic:
                best_sic = sic
                best_lag = k
        except np.linalg.LinAlgError:
            continue
    return best_lag


def adf_test(
    y: np.ndarray,
    lags: str | int = "auto",
    regression: str = "c",
    max_lag: int = 8,
) -> dict:
    """Augmented Dickey-Fuller unit-root test."""
    y_arr = np.asarray(y, dtype=float).ravel()
    T = len(y_arr)
    if T < 6:
        raise ValueError(f"adf_test requires at least 6 observations, got {T}")
    if regression not in ("n", "c", "ct"):
        raise ValueError(f"regression must be 'n', 'c', or 'ct'; got {regression!r}")
    if isinstance(lags, str) and lags == "auto":
        k = _select_lag_sic(y_arr, max_lag=max_lag, regression=regression)
    else:
        k = int(lags)
    X, ydep = _build_adf_design(y_arr, k, regression)
    beta, *_ = np.linalg.lstsq(X, ydep, rcond=None)
    resid = ydep - X @ beta
    dof = max(1, len(ydep) - X.shape[1])
    rss = float(np.sum(resid ** 2))
    cv = _mackinnon_cv(T, regression=regression)
    if not np.isfinite(rss) or rss / dof <= _NEAR_EXACT_VARIANCE_TOL:
        return {
            "stat": float("nan"),
            "p_value": float("nan"),
            "lag_used": int(k),
            "crit_values": {"1%": cv[0], "5%": cv[1], "10%": cv[2]},
            "decision": "accept",
            "regression": regression,
        }
    sigma2 = rss / dof
    XtX_inv = inv_xtx(X, name="adf")
    se = np.sqrt(sigma2 * np.diag(XtX_inv))
    rho_idx = 1 if regression in ("c", "ct") else 0
    t_stat = float(beta[rho_idx] / se[rho_idx]) if se[rho_idx] > 0.0 else float("nan")
    p_value = _interpolate_pvalue(t_stat, cv, reject_low=True)
    return {
        "stat": float(t_stat),
        "p_value": float(p_value),
        "lag_used": int(k),
        "crit_values": {"1%": cv[0], "5%": cv[1], "10%": cv[2]},
        "decision": "reject" if np.isfinite(t_stat) and t_stat < cv[1] else "accept",
        "regression": regression,
    }


def kpss_test(
    y: np.ndarray,
    regression: str = "c",
    lags: int | None = None,
) -> dict:
    """KPSS (1992) stationarity test (null: level- or trend-stationary)."""
    y_arr = np.asarray(y, dtype=float).ravel()
    T = len(y_arr)
    if T < 4:
        raise ValueError(f"kpss_test requires at least 4 observations, got {T}")
    if regression == "c":
        e = y_arr - np.mean(y_arr)
    elif regression == "ct":
        X = np.column_stack([np.ones(T), np.arange(T, dtype=float)])
        beta, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
        e = y_arr - X @ beta
    else:
        raise ValueError(f"regression must be 'c' or 'ct'; got {regression!r}")
    if lags is None:
        lags = int(np.ceil(12 * (T / 100) ** 0.25))
    g0 = float(np.sum(e ** 2) / T)
    cv = _KPSS_CV.get(regression, _KPSS_CV["c"])
    if g0 <= _NEAR_EXACT_VARIANCE_TOL:
        return {
            "stat": float("nan"),
            "p_value": float("nan"),
            "lag_used": int(lags),
            "crit_values": {"1%": cv[0], "5%": cv[1], "10%": cv[2]},
            "decision": "accept",
            "regression": regression,
        }
    s2 = g0
    for ell in range(1, lags + 1):
        if ell >= T:
            break
        cov = float(np.sum(e[ell:] * e[:-ell]) / T)
        s2 += 2.0 * (1.0 - ell / (lags + 1.0)) * cov
    cum_e = np.cumsum(e)
    eta = float(np.sum(cum_e ** 2) / (T ** 2 * max(s2, 1e-12)))
    p_value = _interpolate_pvalue(eta, cv, reject_low=False)
    return {
        "stat": float(eta),
        "p_value": float(p_value),
        "lag_used": int(lags),
        "crit_values": {"1%": cv[0], "5%": cv[1], "10%": cv[2]},
        "decision": "reject" if eta > cv[1] else "accept",
        "regression": regression,
    }


__all__ = [
    "_DF_CV",
    "_DF_CV_RS",
    "_KPSS_CV",
    "_interpolate_pvalue",
    "_mackinnon_cv",
    "_select_lag_sic",
    "adf_test",
    "kpss_test",
]
