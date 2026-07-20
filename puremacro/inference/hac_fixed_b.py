"""Lazarus-Lewis-Stock-Watson (2018) fixed-b HAC.

Same Bartlett kernel as standard Newey-West, but bandwidth = b·T for a
fixed b ∈ (0, 1] rather than the data-dependent rule. The fixed-b
sampling distribution differs from the standard normal, so the
critical values are tabulated separately. For two-sided 90% bands
(α = 0.10) at b = 0.10, the LLSW critical value is approximately 1.86
(vs. 1.645 for normal).

We provide:
- hac_fixed_b(u, X, b=0.1) → sandwich vcov with bandwidth ⌊bT⌋.
- llsw_critical_value(b, alpha=0.10) → tabulated CV.
"""
from __future__ import annotations

import numpy as np

from .._linalg import inv_xtx

# Tabulated LLSW two-sided critical values for the Bartlett kernel.
# Source: Lazarus-Lewis-Stock-Watson (2018), Table 2.
# Keys: (b, alpha) → critical value.
_LLSW_CV = {
    # b = 0.05
    (0.05, 0.10): 1.83,
    (0.05, 0.05): 2.16,
    (0.05, 0.01): 2.79,
    # b = 0.10
    (0.10, 0.10): 1.86,
    (0.10, 0.05): 2.21,
    (0.10, 0.01): 2.90,
    # b = 0.20
    (0.20, 0.10): 1.93,
    (0.20, 0.05): 2.32,
    (0.20, 0.01): 3.13,
    # b = 0.30
    (0.30, 0.10): 2.04,
    (0.30, 0.05): 2.46,
    (0.30, 0.01): 3.40,
    # b = 0.50
    (0.50, 0.10): 2.30,
    (0.50, 0.05): 2.83,
    (0.50, 0.01): 4.07,
    # b = 1.00
    (1.00, 0.10): 3.05,
    (1.00, 0.05): 3.96,
    (1.00, 0.01): 6.29,
}


def llsw_critical_value(b: float, alpha: float = 0.10) -> float:
    """Lookup the LLSW (2018) Bartlett-fixed-b two-sided CV.

    Linearly interpolates between tabulated b values when needed.
    """
    bs = sorted({k[0] for k in _LLSW_CV.keys()})
    if alpha not in {0.01, 0.05, 0.10}:
        # Fallback to nearest tabulated alpha
        alpha = min({0.01, 0.05, 0.10}, key=lambda a: abs(a - alpha))
    if b <= bs[0]:
        return _LLSW_CV[(bs[0], alpha)]
    if b >= bs[-1]:
        return _LLSW_CV[(bs[-1], alpha)]
    # Linear interp between two surrounding b's
    for i in range(len(bs) - 1):
        b_lo, b_hi = bs[i], bs[i + 1]
        if b_lo <= b <= b_hi:
            cv_lo = _LLSW_CV[(b_lo, alpha)]
            cv_hi = _LLSW_CV[(b_hi, alpha)]
            w = (b - b_lo) / (b_hi - b_lo)
            return cv_lo + w * (cv_hi - cv_lo)
    return _LLSW_CV[(0.10, alpha)]  # default


def hac_fixed_b(y, X, b: float = 0.10) -> dict:
    """OLS with LLSW fixed-b Bartlett HAC SE.

    bandwidth = max(1, ⌊b T⌋). For inference, use llsw_critical_value(b)
    instead of the standard normal quantile.

    Returns dict with keys: beta, se, t, vcov, residuals, n_obs,
                              b, lags, llsw_cv_90.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    T, k = X.shape
    L = max(1, int(np.floor(b * T)))
    XtX_inv = inv_xtx(X, name="hac_fixed_b")
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    S = (X * u[:, None]).T @ (X * u[:, None])  # ell=0
    for ell in range(1, L + 1):
        w = 1.0 - ell / (L + 1.0)
        Gamma = (X[ell:] * u[ell:, None]).T @ (X[:-ell] * u[:-ell, None])
        S += w * (Gamma + Gamma.T)
    vcov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(vcov))
    return {
        "beta": beta,
        "se": se,
        "t": beta / se,
        "vcov": vcov,
        "residuals": u,
        "n_obs": int(T),
        "b": float(b),
        "lags": int(L),
        "llsw_cv_90": float(llsw_critical_value(b, alpha=0.10)),
    }


__all__ = ["hac_fixed_b", "llsw_critical_value"]
