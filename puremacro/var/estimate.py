"""Reduced-form VAR(p) estimator and lag-selection helpers."""

from __future__ import annotations

import numpy as np


def estimate_var(Y: np.ndarray, p: int):
    """OLS estimation with a constant.

    Returns
    -------
    VarEstimateResult
        Frozen dataclass with attributes ``A_list``, ``c``, ``Sigma``,
        ``resid``, ``X``. Iterable so existing 5-tuple unpacks continue to work.
    """
    from ._results import VarEstimateResult
    T, n = Y.shape
    X = np.column_stack([np.ones(T - p)] + [Y[p - l - 1 : T - l - 1] for l in range(p)])
    Yd = Y[p:]
    B = np.linalg.lstsq(X, Yd, rcond=None)[0]
    c = B[0]
    A_list = [B[1 + l * n : 1 + (l + 1) * n].T for l in range(p)]
    resid = Yd - X @ B
    Sigma = resid.T @ resid / (T - p - 1 - n * p)
    return VarEstimateResult(A_list=A_list, c=c, Sigma=Sigma, resid=resid, X=X)


def select_lag_bic(Y: np.ndarray, max_p: int = 8) -> int:
    """Pick lag by BIC. Returns p."""
    T, n = Y.shape
    best = (np.inf, 1)
    for p in range(1, max_p + 1):
        _, _, Sigma, _, _ = estimate_var(Y, p)
        ll = -0.5 * (T - p) * (n * np.log(2 * np.pi) + np.log(np.linalg.det(Sigma)) + n)
        k = n * (1 + n * p)
        bic = -2 * ll + k * np.log(T - p)
        if bic < best[0]:
            best = (bic, p)
    return best[1]


def lag_select(Y, maxlags=8, ic="bic"):
    """Choose VAR lag order by information criterion. Returns int p*."""
    Y = np.asarray(Y)
    T, n = Y.shape
    best_p, best_ic = 1, np.inf
    for p in range(1, maxlags + 1):
        try:
            _, _, Sigma, _, _ = estimate_var(Y, p)
            sign, logdet = np.linalg.slogdet(Sigma)
            T_eff = T - p
            k = n * p
            if ic == "aic":
                val = logdet + 2 * k * n / T_eff
            elif ic == "hq":
                val = logdet + 2 * k * n * np.log(np.log(T_eff)) / T_eff
            else:  # bic
                val = logdet + k * n * np.log(T_eff) / T_eff
            if val < best_ic:
                best_ic = val
                best_p = p
        except Exception:
            continue
    return best_p


def companion(A_list):
    """Return (np × np) companion matrix for stability checks."""
    p = len(A_list)
    n = A_list[0].shape[0]
    C = np.zeros((n * p, n * p))
    C[:n] = np.hstack(A_list)
    if p > 1:
        C[n:, :-n] = np.eye(n * (p - 1))
    return C


def is_stable(A_list):
    """Eigenvalues of companion all inside the unit circle."""
    eigs = np.linalg.eigvals(companion(A_list))
    return bool(np.all(np.abs(eigs) < 1.0))
