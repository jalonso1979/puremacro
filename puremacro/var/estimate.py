"""Reduced-form VAR(p) estimator and lag-selection helpers."""

from __future__ import annotations

import numpy as np


def estimate_var(Y: np.ndarray, p: int | None = None, *, lags: int | None = None):
    """OLS estimation with a constant.

    Parameters
    ----------
    Y : array-like of shape (T, n)
        Endogenous variables. Can be a NumPy ndarray or pandas DataFrame.
    p : int, optional
        Lag order (positional or keyword).
    lags : int, optional
        Keyword alias for ``p``.

    Returns
    -------
    VarEstimateResult
        Frozen dataclass with attributes ``A_list``, ``c``, ``Sigma``,
        ``resid``, ``X``, ``names``. Iterable so existing 5-tuple unpacks continue to work.
    """
    from ._results import VarEstimateResult

    if lags is not None:
        if p is not None and p != lags:
            raise ValueError(f"Conflicting lag arguments: p={p}, lags={lags}")
        p = lags
    if p is None:
        raise ValueError("Must specify lag order via positional `p` or keyword `lags=...`")

    names: tuple[str, ...] = ()
    if hasattr(Y, "columns"):
        names = tuple(str(c) for c in Y.columns)

    Y_arr = np.asarray(Y, dtype=float)
    T, n = Y_arr.shape
    # `np.linalg.lstsq` does NOT raise on a non-finite design matrix. LAPACK
    # prints "** On entry to DLASCL parameter number 4 had an illegal value"
    # straight to the terminal -- bypassing sys.stderr, so it survives a `2>`
    # redirect and is invisible to pytest -- and then returns a coefficient
    # matrix of NaN. Every downstream object (Sigma, the residuals, every IRF
    # and every bootstrap band built on them) is silently NaN-valued but
    # perfectly well-formed. This is the whole package's second promise, so it
    # is a named error here rather than garbage downstream.
    # asarray first: callers pass DataFrames here too, and `np.isfinite(df)`
    # returns a DataFrame whose truth value is ambiguous.
    finite = np.isfinite(Y_arr)
    if not finite.all():
        bad = int((~finite).sum())
        raise np.linalg.LinAlgError(
            f"estimate_var: Y contains {bad} non-finite value(s). OLS on them "
            "returns all-NaN coefficients without raising, so every IRF and "
            "band built from this fit would be NaN. Drop or interpolate the "
            "missing observations first."
        )
    X = np.column_stack([np.ones(T - p)] + [Y_arr[p - l - 1 : T - l - 1] for l in range(p)])
    Yd = Y_arr[p:]
    B = np.linalg.lstsq(X, Yd, rcond=None)[0]
    c = B[0]
    A_list = [B[1 + l * n : 1 + (l + 1) * n].T for l in range(p)]
    resid = Yd - X @ B
    Sigma = resid.T @ resid / (T - p - 1 - n * p)
    return VarEstimateResult(A_list=A_list, c=c, Sigma=Sigma, resid=resid, X=X, names=names)


def select_lag_bic(Y: np.ndarray, max_p: int = 8, *, max_lags: int | None = None) -> int:
    """Pick lag by BIC. Returns p."""
    if max_lags is not None:
        max_p = max_lags
    Y_arr = np.asarray(Y, dtype=float)
    T, n = Y_arr.shape
    best = (np.inf, 1)
    for p in range(1, max_p + 1):
        _, _, Sigma, _, _ = estimate_var(Y_arr, p)
        ll = -0.5 * (T - p) * (n * np.log(2 * np.pi) + np.log(np.linalg.det(Sigma)) + n)
        k = n * (1 + n * p)
        bic = -2 * ll + k * np.log(T - p)
        if bic < best[0]:
            best = (bic, p)
    return best[1]


def lag_select(Y, maxlags=8, ic="bic", *, max_lags: int | None = None):
    """Choose VAR lag order by information criterion. Returns int p*."""
    if max_lags is not None:
        maxlags = max_lags
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
