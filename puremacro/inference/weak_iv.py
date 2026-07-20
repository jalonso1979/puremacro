"""Weak-IV-robust confidence bands via Anderson-Rubin F-stat inversion.

Reference: Montiel-Olea, Stock, Watson (2020). Our implementation is a simple
grid inversion; for large grids, consider binary search.

Also provides Cragg-Donald and Kleibergen-Paap weak-instrument rank statistics.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from scipy.stats import chi2

from .._linalg import inv_xtx
from ._results import ARTestResult


def anderson_rubin_band(
    beta_grid: np.ndarray,
    f_stat_fn: Callable[[float], float],
    ci: float = 0.9,
    df: int = 1,
) -> tuple[float, float]:
    """Return (lower, upper) endpoints of the Anderson-Rubin CI.

    Parameters
    ----------
    beta_grid  : 1-D array of candidate β values.
    f_stat_fn  : callable β -> F-statistic.
    ci         : confidence level (default 0.9).
    df         : degrees of freedom for the χ² critical (default 1 = scalar test).
    """
    crit = chi2.ppf(ci, df)
    accepted = np.array([f_stat_fn(b) <= crit for b in beta_grid], dtype=bool)
    if not accepted.any():
        warnings.warn("AR band is empty on the given grid; returning grid endpoints.")
        return float(beta_grid[0]), float(beta_grid[-1])
    idx = np.where(accepted)[0]
    lo = float(beta_grid[idx[0]])
    hi = float(beta_grid[idx[-1]])
    # Check for disconnectedness (holes in accepted region).
    if not accepted[idx[0] : idx[-1] + 1].all():
        warnings.warn(
            "AR confidence set is disconnected; reporting enclosing interval [lo, hi]."
        )
    return lo, hi


def cragg_donald_f(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
) -> float:
    """Cragg-Donald (1993) minimum eigenvalue F-statistic for weak instruments.

    Tests the null hypothesis of weak instruments in IV/TSLS regressions
    with potentially multiple endogenous regressors.

    Parameters
    ----------
    Y : ndarray of shape (n,)
        Dependent variable (not used directly; signature for consistency).
    X : ndarray of shape (n, k)
        Endogenous regressors (n observations, k endogenous variables).
    Z : ndarray of shape (n, l)
        Instruments (l >= k required for identification).

    Returns
    -------
    float
        Cragg-Donald F-statistic: minimum eigenvalue of the concentration matrix
        divided by k (number of endogenous regressors).

    References
    ----------
    Cragg, J.G. and Donald, S.G. (1993). Testing identifiability and specification
        in instrumental variable models. Econometric Theory, 9(2), 222-240.
    Stock, J.H. and Yogo, M. (2005). Testing for weak instruments in linear IV
        regression. In Andrews and Stock (eds), Identification and Inference for
        Econometric Models. Cambridge University Press.
    """
    n = X.shape[0]
    k = X.shape[1] if X.ndim > 1 else 1
    X = np.atleast_2d(X).T if X.ndim == 1 else X
    Z = np.atleast_2d(Z).T if Z.ndim == 1 else Z
    l = Z.shape[1]

    if l < k:
        raise ValueError(
            f"Need at least as many instruments (l={l}) as endogenous regressors (k={k})."
        )

    # Project out any included exogenous regressors (assumed none here for purity).
    # First stage: regress each X_j on Z.
    Mz = np.eye(n) - Z @ np.linalg.solve(Z.T @ Z, Z.T)
    X_tilde = X - Z @ np.linalg.solve(Z.T @ Z, Z.T @ X)   # residuals of X on Z
    X_hat = X - X_tilde                                      # fitted values

    # Concentration matrix: (X_hat' X_hat) scaled by sigma^2 of first-stage residuals.
    # Use average sigma^2 across endogenous regressors.
    sigma2 = np.mean(np.sum(X_tilde ** 2, axis=0)) / (n - l)
    if sigma2 <= 0:
        return 0.0

    conc = X_hat.T @ X_hat / sigma2
    # Minimum eigenvalue of concentration matrix / k.
    eigvals = np.linalg.eigvalsh(conc)
    cd_stat = float(np.min(eigvals)) / k
    return cd_stat


def kleibergen_paap_f(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray | None = None,
) -> float:
    """Kleibergen-Paap (2006) rk F-statistic for weak instruments.

    A heteroskedasticity-robust analog of the Cragg-Donald statistic.
    Returns the rk Wald statistic divided by k (number of endogenous regressors),
    suitable for comparison with Stock-Yogo critical values as an approximation.

    Parameters
    ----------
    Y : ndarray of shape (n,)
        Dependent variable (not used directly; included for API symmetry).
    X : ndarray of shape (n, k) or (n,)
        Endogenous regressors.
    Z : ndarray of shape (n, l) or (n,)
        Instruments (l >= k).
    W : ndarray of shape (n, m) or None
        Included exogenous controls (partialled out if provided).

    Returns
    -------
    float
        Kleibergen-Paap rk F-statistic.

    References
    ----------
    Kleibergen, F. and Paap, R. (2006). Generalized reduced rank tests using
        the singular value decomposition. Journal of Econometrics, 133(1), 97-126.
    """
    n = len(Y) if hasattr(Y, '__len__') else X.shape[0]
    X = np.atleast_2d(X if X.ndim > 1 else X[:, None])
    Z = np.atleast_2d(Z if Z.ndim > 1 else Z[:, None])
    k = X.shape[1]
    l = Z.shape[1]

    if l < k:
        raise ValueError(
            f"Need at least as many instruments (l={l}) as endogenous regressors (k={k})."
        )

    # Partial out included exogenous regressors W if provided.
    if W is not None:
        W = np.atleast_2d(W if W.ndim > 1 else W[:, None])
        Mw = np.eye(n) - W @ np.linalg.solve(W.T @ W, W.T)
        X = Mw @ X
        Z = Mw @ Z

    # First-stage: Pi = (Z'Z)^{-1} Z'X  (l x k matrix of reduced-form coefficients).
    # Compute the inverse first so a singular Z'Z surfaces a named
    # diagnostic instead of a bare "Singular matrix" from np.linalg.solve.
    ZtZ_inv = inv_xtx(Z, name="kleibergen_paap_f")
    Pi_hat = ZtZ_inv @ (Z.T @ X)   # (l, k)

    # Residuals of X projected on Z.
    X_hat = Z @ Pi_hat
    V = X - X_hat   # (n, k) first-stage residuals

    # Heteroskedastic sandwich for vec(Pi_hat).
    # Influence function: psi_t = Z_t ⊗ v_t  for each observation.
    # Omega = (1/n) * sum_t psi_t psi_t'.
    # We compute the robust variance of vec(Pi_hat).
    meat = np.zeros((l * k, l * k))
    for t in range(n):
        psi = np.kron(Z[t], V[t])   # length l*k
        meat += np.outer(psi, psi)
    meat /= n

    # Var(vec(Pi_hat)) = (I_k ⊗ (Z'Z)^{-1}) Omega (I_k ⊗ (Z'Z)^{-1}) / n
    Ik_ZZinv = np.kron(np.eye(k), ZtZ_inv)
    V_Pi = Ik_ZZinv @ meat @ Ik_ZZinv / n

    # rk statistic: minimum singular value of Pi_hat scaled by its s.e.
    # Approximation: Wald statistic for H0: rank(Pi) < k.
    # Use vec(Pi_hat)' V_Pi^{-1} vec(Pi_hat) / (l * k) as a summary F-stat.
    try:
        vec_pi = Pi_hat.ravel(order='F')   # vec in Fortran (column-major) order
        V_Pi_inv = np.linalg.pinv(V_Pi)
        rk_stat = float(vec_pi @ V_Pi_inv @ vec_pi) / (l * k)
    except np.linalg.LinAlgError:
        return 0.0

    return rk_stat


def olea_pflueger_f(
    x_endog: np.ndarray,
    z_inst: np.ndarray,
    cluster: np.ndarray | None = None,
) -> float:
    """Olea-Pflueger (2013) effective F-statistic for weak instruments.

    The effective F is the modern weak-IV-robust replacement for the
    Stock-Yogo Wald F. It is constructed to be conservative under
    heteroskedasticity and (optionally) clustering.

    Parameters
    ----------
    x_endog : ndarray, shape (T,) or (T, 1)
        Endogenous regressor (one-dimensional).
    z_inst : ndarray, shape (T, k)
        Instrument matrix (k instruments).
    cluster : ndarray of int, shape (T,), optional
        Cluster identifier. If provided, the variance is computed
        cluster-robustly; otherwise heteroskedasticity-robust (HC0).

    Returns
    -------
    f_eff : float
        Olea-Pflueger effective F-statistic. Reference cutoffs (5%
        worst-case bias, k=1): F > 23.1 (strong); F < 23.1 means
        weak-IV-robust inference recommended.

    References
    ----------
    Olea, J.L.M. and Pflueger, C. (2013). A robust test for weak
        instruments. JBES 31(3), 358-369.
    """
    x = np.asarray(x_endog).reshape(-1)
    Z = np.asarray(z_inst)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    T, k = Z.shape
    if x.shape[0] != T:
        raise ValueError(
            f"olea_pflueger_f: x_endog length {x.shape[0]} != z_inst rows {T}"
        )
    # Demean (the proxy-SVAR convention; OP also derive without intercept,
    # but demeaning is standard in practice).
    x = x - x.mean()
    Z = Z - Z.mean(axis=0, keepdims=True)
    ZtZ_inv = inv_xtx(Z, name="olea_pflueger_f")
    Pi = ZtZ_inv @ (Z.T @ x)              # shape (k,)
    resid = x - Z @ Pi
    if cluster is None:
        # HC0 sandwich on Z' u u' Z
        ZtuutZ = (Z.T * resid**2) @ Z
    else:
        cluster = np.asarray(cluster).reshape(-1)
        if cluster.shape[0] != T:
            raise ValueError("olea_pflueger_f: cluster length mismatch")
        ZtuutZ = np.zeros((k, k))
        for g in np.unique(cluster):
            mask = cluster == g
            ug = resid[mask]
            Zg = Z[mask]
            score_g = Zg.T @ ug
            ZtuutZ += np.outer(score_g, score_g)
    # Effective F: F_eff = Pi' (Z'Z) Pi / tr( (Z'Z)^{-1} Z' uu' Z )
    num = Pi @ (Z.T @ Z) @ Pi
    denom = np.trace(ZtZ_inv @ ZtuutZ)
    if denom <= 0:
        return float("inf")
    return float(num / denom)


# ====================================================================
# Anderson-Rubin (1949) test + Montiel Olea-Stock-Watson (2021) bands
# ====================================================================
from scipy.stats import f as _f_dist


def anderson_rubin_test(
    beta0: float,
    y: np.ndarray,
    x_endog: np.ndarray,
    z: np.ndarray,
    controls: np.ndarray | None = None,
) -> ARTestResult:
    """Anderson-Rubin (1949) test of H₀: β = β₀ in y = β x + e via instrument z.

    The test regresses w_t ≡ y_t - β₀ x_t on z_t (and any controls) and
    tests whether z's coefficient = 0. Under H₀, z has no explanatory
    power for w_t (because the only variation in w_t orthogonal to
    controls is the structural error). Under H₁, β₀ is wrong so z's
    relationship with x bleeds into w.

    Returns
    -------
    ARTestResult
        Frozen dataclass with fields ``stat`` (F), ``p_value``, ``df_num``,
        ``df_den``, ``residual_ss``.

    References
    ----------
    Anderson, T.W. and Rubin, H. (1949). Estimation of the parameters of
        a single equation in a complete system of stochastic equations.
        Annals of Mathematical Statistics 20(1), 46-63.
    """
    y = np.asarray(y, dtype=float).ravel()
    x_endog = np.asarray(x_endog, dtype=float).ravel()
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    T = len(y)
    w = y - beta0 * x_endog  # residual under H_0

    # Restricted: w = α + (controls) γ + ε
    if controls is None:
        X_r = np.ones((T, 1))
    else:
        controls = np.asarray(controls, dtype=float)
        if controls.ndim == 1:
            controls = controls.reshape(-1, 1)
        X_r = np.column_stack([np.ones(T), controls])
    beta_r, *_ = np.linalg.lstsq(X_r, w, rcond=None)
    rss_r = float(np.sum((w - X_r @ beta_r) ** 2))

    # Unrestricted: w = α + (controls) γ + z δ + ε
    X_u = np.column_stack([X_r, z])
    beta_u, *_ = np.linalg.lstsq(X_u, w, rcond=None)
    rss_u = float(np.sum((w - X_u @ beta_u) ** 2))

    df_num = z.shape[1]
    df_den = T - X_u.shape[1]
    if df_den <= 0 or rss_u <= 0:
        return ARTestResult(
            stat=float("nan"), p_value=float("nan"),
            df_num=int(df_num), df_den=int(df_den),
            residual_ss=float(rss_u),
        )
    F = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
    p_value = 1.0 - _f_dist.cdf(F, df_num, df_den)
    return ARTestResult(
        stat=float(F), p_value=float(p_value),
        df_num=int(df_num), df_den=int(df_den),
        residual_ss=float(rss_u),
    )


def msw_bands(
    grid: np.ndarray,
    y: np.ndarray,
    x_endog: np.ndarray,
    z: np.ndarray,
    alpha: float = 0.10,
    controls: np.ndarray | None = None,
) -> tuple[float, float]:
    """Montiel Olea-Stock-Watson (2021) weak-IV-robust confidence band
    for β via Anderson-Rubin grid inversion.

    Returns the (lo, hi) endpoints of the connected (1-alpha)-confidence
    set: the smallest and largest β₀ in `grid` for which AR fails to reject.

    Parameters
    ----------
    grid : np.ndarray
        Candidate β values to invert. Should bracket the 2SLS estimate
        with enough resolution.
    alpha : float
        Significance level (default 0.10 = 90% CI).
    """
    p_values = np.array([
        anderson_rubin_test(b0, y, x_endog, z, controls=controls).p_value
        for b0 in grid
    ])
    accepted = p_values >= alpha
    if not accepted.any():
        return float(grid[np.argmax(p_values)]), float(grid[np.argmax(p_values)])
    accepted_grid = grid[accepted]
    return float(accepted_grid.min()), float(accepted_grid.max())


__all__ = [
    "anderson_rubin_band",
    "cragg_donald_f",
    "kleibergen_paap_f",
    "olea_pflueger_f",
    "anderson_rubin_test",
    "msw_bands",
    "ARTestResult",
]
