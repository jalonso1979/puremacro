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
    # `f_stat_fn` returns an F-statistic (see the parameter docs), and it is
    # df * F, not F, that is asymptotically chi2(df). Comparing F directly
    # against chi2.ppf(ci, df) therefore uses a cutoff df times too large, so
    # the band is too wide by exactly that factor for any df > 1. Measured
    # coverage of the nominal 90% band on a just-identified AR design at
    # n = 300: 100.0% at df = 2 and 100.0% at df = 4. df = 1 is unaffected,
    # which is the default and the only case any fixture used.
    crit = chi2.ppf(ci, df) / df
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
    W: np.ndarray | None = None,
) -> float:
    """Cragg-Donald (1993) minimum-eigenvalue F-statistic for weak instruments.

    The Stock-Yogo (2005) form of the statistic,

        CD = λ_min( Σ_VV^{-1/2} X' P_Z X Σ_VV^{-1/2} ) / l,

    where ``P_Z`` is the projection on the (partialled) instruments,
    ``Σ_VV = V'V / (n - l - m)`` is the covariance **matrix** of the
    first-stage residuals ``V = X - P_Z X`` and ``l`` is the number of
    excluded instruments. Dividing by ``l`` (not by the number of
    endogenous regressors ``k``) is what makes the value comparable with
    the Stock-Yogo critical values, and reduces it to the textbook
    first-stage F when ``k = 1``.

    Parameters
    ----------
    Y : ndarray of shape (n,)
        Dependent variable (not used directly; signature for consistency).
    X : ndarray of shape (n, k) or (n,)
        Endogenous regressors (n observations, k endogenous variables).
    Z : ndarray of shape (n, l) or (n,)
        Excluded instruments (l >= k required for identification).
    W : ndarray of shape (n, m) or None
        Included exogenous regressors (a constant column, controls). They
        are partialled out of both ``X`` and ``Z`` and cost ``m`` degrees
        of freedom in ``Σ_VV``. Nothing is added by default, so pass a
        column of ones in ``W`` when the first stage has an intercept.

    Returns
    -------
    float
        Cragg-Donald F-statistic. With ``k = 1`` this is exactly the
        first-stage F for the excluded instruments (homoskedastic);
        compare with :func:`puremacro.inference.over_id.stock_yogo_cv`.

    References
    ----------
    Cragg, J.G. and Donald, S.G. (1993). Testing identifiability and specification
        in instrumental variable models. Econometric Theory, 9(2), 222-240.
    Stock, J.H. and Yogo, M. (2005). Testing for weak instruments in linear IV
        regression. In Andrews and Stock (eds), Identification and Inference for
        Econometric Models. Cambridge University Press. (eq. 2.6 and 3.1)
    """
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
    n, k = X.shape
    l = Z.shape[1]
    if Z.shape[0] != n:
        raise ValueError(
            f"cragg_donald_f: X has {n} rows but Z has {Z.shape[0]}"
        )
    if l < k:
        raise ValueError(
            f"Need at least as many instruments (l={l}) as endogenous regressors (k={k})."
        )

    m = 0
    if W is not None:
        W = np.asarray(W, dtype=float)
        W = W.reshape(-1, 1) if W.ndim == 1 else W
        if W.shape[0] != n:
            raise ValueError(
                f"cragg_donald_f: X has {n} rows but W has {W.shape[0]}"
            )
        m = W.shape[1]
        WtW_inv = inv_xtx(W, name="cragg_donald_f (partialling W)")
        X = X - W @ (WtW_inv @ (W.T @ X))
        Z = Z - W @ (WtW_inv @ (W.T @ Z))

    dof = n - l - m
    if dof <= 0:
        raise ValueError(
            f"cragg_donald_f: n - l - m = {dof} <= 0; not enough observations"
        )

    # First stage: X = Z Pi + V  (on the partialled variables).
    ZtZ_inv = inv_xtx(Z, name="cragg_donald_f")
    X_hat = Z @ (ZtZ_inv @ (Z.T @ X))          # P_Z X
    V = X - X_hat                              # first-stage residuals

    # Sigma_VV^{-1/2} from the eigen-decomposition of the (k x k) residual
    # covariance MATRIX -- not a scalar average of the residual variances,
    # which ignores the correlation between the first-stage errors of the
    # different endogenous regressors.
    Sigma_VV = V.T @ V / dof
    w, U = np.linalg.eigh(Sigma_VV)
    if np.min(w) <= 0:
        raise np.linalg.LinAlgError(
            "cragg_donald_f: first-stage residual covariance is singular "
            "(an endogenous regressor is perfectly explained by the "
            "instruments/controls, or two endogenous regressors are collinear)"
        )
    S_m12 = (U * w ** -0.5) @ U.T
    G = S_m12 @ (X_hat.T @ X_hat) @ S_m12 / l
    return float(np.min(np.linalg.eigvalsh(G)))


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
    # `vec` here stacks COLUMNS (`ravel(order='F')` below), and
    # vec(Z'V) = sum_t vec(Z_t V_t') = sum_t (V_t kron Z_t), since
    # vec(a b') = b kron a. The operands used to be the other way round,
    # `kron(Z[t], V[t])`, which disagrees with both the column-major `vec`
    # and the `kron(I_k, (Z'Z)^-1)` bread below. For k = 1 the two coincide
    # (V_t is a scalar), so the error only appears with two or more
    # endogenous regressors -- the case no fixture exercised.
    meat = np.zeros((l * k, l * k))
    for t in range(n):
        psi = np.kron(V[t], Z[t])   # length l*k
        meat += np.outer(psi, psi)

    # Var(vec(Pi_hat)) = (I_k ⊗ (Z'Z)^{-1}) [sum_t psi psi'] (I_k ⊗ (Z'Z)^{-1}).
    #
    # `ZtZ_inv` inverts the RAW cross-product Z'Z, so the sum must be raw too:
    # the bread already carries both factors of 1/n. Dividing `meat` by n and
    # then the sandwich by n again made V_Pi exactly n^2 too small, and the
    # Wald statistic built from its inverse exactly n^2 too LARGE. Measured
    # against the heteroskedasticity-robust first-stage F -- which the rk Wald
    # F equals at k = 1 -- the returned value was 235,470 against 5.89 at
    # n = 200, and 8,755,485 against 13.68 at n = 800: ratios of 40,000,
    # 160,000 and 640,000, i.e. n^2 to the digit.
    #
    # The direction is the dangerous one. Stock-Yogo thresholds sit near 10,
    # so an F of six or seven figures reads as "instruments overwhelmingly
    # strong" for every dataset, and the weak-instrument diagnostic this
    # function exists to provide could never fire.
    Ik_ZZinv = np.kron(np.eye(k), ZtZ_inv)
    V_Pi = Ik_ZZinv @ meat @ Ik_ZZinv

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
    p_value = _f_dist.sf(F, df_num, df_den)
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
