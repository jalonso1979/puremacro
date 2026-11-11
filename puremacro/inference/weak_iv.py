"""Weak-IV-robust confidence bands via Anderson-Rubin F-stat inversion.

Reference: Montiel-Olea, Stock, Watson (2020). Our implementation is a simple
grid inversion; for large grids, consider binary search.

Also provides Cragg-Donald and Kleibergen-Paap weak-instrument rank statistics.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .._linalg import inv_xtx
from ._results import ARTestResult


def _check_lags(lags: int | None, func: str) -> int | None:
    if lags is None:
        return None
    if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)) or int(lags) < 0:
        raise ValueError(f"{func}: lags must be a non-negative integer, got {lags!r}")
    return int(lags)


def _bartlett_lrv(scores: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West Bartlett-kernel long-run covariance of score matrix ``scores``."""
    out = scores.T @ scores
    for ell in range(1, lags + 1):
        weight = 1.0 - ell / (lags + 1.0)
        gamma_ell = scores[ell:].T @ scores[:-ell]
        out = out + weight * (gamma_ell + gamma_ell.T)
    return out


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
        warnings.warn(
            "AR band is empty on the given grid; returning (nan, nan).",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan"), float("nan")
    idx = np.where(accepted)[0]
    lo = float(beta_grid[idx[0]])
    hi = float(beta_grid[idx[-1]])
    # Check for disconnectedness (holes in accepted region).
    if not accepted[idx[0] : idx[-1] + 1].all():
        warnings.warn(
            "AR confidence set is disconnected; reporting enclosing interval [lo, hi].",
            RuntimeWarning,
            stacklevel=2,
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
        compare with :func:`puremacro.inference.over_id.stock_yogo_cv`
        (``criterion="bias"`` for ``l >= 3``, ``criterion="size"`` for
        any ``l``).

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
    """Kleibergen-Paap (2006) rk Wald F-statistic for weak instruments.

    A heteroskedasticity-robust analog of the Cragg-Donald minimum-eigenvalue
    statistic testing ``H0: rank(Pi) = k - 1`` against ``H1: rank(Pi) = k``.
    Returns the Kleibergen-Paap ``rk`` Wald statistic divided by ``l`` (the
    number of excluded instruments), matching the Stock-Yogo scaling of
    :func:`cragg_donald_f` and reducing identically to the HC0 first-stage
    Wald F when ``k = 1``.

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
        Kleibergen-Paap rk Wald F-statistic.

    References
    ----------
    Kleibergen, F. and Paap, R. (2006). Generalized reduced rank tests using
        the singular value decomposition. Journal of Econometrics, 133(1), 97-126.
    """
    X_arr = np.asarray(X, dtype=float)
    Z_arr = np.asarray(Z, dtype=float)
    X_arr = X_arr.reshape(-1, 1) if X_arr.ndim == 1 else X_arr
    Z_arr = Z_arr.reshape(-1, 1) if Z_arr.ndim == 1 else Z_arr
    n, k = X_arr.shape
    l = Z_arr.shape[1]
    if Z_arr.shape[0] != n:
        raise ValueError(
            f"kleibergen_paap_f: X has {n} rows but Z has {Z_arr.shape[0]}"
        )
    if l < k or k < 1:
        raise ValueError(
            f"Need at least as many instruments (l={l}) as endogenous regressors (k={k})."
        )

    m = 0
    if W is not None:
        W_arr = np.asarray(W, dtype=float)
        W_arr = W_arr.reshape(-1, 1) if W_arr.ndim == 1 else W_arr
        if W_arr.shape[0] != n:
            raise ValueError(
                f"kleibergen_paap_f: X has {n} rows but W has {W_arr.shape[0]}"
            )
        m = W_arr.shape[1]
        WtW_inv = inv_xtx(W_arr, name="kleibergen_paap_f (partialling W)")
        X_arr = X_arr - W_arr @ (WtW_inv @ (W_arr.T @ X_arr))
        Z_arr = Z_arr - W_arr @ (WtW_inv @ (W_arr.T @ Z_arr))

    dof = n - l - m
    if dof <= 0:
        raise ValueError(
            f"kleibergen_paap_f: n - l - m = {dof} <= 0; not enough observations"
        )

    # Validate full column rank of Z via inv_xtx so singular designs surface a
    # named diagnostic ("kleibergen_paap_f").
    _ = inv_xtx(Z_arr, name="kleibergen_paap_f")

    # Orthonormalize partialled instruments: Z_arr = Q_Z @ R_Z, Q_Z' Q_Z = I_l.
    Q_Z, _ = np.linalg.qr(Z_arr, mode="reduced")
    X_hat = Q_Z @ (Q_Z.T @ X_arr)
    V = X_arr - X_hat

    Sigma_VV = (V.T @ V) / dof
    eigvals, eigvecs = np.linalg.eigh(Sigma_VV)
    if np.min(eigvals) <= 0:
        raise np.linalg.LinAlgError(
            "kleibergen_paap_f: first-stage residual covariance is singular "
            "(an endogenous regressor is perfectly explained by the "
            "instruments/controls, or two endogenous regressors are collinear)"
        )
    F_scale = (eigvecs * (eigvals ** -0.5)) @ eigvecs.T

    # Standardized reduced-form matrix Theta_hat = Q_Z' X F_scale (l x k)
    # and full SVD Theta_hat = U S V_svd' for reduced-rank null q = k - 1.
    Theta_hat = (Q_Z.T @ X_arr) @ F_scale
    U, svals, Vt = np.linalg.svd(Theta_hat, full_matrices=True)
    if svals[-1] <= 1e-15 * max(1.0, float(svals[0])):
        return 0.0

    U_2 = U[:, k - 1 :]          # (l, l - k + 1)
    v_k = Vt[k - 1, :]           # (k,)
    ell_q = U_2.T @ (Theta_hat @ v_k)  # (l - k + 1,)

    # Observation-level score for ell_q: g_i = (Q_Z U_2)_i * (V F_scale v_k)_i.
    Z_rot = Q_Z @ U_2            # (n, l - k + 1)
    e_rot = V @ (F_scale @ v_k)  # (n,)
    Omega_q = (Z_rot * (e_rot ** 2)[:, None]).T @ Z_rot

    try:
        rk_wald = float(ell_q @ np.linalg.solve(Omega_q, ell_q))
    except np.linalg.LinAlgError:
        rk_wald = float(ell_q @ np.linalg.pinv(Omega_q) @ ell_q)
    return max(0.0, rk_wald / l)


def olea_pflueger_f(
    x_endog: np.ndarray,
    z_inst: np.ndarray,
    cluster: np.ndarray | None = None,
    *,
    controls: np.ndarray | None = None,
    lags: int | None = None,
) -> float:
    """Olea-Pflueger (2013) effective F-statistic for weak instruments.

    The effective F is the modern weak-IV-robust replacement for the
    Stock-Yogo Wald F. It is constructed to be conservative under
    heteroskedasticity, clustering, or Newey-West HAC serial correlation.

    Parameters
    ----------
    x_endog : ndarray, shape (T,) or (T, 1)
        Endogenous regressor (one-dimensional).
    z_inst : ndarray, shape (T, k)
        Instrument matrix (k instruments).
    cluster : ndarray of int, shape (T,), optional
        Cluster identifier. If provided, the variance is computed
        cluster-robustly; cannot be combined with ``lags``.
    controls : ndarray, shape (T,) or (T, p), optional
        Control variables to partial out (along with an intercept) before
        computing the first-stage effective F. When ``None``, ``x_endog`` and
        ``z_inst`` are demeaned.
    lags : int, optional
        Newey-West Bartlett-kernel lag truncation bandwidth. ``lags=0`` equals
        the default Eicker-Huber-White (HC0) variance; for local projections
        at horizon ``h``, pass ``lags=h + 1`` to match :func:`puremacro.lp.lp_iv`.

    Returns
    -------
    f_eff : float
        Olea-Pflueger effective F-statistic. Reference cutoff (5% level,
        10% worst-case Nagar bias, k=1): F > 23.1 (strong); F < 23.1 means
        weak-IV-robust inference recommended. The critical value for any
        ``k``, ``tau`` and level is :func:`puremacro.lp.iv.mop_critical_values`.

    References
    ----------
    Olea, J.L.M. and Pflueger, C. (2013). A robust test for weak
        instruments. JBES 31(3), 358-369.
    """
    lags_int = _check_lags(lags, "olea_pflueger_f")
    if cluster is not None and lags_int is not None:
        raise ValueError("olea_pflueger_f: pass at most one of cluster or lags")
    x = np.asarray(x_endog, dtype=float).reshape(-1)
    Z = np.asarray(z_inst, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    T, k = Z.shape
    if x.shape[0] != T:
        raise ValueError(
            f"olea_pflueger_f: x_endog length {x.shape[0]} != z_inst rows {T}"
        )
    if controls is None:
        x = x - x.mean()
        Z = Z - Z.mean(axis=0, keepdims=True)
    else:
        W = np.asarray(controls, dtype=float)
        if W.ndim == 1:
            W = W.reshape(-1, 1)
        if W.shape[0] != T:
            raise ValueError(
                f"olea_pflueger_f: controls rows {W.shape[0]} != z_inst rows {T}"
            )
        W_ctl = np.column_stack([np.ones(T), W])
        W_inv = np.linalg.pinv(W_ctl.T @ W_ctl)
        x = x - W_ctl @ (W_inv @ (W_ctl.T @ x))
        Z = Z - W_ctl @ (W_inv @ (W_ctl.T @ Z))

    ZtZ_inv = inv_xtx(Z, name="olea_pflueger_f")
    Pi = ZtZ_inv @ (Z.T @ x)              # shape (k,)
    resid = x - Z @ Pi
    if lags_int is not None:
        scores = Z * resid[:, None]
        ZtuutZ = _bartlett_lrv(scores, lags_int)
    elif cluster is None:
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
    *,
    lags: int | None = None,
) -> ARTestResult:
    """Anderson-Rubin (1949) test of H₀: β = β₀ in y = β x + e via instrument z.

    The test regresses w_t ≡ y_t - β₀ x_t on z_t (and any controls) and
    tests whether z's coefficient = 0. Under H₀, z has no explanatory
    power for w_t (because the only variation in w_t orthogonal to
    controls is the structural error). Under H₁, β₀ is wrong so z's
    relationship with x bleeds into w.

    Parameters
    ----------
    beta0 : float
        Hypothesized slope coefficient β₀.
    y, x_endog : ndarray, shape (T,)
        Outcome and endogenous regressor.
    z : ndarray, shape (T,) or (T, k)
        Excluded instrument(s).
    controls : ndarray, shape (T,) or (T, p), optional
        Included exogenous controls (an intercept is always included).
    lags : int, optional
        When ``None`` (default), uses the classical homoskedastic F test.
        When an integer ``lags >= 0`` is passed, computes the Newey-West
        Bartlett-HAC Wald statistic ``W / k`` with asymptotic ``χ²(k)``
        p-value, matching :func:`puremacro.lp.lp_iv` when ``lags = h + 1``.

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
    lags_int = _check_lags(lags, "anderson_rubin_test")
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

    df_num = z.shape[1]
    df_den = T - (X_r.shape[1] + df_num)

    if lags_int is not None:
        W_inv = np.linalg.pinv(X_r.T @ X_r)
        w_tilde = w - X_r @ (W_inv @ (X_r.T @ w))
        Z_tilde = z - X_r @ (W_inv @ (X_r.T @ z))
        ZtZ_inv = np.linalg.pinv(Z_tilde.T @ Z_tilde)
        delta = ZtZ_inv @ (Z_tilde.T @ w_tilde)
        u_hat = w_tilde - Z_tilde @ delta
        rss_u = float(np.sum(u_hat ** 2))
        if df_den <= 0 or rss_u <= 0:
            return ARTestResult(
                stat=float("nan"), p_value=float("nan"),
                df_num=int(df_num), df_den=int(df_den),
                residual_ss=float(rss_u),
            )
        scores = Z_tilde * u_hat[:, None]
        Omega = _bartlett_lrv(scores, lags_int)
        V_delta = ZtZ_inv @ Omega @ ZtZ_inv
        try:
            wald = float(delta @ np.linalg.solve(V_delta, delta))
        except np.linalg.LinAlgError:
            wald = float(delta @ np.linalg.pinv(V_delta) @ delta)
        F = wald / df_num
        p_value = float(chi2.sf(wald, df_num))
        return ARTestResult(
            stat=float(F), p_value=float(p_value),
            df_num=int(df_num), df_den=int(df_den),
            residual_ss=float(rss_u),
        )

    beta_r, *_ = np.linalg.lstsq(X_r, w, rcond=None)
    rss_r = float(np.sum((w - X_r @ beta_r) ** 2))

    # Unrestricted: w = α + (controls) γ + z δ + ε
    X_u = np.column_stack([X_r, z])
    beta_u, *_ = np.linalg.lstsq(X_u, w, rcond=None)
    rss_u = float(np.sum((w - X_u @ beta_u) ** 2))

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
    *,
    lags: int | None = None,
) -> tuple[float, float]:
    """Montiel Olea-Stock-Watson (2021) weak-IV-robust confidence band
    for β via Anderson-Rubin grid inversion.

    Returns the (lo, hi) endpoints of the (1-alpha)-confidence set on
    ``grid``: the smallest and largest β₀ in ``grid`` for which AR fails to
    reject. Emits a :class:`RuntimeWarning` when the accepted set is empty
    (returning ``(nan, nan)``), touches a grid endpoint (potentially
    unbounded), or is disconnected.

    Parameters
    ----------
    grid : np.ndarray
        Candidate β values to invert. Should bracket the 2SLS estimate
        with enough resolution.
    alpha : float
        Significance level (default 0.10 = 90% CI).
    controls : np.ndarray, optional
        Included exogenous controls.
    lags : int, optional
        Newey-West Bartlett-HAC lag truncation bandwidth passed to
        :func:`anderson_rubin_test`.
    """
    lags_int = _check_lags(lags, "msw_bands")
    grid_arr = np.asarray(grid, dtype=float).ravel()
    p_values = np.array([
        anderson_rubin_test(b0, y, x_endog, z, controls=controls, lags=lags_int).p_value
        for b0 in grid_arr
    ])
    accepted = p_values >= alpha
    if not accepted.any():
        warnings.warn(
            "msw_bands: Anderson-Rubin confidence set is empty on the supplied grid; "
            "returning (nan, nan).",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan"), float("nan")
    idx = np.flatnonzero(accepted)
    if not accepted[idx[0] : idx[-1] + 1].all():
        warnings.warn(
            "msw_bands: Anderson-Rubin confidence set is disconnected on the supplied "
            "grid; returning the enclosing interval [lo, hi].",
            RuntimeWarning,
            stacklevel=2,
        )
    if accepted[0] or accepted[-1]:
        warnings.warn(
            "msw_bands: Anderson-Rubin confidence set touches the grid boundary; "
            "the set may be unbounded or require a wider grid.",
            RuntimeWarning,
            stacklevel=2,
        )
    return float(grid_arr[idx[0]]), float(grid_arr[idx[-1]])


def instrument_influence(
    z: np.ndarray | pd.Series,
    controls: np.ndarray | pd.DataFrame | None = None,
    weights: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Per-observation share of residualized instrument variation ``z̃_t² / Σ z̃²``.

    Useful for diagnosing whether first-stage identification is dominated by a
    handful of observations (for example, in the Ramey-Zubairy military-news
    series, 1941Q4 carries ~52% of the slack-regime variation, and excluding
    World War II leaves 1940Q2 carrying ~80-87%).

    Parameters
    ----------
    z : array-like or pd.Series, shape (T,)
        Excluded instrument series.
    controls : array-like or pd.DataFrame, shape (T, p), optional
        Controls to partial out (an intercept is always included).
    weights : array-like or pd.Series, shape (T,), optional
        Non-negative regime or sample weights ``F_t``. When provided, controls
        are partialled out within the weighted regime (i.e. regressing
        ``F_t * z_t`` on ``F_t * [1, controls]``).

    Returns
    -------
    pd.DataFrame
        Indexed like ``z`` (when a Series is passed) and sorted descending by
        ``"share"``, with columns ``["z_tilde", "z_tilde_sq", "share", "cum_share"]``.
    """
    idx = z.index if isinstance(z, pd.Series) else pd.RangeIndex(len(np.asarray(z)))
    z_arr = np.asarray(z, dtype=float).ravel()
    T = z_arr.shape[0]
    if controls is None:
        X = np.ones((T, 1))
    else:
        W = np.asarray(controls, dtype=float)
        if W.ndim == 1:
            W = W.reshape(-1, 1)
        if W.shape[0] != T:
            raise ValueError(
                f"instrument_influence: controls rows {W.shape[0]} != z length {T}"
            )
        X = np.column_stack([np.ones(T), W])
    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape[0] != T:
            raise ValueError(
                f"instrument_influence: weights length {w.shape[0]} != z length {T}"
            )
        z_eff = w * z_arr
        X_eff = w[:, None] * X
    else:
        z_eff = z_arr
        X_eff = X

    valid = np.isfinite(z_eff) & np.all(np.isfinite(X_eff), axis=1)
    z_tilde = np.full(T, np.nan)
    if valid.any():
        Xv = X_eff[valid]
        zv = z_eff[valid]
        X_inv = np.linalg.pinv(Xv.T @ Xv)
        z_tilde[valid] = zv - Xv @ (X_inv @ (Xv.T @ zv))
    z_sq = z_tilde ** 2
    total = float(np.nansum(z_sq))
    share = z_sq / total if total > 0 else np.zeros_like(z_sq)
    out = pd.DataFrame(
        {"z_tilde": z_tilde, "z_tilde_sq": z_sq, "share": share},
        index=idx,
    )
    out = out.dropna(subset=["z_tilde"]).sort_values("share", ascending=False)
    out["cum_share"] = out["share"].cumsum()
    return out


__all__ = [
    "anderson_rubin_band",
    "cragg_donald_f",
    "kleibergen_paap_f",
    "olea_pflueger_f",
    "anderson_rubin_test",
    "msw_bands",
    "instrument_influence",
    "ARTestResult",
]
