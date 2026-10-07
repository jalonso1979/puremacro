"""Arellano-Bond (1991) difference GMM estimator.

Modern best-practice defaults are ON: two-step optimal weighting,
Windmeijer (2005) finite-sample SE correction, Roodman (2009)
instrument collapse. Returns a frozen :class:`GMMResult`.

References
----------
Arellano, M. and Bond, S. (1991). Some tests of specification for panel
    data. Review of Economic Studies 58(2), 277-297.
Windmeijer, F. (2005). A finite sample correction for the variance of
    linear efficient two-step GMM estimators. Journal of Econometrics
    126, 25-51.
Roodman, D. (2009). How to do xtabond2: an introduction to difference
    and system GMM in Stata. Stata Journal 9(1), 86-136.
"""
from __future__ import annotations

import numpy as np

from .._linalg import safe_cholesky
from ._results import GMMResult
from .diagnostics import ar_test, hansen_j, windmeijer_correction
from .instruments import build_instruments


def _inv_psd(A: np.ndarray, *, name: str) -> np.ndarray:
    """Inverse of a presumed-PSD matrix via puremacro._linalg.safe_cholesky.

    Routes through the diagnostic-error contract: a singular weight or
    bread matrix gets a named LinAlgError instead of garbage.
    """
    A_sym = 0.5 * (A + A.T)
    L = safe_cholesky(A_sym, name=name)
    eye = np.eye(L.shape[0])
    L_inv = np.linalg.solve(L, eye)
    return L_inv.T @ L_inv


# ---------------------------------------------------------------------
# Core GMM solver: given Z, X, y, W, return β̂ = (X'Z W Z'X)^{-1} X'Z W Z'y
# ---------------------------------------------------------------------


def _gmm_step(
    Z: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one GMM step: β̂ = (X'ZW Z'X)^{-1} X'ZW Z'y."""
    ZX = Z.T @ X  # (m, k)
    Zy = Z.T @ y  # (m,)
    XZW = ZX.T @ W  # (k, m)
    A = XZW @ ZX  # (k, k)
    b = XZW @ Zy  # (k,)
    A_inv = _inv_psd(A, name=name)
    return A_inv @ b, A_inv


# ---------------------------------------------------------------------
# Weight-matrix builders
# ---------------------------------------------------------------------


def _W_step1(Z: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Step-1 weight ``(Z' H Z)^{-1}``.

    ``H`` is the block-diagonal first-difference covariance matrix
    (Arellano-Bond 1991 eq. (8)).
    """
    M = Z.T @ H @ Z
    return _inv_psd(M, name="ab_gmm step1 weight Z'HZ")


def _W_step2(
    Z: np.ndarray,
    residuals: np.ndarray,
    diff_rows: list,
) -> np.ndarray:
    """Step-2 weight ``(Σ_i Z_i' u_i u_i' Z_i)^{-1}`` (panel-cluster).
    """
    m = Z.shape[1]
    S = np.zeros((m, m))
    # group rows by panel id
    panel_groups: dict[object, list[int]] = {}
    for r, drow in enumerate(diff_rows):
        panel_groups.setdefault(drow["panel"], []).append(r)
    for pid, rows in panel_groups.items():
        Z_i = Z[rows]
        u_i = residuals[rows]
        Zu = Z_i.T @ u_i  # (m,)
        S += np.outer(Zu, Zu)
    return _inv_psd(S, name="ab_gmm step2 weight Σ Z_i' u_i u_i' Z_i")


# ---------------------------------------------------------------------
# Sandwich-form covariance
# ---------------------------------------------------------------------


def _gmm_sandwich_cov(
    Z: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
    name: str,
) -> np.ndarray:
    """Two-step efficient covariance: ``(X'Z W Z'X)^{-1}``.

    Under optimal ``W = (Σ Z_i' u_i u_i' Z_i)^{-1}`` the sandwich
    collapses to this efficient form.
    """
    ZX = Z.T @ X
    A = ZX.T @ W @ ZX
    return _inv_psd(A, name=name)


def _dropped_instrument_notes(dropped: list) -> tuple:
    """Result notes for instrument columns pruned as identically zero."""
    if not dropped:
        return ()
    return (
        f"dropped {len(dropped)} instrument column(s) that are identically "
        "zero on the estimation sample (no moment condition; not counted in "
        "n_instruments or the Hansen J df): " + ", ".join(dropped),
    )


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def ab_gmm(
    y: np.ndarray,
    panel_id: np.ndarray,
    time_id: np.ndarray,
    *,
    lag_dep_var: int = 1,
    lags: int | None = None,
    X_endog: np.ndarray | None = None,
    X_pred: np.ndarray | None = None,
    X_exog: np.ndarray | None = None,
    gmm_lag_window: tuple = (2, None),
    collapse: bool = True,
    two_step: bool = True,
    windmeijer: bool = True,
    names: list | None = None,
    robust: bool = True,
) -> GMMResult:
    """Arellano-Bond (1991) difference GMM.

    Parameters
    ----------
    y : ndarray, shape (NT,)
        Outcome in long format.
    panel_id : ndarray, shape (NT,)
        Panel/unit identifier for each row.
    time_id : ndarray, shape (NT,)
        Integer time identifier for each row. Gaps are allowed.
    lag_dep_var : int, default 1
        Number of lags of ``y`` on the right-hand side.
    lags : int, optional
        Standardized alias for ``lag_dep_var``.
    X_endog : ndarray of shape (NT, k_e) or None, default None
        Endogenous regressors (use lag window ``(lo, hi)`` in
        ``gmm_lag_window``).
    X_pred : ndarray of shape (NT, k_p) or None, default None
        Predetermined regressors (use lag window ``(lo-1, hi-1)``).
    X_exog : ndarray of shape (NT, k_x) or None, default None
        Strictly exogenous regressors (their differences appear in
        ``Z`` directly).
    gmm_lag_window : tuple of (int, int or None), default ``(2, None)``
        Lag window for endogenous regressors. ``None`` upper bound
        means use all available lags.
    collapse : bool, default True
        Roodman (2009) instrument collapse — one column per lag rather
        than one per (lag, t). Recommended ON to control instrument
        proliferation. ``collapse=False`` is the Holtz-Eakin, Newey and
        Rosen (1988) / Arellano-Bond (1991) layout used by Stata
        ``xtabond``: one column per (lag, t). Columns that are identically
        zero on the estimation sample (e.g. the (lag 2, first t) column
        when ``lag_dep_var >= 2``) are dropped and listed in
        ``GMMResult.notes``; they carry no moment condition.
    two_step : bool, default True
        Two-step optimal-weighting GMM. Recommended ON.
    windmeijer : bool, default True
        Apply the Windmeijer (2005) finite-sample correction to the
        two-step covariance ("WC-robust", Stata ``twostep vce(robust)``).
        Only affects two-step. Recommended ON.
    names : list of str or None, default None
        Coefficient names. If None, autogenerated.
    robust : bool, default True
        One-step only (``two_step=False``). ``True`` gives the
        panel-clustered robust sandwich (Stata ``xtabond ..., vce(robust)``).
        ``False`` gives the homoskedastic one-step VCE of Stata
        ``xtabond`` without ``vce(robust)``, [XT] xtdpd Methods and
        formulas p.19: ``sigma2_1 * W_1^{-1}`` with
        ``W_1 = Q_xz (sum_i Z_i'H_i Z_i)^{-1} Q_xz'``, ``H_i`` having 1 on
        the diagonal and -0.5 on the first off-diagonals, and
        ``sigma2_1 = u1'u1 / (n_obs - k)`` from the one-step differenced
        residuals. Ignored when ``two_step=True``.

    Returns
    -------
    GMMResult

    Notes
    -----
    Stata correspondence (verified on ``abdata``, [XT] xtabond
    Examples 1, 2 and 4): with ``collapse=False``, ``lag_dep_var=p`` and
    the strictly exogenous regressors in ``X_exog``,
    ``two_step=False, robust=False`` is ``xtabond ..., lags(p) noconstant``,
    ``two_step=False`` is ``... vce(robust) noconstant`` and the defaults
    ``two_step=True, windmeijer=True`` are ``... twostep vce(robust)
    noconstant``.

    References
    ----------
    See module-level docstring.
    """
    if lags is not None:
        lag_dep_var = lags
    bundle = build_instruments(
        y,
        panel_id,
        time_id,
        lag_dep_var=lag_dep_var,
        X_endog=X_endog,
        X_pred=X_pred,
        X_exog=X_exog,
        gmm_lag_window=gmm_lag_window,
        collapse=collapse,
    )
    Z = bundle["Z"]
    X = bundle["X_diff"]
    y_d = bundle["y_diff"]
    H = bundle["H"]
    diff_rows = bundle["diff_rows"]
    auto_names = bundle["names"]
    notes = _dropped_instrument_notes(bundle.get("dropped_instr_labels", []))

    n_diff = len(diff_rows)
    n_panels = len({d["panel"] for d in diff_rows})
    k = X.shape[1]
    m = Z.shape[1]

    if n_diff < k + 1:
        raise ValueError(
            f"Only {n_diff} usable difference rows for {k} regressors. "
            "Panel is degenerate (too few time periods after differencing)."
        )

    # Step 1
    W1 = _W_step1(Z, H)
    beta1, _A1_inv = _gmm_step(Z, X, y_d, W1, name="ab_gmm step1")
    resid1 = y_d - X @ beta1

    if not two_step:
        # one-step robust covariance via panel cluster
        # Var = (X'ZW1Z'X)^{-1} X'ZW1 S W1Z'X (X'ZW1Z'X)^{-1}
        # with S = Σ Z_i' u_i u_i' Z_i (same panel cluster as W2)
        S = np.zeros((m, m))
        panel_groups: dict[object, list[int]] = {}
        for r, drow in enumerate(diff_rows):
            panel_groups.setdefault(drow["panel"], []).append(r)
        for pid, rows in panel_groups.items():
            Z_i = Z[rows]
            u_i = resid1[rows]
            Zu = Z_i.T @ u_i
            S += np.outer(Zu, Zu)
        bread_inv = _gmm_sandwich_cov(Z, X, W1, name="ab_gmm one-step bread")
        if robust:
            ZX = Z.T @ X
            meat = ZX.T @ W1 @ S @ W1 @ ZX
            cov = bread_inv @ meat @ bread_inv
        else:
            # Stata xtdpd p.19: sigma2_1 * W_1^{-1}, with Stata's H_i = H/2
            # (1 on the diagonal, -0.5 off it). Our H has 2 / -1, so
            # W_1(Stata) = 2 X'Z (Z'HZ)^{-1} Z'X and
            # V = sigma2_1 / 2 * (X'Z (Z'HZ)^{-1} Z'X)^{-1}.
            if n_diff <= k:
                raise ValueError(
                    "one-step homoskedastic VCE needs n_obs > k "
                    f"(n_obs={n_diff}, k={k})."
                )
            sigma2_1 = float(resid1 @ resid1) / (n_diff - k)
            cov = 0.5 * sigma2_1 * bread_inv
        cov = 0.5 * (cov + cov.T)
        beta = beta1
        resid = resid1
        W_for_J = W1
        step = 1
        windmeijer_used = False
    else:
        # Step 2
        W2 = _W_step2(Z, resid1, diff_rows)
        beta2, _ = _gmm_step(Z, X, y_d, W2, name="ab_gmm step2")
        resid2 = y_d - X @ beta2
        cov_uncorr = _gmm_sandwich_cov(Z, X, W2, name="ab_gmm step2 cov")

        if windmeijer:
            # one-step robust covariance V_1 used in the correction
            S1 = np.zeros((m, m))
            panel_groups = {}
            for r, drow in enumerate(diff_rows):
                panel_groups.setdefault(drow["panel"], []).append(r)
            for pid, rows in panel_groups.items():
                Z_i = Z[rows]
                u_i = resid1[rows]
                Zu_i = Z_i.T @ u_i
                S1 += np.outer(Zu_i, Zu_i)
            bread1 = _gmm_sandwich_cov(Z, X, W1, name="ab_gmm V1 bread")
            ZX_full = Z.T @ X
            meat1 = ZX_full.T @ W1 @ S1 @ W1 @ ZX_full
            V_1 = bread1 @ meat1 @ bread1
            V_1 = 0.5 * (V_1 + V_1.T)

            cov = windmeijer_correction(
                beta_hat=beta2,
                Z=Z,
                X_diff=X,
                y_diff=y_d,
                W2=W2,
                residuals_step2=resid2,
                cov_uncorrected=cov_uncorr,
                diff_rows=diff_rows,
                cov_step1=V_1,
                residuals_step1=resid1,
            )
            windmeijer_used = True
        else:
            cov = cov_uncorr
            windmeijer_used = False

        beta = beta2
        resid = resid2
        W_for_J = W2
        step = 2

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    # Hansen J (use the step's optimal weight if two-step; else step1
    # weight is suboptimal and J doesn't have its standard distribution
    # — we still report it for completeness)
    if step == 2:
        J, J_p, J_df = hansen_j(Z, resid, W_for_J, n_regressors=k)
    else:
        # one-step: rebuild the optimal weight from step-1 residuals
        # and report a Sargan-style J at that weight
        W2 = _W_step2(Z, resid1, diff_rows)
        J, J_p, J_df = hansen_j(Z, resid1, W2, n_regressors=k)

    # AR tests on the difference residuals
    ar1, ar1_p = ar_test(resid, diff_rows, lag=1)
    ar2, ar2_p = ar_test(resid, diff_rows, lag=2)

    # naming
    if names is not None:
        if len(names) != k:
            raise ValueError(
                f"names has length {len(names)} but there are {k} coefficients."
            )
        coef_names = tuple(names)
    else:
        coef_names = tuple(auto_names)

    converged = bool(np.all(np.isfinite(beta)) and np.all(np.isfinite(se)))

    return GMMResult(
        coefs=beta,
        se=se,
        cov=cov,
        names=coef_names,
        hansen_j=float(J),
        hansen_j_p=float(J_p),
        hansen_j_df=int(J_df),
        ar1_p=float(ar1_p),
        ar2_p=float(ar2_p),
        n_instruments=int(m),
        n_obs=int(n_diff),
        n_panels=int(n_panels),
        step=int(step),
        windmeijer=bool(windmeijer_used),
        estimator="ab",
        converged=converged,
        notes=notes,
    )


__all__ = ["ab_gmm"]
