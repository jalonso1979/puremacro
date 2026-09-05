"""Barnichon-Brownlees (2019) Smooth Local Projections.

Estimates impulse response functions jointly across all horizons h = 0, ..., H
via penalized Generalized Least Squares (PGLS) or Penalized Least Squares (PLS)
using a cubic B-spline basis and roughness penalty matrix:

    min_θ ||Y - X θ||_2^2 + λ θ' P θ

where:
- X = B ⊗ w̃, with B the (H+1) x K B-spline basis evaluated at horizons 0, ..., H,
  and w̃ the residualized shock variable after partialling out controls Z via FWL.
- P = D_d' D_d is the d-th difference penalty matrix on spline coefficients θ.
- β̂(h) = B_h θ̂ gives the smoothed impulse response at horizon h.

Provides automated data-driven smoothing parameter selection (AIC, BIC, GCV, CV),
analytical sandwich HAC standard errors, and moving block bootstrap inference.

Reference:
    Barnichon, R., & Brownlees, C. (2019). Impulse Response Estimation by Smooth
    Local Projections. The Review of Economics and Statistics, 101(3), 522-530.
"""
from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.stats import norm

from ._results import LPResult


def _build_bspline_basis(
    horizons: np.ndarray,
    n_knots: int | None = None,
    degree: int = 3,
) -> tuple[np.ndarray, int]:
    """Construct a clamped B-spline basis matrix over the horizon grid.

    Parameters
    ----------
    horizons : np.ndarray
        1D array of evaluation horizons (e.g. 0, 1, ..., H).
    n_knots : int or None
        Number of internal knot locations. If None, chosen adaptively based on H.
    degree : int
        Spline polynomial degree (default 3 for cubic B-splines).

    Returns
    -------
    B : np.ndarray
        (H_num, n_basis) basis matrix evaluated at each horizon.
    n_basis : int
        Number of spline basis functions.
    """
    H_num = len(horizons)
    h_min = float(np.min(horizons))
    h_max = float(np.max(horizons))

    # Guard degree against very short horizon grids
    deg = min(int(degree), max(1, H_num - 1))

    if n_knots is None:
        n_knots = min(H_num, max(4, int(np.ceil(H_num / 3))))
    else:
        n_knots = max(2, int(n_knots))

    # Ensure total basis functions do not exceed horizon count
    if n_knots + deg - 1 > H_num:
        n_knots = max(2, H_num - deg + 1)

    knots_inner = np.linspace(h_min, h_max, n_knots)
    t_knots = np.r_[[knots_inner[0]] * deg, knots_inner, [knots_inner[-1]] * deg]
    n_basis = len(t_knots) - deg - 1

    B = np.zeros((H_num, n_basis), dtype=float)
    for j in range(n_basis):
        c = np.zeros(n_basis, dtype=float)
        c[j] = 1.0
        B[:, j] = BSpline(t_knots, c, deg, extrapolate=True)(horizons.astype(float))

    # Clean up numerical fuzz and enforce exact partition of unity
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = B.sum(axis=1, keepdims=True)
    mask = row_sums.ravel() > 1e-12
    if np.any(mask):
        B[mask, :] /= row_sums[mask]

    return B, n_basis


def _difference_penalty_matrix(n_basis: int, order: int = 2) -> np.ndarray:
    """Construct roughness difference penalty matrix P = D_d' D_d."""
    d = min(max(1, int(order)), max(1, n_basis - 1))
    D = np.diff(np.eye(n_basis, dtype=float), n=d, axis=0)
    return D.T @ D


def _prepare_lp_data(
    df: pd.DataFrame | np.ndarray,
    y: str | np.ndarray | None,
    x: str | np.ndarray | None,
    horizons: list[int],
    n_lags: int,
    controls: Sequence[str] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, int]:
    """Prepare aligned time-series arrays and project out controls via FWL.

    Returns
    -------
    w_tilde : np.ndarray (T_eff,)
        Residualized shock variable.
    Y_tilde : np.ndarray (T_eff, H_num)
        Residualized lead variables across all horizons.
    s_ww : float
        Sum of squared residualized shocks (w̃' w̃).
    b_ols : np.ndarray (H_num,)
        Unpenalized OLS LP estimates on common effective sample.
    u_ols : np.ndarray (T_eff, H_num)
        OLS residuals across horizons.
    T_eff : int
        Number of effective observations.
    """
    # Standardize input to DataFrame
    if not (isinstance(df, pd.DataFrame) and isinstance(y, str) and isinstance(x, str)):
        y_vals = np.asarray(df if y is None else y, dtype=float).ravel()
        x_vals = np.asarray(y if x is None else x, dtype=float).ravel()
        if len(y_vals) != len(x_vals):
            raise ValueError(
                f"Length mismatch: y has {len(y_vals)} rows, x has {len(x_vals)} rows."
            )
        data = {"y": y_vals, "x": x_vals}
        ctl_cols: list[str] = []
        if controls is not None:
            C = np.asarray(controls, dtype=float)
            if C.ndim == 1:
                data["c0"] = C
                ctl_cols.append("c0")
            elif C.ndim == 2:
                for j in range(C.shape[1]):
                    col_name = f"c{j}"
                    data[col_name] = C[:, j]
                    ctl_cols.append(col_name)
        df_work = pd.DataFrame(data)
        y_col = "y"
        x_col = "x"
    else:
        df_work = df.copy()
        y_col = y
        x_col = x
        ctl_cols = list(controls or [])

    # Construct lead and lag features on a common sample
    sub = pd.DataFrame(index=df_work.index)
    lead_cols: list[str] = []
    for h in horizons:
        cname = f"__lead_{h}__"
        sub[cname] = df_work[y_col].shift(-h)
        lead_cols.append(cname)

    sub["__x__"] = df_work[x_col]

    for lag in range(1, n_lags + 1):
        sub[f"__x_L{lag}__"] = df_work[x_col].shift(lag)
        sub[f"__y_L{lag}__"] = df_work[y_col].shift(lag)
        for c in ctl_cols:
            sub[f"__{c}_L{lag}__"] = df_work[c].shift(lag)

    for c in ctl_cols:
        sub[f"__{c}__"] = df_work[c]

    # Common balanced sample across all horizons and lags
    sub = sub.dropna()
    T_eff = len(sub)
    if T_eff < max(10, n_lags + 2):
        raise ValueError(
            f"Insufficient effective observations ({T_eff}) for smooth LP estimation. "
            f"Reduce horizons or n_lags."
        )

    w = sub["__x__"].to_numpy(dtype=float)
    Y_leads = sub[lead_cols].to_numpy(dtype=float)

    # Regressors in Z (constant + lags + controls)
    regressors = [np.ones(T_eff, dtype=float)]
    for lag in range(1, n_lags + 1):
        regressors.append(sub[f"__x_L{lag}__"].to_numpy(dtype=float))
        regressors.append(sub[f"__y_L{lag}__"].to_numpy(dtype=float))
        for c in ctl_cols:
            regressors.append(sub[f"__{c}_L{lag}__"].to_numpy(dtype=float))
    for c in ctl_cols:
        regressors.append(sub[f"__{c}__"].to_numpy(dtype=float))

    Z = np.column_stack(regressors)

    # Frisch-Waugh-Lovell projection via robust least squares
    coef_w, _, _, _ = np.linalg.lstsq(Z, w, rcond=1e-12)
    w_tilde = w - Z @ coef_w

    coef_Y, _, _, _ = np.linalg.lstsq(Z, Y_leads, rcond=1e-12)
    Y_tilde = Y_leads - Z @ coef_Y

    s_ww = float(np.sum(w_tilde**2))
    if s_ww < 1e-12:
        raise ValueError("Shock variable x has near-zero variance after partialling out controls.")

    # Unpenalized OLS LP on the common sample
    b_ols = (w_tilde @ Y_tilde) / s_ww
    u_ols = Y_tilde - w_tilde[:, None] * b_ols[None, :]

    return w_tilde, Y_tilde, s_ww, b_ols, u_ols, T_eff


def _select_lambda(
    BtB: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    b_ols: np.ndarray,
    Y_tilde: np.ndarray,
    w_tilde: np.ndarray,
    s_ww: float,
    u_ols: np.ndarray,
    T_eff: int,
    H_num: int,
    selection: str = "aic",
    W_inv: np.ndarray | None = None,
) -> tuple[float, float]:
    """Automated data-driven lambda selection via information criteria or CV."""
    grid = np.logspace(-5, 5, 50)
    n_basis = B.shape[1]
    N = H_num * T_eff
    rss_ols_total = float(np.sum(u_ols**2))

    lhs_matrix = BtB if W_inv is None else B.T @ W_inv @ B
    rhs_vector = B.T @ b_ols if W_inv is None else B.T @ W_inv @ b_ols

    scores: list[float] = []

    if selection == "cv":
        k_folds = min(5, max(2, T_eff // 10))
        fold_indices = np.array_split(np.arange(T_eff), k_folds)

        for lam_val in grid:
            cv_err = 0.0
            for fold in fold_indices:
                train_idx = np.setdiff1d(np.arange(T_eff), fold)
                w_tr = w_tilde[train_idx]
                s_tr = float(np.sum(w_tr**2))
                if s_tr < 1e-12:
                    continue
                Y_tr = Y_tilde[train_idx, :]
                b_tr = (w_tr @ Y_tr) / s_tr

                reg_mat = lhs_matrix + lam_val * P
                try:
                    theta_tr = np.linalg.solve(reg_mat, B.T @ b_tr)
                    beta_tr = B @ theta_tr
                    w_te = w_tilde[fold]
                    Y_te = Y_tilde[fold, :]
                    pred_err = Y_te - w_te[:, None] * beta_tr[None, :]
                    cv_err += float(np.sum(pred_err**2))
                except np.linalg.LinAlgError:
                    cv_err += np.inf
            scores.append(cv_err)
    else:
        for lam_val in grid:
            reg_mat = lhs_matrix + lam_val * P
            try:
                inv_M = np.linalg.solve(reg_mat, np.eye(n_basis))
                theta = inv_M @ rhs_vector
                fit_beta = B @ theta

                # RSS of stacked joint system
                rss = rss_ols_total + s_ww * float(np.sum((b_ols - fit_beta)**2))
                df_lam = float(np.trace(inv_M @ lhs_matrix))

                if selection == "aic":
                    score = float(np.log(max(1e-12, rss / N)) + 2.0 * df_lam / N)
                elif selection == "bic":
                    score = float(np.log(max(1e-12, rss / N)) + np.log(N) * df_lam / N)
                elif selection == "gcv":
                    denom = max(1e-10, (1.0 - df_lam / N) ** 2)
                    score = float((rss / N) / denom)
                else:
                    raise ValueError(
                        f"Unknown selection criterion '{selection}'. "
                        f"Choose from 'aic', 'bic', 'gcv', 'cv'."
                    )
                scores.append(score)
            except np.linalg.LinAlgError:
                scores.append(np.inf)

    best_idx = int(np.argmin(scores))
    best_lam = float(grid[best_idx])

    # Final degrees of freedom
    try:
        inv_best = np.linalg.solve(lhs_matrix + best_lam * P, np.eye(n_basis))
        final_df = float(np.trace(inv_best @ lhs_matrix))
    except np.linalg.LinAlgError:
        final_df = float(n_basis)

    return best_lam, final_df


def _compute_analytical_hac(
    B: np.ndarray,
    BtB: np.ndarray,
    P: np.ndarray,
    lam: float,
    beta_smooth: np.ndarray,
    w_tilde: np.ndarray,
    Y_tilde: np.ndarray,
    s_ww: float,
    horizons: list[int],
    hac_lags: int | None = None,
    W_inv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute analytical sandwich HAC covariance matrix and standard errors."""
    T_eff, H_num = Y_tilde.shape
    n_basis = B.shape[1]

    # Stacked residuals at estimated smooth IRF
    u_mat = Y_tilde - w_tilde[:, None] * beta_smooth[None, :]

    # Score vector per time period t: s_t = w̃_t (B' u_t)
    if W_inv is None:
        S = (u_mat @ B) * w_tilde[:, None]
        lhs_matrix = BtB
    else:
        S = (u_mat @ W_inv @ B) * w_tilde[:, None]
        lhs_matrix = B.T @ W_inv @ B

    # Newey-West Bartlett kernel HAC estimator
    L = hac_lags if hac_lags is not None else max(int(max(horizons)), 1)
    L = min(L, max(1, T_eff - 2))

    Gamma_0 = S.T @ S
    S_hac = Gamma_0.copy()
    for l in range(1, L + 1):
        weight = 1.0 - l / (L + 1.0)
        Gamma_l = S[l:].T @ S[:-l]
        S_hac += weight * (Gamma_l + Gamma_l.T)

    inv_M = np.linalg.solve(lhs_matrix + lam * P, np.eye(n_basis))
    V_theta = inv_M @ (S_hac / (s_ww**2)) @ inv_M
    V_beta = B @ V_theta @ B.T

    se = np.sqrt(np.maximum(0.0, np.diag(V_beta)))
    return se, V_beta, V_theta


def _compute_bootstrap_ci(
    B: np.ndarray,
    BtB: np.ndarray,
    P: np.ndarray,
    lam: float,
    w_tilde: np.ndarray,
    Y_tilde: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 500,
    seed: int | np.random.Generator | None = None,
    W_inv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute moving block bootstrap confidence intervals."""
    T_eff, H_num = Y_tilde.shape
    n_basis = B.shape[1]
    lhs_matrix = BtB if W_inv is None else B.T @ W_inv @ B

    rng = np.random.default_rng(seed)
    block_len = max(2, int(np.ceil(T_eff ** (1.0 / 3.0))))
    n_blocks = int(np.ceil(T_eff / block_len))

    beta_boot = np.zeros((n_boot, H_num), dtype=float)
    theta_boot = np.zeros((n_boot, n_basis), dtype=float)

    reg_mat = lhs_matrix + lam * P

    for m in range(n_boot):
        starts = rng.integers(0, max(1, T_eff - block_len + 1), size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_len) for s in starts])[:T_eff]

        w_b = w_tilde[indices]
        s_ww_b = float(np.sum(w_b**2))
        if s_ww_b < 1e-12:
            s_ww_b = 1e-12

        Y_b = Y_tilde[indices, :]
        b_ols_b = (w_b @ Y_b) / s_ww_b
        rhs_b = B.T @ b_ols_b if W_inv is None else B.T @ W_inv @ b_ols_b

        try:
            th_b = np.linalg.solve(reg_mat, rhs_b)
        except np.linalg.LinAlgError:
            th_b = np.linalg.lstsq(reg_mat, rhs_b, rcond=1e-10)[0]

        theta_boot[m, :] = th_b
        beta_boot[m, :] = B @ th_b

    se_boot = np.std(beta_boot, axis=0)
    lo = np.percentile(beta_boot, 100.0 * (alpha / 2.0), axis=0)
    hi = np.percentile(beta_boot, 100.0 * (1.0 - alpha / 2.0), axis=0)
    V_beta = np.cov(beta_boot, rowvar=False)
    V_theta = np.cov(theta_boot, rowvar=False)

    return se_boot, lo, hi, V_beta, V_theta


def smooth_lp(
    df: pd.DataFrame | np.ndarray,
    y: str | np.ndarray | None = None,
    x: str | np.ndarray | None = None,
    horizons: int | Iterable[int] = 20,
    n_lags: int = 4,
    controls: Sequence[str] | np.ndarray | None = None,
    n_knots: int | None = None,
    degree: int = 3,
    penalty_order: int = 2,
    lam: float | str | None = "auto",
    selection: str = "aic",
    alpha: float = 0.05,
    ci_type: str = "analytic",
    *,
    lambda_: float | None = None,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
    n_boot: int = 500,
    seed: int | np.random.Generator | None = None,
    hac_lags: int | None = None,
    gls: bool = False,
) -> LPResult:
    """Estimate smooth local projections per Barnichon & Brownlees (2019).

    Parameters
    ----------
    df : pd.DataFrame or np.ndarray
        Input dataset containing target variable, shock, and optional controls.
    y : str or np.ndarray
        Name of the dependent response variable, or 1D array.
    x : str or np.ndarray
        Name of the shock / regressor of interest, or 1D array.
    horizons : int or Iterable[int], default 20
        Max horizon (if int) or explicit iterable of horizons (e.g. range(0, 21)).
    n_lags : int, default 4
        Number of autoregressive lags of y, x, and controls.
    controls : Sequence[str], np.ndarray, or None, default None
        Optional additional exogenous control variables.
    n_knots : int or None, default None
        Number of internal spline knots. If None, chosen adaptively based on horizon.
    degree : int, default 3
        B-spline polynomial degree (default 3 for cubic B-spline).
    penalty_order : int, default 2
        Difference order of roughness penalty matrix P = D_d' D_d.
    lam : float, str, or None, default "auto"
        Smoothing parameter λ. If "auto" or None, automatically selected via
        the criterion specified by `selection`. If float, uses fixed λ.
    selection : str, default "aic"
        Criterion for data-driven λ selection: 'aic', 'bic', 'gcv', or 'cv'.
    alpha : float, default 0.05
        Significance level for confidence bands (default 0.05 for 95% coverage).
    ci_type : str, default "analytic"
        Confidence interval method: 'analytic' (sandwich HAC) or 'bootstrap' (block bootstrap).
    lambda_ : float or None, optional
        Backward-compatibility alias for `lam`.
    lags : int or None, optional
        Backward-compatibility alias for `n_lags`.
    horizon : int or None, optional
        Backward-compatibility alias for `horizons`.
    ci : float or None, optional
        Backward-compatibility confidence level (e.g. 0.90 sets alpha = 0.10).
    n_boot : int, default 500
        Number of bootstrap replications when ci_type='bootstrap'.
    seed : int, Generator, or None, optional
        Random seed for bootstrap replication reproducibility.
    hac_lags : int or None, optional
        Bandwidth lag order for Newey-West Bartlett kernel HAC.
    gls : bool, default False
        If True, applies cross-horizon GLS weighting matrix Ω^{-1}.

    Returns
    -------
    LPResult
        Subclass of :class:`pandas.DataFrame` with columns:
        ['h', 'beta', 'se', 'lo', 'hi', 'lambda', 't'].
        Provides `.summary()`, `.plot()`, `.to_markdown()`, `.to_latex()`, `.to_typst()`.
    """
    # Parameter normalization & backward compatibility
    if lags is not None:
        n_lags = lags
    if horizon is not None:
        horizons = range(0, horizon + 1)
    if ci is not None:
        alpha = 1.0 - ci
    if lambda_ is not None:
        lam = lambda_

    if isinstance(horizons, (int, np.integer)):
        horizons_list = list(range(0, int(horizons) + 1))
    else:
        horizons_list = sorted(list(horizons))

    H_num = len(horizons_list)
    h_arr = np.array(horizons_list, dtype=float)

    # 1. Prepare data and partial out controls via FWL
    w_tilde, Y_tilde, s_ww, b_ols, u_ols, T_eff = _prepare_lp_data(
        df=df,
        y=y,
        x=x,
        horizons=horizons_list,
        n_lags=n_lags,
        controls=controls,
    )

    # 2. Build B-spline basis and penalty matrix
    B, n_basis = _build_bspline_basis(h_arr, n_knots=n_knots, degree=degree)
    P = _difference_penalty_matrix(n_basis=n_basis, order=penalty_order)
    BtB = B.T @ B

    # Optional cross-horizon GLS weighting
    W_inv = None
    if gls:
        Omega = (u_ols.T @ u_ols) / T_eff
        # Ridge regularize covariance across horizons
        ridge = 1e-4 * float(np.trace(Omega)) / H_num
        Omega_reg = Omega + ridge * np.eye(H_num)
        try:
            W_inv = np.linalg.inv(Omega_reg)
        except np.linalg.LinAlgError:
            W_inv = np.linalg.pinv(Omega_reg)

    # 3. Data-driven lambda selection
    selection_clean = str(selection).lower().strip()
    if lam is None or lam == "auto":
        lam_val, df_lam = _select_lambda(
            BtB=BtB,
            P=P,
            B=B,
            b_ols=b_ols,
            Y_tilde=Y_tilde,
            w_tilde=w_tilde,
            s_ww=s_ww,
            u_ols=u_ols,
            T_eff=T_eff,
            H_num=H_num,
            selection=selection_clean,
            W_inv=W_inv,
        )
    else:
        lam_val = float(lam)
        lhs_matrix = BtB if W_inv is None else B.T @ W_inv @ B
        try:
            inv_M = np.linalg.solve(lhs_matrix + lam_val * P, np.eye(n_basis))
            df_lam = float(np.trace(inv_M @ lhs_matrix))
        except np.linalg.LinAlgError:
            df_lam = float(n_basis)

    # 4. Final point estimates
    lhs_matrix = BtB if W_inv is None else B.T @ W_inv @ B
    rhs_vector = B.T @ b_ols if W_inv is None else B.T @ W_inv @ b_ols

    reg_mat = lhs_matrix + lam_val * P
    try:
        theta = np.linalg.solve(reg_mat, rhs_vector)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(reg_mat, rhs_vector, rcond=1e-12)[0]

    beta_smooth = B @ theta

    # 5. Inference: Analytical sandwich HAC vs Block Bootstrap
    ci_clean = str(ci_type).lower().strip()
    if ci_clean in ("bootstrap", "boot"):
        se, lo, hi, V_beta, V_theta = _compute_bootstrap_ci(
            B=B,
            BtB=BtB,
            P=P,
            lam=lam_val,
            w_tilde=w_tilde,
            Y_tilde=Y_tilde,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
            W_inv=W_inv,
        )
    else:
        se, V_beta, V_theta = _compute_analytical_hac(
            B=B,
            BtB=BtB,
            P=P,
            lam=lam_val,
            beta_smooth=beta_smooth,
            w_tilde=w_tilde,
            Y_tilde=Y_tilde,
            s_ww=s_ww,
            horizons=horizons_list,
            hac_lags=hac_lags,
            W_inv=W_inv,
        )
        z_crit = float(norm.ppf(1.0 - alpha / 2.0))
        lo = beta_smooth - z_crit * se
        hi = beta_smooth + z_crit * se

    # 6. Format standard LPResult
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(se > 0, beta_smooth / se, np.nan)

    res = LPResult({
        "h": horizons_list,
        "beta": beta_smooth,
        "se": se,
        "lo": lo,
        "hi": hi,
        "lambda": [lam_val] * H_num,
        "t": t_stat,
    })
    res.index = res["h"]
    res.y_name = str(y if y is not None else "y")
    res.x_name = str(x if x is not None else "x")
    res.method = "LP-smooth"
    object.__setattr__(res, "optimal_lambda", lam_val)
    object.__setattr__(res, "df_lambda", df_lam)
    object.__setattr__(res, "selection_criterion", selection_clean)
    object.__setattr__(res, "ci_type", ci_clean)
    object.__setattr__(res, "theta", theta)
    object.__setattr__(res, "vcov", V_beta)
    object.__setattr__(res, "vcov_theta", V_theta)
    object.__setattr__(res, "B", B)
    object.__setattr__(res, "P", P)

    return res


# Backward compatibility alias
lp_smooth = smooth_lp

__all__ = ["smooth_lp", "lp_smooth"]
