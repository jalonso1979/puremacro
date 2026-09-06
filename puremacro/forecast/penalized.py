"""High-Dimensional Penalized Macroeconomic Forecasting (Elastic Net, Adaptive Lasso, Ridge).

Implements penalized estimation for macroeconomic panels following
Zou (2006, *JASA*) and Hastie, Tibshirani & Wainwright (2015):
- Pure NumPy coordinate descent with soft-thresholding operator (alpha > 0).
- Elastic Net mixing parameter alpha in [0, 1] (alpha=1 for Lasso, 0.5 for
  Elastic Net, 0 for Ridge — the ridge path is solved in closed form via
  the SVD and its BIC uses the hat-matrix trace as degrees of freedom).
- Adaptive Lasso weighting w_j = 1 / |beta_OLS, j|^gamma for asymptotic oracle property.
- Optimal regularisation tuning via Bayesian Information Criterion (BIC).
- Direct multi-horizon macroeconomic forecasting: y_{t+h|t} = beta_0 + X_t @ beta.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# For alpha == 0 there is no finite lambda that zeroes every coefficient, so
# the ridge grid is anchored on the spectrum of the (weighted) design:
# lambda_max = _RIDGE_LAMBDA_MAX_FACTOR * e_max, e_max the largest eigenvalue
# of X'X/T, where the effective degrees of freedom are already below P/100.
_RIDGE_LAMBDA_MAX_FACTOR = 10.0


@dataclass(frozen=True)
class PenalizedForecastResult:
    """Results from Penalized Macroeconomic Forecasting.

    Attributes
    ----------
    forecast : float
        Out-of-sample forecast value for y_{T+h}.
    selected_features : list of str
        Features with non-zero estimated coefficients.
    coefficients : pd.Series
        Estimated coefficients for all candidate predictors.
    intercept : float
        Estimated constant term.
    optimal_lambda : float
        Selected regularisation penalty parameter lambda.
    in_sample_r2 : float
        In-sample coefficient of determination R².
    bic_path : pd.Series
        BIC evaluation path across the candidate lambda grid.
    horizon : int
        Forecast horizon h.
    """
    forecast: float
    selected_features: list[str]
    coefficients: pd.Series
    intercept: float
    optimal_lambda: float
    in_sample_r2: float
    bic_path: pd.Series
    horizon: int

    def summary(self) -> str:
        k_nonzero = len(self.selected_features)
        k_total = len(self.coefficients)
        share = 100.0 * k_nonzero / k_total if k_total else float("nan")
        lines = [
            "Penalized Macro Forecasting (Elastic Net / Adaptive Lasso / Ridge)",
            "=" * 72,
            f"Forecast Horizon (h)            : {self.horizon} periods",
            f"Point Forecast                  : {self.forecast:.4f}",
            f"Selected Predictors             : {k_nonzero} / {k_total} ({share:.1f}% of candidates)",
            f"Optimal Penalty (λ)             : {self.optimal_lambda:.6f}",
            f"In-Sample R²                    : {self.in_sample_r2:.4f}",
            "-" * 72,
            "Top Selected Features & Coefficients:",
        ]
        top = self.coefficients[self.coefficients.abs() > 1e-5].sort_values(key=abs, ascending=False).head(8)
        if not top.empty:
            for feat, val in top.items():
                lines.append(f"  {feat:<30s}: {val:+8.4f}")
        else:
            lines.append("  (All candidate features penalized to zero)")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Coefficient table: every candidate with its (un-standardised)
        coefficient and whether it survived the penalty."""
        return pd.DataFrame(
            {
                "coef": np.round(self.coefficients.to_numpy(dtype=float), 6),
                "selected": [f in self.selected_features for f in self.coefficients.index],
            },
            index=pd.Index([str(i) for i in self.coefficients.index], name="feature"),
        )

    def to_markdown(self, **kwargs: Any) -> str:
        """Export the coefficient table to Markdown."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Export the coefficient table to a LaTeX ``tabular``."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Export the coefficient table to a Typst ``#table``."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), **kwargs)

    def plot(self, *, ax: Any = None, title: str | None = None) -> Any:
        """Horizontal bars of the selected coefficients (largest first);
        the BIC path when nothing was selected. Returns the Figure."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.5, 3.8))
        else:
            fig = ax.figure
        sel = self.coefficients[self.coefficients.abs() > 1e-5].sort_values(key=abs)
        if not sel.empty:
            ax.barh([str(i) for i in sel.index], sel.to_numpy(dtype=float), color="#1f77b4")
            ax.axvline(0.0, color="grey", lw=0.6)
            ax.set_xlabel("coefficient (original units)")
            ax.set_title(title or f"Selected predictors (h = {self.horizon}, forecast = {self.forecast:.3f})")
        else:
            ax.plot(self.bic_path.index.to_numpy(dtype=float), self.bic_path.to_numpy(dtype=float), lw=1.5)
            ax.set_xscale("log")
            ax.set_xlabel("lambda")
            ax.set_ylabel("BIC")
            ax.set_title(title or "BIC path (no predictor selected)")
        ax.grid(True, ls=":", alpha=0.5)
        return fig


def _soft_threshold(z: float, tau: float) -> float:
    """Soft-thresholding operator S(z, tau) = sign(z) * max(0, |z| - tau)."""
    if z > tau:
        return z - tau
    elif z < -tau:
        return z + tau
    return 0.0


def _fit_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    alpha: float,
    weights: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[float, np.ndarray]:
    """Pure NumPy Coordinate Descent for Elastic Net / Adaptive Lasso."""
    T, P = X.shape
    beta = np.zeros(P)
    beta_0 = float(np.mean(y))

    # Precompute X column squared norms
    x_sq = np.sum(X ** 2, axis=0) / T

    for it in range(max_iter):
        beta_old = beta.copy()

        # Residual without intercept
        r = y - beta_0 - X @ beta

        for j in range(P):
            if weights[j] >= 1e8:
                beta[j] = 0.0
                continue

            # Partial residual adding back feature j
            r_j = r + X[:, j] * beta[j]
            rho_j = float(np.dot(X[:, j], r_j) / T)

            # Thresholding
            tau_j = lam * alpha * weights[j]
            denom = x_sq[j] + lam * (1.0 - alpha) * weights[j]

            if denom > 1e-12:
                beta[j] = _soft_threshold(rho_j, tau_j) / denom
            else:
                beta[j] = 0.0

            # Update partial residual
            r = r_j - X[:, j] * beta[j]

        beta_0 = float(np.mean(y - X @ beta))

        if np.max(np.abs(beta - beta_old)) < tol:
            break

    return beta_0, beta


def _ridge_path(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    weights: np.ndarray,
) -> list[tuple[float, np.ndarray, float]]:
    """Closed-form weighted ridge path for the objective

        (1/2T) ||y - b0 - X b||^2 + (lam/2) sum_j w_j b_j^2.

    With X~ = X W^{-1/2} and its SVD U d V', the solution is
    b = W^{-1/2} V diag(d / (d^2 + T lam)) U' y_c and the effective degrees
    of freedom are tr(H) = sum_j d_j^2 / (d_j^2 + T lam) (+1 for the
    intercept). ``X`` must have zero-mean columns, so b0 = mean(y).

    Returns a list of ``(b0, b, df)`` per lambda.
    """
    T = X.shape[0]
    y_mean = float(np.mean(y))
    y_c = y - y_mean
    inv_sqrt_w = 1.0 / np.sqrt(np.maximum(weights, 1e-300))
    Xw = X * inv_sqrt_w[None, :]
    U, d, Vt = np.linalg.svd(Xw, full_matrices=False)
    Uty = U.T @ y_c
    out = []
    for lam in lambdas:
        shrink = d / (d ** 2 + T * float(lam))
        b = (Vt.T @ (shrink * Uty)) * inv_sqrt_w
        df = float(np.sum(d ** 2 / (d ** 2 + T * float(lam))))
        out.append((y_mean, b, df))
    return out


def forecast_penalized(
    X_panel: pd.DataFrame | np.ndarray,
    y_target: pd.Series | np.ndarray,
    *,
    horizon: int = 1,
    alpha: float = 0.5,
    adaptive: bool = True,
    n_lambdas: int = 40,
    lambda_min_ratio: float = 1e-3,
) -> PenalizedForecastResult:
    """Direct multi-horizon macroeconomic forecasting with Elastic Net, Adaptive Lasso or Ridge.

    Parameters
    ----------
    X_panel : DataFrame or ndarray of shape (T, P)
        High-dimensional matrix of macroeconomic predictors at time t.
    y_target : Series or ndarray of shape (T,)
        Target variable to forecast (e.g. GDP growth, inflation).
    horizon : int, default 1
        Direct forecast horizon h. The model aligns y_{t+h} on X_t.
    alpha : float, default 0.5
        Elastic Net mixing parameter in ``[0, 1]`` (1.0 = Lasso, 0.5 =
        Elastic Net, 0.0 = Ridge). Values outside ``[0, 1]`` raise
        ``ValueError``. For ``alpha > 0`` the lambda grid runs from the
        closed-form ``lambda_max`` (all coefficients zero) down to
        ``lambda_max * lambda_min_ratio`` and the BIC counts non-zero
        coefficients as degrees of freedom. For ``alpha == 0`` (ridge) no
        finite lambda zeroes the coefficients, so the grid is anchored on
        the largest eigenvalue ``e_max`` of the weighted ``X'X/T``
        (``lambda_max = 100 * e_max``, where the effective degrees of freedom
        are already below ``P/100``), the path is solved in closed form and
        the BIC uses the hat-matrix trace as degrees of freedom.
    adaptive : bool, default True
        If True, applies Adaptive Lasso weighting w_j = 1 / (|beta_Ridge, j| + 1e-3).
    n_lambdas : int, default 40
        Number of candidate regularisation parameters to evaluate on log-scale.
    lambda_min_ratio : float, default 1e-3
        Ratio of minimum to maximum lambda in search grid.

    Returns
    -------
    PenalizedForecastResult
    """
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0, 1] (1 = Lasso, 0 = Ridge); got {alpha!r}")
    if n_lambdas < 1:
        raise ValueError(f"n_lambdas must be >= 1; got {n_lambdas}")
    if not (0.0 < lambda_min_ratio <= 1.0):
        raise ValueError(f"lambda_min_ratio must be in (0, 1]; got {lambda_min_ratio!r}")
    alpha = float(alpha)

    if isinstance(X_panel, pd.DataFrame):
        feat_names = list(X_panel.columns)
        X_mat = X_panel.to_numpy(dtype=float)
    else:
        X_mat = np.asarray(X_panel, dtype=float)
        feat_names = [f"X_{j+1}" for j in range(X_mat.shape[1])]

    if isinstance(y_target, (pd.Series, pd.DataFrame)):
        y_vec = y_target.to_numpy(dtype=float).ravel()
    else:
        y_vec = np.asarray(y_target, dtype=float).ravel()

    T_full, P = X_mat.shape
    if len(y_vec) != T_full:
        raise ValueError(
            f"y_target has {len(y_vec)} rows but X_panel has {T_full}; alignment is positional"
        )

    # 1. Align for direct h-step ahead forecasting: y_{t+h} on X_t
    if horizon > 0:
        X_train = X_mat[:-horizon]
        y_train = y_vec[horizon:]
        X_latest = X_mat[-1]
    else:
        X_train = X_mat
        y_train = y_vec
        X_latest = X_mat[-1]

    T_eff = X_train.shape[0]
    if T_eff < 10:
        raise ValueError(f"forecast_penalized: effective training sample too small ({T_eff} rows)")

    # 2. Standardize X_train
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    X_norm = (X_train - x_mean) / x_std

    # 3. Compute Adaptive Weights (via Ridge Regression)
    if adaptive:
        ridge_pen = 1e-2 * np.eye(P)
        beta_ridge = np.linalg.solve(X_norm.T @ X_norm + ridge_pen, X_norm.T @ (y_train - y_train.mean()))
        weights = 1.0 / (np.abs(beta_ridge) + 1e-3)
        weights = weights / np.median(weights) # Normalize scale
    else:
        weights = np.ones(P)

    # 4. Construct Lambda Grid
    if alpha > 0.0:
        # lambda_max is the smallest penalty that zeroes out all coefficients
        corrs = np.abs(X_norm.T @ (y_train - y_train.mean())) / (T_eff * np.maximum(alpha * weights, 1e-12))
        lambda_max = float(np.max(corrs))
    else:
        # Ridge: anchor the grid on the spectrum of the weighted design.
        Xw = X_norm / np.sqrt(np.maximum(weights, 1e-300))[None, :]
        e_max = float(np.max(np.linalg.svd(Xw, compute_uv=False) ** 2)) / T_eff
        lambda_max = _RIDGE_LAMBDA_MAX_FACTOR * max(e_max, 1e-12)
    lambda_min = lambda_max * lambda_min_ratio
    lambda_grid = np.geomspace(lambda_max, lambda_min, n_lambdas)

    # 5. Evaluate BIC Path
    best_bic = np.inf
    best_lam = float(lambda_grid[0])
    best_b0 = 0.0
    best_b = np.zeros(P)
    bic_dict = {}

    if alpha > 0.0:
        fits = []
        for lam in lambda_grid:
            b0, b = _fit_coordinate_descent(X_norm, y_train, lam, alpha, weights)
            fits.append((b0, b, float(np.sum(np.abs(b) > 1e-5))))
    else:
        fits = _ridge_path(X_norm, y_train, lambda_grid, weights)

    for lam, (b0, b, df_coef) in zip(lambda_grid, fits):
        fitted = b0 + X_norm @ b
        mse = float(np.mean((y_train - fitted) ** 2))
        df_model = df_coef + 1.0
        # BIC formula: T * log(MSE) + df * log(T)
        bic = T_eff * np.log(max(1e-12, mse)) + df_model * np.log(T_eff)
        bic_dict[float(lam)] = bic

        if bic < best_bic:
            best_bic = bic
            best_lam = float(lam)
            best_b0 = b0
            best_b = b.copy()

    # 6. Unstandardize Coefficients
    beta_orig = best_b / x_std
    b0_orig = best_b0 - np.sum(beta_orig * x_mean)

    # 7. Compute Point Forecast
    y_forecast = float(b0_orig + np.dot(X_latest, beta_orig))

    # In-sample R²
    y_fit = b0_orig + X_train @ beta_orig
    ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
    ss_res = np.sum((y_train - y_fit) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12)) if ss_tot > 0 else 0.0

    s_coef = pd.Series(beta_orig, index=feat_names)
    selected = list(s_coef[s_coef.abs() > 1e-5].index)

    return PenalizedForecastResult(
        forecast=y_forecast,
        selected_features=selected,
        coefficients=s_coef,
        intercept=float(b0_orig),
        optimal_lambda=best_lam,
        in_sample_r2=max(0.0, min(1.0, float(r2))),
        bic_path=pd.Series(bic_dict),
        horizon=horizon,
    )


__all__ = [
    "PenalizedForecastResult",
    "forecast_penalized",
]
