"""High-Dimensional Penalized Macroeconomic Forecasting (Elastic Net & Adaptive Lasso).

Implements penalized estimation for macroeconomic panels following
Zou (2006, *JASA*) and Hastie, Tibshirani & Wainwright (2015):
- Pure NumPy coordinate descent with soft-thresholding operator.
- Elastic Net mixing parameter alpha in [0, 1] (alpha=1 for Lasso, 0.5 for Elastic Net).
- Adaptive Lasso weighting w_j = 1 / |beta_OLS, j|^gamma for asymptotic oracle property.
- Optimal regularisation tuning via Bayesian Information Criterion (BIC) or AIC.
- Direct multi-horizon macroeconomic forecasting: y_{t+h|t} = beta_0 + X_t @ beta.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


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
        lines = [
            "Penalized Macro Forecasting (Elastic Net / Adaptive Lasso)",
            "=" * 72,
            f"Forecast Horizon (h)            : {self.horizon} periods",
            f"Point Forecast                  : {self.forecast:.4f}",
            f"Selected Predictors             : {k_nonzero} / {k_total} ({k_nonzero/k_total*100:.1f}% sparsity)",
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


def _prepare_data(
    X_panel: pd.DataFrame | np.ndarray,
    y_target: pd.Series | np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert input data to numpy arrays and extract feature names."""
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

    return X_mat, y_vec, feat_names


def _align_data(
    X_mat: np.ndarray,
    y_vec: np.ndarray,
    horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align for direct h-step ahead forecasting: y_{t+h} on X_t."""
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

    return X_train, y_train, X_latest


def _standardize_data(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize feature matrix to zero mean and unit variance."""
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    X_norm = (X_train - x_mean) / x_std
    return X_norm, x_mean, x_std


def _compute_adaptive_weights(X_norm: np.ndarray, y_train: np.ndarray, adaptive: bool) -> np.ndarray:
    """Compute Adaptive Lasso weights via Ridge Regression."""
    P = X_norm.shape[1]
    if adaptive:
        ridge_pen = 1e-2 * np.eye(P)
        beta_ridge = np.linalg.solve(X_norm.T @ X_norm + ridge_pen, X_norm.T @ (y_train - y_train.mean()))
        weights = 1.0 / (np.abs(beta_ridge) + 1e-3)
        weights = weights / np.median(weights) # Normalize scale
    else:
        weights = np.ones(P)
    return weights


def _construct_lambda_grid(
    X_norm: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    weights: np.ndarray,
    n_lambdas: int,
    lambda_min_ratio: float
) -> np.ndarray:
    """Construct a candidate grid of regularisation parameters."""
    T_eff = X_norm.shape[0]
    corrs = np.abs(X_norm.T @ (y_train - y_train.mean())) / (T_eff * np.maximum(alpha * weights, 1e-4))
    lambda_max = float(np.max(corrs))
    lambda_min = lambda_max * lambda_min_ratio
    lambda_grid = np.geomspace(lambda_max, lambda_min, n_lambdas)
    return lambda_grid


def _evaluate_bic_path(
    X_norm: np.ndarray,
    y_train: np.ndarray,
    lambda_grid: np.ndarray,
    alpha: float,
    weights: np.ndarray
) -> tuple[float, float, np.ndarray, dict]:
    """Evaluate BIC path across lambda grid to find optimal regularisation."""
    T_eff, P = X_norm.shape
    best_bic = np.inf
    best_lam = lambda_grid[0]
    best_b0 = 0.0
    best_b = np.zeros(P)
    bic_dict = {}

    for lam in lambda_grid:
        b0, b = _fit_coordinate_descent(X_norm, y_train, lam, alpha, weights)
        fitted = b0 + X_norm @ b
        mse = float(np.mean((y_train - fitted) ** 2))
        df_model = int(np.sum(np.abs(b) > 1e-5)) + 1
        # BIC formula: T * log(MSE) + df * log(T)
        bic = T_eff * np.log(max(1e-12, mse)) + df_model * np.log(T_eff)
        bic_dict[float(lam)] = bic

        if bic < best_bic:
            best_bic = bic
            best_lam = float(lam)
            best_b0 = b0
            best_b = b.copy()

    return best_lam, best_b0, best_b, bic_dict


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
    """Direct multi-horizon macroeconomic forecasting with Elastic Net or Adaptive Lasso.

    Parameters
    ----------
    X_panel : DataFrame or ndarray of shape (T, P)
        High-dimensional matrix of macroeconomic predictors at time t.
    y_target : Series or ndarray of shape (T,)
        Target variable to forecast (e.g. GDP growth, inflation).
    horizon : int, default 1
        Direct forecast horizon h. The model aligns y_{t+h} on X_t.
    alpha : float, default 0.5
        Elastic Net mixing parameter (1.0 = Lasso, 0.5 = Elastic Net, 0.0 = Ridge).
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
    X_mat, y_vec, feat_names = _prepare_data(X_panel, y_target)

    X_train, y_train, X_latest = _align_data(X_mat, y_vec, horizon)

    X_norm, x_mean, x_std = _standardize_data(X_train)

    weights = _compute_adaptive_weights(X_norm, y_train, adaptive)

    lambda_grid = _construct_lambda_grid(X_norm, y_train, alpha, weights, n_lambdas, lambda_min_ratio)

    best_lam, best_b0, best_b, bic_dict = _evaluate_bic_path(X_norm, y_train, lambda_grid, alpha, weights)

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
