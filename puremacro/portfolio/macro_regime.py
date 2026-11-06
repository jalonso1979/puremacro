"""Macroeconomic regime quadrant classification, All-Weather allocation, and factor attribution.

Implements:
- Dalio / Bridgewater 4-quadrant macro classification (``"goldilocks"``, ``"reflation"``,
  ``"stagflation"``, ``"deflationary_bust"``).
- Regime-conditional All-Weather risk-parity portfolio construction.
- Multi-factor OLS return attribution with Newey-West (1987) HAC standard errors,
  annualized Jensen's alpha, Information Ratio, and rolling factor loadings.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from puremacro.portfolio._results import (
    FactorAttributionResult,
    MacroRegimePortfolioResult,
)
from puremacro.portfolio.allocation import (
    _ensure_psd,
    _extract_returns_and_names,
    _solve_risk_budgeting,
    shrink_covariance,
)

__all__ = [
    "QUADRANTS",
    "classify_macro_quadrants",
    "all_weather_portfolio",
    "factor_attribution",
]

QUADRANTS: tuple[str, str, str, str] = (
    "goldilocks",
    "reflation",
    "stagflation",
    "deflationary_bust",
)


def _hp_cycle(x: np.ndarray, lam: float = 1600.0) -> np.ndarray:
    """Compute Hodrick-Prescott cyclical component ``x - trend``."""
    T = len(x)
    if T < 4:
        return x - float(np.mean(x))
    eye = np.eye(T, dtype=float)
    D2 = np.zeros((T - 2, T), dtype=float)
    idx = np.arange(T - 2)
    D2[idx, idx] = 1.0
    D2[idx, idx + 1] = -2.0
    D2[idx, idx + 2] = 1.0
    trend = np.linalg.solve(eye + float(lam) * (D2.T @ D2), x)
    return x - trend


def _impulse_signal(arr: np.ndarray, method: str) -> np.ndarray:
    """Transform a 1D macro series into a regime impulse signal around 0."""
    m = method.lower().strip()
    if m == "diff":
        out = np.empty_like(arr, dtype=float)
        out[1:] = np.diff(arr)
        out[0] = out[1] if len(arr) > 1 else 0.0
        return out
    if m == "zscore":
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1.0
        return (arr - mu) / (sd if sd > 1e-12 else 1.0)
    if m == "demean":
        return arr - float(np.mean(arr))
    if m == "hp":
        return _hp_cycle(arr)
    raise ValueError(
        f"Unknown classify_macro_quadrants method {method!r}. "
        "Expected one of ('diff', 'zscore', 'demean', 'hp')."
    )


def classify_macro_quadrants(
    growth: Sequence[float] | np.ndarray | pd.Series,
    inflation: Sequence[float] | np.ndarray | pd.Series,
    *,
    method: str = "diff",
) -> pd.Series:
    """Classify each period into one of the four macroeconomic growth-inflation quadrants.

    Quadrant definitions:
    - ``"goldilocks"``: growth impulse >= 0 and inflation impulse < 0
    - ``"reflation"``: growth impulse >= 0 and inflation impulse >= 0
    - ``"stagflation"``: growth impulse < 0 and inflation impulse >= 0
    - ``"deflationary_bust"``: growth impulse < 0 and inflation impulse < 0

    Parameters
    ----------
    growth : Sequence[float] | np.ndarray | pd.Series
        1D macroeconomic growth indicator (e.g., GDP growth, industrial production, PMI).
    inflation : Sequence[float] | np.ndarray | pd.Series
        1D inflation indicator (e.g., CPI YoY, core PCE, breakeven inflation).
    method : str, default ``"diff"``
        Signal extraction method: ``"diff"`` (first differences), ``"zscore"``
        (standardized z-scores), ``"demean"`` (deviation from sample mean), or
        ``"hp"`` (Hodrick-Prescott cycle).

    Returns
    -------
    pd.Series
        Series of quadrant string labels indexed like ``growth``/``inflation``.
    """
    idx = (
        growth.index
        if isinstance(growth, pd.Series)
        else (inflation.index if isinstance(inflation, pd.Series) else None)
    )
    g_arr = np.asarray(growth, dtype=float).ravel()
    pi_arr = np.asarray(inflation, dtype=float).ravel()
    if len(g_arr) != len(pi_arr):
        raise ValueError(
            f"growth length ({len(g_arr)}) and inflation length ({len(pi_arr)}) must match."
        )
    if len(g_arr) < 2:
        raise ValueError("growth and inflation must have at least 2 observations.")

    g_sig = _impulse_signal(g_arr, method)
    pi_sig = _impulse_signal(pi_arr, method)

    labels: list[str] = []
    for g_val, pi_val in zip(g_sig, pi_sig):
        if g_val >= 0.0 and pi_val < 0.0:
            labels.append("goldilocks")
        elif g_val >= 0.0 and pi_val >= 0.0:
            labels.append("reflation")
        elif g_val < 0.0 and pi_val >= 0.0:
            labels.append("stagflation")
        else:
            labels.append("deflationary_bust")

    return pd.Series(labels, index=idx, name="macro_quadrant", dtype="object")


def all_weather_portfolio(
    returns: np.ndarray | pd.DataFrame,
    growth: Sequence[float] | np.ndarray | pd.Series,
    inflation: Sequence[float] | np.ndarray | pd.Series,
    *,
    quadrant_probs: Mapping[str, float] | None = None,
    classify_method: str = "diff",
    periods_per_year: int = 12,
    asset_names: Sequence[str] | None = None,
) -> MacroRegimePortfolioResult:
    """Construct a macro-regime All-Weather portfolio balanced across all four quadrants.

    Estimates regime-conditional expected returns, volatilities, and covariances in
    each of the four growth-inflation quadrants, constructs a risk-budgeted sub-portfolio
    for each quadrant tilted toward assets with stronger risk-adjusted returns in that
    regime, and combines the four sub-portfolios under equal (or custom) quadrant risk budgets.

    Parameters
    ----------
    returns : np.ndarray | pd.DataFrame
        Periodic asset return matrix of shape ``(T, N)``.
    growth : Sequence[float] | np.ndarray | pd.Series
        Macroeconomic growth series of length ``T``.
    inflation : Sequence[float] | np.ndarray | pd.Series
        Macroeconomic inflation series of length ``T``.
    quadrant_probs : Mapping[str, float] | None, optional
        Optional target risk budget / probability weights for the four quadrants
        (defaults to 0.25 for each quadrant in :data:`QUADRANTS`).
    classify_method : str, default ``"diff"``
        Method passed to :func:`classify_macro_quadrants`.
    periods_per_year : int, default 12
        Number of observations per year.
    asset_names : Sequence[str] | None, optional
        Asset names.

    Returns
    -------
    MacroRegimePortfolioResult
    """
    ret_arr, names = _extract_returns_and_names(returns, asset_names)
    if ret_arr.ndim != 2:
        raise ValueError(f"returns must be 2D of shape (T, N); got {ret_arr.shape}")
    T, N = ret_arr.shape

    regime_series = classify_macro_quadrants(growth, inflation, method=classify_method)
    if len(regime_series) != T:
        raise ValueError(
            f"returns length ({T}) must match growth/inflation length ({len(regime_series)})."
        )

    label_arr = regime_series.to_numpy()
    regime_shares: dict[str, float] = {
        q: float(np.mean(label_arr == q)) for q in QUADRANTS
    }

    full_mu = np.mean(ret_arr, axis=0) * float(periods_per_year)
    full_cov = shrink_covariance(ret_arr, method="ledoit_wolf") * float(periods_per_year)
    full_vols = np.sqrt(np.maximum(np.diag(full_cov), 1e-12))

    quad_ret_rows: dict[str, np.ndarray] = {}
    quad_vol_rows: dict[str, np.ndarray] = {}
    quadrant_weights: dict[str, np.ndarray] = {}

    for q in QUADRANTS:
        mask = label_arr == q
        n_q = int(np.sum(mask))
        if n_q >= 3:
            sub_ret = ret_arr[mask, :]
            mu_q = np.mean(sub_ret, axis=0) * float(periods_per_year)
            cov_q = shrink_covariance(sub_ret, method="ledoit_wolf") * float(periods_per_year)
            vol_q = np.sqrt(np.maximum(np.diag(cov_q), 1e-12))
        elif n_q >= 1:
            sub_ret = ret_arr[mask, :]
            mu_q = np.mean(sub_ret, axis=0) * float(periods_per_year)
            cov_q = full_cov
            vol_q = full_vols
        else:
            mu_q = full_mu
            cov_q = full_cov
            vol_q = full_vols

        quad_ret_rows[q] = mu_q
        quad_vol_rows[q] = vol_q

        # Risk budgets tilted toward assets with higher regime Sharpe ratio
        sharpe_q = mu_q / np.maximum(vol_q, 1e-6)
        sharpe_z = sharpe_q - float(np.mean(sharpe_q))
        tilt_budgets = np.exp(np.clip(0.5 * sharpe_z, -1.5, 1.5))
        tilt_budgets = tilt_budgets / float(np.sum(tilt_budgets))

        w_q = _solve_risk_budgeting(cov_q, budgets=tilt_budgets)
        quadrant_weights[q] = w_q

    if quadrant_probs is None:
        q_budgets = {q: 0.25 for q in QUADRANTS}
    else:
        raw_probs = {q: max(float(quadrant_probs.get(q, 0.0)), 0.0) for q in QUADRANTS}
        total_p = sum(raw_probs.values())
        if total_p <= 0.0:
            raise ValueError("quadrant_probs must have positive total weight.")
        q_budgets = {q: raw_probs[q] / total_p for q in QUADRANTS}

    # Combine quadrant sub-portfolios scaled by inverse sub-portfolio volatility
    combined = np.zeros(N, dtype=float)
    for q in QUADRANTS:
        w_q = quadrant_weights[q]
        sub_vol = float(np.sqrt(max(float(w_q @ full_cov @ w_q), 1e-12)))
        combined += (q_budgets[q] / sub_vol) * w_q

    w_all_weather = combined / float(np.sum(combined))
    exp_ret = float(np.dot(w_all_weather, full_mu))
    port_vol = float(np.sqrt(max(float(w_all_weather @ full_cov @ w_all_weather), 1e-18)))
    sharpe = exp_ret / port_vol if port_vol > 1e-12 else 0.0

    quad_ret_df = pd.DataFrame.from_dict(quad_ret_rows, orient="index", columns=list(names))
    quad_vol_df = pd.DataFrame.from_dict(quad_vol_rows, orient="index", columns=list(names))
    quad_ret_df.index.name = "quadrant"
    quad_vol_df.index.name = "quadrant"

    return MacroRegimePortfolioResult(
        weights=w_all_weather,
        asset_names=names,
        quadrant_weights=quadrant_weights,
        quadrant_returns=quad_ret_df,
        quadrant_vols=quad_vol_df,
        regime_labels=regime_series,
        regime_shares=regime_shares,
        expected_return=exp_ret,
        volatility=port_vol,
        sharpe_ratio=sharpe,
        metadata={
            "classify_method": classify_method,
            "quadrant_budgets": q_budgets,
            "periods_per_year": periods_per_year,
        },
    )


def _newey_west_vcov(X: np.ndarray, resid: np.ndarray, hac_lags: int) -> np.ndarray:
    """Compute Newey-West (1987) Bartlett-kernel HAC covariance matrix of OLS estimates."""
    T, p = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    u = resid.ravel()
    xu = X * u[:, None]
    S = xu.T @ xu

    max_lag = min(max(int(hac_lags), 0), T - 2)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - float(lag) / float(max_lag + 1)
        gamma_l = xu[lag:, :].T @ xu[:-lag, :]
        S += weight * (gamma_l + gamma_l.T)

    dof_scale = float(T) / float(max(T - p, 1))
    vcov = dof_scale * (xtx_inv @ S @ xtx_inv)
    return _ensure_psd(vcov, floor=1e-24)


def factor_attribution(
    portfolio_returns: Sequence[float] | np.ndarray | pd.Series,
    factor_returns: np.ndarray | pd.DataFrame,
    *,
    rf: float = 0.0,
    hac_lags: int = 4,
    rolling_window: int | None = None,
    periods_per_year: int = 12,
    factor_names: Sequence[str] | None = None,
) -> FactorAttributionResult:
    """Run multi-factor OLS return attribution with Newey-West HAC standard errors.

    Estimates the factor model:
    ``r_{p,t} - r_{f,t} = alpha_periodic + sum_{k=1}^K beta_k * F_{k,t} + eps_t``
    and reports annualized ``alpha = alpha_periodic * periods_per_year``, Newey-West
    HAC standard errors, t-statistics, p-values, R-squared, annualized Tracking Error,
    Information Ratio, and additive annualized return contributions.

    Parameters
    ----------
    portfolio_returns : Sequence[float] | np.ndarray | pd.Series
        1D periodic portfolio returns of length ``T``.
    factor_returns : np.ndarray | pd.DataFrame
        2D periodic factor returns of shape ``(T, K)`` (or 1D of length ``T``).
    rf : float, default 0.0
        Annualized risk-free rate.
    hac_lags : int, default 4
        Maximum lag order for the Newey-West Bartlett HAC covariance estimator.
    rolling_window : int | None, optional
        If provided, computes rolling OLS alpha and factor betas over windows of
        size ``rolling_window``.
    periods_per_year : int, default 12
        Number of return observations per year.
    factor_names : Sequence[str] | None, optional
        Factor names.

    Returns
    -------
    FactorAttributionResult
    """
    y_raw = np.asarray(portfolio_returns, dtype=float).ravel()
    T = len(y_raw)

    if isinstance(factor_returns, pd.DataFrame):
        F_mat = factor_returns.to_numpy(dtype=float)
        f_names = (
            tuple(str(c) for c in factor_names)
            if factor_names is not None
            else tuple(str(c) for c in factor_returns.columns)
        )
        time_index = factor_returns.index
    else:
        F_mat = np.asarray(factor_returns, dtype=float)
        if F_mat.ndim == 1:
            F_mat = F_mat.reshape(-1, 1)
        K_cols = F_mat.shape[1]
        f_names = (
            tuple(str(c) for c in factor_names)
            if factor_names is not None
            else tuple(f"factor_{k}" for k in range(K_cols))
        )
        time_index = (
            portfolio_returns.index if isinstance(portfolio_returns, pd.Series) else None
        )

    if F_mat.shape[0] != T:
        raise ValueError(
            f"portfolio_returns length ({T}) must match factor_returns rows ({F_mat.shape[0]})."
        )
    K = F_mat.shape[1]
    if T <= K + 1:
        raise ValueError(f"Need T > K + 1 observations for factor regression; got T={T}, K={K}.")

    p_yr = float(periods_per_year)
    rf_periodic = float(rf) / p_yr
    y = y_raw - rf_periodic

    X = np.column_stack([np.ones(T, dtype=float), F_mat])
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ theta
    resid = y - y_hat

    vcov = _newey_west_vcov(X, resid, hac_lags=hac_lags)
    ses = np.sqrt(np.maximum(np.diag(vcov), 1e-24))
    tstats = np.where(ses > 1e-12, theta / ses, np.sign(theta) * 1e6)
    dof = max(T - K - 1, 1)
    pvals = 2.0 * student_t.sf(np.abs(tstats), df=dof)

    alpha_periodic = float(theta[0])
    alpha_ann = alpha_periodic * p_yr
    alpha_se_ann = float(ses[0]) * p_yr
    alpha_tstat = float(tstats[0])
    alpha_pvalue = float(pvals[0])

    betas = theta[1:].astype(float)
    beta_ses = ses[1:].astype(float)
    beta_tstats = tstats[1:].astype(float)
    beta_pvalues = pvals[1:].astype(float)

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 1.0
    r2 = float(np.clip(r2, 0.0, 1.0))
    adj_r2 = 1.0 - (1.0 - r2) * float(T - 1) / float(max(T - K - 1, 1))

    resid_std_periodic = float(np.sqrt(ss_res / float(max(T - K - 1, 1))))
    tracking_error = resid_std_periodic * np.sqrt(p_yr)
    info_ratio = alpha_ann / tracking_error if tracking_error > 1e-12 else 0.0

    factor_means_ann = np.mean(F_mat, axis=0) * p_yr
    return_contributions: dict[str, float] = {"alpha": alpha_ann}
    for k, fname in enumerate(f_names):
        return_contributions[fname] = float(betas[k] * factor_means_ann[k])

    rolling_df: pd.DataFrame | None = None
    if rolling_window is not None:
        win = int(rolling_window)
        if win < K + 2 or win > T:
            raise ValueError(
                f"rolling_window must be between {K + 2} and {T}; got {win}."
            )
        n_win = T - win + 1
        roll_mat = np.zeros((n_win, K + 1), dtype=float)
        for start in range(n_win):
            end = start + win
            th_w, _, _, _ = np.linalg.lstsq(X[start:end, :], y[start:end], rcond=None)
            roll_mat[start, 0] = th_w[0] * p_yr
            roll_mat[start, 1:] = th_w[1:]

        roll_cols = ["alpha", *f_names]
        if time_index is not None:
            rolling_df = pd.DataFrame(roll_mat, index=time_index[win - 1 :], columns=roll_cols)
        else:
            rolling_df = pd.DataFrame(
                roll_mat, index=np.arange(win - 1, T), columns=roll_cols
            )

    return FactorAttributionResult(
        alpha=alpha_ann,
        alpha_se=alpha_se_ann,
        alpha_tstat=alpha_tstat,
        alpha_pvalue=alpha_pvalue,
        betas=betas,
        beta_ses=beta_ses,
        beta_tstats=beta_tstats,
        beta_pvalues=beta_pvalues,
        factor_names=f_names,
        r_squared=r2,
        adj_r_squared=adj_r2,
        tracking_error=tracking_error,
        information_ratio=info_ratio,
        return_contributions=return_contributions,
        rolling_betas=rolling_df,
        metadata={
            "alpha_periodic": alpha_periodic,
            "alpha_se_periodic": float(ses[0]),
            "hac_lags": hac_lags,
            "rf": rf,
            "periods_per_year": periods_per_year,
            "n_obs": T,
        },
    )
