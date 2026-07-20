"""Monthly distributed-lag estimator for climate-fertility analysis.

Two modes:
- Single-region (``region_col=None``): OLS with HC1 SE.
- Panel (``region_col`` set): OLS with cluster-robust SE by region.

Model:
    y_t = α + Σ_k Σ_s β_k^s · shock_s_{t-k}
          + month_FE (optional) + year_FE (optional)
          + region_FE (panel mode) + ε_t

The wide kwarg set lets country pipelines parameterise shock and
response columns without rewriting the estimator.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def make_dl_lags(
    df: pd.DataFrame,
    *,
    cols: Sequence[str],
    n_lags: int,
    sort_by: Sequence[str],
) -> pd.DataFrame:
    """Add ``{col}_lag1..{col}_lag{n_lags}`` columns to a copy of df.

    ``sort_by`` controls within-group shifting. If ``len(sort_by) >= 2``,
    the first element is treated as the group key and shifts are computed
    within each group; otherwise plain ``.shift`` is used. df is sorted
    by ``sort_by`` before shifts.

    Returns a copy of df with the lag columns added.
    """
    out = df.sort_values(list(sort_by)).copy()
    group_key = sort_by[0] if len(sort_by) >= 2 else None
    for col in cols:
        for k in range(1, n_lags + 1):
            if group_key is None:
                out[f"{col}_lag{k}"] = out[col].shift(k)
            else:
                out[f"{col}_lag{k}"] = out.groupby(group_key, observed=True)[col].shift(k)
    return out


def _hc1_sandwich(X: np.ndarray, residuals: np.ndarray, XtX_inv: np.ndarray) -> np.ndarray:
    """HC1 heteroskedasticity-robust covariance."""
    n, k = X.shape
    Xe = X * residuals[:, None]
    meat = Xe.T @ Xe
    return (n / max(n - k, 1)) * XtX_inv @ meat @ XtX_inv


def _design_matrix(
    df: pd.DataFrame,
    *,
    shock_cols: Sequence[str],
    n_lags: int,
    add_month_fe: bool,
    add_year_fe: bool,
    region_col: str | None,
    panel_fe: str,
    month_col: str,
    year_col: str,
) -> tuple[np.ndarray, dict[str, slice]]:
    """Build the regression design matrix and a mapping
    ``{shock_col: slice_into_beta}`` so the estimator can extract each
    shock's coefficient block.
    """
    n = len(df)
    blocks: list[np.ndarray] = [np.ones((n, 1))]  # intercept
    col_slices: dict[str, slice] = {}
    offset = 1  # start past intercept
    for shock in shock_cols:
        cols_for_shock = [shock] + [f"{shock}_lag{k}" for k in range(1, n_lags + 1)]
        block = df[cols_for_shock].to_numpy(dtype=float)
        blocks.append(block)
        col_slices[shock] = slice(offset, offset + block.shape[1])
        offset += block.shape[1]
    # Fixed effects
    if region_col is not None:
        if panel_fe == "region_month":
            rm_key = (
                df[region_col].astype(str) + "_" + df[month_col].astype(int).astype(str)
            )
            rm = pd.get_dummies(rm_key, drop_first=True, dtype=float).to_numpy()
            blocks.append(rm)
        elif panel_fe == "region":
            r = pd.get_dummies(df[region_col], drop_first=True, dtype=float).to_numpy()
            blocks.append(r)
        else:
            raise ValueError(
                f"monthly_dl: panel_fe must be 'region_month' or 'region', got {panel_fe!r}"
            )
        if add_month_fe and panel_fe == "region":
            md = pd.get_dummies(df[month_col].astype(int), prefix="m",
                                drop_first=True, dtype=float).to_numpy()
            blocks.append(md)
    else:
        if add_month_fe:
            md = pd.get_dummies(df[month_col].astype(int), prefix="m",
                                drop_first=True, dtype=float).to_numpy()
            blocks.append(md)
    if add_year_fe:
        yd = pd.get_dummies(df[year_col].astype(int), prefix="y",
                            drop_first=True, dtype=float).to_numpy()
        blocks.append(yd)
    X = np.hstack(blocks)
    return X, col_slices


def monthly_dl(
    df: pd.DataFrame,
    *,
    shock_cols: Sequence[str] = ("cdd", "hdd"),
    response_col: str = "log_births",
    n_lags: int = 12,
    add_month_fe: bool = True,
    add_year_fe: bool = True,
    region_col: str | None = None,
    panel_fe: str = "region_month",
    month_col: str = "calendar_month",
    year_col: str = "year",
) -> dict:
    """Estimate the distributed-lag model.

    Returns dict with per-shock coefficient arrays + SEs, R², n_obs,
    and a ``biological_benchmark`` field (sum of first shock's betas)
    for backward compatibility with the climate_fertility source.
    """
    # 1. Build lags WITHIN regions if panel; else globally.
    sort_by = [region_col, year_col, month_col] if region_col else [year_col, month_col]
    df_lagged = make_dl_lags(df, cols=list(shock_cols), n_lags=n_lags, sort_by=sort_by)
    needed_cols = [response_col, month_col, year_col]
    if region_col is not None:
        needed_cols.append(region_col)
    for shock in shock_cols:
        needed_cols.append(shock)
        for k in range(1, n_lags + 1):
            needed_cols.append(f"{shock}_lag{k}")
    data = df_lagged[needed_cols].dropna()
    if data.empty:
        raise ValueError("monthly_dl: no observations after dropna")
    if n_lags >= len(data):
        raise ValueError(
            f"monthly_dl: n_lags={n_lags} exceeds usable T={len(data)} after lag construction"
        )
    # 2. Design matrix.
    X, col_slices = _design_matrix(
        data, shock_cols=shock_cols, n_lags=n_lags,
        add_month_fe=add_month_fe, add_year_fe=add_year_fe,
        region_col=region_col, panel_fe=panel_fe,
        month_col=month_col, year_col=year_col,
    )
    y = data[response_col].to_numpy(dtype=float)
    n, k = X.shape
    # 3. OLS via pinv (rank-tolerant).
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    residuals = y - X @ beta
    # 4. SE: HC1 for single-region; cluster-by-region for panel.
    if region_col is None:
        V = _hc1_sandwich(X, residuals, XtX_inv)
    else:
        regions = data[region_col].to_numpy()
        unique_regions = np.unique(regions)
        G = len(unique_regions)
        meat = np.zeros((k, k))
        for region in unique_regions:
            mask = regions == region
            score_g = X[mask].T @ residuals[mask]
            meat += np.outer(score_g, score_g)
        correction = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
        V = correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    # 5. Compose return dict.
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    out: dict = {
        "r_squared": float(r_squared),
        "n_obs": int(n),
    }
    if region_col is not None:
        out["n_regions"] = int(len(np.unique(data[region_col])))
    first_shock_sum: float | None = None
    for shock in shock_cols:
        sl = col_slices[shock]
        out[f"{shock}_betas"] = beta[sl].tolist()
        out[f"{shock}_ses"] = se[sl].tolist()
        if first_shock_sum is None:
            first_shock_sum = float(sum(beta[sl]))
    out["biological_benchmark"] = float(first_shock_sum) if first_shock_sum is not None else 0.0
    return out


__all__ = ["monthly_dl", "make_dl_lags"]
