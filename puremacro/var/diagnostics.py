"""VAR diagnostics: Granger causality + block exogeneity (multivariate F-tests).

Granger causality: H_0 — variable `source` has zero coefficients on
all lags in the equation for variable `target`. The test is a Wald F
on those constraints.

Block exogeneity: H_0 — every variable in `source_block` has zero
coefficients on all lags in every equation in `target_block` (joint
restriction).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import f as _f_dist

from .estimate import estimate_var


def _build_var_design(Y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Return X = [const, y_{t-1}, ..., y_{t-p}] and Y_dep = Y[p:]."""
    T, n = Y.shape
    rows = []
    for t in range(p, T):
        row = [1.0]
        for k in range(1, p + 1):
            row.extend(Y[t - k])
        rows.append(row)
    return np.array(rows), Y[p:]


def granger_causality(
    Y: pd.DataFrame,
    source: str,
    target: str,
    p: int,
) -> dict:
    """F-test of H_0: variable `source` does not Granger-cause `target` in a VAR(p).

    Returns
    -------
    dict with keys: stat, p_value, df_num, df_den, restricted_rss, unrestricted_rss
    """
    Y_arr = Y.values.astype(float)
    cols = list(Y.columns)
    if source not in cols or target not in cols:
        raise KeyError(f"source/target not in Y: {source} / {target} vs {cols}")
    n = len(cols)
    target_idx = cols.index(target)
    source_idx = cols.index(source)

    X, Y_dep = _build_var_design(Y_arr, p)
    y_eq = Y_dep[:, target_idx]

    # Unrestricted: all lag coefficients on all variables
    beta_u, *_ = np.linalg.lstsq(X, y_eq, rcond=None)
    rss_u = float(np.sum((y_eq - X @ beta_u) ** 2))
    df_u = X.shape[0] - X.shape[1]

    # Restricted: drop columns corresponding to lags of `source` in equation for `target`
    drop_cols = [1 + k * n + source_idx for k in range(p)]  # const at col 0
    keep = [j for j in range(X.shape[1]) if j not in drop_cols]
    X_r = X[:, keep]
    beta_r, *_ = np.linalg.lstsq(X_r, y_eq, rcond=None)
    rss_r = float(np.sum((y_eq - X_r @ beta_r) ** 2))

    df_num = p
    df_den = df_u
    if rss_u <= 0 or df_den <= 0:
        return {"stat": np.nan, "p_value": np.nan, "df_num": df_num,
                "df_den": df_den, "restricted_rss": rss_r,
                "unrestricted_rss": rss_u}
    F = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
    p_value = _f_dist.sf(F, df_num, df_den)
    return {"stat": float(F), "p_value": float(p_value),
            "df_num": int(df_num), "df_den": int(df_den),
            "restricted_rss": rss_r, "unrestricted_rss": rss_u}


def block_exogeneity(
    Y: pd.DataFrame,
    source_block: Sequence[str],
    target_block: Sequence[str],
    p: int,
) -> dict:
    """F-test of H_0: every variable in `source_block` has zero lag coefficients
    in every equation in `target_block`.

    Joint test across (n_source × n_target × p) restrictions. Computed
    via SUR-equivalent stacked OLS.
    """
    Y_arr = Y.values.astype(float)
    cols = list(Y.columns)
    n = len(cols)
    src_idx = [cols.index(s) for s in source_block]
    tgt_idx = [cols.index(t) for t in target_block]
    n_src, n_tgt = len(src_idx), len(tgt_idx)

    X, Y_dep = _build_var_design(Y_arr, p)
    n_obs = X.shape[0]

    # Run unrestricted OLS equation-by-equation for the target block,
    # collecting residual sum of squares.
    rss_u_total = 0.0
    rss_r_total = 0.0
    keep_cols = [j for j in range(X.shape[1])
                 if j == 0 or (j - 1) % n not in src_idx]
    X_r = X[:, keep_cols]
    for eq in tgt_idx:
        y_eq = Y_dep[:, eq]
        beta_u, *_ = np.linalg.lstsq(X, y_eq, rcond=None)
        rss_u_total += float(np.sum((y_eq - X @ beta_u) ** 2))
        beta_r, *_ = np.linalg.lstsq(X_r, y_eq, rcond=None)
        rss_r_total += float(np.sum((y_eq - X_r @ beta_r) ** 2))

    # Degrees of freedom (system-wide)
    df_num = n_src * n_tgt * p
    df_den = n_tgt * n_obs - n_tgt * X.shape[1]
    if rss_u_total <= 0 or df_den <= 0:
        return {"stat": np.nan, "p_value": np.nan, "df_num": df_num,
                "df_den": df_den, "restricted_rss": rss_r_total,
                "unrestricted_rss": rss_u_total}
    F = ((rss_r_total - rss_u_total) / df_num) / (rss_u_total / df_den)
    p_value = _f_dist.sf(F, df_num, df_den)
    return {"stat": float(F), "p_value": float(p_value),
            "df_num": int(df_num), "df_den": int(df_den),
            "restricted_rss": rss_r_total, "unrestricted_rss": rss_u_total}


__all__ = ["granger_causality", "block_exogeneity"]
