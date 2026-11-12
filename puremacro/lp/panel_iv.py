"""Panel LP-IV: horizon-matched two-way fixed-effects 2SLS with entity-clustered SEs."""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .._linalg import inv_xtx
from ._common import (
    _DEFAULT_ALPHA_10,
    _DEFAULT_HORIZONS_21,
    _DEFAULT_N_LAGS_2,
    format_sample_bounds,
    resolve_lp_kwargs,
    validate_lp_columns_finite,
)
from ._panel_helpers import as_panel_index
from ._results import LPResult


def _demean_two_way(
    arr: np.ndarray,
    entity_idx: np.ndarray,
    time_idx: np.ndarray,
    *,
    tol: float = 1e-11,
    max_iter: int = 500,
) -> np.ndarray:
    """Iteratively remove entity and time means on an exact estimation sample."""
    out = arr.astype(float, copy=True)
    for _ in range(max_iter):
        ent_means = pd.DataFrame(out).groupby(entity_idx).transform("mean").to_numpy()
        out = out - ent_means
        t_means = pd.DataFrame(out).groupby(time_idx).transform("mean").to_numpy()
        out_new = out - t_means
        if np.max(np.abs(out_new - out)) < tol:
            return out_new
        out = out_new
    return out


def _cluster_meat(score: np.ndarray, entity_idx: np.ndarray) -> np.ndarray:
    """Entity-clustered score outer-product meat ``sum_c (sum_{t in c} s_{c,t})(sum_{t in c} s_{c,t})'``."""
    k = score.shape[1]
    meat = np.zeros((k, k), dtype=float)
    for c_key in pd.unique(entity_idx):
        s_c = score[entity_idx == c_key].sum(axis=0)
        meat += np.outer(s_c, s_c)
    return meat


def panel_lp_iv(
    df_wide: pd.DataFrame,
    y: str,
    x: str,
    z: str | Sequence[str],
    horizons: Iterable[int] = _DEFAULT_HORIZONS_21,
    n_lags: int = _DEFAULT_N_LAGS_2,
    controls: Sequence[str] = (),
    alpha: float = _DEFAULT_ALPHA_10,
    entity_level: str = "code",
    time_level: str = "date",
    *,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
    unit_col: str | None = None,
    time_col: str | None = None,
) -> LPResult:
    """Horizon-by-horizon two-way fixed-effects 2SLS panel local projection.

    For each horizon ``h``:
      1. Builds within-entity lags of ``(x, y, controls)`` and the horizon
         difference ``y_{i,t+h} - y_{i,t-1}``.
      2. Restricts to the non-missing sample at horizon ``h`` and applies
         two-way ``(entity, time)`` within-demeaning jointly to the outcome,
         endogenous regressor ``x``, excluded instrument(s) ``z``, and controls.
      3. Computes 2SLS point estimates, structural 2SLS residuals
         ``u_{i,t} = y_{w,i,t} - X_{actual,w,i,t} beta_2sls``, entity-clustered
         sandwich standard errors, and the entity-clustered excluded-instrument
         first-stage Wald ``F`` statistic.

    Returns :class:`LPResult` with columns ``[h, beta, se, t, lo, hi, first_stage_f]``.
    """
    horizons_list, n_lags, alpha = resolve_lp_kwargs(
        horizons,
        n_lags,
        alpha,
        lags=lags,
        horizon=horizon,
        ci=ci,
        name="panel_lp_iv",
        strict_aliases=True,
    )
    z_names = [z] if isinstance(z, str) else [str(col) for col in z]
    if not z_names:
        raise ValueError("panel_lp_iv: at least one instrument column is required in z.")
    ctl = [str(c) for c in controls]

    df_norm, entity_level, time_level = as_panel_index(
        df_wide,
        entity_level=entity_level,
        time_level=time_level,
        unit_col=unit_col,
        time_col=time_col,
        func="panel_lp_iv",
    )
    df = df_norm.copy()
    df.index = df.index.set_names([entity_level, time_level])
    if df.index.has_duplicates:
        raise ValueError(
            "panel_lp_iv: panel MultiIndex contains duplicate (entity, time) observations; "
            "deduplicate or aggregate before estimation."
        )
    validate_lp_columns_finite(df, [y, x, *z_names, *ctl], "panel_lp_iv")
    df = df.sort_index()

    grouped = df.groupby(level=entity_level)
    for lag in range(1, n_lags + 1):
        df[f"__{x}_L{lag}__"] = grouped[x].shift(lag)
        df[f"__{y}_L{lag}__"] = grouped[y].shift(lag)
        for c in ctl:
            df[f"__{c}_L{lag}__"] = grouped[c].shift(lag)

    exog_cols: list[str] = []
    for lag in range(1, n_lags + 1):
        exog_cols += [f"__{x}_L{lag}__", f"__{y}_L{lag}__"]
        for c in ctl:
            exog_cols.append(f"__{c}_L{lag}__")
    for c in ctl:
        exog_cols.append(c)

    z_crit = float(norm.ppf(1.0 - alpha / 2.0))
    k_z = len(z_names)
    rows: list[dict[str, Any]] = []
    n_obs_by_h: dict[int, int] = {}
    sample_start_by_h: dict[int, str | int | None] = {}
    sample_end_by_h: dict[int, str | int | None] = {}
    fs_f_by_h: dict[int, float] = {}

    for h in horizons_list:
        col_dy = f"__dy_h{h}__"
        df[col_dy] = grouped[y].shift(-h) - grouped[y].shift(1)
        unique_cols = list(dict.fromkeys([col_dy, x, *z_names, *exog_cols]))
        sub = df[unique_cols].dropna()
        n_obs = int(len(sub))
        n_obs_by_h[int(h)] = n_obs
        t_idx_index = sub.index.get_level_values(time_level)
        s_start, s_end = format_sample_bounds(t_idx_index)
        sample_start_by_h[int(h)] = s_start
        sample_end_by_h[int(h)] = s_end

        if n_obs <= 1 + len(exog_cols):
            fs_f_by_h[int(h)] = float("nan")
            rows.append(
                {
                    "h": int(h),
                    "beta": np.nan,
                    "se": np.nan,
                    "t": np.nan,
                    "lo": np.nan,
                    "hi": np.nan,
                    "first_stage_f": np.nan,
                }
            )
            continue

        entity_idx = sub.index.get_level_values(entity_level).to_numpy()
        time_idx = t_idx_index.to_numpy()
        dm = _demean_two_way(sub.to_numpy(dtype=float), entity_idx, time_idx)
        col_pos = {c: idx for idx, c in enumerate(unique_cols)}

        y_w = dm[:, col_pos[col_dy]]
        x_w = dm[:, col_pos[x]]
        Z_w = np.column_stack([dm[:, col_pos[zn]] for zn in z_names])
        W_w = (
            np.column_stack([dm[:, col_pos[ec]] for ec in exog_cols])
            if exog_cols
            else np.empty((n_obs, 0), dtype=float)
        )

        # First stage on within-transformed sample: x_w ~ [Z_w, W_w]
        M_fs = np.column_stack([Z_w, W_w])
        MfMf_inv = inv_xtx(M_fs, name=f"panel_lp_iv(first_stage, h={h})")
        pi_hat = MfMf_inv @ (M_fs.T @ x_w)
        x_hat_w = M_fs @ pi_hat
        v_fs = x_w - x_hat_w

        if float(np.max(np.abs(v_fs))) <= 1e-14 * max(1.0, float(np.std(x_w))):
            f_stat = float("inf")
        else:
            S_fs = _cluster_meat(M_fs * v_fs[:, None], entity_idx)
            vcov_fs = MfMf_inv @ S_fs @ MfMf_inv
            pi_z = pi_hat[:k_z]
            V_zz = vcov_fs[:k_z, :k_z]
            if k_z == 1:
                var_z = float(V_zz[0, 0])
                f_stat = float((pi_z[0] ** 2) / var_z) if var_z > 0.0 else float("nan")
            else:
                try:
                    V_zz_inv = np.linalg.pinv(V_zz)
                    f_stat = float((pi_z @ V_zz_inv @ pi_z) / k_z)
                except np.linalg.LinAlgError:
                    f_stat = float("nan")

        # Second stage 2SLS with structural residuals u_2sls = y_w - X_actual_w @ beta
        X_hat_w = np.column_stack([x_hat_w, W_w])
        X_actual_w = np.column_stack([x_w, W_w])
        XhXh_inv = inv_xtx(X_hat_w, name=f"panel_lp_iv(second_stage, h={h})")
        beta_vec = XhXh_inv @ (X_hat_w.T @ y_w)
        beta_h = float(beta_vec[0])
        u_2sls = y_w - X_actual_w @ beta_vec

        S_2sls = _cluster_meat(X_hat_w * u_2sls[:, None], entity_idx)
        vcov_2sls = XhXh_inv @ S_2sls @ XhXh_inv
        se_h = float(np.sqrt(max(0.0, vcov_2sls[0, 0])))
        t_h = beta_h / se_h if se_h > 0.0 else float("nan")

        fs_f_by_h[int(h)] = f_stat
        rows.append(
            {
                "h": int(h),
                "beta": beta_h,
                "se": se_h,
                "t": t_h,
                "lo": beta_h - z_crit * se_h,
                "hi": beta_h + z_crit * se_h,
                "first_stage_f": f_stat,
            }
        )

    res = LPResult(rows)
    if "h" in res.columns:
        res.index = res["h"]
    res.y_name = str(y)
    res.x_name = str(x)
    res.method = "panel_lp_iv"
    res.ci_level = 1.0 - alpha
    weak_hs = [
        int(h) for h, fv in fs_f_by_h.items() if np.isfinite(fv) and float(fv) < 10.0
    ]
    res._set_specification(
        {
            "estimator": "panel_lp_iv",
            "method": "panel_lp_iv",
            "y_name": str(y),
            "x_name": str(x),
            "z_names": list(z_names),
            "controls": list(ctl),
            "n_lags": int(n_lags),
            "horizons": [int(h) for h in horizons_list],
            "estimand": "diff_from_t_minus_1",
            "outcome_formula": "y_{i,t+h} - y_{i,t-1}",
            "fixed_effects": ["entity", "time"],
            "entity_level": str(entity_level),
            "time_level": str(time_level),
            "cov_type": "cluster",
            "cov_summary": f"entity-clustered ({entity_level})",
            "small_sample_correction": False,
            "alpha": float(alpha),
            "ci_level": float(1.0 - alpha),
            "interval_method": "normal_wald",
            "n_obs_by_horizon": n_obs_by_h,
            "sample_start_by_horizon": sample_start_by_h,
            "sample_end_by_horizon": sample_end_by_h,
            "first_stage_f_by_horizon": fs_f_by_h,
            "weak_iv_horizons_10": weak_hs,
        }
    )
    return res


__all__ = ["panel_lp_iv"]
