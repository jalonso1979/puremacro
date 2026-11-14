"""Pesaran (2006) Common Correlated Effects panel LP estimators.

Supports both:
- Pooled CCE (CCEP, ``estimator="pooled"``, default): OLS with entity FE on the
  panel LP augmented with cross-sectional means of ``y``, ``x``, and controls,
  assuming homogeneous factor loadings after cross-sectional projection.
- Mean Group CCE (CCE-MG, ``estimator="mean_group"``): Pesaran (2006)
  unit-by-unit time-series LP regressions augmented with horizon-specific
  cross-sectional means (absorbing heterogeneous factor loadings
  :math:`\\lambda_i' f_t`), cross-unit Mean Group slope averages, and Pesaran
  nonparametric cross-unit standard errors.

References:
- Pesaran (2006). Estimation and inference in large heterogeneous panels
  with a multifactor error structure. Econometrica 74(4): 967-1012.
"""
from __future__ import annotations

from typing import Any, Iterable, Literal, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from ._results import LPResult
from ._common import (
    _DEFAULT_ALPHA_10,
    _DEFAULT_HORIZONS_21,
    _DEFAULT_N_LAGS_2,
    check_regular_index,
    resolve_lp_kwargs,
    validate_lp_columns_finite,
)
from ._panel_helpers import as_panel_index

_MIN_MG_ENTITIES = 3


def _entity_fe_ols(sub: pd.DataFrame, y_col: str, x_cols: Sequence[str],
                   entity_level: str) -> dict[str, np.ndarray]:
    """Entity-FE-only within OLS with cluster-robust SE.

    The CCE estimator already controls for time via the cross-sectional
    means; adding time FE on top creates perfect collinearity with those
    proxies. We therefore demean only by entity (Pesaran 2006, eq. 12).
    """
    arr = sub[[y_col] + list(x_cols)].values.astype(float)
    entity_idx = sub.index.get_level_values(entity_level).values
    ent_means = pd.DataFrame(arr).groupby(entity_idx).transform("mean").values
    arr = arr - ent_means
    y_d = arr[:, 0]
    X_d = arr[:, 1:]
    beta, *_ = np.linalg.lstsq(X_d, y_d, rcond=None)
    u = y_d - X_d @ beta
    # Cluster-robust SE by entity
    score = X_d * u[:, None]
    S = np.zeros((X_d.shape[1], X_d.shape[1]))
    for c in np.unique(entity_idx):
        s_c = score[entity_idx == c].sum(axis=0)
        S += np.outer(s_c, s_c)
    XtX = X_d.T @ X_d
    XtX_inv = np.linalg.pinv(XtX)
    vcov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    return {"beta": beta, "se": se}


def _validate_cce_inputs(
    horizons: Sequence[Any],
    ctl: Sequence[str],
    y: str,
    x: str,
    estimator: str,
    min_obs: int | None,
    func_name: str,
) -> int | None:
    """Validate CCE options, horizons, and variable role uniqueness."""
    if estimator not in ("pooled", "mean_group"):
        raise ValueError(
            f"{func_name}: estimator must be 'pooled' or 'mean_group', got {estimator!r}"
        )
    if not horizons:
        raise ValueError(f"{func_name}: horizons must be non-empty")
    for h_val in horizons:
        if isinstance(h_val, bool) or not isinstance(h_val, (int, np.integer)) or int(h_val) < 0:
            raise ValueError(
                f"{func_name}: horizons must contain non-negative integers, got {h_val!r}"
            )
    if len({int(h_val) for h_val in horizons}) != len(horizons):
        raise ValueError(f"{func_name}: horizons must not contain duplicates")
    if y == x or y in ctl or x in ctl or len(set(ctl)) != len(ctl):
        raise ValueError(
            f"{func_name}: y, x, and controls must be distinct column names "
            f"(got y={y!r}, x={x!r}, controls={list(ctl)!r})"
        )
    if min_obs is not None:
        if isinstance(min_obs, bool) or not isinstance(min_obs, (int, np.integer)) or int(min_obs) < 1:
            raise ValueError(
                f"{func_name}: min_obs must be a positive integer or None, got {min_obs!r}"
            )
        return int(min_obs)
    return None


def _structural_lp_cols(x: str, y: str, ctl: Sequence[str], n_lags: int) -> list[str]:
    """Return ordered LP structural regressors with focal x at index 0."""
    cols = [x]
    for lag in range(1, n_lags + 1):
        cols += [f"__{x}_L{lag}__", f"__{y}_L{lag}__"]
        for c in ctl:
            cols.append(f"__{c}_L{lag}__")
    for c in ctl:
        cols.append(c)
    return cols


def _estimate_unit_cce_ols(
    sub_ent: pd.DataFrame,
    y_col: str,
    x_cols: Sequence[str],
    effective_min_obs: int,
) -> tuple[float | None, str | None]:
    """Run rank-checked unit-level CCE OLS; returns (beta_x, exclusion_reason)."""
    k_regs = 1 + len(x_cols)
    n_ent = len(sub_ent)
    if n_ent < effective_min_obs:
        return (
            None,
            f"insufficient observations ({n_ent} < {effective_min_obs}; k={k_regs})",
        )
    y_vec = sub_ent[y_col].to_numpy(dtype=float)
    X_mat = sub_ent[list(x_cols)].to_numpy(dtype=float)
    Z = np.column_stack([np.ones(n_ent, dtype=float), X_mat])
    col_norms = np.linalg.norm(Z, axis=0)
    if np.any(col_norms <= 1e-12):
        return None, "collinear or zero-variance regressor column"
    # Scale columns before rank check so units with different magnitudes remain comparable.
    Z_scaled = Z / col_norms
    rank = int(np.linalg.matrix_rank(Z_scaled))
    if rank < k_regs:
        return None, f"rank-deficient design matrix (rank {rank} < {k_regs})"
    coef, *_ = np.linalg.lstsq(Z, y_vec, rcond=None)
    beta_i = float(coef[1])
    if not np.isfinite(beta_i):
        return None, "non-finite slope estimate"
    return beta_i, None


def _estimate_cce_mean_group_horizon(
    sub: pd.DataFrame,
    col_dy: str,
    x_cols: Sequence[str],
    unique_entities: pd.Index,
    entity_level: str,
    h: int,
    z_crit: float,
    min_obs_int: int | None,
    func_name: str,
) -> tuple[dict[str, Any], int, int, dict[Any, str]]:
    """Estimate Pesaran (2006) CCE-MG slope and nonparametric SE at horizon h."""
    k_regs = 1 + len(x_cols)
    effective_min_obs = (
        max(10, k_regs + 2) if min_obs_int is None else max(min_obs_int, k_regs + 1)
    )
    grouped_sub = {ent: grp for ent, grp in sub.groupby(level=entity_level, sort=False)}
    unit_betas: list[float] = []
    excluded: dict[Any, str] = {}
    obs_used = 0

    for ent in unique_entities:
        sub_ent = grouped_sub.get(ent)
        if sub_ent is None:
            reason = f"insufficient observations (0 < {effective_min_obs}; k={k_regs})"
            excluded[ent] = reason
            warnings.warn(
                f"{func_name} (estimator='mean_group'): excluded entity {ent!r} at h={h}: {reason}",
                UserWarning,
                stacklevel=3,
            )
            continue
        beta_i, fail_reason = _estimate_unit_cce_ols(
            sub_ent, y_col=col_dy, x_cols=x_cols, effective_min_obs=effective_min_obs
        )
        if fail_reason is not None or beta_i is None:
            reason_msg = fail_reason or "invalid unit regression"
            excluded[ent] = reason_msg
            warnings.warn(
                f"{func_name} (estimator='mean_group'): excluded entity {ent!r} at h={h}: {reason_msg}",
                UserWarning,
                stacklevel=3,
            )
            continue
        unit_betas.append(beta_i)
        obs_used += len(sub_ent)

    n_used = len(unit_betas)
    if n_used < _MIN_MG_ENTITIES:
        raise ValueError(
            f"{func_name} (estimator='mean_group'): insufficient valid entities at horizon h={h}: "
            f"got {n_used}, need at least {_MIN_MG_ENTITIES} (excluded={excluded})."
        )

    betas_arr = np.asarray(unit_betas, dtype=float)
    beta_mg = float(np.mean(betas_arr))
    # Pesaran (2006) eq. (56): nonparametric cross-unit Mean Group variance / SE.
    var_mg = float(np.var(betas_arr, ddof=1) / n_used)
    se_mg = float(np.sqrt(max(var_mg, 0.0)))
    row: dict[str, Any] = {
        "h": h,
        "beta": beta_mg,
        "se": se_mg,
        "lo": beta_mg - z_crit * se_mg,
        "hi": beta_mg + z_crit * se_mg,
        "n_entities_used": int(n_used),
    }
    return row, obs_used, n_used, excluded


def cce_panel_lp(
    df_wide: pd.DataFrame,
    y: str,
    x: str,
    horizons: Iterable[int] = _DEFAULT_HORIZONS_21,
    n_lags: int = _DEFAULT_N_LAGS_2,
    controls: Sequence[str] = (),
    alpha: float = _DEFAULT_ALPHA_10,
    entity_level: str = "code",
    time_level: str = "date",
    *,
    estimator: Literal["pooled", "mean_group"] = "pooled",
    min_obs: int | None = None,
    allow_index_gaps: bool = False,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
    unit_col: str | None = None,
    time_col: str | None = None,
) -> LPResult:
    """Estimate panel LP β_h with Pesaran (2006) Common Correlated Effects.

    Parameters
    ----------
    df_wide:
        Panel DataFrame indexed by ``(entity_level, time_level)`` or flat
        DataFrame with ``unit_col`` and ``time_col``.
    y, x:
        Dependent and shock/treatment column names.
    horizons, n_lags, controls, alpha:
        Horizons ``h >= 0``, lag count ``>= 0``, optional controls, and CI level
        ``alpha`` (equivalently via aliases ``horizon``, ``lags``, ``ci``).
    estimator:
        ``"pooled"`` (default, CCEP) estimates pooled OLS with entity FE and
        entity-clustered SEs, assuming homogeneous factor-loading projections.
        ``"mean_group"`` estimates Pesaran (2006) CCE Mean Group (CCE-MG):
        unit-by-unit time-series regressions augmented with horizon-specific
        cross-sectional averages to absorb heterogeneous factor loadings
        :math:`\\lambda_i' f_t`, averaged across units with Pesaran
        nonparametric cross-unit standard errors.
    min_obs:
        Minimum usable observations per entity (only used when
        ``estimator="mean_group"``; defaults to ``max(10, k_regs + 2)``).
        Insufficient or rank-deficient units are warned and recorded in
        ``res.attrs["specification"]["excluded_entities_by_horizon"]`` rather
        than silently dropped; raises ``ValueError`` if fewer than 3 valid
        entities remain at any horizon.
    allow_index_gaps:
        Allow gaps in dated ``DatetimeIndex``/``PeriodIndex`` per entity.

    Returns
    -------
    LPResult
        DataFrame-backed result with columns ``[h, beta, se, lo, hi]`` (plus
        ``n_entities_used`` for ``estimator="mean_group"``) and full provenance
        in ``attrs["specification"]`` and ``metadata``.
    """
    horizons_iter, n_lags_val, alpha_val = resolve_lp_kwargs(
        horizons,
        n_lags,
        alpha,
        lags=lags,
        horizon=horizon,
        ci=ci,
        name="cce_panel_lp",
        strict_aliases=True,
    )
    horizons_list = [int(h) if isinstance(h, (int, np.integer)) and not isinstance(h, bool) else h for h in horizons_iter]
    ctl = list(controls)
    min_obs_int = _validate_cce_inputs(
        horizons_list,
        ctl,
        y,
        x,
        estimator,
        min_obs,
        func_name="cce_panel_lp",
    )
    horizons_clean = [int(h) for h in horizons_list]
    z_crit = float(norm.ppf(1 - alpha_val / 2))

    df_wide, entity_level, time_level = as_panel_index(
        df_wide,
        entity_level=entity_level,
        time_level=time_level,
        unit_col=unit_col,
        time_col=time_col,
        func="cce_panel_lp",
    )
    if df_wide.index.get_level_values(entity_level).isna().any() or df_wide.index.get_level_values(time_level).isna().any():
        raise ValueError("cce_panel_lp: panel index contains missing entity or time labels")
    if df_wide.index.has_duplicates:
        raise ValueError(
            f"cce_panel_lp: duplicate ({entity_level!r}, {time_level!r}) index entries detected; "
            "deduplicate or aggregate before estimation."
        )
    validate_lp_columns_finite(df_wide, [y, x] + ctl, func="cce_panel_lp")

    df = df_wide.copy()
    df.index = df.index.set_names([entity_level, time_level])
    df = df.sort_index()
    for _, grp in df.groupby(level=entity_level, sort=False):
        check_regular_index(
            grp.index.get_level_values(time_level),
            func="cce_panel_lp",
            cumulative=False,
            allow_gaps=allow_index_gaps,
        )

    unique_entities = df.index.get_level_values(entity_level).unique()
    if estimator == "mean_group" and len(unique_entities) < _MIN_MG_ENTITIES:
        raise ValueError(
            f"cce_panel_lp (estimator='mean_group'): requires at least {_MIN_MG_ENTITIES} entities "
            f"for Pesaran CCE-MG estimation and inference (with <= 2 entities, unit slopes are "
            f"algebraically identical under full cross-sectional mean augmentation), got {len(unique_entities)}."
        )

    grouped_e = df.groupby(level=entity_level)
    # Pre-compute lags after deterministic sort_index()
    for lag in range(1, n_lags_val + 1):
        df[f"__{x}_L{lag}__"] = grouped_e[x].shift(lag)
        df[f"__{y}_L{lag}__"] = grouped_e[y].shift(lag)
        for c in ctl:
            df[f"__{c}_L{lag}__"] = grouped_e[c].shift(lag)

    # Cross-sectional means by time (CCE augmentation)
    grouped_t = df.groupby(level=time_level)
    df["__cs_mean_y__"] = grouped_t[y].transform("mean")
    df["__cs_mean_x__"] = grouped_t[x].transform("mean")
    for c in ctl:
        df[f"__cs_mean_{c}__"] = grouped_t[c].transform("mean")
    if estimator == "mean_group":
        for lag in range(1, n_lags_val + 1):
            df[f"__cs_mean_x_L{lag}__"] = grouped_t[f"__{x}_L{lag}__"].transform("mean")
            for c in ctl:
                df[f"__cs_mean_{c}_L{lag}__"] = grouped_t[f"__{c}_L{lag}__"].transform("mean")

    structural_cols = _structural_lp_cols(x, y, ctl, n_lags_val)
    rows: list[dict[str, Any]] = []
    n_obs_by_h: dict[int, int] = {}
    n_entities_by_h: dict[int, int] = {}
    excluded_by_h: dict[int, dict[Any, str]] = {}

    for h in horizons_clean:
        col_dy = f"__dy_h{h}__"
        df[col_dy] = grouped_e[y].shift(-h) - grouped_e[y].shift(1)

        if estimator == "pooled":
            x_cols = list(structural_cols)
            x_cols.extend(["__cs_mean_y__", "__cs_mean_x__"])
            for c in ctl:
                x_cols.append(f"__cs_mean_{c}__")
            sub = df[[col_dy] + x_cols].dropna()
            n_obs_by_h[h] = int(len(sub))
            if sub.empty:
                rows.append({"h": h, "beta": np.nan, "se": np.nan, "lo": np.nan, "hi": np.nan})
                continue
            out = _entity_fe_ols(sub, y_col=col_dy, x_cols=x_cols, entity_level=entity_level)
            beta_h = float(out["beta"][0])
            se_h = float(out["se"][0])
            rows.append({
                "h": h,
                "beta": beta_h,
                "se": se_h,
                "lo": beta_h - z_crit * se_h,
                "hi": beta_h + z_crit * se_h,
            })
        else:
            col_cs_dy = f"__cs_mean_dy_h{h}__"
            df[col_cs_dy] = df.groupby(level=time_level)[col_dy].transform("mean")
            mg_x_cols = list(structural_cols) + [col_cs_dy, "__cs_mean_x__"]
            for lag in range(1, n_lags_val + 1):
                mg_x_cols.append(f"__cs_mean_x_L{lag}__")
            for c in ctl:
                mg_x_cols.append(f"__cs_mean_{c}__")
                for lag in range(1, n_lags_val + 1):
                    mg_x_cols.append(f"__cs_mean_{c}_L{lag}__")
            sub = df[[col_dy] + mg_x_cols].dropna()
            row, obs_used, n_used, excluded = _estimate_cce_mean_group_horizon(
                sub=sub,
                col_dy=col_dy,
                x_cols=mg_x_cols,
                unique_entities=unique_entities,
                entity_level=entity_level,
                h=h,
                z_crit=z_crit,
                min_obs_int=min_obs_int,
                func_name="cce_panel_lp",
            )
            n_obs_by_h[h] = int(obs_used)
            n_entities_by_h[h] = int(n_used)
            if excluded:
                excluded_by_h[h] = excluded
            rows.append(row)

    res = LPResult(rows)
    if "h" in res.columns:
        res.index = res["h"]
    res.y_name = str(y)
    res.x_name = str(x)
    res.method = "cce_panel_lp"
    res.ci_level = 1.0 - alpha_val

    if estimator == "pooled":
        spec: dict[str, Any] = {
            "estimator": "cce_panel_lp",
            "cce_estimator": "cce_pooled",
            "factor_loadings": "homogeneous",
            "method": "cce_panel_lp",
            "y_name": str(y),
            "x_name": str(x),
            "controls": list(ctl),
            "n_lags": int(n_lags_val),
            "horizons": list(horizons_clean),
            "cov_type": "cluster",
            "ci_level": float(1.0 - alpha_val),
            "cov_summary": f"cluster({entity_level})",
            "n_obs_by_horizon": n_obs_by_h,
        }
    else:
        spec = {
            "estimator": "cce_mean_group",
            "cce_estimator": "cce_mean_group",
            "factor_loadings": "heterogeneous",
            "method": "cce_panel_lp",
            "y_name": str(y),
            "x_name": str(x),
            "controls": list(ctl),
            "n_lags": int(n_lags_val),
            "horizons": list(horizons_clean),
            "cov_type": "pesaran_mg",
            "ci_level": float(1.0 - alpha_val),
            "cov_summary": "Pesaran (2006) CCE-MG nonparametric",
            "min_obs": min_obs_int,
            "n_obs_by_horizon": n_obs_by_h,
            "n_entities_by_horizon": n_entities_by_h,
            "excluded_entities_by_horizon": excluded_by_h,
        }
    res._set_specification(spec)
    return res


def cce_mg_panel_lp(
    df_wide: pd.DataFrame,
    y: str,
    x: str,
    horizons: Iterable[int] = _DEFAULT_HORIZONS_21,
    n_lags: int = _DEFAULT_N_LAGS_2,
    controls: Sequence[str] = (),
    alpha: float = _DEFAULT_ALPHA_10,
    entity_level: str = "code",
    time_level: str = "date",
    *,
    min_obs: int | None = None,
    allow_index_gaps: bool = False,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
    unit_col: str | None = None,
    time_col: str | None = None,
) -> LPResult:
    """Convenience wrapper for Pesaran (2006) CCE Mean Group panel LP.

    Equivalent to ``cce_panel_lp(..., estimator="mean_group")``.
    """
    return cce_panel_lp(
        df_wide,
        y=y,
        x=x,
        horizons=horizons,
        n_lags=n_lags,
        controls=controls,
        alpha=alpha,
        entity_level=entity_level,
        time_level=time_level,
        estimator="mean_group",
        min_obs=min_obs,
        allow_index_gaps=allow_index_gaps,
        lags=lags,
        horizon=horizon,
        ci=ci,
        unit_col=unit_col,
        time_col=time_col,
    )


# Preserve frozen public __all__ snapshot (`cce_mg_panel_lp` stays importable from `puremacro.lp.cce`).
__all__ = ["cce_panel_lp"]
