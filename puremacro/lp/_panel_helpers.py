"""Pure-numpy two-way fixed-effects within transformation for panel LP,
plus the shared horizon-loop engine consumed by ``panel_lp`` /
``panel_lp_dk``.

Replaces linearmodels.PanelOLS(entity_effects=True, time_effects=True).
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .._linalg import inv_xtx


def two_way_fe_within(
    df_wide: pd.DataFrame,
    *,
    y_col: str,
    x_cols: Sequence[str],
    entity_level: str = "code",
    time_level: str = "date",
    tol: float = 1e-10,
    max_iter: int = 200,
) -> dict:
    """Iteratively project out entity and time means until convergence,
    then OLS on the within-projected data. Equivalent to linearmodels'
    PanelOLS with entity_effects=True, time_effects=True (and works on
    unbalanced panels because the iterative algorithm converges either way).

    Standard errors are cluster-robust by entity (the default for two-way
    panel models in macro/finance applications).

    Parameters
    ----------
    df_wide : pd.DataFrame
        MultiIndex (entity, time) with `y_col` and each `x_col` as columns.
    y_col : str
        Outcome column name.
    x_cols : sequence of str
        Regressor column names.
    entity_level : str, default 'code'
        Name of the entity level in the MultiIndex.
    time_level : str, default 'date'
        Name of the time level in the MultiIndex.
    tol : float
        Convergence tolerance for the iterative demeaning.
    max_iter : int
        Hard cap on iterations.

    Returns
    -------
    dict with keys:
        beta, se, t, vcov, residuals, n_obs, entity_keys, time_keys,
        X_within, y_within
    """
    cols = [y_col] + list(x_cols)
    sub = df_wide[cols].dropna().copy()
    if sub.index.nlevels < 2:
        raise ValueError("two_way_fe_within requires a (entity, time) MultiIndex")
    sub.index = sub.index.set_names([entity_level, time_level])
    arr = sub.values.astype(float)
    n_rows = arr.shape[0]

    entity_idx = sub.index.get_level_values(entity_level).values
    time_idx = sub.index.get_level_values(time_level).values

    # Iterative demeaning by entity then by time.
    for _ in range(max_iter):
        ent_means = pd.DataFrame(arr).groupby(entity_idx).transform("mean").values
        arr = arr - ent_means
        t_means = pd.DataFrame(arr).groupby(time_idx).transform("mean").values
        arr_new = arr - t_means
        if np.max(np.abs(arr_new - arr)) < tol:
            arr = arr_new
            break
        arr = arr_new

    y = arr[:, 0]
    X = arr[:, 1:]
    XtX_inv = inv_xtx(X, name="two_way_fe_within")
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta

    # Cluster-robust SE by entity
    score = X * u[:, None]
    S = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(entity_idx):
        s_c = score[entity_idx == c].sum(axis=0)
        S += np.outer(s_c, s_c)
    vcov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(vcov))
    return {
        "beta": beta,
        "se": se,
        "t": beta / se,
        "vcov": vcov,
        "residuals": u,
        "n_obs": int(n_rows),
        "entity_keys": entity_idx,
        "time_keys": time_idx,
        "X_within": X,
        "y_within": y,
        "XtX_inv": XtX_inv,
    }


# ---------------------------------------------------------------------------
# Shared engine for ``panel_lp`` and ``panel_lp_dk``
# ---------------------------------------------------------------------------


SeFn = Callable[[dict], float]
"""Callable that takes the dict returned by ``two_way_fe_within`` and
returns the standard error of the focal (first) regressor."""


def _focal_cluster_se(out: dict) -> float:
    """SE of the first regressor under cluster-robust-by-entity vcov
    (already computed inside ``two_way_fe_within``)."""
    return float(out["se"][0])


def _focal_dk_se(out: dict) -> float:
    """SE of the first regressor under Driscoll-Kraay (1998) HAC.

    Uses the within-projected design returned by ``two_way_fe_within``,
    avoiding the duplicate iterative demeaning that the original
    ``panel_lp_dk`` performed.
    """
    from ..inference.dk import driscoll_kraay

    X = out["X_within"]
    u = out["residuals"]
    score = X * u[:, None]
    L = max(1, int(round(4 * (len(X) / 100) ** (2 / 9))))
    S = driscoll_kraay(score, out["time_keys"], lags=L)
    XtX_inv = out["XtX_inv"]
    vcov = XtX_inv @ S @ XtX_inv
    return float(np.sqrt(np.diag(vcov))[0])


def panel_lp_horizon_loop(
    df_wide: pd.DataFrame,
    *,
    y: str,
    x: str,
    horizons: Iterable[int],
    n_lags: int,
    controls: Sequence[str],
    alpha: float,
    entity_level: str,
    time_level: str,
    se_fn: SeFn,
) -> pd.DataFrame:
    """Run the per-horizon two-way-FE panel LP with a pluggable SE
    estimator. Used by ``panel_lp`` (cluster) and ``panel_lp_dk`` (DK).

    The focal regressor is ``x`` (first column of the design after lag
    construction); ``se_fn`` receives the dict returned by
    :func:`two_way_fe_within` and returns the SE of that regressor.

    Returns a DataFrame with columns ``[h, beta, se, t, lo, hi]``.
    """
    horizons = list(horizons)
    ctl = list(controls)
    z_crit = norm.ppf(1 - alpha / 2)

    df = df_wide.copy()
    df.index = df.index.set_names([entity_level, time_level])
    df = df.sort_index()
    grouped = df.groupby(level=entity_level)
    for lag in range(1, n_lags + 1):
        df[f"__{x}_L{lag}__"] = grouped[x].shift(lag)
        df[f"__{y}_L{lag}__"] = grouped[y].shift(lag)
        for c in ctl:
            df[f"__{c}_L{lag}__"] = grouped[c].shift(lag)

    rows = []
    for h in horizons:
        col_dy = f"__dy_h{h}__"
        df[col_dy] = grouped[y].shift(-h) - grouped[y].shift(1)
        x_cols = [x]
        for lag in range(1, n_lags + 1):
            x_cols += [f"__{x}_L{lag}__", f"__{y}_L{lag}__"]
            for c in ctl:
                x_cols.append(f"__{c}_L{lag}__")
        for c in ctl:
            x_cols.append(c)
        sub = df[[col_dy] + x_cols].dropna()
        if sub.empty:
            rows.append({"h": h, "beta": np.nan, "se": np.nan,
                         "t": np.nan, "lo": np.nan, "hi": np.nan})
            continue
        out = two_way_fe_within(
            sub, y_col=col_dy, x_cols=x_cols,
            entity_level=entity_level, time_level=time_level,
        )
        beta_h = float(out["beta"][0])
        se_h = se_fn(out)
        rows.append({
            "h": h,
            "beta": beta_h,
            "se": se_h,
            "t": beta_h / se_h if se_h > 0 else np.nan,
            "lo": beta_h - z_crit * se_h,
            "hi": beta_h + z_crit * se_h,
        })
    return pd.DataFrame(rows)


__all__ = [
    "two_way_fe_within",
    "panel_lp_horizon_loop",
    "_focal_cluster_se",
    "_focal_dk_se",
]
