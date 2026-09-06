"""Local projection with Lewbel-constructed instruments.

Pools the panel and applies :func:`puremacro.inference.lewbel_iv.lewbel_iv`
horizon-by-horizon. Entity fixed effects via dummy-variable encoding.

This wrapper does not currently expose a clustered-SE option — Lewbel
inference is delicate enough that the homoskedastic 2SLS SE in
``lewbel_iv`` is reported here unchanged. For HAC-clustered LP-IV
inference with external instruments, use :func:`puremacro.lp.iv.lp_iv`.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..inference.lewbel_iv import lewbel_iv
from ._results import LPResult
from ._common import resolve_lp_kwargs


def lp_iv_lewbel(
    panel: pd.DataFrame,
    *,
    y: str,
    x_endog: str,
    heterosk_source: str,
    controls: Sequence[str] = (),
    horizons: Iterable[int] = range(0, 13),
    n_lags: int = 2,
    entity_level: str = "code",
    time_level: str = "date",
    alpha: float = 0.10,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
) -> LPResult:
    """Local-projection IV using Lewbel-constructed instruments.

    Specification at horizon h:
        y_{i, t+h} - y_{i, t-1} = α_i + β_h · x_{i,t} + γ' · W_{i,t} + ε_{i,t,h}
    where ``x`` is the endogenous regressor and instruments are built
    from heteroskedasticity in ``heterosk_source``.

    Returns
    -------
    LPResult with columns ``[h, beta, se, t, lo, hi, first_stage_F, lewbel_p]``.
    """
    horizons, n_lags, alpha = resolve_lp_kwargs(
        horizons, n_lags, alpha, lags=lags, horizon=horizon, ci=ci, name="lp_iv_lewbel")
    horizons = list(horizons)
    controls = list(controls)
    z_crit = norm.ppf(1 - alpha / 2)

    panel = panel.sort_values([entity_level, time_level]).reset_index(drop=True)
    g = panel.groupby(entity_level, observed=True)
    for lag in range(1, n_lags + 1):
        panel[f"{x_endog}_L{lag}"] = g[x_endog].shift(lag)
        panel[f"{y}_L{lag}"] = g[y].shift(lag)
        for c in controls:
            panel[f"{c}_L{lag}"] = g[c].shift(lag)
    panel[f"{y}_Lm1"] = g[y].shift(1)

    rows = []
    for h in horizons:
        panel[f"{y}_lead_h{h}"] = g[y].shift(-h)
        col_lhs = f"{y}_dh{h}"
        panel[col_lhs] = panel[f"{y}_lead_h{h}"] - panel[f"{y}_Lm1"]

        keep_cols = [col_lhs, x_endog, heterosk_source, entity_level]
        for lag in range(1, n_lags + 1):
            keep_cols.append(f"{x_endog}_L{lag}")
            keep_cols.append(f"{y}_L{lag}")
            for c in controls:
                keep_cols.append(f"{c}_L{lag}")
        keep_cols.extend(controls)
        sub = panel[keep_cols].dropna()
        if sub.empty:
            rows.append({"h": h, "beta": np.nan, "se": np.nan, "t": np.nan,
                         "lo": np.nan, "hi": np.nan, "first_stage_F": np.nan,
                         "lewbel_p": np.nan})
            continue

        # Build matrices for lewbel_iv. Entity dummies as part of X_exog.
        ent = pd.get_dummies(sub[entity_level], drop_first=True).to_numpy(dtype=float)
        const = np.ones((len(sub), 1))
        lag_cols = []
        for lag in range(1, n_lags + 1):
            lag_cols.append(sub[f"{x_endog}_L{lag}"].to_numpy())
            lag_cols.append(sub[f"{y}_L{lag}"].to_numpy())
            for c in controls:
                lag_cols.append(sub[f"{c}_L{lag}"].to_numpy())
        ctl_cols = [sub[c].to_numpy() for c in controls]
        if lag_cols or ctl_cols:
            X_exog = np.column_stack([const, ent] + lag_cols + ctl_cols)
        else:
            X_exog = np.column_stack([const, ent])

        X_endog = sub[x_endog].to_numpy().reshape(-1, 1)
        Z_source = sub[heterosk_source].to_numpy().reshape(-1, 1)
        y_lhs = sub[col_lhs].to_numpy()

        res = lewbel_iv(y_lhs, X_endog, X_exog, Z_source)
        beta = float(res.beta[0])
        se = float(res.se[0])
        rows.append({
            "h": h,
            "beta": beta,
            "se": se,
            "t": beta / se if se > 0 else np.nan,
            "lo": beta - z_crit * se,
            "hi": beta + z_crit * se,
            "first_stage_F": float(res.first_stage_F),
            "lewbel_p": float(res.lewbel_diagnostic["p_value"]),
        })
    res = LPResult(rows)
    if "h" in res.columns:
        res.index = res["h"]
    res.y_name = str(y)
    res.x_name = str(x_endog)
    res.method = "LP-IV-Lewbel"
    res.ci_level = 1.0 - alpha
    return res


__all__ = ["lp_iv_lewbel"]
