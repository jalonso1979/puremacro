"""Panel local projection with two-way FE + cluster-robust SE.

Thin wrapper over the shared engine in
:mod:`puremacro.lp._panel_helpers`. Cluster-robust-by-entity SE is the
convention for two-way panel LP in macro applications.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ._panel_helpers import _focal_cluster_se, panel_lp_horizon_loop


def panel_lp(
    df_wide: pd.DataFrame,
    y: str,
    x: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] = (),
    alpha: float = 0.10,
    entity_level: str = "code",
    time_level: str = "date",
) -> pd.DataFrame:
    """Estimate β_h from
        y^{(c)}_{t+h} - y^{(c)}_{t-1} = μ_c + τ_t + β_h x^{(c)}_t + … + ε
    via the two-way FE within transformation, with cluster-robust SE
    by entity.

    Returns DataFrame with columns ``[h, beta, se, t, lo, hi]``.
    """
    return panel_lp_horizon_loop(
        df_wide,
        y=y, x=x,
        horizons=horizons,
        n_lags=n_lags,
        controls=controls,
        alpha=alpha,
        entity_level=entity_level,
        time_level=time_level,
        se_fn=_focal_cluster_se,
    )


def lp_panel_regime_interaction(
    df: pd.DataFrame,
    *,
    y: str,
    x: str,
    regime_col: str,
    horizons: Iterable[int] = range(0, 9),
    n_lags: int = 2,
    controls: Sequence[str] = (),
    alpha: float = 0.10,
    entity_level: str = "code",
    time_level: str = "date",
    regimes: list | None = None,
    time_effects: bool = True,
) -> pd.DataFrame:
    """Estimate a panel LP with regime x shock interaction terms.

    For each horizon h the specification is::

        y_{i,t+h} = alpha_i^h + xi_t^h
                    + sum_R beta_R^h * shock_{i,t} * 1{regime_{i,t}=R}
                    + gamma^h' X_{i,t} + eps_{i,t+h}

    where R ranges over ``regimes``. Estimation is by
    ``linearmodels.PanelOLS`` with entity fixed effects and (optionally)
    time fixed effects. Driscoll-Kraay standard errors (Bartlett kernel,
    bandwidth = floor(0.75 * T^(1/3))) are used throughout.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form panel with columns [entity_level, time_level, y, x,
        regime_col].
    y : str
        Dependent variable name. Canonical alias for legacy ``target``.
    x : str
        Shock / regressor name. Canonical alias for legacy ``shock``.
    regime_col : str
        Column containing regime labels (any comparable type: str or int).
    horizons : iterable of int
        Forecast horizons to estimate (e.g. range(0, 9)).
        Canonical alias for legacy scalar ``horizon`` (which was the max);
        this accepts an explicit iterable so sparse sets are possible.
    n_lags : int
        Number of own-lags of y and x to include as controls.
        Canonical alias for legacy ``lags``.
    controls : sequence of str
        Additional contemporaneous control columns (no lags added).
    alpha : float
        Significance level; CI width = 1 - alpha (e.g. alpha=0.10 -> 90% CI).
        Canonical alias for legacy ``ci`` (note direction flip:
        legacy ci=0.90 <=> canonical alpha=0.10).
    entity_level : str
        Column name for the entity identifier. Canonical alias for
        legacy ``entity_col``.
    time_level : str
        Column name for the time-period identifier. Canonical alias for
        legacy ``time_col``.
    regimes : list | None
        Regime values to interact with the shock. Defaults to the sorted
        unique values of ``regime_col``. Legacy default was
        ['BASE', 'GR', 'POST', 'COVID', 'POST_COVID'].
    time_effects : bool
        Whether to include time fixed effects (default True).

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with columns:
        ``h``             : int — forecast horizon
        ``regime``        : — regime label (same type as values in regime_col)
        ``beta``          : float — point estimate
        ``se``            : float — Driscoll-Kraay standard error
        ``lo``            : float — lower confidence bound (1-alpha CI)
        ``hi``            : float — upper confidence bound (1-alpha CI)
        ``wald_pval``     : float — p-value for Wald equality test across
                            regimes (NaN if fewer than 2 regimes in sample)
        ``n_obs``         : int or NaN — observations used at the regression
                            at the first horizon only; NaN at every other
                            horizon. Inspect ``h == horizons[0]`` rows to
                            recover sample size.
        ``n_entities``    : int or NaN — distinct entities at the first
                            horizon only; NaN elsewhere (same convention as
                            ``n_obs``).

    Notes
    -----
    The Driscoll-Kraay bandwidth is computed once from the full panel as
    ``floor(0.75 * T^(1/3))``, where T is the number of distinct time
    periods. This matches the legacy ``lp_panel.lp_panel_regime_interaction``
    behaviour exactly.
    """
    from linearmodels import PanelOLS  # lazy: Pyodide contract

    controls = list(controls)
    horizons = list(horizons)
    z = norm.ppf(1 - alpha / 2)

    data = df.copy()
    data = data.sort_values([entity_level, time_level])

    # Resolve regime list from data if not provided
    if regimes is None:
        regimes = sorted(data[regime_col].dropna().unique().tolist())

    # Driscoll-Kraay bandwidth (computed once from full panel)
    T_total = data[time_level].nunique()
    bw = max(1, math.floor(0.75 * T_total ** (1 / 3)))

    n_entities_total = data[entity_level].nunique()

    # Build regime x shock interaction columns (safe names using index)
    interact_cols: list[str] = []
    for idx, r in enumerate(regimes):
        col = f"__shock_x_regime_{idx}"
        data[col] = data[x] * (data[regime_col] == r).astype(float)
        interact_cols.append(col)

    # Build lag columns (within-entity)
    for lag in range(1, n_lags + 1):
        data[f"__{y}_lag{lag}"] = data.groupby(entity_level)[y].shift(lag)
        data[f"__{x}_lag{lag}"] = data.groupby(entity_level)[x].shift(lag)

    lag_cols = (
        [f"__{y}_lag{lag}" for lag in range(1, n_lags + 1)]
        + [f"__{x}_lag{lag}" for lag in range(1, n_lags + 1)]
    )

    # Storage: per-horizon, per-regime
    rows: list[dict] = []
    n_obs_h0: int = 0
    n_entities_h0: int = n_entities_total

    for h in horizons:
        dh = data.copy()
        lead_col = f"__{y}_lead{h}"
        dh[lead_col] = dh.groupby(entity_level)[y].shift(-h)

        keep_cols = (
            [entity_level, time_level, lead_col]
            + interact_cols
            + lag_cols
            + controls
        )
        dh = dh[keep_cols].dropna()

        if len(dh) == 0:
            for r in regimes:
                rows.append({
                    "h": h, "regime": r,
                    "beta": np.nan, "se": np.nan,
                    "lo": np.nan, "hi": np.nan,
                    "wald_pval": np.nan,
                    "n_obs": np.nan, "n_entities": np.nan,
                })
            continue

        dh = dh.set_index([entity_level, time_level])

        rhs_cols = interact_cols + lag_cols + controls
        formula_rhs = " + ".join(rhs_cols)
        fe_str = "EntityEffects + TimeEffects" if time_effects else "EntityEffects"
        formula = f"{lead_col} ~ {formula_rhs} + {fe_str}"

        betas_h: dict = {}
        ses_h: dict = {}
        wald_pval_h: float = np.nan
        n_obs_h: int = 0

        try:
            mod = PanelOLS.from_formula(formula, data=dh)
            fit = mod.fit(
                cov_type="kernel",
                kernel="bartlett",
                bandwidth=bw,
                debiased=True,
            )

            for r, col in zip(regimes, interact_cols):
                if col in fit.params.index:
                    betas_h[r] = float(fit.params[col])
                    ses_h[r] = float(fit.std_errors[col])
                else:
                    betas_h[r] = np.nan
                    ses_h[r] = np.nan

            # Wald equality test: H0 all beta_R^h equal across regimes
            valid_cols = [col for col in interact_cols if col in fit.params.index]
            if len(valid_cols) >= 2:
                try:
                    restrictions = ", ".join(
                        f"{valid_cols[i]} = {valid_cols[i + 1]}"
                        for i in range(len(valid_cols) - 1)
                    )
                    wald_res = fit.wald_test(restrictions)
                    wald_pval_h = float(wald_res.pval)
                except Exception:
                    wald_pval_h = np.nan

            n_obs_h = int(fit.nobs)
            if h == horizons[0]:
                n_obs_h0 = n_obs_h
                n_entities_h0 = n_entities_total

        except Exception:
            for r in regimes:
                betas_h[r] = np.nan
                ses_h[r] = np.nan

        for r in regimes:
            b = betas_h.get(r, np.nan)
            s = ses_h.get(r, np.nan)
            rows.append({
                "h": h,
                "regime": r,
                "beta": b,
                "se": s,
                "lo": b - z * s,
                "hi": b + z * s,
                "wald_pval": wald_pval_h,
                "n_obs": n_obs_h0 if h == horizons[0] else np.nan,
                "n_entities": n_entities_h0 if h == horizons[0] else np.nan,
            })

    return pd.DataFrame(rows)


__all__ = ["panel_lp", "lp_panel_regime_interaction"]
