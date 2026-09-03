"""Pesaran-Smith (1995) Mean Group estimator for panel LP.

Estimates a separate Jordà LP per entity, then averages the per-entity
β_h coefficients. Standard errors come from the cross-sectional spread
of the per-entity estimates (Pesaran-Smith 1995 asymptotics).

This is the natural panel-LP estimator under unrestricted heterogeneity
of the slope coefficients across entities — i.e., when β_h^c varies by
country and the pooled estimator would be inconsistent.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .jorda import lp_hac


def mean_group_panel_lp(
    df_wide: pd.DataFrame,
    y: str,
    x: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] | None = None,
    alpha: float = 0.10,
    entity_level: str = "code",
    time_level: str = "date",
    min_obs: int = 30,
) -> pd.DataFrame:
    """Estimate panel LP via Pesaran-Smith Mean Group.

    For each entity c with at least `min_obs` observations, run lp_hac
    individually, collect β_h^c, then return:
        β̂^MG_h = (1/N) Σ_c β_h^c
        SE_h = sqrt( (1 / (N (N-1))) Σ_c (β_h^c - β̂^MG_h)^2 )

    Returns DataFrame with [h, beta, se, lo, hi, n_entities_used].
    """
    horizons = list(horizons); ctl = list(controls or [])
    z_crit = norm.ppf(1 - alpha / 2)
    df = df_wide.copy()
    df.index = df.index.set_names([entity_level, time_level])
    # `lp_hac` builds its leads and lags with POSITIONAL `.shift()`, so the row
    # order is the time order as far as it is concerned. Without this sort an
    # unsorted frame silently produces nonsense lags: on an AR(1)-with-shock
    # panel whose true h=1 response is 0.8, the sorted answer is 0.765 and the
    # same data with its rows shuffled gives 0.055 -- attenuated almost to
    # zero, with no error and no warning. `_panel_helpers.panel_lp_horizon_loop`
    # and `cce.cce_panel_lp` both already sort; this path was the one that did
    # not.
    df = df.sort_index()

    entities = df.index.get_level_values(entity_level).unique()
    per_entity = {}
    for c in entities:
        sub = df.xs(c, level=entity_level).copy()
        if len(sub) < min_obs:
            continue
        sub.index = pd.DatetimeIndex(sub.index) if sub.index.dtype.kind == 'M' else sub.index
        try:
            res = lp_hac(sub, y=y, x=x, horizons=horizons,
                          n_lags=n_lags, controls=controls, alpha=alpha)
            per_entity[c] = res.set_index("h")["beta"]
        except Exception:
            continue

    rows = []
    n_used = len(per_entity)
    for h in horizons:
        if n_used == 0:
            rows.append({"h": h, "beta": np.nan, "se": np.nan,
                         "lo": np.nan, "hi": np.nan, "n_entities_used": 0})
            continue
        betas = np.array([per_entity[c].get(h, np.nan) for c in per_entity])
        betas = betas[np.isfinite(betas)]
        n_h = len(betas)
        if n_h <= 1:
            rows.append({"h": h, "beta": float(np.mean(betas)) if n_h else np.nan,
                         "se": np.nan, "lo": np.nan, "hi": np.nan,
                         "n_entities_used": int(n_h)})
            continue
        beta_mg = float(np.mean(betas))
        # Pesaran-Smith SE
        se_mg = float(np.sqrt(np.sum((betas - beta_mg) ** 2) / (n_h * (n_h - 1))))
        rows.append({
            "h": h, "beta": beta_mg, "se": se_mg,
            "lo": beta_mg - z_crit * se_mg,
            "hi": beta_mg + z_crit * se_mg,
            "n_entities_used": int(n_h),
        })
    return pd.DataFrame(rows)


__all__ = ["mean_group_panel_lp"]
