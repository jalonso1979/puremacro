"""Estimate per-industry sensitivity β_i of log-employment to national EPU shocks.

Uses a simple LP at each horizon: Δlog(emp_{i,t+h}) = α + β_{i,h} × shock_t + ε.
Peak is the most-negative coefficient across horizons (worst-case response).
Optional shrinkage toward the grand mean (James-Stein–style).
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _one_industry_beta(
    dlog_emp: pd.Series,
    shock: pd.Series,
    horizons: Iterable[int],
    min_obs: int = 30,
) -> tuple[float, float, int] | None:
    import statsmodels.api as sm  # lazy: Pyodide contract — see ARCHITECTURE.md

    results = []
    for h in horizons:
        y = dlog_emp.shift(-h).dropna()
        x = shock.loc[y.index]
        if len(y) < min_obs:
            continue
        X = sm.add_constant(x)
        model = sm.OLS(y, X, missing="drop").fit(
            cov_type="HAC", cov_kwds={"maxlags": h + 4}
        )
        results.append((h, model.params.iloc[1], model.bse.iloc[1], int(model.nobs)))
    if not results:
        return None
    peak = min(results, key=lambda r: r[1])  # horizon, beta, se, nobs
    return peak[1], peak[2], peak[3]


def estimate_industry_sensitivities(
    industry_panel: pd.DataFrame,
    shock_panel: pd.DataFrame,
    *,
    horizons: Iterable[int] = range(0, 9),
    shrinkage: float = 0.3,
    min_obs: int = 30,
) -> pd.DataFrame:
    """Return a DataFrame with columns: naics, beta_peak, beta_peak_se, shrunk_beta, obs_count.

    The James-Stein–style shrinkage interpolates toward the cross-industry mean:
        shrunk_beta_i = (1 - shrinkage) × beta_peak_i + shrinkage × mean(beta_peak)

    Setting shrinkage=0 returns raw LP estimates; shrinkage=1 collapses all to the mean.

    "Peak" is the most-negative β_h across horizons (worst-case response). If all
    β_h are positive, peak is the smallest positive (least adverse response).
    """
    horizons = list(horizons)

    n_before = len(industry_panel)
    merged = industry_panel.merge(shock_panel, on="date", how="inner")
    n_after = len(merged)
    if n_before > 0 and (n_before - n_after) / n_before > 0.1:
        logger.warning(
            "Industry panel and shock panel share only %d/%d dates (%.0f%% dropped); "
            "results may be truncated.",
            n_after, n_before, 100 * (n_before - n_after) / n_before,
        )

    out = []
    skipped = 0
    for naics, grp in merged.groupby("naics"):
        grp = grp.sort_values("date")
        result = _one_industry_beta(
            grp.set_index("date")["dlog_emp"],
            grp.set_index("date")["shock_epu_nat"],
            horizons,
            min_obs=min_obs,
        )
        if result is None:
            skipped += 1
            continue
        beta, se, nobs = result
        out.append({"naics": naics, "beta_peak": beta, "beta_peak_se": se,
                    "obs_count": nobs})

    if skipped:
        logger.warning(
            "Skipped %d industries with <%d obs at all horizons", skipped, min_obs
        )

    res = pd.DataFrame(out)
    grand = res["beta_peak"].mean()
    res["shrunk_beta"] = (1 - shrinkage) * res["beta_peak"] + shrinkage * grand
    return res
