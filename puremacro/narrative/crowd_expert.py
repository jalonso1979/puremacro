"""Crowd vs. Expert Macroeconomic Narrative Divergence Index.

Computes the dynamic spread and lead-lag transmission between public discussion
forums (Reddit, Bluesky, HackerNews) and official institutional communications
(Federal Reserve Beige Book, FOMC Minutes, CBO, Central Bank statements):

    Spread_t = S_crowd,t - S_expert,t

Features:
- Standardized sentiment index alignment.
- Lead-lag cross-correlation analysis to test whether online forum discussions
  front-run or trail official central bank sentiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CrowdExpertSpreadResult:
    """Results from Crowd vs. Expert sentiment analysis."""
    spread_series: pd.Series
    crowd_series: pd.Series
    expert_series: pd.Series
    lead_lag_correlations: pd.Series
    optimal_lead_months: int
    crowd_leads_expert: bool


def compute_crowd_expert_spread(
    crowd_series: pd.Series,
    expert_series: pd.Series,
    *,
    max_lags: int = 6,
    standardize: bool = True,
) -> CrowdExpertSpreadResult:
    """Compute the crowd-expert sentiment spread and lead-lag relationship.

    Parameters
    ----------
    crowd_series : pd.Series
        Monthly/quarterly sentiment index from online public discussions (Reddit, Bluesky, etc.).
    expert_series : pd.Series
        Monthly/quarterly sentiment index from central bank or official publications (Beige Book, FOMC).
    max_lags : int, default 6
        Maximum lead/lag horizon (in periods) to compute cross-correlations.
    standardize : bool, default True
        Standardize both series to zero mean and unit variance before computing spread.

    Returns
    -------
    CrowdExpertSpreadResult
    """
    # Align dates
    df = pd.concat({"crowd": crowd_series, "expert": expert_series}, axis=1).dropna()
    if df.empty or len(df) < 5:
        raise ValueError("Insufficient overlapping observations between crowd and expert series.")

    c = df["crowd"].astype(float)
    e = df["expert"].astype(float)

    if standardize:
        c_std = float(c.std()) or 1.0
        e_std = float(e.std()) or 1.0
        c_norm = (c - float(c.mean())) / c_std
        e_norm = (e - float(e.mean())) / e_std
    else:
        c_norm = c
        e_norm = e

    spread = c_norm - e_norm

    # Compute lead-lag cross-correlations corr(crowd_{t+k}, expert_t)
    # k > 0 means crowd at t leads expert at t+k (corr(crowd_t, expert_{t+k}))
    corr_dict: dict[int, float] = {}
    for k in range(-max_lags, max_lags + 1):
        if k > 0:
            # crowd leads expert by k periods: compare crowd[:-k] with expert[k:]
            c_sub = c_norm.iloc[:-k]
            e_sub = e_norm.iloc[k:]
        elif k < 0:
            # expert leads crowd by |k| periods: compare crowd[|k|:] with expert[:k]
            pos_k = abs(k)
            c_sub = c_norm.iloc[pos_k:]
            e_sub = e_norm.iloc[:-pos_k]
        else:
            c_sub = c_norm
            e_sub = e_norm

        if len(c_sub) >= 3:
            r = float(np.corrcoef(c_sub, e_sub)[0, 1])
            corr_dict[k] = r if not np.isnan(r) else 0.0
        else:
            corr_dict[k] = 0.0

    lead_lag_s = pd.Series(corr_dict, name="cross_correlation")
    lead_lag_s.index.name = "lead_horizon_k"

    # Find peak correlation lag
    best_k = int(lead_lag_s.idxmax())
    crowd_leads = best_k > 0

    return CrowdExpertSpreadResult(
        spread_series=spread,
        crowd_series=c_norm,
        expert_series=e_norm,
        lead_lag_correlations=lead_lag_s,
        optimal_lead_months=best_k,
        crowd_leads_expert=crowd_leads,
    )


__all__ = [
    "CrowdExpertSpreadResult",
    "compute_crowd_expert_spread",
]
