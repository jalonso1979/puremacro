"""Tests for crowd vs. expert sentiment divergence index."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.narrative.crowd_expert import compute_crowd_expert_spread


def test_compute_crowd_expert_spread():
    dates = pd.date_range("2023-01-01", periods=18, freq="MS")
    # Crowd sentiment leads expert sentiment by 2 months
    signal = np.sin(np.linspace(0, 3 * np.pi, 18))
    crowd = pd.Series(signal, index=dates)
    expert = pd.Series(np.roll(signal, 2), index=dates)  # expert lags by 2 months
    
    res = compute_crowd_expert_spread(crowd, expert, max_lags=4, standardize=True)
    
    assert len(res.spread_series) == 18
    assert res.optimal_lead_months == 2
    assert res.crowd_leads_expert is True
    assert res.lead_lag_correlations.loc[2] > 0.8
