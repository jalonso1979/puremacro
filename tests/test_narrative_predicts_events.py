"""beta.2 synthetic-DGP sanity test: lp_hac recovers planted coefficients
within tolerance on simulated data and returns null-ish coefficients on
independent series. Pure pytest; no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.lp.jorda import lp_hac


def test_lp_hac_recovers_planted_h1_coefficient():
    """y_{t} = 0.5 * x_{t-1} + eps; lp_hac should give beta_hat ~ 0.5 at h=1.

    With lp_hac's spec dy_h = y.shift(-h) - y.shift(1):
        h=1: dy_1 = y_{t+1} - y_{t-1}
                  = 0.5*(x_t - x_{t-2}) + eps_{t+1} - eps_{t-1}
    so coefficient on x_t is 0.5 (the x_L2 collinearity with y_L1 is
    softened by moderate noise on eps).
    """
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2000Q1", periods=n, freq="QS")
    x = pd.Series(rng.normal(size=n), index=idx, name="shock")
    eps = pd.Series(rng.normal(scale=0.5, size=n), index=idx)
    y = pd.Series(np.nan, index=idx, name="y")
    y.iloc[1:] = (0.5 * x.shift(1).iloc[1:].values
                  + eps.iloc[1:].values)
    y = y.dropna()

    df = pd.concat([y, x], axis=1).dropna()
    fit = lp_hac(df, y="y", x="shock", horizons=[1], n_lags=2)
    beta = float(fit.loc[fit["h"] == 1, "beta"].iloc[0])
    assert 0.35 < beta < 0.65, f"recovered beta_hat = {beta}, expected ~ 0.5"


def test_lp_hac_null_dgp_returns_zero_coefficient():
    """y is i.i.d. independent of x; lp_hac should give beta_hat near 0 (|t| < 2)."""
    rng = np.random.default_rng(7)
    n = 200
    idx = pd.date_range("2000Q1", periods=n, freq="QS")
    x = pd.Series(rng.normal(size=n), index=idx, name="shock")
    y = pd.Series(rng.normal(size=n), index=idx, name="y")
    df = pd.concat([y, x], axis=1).dropna()
    fit = lp_hac(df, y="y", x="shock", horizons=[1], n_lags=2)
    beta = float(fit.loc[fit["h"] == 1, "beta"].iloc[0])
    se = float(fit.loc[fit["h"] == 1, "se"].iloc[0])
    t = beta / se if se else float("nan")
    assert abs(t) < 2.0, f"null DGP got |t|={abs(t):.2f}, should be < 2"
