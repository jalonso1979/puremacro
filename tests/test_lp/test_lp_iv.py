import numpy as np
import pandas as pd
import pytest

from puremacro.lp.iv import lp_iv


def test_lp_iv_returns_documented_columns():
    rng = np.random.default_rng(13)
    T = 200
    z = rng.standard_normal(T)              # instrument
    x = 0.7 * z + 0.3 * rng.standard_normal(T)  # endogenous regressor
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] - 0.4 * x[t - 1] + rng.standard_normal()
    df = pd.DataFrame({"x": x, "y": y, "z": z},
                      index=pd.date_range("2000-01-01", periods=T, freq="MS"))
    out = lp_iv(df, y="y", x="x", z="z", horizons=range(0, 6), n_lags=2)
    assert set(out.columns) >= {"h", "beta", "se", "lo", "hi", "first_stage_f"}
    assert len(out) == 6
    assert np.all(np.isfinite(out["se"]))
    # Strong instrument design: F should be sizeable
    assert out["first_stage_f"].iloc[0] > 50


def test_lp_iv_recovers_iv_slope_under_endogeneity():
    """When x is endogenously correlated with the error AND z is a clean
    instrument, OLS LP biases β_1 but LP-IV recovers ≈ -0.4."""
    rng = np.random.default_rng(17)
    T = 400
    z = rng.standard_normal(T)
    eta = rng.standard_normal(T) * 0.5
    x = 0.6 * z + eta
    y = np.zeros(T)
    for t in range(1, T):
        # Endogeneity: error contains a piece of eta — so OLS biases β
        y[t] = -0.4 * x[t - 1] + 0.5 * y[t - 1] + 0.6 * eta[t - 1] + rng.standard_normal() * 0.3
    df = pd.DataFrame({"x": x, "y": y, "z": z},
                      index=pd.date_range("2000-01-01", periods=T, freq="MS"))
    out = lp_iv(df, y="y", x="x", z="z", horizons=[0, 1, 2, 3], n_lags=1)
    # IV should recover ≈ -0.4 at h=1 (within wide tolerance for finite T)
    assert abs(out.loc[out["h"] == 1, "beta"].iloc[0] - (-0.4)) < 0.25
