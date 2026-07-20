import numpy as np
import pandas as pd
import pytest

from puremacro.lp.jorda import lp_hac


@pytest.mark.pyodide_smoke
def test_lp_hac_returns_documented_columns():
    rng = np.random.default_rng(11)
    T = 200
    x = rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] - 0.3 * x[t - 1] + rng.standard_normal()
    df = pd.DataFrame({"x": x, "y": y},
                      index=pd.date_range("2000-01-01", periods=T, freq="MS"))
    out = lp_hac(df, y="y", x="x", horizons=range(0, 8), n_lags=2, alpha=0.10)
    assert set(out.columns) >= {"h", "beta", "se", "lo", "hi"}
    assert len(out) == 8
    assert np.all(np.isfinite(out["se"]))


def test_lp_hac_recovers_known_response():
    """For a DGP y_t = -0.4 x_{t-1} + 0.5 y_{t-1} + e_t, the LP β at h=1
    should be ≈ -0.4."""
    rng = np.random.default_rng(13)
    T = 400
    x = rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = -0.4 * x[t - 1] + 0.5 * y[t - 1] + rng.standard_normal() * 0.5
    df = pd.DataFrame({"x": x, "y": y},
                      index=pd.date_range("2000-01-01", periods=T, freq="MS"))
    out = lp_hac(df, y="y", x="x", horizons=[0, 1, 2, 3], n_lags=1)
    # h=1: change y_{t+1} - y_{t-1} per unit x_t. With y_{t+1} = -0.4 x_t + ...
    # the LP captures β_1 ≈ -0.4 (give or take noise).
    assert abs(out.loc[out["h"] == 1, "beta"].iloc[0] - (-0.4)) < 0.15
