import numpy as np
import pandas as pd

from puremacro.lp.smooth import lp_smooth


def test_lp_smooth_returns_smoothed_irf():
    rng = np.random.default_rng(23)
    T = 240
    x = rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(5, T):
        y[t] = 0.5 * x[t-1] - 0.3 * x[t-4] + rng.standard_normal() * 0.4
    df = pd.DataFrame({"y": y, "x": x},
                      index=pd.date_range("1990-01-01", periods=T, freq="MS"))
    out = lp_smooth(df, y="y", x="x", horizons=range(0, 12), n_lags=2)
    assert {"h", "beta", "se", "lo", "hi", "lambda"}.issubset(out.columns)
    # Smoothed beta should not have wild horizon-to-horizon jumps
    assert out["beta"].diff().abs().max() < 1.0
    assert out["lambda"].iloc[0] > 0


def test_lp_smooth_explicit_lambda():
    """Passing lambda_ skips GCV; output is monotone in lambda for very large values."""
    rng = np.random.default_rng(29)
    T = 200
    x = rng.standard_normal(T)
    y = 0.5 * np.roll(x, 1) + rng.standard_normal(T) * 0.5
    df = pd.DataFrame({"y": y, "x": x},
                      index=pd.date_range("1990-01-01", periods=T, freq="MS"))
    out_low = lp_smooth(df, y="y", x="x", horizons=range(0, 8), n_lags=1, lambda_=0.001)
    out_high = lp_smooth(df, y="y", x="x", horizons=range(0, 8), n_lags=1, lambda_=1e6)
    # Very high lambda → smoothed beta should be nearly constant across horizons
    assert out_high["beta"].std() < out_low["beta"].std()
