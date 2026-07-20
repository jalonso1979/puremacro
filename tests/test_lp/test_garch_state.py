import numpy as np
import pandas as pd

from puremacro.lp.garch_state import lp_garch_state
from puremacro.lp.garch_in_mean import lp_garch_in_mean


def test_lp_garch_state_runs():
    rng = np.random.default_rng(29)
    T = 240
    x = rng.standard_normal(T)
    y = 0.4 * np.roll(x, 1) + rng.standard_normal(T)
    df = pd.DataFrame({"y": y, "x": x},
                      index=pd.date_range("1990-01-01", periods=T, freq="MS"))
    out = lp_garch_state(df, y="y", x="x", horizons=range(0, 4), n_lags=1)
    assert {"beta_H", "beta_L"}.issubset(out.columns)
    assert np.all(np.isfinite(out[["beta_H", "beta_L"]].values))


def test_lp_garch_in_mean_runs():
    rng = np.random.default_rng(31)
    T = 200
    x = rng.standard_normal(T)
    y = -0.3 * np.roll(x, 1) + rng.standard_normal(T)
    df = pd.DataFrame({"y": y, "x": x},
                      index=pd.date_range("1990-01-01", periods=T, freq="MS"))
    out = lp_garch_in_mean(df, y="y", x="x", horizons=range(0, 4), n_lags=1)
    assert {"h", "beta", "se", "lo", "hi"}.issubset(out.columns)
    assert np.all(np.isfinite(out["beta"]))
