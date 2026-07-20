import numpy as np
import pandas as pd

from puremacro.lp.state_dep import lp_state_dep


def test_lp_state_dep_returns_two_regimes():
    rng = np.random.default_rng(17)
    T = 240
    state = rng.standard_normal(T)
    x = rng.standard_normal(T)
    y = -0.2 * x + 0.4 * state * x + rng.standard_normal(T)
    df = pd.DataFrame({"y": y, "x": x, "state": state},
                      index=pd.date_range("1990-01-01", periods=T, freq="MS"))
    out = lp_state_dep(df, y="y", x="x", state="state",
                       horizons=range(0, 6), n_lags=2)
    assert {"beta_H", "beta_L", "se_H", "se_L"}.issubset(out.columns)
    assert np.all(np.isfinite(out[["beta_H", "beta_L"]].values))
    assert len(out) == 6


def test_lp_state_dep_threshold_mode():
    rng = np.random.default_rng(19)
    T = 200
    state = rng.standard_normal(T)
    x = rng.standard_normal(T)
    y = (state > 0).astype(float) * 0.5 * x + (state <= 0).astype(float) * (-0.3) * x + rng.standard_normal(T) * 0.2
    df = pd.DataFrame({"y": y, "x": x, "state": state},
                      index=pd.date_range("1990-01-01", periods=T, freq="MS"))
    out = lp_state_dep(df, y="y", x="x", state="state",
                       horizons=[0], n_lags=0, transition="threshold")
    # With clean state, beta_H (high state) ≈ 0.5, beta_L (low) ≈ -0.3
    assert out["beta_H"].iloc[0] > 0.2
    assert out["beta_L"].iloc[0] < 0.0
