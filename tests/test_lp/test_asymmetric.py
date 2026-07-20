import numpy as np
import pandas as pd

from puremacro.lp.asymmetric import lp_asymmetric


def test_lp_asymmetric_recovers_known_asymmetry():
    """DGP: y_{t+1} = β_+ * x_t * 1{x>0} + β_- * x_t * 1{x<0} + ε
    with β_+ = 0.5 and β_- = -1.2 (positive shocks weakly raise y,
    negative shocks strongly raise y in absolute terms)."""
    rng = np.random.default_rng(11)
    T = 400
    x = rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        if x[t-1] > 0:
            y[t] = 0.5 * x[t-1] + rng.standard_normal() * 0.3
        else:
            y[t] = -1.2 * x[t-1] + rng.standard_normal() * 0.3
    df = pd.DataFrame({"y": y, "x": x},
                      index=pd.date_range("2000-01-01", periods=T, freq="MS"))
    out = lp_asymmetric(df, y="y", x="x", horizons=[0, 1, 2], n_lags=1)
    # h=1 should recover the asymmetry within 0.3 of true betas
    h1 = out[out["h"] == 1].iloc[0]
    assert abs(h1["beta_pos"] - 0.5) < 0.3
    assert abs(h1["beta_neg"] - (-1.2)) < 0.3


def test_lp_asymmetric_returns_documented_columns():
    rng = np.random.default_rng(13)
    T = 200
    x = rng.standard_normal(T)
    y = -0.4 * x + rng.standard_normal(T)
    df = pd.DataFrame({"y": y, "x": x},
                      index=pd.date_range("2000-01-01", periods=T, freq="MS"))
    out = lp_asymmetric(df, y="y", x="x", horizons=range(0, 5), n_lags=1)
    expected = {"h", "beta_pos", "se_pos", "lo_pos", "hi_pos",
                "beta_neg", "se_neg", "lo_neg", "hi_neg"}
    assert expected.issubset(out.columns)
    assert len(out) == 5
