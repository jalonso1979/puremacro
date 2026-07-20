import numpy as np
import pandas as pd

from puremacro.lp.cce import cce_panel_lp


def test_cce_recovers_slope_with_common_factor():
    """DGP with a strong common factor: OLS panel LP would be biased,
    but CCE should be approximately consistent."""
    rng = np.random.default_rng(11)
    N, T = 12, 80
    factor = rng.standard_normal(T)
    rows = []
    for i in range(N):
        load_y, load_x = rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5)
        x = load_x * factor + rng.standard_normal(T)
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.5 * x[t-1] + load_y * factor[t] + rng.standard_normal() * 0.3
        for t in range(T):
            rows.append({"code": f"E{i:02d}",
                         "date": pd.Timestamp("2010-01-01") + pd.DateOffset(months=t),
                         "x": x[t], "y": y[t]})
    df = pd.DataFrame(rows).set_index(["code", "date"]).sort_index()
    out = cce_panel_lp(df, y="y", x="x", horizons=[0, 1, 2], n_lags=1)
    # CCE should recover β ≈ 0.5 within 0.2 (with common factor absorbed)
    assert abs(out.loc[out["h"] == 1, "beta"].iloc[0] - 0.5) < 0.25


def test_cce_returns_documented_columns():
    rng = np.random.default_rng(13)
    rows = []
    for i in range(6):
        x = rng.standard_normal(50)
        y = 0.3 * np.roll(x, 1) + rng.standard_normal(50)
        for t in range(50):
            rows.append({"code": f"E{i:02d}",
                         "date": pd.Timestamp("2010-01-01") + pd.DateOffset(months=t),
                         "x": x[t], "y": y[t]})
    df = pd.DataFrame(rows).set_index(["code", "date"]).sort_index()
    out = cce_panel_lp(df, y="y", x="x", horizons=range(0, 4), n_lags=1)
    assert {"h", "beta", "se", "lo", "hi"}.issubset(out.columns)
