import numpy as np
import pandas as pd

from puremacro.lp.mean_group import mean_group_panel_lp


def test_mean_group_recovers_average_slope():
    """Heterogeneous betas across entities. MG should recover the
    cross-sectional MEAN of the true β's."""
    rng = np.random.default_rng(11)
    N, T = 8, 100
    rows = []
    true_betas = rng.uniform(0.2, 0.8, size=N)
    for i, b in enumerate(true_betas):
        x = rng.standard_normal(T)
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = b * x[t-1] + rng.standard_normal() * 0.3
        for t in range(T):
            rows.append({"code": f"E{i:02d}",
                         "date": pd.Timestamp("2010-01-01") + pd.DateOffset(months=t),
                         "x": x[t], "y": y[t]})
    df = pd.DataFrame(rows).set_index(["code", "date"]).sort_index()
    out = mean_group_panel_lp(df, y="y", x="x", horizons=[0, 1, 2], n_lags=1)
    # h=1 β should be ≈ mean(true_betas)
    target = np.mean(true_betas)
    assert abs(out.loc[out["h"] == 1, "beta"].iloc[0] - target) < 0.15


def test_mean_group_returns_n_entities_used():
    rng = np.random.default_rng(13)
    rows = []
    for i in range(5):
        x = rng.standard_normal(50)
        y = 0.4 * x + rng.standard_normal(50)
        for t in range(50):
            rows.append({"code": f"E{i:02d}",
                         "date": pd.Timestamp("2010-01-01") + pd.DateOffset(months=t),
                         "x": x[t], "y": y[t]})
    df = pd.DataFrame(rows).set_index(["code", "date"]).sort_index()
    out = mean_group_panel_lp(df, y="y", x="x", horizons=range(0, 4), n_lags=1)
    assert "n_entities_used" in out.columns
    assert out["n_entities_used"].iloc[0] == 5
