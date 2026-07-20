import numpy as np
import pandas as pd
import pytest

from puremacro.lp._panel_helpers import two_way_fe_within


def _make_balanced_panel(rng, N=8, T=40, beta=0.5):
    """Balanced panel with entity + time FE and a true slope of `beta`."""
    rows = []
    entity_fe = rng.standard_normal(N) * 2
    time_fe = rng.standard_normal(T)
    for i in range(N):
        x = rng.standard_normal(T)
        y = entity_fe[i] + time_fe + beta * x + rng.standard_normal(T) * 0.5
        for t in range(T):
            rows.append({"code": f"E{i:02d}", "date": pd.Timestamp("2010-01-01") + pd.DateOffset(months=t),
                         "x": x[t], "y": y[t]})
    return pd.DataFrame(rows).set_index(["code", "date"]).sort_index()


def test_two_way_fe_recovers_slope():
    rng = np.random.default_rng(7)
    df = _make_balanced_panel(rng, N=10, T=60, beta=0.7)
    out = two_way_fe_within(df, y_col="y", x_cols=["x"])
    assert abs(out["beta"][0] - 0.7) < 0.05  # within 5pp of true slope
    assert {"beta", "se", "t", "vcov", "residuals", "n_obs",
            "entity_keys", "time_keys"}.issubset(out.keys())


def test_two_way_fe_parity_with_linearmodels():
    rng = np.random.default_rng(11)
    df = _make_balanced_panel(rng, N=8, T=50, beta=0.4)
    out = two_way_fe_within(df, y_col="y", x_cols=["x"])
    # Reference: linearmodels PanelOLS with entity + time FE, balanced panel.
    from linearmodels.panel import PanelOLS
    ref = PanelOLS.from_formula("y ~ x + EntityEffects + TimeEffects", data=df).fit()
    # PanelOLS prefixes parameter names; for a single regressor it's just 'x'.
    ref_beta = float(ref.params["x"])
    assert abs(out["beta"][0] - ref_beta) < 1e-6, f"beta mismatch: {out['beta'][0]} vs {ref_beta}"
