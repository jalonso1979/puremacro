"""A local projection must not depend on the row order of its input.

`lp_hac` builds leads and lags with positional `.shift()`, so the row order IS
the time order as far as it is concerned. `mean_group_panel_lp` did not sort,
while `_panel_helpers.panel_lp_horizon_loop` and `cce.cce_panel_lp` both do —
so of the three panel-LP paths in this package, one silently disagreed with the
other two on the same data.

The failure is quiet and severe: on the panel below, whose true h=1 response is
0.8, the sorted answer is 0.765 and the identical data with its rows shuffled
gives 0.055. No error, no warning, and an attenuated-toward-zero coefficient is
exactly the shape a reader expects from a noisy estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from puremacro.lp.mean_group import mean_group_panel_lp


def _ar1_with_shock(n_units: int = 12, n_periods: int = 80, seed: int = 0):
    """y_t = 0.5 y_{t-1} + 0.8 x_{t-1} + e_t, so the h=1 response is 0.8."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        x = rng.standard_normal(n_periods)
        e = rng.standard_normal(n_periods)
        y = np.zeros(n_periods)
        for t in range(1, n_periods):
            y[t] = 0.5 * y[t - 1] + 0.8 * x[t - 1] + e[t]
        for t in range(n_periods):
            rows.append({"code": f"c{i}", "date": t, "y": y[t], "x": x[t]})
    return pd.DataFrame(rows).set_index(["code", "date"])


def test_mean_group_is_invariant_to_input_row_order():
    horizons = [0, 1, 2, 3]
    df = _ar1_with_shock()

    ordered = mean_group_panel_lp(df.sort_index(), y="y", x="x",
                                  horizons=horizons, n_lags=2)
    shuffled = mean_group_panel_lp(df.sample(frac=1.0, random_state=3),
                                   y="y", x="x", horizons=horizons, n_lags=2)

    for h in horizons:
        a = float(ordered.loc[ordered.h == h, "beta"].iloc[0])
        b = float(shuffled.loc[shuffled.h == h, "beta"].iloc[0])
        np.testing.assert_allclose(
            b, a, rtol=1e-9, atol=1e-12,
            err_msg=(
                f"h={h}: shuffling the rows of the SAME data moved beta from "
                f"{a:+.6f} to {b:+.6f}"
            ),
        )


def test_mean_group_recovers_the_known_impact():
    """Anchor the level too, so the invariance above cannot hold vacuously."""
    df = _ar1_with_shock()
    res = mean_group_panel_lp(df, y="y", x="x", horizons=[1], n_lags=2)
    beta = float(res.loc[res.h == 1, "beta"].iloc[0])
    assert abs(beta - 0.8) < 0.15, f"h=1 response {beta:+.4f}, expected ~0.80"
