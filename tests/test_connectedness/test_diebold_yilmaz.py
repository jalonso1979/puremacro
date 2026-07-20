import numpy as np
import pandas as pd

from puremacro.connectedness.diebold_yilmaz import spillover_index


def test_spillover_independent_series_low_total():
    """Two independent processes -> total spillover should be small."""
    rng = np.random.default_rng(11)
    T = 300
    df = pd.DataFrame({
        "y1": rng.standard_normal(T).cumsum(),
        "y2": rng.standard_normal(T).cumsum(),
    })
    out = spillover_index(df, var_lags=2, fevd_horizon=10)
    for k in ("total", "directional_to", "directional_from", "net", "pairwise_matrix"):
        assert k in out
    # Total spillover (in %) should be low for independent series
    assert out["total"] < 30


def test_spillover_correlated_series_high_total():
    """Two highly co-moving series (driven by a common factor) -> high spillover."""
    rng = np.random.default_rng(13)
    T = 300
    common = rng.standard_normal(T).cumsum()
    df = pd.DataFrame({
        "y1": common + 0.3 * rng.standard_normal(T),
        "y2": common + 0.3 * rng.standard_normal(T),
    })
    out = spillover_index(df, var_lags=2, fevd_horizon=10)
    assert out["total"] > 30  # most variance comes from "shared" identification
