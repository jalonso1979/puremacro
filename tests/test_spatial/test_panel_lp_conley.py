"""cov_type='conley' on the two-way FE panel local projection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.lp import panel_lp, panel_lp_dk


@pytest.fixture(scope="module")
def regional_panel():
    rng = np.random.default_rng(7)
    ents = [f"R{i:02d}" for i in range(10)]
    T = 60
    coords = pd.DataFrame(
        {"lat": rng.uniform(30.0, 45.0, len(ents)), "lon": rng.uniform(-110.0, -75.0, len(ents))},
        index=ents,
    )
    rows = []
    common = rng.standard_normal(T)
    for en in ents:
        x = rng.standard_normal(T)
        y = np.cumsum(0.4 * x + 0.6 * common + rng.standard_normal(T))
        for t in range(T):
            rows.append({"code": en, "date": t, "x": x[t], "y": y[t]})
    return pd.DataFrame(rows).set_index(["code", "date"]), coords


def test_conley_runs_and_returns_the_lp_result_contract(regional_panel):
    df, coords = regional_panel
    res = panel_lp(df, "y", "x", horizons=range(0, 5), n_lags=1, cov_type="conley", coords=coords, cutoff_km=500.0)
    assert list(res.columns) == ["h", "beta", "se", "t", "lo", "hi"]
    assert np.all(np.isfinite(res["se"])) and np.all(res["se"] > 0)
    assert np.all(res["lo"] <= res["beta"]) and np.all(res["beta"] <= res["hi"])
    # the point estimates do not depend on the covariance type
    base = panel_lp(df, "y", "x", horizons=range(0, 5), n_lags=1)
    np.testing.assert_allclose(res["beta"].to_numpy(), base["beta"].to_numpy())


def test_conley_with_infinite_cutoff_is_driscoll_kraay(regional_panel):
    df, coords = regional_panel
    res_c = panel_lp(df, "y", "x", horizons=range(0, 4), n_lags=1, cov_type="conley",
                     coords=coords, cutoff_km=1e9, kernel="uniform")
    res_dk = panel_lp_dk(df, "y", "x", horizons=range(0, 4), n_lags=1)
    np.testing.assert_allclose(res_c["se"].to_numpy(), res_dk["se"].to_numpy(), rtol=1e-10)


def test_conley_time_lags_and_cutoff_change_the_errors(regional_panel):
    df, coords = regional_panel
    a = panel_lp(df, "y", "x", horizons=[2], n_lags=1, cov_type="conley", coords=coords, cutoff_km=200.0, time_lags=0)
    b = panel_lp(df, "y", "x", horizons=[2], n_lags=1, cov_type="conley", coords=coords, cutoff_km=200.0, time_lags=6)
    c = panel_lp(df, "y", "x", horizons=[2], n_lags=1, cov_type="conley", coords=coords, cutoff_km=5000.0, time_lags=0)
    assert a["se"].iloc[0] != b["se"].iloc[0]
    assert a["se"].iloc[0] != c["se"].iloc[0]


def test_conley_requires_coords_and_cutoff(regional_panel):
    df, coords = regional_panel
    with pytest.raises(ValueError, match="coords"):
        panel_lp(df, "y", "x", horizons=[0], cov_type="conley")
    with pytest.raises(KeyError):
        panel_lp(df, "y", "x", horizons=[0], cov_type="conley", coords=coords.iloc[:-1], cutoff_km=100.0)
    with pytest.raises(ValueError, match="cov_type"):
        panel_lp(df, "y", "x", horizons=[0], cov_type="spherical")
