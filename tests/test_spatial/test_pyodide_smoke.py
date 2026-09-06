"""Browser-kernel smoke test for the spatial package: four-package core only,
tiny inputs, no files, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.bartik import shift_share_iv
from puremacro.lp import panel_lp
from puremacro.spatial import conley_se, contiguity_weights, distance_weights, morans_i


@pytest.mark.pyodide_smoke
def test_spatial_pipeline_runs_on_tiny_inputs():
    rng = np.random.default_rng(0)
    coords = pd.DataFrame(
        {"lat": [40.4, 41.4, 39.5, 37.4, 41.6, 36.7], "lon": [-3.7, 2.2, -0.4, -6.0, -0.9, -4.4]},
        index=list("ABCDEF"),
    )
    W = distance_weights(coords, cutoff=450.0)
    assert W.n == 6 and np.all(np.isfinite(W.to_dense()))
    grid = contiguity_weights({0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]})
    res = morans_i(np.array([1.0, 2.0, 1.5, 2.5]), grid, n_perm=19)
    assert np.isfinite(res.I) and 0.0 <= res.p_sim <= 1.0

    n = 30
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    xy = rng.uniform([36.0, -9.0], [43.5, 3.0], size=(n, 2))
    se = conley_se(X, rng.standard_normal(n), xy, 200.0)
    assert se.shape == (2,) and np.all(se > 0)

    T = 12
    rows = [
        {"code": c, "date": t, "y": float(v), "x": float(u)}
        for c in coords.index
        for t, (v, u) in enumerate(zip(np.cumsum(rng.standard_normal(T)), rng.standard_normal(T)))
    ]
    panel = pd.DataFrame(rows).set_index(["code", "date"])
    irf = panel_lp(panel, "y", "x", horizons=[0, 1], n_lags=1, cov_type="conley", coords=coords, cutoff_km=400.0)
    assert list(irf.columns) == ["h", "beta", "se", "t", "lo", "hi"] and np.all(irf["se"] > 0)

    S = rng.dirichlet(np.ones(4), size=40)
    g = rng.standard_normal(4)
    x = S @ g + rng.standard_normal(40)
    ss = shift_share_iv(pd.DataFrame({"y": x + rng.standard_normal(40), "x": x}), "y", "x", S, g)
    assert np.isfinite(ss.beta) and ss.se_akm > 0 and ss.rotemberg_weights.sum() == pytest.approx(1.0)
