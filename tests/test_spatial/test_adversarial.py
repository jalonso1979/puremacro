"""Adversarial inputs: duplicate coordinates, disconnected graphs, degenerate panels and designs."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.bartik import shift_share_iv
from puremacro.spatial import (
    conley_cov,
    contiguity_weights,
    distance_weights,
    gearys_c,
    knn_weights,
    morans_i,
    spatial_hac_panel_meat,
)

DUP = np.array([[40.0, -3.0], [40.0, -3.0], [41.0, -2.0], [42.0, -1.0]])


def test_duplicate_coordinates_give_finite_weights_and_errors():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        W = distance_weights(DUP, cutoff=300.0, decay="inverse", row_standardize=False)
    assert any("same coordinates" in str(w.message) for w in rec)
    dense = W.to_dense()
    assert np.all(np.isfinite(dense)) and dense[0, 1] == dense.max()
    Wk = knn_weights(DUP, k=2)
    assert list(Wk.n_neighbors) == [2, 2, 2, 2]
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(4), rng.standard_normal(4)])
    cov = conley_cov(X, rng.standard_normal(4), DUP, 50.0)
    assert np.all(np.isfinite(cov)) and np.all(np.diag(cov) > 0)


def test_non_finite_or_malformed_coordinates_raise():
    with pytest.raises(ValueError, match="finite"):
        distance_weights(np.array([[np.nan, 0.0], [1.0, 1.0]]), cutoff=5.0, metric="euclidean")
    with pytest.raises(ValueError, match="two coordinates"):
        distance_weights(np.arange(5.0), cutoff=2.0, metric="euclidean")
    with pytest.raises(ValueError, match="cutoff"):
        conley_cov(np.ones((4, 1)), np.ones(4), DUP, -1.0)


def test_cutoff_below_the_minimum_distance_leaves_only_islands():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        W = distance_weights(DUP[2:], cutoff=10.0)
    assert W.n_islands == 2 and any("island" in str(w.message) for w in rec)
    assert np.all(W.lag(np.array([1.0, 2.0])) == 0.0)


def test_disconnected_graph_still_yields_moran_and_geary():
    W = contiguity_weights({"a": ["b"], "c": ["d"], "e": ["f"]})
    x = pd.Series({"a": 1.0, "b": 2.0, "c": 5.0, "d": 6.0, "e": 9.0, "f": 10.0})  # labelled: order-independent
    res = morans_i(x, W, n_perm=99)
    assert np.isfinite(res.I) and res.I > 0.9
    assert np.isfinite(gearys_c(x, W, n_perm=99).C)
    with pytest.raises(ValueError, match="at least 4"):
        morans_i(np.array([1.0, 2.0, 3.0]), contiguity_weights({"a": ["b"], "b": ["c"]}), n_perm=0)


def test_single_period_and_unbalanced_panels():
    rng = np.random.default_rng(1)
    coords = pd.DataFrame(rng.uniform(0, 10, (5, 2)), index=list("abcde"), columns=["x", "y"])
    ents = np.array(list("abcde"))
    X = np.column_stack([np.ones(5), rng.standard_normal(5)])
    e = rng.standard_normal(5)
    one_period = np.zeros(5, dtype=int)
    S3 = spatial_hac_panel_meat(X, e, coords, ents, one_period, 4.0, 3, metric="euclidean")
    S0 = spatial_hac_panel_meat(X, e, coords, ents, one_period, 4.0, 0, metric="euclidean")
    np.testing.assert_allclose(S3, S0)  # no other period to correlate with
    XtXi = np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(XtXi @ S0 @ XtXi, conley_cov(X, e, coords.to_numpy(), 4.0, metric="euclidean"))
    unbalanced = spatial_hac_panel_meat(X, e, coords, ents, np.array([0, 0, 1, 1, 2]), 4.0, 1, metric="euclidean")
    assert np.all(np.isfinite(unbalanced)) and np.allclose(unbalanced, unbalanced.T)


def test_shift_share_degenerate_designs():
    rng = np.random.default_rng(2)
    n, K = 120, 6
    S_full = rng.dirichlet(np.ones(K), size=n)      # rows sum to one
    S = S_full.copy()
    S[:, 0] = 0.0                                   # a sector nobody is exposed to
    g = rng.standard_normal(K)
    x = S @ g + rng.standard_normal(n)
    df = pd.DataFrame({"y": x + rng.standard_normal(n), "x": x})
    res = shift_share_iv(df, "y", "x", S, g)
    assert res.rotemberg_weights.iloc[0] == 0.0
    assert np.isfinite(res.se_akm) and res.se_akm > 0
    assert shift_share_iv(df, "y", "x", 3 * S, g).beta == pytest.approx(res.beta)  # scale-free in shares
    with pytest.raises(ValueError, match="no variation"):
        shift_share_iv(df, "y", "x", S_full, np.ones(K))   # constant shocks: z is collinear with the intercept
    with pytest.raises(ValueError, match="finite"):
        shift_share_iv(df, "y", "x", np.where(S == 0, np.nan, S), g)
    with pytest.raises(ValueError, match="weights"):
        shift_share_iv(df, "y", "x", S, g, weights=-np.ones(n))
