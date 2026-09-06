"""Spatial weights builders and the SpatialWeights object."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.spatial import (
    SpatialWeights,
    contiguity_weights,
    distance_weights,
    economic_weights,
    haversine_km,
    knn_weights,
    pairwise_distances,
)


def _rook_grid(n: int) -> dict:
    nb: dict = {}
    for i in range(n):
        for j in range(n):
            u = i * n + j
            nb[u] = []
            if i > 0:
                nb[u].append((i - 1) * n + j)
            if i < n - 1:
                nb[u].append((i + 1) * n + j)
            if j > 0:
                nb[u].append(i * n + j - 1)
            if j < n - 1:
                nb[u].append(i * n + j + 1)
    return nb


def test_haversine_matches_known_city_distances():
    london, paris = [51.5074, -0.1278], [48.8566, 2.3522]
    nyc, la = [40.7128, -74.0060], [34.0522, -118.2437]
    assert haversine_km([london], [paris])[0, 0] == pytest.approx(343.6, abs=2.0)
    assert haversine_km([nyc], [la])[0, 0] == pytest.approx(3936.0, abs=6.0)
    D = haversine_km([london, paris])
    assert D[0, 0] == 0.0 and D[0, 1] == pytest.approx(D[1, 0])
    with pytest.raises(ValueError, match="latitude"):
        haversine_km([[51.5074 + 100, -0.1278]])


def test_contiguity_grid_is_symmetric_row_standardised_and_labelled():
    W = contiguity_weights(_rook_grid(4))
    assert W.n == 16
    assert W.ids == tuple(range(16))  # key order, not traversal order: arrays line up with the mapping
    assert W.row_standardized
    assert W.binary().is_symmetric
    np.testing.assert_allclose(W.row_sums(), 1.0)
    assert W.neighbors(0) == {1: 0.5, 4: 0.5}
    assert W.n_islands == 0
    # corner has 2 neighbours, centre has 4
    assert W.n_neighbors[0] == 2 and W.n_neighbors[5] == 4


def test_contiguity_adds_units_seen_only_as_neighbours_and_islands():
    W = contiguity_weights({"a": ["b"], "c": []}, row_standardize=False)
    assert W.ids == ("a", "c", "b")  # keys first, neighbour-only units appended
    assert W.n_islands == 1 and W.islands == ("c",)
    assert W.neighbors("b") == {"a": 1.0}
    with pytest.raises(KeyError):
        contiguity_weights({"a": ["zz"]}, ids=["a", "b"])


def test_knn_picks_nearest_units():
    coords = pd.DataFrame({"x": [0.0, 1.0, 2.0, 10.0, 11.0], "y": 0.0}, index=list("abcde"))
    W = knn_weights(coords, k=2, metric="euclidean")
    assert set(W.neighbors("a")) == {"b", "c"}
    assert set(W.neighbors("e")) == {"d", "c"}
    np.testing.assert_allclose(W.row_sums(), 1.0)
    with pytest.raises(ValueError):
        knn_weights(coords, k=5, metric="euclidean")


def test_distance_weights_decays_and_flags_islands():
    coords = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 3.0], [0.0, 50.0]])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        W = distance_weights(coords, cutoff=2.5, metric="euclidean", decay="inverse", row_standardize=False)
    assert any("island" in str(w.message) for w in rec)
    assert W.n_islands == 1 and W.islands == (3,)
    assert W.neighbors(0) == {1: 1.0}
    assert W.neighbors(1) == pytest.approx({0: 1.0, 2: 0.5})
    Wu = distance_weights(coords, cutoff=2.5, metric="euclidean", decay="uniform", row_standardize=False)
    assert Wu.neighbors(1) == {0: 1.0, 2: 1.0}
    Wg = distance_weights(coords, cutoff=2.5, metric="euclidean", decay="gaussian", bandwidth=1.0, row_standardize=False)
    assert Wg.neighbors(0)[1] == pytest.approx(np.exp(-0.5))
    with pytest.raises(ValueError):
        distance_weights(coords, cutoff=-1.0, metric="euclidean")


def test_economic_weights_are_row_shares_without_self_flows():
    flows = pd.DataFrame([[5, 2, 2], [1, 9, 3], [0, 0, 4]], index=list("abc"), columns=list("abc"), dtype=float)
    W = economic_weights(flows)
    assert W.neighbors("a") == pytest.approx({"b": 0.5, "c": 0.5})
    assert W.neighbors("b") == pytest.approx({"a": 0.25, "c": 0.75})
    assert W.n_islands == 1  # c only flows to itself
    with pytest.raises(ValueError):
        economic_weights(-flows)


def test_lag_aligns_pandas_inputs_by_label():
    W = contiguity_weights({"a": ["b"], "b": ["c"], "c": ["a"]})
    s = pd.Series({"c": 3.0, "a": 1.0, "b": 2.0})
    lag = W.lag(s)
    expected = W.lag(s.loc[list(W.ids)].to_numpy())
    np.testing.assert_allclose(lag.loc[list(W.ids)].to_numpy(), expected)
    with pytest.raises(KeyError):
        W.lag(pd.Series({"a": 1.0, "b": 2.0}))
    df = pd.DataFrame({"u": s, "v": 2 * s})
    lagged = W.lag(df)
    np.testing.assert_allclose(lagged["v"].to_numpy(), 2 * lagged["u"].to_numpy())


def test_weights_object_validation_and_presentation():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy.sparse as sp

    with pytest.raises(ValueError, match="square"):
        SpatialWeights(sp.csr_matrix(np.ones((2, 3))), ids=("a", "b"))
    with pytest.raises(ValueError, match="non-negative"):
        SpatialWeights(sp.csr_matrix(np.array([[0.0, -1.0], [1.0, 0.0]])), ids=("a", "b"))
    W = SpatialWeights(sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]])), ids=("a", "b"))
    assert W.W.diagonal().sum() == 0.0  # self-weights dropped
    W = contiguity_weights(_rook_grid(3))
    assert "SpatialWeights" in W.summary()
    assert W.to_markdown().startswith("|")
    assert "tabular" in W.to_latex()
    assert W.to_typst().startswith("#table(")
    edges = W.to_frame()
    assert set(edges.columns) == {"source", "target", "weight"} and len(edges) == W.W.nnz
    fig = W.plot()
    assert fig is not None
    plt.close("all")
    assert pairwise_distances(np.array([[0.0, 0.0], [3.0, 4.0]]), "euclidean")[0, 1] == pytest.approx(5.0)
