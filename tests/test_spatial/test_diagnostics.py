"""Moran's I and Geary's C: signs on known patterns and agreement with esda."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.spatial import contiguity_weights, gearys_c, morans_i

from .test_weights import _rook_grid


@pytest.fixture(scope="module")
def grid():
    n = 6
    nb = _rook_grid(n)
    W = contiguity_weights(nb)
    checker = np.array([(i + j) % 2 for i in range(n) for j in range(n)], dtype=float)
    gradient = np.array([i + j for i in range(n) for j in range(n)], dtype=float)
    return nb, W, checker, gradient


def test_moran_sign_on_checkerboard_and_gradient(grid):
    _, W, checker, gradient = grid
    neg = morans_i(checker, W, n_perm=199, seed=1)
    pos = morans_i(gradient, W, n_perm=199, seed=1)
    assert neg.I < neg.expected and neg.p_sim < 0.05
    assert pos.I > 0.5 and pos.p_sim < 0.05
    assert neg.expected == pytest.approx(-1.0 / 35)
    assert 0.0 < neg.p_norm < 0.05 and 0.0 < neg.p_rand < 0.05
    c_neg = gearys_c(checker, W, n_perm=199, seed=1)
    c_pos = gearys_c(gradient, W, n_perm=199, seed=1)
    assert c_neg.C > 1.0 and c_pos.C < 1.0


@pytest.mark.reference  # live esda cross-check; the default suite uses the frozen golden in validation/cases_spatial
def test_moran_and_geary_match_esda(grid):
    esda = pytest.importorskip("esda")
    libpysal = pytest.importorskip("libpysal")
    nb, W, checker, gradient = grid
    w = libpysal.weights.W({u: list(v) for u, v in nb.items()})
    w.transform = "r"
    order = list(w.id_order)  # libpysal may reorder ids; align the reference input
    for x in (checker, gradient, np.sin(np.arange(36) / 3.0)):
        x_ref = pd.Series(x, index=W.ids).loc[order].to_numpy()
        ref = esda.Moran(x_ref, w, permutations=0)
        ours = morans_i(x, W, n_perm=0)
        assert ours.I == pytest.approx(ref.I, abs=1e-10)
        assert ours.expected == pytest.approx(ref.EI, abs=1e-12)
        assert ours.variance_norm == pytest.approx(ref.VI_norm, rel=1e-8)
        assert ours.variance_rand == pytest.approx(ref.VI_rand, rel=1e-8)
        assert ours.z_norm == pytest.approx(ref.z_norm, rel=1e-8)
        assert ours.z_rand == pytest.approx(ref.z_rand, rel=1e-8)
        refc = esda.Geary(x_ref, w, permutations=0)
        oursc = gearys_c(x, W, n_perm=0)
        assert oursc.C == pytest.approx(refc.C, abs=1e-10)
        assert oursc.variance_norm == pytest.approx(refc.VC_norm, rel=1e-8)
        assert oursc.variance_rand == pytest.approx(refc.VC_rand, rel=1e-8)
        assert oursc.z_norm == pytest.approx(refc.z_norm, rel=1e-8)
        assert oursc.z_rand == pytest.approx(refc.z_rand, rel=1e-8)


def test_moran_accepts_labelled_series_and_validates(grid):
    _, W, _, gradient = grid
    s = pd.Series(gradient, index=W.ids).sample(frac=1.0, random_state=0)  # shuffled labels
    res = morans_i(s, W, n_perm=0)
    assert res.I == pytest.approx(morans_i(gradient, W, n_perm=0).I)
    with pytest.raises(KeyError):
        morans_i(s.iloc[:-1], W, n_perm=0)
    with pytest.raises(ValueError, match="constant"):
        morans_i(np.ones(36), W, n_perm=0)
    with pytest.raises(ValueError):
        morans_i(np.arange(36.0), contiguity_weights({"a": ["b"], "b": ["c"]}), n_perm=0)


def test_diagnostics_presentation(grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, W, _, gradient = grid
    for res in (morans_i(gradient, W, n_perm=49), gearys_c(gradient, W, n_perm=49)):
        assert res.summary()
        assert res.to_markdown().startswith("|")
        assert "tabular" in res.to_latex()
        assert res.to_typst().startswith("#table(")
        frame = res.to_frame()
        assert {"statistic", "value"} <= set(frame.columns)
        assert res.plot() is not None
        plt.close("all")
