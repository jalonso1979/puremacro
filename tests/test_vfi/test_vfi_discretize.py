from __future__ import annotations

import numpy as np

from puremacro.vfi.discretize import rouwenhorst, tauchen


def test_tauchen_shapes_and_rows_sum_to_one():
    grid, P = tauchen(n=7, rho=0.9, sigma=0.1)
    assert grid.shape == (7,)
    assert P.shape == (7, 7)
    np.testing.assert_allclose(P.sum(axis=1), np.ones(7), atol=1e-12)


def test_tauchen_grid_symmetric_and_monotone():
    grid, _ = tauchen(n=5, rho=0.0, sigma=1.0)
    np.testing.assert_allclose(grid, -grid[::-1], atol=1e-12)
    assert np.all(np.diff(grid) > 0)


def test_tauchen_iid_rows_identical():
    # rho=0 => next state independent of current => all rows equal
    _, P = tauchen(n=6, rho=0.0, sigma=0.5)
    for i in range(1, 6):
        np.testing.assert_allclose(P[i], P[0], atol=1e-12)


def test_tauchen_recovers_persistence():
    grid, P = tauchen(n=21, rho=0.8, sigma=0.2, m=3.0)
    w, V = np.linalg.eig(P.T)
    pi = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    pi = pi / pi.sum()
    mean = pi @ grid
    var = pi @ (grid - mean) ** 2
    Ezz = float(sum(pi[i] * P[i, j] * grid[i] * grid[j]
                    for i in range(21) for j in range(21)))
    rho_hat = (Ezz - mean ** 2) / var
    assert abs(rho_hat - 0.8) < 0.05


def test_rouwenhorst_rows_sum_to_one():
    grid, P = rouwenhorst(5, 0.95, 0.1)
    assert grid.shape == (5,) and P.shape == (5, 5)
    np.testing.assert_allclose(P.sum(axis=1), np.ones(5), atol=1e-12)


def test_rouwenhorst_byte_identical_to_companion_copy():
    # Guards against the duplicated implementation drifting.
    from puremacro.models.nested_dmp.kernels import rouwenhorst as rh_nested

    g1, P1 = rouwenhorst(7, 0.9, 0.15)
    g2, P2 = rh_nested(7, 0.9, 0.15)
    np.testing.assert_array_equal(g1, g2)
    np.testing.assert_array_equal(P1, P2)
