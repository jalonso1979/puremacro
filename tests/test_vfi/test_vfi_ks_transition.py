from __future__ import annotations

import numpy as np

from puremacro.vfi.krusell_smith import ks_exog_transition


def test_combined_transition_row_stochastic_and_values():
    z_vals = np.array([0.5, 1.1])
    P_z = np.array([[0.6, 0.4], [0.2, 0.8]])
    Z_vals = np.array([0.99, 1.01])
    P_Z = np.array([[0.875, 0.125], [0.125, 0.875]])
    K_grid = np.linspace(30.0, 50.0, 5)
    b0 = np.array([0.2, 0.3])
    b1 = np.array([0.95, 0.95])

    Pc, values = ks_exog_transition(z_vals, P_z, Z_vals, P_Z, K_grid, b0, b1)
    n = 2 * 2 * 5
    assert Pc.shape == (n, n)
    np.testing.assert_allclose(Pc.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(Pc >= -1e-15)
    # values enumerate the C-order product (z, Z, K): flat = (iz*nZ + iZ)*nK + iK
    assert values.shape == (n, 3)
    nZ, nK = 2, 5
    for iz in range(2):
        for iZ in range(2):
            for iK in range(5):
                f = (iz * nZ + iZ) * nK + iK
                np.testing.assert_allclose(values[f], [z_vals[iz], Z_vals[iZ], K_grid[iK]])


def test_K_lottery_mean_preserving_and_marginals():
    z_vals = np.array([1.0])
    P_z = np.array([[1.0]])
    Z_vals = np.array([1.0, 1.0])
    P_Z = np.array([[0.7, 0.3], [0.4, 0.6]])
    K_grid = np.linspace(20.0, 60.0, 9)
    b0 = np.array([0.5, 1.0])
    b1 = np.array([0.9, 0.85])

    Pc, values = ks_exog_transition(z_vals, P_z, Z_vals, P_Z, K_grid, b0, b1)
    # flat index here = iZ*nK + iK (nz=1). The K-marginal next-mean from each
    # (Z,K) row must equal the forecast K'(Z,K) = exp(b0[Z] + b1[Z] log K).
    nK = 9
    for iZ in range(2):
        for iK in range(nK):
            f = iZ * nK + iK
            row = Pc[f]                                  # over (Z', K')
            EKp = float(row @ values[:, 2])
            Kp = np.exp(b0[iZ] + b1[iZ] * np.log(K_grid[iK]))
            Kp = min(max(Kp, K_grid[0]), K_grid[-1])     # clamped to grid range
            np.testing.assert_allclose(EKp, Kp, atol=1e-9)
            # the Z' marginal of the row equals P_Z[iZ]
            zmarg = np.array([row[(iZp * nK):(iZp * nK + nK)].sum() for iZp in range(2)])
            np.testing.assert_allclose(zmarg, P_Z[iZ], atol=1e-12)


def test_ks_transition_exported():
    from puremacro.vfi import ks_exog_transition as fn

    assert fn is ks_exog_transition
