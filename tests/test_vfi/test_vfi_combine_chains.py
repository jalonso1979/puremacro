from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi import VFIProblem, stationary_distribution, tauchen
from puremacro.vfi.discretize import combine_markov_chains, rouwenhorst


def _stationary(P):
    w, V = np.linalg.eig(P.T)
    v = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    return v / v.sum()


def test_single_chain_identity():
    g, P = tauchen(4, 0.8, 0.2)
    values, Pc = combine_markov_chains((g, P))
    np.testing.assert_allclose(values, g[:, None])       # (n,1) value column
    np.testing.assert_allclose(Pc, P)


def test_two_chains_kron_shape_and_enumeration():
    g1, P1 = tauchen(3, 0.7, 0.2)
    g2, P2 = rouwenhorst(4, 0.5, 0.1)
    values, Pc = combine_markov_chains((g1, P1), (g2, P2))
    assert values.shape == (12, 2)
    assert Pc.shape == (12, 12)
    np.testing.assert_allclose(Pc.sum(1), 1.0, atol=1e-12)
    np.testing.assert_allclose(Pc, np.kron(P1, P2), atol=1e-12)
    n2 = 4
    for k in range(12):                                  # state k = i*n2+j -> (g1[i], g2[j])
        i, j = k // n2, k % n2
        np.testing.assert_allclose(values[k], [g1[i], g2[j]])


def test_independent_stationary_is_kron_of_marginals():
    g1, P1 = tauchen(3, 0.7, 0.2)
    g2, P2 = tauchen(4, 0.6, 0.15)
    _, Pc = combine_markov_chains((g1, P1), (g2, P2))
    np.testing.assert_allclose(_stationary(Pc),
                               np.kron(_stationary(P1), _stationary(P2)), atol=1e-10)


def test_three_chains_shape_and_stochastic():
    chains = [tauchen(2, 0.5, 0.2), tauchen(3, 0.4, 0.1), rouwenhorst(2, 0.3, 0.1)]
    values, Pc = combine_markov_chains(*chains)
    assert values.shape == (12, 3)
    assert Pc.shape == (12, 12)
    np.testing.assert_allclose(Pc.sum(1), 1.0, atol=1e-12)


def test_validation():
    with pytest.raises(ValueError, match="at least one"):
        combine_markov_chains()
    bad_rows = np.array([[0.5, 0.6], [0.5, 0.5]])        # row 0 sums to 1.1
    with pytest.raises(ValueError, match="row-stochastic"):
        combine_markov_chains((np.array([0.0, 1.0]), bad_rows))
    with pytest.raises(ValueError, match="shape"):        # P not (n,n) vs grid
        combine_markov_chains((np.array([0.0, 1.0, 2.0]), np.eye(2)))


def test_combined_chain_drives_vfi_persistent_plus_transitory():
    # headline use case: additive log-income = persistent AR(1) + transitory iid.
    gp, Pp = tauchen(5, 0.95, 0.15)     # persistent
    gt, Pt = tauchen(3, 0.0, 0.10)      # transitory (rho=0 -> iid rows)
    values, Pc = combine_markov_chains((gp, Pp), (gt, Pt))
    z_income = values.sum(axis=1)       # log income = persistent + transitory
    a_grid = np.linspace(0.0, 50.0, 120)

    def rf(ap, a, z, xp=np):
        c = 1.03 * a + np.exp(z) - ap
        return xp.where(c > 0.0, -1.0 / xp.maximum(c, 1e-12), -np.inf)

    sol = VFIProblem(a_grid=a_grid, z_grid=z_income, P_z=Pc, return_fn=rf,
                     beta=0.95, options=dict(tol=1e-9, n_howard=30)).solve("numpy")
    assert sol.V.shape == (120, 15)
    mu = stationary_distribution(sol.policy_aprime, Pc)
    np.testing.assert_allclose(mu.sum(), 1.0, atol=1e-10)
    np.testing.assert_allclose(mu.sum(axis=0), _stationary(Pc), atol=1e-8)  # z-marginal invariant


def test_combine_exported():
    from puremacro.vfi import combine_markov_chains as fn

    assert fn is combine_markov_chains
