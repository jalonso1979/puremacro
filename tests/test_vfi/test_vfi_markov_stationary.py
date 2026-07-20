from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi import combine_markov_chains, tauchen
from puremacro.vfi.discretize import markov_stationary


def test_two_state_closed_form():
    # P = [[0.9,0.1],[0.2,0.8]] has stationary pi = [2/3, 1/3]
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    pi = markov_stationary(P)
    np.testing.assert_allclose(pi, [2.0 / 3.0, 1.0 / 3.0], atol=1e-12)


def test_is_left_eigenvector_and_simplex():
    _, P = tauchen(n=7, rho=0.9, sigma=0.2)
    pi = markov_stationary(P)
    np.testing.assert_allclose(pi @ P, pi, atol=1e-10)        # pi P = pi
    np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-12)     # on the simplex
    assert np.all(pi >= 0.0)


def test_matches_eig_reference():
    _, P = tauchen(n=5, rho=0.8, sigma=0.15)
    w, V = np.linalg.eig(P.T)
    ref = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    ref = ref / ref.sum()
    np.testing.assert_allclose(markov_stationary(P), ref, atol=1e-12)


def test_kron_chain_stationary_is_product():
    g1, P1 = tauchen(3, 0.7, 0.2)
    g2, P2 = tauchen(4, 0.6, 0.15)
    _, Pc = combine_markov_chains((g1, P1), (g2, P2))
    np.testing.assert_allclose(
        markov_stationary(Pc),
        np.kron(markov_stationary(P1), markov_stationary(P2)),
        atol=1e-10,
    )


def test_validation():
    with pytest.raises(ValueError, match="square"):
        markov_stationary(np.ones((2, 3)))
    with pytest.raises(ValueError, match="row-stochastic"):
        markov_stationary(np.array([[0.5, 0.6], [0.5, 0.5]]))


def test_markov_stationary_exported():
    from puremacro.vfi import markov_stationary as fn

    assert fn is markov_stationary
