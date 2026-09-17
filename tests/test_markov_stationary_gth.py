"""Stationary distributions of nearly decomposable Markov chains.

Regression for the 4.0.1 behaviour of ``vfi.markov_stationary`` and
``dsge.markov_switching.markov_stationary``: both took the eigenvector of ``P.T`` whose
eigenvalue was closest to 1. When states communicate only through tiny probabilities,
several eigenvalues sit within rounding of 1 and that vector is an arbitrary mixture:
off by 6e-5 on a two-state chain with switching probabilities of 1e-13, and with a
*negative* probability (-0.23) on ``tauchen(7, 0.995, 0.1, m=5)``. The GTH
(Grassmann-Taksar-Heyman) elimination used now only adds non-negative numbers.
"""
import numpy as np
import pytest

from puremacro.dsge.markov_switching import markov_stationary as ms_dsge
from puremacro.vfi import tauchen
from puremacro.vfi.discretize import markov_stationary as ms_vfi

BOTH = pytest.mark.parametrize("markov_stationary", [ms_vfi, ms_dsge], ids=["vfi", "dsge"])


@BOTH
@pytest.mark.parametrize("p", [1e-6, 1e-10, 1e-13, 1e-15])
def test_two_state_closed_form_with_tiny_switching(markov_stationary, p):
    q = 2.0 * p
    P = np.array([[1.0 - p, p], [q, 1.0 - q]])
    # pi = (q, p) / (p + q) exactly, whatever the scale of p and q
    np.testing.assert_allclose(markov_stationary(P), [2.0 / 3.0, 1.0 / 3.0], rtol=0, atol=1e-14)


def _gth_high_precision(P, dps=80):
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = dps
    n = P.shape[0]
    A = [[mpmath.mpf(float(P[i, j])) for j in range(n)] for i in range(n)]
    for k in range(n - 1, 0, -1):
        s = mpmath.fsum(A[k][:k])
        for i in range(k):
            A[i][k] /= s
        for i in range(k):
            for j in range(k):
                A[i][j] += A[i][k] * A[k][j]
    pi = [mpmath.mpf(1)] + [mpmath.mpf(0)] * (n - 1)
    for k in range(1, n):
        pi[k] = mpmath.fsum(pi[i] * A[i][k] for i in range(k))
    total = mpmath.fsum(pi)
    return np.array([float(x / total) for x in pi])


@BOTH
@pytest.mark.parametrize("n,rho,m", [(7, 0.995, 5.0), (3, 0.97, 4.0), (5, 0.995, 3.0)])
def test_persistent_tauchen_chain_is_a_valid_distribution(markov_stationary, n, rho, m):
    _, P = tauchen(n, rho, 0.1, m=m)
    pi = markov_stationary(P)
    assert np.all(pi >= 0.0), f"negative probability {pi.min():.3g}"
    assert pi.sum() == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(pi @ P, pi, rtol=0, atol=1e-13)
    # exact stationary distribution of this floating-point matrix, in 80-digit arithmetic
    np.testing.assert_allclose(pi, _gth_high_precision(P), rtol=0, atol=1e-13)


@BOTH
def test_single_closed_class_puts_all_mass_on_it(markov_stationary):
    # state 1 is absorbing and reachable from state 0: the stationary distribution is unique
    P = np.array([[0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.3, 0.3, 0.4]])
    np.testing.assert_allclose(markov_stationary(P), [0.0, 1.0, 0.0], atol=1e-15)


@BOTH
def test_several_closed_classes_raise(markov_stationary):
    P = np.array([[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0],
                  [0.0, 0.0, 0.2, 0.8], [0.0, 0.0, 0.6, 0.4]])
    with pytest.raises(ValueError, match="not unique"):
        markov_stationary(P)
