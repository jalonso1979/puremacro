from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from puremacro.models.nested_dmp import kernels as K

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("numba") is None, reason="numba not installed"
)


def _row_stochastic(n, rng):
    A = rng.random((n, n)) + 1e-3
    return A / A.sum(axis=1, keepdims=True)


def test_numba_bellman_matches_numpy_oracle(rng):
    from puremacro.models.nested_dmp import kernels_numba as KN

    n, beta = 8, 0.96
    P = _row_stochastic(n, rng)
    flow = rng.standard_normal(n)
    oracle = K.bellman_iterate(flow, P, beta, tol=1e-12, max_iter=100_000)
    got = KN.bellman_iterate_numba(flow, P, beta, 1e-12, 100_000)
    np.testing.assert_allclose(got, oracle, rtol=1e-9, atol=1e-12)


def test_numba_batch_matches_looped_single(rng):
    from puremacro.models.nested_dmp import kernels_numba as KN

    B, n, beta = 5, 7, 0.95
    flow_batch = rng.standard_normal((B, n))
    P_batch = np.stack([_row_stochastic(n, rng) for _ in range(B)])
    got = KN.bellman_iterate_batch_numba(flow_batch, P_batch, beta, 1e-12, 100_000)
    for b in range(B):
        single = K.bellman_iterate(
            flow_batch[b], P_batch[b], beta, tol=1e-12, max_iter=100_000
        )
        np.testing.assert_allclose(got[b], single, rtol=1e-9, atol=1e-12)
