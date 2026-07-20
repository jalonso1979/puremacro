"""Axis-convention regression test for ProxySVARResult.

Locks the contract: irf_point / irf_lower / irf_upper are shape (H+1, n, n),
matching the other six *Result dataclasses in var/identify/_results.py.
"""
import numpy as np
import pytest

from puremacro.var.identify.proxy import proxy_svar


@pytest.fixture
def small_svar_inputs():
    rng = np.random.default_rng(2026)
    n = 3
    T = 200
    # Generate a small VAR(1) with a clean shock.
    A = 0.5 * np.eye(n)
    eps = rng.standard_normal((T, n))
    Y = np.zeros((T, n))
    Y[0] = eps[0]
    for t in range(1, T):
        Y[t] = A @ Y[t-1] + eps[t]
    # Proxy correlates with eps[:, 0].
    z = eps[:, 0] + 0.2 * rng.standard_normal(T)
    return Y, z


def test_proxy_svar_irf_point_shape(small_svar_inputs):
    Y, z = small_svar_inputs
    H = 8
    res = proxy_svar(Y, p=1, horizon=H, instrument_series=z, n_boot=20, ci=0.9, seed=0)
    assert res.irf_point.shape == (H + 1, Y.shape[1], Y.shape[1])
    assert res.irf_lower.shape == (H + 1, Y.shape[1], Y.shape[1])
    assert res.irf_upper.shape == (H + 1, Y.shape[1], Y.shape[1])
