"""Every accelerated backend must reproduce the NumPy oracle within tol.

The non-negotiable correctness harness: a
divergence BEYOND a backend's intrinsic numerical precision is a kernel
bug, not a tolerance to widen. Tolerances are matched to each backend's
precision, never loosened to hide bugs:
  - float64 backends (numba) must match the float64 oracle to ~1e-9.
  - MLX runs in float32 on the Apple GPU (float64 is unsupported on
    Metal), so its oracle asserts float32-precision agreement (~1e-5) —
    the tightest physically meaningful bound. A divergence beyond float32
    rounding noise still fails.
Backends absent from the environment are skipped, never asserted against.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.models.nested_dmp import backend as bk
from puremacro.models.nested_dmp import kernels as K

# float64 backends (numba): full double precision vs the numpy oracle.
ATOL = 1e-10
RTOL = 1e-9
# MLX/GPU is float32-native (Metal has no float64); assert float32 precision.
ATOL_F32 = 1e-6
RTOL_F32 = 1e-5


def _row_stochastic(n, rng):
    A = rng.random((n, n)) + 1e-3
    return A / A.sum(axis=1, keepdims=True)


# ---- MLX namespace backend: run the xp-generic kernels under mlx.core ----

mlx_only = pytest.mark.skipif(
    not bk.backend_available("mlx"), reason="mlx not installed"
)


@mlx_only
def test_mlx_bellman_matches_numpy_oracle(rng):
    mx = bk.get_array_namespace("mlx")
    n, beta = 8, 0.96
    P_np = _row_stochastic(n, rng)
    flow_np = rng.standard_normal(n)
    oracle = K.bellman_iterate(flow_np, P_np, beta, tol=1e-12, max_iter=100_000)
    got = K.bellman_iterate(
        mx.array(flow_np), mx.array(P_np), beta, tol=1e-12, max_iter=100_000, xp=mx
    )
    np.testing.assert_allclose(bk.to_numpy(got), oracle, rtol=RTOL_F32, atol=ATOL_F32)


@mlx_only
def test_mlx_belief_update_matches_numpy_oracle(rng):
    mx = bk.get_array_namespace("mlx")
    r_obs = rng.standard_normal(20)
    args = dict(sigma=1.0, r_star=0.0, phi_D=-3.0, phi_H=1.0, signal_sd=1.0)
    oracle = K.belief_update(0.5, r_obs=r_obs, xp=np, **args)
    got = K.belief_update(0.5, r_obs=mx.array(r_obs), xp=mx, **args)
    np.testing.assert_allclose(bk.to_numpy(got), oracle, rtol=RTOL_F32, atol=ATOL_F32)


# ---- Numba backend: compiled kernels vs the oracle ----

numba_only = pytest.mark.skipif(
    not bk.backend_available("numba"), reason="numba not installed"
)


@numba_only
def test_numba_bellman_matches_numpy_oracle(rng):
    from puremacro.models.nested_dmp import kernels_numba as KN

    n, beta = 8, 0.96
    P = _row_stochastic(n, rng)
    flow = rng.standard_normal(n)
    oracle = K.bellman_iterate(flow, P, beta, tol=1e-12, max_iter=100_000)
    got = KN.bellman_iterate_numba(flow, P, beta, 1e-12, 100_000)
    np.testing.assert_allclose(got, oracle, rtol=RTOL, atol=ATOL)


# ---- Equilibrium block: match-value VFI must match the oracle under MLX ----

from puremacro.models.nested_dmp import equilibrium as S  # noqa: E402
from puremacro.models.nested_dmp.params import NestedDMPParameters  # noqa: E402


@mlx_only
def test_mlx_match_value_matches_numpy_oracle():
    mx = bk.get_array_namespace("mlx")
    p = NestedDMPParameters(n_x=15)
    grid, P, _ = S.sigma_scaled_process(p, sigma=0.5)
    J_oracle, _, _, _ = S.match_value(p, 0.5, 0.7, theta=1.0, grid=grid, P=P)
    J_mlx, _, _, _ = S.match_value(p, 0.5, 0.7, theta=1.0, grid=grid, P=P, xp=mx)
    np.testing.assert_allclose(
        bk.to_numpy(J_mlx), np.asarray(J_oracle), rtol=RTOL_F32, atol=ATOL_F32
    )


@mlx_only
def test_mlx_match_value_parity_with_delta():
    mx = bk.get_array_namespace("mlx")
    p = NestedDMPParameters(n_x=15, delta=0.08)
    grid, P, _ = S.sigma_scaled_process(p, sigma=0.5)
    J_oracle, *_ = S.match_value(p, 0.5, 0.7, theta=1.0, grid=grid, P=P)
    J_mlx, *_ = S.match_value(p, 0.5, 0.7, theta=1.0, grid=grid, P=P, xp=mx)
    np.testing.assert_allclose(
        bk.to_numpy(J_mlx), np.asarray(J_oracle), rtol=RTOL_F32, atol=ATOL_F32
    )
