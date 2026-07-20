"""Every accelerated backend reproduces the numpy oracle within tol.

Mirrors tests/test_models/test_companion_oracle.py: tolerances matched to each
backend's precision, never loosened to hide a kernel bug; absent backends are
skipped (find_spec), never asserted against. cupy always skips on a non-CUDA
host. Float32 (MLX) values accumulate rounding over the VFI iteration, so its
bound is looser than the nested-DMP single-step bound but still float32-tight.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro import _backend as bk
from puremacro.vfi.discretize import tauchen
from puremacro.vfi.returnfn import build_return_tensor
from puremacro.vfi.solve import solve_vfi

ATOL, RTOL = 1e-10, 1e-9          # float64 backends (numba, cupy)
ATOL_F32, RTOL_F32 = 1e-3, 1e-4   # MLX float32 (Metal has no float64)


def _savings():
    a = np.linspace(1e-3, 20.0, 40)
    z_grid, P = tauchen(n=5, rho=0.9, sigma=0.1)
    params = {"r": 0.04, "w": 1.0}

    def rf(ap, a, z, r, w, xp=np):
        c = w * xp.exp(z) + (1.0 + r) * a - ap
        return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -1e10)

    return a, z_grid, P, params, rf


numba_only = pytest.mark.skipif(
    not bk.backend_available("numba"), reason="numba not installed"
)
mlx_only = pytest.mark.skipif(
    not bk.backend_available("mlx"), reason="mlx not installed"
)
cupy_only = pytest.mark.skipif(
    not bk.backend_available("cupy"), reason="cupy not installed"
)


def test_numpy_oracle_solves():
    a, z, P, params, rf = _savings()
    R = build_return_tensor(rf, a, z, params, xp=np)
    V, idx, n_ap, n_it, sup = solve_vfi(R, P, 0.95, tol=1e-10)
    assert np.all(np.isfinite(V)) and sup < 1e-10


@numba_only
def test_numba_matches_numpy_oracle():
    from puremacro.vfi import kernels_numba as KN

    a, z, P, params, rf = _savings()
    R = np.ascontiguousarray(build_return_tensor(rf, a, z, params, xp=np))
    V_o, idx_o, *_ = solve_vfi(R, P, 0.95, tol=1e-10)
    V_n, idx_n, n_it, sup = KN.solve_vfi_numba(R, P, 0.95, True, 20, 1e-10, 10_000)
    np.testing.assert_allclose(V_n, V_o, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(idx_n, idx_o)


@mlx_only
def test_mlx_matches_numpy_oracle():
    mx = bk.get_array_namespace("mlx")
    a, z, P, params, rf = _savings()
    R_np = build_return_tensor(rf, a, z, params, xp=np)
    V_o, *_ = solve_vfi(R_np, P, 0.95, tol=1e-10)
    R_mx = build_return_tensor(rf, a, z, params, xp=mx)
    V_m, idx_m, n_ap, n_it, sup = solve_vfi(R_mx, mx.array(P), 0.95, tol=1e-8, xp=mx)
    np.testing.assert_allclose(bk.to_numpy(V_m), V_o, rtol=RTOL_F32, atol=ATOL_F32)


@cupy_only
def test_cupy_matches_numpy_oracle():
    cp = bk.get_array_namespace("cupy")
    a, z, P, params, rf = _savings()
    R_np = build_return_tensor(rf, a, z, params, xp=np)
    V_o, *_ = solve_vfi(R_np, P, 0.95, tol=1e-10)
    R_cp = build_return_tensor(rf, a, z, params, xp=cp)
    V_c, idx_c, n_ap, n_it, sup = solve_vfi(R_cp, cp.array(P), 0.95, tol=1e-10, xp=cp)
    np.testing.assert_allclose(bk.to_numpy(V_c), V_o, rtol=RTOL, atol=ATOL)
