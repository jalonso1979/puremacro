"""Tests for hardware acceleration and backend abstraction across PureMacro solvers.

Verifies:
1. Array namespace and backend detection utilities (_backend.py).
2. Tensor contraction equivalence: xp.sum(a * b[None, :, :], axis=2) vs einsum.
3. Condensed Schur solver with backend="numpy", backend="mlx", and fallback.
4. Allen & Arkolakis spatial general equilibrium with hardware acceleration and fallback.
"""
from __future__ import annotations

import warnings
import numpy as np
import pytest

from puremacro._backend import backend_available, get_array_namespace, to_numpy
from puremacro.spatial.allen_arkolakis import AllenArkolakisModel
from puremacro.trade.calibration import calibrate_trade_model
from puremacro.trade.solver import solve_trade_equilibrium


@pytest.fixture
def toy_trade_calib():
    """Construct a minimal 2-country, 2-sector, 3-final-demand calibration."""
    nc = 2
    ns = 2
    nfd = 3
    data = np.zeros((7, 10), dtype=float)

    # Intermediate transactions
    data[:4, :4] = np.array([
        [20.0, 15.0, 5.0, 2.0],
        [10.0, 25.0, 2.0, 8.0],
        [5.0, 5.0, 12.0, 18.0],
        [10.0, 10.0, 18.0, 22.0],
    ])
    inter_col_sums = data[:4, :4].sum(axis=0)
    y = np.array([100.0, 150.0, 120.0, 180.0])
    va = y - inter_col_sums
    taxes = 0.05 * y
    va_fac = va - taxes
    labor = (2.0 / 3.0) * va_fac
    capital = (1.0 / 3.0) * va_fac

    data[4, :4] = taxes
    data[5, :4] = labor
    data[6, :4] = capital

    # Final demand blocks
    inter_row_sums = data[:4, :4].sum(axis=1)
    fd_row_sums = y - inter_row_sums

    for i in range(4):
        tot_fd = fd_row_sums[i]
        if i < 2:
            data[i, 4] = tot_fd * 0.50
            data[i, 5] = tot_fd * 0.25
            data[i, 6] = tot_fd * 0.05
            data[i, 7] = tot_fd * 0.10
            data[i, 8] = tot_fd * 0.08
            data[i, 9] = tot_fd * 0.02
        else:
            data[i, 4] = tot_fd * 0.10
            data[i, 5] = tot_fd * 0.08
            data[i, 6] = tot_fd * 0.02
            data[i, 7] = tot_fd * 0.50
            data[i, 8] = tot_fd * 0.25
            data[i, 9] = tot_fd * 0.05

    fd_col_sums = data[:4, 4:].sum(axis=0)
    data[4, 4:] = 0.02 * fd_col_sums

    return calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, validate=True)


def test_backend_detection_and_namespace():
    """Verify backend availability, array namespace retrieval, and to_numpy converter."""
    assert backend_available("numpy") is True
    xp_np = get_array_namespace("numpy")
    assert xp_np is np

    arr = np.array([1.0, 2.0, 3.0])
    np_out = to_numpy(arr)
    assert isinstance(np_out, np.ndarray)
    np.testing.assert_array_equal(arr, np_out)

    # Conversion of scalars and lists
    assert to_numpy(42) == 42
    list_out = to_numpy([1.0, 2.0])
    assert isinstance(list_out, np.ndarray)

    with pytest.raises(ValueError, match="Unknown backend"):
        backend_available("nonexistent_backend")

    if backend_available("mlx"):
        xp_mlx = get_array_namespace("mlx")
        mlx_arr = xp_mlx.array([1.0, 2.0, 3.0])
        converted = to_numpy(mlx_arr)
        assert isinstance(converted, np.ndarray)
        np.testing.assert_allclose(converted, [1.0, 2.0, 3.0])


def test_tensor_contraction_equivalence():
    """Verify that xp.sum(a * b[None, :, :], axis=2) exactly equals np.einsum."""
    rng = np.random.default_rng(42)
    M, nc, ns = 20, 5, 4

    a_blocks = rng.standard_normal((M, nc, ns))
    y_blocks = rng.standard_normal((nc, ns))

    einsum_result = np.einsum("min,in->mi", a_blocks, y_blocks)
    broadcast_result = np.sum(a_blocks * y_blocks[None, :, :], axis=2)

    np.testing.assert_allclose(einsum_result, broadcast_result, atol=1e-14, rtol=1e-14)

    if backend_available("mlx"):
        xp_mlx = get_array_namespace("mlx")
        a_mlx = xp_mlx.array(a_blocks.astype(np.float32))
        y_mlx = xp_mlx.array(y_blocks.astype(np.float32))
        mlx_result = to_numpy(xp_mlx.sum(a_mlx * y_mlx[None, :, :], axis=2))

        np.testing.assert_allclose(
            einsum_result.astype(np.float32),
            mlx_result,
            atol=1e-5,
            rtol=1e-5,
        )


def test_trade_equilibrium_backend_acceleration(toy_trade_calib):
    """Verify solve_trade_equilibrium with method='condensed' across backends.

    The comparison uses a tariff-shocked problem so that the solver actually
    iterates (the unshocked calibration is already an equilibrium and returns
    the initial guess after 0 iterations, which cannot detect backend defects).
    """
    ns, nc, nfd = 2, 2, 3
    tau = np.ones((ns, nc, ns, nc))
    tau_fd = np.ones((ns, nc, nfd, nc))
    tau[:, 1, :, 0] = 1.25  # country 0 imposes 25% on imports from country 1
    tau_fd[:, 1, :, 0] = 1.25

    res_np = solve_trade_equilibrium(toy_trade_calib, tau=tau, tau_fd=tau_fd, method="condensed", backend="numpy")
    assert res_np.converged is True
    assert res_np.iterations > 0
    assert res_np.max_residual < 2.5e-3

    if backend_available("mlx"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # a fallback to numpy would hide a defect
            res_mlx = solve_trade_equilibrium(toy_trade_calib, tau=tau, tau_fd=tau_fd, method="condensed", backend="mlx")
        assert res_mlx.converged is True
        assert res_mlx.metadata["backend"] == "mlx"
        np.testing.assert_allclose(res_np.w_sol, res_mlx.w_sol, atol=1e-8)
        np.testing.assert_allclose(res_np.r_sol, res_mlx.r_sol, atol=1e-8)
        np.testing.assert_allclose(res_np.p_sol, res_mlx.p_sol, atol=1e-8)
        np.testing.assert_allclose(res_np.x_sol, res_mlx.x_sol, atol=1e-8)

    # Test unavailable backend fallback
    with pytest.warns(RuntimeWarning, match="falling back to 'numpy'"):
        res_fallback = solve_trade_equilibrium(toy_trade_calib, tau=tau, tau_fd=tau_fd, method="condensed", backend="cupy")
    assert res_fallback.converged is True
    np.testing.assert_allclose(res_np.w_sol, res_fallback.w_sol, atol=1e-12)


def test_allen_arkolakis_backend_acceleration():
    """Verify AllenArkolakisModel.solve_equilibrium with hardware backends and fallback."""
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ])
    model = AllenArkolakisModel.from_coordinates(coords)

    # NumPy baseline
    res_np = model.solve_equilibrium(backend="numpy")
    assert res_np.converged is True
    assert res_np.labor_conservation_residual < 1e-12
    assert res_np.spatial_utility_variance < 1e-7

    # MLX acceleration (if available on Apple Silicon): float32 contraction polished in float64
    if backend_available("mlx"):
        res_mlx = model.solve_equilibrium(backend="mlx")
        assert res_mlx.converged is True
        assert res_mlx.max_residual < 1e-8, "converged must mean the requested tol=1e-8 was met"
        assert res_mlx.labor_conservation_residual < 1e-12
        # Verify close agreement with NumPy (the requested tolerance, not a float32 floor)
        np.testing.assert_allclose(res_np.wages, res_mlx.wages, atol=1e-6, rtol=0)
        np.testing.assert_allclose(res_np.population, res_mlx.population, atol=1e-6, rtol=0)
        np.testing.assert_allclose(res_np.price_index, res_mlx.price_index, atol=1e-6, rtol=0)
        np.testing.assert_allclose(res_np.welfare, res_mlx.welfare, atol=1e-8, rtol=0)

        # Counterfactual on MLX
        cf_mlx = model.solve_counterfactual(
            productivity_new=np.array([1.2, 1.0, 1.0, 1.0, 1.0]),
            backend="mlx",
        )
        assert cf_mlx.converged is True
        assert cf_mlx.welfare_pct is not None
        cf_np = model.solve_counterfactual(
            productivity_new=np.array([1.2, 1.0, 1.0, 1.0, 1.0]),
            backend="numpy",
        )
        np.testing.assert_allclose(cf_mlx.welfare_pct, cf_np.welfare_pct, atol=1e-6, rtol=0)

    # Fallback when backend is unavailable
    with pytest.warns(RuntimeWarning, match="falling back to 'numpy'"):
        res_fallback = model.solve_equilibrium(backend="cupy")
    assert res_fallback.converged is True
    np.testing.assert_allclose(res_np.wages, res_fallback.wages, atol=1e-12)
