"""Automated Test Suite for GPU and Apple Silicon MLX Acceleration Engine.

Verifies:
1. Unified hardware device detection and memory profiling.
2. Batched parallel Jacobian assembly (B=230 and B=307) and pre-inversion GEMM.
3. Directional derivative JVP evaluation and forward AD.
4. PyTorch GPU / MPS / CUDA solver convergence, residual tolerance, and parity.
5. Apple Silicon MLX solver convergence, residual tolerance, and parity.
6. Adaptive homotopy continuation along lambda in [0, 1] for tariff shocks.
7. Walrasian global current account clearing: |sum invforT| < 1e-4.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest

from puremacro.trade.calibration import calibrate_trade_model
from puremacro.trade.data import load_raw_45sector_icio
from puremacro.trade.gpu.backend import (
    detect_device,
    device_context,
    get_memory_usage,
    has_mlx,
    has_torch,
    reset_peak_memory,
    select_compute_device,
    to_numpy,
    to_tensor,
)
from puremacro.trade.gpu.batched_jacobian import BatchedJacobianEvaluator
from puremacro.trade.gpu.homotopy import solve_homotopy_continuation
from puremacro.trade.gpu.mlx_solver import solve_trade_equilibrium_mlx
from puremacro.trade.gpu.solver_gpu import solve_trade_equilibrium_gpu
from puremacro.trade.scenarios import build_tariff_matrices


@pytest.fixture(scope="module")
def calib_45s():
    """Calibrate full 45-sector 77-country empirical trade model."""
    raw = load_raw_45sector_icio()
    return calibrate_trade_model(raw, ns=45, nc=77, nfd=3)


@pytest.fixture(scope="module")
def checkpoint_paths():
    """Locate canonical base and t10 solution checkpoints."""
    env_dir = os.environ.get("IO_CHECKPOINTS_DIR")
    candidates = [
        Path(env_dir) if env_dir else None,
        Path.home() / "Documents" / "RESEARCH" / "IO" / "headlinePaper" / "tables_45s" / "checkpoints",
        Path.cwd().parents[1] / "IO" / "headlinePaper" / "tables_45s" / "checkpoints",
    ]
    for c in candidates:
        if c and c.exists() and (c / "xx_sol_77c_45s_base.npz").exists():
            return {
                "base": c / "xx_sol_77c_45s_base.npz",
                "t10": c / "xx_sol_77c_45s_t10.npz",
            }
    pytest.skip("Canonical 45-sector checkpoints not found.")


class TestDeviceBackend:
    """Verify device selection, capabilities detection, and memory profiling."""

    def test_detect_device_auto(self):
        dev = detect_device()
        assert dev.device in ("cuda", "mps", "mlx", "cpu")
        assert dev.backend in ("torch", "mlx", "numpy")
        assert isinstance(dev.device_name, str)
        assert isinstance(dev.supports_float64, bool)
        assert dev.total_memory_gb is None or isinstance(dev.total_memory_gb, float)

    def test_select_compute_device(self):
        backend, dev = select_compute_device("auto")
        assert backend in ("torch", "mlx", "numpy")
        assert dev in ("cuda", "mps", "mlx", "cpu")

    def test_tensor_conversion_roundtrip(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        t = to_tensor(arr, backend="torch" if has_torch() else "mlx")
        arr_back = to_numpy(t)
        np.testing.assert_allclose(arr, arr_back, rtol=1e-12, atol=1e-12)

    def test_memory_usage_and_reset(self):
        reset_peak_memory()
        mem = get_memory_usage()
        assert "allocated_mb" in mem
        assert "peak_mb" in mem
        assert mem["peak_mb"] >= 0.0


class TestBatchedJacobianEvaluator:
    """Verify pre-inversion, batched parallel evaluation, and JVP."""

    def test_pre_inversion_operators(self, calib_45s):
        tau, taufd, tauf_v, tauf_fd_v = build_tariff_matrices("base", calib_45s)
        evaluator = BatchedJacobianEvaluator(
            calib=calib_45s,
            tau_a=tau,
            taufd_a=taufd,
            tauf_vec=tauf_v,
            tauf_fd_vec=tauf_fd_v,
            device="cpu",
            backend="torch" if has_torch() else "mlx",
            batch_size=230,
        )
        assert evaluator.inv_P_np.shape == (3465, 3465)
        assert evaluator.inv_Y_np.shape == (3465, 3465)
        # Check that (I - A) @ inv(I - A) == I
        prod = (np.eye(3465) - evaluator.a_2d) @ evaluator.inv_Y_np
        np.testing.assert_allclose(prod, np.eye(3465), rtol=1e-10, atol=1e-10)

    def test_batched_jacobian_shape_230(self, calib_45s, checkpoint_paths):
        tau, taufd, tauf_v, tauf_fd_v = build_tariff_matrices("base", calib_45s)
        evaluator = BatchedJacobianEvaluator(
            calib=calib_45s,
            tau_a=tau,
            taufd_a=taufd,
            tauf_vec=tauf_v,
            tauf_fd_vec=tauf_fd_v,
            device="cpu",
            backend="torch" if has_torch() else "mlx",
            batch_size=230,
        )
        ckpt = np.load(checkpoint_paths["base"], allow_pickle=True)
        xx_sol = ckpt["xx_sol"]
        xm = np.concatenate([xx_sol[7007:7084], xx_sol[7084:7161], xx_sol[7161:7237]])

        J = evaluator.evaluate_batched_jacobian(xm)
        assert J.shape == (230, 230)
        assert np.all(np.isfinite(J))

    def test_batched_jacobian_shape_307(self, calib_45s, checkpoint_paths):
        tau, taufd, tauf_v, tauf_fd_v = build_tariff_matrices("base", calib_45s)
        evaluator = BatchedJacobianEvaluator(
            calib=calib_45s,
            tau_a=tau,
            taufd_a=taufd,
            tauf_vec=tauf_v,
            tauf_fd_vec=tauf_fd_v,
            device="cpu",
            backend="torch" if has_torch() else "mlx",
            batch_size=307,
        )
        ckpt = np.load(checkpoint_paths["base"], allow_pickle=True)
        xx_sol = ckpt["xx_sol"]
        xm = np.concatenate([xx_sol[6930:7007], xx_sol[7007:7084], xx_sol[7084:7161], xx_sol[7161:7237]])

        J = evaluator.evaluate_batched_jacobian(xm)
        assert J.shape == (307, 307)
        assert np.all(np.isfinite(J))

    def test_jvp_directional_derivative(self, calib_45s, checkpoint_paths):
        tau, taufd, tauf_v, tauf_fd_v = build_tariff_matrices("base", calib_45s)
        evaluator = BatchedJacobianEvaluator(
            calib=calib_45s,
            tau_a=tau,
            taufd_a=taufd,
            tauf_vec=tauf_v,
            tauf_fd_vec=tauf_fd_v,
            device="cpu",
            backend="torch" if has_torch() else "mlx",
            batch_size=230,
        )
        ckpt = np.load(checkpoint_paths["base"], allow_pickle=True)
        xx_sol = ckpt["xx_sol"]
        xm = np.concatenate([xx_sol[7007:7084], xx_sol[7084:7161], xx_sol[7161:7237]])

        np.random.seed(123)
        v = np.random.randn(230)
        v /= np.linalg.norm(v)

        jvp_val = evaluator.jvp(xm, v)
        assert jvp_val.shape == (230,)
        assert np.all(np.isfinite(jvp_val))


class TestGPUSolverParity:
    """Verify solve_trade_equilibrium_gpu and solve_trade_equilibrium_mlx."""

    def test_gpu_solver_convergence_and_parity(self, calib_45s, checkpoint_paths):
        tau, taufd, tauf_v, tauf_fd_v = build_tariff_matrices("base", calib_45s)
        ckpt = np.load(checkpoint_paths["base"], allow_pickle=True)
        xx_sol_base = ckpt["xx_sol"]

        np.random.seed(42)
        x0_pert = xx_sol_base.copy()
        x0_pert[7007:7084] += 0.01 * np.random.randn(77)

        res = solve_trade_equilibrium_gpu(
            calib=calib_45s,
            tau=tau,
            tau_fd=taufd,
            tauf=tauf_v,
            tauf_fd=tauf_fd_v,
            x0=x0_pert,
            device="auto",
            batch_size=230,
            max_iter=15,
            tol=2.5e-3,
        )

        assert res.converged is True, f"Solver did not converge. Max residual: {res.max_residual}"
        assert res.max_residual < 2.5e-3, f"Max residual {res.max_residual} exceeds 2.5e-3"
        assert res.x_sol.shape == (7237,)

        # Walrasian global current account clearing
        invforT = res.invforT if hasattr(res, "invforT") and res.invforT is not None else res.XN_sol
        if hasattr(res, "invforT") and res.invforT is not None:
            walras_err = abs(float(np.sum(res.invforT)))
            assert walras_err < 1.0e-4, f"Walrasian trade clearing error {walras_err} exceeds 1e-4"

        # Parity against base checkpoint: ||x_gpu - x_cpu||_inf < 1e-3
        log_w_gpu = res.x_sol[7007:7084]
        log_w_base = xx_sol_base[7007:7084]
        w_discrepancy = np.max(np.abs(np.exp(log_w_gpu) - np.exp(log_w_base)))
        assert w_discrepancy < 1.0e-3, f"Wage parity discrepancy {w_discrepancy} exceeds 1e-3"

    @pytest.mark.skipif(not has_mlx(), reason="Apple MLX is not available")
    def test_mlx_solver_convergence_and_parity(self, calib_45s, checkpoint_paths):
        tau, taufd, tauf_v, tauf_fd_v = build_tariff_matrices("base", calib_45s)
        ckpt = np.load(checkpoint_paths["base"], allow_pickle=True)
        xx_sol_base = ckpt["xx_sol"]

        np.random.seed(42)
        x0_pert = xx_sol_base.copy()
        x0_pert[7007:7084] += 0.01 * np.random.randn(77)

        res = solve_trade_equilibrium_mlx(
            calib=calib_45s,
            tau=tau,
            tau_fd=taufd,
            tauf=tauf_v,
            tauf_fd=tauf_fd_v,
            x0=x0_pert,
            batch_size=230,
            max_iter=15,
            tol=2.5e-3,
        )

        assert res.converged is True, f"MLX solver did not converge. Max residual: {res.max_residual}"
        assert res.max_residual < 2.5e-3, f"Max residual {res.max_residual} exceeds 2.5e-3"
        assert res.x_sol.shape == (7237,)

        log_w_mlx = res.x_sol[7007:7084]
        log_w_base = xx_sol_base[7007:7084]
        w_discrepancy = np.max(np.abs(np.exp(log_w_mlx) - np.exp(log_w_base)))
        assert w_discrepancy < 1.0e-3, f"MLX wage parity discrepancy {w_discrepancy} exceeds 1e-3"

    @pytest.mark.skipif(not has_mlx(), reason="Apple MLX is not available")
    def test_mlx_native_cpu_stream_solve(self):
        """Verify that Apple MLX natively solves double-precision linear systems via mx.cpu stream."""
        import mlx.core as mx

        n = 230
        np.random.seed(123)
        A_np = np.random.randn(n, n)
        A_spd_np = A_np.T @ A_np + 1.0 * np.eye(n)
        b_np = np.random.randn(n)

        # Reference solution via NumPy / LAPACK
        x_ref = np.linalg.solve(A_spd_np, b_np)

        # Native MLX CPU stream solve in float64
        with mx.stream(mx.cpu):
            A_mx = mx.array(A_spd_np, dtype=mx.float64)
            b_mx = mx.array(b_np[:, None], dtype=mx.float64)
            x_mx = mx.linalg.solve(A_mx, b_mx, stream=mx.cpu)
            mx.eval(x_mx)
            x_sol = np.array(x_mx).ravel()

        diff = np.max(np.abs(x_sol - x_ref))
        assert diff < 1.0e-10, f"MLX CPU stream solve discrepancy {diff} exceeds 1e-10"


class TestHomotopyContinuation:
    """Verify adaptive homotopy continuation along tariff parameter lambda."""

    def test_homotopy_continuation_to_t10(self, calib_45s, checkpoint_paths):
        tau_base, taufd_base, tauf_v_base, tauf_fd_v_base = build_tariff_matrices("base", calib_45s)
        tau_t10, taufd_t10, tauf_v_t10, tauf_fd_v_t10 = build_tariff_matrices("t10", calib_45s)

        ckpt_base = np.load(checkpoint_paths["base"], allow_pickle=True)
        ckpt_t10 = np.load(checkpoint_paths["t10"], allow_pickle=True)

        res_t10 = solve_homotopy_continuation(
            calib=calib_45s,
            target_tau=tau_t10,
            target_tau_fd=taufd_t10,
            target_tauf=tauf_v_t10,
            target_tauf_fd=tauf_fd_v_t10,
            base_tau=tau_base,
            base_tau_fd=taufd_base,
            base_tauf=tauf_v_base,
            base_tauf_fd=tauf_fd_v_base,
            x0=ckpt_base["xx_sol"],
            device="auto",
            batch_size=230,
            initial_step=0.25,
            tol=2.5e-3,
        )

        assert res_t10.converged is True, f"Homotopy to t10 failed. Max res: {res_t10.max_residual}"
        assert res_t10.max_residual < 2.5e-3, f"Max res {res_t10.max_residual} exceeds 2.5e-3"

        # Compare against canonical t10 checkpoint: ||x_homotopy - x_canon||_inf < 1e-3
        log_w_hom = res_t10.x_sol[7007:7084]
        log_w_canon = ckpt_t10["xx_sol"][7007:7084]
        w_discrepancy = np.max(np.abs(np.exp(log_w_hom) - np.exp(log_w_canon)))
        assert w_discrepancy < 1.0e-3, f"Homotopy wage discrepancy {w_discrepancy} exceeds 1e-3"
