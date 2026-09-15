"""Regression tests for the 3.4.0 pre-release trade/GPU review findings.

Everything here runs on a tiny synthetic calibration, so it is fast and needs no
private data. Accelerator-specific cases skip when torch / mlx are not installed;
the NumPy paths always run, including on CI.

Covered findings:
- ``import puremacro.trade`` must not import torch or mlx (lazy accelerator loading).
- ``select_compute_device`` rejects unknown strings and works without PyTorch.
- GPU / MLX solvers use the NumPy solver's tariff defaults (national rates = 0).
- The macro layout (reduced r == w vs full) is validated against the calibration
  instead of being switched by the magic number ``batch_size == 230``.
- Batched residuals equal the serial residual (including the ytot > 0 mask) and
  Jacobians are evaluated in float64; AD modes are exact and never fall back silently.
- ``solve_trade_equilibrium(method='condensed', backend='mlx')`` matches NumPy on a
  problem that actually iterates, and falls back with a warning if it does not converge.
- ``compute_equilibrium_residuals`` rejects unknown keyword arguments.
- ``TariffScenario.build_*_tariffs`` honour the calibration's codes.
- ``legacy_compat`` only applies to the canonical 5-scenario batch.
- ``AllenArkolakisModel`` with ``backend='mlx'`` honours ``tol``.
"""
from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import pytest

from puremacro._backend import backend_available
from puremacro.trade import TariffScenario, build_tariff_matrices, run_scenario_batch
from puremacro.trade.calibration import calibrate_trade_model
from puremacro.trade.equilibrium import compute_equilibrium_residuals
from puremacro.trade.gpu import (
    BatchedJacobianEvaluator,
    select_compute_device,
    solve_homotopy_continuation,
    solve_trade_equilibrium_gpu,
    solve_trade_equilibrium_mlx,
)
from puremacro.trade.gpu import backend as gpu_backend
from puremacro.trade.solver import solve_trade_equilibrium
from puremacro.trade.tables import to_latex_selected_country_table, to_latex_selected_country_with_row

HAS_TORCH = gpu_backend.has_torch()
HAS_MLX = gpu_backend.has_mlx()
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Synthetic calibrations
# ---------------------------------------------------------------------------

def _make_io_table(nc: int, ns: int, nfd: int, seed: int = 0, labour_share: np.ndarray | None = None) -> np.ndarray:
    """Deterministic balanced IO table with heavier domestic blocks."""
    rng = np.random.default_rng(seed)
    M = ns * nc
    data = np.zeros((M + 3, M + nfd * nc), dtype=float)
    Z = rng.uniform(1.0, 10.0, size=(M, M))
    for c in range(nc):
        Z[c * ns : (c + 1) * ns, c * ns : (c + 1) * ns] *= 3.0
    data[:M, :M] = Z
    col = Z.sum(axis=0)
    row = Z.sum(axis=1)
    y = np.maximum(col, row) * 2.0 + rng.uniform(20.0, 60.0, size=M)
    va = y - col
    taxes = 0.05 * y
    va_fac = va - taxes
    share = np.full(M, 2.0 / 3.0) if labour_share is None else np.tile(np.asarray(labour_share, dtype=float), nc)
    data[M, :M] = taxes
    data[M + 1, :M] = share * va_fac
    data[M + 2, :M] = (1.0 - share) * va_fac
    fd_tot = y - row
    for i in range(M):
        oc = i // ns
        sh = rng.uniform(0.5, 1.5, size=nfd * nc)
        for c in range(nc):
            if c == oc:
                sh[c * nfd : (c + 1) * nfd] *= 4.0
        sh /= sh.sum()
        data[i, M:] = fd_tot[i] * sh
    fd_col = data[:M, M:].sum(axis=0)
    data[M, M:] = 0.02 * fd_col
    return data


@pytest.fixture(scope="module")
def calib3():
    """3-country, 3-sector calibration with uniform capital shares (reduced layout exact)."""
    return calibrate_trade_model(_make_io_table(3, 3, 3), ns=3, nc=3, nfd=3, validate=True)


@pytest.fixture(scope="module")
def calib3_nonuniform():
    """3-country calibration whose sectors have labour shares 0.8 / 0.5 / 0.3 (r == w invalid)."""
    data = _make_io_table(3, 3, 3, seed=1, labour_share=np.array([0.8, 0.5, 0.3]))
    return calibrate_trade_model(data, ns=3, nc=3, nfd=3, validate=True)


@pytest.fixture(scope="module")
def shock3():
    """Country 0 imposes 25% on country 1 and 10% on country 2 (4D multipliers)."""
    nc, ns, nfd = 3, 3, 3
    tau = np.ones((ns, nc, ns, nc))
    tau_fd = np.ones((ns, nc, nfd, nc))
    tau[:, 1, :, 0] = 1.25
    tau_fd[:, 1, :, 0] = 1.25
    tau[:, 2, :, 0] = 1.10
    tau_fd[:, 2, :, 0] = 1.10
    return tau, tau_fd


def _available_gpu_backends() -> list[tuple[str | None, str]]:
    out: list[tuple[str | None, str]] = [("cpu", "numpy")]
    if HAS_TORCH:
        out.append(("cpu", "torch"))
    if HAS_MLX:
        out.append(("mlx", "mlx"))
    return out


# ---------------------------------------------------------------------------
# Lazy accelerator imports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ["puremacro.trade", "puremacro.spatial", "puremacro.trade.gpu"])
def test_import_does_not_load_torch_or_mlx(module: str):
    """Importing the trade / spatial packages must not pull torch or mlx into sys.modules."""
    code = (
        "import sys\n"
        f"import {module}\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'mlx'))\n"
        "print('LOADED=' + ','.join(loaded))\n"
    )
    proc = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("LOADED=")][-1]
    assert line == "LOADED=", f"{module} eagerly imported: {line[len('LOADED='):]}"


def test_has_torch_has_mlx_are_cheap_probes(monkeypatch):
    """has_torch()/has_mlx() use find_spec and never import the libraries themselves."""
    import importlib.util

    calls: list[str] = []
    real = importlib.util.find_spec

    def spy(name, *args, **kwargs):
        calls.append(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", spy)
    assert isinstance(gpu_backend.has_torch(), bool)
    assert isinstance(gpu_backend.has_mlx(), bool)
    assert calls == ["torch", "mlx"]


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["bogus", "gpu0", "cuda0", "apple"])
def test_select_compute_device_rejects_unknown_strings(bad: str):
    with pytest.raises(ValueError, match="Unknown device/backend"):
        select_compute_device(bad)


def test_select_compute_device_numpy_and_cpu_without_torch(monkeypatch):
    assert select_compute_device("numpy") == ("numpy", "cpu")
    # Without PyTorch, 'cpu' must resolve to the NumPy serial evaluator instead of raising.
    monkeypatch.setattr(gpu_backend, "_load_torch", lambda: None)
    assert select_compute_device("cpu") == ("numpy", "cpu")
    with pytest.raises(RuntimeError, match="PyTorch is not installed"):
        select_compute_device("mps")


def test_gpu_solver_runs_on_numpy_backend_without_torch(calib3, shock3, monkeypatch):
    """The public GPU solver must reach the NumPy evaluator when torch/mlx are absent."""
    monkeypatch.setattr(gpu_backend, "_load_torch", lambda: None)
    monkeypatch.setattr(gpu_backend, "has_mlx", lambda: False)
    tau, tau_fd = shock3
    ref = solve_trade_equilibrium(calib3, tau=tau, tau_fd=tau_fd, method="condensed")
    res = solve_trade_equilibrium_gpu(calib3, tau=tau, tau_fd=tau_fd, device="cpu")
    assert res.converged
    assert res.metadata["backend"] == "numpy"
    assert set(res.metadata["jacobian_evaluations"]) == {"numpy:cpu:float64"}
    np.testing.assert_allclose(res.w_sol, ref.w_sol, atol=1e-5)


# ---------------------------------------------------------------------------
# Tariff defaults and layout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device,backend", _available_gpu_backends())
def test_gpu_solver_default_call_matches_numpy_solver(calib3, device, backend):
    """With all defaults the GPU/MLX solver must solve the same baseline as the NumPy solver."""
    ref = solve_trade_equilibrium(calib3, method="condensed")
    assert ref.converged and ref.iterations == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = solve_trade_equilibrium_gpu(calib3, device=device, backend=backend)
    assert res.converged
    assert res.iterations == 0, "default national tariff rates must be zero, like the NumPy solver"
    np.testing.assert_allclose(res.w_sol, ref.w_sol, atol=1e-12)
    np.testing.assert_allclose(res.T_sol, ref.T_sol, atol=1e-10)
    np.testing.assert_allclose(res.x_sol, ref.x_sol, atol=1e-10)


@pytest.mark.skipif(not HAS_MLX, reason="Apple MLX is not available")
def test_mlx_solver_default_call_matches_numpy_solver(calib3):
    ref = solve_trade_equilibrium(calib3, method="condensed")
    res = solve_trade_equilibrium_mlx(calib3)
    assert res.converged and res.iterations == 0
    assert res.metadata["method"] == "mlx_accelerated"
    np.testing.assert_allclose(res.w_sol, ref.w_sol, atol=1e-12)


@pytest.mark.parametrize("device,backend", _available_gpu_backends())
def test_gpu_solver_shock_matches_numpy_solver(calib3, shock3, device, backend):
    tau, tau_fd = shock3
    ref = solve_trade_equilibrium(calib3, tau=tau, tau_fd=tau_fd, method="condensed")
    assert ref.converged and ref.iterations > 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = solve_trade_equilibrium_gpu(calib3, tau=tau, tau_fd=tau_fd, device=device, backend=backend)
    assert res.converged and res.iterations > 0
    assert res.metadata["jacobian_dtype"] == "float64"
    assert all(key.endswith(":float64") for key in res.metadata["jacobian_evaluations"])
    np.testing.assert_allclose(res.w_sol, ref.w_sol, atol=1e-5)
    np.testing.assert_allclose(res.p_sol, ref.p_sol, atol=1e-4)


def test_batch_size_magic_number_is_rejected_on_small_calibration(calib3, shock3):
    tau, tau_fd = shock3
    with pytest.raises(ValueError, match="batch_size=230 is inconsistent with a 3-country calibration"):
        solve_trade_equilibrium_gpu(calib3, tau=tau, tau_fd=tau_fd, device="cpu", backend="numpy", batch_size=230)
    with pytest.raises(ValueError, match="implies factor_equivalence=True"):
        solve_trade_equilibrium_gpu(
            calib3, tau=tau, tau_fd=tau_fd, device="cpu", backend="numpy", batch_size=8, factor_equivalence=False
        )
    # The dimension-consistent values still select the layout
    r8 = solve_trade_equilibrium_gpu(calib3, tau=tau, tau_fd=tau_fd, device="cpu", backend="numpy", batch_size=8)
    r11 = solve_trade_equilibrium_gpu(calib3, tau=tau, tau_fd=tau_fd, device="cpu", backend="numpy", batch_size=11)
    assert (r8.metadata["factor_equivalence"], r8.metadata["batch_size"]) == (True, 8)
    assert (r11.metadata["factor_equivalence"], r11.metadata["batch_size"]) == (False, 11)
    assert r8.converged and r11.converged
    np.testing.assert_allclose(r8.w_sol, r11.w_sol, atol=1e-5)


def test_reduced_layout_is_refused_when_capital_shares_differ(calib3_nonuniform):
    nc, ns, nfd = 3, 3, 3
    tau = np.ones((ns, nc, ns, nc))
    tau_fd = np.ones((ns, nc, nfd, nc))
    tau[:, 1, :, 0] = 1.30
    tau_fd[:, 1, :, 0] = 1.30
    ref = solve_trade_equilibrium(calib3_nonuniform, tau=tau, tau_fd=tau_fd, method="condensed")
    assert ref.converged
    assert np.max(np.abs(ref.r_sol / ref.w_sol - 1.0)) > 1e-5, "r != w in this calibration"

    with pytest.raises(ValueError, match="factor_equivalence=True .* is invalid"):
        solve_trade_equilibrium_gpu(
            calib3_nonuniform, tau=tau, tau_fd=tau_fd, device="cpu", backend="numpy", factor_equivalence=True
        )
    # Default: the full layout is selected automatically and matches the NumPy solver
    res = solve_trade_equilibrium_gpu(calib3_nonuniform, tau=tau, tau_fd=tau_fd, device="cpu", backend="numpy")
    assert res.converged
    assert res.metadata["factor_equivalence"] is False
    np.testing.assert_allclose(res.w_sol, ref.w_sol, atol=1e-5)
    np.testing.assert_allclose(res.r_sol / res.w_sol, ref.r_sol / ref.w_sol, atol=1e-5)


def test_initial_guess_accepts_macro_vectors_of_either_layout(calib3, shock3):
    tau, tau_fd = shock3
    nc = 3
    ref = solve_trade_equilibrium(calib3, tau=tau, tau_fd=tau_fd, method="condensed")
    kw = dict(tau=tau, tau_fd=tau_fd, device="cpu", backend="numpy")
    xm_reduced = np.concatenate([np.zeros(nc), calib3.T.ravel(), calib3.invforT.ravel()[: nc - 1]])
    xm_full = np.concatenate([np.zeros(2 * nc), calib3.T.ravel(), calib3.invforT.ravel()[: nc - 1]])
    for x0 in (ref.x_sol, xm_reduced, xm_full):
        res = solve_trade_equilibrium_gpu(calib3, x0=x0, **kw)
        assert res.converged
        np.testing.assert_allclose(res.w_sol, ref.w_sol, atol=1e-5)
    with pytest.raises(ValueError, match="Expected 8 .* 11 .* 29"):
        solve_trade_equilibrium_gpu(calib3, x0=np.zeros(9), **kw)


def test_homotopy_continuation_on_numpy_backend(calib3):
    nc, ns, nfd = 3, 3, 3
    tau = np.ones((ns, nc, ns, nc))
    tau_fd = np.ones((ns, nc, nfd, nc))
    tau[:, 1, :, 0] = 2.45
    tau_fd[:, 1, :, 0] = 2.45
    ref = solve_trade_equilibrium(calib3, tau=tau, tau_fd=tau_fd, method="condensed")
    res = solve_homotopy_continuation(calib3, tau, tau_fd, device="cpu", backend="numpy")
    assert res.converged
    np.testing.assert_allclose(res.w_sol, ref.w_sol, atol=1e-5)


# ---------------------------------------------------------------------------
# Batched evaluator: residual parity, masking, Jacobian accuracy, AD modes
# ---------------------------------------------------------------------------

def _central_difference_jacobian(ev: BatchedJacobianEvaluator, xm: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    B = len(xm)
    J = np.empty((B, B))
    for j in range(B):
        e = np.zeros(B)
        e[j] = eps
        f_plus, _, _, _, _ = ev.eval_macro_single(xm + e)
        f_minus, _, _, _, _ = ev.eval_macro_single(xm - e)
        J[:, j] = (f_plus - f_minus) / (2.0 * eps)
    return J


def _evaluator_backends() -> list[tuple[str, str]]:
    out = []
    if HAS_TORCH:
        out.append(("torch", "cpu"))
    if HAS_MLX:
        out.append(("mlx", "cpu"))
    return out


def _batched_residual(ev: BatchedJacobianEvaluator, xm: np.ndarray) -> np.ndarray:
    if ev.backend == "torch":
        torch = ev._torch
        return ev.eval_macro_batched_torch(torch.as_tensor(xm[None, :], dtype=ev.dtype, device=ev.device)).cpu().numpy()[0]
    mx = ev._mx
    return np.asarray(ev.eval_macro_batched_mlx(mx.array(xm[None, :], dtype=mx.float64)))[0]


@pytest.mark.parametrize("backend,device", _evaluator_backends())
@pytest.mark.parametrize("factor_equivalence", [True, False])
def test_batched_residual_matches_serial_including_inactive_sectors(calib3, shock3, backend, device, factor_equivalence):
    tau, tau_fd = shock3
    z = np.zeros(3)
    ev = BatchedJacobianEvaluator(calib3, tau, tau_fd, z, z, device=device, backend=backend, factor_equivalence=factor_equivalence)
    xm = ev._macro_from_x0(None)
    f0, _, _, _, y = ev.eval_macro_single(xm)
    assert np.all(y > 0)
    assert np.max(np.abs(_batched_residual(ev, xm) - f0)) < 1e-11 * max(1.0, np.max(np.abs(f0)))

    # A large trade-balance guess drives some gross outputs negative: the batched
    # kernels must apply the same ytot > 0 mask as the serial residual.
    xm_neg = xm.copy()
    xm_neg[-(3 - 1):] = 3e4 * np.array([1.0, -0.5])
    f_neg, _, _, _, y_neg = ev.eval_macro_single(xm_neg)
    assert np.any(y_neg <= 0), "test point must have non-positive outputs"
    assert np.max(np.abs(_batched_residual(ev, xm_neg) - f_neg)) < 1e-11 * np.max(np.abs(f_neg))


@pytest.mark.parametrize("backend,device", _evaluator_backends())
def test_float64_jacobian_and_ad_modes_are_accurate(calib3, shock3, backend, device):
    tau, tau_fd = shock3
    z = np.zeros(3)
    ev = BatchedJacobianEvaluator(calib3, tau, tau_fd, z, z, device=device, backend=backend)
    xm = ev._macro_from_x0(None)
    xm[: ev.n_factor_vars] += 0.02
    f0, _, _, _, _ = ev.eval_macro_single(xm)
    J_ref = _central_difference_jacobian(ev, xm)
    scale = np.max(np.abs(J_ref))

    J_fd = ev.evaluate_batched_jacobian(xm, f_base=f0, ad_mode="finite_diff")
    assert np.max(np.abs(J_fd - J_ref)) / scale < 1e-3  # one-sided differences, h = 1e-4

    for mode in ("forward", "vjp"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # a silent FD fallback would show up as FD error
            J_ad = ev.evaluate_batched_jacobian(xm, f_base=f0, ad_mode=mode)
        assert np.max(np.abs(J_ad - J_ref)) / scale < 1e-7, mode

    v = np.random.default_rng(0).standard_normal(len(xm))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        jv = ev.jvp(xm, v)
    assert np.max(np.abs(jv - J_ref @ v)) / np.max(np.abs(J_ref @ v)) < 1e-7

    with pytest.raises(ValueError, match="Unknown ad_mode"):
        ev.evaluate_batched_jacobian(xm, f_base=f0, ad_mode="reverse")


def test_numpy_evaluator_warns_when_ad_is_requested(calib3, shock3):
    tau, tau_fd = shock3
    z = np.zeros(3)
    ev = BatchedJacobianEvaluator(calib3, tau, tau_fd, z, z, device="cpu", backend="numpy")
    xm = ev._macro_from_x0(None)
    with pytest.warns(RuntimeWarning, match="requires the torch or mlx backend"):
        ev.evaluate_batched_jacobian(xm, ad_mode="forward")


@pytest.mark.skipif(not HAS_MLX, reason="Apple MLX is not available")
def test_mlx_batched_eval_defaults_to_float64_cpu_stream(calib3, shock3):
    import mlx.core as mx

    tau, tau_fd = shock3
    z = np.zeros(3)
    ev = BatchedJacobianEvaluator(calib3, tau, tau_fd, z, z, device="mlx", backend="mlx")
    assert (ev.backend, ev.device) == ("mlx", "gpu")
    xm = ev._macro_from_x0(None)
    f0, _, _, _, _ = ev.eval_macro_single(xm)
    # stream=None used to raise TypeError (mx.DeviceType == None) and silently disabled AD
    out = ev.eval_macro_batched_mlx(mx.array(np.tile(xm, (2, 1)), dtype=mx.float64))
    assert out.dtype == mx.float64
    np.testing.assert_allclose(np.asarray(out)[0], f0, atol=1e-11)
    # a float32 input (mx.array's default for NumPy float64) is upcast and evaluated on the CPU stream
    out32 = ev.eval_macro_batched_mlx(mx.array(np.tile(xm, (2, 1))))
    assert out32.dtype == mx.float64
    np.testing.assert_allclose(np.asarray(out32)[0], f0, atol=1e-4)
    # 'cpu' / 'gpu' string streams are accepted; the default Jacobian is the float64 one
    J_cpu = ev.evaluate_batched_jacobian(xm, f_base=f0, stream="cpu")
    J_default = ev.evaluate_batched_jacobian(xm, f_base=f0)
    np.testing.assert_allclose(J_default, J_cpu, rtol=0, atol=0)
    J_gpu = ev.evaluate_batched_jacobian(xm, f_base=f0, stream="gpu")
    assert J_gpu.shape == J_cpu.shape and np.all(np.isfinite(J_gpu))


@pytest.mark.skipif(not HAS_MLX, reason="Apple MLX is not available")
def test_explicit_float32_device_warns_and_polishes_in_float64(calib3, shock3):
    tau, tau_fd = shock3
    ref = solve_trade_equilibrium(calib3, tau=tau, tau_fd=tau_fd, method="condensed")
    with pytest.warns(RuntimeWarning, match="only supports float32"):
        res = solve_trade_equilibrium_gpu(calib3, tau=tau, tau_fd=tau_fd, device="mlx")
    assert res.converged
    assert res.metadata["requested_device"] == "gpu"
    assert res.metadata["jacobian_device"] == "mlx:cpu"
    assert set(res.metadata["jacobian_evaluations"]) == {"mlx:cpu:float64"}
    np.testing.assert_allclose(res.w_sol, ref.w_sol, atol=1e-5)


# ---------------------------------------------------------------------------
# Condensed solver backend
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not backend_available("mlx"), reason="Apple MLX is not available")
def test_condensed_mlx_backend_matches_numpy_on_iterating_problem(calib3, shock3):
    tau, tau_fd = shock3
    res_np = solve_trade_equilibrium(calib3, tau=tau, tau_fd=tau_fd, method="condensed", backend="numpy")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # no fallback allowed
        res_mx = solve_trade_equilibrium(calib3, tau=tau, tau_fd=tau_fd, method="condensed", backend="mlx")
    assert res_np.converged and res_mx.converged
    assert res_np.iterations > 0, "the test problem must actually iterate"
    assert res_mx.metadata["backend"] == "mlx"
    np.testing.assert_allclose(res_mx.x_sol, res_np.x_sol, atol=1e-8)
    np.testing.assert_allclose(res_mx.w_sol, res_np.w_sol, atol=1e-8)


@pytest.mark.skipif(not backend_available("mlx"), reason="Apple MLX is not available")
def test_condensed_mlx_non_convergence_falls_back_to_numpy_with_warning(calib3, shock3):
    tau, tau_fd = shock3
    with pytest.warns(RuntimeWarning, match="Backend 'mlx' did not converge .* falling back to 'numpy'"):
        res = solve_trade_equilibrium(calib3, tau=tau, tau_fd=tau_fd, method="condensed", backend="mlx", max_iter=1)
    ref = solve_trade_equilibrium(calib3, tau=tau, tau_fd=tau_fd, method="condensed", backend="numpy", max_iter=1)
    assert res.converged == ref.converged
    np.testing.assert_allclose(res.x_sol, ref.x_sol, atol=1e-12)


def test_compute_equilibrium_residuals_rejects_unknown_kwargs(calib3):
    x0 = solve_trade_equilibrium(calib3, method="condensed").x_sol
    with pytest.raises(TypeError, match="fiscal_closur"):
        compute_equilibrium_residuals(x0, calib3, fiscal_closur="labor_tax")
    with pytest.raises(TypeError, match="sigma_typo"):
        compute_equilibrium_residuals(x0, calib3, sigma_typo=0.5)


# ---------------------------------------------------------------------------
# TariffScenario tensor builders and table layout
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def calib_can_usa():
    return calibrate_trade_model(
        _make_io_table(2, 2, 3), ns=2, nc=2, nfd=3, validate=True,
        country_codes=("CAN", "USA"), sector_codes=("S1", "S2"),
    )


def test_tariff_scenario_builders_match_solver_layout_for_non_canonical_codes(calib_can_usa):
    ns, nc, nfd = 2, 2, 3
    scen = TariffScenario(name="t25", us_import_tariffs={"CAN": 0.25}, sectoral_tariffs={("CAN", "USA", "S2"): 0.5})
    tau, tau_fd, _, _ = build_tariff_matrices(scen, calib_can_usa)
    ref = tau.transpose(1, 0, 2, 3).reshape(ns * nc, ns, nc)
    ref_fd = tau_fd.transpose(1, 0, 2, 3).reshape(ns * nc, nfd, nc)
    assert int((ref != 1.0).sum()) == 4

    got = scen.build_intermediate_tariffs(ns, nc, country_codes=calib_can_usa.country_codes, sector_codes=calib_can_usa.sector_codes)
    got_fd = scen.build_final_demand_tariffs(ns, nc, nfd, country_codes=calib_can_usa.country_codes, sector_codes=calib_can_usa.sector_codes)
    np.testing.assert_array_equal(got, ref)
    np.testing.assert_array_equal(got_fd, ref_fd)

    # Mapping a 2-country model onto the first two canonical codes used to return all ones silently.
    with pytest.raises(ValueError, match="nc=2 does not match the 77 canonical country codes"):
        scen.build_intermediate_tariffs(ns, nc)
    with pytest.raises(ValueError, match="country_codes has 1 entries"):
        scen.build_final_demand_tariffs(ns, nc, nfd, country_codes=("CAN",))
    with pytest.raises(ValueError, match="sectoral_tariffs"):
        scen.build_intermediate_tariffs(45, 77)


def test_tariff_scenario_builders_canonical_defaults():
    scen = TariffScenario(name="t10_25", default_us_tariff=0.10, us_import_tariffs={"CAN": 0.25})
    tau_a = scen.build_intermediate_tariffs()
    assert tau_a.shape == (11 * 77, 11, 77)
    usa = list(__import__("puremacro.trade.data", fromlist=["CANONICAL_COUNTRY_CODES"]).CANONICAL_COUNTRY_CODES).index("USA")
    can = list(__import__("puremacro.trade.data", fromlist=["CANONICAL_COUNTRY_CODES"]).CANONICAL_COUNTRY_CODES).index("CAN")
    assert np.all(tau_a[can * 11 : (can + 1) * 11, :, usa] == 1.25)
    assert np.all(tau_a[usa * 11 : (usa + 1) * 11, :, usa] == 1.0)
    assert np.all(tau_a[:, :, can] == 1.0)
    # no sectoral overrides -> placeholder sector codes are fine for a 45-sector layout
    assert scen.build_final_demand_tariffs(45, 77).shape == (45 * 77, 3, 77)


def test_legacy_compat_only_applies_to_five_scenario_batches(calib_can_usa):
    batch = run_scenario_batch(
        [
            TariffScenario(name="base"),
            TariffScenario(name="t10", default_us_tariff=0.10),
            TariffScenario(name="t25", default_us_tariff=0.25),
        ],
        calib_can_usa,
        method="condensed",
    )
    legacy = to_latex_selected_country_table(batch, countries=["CAN", "USA"], legacy_compat=True)
    clean = to_latex_selected_country_table(batch, countries=["CAN", "USA"], legacy_compat=False)
    assert legacy == clean
    assert r"\begin{tabular}{lrrr}" in legacy
    assert "Country & t10 & t25 & Base \\\\" in legacy
    with_row = to_latex_selected_country_with_row(batch, legacy_compat=True)
    assert "& Base \\\\" in with_row


# ---------------------------------------------------------------------------
# Allen-Arkolakis backend tolerance
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not backend_available("mlx"), reason="Apple MLX is not available")
def test_allen_arkolakis_mlx_honours_requested_tolerance():
    from puremacro.spatial.allen_arkolakis import AllenArkolakisModel

    rng = np.random.default_rng(0)
    model = AllenArkolakisModel.from_coordinates(rng.uniform(0.0, 10.0, size=(40, 2)))
    res_np = model.solve_equilibrium(tol=1e-10, backend="numpy")
    res_mx = model.solve_equilibrium(tol=1e-10, backend="mlx")
    assert res_np.converged and res_mx.converged
    assert res_mx.max_residual < 1e-10, "converged=True must mean the requested tol was met"
    np.testing.assert_allclose(res_mx.wages, res_np.wages, atol=1e-8, rtol=0)
    np.testing.assert_allclose(res_mx.population, res_np.population, atol=1e-8, rtol=0)
    assert abs(res_mx.welfare - res_np.welfare) < 1e-10
