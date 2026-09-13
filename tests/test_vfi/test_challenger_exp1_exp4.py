"""Adversarial Empirical Verification Suite for Experiment 1 and Experiment 4.

Tasks:
1. Experiment 1: Smooth Neoclassical Growth (Brock-Mirman 1972)
   - Extreme parameter stress tests (alpha=0.85, beta=0.99, wide capital boundaries).
   - Empirical spectral convergence O(c^-N) for Chebyshev Collocation vs quadratic O(h^2) for FEM.
   - Dense out-of-sample Euler equation residuals on 5,000+ points.
   - Economic monotonicity and feasibility (strictly positive consumption, non-explosive dynamics).
2. Experiment 4: Multi-Backend Runtime & Scaling
   - Numerical parity across NumPy, Numba, and MLX (max relative difference < 1e-4).
   - Defensive graceful fallback for unavailable backends (e.g. CuPy on macOS) with zero unhandled exceptions.
   - Strict validation of invalid backend requests raising ValueError.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pytest

from puremacro import _backend as bk
from puremacro.vfi.collocation import CollocationProblem, CollocationSolution
from puremacro.vfi.fem import FEMProblem, FEMSolution


# ===========================================================================
# Fixtures & Analytical Reference Oracles
# ===========================================================================

@pytest.fixture
def canonical_bm() -> Dict[str, Any]:
    """Canonical Brock-Mirman (1972) neoclassical growth calibration."""
    alpha = 0.36
    beta = 0.96
    delta = 1.0
    k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
    domain = (0.5 * k_ss, 1.5 * k_ss)

    def g_star(k: np.ndarray | float) -> np.ndarray | float:
        k_arr = np.asarray(k, dtype=np.float64)
        val = alpha * beta * (k_arr**alpha)
        return float(val) if np.ndim(k) == 0 else val

    return {
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "k_ss": k_ss,
        "domain": domain,
        "g_star": g_star,
    }


def compute_euler_residual_1d(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    k_grid: np.ndarray,
    alpha: float = 0.36,
    beta: float = 0.96,
    delta: float = 1.0,
) -> np.ndarray:
    """Evaluate continuous Euler equation residual: R(k) = 1 - beta * (c / c') * f'(k')."""
    k = np.asarray(k_grid, dtype=np.float64)
    kp = policy_fn(k)
    kpp = policy_fn(kp)

    c = np.maximum(k**alpha + (1.0 - delta) * k - kp, 1e-14)
    cp = np.maximum(kp**alpha + (1.0 - delta) * kp - kpp, 1e-14)
    fkp = alpha * (kp ** (alpha - 1.0)) + (1.0 - delta)

    return 1.0 - beta * (c / cp) * fkp


# ===========================================================================
# 1. Adversarial Verification of Experiment 1 (Smooth Neoclassical Growth)
# ===========================================================================

class TestAdversarialExperiment1:
    """Adversarial stress-testing of Experiment 1 (Smooth Neoclassical Growth)."""

    @pytest.mark.parametrize(
        "alpha,beta",
        [
            (0.25, 0.90),
            (0.25, 0.99),
            (0.36, 0.96),
            (0.60, 0.96),
            (0.85, 0.90),
            (0.85, 0.99),
        ],
    )
    def test_extreme_alpha_beta_sweeps(self, alpha: float, beta: float):
        """Verify solver stability and precision across extreme capital share and discount factors."""
        k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
        domain = (0.5 * k_ss, 1.5 * k_ss)
        eval_k = np.linspace(domain[0], domain[1], 5000)

        def g_star(k: np.ndarray) -> np.ndarray:
            return alpha * beta * (np.asarray(k, dtype=np.float64) ** alpha)

        # 1. Collocation N=8
        prob_c = CollocationProblem(
            domain=domain,
            orders=8,
            method="euler",
            params={"alpha": alpha, "delta": 1.0},
            beta=beta,
        )
        sol_c = prob_c.solve(backend="numpy")
        pol_c = sol_c.policy(eval_k)
        assert not np.any(np.isnan(pol_c)), f"Collocation policy contained NaN for alpha={alpha}, beta={beta}"
        assert not np.any(np.isinf(pol_c)), f"Collocation policy contained Inf for alpha={alpha}, beta={beta}"

        rel_err_c = float(np.max(np.abs(pol_c - g_star(eval_k)) / g_star(eval_k)))
        assert rel_err_c < 1e-4, f"Collocation relative error {rel_err_c:.2e} >= 1e-4 for alpha={alpha}, beta={beta}"

        # 2. FEM E=50
        prob_f = FEMProblem(
            domain=domain,
            elements=50,
            method="euler",
            projection="galerkin",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k**alpha,
            beta=beta,
            params={"alpha": alpha, "gamma": 1.0},
        )
        sol_f = prob_f.solve(backend="numpy")
        pol_f = sol_f.policy(eval_k)
        assert not np.any(np.isnan(pol_f)), f"FEM policy contained NaN for alpha={alpha}, beta={beta}"
        assert not np.any(np.isinf(pol_f)), f"FEM policy contained Inf for alpha={alpha}, beta={beta}"

        rel_err_f = float(np.max(np.abs(pol_f - g_star(eval_k)) / g_star(eval_k)))
        assert rel_err_f < 1e-4, f"FEM relative error {rel_err_f:.2e} >= 1e-4 for alpha={alpha}, beta={beta}"

    def test_extreme_regime_alpha085_beta099_wide_domain(self):
        """Adversarial stress test on extreme regime: alpha=0.85, beta=0.99, wide domain [0.2*k_ss, 3.0*k_ss]."""
        alpha = 0.85
        beta = 0.99
        k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
        domain = (0.2 * k_ss, 3.0 * k_ss)
        eval_k = np.linspace(domain[0], domain[1], 5000)

        def g_star(k: np.ndarray) -> np.ndarray:
            return alpha * beta * (np.asarray(k, dtype=np.float64) ** alpha)

        # Collocation N=12
        prob_c = CollocationProblem(
            domain=domain,
            orders=12,
            method="euler",
            params={"alpha": alpha, "delta": 1.0},
            beta=beta,
        )
        sol_c = prob_c.solve(backend="numpy")
        pol_c = sol_c.policy(eval_k)
        rel_err_c = float(np.max(np.abs(pol_c - g_star(eval_k)) / g_star(eval_k)))
        assert rel_err_c < 1e-4, f"Collocation N=12 relative error {rel_err_c:.2e} on extreme domain >= 1e-4"

        res_c = compute_euler_residual_1d(sol_c.policy, eval_k, alpha=alpha, beta=beta, delta=1.0)
        max_res_c = float(np.max(np.abs(res_c)))
        assert max_res_c < 1e-4, f"Collocation N=12 max Euler residual {max_res_c:.2e} >= 1e-4"

        # FEM E=80
        prob_f = FEMProblem(
            domain=domain,
            elements=80,
            method="euler",
            projection="galerkin",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k**alpha,
            beta=beta,
            params={"alpha": alpha, "gamma": 1.0},
        )
        sol_f = prob_f.solve(backend="numpy")
        pol_f = sol_f.policy(eval_k)
        rel_err_f = float(np.max(np.abs(pol_f - g_star(eval_k)) / g_star(eval_k)))
        assert rel_err_f < 5e-4, f"FEM E=80 relative error {rel_err_f:.2e} on extreme domain >= 5e-4"

    def test_empirical_spectral_convergence_dense_grid(self, canonical_bm):
        """Verify exponential/spectral error decay O(c^-N) for Chebyshev Collocation on 5,000 dense points."""
        bm = canonical_bm
        eval_k = np.linspace(bm["domain"][0], bm["domain"][1], 5000)
        orders = [4, 6, 8, 10, 12]
        l2_errors: List[float] = []

        for N in orders:
            prob = CollocationProblem(
                domain=bm["domain"],
                orders=N,
                method="euler",
                params={"alpha": bm["alpha"], "delta": bm["delta"]},
                beta=bm["beta"],
            )
            sol = prob.solve(backend="numpy")
            pol = sol.policy(eval_k)
            l2_err = float(np.sqrt(np.mean((pol - bm["g_star"](eval_k)) ** 2)))
            l2_errors.append(l2_err)

        # 1. Strict monotonic decay
        for i in range(len(l2_errors) - 1):
            assert l2_errors[i + 1] < l2_errors[i], (
                f"Collocation error did not decay monotonically from N={orders[i]} "
                f"({l2_errors[i]:.2e}) to N={orders[i+1]} ({l2_errors[i+1]:.2e})"
            )

        # 2. Log-linear spectral regression: log(Error_N) = -c * N + d
        log_errs = np.log(l2_errors)
        slope, intercept = np.polyfit(orders, log_errs, 1)
        # Coefficient of determination R^2
        ss_tot = float(np.sum((log_errs - np.mean(log_errs)) ** 2))
        ss_res = float(np.sum((log_errs - (slope * np.array(orders) + intercept)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot)

        # Assert exponential decay rate c > 0 and high linear correlation R^2 > 0.95
        assert slope < -0.8, f"Decay slope {slope:.3f} indicates slower than expected exponential convergence"
        assert r2 > 0.95, f"Log-linear fit R^2 {r2:.4f} < 0.95 (spectral decay hypothesis violated)"
        # Final error at N=12 below 1e-9
        assert l2_errors[-1] < 1e-9, f"N=12 L2 error {l2_errors[-1]:.2e} >= 1e-9"

    def test_empirical_quadratic_fem_convergence_dense_grid(self, canonical_bm):
        """Verify algebraic quadratic error decay O(h^2) for FEM Galerkin on 5,000 dense points."""
        bm = canonical_bm
        eval_k = np.linspace(bm["domain"][0], bm["domain"][1], 5000)
        elements = [10, 20, 40, 80, 160]
        l2_errors: List[float] = []

        for E in elements:
            prob = FEMProblem(
                domain=bm["domain"],
                elements=E,
                method="euler",
                projection="galerkin",
                return_fn=lambda c: np.log(c),
                transition_fn=lambda k: k ** bm["alpha"],
                beta=bm["beta"],
                params={"alpha": bm["alpha"], "gamma": 1.0},
            )
            sol = prob.solve(backend="numpy")
            pol = sol.policy(eval_k)
            l2_err = float(np.sqrt(np.mean((pol - bm["g_star"](eval_k)) ** 2)))
            l2_errors.append(l2_err)

        # 1. Strict monotonic decay
        for i in range(len(l2_errors) - 1):
            assert l2_errors[i + 1] < l2_errors[i], (
                f"FEM error did not decay monotonically from E={elements[i]} to E={elements[i+1]}"
            )

        # 2. Empirical order of convergence: EOC = - (log(err_{k+1}) - log(err_k)) / (log(E_{k+1}) - log(E_k))
        eoc_list: List[float] = []
        for i in range(len(elements) - 1):
            eoc = - (np.log(l2_errors[i + 1]) - np.log(l2_errors[i])) / (
                np.log(elements[i + 1]) - np.log(elements[i])
            )
            eoc_list.append(float(eoc))

        for idx, eoc in enumerate(eoc_list):
            assert 1.70 <= eoc <= 2.30, (
                f"FEM step E={elements[idx]}->{elements[idx+1]} EOC {eoc:.2f} deviated from theoretical 2.0 (O(h^2))"
            )

    def test_out_of_sample_dense_euler_residuals(self, canonical_bm):
        """Verify continuous Euler residuals across 5,000 strictly out-of-sample uniform points."""
        bm = canonical_bm
        # Strictly random out-of-sample continuous points
        np.random.seed(12345)
        k_out_of_sample = np.sort(np.random.uniform(bm["domain"][0], bm["domain"][1], 5000))

        # Collocation N=8
        prob_c = CollocationProblem(
            domain=bm["domain"],
            orders=8,
            method="euler",
            params={"alpha": bm["alpha"], "delta": bm["delta"]},
            beta=bm["beta"],
        )
        sol_c = prob_c.solve(backend="numpy")
        res_c = compute_euler_residual_1d(sol_c.policy, k_out_of_sample, alpha=bm["alpha"], beta=bm["beta"])
        max_res_c = float(np.max(np.abs(res_c)))
        assert max_res_c < 1e-4, f"Collocation N=8 max out-of-sample Euler residual {max_res_c:.2e} >= 1e-4"

        # FEM E=50
        prob_f = FEMProblem(
            domain=bm["domain"],
            elements=50,
            method="euler",
            projection="galerkin",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"], "gamma": 1.0},
        )
        sol_f = prob_f.solve(backend="numpy")
        res_f = compute_euler_residual_1d(sol_f.policy, k_out_of_sample, alpha=bm["alpha"], beta=bm["beta"])
        max_res_f = float(np.max(np.abs(res_f)))
        assert max_res_f < 1e-4, f"FEM E=50 max out-of-sample Euler residual {max_res_f:.2e} >= 1e-4"

    def test_policy_monotonicity_and_consumption_feasibility(self, canonical_bm):
        """Verify economic properties: strict monotonicity dg/dk > 0 and positive consumption."""
        bm = canonical_bm
        eval_k = np.linspace(bm["domain"][0], bm["domain"][1], 5000)

        # Check Collocation
        prob_c = CollocationProblem(
            domain=bm["domain"],
            orders=8,
            method="euler",
            params={"alpha": bm["alpha"], "delta": bm["delta"]},
            beta=bm["beta"],
        )
        sol_c = prob_c.solve(backend="numpy")
        pol_c = sol_c.policy(eval_k)
        diff_c = np.diff(pol_c)
        assert np.all(diff_c > 0), "Collocation policy is not strictly monotonically increasing"
        c_c = eval_k**bm["alpha"] - pol_c
        assert np.all(c_c > 0), "Collocation implied consumption is non-positive"

        # Check FEM
        prob_f = FEMProblem(
            domain=bm["domain"],
            elements=50,
            method="euler",
            projection="galerkin",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"], "gamma": 1.0},
        )
        sol_f = prob_f.solve(backend="numpy")
        pol_f = sol_f.policy(eval_k)
        diff_f = np.diff(pol_f)
        assert np.all(diff_f > 0), "FEM policy is not strictly monotonically increasing"
        c_f = eval_k**bm["alpha"] - pol_f
        assert np.all(c_f > 0), "FEM implied consumption is non-positive"


# ===========================================================================
# 2. Adversarial Verification of Experiment 4 (Multi-Backend Acceleration)
# ===========================================================================

class TestAdversarialExperiment4:
    """Adversarial testing of Experiment 4 (Multi-Backend Runtime & Scaling)."""

    def test_multi_backend_numerical_parity(self, canonical_bm):
        """Verify numerical parity across all available backends: max relative difference < 1e-4."""
        bm = canonical_bm
        eval_k = np.linspace(bm["domain"][0], bm["domain"][1], 5000)

        # Baseline NumPy solutions
        prob_c = CollocationProblem(
            domain=bm["domain"],
            orders=8,
            method="euler",
            params={"alpha": bm["alpha"], "delta": bm["delta"]},
            beta=bm["beta"],
        )
        sol_c_np = prob_c.solve(backend="numpy")
        pol_c_np = sol_c_np.policy(eval_k)

        prob_f = FEMProblem(
            domain=bm["domain"],
            elements=50,
            method="euler",
            projection="galerkin",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"], "gamma": 1.0},
        )
        sol_f_np = prob_f.solve(backend="numpy")
        pol_f_np = sol_f_np.policy(eval_k)

        for backend in ["numpy", "numba", "mlx"]:
            if not bk.backend_available(backend):
                continue

            # Collocation parity
            sol_c = prob_c.solve(backend=backend)
            pol_c = sol_c.policy(eval_k)
            rel_diff_c = float(np.max(np.abs(pol_c - pol_c_np) / np.maximum(pol_c_np, 1e-12)))
            assert rel_diff_c < 1e-4, (
                f"Collocation parity error on backend '{backend}' {rel_diff_c:.2e} >= 1e-4"
            )

            # FEM parity
            sol_f = prob_f.solve(backend=backend)
            pol_f = sol_f.policy(eval_k)
            rel_diff_f = float(np.max(np.abs(pol_f - pol_f_np) / np.maximum(pol_f_np, 1e-12)))
            assert rel_diff_f < 1e-4, (
                f"FEM parity error on backend '{backend}' {rel_diff_f:.2e} >= 1e-4"
            )

    def test_graceful_fallback_unavailable_backend(self, canonical_bm):
        """Verify that requesting an unavailable backend (CuPy on macOS) falls back gracefully to NumPy."""
        bm = canonical_bm
        eval_k = np.linspace(bm["domain"][0], bm["domain"][1], 2000)

        # Reference NumPy solves
        prob_c = CollocationProblem(
            domain=bm["domain"],
            orders=8,
            method="euler",
            params={"alpha": bm["alpha"], "delta": bm["delta"]},
            beta=bm["beta"],
        )
        sol_c_np = prob_c.solve(backend="numpy")
        pol_c_np = sol_c_np.policy(eval_k)

        prob_f = FEMProblem(
            domain=bm["domain"],
            elements=50,
            method="euler",
            projection="galerkin",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"], "gamma": 1.0},
        )
        sol_f_np = prob_f.solve(backend="numpy")
        pol_f_np = sol_f_np.policy(eval_k)

        # 1. Collocation CuPy fallback
        with warnings.catch_warnings(record=True) as w_c:
            warnings.simplefilter("always")
            sol_c_fallback = prob_c.solve(backend="cupy")
            pol_c_fallback = sol_c_fallback.policy(eval_k)

            # Assert warning was raised with fallback message
            assert len(w_c) >= 1, "Collocation failed to emit warning for unavailable CuPy backend"
            warn_msg = str(w_c[-1].message)
            assert "cupy" in warn_msg and "falling back to 'numpy'" in warn_msg, (
                f"Unexpected warning message: {warn_msg}"
            )
            # Assert exact match to NumPy reference (< 1e-12)
            max_diff_c = float(np.max(np.abs(pol_c_fallback - pol_c_np)))
            assert max_diff_c < 1e-12, f"Collocation fallback differed from NumPy: diff={max_diff_c:.2e}"

        # 2. FEM CuPy fallback
        with warnings.catch_warnings(record=True) as w_f:
            warnings.simplefilter("always")
            sol_f_fallback = prob_f.solve(backend="cupy")
            pol_f_fallback = sol_f_fallback.policy(eval_k)

            assert len(w_f) >= 1, "FEM failed to emit warning for unavailable CuPy backend"
            warn_msg = str(w_f[-1].message)
            assert "cupy" in warn_msg and "falling back to 'numpy'" in warn_msg, (
                f"Unexpected warning message: {warn_msg}"
            )
            max_diff_f = float(np.max(np.abs(pol_f_fallback - pol_f_np)))
            assert max_diff_f < 1e-12, f"FEM fallback differed from NumPy: diff={max_diff_f:.2e}"

    def test_unsupported_backend_raises_value_error(self, canonical_bm):
        """Verify that requesting an unrecognized backend raises clean ValueError without crash."""
        bm = canonical_bm
        prob_c = CollocationProblem(
            domain=bm["domain"], orders=8, method="euler", params={"alpha": bm["alpha"]}, beta=bm["beta"]
        )
        prob_f = FEMProblem(
            domain=bm["domain"],
            elements=20,
            method="euler",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            params={"alpha": bm["alpha"]},
            beta=bm["beta"],
        )

        with pytest.raises(ValueError, match="Unknown backend 'invalid_backend'"):
            prob_c.solve(backend="invalid_backend")

        with pytest.raises(ValueError, match="Unknown backend 'invalid_backend'"):
            prob_f.solve(backend="invalid_backend")

    def test_multi_backend_extreme_parameter_parity(self):
        """Verify numerical parity across backends on extreme parameter calibration (alpha=0.85, beta=0.99)."""
        alpha = 0.85
        beta = 0.99
        k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
        domain = (0.5 * k_ss, 1.5 * k_ss)
        eval_k = np.linspace(domain[0], domain[1], 5000)

        prob_c = CollocationProblem(
            domain=domain,
            orders=8,
            method="euler",
            params={"alpha": alpha, "delta": 1.0},
            beta=beta,
        )
        sol_c_np = prob_c.solve(backend="numpy")
        pol_c_np = sol_c_np.policy(eval_k)

        prob_f = FEMProblem(
            domain=domain,
            elements=50,
            method="euler",
            projection="galerkin",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k**alpha,
            beta=beta,
            params={"alpha": alpha, "gamma": 1.0},
        )
        sol_f_np = prob_f.solve(backend="numpy")
        pol_f_np = sol_f_np.policy(eval_k)

        for backend in ["numpy", "numba", "mlx"]:
            if not bk.backend_available(backend):
                continue

            # Collocation
            sol_c = prob_c.solve(backend=backend)
            diff_c = float(np.max(np.abs(sol_c.policy(eval_k) - pol_c_np) / np.maximum(pol_c_np, 1e-12)))
            assert diff_c < 1e-4, f"Collocation backend '{backend}' extreme parity error {diff_c:.2e} >= 1e-4"

            # FEM
            sol_f = prob_f.solve(backend=backend)
            diff_f = float(np.max(np.abs(sol_f.policy(eval_k) - pol_f_np) / np.maximum(pol_f_np, 1e-12)))
            assert diff_f < 1e-4, f"FEM backend '{backend}' extreme parity error {diff_f:.2e} >= 1e-4"
