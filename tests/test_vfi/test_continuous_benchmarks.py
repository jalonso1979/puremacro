"""Canonical benchmark and verification test suite for continuous projection solvers in puremacro.vfi.

Verifies:
1. Public API exports and imports from `puremacro.vfi`:
   `CollocationBasis`, `CollocationProblem`, `CollocationSolution`, `solve_collocation`,
   `FEMMesh`, `FEMProblem`, `FEMSolution`, `solve_fem`.
2. Closed-form analytical benchmark: Brock-Mirman (1972) neoclassical growth model:
   - True policy: g*(k) = alpha * beta * k^alpha
   - True value: V*(k) = (alpha / (1 - alpha * beta)) * ln(k) + const
   - Policy relative error: max |g(k) - g*(k)| / g*(k) < 10^-4 on >= 1,000 continuous points
     for both Collocation and FEM solvers.
3. Continuous Euler equation residuals:
   - max |1 - beta * (u'(c') / u'(c)) * f'(k')| < 10^-4 across >= 1,000 continuous points
     for both Collocation and FEM solvers.
4. Stochastic Brock-Mirman neoclassical growth with discrete TFP productivity shocks:
   - True policy: g*(k, z) = alpha * beta * z * k^alpha
   - Policy relative error < 10^-4 across all shock regimes.
   - Monotonic policy ordering across productivity levels.
5. Multi-backend parity and acceleration (NumPy, Numba, MLX):
   - Consistent solutions across backends with relative difference < 10^-4 on dense grids.
   - Graceful fallback and error reporting for unavailable optional backends.
6. Puremacro presentation contract compliance:
   - .summary(), .plot(), .to_frame(), .to_markdown(), .to_latex(), .to_typst().
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for CI/test runners
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro import _backend as bk
import puremacro.vfi as vfi
from puremacro.vfi import (
    CollocationBasis,
    CollocationProblem,
    CollocationSolution,
    FEMMesh,
    FEMProblem,
    FEMSolution,
    solve_collocation,
    solve_fem,
)


# ===========================================================================
# Analytical Fixtures: Brock-Mirman (1972) Neoclassical Growth Model
# ===========================================================================

@pytest.fixture
def bm_params() -> Dict[str, Any]:
    """Return canonical Brock-Mirman parameters and exact closed-form functions."""
    alpha = 0.36
    beta = 0.96
    k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
    k_min = 0.5 * k_ss
    k_max = 1.5 * k_ss
    domain = (k_min, k_max)

    A = alpha / (1.0 - alpha * beta)
    B = (
        np.log(1.0 - alpha * beta)
        + (alpha * beta * np.log(alpha * beta)) / (1.0 - alpha * beta)
    ) / (1.0 - beta)

    def g_star(k: np.ndarray | float, z: float = 1.0) -> np.ndarray | float:
        k_arr = np.asarray(k, dtype=np.float64)
        val = alpha * beta * z * (k_arr**alpha)
        return float(val) if np.ndim(k) == 0 else val

    def c_star(k: np.ndarray | float, z: float = 1.0) -> np.ndarray | float:
        k_arr = np.asarray(k, dtype=np.float64)
        val = (1.0 - alpha * beta) * z * (k_arr**alpha)
        return float(val) if np.ndim(k) == 0 else val

    def V_star(k: np.ndarray | float) -> np.ndarray | float:
        k_arr = np.asarray(k, dtype=np.float64)
        val = A * np.log(k_arr) + B
        return float(val) if np.ndim(k) == 0 else val

    return {
        "alpha": alpha,
        "beta": beta,
        "delta": 1.0,
        "k_ss": k_ss,
        "domain": domain,
        "A": A,
        "B": B,
        "g_star": g_star,
        "c_star": c_star,
        "V_star": V_star,
        "transition_fn": lambda k: k**alpha,
        "return_fn": lambda c: np.log(np.maximum(c, 1e-14)),
    }


def compute_continuous_euler_residual(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    eval_grid: np.ndarray,
    alpha: float = 0.36,
    beta: float = 0.96,
    z: float = 1.0,
) -> np.ndarray:
    """Evaluate continuous Euler equation residuals across an arbitrary evaluation grid."""
    k = np.asarray(eval_grid, dtype=np.float64)
    kp = policy_fn(k)
    kpp = policy_fn(kp)
    c = np.maximum(z * (k**alpha) - kp, 1e-12)
    cp = np.maximum(z * (kp**alpha) - kpp, 1e-12)
    fkp = alpha * z * (kp ** (alpha - 1.0))
    return 1.0 - beta * (c / cp) * fkp


# ===========================================================================
# 1. Public API Exports & Imports
# ===========================================================================

class TestPublicAPIExports:
    """Verify that all continuous projection classes and solvers are exported from puremacro.vfi."""

    def test_vfi_all_contains_continuous_solvers(self):
        """Verify __all__ contains Collocation and FEM public symbols."""
        expected_symbols = [
            "CollocationBasis",
            "CollocationProblem",
            "CollocationSolution",
            "solve_collocation",
            "FEMMesh",
            "FEMProblem",
            "FEMSolution",
            "solve_fem",
        ]
        for sym in expected_symbols:
            assert sym in vfi.__all__, f"{sym} is missing from puremacro.vfi.__all__"
            assert hasattr(vfi, sym), f"{sym} is not accessible on puremacro.vfi"

    def test_direct_imports_from_puremacro_vfi(self):
        """Verify direct import statement works identically to submodule imports."""
        from puremacro.vfi import (
            CollocationBasis as CB,
            CollocationProblem as CP,
            CollocationSolution as CS,
            FEMMesh as FM,
            FEMProblem as FP,
            FEMSolution as FS,
            solve_collocation as sc,
            solve_fem as sf,
        )

        assert CB is CollocationBasis
        assert CP is CollocationProblem
        assert CS is CollocationSolution
        assert FM is FEMMesh
        assert FP is FEMProblem
        assert FS is FEMSolution
        assert sc is solve_collocation
        assert sf is solve_fem

    def test_functional_entry_points(self, bm_params):
        """Verify high-level solve_collocation and solve_fem functional APIs."""
        p = bm_params

        # Functional solve_collocation
        sol_c = solve_collocation(
            domain=p["domain"],
            orders=6,
            method="euler",
            params={"alpha": p["alpha"], "delta": 1.0},
            beta=p["beta"],
        )
        assert isinstance(sol_c, CollocationSolution)
        assert sol_c.converged

        # Functional solve_fem with domain passed positionally
        sol_f = solve_fem(
            p["domain"],
            elements=40,
            method="euler",
            projection="galerkin",
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=p["beta"],
            params={"alpha": p["alpha"], "gamma": 1.0},
        )
        assert isinstance(sol_f, FEMSolution)
        assert sol_f.converged

        # Functional solve_fem with FEMProblem instance
        prob_f = FEMProblem(
            domain=p["domain"],
            elements=30,
            method="euler",
            projection="galerkin",
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=p["beta"],
            params={"alpha": p["alpha"], "gamma": 1.0},
        )
        sol_f_prob = solve_fem(prob_f)
        assert isinstance(sol_f_prob, FEMSolution)
        assert sol_f_prob.converged


# ===========================================================================
# 2. Brock-Mirman Analytical Benchmark: Policy Relative Error < 10^-4
# ===========================================================================

class TestBrockMirmanAnalyticalBenchmark:
    """Benchmark Collocation and FEM against analytical Brock-Mirman solution."""

    def test_collocation_euler_policy_relative_error_1000_points(self, bm_params):
        """Verify Collocation Euler policy achieves relative error < 10^-4 on 1,000 points."""
        p = bm_params
        domain = p["domain"]
        alpha = p["alpha"]
        beta = p["beta"]

        prob = CollocationProblem(
            domain=domain,
            orders=6,
            method="euler",
            params={"alpha": alpha, "delta": 1.0},
            beta=beta,
        )
        sol = prob.solve(backend="numpy")
        assert sol.converged

        eval_k = np.linspace(domain[0], domain[1], 1000)
        g_approx = sol.policy(eval_k)
        g_true = p["g_star"](eval_k)

        max_rel_error = np.max(np.abs(g_approx - g_true) / g_true)
        assert max_rel_error < 1e-4, f"Collocation relative policy error {max_rel_error:.2e} >= 1e-4"

    def test_fem_euler_policy_relative_error_1000_points(self, bm_params):
        """Verify FEM Galerkin policy achieves relative error < 10^-4 on 1,000 points."""
        p = bm_params
        domain = p["domain"]
        alpha = p["alpha"]
        beta = p["beta"]

        prob = FEMProblem(
            domain=domain,
            elements=50,
            method="euler",
            projection="galerkin",
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=beta,
            params={"alpha": alpha, "gamma": 1.0},
        )
        sol = prob.solve(backend="numpy")
        assert sol.converged

        eval_k = np.linspace(domain[0], domain[1], 1000)
        g_approx = sol.policy(eval_k)
        g_true = p["g_star"](eval_k)

        max_rel_error = np.max(np.abs(g_approx - g_true) / g_true)
        assert max_rel_error < 1e-4, f"FEM relative policy error {max_rel_error:.2e} >= 1e-4"

    def test_fem_collocation_euler_policy_relative_error(self, bm_params):
        """Verify FEM nodal collocation achieves accurate policy approximation."""
        p = bm_params
        prob = FEMProblem(
            domain=p["domain"],
            elements=40,
            method="euler",
            projection="collocation",
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=p["beta"],
            params={"alpha": p["alpha"], "gamma": 1.0},
        )
        sol = prob.solve(backend="numpy")
        assert sol.converged

        eval_k = np.linspace(p["domain"][0], p["domain"][1], 1000)
        g_approx = sol.policy(eval_k)
        g_true = p["g_star"](eval_k)

        max_rel_error = np.max(np.abs(g_approx - g_true) / g_true)
        assert max_rel_error < 1e-3

    def test_collocation_bellman_value_and_policy_benchmark(self, bm_params):
        """Verify continuous Bellman Collocation recovers V*(k) = A*ln(k) + B and g*(k)."""
        p = bm_params
        domain = p["domain"]
        alpha = p["alpha"]
        beta = p["beta"]

        prob = CollocationProblem(
            domain=domain,
            orders=8,
            method="bellman",
            params={"alpha": alpha, "delta": 1.0},
            beta=beta,
            options={"max_iter": 100, "n_howard": 15, "tol": 1e-10},
        )
        sol = prob.solve(backend="numpy")
        assert sol.converged

        eval_k = np.linspace(domain[0], domain[1], 1000)
        v_approx = sol.value(eval_k)
        v_true = p["V_star"](eval_k)

        max_val_err = np.max(np.abs(v_approx - v_true))
        assert max_val_err < 1e-4, f"Collocation value function error {max_val_err:.2e} >= 1e-4"

        g_approx = sol.policy(eval_k)
        g_true = p["g_star"](eval_k)
        max_rel_policy_err = np.max(np.abs(g_approx - g_true) / g_true)
        assert max_rel_policy_err < 1e-4, f"Collocation Bellman policy error {max_rel_policy_err:.2e} >= 1e-4"

    def test_fem_bellman_monotonicity_and_policy_benchmark(self, bm_params):
        """Verify continuous Bellman FEM produces strictly monotonic policy and concave value function."""
        p = bm_params
        prob = FEMProblem(
            domain=p["domain"],
            elements=25,
            method="bellman",
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=p["beta"],
            params={"alpha": p["alpha"], "gamma": 1.0},
            options={"tol": 1e-6, "howard": True, "n_howard": 10},
        )
        sol = prob.solve(backend="numpy")
        assert sol.converged

        nodes = sol.mesh.nodes
        pol = sol.policy(nodes)
        val = sol.value(nodes)

        # Monotonic policy
        assert np.all(np.diff(pol) >= -1e-7)
        assert pol[-1] > pol[0]

        # Strictly increasing and concave value function
        assert np.all(np.diff(val) > 0.0)
        assert np.all(np.diff(val, 2) <= 1e-4)


# ===========================================================================
# 3. Continuous Euler Equation Residuals: max |R(k)| < 10^-4 on >= 1,000 Points
# ===========================================================================

class TestContinuousEulerEquationResiduals:
    """Verify continuous Euler residuals across dense out-of-sample evaluation grids."""

    @pytest.mark.parametrize("n_points", [1000, 1500, 2000])
    def test_collocation_dense_euler_residuals(self, bm_params, n_points: int):
        """Verify Collocation Euler residuals < 10^-4 on dense continuous grids."""
        p = bm_params
        sol = CollocationProblem(
            domain=p["domain"],
            orders=6,
            method="euler",
            params={"alpha": p["alpha"], "delta": 1.0},
            beta=p["beta"],
        ).solve(backend="numpy")

        eval_k = np.linspace(p["domain"][0], p["domain"][1], n_points)
        residuals = compute_continuous_euler_residual(
            sol.policy, eval_k, alpha=p["alpha"], beta=p["beta"]
        )
        max_res = np.max(np.abs(residuals))
        assert max_res < 1e-4, f"Collocation max Euler residual {max_res:.2e} >= 1e-4 on {n_points} points"

    @pytest.mark.parametrize("n_points", [1000, 1500, 2000])
    def test_fem_dense_euler_residuals(self, bm_params, n_points: int):
        """Verify FEM Galerkin Euler residuals < 10^-4 on dense continuous grids."""
        p = bm_params
        sol = FEMProblem(
            domain=p["domain"],
            elements=50,
            method="euler",
            projection="galerkin",
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=p["beta"],
            params={"alpha": p["alpha"], "gamma": 1.0},
        ).solve(backend="numpy")

        eval_k = np.linspace(p["domain"][0], p["domain"][1], n_points)
        residuals = sol.euler_residual(eval_k)
        max_res = np.max(np.abs(residuals))
        assert max_res < 1e-4, f"FEM max Euler residual {max_res:.2e} >= 1e-4 on {n_points} points"

    def test_euler_residuals_out_of_sample_boundary_safety(self, bm_params):
        """Verify solver stability and accurate residuals up to domain boundaries."""
        p = bm_params
        sol = CollocationProblem(
            domain=p["domain"],
            orders=8,
            method="euler",
            params={"alpha": p["alpha"], "delta": 1.0},
            beta=p["beta"],
        ).solve()

        # Evaluate at boundaries
        k_bounds = np.array([p["domain"][0], p["domain"][1]])
        res_bounds = compute_continuous_euler_residual(
            sol.policy, k_bounds, alpha=p["alpha"], beta=p["beta"]
        )
        assert np.all(np.abs(res_bounds) < 1e-4)


# ===========================================================================
# 4. Stochastic Brock-Mirman Model with Productivity Shocks
# ===========================================================================

class TestStochasticBrockMirmanModel:
    """Benchmark stochastic neoclassical growth with discrete TFP states."""

    @pytest.mark.parametrize("z", [0.90, 1.00, 1.10])
    def test_stochastic_tfp_policy_accuracy(self, bm_params, z: float):
        """Verify policy relative error < 10^-4 for each TFP productivity regime."""
        p = bm_params
        alpha = p["alpha"]
        beta = p["beta"]
        k_ss_z = float((alpha * beta * z) ** (1.0 / (1.0 - alpha)))
        domain_z = (0.5 * k_ss_z, 1.5 * k_ss_z)

        # 1. Collocation with explicit TFP residual function
        def user_euler_z(policy_fn, k, params):
            kp = policy_fn(k)
            kpp = policy_fn(kp)
            c = np.maximum(z * (k**alpha) - kp, 1e-12)
            cp = np.maximum(z * (kp**alpha) - kpp, 1e-12)
            return 1.0 - beta * (c / cp) * alpha * z * (kp ** (alpha - 1.0))

        prob_coll = CollocationProblem(
            domain=domain_z,
            orders=6,
            method="euler",
            euler_residual_fn=user_euler_z,
            params={"alpha": alpha, "z": z},
            beta=beta,
        )
        sol_coll = prob_coll.solve()

        eval_k = np.linspace(domain_z[0], domain_z[1], 1000)
        g_true_z = p["g_star"](eval_k, z=z)
        rel_err_coll = np.max(np.abs(sol_coll.policy(eval_k) - g_true_z) / g_true_z)
        assert rel_err_coll < 1e-4, f"Collocation rel error {rel_err_coll:.2e} >= 1e-4 for z={z}"

        # 2. FEM with TFP transition function
        prob_fem = FEMProblem(
            domain=domain_z,
            elements=50,
            method="euler",
            projection="galerkin",
            return_fn=p["return_fn"],
            transition_fn=lambda k: z * (k**alpha),
            beta=beta,
            params={"alpha": alpha, "gamma": 1.0, "z": z},
        )
        sol_fem = prob_fem.solve()
        rel_err_fem = np.max(np.abs(sol_fem.policy(eval_k) - g_true_z) / g_true_z)
        assert rel_err_fem < 1e-4, f"FEM rel error {rel_err_fem:.2e} >= 1e-4 for z={z}"

    def test_stochastic_productivity_monotonic_ordering(self, bm_params):
        """Verify capital accumulation is strictly increasing in productivity shock z."""
        p = bm_params
        alpha = p["alpha"]
        beta = p["beta"]
        z_low, z_high = 0.95, 1.05

        def make_problem(z_val):
            k_ss_z = float((alpha * beta * z_val) ** (1.0 / (1.0 - alpha)))
            domain_z = (0.5 * k_ss_z, 1.5 * k_ss_z)
            return FEMProblem(
                domain=domain_z,
                elements=40,
                method="euler",
                projection="galerkin",
                return_fn=p["return_fn"],
                transition_fn=lambda k: z_val * (k**alpha),
                beta=beta,
                params={"alpha": alpha, "gamma": 1.0},
            )

        sol_low = make_problem(z_low).solve()
        sol_high = make_problem(z_high).solve()

        common_grid = np.linspace(p["domain"][0], p["domain"][1], 500)
        g_low = sol_low.policy(common_grid)
        g_high = sol_high.policy(common_grid)

        assert np.all(g_high > g_low), "Policy must be strictly higher in high-TFP regime"


# ===========================================================================
# 5. Multi-Backend Parity & Consistency (< 10^-4)
# ===========================================================================

class TestMultiBackendParity:
    """Verify solution parity across installed compute backends (NumPy, Numba, MLX)."""

    def test_collocation_multi_backend_parity(self, bm_params):
        """Verify Collocation gives identical solutions within 10^-4 across all available backends."""
        p = bm_params
        prob = CollocationProblem(
            domain=p["domain"],
            orders=6,
            method="euler",
            params={"alpha": p["alpha"], "delta": 1.0},
            beta=p["beta"],
        )

        sol_numpy = prob.solve(backend="numpy")
        eval_k = np.linspace(p["domain"][0], p["domain"][1], 1000)
        pol_ref = sol_numpy.policy(eval_k)

        for backend in bk.available_backends():
            if backend == "numpy":
                continue
            sol_b = prob.solve(backend=backend)
            assert sol_b.converged
            pol_b = sol_b.policy(eval_k)
            max_rel_diff = np.max(np.abs(pol_b - pol_ref) / pol_ref)
            assert (
                max_rel_diff < 1e-4
            ), f"Collocation backend {backend} relative difference {max_rel_diff:.2e} >= 1e-4"

    def test_fem_multi_backend_parity(self, bm_params):
        """Verify FEM gives identical solutions within 10^-4 across all available backends."""
        p = bm_params
        prob = FEMProblem(
            domain=p["domain"],
            elements=50,
            method="euler",
            projection="galerkin",
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=p["beta"],
            params={"alpha": p["alpha"], "gamma": 1.0},
        )

        sol_numpy = prob.solve(backend="numpy")
        eval_k = np.linspace(p["domain"][0], p["domain"][1], 1000)
        pol_ref = sol_numpy.policy(eval_k)

        for backend in bk.available_backends():
            if backend == "numpy":
                continue
            sol_b = prob.solve(backend=backend)
            assert sol_b.converged
            pol_b = sol_b.policy(eval_k)
            max_rel_diff = np.max(np.abs(pol_b - pol_ref) / pol_ref)
            assert (
                max_rel_diff < 1e-4
            ), f"FEM backend {backend} relative difference {max_rel_diff:.2e} >= 1e-4"

    def test_unavailable_backend_graceful_handling(self, bm_params):
        """Verify requesting an uninstalled backend raises informative exception rather than crashing."""
        p = bm_params
        prob = CollocationProblem(
            domain=p["domain"],
            orders=5,
            params={"alpha": p["alpha"]},
        )
        with pytest.raises(ValueError, match="Unknown backend"):
            prob.solve(backend="nonexistent_backend_name")


# ===========================================================================
# 6. Presentation Contract Compliance
# ===========================================================================

class TestPresentationContract:
    """Verify that CollocationSolution and FEMSolution satisfy the puremacro presentation contract."""

    def test_collocation_presentation_contract(self, bm_params):
        """Verify CollocationSolution implements .summary, .plot, .to_frame, .to_markdown, .to_latex, .to_typst."""
        p = bm_params
        sol = solve_collocation(
            domain=p["domain"],
            orders=6,
            method="euler",
            params={"alpha": p["alpha"], "delta": 1.0},
            beta=p["beta"],
        )

        # 1. .summary()
        summary_df = sol.summary()
        assert isinstance(summary_df, pd.DataFrame)
        assert not summary_df.empty
        assert "Method" in summary_df.index
        assert "Converged" in summary_df.index

        # 2. .to_frame()
        frame = sol.to_frame()
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) == len(summary_df)

        # 3. .to_markdown()
        md = sol.to_markdown()
        assert isinstance(md, str)
        assert "Method" in md
        assert "|" in md

        # 4. .to_latex()
        latex = sol.to_latex()
        assert isinstance(latex, str)
        assert "begin{tabular}" in latex

        # 5. .to_typst()
        typst = sol.to_typst()
        assert isinstance(typst, str)
        assert "#table" in typst

        # 6. .plot()
        fig = sol.plot(show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_fem_presentation_contract(self, bm_params):
        """Verify FEMSolution implements .summary, .plot, .to_frame, .to_markdown, .to_latex, .to_typst."""
        p = bm_params
        sol = solve_fem(
            p["domain"],
            elements=30,
            method="euler",
            projection="galerkin",
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=p["beta"],
            params={"alpha": p["alpha"], "gamma": 1.0},
        )

        # 1. .summary()
        summary_df = sol.summary()
        assert isinstance(summary_df, pd.DataFrame)
        assert not summary_df.empty
        assert "Method" in summary_df.index
        assert "Converged" in summary_df.index

        # 2. .to_frame()
        frame = sol.to_frame()
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) == len(summary_df)

        # 3. .to_markdown()
        md = sol.to_markdown()
        assert isinstance(md, str)
        assert "Method" in md
        assert "|" in md

        # 4. .to_latex()
        latex = sol.to_latex()
        assert isinstance(latex, str)
        assert "begin{tabular}" in latex

        # 5. .to_typst()
        typst = sol.to_typst()
        assert isinstance(typst, str)
        assert "#table" in typst

        # 6. .plot()
        fig = sol.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
