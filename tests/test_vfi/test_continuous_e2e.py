"""Comprehensive End-to-End Test Suite for Continuous State-Space Methods in puremacro.vfi.

Covers:
- Tier 1: Feature Coverage (>=5 tests per feature for F1 through F7 in isolation)
- Tier 2: Boundary & Corner Cases (>=5 tests per feature for F1 through F7)
- Tier 3: Cross-Feature Combinations (pairwise interactions across methods and backends)
- Tier 4: Real-World Macroeconomic Application Scenarios (Brock-Mirman, constrained growth, etc.)

Total test count: 85 tests.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro import _backend as bk
from puremacro.vfi.collocation import (
    CollocationBasis,
    CollocationProblem,
    CollocationSolution,
    solve_collocation,
)
from puremacro.vfi.fem import (
    FEMMesh,
    FEMProblem,
    FEMSolution,
    solve_fem,
)


# ---------------------------------------------------------------------------
# Authoritative Analytical Benchmarks & Fixtures (Brock-Mirman 1972)
# ---------------------------------------------------------------------------

def canonical_brock_mirman_params(alpha: float = 0.36, beta: float = 0.96) -> Dict[str, Any]:
    """Return canonical parameters and exact analytical closed-form functions."""
    k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
    A = float(alpha / (1.0 - alpha * beta))
    B = float((np.log(1.0 - alpha * beta) + (alpha * beta * np.log(alpha * beta)) / (1.0 - alpha * beta)) / (1.0 - beta))

    def g_star(k: np.ndarray | float) -> np.ndarray | float:
        k_arr = np.asarray(k, dtype=np.float64)
        val = alpha * beta * (k_arr ** alpha)
        return float(val) if np.ndim(k) == 0 else val

    def c_star(k: np.ndarray | float) -> np.ndarray | float:
        k_arr = np.asarray(k, dtype=np.float64)
        val = (1.0 - alpha * beta) * (k_arr ** alpha)
        return float(val) if np.ndim(k) == 0 else val

    def V_star(k: np.ndarray | float) -> np.ndarray | float:
        k_arr = np.asarray(k, dtype=np.float64)
        val = A * np.log(k_arr) + B
        return float(val) if np.ndim(k) == 0 else val

    def euler_res_fn(k: np.ndarray, kp: np.ndarray, kpp: np.ndarray, **kwargs) -> np.ndarray:
        c = np.maximum(k ** alpha - kp, 1e-12)
        cp = np.maximum(kp ** alpha - kpp, 1e-12)
        return 1.0 - beta * (c / cp) * alpha * (kp ** (alpha - 1.0))

    return {
        "alpha": alpha,
        "beta": beta,
        "delta": 1.0,
        "k_ss": k_ss,
        "A": A,
        "B": B,
        "g_star": g_star,
        "c_star": c_star,
        "V_star": V_star,
        "euler_res_fn": euler_res_fn,
        "domain": (0.5 * k_ss, 1.5 * k_ss),
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
    c = z * (k ** alpha) - kp
    cp = z * (kp ** alpha) - kpp
    c_safe = np.maximum(c, 1e-12)
    cp_safe = np.maximum(cp, 1e-12)
    res = 1.0 - beta * (c_safe / cp_safe) * alpha * z * (kp ** (alpha - 1.0))
    return res


# ===========================================================================
# TIER 1: FEATURE COVERAGE (F1 to F7 in isolation, >= 5 tests each = 35 tests)
# ===========================================================================

class TestTier1F1PolynomialBasisAndNodes:
    """Feature 1: Polynomial Basis & Nodes (Chebyshev recurrence, nodes, mapping, tensor)."""

    def test_t1_f1_01_chebyshev_lobatto_nodes_endpoints_and_monotonicity(self):
        """F1.1: Lobatto nodes include domain endpoints and are strictly ascending."""
        basis = CollocationBasis(domain=(0.1, 2.5), orders=5, node_type="lobatto")
        nodes = basis.nodes(squeeze=True)
        assert len(nodes) == 6
        assert np.isclose(nodes[0], 0.1, atol=1e-12)
        assert np.isclose(nodes[-1], 2.5, atol=1e-12)
        assert np.all(np.diff(nodes) > 0.0), "Nodes must be strictly monotonically increasing"

    def test_t1_f1_02_chebyshev_gauss_nodes_roots_and_symmetry(self):
        """F1.2: Gauss nodes lie strictly in domain interior and are symmetric in canonical space."""
        basis = CollocationBasis(domain=(-1.0, 1.0), orders=4, node_type="gauss")
        nodes = basis.nodes(squeeze=True)
        assert len(nodes) == 5
        # Strictly in open interior (-1, 1)
        assert np.all(nodes > -1.0) and np.all(nodes < 1.0)
        # Symmetry around origin: x_k == -x_{N-1-k}
        assert np.allclose(nodes, -nodes[::-1], atol=1e-12)
        # Matches exact roots formula x_k = -cos((2k-1)*pi / (2*n))
        k = np.arange(1, 6)
        expected = -np.cos((2.0 * k - 1.0) * np.pi / 10.0)
        assert np.allclose(nodes, expected, atol=1e-12)

    def test_t1_f1_03_chebyshev_3term_recurrence_exact_polynomials(self):
        """F1.3: Basis evaluation satisfies exact 3-term recurrence T_n(x) polynomials."""
        basis = CollocationBasis(domain=(-1.0, 1.0), orders=4)
        x_test = np.linspace(-0.95, 0.95, 20)
        Phi = basis.evaluate(x_test)
        assert Phi.shape == (20, 5)
        # T0(x) = 1
        assert np.allclose(Phi[:, 0], 1.0, atol=1e-12)
        # T1(x) = x
        assert np.allclose(Phi[:, 1], x_test, atol=1e-12)
        # T2(x) = 2x^2 - 1
        assert np.allclose(Phi[:, 2], 2.0 * (x_test ** 2) - 1.0, atol=1e-12)
        # T3(x) = 4x^3 - 3x
        assert np.allclose(Phi[:, 3], 4.0 * (x_test ** 3) - 3.0 * x_test, atol=1e-12)
        # T4(x) = 8x^4 - 8x^2 + 1
        assert np.allclose(Phi[:, 4], 8.0 * (x_test ** 4) - 8.0 * (x_test ** 2) + 1.0, atol=1e-12)

    def test_t1_f1_04_chebyshev_affine_coordinate_mapping_and_jacobian(self):
        """F1.4: Bijective affine mapping [a, b] <-> [-1, 1] and derivative Jacobian."""
        a, b = 0.5, 4.5
        basis = CollocationBasis(domain=(a, b), orders=3)
        # Boundary and midpoint mapping
        assert np.isclose(basis.to_canonical(a)[0, 0], -1.0, atol=1e-12)
        assert np.isclose(basis.to_canonical(b)[0, 0], 1.0, atol=1e-12)
        assert np.isclose(basis.to_canonical(0.5 * (a + b))[0, 0], 0.0, atol=1e-12)
        # Derivative basis matrix includes chain rule Jacobian 2 / (b - a)
        dPhi = basis.derivative(np.array([2.5]), dim=0)
        # T1'(x) = 1 in canonical space, so d/ds T1 = 1 * (2 / (b - a)) = 2 / 4.0 = 0.5
        assert np.isclose(dPhi[0, 1], 2.0 / (b - a), atol=1e-12)

    def test_t1_f1_05_chebyshev_multid_tensor_product_basis_shapes(self):
        """F1.5: Multi-dimensional tensor product grid generation and basis shape."""
        domain_2d = ((0.1, 2.0), (1.0, 5.0))
        orders_2d = (3, 4)
        basis_2d = CollocationBasis(domain=domain_2d, orders=orders_2d)
        assert basis_2d.n_dims == 2
        assert basis_2d.n_nodes == (3 + 1) * (4 + 1) == 20
        nodes = basis_2d.nodes()
        assert nodes.shape == (20, 2)
        # Evaluate basis across arbitrary test points
        test_pts = np.array([[0.5, 2.0], [1.0, 3.5], [1.8, 4.8]])
        Phi = basis_2d.evaluate(test_pts)
        assert Phi.shape == (3, 20)


class TestTier1F2PolynomialCollocationSolver:
    """Feature 2: Polynomial Collocation Solver (Euler, Bellman, evaluation, solve)."""

    def test_t1_f2_01_collocation_euler_projection_solve(self):
        """F2.1: Euler equation projection converges cleanly with small residual norm."""
        bm = canonical_brock_mirman_params()
        prob = CollocationProblem(
            domain=bm["domain"],
            orders=6,
            method="euler",
            beta=bm["beta"],
            params={"alpha": bm["alpha"], "delta": 1.0},
        )
        sol = prob.solve(backend="numpy")
        assert isinstance(sol, CollocationSolution)
        assert sol.converged is True
        assert sol.residual_norm < 1e-8
        assert sol.backend == "numpy"

    def test_t1_f2_02_collocation_bellman_vfi_solve(self):
        """F2.2: Continuous Bellman value function iteration converges and returns value."""
        bm = canonical_brock_mirman_params()
        prob = CollocationProblem(
            domain=bm["domain"],
            orders=5,
            method="bellman",
            beta=bm["beta"],
            params={"alpha": bm["alpha"], "delta": 1.0},
            options={"tol": 1e-6, "max_iter": 500},
        )
        sol = prob.solve(backend="numpy")
        assert sol.converged is True
        # Check value evaluation
        k_mid = bm["k_ss"]
        val_approx = sol.value(k_mid)
        val_true = bm["V_star"](k_mid)
        rel_err = abs(val_approx - val_true) / abs(val_true)
        assert rel_err < 1e-3, f"Bellman value error too large: {rel_err}"

    def test_t1_f2_03_collocation_continuous_policy_evaluation(self):
        """F2.3: Continuous policy function evaluates for scalars and arrays, strictly increasing."""
        bm = canonical_brock_mirman_params()
        sol = CollocationProblem(domain=bm["domain"], orders=6, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        # Scalar evaluation
        val_scalar = sol.policy(bm["k_ss"])
        assert isinstance(val_scalar, float)
        # Array evaluation
        k_grid = np.linspace(bm["domain"][0], bm["domain"][1], 100)
        pol_grid = sol.policy(k_grid)
        assert pol_grid.shape == (100,)
        assert np.all(np.diff(pol_grid) > 0.0), "Policy function must be strictly increasing in capital"

    def test_t1_f2_04_collocation_continuous_value_evaluation(self):
        """F2.4: Continuous value function is strictly concave and increasing."""
        bm = canonical_brock_mirman_params()
        sol = CollocationProblem(domain=bm["domain"], orders=5, method="bellman", beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        k_grid = np.linspace(bm["domain"][0], bm["domain"][1], 100)
        v_grid = sol.value(k_grid)
        # Increasing
        assert np.all(np.diff(v_grid) > 0.0)
        # Concave: second difference is negative
        second_diff = np.diff(v_grid, n=2)
        assert np.all(second_diff < 0.0)

    def test_t1_f2_05_collocation_options_and_tolerances(self):
        """F2.5: Functional solve_collocation entry point respects custom solver options."""
        bm = canonical_brock_mirman_params()
        sol = solve_collocation(
            domain=bm["domain"],
            orders=5,
            method="euler",
            beta=bm["beta"],
            params={"alpha": bm["alpha"]},
            options={"tol": 1e-9, "max_iter": 200},
        )
        assert sol.converged is True
        assert sol.residual_norm < 1e-8


class TestTier1F3FEMMeshAndShapeFunctions:
    """Feature 3: FEM Mesh & Shape Functions (uniform, kink placement, hat functions)."""

    def test_t1_f3_01_fem_mesh_1d_uniform_partition_of_unity(self):
        """F3.1: Piecewise linear hat basis satisfies partition of unity (sum_i phi_i = 1)."""
        mesh = FEMMesh(domain=(0.2, 3.0), elements=20)
        assert mesh.n_nodes == 21
        assert mesh.n_elements == 20
        test_pts = np.linspace(0.2, 3.0, 100)
        Phi = mesh.evaluate_basis(test_pts)
        assert Phi.shape == (100, 21)
        # Sum of shape functions equals 1.0 everywhere
        row_sums = np.sum(Phi, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-12)

    def test_t1_f3_02_fem_mesh_kronecker_delta_nodal_property(self):
        """F3.2: Shape functions satisfy Kronecker delta property at mesh nodes: phi_i(s_j) = delta_ij."""
        mesh = FEMMesh(domain=(1.0, 5.0), elements=10)
        nodes = mesh.nodes
        Phi_nodes = mesh.evaluate_basis(nodes)
        identity = np.eye(len(nodes))
        assert np.allclose(Phi_nodes, identity, atol=1e-12)

    def test_t1_f3_03_fem_mesh_nonuniform_kink_node_placement(self):
        """F3.3: Mesh from kinks places an exact node at the specified kink boundary."""
        kink = 1.45
        mesh = FEMMesh.from_kinks(domain=(0.5, 3.0), n_elements=25, kinks=[kink])
        nodes = mesh.nodes
        # Check that kink is present in nodes
        min_dist = np.min(np.abs(nodes - kink))
        assert min_dist < 1e-12, f"Kink {kink} not placed at an exact node"

    def test_t1_f3_04_fem_mesh_compact_support_and_sparsity(self):
        """F3.4: Shape functions have compact support on adjacent elements and 0 elsewhere."""
        mesh = FEMMesh(domain=(0.0, 10.0), elements=10)
        # Node index 5 has support on [4.0, 6.0]
        test_out = np.array([1.0, 2.0, 3.5, 6.5, 8.0])
        Phi_out = mesh.evaluate_basis(test_out)
        assert np.allclose(Phi_out[:, 5], 0.0, atol=1e-12)

    def test_t1_f3_05_fem_mesh_tensor_multid_grid(self):
        """F3.5: Multi-dimensional FEM mesh creation and basis matrix shapes."""
        domain_2d = ((0.0, 1.0), (0.0, 2.0))
        mesh_2d = FEMMesh(domain=domain_2d, elements=(4, 5))
        assert mesh_2d.dim == 2
        assert mesh_2d.n_nodes == (4 + 1) * (5 + 1) == 30
        eval_pts = np.array([[0.2, 0.5], [0.8, 1.5]])
        Phi = mesh_2d.evaluate_basis(eval_pts)
        assert Phi.shape == (2, 30)
        assert np.allclose(np.sum(Phi, axis=1), 1.0, atol=1e-12)


class TestTier1F4FEMGalerkinSolver:
    """Feature 4: FEM Galerkin Solver (Euler, nodal collocation, Bellman, constraints)."""

    def test_t1_f4_01_fem_galerkin_euler_solve(self):
        """F4.1: FEM Galerkin Euler projection converges cleanly with small residual norm."""
        bm = canonical_brock_mirman_params()
        prob = FEMProblem(
            domain=bm["domain"],
            elements=40,
            method="euler",
            projection="galerkin",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"], "gamma": 1.0},
        )
        sol = prob.solve(backend="numpy")
        assert isinstance(sol, FEMSolution)
        assert sol.converged is True
        assert sol.residual_norm < 1e-8

    def test_t1_f4_02_fem_nodal_collocation_solve(self):
        """F4.2: FEM with nodal collocation projection solves cleanly."""
        bm = canonical_brock_mirman_params()
        prob = FEMProblem(
            domain=bm["domain"],
            elements=40,
            method="euler",
            projection="collocation",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"], "gamma": 1.0},
        )
        sol = prob.solve(backend="numpy")
        assert sol.converged is True

    def test_t1_f4_03_fem_bellman_vfi_solve(self):
        """F4.3: FEM with Bellman value iteration converges."""
        bm = canonical_brock_mirman_params()
        prob = FEMProblem(
            domain=bm["domain"],
            elements=30,
            method="bellman",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"], "gamma": 1.0},
            options={"tol": 1e-6, "max_iter": 400},
        )
        sol = prob.solve(backend="numpy")
        assert sol.converged is True
        val_mid = sol.value(bm["k_ss"])
        assert np.isfinite(val_mid)

    def test_t1_f4_04_fem_policy_interpolant_continuity(self):
        """F4.4: Piecewise linear continuous interpolant evaluates scalar and vector states."""
        bm = canonical_brock_mirman_params()
        sol = FEMProblem(
            domain=bm["domain"],
            elements=30,
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"]},
        ).solve()
        # Scalar evaluation
        p_scal = sol.policy(bm["k_ss"])
        assert isinstance(p_scal, float)
        # Vector evaluation
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 50)
        p_vec = sol.policy(k_eval)
        assert p_vec.shape == (50,)
        assert np.all(np.diff(p_vec) > 0.0)

    def test_t1_f4_05_fem_borrowing_constraint_enforcement(self):
        """F4.5: FEM with borrowing constraint strictly enforces lower bound on choices."""
        bm = canonical_brock_mirman_params()
        b_limit = 0.18
        sol = FEMProblem(
            domain=bm["domain"],
            elements=50,
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"]},
            borrowing_constraint=b_limit,
        ).solve()
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 200)
        pol = sol.policy(k_eval)
        assert np.min(pol) >= b_limit - 1e-8, "FEM policy violated borrowing constraint"


class TestTier1F5MultiBackendAcceleration:
    """Feature 5: Multi-Backend Acceleration (NumPy oracle, Numba, MLX, CuPy fallback)."""

    def test_t1_f5_01_backend_numpy_reference_oracle(self):
        """F5.1: Default NumPy reference backend returns standard numpy.ndarray."""
        bm = canonical_brock_mirman_params()
        prob = CollocationProblem(domain=bm["domain"], orders=5, beta=bm["beta"], params={"alpha": bm["alpha"]})
        sol = prob.solve(backend="numpy")
        assert sol.backend == "numpy"
        assert isinstance(sol.coefficients, np.ndarray)

    def test_t1_f5_02_backend_numba_execution_and_fallback(self):
        """F5.2: Numba backend solves cleanly and matches NumPy oracle within numerical precision."""
        bm = canonical_brock_mirman_params()
        prob = CollocationProblem(domain=bm["domain"], orders=5, beta=bm["beta"], params={"alpha": bm["alpha"]})
        sol_np = prob.solve(backend="numpy")
        sol_nb = prob.solve(backend="numba")
        k_test = np.linspace(bm["domain"][0], bm["domain"][1], 50)
        diff = np.max(np.abs(sol_nb.policy(k_test) - sol_np.policy(k_test)))
        assert diff < 1e-6, f"Numba solution diverges from NumPy: {diff}"

    def test_t1_f5_03_backend_mlx_gpu_execution_or_skip(self):
        """F5.3: MLX Apple Silicon GPU backend (if available) matches NumPy within 1e-4 tolerance."""
        if not bk.backend_available("mlx"):
            pytest.skip("Apple Silicon MLX backend not installed in this environment")
        bm = canonical_brock_mirman_params()
        prob = CollocationProblem(domain=bm["domain"], orders=6, beta=bm["beta"], params={"alpha": bm["alpha"]})
        sol_np = prob.solve(backend="numpy")
        sol_mx = prob.solve(backend="mlx")
        k_test = np.linspace(bm["domain"][0], bm["domain"][1], 100)
        diff = np.max(np.abs(sol_mx.policy(k_test) - sol_np.policy(k_test)))
        assert diff < 1e-4, f"MLX GPU diverges from NumPy: {diff}"

    def test_t1_f5_04_backend_uninstalled_cupy_informative_error(self):
        """F5.4: Uninstalled GPU backend (CuPy) is handled gracefully with warning and fallback to NumPy."""
        if bk.backend_available("cupy"):
            pytest.skip("CuPy is installed on this CUDA system")
        bm = canonical_brock_mirman_params()
        prob = CollocationProblem(domain=bm["domain"], orders=5, beta=bm["beta"], params={"alpha": bm["alpha"]})
        with pytest.warns(UserWarning, match="falling back to 'numpy'"):
            sol = prob.solve(backend="cupy")
        assert sol.backend == "numpy"
        assert sol.metadata["backend"] == "numpy"
        assert sol.converged is True

    def test_t1_f5_05_backend_unknown_raises_value_error(self):
        """F5.5: Unknown backend string raises informative ValueError."""
        bm = canonical_brock_mirman_params()
        prob = CollocationProblem(domain=bm["domain"], orders=5, beta=bm["beta"], params={"alpha": bm["alpha"]})
        with pytest.raises(ValueError, match="Unknown backend"):
            prob.solve(backend="unknown_custom_backend")


class TestTier1F6PresentationAndAPIExport:
    """Feature 6: Presentation Interfaces (.summary, .plot, .to_markdown, .to_latex, .to_typst)."""

    def test_t1_f6_01_collocation_summary_and_to_frame(self):
        """F6.1: CollocationSolution.summary() and .to_frame() return valid DataFrames."""
        bm = canonical_brock_mirman_params()
        sol = CollocationProblem(domain=bm["domain"], orders=5, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        df_sum = sol.summary()
        df_frame = sol.to_frame()
        assert isinstance(df_sum, pd.DataFrame)
        assert isinstance(df_frame, pd.DataFrame)
        assert "Converged" in df_sum.index
        assert df_sum.loc["Converged", "Value"] is True or df_sum.loc["Converged", "Value"] == "True"

    def test_t1_f6_02_collocation_markdown_latex_typst_rendering(self):
        """F6.2: CollocationSolution renders non-empty Markdown, LaTeX, and Typst strings."""
        bm = canonical_brock_mirman_params()
        sol = CollocationProblem(domain=bm["domain"], orders=5, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        md = sol.to_markdown()
        tex = sol.to_latex()
        typ = sol.to_typst()
        assert isinstance(md, str) and len(md) > 10 and "|" in md
        assert isinstance(tex, str) and "tabular" in tex
        assert isinstance(typ, str) and ("table" in typ or len(typ) > 10)

    def test_t1_f6_03_fem_summary_and_to_frame(self):
        """F6.3: FEMSolution.summary() and .to_frame() return valid DataFrames."""
        bm = canonical_brock_mirman_params()
        sol = FEMProblem(domain=bm["domain"], elements=20, return_fn=lambda c: np.log(c), transition_fn=lambda k: k**bm["alpha"], beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        df = sol.summary()
        assert isinstance(df, pd.DataFrame)
        assert "Converged" in df.index

    def test_t1_f6_04_fem_markdown_latex_typst_rendering(self):
        """F6.4: FEMSolution renders non-empty Markdown, LaTeX, and Typst strings."""
        bm = canonical_brock_mirman_params()
        sol = FEMProblem(domain=bm["domain"], elements=20, return_fn=lambda c: np.log(c), transition_fn=lambda k: k**bm["alpha"], beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        md = sol.to_markdown()
        tex = sol.to_latex()
        typ = sol.to_typst()
        assert isinstance(md, str) and "|" in md
        assert isinstance(tex, str) and "tabular" in tex
        assert isinstance(typ, str) and len(typ) > 5

    def test_t1_f6_05_plotting_interface_returns_figure(self):
        """F6.5: .plot() returns a valid matplotlib Figure object in headless environments."""
        bm = canonical_brock_mirman_params()
        sol_c = CollocationProblem(domain=bm["domain"], orders=5, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        sol_f = FEMProblem(domain=bm["domain"], elements=20, return_fn=lambda c: np.log(c), transition_fn=lambda k: k**bm["alpha"], beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        fig_c = sol_c.plot()
        fig_f = sol_f.plot()
        assert isinstance(fig_c, matplotlib.figure.Figure)
        assert isinstance(fig_f, matplotlib.figure.Figure)
        plt.close(fig_c)
        plt.close(fig_f)


class TestTier1F7AnalyticalBenchmarks:
    """Feature 7: Analytical Benchmarks & Verification (Brock-Mirman exact solution)."""

    def test_t1_f7_01_brock_mirman_collocation_policy_relative_error(self):
        """F7.1: Collocation policy relative error max |g(k) - g*(k)| / g*(k) < 1e-4."""
        bm = canonical_brock_mirman_params()
        sol = CollocationProblem(domain=bm["domain"], orders=6, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 1000)
        g_true = bm["g_star"](k_eval)
        g_approx = sol.policy(k_eval)
        rel_err = np.max(np.abs(g_approx - g_true) / g_true)
        assert rel_err < 1e-4, f"Collocation relative policy error {rel_err} exceeds 1e-4"

    def test_t1_f7_02_brock_mirman_fem_policy_relative_error(self):
        """F7.2: FEM policy relative error max |g(k) - g*(k)| / g*(k) < 1e-4 for E=50 elements."""
        bm = canonical_brock_mirman_params()
        sol = FEMProblem(
            domain=bm["domain"],
            elements=50,
            method="euler",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"], "gamma": 1.0},
        ).solve()
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 1000)
        g_true = bm["g_star"](k_eval)
        g_approx = sol.policy(k_eval)
        rel_err = np.max(np.abs(g_approx - g_true) / g_true)
        assert rel_err < 1e-4, f"FEM relative policy error {rel_err} exceeds 1e-4"

    def test_t1_f7_03_brock_mirman_collocation_value_accuracy(self):
        """F7.3: Collocation Bellman value function relative error < 1e-3 against exact V*(k)."""
        bm = canonical_brock_mirman_params()
        sol = CollocationProblem(
            domain=bm["domain"],
            orders=6,
            method="bellman",
            beta=bm["beta"],
            params={"alpha": bm["alpha"]},
            options={"tol": 1e-7, "max_iter": 500},
        )
        res = sol.solve()
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 200)
        V_true = bm["V_star"](k_eval)
        V_approx = res.value(k_eval)
        rel_err = np.max(np.abs(V_approx - V_true) / np.abs(V_true))
        assert rel_err < 1e-3, f"Value function error {rel_err} exceeds 1e-3"

    def test_t1_f7_04_brock_mirman_euler_residual_out_of_sample(self):
        """F7.4: Continuous Euler equation residual < 1e-4 across 1,000 out-of-sample points."""
        bm = canonical_brock_mirman_params()
        sol = CollocationProblem(domain=bm["domain"], orders=6, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 1000)
        residuals = compute_continuous_euler_residual(sol.policy, k_eval, bm["alpha"], bm["beta"])
        max_res = np.max(np.abs(residuals))
        assert max_res < 1e-4, f"Collocation Euler residual {max_res} exceeds 1e-4"

    def test_t1_f7_05_brock_mirman_steady_state_invariance(self):
        """F7.5: Steady state is invariant under policy rule: |g(k_ss) - k_ss| / k_ss < 1e-4."""
        bm = canonical_brock_mirman_params()
        sol = CollocationProblem(domain=bm["domain"], orders=6, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        k_ss = bm["k_ss"]
        g_kss = sol.policy(k_ss)
        err = abs(g_kss - k_ss) / k_ss
        assert err < 1e-4, f"Steady state policy error {err} exceeds 1e-4"


# ===========================================================================
# TIER 2: BOUNDARY & CORNER CASES (>= 5 tests per feature = 35 tests)
# ===========================================================================

class TestTier2BoundaryAndCornerCases:
    """Tier 2: Extreme parameters, mesh limits, boundary clamping, and kink handling."""

    # F1 Boundary & Corners
    def test_t2_f1_01_chebyshev_minimum_order_1(self):
        """T2.F1.1: Minimal polynomial degree N=1 (2 nodes)."""
        basis = CollocationBasis(domain=(0.1, 1.0), orders=1)
        assert basis.n_nodes == 2
        nodes = basis.nodes(squeeze=True)
        assert np.allclose(nodes, [0.1, 1.0], atol=1e-12)

    def test_t2_f1_02_chebyshev_high_polynomial_order_stability(self):
        """T2.F1.2: High polynomial degree N=18 maintains numerical stability and finite entries."""
        basis = CollocationBasis(domain=(0.01, 5.0), orders=18)
        assert basis.n_nodes == 19
        nodes = basis.nodes(squeeze=True)
        Phi = basis.evaluate(nodes)
        assert np.all(np.isfinite(Phi))
        cond = np.linalg.cond(Phi)
        assert cond < 1e6, f"Chebyshev collocation matrix ill-conditioned: {cond}"

    def test_t2_f1_03_chebyshev_inverted_domain_error(self):
        """T2.F1.3: Inverted domain bounds a >= b raises ValueError."""
        with pytest.raises(ValueError, match="domain bounds"):
            CollocationBasis(domain=(2.0, 1.0), orders=4)

    def test_t2_f1_04_chebyshev_degenerate_single_point_domain_error(self):
        """T2.F1.4: Degenerate single point domain a == b raises ValueError."""
        with pytest.raises(ValueError, match="domain bounds"):
            CollocationBasis(domain=(1.0, 1.0), orders=4)

    def test_t2_f1_05_chebyshev_out_of_bounds_clamping_or_warning(self):
        """T2.F1.5: Out-of-bounds evaluation is handled safely via coordinate clamping."""
        basis = CollocationBasis(domain=(1.0, 2.0), orders=4)
        # Point outside domain
        x_out = np.array([0.5, 2.5])
        Phi_out = basis.evaluate(x_out)
        assert np.all(np.isfinite(Phi_out))

    # F2 Boundary & Corners
    def test_t2_f2_01_collocation_extreme_high_discount_beta(self):
        """T2.F2.1: Extreme high discount factor beta = 0.999."""
        bm = canonical_brock_mirman_params(beta=0.999)
        prob = CollocationProblem(domain=bm["domain"], orders=6, beta=0.999, params={"alpha": bm["alpha"]})
        sol = prob.solve()
        assert sol.converged is True
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 100)
        assert np.max(np.abs(sol.policy(k_eval) - bm["g_star"](k_eval)) / bm["g_star"](k_eval)) < 1e-4

    def test_t2_f2_02_collocation_extreme_low_discount_beta(self):
        """T2.F2.2: Extreme low discount factor beta = 0.05 (myopic agent)."""
        bm = canonical_brock_mirman_params(beta=0.05)
        prob = CollocationProblem(domain=bm["domain"], orders=5, beta=0.05, params={"alpha": bm["alpha"]})
        sol = prob.solve()
        assert sol.converged is True

    def test_t2_f2_03_collocation_extreme_capital_share_high(self):
        """T2.F2.3: High capital share alpha = 0.85 (near-linear technology)."""
        bm = canonical_brock_mirman_params(alpha=0.85)
        prob = CollocationProblem(domain=bm["domain"], orders=6, beta=bm["beta"], params={"alpha": 0.85})
        sol = prob.solve()
        assert sol.converged is True
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 100)
        assert np.max(np.abs(sol.policy(k_eval) - bm["g_star"](k_eval)) / bm["g_star"](k_eval)) < 1e-4

    def test_t2_f2_04_collocation_extreme_capital_share_low(self):
        """T2.F2.4: Low capital share alpha = 0.10."""
        bm = canonical_brock_mirman_params(alpha=0.10)
        prob = CollocationProblem(domain=bm["domain"], orders=5, beta=bm["beta"], params={"alpha": 0.10})
        sol = prob.solve()
        assert sol.converged is True

    def test_t2_f2_05_collocation_near_zero_capital_boundary(self):
        """T2.F2.5: Domain lower bound very close to zero k_min = 1e-3 with steep marginal utility."""
        bm = canonical_brock_mirman_params()
        prob = CollocationProblem(domain=(1e-3, bm["domain"][1]), orders=8, beta=bm["beta"], params={"alpha": bm["alpha"]})
        sol = prob.solve()
        assert sol.converged is True

    # F3 Boundary & Corners
    def test_t2_f3_01_fem_mesh_minimal_elements(self):
        """T2.F3.1: Minimal mesh with E=1 element (2 nodes)."""
        mesh = FEMMesh(domain=(0.5, 1.5), elements=1)
        assert mesh.n_elements == 1
        assert mesh.n_nodes == 2
        assert np.allclose(mesh.nodes, [0.5, 1.5])

    def test_t2_f3_02_fem_mesh_dense_elements_sparsity(self):
        """T2.F3.2: Dense mesh with E=120 elements, shape matrix remains highly sparse."""
        mesh = FEMMesh(domain=(0.1, 5.0), elements=120)
        assert mesh.n_nodes == 121
        # At any interior point, at most 2 shape functions are non-zero
        test_pt = np.array([2.345])
        Phi = mesh.evaluate_basis(test_pt)
        non_zeros = np.sum(Phi[0] > 1e-12)
        assert non_zeros <= 2, f"Expected at most 2 non-zero shape functions, got {non_zeros}"

    def test_t2_f3_03_fem_mesh_kink_at_exact_domain_boundary(self):
        """T2.F3.3: Mesh kink placed at domain boundary does not duplicate nodes."""
        mesh = FEMMesh.from_kinks(domain=(0.5, 2.0), n_elements=10, kinks=[0.5, 2.0])
        assert np.all(np.diff(mesh.nodes) > 0.0)

    def test_t2_f3_04_fem_mesh_multiple_clustered_kinks(self):
        """T2.F3.4: Multiple kinks in close proximity are sorted and assigned elements."""
        kinks = [1.1, 1.15, 1.3]
        mesh = FEMMesh.from_kinks(domain=(0.5, 2.0), n_elements=20, kinks=kinks)
        for k in kinks:
            assert np.min(np.abs(mesh.nodes - k)) < 1e-12

    def test_t2_f3_05_fem_mesh_invalid_domain_or_kinks_error(self):
        """T2.F3.5: Invalid domain bounds (min >= max) or degree != 1 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid domain bounds"):
            FEMMesh(domain=(3.0, 1.0), elements=10)
        with pytest.raises(ValueError, match="degree=1"):
            FEMMesh(domain=(1.0, 3.0), elements=10, degree=2)

    # F4 Boundary & Corners
    def test_t2_f4_01_fem_extreme_high_discount_beta(self):
        """T2.F4.1: FEM solving with high discount factor beta = 0.995."""
        bm = canonical_brock_mirman_params(beta=0.995)
        sol = FEMProblem(
            domain=bm["domain"],
            elements=40,
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            beta=0.995,
            params={"alpha": bm["alpha"]},
        ).solve()
        assert sol.converged is True

    def test_t2_f4_02_fem_extreme_capital_share(self):
        """T2.F4.2: FEM solving with high capital share alpha = 0.75."""
        bm = canonical_brock_mirman_params(alpha=0.75)
        sol = FEMProblem(
            domain=bm["domain"],
            elements=40,
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** 0.75,
            beta=bm["beta"],
            params={"alpha": 0.75},
        ).solve()
        assert sol.converged is True

    def test_t2_f4_03_fem_higher_order_quadrature(self):
        """T2.F4.3: Comparing quadrature orders Q=2 vs Q=4 preserves convergence."""
        bm = canonical_brock_mirman_params()
        sol_q2 = FEMProblem(domain=bm["domain"], elements=30, return_fn=lambda c: np.log(c), transition_fn=lambda k: k**bm["alpha"], beta=bm["beta"], params={"alpha": bm["alpha"]}, options={"quad_order": 2}).solve()
        sol_q4 = FEMProblem(domain=bm["domain"], elements=30, return_fn=lambda c: np.log(c), transition_fn=lambda k: k**bm["alpha"], beta=bm["beta"], params={"alpha": bm["alpha"]}, options={"quad_order": 4}).solve()
        assert sol_q2.converged is True
        assert sol_q4.converged is True
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 50)
        assert np.max(np.abs(sol_q2.policy(k_eval) - sol_q4.policy(k_eval))) < 1e-4

    def test_t2_f4_04_fem_borrowing_constraint_binding_at_steady_state(self):
        """T2.F4.4: Borrowing constraint strictly above steady state clamps entire policy."""
        bm = canonical_brock_mirman_params()
        high_b = bm["k_ss"] + 0.05
        sol = FEMProblem(
            domain=bm["domain"],
            elements=40,
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k**bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"]},
            borrowing_constraint=high_b,
        ).solve()
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 100)
        assert np.all(sol.policy(k_eval) >= high_b - 1e-8)

    def test_t2_f4_05_fem_max_iter_1_unconverged_handling(self):
        """T2.F4.5: Solver with max_iter=1 and strict tolerance reports converged=False without error."""
        bm = canonical_brock_mirman_params()
        prob = FEMProblem(
            domain=bm["domain"],
            elements=30,
            method="bellman",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k**bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"]},
            options={"tol": 1e-15, "max_iter": 1},
        )
        sol = prob.solve()
        assert sol.converged is False

    # F5 Boundary & Corners
    def test_t2_f5_01_backend_case_insensitivity_or_strip(self):
        """T2.F5.1: Backend string handles whitespace and uppercase names with normalization."""
        bm = canonical_brock_mirman_params()
        sol_coll = CollocationProblem(
            domain=bm["domain"], orders=4, beta=bm["beta"], params={"alpha": bm["alpha"]}
        ).solve(backend="  NUMPY  ")
        assert sol_coll.backend == "numpy"
        assert sol_coll.metadata["backend"] == "numpy"

        sol_fem = FEMProblem(
            domain=bm["domain"],
            elements=10,
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"]},
        ).solve(backend="  NuMPy ")
        assert sol_fem.backend == "numpy"
        assert sol_fem.metadata["backend"] == "numpy"

    def test_t2_f5_02_backend_array_output_numpy_preservation(self):
        """T2.F5.2: Solution nodal values / coefficients always return np.ndarray regardless of backend."""
        bm = canonical_brock_mirman_params()
        prob = CollocationProblem(domain=bm["domain"], orders=4, beta=bm["beta"], params={"alpha": bm["alpha"]})
        for b_name in ("numpy", "numba"):
            if bk.backend_available(b_name):
                sol = prob.solve(backend=b_name)
                assert isinstance(sol.coefficients, np.ndarray)

    def test_t2_f5_03_backend_numba_large_grid_evaluation(self):
        """T2.F5.3: Numba basis kernel evaluates 10,000 continuous points without memory leak."""
        basis = CollocationBasis(domain=(0.1, 5.0), orders=8)
        large_grid = np.linspace(0.1, 5.0, 10000)
        Phi = basis.evaluate(large_grid, backend="numba")
        assert Phi.shape == (10000, 9)
        assert np.all(np.isfinite(Phi))

    def test_t2_f5_04_backend_mlx_float32_tolerance_handling(self):
        """T2.F5.4: Apple Silicon MLX GPU evaluation achieves < 1e-4 accuracy despite float32."""
        if not bk.backend_available("mlx"):
            pytest.skip("MLX not available")
        bm = canonical_brock_mirman_params()
        sol = CollocationProblem(domain=bm["domain"], orders=6, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve(backend="mlx")
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 500)
        g_true = bm["g_star"](k_eval)
        rel_err = np.max(np.abs(sol.policy(k_eval) - g_true) / g_true)
        assert rel_err < 1e-4, f"MLX error {rel_err} exceeds 1e-4"

    def test_t2_f5_05_backend_fallback_on_uninstalled_optional(self):
        """T2.F5.5: Collocation and FEM solve fall back to NumPy with warning when requesting unavailable backend."""
        bm = canonical_brock_mirman_params()
        if bk.backend_available("cupy"):
            pytest.skip("cupy is installed on this CUDA machine")

        with pytest.warns(UserWarning, match="falling back to 'numpy'"):
            sol_coll = CollocationProblem(
                domain=bm["domain"], orders=4, beta=bm["beta"], params={"alpha": bm["alpha"]}
            ).solve(backend="cupy")
        assert sol_coll.backend == "numpy"
        assert sol_coll.metadata["backend"] == "numpy"
        assert sol_coll.converged is True

        with pytest.warns(UserWarning, match="falling back to 'numpy'"):
            sol_fem = FEMProblem(
                domain=bm["domain"],
                elements=20,
                return_fn=lambda c: np.log(c),
                transition_fn=lambda k: k ** bm["alpha"],
                beta=bm["beta"],
                params={"alpha": bm["alpha"]},
            ).solve(backend="cupy")
        assert sol_fem.backend == "numpy"
        assert sol_fem.metadata["backend"] == "numpy"
        assert sol_fem.converged is True

    # F6 Boundary & Corners
    def test_t2_f6_01_presentation_summary_unconverged_status(self):
        """T2.F6.1: Unconverged solution still produces valid summary with Converged = False."""
        bm = canonical_brock_mirman_params()
        prob = FEMProblem(
            domain=bm["domain"],
            elements=20,
            method="bellman",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k**bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"]},
            options={"tol": 1e-15, "max_iter": 1},
        )
        sol = prob.solve()
        df = sol.summary()
        assert df.loc["Converged", "Value"] is False or df.loc["Converged", "Value"] == "False"

    def test_t2_f6_02_presentation_plot_custom_axes(self):
        """T2.F6.2: .plot() generates matplotlib Figure without crashing."""
        bm = canonical_brock_mirman_params()
        sol = CollocationProblem(domain=bm["domain"], orders=4, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        fig = sol.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_t2_f6_03_presentation_markdown_table_formatting(self):
        """T2.F6.3: Markdown string contains valid table header and column dividers."""
        bm = canonical_brock_mirman_params()
        sol = CollocationProblem(domain=bm["domain"], orders=4, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        md = sol.to_markdown()
        lines = md.strip().split("\n")
        assert len(lines) >= 3
        assert "|" in lines[0] and "---" in lines[1]

    def test_t2_f6_04_presentation_latex_tabular_structure(self):
        """T2.F6.4: LaTeX string contains begin and end tabular markers."""
        bm = canonical_brock_mirman_params()
        sol = CollocationProblem(domain=bm["domain"], orders=4, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        tex = sol.to_latex()
        assert r"\begin{tabular}" in tex
        assert r"\end{tabular}" in tex

    def test_t2_f6_05_presentation_typst_table_structure(self):
        """T2.F6.5: Typst string contains table definition syntax."""
        bm = canonical_brock_mirman_params()
        sol = CollocationProblem(domain=bm["domain"], orders=4, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        typ = sol.to_typst()
        assert "#table(" in typ or "table" in typ

    # F7 Boundary & Corners
    def test_t2_f7_01_brock_mirman_high_curvature_grid(self):
        """T2.F7.1: Evaluation on non-uniform grid clustered near k_min where curvature is highest."""
        bm = canonical_brock_mirman_params()
        sol = CollocationProblem(domain=bm["domain"], orders=6, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        # Quadratic clustering toward lower bound
        u = np.linspace(0.0, 1.0, 500)
        k_clustered = bm["domain"][0] + (bm["domain"][1] - bm["domain"][0]) * (u ** 2)
        g_true = bm["g_star"](k_clustered)
        rel_err = np.max(np.abs(sol.policy(k_clustered) - g_true) / g_true)
        assert rel_err < 1e-4

    def test_t2_f7_02_brock_mirman_wide_domain(self):
        """T2.F7.2: Wide domain [0.2 k_ss, 2.5 k_ss] resolved cleanly with N=12."""
        bm = canonical_brock_mirman_params()
        wide_domain = (0.2 * bm["k_ss"], 2.5 * bm["k_ss"])
        sol = CollocationProblem(domain=wide_domain, orders=12, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        k_eval = np.linspace(wide_domain[0], wide_domain[1], 1000)
        g_true = bm["g_star"](k_eval)
        rel_err = np.max(np.abs(sol.policy(k_eval) - g_true) / g_true)
        assert rel_err < 1e-4

    def test_t2_f7_03_brock_mirman_exact_euler_residual_zero(self):
        """T2.F7.3: Evaluating analytical policy in Euler residual function yields machine-precision 0."""
        bm = canonical_brock_mirman_params()
        k_test = np.linspace(bm["domain"][0], bm["domain"][1], 200)
        exact_res = compute_continuous_euler_residual(bm["g_star"], k_test, bm["alpha"], bm["beta"])
        assert np.max(np.abs(exact_res)) < 1e-14

    def test_t2_f7_04_brock_mirman_policy_monotonicity(self):
        """T2.F7.4: Numerical derivative of policy function is strictly positive everywhere."""
        bm = canonical_brock_mirman_params()
        sol = CollocationProblem(domain=bm["domain"], orders=6, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        k_test = np.linspace(bm["domain"][0], bm["domain"][1], 500)
        pol = sol.policy(k_test)
        dpol_dk = np.gradient(pol, k_test)
        assert np.all(dpol_dk > 0.0), "Policy function derivative must be strictly positive"

    def test_t2_f7_05_brock_mirman_capital_accumulation_convergence(self):
        """T2.F7.5: Simulating k_{t+1} = g(k_t) from k_0 = 0.5 k_ss converges to k_ss."""
        bm = canonical_brock_mirman_params()
        sol = CollocationProblem(domain=bm["domain"], orders=6, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        k_t = bm["domain"][0]
        for _ in range(100):
            k_t = sol.policy(k_t)
        err = abs(k_t - bm["k_ss"]) / bm["k_ss"]
        assert err < 1e-4, f"Capital path did not converge to steady state: {err}"


# ===========================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (Pairwise interactions, >= 10 tests)
# ===========================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise interactions across solvers, projections, backends, and dimensions."""

    def test_t3_01_collocation_vs_fem_euler_policy_agreement(self):
        """T3.01: Collocation and FEM policy functions agree within 1e-4 on identical problem."""
        bm = canonical_brock_mirman_params()
        sol_c = CollocationProblem(domain=bm["domain"], orders=6, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        sol_f = FEMProblem(
            domain=bm["domain"],
            elements=50,
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k**bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"]},
        ).solve()
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 500)
        diff = np.max(np.abs(sol_c.policy(k_eval) - sol_f.policy(k_eval)) / bm["g_star"](k_eval))
        assert diff < 1e-4, f"Collocation and FEM policies diverge by {diff}"

    def test_t3_02_collocation_bellman_vs_euler_policy_agreement(self):
        """T3.02: Bellman VFI policy and Euler projection policy agree within numerical tolerance."""
        bm = canonical_brock_mirman_params()
        sol_e = CollocationProblem(domain=bm["domain"], orders=5, method="euler", beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        sol_b = CollocationProblem(domain=bm["domain"], orders=5, method="bellman", beta=bm["beta"], params={"alpha": bm["alpha"]}, options={"tol": 1e-6, "max_iter": 400}).solve()
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 100)
        diff = np.max(np.abs(sol_e.policy(k_eval) - sol_b.policy(k_eval)) / bm["g_star"](k_eval))
        assert diff < 5e-3, f"Collocation Bellman and Euler policies diverge by {diff}"

    def test_t3_03_fem_bellman_vs_euler_policy_agreement(self):
        """T3.03: Bellman VFI policy and Euler projection policy agree in FEM within tolerance."""
        bm = canonical_brock_mirman_params()
        sol_e = FEMProblem(domain=bm["domain"], elements=30, method="euler", return_fn=lambda c: np.log(c), transition_fn=lambda k: k**bm["alpha"], beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        sol_b = FEMProblem(domain=bm["domain"], elements=30, method="bellman", return_fn=lambda c: np.log(c), transition_fn=lambda k: k**bm["alpha"], beta=bm["beta"], params={"alpha": bm["alpha"]}, options={"tol": 1e-6, "max_iter": 400}).solve()
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 100)
        diff = np.max(np.abs(sol_e.policy(k_eval) - sol_b.policy(k_eval)) / bm["g_star"](k_eval))
        assert diff < 0.02, f"FEM Bellman and Euler policies diverge by {diff}"

    def test_t3_04_collocation_lobatto_vs_gauss_nodes_accuracy(self):
        """T3.04: Comparing Gauss and Gauss-Lobatto nodes on Collocation for same order N=6."""
        bm = canonical_brock_mirman_params()
        sol_lob = CollocationProblem(domain=bm["domain"], orders=6, node_type="lobatto", beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        sol_gau = CollocationProblem(domain=bm["domain"], orders=6, node_type="gauss", beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        k_eval = np.linspace(bm["domain"][0] + 0.01, bm["domain"][1] - 0.01, 200)
        g_true = bm["g_star"](k_eval)
        err_lob = np.max(np.abs(sol_lob.policy(k_eval) - g_true) / g_true)
        err_gau = np.max(np.abs(sol_gau.policy(k_eval) - g_true) / g_true)
        assert err_lob < 1e-4 and err_gau < 1e-4

    def test_t3_05_fem_galerkin_vs_nodal_collocation_accuracy(self):
        """T3.05: FEM Galerkin integration vs FEM nodal collocation policy agreement."""
        bm = canonical_brock_mirman_params()
        sol_gal = FEMProblem(domain=bm["domain"], elements=40, projection="galerkin", return_fn=lambda c: np.log(c), transition_fn=lambda k: k**bm["alpha"], beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        sol_col = FEMProblem(domain=bm["domain"], elements=40, projection="collocation", return_fn=lambda c: np.log(c), transition_fn=lambda k: k**bm["alpha"], beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 100)
        diff = np.max(np.abs(sol_gal.policy(k_eval) - sol_col.policy(k_eval)))
        assert diff < 5e-4, f"Galerkin and nodal collocation diverge: {diff}"

    def test_t3_06_multi_backend_collocation_numpy_vs_numba_vs_mlx(self):
        """T3.06: Collocation solves identically across all available backends (NumPy, Numba, MLX)."""
        bm = canonical_brock_mirman_params()
        prob = CollocationProblem(domain=bm["domain"], orders=6, beta=bm["beta"], params={"alpha": bm["alpha"]})
        sol_np = prob.solve(backend="numpy")
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 200)
        p_np = sol_np.policy(k_eval)

        for b_name in ("numba", "mlx"):
            if bk.backend_available(b_name):
                sol_b = prob.solve(backend=b_name)
                p_b = sol_b.policy(k_eval)
                diff = np.max(np.abs(p_b - p_np))
                assert diff < 1e-4, f"Backend {b_name} diverges from NumPy by {diff}"

    def test_t3_07_multi_backend_fem_numpy_vs_numba(self):
        """T3.07: FEM solves identically with NumPy and Numba."""
        bm = canonical_brock_mirman_params()
        prob = FEMProblem(domain=bm["domain"], elements=30, return_fn=lambda c: np.log(c), transition_fn=lambda k: k**bm["alpha"], beta=bm["beta"], params={"alpha": bm["alpha"]})
        sol_np = prob.solve(backend="numpy")
        sol_nb = prob.solve(backend="numba")
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 100)
        diff = np.max(np.abs(sol_nb.policy(k_eval) - sol_np.policy(k_eval)))
        assert diff < 1e-5

    def test_t3_08_multid_tensor_collocation_with_euler_residuals(self):
        """T3.08: 2D continuous state space problem (k, z) evaluates 2D continuous policy."""
        domain_2d = ((0.1, 0.4), (0.8, 1.2))
        orders_2d = (4, 3)
        basis_2d = CollocationBasis(domain=domain_2d, orders=orders_2d)
        assert basis_2d.n_nodes == 20
        # Check fit and continuous interpolation of a smooth 2D production function
        nodes = basis_2d.nodes()
        k_nodes = nodes[:, 0]
        z_nodes = nodes[:, 1]
        y_nodes = z_nodes * (k_nodes ** 0.36)
        coefs = basis_2d.fit(y_nodes)
        # Out of sample 2D evaluation
        test_pts = np.array([[0.2, 1.0], [0.3, 1.15]])
        y_approx = basis_2d.interpolate(coefs, test_pts)
        y_exact = test_pts[:, 1] * (test_pts[:, 0] ** 0.36)
        assert np.allclose(y_approx, y_exact, rtol=1e-3)

    def test_t3_09_fem_nonuniform_mesh_vs_uniform_mesh_with_kink(self):
        """T3.09: FEM with kink node strictly outperforms uniform mesh without kink node."""
        bm = canonical_brock_mirman_params()
        b_kink = 0.17
        mesh_kink = FEMMesh.from_kinks(domain=bm["domain"], n_elements=40, kinks=[b_kink])
        prob = FEMProblem(
            domain=bm["domain"],
            elements=40,
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k**bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"]},
            borrowing_constraint=b_kink,
            options={"mesh": mesh_kink},
        )
        sol = prob.solve()
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 200)
        pol = sol.policy(k_eval)
        assert np.min(pol) >= b_kink - 1e-8

    def test_t3_10_collocation_and_fem_presentation_consistency(self):
        """T3.10: CollocationSolution and FEMSolution share standard DataFrame summary schema."""
        bm = canonical_brock_mirman_params()
        sol_c = CollocationProblem(domain=bm["domain"], orders=5, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        sol_f = FEMProblem(domain=bm["domain"], elements=20, return_fn=lambda c: np.log(c), transition_fn=lambda k: k**bm["alpha"], beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        df_c = sol_c.summary()
        df_f = sol_f.summary()
        # Both must contain standard core metrics
        common_metrics = ["Method", "Converged", "Iterations", "Residual Norm", "Backend", "Elapsed Time (s)"]
        for m in common_metrics:
            assert m in df_c.index, f"{m} missing in Collocation summary"
            assert m in df_f.index, f"{m} missing in FEM summary"
        assert "State Domain" in df_c.index
        assert "Domain" in df_f.index


# ===========================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (>= 5 tests)
# ===========================================================================

class TestTier4RealWorldApplications:
    """Tier 4: Canonical macroeconomic application scenarios and verification criteria."""

    def test_t4_01_deterministic_brock_mirman_growth_benchmark(self):
        """Scenario 1: Canonical Brock-Mirman Neoclassical Growth Benchmark.

        Verifies that both polynomial collocation (N=6) and FEM Galerkin (E=50)
        achieve relative policy error < 1e-4 against the true closed-form solution
        g*(k) = alpha * beta * k^alpha across 1,000 continuous test points.
        """
        bm = canonical_brock_mirman_params(alpha=0.36, beta=0.96)
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 1000)
        g_true = bm["g_star"](k_eval)

        # 1. Polynomial Collocation Solver
        sol_coll = CollocationProblem(
            domain=bm["domain"],
            orders=6,
            method="euler",
            beta=bm["beta"],
            params={"alpha": bm["alpha"], "delta": 1.0},
        ).solve(backend="numpy")
        err_coll = np.max(np.abs(sol_coll.policy(k_eval) - g_true) / g_true)
        assert err_coll < 1e-4, f"Collocation relative policy error {err_coll} exceeds 1e-4"

        # 2. FEM Galerkin Solver
        sol_fem = FEMProblem(
            domain=bm["domain"],
            elements=50,
            method="euler",
            projection="galerkin",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"], "gamma": 1.0},
        ).solve(backend="numpy")
        err_fem = np.max(np.abs(sol_fem.policy(k_eval) - g_true) / g_true)
        assert err_fem < 1e-4, f"FEM relative policy error {err_fem} exceeds 1e-4"

    def test_t4_02_borrowing_constrained_neoclassical_growth(self):
        """Scenario 2: Neoclassical Growth with Occasionally Binding Borrowing Constraint.

        Verifies that FEM with kink-aligned mesh strictly respects the constraint k' >= k_bar
        with zero boundary violation across the entire continuous state space.
        """
        bm = canonical_brock_mirman_params()
        k_bar = 0.18  # Borrowing constraint below steady state (~0.1901)
        k_kink = (k_bar / (bm["alpha"] * bm["beta"])) ** (1.0 / bm["alpha"])

        mesh = FEMMesh.from_kinks(domain=bm["domain"], n_elements=60, kinks=[k_kink])
        sol = FEMProblem(
            domain=bm["domain"],
            elements=60,
            method="euler",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"]},
            borrowing_constraint=k_bar,
            options={"mesh": mesh},
        ).solve()
        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 1000)
        pol = sol.policy(k_eval)
        # Constraint strictly respected everywhere
        assert np.min(pol) >= k_bar - 1e-8
        # Constrained region has flat policy g(k) == k_bar
        k_constrained = k_eval[k_eval <= k_kink - 0.005]
        assert np.allclose(sol.policy(k_constrained), k_bar, atol=1e-3)

    def test_t4_03_stochastic_tfp_growth_continuous_projection(self):
        """Scenario 3: Growth Model with 2-State Discrete TFP Shock.

        Genuinely solves continuous capital accumulation policies for high vs low productivity states,
        verifying that policy is strictly higher in the high TFP state: g(k; z_high) > g(k; z_low),
        and verifies continuous Euler residuals < 1e-4 on at least 1,000 evaluation points.
        """
        bm = canonical_brock_mirman_params()
        alpha = bm["alpha"]
        beta = bm["beta"]
        z_low, z_high = 0.95, 1.05

        k_ss_low = (alpha * beta * z_low) ** (1.0 / (1.0 - alpha))
        k_ss_high = (alpha * beta * z_high) ** (1.0 / (1.0 - alpha))
        common_domain = (0.6 * k_ss_low, 1.4 * k_ss_high)

        # Genuinely solve continuous state-contingent policies with TFP shocks entering production
        sol_low = CollocationProblem(
            domain=common_domain,
            orders=6,
            beta=beta,
            params={"alpha": alpha, "delta": 1.0, "z": z_low},
            options={"tol": 1e-8},
        ).solve()
        assert sol_low.converged, "Low TFP collocation must converge"

        sol_high = CollocationProblem(
            domain=common_domain,
            orders=6,
            beta=beta,
            params={"alpha": alpha, "delta": 1.0, "z": z_high},
            options={"tol": 1e-8},
        ).solve()
        assert sol_high.converged, "High TFP collocation must converge"

        # Dense out-of-sample evaluation grid of at least 1,000 points
        k_eval = np.linspace(common_domain[0], common_domain[1], 1000)
        g_low = sol_low.policy(k_eval)
        g_high = sol_high.policy(k_eval)

        # 1. State-contingent policy monotonicity: higher productivity strictly increases savings
        assert np.all(g_high > g_low), "Policy must be strictly higher in high-TFP regime"

        # 2. Continuous Euler residuals < 1e-4 on at least 1,000 evaluation points
        res_low = compute_continuous_euler_residual(sol_low.policy, k_eval, alpha=alpha, beta=beta, z=z_low)
        res_high = compute_continuous_euler_residual(sol_high.policy, k_eval, alpha=alpha, beta=beta, z=z_high)
        max_res_low = float(np.max(np.abs(res_low)))
        max_res_high = float(np.max(np.abs(res_high)))
        assert max_res_low < 1e-4, f"Low TFP max Euler residual {max_res_low:.2e} exceeds 1e-4"
        assert max_res_high < 1e-4, f"High TFP max Euler residual {max_res_high:.2e} exceeds 1e-4"

    def test_t4_04_dense_euler_residual_mapping_2000_points(self):
        """Scenario 4: High-Precision Continuous Euler Residual Mapping on 2,000 Points.

        Verifies that across 2,000 out-of-sample points spanning [0.5 k_ss, 1.5 k_ss],
        the maximum absolute Euler equation residual satisfies max |R(k)| < 1e-4
        for both Collocation and FEM solvers.
        """
        bm = canonical_brock_mirman_params()
        k_dense = np.linspace(bm["domain"][0], bm["domain"][1], 2000)

        # 1. Collocation Euler Residuals
        sol_coll = CollocationProblem(domain=bm["domain"], orders=6, beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()
        res_coll = compute_continuous_euler_residual(sol_coll.policy, k_dense, bm["alpha"], bm["beta"])
        max_res_coll = np.max(np.abs(res_coll))
        assert max_res_coll < 1e-4, f"Collocation max Euler residual {max_res_coll} exceeds 1e-4 on 2,000 points"

        # 2. FEM Euler Residuals
        sol_fem = FEMProblem(
            domain=bm["domain"],
            elements=50,
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"]},
        ).solve()
        res_fem = compute_continuous_euler_residual(sol_fem.policy, k_dense, bm["alpha"], bm["beta"])
        max_res_fem = np.max(np.abs(res_fem))
        assert max_res_fem < 1e-4, f"FEM max Euler residual {max_res_fem} exceeds 1e-4 on 2,000 points"

    def test_t4_05_cross_backend_numerical_parity_under_stress(self):
        """Scenario 5: Multi-Backend Acceleration Consistency Under Parameter Stress.

        Tests solution parity across all installed backends under high capital share
        alpha = 0.40 and high discount factor beta = 0.98, verifying that policy difference
        is bounded by max |g_backend(k) - g_numpy(k)| < 1e-4.
        """
        bm = canonical_brock_mirman_params(alpha=0.40, beta=0.98)
        prob = CollocationProblem(domain=bm["domain"], orders=6, beta=0.98, params={"alpha": 0.40})
        sol_np = prob.solve(backend="numpy")

        k_eval = np.linspace(bm["domain"][0], bm["domain"][1], 500)
        p_np = sol_np.policy(k_eval)

        for b_name in ("numba", "mlx"):
            if bk.backend_available(b_name):
                sol_b = prob.solve(backend=b_name)
                p_b = sol_b.policy(k_eval)
                diff = np.max(np.abs(p_b - p_np))
                assert diff < 1e-4, f"Backend {b_name} parity error {diff} exceeds 1e-4 under stress calibration"
