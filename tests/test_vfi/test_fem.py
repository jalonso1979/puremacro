"""Unit and regression tests for Finite Element Method (FEM / Galerkin) solver.

Verifies:
- FEMMesh: 1D and tensor-product multi-D meshes, uniform, clustered, and kink-aligned meshes.
- Basis evaluation: partition of unity, Kronecker delta property, non-negativity.
- Gauss-Legendre quadrature exactness.
- Defensive validation in FEMProblem.
- Analytical Brock-Mirman neoclassical growth benchmark (policy error < 1e-4, Euler residual < 1e-4).
- Borrowing constraint handling, Kuhn-Tucker complementarity, and kink placement (zero Gibbs ringing).
- Continuous Bellman value function iteration with Howard acceleration.
- Full presentation contract (.summary(), .plot(), .to_frame(), .to_markdown(), .to_latex(), .to_typst()).
- Multi-backend execution and graceful fallback.
- solve_fem functional interface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import puremacro._backend as bk
from puremacro.vfi.fem import (
    FEMMesh,
    FEMProblem,
    FEMSolution,
    gauss_legendre_quadrature,
    solve_fem,
)


# ---------------------------------------------------------------------------
# Test Fixtures & Analytical Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def brock_mirman_params():
    """Canonical Brock-Mirman neoclassical growth parameters."""
    alpha = 0.36
    beta = 0.96
    k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
    domain = (0.5 * k_ss, 1.5 * k_ss)
    return {
        "alpha": alpha,
        "beta": beta,
        "k_ss": k_ss,
        "domain": domain,
        "transition_fn": lambda k: k**alpha,
        "return_fn": lambda c: np.log(np.maximum(c, 1e-14)),
    }


# ---------------------------------------------------------------------------
# 1. Mesh and Basis Function Tests
# ---------------------------------------------------------------------------


class TestFEMMesh:
    """Tests for 1D and tensor product FEMMesh."""

    def test_uniform_mesh_1d_properties(self):
        mesh = FEMMesh(domain=(0.1, 2.0), elements=10)
        assert mesh.dim == 1
        assert mesh.n_elements == 10
        assert mesh.n_nodes == 11
        assert mesh.degree == 1
        assert np.isclose(mesh.nodes[0], 0.1)
        assert np.isclose(mesh.nodes[-1], 2.0)
        assert np.all(np.diff(mesh.nodes) > 0)
        # Check equidistant spacing
        diffs = np.diff(mesh.nodes)
        assert np.allclose(diffs, (2.0 - 0.1) / 10)

    def test_clustered_mesh(self):
        mesh = FEMMesh.create_clustered(domain=(0.0, 1.0), elements=20, clustering=2.5)
        assert mesh.n_nodes == 21
        assert np.isclose(mesh.nodes[0], 0.0)
        assert np.isclose(mesh.nodes[-1], 1.0)
        # First element should be smaller than last element due to power clustering
        h_first = mesh.nodes[1] - mesh.nodes[0]
        h_last = mesh.nodes[-1] - mesh.nodes[-2]
        assert h_first < h_last

    def test_from_kinks_mesh(self):
        kink_point = 0.42
        mesh = FEMMesh.from_kinks(domain=(0.0, 1.0), n_elements=30, kinks=[kink_point])
        # Verify that kink_point is exactly an element boundary node
        assert any(np.isclose(node, kink_point, atol=1e-12) for node in mesh.nodes)
        assert np.all(np.diff(mesh.nodes) > 0)
        assert mesh.n_elements >= 2

    def test_basis_partition_of_unity_1d(self):
        mesh = FEMMesh(domain=(0.5, 3.5), elements=15)
        test_points = np.linspace(0.5, 3.5, 200)
        Phi = mesh.evaluate_basis(test_points)
        assert Phi.shape == (200, 16)
        # Partition of unity: sum of shape functions equals 1.0 everywhere
        row_sums = np.sum(Phi, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-14)
        # Non-negativity
        assert np.all(Phi >= -1e-14)

    def test_basis_kronecker_delta_1d(self):
        mesh = FEMMesh(domain=(1.0, 5.0), elements=8)
        # At mesh nodes, Phi must be the identity matrix
        Phi_nodes = mesh.evaluate_basis(mesh.nodes)
        assert np.allclose(Phi_nodes, np.eye(9), atol=1e-14)

    def test_tensor_product_mesh_2d(self):
        dom2d = ((0.0, 1.0), (10.0, 20.0))
        elem2d = (5, 4)
        mesh = FEMMesh(domain=dom2d, elements=elem2d)
        assert mesh.dim == 2
        assert mesh.n_elements == 20
        assert mesh.n_nodes == 6 * 5  # (5+1) * (4+1) = 30
        assert mesh.nodes.shape == (30, 2)

        # Kronecker delta at 2D nodes
        Phi_nodes = mesh.evaluate_basis(mesh.nodes)
        assert np.allclose(Phi_nodes, np.eye(30), atol=1e-14)

        # Partition of unity at test points
        test_pts = np.array([[0.25, 12.5], [0.8, 18.2], [0.0, 10.0], [1.0, 20.0]])
        Phi_pts = mesh.evaluate_basis(test_pts)
        assert np.allclose(np.sum(Phi_pts, axis=1), 1.0, atol=1e-14)

    def test_invalid_mesh_inputs_raise(self):
        with pytest.raises(ValueError, match="Invalid domain bounds"):
            FEMMesh(domain=(2.0, 1.0), elements=10)

        with pytest.raises(ValueError, match="Number of elements"):
            FEMMesh(domain=(0.0, 1.0), elements=0)

        with pytest.raises(ValueError, match="strictly monotonically increasing"):
            FEMMesh(domain=(0.0, 1.0), elements=3, nodes=np.array([0.0, 0.5, 0.4, 1.0]))

        with pytest.raises(ValueError, match="degree=1"):
            FEMMesh(domain=(0.0, 1.0), elements=10, degree=2)


# ---------------------------------------------------------------------------
# 2. Gauss-Legendre Quadrature Tests
# ---------------------------------------------------------------------------


class TestGaussLegendreQuadrature:
    """Tests for Gauss-Legendre quadrature integration."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    def test_weights_sum_to_two(self, order):
        xi, w = gauss_legendre_quadrature(order)
        assert len(xi) == order
        assert len(w) == order
        assert np.isclose(np.sum(w), 2.0, atol=1e-14)
        assert np.all(xi >= -1.0) and np.all(xi <= 1.0)

    def test_quadrature_exactness_degree_three(self):
        # 3-point quadrature integrates up to degree 2*3 - 1 = 5 exactly
        xi, w = gauss_legendre_quadrature(order=3)
        # int_{-1}^1 (3x^4 - 2x^2 + 5) dx = 3(2/5) - 2(2/3) + 5(2) = 6/5 - 4/3 + 10 = 1.2 - 1.333333 + 10 = 9.866667
        true_val = 6.0 / 5.0 - 4.0 / 3.0 + 10.0
        approx_val = np.sum(w * (3.0 * xi**4 - 2.0 * xi**2 + 5.0))
        assert np.isclose(approx_val, true_val, atol=1e-14)

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            gauss_legendre_quadrature(0)


# ---------------------------------------------------------------------------
# 3. FEMProblem Validation Tests
# ---------------------------------------------------------------------------


class TestFEMProblemValidation:
    """Tests for defensive parameter checking in FEMProblem."""

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError, match="beta"):
            FEMProblem(domain=(0.1, 1.0), beta=1.0, return_fn=lambda c: c, transition_fn=lambda k: k)

        with pytest.raises(ValueError, match="beta"):
            FEMProblem(domain=(0.1, 1.0), beta=-0.1, return_fn=lambda c: c, transition_fn=lambda k: k)

    def test_invalid_method_and_projection_raise(self):
        with pytest.raises(ValueError, match="Invalid method"):
            FEMProblem(domain=(0.1, 1.0), method="spectral", return_fn=lambda c: c)

        with pytest.raises(ValueError, match="Invalid projection"):
            FEMProblem(domain=(0.1, 1.0), projection="least_squares", return_fn=lambda c: c)

    def test_missing_functions_raise(self):
        with pytest.raises(ValueError, match="method='euler'"):
            FEMProblem(domain=(0.1, 1.0), method="euler")

        with pytest.raises(ValueError, match="method='bellman'"):
            FEMProblem(domain=(0.1, 1.0), method="bellman")

    def test_invalid_borrowing_constraint_raises(self):
        with pytest.raises(ValueError, match="Borrowing constraint"):
            FEMProblem(
                domain=(0.5, 2.0),
                borrowing_constraint=0.2,
                return_fn=lambda c: c,
                transition_fn=lambda k: k,
            )


# ---------------------------------------------------------------------------
# 4. Brock-Mirman Neoclassical Growth Benchmark Tests
# ---------------------------------------------------------------------------


class TestFEMBrockMirmanBenchmark:
    """Benchmark tests against the closed-form Brock-Mirman model."""

    def test_galerkin_euler_accuracy_and_convergence(self, brock_mirman_params):
        p = brock_mirman_params
        alpha = p["alpha"]
        beta = p["beta"]
        domain = p["domain"]

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

        assert sol.converged is True
        assert sol.residual_norm < 1e-6

        # Evaluate continuous policy over 1,000 dense points
        test_k = np.linspace(domain[0], domain[1], 1000)
        g_true = alpha * beta * test_k**alpha
        g_approx = sol.policy(test_k)

        # Requirement: relative error < 1e-4
        max_rel_error = np.max(np.abs(g_approx - g_true) / g_true)
        assert max_rel_error < 1e-4, f"Policy relative error {max_rel_error} exceeds 1e-4"

        # Requirement: continuous Euler equation residual < 1e-4
        max_euler_res = sol.metadata["max_euler_residual"]
        assert max_euler_res < 1e-4, f"Max Euler residual {max_euler_res} exceeds 1e-4"

    def test_collocation_euler_convergence(self, brock_mirman_params):
        p = brock_mirman_params
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
        assert sol.converged is True

        test_k = np.linspace(p["domain"][0], p["domain"][1], 500)
        g_true = p["alpha"] * p["beta"] * test_k ** p["alpha"]
        g_approx = sol.policy(test_k)
        max_rel_error = np.max(np.abs(g_approx - g_true) / g_true)
        assert max_rel_error < 1e-3

    def test_explicit_euler_residual_fn(self, brock_mirman_params):
        p = brock_mirman_params
        alpha = p["alpha"]
        beta = p["beta"]

        def custom_euler(s, sp, sn, **kwargs):
            c = s**alpha - sp
            cp = sp**alpha - sn
            return 1.0 - beta * (c / cp) * alpha * sp ** (alpha - 1.0)

        prob = FEMProblem(
            domain=p["domain"],
            elements=50,
            method="euler",
            projection="galerkin",
            euler_residual_fn=custom_euler,
            beta=beta,
            params={"alpha": alpha},
        )
        sol = prob.solve()
        assert sol.converged is True
        assert sol.metadata["max_euler_residual"] < 1e-4

    def test_continuous_evaluation_types(self, brock_mirman_params):
        p = brock_mirman_params
        prob = FEMProblem(
            domain=p["domain"],
            elements=20,
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=p["beta"],
            params={"alpha": p["alpha"], "gamma": 1.0},
        )
        sol = prob.solve()

        # Scalar evaluation
        k_mid = 0.5 * (p["domain"][0] + p["domain"][1])
        pol_scalar = sol.policy(k_mid)
        val_scalar = sol.value(k_mid)
        assert isinstance(pol_scalar, float)
        assert isinstance(val_scalar, float)

        # Array evaluation
        k_arr = np.array([p["domain"][0], k_mid, p["domain"][1]])
        pol_arr = sol.policy(k_arr)
        val_arr = sol.value(k_arr)
        assert isinstance(pol_arr, np.ndarray) and len(pol_arr) == 3
        assert isinstance(val_arr, np.ndarray) and len(val_arr) == 3


# ---------------------------------------------------------------------------
# 5. Borrowing Constraint and Policy Kink Tests
# ---------------------------------------------------------------------------


class TestFEMBorrowingConstraintAndKinks:
    """Tests for occasionally binding borrowing constraints and policy kinks."""

    def test_kink_aligned_mesh_prevents_oscillations(self, brock_mirman_params):
        p = brock_mirman_params
        alpha = p["alpha"]
        beta = p["beta"]
        b_const = 0.18  # Borrowing constraint

        # Calculate exact kink where unconstrained policy hits b_const:
        # alpha * beta * (k*)^alpha = b_const => k* = (b_const / (alpha * beta))^(1/alpha)
        k_kink = (b_const / (alpha * beta)) ** (1.0 / alpha)
        domain = (0.10, 0.30)
        assert domain[0] < k_kink < domain[1]

        # Kink-aligned mesh with exact node at k_kink
        mesh = FEMMesh.from_kinks(domain=domain, n_elements=40, kinks=[k_kink])
        prob = FEMProblem(
            domain=domain,
            elements=40,
            method="euler",
            projection="galerkin",
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=beta,
            borrowing_constraint=b_const,
            params={"alpha": alpha, "gamma": 1.0},
            options={"mesh": mesh},
        )
        sol = prob.solve()
        assert sol.converged is True

        # Dense continuous test grid spanning constrained and unconstrained zones
        dense_k = np.linspace(domain[0], domain[1], 500)
        pol_dense = sol.policy(dense_k)

        # 1. No constraint violations: policy must be >= borrowing constraint everywhere
        assert np.all(pol_dense >= b_const - 1e-10)

        # 2. Left of kink: policy must be flat at borrowing constraint (zero overshoot/ringing)
        constrained_k = dense_k[dense_k <= k_kink]
        pol_constrained = sol.policy(constrained_k)
        assert np.allclose(pol_constrained, b_const, atol=1e-6)

        # 3. Right of kink: policy strictly increases with state
        unconstrained_k = dense_k[dense_k > k_kink + 0.01]
        pol_unconstrained = sol.policy(unconstrained_k)
        assert np.all(np.diff(pol_unconstrained) > 0)

    def test_min_complementarity_option(self, brock_mirman_params):
        p = brock_mirman_params
        prob = FEMProblem(
            domain=(0.10, 0.30),
            elements=30,
            borrowing_constraint=0.17,
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=p["beta"],
            params={"alpha": p["alpha"], "gamma": 1.0},
            options={"complementarity": "min"},
        )
        sol = prob.solve()
        assert sol.converged is True
        pol_vals = sol.policy(np.linspace(0.10, 0.30, 200))
        assert np.all(pol_vals >= 0.17 - 1e-8)


# ---------------------------------------------------------------------------
# 6. Bellman Value Function Iteration Tests
# ---------------------------------------------------------------------------


class TestFEMBellmanIteration:
    """Tests for continuous Bellman value function iteration."""

    def test_bellman_convergence_and_monotonicity(self, brock_mirman_params):
        p = brock_mirman_params
        alpha = p["alpha"]
        beta = p["beta"]

        prob = FEMProblem(
            domain=p["domain"],
            elements=25,
            method="bellman",
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=beta,
            params={"alpha": alpha, "gamma": 1.0},
            options={"tol": 1e-6, "howard": True, "n_howard": 10},
        )
        sol = prob.solve()
        assert sol.converged is True
        assert sol.metadata["method"] == "bellman"

        # Policy should be monotonic in state (allowing for 1e-7 numerical solver noise)
        nodes = sol.mesh.nodes
        pol = sol.policy(nodes)
        assert np.all(np.diff(pol) >= -1e-7)
        assert pol[-1] > pol[0]

        # Value function should be strictly increasing and concave
        val = sol.value(nodes)
        assert np.all(np.diff(val) > 0)
        # Second difference <= 0 (concavity)
        assert np.all(np.diff(val, 2) <= 1e-4)


# ---------------------------------------------------------------------------
# 7. Presentation Contract Tests
# ---------------------------------------------------------------------------


class TestFEMPresentationContract:
    """Tests verifying full compliance with puremacro presentation contract."""

    @pytest.fixture
    def solved_solution(self, brock_mirman_params):
        p = brock_mirman_params
        prob = FEMProblem(
            domain=p["domain"],
            elements=20,
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=p["beta"],
            params={"alpha": p["alpha"], "gamma": 1.0},
        )
        return prob.solve()

    def test_summary_dataframe(self, solved_solution):
        df = solved_solution.summary()
        assert isinstance(df, pd.DataFrame)
        assert "Value" in df.columns
        assert "Method" in df.index
        assert "Projection" in df.index
        assert "Converged" in df.index
        assert "Max Euler Residual" in df.index
        assert bool(df.loc["Converged", "Value"]) is True

    def test_to_frame_equals_summary(self, solved_solution):
        df_frame = solved_solution.to_frame()
        df_summary = solved_solution.summary()
        pd.testing.assert_frame_equal(df_frame, df_summary)

    def test_to_markdown_formatting(self, solved_solution):
        md = solved_solution.to_markdown()
        assert isinstance(md, str)
        assert "Metric" in md
        assert "Method" in md
        assert "---" in md
        assert "euler" in md

    def test_to_latex_formatting(self, solved_solution):
        latex = solved_solution.to_latex()
        assert isinstance(latex, str)
        assert r"\begin{tabular}" in latex
        assert r"\end{tabular}" in latex
        assert r"\\" in latex

    def test_to_typst_formatting(self, solved_solution):
        typst = solved_solution.to_typst()
        assert isinstance(typst, str)
        assert "#table(" in typst
        assert "columns:" in typst

    def test_plot_figure(self, solved_solution):
        fig = solved_solution.plot(n_points=100)
        # Verify it returns a valid matplotlib Figure
        assert fig is not None
        assert hasattr(fig, "axes")
        assert len(fig.axes) == 3


# ---------------------------------------------------------------------------
# 8. Multi-Backend & Functional solve_fem Tests
# ---------------------------------------------------------------------------


class TestFEMBackendsAndFunctionalAPI:
    """Tests for compute backends and solve_fem functional wrapper."""

    def test_solve_backend_numpy(self, brock_mirman_params):
        p = brock_mirman_params
        prob = FEMProblem(
            domain=p["domain"],
            elements=15,
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=p["beta"],
            params={"alpha": p["alpha"], "gamma": 1.0},
        )
        sol = prob.solve(backend="numpy")
        assert sol.backend == "numpy"
        assert sol.converged is True

    def test_solve_backend_fallback_cupy(self, brock_mirman_params):
        # cupy is not installed in this environment; verify graceful fallback to numpy
        p = brock_mirman_params
        prob = FEMProblem(
            domain=p["domain"],
            elements=15,
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=p["beta"],
            params={"alpha": p["alpha"], "gamma": 1.0},
        )
        sol = prob.solve(backend="cupy")
        # Should gracefully succeed using numpy without raising an exception
        assert sol.backend == "numpy"
        assert sol.converged is True

    def test_solve_fem_functional_with_problem_instance(self, brock_mirman_params):
        p = brock_mirman_params
        prob = FEMProblem(
            domain=p["domain"],
            elements=15,
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=p["beta"],
            params={"alpha": p["alpha"], "gamma": 1.0},
        )
        sol = solve_fem(prob, backend="numpy")
        assert isinstance(sol, FEMSolution)
        assert sol.converged is True

    def test_solve_fem_functional_with_args(self, brock_mirman_params):
        p = brock_mirman_params
        sol = solve_fem(
            p["domain"],
            elements=15,
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=p["beta"],
            params={"alpha": p["alpha"], "gamma": 1.0},
            backend="numpy",
        )
        assert isinstance(sol, FEMSolution)
        assert sol.converged is True

    def test_tensor_product_mesh_3d(self):
        dom3d = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0))
        elem3d = (3, 2, 2)
        mesh = FEMMesh(domain=dom3d, elements=elem3d)
        assert mesh.dim == 3
        assert mesh.n_elements == 12
        assert mesh.n_nodes == 4 * 3 * 3  # 36
        assert mesh.nodes.shape == (36, 3)

        # Partition of unity at test points
        pts = np.array([[0.5, 1.5, 2.5], [0.1, 1.9, 2.2]])
        Phi = mesh.evaluate_basis(pts)
        assert Phi.shape == (2, 36)
        assert np.allclose(np.sum(Phi, axis=1), 1.0, atol=1e-14)

    def test_mesh_repr_and_clamping(self):
        mesh = FEMMesh((0.0, 10.0), elements=5)
        assert "FEMMesh" in repr(mesh)
        assert "elements=(5,)" in repr(mesh)
        assert "n_nodes=6" in repr(mesh)

        # Points outside bounds should be clamped gracefully
        pts_outside = np.array([-1.0, 11.0])
        Phi = mesh.evaluate_basis(pts_outside)
        assert np.allclose(np.sum(Phi, axis=1), 1.0)
        assert np.isclose(Phi[0, 0], 1.0)  # clamped to left boundary
        assert np.isclose(Phi[1, -1], 1.0)  # clamped to right boundary

    def test_brock_mirman_quadratic_convergence(self, brock_mirman_params):
        """Verify algebraic O(h^2) error convergence for piecewise linear FEM."""
        p = brock_mirman_params
        test_k = np.linspace(p["domain"][0], p["domain"][1], 1000)
        g_true = p["alpha"] * p["beta"] * test_k ** p["alpha"]

        # Low element count: 20 elements
        prob_20 = FEMProblem(
            domain=p["domain"],
            elements=20,
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=p["beta"],
            params={"alpha": p["alpha"], "gamma": 1.0},
        )
        sol_20 = prob_20.solve()
        err_20 = np.max(np.abs(sol_20.policy(test_k) - g_true))

        # High element count: 80 elements (h reduced by 4x => error reduced by ~16x)
        prob_80 = FEMProblem(
            domain=p["domain"],
            elements=80,
            return_fn=p["return_fn"],
            transition_fn=p["transition_fn"],
            beta=p["beta"],
            params={"alpha": p["alpha"], "gamma": 1.0},
        )
        sol_80 = prob_80.solve()
        err_80 = np.max(np.abs(sol_80.policy(test_k) - g_true))

        # Error should decrease by at least 10x
        assert err_80 < err_20 / 10.0
        assert err_80 < 1e-5

    @pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
    def test_alternative_root_methods(self, brock_mirman_params):
        p = brock_mirman_params
        for r_method in ("lm", "broyden1"):
            prob = FEMProblem(
                domain=p["domain"],
                elements=20,
                return_fn=p["return_fn"],
                transition_fn=p["transition_fn"],
                beta=p["beta"],
                params={"alpha": p["alpha"], "gamma": 1.0},
                options={"root_method": r_method, "tol": 1e-7},
            )
            sol = prob.solve()
            assert sol.converged is True

    def test_missing_residual_fn_raises_runtime_error(self):
        mesh = FEMMesh((0.0, 1.0), 5)
        sol = FEMSolution(
            nodal_values=np.zeros(6),
            mesh=mesh,
            converged=True,
            n_iter=1,
            residual_norm=0.0,
            backend="numpy",
            metadata={},
        )
        with pytest.raises(RuntimeError, match="not available in metadata"):
            sol.euler_residual(np.array([0.5]))

    @pytest.mark.skipif(not bk.backend_available("mlx"), reason="mlx backend not installed")
    def test_mlx_backend_basis_parity(self):
        mx = bk.get_array_namespace("mlx")
        mesh = FEMMesh((0.0, 1.0), 10)
        eval_pts = np.linspace(0.0, 1.0, 50)
        phi_mlx = mesh.evaluate_basis(eval_pts, xp=mx)
        phi_np = mesh.evaluate_basis(eval_pts, xp=np)
        assert np.allclose(bk.to_numpy(phi_mlx), phi_np, atol=1e-6)
