"""Adversarial Empirical Stress Test Suite for Continuous Projection & Dynamic Programming in puremacro.vfi.

Tests:
1. Extreme parameter regimes: discount factors beta in [0.01, 0.999], capital share alpha in [0.01, 0.99].
2. Wide state space domains: k in [10^-3, 10^3] with multi-scale and geometric mesh handling.
3. High polynomial degrees (N up to 100) and ultra-fine FEM meshes (E up to 500).
4. Adversarial kink handling with tight borrowing constraints: FEM exact node placement vs global Chebyshev Gibbs ringing.
5. Zero boundary violations and numerical stability without NaN/Inf across dense 5,000-point grids.
6. Multi-backend numerical parity under stress (NumPy, Numba, MLX).
7. 2D tensor product basis stress testing.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
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


# ===========================================================================
# 1. Extreme Parameter Regimes: Discount Factor beta in [0.01, 0.999]
# ===========================================================================

class TestAdversarialExtremeBeta:
    """Stress-test solvers under extreme patient and impatient discount factors."""

    @pytest.mark.parametrize("beta", [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999])
    def test_adv_extreme_beta_collocation(self, beta: float) -> None:
        """Collocation Euler projection solves stably across extreme beta in [0.01, 0.999]."""
        alpha = 0.36
        k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
        domain = (0.2 * k_ss, 2.0 * k_ss)

        prob = CollocationProblem(
            domain=domain,
            orders=8,
            beta=beta,
            params={"alpha": alpha, "delta": 1.0},
        )
        sol = prob.solve()

        assert sol.converged, f"Collocation failed to converge for beta={beta}"
        assert sol.residual_norm < 1e-6, f"Residual norm {sol.residual_norm} too high for beta={beta}"

        # Evaluate on dense grid
        test_k = np.linspace(domain[0], domain[1], 500)
        pol = sol.policy(test_k)
        assert np.all(np.isfinite(pol)), f"Non-finite policy values found for beta={beta}"
        assert np.all(pol > 0), f"Non-positive policy values found for beta={beta}"

        # Analytical Brock-Mirman check: relative error < 1e-3
        true_pol = alpha * beta * (test_k ** alpha)
        rel_err = np.max(np.abs(pol - true_pol) / true_pol)
        assert rel_err < 1e-3, f"Relative policy error {rel_err:.2e} exceeded 1e-3 for beta={beta}"

    @pytest.mark.parametrize("beta", [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999])
    def test_adv_extreme_beta_fem(self, beta: float) -> None:
        """FEM Euler Galerkin solves stably across extreme beta in [0.01, 0.999]."""
        alpha = 0.36
        k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
        domain = (0.2 * k_ss, 2.0 * k_ss)

        prob = FEMProblem(
            domain=domain,
            elements=50,
            beta=beta,
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** alpha,
            params={"alpha": alpha, "gamma": 1.0},
        )
        sol = prob.solve()

        assert sol.converged, f"FEM failed to converge for beta={beta}"
        assert sol.residual_norm < 1e-6, f"Residual norm {sol.residual_norm} too high for beta={beta}"

        test_k = np.linspace(domain[0], domain[1], 500)
        pol = sol.policy(test_k)
        assert np.all(np.isfinite(pol)), f"Non-finite policy values found for beta={beta}"
        assert np.all(pol > 0), f"Non-positive policy values found for beta={beta}"

        true_pol = alpha * beta * (test_k ** alpha)
        rel_err = np.max(np.abs(pol - true_pol) / true_pol)
        assert rel_err < 1e-3, f"FEM relative policy error {rel_err:.2e} exceeded 1e-3 for beta={beta}"


# ===========================================================================
# 2. Extreme Parameter Regimes: Production Elasticity alpha in [0.01, 0.99]
# ===========================================================================

class TestAdversarialExtremeAlpha:
    """Stress-test solvers under extreme low and high capital elasticity."""

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.33, 0.5, 0.7, 0.85, 0.95, 0.99])
    def test_adv_extreme_alpha_collocation(self, alpha: float) -> None:
        """Collocation Euler projection solves stably across extreme alpha in [0.01, 0.99]."""
        beta = 0.96
        k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
        domain = (0.2 * k_ss, 2.0 * k_ss)

        prob = CollocationProblem(
            domain=domain,
            orders=8,
            beta=beta,
            params={"alpha": alpha, "delta": 1.0},
        )
        sol = prob.solve()

        assert sol.converged, f"Collocation failed to converge for alpha={alpha}"
        assert sol.residual_norm < 1e-6, f"Residual norm {sol.residual_norm} too high for alpha={alpha}"

        test_k = np.linspace(domain[0], domain[1], 500)
        pol = sol.policy(test_k)
        assert np.all(np.isfinite(pol))
        assert np.all(pol > 0)

        true_pol = alpha * beta * (test_k ** alpha)
        rel_err = np.max(np.abs(pol - true_pol) / true_pol)
        assert rel_err < 1e-3, f"Relative policy error {rel_err:.2e} exceeded 1e-3 for alpha={alpha}"

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.33, 0.5, 0.7, 0.85, 0.95, 0.99])
    def test_adv_extreme_alpha_fem(self, alpha: float) -> None:
        """FEM Galerkin solves stably across extreme alpha in [0.01, 0.99]."""
        beta = 0.96
        k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
        domain = (0.2 * k_ss, 2.0 * k_ss)

        prob = FEMProblem(
            domain=domain,
            elements=50,
            beta=beta,
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** alpha,
            params={"alpha": alpha, "gamma": 1.0},
        )
        sol = prob.solve()

        assert sol.converged, f"FEM failed to converge for alpha={alpha}"
        assert sol.residual_norm < 1e-6, f"Residual norm {sol.residual_norm} too high for alpha={alpha}"

        test_k = np.linspace(domain[0], domain[1], 500)
        pol = sol.policy(test_k)
        assert np.all(np.isfinite(pol))
        assert np.all(pol > 0)

        true_pol = alpha * beta * (test_k ** alpha)
        rel_err = np.max(np.abs(pol - true_pol) / true_pol)
        assert rel_err < 1e-3, f"FEM relative policy error {rel_err:.2e} exceeded 1e-3 for alpha={alpha}"

    @pytest.mark.parametrize(
        "beta,alpha",
        [
            (0.01, 0.05),
            (0.01, 0.95),
            (0.999, 0.05),
            (0.999, 0.95),
            (0.01, 0.99),
        ],
    )
    def test_adv_joint_extreme_beta_alpha_corners(self, beta: float, alpha: float) -> None:
        """Corner parameter combinations of extreme beta and alpha on domain in [10^-3, 10^3] converge."""
        domain = (0.001, 10.0)

        col_prob = CollocationProblem(domain=domain, orders=8, beta=beta, params={"alpha": alpha, "delta": 1.0})
        col_sol = col_prob.solve()

        fem_prob = FEMProblem(
            domain=domain,
            elements=50,
            beta=beta,
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** alpha,
            params={"alpha": alpha, "gamma": 1.0},
        )
        fem_sol = fem_prob.solve()

        assert col_sol.converged
        assert fem_sol.converged

        test_k = np.linspace(domain[0], domain[1], 200)
        col_pol = col_sol.policy(test_k)
        fem_pol = fem_sol.policy(test_k)

        assert np.all(np.isfinite(col_pol))
        assert np.all(np.isfinite(fem_pol))
        assert np.all(col_pol > 0)
        assert np.all(fem_pol > 0)


# ===========================================================================
# 3. Wide and Asymmetric State Space Boundaries: k in [10^-3, 10^3]
# ===========================================================================

class TestAdversarialWideDomainAndMultiScale:
    """Stress-test domain spanning 6 orders of magnitude [10^-3, 10^3]."""

    def test_adv_wide_domain_collocation_conditioning_and_stability(self) -> None:
        """Chebyshev collocation basis on [10^-3, 10^3] remains well-conditioned."""
        domain = (1e-3, 1e3)
        for N in [8, 15, 25]:
            basis = CollocationBasis(domain=domain, orders=N)
            nodes = basis.nodes()
            Phi = basis.evaluate(nodes)
            cond = np.linalg.cond(Phi)

            assert cond < 5.0, f"Chebyshev basis condition number {cond:.2f} exceeded 5.0 for N={N}"
            assert np.all(np.isfinite(Phi)), f"Non-finite basis matrix entries at N={N}"

            prob = CollocationProblem(
                domain=domain, orders=N, beta=0.96, params={"alpha": 0.36, "delta": 1.0}
            )
            sol = prob.solve()
            assert sol.converged
            assert np.all(np.isfinite(sol.coefficients))

            test_k = np.geomspace(domain[0], domain[1], 500)
            pol = sol.policy(test_k)
            assert np.all(np.isfinite(pol))
            assert np.all(pol > 0)

    def test_adv_wide_domain_fem_geometric_mesh(self) -> None:
        """FEM with geometric multi-scale mesh achieves < 1e-3 relative error on [10^-3, 10^3]."""
        domain = (1e-3, 1e3)
        alpha = 0.36
        beta = 0.96

        # Geometric mesh distributing elements across 6 orders of magnitude
        log_nodes = np.geomspace(domain[0], domain[1], 151)
        mesh_geom = FEMMesh(domain=domain, elements=150, nodes=log_nodes)

        fem_prob = FEMProblem(
            domain=domain,
            elements=150,
            beta=beta,
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** alpha,
            params={"alpha": alpha, "gamma": 1.0},
            options={"mesh": mesh_geom},
        )
        fem_sol = fem_prob.solve()

        assert fem_sol.converged
        assert fem_sol.residual_norm < 1e-6

        # Out-of-sample test on 1,000 log-spaced continuous points
        test_k = np.geomspace(domain[0], domain[1], 1000)
        true_pol = alpha * beta * (test_k ** alpha)
        fem_pol = fem_sol.policy(test_k)

        assert np.all(np.isfinite(fem_pol))
        assert np.all(fem_pol > 0)

        rel_err = np.max(np.abs(fem_pol - true_pol) / true_pol)
        assert rel_err < 1e-3, f"FEM geometric mesh error {rel_err:.2e} exceeded 1e-3 on [1e-3, 1e3]"

    def test_adv_wide_domain_asymmetric_bounds(self) -> None:
        """Highly asymmetric domain [0.001, 50.0] preserves monotonicity and zero violations."""
        domain = (0.001, 50.0)
        alpha = 0.36
        beta = 0.96

        col_prob = CollocationProblem(domain=domain, orders=12, beta=beta, params={"alpha": alpha, "delta": 1.0})
        col_sol = col_prob.solve()

        mesh_clustered = FEMMesh.create_clustered(domain=domain, elements=80, clustering=2.0)
        fem_prob = FEMProblem(
            domain=domain,
            elements=80,
            beta=beta,
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** alpha,
            params={"alpha": alpha, "gamma": 1.0},
            options={"mesh": mesh_clustered},
        )
        fem_sol = fem_prob.solve()

        test_k = np.linspace(domain[0], domain[1], 1000)
        col_pol = col_sol.policy(test_k)
        fem_pol = fem_sol.policy(test_k)

        assert np.all(np.diff(fem_pol) > 0), "FEM policy is not strictly increasing on asymmetric domain"
        assert np.all(col_pol > domain[0])
        assert np.all(fem_pol > domain[0])


# ===========================================================================
# 4. High Degree Polynomials (N >= 25) and Ultra-Fine Meshes (E >= 150)
# ===========================================================================

class TestAdversarialHighDegreeAndFineMesh:
    """Stress-test numerical condition numbers and matrix stability at scale."""

    @pytest.mark.parametrize("N", [25, 35, 50, 75, 100])
    def test_adv_chebyshev_basis_conditioning_high_orders(self, N: int) -> None:
        """Chebyshev basis matrix condition number remains bounded (< 3.0) up to N=100."""
        basis = CollocationBasis(domain=(0.1, 2.0), orders=N)
        nodes = basis.nodes()
        Phi = basis.evaluate(nodes)
        cond = np.linalg.cond(Phi)

        assert cond < 3.0, f"Condition number {cond:.2f} too high for order N={N}"
        assert np.all(np.isfinite(Phi)), f"Non-finite entries in Phi for N={N}"

        # Interpolation exactness on smooth test function
        y_exact = np.sin(nodes[:, 0])
        c = basis.fit(y_exact)
        dense_k = np.linspace(0.1, 2.0, 500)
        y_interp = basis.interpolate(c, dense_k)
        interp_err = np.max(np.abs(y_interp - np.sin(dense_k)))
        assert interp_err < 1e-12, f"Interpolation error {interp_err:.2e} too high for N={N}"

    @pytest.mark.parametrize("E", [150, 250, 350, 500])
    def test_adv_fem_mesh_fine_resolution_partition_of_unity(self, E: int) -> None:
        """Fine FEM meshes up to E=500 satisfy partition of unity to machine precision."""
        mesh = FEMMesh(domain=(0.1, 2.0), elements=E)
        dense_k = np.linspace(0.1, 2.0, 1000)
        Phi = mesh.evaluate_basis(dense_k)

        assert np.all(np.isfinite(Phi))
        pou_err = np.max(np.abs(np.sum(Phi, axis=1) - 1.0))
        assert pou_err < 1e-14, f"Partition of unity violated (error: {pou_err}) for E={E}"

    def test_adv_2d_high_order_tensor_collocation(self) -> None:
        """2D Chebyshev collocation basis with (N1, N2) = (15, 15) remains well-conditioned."""
        domain = ((0.1, 2.0), (0.5, 3.0))
        orders = (15, 15)
        basis = CollocationBasis(domain=domain, orders=orders)
        assert basis.n_nodes == 256

        nodes = basis.nodes()
        Phi = basis.evaluate(nodes)
        cond = np.linalg.cond(Phi)
        assert cond < 10.0, f"2D Chebyshev condition number {cond:.2f} exceeded 10.0"
        assert np.all(np.isfinite(Phi))

        # Test partial derivatives along both dimensions
        dPhi_0 = basis.derivative(nodes, dim=0)
        dPhi_1 = basis.derivative(nodes, dim=1)
        assert np.all(np.isfinite(dPhi_0))
        assert np.all(np.isfinite(dPhi_1))

    def test_adv_2d_fine_mesh_fem(self) -> None:
        """2D FEM tensor mesh with (25, 25) elements (676 nodes) satisfies partition of unity."""
        domain = ((0.1, 2.0), (0.5, 3.0))
        mesh = FEMMesh(domain=domain, elements=(25, 25))
        assert mesh.n_nodes == 26 * 26

        pts = np.column_stack([
            np.linspace(0.1, 2.0, 200),
            np.linspace(0.5, 3.0, 200),
        ])
        Phi = mesh.evaluate_basis(pts)
        assert np.all(np.isfinite(Phi))
        pou_err = np.max(np.abs(np.sum(Phi, axis=1) - 1.0))
        assert pou_err < 1e-14, f"2D FEM partition of unity error {pou_err:.2e} exceeded 1e-14"


# ===========================================================================
# 5. Adversarial Kink & Borrowing Constraint Tests: FEM vs Global Chebyshev
# ===========================================================================

class TestAdversarialKinkBorrowingConstraints:
    """Adversarial stress-testing of policy kinks under tight borrowing constraints."""

    def test_adv_kink_gibbs_ringing_and_boundary_violations(self) -> None:
        """Compare FEM exact node placement against Chebyshev Gibbs oscillations under kinks.

        Under a tight borrowing constraint k' >= k_bar:
        - Global Chebyshev polynomials suffer from Gibbs oscillations and severe boundary
          violations (g(k) < k_bar) that persist even as polynomial degree N increases.
        - FEM with an exact mesh node placed at the kink completely eliminates Gibbs ringing
          and achieves zero boundary violations (<= 1e-14).
        """
        alpha = 0.36
        beta = 0.96
        k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
        k_min, k_max = 0.1 * k_ss, 2.0 * k_ss
        domain = (k_min, k_max)

        # Constraint binds heavily over lower 50% of domain: k <= k_star
        k_star = 0.5 * k_ss
        k_bar = alpha * beta * (k_star ** alpha)

        def true_constrained_policy(k: np.ndarray) -> np.ndarray:
            return np.maximum(k_bar, alpha * beta * (k ** alpha))

        dense_k = np.linspace(k_min, k_max, 3000)
        constrained_mask = dense_k <= k_star

        # 1. Evaluate Chebyshev polynomials across degrees N in [6, 12, 20, 30]
        cheb_violations = []
        cheb_ringings = []
        for N in [6, 12, 20, 30]:
            basis = CollocationBasis(domain=domain, orders=N)
            nodes = basis.nodes()
            y_nodes = true_constrained_policy(nodes[:, 0])
            c = basis.fit(y_nodes)
            g_cheb = basis.interpolate(c, dense_k)

            # Boundary violation: depth where g(k) < k_bar
            violation = np.max(np.maximum(0.0, k_bar - g_cheb))
            # Gibbs ringing on flat constrained interval [k_min, k_star]
            ringing = np.max(np.abs(g_cheb[constrained_mask] - k_bar))

            cheb_violations.append(violation)
            cheb_ringings.append(ringing)

        # Chebyshev must demonstrate non-zero boundary violation and non-vanishing Gibbs ringing
        assert np.max(cheb_violations) > 1e-4, "Chebyshev should exhibit non-negligible boundary violations"
        assert np.max(cheb_ringings) > 5e-4, "Chebyshev should exhibit non-negligible Gibbs oscillations"

        # 2. Evaluate FEM with exact node placement at k_star across E in [20, 50, 100]
        for E in [20, 50, 100]:
            mesh_kink = FEMMesh.from_kinks(domain=domain, n_elements=E, kinks=[k_star])
            assert np.any(np.isclose(mesh_kink.nodes, k_star, atol=1e-12)), "k_star must be an exact node"

            y_nodes = true_constrained_policy(mesh_kink.nodes)
            Phi = mesh_kink.evaluate_basis(dense_k)
            g_fem = Phi @ y_nodes

            max_fem_violation = np.max(np.maximum(0.0, k_bar - g_fem))
            fem_ringing = np.max(np.abs(g_fem[constrained_mask] - k_bar))

            # FEM exact node placement eliminates Gibbs ringing to machine precision
            assert max_fem_violation < 1e-14, (
                f"FEM boundary violation {max_fem_violation:.2e} exceeded machine precision for E={E}"
            )
            assert fem_ringing < 1e-14, (
                f"FEM Gibbs ringing {fem_ringing:.2e} exceeded machine precision for E={E}"
            )

    def test_adv_fem_borrowing_constraint_galerkin_solver(self) -> None:
        """FEM Galerkin solver with borrowing_constraint yields zero boundary violations."""
        alpha = 0.36
        beta = 0.96
        k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
        domain = (0.1 * k_ss, 2.0 * k_ss)

        k_star = 0.5 * k_ss
        k_bar = alpha * beta * (k_star ** alpha)

        fem_prob = FEMProblem(
            domain=domain,
            elements=50,
            beta=beta,
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** alpha,
            params={"alpha": alpha, "gamma": 1.0},
            borrowing_constraint=k_bar,
            options={"kinks": [k_star]},
        )
        fem_sol = fem_prob.solve()

        assert fem_sol.converged
        assert fem_sol.residual_norm < 1e-6

        # Continuous test on 2,000 evaluation points
        dense_k = np.linspace(domain[0], domain[1], 2000)
        g_fem = fem_sol.policy(dense_k)

        # 1. Zero boundary violations: g(k) >= k_bar everywhere
        violations = np.maximum(0.0, k_bar - g_fem)
        assert np.max(violations) == 0.0, "FEM policy violated borrowing lower bound"

        # 2. Constrained flat region test: g(k) == k_bar on [k_min, k_star]
        constr_mask = dense_k <= k_star
        assert np.max(np.abs(g_fem[constr_mask] - k_bar)) < 1e-6

        # 3. Unconstrained region test: relative error vs unconstrained analytical policy < 5e-4
        unconstr_mask = dense_k > k_star
        true_unconstr = alpha * beta * (dense_k[unconstr_mask] ** alpha)
        rel_err = np.max(np.abs(g_fem[unconstr_mask] - true_unconstr) / true_unconstr)
        assert rel_err < 5e-4, f"Unconstrained region relative error {rel_err:.2e} exceeded 5e-4"

    def test_adv_fem_borrowing_constraint_collocation_solver(self) -> None:
        """FEM nodal collocation solver with borrowing_constraint yields zero boundary violations."""
        alpha = 0.36
        beta = 0.96
        k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
        domain = (0.1 * k_ss, 2.0 * k_ss)

        k_star = 0.4 * k_ss
        k_bar = alpha * beta * (k_star ** alpha)

        fem_prob = FEMProblem(
            domain=domain,
            elements=60,
            projection="collocation",
            beta=beta,
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** alpha,
            params={"alpha": alpha, "gamma": 1.0},
            borrowing_constraint=k_bar,
            options={"kinks": [k_star]},
        )
        fem_sol = fem_prob.solve()

        assert fem_sol.converged
        dense_k = np.linspace(domain[0], domain[1], 2000)
        g_fem = fem_sol.policy(dense_k)

        violations = np.maximum(0.0, k_bar - g_fem)
        assert np.max(violations) == 0.0, "FEM collocation policy violated borrowing constraint"

    def test_adv_fem_kink_near_domain_boundaries(self) -> None:
        """FEM mesh handles kinks placed close to domain boundaries without degenerate elements."""
        domain = (0.01, 1.0)
        # Kinks very close to lower and upper boundary
        kinks = [0.015, 0.985]
        mesh = FEMMesh.from_kinks(domain=domain, n_elements=40, kinks=kinks)

        assert mesh.n_elements >= 40
        assert np.all(np.diff(mesh.nodes) > 0), "Mesh nodes must be strictly monotonic"
        # Confirm kinks are exact nodes
        for k in kinks:
            assert np.any(np.isclose(mesh.nodes, k, atol=1e-12))


# ===========================================================================
# 6. Boundary Violations, Numerical Stability & Zero NaN/Inf
# ===========================================================================

class TestAdversarialBoundaryViolationsAndNumericalStability:
    """Stress-test numerical stability, finite bounds, and boundary preservation."""

    def test_adv_boundary_invariance_dense_5000_points(self) -> None:
        """Dense evaluation across 5,000 points strictly satisfies physical resource limits."""
        alpha = 0.36
        beta = 0.96
        k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
        domain = (0.2 * k_ss, 2.5 * k_ss)

        prob = CollocationProblem(domain=domain, orders=10, beta=beta, params={"alpha": alpha, "delta": 1.0})
        sol = prob.solve()

        dense_k = np.linspace(domain[0], domain[1], 5000)
        pol = sol.policy(dense_k)

        # 1. Zero NaNs or Infs
        assert np.all(np.isfinite(pol))

        # 2. Positivity of capital: g(k) > 0
        assert np.all(pol > 0)

        # 3. Feasibility of consumption: c(k) = k^alpha - g(k) > 0
        f_k = dense_k ** alpha
        c = f_k - pol
        assert np.all(c > 0), "Consumption became non-positive at some continuous point"

    def test_adv_continuous_euler_residuals_bounded_across_dense_grid(self) -> None:
        """Euler equation residuals evaluated on 2,000 points remain strictly bounded (< 1e-4)."""
        alpha = 0.36
        beta = 0.96
        k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
        domain = (0.3 * k_ss, 1.8 * k_ss)

        fem_prob = FEMProblem(
            domain=domain,
            elements=100,
            beta=beta,
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** alpha,
            params={"alpha": alpha, "gamma": 1.0},
        )
        fem_sol = fem_prob.solve()

        dense_k = np.linspace(domain[0], domain[1], 2000)
        res = fem_sol.euler_residual(dense_k)

        assert np.all(np.isfinite(res)), "Non-finite Euler residuals detected"
        max_res = np.max(np.abs(res))
        assert max_res < 1e-4, f"Continuous Euler residual {max_res:.2e} exceeded 1e-4"

    def test_adv_multi_backend_stress_parity(self) -> None:
        """NumPy, Numba, and MLX produce consistent solutions under stressful parameters."""
        domain = (0.05, 0.5)
        alpha = 0.85
        beta = 0.995

        solutions = {}
        for b in ["numpy", "numba", "mlx"]:
            if not bk.backend_available(b):
                continue
            prob = CollocationProblem(
                domain=domain, orders=10, beta=beta, params={"alpha": alpha, "delta": 1.0}
            )
            sol = prob.solve(backend=b)
            assert sol.converged
            solutions[b] = sol

        test_k = np.linspace(domain[0], domain[1], 300)
        numpy_pol = solutions["numpy"].policy(test_k)

        for b, sol in solutions.items():
            if b == "numpy":
                continue
            b_pol = sol.policy(test_k)
            max_diff = np.max(np.abs(b_pol - numpy_pol) / numpy_pol)
            assert max_diff < 1e-4, f"Backend '{b}' diverged from NumPy (rel diff: {max_diff:.2e})"

    def test_adv_defensive_validation_degenerate_inputs(self) -> None:
        """Defensive validation rejects non-finite, inverted, or illegal inputs immediately."""
        # Non-finite domain
        with pytest.raises(ValueError):
            CollocationBasis(domain=(np.nan, 2.0), orders=5)
        with pytest.raises(ValueError):
            CollocationBasis(domain=(0.1, np.inf), orders=5)

        # Inverted domain
        with pytest.raises(ValueError):
            CollocationProblem(domain=(2.0, 0.1), orders=5, beta=0.96)
        with pytest.raises(ValueError):
            FEMProblem(domain=(2.0, 0.1), elements=20, beta=0.96, return_fn=lambda c: c)

        # Illegal discount factor
        with pytest.raises(ValueError):
            CollocationProblem(domain=(0.1, 2.0), orders=5, beta=0.0)
        with pytest.raises(ValueError):
            CollocationProblem(domain=(0.1, 2.0), orders=5, beta=1.0)
        with pytest.raises(ValueError):
            FEMProblem(domain=(0.1, 2.0), elements=20, beta=1.05, return_fn=lambda c: c)

        # Borrowing constraint below domain lower bound
        with pytest.raises(ValueError):
            FEMProblem(
                domain=(0.5, 2.0),
                elements=20,
                beta=0.96,
                return_fn=lambda c: c,
                borrowing_constraint=0.1,  # 0.1 < 0.5
            )
