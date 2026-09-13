"""Adversarial Empirical Verification Suite for Continuous Projection Solvers.

Challenger 2 Suite:
1. Ultra-dense out-of-sample Euler residual mapping (5,000 to 10,000 continuous points) for Collocation and FEM.
2. Exact boundary evaluation and out-of-domain extrapolation defense without NaN/Inf.
3. Multi-backend parity and acceleration stress benchmarks (NumPy vs Numba vs MLX).
4. Constrained growth kink defense and wide-domain high-curvature stress.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro import _backend as bk
from puremacro.vfi.collocation import CollocationBasis, CollocationProblem, CollocationSolution
from puremacro.vfi.fem import FEMMesh, FEMProblem, FEMSolution


@pytest.fixture
def canonical_bm():
    """Canonical Brock-Mirman neoclassical growth parameters."""
    alpha = 0.36
    beta = 0.96
    k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
    domain = (0.5 * k_ss, 1.5 * k_ss)
    
    def g_star(k):
        return alpha * beta * (np.asarray(k, dtype=np.float64) ** alpha)
        
    return {
        "alpha": alpha,
        "beta": beta,
        "k_ss": k_ss,
        "domain": domain,
        "g_star": g_star,
    }


def compute_euler_residual(policy_fn, k_grid, alpha=0.36, beta=0.96):
    """Compute continuous Euler residual: E(k) = 1 - beta * (u'(c') / u'(c)) * f'(k')."""
    k = np.asarray(k_grid, dtype=np.float64)
    kp = policy_fn(k)
    kpp = policy_fn(kp)
    c = np.maximum(k**alpha - kp, 1e-12)
    cp = np.maximum(kp**alpha - kpp, 1e-12)
    fkp = alpha * (kp ** (alpha - 1.0))
    return 1.0 - beta * (c / cp) * fkp


class TestUltraDenseEulerResiduals:
    """1. High-Density Out-of-Sample Euler Residual Mapping (5,000 to 10,000 continuous points)."""

    @pytest.mark.parametrize("n_points", [5000, 10000])
    def test_collocation_dense_euler_residuals(self, canonical_bm, n_points):
        """Verify Collocation achieves max |E(k)| < 1e-4 across 5,000 and 10,000 continuous points."""
        bm = canonical_bm
        prob = CollocationProblem(
            domain=bm["domain"],
            orders=8,
            method="euler",
            params={"alpha": bm["alpha"], "delta": 1.0},
            beta=bm["beta"],
        )
        sol = prob.solve(backend="numpy")
        assert sol.converged, "Collocation failed to converge"

        eval_grid = np.linspace(bm["domain"][0], bm["domain"][1], n_points)
        euler_res = compute_euler_residual(sol.policy, eval_grid, bm["alpha"], bm["beta"])
        max_res = np.max(np.abs(euler_res))

        assert max_res < 1e-4, f"Collocation max Euler residual {max_res:.2e} >= 1e-4 on {n_points} points"
        # Also check relative policy error against analytical benchmark
        g_true = bm["g_star"](eval_grid)
        rel_err = np.max(np.abs(sol.policy(eval_grid) - g_true) / g_true)
        assert rel_err < 1e-4, f"Collocation relative policy error {rel_err:.2e} >= 1e-4"

    @pytest.mark.parametrize("n_points", [5000, 10000])
    def test_fem_dense_euler_residuals(self, canonical_bm, n_points):
        """Verify FEM Galerkin achieves max |E(k)| < 1e-4 across 5,000 and 10,000 continuous points."""
        bm = canonical_bm
        prob = FEMProblem(
            domain=bm["domain"],
            elements=60,
            method="euler",
            projection="galerkin",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"], "gamma": 1.0},
        )
        sol = prob.solve(backend="numpy")
        assert sol.converged, "FEM failed to converge"

        eval_grid = np.linspace(bm["domain"][0], bm["domain"][1], n_points)
        euler_res = compute_euler_residual(sol.policy, eval_grid, bm["alpha"], bm["beta"])
        max_res = np.max(np.abs(euler_res))

        assert max_res < 1e-4, f"FEM max Euler residual {max_res:.2e} >= 1e-4 on {n_points} points"
        # Also check relative policy error
        g_true = bm["g_star"](eval_grid)
        rel_err = np.max(np.abs(sol.policy(eval_grid) - g_true) / g_true)
        assert rel_err < 1e-4, f"FEM relative policy error {rel_err:.2e} >= 1e-4"

    def test_clustered_boundary_grid_euler_residuals(self, canonical_bm):
        """Verify Euler residuals remain < 1e-4 on grids heavily clustered near domain boundaries."""
        bm = canonical_bm
        prob_c = CollocationProblem(
            domain=bm["domain"],
            orders=8,
            method="euler",
            params={"alpha": bm["alpha"], "delta": 1.0},
            beta=bm["beta"],
        )
        sol_c = prob_c.solve(backend="numpy")

        prob_f = FEMProblem(
            domain=bm["domain"],
            elements=60,
            method="euler",
            projection="galerkin",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"]},
        )
        sol_f = prob_f.solve(backend="numpy")

        # Chebyshev-Gauss-Lobatto evaluation grid of 5,000 points (concentrates near boundaries)
        theta = np.linspace(0.0, np.pi, 5000)
        x_lobatto = -np.cos(theta)
        k_min, k_max = bm["domain"]
        grid_boundary_clustered = 0.5 * (k_min + k_max) + 0.5 * (k_max - k_min) * x_lobatto

        res_c = np.max(np.abs(compute_euler_residual(sol_c.policy, grid_boundary_clustered, bm["alpha"], bm["beta"])))
        res_f = np.max(np.abs(compute_euler_residual(sol_f.policy, grid_boundary_clustered, bm["alpha"], bm["beta"])))

        assert res_c < 1e-4, f"Collocation boundary-clustered Euler residual {res_c:.2e} >= 1e-4"
        assert res_f < 1e-4, f"FEM boundary-clustered Euler residual {res_f:.2e} >= 1e-4"

    @pytest.mark.parametrize("alpha,beta", [
        (0.15, 0.80),
        (0.15, 0.99),
        (0.40, 0.98),
        (0.70, 0.90),
    ])
    def test_parameter_stress_dense_residuals(self, alpha, beta):
        """Stress test Euler residuals across non-standard economic parameters on 5,000 points."""
        k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
        domain = (0.5 * k_ss, 1.5 * k_ss)
        grid_5k = np.linspace(domain[0], domain[1], 5000)

        sol_c = CollocationProblem(
            domain=domain, orders=8, method="euler",
            params={"alpha": alpha, "delta": 1.0}, beta=beta
        ).solve()
        res_c = np.max(np.abs(compute_euler_residual(sol_c.policy, grid_5k, alpha, beta)))
        assert res_c < 1e-4, f"Collocation residual {res_c:.2e} >= 1e-4 for alpha={alpha}, beta={beta}"

        sol_f = FEMProblem(
            domain=domain, elements=60, method="euler", projection="galerkin",
            return_fn=lambda c: np.log(c), transition_fn=lambda k: k ** alpha,
            beta=beta, params={"alpha": alpha}
        ).solve()
        res_f = np.max(np.abs(compute_euler_residual(sol_f.policy, grid_5k, alpha, beta)))
        assert res_f < 1e-4, f"FEM residual {res_f:.2e} >= 1e-4 for alpha={alpha}, beta={beta}"


class TestBoundaryAndExtrapolationDefense:
    """2. Boundary Evaluation & Extrapolation Safety."""

    def test_exact_domain_endpoints_evaluation(self, canonical_bm):
        """Evaluate strictly at domain endpoints k = k_min and k = k_max."""
        bm = canonical_bm
        k_min, k_max = bm["domain"]

        sol_c = CollocationProblem(domain=bm["domain"], orders=8, beta=bm["beta"], params={"alpha": bm["alpha"], "delta": 1.0}).solve()
        sol_f = FEMProblem(domain=bm["domain"], elements=60, return_fn=lambda c: np.log(c), transition_fn=lambda k: k ** bm["alpha"], beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()

        for endpt in [k_min, k_max]:
            p_c = sol_c.policy(endpt)
            p_f = sol_f.policy(endpt)
            assert np.isfinite(p_c) and not np.isnan(p_c), f"Collocation NaN/Inf at endpoint {endpt}"
            assert np.isfinite(p_f) and not np.isnan(p_f), f"FEM NaN/Inf at endpoint {endpt}"
            assert p_c > 0.0 and p_f > 0.0, f"Non-positive capital policy at endpoint {endpt}"

            # Endpoint policy matches true analytical policy within 1e-4
            g_exact = bm["g_star"](endpt)
            assert abs(p_c - g_exact) / g_exact < 1e-4
            assert abs(p_f - g_exact) / g_exact < 1e-4

    @pytest.mark.parametrize("eps", [1e-8, 1e-6, 1e-4, 1e-2, 0.05, 0.1])
    def test_out_of_domain_extrapolation_clamping(self, canonical_bm, eps):
        """Evaluate query points outside domain [k_min - eps, k_max + eps] to confirm defense."""
        bm = canonical_bm
        k_min, k_max = bm["domain"]
        k_below = k_min - eps
        k_above = k_max + eps

        sol_c = CollocationProblem(domain=bm["domain"], orders=8, beta=bm["beta"], params={"alpha": bm["alpha"], "delta": 1.0}).solve()
        sol_f = FEMProblem(domain=bm["domain"], elements=60, return_fn=lambda c: np.log(c), transition_fn=lambda k: k ** bm["alpha"], beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()

        # Check Collocation policy & basis
        p_c_below = sol_c.policy(k_below)
        p_c_above = sol_c.policy(k_above)
        phi_c = sol_c.basis.evaluate(np.array([k_below, k_above]))
        assert np.all(np.isfinite(phi_c))
        assert np.isfinite(p_c_below) and np.isfinite(p_c_above)
        assert not np.isnan(p_c_below) and not np.isnan(p_c_above)

        # Clamping behavior: out-of-bounds evaluation clamped to domain boundaries
        p_c_min = sol_c.policy(k_min)
        p_c_max = sol_c.policy(k_max)
        assert np.isclose(p_c_below, p_c_min, atol=1e-12)
        assert np.isclose(p_c_above, p_c_max, atol=1e-12)

        # Check FEM policy & basis
        p_f_below = sol_f.policy(k_below)
        p_f_above = sol_f.policy(k_above)
        phi_f = sol_f.mesh.evaluate_basis(np.array([k_below, k_above]))
        assert np.all(np.isfinite(phi_f))
        assert np.isfinite(p_f_below) and np.isfinite(p_f_above)
        assert not np.isnan(p_f_below) and not np.isnan(p_f_above)

        # FEM clamping behavior
        p_f_min = sol_f.policy(k_min)
        p_f_max = sol_f.policy(k_max)
        assert np.isclose(p_f_below, p_f_min, atol=1e-12)
        assert np.isclose(p_f_above, p_f_max, atol=1e-12)

    def test_continuous_span_crossing_boundaries(self, canonical_bm):
        """Verify evaluation on a 5,000-point grid spanning continuously across both boundaries."""
        bm = canonical_bm
        k_min, k_max = bm["domain"]
        grid_extended = np.linspace(k_min - 0.05, k_max + 0.05, 5000)

        sol_c = CollocationProblem(domain=bm["domain"], orders=8, beta=bm["beta"], params={"alpha": bm["alpha"], "delta": 1.0}).solve()
        sol_f = FEMProblem(domain=bm["domain"], elements=60, return_fn=lambda c: np.log(c), transition_fn=lambda k: k ** bm["alpha"], beta=bm["beta"], params={"alpha": bm["alpha"]}).solve()

        pol_c = sol_c.policy(grid_extended)
        pol_f = sol_f.policy(grid_extended)

        assert np.all(np.isfinite(pol_c))
        assert np.all(np.isfinite(pol_f))
        assert not np.any(np.isnan(pol_c))
        assert not np.any(np.isnan(pol_f))

        # Check that interior values remain strictly monotonic
        interior_mask = (grid_extended >= k_min) & (grid_extended <= k_max)
        assert np.all(np.diff(pol_c[interior_mask]) > 0.0)
        assert np.all(np.diff(pol_f[interior_mask]) > 0.0)


class TestMultiBackendStressParity:
    """3. Multi-Backend Numerical Parity (NumPy vs Numba vs MLX)."""

    def test_collocation_dense_parity_numpy_numba_mlx(self, canonical_bm):
        """Verify Collocation solutions across NumPy, Numba, and MLX agree to < 1e-4 on 10,000 points."""
        bm = canonical_bm
        prob = CollocationProblem(
            domain=bm["domain"],
            orders=8,
            method="euler",
            params={"alpha": bm["alpha"], "delta": 1.0},
            beta=bm["beta"],
        )
        sol_np = prob.solve(backend="numpy")
        assert sol_np.converged

        eval_grid_10k = np.linspace(bm["domain"][0], bm["domain"][1], 10000)
        p_np = sol_np.policy(eval_grid_10k)

        for b_name in ("numba", "mlx"):
            if bk.backend_available(b_name):
                sol_b = prob.solve(backend=b_name)
                assert sol_b.converged
                p_b = sol_b.policy(eval_grid_10k)

                max_rel_diff = np.max(np.abs(p_b - p_np) / np.abs(p_np))
                assert max_rel_diff < 1e-4, f"Collocation NumPy vs {b_name} relative diff {max_rel_diff:.2e} >= 1e-4"

    def test_fem_dense_parity_numpy_numba_mlx(self, canonical_bm):
        """Verify FEM solutions across NumPy, Numba, and MLX agree to < 1e-4 on 10,000 points."""
        bm = canonical_bm
        prob = FEMProblem(
            domain=bm["domain"],
            elements=60,
            method="euler",
            projection="galerkin",
            return_fn=lambda c: np.log(c),
            transition_fn=lambda k: k ** bm["alpha"],
            beta=bm["beta"],
            params={"alpha": bm["alpha"]},
        )
        sol_np = prob.solve(backend="numpy")
        assert sol_np.converged

        eval_grid_10k = np.linspace(bm["domain"][0], bm["domain"][1], 10000)
        p_np = sol_np.policy(eval_grid_10k)

        for b_name in ("numba", "mlx"):
            if bk.backend_available(b_name):
                sol_b = prob.solve(backend=b_name)
                assert sol_b.converged
                p_b = sol_b.policy(eval_grid_10k)

                max_rel_diff = np.max(np.abs(p_b - p_np) / np.abs(p_np))
                assert max_rel_diff < 1e-4, f"FEM NumPy vs {b_name} relative diff {max_rel_diff:.2e} >= 1e-4"

    def test_2d_basis_multi_backend_parity(self):
        """Verify 2D tensor product basis evaluation parity across NumPy and MLX on 5,000 points."""
        basis = CollocationBasis(domain=((0.1, 1.0), (0.8, 1.2)), orders=(6, 4))
        k_pts = np.linspace(0.1, 1.0, 100)
        z_pts = np.linspace(0.8, 1.2, 50)
        grid_2d = np.column_stack([np.repeat(k_pts, 50), np.tile(z_pts, 100)])

        phi_np = basis.evaluate(grid_2d, backend="numpy")
        assert phi_np.shape == (5000, 35)

        if bk.backend_available("mlx"):
            phi_mlx = basis.evaluate(grid_2d, backend="mlx")
            max_diff = np.max(np.abs(phi_mlx - phi_np))
            assert max_diff < 1e-5, f"2D MLX basis max difference {max_diff:.2e} >= 1e-5"
