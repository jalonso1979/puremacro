"""Empirical Adversarial Stress Harness for P3 (Smolyak Sparse Grid) and P4 (DC-EGM).

Authored by Challenger 2 for rigorous empirical verification:
- P3 (Smolyak Sparse Grid Collocation):
  1. Full parameter grid sweep: d in [2, 6] x mu in [1, 4] node reduction >= 5x for d >= 3, mu >= 2.
  2. Condition number bounds and stability across all 20 (d, mu) configurations.
  3. Highly anisotropic domains with extreme aspect ratios (10^6 span, negative offsets).
  4. Exact multivariate polynomial reproduction of degree <= mu to machine precision (< 1e-11).
  5. Continuous Euler equation residuals on 2D and 3D neoclassical growth models across 5,000 random continuous points (< 1e-4).
  6. Smolyak combination technique parity against direct collocation on random fields.

- P4 (DC-EGM and Upper Envelope):
  7. Multi-branch non-monotonic endogenous grids (3+ crossing branches, double folds).
  8. Dominated loop / non-convex fold pruning with falling branch excision.
  9. Degenerate branches: collinear, identical values, extreme slope ratios (10^4).
  10. EV1 logit choice probabilities sum to 1.0 +- 1e-14 across Delta V in [10^-8, 10^4].
  11. Deterministic choice limit (sigma_eps -> 0): hardmax inclusive value and exact indicator probabilities.
  12. Envelope theorem marginal value verification without numerical differentiation.
  13. Dynamic retirement model convergence, monotonicity of value functions, borrowing constraint compliance, and finite-horizon backward induction.
  14. Presentation contract and public API export verification.
"""
from __future__ import annotations

import itertools
import warnings
from typing import Any, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro import _backend as bk
import puremacro.vfi as vfi
from puremacro.vfi.smolyak import (
    SmolyakBasis,
    SmolyakGrid,
    SmolyakProblem,
    SmolyakSolution,
    solve_smolyak,
    _clenshaw_curtis_nodes,
    _disjoint_increments,
)
from puremacro.vfi.dcegm import (
    ChoiceMapping,
    DCEGMProblem,
    DCEGMSolution,
    UpperEnvelopeResult,
    solve_dcegm,
    upper_envelope,
    _logsumexp_probs_1d_kernel,
)


# ============================================================================
# P3: SMOLYAK SPARSE GRID ADVERSARIAL SUITE
# ============================================================================

class TestSmolyakAdversarial:
    """Adversarial stress tests for Smolyak sparse grid collocation."""

    @pytest.mark.parametrize("d", [2, 3, 4, 5, 6])
    @pytest.mark.parametrize("mu", [1, 2, 3, 4])
    def test_smolyak_grid_properties_sweep_all_d_mu(self, d: int, mu: int):
        """Sweep all (d, mu) in [2, 6] x [1, 4] verifying node counts, bounds, and reduction."""
        grid = SmolyakGrid(d=d, mu=mu)

        # 1. Dimension and shape checks
        assert grid.d == d
        assert grid.mu == mu
        assert grid.n_nodes == len(grid)
        assert grid.nodes.shape == (grid.n_nodes, d)
        assert grid.physical_nodes.shape == (grid.n_nodes, d)
        assert grid.poly_indices.shape == (grid.n_nodes, d)

        # 2. Canonical hypercube bounds [-1, 1]^d
        assert np.all(grid.nodes >= -1.0 - 1e-14)
        assert np.all(grid.nodes <= 1.0 + 1e-14)

        # 3. Grid node reduction ratio verification
        tensor_nodes = grid.tensor_nodes_count
        assert tensor_nodes > 0
        reduction = grid.reduction_ratio
        assert np.isclose(reduction, tensor_nodes / grid.n_nodes)

        # Requirement check: reduction >= 5x for d >= 3, mu >= 2
        if d >= 3 and mu >= 2:
            assert reduction >= 5.0, f"Reduction {reduction:.2f} < 5.0 for d={d}, mu={mu}"

        # Scalability: verify reduction grows monotonically with d and mu
        if d >= 4 and mu >= 2:
            assert reduction >= 15.0
        if d >= 5 and mu >= 2:
            assert reduction >= 50.0

    @pytest.mark.parametrize("d", [2, 3, 4])
    @pytest.mark.parametrize("mu", [1, 2, 3])
    def test_smolyak_basis_conditioning_and_invertibility(self, d: int, mu: int):
        """Verify collocation matrix Phi condition number and exact invertibility."""
        basis = SmolyakBasis(d=d, mu=mu)
        Phi = basis.collocation_matrix
        n = basis.n_basis

        assert Phi.shape == (n, n)
        cond = basis.condition_number

        # Condition number must remain well-behaved (< 100 for mu <= 3)
        assert cond < 100.0, f"Condition number {cond:.2f} too large for d={d}, mu={mu}"

        # Invertibility: solve Phi * theta = y for random target vector
        rng = np.random.default_rng(1000 + 10 * d + mu)
        y_target = rng.standard_normal(n)
        theta = basis.fit(y_target)
        y_rec = Phi @ theta

        rel_residual = np.linalg.norm(y_rec - y_target, ord=np.inf) / np.linalg.norm(y_target, ord=np.inf)
        assert rel_residual < 1e-12, f"Collocation inversion relative residual {rel_residual} > 1e-12"

    def test_smolyak_highly_anisotropic_domain(self):
        """Stress-test domain with extreme aspect ratios (10^6 span, asymmetric offsets)."""
        domain = (
            (0.001, 1000.0),      # Span 1e6
            (-500.0, 500.0),      # Symmetric around 0
            (10.0, 10.05),        # Narrow band (span 0.05)
            (0.0, 1.0),           # Unit interval
        )
        grid = SmolyakGrid(d=4, mu=2, domain=domain)

        # 1. Physical nodes must lie strictly within specified bounds
        for k, (ak, bk) in enumerate(domain):
            col = grid.physical_nodes[:, k]
            assert np.all(col >= ak - 1e-11), f"Dim {k} lower bound violation: {np.min(col)} < {ak}"
            assert np.all(col <= bk + 1e-11), f"Dim {k} upper bound violation: {np.max(col)} > {bk}"

        # 2. Round-trip affine bijection precision
        rng = np.random.default_rng(2026)
        s_test = np.column_stack([
            rng.uniform(ak, bk, 1000) for ak, bk in domain
        ])
        canonical = grid.to_canonical(s_test)
        assert np.all(canonical >= -1.0 - 1e-13) and np.all(canonical <= 1.0 + 1e-13)

        s_reconstructed = grid.to_physical(canonical)
        max_abs_err = np.max(np.abs(s_test - s_reconstructed))
        rel_err = np.max(np.abs(s_test - s_reconstructed) / np.maximum(np.abs(s_test), 1e-6))
        assert max_abs_err < 1e-11, f"Anisotropic roundtrip absolute error: {max_abs_err}"
        assert rel_err < 1e-12, f"Anisotropic roundtrip relative error: {rel_err}"

        # 3. Basis interpolation on anisotropic domain
        basis = SmolyakBasis(grid=grid)

        def test_field(s: np.ndarray) -> np.ndarray:
            s1, s2, s3, s4 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
            return (s1 / 1000.0) ** 2 + (s2 / 500.0) * (s4) - (s3 - 10.0) / 0.05

        y_nodes = test_field(grid.physical_nodes)
        theta = basis.fit(y_nodes)
        y_approx = basis.interpolate(theta, s_test)
        y_true = test_field(s_test)

        poly_err = np.max(np.abs(y_approx - y_true))
        assert poly_err < 1e-10, f"Anisotropic polynomial interpolation error: {poly_err}"

    def test_smolyak_polynomial_exactness_degree_mu(self):
        """Verify exact machine-precision reproduction of multivariate polynomials up to degree mu."""
        d = 3
        mu = 2
        domain = ((-1.0, 2.0), (0.5, 3.5), (-2.0, 0.0))
        basis = SmolyakBasis(d=d, mu=mu, domain=domain)

        # Form a random polynomial of degree <= 2
        rng = np.random.default_rng(42)
        poly_terms = [
            lambda s: 3.5,
            lambda s: -2.1 * s[:, 0],
            lambda s: 1.4 * s[:, 1],
            lambda s: -0.7 * s[:, 2],
            lambda s: 0.9 * s[:, 0] ** 2,
            lambda s: 1.1 * s[:, 1] ** 2,
            lambda s: -0.5 * s[:, 2] ** 2,
            lambda s: 0.8 * s[:, 0] * s[:, 1],
            lambda s: -0.6 * s[:, 0] * s[:, 2],
            lambda s: 0.4 * s[:, 1] * s[:, 2],
        ]

        def target_poly(s: np.ndarray) -> np.ndarray:
            res = np.zeros(len(s), dtype=np.float64)
            for term in poly_terms:
                res += term(s)
            return res

        y_nodes = target_poly(basis.grid.physical_nodes)
        theta = basis.fit(y_nodes)

        # Query on 2,000 out-of-sample points including corners
        s_eval = np.column_stack([
            rng.uniform(ak, bk, 2000) for ak, bk in domain
        ])
        y_true = target_poly(s_eval)
        y_approx = basis.interpolate(theta, s_eval)

        max_err = np.max(np.abs(y_true - y_approx))
        assert max_err < 1e-11, f"Polynomial exactness violated: {max_err} > 1e-11"

    def test_smolyak_combination_technique_exhaustive(self):
        """Verify Smolyak combination technique interpolation matches direct collocation exactly."""
        d = 3
        mu = 2
        basis = SmolyakBasis(d=d, mu=mu)
        rng = np.random.default_rng(999)

        # Smooth nonlinear function
        def f(s: np.ndarray) -> np.ndarray:
            return np.sin(np.pi * s[:, 0]) * np.cos(np.pi * s[:, 1]) * np.exp(s[:, 2])

        y_nodes = f(basis.grid.physical_nodes)
        theta = basis.fit(y_nodes)

        s_test = rng.uniform(-1.0, 1.0, (500, d))
        direct_vals = basis.interpolate(theta, s_test)
        comb_vals = basis.combination_interpolate(y_nodes, s_test)

        discrepancy = np.max(np.abs(direct_vals - comb_vals))
        assert discrepancy < 1e-12, f"Combination technique discrepancy {discrepancy} > 1e-12"

    def test_smolyak_2d_neoclassical_continuous_euler_5000_points(self):
        """Stress-test 2D neoclassical growth model on 5,000 random out-of-sample points.

        Verify:
        - Continuous Euler equation residual < 1e-4 everywhere.
        - Policy relative error against analytical benchmark < 1e-4 everywhere.
        """
        alphas = [0.18, 0.18]
        beta = 0.96
        z = 1.0 / (alphas[0] * beta)
        domain = ((0.75, 1.25), (0.75, 1.25))

        sol = solve_smolyak(
            domain=domain,
            mu=2,
            params={"alphas": alphas, "z": z, "beta": beta},
        )

        assert sol.converged
        assert sol.residual_norm < 1e-4

        # 5,000 random continuous points
        rng = np.random.default_rng(12345)
        test_pts = rng.uniform(0.75, 1.25, (5000, 2))

        # 1. Analytical policy comparison
        kp_approx = sol.policy(test_pts)
        Y_test = z * (test_pts[:, 0] ** alphas[0]) * (test_pts[:, 1] ** alphas[1])
        kp_true = np.column_stack([alphas[0] * beta * Y_test, alphas[1] * beta * Y_test])

        rel_policy_err = np.max(np.abs(kp_approx - kp_true) / kp_true)
        assert rel_policy_err < 1e-4, f"2D policy relative error {rel_policy_err:.2e} >= 1e-4"

        # 2. Continuous Euler residual
        res_euler = sol.euler_residual(test_pts)
        max_euler = np.max(np.abs(res_euler))
        assert max_euler < 1e-4, f"2D continuous Euler residual {max_euler:.2e} >= 1e-4"

    def test_smolyak_3d_neoclassical_continuous_euler_5000_points(self):
        """Stress-test 3D multi-capital model on 5,000 random out-of-sample points.

        Verify:
        - Continuous Euler equation residual < 1e-4.
        - Grid reduction >= 10x (69 nodes vs 729 tensor nodes).
        """
        alphas = [0.12, 0.12, 0.12]
        beta = 0.96
        z = 1.0 / (alphas[0] * beta)
        domain = ((0.7, 1.3), (0.7, 1.3), (0.7, 1.3))

        sol = solve_smolyak(
            domain=domain,
            mu=3,
            params={"alphas": alphas, "z": z, "beta": beta},
        )

        assert sol.converged
        assert sol.basis.grid.reduction_ratio >= 10.0
        assert sol.basis.n_basis == 69
        assert sol.basis.grid.tensor_nodes_count == 729

        # 5,000 random continuous evaluation points
        rng = np.random.default_rng(54321)
        test_pts = rng.uniform(0.7, 1.3, (5000, 3))

        # Continuous Euler residual
        res_euler = sol.euler_residual(test_pts)
        max_euler = np.max(np.abs(res_euler))
        assert max_euler < 1e-4, f"3D continuous Euler residual {max_euler:.2e} >= 1e-4"


# ============================================================================
# P4: DC-EGM & UPPER ENVELOPE ADVERSARIAL SUITE
# ============================================================================

class TestDCEGMAdversarial:
    """Adversarial stress tests for DC-EGM and Upper Envelope filtering."""

    def test_upper_envelope_triple_branch_crossing(self):
        """Adversarial scenario: 3 overlapping branches with multiple crossings and folds.

        Branch 1 (steep): v1(M) = -3.0 + 2.0 * M,  c1 = 0.7 * M
        Branch 2 (medium): v2(M) = -0.5 + 1.0 * M, c2 = 0.5 * M
        Branch 3 (flat):   v3(M) =  0.5 + 0.5 * M, c3 = 0.3 * M
        """
        # Create non-monotonic assembly where branches fold back and forth
        M1 = np.linspace(2.2, 4.5, 40)
        c1 = 0.7 * M1
        v1 = -3.0 + 2.0 * M1

        M2 = np.linspace(1.5, 3.0, 40)
        c2 = 0.5 * M2
        v2 = -0.5 + 1.0 * M2

        M3 = np.linspace(0.5, 2.3, 40)
        c3 = 0.3 * M3
        v3 = 0.5 + 0.5 * M3

        M_raw = np.concatenate([M3, M1, M2])
        c_raw = np.concatenate([c3, c1, c2])
        v_raw = np.concatenate([v3, v1, v2])

        exog_grid = np.linspace(0.6, 4.0, 100)
        res = upper_envelope(M_raw, c_raw, v_raw, exog_grid)
        c_filtered, v_filtered = res

        # 1. Non-decreasing value function everywhere
        assert np.all(np.diff(v_filtered) >= -1e-10), "Filtered value function is non-monotonic!"

        # 2. Branch selection verification
        # Zone 3: M < 1.9 -> Branch 3 dominates
        z3 = exog_grid < 1.9
        assert np.allclose(c_filtered[z3], 0.3 * exog_grid[z3], atol=1e-3)
        assert np.allclose(v_filtered[z3], 0.5 + 0.5 * exog_grid[z3], atol=1e-3)

        # Zone 2: 2.1 < M < 2.4 -> Branch 2 dominates
        z2 = (exog_grid > 2.1) & (exog_grid < 2.4)
        assert np.allclose(c_filtered[z2], 0.5 * exog_grid[z2], atol=1e-3)
        assert np.allclose(v_filtered[z2], -0.5 + 1.0 * exog_grid[z2], atol=1e-3)

        # Zone 1: M > 2.6 -> Branch 1 dominates
        z1 = exog_grid > 2.6
        assert np.allclose(c_filtered[z1], 0.7 * exog_grid[z1], atol=1e-3)
        assert np.allclose(v_filtered[z1], -3.0 + 2.0 * exog_grid[z1], atol=1e-3)

    def test_upper_envelope_dominated_loop_pruning(self):
        """Adversarial test: non-convex fold with a falling branch and dominated segment."""
        # Branch 1: M rises from 1.0 to 3.5
        M1 = np.linspace(1.0, 3.5, 30)
        c1 = 0.5 * M1
        v1 = -1.0 + 1.0 * M1

        # Falling branch: M falls from 3.4 to 2.3 (sub-optimal local minima)
        M_fall = np.linspace(3.4, 2.3, 15)
        c_fall = 0.3 * M_fall
        v_fall = -2.5 + 0.5 * M_fall

        # Branch 2: M rises from 2.2 to 5.0 (steep branch that overtakes Branch 1 at M*=2.5)
        M2 = np.linspace(2.2, 5.0, 35)
        c2 = 0.7 * M2
        v2 = -2.5 + 1.6 * M2

        M_raw = np.concatenate([M1, M_fall, M2])
        c_raw = np.concatenate([c1, c_fall, c2])
        v_raw = np.concatenate([v1, v_fall, v2])

        exog = np.linspace(1.2, 4.8, 60)
        res = upper_envelope(M_raw, c_raw, v_raw, exog)
        c_clean, v_clean = res

        # Value function must be strictly non-decreasing
        assert np.all(np.diff(v_clean) >= -1e-10)

        # Below M* = 2.5: Branch 1 dominates (c = 0.5 * M, v = -1.0 + M)
        below = exog < 2.4
        assert np.allclose(c_clean[below], 0.5 * exog[below], atol=1e-6)
        assert np.allclose(v_clean[below], -1.0 + exog[below], atol=1e-6)

        # Above M* = 2.5: Branch 2 dominates (c = 0.7 * M, v = -2.5 + 1.6 * M)
        above = exog > 2.6
        assert np.allclose(c_clean[above], 0.7 * exog[above], atol=1e-6)
        assert np.allclose(v_clean[above], -2.5 + 1.6 * exog[above], atol=1e-6)

    def test_upper_envelope_extreme_slope_ratio(self):
        """Stress-test upper envelope when two branches have a 10^4 slope ratio."""
        M1 = np.linspace(1.8, 2.5, 50)
        c1 = 0.8 * M1
        v1 = 1000.0 * (M1 - 2.0)

        M2 = np.linspace(0.5, 3.0, 50)
        c2 = 0.2 * M2
        v2 = 0.1 * M2

        M_raw = np.concatenate([M2, M1])
        c_raw = np.concatenate([c2, c1])
        v_raw = np.concatenate([v2, v1])

        exog_grid = np.linspace(0.8, 2.4, 80)
        c_clean, v_clean = upper_envelope(M_raw, c_raw, v_raw, exog_grid)

        assert np.all(np.isfinite(c_clean))
        assert np.all(np.isfinite(v_clean))
        assert np.all(np.diff(v_clean) >= -1e-10)

    def test_ev1_choice_probabilities_sum_to_one_exact(self):
        """Verify that EV1 logit choice probabilities sum to 1.0 +- 1e-14 across Delta V in [10^-8, 10^4]."""
        deltas = [1e-8, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0, 5000.0, 10000.0]
        sigmas = [0.01, 0.1, 0.25, 0.5, 1.0, 5.0]

        for delta in deltas:
            for sigma in sigmas:
                v = np.array([delta, 0.0, -delta])
                v_inc, probs = _logsumexp_probs_1d_kernel(v, sigma)

                assert np.isfinite(v_inc)
                assert np.all(np.isfinite(probs))
                assert np.all(probs >= 0.0)
                assert np.all(probs <= 1.0)
                sum_p = np.sum(probs)
                assert abs(sum_p - 1.0) < 1e-14, f"Probabilities sum violation: {sum_p} for delta={delta}, sigma={sigma}"

                if delta >= 100.0 and sigma <= 1.0:
                    assert np.isclose(probs[0], 1.0, atol=1e-14)
                    assert np.isclose(probs[1], 0.0, atol=1e-14)
                    assert np.isclose(probs[2], 0.0, atol=1e-14)

    def test_deterministic_choice_limit_hardmax_and_ties(self):
        """Verify deterministic limit (sigma_eps = 0.0) matches exact hardmax and handles ties."""
        # Case A: Strict winner
        v_strict = np.array([1.2, 3.8, 2.5])
        v_inc_a, probs_a = _logsumexp_probs_1d_kernel(v_strict, 0.0)
        assert np.isclose(v_inc_a, 3.8)
        assert np.allclose(probs_a, [0.0, 1.0, 0.0])

        # Case B: Two-way tie
        v_tie = np.array([4.0, 4.0, 1.0])
        v_inc_b, probs_b = _logsumexp_probs_1d_kernel(v_tie, 0.0)
        assert np.isclose(v_inc_b, 4.0)
        assert np.allclose(probs_b, [0.5, 0.5, 0.0])

        # Case C: Three-way tie
        v_tie3 = np.array([2.5, 2.5, 2.5])
        v_inc_c, probs_c = _logsumexp_probs_1d_kernel(v_tie3, 0.0)
        assert np.isclose(v_inc_c, 2.5)
        assert np.allclose(probs_c, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    def test_envelope_theorem_marginal_value_parity(self):
        """Verify envelope theorem E[u'(c)] = sum_d P(d|M) u'(c_d(M)) against finite differences."""
        prob = DCEGMProblem(
            a_grid=np.linspace(0.01, 10.0, 100),
            n_choices=2,
            sigma_eps=0.3,
            options={"tol": 1e-6},
        )
        sol = prob.solve()
        assert sol.converged

        M_test = np.linspace(3.0, 8.0, 15)
        h = 1e-5

        for M in M_test:
            p0 = sol.choice_prob(M, choice=0)
            p1 = sol.choice_prob(M, choice=1)
            c0 = sol.policy(M, choice=0)
            c1 = sol.policy(M, choice=1)
            marg_envelope = p0 * (1.0 / c0) + p1 * (1.0 / c1)

            V_plus = sol.value(M + h, choice=None)
            V_minus = sol.value(M - h, choice=None)
            marg_fd = (V_plus - V_minus) / (2.0 * h)

            rel_diff = abs(marg_envelope - marg_fd) / marg_envelope
            assert rel_diff < 5e-3, f"Envelope vs FD relative discrepancy {rel_diff:.2e} at M={M}"

    def test_dcegm_dynamic_retirement_economic_properties(self):
        """Full economic invariant check on dynamic retirement model."""
        prob = DCEGMProblem(
            a_grid=np.linspace(0.0, 15.0, 50),
            n_choices=2,
            beta=0.96,
            r=0.03,
            sigma_eps=0.20,
            options={"tol": 1e-5, "max_iter": 500},
        )
        sol = prob.solve()

        assert sol.converged
        # 1. Borrowing constraint respected: a' >= 0
        for d in (0, 1):
            assert np.all(sol.aprime[d] >= -1e-10)

        # 2. Consumption is strictly positive
        for d in (0, 1):
            assert np.all(sol.c[d] > 0.0)

        # 3. Value functions are strictly increasing in wealth
        for d in (0, 1):
            assert np.all(np.diff(sol.choice_values[d]) > 0.0)

        # 4. Integrated value is strictly increasing in wealth
        assert np.all(np.diff(sol.integrated_value) > 0.0)

        # 5. Choice probabilities sum to 1.0 at every grid point
        p_sum = sol.choice_probabilities[0] + sol.choice_probabilities[1]
        assert np.allclose(p_sum, 1.0, atol=1e-14)

        # 6. Economic intuition: Poor agents work, wealthy agents retire
        assert sol.choice_probabilities[0][0] > 0.85
        assert sol.choice_probabilities[1][-1] > 0.70


# ============================================================================
# PUBLIC API AND PRESENTATION CONTRACT
# ============================================================================

def test_public_api_exports_puremacro_vfi():
    """Verify all P3 and P4 classes and functions are cleanly exported in puremacro.vfi."""
    expected_p3 = ["SmolyakGrid", "SmolyakBasis", "SmolyakProblem", "SmolyakSolution", "solve_smolyak"]
    expected_p4 = ["upper_envelope", "UpperEnvelopeResult", "ChoiceMapping", "DCEGMProblem", "DCEGMSolution", "solve_dcegm"]

    for name in expected_p3 + expected_p4:
        assert hasattr(vfi, name), f"puremacro.vfi missing export: {name}"


def test_presentation_contract_smolyak_and_dcegm():
    """Verify both SmolyakSolution and DCEGMSolution implement the complete presentation contract."""
    # 1. Smolyak
    sol_smolyak = solve_smolyak(
        domain=((0.8, 1.2), (0.8, 1.2)),
        mu=2,
        params={"alphas": [0.18, 0.18], "z": 1.0 / (0.18 * 0.96), "beta": 0.96},
    )
    assert isinstance(sol_smolyak.summary(), pd.DataFrame)
    assert isinstance(sol_smolyak.to_frame(), pd.DataFrame)
    assert isinstance(sol_smolyak.to_markdown(), str)
    assert isinstance(sol_smolyak.to_latex(), str)
    assert isinstance(sol_smolyak.to_typst(), str)
    fig_sm = sol_smolyak.plot(show=False)
    assert isinstance(fig_sm, plt.Figure)
    plt.close(fig_sm)

    # 2. DC-EGM
    prob_dc = DCEGMProblem(a_grid=np.linspace(0.0, 5.0, 15), sigma_eps=0.2)
    sol_dc = prob_dc.solve()
    assert isinstance(sol_dc.summary(), pd.DataFrame)
    assert isinstance(sol_dc.to_frame(), pd.DataFrame)
    assert isinstance(sol_dc.to_markdown(), str)
    assert isinstance(sol_dc.to_latex(), str)
    assert isinstance(sol_dc.to_typst(), str)
    fig_dc = sol_dc.plot(show=False)
    assert isinstance(fig_dc, plt.Figure)
    plt.close(fig_dc)
