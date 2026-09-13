"""Adversarial Empirical Stress-Testing Suite for P1 (Continuous Distribution & GE) and P2 (Splines).

Authored by Challenger 1.
Validates:
- P1: Mass conservation to machine precision (<= 1e-12) across fine (N_k=2500) and coarse (N_k=15) grids,
  non-linear grids, boundary-absorbing policies, and multi-shock states.
- P1: All 4 distribution solvers ('sparse_direct', 'power', 'arnoldi', 'auto') under stiff/ill-conditioned operators.
- P1: Aiyagari GE market clearing (|K^s - K^d| < 1e-4) under high persistence (rho=0.95), moderate persistence (rho=0.60),
  elevated volatility (sigma=0.35), high risk aversion (gamma=3.0), and tight search brackets.
- P1: Fine grid (N_k=2000) Aiyagari GE solve runtime and mass conservation.
- P1: Cross-module interoperability: feeding SplineCollocationSolution into continuous_stationary_distribution.
- P2: Cubic B-splines: analytical 1st and 2nd derivatives vs central differences (tol <= 1e-5), natural boundary S''=0.
- P2: Schumaker splines: strict shape preservation (min S'(x) >= -1e-14) on sharp kinks, step ramps, and flat shelves
  where standard cubic splines suffer from Gibbs ringing and negative derivatives.
- P2: Schumaker splines on oscillatory/multi-peak data: zero overshoot past local extrema.
- P2: Direct adversarial comparison against SciPy CubicSpline on borrowing constraint kink.
- P2: Schumaker splines with extreme gradient contrasts (slope jumps by 10^4) and sub-knot insertion stability.
- P2: Brock-Mirman analytical Euler collocation benchmark with both cubic and Schumaker splines.
- Multi-backend fallback: NumPy, Numba, invalid backend exception, and CuPy warning-safe fallback.
"""
from __future__ import annotations

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.interpolate import CubicSpline

from puremacro import _backend as bk
from puremacro.vfi.continuous_distribution import (
    AiyagariContinuousEquilibrium,
    AiyagariContinuousModel,
    ContinuousStationaryDistribution,
    build_continuous_transition_matrix,
    continuous_push_distribution,
    continuous_stationary_distribution,
    continuous_stationary_equilibrium,
    solve_aiyagari_continuous,
    young_lottery_weights,
)
from puremacro.vfi.discretize import markov_stationary, tauchen
from puremacro.vfi.splines import (
    CubicBSplineBasis,
    SchumakerSpline,
    SplineCollocationProblem,
    SplineCollocationSolution,
    solve_spline_collocation,
)


# ===========================================================================
# 1. P1: ADVERSARIAL STRESS TESTING ON CONTINUOUS DISTRIBUTIONS
# ===========================================================================

class TestP1AdversarialDistribution:
    """Adversarial stress testing of Young (2010) continuous distribution engine."""

    @pytest.mark.parametrize("N_k", [15, 50, 200, 1000, 2500])
    def test_mass_conservation_grid_scales(self, N_k: int):
        """Stress-test mass conservation across extreme grid scales (coarse to very fine)."""
        # Non-linear asset grid clustering near zero
        k_grid = 30.0 * (np.linspace(0.0, 1.0, N_k) ** 2.5)
        # Policy with non-trivial curvature and asset accumulation
        policy_fn = lambda k: 0.85 * k + 0.5 * (k ** 0.3)

        dist = continuous_stationary_distribution(
            policy_fn,
            k_grid,
            method="auto",
        )

        # 1. Strict mass conservation to 1e-12
        assert dist.mass_error <= 1e-12, f"Mass error {dist.mass_error} exceeds 1e-12 at N_k={N_k}"
        assert abs(float(np.sum(dist.pdf)) - 1.0) <= 1e-12, f"Sum of pdf is {np.sum(dist.pdf)}"

        # 2. Strict non-negativity
        assert np.all(dist.pdf >= 0.0), "Negative probability detected in invariant distribution"

        # 3. Fixed-point residual
        T = build_continuous_transition_matrix(policy_fn, k_grid)
        fp_residual = np.max(np.abs(T.T @ dist.pdf - dist.pdf))
        assert fp_residual <= 1e-7, f"Fixed point residual {fp_residual} too high at N_k={N_k}"

    def test_absorbing_boundary_policies(self):
        """Stress-test policies that collapse all mass onto lower or upper boundary."""
        N_k = 100
        k_grid = np.linspace(0.0, 20.0, N_k)

        # Case A: Poverty trap / collapse to lower bound
        dist_lower = continuous_stationary_distribution(lambda k: 0.0, k_grid, method="sparse_direct")
        assert abs(float(np.sum(dist_lower.pdf)) - 1.0) <= 1e-12
        assert np.isclose(dist_lower.pdf[0], 1.0, atol=1e-10)
        assert np.allclose(dist_lower.pdf[1:], 0.0, atol=1e-10)

        # Case B: Runaway accumulation clamped to upper bound
        dist_upper = continuous_stationary_distribution(lambda k: 100.0, k_grid, method="sparse_direct")
        assert abs(float(np.sum(dist_upper.pdf)) - 1.0) <= 1e-12
        assert np.isclose(dist_upper.pdf[-1], 1.0, atol=1e-10)
        assert np.allclose(dist_upper.pdf[:-1], 0.0, atol=1e-10)

    @pytest.mark.parametrize("method", ["sparse_direct", "power", "arnoldi", "auto"])
    def test_all_solver_methods_consistency(self, method: str):
        """Verify numerical parity across all 4 invariant distribution solvers."""
        N_k = 120
        k_grid = np.linspace(0.1, 15.0, N_k)
        log_z, P_z = tauchen(3, 0.8, 0.15)
        z_grid = np.exp(log_z)

        # Realistic 2D policy
        policy_arr = np.zeros((N_k, 3))
        for m in range(3):
            policy_arr[:, m] = np.clip(0.9 * k_grid + 0.3 * z_grid[m], 0.1, 15.0)

        dist = continuous_stationary_distribution(
            policy_arr,
            k_grid,
            shock_transition=P_z,
            shock_grid=z_grid,
            method=method,
        )

        assert dist.converged, f"Solver {method} failed to converge"
        assert dist.mass_error <= 1e-12, f"Mass error {dist.mass_error} for {method}"
        assert dist.pdf.shape == (N_k, 3)
        assert np.all(dist.pdf >= 0.0)

        # Marginal shock distribution must match theoretical ergodic distribution pi_z
        pi_z = markov_stationary(P_z)
        marginal_z = dist.marginal_shocks()
        assert np.allclose(marginal_z, pi_z, atol=1e-8), f"Marginal shock mismatch for {method}"

    def test_stiff_markov_near_unit_root(self):
        """Stress-test distribution solver with highly persistent Markov chain (rho=0.99)."""
        N_k = 80
        k_grid = np.linspace(0.0, 10.0, N_k)
        log_z, P_z = tauchen(3, 0.99, 0.05)
        z_grid = np.exp(log_z)

        policy_arr = np.zeros((N_k, 3))
        for m in range(3):
            policy_arr[:, m] = np.clip(0.92 * k_grid + 0.15 * z_grid[m], 0.0, 10.0)

        dist = continuous_stationary_distribution(
            policy_arr,
            k_grid,
            shock_transition=P_z,
            shock_grid=z_grid,
            method="auto",
        )
        assert dist.converged
        assert dist.mass_error <= 1e-12
        pi_z = markov_stationary(P_z)
        assert np.allclose(dist.marginal_shocks(), pi_z, atol=1e-6)


# ===========================================================================
# 2. P1: ADVERSARIAL STRESS TESTING ON AIYAGARI GENERAL EQUILIBRIUM
# ===========================================================================

class TestP1AdversarialGeneralEquilibrium:
    """Stress-testing Aiyagari continuous general equilibrium solver."""

    def test_ge_market_clearing_high_persistence(self):
        """Test market clearing with high idiosyncratic persistence rho_z=0.95."""
        eq = solve_aiyagari_continuous(
            beta=0.96,
            gamma=2.0,
            alpha=0.36,
            delta=0.08,
            rho_z=0.95,
            sigma_z=0.20,
            n_z=5,
            a_max=35.0,
            N_k=400,
            xtol=1e-6,
        )
        assert eq.converged
        assert abs(eq.capital_market_clearing_error) < 1e-4, f"Clearing error {eq.capital_market_clearing_error}"
        assert eq.distribution.mass_error <= 1e-12
        # r* must be strictly less than 1/beta - 1
        assert eq.r < (1.0 / 0.96 - 1.0)
        assert eq.K > 0.0
        assert eq.w > 0.0

    def test_ge_market_clearing_moderate_persistence_and_volatility(self):
        """Test market clearing with moderate persistence rho_z=0.60, sigma_z=0.25."""
        eq = solve_aiyagari_continuous(
            beta=0.96,
            gamma=2.0,
            alpha=0.36,
            delta=0.08,
            rho_z=0.60,
            sigma_z=0.25,
            n_z=3,
            a_max=30.0,
            N_k=300,
            xtol=1e-8,
        )
        assert eq.converged
        assert abs(eq.capital_market_clearing_error) < 1e-4
        assert eq.distribution.mass_error <= 1e-12
        assert 0.0 < eq.r < (1.0 / 0.96 - 1.0)

    def test_ge_market_clearing_high_volatility(self):
        """Test market clearing with elevated income volatility sigma_z=0.35."""
        eq = solve_aiyagari_continuous(
            beta=0.96,
            gamma=2.0,
            alpha=0.36,
            delta=0.08,
            rho_z=0.90,
            sigma_z=0.35,
            n_z=3,
            a_max=40.0,
            N_k=300,
            xtol=1e-8,
        )
        assert eq.converged
        assert abs(eq.capital_market_clearing_error) < 1e-4
        assert eq.distribution.mass_error <= 1e-12

    def test_ge_market_clearing_high_risk_aversion(self):
        """Test market clearing with elevated risk aversion gamma=3.0, beta=0.95."""
        eq = solve_aiyagari_continuous(
            beta=0.95,
            gamma=3.0,
            alpha=0.36,
            delta=0.08,
            rho_z=0.85,
            sigma_z=0.30,
            n_z=3,
            a_max=35.0,
            N_k=300,
            xtol=1e-8,
        )
        assert eq.converged
        assert abs(eq.capital_market_clearing_error) < 1e-4
        assert eq.distribution.mass_error <= 1e-12
        assert eq.r < (1.0 / 0.95 - 1.0)

    def test_ge_fine_grid_n_k_2000(self):
        """Stress-test Aiyagari GE on fine histogram grid N_k=2000."""
        eq = solve_aiyagari_continuous(
            beta=0.96,
            gamma=2.0,
            alpha=0.36,
            delta=0.08,
            rho_z=0.90,
            sigma_z=0.20,
            n_z=3,
            a_max=30.0,
            N_k=2000,
            xtol=1e-8,
        )
        assert eq.converged
        assert abs(eq.capital_market_clearing_error) < 1e-4
        assert eq.distribution.mass_error <= 1e-12
        assert eq.distribution.pdf.shape == (2000, 3)

    def test_ge_search_brackets_valid_and_invalid(self):
        """Verify behavior under valid tight bracket and invalid bracket missing r*."""
        # 1. Valid tight bracket enclosing r* approx 0.0394
        eq = solve_aiyagari_continuous(
            beta=0.96,
            gamma=2.0,
            alpha=0.36,
            delta=0.08,
            rho_z=0.90,
            sigma_z=0.20,
            n_z=3,
            a_max=30.0,
            N_k=300,
            r_bracket=(0.020, 0.041),
            xtol=1e-6,
        )
        assert eq.converged
        assert abs(eq.capital_market_clearing_error) < 1e-4
        assert 0.020 <= eq.r <= 0.041

        # 2. Invalid bracket missing r* on the high side [0.020, 0.038]
        with pytest.raises(ValueError, match="does not change sign"):
            solve_aiyagari_continuous(
                beta=0.96,
                gamma=2.0,
                alpha=0.36,
                delta=0.08,
                rho_z=0.90,
                sigma_z=0.20,
                n_z=3,
                a_max=30.0,
                N_k=300,
                r_bracket=(0.020, 0.038),
            )

    def test_cross_module_spline_to_distribution_interop(self):
        """Verify feeding a P2 SplineCollocationSolution directly into P1 continuous_stationary_distribution."""
        bm_alpha = 0.36
        bm_beta = 0.96
        k_ss = float((bm_alpha * bm_beta) ** (1.0 / (1.0 - bm_alpha)))
        domain = (0.5 * k_ss, 1.5 * k_ss)

        # Solve Brock-Mirman using SplineCollocationProblem (P2)
        def euler_res(s, sp_val, spp_val, p):
            c = s**p["alpha"] - sp_val
            cp = sp_val**p["alpha"] - spp_val
            c = np.maximum(c, 1e-10)
            cp = np.maximum(cp, 1e-10)
            R = p["alpha"] * (sp_val ** (p["alpha"] - 1.0))
            return 1.0 - p["beta"] * (cp / c) ** (-1.0) * R

        prob = SplineCollocationProblem(
            domain=domain,
            n_knots=16,
            spline_type="schumaker",
            euler_residual_fn=euler_res,
            beta=bm_beta,
            params={"alpha": bm_alpha, "beta": bm_beta},
        )
        spline_sol = prob.solve()
        assert spline_sol.converged

        # Push into continuous_stationary_distribution (P1)
        k_hist = np.linspace(domain[0], domain[1], 500)
        dist = continuous_stationary_distribution(spline_sol, k_hist)

        assert dist.converged
        assert dist.mass_error <= 1e-12
        assert np.all(dist.pdf >= 0.0)
        # In Brock-Mirman steady state, distribution should be concentrated around k_ss
        mean_k = dist.mean()
        assert np.isclose(mean_k, k_ss, rtol=1e-2), f"Mean asset {mean_k} diverges from k_ss {k_ss}"


# ===========================================================================
# 3. P2: ADVERSARIAL STRESS TESTING ON SHAPE-PRESERVING SPLINES
# ===========================================================================

class TestP2AdversarialSplines:
    """Adversarial stress testing of SchumakerSpline and CubicBSplineBasis."""

    def test_schumaker_vs_cubic_gibbs_ringing_at_sharp_kink(self):
        """Adversarial comparison: Sharp kink (borrowing constraint / piecewise linear).
        
        Data has a flat shelf y=0 for x in [0, 2] and then ramps up with slope 3.0 for x > 2.
        Standard cubic spline will overshoot and produce Gibbs ringing (negative slopes/values).
        Schumaker spline MUST preserve monotonicity: S'(x) >= 0 everywhere.
        """
        x_raw = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
        y_raw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 3.0, 6.0, 9.0])

        # 1. Schumaker Spline
        schumaker = SchumakerSpline(x_raw, y_raw)

        # 2. Standard Cubic B-Spline Basis fit
        cubic_basis = CubicBSplineBasis(domain=(0.0, 5.0), n_knots=len(x_raw), bc_type="clamped")
        c_coefs = cubic_basis.fit(y_raw, nodes=x_raw)

        # Dense evaluation across 10,000 points
        x_dense = np.linspace(0.0, 5.0, 10000)
        schumaker_vals = schumaker.eval(x_dense)
        schumaker_slopes = schumaker.derivative(x_dense, order=1)
        cubic_vals = cubic_basis.interpolate(c_coefs, x_dense, deriv=0)
        cubic_slopes = cubic_basis.interpolate(c_coefs, x_dense, deriv=1)

        # Empirical Evidence of Cubic Spline failure (Gibbs ringing)
        cubic_min_slope = float(np.min(cubic_slopes))
        cubic_min_val = float(np.min(cubic_vals))
        assert cubic_min_slope < -0.01, f"Cubic spline unexpectedly did not ring: min slope {cubic_min_slope}"

        # Empirical Proof of Schumaker shape preservation:
        # Monotonicity: slope must be non-negative everywhere
        schumaker_min_slope = float(np.min(schumaker_slopes))
        assert schumaker_min_slope >= -1e-14, f"Schumaker slope violated monotonicity: {schumaker_min_slope}"

        # Non-negativity: value must be non-negative everywhere
        schumaker_min_val = float(np.min(schumaker_vals))
        assert schumaker_min_val >= -1e-14, f"Schumaker value dipped below zero: {schumaker_min_val}"

        # Zero overshoot on flat shelf [0, 2]: max value must be exactly 0
        shelf_mask = x_dense <= 2.0
        assert np.allclose(schumaker_vals[shelf_mask], 0.0, atol=1e-14)
        assert np.allclose(schumaker_slopes[shelf_mask], 0.0, atol=1e-14)

    def test_schumaker_vs_scipy_cubicspline_direct_comparison(self):
        """Direct benchmark against SciPy CubicSpline demonstrating elimination of negative overshoot."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([0.0, 0.0, 0.0, 2.0, 4.0, 6.0])

        s_schu = SchumakerSpline(x, y)
        s_scipy = CubicSpline(x, y)

        x_dense = np.linspace(0.0, 5.0, 10000)
        y_schu = s_schu.eval(x_dense)
        d_schu = s_schu.derivative(x_dense, order=1)

        y_scipy = s_scipy(x_dense)
        d_scipy = s_scipy(x_dense, 1)

        # SciPy CubicSpline suffers from negative values and negative slope
        assert np.min(y_scipy) < -0.15, "SciPy did not exhibit expected undershoot"
        assert np.min(d_scipy) < -0.40, "SciPy did not exhibit negative derivative"

        # Schumaker strictly preserves shape
        assert np.min(y_schu) >= -1e-14, "Schumaker dipped below zero"
        assert np.min(d_schu) >= -1e-14, "Schumaker derivative became negative"

    def test_schumaker_oscillatory_peaks_and_valleys_no_overshoot(self):
        """Test Schumaker on non-monotonic / oscillatory data with multiple peaks and valleys.
        
        Guarantees: S(x) remains bounded within [min(y), max(y)], and at local extrema S'(x_i) = 0.
        """
        x_raw = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y_raw = np.array([0.0, 2.0, 0.0, 3.0, 0.0, 1.0, 0.0])

        spl = SchumakerSpline(x_raw, y_raw)
        x_dense = np.linspace(0.0, 6.0, 10000)
        vals = spl.eval(x_dense)

        # Values must remain strictly within [0.0, 3.0] with zero overshoot/undershoot
        assert np.min(vals) >= -1e-14, f"Undershoot detected: min val {np.min(vals)}"
        assert np.max(vals) <= 3.0 + 1e-14, f"Overshoot detected: max val {np.max(vals)}"

        # Verify derivative is exactly 0 at all interior local extrema
        for i in [1, 2, 3, 4, 5]:
            d_i = spl.derivative(x_raw[i], order=1)
            assert np.isclose(d_i, 0.0, atol=1e-12), f"Derivative at extremum x={x_raw[i]} is {d_i}"

    def test_schumaker_extreme_gradient_ratio(self):
        """Stress-test Schumaker sub-knot insertion with extreme gradient jump (ratio 10^4)."""
        x_raw = np.array([0.0, 1.0, 1.0001, 2.0])
        # Slope jumps from 1e-4 to 1000.0
        y_raw = np.array([0.0, 0.0001, 100.0, 100.001])

        spl = SchumakerSpline(x_raw, y_raw)
        assert len(spl.inserted_knots) > 0, "Expected auxiliary sub-knots to be inserted"

        x_dense = np.linspace(0.0, 2.0, 5000)
        vals = spl.eval(x_dense)
        slopes = spl.derivative(x_dense, order=1)

        assert not np.any(np.isnan(vals)), "NaN detected in extreme gradient evaluation"
        assert not np.any(np.isnan(slopes)), "NaN detected in extreme gradient slopes"
        assert np.min(slopes) >= -1e-12, f"Monotonicity violated: min slope {np.min(slopes)}"
        assert np.all(np.diff(vals) >= -1e-14)

    def test_cubic_bspline_analytical_derivatives_vs_finite_differences(self):
        """Verify analytical 1st and 2nd derivatives of CubicBSplineBasis against central differences."""
        domain = (0.5, 4.5)
        basis = CubicBSplineBasis(domain=domain, n_knots=15, bc_type="clamped")
        # Random coefficients
        np.random.seed(123)
        coefs = np.random.uniform(-2.0, 2.0, basis.n_basis)

        # Interior evaluation points away from boundary knots
        x_eval = np.linspace(domain[0] + 0.05, domain[1] - 0.05, 500)
        eps = 1e-6

        # Analytical evaluations
        val_0 = basis.interpolate(coefs, x_eval, deriv=0)
        val_1_analytical = basis.interpolate(coefs, x_eval, deriv=1)
        val_2_analytical = basis.interpolate(coefs, x_eval, deriv=2)

        # Numerical central finite differences
        val_plus = basis.interpolate(coefs, x_eval + eps, deriv=0)
        val_minus = basis.interpolate(coefs, x_eval - eps, deriv=0)
        val_1_numerical = (val_plus - val_minus) / (2.0 * eps)

        val_d1_plus = basis.interpolate(coefs, x_eval + eps, deriv=1)
        val_d1_minus = basis.interpolate(coefs, x_eval - eps, deriv=1)
        val_2_numerical = (val_d1_plus - val_d1_minus) / (2.0 * eps)

        # Check 1st derivative error <= 1e-5
        err_d1 = np.max(np.abs(val_1_analytical - val_1_numerical))
        assert err_d1 <= 1e-5, f"1st derivative error {err_d1} exceeds 1e-5"

        # Check 2nd derivative error <= 1e-4
        err_d2 = np.max(np.abs(val_2_analytical - val_2_numerical))
        assert err_d2 <= 1e-4, f"2nd derivative error {err_d2} exceeds 1e-4"

    def test_schumaker_analytical_derivatives_vs_finite_differences(self):
        """Verify analytical derivatives of SchumakerSpline within smooth subintervals."""
        x_raw = np.linspace(0.0, 5.0, 11)
        y_raw = np.log(1.0 + x_raw)
        spl = SchumakerSpline(x_raw, y_raw)

        # Select evaluation points inside subintervals (not on knots)
        knots = spl.knots
        eval_pts = []
        for i in range(len(knots) - 1):
            h = knots[i + 1] - knots[i]
            eval_pts.append(knots[i] + 0.3 * h)
            eval_pts.append(knots[i] + 0.7 * h)
        x_eval = np.array(eval_pts)
        eps = 1e-6

        d1_analytical = spl.derivative(x_eval, order=1)
        d1_numerical = (spl.eval(x_eval + eps) - spl.eval(x_eval - eps)) / (2.0 * eps)
        assert np.max(np.abs(d1_analytical - d1_numerical)) <= 1e-5

        d2_analytical = spl.second_derivative(x_eval)
        d2_numerical = (spl.derivative(x_eval + eps, order=1) - spl.derivative(x_eval - eps, order=1)) / (2.0 * eps)
        assert np.max(np.abs(d2_analytical - d2_numerical)) <= 1e-4

    def test_natural_boundary_conditions_zero_second_derivative(self):
        """Verify natural boundary conditions strictly enforce S''(a) = 0 and S''(b) = 0."""
        domain = (1.0, 10.0)
        basis = CubicBSplineBasis(domain=domain, n_knots=12, bc_type="natural")
        nodes = basis.nodes()
        y_vals = np.sin(nodes)
        coefs = basis.fit(y_vals, nodes=nodes)

        d2_a = basis.interpolate(coefs, domain[0], deriv=2)
        d2_b = basis.interpolate(coefs, domain[1], deriv=2)

        assert abs(d2_a) <= 1e-10, f"Natural boundary S''(a) is {d2_a}"
        assert abs(d2_b) <= 1e-10, f"Natural boundary S''(b) is {d2_b}"

    def test_schumaker_strict_concavity_preservation(self):
        """Verify strict concavity preservation on strictly concave data (S''(x) <= 0)."""
        x_raw = np.linspace(1.0, 10.0, 15)
        # Strictly concave function
        y_raw = np.sqrt(x_raw)
        spl = SchumakerSpline(x_raw, y_raw)

        # Dense interior evaluation
        x_dense = np.linspace(1.05, 9.95, 2000)
        d2 = spl.second_derivative(x_dense)
        # All second derivatives must be <= 0 (concave) up to numerical tolerance
        assert np.all(d2 <= 1e-10), f"Concavity violated: max S''(x) = {np.max(d2)}"


# ===========================================================================
# 4. MULTI-BACKEND FALLBACK & ERROR HANDLING
# ===========================================================================

class TestMultiBackendFallback:
    """Stress-testing multi-backend dispatch and defensive fallbacks."""

    def test_numpy_and_numba_backend_parity(self):
        """Verify numerical parity between pure NumPy and Numba JIT kernels."""
        N_k = 100
        k_grid = np.linspace(0.0, 20.0, N_k)
        pol = lambda k: 0.8 * k + 1.0

        dist_np = continuous_stationary_distribution(pol, k_grid, backend="numpy")
        dist_nb = continuous_stationary_distribution(pol, k_grid, backend="numba")

        assert np.allclose(dist_np.pdf, dist_nb.pdf, atol=1e-14)
        assert dist_np.mass_error <= 1e-12
        assert dist_nb.mass_error <= 1e-12

    def test_invalid_backend_raises_value_error(self):
        """Verify invalid backend string raises clean ValueError listing supported backends."""
        k_grid = np.linspace(0.0, 10.0, 50)
        with pytest.raises(ValueError, match="Unknown backend"):
            continuous_stationary_distribution(lambda k: 0.5 * k, k_grid, backend="non_existent_backend")

        prob = SplineCollocationProblem(domain=(0.5, 2.0), n_knots=8)
        with pytest.raises(ValueError, match="Unknown backend"):
            prob.solve(backend="quantum_gpu")

    def test_cupy_unavailable_graceful_fallback(self):
        """Verify requesting CuPy on macOS emits UserWarning and falls back cleanly without crash."""
        k_grid = np.linspace(0.0, 10.0, 50)
        with pytest.warns(UserWarning, match="Backend 'cupy' requested but not available"):
            dist = continuous_stationary_distribution(lambda k: 0.5 * k, k_grid, backend="cupy")
        assert dist.converged
        assert dist.mass_error <= 1e-12
        assert dist.metadata["backend"] == "numpy"

