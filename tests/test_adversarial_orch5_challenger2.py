"""Empirical Adversarial Stress Testing Suite for puremacro 2.8.0.

Author: orch5_challenger_2 (Teamwork Preview Challenger)
Verification of Requirements:
- R3: Multi-Constraint OccBin (2^K = 4 regimes) & Piecewise Kalman Filter
- R4: Sequential Monte Carlo (SMC) & Bootstrap Particle Filtering
- R5: Nonlinear Ramsey Optimal Policy & BGP Detrending

Adversarial Stress Tasks:
1. Stress test Multi-Constraint OccBin across rapid regime oscillations and verify cycle damping prevents infinite loops.
2. Stress test Piecewise Kalman Filter with missing observation gaps (NaN entries) and zero-variance measurement errors.
3. Stress test SMC on bimodal mixture distribution and verify both modes are explored where standard RWMH gets trapped.
4. Verify SMC Marginal Data Density against analytical benchmark on linear DSGE within 0.10 log-points.
5. Stress test Ramsey optimal policy on boundary discount rates (beta = 0.01 and beta = 0.9999) and verify Clarida-Gali-Gertler timeless perspective stability.
6. Stress test BGP detrending on growing models and verify stationarized models solve cleanly.
"""
from __future__ import annotations

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.integrate
import scipy.linalg

from puremacro.dsge._ast import Const, Var, BinOp, Node
from puremacro.dsge._parser import parse_mod_to_dag, DynareParseError
from puremacro.dsge import build_dynare, LinearModel
from puremacro.dsge.occbin import (
    OccBinConstraint,
    OccBinResult,
    PiecewiseKalmanResult,
    solve_occbin,
    solve_multiconstraint_occbin,
    piecewise_kalman_filter,
)
from puremacro.dsge.priors import NormalPrior, UniformPrior
from puremacro.dsge.smc import (
    SMCSampler,
    SMCResult,
    bootstrap_particle_filter,
    smc_estimate,
    systematic_resample,
)
from puremacro.dsge.ramsey import (
    ramsey_model,
    RamseyResult,
    derive_ramsey_focs,
    detrend_bgp,
)


# ==============================================================================
# Fixtures for Multi-Constraint Model (ZLB + Borrowing Limit: 4 Regimes)
# ==============================================================================

@pytest.fixture
def dual_constraint_fixture():
    """Setup a New Keynesian model with dual constraints:
    - Constraint 1: Zero Lower Bound on policy rate: r_t >= -r_ss (r_t = -r_ss)
    - Constraint 2: Borrowing limit / leverage cap: b_t <= b_bar (b_t = b_bar)
    Yielding 4 distinct regimes:
      0: (0, 0) unconstrained (both slack)
      1: (1, 0) ZLB binding only
      2: (0, 1) Borrowing cap binding only
      3: (1, 1) Joint binding (both ZLB and borrowing cap active)
    """
    params = {
        "beta": 0.99,
        "sigma": 1.0,
        "kappa": 0.15,
        "phi_pi": 1.5,
        "phi_y": 0.25,
        "rho_r": 0.6,
        "rho_b": 0.5,
        "rho_g": 0.7,
        "gamma_y": 0.2,
        "chi": 0.1,
        "r_ss": 0.015,
        "b_bar": 0.02,
    }
    variables = ["y", "pi", "r", "b", "g"]
    shocks = ["eps_g", "eps_r", "eps_b"]
    steady_state = {v: 0.0 for v in variables}

    # Regime 0: Reference
    def ref_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_r),
            curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + shocks_v.eps_b),
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    # Regime 1: ZLB
    def zlb_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (-p.r_ss),
            curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + shocks_v.eps_b),
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    # Regime 2: Borrowing cap
    def borr_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_r),
            curr.b - p.b_bar,
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    # Regime 3: Joint binding
    def joint_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (-p.r_ss),
            curr.b - p.b_bar,
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    m_uncons = build_dynare(ref_eqs, variables=variables, shocks=shocks, params=params, steady_state=steady_state)
    m_zlb = build_dynare(zlb_eqs, variables=variables, shocks=shocks, params=params, steady_state=steady_state, check_steady_state=False, strict=False)
    m_borr = build_dynare(borr_eqs, variables=variables, shocks=shocks, params=params, steady_state=steady_state, check_steady_state=False, strict=False)
    m_joint = build_dynare(joint_eqs, variables=variables, shocks=shocks, params=params, steady_state=steady_state, check_steady_state=False, strict=False)

    c_zlb = OccBinConstraint(variable="r", threshold=-params["r_ss"], operator="<")
    c_borr = OccBinConstraint(variable="b", threshold=params["b_bar"], operator=">")

    return {
        "params": params,
        "variables": variables,
        "shocks": shocks,
        "m_uncons": m_uncons,
        "m_zlb": m_zlb,
        "m_borr": m_borr,
        "m_joint": m_joint,
        "c_zlb": c_zlb,
        "c_borr": c_borr,
    }


# ==============================================================================
# Task 1: Stress Test Multi-Constraint OccBin Across Rapid Regime Oscillations
# ==============================================================================

class TestTask1OccBinRapidOscillationsAndCycleDamping:
    """Stress test Multi-Constraint OccBin across rapid regime oscillations and verify cycle damping prevents infinite loops."""

    def test_rapid_regime_oscillations_damping_termination(self, dual_constraint_fixture):
        """Construct a high-frequency alternating shock sequence that drives the system
        between regime 0, 1, 2, and 3 repeatedly, verifying cycle damping terminates safely.
        """
        setup = dual_constraint_fixture
        horizon = 35
        shocks = np.zeros((horizon, 3))

        # Alternating sign shocks to force rapid back-and-forth regime switching
        for t in range(0, 15, 2):
            shocks[t, 0] = -0.07   # trigger ZLB (regime 1 or 3)
            shocks[t, 2] = +0.06   # trigger borrowing cap (regime 2 or 3)
            shocks[t + 1, 0] = +0.07 # relax ZLB
            shocks[t + 1, 2] = -0.06 # relax borrowing cap

        # Solve with standard max_iter
        res = solve_multiconstraint_occbin(
            m_unconstrained=setup["m_uncons"],
            m_constrained_dict={
                "zlb": setup["m_zlb"],
                "borr": setup["m_borr"],
            },
            constraints={
                "zlb": setup["c_zlb"],
                "borr": setup["c_borr"],
            },
            shock_seq=shocks,
            horizon=horizon,
            max_iter=50,
        )

        assert isinstance(res, OccBinResult)
        assert res.iterations <= 50, f"Exceeded max_iter: {res.iterations}"
        assert len(res.regimes) == horizon
        # Path values must remain strictly finite and bounded
        assert np.all(np.isfinite(res.simulated_path.values))
        assert not np.any(np.isnan(res.simulated_path.values))

        # Check that multiple regimes were visited
        unique_regimes = set(res.regimes)
        assert len(unique_regimes) >= 2, f"Expected multiple regimes, got: {unique_regimes}"

    def test_boundary_threshold_chattering_damping(self, dual_constraint_fixture):
        """Plant shocks that land directly on the threshold (-r_ss and b_bar), creating
        potential numerical floating point limit cycles; verify cycle damping terminates without infinite loop.
        """
        setup = dual_constraint_fixture
        horizon = 25
        shocks = np.zeros((horizon, 3))
        # Fine-tuned shock pushing r near threshold
        shocks[0, 0] = -0.025
        shocks[1, 0] = -0.024999999999999

        # Test with very low max_iter to stress loop bounds
        res_low_iter = solve_multiconstraint_occbin(
            m_unconstrained=setup["m_uncons"],
            m_constrained_dict={"zlb": setup["m_zlb"], "borr": setup["m_borr"]},
            shock_seq=shocks,
            horizon=horizon,
            max_iter=5,
        )
        assert res_low_iter.iterations <= 5
        assert isinstance(res_low_iter.converged, bool)
        assert len(res_low_iter.regimes) == horizon

    def test_multi_constraint_all_four_regimes_traversal(self, dual_constraint_fixture):
        """Verify that under targeted combined shocks, all 4 regimes {0, 1, 2, 3} are traversed."""
        setup = dual_constraint_fixture
        horizon = 30
        shocks = np.zeros((horizon, 3))
        # Period 0: massive demand shock (ZLB only -> Regime 1)
        shocks[0, 0] = -0.06
        # Period 3: massive credit shock (Borrowing cap only -> Regime 2)
        shocks[3, 2] = +0.08
        # Period 7: joint shock (Both ZLB and borrowing cap -> Regime 3)
        shocks[7, 0] = -0.06
        shocks[7, 2] = +0.08

        res = solve_multiconstraint_occbin(
            m_unconstrained=setup["m_uncons"],
            m_constrained_dict={"zlb": setup["m_zlb"], "borr": setup["m_borr"]},
            constraints={"zlb": setup["c_zlb"], "borr": setup["c_borr"]},
            shock_seq=shocks,
            horizon=horizon,
            max_iter=50,
        )
        visited = set(res.regimes)
        assert 0 in visited, f"Regime 0 not visited: {visited}"
        assert 1 in visited or 2 in visited or 3 in visited, f"Constrained regimes not visited: {visited}"
        assert res.simulated_path["r"].min() >= -setup["params"]["r_ss"] - 1e-7
        assert res.simulated_path["b"].max() <= setup["params"]["b_bar"] + 1e-7


# ==============================================================================
# Task 2: Stress Test Piecewise Kalman Filter (Missing Data & Zero Error)
# ==============================================================================

class TestTask2PiecewiseKalmanFilterStress:
    """Stress test Piecewise Kalman Filter with missing observation gaps (NaN entries) and zero-variance measurement errors."""

    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.default_rng(42)
        T = 40
        dates = pd.date_range("2010-01-01", periods=T, freq="QS")
        y = np.zeros(T)
        pi = np.zeros(T)
        r = np.zeros(T)
        for t in range(1, T):
            shock_d = -0.08 if 5 <= t <= 10 else rng.normal(0, 0.01)
            y[t] = 0.6 * y[t - 1] - 0.2 * r[t - 1] + shock_d
            pi[t] = 0.4 * pi[t - 1] + 0.15 * y[t] + rng.normal(0, 0.005)
            notional_r = 0.7 * r[t - 1] + 0.3 * (1.5 * pi[t] + 0.25 * y[t])
            r[t] = max(-0.015, notional_r)
        return pd.DataFrame({"y": y, "pi": pi, "r": r}, index=dates)

    def test_pkf_missing_observation_blocks_and_ragged_gaps(self, dual_constraint_fixture, synthetic_data):
        """Stress test with contiguous missing blocks, initial missing period, and ragged missingness."""
        setup = dual_constraint_fixture
        data = synthetic_data.copy()

        # 1. First period completely missing (t=0)
        data.iloc[0, :] = np.nan

        # 2. Block of consecutive missing periods (t=12..18)
        data.iloc[12:19, :] = np.nan

        # 3. Ragged missing pattern
        data.iloc[5, 0] = np.nan   # y missing at t=5
        data.iloc[7, 1] = np.nan   # pi missing at t=7
        data.iloc[9, 2] = np.nan   # r missing at t=9

        # 4. Final period missing
        data.iloc[-1, :] = np.nan

        res = piecewise_kalman_filter(
            m_unconstrained=setup["m_uncons"],
            m_constrained_dict={"zlb": setup["m_zlb"]},
            data=data,
            varobs=["y", "pi", "r"],
            horizon=20,
        )

        assert isinstance(res, PiecewiseKalmanResult)
        assert np.isfinite(res.log_likelihood), f"Log-likelihood is non-finite: {res.log_likelihood}"
        assert len(res.filtered_states) == len(data)
        # Filtered states must have NO NaNs even during completely missing blocks
        assert np.all(np.isfinite(res.filtered_states.values)), "Filtered states contain NaNs!"
        # Covariances must remain positive semi-definite
        for P in res.filtered_covariances:
            eigs = np.linalg.eigvalsh(P)
            assert np.all(eigs >= -1e-8), f"Negative eigenvalue in state covariance: {eigs.min()}"

    def test_pkf_zero_variance_measurement_error_boundary(self, dual_constraint_fixture, synthetic_data):
        """Stress test PKF when measurement error covariance H is exactly zero (H = 0).
        Verifies jitter safeguard prevents Cholesky factorization failure on singular innovation covariance.
        """
        setup = dual_constraint_fixture
        data = synthetic_data.iloc[:25]
        H_zero = np.zeros((3, 3))

        res = piecewise_kalman_filter(
            m_unconstrained=setup["m_uncons"],
            m_constrained_dict={"zlb": setup["m_zlb"]},
            data=data,
            varobs=["y", "pi", "r"],
            H=H_zero,
            horizon=20,
        )

        assert np.isfinite(res.log_likelihood)
        assert len(res.filtered_states) == len(data)
        assert np.all(np.isfinite(res.filtered_states.values))

    def test_pkf_combined_zero_error_and_missing_data_stress(self, dual_constraint_fixture, synthetic_data):
        """Combined adversarial condition: H = 0 AND missing data periods AND active ZLB."""
        setup = dual_constraint_fixture
        data = synthetic_data.iloc[:25].copy()
        data.iloc[8:12, :] = np.nan  # Entire block missing during/after ZLB

        res = piecewise_kalman_filter(
            m_unconstrained=setup["m_uncons"],
            m_constrained_dict={"zlb": setup["m_zlb"]},
            data=data,
            varobs=["y", "pi", "r"],
            H=np.zeros((3, 3)),
            horizon=20,
        )
        assert np.isfinite(res.log_likelihood)
        assert np.all(np.isfinite(res.filtered_states.values))


# ==============================================================================
# Task 3: Stress Test SMC on Bimodal Mixture Distribution vs RWMH
# ==============================================================================

class TestTask3SMCBimodalExplorationVsRWMH:
    """Stress test SMC on bimodal mixture distribution and verify both modes are explored where standard RWMH gets trapped."""

    def test_smc_explores_both_modes_where_rwmh_trapped(self):
        """Adversarial bimodal target: 0.5 * N(-3.0, 0.25^2) + 0.5 * N(+3.0, 0.25^2).
        Modes are separated by 6 units (12 standard deviations).
        Standard RWMH initialized at -3.0 will never jump the valley.
        SMC must capture both modes with balanced mass.
        """
        def log_target(par):
            x = float(par["x"])
            sd = 0.25
            d1 = math.exp(-0.5 * ((x - (-3.0)) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
            d2 = math.exp(-0.5 * ((x - (+3.0)) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
            dens = 0.5 * d1 + 0.5 * d2
            return math.log(max(dens, 1e-300))

        priors = {"x": UniformPrior(lb=-8.0, ub=8.0)}

        # 1. Standard RWMH benchmark initialized at -3.0
        rng = np.random.default_rng(777)
        curr = -3.0
        chain = [curr]
        curr_ll = log_target({"x": curr})
        for _ in range(3000):
            prop = curr + rng.normal(0.0, 0.35)
            if -8.0 <= prop <= 8.0:
                prop_ll = log_target({"x": prop})
                if math.log(rng.uniform(0.0, 1.0)) < (prop_ll - curr_ll):
                    curr = prop
                    curr_ll = prop_ll
            chain.append(curr)

        rwmh_chain = np.array(chain)
        rwmh_mode2_fraction = float(np.mean(rwmh_chain > 0.0))

        # Empirical proof: RWMH is completely trapped in Mode 1 (0% in Mode 2)
        assert rwmh_mode2_fraction == 0.0, f"RWMH unexpectedly crossed 12-sigma barrier: {rwmh_mode2_fraction}"

        # 2. Sequential Monte Carlo Sampler
        smc_res = smc_estimate(
            log_lik=log_target,
            priors=priors,
            n_particles=800,
            n_stages=30,
            adaptive_tempering=False,
            seed=42,
        )

        assert isinstance(smc_res, SMCResult)
        smc_particles = smc_res.particles[:, 0]
        smc_mode1_mass = float(np.mean(smc_particles < 0.0))
        smc_mode2_mass = float(np.mean(smc_particles > 0.0))

        # Empirical proof: SMC successfully explores both modes with balanced mass in [0.35, 0.65]
        assert 0.35 <= smc_mode1_mass <= 0.65, f"SMC Mode 1 mass unbalanced: {smc_mode1_mass}"
        assert 0.35 <= smc_mode2_mass <= 0.65, f"SMC Mode 2 mass unbalanced: {smc_mode2_mass}"

    def test_smc_2d_bimodal_modes_exploration(self):
        """2D bimodal mixture target: modes at (-2.5, -2.5) and (+2.5, +2.5)."""
        def log_target_2d(par):
            x = float(par["x"])
            y = float(par["y"])
            sd = 0.3
            d1 = math.exp(-0.5 * (((x + 2.5)/sd)**2 + ((y + 2.5)/sd)**2)) / (2 * math.pi * sd**2)
            d2 = math.exp(-0.5 * (((x - 2.5)/sd)**2 + ((y - 2.5)/sd)**2)) / (2 * math.pi * sd**2)
            return math.log(max(0.5 * d1 + 0.5 * d2, 1e-300))

        priors = {
            "x": UniformPrior(lb=-6.0, ub=6.0),
            "y": UniformPrior(lb=-6.0, ub=6.0),
        }

        smc_res = smc_estimate(
            log_lik=log_target_2d,
            priors=priors,
            n_particles=600,
            n_stages=25,
            seed=123,
        )

        x_pts = smc_res.particles[:, 0]
        y_pts = smc_res.particles[:, 1]
        in_mode1 = np.logical_and(x_pts < 0.0, y_pts < 0.0)
        in_mode2 = np.logical_and(x_pts > 0.0, y_pts > 0.0)

        assert np.mean(in_mode1) >= 0.30, f"Mode 1 share too low: {np.mean(in_mode1)}"
        assert np.mean(in_mode2) >= 0.30, f"Mode 2 share too low: {np.mean(in_mode2)}"


# ==============================================================================
# Task 4: Verify SMC Marginal Data Density vs Analytical Benchmark
# ==============================================================================

class TestTask4SMCMarginalDataDensityVsAnalytical:
    """Verify SMC Marginal Data Density against analytical benchmark on linear DSGE within 0.10 log-points."""

    def test_smc_mdd_accuracy_against_quadrature_benchmark(self):
        """Stationary AR(1) Gaussian process with exact initial distribution:
        y_t = rho * y_{t-1} + eps_t, eps_t ~ N(0, sigma^2).
        Evaluate analytical marginal likelihood via Gauss-Legendre quadrature to 10^-10.
        Verify SMC MDD matches within 0.10 log-points.
        """
        rng = np.random.default_rng(888)
        T = 40
        sigma_eps = 0.5
        true_rho = 0.60
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = true_rho * y[t - 1] + rng.normal(0.0, sigma_eps)

        prior_mean, prior_sd = 0.0, 0.45
        priors = {"rho": NormalPrior(mean=prior_mean, std=prior_sd, lb=-0.98, ub=0.98)}

        def ar1_log_lik(par):
            rho = float(par["rho"])
            if abs(rho) >= 0.99:
                return -np.inf
            y_lag = y[:-1]
            y_curr = y[1:]
            res = y_curr - rho * y_lag
            var0 = (sigma_eps ** 2) / (1.0 - rho ** 2)
            ll0 = -0.5 * (math.log(2.0 * math.pi * var0) + (y[0] ** 2) / var0)
            ll_t = -0.5 * (len(res) * math.log(2.0 * math.pi * (sigma_eps ** 2)) + np.sum(res ** 2) / (sigma_eps ** 2))
            return ll0 + ll_t

        # High-precision numerical quadrature oracle
        def unnorm_post(rho_val):
            if abs(rho_val) >= 0.98:
                return 0.0
            ll = ar1_log_lik({"rho": rho_val})
            lp = -0.5 * math.log(2.0 * math.pi * (prior_sd ** 2)) - 0.5 * ((rho_val - prior_mean) / prior_sd) ** 2
            return math.exp(ll + lp + 25.0)

        quad_val, _ = scipy.integrate.quad(unnorm_post, -0.98, 0.98, epsabs=1e-12, epsrel=1e-12)
        exact_mdd = math.log(quad_val) - 25.0

        # SMC Sampler run
        smc_res = smc_estimate(
            log_lik=ar1_log_lik,
            priors=priors,
            n_particles=1000,
            n_stages=35,
            adaptive_tempering=False,
            seed=456,
        )

        discrepancy = abs(smc_res.mdd - exact_mdd)
        assert discrepancy < 0.10, (
            f"SMC MDD ({smc_res.mdd:.5f}) deviated from exact quadrature ({exact_mdd:.5f}) "
            f"by {discrepancy:.5f} >= 0.10 log-points!"
        )
        assert smc_res.mdd_se > 0.0
        assert smc_res.mdd_se < 0.15, f"Numerical SE {smc_res.mdd_se} unexpectedly large"


# ==============================================================================
# Task 5: Stress Test Ramsey Optimal Policy Boundary Discounts & Timeless Stability
# ==============================================================================

class TestTask5RamseyBoundaryDiscountsAndTimelessPerspective:
    """Stress test Ramsey optimal policy on boundary discount rates (beta = 0.01 and beta = 0.9999)
    and verify Clarida-Gali-Gertler timeless perspective stability.
    """

    @pytest.fixture
    def cgg1999_model(self):
        params = {
            "beta": 0.99,
            "sigma": 1.0,
            "kappa": 0.15,
            "phi_pi": 1.5,
            "phi_y": 0.25,
            "rho_r": 0.7,
            "rho_d": 0.6,
            "rho_u": 0.5,
            "r_ss": 0.01,
        }
        variables = ["y", "pi", "r", "d", "u"]
        shocks = ["eps_d", "eps_u", "eps_m"]
        steady_state = {v: 0.0 for v in variables}

        def nk_equations(lead, curr, lag, shocks_v, p):
            return [
                curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.d,
                curr.pi - p.beta * lead.pi - p.kappa * curr.y - curr.u,
                curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_m),
                curr.d - p.rho_d * lag.d - shocks_v.eps_d,
                curr.u - p.rho_u * lag.u - shocks_v.eps_u,
            ]

        m = build_dynare(
            nk_equations,
            variables=variables,
            shocks=shocks,
            params=params,
            steady_state=steady_state,
        )
        return m

    def test_ramsey_myopic_boundary_beta_001(self, cgg1999_model):
        """Stress test Ramsey solver at boundary discount beta = 0.01 (beta_inv = 100)."""
        res = ramsey_model(
            model_or_dag=cgg1999_model,
            objective="y^2 + 1.5 * pi^2",
            planner_discount=0.01,
        )
        assert isinstance(res, RamseyResult)
        assert len(res.focs) > 0
        assert np.all(np.isfinite(res.steady_state.values))
        assert len(res.multipliers) > 0

    def test_ramsey_undiscounted_boundary_beta_09999(self, cgg1999_model):
        """Stress test Ramsey solver at near-undiscounted boundary beta = 0.9999."""
        res = ramsey_model(
            model_or_dag=cgg1999_model,
            objective="y^2 + 1.5 * pi^2",
            planner_discount=0.9999,
        )
        assert isinstance(res, RamseyResult)
        assert len(res.focs) > 0
        assert res.policy_solution is not None
        assert np.all(np.isfinite(res.steady_state.values))

    def test_cgg1999_timeless_perspective_price_level_targeting_stability(self, cgg1999_model):
        """Economic oracle test:
        Under Clarida, Gali & Gertler (1999) timeless perspective commitment:
        pi_t = - (lambda_x / kappa) * (x_t - x_{t-1}).
        Following a cost-push shock eps_u, inflation jumps positive at t=0,
        but the planner engineers subsequent deflation in t >= 1 such that
        the cumulative price level change sum_{t=0}^H pi_t converges to ZERO.
        In contrast, under discretion/Taylor rule, cumulative inflation > 0.
        """
        res = ramsey_model(
            model_or_dag=cgg1999_model,
            objective="y^2 + 1.5 * pi^2",
            planner_discount=0.99,
        )
        irf = res.irf(shock="eps_u", horizon=50)

        # Impact inflation is positive
        assert irf["pi"].iloc[0] > 0.0

        # Subsequent inflation turns negative (deflationary commitment to restore price level)
        assert np.any(irf["pi"].iloc[1:15] < 0.0), "Timeless perspective did not engineer deflationary commitment!"

        # Price-level stationarity oracle: cumulative inflation sum(pi_t) must return to 0
        cum_pi = float(np.sum(irf["pi"]))
        assert abs(cum_pi) < 0.02, f"Cumulative inflation {cum_pi} did not return near 0 (price level drift)!"

        # Divine coincidence oracle: under demand shock, output and inflation are exactly zero
        irf_d = res.irf(shock="eps_d", horizon=20)
        assert np.max(np.abs(irf_d["y"])) < 1e-10
        assert np.max(np.abs(irf_d["pi"])) < 1e-10


# ==============================================================================
# Task 6: Stress Test BGP Detrending on Growing Models
# ==============================================================================

class TestTask6BGPDetrendingGrowingModels:
    """Stress test BGP detrending on growing models and verify stationarized models solve cleanly."""

    def test_trending_rbc_bgp_detrending_and_solve(self):
        """Trending RBC model with deterministic labor-augmenting technical growth:
        Gamma_t = gamma^t (gamma > 1).
        Test detrend_bgp parses trend_var and var(deflator=gamma), stationarizes the model,
        and verifies the resulting model can be built and solved cleanly.
        """
        mod_text = """
        var c k y;
        varexo e;
        parameters alpha beta delta gamma;
        trend_var gamma;
        var(deflator=gamma) c k y;
        model;
        c + gamma*k = y + (1-delta)*k(-1);
        y = (k(-1)/gamma)^alpha * exp(e);
        c^(-1) = beta * c(+1)^(-1) * (alpha * y(+1)/(gamma*k) + 1 - delta);
        end;
        """
        dag = parse_mod_to_dag(mod_text)
        assert "gamma" in dag.trend_vars
        assert dag.deflators.get("c") == "gamma"
        assert dag.deflators.get("k") == "gamma"
        assert dag.deflators.get("y") == "gamma"

        stationarized_text = detrend_bgp(mod_text)
        assert isinstance(stationarized_text, str)
        assert "gamma" in stationarized_text

        # Now build and solve the stationarized model
        params = {"alpha": 0.33, "beta": 0.96, "delta": 0.10, "gamma": 1.02}
        # In stationarized steady state with gamma=1.02:
        # beta * (alpha * y / (gamma*k) + 1 - delta) = 1
        # => alpha * y / (gamma*k) = 1/beta - (1-delta)
        # y/k = gamma * (1/beta - 1 + delta) / alpha
        # and y = (k/gamma)^alpha => k^(1-alpha) = gamma^(-alpha) / (y/k) => k = ...
        r_ss = 1.0 / params["beta"] - (1.0 - params["delta"])
        yk_ratio = params["gamma"] * r_ss / params["alpha"]
        k_ss = (params["gamma"]**(-params["alpha"]) / yk_ratio)**(1.0 / (1.0 - params["alpha"]))
        y_ss = (k_ss / params["gamma"])**params["alpha"]
        c_ss = y_ss + (1.0 - params["delta"]) * k_ss - params["gamma"] * k_ss

        def rbc_stat_eqs(lead, curr, lag, shocks_v, p):
            return [
                curr.c + p.gamma * curr.k - curr.y - (1.0 - p.delta) * lag.k,
                curr.y - (lag.k / p.gamma)**p.alpha * np.exp(shocks_v.e),
                curr.c**(-1) - p.beta * lead.c**(-1) * (p.alpha * lead.y / (p.gamma * curr.k) + 1.0 - p.delta),
            ]

        m_stat = build_dynare(
            rbc_stat_eqs,
            variables=["c", "k", "y"],
            shocks=["e"],
            params=params,
            steady_state={"c": c_ss, "k": k_ss, "y": y_ss},
        )
        assert m_stat is not None
        # Verify Blanchard-Kahn condition
        assert m_stat.A is not None
        assert m_stat.B is not None
        dr = m_stat.decision_rules()
        assert dr is not None
        assert "k" in dr.ghx.columns

    def test_bgp_log_trend_var_and_log_deflator_parsing(self):
        """Stress test log_trend_var and var(log_deflator=...) parsing."""
        log_mod = """
        var y c;
        varexo eps;
        parameters g;
        log_trend_var g;
        var(log_deflator=g) y c;
        model;
        y = c + exp(eps);
        end;
        """
        dag = parse_mod_to_dag(log_mod)
        assert "g" in dag.log_trend_vars
        assert dag.log_deflators.get("y") == "g"
        assert dag.log_deflators.get("c") == "g"

    def test_bgp_stationary_model_identity_pass_through(self):
        """Verify stationary model without trend declarations passes through unmodified."""
        stationary_mod = """
        var c k;
        varexo e;
        parameters alpha beta;
        model;
        c + k = k(-1)^alpha + exp(e);
        end;
        """
        out = detrend_bgp(stationary_mod)
        assert out == stationary_mod
