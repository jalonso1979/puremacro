"""Adversarial Empirical Stress Tests for puremacro DSGE Particle Filtering.

Executed by challenger_1 subagent to empirically stress-test `puremacro/dsge/particle_filter.py`.

Stress Dimensions:
1. Multi-seed Kalman filter log-likelihood parity at N=10,000 particles (unbiasedness within +/- 1.5 log-points).
2. Auxiliary Particle Filter (APF) under near-zero measurement errors (1e-3 down to 1e-10).
3. Extreme 50-sigma and 100-sigma observation outliers (isolated, joint, and consecutive bursts).
4. Stochastic volatility with near-unit-root persistence (rho_h = 0.999, 0.9999) and heavy volatility shocks.
5. Order-2 and Order-3 pruned DSGE state spaces under fat-tailed Student-t innovations (df=2.1, 3.0).
6. Performance: Verification of zero Python loops over particles via AST analysis and execution timing.
"""
from __future__ import annotations

import ast
import inspect
import math
import time
import numpy as np
import pandas as pd
import pytest
import scipy.linalg

from puremacro.dsge import build_dynare, load_mod
from puremacro.dsge.observation import make_state_space_from_varobs
from puremacro.dsge.particle_filter import (
    ParticleFilterResult,
    StochasticVolatilitySpec,
    particle_filter,
)
from puremacro.state_space import kalman_filter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def canonical_nk_model():
    """3-equation New Keynesian model for parity and outlier testing."""
    params = {
        "beta": 0.99,
        "sigma": 1.0,
        "kappa": 0.15,
        "phi_pi": 1.5,
        "phi_y": 0.25,
        "rho_r": 0.7,
        "rho_d": 0.6,
        "r_ss": 0.01,
    }
    variables = ["y", "pi", "r", "d"]
    shocks = ["eps_d", "eps_m"]
    steady_state = {v: 0.0 for v in variables}

    def nk_equations(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.d,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_m),
            curr.d - p.rho_d * lag.d - shocks_v.eps_d,
        ]

    m = build_dynare(
        nk_equations,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
        check_steady_state=False,
        strict=False,
    )
    return m


@pytest.fixture(scope="module")
def canonical_nk_data():
    """Deterministic synthetic data for exact Kalman comparison."""
    rng = np.random.default_rng(12345)
    T = 20
    dates = pd.date_range("2010-01-01", periods=T, freq="QS")
    y = np.zeros(T)
    pi = np.zeros(T)
    r = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] - 0.15 * r[t - 1] + rng.normal(0, 0.08)
        pi[t] = 0.45 * pi[t - 1] + 0.12 * y[t] + rng.normal(0, 0.08)
        r[t] = 0.65 * r[t - 1] + 0.35 * (1.5 * pi[t] + 0.2 * y[t]) + rng.normal(0, 0.04)
    return pd.DataFrame({"y": y, "pi": pi, "r": r}, index=dates)


@pytest.fixture(scope="module")
def nonlinear_rbc_model():
    """Nonlinear stochastic RBC model with capital accumulation."""
    mod_code = """
    var c k z;
    varexo eps;
    parameters beta alpha delta rho sigma_pref sigma_eps;
    beta = 0.99;
    alpha = 0.33;
    delta = 0.025;
    rho = 0.95;
    sigma_pref = 1.0;
    sigma_eps = 0.01;

    model;
    exp(-sigma_pref * c) - beta * exp(-sigma_pref * c(+1)) * (alpha * exp(z(+1)) * exp((alpha - 1.0) * k) + 1.0 - delta);
    exp(c) + exp(k) - exp(z) * exp(alpha * k(-1)) - (1.0 - delta) * exp(k(-1));
    z - rho * z(-1) - sigma_eps * eps;
    end;

    initval;
    k = 3.8;
    c = 0.8;
    z = 0.0;
    end;

    steady;
    """
    return load_mod(mod_code)


# ---------------------------------------------------------------------------
# 1. Multi-Seed Kalman Parity & Asymptotic Unbiasedness
# ---------------------------------------------------------------------------

class TestEmpiricalKalmanParityMultiSeed:
    """Stress-test Kalman parity across multiple seeds at N=10,000 particles."""

    SEEDS = [42, 101, 202, 303, 404, 777, 999]

    def test_bpf_multi_seed_parity_and_unbiasedness(self, canonical_nk_model, canonical_nk_data):
        """BPF log-likelihood across 7 seeds: all within +/- 1.5 log-points, mean error < 0.5."""
        m = canonical_nk_model
        df = canonical_nk_data
        varobs = ["y", "pi", "r"]
        me = {"y": 0.1, "pi": 0.1, "r": 0.1}

        # Exact analytical Kalman Filter
        ssm = make_state_space_from_varobs(m, varobs, measurement_error=me)
        P_stat = scipy.linalg.solve_discrete_lyapunov(ssm.T, ssm.R @ ssm.Q @ ssm.R.T)
        k_res = kalman_filter(df[varobs].to_numpy(), ssm, P0=P_stat)
        kalman_ll = float(k_res["loglik"])

        errors = []
        for seed in self.SEEDS:
            pf_res = particle_filter(
                m,
                df,
                varobs,
                n_particles=10_000,
                method="bootstrap",
                resampling_method="systematic",
                measurement_error=me,
                seed=seed,
            )
            diff = pf_res.log_likelihood - kalman_ll
            errors.append(diff)
            assert abs(diff) <= 1.5, (
                f"BPF seed={seed} error={diff:.4f} exceeds +/- 1.5 log-point tolerance "
                f"(PF={pf_res.log_likelihood:.4f}, Kalman={kalman_ll:.4f})"
            )

        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))
        max_error = float(np.max(np.abs(errors)))

        # Verify asymptotic unbiasedness
        assert abs(mean_error) < 0.5, f"BPF mean error {mean_error:.4f} is too biased"
        assert max_error <= 1.5, f"BPF max error {max_error:.4f} exceeds 1.5"

    def test_apf_multi_seed_parity_and_unbiasedness(self, canonical_nk_model, canonical_nk_data):
        """APF log-likelihood across 7 seeds: all within +/- 1.5 log-points, mean error < 0.5."""
        m = canonical_nk_model
        df = canonical_nk_data
        varobs = ["y", "pi", "r"]
        me = {"y": 0.1, "pi": 0.1, "r": 0.1}

        ssm = make_state_space_from_varobs(m, varobs, measurement_error=me)
        P_stat = scipy.linalg.solve_discrete_lyapunov(ssm.T, ssm.R @ ssm.Q @ ssm.R.T)
        k_res = kalman_filter(df[varobs].to_numpy(), ssm, P0=P_stat)
        kalman_ll = float(k_res["loglik"])

        errors = []
        for seed in self.SEEDS:
            pf_res = particle_filter(
                m,
                df,
                varobs,
                n_particles=10_000,
                method="auxiliary",
                resampling_method="systematic",
                measurement_error=me,
                seed=seed,
            )
            diff = pf_res.log_likelihood - kalman_ll
            errors.append(diff)
            assert abs(diff) <= 1.5, (
                f"APF seed={seed} error={diff:.4f} exceeds +/- 1.5 log-point tolerance "
                f"(PF={pf_res.log_likelihood:.4f}, Kalman={kalman_ll:.4f})"
            )

        mean_error = float(np.mean(errors))
        max_error = float(np.max(np.abs(errors)))

        assert abs(mean_error) < 0.5, f"APF mean error {mean_error:.4f} is too biased"
        assert max_error <= 1.5, f"APF max error {max_error:.4f} exceeds 1.5"


# ---------------------------------------------------------------------------
# 2. Auxiliary Particle Filter Under Near-Zero Measurement Errors
# ---------------------------------------------------------------------------

class TestEmpiricalNearZeroMeasurementError:
    """Stress-test Auxiliary Particle Filter under extreme near-zero measurement errors."""

    @pytest.mark.parametrize("error_scale", [1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10])
    def test_apf_near_zero_measurement_error_stability(self, canonical_nk_model, canonical_nk_data, error_scale):
        """APF must not raise LinAlgError or produce NaNs under near-zero measurement errors."""
        m = canonical_nk_model
        df = canonical_nk_data
        varobs = ["y", "pi", "r"]
        me = {v: error_scale for v in varobs}

        res = particle_filter(
            m,
            df,
            varobs,
            n_particles=2_000,
            method="auxiliary",
            measurement_error=me,
            ridge=1e-6,
            seed=42,
        )

        assert np.isfinite(res.log_likelihood), f"APF produced non-finite LL with error_scale={error_scale}"
        assert not res.filtered_states.isna().any().any(), "Filtered states contain NaNs"
        assert not np.isinf(res.filtered_states.to_numpy()).any(), "Filtered states contain Infs"
        assert not res.filtered_states_std.isna().any().any(), "Filtered states std contain NaNs"
        assert np.all(res.ess > 0), "ESS must remain strictly positive"
        assert np.all(np.isfinite(res.log_likelihood_contributions)), "LL contributions contain non-finite values"

    def test_bpf_vs_apf_near_zero_error_comparison(self, canonical_nk_model, canonical_nk_data):
        """Both BPF and APF must successfully evaluate under tight error scale (1e-5)."""
        m = canonical_nk_model
        df = canonical_nk_data
        varobs = ["y", "pi", "r"]
        me = {v: 1e-5 for v in varobs}

        res_bpf = particle_filter(m, df, varobs, n_particles=2_000, method="bootstrap", measurement_error=me, seed=42)
        res_apf = particle_filter(m, df, varobs, n_particles=2_000, method="auxiliary", measurement_error=me, seed=42)

        assert np.isfinite(res_bpf.log_likelihood)
        assert np.isfinite(res_apf.log_likelihood)
        assert not res_bpf.filtered_states.isna().any().any()
        assert not res_apf.filtered_states.isna().any().any()


# ---------------------------------------------------------------------------
# 3. Extreme 50-Sigma and 100-Sigma Observation Outliers
# ---------------------------------------------------------------------------

class TestEmpiricalObservationOutliers:
    """Stress-test 50-sigma and 100-sigma outliers, isolated, simultaneous, and consecutive."""

    def test_single_variable_50_and_100_sigma_outliers(self, canonical_nk_model, canonical_nk_data):
        """Inject 50-sigma and 100-sigma shocks on a single variable."""
        m = canonical_nk_model
        varobs = ["y", "pi", "r"]

        for sigma_factor in [50.0, 100.0]:
            df_outlier = canonical_nk_data.copy()
            # Estimate normal standard deviation
            std_y = df_outlier["y"].std()
            df_outlier.iloc[10, 0] = sigma_factor * std_y

            for method in ["bootstrap", "auxiliary"]:
                res = particle_filter(m, df_outlier, varobs, n_particles=1_000, method=method, seed=42)
                assert np.isfinite(res.log_likelihood), f"{method} {sigma_factor}-sigma produced non-finite LL"
                assert not res.filtered_states.isna().any().any(), f"{method} states contain NaNs"
                assert not np.isinf(res.filtered_states.to_numpy()).any(), f"{method} states contain Infs"
                # Check that after the outlier at t=10, filter continues smoothly
                assert np.all(res.ess > 0), "ESS must remain strictly positive"

    def test_simultaneous_multi_variable_100_sigma_outlier(self, canonical_nk_model, canonical_nk_data):
        """Simultaneous 100-sigma shock across all observed variables."""
        m = canonical_nk_model
        varobs = ["y", "pi", "r"]
        df_outlier = canonical_nk_data.copy()
        for col in varobs:
            df_outlier.loc[df_outlier.index[8], col] = 100.0 * max(0.1, df_outlier[col].std())

        for method in ["bootstrap", "auxiliary"]:
            res = particle_filter(m, df_outlier, varobs, n_particles=1_000, method=method, seed=42)
            assert np.isfinite(res.log_likelihood)
            # Low-weight outlier rejuvenation caps incremental log-likelihood and avoids NaNs/Infs
            assert np.isfinite(res.log_likelihood_contributions.iloc[8])
            assert res.ess.iloc[8] > 0

    def test_consecutive_multi_period_outlier_burst(self, canonical_nk_model, canonical_nk_data):
        """Consecutive 3-period catastrophic outlier shockwave."""
        m = canonical_nk_model
        varobs = ["y", "pi", "r"]
        df_outlier = canonical_nk_data.copy()
        # Periods 8, 9, 10 hit by 50-sigma shock
        for t_idx in [8, 9, 10]:
            df_outlier.iloc[t_idx, 0] = 50.0
            df_outlier.iloc[t_idx, 1] = -50.0

        for method in ["bootstrap", "auxiliary"]:
            res = particle_filter(m, df_outlier, varobs, n_particles=1_000, method=method, seed=42)
            assert np.isfinite(res.log_likelihood)
            assert not res.filtered_states.isna().any().any()
            # Verify post-burst recovery: by period 15, states are bounded
            post_burst_y = res.filtered_states.iloc[15:]["y"].to_numpy()
            assert np.all(np.abs(post_burst_y) < 20.0), f"States failed to recover after shock: {post_burst_y}"

    def test_outliers_under_student_t_measurement_errors(self, canonical_nk_model, canonical_nk_data):
        """50-sigma and 100-sigma outliers under Student-t measurement errors."""
        m = canonical_nk_model
        varobs = ["y", "pi", "r"]
        df_outlier = canonical_nk_data.copy()
        df_outlier.iloc[10, 0] = 100.0

        res = particle_filter(
            m,
            df_outlier,
            varobs,
            n_particles=1_000,
            method="bootstrap",
            measurement_dist="student_t",
            measurement_df=3.0,
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)
        assert not res.filtered_states.isna().any().any()


# ---------------------------------------------------------------------------
# 4. Stochastic Volatility Near-Unit-Root & Heavy Volatility Shocks
# ---------------------------------------------------------------------------

class TestEmpiricalStochasticVolatilityStress:
    """Stress-test near-unit-root persistence (rho_h = 0.999, 0.9999) and extreme volatility."""

    @pytest.mark.parametrize("rho_val", [0.99, 0.999, 0.9999])
    def test_near_unit_root_persistence(self, canonical_nk_model, canonical_nk_data, rho_val):
        """Stochastic volatility with rho_h -> 1.0 must not overflow or produce NaNs."""
        m = canonical_nk_model
        df = canonical_nk_data
        varobs = ["y", "pi", "r"]

        sv = StochasticVolatilitySpec(
            rho=rho_val,
            sigma_eta=0.10,
            base_scale=0.01,
            h0=0.0,
        )
        res = particle_filter(
            m,
            df,
            varobs,
            n_particles=1_500,
            stochastic_volatility=sv,
            seed=42,
        )

        assert np.isfinite(res.log_likelihood)
        assert res.filtered_volatility is not None
        assert np.all(res.filtered_volatility.to_numpy() > 0.0)
        assert not res.filtered_volatility.isna().any().any()
        assert not res.filtered_states.isna().any().any()

    def test_heavy_volatility_innovations_and_extreme_initial_shock(self, canonical_nk_model, canonical_nk_data):
        """Heavy volatility shocks (sigma_eta = 1.0) and extreme initial log-volatility (h0 = +3.0)."""
        m = canonical_nk_model
        df = canonical_nk_data
        varobs = ["y", "pi", "r"]

        # h0 = 3.0 means initial shock std is exp(3.0) ~ 20.08 times base scale
        sv_surge = StochasticVolatilitySpec(
            rho=0.95,
            sigma_eta=1.0,
            base_scale=0.01,
            h0=3.0,
        )
        res_surge = particle_filter(
            m,
            df,
            varobs,
            n_particles=1_500,
            stochastic_volatility=sv_surge,
            seed=42,
        )
        assert np.isfinite(res_surge.log_likelihood)
        assert not res_surge.filtered_states.isna().any().any()
        assert np.all(res_surge.filtered_volatility.to_numpy() > 0.0)

        # h0 = -4.0 means initial shock std is exp(-4.0) ~ 0.018 times base scale (near-quenched)
        sv_quenched = StochasticVolatilitySpec(
            rho=0.95,
            sigma_eta=0.5,
            base_scale=0.01,
            h0=-4.0,
        )
        res_quenched = particle_filter(
            m,
            df,
            varobs,
            n_particles=1_500,
            stochastic_volatility=sv_quenched,
            seed=42,
        )
        assert np.isfinite(res_quenched.log_likelihood)
        assert not res_quenched.filtered_states.isna().any().any()


# ---------------------------------------------------------------------------
# 5. Order-2 and Order-3 Pruned State Spaces with Fat-Tailed Innovations
# ---------------------------------------------------------------------------

class TestEmpiricalPrunedStateSpacesFatTails:
    """Stress-test Order-2 and Order-3 pruned solutions under fat-tailed Student-t innovations."""

    @pytest.mark.parametrize("df_val", [2.1, 3.0, 5.0])
    def test_order2_pruned_with_student_t_innovations(self, nonlinear_rbc_model, df_val):
        """Order-2 pruned model with Student-t structural innovations."""
        m = nonlinear_rbc_model
        sol2 = m.solve(order=2)
        sim = sol2.simulate(periods=20, seed=123).to_frame() + sol2.steady_state
        varobs = ["c", "k"]

        res = particle_filter(
            sol2,
            sim,
            varobs,
            n_particles=1_500,
            method="bootstrap",
            innovation_dist="student_t",
            innovation_df=df_val,
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)
        assert not res.filtered_states.isna().any().any()
        assert not np.isinf(res.filtered_states.to_numpy()).any()

    @pytest.mark.parametrize("df_val", [2.5, 4.0])
    def test_order3_pruned_with_student_t_innovations(self, nonlinear_rbc_model, df_val):
        """Order-3 pruned model with Student-t structural innovations."""
        m = nonlinear_rbc_model
        sol3 = m.solve(order=3)
        sim = sol3.simulate(periods=20, seed=456).to_frame() + sol3.steady_state
        varobs = ["c", "k"]

        res = particle_filter(
            sol3,
            sim,
            varobs,
            n_particles=1_500,
            method="bootstrap",
            innovation_dist="student_t",
            innovation_df=df_val,
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)
        assert not res.filtered_states.isna().any().any()
        assert not np.isinf(res.filtered_states.to_numpy()).any()

    def test_order2_pruned_apf_with_mixture_innovations(self, nonlinear_rbc_model):
        """Auxiliary Particle Filter on Order-2 pruned model with Gaussian mixture innovations."""
        m = nonlinear_rbc_model
        sol2 = m.solve(order=2)
        sim = sol2.simulate(periods=20, seed=789).to_frame() + sol2.steady_state
        varobs = ["c", "k"]

        res = particle_filter(
            sol2,
            sim,
            varobs,
            n_particles=1_500,
            method="auxiliary",
            innovation_dist="mixture",
            mixture_params={"p_crisis": 0.10, "scale_crisis": 6.0},
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)
        assert not res.filtered_states.isna().any().any()


# ---------------------------------------------------------------------------
# 6. Performance & Zero Python Loops Over Particles Verification
# ---------------------------------------------------------------------------

class TestEmpiricalPerformanceAndLooplessness:
    """Empirically verify zero Python loops over particles and vectorized throughput."""

    def test_ast_verification_zero_particle_loops(self):
        """Static AST analysis: assert zero Python loops over particles in particle_filter.py."""
        import puremacro.dsge.particle_filter as pf_mod

        src = inspect.getsource(pf_mod)
        tree = ast.parse(src)

        # Locate the particle_filter function definition
        pf_func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "particle_filter":
                pf_func_def = node
                break

        assert pf_func_def is not None, "Could not find particle_filter in AST"

        # Walk all loops inside particle_filter
        forbidden_loop_targets = []
        for child in ast.walk(pf_func_def):
            if isinstance(child, (ast.For, ast.While)):
                # Inspect loop target name or call
                if isinstance(child, ast.For) and isinstance(child.target, ast.Name):
                    var_name = child.target.id
                    # Loop over t (periods) or burn_in or variable names is allowed
                    # Loop over particle indices (e.g. i, p, part) iterating over n_particles is forbidden
                    if isinstance(child.iter, ast.Call) and isinstance(child.iter.func, ast.Name):
                        if child.iter.func.id == "range" and child.iter.args:
                            arg0 = child.iter.args[0]
                            if isinstance(arg0, ast.Name) and "particle" in arg0.id.lower():
                                forbidden_loop_targets.append(f"for {var_name} in range({arg0.id})")

        assert len(forbidden_loop_targets) == 0, (
            f"Found forbidden Python loop(s) over particles: {forbidden_loop_targets}"
        )

    def test_vectorized_throughput_and_scaling(self, canonical_nk_model, canonical_nk_data):
        """Empirically measure runtime across particle counts: assert N=10,000 runs in < 0.5s."""
        m = canonical_nk_model
        df = canonical_nk_data
        varobs = ["y", "pi", "r"]

        timings = {}
        for N in [1_000, 5_000, 10_000, 20_000]:
            t0 = time.perf_counter()
            res = particle_filter(
                m,
                df,
                varobs,
                n_particles=N,
                method="bootstrap",
                seed=42,
            )
            elapsed = time.perf_counter() - t0
            timings[N] = elapsed
            assert np.isfinite(res.log_likelihood)

        # Assertion: N=10,000 over T=20 must execute in < 0.50 seconds
        assert timings[10_000] < 0.50, (
            f"N=10,000 execution took {timings[10_000]:.3f}s (exceeds 0.50s threshold)"
        )
        # Assertion: N=20,000 over T=20 must execute in < 1.00 second
        assert timings[20_000] < 1.00, (
            f"N=20,000 execution took {timings[20_000]:.3f}s (exceeds 1.00s threshold)"
        )
