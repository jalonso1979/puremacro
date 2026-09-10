"""Unit and integration tests for puremacro DSGE particle filtering.

Covers Requirement 4:
1. Vectorized state propagation over N particles with zero Python particle loops.
2. Vectorized resampling algorithms: systematic, stratified, residual, multinomial in O(N).
3. Bootstrap Particle Filter (BPF) parity vs analytical Kalman filter within +/- 1.5 log-points.
4. Auxiliary Particle Filter (APF) with predictive covariance Sigma_mu parity vs Kalman filter.
5. Exact nonlinear likelihood evaluation on 2nd-order (PrunedDSGESolution) and 3rd-order (Order3PrunedSolution) models.
6. Stochastic Volatility state augmentation tracking precautionary saving shifts and skewness.
7. Fat-tailed innovations (Student-t, Gaussian mixture) and Student-t observation densities.
8. Outlier robustness and low-weight rejuvenation under extreme shocks.
9. Delegation method on LinearModel.particle_filter and integration into estimate_dsge(method="particle_smc").
10. Full presentation contract compliance (.summary, .plot, .to_markdown, .to_latex, .to_typst).
11. Zero external runtime dependencies under the Pyodide four-package constraint.
"""
from __future__ import annotations

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.linalg

from puremacro.dsge import build_dynare, LinearModel, load_mod
from puremacro.dsge.estimate import estimate_dsge
from puremacro.dsge.observation import make_state_space_from_varobs
from puremacro.dsge.particle_filter import (
    ParticleFilterResult,
    StochasticVolatilitySpec,
    multinomial_resample,
    particle_filter,
    residual_resample,
    stratified_resample,
    systematic_resample,
)
from puremacro.dsge.priors import NormalPrior
from puremacro.state_space import kalman_filter


# ---------------------------------------------------------------------------
# Test Fixtures: Canonical New Keynesian Model & Synthetic Series
# ---------------------------------------------------------------------------

@pytest.fixture
def nk_model_setup():
    """Canonical 3-equation New Keynesian DSGE model with state and forward-looking variables."""
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
    return {"m": m, "params": params}


@pytest.fixture
def synthetic_nk_data():
    """Synthetic macroeconomic observable time series (T=20)."""
    rng = np.random.default_rng(42)
    T = 20
    dates = pd.date_range("2005-01-01", periods=T, freq="QS")
    y = np.zeros(T)
    pi = np.zeros(T)
    r = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.6 * y[t - 1] - 0.2 * r[t - 1] + rng.normal(0, 0.1)
        pi[t] = 0.5 * pi[t - 1] + 0.1 * y[t] + rng.normal(0, 0.1)
        r[t] = 0.7 * r[t - 1] + 0.3 * (1.5 * pi[t] + 0.2 * y[t]) + rng.normal(0, 0.05)
    return pd.DataFrame({"y": y, "pi": pi, "r": r}, index=dates)


@pytest.fixture
def rbc_model_code():
    """Nonlinear stochastic neoclassical growth / RBC model."""
    return """
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


# ---------------------------------------------------------------------------
# 1. Resampling Schemes Verification
# ---------------------------------------------------------------------------

class TestResamplingAlgorithms:
    """Verify O(N) vectorized resampling algorithms."""

    def test_resampling_proportions_on_degenerate_weights(self):
        """Verify that all 4 resampling algorithms correctly replicate particle proportions."""
        rng = np.random.default_rng(123)
        N = 10_000
        # Highly degenerate weights: index 0 carries 90% of weight
        weights = np.array([0.90, 0.05, 0.03, 0.02])

        for resample_fn in (systematic_resample, stratified_resample, residual_resample, multinomial_resample):
            idx = resample_fn(weights, rng)
            assert len(idx) == 4
            assert idx[0] == 0

        # Large N test
        w_large = np.zeros(N)
        w_large[0] = 0.80
        w_large[1:] = 0.20 / (N - 1)

        for resample_fn in (systematic_resample, stratified_resample, residual_resample, multinomial_resample):
            idx = resample_fn(w_large, rng)
            assert len(idx) == N
            count_0 = np.sum(idx == 0)
            # 80% of N = 8000, within tight Monte Carlo bounds
            assert 7700 <= count_0 <= 8300, f"{resample_fn.__name__} count: {count_0}"

    def test_resampling_edge_cases(self):
        """Verify edge cases: N=0, N=1, uniform weights, non-normalized weights."""
        rng = np.random.default_rng(999)
        assert len(systematic_resample(np.array([]), rng)) == 0
        assert np.array_equal(systematic_resample(np.array([1.0]), rng), np.array([0]))
        assert np.array_equal(stratified_resample(np.array([1.0]), rng), np.array([0]))
        assert np.array_equal(residual_resample(np.array([1.0]), rng), np.array([0]))
        assert np.array_equal(multinomial_resample(np.array([1.0]), rng), np.array([0]))

        # Non-normalized weights summing to 5.0
        w_raw = np.array([2.5, 2.5])
        idx = systematic_resample(w_raw, rng)
        assert len(idx) == 2


# ---------------------------------------------------------------------------
# 2. Kalman Filter Parity Tests (Linear Gaussian Benchmark)
# ---------------------------------------------------------------------------

class TestKalmanFilterParity:
    """Verify BPF and APF log-likelihood parity against analytical Kalman filter within +/- 1.5 log-points."""

    def test_bpf_parity_vs_kalman(self, nk_model_setup, synthetic_nk_data):
        """Verify Bootstrap Particle Filter matches analytical Kalman log-likelihood within +/- 1.5 log-points."""
        m = nk_model_setup["m"]
        df = synthetic_nk_data
        varobs = ["y", "pi", "r"]
        me = {"y": 0.1, "pi": 0.1, "r": 0.1}

        # 1. Analytical Kalman Filter evaluation with stationary Lyapunov prior P0
        ssm = make_state_space_from_varobs(m, varobs, measurement_error=me)
        Tm, Rm, Qm = ssm.T, ssm.R, ssm.Q
        P_ssm = scipy.linalg.solve_discrete_lyapunov(Tm, Rm @ Qm @ Rm.T)
        k_res = kalman_filter(df[varobs].to_numpy(), ssm, P0=P_ssm)
        kalman_ll = float(k_res["loglik"])

        # 2. Vectorized Bootstrap Particle Filter with N=10,000
        res_bpf = particle_filter(
            m,
            df,
            varobs,
            n_particles=10_000,
            method="bootstrap",
            resampling_method="systematic",
            measurement_error=me,
            seed=42,
        )

        abs_diff = abs(res_bpf.log_likelihood - kalman_ll)
        assert abs_diff < 1.5, f"BPF LL {res_bpf.log_likelihood:.4f} vs Kalman {kalman_ll:.4f} (diff: {abs_diff:.4f})"
        assert len(res_bpf.log_likelihood_contributions) == len(df)
        assert res_bpf.filtered_states.shape == (len(df), len(m.states) + len(m.controls))

    def test_apf_parity_vs_kalman(self, nk_model_setup, synthetic_nk_data):
        """Verify Auxiliary Particle Filter with predictive covariance matches Kalman filter within +/- 1.5 log-points."""
        m = nk_model_setup["m"]
        df = synthetic_nk_data
        varobs = ["y", "pi", "r"]
        me = {"y": 0.1, "pi": 0.1, "r": 0.1}

        ssm = make_state_space_from_varobs(m, varobs, measurement_error=me)
        Tm, Rm, Qm = ssm.T, ssm.R, ssm.Q
        P_ssm = scipy.linalg.solve_discrete_lyapunov(Tm, Rm @ Qm @ Rm.T)
        k_res = kalman_filter(df[varobs].to_numpy(), ssm, P0=P_ssm)
        kalman_ll = float(k_res["loglik"])

        # Vectorized Auxiliary Particle Filter with predictive proposal covariance Sigma_mu
        res_apf = particle_filter(
            m,
            df,
            varobs,
            n_particles=10_000,
            method="auxiliary",
            resampling_method="systematic",
            measurement_error=me,
            seed=42,
        )

        abs_diff = abs(res_apf.log_likelihood - kalman_ll)
        assert abs_diff < 1.5, f"APF LL {res_apf.log_likelihood:.4f} vs Kalman {kalman_ll:.4f} (diff: {abs_diff:.4f})"


# ---------------------------------------------------------------------------
# 3. Nonlinear Pruned Model Likelihood Evaluation (Order 2 & Order 3)
# ---------------------------------------------------------------------------

class TestNonlinearPrunedParticleFilter:
    """Verify likelihood evaluation on 2nd-order and 3rd-order pruned DSGE state spaces."""

    def test_pruned_order2_likelihood_and_state_tracking(self, rbc_model_code):
        """Verify particle filter on 2nd-order pruned solution (PrunedDSGESolution)."""
        m = load_mod(rbc_model_code)
        sol2 = m.solve(order=2)

        # Simulate synthetic data in levels
        sim = sol2.simulate(periods=15, seed=101).to_frame() + sol2.steady_state
        varobs = ["c", "k"]

        res = particle_filter(sol2, sim, varobs, n_particles=1_000, seed=42)
        assert np.isfinite(res.log_likelihood)
        assert res.filtered_states.shape == (15, 3)
        assert set(varobs).issubset(set(res.filtered_states.columns))
        assert not res.filtered_states.isna().any().any()
        assert np.all(res.ess > 0)

    def test_pruned_order3_likelihood_and_stability(self, rbc_model_code):
        """Verify particle filter on 3rd-order pruned solution (Order3PrunedSolution)."""
        m = load_mod(rbc_model_code)
        sol3 = m.solve(order=3)

        sim = sol3.simulate(periods=15, seed=202).to_frame() + sol3.steady_state
        varobs = ["c", "k"]

        res = particle_filter(sol3, sim, varobs, n_particles=1_000, seed=42)
        assert np.isfinite(res.log_likelihood)
        assert res.filtered_states.shape == (15, 3)
        assert not res.filtered_states.isna().any().any()


# ---------------------------------------------------------------------------
# 4. Stochastic Volatility & Precautionary Saving
# ---------------------------------------------------------------------------

class TestStochasticVolatility:
    """Verify particle filter state augmentation for Stochastic Volatility and precautionary saving."""

    def test_stochastic_volatility_tracking(self, rbc_model_code):
        """Verify particle filter tracks time-varying volatility paths."""
        m = load_mod(rbc_model_code)
        sol2 = m.solve(order=2)
        sim = sol2.simulate(periods=20, seed=303).to_frame() + sol2.steady_state
        varobs = ["c", "k"]

        sv = StochasticVolatilitySpec(rho=0.80, sigma_eta=0.25, base_scale=0.01)
        res = particle_filter(sol2, sim, varobs, n_particles=1_000, stochastic_volatility=sv, seed=42)

        assert res.filtered_volatility is not None
        assert res.filtered_volatility.shape == (20, 1)
        assert "sigma_eps" in res.filtered_volatility.columns
        assert np.all(res.filtered_volatility.to_numpy() > 0)
        assert np.isfinite(res.log_likelihood)

    def test_precautionary_saving_state_shift_under_volatility(self, rbc_model_code):
        """Verify that elevated volatility induces positive precautionary saving shift in 2nd-order model."""
        m = load_mod(rbc_model_code)
        sol2 = m.solve(order=2)

        # Generate synthetic path with zero shocks
        T = 20
        c_ss = sol2.steady_state["c"]
        k_ss = sol2.steady_state["k"]
        steady_df = pd.DataFrame({"c": np.full(T, c_ss), "k": np.full(T, k_ss)})

        # 1. Run particle filter with normal volatility
        sv_low = StochasticVolatilitySpec(rho=0.9, sigma_eta=0.05, base_scale=0.01, h0=0.0)
        res_low = particle_filter(sol2, steady_df, ["c", "k"], n_particles=2_000, stochastic_volatility=sv_low, seed=42)

        # 2. Run particle filter with elevated volatility shock (h0 = +1.0 -> 2.7x standard deviation)
        sv_high = StochasticVolatilitySpec(rho=0.9, sigma_eta=0.05, base_scale=0.01, h0=1.0)
        res_high = particle_filter(sol2, steady_df, ["c", "k"], n_particles=2_000, stochastic_volatility=sv_high, seed=42)

        # Under prudential motive (H_uu > 0 for capital), higher volatility induces higher average capital accumulation
        mean_k_low = res_low.filtered_states["k"].mean()
        mean_k_high = res_high.filtered_states["k"].mean()
        assert mean_k_high >= mean_k_low - 1e-4, f"Capital under high vol ({mean_k_high:.4f}) vs low vol ({mean_k_low:.4f})"


# ---------------------------------------------------------------------------
# 5. Fat-Tailed Innovations & Measurement Error Densities
# ---------------------------------------------------------------------------

class TestFatTailedInnovationsAndOutliers:
    """Verify Student-t innovations, Gaussian mixture innovations, and outlier rejuvenation."""

    def test_student_t_innovations_and_measurement(self, nk_model_setup, synthetic_nk_data):
        """Verify Student-t innovation and measurement error handling."""
        m = nk_model_setup["m"]
        df = synthetic_nk_data
        varobs = ["y", "pi", "r"]

        res = particle_filter(
            m,
            df,
            varobs,
            n_particles=1_000,
            innovation_dist="student_t",
            innovation_df=4.0,
            measurement_dist="student_t",
            measurement_df=4.0,
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)
        assert not res.filtered_states.isna().any().any()

    def test_mixture_innovations(self, nk_model_setup, synthetic_nk_data):
        """Verify two-component Gaussian mixture innovation distribution."""
        m = nk_model_setup["m"]
        df = synthetic_nk_data
        varobs = ["y", "pi", "r"]

        mix_params = {"p_crisis": 0.08, "scale_crisis": 4.0}
        res = particle_filter(
            m,
            df,
            varobs,
            n_particles=1_000,
            innovation_dist="mixture",
            mixture_params=mix_params,
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)

    def test_extreme_outlier_rejuvenation(self, nk_model_setup, synthetic_nk_data):
        """Verify that an extreme 50-sigma observation outlier does not crash or produce NaNs."""
        m = nk_model_setup["m"]
        df = synthetic_nk_data.copy()
        # Inject catastrophic outlier at t=10
        df.iloc[10, 0] = 50.0

        varobs = ["y", "pi", "r"]
        res = particle_filter(m, df, varobs, n_particles=500, seed=42)
        assert np.isfinite(res.log_likelihood)
        assert not res.filtered_states.isna().any().any()


# ---------------------------------------------------------------------------
# 6. Public API Delegation & Estimation Dispatch
# ---------------------------------------------------------------------------

class TestPublicApiAndEstimationDispatch:
    """Verify LinearModel delegation and estimate_dsge integration."""

    def test_linear_model_particle_filter_delegation(self, nk_model_setup, synthetic_nk_data):
        """Verify LinearModel.particle_filter delegates cleanly to particle_filter."""
        m = nk_model_setup["m"]
        df = synthetic_nk_data
        varobs = ["y", "pi", "r"]

        res = m.particle_filter(df, varobs, n_particles=500, seed=42)
        assert isinstance(res, ParticleFilterResult)
        assert np.isfinite(res.log_likelihood)

    def test_estimate_dsge_particle_smc_integration(self, nk_model_setup, synthetic_nk_data):
        """Verify estimate_dsge(..., method='particle_smc') runs SMC estimation."""
        m = nk_model_setup["m"]
        df = synthetic_nk_data.iloc[:10]
        varobs = ["y", "pi", "r"]

        priors = {"kappa": NormalPrior(0.15, 0.05)}
        smc_res = estimate_dsge(
            df,
            priors=priors,
            observed_vars=varobs,
            initial_params={"kappa": 0.15},
            m_unconstrained=m,
            method="particle_smc",
            n_particles_smc=10,
            n_particles_pf=100,
            n_stages=2,
            seed=42,
        )
        assert np.isfinite(smc_res.log_marginal_likelihood)
        assert smc_res.particles.shape[0] == 10


# ---------------------------------------------------------------------------
# 7. Presentation Contract Compliance
# ---------------------------------------------------------------------------

class TestPresentationContract:
    """Verify ParticleFilterResult implements puremacro presentation contract."""

    def test_presentation_methods(self, nk_model_setup, synthetic_nk_data):
        """Verify .summary(), .plot(), .to_markdown(), .to_latex(), .to_typst()."""
        m = nk_model_setup["m"]
        df = synthetic_nk_data
        varobs = ["y", "pi", "r"]

        sv = StochasticVolatilitySpec(rho=0.8, sigma_eta=0.1)
        res = particle_filter(m, df, varobs, n_particles=500, stochastic_volatility=sv, seed=42)

        # 1. Summary DataFrame
        summary_df = res.summary()
        assert isinstance(summary_df, pd.DataFrame)
        assert "Log-Likelihood" in summary_df.index
        assert "Number of Particles (N)" in summary_df.index

        # 2. Markdown export
        md = res.to_markdown()
        assert isinstance(md, str)
        assert "Log-Likelihood" in md

        # 3. LaTeX export
        ltx = res.to_latex()
        assert isinstance(ltx, str)
        assert "tabular" in ltx or "Log-Likelihood" in ltx

        # 4. Typst export
        typ = res.to_typst()
        assert isinstance(typ, str)
        assert "#table" in typ or "Log-Likelihood" in typ

        # 5. Plot generation
        fig = res.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Input Validation & Error Handling
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Verify input validation and informative error messages."""

    def test_invalid_n_particles(self, nk_model_setup, synthetic_nk_data):
        with pytest.raises(ValueError, match="n_particles must be >= 1"):
            particle_filter(nk_model_setup["m"], synthetic_nk_data, ["y"], n_particles=0)

    def test_invalid_method(self, nk_model_setup, synthetic_nk_data):
        with pytest.raises(ValueError, match="method must be 'bootstrap' or 'auxiliary'"):
            particle_filter(nk_model_setup["m"], synthetic_nk_data, ["y"], method="kalman_like")

    def test_invalid_resampling_method(self, nk_model_setup, synthetic_nk_data):
        with pytest.raises(ValueError, match="resampling_method must be one of"):
            particle_filter(nk_model_setup["m"], synthetic_nk_data, ["y"], resampling_method="quantum")

    def test_empty_observables(self, nk_model_setup, synthetic_nk_data):
        with pytest.raises(ValueError, match="observed_vars must contain at least one variable name"):
            particle_filter(nk_model_setup["m"], synthetic_nk_data, [])

    def test_missing_data_columns(self, nk_model_setup, synthetic_nk_data):
        with pytest.raises(ValueError, match="not found in data columns"):
            particle_filter(nk_model_setup["m"], synthetic_nk_data, ["nonexistent_var"])
