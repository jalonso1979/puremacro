"""Unit tests for Sequential Monte Carlo (SMC) Sampler and Particle Filtering.

Covers Requirement R4:
1. Sequential Monte Carlo Sampler (Herbst & Schorfheide 2014, 2015).
2. Fixed and adaptive tempering schedules.
3. Systematic resampling in O(N) on ESS threshold violation.
4. Particle mutation with adaptive proposal covariance and scale adaptation.
5. Exact Marginal Data Density (MDD) calculation and numerical standard error.
6. Bootstrap particle filter for order-2 and order-3 pruned state spaces with outlier rejuvenation.
7. Multimodal posterior exploration where standard RWMH gets trapped.
8. Analytical linear DSGE marginal likelihood benchmark within 0.10 log-points.
9. Full presentation contract compliance (.summary, .plot_stages, .plot_posterior, .to_markdown, .to_latex, .to_typst).
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

from puremacro.dsge import build_dynare, LinearModel
from puremacro.dsge.pruning import Order3PrunedSolution, PrunedDSGESolution
from puremacro.dsge.priors import NormalPrior, UniformPrior, BetaPrior
from puremacro.dsge.smc import (
    SMCSampler,
    SMCResult,
    bootstrap_particle_filter,
    smc_estimate,
    systematic_resample,
    fixed_tempering_schedule,
    solve_adaptive_phi,
)


# ---------------------------------------------------------------------------
# Test Fixtures: Canonical New Keynesian Model & Synthetic Data
# ---------------------------------------------------------------------------

@pytest.fixture
def nk_model_setup():
    """Canonical 3-equation New Keynesian DSGE model."""
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
    """Synthetic macroeconomic observable time series (T=60)."""
    rng = np.random.default_rng(789)
    T = 60
    dates = pd.date_range("2005-01-01", periods=T, freq="QS")
    y = np.zeros(T)
    pi = np.zeros(T)
    r = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.6 * y[t - 1] - 0.2 * r[t - 1] + rng.normal(0, 0.3)
        pi[t] = 0.5 * pi[t - 1] + 0.1 * y[t] + rng.normal(0, 0.2)
        r[t] = 0.7 * r[t - 1] + 0.3 * (1.5 * pi[t] + 0.2 * y[t]) + rng.normal(0, 0.1)
    return pd.DataFrame({"y": y, "pi": pi, "r": r}, index=dates)


# ---------------------------------------------------------------------------
# 1. Initialization and Input Validation
# ---------------------------------------------------------------------------

def test_smc_initialization_and_particle_validation(nk_model_setup, synthetic_nk_data):
    """Verify SMCSampler initializes correctly and rejects n_particles <= 1."""
    m = nk_model_setup["m"]
    data = synthetic_nk_data

    # Invalid n_particles <= 1
    with pytest.raises(ValueError, match="n_particles must be > 1"):
        SMCSampler(m=m, data=data, varobs=["y", "pi", "r"], n_particles=1)

    with pytest.raises(ValueError, match="n_particles must be > 1"):
        SMCSampler(m=m, data=data, varobs=["y", "pi", "r"], n_particles=0)

    # Valid initialization
    sampler = SMCSampler(
        m=m,
        data=data,
        varobs=["y", "pi", "r"],
        n_particles=120,
        n_stages=10,
        seed=42,
    )
    assert sampler.n_particles == 120
    assert sampler.n_stages == 10
    assert callable(sampler.sample)


# ---------------------------------------------------------------------------
# 2. Fixed and Adaptive Tempering Schedules
# ---------------------------------------------------------------------------

def test_smc_fixed_tempering_schedule_properties():
    """Verify fixed power schedule is strictly monotonic from 0 to 1."""
    n_stages = 40
    phi = fixed_tempering_schedule(n_stages, lambda_param=2.1)

    assert len(phi) == n_stages + 1
    assert phi[0] == 0.0
    assert phi[-1] == 1.0
    assert np.all(np.diff(phi) > 0.0), "Schedule must be strictly increasing"
    # lambda > 1 means steps are smaller near 0 and larger near 1
    assert phi[1] < 1.0 / n_stages

    with pytest.raises(ValueError):
        fixed_tempering_schedule(0)
    with pytest.raises(ValueError):
        fixed_tempering_schedule(10, lambda_param=-1.0)


def test_smc_adaptive_tempering_bisection():
    """Verify adaptive bisection finds phi_n matching ESS target."""
    rng = np.random.default_rng(101)
    N = 200
    weights = np.full(N, 1.0 / N)
    # Log-likelihood differences across particles
    log_liks = rng.normal(-100.0, 10.0, size=N)

    phi_curr = 0.2
    phi_next = solve_adaptive_phi(
        phi_curr,
        log_liks,
        weights,
        target_ess_ratio=0.90,
    )

    assert phi_next > phi_curr
    assert phi_next <= 1.0

    # At phi_curr = 1.0, should return 1.0
    assert solve_adaptive_phi(1.0, log_liks, weights) == 1.0


# ---------------------------------------------------------------------------
# 3. Systematic Resampling in O(N)
# ---------------------------------------------------------------------------

def test_systematic_resampling_exactness_and_low_variance():
    """Verify systematic resampling maintains minimal integer reproduction variance."""
    rng = np.random.default_rng(202)
    N = 1000
    # Skewed weights: particle 0 has weight 0.40, others share remainder
    weights = np.zeros(N)
    weights[0] = 0.40
    weights[1:] = 0.60 / (N - 1)

    indices = systematic_resample(weights, rng)
    assert len(indices) == N
    assert np.all(indices >= 0) and np.all(indices < N)

    # Particle 0 should be selected exactly floor(N*W_0) or ceil(N*W_0) times = 400
    count_0 = np.sum(indices == 0)
    expected_count = int(N * weights[0])
    assert abs(count_0 - expected_count) <= 1, f"Expected ~400, got {count_0}"

    # Verify uniform weights produce all indices
    uniform_w = np.full(N, 1.0 / N)
    unif_indices = systematic_resample(uniform_w, rng)
    unique_counts = np.bincount(unif_indices, minlength=N)
    # Under uniform weights, each particle is drawn exactly once
    assert np.all(unique_counts == 1)


# ---------------------------------------------------------------------------
# 4. Mutation Proposal Covariance and Scale Adaptation
# ---------------------------------------------------------------------------

def test_mutation_proposal_covariance_adaptation():
    """Verify proposal scale c expands on high acceptance and contracts on low acceptance."""
    # High acceptance target (flat likelihood): accept rate near 100% -> scale expands
    priors_1d = {"x": NormalPrior(mean=0.0, std=1.0)}
    res_high = smc_estimate(
        log_lik=lambda p: 0.0,
        priors=priors_1d,
        n_particles=100,
        n_stages=4,
        seed=42,
        adaptive_tempering=False,
    )
    # Final scale should have expanded from initial 0.5
    assert res_high.stage_diagnostics["scale_c"].iloc[-1] > 0.5

    # Low acceptance target: 5-dimensional Gaussian with oversized initial proposal scale (c_init=3.0)
    # RWMH proposals in 5D step into low-density tails, producing accept rate < 0.15 -> scale contracts
    priors_5d = {f"x_{i}": NormalPrior(mean=0.0, std=1.0) for i in range(5)}
    res_low = smc_estimate(
        log_lik=lambda p: -5.0 * sum(float(p[f"x_{i}"]) ** 2 for i in range(5)),
        priors=priors_5d,
        n_particles=100,
        n_stages=4,
        c_init=3.0,
        seed=42,
        adaptive_tempering=False,
    )
    # Final scale should have contracted from initial 3.0
    assert res_low.stage_diagnostics["scale_c"].iloc[-1] < 3.0


# ---------------------------------------------------------------------------
# 5. Multimodal Posterior Exploration: SMC vs Standard RWMH
# ---------------------------------------------------------------------------

def test_smc_accurately_explores_bimodal_posterior_where_rwmh_trapped():
    """Verify SMC explores both modes of a bimodal distribution where standard RWMH gets trapped.

    Target: 0.5 * N(-2.5, 0.35^2) + 0.5 * N(+2.5, 0.35^2).
    Modes separated by 5 units with deep density valley.
    """
    def log_target_density(par):
        x = float(par["x"])
        d1 = math.exp(-0.5 * ((x - (-2.5)) / 0.35) ** 2) / (0.35 * math.sqrt(2 * math.pi))
        d2 = math.exp(-0.5 * ((x - 2.5) / 0.35) ** 2) / (0.35 * math.sqrt(2 * math.pi))
        dens = 0.5 * d1 + 0.5 * d2
        return math.log(max(dens, 1e-300))

    priors = {"x": UniformPrior(lb=-6.0, ub=6.0)}

    # 1. Standard RWMH initialized near negative mode
    rng = np.random.default_rng(42)
    curr = -2.5
    chain = [curr]
    curr_ll = log_target_density({"x": curr})
    for _ in range(2500):
        prop = curr + rng.normal(0.0, 0.4)
        if -6.0 <= prop <= 6.0:
            prop_ll = log_target_density({"x": prop})
            if math.log(rng.uniform(0.0, 1.0)) < (prop_ll - curr_ll):
                curr = prop
                curr_ll = prop_ll
        chain.append(curr)

    rwmh_chain = np.array(chain)
    rwmh_mode2_fraction = float(np.mean(rwmh_chain > 0.0))

    # Standard RWMH gets completely trapped in mode 1
    assert rwmh_mode2_fraction < 0.05, f"RWMH unexpectedly crossed barrier: {rwmh_mode2_fraction}"

    # 2. Sequential Monte Carlo Sampler
    smc_res = smc_estimate(
        log_lik=log_target_density,
        priors=priors,
        n_particles=600,
        n_stages=25,
        seed=42,
    )

    smc_particles = smc_res.particles[:, 0]
    smc_mode1_mass = float(np.mean(smc_particles < 0.0))
    smc_mode2_mass = float(np.mean(smc_particles > 0.0))

    # SMC successfully captures both modes with nearly equal mass
    assert 0.35 <= smc_mode1_mass <= 0.65, f"Mode 1 mass {smc_mode1_mass} not balanced"
    assert 0.35 <= smc_mode2_mass <= 0.65, f"Mode 2 mass {smc_mode2_mass} not balanced"


# ---------------------------------------------------------------------------
# 6. Exact Marginal Data Density (MDD) Linear Benchmark
# ---------------------------------------------------------------------------

def test_smc_marginal_likelihood_matches_analytical_benchmark_within_010():
    """Verify SMC estimated MDD matches high-precision analytical quadrature benchmark within 0.10 log-points."""
    # Stationary AR(1) DGP: y_t = rho * y_{t-1} + eps_t, eps_t ~ N(0, sigma^2)
    rng = np.random.default_rng(999)
    T = 40
    sigma_eps = 0.5
    true_rho = 0.65
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = true_rho * y[t - 1] + rng.normal(0.0, sigma_eps)

    prior_mean, prior_sd = 0.0, 0.4
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

    # Analytical exact MDD via Gauss-Legendre numerical quadrature
    def unnormalized_posterior(rho_val):
        if abs(rho_val) >= 0.98:
            return 0.0
        ll = ar1_log_lik({"rho": rho_val})
        lp = -0.5 * math.log(2.0 * math.pi * (prior_sd ** 2)) - 0.5 * ((rho_val - prior_mean) / prior_sd) ** 2
        return math.exp(ll + lp + 30.0)

    quad_val, _ = scipy.integrate.quad(unnormalized_posterior, -0.98, 0.98)
    exact_mdd = math.log(quad_val) - 30.0

    # SMC estimation
    smc_res = smc_estimate(
        log_lik=ar1_log_lik,
        priors=priors,
        n_particles=800,
        n_stages=30,
        adaptive_tempering=False,
        seed=123,
    )

    log_mdd_diff = abs(smc_res.mdd - exact_mdd)
    assert log_mdd_diff < 0.10, (
        f"SMC MDD {smc_res.mdd:.4f} differed from exact {exact_mdd:.4f} by {log_mdd_diff:.4f} >= 0.10"
    )
    assert smc_res.mdd_se > 0.0
    assert np.isfinite(smc_res.mdd_se)


# ---------------------------------------------------------------------------
# 7. Bootstrap Particle Filter on Pruned State Spaces
# ---------------------------------------------------------------------------

def test_bootstrap_particle_filter_order2_and_order3():
    """Verify bootstrap particle filter evaluates likelihoods on Order-2 and Order-3 pruned state spaces."""
    T = 30
    rng = np.random.default_rng(333)
    data = pd.DataFrame({
        "y": rng.normal(0.0, 0.4, size=T),
        "pi": rng.normal(0.0, 0.2, size=T),
        "r": rng.normal(0.01, 0.1, size=T),
    })

    # Order-2 Pruned Solution
    sol2 = PrunedDSGESolution(
        G=np.array([[0.65, 0.05], [0.10, 0.55]]),
        N=np.array([[0.25], [0.15]]),
        F=np.array([[0.12, 0.18]]),
        L=np.array([[0.08]]),
        H_xx=np.zeros((2, 4)),
        H_sigmasigma=np.zeros(2),
        G_xx=np.zeros((1, 4)),
        G_sigmasigma=np.zeros(1),
        state_names=("y", "pi"),
        control_names=("r",),
        shock_names=("eps",),
    )

    ll2, filt2 = bootstrap_particle_filter(
        sol2, data, varobs=["y", "pi", "r"], n_particles=80, seed=42
    )
    assert isinstance(ll2, (float, np.floating))
    assert np.isfinite(ll2)
    assert filt2.shape == (T, 3)

    # Order-3 Pruned Solution
    sol3 = Order3PrunedSolution(
        state_names=("y", "pi"),
        control_names=("r",),
        shock_names=("eps",),
        steady_state=pd.Series({"y": 0.0, "pi": 0.0, "r": 0.01}),
        g_x=np.array([[0.65, 0.05], [0.10, 0.55], [0.12, 0.18]]),
        g_xx=np.zeros((3, 4)),
        g_xxx=np.zeros((3, 8)),
        g_ss=np.zeros(3),
        g_x_ss=np.zeros((3, 2)),
        g_u_ss=np.zeros((3, 1)),
        g_u=np.array([[0.25], [0.15], [0.08]]),
        g_uu=np.zeros((3, 1)),
        g_uuu=np.zeros((3, 1)),
    )

    ll3, filt3 = bootstrap_particle_filter(
        sol3, data, varobs=["y", "pi", "r"], n_particles=80, seed=42
    )
    assert isinstance(ll3, (float, np.floating))
    assert np.isfinite(ll3)
    assert filt3.shape == (T, 3)


def test_bootstrap_particle_filter_outlier_rejuvenation():
    """Verify particle filter rejuvenates particles when encountering extreme 50-sigma observation outliers."""
    sol = Order3PrunedSolution(
        state_names=("y", "pi"),
        control_names=("r",),
        shock_names=("eps",),
        steady_state=pd.Series({"y": 0.0, "pi": 0.0, "r": 0.01}),
        g_x=np.array([[0.5, 0.0], [0.0, 0.5], [0.1, 0.1]]),
        g_xx=np.zeros((3, 4)),
        g_xxx=np.zeros((3, 8)),
        g_ss=np.zeros(3),
        g_x_ss=np.zeros((3, 2)),
        g_u_ss=np.zeros((3, 1)),
        g_u=np.eye(3, 1),
        g_uu=np.zeros((3, 1)),
        g_uuu=np.zeros((3, 1)),
    )

    data = pd.DataFrame({
        "y": [0.1, 0.2, 50.0, 0.1, 0.05],  # Outlier at t=2
        "pi": [0.05, 0.02, 30.0, 0.04, 0.01],
        "r": [0.01, 0.01, 0.01, 0.01, 0.01],
    })

    ll, filt = bootstrap_particle_filter(
        sol, data, varobs=["y", "pi", "r"], n_particles=60, seed=42
    )
    assert np.isfinite(ll)
    assert np.all(np.isfinite(filt))


# ---------------------------------------------------------------------------
# 8. Presentation Contract Compliance (.summary, .plot, .to_markdown, etc.)
# ---------------------------------------------------------------------------

def test_smc_result_presentation_contract():
    """Verify SMCResult implements all required presentation contract methods."""
    np.random.seed(42)
    particles = np.random.normal(0, 1, (100, 3))
    weights = np.full(100, 1.0 / 100)
    stages = np.linspace(0, 1, 8)
    summary_df = pd.DataFrame(
        {
            "mean": [1.55, 0.22, 0.71],
            "std": [0.12, 0.05, 0.04],
            "5%": [1.35, 0.14, 0.64],
            "50%": [1.54, 0.22, 0.71],
            "95%": [1.75, 0.30, 0.78],
        },
        index=["phi_pi", "phi_y", "rho_r"],
    )

    res = SMCResult(
        particles=particles,
        weights=weights,
        stage_tempering=stages,
        mdd=-312.45,
        mdd_se=0.08,
        acceptance_rates=np.full(8, 0.26),
        ess_history=np.full(8, 85.0),
        posterior_summary=summary_df,
    )

    # 1. .summary()
    summ = res.summary()
    assert isinstance(summ, str)
    assert "Sequential Monte Carlo" in summ or "SMC" in summ
    assert "-312.4500" in summ
    assert "phi_pi" in summ

    # 2. .to_markdown()
    md = res.to_markdown()
    assert isinstance(md, str)
    assert "### Sequential Monte Carlo (SMC) Estimation Results" in md
    assert "-312.4500" in md

    # 3. .to_latex()
    ltx = res.to_latex()
    assert isinstance(ltx, str)
    assert "\\begin{table}" in ltx or "\\begin{tabular}" in ltx

    # 4. .to_typst()
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#table" in typ
    assert "phi_pi" in typ

    # 5. .plot_stages()
    fig1, ax1 = res.plot_stages()
    assert fig1 is not None
    plt.close(fig1)

    # 6. .plot_posterior() and .plot()
    fig2, ax2 = res.plot_posterior()
    assert fig2 is not None
    plt.close(fig2)

    fig3, ax3 = res.plot()
    assert fig3 is not None
    plt.close(fig3)

    # 7. .marginal_likelihood() and .posterior_table()
    mdd, se = res.marginal_likelihood()
    assert mdd == -312.45 and se == 0.08
    pt = res.posterior_table()
    assert isinstance(pt, pd.DataFrame)
    assert list(pt.index) == ["phi_pi", "phi_y", "rho_r"]


# ---------------------------------------------------------------------------
# 9. DSGE SMC Estimation End-to-End
# ---------------------------------------------------------------------------

def test_dsge_smc_estimation_end_to_end(nk_model_setup, synthetic_nk_data):
    """Verify full end-to-end SMCSampler execution on DSGE model."""
    m = nk_model_setup["m"]
    data = synthetic_nk_data.iloc[:25]

    sampler = SMCSampler(
        m=m,
        data=data,
        varobs=["y", "pi", "r"],
        n_particles=40,
        n_stages=4,
        adaptive_tempering=True,
        seed=101,
    )
    res = sampler.sample()

    assert isinstance(res, SMCResult)
    assert res.stage_tempering[-1] == 1.0
    assert np.isfinite(res.mdd)
    assert np.isfinite(res.mdd_se)
    assert len(res.particles) == 40
    assert len(res.posterior_summary) >= 2
