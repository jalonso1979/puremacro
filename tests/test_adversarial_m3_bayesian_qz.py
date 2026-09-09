"""Adversarial stress test suite for Bayesian IRFs, Prior Predictive Analysis, and Tunable QZ Criterium.

Author: m3_challenger_2 (teamwork_preview_challenger)
Scope:
1. bayesian_irf:
   - Determinacy filtering and determinacy_rate accounting.
   - Non-biasing of posterior medians under mixed determinate, indeterminate, and explosive draws.
   - Error handling for 100% indeterminate draws (RuntimeError) and empty draws (ValueError).
   - Input format flexibility (pd.DataFrame, np.ndarray 2D/3D with burn_in, BayesianEstimationResult, dict, argument order swap).
2. Credible Intervals Monotonicity:
   - Strict nested inclusion property across multi-band envelopes (0.50, 0.68, 0.90, 0.95, 0.99):
     l_99 <= l_95 <= l_90 <= l_68 <= l_50 <= median <= u_50 <= u_68 <= u_90 <= u_95 <= u_99.
   - Percentage vs decimal specification equivalence.
3. prior_predictive:
   - Mixed parameter priors spanning all 5 prior distributions (Beta, Gamma, Normal, InvGamma, Uniform).
   - Non-zero simulated theoretical moments (variances and standard deviations > 0).
   - Custom draw injection (DataFrame and ndarray).
   - Degenerate indeterminate prior handling.
   - Presentation and report generation contract (.summary(), .plot(), .to_frame(), .to_markdown(), .to_latex(), .to_typst()).
4. Tunable qz_criterium:
   - Boundary classification on exact unit roots (lambda = 1.0):
     - qz_criterium = 1.0 + 1e-5: accepts unit roots as non-explosive across klein_solve, gensys, and LinearModel.solve.
     - qz_criterium = 1.0 - 1e-5: treats unit roots as explosive, raising BlanchardKahnError (strict=True) or returning eu=(0, 0).
   - Cointegrated bivariate system test.
   - Dynamic DSGE with permanent technology shock.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    build_dynare,
    bayesian_irf,
    BayesianIRFResult,
    prior_predictive,
    PriorPredictiveResult,
    klein_solve,
    BetaPrior,
    GammaPrior,
    NormalPrior,
    InvGammaPrior,
    UniformPrior,
)
from puremacro.dsge.bayesian import BayesianEstimationResult
from puremacro.dsge.gensys import gensys
from puremacro.dsge.klein import BlanchardKahnError


@pytest.fixture
def nk_model_fixture():
    """Canonical 3-equation New Keynesian model with monetary and technology shocks."""
    params = {
        "beta": 0.99,
        "sigma": 1.0,
        "kappa": 0.15,
        "phi_pi": 1.5,
        "phi_y": 0.25,
        "rho_r": 0.7,
        "rho_a": 0.6,
    }
    variables = ["y", "pi", "r", "a"]
    shocks = ["e_d", "e_m"]
    steady_state = {v: 0.0 for v in variables}

    def nk_equations(lead, curr, lag, shocks_v, p):
        return [
            curr.y - (lead.y - (curr.r - lead.pi) / p.sigma + curr.a),
            curr.pi - (p.beta * lead.pi + p.kappa * curr.y),
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.e_m),
            curr.a - (p.rho_a * lag.a + shocks_v.e_d),
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


# =============================================================================
# 1. BAYESIAN IRF: DETERMINACY FILTERING & MEDIAN INVARIANCE
# =============================================================================

def test_adversarial_bayesian_irf_determinacy_filtering_and_median_invariance(nk_model_fixture):
    """Verify indeterminate/explosive draws are filtered with zero bias on posterior median."""
    m = nk_model_fixture
    rng = np.random.default_rng(2026)

    # 40 Determinate parameter draws (Taylor principle satisfied: phi_pi = 1.5)
    n_det = 40
    det_draws = pd.DataFrame({
        "phi_pi": np.full(n_det, 1.5),
        "kappa": rng.uniform(0.10, 0.25, size=n_det),
        "rho_r": rng.uniform(0.50, 0.80, size=n_det),
        "rho_a": np.full(n_det, 0.60),
    })

    # Baseline: Bayesian IRF on determinate draws only
    res_det = bayesian_irf(m, draws=det_draws, shock="e_m", periods=15)
    assert res_det.n_draws == 40
    assert res_det.n_valid == 40
    assert res_det.determinacy_rate == 1.0

    # 20 Indeterminate parameter draws (Taylor principle violated: phi_pi = 0.5 < 1.0)
    n_indet = 20
    indet_draws = pd.DataFrame({
        "phi_pi": np.full(n_indet, 0.5),
        "kappa": rng.uniform(0.10, 0.25, size=n_indet),
        "rho_r": rng.uniform(0.50, 0.80, size=n_indet),
        "rho_a": np.full(n_indet, 0.60),
    })

    # 15 Explosive parameter draws (Unstable state autoregression: rho_a = 1.5 > 1.0)
    n_expl = 15
    expl_draws = pd.DataFrame({
        "phi_pi": np.full(n_expl, 1.5),
        "kappa": rng.uniform(0.10, 0.25, size=n_expl),
        "rho_r": rng.uniform(0.50, 0.80, size=n_expl),
        "rho_a": np.full(n_expl, 1.50),
    })

    # Mixed draws: 40 det + 20 indet + 15 expl = 75 total
    mixed_draws = pd.concat([det_draws, indet_draws, expl_draws], ignore_index=True)
    res_mixed = bayesian_irf(m, draws=mixed_draws, shock="e_m", periods=15)

    assert res_mixed.n_draws == 75
    assert res_mixed.n_valid == 40
    np.testing.assert_allclose(res_mixed.determinacy_rate, 40 / 75)

    # CRITICAL: Posterior median must be mathematically identical to determinate benchmark
    max_median_diff = np.max(np.abs(res_mixed.median.to_numpy() - res_det.median.to_numpy()))
    assert max_median_diff < 1e-13, f"Indeterminate draws biased median by {max_median_diff}"


def test_adversarial_bayesian_irf_all_draws_indeterminate_raises(nk_model_fixture):
    """Verify RuntimeError is raised when 100% of parameter draws fail Blanchard-Kahn."""
    m = nk_model_fixture
    all_indet = pd.DataFrame({"phi_pi": np.full(30, 0.3)})
    with pytest.raises(RuntimeError, match="All 30 parameter draws were indeterminate or failed to solve"):
        bayesian_irf(m, draws=all_indet, shock="e_m")


def test_adversarial_bayesian_irf_empty_draws_raises(nk_model_fixture):
    """Verify ValueError is raised when draws container is empty."""
    m = nk_model_fixture
    empty_df = pd.DataFrame({"phi_pi": []})
    with pytest.raises(ValueError, match="draws is empty"):
        bayesian_irf(m, draws=empty_df)


def test_adversarial_bayesian_irf_mcmc_chains_burn_in(nk_model_fixture):
    """Verify 3D MCMC chains array correctly applies burn_in discard across chains."""
    m = nk_model_fixture
    # 3 chains of length 50 with 2 parameters (phi_pi, kappa)
    n_chains, n_draws, n_params = 3, 50, 2
    chains = np.zeros((n_chains, n_draws, n_params))
    # Burn-in draws (first 20) violate Taylor principle (phi_pi = 0.5)
    chains[:, :20, 0] = 0.5
    chains[:, :20, 1] = 0.15
    # Retained draws (remaining 30) satisfy Taylor principle (phi_pi = 1.5)
    chains[:, 20:, 0] = 1.5
    chains[:, 20:, 1] = 0.15

    # Run with burn_in = 20: all retained draws should be determinate!
    res = bayesian_irf(m, draws=chains, param_names=["phi_pi", "kappa"], burn_in=20, shock="e_m")
    assert res.n_draws == 3 * 30  # 90 post-burn draws
    assert res.n_valid == 90
    assert res.determinacy_rate == 1.0


def test_adversarial_bayesian_irf_estimation_result_input(nk_model_fixture):
    """Verify bayesian_irf accepts BayesianEstimationResult object seamlessly."""
    m = nk_model_fixture
    chains = np.zeros((2, 25, 2))
    chains[:, :, 0] = 1.5
    chains[:, :, 1] = 0.15
    summary = pd.DataFrame({"mean": [1.5, 0.15]}, index=["phi_pi", "kappa"])

    est_res = BayesianEstimationResult(
        mode=np.array([1.5, 0.15]),
        mode_se=np.array([0.05, 0.01]),
        param_names=["phi_pi", "kappa"],
        log_posterior_mode=-12.5,
        chains=chains,
        acceptance_rate=0.28,
        posterior_summary=summary,
        diagnostics={},
    )
    res = bayesian_irf(m, draws=est_res, burn_in=5, shock="e_m", periods=10)
    assert res.n_draws == 2 * 20  # 40 post-burn draws
    assert res.n_valid == 40
    assert res.determinacy_rate == 1.0


def test_adversarial_bayesian_irf_argument_order_swap(nk_model_fixture):
    """Verify bayesian_irf supports inverted argument order bayesian_irf(draws, model)."""
    m = nk_model_fixture
    draws = pd.DataFrame({"phi_pi": [1.4, 1.5, 1.6]})
    res = bayesian_irf(draws, m, shock="e_m", periods=5)
    assert isinstance(res, BayesianIRFResult)
    assert res.n_draws == 3
    assert res.n_valid == 3


# =============================================================================
# 2. CREDIBLE INTERVALS: MONOTONIC INCLUSION PROPERTY
# =============================================================================

def test_adversarial_credible_intervals_monotonic_inclusion(nk_model_fixture):
    """Verify strict nested inclusion property across equal-tailed credible bands:
    band_99_lower <= band_95_lower <= band_90_lower <= band_68_lower <= band_50_lower
    <= median <=
    band_50_upper <= band_68_upper <= band_90_upper <= band_95_upper <= band_99_upper.
    """
    m = nk_model_fixture
    priors = {
        "kappa": GammaPrior(mean=0.15, std=0.05),
        "rho_r": BetaPrior(mean=0.70, std=0.10),
        "rho_a": BetaPrior(mean=0.60, std=0.10),
        "phi_pi": NormalPrior(mean=1.50, std=0.20),
        "phi_y": GammaPrior(mean=0.25, std=0.10),
    }

    test_bands = (0.50, 0.68, 0.90, 0.95, 0.99)
    res = bayesian_irf(
        m,
        priors=priors,
        shock="e_m",
        periods=35,
        n_draws=80,
        bands=test_bands,
        seed=31415,
    )

    med = res.median
    low_50, up_50 = res.bands[0.50]
    low_68, up_68 = res.bands[0.68]
    low_90, up_90 = res.bands[0.90]
    low_95, up_95 = res.bands[0.95]
    low_99, up_99 = res.bands[0.99]

    tol = 1e-12
    for v in m.variables:
        # Lower bands monotonic ordering: l99 <= l95 <= l90 <= l68 <= l50
        assert np.all(low_99[v] <= low_95[v] + tol), f"Violation low_99 <= low_95 for {v}"
        assert np.all(low_95[v] <= low_90[v] + tol), f"Violation low_95 <= low_90 for {v}"
        assert np.all(low_90[v] <= low_68[v] + tol), f"Violation low_90 <= low_68 for {v}"
        assert np.all(low_68[v] <= low_50[v] + tol), f"Violation low_68 <= low_50 for {v}"

        # Median sits inside central band
        assert np.all(low_50[v] <= med[v] + tol), f"Violation low_50 <= median for {v}"
        assert np.all(med[v] <= up_50[v] + tol), f"Violation median <= up_50 for {v}"

        # Upper bands monotonic ordering: u50 <= u68 <= u90 <= u95 <= u99
        assert np.all(up_50[v] <= up_68[v] + tol), f"Violation up_50 <= up_68 for {v}"
        assert np.all(up_68[v] <= up_90[v] + tol), f"Violation up_68 <= up_90 for {v}"
        assert np.all(up_90[v] <= up_95[v] + tol), f"Violation up_90 <= up_95 for {v}"
        assert np.all(up_95[v] <= up_99[v] + tol), f"Violation up_95 <= up_99 for {v}"


def test_adversarial_credible_intervals_percentage_vs_decimal(nk_model_fixture):
    """Verify bands=(68, 90, 95) and bands=(0.68, 0.90, 0.95) evaluate identically."""
    m = nk_model_fixture
    draws = pd.DataFrame({"phi_pi": np.linspace(1.2, 1.8, 30)})

    res_dec = bayesian_irf(m, draws=draws, bands=(0.68, 0.90, 0.95), shock="e_m", periods=8)
    res_pct = bayesian_irf(m, draws=draws, bands=(68, 90, 95), shock="e_m", periods=8)

    for b in [0.68, 0.90, 0.95]:
        low_d, up_d = res_dec.bands[b]
        low_p, up_p = res_pct.bands[b]
        np.testing.assert_allclose(low_d.to_numpy(), low_p.to_numpy(), atol=1e-14)
        np.testing.assert_allclose(up_d.to_numpy(), up_p.to_numpy(), atol=1e-14)


# =============================================================================
# 3. PRIOR PREDICTIVE: MIXED PRIORS & MOMENTS EVALUATION
# =============================================================================

def test_adversarial_prior_predictive_mixed_priors_non_zero_moments(nk_model_fixture):
    """Verify prior_predictive with mixed prior distributions generates non-zero moments."""
    m = nk_model_fixture
    # Mixed prior distributions covering all 5 supported types
    mixed_priors = {
        "rho_r": BetaPrior(mean=0.70, std=0.08),       # Beta
        "kappa": GammaPrior(mean=0.15, std=0.04),     # Gamma
        "phi_pi": NormalPrior(mean=1.50, std=0.15),   # Normal
        "sigma": InvGammaPrior(mean=1.00, std=0.20),  # InvGamma
        "phi_y": UniformPrior(lb=0.10, ub=0.40),      # Uniform
    }

    res_pp = prior_predictive(
        m,
        priors=mixed_priors,
        n_draws=40,
        moments=True,
        seed=999,
    )

    assert isinstance(res_pp, PriorPredictiveResult)
    assert res_pp.n_draws == 40
    assert res_pp.n_valid == 40
    assert res_pp.determinacy_rate == 1.0
    assert len(res_pp.prior_draws) == 40
    assert len(res_pp.valid_draws) == 40

    moments = res_pp.prior_moments
    assert isinstance(moments, pd.DataFrame)
    assert list(moments.columns) == ["mean", "std", "p5", "median", "p95"]

    # Verify that variance and standard deviation rows are strictly positive
    var_rows = [idx for idx in moments.index if "Variance" in idx or "Std" in idx]
    assert len(var_rows) >= len(m.variables)
    for r in var_rows:
        mean_val = moments.loc[r, "mean"]
        med_val = moments.loc[r, "median"]
        p95_val = moments.loc[r, "p95"]
        assert mean_val > 0.0, f"Expected positive mean for {r}, got {mean_val}"
        assert med_val > 0.0, f"Expected positive median for {r}, got {med_val}"
        assert p95_val >= med_val, f"Quantile ordering failed for {r}"


def test_adversarial_prior_predictive_custom_draws_injection(nk_model_fixture):
    """Verify prior_predictive functions when custom parameter draws are supplied directly."""
    m = nk_model_fixture
    custom_draws = pd.DataFrame({
        "phi_pi": [1.4, 1.5, 1.6, 1.7],
        "kappa": [0.12, 0.14, 0.16, 0.18],
    })

    res = prior_predictive(m, draws=custom_draws, moments=True)
    assert res.n_draws == 4
    assert res.n_valid == 4
    assert res.determinacy_rate == 1.0
    assert len(res.prior_moments) > 0


def test_adversarial_prior_predictive_all_indeterminate_priors(nk_model_fixture):
    """Verify prior_predictive handles completely indeterminate priors without crashing."""
    m = nk_model_fixture
    indet_priors = {
        "phi_pi": UniformPrior(lb=0.2, ub=0.6),  # Violates Taylor principle everywhere
    }
    res = prior_predictive(m, priors=indet_priors, n_draws=20, moments=True)
    assert res.n_draws == 20
    assert res.n_valid == 0
    assert res.determinacy_rate == 0.0
    assert res.prior_moments.empty


def test_adversarial_prior_predictive_presentation_methods(nk_model_fixture):
    """Verify all presentation methods on PriorPredictiveResult."""
    m = nk_model_fixture
    priors = {"kappa": GammaPrior(mean=0.15, std=0.03)}
    res = prior_predictive(m, priors=priors, n_draws=10, moments=True, seed=1)

    # Text summary
    summary = res.summary()
    assert "Prior Predictive Analysis" in summary
    assert "Determinacy Rate" in summary

    # Table exports
    assert isinstance(res.to_frame(), pd.DataFrame)
    assert "|" in res.to_markdown()
    assert "\\begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()

    # Fan chart plot
    fig = res.plot()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# =============================================================================
# 4. TUNABLE QZ CRITERIUM: UNIT ROOT (LAMBDA = 1.0) BEHAVIOR
# =============================================================================

def test_adversarial_qz_criterium_scalar_unit_root():
    """Verify unit root lambda = 1.0 accepted under 1.0 + 1e-5 and treated as explosive under 1.0 - 1e-5."""
    # Scalar random walk: x_t = x_{t-1} + u_t
    # In Klein form: A E_t[x_{t+1}] = B x_t + C u_t  with A = [1], B = [1], C = [1]
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])

    # 1. qz_criterium = 1.0 + 1e-5: accepts unit root
    sol_acc = klein_solve(A, B, n_pre=1, C=C, strict=False, qz_criterium=1.0 + 1e-5)
    assert sol_acc.eu == (1, 1)
    np.testing.assert_allclose(sol_acc.G[0, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(sol_acc.N[0, 0], 1.0, atol=1e-12)

    # 2. qz_criterium = 1.0 - 1e-5: treats unit root as explosive
    # strict=False returns eu=(0, 0) and zero matrices
    sol_rej = klein_solve(A, B, n_pre=1, C=C, strict=False, qz_criterium=1.0 - 1e-5)
    assert sol_rej.eu == (0, 0)
    assert sol_rej.G[0, 0] == 0.0

    # strict=True raises BlanchardKahnError
    with pytest.raises(BlanchardKahnError) as exc_info:
        klein_solve(A, B, n_pre=1, C=C, strict=True, qz_criterium=1.0 - 1e-5)
    assert "no stable solution" in str(exc_info.value)
    assert exc_info.value.n_unstable == 1
    assert exc_info.value.n_fwd == 0

    # 3. Sims gensys counterpart
    g0 = np.array([[1.0]])
    g1 = np.array([[1.0]])
    psi = np.array([[1.0]])
    pi = np.zeros((1, 0))

    gen_acc = gensys(g0, g1, psi, pi, qz_criterium=1.0 + 1e-5)
    assert gen_acc.eu == (1, 1)
    np.testing.assert_allclose(gen_acc.G[0, 0], 1.0, atol=1e-12)

    gen_rej = gensys(g0, g1, psi, pi, qz_criterium=1.0 - 1e-5)
    assert gen_rej.eu == (0, 0)
    assert gen_rej.G[0, 0] == 0.0


def test_adversarial_qz_criterium_cointegrated_system():
    """Verify tunable qz_criterium on a cointegrated bivariate system (lambda_1 = 1.0, lambda_2 = 0.5)."""
    # System:
    # x1_t = x1_{t-1} + u1_t
    # x2_t = 0.5 * x1_t + 0.5 * x2_{t-1} + u2_t
    A = np.eye(2)
    B = np.array([[1.0, 0.0],
                  [0.5, 0.5]])
    C = np.eye(2)

    # Accepts unit root
    sol_acc = klein_solve(A, B, n_pre=2, C=C, strict=False, qz_criterium=1.0 + 1e-5)
    assert sol_acc.eu == (1, 1)
    np.testing.assert_allclose(sol_acc.G, B, atol=1e-12)

    # Rejects unit root
    sol_rej = klein_solve(A, B, n_pre=2, C=C, strict=False, qz_criterium=1.0 - 1e-5)
    assert sol_rej.eu == (0, 0)

    # Gensys counterpart
    gen_acc = gensys(A, B, C, np.zeros((2, 0)), qz_criterium=1.0 + 1e-5)
    assert gen_acc.eu == (1, 1)
    np.testing.assert_allclose(gen_acc.G, B, atol=1e-12)

    gen_rej = gensys(A, B, C, np.zeros((2, 0)), qz_criterium=1.0 - 1e-5)
    assert gen_rej.eu == (0, 0)


def test_adversarial_qz_criterium_linear_model_solve():
    """Verify LinearModel.solve propagates qz_criterium and alters unit-root solvability."""
    def rw_model(lead, curr, lag, shocks, p):
        return [curr.x - (p.rho * lag.x + shocks.e)]

    # Build model with unit root rho = 1.0 under qz_criterium = 1.0 + 1e-5
    m_acc = build_dynare(
        rw_model,
        variables=["x"],
        shocks=["e"],
        params={"rho": 1.0},
        steady_state={"x": 0.0},
        check_steady_state=False,
        strict=False,
        qz_criterium=1.0 + 1e-5,
    )
    assert m_acc.is_determinate is True
    assert m_acc._qz_criterium == 1.0 + 1e-5

    # Re-solve with qz_criterium = 1.0 + 1e-5: succeeds
    m_re_acc = m_acc.solve(order=1, qz_criterium=1.0 + 1e-5)
    assert m_re_acc.is_determinate is True

    # Re-solve with qz_criterium = 1.0 - 1e-5: raises BlanchardKahnError
    with pytest.raises(BlanchardKahnError):
        m_acc.solve(order=1, qz_criterium=1.0 - 1e-5)


def test_adversarial_qz_criterium_in_bayesian_irf(nk_model_fixture):
    """Verify qz_criterium parameter can be tuned inside bayesian_irf."""
    m = nk_model_fixture
    draws = pd.DataFrame({"phi_pi": [1.5]})
    # Normal execution with default criterium
    res = bayesian_irf(m, draws=draws, qz_criterium=1.0 + 1e-5, shock="e_m", periods=5)
    assert res.determinacy_rate == 1.0
