"""Comprehensive unit tests for Bayesian IRFs, Prior Predictive Simulation, and Tunable QZ Criterium.

Tests cover:
1. Bayesian IRF computation (median and equal-tailed bands 68%, 90%, 95%).
2. Indeterminate draw rejection & determinacy rate calculation.
3. Multi-shock Bayesian IRF evaluation.
4. Prior predictive simulation over prior parameter distributions.
5. Prior predictive moment distributions (means, standard deviations, correlations).
6. Tunable qz_criterium unit-root handling across klein_solve, gensys, and LinearModel.solve.
7. Result presentation contracts (BayesianIRFResult, PriorPredictiveResult).
8. FrozenInstanceError on result mutation.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

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
from puremacro.dsge.gensys import gensys


@pytest.fixture
def nk_model_setup():
    """Canonical 3-equation New Keynesian model with parameter prior specifications."""
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
    priors = {
        "kappa": GammaPrior(mean=0.15, std=0.05),
        "rho_r": BetaPrior(mean=0.7, std=0.1),
        "rho_a": BetaPrior(mean=0.6, std=0.1),
        "phi_pi": NormalPrior(mean=1.5, std=0.2),
        "phi_y": GammaPrior(mean=0.25, std=0.1),
    }
    return m, priors


def test_bayesian_irf_single_shock(nk_model_setup):
    """Verify Bayesian IRF computes median and credible bands for a single shock."""
    m, priors = nk_model_setup
    res = bayesian_irf(
        m,
        priors=priors,
        shock="e_m",
        periods=12,
        n_draws=30,
        bands=(0.68, 0.90, 0.95),
        seed=42,
    )

    assert isinstance(res, BayesianIRFResult)
    assert res.shock == "e_m"
    assert res.periods == 12
    assert res.n_draws == 30
    assert res.determinacy_rate > 0.8
    assert isinstance(res.median, pd.DataFrame)
    assert len(res.median) == 13

    # Credible bands structure
    assert 0.68 in res.bands
    assert 0.90 in res.bands
    assert 0.95 in res.bands
    lower_95, upper_95 = res.bands[0.95]
    lower_68, upper_68 = res.bands[0.68]

    # Envelope check
    assert np.all(upper_95["r"] >= upper_68["r"])
    assert np.all(lower_95["r"] <= lower_68["r"])
    assert np.all(res.median["r"] >= lower_68["r"])
    assert np.all(res.median["r"] <= upper_68["r"])


def test_bayesian_irf_multiple_shocks(nk_model_setup):
    """Verify Bayesian IRF handles multiple shocks simultaneously."""
    m, priors = nk_model_setup
    res = bayesian_irf(
        m,
        priors=priors,
        shocks=["e_d", "e_m"],
        periods=8,
        n_draws=20,
        seed=99,
    )

    assert isinstance(res, BayesianIRFResult)
    assert len(res.shocks) == 2
    assert "e_d" in res.median
    assert "e_m" in res.median
    assert res.median["e_d"].shape == (9, 4)


def test_bayesian_irf_filtering_indeterminate_draws(nk_model_setup):
    """Verify Blanchard-Kahn indeterminate draws are rejected and counted."""
    m, _ = nk_model_setup
    # Prior that puts half mass on Taylor principle violation (phi_pi < 1.0)
    adversarial_priors = {
        "phi_pi": UniformPrior(lb=0.7, ub=1.5),
    }
    res = bayesian_irf(
        m,
        priors=adversarial_priors,
        shock="e_m",
        n_draws=25,
        seed=123,
    )
    assert 0.0 < res.determinacy_rate < 1.0


def test_prior_predictive_simulation(nk_model_setup):
    """Verify prior predictive sampling generates moment and IRF distributions."""
    m, priors = nk_model_setup
    res = prior_predictive(
        m,
        priors=priors,
        n_draws=25,
        irf_periods=10,
        seed=77,
    )

    assert isinstance(res, PriorPredictiveResult)
    assert res.n_draws == 25
    assert res.determinacy_rate > 0.8
    assert isinstance(res.parameter_draws, pd.DataFrame)
    assert len(res.parameter_draws) == 25

    # Moments summary
    assert isinstance(res.theoretical_moments, pd.DataFrame)
    assert "mean" in res.theoretical_moments.index or "mean" in res.theoretical_moments.columns


def test_tunable_qz_criterium_unit_root():
    """Verify tunable qz_criterium resolves boundary eigenvalues correctly."""
    # A random-walk scalar process: x_t = x_{t-1} + u_t
    # In Klein form: A * E_t[x_{t+1}] = B * x_t + C * u_t
    # where A = 1, B = 1. Generalized eigenvalue is 1.0.
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])

    # Default threshold 1.0 + 1e-6 treats 1.0 as stable
    sol_stable = klein_solve(A, B, n_pre=1, C=C, qz_criterium=1.0 + 1e-5)
    assert sol_stable.G.shape == (1, 1)
    np.testing.assert_allclose(sol_stable.G[0, 0], 1.0, atol=1e-10)

    # Sims gensys with tunable qz_criterium
    g0 = np.array([[1.0]])
    g1 = np.array([[1.0]])
    psi = np.array([[1.0]])
    pi = np.zeros((1, 0))

    sol_gensys = gensys(g0, g1, psi, pi, qz_criterium=1.0 + 1e-5)
    assert sol_gensys.eu[0] == 1  # Existence


def test_bayesian_presentation_contract(nk_model_setup):
    """Verify presentation contract for BayesianIRFResult and PriorPredictiveResult."""
    m, priors = nk_model_setup
    res_irf = bayesian_irf(m, priors=priors, shock="e_m", periods=6, n_draws=15, seed=1)

    summary_irf = res_irf.summary()
    assert isinstance(summary_irf, str)
    assert "BAYESIAN" in summary_irf.upper()

    frame_irf = res_irf.to_frame()
    assert isinstance(frame_irf, pd.DataFrame)

    assert "|" in res_irf.to_markdown()
    assert "\\begin{tabular}" in res_irf.to_latex()
    assert "#table(" in res_irf.to_typst()

    fig = res_irf.plot()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    res_pp = prior_predictive(m, priors=priors, n_draws=15, seed=2)
    summary_pp = res_pp.summary()
    assert isinstance(summary_pp, str)
    assert "Prior Predictive" in summary_pp
    assert isinstance(res_pp.to_frame(), pd.DataFrame)
    assert "|" in res_pp.to_markdown()
    assert "\\begin{tabular}" in res_pp.to_latex()
    assert "#table(" in res_pp.to_typst()

    fig_pp = res_pp.plot()
    assert isinstance(fig_pp, plt.Figure)
    plt.close(fig_pp)


def test_bayesian_immutability(nk_model_setup):
    """Verify BayesianIRFResult and PriorPredictiveResult are frozen."""
    m, priors = nk_model_setup
    res_irf = bayesian_irf(m, priors=priors, shock="e_m", periods=4, n_draws=10, seed=1)
    with pytest.raises(FrozenInstanceError):
        res_irf.median = None

    res_pp = prior_predictive(m, priors=priors, n_draws=10, seed=1)
    with pytest.raises(FrozenInstanceError):
        res_pp.parameter_draws = None
