"""Adversarial challenge and empirical stress-test suite for DSGE News & Anticipated Shocks Engine.

Written by orch8_challenger_m3_1 to independently verify mathematical properties,
edge conditions, and cross-model stability of Milestone 3.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    LinearModel,
    ModelError,
    build_dynare,
    build,
    decompose_news,
    news_irf,
    parse_mod,
    load_mod,
    plot_news_vs_surprise,
    NewsIRFResult,
    NewsDecompositionResult,
)
from puremacro.dsge.news import augment_news_state_space


@pytest.fixture(scope="module")
def nk_3eq_model() -> tuple[LinearModel, dict[str, float]]:
    """Standard 3-equation New Keynesian model with persistent AR(1) shock processes."""
    params = dict(
        beta=0.99,
        sigma=1.0,
        kappa=0.15,
        phi_pi=1.5,
        phi_y=0.125,
        rho_g=0.8,
        rho_u=0.7,
        rho_v=0.5,
    )

    def nk_eqs(lead, curr, lag, e, p):
        return [
            curr.y - (lead.y - (1.0 / p.sigma) * (curr.i - lead.pi) + curr.g),
            curr.pi - (p.beta * lead.pi + p.kappa * curr.y + curr.u),
            curr.i - (p.phi_pi * curr.pi + p.phi_y * curr.y + curr.v),
            curr.g - p.rho_g * lag.g - e.eps_g,
            curr.u - p.rho_u * lag.u - e.eps_u,
            curr.v - p.rho_v * lag.v - e.eps_v,
        ]

    m = build_dynare(
        nk_eqs,
        variables=["y", "pi", "i", "g", "u", "v"],
        shocks=["eps_g", "eps_u", "eps_v"],
        params=params,
        guess=dict(y=0.0, pi=0.0, i=0.0, g=0.0, u=0.0, v=0.0),
    )
    return m, params


@pytest.fixture(scope="module")
def rbc_dynare_model() -> tuple[LinearModel, dict[str, float]]:
    """Standard RBC model with capital accumulation and TFP shock."""
    params = dict(alpha=0.33, beta=0.99, delta=0.025, sigma=1.0, rho=0.95)

    def rbc_eqs(lead, curr, lag, e, p):
        return [
            curr.c**-p.sigma - p.beta * lead.c**-p.sigma * (p.alpha * lead.z * curr.k**(p.alpha - 1) + 1 - p.delta),
            curr.c + curr.k - (1 - p.delta) * lag.k - curr.z * lag.k**p.alpha,
            curr.z - (1 - p.rho) - p.rho * lag.z - e.eps,
        ]

    m = build_dynare(
        rbc_eqs,
        variables=["c", "k", "z"],
        shocks=["eps"],
        params=params,
        guess=dict(c=2.0, k=25.0, z=1.0),
    )
    return m, params


# =========================================================================
# 1. COMPANION STATE-SPACE EIGENVALUES & BLANCHARD-KAHN INVARIANCE
# =========================================================================

@pytest.mark.parametrize("H", [1, 2, 4, 8, 12, 16, 24])
def test_companion_eigenvalue_properties_nk_model(nk_3eq_model, H: int):
    """Verify companion state-space introduces exactly H zero eigenvalues and preserves economic roots."""
    m, _ = nk_3eq_model
    assert m.is_determinate

    orig_evals = np.sort(np.abs(m.eigenvalues))
    orig_n_states = len(m.states)

    for shock in ["eps_g", "eps_u", "eps_v"]:
        m_aug = augment_news_state_space(m, shock=shock, max_lead=H)
        assert m_aug.is_determinate
        assert tuple(m_aug.solution.eu) == (1, 1)

        # State and eigenvalue counts increase by exactly H
        assert len(m_aug.states) == orig_n_states + H
        assert len(m_aug.eigenvalues) == len(m.eigenvalues) + H

        # Exactly H eigenvalues are identically zero (< 1e-12)
        aug_evals = np.sort(np.abs(m_aug.eigenvalues))
        zero_evals = aug_evals[:H]
        np.testing.assert_allclose(zero_evals, 0.0, atol=1e-12)

        # Non-zero economic eigenvalues preserved within 1e-10
        econ_evals = aug_evals[H:]
        finite_mask = (orig_evals < 1e10) & np.isfinite(orig_evals)
        np.testing.assert_allclose(econ_evals[finite_mask], orig_evals[finite_mask], atol=1e-10)


@pytest.mark.parametrize("H", [1, 2, 4, 8, 16, 24])
def test_companion_eigenvalue_properties_rbc_model(rbc_dynare_model, H: int):
    """Verify companion state-space augmentation for RBC with capital up to H=24."""
    m, _ = rbc_dynare_model
    assert m.is_determinate

    orig_evals = np.sort(np.abs(m.eigenvalues))
    orig_n_states = len(m.states)

    m_aug = augment_news_state_space(m, shock="eps", max_lead=H)
    assert m_aug.is_determinate
    assert tuple(m_aug.solution.eu) == (1, 1)

    assert len(m_aug.states) == orig_n_states + H
    assert len(m_aug.eigenvalues) == len(m.eigenvalues) + H

    aug_evals = np.sort(np.abs(m_aug.eigenvalues))
    zero_evals = aug_evals[:H]
    np.testing.assert_allclose(zero_evals, 0.0, atol=1e-12)

    econ_evals = aug_evals[H:]
    finite_mask = (orig_evals < 1e10) & np.isfinite(orig_evals)
    np.testing.assert_allclose(econ_evals[finite_mask], orig_evals[finite_mask], atol=1e-10)


# =========================================================================
# 2. PREDETERMINED STATES ZERO REVISION BEFORE REALIZATION (t < k)
# =========================================================================

@pytest.mark.parametrize("k", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("shock", ["eps_g", "eps_u", "eps_v"])
def test_predetermined_states_strictly_zero_nk(nk_3eq_model, k: int, shock: str):
    """In 3-equation NK model, all predetermined states (g, u, v) exhibit zero revision for t < k."""
    m, _ = nk_3eq_model
    res = news_irf(m, shock=shock, lead=k, horizon=25, size=1.0)
    df = res.irf

    for t in range(k):
        norm_s = np.linalg.norm([df.loc[t, "g"], df.loc[t, "u"], df.loc[t, "v"]], ord=np.inf)
        assert norm_s < 1e-12, f"Predetermined state revision at t={t} < k={k}: norm={norm_s}"


@pytest.mark.parametrize("k", [1, 2, 4, 8, 16])
def test_predetermined_exogenous_state_strictly_zero_rbc(rbc_dynare_model, k: int):
    """In RBC model, exogenous state z exhibits zero revision for t < k."""
    m, _ = rbc_dynare_model
    res = news_irf(m, shock="eps", lead=k, horizon=25, size=1.0)
    df = res.irf

    for t in range(k):
        assert abs(df.loc[t, "z"]) < 1e-12, f"RBC z revision at t={t} < k={k}: z={df.loc[t, 'z']}"


# =========================================================================
# 3. FORWARD-LOOKING JUMP DYNAMICS AT ANNOUNCEMENT (t = 0)
# =========================================================================

@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_forward_looking_immediate_jump_nk(nk_3eq_model, k: int):
    """Forward-looking jump controls (y, pi, i) react strictly on impact at t=0 upon announcement."""
    m, _ = nk_3eq_model
    for shock in ["eps_g", "eps_u", "eps_v"]:
        res = news_irf(m, shock=shock, lead=k, horizon=20, size=1.0)
        df = res.irf
        jump_norm = np.linalg.norm([df.loc[0, "y"], df.loc[0, "pi"], df.loc[0, "i"]], ord=np.inf)
        assert jump_norm > 1e-3, f"Forward variables failed to jump at t=0: {jump_norm}"


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_forward_looking_immediate_jump_rbc(rbc_dynare_model, k: int):
    """In RBC model, forward-looking consumption jumps immediately at t=0."""
    m, _ = rbc_dynare_model
    res_pos = news_irf(m, shock="eps", lead=k, horizon=20, size=1.0)
    res_neg = news_irf(m, shock="eps", lead=k, horizon=20, size=-1.0)

    # Positive news induces positive consumption jump
    assert res_pos.irf.loc[0, "c"] > 0.01
    # Negative news induces negative consumption jump
    assert res_neg.irf.loc[0, "c"] < -0.01


# =========================================================================
# 4. STRUCTURAL SHOCK EXACT REALIZATION AT t = k
# =========================================================================

@pytest.mark.parametrize("k", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("size", [0.25, 1.0, 2.5, -1.8])
def test_shock_exact_realization_and_decay_nk(nk_3eq_model, k: int, size: float):
    """Structural shock realizes exactly at t=k (shock_var == size) and decays geometrically."""
    m, params = nk_3eq_model
    for shock in ["eps_g", "eps_u", "eps_v"]:
        shock_var = shock.replace("eps_", "")
        rho = params[f"rho_{shock_var}"]
        res = news_irf(m, shock=shock, lead=k, horizon=k + 15, size=size)
        df = res.irf

        # Exact realization at t=k
        np.testing.assert_allclose(df.loc[k, shock_var], size, atol=1e-10, rtol=1e-8)

        # Decay for t > k
        for h in range(k + 1, k + 6):
            expected = size * (rho ** (h - k))
            np.testing.assert_allclose(df.loc[h, shock_var], expected, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("k", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("size", [0.5, 1.0, -2.0])
def test_shock_exact_realization_and_decay_rbc(rbc_dynare_model, k: int, size: float):
    """RBC TFP shock realizes exactly at t=k and decays according to AR(1) parameter rho."""
    m, params = rbc_dynare_model
    rho = params["rho"]
    res = news_irf(m, shock="eps", lead=k, horizon=k + 20, size=size)
    df = res.irf

    np.testing.assert_allclose(df.loc[k, "z"], size, atol=1e-10, rtol=1e-8)
    for h in range(k + 1, k + 8):
        expected = size * (rho ** (h - k))
        np.testing.assert_allclose(df.loc[h, "z"], expected, atol=1e-8, rtol=1e-6)


# =========================================================================
# 5. LEAD = 0 SURPRISE IRF REPRODUCTION (< 10^-12 PRECISION)
# =========================================================================

@pytest.mark.parametrize("size", [0.01, 0.5, 1.0, 5.0, -3.2])
def test_lead_zero_reproduces_surprise_irf_nk(nk_3eq_model, size: float):
    """When lead=0, news_irf matches model.irf to < 10^-12 precision across all variables."""
    m, _ = nk_3eq_model
    for shock in ["eps_g", "eps_u", "eps_v"]:
        res_news = news_irf(m, shock=shock, lead=0, horizon=40, size=size)
        df_surp = m.irf(shock, horizon=40, size=size)

        max_diff = np.max(np.abs(res_news.irf.to_numpy() - df_surp.to_numpy()))
        assert max_diff < 1e-12, f"Shock {shock}, size {size}: max diff = {max_diff}"


@pytest.mark.parametrize("size", [0.005, 1.0, 10.0, -2.5])
def test_lead_zero_reproduces_surprise_irf_rbc(rbc_dynare_model, size: float):
    """RBC model lead=0 matches surprise IRF to < 10^-12 precision."""
    m, _ = rbc_dynare_model
    res_news = news_irf(m, shock="eps", lead=0, horizon=40, size=size)
    df_surp = m.irf("eps", horizon=40, size=size)

    max_diff = np.max(np.abs(res_news.irf.to_numpy() - df_surp.to_numpy()))
    assert max_diff < 1e-12, f"RBC size {size}: max diff = {max_diff}"


# =========================================================================
# 6. FEVD VARIANCE DECOMPOSITION PROPERTIES
# =========================================================================

@pytest.mark.parametrize("max_lead", [1, 2, 4, 8])
def test_news_decomposition_variance_shares_nk(nk_3eq_model, max_lead: int):
    """Forecast error variance shares sum to 1.0 and remain within [0, 1]. Inactive variables handled."""
    m, _ = nk_3eq_model
    for shock in ["eps_g", "eps_u", "eps_v"]:
        decomp = decompose_news(m, shock=shock, horizon=30, max_lead=max_lead)
        df = decomp.variance_shares

        # Every row sums to 1.0
        row_sums = df.sum(axis=1).to_numpy()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

        # All shares in [0, 1]
        vals = df.to_numpy()
        assert (vals >= -1e-12).all() and (vals <= 1.0 + 1e-12).all()

        # Dynamic shares sum to 1.0 at each horizon step
        assert decomp.dynamic_shares is not None
        for var in m.variables:
            dyn_sums = decomp.dynamic_shares[var].sum(axis=1).to_numpy()
            np.testing.assert_allclose(dyn_sums, 1.0, atol=1e-12)


# =========================================================================
# 7. DYNARE .MOD PARSER ANTICIPATED SHOCKS INTEGRATION
# =========================================================================

def test_dynare_parser_multi_shock_anticipated_syntax():
    """Test full Dynare parser syntax for anticipated shocks with ranges and comma lists."""
    mod_text = """
    var y, pi, i, g, u;
    varexo eps_g, eps_u;
    parameters beta, sigma, kappa, phi_pi, phi_y, rho_g, rho_u;
    beta = 0.99; sigma = 1.0; kappa = 0.15; phi_pi = 1.5; phi_y = 0.125; rho_g = 0.8; rho_u = 0.7;

    initval;
    y = 0.0; pi = 0.0; i = 0.0; g = 0.0; u = 0.0;
    end;

    model;
    y = y(+1) - (1/sigma)*(i - pi(+1)) + g;
    pi = beta * pi(+1) + kappa * y + u;
    i = phi_pi * pi + phi_y * y;
    g = rho_g * g(-1) + eps_g;
    u = rho_u * u(-1) + eps_u;
    end;

    shocks;
    var eps_g; stderr 0.01;
    var eps_u; stderr 0.01;
    var eps_g; periods 1:4; values 0.02;
    var eps_u; periods 2, 4, 6; values 0.01, 0.03, 0.05;
    end;
    """
    m = load_mod(mod_text)
    assert m.anticipated_shocks is not None
    assert "eps_g" in m.anticipated_shocks
    assert "eps_u" in m.anticipated_shocks
    assert m.anticipated_shocks["eps_g"]["periods"] == [1, 2, 3, 4]
    assert m.anticipated_shocks["eps_g"]["values"] == [0.02, 0.02, 0.02, 0.02]
    assert m.anticipated_shocks["eps_u"]["periods"] == [2, 4, 6]
    assert m.anticipated_shocks["eps_u"]["values"] == [0.01, 0.03, 0.05]


# =========================================================================
# 8. PRESENTATION CONTRACT AND HEADLESS VISUALIZATION
# =========================================================================

def test_presentation_contract_and_headless_plots(nk_3eq_model):
    """Verify 6 presentation methods on NewsIRFResult and NewsDecompositionResult with headless plots."""
    m, _ = nk_3eq_model
    res = news_irf(m, shock="eps_v", lead=3, horizon=15)

    assert isinstance(res.to_frame(), pd.DataFrame)
    assert len(res.summary()) > 0
    assert len(res.to_markdown()) > 0
    assert len(res.to_latex()) > 0
    assert len(res.to_typst()) > 0

    fig1, ax1 = res.plot()
    assert isinstance(fig1, plt.Figure)
    fig1.canvas.draw()
    plt.close(fig1)

    decomp = decompose_news(m, shock="eps_v", horizon=15, max_lead=4)
    assert isinstance(decomp.to_frame(), pd.DataFrame)
    assert len(decomp.summary()) > 0
    assert len(decomp.to_markdown()) > 0
    assert len(decomp.to_latex()) > 0
    assert len(decomp.to_typst()) > 0

    fig2, ax2 = decomp.plot()
    assert isinstance(fig2, plt.Figure)
    fig2.canvas.draw()
    plt.close(fig2)

    fig3, ax3 = plot_news_vs_surprise(m, shock="eps_v", leads=(0, 2, 4, 8))
    assert isinstance(fig3, plt.Figure)
    fig3.canvas.draw()
    plt.close(fig3)
