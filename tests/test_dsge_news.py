"""Comprehensive unit test suite for puremacro DSGE news and anticipated shocks engine."""
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
    build,
    build_dynare,
    decompose_news,
    load_mod,
    news_irf,
    parse_mod,
    plot_news_vs_surprise,
    NewsIRFResult,
    NewsDecompositionResult,
)
from puremacro.dsge.news import augment_news_state_space


@pytest.fixture
def rbc_dynare_model() -> LinearModel:
    """Standard 3-variable RBC model under Dynare lead-lag canonical form."""
    def rbc_equations(lead, curr, lag, e, p):
        return [
            curr.c**-p.sigma - p.beta * lead.c**-p.sigma * (p.alpha * lead.z * curr.k**(p.alpha - 1) + 1 - p.delta),
            curr.c + curr.k - (1 - p.delta) * lag.k - curr.z * lag.k**p.alpha,
            curr.z - (1 - p.rho) - p.rho * lag.z - e.eps,
        ]

    m = build_dynare(
        rbc_equations,
        variables=["c", "k", "z"],
        shocks=["eps"],
        params=dict(alpha=0.33, beta=0.99, delta=0.025, sigma=1.0, rho=0.95),
        guess=dict(c=2.0, k=25.0, z=1.0),
    )
    return m


@pytest.fixture
def rbc_klein_model() -> LinearModel:
    """Standard 3-variable RBC model under Klein timing (build())."""
    def rbc_klein_eqs(xp, x, e, p):
        return [
            1 / x.c - p.beta * (p.alpha * xp.z * xp.k**(p.alpha - 1)) / xp.c,
            x.c + xp.k - x.z * x.k**p.alpha,
            xp.z - x.z**p.rho * np.exp(e.eps),
        ]

    m = build(
        rbc_klein_eqs,
        variables=["c", "k", "z"],
        states=["k", "z"],
        shocks=["eps"],
        params=dict(alpha=0.33, beta=0.99, rho=0.95),
        guess=dict(c=0.5, k=0.1, z=1.0),
    )
    return m


def test_blanchard_kahn_invariance_and_eigenvalues(rbc_dynare_model: LinearModel):
    """Verify companion state-space augmentation adds H zero eigenvalues and preserves BK determinacy."""
    m = rbc_dynare_model
    assert m.is_determinate

    orig_evals = np.sort(np.abs(m.eigenvalues))
    orig_n_states = len(m.states)

    for H in [1, 2, 4, 8]:
        m_aug = augment_news_state_space(m, shock="eps", max_lead=H)
        assert isinstance(m_aug, LinearModel)
        assert m_aug.is_determinate
        assert tuple(m_aug.solution.eu) == (1, 1)

        # State count increases by exactly H
        assert len(m_aug.states) == orig_n_states + H
        assert len(m_aug.eigenvalues) == len(m.eigenvalues) + H

        # Exactly H eigenvalues are identically zero
        aug_evals = np.sort(np.abs(m_aug.eigenvalues))
        zero_evals = aug_evals[:H]
        np.testing.assert_allclose(zero_evals, 0.0, atol=1e-12)

        # Non-zero finite economic eigenvalues match the base model
        economic_evals = aug_evals[H:]
        finite_mask = (orig_evals < 1e10) & np.isfinite(orig_evals)
        np.testing.assert_allclose(economic_evals[finite_mask], orig_evals[finite_mask], atol=1e-10)


def test_zero_predetermined_variable_revision_before_realization(rbc_dynare_model: LinearModel):
    """News shock with lead k produces zero surprise revision for predetermined exogenous states at t < k."""
    m = rbc_dynare_model
    lead = 4
    res = news_irf(m, shock="eps", lead=lead, horizon=20, size=1.0)
    df = res.to_frame()

    # Exogenous technology state z must be exactly 0 for t = 0, 1, ..., lead - 1
    for t in range(lead):
        assert abs(df.loc[t, "z"]) < 1e-12, f"Expected z_{t} == 0 before realization at lead {lead}, got {df.loc[t, 'z']}"


def test_immediate_impact_jump_forward_looking(rbc_dynare_model: LinearModel):
    """Forward-looking controls jump on impact at t=0 to reflect anticipated future news."""
    m = rbc_dynare_model
    lead = 3
    res_news = news_irf(m, shock="eps", lead=lead, horizon=20, size=1.0)
    res_surp = news_irf(m, shock="eps", lead=0, horizon=20, size=1.0)

    c_news_0 = res_news.irf.loc[0, "c"]
    c_surp_0 = res_surp.irf.loc[0, "c"]

    # Forward-looking consumption jumps positively on impact at t=0
    assert c_news_0 > 0.1, f"Expected positive jump in consumption at t=0, got {c_news_0}"
    # News response is smaller than contemporaneous surprise shock impact
    assert c_news_0 < c_surp_0


def test_shock_realization_at_k(rbc_dynare_model: LinearModel):
    """At t=k, structural innovation realizes (z_k == size), and smoothly returns to steady state for t > k."""
    m = rbc_dynare_model
    size = 1.5
    lead = 3
    res = news_irf(m, shock="eps", lead=lead, horizon=400, size=size)
    df = res.irf

    # Realization at t = lead
    np.testing.assert_allclose(df.loc[lead, "z"], size, rtol=1e-8, atol=1e-10)

    # For t > lead, z decays by rho = 0.95
    rho = 0.95
    for h in range(lead + 1, lead + 10):
        expected_z = size * (rho ** (h - lead))
        np.testing.assert_allclose(
            df.loc[h, "z"], expected_z, rtol=1e-6, atol=1e-8,
            err_msg=f"Discrepancy in shock decay at horizon {h}",
        )

    # Terminal convergence to steady state
    np.testing.assert_allclose(df.iloc[-1].to_numpy(), 0.0, atol=1e-2)


def test_lead_zero_matches_surprise_irf(rbc_dynare_model: LinearModel):
    """When lead=0, news_irf matches model.irf within 10^-12 (machine precision)."""
    m = rbc_dynare_model
    for size in [0.5, 1.0, 2.5]:
        res_news = news_irf(m, shock="eps", lead=0, horizon=40, size=size)
        df_surp = m.irf("eps", horizon=40, size=size)

        max_abs_diff = np.max(np.abs(res_news.irf.to_numpy() - df_surp.to_numpy()))
        assert max_abs_diff < 1e-12, f"Discrepancy at size {size}: max diff = {max_abs_diff}"

        # Metadata attributes
        assert res_news.lead == 0
        assert res_news.shock == "eps"
        assert res_news.size == size
        assert res_news.horizon == 40
        assert res_news.model is m


def test_decompose_news_variance_shares_sum_to_one(rbc_dynare_model: LinearModel):
    """Forecast error variance shares sum to 1.0 across surprise and all news leads."""
    m = rbc_dynare_model
    max_lead = 6
    horizon = 40
    res = decompose_news(m, shock="eps", horizon=horizon, max_lead=max_lead)

    assert isinstance(res, NewsDecompositionResult)
    assert res.shock == "eps"
    assert res.horizon == horizon
    assert res.max_lead == max_lead
    assert res.model is m

    df = res.variance_shares
    assert df.shape == (len(m.variables), 1 + max_lead)
    expected_cols = ["surprise"] + [f"news_{k}" for k in range(1, max_lead + 1)]
    assert list(df.columns) == expected_cols

    # Every row sums to 1.0 within machine precision
    row_sums = df.sum(axis=1).to_numpy()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    # All shares are in [0, 1]
    assert (df.to_numpy() >= -1e-12).all()
    assert (df.to_numpy() <= 1.0 + 1e-12).all()

    # Total news property
    total_news = res.total_news
    assert isinstance(total_news, pd.Series)
    np.testing.assert_allclose(total_news + df["surprise"], 1.0, atol=1e-12)

    # Dynamic shares dictionary
    assert res.dynamic_shares is not None
    for var in m.variables:
        dyn_df = res.dynamic_shares[var]
        assert len(dyn_df) == horizon + 1
        np.testing.assert_allclose(dyn_df.sum(axis=1).to_numpy(), 1.0, atol=1e-12)


def test_plot_news_vs_surprise_headless(rbc_dynare_model: LinearModel):
    """plot_news_vs_surprise renders headlessly with Matplotlib Agg without error."""
    m = rbc_dynare_model
    fig, axes = plot_news_vs_surprise(
        m, shock="eps", leads=(0, 2, 4, 8), variables=["c", "k", "z"], horizon=20
    )
    assert isinstance(fig, plt.Figure)
    assert len(axes) >= 3

    # Canvas draw smoke test
    fig.canvas.draw()
    plt.close(fig)


def test_dynare_parser_anticipated_shocks_colon_syntax():
    """Dynare parser parses 'periods 1:4; values 0.01;' syntax into anticipated_shocks."""
    mod_text = """
    var c, k, z;
    varexo eps_a, eps_b;
    parameters alpha, beta, delta, rho, sigma;
    alpha = 0.33;
    beta = 0.99;
    delta = 0.025;
    rho = 0.95;
    sigma = 1.0;

    initval;
    c = 2.0;
    k = 25.0;
    z = 1.0;
    end;

    model;
    c^(-sigma) = beta * c(+1)^(-sigma) * (alpha * z(+1) * k^(alpha-1) + 1 - delta);
    c + k - (1-delta)*k(-1) = z * k(-1)^alpha;
    z = (1 - rho) + rho * z(-1) + eps_a + eps_b;
    end;

    shocks;
    var eps_a; stderr 0.01;
    var eps_b; stderr 0.01;
    var eps_a; periods 1:4; values 0.01;
    end;
    """
    parsed = parse_mod(mod_text)
    assert "anticipated" in parsed["shocks_config"]
    assert "eps_a" in parsed["shocks_config"]["anticipated"]
    assert parsed["shocks_config"]["anticipated"]["eps_a"]["periods"] == [1, 2, 3, 4]
    assert parsed["shocks_config"]["anticipated"]["eps_a"]["values"] == [0.01, 0.01, 0.01, 0.01]

    # Model carries anticipated_shocks attribute
    m = load_mod(mod_text)
    assert m.anticipated_shocks is not None
    assert "eps_a" in m.anticipated_shocks
    assert m.anticipated_shocks["eps_a"]["periods"] == [1, 2, 3, 4]


def test_dynare_parser_anticipated_shocks_comma_syntax():
    """Dynare parser parses 'periods 1, 2, 3; values 0.02, 0.01, 0.005;' comma syntax."""
    mod_text = """
    var c, k, z;
    varexo eps_a;
    parameters alpha, beta, delta, rho, sigma;
    alpha = 0.33;
    beta = 0.99;
    delta = 0.025;
    rho = 0.95;
    sigma = 1.0;

    initval;
    c = 2.0;
    k = 25.0;
    z = 1.0;
    end;

    model;
    c^(-sigma) = beta * c(+1)^(-sigma) * (alpha * z(+1) * k^(alpha-1) + 1 - delta);
    c + k - (1-delta)*k(-1) = z * k(-1)^alpha;
    z = (1 - rho) + rho * z(-1) + eps_a;
    end;

    shocks;
    var eps_a; stderr 0.01;
    var eps_a; periods 1, 2, 3; values 0.02, 0.01, 0.005;
    end;
    """
    m = load_mod(mod_text)
    assert m.anticipated_shocks is not None
    assert "eps_a" in m.anticipated_shocks
    assert m.anticipated_shocks["eps_a"]["periods"] == [1, 2, 3]
    assert m.anticipated_shocks["eps_a"]["values"] == [0.02, 0.01, 0.005]


def test_dynare_parser_step_colon_syntax():
    """Dynare parser parses 'periods 1:2:7; values 0.05;' step range syntax."""
    mod_text = """
    var c, k, z;
    varexo eps_a;
    parameters alpha, beta, delta, rho, sigma;
    alpha = 0.33;
    beta = 0.99;
    delta = 0.025;
    rho = 0.95;
    sigma = 1.0;

    initval;
    c = 2.0;
    k = 25.0;
    z = 1.0;
    end;

    model;
    c^(-sigma) = beta * c(+1)^(-sigma) * (alpha * z(+1) * k^(alpha-1) + 1 - delta);
    c + k - (1-delta)*k(-1) = z * k(-1)^alpha;
    z = (1 - rho) + rho * z(-1) + eps_a;
    end;

    shocks;
    var eps_a; stderr 0.01;
    var eps_a; periods 1:2:7; values 0.05;
    end;
    """
    m = load_mod(mod_text)
    assert m.anticipated_shocks is not None
    assert m.anticipated_shocks["eps_a"]["periods"] == [1, 3, 5, 7]
    assert m.anticipated_shocks["eps_a"]["values"] == [0.05, 0.05, 0.05, 0.05]


def test_news_irf_result_presentation_contract(rbc_dynare_model: LinearModel):
    """Test full presentation contract: .to_frame(), .summary(), .plot(), markdown/latex/typst."""
    m = rbc_dynare_model
    res = news_irf(m, shock="eps", lead=2, horizon=15)

    df = res.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 16

    summary_str = res.summary()
    assert isinstance(summary_str, str)
    assert "News Shock IRF Summary" in summary_str
    assert "Impact (t=0)" in summary_str

    md_str = res.to_markdown()
    assert isinstance(md_str, str)
    assert "|" in md_str

    latex_str = res.to_latex()
    assert isinstance(latex_str, str)
    assert "\\begin{tabular}" in latex_str or "\\toprule" in latex_str or "\\begin" in latex_str

    typst_str = res.to_typst()
    assert isinstance(typst_str, str)
    assert "#table" in typst_str or "[" in typst_str

    fig, ax = res.plot(variables=["c", "k"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Decomposition result presentation contract
    decomp = decompose_news(m, shock="eps", horizon=15, max_lead=3)
    assert isinstance(decomp.to_frame(), pd.DataFrame)
    assert "News Shock Variance Decomposition" in decomp.summary()
    assert "|" in decomp.to_markdown()
    assert len(decomp.to_latex()) > 0
    assert len(decomp.to_typst()) > 0

    fig_d, ax_d = decomp.plot()
    assert isinstance(fig_d, plt.Figure)
    plt.close(fig_d)


def test_news_irf_klein_timing(rbc_klein_model: LinearModel):
    """Verify news IRF and state-space augmentation on models built with build() (Klein timing)."""
    m = rbc_klein_model
    assert m.timing == "klein"

    # lead=0 matches surprise IRF to machine precision
    res_0 = news_irf(m, shock="eps", lead=0, horizon=10)
    irf_surp = m.irf("eps", horizon=10)
    np.testing.assert_allclose(res_0.irf.to_numpy(), irf_surp.to_numpy(), atol=1e-12)

    # lead=2 news shock
    res_2 = news_irf(m, shock="eps", lead=2, horizon=10)
    df_2 = res_2.irf
    # At t=0 and t=1, exogenous state z is zero
    assert abs(df_2.loc[0, "z"]) < 1e-12
    assert abs(df_2.loc[1, "z"]) < 1e-12


def test_linearmodel_convenience_method(rbc_dynare_model: LinearModel):
    """LinearModel.news_irf delegates accurately to puremacro.dsge.news.news_irf."""
    m = rbc_dynare_model
    res1 = m.news_irf("eps", lead=3, horizon=25, size=2.0)
    res2 = news_irf(m, "eps", lead=3, horizon=25, size=2.0)

    assert isinstance(res1, NewsIRFResult)
    np.testing.assert_allclose(res1.irf.to_numpy(), res2.irf.to_numpy(), atol=1e-15)


def test_error_handling_invalid_inputs(rbc_dynare_model: LinearModel):
    """Verify informative error handling on invalid parameters."""
    m = rbc_dynare_model

    with pytest.raises(ModelError, match="no shock named 'nonexistent'"):
        news_irf(m, shock="nonexistent")

    with pytest.raises(ValueError, match="lead must be non-negative"):
        news_irf(m, shock="eps", lead=-1)

    with pytest.raises(ValueError, match="horizon must be non-negative"):
        news_irf(m, shock="eps", horizon=-5)

    with pytest.raises(ModelError, match="no shock named 'nonexistent'"):
        augment_news_state_space(m, shock="nonexistent")

    with pytest.raises(ValueError, match="max_lead must be at least 1"):
        augment_news_state_space(m, shock="eps", max_lead=0)

    with pytest.raises(ModelError, match="no shock named 'nonexistent'"):
        decompose_news(m, shock="nonexistent")

    with pytest.raises(ValueError, match="max_lead must be at least 1"):
        decompose_news(m, shock="eps", max_lead=0)
