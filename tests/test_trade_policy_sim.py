"""Comprehensive unit and integration tests for TradePolicySimulator.

Verifies:
1. General equilibrium goods market clearing: max_i |X_i - (Y_i + R_i + D_i)| < 10^-6
   and labor market clearing under arbitrary bilateral tariff adjustments.
2. Exact Hat Algebra identity shock invariance.
3. Terms-of-trade effects (ToT_hat) and real wage adjustments.
4. Caliendo-Parro (2015) welfare decomposition (terms of trade, I-O multiplier, tariff revenue).
5. Trade diversion effects (e.g. US tariff on China diverting demand toward Mexico).
6. Reciprocal trade war counterfactuals.
7. Presentation export methods (.summary, .sector_summary, .to_frame, .to_markdown, .to_latex, .to_typst, .plot).
8. Presets and ICIO benchmark loader.
"""
from __future__ import annotations

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from puremacro.models.trade_policy import (
    TradePolicySimulationResult,
    TradePolicySimulator,
)
from puremacro.trade.caliendo_parro import CaliendoParroModel


@pytest.fixture
def sim_nafta_china() -> TradePolicySimulator:
    """Canonical NAFTA-China 3-country, 2-sector simulator."""
    return TradePolicySimulator.from_preset("nafta_china")


@pytest.fixture
def sim_symmetric() -> TradePolicySimulator:
    """Symmetric 3-country simulator."""
    return TradePolicySimulator.from_preset("symmetric_3c")


def test_identity_shock_invariance(sim_nafta_china: TradePolicySimulator) -> None:
    """Identity tariff shock (no change) must leave all economic variables unchanged."""
    res = sim_nafta_china.simulate_tariff_counterfactual(tol=1e-12)

    assert res.converged
    np.testing.assert_allclose(res.w_hat, 1.0, atol=1e-10)
    np.testing.assert_allclose(res.P_index_hat, 1.0, atol=1e-10)
    np.testing.assert_allclose(res.real_wage_hat, 1.0, atol=1e-10)
    np.testing.assert_allclose(res.terms_of_trade_hat, 1.0, atol=1e-10)
    np.testing.assert_allclose(res.welfare_pct, 0.0, atol=1e-10)
    assert res.market_clearing_residual < 1e-6


def test_market_clearing_convergence_bilateral_tariff(sim_nafta_china: TradePolicySimulator) -> None:
    """Bilateral tariff shock must converge with market clearing residual < 10^-6."""
    res = sim_nafta_china.simulate_bilateral_tariff("USA", "CHN", tariff_rate=0.25, tol=1e-10)

    assert res.converged
    assert res.market_clearing_residual < 1e-6

    # Goods market clearing check: X_i = Y_i + R_i + D_i
    X_tot = np.sum(res.X_prime, axis=1)
    Y_tot = np.sum(res.Y_prime, axis=1)
    R_tot = res.tariff_revenue_prime
    D_tot = sim_nafta_china.model.deficits
    diff = np.abs(X_tot - (Y_tot + R_tot + D_tot))
    assert np.max(diff) < 1e-6


def test_terms_of_trade_and_trade_diversion(sim_nafta_china: TradePolicySimulator) -> None:
    """Tariff on China reduces China's ToT and diverts US import demand toward Mexico."""
    res = sim_nafta_china.simulate_bilateral_tariff("USA", "CHN", tariff_rate=0.30, tol=1e-10)

    # Country index: 0=MEX, 1=USA, 2=CHN
    idx_mex, idx_usa, idx_chn = 0, 1, 2

    # US collects positive tariff revenue
    assert res.tariff_revenue_prime[idx_usa] > 0.0
    assert res.tariff_revenue_prime[idx_mex] == 0.0

    # China terms of trade deteriorates (less than 1.0)
    assert res.terms_of_trade_hat[idx_chn] < 1.0

    # Mexico terms of trade relative to China improves
    assert res.terms_of_trade_hat[idx_mex] > res.terms_of_trade_hat[idx_chn]
    assert res.real_wage_hat[idx_mex] > 1.0

    # US manufacturing trade share from Mexico increases (trade diversion)
    base_share_mex = sim_nafta_china.model.trade_shares[0, idx_usa, idx_mex]
    prime_share_mex = res.pi_prime[0, idx_usa, idx_mex]
    assert prime_share_mex > base_share_mex

    # US manufacturing trade share from China decreases
    base_share_chn = sim_nafta_china.model.trade_shares[0, idx_usa, idx_chn]
    prime_share_chn = res.pi_prime[0, idx_usa, idx_chn]
    assert prime_share_chn < base_share_chn

    # Mexico gains positive welfare from trade diversion
    assert res.welfare_pct[idx_mex] > 0.0


def test_trade_war_simulation(sim_nafta_china: TradePolicySimulator) -> None:
    """Reciprocal trade war between USA and China hurts both belligerents."""
    res = sim_nafta_china.simulate_trade_war(
        coalition_a=["USA"],
        coalition_b=["CHN"],
        tariff_rate_a=0.25,
        tariff_rate_b=0.25,
        tol=1e-10,
    )

    assert res.converged
    assert res.market_clearing_residual < 1e-6

    # Both belligerents experience welfare loss
    assert res.welfare_pct[1] < 0.0  # USA
    assert res.welfare_pct[2] < 0.0  # CHN

    # Non-participating partner (MEX) benefits from trade diversion
    assert res.welfare_pct[0] > 0.0  # MEX


def test_arbitrary_tariffs_randomized_convergence(sim_symmetric: TradePolicySimulator) -> None:
    """Randomized arbitrary bilateral tariff matrices must satisfy market clearing < 10^-6."""
    rng = np.random.default_rng(42)
    J, N = sim_symmetric.model.J, sim_symmetric.model.N

    for trial in range(5):
        # Generate random tariff rates between 0% and 50%
        random_rates = rng.uniform(0.0, 0.5, size=(J, N, N))
        for j in range(J):
            np.fill_diagonal(random_rates[j], 0.0)  # zero domestic tariff
        tariffs_new = 1.0 + random_rates

        res = sim_symmetric.simulate_arbitrary_tariffs(tariffs_new, tol=1e-10)
        assert res.converged, f"Trial {trial} failed to converge"
        assert res.market_clearing_residual < 1e-6, f"Trial {trial} residual {res.market_clearing_residual} >= 1e-6"


def test_welfare_decomposition_consistency(sim_nafta_china: TradePolicySimulator) -> None:
    """Welfare decomposition terms must sum to total log welfare change."""
    res = sim_nafta_china.simulate_bilateral_tariff("USA", "CHN", tariff_rate=0.20, tol=1e-10)

    tot = res.welfare_decomposition["terms_of_trade"]
    io = res.welfare_decomposition["input_output"]
    rev = res.welfare_decomposition["tariff_revenue"]
    total = res.welfare_decomposition["total"]

    sum_decomp = tot + io + rev
    np.testing.assert_allclose(sum_decomp, total, atol=1e-12)


def test_presentation_export_methods(sim_nafta_china: TradePolicySimulator) -> None:
    """Presentation methods (.summary, .sector_summary, .to_markdown, .to_latex, .to_typst, .plot) work."""
    res = sim_nafta_china.simulate_bilateral_tariff("USA", "CHN", tariff_rate=0.15, tol=1e-10)

    # Summary DataFrame
    df = res.summary()
    assert isinstance(df, pd.DataFrame)
    assert "welfare_pct" in df.columns
    assert "terms_of_trade_hat" in df.columns
    assert list(df.index) == ["MEX", "USA", "CHN"]

    # Sector summary
    sec_df = res.sector_summary()
    assert isinstance(sec_df, pd.DataFrame)
    assert len(sec_df) == 3 * 2

    # to_frame
    frame = res.to_frame()
    assert isinstance(frame, pd.DataFrame)

    # Markdown
    md = res.to_markdown()
    assert isinstance(md, str)
    assert "| country |" in md
    assert "MEX" in md

    # LaTeX
    tex = res.to_latex()
    assert isinstance(tex, str)
    assert "\\begin{table}" in tex or "\\begin{tabular}" in tex

    # Typst
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ

    # Plot
    fig = res.plot(kind="welfare")
    assert isinstance(fig, matplotlib.figure.Figure)

    fig_tot = res.plot(kind="terms_of_trade")
    assert isinstance(fig_tot, matplotlib.figure.Figure)

    fig_rw = res.plot(kind="real_wage")
    assert isinstance(fig_rw, matplotlib.figure.Figure)


def test_flexible_counterfactual_inputs(sim_nafta_china: TradePolicySimulator) -> None:
    """simulate_tariff_counterfactual supports dict and array specifications."""
    # Dict with (imp, exp)
    res_dict = sim_nafta_china.simulate_tariff_counterfactual({("USA", "CHN"): 0.25})
    assert res_dict.converged
    assert res_dict.market_clearing_residual < 1e-6

    # Dict with (imp, exp, sec)
    res_sec = sim_nafta_china.simulate_tariff_counterfactual({("USA", "CHN", "Manufactures"): 0.25})
    assert res_sec.converged
    assert res_sec.market_clearing_residual < 1e-6

    # Non-tradable tariff error
    with pytest.raises(ValueError, match="Cannot apply tariff to non-tradable"):
        sim_nafta_china.simulate_tariff_counterfactual({("USA", "CHN", "Services"): 0.25})


def test_preset_and_icio_loader() -> None:
    """Presets and ICIO loaders instantiate valid simulators."""
    sim_icio = TradePolicySimulator.from_icio(2021)
    assert isinstance(sim_icio, TradePolicySimulator)
    assert sim_icio.model.N == 3

    with pytest.raises(ValueError, match="Unknown preset"):
        TradePolicySimulator.from_preset("invalid_preset_name")
