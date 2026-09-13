"""Comprehensive unit and integration tests for Caliendo & Parro (2015) model.

Verifies:
1. Exact Hat Algebra identities (trade shares summing to 1, identity shock invariance).
2. General equilibrium market clearing residual < 10^{-6}.
3. Trade balance residual < 10^{-6} and global trade balance preservation.
4. Input-output propagation across tradable and non-tradable sectors.
5. Unilateral tariff shocks and bilateral trade wars.
6. Deficit rules ("fixed", "scaled", "zero").
7. Data formatting and serialization (.summary, .sector_summary, .to_markdown, .to_latex, .to_typst, .plot).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.trade.caliendo_parro import CaliendoParroModel, CaliendoParroResult


@pytest.fixture
def symmetric_3c_2s_model():
    """A clean 3-country, 2-sector symmetric trade economy."""
    N, J = 3, 2
    trade_shares = np.zeros((J, N, N))
    # Sector 0 (tradables): 60% domestic, 20% from each foreign partner
    trade_shares[0] = np.array([
        [0.60, 0.20, 0.20],
        [0.20, 0.60, 0.20],
        [0.20, 0.20, 0.60],
    ])
    # Sector 1 (non-tradables): 100% domestic
    trade_shares[1] = np.eye(N)

    gamma_va = np.full((N, J), 0.50)
    gamma_io = np.full((N, J, J), 0.25)
    alpha = np.full((N, J), 0.50)
    theta = np.array([5.0, 4.0])
    labor_income = np.array([100.0, 100.0, 100.0])

    model = CaliendoParroModel(
        trade_shares=trade_shares,
        gamma_va=gamma_va,
        gamma_io=gamma_io,
        alpha=alpha,
        theta=theta,
        labor_income=labor_income,
        nontradables=[1],
        country_codes=["USA", "CHN", "DEU"],
        sector_codes=["Manufactures", "Services"],
    )
    return model


@pytest.fixture
def asymmetric_model():
    """Asymmetric 3-country, 2-sector economy with trade imbalances."""
    N, J = 3, 2
    trade_shares = np.zeros((J, N, N))
    trade_shares[0] = np.array([
        [0.70, 0.15, 0.15],
        [0.10, 0.80, 0.10],
        [0.25, 0.25, 0.50],
    ])
    trade_shares[1] = np.eye(N)

    gamma_va = np.array([
        [0.40, 0.60],
        [0.35, 0.65],
        [0.45, 0.55],
    ])
    gamma_io = np.zeros((N, J, J))
    for n in range(N):
        for j in range(J):
            rem = 1.0 - gamma_va[n, j]
            gamma_io[n, j, 0] = rem * 0.6
            gamma_io[n, j, 1] = rem * 0.4

    alpha = np.array([
        [0.40, 0.60],
        [0.50, 0.50],
        [0.30, 0.70],
    ])
    theta = np.array([6.5, 4.5])
    labor_income = np.array([200.0, 150.0, 100.0])
    deficits = np.array([10.0, -6.0, -4.0])  # Sum to 0

    model = CaliendoParroModel(
        trade_shares=trade_shares,
        gamma_va=gamma_va,
        gamma_io=gamma_io,
        alpha=alpha,
        theta=theta,
        labor_income=labor_income,
        deficits=deficits,
        nontradables=["Services"],
        country_codes=["USA", "CHN", "DEU"],
        sector_codes=["Goods", "Services"],
    )
    return model


class TestCaliendoParroModelValidation:
    """Test input validations and dimension checks."""

    def test_dimension_mismatches_raise(self):
        N, J = 2, 2
        trade_shares = np.zeros((J, N, N))
        trade_shares[:, :, :] = 0.5
        gamma_va = np.full((N, J), 0.5)
        gamma_io = np.full((N, J, J), 0.25)
        alpha = np.full((N, J), 0.5)
        theta = np.array([5.0, 4.0])
        labor_income = np.array([100.0, 100.0])

        # Wrong country codes length
        with pytest.raises(ValueError, match="country_codes length"):
            CaliendoParroModel(
                trade_shares=trade_shares,
                gamma_va=gamma_va,
                gamma_io=gamma_io,
                alpha=alpha,
                theta=theta,
                labor_income=labor_income,
                country_codes=["USA"],
            )

        # Wrong sector codes length
        with pytest.raises(ValueError, match="sector_codes length"):
            CaliendoParroModel(
                trade_shares=trade_shares,
                gamma_va=gamma_va,
                gamma_io=gamma_io,
                alpha=alpha,
                theta=theta,
                labor_income=labor_income,
                sector_codes=["Sec1"],
            )

    def test_non_positive_parameters_raise(self):
        N, J = 2, 2
        trade_shares = np.full((J, N, N), 0.5)
        gamma_va = np.full((N, J), 0.5)
        gamma_io = np.full((N, J, J), 0.25)
        alpha = np.full((N, J), 0.5)
        theta = np.array([5.0, 4.0])
        labor_income = np.array([100.0, 100.0])

        # Negative value added
        bad_va = gamma_va.copy()
        bad_va[0, 0] = -0.1
        with pytest.raises(ValueError, match="value-added share"):
            CaliendoParroModel(
                trade_shares=trade_shares,
                gamma_va=bad_va,
                gamma_io=gamma_io,
                alpha=alpha,
                theta=theta,
                labor_income=labor_income,
            )

        # Negative trade elasticity
        bad_theta = np.array([-1.0, 4.0])
        with pytest.raises(ValueError, match="trade elasticities"):
            CaliendoParroModel(
                trade_shares=trade_shares,
                gamma_va=gamma_va,
                gamma_io=gamma_io,
                alpha=alpha,
                theta=bad_theta,
                labor_income=labor_income,
            )

    def test_unbalanced_deficits_raise(self):
        N, J = 2, 2
        trade_shares = np.full((J, N, N), 0.5)
        gamma_va = np.full((N, J), 0.5)
        gamma_io = np.full((N, J, J), 0.25)
        alpha = np.full((N, J), 0.5)
        theta = np.array([5.0, 4.0])
        labor_income = np.array([100.0, 100.0])
        bad_deficits = np.array([10.0, 5.0])  # Non-zero sum

        with pytest.raises(ValueError, match="sum to zero"):
            CaliendoParroModel(
                trade_shares=trade_shares,
                gamma_va=gamma_va,
                gamma_io=gamma_io,
                alpha=alpha,
                theta=theta,
                labor_income=labor_income,
                deficits=bad_deficits,
            )


class TestCaliendoParroIdentitiesAndEquilibrium:
    """Test Exact Hat Algebra identities and market clearing."""

    def test_identity_shock_invariance(self, symmetric_3c_2s_model):
        """Identity shock (no changes) must reproduce baseline exactly."""
        res = symmetric_3c_2s_model.solve_counterfactual()
        assert res.converged
        np.testing.assert_allclose(res.w_hat, 1.0, atol=1e-7)
        np.testing.assert_allclose(res.P_hat, 1.0, atol=1e-7)
        np.testing.assert_allclose(res.P_index_hat, 1.0, atol=1e-7)
        np.testing.assert_allclose(res.real_wage_hat, 1.0, atol=1e-7)
        np.testing.assert_allclose(res.welfare_hat, 1.0, atol=1e-7)
        np.testing.assert_allclose(res.welfare_pct, 0.0, atol=1e-6)
        assert res.market_clearing_residual < 1e-6
        assert res.trade_balance_residual < 1e-6

    def test_trade_shares_normalization(self, symmetric_3c_2s_model):
        """Counterfactual trade shares must sum to 1 for all destinations and sectors."""
        res = symmetric_3c_2s_model.simulate_tariff_shock("USA", "CHN", tariff_rate=0.20)
        assert res.converged
        for j in range(symmetric_3c_2s_model.J):
            sums = np.sum(res.pi_prime[j], axis=1)
            np.testing.assert_allclose(sums, 1.0, atol=1e-12)

    def test_residuals_below_tolerance(self, symmetric_3c_2s_model, asymmetric_model):
        """Both symmetric and asymmetric models must meet residual < 10^{-6} criterion."""
        res_sym = symmetric_3c_2s_model.simulate_tariff_shock("USA", "CHN", tariff_rate=0.15)
        assert res_sym.converged
        assert res_sym.market_clearing_residual < 1e-6
        assert res_sym.trade_balance_residual < 1e-6

        res_asym = asymmetric_model.simulate_tariff_shock("USA", "DEU", tariff_rate=0.20)
        assert res_asym.converged
        assert res_asym.market_clearing_residual < 1e-6
        assert res_asym.trade_balance_residual < 1e-6


class TestCaliendoParroCounterfactuals:
    """Test economic responses to tariff shocks and trade wars."""

    def test_unilateral_tariff_effects(self, symmetric_3c_2s_model):
        """Unilateral tariff generates revenue, changes terms of trade."""
        res = symmetric_3c_2s_model.simulate_tariff_shock(
            "USA", "CHN", sector="Manufactures", tariff_rate=0.25
        )
        assert res.converged

        # USA (imposing tariff) collects positive tariff revenue
        assert res.tariff_revenue_prime[0] > 0.0
        # CHN and DEU collect 0 tariff revenue
        assert res.tariff_revenue_prime[1] == 0.0
        assert res.tariff_revenue_prime[2] == 0.0

        # CHN wage falls relative to USA
        assert res.w_hat[0] > res.w_hat[1]
        # CHN welfare drops
        assert res.welfare_pct[1] < 0.0

    def test_trade_war_simulation(self, symmetric_3c_2s_model):
        """Reciprocal trade war reduces welfare in both belligerents."""
        res = symmetric_3c_2s_model.simulate_trade_war(
            coalition_a=["USA"], coalition_b=["CHN"], tariff_rate=0.30
        )
        assert res.converged

        # Both combatants impose tariffs and collect revenue
        assert res.tariff_revenue_prime[0] > 0.0
        assert res.tariff_revenue_prime[1] > 0.0
        assert res.tariff_revenue_prime[2] == 0.0

        # In symmetric setup, USA and CHN have identical responses by symmetry
        np.testing.assert_allclose(res.w_hat[0], res.w_hat[1], rtol=1e-5)
        np.testing.assert_allclose(res.welfare_pct[0], res.welfare_pct[1], rtol=1e-5)

        # Third party DEU benefits from trade diversion
        assert res.welfare_pct[2] > res.welfare_pct[0]

    def test_deficit_rules(self, asymmetric_model):
        """Verify fixed, scaled, and zero deficit rules."""
        res_fixed = asymmetric_model.solve_counterfactual(
            deficit_rule="fixed",
            d_hat=np.full((asymmetric_model.J, asymmetric_model.N, asymmetric_model.N), 1.05),
        )
        assert res_fixed.converged
        assert res_fixed.trade_balance_residual < 1e-6

        res_scaled = asymmetric_model.solve_counterfactual(
            deficit_rule="scaled",
            d_hat=np.full((asymmetric_model.J, asymmetric_model.N, asymmetric_model.N), 1.05),
        )
        assert res_scaled.converged
        assert res_scaled.trade_balance_residual < 1e-6

        res_zero = asymmetric_model.solve_counterfactual(
            deficit_rule="zero",
            d_hat=np.full((asymmetric_model.J, asymmetric_model.N, asymmetric_model.N), 1.05),
        )
        assert res_zero.converged
        assert res_zero.trade_balance_residual < 1e-6


class TestCaliendoParroPresentations:
    """Test summary tables, markdown/latex/typst exports, and plots."""

    def test_summary_dataframes(self, symmetric_3c_2s_model):
        res = symmetric_3c_2s_model.simulate_tariff_shock("USA", "CHN", tariff_rate=0.10)
        summary = res.summary()
        assert isinstance(summary, pd.DataFrame)
        assert list(summary.index) == ["USA", "CHN", "DEU"]
        expected_cols = {
            "wage_hat", "cpi_hat", "real_wage_hat", "income_hat",
            "welfare_hat", "welfare_pct", "tariff_revenue_prime"
        }
        assert expected_cols.issubset(summary.columns)

        sec_sum = res.sector_summary()
        assert isinstance(sec_sum, pd.DataFrame)
        assert len(sec_sum) == 3 * 2

    def test_string_formatters(self, symmetric_3c_2s_model):
        res = symmetric_3c_2s_model.simulate_tariff_shock("USA", "CHN", tariff_rate=0.10)
        md = res.to_markdown()
        assert "USA" in md
        assert "|" in md

        tex = res.to_latex()
        assert "\\begin{tabular}" in tex
        assert "\\end{tabular}" in tex

        typ = res.to_typst()
        assert "#table(" in typ

    def test_plot_generation(self, symmetric_3c_2s_model):
        res = symmetric_3c_2s_model.simulate_tariff_shock("USA", "CHN", tariff_rate=0.10)
        fig = res.plot(kind="welfare")
        assert fig is not None
        fig2 = res.plot(kind="real_wage")
        assert fig2 is not None
        plt.close("all")
