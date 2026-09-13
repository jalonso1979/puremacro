"""Adversarial stress-test suite for Caliendo-Parro (2015) and Allen-Arkolakis (2014) GE engines.

Author: Challenger 2 (orchestrator_16 adversarial verification)
Scope:
1. Caliendo-Parro (2015):
   - Near-autarky trade costs (tau -> 100, 500)
   - Extreme trade elasticities (theta = 1.05 and theta = 25.0)
   - Multi-country multi-sector tariff retaliation wars
   - Non-tradable sector shocks and input-output propagation
   - Verification of goods market clearing (< 1e-6) and trade balance residuals (< 1e-6)
2. Allen-Arkolakis (2014):
   - Exact uniqueness boundary alpha + beta = theta / (1 + theta)
   - Asymmetric trade cost matrices (tau_{ij} != tau_{ji})
   - Localized climate and environmental shocks (mild, severe, and 99% catastrophe)
   - Verification of labor mass conservation (|sum L_i - L_bar| < 1e-12)
   - Verification of spatial price and utility equalization (std(u_i) / u_bar < 1e-7)
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.spatial.allen_arkolakis import AllenArkolakisModel
from puremacro.trade.caliendo_parro import CaliendoParroModel


# =============================================================================
# CALIENDO-PARRO (2015) ADVERSARIAL TEST SUITE
# =============================================================================

class TestCaliendoParroAdversarial:
    """Adversarial challenge tests for Caliendo-Parro (2015) trade GE engine."""

    def test_near_autarky_symmetric(self):
        """Stress-test near-autarky trade costs (tau -> 100 and 500) in symmetric economy."""
        N, J = 3, 2
        trade_shares = np.zeros((J, N, N))
        trade_shares[0] = np.array([
            [0.60, 0.20, 0.20],
            [0.20, 0.60, 0.20],
            [0.20, 0.20, 0.60],
        ])
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

        for tau_val in [100.0, 500.0]:
            tariffs_new = np.ones((J, N, N))
            for n in range(N):
                for i in range(N):
                    if n != i:
                        tariffs_new[0, n, i] = tau_val

            res = model.solve_counterfactual(tariffs_new=tariffs_new)
            assert res.converged, f"Failed to converge at tau={tau_val}"
            assert res.market_clearing_residual < 1e-6, f"Market clearing res {res.market_clearing_residual} >= 1e-6"
            assert res.trade_balance_residual < 1e-6, f"Trade balance res {res.trade_balance_residual} >= 1e-6"

            # Foreign trade shares must collapse toward 0 (< 1e-10)
            off_diag = res.pi_prime[0] - np.diag(np.diag(res.pi_prime[0]))
            assert np.max(off_diag) < 1e-10, f"Off-diagonal trade shares {np.max(off_diag)} >= 1e-10"
            np.testing.assert_allclose(np.diag(res.pi_prime[0]), 1.0, atol=1e-10)
            # Non-tradable sector trade shares strictly identity
            np.testing.assert_allclose(res.pi_prime[1], np.eye(N), atol=1e-12)

    def test_near_autarky_asymmetric_zero_deficits(self):
        """Stress-test near-autarky trade costs in asymmetric economy with zero deficits."""
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
        deficits = np.array([10.0, -6.0, -4.0])

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

        tariffs_new = np.ones((J, N, N))
        for n in range(N):
            for i in range(N):
                if n != i:
                    tariffs_new[0, n, i] = 100.0

        # Under autarky, deficit_rule="zero" is the economically valid specification
        res = model.solve_counterfactual(tariffs_new=tariffs_new, deficit_rule="zero")
        assert res.converged
        assert res.market_clearing_residual < 1e-6
        assert res.trade_balance_residual < 1e-6
        np.testing.assert_allclose(res.w_hat, 1.0, atol=1e-8)
        np.testing.assert_allclose(res.pi_prime[0], np.eye(N), atol=1e-10)

    @pytest.mark.parametrize("th_val", [1.05, 25.0])
    def test_extreme_trade_elasticities(self, th_val: float):
        """Stress-test extreme trade elasticities theta=1.05 (near Cobb-Douglas) and theta=25.0 (homogeneous)."""
        N, J = 3, 2
        trade_shares = np.zeros((J, N, N))
        trade_shares[0] = np.array([
            [0.70, 0.15, 0.15],
            [0.10, 0.80, 0.10],
            [0.25, 0.25, 0.50],
        ])
        trade_shares[1] = np.eye(N)

        gamma_va = np.array([[0.40, 0.60], [0.35, 0.65], [0.45, 0.55]])
        gamma_io = np.zeros((N, J, J))
        for n in range(N):
            for j in range(J):
                rem = 1.0 - gamma_va[n, j]
                gamma_io[n, j, 0] = rem * 0.6
                gamma_io[n, j, 1] = rem * 0.4

        alpha = np.array([[0.40, 0.60], [0.50, 0.50], [0.30, 0.70]])
        labor_income = np.array([200.0, 150.0, 100.0])

        model = CaliendoParroModel(
            trade_shares=trade_shares,
            gamma_va=gamma_va,
            gamma_io=gamma_io,
            alpha=alpha,
            theta=np.array([th_val, 4.0]),
            labor_income=labor_income,
            nontradables=["Services"],
            country_codes=["USA", "CHN", "DEU"],
            sector_codes=["Goods", "Services"],
        )

        # Unilateral tariff shock
        res_tariff = model.simulate_tariff_shock("USA", "CHN", tariff_rate=0.20)
        assert res_tariff.converged, f"Tariff shock failed to converge at theta={th_val}"
        assert res_tariff.market_clearing_residual < 1e-6
        assert res_tariff.trade_balance_residual < 1e-6

        # Reciprocal trade war
        res_war = model.simulate_trade_war(["USA"], ["CHN"], tariff_rate=0.25)
        assert res_war.converged, f"Trade war failed to converge at theta={th_val}"
        assert res_war.market_clearing_residual < 1e-6
        assert res_war.trade_balance_residual < 1e-6

    def test_multi_country_tariff_retaliation_war(self):
        """Stress-test 5-country, 3-sector complex tariff retaliation war with trade diversion."""
        N, J = 5, 3
        rng = np.random.default_rng(123)

        trade_shares = np.zeros((J, N, N))
        for j in range(2):
            raw = rng.uniform(0.1, 0.5, size=(N, N))
            np.fill_diagonal(raw, rng.uniform(2.0, 4.0, size=N))
            trade_shares[j] = raw / np.sum(raw, axis=1, keepdims=True)
        trade_shares[2] = np.eye(N)  # Non-tradable sector

        gamma_va = rng.uniform(0.3, 0.6, size=(N, J))
        gamma_io = np.zeros((N, J, J))
        for n in range(N):
            for j in range(J):
                rem = 1.0 - gamma_va[n, j]
                io_raw = rng.uniform(0.1, 0.5, size=J)
                gamma_io[n, j] = rem * (io_raw / np.sum(io_raw))

        alpha_raw = rng.uniform(0.2, 0.8, size=(N, J))
        alpha = alpha_raw / np.sum(alpha_raw, axis=1, keepdims=True)

        theta = np.array([5.0, 8.0, 4.0])
        labor_income = np.array([500.0, 400.0, 350.0, 200.0, 150.0])
        countries = ["USA", "CHN", "EU", "JPN", "GBR"]
        sectors = ["Mfg", "Agri", "Services"]

        model = CaliendoParroModel(
            trade_shares=trade_shares,
            gamma_va=gamma_va,
            gamma_io=gamma_io,
            alpha=alpha,
            theta=theta,
            labor_income=labor_income,
            nontradables=[2],
            country_codes=countries,
            sector_codes=sectors,
        )

        for rate in [0.20, 0.40]:
            res = model.simulate_trade_war(
                coalition_a=["USA", "GBR"],
                coalition_b=["CHN", "EU"],
                tariff_rate=rate,
            )
            assert res.converged
            assert res.market_clearing_residual < 1e-6
            assert res.trade_balance_residual < 1e-6

            # Non-tradable goods market clearing Y_n^2 == X_n^2
            np.testing.assert_allclose(res.Y_prime[:, 2], res.X_prime[:, 2], rtol=1e-6)
            # Non-tradable bilateral shares remain strictly identity
            np.testing.assert_allclose(res.pi_prime[2], np.eye(N), atol=1e-12)
            # Neutral JPN benefits from trade diversion
            assert res.welfare_pct[3] > 0.0

    def test_nontradable_sector_shock_propagation(self):
        """Verify input-output propagation into non-tradables and exact goods market clearing."""
        N, J = 3, 2
        trade_shares = np.array([
            [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]],
            np.eye(3),
        ])
        gamma_va = np.full((N, J), 0.5)
        gamma_io = np.zeros((N, J, J))
        gamma_io[:, 0, 0] = 0.25
        gamma_io[:, 0, 1] = 0.25
        gamma_io[:, 1, 0] = 0.30
        gamma_io[:, 1, 1] = 0.20

        alpha = np.full((N, J), 0.5)
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
        )

        res = model.simulate_tariff_shock("C0", "C1", sector="S0", tariff_rate=0.50)
        assert res.converged
        assert res.market_clearing_residual < 1e-6
        assert res.trade_balance_residual < 1e-6

        # Non-tradable prices must rise in C0 due to expensive intermediate inputs
        assert res.P_hat[0, 1] > 1.0
        # Non-tradable domestic market clearing
        np.testing.assert_allclose(res.Y_prime[:, 1], res.X_prime[:, 1], atol=1e-8)


# =============================================================================
# ALLEN-ARKOLAKIS (2014) ADVERSARIAL TEST SUITE
# =============================================================================

class TestAllenArkolakisAdversarial:
    """Adversarial challenge tests for Allen-Arkolakis (2014) spatial GE engine."""

    def test_exact_uniqueness_boundary(self):
        """Stress-test exact uniqueness boundary alpha + beta = theta / (1 + theta)."""
        # Parameter set: theta = 1.0, beta = -0.20, alpha = 0.70
        # Condition: alpha + beta = 0.50 == theta / (1 + theta) = 0.50, and alpha <= 1 / theta = 1.0
        trade_costs = np.array([
            [1.0, 1.3],
            [1.3, 1.0],
        ])
        fund_A = np.array([1.0, 1.2])
        fund_a = np.array([1.0, 0.9])
        total_pop = 100.0

        model = AllenArkolakisModel(
            trade_costs=trade_costs,
            fundamental_productivity=fund_A,
            fundamental_amenity=fund_a,
            theta=1.0,
            alpha=0.70,
            beta=-0.20,
            total_population=total_pop,
        )
        assert model.is_unique

        res = model.solve_equilibrium()
        assert res.converged
        # Verify exact labor mass conservation
        assert res.labor_conservation_residual < 1e-12
        np.testing.assert_allclose(np.sum(res.population), total_pop, atol=1e-12)
        # Verify spatial utility equalization
        assert res.spatial_utility_variance < 1e-7

    def test_uniqueness_boundary_positive_beta(self):
        """Stress-test boundary with beta > 0 (theta = 2.0, alpha = 0.5, beta = 1/6)."""
        trade_costs = np.array([
            [1.0, 1.3],
            [1.3, 1.0],
        ])
        fund_A = np.array([1.0, 1.2])
        fund_a = np.array([1.0, 0.9])
        total_pop = 100.0

        with pytest.warns(UserWarning, match="beta is non-negative"):
            model = AllenArkolakisModel(
                trade_costs=trade_costs,
                fundamental_productivity=fund_A,
                fundamental_amenity=fund_a,
                theta=2.0,
                alpha=0.50,
                beta=2.0 / 3.0 - 0.50,  # 1/6
                total_population=total_pop,
            )
        assert model.is_unique

        res = model.solve_equilibrium()
        assert res.converged
        assert res.labor_conservation_residual < 1e-12
        assert res.spatial_utility_variance < 1e-7

    def test_asymmetric_trade_cost_matrix(self):
        """Stress-test strongly asymmetric trade cost matrix tau_{ij} != tau_{ji}."""
        trade_costs = np.array([
            [1.0, 1.2, 1.8, 2.5],
            [1.5, 1.0, 1.3, 2.0],
            [2.2, 1.6, 1.0, 1.4],
            [3.0, 2.4, 1.7, 1.0],
        ])
        fund_A = np.array([1.2, 1.0, 0.9, 1.1])
        fund_a = np.array([1.0, 1.3, 0.8, 1.0])
        total_pop = 250.0

        model = AllenArkolakisModel(
            trade_costs=trade_costs,
            fundamental_productivity=fund_A,
            fundamental_amenity=fund_a,
            theta=4.0,
            alpha=0.08,
            beta=-0.35,
            total_population=total_pop,
            region_names=["North", "East", "South", "West"],
        )

        res = model.solve_equilibrium(tol=1e-10, max_iter=3000)
        assert res.converged
        assert res.labor_conservation_residual < 1e-12
        np.testing.assert_allclose(np.sum(res.population), total_pop, atol=1e-12)
        assert res.spatial_utility_variance < 1e-7

        # Check goods market clearing: income == sales
        income = res.wages * res.population
        sales = np.sum(res.trade_shares * income[:, None], axis=0)
        max_mc_res = float(np.max(np.abs(income - sales)))
        assert max_mc_res < 1e-6, f"Goods market clearing error {max_mc_res} >= 1e-6"

    @pytest.mark.parametrize("shock_level", [0.70, 0.20, 0.01])
    def test_localized_climate_shocks(self, shock_level: float):
        """Stress-test mild (30%), severe (80%), and catastrophic (99%) localized productivity shocks."""
        coords = np.array([
            [40.7128, -74.0060],  # NYC
            [34.0522, -118.2437], # LA
            [41.8781, -87.6298],  # Chicago
            [29.7604, -95.3698],  # Houston
        ])
        names = ["NYC", "LA", "CHI", "HOU"]
        total_pop = 1000.0

        model = AllenArkolakisModel.from_coordinates(
            coords,
            region_names=names,
            theta=4.0,
            alpha=0.08,
            beta=-0.35,
            total_population=total_pop,
        )

        res = model.simulate_climate_shock(productivity_shocks={"HOU": shock_level})
        assert res.converged
        # Strict labor mass conservation
        assert res.labor_conservation_residual < 1e-12
        np.testing.assert_allclose(np.sum(res.population), total_pop, atol=1e-12)
        # Spatial utility equalization
        assert res.spatial_utility_variance < 1e-7
        # Populations strictly positive
        assert np.all(res.population > 0.0)
        # Shocked location loses population
        assert res.L_hat is not None and res.L_hat[3] < 1.0
        # Aggregate welfare falls
        assert res.welfare_pct is not None and res.welfare_pct < 0.0
