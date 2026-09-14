"""Comprehensive unit and integration tests for Allen & Arkolakis (2014) model.

Verifies:
1. Model initialization, pre-flight uniqueness check, and from_coordinates constructor.
2. Labor mass conservation (|\\sum L_i - \\bar{L}| < 10^{-12}).
3. Spatial price and utility equalization (std(u_i) / \\bar{u} < 10^{-7}).
4. Goods market clearing condition.
5. Symmetry preservation in symmetric geographies.
6. Counterfactual transport infrastructure shocks (welfare gains and spatial reallocation).
7. Counterfactual climate and environmental shocks.
8. Presentation exports (.summary, .to_markdown, .to_latex, .to_typst, .plot).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.spatial.allen_arkolakis import AllenArkolakisModel, AllenArkolakisResult


@pytest.fixture
def symmetric_2loc_model():
    """A perfectly symmetric 2-location geography."""
    N = 2
    trade_costs = np.array([
        [1.0, 1.25],
        [1.25, 1.0],
    ])
    fund_A = np.ones(N)
    fund_a = np.ones(N)
    return AllenArkolakisModel(
        trade_costs=trade_costs,
        fundamental_productivity=fund_A,
        fundamental_amenity=fund_a,
        theta=4.0,
        alpha=0.10,
        beta=-0.30,
        total_population=100.0,
        region_names=["LocA", "LocB"],
    )


@pytest.fixture
def us_4cities_model():
    """A 4-city US geography initialized via coordinates."""
    coords = np.array([
        [40.7128, -74.0060],  # NYC
        [34.0522, -118.2437], # LA
        [41.8781, -87.6298],  # Chicago
        [29.7604, -95.3698],  # Houston
    ])
    names = ["NYC", "LA", "CHI", "HOU"]
    return AllenArkolakisModel.from_coordinates(
        coords,
        region_names=names,
        theta=4.0,
        alpha=0.08,
        beta=-0.35,
        total_population=100.0,
    )


class TestAllenArkolakisValidationAndUniqueness:
    """Test parameter checks and uniqueness theorem verification."""

    def test_dimension_and_range_validation(self):
        # Non-square trade costs
        with pytest.raises(ValueError, match="trade_costs must be square"):
            AllenArkolakisModel(
                trade_costs=np.ones((3, 2)),
                fundamental_productivity=np.ones(3),
                fundamental_amenity=np.ones(3),
            )

        # Trade costs < 1
        bad_tau = np.array([[1.0, 0.8], [1.2, 1.0]])
        with pytest.raises(ValueError, match="trade costs must be >= 1.0"):
            AllenArkolakisModel(
                trade_costs=bad_tau,
                fundamental_productivity=np.ones(2),
                fundamental_amenity=np.ones(2),
            )

        # Non-positive fundamentals
        with pytest.raises(ValueError, match="Fundamental productivities must be strictly positive"):
            AllenArkolakisModel(
                trade_costs=np.ones((2, 2)),
                fundamental_productivity=np.array([1.0, -0.5]),
                fundamental_amenity=np.ones(2),
            )

    def test_uniqueness_preflight_check(self):
        """Pre-flight check correctly identifies uniqueness domain."""
        theta = 4.0
        # Bound: alpha + beta <= 4 / 5 = 0.8, and alpha <= 1 / 4 = 0.25
        # Valid case:
        model_valid = AllenArkolakisModel(
            trade_costs=np.ones((2, 2)),
            fundamental_productivity=np.ones(2),
            fundamental_amenity=np.ones(2),
            theta=theta,
            alpha=0.10,
            beta=-0.30,
        )
        assert model_valid.is_unique

        # Invalid case: high agglomeration violating uniqueness
        with pytest.warns(UserWarning, match="uniqueness condition"):
            model_invalid = AllenArkolakisModel(
                trade_costs=np.ones((2, 2)),
                fundamental_productivity=np.ones(2),
                fundamental_amenity=np.ones(2),
                theta=theta,
                alpha=0.90,
                beta=0.10,
            )
            assert not model_invalid.is_unique

    def test_from_coordinates_constructor(self):
        coords = np.array([[0.0, 0.0], [3.0, 4.0]])
        model = AllenArkolakisModel.from_coordinates(
            coords,
            distance_cost_param=0.1,
            distance_exponent=1.0,
            metric="euclidean",
            total_population=50.0,
        )
        assert model.N == 2
        # Euclidean distance is 5.0, so tau_01 = 1 + 0.1 * 5.0 = 1.5
        np.testing.assert_allclose(model.trade_costs[0, 1], 1.5)
        np.testing.assert_allclose(model.trade_costs[1, 0], 1.5)
        np.testing.assert_allclose(model.trade_costs[0, 0], 1.0)


class TestAllenArkolakisEquilibriumProperties:
    """Test core equilibrium theorems and conservation laws."""

    def test_labor_mass_conservation(self, us_4cities_model):
        """Total population must be conserved to within 10^{-12}."""
        res = us_4cities_model.solve_equilibrium()
        assert res.converged
        assert res.labor_conservation_residual < 1e-12
        np.testing.assert_allclose(
            np.sum(res.population), us_4cities_model.total_population, atol=1e-12
        )

    def test_spatial_utility_equalization(self, us_4cities_model):
        """Utility must be equalized across all populated locations (std / mean < 10^{-7})."""
        res = us_4cities_model.solve_equilibrium()
        assert res.converged
        assert res.spatial_utility_variance < 1e-7

    def test_symmetry_preservation(self, symmetric_2loc_model):
        """In a symmetric geography, wages, populations, and prices are identical."""
        res = symmetric_2loc_model.solve_equilibrium()
        assert res.converged
        np.testing.assert_allclose(res.population[0], res.population[1], rtol=1e-6)
        np.testing.assert_allclose(res.wages[0], res.wages[1], rtol=1e-6)
        np.testing.assert_allclose(res.price_index[0], res.price_index[1], rtol=1e-6)
        np.testing.assert_allclose(res.population[0], 50.0, rtol=1e-6)

    def test_goods_market_clearing(self, us_4cities_model):
        """Total income equals total sales for every location."""
        res = us_4cities_model.solve_equilibrium()
        income = res.wages * res.population
        # Destination i spends income[i], of which share pi[i, j] is on origin j
        # Total sales of origin j: sum_i pi[i, j] * income[i]
        sales = np.sum(res.trade_shares * income[:, None], axis=0)
        np.testing.assert_allclose(income, sales, rtol=1e-6)


class TestAllenArkolakisCounterfactuals:
    """Test policy experiments: transport infrastructure and climate shocks."""

    def test_infrastructure_shock(self, us_4cities_model):
        """Lowering trade costs between NYC and Chicago increases aggregate welfare."""
        res = us_4cities_model.simulate_infrastructure_shock(
            "NYC", "CHI", cost_reduction=0.30
        )
        assert res.converged
        # Aggregate welfare increases
        assert res.welfare_pct is not None and res.welfare_pct > 0.0
        # Connected cities attract population
        assert res.L_hat is not None
        assert res.L_hat[0] > 1.0  # NYC
        assert res.L_hat[2] > 1.0  # Chicago
        # Remote cities lose population share
        assert res.L_hat[1] < 1.0  # LA
        assert res.L_hat[3] < 1.0  # Houston

    def test_climate_shock(self, us_4cities_model):
        """A negative productivity shock to Houston induces out-migration and reduces welfare."""
        res = us_4cities_model.simulate_climate_shock(
            productivity_shocks={"HOU": 0.85}
        )
        assert res.converged
        # Aggregate welfare falls
        assert res.welfare_pct is not None and res.welfare_pct < 0.0
        # Houston loses population
        assert res.L_hat is not None
        assert res.L_hat[3] < 0.90
        # Other regions absorb the migrants
        assert res.L_hat[0] > 1.0
        assert res.L_hat[1] > 1.0
        assert res.L_hat[2] > 1.0

    def test_amenity_shock(self, us_4cities_model):
        """A negative amenity shock to LA induces out-migration."""
        res = us_4cities_model.simulate_climate_shock(
            amenity_shocks={"LA": 0.80}
        )
        assert res.converged
        assert res.L_hat is not None
        assert res.L_hat[1] < 0.90
        assert res.welfare_pct is not None and res.welfare_pct < 0.0


class TestAllenArkolakisPresentations:
    """Test summary, tables, and plots."""

    def test_summary_and_frame(self, us_4cities_model):
        res = us_4cities_model.simulate_infrastructure_shock("NYC", "CHI", cost_reduction=0.20)
        df = res.summary()
        assert isinstance(df, pd.DataFrame)
        assert list(df.index) == ["NYC", "LA", "CHI", "HOU"]
        expected_cols = {"wage", "population", "price_index", "real_wage", "cma", "fma", "wage_hat", "population_hat"}
        assert expected_cols.issubset(df.columns)
        pd.testing.assert_frame_equal(df, res.to_frame())

    def test_markup_formatters(self, us_4cities_model):
        res = us_4cities_model.solve_equilibrium()
        md = res.to_markdown()
        assert "NYC" in md
        assert "|" in md

        latex = res.to_latex()
        assert "\\begin{tabular}" in latex
        assert "NYC" in latex

        typst = res.to_typst()
        assert "#table(" in typst
        assert "NYC" in typst

    def test_plots(self, us_4cities_model):
        res = us_4cities_model.simulate_infrastructure_shock("NYC", "CHI", cost_reduction=0.20)
        fig1 = res.plot(kind="spatial")
        assert fig1 is not None
        fig2 = res.plot(kind="population")
        assert fig2 is not None
        fig3 = res.plot(kind="counterfactual")
        assert fig3 is not None
        plt.close("all")


class TestAdversarialRemediations:
    """Tests verifying fixes for CR-01, CR-02, and CR-08."""

    def test_beta_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="Amenity congestion elasticity beta cannot be zero"):
            AllenArkolakisModel(
                trade_costs=np.ones((2, 2)),
                fundamental_productivity=np.ones(2),
                fundamental_amenity=np.ones(2),
                theta=4.0,
                alpha=0.1,
                beta=0.0,
            )

    def test_trade_shares_properties(self, us_4cities_model):
        res = us_4cities_model.solve_equilibrium()
        np.testing.assert_array_equal(res.trade_shares_dest_origin, res.trade_shares)
        np.testing.assert_array_equal(res.trade_shares_origin_dest, res.trade_shares.T)

    def test_positive_climate_shock(self, us_4cities_model):
        # A positive productivity shock to Houston (+15%) using is_percentage_change=True
        res_pos = us_4cities_model.simulate_climate_shock(
            productivity_shocks={"HOU": 0.15},
            is_percentage_change=True,
        )
        assert res_pos.converged
        assert res_pos.welfare_pct is not None and res_pos.welfare_pct > 0.0
        assert res_pos.L_hat is not None
        assert res_pos.L_hat[3] > 1.0  # Houston attracts workers

        # Shock <= -1.0 raises ValueError
        with pytest.raises(ValueError, match="cannot be <= -1.0"):
            us_4cities_model.simulate_climate_shock(productivity_shocks={"HOU": -1.0})

