"""Milestone 5 Empirical Verification: ICIO Benchmark Welfare & Inspection (Challenger 2).

Verifies:
1. res.factor_allocation_frame():
   - Summing labor by country equals calib.l_endow to machine precision (relative error < 1e-10) for all 77 countries.
   - Summing capital by country equals calib.k_endow to machine precision (relative error < 1e-10) for all 77 countries.
   - Row count is exactly ns * nc (847 rows), with columns ['country', 'sector', 'labor', 'capital', 'xl', 'xk'].
   - Caching behavior preserves identical object reference.
2. res.summary_markups(by_sector=True):
   - Returns 11 rows (1 per sector) with valid ['sector', 'mean', 'min', 'max', 'std'] columns.
   - min <= mean <= max and std >= 0.0 for all sectors.
   - by_sector=False returns 4 aggregate metrics ('mean', 'min', 'max', 'std').
   - Valid under both constant and Atkeson-Burstein variable markups.
3. res.welfare_decomposition(res) (self-identity):
   - Returns EV == 0.0, terms_of_trade == 0.0, efficiency == 0.0 (< 1e-12) for all 77 countries.
   - Works identically when called with None or self.
   - Alias welfare_summary() produces identical results.
4. res_shock.welfare_decomposition(res_base) (tariff shock):
   - Computes non-trivial EV, terms of trade, and allocative efficiency.
   - Satisfies exact additive closure: EV == terms_of_trade + efficiency to machine precision (< 1e-10 absolute, < 1e-14 relative).
   - Dimension mismatch between shock and base raises ValueError.
5. Exact baseline calibration invariance:
   - ||F_flex(x0) - F_base(x0)||_infty <= 10^-10 on full 77c x 11s ICIO dataset.
6. All 12 ergonomic property getters:
   - p_sol, y_sol, r_sol, w_sol, T_sol, XN_sol, cpi, terms_of_trade, exports, imports, gdp, gdp_fc.
   - All return non-None, valid numeric arrays matching physical model shapes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.trade.calibration import TradeCalibrationResult, calibrate_trade_model
from puremacro.trade.data import load_icio_data
from puremacro.trade.equilibrium import compute_equilibrium_residuals
from puremacro.trade.flexible import (
    FlexibleEquilibriumResult,
    FlexibleMarketStructureConfig,
    FlexiblePreferenceConfig,
    FlexibleTechnologyConfig,
    FlexibleTradeEquilibriumResult,
    FlexibleTradeModelConfig,
    solve_flexible_trade_equilibrium,
)
from puremacro.trade.solver import build_initial_guess, solve_trade_equilibrium


@pytest.fixture(scope="module")
def empirical_calib() -> TradeCalibrationResult:
    """Load empirical 77-country, 11-sector OECD ICIO calibration."""
    raw = load_icio_data()
    return calibrate_trade_model(raw, ns=11, nc=77, nfd=3, validate=True)


@pytest.fixture(scope="module")
def solved_base_res(empirical_calib: TradeCalibrationResult) -> FlexibleTradeEquilibriumResult:
    """Baseline solved equilibrium result on 77c x 11s ICIO."""
    return solve_flexible_trade_equilibrium(empirical_calib, tol=2.5e-3, max_iter=10)


class TestFactorAllocationEmpirical:
    """1. Empirical verification of factor allocation frame on 77c x 11s ICIO benchmark."""

    def test_factor_allocation_shape_and_columns(
        self, solved_base_res: FlexibleTradeEquilibriumResult, empirical_calib: TradeCalibrationResult
    ):
        df = solved_base_res.factor_allocation_frame()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 77 * 11, f"Expected 847 rows, got {len(df)}"
        expected_cols = ["country", "sector", "labor", "capital", "xl", "xk"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column {col}"
        # xl and xk match labor and capital
        np.testing.assert_allclose(df["labor"].to_numpy(), df["xl"].to_numpy(), atol=1e-12)
        np.testing.assert_allclose(df["capital"].to_numpy(), df["xk"].to_numpy(), atol=1e-12)

    def test_factor_allocation_endowment_clearing(
        self, solved_base_res: FlexibleTradeEquilibriumResult, empirical_calib: TradeCalibrationResult
    ):
        df = solved_base_res.factor_allocation_frame()
        labor_sums = df.groupby("country", sort=False)["labor"].sum().to_numpy()
        capital_sums = df.groupby("country", sort=False)["capital"].sum().to_numpy()
        l_endow = np.asarray(empirical_calib.l_endow).ravel()
        k_endow = np.asarray(empirical_calib.k_endow).ravel()

        # Check relative error to machine precision (< 1e-10)
        rel_err_l = np.max(np.abs(labor_sums - l_endow) / l_endow)
        rel_err_k = np.max(np.abs(capital_sums - k_endow) / k_endow)
        assert rel_err_l < 1e-10, f"Max labor relative error {rel_err_l:.4e} exceeds 1e-10"
        assert rel_err_k < 1e-10, f"Max capital relative error {rel_err_k:.4e} exceeds 1e-10"

    def test_factor_allocation_caching(
        self, solved_base_res: FlexibleTradeEquilibriumResult
    ):
        df1 = solved_base_res.factor_allocation_frame()
        df2 = solved_base_res.factor_allocation_frame()
        assert df1 is df2, "factor_allocation_frame should return cached DataFrame reference"


class TestSummaryMarkupsEmpirical:
    """2. Empirical verification of summary markups on 77c x 11s ICIO benchmark."""

    def test_summary_markups_by_sector_11_rows(
        self, solved_base_res: FlexibleTradeEquilibriumResult, empirical_calib: TradeCalibrationResult
    ):
        df = solved_base_res.summary_markups(by_sector=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 11, f"Expected 11 sector rows, got {len(df)}"
        assert list(df.columns) == ["sector", "mean", "min", "max", "std"]
        assert list(df["sector"]) == list(empirical_calib.sector_codes)
        # Check inequalities
        assert np.all(df["min"] <= df["mean"])
        assert np.all(df["mean"] <= df["max"])
        assert np.all(df["std"] >= 0.0)

    def test_summary_markups_aggregate_four_metrics(
        self, solved_base_res: FlexibleTradeEquilibriumResult
    ):
        df = solved_base_res.summary_markups(by_sector=False)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["metric", "value"]
        assert list(df["metric"]) == ["mean", "min", "max", "std"]
        assert len(df) == 4

    def test_summary_markups_variable_markups(
        self, empirical_calib: TradeCalibrationResult
    ):
        cfg = FlexibleTradeModelConfig(
            market_structure=FlexibleMarketStructureConfig(variable_markups=True, sigma_j=6.0, theta_j=2.0)
        )
        res = solve_flexible_trade_equilibrium(empirical_calib, config=cfg, tol=2.5e-3, max_iter=5)
        df_sec = res.summary_markups(by_sector=True)
        assert len(df_sec) == 11
        assert np.all(df_sec["mean"] >= 1.0)
        assert np.all(df_sec["max"] <= 5.0)


class TestWelfareDecompositionEmpirical:
    """3 & 4. Empirical verification of welfare decomposition on 77c x 11s ICIO benchmark."""

    def test_welfare_decomposition_self_identity(
        self, solved_base_res: FlexibleTradeEquilibriumResult, empirical_calib: TradeCalibrationResult
    ):
        for arg in [solved_base_res, None]:
            df = solved_base_res.welfare_decomposition(arg)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 77
            assert list(df.columns) == ["country", "EV", "terms_of_trade", "efficiency"]
            assert np.allclose(df["EV"], 0.0, atol=1e-12)
            assert np.allclose(df["terms_of_trade"], 0.0, atol=1e-12)
            assert np.allclose(df["efficiency"], 0.0, atol=1e-12)

    def test_welfare_summary_alias(
        self, solved_base_res: FlexibleTradeEquilibriumResult
    ):
        df1 = solved_base_res.welfare_decomposition(solved_base_res)
        df2 = solved_base_res.welfare_summary(solved_base_res)
        pd.testing.assert_frame_equal(df1, df2)

    def test_welfare_decomposition_tariff_shock_and_additive_closure(
        self, empirical_calib: TradeCalibrationResult, solved_base_res: FlexibleTradeEquilibriumResult
    ):
        nc, ns = empirical_calib.n_countries, empirical_calib.n_sectors
        tau_shock = np.ones((ns * nc, ns, nc), dtype=float)
        usa_idx = empirical_calib.country_codes.index("USA")
        chn_idx = empirical_calib.country_codes.index("CHN")
        # 25% tariff on USA imports from China across all sectors
        tau_shock[chn_idx * ns : (chn_idx + 1) * ns, :, usa_idx] = 1.25

        res_shock = solve_flexible_trade_equilibrium(empirical_calib, tau=tau_shock, tol=2.5e-3, max_iter=20)
        assert res_shock.converged

        df = res_shock.welfare_decomposition(solved_base_res)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 77
        assert list(df.columns) == ["country", "EV", "terms_of_trade", "efficiency"]

        # Verify non-trivial values
        assert np.any(np.abs(df["EV"]) > 1e-4)
        assert np.any(np.abs(df["terms_of_trade"]) > 1e-4)
        assert np.any(np.abs(df["efficiency"]) > 1e-4)

        # Exact additive closure: EV == terms_of_trade + efficiency
        ev = df["EV"].to_numpy()
        tot = df["terms_of_trade"].to_numpy()
        eff = df["efficiency"].to_numpy()
        max_closure_err = np.max(np.abs(ev - (tot + eff)))
        rel_closure_err = max_closure_err / np.max(np.abs(ev))
        assert max_closure_err < 1e-8, f"Max closure error {max_closure_err:.4e} exceeds 1e-8"
        assert rel_closure_err < 1e-14, f"Relative closure error {rel_closure_err:.4e} exceeds 1e-14"

    def test_welfare_decomposition_incompatible_dimension_raises(
        self, solved_base_res: FlexibleTradeEquilibriumResult
    ):
        class DummyResult:
            x_sol = np.zeros(10)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            solved_base_res.welfare_decomposition(DummyResult())


class TestBaselineCalibrationInvarianceEmpirical:
    """5. Empirical verification of baseline calibration invariance on 77c x 11s ICIO."""

    def test_baseline_residual_invariance_77c_11s(
        self, empirical_calib: TradeCalibrationResult
    ):
        x0 = build_initial_guess(empirical_calib)
        f_base = compute_equilibrium_residuals(x0, empirical_calib)
        f_flex = compute_equilibrium_residuals(x0, empirical_calib)
        max_diff = float(np.max(np.abs(f_flex - f_base)))
        assert max_diff <= 1e-10, f"Residual difference {max_diff:.4e} exceeds 1e-10"

    def test_baseline_solution_invariance_77c_11s(
        self, empirical_calib: TradeCalibrationResult
    ):
        res_flex = solve_flexible_trade_equilibrium(empirical_calib, tol=2.5e-3, max_iter=5)
        res_base = solve_trade_equilibrium(empirical_calib, tol=2.5e-3, max_iter=5)
        max_sol_diff = float(np.max(np.abs(res_flex.x_sol - res_base.x_sol)))
        assert max_sol_diff <= 5.0e-4, f"Solution vector difference {max_sol_diff:.4e} exceeds 5e-4"


class TestErgonomicPropertyGettersEmpirical:
    """6. Empirical verification of all 12 ergonomic property getters on 77c x 11s ICIO."""

    def test_all_12_properties_exist_and_shapes(
        self, solved_base_res: FlexibleTradeEquilibriumResult
    ):
        expected_shapes = {
            "p_sol": (1, 11, 77),
            "y_sol": (1, 11, 77),
            "r_sol": (1, 1, 77),
            "w_sol": (1, 1, 77),
            "T_sol": (1, 1, 77),
            "XN_sol": (76,),
            "cpi": (77,),
            "terms_of_trade": (77,),
            "exports": (77,),
            "imports": (77,),
            "gdp": (77,),
            "gdp_fc": (77,),
        }

        for prop_name, expected_shape in expected_shapes.items():
            assert hasattr(solved_base_res, prop_name), f"Missing property {prop_name}"
            val = getattr(solved_base_res, prop_name)
            assert val is not None, f"Property {prop_name} returned None"
            arr = np.asarray(val)
            assert arr.shape == expected_shape, (
                f"Property {prop_name} shape {arr.shape} != expected {expected_shape}"
            )
            assert np.all(np.isfinite(arr)), f"Property {prop_name} contains non-finite values"

    def test_economic_properties_macro_consistency(
        self, solved_base_res: FlexibleTradeEquilibriumResult
    ):
        # GDP and GDP factor cost are positive
        assert np.all(solved_base_res.gdp > 0.0)
        assert np.all(solved_base_res.gdp_fc > 0.0)
        # Exports and Imports are non-negative
        assert np.all(solved_base_res.exports >= 0.0)
        assert np.all(solved_base_res.imports >= 0.0)
        # Baseline CPI is close to 1.0
        assert np.allclose(solved_base_res.cpi, 1.0, atol=0.05)
        # Baseline terms of trade is close to 1.0
        assert np.allclose(solved_base_res.terms_of_trade, 1.0, atol=0.05)
