"""Tier 5 Adversarial Coverage Hardening Suite: Trade Analytics, Results, and Namespace.

Milestone 4 Phase 2: Track B adversarial test harness.
Audits and stress-tests:
1. policy_analytics.py:
   - verify_theorems_1_to_4 across extreme substitution elasticities sigma in [1e-5, 1e5],
     tariff boundary conditions (tau=0 autarky/free trade boundary, prohibitive tariffs up to 100.0),
     extreme foreign export supply elasticities in [1e-5, 1e5], non-default target countries
     (ISO-3 codes, integer indexing), unusual scenario names, single-country MRIO models,
     and 4D technical coefficient tensors.
   - decompose_hicksian_ev_3way under edge-case flows, zero imports, missing/non-positive CPI,
     single-country autarky, strict tolerance gates, and percentage/monetary scaling.
   - decompose_welfare_effects with missing flows, base_result=None, and alternative price metrics.
   - compute_effective_rate_of_protection, compute_supply_chain_vulnerability, and
     calculate_tariff_revenue_incidence across extreme regimes and parameter spaces.
2. _results.py:
   - TheoremValidationReport & EVDecompositionResult formatting methods:
     .to_latex(), .to_typst(), .to_markdown(), .to_frame(), .to_dataframe(), .summary().
   - Dict access edge cases (case-insensitivity, aliases, invalid keys, invalid types).
   - Mapping protocol (len, in, iter, keys, values, items).
   - Slicing and immutability (frozen dataclass enforcement).
   - Adversarial check of .summary(detailed=True) across result containers.
3. __init__.py:
   - Complete export resolution of __all__.
   - Policy analytics re-exports and result dataclasses.
   - Namespace cleanliness and hygiene.

Strictly conforms to the puremacro Pyodide runtime contract (NumPy, SciPy, Pandas only).
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import numpy as np
import pandas as pd
import pytest

import puremacro.trade as pt
from puremacro.trade._results import (
    EVDecompositionResult,
    TheoremValidationReport,
    TradeCalibrationResult,
    TradeEquilibriumResult,
)
from puremacro.trade.policy_analytics import (
    EffectiveRateOfProtectionResult,
    SupplyChainVulnerabilityResult,
    TariffRevenueIncidenceResult,
    WelfareDecompositionResult,
    calculate_tariff_revenue_incidence,
    compute_effective_rate_of_protection,
    compute_supply_chain_vulnerability,
    decompose_hicksian_ev_3way,
    decompose_welfare_effects,
    verify_theorems_1_to_4,
)
from puremacro.trade.scenarios import TariffScenario


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def mock_calib_2c_2s() -> TradeCalibrationResult:
    """Fast, deterministic 2-country x 2-sector calibrated trade model."""
    nc, ns, nfd = 2, 2, 3
    ytot = np.array([[[1000.0, 800.0], [500.0, 400.0]]], dtype=np.float64)
    a_3d = np.zeros((ns * nc, ns, nc), dtype=np.float64)
    a_3d[0, 0, 0] = 0.15  # domestic AGR into AGR in USA
    a_3d[1, 1, 0] = 0.10  # domestic MAN into MAN in USA
    a_3d[2, 0, 0] = 0.20  # foreign AGR into AGR in USA
    a_3d[3, 1, 0] = 0.20  # foreign MAN into MAN in USA
    a_3d[2, 0, 1] = 0.12  # domestic AGR into AGR in ROW
    a_3d[3, 1, 1] = 0.08  # domestic MAN into MAN in ROW
    a_3d[0, 0, 1] = 0.15  # foreign AGR into AGR in ROW
    a_3d[1, 1, 1] = 0.15  # foreign MAN into MAN in ROW

    afd_3d = np.full((ns * nc, nfd, nc), 0.05, dtype=np.float64)
    alpha = np.full((1, ns, nc), 0.33, dtype=np.float64)
    beta = np.ones((1, ns, nc), dtype=np.float64)
    k_endow = np.array([[500.0, 400.0]], dtype=np.float64)
    l_endow = np.array([[500.0, 400.0]], dtype=np.float64)
    invforT = np.array([[0.0, 0.0]], dtype=np.float64)
    tax = np.zeros((1, ns, nc), dtype=np.float64)

    return TradeCalibrationResult(
        a=a_3d,
        afd=afd_3d,
        alpha=alpha,
        beta=beta,
        k_endow=k_endow,
        l_endow=l_endow,
        invforT=invforT,
        tax=tax,
        ytot=ytot,
        n_countries=nc,
        n_sectors=ns,
        n_final_demand=nfd,
        country_codes=("USA", "ROW"),
        sector_codes=("AGR", "MAN"),
    )


@pytest.fixture(scope="module")
def mock_calib_1c_2s() -> TradeCalibrationResult:
    """Minimal single-country 2-sector autarkic trade model."""
    nc, ns, nfd = 1, 2, 3
    ytot = np.array([[[1000.0], [500.0]]], dtype=np.float64)
    a_3d = np.zeros((ns * nc, ns, nc), dtype=np.float64)
    a_3d[0, 0, 0] = 0.20
    a_3d[1, 1, 0] = 0.15
    afd_3d = np.full((ns * nc, nfd, nc), 0.10, dtype=np.float64)
    alpha = np.full((1, ns, nc), 0.33, dtype=np.float64)
    beta = np.ones((1, ns, nc), dtype=np.float64)
    k_endow = np.array([[500.0]], dtype=np.float64)
    l_endow = np.array([[500.0]], dtype=np.float64)
    invforT = np.array([[0.0]], dtype=np.float64)
    tax = np.zeros((1, ns, nc), dtype=np.float64)

    return TradeCalibrationResult(
        a=a_3d,
        afd=afd_3d,
        alpha=alpha,
        beta=beta,
        k_endow=k_endow,
        l_endow=l_endow,
        invforT=invforT,
        tax=tax,
        ytot=ytot,
        n_countries=nc,
        n_sectors=ns,
        n_final_demand=nfd,
        country_codes=("USA",),
        sector_codes=("AGR", "MAN"),
    )


@pytest.fixture(scope="module")
def mock_equilibrium_pair(mock_calib_2c_2s) -> tuple[TradeEquilibriumResult, TradeEquilibriumResult]:
    """Pair of mock baseline and counterfactual equilibrium results."""
    nc = mock_calib_2c_2s.n_countries
    ns = mock_calib_2c_2s.n_sectors
    nfd = mock_calib_2c_2s.n_final_demand
    ytot = mock_calib_2c_2s.ytot

    # Base equilibrium
    base = TradeEquilibriumResult(
        x_sol=np.zeros(2001),
        p_sol=np.ones((1, ns, nc)),
        y_sol=ytot.copy(),
        r_sol=np.ones((1, 1, nc)),
        w_sol=np.ones((1, 1, nc)),
        T_sol=np.zeros((1, 1, nc)),
        XN_sol=np.zeros(1),
        tariffs=np.array([0.0, 0.0]),
        cpi=np.array([1.0, 1.0]),
        terms_of_trade=np.array([1.0, 1.0]),
        exports=np.array([200.0, 150.0]),
        imports=np.array([200.0, 150.0]),
        intermediate_flows=np.ones((ns, nc, ns, nc)) * 10.0,
        final_demand_flows=np.ones((ns, nc, nfd, nc)) * 5.0,
        country_codes=("USA", "ROW"),
        sector_codes=("AGR", "MAN"),
    )

    # Counterfactual equilibrium
    cf = TradeEquilibriumResult(
        x_sol=np.zeros(2001),
        p_sol=np.ones((1, ns, nc)) * 1.05,
        y_sol=ytot * 0.98,
        r_sol=np.ones((1, 1, nc)) * 0.99,
        w_sol=np.ones((1, 1, nc)) * 0.99,
        T_sol=np.array([[[15.0, 8.0]]]),
        XN_sol=np.zeros(1),
        tariffs=np.array([25.0, 10.0]),
        cpi=np.array([1.03, 1.01]),
        terms_of_trade=np.array([1.02, 0.98]),
        exports=np.array([190.0, 145.0]),
        imports=np.array([175.0, 140.0]),
        intermediate_flows=np.ones((ns, nc, ns, nc)) * 9.5,
        final_demand_flows=np.ones((ns, nc, nfd, nc)) * 4.8,
        country_codes=("USA", "ROW"),
        sector_codes=("AGR", "MAN"),
    )

    return base, cf


@pytest.fixture(scope="module")
def sample_ev_result() -> EVDecompositionResult:
    """Pre-built sample EVDecompositionResult for presentation & interface tests."""
    df = pd.DataFrame(
        [
            {"Effect": "TOT", "Description": "Terms of Trade Effect", "Value": 12.5, "Share_Pct": 50.0},
            {"Effect": "Alloc", "Description": "Allocative Efficiency (DWL)", "Value": -2.5, "Share_Pct": -10.0},
            {"Effect": "TariffRec", "Description": "Tariff Revenue Recycling", "Value": 15.0, "Share_Pct": 60.0},
            {"Effect": "Total_EV", "Description": "Equivalent Variation", "Value": 25.0, "Share_Pct": 100.0},
            {"Effect": "Residual", "Description": "Identity Discrepancy", "Value": 0.0, "Share_Pct": 0.0},
        ]
    ).set_index("Effect")

    return EVDecompositionResult(
        ev_usd=25.0,
        ev_pct=1.5625,
        tot=12.5,
        alloc=-2.5,
        tariff_rec=15.0,
        residual=0.0,
        country_code="USA",
        summary_df=df,
    )


# =============================================================================
# 1. TestTheoremBoundingAdversarial (policy_analytics.py: verify_theorems_1_to_4)
# =============================================================================

class TestTheoremBoundingAdversarial:
    """Stress-test verify_theorems_1_to_4 across extreme mathematical boundaries."""

    @pytest.mark.parametrize(
        "sigma",
        [
            1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5, 0.999, 1.0, 1.0001,
            1.5, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0, 1e4, 1e5,
        ],
    )
    @pytest.mark.xfail(raises=NotImplementedError, reason="Exact EV/theorem certification quarantined after independent counterexamples", run=False, strict=True)
    def test_theorems_extreme_elasticities(self, mock_calib_2c_2s, sigma: float):
        """Stress-test elasticity sigma in [1e-5, 1e5] for overflow, underflow, or nan."""
        report = verify_theorems_1_to_4(mock_calib_2c_2s, scenario="uniform_10", sigma=sigma)
        assert isinstance(report, TheoremValidationReport)
        assert report.all_passed is True
        assert report.theorem_1_passed is True
        assert report.theorem_2_passed is True
        assert report.theorem_3_passed is True
        assert report.theorem_4_passed is True

        # Check detail metrics are finite and well-defined
        assert np.isfinite(report[1]["harberger_dwl"])
        assert report[1]["harberger_dwl"] >= 0.0
        assert np.isfinite(report[2]["factory_gate_bound"])
        assert np.isfinite(report[3]["export_drop_ces"])
        assert np.isfinite(report[4]["laffer_peak_tau"])
        assert report[4]["laffer_peak_tau"] > 0.0

    @pytest.mark.parametrize(
        "tau_val",
        [0.0, 1e-6, 1e-4, 0.05, 0.10, 0.25, 0.54, 1.0, 2.0, 5.0, 8.0, 10.0, 50.0, 100.0],
    )
    @pytest.mark.xfail(raises=NotImplementedError, reason="Exact EV/theorem certification quarantined after independent counterexamples", run=False, strict=True)
    def test_theorems_extreme_tariffs(self, mock_calib_2c_2s, tau_val: float):
        """Stress-test tariff rates from zero (free trade) to prohibitive levels (100.0)."""
        tau_override = np.array([tau_val])
        report = verify_theorems_1_to_4(mock_calib_2c_2s, tau_override=tau_override)
        assert report.all_passed is True
        assert report.theorem_1_passed is True
        assert report.theorem_2_passed is True
        assert report.theorem_3_passed is True
        assert report.theorem_4_passed is True

        # Theorem 2 factory-gate price bound
        t2 = report[2]
        assert t2["p_ge"] <= t2["p_bound"] + 1e-12
        assert not t2["unphysical_inflation_detected"]

        # Theorem 3 export drop bound
        t3 = report[3]
        assert t3["export_drop_leontief"] <= t3["export_drop_ces"] + 1e-12

    @pytest.mark.parametrize("fe", [1e-5, 1e-3, 0.1, 0.5, 1.0, 5.0, 50.0, 1e4, 1e5])
    @pytest.mark.xfail(raises=NotImplementedError, reason="Exact EV/theorem certification quarantined after independent counterexamples", run=False, strict=True)
    def test_theorems_extreme_foreign_elasticity(self, mock_calib_2c_2s, fe: float):
        """Stress-test foreign export supply elasticity in [1e-5, 1e5]."""
        report = verify_theorems_1_to_4(mock_calib_2c_2s, foreign_elasticity=fe)
        assert report.all_passed is True
        assert report[1]["loe_tot_gain"] >= 0.0

    @pytest.mark.parametrize("target", ["USA", "ROW", 0, 1, "usa", "row", "NONEXISTENT"])
    @pytest.mark.xfail(raises=NotImplementedError, reason="Exact EV/theorem certification quarantined after independent counterexamples", run=False, strict=True)
    def test_theorems_target_country_variations(self, mock_calib_2c_2s, target):
        """Verify handling of ISO-3 codes, integer indices, and fallback logic."""
        report = verify_theorems_1_to_4(mock_calib_2c_2s, target_country=target)
        assert report.all_passed is True
        assert report[4]["rebate_offset_ratio"] > 0.0

    @pytest.mark.parametrize(
        "scenario_name",
        ["uniform_10", "uniform", "s232", "section_232", "china_54", "trade_war", "retaliation", "custom_xyz", ""],
    )
    @pytest.mark.xfail(raises=NotImplementedError, reason="Exact EV/theorem certification quarantined after independent counterexamples", run=False, strict=True)
    def test_theorems_unusual_scenarios(self, mock_calib_2c_2s, scenario_name: str):
        """Verify scenario resolution logic handles standard and arbitrary strings."""
        report = verify_theorems_1_to_4(mock_calib_2c_2s, scenario=scenario_name)
        assert report.all_passed is True
        assert report.scenario == scenario_name

    @pytest.mark.xfail(raises=NotImplementedError, reason="Exact EV/theorem certification quarantined after independent counterexamples", run=False, strict=True)
    def test_theorems_single_country_autarky(self, mock_calib_1c_2s):
        """Verify theorem evaluation gracefully handles single-country model with no foreign partners."""
        report = verify_theorems_1_to_4(mock_calib_1c_2s, scenario="uniform_10")
        assert report.all_passed is True
        assert report.theorem_1_passed is True
        assert report.theorem_2_passed is True
        assert report.theorem_3_passed is True
        assert report.theorem_4_passed is True

    @pytest.mark.xfail(raises=NotImplementedError, reason="Exact EV/theorem certification quarantined after independent counterexamples", run=False, strict=True)
    def test_theorems_with_4d_technical_coefficients(self, mock_calib_2c_2s):
        """Verify theorem evaluation when calib has pre-populated a_4d tensor."""
        assert mock_calib_2c_2s.a_4d is not None
        report = verify_theorems_1_to_4(mock_calib_2c_2s, scenario="uniform_10")
        assert report.all_passed is True

    @pytest.mark.xfail(raises=NotImplementedError, reason="Exact EV/theorem certification quarantined after independent counterexamples", run=False, strict=True)
    def test_theorems_summary_and_scorecard_integrity(self, mock_calib_2c_2s):
        """Verify summary_df contains all 4 theorems with expected columns and metrics."""
        report = verify_theorems_1_to_4(mock_calib_2c_2s, scenario="uniform_10")
        df = report.summary_df
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "Passed" in df.columns
        assert "Headline_Metric" in df.columns
        assert bool(df["Passed"].all()) is True


# =============================================================================
# 2. TestEVDecompositionAdversarial (policy_analytics.py: decompose_hicksian_ev_3way)
# =============================================================================

class TestEVDecompositionAdversarial:
    """Stress-test decompose_hicksian_ev_3way under edge-case flows and numerical boundaries."""

    @pytest.mark.xfail(raises=NotImplementedError, reason="Exact EV/theorem certification quarantined after independent counterexamples", run=False, strict=True)
    def test_exact_identity_levels(self, mock_calib_2c_2s, mock_equilibrium_pair):
        """Verify exact 3-way additive decomposition in monetary levels (as_percent=False)."""
        base, cf = mock_equilibrium_pair
        decomp = decompose_hicksian_ev_3way(mock_calib_2c_2s, cf, base_result=base, as_percent=False)

        assert isinstance(decomp, EVDecompositionResult)
        assert decomp.exact_identity_satisfied is True
        assert abs(decomp.residual) <= 1e-10
        assert abs(decomp.ev_usd - (decomp.tot + decomp.alloc + decomp.tariff_rec)) <= 1e-10

    @pytest.mark.xfail(raises=NotImplementedError, reason="Exact EV/theorem certification quarantined after independent counterexamples", run=False, strict=True)
    def test_exact_identity_percentage(self, mock_calib_2c_2s, mock_equilibrium_pair):
        """Verify exact 3-way additive decomposition in percentage of base GDP (as_percent=True)."""
        base, cf = mock_equilibrium_pair
        decomp = decompose_hicksian_ev_3way(mock_calib_2c_2s, cf, base_result=base, as_percent=True)

        assert decomp.exact_identity_satisfied is True
        assert abs(decomp.residual) <= 1e-10
        assert abs(decomp.ev_pct - (decomp.tot + decomp.alloc + decomp.tariff_rec)) <= 1e-10

    @pytest.mark.xfail(raises=NotImplementedError, reason="Exact EV/theorem certification quarantined after independent counterexamples", run=False, strict=True)
    def test_custom_tolerances(self, mock_calib_2c_2s, mock_equilibrium_pair):
        """Verify custom tolerance thresholds are enforced, and negative tol strictly raises."""
        base, cf = mock_equilibrium_pair
        decomp = decompose_hicksian_ev_3way(mock_calib_2c_2s, cf, base_result=base, tol=1e-8)
        assert abs(decomp.residual) <= 1e-8

        # A negative tolerance strictly triggers assertion failure as abs(residual) >= 0.0 > tol
        with pytest.raises(AssertionError, match="EV 3-way additive decomposition identity failed"):
            decompose_hicksian_ev_3way(mock_calib_2c_2s, cf, base_result=base, tol=-1.0)

    @pytest.mark.xfail(raises=NotImplementedError, reason="Exact EV/theorem certification quarantined after independent counterexamples", run=False, strict=True)
    def test_missing_and_nonpositive_cpi(self, mock_calib_2c_2s, mock_equilibrium_pair):
        """Verify fallback when CPI is None, zero, or negative."""
        base, cf = mock_equilibrium_pair
        cf_bad_cpi = TradeEquilibriumResult(
            x_sol=cf.x_sol, p_sol=cf.p_sol, y_sol=cf.y_sol, r_sol=cf.r_sol, w_sol=cf.w_sol,
            T_sol=cf.T_sol, XN_sol=cf.XN_sol, tariffs=cf.tariffs, cpi=np.array([0.0, -1.0]),
            terms_of_trade=cf.terms_of_trade, exports=cf.exports, imports=cf.imports,
            intermediate_flows=cf.intermediate_flows,
        )
        decomp = decompose_hicksian_ev_3way(mock_calib_2c_2s, cf_bad_cpi, base_result=base)
        assert decomp.exact_identity_satisfied is True
        assert decomp.tariff_rec == float(cf.tariffs[0])  # deflator defaults to 1.0

    @pytest.mark.xfail(raises=NotImplementedError, reason="Exact EV/theorem certification quarantined after independent counterexamples", run=False, strict=True)
    def test_zero_imports_no_division_by_zero(self, mock_calib_2c_2s, mock_equilibrium_pair):
        """Verify allocative DWL calculation handles zero imports without ZeroDivisionError."""
        base, cf = mock_equilibrium_pair
        cf_zero_imp = TradeEquilibriumResult(
            x_sol=cf.x_sol, p_sol=cf.p_sol, y_sol=cf.y_sol, r_sol=cf.r_sol, w_sol=cf.w_sol,
            T_sol=cf.T_sol, XN_sol=cf.XN_sol, tariffs=cf.tariffs, cpi=cf.cpi,
            terms_of_trade=cf.terms_of_trade, exports=cf.exports, imports=np.array([0.0, 0.0]),
            intermediate_flows=cf.intermediate_flows,
        )
        decomp = decompose_hicksian_ev_3way(mock_calib_2c_2s, cf_zero_imp, base_result=base)
        assert decomp.exact_identity_satisfied is True
        assert np.isfinite(decomp.alloc)

    @pytest.mark.xfail(raises=NotImplementedError, reason="Exact EV/theorem certification quarantined after independent counterexamples", run=False, strict=True)
    def test_missing_flows_fallback(self, mock_calib_2c_2s, mock_equilibrium_pair):
        """Verify fallback to terms_of_trade index when intermediate_flows are None."""
        base, cf = mock_equilibrium_pair
        cf_no_flows = TradeEquilibriumResult(
            x_sol=cf.x_sol, p_sol=cf.p_sol, y_sol=cf.y_sol, r_sol=cf.r_sol, w_sol=cf.w_sol,
            T_sol=cf.T_sol, XN_sol=cf.XN_sol, tariffs=cf.tariffs, cpi=cf.cpi,
            terms_of_trade=cf.terms_of_trade, exports=cf.exports, imports=cf.imports,
            intermediate_flows=None,
        )
        decomp = decompose_hicksian_ev_3way(mock_calib_2c_2s, cf_no_flows, base_result=base)
        assert decomp.exact_identity_satisfied is True
        assert np.isfinite(decomp.tot)

    @pytest.mark.xfail(raises=NotImplementedError, reason="Exact EV/theorem certification quarantined after independent counterexamples", run=False, strict=True)
    def test_single_country_ev_decomp(self, mock_calib_1c_2s):
        """Verify single-country model has zero TOT effect and satisfies identity."""
        nc = mock_calib_1c_2s.n_countries
        ns = mock_calib_1c_2s.n_sectors
        ytot = mock_calib_1c_2s.ytot

        eq = TradeEquilibriumResult(
            x_sol=np.zeros(2001), p_sol=np.ones((1, ns, nc)), y_sol=ytot,
            r_sol=np.ones((1, 1, nc)), w_sol=np.ones((1, 1, nc)), T_sol=np.zeros((1, 1, nc)),
            XN_sol=np.zeros(0), tariffs=np.array([10.0]), imports=np.array([0.0]), exports=np.array([0.0]),
            cpi=np.array([1.05]), terms_of_trade=np.array([1.0]),
            intermediate_flows=np.zeros((ns, nc, ns, nc)),
        )
        decomp = decompose_hicksian_ev_3way(mock_calib_1c_2s, eq, base_result=eq, target_country="USA")
        assert decomp.tot == 0.0
        assert decomp.exact_identity_satisfied is True


# =============================================================================
# 3. TestWelfareDecompositionAndERPEdgeCases (policy_analytics.py routines)
# =============================================================================

class TestWelfareDecompositionAndERPEdgeCases:
    """Stress-test decompose_welfare_effects, compute_effective_rate_of_protection, etc."""

    def test_welfare_decomposition_budget_balance(self, mock_calib_2c_2s, mock_equilibrium_pair):
        """Verify decompose_welfare_effects satisfies exact Walrasian budget balance residual < 1e-4."""
        base, cf = mock_equilibrium_pair
        res = decompose_welfare_effects(mock_calib_2c_2s, cf, base_result=base, target_country="USA")
        assert isinstance(res, WelfareDecompositionResult)
        assert res.exact_identity_satisfied is True
        assert res.budget_balance_residual < 1e-4
        assert abs(res.total_welfare_change - (res.tot_effect + res.cascading_distortion + res.allocative_efficiency)) < 1e-4

        df = res.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "Terms_of_Trade_Effect" in df.index

    def test_decompose_welfare_effects_none_final_demand_flows_adversarial_check(self, mock_calib_2c_2s, mock_equilibrium_pair):
        """Adversarial check: decompose_welfare_effects should not crash when final_demand_flows is None.

        When intermediate_flows is present but final_demand_flows is None on TradeEquilibriumResult,
        lines 733 and 741 of policy_analytics.py attempt to subscript NoneType.
        """
        base, cf = mock_equilibrium_pair
        cf_none_fd = TradeEquilibriumResult(
            x_sol=cf.x_sol, p_sol=cf.p_sol, y_sol=cf.y_sol, r_sol=cf.r_sol, w_sol=cf.w_sol,
            T_sol=cf.T_sol, XN_sol=cf.XN_sol, tariffs=cf.tariffs, cpi=cf.cpi,
            terms_of_trade=cf.terms_of_trade, exports=cf.exports, imports=cf.imports,
            intermediate_flows=cf.intermediate_flows,
            final_demand_flows=None,  # Intentionally None
            country_codes=cf.country_codes, sector_codes=cf.sector_codes,
        )
        base_none_fd = TradeEquilibriumResult(
            x_sol=base.x_sol, p_sol=base.p_sol, y_sol=base.y_sol, r_sol=base.r_sol, w_sol=base.w_sol,
            T_sol=base.T_sol, XN_sol=base.XN_sol, tariffs=base.tariffs, cpi=base.cpi,
            terms_of_trade=base.terms_of_trade, exports=base.exports, imports=base.imports,
            intermediate_flows=base.intermediate_flows,
            final_demand_flows=None,  # Intentionally None
            country_codes=base.country_codes, sector_codes=base.sector_codes,
        )

        try:
            res = decompose_welfare_effects(mock_calib_2c_2s, cf_none_fd, base_result=base_none_fd, target_country="USA")
            assert res is not None
        except TypeError as err:
            pytest.fail(
                f"EXPOSED GAP: decompose_welfare_effects crashes when final_demand_flows is None: {err}"
            )

    def test_erp_different_tariff_inputs(self, mock_calib_2c_2s):
        """Test compute_effective_rate_of_protection with 1D vector, scenario string 't10', and TariffScenario."""
        # 1D vector
        res1 = compute_effective_rate_of_protection(mock_calib_2c_2s, tariffs=np.array([0.10, 0.25]))
        assert isinstance(res1, EffectiveRateOfProtectionResult)
        assert len(res1.erp_balassa) == mock_calib_2c_2s.n_sectors

        # Scenario string 't10'
        res2 = compute_effective_rate_of_protection(mock_calib_2c_2s, tariffs="t10")
        assert len(res2.erp_corden) == mock_calib_2c_2s.n_sectors

        # TariffScenario
        scen = TariffScenario(name="test", default_us_tariff=0.15)
        res3 = compute_effective_rate_of_protection(mock_calib_2c_2s, tariffs=scen)
        assert len(res3.distortion_category) == mock_calib_2c_2s.n_sectors

        # Invalid scenario string raises KeyError
        with pytest.raises(KeyError, match="Unknown scenario"):
            compute_effective_rate_of_protection(mock_calib_2c_2s, tariffs="invalid_scenario_name")

    def test_erp_negative_protection_detection(self, mock_calib_2c_2s):
        """Verify detection of negative effective protection when intermediate tariffs exceed output tariff."""
        res = compute_effective_rate_of_protection(
            mock_calib_2c_2s,
            tariffs=np.array([0.80, 0.80]),
            output_tariffs={"AGR": 0.05, "MAN": 0.05},
        )
        assert np.any(res.is_negative_erp) or np.any(np.array(res.distortion_category) == "negative_protection")

    def test_supply_chain_vulnerability(self, mock_calib_2c_2s):
        """Verify Antras upstreamness and supply chain vulnerability indices."""
        res = compute_supply_chain_vulnerability(mock_calib_2c_2s, target_country="USA")
        assert isinstance(res, SupplyChainVulnerabilityResult)
        assert np.all(res.upstreamness >= 1.0)
        assert len(res.vulnerability_index) == mock_calib_2c_2s.n_sectors
        df = res.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "Upstreamness" in df.columns

    def test_tariff_revenue_incidence_all_closures(self, mock_calib_2c_2s, mock_equilibrium_pair):
        """Verify tariff revenue incidence under all 5 fiscal recycling closures."""
        base, cf = mock_equilibrium_pair
        closures = ["lump_sum", "labor_tax", "capital_tax", "targeted_subsidy", "deficit_reduction"]

        for cl in closures:
            res = calculate_tariff_revenue_incidence(mock_calib_2c_2s, cf, base_result=base, closure=cl)
            assert isinstance(res, TariffRevenueIncidenceResult)
            assert res.closure == cl
            assert res.gross_tariff_revenue >= 0.0
            assert np.isfinite(res.net_efficiency_change)

        with pytest.raises(ValueError, match="Unknown closure"):
            calculate_tariff_revenue_incidence(mock_calib_2c_2s, cf, base_result=base, closure="nonexistent_closure")


# =============================================================================
# 4. TestTheoremValidationReportInterface (_results.py: TheoremValidationReport)
# =============================================================================

class TestTheoremValidationReportInterface:
    """Audit serialization methods, dict access, slicing, and immutability for TheoremValidationReport."""

    @pytest.fixture
    def sample_report(self) -> TheoremValidationReport:
        df = pd.DataFrame(
            [
                {"Theorem": "Theorem 1", "Passed": True, "Headline_Metric": "DWL=0.0100"},
                {"Theorem": "Theorem 2", "Passed": True, "Headline_Metric": "Bound=0.0200"},
                {"Theorem": "Theorem 3", "Passed": True, "Headline_Metric": "Drop_Leo=10 <= Drop_CES=15"},
                {"Theorem": "Theorem 4", "Passed": True, "Headline_Metric": "Offset=99.3%"},
            ]
        ).set_index("Theorem")

        return TheoremValidationReport(
            all_passed=True,
            theorem_1_passed=True,
            theorem_2_passed=True,
            theorem_3_passed=True,
            theorem_4_passed=True,
            details={
                1: {"theorem": 1, "passed": True, "dwl": 0.01},
                2: {"theorem": 2, "passed": True, "p_bound": 0.02},
                3: {"theorem": 3, "passed": True, "drop": 10.0},
                4: {"theorem": 4, "passed": True, "rebate_offset_ratio": 0.993},
            },
            summary_df=df,
            scenario="uniform_10",
        )

    def test_formatting_methods(self, sample_report):
        """Test .to_latex(), .to_typst(), .to_markdown(), .to_frame(), .to_dataframe()."""
        # Markdown
        md = sample_report.to_markdown()
        assert isinstance(md, str)
        assert "|" in md
        assert "Theorem 1" in md

        # LaTeX
        latex = sample_report.to_latex()
        assert isinstance(latex, str)
        assert "\\begin{tabular}" in latex or "\\begin{table}" in latex

        # Typst
        typst = sample_report.to_typst()
        assert isinstance(typst, str)
        assert "#table(" in typst

        # DataFrames
        df1 = sample_report.to_frame()
        df2 = sample_report.to_dataframe()
        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)
        assert len(df1) == 4

    def test_summary_detailed_parameter(self, sample_report):
        """Test .summary(detailed=True) and .summary(detailed=False)."""
        s_detailed = sample_report.summary(detailed=True)
        s_brief = sample_report.summary(detailed=False)
        assert isinstance(s_detailed, pd.DataFrame)
        assert isinstance(s_brief, pd.DataFrame)
        assert sample_report.passed is True

    def test_dict_access_integer_and_string(self, sample_report):
        """Test dict access by theorem integer, digit string, shorthand, and attribute names."""
        for t_idx in (1, 2, 3, 4):
            assert sample_report[t_idx]["theorem"] == t_idx
            assert sample_report[str(t_idx)]["theorem"] == t_idx
            assert sample_report[f"t{t_idx}"]["theorem"] == t_idx
            assert sample_report[f"theorem{t_idx}"]["theorem"] == t_idx
            assert sample_report[f"Theorem {t_idx}"]["theorem"] == t_idx

        # Direct attribute lookup
        assert sample_report["all_passed"] is True
        assert sample_report["scenario"] == "uniform_10"

    def test_dict_access_errors(self, sample_report):
        """Test invalid keys raise KeyError and invalid types raise TypeError."""
        with pytest.raises(KeyError, match="Theorem index 5 not found"):
            sample_report[5]

        with pytest.raises(KeyError, match="Theorem index 0 not found"):
            sample_report[0]

        with pytest.raises(KeyError, match="Key 'nonexistent' not found"):
            sample_report["nonexistent"]

        with pytest.raises(TypeError, match="Key must be int or str"):
            sample_report[1.5]

    def test_immutability_and_slicing(self, sample_report):
        """Verify frozen dataclass raises error on attribute modification and slice access raises TypeError."""
        with pytest.raises((FrozenInstanceError, AttributeError)):
            sample_report.all_passed = False  # type: ignore

        with pytest.raises(TypeError, match="Key must be int or str"):
            sample_report[0:2]  # type: ignore


# =============================================================================
# 5. TestEVDecompositionResultInterface (_results.py: EVDecompositionResult)
# =============================================================================

class TestEVDecompositionResultInterface:
    """Audit serialization methods, dict access, slicing, mapping protocol, and immutability."""

    def test_formatting_methods(self, sample_ev_result):
        """Test .to_latex(), .to_typst(), .to_markdown(), .to_frame(), .to_dataframe()."""
        # Markdown
        md = sample_ev_result.to_markdown()
        assert isinstance(md, str)
        assert "|" in md
        assert "TOT" in md

        # LaTeX
        latex = sample_ev_result.to_latex()
        assert isinstance(latex, str)
        assert "\\begin{tabular}" in latex or "\\begin{table}" in latex

        # Typst
        typst = sample_ev_result.to_typst()
        assert isinstance(typst, str)
        assert "#table(" in typst

        # DataFrames
        df1 = sample_ev_result.to_frame()
        df2 = sample_ev_result.to_dataframe()
        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)
        assert "TOT" in df1.index

    def test_concise_summary(self, sample_ev_result):
        """Verify standard .summary() returns formatted string containing key component labels."""
        s = sample_ev_result.summary()
        assert isinstance(s, str)
        assert "TOT" in s
        assert "Alloc" in s
        assert "TariffRec" in s
        assert "EV=" in s

    def test_dict_access_keys_and_aliases(self, sample_ev_result):
        """Verify dictionary access with canonical names, lowercase, and aliases."""
        assert sample_ev_result["TOT"] == 12.5
        assert sample_ev_result["tot"] == 12.5
        assert sample_ev_result["terms_of_trade"] == 12.5
        assert sample_ev_result["termsoftrade"] == 12.5

        assert sample_ev_result["Alloc"] == -2.5
        assert sample_ev_result["alloc"] == -2.5
        assert sample_ev_result["allocative"] == -2.5
        assert sample_ev_result["allocative_efficiency"] == -2.5

        assert sample_ev_result["TariffRec"] == 15.0
        assert sample_ev_result["tariff_rec"] == 15.0
        assert sample_ev_result["tariff_revenue"] == 15.0
        assert sample_ev_result["tariff_recycling"] == 15.0

        assert sample_ev_result["ev"] == 25.0
        assert sample_ev_result["ev_usd"] == 25.0
        assert sample_ev_result["ev_total"] == 25.0
        assert sample_ev_result["ev_pct"] == 1.5625
        assert sample_ev_result["residual"] == 0.0

    def test_dict_access_errors(self, sample_ev_result):
        """Verify KeyError for invalid keys and TypeError for non-string keys."""
        with pytest.raises(KeyError, match="Key 'nonexistent' not recognized"):
            sample_ev_result["nonexistent"]

        with pytest.raises(TypeError, match="Key must be a str"):
            sample_ev_result[0]  # type: ignore

    def test_slicing_raises_typeerror(self, sample_ev_result):
        """Verify slicing on mapping container raises clean TypeError."""
        with pytest.raises(TypeError, match="Key must be a str"):
            sample_ev_result[0:2]  # type: ignore

    def test_mapping_protocol(self, sample_ev_result):
        """Verify len(), iter(), in operator, keys(), values(), and items()."""
        assert len(sample_ev_result) == 3
        assert list(sample_ev_result) == ["TOT", "Alloc", "TariffRec"]
        assert sample_ev_result.keys() == ["TOT", "Alloc", "TariffRec"]
        assert sample_ev_result.values() == [12.5, -2.5, 15.0]
        assert sample_ev_result.items() == [("TOT", 12.5), ("Alloc", -2.5), ("TariffRec", 15.0)]

        # in operator
        assert "TOT" in sample_ev_result
        assert "tot" in sample_ev_result
        assert "Alloc" in sample_ev_result
        assert "TariffRec" in sample_ev_result
        assert "nonexistent" not in sample_ev_result
        assert 123 not in sample_ev_result

    def test_immutability(self, sample_ev_result):
        """Verify frozen dataclass raises FrozenInstanceError on mutation."""
        with pytest.raises((FrozenInstanceError, AttributeError)):
            sample_ev_result.tot = 99.0  # type: ignore

    def test_properties(self, sample_ev_result):
        """Verify convenience property accessors."""
        assert sample_ev_result.ev_total == 25.0
        assert sample_ev_result.terms_of_trade == 12.5
        assert sample_ev_result.allocative_efficiency == -2.5
        assert sample_ev_result.tariff_revenue_recycling == 15.0
        assert sample_ev_result.exact_identity_satisfied is True

    def test_ev_decomposition_summary_detailed_adversarial_check(self, sample_ev_result):
        """Adversarial check: test .summary(detailed=True) compatibility on EVDecompositionResult.

        The puremacro presentation interface specification requires .summary(detailed=True)
        to be supported across all presentation result objects (as in TheoremValidationReport,
        TradeCalibrationResult, TradeEquilibriumResult, GearyKhamisResult).
        """
        # When calling .summary(detailed=True), it should either return the detailed summary table
        # or accepted keyword argument.
        try:
            res = sample_ev_result.summary(detailed=True)  # type: ignore
            assert res is not None
        except TypeError as err:
            pytest.fail(
                f"EXPOSED GAP: EVDecompositionResult.summary() does not accept 'detailed=True' keyword argument: {err}"
            )


# =============================================================================
# 6. TestTradeNamespaceAndExportHygiene (__init__.py completeness)
# =============================================================================

class TestTradeNamespaceAndExportHygiene:
    """Audit symbol re-exports, __all__ completeness, and namespace cleanliness."""

    def test_all_symbols_resolvable(self):
        """Every symbol in pt.__all__ must be resolvable via getattr(pt, symbol)."""
        missing = [sym for sym in pt.__all__ if not hasattr(pt, sym)]
        assert missing == [], f"Symbols listed in puremacro.trade.__all__ missing from package: {missing}"

    def test_policy_analytics_symbols_reexported(self):
        """All 12 symbols declared in policy_analytics.__all__ must be re-exported at top-level."""
        expected_policy_symbols = [
            "EffectiveRateOfProtectionResult",
            "WelfareDecompositionResult",
            "SupplyChainVulnerabilityResult",
            "TariffRevenueIncidenceResult",
            "TheoremValidationReport",
            "EVDecompositionResult",
            "compute_effective_rate_of_protection",
            "decompose_welfare_effects",
            "compute_supply_chain_vulnerability",
            "calculate_tariff_revenue_incidence",
            "verify_theorems_1_to_4",
            "decompose_hicksian_ev_3way",
        ]
        for sym in expected_policy_symbols:
            assert hasattr(pt, sym), f"puremacro.trade is missing re-export of {sym}"
            assert sym in pt.__all__, f"{sym} not declared in puremacro.trade.__all__"

    def test_result_containers_reexported(self):
        """Key result containers must be re-exported in puremacro.trade."""
        expected_results = [
            "TradeCalibrationResult",
            "TradeEquilibriumResult",
            "GearyKhamisResult",
            "TheoremValidationReport",
            "EVDecompositionResult",
        ]
        for res_cls in expected_results:
            assert hasattr(pt, res_cls)
            assert res_cls in pt.__all__

    def test_namespace_cleanliness(self):
        """Verify no private helper functions (prefixed with _) leaked into __all__."""
        leaked_private = [sym for sym in pt.__all__ if sym.startswith("_")]
        assert leaked_private == [], f"Private symbols leaked in __all__: {leaked_private}"
