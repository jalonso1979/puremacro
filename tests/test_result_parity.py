"""Comprehensive Unit Tests for Milestone 1: Result-Object Parity, Presentations, and Elevations.

Tests:
1. Result-Object Parity & .to_typst() for all 11 trade result classes:
   - TradeCalibrationResult
   - TradeEquilibriumResult
   - GearyKhamisResult
   - ScenarioBatchResult
   - RetaliationGameResult
   - JCurveDynamicResult
   - RevenueRecyclingResult
   - CapacityBottleneckResult
   - OptimalTariffResult
   - NashTariffResult
   - WelfarePayoffMatrixResult
2. BVAR_SVForecast export methods (.to_markdown(), .to_latex(), .to_typst()).
3. ScoreDiagnosticsResult .to_typst() export.
4. Elevated VFISolution (.summary(), .to_frame(), export quintet, headless .plot()).
5. Elevated HJBSolution (.summary(), .to_frame(), export quintet, headless .plot(), and dict mapping protocol).
6. Deprecation warning update verification (retiring stale 2.0.0 references).
"""
from __future__ import annotations

import warnings
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge._gradients import ScoreDiagnosticsResult
from puremacro.trade._results import (
    CapacityBottleneckResult,
    GearyKhamisResult,
    JCurveDynamicResult,
    NashTariffResult,
    OptimalTariffResult,
    RetaliationGameResult,
    RevenueRecyclingResult,
    ScenarioBatchResult,
    TradeCalibrationResult,
    TradeEquilibriumResult,
    WelfarePayoffMatrixResult,
)
from puremacro.var.bvar_sv import BVAR_SVForecast
from puremacro.vfi import HJBSolution, VFIProblem, VFISolution, solve_hjb_achdou


@pytest.fixture(autouse=True)
def close_figures():
    """Ensure all figures are closed after each test."""
    yield
    plt.close("all")


def _make_dummy_equilibrium(nc: int = 2, ns: int = 2) -> TradeEquilibriumResult:
    """Helper to create a minimal TradeEquilibriumResult."""
    codes = ("CAN", "USA")
    s_codes = ("S01", "S02")
    return TradeEquilibriumResult(
        x_sol=np.zeros(2 * ns * nc + 2 * nc + nc + nc - 1),
        p_sol=np.ones((1, ns, nc)),
        y_sol=np.ones((1, ns, nc)) * 50.0,
        r_sol=np.ones((1, 1, nc)),
        w_sol=np.ones((1, 1, nc)),
        T_sol=np.ones((1, 1, nc)) * 5.0,
        XN_sol=np.array([1.0]),
        c_sol=np.ones((1, 3, nc)) * 20.0,
        pfd_sol=np.ones((1, 3, nc)),
        terms_of_trade=np.array([1.02, 0.98]),
        country_codes=codes,
        sector_codes=s_codes,
    )


# ---------------------------------------------------------------------------
# 1. Trade Result Classes: .to_typst() Parity
# ---------------------------------------------------------------------------

def test_trade_calibration_result_to_typst():
    calib = TradeCalibrationResult(
        a=np.zeros((4, 2, 2)),
        afd=np.zeros((4, 3, 2)),
        alpha=np.full((1, 2, 2), 1 / 3),
        beta=np.ones((1, 2, 2)),
        k_endow=np.ones((1, 2)),
        l_endow=np.ones((1, 2)),
        invforT=np.zeros((1, 2)),
        tax=np.zeros((1, 2, 2)),
        ytot=np.ones((1, 2, 2)),
        n_countries=2,
        n_sectors=2,
        country_codes=("CAN", "USA"),
        sector_codes=("S01", "S02"),
    )
    typ = calib.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ
    assert "columns:" in typ


def test_trade_equilibrium_result_to_typst():
    eq = _make_dummy_equilibrium()
    typ = eq.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ
    assert "columns:" in typ


def test_geary_khamis_result_to_typst():
    gk = GearyKhamisResult(
        pi=np.ones(3),
        ppp=np.ones(2),
        real_gdp=np.array([100.0, 200.0]),
        nominal_gdp=np.array([100.0, 200.0]),
        gdp_growth=np.array([0.0, 0.0]),
        country_codes=("CAN", "USA"),
    )
    typ = gk.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ


def test_scenario_batch_result_to_typst():
    eq = _make_dummy_equilibrium()
    gk = GearyKhamisResult(
        pi=np.ones(3),
        ppp=np.ones(2),
        real_gdp=np.array([100.0, 200.0]),
        nominal_gdp=np.array([100.0, 200.0]),
        gdp_growth=np.array([0.0, 0.0]),
        country_codes=("CAN", "USA"),
    )
    batch = ScenarioBatchResult(
        scenarios={"base": eq, "tariff10": eq},
        geary_khamis={"base": gk, "tariff10": gk},
        baseline_scenario="base",
        country_codes=("CAN", "USA"),
        sector_codes=("S01", "S02"),
    )
    typ_selected = batch.to_typst(table_type="selected")
    assert isinstance(typ_selected, str)
    assert "#table(" in typ_selected

    typ_mean = batch.to_typst(table_type="mean")
    assert isinstance(typ_mean, str)
    assert "#table(" in typ_mean


def test_retaliation_game_result_to_typst():
    eq = _make_dummy_equilibrium()
    res = RetaliationGameResult(
        scenario_name="retaliation_test",
        equilibrium=eq,
        initial_scenario=None,
        tau_final=np.ones((2, 2, 2, 2)),
        tau_fd_final=np.ones((2, 2, 3, 2)),
        strategic_players=("CAN", "USA"),
        partner_tariffs={"CAN": {"S01": 0.1}},
        partner_duties_collected={"CAN": 50.0},
        us_duties_collected={"CAN": 45.0},
        outer_iterations=3,
        converged=True,
        outer_error=1e-5,
    )
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ


def test_jcurve_dynamic_result_to_typst():
    eq = _make_dummy_equilibrium()
    res = JCurveDynamicResult(
        scenario_name="jcurve_test",
        quarters=(0.0, 1.0, 2.0),
        sigma_path=(1.0, 2.0, 4.0),
        t_star=1.5,
        half_life=4.6,
        trade_balance_path=np.array([-10.0, -15.0, 5.0]),
        us_exports_path=np.array([100.0, 105.0, 120.0]),
        us_imports_path=np.array([110.0, 120.0, 115.0]),
        us_gdp_path=np.array([1000.0, 995.0, 1010.0]),
        us_cpi_path=np.array([1.0, 1.02, 1.01]),
        equilibria=(eq, eq, eq),
    )
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ


def test_revenue_recycling_result_to_typst():
    eq = _make_dummy_equilibrium()
    res = RevenueRecyclingResult(
        scenario_name="recycling_test",
        closure="lump_sum",
        equilibrium=eq,
        tariff_revenue=100.0,
        household_transfer=100.0,
        factor_tax_cut=0.0,
        subsidy_rate=0.0,
        welfare_decomposition={
            "terms_of_trade": 10.0,
            "deadweight_loss": -5.0,
            "fiscal_dividend": 2.0,
            "net_welfare_change": 7.0,
        },
    )
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ


def test_capacity_bottleneck_result_to_typst():
    eq = _make_dummy_equilibrium()
    res = CapacityBottleneckResult(
        scenario_name="bottleneck_test",
        equilibrium=eq,
        capacity_margins={"S01": 0.05},
        capacity_limits={"S01": 105.0},
        output_levels={"S01": 102.0},
        capacity_utilization={"S01": 0.97},
        price_escalation={"S01": 3.5},
        penalty_multipliers={"S01": 0.02},
    )
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ


def test_optimal_tariff_result_to_typst():
    eq = _make_dummy_equilibrium()
    res = OptimalTariffResult(
        country_code="USA",
        optimal_tariff_rate=0.15,
        welfare_gain_pct=1.2,
        baseline_welfare=100.0,
        optimal_welfare=101.2,
        welfare_metric="geary_khamis",
        terms_of_trade_initial=1.0,
        terms_of_trade_optimal=1.04,
        tariff_grid=np.linspace(0.0, 0.3, 10),
        welfare_curve=np.linspace(100.0, 101.2, 10),
        equilibrium=eq,
    )
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ


def test_nash_tariff_result_to_typst():
    eq = _make_dummy_equilibrium()
    res = NashTariffResult(
        strategic_players=("USA", "CHN"),
        nash_tariffs={"USA": 0.18, "CHN": 0.12},
        welfare_changes_pct={"USA": -0.5, "CHN": -0.8},
        terms_of_trade_changes_pct={"USA": 1.2, "CHN": -0.3},
        world_welfare_change_pct=-0.6,
        outer_iterations=4,
        converged=True,
        outer_error=1e-5,
        equilibrium=eq,
        tau_nash=np.ones((2, 2, 2, 2)),
        tau_fd_nash=np.ones((2, 2, 3, 2)),
    )
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ


def test_welfare_payoff_matrix_result_to_typst():
    eq = _make_dummy_equilibrium()
    res = WelfarePayoffMatrixResult(
        players=("USA", "CHN"),
        strategies=("Cooperate (0%)", "Defect (Nash)"),
        payoff_matrix=np.array([[[0.0, 0.0], [-1.0, 0.5]], [[0.5, -1.0], [-0.5, -0.5]]]),
        scenarios={"CC": eq, "CD": eq, "DC": eq, "DD": eq},
        summary_df=pd.DataFrame({"Player": ["USA", "CHN"], "Cooperate": [0.0, 0.0]}),
    )
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ


# ---------------------------------------------------------------------------
# 2. VAR and DSGE Export Parity
# ---------------------------------------------------------------------------

def test_bvar_sv_forecast_exports():
    forecast = BVAR_SVForecast(
        paths=np.ones((20, 4, 2)),
        h_paths=np.zeros((20, 4, 2)),
        index=pd.RangeIndex(4),
        names=["inflation", "rate"],
        history=pd.DataFrame(np.ones((10, 2)), columns=["inflation", "rate"]),
        ci=0.9,
    )
    md = forecast.to_markdown()
    assert isinstance(md, str)
    assert "inflation" in md
    assert "|" in md

    tex = forecast.to_latex()
    assert isinstance(tex, str)
    assert "\\begin{tabular}" in tex

    typ = forecast.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ


def test_score_diagnostics_result_to_typst():
    res = ScoreDiagnosticsResult(
        loglik=-120.5,
        gradient=np.array([0.05, -0.12]),
        param_names=("alpha", "beta"),
        elapsed_sec=0.015,
    )
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ
    assert "alpha" in typ
    assert "beta" in typ


# ---------------------------------------------------------------------------
# 3. Elevated VFISolution
# ---------------------------------------------------------------------------

def test_vfi_solution_elevation():
    sol = VFISolution(
        V=np.ones((15, 3)) * 4.2,
        policy_aprime=np.zeros((15, 3), dtype=int),
        policy_d=None,
        n_iter=12,
        sup_norm=5e-8,
        backend="numpy",
        endo_shape=(15,),
        a_grid=np.linspace(0.0, 10.0, 15),
        z_grid=np.array([0.5, 1.0, 1.5]),
    )

    # .summary()
    sm = sol.summary()
    assert isinstance(sm, pd.DataFrame)
    assert "Endogenous Grid Dimension (n_a)" in sm.index
    assert sm.loc["Total State Space", "Value"] == "45"
    assert sm.loc["Iterations", "Value"] == "12"

    # .to_frame()
    df = sol.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 45
    assert "V" in df.columns
    assert "policy_aprime" in df.columns
    assert "a" in df.columns
    assert "z" in df.columns
    assert "aprime_val" in df.columns

    # formatters
    assert isinstance(sol.to_markdown(), str)
    assert "\\begin{tabular}" in sol.to_latex()
    assert "#table(" in sol.to_typst()

    # .plot()
    fig = sol.plot(show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2

    # plot with pre-allocated axes
    fig2, (ax1, ax2) = plt.subplots(1, 2)
    sol.plot(ax=(ax1, ax2), show=False)
    assert ax1.get_title() == "Value Function V(a, z)"
    assert ax2.get_title() == "Policy Function a'(a, z)"


def test_vfi_problem_solve_returns_elevated_solution():
    """Verify that VFIProblem.solve populates a_grid and z_grid on VFISolution."""
    a_grid = np.linspace(0.1, 5.0, 10)
    z_grid = np.array([0.8, 1.2])
    P_z = np.array([[0.9, 0.1], [0.1, 0.9]])

    def ret_fn(ap, a, z, xp=np):
        c = z * (a ** 0.3) - ap
        return xp.where(c > 0, xp.log(xp.maximum(c, 1e-10)), -1e10)

    prob = VFIProblem(
        a_grid=a_grid,
        z_grid=z_grid,
        P_z=P_z,
        return_fn=ret_fn,
        beta=0.95,
        options={"max_iter": 50, "tol": 1e-5},
    )
    sol = prob.solve(backend="numpy")
    assert isinstance(sol, VFISolution)
    assert sol.a_grid is not None
    assert sol.z_grid is not None
    assert len(sol.a_grid) == 10
    assert len(sol.z_grid) == 2
    assert isinstance(sol.summary(), pd.DataFrame)
    assert len(sol.to_frame()) == 20


# ---------------------------------------------------------------------------
# 4. Elevated HJBSolution & solve_hjb_achdou
# ---------------------------------------------------------------------------

def test_hjb_solution_elevation_and_dict_compatibility():
    sol = solve_hjb_achdou(Na=30, max_iter=25)
    assert isinstance(sol, HJBSolution)

    # Dict mapping backward-compatibility checks
    assert sol["V"].shape == (30, 2)
    assert sol["c_policy"].shape == (30, 2)
    assert sol["s_drift"].shape == (30, 2)
    assert len(sol["a_grid"]) == 30
    assert len(sol["e_grid"]) == 2
    assert "V" in sol
    assert "c_policy" in sol
    assert "nonexistent_key" not in sol
    assert sol.get("n_iter") > 0
    assert sol.get("missing", 999) == 999
    assert len(sol.keys()) >= 7
    assert len(sol.values()) >= 7
    assert len(sol.items()) >= 7

    # Summary
    sm = sol.summary()
    assert isinstance(sm, pd.DataFrame)
    assert "Asset Grid Points (Na)" in sm.index
    assert sm.loc["Asset Grid Points (Na)", "Value"] == "30"
    assert sm.loc["Income Shock States (Ne)", "Value"] == "2"

    # Tabular DataFrame
    df = sol.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 60
    assert set(df.columns) >= {"asset_a", "prod_e", "value_V", "consumption_c", "savings_drift_s"}

    # Formatters
    assert isinstance(sol.to_markdown(), str)
    assert "\\begin{tabular}" in sol.to_latex()
    assert "#table(" in sol.to_typst()

    # Plotting
    fig = sol.plot(show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 3


# ---------------------------------------------------------------------------
# 5. Stale Deprecation Warnings Retired
# ---------------------------------------------------------------------------

def test_garch_utils_deprecation_warning():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        import puremacro.lp.garch_utils  # noqa: F401
    matches = [w for w in record if "puremacro.lp.garch_utils" in str(w.message)]
    assert len(matches) > 0
    msg = str(matches[0].message)
    assert "a future major release (4.0.0)" in msg
    assert "2.0.0" not in msg


def test_sigma_numpy_deprecation_warning():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        from puremacro.sigma.sigma_numpy import SigmaObject
        SigmaObject(sigma=np.array([0.1, 0.2]), R=np.eye(2), labels=["A", "B"])
    matches = [w for w in record if "SigmaObject is deprecated" in str(w.message)]
    assert len(matches) > 0
    msg = str(matches[0].message)
    assert "a future major release (4.0.0)" in msg
    assert "2.0.0" not in msg
