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
5b. Report quintet (.to_dataframe(), .to_markdown(), .to_latex(), .to_typst(), .summary()) of the result
    classes of trade.{ces_newton, household, continuation, stability, mrio, condensed, dynamic} and
    dsge.stacked_newton, each built from the smallest fixture the module's own tests use.
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
    title_loc = plt.rcParams["axes.titlelocation"]
    assert ax1.get_title(loc=title_loc) == "Value Function V(a, z)"
    assert ax2.get_title(loc=title_loc) == "Policy Function a'(a, z)"


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


def test_vfi_solution_multi_asset_ragged_grids_to_frame_and_plot():
    """Multi-asset (and multi-shock) VFISolution with unequal grid lengths: to_frame() emits
    per-component state/policy columns and plot() falls back to the flat index axis,
    instead of raising on np.asarray of a ragged list."""
    a1 = np.linspace(0.0, 1.0, 4)
    a2 = np.linspace(0.0, 2.0, 3)
    z_grid = np.array([0.9, 1.1])
    P_z = np.array([[0.9, 0.1], [0.1, 0.9]])

    def ret_fn(ap1, ap2, x1, x2, z, xp=np):
        c = z + 1.02 * (x1 + x2) - ap1 - ap2
        return xp.where(c > 0, xp.log(xp.maximum(c, 1e-10)), -1e10)

    sol = VFIProblem(a_grid=[a1, a2], z_grid=z_grid, P_z=P_z, return_fn=ret_fn,
                     beta=0.9, options={"tol": 1e-6}).solve()
    assert sol.endo_shape == (4, 3)

    df = sol.to_frame()
    assert len(df) == 12 * 2
    assert {"a_idx", "z_idx", "a_0", "a_1", "z", "V", "policy_aprime", "aprime_0", "aprime_1"} <= set(df.columns)
    assert "a" not in df.columns and "aprime_val" not in df.columns
    # State columns unravel the flat endogenous index in C order (component 1 fastest)
    a_flat = df["a_idx"].to_numpy()
    np.testing.assert_array_equal(df["a_0"].to_numpy(), a1[a_flat // 3])
    np.testing.assert_array_equal(df["a_1"].to_numpy(), a2[a_flat % 3])
    # Policy columns agree with policy_components()
    c0, c1 = sol.policy_components()
    np.testing.assert_array_equal(df["aprime_0"].to_numpy(), a1[c0.ravel()])
    np.testing.assert_array_equal(df["aprime_1"].to_numpy(), a2[c1.ravel()])

    fig = sol.plot(show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert fig.axes[0].get_xlabel() == "Asset Index (a_idx)"
    assert isinstance(sol.summary(), pd.DataFrame)
    assert "#table(" in sol.to_typst()

    # Multi-shock exogenous grids get z_<m> columns
    def ret_fn_z(ap, x, z1, z2, xp=np):
        c = z1 * z2 + x - ap
        return xp.where(c > 0, xp.log(xp.maximum(c, 1e-10)), -1e10)

    sol_z = VFIProblem(a_grid=np.linspace(0.0, 1.0, 5), z_grid=[np.array([0.9, 1.1]), np.array([0.5, 1.0, 1.5])],
                       P_z=np.full((6, 6), 1.0 / 6.0), return_fn=ret_fn_z, beta=0.9,
                       options={"tol": 1e-6}).solve()
    df_z = sol_z.to_frame()
    assert {"a", "z_0", "z_1", "aprime_val"} <= set(df_z.columns)
    assert len(df_z) == 5 * 6
    assert sol_z.plot(show=False) is not None


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
# 4b. MRIO engines and the stacked Newton-Krylov: the report quintet on real results
# ---------------------------------------------------------------------------
# One short test per result class, each built from the smallest fixture the module's own
# tests use (analytic or two-country tables). Each result must render the quintet.


def _assert_quintet(obj, **frame_kwargs):
    frame = obj.to_dataframe(**frame_kwargs)
    assert isinstance(frame, pd.DataFrame) and not frame.empty
    for method in ("to_markdown", "to_latex", "to_typst"):
        text = getattr(obj, method)()
        assert isinstance(text, str) and text.strip(), method
    if hasattr(obj, "summary"):
        assert isinstance(obj.summary(), str) and obj.summary().strip()


def _two_country_trade_calibration():
    """The two-country, one-sector table of docs/trade_ces_newton.md."""
    from puremacro.trade import calibrate_trade_model

    Z = np.array([[10.0, 12.0], [8.0, 14.0]])
    F = np.array([[30.0, 8.0, 20.0, 20.0], [15.0, 15.0, 50.0, 18.0]])
    production_tax = np.array([4.0, 6.0])
    final_tax = np.array([2.0, 1.0, 3.0, 2.0])
    output = Z.sum(1) + F.sum(1)
    va = output - Z.sum(0) - production_tax
    table = np.vstack([np.hstack([Z, F]), np.r_[production_tax, final_tax],
                       np.r_[2 * va / 3, np.zeros(4)], np.r_[va / 3, np.zeros(4)]])
    return calibrate_trade_model(table, ns=1, nc=2, nfd=2, country_codes=["A", "B"])


def _two_country_tariffs():
    tau = np.ones((2, 1, 2))
    tau[1, 0, 0] = 1.2
    tau_fd = np.ones((2, 2, 2))
    tau_fd[1, :, 0] = 1.2
    return tau, tau_fd


def test_ces_block_newton_result_quintet():
    from puremacro.trade.ces_newton import CESBlockNewtonResult, NestedCESTechnology, solve_ces_block_newton

    tau, tau_fd = _two_country_tariffs()
    tech = NestedCESTechnology(sigma_va_materials=0.3, sigma_sectors=0.5, sigma_origins=1.5, rho_va=0.8)
    res = solve_ces_block_newton(_two_country_trade_calibration(), tau, tau_fd, technology=tech)
    assert isinstance(res, CESBlockNewtonResult) and res.converged
    _assert_quintet(res)


@pytest.fixture(scope="module")
def household_prefs():
    from puremacro.trade.household import calibrate_household

    x0 = np.array([[25.0, 10.0], [45.0, 60.0], [30.0, 30.0]])
    eta = np.array([[0.6, 0.5], [0.9, 1.1], [1.3, 1.2]])
    return x0, eta, calibrate_household(x0, "stone_geary", expenditure_elasticities=eta,
                                        supernumerary_share=[0.4, 0.5],
                                        sector_codes=["AGR", "MAN", "SRV"], country_codes=["USA", "ROW"])


def test_household_calibration_result_quintet(household_prefs):
    from puremacro.trade.household import HouseholdCalibrationResult

    _, _, prefs = household_prefs
    assert isinstance(prefs.calibration, HouseholdCalibrationResult)
    _assert_quintet(prefs.calibration)


def test_household_demand_result_quintet(household_prefs):
    from puremacro.trade.household import HouseholdDemandResult

    _, _, prefs = household_prefs
    prices = np.array([[1.3, 0.8], [0.9, 1.25], [1.1, 1.05]])
    state = prefs.evaluate(prices, np.array([105.0, 110.0]), validate=True)
    assert isinstance(state, HouseholdDemandResult)
    _assert_quintet(state)


def test_household_welfare_result_quintet(household_prefs):
    from puremacro.trade.household import HouseholdWelfareResult

    _, _, prefs = household_prefs
    prices = np.array([[1.3, 0.8], [0.9, 1.25], [1.1, 1.05]])
    welfare = prefs.welfare(prices, np.array([105.0, 110.0]), validate=True)
    assert isinstance(welfare, HouseholdWelfareResult)
    _assert_quintet(welfare)


def test_supernumerary_fit_result_quintet(household_prefs):
    from puremacro.trade.household import SupernumeraryFitResult, fit_supernumerary_share

    x0, eta, prefs = household_prefs
    own = -prefs.supernumerary_share * prefs.eta * (1 - prefs.beta)
    fit = fit_supernumerary_share(x0, eta, own)
    assert isinstance(fit, SupernumeraryFitResult)
    _assert_quintet(fit)


@pytest.fixture(scope="module")
def continuation_problem():
    from dataclasses import replace

    from puremacro.trade._oecd_icio import condense_final_demand
    from puremacro.trade.data import generate_synthetic_mrio, package_mrio_to_calibration_result

    raw = generate_synthetic_mrio("oecd", custom_c=3, custom_s=3, seed=2)
    raw = replace(raw, taxes_less_subsidies_fd=np.zeros(raw.C * raw.K_F))
    calib = package_mrio_to_calibration_result(condense_final_demand(raw))
    nc, ns, nfd = calib.nc, calib.ns, calib.n_final_demand
    tau, tau_fd = np.ones((ns * nc, ns, nc)), np.ones((ns * nc, nfd, nc))
    tau[ns:, :, 0], tau_fd[ns:, :, 0] = 1.1, 1.1
    return calib, tau, tau_fd


def test_parameter_continuation_result_quintet(continuation_problem):
    from puremacro.trade.continuation import ParameterContinuationResult, sigma_path

    calib, tau, tau_fd = continuation_problem
    path = sigma_path(calib, tau, tau_fd, 0.5, tol=1e-9)
    assert isinstance(path, ParameterContinuationResult)
    _assert_quintet(path)


def test_direct_target_result_quintet(continuation_problem):
    from puremacro.trade.continuation import DirectTargetResult, try_starts

    calib, tau, tau_fd = continuation_problem
    multi = try_starts(calib, dict(tau=tau, tau_fd=tau_fd, sigma=0.5, tol=1e-9), [{"name": "calibrated", "x0": None}])
    assert isinstance(multi, DirectTargetResult)
    _assert_quintet(multi)


def test_reduced_stability_result_quintet():
    from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium
    from puremacro.trade.stability import StabilityResult, reduced_stability

    y0, alpha = np.array([100.0, 150.0]), np.array([0.3, 0.4])
    F = np.array([[60.0, 40.0], [40.0, 110.0]])
    data = np.zeros((5, 6))
    data[3, :2], data[4, :2] = (1 - alpha) * y0, alpha * y0
    for c in range(2):
        data[:2, 2 + 2 * c:4 + 2 * c] = F[:, c][:, None] * np.array([0.7, 0.3])[None, :]
    calib = calibrate_trade_model(data, ns=1, nc=2, nfd=2, country_codes=["A", "B"])
    base = solve_trade_equilibrium(calib, accounting="consistent", tol=1e-12)
    report = reduced_stability(calib, base)
    assert isinstance(report, StabilityResult)
    _assert_quintet(report)


def _mrio_toy_table():
    from puremacro.trade import mrio as m

    z = np.array([[1, 0.1, 0.2, 0], [0.3, 1, 0, 0.1], [0.1, 0, 2, 0.2], [0, 0.2, 0.1, 1]])
    C = np.array([[4.0, 0.5], [1.0, 1.0], [0.4, 5.0], [1.0, 3.0]])
    G = np.array([[1.0, 0.1], [2.0, 0.2], [0.1, 1.5], [0.2, 2.0]])
    X = np.array([[0.1, 0.2], [0, 0], [0.3, 0.2], [0, 0]])
    F = np.stack([C, G, X], axis=2)
    output = z.sum(1) + F.sum(axis=(1, 2))
    taxes = np.array([0.1, -0.1, 0.2, 0.1])
    va = output - z.sum(0) - taxes
    return m.MRIOTable.from_arrays(z, F, va, taxes, np.array([[0.2, 0.1, 0.0], [0.4, 0.3, 0.0]]),
                                   country_codes=("A", "B"), sector_codes=("goods", "services"),
                                   fd_codes=("C", "G", "X"), output=output,
                                   merchandise_mask=np.array([True, False]))


def test_mrio_accounting_report_quintet():
    from puremacro.trade.mrio import MRIOAccountingReport

    table = _mrio_toy_table()
    _assert_quintet(table)
    report = table.accounting_report()
    assert isinstance(report, MRIOAccountingReport)
    _assert_quintet(report)


def test_mrio_build_report_quintet():
    from puremacro.trade.mrio import MRIOBuildReport, regularize_table

    balanced, report = regularize_table(_mrio_toy_table())
    assert isinstance(report, MRIOBuildReport) and report.passed
    _assert_quintet(report)


def test_coarse_tariff_result_quintet():
    from puremacro.trade.mrio import CoarseTariffResult, Concordance, coarse_tariff_rates

    table = _mrio_toy_table()
    one = Concordance.from_mapping("one", {"goods": "ALL", "services": "ALL"})
    rates = np.array([[0.0, 0.0], [0.1, 0.0]])       # A taxes B's goods at 10%
    res = coarse_tariff_rates(rates, table, one, "output", importer="A")
    assert isinstance(res, CoarseTariffResult)
    _assert_quintet(res)


@pytest.fixture(scope="module")
def condensed_solution():
    from puremacro.trade.condensed import (BalancedIOTable, build_tariff_wedges, calibrate_condensed,
                                           compute_measures, solve_condensed)
    from tools.reference_validation.validate_oecd import load_fixture

    table = BalancedIOTable.from_raw(load_fixture(), reference_country="REST")
    calib = calibrate_condensed(table)
    wedges = build_tariff_wedges(calib, 0.10, importer="USA")
    result = solve_condensed(calib, wedges)
    return calib, wedges, result, compute_measures(calib, wedges, result.state, country="USA")


def test_condensed_equilibrium_result_quintet(condensed_solution):
    from puremacro.trade.condensed import CondensedEquilibriumResult

    result = condensed_solution[2]
    assert isinstance(result, CondensedEquilibriumResult) and result.passed
    _assert_quintet(result)


def test_raw_flow_certificate_quintet(condensed_solution):
    from puremacro.trade.condensed import RawFlowCertificate

    certificate = condensed_solution[2].certificate
    assert isinstance(certificate, RawFlowCertificate) and certificate.passed
    _assert_quintet(certificate)


def test_condensed_measures_result_quintet(condensed_solution):
    from puremacro.trade.condensed import CondensedMeasuresResult

    measures = condensed_solution[3]
    assert isinstance(measures, CondensedMeasuresResult)
    _assert_quintet(measures)


@pytest.fixture(scope="module")
def dynamic_fixture():
    from puremacro.trade.dynamic import (DynamicEconomy, DynamicTariff, analytic_two_country_accounts,
                                         calibrate_dynamic, solve_dynamic_steady_state,
                                         solve_dynamic_transition, tariff_path)

    calibration = calibrate_dynamic(analytic_two_country_accounts())
    economy = DynamicEconomy(calibration, adjustment_cost=2, risk_aversion=2)
    rates = np.zeros((calibration.n_cells, calibration.n_countries))
    rates[calibration.country == 0, 1] = 0.06
    rates[calibration.country == 1, 0] = 0.08
    shock = DynamicTariff.build(rates, label="analytic tariff")
    baseline = solve_dynamic_steady_state(economy)
    terminal = solve_dynamic_steady_state(economy, shock, tol=1e-11)
    run = solve_dynamic_transition(economy, tariff_path(economy.zero_policy(), shock, horizon=24),
                                   terminal=terminal, tol=1e-10)
    return calibration, economy, baseline, terminal, run


def test_dynamic_calibration_quintet(dynamic_fixture):
    from puremacro.trade.dynamic import DynamicCalibration

    calibration = dynamic_fixture[0]
    assert isinstance(calibration, DynamicCalibration)
    _assert_quintet(calibration)


def test_dynamic_steady_state_result_quintet(dynamic_fixture):
    from puremacro.trade.dynamic import DynamicSteadyStateResult

    terminal = dynamic_fixture[3]
    assert isinstance(terminal, DynamicSteadyStateResult)
    _assert_quintet(terminal)


def test_dynamic_transition_and_welfare_results_quintet(dynamic_fixture):
    from puremacro.trade.dynamic import ConsumptionEquivalentResult, DynamicTransitionResult

    run = dynamic_fixture[4]
    assert isinstance(run, DynamicTransitionResult)
    _assert_quintet(run)
    assert isinstance(run.welfare, ConsumptionEquivalentResult)
    _assert_quintet(run.welfare)


def test_dynamic_stability_result_quintet(dynamic_fixture):
    from puremacro.trade.dynamic import DynamicStabilityResult, stability_report

    _, economy, baseline, _, _ = dynamic_fixture
    report = stability_report(economy, baseline)
    assert isinstance(report, DynamicStabilityResult)
    _assert_quintet(report)


def test_stacked_newton_results_quintet():
    from puremacro.dsge.stacked_newton import (HorizonComparison, StackedNewtonResult, StackedProblem,
                                               compare_horizons, solve_stacked_newton_krylov)

    alpha, beta, delta = 0.33, 0.96, 0.10
    k_ss = (alpha / (1 / beta - (1 - delta))) ** (1 / (1 - alpha))
    c_ss = k_ss ** alpha - delta * k_ss

    def equations_fn(y_plus, y_curr, y_lag, eps):
        euler = 1 / y_curr[0] - beta / y_plus[0] * (alpha * float(eps) * y_curr[1] ** (alpha - 1) + 1 - delta)
        resource = y_curr[1] - (float(eps) * y_lag[1] ** alpha + (1 - delta) * y_lag[1] - y_curr[0])
        return [euler, resource]

    y_ss, y_init = np.array([c_ss, k_ss]), np.array([c_ss, 0.5 * k_ss])
    short = solve_stacked_newton_krylov(StackedProblem(equations_fn, y_init, y_ss, np.ones(40),
                                                       variable_names=["c", "k"]), tol=1e-8)
    long = solve_stacked_newton_krylov(StackedProblem(equations_fn, y_init, y_ss, np.ones(80),
                                                      variable_names=["c", "k"]), tol=1e-8)
    assert isinstance(short, StackedNewtonResult)
    _assert_quintet(short)
    comparison = compare_horizons(short, long, periods=20, tolerance=1e-5)
    assert isinstance(comparison, HorizonComparison)
    _assert_quintet(comparison)


# ---------------------------------------------------------------------------
# 5. Stale Deprecation Warnings Retired
# ---------------------------------------------------------------------------

def test_garch_utils_deprecation_warning():
    import sys
    sys.modules.pop("puremacro.lp.garch_utils", None)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        import puremacro.lp.garch_utils  # noqa: F401
    matches = [w for w in record if "puremacro.lp.garch_utils" in str(w.message)]
    assert len(matches) > 0
    msg = str(matches[0].message)
    # 4.0.0 shipped (2026-09-16) keeping the shim; the target was retargeted to 5.0.0 on
    # 2026-09-22 and tests/test_docs_hygiene_audit_fixes.py guards it against drifting again.
    assert "a future major release (5.0.0)" in msg
    assert "2.0.0" not in msg and "4.0.0" not in msg


def test_sigma_numpy_deprecation_warning():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        from puremacro.sigma.sigma_numpy import SigmaObject
        SigmaObject(sigma=np.array([0.1, 0.2]), R=np.eye(2), labels=["A", "B"])
    matches = [w for w in record if "SigmaObject is deprecated" in str(w.message)]
    assert len(matches) > 0
    msg = str(matches[0].message)
    # 4.0.0 shipped keeping this shim too; retargeted to 5.0.0 with garch_utils (2026-09-23).
    assert "a future major release (5.0.0)" in msg
    assert "2.0.0" not in msg and "4.0.0" not in msg
