"""Comprehensive Unit Tests for Milestone 1: Result-Object Parity, Presentations, and Elevations.

Tests:
2. BVAR_SVForecast export methods (.to_markdown(), .to_latex(), .to_typst()).
3. ScoreDiagnosticsResult .to_typst() export.
4. Elevated VFISolution (.summary(), .to_frame(), export quintet, headless .plot()).
5. Elevated HJBSolution (.summary(), .to_frame(), export quintet, headless .plot(), and dict mapping protocol).
5b. Report quintet (.to_dataframe(), .to_markdown(), .to_latex(), .to_typst(), .summary()) of the result
    classes of dsge.stacked_newton, built from the smallest fixture the module's own tests use.
6. Deprecation warning update verification (retiring stale 2.0.0 references).
7. ``.to_typst()`` of the trade result classes (calibration, equilibrium,
   Geary-Khamis, scenario batch, optimal tariff, Nash tariff, payoff matrix).
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
from puremacro.var.bvar_sv import BVAR_SVForecast
from puremacro.vfi import HJBSolution, VFIProblem, VFISolution, solve_hjb_achdou


@pytest.fixture(autouse=True)
def close_figures():
    """Ensure all figures are closed after each test."""
    yield
    plt.close("all")


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
# 4b. The stacked Newton-Krylov: the report quintet on real results
# ---------------------------------------------------------------------------
# Built from the smallest fixture the module's own tests use (an analytic growth model).
# Each result must render the quintet.


def _assert_quintet(obj, **frame_kwargs):
    frame = obj.to_dataframe(**frame_kwargs)
    assert isinstance(frame, pd.DataFrame) and not frame.empty
    for method in ("to_markdown", "to_latex", "to_typst"):
        text = getattr(obj, method)()
        assert isinstance(text, str) and text.strip(), method
    if hasattr(obj, "summary"):
        assert isinstance(obj.summary(), str) and obj.summary().strip()


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
# 4c. Trade result classes: .to_typst()
# ---------------------------------------------------------------------------

def _dummy_trade_equilibrium(nc: int = 2, ns: int = 2):
    """A minimal TradeEquilibriumResult with neutral codes."""
    from puremacro.trade import TradeEquilibriumResult

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
        country_codes=("C01", "C02"),
        sector_codes=("S01", "S02"),
    )


def _dummy_geary_khamis():
    from puremacro.trade import GearyKhamisResult

    return GearyKhamisResult(
        pi=np.ones(3),
        ppp=np.ones(2),
        real_gdp=np.array([100.0, 200.0]),
        nominal_gdp=np.array([100.0, 200.0]),
        gdp_growth=np.array([0.0, 0.0]),
        country_codes=("C01", "C02"),
    )


def _assert_typst_table(obj, **kwargs):
    typ = obj.to_typst(**kwargs)
    assert isinstance(typ, str)
    assert "#table(" in typ
    assert "columns:" in typ


def test_trade_calibration_result_to_typst():
    from puremacro.trade import TradeCalibrationResult

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
        country_codes=("C01", "C02"),
        sector_codes=("S01", "S02"),
    )
    _assert_typst_table(calib)


def test_trade_equilibrium_result_to_typst():
    _assert_typst_table(_dummy_trade_equilibrium())


def test_geary_khamis_result_to_typst():
    _assert_typst_table(_dummy_geary_khamis())


def test_scenario_batch_result_to_typst():
    from puremacro.trade import ScenarioBatchResult

    eq, gk = _dummy_trade_equilibrium(), _dummy_geary_khamis()
    batch = ScenarioBatchResult(
        scenarios={"base": eq, "tariff10": eq},
        geary_khamis={"base": gk, "tariff10": gk},
        baseline_scenario="base",
        country_codes=("C01", "C02"),
        sector_codes=("S01", "S02"),
    )
    _assert_typst_table(batch)
    _assert_typst_table(batch, metric="real_gdp_growth")


def test_optimal_tariff_result_to_typst():
    from puremacro.trade import OptimalTariffResult

    res = OptimalTariffResult(
        country_code="C01",
        optimal_tariff_rate=0.15,
        welfare_gain_pct=1.2,
        baseline_welfare=100.0,
        optimal_welfare=101.2,
        welfare_metric="geary_khamis",
        terms_of_trade_initial=1.0,
        terms_of_trade_optimal=1.04,
        tariff_grid=np.linspace(0.0, 0.3, 10),
        welfare_curve=np.linspace(100.0, 101.2, 10),
        equilibrium=_dummy_trade_equilibrium(),
    )
    _assert_typst_table(res)


def test_nash_tariff_result_to_typst():
    from puremacro.trade import NashTariffResult

    res = NashTariffResult(
        strategic_players=("C01", "C02"),
        nash_tariffs={"C01": 0.18, "C02": 0.12},
        welfare_changes_pct={"C01": -0.5, "C02": -0.8},
        terms_of_trade_changes_pct={"C01": 1.2, "C02": -0.3},
        world_welfare_change_pct=-0.6,
        outer_iterations=4,
        converged=True,
        outer_error=1e-5,
        equilibrium=_dummy_trade_equilibrium(),
        tau_nash=np.ones((2, 2, 2, 2)),
        tau_fd_nash=np.ones((2, 2, 3, 2)),
    )
    _assert_typst_table(res)


def test_welfare_payoff_matrix_result_to_typst():
    from puremacro.trade import WelfarePayoffMatrixResult

    eq = _dummy_trade_equilibrium()
    res = WelfarePayoffMatrixResult(
        players=("C01", "C02"),
        strategies=("Cooperate (0%)", "Defect (Nash)"),
        payoff_matrix=np.array([[[0.0, 0.0], [-1.0, 0.5]], [[0.5, -1.0], [-0.5, -0.5]]]),
        scenarios={"CC": eq, "CD": eq, "DC": eq, "DD": eq},
        summary_df=pd.DataFrame({"Player": ["C01", "C02"], "Cooperate": [0.0, 0.0]}),
    )
    _assert_typst_table(res)


# ---------------------------------------------------------------------------
# 5. Stale Deprecation Warnings Retired
# ---------------------------------------------------------------------------

def test_garch_utils_removed_in_v600():
    import sys
    sys.modules.pop("puremacro.lp.garch_utils", None)
    with pytest.raises(ImportError) as excinfo:
        import puremacro.lp.garch_utils  # noqa: F401
    msg = str(excinfo.value)
    assert "puremacro 6.0.0" in msg
    assert "puremacro.lp._garch_utils" in msg

    # Canonical path works cleanly
    from puremacro.lp._garch_utils import align_series_for_lp
    assert callable(align_series_for_lp)


def test_sigma_numpy_removed_in_v600():
    with pytest.raises((ImportError, AttributeError)) as excinfo:
        from puremacro.sigma.sigma_numpy import SigmaObject  # noqa: F401
    msg = str(excinfo.value)
    assert "6.0.0" in msg or "SigmaObject" in msg

    # Canonical path works cleanly
    from puremacro.volatility.sigma import SigmaObject
    obj = SigmaObject(sig=np.array([0.1, 0.2]), R=np.eye(2), sectors=["A", "B"])
    assert obj.var(np.array([1.0, 1.0])) > 0


# ---------------------------------------------------------------------------
# 8. Generic GMM result (puremacro.inference.gmm)
# ---------------------------------------------------------------------------

def test_gmm_fit_result_quintet_and_plot():
    from puremacro.inference import GMMFitResult, gmm

    rng = np.random.default_rng(0)
    T = 200
    z = rng.standard_normal((T, 2))
    x = z @ np.array([1.0, 0.5]) + rng.standard_normal(T)
    y = 1.0 + 0.5 * x + rng.standard_normal(T)
    Z = np.column_stack([np.ones(T), z])
    X = np.column_stack([np.ones(T), x])
    res = gmm(lambda th, d: d[2] * (d[0] - d[1] @ th)[:, None], np.zeros(2),
              (y, X, Z), hac_lags=0)
    assert isinstance(res, GMMFitResult)
    assert isinstance(res.to_dataframe(), pd.DataFrame)
    assert isinstance(res.to_frame(), pd.DataFrame)
    assert "|" in res.to_markdown()
    assert "\\begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()
    assert isinstance(res.summary(), str)
    fig = res.plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    _, ax = plt.subplots()
    assert res.plot(ax=ax) is ax


# ---------------------------------------------------------------------------
# 9. Asset-pricing results (puremacro.asset_pricing)
# ---------------------------------------------------------------------------

def _ap_results():
    from puremacro import asset_pricing as ap

    rng = np.random.default_rng(0)
    T, N, K = 120, 5, 2
    idx = pd.date_range("2000-01-01", periods=T, freq="MS")
    f = pd.DataFrame(0.005 + 0.04 * rng.standard_normal((T, K)), index=idx, columns=["mkt", "hml"])
    r = pd.DataFrame(f.to_numpy() @ rng.uniform(0.5, 1.5, (K, N))
                     + 0.02 * rng.standard_normal((T, N)), index=idx,
                     columns=[f"p{i}" for i in range(N)])
    g = pd.Series(np.exp(0.005 + 0.02 * rng.standard_normal(T)), index=idx)
    gross = 1.0 + r.iloc[:, :2]
    return {
        "RiskSummaryResult": ap.risk_summary(r, periods_per_year=12),
        "VaRBacktestResult": ap.var_backtest(r["p0"], 0.05),
        "TimeSeriesTestResult": ap.time_series_test(r, f),
        "FamaMacBethResult": ap.fama_macbeth(r, f),
        "SDFResult": ap.sdf_gmm(r, f),
        "HJDistanceResult": ap.hj_distance(r, f, n_sim=1_000),
        "CCAPMResult": ap.ccapm(gross, g, instruments=np.ones((T, 1))),
    }


@pytest.mark.parametrize("cls_name", [
    "RiskSummaryResult", "VaRBacktestResult", "TimeSeriesTestResult", "FamaMacBethResult",
    "SDFResult", "HJDistanceResult", "CCAPMResult",
])
def test_asset_pricing_result_quintet_and_plot(cls_name):
    from puremacro import asset_pricing as ap

    res = _ap_results()[cls_name]
    assert type(res) is getattr(ap, cls_name)
    assert isinstance(res.to_dataframe(), pd.DataFrame)
    assert isinstance(res.to_frame(), pd.DataFrame)
    assert "|" in res.to_markdown()
    assert "\\begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()
    assert isinstance(res.summary(), str)
    if cls_name == "VaRBacktestResult":
        assert not hasattr(res, "plot")
        return
    fig = res.plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    _, ax = plt.subplots()
    assert res.plot(ax=ax) is ax


def test_irf_comparison_result_quintet_and_plot():
    from puremacro.lp import lp_hac
    from puremacro.reports import compare_irfs

    rng = np.random.default_rng(11)
    n = 80
    x = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.4 * y[t - 1] + 0.6 * x[t] + 0.3 * rng.standard_normal()
    df = pd.DataFrame({"y": y, "x": x})
    res = compare_irfs(
        {
            "L1": lp_hac(df, "y", "x", horizons=range(4), lags=1),
            "L2": lp_hac(df, "y", "x", horizons=range(4), lags=2),
        }
    )

    assert isinstance(res.to_frame(), pd.DataFrame)
    assert isinstance(res.to_dataframe(), pd.DataFrame)
    assert isinstance(res.summary(), str) and res.summary()
    assert "|" in res.to_markdown()
    assert "\\begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()

    fig = res.plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    _, ax = plt.subplots()
    assert res.plot(ax=ax) is ax


def test_dsge_validation_scorecard_quintet_and_plot():
    from puremacro.dsge import NormalPrior, nuts_sample, prior_posterior_comparison, score_irf_against_targets
    from puremacro.structural.bridge import MomentTargets

    priors = {"theta": NormalPrior(mean=0.0, std=1.0)}
    nuts_res = nuts_sample(
        lambda x: (-0.5 * float(x[0] ** 2), -x),
        init_params=np.array([0.0]),
        n_draws=40,
        n_chains=2,
        warmup=20,
        param_names=("theta",),
        seed=0,
        priors=priors,
    )
    ppc = prior_posterior_comparison(nuts_res)
    for res_obj in (
        ppc,
        score_irf_against_targets(
            pd.DataFrame({"y": [0.2, 0.1, 0.05]}, index=[0, 1, 2]),
            MomentTargets.from_irf(
                np.array([0.2, 0.1, 0.05]),
                covariance=np.eye(3) * 0.01,
                response="y",
                shock="eps",
                units="pct",
            ),
        ),
    ):
        assert isinstance(res_obj.to_frame(), pd.DataFrame)
        assert isinstance(res_obj.to_dataframe(), pd.DataFrame)
        assert isinstance(res_obj.summary(), str) and res_obj.summary()
        assert "|" in res_obj.to_markdown()
        assert "\\begin{tabular}" in res_obj.to_latex()
        assert "#table(" in res_obj.to_typst()

        fig = res_obj.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        _, ax = plt.subplots()
        assert res_obj.plot(ax=ax) is ax



def test_scenario_comparison_result_quintet_and_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from puremacro.trade import (
        TariffScenario,
        calibrate_trade_model,
        run_scenario_batch,
        summarize_scenario_batch,
    )
    from puremacro.trade.data import synthetic_codes, synthetic_icio_table

    table = synthetic_icio_table(2, 2, nfd=3, seed=7)
    countries, sectors = synthetic_codes(2, 2)
    calib = calibrate_trade_model(
        table, ns=2, nc=2, nfd=3, country_codes=countries, sector_codes=sectors
    )
    batch = run_scenario_batch(
        calib, [TariffScenario("C01", {("C02", "*"): 0.10}, name="tariff_ten")], baseline_name="base"
    )
    comp = summarize_scenario_batch(batch, calib)

    assert isinstance(comp.to_frame(), pd.DataFrame)
    assert isinstance(comp.to_dataframe(), pd.DataFrame)
    assert isinstance(comp.summary(), str) and comp.summary()
    assert "|" in comp.to_markdown()
    assert "\\begin{tabular}" in comp.to_latex()
    assert "#table(" in comp.to_typst()

    fig = comp.plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
    fig2, ax = plt.subplots()
    assert comp.plot(ax=ax) is ax
    plt.close(fig2)



def test_preflight_report_quintet_and_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from puremacro.data import preflight_panel

    rng = np.random.default_rng(19)
    dates = pd.date_range("2015-01-01", periods=24, freq="QS")
    df = pd.DataFrame({
        "unit": ["A"] * 24 + ["B"] * 24,
        "date": list(dates) * 2,
        "y": rng.normal(0.0, 1.0, size=48),
        "x": rng.normal(0.0, 1.0, size=48),
    })
    rep = preflight_panel(df, unit_col="unit", time_col="date")

    assert isinstance(rep.to_frame(), pd.DataFrame)
    assert isinstance(rep.to_dataframe(), pd.DataFrame)
    assert isinstance(rep.summary(), str) and rep.summary()
    assert "|" in rep.to_markdown()
    assert "\\begin{tabular}" in rep.to_latex()
    assert "#table(" in rep.to_typst()
    assert rep.to_json()

    fig = rep.plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
    fig2, ax = plt.subplots()
    assert rep.plot(ax=ax) is ax
    plt.close(fig2)
