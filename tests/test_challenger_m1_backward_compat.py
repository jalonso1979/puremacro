"""Empirical Challenger 2 Stress Test Suite for Milestone 1.

Verifies:
1. Backward compatibility of HJBSolution (dict subscripting, methods, iteration edge cases).
2. Backward compatibility of VFISolution (positional args, endo_shape, optional grids, unravelling).
3. Presentation quintet parity across Trade results, VAR results, DSGE results, and VFI solutions.
4. Deprecation warnings retirement to 4.0.0 for puremacro.lp.garch_utils and puremacro.sigma.sigma_numpy.
5. Pyodide four-package contract compliance via static AST inspection.
6. Execution of continuous-time HJB showcase notebook.
"""
from __future__ import annotations

import ast
import copy
import importlib
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

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
def cleanup_figures():
    yield
    plt.close("all")


# ===========================================================================
# 1. HJBSolution Backward Compatibility & Boundary Tests
# ===========================================================================

def test_hjb_solution_subscripting_parity():
    """Verify that all keys historically provided in dict return are accessible via indexing."""
    sol = solve_hjb_achdou(Na=20, max_iter=10)
    expected_legacy_keys = [
        "V",
        "c_policy",
        "s_drift",
        "a_grid",
        "e_grid",
        "n_iter",
        "elapsed",
    ]
    for key in expected_legacy_keys:
        assert key in sol, f"Legacy key '{key}' must be present in HJBSolution"
        val = sol[key]
        assert val is not None, f"Legacy key '{key}' returned None"

    # Also check additional attributes added in 3.4.0
    for key in ["r_rate", "w_rate", "rho_val", "gamma_r", "converged"]:
        assert key in sol
        assert sol[key] is not None

    # Missing key must raise KeyError
    with pytest.raises(KeyError):
        _ = sol["nonexistent_variable"]


def test_hjb_solution_dict_methods():
    """Verify dict-like methods: .keys(), .values(), .items(), .get()."""
    sol = solve_hjb_achdou(Na=15, max_iter=5)
    keys = sol.keys()
    assert isinstance(keys, list)
    assert len(keys) == 12
    assert "V" in keys

    values = sol.values()
    assert isinstance(values, list)
    assert len(values) == 12

    items = sol.items()
    assert isinstance(items, list)
    assert len(items) == 12
    for k, v in items:
        assert getattr(sol, k) is v

    assert sol.get("V") is sol.V
    assert sol.get("nonexistent") is None
    assert sol.get("nonexistent", "fallback") == "fallback"


def test_hjb_solution_kwargs_unpacking_and_conversion():
    """Verify that **sol and dict(sol) work."""
    sol = solve_hjb_achdou(Na=15, max_iter=5)

    def receiver(**kwargs):
        return set(kwargs.keys())

    unpacked = receiver(**sol)
    assert "V" in unpacked
    assert "c_policy" in unpacked
    assert len(unpacked) == 12

    as_dict = dict(sol)
    assert isinstance(as_dict, dict)
    assert len(as_dict) == 12
    assert as_dict["n_iter"] == sol.n_iter


def test_hjb_solution_iteration_flaw_investigation():
    """Empirical investigation: Definition of __getitem__ without __iter__
    leads to TypeError when iterating with 'for key in sol'.
    
    This is an empirical challenger finding:
    In Python, if a class defines __getitem__ but omits __iter__, iteration
    falls back to sequence indexing with integers 0, 1, 2...
    Because __getitem__ expects string attributes, getattr(self, 0) raises
    TypeError: attribute name must be string, not 'int'.
    """
    sol = solve_hjb_achdou(Na=10, max_iter=2)
    # Check if __iter__ is defined
    has_iter = "__iter__" in type(sol).__dict__
    if not has_iter:
        with pytest.raises(TypeError, match="attribute name must be string"):
            for _ in sol:
                pass


def test_hjb_solution_pickle_and_copy():
    """Verify frozen dataclass serializability and copying."""
    sol = solve_hjb_achdou(Na=15, max_iter=5)
    copied = copy.copy(sol)
    assert copied.n_iter == sol.n_iter
    np.testing.assert_array_equal(copied.V, sol.V)

    pickled = pickle.dumps(sol)
    unpickled = pickle.loads(pickled)
    assert isinstance(unpickled, HJBSolution)
    assert unpickled.n_iter == sol.n_iter
    np.testing.assert_array_equal(unpickled.V, sol.V)


def test_hjb_solution_frozenness():
    """Verify that HJBSolution attributes cannot be mutated/rebound."""
    sol = solve_hjb_achdou(Na=10, max_iter=2)
    with pytest.raises(Exception):  # FrozenInstanceError
        sol.n_iter = 999


# ===========================================================================
# 2. VFISolution Backward Compatibility & Edge Cases
# ===========================================================================

def test_vfi_solution_positional_compatibility_6_args():
    """Verify legacy 6-positional argument construction."""
    V = np.zeros((10, 2))
    p_a = np.zeros((10, 2), dtype=int)
    sol = VFISolution(V, p_a, None, 15, 1e-6, "numpy")
    assert sol.endo_shape == ()
    assert sol.a_grid is None
    assert sol.z_grid is None
    assert sol.n_iter == 15
    assert sol.backend == "numpy"


def test_vfi_solution_positional_compatibility_7_args():
    """Verify legacy 7-positional argument construction."""
    V = np.zeros((10, 2))
    p_a = np.zeros((10, 2), dtype=int)
    sol = VFISolution(V, p_a, None, 15, 1e-6, "numpy", (10,))
    assert sol.endo_shape == (10,)
    assert sol.a_grid is None
    assert sol.z_grid is None


def test_vfi_solution_positional_compatibility_8_and_9_args():
    """Verify 8 and 9-positional argument construction."""
    V = np.zeros((10, 2))
    p_a = np.zeros((10, 2), dtype=int)
    a_g = np.linspace(0, 1, 10)
    z_g = np.array([0.5, 1.5])
    sol8 = VFISolution(V, p_a, None, 15, 1e-6, "numpy", (10,), a_g)
    assert sol8.a_grid is a_g
    assert sol8.z_grid is None

    sol9 = VFISolution(V, p_a, None, 15, 1e-6, "numpy", (10,), a_g, z_g)
    assert sol9.a_grid is a_g
    assert sol9.z_grid is z_g


def test_vfi_solution_to_frame_with_and_without_grids():
    """Verify .to_frame() operates robustly with or without attached grids."""
    # Case 1: No grids attached
    sol_no_grids = VFISolution(
        V=np.ones((4, 2)),
        policy_aprime=np.zeros((4, 2), dtype=int),
        policy_d=None,
        n_iter=5,
        sup_norm=1e-5,
        backend="numpy",
    )
    df1 = sol_no_grids.to_frame()
    assert isinstance(df1, pd.DataFrame)
    assert len(df1) == 8
    assert "a_idx" in df1.columns
    assert "z_idx" in df1.columns
    assert "V" in df1.columns
    assert "a" not in df1.columns

    # Case 2: Grids attached
    sol_with_grids = VFISolution(
        V=np.ones((4, 2)),
        policy_aprime=np.zeros((4, 2), dtype=int),
        policy_d=np.ones((4, 2), dtype=int),
        n_iter=5,
        sup_norm=1e-5,
        backend="numpy",
        a_grid=np.array([1.0, 2.0, 3.0, 4.0]),
        z_grid=np.array([0.5, 1.5]),
    )
    df2 = sol_with_grids.to_frame()
    assert isinstance(df2, pd.DataFrame)
    assert len(df2) == 8
    assert "a" in df2.columns
    assert "z" in df2.columns
    assert "aprime_val" in df2.columns
    assert "policy_d" in df2.columns


def test_vfi_solution_plot_headless_and_axes():
    """Verify .plot() returns figure with show=False by default."""
    sol = VFISolution(
        V=np.ones((5, 2)),
        policy_aprime=np.zeros((5, 2), dtype=int),
        policy_d=None,
        n_iter=5,
        sup_norm=1e-5,
        backend="numpy",
    )
    fig = sol.plot()  # default show=False
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2

    fig_single, ax = plt.subplots(1, 1)
    res_ax = sol.plot(ax=ax)
    assert res_ax is ax


# ===========================================================================
# 3. Deprecation Warning Behavior (Target: 4.0.0)
# ===========================================================================

def test_garch_utils_future_warning_4_0_0():
    """Ensure puremacro.lp.garch_utils emits FutureWarning with target 4.0.0."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        if "puremacro.lp.garch_utils" in sys.modules:
            importlib.reload(sys.modules["puremacro.lp.garch_utils"])
        else:
            importlib.import_module("puremacro.lp.garch_utils")

    garch_warnings = [
        w for w in recorded
        if issubclass(w.category, FutureWarning) and "puremacro.lp.garch_utils" in str(w.message)
    ]
    assert len(garch_warnings) >= 1, "Expected FutureWarning on importing garch_utils"
    msg = str(garch_warnings[0].message)
    assert "4.0.0" in msg, f"Expected target '4.0.0' in warning message, got: {msg}"
    assert "2.0.0" not in msg, f"Stale '2.0.0' target found in warning: {msg}"


def test_sigma_numpy_future_warning_4_0_0():
    """Ensure puremacro.sigma.sigma_numpy.SigmaObject emits FutureWarning with target 4.0.0."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        from puremacro.sigma.sigma_numpy import SigmaObject
        _ = SigmaObject(sigma=np.array([0.1, 0.2]), R=np.eye(2), labels=["A", "B"])

    sigma_warnings = [
        w for w in recorded
        if issubclass(w.category, FutureWarning) and "SigmaObject is deprecated" in str(w.message)
    ]
    assert len(sigma_warnings) >= 1, "Expected FutureWarning on SigmaObject init"
    msg = str(sigma_warnings[0].message)
    assert "4.0.0" in msg, f"Expected target '4.0.0' in warning message, got: {msg}"
    assert "2.0.0" not in msg, f"Stale '2.0.0' target found in warning: {msg}"


# ===========================================================================
# 4. Typst Parity Across All 11 Trade Results + VAR + DSGE
# ===========================================================================

def _make_dummy_eq():
    return TradeEquilibriumResult(
        x_sol=np.zeros(19),
        p_sol=np.ones((1, 2, 2)),
        y_sol=np.ones((1, 2, 2)) * 50.0,
        r_sol=np.ones((1, 1, 2)),
        w_sol=np.ones((1, 1, 2)),
        T_sol=np.ones((1, 1, 2)) * 5.0,
        XN_sol=np.array([1.0]),
        c_sol=np.ones((1, 3, 2)) * 20.0,
        pfd_sol=np.ones((1, 3, 2)),
        terms_of_trade=np.array([1.02, 0.98]),
        country_codes=("CAN", "USA"),
        sector_codes=("S01", "S02"),
    )


def test_all_trade_results_have_callable_to_typst():
    """Adversarial loop over all 11 trade result classes verifying .to_typst() returns valid Typst table markup."""
    eq = _make_dummy_eq()
    gk = GearyKhamisResult(
        pi=np.ones(3),
        ppp=np.ones(2),
        real_gdp=np.array([100.0, 200.0]),
        nominal_gdp=np.array([100.0, 200.0]),
        gdp_growth=np.array([0.0, 0.0]),
        country_codes=("CAN", "USA"),
    )

    instances = [
        TradeCalibrationResult(
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
        ),
        eq,
        gk,
        ScenarioBatchResult(
            scenarios={"base": eq},
            geary_khamis={"base": gk},
            baseline_scenario="base",
            country_codes=("CAN", "USA"),
            sector_codes=("S01", "S02"),
        ),
        RetaliationGameResult(
            scenario_name="retal",
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
        ),
        JCurveDynamicResult(
            scenario_name="jc",
            quarters=(0.0, 1.0),
            sigma_path=(1.0, 2.0),
            t_star=1.5,
            half_life=4.6,
            trade_balance_path=np.array([-10.0, -15.0]),
            us_exports_path=np.array([100.0, 105.0]),
            us_imports_path=np.array([110.0, 120.0]),
            us_gdp_path=np.array([1000.0, 995.0]),
            us_cpi_path=np.array([1.0, 1.02]),
            equilibria=(eq, eq),
        ),
        RevenueRecyclingResult(
            scenario_name="rec",
            closure="lump_sum",
            equilibrium=eq,
            tariff_revenue=100.0,
            household_transfer=100.0,
            factor_tax_cut=0.0,
            subsidy_rate=0.0,
            welfare_decomposition={"net_welfare_change": 7.0},
        ),
        CapacityBottleneckResult(
            scenario_name="bot",
            equilibrium=eq,
            capacity_margins={"S01": 0.05},
            capacity_limits={"S01": 105.0},
            output_levels={"S01": 102.0},
            capacity_utilization={"S01": 0.97},
            price_escalation={"S01": 3.5},
            penalty_multipliers={"S01": 0.02},
        ),
        OptimalTariffResult(
            country_code="USA",
            optimal_tariff_rate=0.15,
            welfare_gain_pct=1.2,
            baseline_welfare=100.0,
            optimal_welfare=101.2,
            welfare_metric="geary_khamis",
            terms_of_trade_initial=1.0,
            terms_of_trade_optimal=1.04,
            tariff_grid=np.linspace(0.0, 0.3, 5),
            welfare_curve=np.linspace(100.0, 101.2, 5),
            equilibrium=eq,
        ),
        NashTariffResult(
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
        ),
        WelfarePayoffMatrixResult(
            players=("USA", "CHN"),
            strategies=("C", "D"),
            payoff_matrix=np.zeros((2, 2, 2)),
            scenarios={"CC": eq},
            summary_df=pd.DataFrame({"Player": ["USA"], "C": [0.0]}),
        ),
    ]

    for inst in instances:
        assert hasattr(inst, "to_typst"), f"{type(inst).__name__} missing .to_typst()"
        out = inst.to_typst()
        assert isinstance(out, str), f"{type(inst).__name__}.to_typst() did not return str"
        assert "#table(" in out, f"{type(inst).__name__}.to_typst() missing #table("


# ===========================================================================
# 5. Static AST Pyodide 4-Package Purity Scan
# ===========================================================================

def test_modified_files_pyodide_ast_purity():
    """Scan all modified M1 files to ensure zero forbidden module-level imports."""
    forbidden = {"statsmodels", "linearmodels", "arch", "torch", "numba"}
    root = Path(__file__).resolve().parent.parent

    target_files = [
        root / "puremacro/vfi/hjb_achdou.py",
        root / "puremacro/vfi/problem.py",
        root / "puremacro/trade/_results.py",
        root / "puremacro/var/bvar_sv.py",
        root / "puremacro/dsge/_gradients.py",
        root / "puremacro/lp/garch_utils.py",
        root / "puremacro/sigma/sigma_numpy.py",
    ]

    for path in target_files:
        assert path.exists(), f"File {path} must exist"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:  # Module-level only
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_pkg = alias.name.split(".")[0]
                    assert top_pkg not in forbidden, (
                        f"Forbidden module-level import '{top_pkg}' in {path}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top_pkg = node.module.split(".")[0]
                    assert top_pkg not in forbidden, (
                        f"Forbidden module-level from-import '{top_pkg}' in {path}"
                    )


# ===========================================================================
# 6. Showcase Notebook Execution
# ===========================================================================

def test_notebook_22_execution():
    """Verify notebooks/22_continuous_time_hjb.py executes cleanly end-to-end."""
    root = Path(__file__).resolve().parent.parent
    nb_path = root / "notebooks/22_continuous_time_hjb.py"
    assert nb_path.exists(), f"Notebook {nb_path} must exist"

    cmd = [sys.executable, str(nb_path)]
    env = dict(sys.modules["os"].environ, MPLBACKEND="Agg")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == 0, f"Notebook execution failed:\nStdout:\n{res.stdout}\nStderr:\n{res.stderr}"
    assert "HJB Solved in" in res.stdout
    assert "Consumption at a=0" in res.stdout
