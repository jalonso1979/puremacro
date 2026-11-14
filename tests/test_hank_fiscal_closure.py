"""Tests for explicit fiscal closures in ``puremacro.models.hank_fiscal``."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
import typing

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.models.hank_fiscal import (
    FiscalHANKResult,
    compare_fiscal_closures,
    solve_hank_fiscal,
)
from puremacro.models.hank_sequence_space import (
    solve_hank_sequence_space,
    solve_nonlinear_transition,
)
from puremacro.solvers.diagnostics import AccuracyStatus, ConvergenceStatus


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _ar1_shock(horizon: int, impact: float = 0.01, rho: float = 0.7) -> np.ndarray:
    """Return an AR(1) fiscal spending shock path of length ``horizon``."""
    return impact * (rho ** np.arange(horizon, dtype=float))


def _finite_horizon_ibc_residual(res: FiscalHANKResult) -> float:
    """Independently compute the finite-horizon present-value IBC residual."""
    r = np.asarray(res.r, dtype=float)
    g = np.asarray(res.G, dtype=float)
    t = np.asarray(res.T, dtype=float)
    b = np.asarray(res.B, dtype=float)
    b_init = float(res.steady_state["B_ss"])
    discount = 1.0 / np.cumprod(1.0 + r)
    pv_primary_surplus = float(np.sum(discount * (t - g)))
    pv_terminal_debt = float(discount[-1] * b[-1])
    return float(abs(b_init - (pv_primary_surplus + pv_terminal_debt)))


@pytest.mark.parametrize("tax_type", ["proportional", "lump_sum"])
def test_balanced_budget_exact_clearing_and_identities(tax_type: str) -> None:
    """Balanced-budget closure satisfies goods, fiscal, and asset-market clearing."""
    horizon = 80
    dG = _ar1_shock(horizon, impact=0.01, rho=0.7)
    res = solve_hank_fiscal(
        shock_seq=dG,
        closure="balanced_budget",
        tax_type=tax_type,
        horizon=horizon,
        n_a=100,
        a_max=100.0,
        tol=1e-10,
    )

    assert isinstance(res, FiscalHANKResult)
    assert res.converged is True
    assert isinstance(res.convergence_status, ConvergenceStatus)
    assert res.convergence_status.converged is True
    assert isinstance(res.accuracy_status, AccuracyStatus)
    assert res.accuracy_status.is_accurate is True
    assert res.support_status.is_supported is True
    assert res.solver_diagnostics.overall_status == "PASS"

    # Reported audit metrics.
    assert res.goods_market_residual < 1e-9
    assert res.gov_budget_residual < 1e-12
    assert res.gov_intertemporal_residual < 1e-12
    assert res.asset_market_residual < 1e-8
    assert res.terminal_debt_gap < 1e-12

    # Independent verification of every period's identities.
    goods_diff = float(np.max(np.abs(res.Y - res.C - res.G)))
    assert goods_diff == pytest.approx(res.goods_market_residual, abs=1e-14)

    b_ss = float(res.steady_state["B_ss"])
    b_prev = np.concatenate([[b_ss], res.B[:-1]])
    gov_flow_diff = float(np.max(np.abs(res.B - ((1.0 + res.r) * b_prev + res.G - res.T))))
    assert gov_flow_diff == pytest.approx(res.gov_budget_residual, abs=1e-14)

    walras_diff = float(np.max(np.abs(res.A - res.B)))
    assert walras_diff == pytest.approx(res.asset_market_residual, abs=1e-14)
    assert _finite_horizon_ibc_residual(res) < 1e-12

    # Balanced budget keeps government debt pegged to steady state at all t.
    assert np.allclose(res.B, b_ss, atol=1e-12, rtol=0.0)
    expected_taxes = res.G + res.r * b_ss
    assert np.allclose(res.T, expected_taxes, atol=1e-11, rtol=0.0)

    if tax_type == "proportional":
        assert np.allclose(res.tau, res.T / res.Y, atol=1e-12, rtol=0.0)
    else:
        tau_ss = float(res.steady_state["tau_ss"])
        assert np.allclose(res.tau, tau_ss, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("tax_type", ["proportional", "lump_sum"])
def test_debt_rule_clearing_ibc_and_terminal_stability(tax_type: str) -> None:
    """Stabilizing debt rule satisfies Walras's Law, IBC, and debt stationarity."""
    horizon = 160
    phi_B = 0.12
    phi_G = 0.20
    dG = _ar1_shock(horizon, impact=0.01, rho=0.65)
    res = solve_hank_fiscal(
        shock_seq=dG,
        closure="debt_rule",
        tax_type=tax_type,
        horizon=horizon,
        phi_B=phi_B,
        phi_G=phi_G,
        n_a=100,
        a_max=100.0,
        tol=1e-10,
    )

    assert res.converged is True
    assert res.convergence_status.converged is True
    assert res.accuracy_status.is_accurate is True
    assert res.support_status.is_supported is True
    assert res.solver_diagnostics.overall_status == "PASS"

    assert res.goods_market_residual < 1e-9
    assert res.gov_budget_residual < 1e-12
    assert res.gov_intertemporal_residual < 1e-12
    assert res.asset_market_residual < 1e-8
    assert _finite_horizon_ibc_residual(res) < 1e-12

    # Verify the exact linear feedback fiscal rule T_t = T_ss + phi_B dB_{t-1} + phi_G dG_t.
    t_ss = float(res.steady_state["T_ss"])
    dB_prev = np.concatenate([[0.0], res.dB[:-1]])
    expected_T = t_ss + phi_B * dB_prev + phi_G * res.dG
    assert np.allclose(res.T, expected_T, atol=1e-12, rtol=0.0)

    # Impact deficit increases debt since phi_G < 1, then decays asymptotically toward B_ss.
    assert res.dB[0] > 0.0
    peak_dB = float(np.max(np.abs(res.dB)))
    assert peak_dB > 1e-3
    assert res.terminal_debt_gap < 1e-4 * peak_dB

    # Households hold exactly the newly issued government debt at every date.
    assert np.allclose(res.A, res.B, atol=1e-8, rtol=0.0)


def test_zero_shock_steady_state_positive_g_ss_and_monetary_coexistence() -> None:
    """Zero shock sits on steady state; g_ss > 0 and monetary shocks clear all markets."""
    horizon = 80
    zero_res = solve_hank_fiscal(
        shock_seq=np.zeros(horizon),
        closure="balanced_budget",
        tax_type="proportional",
        horizon=horizon,
        n_a=100,
        a_max=100.0,
        tol=1e-10,
    )
    assert zero_res.converged is True
    assert zero_res.iterations == 0
    assert float(np.max(np.abs(zero_res.dY))) < 1e-12
    assert float(np.max(np.abs(zero_res.dC))) < 1e-12
    assert zero_res.asset_market_residual < 1e-10
    assert zero_res.goods_market_residual < 1e-11

    # Positive steady-state spending g_ss > 0 plus joint fiscal and monetary shocks.
    g_shock = _ar1_shock(horizon, impact=0.008, rho=0.6)
    m_shock = _ar1_shock(horizon, impact=-0.002, rho=0.5)
    joint_res = solve_hank_fiscal(
        shock_seq=g_shock,
        monetary_shock_seq=m_shock,
        closure="debt_rule",
        tax_type="lump_sum",
        horizon=horizon,
        phi_B=0.10,
        phi_G=0.10,
        g_ss=0.05,
        n_a=100,
        a_max=100.0,
        tol=1e-10,
    )
    assert joint_res.converged is True
    assert joint_res.steady_state["G_ss"] == pytest.approx(0.05, abs=1e-14)
    assert joint_res.steady_state["T_ss"] == pytest.approx(
        0.05 + joint_res.steady_state["r_ss"] * joint_res.steady_state["B_ss"],
        abs=1e-13,
    )
    assert joint_res.goods_market_residual < 1e-9
    assert joint_res.gov_budget_residual < 1e-12
    assert joint_res.gov_intertemporal_residual < 1e-12
    assert joint_res.asset_market_residual < 1e-8


def test_financed_closures_differ_from_unfinanced_g_shock() -> None:
    """Financed fiscal closures eliminate the unbacked-wealth inflation of shock_var='G'."""
    horizon = 100
    dG = _ar1_shock(horizon, impact=0.01, rho=0.7)

    # Same household parameters on a clean grid with n_a=100, a_max=100.0.
    ss = solve_hank_sequence_space(T=horizon, n_a=100, a_max=100.0)
    unfinanced = solve_nonlinear_transition(
        ss, dG, shock_var="G", max_iter=100, tol=1e-10
    )
    balanced_prop = solve_hank_fiscal(
        ss_model=ss,
        shock_seq=dG,
        closure="balanced_budget",
        tax_type="proportional",
        horizon=horizon,
        tol=1e-9,
    )
    balanced_ls = solve_hank_fiscal(
        ss_model=ss,
        shock_seq=dG,
        closure="balanced_budget",
        tax_type="lump_sum",
        horizon=horizon,
        tol=1e-9,
    )
    debt_prop = solve_hank_fiscal(
        ss_model=ss,
        shock_seq=dG,
        closure="debt_rule",
        tax_type="proportional",
        horizon=horizon,
        phi_B=0.05,
        phi_G=0.0,
        tol=1e-9,
    )

    assert unfinanced.converged is True
    unfinanced_dY = np.asarray(unfinanced.nonlinear_path, dtype=float)
    unfinanced_r = np.concatenate(
        [[float(ss.r_ss)], float(ss.r_ss) + np.asarray(unfinanced.irf_rate_nonlinear[:-1], dtype=float)]
    )
    unfinanced_cum = float(np.sum(unfinanced_dY) / np.sum(dG))

    # In unfinanced shock_var="G", household taxes only adjust for steady-state debt service
    # rather than dG_t itself; accumulating the true bond law under passive steady-state taxes
    # yields an unbacked debt buildup not absorbed by household asset accumulation.
    b_unbacked = np.empty(horizon, dtype=float)
    b_curr = float(ss.government_debt)
    t_ss = float(ss.r_ss) * b_curr
    for t in range(horizon):
        b_curr = (1.0 + unfinanced_r[t]) * b_curr + dG[t] - t_ss
        b_unbacked[t] = b_curr
    assert abs(b_unbacked[-1] - float(ss.government_debt)) > 0.03

    # Financed spending crowds out private consumption via taxes / real interest rates,
    # keeping financed multipliers far below the unfinanced "free-lunch" multiplier.
    assert balanced_prop.cumulative_multiplier < 0.70
    assert balanced_ls.cumulative_multiplier < 0.70
    assert debt_prop.cumulative_multiplier < 0.85
    assert unfinanced_cum - balanced_prop.cumulative_multiplier > 0.50
    assert unfinanced_cum - debt_prop.cumulative_multiplier > 0.40

    # Incidence differs between proportional income taxation and uniform lump-sum taxes.
    assert abs(balanced_prop.impact_multiplier - balanced_ls.impact_multiplier) > 1e-3
    assert (
        abs(balanced_prop.cumulative_multiplier - balanced_ls.cumulative_multiplier)
        > 1e-3
    )


def test_ricardian_failure_and_debt_rule_parameter_comparative_statics() -> None:
    """HANK violates Ricardian equivalence: deficit financing boosts short-run demand."""
    horizon = 120
    dG = _ar1_shock(horizon, impact=0.01, rho=0.6)
    ss = solve_hank_sequence_space(T=horizon, n_a=100, a_max=100.0)

    balanced_ls = solve_hank_fiscal(
        ss_model=ss,
        shock_seq=dG,
        closure="balanced_budget",
        tax_type="lump_sum",
        horizon=horizon,
        tol=1e-9,
    )
    deficit_low = solve_hank_fiscal(
        ss_model=ss,
        shock_seq=dG,
        closure="debt_rule",
        tax_type="lump_sum",
        horizon=horizon,
        phi_B=0.05,
        phi_G=0.0,
        tol=1e-9,
    )
    deficit_mid = solve_hank_fiscal(
        ss_model=ss,
        shock_seq=dG,
        closure="debt_rule",
        tax_type="lump_sum",
        horizon=horizon,
        phi_B=0.05,
        phi_G=0.5,
        tol=1e-9,
    )
    deficit_high_phi_g = solve_hank_fiscal(
        ss_model=ss,
        shock_seq=dG,
        closure="debt_rule",
        tax_type="lump_sum",
        horizon=horizon,
        phi_B=0.15,
        phi_G=0.95,
        tol=1e-9,
    )

    # Even with lump-sum taxes, borrowing instead of taxing immediately raises impact C and Y.
    assert deficit_low.dC[0] > balanced_ls.dC[0] + 1e-4
    assert deficit_low.impact_multiplier > balanced_ls.impact_multiplier + 1e-4
    assert deficit_low.cumulative_multiplier > balanced_ls.cumulative_multiplier + 1e-3

    # Raising upfront tax financing phi_G lowers peak debt and reduces deficit stimulus.
    assert float(np.max(np.abs(deficit_low.dB))) > float(np.max(np.abs(deficit_mid.dB)))
    assert float(np.max(np.abs(deficit_mid.dB))) > float(np.max(np.abs(deficit_high_phi_g.dB)))
    assert deficit_low.dC[0] > deficit_mid.dC[0] > deficit_high_phi_g.dC[0]


@pytest.mark.parametrize("bad_phi_B", [-0.02, 0.0, 0.005, 2.01])
def test_explosive_debt_rule_raises_by_default(bad_phi_B: float) -> None:
    """Rules violating r_ss < phi_B < 2 + r_ss raise ValueError unless explicitly allowed."""
    horizon = 40
    with pytest.raises(ValueError, match="phi_B"):
        solve_hank_fiscal(
            shock_seq=_ar1_shock(horizon),
            closure="debt_rule",
            tax_type="proportional",
            horizon=horizon,
            r_ss=0.005,
            phi_B=bad_phi_B,
            phi_G=0.0,
        )


def test_explosive_debt_rule_flagged_non_convergent_when_allowed() -> None:
    """With allow_unstable_rule=True, phi_B <= r_ss emits warning and marks FAILED."""
    horizon = 60
    dG = _ar1_shock(horizon, impact=0.01, rho=0.6)
    with pytest.warns(RuntimeWarning, match="stationary stability"):
        res = solve_hank_fiscal(
            shock_seq=dG,
            closure="debt_rule",
            tax_type="lump_sum",
            horizon=horizon,
            r_ss=0.005,
            phi_B=0.002,
            phi_G=0.0,
            allow_unstable_rule=True,
        )

    assert res.converged is False
    assert res.convergence_status.converged is False
    assert res.support_status.is_supported is False
    assert res.support_status.regime == "non_stationary_explosive"
    assert res.solver_diagnostics.overall_status == "FAILED"
    assert abs(res.dB[-1]) > abs(res.dB[0])


def test_boundary_asset_truncation_is_surfaced_not_hidden() -> None:
    """Coarse asset grid with upper-boundary mass fails audit honestly."""
    horizon = 40
    coarse_ss = solve_hank_sequence_space(T=horizon, n_a=50, a_max=30.0)
    with pytest.warns(RuntimeWarning, match="Upper asset grid truncation"):
        res = solve_hank_fiscal(
            ss_model=coarse_ss,
            shock_seq=_ar1_shock(horizon, impact=0.01, rho=0.6),
            closure="balanced_budget",
            tax_type="proportional",
            horizon=horizon,
        )

    assert res.support_status.is_supported is False
    assert res.support_status.regime == "boundary_asset_truncation"
    assert res.solver_diagnostics.overall_status != "PASS"
    assert res.asset_market_residual > 1e-6


def test_nonlinear_asymmetry_for_large_fiscal_shocks() -> None:
    """Large fiscal expansions vs contractions exhibit nonlinear asymmetry."""
    horizon = 80
    ss = solve_hank_sequence_space(T=horizon, n_a=100, a_max=100.0)
    pos_shock = _ar1_shock(horizon, impact=0.05, rho=0.65)
    neg_shock = -pos_shock

    pos_res = solve_hank_fiscal(
        ss_model=ss,
        shock_seq=pos_shock,
        closure="balanced_budget",
        tax_type="lump_sum",
        horizon=horizon,
        tol=1e-9,
    )
    neg_res = solve_hank_fiscal(
        ss_model=ss,
        shock_seq=neg_shock,
        closure="balanced_budget",
        tax_type="lump_sum",
        horizon=horizon,
        tol=1e-9,
    )

    assert pos_res.converged is True
    assert neg_res.converged is True
    assert pos_res.support_status.is_supported is True
    assert neg_res.support_status.is_supported is True
    assert pos_res.goods_market_residual < 1e-9
    assert neg_res.goods_market_residual < 1e-9
    assert pos_res.asset_market_residual < 1e-8
    assert neg_res.asset_market_residual < 1e-8

    asymmetry = float(np.max(np.abs(pos_res.dY + neg_res.dY)))
    assert asymmetry > 1e-5


def test_compare_fiscal_closures_and_presentation_outputs() -> None:
    """Comparison helper, tabular exporters, plots, and validation behave cleanly."""
    horizon = 60
    dG = _ar1_shock(horizon, impact=0.01, rho=0.7)
    ss = solve_hank_sequence_space(T=horizon, n_a=100, a_max=100.0)

    df = compare_fiscal_closures(
        ss_model=ss,
        shock_seq=dG,
        horizon=horizon,
        phi_B=0.08,
        phi_G=0.0,
        include_unfinanced=True,
        tol=1e-9,
    )
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == [
        "balanced_budget_proportional",
        "balanced_budget_lump_sum",
        "debt_rule_proportional",
        "debt_rule_lump_sum",
        "unfinanced_goods_only",
    ]
    financed = df.loc[df["closure"] != "unfinanced_goods_only"]
    assert (financed["status"] == "PASS").all()
    assert (financed["goods_residual"] < 1e-8).all()
    assert (financed["gov_budget_residual"] < 1e-11).all()
    assert (financed["asset_residual"] < 1e-7).all()

    res = solve_hank_fiscal(
        ss_model=ss,
        shock_seq=dG,
        closure="debt_rule",
        tax_type="proportional",
        horizon=horizon,
        phi_B=0.08,
        tol=1e-9,
    )
    summary_text = res.summary()
    assert "Financed Sequence-Space HANK Fiscal Transition" in summary_text
    assert "Asset Residual ||A-B||_inf" in summary_text

    frame = res.to_frame()
    assert len(frame) == horizon
    for col in (
        "Y",
        "C",
        "G",
        "T",
        "B",
        "A",
        "r",
        "w",
        "Goods_Residual",
        "Gov_Budget_Residual",
        "Asset_Residual",
    ):
        assert col in frame.columns

    md = res.to_markdown()
    tex = res.to_latex()
    typ = res.to_typst()
    assert "Y" in md
    assert "\\begin" in tex
    assert "#table" in typ

    for style in ("publication", "grayscale"):
        fig = res.plot(style=style)
        assert len(fig.axes) == 4
        plt.close(fig)

    with pytest.raises(ValueError, match="closure"):
        solve_hank_fiscal(shock_seq=dG, closure="monetized", horizon=horizon)
    with pytest.raises(ValueError, match="tax_type"):
        solve_hank_fiscal(shock_seq=dG, tax_type="capital", horizon=horizon)
    with pytest.raises(ValueError, match="horizon"):
        solve_hank_fiscal(horizon=0)
    with pytest.raises(ValueError, match="1-dimensional"):
        solve_hank_fiscal(shock_seq=np.ones((2, 2)), horizon=horizon)
    with pytest.raises(ValueError, match="non-empty"):
        solve_hank_fiscal(shock_seq=[], horizon=horizon)
    with pytest.raises(ValueError, match="g_ss"):
        solve_hank_fiscal(shock_seq=dG, g_ss=1.0, horizon=horizon)


@pytest.mark.mechanism_control
def test_pyodide_import_blocker_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control proving the MetaPathFinder blocker actually intercepts imports."""
    import importlib
    import importlib.abc

    blocked = ("pdfplumber", "statsmodels")

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):  # type: ignore[override]
            if fullname.split(".")[0] in blocked:
                raise ImportError(f"blocked in pyodide: {fullname}")
            return None

    monkeypatch.delitem(sys.modules, "pdfplumber", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Blocker(), *sys.meta_path])
    with pytest.raises(ImportError, match="blocked in pyodide"):
        importlib.import_module("pdfplumber")


def test_pyodide_import_invariant_and_type_hints() -> None:
    """Module resolves type hints cleanly and respects Pyodide top-level imports."""
    hints = typing.get_type_hints(solve_hank_fiscal)
    assert hints["return"] is FiscalHANKResult

    src_path = _REPO_ROOT / "puremacro" / "models" / "hank_fiscal.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    allowed_top = (set(sys.stdlib_module_names) - {"sqlite3", "ssl"}) | {
        "__future__",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "requests",
    }
    top_roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_roots.add(node.module.split(".")[0])
    disallowed = sorted(top_roots - allowed_top)
    assert disallowed == [], f"Non-Pyodide top-level imports: {disallowed}"

    probe = "\n".join(
        [
            "import importlib.abc, sys",
            "BLOCKED = ('statsmodels', 'linearmodels', 'arch', 'bs4', 'pdfplumber', "
            "'pypdf', 'pyarrow', 'openpyxl', 'numba', 'torch', 'jax', 'sqlite3')",
            "class Blocker(importlib.abc.MetaPathFinder):",
            "    def find_spec(self, fullname, path, target=None):",
            "        if fullname.split('.')[0] in BLOCKED:",
            "            raise ImportError(f'blocked in pyodide: {fullname}')",
            "sys.meta_path.insert(0, Blocker())",
            "import puremacro.models.hank_fiscal as hf",
            "res = hf.solve_hank_fiscal(horizon=20, n_a=60, a_max=100.0)",
            "assert res.converged is True",
            "loaded = [m for m in sys.modules if m.split('.')[0] in BLOCKED]",
            "assert not loaded, loaded",
        ]
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT)
    subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
