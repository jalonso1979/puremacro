"""Presentation contract tests for puremacro DSGE Tier 1 result objects.

Tests the full 6-method presentation contract across all 5 DSGE result dataclasses:
1. EigenvalueTable
2. ModelDiagnosticsResult
3. IdentificationResult
4. OSRResult
5. PolicyResult

Presentation methods verified for each class:
- .to_frame() -> pandas.DataFrame
- .summary() -> str
- .plot() -> matplotlib Axes / Figure (with valid matplotlib.figure.Figure)
- .to_markdown() -> GitHub-flavored Markdown table
- .to_latex() -> LaTeX tabular
- .to_typst() -> Typst #table(...)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from puremacro.dsge import (
    DiagnosticFinding,
    EigenvalueTable,
    IdentificationResult,
    ModelDiagnosticsResult,
    OSRResult,
    PolicyResult,
    build,
    discretionary_policy,
    identification,
    load_mod,
    lq_commitment,
    model_diagnostics,
    osr,
)


# ===========================================================================
# Fixtures / Helper model constructors
# ===========================================================================

@pytest.fixture(scope="module")
def canonical_growth_model():
    """Canonical Hansen (1985) style neoclassical growth model."""
    mod_str = """
    var c k y a;
    varexo eps_a;
    parameters beta alpha delta rho;
    beta = 0.99;
    alpha = 0.36;
    delta = 0.025;
    rho = 0.95;

    model;
      1/c = beta * (1/c(+1)) * (alpha * y(+1)/k + 1 - delta);
      y = a * k(-1)^alpha;
      y = c + k - (1 - delta) * k(-1);
      log(a) = rho * log(a(-1)) + eps_a;
    end;

    initval;
      k = 38.0;
      c = 2.75;
      y = 3.7;
      a = 1.0;
    end;

    shocks;
      var eps_a; stderr 0.01;
    end;
    """
    return load_mod(mod_str)


@pytest.fixture(scope="module")
def simple_nk_model():
    """Simple 3-equation New Keynesian model for policy and OSR tests."""
    mod_str = """
    var y pi i u r_nat;
    varexo eps_r eps_u;
    parameters sigma beta kappa phi_pi phi_y rho_i rho_u;
    sigma = 1.0;
    beta = 0.99;
    kappa = 0.15;
    phi_pi = 1.5;
    phi_y = 0.5;
    rho_i = 0.0;
    rho_u = 0.8;

    model;
      y = y(+1) - (1/sigma) * (i - pi(+1) - r_nat);
      pi = beta * pi(+1) + kappa * y + u;
      i = phi_pi * pi + phi_y * y + eps_r;
      u = rho_u * u(-1) + eps_u;
      r_nat = 0.0;
    end;

    initval;
      y = 0; pi = 0; i = 0; u = 0; r_nat = 0;
    end;

    shocks;
      var eps_r; stderr 0.25;
      var eps_u; stderr 0.50;
    end;
    """
    return load_mod(mod_str)


def _assert_figure_valid(out, ax_passed=None):
    """Helper verifying that plot output produces a valid Figure."""
    if ax_passed is not None:
        assert out is ax_passed
        fig = ax_passed.figure
    else:
        fig = out.figure if hasattr(out, "figure") else out
    assert isinstance(fig, Figure)
    plt.close(fig)


# ===========================================================================
# 1. EigenvalueTable Tests
# ===========================================================================

def test_eigenvalue_table_presentation_from_model(canonical_growth_model):
    """Test all 6 presentation methods on EigenvalueTable from model.check()."""
    table = canonical_growth_model.check()
    assert isinstance(table, EigenvalueTable)

    # 1. to_frame()
    df = table.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["modulus", "real", "imag", "is_explosive", "is_unit_root"]
    assert len(df) == len(table.eigenvalues)

    # 2. summary()
    s = table.summary()
    assert isinstance(s, str)
    assert "EIGENVALUES & BLANCHARD-KAHN DIAGNOSTICS" in s
    assert "Determinacy" in s
    assert "EIGENVALUE SPECTRUM" in s

    # 3. plot()
    # ax=None
    out = table.plot()
    _assert_figure_valid(out)

    # ax provided
    fig, ax = plt.subplots(figsize=(6, 6))
    out_ax = table.plot(ax=ax)
    _assert_figure_valid(out_ax, ax_passed=ax)

    # 4. to_markdown()
    md = table.to_markdown()
    assert isinstance(md, str)
    assert "|" in md
    assert "modulus" in md

    # 5. to_latex()
    tex = table.to_latex()
    assert isinstance(tex, str)
    assert r"\begin{tabular}" in tex
    assert r"\end{tabular}" in tex

    # 6. to_typst()
    typ = table.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ


def test_eigenvalue_table_offending_loadings_presentation():
    """Test EigenvalueTable presentation when Blanchard-Kahn determinacy fails with loadings."""
    roots = np.array([0.5, 0.95, 1.05 + 0.1j, 1.05 - 0.1j, 2.5])
    mod = np.abs(roots)
    re = np.real(roots)
    im = np.imag(roots)
    exp = mod >= 1.0 + 1e-6
    unit = np.abs(mod - 1.0) <= 1e-5

    table = EigenvalueTable(
        eigenvalues=roots,
        modulus=mod,
        real=re,
        imag=im,
        is_explosive=exp,
        is_unit_root=unit,
        n_explosive=int(np.sum(exp)),
        n_forward=1,  # 3 explosive vs 1 forward -> indeterminacy / failure
        is_determinate=False,
        bk_status="Blanchard-Kahn order condition failed: 3 explosive eigenvalues > 1 forward variables.",
        loadings={2: {"c": 0.82, "k": 0.55}, 4: {"k": 0.95, "y": 0.31}},
        variables=("c", "k", "y", "a"),
    )

    assert not table.is_determinate
    assert not table.bk_satisfied
    assert table.n_stable == 2
    assert table.offending_loadings is not None

    # Frame
    df = table.to_frame()
    assert len(df) == 5
    assert df.loc["root_3", "is_explosive"]

    # Summary
    s = table.summary()
    assert "DETERMINACY FAILED" in s
    assert "OFFENDING ROOT VARIABLE LOADINGS" in s
    assert "Root #3" in s
    assert "c: 0.8200" in s

    # Plot
    fig, ax = plt.subplots()
    out_ax = table.plot(ax=ax)
    _assert_figure_valid(out_ax, ax_passed=ax)

    # Tables
    assert "|" in table.to_markdown()
    assert r"\begin{tabular}" in table.to_latex()
    assert "#table(" in table.to_typst()


def test_eigenvalue_table_degenerate_empty():
    """Test EigenvalueTable on degenerate empty root spectrum."""
    table = EigenvalueTable(
        eigenvalues=np.array([], dtype=complex),
        modulus=np.array([], dtype=float),
        real=np.array([], dtype=float),
        imag=np.array([], dtype=float),
        is_explosive=np.array([], dtype=bool),
        is_unit_root=np.array([], dtype=bool),
        n_explosive=0,
        n_forward=0,
        is_determinate=True,
        bk_status="Empty spectrum",
        loadings=None,
        variables=(),
    )
    df = table.to_frame()
    assert len(df) == 0
    assert "EIGENVALUES" in table.summary()
    out = table.plot()
    _assert_figure_valid(out)
    assert "|" in table.to_markdown()
    assert r"\begin{tabular}" in table.to_latex()
    assert "#table(" in table.to_typst()


# ===========================================================================
# 2. ModelDiagnosticsResult Tests
# ===========================================================================

def test_model_diagnostics_result_presentation_from_model(canonical_growth_model):
    """Test all 6 presentation methods on ModelDiagnosticsResult from model_diagnostics()."""
    diag = canonical_growth_model.model_diagnostics()
    assert isinstance(diag, ModelDiagnosticsResult)

    # 1. to_frame()
    df = diag.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert "category" in df.columns
    assert "severity" in df.columns
    assert "message" in df.columns

    # 2. summary()
    s = diag.summary()
    assert isinstance(s, str)
    assert "DSGE MODEL DIAGNOSTICS" in s
    assert "Static Jacobian rank" in s
    assert "Dynamic pencil regular" in s

    # 3. plot()
    out = diag.plot()
    _assert_figure_valid(out)

    fig, ax = plt.subplots()
    out_ax = diag.plot(ax=ax)
    _assert_figure_valid(out_ax, ax_passed=ax)

    # 4. to_markdown()
    md = diag.to_markdown()
    assert isinstance(md, str)
    assert "|" in md

    # 5. to_latex()
    tex = diag.to_latex()
    assert isinstance(tex, str)
    assert r"\begin{tabular}" in tex

    # 6. to_typst()
    typ = diag.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ


def test_model_diagnostics_result_with_findings_and_collinearity():
    """Test ModelDiagnosticsResult presentation when findings, collinearity, and unused items exist."""
    findings = (
        DiagnosticFinding(category="static_rank", severity="error", message="Static Jacobian is rank-deficient by 1."),
        DiagnosticFinding(category="incidence", severity="warning", message="Variable 'z' appears in zero equations."),
    )
    mat = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
    ], dtype=bool)

    diag = ModelDiagnosticsResult(
        passed=False,
        findings=findings,
        static_rank=2,
        static_n_vars=3,
        collinear_equations=("eq1 + eq2 - eq3 = 0",),
        collinear_variables=("x1 - x2 = 0",),
        unused_variables=("z",),
        unused_equations=("eq4",),
        is_pencil_regular=False,
        stochastic_singularity=True,
        unit_roots=(0,),
        incidence_matrix=mat,
        eval_points=5,
        variable_names=("x1", "x2", "z"),
        equation_names=("eq1", "eq2", "eq3"),
    )

    assert not diag.passed
    assert diag.rank_deficient
    assert diag.n_vars == 3
    assert not diag.pencil_regular
    assert diag.unit_roots_detected == 1
    assert diag.redundant_equations == ("eq4",)

    # Frame
    df = diag.to_frame()
    assert len(df) == 2
    assert "static_rank" in df["category"].values

    # Summary
    s = diag.summary()
    assert "FAILED" in s
    assert "RANK DEFICIENT" in s
    assert "Collinear equations" in s
    assert "Unused variables       : 1 (z)" in s

    # Plot
    fig, ax = plt.subplots()
    out_ax = diag.plot(ax=ax)
    _assert_figure_valid(out_ax, ax_passed=ax)

    # Tables
    assert "|" in diag.to_markdown()
    assert r"\begin{tabular}" in diag.to_latex()
    assert "#table(" in diag.to_typst()


def test_model_diagnostics_result_empty_and_none_matrix():
    """Test ModelDiagnosticsResult edge cases: no findings and None incidence matrix."""
    diag = ModelDiagnosticsResult(
        passed=True,
        findings=(),
        static_rank=4,
        static_n_vars=4,
        collinear_equations=(),
        collinear_variables=(),
        incidence_matrix=None,
    )
    df = diag.to_frame()
    assert len(df) == 1
    assert df.loc[0, "severity"] == "info"

    s = diag.summary()
    assert "PASSED (0 errors, 0 warnings)" in s

    out = diag.plot()
    _assert_figure_valid(out)


# ===========================================================================
# 3. IdentificationResult Tests
# ===========================================================================

def test_identification_result_presentation_from_model(canonical_growth_model):
    """Test all 6 presentation methods on IdentificationResult from identification()."""
    ident = canonical_growth_model.identification(varobs=["y", "c"])
    assert isinstance(ident, IdentificationResult)

    # 1. to_frame()
    df = ident.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert "j1_collinearity" in df.columns
    assert "j2_collinearity" in df.columns
    assert "sensitivity" in df.columns
    assert "strength" in df.columns
    assert "identified" in df.columns
    assert len(df) == len(ident.param_names)

    # 2. summary()
    s = ident.summary()
    assert isinstance(s, str)
    assert "PARAMETER IDENTIFICATION ANALYSIS" in s
    assert "J1 (Solution) rank" in s
    assert "J2 (Moments) rank" in s
    assert "PARAMETER IDENTIFICATION SUMMARY" in s

    # 3. plot()
    out = ident.plot()
    _assert_figure_valid(out)

    fig, ax = plt.subplots()
    out_ax = ident.plot(ax=ax)
    _assert_figure_valid(out_ax, ax_passed=ax)

    # 4. to_markdown()
    md = ident.to_markdown()
    assert isinstance(md, str)
    assert "|" in md

    # 5. to_latex()
    tex = ident.to_latex()
    assert isinstance(tex, str)
    assert r"\begin{tabular}" in tex

    # 6. to_typst()
    typ = ident.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ


def test_identification_result_unidentified_model():
    """Test IdentificationResult presentation for an unidentified model with null-space combinations."""
    p_names = ("alpha", "beta", "gamma")
    j1_coll = pd.DataFrame({"r2": [0.15, 0.9999, 0.9999]}, index=list(p_names))
    j2_coll = pd.DataFrame({"r2": [0.20, 1.0000, 1.0000]}, index=list(p_names))
    st_df = pd.DataFrame({
        "sensitivity": [0.45, 0.00, 0.00],
        "strength": [0.35, 0.00, 0.00],
        "normalized_strength": [1.00, 0.00, 0.00],
    }, index=list(p_names))

    ident = IdentificationResult(
        is_identified=False,
        j1_rank=2,
        j1_n_params=3,
        j1_null_space=np.array([[0.0, 0.7071, -0.7071]]),
        j1_null_combinations=("0.7071 * beta - 0.7071 * gamma = 0",),
        j1_collinearity=j1_coll,
        j2_rank=2,
        j2_n_params=3,
        j2_null_space=np.array([[0.0, 0.7071, -0.7071]]),
        j2_null_combinations=("0.7071 * beta - 0.7071 * gamma = 0",),
        j2_collinearity=j2_coll,
        strength=st_df,
        param_names=p_names,
        varobs=("y",),
        lags=2,
    )

    assert not ident.is_identified
    assert ident.rank_deficient
    assert ident.j1_rank_deficient
    assert ident.j2_rank_deficient
    assert ident.n_params == 3

    # Frame
    df = ident.to_frame()
    assert df.loc["alpha", "identified"]
    assert not df.loc["beta", "identified"]

    # Summary
    s = ident.summary()
    assert "UNIDENTIFIED (RANK DEFICIENT)" in s
    assert "J1 NULL SPACE PARAMETER COMBINATIONS" in s
    assert "0.7071 * beta - 0.7071 * gamma = 0" in s

    # Plot
    fig, ax = plt.subplots()
    out_ax = ident.plot(ax=ax)
    _assert_figure_valid(out_ax, ax_passed=ax)

    # Tables
    assert "|" in ident.to_markdown()
    assert r"\begin{tabular}" in ident.to_latex()
    assert "#table(" in ident.to_typst()


def test_identification_result_empty_parameters():
    """Test IdentificationResult edge case with empty parameter tuple."""
    ident = IdentificationResult(
        is_identified=True,
        j1_rank=0,
        j1_n_params=0,
        j1_null_space=np.zeros((0, 0)),
        j1_null_combinations=(),
        j1_collinearity=pd.DataFrame(columns=["r2"]),
        j2_rank=0,
        j2_n_params=0,
        j2_null_space=np.zeros((0, 0)),
        j2_null_combinations=(),
        j2_collinearity=pd.DataFrame(columns=["r2"]),
        strength=pd.DataFrame(columns=["sensitivity", "strength", "normalized_strength"]),
        param_names=(),
        varobs=(),
    )
    df = ident.to_frame()
    assert len(df) == 0
    s = ident.summary()
    assert "PARAMETER IDENTIFICATION ANALYSIS" in s
    out = ident.plot()
    _assert_figure_valid(out)


# ===========================================================================
# 4. OSRResult Tests
# ===========================================================================

def test_osr_result_presentation_from_model(simple_nk_model):
    """Test all 6 presentation methods on OSRResult from osr()."""
    res = simple_nk_model.osr(
        rule_params=["phi_pi", "phi_y"],
        target_vars=["pi", "y"],
        weights={"pi": 1.0, "y": 0.5},
        maxiter=30,
    )
    assert isinstance(res, OSRResult)

    # 1. to_frame()
    df = res.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert "var_initial" in df.columns
    assert "var_optimal" in df.columns
    assert "weight" in df.columns
    assert len(df) == 2

    # 2. summary()
    s = res.summary()
    assert isinstance(s, str)
    assert "OPTIMAL SIMPLE RULES (OSR) OPTIMIZATION" in s
    assert "RULE PARAMETERS" in s
    assert "phi_pi" in s
    assert "TARGET VARIABLE VARIANCES" in s

    # 3. plot()
    out = res.plot()
    _assert_figure_valid(out)

    fig, ax = plt.subplots()
    out_ax = res.plot(ax=ax)
    _assert_figure_valid(out_ax, ax_passed=ax)

    # 4. to_markdown()
    md = res.to_markdown()
    assert isinstance(md, str)
    assert "|" in md

    # 5. to_latex()
    tex = res.to_latex()
    assert isinstance(tex, str)
    assert r"\begin{tabular}" in tex

    # 6. to_typst()
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ


def test_osr_result_synthetic_and_edge_cases():
    """Test OSRResult aliases, edge cases, and custom formatting."""
    var_table = pd.DataFrame({
        "var_initial": [0.045, 0.012],
        "var_optimal": [0.015, 0.008],
        "weight": [1.0, 0.5],
        "loss_initial": [0.045, 0.006],
        "loss_optimal": [0.015, 0.004],
    }, index=["pi", "y"])

    res = OSRResult(
        optimal_params={"phi_pi": 2.1, "phi_y": 0.3},
        initial_params={"phi_pi": 1.5, "phi_y": 0.5},
        loss_opt=0.019,
        loss_initial=0.051,
        rule_params=("phi_pi", "phi_y"),
        target_vars=("pi", "y"),
        weights={"pi": 1.0, "y": 0.5},
        variance_table=var_table,
        converged=True,
        message="Optimization terminated successfully.",
        n_evaluations=42,
    )

    assert res.loss_init == res.loss_initial
    assert res.loss_calib == res.loss_initial

    df = res.to_frame()
    assert df.loc["pi", "var_optimal"] == 0.015

    s = res.summary()
    assert "Loss reduction          : 62.75%" in s

    fig, ax = plt.subplots()
    out_ax = res.plot(ax=ax)
    _assert_figure_valid(out_ax, ax_passed=ax)

    # Empty target variables edge case
    empty_res = OSRResult(
        optimal_params={},
        initial_params={},
        loss_opt=0.0,
        loss_initial=0.0,
        rule_params=(),
        target_vars=(),
        weights={},
        variance_table=pd.DataFrame(),
        converged=False,
        message="No variables",
        n_evaluations=0,
    )
    assert len(empty_res.to_frame()) == 0
    out_empty = empty_res.plot()
    _assert_figure_valid(out_empty)


# ===========================================================================
# 5. PolicyResult Tests
# ===========================================================================

def test_policy_result_presentation_discretion(simple_nk_model):
    """Test all 6 presentation methods on PolicyResult under discretionary policy."""
    pol = discretionary_policy(
        simple_nk_model,
        target_vars=["pi", "y"],
        weights={"pi": 1.0, "y": 0.25},
        instruments=["i"],
        beta=0.99,
    )
    assert isinstance(pol, PolicyResult)
    assert pol.regime == "discretion"
    assert len(pol.multipliers) == 0

    # 1. to_frame()
    df = pol.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert "i" in df.index

    # 2. summary()
    s = pol.summary()
    assert isinstance(s, str)
    assert "OPTIMAL POLICY REGIME: DISCRETION" in s
    assert "POLICY REACTION FUNCTIONS" in s
    assert "pi (w=1.0)" in s

    # 3. plot()
    out = pol.plot(periods=10)
    _assert_figure_valid(out)

    fig, ax = plt.subplots()
    out_ax = pol.plot(ax=ax, periods=10)
    _assert_figure_valid(out_ax, ax_passed=ax)

    # 4. to_markdown()
    md = pol.to_markdown()
    assert isinstance(md, str)
    assert "|" in md

    # 5. to_latex()
    tex = pol.to_latex()
    assert isinstance(tex, str)
    assert r"\begin{tabular}" in tex

    # 6. to_typst()
    typ = pol.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ


def test_policy_result_presentation_commitment(simple_nk_model):
    """Test all 6 presentation methods on PolicyResult under linear-quadratic commitment."""
    pol = lq_commitment(
        simple_nk_model,
        target_vars=["pi", "y"],
        weights={"pi": 1.0, "y": 0.25},
        instruments=["i"],
        beta=0.99,
    )
    assert isinstance(pol, PolicyResult)
    assert pol.regime == "commitment"
    assert len(pol.multipliers) > 0

    # Frame
    df = pol.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert "i" in df.index

    # Summary
    s = pol.summary()
    assert "OPTIMAL POLICY REGIME: COMMITMENT" in s
    assert "Lagrange multipliers" in s

    # Plot
    fig, ax = plt.subplots()
    out_ax = pol.plot(ax=ax, periods=12)
    _assert_figure_valid(out_ax, ax_passed=ax)

    # Tables
    assert "|" in pol.to_markdown()
    assert r"\begin{tabular}" in pol.to_latex()
    assert "#table(" in pol.to_typst()

    # Alias check
    assert pol.augmented_model is pol.linear_model


def test_policy_result_without_linear_model():
    """Test PolicyResult presentation when linear_model is None."""
    rules = pd.DataFrame({"u": [0.45], "eps_r": [0.0]}, index=["i"])
    pol = PolicyResult(
        regime="commitment",
        target_vars=("pi", "y"),
        weights={"pi": 1.0, "y": 0.5},
        instruments=("i",),
        beta=0.99,
        loss=0.0125,
        policy_rules=rules,
        transition_matrix=np.eye(2),
        impact_matrix=np.eye(2),
        multipliers=("mult_pi",),
        linear_model=None,
    )

    df = pol.to_frame()
    assert df.loc["i", "u"] == 0.45
    s = pol.summary()
    assert "Lagrange multipliers        : mult_pi" in s

    # Plot should gracefully indicate no linear model attached rather than crashing
    out = pol.plot()
    _assert_figure_valid(out)

    assert "|" in pol.to_markdown()
    assert r"\begin{tabular}" in pol.to_latex()
    assert "#table(" in pol.to_typst()
