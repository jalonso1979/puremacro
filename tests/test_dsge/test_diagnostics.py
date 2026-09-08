"""Tests for puremacro.dsge diagnostics, residual evaluation, and eigenvalue spectrum."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import puremacro.dsge as dsge
from puremacro.dsge import (
    DiagnosticFinding,
    EigenvalueTable,
    ModelDiagnosticsResult,
    check,
    model_diagnostics,
    resid,
)
from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.klein import KleinSolution

# --- Canonical Growth Model Fixture (Neoclassical growth) ---------------------

ALPHA, BETA, RHO = 0.33, 0.98, 0.9
GROWTH_PARAMS = dict(alpha=ALPHA, beta=BETA, rho=RHO)
GROWTH_GUESS = dict(c=0.5, k=0.1, z=1.0)


def growth_equations(xp, x, e, p):
    return [
        1 / x.c - p.beta * (p.alpha * xp.z * xp.k ** (p.alpha - 1)) / xp.c,
        x.c + xp.k - x.z * x.k ** p.alpha,
        xp.z - x.z ** p.rho * np.exp(e.eps),
    ]


@pytest.fixture
def growth_model():
    return dsge.build(
        growth_equations,
        variables=["c", "k", "z"],
        states=["k", "z"],
        shocks=["eps"],
        params=GROWTH_PARAMS,
        guess=GROWTH_GUESS,
    )


# --- Fertility Model Helper (Broken BGP endpoint replication) ----------------

def _build_broken_fertility_model():
    """Build the Alonso-Ortiz fertility model at the broken BGP calibration point."""
    fert_script_dir = Path(__file__).resolve().parents[2] / "docs/research/fertility_bk_diagnosis/scripts"
    sys.path.insert(0, str(fert_script_dir))
    import common as cm

    params = cm.build_params()
    z_ss = cm.z_from_params(params)

    def fert_eqs(lead, curr, lag, shocks, p):
        z_lead = np.array([getattr(lead, v) for v in cm.VAR_NAMES])
        z_curr = np.array([getattr(curr, v) for v in cm.VAR_NAMES])
        z_lag = np.array([getattr(lag, v) for v in cm.VAR_NAMES])
        eps = np.array([getattr(shocks, s) for s in cm.SHOCK_NAMES])
        p_dict = {k: getattr(p, k) for k in p._names} if hasattr(p, "_names") else dict(p)
        return cm.model_residuals(z_lead, z_curr, z_lag, eps, p_dict)

    return build_dynare(
        fert_eqs,
        variables=cm.VAR_NAMES,
        shocks=cm.SHOCK_NAMES,
        params=params,
        steady_state=dict(zip(cm.VAR_NAMES, z_ss)),
        check_steady_state=False,
        strict=False,
        method="central",
    )


# --- 1. check() Tests ---------------------------------------------------------

def test_check_determinate_model(growth_model):
    """check() correctly classifies a well-behaved saddle-path determinate model."""
    table = growth_model.check()
    assert isinstance(table, EigenvalueTable)
    assert table.is_determinate is True
    assert table.bk_satisfied is True
    assert table.n_forward == 1
    assert table.n_explosive == 1
    assert table.n_stable == 2
    assert table.loadings is None
    assert "Blanchard-Kahn condition satisfied" in table.bk_status

    # Test .to_frame()
    df = table.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["modulus", "real", "imag", "is_explosive", "is_unit_root"]

    # Test presentation formats
    summary_text = table.summary()
    assert "EIGENVALUES & BLANCHARD-KAHN DIAGNOSTICS" in summary_text
    assert "UNIQUE STABLE EQUILIBRIUM" in summary_text

    md = table.to_markdown()
    assert isinstance(md, str) and "modulus" in md

    latex = table.to_latex()
    assert isinstance(latex, str) and "\\begin{tabular}" in latex

    typst = table.to_typst()
    assert isinstance(typst, str) and "#table(" in typst


def test_check_unit_circle_plot(growth_model):
    """check().plot() generates a publication-quality complex plane unit-circle diagram."""
    table = growth_model.check()
    fig, ax = plt.subplots(figsize=(6, 6))
    out_ax = table.plot(ax=ax)
    assert out_ax is ax
    assert ax.get_xlabel() == "Re(lambda)"
    assert ax.get_ylabel() == "Im(lambda)"
    assert "Eigenvalue Spectrum" in ax.get_title()
    # Check that scatter collections and plot lines were added
    assert len(ax.collections) >= 2  # stable scatter + explosive scatter
    assert len(ax.lines) >= 3        # unit circle + axes lines
    plt.close(fig)


def test_check_fertility_model_offending_loadings():
    """check() isolates the offending root |λ| ≈ 1.0685 loading chiefly on k (1.000) and n (0.488)."""
    model = _build_broken_fertility_model()
    table = check(model)

    assert table.is_determinate is False
    assert table.bk_satisfied is False
    assert table.n_explosive == 13
    assert table.n_forward == 12
    assert table.loadings is not None

    # Offending root is the boundary crosser around 1.0685
    assert len(table.loadings) >= 1
    offending_idx = list(table.loadings.keys())[0]
    offending_mod = table.modulus[offending_idx]
    assert np.isclose(offending_mod, 1.0685, atol=1e-3)

    lds = table.loadings[offending_idx]
    # Dominant loading is on capital 'k' (1.000), secondary on population/fertility 'n' (~0.488)
    assert lds["k"] == pytest.approx(1.0, rel=1e-4)
    assert lds["n"] == pytest.approx(0.488, rel=5e-2)
    assert "1.0685" in table.bk_status
    assert "k" in table.bk_status
    assert "n" in table.bk_status

    # Verify summary displays the loadings section
    summary_text = table.summary()
    assert "OFFENDING ROOT VARIABLE LOADINGS" in summary_text
    assert "k: 1.0000" in summary_text

    # Verify plot highlights offending root
    fig, ax = plt.subplots(figsize=(6, 6))
    out_ax = table.plot(ax=ax)
    assert out_ax is ax
    plt.close(fig)


def test_check_indeterminacy_model():
    """check() identifies missing explosive roots and variable loadings under indeterminacy."""
    # Build 3-equation New Keynesian model with Taylor rule inflation coefficient < 1 (phi_pi = 0.5)
    def nk_eqs(lead, curr, lag, shocks, p):
        # E_t x_{t+1}, x_t, x_{t-1}, eps_t
        # y = lead.y - (1/sigma) * (curr.i - lead.pi)
        # pi = beta * lead.pi + kappa * curr.y
        # i = phi_pi * curr.pi + eps
        eq1 = curr.y - lead.y + (1.0 / p.sigma) * (curr.i - lead.pi)
        eq2 = curr.pi - p.beta * lead.pi - p.kappa * curr.y
        eq3 = curr.i - p.phi_pi * curr.pi - shocks.eps_i
        return [eq1, eq2, eq3]

    m_indet = build_dynare(
        nk_eqs,
        variables=["y", "pi", "i"],
        shocks=["eps_i"],
        states=["i"],  # predetermined dummy to allow lag-lead structure
        params={"beta": 0.99, "sigma": 1.0, "kappa": 0.1, "phi_pi": 0.5},
        guess={"y": 0.0, "pi": 0.0, "i": 0.0},
        strict=False,
    )
    table = m_indet.check()
    assert table.is_determinate is False
    assert table.n_explosive < table.n_forward
    assert table.loadings is not None
    assert "indeterminacy" in table.bk_status.lower() or "missing" in table.bk_status.lower()


def test_check_unit_roots():
    """check() flags unit roots and reports them in summary."""
    # Growth model with random walk TFP (rho = 1.0)
    rw_params = dict(GROWTH_PARAMS, rho=1.0)
    m_rw = dsge.build(
        growth_equations,
        variables=["c", "k", "z"],
        states=["k", "z"],
        shocks=["eps"],
        params=rw_params,
        guess=GROWTH_GUESS,
    )
    table = m_rw.check()
    assert np.any(table.is_unit_root)
    assert int(np.sum(table.is_unit_root)) >= 1
    assert "Unit roots   : 1" in table.summary()


# --- 2. resid() Tests ---------------------------------------------------------

def test_resid_exact_steady_state(growth_model):
    """resid() on an exact steady state yields near machine-zero residuals."""
    r = growth_model.resid()
    assert isinstance(r, pd.Series)
    assert len(r) == 3
    assert np.max(np.abs(r.values)) < 1e-12
    # Verify sorting: abs(r[0]) >= abs(r[1]) >= ...
    abs_vals = np.abs(r.values)
    assert np.all(abs_vals[:-1] >= abs_vals[1:])


def test_resid_fertility_broken_bgp_replicates_1_48():
    """resid() reproduces the non-zero equation residual norm ≈ 1.48 on the fertility benchmark."""
    model = _build_broken_fertility_model()
    r = resid(model)

    assert isinstance(r, pd.Series)
    assert len(r) == 12

    # Verify reproduction of norm ≈ 1.48 from FINDINGS.md
    norm_l2 = float(np.linalg.norm(r.values))
    norm_linf = float(np.max(np.abs(r.values)))
    assert np.isclose(norm_l2, 1.4837, atol=1e-2)
    assert np.isclose(norm_linf, 1.4830, atol=1e-2)

    # Largest residual heads the series
    assert abs(r.iloc[0]) == pytest.approx(norm_linf, rel=1e-6)
    # Series is monotonically decreasing in absolute magnitude
    abs_vals = np.abs(r.values)
    assert np.all(abs_vals[:-1] >= abs_vals[1:])


# --- 3. model_diagnostics() Tests ---------------------------------------------

def test_model_diagnostics_healthy_model(growth_model):
    """model_diagnostics() on a healthy model confirms full rank, regular pencil, and passing status."""
    res = growth_model.model_diagnostics()
    assert isinstance(res, ModelDiagnosticsResult)
    assert res.passed is True
    assert res.static_rank == 3
    assert res.n_vars == 3
    assert res.rank_deficient is False
    assert res.is_pencil_regular is True
    assert res.pencil_regular is True
    assert res.stochastic_singularity is False
    assert len(res.unused_variables) == 0
    assert len(res.unused_equations) == 0
    assert len(res.collinear_equations) == 0
    assert len(res.collinear_variables) == 0

    # Summary and reporting methods
    summary = res.summary()
    assert "DSGE MODEL DIAGNOSTICS" in summary
    assert "Overall status         : PASSED" in summary
    assert "FULL RANK" in summary

    df = res.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert "category" in df.columns

    md = res.to_markdown()
    assert isinstance(md, str) and "passed" in md.lower()

    latex = res.to_latex()
    assert isinstance(latex, str) and "\\begin{tabular}" in latex

    typst = res.to_typst()
    assert isinstance(typst, str) and "#table(" in typst

    fig, ax = plt.subplots()
    out_ax = res.plot(ax=ax)
    assert out_ax is ax
    plt.close(fig)


def test_model_diagnostics_planted_collinear_equations():
    """model_diagnostics() flags rank deficiency and isolates collinear equation subsets."""
    from puremacro.dsge.klein import KleinSolution

    def collinear_eqs(lead, curr, lag, shocks, p):
        eq1 = curr.c - 0.9 * lag.c
        eq2 = curr.c - 0.9 * lag.c  # Exact duplicate
        eq3 = curr.z - 0.8 * lag.z - shocks.eps
        return [eq1, eq2, eq3]

    sol = KleinSolution(
        G=np.eye(2),
        F=np.eye(1, 2),
        N=np.ones((2, 1)),
        L=np.ones((1, 1)),
        eu=(0, 0),
        eigenvalues=np.array([0.9, 0.9, 0.8]),
    )
    A_plus = np.zeros((3, 3))
    A_0 = np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 0, 1.0]])
    A_minus = np.array([[-0.9, 0, 0], [-0.9, 0, 0], [0, 0, -0.8]])

    m = dsge.LinearModel(
        variables=("c", "k", "z"),
        states=("c", "z"),
        controls=("k",),
        shocks=("eps",),
        steady_state=pd.Series([0.0, 0.0, 0.0], index=["c", "k", "z"]),
        units={"c": "level", "k": "level", "z": "level"},
        solution=sol,
        A=np.eye(3),
        B=np.eye(3),
        C=np.zeros((3, 1)),
        method="complex",
        residual_norm=0.0,
        _dynare_equations=collinear_eqs,
        _A_plus=A_plus,
        _A_0=A_0,
        _A_minus=A_minus,
        timing="dynare",
    )
    res = model_diagnostics(m)
    assert res.passed is False
    assert res.rank_deficient is True
    assert res.static_rank < res.static_n_vars
    assert len(res.collinear_equations) >= 1
    assert any(f.category == "static_rank" and f.severity == "error" for f in res.findings)


def test_model_diagnostics_unused_variables():
    """model_diagnostics() detects variables that do not enter any equation."""
    from puremacro.dsge.klein import KleinSolution

    def unused_var_eqs(lead, curr, lag, shocks, p):
        eq1 = curr.c - 0.5 * lag.c - shocks.eps1
        eq2 = curr.k - 0.6 * lag.k - shocks.eps2
        # eq3 has no dependency on 'w'
        eq3 = curr.z - 0.7 * lag.z
        return [eq1, eq2, eq3]

    sol = KleinSolution(
        G=np.eye(3),
        F=np.eye(1, 3),
        N=np.ones((3, 2)),
        L=np.ones((1, 2)),
        eu=(0, 0),
        eigenvalues=np.array([0.5, 0.6, 0.7, 0.0]),
    )
    m = dsge.LinearModel(
        variables=("c", "k", "z", "w"),
        states=("c", "k", "z"),
        controls=("w",),
        shocks=("eps1", "eps2"),
        steady_state=pd.Series([0.0, 0.0, 0.0, 0.0], index=["c", "k", "z", "w"]),
        units={"c": "level", "k": "level", "z": "level", "w": "level"},
        solution=sol,
        A=np.eye(4),
        B=np.eye(4),
        C=np.zeros((4, 2)),
        method="complex",
        residual_norm=0.0,
        _dynare_equations=unused_var_eqs,
        timing="dynare",
    )
    res = model_diagnostics(m)
    assert res.passed is False
    assert "w" in res.unused_variables
    assert any(f.category == "incidence" and "w" in f.message for f in res.findings)


def test_model_diagnostics_stochastic_singularity(growth_model):
    """model_diagnostics() warns when n_varobs exceeds structural shocks + measurement errors."""
    import dataclasses
    # Attach 2 observables ("c", "k") when only 1 shock ("eps") exists
    m_singular = dataclasses.replace(growth_model, _varobs=("c", "k"))
    res = model_diagnostics(m_singular)

    assert res.stochastic_singularity is True
    assert any(f.category == "stochastic_singularity" and f.severity == "warning" for f in res.findings)


def test_model_diagnostics_singular_pencil():
    """model_diagnostics() detects when the dynamic matrix pencil (A, B) is singular."""
    import dataclasses
    # Create model whose A and B matrices have an all-zero row/col (coincident zeros)
    A = np.zeros((2, 2))
    B = np.zeros((2, 2))
    A[0, 0] = 1.0
    B[0, 0] = 0.5
    # Second row/col is identically 0: singular pencil

    from puremacro.dsge.klein import KleinSolution
    sol = KleinSolution(G=np.eye(1), F=np.eye(1), N=np.ones((1, 1)), L=np.ones((1, 1)), eu=(0, 0), eigenvalues=np.array([0.5, np.nan]))
    m = dsge.LinearModel(
        variables=("x", "y"),
        states=("x",),
        controls=("y",),
        shocks=("e",),
        steady_state=pd.Series([0.0, 0.0], index=["x", "y"]),
        units={"x": "level", "y": "level"},
        solution=sol,
        A=A,
        B=B,
        C=np.zeros((2, 1)),
        method="complex",
        residual_norm=0.0,
    )
    res = model_diagnostics(m)
    assert res.passed is False
    assert res.is_pencil_regular is False
    assert any(f.category == "pencil" and f.severity == "error" for f in res.findings)


# --- 4. Method Wiring on LinearModel ------------------------------------------

def test_linear_model_methods_wired(growth_model):
    """LinearModel delegates .check(), .resid(), and .model_diagnostics() seamlessly."""
    t = growth_model.check()
    assert isinstance(t, EigenvalueTable)

    r = growth_model.resid()
    assert isinstance(r, pd.Series)

    d = growth_model.model_diagnostics()
    assert isinstance(d, ModelDiagnosticsResult)


# --- 5. Adversarial Stress Testing & Edge Cases -------------------------------

def test_infinite_eigenvalue_offending_loadings():
    """check() must not crash with LinAlgError when an offending explosive root is infinite."""
    # A has rank 1, so generalized eigenvalue beta/alpha has an infinite root.
    # When n_forward = 0, both roots are excess explosive roots.
    # The second root is infinite: mu = inf.
    A = np.array([[1.0, 0.0], [0.0, 0.0]])
    B = np.array([[2.0, 0.0], [0.0, 1.0]])

    sol = KleinSolution(
        G=np.eye(2),
        F=np.zeros((0, 2)),
        N=np.zeros((2, 1)),
        L=np.zeros((0, 1)),
        eu=(0, 0),
        eigenvalues=np.array([2.0, np.inf]),
    )
    m = dsge.LinearModel(
        variables=("x", "y"),
        states=("x", "y"),
        controls=(),
        shocks=("e",),
        steady_state=pd.Series([0.0, 0.0], index=["x", "y"]),
        units={"x": "level", "y": "level"},
        solution=sol,
        A=A,
        B=B,
        C=np.zeros((2, 1)),
        method="complex",
        residual_norm=0.0,
    )

    table = check(m)
    assert isinstance(table, EigenvalueTable)
    assert table.is_determinate is False
    assert table.n_explosive == 2
    assert table.loadings is not None
    # Offending roots must include the infinite root without crashing
    assert 1 in table.loadings
    assert "y" in table.loadings[1]


def test_nan_steady_state_residual_detection():
    """model_diagnostics() must flag an error when steady state residuals evaluate to NaN."""
    def nan_eqs(lead, curr, lag, shocks, p):
        # 0.0 / 0.0 evaluates to NaN
        return [curr.x / curr.y, curr.y - 1.0]

    sol = KleinSolution(
        G=np.eye(1),
        F=np.eye(1, 1),
        N=np.zeros((1, 1)),
        L=np.zeros((1, 1)),
        eu=(1, 1),
        eigenvalues=np.array([0.5, 1.5]),
    )
    m = dsge.LinearModel(
        variables=("x", "y"),
        states=("x",),
        controls=("y",),
        shocks=("e",),
        steady_state=pd.Series([0.0, 0.0], index=["x", "y"]),
        units={"x": "level", "y": "level"},
        solution=sol,
        A=np.eye(2),
        B=np.eye(2),
        C=np.zeros((2, 1)),
        method="complex",
        residual_norm=0.0,
        _dynare_equations=nan_eqs,
        timing="dynare",
    )

    diag = model_diagnostics(m)
    assert diag.passed is False
    ss_errors = [f for f in diag.findings if f.category == "steady_state" and f.severity == "error"]
    assert len(ss_errors) >= 1, "model_diagnostics failed to flag NaN steady state as an error"


def test_underdetermined_collinear_variable_reporting():
    """model_diagnostics() must report collinear / unconstrained variables when n_eq < n_vars."""
    def underdet_eqs(lead, curr, lag, shocks, p):
        # 2 equations for 3 variables (x, y, z); z is completely unconstrained
        return [curr.x - 0.5 * lag.x, curr.y - 0.5 * lag.y]

    sol = KleinSolution(
        G=np.eye(3),
        F=np.zeros((0, 3)),
        N=np.zeros((3, 1)),
        L=np.zeros((0, 1)),
        eu=(0, 0),
        eigenvalues=np.array([0.5, 0.5, 0.0]),
    )
    m = dsge.LinearModel(
        variables=("x", "y", "z"),
        states=("x", "y", "z"),
        controls=(),
        shocks=("e",),
        steady_state=pd.Series([0.0, 0.0, 0.0], index=["x", "y", "z"]),
        units={"x": "level", "y": "level", "z": "level"},
        solution=sol,
        A=np.eye(3),
        B=np.eye(3),
        C=np.zeros((3, 1)),
        method="complex",
        residual_norm=0.0,
        _dynare_equations=underdet_eqs,
        timing="dynare",
    )
    diag = model_diagnostics(m)
    assert diag.passed is False
    assert diag.rank_deficient is True
    assert diag.static_rank == 2
    assert diag.static_n_vars == 3
    # Deficiency is 1: collinear_variables MUST identify the unconstrained variable combination
    assert len(diag.collinear_variables) >= 1, "collinear_variables was empty for an underdetermined system"
    assert "z" in diag.collinear_variables[0]


def test_empty_model_diagnostics():
    """check(), resid(), and model_diagnostics() must handle an empty (0-variable) model gracefully."""
    sol = KleinSolution(
        G=np.zeros((0, 0)),
        F=np.zeros((0, 0)),
        N=np.zeros((0, 0)),
        L=np.zeros((0, 0)),
        eu=(1, 1),
        eigenvalues=np.array([]),
    )
    m = dsge.LinearModel(
        variables=(),
        states=(),
        controls=(),
        shocks=(),
        steady_state=pd.Series([], dtype=float),
        units={},
        solution=sol,
        A=np.zeros((0, 0)),
        B=np.zeros((0, 0)),
        C=np.zeros((0, 0)),
        method="complex",
        residual_norm=0.0,
    )

    r = resid(m)
    assert isinstance(r, pd.Series)
    assert len(r) == 0

    table = check(m)
    assert isinstance(table, EigenvalueTable)
    assert table.is_determinate is True
    assert len(table.eigenvalues) == 0

    diag = model_diagnostics(m)
    assert isinstance(diag, ModelDiagnosticsResult)
    assert diag.passed is True
    assert diag.static_rank == 0


def test_mismatched_model_A_shape_fallback_incidence():
    """model_diagnostics fallback incidence must not crash when model.A shape differs from n_eq."""
    # J_static is (3, 2) from _A_0 + _A_minus, but model.A is (2, 2)
    A_0 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    A_minus = np.zeros((3, 2))
    A_plus = np.zeros((3, 2))

    sol = KleinSolution(G=np.eye(2), F=np.zeros((0, 2)), N=np.zeros((2, 1)), L=np.zeros((0, 1)), eu=(0, 0), eigenvalues=np.array([0.5, 0.5]))
    m = dsge.LinearModel(
        variables=("x", "y"),
        states=("x", "y"),
        controls=(),
        shocks=("e",),
        steady_state=pd.Series([0.0, 0.0], index=["x", "y"]),
        units={"x": "level", "y": "level"},
        solution=sol,
        A=np.eye(2),
        B=np.eye(2),
        C=np.zeros((2, 1)),
        method="complex",
        residual_norm=0.0,
        _A_plus=A_plus,
        _A_0=A_0,
        _A_minus=A_minus,
    )
    # Should not raise ValueError: non-broadcastable output operand
    diag = model_diagnostics(m)
    assert isinstance(diag, ModelDiagnosticsResult)
    assert diag.incidence_matrix.shape == (3, 2)


def test_singular_pencil_summary_and_plot():
    """Singular pencil in check() must produce valid summary, plot, and presentation formats."""
    A = np.zeros((2, 2))
    B = np.zeros((2, 2))
    A[0, 0] = 1.0
    B[0, 0] = 0.5

    sol = KleinSolution(G=np.eye(1), F=np.eye(1), N=np.ones((1, 1)), L=np.ones((1, 1)), eu=(0, 0), eigenvalues=np.array([0.5, np.nan]))
    m = dsge.LinearModel(
        variables=("x", "y"),
        states=("x",),
        controls=("y",),
        shocks=("e",),
        steady_state=pd.Series([0.0, 0.0], index=["x", "y"]),
        units={"x": "level", "y": "level"},
        solution=sol,
        A=A,
        B=B,
        C=np.zeros((2, 1)),
        method="complex",
        residual_norm=0.0,
    )
    table = check(m)
    assert table.is_determinate is False
    assert "Singular matrix pencil" in table.bk_status

    # Presentation contract
    summary = table.summary()
    assert "Singular matrix pencil" in summary
    assert "DETERMINACY FAILED" in summary

    md = table.to_markdown()
    assert isinstance(md, str)

    latex = table.to_latex()
    assert isinstance(latex, str)

    typst = table.to_typst()
    assert isinstance(typst, str)

    fig, ax = plt.subplots()
    out_ax = table.plot(ax=ax)
    assert out_ax is ax
    plt.close(fig)
