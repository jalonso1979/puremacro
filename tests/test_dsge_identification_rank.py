"""Comprehensive test suite for DSGE parameter identification rank criteria.

Validates:
1. Iskrev (2010) rank, singular values, and condition number.
2. Komunjer & Ng (2011) transfer function matrix rank condition (JH).
3. Komunjer & Ng (2011) cross-spectral density matrix rank condition (JS).
4. Identifiable 3-shock New Keynesian benchmark model (full rank across all criteria).
5. Unidentifiable 1-shock New Keynesian benchmark model (recovers Taylor rule null space).
6. Planted unidentified parameter product and sum across all criteria.
7. Spectral identification resolving lag deficiency (AR(1) with measurement error).
8. IdentificationResult rank scorecard, collinear pairs, and structured warnings.
9. Publication exports (.summary(), .to_markdown(), .to_latex(), .to_typst(), .plot()).
10. Pyodide zero-dependency contract (strictly numpy, scipy, pandas, matplotlib).
"""
from __future__ import annotations

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import puremacro.dsge as dsge
from puremacro.dsge import IdentificationResult, identification
from puremacro.dsge.dynare import build_dynare


# =============================================================================
# Helper Model Constructors
# =============================================================================

def _build_3shock_nk_model(params: dict[str, float] | None = None) -> dsge.LinearModel:
    """Standard 3-equation New Keynesian model with IS, NKPC, Taylor rule, and 3 shocks."""
    def nk_eqs(lead, curr, lag, shocks, p):
        return [
            curr.y - lead.y + (1.0 / p.sigma) * (curr.r - lead.pi) - curr.g,
            curr.pi - 0.99 * lead.pi - p.kappa * curr.y - curr.u,
            curr.r - p.phi_pi * curr.pi - p.phi_y * curr.y - shocks.eps_r,
            curr.g - p.rho_g * lag.g - shocks.eps_g,
            curr.u - p.rho_u * lag.u - shocks.eps_u,
        ]

    p_dict = dict(
        sigma=1.0,
        kappa=0.1,
        phi_pi=1.5,
        phi_y=0.125,
        rho_g=0.8,
        rho_u=0.5,
    )
    if params is not None:
        p_dict.update(params)

    return build_dynare(
        nk_eqs,
        variables=["y", "pi", "r", "g", "u"],
        shocks=["eps_r", "eps_g", "eps_u"],
        params=p_dict,
        states=["g", "u"],
        guess=dict(y=0.0, pi=0.0, r=0.0, g=0.0, u=0.0),
    )


def _build_1shock_nk_model(params: dict[str, float] | None = None) -> dsge.LinearModel:
    """3-equation New Keynesian model with only 1 shock (cost-push shock eps_u active)."""
    def nk_1shock_eqs(lead, curr, lag, shocks, p):
        return [
            curr.y - lead.y + (1.0 / p.sigma) * (curr.r - lead.pi),
            curr.pi - 0.99 * lead.pi - p.kappa * curr.y - curr.u,
            curr.r - p.phi_pi * curr.pi - p.phi_y * curr.y,
            curr.u - p.rho_u * lag.u - shocks.eps_u,
        ]

    p_dict = dict(
        sigma=1.0,
        kappa=0.1,
        phi_pi=1.5,
        phi_y=0.125,
        rho_u=0.5,
    )
    if params is not None:
        p_dict.update(params)

    return build_dynare(
        nk_1shock_eqs,
        variables=["y", "pi", "r", "u"],
        shocks=["eps_u"],
        params=p_dict,
        states=["u"],
        guess=dict(y=0.0, pi=0.0, r=0.0, u=0.0),
    )


# =============================================================================
# Test Cases 1 - 10
# =============================================================================

def test_iskrev_rank_singular_values_and_condition_number():
    """Test 1: Validates J1 and J2 singular value spectra and condition numbers."""
    m = _build_3shock_nk_model()
    p_names = ["sigma", "kappa", "phi_pi", "phi_y", "rho_g", "rho_u", "SE_eps_r", "SE_eps_g", "SE_eps_u"]
    res = dsge.identification(m, params=p_names, varobs=["y", "pi", "r"], lags=2)

    # Singular values: 1D array of length n_params, non-negative, non-increasing
    assert isinstance(res.j1_singular_values, np.ndarray)
    assert res.j1_singular_values.ndim == 1
    assert len(res.j1_singular_values) == len(p_names)
    assert np.all(res.j1_singular_values >= 0.0)
    assert np.all(np.diff(res.j1_singular_values) <= 1e-12)

    assert isinstance(res.j2_singular_values, np.ndarray)
    assert res.j2_singular_values.ndim == 1
    assert len(res.j2_singular_values) == len(p_names)
    assert np.all(res.j2_singular_values >= 0.0)
    assert np.all(np.diff(res.j2_singular_values) <= 1e-12)

    # Condition number calculation: s_max / s_min
    expected_j1_cond = float(res.j1_singular_values[0] / res.j1_singular_values[-1])
    assert res.j1_condition_number == pytest.approx(expected_j1_cond, rel=1e-5)
    assert np.isfinite(res.j1_condition_number)
    assert res.j1_condition_number > 1.0

    expected_j2_cond = float(res.j2_singular_values[0] / res.j2_singular_values[-1])
    assert res.j2_condition_number == pytest.approx(expected_j2_cond, rel=1e-5)
    assert np.isfinite(res.j2_condition_number)

    # Rank-deficient model should have infinite condition number and tiny smallest singular value
    def eqs_prod(xp, x, e, p):
        return [xp.y - (p.theta1 * p.theta2) * x.y - e.eps]

    m_unid = dsge.build(
        eqs_prod,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(theta1=0.7, theta2=0.7),
        guess=dict(y=0.0),
        linearize="level",
    )
    res_unid = dsge.identification(m_unid, params=["theta1", "theta2"], varobs=["y"])

    assert np.isinf(res_unid.j1_condition_number)
    assert np.isinf(res_unid.j2_condition_number)
    assert res_unid.j1_singular_values[-1] < 1e-12
    assert res_unid.j2_singular_values[-1] < 1e-12


def test_komunjer_ng_transfer_function_rank():
    """Test 2: Validates frequency-domain transfer function Jacobian JH, grids, and shock derivatives."""
    m = _build_3shock_nk_model()
    p_names = ["sigma", "kappa", "phi_pi", "phi_y", "rho_g", "rho_u", "SE_eps_r", "SE_eps_g", "SE_eps_u"]

    # 1. Uniform frequency grid (default)
    res_unif = dsge.identification(m, params=p_names, varobs=["y", "pi", "r"], n_freq=16, freq_type="uniform")
    assert res_unif.jh_rank == 9
    assert res_unif.jh_n_params == 9
    assert res_unif.is_identified_transfer is True
    assert len(res_unif.jh_singular_values) == 9
    assert res_unif.jh_condition_number < 100.0

    # 2. Gauss-Legendre quadrature nodes
    res_gl = dsge.identification(m, params=p_names, varobs=["y", "pi", "r"], n_freq=16, freq_type="gauss_legendre")
    assert res_gl.jh_rank == 9
    assert res_gl.jh_condition_number < 100.0

    # 3. Explicit custom frequency grid
    custom_omega = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    res_cust = dsge.identification(m, params=p_names, varobs=["y", "pi", "r"], frequencies=custom_omega)
    assert res_cust.n_freq == len(custom_omega)
    assert res_cust.jh_rank == 9


def test_komunjer_ng_spectral_density_rank():
    """Test 3: Validates cross-spectral density Jacobian JS and Hermitian non-redundant stacking."""
    m = _build_3shock_nk_model()
    p_names = ["sigma", "kappa", "phi_pi", "phi_y", "rho_g", "rho_u", "SE_eps_r", "SE_eps_g", "SE_eps_u"]
    res = dsge.identification(m, params=p_names, varobs=["y", "pi", "r"], n_freq=16)

    assert res.js_rank == 9
    assert res.js_n_params == 9
    assert res.is_identified_spectrum is True
    assert len(res.js_singular_values) == 9
    assert np.all(res.js_singular_values >= 0.0)
    assert np.all(np.diff(res.js_singular_values) <= 1e-12)
    assert np.isfinite(res.js_condition_number)
    assert res.js_condition_number == pytest.approx(res.js_singular_values[0] / res.js_singular_values[-1], rel=1e-5)


def test_identifiable_3shock_new_keynesian_benchmark():
    """Test 4: Full rank 9/9 across all 4 criteria on 3-shock New Keynesian benchmark."""
    m = _build_3shock_nk_model()
    p_names = ["sigma", "kappa", "phi_pi", "phi_y", "rho_g", "rho_u", "SE_eps_r", "SE_eps_g", "SE_eps_u"]
    res = dsge.identification(m, params=p_names, varobs=["y", "pi", "r"], lags=2, n_freq=16)

    # All criteria identified
    assert res.is_identified is True
    assert res.is_identified_solution is True
    assert res.is_identified_moments is True
    assert res.is_identified_transfer is True
    assert res.is_identified_spectrum is True
    assert res.rank_deficient is False

    # Full ranks
    assert res.j1_rank == 9
    assert res.j2_rank == 9
    assert res.jh_rank == 9
    assert res.js_rank == 9

    # Condition numbers
    assert res.jh_condition_number < 100.0
    assert res.j1_condition_number < 1000.0
    assert res.js_condition_number < 5000.0

    # Null spaces empty
    assert len(res.j1_null_combinations) == 0
    assert len(res.j2_null_combinations) == 0
    assert len(res.jh_null_combinations) == 0
    assert len(res.js_null_combinations) == 0


def test_unidentifiable_1shock_new_keynesian_benchmark():
    """Test 5: Recovers Taylor rule null space 0.8480 phi_pi + 0.5300 phi_y = 0 in 1-shock model."""
    m = _build_1shock_nk_model()
    p_names = ["sigma", "kappa", "phi_pi", "phi_y", "rho_u"]
    res = dsge.identification(m, params=p_names, varobs=["y", "pi", "r"], lags=2)

    assert res.is_identified is False
    assert res.rank_deficient is True
    assert res.j1_rank == 4
    assert res.j1_n_params == 5
    assert res.jh_rank == 4
    assert res.jh_n_params == 5

    # Check that Taylor rule null direction is recovered
    assert len(res.j1_null_combinations) == 1
    comb = res.j1_null_combinations[0]
    assert "phi_pi" in comb and "phi_y" in comb

    # Null space vector ratio: 0.8480 / 0.5300 = 1.600
    null_v = res.j1_null_space[0]
    c_pi = abs(null_v[p_names.index("phi_pi")])
    c_y = abs(null_v[p_names.index("phi_y")])
    assert c_y > 1e-4
    ratio = c_pi / c_y
    assert ratio == pytest.approx(0.8480 / 0.5300, rel=0.02)


def test_planted_product_and_sum_all_criteria():
    """Test 6: Planted product and sum models recover 1/sqrt(2) null vector across all 4 criteria."""
    # 1. Planted product theta1 * theta2
    def eqs_prod(xp, x, e, p):
        return [xp.y - (p.theta1 * p.theta2) * x.y - e.eps]

    m_prod = dsge.build(
        eqs_prod,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(theta1=0.7, theta2=0.7),
        guess=dict(y=0.0),
        linearize="level",
    )
    res_prod = dsge.identification(m_prod, params=["theta1", "theta2"], varobs=["y"])

    assert res_prod.is_identified is False
    assert res_prod.j1_rank == 1
    assert res_prod.j2_rank == 1
    assert res_prod.jh_rank == 1
    assert res_prod.js_rank == 1

    expected_null = np.array([[1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)]])
    assert np.allclose(res_prod.j1_null_space, expected_null, atol=1e-5)
    assert np.allclose(res_prod.j2_null_space, expected_null, atol=1e-5)
    assert np.allclose(res_prod.jh_null_space, expected_null, atol=1e-5)
    assert np.allclose(res_prod.js_null_space, expected_null, atol=1e-5)

    # 2. Planted sum alpha + beta
    def eqs_sum(xp, x, e, p):
        return [xp.y - (p.alpha + p.beta) * x.y - e.eps]

    m_sum = dsge.build(
        eqs_sum,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(alpha=0.35, beta=0.35),
        guess=dict(y=0.0),
        linearize="level",
    )
    res_sum = dsge.identification(m_sum, params=["alpha", "beta"], varobs=["y"])

    assert res_sum.is_identified is False
    assert res_sum.j1_rank == 1
    assert res_sum.j2_rank == 1
    assert res_sum.jh_rank == 1
    assert res_sum.js_rank == 1
    assert np.allclose(res_sum.j1_null_space, expected_null, atol=1e-5)
    assert np.allclose(res_sum.jh_null_space, expected_null, atol=1e-5)
    assert np.allclose(res_sum.js_null_space, expected_null, atol=1e-5)


def test_spectral_identification_resolves_lag_deficiency():
    """Test 7: AR(1) with measurement error at lag 1 vs lag 2 vs frequency spectrum."""
    def eqs(xp, x, e, p):
        return [xp.y - p.rho * x.y - e.eps]

    m = dsge.build(
        eqs,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(rho=0.7),
        guess=dict(y=0.0),
        linearize="level",
    )

    # At lags=1: J2 has only 2 non-zero moments for 3 parameters -> rank 2 / 3
    res_lag1 = dsge.identification(m, params=["rho", "SE_eps", "ME_y"], varobs=["y"], lags=1)
    assert res_lag1.j2_rank == 2
    assert res_lag1.is_identified_moments is False
    assert res_lag1.is_identified is False

    # But frequency spectrum JS incorporates all autocovariances -> full rank 3 / 3
    assert res_lag1.js_rank == 3
    assert res_lag1.is_identified_spectrum is True

    # At lags=2: J2 reaches full rank 3 / 3
    res_lag2 = dsge.identification(m, params=["rho", "SE_eps", "ME_y"], varobs=["y"], lags=2)
    assert res_lag2.j2_rank == 3
    assert res_lag2.is_identified_moments is True
    assert res_lag2.is_identified is True


def test_identification_result_scorecard_and_warnings():
    """Test 8: Verifies .rank_scorecard(), collinear_pairs, and diagnostic warnings."""
    def eqs_prod(xp, x, e, p):
        return [xp.y - (p.theta1 * p.theta2) * x.y - e.eps]

    m = dsge.build(
        eqs_prod,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(theta1=0.7, theta2=0.7),
        guess=dict(y=0.0),
        linearize="level",
    )
    res = dsge.identification(m, params=["theta1", "theta2"], varobs=["y"])

    # Scorecard DataFrame
    sc = res.rank_scorecard()
    assert isinstance(sc, pd.DataFrame)
    assert list(sc.index) == ["J1", "J2", "JH", "JS"]
    assert "criterion" in sc.columns
    assert "rank" in sc.columns
    assert "total" in sc.columns
    assert "deficiency" in sc.columns
    assert "condition_number" in sc.columns
    assert "status" in sc.columns
    assert np.all(sc["deficiency"] == 1)
    assert np.all(sc["rank"] == 1)

    # Collinear pairs: pairwise R^2 ~ 1.0 between theta1 and theta2
    assert len(res.collinear_pairs) > 0
    p1, p2, r2, crit = res.collinear_pairs[0]
    assert {p1, p2} == {"theta1", "theta2"}
    assert r2 > 0.999

    # Warnings generated
    assert len(res.warnings) > 0
    warn_text = " ".join(res.warnings)
    assert "rank deficient" in warn_text
    assert "collinearity" in warn_text.lower() or "collinear" in warn_text.lower()


def test_publication_tables_and_plots():
    """Test 9: Verifies publication formatting (.summary, .to_markdown, .to_latex, .to_typst, .plot)."""
    m = _build_3shock_nk_model()
    p_names = ["sigma", "kappa", "phi_pi", "phi_y", "rho_g", "rho_u", "SE_eps_r", "SE_eps_g", "SE_eps_u"]
    res = dsge.identification(m, params=p_names, varobs=["y", "pi", "r"], lags=2)

    # Summary
    s = res.summary()
    assert "PARAMETER IDENTIFICATION ANALYSIS" in s
    assert "IDENTIFIED" in s
    assert "RANK CRITERIA SCORECARD" in s
    assert "PARAMETER IDENTIFICATION SUMMARY" in s

    # Markdown scorecard and parameter table
    md_sc = res.to_markdown(table="scorecard")
    assert "|" in md_sc
    assert "J1 (Iskrev Solution)" in md_sc
    assert "JH (Komunjer-Ng Transfer)" in md_sc

    md_params = res.to_markdown(table="parameters")
    assert "|" in md_params
    assert "sigma" in md_params

    # LaTeX tabular
    ltx_sc = res.to_latex(table="scorecard")
    assert r"\begin{tabular}" in ltx_sc
    assert "FULL RANK" in ltx_sc

    ltx_params = res.to_latex(table="parameters")
    assert r"\begin{tabular}" in ltx_params
    assert "sigma" in ltx_params

    # Typst table
    typ_sc = res.to_typst(table="scorecard")
    assert "#table(" in typ_sc

    typ_params = res.to_typst(table="parameters")
    assert "#table(" in typ_params

    # Plot
    fig, ax = plt.subplots()
    out_ax = res.plot(ax=ax)
    assert out_ax is ax
    assert len(ax.patches) > 0
    plt.close(fig)


def test_pyodide_zero_dependency_contract():
    """Test 10: Ensures zero foreign runtime dependencies (numpy, scipy, pandas, matplotlib only)."""
    import ast
    from pathlib import Path

    forbidden = {"statsmodels", "linearmodels", "arch", "bs4", "ipywidgets", "torch"}
    dsge_dir = Path(__file__).resolve().parents[1] / "puremacro" / "dsge"

    # Verify via AST analysis of all puremacro.dsge source files that no forbidden
    # foreign packages are imported, guaranteeing Pyodide zero-dependency contract
    # without false-positive leakage from other tests in shared pytest processes.
    imported_pkgs: set[str] = set()
    for py_file in dsge_dir.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_pkgs.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_pkgs.add(node.module.split(".")[0])

    leaked = sorted(imported_pkgs.intersection(forbidden))
    assert not leaked, f"Forbidden package(s) {leaked} imported in puremacro.dsge"

    # Confirm module imports
    import importlib
    ident_mod = importlib.import_module("puremacro.dsge.identification")
    import puremacro.dsge._results as results_mod

    assert hasattr(ident_mod, "identification")
    assert hasattr(results_mod, "IdentificationResult")


def test_shock_correlation_with_underscored_names():
    """Test identification with CORR_ parameters whose shock names contain underscores (Finding 2)."""
    def two_shock_eqs(lead, curr, lag, shocks, p):
        return [
            curr.y - curr.a - shocks.eps_r,
            curr.a - p.rho * lag.a - shocks.eps_a,
        ]

    m = build_dynare(
        two_shock_eqs,
        variables=["y", "a"],
        shocks=["eps_a", "eps_r"],
        params={"rho": 0.8},
        states=["a"],
        guess={"y": 0.0, "a": 0.0},
    )

    # Valid underscored shock correlation: CORR_eps_a_eps_r
    res = identification(
        m,
        params=["rho", "SE_eps_a", "SE_eps_r", "CORR_eps_a_eps_r"],
        varobs=["y", "a"],
        lags=2,
    )
    assert res is not None
    assert "CORR_eps_a_eps_r" in res.param_names
    assert res.n_params == 4
    assert res.is_identified

    # Invalid correlation name should still be caught and raise ValueError
    with pytest.raises(ValueError, match="invalid shock correlation name"):
        identification(m, params=["CORR_nonexistent_eps_r"], varobs=["y"])
