"""Unit test suite for Optimal Discretionary vs Commitment Policy (R1).

Validates:
1. Dennis (2007) policy iteration convergence ||F_{k+1} - F_k||_infty < 10^{-9}.
2. Positive semi-definiteness and Bellman optimality of the Riccati value matrix V.
3. Theoretical inflation bias when y* > 0 and zero inflation bias when y* = 0.
4. Stabilization bias Loss^{disc} - Loss^{comm} > 0 under persistent cost-push shocks.
5. Top-level optimal_policy() dispatch for rule="discretion" and rule="commitment".
6. String quadratic loss parsing with variable weights and target offsets.
7. LinearModel.optimal_policy() convenience method.
8. Full 6-method presentation contract (.summary, .plot, .to_frame, .to_markdown, .to_latex, .to_typst).
9. Robust error handling for invalid rules and unknown instruments.
"""
from __future__ import annotations

import dataclasses
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.build import LinearModel
from puremacro.dsge.policy import (
    discretionary_policy,
    lq_commitment,
    optimal_policy,
)
from puremacro.dsge._results import DiscretionaryPolicyResult, PolicyResult


# ==============================================================================
# Canonical 3-Equation New Keynesian Fixture
# ==============================================================================

@pytest.fixture
def nk_model() -> LinearModel:
    """Canonical 3-equation New Keynesian model with persistent cost-push shock."""
    mod = """
    var y pi r u;
    varexo eps_u;
    parameters beta sigma kappa phi_pi phi_y rho_u;
    beta = 0.99;
    sigma = 1.0;
    kappa = 0.5;
    phi_pi = 1.5;
    phi_y = 0.5;
    rho_u = 0.5;

    model;
    y = y(+1) - (1/sigma)*(r - pi(+1));
    pi = beta*pi(+1) + kappa*y + u;
    r = phi_pi*pi + phi_y*y;
    u = rho_u*u(-1) + eps_u;
    end;

    shocks;
    var eps_u; stderr 1.0;
    end;
    """
    return build_dynare(mod)


# ==============================================================================
# 1. Convergence & Decision Rule Tests
# ==============================================================================

def test_discretionary_policy_convergence_tolerance_1e9(nk_model: LinearModel):
    """Verify Dennis (2007) policy iteration converges with ||F_{k+1} - F_k||_infty < 10^{-9}."""
    res = discretionary_policy(
        nk_model,
        target_vars=["pi", "y"],
        weights={"pi": 1.0, "y": 0.25},
        instruments="r",
        tol=1e-9,
        max_iter=2000,
    )

    assert isinstance(res, DiscretionaryPolicyResult)
    assert isinstance(res, PolicyResult)
    assert res.converged is True
    assert res.diff < 1e-9, f"Convergence diff {res.diff} must be < 1e-9"
    assert 0 < res.iterations < 100, f"Dennis iteration should converge rapidly, took {res.iterations}"

    # Verify feedback matrix F
    assert isinstance(res.F, pd.DataFrame)
    assert "r" in res.F.index
    assert "u" in res.F.columns
    # Reaction on cost-push shock u: around 0.4983 in canonical calibration
    assert 0.45 < res.F.loc["r", "u"] < 0.55

    # Verify policy_rules DataFrame matches F
    assert isinstance(res.policy_rules, pd.DataFrame)
    assert "r" in res.policy_rules.index
    np.testing.assert_allclose(res.F.values, res.policy_rules.values)

    # Verify transition and impact matrices
    assert isinstance(res.transition_matrix, pd.DataFrame)
    assert isinstance(res.impact_matrix, pd.DataFrame)
    assert res.transition.shape == (4, 4)
    assert res.impact.shape == (4, 1)


# ==============================================================================
# 2. Riccati Matrix V & Bellman Optimality Tests
# ==============================================================================

def test_riccati_matrix_psd_and_bellman_optimality(nk_model: LinearModel):
    """Verify Riccati matrix V is positive semi-definite and satisfies Bellman equation."""
    weights = {"pi": 1.0, "y": 0.25}
    res = discretionary_policy(
        nk_model,
        target_vars=["pi", "y"],
        weights=weights,
        instruments="r",
        beta=0.99,
        tol=1e-9,
    )

    V = res.V
    assert isinstance(V, np.ndarray)
    assert V.shape == (len(nk_model.variables), len(nk_model.variables))

    # 1. Symmetry
    np.testing.assert_allclose(V, V.T, atol=1e-10, err_msg="Riccati matrix V must be symmetric")

    # 2. Positive semi-definiteness: all eigenvalues >= 0
    eigs = np.linalg.eigvalsh(V)
    assert np.all(eigs >= -1e-12), f"Min eigenvalue {np.min(eigs)} should be >= 0 (PSD)"

    # 3. Bellman optimality equation: V = G' W G + beta * G' V G
    G = res.transition
    variables = list(nk_model.variables)
    W_full = np.zeros_like(V)
    for v, w in weights.items():
        idx = variables.index(v)
        W_full[idx, idx] = float(w)

    bellman_rhs = G.T @ W_full @ G + 0.99 * G.T @ V @ G
    bellman_error = float(np.max(np.abs(V - bellman_rhs)))
    assert bellman_error < 1e-6, f"Bellman residual {bellman_error:.2e} should be < 1e-6"


# ==============================================================================
# 3. Inflation Bias & Stabilization Bias Tests
# ==============================================================================

def test_inflation_bias_positive_when_target_output_positive(nk_model: LinearModel):
    """Verify inflation bias is strictly positive when y* > 0 and matches theoretical formula."""
    y_star = 0.05
    res = discretionary_policy(
        nk_model,
        target_vars=["pi", "y"],
        weights={"pi": 1.0, "y": 0.25},
        instruments="r",
        beta=0.99,
        y_star=y_star,
    )

    # Theoretical New Keynesian inflation bias:
    # Bias_inf = (kappa * lambda_y) / (lambda_y * (1 - beta) + kappa^2) * y*
    kappa = 0.5
    lambda_y = 0.25
    beta = 0.99
    expected_bias = (kappa * lambda_y) / (lambda_y * (1.0 - beta) + kappa**2) * y_star

    assert res.inflation_bias > 0.0, "Inflation bias must be strictly positive when y* > 0"
    np.testing.assert_allclose(res.inflation_bias, expected_bias, rtol=1e-5)


def test_inflation_bias_zero_when_no_target_output(nk_model: LinearModel):
    """Verify inflation bias is exactly zero when y* = 0."""
    res = discretionary_policy(
        nk_model,
        target_vars=["pi", "y"],
        weights={"pi": 1.0, "y": 0.25},
        instruments="r",
        beta=0.99,
        y_star=0.0,
    )
    assert res.inflation_bias == 0.0


def test_stabilization_bias_strictly_positive(nk_model: LinearModel):
    """Verify stabilization bias Loss^{disc} - Loss^{comm} > 0 under persistent shocks."""
    weights = {"pi": 1.0, "y": 0.25}
    res_disc = discretionary_policy(
        nk_model,
        target_vars=["pi", "y"],
        weights=weights,
        instruments="r",
        compare_commitment=True,
    )

    assert res_disc.stabilization_bias > 0.0, "Stabilization bias must be strictly positive"
    assert res_disc.commitment_result is not None
    assert isinstance(res_disc.commitment_result, PolicyResult)
    assert res_disc.loss > res_disc.commitment_result.loss

    expected_bias = float(res_disc.loss - res_disc.commitment_result.loss)
    np.testing.assert_allclose(res_disc.stabilization_bias, expected_bias, atol=1e-10)


# ==============================================================================
# 4. Top-Level optimal_policy Dispatch Tests
# ==============================================================================

def test_optimal_policy_dispatch_discretion(nk_model: LinearModel):
    """Verify optimal_policy with rule='discretion' returns DiscretionaryPolicyResult."""
    res = optimal_policy(
        nk_model,
        loss={"pi": 1.0, "y": 0.25},
        rule="discretion",
        instruments="r",
        y_star=0.04,
    )
    assert isinstance(res, DiscretionaryPolicyResult)
    assert res.regime == "discretion"
    assert res.inflation_bias > 0.0
    assert res.stabilization_bias > 0.0
    assert res.multipliers == ()


def test_optimal_policy_dispatch_commitment(nk_model: LinearModel):
    """Verify optimal_policy with rule='commitment' returns PolicyResult with multipliers."""
    res = optimal_policy(
        nk_model,
        loss={"pi": 1.0, "y": 0.25},
        rule="commitment",
        instruments="r",
    )
    assert isinstance(res, PolicyResult)
    assert res.regime == "commitment"
    assert len(res.multipliers) > 0


def test_optimal_policy_string_loss_parsing(nk_model: LinearModel):
    """Verify string quadratic loss parsing in optimal_policy."""
    # Simple quadratic expression
    res = optimal_policy(nk_model, loss="pi^2 + 0.25 * y^2", rule="discretion", instruments="r")
    assert isinstance(res, DiscretionaryPolicyResult)
    assert res.weights["pi"] == 1.0
    assert res.weights["y"] == 0.25

    # Quadratic expression with explicit target offset (y - 0.05)^2
    res_tgt = optimal_policy(
        nk_model,
        loss="pi^2 + 0.25 * (y - 0.05)^2",
        rule="discretion",
        instruments="r",
    )
    assert res_tgt.inflation_bias > 0.0
    assert np.isclose(res_tgt.inflation_bias, 0.024752, atol=1e-4)


def test_linearmodel_optimal_policy_method(nk_model: LinearModel):
    """Verify model.optimal_policy() convenience method on LinearModel."""
    res_disc = nk_model.optimal_policy(loss={"pi": 1.0, "y": 0.25}, rule="discretion", instruments="r")
    assert isinstance(res_disc, DiscretionaryPolicyResult)
    assert res_disc.converged is True

    res_comm = nk_model.optimal_policy(loss={"pi": 1.0, "y": 0.25}, rule="commitment", instruments="r")
    assert isinstance(res_comm, PolicyResult)
    assert res_comm.regime == "commitment"


# ==============================================================================
# 5. Presentation Contract Tests
# ==============================================================================

def test_discretionary_policy_result_presentation_contract(nk_model: LinearModel):
    """Verify DiscretionaryPolicyResult satisfies the 6-method presentation contract."""
    res = discretionary_policy(
        nk_model,
        target_vars=["pi", "y"],
        weights={"pi": 1.0, "y": 0.25},
        instruments="r",
        y_star=0.05,
    )

    # 1. to_frame()
    df = res.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert "r" in df.index

    # 2. summary()
    s = res.summary()
    assert isinstance(s, str)
    assert "OPTIMAL POLICY REGIME: DISCRETION" in s
    assert "POLICY REACTION FUNCTIONS" in s
    assert "pi (w=1.0)" in s
    assert "Inflation bias" in s
    assert "Stabilization bias" in s

    # 3. to_markdown()
    md = res.to_markdown()
    assert isinstance(md, str)
    assert "|" in md

    # 4. to_latex()
    ltx = res.to_latex()
    assert isinstance(ltx, str)
    assert "tabular" in ltx

    # 5. to_typst()
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#table" in typ

    # 6. plot() standard and compare_commitment
    ax = res.plot(periods=8)
    assert ax is not None

    ax_comp = res.plot(periods=8, compare_commitment=True)
    assert ax_comp is not None


# ==============================================================================
# 6. Error Handling Tests
# ==============================================================================

def test_optimal_policy_invalid_rule_raises_value_error(nk_model: LinearModel):
    """Verify invalid policy rule raises ValueError."""
    with pytest.raises(ValueError, match="Unknown policy rule"):
        optimal_policy(nk_model, loss={"pi": 1.0}, rule="unknown_regime", instruments="r")


def test_optimal_policy_unknown_instrument_raises_value_error(nk_model: LinearModel):
    """Verify unknown instrument name raises ValueError."""
    with pytest.raises(ValueError, match="not found in model variables"):
        optimal_policy(nk_model, loss={"pi": 1.0}, rule="discretion", instruments="nonexistent_var")


def test_discretionary_policy_warning_on_non_convergence(nk_model: LinearModel):
    """Verify warning is emitted when max_iter is too small to converge."""
    with pytest.warns(UserWarning, match="Dennis .* iteration did not converge"):
        res = discretionary_policy(
            nk_model,
            target_vars=["pi", "y"],
            weights={"pi": 1.0, "y": 0.25},
            instruments="r",
            max_iter=2,
            tol=1e-15,
        )
        assert res.converged is False


# ==============================================================================
# 7. Non-Default Parameter Calibration & High Persistence Stress Tests
# ==============================================================================

def test_inflation_bias_with_non_default_kappa():
    """Verify dynamic inflation bias computation for non-default kappa values (kappa=0.2, 0.05)."""
    mod_base = """
    var y pi r u;
    varexo eps_u;
    parameters beta sigma kappa phi_pi phi_y rho_u;
    beta = 0.99; sigma = 1.0; kappa = {kappa_val}; phi_pi = 1.5; phi_y = 0.5; rho_u = 0.5;
    model;
    y = y(+1) - (1/sigma)*(r - pi(+1));
    pi = beta*pi(+1) + kappa*y + u;
    r = phi_pi*pi + phi_y*y;
    u = rho_u*u(-1) + eps_u;
    end;
    shocks; var eps_u; stderr 1.0; end;
    """

    # 1. Test kappa = 0.2: theoretical bias is (0.2 * 0.25) / (0.25 * 0.01 + 0.04) * 0.05 = 0.05882353
    m_02 = build_dynare(mod_base.format(kappa_val="0.2"))
    res_02 = discretionary_policy(
        m_02, target_vars=["pi", "y"], weights={"pi": 1.0, "y": 0.25}, instruments="r", y_star=0.05
    )
    expected_02 = (0.2 * 0.25) / (0.25 * (1.0 - 0.99) + 0.2**2) * 0.05
    np.testing.assert_allclose(res_02.inflation_bias, expected_02, rtol=1e-5)
    np.testing.assert_allclose(res_02.inflation_bias, 0.058824, atol=1e-5)

    # 2. Test kappa = 0.05: theoretical bias is (0.05 * 0.25) / (0.25 * 0.01 + 0.05**2) * 0.05 = 0.125
    m_005 = build_dynare(mod_base.format(kappa_val="0.05"))
    res_005 = discretionary_policy(
        m_005, target_vars=["pi", "y"], weights={"pi": 1.0, "y": 0.25}, instruments="r", y_star=0.05
    )
    expected_005 = (0.05 * 0.25) / (0.25 * (1.0 - 0.99) + 0.05**2) * 0.05
    np.testing.assert_allclose(res_005.inflation_bias, expected_005, rtol=1e-5)
    np.testing.assert_allclose(res_005.inflation_bias, 0.125000, atol=1e-5)

    # 3. Test explicit kappa argument override on m_02
    res_override = discretionary_policy(
        m_02, target_vars=["pi", "y"], weights={"pi": 1.0, "y": 0.25}, instruments="r", y_star=0.05, kappa=0.05
    )
    np.testing.assert_allclose(res_override.inflation_bias, expected_005, rtol=1e-5)

    # 4. Test kwargs override via optimal_policy
    res_opt = optimal_policy(
        m_02, loss={"pi": 1.0, "y": 0.25}, rule="discretion", instruments="r", y_star=0.05, kappa=0.05
    )
    np.testing.assert_allclose(res_opt.inflation_bias, expected_005, rtol=1e-5)

    # 5. Test structural Phillips curve extraction from dynamic Jacobian when parameters are unassigned
    m_unassigned = dataclasses.replace(m_02, _params={})
    res_jac = discretionary_policy(
        m_unassigned, target_vars=["pi", "y"], weights={"pi": 1.0, "y": 0.25}, instruments="r", y_star=0.05
    )
    np.testing.assert_allclose(res_jac.inflation_bias, expected_02, rtol=1e-5)


def test_bellman_residual_high_persistence():
    """Verify Riccati continuation value V satisfies Bellman equation under high persistence (rho_u=0.95)."""
    mod_high_rho = """
    var y pi r u;
    varexo eps_u;
    parameters beta sigma kappa phi_pi phi_y rho_u;
    beta = 0.99; sigma = 1.0; kappa = 0.5; phi_pi = 1.5; phi_y = 0.5; rho_u = 0.95;
    model;
    y = y(+1) - (1/sigma)*(r - pi(+1));
    pi = beta*pi(+1) + kappa*y + u;
    r = phi_pi*pi + phi_y*y;
    u = rho_u*u(-1) + eps_u;
    end;
    shocks; var eps_u; stderr 1.0; end;
    """
    m = build_dynare(mod_high_rho)
    weights = {"pi": 1.0, "y": 0.25}
    res = discretionary_policy(m, target_vars=["pi", "y"], weights=weights, instruments="r", beta=0.99)

    G = res.transition
    V = res.V
    variables = list(m.variables)
    W_full = np.zeros_like(V)
    for var, wt in weights.items():
        idx = variables.index(var)
        W_full[idx, idx] = float(wt)

    # Verify Bellman equation: V = G' W G + beta * G' V G
    bellman_rhs = G.T @ W_full @ G + 0.99 * G.T @ V @ G
    residual = float(np.max(np.abs(V - bellman_rhs)))
    assert residual < 1e-6, f"Bellman residual {residual:.2e} must be < 1e-6 at rho_u=0.95"

    # Verify positive semi-definiteness
    eigs = np.linalg.eigvalsh(V)
    assert np.all(eigs >= -1e-12), f"Min eigenvalue {np.min(eigs)} should be >= 0 (PSD)"
