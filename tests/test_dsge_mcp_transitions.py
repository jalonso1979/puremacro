"""Tests for DSGE Deterministic Transitions & Mixed Complementarity Problems (MCP via Semismooth Newton).

Validates:
1. Grammar & Parsing of M2 blocks: histval, endval, varexo_det.
2. Static steady-state solver: find_steady_state for permanent exogenous shocks.
3. Deterministic transitions between distinct steady states with y_0 = y_init and y_{T+1} = y_end.
4. Semismooth Newton MCP solver with Fischer-Burmeister complementarity:
   - Lower bound (ZLB: y >= lb)
   - Pinned rate (lb == ub)
   - Never-binding bounds (y >= -inf matches unconstrained)
   - Two-sided bounds (rate collar: lb <= y <= ub)
5. Anticipated shocks (varexo_det) vs rolling surprise shocks (simulate_surprise_shocks).
6. MCPResult presentation methods: summary(), plot(), to_markdown(), to_latex(), to_typst().
7. Integration with LinearModel and load_mod.
"""
from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    LinearModel,
    load_mod,
    parse_mod,
    solve_perfect_foresight,
)
from puremacro.dsge._parser import parse_mod_to_dag
from puremacro.dsge.perfect_foresight import (
    MCPResult,
    PerfectForesightResult,
    find_steady_state,
    simulate_surprise_shocks,
)


@pytest.fixture
def ramsey_model():
    """Deterministic Neoclassical Growth Model.

    Euler equation:
        c_t^(-sigma) = beta * c_{t+1}^(-sigma) * (alpha * A_t * k_t^(alpha-1) + 1 - delta)
    Resource constraint:
        k_t = A_t * k_{t-1}^alpha + (1-delta) * k_{t-1} - c_t
    """
    alpha = 0.33
    beta = 0.96
    delta = 0.10
    sigma = 1.0
    A_ss = 1.0

    r_ss = 1.0 / beta - (1.0 - delta)
    k_ss = (alpha * A_ss / r_ss) ** (1.0 / (1.0 - alpha))
    y_ss_val = A_ss * k_ss**alpha
    c_ss = y_ss_val - delta * k_ss

    def equations_fn(yp, yc, yl, eps):
        c_p, k_p = yp
        c, k = yc
        c_m, k_m = yl
        A = float(eps) if np.ndim(eps) == 0 else float(eps[0])
        euler = c ** (-sigma) - beta * c_p ** (-sigma) * (alpha * A * k ** (alpha - 1.0) + 1.0 - delta)
        res_c = k - (A * k_m**alpha + (1.0 - delta) * k_m - c)
        return [euler, res_c]

    return {
        "equations_fn": equations_fn,
        "y_ss": np.array([c_ss, k_ss]),
        "k_ss": k_ss,
        "c_ss": c_ss,
        "params": dict(alpha=alpha, beta=beta, delta=delta, sigma=sigma, A_ss=A_ss),
    }


@pytest.fixture
def nk_zlb_model():
    """3-equation New Keynesian model with Taylor rule subject to ZLB.

    Variables: y = [pi, x, r]
    Equations:
    1. Phillips Curve:
       pi_t = beta * pi_{t+1} + kappa * x_t  =>  beta * pi_{t+1} + kappa * x_t - pi_t = 0
    2. Dynamic IS:
       x_t = x_{t+1} - (1/sigma) * (r_t - pi_{t+1} - r_nat_t)
       => x_{t+1} - (1/sigma)*(r_t - pi_{t+1} - r_nat_t) - x_t = 0
    3. Monetary Policy Taylor Rule:
       r_t = r_nat_t + phi_pi * pi_t + phi_x * x_t
       => r_nat_t + phi_pi * pi_t + phi_x * x_t - r_t = 0
    Zero lower bound constraint: r_t >= 0.0.
    """
    beta = 0.99
    sigma = 1.0
    kappa = 0.1
    phi_pi = 1.5
    phi_x = 0.5
    r_target = 0.02

    def equations_fn(yp, yc, yl, eps):
        pi_p, x_p, r_p = yp
        pi, x, r = yc
        pi_m, x_m, r_m = yl
        r_nat = float(eps) if np.ndim(eps) == 0 else float(eps[0])

        f1 = beta * pi_p + kappa * x - pi
        f2 = x_p - (1.0 / sigma) * (r - pi_p - r_nat) - x
        f3 = r_nat + phi_pi * pi + phi_x * x - r
        return [f1, f2, f3]

    y_ss = np.array([0.0, 0.0, r_target])
    return {
        "equations_fn": equations_fn,
        "y_ss": y_ss,
        "variable_names": ["pi", "x", "r"],
        "r_target": r_target,
    }


# =========================================================================
# 1. Grammar & Parsing of M2 Blocks
# =========================================================================

def test_grammar_and_parsing_m2_blocks():
    """Verify AST DAG parsing and dictionary generation for histval, endval, varexo_det."""
    mod_text = """
    var y c k;
    varexo e_a;
    varexo_det g_exo tau;

    parameters alpha beta delta;
    alpha = 0.33;
    beta = 0.96;
    delta = 0.10;

    model;
    c = y - delta * k(-1);
    y = k(-1)^alpha;
    k = y - c + (1-delta)*k(-1);
    end;

    histval;
    k(-1) = 2.5;
    y = 0.8;
    end;

    endval;
    k = 4.2;
    c = 1.5;
    end;
    """
    # 1. Test AST DAG parser
    dag = parse_mod_to_dag(mod_text)
    assert "varexo_det" in dir(dag) or hasattr(dag, "varexo_det")
    assert list(dag.varexo_det) == ["g_exo", "tau"]
    assert dag.histval_values["k"] == 2.5 and dag.histval_values["y"] == 0.8
    assert dag.endval_values["k"] == 4.2 and dag.endval_values["c"] == 1.5

    # 2. Test to_dynare_dict
    d = dag.to_dynare_dict()
    assert d["varexo_det"] == ["g_exo", "tau"]
    assert d["histval"]["k"] == 2.5 and d["histval"]["y"] == 0.8
    assert d["endval"]["k"] == 4.2 and d["endval"]["c"] == 1.5

    # 3. Test parse_mod
    parsed = parse_mod(mod_text)
    assert parsed["varexo_det"] == ["g_exo", "tau"]
    assert parsed["histval"]["k"] == 2.5 and parsed["histval"]["y"] == 0.8
    assert parsed["endval"]["k"] == 4.2 and parsed["endval"]["c"] == 1.5



# =========================================================================
# 2. Static Steady-State Solver
# =========================================================================

def test_find_steady_state_static(ramsey_model):
    """Verify find_steady_state computes exact steady state under permanent shock."""
    eqs = ramsey_model["equations_fn"]
    y_ss_base = ramsey_model["y_ss"]
    params = ramsey_model["params"]

    # Technology shifts permanently from A=1.0 to A=1.25
    A_new = 1.25
    r_ss = 1.0 / params["beta"] - (1.0 - params["delta"])
    k_ss_new = (params["alpha"] * A_new / r_ss) ** (1.0 / (1.0 - params["alpha"]))
    y_val_new = A_new * k_ss_new ** params["alpha"]
    c_ss_new = y_val_new - params["delta"] * k_ss_new
    expected_new_ss = np.array([c_ss_new, k_ss_new])

    computed_ss = find_steady_state(
        eqs,
        variables=["c", "k"],
        shocks=["A"],
        exo_values=[A_new],
        guess=y_ss_base,
    )

    np.testing.assert_allclose(computed_ss, expected_new_ss, rtol=1e-8, atol=1e-8)
    res = eqs(computed_ss, computed_ss, computed_ss, A_new)
    assert np.max(np.abs(res)) < 1e-10


# =========================================================================
# 3. Deterministic Transitions Between Distinct Steady States
# =========================================================================

def test_deterministic_transition_distinct_steady_states(ramsey_model):
    """Test dynamic transition between low capital steady state and high capital steady state."""
    eqs = ramsey_model["equations_fn"]
    y_ss_base = ramsey_model["y_ss"]
    params = ramsey_model["params"]

    # Permanent shock: A increases to 1.15 permanently
    A_new = 1.15
    y_ss_end = find_steady_state(
        eqs,
        variables=["c", "k"],
        shocks=["A"],
        exo_values=[A_new],
        guess=y_ss_base,
    )

    # Initial state is the base steady state
    y_init = y_ss_base.copy()
    T = 180
    exo_path = np.full(T, A_new)

    res = solve_perfect_foresight(
        eqs,
        y_init=y_init,
        y_ss=y_ss_end,
        exogenous_path=exo_path,
        n_periods=T,
        variable_names=["c", "k"],
        tol=1e-11,
    )


    assert isinstance(res, PerfectForesightResult)
    assert res.converged is True
    assert res.residual_norm < 1e-9
    assert res.terminal_error < 1e-10

    # Economic validation: capital k_t accumulates smoothly towards new steady state
    k_path = res.path["k"].to_numpy()
    assert k_path[0] > y_init[1]  # starts growing
    assert k_path[-1] == pytest.approx(y_ss_end[1], rel=1e-4)
    # Monotonic accumulation during transition phase
    assert np.all(np.diff(k_path[:120]) > 0)
    assert np.all(np.diff(k_path) >= -1e-12)



# =========================================================================
# 4. Semismooth Newton MCP Solver: Zero Lower Bound (ZLB)
# =========================================================================

def test_semismooth_newton_mcp_zlb_nk(nk_zlb_model):
    """Semismooth Newton solver enforces ZLB r_t >= 0 under deflationary natural rate shock."""
    eqs = nk_zlb_model["equations_fn"]
    y_ss = nk_zlb_model["y_ss"]
    v_names = nk_zlb_model["variable_names"]

    T = 20
    # Natural rate shock drops to -0.04 for 4 periods, causing unconstrained rate to be negative
    r_nat_path = np.full(T, 0.02)
    r_nat_path[0:4] = -0.04

    # 1. Unconstrained solve (rate goes negative)
    res_unc = solve_perfect_foresight(
        eqs,
        y_init=y_ss,
        y_ss=y_ss,
        exogenous_path=r_nat_path,
        n_periods=T,
        variable_names=v_names,
    )
    assert res_unc.converged is True
    assert np.min(res_unc.path["r"]) < -0.01  # violates ZLB

    # 2. Constrained MCP solve with Semismooth Newton
    res_mcp = solve_perfect_foresight(
        eqs,
        y_init=y_ss,
        y_ss=y_ss,
        exogenous_path=r_nat_path,
        n_periods=T,
        variable_names=v_names,
        mcp=True,
        mcp_bounds={"r": (0.0, None)},
    )

    assert isinstance(res_mcp, MCPResult)
    assert res_mcp.converged is True
    assert res_mcp.iterations <= 15
    assert res_mcp.residual_norm < 1e-8
    assert res_mcp.complementarity_residual < 1e-8

    # ZLB is strictly respected
    min_r = np.min(res_mcp.path["r"])
    assert min_r >= -1e-14
    assert np.all(res_mcp.path["r"] >= -1e-14)

    # Binding periods detection (1-indexed)
    assert "r" in res_mcp.binding_periods
    binding_t = res_mcp.binding_periods["r"]
    assert len(binding_t) > 0
    # Periods 1..4 must be binding
    for p in (1, 2, 3, 4):
        assert p in binding_t
        assert res_mcp.path["r"].loc[p] == pytest.approx(0.0, abs=1e-12)


# =========================================================================
# 5. MCP Bound Variants: Pinned Rate & Never-Binding Bounds
# =========================================================================

def test_mcp_pinned_rate_forward_guidance(nk_zlb_model):
    """Test pinned rate bound (lb == ub) enforcing exact policy peg."""
    eqs = nk_zlb_model["equations_fn"]
    y_ss = nk_zlb_model["y_ss"]
    v_names = nk_zlb_model["variable_names"]

    T = 15
    r_nat_path = np.full(T, 0.02)
    pinned_val = 0.01

    res_pinned = solve_perfect_foresight(
        eqs,
        y_init=y_ss,
        y_ss=y_ss,
        exogenous_path=r_nat_path,
        n_periods=T,
        variable_names=v_names,
        mcp=True,
        mcp_bounds={"r": (pinned_val, pinned_val)},
    )

    assert res_pinned.converged is True
    np.testing.assert_allclose(res_pinned.path["r"], pinned_val, atol=1e-12)
    assert res_pinned.binding_periods["r"] == list(range(1, T + 1))


def test_mcp_never_binding_matches_unconstrained(nk_zlb_model):
    """When bounds are not binding, MCP solution matches unconstrained Newton solution."""
    eqs = nk_zlb_model["equations_fn"]
    y_ss = nk_zlb_model["y_ss"]
    v_names = nk_zlb_model["variable_names"]

    T = 15
    r_nat_path = np.full(T, 0.02)
    r_nat_path[0:3] = 0.01  # mild shock, rate stays above 0.01

    res_unc = solve_perfect_foresight(
        eqs,
        y_init=y_ss,
        y_ss=y_ss,
        exogenous_path=r_nat_path,
        n_periods=T,
        variable_names=v_names,
    )

    # Bound r >= -5.0 is far below actual values
    res_mcp = solve_perfect_foresight(
        eqs,
        y_init=y_ss,
        y_ss=y_ss,
        exogenous_path=r_nat_path,
        n_periods=T,
        variable_names=v_names,
        mcp=True,
        mcp_bounds={"r": (-5.0, None)},
    )

    assert res_mcp.converged is True
    assert res_mcp.binding_periods["r"] == []
    pd.testing.assert_frame_equal(res_unc.path, res_mcp.path, check_exact=False, atol=1e-8)


def test_mcp_two_sided_bounds(nk_zlb_model):
    """Test rate collar with two-sided bounds: lb <= r_t <= ub."""
    eqs = nk_zlb_model["equations_fn"]
    y_ss = nk_zlb_model["y_ss"]
    v_names = nk_zlb_model["variable_names"]

    T = 20
    # Natural rate swings: deeply negative then strongly positive
    r_nat_path = np.full(T, 0.02)
    r_nat_path[0:3] = -0.04   # hits lower bound
    r_nat_path[6:9] = 0.08    # hits upper bound

    lb = 0.005
    ub = 0.035

    res_collar = solve_perfect_foresight(
        eqs,
        y_init=y_ss,
        y_ss=y_ss,
        exogenous_path=r_nat_path,
        n_periods=T,
        variable_names=v_names,
        mcp=True,
        mcp_bounds={"r": (lb, ub)},
    )

    assert res_collar.converged is True
    r_vals = res_collar.path["r"].to_numpy()
    assert np.all(r_vals >= lb - 1e-12)
    assert np.all(r_vals <= ub + 1e-12)

    binding_t = res_collar.binding_periods["r"]
    assert 1 in binding_t
    assert 7 in binding_t


# =========================================================================
# 6. Anticipated Shocks vs Rolling Surprise Shocks
# =========================================================================

def test_anticipated_vs_surprise_shocks(ramsey_model):
    """Compare anticipated future shock against rolling surprise shock."""
    eqs = ramsey_model["equations_fn"]
    y_ss = ramsey_model["y_ss"]
    v_names = ["c", "k"]

    T = 30
    # Shock occurs at t=6: A jumps from 1.0 to 1.10
    shock_t6 = 5  # 0-based index 5 = period 6
    exo_path = np.ones(T)
    exo_path[shock_t6:] = 1.10

    # 1. Anticipated shock: forward-looking agents anticipate at t=1
    res_ant = solve_perfect_foresight(
        eqs,
        y_init=y_ss,
        y_ss=y_ss,
        exogenous_path=exo_path,
        n_periods=T,
        variable_names=v_names,
    )

    # 2. Rolling surprise shock: agents believe A=1.0 until t=6
    surprise_arr = np.zeros(T)
    surprise_arr[shock_t6] = 0.10  # surprise innovation

    res_sur = simulate_surprise_shocks(
        eqs,
        surprise_shocks=surprise_arr + 1.0,
        baseline_shocks=np.ones(T),
        horizon=T,
        y_init=y_ss,
        y_ss=y_ss,
        variable_names=v_names,
    )


    # In period 1..5:
    # Under anticipation, consumption c_t changes before shock hits (Euler expectation)
    # Under surprise, the model remains exactly at initial steady state until t=6
    ant_c = res_ant.path["c"].to_numpy()
    sur_c = res_sur.path["c"].to_numpy()

    assert not np.isclose(ant_c[0], y_ss[0], atol=1e-5)
    np.testing.assert_allclose(sur_c[0:shock_t6], y_ss[0], atol=1e-8)


# =========================================================================
# 7. MCPResult Presentation Methods & Immutability
# =========================================================================

def test_mcp_result_presentation_methods(nk_zlb_model):
    """Verify summary, plot, to_markdown, to_latex, to_typst, and immutability of MCPResult."""
    eqs = nk_zlb_model["equations_fn"]
    y_ss = nk_zlb_model["y_ss"]
    v_names = nk_zlb_model["variable_names"]

    T = 10
    r_nat_path = np.full(T, -0.02)

    res = solve_perfect_foresight(
        eqs,
        y_init=y_ss,
        y_ss=y_ss,
        exogenous_path=r_nat_path,
        n_periods=T,
        variable_names=v_names,
        mcp=True,
        mcp_bounds={"r": (0.0, None)},
    )
    assert isinstance(res, MCPResult)

    # 1. Summary
    s = res.summary()
    assert "MIXED COMPLEMENTARITY PROBLEM (MCP)" in s
    assert "Convergence status" in s
    assert "CONVERGED" in s
    assert "Active constraint periods:" in s
    assert "r:" in s


    # 2. Tables: markdown, latex, typst
    md = res.to_markdown()
    assert "|  t |" in md or "| t |" in md or " r " in md
    assert "|" in md


    ltx = res.to_latex()
    assert "\\begin{tabular}" in ltx
    assert "\\end{tabular}" in ltx

    typ = res.to_typst()
    assert "#table(" in typ

    # 3. Plotting
    fig = res.plot(variables=["r"])
    assert isinstance(fig, Figure)
    plt.close(fig)

    fig_pub = res.plot(style="publication")
    assert isinstance(fig_pub, Figure)
    plt.close(fig_pub)

    # 4. Immutability
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.converged = False  # type: ignore[misc]


# =========================================================================
# 8. Integration with LinearModel and load_mod
# =========================================================================

def test_linear_model_integration():
    """Verify LinearModel methods perfect_foresight and simulate_surprise_shocks."""
    mod_text = """
    var y c k;
    varexo e_a;

    parameters alpha beta delta A_bar;
    alpha = 0.33;
    beta = 0.96;
    delta = 0.10;
    A_bar = 1.0;

    model;
    c = y - delta * k(-1);
    y = A_bar * k(-1)^alpha * exp(e_a);
    k = y - c + (1-delta)*k(-1);
    end;

    initval;
    k = 3.5;
    y = 1.5;
    c = 1.15;
    e_a = 0;
    end;
    steady;
    """
    model = load_mod(mod_text)
    assert isinstance(model, LinearModel)

    # Method dispatch from model
    res_pf = model.perfect_foresight(periods=20, shocks=np.zeros(20))
    assert res_pf.converged is True
    assert len(res_pf.path) == 20

    res_sur = model.simulate_surprise_shocks(surprise_shocks=np.zeros(20), horizon=20)
    assert res_sur.converged is True
    assert len(res_sur.path) == 20


# =========================================================================
# 9. Validation and Error Handling
# =========================================================================

def test_mcp_validation_and_errors(nk_zlb_model):
    """Test validation of bounds, unknown variables, and solver options."""
    eqs = nk_zlb_model["equations_fn"]
    y_ss = nk_zlb_model["y_ss"]
    v_names = nk_zlb_model["variable_names"]

    # Unknown variable in mcp_bounds
    with pytest.raises(ValueError, match="MCP variable 'unknown_var' not in model variables"):
        solve_perfect_foresight(
            eqs,
            y_init=y_ss,
            y_ss=y_ss,
            exogenous_path=np.ones(10),
            n_periods=10,
            variable_names=v_names,
            mcp=True,
            mcp_bounds={"unknown_var": (0.0, None)},
        )

    # Negative periods
    with pytest.raises(ValueError, match="n_periods must be at least 1"):
        solve_perfect_foresight(
            eqs,
            y_init=y_ss,
            y_ss=y_ss,
            exogenous_path=np.ones(10),
            n_periods=0,
            variable_names=v_names,
        )

    # Graceful handling when max_iter=1 on difficult problem
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res_fail = solve_perfect_foresight(
            eqs,
            y_init=y_ss,
            y_ss=y_ss,
            exogenous_path=np.full(20, -0.1),
            n_periods=20,
            variable_names=v_names,
            max_iter=1,
            mcp=True,
            mcp_bounds={"r": (0.0, None)},
        )
        assert res_fail.converged is False
        assert len(w) >= 1
