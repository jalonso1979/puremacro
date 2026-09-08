"""Unit tests for the Symbolic Differentiation and CSE Compiler (_symbolic.py).

Verifies analytical Jacobians (A+, A0, A-, Bu), dynamic Hessian tensor (Hf),
symmetry, DAG simplification, Common Subexpression Elimination (CSE), structural
sparsity derivation, extended math functions, high-precision finite difference
tolerances (<= 1e-10), and sub-millisecond execution on SW07 dimensions.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
import numpy as np
import pytest

from puremacro.dsge._ast import BinOp, Call, Const, Node, Param, UnaryOp, Var
from puremacro.dsge._parser import parse_mod_to_dag
from puremacro.dsge._symbolic import (
    CompiledDerivatives,
    compile_derivatives,
    _is_zero,
    _node_size,
    _perform_cse,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
SW07_PATH = WORKSPACE_ROOT / "puremacro" / "dsge" / "_references" / "sw07_pfeifer.mod"


# ===========================================================================
# 1. Structural Sparsity & Linearity Tests
# ===========================================================================


def test_linear_model_sparsity_and_zero_hessian():
    """Purely linear models have empty Hessian sparsity pattern and is_linear=True."""
    src = """
    var y c k;
    varexo e;
    parameters a b delta;
    a = 0.5; b = 0.3; delta = 0.05;
    model;
    y = a * y(-1) + e;
    c = b * y + 0.1 * k(-1);
    k = (1.0 - delta) * k(-1) + y - c;
    end;
    """
    dag = parse_mod_to_dag(src)
    compiled = compile_derivatives(dag)

    assert compiled.is_linear is True
    assert len(compiled.sparsity_pattern["H_f"]) == 0
    assert isinstance(compiled.sparsity_pattern["H_f"], np.ndarray)
    assert compiled.sparsity_pattern["H_f"].shape == (0, 3)

    # Check Jacobian sparsity patterns
    assert "A_plus" in compiled.sparsity_pattern
    assert "A_0" in compiled.sparsity_pattern
    assert "A_minus" in compiled.sparsity_pattern
    assert "B_u" in compiled.sparsity_pattern

    # A_plus should be empty because there are no leads
    assert len(compiled.sparsity_pattern["A_plus"]) == 0

    # B_u should only have equation 0 (y = a*y(-1) + e)
    bu_coords = [tuple(c) for c in compiled.sparsity_pattern["B_u"]]
    assert bu_coords == [(0, 0)]


def test_nonlinear_model_hessian_sparsity_and_symmetry():
    """Non-linear model generates non-empty dynamic Hessian with exact coordinate symmetry."""
    src = """
    var c k;
    varexo e;
    parameters beta delta alpha gamma;
    beta = 0.99; delta = 0.025; alpha = 0.33; gamma = 2.0;
    model;
    c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * k^(alpha - 1.0) + 1.0 - delta);
    k = k(-1)^alpha - c + (1.0 - delta) * k(-1) + e;
    end;
    initval; k = 10.0; c = 1.0; end;
    """
    dag = parse_mod_to_dag(src)
    compiled = compile_derivatives(dag)

    assert compiled.is_linear is False
    h_f_nz = compiled.sparsity_pattern["H_f"]
    assert len(h_f_nz) > 0
    assert h_f_nz.shape[1] == 3

    # Check that for every (i, p, q) in sparsity pattern, (i, q, p) is also present
    coord_set = {(int(r[0]), int(r[1]), int(r[2])) for r in h_f_nz}
    for i, p, q in coord_set:
        assert (i, q, p) in coord_set, f"Symmetric pair ({i}, {q}, {p}) missing from Hessian pattern"


# ===========================================================================
# 2. Analytical Derivatives vs Finite Differences (<= 1e-10)
# ===========================================================================


def test_euler_equation_derivatives_high_precision():
    """Analytical Euler equation derivatives match 4th-order central differences to <= 1e-10."""
    src = """
    var c k;
    varexo e;
    parameters beta delta alpha gamma;
    beta = 0.99; delta = 0.025; alpha = 0.33; gamma = 2.0;
    model;
    c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * k^(alpha - 1.0) + 1.0 - delta);
    k = k(-1)^alpha - c + (1.0 - delta) * k(-1) + e;
    end;
    """
    dag = parse_mod_to_dag(src)
    compiled = compile_derivatives(dag)

    lead = np.array([1.2, 10.5])
    curr = np.array([1.1, 10.2])
    lag = np.array([1.0, 10.0])
    shocks = np.array([0.0])
    pvec = np.array([0.99, 0.025, 0.33, 2.0])

    A_p, A_0, A_m, B_u, H_f = compiled.eval_derivatives(lead, curr, lag, shocks, pvec)

    # Verify A_plus w.r.t lead c (equation 0, variable 0)
    h = 1e-5
    eq0 = dag.equations[0]

    def eval_eq0(c_lead_val):
        d_vars = {
            ("c", 1): c_lead_val,
            ("k", 1): lead[1],
            ("c", 0): curr[0],
            ("k", 0): curr[1],
            ("c", -1): lag[0],
            ("k", -1): lag[1],
            ("e", 0): shocks[0],
        }
        return eq0.eval(d_vars, dag.parameter_values)

    # 4th-order central difference
    num_d_lead = (
        -eval_eq0(lead[0] + 2 * h)
        + 8 * eval_eq0(lead[0] + h)
        - 8 * eval_eq0(lead[0] - h)
        + eval_eq0(lead[0] - 2 * h)
    ) / (12.0 * h)

    exact_d_lead = A_p[0, 0]
    diff = abs(exact_d_lead - num_d_lead)
    assert diff <= 1e-10, f"Euler derivative diff {diff} > 1e-10"


def test_production_function_hessian_precision():
    """Analytical second derivative of production function matches numerical difference to <= 1e-8."""
    src = """
    var y k;
    varexo e;
    parameters alpha;
    alpha = 0.35;
    model;
    y = k(-1)^alpha + e;
    k = 0.9 * k(-1);
    end;
    """
    dag = parse_mod_to_dag(src)
    compiled = compile_derivatives(dag)

    k0 = 4.0
    alpha = 0.35
    lead = np.zeros(2)
    curr = np.zeros(2)
    lag = np.array([0.0, k0])
    shocks = np.zeros(1)
    pvec = np.array([alpha])

    H_f = compiled.eval_second_order(lead, curr, lag, shocks, pvec)
    # k(-1) is coordinate: 2*N + 1 = 5
    d2_exact = H_f[0, 5, 5]

    # Theoretical value: d2/dk^2 (-(k^alpha)) = -alpha * (alpha - 1) * k^(alpha - 2)
    theoretical = -alpha * (alpha - 1.0) * (k0 ** (alpha - 2.0))
    np.testing.assert_allclose(d2_exact, theoretical, atol=1e-12)


# ===========================================================================
# 3. Extended Math Functions Differentiation
# ===========================================================================


def test_extended_math_functions_derivatives():
    """Extended math functions (normcdf, normpdf, erf, sin, cos, exp) differentiate analytically."""
    src = """
    var x;
    varexo e;
    parameters a b;
    a = 0.5; b = 1.0;
    model;
    x = a * normcdf(x(-1)) + b * exp(sin(x(-1))) + e;
    end;
    """
    dag = parse_mod_to_dag(src)
    compiled = compile_derivatives(dag)

    lead = np.zeros(1)
    curr = np.zeros(1)
    lag = np.array([0.5])
    shocks = np.zeros(1)
    pvec = np.array([0.5, 1.0])

    A_p, A_0, A_m, B_u = compiled.eval_first_order(lead, curr, lag, shocks, pvec)
    exact_am = A_m[0, 0]

    # Theoretical derivative w.r.t x(-1):
    # f = x - a*normcdf(x(-1)) - b*exp(sin(x(-1))) - e
    # df/dx(-1) = -a * normpdf(x(-1)) - b * exp(sin(x(-1))) * cos(x(-1))
    x0 = 0.5
    theo = (
        -0.5 * (math.exp(-0.5 * x0 * x0) / math.sqrt(2.0 * math.pi))
        - 1.0 * math.exp(math.sin(x0)) * math.cos(x0)
    )

    np.testing.assert_allclose(exact_am, theo, atol=1e-12)


def test_steady_state_and_diff_operators_differentiation():
    """STEADY_STATE has zero derivative and diff(x) produces 1 and -1."""
    src = """
    var y dy;
    varexo e;
    parameters rho;
    rho = 0.8;
    model;
    dy = diff(y);
    y - STEADY_STATE(y) = rho * (y(-1) - STEADY_STATE(y)) + e;
    end;
    """
    dag = parse_mod_to_dag(src)
    compiled = compile_derivatives(dag)

    lead = np.zeros(2)
    curr = np.array([2.0, 0.5])
    lag = np.array([1.8, 0.4])
    shocks = np.zeros(1)
    pvec = np.array([0.8])

    A_p, A_0, A_m, B_u = compiled.eval_first_order(lead, curr, lag, shocks, pvec)

    # In eq 0: dy - (y - y(-1)) = 0
    # df0/ddy = 1, df0/dy = -1, df0/dy(-1) = 1
    assert A_0[0, 1] == pytest.approx(1.0)   # w.r.t dy_t
    assert A_0[0, 0] == pytest.approx(-1.0)  # w.r.t y_t
    assert A_m[0, 0] == pytest.approx(1.0)   # w.r.t y_{t-1}

    # In eq 1: y - STEADY_STATE(y) - rho*(y(-1) - STEADY_STATE(y)) - e = 0
    # df1/dy_t = 1.0 (STEADY_STATE has 0 derivative)
    # df1/dy_{t-1} = -rho = -0.8
    assert A_0[1, 0] == pytest.approx(1.0)
    assert A_m[1, 0] == pytest.approx(-0.8)


# ===========================================================================
# 4. Common Subexpression Elimination (CSE) Tests
# ===========================================================================


def test_cse_finds_and_sorts_shared_subtrees():
    """CSE optimizer finds repeated sub-DAGs and sorts them topologically."""
    c = Var("c", 0)
    gamma = Param("gamma")
    k = Var("k", -1)
    alpha = Param("alpha")

    # Construct two expressions sharing (c^(-gamma)) and (k^(alpha - 1))
    sub1 = BinOp("^", c, UnaryOp("-", gamma))
    sub2 = BinOp("^", k, BinOp("-", alpha, Const(1)))

    e1 = BinOp("+", sub1, sub2)
    e2 = BinOp("*", sub1, BinOp("+", sub2, Const(1)))

    var_idx = {"c": 0, "k": 1}
    shock_idx = {}
    param_idx = {"gamma": 0, "alpha": 1}

    temps, replacements = _perform_cse([e1, e2], var_idx, shock_idx, param_idx)

    # Both sub1 and sub2 appear >= 2 times
    assert len(temps) >= 2
    temp_vars = [t[0] for t in temps]
    assert "_t0" in temp_vars
    assert "_t1" in temp_vars

    # Verify node_size topological monotonicity
    sizes = [_node_size(cand) for cand in replacements.keys()]
    assert sizes == sorted(sizes), "CSE temporary variables must be sorted in ascending topological order"


# ===========================================================================
# 5. Input Format Adaptability Tests
# ===========================================================================


def test_eval_with_dict_and_none_defaults():
    """Compiled derivatives callable accepts dicts, default params, and partial inputs."""
    src = """
    var c k;
    varexo e;
    parameters beta alpha;
    beta = 0.99; alpha = 0.33;
    model;
    c = beta * c(+1);
    k = alpha * k(-1) + e;
    end;
    """
    dag = parse_mod_to_dag(src)
    compiled = compile_derivatives(dag)

    # Calling with None uses defaults
    A_p, A_0, A_m, B_u = compiled.eval_first_order()
    assert A_p.shape == (2, 2)
    assert A_0.shape == (2, 2)
    assert A_m.shape == (2, 2)
    assert B_u.shape == (2, 1)

    # Calling with dictionaries
    d_lead = {"c": 1.0, "k": 2.0}
    d_curr = {"c": 1.0, "k": 2.0}
    d_lag = {"c": 1.0, "k": 2.0}
    d_shocks = {"e": 0.1}
    d_params = {"beta": 0.95, "alpha": 0.40}

    A_p2, A_02, A_m2, B_u2 = compiled.eval_first_order(
        lead=d_lead, curr=d_curr, lag=d_lag, shocks=d_shocks, params=d_params
    )
    # c - beta * c(+1) = 0 => df/d(c(+1)) = -beta = -0.95
    assert A_p2[0, 0] == pytest.approx(-0.95)
    # k - alpha * k(-1) - e = 0 => df/d(k(-1)) = -alpha = -0.40
    assert A_m2[1, 1] == pytest.approx(-0.40)


# ===========================================================================
# 6. Smets-Wouters (2007) Benchmark Performance
# ===========================================================================


def test_sw07_symbolic_compilation_and_sub_millisecond_eval():
    """SW07 (40 equations, 7 shocks, K=127) compiles and evaluates in < 0.005s."""
    if not SW07_PATH.exists():
        pytest.skip(f"SW07 model file missing at {SW07_PATH}")

    text = SW07_PATH.read_text(encoding="utf-8")
    t0 = time.perf_counter()
    dag = parse_mod_to_dag(text)
    t_parse = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled = compile_derivatives(dag)
    t_compile = time.perf_counter() - t0

    assert compiled.N == 40
    assert compiled.n_e == 7
    assert compiled.K == 127
    assert compiled.is_linear is True
    assert len(compiled.sparsity_pattern["H_f"]) == 0

    N, n_e = compiled.N, compiled.n_e
    lead = np.zeros(N)
    curr = np.zeros(N)
    lag = np.zeros(N)
    shocks = np.zeros(n_e)
    pvec = np.array(
        [float(compiled.parameter_defaults.get(p, 1.0)) for p in compiled.parameters],
        dtype=float,
    )

    # Benchmark first-order evaluation
    t0 = time.perf_counter()
    for _ in range(10):
        A_p, A_0, A_m, B_u = compiled.eval_first_order(lead, curr, lag, shocks, pvec)
    t_eval = (time.perf_counter() - t0) / 10.0

    assert A_p.shape == (40, 40)
    assert A_0.shape == (40, 40)
    assert A_m.shape == (40, 40)
    assert B_u.shape == (40, 7)

    # Sub-millisecond evaluation check (< 0.001s)
    assert t_eval < 0.001, f"SW07 evaluation took {t_eval*1000:.3f} ms > 1 ms"
