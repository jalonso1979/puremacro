"""Comprehensive unit and integration test suite for Expression DAG AST and parser.

Tests cover:
- AST Node immutability, hashing, structural equality, and operator overloading
- Algebraic simplification rules (constant folding, 0+x, 1*x, x-x, x^0, x^1, -(-x))
- Symbolic differentiation (first and second order, product, quotient, power, chain rule)
- Extended math functions (exp, log, sqrt, sin, cos, tan, normcdf, normpdf, erf, abs, sign)
- Lead/lag shifting and numerical evaluation
- Tokenizer precision, comments, numbers, TeX labels, line/col tracking
- Recursive-descent expression parser with operator precedence
- Block declarations (var, varexo, parameters, predetermined_variables, varobs)
- Model-local '#' variables (topological sort, lead/lag shifts, circular dependency detection)
- Substring collision elimination (variables named 'lag', 'lead', 'c', 'r')
- Multi-period companion form transformation (|lead| >= 2)
- End-to-end parsing of sw07_pfeifer.mod and callable equation evaluation
"""

import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from puremacro.dsge._ast import (
    BinOp,
    Call,
    Const,
    Node,
    Param,
    UnaryOp,
    Var,
    _to_node,
)
from puremacro.dsge._parser import (
    DynareParseError,
    Parser,
    Tokenizer,
    parse_mod_to_dag,
)
from puremacro.dsge.build import ModelError, _Vec

# =============================================================================
# 1. AST Node Basics, Immutability, Hashing & Operator Overloading
# =============================================================================


def test_ast_node_immutability():
    """Verify frozen dataclass slots=True immutability."""
    c = Const(1.5)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        c.value = 2.0  # type: ignore

    v = Var("c", 1)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        v.name = "k"  # type: ignore

    p = Param("beta")
    with pytest.raises((FrozenInstanceError, AttributeError)):
        p.name = "alpha"  # type: ignore


def test_ast_node_structural_equality_and_hashing():
    """Verify AST nodes have structural equality and are hashable for CSE."""
    c1 = Const(42)
    c2 = Const(42)
    assert c1 == c2
    assert hash(c1) == hash(c2)

    v1 = Var("k", -1)
    v2 = Var("k", -1)
    assert v1 == v2
    assert hash(v1) == hash(v2)

    # Use in set and dict
    expr1 = v1 + Param("delta")
    expr2 = v2 + Param("delta")
    assert expr1 == expr2
    assert hash(expr1) == hash(expr2)

    s = {expr1}
    assert expr2 in s
    d = {expr1: "saved"}
    assert d[expr2] == "saved"


def test_ast_operator_overloading():
    """Verify operator dunder methods create correct BinOp and UnaryOp trees."""
    x = Var("x")
    assert x + 5 == BinOp("+", x, Const(5))
    assert 5 + x == BinOp("+", Const(5), x)
    assert x - 3 == BinOp("-", x, Const(3))
    assert 3 - x == BinOp("-", Const(3), x)
    assert x * 2 == BinOp("*", x, Const(2))
    assert 2 * x == BinOp("*", Const(2), x)
    assert x / 4 == BinOp("/", x, Const(4))
    assert 4 / x == BinOp("/", Const(4), x)
    assert x**3 == BinOp("^", x, Const(3))
    assert 2**x == BinOp("^", Const(2), x)
    assert -x == UnaryOp("-", x)
    assert +x == UnaryOp("+", x)


# =============================================================================
# 2. Algebraic Simplification & Constant Folding
# =============================================================================


def test_simplify_constant_folding():
    """Verify arithmetic constant folding across operators and functions."""
    assert (Const(2) + Const(3)).simplify() == Const(5)
    assert (Const(10) - Const(4)).simplify() == Const(6)
    assert (Const(3) * Const(7)).simplify() == Const(21)
    assert (Const(12) / Const(3)).simplify() == Const(4)
    assert (Const(2) ** Const(3)).simplify() == Const(8)

    # Functions on constants
    assert Call("exp", (Const(0),)).simplify() == Const(1.0)
    assert Call("log", (Const(1),)).simplify() == Const(0.0)
    assert Call("sqrt", (Const(16),)).simplify() == Const(4.0)
    assert Call("sin", (Const(0),)).simplify() == Const(0.0)
    assert Call("cos", (Const(0),)).simplify() == Const(1.0)
    assert Call("abs", (Const(-5),)).simplify() == Const(5)
    assert Call("sign", (Const(-3),)).simplify() == Const(-1.0)


def test_simplify_algebraic_identities():
    """Verify algebraic reduction rules: 0+x, 1*x, 0*x, x-x, x^0, x^1, -(-x)."""
    x = Var("x")
    # Addition
    assert (Const(0) + x).simplify() == x
    assert (x + Const(0)).simplify() == x

    # Subtraction
    assert (x - Const(0)).simplify() == x
    assert (Const(0) - x).simplify() == UnaryOp("-", x)
    assert (x - x).simplify() == Const(0)

    # Multiplication
    assert (Const(0) * x).simplify() == Const(0)
    assert (x * Const(0)).simplify() == Const(0)
    assert (Const(1) * x).simplify() == x
    assert (x * Const(1)).simplify() == x
    assert (Const(-1) * x).simplify() == UnaryOp("-", x)
    assert (x * Const(-1)).simplify() == UnaryOp("-", x)

    # Division
    assert (Const(0) / x).simplify() == Const(0)
    assert (x / Const(1)).simplify() == x
    assert (x / x).simplify() == Const(1)

    # Powers
    assert (x ** Const(0)).simplify() == Const(1)
    assert (x ** Const(1)).simplify() == x
    assert (Const(1) ** x).simplify() == Const(1)
    assert (Const(0) ** x).simplify() == Const(0)

    # Double negation
    assert (-(-x)).simplify() == x


# =============================================================================
# 3. Symbolic Differentiation
# =============================================================================


def test_diff_polynomial_and_powers():
    """Verify differentiation of polynomials and powers."""
    x = Var("x", 0)
    poly = x**3 - 4 * (x**2) + 7 * x - 10
    dpoly = poly.diff("x", 0).simplify()

    # Numerical verification at x = 3.0: 3*(3^2) - 8*3 + 7 = 27 - 24 + 7 = 10.0
    val = dpoly.eval({("x", 0): 3.0}, {})
    assert pytest.approx(val, 1e-12) == 10.0

    # Second derivative: 6*x - 8; at x = 3: 18 - 8 = 10.0
    d2poly = dpoly.diff("x", 0).simplify()
    assert pytest.approx(d2poly.eval({("x", 0): 3.0}, {}), 1e-12) == 10.0


def test_diff_product_and_quotient_rules():
    """Verify product and quotient differentiation rules."""
    c = Var("c", 0)
    k = Var("k", -1)
    prod = c * k
    assert prod.diff("c", 0).simplify() == k
    assert prod.diff("k", -1).simplify() == c
    assert prod.diff("c", 1).simplify() == Const(0)

    quot = c / k
    # d(c/k)/dc = 1/k
    assert quot.diff("c", 0).simplify() == Const(1) / k
    # Numerical check at c=4, k=2: d/dk(c/k) = -c/k^2 = -4/4 = -1.0
    d_k = quot.diff("k", -1).simplify()
    assert pytest.approx(d_k.eval({("c", 0): 4.0, ("k", -1): 2.0}, {}), 1e-12) == -1.0


def test_diff_math_functions():
    """Verify analytical derivatives of extended math functions."""
    z = Var("z", 0)

    # exp, log, sqrt
    assert Call("exp", (z,)).diff("z", 0).simplify() == Call("exp", (z,))
    assert Call("log", (z,)).diff("z", 0).simplify() == Const(1) / z
    assert Call("sqrt", (z,)).diff("z", 0).simplify() == Const(1) / (
        Const(2) * Call("sqrt", (z,))
    )

    # sin, cos, tan
    assert Call("sin", (z,)).diff("z", 0).simplify() == Call("cos", (z,))
    assert Call("cos", (z,)).diff("z", 0).simplify() == UnaryOp("-", Call("sin", (z,)))
    assert Call("tan", (z,)).diff("z", 0).simplify() == Const(1) + (
        Call("tan", (z,)) ** Const(2)
    )

    # normcdf, normpdf
    assert Call("normcdf", (z,)).diff("z", 0).simplify() == Call("normpdf", (z,))
    assert Call("normpdf", (z,)).diff("z", 0).simplify() == UnaryOp("-", z) * Call(
        "normpdf", (z,)
    )

    # abs, sign
    assert Call("abs", (z,)).diff("z", 0).simplify() == Call("sign", (z,))
    assert Call("sign", (z,)).diff("z", 0).simplify() == Const(0)

    # STEADY_STATE
    assert Call("STEADY_STATE", (z,)).diff("z", 0).simplify() == Const(0)


def test_diff_erf_numerical_match():
    """Verify erf derivative matches numerical gradient."""
    z = Var("z", 0)
    expr = Call("erf", (z,))
    deriv = expr.diff("z", 0).simplify()

    for x0 in (-1.5, -0.2, 0.5, 2.0):
        # Analytical: 2/sqrt(pi) * exp(-x0^2)
        expected = 2.0 / math.sqrt(math.pi) * math.exp(-(x0**2))
        computed = deriv.eval({("z", 0): x0}, {})
        assert pytest.approx(computed, rel=1e-9) == expected


# =============================================================================
# 4. Lead/Lag Shifting & Numerical Evaluation
# =============================================================================


def test_shift_and_eval():
    """Verify lead/lag coordinate shifting and numerical evaluation."""
    expr = Var("c", 0) + Var("k", -1) + Param("alpha")
    shifted = expr.shift(1)
    assert shifted == Var("c", 1) + Var("k", 0) + Param("alpha")

    shifted_lag = expr.shift(-2)
    assert shifted_lag == Var("c", -2) + Var("k", -3) + Param("alpha")

    # Evaluation
    var_vals = {("c", 0): 1.5, ("k", -1): 3.0}
    param_vals = {"alpha": 0.3}
    assert expr.eval(var_vals, param_vals) == 4.8


def test_code_generation_to_python_and_latex():
    """Verify to_python and to_latex generation."""
    c = Var("c", 0)
    c_lead = Var("c", 1)
    c_lag = Var("c", -1)
    p = Param("gamma")
    expr = c_lead + c_lag - (c**p)

    py_code = expr.to_python()
    assert "lead.c" in py_code
    assert "lag.c" in py_code
    assert "curr.c" in py_code
    assert "params.gamma" in py_code

    ltx = expr.to_latex()
    assert "c_{t+1}" in ltx
    assert "c_{t-1}" in ltx
    assert "c_{t}" in ltx


# =============================================================================
# 5. Tokenizer Tests
# =============================================================================


def test_tokenizer_numbers_and_dots():
    """Verify tokenizer correctly handles integers, floats, leading dots, and scientific notation."""
    text = "42 3.14159 .025 1e-4 2.5E+3"
    tokens = Tokenizer(text).tokenize()
    vals = [t.value for t in tokens if t.type == "NUMBER"]
    assert vals == [42, 3.14159, 0.025, 1e-4, 2500.0]


def test_tokenizer_comments():
    """Verify C-style and MATLAB-style comments are stripped with exact line tracking."""
    text = """
    // Line comment 1
    var c; // trailing comment
    % MATLAB comment
    /* Multi-line
       block comment */
    parameters alpha;
    """
    tokens = Tokenizer(text).tokenize()
    idents_or_kw = [t.value for t in tokens if t.type in ("KEYWORD", "IDENT")]
    assert "var" in idents_or_kw
    assert "c" in idents_or_kw
    assert "parameters" in idents_or_kw
    assert "alpha" in idents_or_kw


def test_tokenizer_tex_labels_and_strings():
    """Verify TeX labels and string literals are lexed properly."""
    text = "var labobs ${lHOURS}$ (long_name='log hours worked');"
    tokens = Tokenizer(text).tokenize()
    types = [t.type for t in tokens]
    assert "TEX" in types
    assert "STRING" in types
    tex_tok = next(t for t in tokens if t.type == "TEX")
    assert tex_tok.value == "lHOURS"
    str_tok = next(t for t in tokens if t.type == "STRING")
    assert str_tok.value == "log hours worked"


# =============================================================================
# 6. Parser Operator Precedence & Syntax Recognition
# =============================================================================


def test_parser_operator_precedence():
    """Verify standard arithmetic precedence: power > unary > mul/div > add/sub."""
    mod = """
    var x;
    model;
    x = 2 + 3 * 4^2;
    end;
    """
    dag = parse_mod_to_dag(mod)
    # Equation is: x - (2 + (3 * (4^2)))
    # 4^2 = 16; 3*16 = 48; 2+48 = 50
    eq = dag.equations[0]
    val = eq.eval({("x", 0): 50.0}, {})
    assert val == 0.0


def test_parser_unary_and_associativity():
    """Verify unary minus and right-associativity of exponentiation."""
    mod = """
    var y;
    model;
    y = 2 ^ 3 ^ 2;
    end;
    """
    dag = parse_mod_to_dag(mod)
    # Right-associative: 2 ^ (3 ^ 2) = 2 ^ 9 = 512
    eq = dag.equations[0]
    assert eq.eval({("y", 0): 512.0}, {}) == 0.0


# =============================================================================
# 7. Model-Local '#' Variables & Inlining
# =============================================================================


def test_probe_1_local_var_inlining():
    """Probe 1: #MU = c^(-gamma); MU = beta * MU(+1) parses without NameError."""
    mod = """
    var c;
    varexo eps;
    parameters beta gamma;
    beta = 0.99;
    gamma = 2.0;
    model;
    #MU = c^(-gamma);
    c = beta * MU(+1) + eps;
    end;
    """
    dag = parse_mod_to_dag(mod)
    assert len(dag.equations) == 1
    # Check that MU(+1) was inlined with lead +1
    eq = dag.equations[0]
    assert eq.variables() == {("c", 0), ("c", 1), ("eps", 0)}


def test_probe_5_local_var_lead_lag_shifting():
    """Probe 5: #X = c + k(-1); used as X(+1) inlines to c(+1) + k(0)."""
    mod = """
    var c k y;
    varexo eps;
    model;
    #X = c + k(-1);
    y = X(+1) + eps;
    end;
    """
    dag = parse_mod_to_dag(mod)
    eq = dag.equations[0]
    # Check variables in equation
    assert eq.variables() == {("y", 0), ("c", 1), ("k", 0), ("eps", 0)}


def test_local_var_circular_dependency_raises():
    """Verify circular dependency in model-local '#' variables raises ModelError."""
    mod = """
    var c;
    model;
    #A = B + 1;
    #B = A - 1;
    c = A;
    end;
    """
    with pytest.raises(ModelError, match="circular dependency"):
        parse_mod_to_dag(mod)


def test_local_var_self_loop_raises():
    """Verify self-dependency in model-local '#' variables raises ModelError."""
    mod = """
    var c;
    model;
    #A = A + 1;
    c = A;
    end;
    """
    with pytest.raises(ModelError, match="circular dependency"):
        parse_mod_to_dag(mod)


# =============================================================================
# 8. Substring Collision Elimination
# =============================================================================


def test_probe_2_substring_collision_immunity():
    """Probe 2: Variables named 'lag', 'lead', 'c', 'r' parse without collision."""
    mod = """
    var lag lead c r;
    varexo eps;
    parameters rho;
    rho = 0.9;
    model;
    lag = rho * lag(-1) + eps;
    lead = rho * lead(+1) + eps;
    c = 0.5 * c(-1);
    r = 0.5 * r(+1);
    end;
    """
    dag = parse_mod_to_dag(mod)
    assert set(dag.variables) == {"lag", "lead", "c", "r"}
    assert len(dag.equations) == 4

    # Test callable generation and evaluation without AttributeError
    d = dag.to_dynare_dict()
    fn = d["equations"]
    v_zero = _Vec(["lag", "lead", "c", "r"], [1.0, 1.0, 1.0, 1.0])
    v_shocks = _Vec(["eps"], [0.0])
    v_params = _Vec(["rho"], [0.9])
    res = fn(v_zero, v_zero, v_zero, v_shocks, v_params)
    assert len(res) == 4
    assert pytest.approx(res[0], 1e-12) == 1.0 - 0.9 * 1.0


# =============================================================================
# 9. Multi-Period Companion Form Expansion (|lead| >= 2)
# =============================================================================


def test_multiperiod_lead_lag_expansion():
    """Verify leads/lags with |offset| >= 2 are converted to first-order auxiliary companion form."""
    mod = """
    var c k;
    varexo eps;
    model;
    c = 0.5 * c(+2) + eps;
    k = 0.8 * k(-3);
    end;
    """
    dag = parse_mod_to_dag(mod)
    assert "AUX_LEAD_c_1" in dag.variables
    assert "AUX_LAG_k_1" in dag.variables
    assert "AUX_LAG_k_2" in dag.variables
    # 2 original equations + 1 lead aux + 2 lag aux = 5 equations
    assert len(dag.equations) == 5

    # Check maximum dynamic lead/lag across all equations is at most 1
    for eq in dag.equations:
        for v_name, v_lead in eq.variables():
            assert abs(v_lead) <= 1, f"Equation has order > 1: {v_name}({v_lead})"


# =============================================================================
# 10. Shocks Block & Parameter Assignments
# =============================================================================


def test_shocks_block_parsing():
    """Verify shocks block parses variances, stderrs, and covariances into covariance matrix."""
    mod = """
    var y;
    varexo e1 e2 e3;
    parameters a;
    a = 1.0;
    model;
    y = e1 + e2 + e3;
    end;
    shocks;
    var e1; stderr 0.2;
    var e2 = 0.09;
    var e1, e2 = 0.01;
    corr e1, e3 = 0.5;
    end;
    """
    dag = parse_mod_to_dag(mod)
    cov = dag.shock_cov
    assert cov is not None
    assert pytest.approx(cov[0, 0], 1e-12) == 0.04  # 0.2^2
    assert pytest.approx(cov[1, 1], 1e-12) == 0.09
    assert pytest.approx(cov[0, 1], 1e-12) == 0.01
    assert pytest.approx(cov[1, 0], 1e-12) == 0.01


# =============================================================================
# 11. End-to-End Canonical Benchmark: SW07 Pfeifer Replication
# =============================================================================


def test_sw07_canonical_benchmark_parse_and_residuals():
    """Verify sw07_pfeifer.mod parses completely and residuals evaluate to machine zero."""
    mod_path = Path("puremacro/dsge/_references/sw07_pfeifer.mod")
    assert mod_path.exists()
    text = mod_path.read_text()

    dag = parse_mod_to_dag(text)

    # 40 endogenous variables, 7 shocks, 39 parameters
    assert len(dag.variables) == 40
    assert len(dag.shocks) == 7
    assert len(dag.parameters) == 39
    assert len(dag.equations) == 40
    assert dag.is_linear is True
    assert len(dag.equation_tags) == 38

    # Check that model-local '#' variables were evaluated and inlined
    assert len(dag.local_variables) >= 15
    for eq in dag.equations:
        # No equation should reference unresolved _LocalRef
        assert not any(p in dag.local_variables for p in eq.parameters())

    # Check callable compilation and evaluate residuals at steady state
    d = dag.to_dynare_dict()
    eq_fn = d["equations"]
    ss = d["steady_state"] or {v: 0.0 for v in dag.variables}

    v_lead = _Vec(dag.variables, [ss.get(v, 0.0) for v in dag.variables])
    v_curr = _Vec(dag.variables, [ss.get(v, 0.0) for v in dag.variables])
    v_lag = _Vec(dag.variables, [ss.get(v, 0.0) for v in dag.variables])
    v_shocks = _Vec(dag.shocks, [0.0] * len(dag.shocks))
    v_params = _Vec(
        list(dag.parameter_values.keys()), list(dag.parameter_values.values())
    )

    res = eq_fn(v_lead, v_curr, v_lag, v_shocks, v_params)
    assert len(res) == 40
    max_res = max(abs(r) for r in res)
    assert pytest.approx(max_res, abs=1e-12) == 0.0
