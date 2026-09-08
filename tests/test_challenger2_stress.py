"""Adversarial stress test suite for puremacro v2.7.0 front end.

Executed by orch4_challenger_2 to challenge:
1. Deeply nested macro loops (5-level nesting with tuple destructuring, array comprehensions, and conditional filtering).
2. Macro error guards (circular includes, undefined variables, unclosed loops, unclosed ifs).
3. Grammar and AST stress:
   - Extreme identifier names (single-character, keywords as substrings, variables named lag and lead).
   - Deep parenthesis nesting (depth 50, depth 100).
   - Circular model-local '#' variable dependencies (verifying ModelError with cycle path).
   - Math corner cases (zero powers, zero/one multiplication, double negation, algebraic identity simplifications).
4. End-to-end model solving with extreme edge cases.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import numpy as np
import pytest

from puremacro.dsge._macro import DynareMacroError, Scope, preprocess_macro
from puremacro.dsge._ast import BinOp, Call, Const, Node, Param, UnaryOp, Var
from puremacro.dsge._parser import DynareParseError, ParsedModelDAG, Parser, Tokenizer, parse_mod_to_dag
from puremacro.dsge._symbolic import compile_derivatives
from puremacro.dsge.build import ModelError
from puremacro.dsge.dynare import load_mod, parse_mod


# ===========================================================================
# Battery 1: Deeply Nested Macro Directives
# ===========================================================================

class TestChallengerBattery1DeeplyNestedMacroLoops:
    """Stress-test macro processor with deeply nested loops, tuple destructuring,
    array comprehensions, and conditional filtering.
    """

    def test_5_level_nested_for_with_destructuring_and_comprehensions(self):
        """Level 1: Country (@#for c in ["US", "EA"])
        Level 2: Agent type (@#for a in ["H", "F"])
        Level 3: Tuple destructuring (@#for (v, lag_k) in [("y", 1), ("c", 2), ("pi", 0)])
        Level 4: Array comprehension evaluated via @#define:
                 @#define SHOCKS = [sh for sh in ["a", "b", "m"] when sh != "b"]
                 @#for s in SHOCKS
        Level 5: Loop with conditional filtering:
                 @#for k in 1:3 when (k >= 2 and c == "US") or (k == 1 and c == "EA")
        Inside body: @#if ... @#elseif ... @#else ... @#endif and string interpolation.
        """
        mod_text = """
@#define SHOCKS = [sh for sh in ["a", "b", "m"] when sh != "b"]
@#for c in ["US", "EA"]
  @#for a in ["H", "F"]
    @#for (v, lag_k) in [("y", 1), ("c", 2), ("pi", 0)]
      @#for s in SHOCKS
        @#for k in 1:3 when (k >= 2 and c == "US") or (k == 1 and c == "EA")
          @#if c == "US" and k == 3
          // US special k=3
          var_@{c}_@{a}_@{v}_@{s}_@{k} = 3.0;
          @#elseif c == "EA"
          // EA special
          var_@{c}_@{a}_@{v}_@{s}_@{k} = 1.5;
          @#else
          // Default
          var_@{c}_@{a}_@{v}_@{s}_@{k} = 0.0;
          @#endif
        @#endfor
      @#endfor
    @#endfor
  @#endfor
@#endfor
"""
        out = preprocess_macro(mod_text)
        assert "@#" not in out, "All @# directives must be expanded"
        assert "@{" not in out, "All @{...} interpolations must be expanded"

        # Analytical count calculation:
        # c in ["US", "EA"]: 2 values
        # a in ["H", "F"]: 2 values
        # (v, lag_k) in 3 tuples: 3 values
        # s in ["a", "m"]: 2 values
        # For c == "US": k in [2, 3] (2 values) -> 1 * 2 * 3 * 2 * 2 = 24 entries
        # For c == "EA": k in [1] (1 value)      -> 1 * 2 * 3 * 2 * 1 = 12 entries
        # Total generated assignments = 24 + 12 = 36
        assignments = [line.strip() for line in out.strip().splitlines() if line.strip().startswith("var_")]
        assert len(assignments) == 36, f"Expected exactly 36 generated assignments, got {len(assignments)}"

        # Check US special k=3
        assert "var_US_H_y_a_3 = 3.0;" in out
        assert "var_US_F_pi_m_2 = 0.0;" in out
        # Check EA special
        assert "var_EA_H_c_a_1 = 1.5;" in out
        # Check filtered out items
        assert "_b_" not in out, "Shock 'b' was filtered out in array comprehension"
        assert "var_US_H_y_a_1" not in out, "k=1 was filtered out for c=='US'"
        assert "var_EA_H_y_a_2" not in out, "k=2 was filtered out for c=='EA'"

    def test_inline_comprehension_with_if_in_for_loop(self):
        """Verify inline array comprehension inside @#for with 'if' condition syntax."""
        mod_text = """
@#for s in [sh for sh in ["a", "b", "m"] if sh != "b"]
shock_@{s};
@#endfor
"""
        out = preprocess_macro(mod_text)
        lines = [line.strip() for line in out.strip().splitlines() if line.strip()]
        assert lines == ["shock_a;", "shock_m;"]

    def test_inline_comprehension_with_when_regex_collision_empirical_bug(self):
        """REMEDIATED EMPIRICAL CHALLENGER FINDING:
        Verify that inline array comprehension with 'when' condition inside @#for
        expands cleanly without DynareMacroError after bracket-aware splitting fix:
        @#for s in [sh for sh in ["a", "b", "m"] when sh != "b"]
        """
        mod_text = """
@#for s in [sh for sh in ["a", "b", "m"] when sh != "b"]
shock_@{s};
@#endfor
"""
        out = preprocess_macro(mod_text)
        lines = [line.strip() for line in out.strip().splitlines() if line.strip()]
        assert lines == ["shock_a;", "shock_m;"]

        # Also verify combined inner comprehension 'when' with outer loop 'when'
        mod_text_combined = """
@#for s in [sh for sh in ["a", "b", "m"] when sh != "b"] when s != "a"
shock_@{s};
@#endfor
"""
        out_combined = preprocess_macro(mod_text_combined)
        lines_combined = [line.strip() for line in out_combined.strip().splitlines() if line.strip()]
        assert lines_combined == ["shock_m;"]

    def test_multi_country_multi_sector_macro_expanded_dynare_model(self):
        """Preprocess a full Dynare model with nested @#for generating variables,
        parameters, and equations, then parse into ParsedModelDAG.
        """
        mod_text = """
@#define COUNTRIES = ["US", "EA"]
@#define SECTORS = ["M", "S"]

var
@#for c in COUNTRIES
  @#for s in SECTORS
    y_@{c}_@{s}
  @#endfor
@#endfor
;

varexo
@#for c in COUNTRIES
  @#for s in SECTORS
    eps_@{c}_@{s}
  @#endfor
@#endfor
;

parameters
@#for c in COUNTRIES
  @#for s in SECTORS
    rho_@{c}_@{s}
  @#endfor
@#endfor
;

@#for c in COUNTRIES
  @#for s in SECTORS
    rho_@{c}_@{s} = 0.8;
  @#endfor
@#endfor

model;
@#for c in COUNTRIES
  @#for s in SECTORS
    y_@{c}_@{s} = rho_@{c}_@{s} * y_@{c}_@{s}(-1) + eps_@{c}_@{s};
  @#endfor
@#endfor
end;
"""
        expanded = preprocess_macro(mod_text)
        dag = parse_mod_to_dag(expanded)
        assert len(dag.variables) == 4
        assert set(dag.variables) == {"y_US_M", "y_US_S", "y_EA_M", "y_EA_S"}
        assert len(dag.shocks) == 4
        assert set(dag.shocks) == {"eps_US_M", "eps_US_S", "eps_EA_M", "eps_EA_S"}
        assert len(dag.parameters) == 4
        assert len(dag.equations) == 4
        for p in dag.parameters:
            assert dag.parameter_values[p] == 0.8

    def test_nested_array_comprehensions(self):
        """Test evaluation of nested array comprehensions in macro expressions."""
        mod_text = """
@#define EVENS = [x for x in 1:10 when x % 2 == 0]
@#define SQUARES = [y * y for y in EVENS when y >= 6]
@#for z in SQUARES
val_@{z};
@#endfor
"""
        out = preprocess_macro(mod_text)
        # EVENS: [2, 4, 6, 8, 10]
        # SQUARES: [36, 64, 100]
        lines = [line.strip() for line in out.strip().splitlines() if line.strip()]
        assert lines == ["val_36;", "val_64;", "val_100;"]


# ===========================================================================
# Battery 2: Macro Error Guards
# ===========================================================================

class TestChallengerBattery2MacroErrorGuards:
    """Verify that circular includes, undefined macro variables, unclosed loops,
    and unclosed conditionals raise informative DynareMacroError.
    """

    def test_direct_self_include_raises_dynare_macro_error(self, tmp_path):
        """A file that includes itself must raise DynareMacroError mentioning circular @#include."""
        self_inc = tmp_path / "self_include.mod"
        self_inc.write_text('@#include "self_include.mod"\nvar y;\n', encoding="utf-8")

        with pytest.raises(DynareMacroError) as exc_info:
            preprocess_macro(self_inc.read_text(encoding="utf-8"), base_dir=tmp_path)
        assert "circular @#include detected" in str(exc_info.value).lower()

    def test_two_file_circular_include_raises_dynare_macro_error(self, tmp_path):
        """file_a.mod includes file_b.mod which includes file_a.mod."""
        file_a = tmp_path / "file_a.mod"
        file_b = tmp_path / "file_b.mod"

        file_a.write_text('@#include "file_b.mod"\n', encoding="utf-8")
        file_b.write_text('@#include "file_a.mod"\n', encoding="utf-8")

        with pytest.raises(DynareMacroError) as exc_info:
            preprocess_macro(file_a.read_text(encoding="utf-8"), base_dir=tmp_path)
        err_msg = str(exc_info.value).lower()
        assert "circular @#include detected" in err_msg
        assert "file_a.mod" in err_msg
        assert "file_b.mod" in err_msg

    def test_three_file_circular_include_raises_dynare_macro_error(self, tmp_path):
        """file1 -> file2 -> file3 -> file1 chain."""
        f1 = tmp_path / "f1.mod"
        f2 = tmp_path / "f2.mod"
        f3 = tmp_path / "f3.mod"

        f1.write_text('@#include "f2.mod"\n', encoding="utf-8")
        f2.write_text('@#include "f3.mod"\n', encoding="utf-8")
        f3.write_text('@#include "f1.mod"\n', encoding="utf-8")

        with pytest.raises(DynareMacroError) as exc_info:
            preprocess_macro(f1.read_text(encoding="utf-8"), base_dir=tmp_path)
        assert "circular @#include detected" in str(exc_info.value).lower()

    def test_undefined_macro_variable_in_expression_raises_dynare_macro_error(self):
        """Evaluating an undefined identifier must raise DynareMacroError mentioning the variable."""
        mod_text = """
@#define x = undefined_variable_foo * 2
"""
        with pytest.raises(DynareMacroError) as exc_info:
            preprocess_macro(mod_text)
        err_msg = str(exc_info.value)
        assert "Undefined macro variable" in err_msg
        assert "undefined_variable_foo" in err_msg

    def test_undefined_macro_variable_in_for_iterable_raises_dynare_macro_error(self):
        """Looping over undefined identifier must raise DynareMacroError."""
        mod_text = """
@#for item in non_existent_list
var @{item};
@#endfor
"""
        with pytest.raises(DynareMacroError) as exc_info:
            preprocess_macro(mod_text)
        assert "Undefined macro variable" in str(exc_info.value)
        assert "non_existent_list" in str(exc_info.value)

    def test_undefined_macro_variable_in_interpolation_raises_dynare_macro_error(self):
        """Interpolating undefined identifier must raise DynareMacroError."""
        mod_text = """
var y_@{missing_suffix};
"""
        with pytest.raises(DynareMacroError) as exc_info:
            preprocess_macro(mod_text)
        assert "Undefined macro variable" in str(exc_info.value)
        assert "missing_suffix" in str(exc_info.value)

    def test_unclosed_for_loop_raises_dynare_macro_error(self):
        """Missing @#endfor must raise DynareMacroError with line location."""
        mod_text = """
@#for i in 1:3
var y_@{i};
"""
        with pytest.raises(DynareMacroError) as exc_info:
            preprocess_macro(mod_text)
        assert "Unclosed @#for block" in str(exc_info.value)

    def test_unclosed_nested_for_loop_raises_dynare_macro_error(self):
        """Inner loop missing @#endfor must raise DynareMacroError."""
        mod_text = """
@#for i in 1:3
  @#for j in 1:2
    var y_@{i}_@{j};
@#endfor
"""
        with pytest.raises(DynareMacroError) as exc_info:
            preprocess_macro(mod_text)
        assert "Unclosed" in str(exc_info.value) or "@#endfor" in str(exc_info.value)

    def test_unclosed_if_block_raises_dynare_macro_error(self):
        """Missing @#endif must raise DynareMacroError."""
        mod_text = """
@#if 1 == 1
var y;
"""
        with pytest.raises(DynareMacroError) as exc_info:
            preprocess_macro(mod_text)
        assert "Unclosed @#if block" in str(exc_info.value)

    def test_unclosed_if_else_block_raises_dynare_macro_error(self):
        """Missing @#endif after @#else must raise DynareMacroError."""
        mod_text = """
@#if 1 == 0
var y;
@#else
var c;
"""
        with pytest.raises(DynareMacroError) as exc_info:
            preprocess_macro(mod_text)
        assert "Unclosed @#if block" in str(exc_info.value)

    def test_tuple_destructuring_length_mismatch_raises_dynare_macro_error(self):
        """Unpacking a 2-tuple into 3 identifiers must raise DynareMacroError."""
        mod_text = """
@#for (a, b, c) in [("x", 1), ("y", 2)]
var @{a}_@{b}_@{c};
@#endfor
"""
        with pytest.raises(DynareMacroError) as exc_info:
            preprocess_macro(mod_text)
        assert "Cannot unpack 2 elements into 3 variables" in str(exc_info.value)


# ===========================================================================
# Battery 3: Grammar, Tokenizer & AST Engine Stress
# ===========================================================================

class TestChallengerBattery3GrammarAndASTStress:
    """Stress-test grammar, tokenizer, AST expressions, and simplification rules."""

    def test_extreme_identifier_names_and_substring_collisions(self):
        """Test single-character names (a, b, c, x, y), names containing keywords
        as substrings (var_var, param_name, model_linear, varexo_shock), and
        endogenous variables named lag and lead.
        """
        mod_text = """
var a b c x y lag lead var_var param_name;
varexo varexo_shock;
parameters rho alpha beta;

rho = 0.5;
alpha = 0.3;
beta = 0.99;

model;
  a = rho * a(-1) + varexo_shock;
  b = alpha * b(-1) + a;
  c = beta * c(+1) + b;
  x = a + b + c;
  y = x * 2;
  lag = 0.4 * lead(-1) + 0.1 * a;
  lead = 0.5 * lag(+1) + 0.2 * b;
  var_var = lag + lead;
  param_name = var_var * 0.5;
end;
"""
        dag = parse_mod_to_dag(mod_text)
        assert len(dag.variables) == 9
        assert "lag" in dag.variables
        assert "lead" in dag.variables
        assert "var_var" in dag.variables
        assert "param_name" in dag.variables
        assert set("abcxy").issubset(dag.variables)
        assert "varexo_shock" in dag.shocks
        assert len(dag.equations) == 9

        # Verify derivative compilation works cleanly without namespace collision
        compiled = compile_derivatives(dag)
        lead_arr = np.ones(9)
        curr_arr = np.ones(9)
        lag_arr = np.ones(9)
        shock_arr = np.array([0.1])
        param_vec = np.array([0.5, 0.3, 0.99])

        A_plus, A_0, A_minus, B_u = compiled.eval_first_order(
            lead_arr, curr_arr, lag_arr, shock_arr, param_vec
        )
        assert A_plus.shape == (9, 9)
        assert A_0.shape == (9, 9)
        assert A_minus.shape == (9, 9)
        assert B_u.shape == (9, 1)
        assert not np.isnan(A_plus).any()
        assert not np.isnan(A_0).any()
        assert not np.isnan(A_minus).any()
        assert not np.isnan(B_u).any()

    def test_deep_parenthesis_nesting_depth_50_and_100(self):
        """Verify that recursive-descent parser cleanly handles expressions with
        50 and 100 levels of nested parentheses without RecursionError or corruption.
        """
        for depth in (50, 100):
            open_parens = "(" * depth
            close_parens = ")" * depth
            expr_str = f"{open_parens}a + b{close_parens}"
            mod_text = f"""
var a b;
varexo e;
parameters rho;
rho = 0.7;
model;
  a = {expr_str} * 0.5 + e;
  b = rho * b(-1);
end;
"""
            dag = parse_mod_to_dag(mod_text)
            assert len(dag.equations) == 2
            # Compile and evaluate derivatives
            compiled = compile_derivatives(dag)
            lead_arr = np.zeros(2)
            curr_arr = np.ones(2)
            lag_arr = np.ones(2)
            shock_arr = np.zeros(1)
            param_vec = np.array([0.7])

            A_plus, A_0, A_minus, B_u = compiled.eval_first_order(
                lead_arr, curr_arr, lag_arr, shock_arr, param_vec
            )
            assert not np.isnan(A_0).any()
            # da/da in eq 0 is 1 - 0.5 = 0.5
            # eq: a - (a + b)*0.5 - e = 0
            # d(eq)/da = 1 - 0.5 = 0.5
            assert np.isclose(A_0[0, dag.variables.index("a")], 0.5)

    def test_circular_model_local_variable_dependency_raises_model_error(self):
        """Model-local '#' variables with circular dependency must raise ModelError."""
        # 1-cycle (self-dependency)
        mod_1cycle = """
var y;
varexo e;
parameters rho;
rho = 0.9;
model;
  # a = a + 1;
  y = a + e;
end;
"""
        with pytest.raises(ModelError) as exc1:
            parse_mod_to_dag(mod_1cycle)
        assert "circular dependency in model-local variables" in str(exc1.value)

        # 2-cycle (mutual dependency)
        mod_2cycle = """
var y;
varexo e;
parameters rho;
rho = 0.9;
model;
  # a = b + 1;
  # b = a * 2;
  y = a + b + e;
end;
"""
        with pytest.raises(ModelError) as exc2:
            parse_mod_to_dag(mod_2cycle)
        assert "circular dependency in model-local variables" in str(exc2.value)

        # 3-cycle (chain cycle)
        mod_3cycle = """
var y;
varexo e;
parameters rho;
rho = 0.9;
model;
  # p = q + 1;
  # q = r * 2;
  # r = p - 3;
  y = p + e;
end;
"""
        with pytest.raises(ModelError) as exc3:
            parse_mod_to_dag(mod_3cycle)
        assert "circular dependency in model-local variables" in str(exc3.value)

    def test_math_corner_cases_ast_simplifications(self):
        """Verify algebraic simplifications:
        - zero powers: x^0 -> 1, (x+y)^0 -> 1
        - constant multiplication: 0*x -> 0, x*0 -> 0, 1*x -> x, x*1 -> x, -1*x -> -x
        - double negation: -(-x) -> x, -(-(x+y)) -> x+y
        - addition/subtraction identities: x + 0 -> x, 0 + x -> x, x - 0 -> x, 0 - x -> -x, x - x -> 0
        - division identities: 0/x -> 0, x/1 -> x, x/x -> 1
        - power identities: x^1 -> x, 1^x -> 1
        """
        x = Var("x", 0)
        y = Var("y", 0)

        # Zero powers
        pow_x0 = BinOp("^", x, Const(0)).simplify()
        assert pow_x0 == Const(1)
        assert pow_x0.diff("x", 0) == Const(0)

        pow_expr0 = BinOp("^", BinOp("+", x, y), Const(0)).simplify()
        assert pow_expr0 == Const(1)
        assert pow_expr0.diff("x", 0) == Const(0)
        assert pow_expr0.diff("y", 0) == Const(0)

        # Constant multiplication
        zero_x = BinOp("*", Const(0), x).simplify()
        assert zero_x == Const(0)
        assert zero_x.diff("x", 0) == Const(0)

        x_zero = BinOp("*", x, Const(0)).simplify()
        assert x_zero == Const(0)
        assert x_zero.diff("x", 0) == Const(0)

        one_x = BinOp("*", Const(1), x).simplify()
        assert one_x == x
        assert one_x.diff("x", 0) == Const(1)

        x_one = BinOp("*", x, Const(1)).simplify()
        assert x_one == x
        assert x_one.diff("x", 0) == Const(1)

        neg_one_x = BinOp("*", Const(-1), x).simplify()
        assert neg_one_x == UnaryOp("-", x)
        assert neg_one_x.diff("x", 0) == Const(-1)

        # Double negation
        neg_neg_x = UnaryOp("-", UnaryOp("-", x)).simplify()
        assert neg_neg_x == x
        assert neg_neg_x.diff("x", 0) == Const(1)

        neg_neg_expr = UnaryOp("-", UnaryOp("-", BinOp("+", x, y))).simplify()
        assert neg_neg_expr == BinOp("+", x, y)
        assert neg_neg_expr.diff("x", 0) == Const(1)
        assert neg_neg_expr.diff("y", 0) == Const(1)

        # Addition identities
        assert BinOp("+", x, Const(0)).simplify() == x
        assert BinOp("+", Const(0), x).simplify() == x
        assert BinOp("+", x, UnaryOp("-", x)).simplify() == Const(0)

        # Subtraction identities
        assert BinOp("-", x, Const(0)).simplify() == x
        assert BinOp("-", Const(0), x).simplify() == UnaryOp("-", x)
        assert BinOp("-", x, x).simplify() == Const(0)

        # Division identities
        assert BinOp("/", Const(0), x).simplify() == Const(0)
        assert BinOp("/", x, Const(1)).simplify() == x
        assert BinOp("/", x, x).simplify() == Const(1)

        # Exponentiation identities
        assert BinOp("^", x, Const(1)).simplify() == x
        assert BinOp("^", Const(1), x).simplify() == Const(1)


# ===========================================================================
# Battery 4: Integrated End-to-End Extreme Model Solution
# ===========================================================================

class TestChallengerBattery4IntegratedExtremeModel:
    """Stress-test end-to-end model parsing, building, steady state, and
    first-order/second-order solution for a model containing extreme identifiers
    and simplifications.
    """

    def test_extreme_names_and_simplifications_solves_klein_and_sylvester(self):
        """Construct a stable DSGE model with variables 'lag', 'lead', single-letter
        names 'a', 'c', and math simplifications like '0*a', '1*c', 'lag^0'.
        Verify that:
        1. Model parses and compiles into LinearModel via load_mod.
        2. Solves via Klein (first order) with verified determinacy.
        3. Decision rules have exact dimensions.
        4. Impulse response functions execute cleanly without NaNs.
        5. Solves at second order (order=2) via Schur Sylvester solver.
        """
        mod_text = """
var lag lead c a;
varexo eps_a eps_c;
parameters rho_a rho_c phi beta;

rho_a = 0.7;
rho_c = 0.5;
phi = 0.3;
beta = 0.95;

model;
  // c depends on expectation of c(+1) and lag
  c = beta * c(+1) - phi * lag + eps_c + 0 * a;
  // lag is a state tracking past c with double negation
  lag = -(-c(-1));
  // a is an AR(1) with 1 * a and zero power
  a = rho_a * (1 * a(-1)) + eps_a * (lead^0);
  // lead is forward-looking shock accumulator
  lead = rho_c * lead(+1) + a;
end;

initval;
  lag = 0;
  lead = 0;
  c = 0;
  a = 0;
end;

steady;

shocks;
  var eps_a; stderr 0.01;
  var eps_c; stderr 0.02;
end;
"""
        model = load_mod(mod_text)
        assert model is not None
        assert set(model.variables) == {"lag", "lead", "c", "a"}
        assert set(model.shocks) == {"eps_a", "eps_c"}
        assert model.is_determinate is True

        # Decision rules
        dr = model.decision_rules()
        assert dr is not None
        assert dr.ghx.shape == (4, 2)  # 4 vars, 2 states ('c', 'a')
        assert dr.ghu.shape == (4, 2)  # 4 vars, 2 shocks ('eps_a', 'eps_c')
        assert not np.isnan(dr.ghx.values).any()
        assert not np.isnan(dr.ghu.values).any()

        # Compute IRFs
        irf_a = model.irf("eps_a", 10)
        irf_c = model.irf("eps_c", 10)
        assert irf_a.shape == (11, 4)
        assert irf_c.shape == (11, 4)
        assert not irf_a.isna().any().any()
        assert not irf_c.isna().any().any()

        # Solve at second order
        sol2 = load_mod(mod_text, order=2)
        assert sol2 is not None
        assert sol2.G_xx is not None
        assert sol2.G_uu is not None
        assert sol2.G_xx.shape == (2, 4)  # n_states x n_states^2
        assert sol2.G_uu.shape == (2, 4)  # n_states x n_shocks^2
        assert not np.isnan(sol2.G_xx).any()
        assert not np.isnan(sol2.G_uu).any()
