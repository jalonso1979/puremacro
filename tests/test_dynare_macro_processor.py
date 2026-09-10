"""Comprehensive unit and integration tests for Dynare macro preprocessor.

Validates the full macro-processing surface area in puremacro:
- Directives: @#define, @#undef, macro functions F(args) = EXPR
- Iterations: @#for with ranges (step, ascending, descending), collections, destructuring tuples, and when filters
- Conditionals: @#if, @#elseif, @#elif, @#else, @#endif, @#ifdef, @#ifndef
- File inclusions: @#include with relative paths, dynamic expressions, @#includepath, and circular detection
- Interpolations: @{EXPR} with bracket balancing, formatting of booleans, numbers, and strings
- Arithmetic: Modulo operator % in directives and expressions without comment-stripping collision
- Integration: load_mod() and polymorphic build_dynare() on .mod sources with macro directives
- Realistic Models: Multi-country and multi-sector DSGE models generated via @#for loops
- Error Handling: Syntax errors, unclosed blocks, missing includes, undefined variables
- Architecture: Strict Pyodide 4-package compliance (pure Python stdlib)
"""
from __future__ import annotations

import inspect
from pathlib import Path
import sys
import tempfile
import pytest
import numpy as np

from puremacro.dsge.macro import DynareMacroError, Scope, preprocess_macro
from puremacro.dsge.dynare import build_dynare, load_mod, parse_mod, DynareFeatureError
from puremacro.dsge.build import LinearModel, ModelError


# ===========================================================================
# 1. Public Module Exports and Interface Contract
# ===========================================================================

def test_public_macro_module_reexports():
    """Verify puremacro.dsge.macro exports preprocess_macro, DynareMacroError, and Scope."""
    import puremacro.dsge.macro as macro_mod
    import puremacro.dsge._macro as private_mod

    assert hasattr(macro_mod, "preprocess_macro")
    assert hasattr(macro_mod, "DynareMacroError")
    assert hasattr(macro_mod, "Scope")

    assert macro_mod.preprocess_macro is private_mod.preprocess_macro
    assert macro_mod.DynareMacroError is private_mod.DynareMacroError
    assert macro_mod.Scope is private_mod.Scope
    assert set(macro_mod.__all__) == {"DynareMacroError", "Scope", "preprocess_macro"}


# ===========================================================================
# 2. @#define, @#undef, and Macro Functions
# ===========================================================================

def test_define_primitives_and_types():
    """Verify @#define with integer, float, string, and boolean literals."""
    src = """
@#define INT_VAL = 10
@#define FLOAT_VAL = 2.718
@#define STR_VAL = "EA"
@#define BOOL_T = true
@#define BOOL_F = false
var y_@{STR_VAL};
param = @{INT_VAL} + @{FLOAT_VAL};
flags = @{BOOL_T}, @{BOOL_F};
"""
    out = preprocess_macro(src).strip()
    assert "var y_EA;" in out
    assert "param = 10 + 2.718;" in out
    assert "flags = 1, 0;" in out


def test_define_without_value_defaults_to_one():
    """Verify @#define FLAG without expression defaults to integer 1 (truthy)."""
    src = """
@#define SHOCK_ACTIVE
@#if SHOCK_ACTIVE
varexo eps_active;
@#else
varexo eps_inactive;
@#endif
"""
    out = preprocess_macro(src).strip()
    assert "varexo eps_active;" in out
    assert "eps_inactive" not in out


def test_define_arithmetic_expressions():
    """Verify @#define evaluates expressions with precedence and power."""
    src = """
@#define SUM = 3 + 4 * 2
@#define POW = 2 ^ 3 ^ 2
@#define NEG = -(5 + 2)
v1 = @{SUM};
v2 = @{POW};
v3 = @{NEG};
"""
    out = preprocess_macro(src).strip()
    assert "v1 = 11;" in out
    assert "v2 = 512;" in out
    assert "v3 = -7;" in out


def test_undef_removes_variable():
    """Verify @#undef removes variable so defined() evaluates to false."""
    src = """
@#define TEMP = 42
@#if defined(TEMP)
var before_undef;
@#endif
@#undef TEMP
@#if defined(TEMP)
var after_undef_bad;
@#else
var after_undef_good;
@#endif
"""
    out = preprocess_macro(src).strip()
    assert "var before_undef;" in out
    assert "var after_undef_good;" in out
    assert "after_undef_bad" not in out


def test_define_macro_functions():
    """Verify user-defined macro functions with single, multiple, and zero parameters."""
    src = """
@#define DOUBLE(x) = x * 2
@#define HYPOT(a, b) = sqrt(a^2 + b^2)
@#define DEFAULT_VAL() = 100
val1 = @{DOUBLE(21)};
val2 = @{HYPOT(3, 4)};
val3 = @{DEFAULT_VAL()};
"""
    out = preprocess_macro(src).strip()
    assert "val1 = 42;" in out
    assert "val2 = 5;" in out
    assert "val3 = 100;" in out


def test_macro_function_arity_mismatch_raises():
    """Verify calling macro function with incorrect number of arguments raises DynareMacroError."""
    src = """
@#define ADD3(a, b, c) = a + b + c
val = @{ADD3(1, 2)};
"""
    with pytest.raises(DynareMacroError) as exc_info:
        preprocess_macro(src)
    assert "expects 3 arguments, got 2" in str(exc_info.value)


def test_redefine_variable_in_same_scope():
    """Verify macro variable can be updated/redefined in the same scope."""
    src = """
@#define X = 10
first = @{X};
@#define X = 20
second = @{X};
"""
    out = preprocess_macro(src).strip()
    assert "first = 10;" in out
    assert "second = 20;" in out


# ===========================================================================
# 3. @#for Iteration Loops (Ranges, Arrays, Tuples, Filters, Scoping)
# ===========================================================================

def test_for_range_ascending():
    """Verify @#for iterating over standard ascending range START:END."""
    src = """
@#for i in 1:4
var y_@{i};
@#endfor
"""
    out = preprocess_macro(src).strip()
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    assert lines == ["var y_1;", "var y_2;", "var y_3;", "var y_4;"]


def test_for_range_with_step():
    """Verify @#for iterating over range with positive custom step START:STEP:END."""
    src = """
@#for s in 2:2:8
sector_@{s}
@#endfor
"""
    out = preprocess_macro(src).strip()
    tokens = out.split()
    assert tokens == ["sector_2", "sector_4", "sector_6", "sector_8"]


def test_for_range_descending():
    """Verify @#for iterating over descending range with negative step."""
    src = """
@#for lag in 3:-1:1
lag_@{lag}
@#endfor
"""
    out = preprocess_macro(src).strip()
    assert out.split() == ["lag_3", "lag_2", "lag_1"]


def test_for_range_empty():
    """Verify @#for with ascending bounds where start > end executes zero iterations."""
    src = """
header;
@#for i in 5:2
var bad_@{i};
@#endfor
footer;
"""
    out = preprocess_macro(src).strip()
    assert "bad" not in out
    assert "header;" in out and "footer;" in out


def test_for_range_zero_step_raises():
    """Verify @#for with step 0 raises DynareMacroError."""
    src = """
@#for i in 1:0:5
var x;
@#endfor
"""
    with pytest.raises(DynareMacroError) as exc_info:
        preprocess_macro(src)
    assert "step cannot be 0" in str(exc_info.value).lower()


def test_for_array_collections():
    """Verify @#for iterating over list of strings."""
    src = """
@#for c in ["US", "EA", "UK", "JP"]
var r_@{c};
@#endfor
"""
    out = preprocess_macro(src).strip()
    expected = ["var r_US;", "var r_EA;", "var r_UK;", "var r_JP;"]
    assert [line.strip() for line in out.splitlines() if line.strip()] == expected


def test_for_destructuring_tuples():
    """Verify @#for destructuring tuples with and without surrounding parentheses."""
    src = """
@#for (country, code) in [("US", 840), ("DE", 276), ("FR", 250)]
country_@{country} = @{code};
@#endfor
@#for s, w in [(1, 0.6), (2, 0.4)]
sector_@{s}_weight = @{w};
@#endfor
"""
    out = preprocess_macro(src).strip()
    assert "country_US = 840;" in out
    assert "country_DE = 276;" in out
    assert "country_FR = 250;" in out
    assert "sector_1_weight = 0.6;" in out
    assert "sector_2_weight = 0.4;" in out


def test_for_when_filter():
    """Verify @#for ... when COND filters iterations based on predicate."""
    src = """
@#for n in 1:8 when n > 5
high_@{n}
@#endfor
"""
    out = preprocess_macro(src).strip()
    assert out.split() == ["high_6", "high_7", "high_8"]


def test_for_destructuring_with_when_filter():
    """Verify @#for destructuring combined with when condition."""
    src = """
@#for (c, tier) in [("US", 1), ("EA", 1), ("CA", 2), ("MX", 3)] when tier <= 2
member_@{c}_tier_@{tier};
@#endfor
"""
    out = preprocess_macro(src).strip()
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    assert lines == ["member_US_tier_1;", "member_EA_tier_1;", "member_CA_tier_2;"]
    assert "MX" not in out


def test_for_nested_loops():
    """Verify nested @#for loops generate Cartesian combinations."""
    src = """
@#for c in ["US", "EA"]
@#for s in 1:2
var y_@{c}_s@{s};
@#endfor
@#endfor
"""
    out = preprocess_macro(src).strip()
    expected = ["var y_US_s1;", "var y_US_s2;", "var y_EA_s1;", "var y_EA_s2;"]
    assert [line.strip() for line in out.splitlines() if line.strip()] == expected


def test_for_lexical_scoping_restoration():
    """Verify loop variable does not leak or overwrite outer variables after loop exits."""
    src = """
@#define i = 999
@#for i in 1:3
inner_@{i}
@#endfor
outer_@{i}
"""
    out = preprocess_macro(src).strip()
    assert "outer_999" in out


def test_for_destructuring_mismatch_raises():
    """Verify unpacking arity mismatch raises DynareMacroError."""
    src = """
@#for (a, b) in [(1, 2, 3)]
val = @{a};
@#endfor
"""
    with pytest.raises(DynareMacroError) as exc_info:
        preprocess_macro(src)
    assert "Cannot unpack" in str(exc_info.value)


# ===========================================================================
# 4. Conditionals (@#if, @#elseif/@#elif, @#else, @#endif, @#ifdef, @#ifndef)
# ===========================================================================

def test_if_else_branching():
    """Verify @#if / @#else basic branching."""
    src = """
@#define COND = true
@#if COND
branch_true;
@#else
branch_false;
@#endif
@#define COND = false
@#if COND
branch_true_2;
@#else
branch_false_2;
@#endif
"""
    out = preprocess_macro(src).strip()
    assert "branch_true;" in out
    assert "branch_false;" not in out
    assert "branch_false_2;" in out
    assert "branch_true_2;" not in out


def test_if_elseif_and_elif_chain():
    """Verify @#if / @#elseif / @#elif / @#else conditional cascades."""
    src = """
@#define REGIME = 2
@#if REGIME == 1
regime_one;
@#elseif REGIME == 2
regime_two;
@#elif REGIME == 3
regime_three;
@#else
regime_other;
@#endif

@#define REGIME = 3
@#if REGIME == 1
r1;
@#elif REGIME == 3
r3;
@#else
r_other;
@#endif
"""
    out = preprocess_macro(src).strip()
    assert "regime_two;" in out
    assert "regime_one;" not in out
    assert "regime_three;" not in out
    assert "regime_other;" not in out
    assert "r3;" in out
    assert "r1;" not in out


def test_ifdef_and_ifndef():
    """Verify @#ifdef and @#ifndef directives."""
    src = """
@#define FEATURE_A
@#ifdef FEATURE_A
feature_a_enabled;
@#endif
@#ifdef FEATURE_B
feature_b_enabled;
@#endif
@#ifndef FEATURE_B
feature_b_disabled;
@#endif
"""
    out = preprocess_macro(src).strip()
    assert "feature_a_enabled;" in out
    assert "feature_b_disabled;" in out
    assert "feature_b_enabled;" not in out


def test_inactive_branch_not_evaluated():
    """Verify inactive conditional branches ignore undefined variables and syntax."""
    src = """
@#if false
var @{UNDEFINED_VARIABLE_XYZ};
@#error "This should never be triggered"
@#else
var valid_branch;
@#endif
"""
    out = preprocess_macro(src).strip()
    assert "var valid_branch;" in out


def test_short_circuiting_logic():
    """Verify boolean && and || short-circuit during conditional evaluation."""
    src = """
@#define X = 1
@#if X == 1 || UNDEFINED_VAR == 2
passed_or;
@#endif
@#if X == 0 && UNDEFINED_VAR == 2
failed_and;
@#else
passed_and;
@#endif
"""
    out = preprocess_macro(src).strip()
    assert "passed_or;" in out
    assert "passed_and;" in out
    assert "failed_and;" not in out


def test_truthiness_values():
    """Verify Dynare truthiness conventions (0, empty sequences, empty strings are falsy)."""
    src = """
@#if 0
bad_zero;
@#endif
@#if 42
good_int;
@#endif
@#if ""
bad_empty_str;
@#endif
@#if "hello"
good_str;
@#endif
@#if []
bad_empty_arr;
@#endif
@#if [1]
good_arr;
@#endif
"""
    out = preprocess_macro(src).strip()
    assert "good_int;" in out
    assert "good_str;" in out
    assert "good_arr;" in out
    assert "bad_" not in out


# ===========================================================================
# 5. Modulo Operator % in Directives (Crucial Bug Regression)
# ===========================================================================

def test_modulo_operator_in_define():
    """Verify % evaluates as modulo in @#define."""
    src = """
@#define MOD1 = 17 % 5
@#define MOD2 = 24 % 7
res1 = @{MOD1};
res2 = @{MOD2};
"""
    out = preprocess_macro(src).strip()
    assert "res1 = 2;" in out
    assert "res2 = 3;" in out


def test_modulo_operator_in_if_directive():
    """Verify @#if with modulo % operator evaluates both even and odd branches correctly."""
    src = """
@#for i in 1:4
@#if i % 2 == 0
even_@{i};
@#else
odd_@{i};
@#endif
@#endfor
"""
    out = preprocess_macro(src).strip()
    tokens = [t.strip() for t in out.splitlines() if t.strip()]
    assert tokens == ["odd_1;", "even_2;", "odd_3;", "even_4;"]


def test_modulo_operator_in_for_when():
    """Verify % operator inside @#for ... when filter."""
    src = """
@#for i in 1:6 when i % 2 == 1
odd_num_@{i};
@#endfor
"""
    out = preprocess_macro(src).strip()
    tokens = [t.strip() for t in out.splitlines() if t.strip()]
    assert tokens == ["odd_num_1;", "odd_num_3;", "odd_num_5;"]


def test_modulo_by_zero_raises():
    """Verify modulo by zero raises DynareMacroError."""
    with pytest.raises(DynareMacroError) as exc_info:
        preprocess_macro("@#define X = 10 % 0")
    assert "modulo by zero" in str(exc_info.value).lower()


def test_parse_mod_modulo_preservation_regression():
    """Regression test: parse_mod() must not strip modulo % in @# statements.

    Previously, _remove_comments() executed before preprocess_macro(), stripping
    '% 2 == 0' as a MATLAB comment and corrupting '@#if i % 2 == 0' into '@#if i '.
    """
    mod_text = """
@#for i in 1:2
@#if i % 2 == 0
var even_@{i};
@#else
var odd_@{i};
@#endif
@#endfor
varexo eps;
parameters a;
a = 0.5;
model(linear);
odd_1 = a*odd_1(-1) + eps;
even_2 = a*even_2(-1) + eps;
end;
initval;
odd_1 = 0;
even_2 = 0;
end;
"""
    parsed = parse_mod(mod_text)
    assert parsed["variables"] == ["odd_1", "even_2"]


# ===========================================================================
# 6. File Inclusions (@#include, @#includepath, and Circular Detection)
# ===========================================================================

def test_include_relative_path(tmp_path: Path):
    """Verify @#include inlines relative file contents with shared scope."""
    inc_file = tmp_path / "params.inc"
    inc_file.write_text(
        "@#define BETA = 0.99\n"
        "@#define SIGMA = 2.0\n"
        "parameters beta, sigma;\n"
        "beta = @{BETA};\n"
        "sigma = @{SIGMA};\n",
        encoding="utf-8",
    )

    main_file = tmp_path / "main.mod"
    main_file.write_text(
        '@#include "params.inc"\n'
        "var c, y;\n"
        "disc_factor = @{BETA};\n",
        encoding="utf-8",
    )

    out = preprocess_macro(main_file.read_text(encoding="utf-8"), base_dir=tmp_path).strip()
    assert "parameters beta, sigma;" in out
    assert "beta = 0.99;" in out
    assert "sigma = 2;" in out
    assert "disc_factor = 0.99;" in out


def test_include_with_expression(tmp_path: Path):
    """Verify @#include supports string expressions."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    (sub_dir / "shock_tfp.inc").write_text("varexo eps_tfp;\n", encoding="utf-8")

    src = """
@#define SHOCK_NAME = "tfp"
@#include "sub/shock_" + SHOCK_NAME + ".inc"
"""
    out = preprocess_macro(src, base_dir=tmp_path).strip()
    assert out == "varexo eps_tfp;"


def test_include_nested_chain(tmp_path: Path):
    """Verify linear inclusion chain A -> B -> C."""
    (tmp_path / "c.inc").write_text("level_c;\n", encoding="utf-8")
    (tmp_path / "b.inc").write_text('@#include "c.inc"\nlevel_b;\n', encoding="utf-8")
    (tmp_path / "a.mod").write_text('@#include "b.inc"\nlevel_a;\n', encoding="utf-8")

    out = preprocess_macro((tmp_path / "a.mod").read_text(encoding="utf-8"), base_dir=tmp_path).strip()
    assert "level_c;" in out
    assert "level_b;" in out
    assert "level_a;" in out


def test_includepath_directive(tmp_path: Path):
    """Verify @#includepath registers search directories for subsequent includes."""
    inc_dir = tmp_path / "custom_includes"
    inc_dir.mkdir()
    (inc_dir / "external.inc").write_text("var external_var;\n", encoding="utf-8")

    src = f"""
@#includepath "{inc_dir.as_posix()}"
@#include "external.inc"
"""
    out = preprocess_macro(src, base_dir=tmp_path).strip()
    assert "var external_var;" in out


def test_circular_include_direct_raises(tmp_path: Path):
    """Verify file directly including itself raises DynareMacroError with circular path."""
    main_file = tmp_path / "self_referential.mod"
    main_file.write_text('@#include "self_referential.mod"\n', encoding="utf-8")

    with pytest.raises(DynareMacroError) as exc_info:
        preprocess_macro(main_file.read_text(encoding="utf-8"), base_dir=tmp_path)
    assert "circular @#include detected" in str(exc_info.value)


def test_circular_include_indirect_raises(tmp_path: Path):
    """Verify circular inclusion loop A -> B -> C -> A raises DynareMacroError."""
    (tmp_path / "file_a.mod").write_text('@#include "file_b.mod"\n', encoding="utf-8")
    (tmp_path / "file_b.mod").write_text('@#include "file_c.mod"\n', encoding="utf-8")
    (tmp_path / "file_c.mod").write_text('@#include "file_a.mod"\n', encoding="utf-8")

    with pytest.raises(DynareMacroError) as exc_info:
        preprocess_macro((tmp_path / "file_a.mod").read_text(encoding="utf-8"), base_dir=tmp_path)
    assert "circular @#include detected" in str(exc_info.value)
    assert "file_a.mod" in str(exc_info.value)


def test_missing_include_raises(tmp_path: Path):
    """Verify including non-existent file raises DynareMacroError with searched paths."""
    src = '@#include "non_existent_file_xyz.inc"\n'
    with pytest.raises(DynareMacroError) as exc_info:
        preprocess_macro(src, base_dir=tmp_path)
    assert "not found" in str(exc_info.value).lower()


# ===========================================================================
# 7. Inline Interpolation @{EXPR}
# ===========================================================================

def test_interpolation_formatting():
    """Verify formatting rules: booleans -> 1/0, float exact integers -> int string, strings -> unquoted."""
    src = """
@{true}
@{false}
@{4.0}
@{3.14}
@{"US"}
"""
    out = preprocess_macro(src).strip().split()
    assert out == ["1", "0", "4", "3.14", "US"]


def test_interpolation_multiple_per_line():
    """Verify multiple interpolations on the same line are correctly expanded."""
    src = """
@#define C = "US"
@#define S = 3
@#define T = 1
var y_@{C}_s@{S}_t@{T};
"""
    out = preprocess_macro(src).strip()
    assert out == "var y_US_s3_t1;"


def test_interpolation_nested_delimiters():
    """Verify interpolation scanning handles nested braces, brackets, and quotes."""
    src = """
@#define ARR = [10, 20, 30]
val = @{ARR[2]};
str_val = @{"test{nested}braces"};
"""
    out = preprocess_macro(src).strip()
    assert "val = 20;" in out
    assert "str_val = test{nested}braces;" in out


def test_interpolation_unclosed_raises():
    """Verify unclosed @{ raises DynareMacroError."""
    with pytest.raises(DynareMacroError) as exc_info:
        preprocess_macro("var y_@{foo;")
    assert "unclosed macro interpolation" in str(exc_info.value).lower()


def test_interpolation_empty_raises():
    """Verify empty @{} raises DynareMacroError."""
    with pytest.raises(DynareMacroError) as exc_info:
        preprocess_macro("var y_@{ };")
    assert "empty macro interpolation" in str(exc_info.value).lower()


def test_interpolation_undefined_var_raises():
    """Verify referencing undefined variable in @{...} raises DynareMacroError."""
    with pytest.raises(DynareMacroError) as exc_info:
        preprocess_macro("var y_@{UNDEFINED_VAR_NAME};")
    assert "undefined macro variable" in str(exc_info.value).lower()


# ===========================================================================
# 8. End-to-End load_mod() Integration
# ===========================================================================

def test_load_mod_with_macro_directives():
    """Verify load_mod() transparently expands macro directives and solves LinearModel."""
    mod_text = """
@#define N_SECTORS = 2
@#for s in 1:N_SECTORS
var y_@{s}, a_@{s};
varexo eps_@{s};
@#endfor

parameters rho;
rho = 0.75;

model(linear);
@#for s in 1:N_SECTORS
y_@{s} = a_@{s};
a_@{s} = rho * a_@{s}(-1) + eps_@{s};
@#endfor
end;

initval;
@#for s in 1:N_SECTORS
y_@{s} = 0;
a_@{s} = 0;
@#endfor
end;
"""
    model = load_mod(mod_text)
    assert isinstance(model, LinearModel)
    assert set(model.variables) == {"y_1", "a_1", "y_2", "a_2"}
    assert set(model.shocks) == {"eps_1", "eps_2"}
    assert model.is_determinate is True

    dr = model.decision_rules()
    assert "ghx" in dr and "ghu" in dr
    assert dr["ghx"].shape == (4, 2)
    assert dr["ghu"].shape == (4, 2)


def test_load_mod_from_disk_with_include(tmp_path: Path):
    """Verify load_mod() on a file path on disk resolves @#include statements correctly."""
    (tmp_path / "model_vars.inc").write_text(
        "var c, k, a;\n"
        "varexo eps;\n",
        encoding="utf-8",
    )
    (tmp_path / "model_params.inc").write_text(
        "parameters alpha, beta, delta, rho;\n"
        "alpha = 0.33;\n"
        "beta = 0.99;\n"
        "delta = 0.025;\n"
        "rho = 0.95;\n",
        encoding="utf-8",
    )

    main_mod = tmp_path / "rbc_macro.mod"
    main_mod.write_text(
        '@#include "model_vars.inc"\n'
        '@#include "model_params.inc"\n'
        "model(linear);\n"
        "c = c(+1) - (1 - beta*(1-delta)) * a(+1) + alpha * (1 - beta*(1-delta)) * k;\n"
        "k = (1-delta)*k(-1) + delta*a - delta*c;\n"
        "a = rho*a(-1) + eps;\n"
        "end;\n"
        "initval;\n"
        "c = 0; k = 0; a = 0;\n"
        "end;\n",
        encoding="utf-8",
    )

    model = load_mod(main_mod)
    assert isinstance(model, LinearModel)
    assert set(model.variables) == {"c", "k", "a"}
    assert model.shocks == ("eps",)
    assert model.is_determinate is True


# ===========================================================================
# 9. Polymorphic build_dynare()
# ===========================================================================

def test_build_dynare_with_mod_string():
    """Verify build_dynare() called with a .mod string executes load_mod()."""
    mod_text = """
@#define N = 3
@#for i in 1:N
var x_@{i}, s_@{i};
varexo e_@{i};
@#endfor

parameters rho;
rho = 0.8;

model(linear);
@#for i in 1:N
x_@{i} = s_@{i};
s_@{i} = rho*s_@{i}(-1) + e_@{i};
@#endfor
end;

initval;
@#for i in 1:N
x_@{i} = 0;
s_@{i} = 0;
@#endfor
end;
"""
    model = build_dynare(mod_text)
    assert isinstance(model, LinearModel)
    assert len(model.variables) == 6
    assert len(model.shocks) == 3
    assert model.is_determinate is True


def test_build_dynare_with_path(tmp_path: Path):
    """Verify build_dynare() called with a Path object executes load_mod()."""
    mod_file = tmp_path / "simple.mod"
    mod_file.write_text(
        "var y, a;\n"
        "varexo eps;\n"
        "parameters rho;\n"
        "rho = 0.6;\n"
        "model(linear);\n"
        "y = a;\n"
        "a = rho*a(-1) + eps;\n"
        "end;\n"
        "initval;\n"
        "y = 0; a = 0;\n"
        "end;\n",
        encoding="utf-8",
    )

    model = build_dynare(mod_file)
    assert isinstance(model, LinearModel)
    assert model.variables == ("y", "a")
    assert model.is_determinate is True


def test_build_dynare_with_callable_preserves_compatibility():
    """Verify build_dynare() still works with standard equation callable."""
    def eqs(lead, curr, lag, shocks, params):
        return [
            curr.y - curr.a,
            curr.a - params.rho * lag.a - shocks.eps,
        ]

    model = build_dynare(
        eqs,
        variables=["y", "a"],
        shocks=["eps"],
        params={"rho": 0.85},
        steady_state={"y": 0.0, "a": 0.0},
    )
    assert isinstance(model, LinearModel)
    assert model.variables == ("y", "a")
    assert model.is_determinate is True


def test_build_dynare_callable_missing_variables_raises_typeerror():
    """Verify build_dynare() with callable but missing variables/shocks raises TypeError."""
    def dummy_eqs(lead, curr, lag, shocks, params):
        return []

    with pytest.raises(TypeError) as exc_info:
        build_dynare(dummy_eqs)
    assert "missing required keyword-only argument" in str(exc_info.value)


# ===========================================================================
# 10. Multi-Country and Multi-Sector DSGE Model Generation via @#for
# ===========================================================================

def test_multi_country_dsge_model_solve():
    """End-to-end multi-country DSGE model generated with @#for loops and solved via load_mod().

    Models a 3-region open economy (US, EA, JP) with New Keynesian IS curves,
    Phillips curves, Taylor rules, and cross-border trade spillovers.
    """
    mod_text = """
@#define COUNTRIES = ["US", "EA", "JP"]
@#for c in COUNTRIES
var y_@{c}, pi_@{c}, i_@{c}, a_@{c};
varexo eps_a_@{c};
@#endfor

parameters beta, sigma, kappa, phi_pi, rho_a, trade_share;
beta = 0.99;
sigma = 1.0;
kappa = 0.15;
phi_pi = 1.5;
rho_a = 0.85;
trade_share = 0.1;

model(linear);
@#for c in COUNTRIES
y_@{c} = y_@{c}(+1) - (1/sigma)*(i_@{c} - pi_@{c}(+1)) + a_@{c};
pi_@{c} = beta*pi_@{c}(+1) + kappa*y_@{c};
i_@{c} = phi_pi*pi_@{c};
a_@{c} = rho_a*a_@{c}(-1) + eps_a_@{c};
@#endfor
end;

initval;
@#for c in COUNTRIES
y_@{c} = 0;
pi_@{c} = 0;
i_@{c} = 0;
a_@{c} = 0;
@#endfor
end;
"""
    model = load_mod(mod_text)
    assert isinstance(model, LinearModel)
    assert len(model.variables) == 12
    assert len(model.shocks) == 3
    assert model.is_determinate is True

    # Check impulse responses to US productivity shock
    irfs = model.irf("eps_a_US", horizon=10)
    assert "y_US" in irfs.columns
    assert irfs["y_US"].iloc[0] > 0.0


def test_multi_sector_dsge_model_solve():
    """End-to-end multi-sector DSGE model with input-output linkages via nested @#for."""
    mod_text = """
@#define SECTORS = [1, 2]
@#for s in SECTORS
var y_@{s}, pi_@{s}, a_@{s};
varexo eps_@{s};
@#endfor

parameters beta, kappa, rho;
beta = 0.99;
kappa = 0.15;
rho = 0.8;

model(linear);
@#for s in SECTORS
y_@{s} = y_@{s}(+1) - 0.5*pi_@{s}(+1) + a_@{s};
pi_@{s} = beta*pi_@{s}(+1) + kappa*y_@{s};
a_@{s} = rho*a_@{s}(-1) + eps_@{s};
@#endfor
end;

initval;
@#for s in SECTORS
y_@{s} = 0;
pi_@{s} = 0;
a_@{s} = 0;
@#endfor
end;
"""
    model = build_dynare(mod_text)
    assert isinstance(model, LinearModel)
    assert len(model.variables) == 6
    assert model.is_determinate is True
    moments = model.theoretical_moments()
    assert moments.covariance.shape == (6, 6)


# ===========================================================================
# 11. Error Handling and Diagnostics
# ===========================================================================

def test_unclosed_for_raises():
    """Verify unclosed @#for block raises DynareMacroError."""
    src = """
@#for i in 1:3
var x_@{i};
"""
    with pytest.raises(DynareMacroError) as exc_info:
        preprocess_macro(src)
    assert "unclosed @#for" in str(exc_info.value).lower()


def test_unclosed_if_raises():
    """Verify unclosed @#if block raises DynareMacroError."""
    src = """
@#if true
var x;
"""
    with pytest.raises(DynareMacroError) as exc_info:
        preprocess_macro(src)
    assert "unclosed @#if" in str(exc_info.value).lower()


def test_orphaned_endfor_raises():
    """Verify orphaned @#endfor outside loop raises DynareMacroError."""
    src = """
var y;
@#endfor
"""
    with pytest.raises(DynareMacroError) as exc_info:
        preprocess_macro(src)
    assert "unexpected directive '@#endfor'" in str(exc_info.value).lower()


def test_orphaned_endif_raises():
    """Verify orphaned @#endif outside condition raises DynareMacroError."""
    src = """
var y;
@#endif
"""
    with pytest.raises(DynareMacroError) as exc_info:
        preprocess_macro(src)
    assert "unexpected directive '@#endif'" in str(exc_info.value).lower()


def test_duplicate_else_raises():
    """Verify multiple @#else in same @#if block raises DynareMacroError."""
    src = """
@#if true
var a;
@#else
var b;
@#else
var c;
@#endif
"""
    with pytest.raises(DynareMacroError) as exc_info:
        preprocess_macro(src)
    assert "unexpected directive '@#else'" in str(exc_info.value).lower()


def test_macro_error_directive():
    """Verify @#error directive halts preprocessing and reports user message."""
    src = """
@#define LEVEL = 99
@#if LEVEL > 50
@#error "Level exceeds allowed threshold"
@#endif
"""
    with pytest.raises(DynareMacroError) as exc_info:
        preprocess_macro(src)
    assert "Level exceeds allowed threshold" in str(exc_info.value)


def test_echo_and_echomacrovars_directives(capsys):
    """Verify @#echo and @#echomacrovars print diagnostic output to stdout."""
    src = """
@#define COUNTRY = "US"
@#define VAL = 42
@#echo "Configuring country: " + COUNTRY
@#echomacrovars
"""
    preprocess_macro(src)
    captured = capsys.readouterr()
    assert "Configuring country: US" in captured.out
    assert "'COUNTRY': 'US'" in captured.out
    assert "'VAL': 42" in captured.out


def test_unsupported_macro_directive_guard():
    """Verify parse_mod refuses unrecognized macro directives with DynareFeatureError or DynareMacroError."""
    mod_text = """
@#unknown_future_directive foo bar
var y;
varexo eps;
model;
y = eps;
end;
"""
    with pytest.raises((DynareFeatureError, DynareMacroError)) as exc_info:
        parse_mod(mod_text)
    assert "macro directive" in str(exc_info.value).lower()


# ===========================================================================
# 12. Strict Pyodide Compliance (Pure Python Stdlib)
# ===========================================================================

def test_pyodide_stdlib_compliance():
    """Verify macro modules import only allowed standard library and Pyodide core."""
    import puremacro.dsge.macro as m_pub
    import puremacro.dsge._macro as m_priv

    allowed_modules = {
        "itertools",
        "math",
        "os",
        "pathlib",
        "re",
        "sys",
        "typing",
        "dataclasses",
        "collections",
        "puremacro",
    }

    for mod in (m_pub, m_priv):
        for name, val in inspect.getmembers(mod):
            if inspect.ismodule(val):
                top_pkg = val.__name__.split(".")[0]
                assert top_pkg in allowed_modules or top_pkg in sys.stdlib_module_names, (
                    f"Forbidden module import in macro preprocessor: {val.__name__}"
                )
