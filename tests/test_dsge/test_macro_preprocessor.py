"""Comprehensive unit tests for puremacro's Dynare macro preprocessor."""
from __future__ import annotations

import math
from pathlib import Path
import pytest

from puremacro.dsge._macro import DynareMacroError, Scope, preprocess_macro


# ===========================================================================
# 1. @#define Directives (Variables, Constants, Functions)
# ===========================================================================

def test_define_simple_variable():
    src = """
@#define N = 4
var y_@{N};
"""
    out = preprocess_macro(src).strip()
    assert out == "var y_4;"


def test_define_without_value_defaults_to_one():
    src = """
@#define FLAG
@#if FLAG
var active;
@#else
var inactive;
@#endif
"""
    out = preprocess_macro(src).strip()
    assert out == "var active;"


def test_define_types():
    src = """
@#define INT_VAL = 42
@#define FLOAT_VAL = 3.1415
@#define STR_VAL = "US"
@#define BOOL_VAL = true
var y_@{STR_VAL} = @{INT_VAL} + @{FLOAT_VAL} + @{BOOL_VAL};
"""
    out = preprocess_macro(src).strip()
    assert out == 'var y_US = 42 + 3.1415 + 1;'


def test_define_macro_function():
    src = """
@#define ADD(a, b) = a + b
@#define MULT(x, y) = x * y
val = @{ADD(10, 20)};
prod = @{MULT(3, 4)};
"""
    out = preprocess_macro(src).strip()
    assert "val = 30;" in out
    assert "prod = 12;" in out


def test_define_macro_function_zero_args():
    src = """
@#define GET_PI() = 3.14
pi = @{GET_PI()};
"""
    out = preprocess_macro(src).strip()
    assert "pi = 3.14;" in out


def test_define_macro_function_arity_mismatch():
    src = """
@#define F(x, y) = x + y
val = @{F(1)};
"""
    with pytest.raises(DynareMacroError) as exc:
        preprocess_macro(src)
    assert "expects 2 arguments" in str(exc.value)


def test_define_undef():
    src = """
@#define X = 10
@#undef X
@#if defined(X)
var x_defined;
@#else
var x_undefined;
@#endif
"""
    out = preprocess_macro(src).strip()
    assert out == "var x_undefined;"


def test_malformed_define_raises():
    with pytest.raises(DynareMacroError):
        preprocess_macro("@#define 123_invalid = 1")


# ===========================================================================
# 2. @#for Loops (Ranges, Arrays, Tuples, Destructuring, Scoping)
# ===========================================================================

def test_for_over_range():
    src = """
@#for i in 1:3
var y_@{i};
@#endfor
"""
    out = preprocess_macro(src).strip()
    expected = "var y_1;\nvar y_2;\nvar y_3;"
    assert out == expected


def test_for_over_step_range():
    src = """
@#for i in 1:2:5
var y_@{i};
@#endfor
"""
    out = preprocess_macro(src).strip()
    expected = "var y_1;\nvar y_3;\nvar y_5;"
    assert out == expected


def test_for_over_descending_range():
    src = """
@#for i in 3:-1:1
var y_@{i};
@#endfor
"""
    out = preprocess_macro(src).strip()
    expected = "var y_3;\nvar y_2;\nvar y_1;"
    assert out == expected


def test_for_over_array():
    src = """
@#for c in ["US", "EA", "UK"]
var y_@{c};
@#endfor
"""
    out = preprocess_macro(src).strip()
    expected = "var y_US;\nvar y_EA;\nvar y_UK;"
    assert out == expected


def test_for_destructuring():
    src = """
@#define countries = [(1, "US"), (2, "EA")]
@#for (id, name) in countries
var y_@{name}_@{id};
@#endfor
"""
    out = preprocess_macro(src).strip()
    expected = "var y_US_1;\nvar y_EA_2;"
    assert out == expected


def test_for_destructuring_without_parens():
    src = """
@#for id, name in [(1, "US"), (2, "EA")]
var y_@{name}_@{id};
@#endfor
"""
    out = preprocess_macro(src).strip()
    expected = "var y_US_1;\nvar y_EA_2;"
    assert out == expected


def test_for_when_condition():
    src = """
@#for i in 1:6 when i % 2 == 0
var even_@{i};
@#endfor
"""
    out = preprocess_macro(src).strip()
    expected = "var even_2;\nvar even_4;\nvar even_6;"
    assert out == expected


def test_for_destructuring_with_when():
    src = """
@#for (id, name) in [(1, "US"), (2, "EA"), (3, "SKIP")] when name != "SKIP"
var y_@{name};
@#endfor
"""
    out = preprocess_macro(src).strip()
    expected = "var y_US;\nvar y_EA;"
    assert out == expected


def test_for_nested_loops():
    src = """
@#for i in 1:2
@#for j in 1:2
x_@{i}_@{j} = y;
@#endfor
@#endfor
"""
    out = preprocess_macro(src).strip()
    expected = "x_1_1 = y;\nx_1_2 = y;\nx_2_1 = y;\nx_2_2 = y;"
    assert out == expected


def test_for_lexical_scope_restoration():
    src = """
@#define i = 999
@#for i in 1:2
loop_@{i} = 1;
@#endfor
outer = @{i};
"""
    out = preprocess_macro(src).strip()
    assert "loop_1 = 1;" in out
    assert "loop_2 = 1;" in out
    assert "outer = 999;" in out


def test_for_undefined_restoration():
    src = """
@#for temp_var in 1:2
x = @{temp_var};
@#endfor
@#if defined(temp_var)
var leak;
@#else
var clean;
@#endif
"""
    out = preprocess_macro(src).strip()
    assert "var clean;" in out


def test_for_mutates_outer_variable():
    src = """
@#define total = 0
@#for i in 1:3
@#define total = total + i
@#endfor
final_total = @{total};
"""
    out = preprocess_macro(src).strip()
    assert "final_total = 6;" in out


def test_for_over_non_sequence_raises():
    with pytest.raises(DynareMacroError):
        preprocess_macro("@#for i in 42\nx;\n@#endfor")


def test_for_destructuring_mismatch_raises():
    with pytest.raises(DynareMacroError):
        preprocess_macro("@#for (a, b) in [(1, 2, 3)]\nx;\n@#endfor")


def test_unclosed_for_raises():
    with pytest.raises(DynareMacroError) as exc:
        preprocess_macro("line 1\n@#for i in 1:2\nline 3")
    assert "Unclosed @#for" in str(exc.value)
    assert "line 2" in str(exc.value)


# ===========================================================================
# 3. @#if, @#elseif, @#else, @#endif (Conditionals & Existence Guards)
# ===========================================================================

def test_if_true_false():
    src = """
@#if true
var included;
@#endif
@#if false
var excluded;
@#endif
"""
    out = preprocess_macro(src).strip()
    assert "var included;" in out
    assert "var excluded;" not in out


def test_if_else():
    src = """
@#define MODE = "dev"
@#if MODE == "prod"
var prod_var;
@#else
var dev_var;
@#endif
"""
    out = preprocess_macro(src).strip()
    assert "var dev_var;" in out
    assert "var prod_var;" not in out


def test_if_elseif_chain():
    src = """
@#define CO = "EA"
@#if CO == "US"
var y_US;
@#elseif CO == "EA"
var y_EA;
@#elseif CO == "UK"
var y_UK;
@#else
var y_ROW;
@#endif
"""
    out = preprocess_macro(src).strip()
    assert out == "var y_EA;"


def test_if_elif_synonym():
    src = """
@#define N = 2
@#if N == 1
var one;
@#elif N == 2
var two;
@#else
var other;
@#endif
"""
    out = preprocess_macro(src).strip()
    assert out == "var two;"


def test_ifdef_and_ifndef():
    src = """
@#define MY_VAR = 10
@#ifdef MY_VAR
var found;
@#else
var not_found;
@#endif
@#ifndef OTHER_VAR
var absent;
@#endif
"""
    out = preprocess_macro(src).strip()
    assert "var found;" in out
    assert "var absent;" in out


def test_inactive_branches_not_evaluated():
    src = """
@#if false
@#error "Should not be raised!"
@#define BAD = 1 / 0
var @{undefined_variable};
@#endif
var safe;
"""
    out = preprocess_macro(src).strip()
    assert out == "var safe;"


def test_short_circuiting_and_or():
    src = """
@#if defined(FOO) && FOO > 10
var branch_1;
@#else
var branch_safe;
@#endif
"""
    out = preprocess_macro(src).strip()
    assert out == "var branch_safe;"


def test_unclosed_if_raises():
    with pytest.raises(DynareMacroError) as exc:
        preprocess_macro("line 1\n@#if true\nline 3")
    assert "Unclosed @#if" in str(exc.value)
    assert "line 2" in str(exc.value)


def test_unexpected_endif_raises():
    with pytest.raises(DynareMacroError):
        preprocess_macro("@#endif")


def test_unexpected_else_raises():
    with pytest.raises(DynareMacroError):
        preprocess_macro("@#else")


# ===========================================================================
# 4. @#include, @#includepath, and Circular Include Detection
# ===========================================================================

def test_include_simple(tmp_path: Path):
    inc_file = tmp_path / "included.mod"
    inc_file.write_text("var included_y;\n@#define INC_VAR = 123\n", encoding="utf-8")

    main_src = """
var main_y;
@#include "included.mod"
var from_inc_@{INC_VAR};
"""
    out = preprocess_macro(main_src, base_dir=tmp_path).strip()
    assert "var main_y;" in out
    assert "var included_y;" in out
    assert "var from_inc_123;" in out


def test_include_with_expression(tmp_path: Path):
    inc_file = tmp_path / "model_us.mod"
    inc_file.write_text("var us_gdp;\n", encoding="utf-8")

    main_src = """
@#define CO = "us"
@#include "model_" + CO + ".mod"
"""
    out = preprocess_macro(main_src, base_dir=tmp_path).strip()
    assert "var us_gdp;" in out


def test_circular_include_raises(tmp_path: Path):
    file_a = tmp_path / "a.mod"
    file_b = tmp_path / "b.mod"

    file_a.write_text('@#include "b.mod"\n', encoding="utf-8")
    file_b.write_text('@#include "a.mod"\n', encoding="utf-8")

    with pytest.raises(DynareMacroError) as exc:
        preprocess_macro('@#include "a.mod"\n', base_dir=tmp_path)
    assert "circular @#include detected" in str(exc.value)


def test_missing_include_raises(tmp_path: Path):
    with pytest.raises(DynareMacroError) as exc:
        preprocess_macro('@#include "nonexistent.mod"', base_dir=tmp_path)
    assert "not found" in str(exc.value)


def test_includepath_directive(tmp_path: Path):
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    inc_file = sub_dir / "deep.mod"
    inc_file.write_text("var deep_y;\n", encoding="utf-8")

    main_src = f"""
@#includepath "{sub_dir}"
@#include "deep.mod"
"""
    out = preprocess_macro(main_src, base_dir=tmp_path).strip()
    assert "var deep_y;" in out


# ===========================================================================
# 5. Inline Interpolation @{...}
# ===========================================================================

def test_interpolation_multiple_per_line():
    src = """
@#define A = 1
@#define B = 2
var x_@{A}_@{B} = @{A + B};
"""
    out = preprocess_macro(src).strip()
    assert out == "var x_1_2 = 3;"


def test_interpolation_nested_brackets_and_strings():
    src = """
@#define arr = [10, 20, 30]
val = @{ arr[2] };
str_val = @{ "hello } world" };
"""
    out = preprocess_macro(src).strip()
    assert "val = 20;" in out
    assert 'str_val = hello } world;' in out


def test_unclosed_interpolation_raises():
    with pytest.raises(DynareMacroError) as exc:
        preprocess_macro("var x_@{abc;\n")
    assert "Unclosed macro interpolation" in str(exc.value)


def test_empty_interpolation_raises():
    with pytest.raises(DynareMacroError) as exc:
        preprocess_macro("var x_@{ };\n")
    assert "Empty macro interpolation" in str(exc.value)


# ===========================================================================
# 6. Array Comprehensions
# ===========================================================================

def test_array_comprehension_basic():
    src = """
@#define squares = [x * x for x in 1:4]
@#for s in squares
val_@{s} = 1;
@#endfor
"""
    out = preprocess_macro(src).strip()
    expected = "val_1 = 1;\nval_4 = 1;\nval_9 = 1;\nval_16 = 1;"
    assert out == expected


def test_array_comprehension_with_when():
    src = """
@#define evens = [x for x in 1:6 when x % 2 == 0]
@#for e in evens
even_@{e} = 1;
@#endfor
"""
    out = preprocess_macro(src).strip()
    expected = "even_2 = 1;\neven_4 = 1;\neven_6 = 1;"
    assert out == expected


def test_array_comprehension_destructuring():
    src = """
@#define items = [(1, "A"), (2, "B"), (3, "C")]
@#define names = [name for (id, name) in items when id > 1]
@#for n in names
name_@{n} = 1;
@#endfor
"""
    out = preprocess_macro(src).strip()
    expected = "name_B = 1;\nname_C = 1;"
    assert out == expected


# ===========================================================================
# 7. Built-in Functions & Complex Operators
# ===========================================================================

def test_math_builtins():
    src = """
@#define EXP_V = exp(0)
@#define LOG_V = log(exp(1))
@#define SQRT_V = sqrt(16)
@#define CBRT_V = cbrt(27)
@#define ABS_V = abs(-5)
@#define SIGN_V = sign(-42)
@#define ROUND_V = round(3.7)
@#define FLOOR_V = floor(3.7)
@#define CEIL_V = ceil(3.2)
@#define MAX_V = max(1, 10, 5)
@#define MIN_V = min([2, -1, 8])
res = @{EXP_V}, @{LOG_V}, @{SQRT_V}, @{CBRT_V}, @{ABS_V}, @{SIGN_V}, @{ROUND_V}, @{FLOOR_V}, @{CEIL_V}, @{MAX_V}, @{MIN_V};
"""
    out = preprocess_macro(src).strip()
    assert "res = 1, 1, 4, 3, 5, -1, 4, 3, 4, 10, -1;" in out


def test_statistical_builtins():
    src = """
@#define NCDF = normcdf(0.0)
@#define ERF_0 = erf(0.0)
ncdf = @{NCDF};
erf0 = @{ERF_0};
"""
    out = preprocess_macro(src).strip()
    assert "ncdf = 0.5;" in out
    assert "erf0 = 0;" in out


def test_collection_builtins():
    src = """
@#define arr = [1, 2, 3, 4]
@#define empty_arr = []
len = @{length(arr)};
is_empty = @{empty(empty_arr)};
is_not_empty = @{empty(arr)};
total = @{sum(arr)};
"""
    out = preprocess_macro(src).strip()
    assert "len = 4;" in out
    assert "is_empty = 1;" in out
    assert "is_not_empty = 0;" in out
    assert "total = 10;" in out


def test_type_check_builtins():
    src = """
@#define b = true
@#define r = 3.14
@#define s = "text"
@#define t = (1, 2)
@#define a = [1, 2]
res = @{isboolean(b)}, @{isreal(r)}, @{isreal(b)}, @{isstring(s)}, @{istuple(t)}, @{isarray(a)};
"""
    out = preprocess_macro(src).strip()
    # Notice isreal(b) must be 0 (bool is not real)
    assert "res = 1, 1, 0, 1, 1, 1;" in out


def test_cartesian_product_and_power():
    src = """
@#define prod = [1, 2] * ["a", "b"]
@#for (num, char) in prod
pair_@{num}_@{char} = 1;
@#endfor
"""
    out = preprocess_macro(src).strip()
    expected = "pair_1_a = 1;\npair_1_b = 1;\npair_2_a = 1;\npair_2_b = 1;"
    assert out == expected


def test_cartesian_power():
    src = """
@#define grid = [0, 1] ^ 2
@#for (x, y) in grid
pt_@{x}_@{y} = 1;
@#endfor
"""
    out = preprocess_macro(src).strip()
    expected = "pt_0_0 = 1;\npt_0_1 = 1;\npt_1_0 = 1;\npt_1_1 = 1;"
    assert out == expected


def test_set_operations_on_arrays():
    src = """
@#define A = [1, 2, 3]
@#define B = [2, 3, 4]
@#define DIFF = A - B
@#define INTER = A & B
@#define UNION = A | B
diff_len = @{length(DIFF)};
diff_elem = @{DIFF[1]};
inter_len = @{length(INTER)};
union_len = @{length(UNION)};
"""
    out = preprocess_macro(src).strip()
    assert "diff_len = 1;" in out
    assert "diff_elem = 1;" in out
    assert "inter_len = 2;" in out
    assert "union_len = 4;" in out


def test_type_casting():
    src = """
@#define B = (bool) 1
@#define R = (real) "3.14"
@#define I = (integer) "42"
@#define S = (string) 100
val = @{B}, @{R}, @{I}, @{S};
"""
    out = preprocess_macro(src).strip()
    assert "val = 1, 3.14, 42, 100;" in out


# ===========================================================================
# 8. Diagnostics, Line Continuations & Comments
# ===========================================================================

def test_line_continuation():
    src = """
@#define arr = \\
    [1, 2, \\
     3]
@#for i in arr
val_@{i};
@#endfor
"""
    out = preprocess_macro(src).strip()
    expected = "val_1;\nval_2;\nval_3;"
    assert out == expected


def test_trailing_comment_on_directive():
    src = """
@#define N = 2 // number of countries
@#define S = "val // not comment" % trailing comment
val = @{N}, @{S};
"""
    out = preprocess_macro(src).strip()
    assert 'val = 2, val // not comment;' in out


def test_commented_directive_is_ignored():
    src = """
// @#define N = 2
% @#define N = 3
/*
@#define N = 4
*/
@#if defined(N)
var defined;
@#else
var not_defined;
@#endif
"""
    out = preprocess_macro(src)
    assert "var not_defined;" in out
    assert "// @#define N = 2" in out
    assert "% @#define N = 3" in out


def test_error_directive_raises():
    src = """
@#define CO = "FR"
@#if CO != "US" && CO != "EA"
@#error "Unsupported country: " + CO
@#endif
"""
    with pytest.raises(DynareMacroError) as exc:
        preprocess_macro(src)
    assert "Unsupported country: FR" in str(exc.value)


def test_echo_and_echomacrovars(capsys):
    src = """
@#define N = 5
@#echo "Testing echo with N = " + (string) N
@#echomacrovars
var y;
"""
    out = preprocess_macro(src).strip()
    assert out == "var y;"
    captured = capsys.readouterr()
    assert "Testing echo with N = 5" in captured.out
    assert "N" in captured.out


# ===========================================================================
# 9. Realistic DSGE Multi-Country Macro Benchmark
# ===========================================================================

def test_realistic_multi_country_dsge():
    src = """
// Macro-expanded multi-country DSGE model
@#define countries = ["US", "EA"]
@#define N = length(countries)

var
@#for c in countries
  y_@{c} c_@{c} i_@{c}
@#endfor
;

varexo
@#for c in countries
  eps_@{c}
@#endfor
;

parameters
@#for c in countries
  alpha_@{c} beta_@{c}
@#endfor
;

@#for c in countries
alpha_@{c} = 0.33;
beta_@{c} = 0.99;
@#endfor

model;
@#for c in countries
  c_@{c}^(-1.0) = beta_@{c} * c_@{c}(+1)^(-1.0) * (alpha_@{c} * y_@{c}(+1) + 1.0);
  y_@{c} = c_@{c} + i_@{c} + eps_@{c};
@#endfor
end;
"""
    out = preprocess_macro(src).strip()
    assert "y_US c_US i_US" in out
    assert "y_EA c_EA i_EA" in out
    assert "eps_US" in out
    assert "eps_EA" in out
    assert "alpha_US = 0.33;" in out
    assert "alpha_EA = 0.33;" in out
    assert "c_US^(-1.0) = beta_US * c_US(+1)^(-1.0) * (alpha_US * y_US(+1) + 1.0);" in out
    assert "c_EA^(-1.0) = beta_EA * c_EA(+1)^(-1.0) * (alpha_EA * y_EA(+1) + 1.0);" in out


# ===========================================================================
# 10. Robustness & Edge Cases
# ===========================================================================

def test_built_in_mod_function():
    src = """
@#for i in 1:6 when mod(i, 2) == 0
var mod_even_@{i};
@#endfor
"""
    out = preprocess_macro(src).strip()
    assert out == "var mod_even_2;\nvar mod_even_4;\nvar mod_even_6;"


def test_division_by_zero_raises():
    src = "@#define X = 10 / 0"
    with pytest.raises(DynareMacroError) as exc:
        preprocess_macro(src)
    assert "Division by zero" in str(exc.value)


def test_modulo_by_zero_raises():
    src = "@#define X = 10 % 0"
    with pytest.raises(DynareMacroError) as exc:
        preprocess_macro(src)
    assert "Modulo by zero" in str(exc.value)


def test_index_out_of_bounds_raises():
    src = """
@#define arr = [1, 2]
val = @{arr[3]};
"""
    with pytest.raises(DynareMacroError) as exc:
        preprocess_macro(src)
    assert "out of bounds" in str(exc.value)


def test_index_zero_out_of_bounds_raises():
    src = """
@#define arr = [1, 2]
val = @{arr[0]};
"""
    with pytest.raises(DynareMacroError) as exc:
        preprocess_macro(src)
    assert "out of bounds" in str(exc.value)


def test_index_non_sequence_raises():
    src = """
@#define num = 42
val = @{num[1]};
"""
    with pytest.raises(DynareMacroError) as exc:
        preprocess_macro(src)
    assert "Cannot index" in str(exc.value)


def test_call_non_callable_raises():
    src = """
@#define num = 42
val = @{num()};
"""
    with pytest.raises(DynareMacroError) as exc:
        preprocess_macro(src)
    assert "not callable" in str(exc.value)


def test_range_step_zero_raises():
    src = "@#for i in 1:0:5\nx;\n@#endfor"
    with pytest.raises(DynareMacroError) as exc:
        preprocess_macro(src)
    assert "Range step cannot be 0" in str(exc.value)


def test_range_indexing_and_slicing():
    src = """
@#define arr = ["a", "b", "c", "d"]
slice = @{arr[1:3]};
multi = @{arr[[1, 4]]};
str_slice = @{"hello"[1:3]};
"""
    out = preprocess_macro(src).strip()
    assert 'slice = ["a", "b", "c"];' in out
    assert 'multi = ["a", "d"];' in out
    assert 'str_slice = hel;' in out


def test_linear_chain_includes(tmp_path: Path):
    file_c = tmp_path / "c.mod"
    file_c.write_text("@#define C_VAL = 300\nvar c_var;\n", encoding="utf-8")

    file_b = tmp_path / "b.mod"
    file_b.write_text('@#include "c.mod"\n@#define B_VAL = 200\nvar b_var;\n', encoding="utf-8")

    file_a = tmp_path / "a.mod"
    file_a.write_text('@#include "b.mod"\nvar a_var_@{B_VAL}_@{C_VAL};\n', encoding="utf-8")

    out = preprocess_macro('@#include "a.mod"\n', base_dir=tmp_path).strip()
    assert "var c_var;" in out
    assert "var b_var;" in out
    assert "var a_var_200_300;" in out


def test_three_file_circular_include(tmp_path: Path):
    file_a = tmp_path / "a.mod"
    file_b = tmp_path / "b.mod"
    file_c = tmp_path / "c.mod"

    file_a.write_text('@#include "b.mod"\n', encoding="utf-8")
    file_b.write_text('@#include "c.mod"\n', encoding="utf-8")
    file_c.write_text('@#include "a.mod"\n', encoding="utf-8")

    with pytest.raises(DynareMacroError) as exc:
        preprocess_macro('@#include "a.mod"\n', base_dir=tmp_path)
    assert "circular @#include detected" in str(exc.value)


def test_truthiness_conditions():
    src = """
@#if 0
branch_zero;
@#endif
@#if 42
branch_nonzero;
@#endif
@#if ""
branch_empty_str;
@#endif
@#if "abc"
branch_full_str;
@#endif
@#if []
branch_empty_arr;
@#endif
@#if [1]
branch_full_arr;
@#endif
"""
    out = preprocess_macro(src)
    assert "branch_zero;" not in out
    assert "branch_nonzero;" in out
    assert "branch_empty_str;" not in out
    assert "branch_full_str;" in out
    assert "branch_empty_arr;" not in out
    assert "branch_full_arr;" in out


def test_dynare_macro_error_attributes():
    with pytest.raises(DynareMacroError) as exc:
        preprocess_macro("line 1\nline 2\n@#if true\nline 4")
    err = exc.value
    assert err.line == 3
    assert err.column is not None
    assert err.col == err.column
    assert "line 3" in str(err)


def test_pyodide_compliance():
    """Verify that _macro.py uses only standard library modules."""
    import ast
    macro_path = Path(__file__).resolve().parent.parent.parent / "puremacro" / "dsge" / "_macro.py"
    assert macro_path.is_file()
    tree = ast.parse(macro_path.read_text(encoding="utf-8"))

    stdlib_modules = {
        "__future__", "itertools", "math", "pathlib", "re", "typing", "sys", "os", "collections"
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_root = alias.name.split(".")[0]
                assert mod_root in stdlib_modules, f"Unauthorized import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod_root = node.module.split(".")[0]
                assert mod_root in stdlib_modules, f"Unauthorized from-import: {node.module}"

