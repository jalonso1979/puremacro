"""Dynare macro directives in puremacro 2.7.0.

In puremacro 2.7.0, valid macro directives (@#define, @#for, @#if, @#include, @{})
are legitimately processed and expanded by the macro pre-pass prior to parsing.

Unsupported or malformed macro directives must never be silently ignored; they must
fail loudly by raising DynareFeatureError or DynareMacroError.
"""
from __future__ import annotations

import pytest

from puremacro.dsge import DynareFeatureError, load_mod, parse_mod
from puremacro.dsge._macro import DynareMacroError


_RBC_BODY = """
  c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * exp(a(+1)) * k^(alpha - 1.0) + 1.0 - delta);
  k = exp(a) * k(-1)^alpha - c + (1.0 - delta) * k(-1);
  a = rho * a(-1) + eps;
"""

_BASE = """
var c k a;
varexo eps;
parameters alpha beta delta gamma rho;
alpha = 0.30; beta = 0.99; delta = 0.025; gamma = 1.0; rho = 0.80;
model;
""" + _RBC_BODY + """
end;
initval; k = 38.0; a = 0.0; c = 2.0; end;
shocks; var eps; stderr 0.01; end;
"""


@pytest.mark.parametrize("directive", [
    "@#unsupported_directive",
    "@#unknown_macro_tag",
    "@#custom_pragma_tag",
])
def test_unsupported_macro_directive_raises(directive):
    with pytest.raises((DynareFeatureError, DynareMacroError)):
        parse_mod(directive + "\n" + _BASE)


@pytest.mark.parametrize("directive", [
    "@#define",  # malformed
    "@#for",     # malformed
    "@#if",      # unclosed
])
def test_malformed_macro_directive_raises(directive):
    with pytest.raises((DynareFeatureError, DynareMacroError)):
        parse_mod(directive + "\n" + _BASE)


@pytest.mark.parametrize("directive", [
    "@#define N = 2",
    "@#if 1 == 1\n@#endif",
    "@#for i in 1:3\n@#endfor",
    "@#define RHO_VAL = 0.80",
])
def test_valid_macro_directives_expand_and_parse(directive):
    parsed = parse_mod(directive + "\n" + _BASE)
    assert parsed["variables"] == ["c", "k", "a"]
    assert parsed["shocks"] == ["eps"]


def test_macro_interpolation_succeeds():
    src = "@#define rho_val = 0.80\n" + _BASE.replace("rho = 0.80;", "rho = @{rho_val};")
    parsed = parse_mod(src)
    assert parsed["params"]["rho"] == pytest.approx(0.80)


def test_directive_inside_a_comment_is_still_ignored():
    """``% @#define N = 2`` is a comment, not a directive."""
    parse_mod("% @#define N = 2\n" + _BASE)
    parse_mod("// @#define N = 2\n" + _BASE)


def test_guard_fires_through_load_mod():
    with pytest.raises((DynareFeatureError, DynareMacroError)):
        load_mod("@#unsupported_directive\n" + _BASE)


def test_guard_is_not_a_valueerror():
    """A caller doing ``except ValueError`` around parse_mod must not swallow
    it: this failure is different in kind from a malformed declaration."""
    assert not issubclass(DynareFeatureError, ValueError)
    assert not issubclass(DynareMacroError, ValueError)
    with pytest.raises((DynareFeatureError, DynareMacroError)):
        try:
            load_mod("@#unsupported_directive\n" + _BASE)
        except ValueError:  # pragma: no cover
            pytest.fail("Macro error was caught as a ValueError")


def test_a_clean_file_still_parses():
    """Positive control: without any directives, nothing changed."""
    parsed = parse_mod(_BASE)
    assert parsed["variables"] == ["c", "k", "a"]
    assert parsed["shocks"] == ["eps"]
