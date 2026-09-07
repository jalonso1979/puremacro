"""Dynare macro directives are not implemented — they must not be ignored.

Every other unsupported .mod construct fails loudly: a model-local ``#``
variable over an endogenous variable, ``STEADY_STATE()``, ``normcdf`` all raise
a ``NameError`` when the compiled equations run. The macro processor did not:
``@#define N = 2`` was skipped as junk, the file loaded clean, and a *different
model from the one the file describes* was solved with no complaint.

These tests pin the guard that closes that hole.
"""
from __future__ import annotations

import pytest

from puremacro.dsge import DynareFeatureError, load_mod, parse_mod


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
    "@#define N = 2",
    '@#include "common.mod"',
    "@#if N == 2\n@#endif",
    "@#for i in 1:3\n@#endfor",
    "@#echomacrovars",
])
def test_macro_directive_raises(directive):
    with pytest.raises(DynareFeatureError) as exc:
        parse_mod(directive + "\n" + _BASE)
    msg = str(exc.value)
    assert "macro" in msg.lower()
    assert "2.7.0" in msg


def test_macro_interpolation_raises():
    src = _BASE.replace("rho = 0.80;", "rho = @{rho_val};")
    with pytest.raises(DynareFeatureError):
        parse_mod(src)


def test_directive_inside_a_comment_is_still_ignored():
    """``% @#define N = 2`` is a comment, not a directive. The guard runs after
    _remove_comments precisely so this stays legal."""
    parse_mod("% @#define N = 2\n" + _BASE)
    parse_mod("// @#define N = 2\n" + _BASE)


def test_guard_fires_through_load_mod():
    with pytest.raises(DynareFeatureError):
        load_mod("@#define N = 2\n" + _BASE)


def test_guard_is_not_a_valueerror():
    """A caller doing ``except ValueError`` around parse_mod must not swallow
    it: this failure is different in kind from a malformed declaration."""
    assert not issubclass(DynareFeatureError, ValueError)
    with pytest.raises(DynareFeatureError):
        try:
            load_mod("@#define N = 2\n" + _BASE)
        except ValueError:  # pragma: no cover - the point of the test
            pytest.fail("DynareFeatureError was caught as a ValueError")


def test_a_clean_file_still_parses():
    """Positive control: without the guard firing, nothing changed."""
    parsed = parse_mod(_BASE)
    assert parsed["variables"] == ["c", "k", "a"]
    assert parsed["shocks"] == ["eps"]
