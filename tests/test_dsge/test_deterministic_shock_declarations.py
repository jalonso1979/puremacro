"""Dated shock declarations preserve cumulative and overwrite semantics."""
import numpy as np
import pytest

from puremacro.dsge import parse_mod, read_mod


BASE = "var x; varexo e; model; x=.5*x(-1)+e; end;"


def test_deterministic_model_can_omit_varexo():
    parsed = parse_mod("var x; model; x=.5*x(-1); end;")
    assert parsed["shocks"] == []
    assert parsed["_compiled"].n_e == 0


def test_dated_shocks_accumulate_and_only_replace_the_redeclared_date():
    parsed = parse_mod(BASE + """
        shocks; var e; periods 1,3; values 1,3; end;
        shocks; var e; periods 2,3; values 2,30; end;
    """)
    assert parsed["anticipated_shocks"]["e"] == {"periods": [1, 2, 3], "values": [1, 2, 30]}


def test_shocks_overwrite_clears_prior_dates_and_covariance():
    parsed = parse_mod(BASE + """
        shocks; var e; stderr 2; var e; periods 1; values 4; end;
        shocks(overwrite); var e; periods 2; values 3; end;
    """)
    assert parsed["anticipated_shocks"]["e"] == {"periods": [2], "values": [3]}
    assert parsed["shock_cov"][0, 0] == 0


def test_dated_values_resolve_preceding_calibration_in_structured_parser():
    parsed = parse_mod(BASE + """
        parameters impulse scale; scale=2; impulse=scale/4;
        shocks; var e; periods 1,2; values impulse,2*impulse; end;
    """)
    assert "_dag" in parsed
    assert parsed["anticipated_shocks"]["e"]["values"] == [0.5, 1.0]


@pytest.mark.parametrize("values", ["unresolved", "exp(1000)"])
def test_invalid_dated_value_expressions_do_not_enter_legacy_fallback(values):
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="Shock values"):
        parse_mod(BASE + f"shocks; var e; periods 1; values {values}; end;")


@pytest.mark.parametrize("periods,values", [("0", "1"), ("1,2", "1,2,3"), ("1.5", "1"), ("1:2.5", "1"), ("1:1.5:3", "1")])
def test_invalid_shock_dates_or_value_counts_are_rejected(periods, values):
    with pytest.raises(ValueError, match="Shock periods"):
        parse_mod(BASE + f"shocks; var e; periods {periods}; values {values}; end;")


@pytest.mark.parametrize("values,expected", [
    ("0.02 -0.01 -0.01", [0.02, -0.01, -0.01]),
    ("0.02, -0.01, -0.01", [0.02, -0.01, -0.01]),
    ("-0.01 -0.02 +0.03", [-0.01, -0.02, 0.03]),
    ("1e-2 -2E-2 +3e+0", [0.01, -0.02, 3.0]),
    (".02 -.01 +.03", [0.02, -0.01, 0.03]),
    ("0.02 /* next date */ -0.01\n -0.01", [0.02, -0.01, -0.01]),
    ("(0.03-0.01) (-0.01) (-0.02+0.01)", [0.02, -0.01, -0.01]),
    ("(0.03-0.01), -0.01, (-0.02+0.01)", [0.02, -0.01, -0.01]),
    ("[0.02 -0.01 -0.01]", [0.02, -0.01, -0.01]),
])
def test_dated_values_keep_signed_entries_separate(values, expected):
    # Dynare's grammar permits optional commas, with expressions in parentheses:
    # https://www.dynare.org/manual/the-model-file.html#block-shocks
    parsed = parse_mod(BASE + f"shocks; var e; periods 1:3; values {values}; end;")
    assert "_dag" in parsed  # no silent rescue by the legacy regex parser
    assert parsed["anticipated_shocks"]["e"]["periods"] == [1, 2, 3]
    np.testing.assert_allclose(parsed["anticipated_shocks"]["e"]["values"], expected,
                               rtol=0, atol=1e-15)


@pytest.mark.parametrize("values,expected", [
    ("(impulse-scale/4) (-impulse) (2*(impulse+scale/4))", [0.25, -0.5, 1.5]),
    ("impulse -impulse 2*impulse", [0.5, -0.5, 1.0]),
    ("impulse,2*impulse,-impulse/2", [0.5, 1.0, -0.25]),
    ("-impulse^2 +impulse^2 (impulse-impulse)", [-0.25, 0.25, 0.0]),
])
def test_signed_values_preserve_calibration_and_expression_precedence(values, expected):
    source = BASE + "parameters impulse scale; scale=1; impulse=scale/2;"
    parsed = parse_mod(source + f"shocks; var e; periods 1:3; values {values}; end;")
    assert "_dag" in parsed
    np.testing.assert_allclose(parsed["anticipated_shocks"]["e"]["values"], expected,
                               rtol=0, atol=1e-15)


@pytest.mark.parametrize("values", ["-.01", "(-.02/2)", "[-.01]"])
def test_single_negative_value_broadcasts_over_period_range(values):
    parsed = parse_mod(BASE + f"shocks; var e; periods 2:4; values {values}; end;")
    assert parsed["anticipated_shocks"]["e"] == {
        "periods": [2, 3, 4], "values": [-0.01, -0.01, -0.01],
    }


@pytest.mark.pyodide_smoke
def test_signed_shock_list_reaches_source_ordered_simulation():
    result = read_mod(BASE + """
        initval; x=0; e=0; end;
        shocks; var e; periods 1:3; values 0.02 -0.01 -0.01; end;
        simul(periods=4);
    """).execute_mod_commands()
    # Independently apply x_t = .5*x_{t-1} + e_t, starting at zero.
    np.testing.assert_allclose(result.perfect_foresight.path.x,
                               [0.02, 0.0, -0.01, -0.005], rtol=0, atol=1e-10)


@pytest.mark.parametrize("loader", [parse_mod, read_mod])
@pytest.mark.parametrize("values", [
    "(0.02, -0.01, -0.01)",  # not a parenthesized expression
    "[0.02 -0.01 -0.01",     # missing closing square bracket
    "0.02 -0.01 -0.01)",     # unmatched closing parenthesis
    "0.02,, -0.01, -0.01",   # empty entry
])
def test_malformed_shock_values_fail_with_location_without_fallback(loader, values):
    source = BASE + f"\nshocks; var e; periods 1:3; values {values}; end;"
    with pytest.raises(ValueError, match="Invalid shock values") as caught:
        loader(source)
    # parse_mod must not use its legacy parser; read_mod's inspection must retain
    # the same meaningful message and source location rather than losing shocks.
    assert "Line 2, Col" in str(caught.value)
