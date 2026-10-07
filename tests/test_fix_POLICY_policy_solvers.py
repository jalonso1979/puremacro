"""Regression tests for the optimal-policy solver defects of the 2026-09-30 review.

1. ``stabilization_bias`` must be NaN, with a warning, whenever the comparison
   cannot be computed (the commitment solve raises, or a loss is NaN); it must
   never come back as a silent 0.0.
2. The ``timeless`` flag of ``lq_commitment`` and ``ramsey_model`` never changed
   the solution. It is deprecated: passing it warns, and the docstrings no
   longer equate lambda_{-1} = 0 (Ramsey from the steady state) with the
   timeless perspective.
3. The low-beta reversal (discretion beats the timeless rule under the
   unconditional loss) is Sauer (2010, IJCB 6(2), Prop. 2), not Jensen and
   McCallum (2002), whose result is that the timeless rule is not the best rule
   of its own form.
4. ``RamseyResult.focs`` are readable equations (Dynare syntax that parses back
   to the same expressions), and ``foc_nodes`` match the augmented system the
   solver builds from the linear model matrices.
5. ``osr`` locates the Clarida-Gali-Gertler simple-rule allocation to better
   than 1e-6 at the random calibrations where SciPy's default Nelder-Mead
   tolerances (xatol = fatol = 1e-4) stopped at errors up to 2.7e-5.

The CGG (1999) closed forms used as oracles: under discretion the unconditional
loss is alpha q^2 (alpha + lam^2) Var(u) with q = 1/(lam^2 + alpha(1 - beta rho));
the optimal simple rule x_t = -omega_c u_t has omega_c = lam/(lam^2 + alpha(1 - beta rho)^2).
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

import puremacro.dsge.policy as policy_mod
from puremacro.dsge._parser import Parser, Tokenizer
from puremacro.dsge._results import DiscretionaryPolicyResult, PolicyResult
from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.policy import discretionary_policy, lq_commitment, optimal_policy, osr
from puremacro.dsge.ramsey import RamseyResult, derive_ramsey_focs, ramsey_model

CALIB = {"beta": 0.99, "phi": 1.0, "lam": 0.1, "alpha": 0.25, "rho": 0.5, "mu": 0.8}


def _cgg_model(params, rule="i = phi_pi*pi + phi_x*x;"):
    values = {"phi_pi": 1.5, "phi_x": 0.5, **params}
    text = ("var x pi i g u;\nvarexo eps_g eps_u;\n"
            f"parameters {' '.join(values)};\n"
            + "".join(f"{k} = {float(v)!r};\n" for k, v in values.items())
            + "model;\n"
            "x = x(+1) - phi*(i - pi(+1)) + g;\n"
            "pi = lam*x + beta*pi(+1) + u;\n"
            f"{rule}\n"
            "g = mu*g(-1) + eps_g;\nu = rho*u(-1) + eps_u;\nend;\n"
            "shocks;\nvar eps_g; stderr 1;\nvar eps_u; stderr 1;\nend;\n")
    return build_dynare(text)


def _loss_disc_closed_form(beta, lam, alpha, rho, **_):
    q = 1.0 / (lam**2 + alpha * (1.0 - beta * rho))
    return alpha * q**2 * (alpha + lam**2) / (1.0 - rho**2)


def _solve(model, beta=0.99, alpha=0.25, **kwargs):
    return discretionary_policy(model, ["pi", "x"], {"pi": 1.0, "x": alpha}, "i", beta=beta, **kwargs)


@pytest.fixture(scope="module")
def cgg():
    return _cgg_model(CALIB)


# ---------------------------------------------------------------------------
# 1. stabilization_bias is NaN (with a warning) when it cannot be computed
# ---------------------------------------------------------------------------

def test_computed_bias_unchanged_and_silent(cgg):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = _solve(cgg)
    # Dennis iteration stops at ||F_{k+1} - F_k|| < 1e-9 (default tol), about 2e-9 relative here.
    assert res.loss == pytest.approx(_loss_disc_closed_form(**CALIB), rel=1e-8)
    assert res.stabilization_bias == pytest.approx(res.loss - res.commitment_result.loss, abs=1e-12)
    assert res.stabilization_bias == pytest.approx(1.513166984879534, rel=1e-9)
    assert "Stabilization bias" in res.summary()


def test_bias_is_nan_and_warns_when_commitment_solve_raises(cgg, monkeypatch):
    def failing_commitment(*args, **kwargs):
        raise np.linalg.LinAlgError("forced QZ failure")

    monkeypatch.setattr(policy_mod, "lq_commitment", failing_commitment)
    with pytest.warns(RuntimeWarning, match=r"(?s)commitment solve failed.*LinAlgError.*forced QZ failure"):
        res = _solve(cgg)
    assert np.isnan(res.stabilization_bias)
    assert res.commitment_result is None
    assert np.isfinite(res.loss)  # the discretion solution itself is unaffected
    assert "unavailable" in res.summary()


@pytest.mark.parametrize("criterion,field", [("unconditional", "loss"), ("conditional", "conditional_loss")])
def test_bias_is_nan_and_warns_when_commitment_loss_is_nan(cgg, monkeypatch, criterion, field):
    true_commitment = policy_mod.lq_commitment

    def nan_loss_commitment(*args, **kwargs):
        return dataclasses.replace(true_commitment(*args, **kwargs), **{field: float("nan")})

    monkeypatch.setattr(policy_mod, "lq_commitment", nan_loss_commitment)
    with pytest.warns(RuntimeWarning, match=f"commitment {criterion} loss is NaN"):
        res = _solve(cgg, loss_criterion=criterion)
    assert np.isnan(res.stabilization_bias)
    assert res.commitment_result is not None
    # The summary names the missing loss (and no longer reads "a unconditional loss").
    assert f"unavailable (the commitment {criterion} loss is not finite)" in res.summary()


def test_bias_is_nan_and_warns_when_discretion_loss_does_not_exist():
    # A unit-root cost-push shock has no stationary distribution: neither
    # unconditional loss exists, so the unconditional bias cannot be computed.
    model = _cgg_model({**CALIB, "rho": 1.0})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _solve(model)
    messages = [str(w.message) for w in caught]
    assert np.isnan(res.stabilization_bias)
    assert np.isnan(res.loss)
    assert res.commitment_result is None or np.isnan(res.commitment_result.loss)
    assert any("unconditional loss" in m and "NaN" in m for m in messages), messages


def test_bias_not_computed_is_nan_without_warning(cgg):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = _solve(cgg, compare_commitment=False)
    assert np.isnan(res.stabilization_bias)
    assert res.commitment_result is None
    assert "not computed" in res.summary()


def test_commitment_loss_is_nan_when_no_stationary_distribution():
    model = _cgg_model({**CALIB, "rho": 1.0})
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        comm = lq_commitment(model, ["pi", "x"], {"pi": 1.0, "x": 0.25}, "i", beta=0.99)
    # Before the fix this was a finite garbage number (about 4.9e15).
    assert np.isnan(comm.loss)


def test_result_default_bias_is_nan():
    field = {f.name: f for f in dataclasses.fields(DiscretionaryPolicyResult)}["stabilization_bias"]
    assert np.isnan(field.default)


# ---------------------------------------------------------------------------
# 2. The timeless flag is deprecated; docstrings describe what is solved
# ---------------------------------------------------------------------------

def test_lq_commitment_timeless_flag_warns_and_changes_nothing(cgg):
    args = (cgg, ["pi", "x"], {"pi": 1.0, "x": 0.25}, "i")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        base = lq_commitment(*args, beta=0.99)
    for flag in (True, False):
        with pytest.warns(FutureWarning, match="timeless"):
            other = lq_commitment(*args, beta=0.99, timeless=flag)
        np.testing.assert_array_equal(other.transition_matrix, base.transition_matrix)
        assert other.loss == base.loss and other.conditional_loss == base.conditional_loss


def test_optimal_policy_forwards_timeless_deprecation(cgg):
    with pytest.warns(FutureWarning, match="timeless"):
        optimal_policy(cgg, {"pi": 1.0, "x": 0.25}, "commitment", instruments="i", timeless=False)


def test_ramsey_model_timeless_flag_warns_and_changes_nothing(cgg):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        base = ramsey_model(cgg, "pi^2 + 0.25*x^2", 0.99, "i")
    with pytest.warns(FutureWarning, match="timeless"):
        other = ramsey_model(cgg, "pi^2 + 0.25*x^2", 0.99, "i", timeless=False)
    np.testing.assert_array_equal(other.augmented_model.solution.G, base.augmented_model.solution.G)


@pytest.mark.parametrize("obj", [lq_commitment, ramsey_model])
def test_docstrings_do_not_equate_zero_multiplier_with_timeless(obj):
    doc = " ".join(obj.__doc__.split())
    assert "timeless-perspective initial conditions (lambda_{-1} = 0)" not in doc
    assert "timeless-perspective stationary initial condition" not in doc
    assert "lambda_{-1} = 0" in doc and "Ramsey" in doc
    assert "deprecated" in doc.lower()


def test_module_docstring_does_not_equate_zero_multiplier_with_timeless():
    doc = " ".join(policy_mod.__doc__.split())
    assert "timeless perspective (lambda_{-1} = 0)" not in doc


# ---------------------------------------------------------------------------
# 3. Citations for the unconditional reversal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("doc", [discretionary_policy.__doc__, DiscretionaryPolicyResult.__doc__],
                         ids=["discretionary_policy", "DiscretionaryPolicyResult"])
def test_low_beta_reversal_is_credited_to_sauer(doc):
    flat = " ".join(doc.split())
    assert "Sauer" in flat and "Prop. 2" in flat
    assert "turns negative at low beta (Jensen and McCallum 2002)" not in flat
    assert "loses on average (Jensen and McCallum 2002)" not in flat


# ---------------------------------------------------------------------------
# 4. Readable first-order conditions that match the solved system
# ---------------------------------------------------------------------------

def _parse(text, names):
    toks = Tokenizer(text + ";").tokenize()
    parser = Parser(toks, text + ";")
    parser.variables = list(names)
    parser.parameters = []
    return parser.parse_expression()


def test_ramsey_focs_are_readable_and_parse_back(cgg):
    res = ramsey_model(cgg, "pi^2 + 0.25*x^2", 0.99, "i")
    assert len(res.focs) == len(cgg.variables) + len(res.multipliers)
    for foc in res.focs:
        for raw in ("BinOp(", "Const(", "Var(", "UnaryOp("):
            assert raw not in foc
    names = list(cgg.variables) + list(res.multipliers) + list(cgg.shocks)
    rng = np.random.default_rng(0)
    for var, node, text in zip(cgg.variables, res.foc_nodes, res.focs):
        head, expr, tail = (s.strip() for s in text.split(" = "))
        assert head == f"d L / d {var}" and tail == "0"
        parsed = _parse(expr, names)
        coords = node.variables() | parsed.variables()
        values = {c: float(rng.normal()) for c in coords}
        assert parsed.eval(values, {}) == pytest.approx(node.eval(values, {}), rel=1e-9, abs=1e-12)
    # CGG: dL/dx = 2*0.25*x - lam*zeta, so the Phillips-curve multiplier enters with -0.1.
    assert "0.5*x" in res.focs[0]


def test_ramsey_to_latex_renders_equations(cgg):
    res = ramsey_model(cgg, "pi^2 + 0.25*x^2", 0.99, "i")
    tex = res.to_latex()
    assert "BinOp" not in tex and "Const(" not in tex
    assert r"\frac{\partial \mathcal{L}}{\partial x_{t}}" in tex
    assert r"\pi_{t}" in tex
    assert tex.count(r"\\") >= len(res.focs)


def test_derive_ramsey_focs_strings_are_readable():
    from puremacro.dsge._ast import Const, Var

    U = Const(0.5) * Var("x", 0) ** 2 + Var("pi", 0) ** 2
    nkpc = Var("pi", 0) - Const(0.99) * Var("pi", 1) - Const(0.15) * Var("x", 0)
    _, strings, _ = derive_ramsey_focs(U, [nkpc], ["x", "pi"], beta=0.99)
    assert strings[0] == "d L / d x = x - 0.15*mult_0 = 0"
    assert strings[1] == "d L / d pi = 2*pi + mult_0 - mult_0(-1) = 0"
    assert strings[2] == "d L / d mult_0 = pi - 0.99*pi(+1) - 0.15*x = 0"


def test_foc_nodes_match_augmented_system(cgg):
    """The printed FOCs are the rows the solver assembles from the linear model matrices."""
    beta = 0.99
    res = ramsey_model(cgg, "pi^2 + 0.25*x^2", beta, "i")
    aug = res.augmented_model
    n, m = len(cgg.variables), len(res.multipliers)
    zero = {}
    for k, node in enumerate(res.foc_nodes):
        row = m + k
        for j, v in enumerate(cgg.variables):
            assert float(node.diff(v, 0).simplify().eval(zero, {})) == pytest.approx(aug._A_0[row, j], abs=1e-12)
        for i, mult in enumerate(res.multipliers):
            for lead, mat in ((0, aug._A_0), (-1, aug._A_minus), (1, aug._A_plus)):
                coef = float(node.diff(mult, lead).simplify().eval(zero, {}))
                assert coef == pytest.approx(mat[row, n + i], abs=1e-12)


# Nested differences inside a product used to lose their inner parentheses:
# a*(b - (c + d)) printed as a*(b - c + d). Each case: input, expected text.
_NESTED_CASES = [
    ("a*(b - (c + d))", "a*(b - c - d)"),
    ("a*(b - (c - d))", "a*(b - c + d)"),
    ("a*(b - (c - (d - a)))", "a*(b - c + d - a)"),
    ("-(a - (b + c))*d", "-(a - b - c)*d"),
    ("a/(b - (c + d))", "a/(b - c - d)"),
    ("(a - (b + c))^2", "(a - b - c)^2"),
    ("exp(a*(b - (c + d)))", "exp(a*(b - c - d))"),
    ("(a > b - (c + d))*a", "(a > b - c - d)*a"),
]


def _latex_group(s, i):
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1:j], j + 1
    raise ValueError(s)


def _latex_to_dynare(s):
    """Translate the LaTeX the FOC printer emits back into Dynare syntax (test helper)."""
    import re

    out, i = [], 0
    while i < len(s):
        if s.startswith(r"\frac", i):
            a, i = _latex_group(s, i + 5)
            b, i = _latex_group(s, i)
            out.append(f"(({_latex_to_dynare(a)})/({_latex_to_dynare(b)}))")
        elif s.startswith(r"\sqrt", i):
            a, i = _latex_group(s, i + 5)
            out.append(f"sqrt({_latex_to_dynare(a)})")
        elif s[i] == "{":
            a, i = _latex_group(s, i)
            if s.startswith("^", i):
                b, i = _latex_group(s, i + 1)
                out.append(f"(({_latex_to_dynare(a)})^({_latex_to_dynare(b)}))")
            else:
                out.append(f"({_latex_to_dynare(a)})")
        elif s.startswith(r"\left(", i):
            out.append("(")
            i += 6
        elif s.startswith(r"\right)", i):
            out.append(")")
            i += 7
        elif s.startswith(r"\,", i):
            out.append("*")
            i += 2
        elif s.startswith(r"\times 10^", i):
            b, i = _latex_group(s, i + 10)
            out.append(f"*10^({b})")
        elif s.startswith(r"\exp", i):
            out.append("exp")
            i += 4
        elif s.startswith(r"\ln", i):
            out.append("log")
            i += 3
        elif s.startswith(r"\mathrm", i) or s.startswith(r"\operatorname", i):
            a, i = _latex_group(s, i + (7 if s.startswith(r"\mathrm", i) else 13))
            out.append(a.replace(r"\_", "_"))
        elif s[i] == "\\":
            name = re.match(r"\\([A-Za-z]+)", s[i:]).group(1)
            out.append(name)
            i += len(name) + 1
        elif s[i] == "_":
            a, i = _latex_group(s, i + 1)
            out.append(f"({a[1:]})" if a[1:] else "")
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _parse_expr(text, names, params=()):
    toks = Tokenizer(text + ";").tokenize()
    parser = Parser(toks, text + ";")
    parser.variables = list(names)
    parser.parameters = list(params)
    return parser.parse_expression()


def _assert_same_value(node, text, names, params, rng, rel=1e-9, abs_tol=1e-12):
    parsed = _parse_expr(text, names, params)
    values = {c: float(rng.uniform(0.5, 1.5)) for c in node.variables() | parsed.variables()}
    assert parsed.eval(values, params) == pytest.approx(node.eval(values, params), rel=rel, abs=abs_tol), text


@pytest.mark.parametrize("source,expected", _NESTED_CASES)
def test_format_equation_keeps_signs_of_nested_differences(source, expected):
    from puremacro.dsge.ramsey import _format_equation, _format_equation_latex

    names = ["a", "b", "c", "d"]
    node = _parse_expr(source, names)
    text = _format_equation(node)
    assert text == expected
    rng = np.random.default_rng(3)
    _assert_same_value(node, text, names, {}, rng)
    _assert_same_value(node, _latex_to_dynare(_format_equation_latex(node)), names, {}, rng)


def _random_node(rng, depth):
    from puremacro.dsge._ast import BinOp, Call, Const, Param, UnaryOp, Var

    if depth == 0 or rng.random() < 0.25:
        r = rng.random()
        if r < 0.5:
            return Var("abcd"[rng.integers(4)], int(rng.integers(-1, 2)))
        if r < 0.7:
            return Param("pq"[rng.integers(2)])
        return Const(float(np.round(rng.uniform(-3, 3), 3)))
    r = rng.random()
    if r < 0.12:
        return UnaryOp("-", _random_node(rng, depth - 1))
    if r < 0.2:
        return Call(("exp", "log")[rng.integers(2)], (_random_node(rng, depth - 1),))
    op = "+-*/^"[rng.integers(5)]
    if op == "^":
        return BinOp("^", _random_node(rng, depth - 1), Const(float(rng.integers(-2, 4))))
    return BinOp(op, _random_node(rng, depth - 1), _random_node(rng, depth - 1))


def test_formatted_equations_round_trip_on_random_expressions():
    """Text and LaTeX forms evaluate to the node's value on 1500 random trees."""
    import math

    from puremacro.dsge.ramsey import _format_equation, _format_equation_latex

    names, params = list("abcd"), {"p": 0.7, "q": 1.3}
    rng = np.random.default_rng(2026)
    checked, bad = 0, []
    while checked < 1500:
        node = _random_node(rng, 4)
        values = {c: float(rng.uniform(0.5, 1.5)) for c in node.variables()}
        try:
            v0 = node.eval(values, params)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if not np.isfinite(v0) or abs(v0) > 1e8:
            continue
        checked += 1
        for text in (_format_equation(node), _latex_to_dynare(_format_equation_latex(node))):
            parsed = _parse_expr(text, names, params)
            v1 = parsed.eval({**{c: 1.0 for c in parsed.variables()}, **values}, params)
            # Ten printed significant digits, amplified at most modestly by the tree.
            if not math.isclose(v0, v1, rel_tol=1e-7, abs_tol=1e-9):
                bad.append((text, v0, v1))
    assert not bad, bad[:5]


_EULER_TAU_MOD = """
var c k y;
varexo e;
parameters bet alph delt tau;
bet = 0.96; alph = 0.33; delt = 0.1; tau = 0.02;
model;
  y = exp(0.1*e) * k(-1)^alph;
  c + k = y + (1-delt)*k(-1);
  1/c = bet/c(+1) * (alph*y(+1)/k + 1 - (delt + tau));
end;
steady_state_model;
k = ((1/bet - 1 + delt + tau)/alph)^(1/(alph-1));
y = k^alph;
c = y - delt*k;
end;
"""


def test_every_ramsey_foc_line_parses_back_including_constraints_and_latex():
    """Every printed condition (variables and constraints, text and LaTeX) equals its node."""
    import re

    from puremacro.dsge._parser import parse_mod_to_dag
    from puremacro.dsge.ramsey import _format_equation_latex

    dag = parse_mod_to_dag(_EULER_TAU_MOD)
    res = ramsey_model(dag, "log(c)", 0.96)
    nodes = list(res.foc_nodes) + list(res._constraint_nodes)
    assert len(nodes) == len(res.focs) == len(dag.variables) + len(res.multipliers)
    names = list(dag.variables) + list(res.multipliers) + list(dag.shocks)
    params = dict(dag.parameter_values)
    rng = np.random.default_rng(11)
    for node, line in zip(nodes, res.focs):
        head, expr, tail = (s.strip() for s in line.split(" = "))
        assert head.startswith("d L / d ") and tail == "0"
        # Coefficients print to ten significant digits (1/0.96 -> 1.041666667); terms of
        # order one that nearly cancel then leave an absolute error of about 1e-10.
        _assert_same_value(node, expr, names, params, rng, abs_tol=1e-8)
        _assert_same_value(node, _latex_to_dynare(_format_equation_latex(node)), names, params, rng,
                           abs_tol=1e-8)
    euler = res.focs[-1]
    assert "1 - delt - tau" in euler and "1 - delt + tau" not in euler
    assert "1 - delt - tau" in res.focs[0]  # d L / d c carries the lagged Euler multiplier
    # Every derived line appears in each presentation with the same signs.
    assert "1 - delt - tau" in res.to_markdown() and "1 - delt - tau" in res.summary()
    tex_rows = [r for r in res.to_latex().splitlines() if r.startswith(r"\frac{\partial")]
    assert len(tex_rows) == len(res.focs)
    assert all(re.search(r"- \\mathrm\{delt\} - \\tau", r) for r in (tex_rows[0], tex_rows[-1]))


def test_ramsey_headers_do_not_label_the_solution_timeless():
    res = ramsey_model(_cgg_model(CALIB), "pi^2 + 0.25*x^2", 0.99, "i")
    for text in (res.to_markdown(), res.to_typst()):
        assert "(Timeless Perspective)" not in text
        assert "commitment from the steady state" in text


def test_ramsey_docstring_says_foc_nodes_are_for_display():
    doc = " ".join(ramsey_model.__doc__.split())
    assert "foc_nodes" in doc and "display" in doc
    rdoc = " ".join(RamseyResult.__doc__.split())
    assert "foc_nodes" in rdoc


# ---------------------------------------------------------------------------
# 5. osr accuracy at the CGG random calibrations
# ---------------------------------------------------------------------------

def _cgg_draws(n=25, seed=1999):
    rng = np.random.default_rng(seed)
    return [{"beta": rng.uniform(0.9, 0.995), "phi": rng.uniform(0.3, 2.0), "lam": rng.uniform(0.02, 0.5),
             "alpha": rng.uniform(0.02, 1.0), "rho": rng.uniform(0.0, 0.9), "mu": rng.uniform(0.0, 0.9)}
            for _ in range(n)]


def _osr_allocation_error(p, **kwargs):
    tracking = _cgg_model(p, rule="i = g/phi + phi_pi*pi + phi_x*x;")
    best = osr(tracking, ["phi_pi", "phi_x"], {"pi": 1.0, "x": p["alpha"]},
               bounds={"phi_pi": (1.0, 10.0), "phi_x": (0.0, 10.0)}, **kwargs)
    omega_c = p["lam"] / (p["lam"] ** 2 + p["alpha"] * (1.0 - p["beta"] * p["rho"]) ** 2)
    x0 = best.optimal_model.irf("eps_u", horizon=1)["x"].iloc[0]
    return best, abs(x0 / omega_c + 1.0)


# Draws of notebook 66's loop (seed 1999) where the old defaults missed 1e-6:
# 3, 12, 14, 17, 22, 24 gave 9.1e-6, 2.7e-5, 1.2e-6, 2.7e-5, 1.3e-6, 9.3e-6.
_FAILING_DRAWS = (3, 12, 14, 17, 22, 24)


@pytest.mark.parametrize("draw", _FAILING_DRAWS)
def test_osr_allocation_accuracy_at_random_calibrations(draw):
    best, err = _osr_allocation_error(_cgg_draws()[draw])
    assert best.converged
    assert err < 1e-6


def test_osr_old_tolerances_reproduce_the_defect():
    # With SciPy's defaults the allocation error at draw 12 was 2.7e-5.
    _, err = _osr_allocation_error(_cgg_draws()[12], xatol=1e-4, fatol=1e-4)
    assert err > 1e-6


def test_osr_options_passthrough_overrides():
    p = _cgg_draws()[12]
    best, _ = _osr_allocation_error(p, options={"maxiter": 3})
    assert not best.converged
    assert "maximum number of iterations" in best.message.lower()


_NONLINEAR_MOD = """
var c k y;
varexo e;
parameters bet alph delt;
bet = 0.96; alph = 0.33; delt = 0.1;
model;
  y = exp(0.1*e) * k(-1)^alph;
  c + k = y + (1-delt)*k(-1);
  1/c = bet/c(+1) * (alph*y(+1)/k + 1 - delt);
end;
steady_state_model;
k = ((1/bet - 1 + delt)/alph)^(1/(alph-1));
y = k^alph;
c = y - delt*k;
end;
"""


def test_nonlinear_dag_focs_parse_back_and_shocks_evaluate_at_zero():
    """A shock inside a nonlinear term used to raise KeyError when linearising the DAG."""
    from puremacro.dsge._parser import parse_mod_to_dag

    dag = parse_mod_to_dag(_NONLINEAR_MOD)
    res = ramsey_model(dag, "log(c) - 0.5*(y/c)^2/(1 + k)", 0.96)
    names = list(dag.variables) + list(res.multipliers) + list(dag.shocks)
    rng = np.random.default_rng(1)
    for node, text in zip(res.foc_nodes, res.focs):
        assert "BinOp(" not in text
        expr = text.split(" = ")[1]
        toks = Tokenizer(expr + ";").tokenize()
        parser = Parser(toks, expr + ";")
        parser.variables = names
        parser.parameters = list(dag.parameter_values)
        parsed = parser.parse_expression()
        values = {c: float(rng.uniform(0.5, 1.5)) for c in node.variables() | parsed.variables()}
        # The display rounds constants to ten significant digits (_format_number),
        # so the parsed text agrees to about 1e-9 at best; it missed rel=1e-9 by 5%
        # on Windows CI. 1e-7 still catches any wrongly rendered term.
        assert parsed.eval(values, dag.parameter_values) == pytest.approx(
            node.eval(values, dag.parameter_values), rel=1e-7)
    # The shock enters the technology row of B_u with d y / d e = -0.1 * k^alph at e = 0.
    k_ss = ((1 / 0.96 - 1 + 0.1) / 0.33) ** (1 / (0.33 - 1))
    assert res.augmented_model._B_u[0, 0] == pytest.approx(-0.1 * k_ss**0.33, rel=1e-12)


# ---------------------------------------------------------------------------
# 6. osr never reports a placeholder number as a loss or a reduction
# ---------------------------------------------------------------------------

_OSR_ARGS = dict(bounds={"phi_pi": (1.0, 10.0), "phi_x": (0.0, 10.0)}, options={"maxiter": 6})


def _osr_tracking_model():
    return _cgg_model(CALIB, rule="i = g/phi + phi_pi*pi + phi_x*x;")


def test_osr_baseline_failure_is_nan_not_the_penalty(monkeypatch):
    model = _osr_tracking_model()
    cls = type(model)
    real = cls.theoretical_moments
    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:  # the baseline evaluation only
            raise np.linalg.LinAlgError("forced baseline failure")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(cls, "theoretical_moments", flaky)
    with pytest.warns(RuntimeWarning, match="baseline moments.*loss_initial is NaN"):
        best = osr(model, ["phi_pi", "phi_x"], {"pi": 1.0, "x": 0.25}, **_OSR_ARGS)
    assert np.isnan(best.loss_initial)  # was 1e6, the `penalty` argument
    assert np.isfinite(best.loss_opt)
    assert best.variance_table["variance_reduction_pct"].isna().all()  # was 0.0
    assert "Loss reduction          : unavailable" in best.summary()  # was "0.00%"


def _patch_resolve_after_minimize(monkeypatch, replacement):
    real_minimize = policy_mod.scipy.optimize.minimize
    real_solve = policy_mod._solve_with_params
    state = {"done": False}

    def minimize(*args, **kwargs):
        res = real_minimize(*args, **kwargs)
        state["done"] = True
        return res

    def solve(model, params):
        return replacement(real_solve, model, params) if state["done"] else real_solve(model, params)

    monkeypatch.setattr(policy_mod.scipy.optimize, "minimize", minimize)
    monkeypatch.setattr(policy_mod, "_solve_with_params", solve)


def test_osr_failed_resolve_is_nan_not_the_objective_value(monkeypatch):
    from puremacro.dsge.build import ModelError

    def fail(real_solve, model, params):
        raise ModelError("forced re-solve failure")

    _patch_resolve_after_minimize(monkeypatch, fail)
    with pytest.warns(RuntimeWarning, match=r"re-solving the model.*ModelError.*loss_opt is NaN"):
        best = osr(_osr_tracking_model(), ["phi_pi", "phi_x"], {"pi": 1.0, "x": 0.25}, **_OSR_ARGS)
    assert np.isnan(best.loss_opt)  # was the optimizer's last objective value
    assert best.optimal_model is None
    assert np.isfinite(best.loss_initial)
    assert best.variance_table["variance_reduction_pct"].isna().all()
    assert "unavailable" in best.summary()


def test_osr_indeterminate_resolve_is_nan(monkeypatch):
    def indeterminate(real_solve, model, params):
        return real_solve(model, {"phi_pi": 0.5, "phi_x": 0.0})  # violates the Taylor principle

    _patch_resolve_after_minimize(monkeypatch, indeterminate)
    with pytest.warns(RuntimeWarning, match="not determinate"):
        best = osr(_osr_tracking_model(), ["phi_pi", "phi_x"], {"pi": 1.0, "x": 0.25}, **_OSR_ARGS)
    assert np.isnan(best.loss_opt)


def test_osr_computed_losses_unchanged_and_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        best = osr(_osr_tracking_model(), ["phi_pi", "phi_x"], {"pi": 1.0, "x": 0.25}, **_OSR_ARGS)
    assert np.isfinite(best.loss_initial) and np.isfinite(best.loss_opt)
    table = best.variance_table
    expected = 100.0 * (table["var_initial"] - table["var_optimal"]) / table["var_initial"]
    np.testing.assert_allclose(table["variance_reduction_pct"], expected, rtol=1e-12)
    assert f"{100.0 * (best.loss_initial - best.loss_opt) / best.loss_initial:.2f}%" in best.summary()
