"""Model-local ``#`` variables follow the parameters on every re-solve (LOCALS).

Dynare Reference Manual, section 4.5 "Model declaration": "every time this
variable appears in the model, Dynare will substitute it by the expression
assigned to the variable (if the model-local variable appears with a lead or
a lag ... shifting the expression accordingly)". A local such as
``#k = 2*a;`` is therefore a symbolic function of ``a``: re-solving the model
at ``a = 2`` must give exactly what a fresh load of the file edited to
``a = 2`` gives.

Before the fix the AST parser folded every parameter-only local to a number
at the file's calibration and the regex fallback stored it as a fake
parameter, so every re-solve path (``load_mod(params=)``, ``osr`` /
``_solve_with_params``, ``build_dynare(model._dynare_equations)`` as used by
estimation, SMC, widgets and gradients) kept the stale value. In SW07 the
estimated ``constebeta`` enters the model only through locals, so its
likelihood was exactly flat. Names that are Python keywords (a parameter
called ``lambda``) could not be compiled at all.

The closed form used throughout: for ``x = rho*x(-1) + e`` with unit shock
variance, ``Var(x) = 1/(1 - rho^2)``; ``y = k*x`` gives
``Var(y) = k^2/(1 - rho^2)``.
"""
from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import puremacro.dsge as D
import puremacro.dsge.dynare as dyn
from puremacro.dsge._parser import parse_mod_to_dag
from puremacro.dsge.build import ModelError, _Vec, _make_observation_eq
from puremacro.dsge.dynare import (
    DynareFeatureError,
    _mod_steady_state_at,
    build_dynare,
    load_mod,
    parse_mod,
)
from puremacro.dsge.policy import _solve_with_params

SW07 = Path(D.__file__).parent / "_references" / "sw07_pfeifer.mod"

TOY = """
var x y; varexo e; parameters rho a;
rho = 0.5; a = 1;
model;
  #k = 2*a;
  x = rho*x(-1) + e;
  y = k*x;
end;
shocks; var e; stderr 1; end;
"""


def _var(m, v="y"):
    return float(m.theoretical_moments().covariance.loc[v, v])


def _sd(m, v):
    return math.sqrt(_var(m, v))


@pytest.fixture
def regex_path(monkeypatch):
    """Force parse_mod onto the legacy regex reader."""

    def boom(*args, **kwargs):
        raise RuntimeError("forced regex fallback")

    monkeypatch.setattr(dyn, "parse_mod_to_dag", boom)
    return None


# ---------------------------------------------------------------------------
# Toy model: variance ratio 4 when a doubles, on every path
# ---------------------------------------------------------------------------


def test_toy_fresh_loads_match_the_closed_form():
    assert _var(load_mod(TOY)) == pytest.approx(4.0 / 0.75, rel=1e-10)
    assert _var(load_mod(TOY.replace("a = 1;", "a = 2;"))) == pytest.approx(
        16.0 / 0.75, rel=1e-10
    )


def test_toy_load_mod_params_override_reaches_the_local():
    base = _var(load_mod(TOY))
    assert _var(load_mod(TOY, params={"a": 2.0})) / base == pytest.approx(4.0, rel=1e-10)


def test_toy_osr_resolve_reaches_the_local():
    m = load_mod(TOY)
    assert _var(_solve_with_params(m, {"a": 2.0})) / _var(m) == pytest.approx(4.0, rel=1e-10)


def test_toy_build_dynare_from_the_compiled_callable_reaches_the_local():
    """The pattern estimation, SMC, widgets and gradients use."""
    m = load_mod(TOY)
    m2 = build_dynare(
        m._dynare_equations, variables=m.variables, shocks=m.shocks,
        params={**m._params, "a": 2.0}, guess=m.steady_state.to_dict(),
        shock_cov=m._shock_cov,
    )
    assert _var(m2) / _var(m) == pytest.approx(4.0, rel=1e-10)
    # Analytic Jacobian: dy-equation coefficient on x is -k = -2a.
    iy, ix = list(m.variables).index("y"), list(m.variables).index("x")
    eq_y = int(np.argmax(np.abs(m2._A_0[:, iy])))
    assert m2._A_0[eq_y, ix] / m2._A_0[eq_y, iy] == pytest.approx(-4.0, rel=1e-12)


def test_toy_widget_and_bayesian_resolvers_reach_the_local():
    from puremacro.dsge.bayesian import _resolve_model_with_params
    from puremacro.dsge.widgets import _fast_resolve

    m = load_mod(TOY)
    assert _var(_fast_resolve(m, {"a": 2.0})) / _var(m) == pytest.approx(4.0, rel=1e-10)
    m_b = _resolve_model_with_params(m, {"a": 2.0})
    # bayesian's resolver drops shock_cov (unit variance), which is the
    # file's value here, so the ratio is still exact.
    assert _var(m_b) / _var(m) == pytest.approx(4.0, rel=1e-10)


def test_toy_gradient_resolver_reaches_the_local():
    from puremacro.dsge._gradients import _re_solve_model

    m = load_mod(TOY)
    m_g = _re_solve_model(m, {**m._params, "a": 2.0})
    assert _var(m_g) / _var(m) == pytest.approx(4.0, rel=1e-10)


def test_toy_estimation_state_space_moves_with_the_local():
    text = TOY + "varobs y;\nestimated_params; a, normal_pdf, 1, 0.5; end;\n"
    m = load_mod(text)
    obs = _make_observation_eq(m, m._estimated_params.specs, ["y"])
    from puremacro.dsge.estimate import _stationary_init

    def obs_variance(ssm):
        _, P0 = _stationary_init(ssm)
        Z = np.atleast_2d(np.asarray(ssm.Z, dtype=float))
        H = np.atleast_2d(np.asarray(ssm.H, dtype=float))
        return float((Z @ P0 @ Z.T + H)[0, 0])

    s1, s2 = obs({"a": 1.0}), obs({"a": 2.0})
    assert obs.n_solves == 2
    assert obs_variance(s1) == pytest.approx(4.0 / 0.75, rel=1e-8)
    assert obs_variance(s2) / obs_variance(s1) == pytest.approx(4.0, rel=1e-8)


def test_local_used_with_leads_and_lags_is_shifted_and_tracks_parameters():
    with_local = """
    var x y; varexo e; parameters rho a; rho = 0.5; a = 2;
    model;
      #k = a*x;
      x = rho*x(-1) + e;
      y = k(-1) + 0.5*k(+1);
    end;
    shocks; var e; stderr 1; end;
    """
    inline = with_local.replace("#k = a*x;", "").replace(
        "y = k(-1) + 0.5*k(+1);", "y = a*x(-1) + 0.5*a*x(+1);"
    )
    for p in ({}, {"a": 3.0}):
        a, b = load_mod(with_local, params=p), load_mod(inline, params=p)
        assert _var(a) == pytest.approx(_var(b), rel=1e-12)


# ---------------------------------------------------------------------------
# The parsed DAG keeps locals symbolic
# ---------------------------------------------------------------------------


def test_dag_locals_are_symbolic_and_evaluable():
    dag = parse_mod_to_dag(TOY)
    assert dag.local_variables["k"].parameters() == {"a"}
    assert dag.evaluate_locals() == {"k": 2.0}
    assert dag.evaluate_locals({"a": 3.0}) == {"k": 6.0}
    # The equation still contains the parameter, not the number 2.
    assert any("a" in eq.parameters() for eq in dag.equations)


def test_sw07_locals_follow_constebeta_and_csigma():
    dag = parse_mod_to_dag(SW07.read_text(encoding="utf-8"))
    vals = dag.evaluate_locals()
    assert vals["cbeta"] == pytest.approx(1.0 / (1.0 + 0.742 / 100.0), rel=1e-14)
    cgamma = 1.0 + 0.3982 / 100.0
    assert vals["cbetabar"] == pytest.approx(vals["cbeta"] * cgamma ** (-1.5), rel=1e-14)
    moved = dag.evaluate_locals({"constebeta": 1.5, "csigma": 2.5})
    assert moved["cbeta"] == pytest.approx(1.0 / 1.015, rel=1e-14)
    assert moved["cbetabar"] == pytest.approx(moved["cbeta"] * cgamma ** (-2.5), rel=1e-14)
    assert "constebeta" in set().union(*(eq.parameters() for eq in dag.equations))


# ---------------------------------------------------------------------------
# SW07: a re-solve equals a fresh load; constebeta moves the likelihood
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sw07():
    text = SW07.read_text(encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = load_mod(text, order=1)
        fresh = load_mod(text.replace("csigma=1.5;", "csigma=2.5;"), order=1)
        over = load_mod(text, order=1, params={"csigma": 2.5})
    return text, base, fresh, over


def test_sw07_load_mod_params_equals_a_fresh_load(sw07):
    _, base, fresh, over = sw07
    for v in ("y", "c", "robs", "pinf", "inve"):
        assert _sd(over, v) == pytest.approx(_sd(fresh, v), rel=1e-10)
    assert _sd(fresh, "y") != pytest.approx(_sd(base, "y"), rel=1e-3)
    # steady_state_model re-evaluated at the override: robs = conster.
    assert over.steady_state["robs"] == pytest.approx(fresh.steady_state["robs"], rel=1e-12)
    assert fresh.steady_state["robs"] != pytest.approx(base.steady_state["robs"], rel=1e-3)
    np.testing.assert_allclose(over.solution.G, fresh.solution.G, atol=1e-10)
    np.testing.assert_allclose(over.solution.F, fresh.solution.F, atol=1e-10)


def test_sw07_osr_resolve_equals_a_fresh_load(sw07):
    _, base, fresh, _ = sw07
    re_solved = _solve_with_params(base, {"csigma": 2.5})
    for v in ("y", "c", "robs"):
        assert _sd(re_solved, v) == pytest.approx(_sd(fresh, v), rel=1e-10)


def test_sw07_steady_state_helper_matches_a_fresh_load(sw07):
    _, base, fresh, _ = sw07
    ss = _mod_steady_state_at(base, {**base._params, "csigma": 2.5})
    assert ss["robs"] == pytest.approx(fresh.steady_state["robs"], rel=1e-12)
    m = build_dynare(
        base._dynare_equations, variables=base.variables, shocks=base.shocks,
        params={**base._params, "csigma": 2.5}, steady_state=ss,
        shock_cov=base._shock_cov,
    )
    assert _sd(m, "robs") == pytest.approx(_sd(fresh, "robs"), rel=1e-10)
    assert _mod_steady_state_at(load_mod(TOY), {"a": 2.0}) is None


def test_sw07_constebeta_moves_the_state_space_and_the_likelihood(sw07):
    from puremacro.dsge.estimate import _stationary_init
    from puremacro.state_space import kalman_filter

    _, base, _, _ = sw07
    obs = _make_observation_eq(base, base._estimated_params.specs, list(base._varobs))
    th = base._estimated_params.initial_params()
    lo, hi = dict(th, constebeta=0.5), dict(th, constebeta=1.5)
    s_lo, s_hi = obs(lo), obs(hi)
    assert np.abs(np.asarray(s_lo.T) - np.asarray(s_hi.T)).max() > 1e-3
    assert np.abs(np.asarray(s_lo.d) - np.asarray(s_hi.d)).max() > 1e-2

    raw = pd.read_csv(Path(D.__file__).parent / "_sw07_data.csv", comment="#")
    data = raw.rename(columns={
        "gdp_growth": "dy", "cons_growth": "dc", "inv_growth": "dinve",
        "wage_growth": "dw", "log_hours": "labobs", "infl": "pinfobs",
        "ffr": "robs",
    })[list(base._varobs)].to_numpy(float)

    def loglik(ssm):
        a0, P0 = _stationary_init(ssm)
        return kalman_filter(data, ssm, a0=a0, P0=P0)["loglik"]

    ll_lo, ll_hi = loglik(s_lo), loglik(s_hi)
    assert np.isfinite(ll_lo) and np.isfinite(ll_hi)
    assert abs(ll_lo - ll_hi) > 1.0


def test_sw07_hoisted_equations_equal_the_plain_rendering():
    """compile_equations hoists parameter work; the arithmetic is unchanged."""
    import scipy

    dag = parse_mod_to_dag(SW07.read_text(encoding="utf-8"))
    fast = dag.compile_equations()
    exprs = [eq.to_python(shock_names=set(dag.shocks)) for eq in dag.equations]
    src = "def f(lead, curr, lag, shocks, params):\n    return [" + ", ".join(exprs) + "]\n"
    scope = {"np": np, "math": math, "scipy": scipy}
    exec(src, scope)
    plain = scope["f"]
    rng = np.random.default_rng(0)
    V, S, P = list(dag.variables), list(dag.shocks), list(dag.parameter_values)
    for _ in range(5):
        xs = [_Vec(V, rng.standard_normal(len(V))) for _ in range(3)]
        e = _Vec(S, rng.standard_normal(len(S)))
        p = _Vec(P, np.array([dag.parameter_values[k] * (1 + 0.1 * rng.standard_normal()) for k in P]))
        np.testing.assert_allclose(fast(*xs, e, p), plain(*xs, e, p), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# steady_state_model / initval re-evaluated under load_mod(params=)
# ---------------------------------------------------------------------------

GROWTH = """
var k c; varexo e; parameters alpha beta delta;
alpha = 0.33; beta = 0.99; delta = 0.025;
model;
  #mpk = alpha*k^(alpha-1);
  1/c = beta/c(+1)*(mpk(+1)*0 + alpha*k^(alpha-1) + 1 - delta);
  c + k = k(-1)^alpha + (1-delta)*k(-1) + e;
end;
steady_state_model;
  k = (alpha/(1/beta - 1 + delta))^(1/(1-alpha));
  c = k^alpha - delta*k;
end;
shocks; var e; stderr 0.01; end;
"""


def _growth_ss(alpha, beta=0.99, delta=0.025):
    k = (alpha / (1 / beta - 1 + delta)) ** (1 / (1 - alpha))
    return k, k ** alpha - delta * k


def test_load_mod_params_reevaluates_steady_state_model():
    """Before the fix the calibrated steady state was passed as exact and the
    residual check raised SteadyStateError at any override it depends on."""
    m = load_mod(GROWTH, params={"alpha": 0.36})
    k, c = _growth_ss(0.36)
    assert m.steady_state["k"] == pytest.approx(k, rel=1e-12)
    assert m.steady_state["c"] == pytest.approx(c, rel=1e-12)
    fresh = load_mod(GROWTH.replace("alpha = 0.33;", "alpha = 0.36;"))
    np.testing.assert_allclose(m.solution.G, fresh.solution.G, atol=1e-12)


def test_load_mod_params_reevaluates_initval():
    text = GROWTH.replace("steady_state_model;", "initval;")
    m = load_mod(text, params={"alpha": 0.36})
    k, c = _growth_ss(0.36)
    assert m.steady_state["k"] == pytest.approx(k, rel=1e-8)
    dag = parse_mod_to_dag(text)
    assert dag.evaluate_initval({"alpha": 0.36})["k"] == pytest.approx(k, rel=1e-12)


def test_aux_variables_inherit_their_root_in_reevaluated_steady_states():
    text = """
    var x_long y; varexo e; parameters rho m;
    rho = 0.5; m = 1;
    model;
      x_long = m*(1-rho) + rho*x_long(-1) + e;
      y = x_long(-2);
    end;
    steady_state_model; x_long = m; y = m; end;
    shocks; var e; stderr 1; end;
    """
    dag = parse_mod_to_dag(text)
    assert dag.auxiliary_variables == {
        "AUX_LAG_x_long_1": {"root": "x_long", "kind": "lag", "step": 1}
    }
    ss = dag.evaluate_steady_state({"m": 3.0})
    assert ss == {"x_long": 3.0, "y": 3.0, "AUX_LAG_x_long_1": 3.0}
    assert load_mod(text, params={"m": 3.0}).steady_state["AUX_LAG_x_long_1"] == 3.0


# ---------------------------------------------------------------------------
# Legacy regex reader: locals substituted, never frozen, never silent
# ---------------------------------------------------------------------------


def test_regex_path_does_not_store_locals_as_parameters(regex_path):
    parsed = parse_mod(TOY)
    assert "_dag" not in parsed
    assert parsed["params"] == {"rho": 0.5, "a": 1.0}
    base = _var(load_mod(TOY))
    assert base == pytest.approx(4.0 / 0.75, rel=1e-10)
    assert _var(load_mod(TOY, params={"a": 2.0})) / base == pytest.approx(4.0, rel=1e-10)
    m = load_mod(TOY)
    assert _var(_solve_with_params(m, {"a": 2.0})) / base == pytest.approx(4.0, rel=1e-10)


def test_regex_path_local_over_an_endogenous_variable(regex_path):
    text = TOY.replace("#k = 2*a;", "#k = 2*a*x;").replace("y = k*x;", "y = k;")
    assert _var(load_mod(text)) == pytest.approx(4.0 / 0.75, rel=1e-10)


def test_regex_path_shifts_locals_used_with_leads_and_lags(regex_path):
    with_local = """
    var x y; varexo e; parameters rho a; rho = 0.5; a = 2;
    model;
      #k = a*x;
      #kk = k + 0*a;
      x = rho*x(-1) + e;
      y = kk(-1) + 0.5*k(+1);
    end;
    shocks; var e; stderr 1; end;
    """
    inline = with_local.replace("#k = a*x;", "").replace("#kk = k + 0*a;", "").replace(
        "y = kk(-1) + 0.5*k(+1);", "y = a*x(-1) + 0.5*a*x(+1);"
    )
    for p in ({}, {"a": 3.0}):
        assert _var(load_mod(with_local, params=p)) == pytest.approx(
            _var(load_mod(inline, params=p)), rel=1e-12
        )


def test_regex_path_rejects_malformed_and_circular_locals(regex_path):
    with pytest.raises(DynareFeatureError, match="model-local"):
        parse_mod(TOY.replace("#k = 2*a;", "# = 2*a;"))
    with pytest.raises(ModelError, match="circular"):
        parse_mod(TOY.replace("#k = 2*a;", "#k = 2*j; #j = k;"))


def test_regex_path_reevaluates_steady_state_model(regex_path):
    m = load_mod(GROWTH, params={"alpha": 0.36})
    k, _ = _growth_ss(0.36)
    assert m.steady_state["k"] == pytest.approx(k, rel=1e-12)


# ---------------------------------------------------------------------------
# Python-keyword names
# ---------------------------------------------------------------------------

KW = """
var x yield; varexo e; parameters rho lambda;
rho = 0.5; lambda = 2;
model;
  #k = 2*lambda;
  x = rho*x(-1) + e;
  yield = lambda*x + 0*k;
end;
steady_state_model; x = 0; yield = lambda*0; end;
shocks; var e; stderr 1; end;
"""


def test_keyword_parameter_and_variable_names_parse_and_solve():
    parsed = parse_mod(KW)
    assert "_dag" in parsed, "the AST path must handle keyword names itself"
    m = load_mod(KW)
    assert _var(m, "yield") == pytest.approx(4.0 / 0.75, rel=1e-10)
    m2 = load_mod(KW, params={"lambda": 3.0})
    assert _var(m2, "yield") == pytest.approx(9.0 / 0.75, rel=1e-10)
    assert _var(_solve_with_params(m, {"lambda": 3.0}), "yield") == pytest.approx(
        9.0 / 0.75, rel=1e-10
    )


def test_keyword_name_inside_a_local_tracks_the_parameter():
    text = KW.replace("yield = lambda*x + 0*k;", "yield = k*x;")
    base = _var(load_mod(text), "yield")
    assert base == pytest.approx(16.0 / 0.75, rel=1e-10)
    assert _var(load_mod(text, params={"lambda": 1.0}), "yield") / base == pytest.approx(0.25, rel=1e-10)


def test_keyword_names_on_the_regex_path(regex_path):
    m = load_mod(KW)
    assert _var(m, "yield") == pytest.approx(4.0 / 0.75, rel=1e-10)
    assert _var(load_mod(KW, params={"lambda": 3.0}), "yield") == pytest.approx(9.0 / 0.75, rel=1e-10)
    text = KW.replace("lambda = 2;", "lambda = 2; rho = lambda/4;")
    assert parse_mod(text)["params"]["rho"] == pytest.approx(0.5)


def test_generated_code_uses_getattr_only_for_keywords():
    from puremacro.dsge._ast import Param, Var

    assert Param("lambda").to_python() == "getattr(params, 'lambda')"
    assert Param("beta").to_python() == "params.beta"
    assert Var("yield", 1).to_python() == "getattr(lead, 'yield')"
    assert Var("class", 0).to_python(shock_names={"class"}) == "getattr(shocks, 'class')"
