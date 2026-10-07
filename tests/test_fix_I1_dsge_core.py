"""Re-solve paths of the DSGE core see the model a fresh load would give (I1-dsge-core).

Every consumer that re-solves a ``build_dynare`` / ``load_mod`` model at new
parameter values (estimation, ``osr``, widgets, prior/posterior IRFs, SMC,
analytic gradients) must give what ``load_mod(text, params=...)`` gives at
those values. Up to 4.3.0 they did not, in three ways:

* **Frozen state set.** ``build_dynare`` detects states as the variables whose
  column of ``df/dy_{t-1}`` is non-zero *at the calibration*. Re-solving with
  ``states=model.states`` dropped every lag whose coefficient is calibrated to
  0: SW07 calibrates ``crhoms``, ``crhopinf``, ``crhow``, ``cmap`` and ``cmaw``
  to 0 and estimates all five, so its log-likelihood at the ``estimated_params``
  starting values was -1766.35 instead of -890.996 (the value with the states
  detected at that point, which is Dynare's ``M_.state_var``).
* **Shock covariance.** ``bayesian._resolve_model_with_params``, the SMC
  likelihood and the NUTS gradient re-solve passed no ``shock_cov``, so the
  declared ``shocks;`` block became the identity: SW07 sd(y) 28.894 instead of
  21.695 at unchanged parameters.
* **Stale steady state.** Most paths passed the calibration's steady state with
  ``check_steady_state=False``, so a parameter that moves the steady state
  (SW07's ``robs`` with ``csigma``) left it, and the measurement intercept, at
  the calibration's value.

The toy model with a zero-calibrated lag: ``y = rho*y(-1) + a*x + ey``,
``x = c*x(-1) + ex``. At ``rho = 0`` the automatic state set is ``('x',)``.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import puremacro.dsge as D
from puremacro.dsge.build import (
    ModelError,
    _make_observation_eq,
    _resolve_build_model,
    build,
)
from puremacro.dsge.dynare import (
    _resolve_dynare_model,
    _resolve_states,
    build_dynare,
    load_mod,
)
from puremacro.dsge.estimate import _stationary_init
from puremacro.dsge.observation import make_state_space_from_varobs
from puremacro.state_space import kalman_filter

SW07 = Path(D.__file__).parent / "_references" / "sw07_pfeifer.mod"

TOYLAG = """
var y x; varexo ey ex; parameters rho a c;
rho = 0; a = 1; c = 0.7;
model;
  y = rho*y(-1) + a*x + ey;
  x = c*x(-1) + ex;
end;
shocks; var ey; stderr 0.5; var ex; stderr 1; end;
varobs y;
estimated_params; rho, normal_pdf, 0.5, 0.3; end;
"""

# Steady state that moves with a parameter; the file gives it analytically.
TOYMEAN = """
var y; varexo e; parameters rho mu;
rho = 0.6; mu = 1;
model;
  y = mu + rho*(y(-1) - mu) + e;
end;
steady_state_model; y = mu; end;
shocks; var e; stderr 0.5; end;
varobs y;
estimated_params; mu, normal_pdf, 1, 2; end;
"""


def _var(m, v="y"):
    return float(m.theoretical_moments().covariance.loc[v, v])


def _simulate_toylag(rho=0.8, T=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.7 * x[t - 1] + rng.standard_normal()
        y[t] = rho * y[t - 1] + x[t] + 0.5 * rng.standard_normal()
    return y[:, None]


def _loglik(ssm, data):
    a0, P0 = _stationary_init(ssm)
    return float(kalman_filter(data, ssm, a0=a0, P0=P0)["loglik"])


# ---------------------------------------------------------------------------
# (a) The state set is not frozen at the calibration
# ---------------------------------------------------------------------------


def test_resolved_state_set_adds_zero_calibrated_lags():
    m = load_mod(TOYLAG)
    assert m.states == ("x",)                       # detected at rho = 0
    assert _resolve_states(m) == ("y", "x")         # Dynare's M_.state_var


def test_resolved_state_set_is_unchanged_without_inert_lags():
    m = load_mod(TOYLAG.replace("rho = 0;", "rho = 0.3;"))
    assert _resolve_states(m) == m.states == ("y", "x")


def test_estimation_state_space_equals_a_fresh_load_at_a_nonzero_lag():
    m = load_mod(TOYLAG)
    data = _simulate_toylag()
    eq = _make_observation_eq(m, m._estimated_params.specs, ["y"])
    fresh = load_mod(TOYLAG, params={"rho": 0.5})
    assert fresh.states == ("y", "x")
    ll_eq = _loglik(eq({"rho": 0.5}), data)
    ll_fresh = _loglik(make_state_space_from_varobs(fresh, ["y"]), data)
    assert ll_eq == pytest.approx(ll_fresh, rel=1e-10)
    # What the frozen state set gave: the y(-1) term silently dropped.
    frozen = build_dynare(
        m._dynare_equations, variables=m.variables, shocks=m.shocks,
        params={**m._params, "rho": 0.5}, guess=m.steady_state.to_dict(),
        states=m.states, shock_cov=m._shock_cov, verify_derivatives=False,
    )
    ll_frozen = _loglik(make_state_space_from_varobs(frozen, ["y"]), data)
    assert abs(ll_eq - ll_frozen) > 10.0


def test_inert_added_states_leave_the_likelihood_unchanged():
    m = load_mod(TOYLAG)
    data = _simulate_toylag()
    eq = _make_observation_eq(m, m._estimated_params.specs, ["y"])
    ll_eq = _loglik(eq({"rho": 0.0}), data)
    ll_base = _loglik(make_state_space_from_varobs(m, ["y"]), data)
    assert ll_eq == pytest.approx(ll_base, rel=1e-12)


def test_estimate_moves_a_parameter_calibrated_to_zero():
    """Up to 4.3.0 the likelihood was flat in rho: the mode search stayed at the start."""
    m = load_mod(TOYLAG)
    df = pd.DataFrame({"y": _simulate_toylag(rho=0.8, T=300, seed=3)[:, 0]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = m.estimate(df, n_draws=200, n_chains=1, burn_in=50, seed=0)
    assert res.mode["rho"] == pytest.approx(0.8, abs=0.1)


def test_sw07_likelihood_at_the_initial_values_uses_every_lag():
    base = load_mod(SW07)
    assert set(_resolve_states(base)) - set(base.states) == {
        "ms", "spinf", "sw", "epinfma", "ewma",
    }
    obs = list(base._varobs)
    specs = base._estimated_params.specs
    th = base._estimated_params.initial_params()
    ssm = _make_observation_eq(base, specs, obs)(th)
    merged = {**base._params, **{s.name: th[s.name] for s in specs if s.kind == "param"}}
    auto = build_dynare(
        base._dynare_equations, variables=base.variables, shocks=base.shocks,
        params=merged, guess=base.steady_state.to_dict(), verify_derivatives=False,
    )
    assert {"ms", "spinf", "sw", "epinfma", "ewma"} <= set(auto.states)
    raw = pd.read_csv(Path(D.__file__).parent / "_sw07_data.csv", comment="#")
    data = raw.rename(columns={
        "gdp_growth": "dy", "cons_growth": "dc", "inv_growth": "dinve",
        "wage_growth": "dw", "log_hours": "labobs", "infl": "pinfobs",
        "ffr": "robs",
    })[obs].to_numpy(float)
    ll = _loglik(ssm, data)
    ll_auto = _loglik(make_state_space_from_varobs(auto, obs, shock_cov=np.asarray(ssm.Q)), data)
    assert ll == pytest.approx(ll_auto, rel=1e-9)


@pytest.mark.parametrize("path", ["osr", "widgets", "bayesian", "gradients", "resolve"])
def test_every_resolve_path_keeps_a_zero_calibrated_lag(path):
    from puremacro.dsge._gradients import _re_solve_model
    from puremacro.dsge.bayesian import _resolve_model_with_params
    from puremacro.dsge.policy import _solve_with_params
    from puremacro.dsge.widgets import _fast_resolve

    m = load_mod(TOYLAG)
    new = {"rho": 0.5}
    solved = {
        "osr": lambda: _solve_with_params(m, new),
        "widgets": lambda: _fast_resolve(m, new),
        "bayesian": lambda: _resolve_model_with_params(m, new),
        "gradients": lambda: _re_solve_model(m, {**m._params, **new}),
        "resolve": lambda: _resolve_dynare_model(m, new),
    }[path]()
    fresh = load_mod(TOYLAG, params=new)
    assert _var(solved) == pytest.approx(_var(fresh), rel=1e-10)
    assert "y" in solved.states


def test_callable_zero_lag_found_by_the_generic_probe():
    def eqs(lead, curr, lag, shocks, p):
        return [curr.y - p.rho * lag.y - curr.x - shocks.ey,
                curr.x - p.c * lag.x - shocks.ex]

    kw = dict(variables=["y", "x"], shocks=["ey", "ex"], guess={"y": 0.0, "x": 0.0})
    m = build_dynare(eqs, params={"rho": 0.0, "c": 0.7}, **kw)
    assert m.states == ("x",)
    assert _resolve_states(m) == ("y", "x")
    fresh = build_dynare(eqs, params={"rho": 0.5, "c": 0.7}, **kw)
    assert _var(_resolve_dynare_model(m, {"rho": 0.5})) == pytest.approx(_var(fresh), rel=1e-10)


def test_callable_lag_the_probe_misses_is_caught_after_the_solve():
    """A coefficient rho**6 is below the detection threshold at the probe point."""
    def eqs(lead, curr, lag, shocks, p):
        return [curr.y - p.rho ** 6 * lag.y - curr.x - shocks.ey,
                curr.x - p.c * lag.x - shocks.ex]

    kw = dict(variables=["y", "x"], shocks=["ey", "ex"], guess={"y": 0.0, "x": 0.0})
    m = build_dynare(eqs, params={"rho": 0.0, "c": 0.7}, **kw)
    assert _resolve_states(m) == ("x",)
    re = _resolve_dynare_model(m, {"rho": 0.9})
    fresh = build_dynare(eqs, params={"rho": 0.9, "c": 0.7}, **kw)
    assert re.states == ("y", "x")
    assert _var(re) == pytest.approx(_var(fresh), rel=1e-10)
    # The grown set is kept for later re-solves of the same model.
    assert _resolve_states(m) == ("y", "x")


# ---------------------------------------------------------------------------
# (b) The declared shock covariance survives every re-solve
# ---------------------------------------------------------------------------


def test_bayesian_resolver_keeps_the_declared_shock_covariance_sw07():
    from puremacro.dsge.bayesian import _resolve_model_with_params

    m = load_mod(SW07)
    sd = math.sqrt(_var(m))
    assert sd == pytest.approx(21.6952, abs=1e-3)
    assert math.sqrt(_var(_resolve_model_with_params(m, {}))) == pytest.approx(sd, rel=1e-10)


def test_bayesian_resolver_keeps_the_declared_shock_covariance_toy():
    from puremacro.dsge.bayesian import _resolve_model_with_params

    m = load_mod(TOYLAG)
    np.testing.assert_allclose(_resolve_model_with_params(m, {"rho": 0.5})._shock_cov,
                               np.diag([0.25, 1.0]))


# ---------------------------------------------------------------------------
# (c) Parameter-dependent steady states and model-locals on every path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["osr", "widgets", "bayesian", "gradients"])
def test_every_resolve_path_moves_the_steady_state(path):
    from puremacro.dsge._gradients import _re_solve_model
    from puremacro.dsge.bayesian import _resolve_model_with_params
    from puremacro.dsge.policy import _solve_with_params
    from puremacro.dsge.widgets import _fast_resolve

    m = load_mod(TOYMEAN)
    new = {"mu": 3.0}
    solved = {
        "osr": lambda: _solve_with_params(m, new),
        "widgets": lambda: _fast_resolve(m, new),
        "bayesian": lambda: _resolve_model_with_params(m, new),
        "gradients": lambda: _re_solve_model(m, {**m._params, **new}),
    }[path]()
    assert solved.steady_state["y"] == pytest.approx(3.0, abs=1e-12)
    assert _var(solved) == pytest.approx(0.25 / (1 - 0.36), rel=1e-10)


def test_callable_steady_state_is_solved_for_when_it_moves():
    def eqs(lead, curr, lag, shocks, p):
        return [curr.y - p.mu - p.rho * (lag.y - p.mu) - shocks.e]

    m = build_dynare(eqs, variables=["y"], shocks=["e"], params={"rho": 0.6, "mu": 1.0},
                     guess={"y": 0.0})
    assert m.steady_state["y"] == pytest.approx(1.0, abs=1e-10)
    assert _resolve_dynare_model(m, {"mu": 3.0}).steady_state["y"] == pytest.approx(3.0, abs=1e-10)


def test_build_model_resolve_moves_the_steady_state_and_keeps_linearisation():
    def eqs(xp, x, e, p):
        return [xp.y - p.mu - p.rho * (x.y - p.mu) - e.e]

    m = build(eqs, variables=["y"], states=["y"], shocks=["e"],
              params={"rho": 0.6, "mu": 1.0}, guess={"y": 1.0}, linearize="level")
    re = _resolve_build_model(m, {"mu": 3.0})
    assert re.steady_state["y"] == pytest.approx(3.0, abs=1e-10)
    assert re.units == {"y": "level"}
    # The stored steady state is reused, bit for bit, when it still solves.
    same = _resolve_build_model(m, {"rho": 0.5})
    assert same.steady_state["y"] == m.steady_state["y"]


def test_widgets_resolve_of_a_build_model_no_longer_raises_typeerror():
    from puremacro.dsge.widgets import _fast_resolve

    def eqs(xp, x, e, p):
        return [xp.y - p.rho * x.y - e.e]

    m = build(eqs, variables=["y"], states=["y"], shocks=["e"], params={"rho": 0.5},
              guess={"y": 0.0}, linearize="level")
    re = _fast_resolve(m, {"rho": 0.8})
    assert float(re.solution.G[0, 0]) == pytest.approx(0.8, abs=1e-12)


TOYLOCAL = """
var x y; varexo e; parameters rho a;
rho = 0.5; a = 1;
model;
  #k = 2*a;
  x = rho*x(-1) + e;
  y = k*x;
end;
shocks; var e; stderr 1; end;
varobs y;
estimated_params; a, normal_pdf, 1, 0.5; end;
"""


def test_smc_likelihood_follows_model_locals_and_the_shock_block():
    from puremacro.dsge.smc import SMCSampler

    m = load_mod(TOYLOCAL.replace("stderr 1;", "stderr 0.5;"))
    rng = np.random.default_rng(1)
    x = np.zeros(120)
    for t in range(1, 120):
        x[t] = 0.5 * x[t - 1] + 0.5 * rng.standard_normal()
    df = pd.DataFrame({"y": 4.0 * x})
    smc = SMCSampler(m, df, ["y"], n_particles=10, seed=0)
    fresh = load_mod(TOYLOCAL.replace("stderr 1;", "stderr 0.5;").replace("a = 1;", "a = 2;"))
    ll_fresh = _loglik(make_state_space_from_varobs(fresh, ["y"], ridge=1e-4),
                       df[["y"]].to_numpy(float))
    assert smc._log_lik({"a": 2.0}) == pytest.approx(ll_fresh, rel=1e-10)


def test_smc_likelihood_is_not_flat_in_a_shock_standard_deviation():
    from puremacro.dsge.smc import SMCSampler

    text = TOYLOCAL.replace(
        "estimated_params; a, normal_pdf, 1, 0.5; end;",
        "estimated_params; stderr e, inv_gamma_pdf, 1, 2; end;",
    )
    m = load_mod(text)
    df = pd.DataFrame({"y": np.random.default_rng(2).standard_normal(80)})
    smc = SMCSampler(m, df, ["y"], n_particles=10, seed=0)
    assert list(smc._priors) == ["SE_e"]
    lo, hi = smc._log_lik({"SE_e": 0.3}), smc._log_lik({"SE_e": 1.5})
    assert np.isfinite(lo) and np.isfinite(hi)
    assert abs(lo - hi) > 1.0


def test_smc_likelihood_of_a_build_model_follows_the_particles():
    """A build() model used to be evaluated at its calibration for every particle."""
    from puremacro.dsge.priors import BetaPrior
    from puremacro.dsge.smc import SMCSampler

    def eqs(xp, x, e, p):
        return [xp.y - p.rho * x.y - e.e]

    m = build(eqs, variables=["y"], states=["y"], shocks=["e"], params={"rho": 0.5},
              guess={"y": 0.0}, linearize="level")
    df = pd.DataFrame({"y": _simulate_toylag(rho=0.8, T=100)[:, 0]})
    smc = SMCSampler(m, df, ["y"], n_particles=10, seed=0,
                     priors={"rho": BetaPrior(mean=0.5, std=0.2)})
    lo, hi = smc._log_lik({"rho": 0.2}), smc._log_lik({"rho": 0.8})
    assert np.isfinite(lo) and np.isfinite(hi) and hi - lo > 1.0


def test_nuts_gradient_path_follows_locals_and_steady_state():
    from puremacro.dsge._gradients import kalman_score, log_posterior_and_gradient
    from puremacro.dsge.priors import NormalPrior, log_prior

    m = load_mod(TOYMEAN)
    rng = np.random.default_rng(4)
    y = np.zeros(150)
    y[0] = 3.0
    for t in range(1, 150):
        y[t] = 3.0 + 0.6 * (y[t - 1] - 3.0) + 0.5 * rng.standard_normal()
    data = y[:, None]
    priors = {"mu": NormalPrior(mean=1.0, std=2.0)}

    def lp_at(mu):
        return log_posterior_and_gradient(np.array([mu]), ["mu"], m, data, ["y"], priors)

    lp, g = lp_at(3.0)
    fresh = load_mod(TOYMEAN, params={"mu": 3.0})
    ll_fresh, _ = kalman_score(data, make_state_space_from_varobs(fresh, ["y"]), [])
    assert lp - log_prior({"mu": 3.0}, priors) == pytest.approx(ll_fresh, rel=1e-10)
    # The intercept derivative reaches the analytic score.
    h = 1e-5
    fd = (lp_at(3.0 + h)[0] - lp_at(3.0 - h)[0]) / (2 * h)
    assert g[0] == pytest.approx(fd, rel=1e-5, abs=1e-6)


def test_state_space_sensitivity_of_a_zero_calibrated_lag():
    from puremacro.dsge._gradients import build_state_space_sensitivities

    m = load_mod(TOYLAG)
    sens = build_state_space_sensitivities(m, ["y"], ["rho"])["rho"]
    # alpha = [y_{t-1}, x_{t-1}; ey, ex]: d y_t / d y_{t-1} = rho, so dT[0, 0] = 1.
    assert sens.dT.shape == (4, 4)
    assert sens.dT[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_nuts_gradient_includes_a_zero_calibrated_lag():
    from puremacro.dsge._gradients import log_posterior_and_gradient
    from puremacro.dsge.priors import NormalPrior

    m = load_mod(TOYLAG)
    data = _simulate_toylag(rho=0.8, T=150, seed=5)
    priors = {"rho": NormalPrior(mean=0.5, std=0.3)}

    def lp_at(r):
        return log_posterior_and_gradient(np.array([r]), ["rho"], m, data, ["y"], priors)

    lp0, g0 = lp_at(0.0)
    h = 1e-5
    fd = (lp_at(h)[0] - lp_at(-h)[0]) / (2 * h)
    assert abs(fd) > 1.0
    assert g0[0] == pytest.approx(fd, rel=1e-4)


# ---------------------------------------------------------------------------
# (d) STEADY: allow_singular reaches build / build_dynare / load_mod
# ---------------------------------------------------------------------------


def _random_walk(xp, x, e, p):
    return [xp.y - x.y - e.eps, x.c - 2.0 * x.y]


def _duplicated(xp, x, e, p):
    return [xp.k - 0.5 * x.k - e.eps, xp.w - x.w + xp.k - 0.5 * x.k, x.c - 0.5 * x.k]


def test_package_exports_the_singularity_opt_in():
    from puremacro.dsge.steady import (
        StructuralSingularityWarning as W,
        allow_structural_singularity as A,
    )

    assert D.StructuralSingularityWarning is W
    assert D.allow_structural_singularity is A
    assert {"StructuralSingularityWarning", "allow_structural_singularity"} <= set(D.__all__)


def test_build_forwards_allow_singular_and_keeps_the_info():
    kw = dict(variables=["y", "c"], states=["y"], shocks=["eps"], params={},
              guess={"y": 4.0, "c": 3.0}, linearize="level")
    with pytest.raises(D.StructuralSingularityError):
        build(_random_walk, allow_singular=False, **kw)
    with pytest.warns(D.StructuralSingularityWarning):
        m = build(_random_walk, **kw)
    assert m.steady_state_info["structurally_singular"] is True
    assert m.steady_state_info["singularity_kind"] == "identity_rows"

    kw2 = dict(variables=["k", "w", "c"], states=["k", "w"], shocks=["eps"], params={},
               guess={"k": 0.3, "w": 1.0, "c": 0.2}, linearize="level")
    with pytest.raises(D.StructuralSingularityError):
        build(_duplicated, **kw2)
    with pytest.warns(D.StructuralSingularityWarning):
        m2 = build(_duplicated, allow_singular=True, **kw2)
    assert m2.steady_state_info["singularity_kind"] == "general"


def test_build_dynare_and_load_mod_forward_allow_singular():
    def rw(lead, curr, lag, shocks, p):
        return [curr.y - lag.y - shocks.eps, curr.c - 2.0 * curr.y]

    kw = dict(variables=["y", "c"], shocks=["eps"], params={}, guess={"y": 4.0, "c": 3.0},
              qz_criterium=1.0 + 1e-6)
    with pytest.raises(D.StructuralSingularityError):
        build_dynare(rw, allow_singular=False, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build_dynare(rw, **kw)
    assert m.steady_state_info["structurally_singular"] is True

    text = """
    var y c; varexo eps;
    model; y = y(-1) + eps; c = 2*y; end;
    initval; y = 4; c = 3; end;
    """
    with pytest.raises(D.StructuralSingularityError):
        load_mod(text, allow_singular=False, qz_criterium=1.0 + 1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m2 = load_mod(text, qz_criterium=1.0 + 1e-6)
    assert m2.steady_state_info["structurally_singular"] is True
    assert m2._qz_criterium == 1.0 + 1e-6


def test_regular_model_info_and_supplied_steady_state():
    m = load_mod(TOYMEAN)                       # steady_state_model: supplied
    assert m.steady_state_info is None
    m2 = load_mod(TOYMEAN.replace("steady_state_model; y = mu; end;", "initval; y = 0; end;"))
    info = m2.steady_state_info
    assert info["structurally_singular"] is False and info["converged"] is True


# ---------------------------------------------------------------------------
# (e) (f) Method wrappers forward the function's arguments
# ---------------------------------------------------------------------------


def test_osr_method_forwards_tolerances(monkeypatch):
    import puremacro.dsge.policy as policy

    seen = {}

    def fake_osr(model, **kw):
        seen.update(kw)
        return "ok"

    monkeypatch.setattr(policy, "osr", fake_osr)
    m = load_mod(TOYLAG)
    assert m.osr(["c"], {"y": 1.0}, xatol=1e-4, fatol=1e-3, options={"maxfev": 7}) == "ok"
    assert seen["xatol"] == 1e-4 and seen["fatol"] == 1e-3 and seen["options"] == {"maxfev": 7}


def test_identification_method_forwards_new_arguments(monkeypatch):
    import importlib

    # `puremacro.dsge.identification` is the function (the package rebinds the
    # name); the submodule is in sys.modules.
    ident = importlib.import_module("puremacro.dsge.identification")

    seen = {}

    def fake(model, **kw):
        seen.update(kw)
        return "ok"

    monkeypatch.setattr(ident, "identification", fake)
    m = load_mod(TOYLAG)
    out = m.identification(params=["rho"], fixed_params={"c": 0.6},
                           measurement_error={"y": 0.1}, n_freq=8,
                           frequencies=[0.5, 1.0], freq_type="gauss_legendre")
    assert out == "ok"
    assert seen["fixed_params"] == {"c": 0.6}
    assert seen["measurement_error"] == {"y": 0.1}
    assert seen["n_freq"] == 8 and seen["frequencies"] == [0.5, 1.0]
    assert seen["freq_type"] == "gauss_legendre"


def test_identification_method_fixed_params_reach_the_evaluation_point():
    m = load_mod(TOYLAG)
    a = m.identification(params=["rho"], fixed_params={"c": 0.3})
    b = D.identification(m, params=["rho"], fixed_params={"c": 0.3})
    np.testing.assert_allclose(a.j2_singular_values, b.j2_singular_values, rtol=1e-12)
    c = m.identification(params=["rho"])
    assert not np.allclose(a.j2_singular_values, c.j2_singular_values)


def test_estimate_rejects_unknown_fixed_params_without_the_preflight():
    m = load_mod(TOYLAG)
    df = pd.DataFrame({"y": _simulate_toylag(T=40)[:, 0]})
    with pytest.raises(ValueError, match=r"estimate\(\): fixed_params names \['cc'\]"):
        m.estimate(df, fixed_params={"cc": 0.5}, n_draws=10, burn_in=5)


def test_nuts_target_holds_fixed_params_fixed(monkeypatch):
    """The analytic NUTS target is observation_eq's log-posterior, fixed_params included.

    Up to 4.3.0 estimate() gave fixed_params to observation_eq only: the mode
    search ran at c = 0.3 and NUTS sampled the posterior at the calibrated
    c = 0.7 (log-likelihood -347.84 instead of -516.21 at rho = 0.5).
    """
    import puremacro.dsge._gradients as grads
    import puremacro.dsge.nuts as nuts_mod
    from puremacro.dsge.priors import log_prior

    captured = {"analytic_calls": 0}
    analytic = grads.log_posterior_and_gradient

    def counting(*a, **kw):
        captured["analytic_calls"] += 1
        return analytic(*a, **kw)

    def fake_nuts(target_fn, **kw):
        captured["target"] = target_fn
        captured["mode"] = kw["mode"]
        return "sampled"

    monkeypatch.setattr(grads, "log_posterior_and_gradient", counting)
    monkeypatch.setattr(nuts_mod, "nuts_sample", fake_nuts)

    m = load_mod(TOYLAG)
    data = _simulate_toylag(rho=0.8, T=200, seed=0)
    df = pd.DataFrame({"y": data[:, 0]})
    fixed = {"c": 0.3}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert m.estimate(df, method="nuts", fixed_params=fixed,
                          n_draws=10, n_chains=1, burn_in=5) == "sampled"

    target = captured["target"]
    lp, g = target(np.array([0.5]))
    assert captured["analytic_calls"] == 1          # the analytic gradient path

    specs = m._estimated_params.specs
    priors = m._estimated_params.priors()
    lprior = log_prior({"rho": 0.5}, priors)
    eq = _make_observation_eq(m, specs, ["y"], fixed_params=fixed)
    assert lp == pytest.approx(_loglik(eq({"rho": 0.5}), data) + lprior, rel=1e-9)
    eq_calib = _make_observation_eq(m, specs, ["y"])
    assert abs(lp - (_loglik(eq_calib({"rho": 0.5}), data) + lprior)) > 10.0

    h = 1e-5
    fd = (target(np.array([0.5 + h]))[0] - target(np.array([0.5 - h]))[0]) / (2 * h)
    assert g[0] == pytest.approx(fd, rel=1e-4)
    assert captured["mode"]["c"] == 0.3


# ---------------------------------------------------------------------------
# (h) Split R-hat through mcmc.gelman_rubin(split=True)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_draws", [100, 101])
def test_bayesian_split_rhat_matches_the_mcmc_helper(n_draws):
    """Even n: equal to the old hand-rolled split to 1e-12. Odd n: the middle draw goes.

    The old code interleaved the halves (chain 0 first half, chain 0 second
    half, chain 1 first half, ...); gelman_rubin(split=True) stacks every
    first half before every second half. The statistic is the same, but the
    sums run in a different order, so the two agree to rounding, not bit for
    bit.
    """
    from puremacro.dsge.bayesian import estimate_dsge_bayesian
    from puremacro.dsge.priors import BetaPrior, InvGammaPrior
    from puremacro.mcmc import gelman_rubin

    rng = np.random.default_rng(0)
    y = np.zeros(120)
    for t in range(1, 120):
        y[t] = 0.7 * y[t - 1] + 0.4 * rng.standard_normal()

    def loglik(p):
        rho, sig = (p["rho"], p["sigma"]) if isinstance(p, dict) else (p[0], p[1])
        e = y[1:] - rho * y[:-1]
        return float(-0.5 * np.sum(e ** 2) / sig ** 2 - len(e) * np.log(sig))

    priors = {
        "rho": BetaPrior(mean=0.6, std=0.15, lb=0.01, ub=0.99),
        "sigma": InvGammaPrior(mean=0.3, std=2.0, lb=0.01, ub=3.0),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = estimate_dsge_bayesian(loglik, priors, initial_params=np.array([0.6, 0.3]),
                                     n_draws=n_draws, n_burn=20, n_chains=2, seed=1)
    for i, name in enumerate(res.param_names):
        chains = res.chains[:, :, i]
        expect = float(gelman_rubin(chains, split=True)["R_hat"])
        assert res.diagnostics[f"r_hat_{name}"] == expect
        if n_draws % 2 == 0:
            half = n_draws // 2
            interleaved = np.empty((2 * chains.shape[0], half))
            for c in range(chains.shape[0]):
                interleaved[2 * c] = chains[c, :half]
                interleaved[2 * c + 1] = chains[c, half:2 * half]
            old = gelman_rubin(interleaved)["R_hat"]
            assert res.diagnostics[f"r_hat_{name}"] == pytest.approx(old, rel=1e-12)
