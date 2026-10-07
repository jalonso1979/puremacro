"""Follow-up to the DSGE-core re-solve fixes (FU-DSGE).

The shared re-solve (``_resolve_dynare_model``) made the reference model of
every consumer follow the parameters. What the review of that work found still
stale or missing, and what these tests pin down:

* **OccBin regimes (major).** The estimation likelihoods re-solved the
  reference at each draw but linearised every constrained regime at the
  calibration's steady state. OccBin linearises all regimes at one point, the
  reference's steady state; in a nonlinear model the stale Jacobian rows
  differed from the reference and were spliced into the binding regime as if
  the constraint had rewritten them (with two constraints the overlap check
  raised and the draw scored -inf).
* **Parsed models.** ``bayesian._resolve_model_with_params`` and
  ``widgets._fast_resolve`` solved a ``ParsedModelDAG`` at the calibration's
  steady state, with the draw alone as its parameters and no shock covariance.
* **Every re-solve path, numerical steady state.** A ``.mod`` without a
  ``steady_state_model`` block whose steady state moves with a parameter.
* **allow_singular** survives re-solves; **fixed_params** names are checked on
  every estimation route; **sensitivities** carry their state space;
  the **analytic score** keeps the steady-state term for tiny moves;
  **prior_predictive** maps ``SE_`` draws into the shock covariance and reads
  the file's priors; ``method='particle_smc'`` re-solves each particle.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

import puremacro.dsge.estimate as E
from puremacro.dsge._parser import parse_mod_to_dag
from puremacro.dsge.build import _make_observation_eq, _resolve_build_model, build
from puremacro.dsge.dynare import (
    _resolve_dynare_model,
    _resolve_parsed_dag_model,
    _shock_scale_overrides,
    build_dynare,
    load_mod,
)
from puremacro.dsge.occbin import (
    OccBinConstraint,
    _differing_rows,
    _extract_model_matrices,
    piecewise_kalman_filter,
)
from puremacro.dsge.priors import NormalPrior, log_prior


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# Steady state y = mu, given analytically.
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

# The same model with no steady_state_model block: the steady state must be
# solved for numerically at every new parameter vector.
TOYMEAN_NUM = """
var y; varexo e; parameters rho mu;
rho = 0.6; mu = 1;
model;
  y = mu + rho*(y(-1) - mu) + e;
end;
initval; y = 1; end;
shocks; var e; stderr 0.5; end;
varobs y;
estimated_params; mu, normal_pdf, 1, 2; end;
"""

VAR_Y = 0.25 / (1 - 0.36)


def _var(m, v="y"):
    return float(m.theoretical_moments().covariance.loc[v, v])


# ---------------------------------------------------------------------------
# OccBin: every regime at the reference's steady state
# ---------------------------------------------------------------------------

# z = y^2 makes row 1 nonlinear, so its Jacobian depends on the linearisation
# point; y reacts to r(-1), so a binding peg on r moves the observable.
OCC = """
var y z r s; varexo e u w; parameters rho mu;
rho = 0.6; mu = 1;
model;
  y = mu + rho*(y(-1) - mu) + 0.05*(z - mu^2) - 0.1*(r(-1) - mu^2) + e;
  z = y^2;
  r = z + u;
  s = z + w;
end;
steady_state_model; y = mu; z = mu^2; r = mu^2; s = mu^2; end;
shocks; var e; stderr 0.1; var u; stderr 0.2; var w; stderr 0.2; end;
"""


def _occ_setup():
    ref = load_mod(OCC)
    # Pegged regimes (r = 3.5, s = 4.4 in levels) and the filter's constraints
    # in deviations from the steady state at mu = 2 (r_ss = s_ss = 4).
    peg_r = E._auto_build_pegged_model(ref, OccBinConstraint(variable="r", threshold=3.5))
    peg_s = E._auto_build_pegged_model(ref, OccBinConstraint(variable="s", threshold=4.4))
    c_r = OccBinConstraint(variable="r", threshold=-0.5, operator="<")
    c_s = OccBinConstraint(variable="s", threshold=0.4, operator=">")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": 0.15 * rng.standard_normal(60)})
    return ref, peg_r, peg_s, c_r, c_s, df


def _one_point_loglik(ref, regimes, constraints, df, mu):
    """Piecewise log-likelihood with every regime linearised at the reference's steady state."""
    ref2 = _resolve_dynare_model(ref, {"mu": mu}, strict=False)
    cons = {
        k: build_dynare(m._dynare_equations, variables=m.variables, shocks=m.shocks,
                        params={**m._params, "mu": mu}, steady_state=ref2.steady_state,
                        check_steady_state=False, strict=False)
        for k, m in regimes.items()
    }
    return piecewise_kalman_filter(ref2, cons, df, varobs=["y"], constraints=constraints,
                                   horizon=30, Q=ref2._shock_cov)


def test_auto_pegged_regime_differs_from_the_reference_only_in_the_peg_row():
    ref, peg_r, peg_s, *_ = _occ_setup()
    ref_mats = _extract_model_matrices(ref)
    assert _differing_rows(ref_mats, _extract_model_matrices(peg_r, ref_model=ref)).tolist() == [2]
    assert _differing_rows(ref_mats, _extract_model_matrices(peg_s, ref_model=ref)).tolist() == [3]
    # Linearised at the reference's steady state, not at the threshold.
    assert peg_r.steady_state.equals(ref.steady_state)


@pytest.mark.parametrize("two", [False, True])
def test_piecewise_likelihood_linearises_every_regime_at_the_reference_steady_state(two):
    ref, peg_r, peg_s, c_r, c_s, df = _occ_setup()
    regimes = {"zlb": peg_r, "cap": peg_s} if two else {"zlb": peg_r}
    constraints = {"zlb": c_r, "cap": c_s} if two else {"zlb": c_r}
    priors = {"mu": NormalPrior(mean=1.0, std=1.0)}
    nlp = E._make_piecewise_neg_log_posterior(df, ref, regimes, constraints, priors,
                                              ["mu"], None, ["y"])
    ll = -nlp(np.array([2.0])) - log_prior({"mu": 2.0}, priors)
    want = _one_point_loglik(ref, regimes, constraints, df, 2.0)
    # The constraint binds in part of the sample, so the regimes matter.
    assert 0 < int((np.asarray(want.regime_history) != 0).sum()) < len(df)
    # Before this fix: 34.897 with one constraint, -inf (overlapping rows) with two.
    assert ll == pytest.approx(want.log_likelihood, rel=1e-12, abs=1e-10)
    assert ll == pytest.approx(35.2187781149, abs=1e-6)


def test_piecewise_regimes_seen_by_the_filter_share_the_reference_steady_state(monkeypatch):
    ref, peg_r, _, c_r, _, df = _occ_setup()
    seen = {}
    real = E.piecewise_kalman_filter

    def spy(ref_m, cons, data, **kw):
        seen["ref"], seen["cons"], seen["Q"] = ref_m, cons, kw.get("Q")
        return real(ref_m, cons, data, **kw)

    monkeypatch.setattr(E, "piecewise_kalman_filter", spy)
    nlp = E._make_piecewise_neg_log_posterior(df, ref, {"zlb": peg_r}, {"zlb": c_r},
                                              {"mu": NormalPrior(mean=1.0, std=1.0)},
                                              ["mu"], None, ["y"])
    assert np.isfinite(nlp(np.array([2.0])))
    ref_m, cons_m = seen["ref"], seen["cons"]["zlb"]
    assert float(ref_m.steady_state["y"]) == pytest.approx(2.0)
    assert cons_m.steady_state.equals(ref_m.steady_state)
    rows = _differing_rows(_extract_model_matrices(ref_m),
                           _extract_model_matrices(cons_m, ref_model=ref_m))
    assert rows.tolist() == [2]
    # The declared shocks; block reaches the filter (it used Q = I).
    np.testing.assert_allclose(seen["Q"], np.diag([0.01, 0.04, 0.04]))


@pytest.mark.parametrize("given", ["model", "mapping", "auto"])
def test_differentiable_occbin_regime_follows_the_reference_steady_state(monkeypatch, given):
    ref, peg_r, _, c_r, _, df = _occ_setup()
    seen = {}

    def spy(ref_m, cons_m, constraint, **kw):
        seen["ref"], seen["cons"] = ref_m, cons_m
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(E, "solve_differentiable_occbin", spy)
    constrained = {"model": peg_r, "mapping": {"zlb": peg_r}, "auto": None}[given]
    nlp = E._make_differentiable_occbin_neg_log_posterior(
        df, ref, c_r, constrained, {"mu": NormalPrior(mean=1.0, std=1.0)}, ["mu"], None, ["y"],
    )
    nlp(np.array([2.0]))
    ref_m, cons_m = seen["ref"], seen["cons"]
    assert float(ref_m.steady_state["y"]) == pytest.approx(2.0)
    assert cons_m.steady_state.equals(ref_m.steady_state)
    rows = _differing_rows(_extract_model_matrices(ref_m),
                           _extract_model_matrices(cons_m, ref_model=ref_m))
    # The auto peg replaces the first row with r at t (row 2 here).
    assert rows.tolist() == [2]


# ---------------------------------------------------------------------------
# Parsed models (ParsedModelDAG)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [TOYMEAN, TOYMEAN_NUM], ids=["block", "numerical"])
@pytest.mark.parametrize("path", ["bayesian", "widgets", "helper"])
def test_parsed_model_resolve_moves_the_steady_state_and_keeps_the_shocks(text, path):
    from puremacro.dsge.bayesian import _resolve_model_with_params
    from puremacro.dsge.widgets import _fast_resolve

    dag = parse_mod_to_dag(text)
    new = {"mu": 3.0}
    solved = {
        "bayesian": lambda: _resolve_model_with_params(dag, new),
        "widgets": lambda: _fast_resolve(dag, new),
        "helper": lambda: _resolve_parsed_dag_model(dag, new),
    }[path]()
    fresh = load_mod(text, params=new)
    # Before this fix: y_ss = 1.0 and var(y) = 1.5625 (identity covariance).
    assert float(solved.steady_state["y"]) == pytest.approx(3.0, abs=1e-10)
    assert float(fresh.steady_state["y"]) == pytest.approx(3.0, abs=1e-10)
    assert _var(solved) == pytest.approx(VAR_Y, rel=1e-10)


def test_parsed_model_slider_on_one_parameter_keeps_the_calibration():
    from puremacro.dsge.widgets import _fast_resolve

    dag = parse_mod_to_dag(TOYMEAN)
    # Before this fix: AttributeError, 'no parameter named mu' (the slider values
    # were the only parameters passed).
    m = _fast_resolve(dag, {"rho": 0.8})
    assert float(m.steady_state["y"]) == pytest.approx(1.0)
    assert _var(m) == pytest.approx(0.25 / (1 - 0.64), rel=1e-10)


# ---------------------------------------------------------------------------
# Every re-solve path on a .mod whose steady state is solved numerically
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [TOYMEAN, TOYMEAN_NUM], ids=["block", "numerical"])
@pytest.mark.parametrize("path", ["osr", "widgets", "bayesian", "gradients", "resolve"])
def test_every_resolve_path_moves_a_parameter_dependent_steady_state(text, path):
    from puremacro.dsge._gradients import _re_solve_model
    from puremacro.dsge.bayesian import _resolve_model_with_params
    from puremacro.dsge.policy import _solve_with_params
    from puremacro.dsge.widgets import _fast_resolve

    m = load_mod(text)
    new = {"mu": 3.0}
    solved = {
        "osr": lambda: _solve_with_params(m, new),
        "widgets": lambda: _fast_resolve(m, new),
        "bayesian": lambda: _resolve_model_with_params(m, new),
        "gradients": lambda: _re_solve_model(m, {**m._params, **new}),
        "resolve": lambda: _resolve_dynare_model(m, new),
    }[path]()
    assert float(solved.steady_state["y"]) == pytest.approx(3.0, abs=1e-9)
    assert _var(solved) == pytest.approx(VAR_Y, rel=1e-9)


@pytest.mark.parametrize("text", [TOYMEAN, TOYMEAN_NUM], ids=["block", "numerical"])
def test_estimation_observation_eq_moves_the_measurement_intercept(text):
    m = load_mod(text)
    specs = tuple(s for s in m._estimated_params.specs if s.name == "mu")
    eq = _make_observation_eq(m, specs, ["y"])
    assert float(np.asarray(eq({"mu": 3.0}).d)[0]) == pytest.approx(3.0, abs=1e-9)
    assert float(np.asarray(eq({"mu": -2.0}).d)[0]) == pytest.approx(-2.0, abs=1e-9)


@pytest.mark.parametrize("text", [TOYMEAN, TOYMEAN_NUM], ids=["block", "numerical"])
def test_smc_likelihood_moves_the_steady_state(text, monkeypatch):
    import puremacro.dsge.observation as obs_mod
    from puremacro.dsge.smc import SMCSampler

    seen = []
    real = obs_mod.make_state_space_from_varobs

    def spy(model, varobs, **kw):
        seen.append(model)
        return real(model, varobs, **kw)

    monkeypatch.setattr(obs_mod, "make_state_space_from_varobs", spy)
    m = load_mod(text)
    df = pd.DataFrame({"y": 3.0 + 0.5 * np.random.default_rng(0).standard_normal(40)})
    s = SMCSampler(m, df, ["y"], n_particles=10, n_stages=2, seed=0)
    ll_3 = s._log_lik({"mu": 3.0})
    assert float(seen[-1].steady_state["y"]) == pytest.approx(3.0, abs=1e-9)
    ll_1 = s._log_lik({"mu": 1.0})
    assert ll_3 > ll_1 + 50.0


@pytest.mark.parametrize("text", [TOYMEAN, TOYMEAN_NUM], ids=["block", "numerical"])
def test_particle_smc_resolves_each_particle(text, monkeypatch):
    """estimate(method='particle_smc') filtered every particle through the calibration."""
    import importlib

    # ``puremacro.dsge.particle_filter`` is also the name of the function the
    # package re-exports; the module itself is what estimate_dsge imports from.
    pf_mod = importlib.import_module("puremacro.dsge.particle_filter")
    smc_mod = importlib.import_module("puremacro.dsge.smc")

    seen = []

    class _Res:
        log_likelihood = -1.0

    def fake_pf(model, data, observed_vars, **kw):
        seen.append(model)
        return _Res()

    def fake_smc(log_lik, priors, **kw):
        return log_lik(np.array([3.0]))

    monkeypatch.setattr(pf_mod, "particle_filter", fake_pf)
    monkeypatch.setattr(smc_mod, "smc_estimate", fake_smc)
    m = load_mod(text)
    df = pd.DataFrame({"y": np.zeros(20)})
    assert m.estimate(df, method="particle_smc", n_draws=10) == -1.0
    assert float(seen[-1].steady_state["y"]) == pytest.approx(3.0, abs=1e-9)
    assert _var(seen[-1]) == pytest.approx(VAR_Y, rel=1e-9)


# ---------------------------------------------------------------------------
# allow_singular survives every re-solve
# ---------------------------------------------------------------------------


def _dup(xp, x, e, p):  # a general (not unit-root) structural singularity
    return [xp.k - 0.5 * x.k - e.eps, xp.w - x.w + xp.k - 0.5 * x.k, x.c - p.a - 0.5 * x.k]


def _dup_dyn(lead, curr, lag, sh, p):
    return [curr.k - 0.5 * lag.k - sh.eps, curr.w - lag.w + curr.k - 0.5 * lag.k,
            curr.c - p.a - 0.5 * curr.k]


DUP_MOD = """
var k w c; varexo eps; parameters a;
a = 0;
model;
  k = 0.5*k(-1) + eps;
  w = w(-1) - k + 0.5*k(-1);
  c = a + 0.5*k;
end;
initval; k = 0.3; w = 1; c = 0.2; end;
"""


def _resolvers():
    from puremacro.dsge.bayesian import _resolve_model_with_params
    from puremacro.dsge.policy import _solve_with_params
    from puremacro.dsge.widgets import _fast_resolve

    return {
        "osr": lambda m: _solve_with_params(m, {"a": 1.0}),
        "widgets": lambda m: _fast_resolve(m, {"a": 1.0}),
        "bayesian": lambda m: _resolve_model_with_params(m, {"a": 1.0}, qz_criterium=1 + 1e-6),
    }


@pytest.mark.parametrize("path", ["osr", "widgets", "bayesian", "shared"])
def test_build_allow_singular_survives_resolves(path):
    m = build(_dup, variables=["k", "w", "c"], states=["k", "w"], shocks=["eps"],
              params={"a": 0.0}, guess={"k": 0.3, "w": 1.0, "c": 0.2}, linearize="level",
              allow_singular=True)
    assert m._allow_singular is True
    fn = dict(_resolvers(), shared=lambda mm: _resolve_build_model(mm, {"a": 1.0}))[path]
    assert float(fn(m).steady_state["c"]) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("source", ["build_dynare", "load_mod"])
@pytest.mark.parametrize("path", ["osr", "widgets", "bayesian", "shared"])
def test_dynare_allow_singular_survives_resolves(source, path):
    if source == "build_dynare":
        m = build_dynare(_dup_dyn, variables=["k", "w", "c"], shocks=["eps"], params={"a": 0.0},
                         guess={"k": 0.3, "w": 1.0, "c": 0.2}, allow_singular=True,
                         qz_criterium=1 + 1e-6, strict=False)
    else:
        m = load_mod(DUP_MOD, allow_singular=True, qz_criterium=1 + 1e-6, strict=False)
    assert m._allow_singular is True
    fn = dict(_resolvers(), shared=lambda mm: _resolve_dynare_model(mm, {"a": 1.0}, strict=False))[path]
    assert float(fn(m).steady_state["c"]) == pytest.approx(1.0, abs=1e-9)


def test_explicit_allow_singular_false_overrides_the_stored_opt_in():
    from puremacro.dsge.steady import StructuralSingularityError

    m = build_dynare(_dup_dyn, variables=["k", "w", "c"], shocks=["eps"], params={"a": 0.0},
                     guess={"k": 0.3, "w": 1.0, "c": 0.2}, allow_singular=True,
                     qz_criterium=1 + 1e-6, strict=False)
    with pytest.raises(StructuralSingularityError):
        _resolve_dynare_model(m, {"a": 1.0}, strict=False, allow_singular=False)


# ---------------------------------------------------------------------------
# fixed_params names are checked whatever the estimation route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kw", [
    dict(method="kalman"),
    dict(method="nuts"),
    dict(method="piecewise_kalman"),
    dict(method="particle_smc"),
    dict(method="nuts", constraint=OccBinConstraint(variable="y", threshold=0.0)),
], ids=["kalman", "nuts", "piecewise_kalman", "particle_smc", "nuts_occbin"])
def test_every_estimation_route_rejects_unknown_fixed_params(kw, monkeypatch):
    def boom(*a, **k):  # nothing may run before the check
        raise AssertionError("estimation ran with an unknown fixed_params name")

    monkeypatch.setattr(E, "estimate_dsge", boom)
    m = load_mod(TOYMEAN)
    df = pd.DataFrame({"y": np.zeros(20)})
    with pytest.raises(ValueError, match=r"estimate\(\): fixed_params names \['typo_phi'\]"):
        m.estimate(df, fixed_params={"typo_phi": 9.0}, n_draws=10, burn_in=5, **kw)


# ---------------------------------------------------------------------------
# State-space sensitivities carry the state space they belong to
# ---------------------------------------------------------------------------

TOYLAG = """
var y x; varexo ey ex; parameters rho a c;
rho = 0; a = 1; c = 0.7;
model;
  y = rho*y(-1) + a*x + ey;
  x = c*x(-1) + ex;
end;
shocks; var ey; stderr 0.5; var ex; stderr 1; end;
"""


def test_sensitivities_carry_their_state_space_and_kalman_score_checks_shapes():
    from puremacro.dsge._gradients import build_state_space_sensitivities, kalman_score
    from puremacro.dsge.observation import make_state_space_from_varobs

    m = load_mod(TOYLAG)
    y = 0.5 * np.random.default_rng(3).standard_normal((50, 1))
    sens = build_state_space_sensitivities(m, ["y"], ["rho"])
    assert isinstance(sens, dict)
    assert sens.state_space.T.shape == sens["rho"].dT.shape == (4, 4)
    ll, g = kalman_score(y, sens.state_space, sens)

    # The score on the matching state space is the derivative of its loglik.
    h = 1e-6

    def ll_at(rho):
        mm = _resolve_dynare_model(m, {"rho": rho})
        return kalman_score(y, make_state_space_from_varobs(mm, ["y"]), {})[0]

    assert ll == pytest.approx(ll_at(0.0), rel=1e-12)
    assert g[0] == pytest.approx((ll_at(h) - ll_at(-h)) / (2 * h), rel=1e-5, abs=1e-6)

    # A state space built from the calibration's model has 3 states, not 4:
    # before this fix a bare matmul ValueError (4.3.0 ran but returned d/drho = 0).
    small = make_state_space_from_varobs(m, ["y"])
    assert small.T.shape == (3, 3)
    with pytest.raises(ValueError, match="same solved model"):
        kalman_score(y, small, sens)


# ---------------------------------------------------------------------------
# The analytic score keeps the steady-state term for small moves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k, mu", [(1e-4, 3.0), (1e-5, 3.0), (1e-4, 300.0), (1.0, 3.0)])
@pytest.mark.parametrize("block", [True, False], ids=["block", "numerical"])
def test_analytic_score_matches_finite_differences_for_a_tiny_steady_state_effect(k, mu, block):
    from puremacro.dsge._gradients import log_posterior_and_gradient

    ss_block = "steady_state_model; y = k*mu; end;" if block else f"initval; y = {k * mu}; end;"
    mod = f"""
var y; varexo e; parameters rho mu k;
rho = 0.6; mu = {mu}; k = {k};
model;
  y = k*mu + rho*(y(-1) - k*mu) + e;
end;
{ss_block}
shocks; var e; stderr 0.5; end;
"""
    m = load_mod(mod)
    y = 2.0 + 0.5 * np.random.default_rng(0).standard_normal((80, 1))
    priors = {"mu": NormalPrior(mean=mu, std=10.0)}
    vec = np.array([mu])
    _, g = log_posterior_and_gradient(vec, ["mu"], m, y, ["y"], priors)
    # Exact: mu enters only through the intercept k*mu. With the stationary
    # start, a_0 = y_0 - k*mu has variance 0.25/(1 - 0.36) and
    # a_t - 0.6 a_{t-1} has variance 0.25; the prior is centred at mu.
    a = y[:, 0] - k * mu
    d_ll = k * (a[0] / VAR_Y + (1 - 0.6) * np.sum(a[1:] - 0.6 * a[:-1]) / 0.25)
    # Before this fix, for the numerical steady state at k = 1e-4, mu = 3:
    # 0.000002 against 0.011 (the unmoved steady state passed as exact).
    assert g[0] == pytest.approx(d_ll, rel=1e-6, abs=1e-9)


# ---------------------------------------------------------------------------
# prior_predictive: SE_ draws and the file's priors
# ---------------------------------------------------------------------------

TOYSE = """
var y; varexo e; parameters rho mu;
rho = 0.6; mu = 1;
model;
  y = mu + rho*(y(-1) - mu) + e;
end;
steady_state_model; y = mu; end;
shocks; var e; stderr 0.5; end;
varobs y;
estimated_params; mu, normal_pdf, 1, 2; stderr e, inv_gamma_pdf, 0.5, 2; end;
"""


def test_prior_predictive_reads_the_file_priors_and_scales_the_shocks():
    from puremacro.dsge.bayesian import prior_predictive

    m = load_mod(TOYSE)
    # Before this fix: AttributeError ('EstimatedParams' object has no attribute 'get').
    res = prior_predictive(m, n_draws=40, irf=False, seed=1)
    assert set(res.param_names) == {"mu", "SE_e"}
    sd = res.valid_draws["SE_e"].to_numpy()
    assert res.n_valid == 40
    # Std.Dev.(y) = SE_e / sqrt(1 - rho^2) at every draw; it used to be the
    # constant 1.154701 (the identity covariance, SE_e ignored).
    assert res.prior_moments.loc["y_Std.Dev.", "mean"] == pytest.approx(
        float(np.mean(np.abs(sd))) / math.sqrt(1 - 0.36), rel=1e-10)
    assert res.prior_moments.loc["y_Std.Dev.", "std"] > 0.0


def test_shock_scale_overrides_keep_declared_correlations():
    m = build_dynare(
        lambda lead, curr, lag, sh, p: [curr.y - p.rho * lag.y - sh.e1 - sh.e2,
                                        curr.x - 0.5 * lag.x - sh.e1],
        variables=["y", "x"], shocks=["e1", "e2"], params={"rho": 0.5},
        steady_state={"y": 0.0, "x": 0.0},
        shock_cov=np.array([[1.0, 0.5], [0.5, 1.0]]),
    )
    structural, cov = _shock_scale_overrides(m, {"rho": 0.7, "SE_e1": 2.0})
    assert structural == {"rho": 0.7}
    np.testing.assert_allclose(cov, [[4.0, 1.0], [1.0, 1.0]])
    _, cov2 = _shock_scale_overrides(m, {"CORR_e1_e2": -0.2, "SE_e2": 3.0})
    np.testing.assert_allclose(cov2, [[1.0, -0.6], [-0.6, 9.0]])
    assert _shock_scale_overrides(m, {"rho": 0.7}) == ({"rho": 0.7}, None)
