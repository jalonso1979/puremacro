"""``LinearModel.estimate`` — Bayesian estimation straight from a .mod file.

This is the connector the whole phase exists for. Everything it needs already
existed: priors that read Dynare's conventions, a Kalman filter with exact
diffuse initialisation, an audited Metropolis driver. What was missing was the
wiring, and in particular a measurement equation that did not have to be
hand-written per model.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import load_mod


_AR1_MOD = """
var y a;
varexo eps;
parameters rho;
rho = 0.7;
model;
  y = a;
  a = rho * a(-1) + eps;
end;
initval; a = 0; y = 0; end;
shocks; var eps; stderr 0.5; end;
varobs y;
estimated_params;
  rho, beta_pdf, 0.5, 0.2;
  stderr eps, inv_gamma_pdf, 0.5, 2;
end;
"""

_RHO_TRUE, _SIGMA_TRUE = 0.7, 0.5


def _simulate_ar1(T=400, seed=0):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(T) * _SIGMA_TRUE
    a = np.zeros(T)
    for t in range(1, T):
        a[t] = _RHO_TRUE * a[t - 1] + eps[t]
    return pd.DataFrame({"y": a})


@pytest.fixture(scope="module")
def ar1():
    return load_mod(_AR1_MOD)


def test_estimate_defaults_to_the_mod_files_varobs_and_estimated_params(ar1):
    """The caller should not have to repeat what the file already declares."""
    res = ar1.estimate(_simulate_ar1(T=120), n_draws=60, n_chains=1,
                       burn_in=200, seed=0)
    assert set(res.param_names) == {"rho", "SE_eps"}
    assert res.model_name != "unknown"


def test_estimate_recovers_known_parameters(ar1):
    """Data simulated at rho=0.7, sigma=0.5. The posterior must find them."""
    res = ar1.estimate(_simulate_ar1(T=400, seed=1), n_draws=800, n_chains=1,
                       burn_in=500, seed=0)
    summ = res.summary()
    for name, truth in (("rho", _RHO_TRUE), ("SE_eps", _SIGMA_TRUE)):
        mean, sd = summ.loc[name, "mean"], summ.loc[name, "std"]
        assert abs(mean - truth) < 3.0 * sd + 0.05, (name, mean, sd, truth)
    assert 0.05 <= res.accept_rates[0] <= 0.75


def test_estimate_result_supports_the_marginal_likelihood(ar1):
    res = ar1.estimate(_simulate_ar1(T=150), n_draws=200, n_chains=1,
                       burn_in=300, seed=0)
    assert res.log_post_mode is not None and np.isfinite(res.log_post_mode)
    assert np.isfinite(res.log_mdd("laplace"))


# --------------------------------------------------------------------------
# The closure: what re-solves, and what does not
# --------------------------------------------------------------------------
def _closure(model):
    from puremacro.dsge.build import _make_observation_eq

    return _make_observation_eq(
        model, model._estimated_params.specs, list(model._varobs),
    )


def test_shock_std_draws_do_not_trigger_a_re_solve(ar1):
    """Only kind == 'param' changes the model. A standard deviation writes
    straight into Q, and at 0.057 s per SW07 solve that distinction is the
    difference between a usable sampler and an unusable one."""
    eq = _closure(ar1)
    assert eq.n_solves == 0
    eq({"rho": 0.8, "SE_eps": 0.01})
    assert eq.n_solves == 1
    eq({"rho": 0.8, "SE_eps": 0.02})
    assert eq.n_solves == 1          # Q only — no re-solve
    eq({"rho": 0.7, "SE_eps": 0.02})
    assert eq.n_solves == 2          # structural — re-solved


def test_shock_std_reaches_q_even_without_a_re_solve(ar1):
    """The saving must not cost correctness: Q has to track the draw."""
    eq = _closure(ar1)
    ssm_a = eq({"rho": 0.8, "SE_eps": 0.01})
    ssm_b = eq({"rho": 0.8, "SE_eps": 0.02})
    np.testing.assert_allclose(ssm_a.Q, [[0.01 ** 2]])
    np.testing.assert_allclose(ssm_b.Q, [[0.02 ** 2]])
    np.testing.assert_array_equal(ssm_a.T, ssm_b.T)   # same solved model


def test_warm_start_does_not_change_the_answer(ar1):
    """The steady state is warm-started from the previous accepted draw. The
    speed-up must not buy a different model."""
    eq_warm = _closure(ar1)
    eq_warm({"rho": 0.55, "SE_eps": 0.5})
    warm = eq_warm({"rho": 0.85, "SE_eps": 0.5})
    cold = _closure(ar1)({"rho": 0.85, "SE_eps": 0.5})
    np.testing.assert_allclose(warm.T, cold.T, rtol=0, atol=1e-10)
    np.testing.assert_allclose(warm.Z, cold.Z, rtol=0, atol=1e-10)
    np.testing.assert_allclose(warm.d, cold.d, rtol=0, atol=1e-10)


def test_a_blanchard_kahn_violation_becomes_minus_inf_not_an_exception(ar1):
    """rho > 1 has no stable solution. The sampler must reject that draw, not
    crash — BlanchardKahnError subclasses RuntimeError, which
    _make_neg_log_posterior already treats as an infeasible draw."""
    from puremacro.dsge.estimate import _make_neg_log_posterior
    from puremacro.dsge.klein import BlanchardKahnError

    eq = _closure(ar1)
    with pytest.raises(BlanchardKahnError):
        eq({"rho": 1.5, "SE_eps": 0.5})

    priors = {
        "rho": {"dist": "uniform", "mean": 1.0, "std": 0.5, "lb": 0.01, "ub": 2.0},
        "SE_eps": {"dist": "invgamma", "mean": 0.5, "std": 2.0, "lb": 0.01, "ub": 5.0},
    }
    y = _simulate_ar1(T=80).to_numpy()
    f = _make_neg_log_posterior(y, eq, priors, ("rho", "SE_eps"), {})
    assert f(np.array([1.5, 0.5])) == np.inf      # rejected, not raised
    assert np.isfinite(f(np.array([0.7, 0.5])))


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------
def test_stochastic_singularity_is_reported_before_sampling(ar1):
    """`y = a` exactly, so observing both gives two series driven by one
    shock: the likelihood is -inf everywhere, whatever the starting point."""
    data = _simulate_ar1(T=100)
    data["a"] = data["y"]
    with pytest.raises(ValueError, match="stochastically singular"):
        ar1.estimate(data, varobs=["y", "a"], n_draws=20, n_chains=1,
                     burn_in=200, seed=0)


def test_a_model_without_mod_metadata_says_what_is_missing():
    from puremacro.dsge import build

    def eqs(xp, x, e, p):
        return [xp.a - p.rho * x.a - e.eps]

    m = build(eqs, variables=["a"], states=["a"], shocks=["eps"],
              params={"rho": 0.8}, guess={"a": 0.0})
    with pytest.raises(ValueError, match="estimated_params"):
        m.estimate(pd.DataFrame({"a": np.zeros(50)}))


def test_estimate_rejects_a_second_order_solution():
    sol = load_mod(_AR1_MOD, order=2)
    assert not hasattr(sol, "estimate")


@pytest.mark.slow
def test_sw07_end_to_end_from_the_mod_file():
    """The headline: sw07_pfeifer.mod straight to a posterior, with no
    hand-written observation equation. A smoke test of the whole path — 60
    draws is not a posterior, and the assertions say only that the path runs
    and produces finite, in-support numbers for all 36 estimated parameters."""
    from pathlib import Path

    import puremacro.dsge as D

    ref = Path(D.__file__).parent / "_references" / "sw07_pfeifer.mod"
    model = load_mod(ref)
    assert len(model._estimated_params.specs) == 36

    csv = Path(D.__file__).parent / "_sw07_data.csv"
    raw = pd.read_csv(csv, comment="#")
    data = raw.rename(columns={
        "gdp_growth": "dy", "cons_growth": "dc", "inv_growth": "dinve",
        "wage_growth": "dw", "log_hours": "labobs", "infl": "pinfobs",
        "ffr": "robs",
    })[list(model._varobs)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.estimate(data, n_draws=60, n_chains=1, burn_in=200, seed=0)

    assert len(res.param_names) == 36
    assert np.isfinite(res.draws).all()
    assert np.isfinite(res.log_posterior_trace).all()
    assert res.log_post_mode is not None and np.isfinite(res.log_post_mode)


def test_posterior_mode_matches_ordinary_least_squares_on_the_same_series(ar1):
    """An independent check of the whole connector.

    For ``y_t = a_t``, ``a_t = rho a_{t-1} + eps_t`` observed without
    measurement error, the likelihood is the AR(1) likelihood, so with a prior
    that is nearly flat over the relevant range the posterior mode must sit
    essentially where OLS does. Nothing in this comparison goes through the
    .mod parser, the varobs connector or the Kalman filter — which is what
    makes it worth running.
    """
    data = _simulate_ar1(T=600, seed=5)
    a = data["y"].to_numpy()
    y, ylag = a[1:], a[:-1]
    rho_ols = float(ylag @ y / (ylag @ ylag))
    sigma_ols = float((y - rho_ols * ylag).std(ddof=1))

    flat = {
        "rho": {"dist": "uniform", "mean": 0.5, "std": 0.28, "lb": 0.01, "ub": 0.99},
        "SE_eps": {"dist": "uniform", "mean": 1.0, "std": 0.55, "lb": 0.05, "ub": 2.0},
    }
    res = ar1.estimate(data, priors=flat, n_draws=20, n_chains=1,
                       burn_in=200, seed=0, mode_compute="csminwel")
    assert res.mode["rho"] == pytest.approx(rho_ols, abs=5e-3)
    assert res.mode["SE_eps"] == pytest.approx(sigma_ols, abs=5e-3)


def test_a_plain_dict_prior_override_keeps_the_shock_std_semantics(ar1):
    """Regression. A ``{name: prior}`` override carries no ``kind``, and
    labelling every entry "param" made SE_eps a structural parameter the
    equations never read: Q stayed at the declared 0.25 and the posterior was
    FLAT in it. The symptom was four optimisers agreeing on the log posterior
    (-422.030) at SE_eps values from 0.96 to 1.36.
    """
    from puremacro.dsge.build import _make_observation_eq

    flat = {
        "rho": {"dist": "uniform", "mean": 0.5, "std": 0.28, "lb": 0.01, "ub": 0.99},
        "SE_eps": {"dist": "uniform", "mean": 1.0, "std": 0.55, "lb": 0.05, "ub": 2.0},
    }
    from puremacro.dsge.priors import ensure_prior
    from puremacro.dsge._estimated_params import EstimatedParamSpec

    declared = {s.name: s for s in ar1._estimated_params.specs}
    specs = tuple(
        EstimatedParamSpec(kind=declared[k].kind, target=declared[k].target, name=k,
                           prior=ensure_prior(v), init=None,
                           lb=ensure_prior(v).lb, ub=ensure_prior(v).ub)
        for k, v in flat.items()
    )
    eq = _make_observation_eq(ar1, specs, ["y"])
    np.testing.assert_allclose(eq({"rho": 0.7, "SE_eps": 0.3}).Q, [[0.09]])
    np.testing.assert_allclose(eq({"rho": 0.7, "SE_eps": 1.2}).Q, [[1.44]])

    # End to end: the likelihood must now move with SE_eps.
    data = _simulate_ar1(T=300, seed=5)
    res = ar1.estimate(data, priors=flat, n_draws=20, n_chains=1,
                       burn_in=200, seed=0)
    assert res.mode["SE_eps"] != pytest.approx(1.0, abs=1e-3)


def test_a_prior_naming_nothing_the_model_knows_is_refused(ar1):
    with pytest.raises(Exception, match="neither a declared model parameter"):
        ar1.estimate(_simulate_ar1(T=60), n_draws=10, n_chains=1, burn_in=200,
                     seed=0,
                     priors={"nonsense": {"dist": "normal", "mean": 0.0,
                                          "std": 1.0, "lb": -3.0, "ub": 3.0}})


def test_a_structural_name_the_model_does_not_have_is_refused(ar1):
    """Defence in depth inside the closure itself."""
    from puremacro.dsge.build import _make_observation_eq
    from puremacro.dsge._estimated_params import EstimatedParamSpec
    from puremacro.dsge.priors import NormalPrior

    bogus = (EstimatedParamSpec(kind="param", target=("SE_eps",), name="SE_eps",
                                prior=NormalPrior(), init=None, lb=-9.0, ub=9.0),)
    with pytest.raises(Exception, match="no such parameters"):
        _make_observation_eq(ar1, bogus, ["y"])


def test_unpriored_param_in_estimated_params_is_refused():
    mod_text = """
    var y;
    varexo eps;
    parameters rho;
    rho = 0.7;
    model;
      y = rho * y(-1) + eps;
    end;
    initval; y = 0; end;
    shocks; var eps; stderr 0.5; end;
    varobs y;
    estimated_params;
      rho, 0.7;
    end;
    """
    model = load_mod(mod_text)
    data = pd.DataFrame({"y": [0.1, 0.2, 0.3, 0.4]})
    from puremacro.dsge.build import ModelError
    with pytest.raises(ModelError, match="requires a prior for every parameter in estimated_params"):
        model.estimate(data)

