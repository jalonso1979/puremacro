"""Tests for puremacro.dsge.estimate_dsge — the generic Bayesian DSGE engine."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.state_space import StateSpaceModel


_AR1_PRIORS = {
    "rho":   {"dist": "beta",     "mean": 0.5, "std": 0.2, "lb": 0.001, "ub": 0.99},
    "sigma": {"dist": "invgamma", "mean": 0.1, "std": 2.0, "lb": 0.01,  "ub": 5.0},
}


def _ar1_state_space(params: dict) -> StateSpaceModel:
    """y_t = x_t,  x_t = rho * x_{t-1} + sigma * eps_t."""
    rho = params["rho"]
    sigma = params["sigma"]
    return StateSpaceModel(
        T=np.array([[rho]]),
        Z=np.array([[1.0]]),
        R=np.array([[1.0]]),
        Q=np.array([[sigma ** 2]]),
        H=np.array([[1e-8]]),
        c=np.zeros(1),
        d=np.zeros(1),
    )


def _simulate_ar1(rho_true: float, sigma_true: float, T: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(T) * sigma_true
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = rho_true * x[t - 1] + eps[t]
    return pd.DataFrame({"y": x})


def test_estimate_dsge_returns_dsgeposteriorresult():
    from puremacro.dsge.estimate import estimate_dsge
    from puremacro.dsge._results import DSGEPosteriorResult
    data = _simulate_ar1(0.7, 0.5, T=200, seed=0)
    res = estimate_dsge(
        data, observation_eq=_ar1_state_space, priors=_AR1_PRIORS,
        observed_vars=["y"], initial_params={"rho": 0.5, "sigma": 0.4},
        n_chains=1, n_draws=300, burn_in=100, seed=0,
    )
    assert isinstance(res, DSGEPosteriorResult)
    assert res.draws.shape == (1, 300, 2)
    assert res.param_names == ("rho", "sigma")


def test_estimate_dsge_validates_missing_observed_var():
    from puremacro.dsge.estimate import estimate_dsge
    data = pd.DataFrame({"not_y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="missing columns"):
        estimate_dsge(
            data, observation_eq=_ar1_state_space, priors=_AR1_PRIORS,
            observed_vars=["y"], initial_params={"rho": 0.5, "sigma": 0.4},
            n_chains=1, n_draws=10, burn_in=5, seed=0,
        )


@pytest.mark.slow
def test_estimate_dsge_toy_ar1_recovers_rho():
    from puremacro.dsge.estimate import estimate_dsge
    data = _simulate_ar1(rho_true=0.7, sigma_true=0.5, T=500, seed=0)
    res = estimate_dsge(
        data, observation_eq=_ar1_state_space, priors=_AR1_PRIORS,
        observed_vars=["y"], initial_params={"rho": 0.5, "sigma": 0.4},
        n_chains=1, n_draws=2000, burn_in=500, seed=0,
    )
    posterior_rho_mean = float(res.draws[0, :, 0].mean())
    assert abs(posterior_rho_mean - 0.7) < 0.1, (
        f"posterior mean rho={posterior_rho_mean:.3f} far from true 0.7"
    )


def test_estimate_dsge_model_name_field_set():
    from puremacro.dsge.estimate import estimate_dsge
    data = _simulate_ar1(0.5, 0.3, T=100, seed=1)
    res = estimate_dsge(
        data, observation_eq=_ar1_state_space, priors=_AR1_PRIORS,
        observed_vars=["y"], initial_params={"rho": 0.5, "sigma": 0.3},
        model_name="ToyAR1",
        n_chains=1, n_draws=200, burn_in=50, seed=0,
    )
    assert res.model_name == "ToyAR1"


def test_estimate_dsge_kalman_singular_returns_neg_inf_in_log_posterior():
    """A pathological observation_eq that raises LinAlgError should be
    caught and produce -inf log-posterior, not crash the MCMC."""
    from puremacro.dsge.estimate import _make_neg_log_posterior

    priors = {"x": {"dist": "normal", "mean": 0.0, "std": 1.0, "lb": -10, "ub": 10}}

    def bad_obs(params):
        raise np.linalg.LinAlgError("simulated singular")

    nlp = _make_neg_log_posterior(
        y=np.zeros((10, 1)),
        observation_eq=bad_obs,
        priors=priors,
        names=("x",),
        fixed_params={},
    )
    assert nlp(np.array([0.5])) == np.inf


def test_estimate_dsge_hessian_overflow_falls_back_to_prior_stds(monkeypatch):
    """OverflowError inside the numerical Hessian (seen on some platforms
    when finite-differencing an exploding posterior) must not kill the
    estimation — it falls back to diag(prior_stds**2) proposals."""
    import puremacro.dsge.estimate as est_mod
    from puremacro.dsge.estimate import estimate_dsge

    def exploding_hessian(f, x, h=1e-4):
        raise OverflowError("math range error (simulated scipy numdiff)")

    monkeypatch.setattr(est_mod, "numerical_hessian", exploding_hessian)
    data = _simulate_ar1(0.7, 0.5, T=200, seed=0)
    with pytest.warns(UserWarning, match="OverflowError.*falling back"):
        res = estimate_dsge(
            data, observation_eq=_ar1_state_space, priors=_AR1_PRIORS,
            observed_vars=["y"], initial_params={"rho": 0.5, "sigma": 0.4},
            n_chains=1, n_draws=200, burn_in=50, seed=0,
        )
    assert res.draws.shape == (1, 200, 2)
    assert np.isfinite(res.draws).all()


# ---------------------------------------------------------------------------
# 2.4.1 regression tests (DSGE estimation audit)
# ---------------------------------------------------------------------------

def _ar1_with_measurement_error(params: dict) -> StateSpaceModel:
    """x_t = rho x_{t-1} + sigma_e eps_t,  y_t = x_t + sigma_m u_t."""
    return StateSpaceModel(
        T=np.array([[params["rho"]]]),
        Z=np.array([[1.0]]),
        R=np.array([[1.0]]),
        Q=np.array([[params["sigma_e"] ** 2]]),
        H=np.array([[params["sigma_m"] ** 2]]),
        c=np.zeros(1),
        d=np.zeros(1),
    )


def test_kalman_likelihood_uses_stationary_p0_not_the_diffuse_default():
    """The driver's log-likelihood must equal the exact Gaussian likelihood.

    A solved linear DSGE is stationary, so the Kalman recursion has to start
    from the unconditional state covariance solving ``P = T P T' + R Q R'``.
    ``kalman_filter``'s own default is ``P0 = 1e6 * I``, under which the
    likelihood LEVEL is a function of that arbitrary constant (here it is
    -6.21 log points off, and it moves by ~2.3 points per factor of 100),
    which makes marginal likelihoods and Bayes factors meaningless.
    """
    from scipy.stats import multivariate_normal
    from puremacro.dsge.estimate import _make_neg_log_posterior
    from puremacro.dsge.priors import log_prior

    rho, sig_e, sig_m, T = 0.85, 1.0, 0.4, 200
    rng = np.random.default_rng(7)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + sig_e * rng.standard_normal()
    y = (x + sig_m * rng.standard_normal(T))[:, None]

    # Closed-form log-likelihood of the stacked data under the stationary
    # AR(1)-plus-noise: gamma(h) = sigma_e^2/(1-rho^2) * rho^|h|, plus
    # sigma_m^2 on the diagonal.
    idx = np.arange(T)
    cov = (sig_e ** 2 / (1 - rho ** 2)) * rho ** np.abs(idx[:, None] - idx[None, :])
    cov = cov + sig_m ** 2 * np.eye(T)
    exact = float(multivariate_normal.logpdf(y.ravel(), mean=np.zeros(T), cov=cov))

    priors = {"rho": {"dist": "beta", "mean": 0.5, "std": 0.2,
                      "lb": 0.001, "ub": 0.99}}
    fixed = {"sigma_e": sig_e, "sigma_m": sig_m}
    nlp = _make_neg_log_posterior(
        y, _ar1_with_measurement_error, priors, ("rho",), fixed,
    )
    lp = log_prior({"rho": rho}, priors)
    got = -float(nlp(np.array([rho]))) - lp
    assert got == pytest.approx(exact, abs=1e-8), (
        f"driver loglik {got!r} != exact {exact!r} (diff {got - exact:.4f})"
    )


def test_stationary_init_solves_the_lyapunov_equation():
    from puremacro.dsge.estimate import _stationary_init

    ssm = _ar1_with_measurement_error(
        {"rho": 0.9, "sigma_e": 0.5, "sigma_m": 0.1}
    )
    a0, P0 = _stationary_init(ssm)
    assert a0 == pytest.approx(np.zeros(1))
    assert float(P0[0, 0]) == pytest.approx(0.25 / (1 - 0.81), rel=1e-12)


def test_stationary_init_refuses_a_unit_root_instead_of_guessing():
    from puremacro.dsge.estimate import _stationary_init

    ssm = _ar1_with_measurement_error(
        {"rho": 1.0, "sigma_e": 0.5, "sigma_m": 0.1}
    )
    assert _stationary_init(ssm) == (None, None)


def test_unit_root_draw_warns_by_name_rather_than_silently_going_diffuse():
    from puremacro.dsge.estimate import _make_neg_log_posterior

    y = _simulate_ar1(0.9, 0.5, T=50, seed=3)[["y"]].to_numpy()
    priors = {"rho": {"dist": "normal", "mean": 0.5, "std": 0.5,
                      "lb": 0.0, "ub": 1.5}}
    nlp = _make_neg_log_posterior(
        y, _ar1_with_measurement_error, priors, ("rho",),
        {"sigma_e": 0.5, "sigma_m": 0.1},
    )
    with pytest.warns(UserWarning, match=r"estimate_dsge:.*unit root"):
        nlp(np.array([1.0]))


def test_nearest_pd_returns_a_positive_definite_matrix():
    """cholesky() succeeding is not the same as being positive definite.

    The old Higham-style loop exited as soon as cholesky did not raise,
    which LAPACK can do on a matrix whose minimum eigenvalue is ~-1.7e-9.
    """
    from puremacro.dsge.estimate import _nearest_pd

    rng = np.random.default_rng(0)
    Qm, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    A = Qm @ np.diag([1e8, 1e7, 1e6, 1e5, 1e4, 1e-9]) @ Qm.T
    A = 0.5 * (A + A.T)
    out = _nearest_pd(A)
    assert float(np.min(np.linalg.eigvalsh(out))) > 0.0
    # and on an outright indefinite matrix
    B = Qm @ np.diag([1.0, 1.0, 1.0, -3.0, -7.0, 2.0]) @ Qm.T
    assert float(np.min(np.linalg.eigvalsh(_nearest_pd(B)))) > 0.0


def test_nearest_pd_rejects_non_finite_input():
    from puremacro.dsge.estimate import _nearest_pd

    with pytest.raises(ValueError, match="non-finite"):
        _nearest_pd(np.array([[1.0, np.nan], [np.nan, 1.0]]))


def test_initial_vec_warns_and_never_returns_minus_inf():
    """Out-of-support initial values were snapped to lb+1e-3 in silence,
    and an unbounded prior produced -inf."""
    from puremacro.dsge.estimate import _initial_vec_from_dict

    priors = {"p": {"dist": "beta", "mean": 0.5, "std": 0.2,
                    "lb": 0.01, "ub": 0.99}}
    with pytest.warns(UserWarning, match=r"initial_params\['p'\].*outside"):
        vec = _initial_vec_from_dict({"p": 0.0}, priors)
    assert vec[0] == pytest.approx(0.011)

    unbounded = {"m": {"dist": "normal", "mean": 1.25, "std": 1.0,
                       "lb": -np.inf, "ub": np.inf}}
    with pytest.warns(UserWarning, match="no entry for 'm'"):
        vec = _initial_vec_from_dict({}, unbounded)
    assert np.isfinite(vec).all()
    assert vec[0] == pytest.approx(1.25)


def test_infinite_prior_bounds_do_not_raise_overflowerror():
    """rng.uniform(-inf, inf) raises OverflowError; NormalPrior's defaults
    are exactly (-inf, inf), so this was reachable from the public API."""
    from puremacro.dsge.estimate import _find_finite_start

    priors = {"m": {"dist": "normal", "mean": 0.0, "std": 1.0,
                    "lb": -np.inf, "ub": np.inf}}
    rng = np.random.default_rng(0)
    vec = _find_finite_start(lambda v: float(v[0] ** 2), rng, priors)
    assert np.isfinite(vec).all()


def test_find_finite_start_reports_the_underlying_failure():
    from puremacro.dsge.estimate import _find_finite_start, _LikelihoodFailure

    priors = {"m": {"dist": "normal", "mean": 0.0, "std": 1.0,
                    "lb": -2.0, "ub": 2.0}}
    failure = _LikelihoodFailure()
    failure.record(np.linalg.LinAlgError("F is not positive definite"), np.zeros(1))
    with pytest.raises(ValueError, match="F is not positive definite"):
        _find_finite_start(lambda v: np.inf, np.random.default_rng(0), priors,
                           max_tries=5, failure=failure)


def test_stochastically_singular_model_is_named_as_such():
    """Three observables driven by one shock and no measurement error.

    The old message was 'log_posterior_fn(init) is not finite; pick a
    better starting point' -- a statement about starting values for what is
    a structural property of the model.
    """
    from puremacro.dsge.estimate import estimate_dsge

    rng = np.random.default_rng(0)
    z = rng.standard_normal(60)
    data = pd.DataFrame({"a": z, "b": 2 * z, "c": 3 * z})

    def singular_obs(params):
        return StateSpaceModel(
            T=np.array([[params["rho"]]]),
            Z=np.array([[1.0], [2.0], [3.0]]),
            R=np.array([[1.0]]),
            Q=np.array([[0.25]]),
            H=np.zeros((3, 3)),
        )

    priors = {"rho": {"dist": "beta", "mean": 0.5, "std": 0.2,
                      "lb": 0.01, "ub": 0.99}}
    with pytest.raises(ValueError, match="stochastically singular"):
        estimate_dsge(
            data, observation_eq=singular_obs, priors=priors,
            observed_vars=["a", "b", "c"], initial_params={"rho": 0.5},
            n_chains=1, n_draws=10, burn_in=5, seed=0,
        )


def _cliff_state_space(params: dict) -> StateSpaceModel:
    """Feasible only for rho <= 0.5, mimicking a DSGE that fails to solve."""
    if params["rho"] > 0.5:
        raise np.linalg.LinAlgError("simulated non-PD innovation covariance")
    return _ar1_state_space(params)


def test_mode_search_that_never_moves_is_reported_as_such():
    """The worst failure in this module: L-BFGS-B stops at iteration 1 with
    success=True whenever the objective is +inf a finite-difference step
    away, and `initial_params` is handed straight back as `mode` with
    converged_mle=True. On SW07 that left ~2330 log-posterior points on the
    table without a word to the user. It must at minimum say so.
    """
    from puremacro.dsge.estimate import estimate_dsge

    data = _simulate_ar1(0.7, 0.5, T=200, seed=0)
    with pytest.warns(UserWarning, match="without moving"):
        res = estimate_dsge(
            data, observation_eq=_cliff_state_space, priors=_AR1_PRIORS,
            observed_vars=["y"], initial_params={"rho": 0.5, "sigma": 0.4},
            n_chains=1, n_draws=50, burn_in=20, seed=0,
        )
    assert res.mode["rho"] == pytest.approx(0.5)


def test_optimiser_objective_is_finite_on_an_infeasible_draw():
    """The MCMC target must keep +inf (reject the draw); the optimiser's
    copy must not, or its finite-difference gradient is undefined."""
    from puremacro.dsge.estimate import _make_neg_log_posterior, _OPT_PENALTY

    y = _simulate_ar1(0.7, 0.5, T=50, seed=0)[["y"]].to_numpy()
    bad = np.array([0.9, 0.4])
    mcmc_target = _make_neg_log_posterior(
        y, _cliff_state_space, _AR1_PRIORS, ("rho", "sigma"), {},
    )
    opt_target = _make_neg_log_posterior(
        y, _cliff_state_space, _AR1_PRIORS, ("rho", "sigma"), {},
        penalty=_OPT_PENALTY,
    )
    assert mcmc_target(bad) == np.inf
    assert np.isfinite(opt_target(bad))
    assert opt_target(bad) >= 1e9


def test_mode_search_keeps_the_best_point_when_it_hits_the_iteration_cap(monkeypatch):
    """`success=False` used to discard the optimiser's answer entirely.

    On Smets-Wouters (2007) the bounded L-BFGS-B run stops with
    'STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT' after taking the
    negative log-posterior from 3431.53 down to 1086.79 (max |dx| = 1.86).
    The old code threw that away and reported `initial_params` as the mode,
    which is where the ~2330 lost log-posterior points came from.
    """
    import puremacro.dsge.mode as mode_mod
    from puremacro.dsge.estimate import estimate_dsge

    # The scipy call moved behind ``mode.find_mode`` in 2.6.0 when
    # ``mode_compute`` gained a menu. The seam this test injects at moved with
    # it; what is under test — a capped optimiser reporting success=False, and
    # its best point being kept with a warning — is unchanged.
    real_minimize = mode_mod._scipy_minimize
    better = {}

    def capped_minimize(fun, x0, **kwargs):
        kwargs.setdefault("options", {})
        kwargs["options"] = {**kwargs["options"], "maxiter": 2}
        out = real_minimize(fun, x0, **kwargs)
        # Force the "ran out of iterations" report regardless of scipy's
        # own verdict, which is the case under test.
        out.success = False
        out.message = "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
        better["x"] = np.asarray(out.x, dtype=float)
        better["improved"] = float(fun(x0)) - float(out.fun)
        return out

    monkeypatch.setattr(mode_mod, "_scipy_minimize", capped_minimize)
    data = _simulate_ar1(0.7, 0.5, T=200, seed=0)
    init = {"rho": 0.05, "sigma": 1.5}
    with pytest.warns(UserWarning, match="stopped before converging"):
        res = estimate_dsge(
            data, observation_eq=_ar1_state_space, priors=_AR1_PRIORS,
            observed_vars=["y"], initial_params=init,
            n_chains=1, n_draws=50, burn_in=20, seed=0,
        )
    assert better["improved"] > 0.0, "the capped run did not improve at all"
    assert res.mode["rho"] == pytest.approx(float(better["x"][0]), rel=1e-12)
    assert res.mode["sigma"] == pytest.approx(float(better["x"][1]), rel=1e-12)
    assert abs(res.mode["rho"] - init["rho"]) > 1e-6


def test_estimate_dsge_rejects_an_infeasible_prior_by_name():
    from puremacro.dsge.estimate import estimate_dsge

    bad = {"rho": {"dist": "beta", "mean": 0.5, "std": 0.6,
                   "lb": 0.001, "ub": 0.99}}
    data = _simulate_ar1(0.7, 0.5, T=50, seed=0)
    with pytest.raises(ValueError, match=r"estimate_dsge: prior 'rho'"):
        estimate_dsge(
            data, observation_eq=_ar1_state_space, priors=bad,
            observed_vars=["y"], initial_params={"rho": 0.5},
            n_chains=1, n_draws=10, burn_in=5, seed=0,
        )
