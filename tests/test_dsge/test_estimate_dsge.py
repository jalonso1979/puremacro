"""Tests for puremacro.dsge.estimate_dsge — the generic Bayesian DSGE engine."""
from __future__ import annotations

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
