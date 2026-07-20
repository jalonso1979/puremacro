"""Tests for the model-agnostic prior framework."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats


# Tiny toy priors dict for testing.
_TOY_PRIORS = {
    "rho":   {"dist": "beta",     "mean": 0.5, "std": 0.2, "lb": 0.001, "ub": 0.99},
    "mu":    {"dist": "normal",   "mean": 0.0, "std": 1.0, "lb": -5.0,  "ub": 5.0},
    "sigma": {"dist": "invgamma", "mean": 0.1, "std": 2.0, "lb": 0.01,  "ub": 5.0},
    "alpha": {"dist": "gamma",    "mean": 1.0, "std": 0.5, "lb": 0.001, "ub": 10.0},
}


def test_logpdf_beta_matches_scipy():
    from puremacro.dsge.priors import _logpdf_beta
    mean, std = 0.5, 0.2
    a = mean * (mean * (1 - mean) / std**2 - 1)
    b = a * (1 - mean) / mean
    x = 0.4
    expected = float(stats.beta.logpdf(x, a, b))
    assert _logpdf_beta(x, mean, std) == pytest.approx(expected, rel=1e-12)


def test_logpdf_invgamma_dynare_convention():
    from puremacro.dsge.priors import _logpdf_invgamma
    s, nu = 0.1, 2.0
    x = 0.15
    expected = float(stats.invgamma.logpdf(x, a=nu/2, scale=s**2 * nu / 2))
    assert _logpdf_invgamma(x, s, nu) == pytest.approx(expected, rel=1e-12)


def test_logpdf_normal_matches_scipy():
    from puremacro.dsge.priors import _logpdf_normal
    expected = float(stats.norm.logpdf(0.3, loc=0.0, scale=1.0))
    assert _logpdf_normal(0.3, 0.0, 1.0) == pytest.approx(expected, rel=1e-12)


def test_logpdf_gamma_matches_scipy():
    from puremacro.dsge.priors import _logpdf_gamma
    mean, std = 1.0, 0.5
    k = (mean / std) ** 2
    theta = std**2 / mean
    expected = float(stats.gamma.logpdf(0.8, a=k, scale=theta))
    assert _logpdf_gamma(0.8, mean, std) == pytest.approx(expected, rel=1e-12)


def test_log_prior_sums_across_params():
    from puremacro.dsge.priors import log_prior, _logpdf_beta, _logpdf_normal, _logpdf_invgamma, _logpdf_gamma
    params = {"rho": 0.5, "mu": 0.0, "sigma": 0.1, "alpha": 1.0}
    expected = (
        _logpdf_beta(0.5, 0.5, 0.2)
        + _logpdf_normal(0.0, 0.0, 1.0)
        + _logpdf_invgamma(0.1, 0.1, 2.0)
        + _logpdf_gamma(1.0, 1.0, 0.5)
    )
    assert log_prior(params, _TOY_PRIORS) == pytest.approx(expected, rel=1e-12)


def test_log_prior_returns_neg_inf_outside_bounds():
    from puremacro.dsge.priors import log_prior
    params = {"rho": 1.5, "mu": 0.0, "sigma": 0.1, "alpha": 1.0}  # rho > ub
    assert log_prior(params, _TOY_PRIORS) == -math.inf


def test_log_prior_raises_on_unknown_dist():
    from puremacro.dsge.priors import log_prior
    weird = {"x": {"dist": "weibull", "mean": 1.0, "std": 1.0, "lb": 0.0, "ub": 10.0}}
    with pytest.raises(ValueError, match="unknown distribution"):
        log_prior({"x": 1.0}, weird)


def test_param_bounds_returns_list_of_tuples_in_order():
    from puremacro.dsge.priors import param_bounds
    bounds = param_bounds(_TOY_PRIORS)
    assert bounds == [(0.001, 0.99), (-5.0, 5.0), (0.01, 5.0), (0.001, 10.0)]


def test_param_names_preserves_dict_order():
    from puremacro.dsge.priors import param_names
    assert param_names(_TOY_PRIORS) == ("rho", "mu", "sigma", "alpha")
