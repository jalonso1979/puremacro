"""Tests for puremacro.dsge.sw07_priors."""
import math

import numpy as np
import pytest


def test_priors_dict_has_expected_size():
    from puremacro.dsge.sw07_priors import PRIORS
    assert 30 <= len(PRIORS) <= 40


def test_log_prior_finite_at_prior_means():
    """At the prior-mean parameter vector, log_prior must be finite."""
    from puremacro.dsge.sw07_priors import PRIORS, log_prior
    means = {name: spec["mean"] for name, spec in PRIORS.items()}
    val = log_prior(means)
    assert math.isfinite(val)


def test_log_prior_minus_inf_out_of_support_beta():
    """A beta parameter set outside (0, 1) returns -inf."""
    from puremacro.dsge.sw07_priors import PRIORS, log_prior
    beta_param = next(name for name, spec in PRIORS.items() if spec["dist"] == "beta")
    means = {name: spec["mean"] for name, spec in PRIORS.items()}
    means[beta_param] = 1.5
    val = log_prior(means)
    assert val == -math.inf


def test_log_prior_minus_inf_out_of_support_invgamma():
    """A negative invgamma parameter returns -inf."""
    from puremacro.dsge.sw07_priors import PRIORS, log_prior
    invgamma_param = next(name for name, spec in PRIORS.items() if spec["dist"] == "invgamma")
    means = {name: spec["mean"] for name, spec in PRIORS.items()}
    means[invgamma_param] = -1.0
    val = log_prior(means)
    assert val == -math.inf


def test_log_prior_density_matches_scipy_beta():
    """Spot-check one beta prior against scipy.stats.beta.logpdf."""
    from scipy.stats import beta as beta_dist
    from puremacro.dsge.priors import _logpdf_beta
    from puremacro.dsge.sw07_priors import PRIORS

    name = next(n for n, spec in PRIORS.items() if spec["dist"] == "beta")
    spec = PRIORS[name]
    x = 0.4
    a = spec["mean"] * (spec["mean"] * (1 - spec["mean"]) / spec["std"]**2 - 1)
    b = a * (1 - spec["mean"]) / spec["mean"]
    expected = beta_dist.logpdf(x, a, b)
    actual = _logpdf_beta(x, spec["mean"], spec["std"])
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)


def test_log_prior_sums_components():
    """log_prior(params) = sum of individual logpdfs."""
    from puremacro.dsge.priors import _logpdf_for_spec
    from puremacro.dsge.sw07_priors import PRIORS, log_prior

    means = {name: spec["mean"] for name, spec in PRIORS.items()}
    total = log_prior(means)
    manual = sum(_logpdf_for_spec(spec, means[name]) for name, spec in PRIORS.items())
    assert math.isclose(total, manual, rel_tol=1e-12)


def test_prior_means_returns_dict():
    from puremacro.dsge.sw07_priors import PRIORS, prior_means
    out = prior_means()
    assert set(out.keys()) == set(PRIORS.keys())
    for k in out:
        assert out[k] == PRIORS[k]["mean"]


def test_param_bounds_returns_list_of_tuples():
    """param_bounds returns a list of (lb, ub) tuples in PRIORS dict order."""
    from puremacro.dsge.sw07_priors import PRIORS, param_bounds
    bounds = param_bounds()
    assert len(bounds) == len(PRIORS)
    for (lb, ub), (name, spec) in zip(bounds, PRIORS.items()):
        assert lb == spec["lb"]
        assert ub == spec["ub"]


def test_param_names_returns_tuple():
    from puremacro.dsge.sw07_priors import PRIORS, param_names
    names = param_names()
    assert isinstance(names, tuple)
    assert tuple(PRIORS.keys()) == names
