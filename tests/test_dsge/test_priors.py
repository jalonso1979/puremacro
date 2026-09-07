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


def test_logpdf_invgamma_integrates_to_one_with_requested_moments():
    """The invgamma prior must BE the distribution its spec advertises.

    ``{"dist": "invgamma", "mean": m, "std": s}`` is Dynare's
    ``INV_GAMMA_PDF, m, s``: the density must integrate to 1 and have mean
    ``m`` and standard deviation ``s``.  The pre-2.4.1 implementation read
    ``(mean, std)`` as Dynare's internal ``(s, nu)`` and evaluated
    ``invgamma(a=nu/2, scale=s**2*nu/2)`` on ``x`` rather than on ``x**2``;
    for ``(0.1, 2.0)`` that density had mode 0.005 against Dynare's 0.0461
    and a divergent mean.  This check is deliberately independent of the
    implementation: quadrature, not a restatement of the same call.
    """
    from scipy import integrate
    from puremacro.dsge.priors import _logpdf_invgamma

    for mean, std in [(0.1, 2.0), (0.3, 2.0), (0.1, 0.05), (1.0, 0.5)]:
        f = lambda x: math.exp(_logpdf_invgamma(x, mean, std))  # noqa: E731
        mass = (integrate.quad(f, 1e-10, 1.0, limit=500)[0]
                + integrate.quad(f, 1.0, np.inf, limit=500)[0])
        m1 = (integrate.quad(lambda x: x * f(x), 1e-10, 1.0, limit=500)[0]
              + integrate.quad(lambda x: x * f(x), 1.0, np.inf, limit=500)[0])
        m2 = (integrate.quad(lambda x: x * x * f(x), 1e-10, 1.0, limit=500)[0]
              + integrate.quad(lambda x: x * x * f(x), 1.0, np.inf, limit=500)[0])
        assert mass == pytest.approx(1.0, abs=1e-6), (mean, std, mass)
        assert m1 == pytest.approx(mean, rel=1e-5), (mean, std, m1)
        assert math.sqrt(m2 - m1 ** 2) == pytest.approx(std, rel=1e-4)


def test_logpdf_invgamma_is_dynare_lpdfig1():
    """The family is Dynare's type-1: x**2 ~ InvGamma(nu/2, s/2)."""
    from puremacro.dsge.priors import (
        _invgamma_s_nu, _logpdf_invgamma, _logpdf_invgamma_s_nu,
    )

    s, nu = _invgamma_s_nu(0.1, 2.0)
    # Dynare's inverse_gamma_specification for (mean=0.1, std=2).
    assert s == pytest.approx(0.00638024, rel=1e-5)
    assert nu == pytest.approx(2.00159108, rel=1e-7)
    for x in (0.02, 0.05, 0.1, 0.4618, 1.0, 2.0):
        # change of variables from scipy on u = x**2
        expected = float(
            stats.invgamma.logpdf(x ** 2, a=nu / 2.0, scale=s / 2.0)
            + math.log(2.0 * x)
        )
        assert _logpdf_invgamma_s_nu(x, s, nu) == pytest.approx(expected, rel=1e-12)
        assert _logpdf_invgamma(x, 0.1, 2.0) == pytest.approx(expected, rel=1e-12)


def test_invgamma_spec_accepts_explicit_s_nu():
    """An explicit {'s': .., 'nu': ..} bypasses the mean/std solve."""
    from puremacro.dsge.priors import _logpdf_for_spec, _logpdf_invgamma_s_nu

    spec = {"dist": "invgamma", "mean": 0.1, "std": 2.0,
            "s": 0.5, "nu": 3.0, "lb": 0.0, "ub": 10.0}
    assert _logpdf_for_spec(spec, 0.7) == pytest.approx(
        _logpdf_invgamma_s_nu(0.7, 0.5, 3.0), rel=1e-14
    )


def test_infeasible_beta_spec_raises_instead_of_nan():
    """std**2 >= mean*(1-mean) has no Beta; scipy silently returns NaN."""
    from puremacro.dsge.priors import _logpdf_beta, log_prior

    with pytest.raises(ValueError, match="no std"):
        _logpdf_beta(0.5, 0.5, 0.6)
    bad = {"p": {"dist": "beta", "mean": 0.5, "std": 0.6, "lb": 0.0, "ub": 1.0}}
    with pytest.raises(ValueError, match="no std"):
        log_prior({"p": 0.5}, bad)


def test_validate_priors_names_the_offender():
    from puremacro.dsge.priors import _validate_priors

    with pytest.raises(ValueError, match="myparam"):
        _validate_priors(
            {"myparam": {"dist": "beta", "mean": 0.5, "std": 0.6,
                         "lb": 0.0, "ub": 1.0}},
            caller="estimate_dsge",
        )
    with pytest.raises(ValueError, match="support is empty"):
        _validate_priors(
            {"q": {"dist": "normal", "mean": 0.0, "std": 1.0,
                   "lb": 1.0, "ub": 1.0}},
        )
    with pytest.raises(ValueError, match="mean > 0"):
        _validate_priors(
            {"sig": {"dist": "invgamma", "mean": 0.0, "std": 1.0,
                     "lb": 0.0, "ub": 5.0}},
        )


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
    from puremacro.dsge.priors import (
        log_prior, _logpdf_beta, _logpdf_normal, _logpdf_invgamma, _logpdf_gamma,
    )
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
