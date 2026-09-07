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
    # "weibull" was the example here until 2.6.0 added it as a real family;
    # the test needs a name no family will plausibly claim.
    weird = {"x": {"dist": "dirichlet", "mean": 1.0, "std": 1.0, "lb": 0.0, "ub": 10.0}}
    with pytest.raises(ValueError, match="unknown distribution"):
        log_prior({"x": 1.0}, weird)


def test_param_bounds_returns_list_of_tuples_in_order():
    from puremacro.dsge.priors import param_bounds
    bounds = param_bounds(_TOY_PRIORS)
    assert bounds == [(0.001, 0.99), (-5.0, 5.0), (0.01, 5.0), (0.001, 10.0)]


def test_param_names_preserves_dict_order():
    from puremacro.dsge.priors import param_names
    assert param_names(_TOY_PRIORS) == ("rho", "mu", "sigma", "alpha")


# ---------------------------------------------------------------------------
# 2.6.0 — the shapes an `estimated_params` block can name that had no class.
#
# Conventions pinned here follow Dynare's *_specification helpers, not a
# reading of the manual's prose. In every case PRIOR_P1 / PRIOR_P2 are the
# mean and standard deviation of the ACTUAL variable, and PRIOR_P3 / PRIOR_P4
# shift and scale its support. Getting this backwards silently re-specifies
# every prior in a block, so each test states the mapping it asserts.
# ---------------------------------------------------------------------------
def test_generalised_beta_matches_scipy_on_a_shifted_support():
    """``beta_pdf, P1, P2, P3, P4``: P1/P2 describe x on [P3, P4].

    Dynare's beta_specification standardises with m01 = (P1-P3)/(P4-P3) and
    s01 = P2/(P4-P3), so the density carries a 1/(P4-P3) Jacobian.
    """
    from puremacro.dsge.priors import BetaPrior

    p = BetaPrior(mean=0.5, std=0.1, shift=0.2, scale=0.6)
    m01, s01 = (0.5 - 0.2) / 0.6, 0.1 / 0.6
    a = m01 * (m01 * (1 - m01) / s01**2 - 1)
    b = a * (1 - m01) / m01
    x = 0.45
    expected = float(stats.beta.logpdf((x - 0.2) / 0.6, a, b)) - math.log(0.6)
    assert p.logpdf(x) == pytest.approx(expected, rel=1e-12)


def test_plain_beta_is_unchanged_by_the_shift_scale_fields():
    """Back-compat: the default (shift=0, scale=1) reproduces 2.5.0 exactly."""
    from puremacro.dsge.priors import BetaPrior, _logpdf_beta

    p = BetaPrior(mean=0.5, std=0.2, lb=0.001, ub=0.99)
    assert p.logpdf(0.4) == pytest.approx(_logpdf_beta(0.4, 0.5, 0.2), rel=1e-14)
    assert p.to_dict()["shift"] == 0.0 and p.to_dict()["scale"] == 1.0


def test_generalised_beta_defaults_its_bounds_to_the_support():
    from puremacro.dsge.priors import BetaPrior

    p = BetaPrior(mean=0.5, std=0.1, shift=0.2, scale=0.6)
    assert (p.lb, p.ub) == (0.2, 0.8)
    assert p.logpdf(0.1) == -math.inf and p.logpdf(0.9) == -math.inf


def test_shifted_gamma_matches_scipy():
    """``gamma_pdf, P1, P2, P3``: P1 is the mean of x, so x - P3 has mean
    P1 - P3 and standard deviation P2 (Dynare's gamma_specification)."""
    from puremacro.dsge.priors import GammaPrior

    mean, std, shift = 1.0, 0.5, 0.25
    p = GammaPrior(mean=mean, std=std, shift=shift)
    k = ((mean - shift) / std) ** 2
    theta = std**2 / (mean - shift)
    x = 1.4
    expected = float(stats.gamma.logpdf(x - shift, a=k, scale=theta))
    assert p.logpdf(x) == pytest.approx(expected, rel=1e-12)


def test_shifted_gamma_has_the_advertised_mean():
    """Quadrature, not a restatement of the same call."""
    from scipy import integrate
    from puremacro.dsge.priors import GammaPrior

    p = GammaPrior(mean=1.0, std=0.5, shift=0.25)
    f = lambda x: math.exp(p.logpdf(x))  # noqa: E731
    mass = integrate.quad(f, 0.25, np.inf, limit=500)[0]
    m1 = integrate.quad(lambda x: x * f(x), 0.25, np.inf, limit=500)[0]
    assert mass == pytest.approx(1.0, abs=1e-8)
    assert m1 == pytest.approx(1.0, rel=1e-6)


def test_plain_gamma_is_unchanged_by_the_shift_field():
    from puremacro.dsge.priors import GammaPrior, _logpdf_gamma

    p = GammaPrior(mean=1.0, std=0.5)
    assert p.logpdf(0.8) == pytest.approx(_logpdf_gamma(0.8, 1.0, 0.5), rel=1e-14)


def test_weibull_from_shape_and_scale_matches_scipy():
    from puremacro.dsge.priors import WeibullPrior

    p = WeibullPrior(shape=2.0, scale=1.5)
    assert p.logpdf(1.1) == pytest.approx(
        float(stats.weibull_min.logpdf(1.1, c=2.0, scale=1.5)), rel=1e-12)


def test_weibull_from_mean_and_std_recovers_those_moments():
    """``weibull_pdf, P1, P2`` is mean/std like every other family here."""
    from scipy.special import gamma as gamma_fn
    from puremacro.dsge.priors import WeibullPrior

    for mean, std in [(1.0, 0.5), (2.0, 0.4), (0.3, 0.3)]:
        p = WeibullPrior(mean=mean, std=std)
        k, lam = p.shape, p.scale
        got_mean = lam * gamma_fn(1.0 + 1.0 / k)
        got_var = lam**2 * (gamma_fn(1.0 + 2.0 / k) - gamma_fn(1.0 + 1.0 / k) ** 2)
        assert got_mean == pytest.approx(mean, rel=1e-9)
        assert math.sqrt(got_var) == pytest.approx(std, rel=1e-9)


def test_weibull_rejects_both_parameterisations_at_once():
    from puremacro.dsge.priors import WeibullPrior

    with pytest.raises(ValueError, match="not both"):
        WeibullPrior(mean=1.0, std=0.5, shape=2.0, scale=1.5)


def test_inv_gamma_type2_is_dynare_lpdfig2():
    """Type 2 puts the inverse gamma on x itself (a variance); type 1 puts it
    on x**2 (a standard deviation). They are different densities."""
    from puremacro.dsge.priors import InvGammaPrior

    p2 = InvGammaPrior(s=0.1, nu=4.0, kind="type2")
    x = 0.05
    assert p2.logpdf(x) == pytest.approx(
        float(stats.invgamma.logpdf(x, a=4.0 / 2.0, scale=0.1 / 2.0)), rel=1e-12)
    p1 = InvGammaPrior(s=0.1, nu=4.0)
    assert p1.logpdf(x) != pytest.approx(p2.logpdf(x), rel=1e-6)


def test_inv_gamma_type2_mean_std_round_trip():
    """nu = 2 (mean/std)**2 + 4 and s = mean (nu - 2), checked by quadrature."""
    from scipy import integrate
    from puremacro.dsge.priors import InvGammaPrior

    p = InvGammaPrior(mean=0.5, std=0.2, kind="type2")
    assert p.nu == pytest.approx(2.0 * (0.5 / 0.2) ** 2 + 4.0, rel=1e-12)
    assert p.s == pytest.approx(0.5 * (p.nu - 2.0), rel=1e-12)
    f = lambda x: math.exp(p.logpdf(x))  # noqa: E731
    m1 = integrate.quad(lambda x: x * f(x), 1e-12, np.inf, limit=500)[0]
    m2 = integrate.quad(lambda x: x * x * f(x), 1e-12, np.inf, limit=500)[0]
    assert m1 == pytest.approx(0.5, rel=1e-6)
    assert math.sqrt(m2 - m1**2) == pytest.approx(0.2, rel=1e-5)


def test_inv_gamma_type1_is_still_the_default():
    """Back-compat: no `kind` means type 1, byte-for-byte as in 2.5.0."""
    from puremacro.dsge.priors import InvGammaPrior, _logpdf_invgamma

    p = InvGammaPrior(mean=0.1, std=2.0)
    assert p.logpdf(0.4618) == pytest.approx(_logpdf_invgamma(0.4618, 0.1, 2.0), rel=1e-14)
    assert p.to_dict()["kind"] == "type1"


@pytest.mark.parametrize("spec,cls", [
    ({"dist": "beta", "mean": 0.5, "std": 0.1, "shift": 0.2, "scale": 0.6}, "BetaPrior"),
    ({"dist": "gamma", "mean": 1.0, "std": 0.5, "shift": 0.25}, "GammaPrior"),
    ({"dist": "weibull", "mean": 1.0, "std": 0.5}, "WeibullPrior"),
    ({"dist": "weibull", "shape": 2.0, "scale": 1.5}, "WeibullPrior"),
    ({"dist": "invgamma", "s": 0.1, "nu": 4.0, "kind": "type2"}, "InvGammaPrior"),
])
def test_ensure_prior_round_trips_the_new_specs(spec, cls):
    from puremacro.dsge import priors as P

    p = P.ensure_prior(spec)
    assert type(p).__name__ == cls
    assert p.to_dict()["dist"] == spec["dist"]
    # A dict spec and the class it builds must agree on the density.
    x = 0.45 if spec["dist"] == "beta" else 1.1
    assert P._logpdf_for_spec(p.to_dict(), x) == pytest.approx(p.logpdf(x), rel=1e-14)


def test_validate_priors_rejects_a_nonpositive_beta_scale():
    from puremacro.dsge.priors import _validate_priors

    with pytest.raises(ValueError, match="scale"):
        _validate_priors({"p": {"dist": "beta", "mean": 0.5, "std": 0.1,
                                "shift": 0.2, "scale": 0.0, "lb": 0.0, "ub": 1.0}})


def test_validate_priors_rejects_an_unknown_invgamma_kind():
    from puremacro.dsge.priors import _validate_priors

    with pytest.raises(ValueError, match="type3"):
        _validate_priors({"s": {"dist": "invgamma", "mean": 0.1, "std": 2.0,
                                "kind": "type3", "lb": 0.0, "ub": 5.0}})


def test_validate_priors_rejects_a_gamma_shift_above_the_mean():
    from puremacro.dsge.priors import _validate_priors

    with pytest.raises(ValueError, match="shift"):
        _validate_priors({"g": {"dist": "gamma", "mean": 1.0, "std": 0.5,
                                "shift": 1.5, "lb": 0.0, "ub": 10.0}})
