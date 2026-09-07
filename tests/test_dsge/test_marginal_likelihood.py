"""Marginal likelihood and model comparison.

The reference throughout is a conjugate Gaussian model — ``y_i ~ N(theta, s^2)``
iid with prior ``theta ~ N(m0, t0^2)`` — where ``log p(y)`` is available in
closed form and the posterior is Gaussian, so the Laplace approximation is not
an approximation at all but the answer.

The tolerances below were measured, not guessed: Laplace lands within 1.4e-14,
and the modified harmonic mean at 2000 draws is within 0.006 with a spread of
at most 0.14 across truncation levels.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from puremacro.dsge._results import DSGEPosteriorResult
from puremacro.dsge.marginal import (
    harmonic_mean_mdd,
    laplace_mdd,
    model_comparison,
)


_N, _S, _M0, _T0 = 50, 1.3, 0.2, 0.8


def _conjugate():
    """``(y, exact log p(y), posterior mean, posterior sd, log_post fn)``."""
    y = np.random.default_rng(0).normal(1.0, _S, size=_N)
    Sigma = _S ** 2 * np.eye(_N) + _T0 ** 2 * np.ones((_N, _N))
    exact = float(multivariate_normal(mean=np.full(_N, _M0), cov=Sigma).logpdf(y))
    prec = _N / _S ** 2 + 1 / _T0 ** 2
    post_mean = (y.sum() / _S ** 2 + _M0 / _T0 ** 2) / prec

    def log_post(th):
        return float(norm.logpdf(y, loc=th, scale=_S).sum()
                     + norm.logpdf(th, loc=_M0, scale=_T0))

    return y, exact, post_mean, 1.0 / math.sqrt(prec), log_post, prec


def test_laplace_is_exact_for_a_gaussian_posterior():
    """The posterior here IS Gaussian, so Laplace is not an approximation."""
    _, exact, post_mean, _, log_post, prec = _conjugate()
    got = laplace_mdd(log_post(post_mean), np.array([[1.0 / prec]]))
    assert got == pytest.approx(exact, rel=1e-12)


def test_laplace_rejects_a_non_positive_definite_hessian_inv():
    """A non-PD inverse Hessian means the point is not a maximum, so the
    approximation around it has no meaning."""
    with pytest.raises(ValueError, match="positive definite"):
        laplace_mdd(-10.0, np.array([[1.0, 0.0], [0.0, -0.5]]))


def test_laplace_rejects_a_non_finite_mode_value():
    with pytest.raises(ValueError, match="finite"):
        laplace_mdd(-np.inf, np.eye(2))


def test_harmonic_mean_recovers_the_analytic_value():
    _, exact, post_mean, post_sd, log_post, _ = _conjugate()
    draws = np.random.default_rng(2).normal(post_mean, post_sd, size=(2000, 1))
    lp = np.array([log_post(t[0]) for t in draws])
    res = harmonic_mean_mdd(draws, lp)
    assert res.estimate == pytest.approx(exact, abs=0.05)
    assert res.converged is True
    assert res.spread < 0.2
    assert res.n_draws == 2000


@pytest.mark.parametrize("seed", [0, 1])
def test_harmonic_mean_flags_itself_when_the_draws_are_few_for_the_dimension(seed):
    """Measured: 30 draws in 5 dimensions swing by 1.1-1.5 log points across
    truncation levels. The estimator is supposed to be invariant to the
    truncation, so it must say it has not converged rather than return a
    number that looks like an answer."""
    d, m = 5, 30
    draws = np.random.default_rng(seed).standard_normal((m, d))
    lp = multivariate_normal(np.zeros(d), np.eye(d)).logpdf(draws)
    with pytest.warns(UserWarning, match="not converged"):
        res = harmonic_mean_mdd(draws, lp)
    assert res.converged is False
    assert res.spread > 1.0
    assert "NOT converged" in res.summary()


def test_harmonic_mean_refuses_a_degenerate_chain():
    """A chain that never moved has a singular covariance, so Geweke's
    weighting function is undefined. That is the shape of a stuck sampler."""
    with pytest.raises(ValueError, match="singular"):
        harmonic_mean_mdd(np.full((100, 2), 0.3), np.zeros(100))


def test_harmonic_mean_accepts_the_chains_by_draws_by_params_layout():
    """DSGEPosteriorResult.draws is 3-D; it must not need reshaping by hand."""
    _, exact, post_mean, post_sd, log_post, _ = _conjugate()
    rng = np.random.default_rng(4)
    draws = rng.normal(post_mean, post_sd, size=(2, 1000, 1))
    lp = np.array([[log_post(t[0]) for t in chain] for chain in draws])
    res = harmonic_mean_mdd(draws, lp)
    assert res.n_draws == 2000
    assert res.estimate == pytest.approx(exact, abs=0.05)


def test_harmonic_mean_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="disagree"):
        harmonic_mean_mdd(np.random.default_rng(0).standard_normal((50, 2)),
                          np.zeros(40))


def test_harmonic_mean_presentation_contract():
    _, _, post_mean, post_sd, log_post, _ = _conjugate()
    draws = np.random.default_rng(6).normal(post_mean, post_sd, size=(500, 1))
    lp = np.array([log_post(t[0]) for t in draws])
    res = harmonic_mean_mdd(draws, lp)
    assert isinstance(res.summary(), str) and res.summary()
    for fn in (res.to_markdown, res.to_latex, res.to_typst):
        assert isinstance(fn(), str) and fn()
    import matplotlib
    matplotlib.use("Agg")
    assert res.plot() is not None


# --------------------------------------------------------------------------
# model_comparison
# --------------------------------------------------------------------------
def _fake_result(name: str, log_post_mode: float | None, k: int = 1):
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    return DSGEPosteriorResult(
        draws=rng.standard_normal((1, 200, k)),
        param_names=tuple(f"p{i}" for i in range(k)),
        log_posterior_trace=rng.standard_normal((1, 200)),
        accept_rates=(0.3,),
        mode={f"p{i}": 0.0 for i in range(k)},
        mode_hessian_inv=np.eye(k),
        n_burn_in=50,
        data_n_obs=100,
        seed=0,
        model_name=name,
        log_post_mode=log_post_mode,
    )


def test_log_mdd_accessor_matches_the_free_function():
    res = _fake_result("A", -120.0)
    assert res.log_mdd("laplace") == pytest.approx(
        laplace_mdd(-120.0, np.eye(1)), rel=1e-14)


def test_log_mdd_says_so_when_log_post_mode_is_missing():
    with pytest.raises(ValueError, match="predates"):
        _fake_result("old", None).log_mdd("laplace")


def test_log_mdd_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="laplace"):
        _fake_result("A", -120.0).log_mdd("bridge")


def test_model_comparison_posterior_probabilities_sum_to_one():
    table = model_comparison([_fake_result("A", -120.0), _fake_result("B", -122.0)])
    assert list(table.index) == ["A", "B"]        # sorted by probability
    assert table["posterior_prob"].sum() == pytest.approx(1.0, rel=1e-12)
    assert table.loc["A", "posterior_prob"] > table.loc["B", "posterior_prob"]
    assert table.loc["A", "log_bayes_factor_vs_best"] == pytest.approx(0.0)
    assert table.loc["B", "log_bayes_factor_vs_best"] == pytest.approx(-2.0)


def test_model_comparison_refuses_to_mix_methods():
    """A Laplace value and a harmonic-mean value are not on the same footing.
    A model that cannot supply the requested estimator must raise, not fall
    back to the other one."""
    with pytest.raises(ValueError, match="same estimator"):
        model_comparison({"A": _fake_result("A", -120.0),
                          "B": _fake_result("B", None)}, method="laplace")


def test_model_comparison_honours_model_priors():
    equal = model_comparison({"A": _fake_result("A", -120.0),
                              "B": _fake_result("B", -120.0)})
    assert equal["posterior_prob"].to_numpy() == pytest.approx([0.5, 0.5])
    tilted = model_comparison({"A": _fake_result("A", -120.0),
                               "B": _fake_result("B", -120.0)},
                              model_priors={"A": 3.0, "B": 1.0})
    assert tilted.loc["A", "posterior_prob"] == pytest.approx(0.75)


def test_model_comparison_needs_at_least_two_models():
    with pytest.raises(ValueError, match="at least two"):
        model_comparison([_fake_result("A", -120.0)])


def test_model_comparison_rejects_duplicate_model_names():
    with pytest.raises(ValueError, match="not unique"):
        model_comparison([_fake_result("A", -120.0), _fake_result("A", -121.0)])


def test_model_comparison_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="laplace"):
        model_comparison([_fake_result("A", -1.0), _fake_result("B", -2.0)],
                         method="bridge")


def test_estimate_dsge_records_the_log_posterior_at_the_mode():
    """The Laplace estimator needs it, and nothing else records it."""
    import pandas as pd
    from puremacro.dsge import estimate_dsge
    from puremacro.state_space import StateSpaceModel

    def ar1_ss(params):
        return StateSpaceModel(
            T=np.array([[params["rho"]]]), Z=np.array([[1.0]]),
            R=np.array([[1.0]]), Q=np.array([[params["sigma"] ** 2]]),
            H=np.array([[1e-8]]), c=np.zeros(1), d=np.zeros(1),
        )

    rng = np.random.default_rng(0)
    eps = rng.standard_normal(200) * 0.5
    x = np.zeros(200)
    for t in range(1, 200):
        x[t] = 0.7 * x[t - 1] + eps[t]
    data = pd.DataFrame({"y": x})

    priors = {
        "rho": {"dist": "beta", "mean": 0.5, "std": 0.2, "lb": 0.001, "ub": 0.99},
        "sigma": {"dist": "invgamma", "mean": 0.1, "std": 2.0, "lb": 0.01, "ub": 5.0},
    }
    res = estimate_dsge(
        data, observation_eq=ar1_ss, priors=priors, observed_vars=["y"],
        initial_params={"rho": 0.5, "sigma": 0.4},
        n_draws=60, n_chains=1, burn_in=120, seed=0,
    )
    assert res.log_post_mode is not None and np.isfinite(res.log_post_mode)
    # It must be the value at the reported mode, not at the starting point.
    assert res.log_post_mode == pytest.approx(res.log_posterior_trace.max(), abs=15.0)
    assert np.isfinite(res.log_mdd("laplace"))
