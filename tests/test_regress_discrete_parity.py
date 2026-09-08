"""Parity of ``puremacro.regress.discrete`` against statsmodels 0.14.6.

Every test that names statsmodels is guarded with
``pytest.importorskip("statsmodels")`` — statsmodels is a dev dependency
and must never be importable from the shipped module (``ARCHITECTURE.md``,
"Pyodide-compatibility contract"). The target is ``atol=1e-8`` on
``params`` / ``bse`` / ``pvalues`` / ``llf``.

One parity claim is deliberately *not* made at that tolerance, and
:func:`test_poisson_nonrobust_bse_is_the_converged_information_matrix`
pins it instead of hiding it: statsmodels' GLM reports a non-robust
covariance built from the IRLS weights of the iteration *before* the one
whose coefficients it returns, so its ``bse`` can sit up to ~2e-7 away
from the information matrix evaluated at its own point estimate. That
test measures the gap in both directions — against the default fit
(bounded, non-zero) and against ``fit(tol=1e-14)`` (exact) — so the
finding cannot quietly grow.

Four tests exist purely as positive controls, per ``CONTRIBUTING.md``
§"Making sure a test can fail":
:func:`test_separation_fixture_really_is_separated` proves the
separation fixture produces the condition it claims (statsmodels returns
a coefficient of -671 on it);
:func:`test_default_maxiter_converges_without_warning` proves the
convergence-failure test is failing for the stated reason rather than
because everything warns;
:func:`test_separation_still_fires_after_the_design_gate_was_equilibrated`
proves that equilibrating the inversion — which is what lets a badly
scaled design through — did not buy that parity by disabling the
separation guard; and
:func:`test_collapsed_weights_alone_are_not_separation` proves the other
half of that guard's conjunction is load-bearing, with a converging
wide-exposure Poisson whose weights span ``9.4e-09``.

Several tests assert on a *refusal* where statsmodels returns a number.
Each states the number being refused as a measurement, so the divergence
stays auditable rather than becoming folklore: a cluster covariance on
one cluster, for instance, comes back from statsmodels at ``2.7e-17``
with no warning, which
:func:`test_single_cluster_raises_a_named_error` pins.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from puremacro.regress.discrete import (
    ConvergenceWarning,
    DiscreteResult,
    PerfectSeparationError,
    logit,
    poisson,
)

ATOL = 1e-8


# ---------------------------------------------------------------------------
# Fixtures — seeded, separation-free designs
# ---------------------------------------------------------------------------


def _binary_design(n=300, seed=20260906):
    """A logit design with both outcomes well represented and no separation."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = pd.DataFrame({"const": 1.0, "x1": x1, "x2": x2})
    p = 1.0 / (1.0 + np.exp(-(-0.3 + 0.8 * x1 - 0.5 * x2)))
    y = pd.Series((rng.uniform(size=n) < p).astype(float), name="y")
    return y, X


def _count_design(n=200, seed=7, with_offset=False):
    """A Poisson design; ``with_offset`` returns a non-trivial log-exposure."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = pd.DataFrame({"const": 1.0, "x1": x1, "x2": x2})
    off = 0.2 * rng.normal(size=n) if with_offset else np.zeros(n)
    y = pd.Series(rng.poisson(np.exp(0.4 + 0.3 * x1 - 0.2 * x2 + off)).astype(float),
                  name="cnt")
    return y, X, off


def _separated_design():
    """``y = 1{x > 19.5}`` on a 40-point grid — complete separation."""
    x = np.arange(40.0)
    X = pd.DataFrame({"const": 1.0, "x": x})
    y = pd.Series((x > 19.5).astype(float))
    return y, X


# ---------------------------------------------------------------------------
# Logit parity
# ---------------------------------------------------------------------------


def test_logit_matches_statsmodels_logit():
    """The headline claim: ``sm.Logit(y, X).fit(disp=False)``, to 1e-8."""
    sm = pytest.importorskip("statsmodels.api")
    y, X = _binary_design()

    ref = sm.Logit(y, X).fit(disp=False)
    res = logit(y, X)

    assert np.max(np.abs(res.params.to_numpy() - ref.params.to_numpy())) < ATOL
    assert np.max(np.abs(res.bse.to_numpy() - ref.bse.to_numpy())) < ATOL
    assert np.max(np.abs(res.pvalues.to_numpy() - ref.pvalues.to_numpy())) < ATOL
    assert np.max(np.abs(res.tvalues.to_numpy() - ref.tvalues.to_numpy())) < ATOL
    assert abs(res.llf - ref.llf) < ATOL
    assert abs(res.aic - ref.aic) < ATOL
    assert abs(res.bic - ref.bic) < ATOL
    assert abs(res.prsquared - ref.prsquared) < ATOL
    assert res.nobs == int(ref.nobs)
    assert res.df_resid == ref.df_resid
    assert res.df_model == ref.df_model
    assert res.use_t is False and ref.use_t is False
    assert res.converged is True

    ci, ci_ref = res.conf_int(), ref.conf_int()
    assert list(ci.columns) == list(ci_ref.columns) == [0, 1]
    assert np.max(np.abs(ci.to_numpy() - ci_ref.to_numpy())) < ATOL
    assert np.max(np.abs(np.asarray(res.cov_params()) -
                         np.asarray(ref.cov_params()))) < ATOL


def test_logit_llnull_is_exact_where_statsmodels_is_only_close():
    """The second documented gap, and it runs the other way.

    ``DiscreteResults.llnull`` refits the intercept-only model with BFGS
    (``discrete_model.py``, ``_get_llnull``), so it lands ~1e-8 short of
    the maximum. The intercept-only logit has a closed form —
    ``n1 log(pbar) + n0 log(1-pbar)`` — which this module reaches
    exactly, being a one-parameter Newton problem run to ``tol=1e-10``.
    ``prsquared`` inherits the difference, attenuated to ~1e-11, which is
    why the headline test can still assert it at 1e-8.
    """
    sm = pytest.importorskip("statsmodels.api")
    y, X = _binary_design()
    res = logit(y, X)
    ref = sm.Logit(y, X).fit(disp=False)

    n = float(len(y))
    n1 = float(y.sum())
    pbar = n1 / n
    exact = n1 * np.log(pbar) + (n - n1) * np.log(1.0 - pbar)

    assert abs(res.llnull - exact) < 1e-12
    gap = abs(res.llnull - float(ref.llnull))
    assert 1e-12 < gap < 1e-6, gap
    assert res.llnull > float(ref.llnull)   # ours is the larger, i.e. the maximum
    assert abs(res.prsquared - float(ref.prsquared)) < ATOL


def test_logit_covariance_is_the_inverse_observed_information():
    """``(X' diag(p(1-p)) X)^-1``, stated independently of statsmodels."""
    y, X = _binary_design()
    res = logit(y, X)
    Xv, b = X.to_numpy(), res.params.to_numpy()
    p = 1.0 / (1.0 + np.exp(-(Xv @ b)))
    expected = np.linalg.inv(Xv.T @ (Xv * (p * (1.0 - p))[:, None]))
    assert np.max(np.abs(res.vcov - expected)) < 1e-12
    # ... and the p-values really are normal-tail, not t.
    z = b / np.sqrt(np.diag(expected))
    assert np.max(np.abs(res.pvalues.to_numpy() - 2.0 * stats.norm.sf(np.abs(z)))) < 1e-15


def test_logit_returns_series_indexed_by_column_name():
    """``N04:1289`` reads ``params`` / ``bse`` / ``pvalues`` by column name."""
    y, X = _binary_design()
    res = logit(y, X)
    for attr in ("params", "bse", "pvalues", "tvalues"):
        s = getattr(res, attr)
        assert isinstance(s, pd.Series)
        assert list(s.index) == ["const", "x1", "x2"]
    # The exact N04 access pattern, including the odds-ratio transform.
    assert np.isfinite(float(np.exp(res.params["x1"])))
    assert np.isfinite(float(res.pvalues["x2"]))
    assert float(int(res.nobs)) == 300.0
    lo, hi = res.conf_int().loc["x1"]
    assert lo < float(res.params["x1"]) < hi


def test_logit_with_ndarray_input_returns_ndarrays():
    """Bare ndarray in, bare ndarray out — the statsmodels rule."""
    y, X = _binary_design()
    res = logit(y.to_numpy(), X.to_numpy())
    assert isinstance(res.params, np.ndarray)
    assert isinstance(res.bse, np.ndarray)
    assert res.names is None
    assert isinstance(res.cov_params(), np.ndarray)
    assert res.conf_int().shape == (3, 2)
    named = logit(y, X)
    assert np.max(np.abs(res.params - named.params.to_numpy())) < 1e-15


def test_logit_hc_covariances_match_and_are_all_the_same_matrix():
    """statsmodels aliases HC0-HC3 for an MLE result; so do we."""
    sm = pytest.importorskip("statsmodels.api")
    y, X = _binary_design()

    ref0 = sm.Logit(y, X).fit(disp=False, cov_type="HC0")
    res0 = logit(y, X, cov_type="HC0")
    assert np.max(np.abs(res0.bse.to_numpy() - ref0.bse.to_numpy())) < ATOL
    assert np.max(np.abs(res0.pvalues.to_numpy() - ref0.pvalues.to_numpy())) < ATOL

    for kind in ("HC1", "HC2", "HC3"):
        ours = logit(y, X, cov_type=kind)
        theirs = sm.Logit(y, X).fit(disp=False, cov_type=kind)
        assert np.max(np.abs(ours.bse.to_numpy() - theirs.bse.to_numpy())) < ATOL
        # The claim that makes this an alias and not four estimators.
        assert np.max(np.abs(ours.vcov - res0.vcov)) == 0.0
    # And it is genuinely different from the non-robust covariance, so the
    # assertion above is not vacuous.
    assert np.max(np.abs(res0.vcov - logit(y, X).vcov)) > 1e-4


def test_logit_cluster_matches_statsmodels_including_df_resid_inference():
    sm = pytest.importorskip("statsmodels.api")
    y, X = _binary_design(n=240, seed=11)
    groups = np.repeat(np.arange(12), 20)

    ref = sm.Logit(y, X).fit(disp=False, cov_type="cluster",
                             cov_kwds={"groups": groups})
    res = logit(y, X, cov_type="cluster", cov_kwds={"groups": groups})

    assert np.max(np.abs(res.bse.to_numpy() - ref.bse.to_numpy())) < ATOL
    assert np.max(np.abs(res.pvalues.to_numpy() - ref.pvalues.to_numpy())) < ATOL
    assert res.n_groups == 12
    assert res.df_resid_inference == float(ref.df_resid_inference) == 11.0
    assert res.use_t is False  # cluster does not flip use_t for an MLE fit


def test_logit_use_t_switches_the_tail_and_the_interval():
    """``use_t=True`` is opt-in and uses ``df_resid_inference``."""
    y, X = _binary_design()
    res_z = logit(y, X)
    res_t = logit(y, X, use_t=True)
    assert np.max(np.abs(res_t.params.to_numpy() - res_z.params.to_numpy())) == 0.0
    t = res_t.params.to_numpy() / res_t.bse.to_numpy()
    expected = 2.0 * stats.t.sf(np.abs(t), res_t.df_resid_inference)
    assert np.max(np.abs(res_t.pvalues.to_numpy() - expected)) < 1e-15
    # t tails are fatter, so every p-value rises and every interval widens.
    assert np.all(res_t.pvalues.to_numpy() >= res_z.pvalues.to_numpy())
    width_t = np.diff(res_t.conf_int().to_numpy(), axis=1)
    width_z = np.diff(res_z.conf_int().to_numpy(), axis=1)
    assert np.all(width_t > width_z)


# ---------------------------------------------------------------------------
# Perfect separation
# ---------------------------------------------------------------------------


def test_logit_perfect_separation_raises():
    """The MLE does not exist, so the estimator refuses to invent one.

    The message is asserted, not just the type: complete separation and
    quasi-complete separation reach the same exception through different
    guards, and without pinning the wording the *complete*-separation
    check could be deleted with this test still green (the singular
    information matrix would raise the other message a few iterations
    later).
    """
    y, X = _separated_design()
    with pytest.raises(PerfectSeparationError,
                       match="complete separation at iteration"):
        logit(y, X)


def test_separation_fixture_really_is_separated():
    """Positive control: statsmodels returns garbage on the same design.

    Without this the raise above could be firing on a design that is
    merely awkward. statsmodels warns and hands back a coefficient three
    orders of magnitude too large with a standard error five orders
    larger still — which is precisely the outcome the raise prevents.
    """
    sm = pytest.importorskip("statsmodels.api")
    y, X = _separated_design()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sm.Logit(y, X).fit(disp=False)
    assert abs(float(ref.params["x"])) > 30.0
    assert float(ref.bse["x"]) > 1e3
    assert ref.mle_retvals["converged"] is False


def test_logit_constant_outcome_raises():
    """A degenerate outcome is separation's trivial case, in both directions."""
    X = pd.DataFrame({"const": 1.0, "x": np.linspace(-1.0, 1.0, 30)})
    with pytest.raises(PerfectSeparationError, match="constant"):
        logit(np.zeros(30), X)
    with pytest.raises(PerfectSeparationError, match="constant"):
        logit(np.ones(30), X)


def test_quasi_separation_raises_rather_than_diverging():
    """A dummy that is 1 only where ``y`` is 1 drives one coefficient off."""
    rng = np.random.default_rng(3)
    n = 120
    x = rng.normal(size=n)
    y = (rng.uniform(size=n) < 0.5).astype(float)
    flag = np.zeros(n)
    flag[np.flatnonzero(y == 1.0)[:20]] = 1.0   # 1 implies y == 1
    X = pd.DataFrame({"const": 1.0, "x": x, "flag": flag})
    with pytest.raises(PerfectSeparationError, match="quasi-complete separation"):
        logit(y, X)
    # Control: the same design without the separating column fits fine, so
    # the raise is attributable to `flag` and not to the sample.
    assert logit(y, X.drop(columns="flag")).converged is True


def test_a_well_posed_design_does_not_trip_the_separation_guard():
    """The guard must not fire on a merely well-fitting model."""
    rng = np.random.default_rng(19)
    n = 400
    x = rng.normal(size=n)
    # Strong signal: |eta| up to ~12, fitted probabilities out to 1e-6 —
    # close to the boundary but not on it.
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-4.0 * x))).astype(float)
    X = pd.DataFrame({"const": 1.0, "x": x})
    res = logit(y, X)
    assert res.converged is True
    assert np.min(res.mu) < 1e-4 and np.max(res.mu) > 1.0 - 1e-4


# ---------------------------------------------------------------------------
# Convergence failure
# ---------------------------------------------------------------------------


def test_logit_convergence_failure_warns_and_flags():
    y, X = _binary_design()
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        res = logit(y, X, maxiter=2)
    assert res.converged is False
    assert res.n_iter == 2
    assert np.all(np.isfinite(res.params.to_numpy()))


def test_poisson_convergence_failure_warns_and_flags():
    y, X, _ = _count_design()
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        res = poisson(y, X, maxiter=2)
    assert res.converged is False
    assert res.n_iter == 2


def test_default_maxiter_converges_without_warning():
    """Positive control for the two tests above.

    If every call warned — a stuck ``converged`` flag, say — those tests
    would pass while asserting nothing about ``maxiter``.
    """
    y, X = _binary_design()
    yc, Xc, _ = _count_design()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        assert logit(y, X).converged is True
        assert poisson(yc, Xc).converged is True


# ---------------------------------------------------------------------------
# Poisson parity
# ---------------------------------------------------------------------------


def test_poisson_matches_statsmodels_glm():
    """``sm.GLM(y, X, family=Poisson()).fit()`` — everything but the bse."""
    sm = pytest.importorskip("statsmodels.api")
    y, X, _ = _count_design()

    ref = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    res = poisson(y, X)

    assert np.max(np.abs(res.params.to_numpy() - ref.params.to_numpy())) < ATOL
    assert abs(res.llf - ref.llf) < ATOL
    assert abs(res.deviance - ref.deviance) < ATOL
    assert abs(res.null_deviance - ref.null_deviance) < ATOL
    assert abs(res.pearson_chi2 - ref.pearson_chi2) < ATOL
    assert abs(res.aic - ref.aic) < ATOL
    assert abs(res.bic_llf - ref.bic_llf) < ATOL
    assert res.nobs == int(ref.nobs)
    assert res.df_resid == float(ref.df_resid)
    assert res.df_model == float(ref.df_model)
    assert res.scale == ref.scale == 1.0
    assert res.use_t is False and ref.use_t is False
    # GLMResults.bic is still the deviance form in 0.14.6; reproduce it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        assert abs(res.bic - ref.bic) < ATOL
    assert np.max(np.abs(res.mu - np.asarray(ref.fittedvalues))) < ATOL


def test_poisson_nonrobust_bse_is_the_converged_information_matrix():
    """The one documented parity gap, pinned in both directions.

    ``sm.GLM(...).fit()`` stops when the *deviance* stops moving and then
    reports the covariance of its final weighted least-squares step,
    whose weights came from the previous iterate's mean. Its ``bse`` is
    therefore one Newton step behind its own coefficients. This module
    evaluates the information matrix at the converged coefficients.

    Seed 4 is chosen because statsmodels' deviance criterion stops it at
    four IRLS iterations there, which is when the gap is visible at all;
    on most seeds it takes five and the gap falls to ~1e-12.
    """
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(4)
    n = 200
    X = pd.DataFrame({"const": 1.0, "x1": rng.normal(size=n),
                      "x2": rng.normal(size=n)})
    y = pd.Series(rng.poisson(np.exp(0.4 + 0.3 * X["x1"] - 0.2 * X["x2"])).astype(float))

    res = poisson(y, X)
    default = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    tight = sm.GLM(y, X, family=sm.families.Poisson()).fit(tol=1e-14)

    # Coefficients agree everywhere, at every tolerance.
    assert np.max(np.abs(res.params.to_numpy() - default.params.to_numpy())) < ATOL
    assert np.max(np.abs(res.params.to_numpy() - tight.params.to_numpy())) < ATOL

    # Against a statsmodels fit driven to the same fixed point: exact.
    gap_tight = float(np.max(np.abs(res.bse.to_numpy() - tight.bse.to_numpy())))
    assert gap_tight < 1e-12, gap_tight

    # Against the default fit: non-zero, and bounded. Both halves matter —
    # if the gap ever vanished the module would have started reproducing
    # statsmodels' stale weights, and if it grew past 1e-5 something else
    # broke.
    gap_default = float(np.max(np.abs(res.bse.to_numpy() - default.bse.to_numpy())))
    assert 1e-9 < gap_default < 1e-5, gap_default
    # The difference is statsmodels-internal, not ours: its own default and
    # tight fits disagree by the same amount.
    assert abs(float(np.max(np.abs(default.bse.to_numpy() - tight.bse.to_numpy())))
               - gap_default) < 1e-12


def test_poisson_with_offset_matches_statsmodels_glm():
    sm = pytest.importorskip("statsmodels.api")
    y, X, off = _count_design(with_offset=True)

    ref = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit(tol=1e-14)
    res = poisson(y, X, offset=off)

    assert np.max(np.abs(res.params.to_numpy() - ref.params.to_numpy())) < ATOL
    assert np.max(np.abs(res.bse.to_numpy() - ref.bse.to_numpy())) < ATOL
    assert np.max(np.abs(res.pvalues.to_numpy() - ref.pvalues.to_numpy())) < ATOL
    assert abs(res.llf - ref.llf) < ATOL
    assert abs(res.deviance - ref.deviance) < ATOL
    # The offset is load-bearing: dropping it moves the answer.
    no_off = poisson(y, X)
    assert np.max(np.abs(no_off.params.to_numpy() - res.params.to_numpy())) > 1e-3


def test_poisson_exposure_equals_log_offset():
    sm = pytest.importorskip("statsmodels.api")
    y, X, off = _count_design(with_offset=True)
    expo = np.exp(off)

    ours_exposure = poisson(y, X, exposure=expo)
    ours_offset = poisson(y, X, offset=off)
    assert np.max(np.abs(ours_exposure.params.to_numpy()
                         - ours_offset.params.to_numpy())) < 1e-14

    ref = sm.GLM(y, X, family=sm.families.Poisson(), exposure=expo).fit(tol=1e-14)
    assert np.max(np.abs(ours_exposure.params.to_numpy() - ref.params.to_numpy())) < ATOL


def test_poisson_hac_matches_statsmodels_glm():
    """The N19 call: ``.fit(cov_type='HAC', cov_kwds={'maxlags': L})``."""
    sm = pytest.importorskip("statsmodels.api")
    y, X, _ = _count_design()

    for maxlags in (0, 4, 8):
        ref = sm.GLM(y, X, family=sm.families.Poisson()).fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags})
        res = poisson(y, X, cov_type="HAC", cov_kwds={"maxlags": maxlags})
        assert np.max(np.abs(res.params.to_numpy() - ref.params.to_numpy())) < ATOL
        assert np.max(np.abs(res.bse.to_numpy() - ref.bse.to_numpy())) < ATOL, maxlags
        assert np.max(np.abs(res.pvalues.to_numpy() - ref.pvalues.to_numpy())) < ATOL
        assert np.max(np.abs(res.conf_int().to_numpy()
                             - ref.conf_int().to_numpy())) < ATOL
        assert res.cov_kwds["use_correction"] is False

    # HAC is not the non-robust covariance, so the assertions above bite.
    assert np.max(np.abs(poisson(y, X, cov_type="HAC",
                                 cov_kwds={"maxlags": 8}).bse.to_numpy()
                         - poisson(y, X).bse.to_numpy())) > 1e-4


def test_poisson_hac_with_offset_matches_statsmodels_glm():
    sm = pytest.importorskip("statsmodels.api")
    y, X, off = _count_design(with_offset=True)
    ref = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4})
    res = poisson(y, X, offset=off, cov_type="HAC", cov_kwds={"maxlags": 4})
    assert np.max(np.abs(res.bse.to_numpy() - ref.bse.to_numpy())) < ATOL
    assert np.max(np.abs(res.pvalues.to_numpy() - ref.pvalues.to_numpy())) < ATOL


def test_poisson_hac_use_correction_and_default_bandwidth():
    """``use_correction=True`` scales by ``n/(n-k)``; the default is False."""
    sm = pytest.importorskip("statsmodels.api")
    y, X, _ = _count_design()
    plain = poisson(y, X, cov_type="HAC", cov_kwds={"maxlags": 4})
    corrected = poisson(y, X, cov_type="HAC",
                        cov_kwds={"maxlags": 4, "use_correction": True})
    n, k = 200, 3
    assert np.max(np.abs(corrected.vcov - plain.vcov * (n / (n - k)))) < 1e-14

    # A deliberate divergence, not a parity claim: statsmodels' covtype
    # dispatcher indexes cov_kwds['maxlags'] unconditionally and dies with
    # a bare KeyError when it is absent (base/covtype.py:251) — the
    # cov_hac_simple default it defers to in a comment is never reached.
    # This module falls back to floor(4 (n/100)^(2/9)) instead, the same
    # bandwidth rule puremacro.regress.ols uses.
    with pytest.raises(KeyError, match="maxlags"):
        sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type="HAC")
    auto = poisson(y, X, cov_type="HAC")
    assert auto.cov_kwds["maxlags"] == 4
    explicit = poisson(y, X, cov_type="HAC", cov_kwds={"maxlags": 4})
    assert np.max(np.abs(auto.vcov - explicit.vcov)) == 0.0


def test_poisson_cluster_and_hc0_match_statsmodels_glm():
    sm = pytest.importorskip("statsmodels.api")
    y, X, _ = _count_design(n=240, seed=13)
    groups = np.repeat(np.arange(12), 20)

    ref_c = sm.GLM(y, X, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    res_c = poisson(y, X, cov_type="cluster", cov_kwds={"groups": groups})
    assert np.max(np.abs(res_c.bse.to_numpy() - ref_c.bse.to_numpy())) < ATOL
    assert res_c.n_groups == 12
    assert res_c.df_resid_inference == 11.0

    ref_h = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type="HC0")
    res_h = poisson(y, X, cov_type="HC0")
    assert np.max(np.abs(res_h.bse.to_numpy() - ref_h.bse.to_numpy())) < ATOL


def test_poisson_separation_raises():
    """A regressor that is non-zero only where the count is zero."""
    rng = np.random.default_rng(21)
    n = 150
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(0.5 + 0.3 * x)).astype(float)
    flag = (y == 0).astype(float)          # implies mu -> 0, coefficient -> -inf
    assert flag.sum() >= 10, "fixture must actually contain zero counts"
    X = pd.DataFrame({"const": 1.0, "x": x, "flag": flag})
    with pytest.raises(PerfectSeparationError):
        poisson(y, X)
    # Control: same sample, no separating column.
    assert poisson(y, X.drop(columns="flag")).converged is True


# ---------------------------------------------------------------------------
# The corpus call shapes, end to end
# ---------------------------------------------------------------------------


def test_n19_attribute_reads_all_match_statsmodels():
    """Every attribute ``N19:849-863`` reads off its ``poisson_hac`` result."""
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1644)
    years = np.arange(1470, 1650)
    n = years.size
    dry = rng.normal(size=n)
    log_nonfamine = np.log1p(rng.poisson(6.0, size=n))
    trend = (years - years.mean()) / 100.0
    X = pd.DataFrame({"const": 1.0, "dry": dry,
                      "log_nonfamine": log_nonfamine, "trend": trend})
    famine = pd.Series(
        rng.poisson(np.exp(-0.5 + 0.25 * dry + 0.3 * log_nonfamine)).astype(float))

    ref = sm.GLM(famine, X, family=sm.families.Poisson()).fit(
        cov_type="HAC", cov_kwds={"maxlags": 8})
    res = poisson(famine, X, cov_type="HAC", cov_kwds={"maxlags": 8})

    assert int(res.nobs) == int(ref.nobs)
    assert abs(float(res.params["dry"]) - float(ref.params["dry"])) < ATOL
    assert abs(float(res.bse["dry"]) - float(ref.bse["dry"])) < ATOL
    assert abs(float(res.pvalues["dry"]) - float(ref.pvalues["dry"])) < ATOL
    lo, hi = res.conf_int().loc["dry"]
    lo_r, hi_r = ref.conf_int().loc["dry"]
    assert abs(lo - lo_r) < ATOL and abs(hi - hi_r) < ATOL
    assert abs(res.pearson_chi2 / res.df_resid
               - ref.pearson_chi2 / ref.df_resid) < ATOL


def test_n04_logit_attribute_reads_all_match_statsmodels():
    """``N04:1289-1305``: odds ratios, p-values, ``nobs``, ``summary()``."""
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(640)
    n = 260
    nile_failure = (rng.uniform(size=n) < 0.3).astype(float)
    volc_lag = rng.gamma(1.0, 0.5, size=n)
    tsi_anom = rng.normal(size=n)
    X = pd.DataFrame({"const": 1.0, "nile_failure": nile_failure,
                      "volc_lag": volc_lag, "tsi_anom": tsi_anom})
    eta = -1.2 + 0.4 * nile_failure + 0.3 * volc_lag
    is_famine = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float))

    ref = sm.Logit(is_famine, X).fit(disp=False)
    res = logit(is_famine, X)

    for col in ("nile_failure", "volc_lag", "tsi_anom"):
        assert abs(float(np.exp(res.params[col])) -
                   float(np.exp(ref.params[col]))) < ATOL
        assert abs(float(res.pvalues[col]) - float(ref.pvalues[col])) < ATOL
        assert abs(float(res.bse[col]) - float(ref.bse[col])) < ATOL
    assert float(int(res.nobs)) == float(int(ref.nobs))
    text = res.summary()
    assert "Logit regression results" in text
    assert "nile_failure" in text and "P>|z|" in text


# ---------------------------------------------------------------------------
# Diagnostics, validation, plumbing
# ---------------------------------------------------------------------------


def test_singular_design_raises_a_named_linalg_error():
    """A duplicated column must name the culprits, not return garbage."""
    y, X = _binary_design()
    X = X.assign(dup=X["x1"])
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        logit(y, X)
    yc, Xc, _ = _count_design()
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        poisson(yc, Xc.assign(dup=Xc["x2"]))


def test_near_collinear_design_says_rescaling_will_not_help():
    """The other refusal, and it must not repeat spent advice.

    Every inversion equilibrates first, so by the time ``inv_xtx``
    refuses a full-rank design the columns it saw were already unit-norm
    and "try rescaling regressors" is advice the caller has no way to
    act on. What is left is genuine near-collinearity, which is
    scale-free, and the message names the remedy that works. The fixture
    is the one the message describes: a calendar-year trend raised to
    powers without being centred first.
    """
    rng = np.random.default_rng(11)
    n = 200
    t = np.linspace(1500.0, 1700.0, n)
    X = pd.DataFrame({"const": 1.0, **{f"t{d}": t ** d for d in range(1, 6)}})
    y = pd.Series((rng.uniform(size=n) < 0.5).astype(float))

    with pytest.raises(np.linalg.LinAlgError) as excinfo:
        logit(y, X)
    msg = str(excinfo.value)
    assert not isinstance(excinfo.value, PerfectSeparationError)
    assert "near-collinear" in msg and "rescaling will not help" in msg

    # Control: centring the trend — the remedy the message names — fits,
    # so the refusal is about the column directions and not the degree.
    c = (t - t.mean()) / (t.max() - t.min())
    centred = pd.DataFrame(
        {"const": 1.0, **{f"t{d}": c ** d for d in range(1, 6)}})
    assert logit(y, centred).converged is True


def test_missing_handling():
    y, X = _binary_design(n=120, seed=2)
    X = X.copy()
    X.loc[X.index[:4], "x1"] = np.nan

    with pytest.raises(ValueError, match="missing='raise'"):
        logit(y, X, missing="raise")

    dropped = logit(y, X, missing="drop")
    assert dropped.nobs == 116
    clean = logit(y.iloc[4:], X.iloc[4:])
    assert np.max(np.abs(dropped.params.to_numpy() - clean.params.to_numpy())) < 1e-12

    # The default does not drop, and does not carry the NaN into the fit
    # either. `pytest.raises((LinAlgError, ValueError))` used to stand
    # here and was satisfied by `LinAlgError('SVD did not converge')`
    # escaping the rank check — the exact undiagnosed failure
    # CONTRIBUTING's diagnostic-error contract exists to prevent, and
    # indistinguishable, to that assertion, from the named error below.
    with pytest.raises(ValueError, match="rows contain NaN or inf"):
        logit(y, X)


def test_input_validation():
    y, X = _binary_design(n=80, seed=5)
    with pytest.raises(ValueError, match="unit interval"):
        logit(y * 2.0, X)
    with pytest.raises(ValueError, match="non-negative counts"):
        poisson(y - 1.0, X)
    with pytest.raises(ValueError, match="not recognised"):
        logit(y, X, cov_type="hac-groupsum")
    with pytest.raises(ValueError, match="requires cov_kwds\\['groups'\\]"):
        logit(y, X, cov_type="cluster")
    with pytest.raises(ValueError, match="does not use keywords"):
        logit(y, X, cov_type="HC1", cov_kwds={"maxlags": 3})
    with pytest.raises(ValueError, match="takes no cov_kwds"):
        logit(y, X, cov_kwds={"maxlags": 3})
    with pytest.raises(ValueError, match="exposure must be strictly positive"):
        poisson(np.abs(y), X, exposure=np.zeros(80))
    with pytest.raises(ValueError, match="observations but X has"):
        logit(y.iloc[:-1], X)
    with pytest.raises(ValueError, match="missing must be"):
        logit(y, X, missing="sometimes")


def test_result_helpers():
    y, X, _ = _count_design()
    res = poisson(y, X, cov_type="HAC", cov_kwds={"maxlags": 4})
    assert isinstance(res, DiscreteResult)

    frame = res.to_frame()
    assert list(frame.columns) == ["term", "coef", "std_err", "z", "p_value",
                                   "ci_lo", "ci_hi"]
    assert frame["term"].tolist() == ["const", "x1", "x2"]

    # predict() defaults to the estimation design and offset.
    assert np.max(np.abs(res.predict() - res.mu)) < 1e-15
    assert np.max(np.abs(res.predict(which="linear") - res.linpred)) < 1e-15
    assert np.max(np.abs(res.predict(exog=X.to_numpy()) - res.mu)) < 1e-15
    assert np.max(np.abs(res.predict(exog=X, exposure=np.full(200, np.e))
                         - res.mu * np.e)) < 1e-12

    # fittedvalues reproduces statsmodels' inconsistency on purpose.
    assert np.max(np.abs(res.fittedvalues - res.mu)) == 0.0
    yb, Xb = _binary_design()
    lres = logit(yb, Xb)
    assert np.max(np.abs(lres.fittedvalues - lres.linpred)) == 0.0

    assert np.max(np.abs(res.resid_response - (res.endog - res.mu))) == 0.0
    assert abs(float(np.sum(res.resid_pearson ** 2)) - res.pearson_chi2) < 1e-9
    # Same identity for the binary family, whose variance function differs.
    assert abs(float(np.sum(lres.resid_pearson ** 2)) - lres.pearson_chi2) < 1e-9

    cs = 1.0 - np.exp((res.llnull - res.llf) * (2.0 / res.nobs))
    assert abs(res.pseudo_rsquared("cs") - cs) < 1e-15
    assert abs(res.prsquared - (1.0 - res.llf / res.llnull)) < 1e-15
    with pytest.raises(ValueError, match="mcf"):
        res.pseudo_rsquared("adjusted")
    with pytest.raises(ValueError, match="alpha"):
        res.conf_int(alpha=1.5)
    with pytest.raises(ValueError, match="which must be"):
        res.predict(which="probability")

    # The summary carries the Poisson-only over-dispersion line, and only
    # for the Poisson.
    assert "Pearson chi2" in res.summary()
    assert "Pearson chi2" not in lres.summary()

    # Frozen, as the package's result objects are.
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.llf = 0.0


def test_predict_reuses_the_estimation_offset():
    """``predict()`` with no arguments must not silently drop the offset."""
    y, X, off = _count_design(with_offset=True)
    res = poisson(y, X, offset=off)
    assert np.max(np.abs(res.offset - off)) == 0.0
    assert np.max(np.abs(res.predict() - res.mu)) < 1e-15
    # An offset-free prediction at the same design is a different vector,
    # so the assertion above is not satisfied by ignoring the offset.
    without = res.predict(exog=X)
    assert np.max(np.abs(without - res.mu)) > 1e-3
    assert np.max(np.abs(without * np.exp(off) - res.mu)) < 1e-12


def test_poisson_irls_start_survives_a_count_spike():
    """The starting value is load-bearing, not decoration.

    ``sm.GLM`` starts IRLS from ``mu0 = (y + ybar)/2``; starting from
    ``beta = 0`` (an implied mean of 1) makes the first Newton step
    enormous on a series with a spike, and the loop then fails to come
    back. Measured before the start was adopted: a spike of 5 000 took
    the ``beta = 0`` loop past 100 iterations without converging, and a
    spike of 100 000 overflowed. Both now converge in eight iterations
    and agree with statsmodels to machine precision.
    """
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1644)
    n = 180
    x = rng.normal(size=n)
    trend = np.linspace(-1.0, 1.0, n)
    X = pd.DataFrame({"const": 1.0, "x": x, "trend": trend})
    base = rng.poisson(np.exp(-0.5 + 0.3 * x)).astype(float)

    for spike in (500.0, 5_000.0, 100_000.0):
        y = base.copy()
        y[100:105] = spike
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            res = poisson(y, X)
        ref = sm.GLM(y, X, family=sm.families.Poisson()).fit(tol=1e-14)
        assert res.converged is True
        assert np.max(np.abs(res.params.to_numpy() - ref.params.to_numpy())) < ATOL
        assert np.max(np.abs(res.bse.to_numpy() - ref.bse.to_numpy())) < ATOL


def test_poisson_diverging_linear_predictor_raises():
    """An offset the design cannot absorb overflows ``exp`` — diagnosed."""
    rng = np.random.default_rng(2)
    n = 60
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=n)})
    y = rng.poisson(3.0, size=n).astype(float)
    off = np.where(np.arange(n) % 2 == 0, 800.0, -800.0)
    with pytest.raises(PerfectSeparationError, match="overflowed"):
        poisson(y, X, offset=off)
    # Control: the same offset scaled down to something exp can hold fits.
    assert poisson(y, X, offset=off / 400.0).converged is True


def test_poisson_all_zero_counts_raises():
    rng = np.random.default_rng(8)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=40)})
    with pytest.raises(PerfectSeparationError, match="every count is zero"):
        poisson(np.zeros(40), X)


def test_series_design_is_promoted_to_one_column():
    """A Series ``X`` keeps its name; an unnamed one falls back to ndarrays."""
    rng = np.random.default_rng(31)
    n = 200
    z = pd.Series(rng.normal(size=n), name="z")
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.8 * z))).astype(float)

    named = logit(y, z)
    assert isinstance(named.params, pd.Series)
    assert list(named.params.index) == ["z"]

    unnamed = logit(y, z.rename(None))
    assert isinstance(unnamed.params, np.ndarray)
    assert unnamed.names is None
    assert abs(float(named.params["z"]) - float(unnamed.params[0])) < 1e-15


def test_shape_and_identification_guards():
    y, X = _binary_design(n=60, seed=17)
    with pytest.raises(ValueError, match="offset/exposure length"):
        poisson(np.abs(y), X, offset=np.zeros(59))
    with pytest.raises(ValueError, match="X must be 2-D"):
        logit(np.zeros(8), np.zeros((2, 2, 2)))
    # Fewer observations than parameters: not identified.
    with pytest.raises(ValueError, match="not identified"):
        logit(y.iloc[:3], X.iloc[:3])
    # Everything dropped by missing='drop'.
    X_all_nan = X.copy()
    X_all_nan["x1"] = np.nan
    with pytest.raises(ValueError, match="no observations left"):
        logit(y, X_all_nan, missing="drop")


def test_cov_kwds_guards():
    y, X, _ = _count_design(n=100, seed=23)
    with pytest.raises(TypeError, match="cov_type must be a string"):
        poisson(y, X, cov_type=0)
    with pytest.raises(ValueError, match="maxlags"):
        poisson(y, X, cov_type="HAC", cov_kwds={"maxlags": -1})
    with pytest.raises(ValueError, match="kernel"):
        poisson(y, X, cov_type="HAC", cov_kwds={"maxlags": 2, "kernel": "parzen"})
    with pytest.raises(ValueError, match="groups"):
        poisson(y, X, cov_type="cluster", cov_kwds={"groups": np.arange(7)})
    with pytest.raises(ValueError, match="single group column"):
        poisson(y, X, cov_type="cluster",
                cov_kwds={"groups": np.zeros((100, 3))})
    # The 'uniform' kernel is a real alternative, not a rejected string.
    bart = poisson(y, X, cov_type="HAC", cov_kwds={"maxlags": 3})
    unif = poisson(y, X, cov_type="HAC",
                   cov_kwds={"maxlags": 3, "kernel": "uniform"})
    assert np.max(np.abs(bart.vcov - unif.vcov)) > 1e-8
    # maxlags = 0 is HC0.
    assert np.max(np.abs(poisson(y, X, cov_type="HAC",
                                 cov_kwds={"maxlags": 0}).vcov
                         - poisson(y, X, cov_type="HC0").vcov)) < 1e-15
    # df_correction=False keeps n-k as the t denominator under clustering.
    groups = np.repeat(np.arange(10), 10)
    kept = poisson(y, X, cov_type="cluster",
                   cov_kwds={"groups": groups, "df_correction": False})
    assert kept.df_resid_inference == kept.df_resid == 97.0


def test_module_does_not_import_statsmodels():
    """The whole point of the module. Guards Pyodide release gate 2."""
    import ast
    import pathlib

    import puremacro.regress.discrete as mod

    src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert "statsmodels" not in imported
    assert imported <= {"__future__", "warnings", "dataclasses",
                        "numpy", "pandas", "scipy"}


# ---------------------------------------------------------------------------
# Regressions: the guards that were missing, and the diagnoses that were wrong
# ---------------------------------------------------------------------------


def test_cov_kwds_rejects_unrecognised_keys():
    """A misspelled key must raise, not fall through to a default.

    This is the hole the value-checking tests above could not see. Every
    case here returned a *number* before the key check existed: the HAC
    typo silently re-ran at the fallback bandwidth, 4% away in ``bse``
    from the bandwidth the caller asked for, and the two cluster typos
    produced a covariance bit-identical to the default while the caller
    believed they had turned a correction off. statsmodels ignores all
    four — its own ``base/covtype.py`` docstring carries the TODO — and
    raises ``KeyError('maxlags')`` on the first only by accident, because
    it indexes that key rather than defaulting it.
    """
    y, X, _ = _count_design(n=180, seed=23)
    groups = np.repeat(np.arange(18), 10)

    with pytest.raises(ValueError, match=r"\['maxlag'\] not recognised"):
        poisson(y, X, cov_type="HAC", cov_kwds={"maxlag": 10})
    with pytest.raises(ValueError, match=r"\['use_corection'\] not recognised"):
        poisson(y, X, cov_type="cluster",
                cov_kwds={"groups": groups, "use_corection": False})
    with pytest.raises(ValueError, match=r"\['df_corection'\] not recognised"):
        poisson(y, X, cov_type="cluster",
                cov_kwds={"groups": groups, "df_corection": False})
    # A key that is real for another covariance is still wrong for this one.
    with pytest.raises(ValueError, match=r"\['groups'\] not recognised"):
        poisson(y, X, cov_type="HAC", cov_kwds={"maxlags": 10, "groups": groups})
    with pytest.raises(ValueError, match=r"\['maxlags'\] not recognised"):
        poisson(y, X, cov_type="cluster",
                cov_kwds={"groups": groups, "maxlags": 4})
    # Two at once are both named, so the message does not hide the second.
    with pytest.raises(ValueError, match=r"\['kernal', 'maxlag'\]"):
        poisson(y, X, cov_type="HAC", cov_kwds={"maxlag": 10, "kernal": "bartlett"})

    # Control: every accepted key still works, and the typo really did
    # change the answer it was hiding.
    good = poisson(y, X, cov_type="HAC",
                   cov_kwds={"maxlags": 10, "kernel": "bartlett",
                             "use_correction": True})
    assert good.cov_kwds["maxlags"] == 10
    fallback = poisson(y, X, cov_type="HAC", cov_kwds={"use_correction": True})
    assert fallback.cov_kwds["maxlags"] == 4       # floor(4 (180/100)^(2/9))
    rel = np.max(np.abs(fallback.bse.to_numpy() / good.bse.to_numpy() - 1.0))
    assert rel > 0.03
    kept = poisson(y, X, cov_type="cluster",
                   cov_kwds={"groups": groups, "use_correction": False,
                             "df_correction": False})
    assert kept.cov_kwds["use_correction"] is False


def test_cov_kwds_maxlags_must_be_an_integer():
    """A float bandwidth raises rather than being truncated to 4.

    ``int(4.7)`` is a bandwidth nobody chose. statsmodels raises
    ``TypeError`` from the ``range`` in its kernel loop, so this is
    parity as well as hygiene; ``True`` is accepted by both, being an
    ``int`` subclass, and means one lag.
    """
    sm = pytest.importorskip("statsmodels.api")
    y, X, _ = _count_design(n=180, seed=23)

    with pytest.raises(TypeError, match=r"maxlags.*must be an integer"):
        poisson(y, X, cov_type="HAC", cov_kwds={"maxlags": 4.7})
    with pytest.raises(TypeError, match=r"maxlags.*must be an integer"):
        poisson(y, X, cov_type="HAC", cov_kwds={"maxlags": "4"})
    with pytest.raises(TypeError):
        sm.GLM(y, X, family=sm.families.Poisson()).fit(
            cov_type="HAC", cov_kwds={"maxlags": 4.7})

    # numpy integers and bool are integers, and bool is parity.
    assert poisson(y, X, cov_type="HAC",
                   cov_kwds={"maxlags": np.int64(3)}).cov_kwds["maxlags"] == 3
    res = poisson(y, X, cov_type="HAC", cov_kwds={"maxlags": True})
    ref = sm.GLM(y, X, family=sm.families.Poisson()).fit(
        cov_type="HAC", cov_kwds={"maxlags": True})
    assert res.cov_kwds["maxlags"] == 1
    assert np.max(np.abs(res.bse.to_numpy() - ref.bse.to_numpy())) < ATOL


def _ill_conditioned_design(n=400, seed=0):
    """Full rank, ``cond(X) ~ 1.8e7`` — a population column left in units."""
    rng = np.random.default_rng(seed)
    pop = rng.uniform(2e6, 9e6, size=n)
    z = rng.normal(size=n)
    X = pd.DataFrame({"const": 1.0, "pop": pop, "z": z})
    p = 1.0 / (1.0 + np.exp(-(-1.0 + 3e-7 * pop + 0.4 * z)))
    y = pd.Series((rng.uniform(size=n) < p).astype(float))
    return y, X


def test_badly_scaled_design_fits_instead_of_being_called_separation():
    """A full-rank design in awkward units must be fitted, not diagnosed.

    ``X'WX`` failing to invert used to be reported as quasi-complete
    separation, on the premise that the caller had "already established
    that X is full rank, so a failure here can only come from the
    weights". The premise does not hold: the gate was
    ``np.linalg.matrix_rank`` (a rank test) while ``inv_xtx`` refuses on
    *conditioning*. The message then asserted that observations were
    perfectly predicted while printing the evidence against it — weights
    spanning ``2.5e-01`` to ``2.5e-01``, i.e. constant, at iteration 1,
    before any coefficient had moved. A caller branching on
    ``PerfectSeparationError`` to drop a regressor, as that message
    instructs, took the wrong branch.

    Nor is refusing it the answer, because there is nothing wrong with
    the design: ``cond(X'X)`` is ``3.2e+14`` in persons and ``33`` once
    the columns are scaled to unit norm. The whole of the "ill
    conditioning" is the unit of one column, the inversion is
    equilibrated, and the fit lands on statsmodels.
    """
    sm = pytest.importorskip("statsmodels.api")
    y, X = _ill_conditioned_design()
    Xv = X.to_numpy()
    assert np.linalg.matrix_rank(Xv) == 3, "fixture must be full rank"
    assert np.linalg.cond(Xv.T @ Xv) > 1e13, "fixture must look ill-conditioned"
    norms = np.sqrt(np.einsum("ij,ij->j", Xv, Xv))
    assert np.linalg.cond((Xv / norms).T @ (Xv / norms)) < 1e3, "... only in levels"

    res = logit(y, X)
    ref = sm.Logit(y, X).fit(disp=False)
    assert res.converged is True
    assert np.max(np.abs(res.params.to_numpy() - ref.params.to_numpy())) < 1e-10
    assert np.max(np.abs(res.bse.to_numpy() - ref.bse.to_numpy())) < 1e-10
    # The coefficient at issue is 2.4e-07, so an absolute tolerance would
    # pass on any answer at all; check it relatively too.
    assert np.max(np.abs(res.params.to_numpy() / ref.params.to_numpy() - 1.0)) < 1e-8
    assert np.max(np.abs(res.bse.to_numpy() / ref.bse.to_numpy() - 1.0)) < 1e-8

    # Rescaling by hand is now a no-op rather than a rescue.
    scaled = logit(y, X.assign(pop=X["pop"] / 1e6))
    assert abs(float(scaled.params["z"]) - float(res.params["z"])) < 1e-10
    assert abs(float(scaled.params["pop"]) * 1e-6
               - float(res.params["pop"])) < 1e-16


def test_separation_still_fires_after_the_design_gate_was_equilibrated():
    """The positive control for the test above, in both families.

    Equilibrating the inversion is what lets the badly scaled design
    through, and it also rescues the weighted cross-product in the one
    case where its collapse *was* the diagnosis. Both separations must
    still raise — the logit's through the singular ``X'WX``, the
    Poisson's through the non-convergence conjunction — or the previous
    test bought its parity by disabling the guard.
    """
    rng = np.random.default_rng(3)
    n = 120
    x = rng.normal(size=n)
    y = (rng.uniform(size=n) < 0.5).astype(float)
    flag = np.zeros(n)
    flag[np.flatnonzero(y == 1.0)[:20]] = 1.0   # 1 implies y == 1
    with pytest.raises(PerfectSeparationError, match="separation"):
        logit(y, pd.DataFrame({"const": 1.0, "x": x, "flag": flag}))

    rng = np.random.default_rng(21)
    n = 150
    x = rng.normal(size=n)
    counts = rng.poisson(np.exp(0.5 + 0.3 * x)).astype(float)
    zero = (counts == 0).astype(float)
    assert zero.sum() >= 10, "fixture must actually contain zero counts"
    with pytest.raises(PerfectSeparationError, match="separation"):
        poisson(counts, pd.DataFrame({"const": 1.0, "x": x, "flag": zero}))

    # ... and the badly scaled design is not caught by either of them.
    y_ill, X_ill = _ill_conditioned_design()
    assert logit(y_ill, X_ill).converged is True


def test_collapsed_weights_alone_are_not_separation():
    """The other half of the conjunction, and it is not decoration.

    The Poisson separation guard fires on "did not converge AND the
    weights collapsed". Weight collapse on its own is a perfectly
    ordinary property of a count model with a wide exposure: this fixture
    spans eighteen log-units of exposure, reaches
    ``min(w)/max(w) = 9.4e-09`` — below the ``1e-8`` threshold — and
    converges to the right answer. A guard written on the weights alone
    would reject it.
    """
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(77)
    n = 300
    expo = np.exp(rng.uniform(0.0, 18.0, size=n))
    x = rng.normal(size=n)
    counts = rng.poisson(
        np.minimum(expo * np.exp(-2.0 + 0.3 * x), 1e7)).astype(float)
    X = pd.DataFrame({"const": 1.0, "x": x})

    res = poisson(counts, X, exposure=expo)
    assert res.converged is True
    ratio = float(res.weights.min() / res.weights.max())
    assert ratio < 1e-8, f"fixture must collapse the weights; got {ratio:.2e}"

    ref = sm.GLM(counts, X, family=sm.families.Poisson(),
                 exposure=expo).fit(tol=1e-14)
    assert np.max(np.abs(res.params.to_numpy() - ref.params.to_numpy())) < ATOL


def test_nan_in_endog_is_named_as_missing_data():
    """A NaN response is missing data, not a diverging linear predictor.

    ``np.any(y < 0) or np.any(y > 1)`` is False for every NaN, so the
    unit-interval guard had a hole exactly the width of a missing value
    and the fit went on to report ``max |eta| = nan`` as separation.
    statsmodels writes the same check as ``not np.all((y >= 0) & (y <=
    1))``, which has no hole.
    """
    y, X = _binary_design(n=120, seed=41)
    y = y.copy()
    y.iloc[7] = np.nan
    # The hole itself, so the test names the mechanism and not just the symptom.
    arr = y.to_numpy()
    assert not (np.any(arr < 0.0) or np.any(arr > 1.0))
    assert not np.all((arr >= 0.0) & (arr <= 1.0))

    with pytest.raises(ValueError, match="rows contain NaN or inf"):
        logit(y, X)
    assert logit(y, X, missing="drop").nobs == 119

    yc, Xc, _ = _count_design(n=120, seed=41)
    yc = yc.copy()
    yc.iloc[4] = np.nan
    with pytest.raises(ValueError, match="rows contain NaN or inf"):
        poisson(yc, Xc)
    assert poisson(yc, Xc, missing="drop").nobs == 119

    # The range guards still fire for values that are genuinely out of
    # range, so the NaN-safe rewrite did not neutralise them.
    with pytest.raises(ValueError, match="unit interval"):
        logit(_binary_design(n=120, seed=41)[0] * 2.0, X)
    with pytest.raises(ValueError, match="non-negative counts"):
        poisson(yc.fillna(0.0) - 1.0, Xc)


def test_nan_in_exog_is_named_as_missing_data():
    """A non-finite design must not die inside ``matrix_rank``.

    Before the guard, ``missing='none'`` reached
    ``np.linalg.matrix_rank`` and raised ``LinAlgError('SVD did not
    converge')`` — no name, no mention of missing data, in breach of
    CONTRIBUTING's diagnostic-error contract. An ``inf`` got further
    still and was reported as a diverging linear predictor. statsmodels
    raises ``MissingDataError('exog contains inf or nans')`` on both,
    whatever ``missing`` says.
    """
    y, X = _binary_design(n=120, seed=41)
    yc, Xc, _ = _count_design(n=120, seed=41)

    for bad in (np.nan, np.inf, -np.inf):
        X_bad = X.copy()
        X_bad.iloc[3, 1] = bad
        with pytest.raises(ValueError, match="rows contain NaN or inf") as e:
            logit(y, X_bad)
        assert "in X" in str(e.value) and "position 3" in str(e.value)

        Xc_bad = Xc.copy()
        Xc_bad.iloc[3, 1] = bad
        with pytest.raises(ValueError, match="rows contain NaN or inf"):
            poisson(yc, Xc_bad)

    # A non-finite offset is named too, and 'drop' still drops it.
    off = np.zeros(120)
    off[5] = np.nan
    with pytest.raises(ValueError, match="offset/exposure"):
        poisson(yc, Xc, offset=off)
    assert poisson(yc, Xc, offset=off, missing="drop").nobs == 119


def test_predict_keeps_the_estimation_offset_when_only_exposure_is_given():
    """``predict(exposure=...)`` must not zero the fitted offset.

    ``GLMResults.predict`` replaces the estimation offset only when
    ``offset`` itself is passed (``generalized_linear_model.py:990``);
    reading the rule as "any of the three arguments discards it" costs a
    factor of ``exp(offset)`` on every prediction. Measured on this
    fixture before the fix: ``max|ours - statsmodels| = 4.80`` on a
    series whose mean is 4.06, and ``max|ours - exp(X beta + 1)| = 0``
    exactly — the fitted offset was simply absent.
    """
    sm = pytest.importorskip("statsmodels.api")
    y, X, off = _count_design(n=150, seed=5, with_offset=True)
    assert np.max(np.abs(off)) > 0.1, "fixture must carry a real offset"

    res = poisson(y, X, offset=off)
    ref = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit(tol=1e-14)

    e = np.full(150, np.e)
    other = 0.1 * np.ones(150)
    for kwargs in ({}, {"exposure": e}, {"offset": other},
                   {"offset": other, "exposure": e},
                   {"exog": X, "exposure": e}, {"exog": X}):
        ours = res.predict(**kwargs)
        theirs = np.asarray(ref.predict(**kwargs))
        assert np.max(np.abs(ours - theirs)) < 1e-10, kwargs

    # The exposure path is not satisfied by ignoring the offset: the two
    # differ by exp(off), which is what went unnoticed.
    naive = np.exp(X.to_numpy() @ res.params.to_numpy() + 1.0)
    assert np.max(np.abs(res.predict(exposure=e) - naive)) > 1e-2

    # Length and positivity are checked rather than broadcast.
    with pytest.raises(ValueError, match="offset has 3 entries"):
        res.predict(offset=np.zeros(3))
    with pytest.raises(ValueError, match="exposure must be strictly positive"):
        res.predict(exposure=np.zeros(150))


def test_missing_drop_composes_with_cluster_groups():
    """Dropping rows must not leave the cluster labels one row too long.

    ``_prepare`` shortens ``y`` / ``X`` / ``offset`` while
    ``cov_kwds['groups']`` came from the caller's undropped frame, so the
    two lengths used to disagree and the fit died in the length check
    (statsmodels dies too, with "The weights and list don't have the same
    length"). The labels are subset by the same mask instead; the two
    lengths cannot be confused, because a drop strictly shortens.
    """
    y, X, _ = _count_design(n=120, seed=3)
    X_bad = X.copy()
    X_bad.iloc[0, 1] = np.nan
    groups = np.repeat(np.arange(12), 10)

    res = poisson(y, X_bad, missing="drop", cov_type="cluster",
                  cov_kwds={"groups": groups})
    assert res.nobs == 119 and res.n_groups == 12

    # Identical to doing the drop by hand, labels and all.
    manual = poisson(y.iloc[1:], X.iloc[1:], cov_type="cluster",
                     cov_kwds={"groups": groups[1:]})
    assert np.max(np.abs(res.vcov - manual.vcov)) < 1e-15
    assert res.df_resid_inference == manual.df_resid_inference

    # Labels already matching the retained rows are left alone ...
    pre = poisson(y, X_bad, missing="drop", cov_type="cluster",
                  cov_kwds={"groups": groups[1:]})
    assert np.max(np.abs(pre.vcov - res.vcov)) < 1e-15
    # ... and a length matching neither is still an error naming both numbers.
    with pytest.raises(ValueError, match="7 entries for 119 observations"):
        poisson(y, X_bad, missing="drop", cov_type="cluster",
                cov_kwds={"groups": np.arange(7)})


def test_single_cluster_raises_a_named_error():
    """``G = 1`` is a realistic user error and must not be a ZeroDivisionError.

    Parity here was a bare ``ZeroDivisionError`` from ``1/(1-1.0)`` in the
    small-sample factor — statsmodels dies the same way at
    ``sandwich_covariance.py:538`` — which meets nobody's idea of a named
    diagnostic. Turning the correction off is worse than the crash: the
    meat is then the outer product of the total score, which vanishes at
    the maximum, so every standard error comes back at machine epsilon.
    """
    y, X = _binary_design(n=200, seed=9)
    one = np.zeros(200, dtype=int)

    with pytest.raises(ValueError, match="at least two clusters"):
        logit(y, X, cov_type="cluster", cov_kwds={"groups": one})
    with pytest.raises(ValueError, match="at least two clusters"):
        logit(y, X, cov_type="cluster",
              cov_kwds={"groups": one, "use_correction": False})
    with pytest.raises(ValueError, match="at least two clusters"):
        logit(y, X, cov_type="cluster",
              cov_kwds={"groups": np.full(200, "only", dtype=object)})

    sm = pytest.importorskip("statsmodels.api")
    # What is being refused, stated as a measurement rather than an
    # assertion: statsmodels answers this call with standard errors at
    # machine epsilon and no warning of any kind.
    silent = sm.Logit(y, X).fit(
        disp=False, cov_type="cluster",
        cov_kwds={"groups": one, "use_correction": False})
    assert np.max(silent.bse.to_numpy()) < 1e-15
    assert np.min(np.abs(silent.params.to_numpy())) > 1e-3   # ... on real coefficients

    # Two clusters is the smallest thing that works, and all-singleton
    # clusters keep matching statsmodels — so the guard is a floor, not a
    # blanket refusal.
    assert logit(y, X, cov_type="cluster",
                 cov_kwds={"groups": np.arange(200) % 2}).n_groups == 2
    res = logit(y, X, cov_type="cluster", cov_kwds={"groups": np.arange(200)})
    ref = sm.Logit(y, X).fit(disp=False, cov_type="cluster",
                             cov_kwds={"groups": np.arange(200)})
    assert np.max(np.abs(res.bse.to_numpy() - ref.bse.to_numpy())) < ATOL


def test_declared_divergences_are_actually_declared():
    """PARITY_SPEC §7: a divergence that is not written down is a bug.

    Every guard below refuses an input statsmodels accepts. That is
    defensible — the saturated logit statsmodels fits is separated, and
    its answer is an artefact of where the optimiser stopped — but only
    if a reader of the docstring can find out before the exception does
    it for them. The ``n == k`` guard shipped in neither ``Raises`` nor
    ``Notes``.
    """
    import puremacro.regress.discrete as mod

    with pytest.raises(ValueError, match="not identified"):
        logit(*_saturated_design())
    assert "Declared divergences" in mod.__doc__
    assert "n == k" in logit.__doc__
    assert "Declared divergences" in poisson.__doc__
    # Each guard that refuses an input statsmodels accepts is declared,
    # so a reader finds it before the exception does it for them.
    # Whitespace-normalised: the assertion is about content, not about
    # where the paragraph happens to wrap.
    declared = " ".join(mod.__doc__.split())
    for entry in ("not identified", "naming the cluster count",
                  "naming the key", "n == k", "missing='drop'",
                  "naming the array and the first offending row"):
        assert entry in declared, entry
    # The guard is about identification and nothing else: a design with
    # the same six rows and room to spare fits, and matches statsmodels.
    sm = pytest.importorskip("statsmodels.api")
    y, X = _binary_design(n=12, seed=34)
    res = logit(y, X)
    ref = sm.Logit(y, X).fit(disp=False)
    assert res.nobs == 12
    assert np.max(np.abs(res.params.to_numpy() - ref.params.to_numpy())) < 1e-8


def _saturated_design(n=5):
    """``n`` observations on five parameters — identified only if ``n > 5``."""
    rng = np.random.default_rng(34)
    X = pd.DataFrame(
        np.column_stack([np.ones(n), rng.normal(size=(n, 4))]),
        columns=["const", "a", "b", "c", "d"],
    )
    y = pd.Series(np.resize([0.0, 1.0, 0.0, 1.0, 1.0], n))
    return y, X
