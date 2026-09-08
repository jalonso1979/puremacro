"""Parity and independent-oracle tests for :mod:`puremacro.regress.quantile`.

Three different standards are applied here, deliberately:

1. **statsmodels parity** — the target this module was written against. Every
   such test calls ``pytest.importorskip("statsmodels")`` so the suite still
   runs in a statsmodels-free environment (the Pyodide case, which is the
   whole point of the package).
2. **An independent oracle** — a brute-force ``scipy.optimize`` minimisation
   of the check function, which shares no code with either implementation. If
   puremacro and statsmodels were wrong in the same way, parity alone would
   not notice; this would.
3. **A cross-check against the other quantile fitter puremacro ships**,
   ``puremacro.lp.quantile._qreg``, which solves the exact Koenker-Bassett
   linear program. It is a *different estimator*, so the test records the
   size of the disagreement rather than asserting it away.

Mutation survey (CONTRIBUTING, "Making sure a test can fail")
-------------------------------------------------------------
Nineteen hand-written mutations of ``puremacro/regress/quantile.py`` were run
against this file. Seventeen were caught, including: widening the IRLS
residual floor from 1e-6 to 1e-3; swapping ``q`` for ``1-q`` in the IRLS
weights; changing the Hall-Sheather exponent from 1/3 to 1/2; rescaling the
Epanechnikov kernel; ``min`` → ``max`` in the bandwidth rescaling; 1.34 →
1.35 in the same expression; dropping ``q(1-q)`` from the ``'iid'``
covariance; ``t`` p-values and confidence intervals replaced by normal ones;
ignoring a fixed ``bandwidth=`` argument; ``k_constant`` forced to zero;
``missing='drop'`` silently keeping its rows; ``converged`` hard-wired to
``True``; ``_check_loss`` returning a Python float instead of a numpy scalar;
and removing the ``inv_xtx`` singularity gate.

Two survived. One was not inert after all:

* ``np.where(e > 0, ...)`` → ``np.where(e >= 0, ...)`` in the robust sandwich.
  This was written off as unreachable — the two branches differ only for a
  residual that is *exactly* 0.0, and IRLS floors residual magnitudes at 1e-6
  — but the reasoning confused "no *continuous* fixture reaches it" with "no
  input reaches it". Integer data reaches it constantly: of 600 randomised
  integer designs, 172 produced at least one residual exactly equal to 0.0,
  and the two conventions move ``bse`` by up to 0.18 on them. The module was
  right (it matches statsmodels' strict ``>`` bit for bit on all 172); the
  test suite simply could not see it. ``_tied_integer_design`` and
  ``test_robust_sandwich_tie_convention_on_exact_zero_residuals`` now do, and
  the mutant dies there.
* ``beta = np.ones(k)`` → ``np.zeros(k)`` as the IRLS starting value. The
  initial ``beta`` never enters the arithmetic; it is compared against the
  first least-squares step and discarded, as statsmodels' own comment says.
  It would matter only for a design whose first step landed within ``tol`` of
  the starting vector. Genuinely inert.

``df_resid = nobs - rank`` → ``nobs - exog.shape[1]`` also survives, and is
inert for a reason worth writing down: ``matrix_rank`` calls a design
deficient only past ``cond(X) ≈ 1e13``, and the conditioning gate refuses
everything past ``cond(X'X) = 1e14`` — twelve orders earlier — so the two
spellings can differ only on inputs that never reach the line.
``test_df_resid_uses_rank_and_the_two_spellings_cannot_diverge`` pins both
halves of that argument rather than the mutant.

Section 5 covers the defects an adversarial review found after the above:
a conditioning gate that measured column scale rather than column dependence,
a container rule that keyed off ``X`` alone, positionally-joined pandas
indices, stringified column labels, and an ``inf`` treated as missing. Each of
those tests was run against the pre-fix module and seen to fail there.

One mutation — swapping the private ``_score_at_percentile`` for
``np.percentile`` — survived the parity tests (the difference is 9e-16) *and*
survived a first attempt at a test that exercised the helper in isolation
rather than the call site. It is caught by
``test_reported_bandwidth_uses_the_scipy_percentile_path``, which rebuilds the
reported bandwidth from the fit's own residuals; the isolated helper test is
kept alongside it because it localises the failure.

The convergence-cycle branch of the IRLS loop is **not** covered. It needs
``n_iter >= 300`` plus a genuine limit cycle; statsmodels' own comment on it
reads "should not happen", and forty randomised attempts on heavily-tied
integer data failed to provoke one. Reaching it would require patching the
subject, which is the thing CONTRIBUTING says not to do.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.regress.quantile import QuantRegResult, quantreg

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)

# Why atol=1e-6 and not 1e-10, which the rest of the parity work uses:
# QuantReg has no closed form. Both implementations run the same IRLS loop to
# the same stopping rule, ``max|Δβ| ≤ p_tol = 1e-6``, so the *estimator itself*
# is only defined to about 1e-6 — a caller who changed the BLAS, the summation
# order, or the platform's ``pinv`` could land one iteration earlier or later
# and move the answer by up to that stopping threshold. Promising 1e-10 would
# be promising something about LAPACK, not about this code.
#
# The measured agreement on this machine (numpy 2.x / Accelerate, statsmodels
# 0.14.6) is in fact 0.0 — bit-identical params, bse, bandwidth, sparsity and
# iteration count at all five quantiles, both vcov branches, all five kernels
# and all three bandwidth rules. The looser number is the promise; the exact
# one is the observation.
PARAM_ATOL = 1e-6

# Standard errors are a smooth function of the converged coefficients through
# the sparsity sandwich, so a 1e-6 wobble in β moves bse by roughly the same
# order, scaled by the density estimate. 1e-6 absolute on standard errors that
# are O(0.1-1) here is ~1e-5 relative — defensible, and still four orders
# tighter than any inference anyone draws from them.
BSE_ATOL = 1e-6


def _design(n=240, seed=7):
    """Heteroskedastic, fat-tailed cross-section: the case the sandwich is for."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 0.7 * x1 - 0.4 * x2 + (1 + 0.5 * np.abs(x1)) * rng.standard_t(5, size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    return y, X


def _check_loss(b, y, X, q):
    """Σ ρ_q(y - Xb) with ρ_q(u) = u(q - 1{u<0}).

    Written from the definition, not from either implementation, because it is
    the yardstick both of them are measured against.
    """
    r = np.asarray(y, float) - np.asarray(X, float) @ np.asarray(b, float)
    return float(np.sum(np.where(r < 0, (q - 1) * r, q * r)))


# ---------------------------------------------------------------------------
# 1. statsmodels parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q", QUANTILES)
def test_params_parity_across_quantiles(q):
    sm = pytest.importorskip("statsmodels.api")
    y, X = _design()

    ref = sm.QuantReg(y, X).fit(q=q)
    got = quantreg(y, X, q=q)

    np.testing.assert_allclose(got.params, ref.params, atol=PARAM_ATOL, rtol=0)
    # The iteration count is the sharpest single check that the loop is the
    # same loop and not merely a converging one: a different starting value,
    # a different residual floor, or ``solve`` instead of ``pinv`` all change
    # it while leaving the coefficients close.
    assert got.iterations == ref.iterations


@pytest.mark.parametrize("q", QUANTILES)
def test_standard_error_and_inference_parity(q):
    sm = pytest.importorskip("statsmodels.api")
    y, X = _design()

    ref = sm.QuantReg(y, X).fit(q=q)
    got = quantreg(y, X, q=q)

    np.testing.assert_allclose(got.bse, ref.bse, atol=BSE_ATOL, rtol=0)
    np.testing.assert_allclose(got.tvalues, ref.tvalues, atol=1e-5, rtol=0)
    np.testing.assert_allclose(got.pvalues, ref.pvalues, atol=1e-8, rtol=0)
    np.testing.assert_allclose(got.conf_int(), ref.conf_int(), atol=1e-5, rtol=0)
    np.testing.assert_allclose(got.cov_params(), ref.cov_params(), atol=1e-8, rtol=0)
    # The kernel-density pieces the sandwich is built from, checked separately
    # so a compensating pair of errors cannot hide inside bse.
    assert got.bandwidth == pytest.approx(ref.bandwidth, abs=1e-10)
    assert got.sparsity == pytest.approx(ref.sparsity, abs=1e-8)
    # use_t is True in statsmodels because QuantRegResults inherits
    # RegressionResults with cov_type='nonrobust'; the p-values are t(n-rank).
    assert got.use_t is True and ref.use_t is True


@pytest.mark.parametrize("q", (0.25, 0.5, 0.9))
def test_vcov_iid_parity(q):
    sm = pytest.importorskip("statsmodels.api")
    y, X = _design()

    ref = sm.QuantReg(y, X).fit(q=q, vcov="iid")
    got = quantreg(y, X, q=q, vcov="iid")

    np.testing.assert_allclose(got.bse, ref.bse, atol=BSE_ATOL, rtol=0)
    np.testing.assert_allclose(got.cov_params(), ref.cov_params(), atol=1e-8, rtol=0)

    # 'iid' must not silently be 'robust' — this design is heteroskedastic on
    # purpose. Away from the median the two have to disagree materially. AT
    # the median they are algebraically identical and must agree exactly:
    # d = (q/f̂₀)² = ((1-q)/f̂₀)² = 0.25/f̂₀² is then constant, so the sandwich
    # collapses to 0.25/f̂₀² (X'X)⁻¹, which is the 'iid' formula at q=0.5.
    robust = quantreg(y, X, q=q, vcov="robust")
    spread = np.max(np.abs(np.asarray(got.bse) - np.asarray(robust.bse)))
    if q == 0.5:
        assert spread < 1e-12
    else:
        assert spread > 1e-3


@pytest.mark.parametrize("kernel", ["epa", "biw", "cos", "gau", "par"])
@pytest.mark.parametrize("rule", ["hsheather", "bofinger", "chamberlain"])
def test_kernel_and_bandwidth_rule_parity(kernel, rule):
    sm = pytest.importorskip("statsmodels.api")
    y, X = _design()

    ref = sm.QuantReg(y, X).fit(q=0.7, kernel=kernel, bandwidth=rule)
    got = quantreg(y, X, q=0.7, kernel=kernel, bandwidth=rule)

    np.testing.assert_allclose(got.bse, ref.bse, atol=BSE_ATOL, rtol=0)
    assert got.bandwidth == pytest.approx(ref.bandwidth, abs=1e-10)


def test_result_surface_matches_statsmodels():
    """Every attribute the corpus reads off a ``smf.quantreg`` fit."""
    sm = pytest.importorskip("statsmodels.api")
    y, X = _design()

    ref = sm.QuantReg(y, X).fit(q=0.6)
    got = quantreg(y, X, q=0.6)

    assert got.nobs == ref.nobs
    assert got.df_resid == ref.df_resid
    assert got.df_model == ref.df_model
    assert got.rank == 3
    assert got.scale == ref.scale == 1.0
    assert got.prsquared == pytest.approx(ref.prsquared, abs=1e-10)
    np.testing.assert_allclose(got.resid, ref.resid, atol=1e-8, rtol=0)
    np.testing.assert_allclose(got.fittedvalues, ref.fittedvalues, atol=1e-8, rtol=0)
    # statsmodels NaNs out the least-squares goodness-of-fit surface for a
    # check-loss fit; so do we, rather than quietly inventing an R².
    for name in ("rsquared", "rsquared_adj", "llf", "aic", "bic"):
        assert np.isnan(getattr(got, name)), name
        assert np.isnan(getattr(ref, name)), name
    assert isinstance(got.summary(), str) and "QuantReg" in got.summary()


def test_cov_type_label_diverges_from_statsmodels_deliberately():
    """The one non-numeric divergence, asserted so it cannot drift unnoticed.

    ``QuantReg.fit`` passes its sandwich in through ``normalized_cov_params``
    and never sets ``cov_type``, so statsmodels labels a robust covariance
    ``'nonrobust'``. We report the ``vcov`` that was actually used.
    """
    sm = pytest.importorskip("statsmodels.api")
    y, X = _design()

    assert sm.QuantReg(y, X).fit(q=0.5).cov_type == "nonrobust"
    assert quantreg(y, X, q=0.5).cov_type == "robust"
    assert quantreg(y, X, q=0.5, vcov="iid").cov_type == "iid"


def test_named_design_reproduces_the_corpus_call_site():
    """N03_hyde_spatial_anthropocene.py:1209-1210, without the formula parser.

    The notebook fits ``smf.quantreg("log_popd_1500 ~ sigma_s + I(sigma_s**2)",
    d_env).fit(q=0.9)`` and then reads ``q90.params["sigma_s"]`` and
    ``q90.params["I(sigma_s ** 2)"]`` — patsy's spacing and all — to compute a
    turning point. puremacro has no formula parser, so the caller names the
    columns; this test asserts that naming them is enough.
    """
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1500)
    n = 180
    sigma_s = rng.uniform(0.0, 14.0, size=n)
    log_popd = (
        -1.0 + 0.55 * sigma_s - 0.021 * sigma_s ** 2 + rng.standard_t(6, size=n) * 0.9
    )
    d_env = pd.DataFrame(
        {"const": 1.0, "sigma_s": sigma_s, "I(sigma_s ** 2)": sigma_s ** 2},
        index=[f"ISO{i:03d}" for i in range(n)],
    )
    y = pd.Series(log_popd, index=d_env.index, name="log_popd_1500")

    ref = sm.QuantReg(y, d_env).fit(q=0.9)
    q90 = quantreg(y, d_env, q=0.9)

    assert isinstance(q90.params, pd.Series)
    assert list(q90.params.index) == ["const", "sigma_s", "I(sigma_s ** 2)"]
    np.testing.assert_allclose(q90.params, ref.params, atol=PARAM_ATOL, rtol=0)

    tp_got = float(-q90.params["sigma_s"] / (2 * q90.params["I(sigma_s ** 2)"]))
    tp_ref = float(-ref.params["sigma_s"] / (2 * ref.params["I(sigma_s ** 2)"]))
    # The published number is printed to one decimal; a ratio of two
    # coefficients amplifies the parity gap, so it is checked at the precision
    # that is actually quoted rather than at PARAM_ATOL.
    assert tp_got == pytest.approx(tp_ref, abs=5e-5)
    assert q90.bse.index.equals(q90.params.index)
    assert q90.pvalues.index.equals(q90.params.index)
    assert list(q90.conf_int().columns) == [0, 1]
    assert q90.resid.index.equals(d_env.index)


def test_ndarray_input_returns_ndarrays():
    """statsmodels' pandas-in/pandas-out convention, and its converse.

    Bare ndarrays go in and bare ndarrays come out — but the *names* are
    statsmodels' invented ones, not ``x1 … xk``: ``_make_exog_names``
    (base/data.py:643) renames a zero-variance column ``const`` and closes the
    numbering up around it, so this three-column design with an intercept is
    ``('const', 'x1', 'x2')``. The off-by-one is what makes ``params['const']``
    work on a design built with ``np.column_stack``.
    """
    sm = pytest.importorskip("statsmodels.api")
    y, X = _design(n=120, seed=3)
    res = quantreg(y, X, q=0.4)
    assert isinstance(res.params, np.ndarray)
    assert isinstance(res.bse, np.ndarray)
    assert isinstance(res.cov_params(), np.ndarray)
    assert isinstance(res.conf_int(), np.ndarray)
    assert res.conf_int().shape == (3, 2)

    ref = sm.QuantReg(y, X).fit(q=0.4)
    assert isinstance(ref.params, np.ndarray)
    assert list(res.exog_names) == list(ref.model.exog_names) == ["const", "x1", "x2"]
    # Without a constant the numbering starts at x1 again, for both.
    ref_nc = sm.QuantReg(y, X[:, 1:]).fit(q=0.4)
    assert list(quantreg(y, X[:, 1:], q=0.4).exog_names) == list(
        ref_nc.model.exog_names
    ) == ["x1", "x2"]


def test_max_iter_bookkeeping_matches_statsmodels():
    """A truncated run must be truncated at the same place, and must complain."""
    sm = pytest.importorskip("statsmodels.api")
    y, X = _design()

    with warnings.catch_warnings(record=True) as ref_w:
        warnings.simplefilter("always")
        ref = sm.QuantReg(y, X).fit(q=0.9, max_iter=4)
    with pytest.warns(UserWarning, match="max_iter"):
        got = quantreg(y, X, q=0.9, max_iter=4)

    assert any("Maximum number of iterations" in str(w.message) for w in ref_w)
    assert got.iterations == ref.iterations == 4
    assert got.converged is False
    np.testing.assert_allclose(got.params, ref.params, atol=PARAM_ATOL, rtol=0)


def test_degenerate_bandwidth_warns_where_statsmodels_is_silent():
    """Small ``n`` at an extreme ``q`` pushes ``q ± h₀`` outside ``(0, 1)``.

    statsmodels returns a vector of NaN standard errors with no warning. We
    return the same numbers — parity — but say so, which is the house
    "diagnostic over silent" contract.
    """
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    y = rng.normal(size=18)
    X = np.column_stack([np.ones(18), rng.normal(size=18)])

    ref = sm.QuantReg(y, X).fit(q=0.9)
    with pytest.warns(UserWarning, match=r"outside \(0, 1\)"):
        got = quantreg(y, X, q=0.9)

    assert np.all(np.isnan(ref.bse)) and np.all(np.isnan(got.bse))
    assert np.isnan(ref.bandwidth) and np.isnan(got.bandwidth)
    # The coefficients are unaffected: only the covariance depends on h.
    np.testing.assert_allclose(got.params, ref.params, atol=PARAM_ATOL, rtol=0)


def test_degenerate_constant_response_matches_statsmodels_exactly():
    """A constant ``y`` collapses the bandwidth to zero. Match the wreckage.

    statsmodels emits five ``RuntimeWarning``s from numpy and returns
    ``bse = [nan, nan]``, ``bandwidth = 0.0``, ``sparsity = nan`` and
    ``prsquared = -inf``. Reproducing ``-inf`` is not an accident of
    arithmetic: it requires the pseudo-R² denominator to stay a numpy scalar,
    because a Python float would raise ``ZeroDivisionError`` instead. We
    return the same numbers with one informative warning in place of numpy's
    five.
    """
    sm = pytest.importorskip("statsmodels.api")
    n = 80
    y = np.ones(n)
    X = np.column_stack([np.ones(n), np.arange(float(n))])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = sm.QuantReg(y, X).fit(q=0.5)
        ref_prsquared = ref.prsquared
    with pytest.warns(UserWarning, match="residual density at zero"):
        got = quantreg(y, X, q=0.5)

    np.testing.assert_allclose(got.params, ref.params, atol=1e-12, rtol=0)
    assert np.all(np.isnan(got.bse)) and np.all(np.isnan(ref.bse))
    assert got.bandwidth == ref.bandwidth == 0.0
    assert np.isnan(got.sparsity) and np.isnan(ref.sparsity)
    assert got.prsquared == ref_prsquared == float("-inf")


# ---------------------------------------------------------------------------
# 2. Independent oracle: brute-force minimisation of the check function
# ---------------------------------------------------------------------------
def test_median_matches_bruteforce_check_function_minimisation():
    """LAD by IRLS vs. a derivative-free minimiser of Σ|y - Xb|.

    ``scipy.optimize.minimize`` shares nothing with the IRLS loop — no
    weights, no ``pinv``, no residual floor — so agreement here is evidence
    about the *estimand*, not about the transcription. The objective is
    convex, so any local minimum found is the global one; it is also
    piecewise linear, hence the derivative-free methods and the several
    starting points.
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(20260906)
    n = 120
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.5 + 1.2 * x1 - 0.8 * x2 + rng.standard_normal(n) * (1 + 0.3 * np.abs(x2))
    X = np.column_stack([np.ones(n), x1, x2])
    q = 0.5

    b_irls = np.asarray(quantreg(y, X, q=q).params, float)

    starts = [
        np.zeros(3),
        np.linalg.lstsq(X, y, rcond=None)[0],  # OLS, a neutral start
        b_irls,                                 # and a refinement of the answer
    ]
    best = None
    for start in starts:
        for method, opts in (
            ("Nelder-Mead", dict(maxiter=200000, maxfev=200000,
                                 xatol=1e-12, fatol=1e-12)),
            ("Powell", dict(maxiter=200000, maxfev=200000, xtol=1e-12, ftol=1e-12)),
        ):
            r = minimize(_check_loss, start, args=(y, X, q), method=method,
                         options=opts)
            if best is None or r.fun < best.fun:
                best = r

    obj_irls = _check_loss(b_irls, y, X, q)
    # IRLS may sit fractionally *above* the true minimum — its residual floor
    # keeps it off the exact vertex — but it must not sit meaningfully above,
    # and it must not sit below (that would mean the oracle failed).
    assert obj_irls - best.fun == pytest.approx(0.0, abs=5e-6)
    assert obj_irls >= best.fun - 1e-9
    # Measured: 8.7e-7 in objective, 5.9e-5 in coefficients.
    assert np.max(np.abs(b_irls - best.x)) < 1e-3

    # The oracle has to be able to tell a wrong answer from a right one, or
    # the assertions above are decoration. OLS is a wrong answer for the
    # median of an asymmetric-residual design, and the objective says so.
    assert _check_loss(starts[1], y, X, q) > best.fun + 1e-3


def test_extreme_quantile_first_order_condition():
    """A property no transcription can fake: Σ xᵢ(q - 1{rᵢ<0}) ≈ 0.

    This is the Koenker-Bassett normal equation. It is checked at q=0.9 —
    the quantile the corpus actually fits — because a wrongly-oriented
    weighting scheme (q swapped for 1-q, an easy transcription slip that the
    statsmodels source invites) would fit the 0.1 quantile instead and this
    would catch it where a parity test against the same slip would not.
    """
    y, X = _design(n=400, seed=99)
    for q in (0.1, 0.9):
        b = np.asarray(quantreg(y, X, q=q).params, float)
        r = y - X @ b
        score = X.T @ (q - (r < 0).astype(float))
        # k residuals are ~0 at the optimum and each contributes O(1) to one
        # component of the score; with n=400 that is the resolution available.
        assert np.max(np.abs(score)) < 2.0
        assert np.mean(r < 0) == pytest.approx(q, abs=0.02)


# ---------------------------------------------------------------------------
# 3. Cross-check against the other quantile fitter puremacro ships
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q", QUANTILES)
def test_gap_against_lp_simplex_is_documented_not_asserted_away(q):
    """``_qreg`` solves the exact LP; ``quantreg`` runs statsmodels' IRLS.

    They are different estimators of the same estimand and they do not agree
    to machine precision. What must hold is the ordering: the LP is the exact
    optimiser, so its check loss is never worse, and the excess IRLS pays for
    its ``1e-6`` residual floor is negligible next to any standard error.

    Measured on this design (n=120, k=3): coefficients differ by 7e-7 (q=0.75)
    to 9.3e-5 (q=0.1); the check-loss excess is 7e-7 to 2.1e-6 absolute, which
    is 1.8e-8 to 9.4e-8 *relative* to the loss itself. For comparison, the
    smallest standard error on this design is O(0.05), so the estimator gap is
    three to four orders of magnitude below the sampling noise.
    """
    from puremacro.lp.quantile import _qreg

    rng = np.random.default_rng(20260906)
    n = 120
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.5 + 1.2 * x1 - 0.8 * x2 + rng.standard_normal(n) * (1 + 0.3 * np.abs(x2))
    X = np.column_stack([np.ones(n), x1, x2])

    b_irls = np.asarray(quantreg(y, X, q=q).params, float)
    b_lp = _qreg(y, X, q)

    obj_irls = _check_loss(b_irls, y, X, q)
    obj_lp = _check_loss(b_lp, y, X, q)

    # The LP is exact, so it cannot be beaten by more than floating-point dust.
    assert obj_lp <= obj_irls + 1e-9
    # And IRLS pays a bounded, tiny price for its residual floor.
    assert (obj_irls - obj_lp) / obj_lp < 1e-6
    # The coefficient gap is real and larger than the objective gap suggests —
    # a flat objective near the optimum. This is the number a user porting a
    # call site between the two needs to know, so it is asserted, not ignored.
    assert np.max(np.abs(b_irls - b_lp)) < 1e-3
    assert np.max(np.abs(b_irls - b_lp)) > 1e-9, (
        "the two estimators agreed to machine precision, which they should "
        "not; has _qreg been reimplemented as IRLS?"
    )


def test_parity_tolerance_can_discriminate():
    """A positive control for PARAM_ATOL itself.

    An assertion at ``atol=1e-6`` is only meaningful if 1e-6 is small compared
    with the differences that matter. OLS and the 0.9-quantile fit are
    genuinely different estimators of this design; if PARAM_ATOL could not
    separate them, every parity test above would be vacuous.
    """
    y, X = _design()
    b_q90 = np.asarray(quantreg(y, X, q=0.9).params, float)
    b_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.max(np.abs(b_q90 - b_ols)) > 1e4 * PARAM_ATOL

    # And the 0.1 and 0.9 fits must differ too — a fitter that ignored q
    # would pass everything that only ever looks at one quantile.
    b_q10 = np.asarray(quantreg(y, X, q=0.1).params, float)
    assert np.max(np.abs(b_q90 - b_q10)) > 1e4 * PARAM_ATOL


# ---------------------------------------------------------------------------
# 4. The house contracts: diagnostic errors, input validation, Pyodide
# ---------------------------------------------------------------------------
def test_singular_design_raises_a_named_error():
    """statsmodels min-norms through ``pinv``; we refuse and name the column."""
    rng = np.random.default_rng(1)
    n = 50
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x, 2 * x])  # third column is 2× the second
    y = rng.normal(size=n)

    with pytest.raises(np.linalg.LinAlgError, match=r"quantreg: X'X is singular"):
        quantreg(y, X, q=0.5)


def test_singular_design_is_the_documented_divergence_from_statsmodels():
    """The refusal above is a divergence, so pin what statsmodels does instead."""
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    n = 50
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x, 2 * x])
    y = rng.normal(size=n)

    ref = sm.QuantReg(y, X).fit(q=0.5)
    # statsmodels returns a minimum-norm solution and reports rank 2 of 3;
    # nothing warns. That is the behaviour puremacro declines to reproduce.
    assert np.all(np.isfinite(ref.params))
    assert ref.df_resid == n - 2


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(q=0.0), "strictly between 0 and 1"),
        (dict(q=1.0), "strictly between 0 and 1"),
        (dict(q=-0.2), "strictly between 0 and 1"),
        (dict(kernel="triangle"), "kernel must be one of"),
        (dict(bandwidth="silverman"), "bandwidth must be one of"),
        (dict(bandwidth=0.0), "finite and strictly positive"),
        (dict(bandwidth=-1.0), "finite and strictly positive"),
        (dict(vcov="HC1"), "vcov must be"),
        (dict(missing="ignore"), "missing must be"),
    ],
)
def test_input_validation(kwargs, match):
    y, X = _design(n=60, seed=2)
    with pytest.raises(ValueError, match=match):
        quantreg(y, X, **kwargs)


def test_length_mismatch_is_refused():
    y, X = _design(n=60, seed=2)
    with pytest.raises(ValueError, match="rows"):
        quantreg(y[:50], X, q=0.5)


def test_missing_handling():
    y, X = _design(n=90, seed=5)
    y_nan = y.copy()
    y_nan[3] = np.nan
    X_nan = X.copy()
    X_nan[7, 1] = np.nan

    with pytest.raises(ValueError, match="missing='raise'"):
        quantreg(y_nan, X, q=0.5, missing="raise")

    dropped = quantreg(y_nan, X, q=0.5, missing="drop")
    assert dropped.nobs == 89.0
    both = quantreg(y_nan, X_nan, q=0.5, missing="drop")
    assert both.nobs == 88.0

    # missing='none' is statsmodels' default and does no filtering; a
    # non-finite X is still rejected, exactly where statsmodels rejects it.
    with pytest.raises(ValueError, match="inf or NaN"):
        quantreg(y, X_nan, q=0.5, missing="none")
    # A non-finite y under missing='none' gives NaN coefficients, as in
    # statsmodels — but not silently.
    with pytest.warns(UserWarning, match="non-finite values"):
        res = quantreg(y_nan, X, q=0.5, missing="none")
    assert np.all(np.isnan(np.asarray(res.params, float)))


def test_missing_drop_matches_statsmodels_drop():
    sm = pytest.importorskip("statsmodels.api")
    y, X = _design(n=90, seed=5)
    y_nan = y.copy()
    y_nan[3] = np.nan

    ref = sm.QuantReg(y_nan, X, missing="drop").fit(q=0.5)
    got = quantreg(y_nan, X, q=0.5, missing="drop")
    np.testing.assert_allclose(got.params, ref.params, atol=PARAM_ATOL, rtol=0)
    np.testing.assert_allclose(got.bse, ref.bse, atol=BSE_ATOL, rtol=0)


def test_fixed_bandwidth_overrides_the_rule():
    """A float replaces the *rule output* h₀, the argument of Φ⁻¹(q ± h₀)."""
    sm = pytest.importorskip("statsmodels.api")
    from puremacro.regress.quantile import hall_sheather

    y, X = _design()
    h0 = hall_sheather(float(len(y)), 0.5)

    ref = sm.QuantReg(y, X).fit(q=0.5)  # uses hall_sheather internally
    got = quantreg(y, X, q=0.5, bandwidth=h0)

    assert got.bandwidth_rule.startswith("fixed(")
    assert got.bandwidth == pytest.approx(ref.bandwidth, abs=1e-12)
    np.testing.assert_allclose(got.bse, ref.bse, atol=1e-12, rtol=0)

    # A different fixed bandwidth must actually change the standard errors,
    # or the argument is being ignored.
    other = quantreg(y, X, q=0.5, bandwidth=0.5 * h0)
    assert other.bandwidth != pytest.approx(got.bandwidth, abs=1e-6)


def test_no_constant_design_is_accepted():
    """df_model counts the constant the way statsmodels counts it."""
    sm = pytest.importorskip("statsmodels.api")
    y, X = _design()
    X_nc = X[:, 1:]  # drop the intercept

    ref = sm.QuantReg(y, X_nc).fit(q=0.5)
    got = quantreg(y, X_nc, q=0.5)
    assert got.df_model == ref.df_model == 2.0
    np.testing.assert_allclose(got.params, ref.params, atol=PARAM_ATOL, rtol=0)


def test_module_imports_without_statsmodels():
    """Pyodide contract, checked in a fresh interpreter.

    ``tests/test_pyodide_compat.py`` sweeps the whole package; this asserts it
    for this module specifically, so a regression is attributed here rather
    than to whichever module the sweep happens to reach first.
    """
    import subprocess
    import sys

    code = (
        "import sys; import puremacro.regress.quantile as m; "
        "assert 'statsmodels' not in sys.modules, sorted(sys.modules); "
        "assert callable(m.quantreg); print('clean')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout


def test_score_at_percentile_is_bit_identical_to_scipy():
    """The private percentile helper must be scipy's, bit for bit.

    ``QuantReg`` computes the residual interquartile range with
    ``scipy.stats.scoreatpercentile``, whose ``'fraction'`` interpolation is
    *mathematically* the same as ``np.percentile``'s default but takes a
    different floating-point path. The helper is a transcription of scipy's
    ``_compute_qth_percentile``; if someone replaces it with the "equivalent"
    ``np.percentile``, the parity tests above will not notice (the difference
    is 9e-16, far under their tolerance) but the claim of bit-exact agreement
    with statsmodels stops being true. This test notices.
    """
    scipy_stats = pytest.importorskip("scipy.stats")
    from puremacro.regress.quantile import _score_at_percentile

    rng = np.random.default_rng(7)
    n = int(rng.integers(5, 300))
    a = rng.standard_t(5, size=n) * 3.7 + 1.3

    # This is a case where np.percentile disagrees with scipy in the last bit;
    # it exists precisely so the test can tell the two implementations apart.
    assert np.percentile(a, 90.0) != scipy_stats.scoreatpercentile(a, 90.0)
    assert _score_at_percentile(a, 90.0) == float(
        scipy_stats.scoreatpercentile(a, 90.0)
    )

    for per in (0.0, 10.0, 25.0, 33.3, 50.0, 75.0, 90.0, 100.0):
        assert _score_at_percentile(a, per) == float(
            scipy_stats.scoreatpercentile(a, per)
        ), per


def test_reported_bandwidth_uses_the_scipy_percentile_path():
    """The helper above is only useful if ``quantreg`` actually calls it.

    Testing the helper in isolation would leave the call site free to drift to
    ``np.percentile``, so this rebuilds the reported bandwidth from the fit's
    *own* residuals — which makes it insensitive to BLAS differences across
    platforms, unlike an ``atol=0`` comparison against statsmodels — and
    demands exact equality. The design is chosen so the two percentile paths
    genuinely disagree, and so the residual scale (not ``std(y)``) is the
    binding term in the ``min``; otherwise the assertion would be vacuous.
    """
    scipy_stats = pytest.importorskip("scipy.stats")
    from scipy.stats import norm

    from puremacro.regress.quantile import hall_sheather

    q, n = 0.9, 200
    y, X = _design(n=n, seed=1)
    res = quantreg(y, X, q=q)
    e = np.asarray(res.resid, float)

    iqre_scipy = scipy_stats.scoreatpercentile(e, 75) - scipy_stats.scoreatpercentile(e, 25)
    iqre_numpy = np.percentile(e, 75) - np.percentile(e, 25)
    assert iqre_scipy != iqre_numpy, "fixture no longer separates the two paths"
    assert iqre_scipy / 1.34 < np.std(y), "std(y) would bind, hiding the IQR"

    h0 = hall_sheather(res.nobs, q)
    scale = norm.ppf(q + h0) - norm.ppf(q - h0)
    assert res.bandwidth == min(np.std(y), iqre_scipy / 1.34) * scale
    assert res.bandwidth != min(np.std(y), iqre_numpy / 1.34) * scale


def test_result_is_frozen_and_documented():
    """The result object is a value, not a mutable scratchpad."""
    import dataclasses

    y, X = _design(n=80, seed=4)
    res = quantreg(y, X, q=0.5)
    assert isinstance(res, QuantRegResult)
    assert dataclasses.is_dataclass(res)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.q = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 5. Regressions for the defects found by adversarial review
#
# Every test in this section was written against the pre-fix module and
# watched to fail there before the fix landed (CONTRIBUTING, "Making sure a
# test can fail", habit 1). The failure each one produced is named in its
# docstring, so a future reader can tell what it is guarding rather than
# guessing from the assertion.
# ---------------------------------------------------------------------------
def _year_polynomial(n_years=71, seed=3):
    """``[1, t, t²]`` on raw calendar years — the archetypal macro trend.

    ``cond(X'X) = 1.7e21``. Nothing about it is exotic: it is what you get
    from ``add_constant(np.column_stack([year, year**2]))``, and the corpus
    reads coefficients off designs of exactly this shape.
    """
    t = np.arange(1950, 1950 + n_years, dtype=float)
    X = np.column_stack([np.ones(t.size), t, t ** 2])
    y = (
        2.0
        + 0.01 * (t - 1985)
        + 5e-4 * (t - 1985) ** 2
        + np.random.default_rng(seed).standard_t(5, size=t.size)
    )
    return t, y, X


def test_ill_conditioned_design_is_refused_not_silently_estimated():
    """A raw calendar-year polynomial used to sail through the gate.

    Pre-fix behaviour: accepted, no warning, ``params[2] = 4.75e-06`` against
    the 2.66e-05 the same model gives on a centred design — 82% wrong — with
    a standard error 108 times too small, and a check loss *worse* than the
    centred fit's (34.61343 vs 34.60600), so it was not even a competing
    optimum. That is silent garbage in the one place the module claims to
    beat statsmodels.

    The refusal is a divergence from statsmodels and is asserted as one in
    ``test_ill_conditioned_design_is_the_documented_divergence`` below.
    """
    _, y, X = _year_polynomial()

    with pytest.raises(np.linalg.LinAlgError, match=r"numerically singular"):
        quantreg(y, X, q=0.5)

    # The message has to be actionable, or the refusal just moves the problem.
    with pytest.raises(np.linalg.LinAlgError, match=r"[Cc]entre or rescale"):
        quantreg(y, X, q=0.5)


def test_the_conditioning_gate_is_the_one_that_bites():
    """Positive control: ``inv_xtx`` alone does *not* catch this design.

    Without this assertion the test above would pass just as happily if the
    new gate were deleted and the old one had somehow started working, and a
    future reader would not know which of the two is load-bearing. The
    Cholesky-pivot ratio is 2.7e-3 — three orders *inside* the 1e-6 threshold
    ``inv_xtx`` applies — while the condition number is seven orders past the
    1e14 the same helper's error message advertises.
    """
    from puremacro._linalg import inv_xtx
    from puremacro.regress.quantile import _COND_MAX, _reject_ill_conditioned

    _, _, X = _year_polynomial()

    inv_xtx(X, name="control")  # must NOT raise: this is the gap being closed

    L = np.linalg.cholesky(X.T @ X)
    diag = np.abs(np.diag(L))
    assert diag.min() / diag.max() > 1e-6  # inv_xtx's own test, passed

    sv = np.linalg.svd(X, compute_uv=False)
    assert (sv.max() / sv.min()) ** 2 > 1e7 * _COND_MAX

    with pytest.raises(np.linalg.LinAlgError):
        _reject_ill_conditioned(X, ("const", "x1", "x2"))


def test_ill_conditioned_design_is_the_documented_divergence():
    """What statsmodels does instead, and why refusing is the better answer.

    statsmodels accepts, and the coefficients it returns are not merely
    imprecise — they minimise the check function *less well* than the fit on
    the centred design, which is the same model in different coordinates.
    """
    sm = pytest.importorskip("statsmodels.api")
    t, y, X = _year_polynomial()

    ref = sm.QuantReg(y, X).fit(q=0.5)
    assert np.all(np.isfinite(ref.params))  # accepted, silently

    c = t - t.mean()
    Xc = np.column_stack([np.ones(c.size), c, c ** 2])
    centred = quantreg(y, Xc, q=0.5)  # the same model, and this one is fine

    # Both parameterisations describe the same fitted quadratic, so the check
    # loss is directly comparable. The raw fit is the worse optimum.
    loss_raw = _check_loss(np.asarray(ref.params, float), y, X, 0.5)
    loss_centred = _check_loss(np.asarray(centred.params, float), y, Xc, 0.5)
    assert loss_raw > loss_centred + 1e-4

    # And the quadratic coefficient — the one a call site would quote — is
    # out by tens of percent, with a standard error two orders too tight.
    b2_raw = float(np.asarray(ref.params)[2])
    b2_centred = float(np.asarray(centred.params)[2])
    assert abs(b2_raw - b2_centred) / abs(b2_centred) > 0.5
    assert float(np.asarray(centred.bse)[2]) > 50 * float(np.asarray(ref.bse)[2])


@pytest.mark.parametrize("q", (0.25, 0.5, 0.9))
def test_conditioning_gate_accepts_ordinary_designs(q):
    """The gate must not fire on anything a working notebook would pass it.

    A gate that refuses good designs is as useless as one that accepts bad
    ones, and this is the assertion that would go red if the threshold were
    tightened carelessly. Three ordinary shapes: a plain cross-section, a
    centred trend polynomial, and a design whose columns differ by six orders
    of magnitude in scale but are perfectly well identified.
    """
    sm = pytest.importorskip("statsmodels.api")
    y, X = _design()
    np.testing.assert_allclose(
        quantreg(y, X, q=q).params, sm.QuantReg(y, X).fit(q=q).params,
        atol=PARAM_ATOL, rtol=0,
    )

    t, yp, _ = _year_polynomial()
    c = t - t.mean()
    Xc = np.column_stack([np.ones(c.size), c, c ** 2])
    assert np.linalg.cond(Xc.T @ Xc) < 1e14
    np.testing.assert_allclose(
        quantreg(yp, Xc, q=q).params, sm.QuantReg(yp, Xc).fit(q=q).params,
        atol=PARAM_ATOL, rtol=0,
    )


def test_misaligned_pandas_indices_are_refused_not_positionally_joined():
    """Pre-fix: the two objects were zipped by position, silently.

    With ``dfX`` on labels r0…r199 and ``y`` the same 200 values under the
    same labels in a shuffled order, the module returned a slope of -0.102
    where the label join gives 1.026 — sign-flipped and economically
    meaningless — and handed back a residual series wearing X's labels over
    y's values. statsmodels refuses, and so should this.
    """
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=n)
    idx = [f"r{i}" for i in range(n)]
    dfX = pd.DataFrame({"const": 1.0, "x": x}, index=idx)
    y = pd.Series(2.0 + x + rng.normal(size=n), index=idx)

    shuffled = y.sample(frac=1.0, random_state=0)
    with pytest.raises(ValueError, match="not aligned"):
        quantreg(shuffled, dfX, q=0.5)

    # A disjoint index is refused too, and says something different.
    with pytest.raises(ValueError, match=r"absent from X"):
        quantreg(y.rename(index=lambda s: s + "z"), dfX, q=0.5)

    # Aligned input is unaffected, and re-aligning is the documented remedy.
    got = quantreg(shuffled.reindex(dfX.index), dfX, q=0.5)
    assert float(got.params["x"]) == pytest.approx(1.026, abs=5e-3)
    assert got.resid.index.equals(dfX.index)


def test_misaligned_indices_refusal_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    n = 50
    x = rng.normal(size=n)
    idx = [f"r{i}" for i in range(n)]
    dfX = pd.DataFrame({"const": 1.0, "x": x}, index=idx)
    y = pd.Series(2.0 + x + rng.normal(size=n), index=idx).sample(
        frac=1.0, random_state=0
    )

    with pytest.raises(ValueError, match="not aligned"):
        sm.QuantReg(y, dfX)
    with pytest.raises(ValueError, match="not aligned"):
        quantreg(y, dfX, q=0.5)


def test_pandas_y_with_ndarray_design_returns_named_series():
    """The container rule keys off *either* input, as statsmodels' does.

    ``X = np.column_stack([...])`` with ``y = df["col"]`` is the commonest
    shape a ported call site has, and statsmodels returns a Series indexed
    ``['const', 'x1', …]`` for it (``handle_data_class_factory`` selects
    ``PandasData`` when either argument is pandas). Pre-fix this module keyed
    off ``X`` alone, returned bare ndarrays, and ``params['const']`` died with
    ``IndexError``.
    """
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    y = pd.Series(2.0 + x + rng.normal(size=n), name="y",
                  index=[f"u{i}" for i in range(n)])

    ref = sm.QuantReg(y, X).fit(q=0.9)
    got = quantreg(y, X, q=0.9)

    assert isinstance(got.params, pd.Series)
    assert list(got.params.index) == list(ref.params.index) == ["const", "x1"]
    assert float(got.params["const"]) == pytest.approx(
        float(ref.params["const"]), abs=PARAM_ATOL
    )
    for name in ("bse", "tvalues", "pvalues"):
        assert isinstance(getattr(got, name), pd.Series), name
        assert list(getattr(got, name).index) == ["const", "x1"], name
    # resid and fittedvalues take y's labels, because X has none to give.
    assert isinstance(got.resid, pd.Series)
    assert got.resid.index.equals(y.index)
    assert got.fittedvalues.index.equals(y.index)
    assert ref.resid.index.equals(y.index)
    assert isinstance(got.cov_params(), pd.DataFrame)
    assert list(got.conf_int().index) == ["const", "x1"]

    # The converse still holds: neither input pandas, neither output pandas.
    plain = quantreg(np.asarray(y), X, q=0.9)
    assert isinstance(plain.params, np.ndarray)
    assert isinstance(sm.QuantReg(np.asarray(y), X).fit(q=0.9).params, np.ndarray)


def test_non_string_column_labels_are_preserved():
    """Pre-fix the labels went through ``str()``, so ``params[1]`` KeyError'd.

    ``pd.DataFrame(np.column_stack([...]))`` gives integer labels, which
    statsmodels keeps as integers (``ModelData._get_names`` returns
    ``list(df.columns)`` untouched). A MultiIndex is the one thing it does
    rewrite, joining levels with an underscore.
    """
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(9)
    n = 200
    x = rng.normal(size=n)
    y = 1.0 + x + rng.normal(size=n)

    df = pd.DataFrame(np.column_stack([np.ones(n), x, x ** 2]))
    ref = sm.QuantReg(y, df).fit(q=0.5)
    got = quantreg(y, df, q=0.5)
    assert list(got.params.index) == list(ref.params.index) == [0, 1, 2]
    assert float(got.params[1]) == pytest.approx(float(ref.params[1]), abs=PARAM_ATOL)
    assert got.exog_names == (0, 1, 2)
    # summary() has to survive a label that is not a string.
    assert "QuantReg" in got.summary()

    dfm = pd.DataFrame(
        np.column_stack([np.ones(n), x]),
        columns=pd.MultiIndex.from_tuples([("lev", "const"), ("lev", "x")]),
    )
    refm = sm.QuantReg(y, dfm).fit(q=0.5)
    gotm = quantreg(y, dfm, q=0.5)
    assert list(gotm.params.index) == list(refm.params.index) == ["lev_const", "lev_x"]

    # A Series design contributes its name — but only a truthy one, which is
    # statsmodels' own ``if arr.name:`` and not a mistake being copied.
    named = pd.Series(x, name="gdp")
    assert quantreg(y, named, q=0.5).exog_names == ("gdp",)
    assert tuple(sm.QuantReg(y, named).fit(q=0.5).model.exog_names) == ("gdp",)
    zero_named = pd.Series(x, name=0)
    assert quantreg(y, zero_named, q=0.5).exog_names == ("x1",)
    assert tuple(sm.QuantReg(y, zero_named).fit(q=0.5).model.exog_names) == ("x1",)


def test_inf_is_not_missing_under_missing_drop():
    """``missing='drop'`` drops NaN rows only, as ``_nan_rows`` does.

    Pre-fix, an inf was treated as missing: a y holding one infinity was
    fitted on 199 rows and returned a plausible finite answer where
    statsmodels reports ``nobs=200`` and ``params=[nan, nan]``, and an inf in
    X likewise produced a finite fit where statsmodels raises. A quietly
    moved number is the thing PARITY_SPEC §7 forbids.
    """
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    y = 2.0 + x + rng.normal(size=n)

    y_inf = y.copy()
    y_inf[3] = np.inf
    ref = sm.QuantReg(y_inf, X, missing="drop").fit(q=0.5)
    with pytest.warns(UserWarning, match="non-finite values"):
        got = quantreg(y_inf, X, q=0.5, missing="drop")
    assert got.nobs == ref.nobs == 200.0
    assert np.all(np.isnan(np.asarray(got.params, float)))
    assert np.all(np.isnan(np.asarray(ref.params, float)))

    X_inf = X.copy()
    X_inf[5, 1] = np.inf
    with pytest.raises(ValueError, match="inf or NaN"):
        quantreg(y, X_inf, q=0.5, missing="drop")
    with pytest.raises(Exception, match="inf or nans"):
        sm.QuantReg(y, X_inf, missing="drop").fit(q=0.5)

    # NaN is still missing, and still dropped, and 'raise' still says so.
    y_nan = y.copy()
    y_nan[3] = np.nan
    dropped = quantreg(y_nan, X, q=0.5, missing="drop")
    assert dropped.nobs == 199.0
    np.testing.assert_allclose(
        dropped.params, sm.QuantReg(y_nan, X, missing="drop").fit(q=0.5).params,
        atol=PARAM_ATOL, rtol=0,
    )
    with pytest.raises(ValueError, match="NaN in y or X"):
        quantreg(y_nan, X, q=0.5, missing="raise")


def test_zero_column_design_gets_a_named_refusal():
    """Degenerate, but it must not surface as somebody else's error message.

    Pre-fix: a bare ``ValueError: zero-size array to reduction operation
    minimum which has no identity``, raised from inside ``_linalg`` and
    naming neither the caller nor the problem.
    """
    with pytest.raises(ValueError, match="quantreg: X has no columns"):
        quantreg(np.zeros(10), np.zeros((10, 0)), q=0.5)
    with pytest.raises(ValueError, match="quantreg: X has no columns"):
        quantreg(pd.Series(np.zeros(10)), pd.DataFrame(index=range(10)), q=0.5)


def test_vcov_is_read_only_so_the_result_is_actually_frozen():
    """``.vcov`` and ``.cov_params()`` cannot be driven out of step with ``bse``.

    Pre-fix, ``res.vcov[0, 0] = 1e6`` left ``cov_params()[0, 0] == 1e6`` while
    ``bse[0]`` still reported the old 0.229 — a frozen dataclass wrapping a
    live array.
    """
    y, X = _design(n=80, seed=4)
    res = quantreg(y, X, q=0.5)

    assert res.vcov.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        res.vcov[0, 0] = 1e6
    assert np.asarray(res.cov_params())[0, 0] == pytest.approx(
        float(np.asarray(res.bse)[0]) ** 2, rel=1e-12
    )

    # The pandas view is a copy, so labelling a covariance does not hand out
    # a writable window onto the frozen one.
    df_res = quantreg(y, pd.DataFrame(X, columns=["const", "x1", "x2"]), q=0.5)
    cov = df_res.cov_params()
    cov.iloc[0, 0] = 1e6
    assert df_res.vcov[0, 0] != 1e6


# ---------------------------------------------------------------------------
# 6. Branches the earlier mutation survey could not reach
# ---------------------------------------------------------------------------
def _tied_integer_design():
    """A design that produces residuals equal to *exactly* 0.0.

    The continuous fixture ``_design`` cannot: IRLS floors residual
    magnitudes at 1e-6, and from ``rng.normal`` data no residual ever lands on
    zero. Integer data does it two ways at once — rows where the regressor is
    0 and the response is 0, and rows the fit passes through exactly once β
    settles on the integer 2.0 — which is the ordinary situation for a
    no-intercept regression of a count on an integer exposure.
    """
    rng = np.random.default_rng(179)
    n = 60
    x = rng.integers(0, 4, size=n).astype(float)
    y = np.round(1.5 * x + rng.standard_t(3, size=n))
    return y, x.reshape(-1, 1)


def test_robust_sandwich_tie_convention_on_exact_zero_residuals():
    """``np.where(e > 0, …)`` — strict — is statsmodels', and it matters.

    This is the branch the previous mutation survey listed as "semantically
    inert": ``e > 0`` → ``e >= 0`` left all 63 tests green. It is not inert,
    only unreached — no continuous fixture produces a residual of exactly
    0.0. On this design thirteen of sixty residuals are exactly zero, the two
    conventions move ``bse`` by 80%, and the strict one is bit-identical to
    statsmodels.
    """
    sm = pytest.importorskip("statsmodels.api")
    y, X = _tied_integer_design()
    q = 0.9

    got = quantreg(y, X, q=q)
    e = np.asarray(got.resid, float)
    # The fixture must be able to produce the condition it is testing
    # (CONTRIBUTING, habit 4). If a future edit to the IRLS floor stops the
    # residuals landing on zero, this says so instead of passing vacuously.
    assert int(np.sum(e == 0.0)) == 13

    ref = sm.QuantReg(y, X).fit(q=q)
    np.testing.assert_allclose(got.bse, ref.bse, atol=0.0, rtol=0.0)

    # Rebuild the sandwich both ways from the fit's own pieces. This shares
    # no code with the module beyond the reported sparsity, so it is a check
    # on the convention rather than a restatement of it.
    fhat0 = 1.0 / got.sparsity
    xtxi = np.linalg.pinv(X.T @ X)

    def _bse(mask):
        d = np.where(mask, (q / fhat0) ** 2, ((1 - q) / fhat0) ** 2)
        return np.sqrt(np.diag(xtxi @ (X.T * d) @ X @ xtxi))

    strict, inclusive = _bse(e > 0), _bse(e >= 0)
    np.testing.assert_allclose(got.bse, strict, atol=0.0, rtol=0.0)
    assert np.max(np.abs(strict - inclusive)) > 0.07  # measured: 0.0737
    assert float(inclusive[0]) / float(strict[0]) == pytest.approx(1.796, abs=1e-3)


def test_df_resid_uses_rank_and_the_two_spellings_cannot_diverge():
    """``df_resid = nobs - rank`` vs ``nobs - k``: an invariant, not a choice.

    The mutation survey flagged ``nobs - exog.shape[1]`` as a surviving
    mutant. It survives because it cannot differ: ``matrix_rank``'s tolerance
    declares a design deficient only past ``cond(X) ≈ 1e13``, i.e.
    ``cond(X'X) ≈ 1e26``, and the conditioning gate refuses everything above
    1e14 — twelve orders earlier. So on every input that reaches the line the
    two expressions are equal, and this test pins both halves of that
    argument: full rank on what is accepted, refusal on what is not.
    """
    y, X = _design()
    res = quantreg(y, X, q=0.5)
    assert res.rank == X.shape[1]
    assert res.df_resid == res.nobs - res.rank == res.nobs - X.shape[1]

    # The only way to make the two spellings differ is a design the gate
    # refuses outright.
    rng = np.random.default_rng(1)
    x = rng.normal(size=50)
    deficient = np.column_stack([np.ones(50), x, 2 * x])
    assert np.linalg.matrix_rank(deficient) < deficient.shape[1]
    with pytest.raises(np.linalg.LinAlgError):
        quantreg(rng.normal(size=50), deficient, q=0.5)


def test_badly_scaled_design_is_refused_and_statsmodels_is_wrong_there():
    """A regressor in levels beside an intercept: refused, and rightly.

    ``cond(X'X)`` is 1.1e16 on the raw columns and 3.9 once they are scaled to
    unit norm, so the design is well *identified* and only badly *scaled*.
    ``puremacro.regress.ols`` equilibrates and therefore accepts it; this
    module cannot, because bit-parity means running statsmodels' IRLS — whose
    residual floor is 1e-6 in the units of the raw data — on the array the
    caller passed. So the refusal is the honest answer here, and the test
    exists to record that statsmodels' alternative is not a working one: it
    returns an intercept of 5e-16 for a true 0.86 at a check loss 20% worse
    than the identical model on scaled columns.
    """
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(4)
    n = 200
    pop = np.exp(rng.normal(17.0, 1.2, size=n))  # population in persons
    X = np.column_stack([np.ones(n), pop])
    y = 1.0 + 3e-8 * pop + rng.standard_t(5, size=n)

    with pytest.raises(np.linalg.LinAlgError):
        quantreg(y, X, q=0.5)

    col_norm = np.sqrt((X ** 2).sum(0))
    Xs = X / col_norm
    assert np.linalg.cond(Xs.T @ Xs) < 10.0  # well identified, merely mis-scaled

    # The remedy the docstring recommends: scale, fit, divide back out.
    scaled = quantreg(y, Xs, q=0.5)
    b_scaled = np.asarray(scaled.params, float) / col_norm
    np.testing.assert_allclose(
        b_scaled, np.asarray(sm.QuantReg(y, Xs).fit(q=0.5).params) / col_norm,
        atol=PARAM_ATOL, rtol=0,
    )

    b_raw = np.asarray(sm.QuantReg(y, X).fit(q=0.5).params, float)
    assert abs(b_raw[0]) < 1e-10  # the intercept statsmodels reports for 0.86
    assert 0.5 < b_scaled[0] < 1.5
    assert (
        _check_loss(b_raw, y, X, 0.5) > 1.15 * _check_loss(b_scaled, y, X, 0.5)
    )
