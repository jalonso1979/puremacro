"""Failure-path and API-fidelity regressions for :mod:`puremacro.regress.ols`.

``test_ols_parity.py`` measures agreement with statsmodels on designs that
are *well posed*. This file covers the other half of the contract: the
inputs where the honest answer is an exception rather than a number, and
the statsmodels spellings a ported call site is allowed to keep.

Every test here was written against the pre-fix module and watched to
fail before the fix landed; the three marked "control" are the positive
controls CONTRIBUTING asks for, and pass in both trees on purpose — they
exist so that a guard cannot be widened into uselessness without a test
going red.

statsmodels is a dev-only dependency (ARCHITECTURE.md), so every test
that consults it asks through ``pytest.importorskip``.
"""
import numpy as np
import pandas as pd
import pytest

from puremacro.regress.ols import add_constant, ols

ATOL = 1e-10


def _sm():
    return pytest.importorskip("statsmodels.api")


# ---------------------------------------------------------------------------
# designs
# ---------------------------------------------------------------------------

def _outlier_dummy(n=40, seed=0):
    """``[1, x, e_7]`` — the dummy makes row 7 leverage exactly 1."""
    rng = np.random.default_rng(seed)
    d = np.zeros(n)
    d[7] = 1.0
    X = np.column_stack([np.ones(n), rng.standard_normal(n), d])
    y = X @ np.array([1.0, 0.5, 2.0]) + rng.standard_normal(n)
    return y, X


def _unbalanced_panel_with_singleton(seed=3):
    """Unit dummies on sizes [6, 5, 4, 1]: the last unit is a singleton."""
    rng = np.random.default_rng(seed)
    sizes = [6, 5, 4, 1]
    unit = np.concatenate([np.full(s, g) for g, s in enumerate(sizes)])
    n = unit.size
    x = rng.standard_normal(n)
    D = np.column_stack([(unit == j).astype(float) for j in range(1, 4)])
    X = np.column_stack([np.ones(n), x, D])
    y = X @ np.array([1.0, 0.7, -0.3, 0.4, 2.0]) + rng.standard_normal(n)
    return y, X, unit


def _levels_design(n=300, scale=1e6, seed=7):
    """Full rank, no collinearity, one regressor in levels."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n)
    z = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), scale * base, z])
    y = 1.0 + 0.3 * base + 0.5 * z + rng.standard_normal(n)
    return y, X


def _small_panel(n_units=6, n_periods=10, seed=17):
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)
    n = unit.size
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x1, x2])
    y = X @ np.array([0.5, 1.1, -0.4]) + rng.standard_normal(n)
    return y, X, unit, time


def _plain(n=60, seed=909):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal(n),
                         rng.standard_normal(n)])
    y = X @ np.array([0.4, 1.2, -0.6]) + rng.standard_normal(n)
    return y, X


# ---------------------------------------------------------------------------
# HC2 / HC3 at leverage 1
# ---------------------------------------------------------------------------

def test_hc2_hc3_refuse_a_leverage_one_row():
    """A singleton dummy is 0/0 for the leave-one-out scaling.

    Unfixed, this returned ``bse = [nan, nan, nan]`` — the NaN reaches
    every coefficient, not only the dummy's, and travels on into
    ``tvalues``, ``pvalues``, ``conf_int`` and ``f_test``.
    """
    y, X = _outlier_dummy()
    for kind in ("HC2", "HC3"):
        with pytest.raises(np.linalg.LinAlgError) as exc:
            ols(y, X, cov_type=kind)
        msg = str(exc.value)
        assert kind in msg
        assert "leverage is 1" in msg
        assert "[7]" in msg              # names the offending row
        assert "HC0" in msg              # and the way out


def test_hc0_hc1_are_untouched_by_a_leverage_one_row():
    """The guard is confined to the two estimators that divide by 1 - h."""
    sm = _sm()
    y, X = _outlier_dummy()
    for kind in ("HC0", "HC1"):
        ref = sm.OLS(y, X).fit(cov_type=kind)
        got = ols(y, X, cov_type=kind)
        np.testing.assert_allclose(np.asarray(got.bse), np.asarray(ref.bse),
                                   rtol=0, atol=ATOL)
        assert np.isfinite(np.asarray(got.bse)).all()


def test_unbalanced_panel_singleton_unit_is_diagnosed_not_nan():
    """The corpus-shaped case: unit dummies with one singleton unit."""
    sm = _sm()
    y, X, _ = _unbalanced_panel_with_singleton()
    # HC0/HC1 are the estimators that still have an answer here.
    ref = sm.OLS(y, X).fit(cov_type="HC1")
    got = ols(y, X, cov_type="HC1")
    np.testing.assert_allclose(np.asarray(got.bse), np.asarray(ref.bse),
                               rtol=0, atol=1e-12)
    for kind in ("HC2", "HC3"):
        with pytest.raises(np.linalg.LinAlgError, match="leverage is 1"):
            ols(y, X, cov_type=kind)


def test_high_but_not_unit_leverage_still_fits(request):
    """Control: the guard must not swallow a merely influential row.

    Passes against the unfixed module too — that is the point. If the
    leverage tolerance is ever widened, this goes red.
    """
    sm = _sm()
    y, X = _outlier_dummy()
    X = X.copy()
    X[8, 2] = 0.01            # row 7 no longer alone on the dummy
    h_max = np.max(np.einsum("ij,jk,ik->i", X,
                             np.linalg.inv(X.T @ X), X))
    assert 0.999 < h_max < 1.0            # the fixture really is extreme
    ref = sm.OLS(y, X).fit(cov_type="HC3")
    got = ols(y, X, cov_type="HC3")
    np.testing.assert_allclose(np.asarray(got.bse), np.asarray(ref.bse),
                               rtol=0, atol=1e-8)


def test_the_leverage_tolerance_sits_where_the_precision_runs_out():
    """Pin ``_LEVERAGE_TOL`` to the arithmetic, not to a taste.

    A second observation with a tiny value on the dummy pushes ``1 - h``
    off zero by a controllable amount. Just above the tolerance the
    statsmodels agreement is still worth having; just below it, the
    quotient is rounding error and the fit is refused.
    """
    sm = _sm()
    rng = np.random.default_rng(0)
    n = 40

    def design(second):
        d = np.zeros(n)
        d[7] = 1.0
        d[8] = second
        X = np.column_stack([np.ones(n), rng.standard_normal(n), d])
        return X @ np.array([1.0, 0.5, 2.0]) + rng.standard_normal(n), X

    y, X = design(1e-5)                     # 1 - h ~ 1e-10, above the gate
    got = np.asarray(ols(y, X, cov_type="HC3").bse)
    ref = np.asarray(sm.OLS(y, X).fit(cov_type="HC3").bse)
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=0)

    y, X = design(1e-6)                     # 1 - h ~ 1e-12, below the gate
    with pytest.raises(np.linalg.LinAlgError, match="leverage is 1"):
        ols(y, X, cov_type="HC3")


# ---------------------------------------------------------------------------
# the singularity gate is about collinearity, not units
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [1e0, 1e3, 1e6, 1e9, 1e12])
def test_column_scale_does_not_decide_acceptance(scale):
    """A regressor in levels beside an intercept is a well-posed design.

    Unfixed, ``inv_xtx``'s pivot-ratio gate fired on the column-norm
    disparity rather than on collinearity, so ``scale >= 1e6`` raised
    ``LinAlgError`` on a design statsmodels fits without complaint.
    """
    sm = _sm()
    y, X = _levels_design(scale=scale)
    ref = sm.OLS(y, X).fit(cov_type="HC1")
    got = ols(y, X, cov_type="HC1")
    # rtol rather than atol: the coefficient on a 1e12-scaled column is
    # itself 1e-12, and statsmodels' own pinv loses a few digits on the
    # un-equilibrated design, so 1e-8 relative is the honest bar here.
    np.testing.assert_allclose(np.asarray(got.params),
                               np.asarray(ref.params),
                               rtol=1e-8, atol=0)
    np.testing.assert_allclose(np.asarray(got.bse), np.asarray(ref.bse),
                               rtol=1e-8, atol=0)


def test_rescaling_a_column_moves_no_inference():
    """Scale invariance, stated as a property rather than a threshold."""
    y, X = _levels_design(scale=1.0)
    base = ols(y, X, cov_type="HC1")
    for scale in (1e-8, 1e3, 1e6, 1e11):
        Xs = X.copy()
        Xs[:, 1] = Xs[:, 1] * scale
        got = ols(y, Xs, cov_type="HC1")
        np.testing.assert_allclose(np.asarray(got.tvalues),
                                   np.asarray(base.tvalues),
                                   rtol=1e-9, atol=0)
        np.testing.assert_allclose(np.asarray(got.params)[1] * scale,
                                   np.asarray(base.params)[1],
                                   rtol=1e-9, atol=0)


def test_collinear_design_still_raises_a_named_error():
    """Control: equilibration must not disarm the collinearity gate."""
    rng = np.random.default_rng(8)
    x = rng.standard_normal(40)
    X = np.column_stack([np.ones(40), x, 2.0 * x])
    y = rng.standard_normal(40)
    with pytest.raises(np.linalg.LinAlgError) as exc:
        ols(y, X)
    assert "null space" in str(exc.value)


def test_ill_conditioned_message_does_not_advise_rescaling():
    """At full rank but scale-free ill-conditioning, say what is wrong.

    ``inv_xtx``'s own message ends "Try rescaling regressors", which is
    spent advice once ``ols`` has already scaled every column to unit
    norm.
    """
    t = np.arange(1900.0, 2020.0)
    X = np.column_stack([t ** j for j in range(6)])   # degree-5 year poly
    y = np.random.default_rng(1).standard_normal(t.size)
    with pytest.raises(np.linalg.LinAlgError) as exc:
        ols(y, X)
    msg = str(exc.value)
    assert "unit norm" in msg
    assert "rescaling will not help" in msg


# ---------------------------------------------------------------------------
# pandas alignment
# ---------------------------------------------------------------------------

def test_misaligned_pandas_indexes_are_refused_like_statsmodels():
    """Same labels, permuted order: positional pairing is a wrong number."""
    sm = _sm()
    y, X = _plain()
    idx = pd.Index([f"r{i}" for i in range(len(y))])
    Xd = pd.DataFrame(X, columns=["const", "x1", "x2"], index=idx)
    ys = pd.Series(y, index=idx).sort_values()

    with pytest.raises(ValueError, match="not aligned"):
        sm.OLS(ys, Xd).fit()
    with pytest.raises(ValueError, match="not aligned"):
        ols(ys, Xd)
    # An entirely different index object is refused too.
    with pytest.raises(ValueError, match="not aligned"):
        ols(pd.Series(y), Xd)
    # ... and the aligned call is untouched.
    aligned = ols(ys.reindex(idx), Xd)
    ref = sm.OLS(ys.reindex(idx), Xd).fit()
    np.testing.assert_allclose(np.asarray(aligned.params),
                               np.asarray(ref.params), rtol=0, atol=ATOL)


def test_alignment_check_needs_both_indexes():
    """Control: a pandas y with an ndarray X is still positional."""
    y, X = _plain()
    ys = pd.Series(y, index=[f"r{i}" for i in range(len(y))])
    res = ols(ys, X)
    assert np.isfinite(np.asarray(res.params)).all()
    assert list(res.resid.index) == list(ys.index)


# ---------------------------------------------------------------------------
# non-finite exog
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_exog_raises_under_missing_none(bad):
    """statsmodels raises MissingDataError here even with missing='none'.

    Unfixed this returned ``params = [nan, nan, nan]`` with ``nobs`` and
    ``df_resid`` attached and no exception: ``inv_xtx`` does not save it
    either, because ``np.linalg.cholesky`` accepts a NaN matrix and the
    pivot-ratio guard is ``False`` on NaN.
    """
    sm = _sm()
    y, X = _plain()
    X = X.copy()
    X[9, 1] = bad
    # statsmodels refuses too, though not always at the same place: its
    # check is `np.isfinite(exog.max(axis=0)).all()`, which a -inf slips
    # past, and the SVD then fails instead.
    with pytest.raises(Exception):
        sm.OLS(y, X).fit()
    with pytest.raises(ValueError) as exc:
        ols(y, X)
    assert "inf or nans" in str(exc.value)
    assert "[9]" in str(exc.value)


def test_non_finite_exog_is_still_droppable():
    """Control: the check runs after ``missing='drop'`` has done its work."""
    y, X = _plain()
    X = X.copy()
    X[9, 1] = np.nan
    dropped = ols(y, X, missing="drop")
    assert dropped.nobs == len(y) - 1
    assert np.isfinite(np.asarray(dropped.params)).all()


def test_nan_in_endog_is_still_silent():
    """Control: statsmodels only NaNs silently for a NaN in *endog*."""
    y, X = _plain()
    y = y.copy()
    y[5] = np.nan
    assert np.isnan(np.asarray(ols(y, X).params)).all()


# ---------------------------------------------------------------------------
# maxlags
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cov_type,extra", [
    ("HAC", {}),
    ("hac-panel", {"groups": np.repeat(np.arange(6), 10)}),
    ("hac-groupsum", {"time": np.tile(np.arange(10), 6)}),
])
def test_maxlags_must_be_a_whole_non_negative_number(cov_type, extra):
    """A negative bandwidth published HC0 numbers under a HAC label."""
    y, X, _, _ = _small_panel()
    with pytest.raises(ValueError, match="non-negative"):
        ols(y, X, cov_type=cov_type, cov_kwds={"maxlags": -3, **extra})
    with pytest.raises(ValueError, match="whole number"):
        ols(y, X, cov_type=cov_type, cov_kwds={"maxlags": 3.7, **extra})


def test_negative_maxlags_used_to_be_hc0_in_disguise():
    """Pin the mechanism the guard exists for, not only the message."""
    y, X = _plain()
    hc0 = np.asarray(ols(y, X, cov_type="HC0").bse)
    hac = np.asarray(ols(y, X, cov_type="HAC", cov_kwds={"maxlags": 4}).bse)
    assert not np.allclose(hc0, hac, atol=1e-6)   # they really do differ
    with pytest.raises(ValueError):
        ols(y, X, cov_type="HAC", cov_kwds={"maxlags": -1})


def test_integral_float_maxlags_is_accepted():
    """``np.floor(...)`` returns a float; 4.0 lags is still four lags."""
    sm = _sm()
    y, X = _plain()
    ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    got = ols(y, X, cov_type="HAC", cov_kwds={"maxlags": np.float64(3.0)})
    np.testing.assert_allclose(np.asarray(got.bse), np.asarray(ref.bse),
                               rtol=0, atol=ATOL)


# ---------------------------------------------------------------------------
# column labels
# ---------------------------------------------------------------------------

def test_integer_column_labels_are_not_stringified():
    """``pd.DataFrame(design)`` has integer columns; keep them integers."""
    sm = _sm()
    y, X = _plain()
    Xd = pd.DataFrame(X)
    ref = sm.OLS(y, Xd).fit()
    got = ols(y, Xd)
    assert list(got.params.index) == list(ref.params.index) == [0, 1, 2]
    assert got.params[1] == pytest.approx(ref.params[1], abs=ATOL)
    assert got.bse[2] == pytest.approx(ref.bse[2], abs=ATOL)
    assert list(got.conf_int().index) == [0, 1, 2]
    # summary() has to survive a non-string label.
    assert "coef" in got.summary()


def test_string_column_labels_are_unchanged():
    """Control: the pandas path the corpus actually uses does not move."""
    sm = _sm()
    y, X = _plain()
    Xd = pd.DataFrame(X, columns=["const", "x1", "x2"])
    ref = sm.OLS(y, Xd).fit()
    got = ols(y, Xd)
    assert list(got.params.index) == list(ref.params.index)
    assert got.params["x1"] == pytest.approx(ref.params["x1"], abs=ATOL)


def test_numeric_label_in_a_string_restriction_is_refused():
    """``"1 = 0"`` on integer columns cannot be disambiguated.

    statsmodels raises ``TypeError`` on the same call; guessing which
    side is a coefficient and which is a constant would publish a
    plausible wrong F.
    """
    y, X = _plain()
    res = ols(y, pd.DataFrame(X), cov_type="HC1")
    with pytest.raises(NotImplementedError, match="ambiguous"):
        res.f_test("1 = 0")


# ---------------------------------------------------------------------------
# statsmodels spellings the allow-list used to refuse
# ---------------------------------------------------------------------------

def test_use_t_inside_cov_kwds():
    """``RegressionResults.__init__`` pops it; the keyword still wins."""
    sm = _sm()
    y, X = _plain()
    ref = sm.OLS(y, X).fit(cov_type="HC1", cov_kwds={"use_t": True})
    got = ols(y, X, cov_type="HC1", cov_kwds={"use_t": True})
    assert got.use_t is ref.use_t is True
    np.testing.assert_allclose(np.asarray(got.pvalues),
                               np.asarray(ref.pvalues), rtol=0, atol=ATOL)

    ref2 = sm.OLS(y, X).fit(cov_type="HC1", cov_kwds={"use_t": True},
                            use_t=False)
    got2 = ols(y, X, cov_type="HC1", cov_kwds={"use_t": True}, use_t=False)
    assert got2.use_t is ref2.use_t is False

    # 'nonrobust' never reads cov_kwds in statsmodels, so honouring it
    # there would move a number; it stays refused.
    with pytest.raises(ValueError, match="use_t"):
        ols(y, X, cov_type="nonrobust", cov_kwds={"use_t": False})


@pytest.mark.parametrize("spelling", ["kernel", "weights_func"])
def test_uniform_kernel_for_the_hac_family(spelling):
    """``kernel='uniform'`` really changes the numbers; match them."""
    sm = _sm()
    from statsmodels.stats.sandwich_covariance import weights_uniform
    value = "uniform" if spelling == "kernel" else weights_uniform
    y, X, unit, time = _small_panel()
    cases = [
        ("HAC", {"maxlags": 4}),
        ("hac-panel", {"maxlags": 3, "groups": unit}),
        ("hac-groupsum", {"maxlags": 3, "time": time}),
    ]
    for cov_type, kwds in cases:
        bartlett = sm.OLS(y, X).fit(cov_type=cov_type, cov_kwds=dict(kwds))
        ref = sm.OLS(y, X).fit(cov_type=cov_type,
                               cov_kwds={**kwds, spelling: value})
        got = ols(y, X, cov_type=cov_type, cov_kwds={**kwds, spelling: value})
        # Compare the covariance rather than bse: the uniform kernel is
        # not guaranteed positive semi-definite, and on the groupsum
        # design one diagonal entry of the meat comes out negative in
        # both libraries, which would hide the comparison behind a NaN.
        assert not np.allclose(np.asarray(ref.cov_params()),
                               np.asarray(bartlett.cov_params()),
                               atol=1e-8), cov_type
        np.testing.assert_allclose(np.asarray(got.cov_params()),
                                   np.asarray(ref.cov_params()),
                                   rtol=0, atol=ATOL, err_msg=cov_type)


def test_bartlett_spelled_out_stays_on_the_default_path():
    """Control: naming the default kernel must not perturb it."""
    y, X = _plain()
    base = np.asarray(ols(y, X, cov_type="HAC", cov_kwds={"maxlags": 4}).bse)
    named = np.asarray(ols(y, X, cov_type="HAC",
                           cov_kwds={"maxlags": 4, "kernel": "bartlett"}).bse)
    np.testing.assert_array_equal(base, named)


def test_unknown_kernel_names_the_two_that_exist():
    y, X = _plain()
    with pytest.raises(ValueError, match="bartlett"):
        ols(y, X, cov_type="HAC",
            cov_kwds={"maxlags": 4, "kernel": "quadratic-spectral"})


def test_two_way_cluster_from_a_python_list():
    """``groups=[unit, time]`` is a two-way cluster, not a (2, n) design."""
    sm = _sm()
    y, X, unit, time = _small_panel()
    ref = sm.OLS(y, X).fit(cov_type="cluster",
                           cov_kwds={"groups": [unit, time]}, use_t=True)
    got = ols(y, X, cov_type="cluster", cov_kwds={"groups": [unit, time]},
              use_t=True)
    np.testing.assert_allclose(np.asarray(got.bse), np.asarray(ref.bse),
                               rtol=0, atol=ATOL)
    assert got.df_resid_inference == ref.df_resid_inference
    # and it agrees with the column-stacked spelling of the same thing
    stacked = ols(y, X, cov_type="cluster",
                  cov_kwds={"groups": np.column_stack([unit, time])},
                  use_t=True)
    np.testing.assert_allclose(np.asarray(got.bse), np.asarray(stacked.bse),
                               rtol=0, atol=ATOL)


def test_scalar_weights_are_broadcast_like_wls():
    sm = _sm()
    y, X = _plain()
    ref = sm.WLS(y, X, weights=2.0).fit()
    got = ols(y, X, weights=2.0)
    np.testing.assert_allclose(np.asarray(got.params),
                               np.asarray(ref.params), rtol=0, atol=ATOL)
    np.testing.assert_allclose(np.asarray(got.bse), np.asarray(ref.bse),
                               rtol=0, atol=ATOL)
    # llf carries the 0.5 n log(w) term, so the scalar is not a no-op.
    np.testing.assert_allclose(got.llf, ref.llf, rtol=0, atol=ATOL)
    assert got.llf != ols(y, X).llf


def test_wrong_length_weights_name_both_lengths():
    y, X = _plain()
    with pytest.raises(ValueError) as exc:
        ols(y, X, weights=np.ones(len(y) - 3))
    assert str(len(y) - 3) in str(exc.value) and str(len(y)) in str(exc.value)


# ---------------------------------------------------------------------------
# named diagnostics instead of bare numpy errors
# ---------------------------------------------------------------------------

def test_group_vectors_of_the_wrong_length_are_named():
    """Unfixed: 'array is not broadcastable to correct shape'."""
    y, X, unit, time = _small_panel()
    n = len(y)
    for cov_type, kwds, key in [
        ("cluster", {"groups": unit[:-5]}, "groups"),
        ("hac-groupsum", {"maxlags": 2, "time": time[:-5]}, "time"),
        ("hac-panel", {"maxlags": 2, "groups": unit[:-5]}, "groups"),
        ("hac-panel", {"maxlags": 2, "time": time[:-5]}, "time"),
    ]:
        with pytest.raises(ValueError) as exc:
            ols(y, X, cov_type=cov_type, cov_kwds=kwds)
        msg = str(exc.value)
        assert key in msg
        assert str(n - 5) in msg and str(n) in msg


def test_single_cluster_is_named_not_a_zero_division():
    """Unfixed: ``ZeroDivisionError('float division by zero')``."""
    y, X = _plain()
    g = np.zeros(len(y), dtype=int)
    with pytest.raises(ValueError, match="G/\\(G-1\\)"):
        ols(y, X, cov_type="cluster", cov_kwds={"groups": g})
    with pytest.raises(ValueError, match="G - 1 = 0"):
        ols(y, X, cov_type="cluster",
            cov_kwds={"groups": g, "use_correction": False})
    # With both knobs turned off the estimator is defined, so it runs.
    res = ols(y, X, cov_type="cluster",
              cov_kwds={"groups": g, "use_correction": False,
                        "df_correction": False})
    assert np.isfinite(np.asarray(res.bse)).all()


def test_saturated_design_is_named_not_a_zero_division():
    """``n == rank``: statsmodels returns scale=inf, we say what happened."""
    y, X = _plain()
    with pytest.raises(ValueError, match="saturated"):
        ols(y[:3], X[:3])


def test_add_constant_rejects_a_zero_row_array():
    """statsmodels raises the empty-reduction ValueError on the same input."""
    sm = _sm()
    with pytest.raises(ValueError):
        sm.add_constant(np.zeros((0, 2)))
    with pytest.raises(ValueError, match="zero-row"):
        add_constant(np.zeros((0, 2)))
    # Control: the pandas branch agrees with statsmodels, which does not
    # raise there, so it must keep not raising.
    assert add_constant(pd.DataFrame(np.zeros((0, 2)))).shape == \
        sm.add_constant(pd.DataFrame(np.zeros((0, 2)))).shape
