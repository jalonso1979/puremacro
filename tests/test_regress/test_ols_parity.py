"""statsmodels parity for :mod:`puremacro.regress.ols`.

The sweep in :func:`test_cov_type_parity` is the contract: every
covariance type this module claims to support is fitted twice — once
with ``puremacro.regress.ols.ols`` and once with statsmodels 0.14.6 —
on a design chosen to make that covariance's small-sample factors and
degrees-of-freedom rules bite. ``params``, ``bse``, ``tvalues``,
``pvalues``, ``conf_int``, ``nobs``, ``df_resid``, ``rsquared``, ``aic``
and ``bic`` must agree to ``atol=1e-10``; ``f_test`` / ``t_test`` to
``atol=1e-8``.

The designs are deliberately awkward: AR(1) errors for HAC, a clustered
panel with unequal and singleton clusters, a stacked panel with a common
time shock for Driscoll-Kraay, and a weighted regression whose one call
site (the corpus's single ``sm.WLS``) exercises weights, clustering and
the ``t(G-1)`` denominator together.

statsmodels is a dev-only dependency (see ARCHITECTURE.md), so every
parity test asks for it through ``pytest.importorskip``.
"""
import numpy as np
import pandas as pd
import pytest

from puremacro.regress.ols import (
    FTestResult,
    OLSResult,
    PredictionResult,
    TTestResult,
    add_constant,
    ols,
)

ATOL = 1e-10
ATOL_TEST = 1e-8


def _sm():
    """statsmodels.api, or skip — it never ships in the runtime deps."""
    return pytest.importorskip("statsmodels.api")


# ---------------------------------------------------------------------------
# designs
# ---------------------------------------------------------------------------

def _ar1_timeseries(n=200, rho=0.7, seed=11):
    """Time series with AR(1) errors — the case HAC exists for."""
    rng = np.random.default_rng(seed)
    x1 = np.cumsum(rng.standard_normal(n)) / 10.0
    x2 = rng.standard_normal(n)
    u = np.zeros(n)
    e = rng.standard_normal(n)
    for t in range(1, n):
        u[t] = rho * u[t - 1] + e[t]
    X = np.column_stack([np.ones(n), x1, x2])
    y = X @ np.array([0.4, 1.2, -0.6]) + u
    return y, X


def _clustered(seed=5):
    """Unequal cluster sizes, including two singletons."""
    rng = np.random.default_rng(seed)
    sizes = [25, 18, 14, 12, 11, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1]
    groups = np.concatenate([np.full(s, g) for g, s in enumerate(sizes)])
    n = groups.size
    alpha = rng.standard_normal(len(sizes))[groups]
    x1 = rng.standard_normal(n) + 0.5 * alpha
    x2 = rng.uniform(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    y = X @ np.array([1.0, 0.8, -0.4]) + alpha + rng.standard_normal(n)
    return y, X, groups


def _panel(n_units=8, n_periods=30, seed=7):
    """Unit-major panel with a common time shock (Driscoll-Kraay's target)."""
    rng = np.random.default_rng(seed)
    shock = rng.standard_normal(n_periods)
    rows = []
    for i in range(n_units):
        fe = rng.standard_normal()
        for t in range(n_periods):
            x1 = rng.standard_normal() + 0.3 * shock[t]
            x2 = rng.standard_normal()
            u = 1.5 * shock[t] + 0.5 * rng.standard_normal() + fe
            rows.append((i, t, x1, x2, 0.7 + 0.9 * x1 - 0.3 * x2 + u))
    arr = np.array(rows, dtype=float)
    unit = arr[:, 0].astype(int)
    time = arr[:, 1].astype(int)
    X = np.column_stack([np.ones(len(arr)), arr[:, 2], arr[:, 3]])
    return arr[:, 4], X, unit, time


def _weighted(seed=13):
    """Weighted, clustered event-study-shaped design (mirrors N07:796)."""
    rng = np.random.default_rng(seed)
    sizes = [9, 8, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 1, 1]
    groups = np.concatenate([np.full(s, g) for g, s in enumerate(sizes)])
    n = groups.size
    post = (rng.uniform(size=n) > 0.5).astype(float)
    treat = (groups % 2 == 0).astype(float)
    X = np.column_stack([np.ones(n), post, treat, post * treat])
    w = rng.uniform(0.4, 4.0, n)
    y = X @ np.array([1.0, 0.3, 0.2, -0.5]) + rng.standard_normal(n) / np.sqrt(w)
    return y, X, w, groups


def _as_frame(X, names=None):
    names = names or [f"x{i}" for i in range(X.shape[1])]
    names = ["const"] + names[1:] if np.allclose(X[:, 0], 1.0) else names
    return pd.DataFrame(X, columns=names)


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

def _cases():
    """(id, y, X, weights, cov_type, cov_kwds, use_t) for every cov_type."""
    y_ts, X_ts = _ar1_timeseries()
    y_cl, X_cl, g_cl = _clustered()
    y_pn, X_pn, unit, time = _panel()
    y_w, X_w, w, g_w = _weighted()

    cases = [
        ("nonrobust", y_ts, X_ts, None, "nonrobust", None, None),
        ("nonrobust_use_t_false", y_ts, X_ts, None, "nonrobust", None, False),
        ("HC0", y_ts, X_ts, None, "HC0", None, None),
        ("HC1", y_ts, X_ts, None, "HC1", None, None),
        ("HC1_use_t", y_ts, X_ts, None, "HC1", None, True),
        ("HC2", y_ts, X_ts, None, "HC2", None, None),
        ("HC3", y_ts, X_ts, None, "HC3", None, None),
        ("HAC_default_correction", y_ts, X_ts, None, "HAC",
         {"maxlags": 5}, None),
        ("HAC_use_correction", y_ts, X_ts, None, "HAC",
         {"maxlags": 5, "use_correction": True}, None),
        ("HAC_use_t", y_ts, X_ts, None, "HAC", {"maxlags": 4}, True),
        ("HAC_lag0", y_ts, X_ts, None, "HAC", {"maxlags": 0}, None),
        ("cluster_default", y_cl, X_cl, None, "cluster",
         {"groups": g_cl}, None),
        ("cluster_use_t", y_cl, X_cl, None, "cluster",
         {"groups": g_cl}, True),
        ("cluster_no_correction", y_cl, X_cl, None, "cluster",
         {"groups": g_cl, "use_correction": False}, True),
        ("cluster_no_df_correction", y_cl, X_cl, None, "cluster",
         {"groups": g_cl, "df_correction": False}, True),
        ("cluster_two_way", y_pn, X_pn, None, "cluster",
         {"groups": np.column_stack([unit, time])}, True),
        ("hac_panel_groups", y_pn, X_pn, None, "hac-panel",
         {"maxlags": 3, "groups": unit}, None),
        ("hac_panel_time", y_pn, X_pn, None, "hac-panel",
         {"maxlags": 2, "time": time}, True),
        ("hac_panel_cluster_corr", y_pn, X_pn, None, "hac-panel",
         {"maxlags": 3, "groups": unit, "use_correction": "cluster"}, True),
        ("hac_panel_no_corr", y_pn, X_pn, None, "hac-panel",
         {"maxlags": 3, "groups": unit, "use_correction": False}, None),
        ("hac_groupsum_default", y_pn, X_pn, None, "hac-groupsum",
         {"maxlags": 3, "time": time}, None),
        ("hac_groupsum_use_t", y_pn, X_pn, None, "hac-groupsum",
         {"maxlags": 3, "time": time}, True),
        ("hac_groupsum_hac_corr", y_pn, X_pn, None, "hac-groupsum",
         {"maxlags": 2, "time": time, "use_correction": "hac"}, None),
        ("hac_groupsum_no_corr", y_pn, X_pn, None, "hac-groupsum",
         {"maxlags": 2, "time": time, "use_correction": False}, None),
        ("wls_nonrobust", y_w, X_w, w, "nonrobust", None, None),
        ("wls_HC1", y_w, X_w, w, "HC1", None, None),
        ("wls_HC3", y_w, X_w, w, "HC3", None, None),
        ("wls_HAC", y_w, X_w, w, "HAC", {"maxlags": 3}, None),
        ("wls_cluster_use_t", y_w, X_w, w, "cluster",
         {"groups": g_w}, True),
    ]
    # Every case is run twice: once with a bare ndarray design and once
    # with a DataFrame, because the two take different code paths through
    # both libraries' wrappers.
    out = []
    for cid, y, X, wts, ct, ck, ut in cases:
        out.append(pytest.param(y, X, wts, ct, ck, ut, False, id=cid))
        out.append(
            pytest.param(y, X, wts, ct, ck, ut, True, id=f"{cid}-df")
        )
    return out


def _fit_both(y, X, weights, cov_type, cov_kwds, use_t, as_frame):
    sm = _sm()
    design = _as_frame(X) if as_frame else X
    endog = pd.Series(y, name="y") if as_frame else y
    model = (
        sm.OLS(endog, design)
        if weights is None
        else sm.WLS(endog, design, weights=weights)
    )
    if cov_type == "nonrobust":
        ref = model.fit(use_t=use_t)
    else:
        ref = model.fit(cov_type=cov_type, cov_kwds=dict(cov_kwds or {}),
                        use_t=use_t)
    got = ols(endog, design, weights=weights, cov_type=cov_type,
              cov_kwds=cov_kwds, use_t=use_t)
    return ref, got


@pytest.mark.parametrize(
    "y,X,weights,cov_type,cov_kwds,use_t,as_frame", _cases()
)
def test_cov_type_parity(y, X, weights, cov_type, cov_kwds, use_t, as_frame):
    ref, got = _fit_both(y, X, weights, cov_type, cov_kwds, use_t, as_frame)

    np.testing.assert_allclose(np.asarray(got.params, dtype=float),
                               np.asarray(ref.params, dtype=float),
                               rtol=0, atol=ATOL)
    np.testing.assert_allclose(np.asarray(got.bse, dtype=float),
                               np.asarray(ref.bse, dtype=float),
                               rtol=0, atol=ATOL)
    np.testing.assert_allclose(np.asarray(got.tvalues, dtype=float),
                               np.asarray(ref.tvalues, dtype=float),
                               rtol=0, atol=ATOL)
    np.testing.assert_allclose(np.asarray(got.pvalues, dtype=float),
                               np.asarray(ref.pvalues, dtype=float),
                               rtol=0, atol=ATOL)
    np.testing.assert_allclose(np.asarray(got.conf_int(), dtype=float),
                               np.asarray(ref.conf_int(), dtype=float),
                               rtol=0, atol=ATOL)
    np.testing.assert_allclose(np.asarray(got.conf_int(alpha=0.10),
                                          dtype=float),
                               np.asarray(ref.conf_int(alpha=0.10),
                                          dtype=float),
                               rtol=0, atol=ATOL)
    np.testing.assert_allclose(np.asarray(got.cov_params(), dtype=float),
                               np.asarray(ref.cov_params(), dtype=float),
                               rtol=0, atol=ATOL)
    assert got.nobs == ref.nobs
    assert got.df_resid == ref.df_resid
    assert got.df_model == ref.df_model
    assert got.use_t == ref.use_t
    np.testing.assert_allclose(got.rsquared, ref.rsquared, rtol=0, atol=ATOL)
    np.testing.assert_allclose(got.rsquared_adj, ref.rsquared_adj,
                               rtol=0, atol=ATOL)
    np.testing.assert_allclose(got.scale, ref.scale, rtol=0, atol=ATOL)
    np.testing.assert_allclose(got.llf, ref.llf, rtol=0, atol=ATOL)
    np.testing.assert_allclose(got.aic, ref.aic, rtol=0, atol=ATOL)
    np.testing.assert_allclose(got.bic, ref.bic, rtol=0, atol=ATOL)
    np.testing.assert_allclose(np.asarray(got.resid, dtype=float),
                               np.asarray(ref.resid, dtype=float),
                               rtol=0, atol=ATOL)
    np.testing.assert_allclose(np.asarray(got.fittedvalues, dtype=float),
                               np.asarray(ref.fittedvalues, dtype=float),
                               rtol=0, atol=ATOL)
    # df_resid_inference is the whole point of the cluster family.
    expected_df = getattr(ref, "df_resid_inference", ref.df_resid)
    assert got.df_resid_inference == expected_df


@pytest.mark.parametrize("as_frame", [False, True])
def test_pandas_containers_match_statsmodels(as_frame):
    """Series-vs-ndarray return types follow statsmodels exactly."""
    _sm()
    y, X, groups = _clustered()
    ref, got = _fit_both(y, X, None, "cluster", {"groups": groups}, True,
                         as_frame)
    for attr in ("params", "bse", "tvalues", "pvalues"):
        assert isinstance(getattr(got, attr), pd.Series) is as_frame
        assert isinstance(getattr(ref, attr), pd.Series) is as_frame
    assert isinstance(got.conf_int(), pd.DataFrame) is as_frame
    assert isinstance(got.cov_params(), pd.DataFrame) is as_frame
    if as_frame:
        assert list(got.params.index) == list(ref.params.index)
        assert list(got.conf_int().index) == list(ref.conf_int().index)
        # The corpus indexes by name everywhere: m.params["x1"].
        assert got.params["x1"] == pytest.approx(ref.params["x1"], abs=ATOL)
        assert got.bse["x2"] == pytest.approx(ref.bse["x2"], abs=ATOL)


# ---------------------------------------------------------------------------
# restriction tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cov_type,cov_kwds,use_t",
    [
        ("nonrobust", None, None),
        ("HC1", None, None),
        ("cluster", "groups", True),   # df_denom must be G-1, not n-k
        ("cluster", "groups", False),
    ],
)
def test_f_test_parity(cov_type, cov_kwds, use_t):
    _sm()
    y, X, groups = _clustered()
    kwds = {"groups": groups} if cov_kwds == "groups" else cov_kwds
    ref, got = _fit_both(y, X, None, cov_type, kwds, use_t, True)

    for R in (
        np.array([[0.0, 1.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[0.0, 1.0, -1.0]]),
    ):
        f_ref = ref.f_test(R)
        f_got = got.f_test(R)
        assert isinstance(f_got, FTestResult)
        np.testing.assert_allclose(f_got.fvalue, float(f_ref.fvalue),
                                   rtol=0, atol=ATOL_TEST)
        np.testing.assert_allclose(f_got.pvalue, float(f_ref.pvalue),
                                   rtol=0, atol=ATOL_TEST)
        assert f_got.df_num == f_ref.df_num
        assert f_got.df_denom == f_ref.df_denom


@pytest.mark.parametrize("use_t", [True, False])
def test_t_test_parity_including_non_zero_null(use_t):
    """``t_test`` with a non-zero null — the N03:478 shape."""
    _sm()
    y, X, groups = _clustered()
    ref, got = _fit_both(y, X, None, "cluster", {"groups": groups}, use_t,
                         True)

    for R, q in (
        (np.array([[0.0, 1.0, 0.0]]), None),
        (np.array([[0.0, 1.0, 0.0]]), np.array([1.0])),
        (np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), np.array([0.5, -0.5])),
    ):
        spec = R if q is None else (R, q)
        t_ref = ref.t_test(spec)
        t_got = got.t_test(spec)
        assert isinstance(t_got, TTestResult)
        # statsmodels returns sd / tvalue as a (1, 1) matrix and a 0-d
        # pvalue for a single restriction; we always return length-q
        # vectors, so compare raveled.
        np.testing.assert_allclose(np.ravel(t_got.effect),
                                   np.ravel(t_ref.effect),
                                   rtol=0, atol=ATOL_TEST)
        np.testing.assert_allclose(np.ravel(t_got.sd), np.ravel(t_ref.sd),
                                   rtol=0, atol=ATOL_TEST)
        np.testing.assert_allclose(np.ravel(t_got.tvalue),
                                   np.ravel(np.asarray(t_ref.tvalue)),
                                   rtol=0, atol=ATOL_TEST)
        np.testing.assert_allclose(np.ravel(t_got.pvalue),
                                   np.ravel(np.asarray(t_ref.pvalue)),
                                   rtol=0, atol=ATOL_TEST)
        np.testing.assert_allclose(np.ravel(t_got.conf_int()),
                                   np.ravel(t_ref.conf_int()),
                                   rtol=0, atol=ATOL_TEST)


@pytest.mark.parametrize(
    "spec", ["x1 = 0", "x1 = x2", "x1 = 1", "x1 = 0, x2 = 0", "2*x1 - x2 = 1"]
)
def test_string_restrictions_match_statsmodels(spec):
    """The optional string form, checked against patsy's parser."""
    _sm()
    y, X, groups = _clustered()
    ref, got = _fit_both(y, X, None, "cluster", {"groups": groups}, True, True)
    np.testing.assert_allclose(got.f_test(spec).fvalue,
                               float(ref.f_test(spec).fvalue),
                               rtol=0, atol=ATOL_TEST)
    np.testing.assert_allclose(got.f_test(spec).pvalue,
                               float(ref.f_test(spec).pvalue),
                               rtol=0, atol=ATOL_TEST)
    np.testing.assert_allclose(np.ravel(got.t_test(spec).pvalue),
                               np.ravel(np.asarray(ref.t_test(spec).pvalue)),
                               rtol=0, atol=ATOL_TEST)


def test_unsupported_string_restriction_names_the_array_form():
    y, X, groups = _clustered()
    res = ols(y, _as_frame(X), cov_type="cluster", cov_kwds={"groups": groups})
    with pytest.raises(NotImplementedError, match="restriction array"):
        res.f_test("np.log(x1) = 0")
    # Without column names there is nothing for a string to refer to.
    bare = ols(y, X)
    with pytest.raises(NotImplementedError, match="restriction array"):
        bare.t_test("x1 = 0")


def test_get_prediction_parity():
    _sm()
    y, X = _ar1_timeseries()
    ref, got = _fit_both(y, X, None, "HAC", {"maxlags": 5}, None, True)
    new = _as_frame(np.column_stack([
        np.ones(4), np.linspace(-1, 1, 4), np.linspace(0.5, -0.5, 4)
    ]))
    p_ref = ref.get_prediction(new)
    p_got = got.get_prediction(new)
    assert isinstance(p_got, PredictionResult)
    np.testing.assert_allclose(p_got.predicted_mean,
                               np.asarray(p_ref.predicted_mean),
                               rtol=0, atol=ATOL)
    np.testing.assert_allclose(p_got.se_mean, np.asarray(p_ref.se_mean),
                               rtol=0, atol=ATOL)
    np.testing.assert_allclose(p_got.conf_int(), p_ref.conf_int(),
                               rtol=0, atol=ATOL)
    # In-sample prediction reproduces the fitted values.
    np.testing.assert_allclose(got.get_prediction().predicted_mean,
                               np.asarray(got.fittedvalues, dtype=float),
                               rtol=0, atol=ATOL)


def test_missing_drop_matches_statsmodels():
    sm = _sm()
    y, X = _ar1_timeseries(n=120)
    y = y.copy()
    X = X.copy()
    y[3] = np.nan
    X[17, 2] = np.nan
    y[100] = np.nan
    ref = sm.OLS(y, X, missing="drop").fit(cov_type="HC1")
    got = ols(y, X, cov_type="HC1", missing="drop")
    assert got.nobs == 117 == ref.nobs
    np.testing.assert_allclose(np.asarray(got.params), np.asarray(ref.params),
                               rtol=0, atol=ATOL)
    np.testing.assert_allclose(np.asarray(got.bse), np.asarray(ref.bse),
                               rtol=0, atol=ATOL)


def test_add_constant_matches_statsmodels():
    sm = _sm()
    rng = np.random.default_rng(2)
    arr = rng.standard_normal((6, 2))
    frame = pd.DataFrame(arr, columns=["a", "b"], index=list("uvwxyz"))
    already = np.column_stack([np.ones(6), arr])
    already_df = pd.DataFrame(already, columns=["const", "a", "b"])

    np.testing.assert_array_equal(add_constant(arr), sm.add_constant(arr))
    np.testing.assert_array_equal(add_constant(arr, prepend=False),
                                  sm.add_constant(arr, prepend=False))
    np.testing.assert_array_equal(add_constant(already),
                                  sm.add_constant(already))
    np.testing.assert_array_equal(
        add_constant(already, has_constant="add"),
        sm.add_constant(already, has_constant="add"),
    )
    np.testing.assert_array_equal(add_constant(arr[:, 0]),
                                  sm.add_constant(arr[:, 0]))
    pd.testing.assert_frame_equal(add_constant(frame), sm.add_constant(frame))
    pd.testing.assert_frame_equal(add_constant(frame, prepend=False),
                                  sm.add_constant(frame, prepend=False))
    pd.testing.assert_frame_equal(add_constant(already_df),
                                  sm.add_constant(already_df))
    pd.testing.assert_frame_equal(add_constant(frame["a"]),
                                  sm.add_constant(frame["a"]))


def test_one_regressor_no_constant_matches_statsmodels():
    """k=1 with no constant: R^2 must be the *uncentered* one."""
    sm = _sm()
    rng = np.random.default_rng(4)
    x = rng.standard_normal((40, 1))
    y = 2.0 * x[:, 0] + rng.standard_normal(40)
    ref = sm.OLS(y, x).fit()
    got = ols(y, x)
    assert got.k_constant == 0
    assert got.df_model == ref.df_model == 1.0
    np.testing.assert_allclose(got.rsquared, ref.rsquared, rtol=0, atol=ATOL)
    np.testing.assert_allclose(got.rsquared_adj, ref.rsquared_adj,
                               rtol=0, atol=ATOL)
    np.testing.assert_allclose(np.asarray(got.pvalues),
                               np.asarray(ref.pvalues), rtol=0, atol=ATOL)
    # A 1-D X is accepted and treated as a single column, as statsmodels does.
    flat = ols(y, x[:, 0])
    np.testing.assert_allclose(np.asarray(flat.params),
                               np.asarray(got.params), rtol=0, atol=ATOL)


def test_implicit_constant_detected_like_statsmodels():
    """A full dummy set has no ones column but spans one."""
    sm = _sm()
    rng = np.random.default_rng(6)
    g = np.repeat(np.arange(3), 12)
    D = np.column_stack([(g == j).astype(float) for j in range(3)])
    y = D @ np.array([1.0, 2.0, 3.0]) + rng.standard_normal(36)
    ref = sm.OLS(y, D).fit()
    got = ols(y, D)
    assert got.k_constant == 1
    assert got.df_model == ref.df_model
    np.testing.assert_allclose(got.rsquared, ref.rsquared, rtol=0, atol=ATOL)


# ---------------------------------------------------------------------------
# behaviour that is ours, not statsmodels'
# ---------------------------------------------------------------------------

def test_singular_design_raises_a_named_error():
    rng = np.random.default_rng(8)
    x = rng.standard_normal(40)
    X = np.column_stack([np.ones(40), x, 2.0 * x])  # exact collinearity
    y = rng.standard_normal(40)
    with pytest.raises(np.linalg.LinAlgError) as exc:
        ols(y, X)
    msg = str(exc.value)
    assert "singular" in msg
    assert "null space" in msg          # names the offending columns
    # And the diagnostic is not thrown away by a robust cov_type.
    with pytest.raises(np.linalg.LinAlgError):
        ols(y, X, cov_type="HC1")


def test_nan_does_not_propagate_silently():
    """``missing='none'`` is the statsmodels default; 'raise' is opt-in."""
    y, X = _ar1_timeseries(n=60)
    y = y.copy()
    y[5] = np.nan
    silent = ols(y, X)
    assert np.isnan(np.asarray(silent.params)).all()   # statsmodels' behaviour
    with pytest.raises(ValueError, match="NaN"):
        ols(y, X, missing="raise")
    dropped = ols(y, X, missing="drop")
    assert dropped.nobs == 59
    assert np.isfinite(np.asarray(dropped.params)).all()


def test_unknown_cov_kwds_key_raises():
    y, X, groups = _clustered()
    with pytest.raises(ValueError, match="maxlag"):
        ols(y, X, cov_type="HAC", cov_kwds={"maxlag": 4})
    with pytest.raises(ValueError, match="groups"):
        ols(y, X, cov_type="cluster")


def test_panel_use_correction_true_is_refused():
    """statsmodels silently applies no correction here; we say so."""
    y, X, unit, time = _panel()
    with pytest.raises(ValueError, match="not a bool"):
        ols(y, X, cov_type="hac-groupsum",
            cov_kwds={"maxlags": 2, "time": time, "use_correction": True})
    with pytest.raises(ValueError, match="not a bool"):
        ols(y, X, cov_type="hac-panel",
            cov_kwds={"maxlags": 2, "groups": unit, "use_correction": True})


def test_cov_type_names_are_case_insensitive():
    _sm()
    y, X = _ar1_timeseries(n=80)
    a = ols(y, X, cov_type="hc1")
    b = ols(y, X, cov_type="HC1")
    c = ols(y, X, cov_type="hac", cov_kwds={"maxlags": 3})
    d = ols(y, X, cov_type="HAC", cov_kwds={"maxlags": 3})
    np.testing.assert_allclose(a.cov, b.cov, rtol=0, atol=ATOL)
    np.testing.assert_allclose(c.cov, d.cov, rtol=0, atol=ATOL)
    assert a.cov_type == "HC1" and c.cov_type == "HAC"
    # The old statsmodels aliases still resolve.
    assert ols(y, X, cov_type="nw-panel",
               cov_kwds={"maxlags": 2, "groups": np.repeat([0, 1], 40)},
               ).cov_type == "hac-panel"
    with pytest.raises(ValueError, match="not recognised"):
        ols(y, X, cov_type="HC9")


def test_result_object_shape_and_summary():
    y, X, groups = _clustered()
    res = ols(y, _as_frame(X), cov_type="cluster",
              cov_kwds={"groups": groups}, use_t=True)
    assert isinstance(res, OLSResult)
    assert res.n_groups == 15
    assert res.df_resid_inference == 14.0
    text = res.summary()
    assert "cluster" in text and "const" in text and "R^2" in text
    # wresid is the residual that entered the sandwich.
    np.testing.assert_allclose(np.asarray(res.wresid),
                               np.asarray(res.resid, dtype=float),
                               rtol=0, atol=1e-12)


def test_weights_enter_every_sandwich():
    """A weighted fit must not reduce to the unweighted one anywhere.

    Positive control for the WLS path: if the weights were dropped from
    the sandwich (the silent failure PARITY_SPEC warns about) these
    numbers would coincide.
    """
    _sm()
    y, X, w, groups = _weighted()
    for cov_type, kwds in (("HC1", None), ("HAC", {"maxlags": 2}),
                           ("cluster", {"groups": groups})):
        weighted = ols(y, X, weights=w, cov_type=cov_type, cov_kwds=kwds)
        plain = ols(y, X, cov_type=cov_type, cov_kwds=kwds)
        assert not np.allclose(np.asarray(weighted.bse),
                               np.asarray(plain.bse), atol=1e-6)


def test_module_is_pyodide_clean():
    """No statsmodels import may leak in from the estimator module."""
    import subprocess
    import sys

    code = (
        "import sys, puremacro.regress.ols as m; "
        "assert 'statsmodels' not in sys.modules, sorted(sys.modules)[:0]; "
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_hac_groupsum_with_gapped_time_codes():
    """Empty periods enter the HAC series as zero rows, as in group_sums.

    statsmodels bincounts the integer period codes, so a period with no
    observations is a row of zeros in the Driscoll-Kraay series and
    shifts every Bartlett lag after it. Skipping those rows (the obvious
    implementation) silently changes the answer.
    """
    sm = _sm()
    y, X, unit, time = _panel(n_units=6, n_periods=20)
    gapped = time * 2  # periods 1, 3, 5, ... are now empty
    ref = sm.OLS(y, X).fit(cov_type="hac-groupsum",
                           cov_kwds={"maxlags": 3, "time": gapped})
    got = ols(y, X, cov_type="hac-groupsum",
              cov_kwds={"maxlags": 3, "time": gapped})
    np.testing.assert_allclose(np.asarray(got.bse), np.asarray(ref.bse),
                               rtol=0, atol=ATOL)
    # ... and the gapped codes are genuinely different from the dense ones.
    dense = ols(y, X, cov_type="hac-groupsum",
                cov_kwds={"maxlags": 3, "time": time})
    assert not np.allclose(np.asarray(got.bse), np.asarray(dense.bse),
                           atol=1e-8)


def test_hac_groupsum_with_very_sparse_time_codes():
    """statsmodels relabels the codes when max(time) > 2 * nobs."""
    sm = _sm()
    y, X, unit, time = _panel(n_units=4, n_periods=15)
    sparse = time * 1000
    assert sparse.max() > 2 * len(y)
    ref = sm.OLS(y, X).fit(cov_type="hac-groupsum",
                           cov_kwds={"maxlags": 2, "time": sparse})
    got = ols(y, X, cov_type="hac-groupsum",
              cov_kwds={"maxlags": 2, "time": sparse})
    np.testing.assert_allclose(np.asarray(got.bse), np.asarray(ref.bse),
                               rtol=0, atol=ATOL)


def test_hac_groupsum_rejects_non_integer_time():
    y, X, unit, time = _panel(n_units=3, n_periods=10)
    with pytest.raises(ValueError, match="integer period code"):
        ols(y, X, cov_type="hac-groupsum",
            cov_kwds={"maxlags": 2, "time": time.astype(float)})


def test_hac_default_bandwidth_matches_statsmodels():
    """Omitting maxlags uses statsmodels' floor(4 (T/100)^(2/9)) rule."""
    sm = _sm()
    y, X = _ar1_timeseries(n=200)
    ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": None})
    got = ols(y, X, cov_type="HAC")
    np.testing.assert_allclose(np.asarray(got.bse), np.asarray(ref.bse),
                               rtol=0, atol=ATOL)
    # The rule is a real bandwidth, not lag 0.
    assert not np.allclose(np.asarray(got.bse),
                           np.asarray(ols(y, X, cov_type="HAC",
                                          cov_kwds={"maxlags": 0}).bse),
                           atol=1e-6)


def test_container_rule_is_driven_by_X_not_by_y():
    """Documented departure: statsmodels invents names, we do not.

    With a pandas ``y`` and a bare ndarray ``X``, statsmodels returns
    ``params`` as a Series indexed by the invented labels ``const``,
    ``x1``, ...; here ``X`` decides, so the result is an ndarray. The row
    labels of ``resid`` still follow whichever input carried an index.
    """
    sm = _sm()
    y, X = _ar1_timeseries(n=60)
    ys = pd.Series(y, index=[f"r{i}" for i in range(60)])
    ref = sm.OLS(ys, X).fit()
    got = ols(ys, X)
    assert isinstance(ref.params, pd.Series)
    assert list(ref.params.index) == ["const", "x1", "x2"]
    assert isinstance(got.params, np.ndarray)
    np.testing.assert_allclose(np.asarray(got.params),
                               np.asarray(ref.params), rtol=0, atol=ATOL)
    assert isinstance(got.resid, pd.Series)
    assert list(got.resid.index) == list(ys.index)


def test_rank_deficient_restrictions_warn_and_match_statsmodels():
    """Duplicated restrictions: J drops to the rank, as in wald_test."""
    _sm()
    y, X, groups = _clustered()
    ref, got = _fit_both(y, X, None, "cluster", {"groups": groups}, True, True)
    R = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.warns(Warning, match="does not have full rank"):
        f_ref = ref.f_test(R)
    with pytest.warns(UserWarning, match="does not have full rank"):
        f_got = got.f_test(R)
    assert f_got.df_num == f_ref.df_num == 1
    np.testing.assert_allclose(f_got.fvalue, float(f_ref.fvalue),
                               rtol=0, atol=ATOL_TEST)
    np.testing.assert_allclose(f_got.pvalue, float(f_ref.pvalue),
                               rtol=0, atol=ATOL_TEST)


def test_string_restriction_accepts_operator_bearing_column_names():
    """Real designs have names like ``I(x ** 2)``; the whole side matches."""
    rng = np.random.default_rng(21)
    x = rng.standard_normal(60)
    df = pd.DataFrame({"const": 1.0, "I(x ** 2)": x ** 2,
                       "region_North-East": rng.standard_normal(60)})
    y = df.to_numpy() @ np.array([1.0, 0.5, -0.2]) + rng.standard_normal(60)
    res = ols(y, df, cov_type="HC1")
    for name in ("I(x ** 2)", "region_North-East"):
        got = res.f_test(f"{name} = 0")
        R = np.zeros((1, 3))
        R[0, list(df.columns).index(name)] = 1.0
        np.testing.assert_allclose(got.fvalue, res.f_test(R).fvalue,
                                   rtol=0, atol=ATOL_TEST)
        # and it equals the squared t on that coefficient
        np.testing.assert_allclose(
            got.fvalue, float(res.tvalues[name]) ** 2, rtol=0, atol=1e-8
        )
