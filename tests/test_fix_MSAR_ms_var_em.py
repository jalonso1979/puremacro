"""Regression tests for the MS-VAR EM defect (review key MSAR).

``ms_var_fit`` estimates an MSIH(K)-VAR(p): regime-specific intercepts
``mu_k`` and covariances ``Sigma_k`` with one autoregressive matrix ``A``
shared by all regimes. Until the fix, the M-step updated ``A`` by an
unweighted OLS that ignored ``Sigma_k^{-1}``. That step does not maximise
the EM auxiliary function when the regime covariances differ, so the
log-likelihood fell across iterations (e.g. 28 of 39 steps on the
univariate heteroskedastic DGP below) and the fit stopped below the MLE
of its own model while reporting ``converged=True``.

The tests check, on several DGPs and on Hamilton's GNP growth series:

- the log-likelihood never falls across EM iterations, both through the
  public API (refitting with ``n_iter = 1, 2, ...``) and through the new
  ``loglik_path`` field;
- the reported log-likelihood is the one of the returned parameters,
  recomputed by an independent reference filter;
- the EM fixed point is a stationary point of the likelihood (score ~ 0);
- ``converged`` is set only when the log-likelihood change is below
  ``tol``, and a decrease triggers a ``RuntimeWarning``;
- on data where the model classes coincide, the MLE agrees with
  statsmodels ``MarkovRegression`` (switching intercept and variance,
  shared AR coefficients), after allowing for the different treatment of
  the initial regime probabilities;
- the fit is equivariant to the units of each variable. The first fix
  floored every Sigma_k at 1e-8 times the *average* residual variance,
  which erased the variance switching of a small-scale variable (a rate in
  decimals next to GDP in $bn) and lost hundreds of log-likelihood points;
- a regime that collapses onto the variance floor is reported
  (``sigma_at_floor`` and a ``RuntimeWarning``), and over-parameterised or
  degenerate samples are refused with a ``ValueError``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from puremacro.var.regime import ms_var as ms_mod
from puremacro.var.regime.ms_var import ms_var_fit


# ---------------------------------------------------------------------------
# DGPs and an independent reference likelihood
# ---------------------------------------------------------------------------

def _simulate_msih(T: int = 400, seed: int = 1, n: int = 1) -> np.ndarray:
    """MSIH(2)-VAR(1) with strongly different regime variances."""
    rng = np.random.default_rng(seed)
    P = np.array([[0.95, 0.05], [0.10, 0.90]])
    mu = np.array([[1.0, 0.5][:n], [-1.0, -0.5][:n]])
    sig = [0.3 * np.eye(n), 2.0 * np.eye(n)]
    A = 0.6 * np.eye(n) if n == 1 else np.array([[0.5, 0.1], [0.0, 0.4]])
    s = np.zeros(T, int)
    Y = np.zeros((T, n))
    for t in range(1, T):
        s[t] = rng.choice(2, p=P[s[t - 1]])
        Y[t] = mu[s[t]] + A @ Y[t - 1] + sig[s[t]] @ rng.standard_normal(n)
    return Y


def _simulate_k3(T: int = 500, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    P = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.1, 0.1, 0.8]])
    mu = np.array([0.0, 1.5, -1.5])
    sd = np.array([0.4, 0.8, 1.6])
    s = 0
    y = np.zeros(T)
    for t in range(1, T):
        s = rng.choice(3, p=P[s])
        y[t] = mu[s] + 0.3 * y[t - 1] + sd[s] * rng.standard_normal()
    return y[:, None]


def _design(Y, p):
    T = Y.shape[0]
    Yd = Y[p:]
    if p == 0:
        return Yd, np.zeros((T, 0))
    X = np.column_stack([Y[p - k - 1: T - k - 1] for k in range(p)])
    return Yd, X


def _reference_loglik(Y, p, A, mu, Sigma, P, pi0):
    """Plain Hamilton (1994, ch. 22) filter, written independently of the
    module: log f(y_{p+1..T} | y_{1..p}) with P(s_{p+1}) = pi0."""
    Yd, X = _design(np.asarray(Y, float), p)
    K = len(mu)
    Te, n = Yd.shape
    logd = np.empty((Te, K))
    for k in range(K):
        r = Yd - mu[k] - X @ A.T
        Si = np.linalg.inv(Sigma[k])
        _, ld = np.linalg.slogdet(Sigma[k])
        logd[:, k] = -0.5 * (n * np.log(2 * np.pi) + ld
                             + np.einsum("ti,ij,tj->t", r, Si, r))
    pred = np.asarray(pi0, float)
    ll = 0.0
    filt = np.empty((Te, K))
    for t in range(Te):
        m = logd[t].max()
        joint = pred * np.exp(logd[t] - m)
        s = joint.sum()
        ll += m + np.log(s)
        filt[t] = joint / s
        pred = filt[t] @ P
    return ll, filt


def _decrease_tol(ll):
    return 1e-8 * max(1.0, abs(ll))


# ---------------------------------------------------------------------------
# 1) Monotone log-likelihood — public API only (fails on the pre-fix code)
# ---------------------------------------------------------------------------

def test_loglik_non_decreasing_across_iterations_public_api():
    """Refit with n_iter = 1..30 (tol=0 so no early stop): the k-th fit
    returns the k-th EM iterate, whose log-likelihood must never fall.
    Pre-fix: the log-likelihood peaks at iteration 12 and then falls."""
    Y = _simulate_msih(seed=1, n=1)
    lls = np.array([ms_var_fit(Y, K=2, p=1, n_iter=k, tol=0.0).loglik
                    for k in range(1, 31)])
    d = np.diff(lls)
    assert np.all(d >= -_decrease_tol(lls[-1])), (
        f"log-likelihood fell in {(d < 0).sum()} of {d.size} EM steps; "
        f"worst {d.min():.3e}")


# ---------------------------------------------------------------------------
# 2) Monotone log-likelihood on several DGPs via loglik_path
# ---------------------------------------------------------------------------

def _business_cycle_example():
    from puremacro.examples.ms_var_business_cycle import _simulate
    return _simulate()[0]


@pytest.mark.parametrize(
    "label, make, K, p",
    [
        ("msih-univariate-seed1", lambda: _simulate_msih(seed=1, n=1), 2, 1),
        ("msih-univariate-seed3", lambda: _simulate_msih(seed=3, n=1), 2, 1),
        ("msih-bivariate-seed5", lambda: _simulate_msih(seed=5, n=2), 2, 1),
        ("msih-bivariate-seed7", lambda: _simulate_msih(seed=7, n=2), 2, 1),
        ("three-regime-univariate", _simulate_k3, 3, 1),
        ("business-cycle-example", _business_cycle_example, 2, 1),
        ("bivariate-p2", lambda: _simulate_msih(seed=11, n=2), 2, 2),
    ],
)
def test_loglik_path_is_monotone(label, make, K, p):
    Y = make()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = ms_var_fit(Y, K=K, p=p, n_iter=400, tol=1e-9)
    path = np.asarray(res.loglik_path)
    assert path.shape == (res.n_iter,)
    assert path[-1] == pytest.approx(res.loglik, abs=0, rel=0)
    d = np.diff(path)
    assert np.all(d >= -_decrease_tol(res.loglik)), (
        f"{label}: log-likelihood fell in {(d < 0).sum()} steps, worst {d.min():.3e}")


def test_loglik_path_matches_refits():
    """loglik_path[k-1] is the log-likelihood of the n_iter=k fit."""
    Y = _simulate_msih(seed=7, n=2)
    long = ms_var_fit(Y, K=2, p=1, n_iter=12, tol=0.0)
    for k in (1, 2, 5, 12):
        short = ms_var_fit(Y, K=2, p=1, n_iter=k, tol=0.0)
        assert short.loglik == pytest.approx(long.loglik_path[k - 1], rel=1e-12)


# ---------------------------------------------------------------------------
# 3) The bivariate heteroskedastic DGP: pre-fix fell to -705.95
# ---------------------------------------------------------------------------

def test_bivariate_seed7_reaches_higher_likelihood_and_true_A():
    Y = _simulate_msih(seed=7, n=2)
    res = ms_var_fit(Y, K=2, p=1, n_iter=1000, tol=1e-9)
    assert res.converged
    # Pre-fix: final loglik -705.946 (path max -698.98); a correct ECM
    # reaches about -683.70 from the same start.
    assert res.loglik > -684.0
    A_true = np.array([[0.5, 0.1], [0.0, 0.4]])
    assert np.max(np.abs(res.A - A_true)) < 0.06


# ---------------------------------------------------------------------------
# 4) Reported loglik == loglik of the returned parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, p", [(1, 1), (2, 1), (2, 2)])
def test_reported_loglik_is_that_of_returned_parameters(n, p):
    Y = _simulate_msih(seed=5, n=n)
    res = ms_var_fit(Y, K=2, p=p, n_iter=60)
    ll_ref, filt_ref = _reference_loglik(Y, p, res.A, res.mu, res.Sigma,
                                         res.P, res.pi0)
    assert res.loglik == pytest.approx(ll_ref, abs=1e-8)
    np.testing.assert_allclose(res.filtered_probs, filt_ref, atol=1e-10)


# ---------------------------------------------------------------------------
# 5) The EM fixed point is a stationary point of the likelihood
# ---------------------------------------------------------------------------

def _score_A_mu(Y, p, res, h=1e-6):
    base = dict(A=res.A.copy(), mu=res.mu.copy())
    grads = []
    for name in ("A", "mu"):
        arr = base[name]
        for idx in np.ndindex(arr.shape):
            vals = []
            for sgn in (+1, -1):
                pert = {k: v.copy() for k, v in base.items()}
                pert[name][idx] += sgn * h
                vals.append(_reference_loglik(Y, p, pert["A"], pert["mu"],
                                              res.Sigma, res.P, res.pi0)[0])
            grads.append((vals[0] - vals[1]) / (2 * h))
    return np.array(grads)


@pytest.mark.parametrize("seed, n", [(1, 1), (7, 2)])
def test_em_fixed_point_is_stationary(seed, n):
    """Pre-fix, the score with respect to A at the reported optimum was
    O(1) (e.g. [-4.44, -6.55, -3.31, -3.76] on GNP data)."""
    Y = _simulate_msih(seed=seed, n=n)
    res = ms_var_fit(Y, K=2, p=1, n_iter=5000, tol=1e-11)
    g = _score_A_mu(Y, 1, res)
    assert np.max(np.abs(g)) < 1e-2, g


# ---------------------------------------------------------------------------
# 6) converged flag and the decrease check
# ---------------------------------------------------------------------------

def test_converged_only_when_loglik_change_below_tol():
    Y = _simulate_msih(seed=3, n=1)
    tol = 1e-6
    res = ms_var_fit(Y, K=2, p=1, n_iter=500, tol=tol)
    assert res.converged
    d = np.diff(res.loglik_path)
    assert -_decrease_tol(res.loglik) <= d[-1] < tol
    assert np.all(d[:-1] >= tol)          # no earlier step met the criterion
    short = ms_var_fit(Y, K=2, p=1, n_iter=3, tol=tol)
    assert not short.converged and short.n_iter == 3


def test_small_decrease_is_not_convergence():
    """Pre-fix, ``abs(ll - ll_prev) < tol`` declared convergence while the
    log-likelihood was falling. A fall larger than rounding noise must be
    flagged and must not count as convergence."""
    status = ms_mod._em_step_status
    assert status(-100.0, -100.0 - 1e-4, tol=1e-3) == (False, True)
    assert status(-100.0, -100.0 + 1e-4, tol=1e-3) == (True, False)
    assert status(-100.0, -100.0 - 1e-13, tol=1e-3) == (True, False)
    assert status(-100.0, -99.0, tol=1e-3) == (False, False)


def test_decrease_triggers_warning(monkeypatch):
    """Sabotage one M-step: the monotonicity check must warn."""
    real = ms_mod._m_step
    calls = {"n": 0}

    def bad_m_step(*args, **kwargs):
        out = dict(real(*args, **kwargs))
        calls["n"] += 1
        if calls["n"] == 4:
            out["mu"] = out["mu"] + 5.0
        return out

    monkeypatch.setattr(ms_mod, "_m_step", bad_m_step)
    Y = _simulate_msih(seed=1, n=1)
    with pytest.warns(RuntimeWarning, match="log-likelihood decreased"):
        res = ms_var_fit(Y, K=2, p=1, n_iter=8, tol=1e-12)
    d = np.diff(res.loglik_path)
    # The 4th M-step produces iterate 4 (path[3]); it must be below path[2].
    assert d[2] < -_decrease_tol(res.loglik_path[2])


# ---------------------------------------------------------------------------
# 7) statsmodels parity where the model classes coincide
# ---------------------------------------------------------------------------

def _rgnp():
    mod = pytest.importorskip(
        "statsmodels.tsa.regime_switching.tests.test_markov_autoregression")
    return np.asarray(mod.rgnp, float)


def _ergodic(P):
    K = P.shape[0]
    Amat = np.vstack([(np.eye(K) - P).T, np.ones(K)])
    return np.linalg.lstsq(Amat, np.r_[np.zeros(K), 1.0], rcond=None)[0]


def _sm_model(y, p):
    sm = pytest.importorskip("statsmodels.api")
    Yd, X = _design(np.asarray(y, float)[:, None], p)
    return sm.tsa.MarkovRegression(
        Yd[:, 0], k_regimes=2, trend="c", exog=X if p > 0 else None,
        switching_exog=False, switching_variance=True)


def _unpack_sm(prm, p):
    """statsmodels MarkovRegression params -> (P, mu, ar, sigma2), with
    regimes sorted by intercept and P[i, j] = Pr(s_t=j | s_{t-1}=i)."""
    P = np.array([[prm[0], 1 - prm[0]], [prm[1], 1 - prm[1]]])
    mu = np.asarray(prm[2:4])
    ar = np.asarray(prm[4:4 + p])
    s2 = np.asarray(prm[4 + p:6 + p])
    i = np.argsort(mu)
    return P[np.ix_(i, i)], mu[i], ar, s2[i]


def _unpack_ours(res):
    i = np.argsort(res.mu[:, 0])
    return (res.P[np.ix_(i, i)], res.mu[i, 0], res.A[0].copy(),
            res.Sigma[i, 0, 0])


def _pack_sm(P, mu, ar, s2):
    return np.r_[P[0, 0], P[1, 0], mu, ar, s2]


def _sm_mle(model, start):
    """statsmodels' own MLE, from its default start and from ``start``:
    the higher interior maximum (both variances > 1e-3). The MSIH
    likelihood is unbounded as a variance goes to zero, and statsmodels'
    random search can land there, so it is not used."""
    fits = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kw in ({}, {"start_params": start}):
            try:
                r = model.fit(disp=False, **kw)
            except (RuntimeError, np.linalg.LinAlgError):
                continue
            if np.isfinite(r.llf) and np.all(np.asarray(r.params)[-2:] > 1e-3):
                fits.append(r)
    assert fits, "statsmodels found no interior maximum"
    return max(fits, key=lambda r: r.llf)


def _compare_with_statsmodels(y, p, res, atol_P, atol_mu, atol_ar,
                              atol_s2, ll_gap):
    model = _sm_model(y, p)
    P_o, mu_o, ar_o, s2_o = _unpack_ours(res)
    ours = _pack_sm(P_o, mu_o, ar_o, s2_o)
    r = _sm_mle(model, ours)
    P_s, mu_s, ar_s, s2_s = _unpack_sm(np.asarray(r.params), p)

    # (a) Same objective: at statsmodels' MLE, the independent reference
    #     filter started from the ergodic distribution reproduces its llf.
    ll_ref, _ = _reference_loglik(
        np.asarray(y, float)[:, None], p, ar_s.reshape(1, p),
        mu_s[:, None], s2_s[:, None, None], P_s, _ergodic(P_s))
    assert ll_ref == pytest.approx(r.llf, abs=1e-6)
    # (b) A free initial distribution nests the ergodic one.
    assert res.loglik >= r.llf - 1e-6
    # (c) statsmodels' likelihood at our estimates is close to its maximum,
    #     and the parameters agree.
    assert model.loglike(ours) >= r.llf - ll_gap
    np.testing.assert_allclose(P_o, P_s, atol=atol_P)
    np.testing.assert_allclose(mu_o, mu_s, atol=atol_mu)
    np.testing.assert_allclose(ar_o, ar_s, atol=atol_ar)
    np.testing.assert_allclose(s2_o, s2_s, atol=atol_s2)
    return r


def _simulate_univariate(T, p, seed=123):
    rng = np.random.default_rng(seed)
    P = np.array([[0.95, 0.05], [0.10, 0.90]])
    mu, sd = np.array([1.0, -1.0]), np.array([0.5, 1.5])
    a = 0.5 if p else 0.0
    s, y = 0, np.zeros(T)
    for t in range(1, T):
        s = rng.choice(2, p=P[s])
        y[t] = mu[s] + a * y[t - 1] + sd[s] * rng.standard_normal()
    return y


@pytest.mark.parametrize("p", [0, 1])
def test_long_simulated_series_matches_statsmodels(p):
    """T=3000, so the initial-distribution convention (free pi0 here,
    ergodic in statsmodels) matters only at O(1/T): the MLEs agree to
    about 1e-3. p=0 is a switching-mean-and-variance model; p=1 is
    MSIH(2)-AR(1)."""
    y = _simulate_univariate(3000, p)
    res = ms_var_fit(y, K=2, p=p, n_iter=3000, tol=1e-10)
    assert res.converged
    _compare_with_statsmodels(y, p, res, atol_P=5e-3, atol_mu=5e-3,
                              atol_ar=5e-3, atol_s2=5e-3, ll_gap=5e-3)


@pytest.mark.parametrize("p", [0, 4])
def test_rgnp_msih_matches_statsmodels_markov_regression(p):
    """Hamilton's GNP growth (135 obs, from statsmodels' test module), K=2.
    For p=4 the model is MSIH(2)-AR(4): switching intercept and variance,
    shared AR. That is statsmodels MarkovRegression(trend='c', exog=4
    lags, switching_exog=False, switching_variance=True). For p=0 the
    intercept is the mean. With only 131-135 observations, the
    initial-distribution convention moves the estimates by a few
    hundredths, hence the wider tolerances than in the simulated test.

    Pre-fix (p=4): loglik -180.553 < statsmodels -179.328, and the
    "recession" regime had a positive intercept."""
    y = _rgnp()
    res = ms_var_fit(y[:, None], K=2, p=p, n_iter=3000, tol=1e-10)
    assert res.converged
    assert not np.any(np.diff(res.loglik_path) < -_decrease_tol(res.loglik))
    _compare_with_statsmodels(y, p, res, atol_P=0.03, atol_mu=0.06,
                              atol_ar=0.01, atol_s2=0.03, ll_gap=0.05)
    lo, hi = np.argsort(res.mu[:, 0])
    assert res.mu[lo, 0] < 0 < res.mu[hi, 0]
    if p == 4:
        # MLE of the free-pi0 MSIH-AR(4); statsmodels (ergodic pi0) -179.33.
        assert res.loglik > -179.2


def test_default_call_on_rgnp_reaches_the_mle():
    """Pre-fix, ms_var_fit(rgnp, K=2, p=4) stopped at the old 100-iteration
    cap with converged=False, loglik -180.4825, and a 'low' regime with a
    positive intercept (0.314). The corrected EM reaches the MLE of the
    free-pi0 MSIH-AR(4), -179.1567, in about 120 iterations from every
    seed. Two independent GLS-ECM implementations in the review found the
    same value."""
    y = _rgnp()[:, None]
    for seed in range(4):
        res = ms_var_fit(y, K=2, p=4, seed=seed)
        assert res.converged, seed
        assert res.loglik == pytest.approx(-179.1567, abs=2e-4), seed
        lo = int(np.argmin(res.mu[:, 0]))
        assert res.mu[lo, 0] < 0
        assert res.P[lo, lo] == pytest.approx(0.788, abs=0.005)


# ---------------------------------------------------------------------------
# 8) Units: the fit is equivariant to rescaling each variable
# ---------------------------------------------------------------------------

def _fit_quiet(Y, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        return ms_var_fit(Y, **kw)


def _assert_equivariant(base, scaled, c, p, T_eff):
    """``scaled`` is the fit of ``Y * c`` (c per column); ``base`` of Y.
    The MLE must map as mu -> C mu, A_i -> C A_i C^{-1},
    Sigma -> C Sigma C, and loglik -> loglik - T_eff * sum log|c|."""
    c = np.asarray(c, float)
    assert scaled.n_iter == base.n_iter
    np.testing.assert_allclose(scaled.mu / c, base.mu, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(scaled.A / c[:, None] * np.tile(c, p)[None, :],
                               base.A, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(scaled.Sigma / np.outer(c, c), base.Sigma,
                               rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(scaled.P, base.P, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(scaled.smoothed_probs, base.smoothed_probs,
                               atol=1e-8)
    shift = -T_eff * np.sum(np.log(np.abs(c)))
    assert scaled.loglik == pytest.approx(base.loglik + shift, abs=1e-7)
    np.testing.assert_allclose(scaled.loglik_path, base.loglik_path + shift,
                               atol=1e-7)


@pytest.mark.parametrize("c2", [1e5, 1e-4, -3e3])
def test_rescaling_one_variable_is_exactly_equivariant(c2):
    """Round-0 fix: with column 2 rescaled by 1e4-1e5, the fit lost
    135-664 log-likelihood points against the unit-adjusted optimum (the
    variance floor 1e-8 x average variance bound the small-scale variable);
    the pre-fix absolute ridge 1e-8*I failed at c2=1e-4 (-173 points)."""
    Y = _simulate_msih(T=300, seed=21, n=2)
    base = _fit_quiet(Y, K=2, p=1, n_iter=2000, tol=1e-9)
    c = np.array([1.0, c2])
    scaled = _fit_quiet(Y * c, K=2, p=1, n_iter=2000, tol=1e-9)
    _assert_equivariant(base, scaled, c, p=1, T_eff=Y.shape[0] - 1)


@pytest.mark.parametrize("K, p", [(2, 0), (3, 2)])
def test_rescaling_all_variables_is_equivariant(K, p):
    Y = _simulate_msih(T=400, seed=5, n=2)
    base = _fit_quiet(Y, K=K, p=p, n_iter=300, tol=1e-9)
    c = np.array([2.5e-6, 4e6])
    scaled = _fit_quiet(Y * c, K=K, p=p, n_iter=300, tol=1e-9)
    _assert_equivariant(base, scaled, c, p=p, T_eff=Y.shape[0] - p)


def _simulate_unit_scale(T=300, seed=21):
    rng = np.random.default_rng(seed)
    P = np.array([[0.95, 0.05], [0.10, 0.90]])
    mu = np.array([[0.5, 0.2], [-1.0, -0.4]])
    L = [np.diag([0.5, 0.25]), np.diag([1.5, 1.0])]
    A = np.array([[0.5, 0.1], [0.05, 0.4]])
    s, Y = 0, np.zeros((T, 2))
    for t in range(1, T):
        s = rng.choice(2, p=P[s])
        Y[t] = mu[s] + A @ Y[t - 1] + L[s] @ rng.standard_normal(2)
    return Y


def test_mixed_units_keep_the_variance_switching_of_a_small_variable():
    """GDP change in $bn (sd ~ 100) next to a rate change in decimals
    (sd ~ 0.003). Round-0 fix: both rate variances were floored at the
    same 5.75e-5 (true fit 8.75e-6 and 4.79e-7), and the log-likelihood
    was 422 points below the optimum."""
    Z = _simulate_unit_scale()
    c = np.array([100.0, 0.003])
    fz = _fit_quiet(Z, K=2, p=1, n_iter=2000, tol=1e-9)
    fy = _fit_quiet(Z * c, K=2, p=1, n_iter=2000, tol=1e-9)
    _assert_equivariant(fz, fy, c, p=1, T_eff=Z.shape[0] - 1)
    rate_var = np.sort(fy.Sigma[:, 1, 1])
    assert rate_var[1] / rate_var[0] > 10.0
    assert not fy.sigma_at_floor.any()


# ---------------------------------------------------------------------------
# 9) A regime collapsing onto the variance floor is reported
# ---------------------------------------------------------------------------

def test_collapsed_regime_is_flagged_and_warned():
    """A series with a flat stretch (a rate stuck at a floor): one regime
    collapses onto the flat observations, where the MSIH likelihood is
    unbounded. Its variance sits at the floor 1e-8 x the pooled residual
    variance, and the fit must say so."""
    rng = np.random.default_rng(0)
    e = rng.standard_normal(80)
    y = np.r_[np.zeros(120), 1.0 + 0.5 * e]
    with pytest.warns(RuntimeWarning, match=r"regime\(s\) \[0\].*variance floor"):
        res = ms_var_fit(y, K=2, p=0)
    assert res.sigma_at_floor.tolist() == [True, False]
    pooled = np.mean((y - y.mean()) ** 2)
    assert res.Sigma[0, 0, 0] == pytest.approx(1e-8 * pooled, rel=1e-6)
    assert res.Sigma[1, 0, 0] > 0.1


def test_interior_fit_has_no_regime_at_the_floor():
    res = _fit_quiet(_simulate_msih(seed=1, n=1), K=2, p=1)
    assert res.sigma_at_floor.dtype == bool
    assert res.sigma_at_floor.shape == (2,)
    assert not res.sigma_at_floor.any()


# ---------------------------------------------------------------------------
# 10) Input validation
# ---------------------------------------------------------------------------

def test_overparameterised_sample_is_refused():
    """T=6, n=2, p=2, K=2: 4 observations for 6 coefficients per equation.
    Round-0 fix: ~250 'invalid value in sqrt' warnings and loglik -3e275."""
    Y = np.random.default_rng(0).standard_normal((6, 2))
    with pytest.raises(ValueError, match=r"T - p > K \+ n\*p"):
        ms_var_fit(Y, K=2, p=2)
    with pytest.raises(ValueError, match=r"T - p > K \+ n\*p"):
        ms_var_fit(np.random.default_rng(1).standard_normal(3), K=2, p=1)


def test_smallest_allowed_sample_runs_without_numerical_warnings():
    """T - p = K + n*p + 1: the fit may collapse (floor warning), but it
    must stay finite and monotone, with no numpy 'invalid value' noise."""
    for seed in range(3):
        Y = np.random.default_rng(seed).standard_normal((2 + 2 + 2 * 2 + 1, 2))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            res = ms_var_fit(Y, K=2, p=2, n_iter=200)
        msgs = [str(w.message) for w in rec]
        assert all("variance floor" in m for m in msgs), msgs
        assert np.isfinite(res.loglik)
        d = np.diff(res.loglik_path)
        assert np.all(d >= -_decrease_tol(res.loglik))


@pytest.mark.parametrize("p", [0, 1])
def test_constant_or_zero_variable_is_refused(p):
    rng = np.random.default_rng(0)
    Y = np.column_stack([rng.standard_normal(100), np.full(100, 2.5)])
    with pytest.raises(ValueError, match="zero residual variance"):
        ms_var_fit(Y, K=2, p=p)
    Y[:, 1] = 0.0
    with pytest.raises(ValueError, match="identically zero"):
        ms_var_fit(Y, K=2, p=p)
