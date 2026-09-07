"""Spatial fixed-effects panels: closed-form identities, reductions to
estimators already in the package, invariances, planted-parameter recovery and
the adversarial degenerate cases.

The tests that matter most here pin the two things the design flagged as most
likely to be got wrong:

* the pair (Jacobian factor, variance divisor) must move together --- pinned
  against an *independent* reference that materialises the Lee-Yu ``F``
  matrices explicitly, and by the numerical gradient of the concentrated
  log-likelihood at the reported optimum;
* the Lee-Yu bias under individual effects is ``(0_k, 0, -sigma^2/(T-1))``,
  never ``-sigma^2``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from dataclasses import FrozenInstanceError
from scipy import optimize

from puremacro.inference.dk import driscoll_kraay
from puremacro.lp._panel_helpers import two_way_fe_within
from puremacro.spatial import SpatialWeights, contiguity_weights
from puremacro.spatial.models import spatial_effects
from puremacro.spatial.panel import (
    SpatialPanelResult,
    _hac_meat_from_scores,
    _psd_draws,
    spatial_panel,
)


# ---------------------------------------------------------------------------
# fixtures / DGPs
# ---------------------------------------------------------------------------
def rook(side: int, *, standardize: bool = True) -> SpatialWeights:
    """``side x side`` rook-contiguity lattice, row-standardised by default."""
    nb = {}
    for r in range(side):
        for c in range(side):
            nb[f"u{r * side + c:02d}"] = [
                f"u{rr * side + cc:02d}"
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                if 0 <= rr < side and 0 <= cc < side
            ]
    return contiguity_weights(nb, row_standardize=standardize)


def sar_panel(
    W: SpatialWeights,
    T: int,
    *,
    rho: float = 0.5,
    beta: tuple[float, ...] = (1.0, -0.5),
    sigma: float = 1.0,
    seed: int = 0,
    ent: bool = True,
    tim: bool = True,
) -> pd.DataFrame:
    """Balanced SAR panel ``y = S^-1 (X b + mu + xi + eps)``."""
    rng = np.random.default_rng(seed)
    n = W.n
    S_inv = np.linalg.inv(np.eye(n) - rho * W.to_dense())
    X = [rng.standard_normal((n, T)) for _ in beta]
    M = sum(b * xx for b, xx in zip(beta, X))
    if ent:
        M = M + rng.standard_normal((n, 1))
    if tim:
        M = M + rng.standard_normal((1, T))
    Y = S_inv @ (M + sigma * rng.standard_normal((n, T)))
    idx = pd.MultiIndex.from_product([list(W.ids), range(T)], names=["code", "date"])
    data = {"y": Y.ravel()}
    for j, xx in enumerate(X):
        data[f"x{j}"] = xx.ravel()
    return pd.DataFrame(data, index=idx)


def sem_panel(
    W: SpatialWeights,
    T: int,
    *,
    lam: float = 0.5,
    beta: tuple[float, ...] = (1.0, -0.5),
    seed: int = 0,
    ent: bool = True,
    tim: bool = True,
) -> pd.DataFrame:
    """Balanced SEM panel ``y = X b + mu + xi + S^-1 eps``."""
    rng = np.random.default_rng(seed)
    n = W.n
    S_inv = np.linalg.inv(np.eye(n) - lam * W.to_dense())
    X = [rng.standard_normal((n, T)) for _ in beta]
    Y = sum(b * xx for b, xx in zip(beta, X))
    if ent:
        Y = Y + rng.standard_normal((n, 1))
    if tim:
        Y = Y + rng.standard_normal((1, T))
    Y = Y + S_inv @ rng.standard_normal((n, T))
    idx = pd.MultiIndex.from_product([list(W.ids), range(T)], names=["code", "date"])
    data = {"y": Y.ravel()}
    for j, xx in enumerate(X):
        data[f"x{j}"] = xx.ravel()
    return pd.DataFrame(data, index=idx)


X2 = ["x0", "x1"]


def _blocks(df: pd.DataFrame, W: SpatialWeights, col: str) -> np.ndarray:
    return (df[col].unstack("date").reindex(index=list(W.ids)).to_numpy(float))


def _demean(A: np.ndarray, ent: bool, tim: bool) -> np.ndarray:
    B = A.copy()
    if ent:
        B -= B.mean(axis=1, keepdims=True)
    if tim:
        B -= B.mean(axis=0, keepdims=True)
    return B


def _double_centre(A: np.ndarray) -> np.ndarray:
    return A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()


def _profile_quadratic(df, W, cols, *, ent, tim):
    """``(a, b, c)`` with ``SSR(rho) = a - 2 rho b + rho^2 c``, built here from
    scratch: two OLS passes on the demeaned data, never through the estimator."""
    Wd = W.to_dense()
    yv = _demean(_blocks(df, W, "y"), ent, tim).ravel(order="F")
    wyv = _demean(Wd @ _blocks(df, W, "y"), ent, tim).ravel(order="F")
    Xm = np.column_stack([
        _demean(_blocks(df, W, c), ent, tim).ravel(order="F") for c in cols])
    P = Xm @ np.linalg.solve(Xm.T @ Xm, Xm.T)
    e_O, e_L = yv - P @ yv, wyv - P @ wyv
    return float(e_O @ e_O), float(e_O @ e_L), float(e_L @ e_L)


def _ll_ref(abc, ev, jac_factor, n_div, r):
    """The concentrated log-likelihood, written out independently."""
    a, b, c = abc
    ssr = a - 2.0 * r * b + r * r * c
    return (-0.5 * n_div * (np.log(2 * np.pi) + 1.0) - 0.5 * n_div * np.log(ssr / n_div)
            + jac_factor * float(np.sum(np.log(np.abs(1.0 - r * ev)))))


def _sar_scores(res, df, W, cols, *, ent, tim):
    """The (nT, k+2) ML score matrix, rebuilt from the reported estimates."""
    n, T, k = W.n, res.n_periods, len(cols)
    Wd = W.to_dense()
    r, s2 = res.rho, res.sigma2
    beta = res.beta.to_numpy(float)
    y_dd = _demean(_blocks(df, W, "y"), ent, tim).ravel(order="F")
    wy_dd = _demean(Wd @ _blocks(df, W, "y"), ent, tim).ravel(order="F")
    X_dd = np.column_stack([
        _demean(_blocks(df, W, c), ent, tim).ravel(order="F") for c in cols])
    e = y_dd - r * wy_dd - X_dd @ beta
    G = Wd @ np.linalg.inv(np.eye(n) - r * Wd)
    A = _double_centre(G) if tim else G
    gamma = np.diag(A)
    jac = float(T - 1 if ent else T)
    n_div = float((n - 1 if tim else n) * (T - 1 if ent else T))
    sc = np.empty((n * T, k + 2))
    sc[:, :k] = X_dd * (e[:, None] / s2)
    sc[:, k] = wy_dd * e / s2 - (jac / T) * np.tile(gamma, T)
    sc[:, k + 1] = -(n_div / (n * T)) / (2.0 * s2) + e ** 2 / (2.0 * s2 ** 2)
    return sc, X_dd, e


# ---------------------------------------------------------------------------
# 1. reduction to estimators already in the package
# ---------------------------------------------------------------------------
def test_rho_zero_reduces_to_two_way_fe_within():
    """At rho = 0 the concentrated SAR likelihood collapses to the Gaussian FE
    likelihood, whose MLE is the two-way within estimator. Pins the demeaning,
    the F-order stacking and the two-OLS concentration in one shot."""
    W = rook(6)
    df = sar_panel(W, 10, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_panel(df, "y", X2, W, effects="two-way",
                            rho_bounds=(-1e-10, 1e-10), impacts=False)
    ref = two_way_fe_within(df, y_col="y", x_cols=X2)
    assert abs(res.rho) < 1e-9
    assert np.max(np.abs(res.beta.to_numpy() - ref["beta"])) < 1e-8


def test_transformation_matches_an_explicit_lee_yu_F_reference():
    """Independent reference: build ``F_{n,n-1}`` and ``F_{T,T-1}`` explicitly
    from the SVD of the centring projectors, transform the data, and maximise
    the (n-1)x(T-1) SAR likelihood directly.

    This is the test that catches a mismatched (Jacobian factor, divisor) pair
    --- the single most likely implementation bug. Using divisor n(T-1) with
    Jacobian factor T (a natural-looking half-fix) moves rho_hat by ~1e-2.
    """
    W = rook(6)
    T = 9
    df = sar_panel(W, T, seed=11)
    n = W.n
    Wd = W.to_dense()

    def F_of(m):
        U, _s, _v = np.linalg.svd(np.eye(m) - np.ones((m, m)) / m)
        return U[:, : m - 1]

    Fn, Ft = F_of(n), F_of(T)
    Y = _blocks(df, W, "y")
    Xs = [_blocks(df, W, c) for c in X2]
    Ws = Fn.T @ Wd @ Fn
    Yt = Fn.T @ Y @ Ft
    Xt = [Fn.T @ xx @ Ft for xx in Xs]
    ev = np.linalg.eigvals(Ws)                     # W is row-standardised: NOT symmetric
    Xm = np.column_stack([xx.ravel(order="F") for xx in Xt])
    yv = Yt.ravel(order="F")
    wyv = (Ws @ Yt).ravel(order="F")
    XtXi = np.linalg.inv(Xm.T @ Xm)
    b_O, b_L = XtXi @ Xm.T @ yv, XtXi @ Xm.T @ wyv
    e_O, e_L = yv - Xm @ b_O, wyv - Xm @ b_L
    a, b, c = e_O @ e_O, e_O @ e_L, e_L @ e_L
    N = (n - 1) * (T - 1)

    def ll(r):
        s = a - 2 * r * b + r * r * c
        return (-0.5 * N * (np.log(2 * np.pi) + 1) - 0.5 * N * np.log(s / N)
                + (T - 1) * np.sum(np.log(np.abs(1 - r * ev))))

    opt = optimize.minimize_scalar(lambda r: -ll(r), bounds=(-0.99, 0.99),
                                   method="bounded", options={"xatol": 1e-11})
    rho_ref = float(opt.x)
    beta_ref = b_O - rho_ref * b_L
    sigma2_ref = (a - 2 * rho_ref * b + rho_ref ** 2 * c) / N

    res = spatial_panel(df, "y", X2, W, effects="two-way", impacts=False)
    assert abs(res.rho - rho_ref) < 1e-6
    assert np.max(np.abs(res.beta.to_numpy() - beta_ref)) < 1e-7
    assert abs(res.sigma2 - sigma2_ref) < 1e-6
    assert abs(res.loglik - ll(rho_ref)) < 1e-8


def test_concentrated_score_is_zero_at_the_reported_optimum():
    """The ANALYTIC derivative of an independently written concentrated
    log-likelihood vanishes at the reported rho_hat.

    Nothing here re-enters the estimator: ``(a, b, c)`` come from two OLS
    passes done in this file and the Jacobian from ``numpy.linalg.eigvals`` of
    the dense W with one unit eigenvalue removed (the row-standardised rook
    lattice is *not* symmetric, so the spectrum is complex). The same score
    evaluated with the ``method='direct'`` pair ``(T, nT)`` on the same
    transformation-demeaned data --- the classic half-fix --- is five orders of
    magnitude larger, which is what makes the bound meaningful.
    """
    W = rook(6)
    T = 12
    df = sar_panel(W, T, seed=3)
    n = W.n
    res = spatial_panel(df, "y", X2, W, effects="two-way", impacts=False)

    a, b, c = _profile_quadratic(df, W, X2, ent=True, tim=True)
    ev = np.linalg.eigvals(W.to_dense())
    ev1 = np.delete(ev, int(np.argmin(np.abs(ev - 1.0))))
    r = res.rho
    ssr = a - 2 * r * b + r * r * c
    dssr = -2 * b + 2 * r * c
    dJ = float(np.real(np.sum(-ev1 / (1.0 - r * ev1))))

    grad = -0.5 * ((n - 1) * (T - 1)) * dssr / ssr + (T - 1) * dJ
    grad_direct_pair = -0.5 * (n * T) * dssr / ssr + T * dJ
    assert abs(grad) < 1e-4                       # measured 4.8e-06
    assert abs(grad_direct_pair) > 1.0            # measured 1.53
    assert abs(grad_direct_pair) > 1e4 * abs(grad)
    # the independent objective reproduces the reported log-likelihood exactly
    assert abs(res.loglik - _ll_ref((a, b, c), ev1, T - 1.0, (n - 1) * (T - 1.0), r)) < 1e-9
    # ... and really does peak at rho_hat
    for d in (-0.02, 0.02):
        assert _ll_ref((a, b, c), ev1, T - 1.0, (n - 1) * (T - 1.0), r + d) < res.loglik


def test_sar_information_matrix_matches_an_independent_reconstruction():
    """``I[k,k] = T* (tr A^2 + tr A'A) + m'm/sigma^2`` and
    ``I[:k,k] = X_dd' m_dd / sigma^2`` --- the two most error-prone lines in
    the module --- rebuilt here from scratch and compared with ``inv(vcov)``.

    Both terms are load-bearing, and the test says so numerically: dropping
    the ``m'm/sigma^2`` quadratic term inflates the reported rho standard
    error by 30%, and zeroing the beta-rho cross block moves it by 1.5%.
    ``m_dd`` is the doubly-demeaned systematic spatial lag
    ``W S^-1 (X beta)``; note it is built from the RAW blocks and demeaned
    afterwards, because ``J_n S(rho) != S(rho) J_n``.
    """
    W = rook(6)
    T = 12
    df = sar_panel(W, T, seed=3)
    n, k = W.n, 2
    res = spatial_panel(df, "y", X2, W, effects="two-way", impacts=False)
    Wd = W.to_dense()
    r, s2 = res.rho, res.sigma2
    beta = res.beta.to_numpy(float)

    X_dd = np.column_stack([
        _demean(_blocks(df, W, c), True, True).ravel(order="F") for c in X2])
    S_inv = np.linalg.inv(np.eye(n) - r * Wd)
    G = Wd @ S_inv
    A = _double_centre(G)
    M = sum(bb * _blocks(df, W, c) for bb, c in zip(beta, X2))
    m_dd = _demean(Wd @ (S_inv @ M), True, True).ravel(order="F")
    T_star, N_star = T - 1.0, (n - 1) * (T - 1.0)

    I = np.zeros((k + 2, k + 2))
    I[:k, :k] = X_dd.T @ X_dd / s2
    I[:k, k] = I[k, :k] = X_dd.T @ m_dd / s2
    I[k, k] = T_star * (np.sum(A * A.T) + np.sum(A * A)) + float(m_dd @ m_dd) / s2
    I[k, k + 1] = I[k + 1, k] = T_star * np.trace(A) / s2
    I[k + 1, k + 1] = N_star / (2.0 * s2 * s2)

    got = np.linalg.inv(res.vcov)
    assert np.max(np.abs(I - got)) < 1e-9 * np.max(np.abs(got))   # measured 2e-16

    se_rho = float(np.sqrt(np.linalg.inv(I)[k, k]))
    assert abs(res.se["rho"] - se_rho) < 1e-12
    I_no_quad = I.copy()
    I_no_quad[k, k] -= float(m_dd @ m_dd) / s2
    assert float(np.sqrt(np.linalg.inv(I_no_quad)[k, k])) > 1.25 * se_rho
    I_no_cross = I.copy()
    I_no_cross[:k, k] = I_no_cross[k, :k] = 0.0
    assert abs(float(np.sqrt(np.linalg.inv(I_no_cross)[k, k])) / se_rho - 1.0) > 0.01


# ---------------------------------------------------------------------------
# 2. recovery of planted parameters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("effects", ["individual", "time", "two-way"])
def test_sar_dgp_recovers_rho_and_beta(effects):
    W = rook(8)
    ent = effects in ("individual", "two-way")
    tim = effects in ("time", "two-way")
    df = sar_panel(W, 30, rho=0.5, seed=0, ent=ent, tim=tim)
    res = spatial_panel(df, "y", X2, W, effects=effects, impacts=False)
    assert abs(res.rho - 0.5) < 0.04
    assert np.max(np.abs(res.beta.to_numpy() - np.array([1.0, -0.5]))) < 0.06
    assert res.converged


def test_sem_panel_recovers_lambda_and_has_no_impacts():
    W = rook(8)
    df = sem_panel(W, 30, lam=0.5, seed=4)
    res = spatial_panel(df, "y", X2, W, model="sem", effects="two-way")
    assert abs(res.lam - 0.5) < 0.06
    assert np.isnan(res.rho)
    assert np.max(np.abs(res.beta.to_numpy() - np.array([1.0, -0.5]))) < 0.06
    assert res.impacts is None
    assert res.impacts_frame().empty
    assert "no spatial multiplier" in res.summary()


# ---------------------------------------------------------------------------
# 3. direct vs transformation, and the Lee-Yu correction
# ---------------------------------------------------------------------------
def test_direct_and_transformation_agree_on_beta_and_rho_under_individual_effects():
    """The two concentrated objectives differ by the positive affine map
    ``loglik* = ((T-1)/T)(loglik_direct + const)``, so their rho arguments and
    beta coincide; only the variance divisor differs, by exactly T/(T-1)."""
    W = rook(6)
    T = 10
    df = sar_panel(W, T, seed=2, tim=False)
    tr = spatial_panel(df, "y", X2, W, effects="individual",
                       method="transformation", impacts=False)
    di = spatial_panel(df, "y", X2, W, effects="individual", method="direct",
                       bias_correction="none", impacts=False)
    assert abs(tr.rho - di.rho) < 1e-6              # optimiser tolerance (xatol 1e-8)
    assert np.max(np.abs(tr.beta.to_numpy() - di.beta.to_numpy())) < 1e-6
    assert di.sigma2 / tr.sigma2 == pytest.approx((T - 1) / T, rel=1e-6)
    assert tr.n_eff == W.n * (T - 1) and di.n_eff == W.n * (T - 1)
    assert tr.bias is None and tr.bias_correction == "none"


def test_lee_yu_bias_vector_is_sigma2_only_and_of_order_sigma2_over_T_minus_1():
    """``solve(I, a) == (0_k, 0, -sigma^2/(T-1))`` exactly --- NOT ``-sigma^2``
    (a/I carries the units of the parameter, so the correction is O(1/T))."""
    W = rook(6)
    T = 10
    df = sar_panel(W, T, seed=2, tim=False, beta=(1.0, -0.5, 0.3))
    cols = ["x0", "x1", "x2"]
    raw = spatial_panel(df, "y", cols, W, effects="individual", method="direct",
                        bias_correction="none", impacts=False)
    cor = spatial_panel(df, "y", cols, W, effects="individual", method="direct",
                        bias_correction="lee-yu", impacts=False)
    bias = cor.bias.to_numpy()
    assert cor.bias_correction == "lee-yu"
    assert np.max(np.abs(bias[:-1])) < 1e-10                   # beta and rho untouched
    assert bias[-1] == pytest.approx(-raw.sigma2 / (T - 1), rel=1e-10)
    assert abs(bias[-1] + raw.sigma2) > 0.1 * raw.sigma2       # NOT -sigma^2
    # the corrected sigma2 is exactly the transformation divisor
    assert cor.sigma2 == pytest.approx(raw.sigma2 * T / (T - 1), rel=1e-12)
    tr = spatial_panel(df, "y", cols, W, effects="individual", impacts=False)
    assert cor.sigma2 == pytest.approx(tr.sigma2, rel=1e-6)
    assert np.max(np.abs(cor.beta.to_numpy() - raw.beta.to_numpy())) == 0.0
    assert cor.rho == raw.rho


@pytest.mark.parametrize("T", [5, 10, 20])
def test_individual_effects_sigma2_divisor_identity(T):
    """``sigma2_direct / sigma2_transformation == (T-1)/T`` --- an algebraic
    identity, limited only by the shared rho optimiser tolerance."""
    W = rook(5)
    df = sar_panel(W, T, seed=T, tim=False)
    tr = spatial_panel(df, "y", X2, W, effects="individual", impacts=False)
    di = spatial_panel(df, "y", X2, W, effects="individual", method="direct",
                       bias_correction="none", impacts=False)
    assert di.sigma2 / tr.sigma2 == pytest.approx((T - 1) / T, rel=1e-6)


@pytest.mark.parametrize("T", [10, 20])
def test_lee_yu_correction_shrinks_the_two_way_rho_bias(T):
    """The uncorrected two-way rho bias is negative and does not fall with T
    (it is the O(1/n) time-effects term); the correction removes most of it."""
    W = rook(6)
    reps = 100
    rho0 = 0.5
    unc, cor = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for s in range(reps):
            df = sar_panel(W, T, rho=rho0, beta=(1.0,), seed=s)
            unc.append(spatial_panel(df, "y", "x0", W, effects="two-way",
                                     method="direct", bias_correction="none",
                                     impacts=False).rho)
            cor.append(spatial_panel(df, "y", "x0", W, effects="two-way",
                                     method="direct", bias_correction="lee-yu",
                                     impacts=False).rho)
    b_unc = float(np.mean(unc)) - rho0
    b_cor = float(np.mean(cor)) - rho0
    assert b_unc < -0.02
    assert abs(b_cor) < 0.4 * abs(b_unc)


def test_uncorrected_two_way_rho_bias_scales_like_one_over_n():
    """Pin the ORDER, not the level: n * bias is roughly constant in n."""
    rho0 = 0.5
    reps = 80
    scaled = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for side in (4, 6, 8):
            W = rook(side)
            vals = [
                spatial_panel(sar_panel(W, 10, rho=rho0, beta=(1.0,), seed=s),
                              "y", "x0", W, effects="two-way", method="direct",
                              bias_correction="none", impacts=False).rho
                for s in range(reps)
            ]
            bias = float(np.mean(vals)) - rho0
            assert bias < 0.0, f"bias at n={W.n} is {bias}"
            scaled[W.n] = W.n * bias
    lo, hi = min(scaled.values()), max(scaled.values())
    assert hi / lo < 1.6, scaled


def test_sem_direct_and_transformation_agree_under_individual_effects():
    W = rook(6)
    T = 10
    df = sem_panel(W, T, lam=0.5, seed=5, tim=False)
    tr = spatial_panel(df, "y", X2, W, model="sem", effects="individual",
                       impacts=False)
    di = spatial_panel(df, "y", X2, W, model="sem", effects="individual",
                       method="direct", bias_correction="none", impacts=False)
    assert abs(tr.lam - di.lam) < 1e-6
    assert di.sigma2 / tr.sigma2 == pytest.approx((T - 1) / T, rel=1e-6)


def test_transformation_with_forced_lee_yu_raises():
    W = rook(5)
    df = sar_panel(W, 8, seed=6)
    with pytest.raises(ValueError, match="double-correct"):
        spatial_panel(df, "y", X2, W, method="transformation",
                      bias_correction="lee-yu")


# ---------------------------------------------------------------------------
# 4. the SEM information block (the filtered design)
# ---------------------------------------------------------------------------
def test_sem_beta_information_uses_the_spatially_filtered_design():
    """``I_bb = Xs'Xs/sigma2`` with ``Xs = (I - lam W) X``, not ``X'X/sigma2``.

    The SEM information is block diagonal between beta and (lambda, sigma2),
    so the reported beta covariance must equal ``inv(Xs'Xs/sigma2)`` exactly.
    """
    W = rook(6)
    df = sem_panel(W, 12, lam=0.6, seed=7)
    res = spatial_panel(df, "y", X2, W, model="sem", effects="two-way",
                        impacts=False)
    Wd = W.W
    Xd = np.column_stack([
        _demean(_blocks(df, W, c), True, True).ravel(order="F") for c in X2])
    WXd = np.column_stack([
        _demean(Wd @ _blocks(df, W, c), True, True).ravel(order="F") for c in X2])
    Xs = Xd - res.lam * WXd
    V_filtered = np.linalg.inv(Xs.T @ Xs / res.sigma2)
    V_raw = np.linalg.inv(Xd.T @ Xd / res.sigma2)
    got = res.vcov[:2, :2]
    assert np.max(np.abs(got - V_filtered)) < 1e-10 * max(1.0, np.max(np.abs(got)))
    # and the unfiltered design would have been materially different
    assert np.max(np.abs(V_raw - V_filtered)) > 0.02 * np.max(np.abs(V_filtered))


def test_sem_lags_every_regressor_not_only_the_durbin_subset():
    """The default SEM call (durbin=False) must still form ``W x`` for EVERY
    regressor, not for a Durbin subset.

    Pinned on the estimate itself, not on shapes: ``beta_hat`` must equal the
    GLS coefficient of the fully filtered design ``Xs = (I - lam W) X`` at the
    reported lambda, and must NOT equal the one that filters only the first
    regressor --- the failure mode a lag-the-Durbin-subset implementation
    produces. The same for the reported beta covariance.
    """
    W = rook(5)
    T = 10
    df = sem_panel(W, T, seed=8)
    res = spatial_panel(df, "y", X2, W, model="sem", effects="individual",
                        impacts=False)
    assert list(res.params.index) == ["x0", "x1", "lambda", "sigma2"]
    Wd = W.to_dense()
    lam = res.lam
    yv = _demean(_blocks(df, W, "y"), True, False).ravel(order="F")
    wyv = _demean(Wd @ _blocks(df, W, "y"), True, False).ravel(order="F")
    Xd = np.column_stack([
        _demean(_blocks(df, W, c), True, False).ravel(order="F") for c in X2])
    WXd = np.column_stack([
        _demean(Wd @ _blocks(df, W, c), True, False).ravel(order="F") for c in X2])
    ys = yv - lam * wyv

    Xs_all = Xd - lam * WXd
    beta_all = np.linalg.solve(Xs_all.T @ Xs_all, Xs_all.T @ ys)
    partial = WXd.copy()
    partial[:, 1] = 0.0                      # lag only x0, the "Durbin subset" bug
    Xs_one = Xd - lam * partial
    beta_one = np.linalg.solve(Xs_one.T @ Xs_one, Xs_one.T @ ys)

    got = res.beta.to_numpy(float)
    assert np.max(np.abs(got - beta_all)) < 1e-10
    assert np.max(np.abs(beta_one - beta_all)) > 1e-3
    V_all = np.linalg.inv(Xs_all.T @ Xs_all / res.sigma2)
    V_one = np.linalg.inv(Xs_one.T @ Xs_one / res.sigma2)
    assert np.max(np.abs(res.vcov[:2, :2] - V_all)) < 1e-10 * np.max(np.abs(V_all))
    assert np.max(np.abs(V_one - V_all)) > 0.01 * np.max(np.abs(V_all))
    assert np.all(np.isfinite(res.se.to_numpy()))


# ---------------------------------------------------------------------------
# 5. impacts
# ---------------------------------------------------------------------------
def test_impacts_match_the_dense_multiplier_and_the_closed_form():
    W = rook(6)
    df = sar_panel(W, 12, seed=9)
    res = spatial_panel(df, "y", X2, W, model="sdm", effects="two-way",
                        impacts=True, n_draws=200)
    n = W.n
    Wd = W.to_dense()
    S_inv = np.linalg.inv(np.eye(n) - res.rho * Wd)
    ell = np.ones(n)
    for c in X2:
        b = float(res.params[c])
        th = float(res.params[f"W.{c}"])
        Sr = S_inv @ (b * np.eye(n) + th * Wd)
        assert res.impacts.loc[c, "direct"] == pytest.approx(np.trace(Sr) / n, abs=1e-12)
        assert res.impacts.loc[c, "total"] == pytest.approx(ell @ Sr @ ell / n, abs=1e-12)
        # row-standardised W => closed form
        assert res.impacts.loc[c, "total"] == pytest.approx((b + th) / (1 - res.rho),
                                                            abs=1e-12)
        assert res.impacts.loc[c, "indirect"] == pytest.approx(
            res.impacts.loc[c, "total"] - res.impacts.loc[c, "direct"], abs=1e-12)


def test_impacts_match_the_dense_multiplier_for_an_unstandardised_W():
    """The partial-fraction route (no row standardisation, no truncated power
    series --- a series in rho^q tr(W^q) overflows here)."""
    W = rook(5, standardize=False)
    df = sar_panel(W, 10, rho=0.15, seed=10, tim=False)
    res = spatial_panel(df, "y", X2, W, effects="individual", impacts=True,
                        n_draws=150)
    n = W.n
    S_inv = np.linalg.inv(np.eye(n) - res.rho * W.to_dense())
    ell = np.ones(n)
    for c in X2:
        b = float(res.params[c])
        assert res.impacts.loc[c, "direct"] == pytest.approx(b * np.trace(S_inv) / n,
                                                             abs=1e-10)
        assert res.impacts.loc[c, "total"] == pytest.approx(b * (ell @ S_inv @ ell) / n,
                                                            abs=1e-10)


def test_impacts_are_deterministic_in_the_seed():
    W = rook(6)
    df = sar_panel(W, 10, seed=20)
    kw = dict(effects="two-way", impacts=True, n_draws=250)
    a = spatial_panel(df, "y", X2, W, seed=0, **kw)
    b = spatial_panel(df, "y", X2, W, seed=0, **kw)
    c = spatial_panel(df, "y", X2, W, seed=7, **kw)
    assert np.max(np.abs(a.impacts.to_numpy() - b.impacts.to_numpy())) == 0.0
    assert np.max(np.abs(a.impacts.to_numpy() - c.impacts.to_numpy())) > 0.0
    assert a.n_rejected == 0
    # the plug-in point always lies inside its own interval
    for c_ in X2:
        for key in ("direct", "indirect", "total"):
            assert a.impacts.loc[c_, f"{key}_lo"] <= a.impacts.loc[c_, key] <= \
                   a.impacts.loc[c_, f"{key}_hi"]


def test_sdm_equals_sar_with_wx_appended_and_durbin_subset_is_named():
    W = rook(5)
    df = sar_panel(W, 10, seed=12)
    a = spatial_panel(df, "y", X2, W, model="sdm", effects="two-way", impacts=False)
    b = spatial_panel(df, "y", X2, W, model="sar", durbin=True, effects="two-way",
                      impacts=False)
    assert list(a.params.index) == ["x0", "x1", "W.x0", "W.x1", "rho", "sigma2"]
    assert np.max(np.abs(a.params.to_numpy() - b.params.to_numpy())) == 0.0
    c = spatial_panel(df, "y", X2, W, durbin=["x0"], effects="two-way", impacts=False)
    assert list(c.params.index) == ["x0", "x1", "W.x0", "rho", "sigma2"]


# ---------------------------------------------------------------------------
# 6. covariances
# ---------------------------------------------------------------------------
def _coords_for(W: SpatialWeights, side: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"lat": [float(i // side) for i in range(W.n)],
         "lon": [float(i % side) for i in range(W.n)]},
        index=list(W.ids),
    )


def test_robust_vcov_touches_the_beta_block_only():
    W = rook(6)
    df = sar_panel(W, 14, seed=13)
    coords = _coords_for(W, 6)
    base = spatial_panel(df, "y", X2, W, effects="two-way", impacts=False)
    dk = spatial_panel(df, "y", X2, W, effects="two-way", vcov="dk", impacts=False)
    cy = spatial_panel(df, "y", X2, W, effects="two-way", vcov="conley",
                       coords=coords, cutoff_km=3.0, metric="euclidean",
                       impacts=False)
    for r in (base, dk, cy):
        assert np.all(np.isfinite(r.se.to_numpy()))
        assert np.all(r.se.to_numpy() > 0)
        assert r.vcov.shape == (4, 4)
    for r in (dk, cy):
        # rho and sigma2 rows are the OIM, exactly
        assert np.max(np.abs(r.vcov[2:, :] - base.vcov[2:, :])) == 0.0
        assert np.max(np.abs(r.vcov[:, 2:] - base.vcov[:, 2:])) == 0.0
        assert "BETA block only" in r.vcov_note

    # ... and the beta block moved to the RIGHT value: the sandwich is rebuilt
    # here from an independently written score matrix, with the module's own
    # bandwidth rule floor(4 (T/100)^(2/9)) reproduced explicitly.
    V = base.vcov
    scores, _X_dd, _e = _sar_scores(base, df, W, X2, ent=True, tim=True)
    lags = max(1, int(np.floor(4.0 * (14 / 100.0) ** (2.0 / 9.0))))
    assert lags == 2
    ent_keys = np.tile(np.asarray(list(W.ids), dtype=object), 14)
    tim_keys = np.repeat(np.arange(14), W.n)
    meats = {
        "dk": driscoll_kraay(scores, tim_keys, lags=lags),
        "conley": _hac_meat_from_scores(scores, ent_keys, tim_keys, coords, 3.0,
                                        lags, metric="euclidean"),
    }
    for r, key in ((dk, "dk"), (cy, "conley")):
        sand = V @ meats[key] @ V
        ref = 0.5 * (sand[:2, :2] + sand[:2, :2].T)
        assert np.max(np.abs(r.vcov[:2, :2] - ref)) < 1e-15 * np.max(np.abs(ref))
        # and the robust block is genuinely a different number from the OIM
        assert np.max(np.abs(ref - V[:2, :2])) > 0.1 * np.max(np.abs(V[:2, :2]))


def test_the_jacobian_split_cancels_for_dk_but_not_for_conley():
    """The module docstring claims the arbitrary cross-sectional split of the
    Jacobian term ``-(jT/T) diag(G*)_i`` cancels **exactly** for Driscoll-Kraay
    and **does not** cancel for Conley. Pin both halves.

    Re-allocating ``diag(G*)`` to its own mean keeps the period total (and so
    the exact score) identical, which is why Driscoll-Kraay --- a function of
    the period sums alone --- cannot see it, while Conley's kernel-weighted
    within-period quadratic form can.
    """
    W = rook(6)
    T = 14
    df = sar_panel(W, T, seed=13)
    coords = _coords_for(W, 6)
    base = spatial_panel(df, "y", X2, W, effects="two-way", impacts=False)
    sc, _X, _e = _sar_scores(base, df, W, X2, ent=True, tim=True)
    G = W.to_dense() @ np.linalg.inv(np.eye(W.n) - base.rho * W.to_dense())
    gamma = np.diag(_double_centre(G))
    flat = sc.copy()
    flat[:, 2] += ((T - 1.0) / T) * (np.tile(gamma, T) - gamma.mean())
    assert abs(sc[:, 2].sum() - flat[:, 2].sum()) < 1e-9      # same total score
    assert np.max(np.abs(sc[:, 2] - flat[:, 2])) > 1e-3       # genuinely different split

    ent_keys = np.tile(np.asarray(list(W.ids), dtype=object), T)
    tim_keys = np.repeat(np.arange(T), W.n)
    d1 = driscoll_kraay(sc, tim_keys, lags=1)
    d2 = driscoll_kraay(flat, tim_keys, lags=1)
    assert np.max(np.abs(d1 - d2)) < 1e-9 * np.max(np.abs(d1))       # measured 8e-16 rel
    c1 = _hac_meat_from_scores(sc, ent_keys, tim_keys, coords, 2.0, 1, metric="euclidean")
    c2 = _hac_meat_from_scores(flat, ent_keys, tim_keys, coords, 2.0, 1, metric="euclidean")
    assert np.max(np.abs(c1 - c2)) > 1e-3 * np.max(np.abs(c1))       # measured 1.6e-2 rel
    # it reaches the reported beta SEs only through the small beta-rho bread block
    V = base.vcov
    se1 = np.sqrt(np.diag(V @ c1 @ V)[:2])
    se2 = np.sqrt(np.diag(V @ c2 @ V)[:2])
    assert np.max(np.abs(se2 / se1 - 1.0)) < 0.01


def test_qml_and_cluster_vcov_do_not_exist():
    """The outer-product meat captures 54.0% of the rho score variance; these
    are wrong here, not merely unimplemented."""
    W = rook(5)
    df = sar_panel(W, 10, seed=14)
    for bad in ("qml", "cluster", "hc1"):
        with pytest.raises(ValueError, match="is not recognised"):
            spatial_panel(df, "y", X2, W, vcov=bad)


def test_vcov_accepts_the_shared_covariance_token_vocabulary():
    """The keyword split is deliberate (``vcov=`` for the likelihood-based
    estimators, ``cov_type=`` for the projection ones), but the TOKENS are one
    vocabulary: ``'driscoll-kraay'`` must not work in ``spatial_lp`` and
    raise in ``spatial_panel``."""
    W = rook(5)
    df = sar_panel(W, 10, seed=14)
    coords = _coords_for(W, 5)
    ref = spatial_panel(df, "y", X2, W, vcov="dk", impacts=False)
    for alias in ("driscoll-kraay", "driscoll_kraay", "driscollkraay", "DK",
                  "Driscoll-Kraay"):
        got = spatial_panel(df, "y", X2, W, vcov=alias, impacts=False)
        assert got.vcov_type == "dk"
        np.testing.assert_allclose(got.se.to_numpy(float),
                                   ref.se.to_numpy(float), rtol=0, atol=0)
    ref_c = spatial_panel(df, "y", X2, W, vcov="conley", coords=coords,
                          cutoff_km=2.0, metric="euclidean", impacts=False)
    for alias in ("spatial", "spatial-hac", "spatial_hac", "Conley"):
        got = spatial_panel(df, "y", X2, W, vcov=alias, coords=coords,
                            cutoff_km=2.0, metric="euclidean", impacts=False)
        assert got.vcov_type == "conley"
        np.testing.assert_allclose(got.se.to_numpy(float),
                                   ref_c.se.to_numpy(float), rtol=0, atol=0)
    # the cluster family is still refused, and the message quotes what was passed
    for bad in ("cluster", "clustered", "entity"):
        with pytest.raises(ValueError, match=f"vcov={bad!r} is not recognised"):
            spatial_panel(df, "y", X2, W, vcov=bad)
    # a non-string is still reported as itself, not str()-ed by the normaliser
    with pytest.raises(ValueError, match="vcov=3 is not recognised"):
        spatial_panel(df, "y", X2, W, vcov=3)


def test_hac_meat_from_scores_limiting_cases():
    """cutoff 0 + 0 time lags is the per-cell outer product (kernel_matrix(0)
    is the identity); Driscoll-Kraay with 0 lags is clustering by period."""
    W = rook(6)
    rng = np.random.default_rng(0)
    n, T = W.n, 5
    U = rng.standard_normal((n * T, 4))
    ent = np.tile(np.asarray(list(W.ids), dtype=object), T)
    tim = np.repeat(np.arange(T), n)
    coords = _coords_for(W, 6)
    S = _hac_meat_from_scores(U, ent, tim, coords, 0.0, 0, metric="euclidean")
    assert np.max(np.abs(S - U.T @ U)) < 1e-10
    by_t = np.array([U[tim == t].sum(axis=0) for t in range(T)])
    assert np.max(np.abs(driscoll_kraay(U, tim, 0) - by_t.T @ by_t)) < 1e-12


def test_covariance_keyword_guards():
    W = rook(5)
    df = sar_panel(W, 10, seed=15)
    coords = _coords_for(W, 5)
    with pytest.raises(ValueError, match="needs coords"):
        spatial_panel(df, "y", X2, W, vcov="conley")
    with pytest.raises(ValueError, match="needs cutoff_km"):
        spatial_panel(df, "y", X2, W, vcov="conley", coords=coords)
    with pytest.raises(ValueError, match="only used by vcov='conley'"):
        spatial_panel(df, "y", X2, W, vcov="oim", coords=coords)
    with pytest.raises(ValueError, match="only used by vcov='conley'"):
        spatial_panel(df, "y", X2, W, vcov="dk", cutoff_km=100.0)
    with pytest.raises(ValueError, match="only used by vcov='dk' or 'conley'"):
        spatial_panel(df, "y", X2, W, vcov="oim", time_lags=2)
    with pytest.raises(ValueError, match="needs at least 3 periods"):
        spatial_panel(sar_panel(W, 2, seed=16), "y", X2, W, effects="time",
                      vcov="dk", impacts=False)


# ---------------------------------------------------------------------------
# 7. adversarial / degenerate cases
# ---------------------------------------------------------------------------
def test_time_effects_require_row_sums_of_exactly_one():
    W = rook(5, standardize=False)
    df = sar_panel(W, 8, rho=0.15, seed=17)
    for eff in ("time", "two-way"):
        with pytest.raises(ValueError, match=r"standardize\(\)"):
            spatial_panel(df, "y", X2, W, effects=eff, impacts=False)
    # the same W is fine with individual effects ...
    ok = spatial_panel(df, "y", X2, W, effects="individual", impacts=False)
    assert np.isfinite(ok.rho)
    # ... and with the direct route, whose Jacobian is the plain T ln|S|
    ok2 = spatial_panel(df, "y", X2, W, effects="two-way", method="direct",
                        bias_correction="none", impacts=False)
    assert np.isfinite(ok2.rho)
    # but the Lee-Yu time-effect term assumes W l = l
    with pytest.raises(ValueError, match=r"assumes W l = l"):
        spatial_panel(df, "y", X2, W, effects="two-way", method="direct",
                      bias_correction="lee-yu", impacts=False)


def test_islands_block_time_effects_but_not_individual_effects():
    W = rook(4)
    Wd = W.to_dense().copy()
    Wd[0, :] = 0.0                                  # unit u00 becomes an island
    W_isl = SpatialWeights(sp.csr_matrix(Wd), W.ids, "contiguity", False)
    assert W_isl.n_islands == 1
    df = sar_panel(W, 8, seed=18)
    with pytest.raises(ValueError, match=r"standardize\(\)"):
        spatial_panel(df, "y", X2, W_isl, effects="time", impacts=False)
    res = spatial_panel(df, "y", X2, W_isl, effects="individual", impacts=False)
    assert np.isfinite(res.rho)


def test_disconnected_graph_fits_and_removes_exactly_one_unit_eigenvalue():
    """W with two components has the unit eigenvalue twice; the time-effects
    transformation removes exactly one and rho < 1 still binds."""
    nb = {}
    for blk in (0, 1):
        for r in range(3):
            for c in range(3):
                nb[f"b{blk}_{r}{c}"] = [
                    f"b{blk}_{rr}{cc}"
                    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                    if 0 <= rr < 3 and 0 <= cc < 3
                ]
    W = contiguity_weights(nb)
    T = 10
    df = sar_panel(W, T, seed=19)
    n = W.n
    res = spatial_panel(df, "y", X2, W, effects="two-way", impacts=False)
    assert res.n_eff_entities == n - 1
    assert np.isfinite(res.rho) and res.converged

    ev = np.linalg.eigvals(W.to_dense())
    assert int(np.sum(np.abs(ev - 1.0) < 1e-8)) == 2      # one per component

    # EXACTLY one unit eigenvalue is dropped: pin the reported log-likelihood
    # against the reference Jacobian built three ways. Dropping none or both is
    # a different objective by O(1) --- this is the assertion the old
    # `rho_bounds[1] < 1.0` only pretended to make.
    abc = _profile_quadratic(df, W, X2, ent=True, tim=True)
    jT, Nd = T - 1.0, (n - 1) * (T - 1.0)
    unit = np.flatnonzero(np.abs(ev - 1.0) < 1e-8)
    ll_one = _ll_ref(abc, np.delete(ev, unit[0]), jT, Nd, res.rho)
    ll_none = _ll_ref(abc, ev, jT, Nd, res.rho)
    ll_both = _ll_ref(abc, np.delete(ev, unit), jT, Nd, res.rho)
    assert abs(res.loglik - ll_one) < 1e-9
    assert abs(res.loglik - ll_none) > 1.0
    assert abs(res.loglik - ll_both) > 1.0

    # rho < 1 really does bind: the reported search interval stops just short
    # of the unit eigenvalue, and asking for more is refused by name.
    assert res.rho_bounds[1] == pytest.approx(1.0 - 1e-6, abs=1e-9)
    assert res.invertible_bounds[1] == pytest.approx(1.0 - 1e-6, abs=1e-9)
    with pytest.raises(ValueError, match="invertibility interval"):
        spatial_panel(df, "y", X2, W, effects="two-way", impacts=False,
                      rho_bounds=(0.2, 1.0))


def test_duplicate_coordinates_do_not_break_conley():
    W = rook(6)
    df = sar_panel(W, 12, seed=21)
    coords = pd.DataFrame({"lat": [0.0] * W.n, "lon": [0.0] * W.n},
                          index=list(W.ids))
    res = spatial_panel(df, "y", X2, W, effects="two-way", vcov="conley",
                        coords=coords, cutoff_km=100.0, impacts=False)
    assert np.all(np.isfinite(res.se.to_numpy()))


def test_single_period_and_single_entity():
    W = rook(4)
    df1 = sar_panel(W, 1, seed=22)
    for eff in ("individual", "two-way"):
        with pytest.raises(ValueError, match=r"T-1 = 0"):
            spatial_panel(df1, "y", X2, W, effects=eff, impacts=False)
    assert np.isfinite(spatial_panel(df1, "y", X2, W, effects="time",
                                     impacts=False).rho)
    r0 = spatial_panel(df1, "y", X2, W, effects="none", impacts=False)
    assert list(r0.params.index)[0] == "const"

    W1 = SpatialWeights(sp.csr_matrix((1, 1)), ("only",))
    df_one = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0], "x0": [0.1, 0.2, 0.3], "x1": [1.0, 0.0, 2.0]},
        index=pd.MultiIndex.from_product([["only"], range(3)],
                                         names=["code", "date"]),
    )
    with pytest.raises(ValueError, match=r"n-1 = 0"):
        spatial_panel(df_one, "y", X2, W1, effects="time", impacts=False)


def test_constant_regressor_and_absorbed_regressors():
    W = rook(4)
    df = sar_panel(W, 6, seed=23)
    inv = df.copy()
    inv["ti"] = np.repeat(np.arange(W.n), 6).astype(float)
    with pytest.raises(ValueError, match=r"'ti' is time-invariant"):
        spatial_panel(inv, "y", ["x0", "ti"], W, effects="individual", impacts=False)
    cs = df.copy()
    cs["cs"] = np.tile(np.arange(6), W.n).astype(float)
    with pytest.raises(ValueError, match=r"'cs' is cross-sectionally constant"):
        spatial_panel(cs, "y", ["x0", "cs"], W, effects="time", impacts=False)
    con = df.copy()
    con["c"] = 1.0
    with pytest.raises(ValueError, match=r"'c' is constant"):
        spatial_panel(con, "y", ["x0", "c"], W, effects="two-way", impacts=False)
    with pytest.raises(ValueError, match="already prepends"):
        spatial_panel(con, "y", ["x0", "c"], W, effects="none", impacts=False)


def test_unbalanced_and_duplicated_panels_raise():
    W = rook(4)
    df = sar_panel(W, 6, seed=24)
    with pytest.raises(ValueError, match="not balanced.*balanced_subpanel"):
        spatial_panel(df.drop(df.index[3]), "y", X2, W, effects="two-way",
                      impacts=False)
    nan_df = df.copy()
    nan_df.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        spatial_panel(nan_df, "y", X2, W, effects="two-way", impacts=False)
    dup = pd.concat([df, df.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        spatial_panel(dup, "y", X2, W, effects="two-way", impacts=False)


def test_entity_sets_must_match_the_weights():
    W = rook(4)
    df = sar_panel(W, 6, seed=25)
    with pytest.raises(KeyError, match="weights.ids differ"):
        spatial_panel(df.drop(index="u00", level="code"), "y", X2, W,
                      effects="two-way", impacts=False)
    smaller = contiguity_weights(
        {u: [v for v in W.neighbors(u) if v != "u00"]
         for u in W.ids if u != "u00"})
    with pytest.raises(KeyError, match="weights.ids differ"):
        spatial_panel(df, "y", X2, smaller, effects="two-way", impacts=False)


def test_entity_and_row_order_are_irrelevant():
    W = rook(5)
    df = sar_panel(W, 10, seed=26)
    base = spatial_panel(df, "y", X2, W, effects="two-way", impacts=False)
    shuffled = spatial_panel(df.sample(frac=1.0, random_state=1), "y", X2, W,
                             effects="two-way", impacts=False)
    assert np.max(np.abs(base.params.to_numpy() - shuffled.params.to_numpy())) == 0.0
    perm = list(np.random.default_rng(0).permutation(list(W.ids)))
    W_perm = contiguity_weights({u: list(W.neighbors(u)) for u in perm})
    assert list(W_perm.ids) != list(W.ids)
    permuted = spatial_panel(df, "y", X2, W_perm, effects="two-way", impacts=False)
    assert np.max(np.abs(base.params.to_numpy() - permuted.params.to_numpy())) < 1e-7


def test_rho_pinned_at_a_bound_warns_and_still_reports_finite_ses():
    W = rook(6)
    df = sar_panel(W, 20, rho=0.8, seed=27)
    with pytest.warns(RuntimeWarning, match="within 1e-4 of"):
        res = spatial_panel(df, "y", X2, W, effects="two-way",
                            rho_bounds=(-0.4, 0.4), impacts=False)
    assert res.rho == pytest.approx(0.4, abs=1e-4)
    assert not res.converged
    assert np.all(np.isfinite(res.se.to_numpy()))


def test_unidentified_spatial_parameter_raises():
    """A zero W makes Wy identically zero, so SSR(rho) is constant."""
    W = rook(4)
    Z = SpatialWeights(sp.csr_matrix((W.n, W.n)), W.ids)
    df = sar_panel(W, 8, seed=28)
    with pytest.raises(ValueError, match="not identified"):
        spatial_panel(df, "y", X2, Z, effects="individual", impacts=False)


def test_dense_max_is_a_hard_ceiling():
    W = rook(4)
    df = sar_panel(W, 6, seed=29)
    with pytest.raises(ValueError, match="exceeds dense_max"):
        spatial_panel(df, "y", X2, W, dense_max=5, impacts=False)


def test_enum_typos_and_unknown_keywords():
    W = rook(4)
    df = sar_panel(W, 6, seed=30)
    for kw, bad in (("effects", "twoway"), ("model", "sac"), ("method", "qml"),
                    ("bias_correction", "hahn-kuersteiner"), ("logdet", "chebyshev")):
        with pytest.raises(ValueError, match="is not recognised"):
            spatial_panel(df, "y", X2, W, **{kw: bad})
    with pytest.raises(ValueError, match="logdet='lu' is not available"):
        spatial_panel(df, "y", X2, W, logdet="lu")
    with pytest.raises(TypeError):
        spatial_panel(df, "y", X2, W, error_weights=W)      # SARAR is out of scope
    with pytest.raises(TypeError, match="SpatialWeights"):
        spatial_panel(df, "y", X2, W.to_dense())
    with pytest.raises(ValueError, match="out of scope"):
        spatial_panel(df, "y", X2, W, model="sem", durbin=True)
    with pytest.raises(KeyError, match="not in the frame"):
        spatial_panel(df, "nope", X2, W)


# ---------------------------------------------------------------------------
# 8. fit statistics, recovered effects, presentation contract
# ---------------------------------------------------------------------------
def test_lr_test_matches_a_refit_with_rho_pinned_at_zero():
    W = rook(5)
    df = sar_panel(W, 10, seed=31)
    res = spatial_panel(df, "y", X2, W, effects="two-way", impacts=False)
    assert res.lr_stat == pytest.approx(2.0 * (res.loglik - res.loglik_null), abs=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pinned = spatial_panel(df, "y", X2, W, effects="two-way",
                               rho_bounds=(-1e-10, 1e-10), impacts=False)
    assert pinned.loglik == pytest.approx(res.loglik_null, abs=1e-8)
    assert res.lr_stat > 0 and 0.0 <= res.lr_pvalue <= 1.0


def test_recovered_effects_sum_to_zero_and_decompose_the_residual():
    """``r_it = intercept + mu_i + xi_t + r_dd,it`` exactly, with ``r_dd`` the
    doubly-demeaned array (the within part does NOT vanish)."""
    W = rook(5)
    df = sar_panel(W, 10, seed=32)
    res = spatial_panel(df, "y", X2, W, effects="two-way", impacts=False)
    Y = _blocks(df, W, "y")
    R = Y - res.rho * (W.W @ Y)
    for c in X2:
        R = R - float(res.beta[c]) * _blocks(df, W, c)
    R_dd = _demean(R, True, True)
    rebuilt = (res.intercept + res.entity_effects.to_numpy()[:, None]
               + res.time_effects.to_numpy()[None, :] + R_dd)
    assert np.max(np.abs(rebuilt - R)) < 1e-10
    assert abs(float(res.entity_effects.sum())) < 1e-10
    assert abs(float(res.time_effects.sum())) < 1e-10
    assert list(res.entity_effects.index) == list(W.ids)


def test_effects_are_none_when_not_removed():
    W = rook(5)
    df = sar_panel(W, 10, seed=33, tim=False)
    res = spatial_panel(df, "y", X2, W, effects="individual", impacts=False)
    assert res.time_effects is None and res.entity_effects is not None
    assert res.n_eff_entities == W.n and res.n_eff_periods == 9


def test_within_r2_and_residual_index():
    W = rook(5)
    df = sar_panel(W, 10, seed=34)
    res = spatial_panel(df, "y", X2, W, effects="two-way", impacts=False)
    assert 0.0 < res.r2_within < 1.0
    assert res.resid.index.names == ["code", "date"]
    assert len(res.resid) == W.n * 10
    assert res.n_obs == W.n * 10
    assert res.n_eff == (W.n - 1) * 9


def test_presentation_contract():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    W = rook(5)
    df = sar_panel(W, 10, seed=35)
    res = spatial_panel(df, "y", X2, W, model="sdm", effects="two-way",
                        impacts=True, n_draws=100)
    assert isinstance(res, SpatialPanelResult)
    txt = res.summary()
    assert isinstance(txt, str) and txt
    assert "SDM" in txt and "two-way" in txt and "LeSage-Pace" in txt
    assert "mis-specified W" in txt
    assert list(res.to_frame().columns) == [
        "term", "coef", "se", "z", "p_value", "ci_lower", "ci_upper"]
    assert list(res.impacts_frame().columns) == [
        "variable", "direct", "direct_se", "indirect", "indirect_se",
        "total", "total_se", "total_p"]
    md, tex, typ = res.to_markdown(), res.to_latex(), res.to_typst()
    assert md.startswith("|") and "tabular" in tex and "#table" in typ
    assert isinstance(res.plot(), Figure)
    sem = spatial_panel(sem_panel(W, 10, seed=36), "y", X2, W, model="sem",
                        effects="two-way")
    assert isinstance(sem.plot(), Figure)
    with pytest.raises(FrozenInstanceError):
        res.rho = 0.0


def test_profile_grid_brackets_the_optimum():
    W = rook(5)
    df = sar_panel(W, 10, seed=37)
    res = spatial_panel(df, "y", X2, W, effects="two-way", impacts=False)
    prof = res.profile
    assert list(prof.columns) == ["rho", "loglik"]
    assert len(prof) == 51
    assert prof["loglik"].max() <= res.loglik + 1e-9
    assert prof["rho"].iloc[0] == pytest.approx(res.rho_bounds[0])
    assert prof["rho"].iloc[-1] == pytest.approx(res.rho_bounds[1])


def test_long_form_input_and_custom_level_names():
    W = rook(5)
    df = sar_panel(W, 10, seed=38)
    long = df.reset_index().rename(columns={"code": "state", "date": "quarter"})
    res_long = spatial_panel(long, "y", X2, W, effects="two-way",
                             unit_col="state", time_col="quarter", impacts=False)
    res = spatial_panel(df, "y", X2, W, effects="two-way", impacts=False)
    assert np.max(np.abs(res.params.to_numpy() - res_long.params.to_numpy())) == 0.0
    assert res_long.resid.index.names == ["state", "quarter"]


def test_impacts_with_a_durbin_subset_and_with_effects_none():
    """theta_r is aligned to beta_r BY NAME (0 for a non-Durbin regressor), and
    the constant that effects='none' prepends never appears in the impacts."""
    W = rook(5)
    df = sar_panel(W, 10, seed=40)
    res = spatial_panel(df, "y", X2, W, durbin=["x0"], effects="two-way",
                        impacts=True, n_draws=200)
    n = W.n
    Wd = W.to_dense()
    S_inv = np.linalg.inv(np.eye(n) - res.rho * Wd)
    ell = np.ones(n)
    for c, th in (("x0", float(res.params["W.x0"])), ("x1", 0.0)):
        Sr = S_inv @ (float(res.params[c]) * np.eye(n) + th * Wd)
        assert res.impacts.loc[c, "direct"] == pytest.approx(np.trace(Sr) / n, abs=1e-12)
        assert res.impacts.loc[c, "total"] == pytest.approx(ell @ Sr @ ell / n, abs=1e-12)
    assert np.all(res.impacts[["direct_se", "indirect_se", "total_se"]].to_numpy() > 0)

    plain = spatial_panel(df, "y", X2, W, effects="none", impacts=True, n_draws=100)
    assert list(plain.params.index) == ["const", "x0", "x1", "rho", "sigma2"]
    assert list(plain.impacts.index) == ["x0", "x1"]
    assert plain.entity_effects is None and plain.time_effects is None


# ---------------------------------------------------------------------------
# 11. branches that used to be untested: PSD clipping, a non-PSD sandwich,
#     the grid fallback, and the two spectrum intervals
# ---------------------------------------------------------------------------
def test_psd_draws_clips_negative_eigenvalues_warns_and_is_conservative():
    """``_psd_draws`` is the only reason a sandwich covariance may be simulated
    from at all: ``Generator.multivariate_normal`` accepts a non-PSD covariance
    silently. Clipping must warn, and must shrink (never inflate) the variance."""
    cov = np.diag([1.0, -0.5])
    with pytest.warns(RuntimeWarning, match="positive semi-definite"):
        D = _psd_draws(np.array([2.0, -3.0]), cov, 20000,
                       np.random.default_rng(0), "spatial_panel")
    assert D.shape == (20000, 2)
    C = np.cov(D.T)
    assert C[1, 1] == 0.0                        # the negative direction is exactly zero
    assert abs(C[0, 0] - 1.0) < 0.05             # the positive one survives
    assert np.max(np.abs(D.mean(axis=0) - np.array([2.0, -3.0]))) < 0.05
    # a PSD covariance must go through silently
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ok = _psd_draws(np.zeros(2), np.diag([1.0, 2.0]), 500,
                        np.random.default_rng(0), "spatial_panel")
    assert np.all(np.isfinite(ok))


def test_a_non_psd_sandwich_is_counted_warned_and_reported_as_nan(monkeypatch):
    """``n_nonpsd``, the RuntimeWarning and the nan se/z/p/CI exist for the case
    where ``V S V`` comes back with a non-positive diagonal. A PSD meat can
    never produce it, so the meat is replaced by ``-I`` to reach the branch."""
    monkeypatch.setattr("puremacro.inference.dk.driscoll_kraay",
                        lambda u, t, lags=0: -np.eye(u.shape[1]))
    W = rook(5)
    df = sar_panel(W, 10, seed=13)
    with pytest.warns(RuntimeWarning, match="non-positive variance"):
        res = spatial_panel(df, "y", X2, W, effects="two-way", vcov="dk",
                            impacts=False)
    assert res.n_nonpsd == 2
    assert np.all(np.isnan(res.se[["x0", "x1"]].to_numpy()))
    assert np.all(np.isnan(res.p_values[["x0", "x1"]].to_numpy()))
    assert np.all(np.isnan(res.conf_int.loc[["x0", "x1"]].to_numpy()))
    assert np.all(np.isfinite(res.se[["rho", "sigma2"]].to_numpy()))
    assert "2 parameter(s) had a non-positive sandwich variance" in res.summary()


def test_converged_is_false_when_the_reported_rho_is_a_grid_point(monkeypatch):
    """When the bounded optimiser loses to the 51-point profile grid, rho_hat is
    snapped back to the grid point --- and that is not a converged optimum,
    whatever ``opt.success`` says."""
    W = rook(5)
    df = sar_panel(W, 10, seed=22)
    good = spatial_panel(df, "y", X2, W, effects="two-way", impacts=False)
    assert good.converged

    class _Bad:
        x = 0.0
        fun = 1e9                                # -fun is far below any grid value
        success = True

    monkeypatch.setattr("puremacro.spatial.panel.optimize.minimize_scalar",
                        lambda *a, **k: _Bad())
    res = spatial_panel(df, "y", X2, W, effects="two-way", impacts=False)
    assert res.converged is False
    assert res.rho in set(res.profile["rho"].to_numpy(float))
    assert res.rho == pytest.approx(
        float(res.profile.loc[res.profile["loglik"].idxmax(), "rho"]), abs=0.0)


def _directed_cycle_weights(scale: float) -> SpatialWeights:
    """3-cycle a -> b -> c -> a, scaled. Non-negative, zero diagonal, and NOT
    row-standardised; its spectrum is ``{scale, scale*exp(+-2pi i/3)}``, so the
    largest modulus (``scale``) exceeds the largest REAL eigenvalue's reach on
    the negative side --- exactly the configuration where the modulus interval
    is strictly tighter than the invertibility interval."""
    P = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]) * scale
    return SpatialWeights(sp.csr_matrix(P), ("a", "b", "c"), "custom", False)


def _tiny_panel(ids, T, seed):
    rng = np.random.default_rng(seed)
    n = len(ids)
    X = rng.standard_normal((n, T))
    Y = X + rng.standard_normal((n, 1)) + rng.standard_normal((n, T))
    idx = pd.MultiIndex.from_product([list(ids), range(T)], names=["code", "date"])
    return pd.DataFrame({"y": Y.ravel(), "x": X.ravel()}, index=idx)


@pytest.mark.filterwarnings("ignore:.*within 1e-4 of the search bound.*")
def test_both_spectrum_intervals_are_reported_and_only_the_wide_one_binds():
    """The stability interval (modulus) can be strictly tighter than the
    invertibility interval (real spectrum): a complex eigenvalue never makes
    ``I - rho W`` singular for a real rho. A user interval between the two is a
    warning, not a refusal; outside the wide one it is still a refusal."""
    W = _directed_cycle_weights(2.0)
    df = _tiny_panel(W.ids, 14, seed=5)
    res = spatial_panel(df, "y", "x", W, effects="individual", impacts=False)
    assert res.stability_bounds[0] == pytest.approx(-0.5 + 1e-6, abs=1e-9)
    assert res.invertible_bounds[0] < -0.9
    assert res.stability_bounds[1] == pytest.approx(0.5 - 1e-6, abs=1e-9)
    assert res.rho_bounds == res.stability_bounds
    # S(-0.7) is perfectly invertible; the old code refused it outright
    for lam in np.linalg.eigvals(W.to_dense()):
        assert abs(1.0 - (-0.7) * lam) > 1e-6
    with pytest.warns(RuntimeWarning, match="outside the stability interval"):
        warned = spatial_panel(df, "y", "x", W, effects="individual",
                               impacts=False, rho_bounds=(-0.8, -0.6))
    assert np.isfinite(warned.rho)
    with pytest.raises(ValueError, match="invertibility interval"):
        spatial_panel(df, "y", "x", W, effects="individual", impacts=False,
                      rho_bounds=(-1.5, -0.6))


# ---------------------------------------------------------------------------
# 12. no silent guessing, and no false claim in summary()
# ---------------------------------------------------------------------------
def test_keywords_meaningless_for_the_chosen_mode_raise():
    """G3: a keyword that cannot act in the chosen mode raises naming both the
    argument and the mode, instead of being silently downgraded or stored."""
    W = rook(5)
    df = sar_panel(W, 10, seed=23)
    sem = sem_panel(W, 10, seed=23)
    with pytest.raises(ValueError, match=r"bias_correction='lee-yu' is meaningless"):
        spatial_panel(df, "y", X2, W, effects="none", method="direct",
                      bias_correction="lee-yu", impacts=False)
    with pytest.raises(ValueError, match=r"impacts=True is meaningless for model='sem'"):
        spatial_panel(sem, "y", X2, W, model="sem", impacts=True)
    with pytest.raises(ValueError, match=r"impacts=False runs no impact simulation"):
        spatial_panel(df, "y", X2, W, impacts=False, n_draws=7)
    with pytest.raises(ValueError, match=r"impacts=False runs no impact simulation"):
        spatial_panel(df, "y", X2, W, impacts=False, seed=3)
    with pytest.raises(ValueError, match=r"model='sem' runs no impact simulation"):
        spatial_panel(sem, "y", X2, W, model="sem", n_draws=999, seed=5)
    # the defaults still resolve silently, and only where they mean something
    assert spatial_panel(sem, "y", X2, W, model="sem", effects="two-way").impacts is None
    assert spatial_panel(df, "y", X2, W, effects="two-way", n_draws=50).impacts is not None
    with pytest.raises(ValueError, match="n_draws must be at least 2"):
        spatial_panel(df, "y", X2, W, n_draws=1)


def test_summary_states_the_true_bias_claim_in_every_mode():
    """summary() must never tell a direct-approach user that an uncorrected
    estimate is 'provably zero' bias --- that sentence belongs to the
    transformation route alone."""
    W = rook(5)
    df = sar_panel(W, 10, seed=24)
    trans = spatial_panel(df, "y", X2, W, effects="two-way", impacts=False).summary()
    assert "provably zero" in trans and "direct approach needs one" not in trans

    direct = spatial_panel(df, "y", X2, W, effects="two-way", method="direct",
                           bias_correction="none", impacts=False).summary()
    assert "provably zero" not in direct
    assert "the direct approach needs one" in direct
    assert "O(1/T) + O(1/n) bias" in direct
    assert "bias_correction='lee-yu'" in direct

    plain = spatial_panel(df, "y", X2, W, effects="none", method="direct",
                          impacts=False).summary()
    assert "provably zero" not in plain
    assert "no incidental parameters" in plain

    corrected = spatial_panel(df, "y", X2, W, effects="two-way", method="direct",
                              impacts=False).summary()
    assert "Lee-Yu analytic correction applied" in corrected


def test_intercept_is_the_fitted_constant_when_no_effects_are_removed():
    """effects='none' has no zero-mean effect series to measure from, and the
    residual it would otherwise average is already net of the constant's own
    coefficient (machine zero). Report params['const'] instead."""
    W = rook(5)
    df = sar_panel(W, 10, seed=25, ent=False, tim=False)
    res = spatial_panel(df, "y", X2, W, effects="none", impacts=False)
    assert res.entity_effects is None and res.time_effects is None
    assert res.intercept == pytest.approx(float(res.params["const"]), abs=0.0)
    assert abs(res.intercept) > 1e-6            # a real level, not an inert 1e-16
    # ... and the two-way meaning is unchanged: the level the effects sit on
    two = spatial_panel(df, "y", X2, W, effects="two-way", impacts=False)
    assert abs(float(two.entity_effects.sum())) < 1e-10
    assert abs(float(two.time_effects.sum())) < 1e-10


def test_panel_impacts_agree_with_models_spatial_effects():
    """This module computes the LeSage-Pace decomposition privately (vectorised
    closed forms over the cached spectrum) instead of calling
    :func:`puremacro.spatial.models.spatial_effects`; the module docstring says
    so and says why. Pin the point estimates against that function so the two
    implementations cannot drift apart.

    Only the points: the interval conventions differ on purpose ---
    ``spatial_effects`` reports empirical simulation quantiles and simulation
    p-values, this module reports ``point +/- z se``.
    """
    W = rook(5)
    df = sar_panel(W, 10, seed=9)
    res = spatial_panel(df, "y", X2, W, model="sdm", effects="two-way",
                        impacts=True, n_draws=200)
    ref = spatial_effects(
        W,
        [float(res.params["x0"]), float(res.params["x1"])],
        float(res.rho),
        theta=[float(res.params["W.x0"]), float(res.params["W.x1"])],
        names=list(X2), n_sim=0,
    )
    assert list(ref.variables) == X2
    for key in ("direct", "indirect", "total"):
        assert np.max(np.abs(res.impacts[key].to_numpy(float)
                             - getattr(ref, key))) < 1e-12
    # the CI convention really is the normal one here
    z = 1.959963984540054
    assert np.max(np.abs(
        res.impacts["total_lo"].to_numpy(float)
        - (res.impacts["total"].to_numpy(float)
           - z * res.impacts["total_se"].to_numpy(float)))) < 1e-12
    assert np.all(np.isnan(ref.total_ci))       # n_sim=0 -> no interval there
