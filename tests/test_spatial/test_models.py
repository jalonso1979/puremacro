"""SAR / SEM / SDM / SLX, the LM battery and the LeSage-Pace impacts.

The tests pin behaviour, not implementation:

* **frozen ``spreg`` goldens** for every estimator that has an external
  reference (the literals below were produced once by ``spreg`` 1.9.0 /
  ``libpysal`` 4.15.0 on the seeded DGP in this file, and
  ``test_spreg_goldens_are_still_live`` re-derives them from a live ``spreg``
  under ``-m reference``);
* **closed-form identities** where the module is its own reference — the
  concentrated maximiser equals the root of the analytic score, ``ln L_c(0)``
  equals the Gaussian OLS log-likelihood, ``sdm`` *is* ``sar`` on the augmented
  design, ``sem`` with ``lambda`` pinned to zero *is* OLS, the LeSage-Pace
  total impact is ``(beta + theta)/(1 - rho)`` on a row-standardised
  island-free ``W`` and is **not** on one with an island;
* **reductions** to a brute-force dense ``S_r = A^-1 (beta I + theta W)``,
  which is what replaces a ``spreg`` golden for the SDM impact split (spreg's
  ``spat_impacts='full'`` drops the Durbin term from the direct effect on
  purpose, so goldening it there would pin the wrong number);
* the **adversarial** cases: islands, a disconnected graph, duplicate
  coordinates, a single regressor, a constant regressor, an all-island ``W``,
  and an asymmetric ``W`` with complex eigenvalues.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from puremacro._linalg import inv_xtx
from puremacro.spatial import (
    SpatialWeights,
    contiguity_weights,
    distance_weights,
    economic_weights,
    knn_weights,
)
from puremacro.spatial.models import (
    SpatialEffectsResult,
    SpatialLMResult,
    SpatialModelResult,
    SpatialOLSResult,
    lm_spatial_tests,
    ols_spatial,
    sar,
    sdm,
    sem,
    slx,
    spatial_effects,
)
from puremacro.spatial.models import (
    _admissible_bounds,
    _LogDet,
    _row_power_sums,
    _series_order,
    _spectrum,
    _symmetrizing_scale,
)

from .test_weights import _rook_grid

# ---------------------------------------------------------------------------
# The one DGP every golden is measured on: a 7x7 rook lattice, row-standardised
# contiguity weights, a SAR at rho = 0.5. Seeded at module level so the goldens
# and the puremacro computation see byte-identical inputs.
# ---------------------------------------------------------------------------
SIDE = 7
N = SIDE * SIDE
SEED = 20240906


def _lattice_regression_data(side: int = SIDE, seed: int = SEED, rho: float = 0.5):
    nb = _rook_grid(side)
    W = contiguity_weights(nb)
    n = side * side
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    A = np.linalg.inv(np.eye(n) - rho * W.to_dense())
    y = A @ (1.0 + X @ np.array([1.0, -0.5]) + rng.normal(size=n))
    return y, X, W


#: Frozen ``spreg`` 1.9.0 / ``libpysal`` 4.15.0 output on ``_lattice_regression_data()``.
GOLDEN = {
    "ml_lag": {
        "betas": [1.1981682194626186, 0.8774694626466651, -0.6084907430277315,
                  0.4325402940511969],
        "se": [0.30937423949226284, 0.1436962168800052, 0.13082890133822006,
               0.1365158365217572],
        "logll": -66.67727523347847,
        "sig2": 0.838983124211179,
    },
    "ml_error": {
        "betas": [2.076816935140044, 0.909719936630245, -0.5114861959114841,
                  0.5272210786463556],
        "se": [0.27255791164620397, 0.13767599492979965, 0.13598992941026916,
               0.14229840587077194],
        "logll": -66.64136279474391,
        "sig2": 0.8115003950911429,
    },
    "ml_sdm": {
        "betas": [1.3373387044099376, 0.9230320275789264, -0.5145966110895359,
                  -0.3382735557654345, -0.4561102591561648, 0.38045855322354705],
        "se": [0.352077972039083, 0.13793410626014863, 0.1331374774665623,
               0.299967926828636, 0.29511458843395993, 0.15607172077533576],
        "logll": -63.41376816208441,
        "cfh": [7.260722059291963, 0.026506613008107943],
    },
    "gm_lag": {
        "betas": [0.6337179999631241, 0.9229957551254951, -0.531950229028281,
                  0.7062678645465286],
        "se": [0.46623442277351357, 0.1461355436521712, 0.14211006681911412,
               0.21719041056994726],
    },
    "gm_error": {
        "betas": [2.078215984165867, 0.8909518501067233, -0.5546710541500549,
                  0.4118832851422561],
        "se": [0.22463171806309748, 0.14278417840630536, 0.13847922226411485],
        "sig2": 0.8520378219067434,
    },
    "gm_sdm": {
        "betas": [-0.2100790030436297, 0.9828569141807765, -0.40711248445708154,
                  -0.8916816769272643, 0.31329436581234293, 1.1094657791959992],
        "se": [0.8254554270826239, 0.1439947786027803, 0.14494187513008056,
               0.4018213231795451, 0.48728837252344104, 0.38420874469607946],
    },
    "ols": {
        "betas": [2.090104181476859, 0.8055294826869562, -0.7294389411659121],
        "se": [0.1521005791491661, 0.16621066259786493, 0.15080907721710868],
        "lm_lag": [8.109103826959885, 0.0044043495648006595],
        "lm_error": [6.142739661702839, 0.013195271938423082],
        "rlm_lag": [1.9705884724807343, 0.16038541278661814],
        "rlm_error": [0.0042243072236882034, 0.9481782434237005],
        "lm_sarma": [8.113328134183574, 0.017306656770070664],
        "moran": [0.2745942357739294, 2.8174530204674912, 0.004840619570543918],
    },
}


@pytest.fixture(scope="module")
def lattice():
    return _lattice_regression_data()


def _gaussian_ols_loglik(y: np.ndarray, X: np.ndarray) -> float:
    b = inv_xtx(X) @ (X.T @ y)
    e = y - X @ b
    n = y.shape[0]
    return -0.5 * n * (np.log(2.0 * np.pi) + 1.0 + np.log(float(e @ e) / n))


# ===========================================================================
# 1. External goldens
# ===========================================================================
def test_sar_ml_matches_spreg_golden(lattice):
    y, X, W = lattice
    res = sar(y, X, W)
    g = GOLDEN["ml_lag"]
    np.testing.assert_allclose(res.params.to_numpy(), g["betas"], rtol=1e-8)
    np.testing.assert_allclose(res.bse.to_numpy(), g["se"], rtol=1e-7)
    assert res.loglik == pytest.approx(g["logll"], rel=1e-10)
    assert res.sigma2 == pytest.approx(g["sig2"], rel=1e-8)
    assert list(res.params.index) == ["const", "x1", "x2", "rho"]
    assert res.model == "sar" and res.method == "ml" and res.vcov_type == "analytic"
    assert res.logdet_method == "eig"  # n = 49 <= 1000
    assert res.vcov_names == ("const", "x1", "x2", "rho", "sigma2")
    assert res.vcov.shape == (5, 5)


def test_sem_ml_matches_spreg_golden(lattice):
    y, X, W = lattice
    res = sem(y, X, W)
    g = GOLDEN["ml_error"]
    np.testing.assert_allclose(res.params.to_numpy(), g["betas"], rtol=1e-7)
    np.testing.assert_allclose(res.bse.to_numpy(), g["se"], rtol=1e-6)
    assert res.loglik == pytest.approx(g["logll"], rel=1e-10)
    assert res.sigma2 == pytest.approx(g["sig2"], rel=1e-7)
    assert res.rho is None and res.lam == pytest.approx(g["betas"][-1], rel=1e-7)


def test_sem_beta_block_is_exactly_the_filtered_gls_covariance(lattice):
    """The SEM information matrix is block diagonal, so SE(beta) is exact."""
    y, X, W = lattice
    res = sem(y, X, W)
    Xc = np.column_stack([np.ones(N), X])
    Xs = Xc - res.lam * (W.W @ Xc)
    expect = np.sqrt(np.diag(res.sigma2 * inv_xtx(Xs)))
    np.testing.assert_allclose(res.bse.to_numpy()[:3], expect, rtol=1e-10)
    # ... and the beta-lambda off-diagonal block is exactly zero
    assert np.max(np.abs(res.vcov[:3, 3])) < 1e-12


def test_sdm_ml_matches_spreg_golden_including_the_common_factor_test(lattice):
    y, X, W = lattice
    res = sdm(y, X, W)
    g = GOLDEN["ml_sdm"]
    np.testing.assert_allclose(res.params.to_numpy(), g["betas"], rtol=1e-8)
    np.testing.assert_allclose(res.bse.to_numpy(), g["se"], rtol=1e-7)
    assert res.loglik == pytest.approx(g["logll"], rel=1e-10)
    assert list(res.params.index) == ["const", "x1", "x2", "W_x1", "W_x2", "rho"]
    assert res.durbin_names == ("x1", "x2")
    stat, df, pval = res.common_factor
    assert stat == pytest.approx(g["cfh"][0], rel=1e-8)
    assert df == 2
    assert pval == pytest.approx(g["cfh"][1], rel=1e-8)


def test_gmm_lag_matches_spreg_golden(lattice):
    y, X, W = lattice
    res = sar(y, X, W, method="gmm")
    g = GOLDEN["gm_lag"]
    np.testing.assert_allclose(res.params.to_numpy(), g["betas"], rtol=1e-9)
    np.testing.assert_allclose(res.bse.to_numpy(), g["se"], rtol=1e-9)
    assert res.vcov_type == "classical"  # vcov=None -> 'classical' under gmm
    assert np.isnan(res.loglik) and np.isnan(res.aic)
    assert res.logdet_method == "none"


def test_gmm_error_matches_spreg_golden_with_the_filtered_sigma2(lattice):
    y, X, W = lattice
    res = sem(y, X, W, method="gmm")
    g = GOLDEN["gm_error"]
    # KP(1999) is solved by bounded L-BFGS-B here and by spreg's own optimiser
    # there; the surface is flat, so six significant figures is the honest claim.
    np.testing.assert_allclose(res.params.to_numpy(), g["betas"], rtol=1e-5)
    np.testing.assert_allclose(res.bse.to_numpy()[:3], g["se"], rtol=1e-5)
    assert res.sigma2 == pytest.approx(g["sig2"], rel=1e-5)
    # the filtered sigma2, not the GMM moment-system sigma2
    Xc = np.column_stack([np.ones(N), X])
    Xs = Xc - res.lam * (W.W @ Xc)
    ys = y - res.lam * (W.W @ y)
    e = ys - Xs @ res.beta
    assert res.sigma2 == pytest.approx(float(e @ e) / N, rel=1e-12)
    # no standard error is claimed for the GMM lambda
    assert np.isnan(res.bse["lambda"])
    assert -1.0 < res.lam < 1.0


def test_gmm_sdm_matches_spreg_golden_and_drops_no_instrument_noisily(lattice):
    """``W X_d`` is an included regressor, so it cannot instrument for ``W y``.

    That drop is structural, not a diagnostic, and must not raise a warning on
    the plainest SDM-GMM call.
    """
    y, X, W = lattice
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = sdm(y, X, W, method="gmm")
    assert not [w for w in rec if "instrument" in str(w.message)]
    np.testing.assert_allclose(res.params.to_numpy(), GOLDEN["gm_sdm"]["betas"], rtol=1e-9)
    np.testing.assert_allclose(res.bse.to_numpy(), GOLDEN["gm_sdm"]["se"], rtol=1e-9)
    assert res.common_factor is not None  # the Wald test works off the GMM vcov too


def test_lm_battery_matches_spreg_golden(lattice):
    y, X, W = lattice
    res = lm_spatial_tests(y, X, W)
    g = GOLDEN["ols"]
    for name, (stat, pval) in {
        "lm_lag": (res.lm_lag, res.p_lag),
        "lm_error": (res.lm_error, res.p_error),
        "rlm_lag": (res.rlm_lag, res.p_rlm_lag),
        "rlm_error": (res.rlm_error, res.p_rlm_error),
        "lm_sarma": (res.lm_sarma, res.p_sarma),
    }.items():
        assert stat == pytest.approx(g[name][0], rel=1e-9), name
        assert pval == pytest.approx(g[name][1], rel=1e-9), name
    assert res.moran_i == pytest.approx(g["moran"][0], rel=1e-9)
    assert res.moran_z == pytest.approx(g["moran"][1], rel=1e-9)
    assert res.moran_p == pytest.approx(g["moran"][2], rel=1e-9)
    assert res.moran_traces_exact
    # Both LM statistics reject at 5% and neither robust one does: the honest
    # verdict is "inconclusive", not a pick between sar and sem.
    assert res.p_lag < 0.05 and res.p_error < 0.05
    assert res.p_rlm_lag > 0.05 and res.p_rlm_error > 0.05
    assert res.recommendation == "inconclusive (neither robust LM rejects)"


def test_ols_spatial_matches_spreg_and_carries_the_battery(lattice):
    y, X, W = lattice
    res = ols_spatial(y, X, W)
    g = GOLDEN["ols"]
    np.testing.assert_allclose(res.params.to_numpy(), g["betas"], rtol=1e-10)
    np.testing.assert_allclose(res.bse.to_numpy(), g["se"], rtol=1e-10)
    assert isinstance(res.lm, SpatialLMResult)
    assert res.lm.lm_lag == pytest.approx(g["lm_lag"][0], rel=1e-9)
    # dof-corrected sigma2 for the SEs, ML sigma2 for the log-likelihood
    assert res.sigma2 == pytest.approx(float(res.resid @ res.resid) / (N - 3), rel=1e-12)
    assert res.loglik == pytest.approx(
        _gaussian_ols_loglik(y, np.column_stack([np.ones(N), X])), rel=1e-12)
    hc1 = ols_spatial(y, X, W, cov_type="hc1")
    assert not np.allclose(hc1.bse.to_numpy(), res.bse.to_numpy())


@pytest.mark.reference
def test_spreg_goldens_are_still_live(lattice):
    """Re-derive every frozen literal from a live spreg (opt-in, ``-m reference``)."""
    spreg = pytest.importorskip("spreg")
    libpysal = pytest.importorskip("libpysal")
    y, X, W = lattice
    w = libpysal.weights.W({u: list(v) for u, v in _rook_grid(SIDE).items()})
    w.transform = "r"
    order = list(w.id_order)
    yr = pd.Series(y, index=W.ids).loc[order].to_numpy().reshape(-1, 1)
    Xr = pd.DataFrame(X, index=W.ids).loc[order].to_numpy()
    m = spreg.ML_Lag(yr, Xr, w=w, method="full")
    np.testing.assert_allclose(np.asarray(m.betas).ravel(), GOLDEN["ml_lag"]["betas"], rtol=1e-10)
    np.testing.assert_allclose(np.asarray(m.std_err).ravel(), GOLDEN["ml_lag"]["se"], rtol=1e-10)
    me = spreg.ML_Error(yr, Xr, w=w, method="full")
    np.testing.assert_allclose(np.asarray(me.betas).ravel(), GOLDEN["ml_error"]["betas"], rtol=1e-9)
    md = spreg.ML_Lag(yr, Xr, w=w, method="full", slx_lags=1)
    np.testing.assert_allclose(np.asarray(md.betas).ravel(), GOLDEN["ml_sdm"]["betas"], rtol=1e-9)
    g = spreg.GM_Lag(yr, Xr, w=w, w_lags=2)
    np.testing.assert_allclose(np.asarray(g.betas).ravel(), GOLDEN["gm_lag"]["betas"], rtol=1e-10)
    gs = spreg.GM_Lag(yr, Xr, w=w, w_lags=2, slx_lags=1)
    np.testing.assert_allclose(np.asarray(gs.betas).ravel(), GOLDEN["gm_sdm"]["betas"], rtol=1e-10)
    o = spreg.OLS(yr, Xr, w=w, spat_diag=True, moran=True)
    assert float(o.lm_lag[0]) == pytest.approx(GOLDEN["ols"]["lm_lag"][0], rel=1e-10)
    assert float(o.moran_res[0]) == pytest.approx(GOLDEN["ols"]["moran"][0], rel=1e-10)


# ===========================================================================
# 2. Closed-form identities — the module as its own reference
# ===========================================================================
@pytest.mark.parametrize("rho_true", [-0.6, -0.2, 0.0, 0.3, 0.7])
def test_concentrated_maximiser_equals_the_analytic_score_root(rho_true):
    """The bounded optimum and the ``brentq`` root of the score agree.

    The tolerance is the achievable one. A bounded Brent minimiser cannot
    localise the argmax of a smooth function better than ~``sqrt(eps)`` in
    ``rho``, and the score there is that error times the observed information
    (order ``n * I_rho,rho``), so an absolute 1e-7 bound on the *score* is below
    the floor. The score is therefore checked **relative to the information**.
    """
    y, X, W = _lattice_regression_data(seed=11, rho=rho_true)
    res = sar(y, X, W)
    assert res.score_root is not None
    assert abs(res.rho - res.score_root) < 1e-7
    lam = np.asarray(res.spectrum)
    Xc = np.column_stack([np.ones(N), X])
    Wy = W.W @ y
    XtX_inv = inv_xtx(Xc)
    e0 = y - Xc @ (XtX_inv @ (Xc.T @ y))
    eL = Wy - Xc @ (XtX_inv @ (Xc.T @ Wy))
    c0, c1, c2 = float(e0 @ e0), float(e0 @ eL), float(eL @ eL)
    r = res.rho
    ssr = c0 - 2 * r * c1 + r * r * c2
    score = float(-np.sum(np.real(lam / (1.0 - r * lam))) - N * (r * c2 - c1) / ssr)
    info_rr = res.vcov_names.index("rho")
    information = 1.0 / res.vcov[info_rr, info_rr]
    assert abs(score) / information < 1e-8


def test_the_score_root_disagreement_warning_is_reachable(lattice, monkeypatch):
    """``sar``'s Warns section advertises this warning, so it must be able to fire.

    It could not. The ``brentq`` bracket was ``[rho_hat +- 1e-6]`` while the
    gate demanded a disagreement ``> 1e-6``: any root the bracket could contain
    was inside the gate by construction, and any root outside it failed the
    sign test and left ``score_root is None`` with no warning either — dead
    code on both sides. The bracket is now 100x the gate. Here the bounded
    optimum is displaced by 1e-5 (inside the bracket, outside the gate), which
    is exactly the situation the docstring describes.
    """
    from puremacro.spatial import models as models_mod

    y, X, W = lattice
    base = sar(y, X, W)
    real = models_mod._maximise_profile

    def displaced(objective, bounds, xtol):
        rho_hat, converged, n_iter = real(objective, bounds, xtol)
        return rho_hat + 1e-5, converged, n_iter

    monkeypatch.setattr(models_mod, "_maximise_profile", displaced)
    with pytest.warns(RuntimeWarning, match="root of the analytic score"):
        res = sar(y, X, W)
    # the bounded optimum is kept, the score root is reported alongside it
    assert res.rho == pytest.approx(base.rho + 1e-5, abs=1e-7)
    assert res.score_root == pytest.approx(base.rho, abs=1e-7)
    assert abs(res.score_root - res.rho) == pytest.approx(1e-5, rel=1e-2)


def test_an_interior_optimum_with_no_sign_change_in_the_score_is_reported(lattice, monkeypatch):
    """The other half of the repair: silence used to be the only signal.

    When the bracket does not change sign the refinement cannot run. That is
    the case that actually indicates a problem, and it previously produced no
    warning at all — ``score_root`` simply came back ``None``.
    """
    from puremacro.spatial import models as models_mod

    y, X, W = lattice
    real = models_mod._maximise_profile

    def far_off(objective, bounds, xtol):
        rho_hat, converged, n_iter = real(objective, bounds, xtol)
        return rho_hat - 0.2, converged, n_iter          # still interior

    monkeypatch.setattr(models_mod, "_maximise_profile", far_off)
    with pytest.warns(RuntimeWarning, match="does not change sign"):
        res = sar(y, X, W)
    assert res.score_root is None
    assert res.at_bound is False


def test_a_bound_pinned_optimum_does_not_trigger_the_no_sign_change_warning():
    """An optimum on the boundary is not a stationary point; that is not news."""
    y, X, W = _lattice_regression_data()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = sar(y, X, W, rho_bounds=(-1e-9, 1e-9))
    assert res.at_bound is True
    assert res.score_root is None
    assert not [w for w in rec if "does not change sign" in str(w.message)]
    assert [w for w in rec if "sits on the boundary" in str(w.message)]


@pytest.mark.parametrize("fit", ["sar", "sdm", "sem"])
def test_loglik_at_zero_equals_the_gaussian_ols_loglik(lattice, fit):
    """``ln L_c(0)`` is the OLS Gaussian log-likelihood, to machine precision.

    Evaluated through the same objective the optimiser used — *not* interpolated
    off ``profile_loglik``, whose 201-point grid gives an ``O(h^2 L'')`` error of
    order 1e-3 on a curved profile.
    """
    y, X, W = lattice
    Xc = np.column_stack([np.ones(N), X])
    if fit == "sar":
        res, design = sar(y, X, W), Xc
    elif fit == "sdm":
        res, design = sdm(y, X, W), np.column_stack([Xc, W.W @ X])
    else:
        res, design = sem(y, X, W), Xc
    assert res.loglik_at_zero == pytest.approx(_gaussian_ols_loglik(y, design), abs=1e-10)
    # and the grid is genuinely only good to ~1e-3, which is why it is not used
    interp = float(np.interp(0.0, res.profile_grid, res.profile_loglik))
    assert abs(interp - res.loglik_at_zero) < 1e-1


def test_sdm_is_literally_sar_on_the_augmented_design(lattice):
    """``sdm(y, X, W) == sar(y, [X, W X_ex], W)`` to machine precision.

    Only the **non-constant** block is lagged: for a row-standardised
    island-free ``W``, ``W @ 1 = 1``, so lagging a design that already carries a
    constant produces a duplicate column and the fit never reaches an assertion.
    """
    y, X, W = lattice
    a = sdm(y, X, W)
    b = sar(y, np.column_stack([X, W.lag(X)]), W,
            x_names=["x1", "x2", "W_x1", "W_x2"])
    np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), atol=1e-12)
    np.testing.assert_allclose(a.vcov, b.vcov, atol=1e-12)
    assert a.rho == pytest.approx(b.rho, abs=1e-12)
    assert a.sigma2 == pytest.approx(b.sigma2, abs=1e-12)
    assert a.loglik == pytest.approx(b.loglik, abs=1e-12)


def test_sem_with_lambda_pinned_to_zero_is_ols(lattice):
    y, X, W = lattice
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = sem(y, X, W, lambda_bounds=(-1e-12, 1e-12))
    assert any("boundary" in str(w.message) for w in rec)
    Xc = np.column_stack([np.ones(N), X])
    b = inv_xtx(Xc) @ (Xc.T @ y)
    e = y - Xc @ b
    np.testing.assert_allclose(res.beta, b, atol=1e-10)
    assert res.sigma2 == pytest.approx(float(e @ e) / N, rel=1e-10)
    expect = np.sqrt(np.diag(res.sigma2 * inv_xtx(Xc)))
    np.testing.assert_allclose(res.bse.to_numpy()[:3], expect, rtol=1e-8)


def test_slx_is_ols_on_the_augmented_design(lattice):
    y, X, W = lattice
    res = slx(y, X, W)
    Z = np.column_stack([np.ones(N), X, W.lag(X)])
    b = inv_xtx(Z) @ (Z.T @ y)
    np.testing.assert_allclose(res.params.to_numpy(), b, atol=1e-12)
    assert res.rho is None and res.lam is None
    assert res.durbin_names == ("x1", "x2")
    e = y - Z @ b
    assert res.sigma2 == pytest.approx(float(e @ e) / (N - 5), rel=1e-12)


# ===========================================================================
# 3. LeSage-Pace impacts
# ===========================================================================
def test_effects_at_rho_zero_are_beta_and_zero_indirect():
    W = contiguity_weights(_rook_grid(5))
    beta = np.array([1.0, -0.5])
    eff = spatial_effects(W, beta, 0.0, names=["x1", "x2"])
    np.testing.assert_allclose(eff.direct, beta, atol=1e-14)
    np.testing.assert_allclose(eff.indirect, 0.0, atol=1e-14)
    np.testing.assert_allclose(eff.total, beta, atol=1e-14)
    assert np.all(np.isnan(eff.direct_se))  # n_sim=0 means NaN SEs, never a raise
    # with a Durbin block on a row-standardised island-free W, indirect == theta
    theta = np.array([0.3, -0.2])
    eff2 = spatial_effects(W, beta, 0.0, theta=theta, names=["x1", "x2"])
    np.testing.assert_allclose(eff2.direct, beta, atol=1e-14)
    np.testing.assert_allclose(eff2.indirect, theta, atol=1e-13)


@pytest.mark.parametrize("rho", [-0.8, -0.3, 0.0, 0.3, 0.7, 0.9])
def test_effects_total_has_the_closed_form_on_a_row_standardised_island_free_w(rho):
    W = contiguity_weights(_rook_grid(6))
    beta = np.array([1.0, -0.5])
    theta = np.array([0.4, 0.25])
    eff = spatial_effects(W, beta, rho, theta=theta, names=["x1", "x2"])
    np.testing.assert_allclose(eff.total, (beta + theta) / (1.0 - rho), rtol=1e-11)
    assert eff.s_solve_check < 1e-8


def test_one_island_breaks_the_closed_form_and_the_solve_route_wins():
    """``1'A^-1 1/n = 1/(1-rho)`` needs a row-standardised *island-free* W.

    An island's row of ``A`` is the identity row, so ``(A^-1 1)_i = 1`` while
    every connected unit gets ``1/(1 - rho)``. spreg's ``spmultiplier``
    hard-codes ``1/(1-rho)``; this module always computes ``1'A^-1 1``.
    """
    nb = _rook_grid(5)
    nb[25] = []
    W = contiguity_weights(nb)
    assert W.n == 26 and W.n_islands == 1
    eff = spatial_effects(W, np.array([1.0]), 0.6, names=["x1"])
    assert eff.total[0] == pytest.approx(2.4423076923076925, rel=1e-12)
    assert eff.total[0] != pytest.approx(1.0 / (1.0 - 0.6), rel=1e-4)
    # brute force agrees with the series
    Ainv = np.linalg.inv(np.eye(26) - 0.6 * W.to_dense())
    assert eff.total[0] == pytest.approx(Ainv.sum() / 26, rel=1e-12)
    assert eff.direct[0] == pytest.approx(np.trace(Ainv) / 26, rel=1e-12)


def test_effects_match_a_brute_force_dense_impact_matrix(lattice):
    """The tau/t_W/s/s_W reduction equals ``n^-1 tr S_r`` and ``n^-1 1'S_r 1``.

    This is the pin that replaces a spreg golden for the SDM split: spreg's
    ``spat_impacts='full'`` drops the ``theta_r tr(A^-1 W)/n`` term from the
    direct effect, so goldening the split there would pin the wrong number.
    """
    y, X, W = lattice
    res = sdm(y, X, W)
    eff = res.effects()
    Wd = W.to_dense()
    Ainv = np.linalg.inv(np.eye(N) - res.rho * Wd)
    names = list(res.exog_names)
    for i, v in enumerate(eff.variables):
        br = float(res.beta[names.index(v)])
        th = float(res.theta[list(res.durbin_names).index(v)])
        S = Ainv @ (br * np.eye(N) + th * Wd)
        assert eff.direct[i] == pytest.approx(np.trace(S) / N, rel=1e-11)
        assert eff.total[i] == pytest.approx(S.sum() / N, rel=1e-11)
        assert eff.indirect[i] == pytest.approx(eff.total[i] - eff.direct[i], rel=1e-11)
    assert eff.has_durbin
    assert "const" not in eff.variables


def test_sar_effects_agree_with_the_spreg_convention_when_theta_is_zero(lattice):
    """With ``theta = 0`` the divergence from spreg vanishes by construction."""
    y, X, W = lattice
    res = sar(y, X, W)
    eff = res.effects()
    rho = res.rho
    # row-standardised, island-free: spreg's 1/(1-rho) shortcut is valid here
    np.testing.assert_allclose(eff.total, res.beta[1:] / (1.0 - rho), rtol=1e-11)
    Ainv = np.linalg.inv(np.eye(N) - rho * W.to_dense())
    np.testing.assert_allclose(eff.direct, res.beta[1:] * np.trace(Ainv) / N, rtol=1e-11)
    assert not eff.has_durbin


def test_slx_effects_are_local_with_no_multiplier(lattice):
    y, X, W = lattice
    res = slx(y, X, W)
    eff = res.effects()
    np.testing.assert_allclose(eff.direct, res.beta[1:], atol=1e-13)
    np.testing.assert_allclose(eff.indirect, res.theta * W.s0 / N, atol=1e-13)
    assert eff.rho == 0.0


def test_simulated_total_se_matches_its_closed_form_when_only_beta_varies():
    """A sharp check on the simulation machinery, not a calibration study.

    With ``theta = 0`` and zero variance on ``rho``, ``total_r = beta_r s/n`` is
    linear in ``beta_r``, so the simulated standard deviation must equal
    ``se(beta_r) * s/n`` up to Monte-Carlo error.
    """
    W = contiguity_weights(_rook_grid(6))
    n = W.n
    beta = np.array([1.0, -0.5])
    se_beta = np.array([0.12, 0.07])
    V = np.zeros((2 * 2 + 1, 2 * 2 + 1))
    V[0, 0], V[1, 1] = se_beta[0] ** 2, se_beta[1] ** 2
    rho = 0.4
    eff = spatial_effects(W, beta, rho, vcov=V, names=["x1", "x2"], n_sim=4000, seed=3)
    expect = se_beta * eff.s / n
    np.testing.assert_allclose(eff.total_se, expect, rtol=0.05)
    np.testing.assert_allclose(eff.direct_se, se_beta * eff.tau / n, rtol=0.05)
    assert eff.n_kept == 4000 and eff.n_dropped == 0
    # the quantile interval brackets the point estimate
    assert np.all(eff.total_ci[:, 0] < eff.total)
    assert np.all(eff.total < eff.total_ci[:, 1])


def test_effects_simulation_is_deterministic_in_the_seed(lattice):
    y, X, W = lattice
    res = sar(y, X, W)
    a = res.effects(n_sim=200, seed=7)
    b = res.effects(n_sim=200, seed=7)
    c = res.effects(n_sim=200, seed=8)
    np.testing.assert_array_equal(a.direct_se, b.direct_se)
    np.testing.assert_array_equal(a.total_ci, b.total_ci)
    assert not np.allclose(a.direct_se, c.direct_se)


def test_effects_keep_draws_and_simulation_p_values_agree_with_the_intervals(lattice):
    y, X, W = lattice
    res = sar(y, X, W)
    eff = res.effects(n_sim=500, seed=1, keep_draws=True)
    assert eff.draws is not None and eff.draws.shape == (eff.n_kept, 2, 3)
    np.testing.assert_allclose(eff.draws[:, :, 0].std(axis=0, ddof=1), eff.direct_se,
                               rtol=1e-12)
    # a simulation p-value below alpha and an interval excluding zero cannot disagree
    for i in range(len(eff.variables)):
        excludes_zero = eff.direct_ci[i, 0] > 0 or eff.direct_ci[i, 1] < 0
        assert excludes_zero == (eff.direct_p[i] < eff.alpha)


def test_power_series_and_sparse_solve_agree_on_s(lattice):
    """``s = sum_k rho^k m_k`` is cross-checked against one ``spsolve``."""
    _, _, W = lattice
    m = _row_power_sums(W, 5)
    np.testing.assert_allclose(m, np.full(6, float(N)), rtol=1e-12)  # row-standardised
    for rho in (-0.7, 0.0, 0.5, 0.85):
        eff = spatial_effects(W, np.array([1.0]), rho, names=["x1"])
        assert eff.s_solve_check < 1e-8
        assert eff.trace_method == "spectrum+series"
    # and the truncation order is genuinely finite there, which is what selects
    # the series branch over the sparse-solve one
    assert _series_order(1.0, 0.85) is not None


def test_the_sparse_solve_branch_takes_over_when_the_series_will_not_converge(lattice):
    """``trace_method == 'spectrum+solve'`` — the fallback nothing else covers.

    The series is truncated at :data:`_MAX_SERIES_ORDER`; on a row-standardised
    ``W`` (spectral radius exactly 1) that budget is exhausted just below
    ``rho = 1``, and ``s = 1'A^-1 1`` then has to come from a sparse solve.
    Every other effects test in this file runs at a ``rho`` where the series
    wins, so without this the branch has no coverage at all.
    """
    _, _, W = lattice
    assert _series_order(1.0, 0.999) is None            # the switch condition
    b = np.array([1.0, -0.5])
    th = np.array([0.25, 0.75])
    Wd = W.to_dense()
    for rho in (0.999, 0.9995):
        eff = spatial_effects(W, b, rho, theta=th, names=["x1", "x2"])
        assert eff.trace_method == "spectrum+solve"
        assert eff.s_solve_check < 1e-8
        # against a fully dense brute force, not against the series
        A_inv = np.linalg.inv(np.eye(N) - rho * Wd)
        ones = np.ones(N)
        for j, (bj, tj) in enumerate(zip(b, th)):
            S = A_inv @ (bj * np.eye(N) + tj * Wd)
            assert eff.direct[j] == pytest.approx(float(np.trace(S)) / N, rel=1e-9)
            assert eff.total[j] == pytest.approx(float(ones @ S @ ones) / N, rel=1e-9)
            assert eff.indirect[j] == pytest.approx(eff.total[j] - eff.direct[j], rel=1e-9)


def test_the_sparse_solve_branch_also_serves_the_simulation_loop(lattice):
    """Every draw re-enters ``_scalar_terms`` with ``m=None``; none may fail."""
    _, _, W = lattice
    b = np.array([1.0, -0.5])
    th = np.array([0.25, 0.75])
    V = np.eye(5) * 1e-4
    # at rho = 0.999 roughly half the normal draws land outside (-1, 1); they
    # are dropped, counted and warned about rather than clipped
    with pytest.warns(RuntimeWarning, match="fell outside the admissible interval"):
        eff = spatial_effects(W, b, 0.999, theta=th, names=["x1", "x2"],
                              vcov=V, n_sim=200, seed=3, keep_draws=True)
    assert eff.trace_method == "spectrum+solve"
    assert eff.n_kept > 50 and eff.n_kept + eff.n_dropped == 200
    assert eff.draws is not None and eff.draws.shape == (eff.n_kept, 2, 3)
    assert np.all(np.isfinite(eff.draws))
    np.testing.assert_allclose(eff.draws[:, :, 0].std(axis=0, ddof=1), eff.direct_se, rtol=1e-12)
    # reproducible on the seed
    with pytest.warns(RuntimeWarning, match="fell outside the admissible interval"):
        again = spatial_effects(W, b, 0.999, theta=th, names=["x1", "x2"],
                                vcov=V, n_sim=200, seed=3)
    np.testing.assert_array_equal(again.total, eff.total)
    np.testing.assert_array_equal(again.total_se, eff.total_se)


# ===========================================================================
# 4. Numerics: log-determinant routes, bounds, covariance
# ===========================================================================
#: Measured absolute error of ``logdet='chebyshev'`` at ``order=5`` against the
#: exact eigenvalue log-determinant, on a 20x20 rook lattice (n = 400,
#: row-standardised). These are deterministic dense-linear-algebra numbers, so
#: they are pinned tightly rather than bounded loosely: a *loose* bound is not
#: coverage. The earlier ``0.05 * n * rho**4`` envelope was 13.1 at rho = 0.9
#: against a true error of 0.66 and 0.162 at rho = 0.3 against 2.0e-4 — three
#: orders of headroom, i.e. a parametrised loop that could not fail.
CHEBYSHEV_ORDER5_ERROR = {
    -0.9: 6.630057e-01,
    -0.5: 5.249228e-03,
    0.0: 0.0,
    0.3: 1.953961e-04,
    0.5: 5.249228e-03,
    0.9: 6.630057e-01,
}


@pytest.mark.parametrize("rho", sorted(CHEBYSHEV_ORDER5_ERROR))
def test_logdet_routes_agree_and_chebyshev_error_is_pinned(rho):
    W = contiguity_weights(_rook_grid(20))
    assert W.n == 400
    eig = _LogDet(W, method="eig")
    lu = _LogDet(W, method="lu")
    cheb = _LogDet(W, method="chebyshev", order=5)
    exact = eig(rho)
    # the two exact routes are the same number
    assert lu(rho) == pytest.approx(exact, abs=1e-10)
    err = abs(cheb(rho) - exact)
    expected = CHEBYSHEV_ORDER5_ERROR[rho]
    if expected == 0.0:
        assert err < 1e-14  # ln|I| = 0 is exact on every route
    else:
        assert err == pytest.approx(expected, rel=1e-5), f"rho={rho}: {err:.6e}"


def test_chebyshev_error_grows_with_rho_and_is_even_in_it():
    """The degradation has a shape, and the docstring's claims are pinned."""
    W = contiguity_weights(_rook_grid(20))
    eig = _LogDet(W, method="eig")
    cheb = _LogDet(W, method="chebyshev", order=5)
    err = {r: abs(cheb(r) - eig(r)) for r in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)}
    vals = [err[r] for r in sorted(err)]
    assert all(a < b for a, b in zip(vals, vals[1:])), err  # strictly increasing
    for r in (0.3, 0.5, 0.9):  # the lattice W is symmetric, so the error is even
        assert abs(cheb(-r) - eig(-r)) == pytest.approx(err[r], rel=1e-12)
    # the module docstring's headline claim: usable to ~1e-2 up to |rho| = 0.5,
    # order-of-magnitude only beyond it.
    assert err[0.5] < 1e-2
    assert err[0.9] > 0.1


def test_auto_logdet_switches_at_n_1000(lattice):
    y, X, W = lattice
    assert sar(y, X, W).logdet_method == "eig"
    assert _LogDet(W, method="auto").method == "eig"
    big = contiguity_weights(_rook_grid(33))  # n = 1089 > 1000
    assert _LogDet(big, method="auto").method == "lu"


def test_explicit_bounds_with_the_lu_route_skip_the_dense_eigendecomposition():
    """``logdet='lu'`` is only cheap if nothing else forces the spectrum.

    ``rho_bounds='auto'`` is the one other consumer, so an explicit interval
    plus ``'lu'`` must do no eigendecomposition at all — and must still land on
    the same optimum as the exact ``'eig'`` route.
    """
    y, X, W = _lattice_regression_data(side=10, seed=5)
    fast = sar(y, X, W, logdet="lu", rho_bounds=(-0.99, 0.99))
    assert fast.spectrum is None
    assert np.isnan(fast.rho_bounds_singularity).all()
    assert fast.score_root is None  # the refinement needs the spectrum
    exact = sar(y, X, W)
    assert exact.spectrum is not None and exact.score_root is not None
    assert fast.rho == pytest.approx(exact.rho, abs=1e-7)
    np.testing.assert_allclose(fast.bse.to_numpy(), exact.bse.to_numpy(), rtol=1e-8)
    np.testing.assert_allclose(fast.params.to_numpy(), exact.params.to_numpy(), atol=1e-7)


def test_chebyshev_order_is_validated_and_only_reaches_symmetrisable_weights(lattice):
    y, X, W = lattice
    with pytest.raises(ValueError, match="logdet_order"):
        sar(y, X, W, logdet="eig", logdet_order=5)
    with pytest.raises(ValueError, match=r"logdet_order must lie in \[1, 8\]"):
        sar(y, X, W, logdet="chebyshev", logdet_order=16)
    rng = np.random.default_rng(0)
    Wk = knn_weights(rng.uniform(0, 10, (30, 2)), 4, metric="euclidean")
    Xk = rng.normal(size=(30, 2))
    yk = 1.0 + Xk @ [1.0, -0.5] + rng.normal(size=30)
    with pytest.raises(ValueError, match="symmetrisable"):
        sar(yk, Xk, Wk, logdet="chebyshev")


def test_rho_bounds_intersect_the_spectral_radius_interval():
    """The real-eigenvalue interval alone is useless as an optimiser bracket."""
    rng = np.random.default_rng(4)
    Wk = knn_weights(rng.uniform(0, 10, (40, 2)), 4, metric="euclidean")
    spec, symmetrisable = _spectrum(Wk)
    assert not symmetrisable and np.abs(spec.imag).max() > 1e-3
    searched, singular = _admissible_bounds(spec)
    assert singular[0] < -1.5  # many units wide on one side
    assert -1.0 <= searched[0] and searched[1] <= 1.0
    assert searched[0] > singular[0]
    Xk = rng.normal(size=(40, 2))
    A = np.linalg.inv(np.eye(40) - 0.4 * Wk.to_dense())
    yk = A @ (1.0 + Xk @ [1.0, -0.5] + rng.normal(size=40))
    res = sar(yk, Xk, Wk)
    assert res.rho_bounds == pytest.approx(searched)
    assert res.rho_bounds_singularity == pytest.approx(singular)
    assert res.rho_bounds[0] > res.rho_bounds_singularity[0]
    assert np.iscomplexobj(res.spectrum) and np.abs(res.spectrum.imag).max() > 1e-3


def test_binary_weights_change_the_scale_of_rho():
    W = contiguity_weights(_rook_grid(6), row_standardize=False)
    spec, _ = _spectrum(W)
    searched, _ = _admissible_bounds(spec)
    assert searched[1] == pytest.approx(1.0 / spec.max(), rel=1e-6)
    assert searched[1] < 0.28  # 1/3.6039 on a 6x6 rook lattice


def test_symmetrizing_scale_accepts_only_a_verified_similarity():
    """The pattern is a cheap early-out; the acceptance test is the norm check."""
    W = contiguity_weights(_rook_grid(5))
    d = _symmetrizing_scale(W)
    assert d is not None and np.all(d > 0)
    root = np.sqrt(d)
    S = (sp.diags(root) @ W.W @ sp.diags(1.0 / root)).toarray()
    np.testing.assert_allclose(S, S.T, atol=1e-13)
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(S)),
                               np.sort(np.linalg.eigvals(W.to_dense()).real), atol=1e-13)
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10, (30, 2))
    assert _symmetrizing_scale(distance_weights(coords, 4.0, metric="euclidean")) is not None
    assert _symmetrizing_scale(contiguity_weights(_rook_grid(4), row_standardize=False)) is not None
    # knn: the pattern is asymmetric
    assert _symmetrizing_scale(knn_weights(coords, 4, metric="euclidean")) is None
    # economic flows: the pattern is PERFECTLY symmetric and it is still rejected,
    # because the verification, not the pattern, is what decides.
    flows = rng.uniform(0.1, 1.0, (25, 25))
    We = economic_weights(flows)
    pattern = (We.W != 0).astype(float)
    assert abs(pattern - pattern.T).max() == 0.0
    assert _symmetrizing_scale(We) is None
    assert np.abs(_spectrum(We)[0].imag).max() > 1e-3


def test_analytic_and_hessian_covariance_are_the_same_asymptotic_object():
    """They differ by ``O_p(n^-1/2)``; the gap shrinks with the lattice."""
    gaps = {}
    for side in (6, 14):
        y, X, W = _lattice_regression_data(side=side, seed=7)
        a = sar(y, X, W)
        h = sar(y, X, W, vcov="hessian")
        assert h.vcov_type == "hessian"
        gaps[side] = float(np.abs(h.bse.to_numpy() / a.bse.to_numpy() - 1.0).max())
    assert gaps[14] < 0.05
    assert gaps[14] < gaps[6]


def test_obs_weights_of_ones_reproduce_the_unweighted_fit_bit_for_bit(lattice):
    """``P = I`` must change nothing at all.

    This test is *blind to the weighting convention* by construction — with
    ``P = I`` the correct ``P(Wy)`` and the wrong ``W(Py)`` are the same vector
    — and so is the numerical-Hessian agreement below, whose tolerance is far
    wider than the gap the two conventions produce. The convention is pinned by
    :func:`test_the_weighted_likelihood_lags_first_and_weights_second`, which
    is where the teeth are.
    """
    y, X, W = lattice
    base = sar(y, X, W)
    same = sar(y, X, W, obs_weights=np.ones(N))
    np.testing.assert_array_equal(same.params.to_numpy(), base.params.to_numpy())
    np.testing.assert_array_equal(same.vcov, base.vcov)
    assert same.loglik == base.loglik


def test_weighted_information_matrix_matches_the_weighted_numerical_hessian(lattice):
    y, X, W = lattice
    rng = np.random.default_rng(2)
    w = rng.uniform(0.4, 2.5, N)
    a = sar(y, X, W, obs_weights=w)
    h = sar(y, X, W, obs_weights=w, vcov="hessian")
    assert a.obs_weights is not None
    assert a.obs_weights.mean() == pytest.approx(1.0, rel=1e-12)
    # measured max relative gap on this DGP is 0.0798; 0.12 leaves ~50% margin
    # for the fixed-step stencil, not the 3x the old rtol=0.25 allowed.
    np.testing.assert_allclose(h.bse.to_numpy(), a.bse.to_numpy(), rtol=0.12)
    # weighting genuinely changes the fit
    assert not np.allclose(a.params.to_numpy(), sar(y, X, W).params.to_numpy())


def _weighted_profile_loglik(y, X, W, w, *, lag_first: bool):
    """Reference weighted concentrated log-likelihood, written out longhand.

    ``lag_first=True`` is the convention the module must implement: form the
    spatial lag on the *unweighted* ``y`` and weight it afterwards, ``P(W y)``.
    ``lag_first=False`` is the literal misreading of "replace ``y`` by ``P y``",
    ``W(P y)``, which the design's own critique (finding 17) called out.
    """
    n = y.shape[0]
    ow = np.asarray(w, dtype=float)
    ow = ow / ow.mean()                       # the module normalises to mean one
    P = np.sqrt(ow)
    Wd = W.to_dense()
    Z = np.column_stack([np.ones(n), X])
    Zw = Z * P[:, None]
    yw = y * P
    lagged = (Wd @ y) * P if lag_first else Wd @ (y * P)
    resid_maker = np.eye(n) - Zw @ np.linalg.solve(Zw.T @ Zw, Zw.T)
    e0, eL = resid_maker @ yw, resid_maker @ lagged
    c0, c1, c2 = float(e0 @ e0), float(e0 @ eL), float(eL @ eL)
    lam = np.linalg.eigvals(Wd)
    const = -0.5 * n * (np.log(2.0 * np.pi) + 1.0) + 0.5 * float(np.sum(np.log(ow)))

    def objective(rho: float) -> float:
        ssr = c0 - 2.0 * rho * c1 + rho * rho * c2
        return (const + float(np.sum(np.log(np.abs(1.0 - rho * lam))))
                - 0.5 * n * np.log(ssr / n))

    return objective


def test_the_weighted_likelihood_lags_first_and_weights_second(lattice):
    """``obs_weights`` must give ``P(W y)``, never ``W(P y)``.

    The two conventions are *not* a rounding difference: on this DGP they put
    the maximum 0.093 apart in ``rho`` (a 22% error in the headline parameter)
    and 0.84 apart in the log-likelihood. Nothing else in this file can see
    that — the ones-weights test has ``P = I``, under which the conventions
    coincide, and the Hessian-agreement tolerance is an order of magnitude
    wider than the gap.
    """
    from scipy import optimize

    y, X, W = lattice
    rng = np.random.default_rng(2)
    w = rng.uniform(0.4, 2.5, N)
    res = sar(y, X, W, obs_weights=w)

    def argmax(objective) -> float:
        opt = optimize.minimize_scalar(
            lambda r: -objective(float(r)), bounds=(-0.999, 0.999),
            method="bounded", options={"xatol": 1e-12},
        )
        return float(opt.x)

    right = _weighted_profile_loglik(y, X, W, w, lag_first=True)
    wrong = _weighted_profile_loglik(y, X, W, w, lag_first=False)
    rho_right, rho_wrong = argmax(right), argmax(wrong)

    # (a) the test has teeth: the two conventions genuinely disagree here
    assert abs(rho_right - rho_wrong) > 0.05, (rho_right, rho_wrong)
    # (b) the module implements the right one, to the optimiser's precision
    assert res.rho == pytest.approx(rho_right, abs=1e-6)
    assert abs(res.rho - rho_wrong) > 0.05
    # (c) and reports the matching maximised value of that same objective
    assert res.loglik == pytest.approx(right(res.rho), rel=1e-12)
    assert right(res.rho) > right(rho_wrong)


def test_the_weighted_information_matrix_uses_the_same_weight_orientation(lattice):
    """``diag(w)`` enters the information as ``X' diag(w) X``, not transposed.

    Pins the second surviving mutation the auditor found: swapping the
    observation-weight ratio inside the trace terms left every other weighted
    test green. Here the whole weighted information matrix is rebuilt longhand
    from the analytic SAR blocks and compared element by element.
    """
    y, X, W = lattice
    rng = np.random.default_rng(5)
    w = rng.uniform(0.4, 2.5, N)
    w = w / w.mean()
    res = sar(y, X, W, obs_weights=w)
    n, rho, s2 = N, res.rho, res.sigma2
    Z = np.column_stack([np.ones(n), X])
    Wd = W.to_dense()
    A = np.eye(n) - rho * Wd
    G = Wd @ np.linalg.inv(A)                      # W A^-1
    Zb = Z @ res.beta
    GZb = G @ Zb
    D = np.diag(w)
    k = Z.shape[1]
    info = np.zeros((k + 2, k + 2))
    info[:k, :k] = Z.T @ D @ Z / s2
    info[:k, k] = info[k, :k] = Z.T @ D @ GZb / s2
    info[k, k] = (float(GZb @ (D @ GZb)) / s2
                  + float(np.trace(G @ G)) + float(np.trace(G.T @ (D @ G @ np.diag(1.0 / w)))))
    info[k, k + 1] = info[k + 1, k] = float(np.trace(G)) / s2
    info[k + 1, k + 1] = n / (2.0 * s2 * s2)
    cov = np.linalg.inv(info)
    np.testing.assert_allclose(res.bse.to_numpy(),
                               np.sqrt(np.diag(cov))[: k + 1], rtol=1e-10)


def test_ml_recovers_a_planted_rho_on_a_seeded_dgp():
    """120 replications at n = 100 and 60 at n = 256, rho = 0.5, seed 99.

    The seeds are fixed, so the two means below are deterministic: they are
    regression pins on this experiment, not fresh statistical evidence. The
    absolute bias bound is what has teeth — the previous ``6 * se + 0.03``
    (~0.078 against a measured bias of 0.017) would have passed an estimator
    that systematically returned 0.43.
    """
    def mean_rho(side: int, reps: int) -> tuple[float, float]:
        nb = _rook_grid(side)
        W = contiguity_weights(nb)
        n = side * side
        Wd = W.to_dense()
        A = np.linalg.inv(np.eye(n) - 0.5 * Wd)
        rng = np.random.default_rng(99)
        out = []
        for _ in range(reps):
            X = rng.normal(size=(n, 2))
            y = A @ (1.0 + X @ np.array([1.0, -0.5]) + rng.normal(size=n))
            out.append(sar(y, X, W).rho)
        return float(np.mean(out)), float(np.std(out, ddof=1))

    m10, sd10 = mean_rho(10, 120)
    m16, sd16 = mean_rho(16, 60)
    # downward bias, and small in absolute terms at n = 100
    assert m10 < 0.5
    assert abs(m10 - 0.5) < 0.03
    assert abs(m16 - 0.5) < 0.02
    # deterministic pins on this seeded experiment
    assert m10 == pytest.approx(0.482960, abs=2e-3)
    assert m16 == pytest.approx(0.495308, abs=2e-3)
    # sampling dispersion shrinks with n (0.0814 -> 0.0460 as measured).
    # NOTE: monotone *bias* is deliberately not asserted across two lattice
    # sizes at these replication counts — at side 14 the measured bias is
    # -0.0234 against -0.0170 at side 10, a difference well inside one
    # standard error, so the old `abs(m14 - .5) < abs(m10 - .5) + 0.01`
    # assertion only ever passed on its slack.
    assert sd16 < 0.7 * sd10


# ===========================================================================
# 5. Adversarial inputs
# ===========================================================================
def test_islands_warn_and_the_fit_still_runs():
    nb = _rook_grid(5)
    nb[25] = []
    W = contiguity_weights(nb)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(26, 2))
    A = np.linalg.inv(np.eye(26) - 0.5 * W.to_dense())
    y = A @ (1.0 + X @ [1.0, -0.5] + rng.normal(size=26))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = sar(y, X, W)
    assert any("no neighbours" in str(w.message) for w in rec)
    assert res.n_islands == 1
    assert np.isfinite(res.rho) and res.rho_bounds[1] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.isfinite(res.bse.to_numpy()))


def test_a_disconnected_graph_fits_as_one_model():
    nb = {u: [v for v in (u - 1, u + 1) if 0 <= v < 6] for u in range(6)}
    nb.update({u: [v for v in (u - 1, u + 1) if 6 <= v < 12] for u in range(6, 12)})
    W = contiguity_weights(nb)
    assert W.n_islands == 0
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 2))
    y = 1.0 + X @ [1.0, -0.5] + rng.normal(scale=0.4, size=12)
    res = sar(y, X, W)
    assert np.isfinite(res.rho) and np.all(np.isfinite(res.bse.to_numpy()))
    # the similarity transform must be built component by component
    d = _symmetrizing_scale(W)
    assert d is not None and np.all(d > 0)


def test_duplicate_coordinates_do_not_break_the_fit():
    dup = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                    [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        W = distance_weights(dup, cutoff=1.5, metric="euclidean", decay="inverse")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 1))
    y = 1.0 + X[:, 0] + rng.normal(scale=0.2, size=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = sar(y, X, W)
    assert np.isfinite(res.rho)
    assert np.all(np.isfinite(res.resid))


def test_a_single_regressor_still_fits_and_the_robust_lm_guard_can_fire():
    W = contiguity_weights(_rook_grid(5))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(25, 1))
    y = 1.0 + X[:, 0] + rng.normal(scale=0.3, size=25)
    res = sar(y, X, W)
    assert list(res.params.index) == ["const", "x1", "rho"]
    # a regressor that is an eigenvector of a SYMMETRIC W puts W X b inside the
    # column space of X, which kills the robust LM-lag denominator nJ - T
    Wb = contiguity_weights(_rook_grid(5), row_standardize=False)
    assert Wb.is_symmetric
    vec = np.linalg.eigh(Wb.to_dense())[1][:, -1]
    with pytest.raises(ValueError, match="robust LM-lag denominator"):
        lm_spatial_tests(y, vec.reshape(-1, 1), Wb, add_constant=False)


def test_a_constant_regressor_is_rejected_rather_than_silently_dropped(lattice):
    y, X, W = lattice
    Xc = np.column_stack([np.ones(N), X])
    with pytest.raises(ValueError, match="already contains a constant"):
        sar(y, Xc, W)
    # ... and is accepted, once, with add_constant=False
    res = sar(y, Xc, W, add_constant=False, x_names=["const", "x1", "x2"])
    assert res.const_index == 0
    assert res.params.to_numpy() == pytest.approx(sar(y, X, W).params.to_numpy(), abs=1e-10)


def test_an_all_island_weights_matrix_raises_for_every_spatial_model():
    W = SpatialWeights(sp.csr_matrix((6, 6)), tuple(range(6)))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 1))
    y = rng.normal(size=6)
    for fn in (sar, sem, sdm, lm_spatial_tests):
        with pytest.raises(ValueError, match="no neighbours"):
            fn(y, X, W)


def test_complex_spectrum_is_handled_by_the_log_determinant():
    rng = np.random.default_rng(4)
    Wk = knn_weights(rng.uniform(0, 10, (40, 2)), 4, metric="euclidean")
    spec, _ = _spectrum(Wk)
    assert np.abs(spec.imag).max() > 1e-3
    ld = _LogDet(Wk, method="eig")
    for rho in (-0.5, 0.3, 0.8):
        manual = 0.5 * np.sum(np.log((1 - rho * spec.real) ** 2 + (rho * spec.imag) ** 2))
        assert ld(rho) == pytest.approx(float(manual), abs=1e-12)
        assert ld(rho) == pytest.approx(_LogDet(Wk, method="lu")(rho), abs=1e-9)


def test_boundary_rho_flags_at_bound_and_warns():
    W = contiguity_weights(_rook_grid(5))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(25, 2))
    y = 1.0 + X @ [1.0, -0.5] + rng.normal(scale=0.2, size=25)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = sar(y, X, W, rho_bounds=(-1e-9, 1e-9))
    assert res.at_bound
    assert any("boundary" in str(w.message) for w in rec)


# ===========================================================================
# 6. Contracts: validation, no kwargs sink, presentation
# ===========================================================================
def _tiny():
    W = contiguity_weights(_rook_grid(5))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(25, 2))
    y = 1.0 + X @ [1.0, -0.5] + rng.normal(scale=0.3, size=25)
    return y, X, W


@pytest.mark.parametrize("call,exc,match", [
    (lambda y, X, W: sar(y, X, "not weights"), TypeError, "SpatialWeights"),
    (lambda y, X, W: sar(y[:-1], X[:-1], W), ValueError, "sar"),
    (lambda y, X, W: sar(np.r_[y[:-1], np.nan], X, W), ValueError, "NaN or inf"),
    (lambda y, X, W: sar(y, np.c_[X, np.r_[X[:-1, 0], np.inf]], W), ValueError, "NaN or inf"),
    (lambda y, X, W: sar(y[:6], X[:6], W), ValueError, "observations but the weights"),
    (lambda y, X, W: sar(np.arange(5.0), np.eye(5)[:, :3],
                         contiguity_weights({i: [j for j in (i - 1, i + 1) if 0 <= j < 5]
                                             for i in range(5)})),
     ValueError, r"n > k \+ 2"),
    (lambda y, X, W: sar(y, X, W, method="bayes"), ValueError, "method must be"),
    (lambda y, X, W: sar(y, X, W, vcov="sandwich"), ValueError, "vcov must be"),
    (lambda y, X, W: sar(y, X, W, vcov="robust"), ValueError, "Lin & Lee"),
    (lambda y, X, W: sar(y, X, W, method="gmm", vcov="analytic"), ValueError, "vcov must be"),
    (lambda y, X, W: sar(y, X, W, logdet="montecarlo"), ValueError, "logdet must be"),
    (lambda y, X, W: sar(y, X, W, method="gmm", logdet="eig"), ValueError, "meaningless"),
    (lambda y, X, W: sar(y, X, W, gmm_lags=3), ValueError, "gmm_lags is meaningless"),
    (lambda y, X, W: sar(y, X, W, rho_bounds=(0.2, 0.8)), ValueError, "does not contain 0"),
    (lambda y, X, W: sar(y, X, W, rho_bounds=(0.8, -0.2)), ValueError, "increasing"),
    (lambda y, X, W: sar(y, X, W, alpha=0.0), ValueError, "alpha"),
    (lambda y, X, W: sar(y, X, W, obs_weights=np.zeros(25)), ValueError, "strictly positive"),
    (lambda y, X, W: sar(y, X, W, obs_weights=np.ones(3)), ValueError, "obs_weights has"),
    (lambda y, X, W: sar(y, X, W, method="gmm", obs_weights=np.ones(25)),
     ValueError, "maximum-likelihood feature"),
    (lambda y, X, W: sem(y, X, W, n_iter=2), ValueError, "n_iter is meaningless"),
    (lambda y, X, W: sem(y, X, W, method="gmm", n_iter=0), ValueError, "n_iter must be"),
    (lambda y, X, W: sdm(y, X, W, durbin=[]), ValueError, "durbin is empty"),
    (lambda y, X, W: sdm(y, X, W, durbin=["nope"]), KeyError, "not a regressor"),
    (lambda y, X, W: sdm(y, X, W, durbin=[7]), ValueError, "outside"),
    (lambda y, X, W: sdm(y, X, W, durbin=[-1]), ValueError, "outside"),
    (lambda y, X, W: sdm(y, X, W, durbin=["const"]), ValueError, "constant cannot enter"),
    (lambda y, X, W: sdm(y, X, W, durbin=["x1", "x1"]), ValueError, "twice"),
    (lambda y, X, W: sdm(y, X, W, durbin=["x1"]), ValueError, "FULL Durbin block"),
    (lambda y, X, W: ols_spatial(y, X, W, cov_type="conley"), ValueError, "cov_type must be"),
    (lambda y, X, W: slx(y, X, W, cov_type="conley"), ValueError, "cov_type must be"),
    (lambda y, X, W: spatial_effects(W, np.ones(2), 0.1, theta=np.ones(1)),
     ValueError, "aligned BY VARIABLE"),
    (lambda y, X, W: spatial_effects(W, np.ones(2), 0.1, n_sim=10),
     ValueError, "needs a covariance"),
    (lambda y, X, W: spatial_effects(W, np.ones(2), 5.0), ValueError, "lies outside"),
    (lambda y, X, W: spatial_effects(W, np.ones(2), 0.1, n_sim=-1), ValueError, "n_sim"),
    (lambda y, X, W: spatial_effects(W, np.ones(2), 0.1, vcov=np.eye(4), n_sim=5),
     ValueError, "vcov must be"),
    (lambda y, X, W: sar("y", X, W, data=pd.DataFrame({"y": y})),
     ValueError, "must be a column name"),
    (lambda y, X, W: sar("nope", ["a"], W, data=pd.DataFrame({"y": y, "a": X[:, 0]})),
     KeyError, "not in `data`"),
])
def test_degenerate_inputs_raise_with_an_actionable_message(call, exc, match):
    y, X, W = _tiny()
    with pytest.raises(exc, match=match):
        call(y, X, W)


@pytest.mark.parametrize("fn", [sar, sem, sdm, slx, ols_spatial])
def test_a_constant_outcome_is_rejected_instead_of_fitted(fn):
    """A degenerate outcome must raise, not produce a confident number.

    Before this was pinned, ``sem`` reported lambda with a significant z on a
    y with no variation at all, ``sar``/``sdm`` pinned rho to the admissible
    boundary and ``ols_spatial`` raised a bare ``ZeroDivisionError``.
    """
    y, X, W = _tiny()
    with pytest.raises(ValueError, match=r"outcome is constant"):
        fn(np.full_like(y, 3.0), X, W)


@pytest.mark.parametrize("fn", [sar, sem, sdm, slx, ols_spatial])
def test_an_outcome_the_regressors_fit_exactly_is_rejected(fn):
    y, X, W = _tiny()
    exact = 1.0 + X @ np.array([1.0, -0.5])          # zero residual by construction
    with pytest.raises(ValueError, match=r"no variation left after the regressors"):
        fn(exact, X, W)


def test_the_degenerate_outcome_message_names_the_caller_and_what_is_lost():
    y, X, W = _tiny()
    with pytest.raises(ValueError) as ei:
        sem(np.full_like(y, 3.0), X, W)
    assert str(ei.value).startswith("sem: ")
    assert "the spatial parameter is not identified" in str(ei.value)
    # slx and ols_spatial carry no spatial parameter, so the tail differs
    with pytest.raises(ValueError, match=r"slx: .*nothing is identified"):
        slx(np.full_like(y, 3.0), X, W)


def test_slx_with_an_empty_durbin_block_is_sent_to_ols_not_sar():
    """SLX minus its Durbin block is OLS; following ``sar()`` would add a rho."""
    y, X, W = _tiny()
    with pytest.raises(ValueError, match=r"durbin is empty.*ols_spatial\(\)"):
        slx(y, X, W, durbin=())
    with pytest.raises(ValueError, match=r"durbin is empty.*sar\(\)"):
        sdm(y, X, W, durbin=())

    const_only = np.full((W.n, 1), 2.0)
    with pytest.raises(ValueError, match=r"no non-constant regressor to lag.*ols_spatial\(\)"):
        slx(y, const_only, W, add_constant=False)
    with pytest.raises(ValueError, match=r"no non-constant regressor to lag.*sar\(\)"):
        sdm(y, const_only, W, add_constant=False)


def test_n_draws_is_an_accepted_alias_for_n_sim(lattice):
    y, X, W = lattice
    res = sar(y, X, W)
    a = res.effects(n_sim=200, seed=3)
    b = res.effects(n_draws=200, seed=3)
    assert a.n_sim == b.n_sim == 200
    assert np.allclose(a.indirect_se, b.indirect_se)

    big = np.eye(3) * 0.01
    c = spatial_effects(W, np.ones(2), 0.2, vcov=big, n_sim=50, seed=1)
    d = spatial_effects(W, np.ones(2), 0.2, vcov=big, n_draws=50, seed=1)
    assert c.n_sim == d.n_sim == 50
    assert np.allclose(c.total_se, d.total_se)

    # the defaults are untouched: neither spelling given still means 0 draws
    assert res.effects().n_sim == 0
    assert spatial_effects(W, np.ones(2), 0.2).n_sim == 0

    with pytest.raises(ValueError, match="not both"):
        res.effects(n_sim=10, n_draws=10)
    with pytest.raises(ValueError, match="not both"):
        spatial_effects(W, np.ones(2), 0.2, vcov=big, n_sim=10, n_draws=10)


def test_the_indirect_effect_is_the_off_diagonal_sum_over_n_not_their_mean(lattice):
    """``indirect = (1'S1 - tr S)/n``, which is (n-1) times the mean cross-partial."""
    y, X, W = lattice
    rho, beta = 0.5, np.array([1.0])
    eff = spatial_effects(W, beta, rho, names=["x1"])
    n = W.n
    S = np.linalg.inv(np.eye(n) - rho * W.to_dense()) * beta[0]
    off_sum = float(S.sum() - np.trace(S))
    assert eff.indirect[0] == pytest.approx(off_sum / n, rel=1e-10)
    mean_cross = off_sum / (n * (n - 1))
    assert eff.indirect[0] / mean_cross == pytest.approx(n - 1, rel=1e-10)

    text = sar(y, X, W).effects(n_sim=50, seed=0).summary()
    assert "SUM of all n(n-1) off-diagonal cross-partials" in text
    assert "divided by n" in text
    assert "average of all n(n-1)" not in text


def test_the_lm_battery_documents_the_real_dof_rescaling():
    from puremacro.spatial.models import SpatialLMResult, lm_spatial_tests as _lm
    for doc in (SpatialLMResult.__doc__, _lm.__doc__):
        assert "affine" in doc
        assert "((n - k)/n)^2" in doc
        assert "no constant at all" in doc


def test_effects_raises_for_sem_because_there_is_no_decomposition(lattice):
    y, X, W = lattice
    with pytest.raises(ValueError, match="no LeSage-Pace decomposition"):
        sem(y, X, W).effects()


@pytest.mark.parametrize("fn", [sar, sem, sdm, slx])
def test_a_rank_deficient_design_names_the_collinear_columns(lattice, fn):
    """Every estimator routes its Gram solve through ``inv_xtx`` (build rule G7).

    ``sem`` used to be the exception: it solved the spatially filtered normal
    equations with a bare ``np.linalg.solve``, so the user got numpy's
    context-free ``Singular matrix`` while ``sar``/``sdm``/``slx`` named the
    offending columns — and ``sem``'s own docstring promised the named message.
    Parametrised precisely so that regression cannot come back on one
    estimator while the others stay green.
    """
    y, X, W = lattice
    Xbad = np.column_stack([X, X[:, 0]])
    with pytest.raises(np.linalg.LinAlgError) as exc:
        fn(y, Xbad, W)
    msg = str(exc.value)
    assert msg.startswith(f"{fn.__name__}: X'X is singular"), msg
    assert "Columns most aligned with the null space" in msg
    # sar/sem fit [1, x1, x2, x1]; sdm/slx add the three lagged columns.
    expected_rank = "rank 3 of 4" if fn in (sar, sem) else "rank 5 of 7"
    assert expected_rank in msg, msg


@pytest.mark.parametrize("fn", [sar, sem, sdm, slx, ols_spatial, lm_spatial_tests])
def test_no_kwargs_sink(fn):
    y, X, W = _tiny()
    with pytest.raises(TypeError):
        fn(y, X, W, not_a_real_option=1)


def test_spatial_effects_has_no_kwargs_sink():
    _, _, W = _tiny()
    with pytest.raises(TypeError):
        spatial_effects(W, np.ones(2), 0.1, not_a_real_option=1)


def test_data_and_pandas_input_modes_agree(lattice):
    y, X, W = lattice
    frame = pd.DataFrame({"gdp": y, "a": X[:, 0], "b": X[:, 1]}, index=W.ids)
    from_frame = sar("gdp", ["a", "b"], W, data=frame)
    from_pandas = sar(frame["gdp"], frame[["a", "b"]], W)
    from_arrays = sar(y, X, W, x_names=["a", "b"])
    for other in (from_pandas, from_arrays):
        np.testing.assert_allclose(from_frame.params.to_numpy(), other.params.to_numpy(),
                                   atol=1e-12)
    assert list(from_frame.params.index) == ["const", "a", "b", "rho"]
    assert from_frame.y_name == "gdp"
    # a shuffled index is realigned by label, not by position
    shuffled = frame.sample(frac=1.0, random_state=0)
    assert list(shuffled.index) != list(W.ids)
    np.testing.assert_allclose(
        sar("gdp", ["a", "b"], W, data=shuffled).params.to_numpy(),
        from_frame.params.to_numpy(), atol=1e-12)
    with pytest.raises(KeyError, match="missing from"):
        sar("gdp", ["a", "b"], W, data=frame.iloc[1:])


@pytest.mark.parametrize("maker", [
    lambda y, X, W: sar(y, X, W),
    lambda y, X, W: sem(y, X, W),
    lambda y, X, W: sdm(y, X, W),
    lambda y, X, W: slx(y, X, W),
    lambda y, X, W: sar(y, X, W, method="gmm"),
    lambda y, X, W: ols_spatial(y, X, W),
    lambda y, X, W: lm_spatial_tests(y, X, W),
    lambda y, X, W: sar(y, X, W).effects(n_sim=50, seed=0),
])
def test_presentation_contract(lattice, maker):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y, X, W = lattice
    res = maker(y, X, W)
    assert isinstance(res, (SpatialModelResult, SpatialOLSResult, SpatialLMResult,
                            SpatialEffectsResult))
    text = res.summary()
    assert isinstance(text, str) and len(text) > 50
    df = res.to_frame()
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    assert isinstance(df.index, pd.RangeIndex)
    for fmt in ("markdown", "latex", "typst"):
        out = getattr(res, f"to_{fmt}")()
        assert isinstance(out, str) and len(out) > 20
    fig = res.plot()
    assert fig.__class__.__name__ == "Figure"
    plt.close(fig)


def test_result_objects_are_frozen(lattice):
    y, X, W = lattice
    res = sar(y, X, W)
    with pytest.raises(Exception):
        res.rho = 0.9  # type: ignore[misc]
    assert type(res).__dataclass_params__.frozen
    assert type(res).__dataclass_params__.eq is False


def test_residual_and_fitted_concepts_are_distinct_and_documented(lattice):
    y, X, W = lattice
    res = sar(y, X, W)
    np.testing.assert_allclose(res.resid_reduced, y - res.fitted, atol=1e-12)
    # the structural residual is the one sigma2 is built from
    assert res.sigma2 == pytest.approx(float(res.resid @ res.resid) / N, rel=1e-12)
    assert not np.allclose(res.resid, res.resid_reduced)
    # the reduced form really is A^-1 Z b
    Ainv = np.linalg.inv(np.eye(N) - res.rho * W.to_dense())
    Xc = np.column_stack([np.ones(N), X])
    np.testing.assert_allclose(res.fitted, Ainv @ (Xc @ res.beta), atol=1e-9)


def test_information_criteria_count_sigma2(lattice):
    y, X, W = lattice
    res = sar(y, X, W)
    assert res.n_params == 5  # 3 betas + rho + sigma2
    assert res.aic == pytest.approx(-2 * res.loglik + 2 * res.n_params, rel=1e-12)
    assert res.bic == pytest.approx(-2 * res.loglik + res.n_params * np.log(N), rel=1e-12)
    assert sdm(y, X, W).n_params == 7
    assert sem(y, X, W).n_params == 5


def test_gmm_robust_and_classical_differ_and_both_are_finite(lattice):
    y, X, W = lattice
    a = sar(y, X, W, method="gmm", vcov="classical")
    b = sar(y, X, W, method="gmm", vcov="robust")
    np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), atol=1e-14)
    assert np.all(np.isfinite(b.bse.to_numpy()))
    assert not np.allclose(a.bse.to_numpy(), b.bse.to_numpy())
    assert b.vcov_type == "robust"


@pytest.mark.parametrize("q", [1, 2, 3])
def test_gmm_lags_is_the_order_of_the_instrument_set(lattice, q):
    """``H = [1, X, W X, ..., W^q X]`` — reproduced longhand, not just perturbed.

    The old test asserted only that ``gmm_lags=1`` and ``gmm_lags=3`` give
    different coefficients, which any code that reads the argument at all would
    satisfy. Here the whole Kelejian-Prucha 2SLS is rebuilt from the explicit
    instrument matrix, so the *content* of the set and the meaning of ``q`` are
    both pinned.
    """
    y, X, W = lattice
    Wd = W.to_dense()
    Z_exog = np.column_stack([np.ones(N), X])
    Z = np.column_stack([Z_exog, Wd @ y])          # the endogenous Wy last
    blocks, cur = [Z_exog], X                      # X_ex excludes the constant
    for _ in range(q):
        cur = Wd @ cur
        blocks.append(cur)
    H = np.column_stack(blocks)
    assert H.shape[1] == 3 + 2 * q                 # 1 + 2 exog + 2 per lag order
    Zhat = H @ np.linalg.solve(H.T @ H, H.T @ Z)
    delta = np.linalg.solve(Zhat.T @ Zhat, Zhat.T @ y)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = sar(y, X, W, method="gmm", gmm_lags=q)
    assert not [w for w in rec if "instrument column" in str(w.message)]
    np.testing.assert_allclose(res.params.to_numpy(), delta, rtol=0, atol=1e-11)
    # sigma2 and the classical covariance come off the same projection
    e = y - Z @ delta
    assert res.sigma2 == pytest.approx(float(e @ e) / N, rel=1e-12)
    cov = res.sigma2 * np.linalg.inv(Zhat.T @ Zhat)
    np.testing.assert_allclose(res.bse.to_numpy(), np.sqrt(np.diag(cov)), rtol=1e-10)


def test_gmm_lags_of_different_orders_really_are_different_fits(lattice):
    y, X, W = lattice
    a = sar(y, X, W, method="gmm", gmm_lags=1)
    b = sar(y, X, W, method="gmm", gmm_lags=3)
    assert abs(a.rho - b.rho) > 1e-3


def test_sem_gmm_iteration_converges_and_is_reported(lattice):
    """``n_iter`` iterates the KP moment step to a genuine fixed point.

    Every assertion here can fail: ``n_iter`` used to be checked with
    ``>= 1`` (true by construction) and the one- vs many-step gap with a
    half-unit tolerance on a parameter confined to ``(-1, 1)``.
    """
    y, X, W = lattice
    one = sem(y, X, W, method="gmm", n_iter=1)
    many = sem(y, X, W, method="gmm", n_iter=25)
    more = sem(y, X, W, method="gmm", n_iter=60)

    # the two-step estimator stops after exactly one pass and says so
    assert one.converged and one.n_iter == 1
    # the iterated one stops strictly later, and strictly before its budget
    assert 1 < many.n_iter < 25
    assert many.converged
    # it is a fixed point: a much larger budget changes neither lambda nor the
    # pass count (this is what "converged" has to mean)
    assert more.n_iter == many.n_iter
    assert more.lam == pytest.approx(many.lam, abs=1e-12)
    np.testing.assert_allclose(more.params.to_numpy(), many.params.to_numpy(), atol=1e-12)
    # and iterating genuinely moves lambda: 0.4119 -> 0.5159 on this DGP
    assert abs(many.lam - one.lam) > 0.05
    for res in (one, many):
        assert res.rho_bounds[0] < res.lam < res.rho_bounds[1]


def test_sem_gmm_truncated_before_convergence_warns_and_says_so(lattice):
    """A budget too small to converge must not be reported as converged."""
    y, X, W = lattice
    with pytest.warns(RuntimeWarning, match="had not converged"):
        short = sem(y, X, W, method="gmm", n_iter=3)
    assert short.converged is False
    assert short.n_iter == 3
    full = sem(y, X, W, method="gmm", n_iter=60)
    assert abs(short.lam - full.lam) > 1e-4          # it really had not landed


def test_gmm_rho_outside_the_admissible_interval_is_declared_not_hidden(lattice):
    """The plainest SDM-GMM call on this DGP returns rho = 1.109 — say so.

    Kelejian-Prucha 2SLS is unconstrained. Before the repair this fit reported
    three contradictory things at once and no warning: an interval it was not
    in, ``at_bound=False``, and a ``summary()`` line asserting that
    ``I - rho W`` is non-singular over that interval — while ``fitted`` and
    ``pseudo_r2`` were computed through an explosive multiplier and the
    documented next step, ``.effects()``, raised.
    """
    y, X, W = lattice
    with pytest.warns(RuntimeWarning, match="lies OUTSIDE the admissible interval"):
        res = sdm(y, X, W, method="gmm")
    assert res.rho > res.rho_bounds[1]                     # 1.1095 vs 1.0
    assert res.rho == pytest.approx(1.109466, abs=1e-5)
    assert res.rho_admissible is False
    # nothing is computed through (I - rho W)^-1 in that state
    assert np.isnan(res.pseudo_r2)
    assert np.all(np.isnan(res.fitted))
    assert np.all(np.isnan(res.resid_reduced))
    # but the estimates themselves are still reported and still the goldens
    assert np.all(np.isfinite(res.params.to_numpy()))
    np.testing.assert_allclose(res.params.to_numpy(), GOLDEN["gm_sdm"]["betas"], rtol=1e-9)
    # the summary does not claim an interval search it never did, and flags it
    text = res.summary()
    assert "no interval search" in text
    assert "OUTSIDE the admissible interval" in text
    assert "is non-singular on" not in text
    with pytest.raises(ValueError, match="lies outside the interval"):
        res.effects()


def test_an_admissible_gmm_rho_carries_no_warning_and_a_live_reduced_form(lattice):
    """The flag has two sides: the ordinary case must stay silent and finite."""
    y, X, W = lattice
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = sar(y, X, W, method="gmm")
    assert not [w for w in rec if "admissible interval" in str(w.message)]
    assert res.rho_bounds[0] < res.rho < res.rho_bounds[1]
    assert res.rho_admissible is True
    assert np.isfinite(res.pseudo_r2) and np.all(np.isfinite(res.fitted))
    assert "inside the admissible interval" in res.summary()
    eff = res.effects()                                    # the documented next step works
    assert np.all(np.isfinite(eff.total))


def test_ml_fits_are_always_admissible_by_construction(lattice):
    y, X, W = lattice
    for res in (sar(y, X, W), sdm(y, X, W), sem(y, X, W), slx(y, X, W)):
        assert res.rho_admissible is True


VCOV_TYPES = {
    ("sar", "ml", None): "analytic",
    ("sar", "ml", "hessian"): "hessian",
    ("sar", "gmm", None): "classical",
    ("sar", "gmm", "robust"): "robust",
    ("sem", "ml", None): "analytic",
    ("sem", "gmm", None): "classical",
}


def test_vcov_type_only_ever_takes_the_documented_values(lattice):
    """The reported enumeration must match what is actually written into it.

    ``slx`` is plain OLS and takes ``cov_type``, so it reports the OLS names;
    the field docstring now says so and names ``model == 'slx'`` as the exact
    condition. Downstream code branching on ``vcov_type`` needs the full set.
    """
    y, X, W = lattice
    documented = {"analytic", "hessian", "classical", "robust", "nonrobust", "hc1"}
    seen = set()
    for (model, method, vcov), expected in VCOV_TYPES.items():
        fn = {"sar": sar, "sem": sem}[model]
        res = fn(y, X, W, method=method, vcov=vcov)
        assert res.vcov_type == expected, (model, method, vcov)
        assert res.model != "slx"
        seen.add(res.vcov_type)
    for cov_type in ("nonrobust", "hc1"):
        res = slx(y, X, W, cov_type=cov_type)
        assert res.vcov_type == cov_type
        assert res.model == "slx"                # the documented iff condition
        seen.add(res.vcov_type)
    assert seen == documented
    doc = SpatialModelResult.__doc__
    for name in sorted(documented):
        assert f"``'{name}'``" in doc, name


def test_gmm_takes_an_explicit_interval_and_then_skips_the_eigendecomposition(lattice):
    """The escape a large-``n`` GMM caller is told to take must actually exist.

    ``_spectrum``'s warning used to advise ``logdet='lu'``, which raises under
    ``method='gmm'``; and ``rho_bounds`` was itself rejected there, so the
    dense O(n^3) eigendecomposition was unavoidable *and* unused. It is now
    the documented escape, and it works.
    """
    y, X, W = lattice
    auto = sar(y, X, W, method="gmm")
    explicit = sar(y, X, W, method="gmm", rho_bounds=(-1.0, 1.0))
    assert auto.spectrum is not None                  # needed for 'auto'
    assert explicit.spectrum is None                  # and for nothing else
    assert explicit.rho_bounds == (-1.0, 1.0)
    np.testing.assert_allclose(explicit.params.to_numpy(), auto.params.to_numpy(), atol=1e-12)
    assert explicit.rho_admissible is auto.rho_admissible
    # logdet remains meaningless there, so it is still rejected
    with pytest.raises(ValueError, match="`logdet` is meaningless with method='gmm'"):
        sar(y, X, W, method="gmm", logdet="lu")


def test_the_dense_spectrum_warning_names_an_escape_the_caller_can_take(lattice, monkeypatch):
    y, X, W = lattice
    from puremacro.spatial import models as models_mod

    monkeypatch.setattr(models_mod, "_DENSE_WARN_N", 10)

    def message(**kwargs) -> str:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            sar(y, X, W, **kwargs)
        hits = [str(w.message) for w in rec if "dense spectrum" in str(w.message)]
        return hits[0] if hits else ""

    ml_auto = message()
    assert "rho_bounds=(lo, hi)" in ml_auto and "logdet='lu'" in ml_auto
    gmm_auto = message(method="gmm")
    assert "rho_bounds=(lo, hi)" in gmm_auto
    # never advise logdet='lu' to a caller for whom it raises
    assert "logdet" not in gmm_auto
    # and taking either escape silences it
    assert message(rho_bounds=(-0.99, 0.99), logdet="lu") == ""
    assert message(method="gmm", rho_bounds=(-0.99, 0.99)) == ""


def test_partial_durbin_block_is_allowed_without_the_common_factor_claim(lattice):
    y, X, W = lattice
    res = sdm(y, X, W, durbin=["x1"], common_factor=False)
    assert res.durbin_names == ("x1",)
    assert list(res.params.index) == ["const", "x1", "x2", "W_x1", "rho"]
    assert res.common_factor is None
    eff = res.effects()
    # x2 carries no Durbin term, so its direct effect is the pure SAR one
    Ainv = np.linalg.inv(np.eye(N) - res.rho * W.to_dense())
    i2 = list(eff.variables).index("x2")
    b2 = float(res.beta[list(res.exog_names).index("x2")])
    assert eff.direct[i2] == pytest.approx(b2 * np.trace(Ainv) / N, rel=1e-11)


def test_module_all_is_exactly_the_public_surface():
    import puremacro.spatial.models as mod
    assert set(mod.__all__) == {
        "sar", "sem", "sdm", "slx", "ols_spatial", "lm_spatial_tests", "spatial_effects",
        "SpatialModelResult", "SpatialOLSResult", "SpatialLMResult", "SpatialEffectsResult",
    }
    for name in mod.__all__:
        assert hasattr(mod, name)


def test_importing_the_module_does_not_pull_in_matplotlib():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, puremacro.spatial.models as m; "
         "print('matplotlib' in sys.modules)"],
        cwd=root, capture_output=True, text=True, timeout=300,
        env={"PYTHONPATH": str(root), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", proc.stdout
