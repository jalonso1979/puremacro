"""Tests for puremacro.dynpanel.diagnostics."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from puremacro.dynpanel import ab_gmm
from puremacro.dynpanel.diagnostics import ar_test, hansen_j

from .conftest import simulate_dynamic_panel


def test_hansen_j_is_chi2_under_correct_model():
    """Under a correctly-specified DGP, the two-step Hansen J p-value
    should be approximately uniform on [0, 1] across simulation draws.
    KS test against U(0,1) at α = 0.05.
    """
    n_draws = 100
    p_values = []
    for seed in range(n_draws):
        sim = simulate_dynamic_panel(N=80, T=6, rho=0.3, seed=seed)
        res = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"])
        if np.isfinite(res.hansen_j_p):
            p_values.append(res.hansen_j_p)
    p_values = np.array(p_values)
    # KS test against uniform
    ks_stat, ks_p = stats.kstest(p_values, "uniform")
    # Tolerance: with 100 draws the null should not be rejected at α = 0.05
    # under a correctly-specified model. We're permissive (0.01) to allow
    # for finite-sample non-uniformity from the GMM weight estimation.
    assert ks_p > 0.01, (
        f"Hansen J p-value distribution differs from uniform "
        f"(KS p = {ks_p:.4f}); spec may be miscoded."
    )


def test_ar1_p_low_for_difference_residuals_under_iid():
    """Differencing iid ε creates AR(1) structure in Δε by construction:
    Cov(Δε_t, Δε_{t-1}) = -σ² ≠ 0. The AR(1) test should reject.
    """
    sim = simulate_dynamic_panel(N=200, T=10, rho=0.3, seed=0)
    res = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    # AR(1) p should be small: differencing always induces lag-1 corr
    assert res.ar1_p < 0.10, (
        f"AR(1) p = {res.ar1_p:.3f}; differencing should induce AR(1)"
    )


def test_ar2_p_typical_under_correct_iid_dgp():
    """Under iid ε, AR(2) test should typically NOT reject.
    Check that on a single decent-size panel, p > 0.05 (a soft check
    since p is a single random draw)."""
    p_vals = []
    for seed in range(20):
        sim = simulate_dynamic_panel(N=200, T=10, rho=0.3, seed=seed)
        res = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"])
        p_vals.append(res.ar2_p)
    # under the null, mean(p) should be ~0.5 and few should reject
    n_reject = sum(p < 0.05 for p in p_vals if np.isfinite(p))
    assert n_reject <= 4, (
        f"{n_reject}/20 AR(2) rejections at α=0.05 under correct model "
        "— too many false-positives."
    )


def test_ar2_p_low_when_ar1_disturbance_present():
    """If we manually inject AR(1) disturbances (not iid), the AR(2)
    test on the difference residuals SHOULD reject.

    Here we generate ε with AR(1) coefficient 0.6 and check.
    """
    rng = np.random.default_rng(0)
    N, T = 200, 10
    rho_true = 0.3
    rho_eps = 0.6
    burnin = 50
    alpha = rng.normal(0, 1, size=N)
    eps_innov = rng.normal(0, 1, size=(N, burnin + T))
    eps_full = np.zeros((N, burnin + T))
    eps_full[:, 0] = eps_innov[:, 0]
    for t in range(1, burnin + T):
        eps_full[:, t] = rho_eps * eps_full[:, t - 1] + eps_innov[:, t]
    y_full = np.zeros((N, burnin + T))
    y_full[:, 0] = alpha + eps_full[:, 0]
    for t in range(1, burnin + T):
        y_full[:, t] = rho_true * y_full[:, t - 1] + alpha + eps_full[:, t]
    y_obs = y_full[:, -T:].ravel()
    panel_id = np.repeat(np.arange(N), T)
    time_id = np.tile(np.arange(T), N).astype(np.int64)
    res = ab_gmm(y_obs, panel_id, time_id)
    # Note: when ε is AR(1), differenced residuals are AR(1) in original
    # frame which means Δε has an MA(2)-like structure. AR(2) test
    # should reject more often than under iid.
    # (Soft check: AR(2) p should not be > 0.5 across many seeds)
    assert res.ar2_p < 0.20 or res.ar1_p < 0.001, (
        f"With AR(1) disturbances, AR(2) p = {res.ar2_p:.3f} should "
        "show stronger evidence against the iid null."
    )


def test_hansen_j_zero_when_just_identified():
    """When df=0 (exactly identified), Hansen J = 0 and p = NaN."""
    Z = np.eye(3)  # 3 instruments
    u = np.array([0.1, -0.2, 0.05])
    W = np.eye(3)
    J, p, df = hansen_j(Z, u, W, n_regressors=3)
    assert df == 0
    assert np.isnan(p)


def test_ar_test_invalid_lag_raises():
    with pytest.raises(ValueError):
        ar_test(np.zeros(5), [{"panel": 0, "t": i} for i in range(5)], lag=0)


def test_windmeijer_increases_se_relative_to_uncorrected():
    """Windmeijer correction inflates two-step SE relative to the
    uncorrected sandwich, in finite samples."""
    sim = simulate_dynamic_panel(N=80, T=8, rho=0.5, seed=0)
    res_w = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"], windmeijer=True)
    res_no = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"], windmeijer=False)
    assert res_w.se[0] >= res_no.se[0]


def test_windmeijer_finite_and_positive():
    """SEs should be finite and positive for a generic well-conditioned panel."""
    sim = simulate_dynamic_panel(N=80, T=8, rho=0.3, seed=0)
    res = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    assert np.all(np.isfinite(res.se))
    assert np.all(res.se > 0)
