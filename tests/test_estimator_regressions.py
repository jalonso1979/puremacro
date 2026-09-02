"""Recovery tests for four estimators that shipped returning wrong numbers.

Each of these was green on the existing suite for the same reason: the fixtures
were built from the *simple* case, and every one of these bugs is invisible
there. A DCC fixture with mean-zero returns, a state-space fixture with no
intercept, a proxy fixture with i.i.d. shocks, a bootstrap that never fails —
each removes the exact condition the bug needs.

So these tests are all of one kind: simulate from a DGP that satisfies the
estimator's own assumptions, where the truth is known by construction, and
check that the estimator converges to it. Every one of them fails against the
pre-fix tree.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.garch.dcc import dcc_fit
from puremacro.inference.wild_bootstrap import wild_bootstrap_var
from puremacro.state_space import (StateSpaceModel, kalman_smoother,
                                   simulation_smoother)
from puremacro.var.identify.hetero import rigobon_svar


# --------------------------------------------------------------------------- #
# DCC: mean="constant" standardised the raw returns by a demeaned volatility
# --------------------------------------------------------------------------- #

def _dcc_panel(T, rho, mu, seed):
    """Exactly the DCC model: GARCH(1,1) margins, constant correlation rho."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    z = rng.standard_normal((T, 2)) @ L.T
    eps = np.zeros((T, 2))
    h = np.full((T, 2), 0.05 / (1 - 0.10 - 0.85))
    for t in range(1, T):
        h[t] = 0.05 + 0.10 * eps[t - 1] ** 2 + 0.85 * h[t - 1]
        eps[t] = np.sqrt(h[t]) * z[t]
    eps /= eps.std(axis=0)
    return pd.DataFrame(mu + eps, columns=["a", "b"])


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_dcc_constant_mean_recovers_the_true_correlation(rho):
    """Two series with a large mean are not correlated because of the mean.

    Step 2 of Engle's two-step needs z_t = eps_t / h_t^{1/2} with the same
    eps_t top and bottom. `garch11_fit(mean="constant")` demeans internally and
    never returns mu, so the raw series was divided by a volatility fitted on
    the demeaned one and the mean stayed inside z. Qbar then measured
    mu_i * mu_j: for independent series the probability limit is m^2/(m^2+1)
    with m = mu/sd, i.e. 0.96 at m = 5.
    """
    got = dcc_fit(_dcc_panel(8000, rho, mu=5.0, seed=1), mean="constant")
    assert got.R[:, 0, 1].mean() == pytest.approx(rho, abs=0.05)


def test_dcc_constant_mean_is_invariant_to_the_mean():
    """The whole point of `mean="constant"` is that the level must not matter."""
    hi = dcc_fit(_dcc_panel(8000, 0.5, mu=5.0, seed=1), mean="constant")
    lo = dcc_fit(_dcc_panel(8000, 0.5, mu=0.0, seed=1), mean="constant")
    assert hi.R[:, 0, 1].mean() == pytest.approx(lo.R[:, 0, 1].mean(), abs=1e-9)


def test_dcc_zero_mean_path_is_unchanged():
    """`mean="zero"` documents mean-removal as the caller's job. Keep it so."""
    panel = _dcc_panel(4000, 0.5, mu=0.0, seed=2)
    z = dcc_fit(panel, mean="zero")
    c = dcc_fit(panel, mean="constant")
    # on already-demeaned input the two agree to numerical noise
    assert z.R[:, 0, 1].mean() == pytest.approx(c.R[:, 0, 1].mean(), abs=1e-3)


def test_dcc_rejects_an_unknown_mean_spec():
    with pytest.raises(ValueError, match="mean must be"):
        dcc_fit(_dcc_panel(200, 0.0, 0.0, 0), mean="AR1")


# --------------------------------------------------------------------------- #
# simulation_smoother: the intercepts were left in the y* smoothing pass
# --------------------------------------------------------------------------- #

def _local_level(T, d, c, seed):
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    s = 0.0
    for t in range(T):
        y[t] = s + d + rng.standard_normal()
        s = s + c + rng.standard_normal()
    model = StateSpaceModel(
        T=np.array([[1.0]]), Z=np.array([[1.0]]), Q=np.array([[1.0]]),
        H=np.array([[1.0]]), R=np.array([[1.0]]),
        c=np.array([c]), d=np.array([d]),
    )
    return y[:, None], model


@pytest.mark.parametrize("c, d", [(0.0, 5.0), (0.3, 0.0), (0.4, -3.0)])
def test_simulation_smoother_draws_are_centred_on_the_posterior_mean(c, d):
    """Durbin-Koopman's second pass must run the HOMOGENEOUS model.

    The smoother is affine in (y, a0, c, d) — a_hat(y) = A y + b — and the
    algorithm needs only its linear part, because b cancels between a_hat(y)
    and a_hat(y+). Passing the original model added b back, shifting every
    draw by the whole intercept: with d = 5 the draws sat exactly -5.0 from
    the posterior mean at every t, against a Monte Carlo se of 0.012.
    """
    y, model = _local_level(40, d=d, c=c, seed=0)
    truth = kalman_smoother(y, model, a0=np.zeros(1),
                            P0=1e6 * np.eye(1))["a_smooth"][:, 0]
    draws = np.array([
        simulation_smoother(y, model, rng=np.random.default_rng(k))[:, 0]
        for k in range(3000)
    ])
    bias = draws.mean(axis=0) - truth
    se = draws.std(axis=0).max() / np.sqrt(len(draws))
    assert np.abs(bias).max() < 6.0 * se, (np.abs(bias).max(), se)


def test_simulation_smoother_draws_have_the_posterior_spread():
    """A fix that merely returned the smoothed mean would pass the test above."""
    y, model = _local_level(40, d=5.0, c=0.0, seed=0)
    out = kalman_smoother(y, model, a0=np.zeros(1), P0=1e6 * np.eye(1))
    draws = np.array([
        simulation_smoother(y, model, rng=np.random.default_rng(k))[:, 0]
        for k in range(3000)
    ])
    ratio = draws.var(axis=0)[5:-5] / out["P_smooth"][5:-5, 0, 0]
    assert 0.85 < float(np.median(ratio)) < 1.15, float(np.median(ratio))


# --------------------------------------------------------------------------- #
# wild_bootstrap_var: failed draws were replaced by the point estimate
# --------------------------------------------------------------------------- #

def _var_data(T=300, seed=0):
    rng = np.random.default_rng(seed)
    A = np.array([[0.6, 0.1], [0.05, 0.5]])
    B = np.array([[1.0, 0.0], [0.4, 0.8]])
    Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + B @ rng.standard_normal(2)
    return Y


def _flaky_impact(fail_rate, seed):
    """Cholesky, failing at random independently of the draw's value."""
    state = {"n": 0}
    rng = np.random.default_rng(seed)

    def impact(A_b, Sigma_b, resid_b):
        state["n"] += 1
        if state["n"] > 1 and rng.random() < fail_rate:
            raise np.linalg.LinAlgError("injected identification failure")
        return np.linalg.cholesky(Sigma_b)
    return impact


def test_failed_draws_do_not_shrink_the_band():
    """Substituting the point estimate puts a point mass at theta_hat.

    With a failure fraction f, the reported 100(1-2a)% band was really the
    100(1 - 2a/(1-f))% band — contracting monotonically in f, and collapsing
    to zero width once f >= 1-2a. Failures here are injected at random and
    independent of the draw value, so dropping them is exactly unbiased and
    any contraction is attributable to the substitution alone.
    """
    Y = _var_data()
    widths = {}
    for f in (0.0, 0.4, 0.7):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, lo, hi = wild_bootstrap_var(
                Y, p=1, horizon=8, impact_fn=_flaky_impact(f, 1),
                n_boot=400, ci=0.90, seed=3)
        widths[f] = float((hi - lo).mean())
    assert widths[0.7] > 0.8 * widths[0.0], widths


def test_a_high_failure_rate_is_reported():
    Y = _var_data()
    with pytest.warns(UserWarning, match="failed identification and were dropped"):
        wild_bootstrap_var(Y, p=1, horizon=4, impact_fn=_flaky_impact(0.5, 1),
                           n_boot=100, ci=0.90, seed=3)


def test_a_low_failure_rate_is_not_reported():
    """Positive control for the threshold: it must not fire on a clean run."""
    Y = _var_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wild_bootstrap_var(Y, p=1, horizon=4, impact_fn=_flaky_impact(0.0, 1),
                           n_boot=50, ci=0.90, seed=3)


def test_a_wholly_failed_bootstrap_raises_instead_of_returning_a_zero_band():
    """It used to return lo == hi == point, a zero-width 90% band, silently."""
    Y = _var_data()
    with pytest.raises(np.linalg.LinAlgError, match="all 50 bootstrap draws"):
        wild_bootstrap_var(Y, p=1, horizon=4, impact_fn=_flaky_impact(1.0, 1),
                           n_boot=50, ci=0.90, seed=3)


# --------------------------------------------------------------------------- #
# rigobon_svar: bootstrap blocks were reshuffled but regime labels were not
# --------------------------------------------------------------------------- #

def _rigobon_data(T=2000, seed=0):
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.10], [0.05, 0.40]])
    B = np.array([[1.0, 0.40], [0.5, 0.80]])
    lam = np.array([2.0, 6.0])
    ri = np.zeros(T, dtype=int)
    for k in range(0, T, 200):
        ri[k + 100:k + 200] = 1
    Y = np.zeros((T, 2))
    for t in range(1, T):
        sd = np.sqrt(np.where(ri[t] == 1, lam, 1.0))
        Y[t] = A @ Y[t - 1] + B @ (sd * rng.standard_normal(2))
    return Y, ri, B, lam


def test_rigobon_recovers_the_variance_ratios_and_the_impact_matrix():
    Y, ri, B_true, lam = _rigobon_data()
    res = rigobon_svar(Y, p=1, horizon=8, regime_indicator=ri, n_boot=0)
    assert np.allclose(res.variance_ratios, lam, rtol=0.15)
    assert np.allclose(np.abs(res.B), np.abs(B_true), atol=0.1)


def test_the_rigobon_bootstrap_keeps_each_regime_label_with_its_own_residual():
    """The identifying content is the contrast Sigma_1 - Sigma_0.

    The bootstrap reshuffles residual blocks; pairing them with the labels of
    the calendar dates they land on makes each "regime" a random subset of the
    same mixed distribution, so both bootstrap covariances converge to the same
    matrix and the generalised eigenproblem goes near-degenerate. The draws are
    then arbitrary rotations, and the band they produce is roughly eight times
    too wide on this DGP — a band, but not a confidence band.
    """
    Y, ri, _, _ = _rigobon_data()
    res = rigobon_svar(Y, p=1, horizon=8, regime_indicator=ri,
                       n_boot=200, ci=0.90, seed=1, block_len=13)
    assert res.lower is not None and np.isfinite(res.lower).all()
    # The point estimate must lie inside its own band...
    assert bool(((res.lower <= res.irfs + 1e-9)
                 & (res.irfs <= res.upper + 1e-9)).all())
    # ...and the band must be a sampling distribution, not a rotation cloud.
    # Impact responses here are O(1); rotations gave a mean width near 0.40.
    assert float((res.upper - res.lower).mean()) < 0.20
