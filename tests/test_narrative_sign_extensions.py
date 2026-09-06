"""Tests for narrative SVAR extensions (shock bounds and Bayesian NIW sampling).

The second half of the file holds regression tests for the v2.3.0 audit
(review-r1-narrative M101 / M103 / M105); each docstring describes the
behaviour of the old code that the test would have caught.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.var.identify.narrative_sign import NarrativeRestriction, narrative_sign_svar

SIGN_2 = {0: np.array([[1, 0], [1, 1]])}


@pytest.fixture
def synthetic_var_data():
    rng = np.random.default_rng(123)
    T = 100
    e = rng.standard_normal((T, 2))
    Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t, 0] = 0.6 * Y[t-1, 0] + 0.1 * Y[t-1, 1] + e[t, 0]
        Y[t, 1] = 0.2 * Y[t-1, 0] + 0.5 * Y[t-1, 1] + 0.4 * e[t, 0] + e[t, 1]
    return Y


def test_narrative_restriction_shock_bound(synthetic_var_data):
    Y = synthetic_var_data
    restr = [
        NarrativeRestriction(
            kind="shock_bound",
            date=40,
            shock=0,
            min_magnitude=0.3,
            sign=+1,
        )
    ]
    res = narrative_sign_svar(
        Y,
        p=1,
        horizon=4,
        sign_matrix=SIGN_2,
        restrictions=restr,
        n_draws=300,
        seed=42,
    )
    assert res.n_narrative_accepted > 0
    assert res.irf_median.shape == (5, 2, 2)
    assert res.irf_lower.shape == (5, 2, 2)
    assert res.irf_upper.shape == (5, 2, 2)


def test_narrative_svar_bayes_draws(synthetic_var_data):
    Y = synthetic_var_data
    restr = [
        NarrativeRestriction(
            kind="shock_sign",
            date=21,
            shock=0,
            sign=+1,
        )
    ]
    res = narrative_sign_svar(
        Y,
        p=1,
        horizon=4,
        sign_matrix=SIGN_2,
        restrictions=restr,
        bayes_draws=True,
        n_draws=300,
        seed=42,
    )
    assert res.n_narrative_accepted > 0
    assert res.irf_median.shape == (5, 2, 2)
    assert not np.isnan(res.irf_median).any()


# ---------------------------------------------------------------------------
# Regression tests (v2.3.0 audit)
# ---------------------------------------------------------------------------

def _run2(Y, restr, **kw):
    kw.setdefault("n_draws", 400)
    kw.setdefault("n_weight_sims", 200)
    kw.setdefault("seed", 0)
    return narrative_sign_svar(Y, p=1, horizon=4, sign_matrix=SIGN_2,
                               restrictions=restr, **kw)


def test_shock_bound_default_is_unsigned(synthetic_var_data):
    """M103: the dataclass default sign=+1 leaked into shock_bound, so a
    magnitude-only bound on a date with a negative shock rejected every draw
    (RuntimeError) unless sign=None was passed explicitly; its label even
    read 'sign=+1'. The default is now an unsigned bound: the accepted set
    is the disjoint union of the sign=+1 and sign=-1 sets."""
    Y = synthetic_var_data
    unsigned = NarrativeRestriction(kind="shock_bound", date=40, shock=0, min_magnitude=0.5)
    assert unsigned.sign is None
    assert "sign=" not in unsigned.label()
    zero = NarrativeRestriction(kind="shock_bound", date=40, shock=0, min_magnitude=0.5, sign=0)
    assert zero.sign is None  # 0 is the documented spelling of 'unsigned'

    r_u = _run2(Y, [unsigned])
    r_pos = _run2(Y, [NarrativeRestriction(kind="shock_bound", date=40, shock=0,
                                           min_magnitude=0.5, sign=+1)])
    r_neg = _run2(Y, [NarrativeRestriction(kind="shock_bound", date=40, shock=0,
                                           min_magnitude=0.5, sign=-1)])
    assert r_u.n_traditional_accepted == r_pos.n_traditional_accepted == r_neg.n_traditional_accepted
    assert r_u.n_narrative_accepted == r_pos.n_narrative_accepted + r_neg.n_narrative_accepted
    assert r_u.n_narrative_accepted > max(r_pos.n_narrative_accepted, r_neg.n_narrative_accepted)


def test_bayes_mode_returns_coherent_reduced_form(synthetic_var_data):
    """M105: bayes_draws=True returned B from a posterior (A, Sigma) draw but
    A_list / residuals / intercept from OLS, so B B' != Sigma_OLS (max diff
    0.14) and the historical decomposition mixed incompatible objects. The
    result now carries the median-target draw's own (A, c, Sigma, residuals)
    and the per-draw A's used to extend the IRFs."""
    from puremacro.var.estimate import estimate_var

    Y = synthetic_var_data
    est = estimate_var(Y, 1)
    res = _run2(Y, [NarrativeRestriction(kind="shock_sign", date=21, shock=0, sign=+1)],
                bayes_draws=True, n_draws=300, seed=42)
    assert res.bayes_draws
    assert "Normal-Inverse-Wishart" in res.summary()
    np.testing.assert_allclose(res.B @ res.B.T, res.Sigma, atol=1e-10)
    assert not np.allclose(res.A_list[0], est.A_list[0])
    assert res.accepted_A.shape == (res.n_narrative_accepted, 1, 2, 2)
    assert res.accepted_B.shape == (res.n_narrative_accepted, 2, 2)
    # the representative B is an accepted draw and A_list is that draw's own A
    hits = [k for k in range(res.n_narrative_accepted) if np.allclose(res.accepted_B[k], res.B)]
    assert len(hits) == 1
    np.testing.assert_allclose(res.accepted_A[hits[0]][0], res.A_list[0])
    # residuals are the draw's own: the HD identity holds exactly
    hd = res.historical_decomposition()
    np.testing.assert_allclose(hd["deterministic"] + hd["shocks"].sum(axis=2), Y[1:], atol=1e-9)
    # extending the IRF uses each draw's own A and stays a weighted median
    ext = res.irf(8)
    assert ext.shape == (9, 2, 2)
    np.testing.assert_allclose(ext[:5], res.irf_median, atol=1e-12)
    fe = res.fevd(8)
    np.testing.assert_allclose(fe[:5], res.fevd_median, atol=1e-12)
    np.testing.assert_allclose(fe.sum(axis=2), 1.0, atol=1e-9)


def test_bayes_mode_unstable_posterior_is_not_silently_used(synthetic_var_data, monkeypatch):
    """M101: after 50 failed stability attempts the last (unstable) posterior
    draw was used without any signal — on an explosive DGP (rho = 1.15) the
    median IRF reached 1e46. Now: all draws unstable -> RuntimeError; some
    unstable -> skipped, counted in n_unstable_draws, RuntimeWarning."""
    import puremacro.var.identify.narrative_sign as ns

    rng = np.random.default_rng(3)
    Ye = np.zeros((150, 2))
    for t in range(1, 150):
        Ye[t] = 1.15 * Ye[t - 1] + rng.standard_normal(2)
    with pytest.raises(RuntimeError, match="stable"):
        narrative_sign_svar(Ye, p=1, horizon=4, restrictions=[], bayes_draws=True,
                            n_draws=10, seed=0)

    # Make exactly the first posterior draw fail all 50 attempts.
    calls = {"n": 0}
    orig = ns._is_stable_var

    def flaky(A_list):
        calls["n"] += 1
        return False if calls["n"] <= ns._MAX_STABLE_ATTEMPTS else orig(A_list)

    monkeypatch.setattr(ns, "_is_stable_var", flaky)
    with pytest.warns(RuntimeWarning, match="unstable"):
        res = _run2(synthetic_var_data, [], bayes_draws=True, n_draws=40, seed=1)
    assert res.n_unstable_draws == 1
    assert res.n_traditional_accepted <= 39
    assert "1 unstable draws skipped" in res.summary()
    assert np.isfinite(res.irf_median).all()
