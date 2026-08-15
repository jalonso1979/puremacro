"""Tests for narrative SVAR extensions (shock bounds and Bayesian NIW sampling)."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.var.identify.narrative_sign import NarrativeRestriction, narrative_sign_svar


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
        sign_matrix={0: np.array([[1, 0], [1, 1]])},
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
        sign_matrix={0: np.array([[1, 0], [1, 1]])},
        restrictions=restr,
        bayes_draws=True,
        n_draws=300,
        seed=42,
    )
    assert res.n_narrative_accepted > 0
    assert res.irf_median.shape == (5, 2, 2)
    assert not np.isnan(res.irf_median).any()
