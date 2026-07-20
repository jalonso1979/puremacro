"""Tests for HAR-RV and range-based estimators."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.volatility import (
    har_rv, parkinson, garman_klass, rogers_satchell,
)


def test_har_rv_recovers_persistence_signal(rng):
    # Generate an AR(1)-like RV path; HAR's daily coefficient should
    # pick up the persistence.
    T = 500
    rho = 0.7
    rv = np.empty(T)
    rv[0] = 1.0
    for t in range(1, T):
        rv[t] = max(0.05, 0.2 + rho * rv[t - 1] + 0.1 * rng.standard_normal())
    out = har_rv(rv, horizons=(1, 5, 22))
    # Daily coefficient (β_d) sits at out["beta"][1] (intercept first).
    assert out["beta"][1] > 0.2  # positive persistence loading
    assert "horizons" in out


def test_har_rv_documented_keys(rng):
    rv = np.abs(rng.standard_normal(100)) + 0.1
    out = har_rv(rv, horizons=(1, 5, 22))
    for k in ("beta", "se", "t", "vcov", "residuals", "n_obs", "horizons"):
        assert k in out


def test_har_rv_rejects_too_short_input(rng):
    rv = np.abs(rng.standard_normal(20)) + 0.1
    with pytest.raises(ValueError):
        har_rv(rv, horizons=(1, 5, 22))


def test_parkinson_zero_when_high_equals_low():
    H = np.array([100.0, 105.0, 102.0])
    L = H.copy()
    np.testing.assert_allclose(parkinson(H, L), 0.0, atol=1e-15)


def test_garman_klass_close_to_parkinson_on_typical_input(rng):
    """When close ≈ open and the bar is symmetric, GK should land
    close to Parkinson."""
    n = 100
    O = 100 * np.ones(n)
    spread = np.abs(rng.standard_normal(n)) * 0.5 + 0.1
    H = O + spread
    L = O - spread
    C = O + 0.001 * rng.standard_normal(n)  # close ≈ open
    p = parkinson(H, L)
    gk = garman_klass(O, H, L, C)
    # Both should be positive and on the same order of magnitude.
    assert (p > 0).all() and (gk > 0).all()
    # Ratio bounded.
    ratio = gk.mean() / p.mean()
    assert 0.3 < ratio < 1.5


def test_rogers_satchell_drift_invariance():
    """RS should be invariant to a deterministic drift in log-prices.
    Construct a no-volatility path with positive drift and confirm RS
    returns ≈ 0."""
    n = 50
    drift = 0.001
    O = 100.0 * np.exp(drift * np.arange(n))
    # No intra-day volatility ⇒ H = C, L = O, so RS = 0.
    H = O * np.exp(drift)
    L = O
    C = O * np.exp(drift)
    rs = rogers_satchell(O, H, L, C)
    np.testing.assert_allclose(rs, 0.0, atol=1e-12)


def test_range_estimators_handle_2d_input(rng):
    n_assets, T = 3, 50
    O = np.full((T, n_assets), 100.0)
    spread = np.abs(rng.standard_normal((T, n_assets))) + 0.05
    H = O + spread
    L = O - spread
    C = O + 0.01 * rng.standard_normal((T, n_assets))
    assert parkinson(H, L).shape == (T, n_assets)
    assert garman_klass(O, H, L, C).shape == (T, n_assets)
    assert rogers_satchell(O, H, L, C).shape == (T, n_assets)
