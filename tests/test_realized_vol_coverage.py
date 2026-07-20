"""Meaningful tests for puremacro.realized_vol.

Covers:
  - realized_variance : known-value, scaling, sign-invariance, list input, empty
  - bipower_variation : known-value, scaling, sign-invariance, edge cases (0/1/2 elements)
  - realized_jump     : contract corners (rv<=0, bv>rv clip, normal case)
  - daily_rv_panel    : shape, columns, RJ bounds, BV=0 for 1-element days
  - har_rv            : key set, design shape, reconstruction, SE positivity, R2 float,
                        persistence signal (high-rho => large beta_d), auto hac_lags
                        formula, hac_lags override, log_transform flag, ValueError on short input
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.realized_vol import (
    bipower_variation,
    daily_rv_panel,
    har_rv,
    realized_jump,
    realized_variance,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ===========================================================================
# realized_variance
# ===========================================================================

def test_realized_variance_known_value():
    """RV = sum(r_i^2): verify with hand-calculated answer."""
    r = np.array([0.01, -0.02, 0.015])
    expected = 0.01**2 + 0.02**2 + 0.015**2  # 0.000725
    result = realized_variance(r)
    assert abs(result - expected) < 1e-15


def test_realized_variance_zero_returns():
    """All-zero returns give RV = 0."""
    assert realized_variance(np.zeros(10)) == 0.0


def test_realized_variance_scaling():
    """Scaling returns by k multiplies RV by k^2."""
    r = np.array([0.01, -0.02, 0.015, 0.005, -0.01])
    k = 3.0
    assert abs(realized_variance(k * r) / realized_variance(r) - k**2) < 1e-12


def test_realized_variance_sign_invariance():
    """RV(-r) == RV(r) because we square."""
    r = np.array([0.01, -0.02, 0.015])
    assert realized_variance(-r) == realized_variance(r)


def test_realized_variance_single_element():
    """Single intraday return: RV = r^2."""
    r = np.array([0.05])
    assert abs(realized_variance(r) - 0.05**2) < 1e-15


def test_realized_variance_empty():
    """Empty return array gives RV = 0.0."""
    assert realized_variance(np.array([])) == 0.0


def test_realized_variance_accepts_list():
    """Accepts Python list (not just ndarray)."""
    r_list = [0.01, -0.02, 0.015]
    r_arr = np.array(r_list)
    assert realized_variance(r_list) == realized_variance(r_arr)


def test_realized_variance_returns_float():
    """Return type is a Python float (not ndarray)."""
    result = realized_variance(np.array([0.01, 0.02]))
    assert isinstance(result, float)


# ===========================================================================
# bipower_variation
# ===========================================================================

def test_bipower_variation_known_value_two_elements():
    """BV for two elements: (pi/2) * |r0| * |r1|."""
    r = np.array([0.03, -0.04])
    expected = (np.pi / 2.0) * 0.03 * 0.04
    assert abs(bipower_variation(r) - expected) < 1e-14


def test_bipower_variation_known_value_three_elements():
    """BV for [a, b, c] = (pi/2) * (|a|*|b| + |b|*|c|)."""
    a, b, c = 0.01, -0.02, 0.015
    expected = (np.pi / 2.0) * (abs(a) * abs(b) + abs(b) * abs(c))
    result = bipower_variation(np.array([a, b, c]))
    assert abs(result - expected) < 1e-14


def test_bipower_variation_single_element_returns_zero():
    """Less than 2 returns: BV = 0 (no adjacent pair)."""
    assert bipower_variation(np.array([0.05])) == 0.0


def test_bipower_variation_empty_returns_zero():
    """Empty array: BV = 0."""
    assert bipower_variation(np.array([])) == 0.0


def test_bipower_variation_scaling():
    """Scaling by k multiplies BV by k^2."""
    r = np.array([0.01, -0.02, 0.015, 0.005, -0.01])
    k = 2.5
    assert abs(bipower_variation(k * r) / bipower_variation(r) - k**2) < 1e-12


def test_bipower_variation_sign_invariance():
    """BV(-r) == BV(r) because we take absolute values."""
    r = np.array([0.01, -0.02, 0.015, 0.005])
    assert abs(bipower_variation(-r) - bipower_variation(r)) < 1e-15


def test_bipower_variation_constant_returns():
    """Constant returns r = c: BV = (pi/2) * (n-1) * c^2."""
    c = 0.01
    n = 5
    r = np.full(n, c)
    expected = (np.pi / 2.0) * (n - 1) * c**2
    assert abs(bipower_variation(r) - expected) < 1e-14


def test_bipower_variation_nonnegative(rng):
    """BV is always >= 0 for random returns."""
    r = rng.standard_normal(100) * 0.01
    assert bipower_variation(r) >= 0.0


def test_bipower_variation_returns_float():
    """Return type is a Python float."""
    result = bipower_variation(np.array([0.01, 0.02, 0.03]))
    assert isinstance(result, float)


def test_bipower_variation_accepts_list():
    """Accepts Python list (not just ndarray)."""
    r_list = [0.01, -0.02, 0.015]
    r_arr = np.array(r_list)
    assert bipower_variation(r_list) == bipower_variation(r_arr)


# ===========================================================================
# realized_jump
# ===========================================================================

def test_realized_jump_zero_rv_returns_zero():
    """RV == 0 => RJ = 0 (no division by zero)."""
    assert realized_jump(0.0, 0.0) == 0.0


def test_realized_jump_negative_rv_returns_zero():
    """RV < 0 is an invalid state; contract says return 0."""
    assert realized_jump(-1.0, 0.5) == 0.0


def test_realized_jump_bv_exceeds_rv_clipped_to_zero():
    """When BV > RV (noise), jump component is clipped to 0."""
    rv, bv = 0.0005, 0.001  # bv > rv
    assert realized_jump(rv, bv) == 0.0


def test_realized_jump_known_value():
    """RJ = (RV - BV) / RV when RV > BV."""
    rv, bv = 1.0, 0.6
    expected = (rv - bv) / rv  # 0.4
    assert abs(realized_jump(rv, bv) - expected) < 1e-15


def test_realized_jump_bounds_on_random_data(rng):
    """RJ in [0, 1] for any valid rv > bv input."""
    rvs = np.abs(rng.standard_normal(50)) * 0.001 + 0.0001
    bvs = rvs * rng.uniform(0.0, 1.2, size=50)  # sometimes bv > rv
    for rv, bv in zip(rvs, bvs):
        rj = realized_jump(rv, bv)
        assert 0.0 <= rj <= 1.0


def test_realized_jump_bv_equal_rv_is_zero():
    """When BV == RV exactly, RJ = 0."""
    rv = 0.001
    assert realized_jump(rv, rv) == 0.0


# ===========================================================================
# daily_rv_panel
# ===========================================================================

def test_daily_rv_panel_shape_and_columns(rng):
    """Panel has one row per day and exactly columns RV, BV, RJ."""
    n_days = 10
    days = [rng.standard_normal(78) * 0.01 for _ in range(n_days)]
    panel = daily_rv_panel(days)
    assert isinstance(panel, pd.DataFrame)
    assert panel.shape[0] == n_days
    assert set(panel.columns) == {"RV", "BV", "RJ"}


def test_daily_rv_panel_rv_nonnegative(rng):
    """RV >= 0 for every day."""
    days = [rng.standard_normal(50) * 0.01 for _ in range(20)]
    panel = daily_rv_panel(days)
    assert (panel["RV"] >= 0).all()


def test_daily_rv_panel_bv_nonnegative(rng):
    """BV >= 0 for every day."""
    days = [rng.standard_normal(50) * 0.01 for _ in range(20)]
    panel = daily_rv_panel(days)
    assert (panel["BV"] >= 0).all()


def test_daily_rv_panel_rj_in_unit_interval(rng):
    """RJ in [0, 1] for every day."""
    days = [rng.standard_normal(50) * 0.01 for _ in range(20)]
    panel = daily_rv_panel(days)
    assert (panel["RJ"] >= 0).all()
    assert (panel["RJ"] <= 1.0).all()


def test_daily_rv_panel_single_element_days():
    """Days with only one return have BV=0.
    RJ = max(0, (RV-BV)/RV) = 1 when RV>0 and BV=0.
    """
    days = [np.array([0.05]), np.array([-0.03])]
    panel = daily_rv_panel(days)
    assert panel.shape == (2, 3)
    # BV requires at least two adjacent returns => zero for single-element days
    assert (panel["BV"] == 0.0).all()
    # With BV=0 and RV>0, RJ = (RV-0)/RV = 1.0
    assert (panel["RJ"] == 1.0).all()


def test_daily_rv_panel_empty_list():
    """Empty day list returns empty DataFrame."""
    panel = daily_rv_panel([])
    assert isinstance(panel, pd.DataFrame)
    assert len(panel) == 0


def test_daily_rv_panel_known_rv_value():
    """Check RV in panel matches manual calculation for one day."""
    r = np.array([0.01, -0.02, 0.015])
    panel = daily_rv_panel([r])
    expected_rv = 0.01**2 + 0.02**2 + 0.015**2
    assert abs(panel["RV"].iloc[0] - expected_rv) < 1e-15


def test_daily_rv_panel_known_bv_value():
    """Check BV in panel matches bipower_variation for one day."""
    r = np.array([0.03, -0.04, 0.02])
    panel = daily_rv_panel([r])
    expected_bv = bipower_variation(r)
    assert abs(panel["BV"].iloc[0] - expected_bv) < 1e-15


# ===========================================================================
# har_rv
# ===========================================================================

def test_har_rv_required_keys_present(rng):
    """har_rv returns all documented keys."""
    rv = np.abs(rng.standard_normal(100)) * 0.01 + 0.0001
    out = har_rv(rv, log_transform=False)
    required_keys = {
        "intercept", "se_intercept",
        "beta_d", "se_d",
        "beta_w", "se_w",
        "beta_m", "se_m",
        "fitted", "residuals", "R2",
        "design", "log_transform", "hac_lags",
    }
    assert required_keys.issubset(out.keys())


def test_har_rv_raises_on_short_input(rng):
    """ValueError when fewer than ~30 daily observations."""
    with pytest.raises(ValueError, match="HAR-RV"):
        har_rv(np.abs(rng.standard_normal(25)) + 0.0001)


def test_har_rv_design_shape(rng):
    """Design matrix has shape (T-22, 4): intercept + d + w + m."""
    T = 100
    rv = np.abs(rng.standard_normal(T)) * 0.01 + 0.001
    out = har_rv(rv, log_transform=False)
    assert out["design"].shape == (T - 22, 4)


def test_har_rv_design_intercept_column(rng):
    """First column of design matrix is all-ones (intercept)."""
    rv = np.abs(rng.standard_normal(80)) * 0.01 + 0.001
    out = har_rv(rv, log_transform=False)
    assert np.all(out["design"][:, 0] == 1.0)


def test_har_rv_reconstruction_property(rng):
    """fitted + residuals reconstructs the regression target exactly."""
    T = 100
    rv = np.abs(rng.standard_normal(T)) * 0.01 + 0.001
    out = har_rv(rv, log_transform=False)
    target = rv[22:]  # no log-transform: target is rv[22:]
    recon = out["fitted"] + out["residuals"]
    np.testing.assert_allclose(recon, target, atol=1e-12)


def test_har_rv_log_transform_reconstruction(rng):
    """With log_transform=True, fitted + residuals = log(max(rv, 1e-12))[22:]."""
    T = 100
    rv = np.abs(rng.standard_normal(T)) * 0.01 + 0.001
    out = har_rv(rv, log_transform=True)
    log_rv = np.log(np.maximum(rv, 1e-12))
    target = log_rv[22:]
    recon = out["fitted"] + out["residuals"]
    np.testing.assert_allclose(recon, target, atol=1e-12)


def test_har_rv_se_positive(rng):
    """All standard errors are strictly positive."""
    rv = np.abs(rng.standard_normal(120)) * 0.01 + 0.001
    out = har_rv(rv, log_transform=True)
    for key in ("se_intercept", "se_d", "se_w", "se_m"):
        assert out[key] > 0.0, f"{key} should be positive"


def test_har_rv_r2_is_float(rng):
    """R2 is a Python float."""
    rv = np.abs(rng.standard_normal(80)) * 0.01 + 0.001
    out = har_rv(rv, log_transform=False)
    assert isinstance(out["R2"], float)


def test_har_rv_log_transform_flag(rng):
    """har_rv stores the log_transform flag in the output dict."""
    rv = np.abs(rng.standard_normal(80)) * 0.01 + 0.001
    assert har_rv(rv, log_transform=True)["log_transform"] is True
    assert har_rv(rv, log_transform=False)["log_transform"] is False


def test_har_rv_hac_lags_override(rng):
    """Explicit hac_lags is stored and used."""
    rv = np.abs(rng.standard_normal(80)) * 0.01 + 0.001
    out = har_rv(rv, log_transform=False, hac_lags=2)
    assert out["hac_lags"] == 2


def test_har_rv_auto_hac_lags_formula(rng):
    """Auto hac_lags = floor(4 * (T_eff / 100)^(2/9)) where T_eff = T - 22."""
    T = 300
    rv = np.abs(rng.standard_normal(T)) * 0.01 + 0.001
    out = har_rv(rv, log_transform=False)
    T_eff = T - 22
    expected = int(np.floor(4 * (T_eff / 100) ** (2 / 9)))
    assert out["hac_lags"] == expected


def test_har_rv_captures_persistence_in_beta_d():
    """High AR(1) persistence in RV => large beta_d (daily coefficient)."""
    rng = np.random.default_rng(0)
    T = 500

    # Very persistent
    rho_hi = 0.95
    rv_hi = np.empty(T)
    rv_hi[0] = 0.001
    for t in range(1, T):
        rv_hi[t] = max(1e-8, (1 - rho_hi) * 0.001 + rho_hi * rv_hi[t - 1]
                       + 1e-5 * rng.standard_normal())

    # Low persistence
    rho_lo = 0.05
    rv_lo = np.empty(T)
    rv_lo[0] = 0.001
    for t in range(1, T):
        rv_lo[t] = max(1e-8, (1 - rho_lo) * 0.001 + rho_lo * rv_lo[t - 1]
                       + 1e-5 * rng.standard_normal())

    beta_d_hi = har_rv(rv_hi, log_transform=True)["beta_d"]
    beta_d_lo = har_rv(rv_lo, log_transform=True)["beta_d"]
    assert beta_d_hi > beta_d_lo, (
        f"High-persistence beta_d ({beta_d_hi:.3f}) should exceed "
        f"low-persistence beta_d ({beta_d_lo:.3f})"
    )


def test_har_rv_fitted_shape_matches_T_minus_22(rng):
    """fitted and residuals have length T - 22."""
    T = 80
    rv = np.abs(rng.standard_normal(T)) * 0.01 + 0.001
    out = har_rv(rv, log_transform=False)
    assert len(out["fitted"]) == T - 22
    assert len(out["residuals"]) == T - 22


def test_har_rv_intercept_is_float(rng):
    """intercept and all scalar outputs are Python floats, not ndarrays."""
    rv = np.abs(rng.standard_normal(80)) * 0.01 + 0.001
    out = har_rv(rv, log_transform=False)
    for key in ("intercept", "beta_d", "beta_w", "beta_m",
                "se_intercept", "se_d", "se_w", "se_m", "R2"):
        assert isinstance(out[key], float), f"{key} should be float"
