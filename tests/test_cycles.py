"""Tests for puremacro.cycles.hamilton_filter."""
import numpy as np
import pytest

from puremacro.cycles import hamilton_filter


@pytest.mark.pyodide_smoke
def test_constant_series_cycle_is_zero():
    y = np.ones(60)
    cycle, trend = hamilton_filter(y, h=8, p=4)
    # First 11 (h + p - 1 = 11) are NaN; rest of cycle is ~0
    assert np.all(np.isnan(cycle[:11]))
    np.testing.assert_allclose(cycle[11:], 0.0, atol=1e-10)
    np.testing.assert_allclose(trend[11:], 1.0, atol=1e-10)


def test_linear_trend_cycle_is_zero():
    y = np.arange(80, dtype=float) * 0.5 + 3.0
    cycle, trend = hamilton_filter(y, h=8, p=4)
    np.testing.assert_allclose(cycle[11:], 0.0, atol=1e-9)
    np.testing.assert_allclose(trend[11:], y[11:], atol=1e-9)


def test_high_persistence_ar1_cycle_smaller_than_input():
    rng = np.random.default_rng(0)
    T = 400
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.95 * y[t - 1] + rng.standard_normal()
    cycle, trend = hamilton_filter(y, h=8, p=4)
    # Cycle removes a slow-moving component → variance of cycle < variance of y
    valid = ~np.isnan(cycle)
    assert np.var(cycle[valid]) < np.var(y[valid])


def test_output_shape_and_nan_prefix_default():
    T = 50
    y = np.arange(T, dtype=float)
    cycle, trend = hamilton_filter(y)  # defaults h=8, p=4
    assert cycle.shape == (T,)
    assert trend.shape == (T,)
    n_nan = 8 + 4 - 1  # 11
    assert np.all(np.isnan(cycle[:n_nan]))
    assert np.all(np.isnan(trend[:n_nan]))
    assert np.all(np.isfinite(cycle[n_nan:]))
    assert np.all(np.isfinite(trend[n_nan:]))


def test_custom_kwargs_monthly():
    """h=24, p=12 (Hamilton's monthly convention) → 35 leading NaNs."""
    T = 200
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.standard_normal(T))
    cycle, trend = hamilton_filter(y, h=24, p=12)
    n_nan = 24 + 12 - 1  # 35
    assert np.all(np.isnan(cycle[:n_nan]))
    assert np.all(np.isfinite(cycle[n_nan:]))


def test_too_short_input_raises():
    y = np.arange(10, dtype=float)  # T=10, h+p=12 → not enough
    with pytest.raises(ValueError):
        hamilton_filter(y, h=8, p=4)


def test_cycle_plus_trend_recovers_y():
    rng = np.random.default_rng(2)
    y = np.cumsum(rng.standard_normal(100))
    cycle, trend = hamilton_filter(y, h=8, p=4)
    valid = ~np.isnan(cycle)
    # Where defined: y[t] = cycle[t] + trend[t] (with t indexed in original y)
    np.testing.assert_allclose(cycle[valid] + trend[valid], y[valid], atol=1e-10)


def test_accepts_pandas_series():
    """The reference implementation at MAV root accepts a pd.Series; ours should too."""
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2000-01-01", periods=60, freq="QE")
    y = pd.Series(np.arange(60.0), index=idx)
    cycle, trend = hamilton_filter(y, h=8, p=4)
    assert cycle.shape == (60,)
    assert trend.shape == (60,)
    np.testing.assert_allclose(cycle[11:], 0.0, atol=1e-9)


def test_baxter_king_filter_properties():
    from puremacro.cycles import baxter_king_filter

    rng = np.random.default_rng(42)
    T = 100
    K = 12
    y = np.cumsum(rng.standard_normal(T))
    cycle, trend = baxter_king_filter(y, low=6, high=32, K=K)

    assert cycle.shape == (T,)
    assert trend.shape == (T,)
    assert np.all(np.isnan(cycle[:K]))
    assert np.all(np.isnan(cycle[T - K :]))
    valid = ~np.isnan(cycle)
    assert np.sum(valid) == T - 2 * K
    np.testing.assert_allclose(cycle[valid] + trend[valid], y[valid], atol=1e-10)


def test_christiano_fitzgerald_filter_properties():
    from puremacro.cycles import christiano_fitzgerald_filter

    rng = np.random.default_rng(101)
    T = 80
    y = np.cumsum(rng.standard_normal(T)) + 2.0 * np.arange(T)
    cycle, trend = christiano_fitzgerald_filter(y, low=6, high=32, drift=True)

    assert cycle.shape == (T,)
    assert trend.shape == (T,)
    assert np.all(np.isfinite(cycle))
    assert np.all(np.isfinite(trend))
    np.testing.assert_allclose(cycle + trend, y, atol=1e-10)


def test_beveridge_nelson_filter_properties():
    from puremacro.cycles import beveridge_nelson_filter

    rng = np.random.default_rng(99)
    T = 120
    p = 4
    # Stationary AR(1) in first differences
    e = rng.normal(0, 1, T)
    dy = np.empty(T)
    dy[0] = 0.5
    for t in range(1, T):
        dy[t] = 0.4 * dy[t - 1] + e[t]
    y = np.cumsum(dy)

    cycle, trend = beveridge_nelson_filter(y, p=p)
    assert cycle.shape == (T,)
    assert trend.shape == (T,)
    valid = ~np.isnan(cycle)
    assert np.sum(valid) == T - p
    np.testing.assert_allclose(cycle[valid] + trend[valid], y[valid], atol=1e-10)


def test_filter_errors():
    from puremacro.cycles import baxter_king_filter, christiano_fitzgerald_filter, beveridge_nelson_filter

    y_short = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="too short"):
        baxter_king_filter(y_short, K=12)
    with pytest.raises(ValueError, match="require 1 < low < high"):
        baxter_king_filter(np.ones(50), low=32, high=6)
    with pytest.raises(ValueError, match="too short"):
        christiano_fitzgerald_filter(np.ones(2))
    with pytest.raises(ValueError, match="too short"):
        beveridge_nelson_filter(np.ones(5), p=4)
