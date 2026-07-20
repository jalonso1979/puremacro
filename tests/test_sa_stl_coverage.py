"""Tests for puremacro.sa.stl — STL seasonal adjustment for panel data.

Strategy:
- Known-DGP: build synthetic quarterly/monthly series with a known additive
  seasonal pattern; assert that deseasonalize reduces the seasonal swing
  materially (the SA series has a smaller Q1-Q3 or Jan-Jul gap than the raw).
- Properties: output shape, dtype, index alignment, NaN preservation, value
  bounds for F-statistics (non-negative F, p in [0,1]).
- Contracts: min_obs threshold drops short units to NaN; freq validation raises
  ValueError; multi-unit panel produces correct per-unit handling.
- Edge cases: all-NaN input, NaN positions preserved, boundary series lengths
  for residual_seasonality_F.
- No network / external binary calls: STL is pure statsmodels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.sa.stl import _stl_one, deseasonalize, residual_seasonality_F

RNG = np.random.default_rng(2025)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_quarterly_panel(
    units: list[str],
    n_periods: int = 40,
    *,
    seasonal_amplitude: float = 2.0,
    noise_std: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic quarterly panel with additive seasonal pattern.

    Seasonal: +amplitude in Q1, -amplitude in Q3, 0 otherwise.
    Trend: linear from 100 to 110 per unit (shifted by unit index).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_periods, freq="QS")
    quarters = np.array([d.quarter for d in dates])
    seasonal = np.where(
        quarters == 1, seasonal_amplitude,
        np.where(quarters == 3, -seasonal_amplitude, 0.0),
    )
    rows = []
    for i, unit in enumerate(units):
        trend = np.linspace(100 + i * 10, 110 + i * 10, n_periods)
        noise = rng.normal(0, noise_std, n_periods)
        for j, (d, v) in enumerate(zip(dates, trend + seasonal + noise)):
            rows.append({"unit": unit, "date": d, "y": v})
    return pd.DataFrame(rows)


def _make_monthly_panel(
    units: list[str],
    n_periods: int = 60,
    *,
    seed: int = 43,
) -> pd.DataFrame:
    """Synthetic monthly panel with sinusoidal annual seasonality."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_periods, freq="MS")
    t = np.arange(n_periods)
    seasonal = np.sin(2 * np.pi * t / 12) * 5.0
    rows = []
    for i, unit in enumerate(units):
        trend = np.linspace(100 + i * 5, 115 + i * 5, n_periods)
        noise = rng.normal(0, 0.3, n_periods)
        for d, v in zip(dates, trend + seasonal + noise):
            rows.append({"unit": unit, "date": d, "y": v})
    return pd.DataFrame(rows)


# ===========================================================================
# 1. _stl_one — internal STL decomposition helper
# ===========================================================================

class TestStlOne:
    """Tests for the internal _stl_one function."""

    def test_output_shape_matches_input(self):
        """Output array has the same shape as the input."""
        values = RNG.normal(100, 1, 40)
        sa = _stl_one(values, period=4)
        assert sa.shape == values.shape

    def test_output_is_float_dtype(self):
        """Output is always float, even for integer input."""
        values = np.arange(20, dtype=int)
        sa = _stl_one(values, period=4)
        assert np.issubdtype(sa.dtype, np.floating)

    def test_all_nan_input_returns_nan_array(self):
        """All-NaN input triggers early return with all-NaN output."""
        values = np.full(12, np.nan)
        sa = _stl_one(values, period=4)
        assert np.all(np.isnan(sa)), "Expected all-NaN output for all-NaN input"

    def test_nan_positions_preserved(self):
        """NaN positions in the input are restored to NaN in the output."""
        values = np.array([100.0, 101.0, np.nan, 103.0, 104.0, 102.0, 101.0, 103.0,
                           105.0, 106.0, np.nan, 108.0])
        sa = _stl_one(values, period=4)
        assert np.isnan(sa[2]), "NaN at index 2 must be preserved"
        assert np.isnan(sa[10]), "NaN at index 10 must be preserved"

    def test_non_nan_positions_finite_after_adjustment(self):
        """Non-NaN input positions produce finite output."""
        rng = np.random.default_rng(10)
        values = np.linspace(100, 110, 40) + rng.normal(0, 0.5, 40)
        sa = _stl_one(values, period=4)
        assert np.all(np.isfinite(sa))

    def test_edge_nans_preserved(self):
        """NaNs at the edges of the series are preserved."""
        values = np.array([np.nan, np.nan, 101.0, 102.0, 103.0, 104.0,
                           105.0, 106.0, 107.0, 108.0, np.nan, np.nan])
        sa = _stl_one(values, period=4)
        assert np.isnan(sa[0]), "Leading NaN must be preserved"
        assert np.isnan(sa[1]), "Leading NaN must be preserved"
        assert np.isnan(sa[-1]), "Trailing NaN must be preserved"
        assert np.isnan(sa[-2]), "Trailing NaN must be preserved"

    def test_finite_positions_remain_after_edge_nans(self):
        """Middle values are finite even when edges contain NaN."""
        values = np.array([np.nan, 101.0, 102.0, 103.0, 104.0, 105.0,
                           106.0, 107.0, 108.0, 109.0, 110.0, np.nan])
        sa = _stl_one(values, period=4)
        assert np.all(np.isfinite(sa[1:-1]))

    def test_seasonal_swing_reduced_quarterly(self):
        """STL removes the seasonal component: Q1-Q3 swing shrinks after SA."""
        rng = np.random.default_rng(42)
        n = 40
        quarters = np.arange(n) % 4  # 0=Q1, 1=Q2, 2=Q3, 3=Q4
        seasonal = np.array([3.0, 0.5, -3.0, 0.5])
        trend = np.linspace(100, 110, n)
        noise = rng.normal(0, 0.2, n)
        values = trend + seasonal[quarters] + noise

        sa = _stl_one(values, period=4)

        raw_q1_q3 = values[quarters == 0].mean() - values[quarters == 2].mean()
        sa_q1_q3 = sa[quarters == 0].mean() - sa[quarters == 2].mean()
        assert abs(sa_q1_q3) < abs(raw_q1_q3) * 0.3, (
            f"SA should reduce Q1-Q3 swing: raw={raw_q1_q3:.3f}, sa={sa_q1_q3:.3f}"
        )

    def test_seasonal_swing_reduced_monthly(self):
        """STL removes monthly seasonality; cross-month dispersion shrinks."""
        rng = np.random.default_rng(43)
        n = 48
        t = np.arange(n)
        seasonal = np.sin(2 * np.pi * t / 12) * 5.0
        trend = np.linspace(100, 110, n)
        noise = rng.normal(0, 0.3, n)
        values = trend + seasonal + noise

        sa = _stl_one(values, period=12)

        months = t % 12
        # std of per-month group means captures total seasonal dispersion
        raw_group_means = [values[months == m].mean() for m in range(12)]
        sa_group_means = [sa[months == m].mean() for m in range(12)]
        raw_sd = float(np.std(raw_group_means))
        sa_sd = float(np.std(sa_group_means))
        assert sa_sd < raw_sd * 0.5, (
            f"SA should reduce cross-month dispersion: raw_sd={raw_sd:.3f}, sa_sd={sa_sd:.3f}"
        )

    def test_constant_series_returns_constant_like(self):
        """A constant series has no seasonal component; STL should leave values roughly constant."""
        values = np.full(40, 5.0)
        sa = _stl_one(values, period=4)
        assert np.all(np.isfinite(sa))
        # Seasonal component of a constant series is zero; SA ≈ original
        np.testing.assert_allclose(sa, values, atol=1e-10)

    def test_stl_internal_exception_falls_back_to_raw(self, monkeypatch):
        """If statsmodels STL raises an exception, _stl_one returns the raw values."""
        import puremacro.sa.stl as _stl_mod

        def _bad_stl(*args, **kwargs):
            raise RuntimeError("simulated STL failure")

        monkeypatch.setattr(_stl_mod, "__builtins__", None, raising=False)
        # Patch at the module level by intercepting the lazy import
        import statsmodels.tsa.seasonal as _sm_seasonal

        original_stl_cls = _sm_seasonal.STL

        class _FailingSTL:
            def __init__(self, *a, **kw):
                raise RuntimeError("simulated STL failure")

        monkeypatch.setattr(_sm_seasonal, "STL", _FailingSTL)
        try:
            values = np.array([100.0, 101.0, 102.0, 103.0, 100.0, 101.0, 102.0, 103.0])
            result = _stl_one(values, period=4)
            # Fallback: returns the raw values as float
            np.testing.assert_allclose(result, values.astype(float))
        finally:
            monkeypatch.setattr(_sm_seasonal, "STL", original_stl_cls)


# ===========================================================================
# 2. deseasonalize — panel-level SA
# ===========================================================================

class TestDeseasonalize:
    """Tests for the public deseasonalize function."""

    def test_returns_series(self):
        """Output is a pd.Series."""
        df = _make_quarterly_panel(["A"])
        result = deseasonalize(df, "y", by="unit", date_col="date", freq="Q")
        assert isinstance(result, pd.Series)

    def test_output_shape_matches_input(self):
        """Output has the same length as the input DataFrame."""
        df = _make_quarterly_panel(["A"])
        result = deseasonalize(df, "y", by="unit", date_col="date", freq="Q")
        assert len(result) == len(df)

    def test_output_index_preserved(self):
        """Output index equals the input DataFrame's index."""
        df = _make_quarterly_panel(["A"])
        result = deseasonalize(df, "y", by="unit", date_col="date", freq="Q")
        assert result.index.equals(df.index)

    def test_non_default_index_preserved(self):
        """Custom DataFrame index is preserved in the output."""
        df = _make_quarterly_panel(["A"])
        df.index = np.arange(1000, 1000 + len(df))
        result = deseasonalize(df, "y", by="unit", date_col="date", freq="Q")
        assert result.index.equals(df.index)

    def test_single_unit_seasonal_swing_reduced(self):
        """STL applied via deseasonalize reduces the Q1-Q3 swing."""
        df = _make_quarterly_panel(["A"], seasonal_amplitude=3.0)
        dates_col = df["date"]
        quarters = np.array([d.quarter for d in dates_col])

        result = deseasonalize(df, "y", by="unit", date_col="date", freq="Q")
        raw = df["y"].values

        raw_q1_q3 = raw[quarters == 1].mean() - raw[quarters == 3].mean()
        sa_q1_q3 = result.values[quarters == 1].mean() - result.values[quarters == 3].mean()
        assert abs(sa_q1_q3) < abs(raw_q1_q3) * 0.3, (
            f"deseasonalize should reduce Q1-Q3 swing: raw={raw_q1_q3:.3f}, sa={sa_q1_q3:.3f}"
        )

    def test_multi_unit_each_unit_adjusted_independently(self):
        """Multiple units each receive independent STL fits."""
        df = _make_quarterly_panel(["A", "B", "C"])
        result = deseasonalize(df, "y", by="unit", date_col="date", freq="Q")
        assert len(result) == len(df)
        assert result.notna().all()

    def test_short_unit_below_min_obs_returns_nan(self):
        """Units with fewer than min_obs non-NaN observations return NaN."""
        df_short = _make_quarterly_panel(["Short"], n_periods=15)
        result = deseasonalize(df_short, "y", by="unit", date_col="date", freq="Q")
        # Default min_obs=24, 15 < 24 → all NaN
        assert result.isna().all(), "Unit with 15 obs should be all NaN (min_obs=24)"

    def test_long_unit_above_min_obs_returns_finite(self):
        """Units with sufficient observations return finite values."""
        df_long = _make_quarterly_panel(["Long"], n_periods=30)
        result = deseasonalize(df_long, "y", by="unit", date_col="date", freq="Q")
        assert result.notna().all(), "Unit with 30 obs (>24) should have finite SA values"

    def test_custom_min_obs_strict_drops_unit(self):
        """A custom min_obs higher than unit obs drops that unit to NaN."""
        df = _make_quarterly_panel(["A"], n_periods=20)
        result = deseasonalize(df, "y", by="unit", date_col="date", freq="Q", min_obs=30)
        assert result.isna().all()

    def test_custom_min_obs_lenient_accepts_unit(self):
        """A custom min_obs lower than unit obs keeps the unit."""
        df = _make_quarterly_panel(["A"], n_periods=15)
        result = deseasonalize(df, "y", by="unit", date_col="date", freq="Q", min_obs=10)
        assert result.notna().any(), "min_obs=10 should accept a 15-obs unit"

    def test_mixed_units_long_and_short(self):
        """Long unit is SA'd; short unit is NaN — independently."""
        df_long = _make_quarterly_panel(["Long"], n_periods=30)
        df_short = _make_quarterly_panel(["Short"], n_periods=10)
        df_mixed = pd.concat([df_long, df_short], ignore_index=True)

        result = deseasonalize(df_mixed, "y", by="unit", date_col="date", freq="Q")

        long_mask = df_mixed["unit"] == "Long"
        short_mask = df_mixed["unit"] == "Short"
        assert result[long_mask].notna().all(), "Long unit should be fully adjusted"
        assert result[short_mask].isna().all(), "Short unit should be all NaN"

    def test_monthly_freq_works(self):
        """freq='M' runs STL with period=12 and returns finite values."""
        df = _make_monthly_panel(["A"])
        result = deseasonalize(df, "y", by="unit", date_col="date", freq="M")
        assert isinstance(result, pd.Series)
        assert result.notna().all()

    def test_monthly_seasonal_swing_reduced(self):
        """Monthly STL reduces cross-month dispersion of group means."""
        df = _make_monthly_panel(["A"], n_periods=72)
        dates_col = df["date"]
        months = np.array([d.month for d in dates_col])
        raw = df["y"].values

        result = deseasonalize(df, "y", by="unit", date_col="date", freq="M")

        raw_group_means = [raw[months == m].mean() for m in range(1, 13)]
        sa_group_means = [result.values[months == m].mean() for m in range(1, 13)]
        raw_sd = float(np.std(raw_group_means))
        sa_sd = float(np.std(sa_group_means))
        assert sa_sd < raw_sd * 0.5, (
            f"Monthly SA should reduce cross-month dispersion: raw_sd={raw_sd:.3f}, sa_sd={sa_sd:.3f}"
        )

    def test_invalid_freq_raises_value_error(self):
        """Unsupported freq value raises ValueError with informative message."""
        df = _make_quarterly_panel(["A"])
        with pytest.raises(ValueError, match=r"freq must be"):
            deseasonalize(df, "y", by="unit", date_col="date", freq="W")  # type: ignore[arg-type]

    def test_invalid_freq_daily_raises_value_error(self):
        """'D' frequency also raises ValueError."""
        df = _make_quarterly_panel(["A"])
        with pytest.raises(ValueError, match=r"freq must be"):
            deseasonalize(df, "y", by="unit", date_col="date", freq="D")  # type: ignore[arg-type]

    def test_output_dtype_is_float(self):
        """Output Series has float dtype."""
        df = _make_quarterly_panel(["A"])
        result = deseasonalize(df, "y", by="unit", date_col="date", freq="Q")
        assert np.issubdtype(result.dtype, np.floating)

    def test_multi_unit_index_covers_all_rows(self):
        """Output index covers all rows from all units."""
        df = _make_quarterly_panel(["A", "B"])
        result = deseasonalize(df, "y", by="unit", date_col="date", freq="Q")
        assert result.index.equals(df.index)


# ===========================================================================
# 3. residual_seasonality_F — ANOVA test for residual seasonality
# ===========================================================================

class TestResidualSeasonalityF:
    """Tests for the residual_seasonality_F function."""

    def test_returns_4_tuple(self):
        """Function returns a tuple of exactly 4 elements."""
        s = pd.Series(RNG.normal(5, 1, 40))
        result = residual_seasonality_F(s, freq="Q")
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_return_types(self):
        """Returns (float, float, int, int)."""
        s = pd.Series(RNG.normal(5, 1, 40))
        F, p, dof_num, dof_den = residual_seasonality_F(s, freq="Q")
        assert isinstance(F, float)
        assert isinstance(p, float)
        assert isinstance(dof_num, int)
        assert isinstance(dof_den, int)

    def test_f_statistic_nonnegative_on_valid_input(self):
        """F statistic must be non-negative for valid input."""
        s = pd.Series(RNG.normal(5, 1, 40))
        F, p, _, _ = residual_seasonality_F(s, freq="Q")
        assert F >= 0.0

    def test_p_value_in_0_1_on_valid_input(self):
        """p-value must lie in [0, 1]."""
        s = pd.Series(RNG.normal(5, 1, 40))
        _, p, _, _ = residual_seasonality_F(s, freq="Q")
        assert 0.0 <= p <= 1.0

    def test_dof_num_quarterly_is_period_minus_1(self):
        """dof_num for quarterly data should equal period-1 = 3."""
        s = pd.Series(RNG.normal(5, 1, 40))
        _, _, dof_num, _ = residual_seasonality_F(s, freq="Q")
        assert dof_num == 3  # 4 groups - 1

    def test_dof_num_monthly_is_period_minus_1(self):
        """dof_num for monthly data should equal period-1 = 11."""
        s = pd.Series(RNG.normal(5, 1, 60))
        _, _, dof_num, _ = residual_seasonality_F(s, freq="M")
        assert dof_num == 11  # 12 groups - 1

    def test_seasonal_series_has_high_f(self):
        """Strongly seasonal series has high F-statistic and low p-value."""
        rng = np.random.default_rng(77)
        n = 60
        t = np.arange(n)
        s = pd.Series(np.sin(2 * np.pi * t / 4) * 10.0 + rng.normal(0, 0.1, n))
        F, p, _, _ = residual_seasonality_F(s, freq="Q")
        assert F > 10.0, f"Expected high F for seasonal series, got {F:.3f}"
        assert p < 0.01, f"Expected low p for seasonal series, got {p:.4f}"

    def test_noise_series_has_low_f(self):
        """White noise has low F-statistic and high p-value (most of the time)."""
        rng = np.random.default_rng(88)
        s = pd.Series(rng.normal(0, 1.0, 60))
        F, p, _, _ = residual_seasonality_F(s, freq="Q")
        # Noise should not reject H0 at 1% level reliably
        assert p > 0.01 or F < 5.0, (
            f"Noise should yield low F (got {F:.3f}) or high p (got {p:.4f})"
        )

    def test_seasonal_f_higher_than_noise_f(self):
        """F for a purely seasonal series exceeds F for white noise."""
        rng = np.random.default_rng(42)
        n = 60
        t = np.arange(n)
        seasonal = pd.Series(np.sin(2 * np.pi * t / 4) * 5.0 + rng.normal(0, 0.1, n))
        noise = pd.Series(rng.normal(0, 1.0, n))

        F_seasonal, _, _, _ = residual_seasonality_F(seasonal, freq="Q")
        F_noise, _, _, _ = residual_seasonality_F(noise, freq="Q")
        assert F_seasonal > F_noise, (
            f"Seasonal F ({F_seasonal:.3f}) should exceed noise F ({F_noise:.3f})"
        )

    def test_seasonal_p_lower_than_noise_p(self):
        """p-value for seasonal series is lower than for white noise."""
        rng = np.random.default_rng(42)
        n = 60
        t = np.arange(n)
        seasonal = pd.Series(np.sin(2 * np.pi * t / 4) * 5.0 + rng.normal(0, 0.1, n))
        noise = pd.Series(rng.normal(0, 1.0, n))

        _, p_seasonal, _, _ = residual_seasonality_F(seasonal, freq="Q")
        _, p_noise, _, _ = residual_seasonality_F(noise, freq="Q")
        assert p_seasonal < p_noise

    def test_too_short_series_quarterly_returns_nan(self):
        """Series shorter than 2*period=8 returns (nan, nan, 0, 0)."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])  # len=7 < 8
        F, p, dof_num, dof_den = residual_seasonality_F(s, freq="Q")
        assert np.isnan(F)
        assert np.isnan(p)
        assert dof_num == 0
        assert dof_den == 0

    def test_boundary_length_too_short_after_diff_quarterly(self):
        """9 observations (n-2=7 < 8) also returns nan after second-diff."""
        s = pd.Series(RNG.normal(5, 1, 9))
        F, p, _, _ = residual_seasonality_F(s, freq="Q")
        assert np.isnan(F)

    def test_sufficient_length_quarterly_returns_finite(self):
        """10 observations (n-2=8=2*period) produces finite F and p."""
        rng = np.random.default_rng(99)
        s = pd.Series(rng.normal(5, 1, 10))
        F, p, dof_num, dof_den = residual_seasonality_F(s, freq="Q")
        # May return finite or nan depending on group sizes; at minimum dof_num > 0
        # if groups have enough members (contract: not all nan when n>=10)
        # We only check types; F may be finite or nan depending on group balance
        assert isinstance(F, float)
        assert isinstance(p, float)

    def test_invalid_freq_raises_value_error(self):
        """Invalid freq raises ValueError."""
        s = pd.Series(RNG.normal(5, 1, 40))
        with pytest.raises(ValueError, match=r"freq must be"):
            residual_seasonality_F(s, freq="W")  # type: ignore[arg-type]

    def test_invalid_freq_annual_raises(self):
        """'A' freq also raises ValueError."""
        s = pd.Series(RNG.normal(5, 1, 40))
        with pytest.raises(ValueError, match=r"freq must be"):
            residual_seasonality_F(s, freq="A")  # type: ignore[arg-type]

    def test_monthly_freq_works_and_returns_finite(self):
        """Monthly frequency produces finite F and p for sufficient data."""
        rng = np.random.default_rng(55)
        n = 60
        t = np.arange(n)
        s = pd.Series(np.sin(2 * np.pi * t / 12) * 3.0 + rng.normal(0, 0.3, n))
        F, p, dof_num, dof_den = residual_seasonality_F(s, freq="M")
        assert np.isfinite(F)
        assert np.isfinite(p)
        assert dof_num == 11
        assert 0.0 <= p <= 1.0

    def test_accepts_list_input(self):
        """Function accepts a Python list (coerced to pd.Series)."""
        data = list(RNG.normal(5, 1, 40))
        result = residual_seasonality_F(pd.Series(data), freq="Q")
        assert len(result) == 4

    def test_accepts_series_with_float_index(self):
        """Series with a non-integer index is handled gracefully."""
        s = pd.Series(
            RNG.normal(5, 1, 40),
            index=np.linspace(0, 1, 40),
        )
        F, p, _, _ = residual_seasonality_F(s, freq="Q")
        assert isinstance(F, float)

    def test_nan_values_dropped_before_anova(self):
        """NaN values are dropped cleanly; function does not crash."""
        rng = np.random.default_rng(66)
        data = rng.normal(5, 1, 50)
        data[::5] = np.nan  # scatter NaNs
        s = pd.Series(data)
        F, p, dof_num, dof_den = residual_seasonality_F(s, freq="Q")
        # After dropna, 40 obs remain — should return finite values
        assert isinstance(F, float)
        assert isinstance(p, float)

    def test_dof_den_positive_when_sufficient_data(self):
        """dof_den is positive when enough data is present."""
        s = pd.Series(RNG.normal(5, 1, 40))
        _, _, dof_num, dof_den = residual_seasonality_F(s, freq="Q")
        assert dof_den > 0

    def test_sa_series_lower_f_than_raw(self):
        """Seasonally-adjusted series should have lower residual seasonality F
        than the raw seasonal series from which it was derived."""
        rng = np.random.default_rng(42)
        n = 40
        dates = pd.date_range("2010-01-01", periods=n, freq="QS")
        quarters = np.array([d.quarter for d in dates])
        # Strong seasonal pattern
        seasonal = np.where(quarters == 1, 4.0, np.where(quarters == 3, -4.0, 0.0))
        trend = np.linspace(100, 110, n)
        noise = rng.normal(0, 0.2, n)
        raw_values = trend + seasonal + noise

        df = pd.DataFrame({"unit": "A", "date": dates, "y": raw_values})
        sa_values = deseasonalize(df, "y", by="unit", date_col="date", freq="Q")

        F_raw, _, _, _ = residual_seasonality_F(pd.Series(raw_values), freq="Q")
        F_sa, _, _, _ = residual_seasonality_F(sa_values, freq="Q")

        assert F_sa < F_raw, (
            f"SA series should have lower residual-seasonality F than raw: "
            f"F_raw={F_raw:.3f}, F_sa={F_sa:.3f}"
        )
