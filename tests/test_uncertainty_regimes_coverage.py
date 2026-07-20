"""Coverage tests for puremacro.uncertainty.regimes.

Public API:
    CALENDAR_REGIMES  — dict of fixed regime boundaries
    add_calendar_regime(panel, *, date_col, out_col) -> DataFrame
    bai_perron_breaks(series, *, max_breaks, min_segment) -> list[BreakResult]
    state_indicator_smooth(series, *, lag, gamma, threshold_quantile) -> Series

Strategy:
    - Known-value tests: analytically derivable expected answers.
    - Property tests: bounds, monotonicity, output shape/dtype/columns.
    - Edge cases: short series, NaN-prefix, boundary dates, custom columns.
    - BreakResult typed-dict contract: required keys, index in range, CI ordering.
    - state_indicator_smooth: F=0.5 at threshold, monotone in z, lag shifts index.
    - bai_perron_breaks: slow-marked where the DP is long (large series with 2 breaks).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.uncertainty.regimes import (
    CALENDAR_REGIMES,
    BreakResult,
    add_calendar_regime,
    bai_perron_breaks,
    state_indicator_smooth,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monotone_series(n: int = 100, freq: str = "MS") -> pd.Series:
    """Strictly increasing series 0, 1, …, n-1 with a MonthStart DatetimeIndex."""
    return pd.Series(
        np.arange(n, dtype=float),
        index=pd.date_range("2000-01-01", periods=n, freq=freq),
    )


def _step_series(n_before: int, n_after: int,
                 val_before: float = 0.0,
                 val_after: float = 10.0,
                 noise_sd: float = 0.01,
                 seed: int = 42) -> pd.Series:
    """Series with exactly one mean-shift break."""
    rng = np.random.default_rng(seed)
    n = n_before + n_after
    y = np.concatenate([
        np.full(n_before, val_before) + rng.standard_normal(n_before) * noise_sd,
        np.full(n_after, val_after) + rng.standard_normal(n_after) * noise_sd,
    ])
    return pd.Series(y, index=pd.date_range("2000-01-01", periods=n, freq="MS"))


def _two_step_series(n_seg: int = 60, seed: int = 42) -> pd.Series:
    """Series with exactly two mean-shift breaks (3 segments: 0, 5, 0)."""
    rng = np.random.default_rng(seed)
    n = 3 * n_seg
    y = np.concatenate([
        np.zeros(n_seg) + rng.standard_normal(n_seg) * 0.01,
        np.full(n_seg, 5.0) + rng.standard_normal(n_seg) * 0.01,
        np.zeros(n_seg) + rng.standard_normal(n_seg) * 0.01,
    ])
    return pd.Series(y, index=pd.date_range("2000-01-01", periods=n, freq="MS"))


# ===========================================================================
# CALENDAR_REGIMES constant
# ===========================================================================

class TestCalendarRegimesConstant:
    def test_has_four_entries(self):
        assert len(CALENDAR_REGIMES) == 4

    def test_expected_keys(self):
        expected = {"cold_war_end", "great_moderation", "gfc_era", "pandemic_plus"}
        assert set(CALENDAR_REGIMES.keys()) == expected

    def test_values_are_timestamp_pairs(self):
        for name, (lo, hi) in CALENDAR_REGIMES.items():
            assert isinstance(lo, pd.Timestamp), f"{name}: lo is not Timestamp"
            assert isinstance(hi, pd.Timestamp), f"{name}: hi is not Timestamp"

    def test_all_ranges_are_non_degenerate(self):
        """Each regime spans at least one day."""
        for name, (lo, hi) in CALENDAR_REGIMES.items():
            assert lo < hi, f"{name}: lo >= hi"

    def test_regimes_are_contiguous_and_non_overlapping(self):
        """Adjacent regimes must share a boundary: hi of one equals lo of next minus 1 day."""
        sorted_regimes = sorted(CALENDAR_REGIMES.values(), key=lambda x: x[0])
        for (_, hi), (lo_next, _) in zip(sorted_regimes[:-1], sorted_regimes[1:]):
            gap = lo_next - hi
            assert gap == pd.Timedelta("1 day"), (
                f"Non-contiguous regimes: {hi} vs {lo_next} (gap={gap})"
            )

    def test_cold_war_end_starts_1985(self):
        lo, _ = CALENDAR_REGIMES["cold_war_end"]
        assert lo == pd.Timestamp("1985-01-01")

    def test_pandemic_plus_hi_is_far_future(self):
        _, hi = CALENDAR_REGIMES["pandemic_plus"]
        assert hi >= pd.Timestamp("2099-01-01")


# ===========================================================================
# add_calendar_regime
# ===========================================================================

class TestAddCalendarRegimeBoundaryDates:
    """Each boundary date maps to the expected regime (or NaN outside all ranges)."""

    _boundary_cases = [
        ("1984-12-31", None),            # one day before cold_war_end starts
        ("1985-01-01", "cold_war_end"),  # first day of cold_war_end
        ("1991-12-31", "cold_war_end"),  # last day of cold_war_end
        ("1992-01-01", "great_moderation"),
        ("2007-12-31", "great_moderation"),
        ("2008-01-01", "gfc_era"),
        ("2019-12-31", "gfc_era"),
        ("2020-01-01", "pandemic_plus"),
        ("2099-12-31", "pandemic_plus"),
    ]

    def _make_df(self, dates):
        return pd.DataFrame({"date": [pd.Timestamp(d) for d in dates]})

    @pytest.mark.parametrize("date_str,expected_regime", _boundary_cases)
    def test_boundary(self, date_str, expected_regime):
        df = self._make_df([date_str])
        out = add_calendar_regime(df)
        val = out["regime"].iloc[0]
        if expected_regime is None:
            assert val is np.nan or (isinstance(val, float) and np.isnan(val))
        else:
            assert val == expected_regime


class TestAddCalendarRegimeOutputContract:
    def test_returns_dataframe(self):
        df = pd.DataFrame({"date": pd.date_range("2008-01-01", periods=3, freq="YS")})
        out = add_calendar_regime(df)
        assert isinstance(out, pd.DataFrame)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"date": pd.date_range("2008-01-01", periods=3, freq="YS")})
        df_orig_cols = list(df.columns)
        _ = add_calendar_regime(df)
        assert list(df.columns) == df_orig_cols

    def test_adds_exactly_one_column(self):
        df = pd.DataFrame({"date": [pd.Timestamp("2008-01-01")]})
        n_cols_before = len(df.columns)
        out = add_calendar_regime(df)
        assert len(out.columns) == n_cols_before + 1

    def test_default_out_col_is_regime(self):
        df = pd.DataFrame({"date": [pd.Timestamp("2008-01-01")]})
        out = add_calendar_regime(df)
        assert "regime" in out.columns

    def test_preserves_all_existing_columns(self):
        df = pd.DataFrame({
            "date": [pd.Timestamp("2008-01-01")],
            "value": [42.0],
            "code": ["USA"],
        })
        out = add_calendar_regime(df)
        for col in ["date", "value", "code"]:
            assert col in out.columns
        assert out["value"].iloc[0] == 42.0

    def test_regime_col_dtype_is_object(self):
        """Regime column should be object dtype (strings + NaN)."""
        df = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=5, freq="YS")})
        out = add_calendar_regime(df)
        assert out["regime"].dtype == object

    def test_row_count_preserved(self):
        n = 20
        df = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=n, freq="QS")})
        out = add_calendar_regime(df)
        assert len(out) == n

    def test_index_preserved(self):
        idx = pd.Index([10, 20, 30])
        df = pd.DataFrame({"date": pd.date_range("2008-01-01", periods=3, freq="YS")}, index=idx)
        out = add_calendar_regime(df)
        pd.testing.assert_index_equal(out.index, idx)


class TestAddCalendarRegimeCustomColumns:
    def test_custom_date_col(self):
        df = pd.DataFrame({"when": pd.date_range("2008-01-01", periods=2, freq="YS")})
        out = add_calendar_regime(df, date_col="when")
        assert "regime" in out.columns
        assert (out["regime"] == "gfc_era").all()

    def test_custom_out_col(self):
        df = pd.DataFrame({"date": pd.date_range("2008-01-01", periods=2, freq="YS")})
        out = add_calendar_regime(df, out_col="epoch")
        assert "epoch" in out.columns
        assert "regime" not in out.columns

    def test_custom_date_col_and_out_col(self):
        df = pd.DataFrame({"when": pd.date_range("1992-01-01", periods=3, freq="YS")})
        out = add_calendar_regime(df, date_col="when", out_col="period")
        assert "period" in out.columns
        assert (out["period"] == "great_moderation").all()


class TestAddCalendarRegimeNaN:
    def test_pre_1985_dates_are_nan(self):
        """Dates before 1985-01-01 fall outside all regimes -> NaN."""
        df = pd.DataFrame({"date": pd.date_range("1980-01-01", periods=5, freq="YS")})
        out = add_calendar_regime(df)
        assert out["regime"].isna().all()

    def test_mixed_in_and_out_of_regime(self):
        """Some rows get a regime, pre-1985 rows get NaN."""
        dates = [pd.Timestamp("1983-01-01"), pd.Timestamp("1985-01-01")]
        df = pd.DataFrame({"date": dates})
        out = add_calendar_regime(df)
        assert pd.isna(out["regime"].iloc[0])
        assert out["regime"].iloc[1] == "cold_war_end"

    def test_pandemic_plus_covers_post_2020(self):
        """2020+ dates all get pandemic_plus."""
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5, freq="YS")})
        out = add_calendar_regime(df)
        assert (out["regime"] == "pandemic_plus").all()


class TestAddCalendarRegimeKnownAssignments:
    """Spot-check several well-known dates for correctness."""

    _cases = [
        ("1990-06-01", "cold_war_end"),
        ("2000-01-01", "great_moderation"),
        ("2008-09-15", "gfc_era"),       # Lehman collapse
        ("2020-03-01", "pandemic_plus"),
    ]

    @pytest.mark.parametrize("date_str,expected", _cases)
    def test_known_date(self, date_str, expected):
        df = pd.DataFrame({"date": [pd.Timestamp(date_str)]})
        out = add_calendar_regime(df)
        assert out["regime"].iloc[0] == expected


# ===========================================================================
# bai_perron_breaks
# ===========================================================================

class TestBaiPerronBreaksEmptyReturn:
    def test_too_short_returns_empty_list(self):
        """2*min_segment > len(series) -> []."""
        rng = np.random.default_rng(0)
        s = pd.Series(
            rng.standard_normal(30),
            index=pd.date_range("2000-01-01", periods=30, freq="MS"),
        )
        result = bai_perron_breaks(s, min_segment=24)
        assert result == []

    def test_returns_list_type(self):
        rng = np.random.default_rng(0)
        s = pd.Series(
            rng.standard_normal(100),
            index=pd.date_range("2000-01-01", periods=100, freq="MS"),
        )
        result = bai_perron_breaks(s)
        assert isinstance(result, list)

    def test_nan_prefix_does_not_crash(self):
        """Leading NaNs are dropped by dropna(); surviving series is analysed."""
        rng = np.random.default_rng(0)
        vals = [np.nan] * 5 + list(rng.standard_normal(30))
        s = pd.Series(vals, index=pd.date_range("2000-01-01", periods=35, freq="MS"))
        result = bai_perron_breaks(s, min_segment=24)
        # Series with 30 non-NaN obs is too short for 2*24; should return []
        assert result == []

    def test_all_nan_returns_empty(self):
        """All-NaN series has 0 valid obs after dropna -> []."""
        s = pd.Series(
            [np.nan] * 50,
            index=pd.date_range("2000-01-01", periods=50, freq="MS"),
        )
        result = bai_perron_breaks(s)
        assert result == []


class TestBaiPerronBreaksBreakResultContract:
    """BreakResult typed-dict must have exactly the right keys and types."""

    @pytest.fixture
    def single_break_result(self):
        """A clear one-break series that reliably finds one break."""
        s = _step_series(60, 60, val_before=0.0, val_after=10.0, noise_sd=0.005)
        return bai_perron_breaks(s, max_breaks=5, min_segment=24)

    def test_finds_at_least_one_break(self, single_break_result):
        assert len(single_break_result) >= 1

    def test_break_has_required_keys(self, single_break_result):
        b = single_break_result[0]
        for key in ("date", "index", "ci_lo", "ci_hi"):
            assert key in b, f"Missing key: {key}"

    def test_break_date_is_timestamp(self, single_break_result):
        assert isinstance(single_break_result[0]["date"], pd.Timestamp)

    def test_break_index_is_int(self, single_break_result):
        assert isinstance(single_break_result[0]["index"], int)

    def test_ci_lo_is_timestamp(self, single_break_result):
        assert isinstance(single_break_result[0]["ci_lo"], pd.Timestamp)

    def test_ci_hi_is_timestamp(self, single_break_result):
        assert isinstance(single_break_result[0]["ci_hi"], pd.Timestamp)

    def test_ci_lo_le_date(self, single_break_result):
        b = single_break_result[0]
        assert b["ci_lo"] <= b["date"]

    def test_date_le_ci_hi(self, single_break_result):
        b = single_break_result[0]
        assert b["date"] <= b["ci_hi"]

    def test_index_within_series_bounds(self, single_break_result):
        """Index must be a valid position in the underlying (non-NaN) series."""
        s = _step_series(60, 60, val_before=0.0, val_after=10.0, noise_sd=0.005)
        s_clean = s.dropna()
        for b in single_break_result:
            assert 1 <= b["index"] <= len(s_clean) - 1


class TestBaiPerronBreaksDetection:
    """Verify that well-separated breaks are found near the correct location."""

    def test_one_clear_break_at_midpoint(self):
        """One-break step series: break must be at index 60 (the midpoint)."""
        s = _step_series(60, 60, val_before=0.0, val_after=10.0, noise_sd=0.005)
        breaks = bai_perron_breaks(s, max_breaks=5, min_segment=24)
        assert len(breaks) >= 1
        # Break index should be near 60 (within ±6 months of true break)
        assert abs(breaks[0]["index"] - 60) <= 6

    @pytest.mark.slow
    def test_two_clear_breaks_detected(self):
        """Two-break series: both breaks should be found."""
        s = _two_step_series(n_seg=60)
        breaks = bai_perron_breaks(s, max_breaks=5, min_segment=24)
        assert len(breaks) == 2

    @pytest.mark.slow
    def test_two_break_indices_near_true_breaks(self):
        """True breaks at 60 and 120; detected within ±8 months each."""
        s = _two_step_series(n_seg=60)
        breaks = bai_perron_breaks(s, max_breaks=5, min_segment=24)
        assert len(breaks) == 2
        idxs = sorted(b["index"] for b in breaks)
        assert abs(idxs[0] - 60) <= 8
        assert abs(idxs[1] - 120) <= 8

    def test_max_breaks_limits_output(self):
        """max_breaks=1 can return at most 1 break."""
        s = _two_step_series(n_seg=60)
        breaks = bai_perron_breaks(s, max_breaks=1, min_segment=24)
        assert len(breaks) <= 1

    def test_no_break_series_can_return_empty(self):
        """White-noise series with no structural break typically returns [] or few breaks."""
        rng = np.random.default_rng(7)
        s = pd.Series(
            rng.standard_normal(100),
            index=pd.date_range("2000-01-01", periods=100, freq="MS"),
        )
        breaks = bai_perron_breaks(s, max_breaks=5, min_segment=24)
        # We only assert it's a list of BreakResult dicts, not that it's empty
        # (sequential F may occasionally spuriously detect a break in noise).
        assert isinstance(breaks, list)

    def test_ci_width_matches_min_segment_div_4(self):
        """half_ci = min_segment // 4; the CI span in index terms is 2*half_ci at most."""
        min_seg = 24
        s = _step_series(60, 60, noise_sd=0.005)
        breaks = bai_perron_breaks(s, max_breaks=5, min_segment=min_seg)
        half_ci = max(min_seg // 4, 1)
        s_clean = s.dropna().sort_index()
        for b in breaks:
            i = b["index"]
            lo_idx = s_clean.index.get_loc(b["ci_lo"])
            hi_idx = s_clean.index.get_loc(b["ci_hi"])
            assert hi_idx - i <= half_ci
            assert i - lo_idx <= half_ci

    def test_break_dates_come_from_series_index(self):
        """All returned dates (date, ci_lo, ci_hi) must be in the original series index."""
        s = _step_series(60, 60, noise_sd=0.005)
        breaks = bai_perron_breaks(s, max_breaks=5, min_segment=24)
        idx_set = set(s.dropna().sort_index().index)
        for b in breaks:
            assert b["date"] in idx_set, f"date {b['date']} not in series index"
            assert b["ci_lo"] in idx_set
            assert b["ci_hi"] in idx_set


# ===========================================================================
# state_indicator_smooth
# ===========================================================================

class TestStateIndicatorSmoothOutputContract:
    def test_returns_series(self):
        s = _monotone_series(50)
        F = state_indicator_smooth(s)
        assert isinstance(F, pd.Series)

    def test_name_is_state(self):
        s = _monotone_series(50)
        F = state_indicator_smooth(s)
        assert F.name == "state"

    def test_preserves_index(self):
        s = _monotone_series(50)
        F = state_indicator_smooth(s)
        pd.testing.assert_index_equal(F.index, s.index)

    def test_preserves_length(self):
        s = _monotone_series(50)
        F = state_indicator_smooth(s)
        assert len(F) == len(s)

    def test_dtype_is_float(self):
        s = _monotone_series(50)
        F = state_indicator_smooth(s)
        assert F.dtype.kind == "f"


class TestStateIndicatorSmoothBounds:
    def test_non_nan_values_in_open_unit_interval(self):
        """F ∈ (0, 1) for all non-NaN values."""
        rng = np.random.default_rng(0)
        s = pd.Series(
            rng.standard_normal(200),
            index=pd.date_range("2000-01-01", periods=200, freq="MS"),
        )
        F = state_indicator_smooth(s)
        clean = F.dropna()
        assert (clean > 0).all(), "F has values <= 0"
        assert (clean < 1).all(), "F has values >= 1"

    def test_lag_creates_leading_nans(self):
        """With lag=k, the first k observations should be NaN."""
        s = _monotone_series(100)
        lag = 12
        F = state_indicator_smooth(s, lag=lag)
        assert F.iloc[:lag].isna().all()

    def test_lag_0_no_leading_nans(self):
        """With lag=0, there are no NaN values (rank(pct=True) is always defined)."""
        s = _monotone_series(50)
        F = state_indicator_smooth(s, lag=0)
        assert F.notna().all()

    def test_lag_1_exactly_one_leading_nan(self):
        s = _monotone_series(50)
        F = state_indicator_smooth(s, lag=1)
        assert pd.isna(F.iloc[0])
        assert F.iloc[1:].notna().all()


class TestStateIndicatorSmoothKnownValue:
    """Known-value tests derived from the logistic formula."""

    def test_f_equals_half_at_threshold_quantile(self):
        """When z == threshold_quantile, F = 1/(1+exp(0)) = 0.5 exactly."""
        # Monotone series: each value's percentile rank = (i+1)/n
        n = 100
        s = _monotone_series(n)
        for thr in (0.25, 0.5, 0.75):
            F = state_indicator_smooth(s, lag=0, gamma=1.5, threshold_quantile=thr)
            z = s.rank(pct=True)
            # Find the row where z is exactly thr
            diff = (z - thr).abs()
            closest_idx = diff.idxmin()
            if abs(z.loc[closest_idx] - thr) < 1e-9:
                assert abs(F.loc[closest_idx] - 0.5) < 1e-12, (
                    f"F != 0.5 at z=threshold_quantile={thr}: {F.loc[closest_idx]}"
                )

    def test_known_logistic_value(self):
        """Cross-check the formula: F(z) = 1/(1+exp(-gamma*(z-thr)/0.25))."""
        # Use lag=0 so z = rank(pct=True) of the series itself.
        n = 100
        s = _monotone_series(n)
        F = state_indicator_smooth(s, lag=0, gamma=2.0, threshold_quantile=0.5)
        z = s.rank(pct=True)
        gamma, thr = 2.0, 0.5
        F_expected = 1.0 / (1.0 + np.exp(-gamma * (z - thr) / 0.25))
        np.testing.assert_allclose(F.values, F_expected.values, rtol=1e-12)

    def test_lag_shifts_z_by_k_periods(self):
        """z should be the lagged percentile rank: value at t-lag is ranked at time t."""
        n = 50
        s = _monotone_series(n)
        lag = 3
        F = state_indicator_smooth(s, lag=lag, gamma=1.5, threshold_quantile=0.5)
        # Manually compute: z_t = rank(s.shift(lag), pct=True)
        z = s.shift(lag).rank(pct=True)
        gamma, thr = 1.5, 0.5
        F_expected = 1.0 / (1.0 + np.exp(-gamma * (z - thr) / 0.25))
        F_expected.name = "state"
        np.testing.assert_allclose(
            F.values, F_expected.values, rtol=1e-12, equal_nan=True
        )


class TestStateIndicatorSmoothMonotonicity:
    def test_f_is_monotone_in_z_for_monotone_series(self):
        """For a strictly increasing series with lag=0, rank is increasing,
        and since F is increasing in z, F must be non-decreasing."""
        s = _monotone_series(100)
        F = state_indicator_smooth(s, lag=0, gamma=1.5)
        diffs = F.diff().dropna()
        assert (diffs >= 0).all(), "F is not monotone non-decreasing"

    def test_higher_threshold_gives_lower_f_at_same_z(self):
        """Ceteris paribus, a higher threshold_quantile shifts the curve right,
        lowering F at any fixed z value."""
        rng = np.random.default_rng(99)
        s = pd.Series(
            rng.standard_normal(200),
            index=pd.date_range("2000-01-01", periods=200, freq="MS"),
        )
        F_lo = state_indicator_smooth(s, lag=0, threshold_quantile=0.2)
        F_hi = state_indicator_smooth(s, lag=0, threshold_quantile=0.8)
        # At every defined point, F_hi < F_lo (higher threshold -> lower F)
        assert (F_hi.dropna() < F_lo.dropna()).all()

    def test_higher_gamma_produces_sharper_transition(self):
        """A larger gamma increases the range (max-min) of F over a fixed z range."""
        s = _monotone_series(100)
        F_low_g = state_indicator_smooth(s, lag=0, gamma=0.5)
        F_high_g = state_indicator_smooth(s, lag=0, gamma=10.0)
        range_low = F_low_g.dropna().max() - F_low_g.dropna().min()
        range_high = F_high_g.dropna().max() - F_high_g.dropna().min()
        assert range_high > range_low

    def test_gamma_large_approximates_step_function(self):
        """With very large gamma, F should be close to 0 below threshold and
        close to 1 above threshold for a monotone series."""
        s = _monotone_series(200)
        F = state_indicator_smooth(s, lag=0, gamma=100.0, threshold_quantile=0.5)
        # Bottom quartile should have F << 0.5; top quartile should have F >> 0.5
        bottom = F.iloc[:50].dropna()
        top = F.iloc[150:].dropna()
        assert (bottom < 0.01).all(), "Bottom quartile F should be near 0 with large gamma"
        assert (top > 0.99).all(), "Top quartile F should be near 1 with large gamma"

    def test_default_threshold_0p5_pivots_around_median(self):
        """With the default threshold=0.5, the median rank (z=0.5) maps to F=0.5."""
        n = 100
        s = _monotone_series(n)
        F = state_indicator_smooth(s, lag=0)  # default threshold_quantile=0.5
        # The 50th percentile value (index 49) has z=0.5 -> F=0.5
        assert abs(F.iloc[49] - 0.5) < 1e-10


class TestStateIndicatorSmoothParameterDefaults:
    def test_default_lag_is_12(self):
        """Default lag=12: first 12 values should be NaN."""
        s = _monotone_series(100)
        F = state_indicator_smooth(s)
        assert F.iloc[:12].isna().all()
        assert F.iloc[12:].notna().all()

    def test_default_gamma_is_1p5(self):
        """Verify default gamma=1.5 matches explicit call."""
        s = _monotone_series(100)
        F_default = state_indicator_smooth(s)
        F_explicit = state_indicator_smooth(s, gamma=1.5)
        np.testing.assert_array_equal(F_default.values, F_explicit.values)

    def test_default_threshold_quantile_is_0p5(self):
        """Verify default threshold_quantile=0.5 matches explicit call."""
        s = _monotone_series(100)
        F_default = state_indicator_smooth(s)
        F_explicit = state_indicator_smooth(s, threshold_quantile=0.5)
        np.testing.assert_array_equal(F_default.values, F_explicit.values)
