"""Focused tests for puremacro.regimes — regime-indicator helpers.

Test strategy
-------------
- known_value: break_indices with known breakpoints, verify exact regime
  assignments; dates_to_regimes with exact inclusive intervals, verify correct
  label assignment; regime_dummies with known indicator, verify one-hot
  arithmetic; smoothed_to_indicator with known probability matrix, verify
  argmax and threshold outputs.
- property: output shape/dtype, probability bounds, conservation (dummy rows
  sum to 1 when drop_first=False), monotonicity / consistency between
  smoothed_to_indicator modes.
- edge cases: empty breaks, single-element series, overlapping regime_dates
  intervals (last label wins), drop_first=True, n_regimes explicit override,
  target_regime threshold variations.
- contract: correct __all__ exports, Series vs ndarray input coercion.

No network I/O or external binaries: the module is pure NumPy/Pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.regimes import (
    breaks_to_regimes,
    dates_to_regimes,
    regime_dummies,
    smoothed_to_indicator,
)


# ===========================================================================
# 1. breaks_to_regimes
# ===========================================================================

class TestBreaksToRegimes:
    """Known-value and property tests for breaks_to_regimes."""

    # --- known-value --------------------------------------------------------

    def test_no_breaks_all_regime_zero(self):
        """With no break indices, every observation is regime 0."""
        result = breaks_to_regimes([], T=10)
        np.testing.assert_array_equal(result, np.zeros(10, dtype=int))

    def test_single_break_at_middle(self):
        """Break at k=5 with T=10 → obs 0-4 are regime 0, 5-9 are regime 1."""
        result = breaks_to_regimes([5], T=10)
        expected = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(result, expected)

    def test_single_break_at_1(self):
        """Break at k=1 → only obs 0 is regime 0, rest regime 1."""
        result = breaks_to_regimes([1], T=5)
        expected = np.array([0, 1, 1, 1, 1])
        np.testing.assert_array_equal(result, expected)

    def test_two_breaks_three_regimes(self):
        """Two breaks at k=3, k=7 with T=10 → 3 regimes correctly assigned."""
        result = breaks_to_regimes([3, 7], T=10)
        expected = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        np.testing.assert_array_equal(result, expected)

    def test_breaks_unsorted_input_sorted_internally(self):
        """Break list out of order is sorted before assignment."""
        # breaks=[7, 3] should behave identically to [3, 7]
        result_unsorted = breaks_to_regimes([7, 3], T=10)
        result_sorted = breaks_to_regimes([3, 7], T=10)
        np.testing.assert_array_equal(result_unsorted, result_sorted)

    def test_break_at_t_minus_1(self):
        """Break at T-1 means last obs is regime 1."""
        T = 8
        result = breaks_to_regimes([T - 1], T=T)
        assert result[-1] == 1
        assert all(result[:-1] == 0)

    def test_break_at_0_all_regime_1(self):
        """Break at index 0: all obs become regime 1."""
        result = breaks_to_regimes([0], T=6)
        np.testing.assert_array_equal(result, np.ones(6, dtype=int))

    # --- property -----------------------------------------------------------

    def test_output_shape_matches_T(self):
        """Output array has shape (T,)."""
        for T in [1, 5, 20, 100]:
            result = breaks_to_regimes([], T=T)
            assert result.shape == (T,)

    def test_output_dtype_is_int(self):
        """Output dtype is integer."""
        result = breaks_to_regimes([3], T=10)
        assert np.issubdtype(result.dtype, np.integer)

    def test_regimes_are_nonnegative(self):
        """All regime labels are non-negative integers."""
        result = breaks_to_regimes([2, 5, 8], T=12)
        assert np.all(result >= 0)

    def test_regime_labels_contiguous_from_zero(self):
        """With K breaks, regime labels run from 0 to K (inclusive)."""
        breaks = [3, 6, 9]
        T = 12
        result = breaks_to_regimes(breaks, T=T)
        expected_labels = set(range(len(breaks) + 1))
        assert set(np.unique(result)) == expected_labels

    def test_single_T_no_breaks(self):
        """T=1 with no breaks returns a single-element array."""
        result = breaks_to_regimes([], T=1)
        assert result.shape == (1,)
        assert result[0] == 0

    def test_float_break_indices_converted(self):
        """Float break indices are converted to int (docstring contract)."""
        result = breaks_to_regimes([3.0, 7.0], T=10)
        expected = breaks_to_regimes([3, 7], T=10)
        np.testing.assert_array_equal(result, expected)


# ===========================================================================
# 2. dates_to_regimes
# ===========================================================================

class TestDatesToRegimes:
    """Known-value and property tests for dates_to_regimes."""

    def _make_monthly_index(self, start: str, periods: int) -> pd.DatetimeIndex:
        return pd.date_range(start, periods=periods, freq="MS")

    # --- known-value --------------------------------------------------------

    def test_empty_regime_dict_returns_all_default(self):
        """Empty regime_dates → every date gets default_label."""
        idx = self._make_monthly_index("2000-01-01", 24)
        result = dates_to_regimes(idx, {}, default_label=0)
        assert (result == 0).all()

    def test_single_interval_exact_boundary_inclusive(self):
        """Start and end dates of the interval are included (inclusive)."""
        idx = pd.DatetimeIndex(["2008-01-01", "2008-06-01", "2008-12-01", "2009-01-01"])
        regime_dates = {1: [("2008-01-01", "2008-12-01")]}
        result = dates_to_regimes(idx, regime_dates)
        # 2008-01-01 and 2008-12-01 both inside → label 1
        assert result["2008-01-01"] == 1
        assert result["2008-12-01"] == 1
        # 2009-01-01 outside → default 0
        assert result["2009-01-01"] == 0

    def test_two_intervals_non_overlapping(self):
        """Two non-overlapping intervals for the same label are both covered."""
        idx = pd.DatetimeIndex([
            "2008-01-01", "2009-01-01",   # inside interval 1
            "2010-01-01",                  # outside both
            "2020-02-01", "2020-04-01",   # inside interval 2
        ])
        regime_dates = {1: [("2008-01-01", "2009-06-30"), ("2020-02-01", "2020-04-30")]}
        result = dates_to_regimes(idx, regime_dates, default_label=0)
        assert result["2008-01-01"] == 1
        assert result["2009-01-01"] == 1
        assert result["2010-01-01"] == 0
        assert result["2020-02-01"] == 1
        assert result["2020-04-01"] == 1

    def test_dates_outside_all_intervals_get_default_label(self):
        """Dates that fall in no interval receive default_label."""
        idx = pd.DatetimeIndex(["2005-01-01", "2012-01-01"])
        regime_dates = {1: [("2008-01-01", "2009-06-30")]}
        result = dates_to_regimes(idx, regime_dates, default_label=99)
        assert result["2005-01-01"] == 99
        assert result["2012-01-01"] == 99

    def test_multiple_labels_distinct_intervals(self):
        """Multiple label keys map to distinct date ranges."""
        idx = pd.DatetimeIndex(["2008-01-01", "2015-01-01", "2020-01-01"])
        regime_dates = {
            1: [("2008-01-01", "2009-06-30")],
            2: [("2013-01-01", "2019-12-31")],
            3: [("2020-01-01", "2021-12-31")],
        }
        result = dates_to_regimes(idx, regime_dates, default_label=0)
        assert result["2008-01-01"] == 1
        assert result["2015-01-01"] == 2
        assert result["2020-01-01"] == 3

    def test_overlapping_intervals_last_label_wins(self):
        """When two intervals overlap, the label assigned last (dict iteration order)
        wins — the function overwrites with subsequent labels."""
        idx = pd.DatetimeIndex(["2008-06-01"])
        # label 1 assigned first, then label 2 overwrites the same date
        regime_dates = {
            1: [("2008-01-01", "2009-12-31")],
            2: [("2008-01-01", "2009-12-31")],
        }
        result = dates_to_regimes(idx, regime_dates, default_label=0)
        assert result.iloc[0] == 2

    def test_non_default_default_label(self):
        """default_label != 0 is used for uncovered dates."""
        idx = pd.DatetimeIndex(["1990-01-01", "2025-01-01"])
        result = dates_to_regimes(idx, {}, default_label=-1)
        assert (result == -1).all()

    # --- property -----------------------------------------------------------

    def test_output_is_series(self):
        """Output is a pd.Series."""
        idx = self._make_monthly_index("2000-01-01", 12)
        result = dates_to_regimes(idx, {}, default_label=0)
        assert isinstance(result, pd.Series)

    def test_output_index_matches_input(self):
        """Output Series index equals the input DatetimeIndex."""
        idx = self._make_monthly_index("2000-01-01", 24)
        result = dates_to_regimes(idx, {}, default_label=0)
        assert result.index.equals(idx)

    def test_output_dtype_is_int(self):
        """Output dtype is integer."""
        idx = self._make_monthly_index("2000-01-01", 12)
        result = dates_to_regimes(idx, {}, default_label=0)
        assert np.issubdtype(result.dtype, np.integer)

    def test_output_name_is_regime(self):
        """Output Series has name='regime'."""
        idx = self._make_monthly_index("2000-01-01", 6)
        result = dates_to_regimes(idx, {}, default_label=0)
        assert result.name == "regime"

    def test_output_length_matches_index_length(self):
        """Output length equals input DatetimeIndex length."""
        for n in [1, 10, 100]:
            idx = self._make_monthly_index("2000-01-01", n)
            result = dates_to_regimes(idx, {}, default_label=0)
            assert len(result) == n

    def test_string_timestamps_accepted(self):
        """Start/end strings are accepted and converted via pd.Timestamp."""
        idx = pd.DatetimeIndex(["2008-03-01"])
        # Passes string start/end; should not raise
        result = dates_to_regimes(idx, {1: [("2008-01-01", "2008-12-31")]})
        assert result.iloc[0] == 1


# ===========================================================================
# 3. regime_dummies
# ===========================================================================

class TestRegimeDummies:
    """Known-value and property tests for regime_dummies."""

    # --- known-value --------------------------------------------------------

    def test_two_regime_indicator_ndarray_default(self):
        """2-regime indicator, drop_first=False → columns regime_0 and regime_1."""
        indicator = np.array([0, 0, 1, 1, 0, 1])
        dummies = regime_dummies(indicator)
        assert list(dummies.columns) == ["regime_0", "regime_1"]
        np.testing.assert_array_equal(dummies["regime_0"].values, [1, 1, 0, 0, 1, 0])
        np.testing.assert_array_equal(dummies["regime_1"].values, [0, 0, 1, 1, 0, 1])

    def test_two_regime_indicator_drop_first(self):
        """drop_first=True drops regime_0, keeps only regime_1."""
        indicator = np.array([0, 0, 1, 1])
        dummies = regime_dummies(indicator, drop_first=True)
        assert list(dummies.columns) == ["regime_1"]
        np.testing.assert_array_equal(dummies["regime_1"].values, [0, 0, 1, 1])

    def test_three_regime_all_columns(self):
        """3-regime indicator → columns regime_0, regime_1, regime_2."""
        indicator = np.array([0, 1, 2, 0, 1, 2])
        dummies = regime_dummies(indicator)
        assert list(dummies.columns) == ["regime_0", "regime_1", "regime_2"]
        # Each column is an exact one-hot
        for k in range(3):
            np.testing.assert_array_equal(
                dummies[f"regime_{k}"].values,
                (indicator == k).astype(float),
            )

    def test_three_regime_drop_first(self):
        """3-regime, drop_first=True → columns regime_1, regime_2 only."""
        indicator = np.array([0, 1, 2, 0, 1, 2])
        dummies = regime_dummies(indicator, drop_first=True)
        assert list(dummies.columns) == ["regime_1", "regime_2"]
        assert "regime_0" not in dummies.columns

    def test_explicit_n_regimes_pads_columns(self):
        """n_regimes=4 on a 2-regime indicator creates 4 columns (regime_0..3)."""
        indicator = np.array([0, 1, 0, 1])
        dummies = regime_dummies(indicator, n_regimes=4)
        assert list(dummies.columns) == ["regime_0", "regime_1", "regime_2", "regime_3"]
        # regime_2 and regime_3 are all zeros (no observations)
        np.testing.assert_array_equal(dummies["regime_2"].values, [0, 0, 0, 0])
        np.testing.assert_array_equal(dummies["regime_3"].values, [0, 0, 0, 0])

    def test_single_regime_zero(self):
        """All-zero indicator → one-column dummy when drop_first=False."""
        indicator = np.array([0, 0, 0, 0])
        dummies = regime_dummies(indicator)
        assert list(dummies.columns) == ["regime_0"]
        np.testing.assert_array_equal(dummies["regime_0"].values, [1, 1, 1, 1])

    # --- property -----------------------------------------------------------

    def test_rows_sum_to_one_no_drop(self):
        """Without drop_first, every row sums to exactly 1 (true partition of unity)."""
        indicator = np.array([0, 1, 2, 0, 1, 2, 0])
        dummies = regime_dummies(indicator)
        row_sums = dummies.sum(axis=1).values
        np.testing.assert_allclose(row_sums, 1.0)

    def test_values_are_zero_or_one(self):
        """All dummy values are exactly 0.0 or 1.0."""
        indicator = np.array([0, 1, 2, 0])
        dummies = regime_dummies(indicator)
        vals = dummies.values
        assert np.all((vals == 0.0) | (vals == 1.0))

    def test_output_dtype_float(self):
        """Output values are float (as built: (arr == k).astype(float))."""
        indicator = np.array([0, 1, 0])
        dummies = regime_dummies(indicator)
        assert np.issubdtype(dummies.values.dtype, np.floating)

    def test_output_is_dataframe(self):
        """Output is always a pd.DataFrame."""
        result = regime_dummies(np.array([0, 1]))
        assert isinstance(result, pd.DataFrame)

    def test_series_input_preserves_index(self):
        """When input is a pd.Series, output DataFrame has the same index."""
        idx = pd.Index([10, 20, 30, 40])
        indicator = pd.Series([0, 1, 0, 1], index=idx)
        dummies = regime_dummies(indicator)
        assert dummies.index.equals(idx)

    def test_ndarray_input_has_range_index(self):
        """When input is an ndarray, output has a default RangeIndex."""
        indicator = np.array([0, 1, 0])
        dummies = regime_dummies(indicator)
        pd.testing.assert_index_equal(dummies.index, pd.RangeIndex(3))

    def test_shape_no_drop(self):
        """Output shape is (T, n_regimes)."""
        indicator = np.array([0, 1, 2, 0, 1])
        dummies = regime_dummies(indicator)
        assert dummies.shape == (5, 3)

    def test_shape_drop_first(self):
        """With drop_first=True, n_cols == n_regimes - 1."""
        indicator = np.array([0, 1, 2, 0, 1])
        dummies = regime_dummies(indicator, drop_first=True)
        assert dummies.shape == (5, 2)

    def test_n_regimes_auto_from_max(self):
        """n_regimes inferred as max(indicator)+1 when not supplied."""
        indicator = np.array([0, 1, 3])  # max = 3 → 4 regimes
        dummies = regime_dummies(indicator)
        assert dummies.shape[1] == 4

    def test_column_names_follow_convention(self):
        """Column names follow the 'regime_{k}' convention."""
        indicator = np.array([0, 1, 2])
        dummies = regime_dummies(indicator)
        for k, col in enumerate(dummies.columns):
            assert col == f"regime_{k}"


# ===========================================================================
# 4. smoothed_to_indicator
# ===========================================================================

class TestSmoothedToIndicator:
    """Known-value and property tests for smoothed_to_indicator."""

    # --- known-value --------------------------------------------------------

    def test_argmax_two_regime_clear(self):
        """Clear 2-regime probs: argmax picks the column with higher probability."""
        probs = np.array([
            [0.9, 0.1],   # regime 0
            [0.2, 0.8],   # regime 1
            [0.6, 0.4],   # regime 0
        ])
        result = smoothed_to_indicator(probs)
        np.testing.assert_array_equal(result, [0, 1, 0])

    def test_argmax_three_regime_clear(self):
        """3-regime probs: argmax identifies the dominant regime at each step."""
        probs = np.array([
            [0.1, 0.8, 0.1],  # regime 1
            [0.0, 0.0, 1.0],  # regime 2
            [0.7, 0.2, 0.1],  # regime 0
        ])
        result = smoothed_to_indicator(probs)
        np.testing.assert_array_equal(result, [1, 2, 0])

    def test_target_regime_threshold_default(self):
        """target_regime=1, threshold=0.5: indicator is 1 when P(regime 1) >= 0.5."""
        probs = np.array([
            [0.8, 0.2],   # P(1)=0.2 < 0.5 → 0
            [0.4, 0.6],   # P(1)=0.6 >= 0.5 → 1
            [0.49, 0.51], # P(1)=0.51 >= 0.5 → 1
            [0.55, 0.45], # P(1)=0.45 < 0.5 → 0
        ])
        result = smoothed_to_indicator(probs, threshold=0.5, target_regime=1)
        np.testing.assert_array_equal(result, [0, 1, 1, 0])

    def test_target_regime_threshold_exact_boundary(self):
        """Exactly at threshold → classified as 'in regime' (>= threshold)."""
        probs = np.array([[0.5, 0.5]])
        result = smoothed_to_indicator(probs, threshold=0.5, target_regime=0)
        assert result[0] == 1  # 0.5 >= 0.5

    def test_target_regime_high_threshold(self):
        """threshold=0.8 requires > 80% prob to assign label 1."""
        probs = np.array([
            [0.1, 0.9],   # 0.9 >= 0.8 → 1
            [0.3, 0.7],   # 0.7 < 0.8 → 0
        ])
        result = smoothed_to_indicator(probs, threshold=0.8, target_regime=1)
        np.testing.assert_array_equal(result, [1, 0])

    def test_target_regime_zero_column(self):
        """target_regime=0 checks the first column for threshold."""
        probs = np.array([
            [0.9, 0.1],   # P(0)=0.9 >= 0.5 → 1
            [0.3, 0.7],   # P(0)=0.3 < 0.5 → 0
        ])
        result = smoothed_to_indicator(probs, threshold=0.5, target_regime=0)
        np.testing.assert_array_equal(result, [1, 0])

    def test_target_regime_2_in_three_regime(self):
        """target_regime=2 with 3-regime probs works correctly."""
        probs = np.array([
            [0.1, 0.1, 0.8],  # P(2)=0.8 >= 0.5 → 1
            [0.4, 0.4, 0.2],  # P(2)=0.2 < 0.5 → 0
        ])
        result = smoothed_to_indicator(probs, threshold=0.5, target_regime=2)
        np.testing.assert_array_equal(result, [1, 0])

    def test_all_regime_one_argmax(self):
        """If all rows favor regime 1, argmax output is all ones."""
        probs = np.array([
            [0.2, 0.8],
            [0.1, 0.9],
            [0.3, 0.7],
        ])
        result = smoothed_to_indicator(probs)
        np.testing.assert_array_equal(result, [1, 1, 1])

    # --- property -----------------------------------------------------------

    def test_output_shape_argmax_mode(self):
        """argmax mode output shape is (T,)."""
        T = 20
        probs = np.random.default_rng(0).dirichlet([1, 1], size=T)
        result = smoothed_to_indicator(probs)
        assert result.shape == (T,)

    def test_output_shape_target_regime_mode(self):
        """target_regime mode output shape is (T,)."""
        T = 15
        probs = np.random.default_rng(1).dirichlet([1, 1], size=T)
        result = smoothed_to_indicator(probs, target_regime=0)
        assert result.shape == (T,)

    def test_output_values_argmax_are_valid_regime_indices(self):
        """Argmax output values are valid regime indices in {0, 1, ..., K-1}."""
        K = 3
        T = 30
        probs = np.random.default_rng(2).dirichlet(np.ones(K), size=T)
        result = smoothed_to_indicator(probs)
        assert np.all(result >= 0)
        assert np.all(result < K)

    def test_output_values_target_regime_are_binary(self):
        """target_regime mode output values are only 0 or 1."""
        T = 25
        probs = np.random.default_rng(3).dirichlet([1, 1], size=T)
        result = smoothed_to_indicator(probs, target_regime=1)
        assert set(np.unique(result)).issubset({0, 1})

    def test_output_dtype_argmax_is_int(self):
        """Argmax mode returns integer-dtype array."""
        probs = np.array([[0.7, 0.3], [0.2, 0.8]])
        result = smoothed_to_indicator(probs)
        assert np.issubdtype(result.dtype, np.integer)

    def test_output_dtype_target_regime_is_int(self):
        """target_regime mode returns integer-dtype array."""
        probs = np.array([[0.7, 0.3], [0.2, 0.8]])
        result = smoothed_to_indicator(probs, target_regime=0)
        assert np.issubdtype(result.dtype, np.integer)

    def test_lower_threshold_more_ones(self):
        """Lower threshold → more observations classified as 'in target regime'."""
        rng = np.random.default_rng(42)
        probs = rng.dirichlet([1, 1], size=50)
        result_high = smoothed_to_indicator(probs, threshold=0.8, target_regime=1)
        result_low = smoothed_to_indicator(probs, threshold=0.2, target_regime=1)
        assert result_low.sum() >= result_high.sum()

    def test_ndarray_input_accepted(self):
        """Numpy ndarray input works without error."""
        probs = np.array([[0.6, 0.4], [0.3, 0.7]])
        result = smoothed_to_indicator(probs)
        assert result.shape == (2,)

    def test_list_input_accepted(self):
        """Python list input is coerced to ndarray without error."""
        probs = [[0.9, 0.1], [0.1, 0.9]]
        result = smoothed_to_indicator(probs)
        np.testing.assert_array_equal(result, [0, 1])

    def test_single_observation(self):
        """T=1 works correctly in both argmax and target_regime modes."""
        probs = np.array([[0.3, 0.7]])
        assert smoothed_to_indicator(probs)[0] == 1
        assert smoothed_to_indicator(probs, target_regime=1)[0] == 1

    def test_consistent_with_argmax_at_threshold_0(self):
        """target_regime=k at threshold=0 assigns 1 everywhere (all probs >= 0)."""
        probs = np.random.default_rng(7).dirichlet([1, 1, 1], size=10)
        result = smoothed_to_indicator(probs, threshold=0.0, target_regime=2)
        np.testing.assert_array_equal(result, np.ones(10, dtype=int))

    def test_consistent_with_argmax_at_threshold_1(self):
        """threshold=1.0: only rows with P(target)=1 exactly get label 1."""
        probs = np.array([
            [0.0, 0.0, 1.0],   # P(2)=1 exactly → 1
            [0.5, 0.2, 0.3],   # P(2)=0.3 < 1 → 0
        ])
        result = smoothed_to_indicator(probs, threshold=1.0, target_regime=2)
        np.testing.assert_array_equal(result, [1, 0])


# ===========================================================================
# 5. Public API / __all__
# ===========================================================================

class TestPublicAPI:
    def test_all_exports(self):
        """__all__ contains exactly the four documented functions."""
        import puremacro.regimes as mod
        assert set(mod.__all__) == {
            "breaks_to_regimes",
            "dates_to_regimes",
            "regime_dummies",
            "smoothed_to_indicator",
        }

    def test_all_callables_importable(self):
        """Every name in __all__ is callable."""
        import puremacro.regimes as mod
        for name in mod.__all__:
            assert callable(getattr(mod, name))

    def test_breaks_to_regimes_importable(self):
        from puremacro.regimes import breaks_to_regimes as f
        assert callable(f)

    def test_dates_to_regimes_importable(self):
        from puremacro.regimes import dates_to_regimes as f
        assert callable(f)

    def test_regime_dummies_importable(self):
        from puremacro.regimes import regime_dummies as f
        assert callable(f)

    def test_smoothed_to_indicator_importable(self):
        from puremacro.regimes import smoothed_to_indicator as f
        assert callable(f)


# ===========================================================================
# 6. Integration: pipeline breaks → dummies → smoothed_to_indicator
# ===========================================================================

class TestPipelineIntegration:
    """End-to-end sanity checks combining multiple helpers."""

    def test_breaks_to_dummies_partition_unity(self):
        """breaks_to_regimes piped into regime_dummies: rows sum to 1."""
        indicator = breaks_to_regimes([10, 20], T=30)
        dummies = regime_dummies(indicator)
        row_sums = dummies.sum(axis=1).values
        np.testing.assert_allclose(row_sums, 1.0)

    def test_dates_to_dummies_partition_unity(self):
        """dates_to_regimes piped into regime_dummies: rows sum to 1."""
        idx = pd.date_range("2000-01-01", periods=48, freq="MS")
        regime_dates = {
            1: [("2008-01-01", "2009-06-30")],
            2: [("2013-01-01", "2019-12-31")],
        }
        indicator = dates_to_regimes(idx, regime_dates, default_label=0)
        dummies = regime_dummies(indicator.values, n_regimes=3)
        row_sums = dummies.sum(axis=1).values
        np.testing.assert_allclose(row_sums, 1.0)

    def test_breaks_to_regimes_then_smoothed_indicator_consistency(self):
        """Smoothed probs dominated by one regime → argmax matches breaks_to_regimes."""
        T = 10
        indicator = breaks_to_regimes([5], T=T)  # 0..4 → 0, 5..9 → 1
        # Build fake smoothed probs that perfectly match the hard indicator
        probs = np.zeros((T, 2))
        for t in range(T):
            probs[t, indicator[t]] = 1.0
            probs[t, 1 - indicator[t]] = 0.0
        recovered = smoothed_to_indicator(probs)
        np.testing.assert_array_equal(recovered, indicator)
