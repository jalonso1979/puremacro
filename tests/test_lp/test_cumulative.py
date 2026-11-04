import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.lp._common import check_regular_index, cumulative_windows


def _q(vals, start="2000-01-01"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq="QS"), dtype=float)


def test_windows_h0_is_the_long_difference_bit_for_bit():
    y, x = _q([1., 2., 4., 7.]), _q([1., 0., 2., 1.])
    yw, xw = cumulative_windows(y, x, 0)
    assert yw.equals(y.shift(0) - y.shift(1)) and xw.equals(x)


def test_windows_h2_values_and_edges():
    y, x = _q([1., 2., 4., 7., 11.]), _q([1., 0., 2., 1., 3.])
    yw, xw = cumulative_windows(y, x, 2)
    assert yw.iloc[1] == (2 - 1) + (4 - 1) + (7 - 1) and xw.iloc[1] == 0 + 2 + 1
    assert np.isnan(yw.iloc[0])                                      # no y_{t-1}
    assert yw.iloc[-2:].isna().all() and xw.iloc[-2:].isna().all()    # window runs past the end


def test_interior_missing_value_drops_only_touching_rows():
    y, x = _q([1., 2., np.nan, 7., 11., 16.]), _q([1.] * 6)
    yw, _ = cumulative_windows(y, x, 1)
    assert yw.iloc[1:4].isna().all()                  # t = 1, 2, 3 touch y_2
    assert yw.iloc[4] == (11 - 7) + (16 - 7)


def test_regular_indexes_pass():
    check_regular_index(pd.date_range("1890-01-01", periods=12, freq="QS"), "f")
    check_regular_index(pd.period_range("1890Q1", periods=12, freq="Q"), "f")
    check_regular_index(pd.RangeIndex(12), "f")       # not a time index: not checked


def test_datetime_gap_is_named():
    idx = pd.date_range("1940-01-01", periods=12, freq="QS").delete(5)
    with pytest.raises(ValueError, match=r"lp_iv: cumulative=True .*1941-01-01.*1941-07-01.*NaN"):
        check_regular_index(idx, "lp_iv")


def test_period_gap_and_duplicates_raise():
    with pytest.raises(ValueError, match=r"1940Q3.*1941Q1"):
        check_regular_index(pd.period_range("1940Q1", periods=8, freq="Q").delete(3), "f")
    with pytest.raises(ValueError):
        check_regular_index(pd.DatetimeIndex(["2000-01-01", "2000-04-01", "2000-04-01", "2000-07-01"]), "f")


def test_tz_aware_index_is_measured_without_a_timezone_warning():
    regular = pd.date_range("1940-01-01", periods=12, freq="QS", tz="UTC")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_regular_index(regular, "lp_iv")
        with pytest.raises(ValueError, match=r"1941-01-01.*1941-07-01"):
            check_regular_index(regular.delete(5), "lp_iv")
