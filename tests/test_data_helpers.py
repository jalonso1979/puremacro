"""Tests for puremacro._data_helpers.coerce_array_or_frame."""
import numpy as np
import pandas as pd
import pytest

from puremacro._data_helpers import coerce_array_or_frame, _CoercedData


def test_coerce_1d_array():
    arr = [1.0, 2.0, 3.0]
    res = coerce_array_or_frame(arr, required_dim=1, name="series_a")
    assert isinstance(res, _CoercedData)
    assert res.values.shape == (3,)
    assert res.names == ["series_a"]
    assert res.index is None
    assert res.freq is None
    assert len(res) == 3


def test_coerce_series_preserves_metadata():
    idx = pd.date_range("2020-01-01", periods=4, freq="QE")
    s = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx, name="gdp")
    res = coerce_array_or_frame(s, required_dim=1)
    assert res.values.shape == (4,)
    assert res.names == ["gdp"]
    assert res.index is not None
    assert len(res.index) == 4
    assert res.freq is not None


def test_coerce_dataframe_2d():
    idx = pd.date_range("2020-01-01", periods=3, freq="MS")
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [4.0, 5.0, 6.0]}, index=idx)
    res = coerce_array_or_frame(df, required_dim=2)
    assert res.values.shape == (3, 2)
    assert res.names == ["y", "x"]
    assert res.index is not None
    assert res.freq is not None


def test_coerce_shape_mismatch_raises():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="DataFrame has 2 columns, but required_dim=1"):
        coerce_array_or_frame(df, required_dim=1)

    with pytest.raises(ValueError, match="Expected 1-dimensional data"):
        coerce_array_or_frame(np.ones((2, 2, 2)), required_dim=1)


def test_coerce_none_raises():
    with pytest.raises(ValueError, match="cannot be None"):
        coerce_array_or_frame(None, name="test_arg")
