"""Tests for puremacro.instruments._compose."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument, compose


def _make(values, idx, name, freq="M"):
    return Instrument(
        series=pd.Series(values, index=idx, name=name),
        name=name,
        source="synthetic",
        category="literature",
        frequency=freq,
    )


def _idx(start, n, freq="MS"):
    return pd.date_range(start, periods=n, freq=freq)


# --------------------------------------------------------------------------
# Sum
# --------------------------------------------------------------------------
def test_compose_sum_two_aligned_instruments():
    idx = _idx("2000-01-01", 3)
    a = _make([1.0, 2.0, 3.0], idx, "a")
    b = _make([10.0, 20.0, 30.0], idx, "b")
    out = compose([a, b], op="sum")
    assert isinstance(out, Instrument)
    assert out.category == "composite"
    assert out.frequency == "M"
    assert list(out.series.values) == [11.0, 22.0, 33.0]


def test_compose_sum_propagates_nan_by_default():
    idx = _idx("2000-01-01", 3)
    a = _make([1.0, np.nan, 3.0], idx, "a")
    b = _make([10.0, 20.0, 30.0], idx, "b")
    out = compose([a, b], op="sum")
    assert pd.isna(out.series.iloc[1])


def test_compose_sum_skipna_true_ignores_nan():
    idx = _idx("2000-01-01", 3)
    a = _make([1.0, np.nan, 3.0], idx, "a")
    b = _make([10.0, 20.0, 30.0], idx, "b")
    out = compose([a, b], op="sum", skipna=True)
    assert out.series.iloc[1] == 20.0  # only b contributes


# --------------------------------------------------------------------------
# Mean
# --------------------------------------------------------------------------
def test_compose_mean_three_instruments():
    idx = _idx("2000-01-01", 2)
    a = _make([1.0, 2.0], idx, "a")
    b = _make([3.0, 4.0], idx, "b")
    c = _make([5.0, 6.0], idx, "c")
    out = compose([a, b, c], op="mean")
    assert list(out.series.values) == [3.0, 4.0]


# --------------------------------------------------------------------------
# Weighted mean
# --------------------------------------------------------------------------
def test_compose_weighted_mean_with_explicit_weights():
    idx = _idx("2000-01-01", 2)
    a = _make([1.0, 2.0], idx, "a")
    b = _make([3.0, 4.0], idx, "b")
    out = compose([a, b], op="weighted_mean", weights=[0.25, 0.75])
    # 0.25*1 + 0.75*3 = 2.5; 0.25*2 + 0.75*4 = 3.5
    assert out.series.iloc[0] == pytest.approx(2.5)
    assert out.series.iloc[1] == pytest.approx(3.5)


def test_compose_weighted_mean_requires_weights():
    idx = _idx("2000-01-01", 2)
    a = _make([1.0, 2.0], idx, "a")
    b = _make([3.0, 4.0], idx, "b")
    with pytest.raises(ValueError, match="weights"):
        compose([a, b], op="weighted_mean")


def test_compose_weighted_mean_wrong_length_weights():
    idx = _idx("2000-01-01", 2)
    a = _make([1.0, 2.0], idx, "a")
    b = _make([3.0, 4.0], idx, "b")
    with pytest.raises(ValueError, match="length"):
        compose([a, b], op="weighted_mean", weights=[0.5, 0.3, 0.2])


# --------------------------------------------------------------------------
# Concat
# --------------------------------------------------------------------------
def test_compose_concat_non_overlapping_union():
    idx_a = _idx("2000-01-01", 2)
    idx_b = _idx("2010-01-01", 2)
    a = _make([1.0, 2.0], idx_a, "a")
    b = _make([10.0, 20.0], idx_b, "b")
    out = compose([a, b], op="concat")
    assert len(out.series) == 4
    assert out.series.loc[pd.Timestamp("2000-01-01")] == 1.0
    assert out.series.loc[pd.Timestamp("2010-01-01")] == 10.0


def test_compose_concat_overlapping_later_wins():
    """When two instruments have a value at the same date, the LAST in
    the input list overwrites the earlier."""
    idx = _idx("2000-01-01", 3)
    a = _make([1.0, 2.0, 3.0], idx, "a")
    b = _make([100.0, 200.0, 300.0], idx, "b")
    out = compose([a, b], op="concat")
    # b is last, so its values win at every overlapping date
    assert list(out.series.values) == [100.0, 200.0, 300.0]


# --------------------------------------------------------------------------
# Frequency / alignment / scalar inputs
# --------------------------------------------------------------------------
def test_compose_mismatched_frequencies_raises():
    idx_m = _idx("2000-01-01", 3, freq="MS")
    idx_q = _idx("2000-01-01", 3, freq="QS")
    a = _make([1.0, 2.0, 3.0], idx_m, "a", freq="M")
    b = _make([1.0, 2.0, 3.0], idx_q, "b", freq="Q")
    with pytest.raises(ValueError, match="frequency"):
        compose([a, b], op="sum")


def test_compose_inner_alignment_default():
    """Default alignment is inner-join: drop dates not present in all."""
    idx_a = pd.date_range("2000-01-01", periods=4, freq="MS")
    idx_b = pd.date_range("2000-02-01", periods=4, freq="MS")
    a = _make([1.0, 2.0, 3.0, 4.0], idx_a, "a")
    b = _make([10.0, 20.0, 30.0, 40.0], idx_b, "b")
    out = compose([a, b], op="sum")
    # Common dates: 2000-02, 2000-03, 2000-04 (three months)
    assert len(out.series) == 3


def test_compose_outer_alignment():
    idx_a = pd.date_range("2000-01-01", periods=2, freq="MS")
    idx_b = pd.date_range("2000-02-01", periods=2, freq="MS")
    a = _make([1.0, 2.0], idx_a, "a")
    b = _make([10.0, 20.0], idx_b, "b")
    out = compose([a, b], op="sum", align="outer")
    # Union: 2000-01, 2000-02, 2000-03 (three dates)
    assert len(out.series) == 3
    # Outer-join introduces NaN where one input is missing; sum propagates
    assert pd.isna(out.series.loc[pd.Timestamp("2000-01-01")])
    assert pd.isna(out.series.loc[pd.Timestamp("2000-03-01")])
    assert out.series.loc[pd.Timestamp("2000-02-01")] == 12.0


def test_compose_empty_list_raises():
    with pytest.raises(ValueError, match="empty"):
        compose([], op="sum")


def test_compose_single_instrument_returns_copy():
    idx = _idx("2000-01-01", 3)
    a = _make([1.0, 2.0, 3.0], idx, "a")
    out = compose([a], op="sum")
    assert isinstance(out, Instrument)
    assert out.category == "composite"
    assert list(out.series.values) == [1.0, 2.0, 3.0]
    # Series is a copy (modifying out.series should not touch a.series; both
    # are pandas Series so we just verify they're different objects).
    assert out.series is not a.series


def test_compose_unknown_op_raises():
    idx = _idx("2000-01-01", 2)
    a = _make([1.0, 2.0], idx, "a")
    with pytest.raises(ValueError, match="op"):
        compose([a], op="not_a_real_op")


# --------------------------------------------------------------------------
# Result Instrument shape
# --------------------------------------------------------------------------
def test_compose_result_metadata_records_provenance():
    idx = _idx("2000-01-01", 2)
    a = _make([1.0, 2.0], idx, "a")
    b = _make([3.0, 4.0], idx, "b")
    out = compose([a, b], op="sum")
    assert out.metadata.get("source_instruments") == ["a", "b"]
    assert out.metadata.get("composition_op") == "sum"
    assert out.metadata.get("composition_weights") is None
    assert out.metadata.get("composition_align") == "inner"


def test_compose_result_uses_caller_supplied_name_and_source():
    idx = _idx("2000-01-01", 2)
    a = _make([1.0, 2.0], idx, "a")
    b = _make([3.0, 4.0], idx, "b")
    out = compose([a, b], op="sum", name="my_composite", source="user demo")
    assert out.name == "my_composite"
    assert out.source == "user demo"


def test_compose_result_auto_generates_name_when_none():
    idx = _idx("2000-01-01", 2)
    a = _make([1.0, 2.0], idx, "a")
    b = _make([3.0, 4.0], idx, "b")
    out = compose([a, b], op="sum")
    assert "compose" in out.name or "sum" in out.name
    assert "a" in out.source or "b" in out.source


# --------------------------------------------------------------------------
# Method form: Instrument.compose(*others, **kwargs)
# --------------------------------------------------------------------------
def test_method_compose_matches_function_form():
    idx = _idx("2000-01-01", 3)
    a = _make([1.0, 2.0, 3.0], idx, "a")
    b = _make([10.0, 20.0, 30.0], idx, "b")
    c = _make([100.0, 200.0, 300.0], idx, "c")
    func_result = compose([a, b, c], op="sum")
    method_result = a.compose(b, c, op="sum")
    assert list(func_result.series.values) == list(method_result.series.values)
    assert func_result.metadata == method_result.metadata


def test_method_compose_with_no_others_returns_single_instrument_copy():
    idx = _idx("2000-01-01", 2)
    a = _make([1.0, 2.0], idx, "a")
    out = a.compose(op="sum")
    assert isinstance(out, Instrument)
    assert out.category == "composite"
    assert list(out.series.values) == [1.0, 2.0]


def test_compose_result_series_name_matches_instrument_name():
    """Series.name must match Instrument.name in both single- and
    multi-instrument branches (regression for I1)."""
    idx = _idx("2000-01-01", 3)
    a = _make([1.0, 2.0, 3.0], idx, "original_a_name")
    b = _make([10.0, 20.0, 30.0], idx, "original_b_name")
    # Single-instrument branch
    out_single = compose([a], op="sum")
    assert out_single.series.name == out_single.name
    # Multi-instrument branch
    out_multi = compose([a, b], op="sum")
    assert out_multi.series.name == out_multi.name
    # With caller-supplied name
    out_named = compose([a, b], op="sum", name="user_chose_this")
    assert out_named.series.name == "user_chose_this"
    assert out_named.name == "user_chose_this"


def test_compose_concat_with_explicit_inner_align_still_uses_outer():
    """`align="inner"` is silently overridden to "outer" when op="concat"
    so non-overlapping inputs can be spliced. The metadata still records
    the caller's request."""
    idx_a = _idx("2000-01-01", 2)
    idx_b = _idx("2010-01-01", 2)
    a = _make([1.0, 2.0], idx_a, "a")
    b = _make([10.0, 20.0], idx_b, "b")
    # If align="inner" were honored literally, this would be empty.
    out = compose([a, b], op="concat", align="inner")
    assert len(out.series) == 4
    # Caller's intent is recorded in metadata.
    assert out.metadata["composition_align"] == "inner"


def test_compose_mean_skipna_true_ignores_nan():
    """mean with skipna=True ignores NaN per row."""
    idx = _idx("2000-01-01", 3)
    a = _make([1.0, np.nan, 3.0], idx, "a")
    b = _make([3.0, 4.0, 5.0], idx, "b")
    out = compose([a, b], op="mean", skipna=True)
    # row 0: mean(1, 3) = 2; row 1: mean(NaN, 4) skipna=True -> 4; row 2: mean(3, 5) = 4
    assert out.series.iloc[0] == 2.0
    assert out.series.iloc[1] == 4.0
    assert out.series.iloc[2] == 4.0
