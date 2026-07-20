"""Tests for index_to_quarterly (continuous text-derived series → RiskIndex)."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative import RiskIndex, index_to_quarterly


def _record(date, score):
    """Synthetic per-document score record."""
    return (pd.Timestamp(date), score)


def _kernel_passthrough(records):
    """Trivial kernel: yields (date, score) unchanged."""
    return list(records)


def test_quarterly_mean_aggregation():
    records = [
        _record("2020-01-15", 100.0),
        _record("2020-02-15", 110.0),
        _record("2020-04-15",  90.0),
    ]
    ri = index_to_quarterly(
        records, kernel=_kernel_passthrough,
        country="USA", language="en",
        name="test_index", method="keyword_count",
        corpus="synthetic", normalization="raw",
        agg="mean",
    )
    assert isinstance(ri, RiskIndex)
    assert ri.country == "USA"
    assert ri.series.iloc[0] == pytest.approx(105.0)
    assert ri.series.iloc[1] == pytest.approx(90.0)


def test_quarterly_max_aggregation():
    records = [
        _record("2020-01-15", 100.0),
        _record("2020-02-15", 110.0),
    ]
    ri = index_to_quarterly(
        records, kernel=_kernel_passthrough,
        country="USA", language="en",
        name="test_index", method="keyword_count",
        corpus="synthetic", normalization="raw",
        agg="max",
    )
    assert ri.series.iloc[0] == pytest.approx(110.0)


def test_quarterly_dispersion_aggregation():
    records = [
        _record("2020-01-15", 100.0),
        _record("2020-02-15", 110.0),
        _record("2020-03-15",  90.0),
    ]
    ri = index_to_quarterly(
        records, kernel=_kernel_passthrough,
        country="USA", language="en",
        name="tone_dispersion", method="tone_dispersion",
        corpus="synthetic", normalization="raw",
        agg="dispersion",
    )
    assert ri.series.iloc[0] == pytest.approx(10.0)


def test_empty_records_raises():
    with pytest.raises(ValueError, match="no documents"):
        index_to_quarterly(
            iter([]), kernel=_kernel_passthrough,
            country="USA", language="en",
            name="x", method="keyword_count", corpus="empty",
            normalization="raw",
        )


def test_metadata_records_kernel_count():
    records = [_record("2020-01-15", 100.0), _record("2020-02-15", 110.0)]
    ri = index_to_quarterly(
        records, kernel=_kernel_passthrough,
        country="USA", language="en",
        name="x", method="keyword_count", corpus="synthetic",
        normalization="raw", agg="mean",
    )
    assert ri.metadata.get("n_docs") == 2


def test_index_to_quarterly_actually_applies_zscore_normalization():
    """Slice 2: normalization is no longer just metadata — the series is
    actually normalized."""
    from puremacro.narrative import RiskIndex, index_to_quarterly

    def kernel(records):
        return list(records)  # passthrough

    records = [
        (pd.Timestamp(d), v) for d, v in [
            ("2020-01-15", 90.0), ("2020-02-15", 100.0), ("2020-03-15", 110.0),
            ("2020-04-15", 95.0), ("2020-05-15", 105.0), ("2020-06-15", 100.0),
        ]
    ]
    ri = index_to_quarterly(
        records, kernel=kernel,
        country="USA", language="en",
        name="z_test", method="keyword_count",
        corpus="synthetic", normalization="zscore", agg="mean",
    )
    assert ri.series.dropna().mean() == pytest.approx(0.0, abs=1e-9)


def test_index_to_quarterly_threads_base_period_to_normalize():
    """Slice 3: base_period from metadata is now plumbed through to
    normalize_series."""
    from puremacro.narrative import index_to_quarterly

    def kernel(records):
        # Build values that will create a strong base/post-base separation:
        # base = first 4 obs (mean 100, std small); post = next 4 obs (200).
        return list(records)

    records = [
        (pd.Timestamp("2020-01-15"), 95.0),
        (pd.Timestamp("2020-04-15"), 100.0),
        (pd.Timestamp("2020-07-15"), 105.0),
        (pd.Timestamp("2020-10-15"), 100.0),
        (pd.Timestamp("2021-01-15"), 200.0),
        (pd.Timestamp("2021-04-15"), 200.0),
        (pd.Timestamp("2021-07-15"), 200.0),
        (pd.Timestamp("2021-10-15"), 200.0),
    ]
    ri = index_to_quarterly(
        records, kernel=kernel,
        country="USA", language="en",
        name="bp_test", method="keyword_count",
        corpus="synthetic", normalization="bbd_100",
        agg="mean",
        metadata={"base_period": ("2020-01-01", "2020-12-31")},
    )
    s = ri.series.dropna()
    base_mask = (s.index >= "2020-01-01") & (s.index <= "2020-12-31")
    # Base period mean should be ≈ 100 (the bbd_100 target).
    assert s[base_mask].mean() == pytest.approx(100.0, rel=0.1)
    # Post-base values should be far above 100 (since they were 28x std above base).
    assert (s[~base_mask] > 200).all()
