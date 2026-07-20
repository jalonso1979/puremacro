"""β.1 5-tuple schema: optional magnitude as the 5th slot.

Old 4-tuple connectors keep working unchanged. New connectors yielding
5-tuples (date, text, url, meta, magnitude) validate cleanly. Records
with magnitude=None fall back to 1.0 at aggregation time (tested in β-2).
"""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative.sources._schema import (
    validate_records, SourceRecordValidationError,
)


def test_4tuple_still_valid_and_emits_5tuple():
    rec = (pd.Timestamp("2024-01-01"), "text", "https://x", {"k": "v"})
    out = list(validate_records([rec]))
    assert len(out) == 1
    assert len(out[0]) == 5
    assert out[0][4] is None


def test_5tuple_with_magnitude():
    rec = (pd.Timestamp("2024-01-01"), "layoff", "https://x",
           {"company": "Acme"}, 250.0)
    out = list(validate_records([rec]))
    assert out[0][4] == 250.0


def test_5tuple_rejects_non_numeric_magnitude():
    rec = (pd.Timestamp("2024-01-01"), "x", "https://x", {}, "not a number")
    with pytest.raises(SourceRecordValidationError, match="magnitude"):
        list(validate_records([rec]))


def test_negative_magnitude_rejected():
    rec = (pd.Timestamp("2024-01-01"), "x", "https://x", {}, -5.0)
    with pytest.raises(SourceRecordValidationError, match="magnitude"):
        list(validate_records([rec]))


def test_bool_magnitude_rejected():
    # bool is a subclass of int in Python; we must explicitly reject it.
    rec = (pd.Timestamp("2024-01-01"), "x", "https://x", {}, True)
    with pytest.raises(SourceRecordValidationError, match="magnitude"):
        list(validate_records([rec]))


def test_int_magnitude_coerced_to_float():
    rec = (pd.Timestamp("2024-01-01"), "x", "https://x", {}, 42)
    out = list(validate_records([rec]))
    assert out[0][4] == 42.0
    assert isinstance(out[0][4], float)


def test_3tuple_rejected():
    rec = (pd.Timestamp("2024-01-01"), "x", "https://x")
    with pytest.raises(SourceRecordValidationError, match="3-tuple|expected"):
        list(validate_records([rec]))


def test_index_to_quarterly_uses_magnitude_when_present():
    """β.1: index_to_quarterly should weight per-doc kernel scores by
    `magnitude` when supplied via 5-tuple records + caller passes
    `weight_by='magnitude'`."""
    from puremacro.narrative.aggregate import index_to_quarterly

    records = [
        (pd.Timestamp("2024-01-15"), "x1", "u1", {"bank_code": "X"}, 100.0),
        (pd.Timestamp("2024-02-15"), "x2", "u2", {"bank_code": "X"}, 300.0),
    ]
    def kernel(recs):
        for r in recs:
            yield (r[0], 1.0)  # per-doc score = 1; weight via magnitude downstream

    out = index_to_quarterly(
        records,
        kernel=kernel,
        country="USA", language="en", name="ev",
        method="keyword_count", corpus="test", normalization="raw",
        agg="sum_weighted",
        weight_by="magnitude",
    )
    # 100 + 300 = 400 in 2024Q1.
    assert out.series.loc["2024-01-01"] == 400.0


def test_index_to_quarterly_fallback_to_one_when_magnitude_missing():
    """4-tuple records (no magnitude) should fall back to weight=1.0."""
    from puremacro.narrative.aggregate import index_to_quarterly
    records = [
        (pd.Timestamp("2024-01-15"), "x1", "u1", {"bank_code": "X"}),  # 4-tuple
    ]
    def kernel(recs):
        for r in recs:
            yield (r[0], 1.0)
    out = index_to_quarterly(
        records, kernel=kernel,
        country="USA", language="en", name="ev",
        method="keyword_count", corpus="test", normalization="raw",
        agg="sum_weighted", weight_by="magnitude",
    )
    assert out.series.loc["2024-01-01"] == 1.0


def test_index_to_quarterly_rejects_sum_weighted_without_weight_by():
    from puremacro.narrative.aggregate import index_to_quarterly
    records = [(pd.Timestamp("2024-01-15"), "x", "u", {"bank_code": "X"}, 10.0)]
    def kernel(recs):
        for r in recs:
            yield (r[0], 1.0)
    with pytest.raises(ValueError, match="weight_by"):
        index_to_quarterly(
            records, kernel=kernel,
            country="USA", language="en", name="ev",
            method="keyword_count", corpus="test", normalization="raw",
            agg="sum_weighted",
        )


def test_index_to_quarterly_existing_agg_modes_unchanged():
    """Sanity: existing agg='mean' path still works with 4-tuple records."""
    from puremacro.narrative.aggregate import index_to_quarterly
    records = [
        (pd.Timestamp("2024-01-15"), "x1", "u1", {"bank_code": "X"}),
        (pd.Timestamp("2024-01-20"), "x2", "u2", {"bank_code": "X"}),
    ]
    def kernel(recs):
        for r in recs:
            yield (r[0], 2.0)
    out = index_to_quarterly(
        records, kernel=kernel,
        country="USA", language="en", name="ev",
        method="keyword_count", corpus="test", normalization="raw",
        agg="mean",
    )
    assert out.series.loc["2024-01-01"] == 2.0
