"""Slice 1 of the signal contract — sparsity-report computation."""
from __future__ import annotations

import pandas as pd
import pytest


def _records(*items):
    """Build a list of 4-tuple records: (date_iso, text, url, metadata)."""
    return [(pd.Timestamp(d), txt, "http://test", {}) for d, txt in items]


def test_sparsity_report_counts_docs_per_quarter():
    from puremacro.narrative._signal_quality import compute_sparsity_report

    recs = _records(
        ("2020-01-15", "alpha beta gamma"),
        ("2020-02-10", "delta epsilon zeta eta"),
        ("2020-05-20", "single"),
    )
    rep = compute_sparsity_report(recs)
    # 2020Q1 -> 2 docs, 2020Q2 -> 1 doc.
    assert int(rep.n_docs_per_period.loc[pd.Period("2020Q1", "Q")]) == 2
    assert int(rep.n_docs_per_period.loc[pd.Period("2020Q2", "Q")]) == 1


def test_sparsity_report_computes_average_doc_length():
    from puremacro.narrative._signal_quality import compute_sparsity_report

    recs = _records(
        ("2020-01-15", "a b c"),       # 3 tokens
        ("2020-02-10", "a b c d e"),   # 5 tokens
    )
    rep = compute_sparsity_report(recs)
    # mean of {3, 5} in 2020Q1 = 4.0
    assert rep.avg_doc_length.loc[pd.Period("2020Q1", "Q")] == 4.0


def test_sparsity_report_reports_coverage_gaps_within_range():
    from puremacro.narrative._signal_quality import compute_sparsity_report

    recs = _records(
        ("2020-01-15", "alpha"),
        ("2020-10-15", "omega"),
    )
    rep = compute_sparsity_report(recs)
    # 2020Q2 and 2020Q3 should be gaps (no docs).
    gaps = set(rep.coverage_gaps)
    assert pd.Period("2020Q2", "Q") in gaps
    assert pd.Period("2020Q3", "Q") in gaps
    # 2020Q1 and 2020Q4 are populated → not gaps.
    assert pd.Period("2020Q1", "Q") not in gaps
    assert pd.Period("2020Q4", "Q") not in gaps


def test_sparsity_report_empty_records_returns_empty_report():
    from puremacro.narrative._signal_quality import compute_sparsity_report

    rep = compute_sparsity_report([])
    assert rep.n_docs_per_period.empty
    assert rep.avg_doc_length.empty
    assert rep.coverage_gaps == []


def test_sparsity_report_tolerates_5tuple_records_with_magnitude():
    from puremacro.narrative._signal_quality import compute_sparsity_report

    # 5-tuple: (date, text, url, metadata, magnitude)
    recs = [(pd.Timestamp("2020-01-15"), "alpha beta", "u", {}, 2.0)]
    rep = compute_sparsity_report(recs)
    assert int(rep.n_docs_per_period.loc[pd.Period("2020Q1", "Q")]) == 1
    assert rep.avg_doc_length.loc[pd.Period("2020Q1", "Q")] == 2.0
