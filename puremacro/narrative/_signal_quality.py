"""Sparsity / coverage diagnostics for narrative-index records.

Slice 1 of the signal contract (puremacro 0.65.0). The full
SignalQualityReport schema is in puremacro.narrative.types; this
module fills the sparsity / coverage fields from a materialised list
of records (the same `records_list` that
`aggregate.index_to_quarterly` builds internally). Stability and
calibration fields are added in Slices 2 and 3.

A "record" is a 4-tuple `(date, text, source_url, metadata)` or a
5-tuple `(..., magnitude)` — both shapes are tolerated.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from .types import SignalQualityReport


def compute_sparsity_report(
    records_list: Sequence,
    *,
    freq: str = "Q",
) -> SignalQualityReport:
    """Build a sparsity-only SignalQualityReport from a records list.

    Parameters
    ----------
    records_list : sequence of 4- or 5-tuples `(date, text, ...)`.
    freq : pandas Period frequency for bucketing (default `"Q"`).

    Returns
    -------
    SignalQualityReport with only the sparsity / coverage fields populated.
    All Slice-2/3 fields are left at their defaults (None / empty dict).
    """
    if not records_list:
        return SignalQualityReport(
            n_docs_per_period=pd.Series(dtype="int64"),
            avg_doc_length=pd.Series(dtype="float64"),
            coverage_gaps=[],
        )

    rows = []
    for rec in records_list:
        date = pd.Timestamp(rec[0])
        text = str(rec[1]) if len(rec) > 1 and rec[1] is not None else ""
        rows.append({"date": date, "n_tokens": len(text.split())})

    df = pd.DataFrame(rows)
    df["period"] = df["date"].dt.to_period(freq)

    n_docs = df.groupby("period").size().astype("int64")
    avg_len = df.groupby("period")["n_tokens"].mean().astype("float64")

    full = pd.period_range(df["period"].min(), df["period"].max(), freq=freq)
    populated = set(n_docs.index)
    gaps = [p for p in full if p not in populated]

    return SignalQualityReport(
        n_docs_per_period=n_docs,
        avg_doc_length=avg_len,
        coverage_gaps=gaps,
    )
