"""Slice 1 of the signal contract — RiskIndex schema extension."""
from __future__ import annotations

import pandas as pd
import pytest


def test_signal_quality_report_constructs_with_sparsity_fields_only():
    from puremacro.narrative.types import SignalQualityReport

    n_docs = pd.Series([3, 5, 2], index=pd.period_range("2020Q1", periods=3, freq="Q"))
    avg_len = pd.Series([100.0, 120.0, 80.0], index=n_docs.index)
    report = SignalQualityReport(
        n_docs_per_period=n_docs,
        avg_doc_length=avg_len,
        coverage_gaps=[],
    )
    assert report.n_docs_per_period.iloc[0] == 3
    assert report.avg_doc_length.iloc[1] == 120.0
    assert report.coverage_gaps == []
    # Slice-1 fields default to None / empty:
    assert report.kernel_agreement is None
    assert report.multilingual_parity is None
    assert report.doc_bootstrap_sd is None
    assert report.corpus_loo_max_swing is None
    assert report.benchmark_scores == {}
    assert report.event_panel is None
    assert report.survey_scores == {}


def test_signal_quality_report_summary_returns_one_row_dataframe():
    from puremacro.narrative.types import SignalQualityReport

    n_docs = pd.Series([3, 5], index=pd.period_range("2020Q1", periods=2, freq="Q"))
    avg_len = pd.Series([100.0, 120.0], index=n_docs.index)
    report = SignalQualityReport(
        n_docs_per_period=n_docs,
        avg_doc_length=avg_len,
        coverage_gaps=[pd.Period("2019Q4", freq="Q")],
    )
    df = report.summary()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1
    assert "mean_n_docs" in df.columns
    assert "mean_doc_length" in df.columns
    assert "n_coverage_gaps" in df.columns
    assert df["mean_n_docs"].iloc[0] == 4.0
    assert df["n_coverage_gaps"].iloc[0] == 1


def test_riskindex_defaults_to_none_for_new_fields_and_is_backward_compatible():
    from puremacro.narrative.types import RiskIndex

    s = pd.Series([1.0, 2.0, 3.0],
                  index=pd.date_range("2020-01-01", periods=3, freq="QS"))
    ri = RiskIndex(
        name="test", country="USA", series=s,
        method="keyword_count", corpus="x",
        language="en", normalization="zscore",
    )
    assert ri.quality is None
    assert ri.draws is None
    # Existing methods still work unchanged.
    assert "n_quarters" in ri.diagnostics()
    assert ri.to_frame().shape == (3, 4)


def test_riskindex_accepts_quality_report():
    from puremacro.narrative.types import RiskIndex, SignalQualityReport

    s = pd.Series([1.0], index=pd.date_range("2020-01-01", periods=1, freq="QS"))
    report = SignalQualityReport(
        n_docs_per_period=pd.Series([5], index=pd.period_range("2020Q1", periods=1, freq="Q")),
        avg_doc_length=pd.Series([100.0], index=pd.period_range("2020Q1", periods=1, freq="Q")),
        coverage_gaps=[],
    )
    ri = RiskIndex(
        name="t", country="USA", series=s,
        method="keyword_count", corpus="x",
        language="en", normalization="zscore",
        quality=report,
    )
    assert ri.quality is report
    assert int(ri.quality.n_docs_per_period.iloc[0]) == 5


def test_riskindex_rejects_draws_with_wrong_index():
    from puremacro.narrative.types import RiskIndex

    s = pd.Series([1.0, 2.0],
                  index=pd.date_range("2020-01-01", periods=2, freq="QS"))
    # Draws index differs from series.index → must raise.
    bad_draws = pd.DataFrame(
        [[0.0, 0.1], [0.0, 0.1]],
        index=pd.date_range("2030-01-01", periods=2, freq="QS"),
        columns=pd.MultiIndex.from_tuples([("kernel", 0), ("kernel", 1)],
                                          names=["source", "draw_id"]),
    )
    with pytest.raises(ValueError, match="draws.index"):
        RiskIndex(name="t", country="USA", series=s,
                  method="keyword_count", corpus="x",
                  language="en", normalization="zscore",
                  draws=bad_draws)


def test_riskindex_rejects_draws_without_source_draw_id_multiindex():
    from puremacro.narrative.types import RiskIndex

    s = pd.Series([1.0, 2.0],
                  index=pd.date_range("2020-01-01", periods=2, freq="QS"))
    bad_draws = pd.DataFrame(
        [[0.0, 0.1], [0.0, 0.1]],
        index=s.index,
        columns=["a", "b"],   # flat index, not the required MultiIndex.
    )
    with pytest.raises(ValueError, match="draws.columns"):
        RiskIndex(name="t", country="USA", series=s,
                  method="keyword_count", corpus="x",
                  language="en", normalization="zscore",
                  draws=bad_draws)


def test_riskindex_rejects_draws_with_invalid_source_tag():
    from puremacro.narrative.types import RiskIndex

    s = pd.Series([1.0],
                  index=pd.date_range("2020-01-01", periods=1, freq="QS"))
    bad_draws = pd.DataFrame(
        [[0.0]],
        index=s.index,
        columns=pd.MultiIndex.from_tuples([("not_a_source", 0)],
                                          names=["source", "draw_id"]),
    )
    with pytest.raises(ValueError, match="source"):
        RiskIndex(name="t", country="USA", series=s,
                  method="keyword_count", corpus="x",
                  language="en", normalization="zscore",
                  draws=bad_draws)


def test_puremacro_version_is_wellformed():
    import re

    import puremacro
    assert isinstance(puremacro.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", puremacro.__version__), puremacro.__version__


def test_signal_quality_report_summary_columns_are_pinned():
    """Pin the public summary() column schema so cross-index tables built
    against these names don't break silently. Slice 2 / 3 may ADD columns
    but must NOT rename these."""
    from puremacro.narrative.types import SignalQualityReport

    empty = SignalQualityReport(
        n_docs_per_period=pd.Series(dtype="int64"),
        avg_doc_length=pd.Series(dtype="float64"),
        coverage_gaps=[],
    )
    cols = set(empty.summary().columns)
    expected = {
        "mean_n_docs", "mean_doc_length", "n_coverage_gaps",
        "has_kernel_draws", "has_multiling", "has_doc_boot", "has_corpus_loo",
        "n_benchmark_scores", "has_event_panel", "n_survey_scores",
    }
    assert expected.issubset(cols), (
        f"summary() must keep these column names: {sorted(expected - cols)} missing"
    )
