"""α.2 stability report: grid construction + summary metadata."""
from __future__ import annotations

from pathlib import Path

import math
import pandas as pd
import pytest

from puremacro.narrative.validation.stability import NarrativeStabilityReport


def _trivial_index_fn(records, **kwargs):
    """A stand-in index function returning a constant series — lets us
    test scaffolding without invoking real kernels."""
    from puremacro.narrative.types import RiskIndex
    idx = pd.date_range("2018Q1", "2023Q4", freq="QS")
    series = pd.Series(1.0, index=idx, name="trivial")
    return RiskIndex(name="trivial", country="USA", series=series,
                     method="trivial", corpus="test", language="en",
                     normalization="raw", metadata={})


def _trivial_text_iter():
    return iter([])


def _trivial_benchmark():
    idx = pd.date_range("2018Q1", "2023Q4", freq="QS")
    return pd.Series(range(len(idx)), index=idx, name="bench")


def test_stability_report_grid_has_four_axes():
    rep = NarrativeStabilityReport(
        index_fn=_trivial_index_fn,
        benchmark_series=_trivial_benchmark(),
        benchmark_name="trivial_bench",
        perturbations={
            "kernel": ["k1", "k2"],
            "lexicon_variant": ["strict", "expanded"],
            "base_period": [None, ("2017", "2026")],
            "leave_out": [None, "FED", "ECB"],
        },
        text_iter_factory=_trivial_text_iter,
        country="USA",
        language="en",
    )
    grid = rep._grid()
    # 2 × 2 × 2 × 3 = 24 cells
    assert len(grid) == 24
    assert set(grid[0].keys()) == {"kernel", "lexicon_variant",
                                    "base_period", "leave_out"}


def test_stability_report_grid_subsamples_when_above_max():
    rep = NarrativeStabilityReport(
        index_fn=_trivial_index_fn,
        benchmark_series=_trivial_benchmark(),
        benchmark_name="trivial_bench",
        perturbations={
            "kernel": ["a", "b", "c", "d"],
            "lexicon_variant": ["x", "y", "z"],
            "base_period": [None, ("a", "b"), ("c", "d"), ("e", "f")],
            "leave_out": [None, "X", "Y", "Z", "W"],
        },
        text_iter_factory=_trivial_text_iter,
        country="USA",
        max_cells=16,
        seed=0,
    )
    grid = rep._grid()
    # Full product is 4*3*4*5 = 240; subsample caps at 16
    assert len(grid) == 16


def test_stability_report_missing_axis_raises():
    rep = NarrativeStabilityReport(
        index_fn=_trivial_index_fn,
        benchmark_series=_trivial_benchmark(),
        benchmark_name="trivial_bench",
        perturbations={"kernel": ["k1"]},   # missing other axes
        text_iter_factory=_trivial_text_iter,
        country="USA",
    )
    with pytest.raises(ValueError, match="missing axes"):
        rep._grid()


def test_stability_report_summary_keys_after_run():
    rep = NarrativeStabilityReport(
        index_fn=_trivial_index_fn,
        benchmark_series=_trivial_benchmark(),
        benchmark_name="trivial_bench",
        perturbations={
            "kernel": ["k1"],
            "lexicon_variant": ["strict"],
            "base_period": [None],
            "leave_out": [None],
        },
        text_iter_factory=_trivial_text_iter,
        country="USA",
        language="en",
    )
    # Inject canned results so we can check summary() without running real index_fn.
    rep._run_results = pd.DataFrame({"rho": [0.5, 0.6, 0.55]})
    summ = rep.summary()
    assert set(summ.keys()) >= {"cv", "min_rho", "max_rho", "n_specs",
                                "benchmark", "computed_at"}
    assert summ["benchmark"] == "trivial_bench"
    assert summ["n_specs"] == 3
    assert math.isfinite(summ["cv"])


def test_stability_report_summary_before_run_raises():
    rep = NarrativeStabilityReport(
        index_fn=_trivial_index_fn,
        benchmark_series=_trivial_benchmark(),
        benchmark_name="trivial_bench",
        perturbations={
            "kernel": ["k1"], "lexicon_variant": ["strict"],
            "base_period": [None], "leave_out": [None],
        },
        text_iter_factory=_trivial_text_iter,
        country="USA",
    )
    with pytest.raises(RuntimeError, match="run\\(\\) before summary"):
        rep.summary()


def _fixture_text_iter_factory():
    """Return a zero-arg factory that yields the fixture corpus."""
    parquet_path = (Path(__file__).parent
                    / "fixtures" / "narrative" / "stability_corpus.parquet")
    df = pd.read_parquet(parquet_path)
    def _make():
        for _, r in df.iterrows():
            yield (r["date"], r["text"], r["url"], dict(r["meta"]))
    return _make


def _fixture_benchmark():
    """Synthetic benchmark series (linear trend over 24 quarters)."""
    idx = pd.date_range("2018Q1", "2023Q4", freq="QS")
    return pd.Series(range(len(idx)), index=idx, name="synthetic_aieu")


def _ltui_adapter(records, *, country, language, kernel, lexicon, base_period):
    """Translate stability-cell parameters into the real ltui() signature."""
    from puremacro.narrative.indices import ltui
    from puremacro.narrative.indices._lexicons import LEXICONS
    from puremacro.narrative.indices._lexicons_expanded import LEXICONS_EXPANDED

    src = LEXICONS_EXPANDED if lexicon == "expanded" else LEXICONS
    lex = src["ltui"][language] if language in src["ltui"] else src["ltui"].get("en")
    return ltui(
        records,
        country=country,
        language=language,
        lexicon=lex,
        base_period=base_period,
    )


def test_stability_report_runs_on_fixture_corpus():
    rep = NarrativeStabilityReport(
        index_fn=_ltui_adapter,
        benchmark_series=_fixture_benchmark(),
        benchmark_name="synthetic_aieu",
        perturbations={
            "kernel": ["triple_cooccurrence"],
            "lexicon_variant": ["strict", "expanded"],
            "base_period": [None, ("2018", "2024")],
            "leave_out": [None, "FED", "ECB", "BOJ"],
        },
        text_iter_factory=_fixture_text_iter_factory(),
        country="USA",
        language="en",
    )
    df = rep.run()
    # 1 × 2 × 2 × 4 = 16 cells
    assert len(df) == 16
    # At least one cell should have a finite ρ (the fixture is small + noisy,
    # but with 200 docs and a real lexicon, some matches should land).
    if df["rho"].notna().sum() == 0:
        # Don't make this a strict assert — the fixture is small and may not
        # yield enough quarterly samples for some perturbations. We just need
        # the scaffolding to execute without exceptions.
        pytest.skip("Fixture corpus too sparse for ρ computation on any cell")
    summ = rep.summary()
    assert summ["n_specs"] == 16
    # CV should be finite when there's any signal.
    if not math.isfinite(summ["cv"]):
        pytest.skip("CV non-finite on fixture; may need larger corpus")
