# puremacro 0.65.0 Implementation Plan — signal contract, Slice 1 (schema + sparsity)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Slice 1 of the signal contract as `0.65.0`. Extend `puremacro.narrative.RiskIndex` with two optional fields (`quality: SignalQualityReport | None`, `draws: pd.DataFrame | None`); add a sparsity-only `SignalQualityReport`; wire a `with_quality=False` kwarg through `aggregate.index_to_quarterly` and every canonical index function. Strict backwards-compat: every existing caller behaves identically.

**Architecture:** Centralised through the existing `narrative.aggregate.index_to_quarterly` choke point — every text-based index already calls it, so adding the kwarg there gives universal coverage with one core change. Each top-level index function (and the wrapper indices that re-export `lui`) gains a passthrough `with_quality` kwarg. A new `narrative/_signal_quality.py` module computes the sparsity fields from the materialised records list. The placeholder calibration types (`BenchmarkScore` / `EventPanelScore` / `SurveyScore`) are typed as `Any` in this slice; they get full dataclass implementations in Slice 3 (0.67.0).

**Tech Stack:** numpy, scipy, pandas — already in deps. No new dependencies. Python ≥3.10. Pyodide-pure.

**Spec:** `docs/specs/2026-05-26-signal-contract-design.md`

---

## File map

### New files
- `puremacro/narrative/_signal_quality.py` — `compute_sparsity_report(records_list, freq="QS") -> SignalQualityReport`.
- `tests/test_signal_contract/__init__.py` — empty.
- `tests/test_signal_contract/test_schema_extension.py` — `RiskIndex` schema round-trip + new-field validation + convenience-method behaviour.
- `tests/test_signal_contract/test_sparsity.py` — `compute_sparsity_report` unit tests.
- `tests/test_signal_contract/test_with_quality_indices.py` — per-index parity test + cross-index coverage assertion (every name in `narrative.indices.__all__` accepts `with_quality=`).
- `notebooks/R4_signal_contract/R4_01_schema_demo.ipynb` — built from the paired builder.
- `tools/make_notebook_R4_01.py` — paired builder (per memory rule: notebooks ↔ builders must ship together).
- `docs/SIGNAL_CONTRACT.md` — single-page reference for the schema + opt-in surface; will be expanded in Slice 2 / 3.

### Modified files
- `puremacro/narrative/types.py` — add `SignalQualityReport` dataclass; add placeholder type aliases `BenchmarkScore = Any`, `EventPanelScore = Any`, `SurveyScore = Any` (replaced by real dataclasses in 0.67.0); extend `RiskIndex` with `quality` and `draws` optional fields; extend `__post_init__` to validate `draws` shape; extend `__all__`.
- `puremacro/narrative/aggregate.py` — `index_to_quarterly` gains `with_quality: bool = False`; when True, calls `compute_sparsity_report(records_list)` and passes the result into `RiskIndex(..., quality=...)`.
- `puremacro/narrative/indices/epu.py` — add `with_quality: bool = False`; forward to `index_to_quarterly`.
- `puremacro/narrative/indices/mpu.py` — same.
- `puremacro/narrative/indices/gpr.py` — same.
- `puremacro/narrative/indices/tone.py` — same.
- `puremacro/narrative/indices/wui.py` — same.
- `puremacro/narrative/indices/lui.py` — same.
- `puremacro/narrative/indices/ltui.py` — same on `ltui`, `ltui_up`, `ltui_down`.
- `puremacro/narrative/indices/lwui.py` — same on `lwui`, `lwui_wage`.
- `puremacro/narrative/indices/beige_book.py` — add `with_quality` to `bbui`; forward to inner `lui` call.
- `puremacro/narrative/indices/bluesky.py` — add `with_quality` to `bluesky_ui`; forward to inner `lui` call.
- `puremacro/narrative/indices/us_executive.py` — add `with_quality` to `erpui`, `sotuui`, `cboui`; forward.
- `puremacro/narrative/indices/eu_legislative.py` — add `with_quality` to `eurlex_ui`, `ep_ui`; forward.
- `puremacro/__init__.py` — bump `__version__` to `"0.65.0"`.
- `pyproject.toml` — bump `version` to `"0.65.0"`.
- `CHANGELOG.md` — prepend a `## 0.65.0 (2026-05-26)` section.
- `tests/test_pyodide_compat.py` — no change to the test body, but a manual verification step (Task 9) re-runs it to confirm the new modules don't leak forbidden imports.
- `ARCHITECTURE.md` — add a short "Signal contract" subsection in the result-object area.
- `README.md` — add a 6-line `with_quality=True` code block near the Quickstart.

### Working assumptions (verified via signature dumps)

- `RiskIndex` (puremacro/narrative/types.py:357) currently has fields `name, country, series, method, corpus, language, normalization, metadata` and a `__post_init__` that validates `method` against `VALID_RISKINDEX_METHODS` and `normalization` against `VALID_RISKINDEX_NORMALIZATION`. Convenience methods are `diagnostics()`, `to_frame()`, `as_instrument()`.
- `aggregate.index_to_quarterly` (puremacro/narrative/aggregate.py:171) is the central constructor every text-based index uses. Its full signature is `(records, *, kernel, country, language, name, method, corpus, normalization, freq="QS", agg="mean", weight_by=None, metadata=None) -> RiskIndex`. It materialises `records_list = list(records)` (line 214) before scoring — that's the same list we feed to the sparsity helper, no double iteration.
- `narrative.indices.__all__` lists the public surface: `bluesky_ui, epu, mpu, gpr, tone, wui, lui, ltui, ltui_up, ltui_down, lwui, lwui_wage, bbui, cboui, ep_ui, erpui, eurlex_ui, sotuui` (18 public functions), plus `consensus_disagreement` and kernel exports (no `with_quality=` for those — they're not single-index constructors).
- Wrapper indices (`bbui`, `bluesky_ui`, `erpui`, `sotuui`, `cboui`, `eurlex_ui`, `ep_ui`) wrap `lui` rather than calling `index_to_quarterly` directly. They forward `with_quality` down to the `lui()` call.
- `tests/test_pyodide_compat.py` walks every shippable submodule via `pkgutil` and asserts no forbidden module (`statsmodels`, `linearmodels`, `arch`) lands in `sys.modules`. Skip prefixes include `puremacro.narrative.sources` and `puremacro.narrative.scoring.llm`. We add no imports from those buckets.
- pyproject.toml shows `version = "0.64.0"` (lines 1–9). Bump to `"0.65.0"`.
- `puremacro/__init__.py` exposes `__version__` (line near top). Bump there too.
- Notebooks must be paired with `tools/make_notebook_*.py` builders per the user's memory feedback (else the next builder re-run silently clobbers the executed `.ipynb`). The builder writes the notebook; the user (or a controller task, never a subagent — long nbconvert times out) executes it via `jupyter nbconvert --execute`.
- Per `CONTRIBUTING.md` / repo precedent, pytest is the canonical test runner: `pytest tests/<path>::<name> -v` from the `puremacro/` package directory. Full suite runtime is ~12 minutes; do not run the full suite in tight loops — run the new test files individually during TDD.
- Commit message style (from `git log --oneline`): `chore(puremacro): …`, `feat(R3_12): …`, `docs(spec): …`. Slice 1 commits use `feat(0.65.0): …` for code changes, `docs(0.65.0): …` for documentation, `chore(puremacro): bump to 0.65.0` for the version bump.

---

## Task 1: Add `SignalQualityReport` dataclass (sparsity-only fields populated)

**Files:**
- Modify: `puremacro/narrative/types.py` (append after `RiskIndex` definition, before `__all__`).
- Test: `tests/test_signal_contract/test_schema_extension.py` (create the test file and its directory).

- [ ] **Step 1: Create the test directory and write the failing test**

Create `tests/test_signal_contract/__init__.py` (empty file).

Create `tests/test_signal_contract/test_schema_extension.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_signal_contract/test_schema_extension.py -v
```
Expected: FAIL with `ImportError: cannot import name 'SignalQualityReport'`.

- [ ] **Step 3: Add the dataclass to `puremacro/narrative/types.py`**

Append immediately before the existing `__all__` line (which is the last statement in the file). The full block:

```python
# --- Signal-quality report (Slice 1: sparsity-only fields populated; the
# kernel_agreement / multilingual_parity / *_sd / corpus_loo_max_swing
# fields wire up in Slice 2 — 0.66.0. The calibration fields use Any
# placeholders here; the BenchmarkScore / EventPanelScore / SurveyScore
# dataclasses arrive in Slice 3 — 0.67.0.) ---
BenchmarkScore = Any   # placeholder, replaced in 0.67.0
EventPanelScore = Any  # placeholder, replaced in 0.67.0
SurveyScore = Any      # placeholder, replaced in 0.67.0


@dataclass
class SignalQualityReport:
    """Signal-quality companion to RiskIndex.

    Slice 1 (0.65.0): sparsity / coverage fields only. Stability and
    calibration fields are declared in the schema (default-None / empty
    dict / None) so Slice 2 (draws) and Slice 3 (calibration) can fill
    them in-place without re-versioning the dataclass.
    """
    n_docs_per_period: pd.Series                       # date -> int
    avg_doc_length:    pd.Series                       # date -> float (tokens)
    coverage_gaps:     list[pd.Period]
    kernel_agreement:       pd.Series | None = None
    multilingual_parity:    pd.Series | None = None
    doc_bootstrap_sd:       pd.Series | None = None
    corpus_loo_max_swing:   pd.Series | None = None
    benchmark_scores:       dict[str, Any] = field(default_factory=dict)
    event_panel:            Any | None     = None
    survey_scores:          dict[str, Any] = field(default_factory=dict)
    metadata:               dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Flatten the report to a single-row DataFrame for cross-index tables."""
        return pd.DataFrame([{
            "mean_n_docs":         float(self.n_docs_per_period.mean())
                                    if not self.n_docs_per_period.empty else float("nan"),
            "mean_doc_length":     float(self.avg_doc_length.mean())
                                    if not self.avg_doc_length.empty else float("nan"),
            "n_coverage_gaps":     len(self.coverage_gaps),
            "has_kernel_draws":    self.kernel_agreement is not None,
            "has_multiling":       self.multilingual_parity is not None,
            "has_doc_boot":        self.doc_bootstrap_sd is not None,
            "has_corpus_loo":      self.corpus_loo_max_swing is not None,
            "n_benchmark_scores":  len(self.benchmark_scores),
            "has_event_panel":     self.event_panel is not None,
            "n_survey_scores":     len(self.survey_scores),
        }])
```

Then update the existing `__all__` list at the bottom of the file. The current value is:
```python
__all__ = ["NarrativeEvent", "NarrativeInstrument", "RiskIndex",
           "VALID_KINDS", "VALID_TARGETS_BY_KIND",
           "VALID_TARGETS", "VALID_SIGNS", "VALID_SCORERS",
           "VALID_RISKINDEX_METHODS", "VALID_RISKINDEX_NORMALIZATION"]
```
Replace with:
```python
__all__ = ["NarrativeEvent", "NarrativeInstrument", "RiskIndex",
           "SignalQualityReport",
           "BenchmarkScore", "EventPanelScore", "SurveyScore",
           "VALID_KINDS", "VALID_TARGETS_BY_KIND",
           "VALID_TARGETS", "VALID_SIGNS", "VALID_SCORERS",
           "VALID_RISKINDEX_METHODS", "VALID_RISKINDEX_NORMALIZATION"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_signal_contract/test_schema_extension.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add puremacro/narrative/types.py tests/test_signal_contract/__init__.py tests/test_signal_contract/test_schema_extension.py
git commit -m "feat(0.65.0): add SignalQualityReport dataclass (sparsity-only fields)"
```

---

## Task 2: Add `compute_sparsity_report` helper

**Files:**
- Create: `puremacro/narrative/_signal_quality.py`.
- Test: `tests/test_signal_contract/test_sparsity.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_signal_contract/test_sparsity.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_signal_contract/test_sparsity.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'puremacro.narrative._signal_quality'`.

- [ ] **Step 3: Implement the helper**

Create `puremacro/narrative/_signal_quality.py`:

```python
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

from typing import Iterable, Sequence

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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_signal_contract/test_sparsity.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add puremacro/narrative/_signal_quality.py tests/test_signal_contract/test_sparsity.py
git commit -m "feat(0.65.0): add compute_sparsity_report helper"
```

---

## Task 3: Extend `RiskIndex` with `quality` and `draws` fields

**Files:**
- Modify: `puremacro/narrative/types.py` (the existing `RiskIndex` dataclass and its `__post_init__`).
- Test: append to `tests/test_signal_contract/test_schema_extension.py`.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_signal_contract/test_schema_extension.py`:

```python
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
    assert ri.to_frame().shape == (3, 3)


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
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_signal_contract/test_schema_extension.py -v
```
Expected: the 2 prior tests still pass; 4 new tests fail with `TypeError: RiskIndex.__init__() got an unexpected keyword argument 'quality'` (or similar).

- [ ] **Step 3: Extend `RiskIndex` and its `__post_init__`**

In `puremacro/narrative/types.py`, the existing `RiskIndex` dataclass (around line 357–398) gains two fields and an extended `__post_init__`.

Replace the existing `RiskIndex` body with:

```python
_VALID_DRAW_SOURCES = frozenset({"kernel", "lexicon", "doc", "corpus"})


@dataclass
class RiskIndex:
    """A continuous text-derived risk / uncertainty / tone index.

    ... (existing docstring lines preserved) ...

    Slice 1 of the signal contract (0.65.0) adds two optional fields:

    quality : SignalQualityReport | None
        Sparsity / coverage diagnostics. Populated when the index
        function was called with ``with_quality=True``. None by default
        so every existing caller is unaffected.
    draws : pd.DataFrame | None
        Posterior draws of the index (Slice 2 / 0.66.0). Row index =
        ``series.index``. Column index = MultiIndex ``('source',
        'draw_id')`` with ``source ∈ {kernel, lexicon, doc, corpus}``.
        None in Slice 1.
    """
    name: str
    country: str
    series: pd.Series
    method: str
    corpus: str
    language: str
    normalization: str
    metadata: dict[str, Any] = field(default_factory=dict)
    quality: "SignalQualityReport | None" = None
    draws:   pd.DataFrame | None          = None

    def __post_init__(self):
        if self.method not in VALID_RISKINDEX_METHODS:
            raise ValueError(
                f"RiskIndex: method {self.method!r} not in {VALID_RISKINDEX_METHODS}"
            )
        if self.normalization not in VALID_RISKINDEX_NORMALIZATION:
            raise ValueError(
                f"RiskIndex: normalization {self.normalization!r} not in "
                f"{VALID_RISKINDEX_NORMALIZATION}"
            )
        if self.draws is not None:
            if not self.draws.index.equals(self.series.index):
                raise ValueError(
                    "RiskIndex.draws.index must equal series.index "
                    f"(got draws.index of length {len(self.draws.index)}, "
                    f"series.index of length {len(self.series.index)})"
                )
            cols = self.draws.columns
            if not isinstance(cols, pd.MultiIndex) or cols.nlevels != 2 \
                    or list(cols.names) != ["source", "draw_id"]:
                raise ValueError(
                    "RiskIndex.draws.columns must be a 2-level MultiIndex "
                    "with names ['source', 'draw_id']; got "
                    f"{cols!r} (names={getattr(cols, 'names', None)})"
                )
            bad = set(cols.get_level_values("source")) - _VALID_DRAW_SOURCES
            if bad:
                raise ValueError(
                    f"RiskIndex.draws: source tags {sorted(bad)} not in "
                    f"{sorted(_VALID_DRAW_SOURCES)}"
                )
```

The existing methods `diagnostics()`, `to_frame()`, `as_instrument()` are unchanged in Slice 1. (Convenience methods `.band()`, `.draws_attribution()`, `.mean_draws()` arrive in Slice 2 when draws are actually populated.)

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_signal_contract/test_schema_extension.py -v
```
Expected: 6 passed (the original 2 plus the 4 new).

- [ ] **Step 5: Sanity-run existing narrative tests to confirm zero regression**

```bash
pytest tests/test_narrative.py tests/test_narrative_indices.py -v
```
Expected: no new failures (any pre-existing failures from `_phase2_audit_notes.md` may persist — confirm the count matches the baseline).

- [ ] **Step 6: Commit**

```bash
git add puremacro/narrative/types.py tests/test_signal_contract/test_schema_extension.py
git commit -m "feat(0.65.0): extend RiskIndex with optional quality/draws fields"
```

---

## Task 4: Wire `with_quality=` into `aggregate.index_to_quarterly`

**Files:**
- Modify: `puremacro/narrative/aggregate.py` (lines 171–270 region — the `index_to_quarterly` function).
- Test: append to `tests/test_signal_contract/test_with_quality_indices.py` (create the file).

- [ ] **Step 1: Write the failing test**

Create `tests/test_signal_contract/test_with_quality_indices.py`:

```python
"""Slice 1 — `with_quality=` rollout across canonical narrative indices."""
from __future__ import annotations

import importlib
import inspect

import pandas as pd
import pytest


# 4-tuple records: (date, text, source_url, metadata).
_RECS = [
    (pd.Timestamp("2020-01-15"), "policy uncertainty about economic outlook", "u", {}),
    (pd.Timestamp("2020-02-20"), "fiscal policy and tax reform uncertain", "u", {}),
    (pd.Timestamp("2020-05-10"), "monetary policy stable, economy improving", "u", {}),
]


def test_index_to_quarterly_with_quality_true_attaches_sparsity_report():
    from puremacro.narrative.aggregate import index_to_quarterly
    from puremacro.narrative.types import SignalQualityReport

    def _kernel(records):
        # Trivial kernel: 1.0 per doc.
        return [(r[0], 1.0) for r in records]

    ri = index_to_quarterly(
        _RECS, kernel=_kernel,
        country="USA", language="en",
        name="t", method="keyword_count", corpus="x",
        normalization="zscore",
        with_quality=True,
    )
    assert ri.quality is not None
    assert isinstance(ri.quality, SignalQualityReport)
    assert int(ri.quality.n_docs_per_period.loc[pd.Period("2020Q1", "Q")]) == 2
    assert int(ri.quality.n_docs_per_period.loc[pd.Period("2020Q2", "Q")]) == 1


def test_index_to_quarterly_with_quality_false_keeps_quality_none():
    from puremacro.narrative.aggregate import index_to_quarterly

    def _kernel(records):
        return [(r[0], 1.0) for r in records]

    ri = index_to_quarterly(
        _RECS, kernel=_kernel,
        country="USA", language="en",
        name="t", method="keyword_count", corpus="x",
        normalization="zscore",
        # with_quality defaults to False
    )
    assert ri.quality is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_signal_contract/test_with_quality_indices.py -v
```
Expected: FAIL with `TypeError: index_to_quarterly() got an unexpected keyword argument 'with_quality'`.

- [ ] **Step 3: Extend `index_to_quarterly`**

In `puremacro/narrative/aggregate.py`:

1. Add the import near the top of the file (in the existing imports block, after `from .types import RiskIndex`):
   ```python
   from ._signal_quality import compute_sparsity_report
   ```

2. In the signature of `index_to_quarterly`, append `with_quality` as a new keyword-only argument (after the existing `metadata: dict | None = None,`):
   ```python
       metadata: dict | None = None,
       with_quality: bool = False,
   ) -> RiskIndex:
   ```

3. Add a paragraph to the docstring under Parameters, before the Returns section:
   ```text
       with_quality : if True, compute a sparsity-only ``SignalQualityReport``
           from ``records_list`` and attach it to the returned ``RiskIndex``
           as ``ri.quality``. Default False preserves the 0.64.0 behaviour
           (``ri.quality is None``).
   ```

4. At the end of the function (just before the existing `return RiskIndex(...)`), compute the report when requested. The existing return is:
   ```python
       return RiskIndex(
           name=name, country=country, series=out, method=method,
           corpus=corpus, language=language, normalization=normalization,
           metadata={**(metadata or {}), "n_docs": int(len(records_list))},
       )
   ```
   Replace with:
   ```python
       quality = compute_sparsity_report(records_list) if with_quality else None
       return RiskIndex(
           name=name, country=country, series=out, method=method,
           corpus=corpus, language=language, normalization=normalization,
           metadata={**(metadata or {}), "n_docs": int(len(records_list))},
           quality=quality,
       )
   ```
   (If the actual return-site formatting differs from the snippet above — read the file first — adapt to the existing field order while keeping the `quality=quality` line.)

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_signal_contract/test_with_quality_indices.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Sanity-run narrative tests**

```bash
pytest tests/test_narrative.py tests/test_narrative_indices.py tests/test_narrative_aggregate_kind.py -v
```
Expected: no new failures vs. baseline.

- [ ] **Step 6: Commit**

```bash
git add puremacro/narrative/aggregate.py tests/test_signal_contract/test_with_quality_indices.py
git commit -m "feat(0.65.0): wire with_quality= into index_to_quarterly"
```

---

## Task 5: Add `with_quality=` to canonical text-based indices (EPU/MPU/GPR/TONE/WUI/LUI/LTUI/LWUI)

**Files:** modify each, one tiny diff per file.
- Modify: `puremacro/narrative/indices/epu.py`
- Modify: `puremacro/narrative/indices/mpu.py`
- Modify: `puremacro/narrative/indices/gpr.py`
- Modify: `puremacro/narrative/indices/tone.py`
- Modify: `puremacro/narrative/indices/wui.py`
- Modify: `puremacro/narrative/indices/lui.py`
- Modify: `puremacro/narrative/indices/ltui.py` (three functions: `ltui`, `ltui_up`, `ltui_down`)
- Modify: `puremacro/narrative/indices/lwui.py` (two functions: `lwui`, `lwui_wage`)
- Test: append to `tests/test_signal_contract/test_with_quality_indices.py`.

**Pattern (identical for each function):** add `with_quality: bool = False,` as the last keyword-only argument in the signature; pass `with_quality=with_quality` to the inner `index_to_quarterly` call. Docstring gets a one-line addition: `with_quality : forwarded to :func:\`puremacro.narrative.aggregate.index_to_quarterly\`.`

- [ ] **Step 1: Write the failing per-function test**

Append to `tests/test_signal_contract/test_with_quality_indices.py`:

```python
# 14 canonical text-based index functions whose `with_quality=` plumbing
# is verified directly. The wrapper indices (bbui, bluesky_ui, erpui,
# sotuui, cboui, eurlex_ui, ep_ui) are tested in Task 6 alongside the
# coverage assertion.
_DIRECT_INDICES = [
    "puremacro.narrative.indices.epu:epu",
    "puremacro.narrative.indices.mpu:mpu",
    "puremacro.narrative.indices.gpr:gpr",
    "puremacro.narrative.indices.tone:tone",
    "puremacro.narrative.indices.wui:wui",
    "puremacro.narrative.indices.lui:lui",
    "puremacro.narrative.indices.ltui:ltui",
    "puremacro.narrative.indices.ltui:ltui_up",
    "puremacro.narrative.indices.ltui:ltui_down",
    "puremacro.narrative.indices.lwui:lwui",
    "puremacro.narrative.indices.lwui:lwui_wage",
]


@pytest.mark.parametrize("dotted", _DIRECT_INDICES)
def test_direct_index_accepts_with_quality_kwarg(dotted):
    mod_path, fn_name = dotted.split(":")
    mod = importlib.import_module(mod_path)
    fn = getattr(mod, fn_name)
    sig = inspect.signature(fn)
    assert "with_quality" in sig.parameters, (
        f"{dotted}: missing `with_quality=` kwarg (Slice 1 contract)"
    )
    param = sig.parameters["with_quality"]
    assert param.default is False, (
        f"{dotted}: with_quality default must be False; got {param.default!r}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_signal_contract/test_with_quality_indices.py -v -k with_quality_kwarg
```
Expected: 11 failures (one per `_DIRECT_INDICES` entry), each reporting "missing `with_quality=` kwarg".

- [ ] **Step 3: Edit `puremacro/narrative/indices/epu.py`**

Existing signature ends with:
```python
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
) -> RiskIndex:
```
Replace with:
```python
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
    with_quality: bool = False,
) -> RiskIndex:
```
Existing inner `return index_to_quarterly(...)` ends with `metadata={...})`. Add `with_quality=with_quality,` immediately before the closing `)`. Verify by re-reading the file before the next step.

- [ ] **Step 4: Edit `puremacro/narrative/indices/mpu.py`**

Same pattern. Add `with_quality: bool = False,` as the last kwarg in the signature; add `with_quality=with_quality,` to the inner `index_to_quarterly(...)` call.

- [ ] **Step 5: Edit `puremacro/narrative/indices/gpr.py`**

Same pattern.

- [ ] **Step 6: Edit `puremacro/narrative/indices/tone.py`**

Same pattern. The function has an extra `method=` kwarg before `agg`; insert `with_quality` AFTER `agg=` (last position is the rule of thumb).

- [ ] **Step 7: Edit `puremacro/narrative/indices/wui.py`**

Same pattern.

- [ ] **Step 8: Edit `puremacro/narrative/indices/lui.py`**

The signature ends with `negation: bool = True,`. Insert `with_quality: bool = False,` AFTER `negation`. Inner call to `index_to_quarterly` gets `with_quality=with_quality,`.

- [ ] **Step 9: Edit `puremacro/narrative/indices/ltui.py`**

Three functions in this file: `ltui`, `ltui_up`, `ltui_down`. Each gets `with_quality: bool = False,` as its last kwarg AND forwards it to the inner call (`lui(...)` or `index_to_quarterly(...)` — read the file to confirm which). For each, the diff is: one new kwarg in the signature; one new `with_quality=with_quality,` line in the inner call.

- [ ] **Step 10: Edit `puremacro/narrative/indices/lwui.py`**

Two functions: `lwui`, `lwui_wage`. For each: add `with_quality: bool = False,` as the last keyword-only argument in the signature; add `with_quality=with_quality,` to the inner call (read the file to confirm whether the inner call is `lui(...)` or `index_to_quarterly(...)`).

- [ ] **Step 11: Run test to verify pass**

```bash
pytest tests/test_signal_contract/test_with_quality_indices.py -v -k with_quality_kwarg
```
Expected: 11 passed.

- [ ] **Step 12: Run full narrative tests to confirm zero regression**

```bash
pytest tests/test_narrative.py tests/test_narrative_indices.py -v
```
Expected: no new failures.

- [ ] **Step 13: Commit**

```bash
git add puremacro/narrative/indices/epu.py puremacro/narrative/indices/mpu.py \
        puremacro/narrative/indices/gpr.py puremacro/narrative/indices/tone.py \
        puremacro/narrative/indices/wui.py puremacro/narrative/indices/lui.py \
        puremacro/narrative/indices/ltui.py puremacro/narrative/indices/lwui.py \
        tests/test_signal_contract/test_with_quality_indices.py
git commit -m "feat(0.65.0): add with_quality= to text-based canonical indices"
```

---

## Task 6: Add `with_quality=` to wrapper indices (BBUI / BLUESKY_UI / ERPUI / SOTUUI / CBOUI / EURLEX_UI / EP_UI) + coverage assertion

**Files:**
- Modify: `puremacro/narrative/indices/beige_book.py`
- Modify: `puremacro/narrative/indices/bluesky.py`
- Modify: `puremacro/narrative/indices/us_executive.py` (three functions: `erpui`, `sotuui`, `cboui`)
- Modify: `puremacro/narrative/indices/eu_legislative.py` (two functions: `eurlex_ui`, `ep_ui`)
- Test: append to `tests/test_signal_contract/test_with_quality_indices.py`.

**Pattern:** these all wrap `lui(...)`; the diff is the same `with_quality: bool = False,` signature addition and `with_quality=with_quality` in the inner `lui(...)` call.

- [ ] **Step 1: Append the coverage assertion + wrapper tests**

Append to `tests/test_signal_contract/test_with_quality_indices.py`:

```python
_WRAPPER_INDICES = [
    "puremacro.narrative.indices.beige_book:bbui",
    "puremacro.narrative.indices.bluesky:bluesky_ui",
    "puremacro.narrative.indices.us_executive:erpui",
    "puremacro.narrative.indices.us_executive:sotuui",
    "puremacro.narrative.indices.us_executive:cboui",
    "puremacro.narrative.indices.eu_legislative:eurlex_ui",
    "puremacro.narrative.indices.eu_legislative:ep_ui",
]


@pytest.mark.parametrize("dotted", _WRAPPER_INDICES)
def test_wrapper_index_accepts_with_quality_kwarg(dotted):
    mod_path, fn_name = dotted.split(":")
    mod = importlib.import_module(mod_path)
    fn = getattr(mod, fn_name)
    sig = inspect.signature(fn)
    assert "with_quality" in sig.parameters, (
        f"{dotted}: missing `with_quality=` kwarg (Slice 1 contract)"
    )
    assert sig.parameters["with_quality"].default is False


def test_every_public_index_in__all__has_with_quality():
    """Cross-check: every name in `narrative.indices.__all__` that is a
    single-index constructor (i.e. takes records and returns a RiskIndex
    or DataFrame) accepts `with_quality=`. The excluded names are the
    kernel exports and `consensus_disagreement` — they're tracked here
    explicitly so adding a new index without `with_quality=` is caught."""
    import puremacro.narrative.indices as I

    EXCLUDED = {
        # Cross-source derived (no records argument).
        "consensus_disagreement", "CROSS_SOURCE_GROUPS",
        # Kernel exports (not indices).
        "embedding_similarity_kernel", "build_seed_prototype",
        "make_sentence_transformer_embedder",
        "mnl_kernel", "canonicalize_weights",
        "llm_prob_kernel", "LLMProvider", "MockProvider", "AnthropicProvider",
        # Lexicons constant (not a callable).
        "LEXICONS",
    }
    public_index_names = [n for n in I.__all__ if n not in EXCLUDED]
    missing = []
    for name in public_index_names:
        fn = getattr(I, name)
        if "with_quality" not in inspect.signature(fn).parameters:
            missing.append(name)
    assert not missing, (
        "Slice-1 contract violation: every canonical index in "
        "`narrative.indices.__all__` must accept `with_quality=False`. "
        f"Missing: {missing}"
    )


def test_lui_end_to_end_attaches_quality():
    """One end-to-end check: lui with `with_quality=True` produces a
    RiskIndex whose .quality.n_docs_per_period reflects the input."""
    from puremacro.narrative.indices import lui

    ri = lui(
        _RECS,
        country="USA",
        language="en",
        with_quality=True,
    )
    assert ri.quality is not None
    # The 3 fixture records span 2020Q1 (2) and 2020Q2 (1).
    assert int(ri.quality.n_docs_per_period.loc[pd.Period("2020Q1", "Q")]) == 2
    assert int(ri.quality.n_docs_per_period.loc[pd.Period("2020Q2", "Q")]) == 1
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_signal_contract/test_with_quality_indices.py -v
```
Expected: 7 wrapper failures + 1 coverage failure listing 7 missing names. The end-to-end `lui` test passes (lui got `with_quality=` in Task 5).

- [ ] **Step 3: Edit `puremacro/narrative/indices/beige_book.py`**

Add `with_quality: bool = False,` as the last kwarg in `bbui`'s signature; forward it to the inner `lui(...)` call as `with_quality=with_quality`.

- [ ] **Step 4: Edit `puremacro/narrative/indices/bluesky.py`**

Same pattern in `bluesky_ui`.

- [ ] **Step 5: Edit `puremacro/narrative/indices/us_executive.py`**

Three functions in this file (`erpui`, `sotuui`, `cboui`). Each: `with_quality: bool = False,` last in signature; `with_quality=with_quality,` in the inner `lui(...)` (or `index_to_quarterly(...)`) call.

- [ ] **Step 6: Edit `puremacro/narrative/indices/eu_legislative.py`**

Two functions (`eurlex_ui`, `ep_ui`). Same pattern.

- [ ] **Step 7: Run tests to verify pass**

```bash
pytest tests/test_signal_contract/test_with_quality_indices.py -v
```
Expected: all tests pass (2 + 11 + 7 + 1 coverage + 1 end-to-end = 22 passed).

- [ ] **Step 8: Run full narrative suite**

```bash
pytest tests/test_narrative.py tests/test_narrative_indices.py tests/test_narrative_aggregate_kind.py tests/test_narrative_5tuple_schema.py -v
```
Expected: no new failures vs. baseline.

- [ ] **Step 9: Commit**

```bash
git add puremacro/narrative/indices/beige_book.py puremacro/narrative/indices/bluesky.py \
        puremacro/narrative/indices/us_executive.py puremacro/narrative/indices/eu_legislative.py \
        tests/test_signal_contract/test_with_quality_indices.py
git commit -m "feat(0.65.0): add with_quality= to wrapper indices + coverage test"
```

---

## Task 7: Re-run Pyodide-compat regression

**Files:** none modified — this is a verification task.

- [ ] **Step 1: Run the Pyodide-compat test**

```bash
pytest tests/test_pyodide_compat.py -v
```
Expected: PASS. The new module `puremacro.narrative._signal_quality` and the new symbols in `puremacro.narrative.types` are pure-numpy / pandas; they should not pull in any forbidden imports.

- [ ] **Step 2: If it fails, diagnose**

If `test_pyodide_compat` reports a new forbidden module in `sys.modules`:
1. Run `python -c "import puremacro.narrative._signal_quality; import sys; print([m for m in sys.modules if any(f in m for f in ['statsmodels','linearmodels','arch'])])"` from the package root.
2. Trace the offending import (almost certainly transitive through a kernel module — none should appear in Slice 1, but verify before assuming).
3. If a forbidden import really is required, follow the existing lazy-import pattern documented in `puremacro/narrative/scoring/llm.py` — import inside the function body, not at module top level.

- [ ] **Step 3: No commit needed** — verification-only task.

---

## Task 8: Build R4_01 schema-demo notebook (paired builder + executed notebook)

**Files:**
- Create: `tools/make_notebook_R4_01.py` — paired builder script.
- Create: `notebooks/R4_signal_contract/R4_01_schema_demo.ipynb` — produced by the builder, then executed.

Per the user's memory rule: notebooks and their builders ship together. The builder writes the `.ipynb`; executing the notebook (via `jupyter nbconvert --execute`) populates outputs. **Never re-run the builder over an executed notebook** — it strips outputs.

- [ ] **Step 1: Create the builder**

Create `tools/make_notebook_R4_01.py`:

```python
"""Builder for notebooks/R4_signal_contract/R4_01_schema_demo.ipynb.

R4_01 is the Slice-1 schema demo for the signal contract (puremacro
0.65.0). It demonstrates:
  1. The default behaviour of canonical indices (quality=None, draws=None).
  2. Opt-in `with_quality=True` and the resulting SignalQualityReport.
  3. The cross-index summary table built from SignalQualityReport.summary().

Run:
    python tools/make_notebook_R4_01.py
Then execute (one-shot, ~30s — controller-side, never a subagent):
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/R4_signal_contract/R4_01_schema_demo.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


OUT = Path("notebooks/R4_signal_contract/R4_01_schema_demo.ipynb")


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(
        "# R4_01 — Signal contract Slice 1: schema demo\n\n"
        "Demonstrates the 0.65.0 schema extension to `puremacro.narrative.RiskIndex`:\n\n"
        "- Default behaviour: `ri.quality is None`, `ri.draws is None` — every existing caller is unaffected.\n"
        "- Opt-in: `ri = lui(records, with_quality=True)` populates `ri.quality` with a sparsity-only `SignalQualityReport`.\n"
        "- `report.summary()` flattens the report to one DataFrame row for cross-index comparison.\n\n"
        "Slices 2 (0.66.0 — draws + propagation) and 3 (0.67.0 — calibration) will populate the rest of the report."
    ))

    cells.append(nbf.v4.new_code_cell(
        "from __future__ import annotations\n"
        "import pandas as pd\n"
        "import puremacro\n"
        "from puremacro.narrative.indices import lui, epu\n"
        "from puremacro.narrative.types import SignalQualityReport\n"
        "print('puremacro', puremacro.__version__)"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 1. Backwards-compat: default behaviour"))
    cells.append(nbf.v4.new_code_cell(
        "RECS = [\n"
        "    (pd.Timestamp('2020-01-15'), 'policy uncertainty about economic outlook', 'u', {}),\n"
        "    (pd.Timestamp('2020-02-20'), 'fiscal policy and tax reform uncertain', 'u', {}),\n"
        "    (pd.Timestamp('2020-05-10'), 'monetary policy stable, economy improving', 'u', {}),\n"
        "    (pd.Timestamp('2020-08-01'), 'unemployment falling, recovery uneven', 'u', {}),\n"
        "]\n"
        "ri_default = lui(RECS, country='USA', language='en')\n"
        "print('quality:', ri_default.quality)\n"
        "print('draws:',   ri_default.draws)\n"
        "ri_default.series"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 2. Opt-in: `with_quality=True`"))
    cells.append(nbf.v4.new_code_cell(
        "ri = lui(RECS, country='USA', language='en', with_quality=True)\n"
        "assert isinstance(ri.quality, SignalQualityReport)\n"
        "print('n_docs per quarter:'); print(ri.quality.n_docs_per_period)\n"
        "print('avg doc length:');     print(ri.quality.avg_doc_length)\n"
        "print('coverage gaps:',       ri.quality.coverage_gaps)"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 3. Cross-index summary table"))
    cells.append(nbf.v4.new_code_cell(
        "ri_epu = epu(RECS, country='USA', language='en', with_quality=True)\n"
        "summary = pd.concat([\n"
        "    ri.quality.summary().assign(index='lui'),\n"
        "    ri_epu.quality.summary().assign(index='epu'),\n"
        "], ignore_index=True).set_index('index')\n"
        "summary"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## What's next\n\n"
        "- **Slice 2 (0.66.0)** adds `with_draws='basic'|'full'`, the propagation "
        "kwargs on LP / SVAR, and `plot.irf_with_signal_bands`.\n"
        "- **Slice 3 (0.67.0)** adds `attach_calibration(index, layers=...)` with "
        "benchmark / event-panel / survey scores, populating the calibration block "
        "of the same `SignalQualityReport`."
    ))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the builder**

```bash
python tools/make_notebook_R4_01.py
```
Expected: `wrote notebooks/R4_signal_contract/R4_01_schema_demo.ipynb`.

- [ ] **Step 3: Execute the notebook (controller-side, NOT in a subagent)**

⚠️ Per memory: "Long nbconvert: no subagent — notebooks > 5 min must run in the controller's background or as a standalone script; subagents time out on the Monitor-wait pattern." R4_01 is small (~30s) so a foreground run is fine; do it from the user's terminal or a top-level Bash call, not a delegated subagent.

```bash
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/R4_signal_contract/R4_01_schema_demo.ipynb
```
Expected: notebook executes without error; outputs populated. Verify by opening the file and confirming the markdown cells render and the code cells show their printed output / DataFrame.

- [ ] **Step 4: Commit builder + executed notebook together**

```bash
git add tools/make_notebook_R4_01.py notebooks/R4_signal_contract/R4_01_schema_demo.ipynb
git commit -m "feat(0.65.0): R4_01 schema-demo notebook + paired builder"
```

⚠️ Per memory: "Notebooks ↔ builders are paired — patch the live `.ipynb` AND `tools/make_notebook_NN.py` together, else the next builder re-run silently clobbers the edit." Future edits to R4_01 must touch BOTH files in the same commit.

---

## Task 9: Documentation — `docs/SIGNAL_CONTRACT.md` + ARCHITECTURE.md subsection + README quickstart

**Files:**
- Create: `docs/SIGNAL_CONTRACT.md`
- Modify: `ARCHITECTURE.md` (add a subsection after the Result-object standard).
- Modify: `README.md` (insert a 6-line code block near the Quickstart).

- [ ] **Step 1: Create `docs/SIGNAL_CONTRACT.md`**

```markdown
# Signal contract

> Status: **Slice 1 shipped in 0.65.0** (sparsity-only quality).
> Slice 2 (draws + propagation) and Slice 3 (calibration) are tracked
> in `docs/specs/2026-05-26-signal-contract-design.md`.

The signal contract is the per-`RiskIndex` data shape that lets a
downstream LP / SVAR estimator know how reliable an index reading is
and (later) propagate that reliability into IRF bands.

## Schema (Slice 1)

`puremacro.narrative.types.RiskIndex` carries two optional fields:

```python
@dataclass
class RiskIndex:
    name: str
    country: str
    series: pd.Series
    method: str
    corpus: str
    language: str
    normalization: str
    metadata: dict
    quality: SignalQualityReport | None = None   # 0.65.0+
    draws:   pd.DataFrame | None          = None  # 0.66.0+
```

Both default to `None` — every pre-0.65.0 caller behaves identically.

## Opt-in (Slice 1)

```python
from puremacro.narrative.indices import lui

ri = lui(records, country="USA", language="en", with_quality=True)
ri.quality.summary()      # one-row DataFrame: mean_n_docs, mean_doc_length, n_coverage_gaps, ...
```

The same `with_quality=False` kwarg sits on every canonical index in
`puremacro.narrative.indices.__all__`: `epu`, `mpu`, `gpr`, `tone`,
`wui`, `lui`, `ltui`, `ltui_up`, `ltui_down`, `lwui`, `lwui_wage`,
`bbui`, `cboui`, `ep_ui`, `erpui`, `eurlex_ui`, `sotuui`, `bluesky_ui`.

## `SignalQualityReport` (Slice 1 fields populated)

| Field                 | Populated in | Description                                          |
|-----------------------|--------------|------------------------------------------------------|
| `n_docs_per_period`   | 0.65.0       | docs per quarter that fed the kernel                 |
| `avg_doc_length`      | 0.65.0       | mean tokens per doc per quarter                      |
| `coverage_gaps`       | 0.65.0       | quarters with zero docs inside the date range        |
| `kernel_agreement`    | 0.66.0       | mean pairwise corr across kernel draws (Slice 2)     |
| `multilingual_parity` | 0.66.0       | corr between language subsets (Slice 2)              |
| `doc_bootstrap_sd`    | 0.66.0       | per-period sd across doc-bootstrap draws (Slice 2)   |
| `corpus_loo_max_swing`| 0.66.0       | max \|Δ\| across leave-one-corpus draws (Slice 2)    |
| `benchmark_scores`    | 0.67.0       | per-key Pearson/Spearman/RMSE vs. canonical (S3)     |
| `event_panel`         | 0.67.0       | rank-corr + top-decile hit + AUC vs. event panel     |
| `survey_scores`       | 0.67.0       | per-key Pearson/RMSE vs. survey series               |

## Validation

`RiskIndex.__post_init__` validates `draws` (when set):
- `draws.index` must equal `series.index`.
- `draws.columns` must be a 2-level `pd.MultiIndex` named `['source', 'draw_id']`.
- Every value of the `source` level must be in `{'kernel', 'lexicon', 'doc', 'corpus'}`.

## Spec

The architectural spec — covering Slices 1–3 in full — lives at
`docs/specs/2026-05-26-signal-contract-design.md`.
```

- [ ] **Step 2: Add the ARCHITECTURE.md subsection**

In `ARCHITECTURE.md`, find the "Result-object standard" section (search for the string `Result-object standard` or similar). Append immediately after it:

```markdown
### Signal contract (0.65.0+)

`puremacro.narrative.RiskIndex` carries two optional companion fields —
`quality: SignalQualityReport | None` and `draws: pd.DataFrame | None` —
that let downstream estimators see how reliable the index reading is
and (in 0.66.0+) propagate measurement uncertainty into IRF bands.
Both default to `None`; every pre-0.65.0 caller is unaffected. Opt in
via `with_quality=True` (Slice 1) or `with_draws='basic'|'full'`
(Slice 2). The single-page reference is `docs/SIGNAL_CONTRACT.md`.
The full multi-slice spec is `docs/specs/2026-05-26-signal-contract-design.md`.
```

- [ ] **Step 3: Add the README quickstart block**

In `README.md`, find the existing "Quickstart" section (the code-block that demonstrates `cholesky_svar` + `lp_hac`). Append a third example immediately after the existing two:

````markdown
```python
# Opt into per-period signal-quality diagnostics (sparsity + coverage).
from puremacro.narrative.indices import lui
ri = lui(records, country="USA", language="en", with_quality=True)
ri.quality.summary()
# mean_n_docs, mean_doc_length, n_coverage_gaps, ...
```
````

- [ ] **Step 4: Commit**

```bash
git add docs/SIGNAL_CONTRACT.md ARCHITECTURE.md README.md
git commit -m "docs(0.65.0): signal contract reference + ARCHITECTURE + README quickstart"
```

---

## Task 10: Version bump + CHANGELOG entry

**Files:**
- Modify: `pyproject.toml` (`version = "0.65.0"`)
- Modify: `puremacro/__init__.py` (`__version__ = "0.65.0"`)
- Modify: `CHANGELOG.md` (prepend a 0.65.0 section)

- [ ] **Step 1: Add the failing version-snapshot test**

The repo's `tests/test_public_api.py::test_public_api_matches_snapshot` already exists and is marked as a known pre-existing failure in `_phase2_audit_notes.md`. Do NOT regenerate the snapshot in this slice — the new symbols (`SignalQualityReport`, `BenchmarkScore`, `EventPanelScore`, `SurveyScore`) will surface as snapshot diffs, but the right place to update the snapshot is the release-polish PR, not here.

Instead, add a smoke test that the version string updates correctly. Append to `tests/test_signal_contract/test_schema_extension.py`:

```python
def test_puremacro_version_is_065():
    import puremacro
    assert puremacro.__version__ == "0.65.0"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_signal_contract/test_schema_extension.py::test_puremacro_version_is_065 -v
```
Expected: FAIL — current version is `"0.64.0"`.

- [ ] **Step 3: Bump version in `puremacro/__init__.py`**

Find the existing `__version__ = "0.64.0"` line and change it to `__version__ = "0.65.0"`.

- [ ] **Step 4: Bump version in `pyproject.toml`**

Change `version = "0.64.0"` to `version = "0.65.0"` (around line 7).

- [ ] **Step 5: Prepend the CHANGELOG entry**

Insert this block in `CHANGELOG.md` immediately after the top-level `# Changelog` header and the introductory blurb, and BEFORE the `## 0.64.0 (2026-05-26)` section:

```markdown
## 0.65.0 (2026-05-26)

**Signal contract — Slice 1 (schema + sparsity diagnostics).**

### Added
- `puremacro.narrative.types.SignalQualityReport`: sparsity-only Slice-1 fields
  (`n_docs_per_period`, `avg_doc_length`, `coverage_gaps`); Slices 2 / 3 fields
  declared with default `None` / empty dict for forward-compat.
- `puremacro.narrative.types.RiskIndex` gains two optional fields:
  `quality: SignalQualityReport | None` and `draws: pd.DataFrame | None`.
  `__post_init__` validates `draws` (index must equal `series.index`;
  columns must be a 2-level MultiIndex `['source','draw_id']`; source
  tag must be in `{'kernel','lexicon','doc','corpus'}`).
- `puremacro.narrative._signal_quality.compute_sparsity_report`:
  helper that builds the sparsity / coverage fields from a materialised
  records list.
- `with_quality: bool = False` kwarg on every canonical index in
  `puremacro.narrative.indices.__all__` (epu, mpu, gpr, tone, wui, lui,
  ltui, ltui_up, ltui_down, lwui, lwui_wage, bbui, cboui, ep_ui, erpui,
  eurlex_ui, sotuui, bluesky_ui). Default `False` preserves 0.64.0
  behaviour (`ri.quality is None`). Plumbed centrally through
  `aggregate.index_to_quarterly`.
- `notebooks/R4_signal_contract/R4_01_schema_demo.ipynb` + paired builder
  `tools/make_notebook_R4_01.py`.
- `docs/SIGNAL_CONTRACT.md`: single-page reference; ARCHITECTURE.md
  gains a "Signal contract" subsection; README.md quickstart shows the
  `with_quality=True` path.

### Changed
- None of the existing index function call-sites change behaviour; new
  kwargs default to `with_quality=False`.

### Roadmap
- Slice 2 (0.66.0): draws (kernel / lexicon / doc / corpus) + LP / SVAR
  propagation (`signal_draws=`, `signal_propagation=`, `signal_attribution=`)
  on `lp_hac`, `lp_iv`, `panel_lp_dk`, `cholesky_svar`, `proxy_svar`.
- Slice 3 (0.67.0): three-layer calibration (benchmarks, shipped event
  panel, surveys) via `attach_calibration(index, layers=...)`.
- Full spec: `docs/specs/2026-05-26-signal-contract-design.md`.

### Internal
- `BenchmarkScore` / `EventPanelScore` / `SurveyScore` are `Any` placeholders
  in 0.65.0; full dataclass implementations land in 0.67.0.
- `tests/test_signal_contract/` is the home for the contract's tests
  across all three slices.
```

- [ ] **Step 6: Run the version test to verify pass**

```bash
pytest tests/test_signal_contract/test_schema_extension.py::test_puremacro_version_is_065 -v
```
Expected: PASS.

- [ ] **Step 7: Final full-narrative sanity sweep**

```bash
pytest tests/test_signal_contract -v && \
pytest tests/test_pyodide_compat.py -v && \
pytest tests/test_narrative.py tests/test_narrative_indices.py tests/test_narrative_aggregate_kind.py -v
```
Expected: all `tests/test_signal_contract/` tests pass; Pyodide compat passes; narrative tests show no new failures vs. baseline.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml puremacro/__init__.py CHANGELOG.md \
        tests/test_signal_contract/test_schema_extension.py
git commit -m "chore(puremacro): bump to 0.65.0 — signal contract Slice 1 (schema + sparsity)"
```

---

## Done-definition for Slice 1 (0.65.0)

- [ ] `puremacro.narrative.types.SignalQualityReport` ships, Slice-1 fields populated.
- [ ] `puremacro.narrative.types.RiskIndex` ships with optional `quality` and `draws`; existing callers unaffected.
- [ ] `aggregate.index_to_quarterly` accepts `with_quality=` and attaches the report when True.
- [ ] All 18 canonical index functions in `narrative.indices.__all__` accept `with_quality=False`; the cross-index coverage test enforces it.
- [ ] `tests/test_signal_contract/` is green; full narrative suite shows zero new regressions vs. the baseline in `_phase2_audit_notes.md`.
- [ ] `tests/test_pyodide_compat.py` passes — no new forbidden imports.
- [ ] `notebooks/R4_signal_contract/R4_01_schema_demo.ipynb` executes cleanly and is committed alongside its `tools/make_notebook_R4_01.py` builder.
- [ ] `docs/SIGNAL_CONTRACT.md`, ARCHITECTURE.md, README.md updates committed.
- [ ] `pyproject.toml` and `puremacro/__init__.py` at `0.65.0`; CHANGELOG entry written.

## Out of scope for Slice 1 (queued for follow-up plans)

- Slice 2 (0.66.0): draws generation (kernel / lexicon / doc / corpus); propagation kwargs on the five estimators; `plot.irf_with_signal_bands`; notebook R4_02.
- Slice 3 (0.67.0): calibration registries + shipped event panel + `attach_calibration`; notebook R4_03.
- F1 source-coverage expansion, F2 reliability backbone, F3 unified panel builder, S2 interpretation, S4 cross-source synthesis 2.0, T1 cookbook, T2 onboarding (sibling sub-projects from the brainstorm).
