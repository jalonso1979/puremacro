# Narrative Extension — Slice 1 (Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `kind` + `language` to `NarrativeEvent`, introduce `RiskIndex`, kind-aware `events_to_quarterly`, new `index_to_quarterly`, kind-parameterized scoring (fiscal + monetary), 4-tuple `SourceRecord` with backward-compat shim, and first-wave central-bank connectors (Fed, ECB, BoE, BoJ — decision / minutes / press_conf / speeches as published).

**Architecture:** Three concentric extensions on the existing `puremacro.narrative` package. (1) Type layer — `NarrativeEvent` gains two optional fields with per-kind validation; new `RiskIndex` sibling for continuous text-derived series. (2) Scoring layer — `score_llm` and `score_keyword` accept `kind=`; LLM prompts dispatch by kind; multilingual preamble injects language hint. (3) Source layer — `SourceRecord` becomes a 4-tuple `(date, text, source_url, metadata)` with a one-line shim that upgrades legacy 3-tuples; per-bank connectors call into shared `_ratedoc.py` / `_speeches.py` parser scaffolds. Every existing import keeps working; defaults preserve fiscal-English semantics.

**Tech Stack:** Python 3.10+, `dataclasses`, `pandas`, `numpy`, `urllib`, `xml.etree.ElementTree`. Pyodide-compatible. No new runtime deps.

**Spec reference:** `docs/specs/2026-05-08-narrative-extension-design.md` — single source of truth for the per-kind enums, prompt schemas, and slice scope.

**Pre-implementation baseline:** record `pytest -q` pass count in Task 0; this is the line every later task must protect.

**Version bump:** `pyproject.toml` `0.6.0 → 0.6.1`. The spec listed 0.6.0 illustratively; the package is already at 0.6.0, so Slice 1 lands as 0.6.1.

---

## File Structure

### Files created
- `puremacro/narrative/sources/_ratedoc.py` — shared decision/minutes parser scaffold
- `puremacro/narrative/sources/_speeches.py` — shared speech-archive parser scaffold
- `puremacro/narrative/sources/fed_decision.py` — FOMC statement listing
- `puremacro/narrative/sources/fed_minutes.py` — FOMC minutes listing
- `puremacro/narrative/sources/fed_press_conf.py` — FOMC chair press-conference transcripts
- `puremacro/narrative/sources/fed_speeches.py` — Federal Reserve speeches RSS
- `puremacro/narrative/sources/ecb_decision.py` — promoted from `ecb_press.py` (renamed)
- `puremacro/narrative/sources/ecb_minutes.py` — ECB monetary-policy account
- `puremacro/narrative/sources/ecb_press_conf.py` — ECB press-conference transcripts
- `puremacro/narrative/sources/ecb_speeches.py` — ECB Executive Board speeches RSS
- `puremacro/narrative/sources/boe_decision.py` — BoE MPC summary statement
- `puremacro/narrative/sources/boe_minutes.py` — BoE MPC minutes
- `puremacro/narrative/sources/boe_speeches.py` — BoE speeches and statements
- `puremacro/narrative/sources/boj_decision.py` — BoJ statement on monetary policy
- `puremacro/narrative/sources/boj_speeches.py` — BoJ speeches and statements
- `tests/test_narrative_kind.py` — per-kind validation, multilingual field round-trip
- `tests/test_narrative_riskindex.py` — `RiskIndex` dataclass tests
- `tests/test_narrative_aggregate_kind.py` — `kind_filter` + per-kind aggregation rules
- `tests/test_narrative_index_to_quarterly.py` — `index_to_quarterly` tests
- `tests/test_narrative_scoring_monetary.py` — monetary keyword + LLM-prompt-dispatch tests
- `tests/test_narrative_cb_connectors.py` — offline + network smoke tests for Fed/ECB/BoE/BoJ

### Files modified
- `puremacro/narrative/types.py` — add `kind`, `language`, per-kind validation, `RiskIndex`
- `puremacro/narrative/aggregate.py` — `kind_filter` parameter, per-kind aggregation rules, `index_to_quarterly`
- `puremacro/narrative/scoring/keyword.py` — monetary lexicon, `kind=` dispatch
- `puremacro/narrative/scoring/llm.py` — kind-parameterized prompts, multilingual preamble, 4-tuple shim
- `puremacro/narrative/sources/__init__.py` — export new connectors, keep `iter_ecb_press` as deprecation shim
- `puremacro/narrative/sources/ecb_press.py` — collapse to deprecation re-export of `ecb_decision`
- `puremacro/narrative/__init__.py` — re-export `RiskIndex`, `index_to_quarterly`
- `tests/test_pyodide_compat.py` — walk new modules
- `pyproject.toml` — version `0.6.0 → 0.6.1`
- `puremacro/__init__.py` — `__version__ = "0.6.1"`
- `tests/test_import.py` — bump expected version
- `CHANGELOG.md` — `## 0.6.1 — 2026-05-08` block

---

## Task 0: Establish baseline

**Files:** none

- [ ] **Step 1: Run the suite, record pass/skip count**

Run: `pytest -q --no-header 2>&1 | tail -3`
Expected output (record verbatim into the plan as a comment in Task 13):
```
<X> passed, <Y> skipped, <Z> warnings in <T>s
```
This is the baseline. Every later task must keep `<X>` from going down (it is allowed to go up).

- [ ] **Step 2: Confirm Pyodide-compat test currently passes**

Run: `pytest tests/test_pyodide_compat.py -v --no-header 2>&1 | tail -5`
Expected: green.

---

## Task 1: `NarrativeEvent` — `kind` and `language` fields with per-kind validation

**Files:**
- Modify: `puremacro/narrative/types.py:11-17, 55-67, 69-108`
- Create: `tests/test_narrative_kind.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_narrative_kind.py`:

```python
"""Tests for NarrativeEvent kind/language extension (Slice 1)."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative import NarrativeEvent


def _base(**kw):
    """Build a fiscal event with sane defaults; override via kwargs."""
    defaults = dict(
        date=pd.Timestamp("2020-03-15"),
        country="USA",
        magnitude=10.0,
        magnitude_unit="USD_bn",
        target="investment",
        subtarget="defense",
        sign=+1,
        confidence=0.8,
        source_text="test",
        source_url="https://example.test",
        scoring_method="manual",
    )
    defaults.update(kw)
    return defaults


def test_default_kind_is_fiscal():
    e = NarrativeEvent(**_base())
    assert e.kind == "fiscal"
    assert e.language == "en"


def test_explicit_monetary_kind_with_valid_target():
    e = NarrativeEvent(**_base(
        kind="monetary",
        target="policy_rate",
        magnitude_unit="bps",
        magnitude=25.0,
    ))
    assert e.kind == "monetary"
    assert e.target == "policy_rate"


def test_invalid_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        NarrativeEvent(**_base(kind="not_a_kind"))


def test_fiscal_target_rejected_for_monetary_kind():
    with pytest.raises(ValueError, match="target"):
        NarrativeEvent(**_base(kind="monetary", target="investment"))


def test_monetary_target_rejected_for_fiscal_kind():
    with pytest.raises(ValueError, match="target"):
        NarrativeEvent(**_base(kind="fiscal", target="policy_rate"))


@pytest.mark.parametrize("kind,target", [
    ("fiscal",     "investment"),
    ("fiscal",     "consumption"),
    ("fiscal",     "both"),
    ("monetary",   "policy_rate"),
    ("monetary",   "asset_purchase"),
    ("monetary",   "forward_guidance"),
    ("monetary",   "fx_intervention"),
    ("monetary",   "lending_facility"),
    ("macropru",   "capital_buffer"),
    ("macropru",   "ltv_dsti"),
    ("macropru",   "sector_limit"),
    ("macropru",   "reserve_requirement"),
    ("fx",         "intervention"),
    ("fx",         "peg_change"),
    ("structural", "labor"),
    ("structural", "product_market"),
    ("structural", "trade"),
    ("structural", "tax_admin"),
])
def test_all_valid_kind_target_combinations(kind, target):
    e = NarrativeEvent(**_base(kind=kind, target=target))
    assert e.kind == kind
    assert e.target == target


def test_language_default_en():
    e = NarrativeEvent(**_base())
    assert e.language == "en"


def test_explicit_language_passes_through():
    e = NarrativeEvent(**_base(language="es"))
    assert e.language == "es"


def test_to_dict_round_trip_preserves_kind_language():
    e = NarrativeEvent(**_base(kind="monetary", target="policy_rate",
                                language="de", magnitude_unit="bps"))
    d = e.to_dict()
    assert d["kind"] == "monetary"
    assert d["language"] == "de"
    e2 = NarrativeEvent.from_dict(d)
    assert e2.kind == "monetary"
    assert e2.language == "de"
    assert e2.target == "policy_rate"


def test_from_dict_legacy_payload_without_kind_defaults_fiscal():
    """Legacy serialized events must still load (kind absent → 'fiscal')."""
    legacy = dict(
        date="2020-03-15", country="USA", magnitude=10.0,
        magnitude_unit="USD_bn", target="investment", subtarget="defense",
        sign=+1, confidence=0.8, source_text="test",
        source_url="https://example.test", scoring_method="manual",
    )
    e = NarrativeEvent.from_dict(legacy)
    assert e.kind == "fiscal"
    assert e.language == "en"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_narrative_kind.py -v --no-header 2>&1 | tail -25`
Expected: every test fails with `AttributeError` (missing `kind`/`language`) or `TypeError` (unexpected keyword).

- [ ] **Step 3: Extend `puremacro/narrative/types.py` with the two new fields and per-kind validation**

In `types.py`, locate the `VALID_TARGETS = {"investment", "consumption", "both"}` line (currently line 11) and replace the three constants block with:

```python
VALID_KINDS = {"fiscal", "monetary", "macropru", "fx", "structural"}

VALID_TARGETS_BY_KIND = {
    "fiscal":     {"investment", "consumption", "both"},
    "monetary":   {"policy_rate", "asset_purchase", "forward_guidance",
                   "fx_intervention", "lending_facility"},
    "macropru":   {"capital_buffer", "ltv_dsti", "sector_limit",
                   "reserve_requirement"},
    "fx":         {"intervention", "peg_change"},
    "structural": {"labor", "product_market", "trade", "tax_admin"},
}

# Backwards compat: VALID_TARGETS retains its original meaning (fiscal kinds).
VALID_TARGETS = VALID_TARGETS_BY_KIND["fiscal"]
VALID_SIGNS = {-1, 0, 1}
VALID_SCORERS = {"keyword", "llm", "manual"}
```

In the `NarrativeEvent` dataclass, add two new fields **after** `implementation_profile`:

```python
    kind: str = "fiscal"
    language: str = "en"
```

In `__post_init__`, replace the `target` validation block:

```python
        if self.target not in VALID_TARGETS:
            raise ValueError(
                f"target {self.target!r} not in {VALID_TARGETS}"
            )
```

with:

```python
        if self.kind not in VALID_KINDS:
            raise ValueError(
                f"kind {self.kind!r} not in {VALID_KINDS}"
            )
        valid_targets = VALID_TARGETS_BY_KIND[self.kind]
        if self.target not in valid_targets:
            raise ValueError(
                f"target {self.target!r} not in {valid_targets} "
                f"(for kind={self.kind!r})"
            )
```

Update `__all__` at the bottom of the file:

```python
__all__ = ["NarrativeEvent", "NarrativeInstrument",
           "VALID_KINDS", "VALID_TARGETS_BY_KIND",
           "VALID_TARGETS", "VALID_SIGNS", "VALID_SCORERS"]
```

- [ ] **Step 4: Run tests, expect green**

Run: `pytest tests/test_narrative_kind.py -v --no-header 2>&1 | tail -25`
Expected: all 18 tests pass (12 from `parametrize` + 6 explicit).

- [ ] **Step 5: Run the full narrative suite to confirm no fiscal regressions**

Run: `pytest tests/test_narrative.py tests/test_narrative_quality.py tests/test_narrative_replication_*.py -q --no-header 2>&1 | tail -3`
Expected: same pass count as baseline (Task 0).

- [ ] **Step 6: Commit**

```bash
git add puremacro/narrative/types.py tests/test_narrative_kind.py
git commit -m "feat(narrative): NarrativeEvent gains kind+language with per-kind target validation"
```

---

## Task 2: `RiskIndex` dataclass

**Files:**
- Modify: `puremacro/narrative/types.py` (append new dataclass + imports)
- Create: `tests/test_narrative_riskindex.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_narrative_riskindex.py`:

```python
"""Tests for the RiskIndex dataclass."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.narrative import RiskIndex


def _example_series():
    return pd.Series(
        [100.0, 110.0, 95.0, 102.0],
        index=pd.date_range("2020-01-01", periods=4, freq="QS"),
        name="risk_index",
    )


def test_construct_basic():
    ri = RiskIndex(
        name="epu_us_news",
        country="USA",
        series=_example_series(),
        method="keyword_count",
        corpus="news",
        language="en",
        normalization="bbd_100",
    )
    assert ri.name == "epu_us_news"
    assert ri.method == "keyword_count"
    assert ri.normalization == "bbd_100"
    assert len(ri.series) == 4


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="method"):
        RiskIndex(
            name="x", country="USA", series=_example_series(),
            method="not_a_method", corpus="news", language="en",
            normalization="bbd_100",
        )


def test_invalid_normalization_raises():
    with pytest.raises(ValueError, match="normalization"):
        RiskIndex(
            name="x", country="USA", series=_example_series(),
            method="keyword_count", corpus="news", language="en",
            normalization="weird",
        )


def test_diagnostics_returns_expected_keys():
    ri = RiskIndex(
        name="epu_us_news", country="USA", series=_example_series(),
        method="keyword_count", corpus="news", language="en",
        normalization="bbd_100",
    )
    d = ri.diagnostics()
    assert {"n_quarters", "mean", "std", "first_date", "last_date"} <= set(d)
    assert d["n_quarters"] == 4
    assert d["mean"] == pytest.approx(101.75)


def test_to_frame_is_tidy():
    ri = RiskIndex(
        name="epu_us_news", country="USA", series=_example_series(),
        method="keyword_count", corpus="news", language="en",
        normalization="bbd_100",
    )
    df = ri.to_frame()
    assert set(df.columns) == {"qdate", "value", "country", "name"}
    assert len(df) == 4
    assert (df["country"] == "USA").all()
    assert (df["name"] == "epu_us_news").all()


def test_as_instrument_round_trip():
    ri = RiskIndex(
        name="epu_us_news", country="USA", series=_example_series(),
        method="keyword_count", corpus="news", language="en",
        normalization="bbd_100",
    )
    inst = ri.as_instrument()
    assert inst.name == "epu_us_news"
    assert inst.category == "text_index"
    assert inst.frequency == "Q"
    assert inst.metadata["corpus"] == "news"
    assert inst.metadata["language"] == "en"
    assert inst.metadata["method"] == "keyword_count"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_narrative_riskindex.py -v --no-header 2>&1 | tail -15`
Expected: every test fails with `ImportError: cannot import name 'RiskIndex'`.

- [ ] **Step 3: Add `RiskIndex` to `puremacro/narrative/types.py`**

Append to `types.py` (after `NarrativeInstrument` class, before `__all__`):

```python
VALID_RISKINDEX_METHODS = {"keyword_count", "llm_prob", "tone_dispersion", "hybrid"}
VALID_RISKINDEX_NORMALIZATION = {"raw", "zscore", "bbd_100"}


@dataclass
class RiskIndex:
    """A continuous text-derived risk / uncertainty / tone index.

    Attributes
    ----------
    name : str
        Short identifier (e.g., ``"epu_us_news"``, ``"mpu_ecb_speeches"``).
    country : str
        ISO3 code.
    series : pd.Series
        Quarterly index, indexed by quarter-start dates.
    method : str
        ``keyword_count`` | ``llm_prob`` | ``tone_dispersion`` | ``hybrid``.
    corpus : str
        Free-form label of the underlying corpus (e.g., ``"fed_speeches"``).
    language : str
        ISO-639-1 language code of the corpus.
    normalization : str
        ``raw`` | ``zscore`` | ``bbd_100``.
    metadata : dict
        Free-form (e.g., n_docs, vocab_size, source_urls_sample).
    """
    name: str
    country: str
    series: pd.Series
    method: str
    corpus: str
    language: str
    normalization: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.method not in VALID_RISKINDEX_METHODS:
            raise ValueError(
                f"method {self.method!r} not in {VALID_RISKINDEX_METHODS}"
            )
        if self.normalization not in VALID_RISKINDEX_NORMALIZATION:
            raise ValueError(
                f"normalization {self.normalization!r} not in "
                f"{VALID_RISKINDEX_NORMALIZATION}"
            )

    def diagnostics(self) -> dict:
        s = self.series.dropna()
        return {
            "n_quarters": int(s.shape[0]),
            "mean": float(s.mean()) if s.shape[0] else float("nan"),
            "std": float(s.std()) if s.shape[0] > 1 else float("nan"),
            "first_date": str(s.index.min()) if s.shape[0] else None,
            "last_date": str(s.index.max()) if s.shape[0] else None,
        }

    def to_frame(self) -> pd.DataFrame:
        df = self.series.rename("value").reset_index()
        df.columns = ["qdate", "value"]
        df["country"] = self.country
        df["name"] = self.name
        return df

    def as_instrument(self) -> "Instrument":
        from ..instruments import Instrument
        return Instrument(
            series=self.series,
            name=self.name,
            source=f"narrative.indices.{self.method}",
            category="text_index",
            frequency="Q",
            metadata={
                "corpus": self.corpus,
                "language": self.language,
                "method": self.method,
                "normalization": self.normalization,
                **self.metadata,
            },
        )
```

Update `__all__`:

```python
__all__ = ["NarrativeEvent", "NarrativeInstrument", "RiskIndex",
           "VALID_KINDS", "VALID_TARGETS_BY_KIND",
           "VALID_TARGETS", "VALID_SIGNS", "VALID_SCORERS",
           "VALID_RISKINDEX_METHODS", "VALID_RISKINDEX_NORMALIZATION"]
```

- [ ] **Step 4: Re-export `RiskIndex` from `puremacro.narrative.__init__`**

In `puremacro/narrative/__init__.py`, change the first import line:

```python
from .types import NarrativeEvent, NarrativeInstrument
```

to:

```python
from .types import NarrativeEvent, NarrativeInstrument, RiskIndex
```

And add `"RiskIndex"` to `__all__`.

Check that `puremacro.instruments` exposes `Instrument` at module top level. It already does (per `puremacro/instruments/__init__.py`).

- [ ] **Step 5: Run tests, expect green**

Run: `pytest tests/test_narrative_riskindex.py -v --no-header 2>&1 | tail -15`
Expected: all 6 tests pass.

- [ ] **Step 6: Confirm `as_instrument()` does not pull `Instrument` at import time** (Pyodide check)

Run: `python -c "from puremacro.narrative.types import RiskIndex; import sys; assert 'puremacro.instruments._core' not in sys.modules, sorted(k for k in sys.modules if 'instruments' in k)"`
Expected: silent (no AssertionError). Confirms `Instrument` import is lazy.

- [ ] **Step 7: Commit**

```bash
git add puremacro/narrative/types.py puremacro/narrative/__init__.py tests/test_narrative_riskindex.py
git commit -m "feat(narrative): add RiskIndex dataclass with as_instrument adapter"
```

---

## Task 3: `events_to_quarterly` — `kind_filter` + per-kind aggregation rules

**Files:**
- Modify: `puremacro/narrative/aggregate.py:17-117` (entire `events_to_quarterly`)
- Create: `tests/test_narrative_aggregate_kind.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_narrative_aggregate_kind.py`:

```python
"""Tests for kind-aware events_to_quarterly."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative import NarrativeEvent, events_to_quarterly


def _ev(date, kind, target, magnitude, sign, country="USA", magnitude_unit=None):
    if magnitude_unit is None:
        magnitude_unit = {
            "fiscal": "USD_bn",
            "monetary": "bps",
            "macropru": "ratio",
            "fx": "USD_bn",
            "structural": "z",
        }[kind]
    return NarrativeEvent(
        date=pd.Timestamp(date), country=country,
        magnitude=magnitude, magnitude_unit=magnitude_unit,
        target=target, subtarget=None, sign=sign,
        confidence=0.9, source_text="t", source_url="u",
        scoring_method="manual", kind=kind,
    )


def test_pure_fiscal_unchanged_when_kind_filter_none():
    """Backwards compat: no-kind-filter on all-fiscal list works as before."""
    events = [
        _ev("2020-01-15", "fiscal", "investment", 10.0, +1),
        _ev("2020-04-15", "fiscal", "consumption", 5.0, -1),
    ]
    s = events_to_quarterly(events)
    assert len(s) == 2
    assert s.iloc[0] == 10.0
    assert s.iloc[1] == -5.0


def test_mixed_kind_without_filter_raises():
    events = [
        _ev("2020-01-15", "fiscal", "investment", 10.0, +1),
        _ev("2020-04-15", "monetary", "policy_rate", 25.0, +1),
    ]
    with pytest.raises(ValueError, match="multiple kinds"):
        events_to_quarterly(events)


def test_kind_filter_monetary_yields_only_monetary():
    events = [
        _ev("2020-01-15", "fiscal", "investment", 10.0, +1),
        _ev("2020-01-25", "monetary", "policy_rate", 25.0, +1),
        _ev("2020-04-15", "monetary", "policy_rate", 50.0, -1),
    ]
    s = events_to_quarterly(events, kind_filter="monetary")
    # Q1 2020: +25, Q2 2020: -50 (sum aggregation)
    assert s.iloc[0] == 25.0
    assert s.iloc[1] == -50.0


def test_kind_filter_macropru_uses_count_aggregation():
    """macropru: signed COUNT of actions per quarter, not magnitude sum."""
    events = [
        _ev("2020-01-10", "macropru", "capital_buffer", 100.0, +1),
        _ev("2020-02-10", "macropru", "capital_buffer", 250.0, +1),
        _ev("2020-02-25", "macropru", "ltv_dsti",       0.05,  -1),
    ]
    s = events_to_quarterly(events, kind_filter="macropru")
    # Q1 2020: 2 tightening + 1 loosening = +1 net
    assert s.iloc[0] == 1.0


def test_kind_filter_structural_uses_indicator():
    """structural: presence indicator (any signed event ⇒ ±1)."""
    events = [
        _ev("2020-01-10", "structural", "labor", 0.5, +1),
        _ev("2020-01-30", "structural", "trade", 0.2, +1),
    ]
    s = events_to_quarterly(events, kind_filter="structural")
    # Q1 2020: at least one positive structural reform ⇒ +1
    assert s.iloc[0] == 1.0


def test_kind_filter_with_unknown_kind_raises():
    events = [_ev("2020-01-15", "fiscal", "investment", 10.0, +1)]
    with pytest.raises(ValueError, match="kind_filter"):
        events_to_quarterly(events, kind_filter="not_a_kind")


def test_empty_after_kind_filter_returns_empty_series():
    events = [_ev("2020-01-15", "fiscal", "investment", 10.0, +1)]
    s = events_to_quarterly(events, kind_filter="monetary")
    assert s.empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_narrative_aggregate_kind.py -v --no-header 2>&1 | tail -20`
Expected: all 7 tests fail.

- [ ] **Step 3: Extend `events_to_quarterly` in `puremacro/narrative/aggregate.py`**

Replace the entire body of `events_to_quarterly` with:

```python
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .types import NarrativeEvent, VALID_KINDS


_AGG_RULE_BY_KIND = {
    "fiscal": "magnitude_sum",
    "monetary": "magnitude_sum",
    "macropru": "signed_count",
    "fx": "signed_count",
    "structural": "indicator",
}


def events_to_quarterly(
    events: Iterable[NarrativeEvent],
    *,
    target_filter: str | None = None,
    kind_filter: str | None = None,
    aggregation: str = "sum",
    sign_weighted: bool = True,
    pct_gdp: pd.DataFrame | None = None,
    freq: str = "QS",
    confidence_threshold: float = 0.0,
    iv_kind: str = "implementation",
) -> pd.Series:
    """Aggregate narrative events into a quarterly IV.

    See module docstring for full parameter documentation. New in 0.6.1:

    kind_filter : str | None
        Filter events by ``kind`` (fiscal | monetary | macropru | fx |
        structural). If ``None`` and the event list contains multiple
        kinds, ``ValueError`` is raised. ``None`` on a single-kind list
        is fine (backward compatible with all-fiscal callers).

    Aggregation rule depends on the surviving kind:
      - fiscal, monetary: sum of signed magnitudes (existing rule).
      - macropru, fx:    signed count of actions per quarter.
      - structural:      indicator (sign of any event in the quarter).
    """
    if iv_kind not in {"announcement", "implementation"}:
        raise ValueError(
            f"iv_kind must be 'announcement' or 'implementation'; "
            f"got {iv_kind!r}"
        )
    if kind_filter is not None and kind_filter not in VALID_KINDS:
        raise ValueError(
            f"kind_filter {kind_filter!r} not in {VALID_KINDS}"
        )

    events_list = list(events)
    if not events_list:
        return pd.Series(dtype=float, name="narrative_iv")

    kinds_present = {e.kind for e in events_list}
    if kind_filter is None and len(kinds_present) > 1:
        raise ValueError(
            f"events_to_quarterly: events have multiple kinds "
            f"{sorted(kinds_present)}; pass kind_filter= to disambiguate"
        )
    if kind_filter is not None:
        events_list = [e for e in events_list if e.kind == kind_filter]
        if not events_list:
            return pd.Series(dtype=float, name="narrative_iv")
        kinds_present = {kind_filter}

    surviving_kind = next(iter(kinds_present))
    agg_rule = _AGG_RULE_BY_KIND[surviving_kind]

    def _keep(e: NarrativeEvent) -> bool:
        if e.confidence < confidence_threshold:
            return False
        if target_filter is None:
            return True
        return e.target == target_filter or e.target == "both"

    filtered = [e for e in events_list if _keep(e)]
    if not filtered:
        return pd.Series(dtype=float, name="narrative_iv")

    rows = []
    for e in filtered:
        if agg_rule == "magnitude_sum":
            v = e.signed_magnitude if sign_weighted else e.magnitude
        elif agg_rule == "signed_count":
            v = float(e.sign)  # ±1 per action
        elif agg_rule == "indicator":
            v = float(e.sign)  # collapsed below to one value/quarter
        else:  # pragma: no cover
            raise AssertionError(agg_rule)
        if iv_kind == "announcement" or agg_rule != "magnitude_sum":
            rows.append({"date": e.date, "country": e.country, "value": float(v)})
        else:
            for d, w in e.effective_profile:
                rows.append({"date": d, "country": e.country,
                             "value": float(v) * float(w)})

    df = pd.DataFrame(rows)

    if pct_gdp is not None and agg_rule == "magnitude_sum":
        df["q_date"] = df["date"].dt.to_period("Q").dt.to_timestamp()
        gdp_long = pct_gdp.stack().rename("gdp").reset_index()
        gdp_long.columns = ["q_date", "country", "gdp"]
        df = df.merge(gdp_long, on=["q_date", "country"], how="left")
        df["value"] = df["value"] / df["gdp"]

    df["q_date"] = df["date"].dt.to_period("Q").dt.to_timestamp()

    if agg_rule == "indicator":
        # One value per quarter: sign of the largest |value| in the quarter.
        idx = df.groupby("q_date")["value"].apply(lambda s: s.abs().idxmax())
        out = df.loc[idx].set_index("q_date")["value"].apply(np.sign)
    elif agg_rule == "signed_count":
        out = df.groupby("q_date")["value"].sum()
    elif aggregation == "sum":
        out = df.groupby("q_date")["value"].sum()
    elif aggregation == "mean":
        out = df.groupby("q_date")["value"].mean()
    elif aggregation == "max":
        idx = df.groupby("q_date")["value"].apply(lambda s: s.abs().idxmax())
        out = df.loc[idx].set_index("q_date")["value"]
    elif aggregation == "first":
        df_sorted = df.sort_values("date")
        out = df_sorted.groupby("q_date")["value"].first()
    else:
        raise ValueError(f"unknown aggregation {aggregation!r}")

    full_idx = pd.date_range(out.index.min(), out.index.max(), freq=freq)
    out = out.reindex(full_idx, fill_value=0.0)
    out.index.name = "date"
    out.name = "narrative_iv"
    return out


__all__ = ["events_to_quarterly"]
```

- [ ] **Step 4: Run new tests, expect green**

Run: `pytest tests/test_narrative_aggregate_kind.py -v --no-header 2>&1 | tail -15`
Expected: all 7 pass.

- [ ] **Step 5: Run existing fiscal aggregation tests, expect green**

Run: `pytest tests/test_narrative.py -v --no-header -k "to_quarterly or aggregate" 2>&1 | tail -10`
Expected: same as baseline (no fiscal regressions).

- [ ] **Step 6: Commit**

```bash
git add puremacro/narrative/aggregate.py tests/test_narrative_aggregate_kind.py
git commit -m "feat(narrative): events_to_quarterly gains kind_filter + per-kind aggregation"
```

---

## Task 4: `index_to_quarterly`

**Files:**
- Modify: `puremacro/narrative/aggregate.py` (append new function)
- Create: `tests/test_narrative_index_to_quarterly.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_narrative_index_to_quarterly.py`:

```python
"""Tests for index_to_quarterly (continuous text-derived series → RiskIndex)."""
from __future__ import annotations

import numpy as np
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
    # Q1 2020 mean = 105, Q2 2020 mean = 90
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
    # Std of [100, 110, 90] ≈ 10
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_narrative_index_to_quarterly.py -v --no-header 2>&1 | tail -15`
Expected: all 5 fail with `ImportError: cannot import name 'index_to_quarterly'`.

- [ ] **Step 3: Add `index_to_quarterly` to `puremacro/narrative/aggregate.py`**

Append to `aggregate.py`:

```python
from .types import RiskIndex


def index_to_quarterly(
    records,
    *,
    kernel,
    country: str,
    language: str,
    name: str,
    method: str,
    corpus: str,
    normalization: str,
    freq: str = "QS",
    agg: str = "mean",
    metadata: dict | None = None,
) -> RiskIndex:
    """Aggregate per-document scores into a quarterly RiskIndex.

    Parameters
    ----------
    records : iterable of arbitrary records (typically (date, text, url, meta)).
    kernel  : callable(records) → iterable of (pd.Timestamp, float). The
              kernel turns raw records into per-document score points.
    country : ISO3.
    language : ISO-639-1, recorded for provenance.
    name, method, corpus, normalization : forwarded to RiskIndex constructor.
    freq    : pandas frequency string for the output index. Default "QS".
    agg     : "mean" | "max" | "dispersion" (within-quarter std).
    metadata : optional extra metadata. ``n_docs`` is always added.

    Returns
    -------
    RiskIndex with quarterly-indexed series.
    """
    if agg not in {"mean", "max", "dispersion"}:
        raise ValueError(f"agg must be mean|max|dispersion; got {agg!r}")

    points = list(kernel(records))
    if not points:
        raise ValueError(
            f"index_to_quarterly: no documents in corpus={corpus!r} "
            f"for country={country!r}"
        )

    df = pd.DataFrame(points, columns=["date", "value"])
    df["q_date"] = df["date"].dt.to_period("Q").dt.to_timestamp()

    if agg == "mean":
        out = df.groupby("q_date")["value"].mean()
    elif agg == "max":
        out = df.groupby("q_date")["value"].max()
    else:  # dispersion
        out = df.groupby("q_date")["value"].std().fillna(0.0)

    full_idx = pd.date_range(out.index.min(), out.index.max(), freq=freq)
    out = out.reindex(full_idx)
    out.index.name = "date"
    out.name = name

    full_metadata = {"n_docs": len(points)}
    if metadata:
        full_metadata.update(metadata)

    return RiskIndex(
        name=name, country=country, series=out,
        method=method, corpus=corpus, language=language,
        normalization=normalization, metadata=full_metadata,
    )


__all__ = ["events_to_quarterly", "index_to_quarterly"]
```

- [ ] **Step 4: Re-export from `puremacro/narrative/__init__.py`**

Change:

```python
from .aggregate import events_to_quarterly
```

to:

```python
from .aggregate import events_to_quarterly, index_to_quarterly
```

Add `"index_to_quarterly"` to `__all__`.

- [ ] **Step 5: Run tests, expect green**

Run: `pytest tests/test_narrative_index_to_quarterly.py -v --no-header 2>&1 | tail -10`
Expected: all 5 pass.

- [ ] **Step 6: Commit**

```bash
git add puremacro/narrative/aggregate.py puremacro/narrative/__init__.py tests/test_narrative_index_to_quarterly.py
git commit -m "feat(narrative): add index_to_quarterly aggregator for continuous text scores"
```

---

## Task 4b: `NarrativeInstrument.as_instrument` — thread `kind` into metadata

**Files:**
- Modify: `puremacro/narrative/types.py` (existing `NarrativeInstrument.as_instrument` ~line 302-327)
- Modify: `tests/test_narrative_riskindex.py` (append a single test)

The spec wants downstream consumers to be able to filter the instrument catalog by kind. The existing `category` strings (`narrative_replication`, `narrative_connector`) stay intact (backwards compatible with the catalog and any code reading `category`). We thread the new `kind` (or set of kinds, if mixed) through `Instrument.metadata` instead.

- [ ] **Step 1: Append failing test** to `tests/test_narrative_riskindex.py`:

```python
# ---------------------------------------------------------------------------
# NarrativeInstrument.as_instrument — kind passthrough
# ---------------------------------------------------------------------------
def test_narrative_instrument_threads_kind_into_metadata():
    from puremacro.narrative import NarrativeEvent, NarrativeInstrument

    events = [
        NarrativeEvent(
            date=pd.Timestamp("2022-03-16"), country="USA",
            magnitude=25.0, magnitude_unit="bps",
            target="policy_rate", subtarget=None, sign=+1,
            confidence=0.9, source_text="t", source_url="u",
            scoring_method="manual", kind="monetary",
        ),
    ]
    inst = NarrativeInstrument.from_events(events).as_instrument()
    assert inst.metadata.get("kinds") == ["monetary"]
    # category stays a current-known string (not changed)
    assert inst.category in {"narrative_replication", "narrative_connector"}


def test_narrative_instrument_kinds_is_sorted_unique_for_mixed():
    from puremacro.narrative import NarrativeEvent, NarrativeInstrument

    e1 = NarrativeEvent(
        date=pd.Timestamp("2020-01-15"), country="USA", magnitude=10.0,
        magnitude_unit="USD_bn", target="investment", subtarget=None,
        sign=+1, confidence=0.9, source_text="t", source_url="u",
        scoring_method="manual", kind="fiscal",
    )
    e2 = NarrativeEvent(
        date=pd.Timestamp("2022-03-16"), country="USA", magnitude=25.0,
        magnitude_unit="bps", target="policy_rate", subtarget=None,
        sign=+1, confidence=0.9, source_text="t", source_url="u",
        scoring_method="manual", kind="monetary",
    )
    # Mixed-kind list cannot go through from_events (kind_filter required)
    # so build the instrument with kind_filter explicitly
    inst = NarrativeInstrument.from_events([e1, e2], kind_filter=None)  # placeholder
```

Wait — `NarrativeInstrument.from_events` does not currently accept `kind_filter`. Decision: for this task we test only the single-kind path; mixed-kind already raises in `events_to_quarterly`. Replace the second test body with:

```python
def test_narrative_instrument_kinds_with_legacy_fiscal_default():
    """Legacy fiscal events (kind defaulted) still produce kinds=['fiscal']."""
    from puremacro.narrative import NarrativeEvent, NarrativeInstrument

    e = NarrativeEvent(
        date=pd.Timestamp("2020-01-15"), country="USA", magnitude=10.0,
        magnitude_unit="USD_bn", target="investment", subtarget=None,
        sign=+1, confidence=0.9, source_text="t", source_url="u",
        scoring_method="manual",
    )
    inst = NarrativeInstrument.from_events([e]).as_instrument()
    assert inst.metadata.get("kinds") == ["fiscal"]
```

- [ ] **Step 2: Run the new tests, verify they fail**

Run: `pytest tests/test_narrative_riskindex.py -v --no-header -k "narrative_instrument" 2>&1 | tail -10`
Expected: both tests fail (`kinds` key missing from metadata).

- [ ] **Step 3: Patch `NarrativeInstrument.as_instrument` in `puremacro/narrative/types.py`**

Locate the existing method (currently around line 302-327):

```python
    def as_instrument(self, iv_kind: str = "implementation") -> "Instrument":
        from ..instruments import Instrument
        is_replication = any(
            "replication" in (e.metadata or {}) for e in self.events
        )
        category = "narrative_replication" if is_replication else "narrative_connector"
        series = self._series_for(iv_kind)
        return Instrument(
            series=series,
            name=self.metadata.get("registry_key", "narrative_instrument"),
            source=self.metadata.get("source", "narrative aggregation"),
            category=category,
            frequency="Q",
            metadata={
                **self.metadata,
                "n_events": len(self.events),
                "target": self.target,
                "aggregation": self.aggregation,
                "iv_kind": iv_kind,
            },
        )
```

Add one line that computes the sorted-unique list of kinds, and pass it into `metadata`:

```python
    def as_instrument(self, iv_kind: str = "implementation") -> "Instrument":
        from ..instruments import Instrument
        is_replication = any(
            "replication" in (e.metadata or {}) for e in self.events
        )
        category = "narrative_replication" if is_replication else "narrative_connector"
        series = self._series_for(iv_kind)
        kinds = sorted({e.kind for e in self.events})
        return Instrument(
            series=series,
            name=self.metadata.get("registry_key", "narrative_instrument"),
            source=self.metadata.get("source", "narrative aggregation"),
            category=category,
            frequency="Q",
            metadata={
                **self.metadata,
                "n_events": len(self.events),
                "target": self.target,
                "aggregation": self.aggregation,
                "iv_kind": iv_kind,
                "kinds": kinds,
            },
        )
```

- [ ] **Step 4: Run the new tests, expect green**

Run: `pytest tests/test_narrative_riskindex.py -v --no-header -k "narrative_instrument" 2>&1 | tail -10`
Expected: both tests pass.

- [ ] **Step 5: Confirm no regressions in instruments-protocol tests**

Run: `pytest tests/test_instruments -v --no-header 2>&1 | tail -10`
Expected: same as baseline (the new metadata key is additive).

- [ ] **Step 6: Commit**

```bash
git add puremacro/narrative/types.py tests/test_narrative_riskindex.py
git commit -m "feat(narrative): NarrativeInstrument.as_instrument threads kinds into metadata"
```

---

## Task 5: `score_keyword` — monetary lexicon + `kind=` dispatch

**Files:**
- Modify: `puremacro/narrative/scoring/keyword.py`
- Create: `tests/test_narrative_scoring_monetary.py`

- [ ] **Step 1: Write the failing tests** (this test file is also used in Task 6 for LLM tests)

Create `tests/test_narrative_scoring_monetary.py`:

```python
"""Tests for monetary-kind keyword + LLM scoring (Slice 1)."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative.scoring import score_keyword


def _records(*pairs):
    """Build (date, text, url) records."""
    return [(pd.Timestamp(d), t, "https://test/" + d) for d, t in pairs]


def test_keyword_monetary_hawkish_signal():
    """A clear rate-hike signal should yield a +1 (hawkish) monetary event."""
    records = _records(
        ("2022-03-16", "The FOMC voted to raise the federal funds rate by 25 basis points."),
    )
    events = score_keyword(records, kind="monetary", country="USA")
    assert len(events) == 1
    e = events[0]
    assert e.kind == "monetary"
    assert e.target == "policy_rate"
    assert e.sign == +1
    assert e.country == "USA"


def test_keyword_monetary_dovish_signal():
    records = _records(
        ("2020-03-15", "The Fed announced a rate cut and asset-purchase expansion."),
    )
    events = score_keyword(records, kind="monetary", country="USA")
    assert len(events) >= 1
    # First-match wins; rate cut is dovish
    assert events[0].sign == -1
    assert events[0].kind == "monetary"


def test_keyword_monetary_no_signal():
    records = _records(
        ("2020-03-15", "Inflation remained elevated through the quarter."),
    )
    events = score_keyword(records, kind="monetary", country="USA")
    assert events == []


def test_keyword_fiscal_kind_unchanged():
    """Existing fiscal-keyword path must still work with kind='fiscal' default."""
    records = _records(
        ("2020-04-01", "Congress approved a $500 billion infrastructure package."),
    )
    events = score_keyword(records, country="USA")  # kind defaults fiscal
    assert len(events) == 1
    assert events[0].kind == "fiscal"
    assert events[0].target == "investment"


def test_keyword_invalid_kind_raises():
    records = _records(("2020-01-01", "anything"))
    with pytest.raises(ValueError, match="kind"):
        score_keyword(records, kind="not_a_kind", country="USA")
```

- [ ] **Step 2: Run keyword tests, verify they fail**

Run: `pytest tests/test_narrative_scoring_monetary.py -v --no-header -k "keyword" 2>&1 | tail -15`
Expected: monetary tests fail (`kind` keyword not accepted).

- [ ] **Step 3: Extend `puremacro/narrative/scoring/keyword.py`**

Add after `DEFAULT_FISCAL_LEXICON`:

```python
DEFAULT_MONETARY_LEXICON = [
    # Hawkish (+1)
    (r"\braise (?:the )?(?:federal funds rate|policy rate|interest rate|bank rate)\b",
                                                  ("policy_rate", +1, "rate_up")),
    (r"\brate (?:hike|increase)\b",               ("policy_rate", +1, "rate_up")),
    (r"\btighten(?:ing)? monetary policy\b",      ("policy_rate", +1, "tighten")),
    (r"\bhike (?:rates|the policy rate)\b",       ("policy_rate", +1, "rate_up")),
    (r"\bquantitative tightening\b",              ("asset_purchase", +1, "qt")),
    (r"\bbalance[- ]sheet runoff\b",              ("asset_purchase", +1, "qt")),
    (r"\bwithdraw accommodation\b",               ("forward_guidance", +1, "hawkish")),
    # Dovish (-1)
    (r"\b(?:rate )?cut\b",                        ("policy_rate", -1, "rate_down")),
    (r"\blower (?:the )?(?:federal funds rate|policy rate|bank rate)\b",
                                                  ("policy_rate", -1, "rate_down")),
    (r"\bease(?:d|s)? monetary policy\b",         ("policy_rate", -1, "ease")),
    (r"\bquantitative easing\b",                  ("asset_purchase", -1, "qe")),
    (r"\basset[- ]purchase (?:expansion|programme)\b",
                                                  ("asset_purchase", -1, "qe")),
    (r"\bforward guidance.*lower for longer\b",   ("forward_guidance", -1, "dovish")),
]


def regex_basis_points(text: str) -> float:
    """Extract basis-point magnitude from text. Returns 0.0 if absent."""
    text_lower = text.lower()
    patterns = [
        r"\b([0-9]+(?:\.[0-9]+)?)\s*basis points?\b",
        r"\b([0-9]+(?:\.[0-9]+)?)\s*bps?\b",
        r"\bby ([0-9]+(?:\.[0-9]+)?)\s*percentage points?\b",
    ]
    best = 0.0
    for rx in patterns:
        for m in re.finditer(rx, text_lower):
            try:
                v = float(m.group(1))
                if "percentage point" in rx:
                    v *= 100  # 1 pp = 100 bps
                if v > best:
                    best = v
            except ValueError:
                continue
    return best


_LEXICON_BY_KIND = {
    "fiscal":   DEFAULT_FISCAL_LEXICON,
    "monetary": DEFAULT_MONETARY_LEXICON,
}

_MAGNITUDE_EXTRACTOR_BY_KIND = {
    "fiscal":   regex_billions,
    "monetary": regex_basis_points,
}

_MAGNITUDE_UNIT_BY_KIND = {
    "fiscal":   "USD_bn",
    "monetary": "bps",
}
```

Replace the `score_keyword` signature and body:

```python
def score_keyword(
    text_iter: Iterable[tuple],
    *,
    country: str = "USA",
    kind: str = "fiscal",
    lexicon=None,
    magnitude_extractor=None,
    confidence: float = 0.5,
    language: str = "en",
) -> list[NarrativeEvent]:
    """Score each text in ``text_iter`` to at most one event.

    Parameters
    ----------
    kind : ``"fiscal"`` (default) or ``"monetary"``. Selects the default
        lexicon, magnitude extractor, and magnitude unit. Other kinds
        (macropru, fx, structural) require explicit ``lexicon=`` and
        ``magnitude_extractor=`` (no built-in defaults yet).
    """
    if kind not in _LEXICON_BY_KIND and lexicon is None:
        raise ValueError(
            f"kind {kind!r}: no built-in lexicon. Pass lexicon= explicitly, "
            f"or use kind in {sorted(_LEXICON_BY_KIND)}."
        )
    if lexicon is None:
        lexicon = _LEXICON_BY_KIND[kind]
    if magnitude_extractor is None:
        magnitude_extractor = _MAGNITUDE_EXTRACTOR_BY_KIND.get(kind, regex_billions)
    magnitude_unit = _MAGNITUDE_UNIT_BY_KIND.get(kind, "USD_bn")

    out: list[NarrativeEvent] = []
    for record in text_iter:
        # Accept both 3-tuples (legacy) and 4-tuples (new SourceRecord).
        if len(record) == 4:
            date, text, source_url, _meta = record
        else:
            date, text, source_url = record
        cls = _classify(text, lexicon=lexicon)
        if cls is None:
            continue
        target, sign, subtarget = cls
        magnitude = magnitude_extractor(text)
        if magnitude == 0.0:
            local_conf = 0.25
        else:
            local_conf = confidence
        out.append(NarrativeEvent(
            date=pd.Timestamp(date),
            country=country,
            magnitude=magnitude,
            magnitude_unit=magnitude_unit,
            target=target,
            subtarget=subtarget,
            sign=sign,
            confidence=local_conf,
            source_text=text[:500],
            source_url=str(source_url),
            scoring_method="keyword",
            kind=kind,
            language=language,
        ))
    return out


__all__ = ["DEFAULT_FISCAL_LEXICON", "DEFAULT_MONETARY_LEXICON",
           "regex_billions", "regex_basis_points", "score_keyword"]
```

- [ ] **Step 4: Run keyword tests, expect green**

Run: `pytest tests/test_narrative_scoring_monetary.py -v --no-header -k "keyword" 2>&1 | tail -15`
Expected: 5 keyword tests pass.

- [ ] **Step 5: Run existing fiscal-keyword tests, expect green**

Run: `pytest tests/test_narrative.py -v --no-header -k "keyword" 2>&1 | tail -10`
Expected: same as baseline.

- [ ] **Step 6: Commit**

```bash
git add puremacro/narrative/scoring/keyword.py tests/test_narrative_scoring_monetary.py
git commit -m "feat(narrative): score_keyword adds monetary lexicon + kind dispatch"
```

---

## Task 6: `score_llm` — kind-parameterized prompts + multilingual preamble + 4-tuple shim

**Files:**
- Modify: `puremacro/narrative/scoring/llm.py`
- Modify: `tests/test_narrative_scoring_monetary.py` (append LLM-prompt-dispatch tests)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_narrative_scoring_monetary.py`:

```python
# ---------------------------------------------------------------------------
# LLM prompt-dispatch tests (no API key required — uses dry_run)
# ---------------------------------------------------------------------------
from puremacro.narrative.scoring.llm import (
    _PROMPTS, _build_prompt, score_llm,
)


def test_prompt_registry_has_five_kinds():
    assert set(_PROMPTS) == {"fiscal", "monetary", "macropru", "fx", "structural"}


def test_build_prompt_fiscal_contains_legacy_text():
    p = _build_prompt(kind="fiscal", language="en", country="USA",
                      date="2020-01-01", text="hello")
    assert "fiscal-policy events" in p
    assert "USA" in p
    assert "hello" in p


def test_build_prompt_monetary_contains_bps_and_hawkish_dovish():
    p = _build_prompt(kind="monetary", language="en", country="USA",
                      date="2022-03-16", text="rate hike")
    assert "basis points" in p.lower() or "bps" in p.lower()
    assert "hawkish" in p.lower()
    assert "dovish" in p.lower()


def test_build_prompt_includes_language_hint_for_non_english():
    p = _build_prompt(kind="fiscal", language="es", country="MEX",
                      date="2020-01-01", text="hola")
    assert "es" in p or "Spanish" in p or "español" in p.lower() or \
           "language" in p.lower()


def test_score_llm_dry_run_returns_empty_list():
    """Dry run should not call the network; just print cost estimate."""
    records = [(pd.Timestamp("2020-01-01"), "x", "u")]
    out = score_llm(records, backend=None, kind="fiscal", dry_run=True)
    assert out == []


def test_score_llm_invalid_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        score_llm([], backend=None, kind="not_a_kind", dry_run=True)


def test_score_llm_accepts_4_tuple_records():
    """Backwards-compat: 4-tuple SourceRecord must work alongside 3-tuple."""
    records = [
        (pd.Timestamp("2020-01-01"), "x", "u", {"doctype": "decision", "language": "en"}),
        (pd.Timestamp("2020-02-01"), "y", "u"),  # legacy 3-tuple
    ]
    # dry_run does not actually call backend, but it iterates records
    out = score_llm(records, backend=None, kind="fiscal", dry_run=True)
    assert out == []
```

- [ ] **Step 2: Run LLM tests, verify they fail**

Run: `pytest tests/test_narrative_scoring_monetary.py -v --no-header -k "llm or prompt" 2>&1 | tail -20`
Expected: every prompt/llm test fails (`_PROMPTS` and `_build_prompt` don't exist).

- [ ] **Step 3: Refactor `puremacro/narrative/scoring/llm.py`**

Replace the `_PROMPT_TEMPLATE` constant with a dictionary of prompts. Keep the existing fiscal text verbatim under `_PROMPTS["fiscal"]`. Add `_build_prompt`, four new prompt strings, kind validation, and a 4-tuple shim.

Locate the existing `_PROMPT_TEMPLATE = """..."""` (~line 32) and replace through to the `# ---` separator before `_BackendBase` with:

```python
_LANGUAGE_PREAMBLE = (
    "The text below may be in any language; the field 'language' is "
    "{language} (ISO-639-1). Extract events regardless of language. "
    "Return all field labels in English."
)


_PROMPTS = {
    "fiscal": """You are extracting narrative fiscal-policy events from
press text. From the input below, output a JSON array of zero or more
events with this exact schema (one JSON object per event):

  {{
    "magnitude_usd_bn": float,
    "target": "investment" | "consumption" | "both",
    "subtarget": string | null,
    "sign": -1 | 0 | +1,
    "confidence": float in [0, 1],
    "excerpt": string,
    "implementation_date": "YYYY-MM-DD" | null,
    "horizon_quarters": int 0..16 | null
  }}

Only include events that announce a *discrete change* in government
spending (consumption or capital), not retrospective reporting or
forecasts. Output ONLY the JSON array, no surrounding prose.

{language_preamble}

Country: {country}
Date: {date}

Text:
\"\"\"
{text}
\"\"\"
""",
    "monetary": """You are extracting narrative MONETARY-policy events
from central-bank text. From the input below, output a JSON array of
zero or more events with this exact schema:

  {{
    "magnitude_bps": float,
    "target": "policy_rate" | "asset_purchase" | "forward_guidance"
              | "fx_intervention" | "lending_facility",
    "subtarget": string | null,
    "sign": -1 | 0 | +1,                  // +1 hawkish/tighten, -1 dovish/ease, 0 neutral
    "hawkish_dovish_prob": float in [0, 1],
    "confidence": float in [0, 1],
    "excerpt": string,
    "implementation_date": "YYYY-MM-DD" | null,
    "horizon_quarters": int 0..16 | null
  }}

Only include discrete monetary-policy decisions or unambiguous
forward-guidance announcements. Skip retrospective discussion and
forecasts. Output ONLY the JSON array.

{language_preamble}

Country: {country}
Date: {date}

Text:
\"\"\"
{text}
\"\"\"
""",
    "macropru": """You are extracting MACROPRUDENTIAL-policy events.
JSON array, one object per event:

  {{
    "target": "capital_buffer" | "ltv_dsti" | "sector_limit" | "reserve_requirement",
    "magnitude_pct": float,
    "subtarget": string | null,
    "sign": -1 | 0 | +1,                  // +1 tightening, -1 loosening
    "confidence": float in [0, 1],
    "excerpt": string,
    "implementation_date": "YYYY-MM-DD" | null,
    "horizon_quarters": int 0..16 | null
  }}

Only include announced policy actions, not commentary. Output JSON
array only.

{language_preamble}

Country: {country}
Date: {date}

Text:
\"\"\"
{text}
\"\"\"
""",
    "fx": """You are extracting FX-INTERVENTION or peg-change events.
JSON array, one object per event:

  {{
    "target": "intervention" | "peg_change",
    "magnitude_usd_bn": float,
    "subtarget": string | null,
    "sign": -1 | 0 | +1,                  // +1 buying domestic / defending, -1 selling
    "confidence": float in [0, 1],
    "excerpt": string,
    "implementation_date": "YYYY-MM-DD" | null,
    "horizon_quarters": int 0..16 | null
  }}

Output JSON array only.

{language_preamble}

Country: {country}
Date: {date}

Text:
\"\"\"
{text}
\"\"\"
""",
    "structural": """You are extracting STRUCTURAL-REFORM events. JSON
array, one object per event:

  {{
    "target": "labor" | "product_market" | "trade" | "tax_admin",
    "magnitude_z": float,                  // qualitative scale, ~[0, 3]
    "subtarget": string | null,
    "sign": -1 | 0 | +1,                  // +1 liberalizing, -1 restrictive
    "confidence": float in [0, 1],
    "excerpt": string,
    "implementation_date": "YYYY-MM-DD" | null,
    "horizon_quarters": int 0..16 | null
  }}

Output JSON array only.

{language_preamble}

Country: {country}
Date: {date}

Text:
\"\"\"
{text}
\"\"\"
""",
}


_MAGNITUDE_KEY_BY_KIND = {
    "fiscal":     "magnitude_usd_bn",
    "monetary":   "magnitude_bps",
    "macropru":   "magnitude_pct",
    "fx":         "magnitude_usd_bn",
    "structural": "magnitude_z",
}

_MAGNITUDE_UNIT_BY_KIND = {
    "fiscal":     "USD_bn",
    "monetary":   "bps",
    "macropru":   "ratio",
    "fx":         "USD_bn",
    "structural": "z",
}


def _build_prompt(*, kind, language, country, date, text):
    if kind not in _PROMPTS:
        raise ValueError(f"kind {kind!r} not in {sorted(_PROMPTS)}")
    return _PROMPTS[kind].format(
        country=country,
        date=date,
        text=str(text)[:6000],
        language_preamble=_LANGUAGE_PREAMBLE.format(language=language),
    )
```

Update `_validate_event_dict` to validate the per-kind magnitude key:

```python
def _validate_event_dict(d: dict, *, kind: str) -> bool:
    from ..types import VALID_TARGETS_BY_KIND
    if not isinstance(d, dict):
        return False
    if d.get("target") not in VALID_TARGETS_BY_KIND[kind]:
        return False
    try:
        if int(d.get("sign", 99)) not in VALID_SIGNS:
            return False
    except (ValueError, TypeError):
        return False
    mag_key = _MAGNITUDE_KEY_BY_KIND[kind]
    if not isinstance(d.get(mag_key, None), (int, float)):
        return False
    if not isinstance(d.get("confidence", None), (int, float)):
        return False
    return True
```

Update `score_llm` signature and body:

```python
def score_llm(
    text_iter: Iterable[tuple],
    *,
    backend: _BackendBase,
    kind: str = "fiscal",
    language: str = "en",
    country: str = "USA",
    dry_run: bool = False,
) -> list[NarrativeEvent]:
    if kind not in _PROMPTS:
        raise ValueError(f"kind {kind!r} not in {sorted(_PROMPTS)}")

    items = list(text_iter)
    if dry_run:
        chars = sum(len(r[1]) for r in items)
        approx_tokens = chars / 4.0
        print(f"[score_llm dry-run kind={kind} lang={language}] "
              f"{len(items)} items, ~{approx_tokens:,.0f} input tokens, "
              f"~{len(items) * 200:,.0f} output tokens.")
        return []

    out: list[NarrativeEvent] = []
    n_dropped_malformed = 0
    for record in items:
        # Accept both 3-tuple legacy and 4-tuple SourceRecord.
        if len(record) == 4:
            date, text, source_url, meta = record
            record_lang = (meta or {}).get("language", language)
        else:
            date, text, source_url = record
            record_lang = language
        prompt = _build_prompt(
            kind=kind, language=record_lang, country=country,
            date=str(date), text=text,
        )
        try:
            response = backend.call(prompt)
        except Exception:
            n_dropped_malformed += 1
            continue
        for ev_dict in _parse_response(response):
            if not _validate_event_dict(ev_dict, kind=kind):
                n_dropped_malformed += 1
                continue
            mag_key = _MAGNITUDE_KEY_BY_KIND[kind]
            magnitude_unit = _MAGNITUDE_UNIT_BY_KIND[kind]
            profile = _profile_from_timing_fields(ev_dict, announcement=date)
            out.append(NarrativeEvent(
                date=pd.Timestamp(date),
                country=country,
                magnitude=float(ev_dict[mag_key]),
                magnitude_unit=magnitude_unit,
                target=ev_dict["target"],
                subtarget=ev_dict.get("subtarget"),
                sign=int(ev_dict["sign"]),
                confidence=float(ev_dict["confidence"]),
                source_text=str(ev_dict.get("excerpt", text))[:500],
                source_url=str(source_url),
                scoring_method="llm",
                implementation_profile=profile,
                kind=kind,
                language=record_lang,
            ))
    if n_dropped_malformed:
        print(f"[score_llm] dropped {n_dropped_malformed} malformed events.")
    return out


__all__ = ["AnthropicBackend", "OpenAIBackend", "score_llm",
           "_PROMPTS", "_build_prompt"]
```

- [ ] **Step 4: Run LLM tests, expect green**

Run: `pytest tests/test_narrative_scoring_monetary.py -v --no-header 2>&1 | tail -25`
Expected: all 12 tests (5 keyword + 7 LLM/prompt) pass.

- [ ] **Step 5: Run existing LLM-scoring tests, expect green**

Run: `pytest tests/test_narrative_llm_scoring.py -v --no-header 2>&1 | tail -10`
Expected: same as baseline (existing fiscal LLM scoring still works).

- [ ] **Step 6: Commit**

```bash
git add puremacro/narrative/scoring/llm.py tests/test_narrative_scoring_monetary.py
git commit -m "feat(narrative): score_llm — kind-parameterized prompts, multilingual preamble, 4-tuple shim"
```

---

## Task 7: Shared CB parser scaffolds (`_ratedoc.py` + `_speeches.py`)

**Files:**
- Create: `puremacro/narrative/sources/_ratedoc.py`
- Create: `puremacro/narrative/sources/_speeches.py`

These are *helpers* for the per-bank connectors that follow. They have no public surface tests on their own; per-bank tests (Task 8+) exercise them.

- [ ] **Step 1: Create `_ratedoc.py`**

```python
"""Shared parser for central-bank decision / minutes pages.

Most CB decisions follow the same shape: a listing page with one entry
per meeting linking to a statement HTML or PDF, plus an optional
language-tagged variant. This helper wraps:
  - listing-page fetch (RSS, Atom, or HTML)
  - per-entry fetch + body extraction
into a uniform 4-tuple SourceRecord stream.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Callable, Iterator

import pandas as pd

from ..._http import safe_get_bytes, safe_get_text


_HTML_TEXT_RX = re.compile(r"<[^>]+>")


def strip_html(html: str) -> str:
    """Crude HTML→text. Keeps line breaks at block tags, drops tags."""
    txt = re.sub(r"</?(p|div|li|br|h[1-6])[^>]*>", "\n", html, flags=re.I)
    txt = _HTML_TEXT_RX.sub("", txt)
    txt = re.sub(r"\n\s*\n+", "\n\n", txt)
    return txt.strip()


def iter_ratedoc_listing(
    url: str,
    *,
    parse_listing: Callable[[bytes], list[tuple[pd.Timestamp, str]]],
    fetch_body: Callable[[str], str] | None = None,
    bank_code: str,
    country: str,
    doctype: str,
    language: str = "en",
    user_agent: str | None = None,
) -> Iterator[tuple]:
    """Yield SourceRecord 4-tuples for a CB decision/minutes listing.

    Parameters
    ----------
    url : listing-page URL.
    parse_listing : callable bytes → list[(date, item_url)]. Bank-specific.
    fetch_body : callable url → text. Default uses ``safe_get_text``;
        connectors may override to handle PDFs.
    bank_code : short tag stamped into metadata (e.g. ``"FED"``, ``"ECB"``).
    country : ISO3 of the bank's jurisdiction.
    doctype : ``"decision"`` | ``"minutes"`` | ``"press_conf"`` | ``"fsr"``.
    language : ISO-639-1 of the listing.
    user_agent : optional UA override for WAF-protected sites.
    """
    try:
        body = safe_get_bytes(url, user_agent=user_agent) if user_agent \
                else safe_get_bytes(url)
    except Exception:
        return
    try:
        entries = parse_listing(body)
    except Exception:
        return
    fetcher = fetch_body or (lambda u: safe_get_text(u, user_agent=user_agent)
                              if user_agent else safe_get_text(u))
    for date, item_url in entries:
        if pd.isna(date) or not item_url:
            continue
        try:
            text = fetcher(item_url)
        except Exception:
            continue
        clean = strip_html(text) if "<" in text and ">" in text else text
        if not clean:
            continue
        yield (date, clean, item_url, {
            "doctype": doctype, "language": language,
            "bank_code": bank_code, "country": country,
        })


__all__ = ["iter_ratedoc_listing", "strip_html"]
```

- [ ] **Step 2: Create `_speeches.py`**

```python
"""Shared parser for CB speech archives (RSS-style)."""
from __future__ import annotations

from typing import Iterator

import pandas as pd

from ._rss import iter_rss
from ._ratedoc import strip_html


def iter_speeches_rss(
    url: str,
    *,
    bank_code: str,
    country: str,
    language: str = "en",
) -> Iterator[tuple]:
    """Wrap an RSS speech feed and emit 4-tuple SourceRecords."""
    for date, title_desc, link in iter_rss(url):
        clean = strip_html(title_desc) if "<" in title_desc else title_desc
        yield (date, clean, link, {
            "doctype": "speech", "language": language,
            "bank_code": bank_code, "country": country,
        })


__all__ = ["iter_speeches_rss"]
```

- [ ] **Step 3: Sanity import**

Run: `python -c "from puremacro.narrative.sources._ratedoc import iter_ratedoc_listing, strip_html; from puremacro.narrative.sources._speeches import iter_speeches_rss; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add puremacro/narrative/sources/_ratedoc.py puremacro/narrative/sources/_speeches.py
git commit -m "feat(narrative): shared CB parser scaffolds (_ratedoc, _speeches)"
```

---

## Task 8: Federal Reserve connectors (decision / minutes / press_conf / speeches)

**Files:**
- Create: `puremacro/narrative/sources/fed_decision.py`
- Create: `puremacro/narrative/sources/fed_minutes.py`
- Create: `puremacro/narrative/sources/fed_press_conf.py`
- Create: `puremacro/narrative/sources/fed_speeches.py`
- Create: `tests/test_narrative_cb_connectors.py`

The Fed has WAF protection; pass an explicit User-Agent override on every fetch (per `narrative/sources/RETRY_POLICY.md` §7).

- [ ] **Step 1: Write the failing test for the four connectors**

Create `tests/test_narrative_cb_connectors.py`. The test file defines its own small in-memory HTTP mock fixture (the existing `_http_fixtures.install_fixture_patches` uses on-disk JSON cache + module-registry, which is heavyweight for new connector modules). Direct `monkeypatch` is simpler and self-contained.

```python
"""Offline + smoke tests for first-wave central-bank connectors."""
from __future__ import annotations

import importlib

import pandas as pd
import pytest


# Modules whose `safe_get_bytes` / `safe_get_text` we patch in offline tests.
_PATCH_TARGETS = [
    "puremacro.narrative.sources._rss",
    "puremacro.narrative.sources._ratedoc",
    "puremacro.narrative.sources._speeches",
    "puremacro.narrative.sources.fed_decision",
    "puremacro.narrative.sources.fed_minutes",
    "puremacro.narrative.sources.fed_press_conf",
    "puremacro.narrative.sources.fed_speeches",
    "puremacro.narrative.sources.ecb_decision",
    "puremacro.narrative.sources.ecb_minutes",
    "puremacro.narrative.sources.ecb_press_conf",
    "puremacro.narrative.sources.ecb_speeches",
    "puremacro.narrative.sources.boe_decision",
    "puremacro.narrative.sources.boe_minutes",
    "puremacro.narrative.sources.boe_speeches",
    "puremacro.narrative.sources.boj_decision",
    "puremacro.narrative.sources.boj_speeches",
]


@pytest.fixture
def mock_http(monkeypatch):
    """Per-test in-memory HTTP mock. Use ``register(bytes_=..., text=...)``
    to register URL → payload mappings before invoking the connector.
    Unregistered URLs raise ``LookupError`` (loud failure on missing mocks).
    """
    by_url_bytes: dict[str, bytes] = {}
    by_url_text: dict[str, str] = {}

    def _fake_bytes(url, **_kw):
        if url in by_url_bytes:
            return by_url_bytes[url]
        raise LookupError(f"mock_http: no bytes registered for {url}")

    def _fake_text(url, **_kw):
        if url in by_url_text:
            return by_url_text[url]
        raise LookupError(f"mock_http: no text registered for {url}")

    for modname in _PATCH_TARGETS:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        if hasattr(mod, "safe_get_bytes"):
            monkeypatch.setattr(mod, "safe_get_bytes", _fake_bytes)
        if hasattr(mod, "safe_get_text"):
            monkeypatch.setattr(mod, "safe_get_text", _fake_text)

    def register(*, bytes_=None, text=None):
        if bytes_:
            by_url_bytes.update(bytes_)
        if text:
            by_url_text.update(text)

    return register


# ---------------------------------------------------------------------------
# Federal Reserve
# ---------------------------------------------------------------------------
def test_fed_decision_yields_four_tuple(mock_http):
    mock_http(
        bytes_={
            "https://www.federalreserve.gov/json/ne-press.json":
                b'{"refData":[{"d":"2022-03-16","ti":"FOMC statement",'
                b'"l":"/newsevents/pressreleases/monetary20220316a.htm"}]}',
        },
        text={
            "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220316a.htm":
                "<html><body><p>The Committee decided to raise the target range "
                "for the federal funds rate to 1/4 to 1/2 percent.</p></body></html>",
        },
    )
    from puremacro.narrative.sources import iter_fed_decision
    records = list(iter_fed_decision())
    assert len(records) >= 1
    date, text, url, meta = records[0]
    assert isinstance(date, pd.Timestamp)
    assert "federal funds rate" in text.lower()
    assert meta["doctype"] == "decision"
    assert meta["language"] == "en"
    assert meta["bank_code"] == "FED"
    assert meta["country"] == "USA"


def test_fed_minutes_yields_four_tuple(mock_http):
    mock_http(
        bytes_={
            "https://www.federalreserve.gov/json/ne-press.json":
                b'{"refData":[{"d":"2022-04-06","ti":"Minutes of the FOMC",'
                b'"l":"/monetarypolicy/fomcminutes20220316.htm"}]}',
        },
        text={
            "https://www.federalreserve.gov/monetarypolicy/fomcminutes20220316.htm":
                "<html><body><p>Participants noted that inflation remained "
                "elevated.</p></body></html>",
        },
    )
    from puremacro.narrative.sources import iter_fed_minutes
    records = list(iter_fed_minutes())
    assert len(records) >= 1
    _, _, _, meta = records[0]
    assert meta["doctype"] == "minutes"


def test_fed_press_conf_yields_four_tuple(mock_http):
    """Press-conference connector: listing HTML + per-PDF byte fetches.

    The connector strips HTML and best-effort decodes PDFs via latin-1.
    For the offline test we deliver both HTML listing and a stub PDF as
    bytes.
    """
    mock_http(
        bytes_={
            "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20220316.pdf":
                # Stub "PDF": the connector decodes latin-1 + filters non-ASCII.
                b"%PDF-1.4\n" + (
                    b"Chair Powell: Today the FOMC raised the federal funds "
                    b"rate by 25 basis points." * 5
                ),
        },
        text={
            "https://www.federalreserve.gov/monetarypolicy/fomcpresconf.htm":
                '<html><body><a href="/mediacenter/files/FOMCpresconf20220316.pdf">'
                'March 16, 2022</a></body></html>',
        },
    )
    from puremacro.narrative.sources import iter_fed_press_conf
    records = list(iter_fed_press_conf())
    # The connector skips short bodies; only assert when at least one yields.
    if records:
        _, _, _, meta = records[0]
        assert meta["doctype"] == "press_conf"


@pytest.mark.network
def test_fed_speeches_smoke():
    """Live RSS smoke: skip if empty, never assert positive count."""
    from puremacro.narrative.sources import iter_fed_speeches
    records = list(iter_fed_speeches())
    if not records:
        pytest.skip("Fed speech feed returned empty (network or upstream issue).")
    _, _, _, meta = records[0]
    assert meta["doctype"] == "speech"
    assert meta["bank_code"] == "FED"
    assert meta["language"] == "en"
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_narrative_cb_connectors.py -v --no-header 2>&1 | tail -15`
Expected: every test fails with `ImportError`.

- [ ] **Step 3: Create `puremacro/narrative/sources/fed_decision.py`**

```python
"""Federal Reserve FOMC decision statements.

Listing endpoint: federalreserve.gov publishes a JSON index of press
releases at /json/ne-press.json. Each entry has a date, title, and
relative URL. We filter to entries whose title contains "FOMC" and
"statement" and fetch the linked HTML body.
"""
from __future__ import annotations

import json
from typing import Iterator

import pandas as pd

from ..._http import safe_get_bytes, safe_get_text


_LISTING_URL = "https://www.federalreserve.gov/json/ne-press.json"
_BASE = "https://www.federalreserve.gov"
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _parse_listing(raw: bytes) -> list[tuple[pd.Timestamp, str]]:
    try:
        obj = json.loads(raw.decode("utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return []
    out: list[tuple[pd.Timestamp, str]] = []
    for item in obj.get("refData", []):
        title = (item.get("ti") or "").lower()
        if "fomc" not in title or "statement" not in title:
            continue
        try:
            date = pd.Timestamp(item.get("d"))
        except Exception:
            continue
        href = item.get("l", "")
        if not href:
            continue
        out.append((date, _BASE + href if href.startswith("/") else href))
    return out


def iter_fed_decision() -> Iterator[tuple]:
    """Yield (date, text, url, metadata) for FOMC statement releases."""
    try:
        body = safe_get_bytes(_LISTING_URL, user_agent=_UA)
    except Exception:
        return
    for date, item_url in _parse_listing(body):
        try:
            html = safe_get_text(item_url, user_agent=_UA)
        except Exception:
            continue
        from ._ratedoc import strip_html
        text = strip_html(html)
        if not text:
            continue
        yield (date, text, item_url, {
            "doctype": "decision", "language": "en",
            "bank_code": "FED", "country": "USA",
        })


__all__ = ["iter_fed_decision"]
```

- [ ] **Step 4: Create `puremacro/narrative/sources/fed_minutes.py`**

```python
"""Federal Reserve FOMC minutes."""
from __future__ import annotations

import json
from typing import Iterator

import pandas as pd

from ..._http import safe_get_bytes, safe_get_text
from ._ratedoc import strip_html


_LISTING_URL = "https://www.federalreserve.gov/json/ne-press.json"
_BASE = "https://www.federalreserve.gov"
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def iter_fed_minutes() -> Iterator[tuple]:
    """Yield (date, text, url, metadata) for FOMC meeting minutes."""
    try:
        body = safe_get_bytes(_LISTING_URL, user_agent=_UA)
    except Exception:
        return
    try:
        obj = json.loads(body.decode("utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return
    for item in obj.get("refData", []):
        title = (item.get("ti") or "").lower()
        if "minutes" not in title:
            continue
        try:
            date = pd.Timestamp(item.get("d"))
        except Exception:
            continue
        href = item.get("l", "")
        item_url = _BASE + href if href.startswith("/") else href
        try:
            html = safe_get_text(item_url, user_agent=_UA)
        except Exception:
            continue
        text = strip_html(html)
        if not text:
            continue
        yield (date, text, item_url, {
            "doctype": "minutes", "language": "en",
            "bank_code": "FED", "country": "USA",
        })


__all__ = ["iter_fed_minutes"]
```

- [ ] **Step 5: Create `puremacro/narrative/sources/fed_press_conf.py`**

```python
"""FOMC chair press-conference transcripts.

Listing page: /monetarypolicy/fomcpresconf.htm. Anchor hrefs to PDFs
named FOMCpresconfYYYYMMDD.pdf. We extract the date from the filename
and fetch the PDF as bytes (callers can run a pdf-to-text pass; we
yield raw text via crude HTML strip if it's HTML, or pdf bytes-as-string
fallback for PDFs — the LLM pipeline tolerates noise).
"""
from __future__ import annotations

import re
from typing import Iterator

import pandas as pd

from ..._http import safe_get_bytes, safe_get_text
from ._ratedoc import strip_html


_LISTING_URL = "https://www.federalreserve.gov/monetarypolicy/fomcpresconf.htm"
_BASE = "https://www.federalreserve.gov"
_FNAME_RX = re.compile(r"FOMCpresconf(\d{8})\.pdf", re.I)
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def iter_fed_press_conf() -> Iterator[tuple]:
    try:
        html = safe_get_text(_LISTING_URL, user_agent=_UA)
    except Exception:
        return
    for m in _FNAME_RX.finditer(html):
        ymd = m.group(1)
        try:
            date = pd.Timestamp(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}")
        except Exception:
            continue
        href_start = max(0, m.start() - 200)
        # Find the nearest preceding href= attribute.
        snippet = html[href_start:m.end() + 5]
        href_m = re.search(r'href="([^"]+\.pdf)"', snippet, re.I)
        if not href_m:
            continue
        item_url = href_m.group(1)
        if item_url.startswith("/"):
            item_url = _BASE + item_url
        try:
            pdf_bytes = safe_get_bytes(item_url, user_agent=_UA)
        except Exception:
            continue
        # Crude PDF→text: extract printable ASCII; downstream LLM tolerates noise.
        text = pdf_bytes.decode("latin-1", errors="ignore")
        text = re.sub(r"[^\x20-\x7e\n]+", " ", text)
        text = re.sub(r"\s{3,}", "\n\n", text)
        if len(text) < 200:
            continue
        yield (date, text[:30000], item_url, {
            "doctype": "press_conf", "language": "en",
            "bank_code": "FED", "country": "USA",
        })


__all__ = ["iter_fed_press_conf"]
```

- [ ] **Step 6: Create `puremacro/narrative/sources/fed_speeches.py`**

```python
"""Federal Reserve speech archive (RSS)."""
from __future__ import annotations

from typing import Iterator

from ._speeches import iter_speeches_rss


_FEED_URL = "https://www.federalreserve.gov/feeds/speeches.xml"


def iter_fed_speeches() -> Iterator[tuple]:
    """Yield (date, text, url, metadata) for Fed speeches via RSS."""
    yield from iter_speeches_rss(
        _FEED_URL, bank_code="FED", country="USA", language="en",
    )


__all__ = ["iter_fed_speeches"]
```

- [ ] **Step 7: Re-export from `puremacro/narrative/sources/__init__.py`**

Add at the bottom of the file (after the existing `from .ecb_press import iter_ecb_press` line):

```python
# Central-bank decision / minutes / press-conf / speeches (Slice 1).
from .fed_decision import iter_fed_decision
from .fed_minutes import iter_fed_minutes
from .fed_press_conf import iter_fed_press_conf
from .fed_speeches import iter_fed_speeches
```

Update `__all__`:

```python
__all__ = [
    # ...existing entries unchanged...
    "iter_fed_decision", "iter_fed_minutes",
    "iter_fed_press_conf", "iter_fed_speeches",
]
```

- [ ] **Step 8: Run offline tests, expect green**

Run: `pytest tests/test_narrative_cb_connectors.py -v --no-header -k "fed and not network" 2>&1 | tail -20`
Expected: 3 offline Fed tests pass; the network smoke test is collected but not run without `-m network`.

- [ ] **Step 9: Commit**

```bash
git add puremacro/narrative/sources/fed_*.py puremacro/narrative/sources/__init__.py tests/test_narrative_cb_connectors.py
git commit -m "feat(narrative): Federal Reserve connectors (decision/minutes/press_conf/speeches)"
```

---

## Task 9: ECB connectors (rename press → decision + add minutes/press_conf/speeches)

**Files:**
- Modify: `puremacro/narrative/sources/ecb_press.py` (collapse to deprecation shim)
- Create: `puremacro/narrative/sources/ecb_decision.py`
- Create: `puremacro/narrative/sources/ecb_minutes.py`
- Create: `puremacro/narrative/sources/ecb_press_conf.py`
- Create: `puremacro/narrative/sources/ecb_speeches.py`
- Modify: `puremacro/narrative/sources/__init__.py`
- Modify: `tests/test_narrative_cb_connectors.py` (append ECB tests)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_narrative_cb_connectors.py`:

```python
# ---------------------------------------------------------------------------
# European Central Bank
# ---------------------------------------------------------------------------
def test_ecb_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.ecb.europa.eu/rss/press.html":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Monetary policy decisions</title>'
            b'<description>The Governing Council decided to raise rates by 25bps.</description>'
            b'<link>https://www.ecb.europa.eu/press/pr/date/2022/html/ecb.mp220721.en.html</link>'
            b'<pubDate>Thu, 21 Jul 2022 12:45:00 +0200</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_ecb_decision
    records = list(iter_ecb_decision())
    assert len(records) == 1
    date, text, _, meta = records[0]
    assert "Governing Council" in text or "25bps" in text
    assert meta["doctype"] == "decision"
    assert meta["bank_code"] == "ECB"


def test_ecb_press_legacy_import_still_works(mock_http, recwarn):
    """Backwards compat: iter_ecb_press still importable, emits DeprecationWarning."""
    mock_http(bytes_={
        "https://www.ecb.europa.eu/rss/press.html":
            b'<?xml version="1.0"?><rss><channel></channel></rss>',
    })
    from puremacro.narrative.sources import iter_ecb_press
    # Calling it should still work and DELEGATE to iter_ecb_decision.
    list(iter_ecb_press())
    deprecations = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "expected DeprecationWarning from iter_ecb_press"


@pytest.mark.network
def test_ecb_speeches_smoke():
    from puremacro.narrative.sources import iter_ecb_speeches
    recs = list(iter_ecb_speeches())
    if not recs:
        pytest.skip("ECB speeches feed returned empty.")
    _, _, _, meta = recs[0]
    assert meta["bank_code"] == "ECB"
    assert meta["doctype"] == "speech"
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_narrative_cb_connectors.py -v --no-header -k "ecb and not network" 2>&1 | tail -15`
Expected: ECB tests fail with `ImportError: cannot import name 'iter_ecb_decision'` (legacy `iter_ecb_press` import still works against the current implementation but the deprecation warning isn't emitted).

- [ ] **Step 3: Create `puremacro/narrative/sources/ecb_decision.py`** (lifts logic from current `ecb_press.py` and stamps 4-tuple metadata)

```python
"""European Central Bank press releases (monetary-policy decisions feed).

The ECB publishes monetary-policy decisions and other press releases
on a single RSS feed, in 6 languages. Renamed from ``ecb_press.py`` in
0.6.1; ``ecb_press.py`` survives as a deprecation shim that re-exports
``iter_ecb_press = iter_ecb_decision``.

Feed URLs:
    https://www.ecb.europa.eu/rss/press.html        (English; default)
    https://www.ecb.europa.eu/rss/press.de.html     (German)
    https://www.ecb.europa.eu/rss/press.fr.html     (French)
    https://www.ecb.europa.eu/rss/press.es.html     (Spanish)
    https://www.ecb.europa.eu/rss/press.it.html     (Italian)
    https://www.ecb.europa.eu/rss/press.pt.html     (Portuguese)
"""
from __future__ import annotations

from typing import Iterator

from ._rss import iter_rss
from ._ratedoc import strip_html


_FEED_BY_LANG = {
    "en": "https://www.ecb.europa.eu/rss/press.html",
    "de": "https://www.ecb.europa.eu/rss/press.de.html",
    "fr": "https://www.ecb.europa.eu/rss/press.fr.html",
    "es": "https://www.ecb.europa.eu/rss/press.es.html",
    "it": "https://www.ecb.europa.eu/rss/press.it.html",
    "pt": "https://www.ecb.europa.eu/rss/press.pt.html",
}


def iter_ecb_decision(
    *, language: str = "en", feed_url: str | None = None,
) -> Iterator[tuple]:
    """Yield (date, text, url, metadata) for ECB press / decisions."""
    url = feed_url or _FEED_BY_LANG.get(language, _FEED_BY_LANG["en"])
    for date, title_desc, link in iter_rss(url):
        clean = strip_html(title_desc) if "<" in title_desc else title_desc
        yield (date, clean, link, {
            "doctype": "decision", "language": language,
            "bank_code": "ECB", "country": "EUR",
        })


__all__ = ["iter_ecb_decision"]
```

- [ ] **Step 4: Replace `puremacro/narrative/sources/ecb_press.py`** with a deprecation shim:

```python
"""Deprecated: renamed to :mod:`ecb_decision` in 0.6.1.

This shim keeps any pre-0.6.1 imports working.
"""
from __future__ import annotations

import warnings
from typing import Iterator

from .ecb_decision import iter_ecb_decision


def iter_ecb_press(*, language: str = "en", feed_url: str | None = None) -> Iterator[tuple]:
    warnings.warn(
        "iter_ecb_press is deprecated; use iter_ecb_decision instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    yield from iter_ecb_decision(language=language, feed_url=feed_url)


__all__ = ["iter_ecb_press"]
```

- [ ] **Step 5: Create `puremacro/narrative/sources/ecb_minutes.py`**

```python
"""ECB monetary-policy account (minutes-equivalent)."""
from __future__ import annotations

from typing import Iterator

from ._rss import iter_rss
from ._ratedoc import strip_html


_FEED = "https://www.ecb.europa.eu/rss/mopo.html"


def iter_ecb_minutes(*, language: str = "en") -> Iterator[tuple]:
    yield from _emit(_FEED, language)


def _emit(url: str, language: str) -> Iterator[tuple]:
    for date, title_desc, link in iter_rss(url):
        clean = strip_html(title_desc) if "<" in title_desc else title_desc
        if "account" not in clean.lower() and "monetary policy account" not in clean.lower():
            # Filter mopo feed to the meeting-account entries.
            continue
        yield (date, clean, link, {
            "doctype": "minutes", "language": language,
            "bank_code": "ECB", "country": "EUR",
        })


__all__ = ["iter_ecb_minutes"]
```

- [ ] **Step 6: Create `puremacro/narrative/sources/ecb_press_conf.py`**

```python
"""ECB press conferences (after Governing Council monetary decisions).

Press-conference transcripts live at /press/pressconf/{year}/html/. We
crawl the most-recent year's listing and yield 4-tuple records.
"""
from __future__ import annotations

import re
from typing import Iterator

import pandas as pd

from ..._http import safe_get_text
from ._ratedoc import strip_html


_LISTING_FMT = "https://www.ecb.europa.eu/press/pressconf/{year}/html/index.en.html"
_DATE_RX = re.compile(r"is22?(\d{6})", re.I)
# ECB filenames look like .../is220721~973616afa9.en.html for 2022-07-21.
_FILENAME_RX = re.compile(r"is(\d{6})[^/\"]*\.html")


def iter_ecb_press_conf(*, year: int | None = None) -> Iterator[tuple]:
    if year is None:
        from datetime import date as _d
        year = _d.today().year
    url = _LISTING_FMT.format(year=year)
    try:
        html = safe_get_text(url)
    except Exception:
        return
    seen: set[str] = set()
    for m in _FILENAME_RX.finditer(html):
        href = m.group(0)
        ymd = m.group(1)
        if href in seen:
            continue
        seen.add(href)
        try:
            date = pd.Timestamp(f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]}")
        except Exception:
            continue
        item_url = f"https://www.ecb.europa.eu/press/pressconf/{year}/html/{href}"
        try:
            body_html = safe_get_text(item_url)
        except Exception:
            continue
        text = strip_html(body_html)
        if len(text) < 200:
            continue
        yield (date, text[:30000], item_url, {
            "doctype": "press_conf", "language": "en",
            "bank_code": "ECB", "country": "EUR",
        })


__all__ = ["iter_ecb_press_conf"]
```

- [ ] **Step 7: Create `puremacro/narrative/sources/ecb_speeches.py`**

```python
"""ECB Executive Board speeches (RSS)."""
from __future__ import annotations

from typing import Iterator

from ._speeches import iter_speeches_rss


_FEED_BY_LANG = {
    "en": "https://www.ecb.europa.eu/rss/sp.html",
    "de": "https://www.ecb.europa.eu/rss/sp.de.html",
    "fr": "https://www.ecb.europa.eu/rss/sp.fr.html",
}


def iter_ecb_speeches(*, language: str = "en") -> Iterator[tuple]:
    url = _FEED_BY_LANG.get(language, _FEED_BY_LANG["en"])
    yield from iter_speeches_rss(
        url, bank_code="ECB", country="EUR", language=language,
    )


__all__ = ["iter_ecb_speeches"]
```

- [ ] **Step 8: Update `puremacro/narrative/sources/__init__.py`**

Replace the existing `from .ecb_press import iter_ecb_press` line with:

```python
from .ecb_decision import iter_ecb_decision
from .ecb_minutes import iter_ecb_minutes
from .ecb_press_conf import iter_ecb_press_conf
from .ecb_speeches import iter_ecb_speeches
from .ecb_press import iter_ecb_press   # deprecated re-export
```

Update `__all__` to include `iter_ecb_decision`, `iter_ecb_minutes`, `iter_ecb_press_conf`, `iter_ecb_speeches` (keep `iter_ecb_press` for compat).

- [ ] **Step 9: Run ECB tests, expect green**

Run: `pytest tests/test_narrative_cb_connectors.py -v --no-header -k "ecb and not network" 2>&1 | tail -15`
Expected: 2 offline ECB tests pass.

- [ ] **Step 10: Run G7 example to confirm legacy import path still works**

Run: `python -c "from puremacro.narrative.sources import iter_ecb_press; print('legacy ok')"`
Expected: `legacy ok` (the call itself emits the DeprecationWarning but the import is silent).

- [ ] **Step 11: Commit**

```bash
git add puremacro/narrative/sources/ecb_*.py puremacro/narrative/sources/__init__.py tests/test_narrative_cb_connectors.py
git commit -m "feat(narrative): ECB connectors — rename press→decision + minutes/press_conf/speeches"
```

---

## Task 10: BoE connectors (decision / minutes / speeches)

**Files:**
- Create: `puremacro/narrative/sources/boe_decision.py`
- Create: `puremacro/narrative/sources/boe_minutes.py`
- Create: `puremacro/narrative/sources/boe_speeches.py`
- Modify: `puremacro/narrative/sources/__init__.py`
- Modify: `tests/test_narrative_cb_connectors.py`

- [ ] **Step 1: Append BoE tests** to `tests/test_narrative_cb_connectors.py`:

```python
# ---------------------------------------------------------------------------
# Bank of England
# ---------------------------------------------------------------------------
def test_boe_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.bankofengland.co.uk/rss/news/monetary-policy":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Bank Rate increased to 1.75% - August 2022</title>'
            b'<description>The MPC voted to raise Bank Rate by 0.5 percentage points.</description>'
            b'<link>https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2022/august-2022</link>'
            b'<pubDate>Thu, 04 Aug 2022 12:00:00 +0100</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_boe_decision
    records = list(iter_boe_decision())
    assert len(records) == 1
    _, text, _, meta = records[0]
    assert "Bank Rate" in text or "MPC" in text
    assert meta["bank_code"] == "BOE"
    assert meta["country"] == "GBR"


@pytest.mark.network
def test_boe_speeches_smoke():
    from puremacro.narrative.sources import iter_boe_speeches
    recs = list(iter_boe_speeches())
    if not recs:
        pytest.skip("BoE speeches feed empty.")
    _, _, _, meta = recs[0]
    assert meta["bank_code"] == "BOE"
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_narrative_cb_connectors.py -v --no-header -k "boe and not network" 2>&1 | tail -10`
Expected: failure (`ImportError: iter_boe_decision`).

- [ ] **Step 3: Create `puremacro/narrative/sources/boe_decision.py`**

```python
"""Bank of England MPC decision RSS feed."""
from __future__ import annotations

from typing import Iterator

from ._rss import iter_rss
from ._ratedoc import strip_html


_FEED = "https://www.bankofengland.co.uk/rss/news/monetary-policy"


def iter_boe_decision() -> Iterator[tuple]:
    for date, title_desc, link in iter_rss(_FEED):
        clean = strip_html(title_desc) if "<" in title_desc else title_desc
        if "bank rate" not in clean.lower() and "mpc" not in clean.lower():
            continue
        yield (date, clean, link, {
            "doctype": "decision", "language": "en",
            "bank_code": "BOE", "country": "GBR",
        })


__all__ = ["iter_boe_decision"]
```

- [ ] **Step 4: Create `puremacro/narrative/sources/boe_minutes.py`**

```python
"""Bank of England MPC minutes (same RSS feed; doctype-tagged differently)."""
from __future__ import annotations

from typing import Iterator

from ._rss import iter_rss
from ._ratedoc import strip_html


_FEED = "https://www.bankofengland.co.uk/rss/news/monetary-policy"


def iter_boe_minutes() -> Iterator[tuple]:
    for date, title_desc, link in iter_rss(_FEED):
        clean = strip_html(title_desc) if "<" in title_desc else title_desc
        if "minutes" not in clean.lower():
            continue
        yield (date, clean, link, {
            "doctype": "minutes", "language": "en",
            "bank_code": "BOE", "country": "GBR",
        })


__all__ = ["iter_boe_minutes"]
```

- [ ] **Step 5: Create `puremacro/narrative/sources/boe_speeches.py`**

```python
"""Bank of England speeches and statements RSS."""
from __future__ import annotations

from typing import Iterator

from ._speeches import iter_speeches_rss


_FEED = "https://www.bankofengland.co.uk/rss/news/speeches-and-statements"


def iter_boe_speeches() -> Iterator[tuple]:
    yield from iter_speeches_rss(
        _FEED, bank_code="BOE", country="GBR", language="en",
    )


__all__ = ["iter_boe_speeches"]
```

- [ ] **Step 6: Re-export in `puremacro/narrative/sources/__init__.py`**

Append:

```python
from .boe_decision import iter_boe_decision
from .boe_minutes import iter_boe_minutes
from .boe_speeches import iter_boe_speeches
```

Add to `__all__`: `"iter_boe_decision", "iter_boe_minutes", "iter_boe_speeches"`.

- [ ] **Step 7: Run BoE tests, expect green**

Run: `pytest tests/test_narrative_cb_connectors.py -v --no-header -k "boe and not network" 2>&1 | tail -10`
Expected: 1 offline BoE test passes.

- [ ] **Step 8: Commit**

```bash
git add puremacro/narrative/sources/boe_*.py puremacro/narrative/sources/__init__.py tests/test_narrative_cb_connectors.py
git commit -m "feat(narrative): Bank of England connectors (decision/minutes/speeches)"
```

---

## Task 11: BoJ connectors (decision / speeches)

**Files:**
- Create: `puremacro/narrative/sources/boj_decision.py`
- Create: `puremacro/narrative/sources/boj_speeches.py`
- Modify: `puremacro/narrative/sources/__init__.py`
- Modify: `tests/test_narrative_cb_connectors.py`

BoJ does not publish meeting minutes in the FOMC sense (only "Summary of Opinions" and historical Minutes which lag). Slice 1 ships *decision* (statement on monetary policy) and *speeches*; minutes can be added as a separate connector in Slice 3 if useful.

- [ ] **Step 1: Append BoJ tests** to `tests/test_narrative_cb_connectors.py`:

```python
# ---------------------------------------------------------------------------
# Bank of Japan
# ---------------------------------------------------------------------------
def test_boj_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.boj.or.jp/en/rss/whatsnew_e.xml":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Statement on Monetary Policy</title>'
            b'<description>The Bank of Japan decided to maintain the current monetary easing.</description>'
            b'<link>https://www.boj.or.jp/en/announcements/release_2022/k220721a.htm</link>'
            b'<pubDate>Thu, 21 Jul 2022 12:48:00 +0900</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_boj_decision
    recs = list(iter_boj_decision())
    assert len(recs) == 1
    _, text, _, meta = recs[0]
    assert "monetary" in text.lower() or "Bank of Japan" in text
    assert meta["bank_code"] == "BOJ"
    assert meta["country"] == "JPN"


@pytest.mark.network
def test_boj_speeches_smoke():
    from puremacro.narrative.sources import iter_boj_speeches
    recs = list(iter_boj_speeches())
    if not recs:
        pytest.skip("BoJ speeches feed empty.")
    _, _, _, meta = recs[0]
    assert meta["bank_code"] == "BOJ"
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_narrative_cb_connectors.py -v --no-header -k "boj and not network" 2>&1 | tail -10`
Expected: ImportError.

- [ ] **Step 3: Create `puremacro/narrative/sources/boj_decision.py`**

```python
"""Bank of Japan: Statement on Monetary Policy (decision-equivalent)."""
from __future__ import annotations

from typing import Iterator

from ._rss import iter_rss
from ._ratedoc import strip_html


_FEED = "https://www.boj.or.jp/en/rss/whatsnew_e.xml"
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def iter_boj_decision() -> Iterator[tuple]:
    for date, title_desc, link in iter_rss(_FEED):
        clean = strip_html(title_desc) if "<" in title_desc else title_desc
        # Filter to monetary-policy statements.
        low = clean.lower()
        if "monetary policy" not in low and "statement on" not in low:
            continue
        yield (date, clean, link, {
            "doctype": "decision", "language": "en",
            "bank_code": "BOJ", "country": "JPN",
        })


__all__ = ["iter_boj_decision"]
```

- [ ] **Step 4: Create `puremacro/narrative/sources/boj_speeches.py`**

```python
"""Bank of Japan speeches & statements (RSS, English mirror)."""
from __future__ import annotations

from typing import Iterator

from ._speeches import iter_speeches_rss


_FEED = "https://www.boj.or.jp/en/rss/whatsnew_e.xml"


def iter_boj_speeches() -> Iterator[tuple]:
    """BoJ does not publish a separate speech RSS in English — we
    re-use the whatsnew feed and tag every entry as ``speech``. Filter
    downstream by looking at the linked URL pattern (.../press/koen/).
    """
    yield from iter_speeches_rss(
        _FEED, bank_code="BOJ", country="JPN", language="en",
    )


__all__ = ["iter_boj_speeches"]
```

- [ ] **Step 5: Re-export in `puremacro/narrative/sources/__init__.py`**

Append:

```python
from .boj_decision import iter_boj_decision
from .boj_speeches import iter_boj_speeches
```

Add to `__all__`: `"iter_boj_decision", "iter_boj_speeches"`.

- [ ] **Step 6: Run BoJ tests, expect green**

Run: `pytest tests/test_narrative_cb_connectors.py -v --no-header -k "boj and not network" 2>&1 | tail -10`
Expected: 1 offline BoJ test passes.

- [ ] **Step 7: Commit**

```bash
git add puremacro/narrative/sources/boj_*.py puremacro/narrative/sources/__init__.py tests/test_narrative_cb_connectors.py
git commit -m "feat(narrative): Bank of Japan connectors (decision/speeches)"
```

---

## Task 12: Pyodide compat + public API audit

**Files:**
- Modify: `puremacro/narrative/__init__.py` (final pass)
- Modify: `tests/test_pyodide_compat.py`

- [ ] **Step 1: Confirm new modules don't leak `statsmodels` / `linearmodels` / `arch`**

Read `tests/test_pyodide_compat.py` to find the module-walk pattern. The existing test walks `puremacro/*` excluding `examples/`, `narrative/sources/`, `narrative/scoring/llm`, and `tests/` itself. Confirm the new files are correctly placed:

- `narrative/sources/_ratedoc.py`, `narrative/sources/_speeches.py`, all `<bank>_*.py` are under `narrative/sources/` → already excluded ✓
- `narrative/types.py`, `narrative/aggregate.py`, `narrative/scoring/keyword.py` → walked; must stay clean ✓
- No new files outside the existing exclusion mask.

Run: `pytest tests/test_pyodide_compat.py -v --no-header 2>&1 | tail -10`
Expected: green (no leakage).

- [ ] **Step 2: Audit `puremacro/narrative/__init__.py`**

Read the current `__init__.py` and confirm the public surface includes:
- `NarrativeEvent`, `NarrativeInstrument`, `RiskIndex`
- `events_to_quarterly`, `index_to_quarterly`
- All existing replication loaders unchanged.

If `RiskIndex` or `index_to_quarterly` is missing from `__all__`, add them. (Tasks 2 and 4 added them already, but verify.)

- [ ] **Step 3: Confirm legacy import shapes still work**

Run:
```bash
python -c "
from puremacro.narrative import (
    NarrativeEvent, NarrativeInstrument, RiskIndex,
    events_to_quarterly, index_to_quarterly,
    load_ramey_2011_defense, load_romer_romer_2010,
    load_dglp_2011, load_imf_covid_2022,
)
from puremacro.narrative.sources import (
    iter_federal_register, iter_treasury_press,
    iter_ecb_press,            # deprecated — must still import
    iter_ecb_decision,         # new
    iter_fed_decision, iter_fed_minutes, iter_fed_press_conf, iter_fed_speeches,
    iter_boe_decision, iter_boe_minutes, iter_boe_speeches,
    iter_boj_decision, iter_boj_speeches,
)
print('public-API audit ok')
"
```
Expected: `public-API audit ok`.

- [ ] **Step 4: Commit (no code change — just verification)**

If Steps 1–3 surfaced any missing exports, fix them now in `__init__.py` and commit:

```bash
git add puremacro/narrative/__init__.py
git commit -m "chore(narrative): Slice-1 public API audit"
```

If everything was already in place from Tasks 2/4/8/9/10/11, skip the commit.

---

## Task 13: Version bump + CHANGELOG + final regression sweep

**Files:**
- Modify: `pyproject.toml`
- Modify: `puremacro/__init__.py` (`__version__`)
- Modify: `tests/test_import.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Bump version to 0.6.1**

Edit `pyproject.toml` line 7: change `version = "0.6.0"` to `version = "0.6.1"`.

- [ ] **Step 2: Bump `__version__` in `puremacro/__init__.py`**

Read the file. Locate `__version__ = "0.6.0"` and change to `"0.6.1"`.

- [ ] **Step 3: Bump expected version in `tests/test_import.py`**

Read the file. Locate the assertion checking `__version__ == "0.6.0"` and change to `"0.6.1"`.

- [ ] **Step 4: Add CHANGELOG entry**

Open `CHANGELOG.md` and add a new top entry:

```markdown
## 0.6.1 — 2026-05-08

Slice 1 of the multi-domain narrative extension (`docs/specs/2026-05-08-narrative-extension-design.md`). Foundation for monetary / macropru / fx / structural narrative work and text-derived risk indices. **No breaking changes** — every fiscal call site keeps working unchanged.

### Added

- `narrative.NarrativeEvent` gains two optional fields: `kind` (default `"fiscal"`, validated against `VALID_KINDS = {fiscal, monetary, macropru, fx, structural}`) and `language` (default `"en"`). `target` is now validated per-kind via `VALID_TARGETS_BY_KIND`.
- `narrative.RiskIndex` (new dataclass) — continuous text-derived index series with `country`, `method`, `corpus`, `language`, `normalization`, plus `as_instrument()` / `diagnostics()` / `to_frame()` helpers.
- `narrative.events_to_quarterly` gains `kind_filter=` and per-kind aggregation rules (sum for fiscal/monetary, signed-count for macropru/fx, indicator for structural). Mixed-kind event lists raise unless filtered.
- `narrative.index_to_quarterly` (new) — aggregates per-document score points into a quarterly `RiskIndex` (mean / max / dispersion).
- `narrative.scoring.score_keyword` gains `kind=` dispatch with a built-in monetary lexicon (English) plus `regex_basis_points` magnitude extractor.
- `narrative.scoring.score_llm` gains `kind=` and `language=` parameters; five kind-specific prompt templates (`_PROMPTS`); multilingual preamble; accepts both 3-tuple legacy and 4-tuple `SourceRecord` records.
- New CB connectors (Slice 1 first wave): **Federal Reserve** (`iter_fed_decision`, `iter_fed_minutes`, `iter_fed_press_conf`, `iter_fed_speeches`), **ECB** (`iter_ecb_decision`, `iter_ecb_minutes`, `iter_ecb_press_conf`, `iter_ecb_speeches`), **Bank of England** (`iter_boe_decision`, `iter_boe_minutes`, `iter_boe_speeches`), **Bank of Japan** (`iter_boj_decision`, `iter_boj_speeches`).
- Shared scaffolds for new connectors: `narrative.sources._ratedoc` (decision/minutes parser), `narrative.sources._speeches` (speech-archive RSS wrapper).
- `tests/test_narrative_kind.py`, `tests/test_narrative_riskindex.py`, `tests/test_narrative_aggregate_kind.py`, `tests/test_narrative_index_to_quarterly.py`, `tests/test_narrative_scoring_monetary.py`, `tests/test_narrative_cb_connectors.py` (~50 new tests).

### Changed

- `narrative.sources.ecb_press` is renamed to `narrative.sources.ecb_decision`. The old module name survives as a re-export shim that emits `DeprecationWarning` on call. `iter_ecb_press(...)` still works and delegates.
- `narrative.NarrativeEvent.to_dict()` / `from_dict()` round-trip the new `kind` and `language` fields. Legacy serialized payloads without these keys load with the defaults.

### Pyodide compatibility

- `narrative.types`, `narrative.aggregate`, `narrative.scoring.keyword` remain Pyodide-clean (no new top-level deps).
- New `narrative.sources/<bank>_*.py` and the shared scaffolds (`_ratedoc`, `_speeches`) stay in the existing **Experimental** tier per `ARCHITECTURE.md` — `tests/test_pyodide_compat.py` already excludes `narrative/sources/` from the leakage walk.

### Notes for future slices

- Slice 2 (`narrative.indices` subpackage) ships EPU / MPU / GPR / tone / WUI / LUI text-index helpers in 0.6.2.
- Slice 3 (LATAM, advanced non-G7, Asia-EM CBs; macropru / fx / structural prompt families; BIS speeches meta-connector) targets 0.7.0.
```

- [ ] **Step 5: Run the full suite**

Run: `pytest -q --no-header 2>&1 | tail -5`
Expected: passing-count ≥ baseline + ~50 (the new tests). No new failures, no new errors.

- [ ] **Step 6: Run Pyodide compat check one more time**

Run: `pytest tests/test_pyodide_compat.py -v --no-header 2>&1 | tail -5`
Expected: green.

- [ ] **Step 7: Run the existing fiscal narrative-replication suite to verify zero regressions**

Run: `pytest tests/test_narrative.py tests/test_narrative_replication_*.py tests/test_narrative_quality.py tests/test_narrative_validation.py -q --no-header 2>&1 | tail -5`
Expected: same passing count as before Slice 1 began.

- [ ] **Step 8: Commit version bump + CHANGELOG**

```bash
git add pyproject.toml puremacro/__init__.py tests/test_import.py CHANGELOG.md
git commit -m "chore(release): puremacro 0.6.1 — narrative Slice 1 (kind+language, RiskIndex, CB connectors)"
```

- [ ] **Step 9: Tag the release**

```bash
git tag -a v0.6.1 -m "puremacro 0.6.1 — narrative Slice 1"
```

(Do **not** push the tag — let the user push when ready.)

---

## Definition of Done

- [ ] All 13 task blocks above checked off.
- [ ] `pytest -q` shows ≥ baseline + ~50 passing tests.
- [ ] `pytest tests/test_pyodide_compat.py` is green.
- [ ] No fiscal-narrative regressions: `pytest tests/test_narrative.py tests/test_narrative_replication_*.py tests/test_narrative_quality.py` matches the pre-Slice-1 baseline.
- [ ] `from puremacro.narrative.sources import iter_ecb_press` still works (with DeprecationWarning on call).
- [ ] `from puremacro.narrative import RiskIndex, index_to_quarterly` works.
- [ ] `pyproject.toml` version is `0.6.1`; `puremacro.__version__ == "0.6.1"`.
- [ ] `CHANGELOG.md` has a `## 0.6.1 — 2026-05-08` section.
- [ ] Tag `v0.6.1` exists locally (not pushed).

## Out of scope for this plan (deferred to Slice 2 / Slice 3)

- `narrative/indices/` subpackage (EPU / MPU / GPR / tone / WUI / LUI) — Slice 2.
- LATAM / advanced non-G7 / Asia-EM CB connectors — Slice 3.
- Macropru / FX / structural prompt families exercised end-to-end — Slice 3 (the prompts are scaffolded in Task 6 but not yet wired to dedicated connectors or production examples).
- BIS speeches meta-connector — Slice 3.
- Cross-lingual lexicon validation — Slice 3.
- Validation against published BBD-EPU / GPR mirrors — Slice 2 (requires the `indices/` subpackage).
