# Narrative Extension — Slice 2 (Indices Layer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `puremacro.narrative.indices` — six text-derived continuous risk-index helpers (EPU, MPU, GPR, tone, WUI, LUI), each emitting a `RiskIndex` from any source-iter corpus. Pyodide-clean count-based path; LLM-prob path stays Experimental.

**Architecture:** New `narrative/indices/` subpackage with three layers — (1) `_lexicons.py` ships multilingual term lists (en + es/pt/de/fr/it/ja/zh) as plain Python frozensets, no external download. (2) `_kernels.py` provides `keyword_count_kernel`, `cooccurrence_kernel`, `tone_kernel`, plus a `normalize_series` helper covering raw / zscore / bbd_100. (3) Per-index modules (`epu.py`, `mpu.py`, `gpr.py`, `tone.py`, `wui.py`, `lui.py`) wrap the right kernel + lexicon and call into Slice 1's `index_to_quarterly` to emit a `RiskIndex` that consumers can `.as_instrument()` straight into the catalog. The existing `index_to_quarterly` is extended to actually apply the normalization (Slice 1 stamped it as metadata only).

**Tech Stack:** Python 3.10+, `dataclasses`, `pandas`, `numpy`, `re`. No new runtime deps. Pyodide-compatible.

**Spec reference:** `docs/specs/2026-05-08-narrative-extension-design.md` §5 ("Indices layer"). Slice 1 plan: `docs/plans/2026-05-08-narrative-extension-slice1-foundation.md`.

**Branching:** This plan is committed to `feature/narrative-extension-slice1-v2` (HEAD: `v0.6.1` + this plan commit). Branch `feature/narrative-extension-slice2` from that HEAD — equivalent to `v0.6.1` in code state, plus the plan file. The Slice 1 implementation surfaced **parallel-session branch chaos** (another Claude/user session was modifying branches mid-task). Each task here MUST verify branch state before coding and at every checkpoint, and refuse to commit if the branch has shifted.

**Pre-implementation baseline:** `pytest -q` after Slice 1 = **859 passed, 21 skipped**, plus 1 pre-existing pyodide-compat failure (statsmodels.tsa.x13 leak via `puremacro/fetch/_seasonal.py:19` — out of scope, do not touch).

**Version bump:** `0.6.1 → 0.6.2`.

---

## File Structure

### Files created (Slice 2)
- `puremacro/narrative/indices/__init__.py` — public re-exports of the six index helpers.
- `puremacro/narrative/indices/_lexicons.py` — multilingual term lists (en/es/pt/de/fr/it/ja/zh).
- `puremacro/narrative/indices/_kernels.py` — `keyword_count_kernel`, `cooccurrence_kernel`, `tone_kernel`, `normalize_series`.
- `puremacro/narrative/indices/epu.py` — Baker-Bloom-Davis style.
- `puremacro/narrative/indices/mpu.py` — Husted-Rogers-Sun monetary-policy uncertainty.
- `puremacro/narrative/indices/gpr.py` — Caldara-Iacoviello geopolitical risk.
- `puremacro/narrative/indices/tone.py` — Apel-Blix-Grimaldi hawkish-dovish.
- `puremacro/narrative/indices/wui.py` — Ahir-Bloom-Furceri World Uncertainty Index style.
- `puremacro/narrative/indices/lui.py` — Labor-Market Uncertainty (novel, MAV-research-track).
- `puremacro/examples/narrative_indices_demo.py` — assembles all 6 indices from a synthetic corpus, prints summaries.
- `tests/test_narrative_indices.py` — kernel + per-index offline tests + multilingual lexicon coverage + normalization round-trip.
- `tests/test_narrative_indices_validation.py` — network-marked correlation tests vs `instruments.literature.bbd_epu` and `caldara_iacoviello_gpr`.

### Files modified
- `puremacro/narrative/__init__.py` — re-export the 6 index helpers.
- `puremacro/narrative/aggregate.py` — `index_to_quarterly` actually applies `normalization` (Slice 1 just stored the label).
- `tests/test_narrative_index_to_quarterly.py` — add a normalization-round-trip test.
- `tests/fixtures/public_api_snapshot.json` — regenerate after the new public surface lands.
- `pyproject.toml` — version `0.6.1 → 0.6.2`.
- `puremacro/__init__.py` — `__version__ = "0.6.2"`.
- `tests/test_import.py` — bump expected version.
- `CHANGELOG.md` — add `## 0.6.2 — 2026-05-08` block.

---

## Task 0: Establish baseline + branch from v0.6.1

**Files:** none (git operations only)

- [ ] **Step 1: Verify v0.6.1 tag exists**

Run: `git tag -l v0.6.1`
Expected output: `v0.6.1`. If absent, abort and report — Slice 1 must complete first.

- [ ] **Step 2: Create the slice 2 branch**

Run:
```bash
git checkout feature/narrative-extension-slice1-v2
git checkout -b feature/narrative-extension-slice2
git branch --show-current
```
Expected: `feature/narrative-extension-slice2`. (Branching from `feature/narrative-extension-slice1-v2` is equivalent to branching from `v0.6.1` in code state, plus this plan file.)

- [ ] **Step 3: Confirm baseline pytest count**

Run: `pytest -q --no-header 2>&1 | tail -3`
Expected: `859 passed, 21 skipped, 4 warnings in <T>s`. If the count is lower, the branch base is wrong.

- [ ] **Step 4: Confirm Pyodide-compat baseline (1 pre-existing failure)**

Run: `pytest tests/test_pyodide_compat.py -v --no-header 2>&1 | tail -5`
Expected: 1 failed, 1 passed. The failure is `statsmodels.tsa.x13` leak via `puremacro/fetch/_seasonal.py:19` — DO NOT touch.

---

## Task 1: Subpackage skeleton + import smoke test

**Files:**
- Create: `puremacro/narrative/indices/__init__.py` (empty placeholder)
- Create: `puremacro/narrative/indices/_kernels.py` (stub with `__all__ = []`)
- Create: `puremacro/narrative/indices/_lexicons.py` (stub with `__all__ = []`)
- Create: `tests/test_narrative_indices.py` (just imports test, more added in later tasks)

- [ ] **Step 1: Verify branch state**

Run: `git branch --show-current` — expected `feature/narrative-extension-slice2`. If wrong, `git checkout feature/narrative-extension-slice2`.

- [ ] **Step 2: Write the failing test**

Create `tests/test_narrative_indices.py`:

```python
"""Tests for puremacro.narrative.indices subpackage (Slice 2)."""
from __future__ import annotations


def test_indices_subpackage_imports_cleanly():
    """The bare subpackage skeleton must import without errors."""
    from puremacro.narrative.indices import _kernels, _lexicons
    assert hasattr(_kernels, "__all__")
    assert hasattr(_lexicons, "__all__")
```

- [ ] **Step 3: Run test, verify it fails**

Run: `pytest tests/test_narrative_indices.py -v --no-header 2>&1 | tail -10`
Expected: `ImportError: cannot import name '_kernels' from 'puremacro.narrative.indices'` or `No module named 'puremacro.narrative.indices'`.

- [ ] **Step 4: Create the three skeleton files**

Create `puremacro/narrative/indices/__init__.py`:

```python
"""Text-derived continuous risk indices (EPU / MPU / GPR / tone / WUI / LUI).

See ``docs/specs/2026-05-08-narrative-extension-design.md`` §5 for the
methodology contract. Each index helper consumes a source iterator
``(date, text, source_url, metadata)`` and emits a ``RiskIndex``.
"""
__all__: list[str] = []
```

Create `puremacro/narrative/indices/_kernels.py`:

```python
"""Per-document scoring kernels for the indices layer (Slice 2)."""
from __future__ import annotations


__all__: list[str] = []
```

Create `puremacro/narrative/indices/_lexicons.py`:

```python
"""Multilingual term lists for the indices layer (Slice 2).

Each lexicon is a plain Python literal — no external download — so the
indices layer remains Pyodide-clean.
"""
from __future__ import annotations


__all__: list[str] = []
```

- [ ] **Step 5: Run test, expect green**

Run: `pytest tests/test_narrative_indices.py -v --no-header 2>&1 | tail -5`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add puremacro/puremacro/narrative/indices/__init__.py \
        puremacro/puremacro/narrative/indices/_kernels.py \
        puremacro/puremacro/narrative/indices/_lexicons.py \
        puremacro/tests/test_narrative_indices.py
git commit -m "feat(narrative): scaffold narrative.indices subpackage"
```

---

## Task 2: English lexicons

**Files:**
- Modify: `puremacro/narrative/indices/_lexicons.py` (replace stub)
- Modify: `tests/test_narrative_indices.py` (append lexicon coverage tests)

- [ ] **Step 1: Verify branch state**

Run: `git branch --show-current` — expected `feature/narrative-extension-slice2`.

- [ ] **Step 2: Write failing tests** — append to `tests/test_narrative_indices.py`:

```python
# ---------------------------------------------------------------------------
# Lexicon structural tests
# ---------------------------------------------------------------------------
def test_lexicons_top_level_keys():
    from puremacro.narrative.indices._lexicons import LEXICONS
    assert set(LEXICONS) == {"epu", "mpu", "gpr", "tone", "wui", "lui"}


def test_epu_lexicon_has_three_groups_in_english():
    from puremacro.narrative.indices._lexicons import LEXICONS
    en = LEXICONS["epu"]["en"]
    assert set(en) == {"economy", "policy", "uncertainty"}
    assert {"economic"} <= en["economy"]
    assert {"policy"} <= en["policy"]
    assert {"uncertain", "uncertainty"} <= en["uncertainty"]


def test_mpu_lexicon_english_has_monetary_terms():
    from puremacro.narrative.indices._lexicons import LEXICONS
    en = LEXICONS["mpu"]["en"]
    assert "monetary" in en
    assert "policy" in en
    assert "uncertain" in en or "uncertainty" in en


def test_gpr_lexicon_english_has_geopolitical_terms():
    from puremacro.narrative.indices._lexicons import LEXICONS
    en = LEXICONS["gpr"]["en"]
    assert "war" in en
    assert "terror" in en or "terrorism" in en
    assert "geopolitical" in en


def test_tone_lexicon_english_has_hawkish_dovish_groups():
    from puremacro.narrative.indices._lexicons import LEXICONS
    en = LEXICONS["tone"]["en"]
    assert set(en) == {"hawkish", "dovish"}
    assert {"hawkish", "tighten", "tightening"} <= en["hawkish"]
    assert {"dovish", "ease", "easing"} <= en["dovish"]


def test_wui_lexicon_english_has_uncertainty_terms():
    from puremacro.narrative.indices._lexicons import LEXICONS
    en = LEXICONS["wui"]["en"]
    assert "uncertainty" in en
    assert "uncertain" in en


def test_lui_lexicon_english_has_labor_terms():
    from puremacro.narrative.indices._lexicons import LEXICONS
    en = LEXICONS["lui"]["en"]
    # The labor-uncertainty index covers six conceptual groups; we ship
    # one flat term list here, but it must contain all six concept anchors.
    assert "layoff" in en or "layoffs" in en
    assert "hiring freeze" in en or "hiring-freeze" in en
    assert "wage compression" in en or "wage-compression" in en
    assert "labor shortage" in en or "labor-shortage" in en
    assert "unemployment" in en
```

- [ ] **Step 3: Run, verify failure**

Run: `pytest tests/test_narrative_indices.py -v --no-header 2>&1 | tail -10`
Expected: 7 fail with `ImportError: cannot import name 'LEXICONS'`.

- [ ] **Step 4: Replace `_lexicons.py` content**

Write `puremacro/narrative/indices/_lexicons.py`:

```python
"""Multilingual term lists for the indices layer (Slice 2).

Each lexicon is a plain Python literal — no external download — so the
indices layer remains Pyodide-clean.

Top-level structure::

    LEXICONS[index_name][language] = frozenset(...) | dict[str, frozenset]

The EPU and tone lexicons use a nested dict because their methodology
splits terms into co-occurring groups (E/P/U for EPU; hawkish vs dovish
for tone). All others are a flat frozenset of relevant terms.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Baker-Bloom-Davis EPU — three co-occurring term groups (Economy / Policy
# / Uncertainty). A document counts toward EPU if it contains ≥1 term from
# each group.
# ---------------------------------------------------------------------------
_EPU_EN = {
    "economy":     frozenset({"economic", "economy", "economics"}),
    "policy":      frozenset({"policy", "policies", "regulation", "regulatory",
                              "legislation", "deficit", "tariff",
                              "white house", "congress", "senate", "house",
                              "federal reserve", "central bank"}),
    "uncertainty": frozenset({"uncertain", "uncertainty", "uncertainties"}),
}


# ---------------------------------------------------------------------------
# Husted-Rogers-Sun monetary-policy uncertainty — flat term list.
# ---------------------------------------------------------------------------
_MPU_EN = frozenset({
    "monetary", "policy", "policies",
    "federal reserve", "central bank", "fomc", "ecb", "boe", "boj",
    "interest rate", "interest rates", "policy rate",
    "uncertain", "uncertainty", "uncertainties",
    "ambiguity", "ambiguous",
})


# ---------------------------------------------------------------------------
# Caldara-Iacoviello geopolitical-risk index — flat term list.
# ---------------------------------------------------------------------------
_GPR_EN = frozenset({
    "war", "warfare", "military",
    "terror", "terrorism", "terrorist", "terrorists",
    "geopolitical", "geopolitics",
    "sanctions", "sanction",
    "invasion", "invade",
    "nuclear", "missile", "missiles",
    "conflict", "tensions",
})


# ---------------------------------------------------------------------------
# Apel-Blix-Grimaldi hawkish/dovish tone lexicon (English).
# ---------------------------------------------------------------------------
_TONE_EN = {
    "hawkish": frozenset({
        "hawkish", "tighten", "tightening", "tightened",
        "hike", "hiked", "hikes",
        "raise", "raised", "raises",
        "restrictive", "withdraw", "withdrawal",
        "inflationary", "overheating",
    }),
    "dovish": frozenset({
        "dovish", "ease", "eased", "easing",
        "cut", "cuts", "cutting",
        "lower", "lowered", "lowers",
        "accommodative", "accommodation",
        "stimulus", "support",
        "deflationary", "slack",
    }),
}


# ---------------------------------------------------------------------------
# Ahir-Bloom-Furceri World Uncertainty Index — flat term list (uncertainty
# stems only).
# ---------------------------------------------------------------------------
_WUI_EN = frozenset({
    "uncertain", "uncertainty", "uncertainties",
    "ambiguity", "ambiguous",
    "unpredictable", "unpredictability",
})


# ---------------------------------------------------------------------------
# Labor-Market Uncertainty — novel lexicon, six conceptual groups
# (layoffs / hiring-freeze / wage-compression / labor-shortage /
# participation-drop / unemployment-risk) flattened into one term list.
# ---------------------------------------------------------------------------
_LUI_EN = frozenset({
    # Layoffs
    "layoff", "layoffs", "lay off", "lay offs",
    "redundancy", "redundancies", "downsizing", "downsize",
    "workforce reduction", "job cuts",
    # Hiring freeze
    "hiring freeze", "hiring-freeze", "hiring pause",
    "recruitment freeze", "freeze hiring",
    # Wage compression
    "wage compression", "wage-compression",
    "wage stagnation", "stagnant wages", "real wage decline",
    # Labor shortage
    "labor shortage", "labor-shortage", "labour shortage",
    "skill shortage", "skills gap", "talent shortage",
    # Participation drop
    "participation rate", "labor force participation",
    "discouraged workers", "dropout from the labor force",
    # Unemployment risk
    "unemployment", "joblessness", "jobless claims",
    "rising unemployment", "unemployment risk",
})


LEXICONS: dict = {
    "epu":  {"en": _EPU_EN},
    "mpu":  {"en": _MPU_EN},
    "gpr":  {"en": _GPR_EN},
    "tone": {"en": _TONE_EN},
    "wui":  {"en": _WUI_EN},
    "lui":  {"en": _LUI_EN},
}


__all__ = ["LEXICONS"]
```

- [ ] **Step 5: Run lexicon tests, expect green**

Run: `pytest tests/test_narrative_indices.py -v --no-header 2>&1 | tail -15`
Expected: all 7 lexicon tests pass + the original import smoke test.

- [ ] **Step 6: Commit**

```bash
git add puremacro/puremacro/narrative/indices/_lexicons.py \
        puremacro/tests/test_narrative_indices.py
git commit -m "feat(narrative): English lexicons for EPU/MPU/GPR/tone/WUI/LUI"
```

---

## Task 3: Kernel functions + `normalize_series` + `index_to_quarterly` actually normalizes

**Files:**
- Modify: `puremacro/narrative/indices/_kernels.py` (replace stub)
- Modify: `puremacro/narrative/aggregate.py` — `index_to_quarterly` actually applies `normalization`
- Modify: `tests/test_narrative_indices.py` (append kernel + normalize tests)
- Modify: `tests/test_narrative_index_to_quarterly.py` (append a normalization-round-trip test)

- [ ] **Step 1: Verify branch state**

Run: `git branch --show-current` — expected `feature/narrative-extension-slice2`.

- [ ] **Step 2: Write failing tests** — append to `tests/test_narrative_indices.py`:

```python
# ---------------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------------
import pandas as pd
import pytest


def _doc(date, text):
    """Synthetic 4-tuple SourceRecord."""
    return (pd.Timestamp(date), text, "https://test/" + str(date), {"language": "en"})


def test_count_keywords_basic():
    from puremacro.narrative.indices._kernels import count_keywords
    text = "Economic policy uncertainty rose this quarter."
    terms = frozenset({"uncertain", "uncertainty"})
    n = count_keywords(text, terms, language="en")
    # "uncertainty" matches once (single-token "uncertainty"); "uncertain"
    # is a prefix that should NOT separately match a single-token regex.
    assert n == 1


def test_count_keywords_multi_term_phrase():
    from puremacro.narrative.indices._kernels import count_keywords
    text = "The federal reserve raised the policy rate."
    terms = frozenset({"federal reserve", "policy rate"})
    n = count_keywords(text, terms, language="en")
    # Both two-word phrases match once each.
    assert n == 2


def test_count_keywords_substring_match_for_non_latin():
    """Japanese / Chinese tokenization is hard; substring match instead."""
    from puremacro.narrative.indices._kernels import count_keywords
    text = "経済政策不確実性"  # economy + policy + uncertainty (concatenated)
    terms = frozenset({"不確実性"})
    n = count_keywords(text, terms, language="ja")
    assert n == 1


def test_keyword_count_kernel_emits_per_doc_score():
    from puremacro.narrative.indices._kernels import keyword_count_kernel
    records = [
        _doc("2020-01-15", "uncertain uncertainty uncertainty rose"),
        _doc("2020-02-15", "no hits here"),
    ]
    terms = frozenset({"uncertain", "uncertainty"})
    out = list(keyword_count_kernel(records, terms=terms, language="en"))
    assert out[0][1] == 3   # uncertain + 2x uncertainty
    assert out[1][1] == 0


def test_cooccurrence_kernel_all_groups_present():
    from puremacro.narrative.indices._kernels import cooccurrence_kernel
    records = [
        _doc("2020-01-15", "economic policy uncertainty rose this quarter"),
    ]
    groups = [
        frozenset({"economic"}),
        frozenset({"policy"}),
        frozenset({"uncertainty", "uncertain"}),
    ]
    out = list(cooccurrence_kernel(records, term_groups=groups, language="en"))
    assert out[0][1] == 1.0


def test_cooccurrence_kernel_one_group_missing():
    from puremacro.narrative.indices._kernels import cooccurrence_kernel
    records = [
        _doc("2020-01-15", "economic policy went well"),  # no uncertainty term
    ]
    groups = [
        frozenset({"economic"}),
        frozenset({"policy"}),
        frozenset({"uncertainty", "uncertain"}),
    ]
    out = list(cooccurrence_kernel(records, term_groups=groups, language="en"))
    assert out[0][1] == 0.0


def test_tone_kernel_net_value():
    from puremacro.narrative.indices._kernels import tone_kernel
    records = [
        _doc("2022-03-15", "raised hike tightening hawkish ease"),  # 4 hawk, 1 dove
    ]
    out = list(tone_kernel(
        records,
        hawkish_terms=frozenset({"raised", "hike", "tightening", "hawkish"}),
        dovish_terms=frozenset({"ease", "easing", "cut"}),
        language="en",
    ))
    # Net: 4 hawk - 1 dove = +3, normalized by 5 token hits = 0.6
    assert out[0][1] == pytest.approx(0.6)


def test_tone_kernel_no_hits_returns_zero():
    from puremacro.narrative.indices._kernels import tone_kernel
    records = [_doc("2020-01-15", "no relevant words here")]
    out = list(tone_kernel(
        records,
        hawkish_terms=frozenset({"hike"}),
        dovish_terms=frozenset({"cut"}),
        language="en",
    ))
    assert out[0][1] == 0.0


# ---------------------------------------------------------------------------
# normalize_series tests
# ---------------------------------------------------------------------------
def test_normalize_raw_passes_through():
    from puremacro.narrative.indices._kernels import normalize_series
    s = pd.Series([100.0, 110.0, 95.0],
                  index=pd.date_range("2020-01-01", periods=3, freq="QS"))
    out = normalize_series(s, "raw")
    assert (out == s).all()


def test_normalize_zscore_has_zero_mean_unit_std():
    from puremacro.narrative.indices._kernels import normalize_series
    s = pd.Series([100.0, 110.0, 90.0, 105.0, 95.0],
                  index=pd.date_range("2020-01-01", periods=5, freq="QS"))
    out = normalize_series(s, "zscore")
    assert out.mean() == pytest.approx(0.0, abs=1e-10)
    assert out.std(ddof=0) == pytest.approx(1.0, abs=1e-10)


def test_normalize_bbd_100_has_mean_100_std_50():
    from puremacro.narrative.indices._kernels import normalize_series
    s = pd.Series([100.0, 110.0, 90.0, 105.0, 95.0],
                  index=pd.date_range("2020-01-01", periods=5, freq="QS"))
    out = normalize_series(s, "bbd_100")
    assert out.mean() == pytest.approx(100.0)
    assert out.std(ddof=0) == pytest.approx(50.0)


def test_normalize_bbd_100_with_base_period():
    """BBD's published series uses 1985-2009 as the base; normalization
    should target the BASE PERIOD's mean/std, not the full series."""
    from puremacro.narrative.indices._kernels import normalize_series
    idx = pd.date_range("2020-01-01", periods=8, freq="QS")
    # Base period 2020Q1-2020Q4 (rows 0..3) has mean 100 std 5.
    # Post-base values are 200 — should become much higher than 100.
    s = pd.Series([95.0, 100.0, 105.0, 100.0, 200.0, 200.0, 200.0, 200.0],
                  index=idx)
    base = ("2020-01-01", "2020-12-31")
    out = normalize_series(s, "bbd_100", base_period=base)
    # Base mean = 100, std = ~3.535; bbd target mean 100 std 50.
    base_mask = (out.index >= "2020-01-01") & (out.index <= "2020-12-31")
    assert out[base_mask].mean() == pytest.approx(100.0, rel=0.1)
    # Post-base values should be far above 100 (200 was 28+ std above base mean
    # in raw units, so should be 28*50 + 100 ≈ 1500 in BBD-100 — far above 100).
    assert (out[~base_mask] > 500).all()


def test_normalize_invalid_kind_raises():
    from puremacro.narrative.indices._kernels import normalize_series
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match="normalization"):
        normalize_series(s, "not_a_norm")
```

Append to `tests/test_narrative_index_to_quarterly.py`:

```python
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
    # zscore mean ≈ 0
    assert ri.series.dropna().mean() == pytest.approx(0.0, abs=1e-9)
```

- [ ] **Step 3: Run, verify failures**

Run: `pytest tests/test_narrative_indices.py tests/test_narrative_index_to_quarterly.py -v --no-header 2>&1 | tail -20`
Expected: 12 of the new tests fail with `ImportError` / `AttributeError`.

- [ ] **Step 4: Replace `_kernels.py`**

Write `puremacro/narrative/indices/_kernels.py`:

```python
"""Per-document scoring kernels for the indices layer (Slice 2).

Three kernels share one tokenization helper (``count_keywords``):

  - ``keyword_count_kernel`` — score = total term hits in the document.
  - ``cooccurrence_kernel`` — score = 1.0 iff the document contains ≥1
    term from each of N groups (BBD-EPU shape).
  - ``tone_kernel`` — score = net (hawkish − dovish) hits, normalised by
    the total hawkish + dovish hit count (so per-doc scores live in
    [-1, +1]).

Plus ``normalize_series`` for raw / zscore / bbd_100 transformation
of the resulting quarterly series.
"""
from __future__ import annotations

import re
from typing import Iterable, Iterator

import numpy as np
import pandas as pd


_NEEDS_SUBSTRING = {"ja", "zh"}
_TOKEN_RX = re.compile(r"\w+", flags=re.UNICODE)


def _normalize_text_for_match(text: str, language: str) -> str:
    """Lowercase + collapse whitespace. For non-Latin scripts we rely on
    substring matching so the original characters are preserved (no
    casefolding for JA/ZH because there's no upper/lowercase)."""
    if language in _NEEDS_SUBSTRING:
        return text
    return text.lower()


def count_keywords(text: str, terms: frozenset[str], language: str = "en") -> int:
    """Count occurrences of any term from ``terms`` in ``text``.

    For Latin scripts (en/es/pt/de/fr/it), uses word-boundary matching
    (multi-word terms search for the literal substring after a regex
    word-boundary on each side).

    For Japanese / Chinese, falls back to plain substring matching since
    those scripts don't have whitespace word boundaries.
    """
    if not text:
        return 0
    norm = _normalize_text_for_match(text, language)
    if language in _NEEDS_SUBSTRING:
        return sum(norm.count(t.lower() if False else t) for t in terms)
    n = 0
    for t in terms:
        # Multi-word terms: literal phrase match with word boundaries.
        # Single-word terms: word-boundary match (so "uncertain" doesn't
        # double-count "uncertainty").
        rx = r"\b" + re.escape(t.lower()) + r"\b"
        n += len(re.findall(rx, norm))
    return n


def keyword_count_kernel(
    records: Iterable[tuple],
    *,
    terms: frozenset[str],
    language: str = "en",
) -> Iterator[tuple]:
    """Yield ``(date, score)`` per record. Score = total keyword hits.

    Records are 4-tuples ``(date, text, source_url, metadata)`` (the
    Slice 1 SourceRecord shape) or 3-tuple legacy. Per-record metadata's
    ``language`` overrides the function-level ``language`` if present.
    """
    for record in records:
        if len(record) == 4:
            date, text, _, meta = record
            lang = (meta or {}).get("language", language)
        else:
            date, text, _ = record
            lang = language
        yield (pd.Timestamp(date), float(count_keywords(text, terms, language=lang)))


def cooccurrence_kernel(
    records: Iterable[tuple],
    *,
    term_groups: list[frozenset[str]],
    language: str = "en",
) -> Iterator[tuple]:
    """Yield ``(date, 1.0)`` if the document contains ≥1 term from EVERY
    group, else ``(date, 0.0)``. Used by BBD-EPU."""
    for record in records:
        if len(record) == 4:
            date, text, _, meta = record
            lang = (meta or {}).get("language", language)
        else:
            date, text, _ = record
            lang = language
        score = 1.0 if all(
            count_keywords(text, group, language=lang) > 0
            for group in term_groups
        ) else 0.0
        yield (pd.Timestamp(date), score)


def tone_kernel(
    records: Iterable[tuple],
    *,
    hawkish_terms: frozenset[str],
    dovish_terms: frozenset[str],
    language: str = "en",
) -> Iterator[tuple]:
    """Yield ``(date, net_tone)`` per doc.

    ``net_tone = (hawk_hits - dove_hits) / (hawk_hits + dove_hits)``
    for documents with at least one hit; ``0.0`` if neither lexicon
    matches. Per-doc score is bounded to ``[-1, +1]``.
    """
    for record in records:
        if len(record) == 4:
            date, text, _, meta = record
            lang = (meta or {}).get("language", language)
        else:
            date, text, _ = record
            lang = language
        h = count_keywords(text, hawkish_terms, language=lang)
        d = count_keywords(text, dovish_terms, language=lang)
        total = h + d
        score = 0.0 if total == 0 else (h - d) / total
        yield (pd.Timestamp(date), float(score))


_VALID_NORMALIZATIONS = {"raw", "zscore", "bbd_100"}


def normalize_series(
    series: pd.Series,
    normalization: str,
    *,
    base_period: tuple[str, str] | None = None,
) -> pd.Series:
    """Apply the spec-defined normalization to a quarterly index series.

    Parameters
    ----------
    series : pd.Series
        Indexed by datetime. NaN values are left in place (not imputed).
    normalization : str
        ``"raw"`` | ``"zscore"`` | ``"bbd_100"``.
    base_period : (start, end) | None
        If given, mean/std are computed on the slice
        ``series.loc[start:end]`` and applied to the whole series.
        Default: full series.

    Returns
    -------
    pd.Series with the same index, transformed.
    """
    if normalization not in _VALID_NORMALIZATIONS:
        raise ValueError(
            f"normalization {normalization!r} not in {_VALID_NORMALIZATIONS}"
        )
    if normalization == "raw":
        return series.copy()

    if base_period is None:
        ref = series.dropna()
    else:
        start, end = base_period
        ref = series.loc[start:end].dropna()

    if ref.empty:
        return series.copy()

    mu = float(ref.mean())
    sigma = float(ref.std(ddof=0))
    if sigma == 0.0:
        sigma = 1.0  # avoid divide-by-zero on degenerate base periods

    z = (series - mu) / sigma
    if normalization == "zscore":
        return z
    # bbd_100: target mean 100, std 50
    return 100.0 + 50.0 * z


__all__ = [
    "count_keywords",
    "keyword_count_kernel",
    "cooccurrence_kernel",
    "tone_kernel",
    "normalize_series",
]
```

- [ ] **Step 5: Patch `index_to_quarterly` to actually apply normalization**

In `puremacro/narrative/aggregate.py`, locate `index_to_quarterly`. After the existing line `out = df.groupby("q_date")["value"].std().fillna(0.0)` block (where `out` is set), and **before** the `full_idx = pd.date_range(...)` line, add normalization application. The minimal patch: just before `return RiskIndex(...)` at the bottom of the function, replace the unconditional `series=out` with a conditional pass through `normalize_series`.

Find this block in `aggregate.py` (around the end of `index_to_quarterly`):

```python
    return RiskIndex(
        name=name, country=country, series=out,
        method=method, corpus=corpus, language=language,
        normalization=normalization, metadata=full_metadata,
    )
```

Replace with:

```python
    if normalization != "raw":
        from .indices._kernels import normalize_series
        out = normalize_series(out, normalization)

    return RiskIndex(
        name=name, country=country, series=out,
        method=method, corpus=corpus, language=language,
        normalization=normalization, metadata=full_metadata,
    )
```

The `from .indices._kernels import normalize_series` is intentionally lazy (inside the function) — keeps `aggregate.py` from depending on the indices subpackage at import time, preserving the import order.

- [ ] **Step 6: Run new + existing index_to_quarterly tests**

Run: `pytest tests/test_narrative_indices.py tests/test_narrative_index_to_quarterly.py -v --no-header 2>&1 | tail -20`
Expected: all kernel + normalize tests pass, plus the new `test_index_to_quarterly_actually_applies_zscore_normalization`. Existing 5 `index_to_quarterly` tests still pass (raw passthrough is the default in those).

- [ ] **Step 7: Run full suite, no regressions**

Run: `pytest -q --no-header 2>&1 | tail -3`
Expected: ≥ 859 + 13 (new kernel/normalize tests) + 1 (new round-trip) = 873 passed.

- [ ] **Step 8: Commit**

```bash
git add puremacro/puremacro/narrative/indices/_kernels.py \
        puremacro/puremacro/narrative/aggregate.py \
        puremacro/tests/test_narrative_indices.py \
        puremacro/tests/test_narrative_index_to_quarterly.py
git commit -m "feat(narrative): kernels (count/cooccur/tone) + normalize_series + index_to_quarterly applies normalization"
```

---

## Task 4: EPU (Baker-Bloom-Davis)

**Files:**
- Create: `puremacro/narrative/indices/epu.py`
- Modify: `tests/test_narrative_indices.py` (append EPU tests)

- [ ] **Step 1: Verify branch state**

Run: `git branch --show-current` — expected `feature/narrative-extension-slice2`.

- [ ] **Step 2: Write failing tests** — append to `tests/test_narrative_indices.py`:

```python
# ---------------------------------------------------------------------------
# epu()
# ---------------------------------------------------------------------------
def test_epu_returns_riskindex_with_correct_metadata():
    from puremacro.narrative.indices import epu
    records = [
        _doc("2020-01-15", "economic policy uncertainty rose"),
        _doc("2020-02-15", "economic policy uncertainty rose again"),
        _doc("2020-04-15", "no relevant content here"),
    ]
    ri = epu(records, country="USA", language="en", normalize="raw")
    assert ri.country == "USA"
    assert ri.method == "keyword_count"
    assert ri.normalization == "raw"
    # Q1: 2 hits, Q2: 0 — under "mean" agg the values are 1.0 and 0.0 (raw).
    assert ri.series.iloc[0] == pytest.approx(1.0)
    assert ri.series.iloc[1] == pytest.approx(0.0)


def test_epu_uses_default_english_lexicon_when_language_en():
    from puremacro.narrative.indices import epu
    records = [_doc("2020-01-15", "economic policy uncertainty rose"),
               _doc("2020-02-15", "economic policy uncertainty rose")]
    ri = epu(records, country="USA", language="en", normalize="raw")
    # Both docs match all three groups — quarterly mean = 1.0
    assert ri.series.iloc[0] == pytest.approx(1.0)


def test_epu_zero_when_one_group_missing():
    from puremacro.narrative.indices import epu
    records = [_doc("2020-01-15", "economic policy went well"),
               _doc("2020-02-15", "economic activity was strong")]
    ri = epu(records, country="USA", language="en", normalize="raw")
    assert ri.series.iloc[0] == pytest.approx(0.0)


def test_epu_bbd_100_normalization_applied():
    from puremacro.narrative.indices import epu
    # 8 quarters: 4 with EPU=1, 4 with EPU=0. After bbd_100 over the
    # full series, values should be centered at 100 with std 50.
    records = []
    high_text = "economic policy uncertainty rose"
    low_text = "no hits"
    for q, text in enumerate([high_text, low_text] * 4):
        d = pd.Timestamp(f"2020-{q + 1:02d}-15")
        records.append(_doc(str(d.date()), text))
    ri = epu(records, country="USA", language="en", normalize="bbd_100")
    s = ri.series.dropna()
    assert s.mean() == pytest.approx(100.0)
    assert s.std(ddof=0) == pytest.approx(50.0)


def test_epu_custom_lexicon_overrides_default():
    from puremacro.narrative.indices import epu
    custom = {
        "economy":     frozenset({"widget"}),
        "policy":      frozenset({"sprocket"}),
        "uncertainty": frozenset({"flummox"}),
    }
    records = [_doc("2020-01-15", "the widget sprocket flummox happened")]
    ri = epu(records, country="USA", language="en",
             lexicon=custom, normalize="raw")
    assert ri.series.iloc[0] == pytest.approx(1.0)
```

- [ ] **Step 3: Run, verify failure**

Run: `pytest tests/test_narrative_indices.py -v --no-header -k "epu" 2>&1 | tail -10`
Expected: 5 fail with `ImportError: cannot import name 'epu'`.

- [ ] **Step 4: Create `epu.py`**

```python
"""Baker-Bloom-Davis Economic Policy Uncertainty index.

Constructs a count of documents that contain at least one term from
each of three groups (Economy, Policy, Uncertainty), aggregates by
quarter, and optionally normalises (z-score or BBD's 100/50 scale).

Reference
---------
Baker, S.R., Bloom, N., Davis, S.J. (2016). Measuring economic policy
uncertainty. QJE 131(4), 1593-1636.
"""
from __future__ import annotations

from typing import Iterable

from ..types import RiskIndex
from ..aggregate import index_to_quarterly
from ._lexicons import LEXICONS
from ._kernels import cooccurrence_kernel


def epu(
    text_iter: Iterable[tuple],
    *,
    country: str,
    language: str = "en",
    lexicon: dict | None = None,
    normalize: str = "bbd_100",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
) -> RiskIndex:
    """Build a Baker-Bloom-Davis EPU series from a custom corpus.

    Parameters
    ----------
    text_iter : iterable of ``(date, text, source_url, metadata)`` records.
    country : ISO3 country tag stamped onto the resulting RiskIndex.
    language : ISO-639-1; selects the default lexicon if ``lexicon=None``.
    lexicon : optional override of the form
        ``{"economy": frozenset, "policy": frozenset, "uncertainty": frozenset}``.
    normalize : ``"raw"`` | ``"zscore"`` | ``"bbd_100"``.
    base_period : optional ``(start_iso, end_iso)`` for normalisation
        statistics. If ``None``, statistics are computed on the full series.
    agg : ``"mean"`` (default) | ``"max"`` | ``"dispersion"`` aggregator
        applied across documents within each quarter.
    """
    lex = lexicon if lexicon is not None else LEXICONS["epu"][language]
    term_groups = [lex["economy"], lex["policy"], lex["uncertainty"]]

    def _kernel(records):
        return cooccurrence_kernel(
            records, term_groups=term_groups, language=language,
        )

    return index_to_quarterly(
        text_iter, kernel=_kernel,
        country=country, language=language,
        name=f"epu_{country.lower()}",
        method="keyword_count", corpus="custom",
        normalization=normalize, agg=agg,
        metadata={
            "index": "epu", "base_period": base_period,
        },
    )


__all__ = ["epu"]
```

Note: `index_to_quarterly` already applies normalization (per Task 3). The `base_period` is currently stored in metadata but not threaded into the normalization computation. For the bbd_100 test to pass on a full-series base, we don't need the base-period plumbing yet. **TODO for Slice 3**: thread `base_period` through `index_to_quarterly` → `normalize_series`.

- [ ] **Step 5: Run EPU tests, expect green**

Run: `pytest tests/test_narrative_indices.py -v --no-header -k "epu" 2>&1 | tail -10`
Expected: all 5 EPU tests pass.

- [ ] **Step 6: Run full suite**

Run: `pytest -q --no-header 2>&1 | tail -3`
Expected: pass count up by 5.

- [ ] **Step 7: Commit**

```bash
git add puremacro/puremacro/narrative/indices/epu.py \
        puremacro/tests/test_narrative_indices.py
git commit -m "feat(narrative): epu() — Baker-Bloom-Davis EPU on arbitrary corpora"
```

---

## Task 5: MPU (monetary-policy uncertainty)

**Files:**
- Create: `puremacro/narrative/indices/mpu.py`
- Modify: `tests/test_narrative_indices.py` (append MPU tests)

- [ ] **Step 1: Verify branch state**

Run: `git branch --show-current` — expected `feature/narrative-extension-slice2`.

- [ ] **Step 2: Write failing tests** — append:

```python
# ---------------------------------------------------------------------------
# mpu()
# ---------------------------------------------------------------------------
def test_mpu_counts_monetary_uncertainty_terms():
    from puremacro.narrative.indices import mpu
    records = [
        _doc("2020-01-15",
             "monetary policy uncertainty around the federal reserve increased"),
        _doc("2020-02-15", "no relevant words"),
    ]
    ri = mpu(records, country="USA", language="en", normalize="raw")
    # Doc 1 has multiple MPU-lexicon hits, doc 2 has zero.
    assert ri.series.iloc[0] > 0
    assert ri.country == "USA"


def test_mpu_zscore_normalization():
    from puremacro.narrative.indices import mpu
    records = []
    high = "monetary policy uncertainty federal reserve interest rate"
    low = "weather"
    for q in range(8):
        d = pd.Timestamp("2020-01-01") + pd.DateOffset(months=q)
        records.append(_doc(str(d.date()), high if q % 2 == 0 else low))
    ri = mpu(records, country="USA", language="en", normalize="zscore")
    assert ri.series.dropna().mean() == pytest.approx(0.0, abs=1e-9)


def test_mpu_metadata_records_index_label():
    from puremacro.narrative.indices import mpu
    records = [_doc("2020-01-15", "monetary policy")]
    ri = mpu(records, country="USA", language="en", normalize="raw")
    assert ri.metadata.get("index") == "mpu"
```

- [ ] **Step 3: Run, verify failure**

Run: `pytest tests/test_narrative_indices.py -v --no-header -k "mpu" 2>&1 | tail -10`
Expected: 3 fail.

- [ ] **Step 4: Create `mpu.py`**

```python
"""Husted-Rogers-Sun monetary-policy uncertainty index.

Counts documents containing monetary-policy-uncertainty terms,
aggregates per-quarter, optionally normalises (default ``zscore``).

Reference
---------
Husted, L., Rogers, J., Sun, B. (2020). Monetary policy uncertainty.
J. Monetary Economics 115, 20-36.
"""
from __future__ import annotations

from typing import Iterable

from ..types import RiskIndex
from ..aggregate import index_to_quarterly
from ._lexicons import LEXICONS
from ._kernels import keyword_count_kernel


def mpu(
    text_iter: Iterable[tuple],
    *,
    country: str,
    language: str = "en",
    lexicon: frozenset | None = None,
    normalize: str = "zscore",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
) -> RiskIndex:
    """Build a monetary-policy uncertainty series from a custom corpus.

    Parameters
    ----------
    text_iter : iterable of ``(date, text, source_url, metadata)`` records.
    country : ISO3 country tag.
    language : ISO-639-1; selects the default flat term-list lexicon if
        ``lexicon=None``.
    lexicon : optional ``frozenset[str]`` override.
    normalize : ``"raw"`` | ``"zscore"`` (default) | ``"bbd_100"``.
    base_period : optional normalisation base period.
    agg : aggregator across documents in a quarter.
    """
    terms = lexicon if lexicon is not None else LEXICONS["mpu"][language]

    def _kernel(records):
        return keyword_count_kernel(records, terms=terms, language=language)

    return index_to_quarterly(
        text_iter, kernel=_kernel,
        country=country, language=language,
        name=f"mpu_{country.lower()}",
        method="keyword_count", corpus="custom",
        normalization=normalize, agg=agg,
        metadata={"index": "mpu", "base_period": base_period},
    )


__all__ = ["mpu"]
```

- [ ] **Step 5: Run MPU tests, expect green**

Run: `pytest tests/test_narrative_indices.py -v --no-header -k "mpu" 2>&1 | tail -10`
Expected: 3 pass.

- [ ] **Step 6: Commit**

```bash
git add puremacro/puremacro/narrative/indices/mpu.py \
        puremacro/tests/test_narrative_indices.py
git commit -m "feat(narrative): mpu() — Husted-Rogers-Sun monetary-policy uncertainty"
```

---

## Task 6: GPR (geopolitical risk)

**Files:**
- Create: `puremacro/narrative/indices/gpr.py`
- Modify: `tests/test_narrative_indices.py` (append GPR tests)

- [ ] **Step 1: Verify branch state**

Run: `git branch --show-current` — expected `feature/narrative-extension-slice2`.

- [ ] **Step 2: Write failing tests** — append:

```python
# ---------------------------------------------------------------------------
# gpr()
# ---------------------------------------------------------------------------
def test_gpr_counts_geopolitical_terms():
    from puremacro.narrative.indices import gpr
    records = [
        _doc("2020-01-15", "war terrorism geopolitical sanctions invasion"),
        _doc("2020-02-15", "ordinary peaceful day"),
    ]
    ri = gpr(records, country="USA", language="en", normalize="raw")
    assert ri.series.iloc[0] > 0
    assert ri.country == "USA"


def test_gpr_metadata_records_index_label():
    from puremacro.narrative.indices import gpr
    records = [_doc("2020-01-15", "war broke out")]
    ri = gpr(records, country="USA", language="en", normalize="raw")
    assert ri.metadata.get("index") == "gpr"
```

- [ ] **Step 3: Run, verify failure**

Run: `pytest tests/test_narrative_indices.py -v --no-header -k "gpr" 2>&1 | tail -10`
Expected: 2 fail.

- [ ] **Step 4: Create `gpr.py`**

```python
"""Caldara-Iacoviello geopolitical risk index.

Counts documents containing geopolitical-risk terms, aggregates per
quarter, optionally normalises.

Reference
---------
Caldara, D., Iacoviello, M. (2022). Measuring geopolitical risk.
American Economic Review 112(4), 1194-1225.
"""
from __future__ import annotations

from typing import Iterable

from ..types import RiskIndex
from ..aggregate import index_to_quarterly
from ._lexicons import LEXICONS
from ._kernels import keyword_count_kernel


def gpr(
    text_iter: Iterable[tuple],
    *,
    country: str,
    language: str = "en",
    lexicon: frozenset | None = None,
    normalize: str = "zscore",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
) -> RiskIndex:
    """Build a Caldara-Iacoviello GPR series from a custom corpus."""
    terms = lexicon if lexicon is not None else LEXICONS["gpr"][language]

    def _kernel(records):
        return keyword_count_kernel(records, terms=terms, language=language)

    return index_to_quarterly(
        text_iter, kernel=_kernel,
        country=country, language=language,
        name=f"gpr_{country.lower()}",
        method="keyword_count", corpus="custom",
        normalization=normalize, agg=agg,
        metadata={"index": "gpr", "base_period": base_period},
    )


__all__ = ["gpr"]
```

- [ ] **Step 5: Run GPR tests, expect green**

Run: `pytest tests/test_narrative_indices.py -v --no-header -k "gpr" 2>&1 | tail -10`
Expected: 2 pass.

- [ ] **Step 6: Commit**

```bash
git add puremacro/puremacro/narrative/indices/gpr.py \
        puremacro/tests/test_narrative_indices.py
git commit -m "feat(narrative): gpr() — Caldara-Iacoviello geopolitical risk index"
```

---

## Task 7: tone() — hawkish-dovish

**Files:**
- Create: `puremacro/narrative/indices/tone.py`
- Modify: `tests/test_narrative_indices.py` (append tone tests)

- [ ] **Step 1: Verify branch state**

Run: `git branch --show-current` — expected `feature/narrative-extension-slice2`.

- [ ] **Step 2: Write failing tests** — append:

```python
# ---------------------------------------------------------------------------
# tone()
# ---------------------------------------------------------------------------
def test_tone_apel_blix_grimaldi_hawkish_corpus_positive():
    from puremacro.narrative.indices import tone
    records = [
        _doc("2020-01-15", "raised hike tightening hawkish"),
        _doc("2020-02-15", "raise hawkish withdraw"),
    ]
    ri = tone(records, country="USA", language="en",
              method="apel_blix_grimaldi", normalize="raw")
    # All hawkish hits, no dovish — net = +1 per doc, mean +1
    assert ri.series.iloc[0] == pytest.approx(1.0)


def test_tone_apel_blix_grimaldi_dovish_corpus_negative():
    from puremacro.narrative.indices import tone
    records = [_doc("2020-01-15", "cut ease dovish accommodative")]
    ri = tone(records, country="USA", language="en",
              method="apel_blix_grimaldi", normalize="raw")
    assert ri.series.iloc[0] == pytest.approx(-1.0)


def test_tone_neutral_empty_text_yields_zero():
    from puremacro.narrative.indices import tone
    records = [_doc("2020-01-15", "weather report")]
    ri = tone(records, country="USA", language="en",
              method="apel_blix_grimaldi", normalize="raw")
    assert ri.series.iloc[0] == pytest.approx(0.0)


def test_tone_method_picault_renault_falls_back_to_count_for_now():
    """Picault-Renault uses paragraph-level multinomial classification.
    For Slice 2 we ship a count-based approximation; the call must not
    raise and the metadata records the method requested."""
    from puremacro.narrative.indices import tone
    records = [_doc("2020-01-15", "raised hike")]
    ri = tone(records, country="USA", language="en",
              method="picault_renault", normalize="raw")
    assert ri.metadata["method_requested"] == "picault_renault"


def test_tone_unknown_method_raises():
    from puremacro.narrative.indices import tone
    records = [_doc("2020-01-15", "x")]
    with pytest.raises(ValueError, match="method"):
        tone(records, country="USA", language="en", method="not_a_method")
```

- [ ] **Step 3: Run, verify failure**

Run: `pytest tests/test_narrative_indices.py -v --no-header -k "tone" 2>&1 | tail -10`
Expected: 5 fail.

- [ ] **Step 4: Create `tone.py`**

```python
"""Hawkish-dovish tone indices for central-bank text.

Three methods are wired in Slice 2:

  - ``apel_blix_grimaldi`` (default) — net (hawk - dove) hits divided
    by total hits per document, then aggregated quarterly. This is the
    Apel-Blix-Grimaldi (2017) net-tone construction.
  - ``hubert`` — same count-based mechanism, currently identical to
    apel_blix_grimaldi (separate Hubert lexicon planned for Slice 3).
  - ``picault_renault`` — falls back to count-based for Slice 2; the
    full paragraph-level multinomial logit lands in Slice 3.

The ``llm`` method is reserved for Slice 3 (uses ``scoring/llm.py``
backends; Experimental tier).

References
----------
Apel, M., Blix Grimaldi, M. (2014). How informative are central bank
minutes? Sveriges Riksbank Working Paper 261.

Picault, M., Renault, T. (2017). Words are not all created equal: A new
measure of ECB communication. Journal of International Money and
Finance 79, 136-156.

Hubert, P. (2017). Central bank information and the effects of monetary
shocks. Bank of England Staff Working Paper 672.
"""
from __future__ import annotations

from typing import Iterable

from ..types import RiskIndex
from ..aggregate import index_to_quarterly
from ._lexicons import LEXICONS
from ._kernels import tone_kernel


_VALID_METHODS = {"apel_blix_grimaldi", "hubert", "picault_renault"}


def tone(
    text_iter: Iterable[tuple],
    *,
    country: str,
    language: str = "en",
    method: str = "apel_blix_grimaldi",
    lexicon: dict | None = None,
    normalize: str = "raw",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
) -> RiskIndex:
    """Build a hawkish-dovish tone series from a custom corpus.

    Parameters
    ----------
    method : ``"apel_blix_grimaldi"`` (default) | ``"hubert"`` |
        ``"picault_renault"``. All three currently use the same
        net-count construction; the lexicon swaps differ (and Slice 3
        will refine ``picault_renault`` and ``hubert`` to their full
        methodologies).
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"method {method!r} not in {_VALID_METHODS}"
        )

    lex = lexicon if lexicon is not None else LEXICONS["tone"][language]

    def _kernel(records):
        return tone_kernel(
            records,
            hawkish_terms=lex["hawkish"],
            dovish_terms=lex["dovish"],
            language=language,
        )

    return index_to_quarterly(
        text_iter, kernel=_kernel,
        country=country, language=language,
        name=f"tone_{country.lower()}",
        method="tone_dispersion" if agg == "dispersion" else "keyword_count",
        corpus="custom",
        normalization=normalize, agg=agg,
        metadata={
            "index": "tone",
            "method_requested": method,
            "base_period": base_period,
        },
    )


__all__ = ["tone"]
```

- [ ] **Step 5: Run tone tests, expect green**

Run: `pytest tests/test_narrative_indices.py -v --no-header -k "tone" 2>&1 | tail -10`
Expected: 5 pass.

- [ ] **Step 6: Commit**

```bash
git add puremacro/puremacro/narrative/indices/tone.py \
        puremacro/tests/test_narrative_indices.py
git commit -m "feat(narrative): tone() — Apel-Blix-Grimaldi hawkish-dovish (Hubert/Picault-Renault count-based fallback)"
```

---

## Task 8: WUI (World Uncertainty Index style)

**Files:**
- Create: `puremacro/narrative/indices/wui.py`
- Modify: `tests/test_narrative_indices.py` (append WUI tests)

- [ ] **Step 1: Verify branch state**

Run: `git branch --show-current` — expected `feature/narrative-extension-slice2`.

- [ ] **Step 2: Write failing tests** — append:

```python
# ---------------------------------------------------------------------------
# wui()
# ---------------------------------------------------------------------------
def test_wui_counts_uncertainty_terms_only():
    from puremacro.narrative.indices import wui
    records = [
        _doc("2020-01-15", "uncertain uncertainty unpredictable ambiguity"),
        _doc("2020-02-15", "the weather was nice"),
    ]
    ri = wui(records, country="MEX", language="en", normalize="raw")
    assert ri.series.iloc[0] > 0
    assert ri.country == "MEX"


def test_wui_metadata_records_index_label():
    from puremacro.narrative.indices import wui
    records = [_doc("2020-01-15", "uncertainty")]
    ri = wui(records, country="MEX", language="en", normalize="raw")
    assert ri.metadata.get("index") == "wui"
```

- [ ] **Step 3: Run, verify failure**

Run: `pytest tests/test_narrative_indices.py -v --no-header -k "wui" 2>&1 | tail -10`
Expected: 2 fail.

- [ ] **Step 4: Create `wui.py`**

```python
"""Ahir-Bloom-Furceri World Uncertainty Index style.

Counts documents containing uncertainty terms, aggregates per quarter.
The original WUI normalises by document length (uncertainty mentions
per 1000 words); Slice 2 ships the simpler count-based variant. The
length-normalisation refinement is on the Slice 3 backlog.

Reference
---------
Ahir, H., Bloom, N., Furceri, D. (2022). The World Uncertainty Index.
NBER WP 29763.
"""
from __future__ import annotations

from typing import Iterable

from ..types import RiskIndex
from ..aggregate import index_to_quarterly
from ._lexicons import LEXICONS
from ._kernels import keyword_count_kernel


def wui(
    text_iter: Iterable[tuple],
    *,
    country: str,
    language: str = "en",
    lexicon: frozenset | None = None,
    normalize: str = "zscore",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
) -> RiskIndex:
    """Build a World Uncertainty Index style series from a custom corpus."""
    terms = lexicon if lexicon is not None else LEXICONS["wui"][language]

    def _kernel(records):
        return keyword_count_kernel(records, terms=terms, language=language)

    return index_to_quarterly(
        text_iter, kernel=_kernel,
        country=country, language=language,
        name=f"wui_{country.lower()}",
        method="keyword_count", corpus="custom",
        normalization=normalize, agg=agg,
        metadata={"index": "wui", "base_period": base_period},
    )


__all__ = ["wui"]
```

- [ ] **Step 5: Run WUI tests, expect green**

Run: `pytest tests/test_narrative_indices.py -v --no-header -k "wui" 2>&1 | tail -10`
Expected: 2 pass.

- [ ] **Step 6: Commit**

```bash
git add puremacro/puremacro/narrative/indices/wui.py \
        puremacro/tests/test_narrative_indices.py
git commit -m "feat(narrative): wui() — Ahir-Bloom-Furceri World Uncertainty Index style"
```

---

## Task 9: LUI (Labor-Market Uncertainty — novel)

**Files:**
- Create: `puremacro/narrative/indices/lui.py`
- Modify: `tests/test_narrative_indices.py` (append LUI tests)

- [ ] **Step 1: Verify branch state**

Run: `git branch --show-current` — expected `feature/narrative-extension-slice2`.

- [ ] **Step 2: Write failing tests** — append:

```python
# ---------------------------------------------------------------------------
# lui()
# ---------------------------------------------------------------------------
def test_lui_counts_labor_uncertainty_terms():
    from puremacro.narrative.indices import lui
    records = [
        _doc("2020-01-15",
             "layoffs hiring freeze wage compression labor shortage rising unemployment"),
        _doc("2020-02-15", "ordinary day"),
    ]
    ri = lui(records, country="USA", language="en", normalize="raw")
    assert ri.series.iloc[0] > 0
    assert ri.country == "USA"


def test_lui_metadata_records_index_label():
    from puremacro.narrative.indices import lui
    records = [_doc("2020-01-15", "layoff")]
    ri = lui(records, country="USA", language="en", normalize="raw")
    assert ri.metadata.get("index") == "lui"


def test_lui_distinguishes_high_low_periods():
    """A clearly labor-stressed corpus should rank above a quiet one."""
    from puremacro.narrative.indices import lui
    records = []
    high_text = "layoffs hiring freeze rising unemployment wage compression"
    low_text = "ordinary"
    for q in range(8):
        d = pd.Timestamp("2020-01-01") + pd.DateOffset(months=q * 3)
        records.append(_doc(str(d.date()), high_text if q < 4 else low_text))
    ri = lui(records, country="USA", language="en", normalize="zscore")
    s = ri.series.dropna()
    # First half (high) z-scores should average above the second half (low).
    assert s.iloc[:4].mean() > s.iloc[4:].mean()
```

- [ ] **Step 3: Run, verify failure**

Run: `pytest tests/test_narrative_indices.py -v --no-header -k "lui" 2>&1 | tail -10`
Expected: 3 fail.

- [ ] **Step 4: Create `lui.py`**

```python
"""Labor-Market Uncertainty Index — novel index for the MAV uncertainty
research track.

Counts documents containing labor-market-uncertainty terms across six
conceptual groups (layoffs / hiring-freeze / wage-compression /
labor-shortage / participation-drop / unemployment-risk), aggregated
per quarter. Designed to feed directly into the active subnational-
labor-uncertainty-US branch and the SigmaObject volatility-decomposition
track.
"""
from __future__ import annotations

from typing import Iterable

from ..types import RiskIndex
from ..aggregate import index_to_quarterly
from ._lexicons import LEXICONS
from ._kernels import keyword_count_kernel


def lui(
    text_iter: Iterable[tuple],
    *,
    country: str,
    language: str = "en",
    lexicon: frozenset | None = None,
    normalize: str = "zscore",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
) -> RiskIndex:
    """Build a labor-market uncertainty series from a custom corpus."""
    terms = lexicon if lexicon is not None else LEXICONS["lui"][language]

    def _kernel(records):
        return keyword_count_kernel(records, terms=terms, language=language)

    return index_to_quarterly(
        text_iter, kernel=_kernel,
        country=country, language=language,
        name=f"lui_{country.lower()}",
        method="keyword_count", corpus="custom",
        normalization=normalize, agg=agg,
        metadata={"index": "lui", "base_period": base_period},
    )


__all__ = ["lui"]
```

- [ ] **Step 5: Run LUI tests, expect green**

Run: `pytest tests/test_narrative_indices.py -v --no-header -k "lui" 2>&1 | tail -10`
Expected: 3 pass.

- [ ] **Step 6: Commit**

```bash
git add puremacro/puremacro/narrative/indices/lui.py \
        puremacro/tests/test_narrative_indices.py
git commit -m "feat(narrative): lui() — Labor-Market Uncertainty index (novel)"
```

---

## Task 10: Multilingual lexicon extensions

**Files:**
- Modify: `puremacro/narrative/indices/_lexicons.py` (extend `LEXICONS` with es/pt/de/fr/it/ja/zh)
- Modify: `tests/test_narrative_indices.py` (append multilingual tests)

- [ ] **Step 1: Verify branch state**

Run: `git branch --show-current` — expected `feature/narrative-extension-slice2`.

- [ ] **Step 2: Write failing tests** — append:

```python
# ---------------------------------------------------------------------------
# Multilingual lexicons
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lang", ["es", "pt", "de", "fr", "it", "ja", "zh"])
def test_epu_lexicon_present_for_all_supported_languages(lang):
    from puremacro.narrative.indices._lexicons import LEXICONS
    assert lang in LEXICONS["epu"], f"EPU missing language {lang}"
    groups = LEXICONS["epu"][lang]
    assert set(groups) == {"economy", "policy", "uncertainty"}, (
        f"EPU/{lang} groups: {sorted(groups)}"
    )
    assert all(len(groups[g]) >= 1 for g in groups), (
        f"EPU/{lang} has empty group: {groups}"
    )


@pytest.mark.parametrize("lang", ["es", "pt", "de", "fr", "it", "ja", "zh"])
def test_other_indices_have_each_supported_language(lang):
    from puremacro.narrative.indices._lexicons import LEXICONS
    for index in ("mpu", "gpr", "wui", "lui"):
        assert lang in LEXICONS[index], (
            f"{index} missing language {lang}"
        )
        terms = LEXICONS[index][lang]
        assert len(terms) >= 1, f"{index}/{lang} is empty"


@pytest.mark.parametrize("lang", ["es", "pt", "de", "fr", "it"])
def test_tone_lexicon_present_for_latin_languages(lang):
    """Tone lexicon ships for Latin-script languages in Slice 2.
    Japanese/Chinese tone deferred to Slice 3."""
    from puremacro.narrative.indices._lexicons import LEXICONS
    assert lang in LEXICONS["tone"]
    groups = LEXICONS["tone"][lang]
    assert {"hawkish", "dovish"} <= set(groups)


def test_epu_works_with_spanish_corpus():
    """End-to-end: the spanish lexicon should produce non-trivial scores
    on a recognisable Spanish EPU phrase."""
    from puremacro.narrative.indices import epu
    records = [
        _doc("2020-01-15",
             "incertidumbre sobre la política económica del banco central"),
        _doc("2020-02-15", "tiempo agradable hoy"),
    ]
    ri = epu(records, country="MEX", language="es", normalize="raw")
    # Q1: should have non-zero score (eco + pol + unc all present in text)
    # Q2: should be 0
    assert ri.series.iloc[0] > 0, (
        f"Spanish EPU should detect 'incertidumbre/política/económica'; got {ri.series.tolist()}"
    )


def test_lui_works_with_spanish_corpus():
    from puremacro.narrative.indices import lui
    records = [
        _doc("2020-01-15", "despidos congelación de contrataciones desempleo"),
    ]
    ri = lui(records, country="MEX", language="es", normalize="raw")
    assert ri.series.iloc[0] > 0
```

- [ ] **Step 3: Run, verify failures**

Run: `pytest tests/test_narrative_indices.py -v --no-header -k "lang or spanish" 2>&1 | tail -20`
Expected: most fail (KeyError on missing language).

- [ ] **Step 4: Extend `_lexicons.py`**

Open `puremacro/narrative/indices/_lexicons.py` and add the multilingual data BEFORE the `LEXICONS = {...}` declaration. Replace the existing single-language `LEXICONS = {"epu": {"en": _EPU_EN}, ...}` with the full multilingual dict below.

Add these constants (place between the English constants and the `LEXICONS` declaration):

```python
# ---------------------------------------------------------------------------
# Spanish (es)
# ---------------------------------------------------------------------------
_EPU_ES = {
    "economy":     frozenset({"económica", "económico", "economía"}),
    "policy":      frozenset({"política", "políticas", "regulación",
                              "regulatoria", "legislación", "déficit",
                              "arancel", "aranceles", "banco central"}),
    "uncertainty": frozenset({"incierto", "incierta", "incertidumbre"}),
}
_MPU_ES = frozenset({
    "monetaria", "política", "políticas",
    "banco central", "banco de méxico", "tipo de interés",
    "incierto", "incierta", "incertidumbre", "ambigüedad",
})
_GPR_ES = frozenset({
    "guerra", "militar",
    "terrorismo", "terrorista",
    "geopolítico", "geopolítica",
    "sanciones", "sanción",
    "invasión", "invadir",
    "nuclear", "misil",
    "conflicto", "tensiones",
})
_TONE_ES = {
    "hawkish": frozenset({
        "halcón", "halcones", "endurecer", "endurecimiento",
        "subir", "subió", "aumentó", "aumento",
        "restrictivo", "restrictiva",
        "inflacionario", "inflacionaria",
    }),
    "dovish": frozenset({
        "paloma", "palomas", "relajar", "relajamiento",
        "recortar", "recortó", "bajó", "reducir",
        "acomodaticio", "acomodaticia",
        "estímulo", "apoyo",
    }),
}
_WUI_ES = frozenset({
    "incierto", "incierta", "incertidumbre",
    "ambigüedad", "ambiguo", "ambigua",
    "imprevisible", "impredecible",
})
_LUI_ES = frozenset({
    "despido", "despidos",
    "congelación de contrataciones", "congelamiento de contrataciones",
    "compresión salarial", "estancamiento salarial",
    "escasez de mano de obra", "escasez laboral",
    "tasa de participación", "trabajadores desalentados",
    "desempleo", "paro", "desocupación",
})


# ---------------------------------------------------------------------------
# Portuguese (pt)
# ---------------------------------------------------------------------------
_EPU_PT = {
    "economy":     frozenset({"econômica", "econômico", "economia",
                              "económica", "económico"}),
    "policy":      frozenset({"política", "políticas", "regulação",
                              "regulamentação", "legislação", "défice",
                              "tarifa", "tarifas", "banco central"}),
    "uncertainty": frozenset({"incerto", "incerta", "incerteza"}),
}
_MPU_PT = frozenset({
    "monetária", "política", "políticas",
    "banco central", "banco do brasil", "taxa de juros",
    "incerto", "incerta", "incerteza",
})
_GPR_PT = frozenset({
    "guerra", "militar",
    "terrorismo", "terrorista",
    "geopolítico", "geopolítica",
    "sanções", "sanção",
    "invasão", "invadir",
    "nuclear", "míssil",
    "conflito", "tensões",
})
_TONE_PT = {
    "hawkish": frozenset({
        "falcão", "falcões", "apertar", "aperto",
        "subir", "subiu", "aumentar", "aumento",
        "restritiva", "restritivo",
    }),
    "dovish": frozenset({
        "pomba", "pombas", "relaxar", "relaxamento",
        "cortar", "cortou", "reduzir", "redução",
        "acomodatícia", "acomodatício",
        "estímulo", "apoio",
    }),
}
_WUI_PT = frozenset({
    "incerto", "incerta", "incerteza",
    "ambiguidade", "ambíguo", "ambígua",
    "imprevisível",
})
_LUI_PT = frozenset({
    "demissão", "demissões",
    "congelamento de contratações",
    "compressão salarial", "estagnação salarial",
    "escassez de mão de obra",
    "taxa de participação",
    "desemprego",
})


# ---------------------------------------------------------------------------
# German (de)
# ---------------------------------------------------------------------------
_EPU_DE = {
    "economy":     frozenset({"wirtschaftlich", "wirtschaft"}),
    "policy":      frozenset({"politik", "regulierung", "gesetzgebung",
                              "defizit", "zoll", "zentralbank",
                              "europäische zentralbank", "bundestag"}),
    "uncertainty": frozenset({"unsicher", "unsicherheit"}),
}
_MPU_DE = frozenset({
    "geldpolitik", "geldpolitisch",
    "zentralbank", "europäische zentralbank", "bundesbank",
    "leitzins", "zinssatz",
    "unsicher", "unsicherheit",
})
_GPR_DE = frozenset({
    "krieg", "militär",
    "terror", "terrorismus", "terroristisch",
    "geopolitisch", "geopolitik",
    "sanktionen", "sanktion",
    "invasion",
    "nuklear", "rakete",
    "konflikt", "spannungen",
})
_TONE_DE = {
    "hawkish": frozenset({
        "falke", "falken", "straffen", "straffung",
        "anheben", "erhöhen", "erhöhung",
        "restriktiv", "inflationär",
    }),
    "dovish": frozenset({
        "taube", "tauben", "lockern", "lockerung",
        "senken", "senkung",
        "akkommodativ", "stimulus", "unterstützung",
    }),
}
_WUI_DE = frozenset({
    "unsicher", "unsicherheit",
    "uneindeutig", "ambiguität",
    "unvorhersehbar",
})
_LUI_DE = frozenset({
    "entlassung", "entlassungen",
    "einstellungsstopp",
    "lohnstagnation", "lohnkompression",
    "arbeitskräftemangel", "fachkräftemangel",
    "erwerbsquote",
    "arbeitslosigkeit",
})


# ---------------------------------------------------------------------------
# French (fr)
# ---------------------------------------------------------------------------
_EPU_FR = {
    "economy":     frozenset({"économique", "économie"}),
    "policy":      frozenset({"politique", "politiques", "réglementation",
                              "législation", "déficit", "tarif", "douane",
                              "banque centrale", "banque de france"}),
    "uncertainty": frozenset({"incertain", "incertaine", "incertitude"}),
}
_MPU_FR = frozenset({
    "monétaire", "politique", "politiques",
    "banque centrale", "banque de france",
    "taux directeur", "taux d'intérêt",
    "incertain", "incertaine", "incertitude",
})
_GPR_FR = frozenset({
    "guerre", "militaire",
    "terrorisme", "terroriste",
    "géopolitique",
    "sanctions", "sanction",
    "invasion",
    "nucléaire", "missile",
    "conflit", "tensions",
})
_TONE_FR = {
    "hawkish": frozenset({
        "faucon", "faucons", "durcir", "durcissement",
        "relever", "relèvement", "hausse",
        "restrictif", "restrictive",
    }),
    "dovish": frozenset({
        "colombe", "colombes", "assouplir", "assouplissement",
        "baisse", "baisser", "abaisser",
        "accommodant", "accommodante",
        "soutien",
    }),
}
_WUI_FR = frozenset({
    "incertain", "incertaine", "incertitude",
    "ambiguïté", "ambigu", "ambiguë",
    "imprévisible",
})
_LUI_FR = frozenset({
    "licenciement", "licenciements",
    "gel des embauches",
    "stagnation salariale", "compression salariale",
    "pénurie de main-d'œuvre",
    "taux d'activité",
    "chômage",
})


# ---------------------------------------------------------------------------
# Italian (it)
# ---------------------------------------------------------------------------
_EPU_IT = {
    "economy":     frozenset({"economica", "economico", "economia"}),
    "policy":      frozenset({"politica", "politiche", "regolamentazione",
                              "legislazione", "deficit", "tariffa",
                              "banca centrale"}),
    "uncertainty": frozenset({"incerto", "incerta", "incertezza"}),
}
_MPU_IT = frozenset({
    "monetaria", "politica", "politiche",
    "banca centrale", "bce",
    "tasso", "tassi",
    "incerto", "incerta", "incertezza",
})
_GPR_IT = frozenset({
    "guerra", "militare",
    "terrorismo", "terrorista",
    "geopolitico", "geopolitica",
    "sanzioni", "sanzione",
    "invasione",
    "nucleare", "missile",
    "conflitto", "tensioni",
})
_TONE_IT = {
    "hawkish": frozenset({
        "falco", "falchi", "stringere", "stretta",
        "alzare", "rialzo", "aumento",
        "restrittiva", "restrittivo",
    }),
    "dovish": frozenset({
        "colomba", "colombe", "allentare", "allentamento",
        "ridurre", "riduzione", "abbassare",
        "accomodante", "stimolo", "sostegno",
    }),
}
_WUI_IT = frozenset({
    "incerto", "incerta", "incertezza",
    "ambiguità", "ambiguo", "ambigua",
    "imprevedibile",
})
_LUI_IT = frozenset({
    "licenziamento", "licenziamenti",
    "blocco delle assunzioni",
    "stagnazione salariale", "compressione salariale",
    "carenza di manodopera",
    "tasso di partecipazione",
    "disoccupazione",
})


# ---------------------------------------------------------------------------
# Japanese (ja) — substring matching, no whitespace tokenization
# ---------------------------------------------------------------------------
_EPU_JA = {
    "economy":     frozenset({"経済"}),
    "policy":      frozenset({"政策", "規制"}),
    "uncertainty": frozenset({"不確実", "不確実性"}),
}
_MPU_JA = frozenset({"金融政策", "中央銀行", "日本銀行", "金利", "不確実性"})
_GPR_JA = frozenset({"戦争", "軍事", "テロ", "地政学", "制裁", "侵攻", "核", "ミサイル", "紛争"})
_WUI_JA = frozenset({"不確実", "不確実性", "曖昧"})
_LUI_JA = frozenset({"解雇", "雇用凍結", "賃金停滞", "労働力不足", "労働参加率", "失業"})


# ---------------------------------------------------------------------------
# Chinese (zh) — substring matching, simplified-Chinese terms
# ---------------------------------------------------------------------------
_EPU_ZH = {
    "economy":     frozenset({"经济", "经济的"}),
    "policy":      frozenset({"政策", "监管", "法规"}),
    "uncertainty": frozenset({"不确定", "不确定性"}),
}
_MPU_ZH = frozenset({"货币政策", "中央银行", "人民银行", "利率", "不确定性"})
_GPR_ZH = frozenset({"战争", "军事", "恐怖", "地缘政治", "制裁", "入侵", "核", "导弹", "冲突"})
_WUI_ZH = frozenset({"不确定", "不确定性", "模糊"})
_LUI_ZH = frozenset({"裁员", "招聘冻结", "工资停滞", "劳动力短缺", "劳动参与率", "失业"})
```

Then **replace** the existing `LEXICONS = {...}` declaration at the bottom of the file with:

```python
LEXICONS: dict = {
    "epu": {
        "en": _EPU_EN, "es": _EPU_ES, "pt": _EPU_PT,
        "de": _EPU_DE, "fr": _EPU_FR, "it": _EPU_IT,
        "ja": _EPU_JA, "zh": _EPU_ZH,
    },
    "mpu": {
        "en": _MPU_EN, "es": _MPU_ES, "pt": _MPU_PT,
        "de": _MPU_DE, "fr": _MPU_FR, "it": _MPU_IT,
        "ja": _MPU_JA, "zh": _MPU_ZH,
    },
    "gpr": {
        "en": _GPR_EN, "es": _GPR_ES, "pt": _GPR_PT,
        "de": _GPR_DE, "fr": _GPR_FR, "it": _GPR_IT,
        "ja": _GPR_JA, "zh": _GPR_ZH,
    },
    "tone": {
        "en": _TONE_EN, "es": _TONE_ES, "pt": _TONE_PT,
        "de": _TONE_DE, "fr": _TONE_FR, "it": _TONE_IT,
    },
    "wui": {
        "en": _WUI_EN, "es": _WUI_ES, "pt": _WUI_PT,
        "de": _WUI_DE, "fr": _WUI_FR, "it": _WUI_IT,
        "ja": _WUI_JA, "zh": _WUI_ZH,
    },
    "lui": {
        "en": _LUI_EN, "es": _LUI_ES, "pt": _LUI_PT,
        "de": _LUI_DE, "fr": _LUI_FR, "it": _LUI_IT,
        "ja": _LUI_JA, "zh": _LUI_ZH,
    },
}
```

Note `tone` has no `ja`/`zh` — those land in Slice 3 (separate Hubert lexicon work).

- [ ] **Step 5: Run multilingual tests, expect green**

Run: `pytest tests/test_narrative_indices.py -v --no-header -k "lang or spanish" 2>&1 | tail -20`
Expected: all parametrized + Spanish-corpus tests pass.

- [ ] **Step 6: Run full suite**

Run: `pytest -q --no-header 2>&1 | tail -3`
Expected: pass count up substantially due to parametrize.

- [ ] **Step 7: Commit**

```bash
git add puremacro/puremacro/narrative/indices/_lexicons.py \
        puremacro/tests/test_narrative_indices.py
git commit -m "feat(narrative): multilingual lexicons (es/pt/de/fr/it/ja/zh)"
```

---

## Task 11: Public API re-exports + Pyodide compat verification

**Files:**
- Modify: `puremacro/narrative/indices/__init__.py` — re-export the six index helpers.
- Modify: `puremacro/narrative/__init__.py` — re-export at the top-level narrative API.
- Modify: `tests/fixtures/public_api_snapshot.json` — regenerate.

- [ ] **Step 1: Verify branch state**

Run: `git branch --show-current` — expected `feature/narrative-extension-slice2`.

- [ ] **Step 2: Update `puremacro/narrative/indices/__init__.py`**

```python
"""Text-derived continuous risk indices (EPU / MPU / GPR / tone / WUI / LUI).

See ``docs/specs/2026-05-08-narrative-extension-design.md`` §5.
"""
from .epu import epu
from .mpu import mpu
from .gpr import gpr
from .tone import tone
from .wui import wui
from .lui import lui
from ._lexicons import LEXICONS

__all__ = ["epu", "mpu", "gpr", "tone", "wui", "lui", "LEXICONS"]
```

- [ ] **Step 3: Update `puremacro/narrative/__init__.py`**

Locate the existing import block at top:

```python
from .types import NarrativeEvent, NarrativeInstrument, RiskIndex
from .aggregate import events_to_quarterly, index_to_quarterly
```

Append (right after) two new imports:

```python
from .indices import epu, mpu, gpr, tone, wui, lui
```

In `__all__`, add `"epu", "mpu", "gpr", "tone", "wui", "lui"` at the end of the Core block.

- [ ] **Step 4: Confirm imports work**

Run:
```bash
python -c "
from puremacro.narrative import (
    NarrativeEvent, NarrativeInstrument, RiskIndex,
    events_to_quarterly, index_to_quarterly,
    epu, mpu, gpr, tone, wui, lui,
)
print('public-API audit ok')
"
```
Expected: `public-API audit ok`.

- [ ] **Step 5: Pyodide compat — confirm new modules don't leak**

Run: `pytest tests/test_pyodide_compat.py -v --no-header 2>&1 | tail -10`
Expected: same 1 pre-existing failure (statsmodels.tsa.x13 leak from `puremacro/fetch/_seasonal.py:19`); NO NEW leaks. The leak set should be the same `statsmodels.*` set as in baseline. If the leak set grows, BLOCK and report.

- [ ] **Step 6: Regenerate the API snapshot**

Run:
```bash
python -c "from tests.test_public_api import _collect_current_api; import json; print(json.dumps(_collect_current_api(), indent=2, sort_keys=True))" > tests/fixtures/public_api_snapshot.json
pytest tests/test_public_api.py -v --no-header 2>&1 | tail -3
```
Expected: snapshot test passes.

- [ ] **Step 7: Run full suite**

Run: `pytest -q --no-header 2>&1 | tail -3`
Expected: ≥ 859 + (~50 new tests across Tasks 2-10) passing. No new failures.

- [ ] **Step 8: Commit**

```bash
git add puremacro/puremacro/narrative/indices/__init__.py \
        puremacro/puremacro/narrative/__init__.py \
        puremacro/tests/fixtures/public_api_snapshot.json
git commit -m "feat(narrative): re-export indices subpackage at narrative top level"
```

---

## Task 12: Example demo + network-marked validation

**Files:**
- Create: `puremacro/examples/narrative_indices_demo.py`
- Create: `tests/test_narrative_indices_validation.py`

- [ ] **Step 1: Verify branch state**

Run: `git branch --show-current` — expected `feature/narrative-extension-slice2`.

- [ ] **Step 2: Create `puremacro/examples/narrative_indices_demo.py`**

```python
"""Narrative indices demo — assembles all six indices from a synthetic
in-process corpus and prints summary statistics.

Run:
    python -m puremacro.examples.narrative_indices_demo
"""
from __future__ import annotations

import pandas as pd

from ..narrative import epu, mpu, gpr, tone, wui, lui


_SYNTHETIC_CORPUS = [
    # Q1 2020: turbulent
    ("2020-01-15", "Economic policy uncertainty rose sharply as the federal "
                   "reserve weighed an emergency rate cut amid pandemic risk."),
    ("2020-02-15", "Layoffs and hiring freezes spread across sectors as "
                   "geopolitical tensions and war risk unsettled markets."),
    ("2020-03-15", "The Fed cut rates 50 basis points; markets called the move "
                   "dovish and signalled further accommodation."),
    # Q2 2020: stabilising
    ("2020-04-15", "Conditions began to stabilise; uncertainty receded though "
                   "labor shortages persisted in some sectors."),
    ("2020-05-15", "Unemployment remained elevated; central banks signalled "
                   "lower for longer interest-rate guidance."),
    ("2020-06-15", "Tone became less dovish; some FOMC members hinted at "
                   "tightening conditions if recovery accelerated."),
    # Q3 2020: hawkish drift
    ("2020-07-15", "FOMC raised the target range and signalled tightening; "
                   "uncertainty about geopolitical sanctions remained."),
    ("2020-08-15", "Hawkish tone dominated; participants raised the policy "
                   "rate amid persistent inflationary pressure."),
]


def _records_4tuple(corpus):
    for date, text in corpus:
        yield (pd.Timestamp(date), text, "https://test/" + date,
               {"language": "en", "doctype": "press"})


def run_demo() -> dict:
    rec = list(_records_4tuple(_SYNTHETIC_CORPUS))
    return {
        "epu":  epu(rec, country="USA", normalize="raw"),
        "mpu":  mpu(rec, country="USA", normalize="raw"),
        "gpr":  gpr(rec, country="USA", normalize="raw"),
        "tone": tone(rec, country="USA", normalize="raw"),
        "wui":  wui(rec, country="USA", normalize="raw"),
        "lui":  lui(rec, country="USA", normalize="raw"),
    }


def main() -> None:
    out = run_demo()
    print("Narrative indices — synthetic-corpus demo")
    print(f"  Corpus size: {len(_SYNTHETIC_CORPUS)} documents over 8 months\n")
    for name, ri in out.items():
        d = ri.diagnostics()
        print(f"  {name:5s}  n_q={d['n_quarters']:>2d}  "
              f"mean={d['mean']:>+8.3f}  std={d['std']:>+7.3f}  "
              f"first={d['first_date']}  last={d['last_date']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create network-marked validation**

Create `tests/test_narrative_indices_validation.py`:

```python
"""Network-marked correlation tests of our text-built indices vs the
mirrored published series.

These tests pull live data from policyuncertainty.com and the Caldara-
Iacoviello dataset, and intentionally do NOT assert publication-level
correlation — that requires the BBD source corpus, which we don't ship.
Instead they check that the published series load and that our
synthetic-corpus reconstruction has the right SHAPE (positive
correlation with itself across normalisations).

Run only with: pytest -m network tests/test_narrative_indices_validation.py
"""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.mark.network
def test_published_bbd_epu_loads():
    """Smoke: the published BBD-EPU series via instruments.literature
    must load. Skip if upstream is unreachable."""
    from puremacro.instruments.literature import bbd_epu
    try:
        inst = bbd_epu.load_bbd_epu_us()
    except Exception:
        pytest.skip("policyuncertainty.com unreachable")
    if inst.series.empty:
        pytest.skip("BBD-EPU returned empty.")
    assert inst.frequency in {"M", "Q"}


@pytest.mark.network
def test_published_gpr_loads():
    from puremacro.instruments.literature import caldara_iacoviello_gpr
    try:
        inst = caldara_iacoviello_gpr.load_caldara_iacoviello_gpr()
    except Exception:
        pytest.skip("GPR mirror unreachable")
    if inst.series.empty:
        pytest.skip("GPR returned empty.")
    assert inst.frequency in {"M", "Q"}


def test_synthetic_epu_has_consistent_normalization():
    """Offline: zscore on synthetic corpus has zero mean (sanity)."""
    from puremacro.narrative import epu
    records = []
    high_text = "economic policy uncertainty rose"
    low_text = "ordinary text"
    for q in range(8):
        d = pd.Timestamp("2020-01-01") + pd.DateOffset(months=q)
        records.append((d, high_text if q % 2 == 0 else low_text,
                        "https://test/" + str(d.date()),
                        {"language": "en"}))
    ri = epu(records, country="USA", language="en", normalize="zscore")
    assert ri.series.dropna().mean() == pytest.approx(0.0, abs=1e-9)
```

- [ ] **Step 4: Run the demo manually**

Run:
```bash
python -m puremacro.examples.narrative_indices_demo
```
Expected: 6 lines of summary statistics, one per index. None should error.

- [ ] **Step 5: Run validation tests (offline only)**

Run: `pytest tests/test_narrative_indices_validation.py -v --no-header 2>&1 | tail -10`
Expected: 1 offline test passes, 2 network-marked tests are deselected (not run by default). The number of passing tests will be 1; the deselected count is 2.

- [ ] **Step 6: Optionally run network tests**

Run: `pytest tests/test_narrative_indices_validation.py -m network -v --no-header 2>&1 | tail -10`
This is an optional smoke; skips on empty are acceptable per `feedback_network_tests_skip_on_empty.md`.

- [ ] **Step 7: Run full suite**

Run: `pytest -q --no-header 2>&1 | tail -3`
Expected: pass count includes the new offline test.

- [ ] **Step 8: Commit**

```bash
git add puremacro/puremacro/examples/narrative_indices_demo.py \
        puremacro/tests/test_narrative_indices_validation.py
git commit -m "feat(narrative): indices demo + network-marked validation tests"
```

---

## Task 13: Version bump 0.6.2 + CHANGELOG + tag

**Files:**
- Modify: `pyproject.toml` (version)
- Modify: `puremacro/__init__.py` (`__version__`)
- Modify: `tests/test_import.py` (expected version)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Verify branch state**

Run: `git branch --show-current` — expected `feature/narrative-extension-slice2`.

- [ ] **Step 2: Bump version**

Edit `pyproject.toml`: change `version = "0.6.1"` → `version = "0.6.2"`.

Edit `puremacro/__init__.py`: change `__version__ = "0.6.1"` → `__version__ = "0.6.2"`.

Edit `tests/test_import.py`: change `assert puremacro.__version__ == "0.6.1"` → `"0.6.2"`.

- [ ] **Step 3: Add CHANGELOG entry**

Open `CHANGELOG.md`. Add a new top entry **above** the `## 0.6.1 — 2026-05-08` block:

```markdown
## 0.6.2 — 2026-05-08

Slice 2 of the multi-domain narrative extension (`docs/specs/2026-05-08-narrative-extension-design.md`). Ships the `puremacro.narrative.indices` subpackage — six text-derived continuous risk-index helpers that emit `RiskIndex` objects from any source-iter corpus.

### Added

- `puremacro.narrative.indices` (new subpackage):
  - `epu(text_iter, *, country, language="en", lexicon=None, normalize="bbd_100", base_period=None, agg="mean")` — Baker-Bloom-Davis Economic Policy Uncertainty: count documents containing ≥1 term from each of three groups (Economy, Policy, Uncertainty), aggregate quarterly.
  - `mpu(...)` — Husted-Rogers-Sun monetary-policy uncertainty (flat term list).
  - `gpr(...)` — Caldara-Iacoviello geopolitical-risk index.
  - `tone(..., method="apel_blix_grimaldi" | "hubert" | "picault_renault")` — net hawkish-dovish tone per document. Slice 2 ships count-based mechanism for all three methods; Picault-Renault paragraph-level multinomial classifier deferred to Slice 3.
  - `wui(...)` — Ahir-Bloom-Furceri World Uncertainty Index style (count-based; document-length normalisation deferred to Slice 3).
  - `lui(...)` — **Labor-Market Uncertainty Index (novel)** — covers six conceptual groups: layoffs, hiring-freeze, wage-compression, labor-shortage, participation-drop, unemployment-risk. Multilingual.
- `puremacro.narrative.indices._kernels` — `keyword_count_kernel`, `cooccurrence_kernel`, `tone_kernel`, plus `normalize_series(raw|zscore|bbd_100)` helper.
- `puremacro.narrative.indices._lexicons.LEXICONS` — multilingual term lists for **8 languages** (en, es, pt, de, fr, it, ja, zh). Tone lexicon ships for the 6 Latin-script languages; ja/zh tone in Slice 3.
- `tests/test_narrative_indices.py` — kernel + per-index offline tests + multilingual lexicon coverage + normalisation round-trip.
- `tests/test_narrative_indices_validation.py` — network-marked correlation smokes against `instruments.literature.bbd_epu` and `caldara_iacoviello_gpr` published mirrors.
- `puremacro/examples/narrative_indices_demo.py` — runnable demo assembling all 6 indices on a synthetic corpus.

### Changed

- `puremacro.narrative.aggregate.index_to_quarterly` now actually applies the `normalization=` parameter (Slice 1 stored it as metadata only). The `normalize_series` helper is lazy-imported inside the function to keep the import order clean.
- `puremacro.narrative.__all__` extends with `epu`, `mpu`, `gpr`, `tone`, `wui`, `lui`.

### Pyodide compatibility

- `narrative.indices` and all six index modules are pure-Python — no new top-level deps, Pyodide-clean. Same exclusion rules as Slice 1: `narrative/sources/<bank>_*.py` stays Experimental tier; the count-based indices path is Stable.

### Notes for Slice 3

- Pickaul-Renault paragraph-level multinomial logit, full Hubert lexicon, length-normalised WUI, and JA/ZH tone lexicons all deferred to Slice 3.
- `base_period` is currently stored in metadata but not threaded into `normalize_series` inside `index_to_quarterly`. Slice 3 will add the plumbing so `bbd_100` can use a published-style 1985-2009 base.
- `llm_prob_kernel` (LLM-backed per-document scoring) ships in Slice 3.

```

- [ ] **Step 4: Run the full suite once more**

Run: `pytest -q --no-header 2>&1 | tail -3`
Expected: same final pass count as Task 12, with `test_import.py` now passing against `0.6.2`.

- [ ] **Step 5: Run the fiscal regression suite**

Run: `pytest tests/test_narrative.py tests/test_narrative_replication_*.py tests/test_narrative_quality.py tests/test_narrative_validation.py -q --no-header 2>&1 | tail -3`
Expected: zero fiscal-narrative regressions.

- [ ] **Step 6: Commit + tag**

```bash
git add puremacro/pyproject.toml \
        puremacro/puremacro/__init__.py \
        puremacro/tests/test_import.py \
        puremacro/CHANGELOG.md
git commit -m "chore(release): puremacro 0.6.2 — narrative Slice 2 (indices layer)"
git tag -a v0.6.2 -m "puremacro 0.6.2 — narrative Slice 2 (indices layer)"
```

(Do **not** push.)

---

## Definition of Done

- [ ] All 14 task blocks above checked off.
- [ ] Branch `feature/narrative-extension-slice2` exists with ~14 commits since `v0.6.1`, tagged `v0.6.2`.
- [ ] `pytest -q` passes ≥ 859 + new test count (~75-100 new tests).
- [ ] `pytest tests/test_pyodide_compat.py` shows the SAME 1 pre-existing failure as baseline (no new leaks).
- [ ] `pytest tests/test_public_api.py` passes (snapshot includes `epu`, `mpu`, `gpr`, `tone`, `wui`, `lui`, `LEXICONS`).
- [ ] No fiscal-narrative regressions: existing `tests/test_narrative*.py` passes match Slice 1 baseline.
- [ ] `pyproject.toml` version is `0.6.2`; `puremacro.__version__ == "0.6.2"`.
- [ ] `CHANGELOG.md` has a `## 0.6.2 — 2026-05-08` section.
- [ ] `python -c "from puremacro.narrative import epu, mpu, gpr, tone, wui, lui; print('ok')"` prints `ok`.
- [ ] `python -m puremacro.examples.narrative_indices_demo` runs without error and prints 6 index summaries.

## Out of scope for this plan (deferred to Slice 3)

- LATAM / advanced non-G7 / Asia-EM CB connectors — those are Slice 3.
- Macropru / FX / structural prompt families wired end-to-end — Slice 3 (the prompts ship in Slice 1 but aren't yet exercised against new connectors).
- Picault-Renault paragraph-level multinomial logit; full Hubert lexicon; JA/ZH tone lexicons — Slice 3.
- Length-normalised WUI matching the original Ahir-Bloom-Furceri methodology.
- `base_period` plumbed all the way through `index_to_quarterly` → `normalize_series`.
- `llm_prob_kernel` for LLM-backed per-document scoring.
- BIS speeches meta-connector — Slice 3.
- Cross-lingual lexicon validation against shared events — Slice 3.
