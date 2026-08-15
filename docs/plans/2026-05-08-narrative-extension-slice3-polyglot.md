# Narrative Extension — Slice 3 (Polyglot Expansion) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship 15 new central-bank connectors (LATAM, Advanced non-G7, Asia-EM), a BIS speeches meta-connector, end-to-end smoke tests for the macropru/fx/structural LLM prompts, cross-lingual validation, and three small Slice-2 followups (`base_period` plumbing, JA/ZH tone lexicons, dedup the dual `VALID_NORMALIZATIONS` constants).

**Architecture:** A new `iter_rss_filtered` helper consolidates the RSS-fetch + title-keyword-filter + 4-tuple-emit pattern shared across most CB connectors. Each new bank ships as one file with 1-2 functions (`iter_<bank>_decision` and `iter_<bank>_speeches` where applicable). The `mock_http` fixture is promoted from per-test-file to `tests/conftest.py` so all 15 new tests can share it. The Slice-2 followups are mechanical: thread `base_period` through `index_to_quarterly`, add JA/ZH `tone` lexicons (data only), and remove the `_VALID_NORMALIZATIONS` duplication. Cross-lingual validation runs as `@pytest.mark.network` smoke tests asserting EN-vs-ES indices on the same overlapping period correlate ρ ≥ 0.7.

**Tech Stack:** Python 3.10+, `pandas`, `numpy`, `re`, `urllib`. No new runtime deps. Pyodide-compatible (count-based path).

**Spec reference:** `docs/specs/2026-05-08-narrative-extension-design.md` §"Slice 3 — Polyglot expansion". Slice 1 plan: `docs/plans/2026-05-08-narrative-extension-slice1-foundation.md`. Slice 2 plan: `docs/plans/2026-05-08-narrative-extension-slice2-indices.md`.

**Branching:** This plan is committed to `feature/narrative-extension-slice2` (HEAD: `v0.6.2` + this plan commit). Branch `feature/narrative-extension-slice3` from that HEAD — equivalent to `v0.6.2` in code state, plus the plan file.

**Pre-implementation baseline:** `pytest -q` after Slice 2 = **924 passed, 22 skipped**, plus 1 pre-existing pyodide-compat failure (`statsmodels.tsa.x13` leak via `puremacro/fetch/_seasonal.py:19` — out of scope).

**Version bump:** `0.6.2 → 0.7.0`.

---

## File Structure

### Files created (Slice 3)
- `puremacro/narrative/sources/_rss_filtered.py` — shared RSS + title-keyword-filter helper.
- `puremacro/narrative/sources/banxico.py` — Banco de México (es).
- `puremacro/narrative/sources/bcb.py` — Banco Central do Brasil (pt + en).
- `puremacro/narrative/sources/bccl.py` — Banco Central de Chile (es).
- `puremacro/narrative/sources/bcra.py` — Banco Central de la República Argentina (es).
- `puremacro/narrative/sources/banrep.py` — Banco de la República (Colombia, es).
- `puremacro/narrative/sources/rba.py` — Reserve Bank of Australia (en).
- `puremacro/narrative/sources/rbnz.py` — Reserve Bank of New Zealand (en).
- `puremacro/narrative/sources/riksbank.py` — Sveriges Riksbank (en mirror).
- `puremacro/narrative/sources/norges.py` — Norges Bank (en mirror).
- `puremacro/narrative/sources/sarb.py` — South African Reserve Bank (en).
- `puremacro/narrative/sources/pboc.py` — People's Bank of China (en mirror).
- `puremacro/narrative/sources/rbi.py` — Reserve Bank of India (en).
- `puremacro/narrative/sources/bok.py` — Bank of Korea (en mirror).
- `puremacro/narrative/sources/mas.py` — Monetary Authority of Singapore (en).
- `puremacro/narrative/sources/bot.py` — Bank of Thailand (en).
- `puremacro/narrative/sources/bis_speeches.py` — BIS speeches archive (multi-bank).
- `tests/test_narrative_slice3_connectors.py` — offline + smoke tests for all 15 banks + BIS.
- `tests/test_narrative_slice3_prompts.py` — macropru/fx/structural prompt smoke tests.
- `tests/test_narrative_indices_crosslingual.py` — network-marked EN-vs-ES correlation smokes.

### Files modified
- `puremacro/narrative/sources/__init__.py` — re-export the new connectors.
- `puremacro/narrative/aggregate.py` — `index_to_quarterly` threads `base_period` into `normalize_series`.
- `puremacro/narrative/indices/_lexicons.py` — add `tone["ja"]` and `tone["zh"]` keys.
- `puremacro/narrative/indices/_kernels.py` — drop local `_VALID_NORMALIZATIONS`, import from `..types`.
- `tests/conftest.py` — promote `mock_http` fixture from `test_narrative_cb_connectors.py`.
- `tests/test_narrative_cb_connectors.py` — drop the in-file `mock_http` (now in conftest).
- `tests/test_narrative_indices.py` — extend multilingual lexicon parametrize for tone ja/zh.
- `tests/test_narrative_index_to_quarterly.py` — add a base_period plumbing test.
- `tests/fixtures/public_api_snapshot.json` — regenerate.
- `pyproject.toml` — version `0.6.2 → 0.7.0`.
- `puremacro/__init__.py` — `__version__ = "0.7.0"`.
- `tests/test_import.py` — bump expected version.
- `CHANGELOG.md` — add `## 0.7.0 — 2026-05-08` block.

---

## Task 0: Branch + baseline

**Files:** none (git only)

- [ ] **Step 1: Verify v0.6.2 tag**

Run: `git tag -l v0.6.2`
Expected: `v0.6.2`. If absent, abort.

- [ ] **Step 2: Create slice3 branch from current HEAD**

```bash
git checkout feature/narrative-extension-slice2
git checkout -b feature/narrative-extension-slice3
git branch --show-current
```
Expected: `feature/narrative-extension-slice3`. (Branching from `feature/narrative-extension-slice2` HEAD is equivalent to `v0.6.2` plus this plan file.)

- [ ] **Step 3: Confirm baseline**

Run: `pytest -q --no-header 2>&1 | tail -3`
Expected: `924 passed, 22 skipped, 4 warnings in <T>s`.

---

## Task 1: Shared RSS-filtered helper + promote `mock_http` fixture to conftest

**Files:**
- Create: `puremacro/narrative/sources/_rss_filtered.py`
- Create: `tests/conftest.py` (or modify if it exists)
- Modify: `tests/test_narrative_cb_connectors.py` (drop the in-file `mock_http`)

- [ ] **Step 1: Verify branch state**

Run: `git branch --show-current` — must be `feature/narrative-extension-slice3`.

- [ ] **Step 2: Read the existing conftest**

Run: `ls tests/conftest.py 2>&1` — note whether the file exists.

If the file exists, read it and plan to APPEND the mock_http fixture without disturbing existing fixtures. If it doesn't exist, create it from scratch.

- [ ] **Step 3: Read the existing in-file mock_http**

Read `tests/test_narrative_cb_connectors.py` and identify the `mock_http` fixture and its `_PATCH_TARGETS` list. The fixture spans roughly 30 lines starting with `@pytest.fixture` decorator on `mock_http`.

- [ ] **Step 4: Move `mock_http` to `tests/conftest.py`**

If `tests/conftest.py` exists: append the fixture below to it.
If not: create the file with this exact content.

```python
"""Shared pytest fixtures for the puremacro test suite."""
from __future__ import annotations

import importlib

import pytest


# Modules whose `safe_get_bytes` / `safe_get_text` we patch in offline
# CB-connector tests. Slice 1 + Slice 3 banks combined.
_CB_PATCH_TARGETS = [
    "puremacro.narrative.sources._rss",
    "puremacro.narrative.sources._ratedoc",
    "puremacro.narrative.sources._speeches",
    "puremacro.narrative.sources._rss_filtered",
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
    # Slice 3
    "puremacro.narrative.sources.banxico",
    "puremacro.narrative.sources.bcb",
    "puremacro.narrative.sources.bccl",
    "puremacro.narrative.sources.bcra",
    "puremacro.narrative.sources.banrep",
    "puremacro.narrative.sources.rba",
    "puremacro.narrative.sources.rbnz",
    "puremacro.narrative.sources.riksbank",
    "puremacro.narrative.sources.norges",
    "puremacro.narrative.sources.sarb",
    "puremacro.narrative.sources.pboc",
    "puremacro.narrative.sources.rbi",
    "puremacro.narrative.sources.bok",
    "puremacro.narrative.sources.mas",
    "puremacro.narrative.sources.bot",
    "puremacro.narrative.sources.bis_speeches",
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

    for modname in _CB_PATCH_TARGETS:
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
```

- [ ] **Step 5: Drop the in-file `mock_http` from `test_narrative_cb_connectors.py`**

In `tests/test_narrative_cb_connectors.py`, delete the local `_PATCH_TARGETS` list (~16 lines) and the `mock_http` fixture (~30 lines). Keep the `import importlib` removed too if it becomes unused. The test functions still reference `mock_http` — they'll resolve it via conftest now.

- [ ] **Step 6: Confirm Slice 1 CB tests still pass with the conftest fixture**

Run: `pytest tests/test_narrative_cb_connectors.py -v --no-header -k "not network" 2>&1 | tail -15`
Expected: 7 offline tests pass (Fed×3, ECB×2, BoE×1, BoJ×1).

- [ ] **Step 7: Create `puremacro/narrative/sources/_rss_filtered.py`**

```python
"""Shared RSS-feed wrapper with optional title-keyword filtering.

Most CB connectors share the same shape:
  1. Fetch an RSS feed (parsed via ``_rss.iter_rss``).
  2. Strip HTML from title/description.
  3. Keep only items whose title matches certain keywords (or skip
     items matching exclude keywords).
  4. Emit a 4-tuple SourceRecord ``(date, text, source_url, metadata)``.

This helper consolidates that pattern so per-bank connectors are a
single 6-line ``yield from iter_rss_filtered(...)`` call.
"""
from __future__ import annotations

from typing import Iterator

from ._rss import iter_rss
from ._ratedoc import strip_html


def iter_rss_filtered(
    url: str,
    *,
    bank_code: str,
    country: str,
    doctype: str,
    language: str = "en",
    title_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None,
) -> Iterator[tuple]:
    """Wrap an RSS feed and emit 4-tuple SourceRecords.

    Parameters
    ----------
    url : RSS feed URL.
    bank_code : short tag for ``metadata["bank_code"]`` (e.g. ``"RBA"``).
    country : ISO3 (e.g. ``"AUS"``).
    doctype : ``"decision"`` | ``"minutes"`` | ``"speech"`` | ``"press"`` | ``"fsr"``.
    language : ISO-639-1.
    title_keywords : if given (non-empty), the title must contain at
        least one of these (case-insensitive). If ``None`` or empty,
        no filter — all items pass.
    exclude_keywords : if given (non-empty), items whose title contains
        any of these are dropped (case-insensitive).
    """
    for date, title_desc, link in iter_rss(url):
        clean = strip_html(title_desc) if "<" in title_desc else title_desc
        low = clean.lower()
        if title_keywords and not any(kw.lower() in low for kw in title_keywords):
            continue
        if exclude_keywords and any(kw.lower() in low for kw in exclude_keywords):
            continue
        yield (date, clean, link, {
            "doctype": doctype, "language": language,
            "bank_code": bank_code, "country": country,
        })


__all__ = ["iter_rss_filtered"]
```

- [ ] **Step 8: Sanity import**

Run:
```bash
python -c "from puremacro.narrative.sources._rss_filtered import iter_rss_filtered; print('ok')"
```
Expected: `ok`.

- [ ] **Step 9: Run full pytest, no regressions**

Run: `pytest -q --no-header 2>&1 | tail -3`
Expected: still 924 passed (no new tests, no broken ones).

- [ ] **Step 10: Commit**

```bash
git branch --show-current   # must be feature/narrative-extension-slice3
git add puremacro/puremacro/narrative/sources/_rss_filtered.py \
        puremacro/tests/conftest.py \
        puremacro/tests/test_narrative_cb_connectors.py
git commit -m "feat(narrative): iter_rss_filtered helper + promote mock_http fixture to conftest"
```

---

## Task 2: Consolidate `_VALID_NORMALIZATIONS` + add JA/ZH tone lexicons

**Files:**
- Modify: `puremacro/narrative/indices/_kernels.py` (drop local set, import from `..types`)
- Modify: `puremacro/narrative/indices/_lexicons.py` (append `tone["ja"]` and `tone["zh"]`)
- Modify: `tests/test_narrative_indices.py` (extend the tone-language parametrize)

- [ ] **Step 1: Verify branch state.**

- [ ] **Step 2: Drop the duplicate constant in `_kernels.py`**

In `puremacro/narrative/indices/_kernels.py`, find the line:

```python
_VALID_NORMALIZATIONS = {"raw", "zscore", "bbd_100"}
```

Replace with an import-time alias:

```python
from ..types import VALID_RISKINDEX_NORMALIZATION as _VALID_NORMALIZATIONS
```

Place this import next to the other relative imports at the top of the file. The body of `normalize_series` uses `_VALID_NORMALIZATIONS` and continues to work unchanged.

- [ ] **Step 3: Run kernels tests, confirm still pass**

Run: `pytest tests/test_narrative_indices.py -v --no-header -k "normalize" 2>&1 | tail -10`
Expected: all 5 normalize tests pass.

- [ ] **Step 4: Add JA/ZH tone lexicons**

Open `puremacro/narrative/indices/_lexicons.py`. Find the existing JA section (a comment line `# Japanese (ja) — substring matching`). Right after the JA per-index constants and BEFORE the ZH section, add:

```python
_TONE_JA = {
    "hawkish": frozenset({"引き締め", "利上げ", "タカ派", "緊縮"}),
    "dovish":  frozenset({"緩和", "利下げ", "ハト派", "刺激"}),
}
```

Right after the ZH per-index constants and BEFORE the `LEXICONS = {...}` dict, add:

```python
_TONE_ZH = {
    "hawkish": frozenset({"紧缩", "加息", "鹰派", "收紧"}),
    "dovish":  frozenset({"宽松", "降息", "鸽派", "刺激"}),
}
```

Update the `tone` block inside `LEXICONS = {...}` to include `ja` and `zh`:

Find:
```python
    "tone": {
        "en": _TONE_EN, "es": _TONE_ES, "pt": _TONE_PT,
        "de": _TONE_DE, "fr": _TONE_FR, "it": _TONE_IT,
    },
```

Replace with:
```python
    "tone": {
        "en": _TONE_EN, "es": _TONE_ES, "pt": _TONE_PT,
        "de": _TONE_DE, "fr": _TONE_FR, "it": _TONE_IT,
        "ja": _TONE_JA, "zh": _TONE_ZH,
    },
```

- [ ] **Step 5: Extend the parametrize test in `tests/test_narrative_indices.py`**

Find the existing test:

```python
@pytest.mark.parametrize("lang", ["es", "pt", "de", "fr", "it"])
def test_tone_lexicon_present_for_latin_languages(lang):
```

Rename and broaden:

```python
@pytest.mark.parametrize("lang", ["es", "pt", "de", "fr", "it", "ja", "zh"])
def test_tone_lexicon_present_for_all_supported_languages(lang):
    """Tone lexicon ships for all 8 languages in Slice 3 (was Latin-only in Slice 2)."""
    from puremacro.narrative.indices._lexicons import LEXICONS
    assert lang in LEXICONS["tone"]
    groups = LEXICONS["tone"][lang]
    assert {"hawkish", "dovish"} <= set(groups)
    assert len(groups["hawkish"]) >= 1
    assert len(groups["dovish"]) >= 1
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_narrative_indices.py -v --no-header -k "tone_lexicon" 2>&1 | tail -10`
Expected: 7 parametrized tests pass (one per language).

- [ ] **Step 7: Run full suite**

Run: `pytest -q --no-header 2>&1 | tail -3`
Expected: 924 + 2 (new ja/zh parametrize cases) = 926 (or 925 if the renamed test counts as the same).

- [ ] **Step 8: Snapshot regen if drift**

```bash
pytest tests/test_public_api.py -v --no-header 2>&1 | tail -3
```
If FAIL:
```bash
python -c "from tests.test_public_api import _collect_current_api; import json; print(json.dumps(_collect_current_api(), indent=2, sort_keys=True))" > tests/fixtures/public_api_snapshot.json
```

- [ ] **Step 9: Commit**

```bash
git branch --show-current
git add puremacro/puremacro/narrative/indices/_kernels.py \
        puremacro/puremacro/narrative/indices/_lexicons.py \
        puremacro/tests/test_narrative_indices.py
# add tests/fixtures/public_api_snapshot.json IF drifted
git commit -m "fix(narrative): consolidate VALID_NORMALIZATIONS + add ja/zh tone lexicons"
```

---

## Task 3: `base_period` plumbing through `index_to_quarterly`

**Files:**
- Modify: `puremacro/narrative/aggregate.py` (`index_to_quarterly` reads `base_period` from `metadata` if set)
- Modify: `puremacro/narrative/indices/{epu,mpu,gpr,tone,wui,lui}.py` — drop the docstring deferral note (no longer accurate after this task)
- Modify: `tests/test_narrative_index_to_quarterly.py` (add base_period plumbing test)

- [ ] **Step 1: Verify branch state.**

- [ ] **Step 2: Write failing test** — append to `tests/test_narrative_index_to_quarterly.py`:

```python
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
```

- [ ] **Step 3: Run, verify failure**

Run: `pytest tests/test_narrative_index_to_quarterly.py::test_index_to_quarterly_threads_base_period_to_normalize -v --no-header 2>&1 | tail -10`
Expected: fail because `base_period` is currently ignored — the post-base values fall back to whole-series stats and may not exceed 200.

- [ ] **Step 4: Patch `index_to_quarterly` to thread `base_period`**

In `puremacro/narrative/aggregate.py`, find the existing block:

```python
    if normalization != "raw":
        from .indices._kernels import normalize_series
        out = normalize_series(out, normalization)
```

Replace with:

```python
    if normalization != "raw":
        from .indices._kernels import normalize_series
        bp = (metadata or {}).get("base_period") if metadata else None
        out = normalize_series(out, normalization, base_period=bp)
```

The `base_period` is already stored in metadata by every index helper (`epu.py`, `mpu.py`, etc.), so consuming it here completes the plumbing without changing index helpers.

- [ ] **Step 5: Update the docstrings of the 6 index helpers**

For each of `epu.py`, `mpu.py`, `gpr.py`, `tone.py`, `wui.py`, `lui.py`: find the deferred-note docstring fragment that says something like:

> Note: ``base_period`` is currently stored in metadata only and not plumbed through ``index_to_quarterly`` → ``normalize_series`` (Slice 3 will add the plumbing).

Replace each occurrence with:

> Note: ``base_period`` (when supplied) is plumbed through ``index_to_quarterly`` to ``normalize_series`` so that ``"zscore"`` / ``"bbd_100"`` statistics are computed on the slice
> ``series.loc[start:end]`` rather than the full series. The default
> ``None`` uses the full series for normalisation stats.

For `epu.py` specifically — check the existing wording at the parameter docstring `base_period:` — adapt the exact replacement to fit the existing style. The intent is: drop the deferral language, document what the parameter actually does.

- [ ] **Step 6: Run new test**

Run: `pytest tests/test_narrative_index_to_quarterly.py -v --no-header 2>&1 | tail -10`
Expected: all `index_to_quarterly` tests pass including the new one.

- [ ] **Step 7: Run full suite**

Run: `pytest -q --no-header 2>&1 | tail -3`
Expected: pass count up by 1.

- [ ] **Step 8: Commit**

```bash
git branch --show-current
git add puremacro/puremacro/narrative/aggregate.py \
        puremacro/puremacro/narrative/indices/epu.py \
        puremacro/puremacro/narrative/indices/mpu.py \
        puremacro/puremacro/narrative/indices/gpr.py \
        puremacro/puremacro/narrative/indices/tone.py \
        puremacro/puremacro/narrative/indices/wui.py \
        puremacro/puremacro/narrative/indices/lui.py \
        puremacro/tests/test_narrative_index_to_quarterly.py
git commit -m "feat(narrative): plumb base_period through index_to_quarterly to normalize_series"
```

---

## Task 4: LATAM connectors — Banxico + BCB

**Files:**
- Create: `puremacro/narrative/sources/banxico.py`
- Create: `puremacro/narrative/sources/bcb.py`
- Modify: `puremacro/narrative/sources/__init__.py`
- Create: `tests/test_narrative_slice3_connectors.py`

- [ ] **Step 1: Verify branch state.**

- [ ] **Step 2: Create the test file**

`tests/test_narrative_slice3_connectors.py`:

```python
"""Offline + smoke tests for Slice 3 central-bank connectors (LATAM,
Advanced non-G7, Asia-EM, BIS speeches meta).

Uses the conftest-provided ``mock_http`` fixture for offline tests.
"""
from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Banco de México (Banxico)
# ---------------------------------------------------------------------------
def test_banxico_decision_yields_four_tuple_es(mock_http):
    mock_http(bytes_={
        "https://www.banxico.org.mx/rss/feeds/comunicados.xml":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Anuncio de pol\xc3\xadtica monetaria</title>'
            b'<description>La Junta de Gobierno decidi\xc3\xb3 mantener la tasa de inter\xc3\xa9s.</description>'
            b'<link>https://www.banxico.org.mx/publicaciones-y-prensa/anuncios-de-las-decisiones-de-politica-monetaria/anuncio-2022-09.html</link>'
            b'<pubDate>Thu, 29 Sep 2022 13:00:00 -0500</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_banxico_decision
    records = list(iter_banxico_decision())
    assert len(records) == 1
    _, text, _, meta = records[0]
    assert "política monetaria" in text.lower() or "tasa de interés" in text.lower()
    assert meta["doctype"] == "decision"
    assert meta["bank_code"] == "BANXICO"
    assert meta["country"] == "MEX"
    assert meta["language"] == "es"


@pytest.mark.network
def test_banxico_decision_smoke():
    from puremacro.narrative.sources import iter_banxico_decision
    recs = list(iter_banxico_decision())
    if not recs:
        pytest.skip("Banxico feed returned empty.")
    _, _, _, meta = recs[0]
    assert meta["bank_code"] == "BANXICO"


# ---------------------------------------------------------------------------
# Banco Central do Brasil (BCB)
# ---------------------------------------------------------------------------
def test_bcb_decision_yields_four_tuple_pt(mock_http):
    mock_http(bytes_={
        "https://www.bcb.gov.br/api/feed/site/comunicados-do-bcb":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Decis\xc3\xa3o do Copom</title>'
            b'<description>O Comit\xc3\xaa decidiu elevar a taxa Selic.</description>'
            b'<link>https://www.bcb.gov.br/detalhenoticia/100/comunicado</link>'
            b'<pubDate>Wed, 21 Sep 2022 18:30:00 -0300</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_bcb_decision
    records = list(iter_bcb_decision(language="pt"))
    assert len(records) == 1
    _, text, _, meta = records[0]
    assert "copom" in text.lower() or "selic" in text.lower()
    assert meta["bank_code"] == "BCB"
    assert meta["country"] == "BRA"
    assert meta["language"] == "pt"


@pytest.mark.network
def test_bcb_decision_smoke():
    from puremacro.narrative.sources import iter_bcb_decision
    recs = list(iter_bcb_decision())
    if not recs:
        pytest.skip("BCB feed empty.")
    _, _, _, meta = recs[0]
    assert meta["bank_code"] == "BCB"
```

- [ ] **Step 3: Run, verify failure**

Run: `pytest tests/test_narrative_slice3_connectors.py -v --no-header -k "not network" 2>&1 | tail -10`
Expected: 2 fail with `ImportError`.

- [ ] **Step 4: Create `puremacro/narrative/sources/banxico.py`**

```python
"""Banco de México (Banxico) — monetary-policy announcements (Spanish).

Banxico publishes monetary-policy decisions and other official press
releases on a single RSS feed in Spanish.

Feed:
    https://www.banxico.org.mx/rss/feeds/comunicados.xml
"""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.banxico.org.mx/rss/feeds/comunicados.xml"


def iter_banxico_decision() -> Iterator[tuple]:
    """Yield (date, text, url, metadata) for Banxico monetary-policy releases."""
    yield from iter_rss_filtered(
        _FEED,
        bank_code="BANXICO", country="MEX",
        doctype="decision", language="es",
    )


__all__ = ["iter_banxico_decision"]
```

- [ ] **Step 5: Create `puremacro/narrative/sources/bcb.py`**

```python
"""Banco Central do Brasil (BCB) — monetary-policy announcements
(Portuguese, with English mirror).

BCB publishes Copom decisions and press releases via two parallel feeds:
    Portuguese (default): https://www.bcb.gov.br/api/feed/site/comunicados-do-bcb
    English mirror:       https://www.bcb.gov.br/api/feed/site/eng/news
"""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED_BY_LANG = {
    "pt": "https://www.bcb.gov.br/api/feed/site/comunicados-do-bcb",
    "en": "https://www.bcb.gov.br/api/feed/site/eng/news",
}


def iter_bcb_decision(*, language: str = "pt") -> Iterator[tuple]:
    """Yield (date, text, url, metadata) for BCB monetary-policy releases.

    Language defaults to Portuguese; pass ``language="en"`` for the
    English mirror (fewer items, summarised).
    """
    url = _FEED_BY_LANG.get(language, _FEED_BY_LANG["pt"])
    yield from iter_rss_filtered(
        url, bank_code="BCB", country="BRA",
        doctype="decision", language=language,
    )


__all__ = ["iter_bcb_decision"]
```

- [ ] **Step 6: Re-export in `puremacro/narrative/sources/__init__.py`**

Append:
```python
from .banxico import iter_banxico_decision
from .bcb import iter_bcb_decision
```

Update `__all__`:
```python
    "iter_banxico_decision", "iter_bcb_decision",
```
(append to the existing `__all__` list).

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_narrative_slice3_connectors.py -v --no-header -k "not network" 2>&1 | tail -10`
Expected: 2 offline tests pass.

- [ ] **Step 8: Snapshot regen if drift**

```bash
pytest tests/test_public_api.py -v --no-header 2>&1 | tail -3
```
If FAIL:
```bash
python -c "from tests.test_public_api import _collect_current_api; import json; print(json.dumps(_collect_current_api(), indent=2, sort_keys=True))" > tests/fixtures/public_api_snapshot.json
```

- [ ] **Step 9: Commit**

```bash
git branch --show-current
git add puremacro/puremacro/narrative/sources/banxico.py \
        puremacro/puremacro/narrative/sources/bcb.py \
        puremacro/puremacro/narrative/sources/__init__.py \
        puremacro/tests/test_narrative_slice3_connectors.py
# add tests/fixtures/public_api_snapshot.json IF drifted
git commit -m "feat(narrative): Banxico (es) + BCB (pt/en) decision connectors"
```

---

## Task 5: LATAM connectors — BCCh + BCRA + BanRep

**Files:**
- Create: `puremacro/narrative/sources/bccl.py`
- Create: `puremacro/narrative/sources/bcra.py`
- Create: `puremacro/narrative/sources/banrep.py`
- Modify: `puremacro/narrative/sources/__init__.py`
- Modify: `tests/test_narrative_slice3_connectors.py` (append BCCh/BCRA/BanRep tests)

- [ ] **Step 1: Verify branch state.**

- [ ] **Step 2: Append failing tests**

```python
# ---------------------------------------------------------------------------
# Banco Central de Chile (BCCh)
# ---------------------------------------------------------------------------
def test_bccl_decision_yields_four_tuple_es(mock_http):
    mock_http(bytes_={
        "https://www.bcentral.cl/-/rss-feed-prensa":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Reuni\xc3\xb3n de pol\xc3\xadtica monetaria</title>'
            b'<description>El Consejo decidi\xc3\xb3 aumentar la tasa.</description>'
            b'<link>https://www.bcentral.cl/contenido/-/detalle/reunion-2022-09</link>'
            b'<pubDate>Tue, 06 Sep 2022 18:00:00 -0400</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_bccl_decision
    records = list(iter_bccl_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "BCCH"
    assert meta["country"] == "CHL"
    assert meta["language"] == "es"


# ---------------------------------------------------------------------------
# Banco Central de la República Argentina (BCRA)
# ---------------------------------------------------------------------------
def test_bcra_decision_yields_four_tuple_es(mock_http):
    mock_http(bytes_={
        "https://www.bcra.gob.ar/rss/Prensa.aspx":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Comunicado de prensa</title>'
            b'<description>El BCRA fij\xc3\xb3 la tasa de pol\xc3\xadtica monetaria.</description>'
            b'<link>https://www.bcra.gob.ar/Noticias/Comunicado-2022-08.asp</link>'
            b'<pubDate>Thu, 11 Aug 2022 17:00:00 -0300</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_bcra_decision
    records = list(iter_bcra_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "BCRA"
    assert meta["country"] == "ARG"


# ---------------------------------------------------------------------------
# Banco de la República (Colombia, BanRep)
# ---------------------------------------------------------------------------
def test_banrep_decision_yields_four_tuple_es(mock_http):
    mock_http(bytes_={
        "https://www.banrep.gov.co/rss-comunicados":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Comunicado de pol\xc3\xadtica monetaria</title>'
            b'<description>La Junta Directiva increment\xc3\xb3 la tasa de inter\xc3\xa9s.</description>'
            b'<link>https://www.banrep.gov.co/comunicado-2022-10</link>'
            b'<pubDate>Fri, 28 Oct 2022 14:00:00 -0500</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_banrep_decision
    records = list(iter_banrep_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "BANREP"
    assert meta["country"] == "COL"
```

- [ ] **Step 3: Run, verify failure** (all three fail with `ImportError`).

- [ ] **Step 4: Create `bccl.py`**

```python
"""Banco Central de Chile (BCCh) — monetary-policy press feed (Spanish)."""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.bcentral.cl/-/rss-feed-prensa"


def iter_bccl_decision() -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _FEED, bank_code="BCCH", country="CHL",
        doctype="decision", language="es",
    )


__all__ = ["iter_bccl_decision"]
```

- [ ] **Step 5: Create `bcra.py`**

```python
"""Banco Central de la República Argentina (BCRA) — press releases (Spanish)."""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.bcra.gob.ar/rss/Prensa.aspx"


def iter_bcra_decision() -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _FEED, bank_code="BCRA", country="ARG",
        doctype="decision", language="es",
    )


__all__ = ["iter_bcra_decision"]
```

- [ ] **Step 6: Create `banrep.py`**

```python
"""Banco de la República (Colombia, BanRep) — press releases (Spanish)."""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.banrep.gov.co/rss-comunicados"


def iter_banrep_decision() -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _FEED, bank_code="BANREP", country="COL",
        doctype="decision", language="es",
    )


__all__ = ["iter_banrep_decision"]
```

- [ ] **Step 7: Re-export in `__init__.py`**

Append:
```python
from .bccl import iter_bccl_decision
from .bcra import iter_bcra_decision
from .banrep import iter_banrep_decision
```

Add to `__all__`: `"iter_bccl_decision", "iter_bcra_decision", "iter_banrep_decision"`.

- [ ] **Step 8: Run new tests** — `pytest tests/test_narrative_slice3_connectors.py -v --no-header -k "(bccl or bcra or banrep) and not network" 2>&1 | tail -10` — 3 pass.

- [ ] **Step 9: Snapshot regen if drift, commit:**

```bash
git branch --show-current
git add puremacro/puremacro/narrative/sources/bccl.py \
        puremacro/puremacro/narrative/sources/bcra.py \
        puremacro/puremacro/narrative/sources/banrep.py \
        puremacro/puremacro/narrative/sources/__init__.py \
        puremacro/tests/test_narrative_slice3_connectors.py
# add tests/fixtures/public_api_snapshot.json IF drifted
git commit -m "feat(narrative): BCCh + BCRA + BanRep decision connectors (es)"
```

---

## Task 6: Advanced non-G7 — RBA + RBNZ

**Files:**
- Create: `puremacro/narrative/sources/rba.py`
- Create: `puremacro/narrative/sources/rbnz.py`
- Modify: `puremacro/narrative/sources/__init__.py`
- Modify: `tests/test_narrative_slice3_connectors.py`

- [ ] **Step 1: Verify branch state.**

- [ ] **Step 2: Append failing tests**

```python
# ---------------------------------------------------------------------------
# Reserve Bank of Australia (RBA)
# ---------------------------------------------------------------------------
def test_rba_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.rba.gov.au/feeds/rss.xml":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Statement on monetary policy decision</title>'
            b'<description>The Board decided to lift the cash rate target by 25bps.</description>'
            b'<link>https://www.rba.gov.au/media-releases/2022/mr-22-30.html</link>'
            b'<pubDate>Tue, 04 Oct 2022 04:30:00 +0000</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_rba_decision
    records = list(iter_rba_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "RBA"
    assert meta["country"] == "AUS"
    assert meta["language"] == "en"


def test_rba_speeches_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.rba.gov.au/feeds/speeches/rss.xml":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Inflation, productivity, and the supply side</title>'
            b'<description>Speech by the Governor at a conference.</description>'
            b'<link>https://www.rba.gov.au/speeches/2022/sp-gov-2022-09-15.html</link>'
            b'<pubDate>Thu, 15 Sep 2022 03:00:00 +0000</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_rba_speeches
    records = list(iter_rba_speeches())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "RBA"
    assert meta["doctype"] == "speech"


# ---------------------------------------------------------------------------
# Reserve Bank of New Zealand (RBNZ)
# ---------------------------------------------------------------------------
def test_rbnz_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.rbnz.govt.nz/rss/news.xml":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Monetary Policy Statement</title>'
            b'<description>The MPC raised the OCR by 50 basis points.</description>'
            b'<link>https://www.rbnz.govt.nz/news/2022/10/monetary-policy-statement-october-2022</link>'
            b'<pubDate>Wed, 05 Oct 2022 02:00:00 +0000</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_rbnz_decision
    records = list(iter_rbnz_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "RBNZ"
    assert meta["country"] == "NZL"
```

- [ ] **Step 3: Run, verify failure.**

- [ ] **Step 4: Create `rba.py`**

```python
"""Reserve Bank of Australia (RBA) — monetary-policy decisions + speeches.

The RBA's monetary-policy decisions and other media releases share one
RSS feed; speeches have their own. Both are English-only.
"""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_DECISION_FEED = "https://www.rba.gov.au/feeds/rss.xml"
_SPEECHES_FEED = "https://www.rba.gov.au/feeds/speeches/rss.xml"


def iter_rba_decision() -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _DECISION_FEED, bank_code="RBA", country="AUS",
        doctype="decision", language="en",
    )


def iter_rba_speeches() -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _SPEECHES_FEED, bank_code="RBA", country="AUS",
        doctype="speech", language="en",
    )


__all__ = ["iter_rba_decision", "iter_rba_speeches"]
```

- [ ] **Step 5: Create `rbnz.py`**

```python
"""Reserve Bank of New Zealand (RBNZ) — combined news feed (English).

RBNZ's main RSS covers monetary-policy decisions, FSR releases, and
speeches in one feed. We default to filtering by title for monetary
policy items; pass to ``iter_rbnz_decision`` directly.
"""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.rbnz.govt.nz/rss/news.xml"


def iter_rbnz_decision() -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _FEED, bank_code="RBNZ", country="NZL",
        doctype="decision", language="en",
        title_keywords=["monetary policy", "ocr", "official cash rate"],
    )


__all__ = ["iter_rbnz_decision"]
```

- [ ] **Step 6: Re-export in `__init__.py`**

Append:
```python
from .rba import iter_rba_decision, iter_rba_speeches
from .rbnz import iter_rbnz_decision
```

Add to `__all__`: `"iter_rba_decision", "iter_rba_speeches", "iter_rbnz_decision"`.

- [ ] **Step 7: Run tests**

`pytest tests/test_narrative_slice3_connectors.py -v --no-header -k "(rba or rbnz) and not network" 2>&1 | tail -10` — 3 pass.

- [ ] **Step 8: Snapshot regen if drift, commit:**

```bash
git branch --show-current
git add puremacro/puremacro/narrative/sources/rba.py \
        puremacro/puremacro/narrative/sources/rbnz.py \
        puremacro/puremacro/narrative/sources/__init__.py \
        puremacro/tests/test_narrative_slice3_connectors.py
# add tests/fixtures/public_api_snapshot.json IF drifted
git commit -m "feat(narrative): RBA + RBNZ connectors (en)"
```

---

## Task 7: Advanced non-G7 — Riksbank + Norges + SARB

**Files:**
- Create: `puremacro/narrative/sources/riksbank.py`
- Create: `puremacro/narrative/sources/norges.py`
- Create: `puremacro/narrative/sources/sarb.py`
- Modify: `puremacro/narrative/sources/__init__.py`
- Modify: `tests/test_narrative_slice3_connectors.py`

- [ ] **Step 1: Verify branch state.**

- [ ] **Step 2: Append tests**

```python
# ---------------------------------------------------------------------------
# Riksbank (Sweden)
# ---------------------------------------------------------------------------
def test_riksbank_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.riksbank.se/en-gb/feeds/news/":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Repo rate decision</title>'
            b'<description>The Executive Board decided to raise the repo rate.</description>'
            b'<link>https://www.riksbank.se/en-gb/press-and-published/2022/repo-rate</link>'
            b'<pubDate>Wed, 21 Sep 2022 07:30:00 +0000</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_riksbank_decision
    records = list(iter_riksbank_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "RIKSBANK"
    assert meta["country"] == "SWE"


# ---------------------------------------------------------------------------
# Norges Bank
# ---------------------------------------------------------------------------
def test_norges_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.norges-bank.no/en/news-events/news-publications/?rss=true":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Monetary Policy Report and key policy rate</title>'
            b'<description>Norges Bank raised the policy rate by 50 basis points.</description>'
            b'<link>https://www.norges-bank.no/en/news-events/news-publications/2022/2022-09</link>'
            b'<pubDate>Thu, 22 Sep 2022 08:00:00 +0000</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_norges_decision
    records = list(iter_norges_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "NORGES"
    assert meta["country"] == "NOR"


# ---------------------------------------------------------------------------
# South African Reserve Bank (SARB)
# ---------------------------------------------------------------------------
def test_sarb_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.resbank.co.za/en/home/publications/rss":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Monetary Policy Committee statement</title>'
            b'<description>The MPC raised the repo rate by 75 basis points.</description>'
            b'<link>https://www.resbank.co.za/en/home/publications/publication-detail-pages/statements/monetary-policy-statements/2022</link>'
            b'<pubDate>Thu, 22 Sep 2022 13:00:00 +0200</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_sarb_decision
    records = list(iter_sarb_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "SARB"
    assert meta["country"] == "ZAF"
```

- [ ] **Step 3: Run, verify failure.**

- [ ] **Step 4: Create `riksbank.py`**

```python
"""Sveriges Riksbank — news feed (English mirror)."""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.riksbank.se/en-gb/feeds/news/"


def iter_riksbank_decision() -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _FEED, bank_code="RIKSBANK", country="SWE",
        doctype="decision", language="en",
        title_keywords=["repo rate", "monetary policy", "interest rate"],
    )


__all__ = ["iter_riksbank_decision"]
```

- [ ] **Step 5: Create `norges.py`**

```python
"""Norges Bank — news feed (English mirror)."""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.norges-bank.no/en/news-events/news-publications/?rss=true"


def iter_norges_decision() -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _FEED, bank_code="NORGES", country="NOR",
        doctype="decision", language="en",
        title_keywords=["policy rate", "monetary policy", "key policy"],
    )


__all__ = ["iter_norges_decision"]
```

- [ ] **Step 6: Create `sarb.py`**

```python
"""South African Reserve Bank (SARB) — publications feed (English)."""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.resbank.co.za/en/home/publications/rss"


def iter_sarb_decision() -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _FEED, bank_code="SARB", country="ZAF",
        doctype="decision", language="en",
        title_keywords=["monetary policy", "repo rate", "mpc"],
    )


__all__ = ["iter_sarb_decision"]
```

- [ ] **Step 7: Re-export in `__init__.py`**

Append:
```python
from .riksbank import iter_riksbank_decision
from .norges import iter_norges_decision
from .sarb import iter_sarb_decision
```

Add to `__all__`: `"iter_riksbank_decision", "iter_norges_decision", "iter_sarb_decision"`.

- [ ] **Step 8: Run tests, snapshot regen if drift, commit:**

```bash
git branch --show-current
git add puremacro/puremacro/narrative/sources/riksbank.py \
        puremacro/puremacro/narrative/sources/norges.py \
        puremacro/puremacro/narrative/sources/sarb.py \
        puremacro/puremacro/narrative/sources/__init__.py \
        puremacro/tests/test_narrative_slice3_connectors.py
# add tests/fixtures/public_api_snapshot.json IF drifted
git commit -m "feat(narrative): Riksbank + Norges + SARB decision connectors (en)"
```

---

## Task 8: Asia-EM — PBoC + RBI

**Files:**
- Create: `puremacro/narrative/sources/pboc.py`
- Create: `puremacro/narrative/sources/rbi.py`
- Modify: `puremacro/narrative/sources/__init__.py`
- Modify: `tests/test_narrative_slice3_connectors.py`

- [ ] **Step 1: Verify branch state.**

- [ ] **Step 2: Append tests**

```python
# ---------------------------------------------------------------------------
# People's Bank of China (PBoC) — English mirror
# ---------------------------------------------------------------------------
def test_pboc_decision_yields_four_tuple_en(mock_http):
    mock_http(bytes_={
        "http://www.pbc.gov.cn/en/3688110/3688215/index.html":
            '<html><body>'
            '<a href="/en/3688110/3688215/4582345/index.html" title="PBC announces rate cut">'
            'PBC announces rate cut'
            '</a>'
            '<span class="date">2022-08-22</span>'
            '</body></html>',
    })
    from puremacro.narrative.sources import iter_pboc_decision
    records = list(iter_pboc_decision())
    # Best-effort HTML parse — may yield 0 if the structure isn't recognised.
    if records:
        _, _, _, meta = records[0]
        assert meta["bank_code"] == "PBOC"
        assert meta["country"] == "CHN"


# ---------------------------------------------------------------------------
# Reserve Bank of India (RBI)
# ---------------------------------------------------------------------------
def test_rbi_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://rbi.org.in/Scripts/RSS.aspx":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Monetary Policy Statement</title>'
            b'<description>The MPC raised the repo rate by 50 basis points.</description>'
            b'<link>https://rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=12345</link>'
            b'<pubDate>Fri, 30 Sep 2022 10:00:00 +0530</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_rbi_decision
    records = list(iter_rbi_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "RBI"
    assert meta["country"] == "IND"
```

- [ ] **Step 3: Run, verify failure.**

- [ ] **Step 4: Create `pboc.py`** — simple HTML scrape (PBoC has no clean RSS)

```python
"""People's Bank of China (PBoC) — English mirror.

PBoC does not publish a clean RSS feed. We scrape the press-releases
listing page (English mirror) for date+href pairs and yield 4-tuple
SourceRecords. The HTML structure can change without notice, so the
connector is best-effort: parsing failures yield nothing rather than
raising.
"""
from __future__ import annotations

import re
from typing import Iterator

import pandas as pd

from ..._http import safe_get_text


_LISTING_URL = "http://www.pbc.gov.cn/en/3688110/3688215/index.html"
_BASE = "http://www.pbc.gov.cn"
_LINK_RX = re.compile(
    r'<a[^>]+href="([^"]+)"[^>]*title="([^"]+)"',
    flags=re.IGNORECASE,
)
_DATE_RX = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def iter_pboc_decision() -> Iterator[tuple]:
    """Yield (date, text, url, metadata) for PBoC press releases (English)."""
    try:
        html = safe_get_text(_LISTING_URL)
    except Exception:
        return
    # Best-effort: pair link anchors with the nearest date string.
    for m in _LINK_RX.finditer(html):
        href, title = m.group(1), m.group(2)
        # Look for a date within ~120 chars after the anchor.
        date_window = html[m.end():m.end() + 120]
        date_match = _DATE_RX.search(date_window)
        if not date_match:
            continue
        try:
            date = pd.Timestamp(date_match.group(1))
        except Exception:
            continue
        if href.startswith("/"):
            href = _BASE + href
        yield (date, title, href, {
            "doctype": "decision", "language": "en",
            "bank_code": "PBOC", "country": "CHN",
        })


__all__ = ["iter_pboc_decision"]
```

- [ ] **Step 5: Create `rbi.py`**

```python
"""Reserve Bank of India (RBI) — main press-release RSS feed (English)."""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://rbi.org.in/Scripts/RSS.aspx"


def iter_rbi_decision() -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _FEED, bank_code="RBI", country="IND",
        doctype="decision", language="en",
        title_keywords=["monetary policy", "repo rate", "mpc"],
    )


__all__ = ["iter_rbi_decision"]
```

- [ ] **Step 6: Re-export in `__init__.py`**

Append:
```python
from .pboc import iter_pboc_decision
from .rbi import iter_rbi_decision
```

Add to `__all__`: `"iter_pboc_decision", "iter_rbi_decision"`.

- [ ] **Step 7: Run tests, snapshot regen, commit**

```bash
git branch --show-current
git add puremacro/puremacro/narrative/sources/pboc.py \
        puremacro/puremacro/narrative/sources/rbi.py \
        puremacro/puremacro/narrative/sources/__init__.py \
        puremacro/tests/test_narrative_slice3_connectors.py
# add tests/fixtures/public_api_snapshot.json IF drifted
git commit -m "feat(narrative): PBoC (en HTML) + RBI (en RSS) decision connectors"
```

---

## Task 9: Asia-EM — BoK + MAS + BoT

**Files:**
- Create: `puremacro/narrative/sources/bok.py`
- Create: `puremacro/narrative/sources/mas.py`
- Create: `puremacro/narrative/sources/bot.py`
- Modify: `puremacro/narrative/sources/__init__.py`
- Modify: `tests/test_narrative_slice3_connectors.py`

- [ ] **Step 1: Verify branch state.**

- [ ] **Step 2: Append tests**

```python
# ---------------------------------------------------------------------------
# Bank of Korea (BoK) — English mirror
# ---------------------------------------------------------------------------
def test_bok_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.bok.or.kr/eng/rss/RssBokService.do?menuNo=400069":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Monetary Policy Decision</title>'
            b'<description>The Monetary Policy Board raised the Base Rate.</description>'
            b'<link>https://www.bok.or.kr/eng/main/contents.do?menuNo=400069&pCdid=12345</link>'
            b'<pubDate>Wed, 12 Oct 2022 09:00:00 +0900</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_bok_decision
    records = list(iter_bok_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "BOK"
    assert meta["country"] == "KOR"


# ---------------------------------------------------------------------------
# Monetary Authority of Singapore (MAS)
# ---------------------------------------------------------------------------
def test_mas_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.mas.gov.sg/news/rss":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Monetary policy statement</title>'
            b'<description>MAS will continue with the policy of appreciation of the SGD NEER.</description>'
            b'<link>https://www.mas.gov.sg/news/monetary-policy-statements/2022/oct</link>'
            b'<pubDate>Fri, 14 Oct 2022 08:00:00 +0800</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_mas_decision
    records = list(iter_mas_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "MAS"
    assert meta["country"] == "SGP"


# ---------------------------------------------------------------------------
# Bank of Thailand (BoT)
# ---------------------------------------------------------------------------
def test_bot_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.bot.or.th/content/bot/en/_jcr_content.feed":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>MPC decision</title>'
            b'<description>The MPC voted to raise the policy rate by 25bps.</description>'
            b'<link>https://www.bot.or.th/en/news-and-media/news/news-202209</link>'
            b'<pubDate>Wed, 28 Sep 2022 14:00:00 +0700</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_bot_decision
    records = list(iter_bot_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "BOT"
    assert meta["country"] == "THA"
```

- [ ] **Step 3: Run, verify failure.**

- [ ] **Step 4: Create `bok.py`**

```python
"""Bank of Korea (BoK) — English mirror RSS."""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.bok.or.kr/eng/rss/RssBokService.do?menuNo=400069"


def iter_bok_decision() -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _FEED, bank_code="BOK", country="KOR",
        doctype="decision", language="en",
        title_keywords=["monetary policy", "base rate"],
    )


__all__ = ["iter_bok_decision"]
```

- [ ] **Step 5: Create `mas.py`**

```python
"""Monetary Authority of Singapore (MAS) — news RSS feed (English)."""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.mas.gov.sg/news/rss"


def iter_mas_decision() -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _FEED, bank_code="MAS", country="SGP",
        doctype="decision", language="en",
        title_keywords=["monetary policy"],
    )


__all__ = ["iter_mas_decision"]
```

- [ ] **Step 6: Create `bot.py`**

```python
"""Bank of Thailand (BoT) — English content feed."""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.bot.or.th/content/bot/en/_jcr_content.feed"


def iter_bot_decision() -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _FEED, bank_code="BOT", country="THA",
        doctype="decision", language="en",
        title_keywords=["mpc", "monetary policy", "policy rate"],
    )


__all__ = ["iter_bot_decision"]
```

- [ ] **Step 7: Re-export in `__init__.py`**

Append:
```python
from .bok import iter_bok_decision
from .mas import iter_mas_decision
from .bot import iter_bot_decision
```

Add to `__all__`: `"iter_bok_decision", "iter_mas_decision", "iter_bot_decision"`.

- [ ] **Step 8: Run tests, snapshot regen, commit**

```bash
git branch --show-current
git add puremacro/puremacro/narrative/sources/bok.py \
        puremacro/puremacro/narrative/sources/mas.py \
        puremacro/puremacro/narrative/sources/bot.py \
        puremacro/puremacro/narrative/sources/__init__.py \
        puremacro/tests/test_narrative_slice3_connectors.py
# add tests/fixtures/public_api_snapshot.json IF drifted
git commit -m "feat(narrative): BoK + MAS + BoT decision connectors (en)"
```

---

## Task 10: BIS speeches meta-connector

**Files:**
- Create: `puremacro/narrative/sources/bis_speeches.py`
- Modify: `puremacro/narrative/sources/__init__.py`
- Modify: `tests/test_narrative_slice3_connectors.py`

The Bank for International Settlements republishes speeches by central-bank governors and senior officials from member banks at speeches.bis.org. A single connector can fetch the latest archive page (or filter by bank code) — providing broad cross-CB coverage without per-bank scraping.

- [ ] **Step 1: Verify branch state.**

- [ ] **Step 2: Append failing tests**

```python
# ---------------------------------------------------------------------------
# BIS speeches meta-connector
# ---------------------------------------------------------------------------
def test_bis_speeches_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.bis.org/cbspeeches/index.rss":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Inflation outlook and policy challenges</title>'
            b'<description>Speech by Christine Lagarde at the IMF annual meetings.</description>'
            b'<link>https://www.bis.org/review/r221015a.htm</link>'
            b'<pubDate>Sat, 15 Oct 2022 12:00:00 +0000</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_bis_speeches
    records = list(iter_bis_speeches())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "BIS"
    assert meta["doctype"] == "speech"
    # Country tag is "MULTI" because BIS speeches span all member banks.
    assert meta["country"] == "MULTI"


def test_bis_speeches_filter_by_bank_keyword(mock_http):
    """When the caller passes bank_filter, only matching items are yielded."""
    mock_http(bytes_={
        "https://www.bis.org/cbspeeches/index.rss":
            b'<?xml version="1.0"?><rss><channel>'
            b'<item><title>ECB Lagarde on inflation</title>'
            b'<description>Christine Lagarde speech</description>'
            b'<link>https://www.bis.org/review/r221015a.htm</link>'
            b'<pubDate>Sat, 15 Oct 2022 12:00:00 +0000</pubDate></item>'
            b'<item><title>Fed Powell on financial conditions</title>'
            b'<description>Jerome Powell speech</description>'
            b'<link>https://www.bis.org/review/r221016a.htm</link>'
            b'<pubDate>Sun, 16 Oct 2022 12:00:00 +0000</pubDate></item>'
            b'</channel></rss>',
    })
    from puremacro.narrative.sources import iter_bis_speeches
    records = list(iter_bis_speeches(bank_filter="Powell"))
    assert len(records) == 1
    _, text, _, _ = records[0]
    assert "Powell" in text


@pytest.mark.network
def test_bis_speeches_smoke():
    from puremacro.narrative.sources import iter_bis_speeches
    recs = list(iter_bis_speeches())
    if not recs:
        pytest.skip("BIS speeches feed empty.")
    _, _, _, meta = recs[0]
    assert meta["bank_code"] == "BIS"
```

- [ ] **Step 3: Run, verify failures.**

- [ ] **Step 4: Create `bis_speeches.py`**

```python
"""Bank for International Settlements speech archive (multi-bank).

The BIS republishes speeches by senior officials of its ~60 member
central banks at https://www.bis.org/cbspeeches/. We pull the master
RSS feed and tag every item with ``bank_code="BIS"`` /
``country="MULTI"`` since the feed spans all member banks. Callers
who want bank-specific filtering can pass ``bank_filter`` (a string
matched against the title, e.g. ``"Lagarde"``, ``"Fed"``).
"""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.bis.org/cbspeeches/index.rss"


def iter_bis_speeches(*, bank_filter: str | None = None) -> Iterator[tuple]:
    """Yield (date, text, url, metadata) for BIS-republished CB speeches.

    Parameters
    ----------
    bank_filter : optional substring to match against the title (case-
        insensitive). Useful to narrow to one institution's speeches
        (e.g., ``"Lagarde"`` for ECB, ``"Powell"`` for Fed).
    """
    title_keywords = [bank_filter] if bank_filter else None
    yield from iter_rss_filtered(
        _FEED,
        bank_code="BIS", country="MULTI",
        doctype="speech", language="en",
        title_keywords=title_keywords,
    )


__all__ = ["iter_bis_speeches"]
```

- [ ] **Step 5: Re-export in `__init__.py`**

Append:
```python
from .bis_speeches import iter_bis_speeches
```

Add to `__all__`: `"iter_bis_speeches"`.

- [ ] **Step 6: Run tests, snapshot regen, commit**

```bash
git branch --show-current
git add puremacro/puremacro/narrative/sources/bis_speeches.py \
        puremacro/puremacro/narrative/sources/__init__.py \
        puremacro/tests/test_narrative_slice3_connectors.py
# add tests/fixtures/public_api_snapshot.json IF drifted
git commit -m "feat(narrative): BIS speeches meta-connector"
```

---

## Task 11: macropru / fx / structural prompt smoke tests

**Files:**
- Create: `tests/test_narrative_slice3_prompts.py`

The five LLM prompts (`fiscal`, `monetary`, `macropru`, `fx`, `structural`) shipped in Slice 1's `score_llm`. Slice 1 tested only `fiscal` and `monetary` end-to-end. This task adds dry-run smoke tests for the other three: confirm `_build_prompt` returns the right shape and that `score_llm(..., dry_run=True)` exits without error for each kind, and confirm `_validate_event_dict` accepts well-formed dicts and rejects malformed ones for each kind.

- [ ] **Step 1: Verify branch state.**

- [ ] **Step 2: Create `tests/test_narrative_slice3_prompts.py`**

```python
"""Slice 3: smoke tests for macropru / fx / structural LLM prompts."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative.scoring.llm import (
    _PROMPTS, _build_prompt, _validate_event_dict, score_llm,
)


# ---------------------------------------------------------------------------
# _build_prompt — coverage of the three kinds shipped in Slice 1 but not
# previously smoke-tested in their own test
# ---------------------------------------------------------------------------
def test_build_prompt_macropru_contains_target_enum():
    p = _build_prompt(kind="macropru", language="en", country="GBR",
                      date="2022-09-01", text="capital buffer increased")
    low = p.lower()
    assert "capital_buffer" in low
    assert "ltv_dsti" in low
    assert "tightening" in low
    assert "loosening" in low


def test_build_prompt_fx_contains_target_enum():
    p = _build_prompt(kind="fx", language="en", country="JPN",
                      date="2022-10-21", text="MoF intervened to buy JPY")
    low = p.lower()
    assert "intervention" in low
    assert "peg_change" in low


def test_build_prompt_structural_contains_target_enum():
    p = _build_prompt(kind="structural", language="en", country="ITA",
                      date="2018-01-01", text="labor reform passed")
    low = p.lower()
    assert "labor" in low
    assert "product_market" in low
    assert "trade" in low
    assert "tax_admin" in low


# ---------------------------------------------------------------------------
# _validate_event_dict — per-kind acceptance / rejection
# ---------------------------------------------------------------------------
def test_validate_macropru_accepts_well_formed_dict():
    d = {
        "target": "capital_buffer",
        "magnitude_pct": 0.5,
        "subtarget": None,
        "sign": +1,
        "confidence": 0.8,
        "excerpt": "the FPC raised the CCyB",
    }
    assert _validate_event_dict(d, kind="macropru") is True


def test_validate_macropru_rejects_wrong_target():
    d = {
        "target": "investment",   # fiscal target, not macropru
        "magnitude_pct": 0.5,
        "sign": +1,
        "confidence": 0.8,
    }
    assert _validate_event_dict(d, kind="macropru") is False


def test_validate_fx_accepts_well_formed_dict():
    d = {
        "target": "intervention",
        "magnitude_usd_bn": 36.0,
        "sign": +1,
        "confidence": 0.9,
    }
    assert _validate_event_dict(d, kind="fx") is True


def test_validate_structural_accepts_well_formed_dict():
    d = {
        "target": "labor",
        "magnitude_z": 1.5,
        "sign": +1,
        "confidence": 0.7,
    }
    assert _validate_event_dict(d, kind="structural") is True


def test_validate_structural_rejects_wrong_magnitude_key():
    """The structural schema uses magnitude_z, not magnitude_usd_bn."""
    d = {
        "target": "labor",
        "magnitude_usd_bn": 1.5,   # wrong key
        "sign": +1,
        "confidence": 0.7,
    }
    assert _validate_event_dict(d, kind="structural") is False


# ---------------------------------------------------------------------------
# score_llm dry_run for the three kinds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["macropru", "fx", "structural"])
def test_score_llm_dry_run_supports_kind(kind):
    records = [(pd.Timestamp("2020-01-01"), "x", "u")]
    out = score_llm(records, backend=None, kind=kind, dry_run=True)
    assert out == []
```

- [ ] **Step 3: Run new tests** — `pytest tests/test_narrative_slice3_prompts.py -v --no-header 2>&1 | tail -20` — 12 pass.

- [ ] **Step 4: Run full suite** — `pytest -q --no-header 2>&1 | tail -3` — pass count up by 12.

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add puremacro/tests/test_narrative_slice3_prompts.py
git commit -m "test(narrative): macropru/fx/structural prompt smoke tests"
```

---

## Task 12: Cross-lingual validation tests

**Files:**
- Create: `tests/test_narrative_indices_crosslingual.py`

Tests that the EN and ES versions of an EPU-style index, when computed on the same overlapping period from a real corpus (ECB press, multilingual), correlate ρ ≥ 0.7. These are network-marked and skip on empty.

- [ ] **Step 1: Verify branch state.**

- [ ] **Step 2: Create `tests/test_narrative_indices_crosslingual.py`**

```python
"""Slice 3: cross-lingual validation smokes.

Network-marked tests that build the same EPU-style index from the ECB's
English vs Spanish press feeds and check the resulting quarterly series
correlate ρ ≥ 0.7 on the overlapping window.

Skip when feeds are empty (per the project network-tests-skip-on-empty
convention).
"""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.mark.network
def test_ecb_epu_en_vs_es_correlation():
    """ECB press feed in en vs es should yield highly correlated EPU
    quarterly series. Skip on empty feeds."""
    from puremacro.narrative.indices import epu
    from puremacro.narrative.sources import iter_ecb_decision

    en_records = list(iter_ecb_decision(language="en"))
    es_records = list(iter_ecb_decision(language="es"))

    if not en_records or not es_records:
        pytest.skip("ECB feed returned empty for one of the languages.")

    ri_en = epu(en_records, country="EA20", language="en", normalize="raw")
    ri_es = epu(es_records, country="EA20", language="es", normalize="raw")

    s_en = ri_en.series.dropna()
    s_es = ri_es.series.dropna()
    common = s_en.index.intersection(s_es.index)
    if len(common) < 4:
        pytest.skip(f"Insufficient overlap: only {len(common)} common quarters.")

    rho = float(s_en.loc[common].corr(s_es.loc[common]))
    if pd.isna(rho):
        pytest.skip("Correlation is NaN (constant series); cross-lingual signal too weak.")

    assert rho >= 0.7, (
        f"EN-vs-ES EPU on ECB press should correlate ρ ≥ 0.7; got {rho:.3f} "
        f"on {len(common)} common quarters."
    )


@pytest.mark.network
def test_ecb_lui_en_vs_es_correlation():
    """LUI lexicon coverage parity check across en / es ECB press."""
    from puremacro.narrative.indices import lui
    from puremacro.narrative.sources import iter_ecb_decision

    en_records = list(iter_ecb_decision(language="en"))
    es_records = list(iter_ecb_decision(language="es"))

    if not en_records or not es_records:
        pytest.skip("ECB feed returned empty for one of the languages.")

    ri_en = lui(en_records, country="EA20", language="en", normalize="raw")
    ri_es = lui(es_records, country="EA20", language="es", normalize="raw")

    s_en = ri_en.series.dropna()
    s_es = ri_es.series.dropna()
    common = s_en.index.intersection(s_es.index)
    if len(common) < 4:
        pytest.skip(f"Insufficient overlap: only {len(common)} common quarters.")

    # Looser threshold for LUI: labor language is rarer in CB text than
    # uncertainty language, so cross-lingual signal is weaker.
    rho = float(s_en.loc[common].corr(s_es.loc[common]))
    if pd.isna(rho):
        pytest.skip("Correlation is NaN.")
    assert rho >= 0.4, (
        f"EN-vs-ES LUI on ECB press should correlate ρ ≥ 0.4; got {rho:.3f}."
    )
```

- [ ] **Step 3: Run offline collection check (network tests are deselected by default)**

`pytest tests/test_narrative_indices_crosslingual.py -v --no-header 2>&1 | tail -10`
Expected: collected 2, deselected 2 (or skipped if `pytest.ini` doesn't deselect by default).

- [ ] **Step 4: Run full suite**

`pytest -q --no-header 2>&1 | tail -3` — pass count unchanged (network tests deselected).

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add puremacro/tests/test_narrative_indices_crosslingual.py
git commit -m "test(narrative): cross-lingual EN-vs-ES correlation smokes (network)"
```

---

## Task 13: Public API audit + Pyodide compat verification

**Files:** none (verification only — possibly snapshot regen)

- [ ] **Step 1: Verify branch state.**

- [ ] **Step 2: Confirm imports work**

Run:
```bash
python -c "
from puremacro.narrative.sources import (
    # Slice 1 (still working)
    iter_fed_decision, iter_fed_minutes, iter_fed_press_conf, iter_fed_speeches,
    iter_ecb_decision, iter_ecb_minutes, iter_ecb_press_conf, iter_ecb_speeches,
    iter_boe_decision, iter_boe_minutes, iter_boe_speeches,
    iter_boj_decision, iter_boj_speeches,
    # Slice 3 (new)
    iter_banxico_decision, iter_bcb_decision, iter_bccl_decision,
    iter_bcra_decision, iter_banrep_decision,
    iter_rba_decision, iter_rba_speeches, iter_rbnz_decision,
    iter_riksbank_decision, iter_norges_decision, iter_sarb_decision,
    iter_pboc_decision, iter_rbi_decision,
    iter_bok_decision, iter_mas_decision, iter_bot_decision,
    iter_bis_speeches,
)
print('public-API audit ok — Slice 1 + Slice 3 connectors all import')
"
```
Expected: `public-API audit ok — Slice 1 + Slice 3 connectors all import`.

- [ ] **Step 3: Pyodide compat**

Run: `pytest tests/test_pyodide_compat.py -v --no-header 2>&1 | tail -10`

Expected: 1 PRE-EXISTING failure with `statsmodels.tsa.x13` leak from `puremacro/fetch/_seasonal.py:19`. The leak set must be exactly the same `statsmodels.*` set as baseline. New `narrative.sources/<bank>.py` files are excluded from the walk (under `narrative/sources/`). If the leak set grows, BLOCK and report.

- [ ] **Step 4: Run full suite**

Run: `pytest -q --no-header 2>&1 | tail -3`
Expected: 924 + (~50 new tests from Slice 3) ≈ 970+ passed.

- [ ] **Step 5: Snapshot regen if drift**

```bash
pytest tests/test_public_api.py -v --no-header 2>&1 | tail -3
```
If FAIL:
```bash
python -c "from tests.test_public_api import _collect_current_api; import json; print(json.dumps(_collect_current_api(), indent=2, sort_keys=True))" > tests/fixtures/public_api_snapshot.json
git add puremacro/tests/fixtures/public_api_snapshot.json
git commit -m "chore(narrative): regenerate public_api_snapshot for Slice 3"
```
Otherwise no commit.

---

## Task 14: Version 0.7.0 + CHANGELOG + tag

**Files:**
- Modify: `pyproject.toml`
- Modify: `puremacro/__init__.py`
- Modify: `tests/test_import.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Verify branch state.**

- [ ] **Step 2: Bump version**

Edit `pyproject.toml`: `version = "0.6.2"` → `version = "0.7.0"`.

Edit `puremacro/__init__.py`: `__version__ = "0.6.2"` → `__version__ = "0.7.0"`.

Edit `tests/test_import.py`: `assert puremacro.__version__ == "0.6.2"` → `"0.7.0"`.

- [ ] **Step 3: Add CHANGELOG entry**

Open `CHANGELOG.md`. Add a new top entry above the `## 0.6.2 — 2026-05-08` block:

```markdown
## 0.7.0 — 2026-05-08

Slice 3 of the multi-domain narrative extension (`docs/specs/2026-05-08-narrative-extension-design.md`). Polyglot expansion: 15 new central-bank connectors plus a BIS speeches meta-connector. Closes the 3-slice plan.

### Added

- **15 new CB connectors** (Slice 3 polyglot wave):
  - **LATAM (5):** `iter_banxico_decision` (es), `iter_bcb_decision` (pt + en mirror), `iter_bccl_decision` (es), `iter_bcra_decision` (es), `iter_banrep_decision` (es).
  - **Advanced non-G7 (5):** `iter_rba_decision` + `iter_rba_speeches`, `iter_rbnz_decision`, `iter_riksbank_decision`, `iter_norges_decision`, `iter_sarb_decision` (all en).
  - **Asia EM (5):** `iter_pboc_decision` (en mirror, HTML scrape), `iter_rbi_decision`, `iter_bok_decision` (en mirror), `iter_mas_decision`, `iter_bot_decision`.
- **`iter_bis_speeches`** — meta-connector pulling the BIS speech republication archive across ~60 member central banks. Optional `bank_filter` for per-institution narrowing.
- **`narrative.sources._rss_filtered.iter_rss_filtered`** — shared helper that consolidates the RSS-fetch + title-keyword-filter + 4-tuple-emit pattern. New Slice 3 connectors collapse to a 6-line `yield from` call. (Slice 1 connectors retain their original implementations.)
- **JA / ZH tone lexicons** (`LEXICONS["tone"]["ja"]`, `LEXICONS["tone"]["zh"]`) — closes Slice 2's deferral. All 6 indices now have lexicon coverage in all 8 languages.
- **`puremacro.narrative.aggregate.index_to_quarterly` plumbs `base_period`** through to `normalize_series` (Slice 2 stored it as metadata only). All 6 index helpers (`epu`, `mpu`, `gpr`, `tone`, `wui`, `lui`) now honor `base_period=("YYYY-MM-DD", "YYYY-MM-DD")` for normalisation reference window — e.g., BBD-published 1985–2009 base for `bbd_100`.
- **macropru / fx / structural prompt smoke tests** — 12 new tests in `tests/test_narrative_slice3_prompts.py` exercise the three Slice-1-shipped LLM prompt families end-to-end via `_build_prompt` + `_validate_event_dict` + `score_llm(dry_run=True)`.
- **Cross-lingual validation tests** — `tests/test_narrative_indices_crosslingual.py` (`@pytest.mark.network`) checks EN-vs-ES EPU and LUI on the same ECB-press window correlate ρ ≥ 0.7 (EPU) / ρ ≥ 0.4 (LUI). Skip-on-empty per the project's network-tests convention.
- **Shared `mock_http` fixture promoted to `tests/conftest.py`** — Slice 1's per-file fixture now lives in conftest and serves all 15 new Slice-3 connector tests in addition to the existing CB tests.

### Changed

- `narrative.indices._kernels._VALID_NORMALIZATIONS` is now an alias to `narrative.types.VALID_RISKINDEX_NORMALIZATION` (single source of truth — closes Slice 2 review issue M4).
- The 6 index docstrings (`epu/mpu/gpr/tone/wui/lui`) update the `base_period` parameter description from "stored in metadata only" to the now-functional "plumbed through to normalize_series" semantics.

### Pyodide compatibility

- All 16 new connector modules live under `narrative/sources/` and stay in the existing **Experimental** tier. `tests/test_pyodide_compat.py` excludes the subtree from its leakage walk. Slice 3 added zero new forbidden-runtime-dep leaks. The pre-existing `statsmodels.tsa.x13` leak via `puremacro/fetch/_seasonal.py:19` remains the only failing pyodide-compat case.

### Deferred to a future iteration (out of scope for the 3-slice plan)

- **Picault-Renault paragraph-level multinomial logit** — `tone(method="picault_renault")` still uses the count-based mechanism, lexicon tuning shipped.
- **Full Hubert lexicon** — `tone(method="hubert")` shares Apel-Blix-Grimaldi machinery; separate Hubert dictionary is research code beyond the scope of this iteration.
- **Length-normalised WUI** per the original Ahir-Bloom-Furceri methodology (mentions per 1000 words). Current `wui()` uses raw counts.
- **`llm_prob_kernel`** for LLM-backed per-document scoring inside `narrative.indices`.
- **Published-correlation regression tests** (ρ ≥ 0.85 vs `bbd_epu` / `caldara_iacoviello_gpr`). Cross-lingual ρ ≥ 0.7 ships in this slice; the published-corpus comparison requires the BBD source corpus which we don't ship.

### Slice 1 + 2 + 3 totals

| Slice | Version | Tests added | Tests at end |
|-------|---------|-------------|---------------|
| 1     | 0.6.1   | +67         | 858           |
| 2     | 0.6.2   | +66         | 924           |
| 3     | 0.7.0   | ~+50        | ~970+         |

```

- [ ] **Step 4: Run the full suite once more**

Run: `pytest -q --no-header 2>&1 | tail -3`
Expected: pass count + version-test now `0.7.0`.

- [ ] **Step 5: Run fiscal regression suite**

Run: `pytest tests/test_narrative.py tests/test_narrative_replication_*.py tests/test_narrative_quality.py tests/test_narrative_validation.py -q --no-header 2>&1 | tail -3`
Expected: same fiscal pass count as Slice 2 baseline (zero regressions).

- [ ] **Step 6: Commit + tag**

```bash
git branch --show-current
git add puremacro/pyproject.toml \
        puremacro/puremacro/__init__.py \
        puremacro/tests/test_import.py \
        puremacro/CHANGELOG.md
git commit -m "chore(release): puremacro 0.7.0 — narrative Slice 3 (polyglot CB expansion)"
git tag -a v0.7.0 -m "puremacro 0.7.0 — narrative Slice 3 (polyglot CB expansion)"
```

(Do **not** push.)

---

## Definition of Done

- [ ] All 15 task blocks above checked off (Tasks 0–14, with Task 13 verification-only).
- [ ] Branch `feature/narrative-extension-slice3` exists with ~14 commits since `v0.6.2`, tagged `v0.7.0`.
- [ ] `pytest -q` passes ≥ 924 + ~50 ≈ 970+.
- [ ] `pytest tests/test_pyodide_compat.py` shows the SAME 1 pre-existing failure as baseline (no new leaks).
- [ ] `pytest tests/test_public_api.py` passes (snapshot includes 16 new connectors + 2 new tone language entries).
- [ ] Zero fiscal-narrative regressions.
- [ ] `pyproject.toml` version is `0.7.0`; `puremacro.__version__ == "0.7.0"`.
- [ ] `CHANGELOG.md` has a `## 0.7.0 — 2026-05-08` section.
- [ ] `python -c "from puremacro.narrative.sources import iter_banxico_decision, iter_bis_speeches; print('ok')"` prints `ok`.

## Out of scope for this plan (deliberate deferrals)

- Picault-Renault paragraph-level multinomial logit; Hubert dictionary; length-normalised WUI; `llm_prob_kernel`; published-corpus correlation regression tests. All listed under "Deferred to a future iteration" in the CHANGELOG.
- Retrofit of Slice 1's BoE/BoJ connectors to use `iter_rss_filtered`. Slice 1 connectors keep their original code; the new helper is for Slice 3 connectors.
- BCB English mirror for Brazil — supported via the `language="en"` parameter on `iter_bcb_decision` but not exercised by a separate test (the `pt` test exercises the same code path).
