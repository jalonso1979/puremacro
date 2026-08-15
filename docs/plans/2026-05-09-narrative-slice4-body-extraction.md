# Narrative Extension — Slice 4 (Body Extraction) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace crude `strip_html` with body-aware extraction (per-bank registry + generic fallback). Fix Fed minutes URL pattern. Fix BIS speeches URL. Add opt-in `fetch_body=` to all RSS-based connectors. Re-render notebook 28 to validate the LUI signal.

**Architecture:** New `narrative/sources/_extractors.py` ships `extract_body(html, *, bank_code=None)` that dispatches to per-bank extractors via a `BODY_EXTRACTORS` registry, with a generic heuristic fallback. The 3 body-fetching connectors (`fed_decision`, `fed_minutes`, `ecb_press_conf`) replace `strip_html` with `extract_body`. `iter_rss_filtered` gains `fetch_body: bool = False`; all 25 RSS-based connectors forward it as a passthrough kwarg. Notebook 28 adds `iter_fed_decision` to its corpus and opts into `fetch_body=True` for speeches. Pure stdlib (`re` + existing `_ratedoc.strip_html`); Pyodide-clean.

**Tech Stack:** Python 3.10+, `re`, `urllib`, `pandas`, `numpy`. No new top-level deps.

**Spec reference:** `docs/specs/2026-05-09-narrative-slice4-body-extraction.md`.

**Branching:** Stay on `feature/narrative-extension-slice3` (current head past `v0.7.0` with the schema-fix commit `4ab5f4d`). No new branch.

**Pre-implementation baseline:** `pytest -q` after Slice 3 + the schema fix = **956 passed, 27 skipped**. Pyodide-compat has 1 pre-existing failure (statsmodels.tsa.x13 leak). Out of scope.

---

## File Structure

### Files created
- `puremacro/narrative/sources/_extractors.py` — body extractor registry + dispatcher.
- `tests/test_narrative_extractors.py` — extractor unit tests (in `puremacro/tests/`).
- `tests/test_narrative_fed_url_transform.py` — Fed URL-pattern tests (in `puremacro/tests/`).

### Files modified
- `puremacro/narrative/sources/_rss_filtered.py` — add `fetch_body=` param.
- `puremacro/narrative/sources/fed_decision.py` — use `extract_body`.
- `puremacro/narrative/sources/fed_minutes.py` — URL transformation + `extract_body`.
- `puremacro/narrative/sources/ecb_press_conf.py` — use `extract_body`.
- `puremacro/narrative/sources/bis_speeches.py` — URL fix.
- 25 RSS-based connector files (Slice 1 RSS + all Slice 3) — add `fetch_body: bool = False` passthrough.
- `tools/make_notebook_28_us_lui_text.py` — include `iter_fed_decision`, `fetch_body=True` for speeches.
- `notebooks/28_us_lui_from_fed_text.ipynb` — re-rendered.
- `pyproject.toml` — `0.7.0 → 0.7.1`.
- `puremacro/__init__.py` — `__version__ = "0.7.1"`.
- `tests/test_import.py` — bump expected version.
- `CHANGELOG.md` — add `## 0.7.1 — 2026-05-09` block.

---

## Task 0: Branch + baseline

**Files:** none.

- [ ] **Step 1: Verify branch**

Run: `git branch --show-current`
Expected: `feature/narrative-extension-slice3`. The current HEAD past `v0.7.0` already has commit `4ab5f4d` (Fed JSON schema fix).

- [ ] **Step 2: Confirm baseline**

Run: `cd puremacro && pytest -q --no-header 2>&1 | tail -3`
Expected: `956 passed, 27 skipped, …`.

---

## Task 1: `_extractors.py` registry + Fed/ECB extractors + generic fallback

**Files:**
- Create: `puremacro/narrative/sources/_extractors.py`
- Create: `puremacro/tests/test_narrative_extractors.py`

- [ ] **Step 1: Write failing tests**

Create `puremacro/tests/test_narrative_extractors.py`:

```python
"""Tests for narrative.sources._extractors body-extraction module."""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Fed: <div id="article">…</div>
# ---------------------------------------------------------------------------
def test_extract_fed_finds_article_div():
    from puremacro.narrative.sources._extractors import extract_body
    html = (
        '<html><head><title>x</title></head><body>'
        '<nav>Top menu Economic Research Data Bank Assets</nav>'
        '<div id="article">'
        '<p>Recent indicators suggest that economic activity continued.</p>'
        '<p>The Committee judges that risks to its outlook remain.</p>'
        '</div>'
        '<footer>Footer chrome about uncertainty in policy</footer>'
        '</body></html>'
    )
    body = extract_body(html, bank_code="FED")
    assert "Recent indicators" in body
    assert "Committee judges" in body
    # Navigation chrome should NOT leak in
    assert "Top menu" not in body
    assert "Footer chrome" not in body


def test_extract_fed_falls_back_to_generic_on_no_article_div():
    from puremacro.narrative.sources._extractors import extract_body
    html = (
        '<html><body>'
        '<main><p>The body is in main, not in an article div.</p></main>'
        '</body></html>'
    )
    body = extract_body(html, bank_code="FED")
    assert "body is in main" in body


# ---------------------------------------------------------------------------
# ECB: <main id="main-wrapper"> or <div class="section">
# ---------------------------------------------------------------------------
def test_extract_ecb_finds_main_wrapper():
    from puremacro.narrative.sources._extractors import extract_body
    html = (
        '<html><body>'
        '<nav>menu</nav>'
        '<main id="main-wrapper">'
        '<p>The Governing Council decided today to raise rates.</p>'
        '</main>'
        '</body></html>'
    )
    body = extract_body(html, bank_code="ECB")
    assert "Governing Council" in body
    assert "menu" not in body


# ---------------------------------------------------------------------------
# Generic fallback: largest text-dense container
# ---------------------------------------------------------------------------
def test_extract_default_picks_largest_div():
    from puremacro.narrative.sources._extractors import extract_body
    html = (
        '<html><body>'
        '<div class="menu">x</div>'
        '<div class="content">'
        '<p>This is the actual statement body and it is the longest div on the page.</p>'
        '</div>'
        '<div class="footer">y</div>'
        '</body></html>'
    )
    body = extract_body(html)
    assert "actual statement body" in body


def test_extract_default_strips_script_and_style():
    from puremacro.narrative.sources._extractors import extract_body
    html = (
        '<html><body>'
        '<script>var x = "should not appear"</script>'
        '<style>.foo { color: red; }</style>'
        '<div><p>Real body text.</p></div>'
        '</body></html>'
    )
    body = extract_body(html)
    assert "Real body text" in body
    assert "should not appear" not in body
    assert "color: red" not in body


def test_extract_handles_empty_html():
    from puremacro.narrative.sources._extractors import extract_body
    assert extract_body("") == ""
    assert extract_body("<html></html>") == ""


def test_extract_handles_malformed_html():
    """Crude regex extraction should not crash on broken HTML."""
    from puremacro.narrative.sources._extractors import extract_body
    body = extract_body("<html><body><p>unclosed paragraph<div>nested",
                        bank_code="FED")
    # No assertion on content; just confirm it doesn't raise.
    assert isinstance(body, str)


def test_extract_unknown_bank_code_uses_generic():
    from puremacro.narrative.sources._extractors import extract_body
    html = '<html><body><div><p>Body text here.</p></div></body></html>'
    body = extract_body(html, bank_code="NOT_A_BANK")
    assert "Body text here" in body
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd puremacro && pytest tests/test_narrative_extractors.py -v --no-header 2>&1 | tail -10`
Expected: every test fails with `ModuleNotFoundError: puremacro.narrative.sources._extractors`.

- [ ] **Step 3: Create `_extractors.py`**

```python
"""HTML body extraction for CB pages (per-bank registry + generic fallback).

Replaces the crude ``_ratedoc.strip_html`` for connectors that fetch
HTML body pages. Per-bank precise extractors find a known container
(e.g. Fed's ``<div id="article">``); the generic fallback picks the
largest text-dense block.

Pure stdlib (``re``); Pyodide-clean.
"""
from __future__ import annotations

import re
from typing import Callable

from ._ratedoc import strip_html


# ---------------------------------------------------------------------------
# Whole-document preprocessing: remove obvious chrome before any per-bank
# matching. Order matters (script/style first to avoid grabbing JS strings).
# ---------------------------------------------------------------------------
_CHROME_TAG_RX = re.compile(
    r"<(script|style|nav|header|footer|aside)\b[^>]*>.*?</\1\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _drop_chrome(html: str) -> str:
    return _CHROME_TAG_RX.sub("", html)


# ---------------------------------------------------------------------------
# Per-bank extractors. Each returns either body text or None (caller
# falls back to the generic on None).
# ---------------------------------------------------------------------------
_FED_ARTICLE_RX = re.compile(
    r'<div\b[^>]*\bid\s*=\s*"article"[^>]*>(.*?)</div\s*>',
    flags=re.IGNORECASE | re.DOTALL,
)


def _extract_fed_body(html: str) -> str | None:
    m = _FED_ARTICLE_RX.search(html)
    if not m:
        return None
    return strip_html(m.group(1)).strip() or None


_ECB_MAIN_RX = re.compile(
    r'<main\b[^>]*\bid\s*=\s*"main-wrapper"[^>]*>(.*?)</main\s*>',
    flags=re.IGNORECASE | re.DOTALL,
)
_ECB_SECTION_RX = re.compile(
    r'<div\b[^>]*\bclass\s*=\s*"[^"]*\bsection\b[^"]*"[^>]*>(.*?)</div\s*>',
    flags=re.IGNORECASE | re.DOTALL,
)


def _extract_ecb_body(html: str) -> str | None:
    for rx in (_ECB_MAIN_RX, _ECB_SECTION_RX):
        m = rx.search(html)
        if m:
            text = strip_html(m.group(1)).strip()
            if text:
                return text
    return None


# ---------------------------------------------------------------------------
# Generic fallback: pick the largest text-dense container.
# ---------------------------------------------------------------------------
_BODY_RX = re.compile(r"<body\b[^>]*>(.*?)</body\s*>",
                      flags=re.IGNORECASE | re.DOTALL)
_BLOCK_RX = re.compile(
    r"<(main|article|div)\b[^>]*>(.*?)</\1\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _default_extract_body(html: str) -> str:
    """Generic heuristic: drop chrome, find largest text-dense container."""
    if not html:
        return ""
    cleaned = _drop_chrome(html)
    body_match = _BODY_RX.search(cleaned)
    scope = body_match.group(1) if body_match else cleaned

    best_text = ""
    for m in _BLOCK_RX.finditer(scope):
        candidate = strip_html(m.group(2)).strip()
        if len(candidate) > len(best_text):
            best_text = candidate

    if best_text:
        return best_text
    # Last-resort: strip the entire scope.
    return strip_html(scope).strip()


# ---------------------------------------------------------------------------
# Registry + dispatcher
# ---------------------------------------------------------------------------
BODY_EXTRACTORS: dict[str, Callable[[str], str | None]] = {
    "FED": _extract_fed_body,
    "ECB": _extract_ecb_body,
}


def extract_body(html: str, *, bank_code: str | None = None) -> str:
    """Extract main body text from a CB HTML page.

    Looks up ``bank_code`` in ``BODY_EXTRACTORS``; if registered, tries
    the per-bank extractor and falls back to the generic on None.
    Unknown bank codes go straight to the generic fallback.
    """
    if not html:
        return ""
    extractor = BODY_EXTRACTORS.get(bank_code or "")
    if extractor is not None:
        result = extractor(html)
        if result:
            return result
    return _default_extract_body(html)


__all__ = ["extract_body", "BODY_EXTRACTORS"]
```

- [ ] **Step 4: Run tests, expect green**

Run: `cd puremacro && pytest tests/test_narrative_extractors.py -v --no-header 2>&1 | tail -15`
Expected: all 7 tests pass.

- [ ] **Step 5: Run full suite, no regressions**

Run: `cd puremacro && pytest -q --no-header 2>&1 | tail -3`
Expected: 956 + 7 = 963 passed.

- [ ] **Step 6: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current   # must be feature/narrative-extension-slice3
git add puremacro/puremacro/narrative/sources/_extractors.py \
        puremacro/tests/test_narrative_extractors.py
git commit -m "feat(narrative): _extractors.py with Fed/ECB precise + generic fallback"
```

---

## Task 2: Wire `extract_body` into the 3 body-fetching connectors

**Files:**
- Modify: `puremacro/narrative/sources/fed_decision.py`
- Modify: `puremacro/narrative/sources/fed_minutes.py`
- Modify: `puremacro/narrative/sources/ecb_press_conf.py`

- [ ] **Step 1: Verify branch.**

- [ ] **Step 2: Replace `strip_html(html)` with `extract_body(html, bank_code="FED")` in `fed_decision.py`**

Locate the import line:

```python
from ._ratedoc import strip_html
```

Replace with:

```python
from ._extractors import extract_body
```

Locate the body assignment line:

```python
        text = strip_html(html)
```

Replace with:

```python
        text = extract_body(html, bank_code="FED")
```

- [ ] **Step 3: Same change in `fed_minutes.py`**

Replace `from ._ratedoc import strip_html` with `from ._extractors import extract_body`. Replace `text = strip_html(html)` with `text = extract_body(html, bank_code="FED")`.

- [ ] **Step 4: Same change in `ecb_press_conf.py`**

Replace import `from ._ratedoc import strip_html` with `from ._extractors import extract_body`. Replace `text = strip_html(body_html)` with `text = extract_body(body_html, bank_code="ECB")`.

- [ ] **Step 5: Run existing connector tests, verify still pass**

Run: `cd puremacro && pytest tests/test_narrative_cb_connectors.py -v --no-header -k "(fed or ecb_press_conf) and not network" 2>&1 | tail -15`
Expected: 4 offline tests pass (fed_decision, fed_minutes, fed_press_conf, ecb_press_conf if it has an offline test — otherwise just 3).

- [ ] **Step 6: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current
git add puremacro/puremacro/narrative/sources/fed_decision.py \
        puremacro/puremacro/narrative/sources/fed_minutes.py \
        puremacro/puremacro/narrative/sources/ecb_press_conf.py
git commit -m "feat(narrative): use extract_body in fed_decision, fed_minutes, ecb_press_conf"
```

---

## Task 3: Fed minutes URL transformation

**Files:**
- Modify: `puremacro/narrative/sources/fed_minutes.py`
- Create: `puremacro/tests/test_narrative_fed_url_transform.py`

The Fed JSON's `l` field for minutes points to `/newsevents/pressreleases/monetary{YYYYMMDD}a.htm` (announcement page, mostly chrome). The actual minutes body is at `/monetarypolicy/fomcminutes{YYYYMMDD}.htm`.

- [ ] **Step 1: Verify branch.**

- [ ] **Step 2: Write failing tests**

Create `puremacro/tests/test_narrative_fed_url_transform.py`:

```python
"""Tests for the Fed minutes announcement → body URL transformation."""
from __future__ import annotations


def test_minutes_body_url_transforms_canonical_pattern():
    from puremacro.narrative.sources.fed_minutes import _minutes_body_url
    out = _minutes_body_url(
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20060103a.htm"
    )
    assert out == "https://www.federalreserve.gov/monetarypolicy/fomcminutes20060103.htm"


def test_minutes_body_url_handles_relative_input():
    from puremacro.narrative.sources.fed_minutes import _minutes_body_url
    out = _minutes_body_url("/newsevents/pressreleases/monetary20240501a.htm")
    assert out == "https://www.federalreserve.gov/monetarypolicy/fomcminutes20240501.htm"


def test_minutes_body_url_returns_input_when_pattern_does_not_match():
    from puremacro.narrative.sources.fed_minutes import _minutes_body_url
    other = "https://www.federalreserve.gov/something/else.htm"
    assert _minutes_body_url(other) == other


def test_minutes_body_url_handles_no_a_suffix():
    from puremacro.narrative.sources.fed_minutes import _minutes_body_url
    # Some older URLs lack the trailing 'a' before .htm
    out = _minutes_body_url(
        "/newsevents/pressreleases/monetary20060103.htm"
    )
    assert out == "https://www.federalreserve.gov/monetarypolicy/fomcminutes20060103.htm"
```

- [ ] **Step 3: Run, verify failure**

Run: `cd puremacro && pytest tests/test_narrative_fed_url_transform.py -v --no-header 2>&1 | tail -10`
Expected: 4 fail with `ImportError: cannot import name '_minutes_body_url'`.

- [ ] **Step 4: Add `_minutes_body_url` and use it inside `iter_fed_minutes`**

In `puremacro/narrative/sources/fed_minutes.py`, near the existing imports add:

```python
import re

_FOMC_MINUTES_URL_RX = re.compile(r"monetary(\d{8})a?\.htm", re.IGNORECASE)


def _minutes_body_url(announcement_href: str) -> str:
    """Map the FOMC minutes announcement URL to the actual body URL.

    /newsevents/pressreleases/monetary{YYYYMMDD}a?.htm
        → /monetarypolicy/fomcminutes{YYYYMMDD}.htm

    Returns the input unchanged when the pattern does not match.
    """
    m = _FOMC_MINUTES_URL_RX.search(announcement_href)
    if not m:
        return announcement_href
    ymd = m.group(1)
    return f"{_BASE}/monetarypolicy/fomcminutes{ymd}.htm"
```

(Put `_minutes_body_url` near the top of the file, alongside `_LISTING_URL`, `_BASE`, `_UA`.)

In the body of `iter_fed_minutes`, change the URL-resolution logic so we try the body URL first, fall back to the announcement URL on 404 or short body:

Find:

```python
        href = item.get("l", "")
        if not href:
            continue
        item_url = _BASE + href if href.startswith("/") else href
        try:
            html = safe_get_text(item_url, user_agent=_UA)
        except Exception:
            continue
        text = extract_body(html, bank_code="FED")
        if not text:
            continue
        yield (date, text, item_url, {
            "doctype": "minutes", "language": "en",
            "bank_code": "FED", "country": "USA",
        })
```

Replace with:

```python
        href = item.get("l", "")
        if not href:
            continue
        announcement_url = _BASE + href if href.startswith("/") else href
        body_url = _minutes_body_url(announcement_url)

        # Try body URL first; fall back to announcement on failure or short body.
        text = ""
        chosen_url = body_url
        try:
            html = safe_get_text(body_url, user_agent=_UA)
            text = extract_body(html, bank_code="FED")
        except Exception:
            text = ""
        if len(text) < 5000 and body_url != announcement_url:
            try:
                html = safe_get_text(announcement_url, user_agent=_UA)
                text = extract_body(html, bank_code="FED")
                chosen_url = announcement_url
            except Exception:
                continue
        if not text:
            continue
        yield (date, text, chosen_url, {
            "doctype": "minutes", "language": "en",
            "bank_code": "FED", "country": "USA",
        })
```

- [ ] **Step 5: Run new + existing tests**

Run: `cd puremacro && pytest tests/test_narrative_fed_url_transform.py tests/test_narrative_cb_connectors.py -v --no-header -k "fed_minutes or url_transform" 2>&1 | tail -15`
Expected: all URL-transform tests pass; existing fed_minutes offline test still passes (the mock just stays at the announcement URL since the body URL won't be in the mock fixture).

If `test_fed_minutes_yields_four_tuple` now fails because the connector first tries the body URL and `mock_http` raises `LookupError` for an unmocked URL, update the mock. Add to that test's `mock_http` call a body URL response (since the connector hits both):

```python
mock_http(
    bytes_={...existing JSON listing...},
    text={
        "https://www.federalreserve.gov/monetarypolicy/fomcminutes20220316.htm":
            "<html><body><div id=\"article\"><p>Participants noted that inflation...</p></div></body></html>",
        # Keep the existing announcement URL as a fallback registration:
        "https://www.federalreserve.gov/monetarypolicy/fomcminutes20220316.htm":
            "<html><body><div id=\"article\"><p>Participants noted that inflation remained elevated.</p></div></body></html>",
    },
)
```

(The test should be updated as part of this task to register the body URL with the right HTML containing `<div id="article">`.)

- [ ] **Step 6: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current
git add puremacro/puremacro/narrative/sources/fed_minutes.py \
        puremacro/tests/test_narrative_fed_url_transform.py \
        puremacro/tests/test_narrative_cb_connectors.py
git commit -m "feat(narrative): Fed minutes URL transform (announcement → body) + extract_body wiring"
```

---

## Task 4: BIS speeches URL fix

**Files:**
- Modify: `puremacro/narrative/sources/bis_speeches.py`

The current `https://www.bis.org/cbspeeches/index.rss` is 404. We probe alternatives at implementation time.

- [ ] **Step 1: Verify branch.**

- [ ] **Step 2: Probe candidate URLs from the working tree**

Run each command and note which (if any) returns non-empty bytes:

```bash
python -c "
from puremacro._http import safe_get_bytes
candidates = [
    'https://www.bis.org/list/cbspeeches/index.rss',
    'https://www.bis.org/cbspeeches/index.rss',
    'https://www.bis.org/rss/cbspeeches.rss',
    'https://www.bis.org/cbspeeches/recent.rss',
]
for u in candidates:
    try:
        b = safe_get_bytes(u)
        print(f'OK ({len(b)} bytes): {u}')
    except Exception as e:
        print(f'FAIL: {u} — {e}')
"
```

- [ ] **Step 3: If a candidate URL works, update the connector**

In `puremacro/narrative/sources/bis_speeches.py`, replace the `_FEED` constant with the working URL.

If NO RSS URL works, switch to HTML-listing scraping. Replace the file's `iter_bis_speeches` with:

```python
"""Bank for International Settlements speech archive (multi-bank).

The BIS does not publish a stable RSS feed for the central-bank speeches
archive (the ``/cbspeeches/index.rss`` URL is 404 as of 2026-05). We
scrape the HTML listing at ``https://www.bis.org/cbspeeches/`` and
extract date+title+href trios. The structure can change without
notice; the connector is best-effort and yields nothing on parse
failure (per RETRY_POLICY).
"""
from __future__ import annotations

import re
from typing import Iterator

import pandas as pd

from ..._http import safe_get_text


_LISTING_URL = "https://www.bis.org/cbspeeches/"
_LINK_RX = re.compile(
    r'<a\b[^>]+href="(/review/r\d+[a-z]?\.htm)"[^>]*>(.*?)</a>',
    flags=re.IGNORECASE | re.DOTALL,
)
_DATE_RX = re.compile(
    r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|"
    r"July|August|September|October|November|December)\s+\d{4})\b",
)


def iter_bis_speeches(*, bank_filter: str | None = None) -> Iterator[tuple]:
    """Yield (date, text, url, metadata) for BIS-republished CB speeches.

    Parameters
    ----------
    bank_filter : optional substring to match against the title (case-
        insensitive). Useful to narrow to one institution's speeches
        (e.g., ``"Lagarde"`` for ECB, ``"Powell"`` for Fed).
    """
    try:
        html = safe_get_text(_LISTING_URL)
    except Exception:
        return
    for m in _LINK_RX.finditer(html):
        href, title_html = m.group(1), m.group(2)
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        if not title:
            continue
        if bank_filter and bank_filter.lower() not in title.lower():
            continue
        # Find a date token in the surrounding ~200 chars.
        window = html[max(0, m.start() - 200):m.end() + 200]
        date_match = _DATE_RX.search(window)
        if not date_match:
            continue
        try:
            date = pd.Timestamp(date_match.group(1))
        except Exception:
            continue
        url = f"https://www.bis.org{href}" if href.startswith("/") else href
        yield (date, title, url, {
            "doctype": "speech", "language": "en",
            "bank_code": "BIS", "country": "MULTI",
        })


__all__ = ["iter_bis_speeches"]
```

- [ ] **Step 4: Update offline tests**

In `puremacro/tests/test_narrative_slice3_connectors.py`, replace the BIS test bytes-mock with a text-mock matching the HTML scrape format. Find:

```python
def test_bis_speeches_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.bis.org/cbspeeches/index.rss":
            b'<?xml version="1.0"?><rss>...
```

Replace with the new URL/format that matches whichever path Step 3 chose. If RSS works, leave the test largely as-is but update the URL. If HTML scraping, switch to:

```python
def test_bis_speeches_yields_four_tuple(mock_http):
    mock_http(text={
        "https://www.bis.org/cbspeeches/":
            '<html><body>'
            '<a href="/review/r221015a.htm">Lagarde on inflation</a>'
            '<span>15 October 2022</span>'
            '</body></html>',
    })
    from puremacro.narrative.sources import iter_bis_speeches
    records = list(iter_bis_speeches())
    if records:  # Best-effort scraping; canonical fixture should match.
        _, text, _, meta = records[0]
        assert "Lagarde" in text
        assert meta["bank_code"] == "BIS"
        assert meta["country"] == "MULTI"
```

Same for `test_bis_speeches_filter_by_bank_keyword` — adapt mock to HTML format.

- [ ] **Step 5: Run BIS tests**

Run: `cd puremacro && pytest tests/test_narrative_slice3_connectors.py -v --no-header -k "bis and not network" 2>&1 | tail -10`
Expected: 2 tests pass.

- [ ] **Step 6: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current
git add puremacro/puremacro/narrative/sources/bis_speeches.py \
        puremacro/tests/test_narrative_slice3_connectors.py
git commit -m "fix(narrative): BIS speeches URL — switch to HTML listing scrape (RSS is 404)"
```

---

## Task 5: `iter_rss_filtered` gets `fetch_body=` param

**Files:**
- Modify: `puremacro/narrative/sources/_rss_filtered.py`
- Modify: `puremacro/tests/test_narrative_cb_connectors.py` (add a `fetch_body=True` test)

- [ ] **Step 1: Verify branch.**

- [ ] **Step 2: Write failing test**

Append to `puremacro/tests/test_narrative_cb_connectors.py`:

```python
def test_iter_rss_filtered_fetch_body_replaces_summary(mock_http):
    """When fetch_body=True, the connector pulls the link target and
    replaces the RSS summary text with extract_body(...)."""
    mock_http(
        bytes_={
            "https://example.test/rss.xml":
                b'<?xml version="1.0"?><rss><channel><item>'
                b'<title>Decision title</title>'
                b'<description>Short summary.</description>'
                b'<link>https://example.test/page.html</link>'
                b'<pubDate>Tue, 01 Mar 2022 12:00:00 +0000</pubDate>'
                b'</item></channel></rss>',
        },
        text={
            "https://example.test/page.html":
                '<html><body>'
                '<div id="article"><p>The full body of the decision is here.</p></div>'
                '</body></html>',
        },
    )
    from puremacro.narrative.sources._rss_filtered import iter_rss_filtered
    records = list(iter_rss_filtered(
        "https://example.test/rss.xml",
        bank_code="FED", country="USA", doctype="decision",
        language="en", fetch_body=True,
    ))
    assert len(records) == 1
    _, text, _, _ = records[0]
    assert "full body of the decision" in text
    assert "Short summary" not in text   # body replaces summary
```

- [ ] **Step 3: Run, verify failure**

Run: `cd puremacro && pytest tests/test_narrative_cb_connectors.py -v --no-header -k "fetch_body" 2>&1 | tail -10`
Expected: fail with TypeError or unexpected kwarg.

- [ ] **Step 4: Modify `_rss_filtered.py`**

Open `puremacro/narrative/sources/_rss_filtered.py`. Replace the function with:

```python
"""Shared RSS-feed wrapper with optional title-keyword filtering and
optional body-fetch.
"""
from __future__ import annotations

from typing import Iterator

from ..._http import safe_get_text
from ._rss import iter_rss
from ._ratedoc import strip_html
from ._extractors import extract_body


def iter_rss_filtered(
    url: str,
    *,
    bank_code: str,
    country: str,
    doctype: str,
    language: str = "en",
    title_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None,
    fetch_body: bool = False,
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
    fetch_body : default ``False``. If ``True``, fetch the link target
        URL for each item and replace the RSS-summary text with the
        extracted body. Body fetch failures (or short bodies < 200
        chars) fall back to the RSS summary so the connector never
        yields empty text. Doubles HTTP calls per item; opt-in.
    """
    for date, title_desc, link in iter_rss(url):
        clean = strip_html(title_desc) if "<" in title_desc else title_desc
        low = clean.lower()
        if title_keywords and not any(kw.lower() in low for kw in title_keywords):
            continue
        if exclude_keywords and any(kw.lower() in low for kw in exclude_keywords):
            continue
        if fetch_body and link:
            try:
                body_html = safe_get_text(link)
                body_text = extract_body(body_html, bank_code=bank_code)
                if len(body_text) >= 200:
                    clean = body_text
            except Exception:
                pass
        yield (date, clean, link, {
            "doctype": doctype, "language": language,
            "bank_code": bank_code, "country": country,
        })


__all__ = ["iter_rss_filtered"]
```

- [ ] **Step 5: Run tests, expect green**

Run: `cd puremacro && pytest tests/test_narrative_cb_connectors.py -v --no-header -k "fetch_body or rss_filtered" 2>&1 | tail -10`
Expected: new test passes.

- [ ] **Step 6: Run full suite, no regressions**

Run: `cd puremacro && pytest -q --no-header 2>&1 | tail -3`
Expected: 956 + 7 + 4 + 1 = 968 (give or take).

- [ ] **Step 7: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current
git add puremacro/puremacro/narrative/sources/_rss_filtered.py \
        puremacro/tests/test_narrative_cb_connectors.py
git commit -m "feat(narrative): iter_rss_filtered gains fetch_body= param (default False)"
```

---

## Task 6: Forward `fetch_body=` through all RSS-based connectors

**Files (modify all):**
- Slice 1 (4 RSS connectors): `puremacro/narrative/sources/{fed_speeches,ecb_decision,ecb_minutes,ecb_speeches,boe_decision,boe_minutes,boe_speeches,boj_decision,boj_speeches}.py`
- Slice 3 (16 RSS connectors): `puremacro/narrative/sources/{banxico,bcb,bccl,bcra,banrep,rba,rbnz,riksbank,norges,sarb,rbi,bok,mas,bot,bis_speeches}.py` (some have multiple iter functions)

- [ ] **Step 1: Verify branch.**

- [ ] **Step 2: Apply the same passthrough pattern to every connector**

For each file, change the function signature from:

```python
def iter_<bank>_<doctype>() -> Iterator[tuple]:
    yield from iter_rss_filtered(_FEED, bank_code=..., country=..., doctype=..., language=...)
```

to:

```python
def iter_<bank>_<doctype>(*, fetch_body: bool = False) -> Iterator[tuple]:
    yield from iter_rss_filtered(_FEED, bank_code=..., country=..., doctype=..., language=..., fetch_body=fetch_body)
```

Apply this to every `iter_<bank>_<doctype>` function in the connector files listed above. Skip connectors that don't go through `iter_rss_filtered` (the 3 body-fetching ones from Task 2 already use `extract_body` directly; PBoC HTML-scrape connector doesn't use the helper).

If a connector has multiple iter functions (e.g. `iter_rba_decision` and `iter_rba_speeches`), update both.

If a connector has additional kwargs (e.g. `iter_bcb_decision(*, language: str = "pt")`), put `fetch_body=False` after the existing kwargs:

```python
def iter_bcb_decision(*, language: str = "pt", fetch_body: bool = False) -> Iterator[tuple]:
    if language not in _FEED_BY_LANG:
        raise ValueError(...)
    yield from iter_rss_filtered(
        _FEED_BY_LANG[language], bank_code="BCB", country="BRA",
        doctype="decision", language=language, fetch_body=fetch_body,
    )
```

For `iter_bis_speeches`, only update if Task 4 kept it on `iter_rss_filtered`. If Task 4 switched it to direct HTML scraping, leave it (it doesn't share the pattern; body fetching there is a separate concern).

- [ ] **Step 3: Sanity import**

Run:
```bash
python -c "
import inspect
from puremacro.narrative.sources import (
    iter_fed_speeches, iter_ecb_decision, iter_boe_decision, iter_boj_decision,
    iter_banxico_decision, iter_bcb_decision, iter_bccl_decision,
    iter_bcra_decision, iter_banrep_decision,
    iter_rba_decision, iter_rba_speeches, iter_rbnz_decision,
    iter_riksbank_decision, iter_norges_decision, iter_sarb_decision,
    iter_rbi_decision, iter_bok_decision, iter_mas_decision, iter_bot_decision,
)
for fn in [iter_fed_speeches, iter_ecb_decision, iter_boe_decision,
           iter_boj_decision, iter_banxico_decision, iter_bcb_decision,
           iter_rba_decision, iter_rba_speeches, iter_rbi_decision,
           iter_bok_decision, iter_mas_decision, iter_bot_decision]:
    sig = inspect.signature(fn)
    assert 'fetch_body' in sig.parameters, f'{fn.__name__} missing fetch_body'
print('all checked connectors have fetch_body=')
"
```
Expected: `all checked connectors have fetch_body=`.

- [ ] **Step 4: Run existing connector tests**

Run: `cd puremacro && pytest tests/test_narrative_cb_connectors.py tests/test_narrative_slice3_connectors.py -v --no-header -k "not network" 2>&1 | tail -20`
Expected: all offline tests pass (the new `fetch_body=False` default preserves existing behavior).

- [ ] **Step 5: Run full suite**

Run: `cd puremacro && pytest -q --no-header 2>&1 | tail -3`
Expected: still ≥ 968 passed.

- [ ] **Step 6: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current
git add puremacro/puremacro/narrative/sources/
git commit -m "feat(narrative): forward fetch_body= through all RSS-based connectors"
```

---

## Task 7: Update notebook 28 builder

**Files:**
- Modify: `tools/make_notebook_28_us_lui_text.py`

The current builder fetches FOMC minutes + Fed speeches but NOT FOMC statements. Add `iter_fed_decision` and pass `fetch_body=True` to speeches so the corpus has real body text.

- [ ] **Step 1: Verify branch.**

- [ ] **Step 2: Update the builder**

Open `tools/make_notebook_28_us_lui_text.py`. Find the import in §1's code cell:

```python
from puremacro.narrative.sources import (
    iter_fed_minutes, iter_fed_speeches,
)
```

Change to:

```python
from puremacro.narrative.sources import (
    iter_fed_decision, iter_fed_minutes, iter_fed_speeches,
)
```

In §2 (corpus assembly cell), find:

```python
    for src_iter, src_name in [(iter_fed_minutes, 'minutes'),
                                (iter_fed_speeches, 'speeches')]:
```

Change to:

```python
    for src_iter, src_name, kwargs in [
        (iter_fed_decision, 'decision', {}),
        (iter_fed_minutes, 'minutes', {}),
        (iter_fed_speeches, 'speeches', {'fetch_body': True}),
    ]:
        try:
            for date, text, url, meta in src_iter(**kwargs):
```

This pulls statements (full body via Task 2's extractor), minutes (full body via Task 3's URL transform + extractor), and speeches (full body via Task 5/6's `fetch_body=True`).

- [ ] **Step 3: Re-render the notebook**

Run:
```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
python tools/make_notebook_28_us_lui_text.py
```
Expected: `wrote /…/notebooks/28_us_lui_from_fed_text.ipynb`.

- [ ] **Step 4: Smoke test the regenerated notebook**

Run: `cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples" && pytest tests/test_notebook_28_smoke.py -v --no-header 2>&1 | tail -10`
Expected: 2 tests pass (the smoke test only asserts builder structure, not signal quality — that's Task 8).

- [ ] **Step 5: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current
git add tools/make_notebook_28_us_lui_text.py \
        notebooks/28_us_lui_from_fed_text.ipynb
git commit -m "feat(notebooks): nb28 — include iter_fed_decision + fetch_body for speeches"
```

---

## Task 8: Re-run notebook 28 + validate signal quality

**Files:** none (interactive validation; outputs go to `notebooks/output_*`).

This is a manual / interactive validation step. The implementer (or user) runs the notebook and inspects the outputs.

- [ ] **Step 1: Clear the stale corpus cache**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
rm -f notebooks/data_cache/fed_corpus_28.parquet
```

- [ ] **Step 2: Execute the notebook with refetch enabled**

```bash
PUREMACRO_REFETCH=1 jupyter execute notebooks/28_us_lui_from_fed_text.ipynb \
  --output 28_us_lui_from_fed_text.executed.ipynb
```
Expected output (timing varies; 5–15 minutes total because of N+1 HTTP calls per speech):
```
[NbClientApp] Executing notebooks/28_us_lui_from_fed_text.ipynb
[NbClientApp] Save executed results to notebooks/28_us_lui_from_fed_text.executed.ipynb
```

- [ ] **Step 3: Inspect the corpus + indices**

```bash
cat notebooks/output_tables/28_lui_us_quarterly.meta.json
cat notebooks/output_tables/28_lui_validation_corr.csv
python -c "
import pandas as pd
p = pd.read_parquet('notebooks/output_tables/28_lui_us_quarterly.parquet')
print(p.describe())
print('non-zero quarters per index:')
for c in p.columns:
    print(f'  {c}: {(p[c] != 0).sum()} / {len(p)}')
"
```

Acceptance criteria:
- `corpus_size` ≥ 200 (was 15 / 180 before fixes).
- `n_q` for `lui` vs `urate_us` ≥ 30 (was 0 / 76).
- `rho` for any of (lui, epu, wui) vs (bbd_epu, urate_us) is `>= 0.30` in absolute value. **At least one combination** should hit ≥ 0.30 if the body extraction worked. If all rho < 0.30, the lexicons are still the bottleneck — document and defer to a future iteration but still ship the slice.

- [ ] **Step 4: Commit the executed notebook outputs**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current
git add notebooks/28_us_lui_from_fed_text.executed.ipynb \
        notebooks/output_tables/28_lui_us_quarterly.parquet \
        notebooks/output_tables/28_lui_us_quarterly.meta.json \
        notebooks/output_tables/28_lui_validation_corr.csv \
        notebooks/output_figures/28_lui_us_timeseries.pdf \
        notebooks/data_cache/fed_corpus_28.parquet
git commit -m "feat(notebooks): nb28 executed outputs after Slice-4 body-extraction fixes"
```

If the cache file is too large to commit (> 50 MB), don't commit it; let it stay local-only. The other outputs (PDF, parquet panel, CSV) should be small enough to commit.

---

## Task 9: Version bump 0.7.1 + CHANGELOG + tag

**Files:**
- Modify: `puremacro/pyproject.toml`
- Modify: `puremacro/puremacro/__init__.py`
- Modify: `puremacro/tests/test_import.py`
- Modify: `puremacro/CHANGELOG.md`

- [ ] **Step 1: Verify branch.**

- [ ] **Step 2: Bump version**

Edit `puremacro/pyproject.toml`: `version = "0.7.0"` → `version = "0.7.1"`.

Edit `puremacro/puremacro/__init__.py`: `__version__ = "0.7.0"` → `__version__ = "0.7.1"`.

Edit `puremacro/tests/test_import.py`: `assert puremacro.__version__ == "0.7.0"` → `"0.7.1"`.

- [ ] **Step 3: Add CHANGELOG entry**

Open `puremacro/CHANGELOG.md`. Add a new top entry above the `## 0.7.0 — 2026-05-08` block:

```markdown
## 0.7.1 — 2026-05-09

Slice 4: body extraction + connector bug fixes. Triggered by notebook 28's flat-zero LUI signal — investigation surfaced multiple foundation issues that masked any meaningful signal across CB connectors.

### Fixed

- **Fed JSON listing parser** (Slice 1 schema bug, fixed in commit `4ab5f4d` shipped here): real endpoint serves a top-level list under UTF-8 BOM with key `t`, not the `{"refData": [...]}` shape mocked in Slice 1 tests. Fixed parser handles both shapes.
- **Fed minutes URL pattern**: the JSON `l` field gives the press-release announcement URL; the actual minutes body is at `/monetarypolicy/fomcminutes{YYYYMMDD}.htm`. New `_minutes_body_url` helper transforms the URL; `iter_fed_minutes` tries the body URL first, falls back to the announcement URL on 404 or short body.
- **`strip_html` was too crude for modern Fed pages** — the cleaner kept menu chrome alongside body content, polluting lexicon counts. New `puremacro.narrative.sources._extractors.extract_body(html, *, bank_code=None)` dispatches to per-bank precise extractors via `BODY_EXTRACTORS` registry (Fed, ECB), with a generic heuristic fallback for the long tail.
- **BIS speeches URL was 404**. The Slice 3 RSS URL `https://www.bis.org/cbspeeches/index.rss` doesn't exist. Switched to HTML-listing scraping at `/cbspeeches/` (or RSS if implementer probe finds a working URL).

### Added

- `narrative.sources._extractors.extract_body(html, *, bank_code=None)` — public dispatcher with `BODY_EXTRACTORS` registry (Fed, ECB pre-registered; others use generic).
- `iter_rss_filtered(...)` gains opt-in `fetch_body: bool = False`. When `True`, fetches each item's link target and replaces RSS-summary text with the extracted body. Failures + short bodies fall back to RSS summary; doubles HTTP calls per item.
- All ~25 RSS-based CB connectors gain a `fetch_body: bool = False` passthrough keyword. Backward-compatible.
- Notebook 28 builder now includes `iter_fed_decision` (was missing) and uses `fetch_body=True` for `iter_fed_speeches`.
- `tests/test_narrative_extractors.py` (7 tests).
- `tests/test_narrative_fed_url_transform.py` (4 tests).
- `tests/test_narrative_cb_connectors.py::test_iter_rss_filtered_fetch_body_replaces_summary` (1 test).

### Pyodide compatibility

- `_extractors.py` is pure-Python (`re` + `_ratedoc.strip_html`), no new top-level deps. `narrative.sources/*` stays in Experimental tier; pyodide-compat test exclusions unchanged. Slice 4 added zero new forbidden-runtime-dep leaks.

### Notes for next iteration

- If notebook 28 still shows weak signal after Slice 4, the bottleneck is lexicons, not extraction. Slice 5 candidates: lexicon expansion (LUI especially), length-normalized WUI, Picault-Renault classifier.
- Per-bank precise extractors for Slice-3 banks (BoE, BoJ, LATAM, Asia EM) can be added incrementally as research surfaces signal-quality issues.
- BIS HTML scrape is best-effort; structure can change without notice.
```

- [ ] **Step 4: Run final regression sweep**

Run: `cd puremacro && pytest -q --no-header 2>&1 | tail -3`
Expected: ≥ 968 passed (956 baseline + ~12 new tests across Tasks 1+3+5).

Run fiscal regression: `cd puremacro && pytest tests/test_narrative.py tests/test_narrative_replication_*.py tests/test_narrative_quality.py -q --no-header 2>&1 | tail -3`
Expected: same as baseline (zero fiscal regressions).

- [ ] **Step 5: Commit + tag**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current
git add puremacro/pyproject.toml \
        puremacro/puremacro/__init__.py \
        puremacro/tests/test_import.py \
        puremacro/CHANGELOG.md
git commit -m "chore(release): puremacro 0.7.1 — narrative Slice 4 (body extraction + bug fixes)"
git tag -a v0.7.1 -m "puremacro 0.7.1 — narrative Slice 4 (body extraction + bug fixes)"
```

(Do NOT push.)

---

## Definition of Done

- [ ] All 10 task blocks above checked off (Tasks 0–9).
- [ ] Branch `feature/narrative-extension-slice3` has new commits past `v0.7.0`, tagged `v0.7.1`.
- [ ] `pytest -q` ≥ 968 passed.
- [ ] `pytest tests/test_pyodide_compat.py` shows the SAME 1 pre-existing failure (no new leaks).
- [ ] Zero fiscal-narrative regressions.
- [ ] `pyproject.toml` version is `0.7.1`; `puremacro.__version__ == "0.7.1"`.
- [ ] `CHANGELOG.md` has a `## 0.7.1 — 2026-05-09` section.
- [ ] Notebook 28 has been executed and the executed-output notebook + supporting CSVs/PDF are committed.
- [ ] Signal-quality acceptance criterion either met (one rho ≥ 0.30) or its absence honestly documented in the executed notebook.

## Out of scope (deferred)

- Per-bank precise extractors for Slice-3 banks (BoE, BoJ, LATAM, Asia EM). All use the generic fallback.
- Lexicon expansion (LUI/EPU/WUI). Re-evaluate after Slice 4 produces real bodies.
- Length-normalized WUI per Ahir-Bloom-Furceri.
- Picault-Renault paragraph-level classifier; full Hubert lexicon.
- `llm_prob_kernel` for LLM-backed scoring.
- Notebook 29 (state-panel LP-IV with national LUI as shock).
- Vendor `readability-lxml` or `trafilatura` (would break Pyodide promise).
