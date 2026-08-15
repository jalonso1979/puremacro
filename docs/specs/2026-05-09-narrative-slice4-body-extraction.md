# Narrative Extension — Slice 4 (Body Extraction + Connector Bug Fixes)

**Status:** Drafted 2026-05-09. Triggered by notebook 28 (`docs/specs/2026-05-09-notebook-28-us-lui-design.md`) surfacing real-world foundation bugs that masked the LUI / EPU / WUI signal.
**Driving lens:** Make notebook 28 produce a research-usable LUI series. Generalize the fix so the same infrastructure unlocks deeper signal across all CB connectors.

## Motivation

Notebook 28 ran end-to-end and produced flat-zero indices. Investigation surfaced three foundation bugs and one architectural limitation:

1. **Fed JSON listing parser was wrong** (Slice 1 bug). Fixed in commit `4ab5f4d`. The endpoint serves a top-level list under UTF-8 BOM with key `t`, not the `{"refData": [...]}` shape that the Slice-1 mock implied. Now finds 186 FOMC statements + ~132 minutes since 2016.

2. **Fed minutes URL pattern bug.** The JSON's `l` field gives the press-release announcement URL (`/newsevents/pressreleases/monetary{YYYYMMDD}a.htm`). For *minutes*, that page is mostly site chrome. The actual minutes body lives at `/monetarypolicy/fomcminutes{YYYYMMDD}.htm`. Connector currently fetches the wrong URL and gets ~19K of navigation text per item.

3. **`strip_html` is too crude for modern Fed pages.** Even when the right URL is fetched, `<div id="article">` (the actual body container, ~3-5K chars of statement) sits inside ~80K of page chrome, navigation menus, footer links, and JS embeds. Removing tags but keeping all surrounding text means menu items ("Economic Research Data", "Bank Assets and Liabilities") appear as if they were body content. Lexicon counts then fire on menu words instead of statement words. EPU/WUI returned 0 across 165 minutes; LUI was constant at –0.11 (one labor term in the menu, no signal).

4. **BIS speeches URL is 404.** The Slice-3 connector pointed at `https://www.bis.org/cbspeeches/index.rss`, which doesn't exist. Connector silently returns nothing per `RETRY_POLICY` — masking the configuration bug.

5. **Architectural limitation: RSS-based connectors yield only summary text.** 14 of the 17 CB connectors call `iter_rss_filtered` which emits the RSS-summary text (title + description, typically 1-2 sentences). The actual body lives at the link target. To get a research-grade corpus from any of those banks, we need to fetch + extract the body of each linked page. Currently we don't.

This slice closes #2 (minutes URL), #3 (extraction), #4 (BIS URL), and #5 (RSS body-fetch). Plus reusable infrastructure.

## Non-goals

- **No** vendoring of `readability-lxml` or `trafilatura`. Pyodide-clean stdlib only.
- **No** lexicon expansion (LUI/EPU/WUI term lists stay as-is). Pending evidence from a real corpus that lexicons are the bottleneck rather than extraction quality.
- **No** Picault-Renault paragraph-level classifier, no full Hubert lexicon, no `llm_prob_kernel`. All deferred from earlier slice reviews.
- **No** length-normalized WUI per Ahir-Bloom-Furceri. Same deferral.
- **No** rewrite of how connectors yield records (still 4-tuple `(date, text, source_url, metadata)`).
- **No** notebook 29 (state-panel LP-IV with national LUI as shock). That comes after Slice 4 lands and notebook 28 produces a defensible signal.

## Architecture

### New module: `puremacro/narrative/sources/_extractors.py`

```python
def extract_body(html: str, *, bank_code: str | None = None) -> str:
    """Extract the main body text from a CB HTML page.

    Dispatches by ``bank_code`` to a registered per-bank extractor;
    falls back to the generic heuristic if no override is registered
    or if the override's regex does not match.
    """

BODY_EXTRACTORS: dict[str, Callable[[str], str | None]] = {
    "FED": _extract_fed_body,
    "ECB": _extract_ecb_body,
    # BoE / BoJ + Slice-3 banks default to None (use generic).
}

def _default_extract_body(html: str) -> str:
    """Generic heuristic for unknown banks.

    Strategy:
      1. Drop everything outside <body>...</body>.
      2. Drop <script>, <style>, <nav>, <header>, <footer>, <aside>
         elements (regex-based, since stdlib's html.parser doesn't do
         this cleanly without lxml).
      3. Find the largest text-dense container (<main>, <article>, or
         the largest <div> by stripped text length).
      4. Strip remaining tags via the existing strip_html.
    """

def _extract_fed_body(html: str) -> str | None:
    """Find <div id="article">…</div> on Fed pages. Returns None on miss."""

def _extract_ecb_body(html: str) -> str | None:
    """Find <main id="main-wrapper"> or <div class="section">. None on miss."""
```

Pure stdlib `re` + the existing `strip_html` from `_ratedoc.py`. No new top-level deps. Pyodide-clean.

### `iter_rss_filtered` gets `fetch_body: bool = False`

```python
def iter_rss_filtered(
    url: str,
    *,
    bank_code: str,
    country: str,
    doctype: str,
    language: str = "en",
    title_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None,
    fetch_body: bool = False,                    # NEW
) -> Iterator[tuple]:
    """If fetch_body=True, additionally fetch the link target page and
    replace the summary text with the extracted body. Body fetch
    failures fall back to the summary (connector never yields empty
    text). Doubles HTTP calls per item; opt-in to preserve current
    behavior.
    """
```

### All RSS-based connectors get `fetch_body: bool = False`

Every connector that currently calls `iter_rss_filtered` gets a passthrough parameter:

```python
def iter_boe_decision(*, fetch_body: bool = False) -> Iterator[tuple]:
    yield from iter_rss_filtered(_FEED, ..., fetch_body=fetch_body)
```

Backward-compatible: every existing call site keeps working; new callers can opt in. The full list (~25 connectors): all of `iter_fed_speeches`, `iter_ecb_decision`, `iter_ecb_minutes`, `iter_ecb_speeches`, `iter_boe_decision`, `iter_boe_minutes`, `iter_boe_speeches`, `iter_boj_decision`, `iter_boj_speeches`, `iter_banxico_decision`, `iter_bcb_decision`, `iter_bccl_decision`, `iter_bcra_decision`, `iter_banrep_decision`, `iter_rba_decision`, `iter_rba_speeches`, `iter_rbnz_decision`, `iter_riksbank_decision`, `iter_norges_decision`, `iter_sarb_decision`, `iter_rbi_decision`, `iter_bok_decision`, `iter_mas_decision`, `iter_bot_decision`, `iter_bis_speeches`.

### Connectors that already fetch HTML bodies get `extract_body`

`fed_decision.py`, `fed_minutes.py`, `ecb_press_conf.py` currently call `strip_html(html)`. Each gets replaced with `extract_body(html, bank_code="FED" | "ECB")`.

### Fed minutes URL transformation

In `iter_fed_minutes`, after pulling `href` from the JSON listing, compute the body URL:

```python
import re
_FOMC_BODY_URL_RX = re.compile(r"monetary(\d{8})a?\.htm", re.I)

def _minutes_body_url(announcement_href: str) -> str:
    """Transform announcement URL to actual minutes body URL.

    /newsevents/pressreleases/monetary20060103a.htm
    →  /monetarypolicy/fomcminutes20060103.htm
    """
    m = _FOMC_BODY_URL_RX.search(announcement_href)
    if not m:
        return announcement_href
    ymd = m.group(1)
    return f"/monetarypolicy/fomcminutes{ymd}.htm"
```

Try the body URL first; on 404 / empty / short body (< 5000 chars), fall back to the announcement URL. The fallback handles older Fed pages where the body URL convention may differ.

### BIS speeches URL fix

Probe alternatives and switch to the working one:
- Current: `https://www.bis.org/cbspeeches/index.rss` (404)
- Candidates: `https://www.bis.org/list/cbspeeches/index.rss`, scrape `https://www.bis.org/cbspeeches/` HTML listing.

Resolved during implementation. If no working RSS, switch to HTML-list scraping with a small parser.

### Notebook 28 update

Update `tools/make_notebook_28_us_lui_text.py` to:
1. Include `iter_fed_decision` in the corpus assembly list (currently missing — it's only `iter_fed_minutes` + `iter_fed_speeches`).
2. Pass `fetch_body=True` to `iter_fed_speeches` so we get speech body text instead of RSS summaries.
3. Re-render the notebook from the builder.
4. Document the cache-invalidation trigger (`rm notebooks/data_cache/fed_corpus_28.parquet` after Slice 4 lands, then `PUREMACRO_REFETCH=1`).

## Components

### `_extractors.py` (new)
Single-responsibility module: HTML body extraction, with per-bank dispatch and a generic fallback. ~150 lines, including 2 per-bank precise extractors and the generic heuristic.

### Modified files

- `puremacro/narrative/sources/_rss_filtered.py` — add `fetch_body=` param.
- `puremacro/narrative/sources/fed_decision.py` — use `extract_body`.
- `puremacro/narrative/sources/fed_minutes.py` — URL transform + `extract_body`.
- `puremacro/narrative/sources/ecb_press_conf.py` — use `extract_body`.
- `puremacro/narrative/sources/bis_speeches.py` — fix URL.
- 24 RSS-based connector files — pass `fetch_body=` param through.
- `tools/make_notebook_28_us_lui_text.py` — include `iter_fed_decision` and pass `fetch_body=True`.
- Tests updated to match new behavior.

### New tests
- `tests/test_narrative_extractors.py` — unit tests for `extract_body` (Fed match, ECB match, generic fallback, malformed HTML, empty input).
- `tests/test_narrative_fed_url_transform.py` — unit tests for `_minutes_body_url`.

## Failure handling

| Failure | Behavior |
|---|---|
| `extract_body` finds no body container at all | Falls back to whole-page strip (current behavior) |
| Per-bank extractor regex misses | Falls back to generic |
| `fetch_body=True` link fetch fails | Falls back to RSS summary, prints `[skip body]` once per connector run |
| `fetch_body=True` returns very short body (< 200 chars) | Falls back to RSS summary |
| Fed minutes body URL 404s | Falls back to announcement URL |
| BIS URL still doesn't work after fix attempt | Connector returns empty (per RETRY_POLICY); document the issue and defer to a future iteration |

## Testing strategy

- `tests/test_narrative_extractors.py` — extractor unit tests on canned HTML fixtures (Fed sample, ECB sample, malformed, empty). 6-8 tests.
- `tests/test_narrative_fed_url_transform.py` — URL transformation tests. 3-4 tests.
- Existing connector tests (`tests/test_narrative_cb_connectors.py`, `tests/test_narrative_slice3_connectors.py`) updated where needed; `mock_http` continues to drive offline tests.
- Integration smoke: a manual notebook-28 re-run after Slice 4 commits should produce non-zero LUI/EPU/WUI with non-trivial cross-quarter variance.

## Branching and release

- Stay on `feature/narrative-extension-slice3` — current head; no need for a new branch since this slice continues the same research push.
- Version bump `0.7.0 → 0.7.1` (patch — bug fixes + extractor infra; no API breaks; new `fetch_body=` parameter is backward-compatible default-False).
- Tag `v0.7.1`.

## Out of scope (deferred)

- Length-normalized WUI per Ahir-Bloom-Furceri.
- Picault-Renault paragraph-level multinomial logit; full Hubert lexicon.
- `llm_prob_kernel` for LLM-backed scoring.
- Per-bank precise extractors for the 13 Slice-3 banks beyond the generic fallback. Add as research surfaces signal-quality issues for a specific bank.
- Lexicon expansion (LUI/EPU/WUI term lists). Re-evaluate AFTER Slice 4 produces a real corpus.
- Notebook 29 (state-panel LP-IV with national LUI as shock).
