# Narrative Extension — Slice 5 (LUI Lexicon + Fed Minutes Fix)

**Status:** Drafted 2026-05-09. Triggered by notebook 28's post-Slice-4 result: extraction works (EPU/WUI hit ρ ≥ 0.30 vs benchmarks) but LUI vs urate ρ = +0.18 — diagnosed as the 35-term lexicon being too sparse for real Fed text. Fed minutes per-record extraction also stays thin (~1000 chars/doc avg) because the URL transform shipped in Slice 4 doesn't handle pre-2014 minutes.

**Driving lens:** Lift LUI signal to research-usable strength so notebook 29 (state-panel LP-IV) can use it without becoming a study of measurement noise.

## Motivation

Slice 4 unblocked the foundation: 366-record corpus, EPU vs published BBD-EPU ρ = +0.32, EPU/WUI vs urate |ρ| ≈ 0.40. **But LUI vs urate is only +0.18.** Same corpus, same extraction — the only difference is the lexicon. The 35-term LUI lexicon misses the actual labor-uncertainty vocabulary in Fed minutes.

A complementary issue: the Fed minutes per-record text length averages ~1000 chars even after Slice 4. Live probes show modern minutes (post-2014) extract well at 48K chars/doc, but pre-2014 minutes use a different URL convention. The Slice-4 URL transform `monetary{date}a.htm → fomcminutes{date}.htm` 404s on those older items, falling back to the announcement page (which is mostly chrome). The right fix is to parse the announcement page itself for the actual `<a href="/fomc/minutes/…">` link, which works across all eras.

## Non-goals

- **No** BIS speeches HTML scrape. The `/cbspeeches/` page is JavaScript-rendered (returns 2 anchors total to static fetch). The right fix is a headless-browser path or a different BIS endpoint — both bigger than this slice's frame.
- **No** per-bank Slice-3 extractors. Generic fallback is good enough until research surfaces signal-quality issues for a specific bank.
- **No** Picault-Renault paragraph-level multinomial logit, full Hubert lexicon, length-normalized WUI, `llm_prob_kernel`. All Slice 6.
- **No** notebook 29 (state-panel LP-IV with national LUI as shock). Slice 7, depends on this slice unblocking LUI.
- **No** API changes. Lexicon expansion is data-only; minutes URL fix is purely internal to `iter_fed_minutes`.

## Architecture

### LUI lexicon expansion (`_lexicons.py`)

Each `_LUI_<lang>` constant is replaced with an expanded version organized around the 6 conceptual groups already documented in the spec:

1. **Layoffs** — layoff, redundancy, downsizing, dismissal, job cut, workforce reduction, RIF (reduction in force), termination, severance, mass layoff, …
2. **Hiring freeze** — hiring freeze, hiring pause, recruitment freeze, headcount freeze, suspended hiring, hiring slowdown, no new hires, attrition only, …
3. **Wage compression** — wage stagnation, real wage decline, wage softening, compressed pay, depressed wages, wage moderation, soft wage growth, weakening wage pressure, …
4. **Labor shortage** — skill shortage, talent shortage, tight labor market, labor scarcity, hiring difficulty, hard-to-fill, hiring bottleneck, worker shortage, …
5. **Participation drop** — labor force participation, discouraged workers, withdrawing from the workforce, dropping out, decline in participation, sidelined workers, …
6. **Unemployment risk** — unemployment, jobless claims, layoff risk, employment uncertainty, rising unemployment, weakening employment, softening labor market, deteriorating employment, …

Per-language target sizes (concept-density-aware):
- en: 150
- es, pt, de, fr, it: ≥ 100 each (Romance/Germanic languages have less synonym density than English in some groups)
- ja, zh: ≥ 60 each (concepts in those scripts often share a single canonical phrase per concept; bulking would require synthesizing rare metaphors)

Approximate aggregate: ~1,000 terms across 8 languages.

### Fed minutes URL via announcement-page parsing (`fed_minutes.py`)

Replace the regex URL transform `_minutes_body_url` with a 3-tier resolution:

```python
def _extract_minutes_body_link(announcement_html: str) -> str | None:
    """Find the first link to a minutes body inside the announcement page.

    Looks for <a href="/fomc/minutes/...html"> (older convention) or
    <a href="/monetarypolicy/fomcminutes...html"> (newer). Returns the
    href as found (caller prepends _BASE if relative).
    """

# In iter_fed_minutes:
#   1. Fetch announcement_url (always reliable; JSON `l` field).
#   2. Parse it for the body link. If found → fetch + extract.
#   3. If link not found OR body fetch fails OR body too short:
#      fall back to extract_body(announcement_html).
```

Net change: 1 HTTP call → 2 HTTP calls per minutes record (announcement + body), in exchange for body content that's ~10× longer on older minutes. Acceptable (the connector is already opt-in / cached at the notebook level).

The Slice 4 helper `_minutes_body_url(announcement_href)` is removed (no longer needed; the URL is now extracted from the announcement HTML rather than computed from the announcement URL string).

### Notebook 28 re-run

Same builder shape as Slice 4. The implementation:
1. `rm notebooks/data_cache/fed_corpus_28.parquet`.
2. `PUREMACRO_REFETCH=1 jupyter execute notebooks/28_us_lui_from_fed_text.ipynb …`.
3. Inspect `notebooks/output_tables/28_lui_validation_corr.csv`.

Acceptance criterion: **LUI vs urate ρ ≥ 0.30** (was +0.18). EPU/WUI baselines (ρ ≈ 0.32 / 0.38 vs BBD-EPU; |ρ| ≈ 0.40 vs urate) should hold roughly stable since extraction logic is unchanged.

### Release 0.7.2

Patch release (lexicon data + minutes-URL behavior fix; backward-compatible).
- `pyproject.toml` `0.7.1 → 0.7.2`.
- `puremacro/__init__.py` and `tests/test_import.py` matched.
- `CHANGELOG.md` entry.

## Components

| File | Change |
|---|---|
| `puremacro/narrative/indices/_lexicons.py` | Expand `_LUI_EN`, `_LUI_ES`, `_LUI_PT`, `_LUI_DE`, `_LUI_FR`, `_LUI_IT`, `_LUI_JA`, `_LUI_ZH` to target sizes. |
| `puremacro/narrative/sources/fed_minutes.py` | Add `_extract_minutes_body_link()`; rewrite the URL-resolution block in `iter_fed_minutes` to use 3-tier fallback. Remove the now-unused `_minutes_body_url()`. |
| `puremacro/tests/test_narrative_indices.py` | Update lexicon-coverage parametrize: bump `len ≥ 1` assertion to `len ≥ 60` for ja/zh and `len ≥ 100` for the 6 Latin-script langs. |
| `puremacro/tests/test_narrative_fed_url_transform.py` | Repurpose: instead of testing the regex URL transform (now removed), test `_extract_minutes_body_link` against canned announcement HTML fixtures (modern + 2006-era patterns). 4 tests. |
| `puremacro/tests/test_narrative_cb_connectors.py` | Update `test_fed_minutes_yields_four_tuple` mock: announcement URL HTML now includes `<a href="/monetarypolicy/fomcminutes20220316.htm">` so the parser finds the body link. |
| Notebook 28 cache + outputs | Re-run; commit `notebooks/output_tables/28_lui_*` and `notebooks/data_cache/fed_corpus_28.parquet` updates. |
| `pyproject.toml`, `puremacro/__init__.py`, `tests/test_import.py`, `CHANGELOG.md` | 0.7.1 → 0.7.2 + entry. |

## Failure handling

| Failure | Behavior |
|---|---|
| Lexicon expansion regresses LUI signal (ρ < 0.18 instead of better) | Acceptance miss — investigate and tune in a fix-up commit before tagging. |
| Announcement page has no body link (e.g., one-line release) | Fall back to extracting body from the announcement page itself. |
| Body-link URL fetch fails (404) | Fall back to announcement page. |
| Body fetched but extraction returns short text (< 5000 chars) | Fall back to announcement page. |
| ja / zh lexicon expansion can't reach 60 terms | Set target to 50 minimum and document the gap; defer to future expansion if real corpus density confirms it. |

## Testing strategy

- **Lexicon coverage** (existing parametrize, threshold raised) — confirms each language has ≥ N terms; cheap regression guard against lexicon shrinkage.
- **`_extract_minutes_body_link` unit tests** — 4 tests on canned announcement HTML for modern + 2006-era patterns.
- **Connector offline test** — existing `test_fed_minutes_yields_four_tuple` updated to include a body-link in the mock announcement HTML.
- **Live re-run of notebook 28** is the integration test (correlation acceptance criterion).

## Out of scope (deferred)

- BIS speeches connector — needs JS-rendered scraping work.
- Length-normalized WUI per Ahir-Bloom-Furceri.
- Picault-Renault paragraph-level multinomial logit; full Hubert lexicon.
- `llm_prob_kernel` for LLM-backed scoring.
- Per-bank precise extractors for Slice-3 banks.
- Notebook 29 (state-panel LP-IV with national LUI as shock).
