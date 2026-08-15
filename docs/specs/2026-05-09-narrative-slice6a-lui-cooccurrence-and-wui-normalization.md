# Narrative Extension — Slice 6a (LUI Sentence Co-occurrence + WUI Length-Normalization + Hubert Vocabulary)

**Status:** Drafted 2026-05-09. Triggered by Slice 5's headline miss: lexicon expansion (35 → 145 EN terms) did not lift LUI vs urate (ρ stayed at +0.18). Diagnosed: term-frequency raw counts conflate labor-market *discussion* with labor-market *uncertainty*. Slice 6a fixes the scoring methodology; Slice 6b (next iteration) adds LLM-backed kernels and Picault-Renault-style multinomial logit.

**Driving lens:** Lift LUI to research-usable strength so notebook 29 (state-panel LP-IV) unblocks. Acceptance: ρ ≥ +0.30 vs urate.

## Motivation

Fed minutes are long, multi-topic documents that virtually always discuss labor markets. The Slice 5 LUI score (raw lexicon hits / doc length) reflects *whether labor came up*, not *how uncertain the labor outlook is*. EPU avoids this by requiring document-level co-occurrence of (Economy ∧ Policy ∧ Uncertainty) — the BBD methodology. That works for newspaper articles (short, topic-focused) but underperforms on Fed minutes for the same reason: every multi-page meeting record contains labor + uncertainty markers *somewhere*.

The fix is **sentence-level co-occurrence**: count the fraction of sentences that contain both a labor-domain term AND an uncertainty/risk-tone term. This generalizes BBD to long-document contexts and produces a score interpretable as "% of doc that's labor-uncertainty-flavored."

A complementary issue: WUI is currently a flat hit-count, not the Ahir-Bloom-Furceri "hits per 1000 words" specification. Slice 6a corrects that.

A third small piece: Hubert-style inflation-uncertainty vocabulary (e.g., "anchored expectations," "wage-price spiral," "deviation from target") is missing from WUI. Conservative additions broaden coverage without changing the methodology.

## Non-goals

- **No** `llm_prob_kernel` (LLM-backed scoring). Slice 6b — needs its own design (API choice, async, caching, pyodide carve-out, cost guardrails).
- **No** Picault-Renault paragraph-level multinomial logit. Slice 6b — different modeling track, mostly relevant to MPU/tone, not LUI.
- **No** notebook 29 (state-panel LP-IV with national LUI as shock). Unblocked by this slice but built in Slice 7+.
- **No** API additions to support back-testing the old vs new LUI side-by-side. Hard cutover. Slice 5 LUI was a week old; no published results depend on it.
- **No** changes to EPU/MPU/GPR/tone scoring — those work as designed.
- **No** length-normalization for indices other than WUI. EPU is already normalized via co-occurrence; LUI is becoming a per-sentence ratio (also length-aware).

## Architecture

### S6.1 Sentence-level co-occurrence LUI

Three lexicons per language replace the current flat `_LUI_<lang>`:

- `_LABOR_DOMAIN_<lang>` — broad labor-economics vocabulary (employment, jobs, wages, hiring, layoff, workforce, etc.). ~30-40 terms each.
- `_UNCERTAINTY_TONE_<lang>` — risk/uncertainty markers, polarity-neutral (uncertain, risk, downside, weak, declining, deteriorating, softening, slowing, etc.). ~25 terms each.
- `_LUI_PHRASES_<lang>` — Slice 5's curated phrases (rising unemployment, labor shortage, hiring freeze, etc.). High-precision pre-formed combos. ~145 EN, ≥100 ES/PT/DE/FR/IT, ≥60 JA/ZH (no change in size from Slice 5).

`LEXICONS["lui"][lang]` becomes a dict (matching EPU's nested structure):

```python
LEXICONS["lui"]["en"] = {
    "labor_domain": frozenset({"employment", "jobs", ...}),
    "uncertainty_tone": frozenset({"uncertain", "risk", ...}),
    "phrases": frozenset({"rising unemployment", "labor shortage", ...}),
}
```

A sentence "matches" if:
- (it contains ≥1 labor-domain AND ≥1 uncertainty-tone term), OR
- (it contains ≥1 phrase from `_LUI_PHRASES`)

The phrases path keeps Slice 5's lexicon work productive: high-precision phrases like "rising unemployment" or "labor shortage" hit on their own, no separate uncertainty-tone term required in the same sentence.

Doc score = matched_sentences / total_sentences ∈ [0, 1].

### New kernel: `sentence_cooccurrence_kernel`

In `puremacro/narrative/indices/_kernels.py`:

```python
def sentence_cooccurrence_kernel(records, *, term_groups, phrases=None, language="en"):
    """Yield (date, fraction-of-sentences-matching) per record.

    A sentence matches iff:
      (∀ group in term_groups: count_keywords(s, group) > 0)
      OR
      (phrases is not None AND count_keywords(s, phrases) > 0)

    Score is total_match / total_sentences ∈ [0, 1].
    Empty docs and docs with zero parseable sentences yield score = 0.0.
    """
```

### Sentence splitter

New helper `_split_sentences(text, language)` in `_kernels.py`:

- Latin scripts (en/es/pt/de/fr/it): regex `[.!?]+(?:\s+|$)` boundary, keeping non-empty stripped chunks.
- CJK (ja/zh): regex `[。！？]+`.

Trade-offs: a regex splitter mishandles abbreviations ("Mr.", "U.S.", "etc.") — sometimes splitting mid-sentence. For Fed minutes specifically, this produces an over-count of "sentences" but the ratio is unchanged in expectation. Acceptable for this slice; can be replaced with a stricter tokenizer later if needed.

### Updated `lui.py`

Replaces `keyword_count_kernel` with `sentence_cooccurrence_kernel`. The public API signature stays the same except `lexicon` parameter shape: dict-of-frozensets instead of frozenset.

```python
def lui(text_iter, *, country, language="en", lexicon=None, normalize="zscore",
        base_period=None, agg="mean") -> RiskIndex:
    lex = lexicon if lexicon is not None else LEXICONS["lui"][language]
    term_groups = [lex["labor_domain"], lex["uncertainty_tone"]]
    phrases = lex.get("phrases")

    def _kernel(records):
        return sentence_cooccurrence_kernel(
            records, term_groups=term_groups, phrases=phrases, language=language,
        )

    return index_to_quarterly(...)  # same as before
```

### S6.2 Length-normalized WUI

Add `length_normalize` parameter to `keyword_count_kernel`:

```python
def keyword_count_kernel(records, *, terms, language="en", length_normalize=False):
    """Yield (date, score). 
    
    If length_normalize=False: score = total keyword hits.
    If length_normalize=True: score = (hits / total_word_count) * 1000.
    """
```

`wui.py` passes `length_normalize=True`. Default stays `False` so other callers are unaffected.

`total_word_count` is `len(_TOKEN_RX.findall(text))`. For ja/zh substring-match scripts, fall back to `max(len(text) / 2, 1)` (rough character-pair approximation; refinement deferred).

### S6.3 Hubert-inspired vocabulary expansion

Conservative additions to `_WUI_EN` (~25-35 new terms drawn from peer-reviewed open economic-uncertainty literature, attribution in module docstring). Examples:

- "inflation expectations", "anchored expectations", "unanchored expectations"
- "deviation from target", "off target"
- "second-round effects", "second round effects"
- "wage-price spiral", "price spiral"
- "price stability concerns"
- "policy uncertainty", "policy unpredictability"
- "macroeconomic uncertainty"
- "downside risks", "upside risks"
- "fat tails", "tail risk", "tail risks"
- "stagflation", "stagflationary"
- "headwinds", "crosscurrents"
- "fragile recovery"

Proportional ports for ES/PT/DE/FR/IT (~15 each), JA/ZH (~10 each). Counts stay below the existing language thresholds — purely additive expansion.

## Components

| File | Change |
|---|---|
| `puremacro/narrative/indices/_kernels.py` | Add `_split_sentences(text, language)`; add `sentence_cooccurrence_kernel`; add `length_normalize` parameter to `keyword_count_kernel`. |
| `puremacro/narrative/indices/_lexicons.py` | Restructure `LEXICONS["lui"][lang]`: from `frozenset` to dict-of-frozensets with keys `labor_domain`, `uncertainty_tone`, `phrases`. Add 8 new `_LABOR_DOMAIN_<lang>` constants and 8 new `_UNCERTAINTY_TONE_<lang>` constants. Move existing `_LUI_<lang>` to `_LUI_PHRASES_<lang>` (rename only — content unchanged from Slice 5). Expand `_WUI_<lang>` with Hubert-inspired terms. |
| `puremacro/narrative/indices/lui.py` | Switch from `keyword_count_kernel` to `sentence_cooccurrence_kernel`; update lexicon-shape handling. |
| `puremacro/narrative/indices/wui.py` | Pass `length_normalize=True` to the kernel. Update docstring to remove the "Slice 3 backlog" comment. |
| `puremacro/tests/test_narrative_kernels.py` | Add tests for `_split_sentences` (per-language), `sentence_cooccurrence_kernel`, and length-normalized `keyword_count_kernel`. |
| `puremacro/tests/test_narrative_indices.py` | Update LUI tests for the dict-shape lexicon; verify WUI length-normalization; add Hubert-coverage check. |
| `puremacro/tests/test_narrative_lui_cooccurrence.py` | New file — focused integration tests on the new LUI methodology. |
| Notebook 28 | Re-run; outputs (`28_lui_us_quarterly.parquet`, `.meta.json`, `28_lui_validation_corr.csv`) updated. Validation: LUI vs urate ρ ≥ +0.30. |
| `pyproject.toml`, `puremacro/__init__.py`, `tests/test_import.py`, `CHANGELOG.md` | 0.7.2 → 0.8.0 (minor bump signals breaking change to LUI scoring). |

## Failure handling

| Failure | Behavior |
|---|---|
| LUI ρ stays at ~0.18 even with sentence co-occurrence | Methodology miss — investigate. Likely cause: labor-domain or uncertainty-tone lexicons too narrow/broad. Tune in fix-up commit before tagging 0.8.0. If still flat after one fix attempt, escalate (could indicate Fed minutes are not the right corpus). |
| LUI ρ goes NEGATIVE or near 0 | Lexicon polarity bug — the new uncertainty-tone lexicon may include terms that fire on positive labor news. Audit terms; remove false positives. |
| EPU/WUI regress | WUI length-norm should not regress correlation (just rescales). EPU is untouched by this slice. If either regresses, that's a bug in shared infrastructure (sentence splitter? token regex?) — investigate before tagging. |
| Sentence splitter under-splits (rare punctuation, no `.!?` in CJK doc) | Score = 0 for that doc. Acceptable; test via a unit test that verifies CJK input gets split. |
| Sentence splitter over-splits on abbreviations | Increases denominator; ratio unchanged in expectation. Document the limitation; accept. |

## Testing strategy

- **`_split_sentences` unit tests** — happy path per language, edge cases (empty, no punctuation, mixed scripts).
- **`sentence_cooccurrence_kernel` unit tests** — manual fixtures with known co-occurrence patterns. Verify: doc with all matching sentences scores 1.0; doc with no matches scores 0.0; doc with half scores 0.5; phrase shortcut works without sentence-level co-occurrence.
- **Length-normalized `keyword_count_kernel`** — verify doubling doc length halves the score (constant raw count).
- **Lexicon shape regression** — `LEXICONS["lui"][lang]` is dict with required keys; `LEXICONS["lui"]["en"]["phrases"]` matches Slice 5 content.
- **Live re-run of notebook 28** — integration test on real Fed corpus.
- **Existing parametrized coverage tests** — extend to the new lexicons (≥30 labor_domain terms per Latin lang, ≥25 per CJK; ≥20 uncertainty_tone per Latin, ≥15 per CJK).

## Out of scope (deferred to Slice 6b)

- LLM-backed scoring (`llm_prob_kernel`) — needs API choice, cost guardrails, async/caching.
- Picault-Renault paragraph-level multinomial logit — different modeling track.
- Stricter sentence tokenizer (handling abbreviations) — only relevant if regex splitter proves insufficient.
- Notebook 29 (state-panel LP-IV with national LUI as shock).
- Per-bank precise extractors for Slice-3 banks.
- BIS speeches connector (still JS-rendered).
