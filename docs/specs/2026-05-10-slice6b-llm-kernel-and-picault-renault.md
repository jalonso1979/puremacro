# Slice 6b — LLM Kernel + Picault-Renault MNL (Brainstorm Spec)

**Status:** Drafted 2026-05-10. Surfaces the two remaining major Slice 6b items as design-only specs. Implementation deferred pending user decision on scope + cost.

## Item 1: `llm_prob_kernel` — LLM-backed paragraph-level scoring

### Motivation

Slices 5 + 6a built lexicon-based scoring (`keyword_count_kernel`, `cooccurrence_kernel`, `sentence_cooccurrence_kernel`). These work, but they're upper-bounded by what frozensets can express: they can't capture irony, qualifiers ("inflation pressures *remain anchored*"), or paragraph-level discourse structure ("Although hiring is strong, *the outlook is uncertain*").

An LLM-backed kernel would: send each paragraph to a small instruction-tuned model, ask for a structured probability of "this paragraph expresses [labor-market uncertainty / monetary policy uncertainty / hawkish stance / etc.]", and return a per-paragraph score in [0, 1]. Aggregate to doc → quarter.

### Key design questions

1. **Which API?**
   - Anthropic Claude (Haiku is fast + cheap for classification) — most reliable, requires API key.
   - OpenAI GPT-4o-mini — similar.
   - Local model via Ollama or llama.cpp — free + private, but heavier setup, slower.

2. **Per-paragraph or per-doc?**
   - Per-paragraph: ~50-200 paragraphs/doc × 366 docs ≈ 30K-70K API calls per re-run. At $0.25/1M input tokens, ~$10-30 per full corpus pass.
   - Per-doc: 1 call per doc → 366 calls per re-run, ~$1-3. Loses paragraph-level resolution.

3. **Pyodide compatibility?**
   - LLM calls require network + API client libraries → can't be pyodide-clean.
   - Solution: ship `llm_prob_kernel` as an **opt-in** module gated by a soft dependency. The base `puremacro.narrative.indices` stays pyodide-clean; LLM scoring lives in `puremacro.narrative.llm_kernel` with explicit `pip install puremacro[llm]` extras.

4. **Caching?**
   - Per-paragraph cache keyed on `(provider, model, prompt_hash, paragraph_hash)` → SQLite or parquet.
   - Re-runs hit cache; only new paragraphs incur cost.

5. **Output shape?**
   - Same 5-tuple as lexicon kernels: `(date, score, source_url, metadata)`.
   - Score = mean over paragraphs of LLM-returned P(category | paragraph).

### Components

| File | Purpose |
|---|---|
| `puremacro/narrative/llm_kernel/_client.py` | Provider abstraction (Anthropic / OpenAI / Ollama). |
| `puremacro/narrative/llm_kernel/_cache.py` | SQLite-backed paragraph→score cache. |
| `puremacro/narrative/llm_kernel/kernels.py` | `llm_prob_kernel(records, *, category, provider, model, ...)` — drop-in replacement for `sentence_cooccurrence_kernel`. |
| `pyproject.toml` | Add `[project.optional-dependencies] llm = [anthropic>=0.20, openai>=1.0]` (or just `httpx>=0.27` for a thin wrapper). |
| Tests | Mock the provider; verify caching + aggregation; one `@pytest.mark.network @pytest.mark.expensive` live smoke. |

### Validation target

LUI vs urate ρ ≥ +0.40 (currently +0.33). The LLM kernel should improve discrimination on the same Fed corpus.

### Cost ceiling

Configurable cost cap. Default: refuse to make > 5K API calls per single notebook run unless `PUREMACRO_LLM_BUDGET=spend` is set.

### Open questions for user

- Which provider to default to?
- Per-paragraph vs per-doc?
- Should `puremacro[llm]` extras be the only path, or also support a fully local Ollama variant?

---

## Item 2: Picault-Renault MNL — Paragraph-level multinomial logit (ECB-style)

### Motivation

Picault & Renault (2017) "MoneyTalks: An ECB BVAR-Compatible Communication Index" classifies each ECB-speech paragraph into one of K topical/stance categories (monetary stance: hawkish / neutral / dovish; or economic-activity outlook: improving / stable / deteriorating) using a multinomial logit on bag-of-words features. Their published coefficients form a "lexicon" with per-category weights per term — more granular than a frozenset.

This is a middle ground between (a) plain lexicon counts and (b) full LLM scoring. Coefficients are trained once on labeled ECB data; inference is a single matrix multiply per paragraph. No API calls; fully pyodide-clean.

### Key design questions

1. **Where do the coefficients come from?**
   - Reproduce Picault-Renault (2017) — train MNL on their published labeled dataset. Requires the labeled paragraphs (~5K paragraphs, hand-coded).
   - Use an alternative open dataset (Apel-Blix-Grimaldi has labeled hawkish/dovish; Hubert has labeled inflation-uncertainty).
   - Ship a small MNL trained on our own labeled subsample of FOMC minutes (requires labeling effort).

2. **Single MNL for all categories, or separate models?**
   - Joint: one MNL across `K` categories per paragraph → ensures probabilities sum to 1.
   - Separate: K binary logits → more flexible but less efficient.

3. **Feature engineering?**
   - Plain bag-of-words (token frequencies).
   - TF-IDF.
   - n-grams (bigrams capture "labor market", "interest rate").
   - Picault-Renault used unigrams; we could extend.

4. **Output integration?**
   - New kernel `mnl_kernel(records, *, model_path, language, category)` returning per-doc-aggregate score.
   - Or extend existing `tone_kernel` to a multinomial generalization.

### Components

| File | Purpose |
|---|---|
| `puremacro/narrative/mnl_kernel.py` | `mnl_kernel(records, *, weights, category)` — paragraph splitter + matrix inference. Pure numpy. |
| `puremacro/narrative/mnl_weights/picault_renault_2017.json` | Pre-trained coefficients (binary blob; sparse dict of {term → {category → weight}}). |
| Tests | Synthetic-weights forward-pass test; one fixture file. |

### Validation target

- Reproduce PR-2017's reported correlation with ECB shadow rate (their headline benchmark).
- Apply to Fed minutes; compare to existing tone_kernel + sentence_cooccurrence LUI.

### Open questions for user

- Acquire PR-2017's weights from the paper's replication archive, or train fresh on labeled FOMC data?
- Categories: mirror PR-2017 (5 categories) or simplify to 3 (hawkish/dovish/uncertain)?

---

## Implementation order recommendation

1. **Picault-Renault MNL first** — pyodide-clean, no API costs, ~3-5 day slice. Test if MNL beats lexicons on the existing corpus.
2. **LLM kernel second** — bigger design, real cost, ~5-10 day slice. Worth doing only if MNL shows headroom remains.

If MNL hits ρ ≥ +0.40 vs urate, LLM kernel may be unnecessary for the dissertation; defer.

## Out of scope (this spec)

- Full implementation. This document is design-only.
- Per-bank custom kernel variants.
- Training pipeline UI / hyperparameter sweep.
