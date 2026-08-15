# `puremacro.narrative` — Multi-domain extension (CB sources, monetary/macropru/fx events, text-derived risk indices)

**Status:** Drafted 2026-05-08. Architectural spec for a layered extension; first-slice plan to follow.
**Target releases:** 0.6.0 (Slice 1 — types, scoring, first-wave CB sources), 0.6.1 (Slice 2 — indices), 0.7.0 (Slice 3 — polyglot CB expansion + macropru/fx/structural).
**Driving lenses:** research-infrastructure breadth (multi-domain narrative shocks, multi-language corpora) + composability with the existing `puremacro.instruments` registry.

## Motivation

`puremacro.narrative` today is a fiscal-IV pipeline. It cleanly separates sources, scoring, aggregation, and replication, and emits `NarrativeInstrument` objects that already plug into LP-IV and Proxy-SVAR. That architecture is the right shape — it just covers one domain.

Three holes in the current scope motivate this extension:

1. **Domain coverage.** `NarrativeEvent.target ∈ {investment, consumption, both}` is fiscal-only. Monetary, macroprudential, FX-intervention, and structural-reform announcements all have published narrative datasets and active research literatures, but no home in the package.
2. **Source breadth.** The existing connectors are heavily ministry-side. Central-bank communications — decisions, minutes, press conferences, speeches, financial-stability reports — are largely absent (only ECB press is wired up, and only as an RSS sniff). Cross-country research increasingly needs CB text.
3. **Continuous indices.** `puremacro.instruments.literature` already mirrors the *published* BBD-EPU and Caldara-Iacoviello GPR series, but the package can't *generate* such indices from arbitrary corpora — which blocks novel applications (e.g. a labor-market uncertainty index from CB minutes) and limits cross-country / cross-language coverage.

The active `feature/subnational-labor-uncertainty-us` branch is one direct beneficiary: it currently has no in-house way to construct a text-derived labor-uncertainty series.

## Non-goals

- **No** breaking changes to existing fiscal API. Every current import (`from puremacro.narrative import NarrativeEvent, NarrativeInstrument, events_to_quarterly, ...` and the 11 replication loaders) keeps working unchanged.
- **No** translation layer. Multilingual processing is LLM-first (the model handles language natively); we do not bundle a translation backend.
- **No** novel ML pipeline beyond what `scoring/llm.py` already does. New scoring still goes through Anthropic / OpenAI urllib backends.
- **No** revival of `JKResult`-style high-frequency identification work — that lives in `puremacro.hfi` and stays separate. The new monetary events here are *narrative*, not HF-surprise.
- **No** automatic publication-quality replication of every canonical risk index. We replicate enough to validate (BBD-EPU US, Caldara-Iacoviello GPR US) and ship the lexicons; users wanting exact published series should keep using the `instruments.literature` mirrors.

## Architecture

### Module map (deltas only — existing files unchanged unless marked)

```
puremacro/narrative/
├── types.py                    [extended] NarrativeEvent (+kind, +language, per-kind validation)
│                               [new]      RiskIndex dataclass
├── aggregate.py                [extended] events_to_quarterly gains kind_filter
│                               [new]      index_to_quarterly
├── sources/
│   ├── _ratedoc.py             [new]      Shared decision/minutes parser scaffold
│   ├── _speeches.py            [new]      Shared speech-archive parser scaffold
│   ├── _fsr.py                 [new]      Shared financial-stability-report parser
│   ├── ecb_press.py            [renamed] → ecb_decision.py (re-export shim preserves old import path)
│   ├── ecb_minutes.py          [new]
│   ├── ecb_press_conf.py       [new]
│   ├── ecb_speeches.py         [new]
│   ├── ecb_fsr.py              [new]
│   ├── fed_decision.py         [new]
│   ├── fed_minutes.py          [new]
│   ├── fed_press_conf.py       [new]
│   ├── fed_speeches.py         [new]
│   ├── boe_*.py, boj_*.py,     [new]
│   │   boc_*.py, snb_*.py
│   ├── rba_*.py, rbnz_*.py,    [new]    (Slice 3)
│   │   riksbank_*.py, norges_*.py, sarb_*.py
│   ├── banxico_*.py, bcb_*.py, [new]    (Slice 3 — Spanish/Portuguese)
│   │   bccl_*.py, bcra_*.py, banrep_*.py
│   ├── pboc_*.py, rbi_*.py,    [new]    (Slice 3 — Asia EM)
│   │   bok_*.py, mas_*.py, bot_*.py
│   └── bis_speeches.py         [new]    (Slice 3 — meta-connector)
├── scoring/
│   ├── llm.py                  [extended] kind-parameterized prompts, multilingual preamble
│   ├── keyword.py              [extended] adds monetary lexicon (English)
│   └── manual.py               [unchanged]
├── indices/                    [new subpackage]
│   ├── __init__.py
│   ├── _kernels.py             Rolling document-count kernel; LLM-prob kernel; agg rules
│   ├── _lexicons.py            Multilingual term lists (en/es/pt/de/fr/it/ja/zh)
│   ├── epu.py                  Baker-Bloom-Davis style on arbitrary corpora
│   ├── mpu.py                  Husted-Rogers-Sun style monetary-policy uncertainty
│   ├── gpr.py                  Caldara-Iacoviello style geopolitical risk
│   ├── tone.py                 Hawkish-dovish (Apel-Blix-Grimaldi, Picault-Renault, Hubert)
│   ├── wui.py                  Ahir-Bloom-Furceri World Uncertainty Index style
│   └── lui.py                  Labor-market uncertainty (NEW — bespoke, multilingual)
├── quality/                    [extended] magnitude_harmonizer adds per-kind rules
└── __init__.py                 [extended] re-export new public surface
```

### Conceptual stack

```
Sources (text)  ─┬─►  Scoring (events)   ─►  events_to_quarterly  ─►  NarrativeInstrument
                 │                                                   │
                 └─►  Indices (counts/probs) ─► index_to_quarterly ─► RiskIndex
                                                                     │
                                       both wrap into  ◄─────────────┘
                                       puremacro.instruments.Instrument registry
```

The two pipelines (event-based and index-based) share the source layer and the LLM backend. They diverge at extraction: scoring emits discrete `NarrativeEvent` lists; indices emit per-document scores that are aggregated into a continuous `RiskIndex` series.

## Public API

### `NarrativeEvent` — extended

```python
@dataclass
class NarrativeEvent:
    # ---- existing fields ----
    date: pd.Timestamp
    country: str
    magnitude: float
    magnitude_unit: str
    target: str
    subtarget: str | None
    sign: int
    confidence: float
    source_text: str
    source_url: str
    scoring_method: str
    metadata: dict[str, Any] = field(default_factory=dict)
    implementation_profile: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    # ---- new fields ----
    kind: str = "fiscal"           # fiscal | monetary | macropru | fx | structural
    language: str = "en"           # ISO-639-1
```

`__post_init__` adds two new validations:

- `kind` in `VALID_KINDS = {"fiscal","monetary","macropru","fx","structural"}`.
- `target` in `VALID_TARGETS_BY_KIND[kind]`. Per-kind enums:

| `kind` | valid `target` | `sign` semantics | typical `magnitude_unit` |
|---|---|---|---|
| `fiscal` (existing) | `investment` / `consumption` / `both` | `+1` expansionary, `-1` contractionary | `USD_bn`, `pct_gdp`, `z` |
| `monetary` | `policy_rate` / `asset_purchase` / `forward_guidance` / `fx_intervention` / `lending_facility` | `+1` hawkish (tightening), `-1` dovish | `bps`, `bn_local`, `pct_assets` |
| `macropru` | `capital_buffer` / `ltv_dsti` / `sector_limit` / `reserve_requirement` | `+1` tightening, `-1` loosening | `pct_capital`, `pct_RWA`, `ratio` |
| `fx` | `intervention` / `peg_change` | `+1` defending (buying), `-1` selling | `USD_bn` |
| `structural` | `labor` / `product_market` / `trade` / `tax_admin` | `+1` liberalizing, `-1` restrictive | `z`, `pct_gdp` |

Defaults (`kind="fiscal"`, `language="en"`) ensure every existing call site keeps working with no source change. `magnitude_unit` is not enum-validated (current behavior preserved); harmonization to a per-kind common unit is the job of `quality/magnitude_harmonizer.py`.

### `RiskIndex` — new

```python
@dataclass
class RiskIndex:
    name: str                    # "epu_us_news" / "mpu_ecb_speeches" / "lui_mex_banxico"
    country: str                 # ISO3
    series: pd.Series            # quarterly, qdate-indexed
    method: str                  # keyword_count | llm_prob | tone_dispersion | hybrid
    corpus: str                  # "fed_speeches" / "ecb_press" / etc.
    language: str                # primary language of the corpus
    normalization: str           # raw | zscore | bbd_100
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_instrument(self) -> "Instrument": ...   # adapter into puremacro.instruments
    def diagnostics(self) -> dict: ...             # n_quarters, mean, std, gaps, n_docs
    def to_frame(self) -> pd.DataFrame: ...        # tidy: qdate, value, country, name
```

### Source layer — `SourceRecord` (4-tuple)

New connectors yield `(date, text, source_url, metadata)`. `metadata` is a dict with required keys `doctype` and `language` and optional `bank_code`, `country`, `speaker`, `doc_id`, `embargo_status`. Existing 3-tuple connectors are upgraded by a one-line shim at the scoring/index boundary that fills `metadata={"doctype":"press","language":"en"}`. **Every existing connector keeps its current public signature.**

Document taxonomy (the union; per-bank capability is a matrix):

| `doctype` | Cadence | Content | Use |
|---|---|---|---|
| `decision` | per meeting | Rate decision + statement | Discrete monetary events |
| `minutes` | per meeting (lag) | Account of debate, dissent | Tone variance / dispersion |
| `press_conf` | per meeting (Fed/ECB/BoE/BoJ/some EM) | Q&A transcript | Off-script signal |
| `speech` | weekly–monthly | Individual board-member views | Largest tone corpus |
| `fsr` | semi-annual | Financial-stability report | Macropru events; financial-risk index |

Per-bank coverage in this spec is the union shown in the module map; not every bank publishes every doctype, and missing combinations raise `NotImplementedError` with a clear message rather than yielding empty.

### Scoring — `score_llm` extended

```python
score_llm(text_iter, *, backend, kind="fiscal", language="en",
          country="USA", dry_run=False) -> list[NarrativeEvent]
```

Internally dispatches to one of five prompt templates (`_PROMPTS[kind]`), each with its own JSON schema. A shared preamble adds the language hint:

> *"The text below is in `{language}`. Extract all events regardless of language. Return field labels in English."*

Per-kind prompt summaries:

- **fiscal** (existing): unchanged, kept verbatim.
- **monetary** (new): magnitude in bps, target ∈ {policy_rate, asset_purchase, forward_guidance, fx_intervention, lending_facility}, sign ∈ {-1 dovish, +1 hawkish, 0 neutral}, effective_date, hawkish-dovish probability.
- **macropru** (new): target ∈ {capital_buffer, ltv_dsti, sector_limit, reserve_requirement}, magnitude in pct or ratio, sign tightening/loosening, effective_date.
- **fx** (new): side (buy/sell), magnitude in USD bn (counterparty side), effective_date.
- **structural** (new): target ∈ {labor, product_market, trade, tax_admin}, qualitative magnitude (z-score), sign.

Each prompt instructs the model to drop ambiguous, retrospective, or forecast-only items — same discipline as the existing fiscal prompt.

### Scoring — `score_keyword` extended

```python
score_keyword(text_iter, *, kind="fiscal", country="USA") -> list[NarrativeEvent]
```

Adds English `MONETARY_HAWKISH` / `MONETARY_DOVISH` lexicons. Kind dispatch picks which family to apply. Non-English keyword scoring is **not supported** — multilingual users are pushed to LLM. This keeps the keyword path Pyodide-clean.

### Indices — common API

Every index returns a `RiskIndex`:

```python
RiskIndex = epu(text_iter, *, country, language="en", lexicon=None,
                normalize="bbd_100", base_period=("2010-01","2015-12"))
RiskIndex = mpu(text_iter, *, country, language="en", lexicon=None, normalize="zscore")
RiskIndex = gpr(text_iter, *, country, language="en")
RiskIndex = tone(text_iter, *, country, language="en", method="apel_blix_grimaldi")
RiskIndex = wui(text_iter, *, country, language="en")
RiskIndex = lui(text_iter, *, country, language="en", lexicon=None, normalize="zscore")
```

`text_iter` is the same `(date, text, source_url, metadata)` stream the scoring layer consumes. Lexicons live in `_lexicons.py` as plain Python dicts, no external download.

`tone(method=...)` selects:
- `"apel_blix_grimaldi"` (token-count of hawkish vs. dovish lexicon, normalized).
- `"picault_renault"` (per-paragraph classification, multinomial logit-style with French ECB lexicon).
- `"hubert"` (net hawkishness with Hubert's dictionary).
- `"llm"` (each paragraph scored by the LLM backend; mean per quarter).

### Aggregation — `events_to_quarterly` and `index_to_quarterly`

`events_to_quarterly` gains a `kind_filter` keyword (default `None`). Mixed-kind event lists raise unless filtered. Aggregation rule per kind:

- `fiscal`, `monetary`: sum signed magnitudes (existing rule for fiscal).
- `macropru`, `fx`: count net (signed) actions in quarter (binary-style — magnitudes unit-incompatible across actions).
- `structural`: indicator (presence/absence).

`index_to_quarterly`:

```python
index_to_quarterly(records, *, kernel, country, language="en",
                   freq="QS", agg="mean") -> RiskIndex
```

`kernel` is a callable from `indices/_kernels.py`. `agg ∈ {"mean","max","dispersion"}`. `agg="dispersion"` returns within-quarter standard deviation — useful for tone uncertainty.

### Integration with `puremacro.instruments`

`RiskIndex.as_instrument()` wraps into the existing registry:

```python
Instrument(
    series=self.series,
    name=self.name,
    source=f"narrative.indices.{method}",
    category="text_index",
    frequency="Q",
    metadata={"corpus": ..., "language": ..., "method": ...,
              "normalization": ..., "n_docs": ...}
)
```

`NarrativeInstrument.as_instrument()` (existing) is updated to read `kind` off the events:

- fiscal → `narrative_fiscal_iv`
- monetary → `narrative_monetary_iv`
- macropru → `narrative_macropru_iv`
- mixed → `narrative_mixed_iv`

This puts text-derived indices and new event-based IVs into the same catalog as `bbd_epu`, `caldara_iacoviello_gpr`, `romer_romer_2004`, etc., so downstream LP/SVAR consumers don't care how a series was produced.

### Public API — `puremacro.narrative.__init__`

```python
from .types import NarrativeEvent, NarrativeInstrument, RiskIndex
from .aggregate import events_to_quarterly, index_to_quarterly
from .indices import epu, mpu, gpr, tone, wui, lui
# all existing replication loaders + helpers keep their imports
```

## Slicing — three implementation plans

### Slice 1 (target 0.6.0) — Foundation

- `NarrativeEvent`: add `kind` and `language` fields with per-kind target validation, defaults preserve fiscal API.
- `RiskIndex` dataclass + `index_to_quarterly`.
- `score_llm`: kind-parameterized prompts, ship `fiscal` (existing) + `monetary`.
- `score_keyword`: monetary lexicon (English).
- 4-tuple `SourceRecord` with backward-compat shim.
- First-wave CB connectors: **Fed** (decision/minutes/press_conf/speeches), **ECB** (rename `ecb_press` → `ecb_decision` + minutes/press_conf/speeches), **BoE** (decision/minutes/speeches), **BoJ** (statement/speeches).
- Tests: extend `test_narrative.py` for kind validation + monetary keyword + per-kind `events_to_quarterly`; add offline fixtures for the 4 new banks; extend `test_pyodide_compat.py`.

### Slice 2 (target 0.6.1) — Indices layer

- `indices/_kernels.py`, `indices/_lexicons.py`.
- `epu.py`, `mpu.py`, `gpr.py`, `tone.py`, `wui.py`, `lui.py`.
- Validation tests against `instruments.literature.bbd_epu` (US news subset, ρ ≥ 0.85, 1985–present) and `caldara_iacoviello_gpr` (US, ρ ≥ 0.85).
- `tests/test_narrative_indices.py` (new file).
- Example: `narrative_indices_demo.py` building EPU+MPU+GPR+LUI from existing connectors and validating against the published mirrors.

### Slice 3 (target 0.7.0) — Polyglot expansion

- LATAM CBs: Banxico (es), BCB (pt), BCCh (es), BCRA (es), BanRep (es).
- Advanced non-G7: RBA, RBNZ, Riksbank, Norges, SARB.
- Asia EM: PBoC (zh), RBI (en), BoK (ko), MAS, BoT.
- BIS speeches meta-connector.
- macropru + fx + structural prompt families.
- Cross-lingual lexicon validation: same banks, two languages, indices should correlate ρ ≥ 0.7 on overlapping window.
- Per-bank smoke tests.

## Error handling

- Every new connector honors `narrative/sources/RETRY_POLICY.md`: yield-don't-raise, 30 s default timeout, no exponential backoff, one SSL fallback. Connectors hitting WAFs (Fed, BoJ) pass `user_agent=` overrides.
- `score_llm` keeps the existing drop-malformed counter pattern; per-kind validation rejects events with wrong target/sign before construction (no silent fall-through to `kind="fiscal"`).
- `events_to_quarterly` raises `ValueError("events_to_quarterly: events have multiple kinds [fiscal, monetary]; pass kind_filter= to disambiguate")` when called on mixed-kind events without `kind_filter`.
- Index kernels: empty corpus → `ValueError("indices.<name>: no documents in <corpus_label>")`. Never silently emit NaN.
- Network-marked tests `pytest.skip()` when the fetcher returns empty (per the project's network-tests convention).

## Pyodide-compatibility (per `ARCHITECTURE.md` stability tiers)

| Module | Tier | Notes |
|---|---|---|
| `narrative/types.py` (extended) | **Stable / Pyodide-clean** | Pure dataclass + dict validation |
| `narrative/aggregate.py` | **Stable / Pyodide-clean** | Pure pandas |
| `narrative/scoring/keyword.py` | **Stable / Pyodide-clean** | New monetary lexicon, no top-level deps |
| `narrative/scoring/llm.py` | **Experimental** | Already lazy-imports |
| `narrative/indices/{epu,mpu,gpr,tone,wui,lui,_kernels,_lexicons}` (count path) | **Stable / Pyodide-clean** | Pure Python + pandas |
| `narrative/indices/_kernels.llm_prob_kernel` | **Experimental** | Reuses `scoring/llm` backends |
| `narrative/sources/<bank>_*.py` | **Experimental** | Network-bound, same tier as existing fiscal connectors |

`tests/test_pyodide_compat.py` is extended to walk the new `indices/` subpackage and confirm no `statsmodels` / `linearmodels` / `arch` leak into `sys.modules`.

## Testing strategy

- `tests/test_narrative.py` — extended for kind/language validation, kind_filter aggregation, monetary keyword scoring.
- `tests/test_narrative_indices.py` — new file. EPU vs. published correlation, GPR vs. published correlation, LUI smoke test on a synthetic corpus, normalization round-trip, lexicon-language coverage.
- `tests/test_narrative_offline.py` — extended HTTP fixtures for the new CB connectors (Fed press release page, ECB minutes JSON, BIS speech archive RSS).
- `tests/test_pyodide_compat.py` — confirms the new subpackage stays clean.
- `tests/_http_fixtures.py` — adds CB-shaped fixtures (decision page HTML, minutes PDF stub, speech archive RSS, FSR PDF stub).

Network-marked tests `pytest.skip()` when fetchers return empty, never `assert`. Offline-deterministic tests (kind validation, aggregation, lexicon-based index counts on a fixed synthetic corpus) are the bulk of the suite; live HTTP smoke tests are a thin top layer.

## Backward compatibility

- All existing imports keep working: `from puremacro.narrative import NarrativeEvent, NarrativeInstrument, events_to_quarterly`, all 11 replication loaders, all 20+ existing connectors.
- `ecb_press.py` is renamed to `ecb_decision.py`; `ecb_press.py` becomes a re-export shim that emits a `DeprecationWarning` once per session.
- `NarrativeEvent(...)` calls without `kind=`/`language=` continue to construct fiscal-English events.
- Existing `events_to_quarterly` calls without `kind_filter=` continue to work unchanged when events are all `kind="fiscal"` (the default).

## Out of scope (explicit)

- **Translation backends.** LLM handles non-English natively; no DeepL / GoogleMT integration.
- **High-frequency identification of monetary shocks** — that's `puremacro.hfi`'s job.
- **Replicating every published risk index exactly.** We replicate enough for validation; for exact published series, users should keep using `instruments.literature.{bbd_epu, caldara_iacoviello_gpr}`.
- **Concurrent fetch.** Connectors stay synchronous, per existing `RETRY_POLICY.md`. Parallelism is an opt-in helper module if the need arises.
- **On-disk caching of CB corpora.** Each connector decides whether caching makes sense, per existing convention.
- **Persistent embedding store / vector search over CB text.** Out of scope for this iteration.
