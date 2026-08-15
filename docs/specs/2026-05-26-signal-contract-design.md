# `puremacro.narrative` — Signal contract (S1 + S3)

**Status:** Drafted 2026-05-26. Architectural spec for the signal-quality + measurement-uncertainty extension to `RiskIndex`. Implementation in three slices.
**Target releases:** 0.65.0 (Slice 1 — schema + sparsity), 0.66.0 (Slice 2 — draws + propagation), 0.67.0 (Slice 3 — calibration).
**Driving lenses:** researcher-facing platform breadth; defensibility of narrative indices under referee scrutiny; strict backwards-compat with the 0.64.0 `RiskIndex` API.

## Motivation

After Track A closed (0.64.0), puremacro ships 13+ narrative indices spanning Fed Beige Book, US executive (ERP / SOTU / CBO), EU legislative (EUR-Lex / EP), Bluesky central-bank communication, plus the canonical EPU / GPR / WUI / MPU / Tone. Every index returns a `RiskIndex` with a single point series, a `method` tag, and a free-form `metadata` dict. There is no canonical way to:

1. **Quantify how reliable each period's reading is.** A researcher gets a number; they don't know whether that number rests on 50 documents or 3, whether scoring kernels would agree, or whether dropping one Beige Book district would have flipped it.
2. **Propagate measurement uncertainty into downstream LP / SVAR IRFs.** Standard practice is to treat the index as a deterministic regressor; the resulting bands reflect only data sampling uncertainty, not signal uncertainty.
3. **Validate against any external ground truth.** `narrative/validation/` has the building blocks (`external_benchmarks.py`, `stability.py`, `spec_curve.py`, `gap_filler.py`, `report.py`) but no canonical contract that ties them to `RiskIndex` outputs.

Three holes that get more acute as outside researchers start to use the package: they need to know what to trust, what to band, and what to defend in a referee report.

## Non-goals

- **No** breaking changes to the 0.64.0 `RiskIndex` API. Both new fields are optional with `None` defaults; every existing caller and notebook continues to work unchanged.
- **No** new connectors. F1 (source coverage expansion) is a sibling spec; this one builds on the existing 60+ connectors.
- **No** new estimators. The five propagation targets (`lp_hac`, `lp_iv`, `panel_lp_dk`, `cholesky_svar`, `proxy_svar`) gain three kwargs each; no behavioural change when those kwargs are unused.
- **No** interpretation layer (driver decomposition, exemplar quotes, saliency) — that is S2, a downstream spec. This one stops at "the reading + bands + scores," not "why."
- **No** cross-source synthesis beyond what `consensus_disagreement` already does — S4, also downstream.
- **No** automatic LLM scoring in the default kernel set for `with_draws="basic"`. Network-dependent kernels stay opt-in to preserve the Pyodide story.

## Architecture

### Module map (deltas only)

```
puremacro/narrative/
├── types.py
│     [extended] RiskIndex gains optional `quality`, `draws` fields
│     [new]      SignalQualityReport, BenchmarkScore, EventPanelScore, SurveyScore
├── indices/
│   ├── __init__.py         [extended] private _assemble_riskindex helper
│   ├── _draws_taxonomy.py  [new]      per-index source applicability + group keys
│   ├── draws.py            [new]      kernel_draws, lexicon_draws,
│   │                                  doc_bootstrap_draws, corpus_loo_draws
│   └── (all index files)   [extended] with_draws=, with_quality= kwargs
└── validation/
    └── calibration/        [new subpkg]
        ├── __init__.py     attach_calibration entry point
        ├── benchmarks.py   BENCHMARK_REGISTRY + score_vs_benchmark
        ├── event_panel.py  load + score_vs_event_panel
        ├── surveys.py      SURVEY_REGISTRY + score_vs_survey
        └── _data/
            ├── benchmark_registry.json
            ├── event_panel.json    (seed: USA/GBR/EUR/JPN, ~30-50 events each)
            └── survey_registry.json

puremacro/lp/
├── jorda.py        [extended] signal_draws/propagation/attribution kwargs on lp_hac
├── iv.py           [extended] same on lp_iv
└── panel_dk.py     [extended] same on panel_lp_dk

puremacro/var/identify/
├── cholesky.py     [extended] same on cholesky_svar
└── proxy.py        [extended] same on proxy_svar

puremacro/inference/_results.py
├── LPResult        [extended] optional irf_signal_*, signal_attribution, n_signal_*
└── (SVAR result types) likewise

puremacro/plot.py   [extended] irf_with_signal_bands convenience

docs/SIGNAL_CONTRACT.md     [new] one-page reference, linked from README

notebooks/R4_signal_contract/
├── R4_01_schema_demo.ipynb
├── R4_02_draws_and_irf_bands.ipynb
└── R4_03_calibration_table.ipynb
```

### Schema: `RiskIndex` extension

```python
@dataclass
class RiskIndex:
    name: str
    country: str
    series: pd.Series                                 # unchanged
    method: str
    corpus: str
    language: str
    normalization: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # NEW — both optional, backwards-compat
    quality: "SignalQualityReport | None" = None
    draws:   pd.DataFrame | None           = None
```

**`draws` layout**: row index = `series.index` (same dates). Column index = `pd.MultiIndex` with two levels `('source', 'draw_id')` where `source ∈ {'kernel', 'lexicon', 'doc', 'corpus'}`. A full-draws frame is `n_periods × (n_kernel + n_lex + n_doc + n_corpus)`. Validated in `__post_init__`.

**New convenience methods**:
- `.band(level=0.9, sources='all')` → DataFrame with `lower`, `upper` per period; `sources` filters which `level-0` keys to marginalise over.
- `.draws_attribution()` → per-period variance share by source tag.
- `.mean_draws()` → posterior mean across draws (sanity check vs. `.series`).

Existing `.diagnostics()`, `.to_frame()`, `.as_instrument()` unchanged. `.as_instrument()` propagates `draws` into the `Instrument.metadata['signal_draws']` slot so the LP-IV machinery can pick them up.

### Schema: `SignalQualityReport`

```python
@dataclass
class SignalQualityReport:
    # Sparsity / coverage (cheap, always computable)
    n_docs_per_period:        pd.Series          # date -> int
    avg_doc_length:           pd.Series          # date -> float (tokens)
    coverage_gaps:            list[pd.Period]    # dates with zero docs
    # Stability (derived from draws; None if no draws)
    kernel_agreement:         pd.Series | None
    multilingual_parity:      pd.Series | None
    doc_bootstrap_sd:         pd.Series | None
    corpus_loo_max_swing:     pd.Series | None
    # Calibration (per-layer opt-in)
    benchmark_scores:         dict[str, "BenchmarkScore"] = field(default_factory=dict)
    event_panel:              "EventPanelScore | None"    = None
    survey_scores:            dict[str, "SurveyScore"]    = field(default_factory=dict)
    metadata:                 dict[str, Any]              = field(default_factory=dict)
```

Supporting types (`narrative/validation/calibration/`, all frozen):

```python
@dataclass(frozen=True)
class BenchmarkScore:
    key: str
    rho: float
    rank_rho: float
    rmse: float
    overlap_start: pd.Timestamp
    overlap_end:   pd.Timestamp
    n_obs: int

@dataclass(frozen=True)
class EventPanelScore:
    country: str
    n_events: int
    rank_rho: float
    hit_rate_top_decile: float
    auc: float
    misses: list[pd.Timestamp]

@dataclass(frozen=True)
class SurveyScore:
    key: str
    rho: float
    rmse: float
    overlap_start: pd.Timestamp
    overlap_end:   pd.Timestamp
    n_obs: int
```

Helper: `SignalQualityReport.summary() -> pd.DataFrame` flattens to one row for cross-index tables. Per-index variation is handled by leaving inapplicable fields `None` (e.g. `multilingual_parity` is `None` for monolingual indices like `cboui`; `corpus_loo_max_swing` is `None` for sources without natural subgroups).

### Draws generation

`narrative/indices/draws.py` exports four generator functions:

```python
def kernel_draws(records, *, kernels: tuple[str, ...], base_kwargs: dict) -> pd.DataFrame: ...
def lexicon_draws(records, *, lexicon: dict, n_draws: int = 200,
                  frac_keep: float = 0.8, rng_seed: int | None = None,
                  base_kwargs: dict) -> pd.DataFrame: ...
def doc_bootstrap_draws(records, *, n_draws: int = 500,
                        rng_seed: int | None = None, base_kwargs: dict) -> pd.DataFrame: ...
def corpus_loo_draws(records, *, group_key: str, base_kwargs: dict) -> pd.DataFrame: ...
```

Each returns a per-period DataFrame whose columns are integer `draw_id`s; the assembler stamps the `source` MultiIndex level when constructing `RiskIndex.draws`.

`_draws_taxonomy.py` declares per-index which sources apply and what `group_key` `corpus_loo` uses, e.g.:

```python
DRAW_TAXONOMY: dict[str, dict] = {
    "bbui":       {"sources": ("kernel", "lexicon", "doc", "corpus"),
                   "corpus_group_key": "district"},
    "bluesky_ui": {"sources": ("kernel", "lexicon", "doc", "corpus"),
                   "corpus_group_key": "actor"},
    "eurlex_ui":  {"sources": ("kernel", "lexicon", "doc"),
                   "corpus_group_key": None},
    "gpr":        {"sources": ("kernel", "doc"),
                   "corpus_group_key": None},
    # ... one entry per name in narrative.indices.__all__
}
```

Auto-validated by a test asserting every name in `narrative.indices.__all__` has a taxonomy entry.

**Each index function gains a `with_draws=` kwarg**:
- `False` (default) — no draws. Same speed as today.
- `"basic"` — kernel + doc_bootstrap (cheap subset, in-Pyodide).
- `"full"` — all sources declared in `DRAW_TAXONOMY` for that index.
- list, e.g. `["kernel", "lexicon"]` — caller-specified subset.

Inapplicable sources are skipped with a warning rather than failing.

**Where the assembly happens**: a private helper `_assemble_riskindex(point_series, draws_dict, quality_dict, ...)` in `narrative/indices/__init__.py` glues the four per-source DataFrames into the MultiIndex-columned `RiskIndex.draws` and constructs the `SignalQualityReport`. Index functions call it instead of constructing `RiskIndex` directly, so future schema changes live in one place.

**Cost calibration** (orders of magnitude): `with_draws=False` is `1×`; `"basic"` is roughly minutes on a 13-index full panel; `"full"` adds another `2–3×`. The `"full"` mode is a "run-once, cache the `RiskIndex` to parquet" pattern, not an interactive default. `tools/build_index_parquets.py` becomes the natural home for full-draw materialisation.

**Pyodide compat**: `kernel_draws` with `"llm"` or `"embedding"` kernels triggers the existing lazy-import guards. The `"basic"` default drops both.

### Calibration

`narrative/validation/calibration/` provides three opt-in layers:

- **Layer 1 — benchmarks**. `BENCHMARK_REGISTRY: dict[str, dict[str, str]]` maps `{index_name: {country: benchmark_key}}` to existing `puremacro.fetch.*` calls (e.g. `epu/USA → fetch.epu.bbd_epu_usa`). `score_vs_benchmark(index, key=None)` returns a `BenchmarkScore` on the overlapping z-scored window. `None` key → look up by `index.name` + `index.country`.
- **Layer 2 — event panel**. `event_panel.json` ships ~30–50 dated events per country (USA / GBR / EUR / JPN seed) with HIGH / MED / LOW severity, e.g. `{"date": "2008-09-15", "severity": "HIGH", "kind": "financial", "note": "Lehman"}`. `score_vs_event_panel(index, country=None, panel_path=None)` returns an `EventPanelScore` with rank-correlation, top-decile hit rate, binary AUC, and a `misses` list. Researchers override the panel via `panel_path=` for paper-specific calibration.
- **Layer 3 — surveys**. `SURVEY_REGISTRY` mirrors the benchmark registry but maps to FRED / OECD / ECB survey series (Michigan Sentiment, SPF disagreement, ECB CB EPU survey, etc.). `score_vs_survey(index, key=None)` returns a `SurveyScore`.

**Public entry**:
```python
def attach_calibration(
    index: RiskIndex,
    *,
    layers: tuple[str, ...] = ("benchmark", "event_panel", "survey"),
    country: str | None = None,
    panel_path: str | Path | None = None,
) -> RiskIndex: ...
```
Mutates `index.quality.*` in place and returns the index. Each layer fails soft: missing benchmark fetch → `benchmark_scores[key] = None`; calibration never blocks the index from being returned.

**Maintenance**: registries are JSON-backed (one-line PRs to add coverage). `event_panel.json` is an editorial commitment — explicitly documented as "puremacro's opinionated default chronology," not "the truth." The `panel_path=` override keeps disagreement productive.

A test asserts every name in `narrative.indices.__all__` either has an entry in `BENCHMARK_REGISTRY` or is explicitly waived (cross-source indices).

**Pyodide compat**: `calibration/` is lazy-imported behind the network boundary. The bare `RiskIndex` schema with `quality=None` stays Pyodide-pure.

### Propagation in LP / SVAR

Five estimators gain three identical kwargs:

```python
def lp_hac(
    panel, y, x, horizons, n_lags=2,
    # ... existing kwargs ...
    signal_draws:       pd.DataFrame | None = None,
    signal_propagation: Literal["bootstrap", "rubin", "both"] = "bootstrap",
    signal_attribution: bool                                  = False,
) -> LPResult: ...
```

Identical surface on `lp_iv`, `panel_lp_dk`, `cholesky_svar`, `proxy_svar`.

**Three propagation modes**:
1. **`"bootstrap"`** (default when draws are present). Outer loop over `n_signal_draws`, inner residual bootstrap of size `n_boot`. Final `irf_signal_lower`/`upper` = outer quantiles across all `n_signal_draws × n_boot` stacked IRFs. Most general; expensive.
2. **`"rubin"`**. Per-draw HAC point + SE (no inner bootstrap). Combine via Rubin's rules: `var_total = mean(var_within) + (1 + 1/m) · var_between(point_irfs)`. Bands from `point ± z · sqrt(var_total)`. ~10–100× cheaper; assumes ~normal sampling distribution.
3. **`"both"`** runs both, returns both sets of bands. Diagnostic — documents whether Rubin is a good cheap proxy on this problem.

**Source attribution** (`signal_attribution=True`): reads the MultiIndex level-0 (`source`) of `signal_draws`. For each horizon, decomposes the across-draw variance via a one-way ANOVA partition: `Var_total(h) = Σ_s Var_s(h) + cross_terms`. Reported as `result.signal_attribution`: DataFrame indexed by horizon, columns = source tags + `cross`, values = variance shares summing to 1.

**Plumbing through `Instrument`**: `RiskIndex.as_instrument()` puts `draws` into `Instrument.metadata['signal_draws']`. The five estimators check there as a fallback when `signal_draws=None` but `x` carries them. The two call sites converge on the same code:
```python
res = lp_hac(panel, "y", x=ri.series, signal_draws=ri.draws)
# equivalent to
res = lp_hac(panel, "y", x=ri.as_instrument())
```

### Result-object extensions

`LPResult` and the SVAR result types gain (all optional, default `None`, backwards-compat):

```python
irf_signal_lower:        np.ndarray | None = None
irf_signal_upper:        np.ndarray | None = None
signal_attribution:      pd.DataFrame | None = None
n_signal_draws:          int | None         = None
n_signal_fail:           int | None         = None
signal_propagation:      str | None         = None
# When propagation="both":
irf_signal_lower_rubin:  np.ndarray | None = None
irf_signal_upper_rubin:  np.ndarray | None = None
```

Existing `irf_lower` / `irf_upper` (residual-only bands) unchanged. Researchers compare "data-only" vs. "data + signal" side-by-side.

`puremacro.plot.irf_with_signal_bands(result)` renders three nested bands (point, residual, residual+signal) using existing `plot.py` grayscale primitives. Pyodide-pure.

## Data flow

```
records (iter_X)
  → kernel(s) score per document
  → period aggregation
       ├── point estimate                                → RiskIndex.series
       ├── opt-in: draws (kernel / lexicon / doc / corpus) → RiskIndex.draws
       └── opt-in: SignalQualityReport
            ├── sparsity / coverage / parity   (auto)
            └── attach_calibration()
                 ├── benchmarks      (canonical published siblings)
                 ├── event_panel     (shipped + paper-specific JSON)
                 └── surveys         (FRED / OECD / ECB)

RiskIndex
  → ri.as_instrument()  (carries draws via Instrument.metadata['signal_draws'])
  → lp_hac / lp_iv / panel_lp_dk / cholesky_svar / proxy_svar
       ├── signal_propagation="bootstrap" | "rubin" | "both"
       └── signal_attribution=True/False
  → LPResult / SVARResult with
       ├── irf_lower/upper            (residual-only, existing)
       ├── irf_signal_lower/upper     (data + signal)
       └── signal_attribution         (per-source variance shares)
  → plot.irf_with_signal_bands(result)
```

## Failure semantics

- Inapplicable draws source (no district metadata, lexicon perturbation on non-lexicon kernel) → skip with warning, return other sources.
- Date alignment between draws and panel: inner join, warn on >5% loss.
- Per-draw estimation failure (singular `X'X`, BK violation): caught and counted in `n_signal_fail`. Raises if `n_signal_fail / n_signal_draws > 0.1`.
- `propagation="rubin"` with `n_signal_draws < 10`: warns about unreliable Rubin bands.
- Calibration fetch failure: that benchmark scored `None`; report still emits other layers.

## Pyodide contract

- `narrative/types.py` extensions: pure-Python dataclasses; Pyodide-pure.
- `narrative/indices/draws.py`: pure-numpy / pandas. LLM and embedding kernels lazy-imported behind existing guards.
- `narrative/validation/calibration/`: lazy-imported behind the network boundary. Subpackage import is allowed; `attach_calibration` only fires network calls when invoked.
- `lp/*`, `var/identify/*`, `plot.py` extensions: pure-numpy.

Extended `tests/test_pyodide_compat.py` asserts none of the above leak forbidden modules at import time.

## Testing

Headline tests (sibling test files alongside the modules they cover, plus one top-level `tests/test_signal_contract.py`):

1. **Schema round-trip** — `RiskIndex(..., quality=None, draws=None)` constructs and behaves identical to today on `to_frame()` / `as_instrument()` / `diagnostics()`. Per-index parity test.
2. **Draws shape** — each generator returns DataFrame with `series.index` rows and integer `draw_id` columns; no NaN propagation; negative `n_draws` rejected. `_assemble_riskindex` produces the expected `('source', 'draw_id')` MultiIndex.
3. **Convenience methods** — `.band(level=0.9)` matches `np.quantile` on draws; `.mean_draws() ≈ .series` on a known-noise fixture within tolerance; `.draws_attribution()` shares sum to 1.0.
4. **Taxonomy completeness** — auto-test asserts every name in `narrative.indices.__all__` has a `DRAW_TAXONOMY` entry.
5. **Calibration** — fixture-mocked benchmark fetch → expected `(rho, rmse)`; event-panel scoring on a synthetic spiking series scores high; survey scoring returns null-fields cleanly on no overlap.
6. **Propagation parity** — known-noise signal with closed-form measurement-error correction: `propagation="bootstrap"` and `"rubin"` match the analytic answer within tolerance; `"both"` returns matching pairs.
7. **Per-draw failure** — synthetic draws designed to make `X'X` singular every 10th draw; `n_signal_fail == n_signal_draws / 10`; estimation completes without raising. Raising kicks in once we cross the 10% threshold.
8. **Attribution sanity** — synthetic draws where 80% of variance is intentionally from one source; attribution recovers ≥ 0.7 share for that source.
9. **Pyodide guard** — extends `tests/test_pyodide_compat.py`: importing the new modules does not pull in `arch`, `statsmodels`, network deps, etc.
10. **Performance smoke** — `with_draws="basic"` on one small fixture in <30s on a single core; `with_draws="full"` in <5min. Guardrail, not a benchmark.

## Documentation deliverables

- `ARCHITECTURE.md` gains a short "Signal contract" subsection after the result-object standard.
- `README.md` gains a 6-line code block showing `with_draws=` + `lp_hac(signal_draws=)`.
- One new file: `docs/SIGNAL_CONTRACT.md` — single-page reference for the schema, the four uncertainty sources, the three propagation modes, and the three calibration layers. Linked from README.

## Staging

Each slice is shippable independently. Each slice's done-definition includes its notebook deliverable.

- **Slice 1 — Schema + sparsity diagnostics** → `0.65.0`.
  Extends `RiskIndex` with optional `quality` / `draws` (both `None`). Adds `SignalQualityReport` with only the sparsity / coverage fields populated. Adds `with_quality=` kwarg to all canonical indices (purely additive). Notebook `R4_01_schema_demo.ipynb`.

- **Slice 2 — Draws + propagation** → `0.66.0`.
  Adds `narrative/indices/draws.py` + `_draws_taxonomy.py`. Adds `with_draws=` to all canonical indices. Adds the three propagation kwargs to the five estimators. Extends `LPResult` / SVAR result types. Adds `plot.irf_with_signal_bands`. Notebook `R4_02_draws_and_irf_bands.ipynb`.

- **Slice 3 — Calibration** → `0.67.0`.
  Adds `narrative/validation/calibration/` with the three layers and the shipped `event_panel.json` seed (USA / GBR / EUR / JPN). Wires `attach_calibration` into `with_quality="full"`. Notebook `R4_03_calibration_table.ipynb`.

## Open follow-ups (out of scope for this spec)

- **S2** — interpretation layer (driver decomposition, exemplar quotes, saliency).
- **S4** — cross-source synthesis 2.0 (factor extraction, weighted ensembling, agreement-regime detection).
- Extending propagation to TVP-VAR, regime-switching VAR, and exotic SVAR identifications (sign restrictions, max-share, non-Gaussian, BQ, hetero, news-shock) — same kwarg surface, separate work.
- `event_panel.json` country expansion beyond USA / GBR / EUR / JPN.
- Adoption of `with_quality=` / `with_draws=` by the cross-source synthesis primitive (`consensus_disagreement`) — natural fit but waits for S4.
