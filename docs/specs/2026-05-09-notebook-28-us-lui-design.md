# Notebook 28 — US Labor-Market Uncertainty Index from Fed Text

**Status:** Drafted 2026-05-09. First research-level integration of the `puremacro.narrative.indices` subpackage shipped in v0.6.2 / v0.7.0.
**Driving lens:** "use what we built" — pressure-test Slices 1–3 against an actual research workflow before extending the foundation further.

## Motivation

Slice 1 (v0.6.1) shipped Fed connectors. Slice 2 (v0.6.2) shipped `lui()` (Labor-Market Uncertainty Index). Slice 3 (v0.7.0) closed deferrals. Together they let us build a US LUI quarterly series from a real Fed corpus in ~10 lines. Until now the indices have only been exercised on synthetic test corpora; we don't yet know whether the published Fed text + the shipped LUI lexicon yields a signal that lines up with established labor-uncertainty proxies (jobless claims, BBD-EPU, the national unemployment rate).

The active `feature/subnational-labor-uncertainty-us` research branch (notebooks 21–27) needs a credible US-national LUI series before it can be propagated to state-level work. Notebook 28 is the first step: build the index, plot it, validate it against external benchmarks. State-panel LP-IV using LUI as the shock is deliberately deferred to a future notebook 29.

## Non-goals

- **No** state-panel work. Notebook 28 is US-national only.
- **No** tone/hawkish-dovish axis. Just the uncertainty trio (LUI / EPU / WUI).
- **No** cross-country comparison. US-only — Slice 3 cross-lingual smokes already cover the multilingual axis.
- **No** publication-quality writeup. This is exploratory: confirm the signal exists and looks reasonable before investing in formal econometrics.
- **No** modification of the `puremacro.narrative.indices` package. If lexicon gaps surface during use, document them and defer fixes — don't widen scope into Slice 4.

## Architecture

Three new files; one new directory.

```
notebooks/
├── 28_us_lui_from_fed_text.ipynb           [new — rendered notebook]
└── data_cache/                              [new directory]
    └── fed_corpus_28.parquet                [new — corpus cache, written on first run]

tools/
└── make_notebook_28_us_lui_text.py         [new — paired builder per user convention]

tests/
└── test_notebook_28_smoke.py                [new — builder smoke test, no execution]
```

The builder ↔ notebook pairing follows the established convention (cf. `tools/make_notebook_27_bfs.py`). Per the user-memory rule on `feedback_notebook_builders_paired.md`, the live `.ipynb` and the builder are kept in lock-step.

### Notebook structure (6 sections)

#### §1 Setup

Bootstrap import via `notebooks._bootstrap.setup()`. Standard imports (`numpy`, `pandas`, `matplotlib`). Imports from `puremacro.narrative` (`lui`, `epu`, `wui`) and `puremacro.narrative.sources` (`iter_fed_minutes`, `iter_fed_speeches`). One markdown cell stating the question.

#### §2 Corpus assembly

```python
CACHE_PATH = ROOT / "notebooks" / "data_cache" / "fed_corpus_28.parquet"
REFETCH = os.getenv("PUREMACRO_REFETCH") == "1"

if CACHE_PATH.exists() and not REFETCH:
    corpus_df = pd.read_parquet(CACHE_PATH)
else:
    records = list(iter_fed_minutes()) + list(iter_fed_speeches())
    # Serialize the 4-tuple form: date, text, source_url, language
    corpus_df = pd.DataFrame([
        (date, text, url, meta.get("language", "en"), meta.get("doctype", ""))
        for date, text, url, meta in records
    ], columns=["date", "text", "source_url", "language", "doctype"])
    corpus_df = corpus_df.drop_duplicates("source_url").sort_values("date")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    corpus_df.to_parquet(CACHE_PATH, index=False)

# Reconstitute 4-tuples for the kernels
records = [
    (row.date, row.text, row.source_url,
     {"language": row.language, "doctype": row.doctype})
    for row in corpus_df.itertuples()
]
```

If `records` is empty, the cell prints a warning and writes an empty parquet. Subsequent cells short-circuit on `if not records:`.

#### §3 Compute indices

```python
ri_lui = lui(records, country="USA", language="en", normalize="zscore")
ri_epu = epu(records, country="USA", language="en", normalize="zscore")
ri_wui = wui(records, country="USA", language="en", normalize="zscore")
panel = pd.DataFrame({
    "lui": ri_lui.series, "epu": ri_epu.series, "wui": ri_wui.series,
})
```

#### §4 Time-series plot

Single `matplotlib` figure, 3-line plot of LUI/EPU/WUI z-scored, with shaded NBER recession bars (recessions hard-coded as a small dict in the notebook — there are ~10 since 1970, no need for a fancy data source). Title: "US labor-market & policy uncertainty from Fed text, {first}–{last}". Saved to `output_figures/28_lui_us_timeseries.pdf`.

#### §5 Validation correlations

Build a small DataFrame of comparator series:

```python
benchmarks = {}

# (a) BBD-EPU published series
try:
    bbd = puremacro.instruments.literature.bbd_epu.load()
    benchmarks["bbd_epu"] = bbd.series.resample("QS").mean()
except Exception as e:
    print(f"[skip] BBD-EPU mirror unreachable: {e}")

# (b) National unemployment rate aggregated from state panel (if available)
state_path = ROOT / "data" / "processed" / "state_panel_M.parquet"
if state_path.exists():
    sp = pd.read_parquet(state_path)
    benchmarks["urate_us"] = (
        sp.groupby("date")["urate_laus"].mean()
          .pipe(lambda s: s.set_axis(pd.to_datetime(s.index)))
          .resample("QS").mean()
    )

# Compute correlations on overlapping quarters
corrs = pd.DataFrame({
    name: panel.corrwith(b.reindex(panel.index)) for name, b in benchmarks.items()
})
```

Save table as `output_tables/28_lui_validation_corr.csv`. Print + display in notebook.

If `benchmarks` is empty (no BBD mirror, no state panel), print warning and skip — but the LUI itself is still saved from §6.

#### §6 Save outputs

```python
panel.to_parquet(ROOT / "notebooks" / "output_tables" / "28_lui_us_quarterly.parquet")
```

Plus a small JSON metadata sidecar with corpus size, language, normalization, computation timestamp. Useful for downstream notebook 29 (state-panel LP-IV).

## Components

### `tools/make_notebook_28_us_lui_text.py`

Standard `nbformat` builder following the pattern of `make_notebook_27_bfs.py`. Defines `build() -> NotebookNode` that returns the assembled notebook with the 6 sections above. Saved next to existing `make_notebook_27_bfs.py`. Module is import-safe and idempotent.

### `notebooks/data_cache/`

New directory for corpus caches. Listed in `.gitignore` ONLY if file size is a concern (a typical Fed-minutes+speeches parquet is < 5 MB and tracking it makes runs deterministic across machines). **Decision:** track in git for reproducibility; the file is small and rarely changes.

### `tests/test_notebook_28_smoke.py`

```python
def test_notebook_28_builder_produces_six_sections():
    from tools.make_notebook_28_us_lui_text import build
    nb = build()
    cells = nb["cells"]
    assert len(cells) >= 12  # 6 sections × ~2 cells each minimum
    headings = [c["source"].splitlines()[0]
                for c in cells if c["cell_type"] == "markdown"]
    assert any("§1" in h or "Setup" in h for h in headings)
    assert any("§3" in h or "Compute indices" in h for h in headings)
    assert any("§5" in h or "Validation" in h for h in headings)


def test_notebook_28_imports_resolve():
    """The notebook's imports must work in the current environment."""
    from puremacro.narrative import lui, epu, wui  # noqa: F401
    from puremacro.narrative.sources import (
        iter_fed_minutes, iter_fed_speeches,
    )  # noqa: F401
```

No live notebook execution — that requires network and is the user's responsibility to run interactively.

## Failure handling

| Failure | Behavior |
|---|---|
| Fed connectors return 0 records | §2 writes empty parquet, prints warning, §3–§6 short-circuit with "no corpus" markdown |
| BBD-EPU mirror unreachable | §5 logs the exception and proceeds with whatever benchmarks remain |
| `state_panel_M.parquet` missing | §5 skips that benchmark silently (the file is on the user's `feature/subnational-labor-uncertainty-us` branch and may not exist on every checkout) |
| Cache file exists but stale | User sets `PUREMACRO_REFETCH=1` to force re-fetch |
| Builder smoke-test fails | The builder didn't run cleanly — fix the builder, don't modify the rendered notebook directly |

## Testing strategy

- `tests/test_notebook_28_smoke.py` — two offline tests (builder structure, import resolution).
- The notebook itself is the integration test. Execution is interactive; not part of CI.
- Slice 1–3 connector and index tests cover the underlying primitives.

## Branching and release

- Stay on `feature/narrative-extension-slice3` (already at `v0.7.0`).
- No version bump — research notebook, not a package change.
- Single commit at end with the builder + tests + executed notebook + cache file (if small).

## Out of scope (future iterations)

- **Notebook 29:** state-panel LP-IV with LUI as the national shock and state employment / firm entry as outcomes. Requires the validated LUI from this notebook.
- **Notebook 30:** cross-country LUI panel (USA + LATAM + EU using Slice 3 connectors).
- **Lexicon expansion:** if §5 surfaces ρ < 0.3 vs. urate, that's evidence the LUI lexicon is too thin and warrants Slice-4-equivalent work in `_lexicons.py`. We document the finding; we don't fix it in this iteration.
- **Real-time vintages:** the corpus is point-in-time as of the run; no vintaged versions of the index series.
