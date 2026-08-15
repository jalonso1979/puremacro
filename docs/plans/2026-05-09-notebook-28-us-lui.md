# Notebook 28 — US LUI from Fed Text — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `tools/make_notebook_28_us_lui_text.py` (paired builder), `notebooks/28_us_lui_from_fed_text.ipynb` (rendered output), and `tests/test_notebook_28_smoke.py` (offline smoke tests). The notebook builds a US Labor-Market Uncertainty index from Fed text using the indices subpackage shipped in v0.6.2 / v0.7.0.

**Architecture:** Builder follows the pattern of `tools/make_notebook_27_bfs.py` — a `build() -> NotebookNode` function that assembles markdown + code cells into 6 sections (setup, corpus assembly with on-disk cache, compute indices, time-series plot, validation correlations, save outputs). Two offline smoke tests confirm the builder structure and import resolution. Notebook execution itself is the user's interactive responsibility — out of scope for this plan.

**Tech Stack:** Python 3.10+, `nbformat`, `pandas`, `numpy`, `matplotlib`. Uses `puremacro.narrative.indices` (`lui`, `epu`, `wui`) and `puremacro.narrative.sources` (`iter_fed_minutes`, `iter_fed_speeches`).

**Spec reference:** `docs/specs/2026-05-09-notebook-28-us-lui-design.md`.

**Branching:** Stay on `feature/narrative-extension-slice3` (already at `v0.7.0`). No version bump — this is a research notebook, not a package change.

**Pre-implementation baseline:** `pytest -q` after Slice 3 = **956 passed, 27 skipped**. Plus 1 pre-existing pyodide-compat failure (statsmodels.tsa.x13 leak). Out of scope.

**Repo layout note:** This repo's git toplevel is `uncertainty_examples/` (parent), not `puremacro/`. All paths below are **relative to the git toplevel**:
- `tools/` lives at the toplevel — that's where `make_notebook_27_bfs.py` already is.
- `notebooks/` lives at the toplevel — alongside `_bootstrap.py` and the existing 21–27.
- `tests/` (parent-level) is where notebook smoke tests go (not `puremacro/tests/`).
- `data/processed/state_panel_M.parquet` is at the toplevel.
- The `puremacro/` subdirectory holds the package source + its own tests + its own docs.

Run `git rev-parse --show-toplevel` and confirm it is `uncertainty_examples/` before committing.

---

## File Structure

### Files created
- `tools/make_notebook_28_us_lui_text.py` — paired builder.
- `notebooks/28_us_lui_from_fed_text.ipynb` — rendered notebook (built from the builder).
- `notebooks/data_cache/.gitkeep` — directory marker for the corpus cache.
- `tests/test_notebook_28_smoke.py` — offline smoke tests.

### Files modified
- None. (No `puremacro` package changes; the notebook just consumes the existing public API.)

---

## Task 0: Verify branch + baseline

**Files:** none.

- [ ] **Step 1: Verify branch**

Run: `git branch --show-current`
Expected: `feature/narrative-extension-slice3`. If wrong, switch back.

- [ ] **Step 2: Confirm baseline**

Run: `pytest -q --no-header 2>&1 | tail -3`
Expected: `956 passed, 27 skipped, …`.

- [ ] **Step 3: Confirm Slice-3 connectors + indices import**

Run:
```bash
python -c "
from puremacro.narrative import lui, epu, wui
from puremacro.narrative.sources import iter_fed_minutes, iter_fed_speeches
print('imports ok')
"
```
Expected: `imports ok`.

---

## Task 1: Builder + smoke tests

**Files:**
- Create: `tools/make_notebook_28_us_lui_text.py`
- Create: `tests/test_notebook_28_smoke.py`
- Create: `notebooks/data_cache/.gitkeep` (empty file — directory marker)

- [ ] **Step 1: Verify branch.**

Run: `git branch --show-current` — must be `feature/narrative-extension-slice3`.

- [ ] **Step 2: Create the directory marker**

```bash
mkdir -p notebooks/data_cache
touch notebooks/data_cache/.gitkeep
```

- [ ] **Step 3: Write the failing smoke tests**

Create `tests/test_notebook_28_smoke.py`:

```python
"""Offline smoke tests for notebook 28 (US LUI from Fed text)."""
from __future__ import annotations


def test_notebook_28_builder_produces_six_sections():
    """The builder must emit a notebook with all six sections present."""
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo))
    from tools.make_notebook_28_us_lui_text import build

    nb = build()
    cells = nb["cells"]
    assert len(cells) >= 12, (
        f"expected >= 12 cells across 6 sections, got {len(cells)}"
    )

    md_sources = "\n".join(
        c["source"] for c in cells if c["cell_type"] == "markdown"
    )
    for marker in ("Setup", "Corpus", "indices", "Validation", "Save"):
        assert marker in md_sources, (
            f"section marker {marker!r} not found in any markdown cell"
        )


def test_notebook_28_imports_resolve():
    """The notebook's package imports must work in the current env."""
    from puremacro.narrative import lui, epu, wui  # noqa: F401
    from puremacro.narrative.sources import (
        iter_fed_minutes, iter_fed_speeches,
    )  # noqa: F401
```

- [ ] **Step 4: Run tests, verify they fail**

Run: `pytest tests/test_notebook_28_smoke.py -v --no-header 2>&1 | tail -10`
Expected: `test_notebook_28_builder_produces_six_sections` fails with `ModuleNotFoundError: tools.make_notebook_28_us_lui_text`. The imports test should already pass (the puremacro packages are shipped).

- [ ] **Step 5: Create the builder file**

Create `tools/make_notebook_28_us_lui_text.py` with the full content below.

```python
"""Deterministic regeneration of 28_us_lui_from_fed_text.ipynb.

First research-level integration of the puremacro.narrative.indices
subpackage (v0.6.2 / v0.7.0). Builds a US Labor-Market Uncertainty
index from FOMC minutes + Fed speeches and validates it against
external benchmarks (BBD-EPU and the state-panel-aggregated US
unemployment rate).

Spec: docs/specs/2026-05-09-notebook-28-us-lui-design.md
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def build() -> nbf.NotebookNode:
    nb = new_notebook()
    cells = []

    # ------------------------------------------------------------------
    # §1 Setup
    # ------------------------------------------------------------------
    cells.append(new_markdown_cell(
        "# 28 — US Labor-Market Uncertainty from Fed text\n\n"
        "**Question.** Can the new `puremacro.narrative.indices.lui()` helper "
        "produce a US labor-uncertainty signal from the published Fed corpus "
        "(FOMC minutes + Fed speeches) that lines up with established "
        "labor-market proxies (jobless claims, unemployment rate, BBD-EPU)?\n\n"
        "**Pipeline.** Slice 1 (v0.6.1) shipped the Fed connectors. Slice 2 "
        "(v0.6.2) shipped `lui()`. Slice 3 (v0.7.0) closed deferrals. This "
        "notebook is the first *research-level* exercise of all three slices "
        "end-to-end.\n\n"
        "**Out of scope.** State-panel work (deferred to nb 29). "
        "Cross-country comparison (US-only here). Tone / hawkish-dovish "
        "(uncertainty axis only)."
    ))

    cells.append(new_markdown_cell("## §1 Setup"))

    cells.append(new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path.cwd().resolve()))\n"
        "sys.path.insert(0, str(Path.cwd().resolve().parent))\n"
        "from notebooks._bootstrap import setup\n"
        "ROOT = setup()\n"
    ))

    cells.append(new_code_cell(
        "import os\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from puremacro.narrative import lui, epu, wui\n"
        "from puremacro.narrative.sources import (\n"
        "    iter_fed_minutes, iter_fed_speeches,\n"
        ")\n"
    ))

    # ------------------------------------------------------------------
    # §2 Corpus assembly
    # ------------------------------------------------------------------
    cells.append(new_markdown_cell(
        "## §2 Corpus assembly\n\n"
        "Fetch FOMC minutes and Fed speeches via the Slice-1 connectors. "
        "Cache to `notebooks/data_cache/fed_corpus_28.parquet` so subsequent "
        "runs are offline. Set `PUREMACRO_REFETCH=1` to force a re-fetch."
    ))

    cells.append(new_code_cell(
        "CACHE_PATH = ROOT / 'notebooks' / 'data_cache' / 'fed_corpus_28.parquet'\n"
        "REFETCH = os.getenv('PUREMACRO_REFETCH') == '1'\n"
        "\n"
        "if CACHE_PATH.exists() and not REFETCH:\n"
        "    corpus_df = pd.read_parquet(CACHE_PATH)\n"
        "    print(f'[cache hit] {len(corpus_df)} records from {CACHE_PATH}')\n"
        "else:\n"
        "    rows = []\n"
        "    for src_iter, src_name in [(iter_fed_minutes, 'minutes'),\n"
        "                                (iter_fed_speeches, 'speeches')]:\n"
        "        try:\n"
        "            for date, text, url, meta in src_iter():\n"
        "                rows.append({\n"
        "                    'date': pd.Timestamp(date),\n"
        "                    'text': text,\n"
        "                    'source_url': url,\n"
        "                    'language': meta.get('language', 'en'),\n"
        "                    'doctype': meta.get('doctype', src_name),\n"
        "                })\n"
        "        except Exception as e:\n"
        "            print(f'[skip] {src_name}: {e}')\n"
        "    corpus_df = (pd.DataFrame(rows)\n"
        "                   .drop_duplicates('source_url')\n"
        "                   .sort_values('date')\n"
        "                   .reset_index(drop=True))\n"
        "    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)\n"
        "    corpus_df.to_parquet(CACHE_PATH, index=False)\n"
        "    print(f'[fetched] {len(corpus_df)} records → cached at {CACHE_PATH}')\n"
        "\n"
        "corpus_df.head()"
    ))

    cells.append(new_code_cell(
        "# Reconstitute 4-tuples for the kernels.\n"
        "records = [\n"
        "    (row.date, row.text, row.source_url,\n"
        "     {'language': row.language, 'doctype': row.doctype})\n"
        "    for row in corpus_df.itertuples()\n"
        "]\n"
        "print(f'records: {len(records)}; first {records[0][0] if records else \"-\"}; '\n"
        "      f'last {records[-1][0] if records else \"-\"}')\n"
    ))

    # ------------------------------------------------------------------
    # §3 Compute indices
    # ------------------------------------------------------------------
    cells.append(new_markdown_cell(
        "## §3 Compute indices\n\n"
        "LUI is the headline; EPU and WUI are point-of-comparison signals "
        "computed from the same corpus."
    ))

    cells.append(new_code_cell(
        "if not records:\n"
        "    print('[no corpus] skipping index computation')\n"
        "    panel = pd.DataFrame()\n"
        "else:\n"
        "    ri_lui = lui(records, country='USA', language='en', normalize='zscore')\n"
        "    ri_epu = epu(records, country='USA', language='en', normalize='zscore')\n"
        "    ri_wui = wui(records, country='USA', language='en', normalize='zscore')\n"
        "    panel = pd.DataFrame({\n"
        "        'lui': ri_lui.series,\n"
        "        'epu': ri_epu.series,\n"
        "        'wui': ri_wui.series,\n"
        "    })\n"
        "panel.tail(8)"
    ))

    # ------------------------------------------------------------------
    # §4 Time-series plot
    # ------------------------------------------------------------------
    cells.append(new_markdown_cell(
        "## §4 Time-series plot\n\n"
        "Quarterly z-scored LUI / EPU / WUI with NBER recession bars. "
        "If the LUI signal is real, peaks should cluster around 2008-09 "
        "and 2020Q1-Q2."
    ))

    cells.append(new_code_cell(
        "_NBER_RECESSIONS = [\n"
        "    ('1973-11-01', '1975-03-01'),\n"
        "    ('1980-01-01', '1980-07-01'),\n"
        "    ('1981-07-01', '1982-11-01'),\n"
        "    ('1990-07-01', '1991-03-01'),\n"
        "    ('2001-03-01', '2001-11-01'),\n"
        "    ('2007-12-01', '2009-06-01'),\n"
        "    ('2020-02-01', '2020-04-01'),\n"
        "]\n"
        "\n"
        "if panel.empty:\n"
        "    print('[no panel] skipping plot')\n"
        "else:\n"
        "    fig, ax = plt.subplots(figsize=(10, 4.5))\n"
        "    for col, color, label in [\n"
        "        ('lui', '#d62728', 'LUI (labor-uncertainty)'),\n"
        "        ('epu', '#1f77b4', 'EPU (BBD-style)'),\n"
        "        ('wui', '#2ca02c', 'WUI (uncertainty stems)'),\n"
        "    ]:\n"
        "        if col in panel:\n"
        "            ax.plot(panel.index, panel[col].values,\n"
        "                    color=color, label=label, linewidth=1.4)\n"
        "    for s, e in _NBER_RECESSIONS:\n"
        "        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),\n"
        "                   color='gray', alpha=0.15, zorder=-10)\n"
        "    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)\n"
        "    ax.set_title('US labor-market and policy uncertainty from Fed text (z-score)')\n"
        "    ax.set_ylabel('z-score')\n"
        "    ax.legend(loc='upper left', frameon=False)\n"
        "    fig.tight_layout()\n"
        "    fig_path = ROOT / 'notebooks' / 'output_figures' / '28_lui_us_timeseries.pdf'\n"
        "    fig_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    fig.savefig(fig_path, bbox_inches='tight')\n"
        "    print(f'figure saved: {fig_path}')"
    ))

    # ------------------------------------------------------------------
    # §5 Validation correlations
    # ------------------------------------------------------------------
    cells.append(new_markdown_cell(
        "## §5 Validation correlations\n\n"
        "Correlate our LUI / EPU / WUI against external benchmarks: "
        "(a) the published BBD-EPU mirror via "
        "`puremacro.instruments.literature.bbd_epu`, "
        "(b) the US national unemployment rate aggregated from "
        "`data/processed/state_panel_M.parquet` (if available on this "
        "checkout). High EPU↔BBD-EPU correlation is the obvious sanity "
        "check; LUI↔urate is the headline test."
    ))

    cells.append(new_code_cell(
        "benchmarks: dict[str, pd.Series] = {}\n"
        "\n"
        "# (a) BBD-EPU published mirror.\n"
        "try:\n"
        "    from puremacro.instruments.literature import bbd_epu\n"
        "    inst_bbd = bbd_epu.load()\n"
        "    s_bbd = inst_bbd.series\n"
        "    if not isinstance(s_bbd.index, pd.DatetimeIndex):\n"
        "        s_bbd.index = pd.to_datetime(s_bbd.index)\n"
        "    benchmarks['bbd_epu'] = s_bbd.resample('QS').mean()\n"
        "except Exception as e:\n"
        "    print(f'[skip] BBD-EPU mirror unreachable: {e}')\n"
        "\n"
        "# (b) National unemployment rate from the state panel.\n"
        "state_path = ROOT / 'data' / 'processed' / 'state_panel_M.parquet'\n"
        "if state_path.exists():\n"
        "    sp = pd.read_parquet(state_path)\n"
        "    sp['date'] = pd.to_datetime(sp['date'])\n"
        "    urate_us = (sp.groupby('date')['urate_laus'].mean()\n"
        "                  .resample('QS').mean())\n"
        "    benchmarks['urate_us'] = urate_us\n"
        "else:\n"
        "    print(f'[skip] {state_path} not on this checkout')\n"
        "\n"
        "list(benchmarks.keys())"
    ))

    cells.append(new_code_cell(
        "if panel.empty or not benchmarks:\n"
        "    print('[no correlations to compute]')\n"
        "    corrs = pd.DataFrame()\n"
        "else:\n"
        "    aligned = pd.DataFrame({\n"
        "        **{f'idx_{c}': panel[c] for c in panel.columns},\n"
        "        **{f'bm_{k}': v.reindex(panel.index) for k, v in benchmarks.items()},\n"
        "    })\n"
        "    rho_rows = []\n"
        "    for idx_col in [c for c in aligned.columns if c.startswith('idx_')]:\n"
        "        for bm_col in [c for c in aligned.columns if c.startswith('bm_')]:\n"
        "            sub = aligned[[idx_col, bm_col]].dropna()\n"
        "            if len(sub) < 4:\n"
        "                rho = float('nan')\n"
        "            else:\n"
        "                rho = float(sub.corr().iloc[0, 1])\n"
        "            rho_rows.append({\n"
        "                'index': idx_col[4:],\n"
        "                'benchmark': bm_col[3:],\n"
        "                'rho': rho,\n"
        "                'n_q': int(len(sub)),\n"
        "            })\n"
        "    corrs = pd.DataFrame(rho_rows)\n"
        "    out_csv = ROOT / 'notebooks' / 'output_tables' / '28_lui_validation_corr.csv'\n"
        "    out_csv.parent.mkdir(parents=True, exist_ok=True)\n"
        "    corrs.to_csv(out_csv, index=False)\n"
        "corrs"
    ))

    # ------------------------------------------------------------------
    # §6 Save outputs
    # ------------------------------------------------------------------
    cells.append(new_markdown_cell(
        "## §6 Save outputs\n\n"
        "Persist the quarterly index panel and a small JSON metadata "
        "sidecar for downstream consumption (e.g., notebook 29's "
        "state-panel LP-IV with the LUI as national shock)."
    ))

    cells.append(new_code_cell(
        "import json\n"
        "from datetime import datetime\n"
        "\n"
        "if not panel.empty:\n"
        "    out_panel = ROOT / 'notebooks' / 'output_tables' / '28_lui_us_quarterly.parquet'\n"
        "    out_panel.parent.mkdir(parents=True, exist_ok=True)\n"
        "    panel.to_parquet(out_panel)\n"
        "    meta = {\n"
        "        'corpus_size': int(len(corpus_df)),\n"
        "        'language': 'en',\n"
        "        'normalization': 'zscore',\n"
        "        'first_quarter': str(panel.index.min().date()) if len(panel) else None,\n"
        "        'last_quarter': str(panel.index.max().date()) if len(panel) else None,\n"
        "        'computed_at': datetime.utcnow().isoformat(),\n"
        "        'puremacro_version': '0.7.0',\n"
        "    }\n"
        "    out_meta = ROOT / 'notebooks' / 'output_tables' / '28_lui_us_quarterly.meta.json'\n"
        "    out_meta.write_text(json.dumps(meta, indent=2))\n"
        "    print(f'panel saved: {out_panel}')\n"
        "    print(f'meta:       {out_meta}')\n"
        "else:\n"
        "    print('[no panel — nothing to save]')"
    ))

    nb["cells"] = cells
    return nb


def main() -> None:
    nb = build()
    out = Path(__file__).resolve().parent.parent / "notebooks" / "28_us_lui_from_fed_text.ipynb"
    nbf.write(nb, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run smoke tests, expect green**

Run: `pytest tests/test_notebook_28_smoke.py -v --no-header 2>&1 | tail -10`
Expected: 2 tests pass.

- [ ] **Step 7: Run full suite, no regressions**

Run: `pytest -q --no-header 2>&1 | tail -3`
Expected: at least 956 + 2 = 958 passed.

- [ ] **Step 8: Commit**

All paths below are relative to the git toplevel (`uncertainty_examples/`), not `puremacro/`. From the toplevel:

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current   # must be feature/narrative-extension-slice3
git status -s
git add tools/make_notebook_28_us_lui_text.py \
        tests/test_notebook_28_smoke.py \
        notebooks/data_cache/.gitkeep
git commit -m "feat(notebooks): builder for nb 28 — US LUI from Fed text"
```

---

## Task 2: Render the notebook + commit

**Files:**
- Create: `notebooks/28_us_lui_from_fed_text.ipynb` (output of running the builder)

- [ ] **Step 1: Verify branch.**

- [ ] **Step 2: Run the builder to produce the notebook**

```bash
python -m tools.make_notebook_28_us_lui_text
```

Or equivalently:
```bash
python tools/make_notebook_28_us_lui_text.py
```

Expected output: `wrote /…/notebooks/28_us_lui_from_fed_text.ipynb`.

- [ ] **Step 3: Verify the rendered notebook is valid JSON / nbformat**

Run:
```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/28_us_lui_from_fed_text.ipynb', as_version=4)
print(f'cells: {len(nb.cells)}')
print(f'first markdown: {nb.cells[0].source[:80]}')
"
```
Expected: `cells: 14` (or thereabouts; depends on builder output) and the title `# 28 — US Labor-Market…`.

- [ ] **Step 4: Verify the notebook opens in nbformat round-trip**

Run:
```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/28_us_lui_from_fed_text.ipynb', as_version=4)
nbformat.validate(nb)
print('valid')
"
```
Expected: `valid`.

- [ ] **Step 5: Commit the rendered notebook**

From the toplevel (`uncertainty_examples/`):

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current
git add notebooks/28_us_lui_from_fed_text.ipynb
git commit -m "feat(notebooks): render nb 28 (US LUI from Fed text)"
```

---

## Definition of Done

- [ ] Branch `feature/narrative-extension-slice3` has 2 new commits past `v0.7.0`.
- [ ] `tools/make_notebook_28_us_lui_text.py` exists, ~250-300 lines, builds a 14-cell notebook.
- [ ] `notebooks/28_us_lui_from_fed_text.ipynb` exists, opens in `nbformat.read`, has a 6-section structure.
- [ ] `notebooks/data_cache/` directory exists (with `.gitkeep`).
- [ ] `tests/test_notebook_28_smoke.py` passes 2 tests offline.
- [ ] `pytest -q` ≥ 958 passed (956 baseline + 2 new).
- [ ] No `puremacro/` package modifications. No version bump.

## Out of scope (deliberate)

- **Live notebook execution.** Building the notebook is the implementation task; running it (which needs network for the Fed connectors and BBD-EPU mirror) is the user's interactive step. The user can run `PUREMACRO_REFETCH=1 jupyter execute notebooks/28_us_lui_from_fed_text.ipynb` manually.
- **Caching the executed corpus in git.** The first interactive run will write `notebooks/data_cache/fed_corpus_28.parquet`. Whether to commit that file is a follow-on decision after the user sees its size; the plan only ensures the directory exists.
- **`notebooks/output_figures/28_lui_us_timeseries.pdf` and `output_tables/28_lui_us_quarterly.parquet`** — these are produced only by interactive execution. Not part of this plan.
- **State-panel LP-IV** with LUI as national shock — that's notebook 29.
