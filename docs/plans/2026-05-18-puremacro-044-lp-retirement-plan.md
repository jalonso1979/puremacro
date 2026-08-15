# puremacro 0.44.0 — LP Retirement + Notebook Re-Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Body-rewrite R1_02/R1_03/R1_05 to canonical lp/* APIs, build the data cache, re-execute body-rewrite notebooks against canonical, then delete `puremacro/lp/lp_*.py` + `puremacro/inference/legacy/`. Ship as **0.44.0**.

**Architecture:** Three body-rewrite tasks update notebook cells AND paired builders simultaneously (notebooks↔builders memory pin). A data-build task populates `data/processed/panel_Q.parquet` so notebook re-execution can complete. A re-execution task runs paired builders to regenerate cell outputs (controller-only per long-nbconvert pin). Final deletion tasks retire the now-dead lp/lp_* and inference/legacy paths.

**Tech Stack:** Python ≥3.10, numpy/scipy/pandas/matplotlib, pytest, jupyter nbconvert.

**Source spec:** `docs/specs/2026-05-18-puremacro-044-lp-retirement-design.md` (commit `b03ef6a`).

**Pre-execution state (HEAD `b03ef6a`):**
- 0.43.0 shipped at `1bae0b2`; merged to `main` + pushed.
- Phase-2.5 deferrals alive: `puremacro/lp/lp_*.py` (8 files), `puremacro/inference/legacy/` (10 files).
- Test baseline: 1277 passed, 10 pre-existing failed.
- Canonical signatures verified (2026-05-18):
  - `lp.jorda.lp_hac(df, y, x, horizons=range(0,21), n_lags=2, controls=None, alpha=0.10) -> pd.DataFrame`
  - `lp.iv.lp_iv(df, y, x, z, horizons=range(0,21), n_lags=2, controls=None, alpha=0.10) -> pd.DataFrame`
  - `lp.panel.panel_lp(...) -> pd.DataFrame`
  - `lp.smooth.lp_smooth(...) -> pd.DataFrame`
  - `lp.state_dep.lp_state_dep(...) -> pd.DataFrame`
  - `lp.garch_state.lp_garch_state(...) -> pd.DataFrame`
  - `lp.garch_in_mean.lp_garch_in_mean(...) -> pd.DataFrame`
  - `build_panel.build_all(countries=None, *, fast=False, refresh=False) -> (panel_Q, panel_M)`
- Legacy signatures use `target=/shock=/horizon=/lags=/ci=` and return `LPResult`/`LPIVResult`/`PanelLPResult`/`LPSmoothResult`/`LPStateDepResult` dataclasses with `.beta`/`.se`/`.lower`/`.upper`/`.ci` attributes.

---

## File structure

**Notebooks rewritten + their builders (paired commits):**
- `notebooks/R1_methods/R1_02_lp_menu.ipynb` + `tools/make_notebook_R1_02.py` (Task 1; biggest, ~7 sections)
- `notebooks/R1_methods/R1_03_cross_country.ipynb` + `tools/make_notebook_R1_03.py` (Task 2; small, 2 legacy imports)
- `notebooks/R1_methods/R1_05_publication.ipynb` + `tools/make_notebook_R1_05.py` (Task 3; small, 1 legacy import)

**Deleted files:**
- `puremacro/lp/lp_jorda.py`, `lp_iv.py`, `lp_panel.py`, `lp_panel_dk.py`, `lp_state_dep.py`, `lp_smooth.py`, `lp_garch_state.py`, `lp_garch_in_mean.py` (8 files, Task 6)
- `puremacro/inference/legacy/` (entire directory, 10 files, Task 7)
- `puremacro/lp/garch_utils.py` → rename to `_garch_utils.py` if no external callers (Task 8)

**Docs:**
- `ARCHITECTURE.md` (Task 9) — remove Phase-2.5 consolidation entries; update stability tier table.
- `CHANGELOG.md` (Task 9) — prepend 0.44.0 entry.
- `puremacro/__init__.py`, `pyproject.toml`, `tests/test_import.py`, `tests/fixtures/public_api_snapshot.json` (Task 10).

---

## Task 0: Per-cell audit of R1_02 / R1_03 / R1_05

Read-only mapping of every legacy LP call site + result-access pattern in the three body-rewrite notebooks. Produces a per-cell rewrite map that Tasks 1-3 follow exactly.

- [ ] **Step 1: Confirm pre-task pytest baseline**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest tests/test_var/ tests/test_lp/ tests/test_cholesky_shocks.py tests/test_robustness.py tests/test_pyodide_compat.py -q --tb=no 2>&1 | tail -5
```

Expected: 122 passed (per 0.43.0 final).

- [ ] **Step 2: Inventory R1_02's legacy LP call sites**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
grep -nE "lp_irf\(|lp_iv_irf\(|lp_panel_irf\(|lp_smooth_irf\(|lp_state_dep_irf\(|LPResult|LPIVResult|PanelLPResult|LPSmoothResult|LPStateDepResult|LPGARCHStateResult|LPGIMResult|WeakInstrumentWarning" tools/make_notebook_R1_02.py | head -60
```

Capture every line that calls a legacy LP function or references a legacy result class.

- [ ] **Step 3: Inventory R1_03's legacy LP call sites**

```bash
grep -nE "lp_irf\(|lp_panel_irf\(|RegimeInteractionLPResult|LPResult" tools/make_notebook_R1_03.py | head -20
```

- [ ] **Step 4: Inventory R1_05's legacy LP call sites**

```bash
grep -nE "lp_irf\(|LPResult" tools/make_notebook_R1_05.py | head -10
```

- [ ] **Step 5: Write audit notes**

Create `docs/plans/_044_audit_notes.md`:

```markdown
# 0.44.0 audit notes — 2026-05-18

## Baseline
- pytest (targeted): <N> passed.

## R1_02 legacy LP call sites
<paste step 2 output with one-line notes per call site>

## R1_03 legacy LP call sites
<paste step 3>

## R1_05 legacy LP call sites
<paste step 4>

## Legacy → canonical mapping (verified 2026-05-18)
| Legacy fn | Canonical fn | Kwarg renames | Return type change |
|---|---|---|---|
| lp_irf | lp_hac | target→y, shock→x, horizon→horizons (int→iter), lags→n_lags, ci→alpha (flip), B (bootstrap reps)→dropped | LPResult → pd.DataFrame |
| lp_iv_irf | lp_iv | target→y, shock_variable→x, instrument→z, horizon→horizons, lags→n_lags, ci→alpha, ar_grid_points→dropped | LPIVResult → pd.DataFrame |
| lp_panel_irf | panel_lp | target→y, shock→x, horizon→horizons, lags→n_lags, entity_col→entity_level, time_col→time_level, ci→alpha | PanelLPResult → pd.DataFrame |
| lp_smooth_irf | lp_smooth | target→y, shock→x, horizon→horizons, lags→n_lags, ci→alpha, n_knots+lambda_*+plus → verify canonical kwargs | LPSmoothResult → pd.DataFrame |
| lp_state_dep_irf | lp_state_dep | target→y, shock→x, horizon→horizons, lags→n_lags, state_col→state_var, ci→alpha, B (bootstrap)→dropped | LPStateDepResult → pd.DataFrame |
```

- [ ] **Step 6: Commit audit notes**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
git add docs/plans/_044_audit_notes.md
git commit -m "$(cat <<'EOF'
docs(0.44.0): pre-rewrite audit of R1_02/R1_03/R1_05 LP call sites

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: Body-rewrite R1_02_lp_menu.ipynb + builder

**Files:**
- Modify: `tools/make_notebook_R1_02.py` (the builder; defines the .ipynb cell content).
- Modify: `notebooks/R1_methods/R1_02_lp_menu.ipynb` (the executed .ipynb; outputs preserved at this task — re-execution in Task 5).

**Approach:** R1_02 has ~7 sections each with its own legacy LP demo. Rewrite the BUILDER's cell content (Python source strings) — that's the source of truth. Then apply the same edits to the .ipynb's source cells (without re-executing — outputs stay until Task 5).

The notebook has an internal helper function (legacy line 182) that duck-types over LPResult/PanelLPResult/LPSmoothResult. After rewrite, all canonical functions return DataFrames; the helper becomes a column-access helper.

- [ ] **Step 1: Rewrite the builder's import cell**

Open `tools/make_notebook_R1_02.py`. Find the import-cell block (around line 108) and replace the legacy imports:

```python
# Before (the literal string content in the builder):
"from puremacro.lp.lp_jorda import lp_irf, LPResult                          # §1\n"
"from puremacro.lp.lp_panel import lp_panel_irf, PanelLPResult               # §2\n"
"from puremacro.lp.lp_iv import lp_iv_irf, LPIVResult, WeakInstrumentWarning # §3\n"
"from puremacro.lp.lp_smooth import lp_smooth_irf, LPSmoothResult            # §4\n"
"from puremacro.lp.lp_state_dep import lp_state_dep_irf                    # §5\n"
"from puremacro.lp.lp_garch_state import LPGARCHStateResult                 # §6 result type\n"
"from puremacro.lp.lp_garch_in_mean import LPGIMResult                      # §7 result type\n"

# After:
"from puremacro.lp.jorda import lp_hac           # §1\n"
"from puremacro.lp.panel import panel_lp         # §2\n"
"from puremacro.lp.iv import lp_iv               # §3\n"
"from puremacro.lp.smooth import lp_smooth       # §4\n"
"from puremacro.lp.state_dep import lp_state_dep # §5\n"
"from puremacro.lp.garch_state import lp_garch_state    # §6\n"
"from puremacro.lp.garch_in_mean import lp_garch_in_mean # §7\n"
```

If `WeakInstrumentWarning` is actually referenced in the notebook (not just imported), find its canonical home:

```bash
grep -rn "class WeakInstrumentWarning\|WeakInstrumentWarning =" puremacro/ 2>/dev/null | head -5
```

If it's in `puremacro.inference.weak_iv`, add `from puremacro.inference.weak_iv import WeakInstrumentWarning`. If it's not anywhere canonical, drop the reference + downstream `try/except WeakInstrumentWarning:` clauses.

- [ ] **Step 2: Rewrite each section's LP call site**

For §1 Jordà LP (around line 187 in the builder):

```python
# Before:
"    return lp_irf(\n"
"        df, target=TARGET, shock=shock,\n"
"        horizon=HORIZON, lags=LAGS, ci=CI,\n"
"    )\n"

# After:
"    return lp_hac(\n"
"        df, y=TARGET, x=shock,\n"
"        horizons=range(0, HORIZON + 1), n_lags=LAGS, alpha=1.0 - CI,\n"
"    )\n"
```

Apply the same pattern to §2 (`lp_panel_irf` → `panel_lp` at line 411), §3 (`lp_iv_irf` → `lp_iv` at lines 510, 562), §4 (`lp_smooth_irf` → `lp_smooth` at lines 628, 673, 693), §5 (`lp_state_dep_irf` → `lp_state_dep` at line 835).

For §3 LP-IV, the legacy `instrument=` kwarg becomes canonical `z=`.

For §5 lp_state_dep, the legacy `state_col=` becomes canonical `state_var=` (Task 4 of 0.43.0 confirmed this naming).

For §6/§7 GARCH variants: the legacy notebook only imported the *Result classes, didn't call functions. After rewrite, drop these unused result-class imports — the canonical functions return DataFrames that downstream code can use directly.

- [ ] **Step 3: Rewrite the internal duck-typed helper**

Around line 201 in the builder, the duck-typed helper:

```python
# Before (paraphrased — read the actual content):
"def _plot_band(result, ax, label=None):\n"
"    \"\"\"Duck-typed over LPResult / PanelLPResult / LPSmoothResult.\"\"\"\n"
"    h = np.arange(len(result.beta))\n"
"    ax.plot(h, result.beta, label=label)\n"
"    ax.fill_between(h, result.lower, result.upper, alpha=0.2)\n"

# After (DataFrame-based):
"def _plot_band(result, ax, label=None):\n"
"    \"\"\"Plot a long-form LP result DataFrame's IRF + bands.\"\"\"\n"
"    ax.plot(result['h'], result['beta'], label=label)\n"
"    ax.fill_between(result['h'], result['lo'], result['hi'], alpha=0.2)\n"
```

Verify the canonical DataFrames have columns `h`/`beta`/`lo`/`hi` (lp_hac/lp_iv/lp_smooth/lp_state_dep return these per their signatures verified above). If any has different column names, adapt the helper.

- [ ] **Step 4: Rewrite the wrapper helper (line 182 region)**

If R1_02 has a `lp_irf`-returning wrapper function that wraps lp_irf and adds extra fields, rewrite to call `lp_hac` and return its DataFrame.

- [ ] **Step 5: Mirror edits to the .ipynb cells**

The .ipynb has its own copy of the cell source strings (independent of the builder). The notebooks↔builders memory pin requires editing BOTH together.

Use `jupyter nbconvert --to script notebooks/R1_methods/R1_02_lp_menu.ipynb --stdout` to view current source, then apply identical edits to each affected cell via Read + Edit on the .ipynb file. The .ipynb is JSON; cell source content lives under `cells[i].source` (list of strings).

A reliable approach: edit the .ipynb in place using Edit tool, finding each legacy import string and replacing with the canonical equivalent.

Do NOT run `python tools/make_notebook_R1_02.py` — running the builder would regenerate the .ipynb and CLOBBER its current outputs (per memory pin `feedback_builder_clobbers_outputs`). Task 5 handles re-execution explicitly.

- [ ] **Step 6: Verify .ipynb is still valid JSON**

```bash
jupyter nbconvert --to script "notebooks/R1_methods/R1_02_lp_menu.ipynb" --stdout 2>&1 | head -3
```

Expected: prints script body, no errors.

- [ ] **Step 7: Verify no remaining legacy imports in R1_02**

```bash
grep -nE "lp_jorda|lp_iv|lp_panel|lp_smooth|lp_state_dep|lp_garch_state|lp_garch_in_mean|LPResult|LPIVResult|PanelLPResult|LPSmoothResult|LPGARCHStateResult|LPGIMResult|lp_irf\(|lp_iv_irf\(|lp_panel_irf\(|lp_smooth_irf\(|lp_state_dep_irf\(" tools/make_notebook_R1_02.py notebooks/R1_methods/R1_02_lp_menu.ipynb 2>/dev/null | head -10
```

Expected: zero hits. (Legacy module names may appear in markdown text mentioning "the lp_irf API" historically — that's fine if it's narrative, but if it's a code reference, it needs rewrite.)

- [ ] **Step 8: Commit the pair**

```bash
git add tools/make_notebook_R1_02.py notebooks/R1_methods/R1_02_lp_menu.ipynb
git commit -m "$(cat <<'EOF'
refactor(R1_02): body-rewrite lp menu to canonical lp/* APIs — 0.44.0

R1_02 was the legacy lp_* API demo. Rewritten to demo canonical lp/*:
lp_hac/lp_iv/panel_lp/lp_smooth/lp_state_dep/lp_garch_state/
lp_garch_in_mean. Section structure preserved. Outputs unchanged at
this commit; Task 5 re-executes via the paired builder.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Body-rewrite R1_03_cross_country.ipynb + builder

**Files:** `tools/make_notebook_R1_03.py`, `notebooks/R1_methods/R1_03_cross_country.ipynb`.

Per Task 0 audit, R1_03 has 2 legacy LP imports to address.

- [ ] **Step 1: Locate the two legacy import lines in the builder**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
grep -nE "lp_jorda|lp_panel.*RegimeInteractionLPResult|RegimeInteractionLPResult" tools/make_notebook_R1_03.py | head -5
```

Expected: two hits (one `from puremacro.lp.lp_jorda import lp_irf`, one `from puremacro.lp.lp_panel import RegimeInteractionLPResult  # legacy result type`).

- [ ] **Step 2: Rewrite imports**

```python
# Before:
"from puremacro.lp.lp_jorda import lp_irf\n"
"from puremacro.lp.lp_panel import RegimeInteractionLPResult  # legacy result type\n"

# After:
"from puremacro.lp.jorda import lp_hac\n"
# Drop the RegimeInteractionLPResult import — canonical lp_panel_regime_interaction
# returns DataFrame, no result-class needed.
```

- [ ] **Step 3: Rewrite call sites of `lp_irf` and any references to `RegimeInteractionLPResult`**

Find each `lp_irf(...)` call and rewrite to `lp_hac(df, y=..., x=..., horizons=range(0, HORIZON + 1), n_lags=..., alpha=1.0 - ci_legacy)`.

Find any `isinstance(result, RegimeInteractionLPResult)` checks or `result.beta`-style attribute access — rewrite to DataFrame column access.

- [ ] **Step 4: Mirror to .ipynb**

Apply identical edits to `notebooks/R1_methods/R1_03_cross_country.ipynb`. Do NOT run the builder.

- [ ] **Step 5: Verify .ipynb is valid + no remaining legacy lp_ imports**

```bash
jupyter nbconvert --to script "notebooks/R1_methods/R1_03_cross_country.ipynb" --stdout 2>&1 | head -3
grep -nE "lp_jorda|lp_panel.*Result|lp_irf\(|RegimeInteractionLPResult" tools/make_notebook_R1_03.py notebooks/R1_methods/R1_03_cross_country.ipynb 2>/dev/null
```

Expected: zero legacy hits.

- [ ] **Step 6: Commit**

```bash
git add tools/make_notebook_R1_03.py notebooks/R1_methods/R1_03_cross_country.ipynb
git commit -m "$(cat <<'EOF'
refactor(R1_03): drop lp_jorda + RegimeInteractionLPResult imports — 0.44.0

Last two legacy LP holdovers in R1_03. Outputs unchanged; Task 5 re-execs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Body-rewrite R1_05_publication.ipynb + builder

**Files:** `tools/make_notebook_R1_05.py`, `notebooks/R1_methods/R1_05_publication.ipynb`.

Per Task 0 audit, R1_05 has 1 legacy LP import (`from puremacro.lp.lp_jorda import lp_irf`).

- [ ] **Step 1: Locate the legacy import**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
grep -nE "lp_jorda|lp_irf" tools/make_notebook_R1_05.py | head -5
```

- [ ] **Step 2: Rewrite the import + call site**

```python
# Before:
"from puremacro.lp.lp_jorda import lp_irf\n"

# After:
"from puremacro.lp.jorda import lp_hac\n"
```

Find each `lp_irf(...)` call and rewrite to `lp_hac(...)` with canonical kwargs. Update any `result.beta`/`result.se`/`result.lower`/`result.upper` to DataFrame column access (`result['beta']` etc.).

- [ ] **Step 3: Mirror to .ipynb**

Apply identical edits. Do NOT run the builder.

- [ ] **Step 4: Verify + commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
grep -nE "lp_jorda|lp_irf\(" ../tools/make_notebook_R1_05.py ../notebooks/R1_methods/R1_05_publication.ipynb 2>/dev/null
git add ../tools/make_notebook_R1_05.py ../notebooks/R1_methods/R1_05_publication.ipynb
git commit -m "$(cat <<'EOF'
refactor(R1_05): canonical lp_hac — 0.44.0

Last legacy LP holdover in R1_05. Outputs unchanged; Task 5 re-execs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Build `data/processed/panel_Q.parquet`

**Files:** none modified; this populates the data cache.

R1_01 (already migrated in 0.43.0) and the cross-country notebooks need `panel_Q.parquet` to load country data. Without it, Task 5's re-execution of R1_01 fails immediately with a FileNotFoundError (as observed in 0.43.0 final smoke test).

- [ ] **Step 1: Check if file already exists**

```bash
ls -la "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/data/processed/panel_Q.parquet" 2>&1
```

If it exists, skip to Step 4. Otherwise:

- [ ] **Step 2: Run `build_all` (controller-only, may take 10-30 min)**

Per memory pin `feedback_long_nbconvert_no_subagent`, this should NOT be delegated to a subagent. Run from the controller's background:

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
python -c "
from puremacro.build_panel import build_all
panel_Q, panel_M = build_all(fast=True, refresh=False)
print(f'panel_Q rows: {len(panel_Q)}; panel_M rows: {len(panel_M)}')
"
```

`fast=True` keeps the 6-country filter for a quick build. `refresh=False` uses the existing disk cache (under `~/.cache/puremacro/`) where available.

- [ ] **Step 3: Verify the cache file landed**

```bash
ls -la data/processed/panel_Q.parquet 2>&1
```

Expected: file exists, non-trivial size (> 100 KB).

If the build fails (network unreachable, no FRED API key, etc.), STOP and report. Re-executing R1_01 in Task 5 will be skipped + documented; the rest of Task 5 can proceed for notebooks that don't depend on `panel_Q.parquet`.

- [ ] **Step 4: No commit — `data/processed/panel_Q.parquet` is gitignored / not staged**

Verify with `git status data/processed/panel_Q.parquet` — if it's gitignored or absent from the index, no commit needed.

---

## Task 5: Re-execute body-rewrite notebooks via paired builders

**Controller-only task per memory pin `feedback_long_nbconvert_no_subagent`.** Each builder run regenerates a notebook's outputs (and clobbers the prior outputs per `feedback_builder_clobbers_outputs` — that's the goal here).

The 6 body-rewrite notebooks to re-execute (5 from 0.43.0 + 1 new in 0.44.0; R1_02 is the only new one because R1_03 and R1_05 were in both releases' lists but only get re-executed once with the latest edits):

| # | Notebook | Builder | Source changes |
|---|---|---|---|
| 1 | R1_methods/R1_01_svar_menu.ipynb | tools/make_notebook_R1_01.py | 0.43.0 (svar canonical) |
| 2 | R1_methods/R1_02_lp_menu.ipynb | tools/make_notebook_R1_02.py | 0.44.0 Task 1 (lp canonical) |
| 3 | R1_methods/R1_03_cross_country.ipynb | tools/make_notebook_R1_03.py | 0.43.0 + 0.44.0 Task 2 |
| 4 | R1_methods/R1_04_dsge_compare.ipynb | tools/make_notebook_R1_04.py | 0.43.0 (svar canonical) |
| 5 | R1_methods/R1_05_publication.ipynb | tools/make_notebook_R1_05.py | 0.43.0 + 0.44.0 Task 3 |
| 6 | T_us_national.ipynb | (no builder — direct nbconvert) | 0.43.0 (cholesky_svar canonical) |

Per notebook: re-execute via the builder, inspect diff, commit per notebook.

- [ ] **Step 1: Re-execute R1_01**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
python tools/make_notebook_R1_01.py 2>&1 | tail -10
```

Expected: builds + executes. If `panel_Q.parquet` missing (Task 4 failed), this will FileNotFoundError on the data-load cell. Skip + document if so.

- [ ] **Step 2: Inspect R1_01 diff**

```bash
git diff -- notebooks/R1_methods/R1_01_svar_menu.ipynb | head -80
```

Outputs (cell results) will have changed because the canonical SVAR uses `safe_cholesky` with different conditioning. Source cells should match what was committed in 0.43.0. Sanity-check that no cell raises an exception in its outputs (look for `"output_type": "error"` in the diff).

- [ ] **Step 3: Commit R1_01 re-execution**

```bash
git add notebooks/R1_methods/R1_01_svar_menu.ipynb
git commit -m "$(cat <<'EOF'
nb(R1_01): re-execute against canonical SVAR — 0.44.0

Outputs regenerated post-0.43.0 svar migration. Sanity-checked
against committed source edits; no cell exceptions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4-15: Repeat Steps 1-3 for R1_02, R1_03, R1_04, R1_05**

Same pattern: `python tools/make_notebook_<X>.py`, inspect diff, commit. One commit per notebook so any individual regression can be reverted in isolation.

For each notebook, the commit message is `nb(R<X>): re-execute against canonical — 0.44.0` with a one-line note about which canonical paths the re-execution exercises.

- [ ] **Step 16: Re-execute T_us_national (no builder — direct nbconvert)**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
jupyter nbconvert --to notebook --execute \
  notebooks/T_us_national.ipynb --inplace \
  --ExecutePreprocessor.timeout=900 2>&1 | tail -5
```

Inspect diff + commit. If `panel_Q.parquet` is missing, this may fail too — skip + document.

```bash
git add notebooks/T_us_national.ipynb
git commit -m "$(cat <<'EOF'
nb(T_us_national): re-execute against canonical cholesky_svar — 0.44.0

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Delete `puremacro/lp/lp_*.py`

**Files (deletions):** all 8 `puremacro/lp/lp_*.py` files.

- [ ] **Step 1: Pre-deletion gate — no remaining callers**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git grep -E "from puremacro\.lp\.lp_|import puremacro\.lp\.lp_" -- ':!docs/' ':!CHANGELOG.md' ':!*.bak'
```

Expected: zero hits OUTSIDE the lp/lp_*.py files themselves and the legacy notebook strings in any prior pickle/JSON cache. If a real Python caller hits, STOP and migrate it first.

- [ ] **Step 2: Delete the 8 files**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
git rm puremacro/lp/lp_jorda.py puremacro/lp/lp_iv.py puremacro/lp/lp_panel.py puremacro/lp/lp_panel_dk.py puremacro/lp/lp_state_dep.py puremacro/lp/lp_smooth.py puremacro/lp/lp_garch_state.py puremacro/lp/lp_garch_in_mean.py
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/ -q --tb=line 2>&1 | tail -10
```

Expected: same pre-existing 10 failures. Total count drops by however many tests were exercising the lp_* legacy modules (likely none — those tests were in `tests/test_lp/` exercising canonical paths).

- [ ] **Step 4: Strict DeprecationWarning gate**

```bash
python -m pytest tests/ -q -W "error::DeprecationWarning" \
  --ignore tests/test_pyodide_compat.py \
  --ignore tests/test_public_api.py 2>&1 | tail -10
```

Expected: same pre-existing failures.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore(0.44.0): delete puremacro/lp/lp_*.py (8 files)

All callers (tools/, notebooks/, teaching/, in-package) migrated to
canonical lp/* APIs in 0.43.0 + 0.44.0 Tasks 1-3. The 8 Phase-2.5
banner files now have zero callers and are removed.

Removed:
- puremacro/lp/lp_jorda.py
- puremacro/lp/lp_iv.py
- puremacro/lp/lp_panel.py
- puremacro/lp/lp_panel_dk.py
- puremacro/lp/lp_state_dep.py
- puremacro/lp/lp_smooth.py
- puremacro/lp/lp_garch_state.py
- puremacro/lp/lp_garch_in_mean.py

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Delete `puremacro/inference/legacy/`

**Files (deletion):** entire `puremacro/inference/legacy/` directory (10 files).

- [ ] **Step 1: Pre-deletion gate — no remaining callers**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git grep -E "from puremacro\.inference\.legacy|import puremacro\.inference\.legacy" -- ':!docs/' ':!CHANGELOG.md' ':!*.bak'
```

Expected: zero hits (after Task 6 deleted lp/lp_state_dep.py + lp/lp_smooth.py, the only remaining `inference.legacy` callers should be gone).

- [ ] **Step 2: Delete the directory**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
git rm -r puremacro/inference/legacy/
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/ -q --tb=line 2>&1 | tail -10
```

Expected: same pre-existing failures, no new ones.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore(0.44.0): delete puremacro/inference/legacy/

The 4 distinct files (bootstrap, wild_bootstrap, block_bootstrap,
weak_iv) and 6 byte-identical shim files in inference/legacy/ are
gone now that lp/lp_state_dep.py and lp/lp_smooth.py (their only
remaining callers) are deleted in Task 6.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Decide `puremacro/lp/garch_utils.py` rename

**Files:** `puremacro/lp/garch_utils.py` (rename or leave).

The canonical `lp/garch_state.py` and `lp/garch_in_mean.py` import from `garch_utils.py`. After Task 6 deleted `lp/lp_garch_state.py` and `lp/lp_garch_in_mean.py`, there might still be external callers using the public name.

- [ ] **Step 1: Check for external callers**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git grep -E "from puremacro\.lp\.garch_utils|import puremacro\.lp\.garch_utils" -- ':!docs/' ':!CHANGELOG.md' ':!*.bak' ':!puremacro/puremacro/lp/'
```

Expected: zero hits if only `lp/garch_state.py` + `lp/garch_in_mean.py` use it.

- [ ] **Step 2: If zero external callers, rename to private**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
git mv puremacro/lp/garch_utils.py puremacro/lp/_garch_utils.py
```

Update internal callers:

```bash
sed -i.bak 's|from puremacro\.lp\.garch_utils|from puremacro.lp._garch_utils|g; s|from \.garch_utils|from ._garch_utils|g' puremacro/lp/garch_state.py puremacro/lp/garch_in_mean.py
rm puremacro/lp/garch_state.py.bak puremacro/lp/garch_in_mean.py.bak
```

(macOS `sed -i` requires the extension; clean up the `.bak` files.)

If Step 1 found external callers, SKIP this step and leave `garch_utils.py` as a public name. Document in audit notes.

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_lp/ tests/test_pyodide_compat.py -q --tb=line 2>&1 | tail -5
```

Expected: green.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(lp): rename garch_utils.py → _garch_utils.py (private) — 0.44.0

Helper module is now used only by lp/garch_state.py and
lp/garch_in_mean.py. The public name was only needed for the deleted
lp/lp_garch_state.py + lp/lp_garch_in_mean.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If Step 1 found external callers and Step 2 was skipped, no commit — document the deferral in CHANGELOG (Task 9).

---

## Task 9: ARCHITECTURE.md + CHANGELOG.md for 0.44.0

**Files:** `ARCHITECTURE.md`, `CHANGELOG.md`.

- [ ] **Step 1: Update ARCHITECTURE.md**

Find the "Known consolidation candidates" section. Remove the Phase-2.5 entries for `lp/lp_*.py` and `inference/legacy/`. The remaining entry is `regress/lp.py` (independent implementation, its own future release).

Update the stability tier table:
- REMOVE rows for all 8 `puremacro/lp/lp_*` files.
- REMOVE rows for `puremacro/inference/legacy/*`.
- If Task 8 renamed garch_utils, update its row to `lp/_garch_utils.py (private)`.

- [ ] **Step 2: Prepend 0.44.0 to CHANGELOG.md**

```markdown
## 0.44.0 — 2026-05-18

LP shim retirement + notebook re-execution. The `lp/lp_*.py` and
`inference/legacy/` paths deferred in 0.43.0 are deleted. Three
body-rewrite notebooks (R1_02, R1_03, R1_05) migrated to canonical
`lp/*` APIs. Six body-rewrite notebooks (R1_01, R1_02, R1_03, R1_04,
R1_05, T_us_national) re-executed against canonical.

### Removed (breaking)
- `puremacro.lp.lp_*` — all 8 files (`lp_jorda`, `lp_iv`, `lp_panel`,
  `lp_panel_dk`, `lp_state_dep`, `lp_smooth`, `lp_garch_state`,
  `lp_garch_in_mean`).
- `puremacro.inference.legacy` — entire directory (10 files).
- `puremacro.lp.garch_utils` (if Task 8 succeeded) → renamed to
  private `_garch_utils`.

### Changed
- `R1_02_lp_menu.ipynb` body-rewritten as canonical `lp/*` menu.
  Preserves chapter's pedagogical role on the new API.
- `R1_03_cross_country.ipynb` and `R1_05_publication.ipynb` last
  legacy lp_jorda + RegimeInteractionLPResult imports removed.
- 6 body-rewrite notebooks re-executed against canonical paths
  (R1_01 svar, R1_02 lp, R1_03 cross-country, R1_04 dsge, R1_05
  publication, T_us_national).

### Internal
- `tests/test_lp/` no longer needs to gate any shim contract.
- `ARCHITECTURE.md` cleaned of all Phase-2.5 consolidation entries.

### Pre-existing failures (unchanged from 0.43.0)
- Same 10 orthogonal failures: 7 × `test_qar_skewt_fci`, 2 ×
  `test_narrative_body_extractor_coverage`, 1 ×
  `test_narrative_indices`.

### Still deferred
- `puremacro/regress/lp.py` — independent implementation, NOT a thin
  re-export of `lp.panel`. 2 active callers in
  `tools/run_logurate_revision.py` + `tools/run_paper_extensions.py`.
  Its own follow-up release once a canonical equivalent with the
  same signature ships.
```

- [ ] **Step 3: Run tests + commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest tests/test_var/ tests/test_lp/ tests/test_cholesky_shocks.py tests/test_robustness.py tests/test_pyodide_compat.py -q --tb=line 2>&1 | tail -5
git add ARCHITECTURE.md CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(0.44.0): ARCHITECTURE + CHANGELOG — lp shim retirement complete

The Phase-2.5 lp/lp_*.py + inference/legacy/ entries are gone. The
remaining stayer is regress/lp.py (independent implementation,
separate follow-up).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Version bump to 0.44.0

**Files:** `puremacro/__init__.py`, `pyproject.toml`, `tests/test_import.py`, `tests/fixtures/public_api_snapshot.json`.

- [ ] **Step 1: Bump version literals**

Edit `puremacro/__init__.py`: `__version__ = "0.43.0"` → `"0.44.0"`.

Edit `pyproject.toml`: `version = "0.43.0"` → `"0.44.0"`.

Edit `tests/test_import.py`: any assertion of `"0.43.0"` → `"0.44.0"`.

- [ ] **Step 2: Regenerate public-API snapshot**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -c "
import importlib, dataclasses, json, sys
# Use the same _collect_current_api helper as tests/test_public_api.py
sys.path.insert(0, 'tests')
from test_public_api import _collect_current_api
snapshot = _collect_current_api()
with open('tests/fixtures/public_api_snapshot.json', 'w') as f:
    json.dump(snapshot, f, indent=2, sort_keys=True)
print(f'snapshot keys: {len(snapshot)}')
"
```

If the test file's helper has a different name or signature, inspect first:

```bash
grep -nE "def _collect_current_api\|def.*snapshot" tests/test_public_api.py | head -5
```

Adapt the regeneration script to match. The snapshot should LOSE all entries under `puremacro.lp.lp_*` and `puremacro.inference.legacy`.

- [ ] **Step 3: Run import + public API tests**

```bash
python -m pytest tests/test_import.py tests/test_public_api.py -v
```

Expected: both pass.

- [ ] **Step 4: Commit**

```bash
git add puremacro/__init__.py pyproject.toml tests/test_import.py tests/fixtures/public_api_snapshot.json
git commit -m "$(cat <<'EOF'
chore(release): bump version to 0.44.0

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Final verification gate

- [ ] **Step 1: Full pytest run (12+ min)**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest tests/ -q --tb=line 2>&1 | tail -15
```

Expected: ~1277 - any small count from removed lp/lp_* test coverage. 10 pre-existing failed unchanged.

- [ ] **Step 2: Strongest gate — no legacy paths anywhere**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git grep -E "puremacro\.lp\.lp_|puremacro\.inference\.legacy" -- ':!docs/' ':!CHANGELOG.md' ':!*.bak'
```

Expected: ZERO hits.

- [ ] **Step 3: Strict DeprecationWarning gate**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest tests/ -q -W "error::DeprecationWarning" \
  --ignore tests/test_pyodide_compat.py \
  --ignore tests/test_public_api.py 2>&1 | tail -10
```

Expected: only the same pre-existing 21 failures (skewt + narrative utcnow).

- [ ] **Step 4: Pyodide compat**

```bash
python -m pytest tests/test_pyodide_compat.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Git log + branch state**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git log --oneline b03ef6a^..HEAD | head -25
git status --short | head -5
```

Expected: 0.44.0 commit chain present from `b03ef6a` (spec) through the version-bump commit. No 0.44.0 source files left staged or unstaged.

---

## Self-Review

- [ ] **Spec coverage:** Phase A → Tasks 1-3 (body-rewrites). Phase B → Tasks 4-5 (data build + re-execution). Phase C → Tasks 6-8 (deletions + garch_utils decision). Phase D → Tasks 9-11 (docs + version + final gate).

- [ ] **Placeholder scan:** No "TBD", no "implement later". The `_collect_current_api` regeneration script in Task 10 assumes a specific helper-function name; if it's different, the step instructs adapting — explicit fallback, not a placeholder.

- [ ] **Type consistency:** Canonical lp DataFrame columns assumed are `h`, `beta`, `se`, `lo`, `hi`. Verify this matches `lp_hac` / `lp_iv` / `panel_lp` / `lp_smooth` / `lp_state_dep` by reading each canonical function's return shape during Task 1. If a function uses different column names (e.g. `ci_lo`/`ci_hi`), adapt the rewrites accordingly.

- [ ] **Memory pin compliance:** Task 1/2/3 explicitly cite `feedback_notebook_builders_paired` (edit both files in same commit, do NOT run the builder). Task 4 and 5 explicitly cite `feedback_long_nbconvert_no_subagent` (controller-only background). Task 8's sed-rename has a macOS-specific `-i.bak` pattern that's portable.

- [ ] **Signature verification:** Every canonical lp function signature in this plan was verified against live code on 2026-05-18 (see plan's pre-execution state section). Every legacy function signature was verified against legacy lp/lp_*.py at the same time.
