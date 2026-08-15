# puremacro 0.48.0 — release polish: 1.0 roadmap + examples gallery

**Status:** draft 2026-05-22. Target release: **0.48.0**.

## Why

Two debts surfaced in the brainstorm after 0.47.0 closed the consolidate-and-finish arc:

1. **No declared path to 1.0.** The package has been pre-1.0 across ~30 releases (from at least 0.7.x through 0.47.0). The README states "Pre-1.0; APIs rename freely with consumers updated in the same commit." That convention is fine while there are no external consumers, but it has no end condition. Without a written 1.0 contract, the package will remain pre-1.0 by default — even after the release-gate machinery shipped at 0.46.0 makes 1.0-style discipline cheap.

2. **63 example scripts under `puremacro/examples/` have no execution-health surface.** They are only smoke-tested at the `import` level by `puremacro/tests/test_replication_smoke.py`. The `puremacro/examples/output/` directory holds 6 untracked PNGs from prior local runs — evidence that some examples work, with no record of which. New users landing on the repo cannot tell which examples are runnable, which need network, which need local data, and which are broken.

The brainstorm proposed six items (P1–P6) spanning research / maturity / DX. The user picked P1, P3, P4, P5 to take forward and ordered them P4+P5 → P3 → P1. This spec covers the first cycle: **P4 (collapsed to "1.0 roadmap doc only", PyPI publishing deferred) + P5 (examples gallery + opt-in Gate 5)**.

## Scope

One release. Two artifacts under one "release polish" banner:

- **`docs/1.0_path.md`** — single policy doc declaring what 1.0 means, the deprecation policy that activates at 1.0, the API-freeze contract, the gates that must be green to cut 1.0, and which subpackages are 1.0-blessed vs. experimental.
- **Examples gallery system** — `tools/render_examples_gallery.py` + generated `docs/examples_gallery.{md,json}` + committed `puremacro/examples/output/*.png` + a new opt-in **Gate 5** in `tools/release_check.py` activated by `--examples`.

PyPI wheel publishing is **explicitly out of scope** per the brainstorm conclusion. The "Gates to 1.0" section of the roadmap doc will list PyPI-publish-path as one of the criteria, but the actual publishing work is deferred until closer to 1.0.

## Pre-conditions

- 0.47.0 shipped (tag `v0.47.0` at commit `3446578`, pushed to `origin/feature/subnational-labor-uncertainty-us`).
- `tools/release_check.py` is the pre-tag gate with 4 gates (test baseline / Pyodide / public API snapshot / version sync).
- `tests/known_failures.json` is empty.
- Full-suite test count: 1307 passed, 21 skipped, 31 deselected (post-0.47.0 with the new shape-lock and Klein tests).

## Architecture

Two functional units, joined by the `0.48.0` release commit:

**Unit A — 1.0 roadmap (`docs/1.0_path.md`)**
- Static prose document.
- Sole audience: future maintainers and external would-be users evaluating the package.
- Five sections (detailed below).
- Reviewed at PR time, not gated.

**Unit B — Examples gallery system**
- `tools/render_examples_gallery.py` discovers `puremacro/examples/*.py` (excluding `__init__.py` and any `_*.py`), subprocess-runs each with a configurable timeout (300s default), classifies the result as PASS / SKIP / FAIL, captures PNGs the example wrote to `puremacro/examples/output/`, and emits two generated files: `docs/examples_gallery.md` (human-readable) and `docs/examples_gallery.json` (machine-readable, the gate input).
- `tools/release_check.py` Gate 5 reads `docs/examples_gallery.json` and fails on any FAIL entry. Activated by `--examples` flag; default 4-gate run unchanged.
- Maintainer runs the renderer explicitly (e.g., after fixing a broken example or before tag). The render is a separate step from the gate; the gate only **verifies** the committed JSON, never re-runs examples.

The two units are independent but ship together because both are documentation-tier polish.

## Unit A — `docs/1.0_path.md` (detailed design)

The doc has exactly five sections, in this order:

### § 1. What 1.0 means

Two-paragraph statement. Key claims:
- 1.0 is the point at which the public API stops renaming freely. After 1.0, every public-symbol rename or removal goes through a deprecation cycle (defined in § 2).
- "Public API" is the set of symbols listed in `tests/fixtures/public_api_snapshot.json` at the tag of 1.0. Anything not in that snapshot is implementation detail and may change without notice. Subpackages explicitly excluded from the 1.0 promise are listed in § 5.

### § 2. Deprecation policy (active at 1.0)

- A public symbol marked for removal stays callable for **one minor release** with a `DeprecationWarning`. Example: `puremacro.lp.foo` deprecated in 1.3.0 → still importable in 1.3.X (with warning) → removed in 1.4.0.
- The warning must name (a) the deprecated symbol's fully-qualified path, (b) the version of removal, (c) the canonical replacement (or "no replacement; opening issue X" if none).
- Behavioral changes to existing symbols that change return shape or argument semantics require a **major** bump.
- Internal refactors that don't change the snapshot or behavior require a **patch**.

### § 3. API freeze contract

Lists what `tests/fixtures/public_api_snapshot.json` does and does not contain:
- **Tracked:** `__all__` exports per subpackage (`puremacro.X.symbol` for every `X` reachable from `puremacro/__init__.py`'s import graph), plus dataclass field names in `result_classes`.
- **NOT tracked:** docstrings, internal modules (`_*.py`), `teaching/*`, `examples/*`, function bodies, signature names that aren't `__all__`-exported.
- At 1.0, Gate 3 (public-API snapshot) becomes a **stability gate**, not a "diff against the prior release" gate. Any diff is a deliberate maintainer action; the snapshot file is the source of truth.

### § 4. Gates to 1.0

Concrete checklist. The maintainer commits to all of these being satisfied before tagging 1.0:

- [ ] `tests/known_failures.json` empty for N (= 3) consecutive non-patch releases.
- [ ] Real Pyodide CI green (the P3 deliverable). Currently Gate 2 only checks `sys.modules`; 1.0 requires actually booting Pyodide and running a replication.
- [ ] PyPI publishing path proven (separate spec; deferred). One green test-PyPI release minimum.
- [ ] All 63 examples PASS or SKIP under the gallery system (P5). Zero FAILs.
- [ ] At least one external user has installed and used the package from a release artifact (not git clone). This unblocks the "are we discoverable enough" question.
- [ ] `ARCHITECTURE.md` refreshed within the last release cycle (no drift > 2 releases).
- [ ] A complete `RELEASING.md` covering the full release procedure (currently only `CONTRIBUTING.md` "Before tagging" partially covers this).

### § 5. Scope at 1.0 vs experimental

Three categories:

**1.0-blessed (stable API at 1.0):**
- `puremacro.var.*` (excluding `var/regime/*` which is research-experimental)
- `puremacro.lp.*` (excluding `lp/_garch_utils` which is private)
- `puremacro.inference.*`
- `puremacro.dsge.{klein,gensys}` (excluding `dsge.smets_wouters` which is replication-grade, may evolve)
- `puremacro.var.identify.*`
- `puremacro.volatility.*`
- `puremacro.{cycles, cointegration_modern, factor, korv_gmm, midas, spectral, synthetic_control, wavelet, realized_vol, labor_share}` (the absorbed-in-Phase-5 modules with frozen result classes)

**Research-experimental (out of the 1.0 promise):**
- `puremacro.narrative.*` — schemas may still evolve as the LUI/LWUI work matures.
- `puremacro.uncertainty.*` — early stage.
- `puremacro.models.*` — research models (DMP regime-dependent, etc.) tied to specific papers.
- `puremacro.smm` (if it lands as a separate subpackage by 1.0).

**Side-channel (out of the Pyodide promise + the 1.0 promise):**
- `puremacro.teaching.*` — intentionally wraps statsmodels/linearmodels/arch for MATLAB-parity comparisons. Documented as a teaching artifact, not a research-API surface.
- `puremacro.fetch.*` — depends on external HTTP services that can change schemas independently of puremacro's release cycle.

## Unit B — Examples gallery system (detailed design)

### `tools/render_examples_gallery.py`

Single-file Python script, ~300 LOC, pure-stdlib + subprocess. Lives in `tools/`, never imported by the wheel.

**CLI:**
```
python tools/render_examples_gallery.py [--only NAME] [--timeout SEC] [--limit N]
```
- `--only NAME`: re-render just one example (e.g., `--only bloom2009`). Other entries in the existing JSON are preserved.
- `--timeout SEC`: per-example timeout (default 300).
- `--limit N`: process only the first N examples (for fast iteration during development).

**Algorithm:**

1. **Discover** — list `puremacro/examples/*.py`, excluding `__init__.py` and any leading-underscore files.
2. **For each example** (or just the `--only` target):
   - Snapshot `puremacro/examples/output/` mtimes BEFORE the run.
   - Subprocess-run `python -m puremacro.examples.<name>` with `capture_output=True`, `timeout=300`, `cwd=<repo root>`.
   - Determine status from exit code + stderr + special markers:
     - returncode == 0 → PASS
     - first 10 lines of the example contain `# skip: <reason>` → SKIP (regardless of run; don't even subprocess in this case)
     - returncode != 0 + stderr regex matches one of:
       - `(HTTPError|URLError|ConnectionError|TimeoutError|requests\.exceptions)` → SKIP, reason "network unavailable"
       - `FileNotFoundError.*\.(parquet|csv)` → SKIP, reason "local data file missing"
     - subprocess.TimeoutExpired → FAIL, reason "timeout after 300s"
     - else → FAIL, reason = first 200 chars of stderr
   - Snapshot `examples/output/` mtimes AFTER the run; the diff is the figures attributed to this example.
3. **Aggregate** — build a dict keyed by example name with: `status`, `reason`, `runtime_s`, `figures` (list of relative paths), `last_run` (ISO timestamp).
4. **Emit:**
   - `docs/examples_gallery.json` — full dict + `schema_version: 1`.
   - `docs/examples_gallery.md` — Markdown gallery: top-level summary table (counts of PASS/SKIP/FAIL), then per-example sections sorted by status (FAIL first to surface problems, then PASS, then SKIP).

**Markdown per-example section template:**

```markdown
## bloom2009  ![PASS](https://img.shields.io/badge/status-PASS-green)

- **Runtime:** 12.4 s
- **Last run:** 2026-05-22T14:30:00Z
- **Figures:**
  - ![bloom2009_irf](../puremacro/examples/output/bloom2009_irf.png)

*Reason (for SKIP / FAIL only):*
*…*
```

**Idempotency:** running the renderer twice in a row produces byte-equal JSON for examples that ran deterministically. The Markdown's `last_run` timestamps will update each run; everything else is stable.

**What the renderer does NOT do:**
- Never edits example source files.
- Never amends git commits.
- Never deletes pre-existing PNGs (only adds new ones).
- Never auto-stages or commits its output — the maintainer commits explicitly.

### `docs/examples_gallery.json`

```json
{
  "schema_version": 1,
  "generated_at": "2026-05-22T14:30:00Z",
  "examples": {
    "bloom2009": {
      "status": "PASS",
      "reason": null,
      "runtime_s": 12.4,
      "figures": ["puremacro/examples/output/bloom2009_irf.png"],
      "last_run": "2026-05-22T14:30:00Z"
    },
    "fetch_fred_demo": {
      "status": "SKIP",
      "reason": "network unavailable",
      "runtime_s": 0.8,
      "figures": [],
      "last_run": "2026-05-22T14:30:00Z"
    }
    // ... 61 more
  }
}
```

### Gate 5 in `tools/release_check.py`

Added behind `--examples` flag (default OFF, so the existing 4-gate run stays fast).

**Logic:**
- Read `docs/examples_gallery.json`.
- If file missing → FAIL with "examples gallery not rendered; run tools/render_examples_gallery.py".
- If any entry has `status: FAIL` → FAIL, report lists each failing example name + reason.
- Stale check: if `generated_at` is older than any source-file `mtime` under `puremacro/examples/*.py`, **warn** (not fail). This surfaces "you edited an example but forgot to re-render" cases.
- Else → PASS.

**Report format:**

```
  Gate 5 (examples health): PASS — 30 PASS, 33 SKIP, 0 FAIL
```

or

```
  Gate 5 (examples health): FAIL — 1 example failed
    FAIL bloom2009 — RuntimeError: Sigma not PSD (last 200 chars of stderr...)
```

### `CONTRIBUTING.md` update

Add to the "Before tagging a release" subsection:

> For a full release check including the examples-gallery health, run with `--examples`:
>
> ```bash
> python tools/render_examples_gallery.py   # ~5-15 minutes
> git add docs/examples_gallery.* puremacro/examples/output/
> git commit -m "chore: refresh examples gallery"
> python tools/release_check.py --examples
> ```
>
> The `--examples` flag activates Gate 5, which fails on any example with status FAIL. The render step is slow; the gate itself just reads the committed JSON.

## Data flow

```
Maintainer
   │
   ├── edits puremacro/examples/foo.py
   │
   ├── runs `python tools/render_examples_gallery.py`     ──→  docs/examples_gallery.json
   │                                                            docs/examples_gallery.md
   │                                                            puremacro/examples/output/foo_*.png  (new/updated)
   │
   ├── inspects gallery markdown; commits if happy
   │
   ├── runs `python tools/release_check.py --examples`    ──→  reads docs/examples_gallery.json
   │                                                            verifies no FAILs
   │
   └── tags release if gate green
```

Gate 5 reads the JSON; it does NOT re-run examples. This separation keeps the gate fast (~milliseconds for the JSON read + comparison) and keeps the render explicit (the maintainer chooses when to re-render).

## Error handling

**Renderer:**
- One example crashing never kills the run — each subprocess is isolated, captured, and recorded.
- Timeout (default 300s) → SIGTERM → classify as FAIL with reason "timeout after 300s".
- An example that produces no `output/*.png` is fine — gallery entry just lists "no figures".
- Renderer never raises out of the per-example loop. A bug in the renderer's classification logic that crashes the whole process is caught at the test level (see Testing).

**Gate 5:**
- Missing JSON → FAIL with actionable message.
- Malformed JSON → FAIL with `json.JSONDecodeError` surfaced (per puremacro's "diagnostic over silent" norm).
- Stale JSON → warn, not fail.

**Roadmap doc:** no runtime; no error handling needed.

## Testing

### Renderer tests (`puremacro/tests/test_render_examples_gallery.py`)

Tests for the classification logic (pure functions, no subprocess), against mocked subprocess.run results:

- `test_classify_pass` — returncode 0 → PASS.
- `test_classify_fail_unknown_error` — returncode 1, stderr `ValueError: ...` → FAIL.
- `test_classify_skip_network` — returncode 1, stderr `urllib.error.URLError: ...` → SKIP, reason mentions network.
- `test_classify_skip_data_missing` — returncode 1, stderr `FileNotFoundError: ...panel_Q.parquet` → SKIP, reason mentions data file.
- `test_classify_skip_explicit_comment` — example source starts with `# skip: requires GPU` → SKIP, reason "requires GPU", subprocess never invoked.
- `test_classify_timeout` — subprocess.TimeoutExpired → FAIL, reason "timeout after 300s".
- `test_figure_capture_attributes_only_new_pngs` — when example A produces `a.png` and example B is then run, B's figures list does NOT include `a.png`. (Mtime-snapshot semantic test.)
- `test_markdown_render_golden` — given a synthetic dict of 1 PASS + 1 SKIP + 1 FAIL, the generated markdown matches a fixture file (timestamp lines tolerated as wildcards).
- `test_json_schema` — generated JSON has the expected top-level keys (`schema_version`, `generated_at`, `examples`); each example entry has all of `status`, `reason`, `runtime_s`, `figures`, `last_run`.

### Gate 5 tests (extend `puremacro/tests/test_release_check.py`)

- `test_gate5_all_pass_or_skip` — JSON with 2 PASS + 1 SKIP, no FAIL → gate passes.
- `test_gate5_one_fail` — JSON with one FAIL entry → gate fails, report names the failing example.
- `test_gate5_missing_json` — no file at `docs/examples_gallery.json` → gate fails with actionable message.
- `test_gate5_stale_json_warns` — JSON `generated_at` older than the newest example source mtime → gate passes but emits a warning line in the report.
- `test_main_summary_with_5_gates` — extends `test_main_emits_summary_pass` to verify the "all 5 gates PASS" path when `--examples` is set.

### No tests for the roadmap doc.

Pure prose. Reviewed at PR time.

## Acceptance criteria

1. `docs/1.0_path.md` exists with all five sections enumerated above, populated.
2. `tools/render_examples_gallery.py` exists, runs against all 63 examples without crashing, and emits the JSON + Markdown pair.
3. `docs/examples_gallery.md` is in the repo with one entry per example.
4. `docs/examples_gallery.json` is in the repo, matches the markdown.
5. `puremacro/examples/output/*.png` previously-untracked PNGs are now committed. (Pre-existing untracked: `bvar_fan_chart.png`, `dfm_nowcast_kalman.png`, `spectral_business_cycle.png`, `synthetic_control.png`, `vulnerable_growth.png`, `wavelet_business_cycle.png`. Additional PNGs from the new render added too.)
6. `python tools/release_check.py --examples` exits 0 (Gate 5 PASS), which requires every entry in `examples_gallery.json` to be PASS or SKIP. If first-render discovers FAILs, see Risks below.
7. `CONTRIBUTING.md` "Before tagging a release" subsection updated to document `--examples`.
8. CHANGELOG 0.48.0 entry covering both pieces.
9. Version strings bumped: `pyproject.toml`, `puremacro/__init__.py`, `puremacro/tests/test_import.py`, CHANGELOG heading.

## Risks and mitigations

1. **First-render discovers FAIL examples.** Some of the 63 examples may be FAIL when rendered for the first time (this is genuinely unknown until the renderer runs). If so:
   - If FAIL count is small (≤ 5), fix or `# skip:`-tag them in 0.48.0; expand scope and document each fix in the CHANGELOG.
   - If FAIL count is large (> 5), seed `examples_gallery.json` with the FAILs as known issues (a per-example `known_failure: true` field added to the schema), make Gate 5 tolerate known-failures (the same way `known_failures.json` tolerates known red tests at Gate 1), and queue a 0.48.1 to drain them. Same pattern as 0.46.0 → 0.46.1's pyramid.

2. **`examples/output/*.png` PNGs are heavy** at 100KB-1MB each. 63 examples × even 200KB average ≈ 13MB of repo growth in a single commit. Mitigation: pin `dpi=100` and `figsize=(8, 5)` defaults in the gallery wrapper; for examples that produce large multi-panel figures, accept the size — they're documentation artifacts and the repo is a research workspace, not a wheel.

3. **Network-dependent examples flap between SKIP and PASS** depending on connectivity. The gallery's `last_run` timestamp would update on every render even though the underlying example didn't change. Mitigation: encourage explicit `# skip: network` comments on network-dependent examples (they become unconditional SKIPs, deterministic across renders).

4. **Renderer subprocess is slow at first run** (potentially 30-60 min for all 63 examples including timeouts and slow Bayesian runs). Mitigation: the first run is a one-time discovery cost. Subsequent runs use `--only` to re-render only edited examples. Document this pattern in `CONTRIBUTING.md`.

5. **Gate 5's stale-JSON warning may be noisy** if maintainers commonly edit examples without re-rendering. Mitigation: pre-commit hook (optional, not in 0.48.0 scope) that prompts to re-render when `examples/*.py` is touched. For 0.48.0, the warning text just says "consider re-rendering".

6. **The 1.0 roadmap doc may overcommit** to gates that turn out to be impractical (e.g., "at least one external user installed from artifact" is hard to verify). Mitigation: the doc is editable; revisions in future releases are allowed. The doc declares intent, not contract.

## Out of scope

- **PyPI publishing** (`tools/release_publish.py`, the `twine upload` path). Explicitly deferred per the brainstorm. Listed in § 4 (Gates to 1.0) as a future requirement, not a 0.48.0 deliverable.
- **The other three pitches** from the brainstorm: P3 (real Pyodide CI), P1 (Bayesian DSGE estimation), P2 (mixed-frequency BVAR), P6 (numba JIT). Each has its own spec → plan → impl cycle.
- **Notebooks (R1_xx, R2_xx, T_xx)** in the gallery. The brainstorm chose `puremacro/examples/*.py` only. Notebooks have paired-builder discipline already; they don't fit the "small self-contained script" model the gallery is built for.
- **A pre-commit hook** that auto-runs the renderer on `examples/*.py` changes. Possible follow-on; not required for 0.48.0.
- **An HTML rendering** of the gallery (with thumbnails, sortable tables, etc.). The brainstorm picked Markdown-in-repo over HTML. If HTML becomes desired later, a second renderer reading the JSON can produce it without touching `render_examples_gallery.py`'s core logic.
- **CI** (GitHub Actions, etc.) running the gate automatically. The package's "no CI by design" promise stands; Gate 5 is a discipline tool, not enforcement.
