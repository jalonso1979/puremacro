# puremacro 0.46.x → 0.47.0 — consolidate and finish

**Status:** draft 2026-05-22. Target releases: **0.46.0** (release-gate), **0.46.1** (red-test cleanup), **0.47.0** (mechanical cleanup + Klein hardening).

## Why

The 0.43.0 → 0.45.0 cycle ([`docs/specs/2026-05-18-puremacro-043-canonical-promotion-design.md`](2026-05-18-puremacro-043-canonical-promotion-design.md), [`docs/specs/2026-05-18-puremacro-044-lp-retirement-design.md`](2026-05-18-puremacro-044-lp-retirement-design.md), CHANGELOG 0.45.0) retired the bulk of the legacy `svar/*` and `lp/lp_*` machinery. Four follow-ups were deferred across those releases:

1. **`puremacro/regress/lp.py`** (167 LOC) — independent pure-numpy LP implementation, not a re-export. Two `tools/` callers (`run_logurate_revision.py`, `run_paper_extensions.py`).
2. **`puremacro/lp/garch_utils.py`** (169 LOC) — kept as public name; 3 external callers (`tools/make_notebook_R1_02.py`, `tests/test_garch_utils.py`, the live `notebooks/R1_methods/R1_02_lp_menu.ipynb`).
3. **`ProxySVARResult` axis inconsistency** — `irf_point` ships as `(n, n, H+1)` while the other six `*Result` dataclasses in `var/identify/_results.py` use `(H+1, n, n)`. Cosmetic but lone outlier.
4. **`puremacro/dsge/klein.py` many-unit-eigenvalue Z-partition** — known-broken for systems like SW07; `smets_wouters._solve_F_sylvester` is the de-facto workaround (CHANGELOG 0.45.0 § Internal).

In parallel, the CHANGELOG carried a "Pre-existing failures (unchanged from 0.42.0)" line through 0.43.0 → 0.45.0 listing ~10 red tests across `test_qar_skewt_fci`, `test_narrative_body_extractor_coverage`, and `test_narrative_indices`. A 2026-05-22 spot-check ran those three files in isolation: **118 passed, 6 skipped, 0 failed**. The recorded failure list is stale at the file-list granularity — true red set in the full suite is unknown until baselined.

A separate latent problem: `pyproject.toml::version` was stale at `0.12.1` for ten releases until 0.41.0 caught it. That was a one-off catch by the consistency pass, not a recurring guard. There is no equivalent guard for `puremacro/__init__.py::__version__`, the latest `## X.Y.Z` heading in `CHANGELOG.md`, or `tests/fixtures/public_api_snapshot.json` going stale against a public-API change.

This spec ships a small release-gate (track D) first so the remaining cleanup (tracks B / A / C) lands under it, then closes the four deferred follow-ups across two follow-on releases.

## Scope

Three releases, gate-first:

- **0.46.0** — `tools/release_check.py` + `tests/known_failures.json`. No behavior change. The whitelist is seeded with whatever the full test suite emits as red on the 0.46.0 branch.
- **0.46.1** — Walk the whitelist. Per entry: fix, mark `skip(reason=…)` if env-gated, or document as real known-broken with a tracking note. Shrink whitelist toward zero. No notebook re-execution.
- **0.47.0** — Mechanical removals (A) plus Klein hardening (C). R1_02 + R1_04 notebooks re-execute under paired-builder discipline. `smets_wouters._solve_F_sylvester` deleted.

Each release passes `tools/release_check.py` before tag. 0.46.0 ships the script; thereafter the script is the gate.

## Pre-conditions

- 0.45.0 is on `main` (commit `4b9d851`).
- `data/processed/panel_Q.parquet` available locally (required for 0.47.0's notebook re-execution; not required for 0.46.0 / 0.46.1).
- `tests/test_pyodide_compat.py` green (already enforced; 0.41.0 restoration unchanged).
- `tests/fixtures/public_api_snapshot.json` reflects 0.45.0 (regenerated at commit `4b9d851`).

## 0.46.0 — Release gate

### `tools/release_check.py`

Single CLI entry: `python tools/release_check.py [--no-tests] [--report-only]`. Returns exit 0 on all gates green, exit 1 otherwise. Pure-stdlib + pytest (already a dev dep). Lives in `tools/`, never imported by the wheel.

Four gates, all run; the report prints which passed and which failed even if an earlier one fails (no fail-fast).

**Gate 1 — Test baseline.**

```
pytest puremacro/tests/ tests/ -m "not network" --tb=short -q
```

Collect failing-test `nodeid` set (pytest's full parametrized `nodeid` — `path::name[param]` — so each parametrized variant is its own whitelist entry). Compare to `tests/known_failures.json` entries. Allowed outcome: failing set ⊆ whitelist. Three sub-cases:

- Failing set == whitelist → pass.
- Failing set ⊃ whitelist → fail. New failures listed in report.
- Failing set ⊂ whitelist → warn, not fail. Names the previously-red-now-green tests so the maintainer can shrink the whitelist in the same release.

Skipped tests, xfail, and tests with the `network` marker are excluded from the comparison.

**Gate 2 — Pyodide contract.**

```
pytest tests/test_pyodide_compat.py -q
```

Hard-fail on any failure. This test already exists; the gate just ensures it isn't skipped during a release sprint.

**Gate 3 — Public-API snapshot.**

Regenerate the snapshot via the same helper `tests/test_public_api.py` uses, compare against `tests/fixtures/public_api_snapshot.json`. Any diff → fail with the diff printed. Snapshot is never auto-overwritten by the gate; maintainer regenerates explicitly when intentional.

**Gate 4 — Version sync.**

Parse three values:

- `pyproject.toml::project.version`
- `puremacro/__init__.py::__version__`
- The first `## X.Y.Z` heading in `CHANGELOG.md` (ignore `## X.Y.Z — date` suffix).

All three must be byte-equal as strings. Mismatch → fail with the three values printed side-by-side.

### `tests/known_failures.json`

```json
{
  "schema_version": 1,
  "entries": [
    {
      "nodeid": "tests/test_X.py::test_Y",
      "reason": "short human description of why this is red",
      "since_version": "0.X.Y",
      "owner_note": "where the fix lives or why it stays red"
    }
  ]
}
```

Seeded on the 0.46.0 branch with the actual baseline output of Gate 1 on `main` (unknown until run; CHANGELOG's listed ~10 is stale).

### Acceptance for 0.46.0

- `python tools/release_check.py` exits 0 on the release commit.
- `tests/known_failures.json` exists and matches the live red set.
- No `puremacro/*.py` file modified — release-gate touches only `tools/`, `tests/`, `CONTRIBUTING.md`, and changelog/version metadata.
- `CONTRIBUTING.md` § "How to run things" (or a new "Before tagging a release" subsection) cites `python tools/release_check.py` as the pre-tag step.
- CHANGELOG 0.46.0 entry documents the new gate and what each sub-gate enforces.
- Version strings bumped: `pyproject.toml`, `puremacro/__init__.py`, `tests/test_import.py`, CHANGELOG heading.

## 0.46.1 — Red-test cleanup

### Procedure

1. **Baseline commit.** First commit on `0.46.1` branch is the `known_failures.json` from 0.46.0, unchanged. Establishes the working set.
2. **Per-failure pass.** One commit per failure or per tight cluster (e.g., parametrized variants of the same test → one commit). Each commit either:
   - Fixes the test or the code under test (preferred).
   - Adds `@pytest.mark.skip(reason="...")` or `@pytest.mark.network` if the test is environmentally gated.
   - Re-classifies the failure as truly out-of-scope (rare; requires explicit `owner_note` justification and an issue-style tracking comment in the whitelist entry).
3. **Whitelist shrinks each commit.** Each commit edits `known_failures.json` to remove the resolved entries.
4. **Release-gate stays green throughout.** `release_check.py` runs Gate 1 incrementally — each commit reduces the allowed failing set.

### Out of scope for 0.46.1

- Notebook re-execution. No `.ipynb` outputs change.
- Public-API changes. Snapshot is unchanged.
- New functionality. Fixes only.

### Acceptance for 0.46.1

- `tests/known_failures.json::entries` is empty (best case) or contains only documented `owner_note`'d real-known-broken (each entry's `owner_note` describes its tracking plan).
- All four release-gate gates green.
- CHANGELOG 0.46.1 entry lists each failure resolved and how (one bullet per).

## 0.47.0 — Mechanical cleanup + Klein hardening

Two phases (A and C), three commits in A, one commit in C. R1_02 and R1_04 paper notebooks re-execute under paired-builder discipline.

### Phase A — Mechanical removals

**A1. `regress/lp.py` retirement.**

Pre-step audit:

- Compare `puremacro/regress/lp.py` signature against `puremacro.lp.panel.panel_lp` (and `panel_dk.panel_lp_dk`). If semantics differ in a way that affects the two `tools/` callers, the migration is non-trivial — pause and re-spec. If semantics align modulo kwarg renames, proceed.
- Verify `puremacro/regress/__init__.py` only exports the symbol(s) backing `regress/lp.py`. If so, the entire `regress/` package goes.

Migration:

- Rewrite `tools/run_logurate_revision.py` and `tools/run_paper_extensions.py` to canonical imports. Run both scripts end-to-end against existing input data; outputs must match prior runs within numerical tolerance (1e-10 on coefficients).
- Delete `puremacro/regress/lp.py`, `tests/test_regress_lp.py`, and `puremacro/regress/__init__.py` (and the directory if empty).
- Re-run `tools/release_check.py` — public-API snapshot diff must show the symbol removal cleanly; bump major-component of the public surface accordingly (per `CONTRIBUTING.md` § version table, removing a public symbol = major bump, but pre-1.0 conventions stay as minor + breaking-changes-flagged).

**A2. `garch_utils` → `_garch_utils` rename.**

- Internal-caller scan: no `lp/lp_*.py` should remain (deleted at 0.44.0); verify with grep.
- External-caller migration: rename `puremacro/lp/garch_utils.py` → `puremacro/lp/_garch_utils.py`. Update `tools/make_notebook_R1_02.py` and `tests/test_garch_utils.py` (rename test file to `test__garch_utils.py` and re-pin its imports).
- **Notebook re-execution.** Run `python tools/make_notebook_R1_02.py` after the rename. The builder emits a fresh executed `.ipynb`. Pin RNG seeds where re-execution would introduce gratuitous numerical churn (use the existing seed conventions in the builder).
- **Commit discipline:** the rename, the test-file rename, the builder update, AND the executed notebook must land in **one commit**. Per memory `feedback_notebook_builders_paired`, splitting them invites the next builder run to clobber the executed output.

**A3. `ProxySVARResult` axis flip.**

- Change `puremacro/var/identify/_results.py::ProxySVARResult.irf_point` from shape `(n, n, H+1)` to `(H+1, n, n)`. Same for `irf_lower`, `irf_upper`. Update `__post_init__` shape extraction (currently `H = shape[2] - 1`, `n = shape[0]`).
- Update `puremacro/var/identify/proxy.py` builder — the `point`, `lo`, `hi` arrays returned by the underlying bootstrap must be axis-transposed before constructing the `ProxySVARResult`. Re-use the canonical convention used by `cholesky.py` / `bq.py`.
- Update callers: grep `puremacro/` + `notebooks/` + `tools/` for `result.irf_point[`, `result.irf_lower[`, `result.irf_upper[` referencing a `ProxySVARResult` and rewrite the indexing.
- Notebook re-execution: `notebooks/R1_methods/R1_04_*.ipynb` paired builder runs and commits with the source change.
- Tests: add a single unit test asserting `ProxySVARResult.irf_point.shape == (H+1, n, n)` to lock the convention.

### Phase C — Klein many-unit-eigenvalue hardening

Lift `smets_wouters._solve_F_sylvester` into `puremacro/dsge/klein.py` as a fallback branch.

**C1. Detection.** In `klein.solve()`, after the QZ decomposition, detect whether the Z-partition is rank-deficient or ill-conditioned (concretely: the matrix that gets inverted to extract `F` from the Klein formula has condition number above some threshold, e.g. 1e10). When this triggers, the existing Klein closed-form `F` is corrupted by unit-eigenvalue lag states.

**C2. Sylvester fallback.** When Gate C1 triggers, solve the equilibrium Sylvester system instead. Port the existing logic from `smets_wouters._solve_F_sylvester` verbatim. Document inline why the fallback is needed (cite the SW07 case + paragraph from `smets_wouters.py:683–770`).

**C3. SW07 retraction of workaround.** Once C2 is in `klein.py`, `smets_wouters.py` calls `klein.solve()` directly and deletes its local `_solve_F_sylvester`. The 10 existing SW07 unit tests (consumption-Euler coefs, BK condition, qualitative IRFs, growth-rate IRFs, unit-sd impact values) must pass byte-equal — that is the regression contract.

**C4. New test.** Add `tests/test_dsge/test_klein.py::test_many_unit_eigenvalue_fallback` constructing a small synthetic system with two unit-eigenvalue lag states, verifying that the Sylvester branch triggers and that the policy function satisfies the equilibrium condition to ~1e-14.

### Acceptance for 0.47.0

- `puremacro/regress/` deleted.
- `puremacro/lp/garch_utils.py` renamed to `_garch_utils.py`; one commit also lands the R1_02 executed notebook.
- `ProxySVARResult.irf_point.shape == (H+1, n, n)` everywhere; R1_04 executed notebook committed in same commit as the axis flip.
- `smets_wouters._solve_F_sylvester` deleted; SW07 unit tests pass byte-equal.
- `tools/release_check.py` exits 0.
- `tests/known_failures.json` unchanged from 0.46.1 end-state (no regressions introduced).
- Public-API snapshot regenerated to reflect the three removals.
- CHANGELOG 0.47.0 entry covers A + C with breaking-changes section listing the three removed paths.

## Risks and mitigations

1. **R1_02 / R1_04 re-execution introduces numerical churn unrelated to the rename.** Pin RNG seeds at the builder level; the builders for these chapters already do this for bootstrap-bearing cells. If a cell numerically drifts unexpectedly, treat it as a finding and root-cause before committing the executed `.ipynb`.

2. **`regress/lp.py` is not actually equivalent to `lp.panel.panel_lp`.** A1's pre-step audit is the gate. If signatures or behavior differ, the spec needs amendment — do not force-fit. Treat the migration audit as a separate sub-task that may bail out.

3. **Klein hardening changes SW07 IRFs measurably.** The reference `smets_wouters._solve_F_sylvester` already produces residuals at machine precision; lifting it into `klein.py` should be a pure refactor of the same arithmetic. If C3's byte-equal regression contract fails, the cause is a porting bug, not an algorithmic disagreement — fix the port.

4. **0.46.0's whitelist is much larger than the 3 named files imply.** Treat that as diagnostic value: the gate's job is to surface the real red set. 0.46.1's scope expands to cover whatever's actually there. Document the actual number in the 0.46.1 CHANGELOG entry.

5. **The release-gate becomes burdensome and gets skipped.** Mitigation: keep it under ~150 LOC, single-file, no subprocesses beyond pytest. If the maintainer ever runs `git tag` without `release_check.py` it has already failed by design — there is no enforcement mechanism beyond habit. Document in `CONTRIBUTING.md` as part of "When to bump the version" → "Before tag: run `python tools/release_check.py`."

## Out of scope

- New estimators, fetchers, replications, or paper figures.
- Pyodide actual-runtime CI (the current `sys.modules` check stays; spinning up Pyodide in-loop is a separate spec).
- Type annotations / mypy.
- `regress/__init__.py` survival check beyond A1's audit — if it has other contents, document and defer.
- Notebook re-execution outside of R1_02 + R1_04. R1_01, R1_03, R1_05, T_us_national stay at their 0.43.0/0.44.0 executed outputs unless directly affected by A/C changes.
- Documentation site / mkdocs / Sphinx (separate spec if pursued).
