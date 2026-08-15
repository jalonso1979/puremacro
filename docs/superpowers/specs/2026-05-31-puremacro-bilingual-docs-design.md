# puremacro bilingual (EN+ES) docs & examples + pytest-cov fix — design spec

- **Date:** 2026-05-31
- **Status:** design approved (brainstorming), pending spec review → writing-plans
- **Scope:** `puremacro/` package subtree
- **Branch:** `feature/regime-uncertainty-companion-phase2a`

## 1. Motivation & goal

puremacro is framed for a Spanish course (*Macroeconomía Avanzada*) and its mission
is to be a public good for no-budget, often Spanish-speaking, students/researchers —
yet **all prose is English** (the 2026-05-29 audit called this the *biggest remaining
equity lever*). Goal: deliver a **broad bilingual layer** so a Spanish-speaking user
gets a full native-Spanish experience (README, docs, the browser-playground notebooks,
example navigation), with the English originals untouched. Plus a precursor test-infra
fix the user requested.

**Success criteria.**
1. `pytest --cov` works directly (no numpy double-import), and a coverage CI job exists.
2. A Spanish reader can: read `README.es.md`; read every user-facing doc in Spanish
   (`docs/es/`); run all 12 showcase notebooks with Spanish narration (in the browser
   playground); and navigate all ~70 examples via a Spanish gallery (+ Spanish docstrings
   on ~15-20 flagship examples).
3. English files are unchanged except a one-line `🇬🇧 EN | 🇪🇸 ES` language switcher.
4. Everything stays green: ES notebooks execute, ES examples run, full suite passes.
5. Spanish is native academic-register macro Spanish (not literal/machine translation).

## 2. Non-goals (YAGNI)

- No i18n framework / runtime language switching (static parallel files only).
- No translation of internal docs (`docs/superpowers/*`, `ARCHITECTURE.md`,
  `CHANGELOG.md`, `CONTRIBUTING.md`) — developer-facing, English.
- No `_es.py` copies of example *code* (gallery + flagship docstrings instead — chosen).
- No translation of code identifiers, API names, or maths.

## 3. Phase 0 — pytest-cov + conftest fix (the "do the tests" item)

**Problem (observed by the coverage-sweep agents):** running `pytest --cov` via this
repo's `conftest.py` under Python 3.13 / numpy 2.4.x can raise
`ImportError: cannot load module 'numpy...' more than once per process` — coverage's
auto-start `.pth` (e.g. `a1_coverage.pth` sitecustomize) begins measurement before
numpy is imported, and the conftest's numpy import then collides. (NB: the controller's
full-suite `pytest -m "not network" --cov` runs *did* work, so the failure is
invocation/order-dependent — likely single-file or `--noconftest` invocations.)

**Approach (investigate → fix):** reproduce the exact failing invocation; inspect
`tests/conftest.py` + the coverage `.pth`/`COVERAGE_PROCESS_START` setup; apply the
minimal structural fix (options, in priority order, to be confirmed by the repro):
(a) ensure numpy is imported once before coverage starts (or vice versa) via the
conftest / a `coverage` config `[run] disable_warnings`/`concurrency` setting;
(b) move/guard the conftest import that triggers the early-numpy path;
(c) pin the coverage invocation in `pyproject [tool.coverage.run]` (`source`, `omit`)
so the `.pth` auto-start is not double-counted. Verify `pytest --cov=puremacro -q`
(full + single-file) runs clean. Then add a **coverage CI job** (non-blocking initially)
running `pytest --cov` and reporting the total.

*(This phase is a bug fix; the exact patch is determined by the reproduction. It is
sequenced first so the new Spanish notebooks' coverage can be verified normally.)*

## 4. Phase 1 — bilingual structure (separate Spanish files; English intact)

### 4.1 README + docs
- `README.es.md` — full native-Spanish adaptation of `README.md` (all sections: qué
  incluye, instalación, ejecutar los LLM gratis en local, compatibilidad Pyodide,
  inicio rápido, documentación, convenciones, estado). Add a one-line switcher at the
  TOP of both `README.md` and `README.es.md`.
- `docs/es/<name>.md` — Spanish versions of the 7 user-facing docs:
  `CREDENTIALS`, `CACHE_DB`, `CONNECTOR_HEALTH`, `SIGNAL_CONTRACT`, `examples_gallery`,
  `1.0_path`, `lexicon_review`. Add the switcher line atop each English original.

### 4.2 Notebooks (the equity centerpiece)
- For each of the 12 notebook sources `notebooks/<NN>_<name>.py` (01-11 + `local_llm_uncertainty`),
  create `notebooks/<NN>_<name>_es.py`: **identical code**, **Spanish markdown narration**.
  Built to `.ipynb` by `tools/build_notebooks.py` (globs `notebooks/*.py` minus `_*`).
- The playground glob `[0-1][0-9]_*.ipynb` will include the numeric ES ones
  (`01_..._es.ipynb` … `11_..._es.ipynb`) → bilingual browser playground. The
  non-numeric `local_llm_uncertainty_es.ipynb` stays out of the playground (matches its
  EN counterpart, which is desktop-only) but is still built + execute-gated.
- The execute-all slow gate `tests/test_notebooks/test_notebooks_execute.py` count assert
  **12 → 24**.

### 4.3 Examples (gallery + flagship — chosen)
- `docs/es/examples_gallery.md` — the Spanish examples gallery describing **all ~70**
  examples (mirrors `docs/examples_gallery.md`; this is the same `docs/es/` file already
  listed in §4.1, called out here as the examples deliverable).
- Bilingual module docstrings (add a Spanish block below the English; code unchanged) on
  a **flagship ~15-20** spanning the toolbox, e.g.: `sign_restrictions_uhlig`,
  `lp_smooth_demo`, `la_lp_pmw_demo`, `did_callaway_santanna_demo`, `dcc_volatility`,
  `har_realized_vol`, `vulnerable_growth`, `bloom2009`, `narrative_indices_demo`,
  `narrative_local_llm`, `bvar_fan_chart`, `ms_var_business_cycle`, `tvp_var_demo`,
  `synthetic_control_demo`, `dfm_nowcast_kalman`, `dsge_rbc_klein`, `sigma_decomposition`,
  `labor_flows_demo`, `wavelet_business_cycle`. (Final list pinned in the plan.)

## 5. Phase 2 — Spanish quality bar
Native academic-register Spanish; correct macro/econometrics terminology (*proyección
local, restricciones de signo, identificación, volatilidad realizada, incertidumbre,
crecimiento en riesgo, descomposición, choque/shock*). Keep code, identifiers, API
names, citations, and maths verbatim. EN and ES kept structurally in sync (same headings/
sections) so they don't drift.

## 6. Phase 3 — execution (parallel workflows, batched)
Same Drive-safe pattern proven this session: agents create ONLY their new ES file(s),
**never run git**; the controller commits the verified aggregate per batch and runs the
suite. Batches:
- **B1 — README + docs** (~8 agents): `README.es.md` + the 7 `docs/es/*.md`.
- **B2 — notebooks** (12 agents, one per notebook): create `<NN>_<name>_es.py`, build the
  `.ipynb`, confirm it executes (Mock fallbacks where needed).
- **B3 — examples** (a few agents): the Spanish gallery + flagship bilingual docstrings.
Controller, after each batch: audit `git status` (only intended new/edited files, no agent
commits), build/run, commit.

## 7. Phase 4 — testing & verification
- Phase 0: `pytest --cov` runs clean (full + single-file); coverage CI job added.
- Notebook execute-all gate passes with 24 notebooks (count bumped); the ES notebooks
  execute green (Mock fallbacks; no network).
- A light **bilingual-parity test** (`tests/test_bilingual_docs.py`): every
  `notebooks/<NN>_<name>.py` has an `_es.py` sibling; `README.es.md` and each
  `docs/<name>.md` user-facing doc has an `docs/es/` (or `.es`) counterpart; the EN
  switcher line is present. Pure file-existence checks (fast, no network).
- The switcher edit to English docs is additive (a link line) — not a translation, so it
  honors the repo's no-translate-in-place rule.
- Full default suite stays green; `playground/build_playground.sh` still builds (now with
  the ES notebooks).

## 8. File-by-file (for the plan)
New: `README.es.md`; `docs/es/{credentials,cache_db,connector_health,signal_contract,
examples_gallery,1.0_path,lexicon_review}.md` (the Spanish docs incl. the examples
gallery); `notebooks/<NN>_<name>_es.py` ×12 (+ built `.ipynb`); `tests/test_bilingual_docs.py`.
Modified (additive): English `README.md` + the 7 user-facing docs (switcher line); the
flagship ~15-20 `examples/*.py` (Spanish docstring block); `tests/test_notebooks/
test_notebooks_execute.py` (count 12→24); `pyproject.toml` / CI (Phase 0 coverage); plus
the Phase-0 conftest/coverage fix.

## 9. Risks
- **Spanish quality.** Mitigation: native academic register + a controller review pass of
  a sample from each batch; technical terms kept consistent (a short glossary in the spec/plan).
- **Phase-0 root cause unknown until reproduced.** Mitigation: investigate first; the fix
  is localized to conftest/coverage config.
- **Playground/gate growth** (12→24 notebooks ≈ doubles the execute-gate time + playground
  size). Acceptable; the ES notebooks reuse the EN code (fast, Mock fallbacks).
- **Switcher edits touch English docs.** Additive one-liner only; no content translated.

## 10. Decision log
1. "do the tests" = fix pytest-cov + conftest (+ coverage CI). 2. Bilingual scope = broad
(README + all docs + all 12 notebooks + examples). 3. Examples = Spanish gallery (all ~70)
+ flagship bilingual docstrings (~15-20). 4. Format = separate Spanish files, English
intact + switcher. 5. Execution = parallel workflow batches, agents no-commit, controller
commits/verifies.
