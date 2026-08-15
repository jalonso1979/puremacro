# puremacro notebook pedagogy: enriched showcases + build-your-own-index lab — design spec

- **Date:** 2026-06-01
- **Status:** design approved (brainstorming), pending spec review → writing-plans
- **Scope:** `puremacro/notebooks/` (no library code changes)
- **Branch:** `feature/regime-uncertainty-companion-phase2a`

## 1. Motivation & goal

The 12 showcase notebooks (`notebooks/01`–`12`, bilingual) are publication-quality but
**pedagogically terse**: they state intuition in 1–2 sentences and never write the
governing equations. `06_svar_identification` says "a reduced-form VAR captures dynamics
but not causal structure" without the VAR/structural equations, the identification
counting argument, or why Cholesky implies a recursive ordering. The "Try it" sections
are one line.

The user wants the example notebooks to (a) be **more verbose and pedagogical** — derive
the equations, build intuition, explain technique; (b) include **more notebooks where
students construct their own indices** with puremacro; and (c) convey **how comprehensive
the library is**.

**Approach (chosen):** "Enriched showcase in place." Deepen three flagships
(`01_wealth_inequality`, `06_svar_identification`, `11_narrative_uncertainty`) against a
reusable template, add a new `13_build_your_own_index` lab that tours four index-building
kernels from one library, and document the template so the remaining nine fan out
mechanically. This improves the examples in place (vs. leaving them terse), adds the
build-your-own thread, and conveys breadth — while preserving every hard guarantee
(executable, asserted, bilingual, Pyodide-safe). It complements, not competes with, the
course-companion roadmap (modules 2–8).

**Success criteria.**
1. The three flagships each carry: the method in LaTeX math, an `**Intuition.**`
   paragraph, a "read the output" interpretation, a fill-in-the-blank *Your turn* with a
   passing `assert`, and a "how comprehensive is this?" cross-reference.
2. A new `13_build_your_own_index` (+ `_es`) builds four indices on synthetic data
   (text→EPU, macro-panel→JLN-style, financial→FCI, cross-section→comovement premium),
   each with a fill-in cell, closing with a kernel↔entry-point recap table.
3. Bilingual parity holds: every touched/new EN `.py` has an `_es` twin, code
   byte-identical, prose in academic Spanish. `tests/test_bilingual_docs.py` green.
4. Everything stays green: all sources execute via `tools/build_notebooks.py`; the
   execute-all source-count gate is bumped to match.
5. A `notebooks/_TEMPLATE.md` documents the enriched-showcase structure for fan-out.

## 2. Non-goals (YAGNI)

- **Not** deepening the other nine showcases now (02, 03, 04, 05, 07, 08, 09, 10, 12) —
  the template makes them mechanical follow-ups.
- **Not** building course modules 2–8, slide decks, or AI-tutor cells (separate roadmap).
- **Not** adding real-data index examples — NB13 is fully synthetic/seeded/offline.
- **No** library code, no new public API, no new dependencies. Public-API snapshot
  (`tests/fixtures/public_api_snapshot.json`) must remain untouched.
- **No** translation-in-place (separate `_es.py` files, per the repo convention).
- **No** restructuring of the showcase numbering or the build pipeline.

## 3. The enriched-showcase template (`notebooks/_TEMPLATE.md`)

A markdown doc capturing the per-notebook structure, in order:

1. **Motivating question** — 1–2 sentences (keep today's opener).
2. **The method in math** — one markdown cell, the governing equations in LaTeX, compact
   (≈4–8 lines). Model primitives or estimator definition.
3. **Intuition** — a `**Intuition.**` lead-in paragraph mapping each equation to economics.
   No emoji; bold used sparingly (matches the course `**Recap.**` style and the
   AER-quality writing guidance in memory).
4. **Worked code** — the existing code, plus 1–2 inline comments per block narrating *why*
   (not what). No behavioural change; seeds/asserts preserved.
5. **Read the output** — a markdown cell interpreting the printed numbers and the hero
   figure explicitly (e.g. "r\* = 0.030 sits below 1/β−1 = 0.042 — the precautionary wedge").
6. **Your turn** — a code cell with a *working default* framed `# ← change this …`, a
   downstream `assert` that holds for the default, then 2–3 graded prose prompts
   (basic → stretch). The committed notebook executes green; the "blank" is conceptual.
7. **How comprehensive is this?** — 2–3 lines pointing to the other puremacro entry points
   that use the same machinery (the breadth lever).

House-style constraints (all from existing notebooks / memory): pure-numpy, fixed seeds,
inline asserts on headline numbers, the standard `_nbstyle` preamble (`apply_style()`,
`palette(n)`), Pyodide-clean imports.

## 4. Flagship deepenings (the math to add)

Each edit is **markdown-and-comments only** unless a *Your turn* needs a small,
green-executing default cell. No headline number changes ⇒ existing asserts stay valid.

### 4.1 `01_wealth_inequality` (`puremacro.vfi`)
Add: household Bellman `V(a,z) = max_{a'≥0} u(c) + β E[V(a',z') | z]` s.t.
`c = w·e^z + (1+r)a − a'`, `a' ≥ 0`; AR(1) for `z`; `u(c)=log c` (γ=1); stationary
distribution `μ` as the fixed point of the policy-induced transition; firm side
`r = α(K/L)^{α−1} − δ`, `w = (1−α)(K/L)^α`; market clearing `K = ∫ a dμ`; Huggett bond in
zero net supply `∫ a dμ = 0` with clearing `r < 1/β − 1`; permanent types = mixture over β.
Intuition: precautionary saving, the binding constraint (policy kink), why β-heterogeneity
fattens the top tail. Your turn: change `gamma`, the income-risk process, or the β spread;
assert the Gini moves the expected direction (the file already hints this). Comprehensive
pointer: vfi also does life-cycle/OLG (NB03), Hopenhayn firm dynamics (NB04), two-asset /
EGM / Epstein–Zin (NB05), Krusell–Smith GE transitions (NB02).
Verified imports in use: `aiyagari_steady_state`, `huggett_steady_state`,
`lorenz_and_gini`, `VFIProblem`, `solve_permanent_types`, `markov_stationary`.

### 4.2 `06_svar_identification` (`puremacro.var.identify`)
Add: reduced-form VAR(p) `y_t = Σ_{i=1}^p A_i y_{t−i} + u_t`, `E[u_t u_t'] = Σ_u`;
structural mapping `u_t = B ε_t`, `Σ_u = B B'`; the counting argument — `Σ_u` has
`n(n+1)/2` distinct entries, `B` has `n²` free elements, so identification needs
`n(n−1)/2` extra restrictions; **Cholesky**: `B = chol(Σ_u)` lower-triangular = exactly
`n(n−1)/2` zeros = a recursive ordering; **sign restrictions**: any `B̃ = B Q` with
`Q Q' = I` also satisfies `Σ_u = B̃ B̃'`, keep Haar-rotation draws whose impact IRF signs
match the prior ⇒ set (not point) identification. Intuition: why the ordering is an
economic assumption; why sign-restricted bands are wider. Read-output: impact signs +
band widths. Your turn (already strong): flip a sign in `restrictions` (informative
`RuntimeError`) and re-order the Cholesky variables; assert. Comprehensive pointer:
var/svar also do BQ long-run, proxy-SVAR / external-IV (Mertens–Ravn), FEVD, historical
decomposition (examples + R1 chapters). Verified imports: `cholesky_svar`,
`sign_restriction_svar` (fields `irf_point/irf_lower/irf_upper`, `irf_median/...`,
`n_accepted`, `summary()`).

### 4.3 `11_narrative_uncertainty` (`puremacro.narrative.indices`)
Add: the Baker–Bloom–Davis EPU indicator — a document is flagged iff it contains
`(≥1 economy term) ∧ (≥1 policy term) ∧ (≥1 uncertainty term)`; the index is the scaled
share of flagged docs per period; `bbd_100` normalization rescales to sample mean 100,
sd 50; MPU = flat keyword count, z-scored; the co-occurrence-vs-keyword-count *kernel*
abstraction. Intuition: why three-group co-occurrence suppresses false positives relative
to single-keyword counting. **Your turn upgraded to a fill-in that bridges to NB13**:
define your own three-group lexicon (e.g. climate / housing / migration) and pass it via
`epu(records, …, lexicon={…})`; assert it fires in-window on a synthetic corpus.
Comprehensive pointer: narrative ships ~70 connectors, LUI/LWUI/tone, multilingual
lexicons; NB13 generalizes the idea to non-text indices. Verified imports: `epu`, `mpu`,
`LEXICONS`, `_kernels.cooccurrence_kernel`, `_kernels.keyword_count_kernel`; records are
`(date, text, source_url, metadata)` 4-tuples; `normalize="bbd_100"|"zscore"`.

## 5. New lab `13_build_your_own_index` — "one toolkit, four index kernels"

Spine: every uncertainty/conditions index in puremacro reduces to one of four operations.
Four worked recipes on synthetic, seeded, offline data; each = worked example + fill-in
*Your turn* + `assert`. Recipes 1–2 carry the deepest scaffolding; 3–4 are lighter
"same pattern, another flavour." Pyodide-clean ⇒ the numeric `13_` prefix auto-bundles
into the playground (`build_playground.sh` globs `[0-1][0-9]_*.ipynb`).

| # | Build | Kernel | Verified puremacro API |
|---|---|---|---|
| 1 | Text → EPU | three-group co-occurrence | `narrative.indices.epu(records, country, language, normalize, lexicon=…)` |
| 2 | Macro panel → JLN-*style* uncertainty | common factors + conditional vol | `factor.pca_factors(X, k)` → `garch.garch11_fit(returns)` |
| 3 | Financial panel → FCI | first-PC projection + sign-normalize | `gar.fci(df, tightening_columns=…)` → dict `index/loadings/var_explained` |
| 4 | Cross-section → comovement premium | `σ·R` quadratic form | `sigma.SigmaObject(sigma, R, labels).cov_premium_var(w)` / `.mean_corr()` |

**Recipe 2 (simplified JLN), stated honestly in the notebook:** standardize a synthetic
macro panel `X (T×n)` with an injected high-volatility window; extract `k` factors via
`pca_factors`; for each series regress on the factors and take the residual as a one-step
forecast-error proxy; fit `garch11_fit` to each residual for a conditional-vol path;
average across series ⇒ macro-uncertainty index `U_t`. Label it explicitly as a
*pedagogically simplified* JLN — one-step, factor-projection residuals, GARCH vol — **not**
the full Jurado–Ludvigson–Ng multi-horizon stochastic-volatility machinery; point to
`puremacro.fetch.jln` for the published series. Assert `U_t` is materially higher in the
injected window.

Fill-in targets: (1) swap the lexicon; (2) choose `k` (or swap GARCH for a rolling-std
vol); (3) add an indicator column / set `tightening_columns`; (4) change weights `w` or the
correlation `R`. Each has a working default + a passing assert. Closes with the recap
table above (the comprehensiveness payoff) and a one-line pointer to the deepened NB11.

`13_build_your_own_index_es.py`: code byte-identical, prose in academic Spanish.

## 6. Mechanics & guardrails

- **Source of truth is the `.py`** (jupytext percent). Rebuild with
  `python tools/build_notebooks.py <stem>` (converts + executes, writes the committed
  `.ipynb` *with outputs*). Never hand-edit the `.ipynb`. Patch the `.py` and rebuild
  together (per memory: notebooks ↔ builders are paired; builders clobber outputs).
- **Bilingual parity:** every EN `.py` change is mirrored to its `_es` twin in the same
  change, code byte-identical, prose translated to native academic Spanish.
- **Fill-in cells must execute green** in the committed notebook: working-default value +
  `# ← replace with your own …` comment + a downstream assert that holds for the default.
- **Build long notebooks in the controller's background**, not a subagent (per memory:
  long nbconvert times out the Monitor-wait pattern). `01_wealth_inequality` (VFI GE) is
  the slowest but already builds today, so no new risk.

## 7. Tests & docs to update

- `tests/test_notebooks/test_notebooks_execute.py` — bump the hardcoded source count by
  **+2** (`13_build_your_own_index` EN + ES). (Memory: this gate hardcodes `len(srcs)==N`.)
- `tests/test_bilingual_docs.py` — passes automatically once the `_es` twin exists; verify.
- `notebooks/README.md` — add the `13_build_your_own_index` row to the table and a one-line
  note pointing at `_TEMPLATE.md`.
- No `tests/fixtures/public_api_snapshot.json` change (no library code).
- Optional/deferrable: a pointer to NB13 from `docs/examples_gallery.md` and the
  monorepo `TEACHING.md` (not required for green).

## 8. Phasing (for the implementation plan)

**Signature dump first (plan prerequisite).** Before locking any code, the plan must dump
the actual signatures/fields of the not-yet-verified surfaces and write against them, not
from memory: `aiyagari_steady_state`'s income-risk parameters (for the NB01 *Your turn*),
the `GARCH11Result` conditional-volatility field name (NB13 recipe 2), and the
factor-residual regression step (plain `numpy.linalg.lstsq`). Already verified this
session: `factor.pca_factors`, `gar.fci`, `sigma.SigmaObject` methods, `narrative.epu/mpu`,
`cholesky_svar`/`sign_restriction_svar`, the `_nbstyle` helpers.

1. **Template** — write `notebooks/_TEMPLATE.md` (the contract the rest follow).
2. **Deepen `11_narrative_uncertainty`** first (richest already; its upgraded *Your turn*
   seeds NB13) — EN + ES, rebuild, verify execute + asserts.
3. **Deepen `06_svar_identification`** — EN + ES, rebuild, verify.
4. **Deepen `01_wealth_inequality`** — EN + ES, rebuild, verify (slowest build).
5. **New `13_build_your_own_index`** — prototype each recipe to green first, then write
   EN + ES, rebuild, verify; bump the execute-all source count +2; update README.
6. **Full verification** — `build_notebooks.py --check` (or per-stem), the notebook
   execute gate, `test_bilingual_docs.py`; confirm no public-API drift.

Each phase is independently committable; bilingual twin lands in the same commit as its EN
source.
