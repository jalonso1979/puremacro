# Notebook catalog plan — 23 September 2026

The catalog should grow by **replicating published DSGE and heterogeneous-agent
results against their published numbers**. Three goals point the same way: showing
the newest features, adding DSGE and VFI notebooks, and adding replications. A
replication notebook exercises capabilities that no notebook uses today. It also
closes the package's weakest evidence gap: the validation gallery has one published
target, and the replication gallery's DSGE densities are puremacro's own outputs.

The first two notebooks of that wave are implemented and verified:
[66 — optimal policy, a replication of Clarida, Galí and Gertler (1999)](../../notebooks/66_optimal_policy_cgg1999_replication.py)
([Spanish](../../notebooks/66_optimal_policy_cgg1999_replication_es.py)) checks four
policy solvers against the paper's closed forms, and turned up a documentation error
about the stabilization bias, now fixed. [67 — Bewley classics](../../notebooks/67_bewley_classics_replication.py)
([Spanish](../../notebooks/67_bewley_classics_replication_es.py)) re-solves Huggett's
and Aiyagari's tables; it reproduces Aiyagari's Table I exactly and every published
ordering, and shows that the tables' decimals carry numerical error (see
[below](#notebook-67-what-was-built-and-verified)). [68 — third order you can trust](../../notebooks/68_third_order_perturbation_trust.py)
([Spanish](../../notebooks/68_third_order_perturbation_trust_es.py)) checks the order-3
solver against two closed forms and live Dynare, and shows why tensor comparisons
are needed (see [below](#notebook-68-what-was-built-and-verified)).

The three notebooks exposed three library problems, all fixed on 23 September (see
[below](#library-fixes-that-followed)): `discretionary_policy` gained a conditional-loss
criterion, `solve_aiyagari_continuous` solves economies with negative interest
rates, and the order-3 moments are exact instead of first-order placeholders. The
last fix found that Dynare 8's own order-3 autocorrelations are wrong.

## Scope and method

- Three read-only surveys covered `puremacro/dsge/` (54 modules), `puremacro/vfi/`
  with `puremacro/models/`, and `puremacro/replication/` and `puremacro/validation/`
  with the bundled data and the notebook conventions. Each capability was matched
  against the call sites in `notebooks/`, `notebooks/course/`,
  `notebooks/macro_history_and_climate/` and `curso/notebooks/`.
- The claims used below were re-checked by hand: coverage greps, the notebook 35
  data range and stored output, the SW07 case code, the notebook 22 note, the
  README and hygiene-test rules, and the Dynare 8 example corpus on this machine.
- The survey itself changed no package code. The library fixes it led to are listed in
  [Library fixes that followed](#library-fixes-that-followed).

## Findings

### Capabilities without a notebook

Counts are files in the showcase, course and curso collections that call the
function (English and Spanish counted separately), before notebook 66.

| Area | No notebook | Light coverage |
|---|---|---|
| DSGE policy | Ramsey optimal policy (`ramsey_model`), `osr`, `lq_commitment` on its own | discretion vs commitment (45 only) |
| DSGE estimation | `bayesian_irf`, `prior_predictive`, `bootstrap_particle_filter` (order-2/3 likelihood) | SMC (curso T02_E only, 80 particles on a toy RBC); identification (45 only, a `summary()`); SW07 posteriors (40–100 draws) |
| DSGE analysis | `conditional_forecast`, `shock_groups_decomposition`, HP/bandpass theoretical moments, `simulate_surprise_shocks` | DSGE-VAR and news shocks (45, synthetic data) |
| DSGE solvers | `gensys` and a Klein-versus-gensys comparison, `build` with a callable plus `steady`, `dsge.stacked_newton` | order 3 on a real model (curso T02_B); Markov-switching and differentiable OccBin (46 only) |
| DSGE validation | the live Dynare 7 fixtures at orders 2 and 3 (`tests/fixtures/dynare_live/`, tensor errors at most 1.2e-13), `run_parity_suite` | — |
| VFI | DC-EGM, Smolyak, splines and Schumaker, `estimate_method_of_moments`, `farmer_toda`, `simulate_targeted_transfer` | OLG general equilibrium (prose in 03 only); two-asset HANK called directly (curso T06_C imports it); Krusell–Smith forecast accuracy |
| Newest (unreleased) | `trade.mrio`, `trade.condensed`, `trade.ces_newton`, `trade.household`, `trade.continuation`, `trade.stability`, `trade.dynamic` | — (notebook 64 mentions continuation in a comment) |

There is no tour of the 4.x releases; `00_whats_new` exists for 2.0 and 3.0 only.

### Replication evidence today

| Where | What it checks | Status |
|---|---|---|
| Validation gallery (107 cases) | Algorithmic agreement | One published target (Stock–Yogo critical values); the rest are internal, analytical, package or SciPy comparisons ([09-19 review](../2026-09-19-critical-evaluation/REVIEW.md)) |
| `replication/cases_dsge_estimation.py` | SW07 log posterior, Laplace and harmonic-mean densities | Recorded in the JOSS review notes as puremacro's own outputs, yet `docs/replication.md` calls them published |
| Same file, "structural parameters mode" | Nine Table 1A values at 25% | Scored at the best draw of a 200-draw chain (`argmax` of the trace), not an optimized mode; the targets look like posterior means (for example 5.74 and 1.83) — check against the paper |
| SW07 data | `dsge/_sw07_data.csv` | A one-time FRED rebuild (`tools/build_sw07_data.py`), not the original vintage; the fixture lives in `tests/fixtures`, so the cases fail from an installed wheel |
| `docs/replication.md` | Descriptions of the cases | Lists ξw and φπ while the code checks `csigl` and `crdy`; states Okun ≤ −0.60 while the code uses 0.5 |
| `cases_ha_dsge_causal.py` | Aiyagari and Huggett | Sign checks only, not the published tables |
| `examples/ramey_zubairy_2018_multipliers.py` | RZ multipliers | Reports 0.67 and 0.71 against the 0.66 and 0.71 it attributes to RZ Table 1; no notebook |
| `replication/data/kilian2009.csv` | Kilian (2009) data | Bundled, example and qualitative tests only; no notebook |

### Notebooks and docs that teach something wrong

| Item | Evidence | Suggested action |
|---|---|---|
| [35](../../notebooks/35_empirical_benchmark_replications.py) | `gali1999.csv` runs 1948–2026 while `load_gali1999` documents 1948Q1–1994Q4; hours enter in levels (the Christiano–Eichenbaum–Vigfusson specification, not Galí's differences); the prose hard-codes "−0.4% on impact" and "statistically significant" while the stored output prints `first-stage F (OP): 6.79 [WEAK]` | Rebuild around Kilian (2009), Ramey–Zubairy (2018) and Galí (1999) on its own sample, with published targets |
| [22](../../notebooks/22_continuous_time_hjb.py) | `docs/vfi_hjb_continuous.md` records that its numbers predate the 3.4.0 income-switching generator | Refresh, or fold into 56 |
| [course/04](../../notebooks/course/04_rbc_momentos_es.py) | `verify_dynare_parity(_mod_rbc, _mod_rbc.oo_dr)`, labelled "genuine parity", compares the model with itself | Compare with a real Dynare file, as curso T02_B does |
| [`docs/dsge_phase_c.md`](../../docs/dsge_phase_c.md) §1.4 (and the Spanish mirror) | Stated "The stabilization bias is strictly positive". The library measures it with unconditional losses against the timeless rule; notebook 66 shows it at −0.19 for β = 0.5 with serially uncorrelated cost-push shocks, and negative for β ≤ 0.8 at the baseline calibration (Jensen and McCallum 2002) | **Fixed 23 September** (both languages, also documenting the silent κ and weight defaults of `inflation_bias`), and `discretionary_policy(loss_criterion="conditional")` added the same day. The assertion messages of notebook 45 and `puremacro/examples/dsge_optimal_discretion.py` now name the calibration instead of stating a law |
| Order-3 moments and [`docs/dsge_higher_order.md`](../../docs/dsge_higher_order.md) §2.3-2.4 | `Order3PrunedSolution.theoretical_moments()` and `ergodic_moments()` return the first-order covariance with skewness 0 and kurtosis 3 hard-coded; the docs call them exact closed-form skewness and kurtosis, and the §2.4 example calls `model.solve(order=3, pruning=True)`, which raises `TypeError` (order 3 is always pruned; there is no `pruning` keyword) | **Fixed 23 September**: exact pruned-state-space moments, skewness and kurtosis NaN in `theoretical_moments()` and simulated in `ergodic_moments()` (which had raised `KeyError`), `solve(pruning=...)`, docs rewritten; advisory for 2.8.0 to 4.3.0 |
| `solve_aiyagari_continuous` | Its first consumption guess `r * a + w * z` is negative at negative interest rates, so a bracket that includes r < 0 stops on NaN; Aiyagari's σ = 0.4, ρ = 0.9, μ = 5 equilibrium is negative | **Fixed 23 September**: cash-on-hand guess at r < 0, automatic extension of the default bracket, `n_a` exposed, tests at the negative equilibrium |
| `RamseyResult.focs` | Prints raw AST reprs (`BinOp(op='+', …)`) | Render with the nodes' `to_latex`, or document `foc_nodes` |
| `docs/notebooks.md` row for 41 | "Ramsey transitions" is a Ramsey–Cass–Koopmans growth transition, not Ramsey policy | Reword |

The 09-20 notebook review's open items (28, 32, 33, 47, 49, 59) remain open.

## Proposed notebooks

Status: **done** means implemented and verified in this pass.

### DSGE replications

| # | Notebook | Shows | Oracle | Cost |
|---|---|---|---|---|
| 66 | **Optimal policy, CGG (1999)** — done | `discretionary_policy`, `lq_commitment`, `ramsey_model`, `osr` | Closed forms; 25 random calibrations | Low |
| — | **Order 3 you can trust**: the live Dynare fixtures (including the `ghxxu` permutation bug that paths could not see), then Tallarini (2000) or Basu–Bundick (2017) | `solve_dynare_3rd_order`, pruning, GIRFs, stochastic steady state, `run_parity_suite` | Live Dynare 7 plus the paper | Medium |
| — | **SW07 against the paper**: optimized mode against Table 1A; variance decomposition at the published mode; Laplace density against the published one; the effect of the Kalman initialization; the posterior under data revisions | `estimate_sw07`, `find_mode`, `laplace_mdd`, FEVD | The paper; Pfeifer's Dynare replication as a second route | High (also closes a JOSS blocker) |
| — | **Three samplers, one posterior** (Herbst–Schorfheide 2015 small NK) | RWMH, SMC, NUTS, `prior_predictive`, identification strength, `bayesian_irf`, three density estimators | The book's posterior tables | Medium–high |
| — | **Business-cycle moments done right**: BKK (1992) or King–Rebelo (1999) | HP and bandpass theoretical moments (the 4.1.0 fix), `simul_replic`, `preprocess_macro` | The paper (approximate) and Dynare's `bkk_1992.mod` (exact) | Low |
| — | **Aguiar–Gopinath (2007) for Mexico**, with Schmitt-Grohé–Uribe (2003) closures | Estimation on the package's Latin American data | The paper's Mexico estimates | Medium |
| — | **Davig–Leeper (2007) generalized Taylor principle** | `solve_ms_dsge` with the new convergence gate, mean-square stability | The published determinacy condition; the Farmer–Waggoner–Zha caveat | Low–medium |

Workflow notebooks without a paper: **DSGE from a Python function** (`build`,
`steady`, diagnostics, Klein versus gensys, LaTeX export; the starter the 09-20
review asked for), **SW07 policy scenarios** (conditional forecasts, shock
groups, fan charts) and **particle filter versus Kalman** (exact agreement in the
linear Gaussian case, then a pruned second-order likelihood).

### VFI and heterogeneous-agent replications

| Notebook | Shows | Oracle | Prerequisites | Cost |
|---|---|---|---|---|
| **Bewley classics**: Aiyagari (1994) Table II, Huggett (1993) tables, the Kopecky–Suen (2010) discretization comparison | `stationary_equilibrium`, Tauchen / Rouwenhorst / `farmer_toda`, numba and divide-and-conquer | The published tables | `aiyagari_steady_state` takes σ as the innovation s.d. with 5 states; confirm the paper's convention (unconditional s.d., 7 states); Huggett needs his two-state chain (the example is AR(1)) | Medium |
| **One model, many solvers** (Aruoba, Fernández-Villaverde and Rubio-Ramírez 2006) | Perturbation 1–3, VFI, EGM, collocation (Euler and Bellman), FEM, splines, Smolyak, deep macro | Brock–Mirman closed form; published Euler-error rankings | Pass `sigma`, not `gamma`, to splines and Smolyak (known issue) | Medium |
| **DC-EGM retirement** (Iskhakov, Jørgensen, Rust and Schjerning 2017) | `solve_dcegm`, upper envelope, taste shocks | Brute-force `FiniteHorizonProblem` on a fine grid | Most content already in `docs/dcegm.md` | Low–medium |
| **Forward-guidance puzzle** (McKay, Nakamura and Steinsson 2016) | Sequence-space Jacobian columns; targeted transfers | Complete-markets benchmark (flat by construction); the published figure | — | Low–medium |
| **Krusell–Smith (1998) and Den Haan (2010)** | Forecast-rule accuracy beyond R²; comparison with sequence space | Published forecast rules and accuracy tables; Dynare 8's `krusell_smith_1998.mod` as numbers only | The joint employment–aggregate transition (the current solver is a simplified variant) | Medium–high |
| Bigger bets: OLG social security (steady states), life-cycle method of moments, Arellano (2008) sovereign default | OLG GE, `estimate_method_of_moments`, a new default model | Published tables and moments | Sovereign default needs a new model | High |

### Newest features

- **Long–Plosser (1983) multisector RBC on `dsge.stacked_newton`**: its closed-form
  policy is an exact oracle, Hulten's theorem gives first-order GDP effects, and
  scaling the number of sectors shows why the block-tridiagonal preconditioner matters.
- **"Is this input–output table safe to use?"**: `trade.mrio` checksums,
  `regularize_table` reports and Collatz–Wielandt certificates, with the corrupted
  OECD 2020 export as the case study.
- **Three trade models, one tariff**: condensed, consistent accounting and nested-CES
  block Newton, with exact household welfare under four demand systems.
- **Where equilibria stop existing** (a sequel to 64): the condensed existence
  boundary, `sigma_path` stalling near unit elasticity, `reduced_stability`, and
  dynamic MRIO determinacy.
- **A 4.x tour**, once the unreleased version is tagged.

### Empirical replications ready now

Kilian (2009), Ramey–Zubairy (2018) and Galí (1999) on its own sample and
specification have bundled data and examples. Together they could replace
notebook 35's content.

## Replication contract

Notebook 66 is the template.

1. **A card** at the top: the paper, the table or figure, the data vintage, the
   oracle type (published number, independent software, closed form or
   self-generated), the tolerance and the solvers tested. The verdict is printed
   from a scorecard computed from the checks, never written into the prose.
2. **Two oracles where possible.** The paper is often approximate (different
   solution method or data vintage). An independent implementation gives exact
   numbers: Dynare through the MATLAB R2026a template, Moll's HACT codes or the SSJ
   package. Freeze their outputs as fixtures; never generate a reference with puremacro.
3. **Robustness away from the calibration.** Closed forms should hold at random
   admissible calibrations (66 uses 25 draws).
4. **A caveat section** that states what the check does not establish (66: the loss
   criterion, and linear-quadratic policy only).
5. **Licensing.** Dynare's example models are GPL-3: freeze numbers only, and write
   the models natively. Check the terms of SW's AER data, the Herbst–Schorfheide
   data and Pfeifer's DSGE_mod before bundling.
6. **Optional `ReplicationCase`.** It joins the default test suite, so it must run
   offline in under a second; heavy solves need frozen intermediate results. Update
   `docs/replication.md` and its Spanish mirror with the case.

## Constraints and registration checklist

- Number new showcases at the top level (66 onward): tests pin the course (22) and
  curso (62) notebook counts.
- Keep EN and ES code cells identical and the `# %%` cell counts equal.
- Build with the interpreter that the user-level `python3` kernelspec points at
  (currently the conda `work` environment). `ensure_kernel()` re-registers the
  kernelspec for whichever interpreter runs the tool.
- Register each notebook in `notebooks/README.md`, `docs/notebooks.md`,
  `docs/es/notebooks.md`, both READMEs (the "(00–NN)" range and a "(`NN`)" bullet
  are tested by `tests/test_docs_hygiene_audit_fixes.py`) and `CHANGELOG.md`.
- Move the SW07 fixture into package data before any SW07 replication notebook.
- Respect the documented limits: IFT gradients are representative-agent only, Deep
  Macro is experimental, and numbers computed on the bundled 77x11 table are
  software regressions, not OECD magnitudes.

## Recommended sequence

| Order | Notebook | Why | Cost |
|---|---|---|---|
| 1 | CGG optimal policy (66) | **Done** | Low |
| 2 | Bewley classics (67) | **Done** | Medium |
| 3 | Order 3 you can trust (68) | **Done** | Medium |
| 4 | SW07 against the paper, with `docs/replication.md` corrected | Credibility and a JOSS blocker | High |
| 5 | One model, many solvers | Five solvers without a notebook | Medium |
| 6 | Long–Plosser on stacked Newton–Krylov | Newest DSGE solver with an exact oracle | Medium |
| — | Rebuild 35 (§1.4 of `docs/dsge_phase_c.md` is done) | Remove wrong teaching | Low |

## Library fixes that followed

- **Conditional loss** (`puremacro/dsge/policy.py`, `_results.py`). Every
  `PolicyResult` carries `conditional_loss`, the discounted loss from the steady state:
  under discretion `trace(W X)` with `X = beta G X G' + N Sigma N'`, under commitment
  `L Sigma L' + beta F X_s F'` from zero lagged multipliers (the Ramsey plan).
  `loss_criterion="conditional"` makes `stabilization_bias` compare them. Checks: the
  CGG closed forms (discretion `alpha q^2 (alpha + lambda^2)/(1 - beta rho^2)`,
  commitment a sum of geometric series) at three fixed and twelve random
  calibrations, discounted squared impulse responses, and notebook 66's numbers
  (4.6530 and 3.1112 at the baseline; at beta = 0.5, rho = 0: -0.193 unconditional,
  +0.031 conditional). Notebook 66 now reads the fields and checks both losses at 25
  random calibrations (22 checks).
- **Negative interest rates** (`puremacro/vfi/continuous_distribution.py`). At r < 0 the
  first EGM guess is cash on hand; without `r_bracket`, positive excess supply at
  r = 1e-3 moves the lower end to -delta + 1e-3; `n_a` is a parameter. With `a_max=200`
  and `n_a=800` the packaged solver finds -0.087% for sigma = 0.4, rho = 0.9, mu = 5,
  0.0025 points from notebook 67's solver, which now records that check.
- **Exact order-3 moments** (`puremacro/dsge/_pruned_moments.py`). The pruned state
  `z = [xf; xs; xf⊗xf; xrd; xf⊗xs; xf⊗xf⊗xf]` with every innovation centred on its
  conditional mean (Dynare leaves `xf⊗u⊗u` uncentred), so the innovations form a
  martingale difference sequence and `Gamma_k = C A^k Var(z) C' + C A^(k-1) B Sigma D'`.
  Checks: the system reproduces the pruned simulator period by period to 2e-15; the
  claim's Isserlis closed form to 3e-15; Dynare 8's means and covariances on six
  models to 2e-13; Monte Carlo over 400,000 independent chains (|z| < 1.3).
- **Dynare's autocorrelations.** Dynare 8's order-3 `Corr_yi` drops the correlation
  between its innovation `xf(-1)⊗u⊗u` and earlier innovations. Emulating that omission
  reproduces its table to 5e-14; its own 4,000,000-period simulation gives 0.475 at
  lag 1 for `y` in `correlated_cubic.mod`, as puremacro does, against 0.330 in its
  table (for the claim to a cube: closed form 0.7634, Dynare 0.7085). The frozen numbers
  are in `tests/fixtures/dynare_order3_pruned_moments.json`. Centring the same innovation
  inside Dynare's own `pruned_state_space_system` leaves its means and variances unchanged
  and makes its autocorrelations exact (the claim's to 2e-16, six models to 5e-14). The
  draft issue is [DYNARE_ISSUE_order3_pruned_autocorrelations.md](DYNARE_ISSUE_order3_pruned_autocorrelations.md),
  with a `git apply`-ready patch in `dynare_order3_autocovariance_fix.diff`; posting it is
  the maintainer's decision.
- **Exact pruned moments at order 2.** `theoretical_moments(pruning=True)` on the order-2
  solution, forwarded by `stoch_simul` and `verify_dynare_parity`, uses the order-2 block
  of the same state space (no centring is needed there). It reproduces Dynare 8's
  `stoch_simul(order=2, pruning)` means, covariances and autocorrelations to 7e-14 on the
  five live models and a claim to x^2 (closed form: variance `2 A^2 v^2`,
  autocorrelations `rho^(2k)`). The default keeps Dynare's convention without `pruning`
  and is bit-identical to before.

## Notebook 66: what was built and verified

- **Content.** 19 checks: discretion (paths, lean against the wind, implied rule
  with γπ > 1, demand shocks offset), timeless commitment (closed-form δ path,
  targeting rule, symbolic route equal to the matrix route, zero IS multiplier),
  the simple rule (conservative central banker, `osr` recovering the simple-rule
  loss on an analytic ridge of rules), unconditional loss closed forms, and path
  and loss checks at 25 random calibrations. Every error is at rounding level
  (largest 9.2e-9, from the derivative-free `osr` search, against a 1e-6 relative
  tolerance). Two figures: responses with closed forms overlaid, and the variance
  frontiers beside the commitment/discretion loss ratio by discount factor.
- **Finding.** From the steady state commitment beats discretion at every discount
  factor tried. By the unconditional criterion that `.loss` and `stabilization_bias`
  report, the ranking flips for β ≤ 0.8 at the baseline calibration.
- **Verification.** Built with `tools/build_notebooks.py` for both languages;
  `--validate-only` and `--check` pass; the exercise assertion was tested at the
  corners of its advertised ranges (ρ ∈ {0, 0.95} × β ∈ {0.3, 0.995}) and at the
  prompts' values; the kernelspec was unchanged. Targeted tests (docs hygiene,
  bilingual docs, notebook integrity, build tooling, visual contrast, doc
  elevation, schema suites, trade tutorials): 702 passed, 2 expected failures. The
  slow `test_all_notebooks_execute`, which rebuilds every notebook, was not run.

## Notebook 67: what was built and verified

- **Targets.** Aiyagari (1994) Tables I and II and Huggett (1993) Tables 1 and 2,
  transcribed from scans of the published pages. Both tables check themselves:
  Aiyagari's saving rates equal δα/(r + δ) (his footnote 39) to within 0.005 points,
  and Huggett's annual rates equal q⁻⁶ − 1 within the rounding of the printed digits.
- **Table I, exact.** A seven-state Tauchen chain with a grid of three unconditional
  standard deviations reproduces all eight entries at the printed precision; widths
  2.0 and 2.5 do not. The entries are the coefficient of variation and serial
  correlation of the income *level*, which is why ρ differs between the σ panels.
  This also confirms that Table II's σ is the unconditional s.d. of log income (the
  innovation s.d. is σ(1 − ρ²)^½); `aiyagari_steady_state` takes the innovation s.d.
- **Table II, qualitative.** Every ordering in σ, ρ and μ holds in all 24 cells. The
  rates differ by −0.07 to +0.26 points, the largest in the bottom-right corner
  (published −0.3456%, replicated −0.090%). The numerical error is at most 0.005
  points under grid refinement, and the library's discrete-VFI
  `aiyagari_steady_state` converges to the same corner value (−0.125% at 800 grid
  points, −0.095% at 1,600; not run in the notebook, 47 s). Kirkby's replication
  study (*Computational Economics*, 2023; working-paper version of 27 February 2022)
  says the corner "contained numerical error" and reports a median absolute
  difference of 2.4% between his replication and Aiyagari's numbers (1.5% for Huggett).
- **Discretization.** Rouwenhorst with seven states moves the ρ = 0.9 rates by up
  to 0.81 points relative to Tauchen with seven; 25 Rouwenhorst states change the
  corner by 0.07 points. The corner's "true AR(1)" rate is about +0.65%, not −0.35%.
- **Huggett, qualitative.** The orderings hold (the rate rises as the limit loosens
  and falls with risk aversion, always below the full-insurance 4.17%). Prices differ
  by up to 0.003, which is up to 1.75 annual points because rates compound six
  periods; a discrete-VFI solve agrees with EGM to about 1e-5 on the worst cell (and
  to 1e-5 on two more cells checked outside the notebook with 1,000 grid points).
- **Not used.** Kopecky and Suen's (2010) Table 3 "ρ" rows: the chains are
  identical (their σ and σ_ε ratios are reproduced to four decimals in Table 2), but
  their persistence measure is neither the chain's autocorrelation, its second
  eigenvalue nor a regression slope, so it could not serve as a target.
- **Verification.** Both editions build, `--validate-only` passes, the notebook runs
  in about two minutes, and the exercise assertion holds at the corners of its
  advertised ranges (3 or 25 states × widths 1.5 or 3.5) and at the prompts' values.

## Notebook 68: what was built and verified

- **Exact oracles.** A claim to a cubic payoff, `y = beta E y' + x^3`, has the exact
  solution `y = A x^3 + B x`: every third-order tensor, including the risk terms
  `ghxss = 2 B rho` and `ghuss = 2 B` (0.2959502 and 0.3699377 at the calibration
  where 4.2.0 returned 0.3699377 and 0.4624221), matches to 3e-16 relative over twelve
  calibrations. Brock-Mirman's exact policy gives every tensor up to order 3 as a
  product of powers; the solver matches to 8e-16, with risk terms of 2e-18.
- **Dynare.** The ten live cases of `tests/fixtures/dynare_live` (five models, orders
  2 and 3) are re-solved with the manifest hashes checked: largest tensor error
  1.0e-13, pruned-path error 2.4e-13, sample-moment error 9.5e-14, reproducing the
  20 September report. The fixture paths come from Dynare's `stoch_simul(order=k,
  pruning)` and `simult_`, so this also checks the pruned simulator. The embedded RBC
  text hashes to the file Dynare ran.
- **Blind spot.** An antisymmetric `ghxxu` error of 0.0012928883 (the 20 September
  discrepancy) changes simulated observations by 1e-22; a symmetric one of the same
  size by 2.4e-6. Paths cannot detect the first; only an entry-wise comparison can.
- **Economics.** The RBC's precautionary capital (0.08% of steady state at the
  calibrated volatility) scales exactly with the variance, and the pruned
  decomposition used as a control variate confirms it within 1.4 standard errors.
  Responses to large shocks are asymmetric (3% of the peak at 3 s.d.), a second-order
  effect; third order changes them very little, and five times the volatility moves
  consumption's impact response by about 0.1%. For the cubic claim the whole response
  to small news is the third-order risk term, `B e`, and matches the closed form.
- **Not used.** A Monte Carlo GIRF averaging over future shocks and the
  zero-future-shock `girf()` agree within Monte Carlo error in this RBC (4e-4
  relative against standard errors of 3e-4), so the notebook uses the latter.
- **Exact moments (added with the library fix).** Section 5 checks the claim's
  variance and five autocorrelations against Isserlis' theorem (3e-16 at twelve
  calibrations), Dynare 8's frozen pruned means and covariances (2e-13), and Dynare's
  own long simulations (autocorrelations within 1.5e-3), and plots Dynare's table
  against both oracles.
- **Verification.** Both editions build, `--validate-only` passes, the notebook runs in
  about 13 seconds, and a run with the fixtures removed reports "8 of 15 checks ran and
  pass; 7 need the repository's Dynare fixtures". The exercise assertion holds at the twelve calibrations above and at five corners
  of its advertised ranges (up to B = 78.4 at beta = 0.99, rho = 0.95, s = 0.5).
