# Notebook review and roadmap — 30 September 2026

**Fix the latest notebooks first; several of them teach something false.** Notebooks 62–68 all run and reproduce their stored outputs, but 63, 64 and 65 make claims their own numbers contradict, and 66–68 carry four specific errors (a mislabelled mechanism, an exercise built on a false premise, a misattributed result and an unsupported explanation of the replication gaps). The review also surfaced nineteen library defects, confirmed by independent reproduction, that reach existing notebooks, course lessons and the JOSS replication evidence. The best new material is replications against **independent** oracles that already exist and were read this session (Stata, R packages, published tables), several of which double as the regression tests those defects lacked.

This builds on [the 23 September catalog plan](../2026-09-23-notebook-catalog-plan/PLAN.md) and [the 20 September review](../2026-09-20-notebook-review/REVIEW.md); items already there are marked *PLAN* or *REVIEW*.

## Scope and method

- **Latest notebooks.** 62–68, English and Spanish sources and executed artifacts. 66–68 were reviewed through four lenses (claims against evidence; execution and the exercise; teaching and packaging; replication fidelity), 62–65 through the first three. Every notebook was executed with the build tool's jupytext command into scratch, and exercises were run at the corners of their advertised ranges.
- **Adversarial verification.** 360 findings; each major one was re-checked by three independent verifiers (reproduce, interpretation, counter-evidence), minor ones by one. 355 survived, most with corrected severity or wording; 5 were refuted. See [findings.json](findings.json).
- **Survey for new material.** Fifteen sweeps (library coverage by subpackage, examples without notebooks, empirical and structural replication targets, graduate theory and empirical syllabi, an exercise audit of all notebooks, the ITAM course, newest features, labs, open items from earlier reviews), then two rounds of a completeness critic. 242 raw proposals became 131 vetted ones ([ideas.json](ideas.json)) and about 200 vetted exercises ([EXERCISES.md](EXERCISES.md)).
- **Library defects and oracle numbers.** Twenty-one defect claims that surfaced in the survey were each re-verified by three independent agents, and eleven published numbers that the roadmap relies on were checked against the fetched sources ([defects_and_oracles.json](defects_and_oracles.json)).
- **Not done.** No existing repository file was changed (this folder is new and untracked). The full test suite was not run. One agent's call to a `fetch_*` function wrote a cache entry into `data/raw`; the manifest was restored byte for byte and the cached file removed.

## 1. The latest notebooks

| Notebook | Runs | What is wrong (verified) |
|---|---|---|
| 62 flexible trade | 4 s, reproduces | The bundled 77×11 table is called "OECD", "empirical" and "full-data" although `docs/ADVISORY.md` classifies it as a regression fixture derived from a corrupted export. The one GE result (B bears about all of A's tariff through its terms of trade) is never interpreted, and it sits just above a singularity near σ ≈ 1. The solver-path paragraph does not say that flexible GE exists only under the legacy closure, which flips the relative-price response on this same economy. The rewrite exists only as uncommitted edits. |
| 63 trade wars | 14 s, reproduces | Each country's welfare rises monotonically in its own tariff (to at least 500%), so every "optimal tariff" and "Nash candidate" is the imposed ceiling; the notebook never says so. Payoffs change when the countries are listed in the other order (numeraire foreign-saving closure). Consumption-only EV counts crowded-out investment as a gain, so even a symmetric war "helps" both. No trade economics in the intuition; Read-the-output reads no number. |
| 64 singularities | 7 s, reproduces | "Standard Newton divergence" prints a correct step of 7e-7 as `0.0000 (pathological step)`; J_x "explodes" in the legend but its plotted condition number peaks at 41 under a 10⁴ line; the fold-traversal assert is already true at step 0; Experiment 3 and the solver docstrings still claim a fold at σ = 0.1238 where none exists and plain Newton converges; `solve_cyprus_manifold_step` reports the residual (4.4e-16) of a vector it does not return (the returned step has residual 3.2) and index 14 is Cameroon, not Cyprus; the prohibitive-tariff check accepts any `ValueError`, and 4.3.0 shipped it passing on an unresolved certificate. The corrupted table is called empirical. |
| 65 GVC welfare | 3 s, reproduces | The headline EV depends on country order: A's EV is +0.000892 listed first and −0.087594 listed second; `foreign_saving_units="world_income"` gives 0.006888 either way. The price/factor/transfer split depends on the numeraire. The three "provider layouts" are the same matrix (max difference 6e-17). Your turn has no code cell, and its CES prompt leads into a singular region (σ ≈ 0.4–0.7). |
| 66 CGG (1999) | 6 s, reproduces | Closed forms are right (checked against the JEL text and re-derived). But the robustness claim covers four solvers when only `discretionary_policy` and `lq_commitment` run at the random calibrations; `osr` misses its own 1e-6 tolerance at 6–7 of the 25 draws. The "two independent routes" to commitment share the same matrices. "Discretion beats the timeless rule at low β" is credited to Jensen and McCallum (2002), which does not report it (it compares the timeless rule with an alternative rule, at β = 0.99 and 0.98 only); Sauer (2010, Prop. 2) is the source. |
| 67 Bewley classics | 131 s, reproduces | Transcriptions are exact (all cells checked against the scans). The "likely sources" of the gaps are wrong for Huggett: his p. 962 says the 0.0025 excess-demand criterion keeps rates within a tenth of a percent, and excess demand at his printed prices is 10 to 1990 times that. "Gaps exceed our numerical error" is false for 3 of 24 Aiyagari cells. The second oracles are puremacro code, while Kirkby's (2023) VFI-Toolkit numbers exist and agree (his 27-state Aiyagari table within 0.02 pp in all 24 cells). Several asserts are guaranteed by the `brentq` bracket and cannot fail. |
| 68 third order | 14 s, reproduces | The ergodic-mean shift in capital is called "precautionary capital", but the precautionary term for capital is negative (g_σσ,k = −9.8e-5; the risky steady state is −0.0045%, the ergodic mean +0.0785%): it is a curvature effect. Exercise 1 asks why "both" risk terms flip sign when ρ < 0, but g_xσσ = 2Bρ ∝ ρ² stays positive. The blind-spot lesson is overstated: a reference-free symmetry test catches the antisymmetric g_xxu error, and that error changes no path, response or moment. `rho_claim = 0` is inside the advertised range and crashes. |

**Strengths worth copying.** 66–68 set the standard the rest of the catalog should meet: a replication card, a scorecard whose verdict is computed from the checks, closed forms re-checked at random calibrations (66, 68), transcriptions checked against their own identities before use (67), negative controls that make an exact match informative (67's Table I: only Tauchen widths 2.975–3.035 match), SHA-256 fixture provenance and a graceful no-fixture mode (68). 68's finding that Dynare's order-3 autocorrelations are wrong was re-derived independently from Dynare 8's source and holds.

### Fixes, in priority order

**P0 — teaches something false or reports an arbitrary headline.**

- **64:** drop the Newton-divergence demo (or replace it with natural-parameter continuation that genuinely fails past the fold); relabel the J_x curve; make the fold test able to fail (one sign change of τ_λ, then check x₁ against −√λ); remove "σ_fold = 0.1238" from the notebook and from `trade/solver.py:13-15, 1793-1796`; report ‖S dw − rhs‖ for every returned step and compute `best_res`/`conv` after the clamp in `solve_cyprus_manifold_step` (fix `idx_cyp`); assert a *certified* Hawkins–Simon violation. Longer term, rebuild 64 as PLAN's "Where equilibria stop existing".
- **65 and 63:** solve with `foreign_saving_units="world_income"` and assert the result is unchanged when the countries are swapped (63 first needs `_hicksian_policy._Context` to forward the closure); say that the channel split depends on the numeraire. In 63, state that the optimum is the ceiling and why (Leontief final baskets), or recalibrate to get an interior equilibrium.
- **62, 64 (and 61):** call the bundled table what `docs/ADVISORY.md` says it is and link the advisory.
- **68:** relabel "precautionary capital" as the ergodic (risk-adjusted) mean and show g_σσ,k < 0; reword Exercise 1 ("which risk term changes sign, and why is g_xσσ ∝ ρ²?"); rewrite the blind-spot lesson and add a symmetry-residual row.
- **67:** call the Huggett gap unexplained, print excess demand at the published prices, move Kirkby to section 3 (he supports the chain effect), replace the blanket "beyond numerical error" with per-cell counts.
- **66:** scope the robustness claim to what was tested (or add `ramsey_model`, the conservative banker and `osr` to the loop and report `osr` against its own tolerance); correct the citations in the notebook and in `dsge/policy.py:461`, `dsge/_results.py:2629` and `docs/dsge_phase_c.md:73` (working-tree lines only).

**P1 — misleading, or a check that does not check.** Read-the-output cells in 62–65 interpret no printed number (template item 5); Your-turn cells in 62–65 are knob-turns or prose (62's and 63's asserts cannot fail, 65 has no cell); 67's asserts guaranteed by the bracket; 68's variance-law row is an identity labelled "exact"; 66's `stabilization_bias` silently returns 0.0 when the commitment solve fails, and its `timeless` flag is ignored; 67 should add Kirkby's numbers as a frozen second oracle (numbers only; GPL code).

**Packaging.** The twelve files of 66–68 are untracked while seven catalog sites and the CHANGELOG already link them; 62's rewrite is uncommitted. Stray `Legend` reprs and unrounded floats make every rebuild differ (63, 65–68). Commit by path, one notebook at a time, after checking `git log origin/main` for the parallel session.

## 2. Library defects the review surfaced

Each was reproduced on the working tree by one verifier, checked against the primary source by a second, and searched for counter-evidence by a third. "Confirmed" means all three agreed; "partial" means the defect is real but narrower or different in scope than first claimed.

| Defect | Status | What is wrong | Teaching affected |
|---|---|---|---|
| Two-asset HANK (`models/hank_sequence_space.py:1736-2340`) | confirmed, high | The household loop has no convergence test (a limit cycle, sup-norm change 0.39 after 3,000 iterations) yet reports `converged=True`; all illiquid wealth sits at `a_max` because β(1+r_a) > 1; the fake-news Jacobians have no anticipation terms (`c_next` is the steady state) and no date shift for r_b; the intertemporal budget leaks. The sign of C after a rate cut depends on the horizon: it falls at T = 16, 30 and 40 and rises at 20, the value the unit test happens to use | **46** prints C₀ = −0.0038 while its text says C₀ > 0 (EN and ES); `hank_two_asset.mod`; `tests/test_hank_two_asset.py:203`; CHANGELOG and ARCHITECTURE describe it as working |
| SW07 model (`dsge/smets_wouters.py:495-507`) | confirmed, high | Price and wage markup shocks omit their MA terms and arrive one period late (anticipated shocks); `cmap`/`cmaw` are estimated but never enter; hours lack the ×100 (`tools/build_sw07_data.py:85`); the replication targets −1673.72, −1686.09, −2524.36 are puremacro outputs presented as published, and the "mode" targets are Table 1a posterior means | 41, 42, `docs/replication.md`, the JOSS paper |
| `.mod` model-local `#` variables (`dsge/_parser.py:1659-1665`) | confirmed, high | Parameter-only locals are folded to constants at load, so every re-solve at new parameters (estimation, `osr`, widgets, SMC, gradients, `load_mod(params=)`) keeps the old values; in SW07 `constebeta` has an exactly flat likelihood | 41, 42, T02_E |
| `identification()` | confirmed, high | Evaluates at a mixture of the user's point and the calibration; the one-sided fallback subtracts the wrong baseline | `estimate(check_identification=)`, `docs/dsge_estimation.md`; 45 calls it at the calibration, where the defect does not bite |
| `ms_var_fit` | confirmed, high | The shared AR step ignores regime covariances, so the likelihood can fall across EM iterations while `converged=True`; it fits MSIH rather than Hamilton's model | 16, example `ms_var_business_cycle`, the docstring's Hamilton claim |
| `ab_gmm` / `windmeijer_correction` | confirmed, high | Uncollapsed instruments with a dead column raise; the Windmeijer correction uses step-2 residuals, so SEs are 1.01–1.68× Stata's (step-1 residuals reproduce Stata to 2.6e-6); `bb_gmm` shares the call | no notebook yet; the README parity row |
| Minnesota dummies (`var/bvar.py:241-254`) | partial, high | The inner loop overwrites a cell, so for λ₂ ≠ 1 own-lag and cross-lag tightness are swapped and the own-lag prior mean is λ₂; the NIW prior cannot honour λ₂ ≠ 1 yet accepts it silently. The validation case is pinned at λ₂ = 1, the one value where it cannot see the bug | `docs/var.md`; examples `bvar_fan_chart`, `glp_lambda_search`, `gk_robust_from_gibbs`, `posterior_predictive_fan` |
| Synthetic DiD, one treated unit | partial, high | The donor bootstrap under-covers (49–67% at nominal 90%, depending on the DGP) where the paper and R `synthdid` use placebo variance; SLSQP non-success returns uniform weights, so at large outcome scales SDID silently becomes DiD | 29 (also called in 10 and the 2.0 tour), `docs/did.md`, the `informal_hours` Colombia/Chile tables |
| Callaway–Sant'Anna aggregation | confirmed, medium | `att_overall` and the event study are unweighted means; this is documented, but presented as CS, whose eqs. 3.4, 3.10 and 3.11 weight by cohort size. The teaching has it backwards: puremacro's `sun_abraham` returns CS eq. 3.4/3.10 exactly | 10, 15, T05_C, T05_D, `docs/did.md`. **The working tree adds an assert in 10 (`:176`) that pins the unweighted rule.** |
| `la_lp` | confirmed, medium | Defaults to `max(h)` extra lags at every horizon (docstring says p + h); credits Plagborg-Møller–Wolf, whose paper is about LP = VAR. Montiel Olea–Plagborg-Møller (2021) add one lag | 50, 14, T04_A, T05_B, course 06, `docs/lp.md` |
| Fixed-b critical values (`inference/hac_fixed_b.py:20-47`) | confirmed, medium | Disagree with Kiefer–Vogelsang (2005) Table I: 3.96 against 4.771 at b = 1, two-sided 5%; a nominal 5% test rejects up to 9.5% | no notebook or example yet; any `hac_fixed_b` user |
| Sun–Abraham SE | confirmed, medium | Aggregates cohort SEs as independent; the error goes both ways (0.62× to 1.40× the Monte Carlo truth) | 10, 15, T05_C |
| AKM shift-share SE (`bartik/akm.py:243-254`) | partial, medium | Uses residualised shocks instead of AKM's projection; like-for-like on ADH it is 13.5% too small (0.1816 against 0.2101); the vignette's 0.2404 also needs sector clusters | no notebook yet |
| `geweke_z` | partial, medium | Bandwidth of 5–7 lags biases the long-run variance; a stationary chain is flagged 33% of the time at φ = 0.9 and 74% at 0.99 (R coda: 7–8%). Non-split R-hat is the classic statistic, not a bug | Bayesian DSGE diagnostics (`dsge/bayesian.py`), example `mcmc_diagnostics` |
| `realtime_nowcast(as_of=)` | confirmed, medium | The news decomposition and `vintage_history` use vintages published after `as_of` | none in the catalog (59 uses the latest vintage) |
| Gertler–Karadi units and HFI sign | partial, medium | `to_frame` returns level deviations that T02_F multiplies by 100 and labels "%" (K: −28.66% printed, −5.06% true); `gk2015_surprise` flips sign for futures prices | curso T02_F, example `hfi_gertler_karadi` |
| `bq_svar` cumulation | partial, medium | Cumulates every variable (documented), a trap for the canonical (Δy, u) specification; the result docstring says "level response", false for levels inputs | course 02/05, T02_B, example `gali_1999_hours` |
| `solve_aiyagari_continuous` | confirmed, low | Silent 500-iteration EGM cap, ignored `solver`, hard-coded `converged=True`; r* moves by at most 5e-7 in tests | 67 |
| `steady()` fallback | partial, low | A structurally singular model can return a full-system `hybr` point instead of raising, without a warning | — |
| `hamilton_filter` twins | not a defect | The two implementations give the same numbers; only their return types and signatures differ | — |
| `stock_yogo_cv(1, 10)` returns None | not a defect | Correct: no 10%-bias value exists for one instrument. The gap is that there is no size table (16.38 for one instrument is typed by hand in an example) | — |

**Why the galleries stayed green.** Several references are computed by the code under test (the SW07 targets), or pinned where the defect cannot show (Minnesota at λ₂ = 1, a single-cohort DiD case, Windmeijer checked only for inflation). The "107 checks, all pass" line in `paper/paper.md` rests on these. A rebuilt notebook 12 ("Can the gallery catch a wrong number?") is the natural place to show circular references and the discriminating oracles below.

## 3. Problems across the catalog

- **`notebooks/macro_history_and_climate/` cannot run for anyone.** All five notebooks import `exptools` (not installed anywhere); four hard-code `/Users/jalonso/...` and N12 a `/Volumes/BIGDATA` path. They are tracked, linked from `notebooks/README.md` and tested by `tests/test_v260_showcase_replication_e2e.py`. Retire them from the public tree.
- **Course lesson 04 (`notebooks/course/04_rbc_momentos_es.py`)** hard-codes the model's σ_c/σ_y as 0.44 in ten prose and code lines instead of computing it, and its "genuine parity" check compares the model with its own decision rules (PLAN noted this; still open).
- **Colab students get fixes only through PyPI.** All 62 curso notebooks install `puremacro>=4.0.1,<5` from PyPI, so a library fix on `main` reaches them only after a 4.x release, and a 5.0 never does. Decide the release vehicle before changing defaults.
- **The browser playground ships every showcase and course notebook,** but the Pyodide gate runs five smoke tests; no notebook has been executed in Pyodide. 68 already loses its Dynare oracles there (the fixtures are not in the wheel).
- **Exercises.** Of 71 rated notebooks, 5 have strong exercises, 23 adequate, 31 weak and 12 none (30–33, 35–39, 65, the 2.0 tour, `local_llm_uncertainty`). The root cause is in `_TEMPLATE.md:18-21`, which asks that the assertion hold only at the default. See section 6.
- **Hand-typed numbers in prose contradict outputs** in older notebooks. For example, 14 says "weak proxy ≈ 0" while it prints −4.27; 31 says the bottom 20% have MPCs above 55% while only the bottom decile does (0.66; decile 2 is 0.09); 38 states the noise-hypothesis slope as −1, the error the 20 September review corrected in 48. A test that every number in a Read-the-output cell is formatted from a variable would stop this.
- **Overlap.** 33 duplicates 48 with a leaked target (REVIEW), 22 is superseded by 56 (PLAN), 38 by 58, 50 by 14, 49 by 09. Consolidating before adding keeps the bilingual maintenance and Pyodide budget manageable.

## 4. New replications

Ranked by the quality of the independent oracle per unit of cost. "Checked" means the number was read in the source on 30 September; still re-read the page when freezing a golden.

| Replication | Oracle (independent of puremacro) | Data | Needs first |
|---|---|---|---|
| **SW07 against the paper** (rebuild 41; *PLAN*) | ECB WP 722 Table 1a: mode φ = 5.48, σ_c = 1.39 (5.74 and 1.38 are posterior *means*); Table 2: −905.8 with the 1956–65 training sample (checked). Pfeifer's Dynare replication as the second route | bundled `_sw07_data.csv`, rebuilt with the hours fix | SW07 markup fix, `#` locals fix |
| **California Prop 99 three ways** (rebuild 29) | Arkhangelsky et al. (2021) Table 1: SDID −15.6 (8.4), SC −19.6 (9.9), DID −27.3 (17.7), placebo s.e. (checked); R `synthdid` frozen | public Prop 99 panel, frozen | placebo variance; SLSQP fallback |
| **Arellano–Bond (1991)** | Stata `[XT] xtabond`: one-step L1.n .6862261 (s.e. .1486163 / robust .1445943); Example 4 two-step .6287089 with WC-robust s.e. .1934138 (checked; do not pair .6862261 with .1934138) | `abdata`, frozen | Windmeijer residuals; dead instrument columns |
| **Shift-share IV on the China shock** | R `ShiftShareSE` vignette: IV −0.7742267, AKM s.e. 0.2403730 under its exact specification (checked) | `ADH.rda`, frozen | AKM projection and sector clusters |
| **Lag-augmented LP coverage** | Montiel Olea–Plagborg-Møller (2021) Table 1, ρ = 0.95, T = 240, LP-LA: .878/.838/.806/.814/.833 at h = 1/6/12/36/60 (checked) | simulated | `la_lp` default and attribution |
| **Aguiar–Gopinath (2007) for Mexico** (*PLAN*) | JPE Table 4 Mexico: σ_g 2.81 (.37) … random-walk component .96 (.07) in column 1 (checked; eq. 14 reproduces all eight published ratios) | bundled OECD quarterly accounts and FX | — |
| **Rebuild 35: Kilian (2009), Ramey–Zubairy (2018), Galí (1999)** on their own samples (*PLAN*) | published figures and tables, read before freezing | bundled `kilian2009.csv`, `rz2018.csv`, `gali1999.csv` | Galí sample and differenced hours |
| **Hamilton (2018)** regression filter | WP 23429 Table 2: regression residuals against a random walk for 13 US series, e.g. GDP 3.38/3.69 (checked) — it does not compare HP | ALFRED vintage near the paper, frozen | reconcile the two `hamilton_filter` APIs |
| **Hamilton (1989)** recession dating | `statsmodels` MarkovAutoregression on the same data | `rgnp`, frozen | `ms_var_fit` M-step; model class |
| **Business-cycle moments: Hansen (1985) or BKK** (*PLAN*) | published table; Dynare's `bkk_1992.mod` numbers | none | write the model natively (`lambda` breaks the parser) |
| **Blanchard–Quah (1989)** on the 1988 vintage | closed-form bivariate VAR(1); published tables | Philadelphia Fed vintage | `bq_svar` cumulation mask |
| **OccBin you can trust: Guerrieri–Iacoviello (2015)** | Dynare 8 `occbin` paths, frozen | none | an initial-state argument |
| **Galí–Monacelli (2005)** open-economy sequel to 66 | GM tables; Pfeifer's model run in Dynare 8 | none | native model (keyword names) |
| **Mertens–Ravn (2014)** tax multipliers (rebuild 14 with 50 folded in) | published multipliers | MR data | `la_lp` fix |
| **Cost of business cycles: Lucas (1987), Imrohoroglu (1989)** | closed-form perfect-insurance CE; Kirkby's numbers | none | — |
| **Gains from trade: ACR and Costinot–Rodríguez-Clare** | exact ACR identity; CRC tables | CRC WIOT package | never use the bundled 77×11 table |

Further candidates, each vetted but lower in value per cost: Gertler–Karadi (2015) HFI proxy SVAR, Uhlig (2005) and ADRR (2018) sign restrictions, FRED-MD factors (Stock–Watson 2002), Bloom (2009) and JLN uncertainty extended to Mexico, Jordà–Schularick–Taylor credit booms, DML on the 401(k) data, Hopenhayn–Rogerson (1993), Hsieh–Klenow (2009), Arellano (2008), CKM business-cycle accounting, KORV (2000), Ireland (2004). Details and required changes are in [ideas.json](ideas.json).

## 5. New notebooks and labs

**Theory with closed-form oracles** (high value for the ITAM course; each is cheap because the closed form checks the solver):

- *Who controls inflation?* Leeper (1991) active/passive regions and the fiscal theory of the price level, with Sauer-style closed-form region maps; the fixed-regime half of PLAN's Davig–Leeper item. The FTPL appears nowhere in the catalog.
- *The liquidity trap three ways*: Eggertsson–Woodford closed form against the Markov-switching solver and OccBin; links 57 and T02_D.
- *Diamond (1965) OLG*: dynamic inefficiency and social security, checking the OLG solver against the log-utility closed form (anchor for PLAN's OLG item).
- Further closed-form theory notebooks, vetted: Lucas tree and the equity premium (exact Mehra–Prescott and Burnside prices against perturbation), Pareto wealth tails (Kesten processes), money (MIU, cash-in-advance, Lagos–Wright), menu costs against Calvo, capital taxation (Chamley–Judd, Aiyagari 1995), endogenous growth, time inconsistency beyond LQ, job search and optimal UI.

**Labs with a known truth** (simulation oracle; one lesson each):

- *Instrument strength versus validity* (rebuild 28; *REVIEW*): Wald, Anderson–Rubin and tF coverage under weak and invalid instruments, with a closed-form noncentral-F coverage check.
- *Coverage lab for staggered DiD* (*REVIEW*): pointwise, sup-t and overall-ATT intervals with shared controls; exposes the Sun–Abraham SE and makes the CS weighting visible.
- *LP or VAR?* Bias, variance and coverage by horizon against a DSGE-generated truth; *What does an SVAR recover from a known DSGE?* (invertibility, news shocks).
- *Filter-induced cycles* on random walks (HP, BK, CF, Hamilton, Beveridge–Nelson) with exact spectral oracles; supports lesson 01.
- *Weak identification in a DSGE* (after the `identification()` fix); *MCMC diagnostics against closed forms* (after the `geweke_z` fix), leading into PLAN's "three samplers, one posterior".
- *Grading density forecasts* (PIT, Berkowitz, CRPS, log scores) and *forecast-comparison pitfalls* (look-ahead tuning, nested-model DM size, fixed-b), after the fixed-b table fix.
- *Information-set lab* hosted in 48 (*REVIEW*), which retires 33.
- *Can the gallery catch a wrong number?* (rebuild 12): circular references, blind designs, and the discriminating oracles of section 4 run against the unfixed and fixed library.

**Mexico and Latin America** (bundled `data_curso` data): Aguiar–Gopinath (above), a Banxico Taylor rule benchmarked on Clarida–Galí–Gertler (2000), peso–dollar DCC, Growth-at-Risk for Mexico, the ins and outs of unemployment with a formal/informal split (ENOE), and a GVAR of US shocks reaching Mexico.

**Starters and tours.** One compact starter (simulate first, then replace the data) and PLAN's "DSGE from a Python function"; fold the 2.0 and 3.0 tours into topic notebooks instead of adding a 4.x tour.

Deferred or dropped after vetting: Kydland–Prescott time-to-build (no published-number oracle), lumpy investment (cost), the dynamic-MRIO tariff and "three trade models, one tariff" (unreleased modules and self-generated oracles), Caliendo–Parro NAFTA and KORV (data licence unverified), Kaplan–Moll–Violante (no-go until the two-asset solver is fixed; see section 2).

## 6. Exercises

The audit's lesson is that most exercises cannot fail. Principles for every new and rebuilt notebook:

1. **Every assertion must be able to fail.** Nothing guaranteed by a `brentq` bracket, a library identity or the definition of the band. Amend `_TEMPLATE.md:18-21` so the assertion holds *over the advertised range*, and test the corners.
2. **Predict, then run, then assert** a sign, a direction or an ordering.
3. **Use the planted truth.** Synthetic-DGP notebooks (06, 07, 08, 10, 11, 60) should grade estimates against it, with a Monte Carlo coverage table and MC s.e. √(p(1−p)/R).
4. **Invariance checks**: country order, numeraire, units, re-referencing. The exercise audit reports that 55 fails one (a 1e-6 tariff moves welfare by up to 1.7 pp while a zero tariff short-circuits to zero); this was not re-verified.
5. **Oracle design**: "write the check that would have caught this bug" — Minnesota at λ₂ = 0.5, CS eq. 3.10, an antisymmetric tensor error.

For the latest notebooks (all tested on a modified copy unless noted):

- **68:** *Which risk term flips sign?* (assert `sign(ghuss) == sign(ρ)`, `ghxss > 0`, `ghxss/ghuss == ρ`); *a symmetry test sees what a path check cannot*; *precaution or Jensen?* (decompose mean capital into the risk term −4.5e-5, dispersion +7.1e-4 and shock variance +1.3e-4, which sum exactly to the ergodic mean).
- **67:** Table I against closed-form lognormal moments; how loose must Huggett's limit be for r > 0; unconditional versus innovation s.d. design; the corner's error budget.
- **66:** what share of the commitment gain a conservative central banker delivers; the Jensen–McCallum rule family c ∈ [0,1]; real rates on impact; price-level targeting implements timeless commitment.
- **65:** relabel the countries and see which numbers survive; where the Leontief "gain" comes from (investment); the tariff-incidence boundary.
- **64:** an honest fold test; the first tariff at which cold Newton fails (continuation succeeds at 150%); what a clamped step buys; a factor-price existence boundary near τ ≈ 2.3, far below the Hawkins–Simon threshold of 8.65.
- **63:** small damped steps with large regret; a ceiling as a trade agreement (who wins); when the optimum is interior.
- **62:** who pays A's tariff at σ ∈ {0, 0.5, 2}, and the singularity near 1.

[EXERCISES.md](EXERCISES.md) holds about 200 vetted exercises across the showcases and curso units (for example the Card–Krueger 2×2 on real data for T05_C, Okun's law with four margins for T08_A, Banxico's rule on the determinacy map for T02_C), each with its self-check.

## 7. Suggested order

| When | Items |
|---|---|
| **Now** | P0 fixes to 62–68; retract or fix the two-asset section of 46 and mark the two-asset solver experimental; retire `macro_history_and_climate`; rebuild course 04; commit 66–68 and the 62 rewrite by path. Land the library fixes with every consumer in the same change and an ADVISORY entry: SW07 markup and `#` locals first (JOSS), then CS naming and weights (stop the new assert in 10), `la_lp`, Minnesota, SDID placebo variance, Windmeijer, fixed-b, AKM, Sun–Abraham SE, `identification()`, `ms_var_fit`, `geweke_z`. Relabel the self-generated SW07 targets in `docs/replication.md` and ship the 68 fixtures in the wheel. |
| **Next** | The independent-oracle replications of section 4 that double as regression tests for those fixes (Prop 99, Arellano–Bond, ADH shift-share, MOPM, SW07 against the paper); rebuild 35; consolidations (33→48, 22→56, 38→58, 50→14, 49→09); the Leeper/FTPL and liquidity-trap notebooks; Aguiar–Gopinath for Mexico; the instrument and DiD coverage labs; rebuild 12. |
| **Later** | Remaining replications and theory notebooks; the other labs; rebuild 64 once `trade.condensed`, `continuation` and `stability` are committed and tagged; a playground truth table that runs every browser notebook in Pyodide. |

## 8. Decisions for you

- **Release vehicle for default-changing fixes** (CS aggregation, `la_lp`, `bq_svar`, Minnesota λ₂): a 4.x release with `FutureWarning`s, or 5.0 — which the Colab pin `<5` would never deliver to students.
- **The bundled 77×11 table:** regenerate it from the clean OECD release (breaking the MATLAB parity contract), or keep it as a labelled regression fixture across 61–65.
- **Notebook 63:** keep the corner result as the lesson, or recalibrate for an interior Nash equilibrium.
- **Numbering:** retire notebooks with gaps or renumber (tests pin the counts).
- **Licensing:** freezing numbers from GPL sources (Kirkby, Pfeifer, Dynare examples) and bundling `abdata`, `ADH.rda` and the Prop 99 panel.
- **SW07 target:** ECB WP 722 or the AER tables, and whether SW's original data may be bundled.
- **The drafted Dynare issue** on order-3 pruned autocorrelations (from the 23 September plan).
- **The parallel session's edits:** its new assert in notebook 10 entrenches the unweighted CS rule, and the Jensen–McCallum citations exist only in the working tree.

## Files

- [findings.json](findings.json) — the 355 surviving findings on 62–68 with verifier-corrected wording, plus the 5 refuted.
- [defects_and_oracles.json](defects_and_oracles.json) — the twenty-one library-defect verifications (repro, primary source, affected teaching, fix sketch) and the eleven oracle-number checks with quotes and pages.
- [ideas.json](ideas.json) — the 131 vetted proposals with their required changes.
- [EXERCISES.md](EXERCISES.md) — the exercise program, by notebook.
