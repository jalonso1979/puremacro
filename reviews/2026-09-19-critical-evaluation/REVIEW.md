**Critical evaluation of puremacro 4.2.0 — 19 September 2026**

**Assessment.** Puremacro has a useful numerical core, substantial working functionality, and much more testing than a typical research repository. Its strongest contribution is bringing estimation, economic models, data, and teaching examples into a common Python environment. However, the newest structural-model features have expanded faster than their independent validation. I would not currently rely on the new trade welfare/theorem outputs, the third-order DSGE risk corrections, or the frontier VFI solvers' convergence flags for research conclusions without additional checks. Several problems return plausible numbers and successful status flags.

This is a review, not a repair. No production code or existing tests were edited. The review added this directory and installed pytest in the existing virtual environment.

**Scope and evidence.** Reviewed HEAD `f5c28db` plus the existing working-tree changes. The uncommitted trade work includes roughly 2,980 added lines across the tracked trade files, a new regularization module, new stress tests, and notebooks 63–65. The review covered the package architecture, release/CI configuration, validation design, recent history, documentation, and technical-report claims, with detailed code inspection and numerical probes concentrated on VFI, DSGE/Dynare, and trade. Additional tests cover DML, nowcasting, monetary transmission, the validation gallery, and package-wide import/API contracts. This was not a line-by-line audit of every module or a complete execution of every notebook.

Two disjoint test runs completed:

| Run | Passed | Failed | Skipped | Deselected | Elapsed |
|---|---:|---:|---:|---:|---:|
| VFI, DSGE directory, trade, OccBin and v3.0 tests | 2,668 | 2 | 30 | 7 | 544.62 s |
| Additional DSGE/Dynare, DML, nowcast, monetary, validation and package contracts | 918 | 3 | 2 | 0 | 77.93 s |
| Total | **3,586** | **5** | **32** | **7** | |

Environment: Python 3.14.6; NumPy 2.5.2; SciPy 1.18.1; pandas 3.0.5; pytest 9.1.1. The configured CI matrix covers Python 3.11–3.13, so these observations should also be reproduced on a CI-tested interpreter before attributing all suite failures to a release regression. The closed-form and API-invariant counterexamples below do not depend on interpreting the five test failures.

The reproducible probes are in [reproduce.py](reproduce.py), with results in [observations.json](observations.json). Full test output is preserved in [focused-tests.log](focused-tests.log) and [broad-tests.log](broad-tests.log). No live Dynare execution, GPU/MLX/CUDA validation, full external MRIO archive ingestion, live reference-package drift run, or browser runtime execution was performed. Passing the import-contract tests is narrower than executing all estimators under Pyodide.

**Findings requiring attention before relying on the affected outputs**

**1. Trade: the “exact Hicksian EV” is defined by the decomposition, not independently measured. High priority; uncommitted addition.**

[policy_analytics.py:1621](/Users/jalonso/Documents/RESEARCH/puremacro/puremacro/trade/policy_analytics.py:1621) takes the level of counterfactual tariff revenue divided by CPI without subtracting baseline revenue. At line 1631 it defines `ev_usd = delta_tot_usd + delta_alloc_usd + delta_tariff_rec_usd`, then checks that the same sum equals EV. The zero residual is an arithmetic identity; it cannot certify Hicksian welfare accounting.

The decisive counterexample uses **the same solved tariff equilibrium as both baseline and counterfactual**. True EV must be zero. The function returns **70.9104753232**, with an identity residual of **0**. In addition, its import-price calculation averages source-country prices before weighting by flows, and its Fisher construction omits final-demand trade. These are not sufficient statistics for a general exact expenditure-function decomposition.

Repair: obtain EV independently from the model's expenditure function and utility change at baseline prices; compute changes relative to the supplied baseline; derive channels consistent with the model and fiscal closure. Retain any unexplained remainder as a diagnostic. Harberger triangles can be offered as a local approximation with an explicit approximation error. Notebook 65's machine-precision identity claims should be withdrawn until this is done.

**2. Trade: theorem “verification” mostly imposes the proposed conclusions. High priority; uncommitted addition.**

[policy_analytics.py:1145](/Users/jalonso/Documents/RESEARCH/puremacro/puremacro/trade/policy_analytics.py:1145) reduces the calibration to a few averages and substitutes constructed formulas for solved Leontief and CES equilibria. Examples include setting Leontief output/imports to their baseline values, assigning CES GDP loss to a Harberger formula after computing a different GDP expression, imposing import-volume rankings, and evaluating tariff revenue on an assumed demand curve. At line 1286, `max(tot_gain - dwl_loss_loe, 0.01)` explicitly prevents the reported gain from being negative. At zero tariffs, line 1282 evaluates the large-open-economy calculation at a 10% tariff instead.

Probe: a zero-tariff scenario reports **12.5640058386** of large-economy welfare gains and passes all four checks. The 99.3% rebate “theorem” is also a calibration-dependent numerical claim, not a general theorem established by the function.

Repair: separate analytical examples from empirical tests. State each proposition's assumptions and compute outcomes from independent equilibrium solutions. Permit violations and report why they occur. Test zero shocks, actual network heterogeneity, varying closures, and counterexamples outside the assumptions. Do not interpret hundreds of tests against the same formulas as evidence for the economic propositions.

**3. DSGE: third-order risk slopes fail an exact polynomial benchmark. High priority; committed code.**

The risk-correction assembly in [dynare.py:1520](/Users/jalonso/Documents/RESEARCH/puremacro/puremacro/dsge/dynare.py:1520) is incorrect for this simple model:

```text
x_t = rho*x_(t-1) + e_t,       Var(e_t) = s²
y_t = beta*E_t[y_(t+1)] + x_t³
```

Its exact bounded solution is `y_t = A*x_t³ + B*x_t`, where

```text
A = 1 / (1 - beta*rho³)
B = 3*beta*A*rho*s² / (1 - beta*rho).
```

Using puremacro's documented lagged-state decision-rule convention, the risk terms are `ghxss_y = 2*B*rho` and `ghuss_y = 2*B`. With `beta=.95`, `rho=.8`, `s=.1`:

| Coefficient | Exact | Returned |
|---|---:|---:|
| `ghxss_y` | 0.295950156 | 0.369937695 |
| `ghuss_y` | 0.369937695 | 0.462422118 |

Both are **25% too large**. At `rho=.5`, both are **100% too large**. The cubic coefficients agree with the exact solution to about `1e-15`, isolating the issue to the stochastic risk terms rather than the cubic derivative calculation or variable ordering.

Repair: rederive the complete risk-slope equations, including propagation/timing through the state transition and all variance contractions. Add polynomial closed forms with multiple states and correlated shocks, then compare against independently generated Dynare results. Risk-sensitive IRFs, higher-order moments, and likelihood calculations using these rules inherit the error.

**4. Dynare: a 100% parity score does not establish full second-order parity. High priority; committed code.**

[parity.py:238](/Users/jalonso/Documents/RESEARCH/puremacro/puremacro/dsge/parity.py:238) calculates steady-state discrepancies without including them in pass/fail. It checks `ghxx` and `ghs2` at order two but omits `ghxu` and `ghuu`. Some malformed/missing moment comparisons default to zero deviation; exceptions are swallowed. Column alignment also needs to be by state and shock names, not only by variable rows.

Probe: add **1,000** to every steady state, `ghxu`, and `ghuu` entry in a second-order reference. The dashboard reports **passed=True, score=100.0**. The omitted cross-shock and shock-curvature terms are part of the second-order decision rule in the [official Dynare manual](https://www.dynare.org/manual/the-model-file.html).

Repair: check every required tensor, steady states, labeled row/column ordering, finite values, shapes, covariance assumptions, and requested moments. Distinguish PASS, FAIL, and UNAVAILABLE. A comparison skipped because of an exception must not increase confidence. Also update the parity documentation: the advertised CLI deliberately exits with an error since 4.0.0 ([cli.py:167](/Users/jalonso/Documents/RESEARCH/puremacro/puremacro/dsge/cli.py:167)).

**5. VFI: least-squares termination is accepted as Euler-equation convergence. High priority; committed code.**

In [collocation.py:871](/Users/jalonso/Documents/RESEARCH/puremacro/puremacro/vfi/collocation.py:871), MINPACK LM success sets `converged=True`; the final residual check can only turn the flag on, not off. LM can successfully minimize a nonzero residual without finding a root.

Probe: supply the valid residual callback `R(s)=1`, which has no solution, and request tolerance `1e-10`. Returned result: **converged=True, residual_norm=1.0**. Related success-or-residual patterns occur in FEM and Smolyak, including fixed residual thresholds independent of requested tolerance. Those paths need the same audit.

Repair: compute convergence from the finite, unmodified final equilibrium residual and the requested tolerance. Report optimizer termination separately. For constrained problems, also verify feasibility and complementarity; for projection accuracy, evaluate residuals away from fitting nodes.

**6. Trade: CES equilibrium solutions are postprocessed using Leontief quantities. High priority; committed code, affecting newer analyses.**

[equilibrium.py:313](/Users/jalonso/Documents/RESEARCH/puremacro/puremacro/trade/equilibrium.py:313) correctly changes intermediate demand with `sigma`. However, [postprocessing.py:237](/Users/jalonso/Documents/RESEARCH/puremacro/puremacro/trade/postprocessing.py:237) reconstructs `x_mat = calib.a * ytot` regardless of substitution, and its intermediate price calculation is also Leontief. The solver passes extension settings as metadata, which does not make the postprocessor apply them.

Probe: a two-country/two-sector case with `sigma=2` converges to residual **8.37e-11**. Reported intermediate flows differ from the demand equations at that returned solution by **19.2481** in the table's units, and agree with the Leontief quantities exactly. Trade volumes, tariff revenue, and downstream welfare decompositions can therefore describe a different model from the one solved.

Repair: use a shared economic evaluation routine for both residual construction and postprocessing, with all technology/fiscal/capacity settings forwarded explicitly. Check accounting identities using the returned flows, not just the internal residual vector.

**7. Trade: switching to Keller PAC can silently change the tariff experiment. High priority; uncommitted addition.**

[solver.py:2146](/Users/jalonso/Documents/RESEARCH/puremacro/puremacro/trade/solver.py:2146) selects a target from `tau`/`tauf`, dropping `tau_fd` and `tauf_fd`. The resolved tariff arrays are not passed through to PAC.

Probe: `tau_fd=[.2,0]` produces **73.4607** of tariff revenue under ordinary Newton. With `method="keller_pac"`, the returned solution is exactly the zero-tariff baseline, with zero revenue. Both report convergence.

Other code-review concerns: PAC resets its step from `desired_dlam` each iteration, undoing failed-corrector step halving; `condensed` is ignored despite its documented scaling role; and named scenarios reach a numeric viability check before being resolved. These are distinct from the demonstrated tariff-loss bug.

Repair: pass complete start/target tariff schedules, make method changes preserve the experiment, retain reduced step sizes until an accepted step, and test the actual public PAC solver on a known fold. A separately implemented notebook fold example does not validate the library solver.

**8. MRIO: missing empirical data silently become synthetic calibrations. High priority; uncommitted addition.**

The new loaders default to synthetic fallback. [data.py:1086](/Users/jalonso/Documents/RESEARCH/puremacro/puremacro/trade/data.py:1086) also takes the synthetic path whenever custom dimensions are supplied, before checking the file or the fallback option. FIGARO's synthetic records are renamed `Eurostat_FIGARO_Harmonized`; [the calibration packaging step](/Users/jalonso/Documents/RESEARCH/puremacro/puremacro/trade/data.py:1028) does not retain their provenance, year, currency-conversion assumptions, or regularization history in result metadata.

Probe: a nonexistent file, `fallback_to_synthetic=False`, and custom dimensions return a calibration with **no warning and empty metadata**. Notebook 65 invokes default loaders and describes its cross-database results as empirical. Depending on available files, those results can instead be generated data.

Repair: make synthetic datasets an explicit opt-in, preserve `is_synthetic`, source, year, schema, units, exchange rate, file hash, and balancing adjustments, and display provenance in notebooks. Validate real archive schemas with frozen, labeled source fixtures; shape-compatible random matrices do not test ingestion correctness. Treat the hardcoded labor/capital split and residual-tax reconciliation as modeling choices requiring disclosure.

**9. Caliendo–Parro: the identity shortcut conflates tariffs with iceberg costs. High priority; committed 17 September addition.**

[caliendo_parro.py:638](/Users/jalonso/Documents/RESEARCH/puremacro/puremacro/trade/caliendo_parro.py:638) returns the baseline whenever the product of tariff and iceberg changes is approximately one. Equal delivered trade costs do not mean equal government revenue or national income.

Probe: increase bilateral gross tariffs to 1.2 and offset them with iceberg changes of `1/1.2`. The solver returns zero tariff revenue, zero iterations, and convergence. Its expenditure equations at the returned wages and shares imply **6.89655** revenue per country. This violates the counterfactual's fiscal accounting even before a welfare comparison.

Repair: short-circuit only when tariffs, iceberg costs, and the deficit closure are independently unchanged. Use a tolerance consistent with the solve tolerance. Validate baseline labor-market consistency rather than assigning zero diagnostic residuals. The authors provide a useful independent [Caliendo–Parro replication package](https://faculty.som.yale.edu/lorenzocaliendo/estimates-of-the-trade-and-welfare-effects-of-nafta/); reproducing selected published outcomes would be stronger evidence than identity-shock tests alone.

**10. Nash tariffs: small damped updates can be mistaken for a Nash equilibrium. High priority for equilibrium claims; committed code.**

[optimal_tariffs.py:864](/Users/jalonso/Documents/RESEARCH/puremacro/puremacro/trade/optimal_tariffs.py:864) checks only the change in the relaxed tariff vector. Its local three-point search does not establish a global best response, and the routine does not require every GE evaluation to converge.

Probe with a valid positive relaxation of `1e-8`: it reports convergence after moving the first tariff to **4e-10**. That player's own reported welfare rises from **1027.06966** to **1030.25157** by unilaterally selecting a 20% tariff with the other player's tariff unchanged at zero. This is a profitable deviation, not a Nash certificate.

Repair: check undamped best-response residuals and unilateral deviation gains over the allowed policy interval, alongside GE feasibility. Return maximum regret, boundary/KKT diagnostics, and unresolved inner-solver failures. Until then, describe the routine as a tariff-adjustment heuristic whose candidate equilibria require verification.

**11. VFI Deep Macro: training backpropagates through the next-state cache. High priority for training correctness; committed code.**

[deep_macro.py:1002](/Users/jalonso/Documents/RESEARCH/puremacro/puremacro/vfi/deep_macro.py:1002) evaluates next-period consumption with the same MLP after evaluating current consumption. Each forward pass overwrites the activation cache. At line 1039, the backward pass combines current-output derivatives with next-state activations.

Instrumenting one actual training iteration gives a **0.4136% relative gradient discrepancy** compared with the intended current-state backward pass using identical upstream derivatives. Its magnitude is calibration-dependent; the cache mismatch is structural. Tests of the standalone MLP derivative do not catch this composition error.

Repair: preserve current-state caches or re-run the current forward pass before backpropagating the detached target loss. Clearly distinguish Euler time iteration with detached targets from differentiation of an Euler-residual objective. Check a complete training-step gradient and stochastic conditional expectations, not just activation derivatives or in-distribution training fit.

**Additional confirmed or materially incomplete behavior**

- **VFI auxiliary values use the wrong primitives.** [collocation.py:891](/Users/jalonso/Documents/RESEARCH/puremacro/puremacro/vfi/collocation.py:891) omits productivity `z`/`A` and always uses log utility when reconstructing values from an Euler solution. With log utility and productivity 2, steady-state value is **-34.57019**, versus the analytic **1.53412**, despite convergence. Use the same utility, technology and transition primitives in both policy and value calculations.
- **The MRIO spectral estimate is not a certificate.** [regularize.py:286](/Users/jalonso/Documents/RESEARCH/puremacro/puremacro/trade/regularize.py:286) ignores the returned Collatz–Wielandt bounds and gates viability on an unconverged power-iteration estimate. For `[[0,2],[.2,0]]`, it returns **1.1**, whereas the exact radius is **sqrt(.4)=.6324555**; bounds remain `[.2,2]`. A productive periodic network can be classified as unproductive. Use certified bounds, periodicity-safe iteration, and a fallback when the bounds straddle the threshold; report uncertainty instead of a false certificate.
- **Markov-switching moments survive a failed solve.** [markov_switching.py:1269](/Users/jalonso/Documents/RESEARCH/puremacro/puremacro/dsge/markov_switching.py:1269) evaluates stability and ergodic moments even when the coupled system has not converged. The failing test's scalar polynomial `T²-3T+2.5` has no real root; it should test rejection/nonconvergence, not demand a particular unstable real solution. In the observed run, the routine reports `converged=False` but `mean_square_stable=True` and finite covariance. The residual at the returned matrices is **.58361**, versus reported `diff=.25044`. Mark stability/moments unavailable after failure and recompute diagnostics at the returned iterate.
- **“Exact analytic VFI gradients” and “adjoint distribution sensitivities” overstate the implementation.** [analytic_gradients.py:515](/Users/jalonso/Documents/RESEARCH/puremacro/puremacro/vfi/analytic_gradients.py:515) builds both residual Jacobians by central differences. This is a useful semi-analytic IFT calculation, not a machine-precision analytic derivative. The heterogeneous-agent aggregate branch averages direct policy derivatives at a fixed distribution and leaves `grad_mu=None`; it does not solve the differentiated stationary-distribution equation or the coupled GE response. Either implement those equations or narrow the advertised scope. The module header already partially acknowledges the semi-analytic distinction, but public docstrings and the architecture description do not.

**Interpretation of the five existing-suite failures**

1. `test_check_fertility_model_offending_loadings`: expected indeterminacy; the helper-built model is classified determinate.
2. `test_resid_fertility_broken_bgp_replicates_1_48`: expected residual near 1.4837; observed .3585907. Both fertility tests use a research-script fixture. This review has not determined whether the fixture/goldens are stale or diagnostic behavior regressed; they need reconciliation, not automatic replacement of expected values.
3. `test_explosive_regime_mean_square_unstable`: the non-real-root case discussed above. Both test design and post-failure diagnostics require work.
4. `test_smc_result_presentation_contract`: `SMCResult.to_latex()` calls pandas' Jinja-dependent formatter; Jinja2 is not installed or declared in the base dependency set. Use puremacro's existing dependency-minimal formatter or explicitly document/declare the extra.
5. `test_public_api_matches_snapshot`: the new trade exports drift from the frozen API snapshot. The log also shows an unavailable teaching-module entry in this environment. Review the intended public surface and import errors before regenerating the snapshot.

The new trade tests in the focused run passed. That is specifically why the independent counterexamples matter: passing this suite is not enough to establish the new welfare, theorem, equilibrium, or data-provenance claims.

**Assessment across the package**

| Area | Evidence supporting it | Current limit |
|---|---|---|
| Empirical core: VAR, LP, inference, Kalman/state space | Frozen external references, analytical cases, useful layered API and broad import checks | The whole empirical test suite and live reference drift suite were not run here; avoid extrapolating a small gallery to every estimator |
| Discrete VFI, EGM, distributions, household models | Many economics, recovery, and cross-method tests passed; recent GTH and KFE fixes address real numerical issues | Grid/boundary and transition accuracy still need model-specific checks |
| Continuous VFI/HJB | Substantial implementation and useful explicit treatment of nonuniform-grid densities and GE tolerances | Frontier projection convergence/value/gradient claims need repair; passing mass normalization is not a solve-accuracy certificate |
| DSGE first order and estimation | Strong linear closed forms; many parser, filtering, Bayesian, and estimation tests passed | General Dynare compatibility is broader than the tested subset; estimation diagnostics and calibration validity remain essential |
| DSGE higher order, regimes, and parity | Real tensor, pruning, and switching implementations rather than merely wrappers | The demonstrated third-order and parity errors prevent a blanket replacement/parity claim |
| HANK, monetary, DML and nowcast additions | The additional selected tests passed; HANK has explicit household distribution and sequence-space machinery | Internal identities and synthetic recovery are not external replication or empirical policy validation |
| Trade calibration and baseline solver | Bundled OECD inputs and externally generated reference equilibria are valuable; multiple solver paths and accounting tests | CES output mismatch, CP shortcut, and Nash convergence weaken structural conclusions; new welfare/theorem/loader layer needs immediate attention |
| Presentation, docs, teaching | Broad bilingual material and shared result objects lower the cost of use | The newest narrative sometimes treats illustrative calculations as empirical evidence and a passed identity as model certification |

The technical report should be revised before dissemination. Its abstract and introduction claim universal/bit-for-bit portability and machine-precision external certification. The actual 107-case gallery contains **59 internal, 29 analytical, 13 package, five SciPy, and one published** case; **no trade cases are registered in that gallery**. Its tolerances include numeric, coarse and qualitative checks, not only machine precision. Separate trade tests do exist and should be described separately. The more careful scope statement already in [docs/VALIDATION.md](/Users/jalonso/Documents/RESEARCH/puremacro/docs/VALIDATION.md:104) is a better standard for the report.

Also distinguish “puremacro ships no compiled extension of its own” from “the complete stack has no compiled dependencies.” The installation has five mandatory packages, including requests; optional acceleration and file-format dependencies widen it further. Browser importability, browser execution at realistic sizes, cross-backend numerical agreement, and bitwise identity are different claims. The repository's own MLX tests permit looser float32 tolerances.

**Recommended repair order**

1. Correct or temporarily quarantine the invalid EV/theorem claims and synthetic-as-empirical notebook paths. Fix CES postprocessing and preserve experiment semantics across solvers.
2. Repair third-order risk corrections and the parity dashboard together; require complete analytical and external decision-rule comparisons before advertising higher-order parity.
3. Enforce residual-based convergence throughout VFI, regime models and trade; add unilateral regret to tariff games. Fix the Deep Macro cache and VFI value reconstruction.
4. Preserve MRIO provenance and adjustment reports, and test actual source-file schemas. Add periodic/reducible matrices to spectral certification checks.
5. Separate feature inventory from validation status in documentation. For each public solver, record the economic model, assumptions, supported input class, independent oracle, tested configurations, residual contract, and known limitations. Reconcile existing failures and API snapshots before release.

The best next investment is a smaller set of decisive independent benchmarks and a consistent failure contract. The repository already demonstrates how to do this in its stronger core modules; extending that standard to the newest features will improve credibility more than adding further solver names or stress-test volume.
