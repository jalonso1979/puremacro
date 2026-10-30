# Changelog

This file records user-visible changes per release. Internal refactors that don't change behaviour are listed under "Internal" so a returning user can see what shifted under the hood without surprise.

This changelog describes versioned source changes. Check
[PyPI](https://pypi.org/project/puremacro/) for published distributions.
Versions 5.3.0, 5.4.0 and 5.4.1 were tagged but not published to PyPI; their
additions are included in 5.4.2. Versions 5.8.0, 5.9.0 and 6.0.0 were tagged
but not published; their additions are included in 6.0.1, published on
2026-10-29. See the [release procedure](RELEASING.md).

## 6.1.0 (unreleased)

Development version after 6.0.1; not a published distribution.

- Bounded the short-term monotone policy search's workspace by batching
  independent income columns. Values, choices, equilibrium metadata and random
  paths retain exact parity with the pinned original algorithm. The sovereign
  brief retains a failed crossed refinement through 12,801 × 1,281 and registers
  a further debt refinement without relaxing its tolerances or rewriting the
  historical cases' source provenance.

- `calibrate_one_asset_hank` and `solve_one_asset_hank`: the published ABRS
  one-asset equilibrium with endogenous labor, debt revaluation, asset/labor
  clearing and nonlinear Phillips curve. Independent author-code fixtures
  check the published 500×7 calibration and both shock signs. A fixed-tolerance
  study extends income refinement to 31 states and checks assets and horizons;
  failed intermediate refinements remain visible. The existing reduced HANK
  API keeps its original closure. [Guide](docs/hank_equilibrium.md).
- Independently validated the tariff brief's positive consumption EV with a
  separate Leontief equilibrium solver and direct flow/budget/EV checks for
  `sigma=0` and `sigma=2`. The national record now includes investment and
  spending-abroad quantity changes at baseline prices; bilingual guides explain
  the final-use reallocation and frozen resource-balance/closure checks. The
  default tariff covers all modeled sectors including services; the CES input
  composite substitutes across all country-sector inputs. Both descriptions
  are corrected without changing the recorded equilibrium results.
- Independently checked the stated long-term sovereign operator, haircut/maturity
  states and collateral-planner policies on small grids. Long-term simulation,
  spread diagnostics and survival pricing now use the debt-choice probabilities
  used by the smoothed pricing operator; `policy_b` remains the modal choice.
  Short-term values, prices and simulations retain exact 6.0.1 parity. The
  long-term operator is explicitly documented as a quantal-policy approximation,
  with its deterministic-optimality gap exposed.
- Replaced prescribed collateral diagnostics with reproducible simulated
  measurements. Amplification uses weighted consumption drops at identical
  debt/capital states; undefined cases return `NaN` with a status, and no 1.35
  fallback or lower bound remains. Crisis frequency and grid binding are distinct;
  capacity averages the actual product `kappa*q*k_next`. Infeasible consumption
  grids are rejected, and final Bellman residuals certify both policy problems.
- Haircut-only restructuring reuses the original equilibrium; maturity changes
  preserve its numerical budget and reject nonconvergence. Recovery starts at
  the selected income state and reports debt-grid rounding. The legacy survival
  spread `risk_premium_component` is documented as a compounding adjustment,
  with independent risk-free/constant-hazard checks. Bilingual notebook 73 and
  guides now use the corrected measurements and state their validation scope.

- Added `puremacro.inference.gmm`, a public generic GMM engine: user moment
  functions, two-step, iterated, identity or user weighting, Bartlett HAC
  long-run covariance (package default bandwidth), analytic or numerical
  Jacobians, sandwich standard errors, Hansen J and Wald tests. Returns
  `GMMFitResult`. Linear IV estimates, standard errors and J match
  linearmodels `IV2SLS`/`IVGMM` and `inference.hansen_j` to 1e-8.
  Optimizer stopping is invariant to common changes of moment units, and a
  rank check rejects unidentified parameter combinations before inference.
- Added `puremacro.asset_pricing` (macro-finance): `to_returns`,
  `excess_returns`, `compound` and annualization helpers; `risk_summary`
  (Sharpe ratio with a Lo 2002 HAC standard error, Sortino, moments,
  drawdowns); `value_at_risk` / `expected_shortfall` (historical, normal,
  filtered GARCH) and `var_backtest` (Kupiec, Christoffersen);
  `time_series_test` (GRS and HAC Wald); `fama_macbeth` (Fama-MacBeth,
  Shanken and Newey-West se); `sdf_gmm` and `hj_distance` (Jagannathan-Wang
  p-value); `ccapm` (Hansen-Singleton) with FRED helpers
  `fred_consumption_growth` and `fred_risk_free`. Inputs are refused when they
  contain NaN/inf, look percent-scaled or mix frequencies. Estimates match
  statsmodels and linearmodels to 1e-8 where definitions coincide; the
  remaining differences are documented. Multi-period historical VaR/ES compound
  simple returns; SDF/HJ reject rank-deficient systems before inference. Bilingual guide `docs/asset_pricing.md`.
- Added `puremacro.fetch.kenfrench_factors.fetch` for the Fama-French three-
  and five-factor, momentum and 25 size/book-to-market files. Data are
  downloaded into the user cache and never bundled (registry entry
  `kenfrench.factors`, not redistributable). A network test reproduces the
  Fama and French (2015) GRS of the three-factor model on the 25 portfolios
  (3.57 against the published 3.62).
- `puremacro.sovereign.solve_fiscal_sovereign_default`: a fiscal block on the
  short-term Arellano solver. A fixed tax share `tau` funds public spending
  `g = tau*y - b + q(b', y) b'`, valued as `psi * v(g)` next to private
  consumption `(1 - tau) y`. `FiscalSovereignResult` adds spending,
  primary-balance and debt-to-output policies, private consumption, and a
  `simulate()` built on a vectorized path engine with common random numbers.
  The certified monotone search is the default. With `tau=1` and private
  consumption off the solver is bit-identical to `solve_sovereign_default`.
  `tau` and `psi` have no defaults; `tau=0.2, psi=1` is an illustrative
  calibration, not fitted to any target. Guide:
  [fiscal block](docs/sovereign_fiscal_rules.md).
- `tools/reference_validation/check_fiscal_grids.py`: a paired-seed grid
  ladder for the illustrative fiscal calibration, frozen in
  `tests/fixtures/sovereign_fiscal/fiscal_grid_convergence.json`.

- `solve_sovereign_default` now calls a private Bellman core parameterized by
  resources, a choice utility and an income-only flow. The value and pricing
  fixed points retain exact 6.0.1 parity for dense and monotone short-term debt
  and the smoothed long-term operator. Short-term paths are also unchanged;
  long-term paths now follow the pricing operator’s debt-choice probabilities.
- Sovereign brief resumes validate the baseline and scenario overrides before
  reusing a cached case, preventing stale results from being relabeled.
- Fiscal grid caches now include the effective calibration and policy-search
  settings in both solution and simulation identities. The 6.0.1 parity harness
  independently pins the legacy simulation algorithm on each platform; archived
  floating-point data allow only the documented cross-platform rounding bound.
- `check_arellano_grids.paired_comparison` accepts a `tolerances` mapping;
  its default is unchanged.
- A bilingual sovereign-risk policy brief,
  [`docs/brief_sovereign_risk.md`](docs/brief_sovereign_risk.md) (Spanish:
  `docs/es/brief_sovereign_risk.md`). It shows how the market-reentry
  probability, income volatility and income persistence move average debt,
  spreads, default frequency and the value of market access in the validated
  short-term Arellano model. These are illustrative comparative statics of the
  public calibration, not a claim about any country. Every number is shown with
  its 95% interval across 16 seeds and its paired 3,201 -> 6,401 grid change.
  The numbers come from a frozen converged sweep,
  `tests/fixtures/sovereign_brief/sweep.json`.
- `puremacro.sovereign.ergodic_moments(result, *, periods=500_000,
  seeds=range(1979, 1995), burn=1_000)`: raw ergodic debt (% of mean quarterly
  income), mean spread,
  annual default-entry frequency and spread SD, per seed and with 95%
  t-intervals. It reproduces `SovereignDefaultResult.simulate` draw for draw
  and runs 16 x 500,000 quarters in seconds. Continuing exclusion is not counted
  as default.
- `puremacro.sovereign.value_of_market_access(result)`: the
  consumption-equivalent value (in %) of market access at zero debt, compared
  with permanent autarky without a default cost.
- Example `python -m puremacro.examples.first_brief_sovereign`:
  `--scenario theta=0.15` (one scenario on the converged 6,401 x 641 grid),
  `--sweep --coarse` (all seven scenarios at 801 x 161 in minutes, with a
  coarse-grid caveat), `--output DIR`, `--lang en|es`, `--grid`, `--periods`,
  `--n-seeds`, `--from-record PATH`, and `--freeze PATH` for maintainers.

- Sovereign brief outputs use explicit UTF-8 and portable resource diagnostics.
  Custom scenarios appear in figures, and effect directions compare numerical
  values while treating indistinguishable displayed values as flat.
- Added a first policy brief, `python -m puremacro.examples.first_brief_tariffs`:
  a 10% US tariff on Chinese goods solved with the exact trade engine on the
  bundled OECD ICIO 2019 table, reporting national EV with its purchaser-price,
  factor-income and fiscal split, one alternative assumption (`--experiment
  sigma=2`) and household incidence under proportional and equal rebates. The
  default household groups are synthetic, labelled "SYNTHETIC, method
  illustration only"; no survey data ship, and `--groups CSV` takes your own.
  Outputs carry a manifest with versions, the table's citation and hash, solver
  settings, convergence and timings. New bilingual page "Your first policy
  brief" shows the frozen result, which a test re-renders from the fixture.
  Credit the table as OECD (2023), OECD Inter-Country Input-Output Tables.

- Tariff-brief resource diagnostics work on Windows; reruns remove obsolete
  generated experiment tables and Spanish figures use decimal commas.
- Faster annual-to-quarterly panel expansion preserves duplicate precedence,
  missing-value handling and provenance; BVAR-SV forecasts reuse their design
  matrix with unchanged seeded paths.
- Course scatter charts support arrow keys, Home/End and Escape. Lab CSV buttons
  retain bilingual accessible labels; button transitions respect reduced motion.
- Bilingual roadmap/API guidance now describes the published 6.0.1 release and
  6.1.0 development. CI checks native/Windows typing before numerical tests and
  verifies representative workflows from a clean installed wheel before publish.

## 6.0.1 (2026-10-29)

**Reliability release: corrected sovereign timing and model fidelity, a
grid-converged Arellano benchmark, and a certified fast sovereign solver.**

This is the first version published to PyPI since 5.7.1. The 5.8.0, 5.9.0 and
6.0.0 tags were never published (the 6.0.0 release failed CI on Python 3.11
typing), so installing 6.0.1 from 5.7.1 also brings everything listed under
those three versions below.

### Fixed

- Fixed sovereign simulation timing: a successful reentry draw permits the
  current period's repayment/default decision, matching the Bellman equation.
  Public simulated paths now start and reenter at zero debt on saving grids.
  Solver-generated grids preserve both bounds and the requested size while
  including an exact zero node; mixed saving/borrowing grids need at least
  three points. Reentry probabilities must be finite and within [0, 1].
  Simulation metadata states timing, seed, sample and annualization conventions.
- Sovereign default now supports `sigma_eps=0`, uses stationary mean income for
  the Arellano penalty, reenters at zero debt even on grids allowing saving, and
  returns default decisions consistent with the final values. Spread correlations
  are reported without sign changes; undefined correlations are `NaN` with status
  metadata. No grid-average statistics substitute for missing simulation data.
- Sovereign restructuring preserves the original income discretization,
  default-cost calibration, recovery assumptions and solver settings. Recovery
  paths use the model's actual default output and return to zero debt.
- Model adapters now evaluate the supplied first-order DSGE or BlockDAG
  likelihood. Scores use native Kalman sensitivities or central differences of
  native Whittle likelihoods; they are not universally end-to-end DualArray AD.
  Observed data and parameter names are checked. Unsupported native operations
  fail explicitly instead of returning synthetic substitutes. `model=None`
  demonstrations emit warnings.
- The default nonlinear HANK DAG and aggregate-risk calculations are explicitly
  reduced demonstrations with prescribed response formulas and moments. Unknown
  nonlinear models and unsupported aggregate-risk model/steady-state arguments
  are rejected. The legacy `asymmetry_ratio` is an unclipped nonlinear/linear
  peak ratio, not an independently solved opposite-shock comparison.
- Fixed Python 3.11/NumPy 2.4 type-checking, retired notebook imports and isolated
  offline LLM-cache replay. Corrected bilingual guides and notebook explanations,
  including bridge identity checks and separately computed policy scenarios.
  Notebook 73 distinguishes grid and simulated statistics, uses the long-term
  risk-free price and labels maturity extension as a fractional duration change.
  Collateral amplification diagnostics and term-structure residual components
  retain explicitly documented experimental limitations.

### Added

- `solve_sovereign_default(policy_search="monotone")`: for short-term debt, a
  monotone policy search replaces the dense n_b × n_b × n_y tensors (about
  210 GB each at 6,401 × 641), returns the same values as the dense search (and
  the same policies except at exact value ties between choices with different
  bond proceeds), and is certified after convergence by a blocked exhaustive search (a gap above
  1e-11 raises). The converged 6,401 × 641 public design now solves in 28
  minutes and 2.4 GB, matching the independent reference solution. The default
  remains `"dense"`; convex utility (gamma < 0) is rejected. Price-schedule
  derivatives `dq_db` are now computed in one vectorized step instead of a
  per-node loop; values agree with the loop up to rounding (about 1e-15).
- Arellano grid convergence: the independent reference solver gains a monotone
  policy search certified by an exhaustive Bellman check, which makes debt grids
  of 6,401 nodes affordable. For the public calibration, grid convergence is now
  established at 6,401 × 641 under the pre-registered paired-seed tolerances,
  after the registered precision escalation. The passing record is frozen and
  machine-checked against the bilingual guide. At the archived calibration the
  published spread volatility halves on fine grids, the pattern documented by
  Hatchondo, Martinez and Sapriza (2010). `run_arellano_study --income-states`
  reproduces the 161- and 321-state rows. Notebook 73 states that grid
  resolution is part of the result.
- Added an offline Arellano (2008) validation study using actual archived Fortran
  and MATLAB/Octave references, published-moment comparisons, independent-seed
  simulation intervals and debt/income grid refinements. The bilingual guide
  records the archived calibration's grid sensitivity and distinct
  exclusion/measurement conventions; no published-moments replication or new policy result is claimed.
- Added independent public-calibration parity and paired-seed economic grid
  convergence checks, separating discretization shifts from simulation error.
- Independent matched-grid references cover short-term deterministic-default
  equations and one-asset household policies, distributions and Jacobians.
  These checks do not certify complete published calibrations or policy moments.

### Internal

- Release gate: a default-on Gate 8 runs mypy under the Python 3.11 floor
  interpreter, natively and as `--platform win32`; `--no-tests --mypy-floor`
  adds it to the fast run. The v5.4.0 and v6.0.0 tags passed every local gate
  and failed only the 3.11 CI legs on typing. The release procedure now pushes
  `main` and waits for green CI before the tag. The headless fan-chart test
  closes stray figures before asserting that none leak.

## 6.0.0 (2026-10-25)

**Major Release: Unified Differentiable Model Protocol, Vectorized Tempered SMC Engine, Global Nonlinear HANK with Aggregate Risk, Quantitative Sovereign Default & Sudden Stop Suite, Distributional Local Projections with Exact Adding-Up, Integrated Spatial Climate IAM, and Bilingual Flagship Showcase Notebook 72.**

### Added

- `puremacro.core.protocol` (`DifferentiableModel`, `StateSpaceProtocol`, `wrap_differentiable_model`, `LinearDSGEAdapter`, `BlockDAGAdapter`, `DynamicSpatialAdapter`, `NowcastDFMAdapter`) provides a unified five-method differentiable modeling protocol (`.impulse_response()`, `.simulate()`, `.loglik_and_grad()`, `.counterfactual()`, `.to_interactive_html()`) interoperable with `DualArray` forward automatic differentiation score gradients.
- `puremacro.inference.smc` (`smc_sample`, `SMCResult`, `AdaptiveTemperingSchedule`, `TemperingStageDiagnostics`, `ImportanceWeights`, `RWProposal`, `MALAProposal`, `HMCProposal`) provides a vectorized, tempered Sequential Monte Carlo estimation engine with adaptive ESS targeting, MALA/HMC analytical score gradients, and marginal data density ($\ln p(Y)$) recovery.
- `puremacro.dsge.hank_nonlinear` (`solve_hank_nonlinear_transition`, `solve_hank_aggregate_risk_order2`, `NonlinearHANKTransitionResult`, `HANKAggregateRiskResult`) provides global nonlinear sequence-space general-equilibrium transitions with Good Broyden rank-1 Jacobian updates and 2nd-order aggregate risk perturbation with Winberry–Reiter moment reduction.
- `puremacro.sovereign` (`solve_sovereign_default`, `solve_sudden_stop`, `decompose_spread_term_structure`, `evaluate_debt_restructuring`, `SovereignDefaultResult`, `SuddenStopResult`, `RestructuringResult`) provides quantitative sovereign default under short- and long-term debt, extreme-value taste shocks, `DualArray` dilution derivatives $\partial q/\partial b' \le 0$, Mendoza (2010) collateral sudden stops, spread term-structure decomposition, and debt restructuring counterfactuals.
- `puremacro.lp.distributional` (`estimate_distributional_lp`, `estimate_quantile_lp`, `bridge_hank_distributional_irf`, `DistributionalLPResult`, `DistributionalBridgeResult`) provides functional and distributional local projections enforcing exact cross-sectional adding-up constraints ($\max_h |\sum_b \beta_{b,h}| < 10^{-12}$) via null-space projections, Driscoll–Kraay panel covariance inference, and structural HANK decile bridges.
- `puremacro.climate.spatial_iam` (`SpatialClimateIAM`, `solve_spatial_climate_iam`, `SpatialClimateIAMResult`) couples dynamic spatial trade and migration hat-algebra with a mass-conserving multi-box carbon cycle, regional temperature damage functions, Carbon Border Adjustment Mechanisms (CBAM), and analytical Golosov et al. (2014) Pigouvian carbon taxes.
- Showcase Notebook 72 (`notebooks/72_macro_v6_flagship_workflows.py` and `notebooks/72_macro_v6_flagship_workflows_es.py` + compiled `.ipynb` builds) uniting all six v6.0.0 tracks with a 3-panel publication figure, 14-check scorecard, and interactive parameterized exercises.

### Removed

- Scheduled v6.0.0 deprecations permanently removed:
  * `puremacro.lp.garch_utils` (migrated to `puremacro.lp._garch_utils`).
  * `SigmaObject` in `puremacro.sigma.sigma_numpy` (migrated to `puremacro.volatility.sigma.SigmaObject`).

## 5.9.0 (2026-10-25)

**Pure-NumPy forward AD (`DualArray` / `HyperDualArray`), generalized nonlinear DAG Sequence-Space Jacobians & Whittle HANK estimation, 2nd/3rd-order pruned nonlinear OccBin, Baqaee–Farhi 2nd-order network aggregation, dynamic spatial migration hat-algebra, mixed-frequency nowcast news gradients, self-auditing `PolicyStudio` counterfactual grids, zero-server interactive HTML5/SVG explorer, and bilingual showcase notebook 71.**

### Added

- `puremacro._autodiff` (`DualArray`, `HyperDualArray`, `seed_dual_vector`,
  `jacobian_forward`, `value_and_grad`, `hessian_forward`, `dual_solve`,
  `dual_sylvester_solve`) provides a pure-NumPy vectorized forward-mode automatic
  differentiation engine supporting `@` (`__matmul__`), NumPy ufuncs, implicit
  linear solves, and Sylvester equations, wired into
  `puremacro.vfi.analytic_gradients` (`diff_method="autodiff"`) and
  `puremacro.dsge._gradients.sylvester_order2_tangent`.
- `puremacro.dsge.ssj_dag` (`SimpleBlock`, `HetBlock`, `SolvedBlock`, `BlockDAG`,
  `SSJDAGResult`) and `puremacro.dsge.hank_estimate` (`hank_loglik_whittle`,
  `estimate_hank_whittle`, `estimate_hank_irf_matching`, `HANKEstimationResult`)
  for composable general-equilibrium Sequence-Space Jacobians, damped quasi-Newton
  nonlinear DAG transitions, continuous-time HJB/KFE `HetBlock` integration, and
  frequency-domain Whittle / IRF-matching HANK estimation.
- `puremacro.dsge.occbin_nonlinear` (`solve_nonlinear_occbin`,
  `NonlinearOccBinResult`) extending piecewise-linear OccBin to 2nd- and 3rd-order
  pruned perturbation systems with precautionary risk shifts.
- `puremacro.trade.network_macro` (`baqaee_farhi_decomposition`,
  `solve_exact_ces_network_prices`, `NetworkMacroDecompositionResult`) for
  Baqaee–Farhi (2019, 2020) 2nd-order nonlinear input-output macro aggregation,
  analytical macroeconomic Hessians, and markup allocative efficiency wedges.
- `puremacro.spatial.dynamic_hat` (`DynamicSpatialHatModel`,
  `solve_dynamic_spatial_hat`, `DynamicSpatialHatResult`) for Caliendo–Parro–Rossi-Hansberg
  (2019) dynamic spatial migration hat-algebra transitions.
- `puremacro.study.studio` (`CounterfactualGrid`, `PolicyStudioCell`,
  `PolicyStudioResult`, `run_policy_studio`) and
  `puremacro.reports.result_to_interactive_html` for self-auditing counterfactual
  grids and zero-server interactive HTML5 + inline SVG explorers.
- Bilingual showcase notebook 71 (`notebooks/71_frontier_macro_workflows.py` and
  `notebooks/71_frontier_macro_workflows_es.py`) and end-to-end verification suite
  (`tests/test_e2e_frontier_v59.py`).

## 5.8.0 (2026-10-24)

**Exact Sylvester factorization reuse, sparse transition JVP caching, separable Chebyshev fitting, and constrained Newton–Krylov.**

### Added

- Ordered `.mod` execution now supports mid-script parameter reassignments between
  commands, multi-lag indexed `histval` (`k(0)`, `k(-1)`, ...) and lead-indexed
  `endval` mapped onto parser-generated `AUX_LAG_*` / `AUX_LEAD_*` / `AUX_EXO_LAG_*` /
  `AUX_EXO_LEAD_*` variables, and CLI execution via
  `puremacro-dynare execute <file.mod> [--outdir DIR] [--data CSV]` (or
  `puremacro-dynare <file.mod> --ordered`).
- `diagnose_perturbation(solution, max_curvature_order=3)` reports higher-order
  Schur-block conditioning (`schur_block_condition`), full Kronecker curvature
  operator conditioning (`curvature_operator_condition`, `curvature_coupling_amplification`,
  `curvature_conditioning`), normalized Sylvester equation residuals
  (`sylvester_residuals` for `ghxx`, `ghs2`, `ghxxx`, `ghxss`, `ghuss`), and
  work counts (`sylvester_work` / `SylvesterWorkCounts`).
- `solve_perfect_foresight(..., linear_solver="krylov")`, `solve_transition`, and
  `StackedProblem` now support `terminal_policy` / `terminal_jacobian` and
  `mcp=True` / `mcp_bounds` / `mcp_equations` via active-row preconditioned
  semismooth Newton–Krylov, with physical bound/complementarity diagnostics
  (`physical_mcp_feasibility`, `physical_complementarity_residual`,
  `physical_residual_norm`), `physical_scale` / `normalized_horizon_tolerance`,
  and compiled sparse derivative value-reuse across Newton steps (`TrajectoryJacobian`).
- `solve_time_iteration` and `EquilibriumProblem` add analytic expected-residual
  Jacobian callbacks (`transition_jacobian`, `residual_jacobian`, and automatic
  compiled equation derivative reuse in `EquilibriumProblem.from_model`),
  domain-safe feasible backtracking (`trial_rejections` in `history` and
  `callback_evaluations`), separable 1D Kronecker LU Chebyshev fitting
  (`collocation_matrix_allocated`, `storage_bytes`), and batched shock integration.
- Bilingual showcase notebook `notebooks/70_nonlinear_dsge_workflows_and_accuracy.ipynb`
  (`notebooks/70_nonlinear_dsge_workflows_and_accuracy_es.ipynb`) demonstrating
  exact Sylvester conditioning diagnostics, separable Chebyshev global time
  iteration with analytic Jacobians, matrix-free Newton–Krylov MCP and
  `terminal_policy` transitions, and ordered `.mod` execution with mid-script
  parameter reassignment and lag-indexed `histval`, plus Sylvester residual and
  LU cache-reuse checks in `notebooks/68_third_order_perturbation_trust.ipynb`.

### Fixed

- The higher-order Bartels–Stewart Sylvester solver (`puremacro.dsge._sylvester`)
  now keys its LU cache by exact IEEE-754 floating-point values (`float` and
  4-tuples of `float`) instead of 12-decimal rounded scalars, eliminating the
  near-unit-root factorization collision reported in 5.7.1.
- `_compile_dag_derivatives` and `TrajectoryJacobian` now key their compiled
  derivative caches by symbolic equation structure rather than `id(dag)`,
  preventing stale cache hits after garbage collection.

## 5.7.1 (2026-10-23)

**Fix the opt-in course test bootstrap and publish the retained 5.7.0 features.**

Version 5.7.0 was publicly tagged but was not published to PyPI. Its public tag
and source history remain unchanged. This patch retains the 5.7.0 features
below, including the earlier unpublished 5.6.0 development changes; it does
not change the numerical solvers or economic notebook equations.

### Fixed

- The opt-in T02_F macro-finance notebook test now bootstraps the current
  checkout under test instead of executing the frozen 5.5.0 course bootstrap
  against a newer package. The original course notebook keeps its historical
  edition pin.
- The scheduled/manual showcase notebook test now selects the `showcase`
  suite explicitly, matching its documented discovery scope. Its assertions
  and test identity remain unchanged; historical course validation retains
  its separate execution records.
- The eight amended T02_A–D notebooks and their current setup instructions now
  pin the package, models and helpers to `v5.7.1`. Their new offline execution
  record is separate from the preserved 5.7.0 release-source and development
  records.

### Known issues

- The pre-existing higher-order Sylvester LU cache rounds scalar keys to
  12 decimal places, so distinct Schur products can reuse a factorization.
  A determinate analytic model with roots near `0.9999998` shows relative
  second-/third-order coefficient errors up to approximately `3.8e-7`.
  The existing near-unit-root warning does not quantify this curvature error.
  There is no general drop-in public-API workaround: validate affected
  higher-order coefficients against an independent analytic/dense reference
  and do not assume full floating-point precision near unit roots. Exact cache
  keys and independent operator regression checks are targeted for **5.7.2**;
  that correction is planned, not included here. The
  [release record](https://github.com/jalonso1979/puremacro/releases/tag/v5.7.1)
  includes a public-API reproducer. This example does not demonstrate a general
  unit-invariance failure.

## 5.7.0 (2026-10-23)

**Use one nonlinear model for ordered `.mod` programs, deterministic transitions and comparable local/global policies.**

This release also includes the 5.6.0 development changes below; 5.6.0 was not published to PyPI.

### Added

- `dsge.read_mod` and `NonlinearModel` retain equations, timing and source
  metadata without requiring a perturbation solution. The same residuals feed
  steady states, local approximations, perfect foresight and global policies;
  Python equation callbacks and models without innovations are supported.
- Ordered deterministic `.mod` programs preserve `initval`, `histval`, `endval`,
  steady states, deterministic shock declarations and repeated
  `perfect_foresight_setup`/`perfect_foresight_solver` snapshots. Unsupported
  commands and options fail preflight rather than disappearing from execution.
- `solve_transition` adds adaptive continuation in shocks, terminal values and
  calibration, economic domains, equation/variable scaling, explicit terminal
  policies and budgeted horizon checks. `solve_announced_path` separates news
  from later surprises. Compiled equations supply analytic path derivatives;
  MCP equation mappings and retained audit evidence expose binding constraints.
- Global time iteration connects explicit state/control mappings of the same
  equations to Chebyshev, Smolyak and linear interpolation, Gaussian quadrature,
  constraints and finite refinement budgets. Independent policy audits report
  equilibrium/KKT errors, infeasibility and probability leaving the domain.
  Local orders 1–3 can be evaluated and compared on the same audit points;
  conditioning diagnostics supplement the existing determinacy checks.
- English/Spanish guides explain mathematical timing, shapes, solver selection,
  compatibility limits, runnable examples, diagnostics and reproducibility.
- MAV T02_A–D in English and Spanish use the new workflows, with common model
  helpers and a fresh-process offline execution runner. T02_A compares equation
  stacking, `.mod` transitions, announcement timing and local/global policies;
  T02_D uses the library transition solver for irreversible investment.

### Fixed

- The MAV neoclassical and RBC `.mod` examples now retain future marginal utility
  inside the Euler expectation. Perfect-foresight paths and first-order rules
  are equivalent, while higher-order stochastic solutions now use the stated
  expected marginal return instead of a certainty-only consumption ratio.
- Corrected English/Spanish stacked Newton-Krylov documentation and the module
  and architecture descriptions to reflect the existing perfect-foresight and
  extended-path integration, supported options and numerical failure behavior.
- One-asset EGM now checks the affine budget and supported consumption constraint
  structurally, including intermediate local expressions. Nonlinear budgets,
  asset-dependent custom functions and extra constraints can no longer pass
  merely because they agree at asset grid nodes, then produce invalid continuous
  policies or nonfinite life-cycle values. The next-asset coefficient is checked
  independently of income scale at each calibration, age and discrete choice.
  Custom functions of exogenous income remain supported; general resource
  equations and constraints remain available through VFI.

### Known limits

- These are supported subsets of the Dynare language and solution workflows.
  Global policies require explicit state/control mappings and a chosen domain;
  convergence or passing a finite audit does not establish a globally unique
  equilibrium. Newton-Krylov currently rejects MCP and terminal-policy callbacks.
- The amended T02_A–D notebooks pin their source, models and helpers to `v5.7.0`.
  Their execution evidence covers local Python; it does not certify Colab, Juno
  or a browser notebook session. The full original course remains pinned to 5.5.0.

## 5.6.0 (2026-10-22)

**Write dynamic programs in `.mod` files and preserve complete empirical studies and model scenarios.**

### Added

- Dynare-style dynamic-programming files: `dp.load_mod`, `dp.solve_mod` and
  `puremacro-dp` compile explicit state grids, income processes, objectives,
  constraints and aggregates to the existing VFI, EGM and HJB engines. Loading
  validates without solving; scripts/macros/includes are unsupported. Packaged
  growth, household and lifecycle examples and English/Spanish guides are included.
- `puremacro.study` records explicit LP/recursive-VAR specifications, input
  transforms and units, identification assumptions, effective samples and audit
  evidence. Portable checksum-verified archives and Markdown/LaTeX/Typst reports
  retain inputs; model scenarios remain distinct from statistical estimates.
- Editable HANK aggregate equations with `closure="model"`, explicit household
  input/output mappings and declared shock paths. Homogeneous affine aggregate
  equations can use linear household Jacobians or nonlinear household transitions
  with bounded Broyden iterations. Two-asset EGM solutions now simulate joint
  household panels with bilinear policies and consumption from the budget.
- Rank-normalized folded split R-hat, between-chain bulk/tail ESS and mean MCSE,
  integrated into NUTS summaries. Frozen actual ArviZ reference outputs cover
  offsets, scale differences, ties, heavy tails and serial dependence. Degenerate
  draws are unavailable rather than reported as healthy convergence.
- Narrative signal document/cluster draws, separately labelled specification
  sensitivity, train/holdout calibration and full-path callback propagation.
  Measurement uncertainty remains distinct from estimator sampling uncertainty.
- Optional spatial-LP circular block-pairs bootstrap-t inference with explicit
  stationarity and common-support estimates; a separate GVAR bootstrap reference
  for the stronger independent-country null, refitting both estimation stages.
  Default asymptotic diagnostics remain unchanged. Reproducible size/coverage/power
  experiments retain failures and Monte Carlo uncertainty.
- Three offline workflow templates: a frozen OECD/FRED Mexican recursive VAR,
  a DP household transfer experiment, and an OECD ICIO tariff counterfactual.
  Inputs, units, assumptions and audit limits are retained in report archives.
  Executed English/Spanish notebook 69 covers all three studies, response plots,
  report exports, archive reloads and seeded reproducibility checks.
- Perfect-foresight and nonlinear HANK path audits, plus explicit finite
  refinement ladders with solve/cost budgets. Missing paths or Euler evidence
  remain visible; default auditing never re-solves a model.
- `linear_solver="krylov"` in standard perfect-foresight and extended-path calls,
  with validated options, retained diagnostics and no silent direct-solver
  fallback. Workload scaling benchmarks record numerical checks, repeated
  timings, environment and the scope of memory measurements.

### Known limits

- The new spatial and GVAR bootstrap references are experimental and specific to
  their stated nulls and sampling assumptions. Preset simulations show spatial
  simultaneous undercoverage (88.5% in one design with nominal 95% coverage) and
  weak GVAR power against one integrated leakage alternative (3%). See the
  bilingual dependence-bootstrap guide for the full results and uncertainty.
- Editable HANK closures require homogeneous affine aggregate equations, even
  when household transitions are nonlinear. DP `.mod` files use PureMacro's
  own supported dialect, not a Dynare dynamic-programming extension.
- Numerical convergence and reproducible archives do not establish economic
  validity or causal identification.

## 5.5.0 (2026-10-22)

**Inspect model compatibility and numerical accuracy, and correct steady-state trajectory references.**

### Added

- `puremacro.dsge.inspect_mod` and `puremacro-dynare inspect` report supported,
  ignored, unresolved and unsupported syntax without numerical evaluation or
  solving. Deterministic JSON/text reports retain source locations, include and
  macro-loop provenance, data requirements and bounded preprocessing failures.
- Opt-in `load_mod(command_mode="ordered")` and
  `model.execute_mod_commands(ordered=True)` preserve separate fixed-calibration
  simulation, smoother and shock-decomposition invocations, their explicit and
  effective options, variable selections and full result snapshots. Unsupported
  programs fail preflight; numerical failures retain completed invocation evidence.
  The default legacy runner is unchanged.
- `puremacro.validation.audit_result` separates solver convergence, recomputed
  equations, distribution stationarity, derivative checks and refinement evidence.
  Initial adapters cover stationary one- and two-asset EGM, first-order DSGE and
  existing dynamic-trade certificates. Reports export DataFrames, Markdown and
  strict JSON with explicit `not_checked`/`unsupported` states.
- Household audits expose budget and Euler/KKT maps, worst-state coordinates,
  stationary-weighted error quantiles and mass at artificial grid bounds.
  `DerivativeCheck`, `ResidualCheck` and `RefinementCheck` support explicit
  experiments; default audits never re-solve, refine or use the network.

### Fixed

- `STEADY_STATE(expr)` now uses a fixed reference in nonlinear trajectory and
  derivative evaluation, including perfect foresight/MCP, surprise shocks,
  extended path and OccBin. It previously used current path values despite its
  zero derivative, potentially producing wrong paths ([#177](https://github.com/jalonso1979/puremacro/issues/177)).
  Native Dynare fixtures cover the reported example, an RBC fiscal rule and
  terminal assignments; see the bilingual correctness advisory.
- A bare `.mod` `endval` retains literal terminal values; following `steady` or
  the supported `endval_steady` option requests a terminal equilibrium solve.
  Residual callbacks retain unrelated defaults and propagate their own errors.
  Exogenous variables inside `STEADY_STATE()` are explicitly rejected, as in Dynare.
- Two-asset EGM recognizes its supported resource equation structurally, binds
  asset roles and parameter aliases consistently, and reads discount/curvature
  parameters from the declaration. Arbitrary budget functions cannot be accepted
  solely because they match a finite set of probes.
- Empty Bartik base-year shares avoid pandas/PyArrow's empty-key join failure
  while retaining key dtypes and the numeric share schema.
- Flat-cache migration commits bounded insertion batches before removing any
  original files; existing database keys retain their source files, and a
  transaction failure leaves all originals available for recovery.
- Colab offload links show keyboard focus and hover states. CSV downloads in
  the teaching labs have descriptive English and Spanish accessible labels.

### Performance and maintenance

- Quarterly-to-monthly panel projection batches repeated work while preserving
  mixed date formats, local calendar dates, numeric conversion, output columns
  and duplicate precedence.
- VAR bootstrap lag stacks and BVAR-SV forecast buffers avoid repeated
  concatenation, preserving seeded results. Narrative keyword scoring reuses a
  bounded cache of escaped, whole-word patterns without changing overlap or
  negation rules.
- Consolidated regression coverage for weighted statistics, country codes,
  Markov closed classes and transient states, cache migration and panel projection.
  Broad refactors and a slower document-store batching proposal were not adopted.

### Scope

- Ordered `.mod` execution supports literal options and fixed calibration;
  policy, perfect-foresight, HANK and interleaved state changes require their
  explicit Python workflows. Compatibility inspection is not numerical validation.
- Audit tolerances describe individual checks, not an overall accuracy guarantee.
  Coarse household grids can pass budget/stationarity checks while failing moment
  refinement. Expensive derivative and grid/horizon experiments require opt-in.

## 5.4.2 (2026-10-22)

**Release of the 5.4.1 changes.** The `v5.4.1` tag was never published:
python-dateutil 3.0.0 appeared on PyPI 26 minutes before the tag, and with it
pandas 3.0.6 cannot localise any dateutil timezone (pandas reads a UTC offset
that dateutil 3 returns as `None`), so a store test that builds a dateutil
timestamp failed on the release run. No library code changes.

### Internal

- `test_dateutil_named_timestamp_retains_zone_offset_and_nanoseconds` skips, with
  the reason, when the installed pandas cannot do arithmetic on a dateutil
  timestamp. pandas fails before puremacro is involved, so the store path the
  test covers is unreachable for that combination; it runs normally with
  python-dateutil 2.x or a pandas release that supports dateutil 3.

## 5.4.1 (2026-10-20)

**Python 3.11 typing and Windows CI compatibility, plus the defect fixes from the
20 October 2026 review.** This patch includes the 5.3.0 and 5.4.0 feature set. The `v5.4.0` tag failed five of the nine CI legs
and was never published: the type-checking gate failed on all three Python 3.11
legs, where pip resolves NumPy 2.4 (NumPy 2.5 requires Python 3.12) and its
stubs differ from NumPy 2.5's, and the three Windows legs also failed on
Windows-only code paths. No numerical result changes in the library; the course
model `rbc_canonico.mod` is recalibrated (see Fixed).

### Fixed

- Make QR-only array returns, spatial Boolean reductions, tariff-export labels,
  and plotting-axis containers explicit to the type checker across NumPy 2.4
  and 2.5. Numerical algorithms and tolerances are unchanged.
- Convert the bounded SMC rejection-draw count to a Python integer, preserving
  the seeded draws and subsequent random-number sequence.
- Guard the POSIX-only `resource` and `os.sysconf` fallbacks in the GPU memory
  report and the runtime capability probe behind `sys.platform`, so mypy on
  Windows no longer reports them. Windows already took the zero and `None`
  paths through the caught exceptions.
- Read and write the course-site build's HTML, JavaScript and JSON files as
  UTF-8 explicitly. On Windows the default code page (cp1252) could not decode
  the accented characters in the lab pages and the build failed.
- Resolve the symlink itself, not its `os.readlink` target, in the research-guard
  symlink test; Windows returns the extended-length `\\?\` form from `os.readlink`
  and `Path.resolve()` keeps that prefix when the input already carries it.
- `lp_moment_targets` accepts both the pandas 2 (`'Q'`, `'M'`, `'Y'`) and pandas 3
  (`'QE'`, `'ME'`, `'YE'`, which need pandas 2.2 or later, or `'QS'`/`'MS'`/`'YS'`)
  offset spellings of `frequency` for a `DatetimeIndex`, builds the comparison
  grid at the index's own resolution (a regular index stored at second or
  microsecond resolution was refused as non-consecutive on pandas 2.x), and
  reports an unrecognised alias with its own `ValueError` naming the index kind
  instead of pandas' internal message, including the `AttributeError` pandas
  2.0/2.1 raise for an offset alias such as `'QS'` on a `PeriodIndex`.
  `PeriodIndex` callers using period aliases and the `structural_irf_matching`
  example were already correct.
- `HANKModel.simulate(nonlinear=True)` and `solve_hank_bridge(..., nonlinear=True)`
  on the built-in two-asset household (`hetagent_block` with `assets = 2`, no
  `household_model`) now raise `NotImplementedError` instead of silently
  returning the linear sequence-space transition with `converged=True`. The
  Broyden solver covers the built-in one-asset household only; the DP-bridge
  rejection already existed.
- `CaliendoParroModel.solve_counterfactual` and the `TradePolicySimulator.simulate_*`
  methods that call it issue a `RuntimeWarning` when the wage tatonnement stops
  at `max_iter` without converging; they used to return `converged=False`
  silently. Returned values are unchanged. `max_iter` must be an integer >= 1
  (`ValueError`); `max_iter <= 0` used to crash with an `UnboundLocalError`. This
  warning and the HJB one below point at the caller's line, as the Krusell-Smith
  warning does.
- `solve_hjb_achdou` (and `dp.Model.solve(method="hjb")`) emits a `RuntimeWarning`
  naming the iteration cap and the final sup-norm when `max_iter` is exhausted
  instead of setting `converged=False` silently; the returned `HJBSolution` is
  unchanged.
- The course model `curso/notebooks/modelos/rbc_canonico.mod` sets `mu = 1.455567`,
  the value its header, `initval` block and lesson T02_B derive for `l_ss = 0.33`.
  The shipped `0.822516` made `steady;` walk away from the `initval` to
  steady-state hours of 0.443 and a 0.285% impact response of hours to a
  one-standard-deviation TFP innovation, instead of the 0.393% the RBC lab and
  T02_B report; the `initval` `w` moves to the analytic 2.020270. The notebooks
  and the lab were unaffected because they recalibrate `mu` themselves.
- `tools/release_check.py --examples` (Gate 5) accepts a FAIL listed in the new
  `tests/known_example_failures.json` when the reason recorded in
  `docs/examples_gallery.json` still starts with the entry's `reason_prefix`,
  reporting it as `known FAIL (documented)`; an unlisted FAIL or a drifted reason
  still fails the gate, and a listed example that now passes is reported for
  removal, mirroring `tests/known_failures.json`. The only entry is the RR2010
  example's published t-statistic rounding miss; the example still exits
  nonzero and the gallery still records it as FAIL, so the gate can now pass
  without hiding anything.

### Documentation

- Align the citation abstract with supported local execution and best-effort
  browser execution through Pyodide.
- Explain the native `.mod` runner's category-based dispatch, merged repeated
  simulations, and possible omission of unrecognized outer statements in both
  English and Spanish. Full source-order execution remains a future improvement.
- Record in `RELEASING.md` the two gaps the local pre-tag gate left open: it ran
  under one Python (3.13, NumPy 2.5 stubs) and one platform. The pre-tag recipe
  now also runs mypy under the oldest supported interpreter and with
  `--platform win32`.
- The README, `docs/dsge_v3.md`, `docs/hank_nonlinear.md` (English and Spanish) and
  the `puremacro.dsge.hank` docstrings state that the HANK bridge's `model;`
  block is parsed onto `HANKModel.dag` but not solved: the closure is the
  built-in Phillips curve and Taylor rule on the declared `beta`, `kappa`,
  `phi_pi` (`alpha_ab` for two-asset) with a single monetary shock path, and the
  `shock` argument only labels the result (a `transition_paths` column appears
  only for a declared `varexo`). A test in
  `tests/test_v300_hank.py` pins this so the docs and the behaviour move in step.
- `docs/dp.md`, `docs/es/dp.md` and the `dp.Model.solve()` docstring describe the
  household-solver non-convergence contract: the one-asset `vfi`/`egm` engines
  and DC-EGM carry no `converged` field and raise `RuntimeError` when `max_iter`
  is exhausted; the two-asset EGM solver and `solve_hjb_achdou` emit a
  `RuntimeWarning` and return the last iterate with `converged=False`, as does
  the Krusell-Smith loop at `max_outer`; `dp_sequence_jacobian` and
  `dp_nonlinear_transition` refuse an unconverged two-asset household.
  `docs/vfi_hjb_continuous.md` and its Spanish twin no longer say the HJB solver
  does not warn.
- `docs/structural_bridge.md` and its Spanish twin state which `frequency`
  aliases `lp_moment_targets` accepts for a `PeriodIndex` and a `DatetimeIndex`;
  `docs/policy_simulators.md` and its twin note the new non-convergence warning.
- `RELEASING.md` (Gate 5 row) and `CONTRIBUTING.md` (Gate 5 section) describe
  `tests/known_example_failures.json` and the pinned reason prefix.
- Course text pass: the syllabus table in `T00_syllabus{,_es}.ipynb` lists the
  labs that exist (`T04_C`, `T02_E`/`T02_F`, `T05_G`, `T00_B`; no `T04_B`) with its
  stored output re-rendered in step; the Colab hints for `puremacro.text` (T03_D,
  T03_E, T03_F, T09_B) and the comments around the `puremacro.dp` fallback
  (T03_A, T06_A, T06_B) name the pinned development commit `4a726629` the course
  README declares instead of an unpinned repository install, and no longer call
  5.x unpublished; the browser playground's start page describes the companion
  lessons as Spanish-only and the notebooks as network-free rather than "fully
  synthetic"; the inline `multi_shock_mod` in `T02_B` (EN/ES) now carries the
  same `mu = 1.455567` as the model file (the notebook already overrode it
  through `params=`, so its outputs are unchanged).

### Internal

- The dedicated CI mypy step runs on Ubuntu/Python 3.11 as well as 3.12, so the
  NumPy 2.4 stub line is type-checked outside the test suite too.
- `tests/test_curso_models.py` loads all seven course `.mod` files under
  `curso/notebooks/modelos`, solves the six determinate ones with
  `stoch_simul(order=1, irf=40)` (the horizon four of the files use;
  `new_keynesian_3eq.mod` ships `irf=20` and `nk_zlb_ref.mod` carries no
  `stoch_simul` line), treats `nk_zlb_cons.mod` as the documented
  `BlanchardKahnError` / `strict=False` regime, checks every `initval` block
  against the solved steady state and pins the calibration targets the files
  document. Previously no test loaded any course model.
- `release_check.gate_examples_gallery` gained a `known_failures_path` keyword
  (default `None`, previous behaviour) and `load_known_example_failures`;
  `tests/test_release_check.py` gained eleven regression tests, including one
  that runs Gate 5 on the committed gallery with the repo allowlist.
- `tests/test_course_text_consistency.py` parses the course notebooks and
  `playground/00_start_here.py` statically: every lab id the syllabus names has a
  notebook, every shipped lab is in a module row, the stored table output matches
  the source, the start page names only notebooks the build copies, and every
  `git+` install hint in the course pins the README's commit.
- Regenerated `curso/site/diapositivas/index.html` and the trade lab's cache-busting
  hash from the committed sources: the slide index still listed the page counts
  and sizes of the PDFs replaced on 18 October, so every course-site build (and
  every `tests/test_e2e_upgrade.py` run) rewrote both tracked files.
- Six pre-existing HJB tests that exhaust `max_iter` on purpose now expect the
  new warning or use a converging iteration budget; `_caller_stacklevel` moved
  to the dependency-free `puremacro/_warnings.py` so the trade and HJB solvers
  can use it without importing the continuous-time VFI module.

### Known limitations

- Nonlinear general-equilibrium transitions (`nonlinear=True`) are implemented
  for the built-in one-asset household only; two-asset models, built-in or
  through the DP bridge, raise `NotImplementedError` and keep the linear
  transition (`nonlinear=False`).

## 5.4.0 (2026-10-20)

**Validated two-asset DP bridges, analytic DSGE scores, and release hardening.** This release includes the 5.3.0 additions and fixes the CI failures that blocked publication of that tag.

### Added

- Two-asset `dp.Model.solve(method="egm")` for the supported infinite-horizon liquid/illiquid household with CRRA utility and convex adjustment costs, with stationary distributions, model-local evaluation, and declared aggregate Jacobians and nonlinear household transitions. Unsupported rewards, budgets, constraints, and model configurations raise explicit errors instead of silently solving the built-in household.
- Analytic Kalman score and state-space sensitivity checks against numerical derivatives, including zero-calibrated persistence, fixed or stationary initialization, and zero shock standard deviations.
- Experimental `solve_pruned_occbin`: verified linear OccBin reduction with anticipated shocks, and slack second-order paths with a single initial innovation. Nonlinear binding regimes and nonlinear anticipated paths remain unsupported and raise `NotImplementedError`.
- Bilingual graduate exercise catalogs for showcase notebooks 66–68, a portable course publication script, and JOSS architecture and overview figures.

### Fixed

- Removed a circular `__wrapped__` reference on `LinearModel.stoch_simul` that broke signature inspection, and preserved the parser's existing options dictionary while retaining requested variables in native command metadata.
- Preserved fractional band-pass periods in `stoch_simul` and Baxter–King filter metadata, including narrow bands that integer rounding previously collapsed.
- Preserved legacy list-returning HJB result mapping methods and excluded private class metadata from dictionary access.
- Kept NUTS finite-difference probes within prior bounds; restored state normalization for zero-lag calibrations and numerical derivative fallback for nonanalytic model equations.
- Tempered particle filtering retains pure risk drift and cross terms in higher-order transitions and initializes all pruned state components consistently.
- Two-asset evaluation distinguishes current assets from next-period choices and respects declared aggregates and discount factors. Narrow asset-grid lotteries preserve wealth. Convergence requires a stationary distribution, not just normalized mass. Unsupported panel simulation and nonlinear GE bridges now fail explicitly.
- **Bridge timing correction:** explicit `A="a"` aggregates now mean current assets, consistently with `2*a` and `dp.Model`. Use `A="a(+1)"` to retain the 5.3.0 bridge's end-of-period saving convention; undeclared default `A`/`B` aliases still mean saving.
- Trade regularization rejects unknown keyword arguments; tariff optimization rejects ambiguous multi-country sequences instead of taxing only their first member. Structural local projections reject complex shock sizes instead of discarding imaginary parts.
- Explicitly empty course LLM fallback-cache lists no longer re-enable environment-configured fallbacks.
- Course figure presets and notebook inventory checks cover the new notebooks; release integration tests use the active interpreter and exclude downstream private-research checks from the public package.

### Internal

- Restored package-wide static type checking and added a CI mypy step, with numerical regression coverage for behavioral changes discovered during review.
- Updated the public API snapshot, documentation, manuscript assets, and synchronized release metadata. The release gate now fails closed when pytest exits unsuccessfully without parseable failure identifiers.

### Known limitations

- The existing RR2010 example intentionally exits nonzero: independent software parity, GDP response, and trough horizon pass, but its published t-statistic comparison remains outside the unchanged 0.005 tolerance by about 0.000003925. Optional examples Gate 5 therefore remains failed; this empirical discrepancy is preserved and documented in `docs/research_benchmarks.md`.
- The two-asset EGM adapter is specialized, not a general two-state optimizer. Panel simulation, discrete-choice SSJs, and nonlinear general-equilibrium transitions through the DP bridge are not implemented.
- Nonlinear constrained second-order OccBin requires regime-specific curvature and consistency checks; the experimental entry point rejects those cases rather than reporting a spurious converged path.

## 5.3.0 (2026-10-17)

This release extends `puremacro.dp`, `puremacro.dsge`, and the heterogeneous-agent sequence-space bridge with continuous-time and Krusell–Smith simulation, user-registered functions, declarative `dp.Model` Fake-News Jacobians, dated exogenous shocks and native policy/decomposition commands in `.mod` files, Real Schur Sylvester solvers up to 3rd order, a Tempered Particle Filter, and fast score gradients for NUTS.

### Added

- `puremacro.dp`:
  - `DPSolution.simulate()` now supports continuous-time HJB solutions (`method="hjb"`, using exact matrix-exponential transitions $\exp(Q\,\Delta t)$ for `"poisson"` jump processes and Euler–Maruyama steps with boundary reflection for `"diffusion"` processes) and Krusell–Smith aggregate-uncertainty solutions (`"vfi+ks"`, `"egm+ks"`, evolving aggregate state paths $(Z_t, K_t)$ under the converged log-linear forecast rule and interpolating the 4D policy `pol[i_Z, i_e, i_a, i_K]`).
  - `DPSolution.elapsed`: wall-clock solve time in seconds (`float`) recorded across all solver backends (`vfi`, `egm`, `hjb`, `vfi+ks`, `egm+ks`, and finite-horizon life-cycle backward induction).
  - `DPSolution.to_microframe(columns=None, *, age=None)`: converts the stationary or life-cycle grid-node distribution (`sol.dist > 0`) directly into a `puremacro.micro.MicroFrame` with `SurveyDesign.weights_only("weight")` (`unit="node"`), attaching states, policies, value function, and model `local` expressions (`local_grids`).
  - `Model.functions(**callables)`: registers user-defined vectorised Python/NumPy callables (`f(x, ..., xp=np)`) usable inside `local`, `subject_to`, `aggregate`, `clear`, and `Panel` evaluation across VFI, EGM, and HJB (and in `reward` for `method="vfi"`), with automatic routing to the active array backend (`xp`).
- `puremacro.models.hank_sequence_space` & `puremacro.dsge.hank`:
  - `dp_sequence_jacobian(model_or_sol, inputs, outputs=None, T=300, dx=1e-4)` and `dp_nonlinear_transition(model_or_sol, input_paths, outputs=None)` compute Auclert, Bardóczy, Rognlie & Straub (2021) Fake-News sequence-space Jacobians `J[Y][X]` of shape `(T, T)` and exact non-linear MIT-shock transition paths directly from a declarative `puremacro.dp.Model` or solved `DPSolution`.
  - `HANKModel`, `load_hank_mod`, and `solve_hank_bridge` accept `household_model=dp_model` to wire any declarative 1-state `dp.Model` into general-equilibrium HANK DAGs and `.mod` bridges without hand-writing a custom household class.
- `puremacro.dsge`:
  - Dated exogenous shocks (`e(-k)` lags for $k \ge 1$ and `e(+k)` news leads for $k \ge 1$) are supported in both `#` model-local variables and dynamic equations via automatic auxiliary state and lead expansion (`AUX_EXO_LAG_*`, `AUX_EXO_LEAD_*`) in `puremacro.dsge.parser`.
  - Analytical $O(1)$ compilation and evaluation of `steady_state_model; ... end;` blocks (supporting sequential local helper assignments even when a helper shares a name with a `#` model-local expression), with automatic validation against dynamic residuals and fallback to nonlinear root-finding / homotopy continuation (`solve_algo`, `homotopy`) when residuals exceed `tol`.
  - Native execution of Dynare `.mod` command blocks (`planner_objective;`, `ramsey_policy;`, `optim_weights;`, `osr_params;`, `osr_params_bounds;`, `osr;`, `histval;`, `endval;`, `calib_smoother;`, `shock_decomposition;`) via `DSGEModel.execute_mod_commands(data=...)` returning `DynareCommandResults` (`.ramsey`, `.osr`, `.smoother`, `.decomposition`, `.perfect_foresight`, `.commands_executed`).
  - Recursive Real Schur (Bartels–Stewart) generalized Sylvester solvers for 2nd- and 3rd-order perturbation tensors (`solve_generalized_sylvester_schur` and `solve_order3_sylvester_schur` in `puremacro.dsge._sylvester`), solving $(I_{n_x^k} \otimes \hat{A}) X + (h_x^\top)^{\otimes k} \otimes (A_+ F_{x+}) X = -R$ in pure real arithmetic via $1 \times 1$ and $2 \times 2$ quasi-upper-triangular blocks with bounded LU caching.
  - Tempered Particle Filter (`tempered_particle_filter` and `particle_filter(..., method="tempered")` in `puremacro.dsge.particle_filter`, following Herbst & Schorfheide 2019) for 1st-, 2nd- (`PrunedDSGESolution`), and 3rd-order (`Order3PrunedSolution`) DSGE state-space models, with adaptive $\phi$-bisection bridging to match `target_ineff`, systematic/stratified/residual/multinomial resampling, and vectorised RWMH mutation steps (`ParticleFilterResult.n_tempering_stages`).
  - Fast score gradients in `nuts_sample` and `estimate_dsge(..., method="nuts")` (`puremacro.dsge.nuts`, `puremacro.dsge.estimate`): `grad_log_post_fn` for custom/analytic score gradients, plus `grad_method` (`"forward"`, `"central"`, `"complex"`), `grad_eps`, and `vectorized_grad=True` reusing cached baseline evaluations ($d$ extra evaluations per leapfrog step under `"forward"` vs $2d$ under `"central"`) with automatic interior step reflection near prior bounds.
- `puremacro.llm`:
  - Two-tier read-only + writable completion caching via `open_cache([student_cache, course_cache])` and `CompletionCache(path, fallback_paths=...)` (or `$PUREMACRO_LLM_FALLBACK_CACHE`): lookups consult the primary writable cache first and then read-only fallback tiers without creating or modifying fallback files, while new completions are stored only in the primary cache.

### Fixed

- `puremacro.dp`: Fixed life-cycle (`T=N`) policy and value indexing when `DPSolution.local(...)` or `DPSolution.simulate()` evaluates expressions referencing endogenous choices (`sol.policy[name][t]` and `sol.V[t]` instead of indexing along the exogenous shock axis), and routed `Panel` derived-column evaluation through the model's registered `functions`.
- `puremacro.dsge`: Fixed `steady_state_model;` local helper assignments that shadow `#` model-local expression names so analytical steady-state evaluation preserves local assignments before evaluating dynamic equations.

## 5.2.0 (2026-10-16)

`puremacro.trade` is back. 5.0 and 5.1 shipped without it; this release brings the multi-country, multi-sector trade library back with consistent producer/purchaser accounting as its only accounting, a new bundled OECD table built by a script that ships with it, and no built-in defaults for the table: every dimension, country and sector comes from the table or from an argument. Code written for the 4.x trade API needs the changes listed under "Removed relative to 4.8.1" and "Changed relative to 4.8.1".

### Added

- `puremacro.trade` with the static model (`calibrate_trade_model`, `solve_trade_equilibrium`, `solve_equilibrium`, `TariffScenario`, `run_scenario`, `run_scenario_batch`, Geary-Khamis PPP), Hicksian welfare (`compute_hicksian_welfare`), audited policy recovery (`solve_policy_equilibrium`), optimal tariffs and Nash tariff games (`optimal_tariffs`, `game`), Caliendo-Parro (2015) exact hat algebra (`CaliendoParroModel`), the flexible CGE (`solve_flexible_trade_equilibrium`: nested CES technology, Stone-Geary demand, Armington sourcing, Atkeson-Burstein markups, capacity margins), policy analytics, the exact nested-CES block Newton (`ces_newton`), household demand systems and distributional incidence (`household`, `distributional`), audited parameter continuation (`continuation`), the reduced local-stability diagnostic (`stability`, experimental), native MRIO tables (`mrio`), the condensed one-factor Leontief tariff model (`condensed`), the perfect-foresight dynamic MRIO (`dynamic`, experimental), plots and the optional torch/MLX solvers (`gpu`). `models.TradePolicySimulator` returns with it.
- A bundled table, `puremacro/trade/_datafiles/icio_oecd2023_2019_isic_g20.npz`: the OECD Inter-Country Input-Output tables, 2023 edition, reference year 2019, summed into the 19 G20 member countries plus a rest of the world and the 20 sections of ISIC Rev.4 (shape 403 x 460). `load_icio_data()` reads it by default (`source="oecd_isic"`) and labels it with the table's own codes. `puremacro/trade/_datafiles/SOURCES.md` gives the OECD citation, the terms of use, the concordance and the checksums of the source and of the bundled file. If you publish results computed from it, cite the OECD.
- `tools/build_icio_isic.py` rebuilds that table byte for byte from the OECD file; `--download` fetches the OECD's `2016-2020_SML.zip` (about 85 MB) once into `~/.cache/puremacro/oecd_icio` (or `$PUREMACRO_OECD_ICIO_DIR`). `tools/reference_validation/validate_oecd.py` takes the same option.
- `puremacro.trade.data.synthetic_icio_table(nc, ns, *, nfd=3, seed=0)` builds a seeded synthetic table of any size in the library's layout, with nonnegative final-use shares and positive investment for every seed, and `synthetic_codes(nc, ns)` gives its labels (`C01...`, `S01...`). The tests run on it instead of a real table.
- Docs (EN/ES): `trade_accounting`, `trade_welfare`, `trade_policy`, `trade_continuation`, `trade_stability`, `trade_household`, `trade_distributional`, `trade_ces_newton`, `trade_condensed`, `trade_mrio`, `trade_dynamic`, `trade_gpu`; the trade sections of `spatial_and_trade_ge`, `policy_simulators`, `STRUCTURAL_VALIDATION_STATUS` and `ADVISORY`. Notebooks 55 and 61-65 (EN/ES), the example `puremacro/examples/distributional_trade_ge.py`, the OECD/accounting/welfare/Hicksian validators in `tools/reference_validation/` and three trade cases in the validation gallery.

### Removed relative to 4.8.1

- The legacy accounting. `accounting="legacy"` (the 4.x default), `tariff_revenue_mode="legacy_national"` and the precedence and compatibility switches that went with it are gone from every entry point, `gpu` included; consistent accounting is the only one, so the same call can give different numbers than in 4.x. `accounting="legacy"` raises `ValueError` saying so; a removed keyword raises `TypeError` naming it.
- Named scenarios and the scenario registry (`SCENARIOS`, the canonical-scenario constants and lookup functions, `ExtendedTariffScenario`, `run_tariff_scenario`). `run_scenario`, `solve_keller_pac` and `policy_analytics` take a `TariffScenario` or explicit tariff arrays; a string raises `TypeError("named scenarios were removed; build a TariffScenario")`.
- The 4.x bundled table, the bundled reference solutions and results workbook, their loaders and constants (the reference-solution and workbook loaders, `available_reference_scenarios`, `bundled_icio_path`, `bundled_reference_path`, `bundled_workbook_path`, `load_raw_45sector_icio`, `BUNDLED_ICIO_PROVENANCE`, `OECD2020_ICIO_PROVENANCE`), and that table's code lists (`CANONICAL_COUNTRY_CODES`, `CANONICAL_SECTOR_CODES`, `CANONICAL_SECTOR_NAMES`, `get_country_codes`, `get_sector_codes`, `get_sector_names`). `load_icio_data(source="legacy")` raises `ValueError` naming the bundled source and `synthetic_icio_table`.
- The real-world tariff benchmark (and its re-export in `game`), the analysis stubs and result classes built on the named scenarios (among them `EVDecompositionResult`), the two `ScenarioBatchResult` table renderers for the named scenarios, `plot.plot_country_impacts` and `plot.plot_tariff_escalation_curve`, the `"harberger"` welfare metric (it now gives the "Unknown welfare metric" `ValueError`), and `flexible.compute_homothetic_final_demand`.
- In `mrio` and `condensed`: the coarse tariff rules (`coarse_tariff_rates`, `CoarseTariffResult`), the sector concordance of the 4.x table and its code and name constants, the numbered model labels, `condensed.measures.aggregation_gaps` and the `w_<code>` alias.
- `puremacro.trade.extensions` (scenario extensions built on the named scenarios) and `puremacro.trade.tables` (LaTeX tables of the named scenarios) do not return.

### Changed relative to 4.8.1

- No paper defaults. Dimensions are read from the table (any number of countries, sectors and final-use categories); the fiscal-recycling and capacity branches of `solve_trade_equilibrium` and `solve_flexible_trade_equilibrium` need their country and sector named and raise `TypeError` naming the missing argument; `compute_unilateral_optimal_tariff` and `compute_welfare_payoff_matrix` take their countries as required arguments, and `solve_multilateral_nash_tariffs` and `evaluate_national_welfare` raise `TypeError` without them; the Geary-Khamis numeraire is a country argument, and `GearyKhamisResult`'s summary has no fixed-country PPP row. `ICIOData` needs its country and sector codes.
- Renamed without aliases: the fold helpers are `solve_fold_continuation` and `solve_fold_continuation_step` (they take the stiff coordinate as an argument; the info key is `idx_stiff`), and `CondensedMeasuresResult.headline()` is `key_measures()`.
- `solve_flexible_trade_equilibrium` is rebuilt on the consistent accounting; at default settings its equations are those of `solve_trade_equilibrium`. `method="quasi_condensed"` on `solve_trade_equilibrium` raises `NotImplementedError` pointing to the flexible solver.
- `gpu`: `BatchedJacobianEvaluator(calib, tau_a, taufd_a, ...)` evaluates the consistent-accounting residuals and no longer takes the national tariff-rate vectors; `solve_homotopy_continuation` has no national-rate targets.
- `get_country_code` raises `IndexError` for a position outside the table instead of inventing a code.
- `solve_fold_continuation_step` raises `ValueError` for an `idx_stiff` outside `[0, nc)`; it used to warn and use country 0.
- The condensed path of `solve_flexible_trade_equilibrium` keeps `iterations` within `max_iter`: Newton stops when its line search finds no decrease (`metadata["newton_stalled"]`). Before, a solve that could not converge ran Newton up to 21 times with the full `max_iter` each and could report more iterations than `max_iter`.
- In that condensed path, the tariff homotopy runs only on the iteration budget the direct attempt leaves unused; with fewer iterations left than homotopy steps it is not tried.
- When that condensed solve fails, `iterations` counts the homotopy stages as well as the direct attempt (`metadata["tariff_homotopy_steps"]` is 0 when a homotopy was tried and did not converge).
- `TradePolicySimulator.from_icio(year, *, tables=None, ...)` calibrates to the table of `year` (from `tables`, or the bundled 2019 table) and raises `ValueError` listing the available years for any other year.

### Fixed

- `CaliendoParroModel` mapped the codes USA, CHN and DEU (and ROW) to positions 0-2 whatever the model's own country order, and `manuf`/`serv` to sectors 0-1. Countries and sectors are now found by their own codes (exact, then case-insensitive) or by position; an unknown code raises `KeyError`.
- `TradePolicySimulator.from_icio(year)` ignored `year` and returned the `nafta_china` preset.
- `condensed.table.infer_merchandise_mask` counted utilities (ISIC D and E) as merchandise, and accepted composite sector labels whose content is not documented; those now need an explicit `merchandise_mask`.

## 5.1.0 (2026-10-15)

First 5.x final release. `puremacro.dp` simulates panels of individual histories and reports exact inequality statistics, and it is now the stable way to declare dynamic-programming models; `vfi.Model` is deprecated and will be removed in 6.0. It also carries everything in the 5.0 pre-releases below: the research material is no longer part of the package (4.8.1 is the last release that contains it), `puremacro.llm`, and the review-defect fixes.

### Added

- `DPSolution.simulate(n_agents, periods, seed=..., init=..., burn_in=..., record=...)` simulates individual histories from a `puremacro.dp` solution (discrete VFI and EGM, infinite horizon and life cycle, with or without a discrete choice and taste shocks) and returns a `dp.Panel`. One seeded PCG64 generator draws, each period, the discrete-choice uniforms, then the shock uniforms, then the survival uniforms, so the same seed gives a bit-identical panel. Infinite-horizon panels start in the stationary distribution by default; life-cycle panels start with newborns and apply the declared survival (`panel.alive`). `n_agents * periods` above 5e8 raises `ValueError` with the memory it would need. Continuous-time (HJB) and Krusell-Smith solutions raise `NotImplementedError`.
- `dp.Panel`, a container for simulated histories: `to_frame`, `to_microframe`, `cross_section`, `path`, and `mean`, `quantile`, `gini`, `top_share` and `summary` pooled over periods or grouped `by="t"` or `by="age"`. Dead agents keep their rows and are left out of every statistic.
- Exact statistics on the solved distribution: `DPSolution.gini`, `top_share`, `quantile` and `lorenz`. The Gini is the standard weighted Gini on the Lorenz curve, `1 - sum_i w_i (L_i + L_{i-1})`, the same function as `MicroFrame.gini` (negative values allowed, NaN when the total is not positive). In a life cycle, `age=j` uses one cohort and `age="all"` returns one value per age. Krusell-Smith solutions raise `ValueError` (they have no stationary distribution).
- `DPSolution.population_shares()` gives each age's population share in a life cycle with survival applied. `sol.distribution` still gives every cohort mass 1; the pooled statistics weight cohorts by these shares.
- `docs/dp.md` (EN/ES) has two new sections, "Panels and inequality" and "Porting from vfi.Model". `dp` moves from Experimental to Stable in `ARCHITECTURE.md`.

### Fixed

- `vfi.simulate_panel` raises `TypeError` for a policy that is not an integer array of grid indices. It used to cast a float policy (for example a continuous EGM policy in asset levels) with `astype(np.intp)`, which silently truncated it to wrong indices. Use `DPSolution.simulate` for such solutions.

### Deprecated

- `vfi.Model` issues a `DeprecationWarning` when constructed and will be removed in 6.0. Declare the model with `puremacro.dp`; `docs/dp.md`, "Porting from vfi.Model", shows the translation, and the `dp` version of notebook 21's model picks the same policy on every grid node.

### Changed

- The deprecated `puremacro.lp.garch_utils` module and `puremacro.sigma.sigma_numpy.SigmaObject` announced removal in 5.0.0; they now name 6.0.0, together with `vfi.Model`, so a minor release removes nothing.
- Notebook 21 (EN/ES) declares its precautionary-savings model with `puremacro.dp` instead of `vfi.Model`. It now prints mean assets 6.5733 and a wealth Gini of 0.3494, the exact values of the stationary distribution. It used to print 6.6767 and 0.3924, which came from a 2000-household, 100-period simulation; that Gini was computed after subtracting the minimum asset holding, so it was not the standard Gini. The notebook adds a simulated panel whose cross-sectional Gini converges to the exact value, and its text no longer claims vectorised Bellman updates or pluggable Numba/MLX/CuPy backends.

## 5.0.0a2 (2026-10-14)

Second 5.0 pre-release: the last defects from the September notebook review, the Stock-Yogo tables in full, and a test suite whose timing and network checks no longer depend on the CI runner.

### Added

- `inference.stock_yogo_cv` covers 3 to 30 instruments for the TSLS relative-bias test (it covered 3 to 8 and returned None for 9 to 30). A new keyword-only `criterion="size"` returns the TSLS Wald-test size critical values for 1 to 30 instruments at maximal size 10/15/20/25%. Both tables are for one endogenous regressor, from Stock and Yogo (2002, NBER Technical Working Paper 284, Tables 1 and 2). The bias criterion still returns None for one or two instruments.
- `gali_difference_spec()` in `puremacro.examples.empirical_benchmark_replications` returns the VAR data and the `bq` result for Galí's (1999) difference specification.

### Fixed

- The empirical-benchmark example and showcase notebook 35 (EN/ES) fed hours per capita in levels to `bq`, whose default cumulates every response, so the plotted hours response was a running sum (+12.2 at h=20). Both now use Galí's (1999) difference specification (productivity growth and the first difference of hours, cumulated to levels). On the bundled 1948-2026 data, hours fall 0.83% on impact and settle near -0.40%; productivity rises 0.68% on impact and settles near +0.81%. The notebook prose follows these numbers.
- Notebook 35 (EN/ES), Mertens-Ravn section: the text now quotes the notebook's own printed responses. GDP is -0.74% on impact with a trough of -1.13% at h=3, where the text claimed -1.5% to -2.5% and a multiplier the notebook does not compute. The federal funds rate rises 0.15 pp on impact, where the text said the Fed lowers it. A note says the first-stage F of 6.79 flags a weak instrument. The `load_gali1999` docstring gives the bundled sample (1948Q2-2026Q1, 312 quarters).
- `cycles.hamilton_filter` raises `ValueError` when the sample is too short to identify Hamilton's (2018) regression (T < h + 2p + 1: T <= 16 at h=8, p=4; T <= 48 at h=24, p=12). It used to return a cycle that was exactly zero, because the regression fitted exactly. It also raises `ValueError`, not `LinAlgError`, for NaN or inf input and for invalid `h`/`p`, checked after the `horizon`/`lags` aliases are resolved.
- `dynpanel.ab_gmm` and `dynpanel.bb_gmm` no longer raise `LinAlgError` when a GMM weight matrix is singular, for example with uncollapsed instruments on unbalanced panels with staggered start dates (step 1), or system GMM with year dummies on Stata's abdata (step 2). They use the Moore-Penrose generalized inverse, as xtabond2 does (Roodman 2009), record the rank in `GMMResult.notes` and issue a `RuntimeWarning`. Two-step results with a singular step-2 matrix depend on the generalized inverse and can differ from xtabond2's. Full-rank fits are bit-for-bit unchanged.
- Notebook 46 (EN/ES): the two-asset HANK section states the adjustment cost the solver implements, chi(d,a) = chi_0/(1+chi_1) |d|^(1+chi_1)/(a+a_bar)^chi_1, and its stored outputs come from the current solver: impact consumption is +0.0031 (the notebook still showed -0.0038 from before 4.4.0).
- The Mertens-Ravn proxy-SVAR example prints the published one-instrument 10% maximal-size cutoff 16.38 (it printed a hand-typed 16.4). The Stock-Yogo citations say NBER TWP 284 Tables 1 and 2, and the docstring no longer claims that the F > 10 rule corresponds to 3 instruments at 10% bias (that cell is 9.08).

### Changed

- `data.hamilton_filter` is a thin wrapper around `cycles.hamilton_filter`, the single implementation. Its signature and named `cycle`/`trend` return are unchanged.
- When instrument columns are linearly dependent, `GMMResult.n_instruments` and `hansen_j_df` count the linearly independent instruments (xtabond2's `e(j)`, df = j - k). A singular step-2 matrix does not reduce the degrees of freedom.

### Internal

- Every absolute wall-clock assertion in the tests goes through `budget()`/`budget_ms()` in `tests/_timing.py`, so `PUREMACRO_TIMING_SCALE` widens all of them on CI. Repeated measurements take the best of N after a warm-up instead of the worst. Ratio checks stay unscaled. `tests/test_test_quality.py` fails when a timing measurement is compared with an unscaled number. This removes the SW07 repeated-solve flake and the other unscaled ceilings.
- A new `benchmark` marker covers pure performance checks (speed-up ratios, sub-millisecond and widget-latency checks, SW07 and Sylvester timing). The default run deselects them; the opt-in workflow runs them weekly and on every release tag.
- The test suite runs offline and without the developer's API keys by default. An autouse guard refuses non-loopback connections and DNS lookups in any test not marked `network`, `replication` or `local_llm` and lists every refused attempt at the end of the run. Tests get a missing credentials file and no service key variables, which fixes the order-dependent `test_missing_key_raises_before_any_request`. Five tests that reached eur-lex, OECD or GitHub during a default run no longer do.
- The 16-thread SQLite cache stress test takes the write lock once per writer and sets a busy timeout on every connection, removing the intermittent "database is locked" failure on Windows runners.
- `tests/test_nb46_staleness.py` checks that notebook 46's stated cost formula and stored two-asset outputs match the library.

## 5.0.0a1 (2026-10-14)

This pre-release removes material that is not part of the library and prepares the 5.0 line. 4.8.1 remains on PyPI for anyone who needs what was removed.

### Removed

- The trade/MRIO modules (`puremacro.trade`, `puremacro.models.trade_policy`), their data, notebooks and docs are no longer part of puremacro. The validation gallery therefore has 110 cases in 15 subsystems (the four trade cases are gone), the `gpu` extra is gone with the accelerated trade solvers, and the wheel no longer bundles the OECD input-output table. Users who rely on the trade modules should stay on 4.8.1, the last release that contains them.
- The research studies and applications that had been added to the package since 4.4.0 are no longer part of puremacro: the SW07 estimator studies and their examples, the household incidence application and its dataset, and the cross-country forecasting application and its page. The DSGE fertility model, the identification spec-curve notebook and its panel, the state-dependent local-projection section of notebook 18 (Beveridge curve) that read that panel, and the internal project records (reviews, plans, specs, technical reports) were also removed. 4.8.1 is the last release that contains them. `puremacro.structural` (`MomentTargets`, `fit_structural`, the canonical LP designs), `validation.run_research_benchmarks` and the Romer–Romer (2010) replication stay.

**Declarative dynamic programming front end (`puremacro.dp`).** Declare states,
shocks, choices, a reward, constraints and an optional market-clearing condition
as equations in Dynare expression syntax (parsed by the `.mod` parser), and
`Model.solve()` compiles them to `VFIProblem`, `FiniteHorizonProblem` or
`stationary_equilibrium`. No return-function closures, grid broadcasting or
positional parameter lists. Aiyagari, Huggett, the life-cycle and two-asset
examples written as `dp` models reproduce `vfi.examples` exactly (same prices,
policies and value functions). `vfi.Model` stays supported: `dp` has no panel simulation or Python-callable utilities yet.

**Endogenous grid method for `dp` models.** `Model.solve(method="egm")` compiles
a one-asset model whose budget is `c + a(+1) = R*a + y(z)` to `vfi.solve_egm`.
The compiler checks that form (symbolically for the reward, numerically on the
grid for the budget and constraints) and raises `ModelSpecError` otherwise.
Marginal utility is the symbolic derivative of the reward, so felicities other
than CRRA work (closed-form inverse for `crra` and `log`, bisection otherwise).
Stationary equilibria warm-start each price evaluation. On the Aiyagari example
the equilibrium takes 0.8 s against 4.6 s for discrete VFI, and the EGM value
function is closer to a 1,500-point VFI solution. `vfi.solve_egm` takes
`u_prime`/`u_prime_inv` (any concave felicity) and a `c0` warm start; CRRA calls
are unchanged, and its inversion step is public as `vfi.egm.egm_step`.

Life-cycle models (`horizon(T)`) also solve with `method="egm"`: one EGM step per
age, with age-varying returns, income and parameters, survival, and an optional
terminal value (its slope taken by finite differences). Two-period log-utility
models match their closed form to 1e-12. On the 40-age example the EGM mean
assets by age are within 0.02 of a 1,191-point VFI solution, against 0.22 for
VFI on the same 120-point grid. Value functions interpolate linearly between
nodes, which understates them where they are most curved (near the borrowing
limit with gamma = 2).

A discrete choice (`discrete("h", ...)`, deterministic, no taste shocks) also
compiles to EGM, in both life cycles and infinite horizons: the Euler inversion
for each option is followed by an upper envelope over all segments between
adjacent endogenous points, with the borrowing-limit and top-of-grid corners
admitted only where their Kuhn-Tucker condition holds. The reward may depend on
the option (`crra(c, gamma) - chi*h`) and so may income; the return on assets
may not. A dominated option reproduces plain EGM to 1e-8. On a 30-age
labour-supply model the EGM value function on 100 points is less than half as
far from a 793-point VFI solution as VFI on 100 points, and the two methods agree
on a fine grid (work share by age within 0.02). Aggregates over the discrete
choice converge slowly in the grid for either method because the
distribution's mass sits on grid nodes. `vfi.DCEGMProblem` is unchanged: its
horizon repeats one stationary step and returns only the first period, so it
cannot carry age-varying income or survival.

`Model.taste_shocks(scale)` adds type-I extreme value shocks to the discrete
choice (solved by `method="egm"` only): values are log-sums, choices logit, the
Euler right side averages marginal utility over options and the distribution
splits mass by choice probability, so `mean("h")` is the participation rate.
The last period matches the closed-form logit to 1e-12 and a two-period model
matches a 200,001-point brute-force maximisation to 1e-5.

**Continuous-time `dp` models (`method="hjb"`).** Declare `control("c")`,
`drift("a", "r*a + w*e - c")`, `discount_rate("rho")` and `dp.Jump` shocks
(Poisson generators; several are combined as their product chain), and
`solve()` compiles to `vfi.solve_hjb_achdou`. The compiler checks that the
drift is linear in the control with slope -1 and derives u' and its inverse from
the reward. Prices and `clear()` work as in discrete time, with aggregates
integrated against the KFE density; `distribution` is the mass at each node.
The household matches `solve_hjb_achdou` to 1e-12 and the continuous-time
Aiyagari equilibrium matches `solve_aiyagari_continuous_hjb` (r* to 1e-9).
`solve_hjb_achdou` takes `income` (any (Na, Ne) drift before consumption) and
`utility`/`u_prime`/`u_prime_inv`; default calls are bit-identical.
`Model.solve(method=None)` picks `"hjb"` for a model with a drift and `"vfi"`
otherwise.

**Aggregate risk in `dp` models (Krusell-Smith).** `aggregate_shock("Z", ...)`
and `aggregate_state("K", grid, mean="a")` add the Krusell and Smith (1998)
forecast-rule fixed point, with prices and technology written as model
equations (any firm, any household reward). With the same grids and starting
distribution, the Cobb-Douglas model reproduces `vfi.krusell_smith` bit for bit
(forecast coefficients and capital path). `solve("egm")` solves the household
by EGM and simulates with continuous policies: on a 100-point asset grid it
converges in 36 outer iterations (24 s) to `b1` = 0.965 and 0.969 with R^2 above
0.9999. Discrete VFI policies sit on grid nodes, so on a 60-point grid
simulated capital never moves and the regression is degenerate; at its
defaults (200 points) `vfi.krusell_smith` had not converged after its 60 outer
iterations (470 s, `b1` = 0.897 and 0.954). Loop options go in
`solve(ks=dict(T=..., burn_in=..., seed=..., damping=..., tol=..., max_outer=..., mu0=...))`.
Both `vfi.krusell_smith` and dp's loop now raise a `RuntimeWarning` when they stop
at `max_outer` without converging (they used to return `converged=False`
silently), and when simulated capital never moves under aggregate risk, the
degenerate regression above, which would otherwise pass `tol` and look converged.

`vfi.solve_egm` and dp's EGM accept a return on assets that varies with the
exogenous state (`r` of shape (n_z,); Euler u'(c) = beta E[(1 + r') u'(c')]).
Scalar-`r` calls are unchanged.

### Added

- `puremacro.fetch.registry` lists every data source (series, panels, vintage providers, microdata and the narrative text connectors) with its auth level, credential, terms and whether results may be redistributed; `registry.load(id)` returns the fetcher. `puremacro.fetch.micro` adds `fetch_acs_pums` (Census ACS PUMS, 80 replicate weights, chunked past the API's 50-variable cap), `fetch_cps_basic` (weights only) and `fetch_scf` (Fed SCF summary extract, five implicates, 999 replicate weights) and `fetch_enigh` (INEGI ENIGH, Mexico, 2016+; strata and PSUs, so standard errors come from Taylor linearisation, `SurveyDesign(method="taylor")`). Each returns a `MicroFrame` whose `mean`/`total`/`quantile`/`share` give replicate standard errors, combined across implicates by Rubin's rules. The Census fetchers set the codebook's not-applicable codes (N/A, Not in Universe, Blank, Don't Know, Refused, CPS "Met No Conditions To Assign", including CPS earnings codes served with implied decimals as `-0.01`; e.g. `WAGP == -1` for persons under 15) to `NaN` by default, read from the endpoint's `variables.json`, so they no longer enter means and totals as values; `na_to_nan=False` keeps the raw codes and `MicroFrame.codebook` lists every labelled code.

- `MicroFrame.ratio`, `gini` and `top_share`: ratio of means (or of `q`-quantiles) of two columns, the Gini coefficient and the share held by the top `p`, with the same replicate/implicate standard errors as `mean`. Taylor designs (ENIGH) raise `NotImplementedError` for these rather than return an estimate without its standard error.

- **Worked example: an Aiyagari economy calibrated to the SCF** (`python -m puremacro.examples.scf_aiyagari`). It fetches one SCF wave, estimates wealth-to-income ratios and concentration moments with replicate standard errors, calibrates `beta` of a `dp` Aiyagari model (EGM) to one ratio and reports the untargeted moments next to the data, with the target's sampling error carried through `beta` by the delta method. On SCF 2025, with `beta` calibrated to median wealth over median income (2.63), the model gives a wealth Gini of 0.59 against 0.82, a top 1% share of 0.06 against 0.31 and a top 10% share of 0.38 against 0.72. The mean wealth-to-income ratio (8.52) is out of reach for `beta < 1` (at most 7.55, with r < 0).

- **Narrative corpus is now stored and exportable.** `harvest_narrative_corpus`
  used to keep nothing: its SQLite table was never written (and had no text
  column), and `use_cache`/`db_path` were ignored. A harvest given `db_path=`,
  `store=` or `use_cache=True` (default path: `narrative_corpus.db` beside the
  HTTP cache) now upserts into a `CorpusStore`; without them it stays in memory
  and writes nothing to disk. The store keeps full
  text (latest version, with a flag and the previous hash when a document
  was edited; see `store.revised()`) and a log of harvest runs; `store.load()`
  returns the accumulated corpus. `NarrativeCorpus` gains `to_frame(metadata=
  "promote"|"json"|"drop")`, `to_jsonl`/`from_jsonl` (no dependency),
  `to_parquet`/`from_parquet` (`io` extra), `from_frame`, `from_records` (any
  connector's output, including query-driven ones), and a provenance
  `manifest()` written next to every export, with `verify_manifest` to check a
  re-harvest against it.
- **New `puremacro.text`: tokenizer, segmentation and a sparse document-term
  matrix.** `build_dtm(texts, languages=...)` returns a `DocumentTermMatrix`
  whose `X` is a `scipy.sparse.csr_matrix` with a row-aligned `docs` frame;
  `.filter(min_df, max_df, max_features, terms)`, `.tfidf()` (scikit-learn's
  definition), `.aggregate(by="country", freq="MS")` (term-count panels),
  `.transform(new_texts)`, `.to_tidy()`, and `.save(dir)` writing `X.mtx` +
  `vocab.txt` + `docs.csv` for R (`Matrix::readMM`, quanteda, stm). The default
  `RegexTokenizer` keeps numbers and percentages ("25", "5.25%"; or masks them
  with `numbers="mask"`) and picks stop words from each document's language
  (en, es, pt, de, fr, it). Any `callable(text, language) -> list[str]` (spaCy,
  NLTK) can replace it. `NarrativeCorpus.dtm(...)` and
  `NarrativeCorpus.segment("paragraph"|"sentence"|callable)` build on it.
- `puremacro.llm`: one entry point for local language models (MLX, llama.cpp, Ollama and
  OpenAI-compatible servers) with a device-independent completion cache and memory-based model
  selection. MLX is selected on the iPad under Juno, where `mlx` and `mlx-lm` import; a
  completion there is not yet verified on a device.
  - `complete(prompt, *, model="default", engine="auto", mode=None, system=None,
    max_tokens=256, temperature=0.0, json_mode=False, timeout=120.0, base_url=None,
    cache=None)` and `chat(messages, **kw)`. `engine="auto"` takes the first available of MLX,
    llama.cpp, Ollama on port 11434 and OpenAI-compatible servers on ports 8080 (llamafile,
    llama-server) and 1234 (LM Studio); `base_url=` or `PUREMACRO_LLM_BASE_URL` names one
    server instead.
  - Modes `auto` (cache, then model, then store), `cache` (cache only; a miss raises `CacheMiss`,
    an unreadable cache `CacheUnavailable`; no engine, socket or model) and `off` (model only),
    from `mode=`, else `PUREMACRO_LLM_MODE`.
  - `CompletionCache` (SQLite, `~/.cache/puremacro/llm_cache.sqlite` or `PUREMACRO_LLM_CACHE`)
    keyed on the canonical model alias and the prompt before any chat template, so a row written
    by one engine answers the same call on any other; `export`, `merge` and `stats` to ship one.
    `export(since=...)` takes any ISO 8601 form (`Z`, offsets, fractional seconds) and compares
    in UTC; an answer that is not valid UTF-8 is returned but not stored; a cache path that cannot
    be read raises `CacheUnavailable` in cache mode rather than counting as a miss.
  - `complete()` refuses a `temperature` that is negative, not finite or not a number, and an
    `engine` name it does not know (`validate_engine_name` is public), before the cache is read.
  - A model ladder of three Qwen3 models (Apache-2.0): `qwen3-1.7b` (`small`), `qwen3-4b`
    (`default`, the reference model) and `qwen3-8b` (`large`). `pick_model()` sizes the model to
    the device's memory (`device_budget_gb()`), `ModelTooLarge` refuses a model that does not
    fit, and `PUREMACRO_LLM_MODEL` changes what `default` means. The four models of 4.8.1 keep
    their names.
  - The Qwen3 thinking is switched off on every engine and any `<think>` block left is removed
    (`strip_thinking`).
  - `report()` lists the host, memory, budget, engines, chosen model, cache and mode.
  - `pip install "puremacro[local-llm]"` now also installs `huggingface_hub`, through which
    llama.cpp fetches the registry's GGUF files, and installs `mlx-lm` only on Apple silicon
    (`platform_machine == 'arm64'`), so the extra no longer fails on an Intel Mac, where `mlx`
    has no wheel.
  - The `local_llm_uncertainty` notebooks and the `_tutor.py` copies still show the 4.8.1 model
    names (`qwen2.5-3b-instruct`, `qwen2.5:3b` and the tutor's `qwen2.5:3b-instruct`), which
    still resolve; they are synced in a separate change.

### Changed

- `narrative.topics.TfidfVectorizer` now builds its matrix sparse, so memory
  scales with non-zero cells rather than documents x terms. Its default
  tokenizer and its dense return value are unchanged (results match the
  previous version to 1e-14); pass `sparse_output=True` or `tokenizer=` to opt
  in. `NMF` accepts sparse input, and `DynamicTopicModel` uses it.
- `fetch.cdc_births_state` and `fetch.cdc_births_county` no longer default to a data directory outside the repository. `PUREMACRO_FERTILITY_ROOT` must name the directory that holds the natality extracts; when it is unset, `fetch_monthly`, `fetch_quarterly`, `fetch_yearly` and `fetch` raise `NatalityRootError`, which names the variable, and nothing is written. `build_subnational_panel` then builds without births, as it already did when the files were missing. The county loader's artifact paths are `nchs_artifact_path()` and `combined_artifact_path()` instead of module constants.
- The `fetch` HTTP helpers (`fetch._http.cached_get`, the FRED CSV reader, the CDC natality readers) send a `puremacro/<version>` User-Agent instead of the name of the monorepo the package came from.

- The score cache of `narrative.indices.llm_prob_kernel` now takes its path from
  `PUREMACRO_LLM_SCORES_CACHE` (the default, `~/.cache/puremacro/llm_scores.sqlite`,
  is unchanged). It used to read `PUREMACRO_LLM_CACHE`, which now names the
  completion cache of `puremacro.llm`. They are different caches with different
  tables, and one file for both would let each get in the other's way. If you had
  pointed `PUREMACRO_LLM_CACHE` at a score cache, set the new variable instead.

- The local narrative backends go through `puremacro.llm.complete`, so they read and fill the
  completion cache and follow the mode:
  - `LocalBackend`, `LocalProvider`, `get_default_backend` and `get_default_provider` default to
    `model="default"` (`qwen3-4b`; it was `qwen2.5-3b-instruct`), take a new `mode=` argument,
    and resolve no engine when they are built, so a cache-backed `score_llm` or
    `llm_prob_kernel` runs with no engine installed. `base_url` defaults to None:
    `PUREMACRO_LLM_BASE_URL`, then the default local servers.
  - They check `engine` and `mode` when they are built (`PUREMACRO_LLM_MODE` included): a
    misspelt name raises `ValueError` there, as the engine resolution of 4.8.1 did, and not at
    the first record.
  - `LocalBackend`'s `json_mode` now defaults to False: the prompts ask for a JSON array, which
    llama.cpp's object-only JSON mode cannot produce.
  - `LocalProvider.name` is `"local"` and its `model` the canonical alias, so the score cache of
    `llm_prob_kernel` is shared by every engine. Score-cache rows written by 4.8.1, keyed on the
    engine name, no longer hit.
  - `OllamaBackend` and `OllamaProvider` default to the registry alias `qwen3-4b`: the server
    still gets `qwen3:4b`, and the cache is shared with `LocalBackend("default")`. An explicit
    Ollama tag passes through unchanged and keys the cache on the tag.
  - `OllamaBackend` and `OllamaProvider` now default `base_url` to None, so
    `PUREMACRO_LLM_BASE_URL` applies to them as it does to `LocalBackend(engine='ollama')`;
    without the variable they still use the local Ollama server.
  - In `cache` mode `score_llm` raises `CacheMiss` / `CacheUnavailable` for a record the cache
    does not hold, instead of dropping it. `score_llm` also raises `ModelTooLarge` instead of
    dropping every record as malformed.
  - `get_default_backend` / `get_default_provider` look for engines with
    `puremacro.llm.engines.available_engines(base_url=…)` (in `cache` mode, nothing is probed).
    The print lines `[puremacro] local LLM backend: engine=…`,
    `[puremacro] local LLM provider: engine=…`, `[puremacro] using MockBackend (zero events).`
    and `[puremacro] using MockProvider.` are replaced by the install hint (when no engine is
    found) plus one status line,
    `[puremacro] local LLM: mode=<mode> engine=<name|cache|mock> model=<alias>`.
  - The example `puremacro.examples.narrative_local_llm` uses `"default"` and
    `ollama pull qwen3:4b`.
- `puremacro.narrative._local_engines` is now a compatibility re-export of `puremacro.llm`. Its
  `chat` reads the completion cache, and its `base_url` defaults to None
  (`PUREMACRO_LLM_BASE_URL`, then the default local servers) instead of Ollama's URL.
  `MODEL_ALIASES` is a read-only view of `puremacro.llm.MODELS` built from the registry, with
  exact GGUF file names instead of globs: adding an entry to it changes nothing, and a custom
  model id passes through to the engine unchanged. `HTTPEngine()` keeps its 4.8.1 defaults
  (Ollama API at http://localhost:11434).
- Local engines: MLX is chosen whenever `mlx` and `mlx_lm` can be imported, on any platform;
  llama.cpp offloads every layer to the GPU (`n_gpu_layers=-1`; it ran on the CPU) and loads
  registry files through `huggingface_hub`, which finds a downloaded file offline; `engine="auto"`
  also tries OpenAI-compatible servers on ports 8080 and 1234; and a server URL ending in `/v1`
  (LM Studio's documented form) is accepted.

### Fixed

- The MLX engine was never selected on iPadOS: it required `platform.system() == "Darwin"` and
  `platform.machine() == "arm64"`. Engines are now chosen by what can be imported.
- `score_llm`:
  - It aborted the whole corpus when an event's implementation date fell in the announcement
    quarter, or `horizon_quarters` was 0. Such events now get an empty implementation profile
    (implemented at the announcement), and any event `NarrativeEvent` refuses is dropped as
    malformed instead of raising.
  - Answers with `"sign": +1` (not valid JSON, but what the prompt's own schema line suggests)
    or wrapped in `{"events": [...]}` are now parsed, instead of yielding zero events.
- The llama.cpp ids of 4.8.1 were filename globs (`*Q4_K_M.gguf`), and the one for
  `qwen2.5-3b-instruct` matched no file on macOS and Linux, where matching is case-sensitive.
  Registry entries now name the exact file.

- `fetch.registry` listed the 67 narrative text connectors as `redistributable=True`, although their terms are marked `unaudited`. They are now `redistributable=False` until someone reviews them.

- `test_t2_f22_sw07_order2_repeated_solve_stability` compared its wall time with a fixed 0.20 s instead of going through `tests/_timing.budget`. CI's `PUREMACRO_TIMING_SCALE` therefore did not apply, and the test failed on Ubuntu with Python 3.11 in two PR runs on 2026-10-12 (0.48 s and 0.52 s; the other samples took 0.11 s).

- `fetch._http.cached_get` gains `params=` and `secret_params=`; an API key passed as a secret parameter is no longer written into the cache file name, the manifest, or HTTP error text.

- `harvest_narrative_corpus` fixes that changed what the narrative corpus contains:
  - In a fresh session `harvest_narrative_corpus` returned an empty corpus: it
    looked connectors up with `getattr(sources, module_name)`, which is `None`
    until the lazily loaded submodule has been imported, and skipped them
    silently. Connectors are now imported on demand.
  - Three registry entries named functions that do not exist (`us_federal_register`,
    `us_dod_contracts`, `uk_obr`) and harvested nothing; fixed.
  - Connectors that yield legacy `(date, text, url)` 3-tuples (the RSS/Atom
    helpers: `uk_hmt`, `uk_obr`, `de_bmf`, `fr_tresor`, `it_mef`, `eu_ecfin`,
    `ca_dof`, `us_treasury`, `us_dod_contracts`) were rejected by the schema
    check, so these sources were always empty. They are now accepted.
  - The per-record `doctype` that 33 connectors emit was ignored (harvest read
    `doc_type`); both are now read.
  - A failing connector was swallowed by a bare `except`. Failures are now in
    `corpus.harvest_report` and raise a warning (`on_error="warn"|"raise"|"ignore"`);
    the documents a connector yielded before failing are kept, as before.
  - Newly registered: `us_treasury`, `mas`, `macro_blogs`, `bluesky`; and, via
    `QUERY_SOURCE_REGISTRY` + `source_kwargs`, `gdelt`, `google_news`, `reddit`,
    `hackernews`, `local_csv`. Unknown source names now raise instead of being
    dropped silently.
  - Harvested `doc_id`s now also hash the record's explicit title (or `id`), and
    records sharing a key but differing in text all get text-qualified ids, so
    the ids do not depend on feed order (a reordered `us_warn` feed no longer
    shows up as revisions). A key with a single text keeps its plain id, which is
    what lets the store flag revisions; if a second text later appears under that
    key, both records get new ids and the old row stays in the store.

- `kalman_filter` / `kalman_smoother` with `diffuse_states=[...]` (exact-diffuse initialisation) returned wrong filtered and smoothed states, covariances and log-likelihood whenever a diffuse state's loading was not 1. Four defects, each now checked against an exact batch posterior and the closed-form diffuse log-likelihood (2,400 random 3-state / 2-observable models agree to ≤4e-7 relative error, previously wrong in most of them):
  - The diffuse-step update of the finite covariance scaled `K0 K0'` by `F_* / F_inf` instead of `F_*` (Koopman and Durbin 2000; Durbin and Koopman 2012, §5.2). The two agree only when `F_inf = 1`, as in the local-level `Z = 1` examples and tests. A local level with `Z = 2` gave a first-period variance of 0.0625 instead of `H / Z² = 0.25`.
  - Correlated measurement errors (non-diagonal `H`) were ignored during the diffuse phase, which processes a period's observations one at a time using only `H[k, k]`. Each period is now rotated onto the eigenvectors of `H` first.
  - A fully missing period during the diffuse phase did not carry the diffuse part through the transition (`P_inf ← T P_inf T'`). This changed the log-likelihood for a stationary diffuse state and the filtered states when `T` moves the diffuse direction.
  - The test for a diffuse direction was an absolute `F_inf > 1e-12`. With small loadings, a genuine diffuse direction fell below it; with large ones, the rounding residual of a resolved `P_inf` passed it and triggered spurious diffuse updates, sometimes with non-finite output. `P_inf` is now carried in factored form `A A'`, with one column per diffuse direction not yet pinned down by data. A diffuse update removes one column exactly, so no residual is left in a resolved direction. The cut-off compares `A'z` with its own rounding bound `|A|'|z|`, so it follows the loadings' units and keeps genuine diffuse directions however small their scale (nearly collinear loadings, or a diffuse component shrunk by `T`). Directions that `T` annihilates are dropped by a numerical-rank test on `T A`, with each state scaled to its rounding bound `|T| |A|`. A direction that `T` only shrinks therefore stays diffuse however far below the others it falls; before, `T = diag(1, 1e-16)`, or `diag(1, 1e-4)` over four missing periods, dropped it, which moved `a_filt` by up to 0.46 and `loglik` by 37. Once no column is left, the standard multivariate recursion (which also stores `F` and `K`) takes over. From there `kalman_smoother` uses its Durbin–Koopman recursion. Before, a positive rounding residual in `P_inf` could keep every period on the exact-diffuse path, and with it the smoother on its RTS step: this happened for 43 of 200 random 3-state models over 20 periods.

  The diffuse-period log-likelihood term `-0.5 (log 2π + log F_inf)` is unchanged, but later periods' `F_*` depend on the corrected `P_*`. **`loglik` therefore changes whenever `F_inf ≠ 1`**. Results are unchanged (to rounding) when every diffuse step has `F_inf = 1` (e.g. a local level with `Z = 1`), `H` is diagonal and no period in the diffuse phase is fully missing. No other library code passes `diffuse_states`. Tests in `tests/test_state_space_coverage.py` (`TestExactDiffuseNonUnitLoadings`, `TestExactDiffuseEdgeCases`).
- `kalman_smoother` with `diffuse_states=[...]` returned wrong `a_smooth` and `P_smooth` in every period that ended with part of the diffuse prior still unresolved. It ran the Rauch–Tung–Striebel recursion there on `P_pred` / `P_filt`, which hold only the finite part `P_*` of the covariance while the diffuse phase lasts (the diffuse part `P_inf` was never returned). Those periods now follow the exact initial smoother of Durbin and Koopman (2012, §5.3), the `r⁽⁰⁾, r⁽¹⁾, N⁽⁰⁾, N⁽¹⁾, N⁽²⁾` recursions, run backwards over the filter's observation-by-observation steps, including its rotation onto the eigenvectors of a non-diagonal `H`. They start from the Durbin–Koopman recursion's `r` and `N` at the end of the diffuse phase, with the `1/κ` terms at zero. The periods after the diffuse phase were already exact and are unchanged.
  - Checked against the batch GLS posterior of `α_1..α_T`, with zero prior precision on the diffuse indices of `α_1`. Before the fix:
    - A local linear trend with level and slope both diffuse was off by 0.18 in `a_smooth` and 0.17 in `P_smooth` at t = 0.
    - An AR(0.5) diffuse state loaded with `Z = 1.5` and `y[0]` missing was off by 0.18 and 2.6 at t = 0.
    - With `T` swapping two states and `y[0]` missing, it was off by 0.041 and 1.09 over t = 0, 1.
  - All three now agree to ≤2e-15, as does a 3-state model with correlated `H` and a value missing inside the diffuse phase (≤2e-14 in `P_smooth`).
  - Over 2,898 random models (1–4 states, 1–3 observables, random diffuse subsets, a quarter of the values missing, diagonal or correlated `H`), the median relative error is 5e-14 and the 99th percentile 5e-8.
  - Ten models exceed 1e-6 against a 60-digit evaluation of the same posterior (worst 3e-2 in `P_smooth`, at most 1e-5 in `a_smooth`). In nine of them some observation resolves a diffuse direction while barely loading on it: `F_inf` is at most 2e-4 of `(|z|' sd)²`, where `sd` holds the standard deviations of the `P_inf` entering the period, against a median of 1 over all the models (the tenth, at 2e-6, has 1.5e-2). The backward pass then cancels terms of order `1/F_inf²`.

  Results are unchanged when the diffuse prior is fully resolved within the first period, as in a local level with `Z = 1` (`python -m puremacro.examples.exact_diffuse_kalman`), and without `diffuse_states` (bit-identical). Tests in `tests/test_state_space_coverage.py` (`TestExactDiffuseSmoother`).
- The exact-diffuse filter took a period's observations in the order given, so a diffuse step that barely loads the direction it resolves could come before an observation that loads it strongly. That step leaves a large `P_*` for the next observation to cancel, and the smoother's expansion in `1/κ` then cancels terms of order `1/F_inf²`. Results depended on the order of the observables. In the test model a row loads the last diffuse state with weight 1e-5 and comes first; filtering with the rows swapped moved `P_filt` by 8e-8, and `a_smooth` was 6e-7 off the GLS posterior.
  - Within each diffuse period the filter now takes first, among the observations that still carry diffuse information, the one with the largest `F_inf / F_*`. That ratio is free of the units of `y`, of the states and of the scale of the diffuse factor. The order does not change the exact result.
  - Over the 2,898 random models of the previous entry, models above 1e-6 fall from ten to five and the worst error from 3e-2 to 2e-3. Each of the five left has a weak diffuse step in every order of its observables (smallest `F_inf / F_*` between 2e-8 and 1e-4, the same in every order): its diffuse loadings are nearly collinear within a period.
  - Reversing the observables moves the log-likelihood by at most 5e-8 relative (2e-7 before).
  - Filter output is bit-identical for 2,238 of 3,000 random models and within 1.4e-12 for 99%.
  - Tests: `test_invariant_to_the_order_of_the_observables`, `test_weakly_loaded_observation_before_a_strong_one`.

### Documentation

- New guides, each with a Spanish twin and with code blocks that CI runs: `docs/dp.md` (declarative dynamic programming: VFI, EGM and DC-EGM, continuous time, Krusell-Smith, and what `dp` does not cover yet), `docs/micro_data.md` (the source registry, the ACS, CPS, SCF and ENIGH providers, design-based standard errors and the SCF-calibrated Aiyagari example) and `docs/text_analysis.md` (corpus storage and export, the tokenizer and the sparse document-term matrix). README, README.es and the site navigation link them. `docs/CREDENTIALS.md` now names the microdata providers as the `census` key's consumers.
- The JOSS paper (`paper/paper.md`) describes `puremacro.dp`, the survey microdata and the text tools, uses the SCF example as its worked case, and adds dolo to the state of the field. It stays within the 1,750-word limit (1,730 by `tests/test_paper.py`'s count).
- New guide `docs/local_llm.md` (Spanish twin `docs/es/local_llm.md`): what runs where, installation per platform (iPad, Mac, Windows, Linux, Colab), the model ladder with licences, the modes, shipping a completion cache, the environment variables, the iPad notes and `report()`. Its model-free code blocks run in CI. `docs/tablet.md` gains a section "Local models on the iPad"; README, README.es and the site navigation link the guide, and the README's local-model section now describes `puremacro.llm`.

- Course lesson T02_F (Gertler-Karadi 2011, English and Spanish) reports percent deviations from `to_frame(units="pct")`. It multiplied level deviations by 100 and labelled them %, so it printed capital at -28.66% and bank net worth at -67.88% on impact where the true figures are -5.06% and -49.15% (`docs/ADVISORY.md`, 2026-09-30). The scorecard, the policy table, the three figures, the figures quoted in the text and the lesson's own checks are corrected; the spread in annualised basis points and Q, whose steady state is 1, were already right. The lesson now installs `puremacro>=4.4,<5`, the first versions with `units="pct"`. `tests/test_notebooks/test_curso_t02f_units.py` pins the printed values and, under `-m slow`, re-runs the lesson.
- `docs/monetary_transmission.md` (English and Spanish) documents `MonetaryTransmissionSimulator` and the HANK vs. RANK browser lab, which used to share a page with the removed trade simulator. The page prints the simulator's own values, attributes the 60–80% indirect share to Kaplan, Moll and Violante (2018) and describes the lab as plain JavaScript (d3 and KaTeX from a CDN); recipe 1's shock is a 25 bp quarterly (100 bp annualized) hike.
- README, README.es, ARCHITECTURE.md, `docs/index.md`, `docs/VALIDATION.md` (110 cases in 15 subsystems) and the JOSS paper describe the library without the removed material. `docs/STRUCTURAL_VALIDATION_STATUS.md` points its MATLAB caveat at `tools/reference_validation/README.md`.

### Internal

- `tools/research_guard.py` and `tests/test_no_research_content.py`: a guard that scans the tracked tree and built sdists or wheels for material that does not belong in the public repository (generic forbidden directory prefixes, the SHA-256 digests of moved paths and of a private name list, and machine-specific paths such as a user's home directory), with the tests that pin its behaviour. The guard names no project; `python tools/research_guard.py --git` runs it locally.
- `playground/dist/`, a stale JupyterLite build, is no longer tracked; GitHub Pages rebuilds it.
- `pyproject.toml` and `MANIFEST.in` drop the removed packages, their data files and the `gpu` extra (its torch and `mlx` pins served only the removed solvers; the MLX engine of `puremacro.llm` comes with the `local-llm` extra, and the `accel`/`backend` and `cuda` extras are unchanged).
- The notebook integrity test restores matplotlib's `rcParams` after exercising `_nbstyle.apply_style()`, so `tests/test_dsge/test_diagnostics.py::test_check_unit_circle_plot` passes in any test order.
- `.gitignore` no longer lists LaTeX intermediates of a directory that is not part of the repository.
- `tools/release_check.py` reads a PEP 440 pre-release heading (`## 5.0.0a1`) from this file, so the version-sync gate compares `5.0.0a1` across the five version-bearing files instead of reporting `5.0.0` against `5.0.0a1`; `tests/test_release_check.py` pins every reader on a pre-release value. The deprecation-target tests for the `puremacro.lp.garch_utils` and `SigmaObject` shims compare `packaging` versions, so their 5.0.0 removal stays in the future of 5.0.0a1.

## Releases before 5.0

The complete changelog of earlier releases is not part of the public repository. Development before 0.92.0 (2026-07-20), the first public release, was private.

- **4.8.1** (2026-10-11) — Cross-platform cartridge verification (NPZ creator-OS metadata no longer causes false checksum mismatches), shipping the cross-country panel builders, portable `DataFrame.attrs` provenance and v2 cartridges prepared for 4.7.0 and 4.8.0.
- **4.8.0** (2026-10-10; not on PyPI) — Portable panel provenance: `DataFrame.attrs` survive storage and cartridge verification. Legacy v1 cartridge verification failed on the Windows CI targets and the release workflow blocked the upload; 4.8.1 carries the correction.
- **4.7.0** (2026-10-07; not on PyPI) — Cross-country macro panels: ten one-call panel builders for the World Bank (WDI), the OECD (annual and quarterly national accounts, short-term indicators, labour force and vacancies), Eurostat, the IMF data portal, the BIS, ECB/FRED rates and yields, ILOSTAT, the Penn World Table and the Maddison Project, through urllib with an on-disk cache and one `(code, date)` contract; `splice_sources`; and repairs to the BIS, Eurostat, ILOSTAT and OECD-LFS readers.
- **4.6.0** (2026-10-04) — The validation gallery runs inside Pyodide, the Smets–Wouters Table 2 (training-sample) marginal-likelihood computation is implemented and checked against Dynare 8, and the opt-in slow, reference and replication tests gate every release.
- **4.5.0** (2026-10-03) — Honest convergence and corrected numbers: the wrong-number and false-success known issues of 4.4.0, each checked against an independent reference.
- **4.4.0** (2026-10-02) — Empirical-to-structural estimation (`puremacro.structural`: `MomentTargets`, `fit_structural`, canonical LP designs with joint HAC covariance), independent research benchmarks, a matrix-free stacked Newton-Krylov solver for perfect-foresight transitions (`dsge.stacked_newton`), and the verified fixes of the 30 September library review.
- **4.3.0** (2026-09-20) — Audited structural solvers: `docs/STRUCTURAL_VALIDATION_STATUS.md` states each surface's acceptance contract, evidence and limits, with five models checked against live Dynare 7.0 at orders two and three.
- **4.2.0** (2026-09-18) — Dual-theme grayscale card styling for every notebook (`_nbstyle.py`, synchronized between the showcase and course trees) and a polish pass over the notebook corpus.
- **4.1.1** (2026-09-17) — Restores the fallback exception handling that 4.1.0's explicit exception tuples had dropped across 142 files.
- **4.1.0** (2026-09-17; not on PyPI) — Latin America real-time nowcasting with the Bańbura–Modugno news decomposition, interactive Double ML (IRM and DML-IV), the HANK vs. RANK monetary transmission simulator with its browser lab, showcase notebooks 59 and 60, and three wrong numbers fixed (HP-filtered theoretical moments, LP-IV standard errors, stationary distributions of nearly decomposable chains).
- **4.0.1** (2026-09-16) — An attribution notice for the third-party data that 4.0.0 had started redistributing; no code change.
- **4.0.0** (2026-09-16) — MATLAB removed from the reproduction path: every documented result reproduces with Python and the installed package alone, and the `matlab/` companion toolbox is gone (a major release because public API was removed).
- **3.4.0** (2026-09-15; not on PyPI) — `.to_typst()` export parity across result objects, implicit continuous-time HJB and adjoint KFE solvers (`vfi.hjb_achdou`), hardened multi-constraint OccBin, Double/Debiased Machine Learning (`causal.dml`), and Latin American real-time connectors.
- **3.3.0** (2026-09-13) — Continuous dynamic programming (Chebyshev collocation, FEM Galerkin, shape-preserving splines, Smolyak grids, DC-EGM), Deep Macro PINNs, quantitative spatial general equilibrium (Allen–Arkolakis 2014), and `pyarrow`/`openpyxl` moved to the `io` extra so the base install resolves inside Pyodide.
- **3.2.1** (2026-09-11) — Documentation overhaul and the live MkDocs site on GitHub Pages; PyPI metadata aligned.
- **3.2.0** (2026-09-11) — Applied macroeconomist showcase suite (notebooks 47–50) and the global climate, energy, commodity and financial data ecosystem with modular panel builders.
- **3.1.0** (2026-09-10) — Optimal policy under discretion and commitment, DSGE-VAR, news shocks, nonlinear particle filtering and Markov-switching DSGE; the Dynare macro processor, live IRF parameter sliders and rank identification criteria.
- **3.0.0** (2026-09-09; not on PyPI) — Exact analytic likelihood gradients, pure-Python HMC/NUTS Bayesian estimation, and the heterogeneous-agent (HANK) sequence-space bridge.
- **2.9.0** (2026-09-09) — Dynare parity, Tier 3: the `stoch_simul` filtering and simulation surface, extended path (Fair–Taylor 1983), forecasting and shock decompositions, the parity dashboard and CLI.
- **2.8.0** (2026-09-08) — Dynare parity, Tier 2: order-3 perturbation and pruning, deterministic transitions and a semismooth Newton MCP solver, multi-constraint OccBin with a piecewise Kalman filter, SMC and particle filtering, nonlinear Ramsey policy and BGP detrending.
- **2.7.0** (2026-09-08) — Dynare parity, Tier 0: a native macro preprocessor, recursive-descent AST parser, symbolic differentiation with common-subexpression elimination and a fast Sylvester solver replace the regular-expression front end.
- **2.6.0** (2026-09-07) — Two tracks: a Dynare `.mod` file becomes estimable (posterior, diagnostics, robust steady state and policy regimes), and native regression and inference replace `statsmodels` for the calls an applied notebook makes.
- **2.5.0** (2026-09-06) — Spatial econometrics phases 2–4 (spatial models, panels and local projections, spillover-robust DiD), the wavelet package split, `simex_ols` and runtime helpers; `wavelet_coherence` now discards boundary-contaminated coefficients by default.
- **2.4.0** (2026-09-05) — `puremacro.spatial`, phase 1: spatial weights, Moran's I and Geary's C, Conley and space-time HAC, Conley errors for panel local projections, and shift-share IV with Adão–Kolesár–Morales errors.
- **2.3.1** (2026-09-05) — Correctness release after a full audit of 2.3.0: the DSGE engine solves lead/lag systems, decision-rule control rows are aligned, Blanchard–Kahn violations raise, and the Gertler–Karadi responses are the documented ones.
- **2.3.0** (2026-09-05; not on PyPI) — Narrative sign restrictions, honest DiD sensitivity bounds, smooth local projections, nonlinear sequence-space HANK transitions, the Gertler–Karadi (2011) banking-frictions DSGE and BVAR with stochastic volatility, each with a bilingual guide.
- **2.2.0** (2026-09-05) — The `puremacro-dynare` command-line tool, the sequence-space Fake News algorithm and targeted transfer simulations, and the frontier Dynare showcase notebooks (41, 42).
- **2.1.0** (2026-09-04) — FEVD and historical shock decomposition, OccBin occasionally binding constraints, deterministic perfect-foresight simulation and the Bayesian DSGE estimation pipeline.
- **2.0.1** (2026-09-04) — Packaging and PyPI release update for 2.0.
- **2.0.0** (2026-09-04) — A unified result architecture (`LPResult` and `.plot()`/`.to_latex()`/`.to_typst()`/`.to_markdown()` on every result object), standardized estimator signatures (`lags`, `horizon`, `ci`), the Dynare-compatible solver with second-order pruning, and full type completeness.
- **1.10.0** (2026-09-03) — Three estimator fixes affecting every earlier release: `dols` omitted the contemporaneous term, `fm_ols` dropped the lag-0 block of the long-run covariance, and `kleibergen_paap_f` was `n²` too large.
- **1.9.0** (2026-09-02) — `ana_by_activity` (annual value added and labour input by ISIC activity, with `chain_volume` for additive volumes), the ISIC breakdown of the quarterly labour block, and the seven-estimator fix that reached every release from 0.92.0 to 1.8.0.
- **1.8.0** (2026-08-23) — The longest quarterly national accounts panel the sources support (Spain to 1970, Japan to 1955), spliced onto the OECD spine with the evidence for every join reported.
- **1.7.0** (2026-08-23) — Real-time data: an accessor that returns the archive, six vintage providers behind one panel, the news-vs-noise test as a first-class estimator; cross-country revision work grows from 2 to 34 usable countries.
- **1.6.0** (2026-08-23) — The capital stock the production functions assumed, a panel that imports without a network stack, and two silent data defects fixed (one variable carrying two scales, five countries publishing the wrong kind of volume).
- **1.5.0** (2026-08-22) — Two hours series wrong by an order of magnitude, a labour fetch that no longer drags the national accounts behind it, and the retirement of a route that reached only a third of the block.
- **1.4.0** (2026-08-21) — The labour input the quarterly national accounts measure, which `qna_panel` could not reach.
- **1.3.2** (2026-08-20) — A DSGE solver that no longer depends on the BLAS, an Aiyagari routine that solves, a suite that passes on Windows, and the worked national-accounts example.
- **1.3.1** (2026-08-20) — The `requests`-free fetch path that 1.3.0 was tagged one commit too early to include, and a cartridge path that survives pandas 3.
- **1.3.0** (2026-08-20) — All three measurements of GDP in one panel, availability-driven country discovery, one price reference year across countries, and the expenditure, output and income identities scored inside their own flows.
- **1.2.0** (2026-08-19) — Capability detection, browser networking, portable data cartridges, resumable estimation and a DSGE model DSL, plus a corrected Klein policy function.
- **1.1.0** (2026-08-16) — Quarterly national accounts historical vintages, Mankiw–Shapiro news-vs-noise testing, and multilingual narrative harvesting across 50+ institutional sources.
- **1.0.0** (2026-08-15) — GARCH-MIDAS volatility, business-cycle bandpass filters and Beveridge–Nelson decomposition, weak-IV robust Anderson–Rubin sets for LP-IV, panel LP block bootstrap with sup-t bands, a 73-case validation gallery across 13 subsystems, and the Rosetta Stone syntax cheatsheet.
- **0.96.0** (2026-08-08; not on PyPI) — Six-package base install (`requests` and `pyarrow` promoted out of the extras), a local-install, Spanish-only course companion, sup-t simultaneous IRF bands, the Galí (1999) and Kilian (2009) replications, and the district-level Beige Book uncertainty index.
- **0.95.0** (2026-07-24) — Published to PyPI without a changelog entry of its own; the work since 0.92.0 is recorded under 0.96.0.
- **0.94.0** (2026-07-24) — Published to PyPI without a changelog entry of its own; see 0.96.0.
- **0.93.0** (2026-07-24) — Published to PyPI without a changelog entry of its own; see 0.96.0.
- **0.92.0** (2026-07-20) — First public release (the version had been cut privately on 2026-05-30): free local-LLM backends for narrative scoring (`score_llm` and `llm_prob_kernel` run at $0 locally through MLX, llama.cpp or an Ollama server), on top of the estimators, data layer and `puremacro.vfi` built during the private period.
