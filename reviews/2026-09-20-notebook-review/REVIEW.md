# Notebook review — 20 September 2026

The most useful improvement is to make every example connect its question to
the evidence it actually computes. Several notebooks executed successfully
while teaching an incorrect interpretation. This pass corrects six bilingual
examples and makes the catalog easier to navigate.

## Scope

- Inventoried 136 top-level showcase sources: 68 English/Spanish pairs. Their
  source cells matched their existing notebook artifacts at the start.
- Reviewed the numbered showcases across 01–35, 36–60 and 61–65, with detailed
  computation changes concentrated in 10, 34, 48, 50, 61 and 62.
- Validated the 220 artifacts discovered across the showcase and curso suites.
  This is structural/output validation, not a fresh execution of all 220.
- Preserved the pre-existing edit to `59_latam_realtime_nowcast_and_news_es.ipynb`.
  The course collections and specialized historical/climate collection did not
  receive a detailed pedagogical review in this pass.
- No package algorithms, dependencies or public APIs were changed.

## Corrections made

| Example | Finding | Updated behavior |
|---|---|---|
| [10: staggered DiD](../../notebooks/10_staggered_did.py) | Overall ATT uncertainty was calculated by averaging event-study interval endpoints. Prose also incorrectly said genuine pretreatment leads were unavailable. | Recompute the overall statistic in a whole-unit bootstrap; distinguish estimated leads, the normalized base period and pointwise bands. The default 90% interval is approximately [1.020, 1.252] around ATT 1.129 and misses the planted truth of 1.0. The notebook reports this honestly. |
| [34: penalized forecasting](../../notebooks/34_penalized_macro_forecasting.py) | An in-sample fit was presented as forecasting evidence; prose claimed exact variable selection and correlated predictors despite the actual simulation. | Reserve the final 40 of 160 months, fit and tune on the first 120, and evaluate successive one-step forecasts with fixed coefficients. Compare RMSE/MAE with a training-mean benchmark and report actual signal/noise selection. |
| [48: nowcasting](../../notebooks/48_applied_realtime_nowcasting_and_news.py) | Target-quarter GDP entered bridge estimation. A single-vintage projection table was described as an exact vintage-revision identity. The classical-noise regression slope was misstated as −1. | Hold out target GDP, report its forecast error, derive the implemented projection weights and explain their limited interpretation. Correct the noise-null algebra and discuss the small sample's misclassification of the known noise DGP. |
| [50: fiscal analysis](../../notebooks/50_applied_fiscal_multipliers_and_debt_sustainability.py) | The instrument was manufactured from the endogenous tax residual. Mean responses were called cumulative multipliers. Consolidation units were off by 100× and the exercise compared different initial conditions. | Use bundled observed narrative tax shocks and expose weak identification through effective F and Anderson–Rubin sets. Label average GDP responses accurately. Express balance adjustments in annual percentage points of GDP and compare matched debt paths. |
| [61: policy simulators](../../notebooks/61_quantitative_policy_simulators.py) | A hand-built trade preset was described as empirical OECD data; residual-based welfare components were treated as an exact economic decomposition. HANK/RANK interpretation and MPC values were stale. | State calibration provenance, distinguish model real income from real wages, verify their actual accounting identities, and interpret the computed monetary responses. |
| [62: flexible trade](../../notebooks/62_flexible_trade_cge.py) | Plots mislabeled nominal wages, output and imposed pass-through assumptions. The original wrapper bypassed flexible settings for the default legacy objective. | Inspect ICIO separately; teach cost, demand and markup functions at given prices; solve a small, balanced benchmark with explicit residual checks. The library now routes small active-flexible solves through the quasi-condensed residual and documents the large-ICIO fallback. |

All six English examples have corresponding Spanish updates. The revised
[English](../../docs/notebooks.md) and [Spanish](../../docs/es/notebooks.md)
guides organize learning paths by question and link every showcase's executed
artifact and editable source. Duplicated feature descriptions and incomplete
API snippets were replaced with instructions grounded in the runnable examples.
The [template](../../notebooks/_TEMPLATE.md) now covers provenance, information
sets, units, estimands, uncertainty and valid exercise comparisons.

## Numerical observations

- **34:** holdout RMSE is about 0.506 for Elastic Net, 0.495 for Adaptive Lasso
  and 2.168 for the training-mean benchmark. This is one planted-signal experiment,
  not evidence that either method wins on arbitrary macroeconomic datasets.
- **48:** held-out nowcast 0.7856%, realization 1.1886%, error −0.4030 percentage
  points. A single held-out quarter does not validate a forecasting model.
- **50:** impact effective F is about 1.347 against the reported 10%-bias
  critical value of 23.109. Anderson–Rubin sets include unbounded cases. The
  corrected instrument does not support the old precise multiplier ranking.
  The separate one-percentage-point consolidation experiment gives median
  terminal debt near 66.2% and an any-time 80% threshold breach rate near 9.0%.
- **62:** the small benchmark has a maximum residual around 3.52e-12 and the
  custom tariff experiment around 7.87e-12. These verify the benchmark equations,
  not integration of all flexible extensions into a general equilibrium model.

## Validation completed

- Rebuilt and executed all **12 changed `.ipynb` artifacts**, using
  `tools/build_notebooks.py` with explicit stems. See
  [applied build log](applied-build.log) and
  [foundations/trade build log](foundations-and-trade-build.log).
- `python tools/build_notebooks.py --validate-only`: **220 notebooks, zero
  errors**, after rebuilding. See [artifact validation](artifact-validation.log).
- Notebook integrity, build tooling, bilingual documentation and documentation
  navigation checks: **135 passed**. See [test log](static-tests.log).
- Documentation/navigation and existing challenger schema/output checks:
  **146 passed** in a separate targeted run; this overlaps the preceding suite.
  See [schema test log](docs-and-schema-tests.log).
- All **136 top-level sources match their artifact cells**. Code cells match
  exactly between English and Spanish in all six updated pairs. All **832 catalog
  link targets** exist. The existing edited notebook 59 artifact retains its
  original SHA-256. See [audit record](source-artifact-audit.json).
- Inspected rendered figures from all six updated English examples. Exercise
  checks covered DiD effects 0, 2 and −0.01; penalty mixes 0 and 1; factor counts
  1 and 3; debt starting at .85 with balance shifts 0, 2 and −1; and the optional
  ICIO inspection in 62. These variants passed their stated checks.
- `git diff --check` passes. No full-library test run or fresh execution of all
  220 notebooks was performed. MkDocs is absent from the environment, so the
  documentation checks used navigation tests and local link-target validation.

## Follow-up implementation: flexible-trade settings

After this review identified the bypass, `solve_flexible_trade_equilibrium` was
fixed. The default configuration keeps exact legacy parity. A small model with
an active flexible setting now selects the quasi-condensed path, whose macro
evaluation uses the configured nested-CES cost, intermediate sourcing,
Armington/Stone–Geary demand, capacity penalties and markup primitives. Its
goods block is solved with the implied flexible intermediate coefficient matrix,
and its price residual uses the full nested unit cost instead of adding the
legacy Leontief block a second time. Large country-sector systems keep the
legacy sparse-friendly route because the quasi-condensed Jacobian is dense.

The focused trade regression suite passes **231 tests**, including a new check
that CES and subsistence settings change a tariffed synthetic equilibrium. The
English and Spanish notebook 62 sources and artifacts remain code-cell parity
and validate cleanly. This implementation does not claim full flexible-parameter
integration for large ICIO solves; that is the next engineering boundary.

## Findings to prioritize next

1. **Flexible-trade integration at scale.** Small active-flexible solves now use
   the quasi-condensed residual and a regression test requires CES and
   subsistence settings to change a tariffed synthetic equilibrium. Full 77×11
   ICIO configurations still use the sparse-friendly legacy path because a dense
   quasi-condensed Jacobian is not a practical default at that scale. A future
   sparse flexible Jacobian would remove that boundary.
2. **Sun–Abraham aggregation uncertainty.**
   `puremacro/did/sun_abraham.py` combines cohort standard errors as
   `sqrt(sum(w**2 * se**2))`, omitting cross-cohort covariance. Shared controls can
   induce dependence. Audit with a joint unit bootstrap and a coverage experiment;
   notebook 10's corrected overall interval does not resolve this separate API
   concern. The notebook plots Callaway–Sant'Anna bands.
3. **Policy examples 47 and 49.** In 47, manually assembled text is described as
   official statements and arbitrary coefficient jitter as posterior draws.
   Its macro data are frozen observations, so provenance needs to distinguish
   each component. In 49, simulated bank returns use real ticker labels and a
   toy stress-loss formula is used to suggest a regulatory buffer. Rewrite those
   interpretations before presenting the examples as institutional applications.
4. **Notebook 59 prose.** The guide now identifies its synthetic data and separate
   PIT experiment. Its own prose still describes a styled fan chart as official
   and interprets independent simulated PIT draws as validation of the nowcasting
   engine. Coordinate a source/artifact update with the existing local edit.
5. **Earlier demonstrations.** Notebook 33 also fits target GDP and needs the
   information-set correction implemented in 48; it has been removed from the
   recommended forecasting sequence. Notebook 28 describes HAC/chi-square
   Anderson–Rubin inference as finite-sample exact despite relying on asymptotic
   inference. Notebook 32 describes the prescribed-tax, fixed-saving DICE
   simulator as a welfare optimizer and gives equations inconsistent with its
   implemented damage/net-output formulas. Notebook 35 needs an audit of hours
   response accumulation and of tax-multiplier identification and normalization.

## Brainstorming

- **An information-set lab:** use the same target to compare leaked full-sample
  fit, a chronological holdout, and a genuine release-date backtest. Show exactly
  which observations are available at each decision date, with a simple benchmark
  and a record of revisions. This would connect 34, 48 and 59.
- **An uncertainty lab:** repeat the DiD experiment across independent samples,
  comparing pointwise coverage, simultaneous coverage and coverage of the overall
  ATT. Include shared-control dependence and vary cluster counts. This turns the
  default interval miss into a question students can investigate.
- **Instrument strength versus validity:** vary relevance and exclusion
  separately in a known simulation. Compare OLS, LP-IV, conventional intervals
  and Anderson–Rubin coverage. Include an instrument derived from the regressor
  to show why a large first-stage F does not establish identification.
- **A policy accounting lab:** follow one shock from units and identification
  through a clearly stated policy counterfactual. Require a table defining each
  reported outcome, its numeraire, aggregation weights and uncertainty. The
  fiscal and trade examples would supply two contrasting applications.
- **A compact starter notebook:** build a small reproducible example from data
  to fit, diagnostics, plot and export, then invite the reader to replace the data.
  Keep the learning objective narrow and link to advanced methods for extensions.

These are proposed follow-ups, not new implemented notebooks.
