> 🇬🇧 English · 🇪🇸 [Español](es/ADVISORY.md)

# Correctness advisories

A correctness advisory is issued when a released version of puremacro
returned a **wrong number** — not a crash, not a missing feature, but an
answer that looked well-formed and was not. The distinction matters
because a crash is self-reporting and a wrong number is not: it goes into
a table, a figure, a referee report.

Each advisory names the affected versions, the exact condition under
which the error **vanishes** (so you can rule your own run in or out
without re-running it), and what to do.

---

## 2026-10-03 — the legacy 77x11 aggregation misplaced rest-of-world final demand, all versions; a clean table ships in 4.6.0

**Not a solver defect; a second defect of the legacy data fixture, found while
replacing it. Fixed in 4.6.0 by a new bundled table; the legacy table is kept
unchanged for its MATLAB parity suites.**

While porting the MATLAB aggregation `Agregar_11s.m` to Python
(`tools/build_icio_77c_11s.py`, verified against the bundled
`icio_77c_11s.npz` cell by cell: 99.92% bit-equal, maximum difference 7.6e-5 on
values of order 1e9), two indexing defects of that script surfaced in addition
to the decimal corruption recorded on 2026-09-22. In the block that reassembles
the table around the rest-of-world country (ROW), the loop meant to fill ROW's
final-demand sales to the other 76 countries writes into the intermediate block
instead (`data7fd` written as `data7`), so those sales sit in the intermediate
columns of the first countries while the ROW final-demand block is empty; and
the loop meant to fill the tax and value-added cells of ROW's own final-demand
columns writes into ROW's flow rows (`data5Tfd` written as `data5fd`). On the
legacy table this moves 3,203 cells and 4.9e9 (corrupted-export units) between
blocks; on the clean 2020 release it would move 1.2e5 USD million. Every
number derived from the legacy table inherits both defects.

The 4.6.0 replacement is `icio_77c_11s_oecd2020.npz`, built from the clean
OECD ICIO 2023-edition 2020 file (MD5 `d3e0f4979d85d6c0bb7cf4c43e324287`) with
the corrected aggregation; `MANIFEST_OECD2020.json` records the array digest,
the recipe and the native world totals it conserves (value added 7.97e7 USD
million; output 1.66e8). The author's legacy MATLAB model, run unchanged on the
clean table, is bundled as `trade_reference_solutions_oecd2020.npz` so the
parity check survives the data change, and four gallery cases (`Trade`) check
conservation, column balance, the base equilibrium and that MATLAB parity.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `load_icio_data()` without `source=`, all versions | Returns the legacy table; from 4.6.0 with a `FutureWarning` (once per process) | `source="oecd2020"` | Pass `source="oecd2020"` for anything empirical; `source="legacy"` keeps the regression fixture explicitly. The default switches to `"oecd2020"` in 5.0. |
| Notebooks 61, 62, 64 and their Spanish twins; the trade documentation snippets | They read the legacy table (now pinned with `source="legacy"` so their numbers are unchanged) | — | Their magnitudes remain illustrations of the solver on a regression fixture, not OECD-based estimates, exactly as the 2026-09-22 entry says. |
| `trade_reference_solutions.npz`, `trade_results_workbook.npz` | MATLAB solutions of the legacy table | `load_reference_solution(..., source="oecd2020")` | Use the clean-table references for new parity work; `REFERENCE_MANIFEST_OECD2020.json` lists the scenarios available. |
| `load_oecd_icio_granular`, `puremacro.trade.mrio.read_oecd_native`, `load_raw_45sector_icio` | None: they read the native files by labels | — | Unchanged. |

---

## 2026-10-02 — `solve_trade_equilibrium(method="quasi_condensed")` reported `converged=True` with a large residual, versions 4.2.0 to 4.4.0

**Fixed after 4.4.0** (see the Unreleased section of `CHANGELOG.md`). The
`quasi_condensed` branch of `solve_trade_equilibrium` took its convergence flag
from the quasi-condensed (flexible) solver but its residual, flows and prices
from the legacy post-processing, evaluated at the solver's default MATLAB
pricing convention (`replicate_matlab_precedence=True`), which the
quasi-condensed solve does not use. The flag and the reported residual
therefore disagreed. On the notebook-62 2x2 calibration with a 25% tariff and
`tol=1e-10`, 4.4.0 returned `converged=True` with `max_residual` 0.0487
(`residual_norm` 0.155); with `sigma_y=0.5`, 0.498. The residual is now the one
the flag was judged on, the flows are post-processed at the quasi-condensed
convention and agree with a legacy Newton solve of the same model
(`replicate_matlab_precedence=False`) to 1e-8, and active flexible settings
raise `ValueError` because the legacy flow fields of `TradeEquilibriumResult`
cannot describe them.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `solve_trade_equilibrium(..., method="quasi_condensed")` | any call | — | `x_sol` was the quasi-condensed solution and the flag was right for that model; the residual fields and the flows were not. Re-run on the fixed version, or use `solve_flexible_trade_equilibrium(..., method="quasi_condensed")`. |
| same, with `sigma_y > 0` or an active `config` | flows, prices, CPI, terms of trade | — | These describe the legacy model, not the solved one; take the flows from `solve_flexible_trade_equilibrium`'s `metadata`. |
| same, with `fiscal_closure`/`recycling_params`/`sigma`/`capacity_margins` | the setting was ignored by the solve | the setting was at its default | The returned equilibrium is not a solution of the requested model. |

`solve_flexible_trade_equilibrium` never used this branch; its own
`converged` and `residual_norm` were consistent.

### Are you affected?

```python
res = solve_trade_equilibrium(calib, ..., method="quasi_condensed")
res.converged and res.max_residual > res.metadata["tol"]   # True → the flag and residual disagreed
```

### What to re-run

- Any table built from the flows, prices, CPI or terms of trade of a
  `solve_trade_equilibrium(..., method="quasi_condensed")` result.

---

## 2026-10-02 — Flow properties of `FlexibleTradeEquilibriumResult` ignored the solved model, versions 4.2.0 to 4.4.0

**Fixed after 4.4.0** (see the Unreleased section of `CHANGELOG.md`). The
`exports`, `imports`, `cpi`, `terms_of_trade`, `gdp` and `gdp_fc` properties of
a `solve_flexible_trade_equilibrium` result evaluated the legacy
Cobb-Douglas/Leontief flow equations at `x_sol` with no tariffs and the MATLAB
pricing convention, whatever the call. Measured on 4.4.0, notebook-62 2x2
calibration, 25% tariff on intermediates from country B:

- legacy route (default configuration): exports `[60.73, 68.94]` against
  `[60.73, 69.19]` from `solve_trade_equilibrium` with the same arguments;
- quasi-condensed route, default configuration: exports `[55.87, 63.12]`
  against `[56.73, 64.26]` from a legacy Newton solve of the same model;
- quasi-condensed route, `sigma_trade=5`: exports `[58.20, 59.86]` against the
  solved bilateral flows `[62.15, 58.26]`.

`cpi`, `terms_of_trade` and `gdp`/`gdp_fc` matched on the two default-
configuration cases (they depend on the state, not on the tariff-dependent
quantities); for an active configuration the legacy CPI and terms of trade
describe a different model, and with `variable_markups=True` `gdp` omitted
markup profits. The properties now describe the solved model;
`cpi`/`terms_of_trade` raise `NotImplementedError` for an active configuration.
4.4.0 documented the quasi-condensed half of this as a known issue; the
legacy-route half was not known.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `exports`, `imports` | any `tau`, `tau_fd`, `tauf`, `tauf_fd`; any active configuration on the quasi-condensed route | no tariffs and the default configuration on the legacy route | Re-run, or take row/column sums of `metadata["bilateral_trade"]` (quasi-condensed route). |
| `cpi`, `terms_of_trade` | active flexible configuration | default configuration | No replacement index exists for the flexible model; use the metadata flows. |
| `gdp` | `variable_markups=True` | no markups | Add `metadata["markup_profits"]`, or use `metadata["household_income"]`. |
| `welfare_decomposition` | — | always: it keeps its historical proxy and values | Unchanged. |

### Are you affected?

```python
res = solve_flexible_trade_equilibrium(calib, ...)
tariffs = any(v is not None for v in res.metadata["tariff_inputs"].values())
tariffs or bool(res.metadata["active_flexible_settings"])   # True → re-run
```

### What to re-run

- Trade volumes, trade balances and terms-of-trade effects read from these
  properties for tariff scenarios or active flexible settings.

---

## 2026-10-02 — `minnesota_gibbs` Σ posterior degrees of freedom, versions 0.92.0 to 4.4.0

**Fixed after 4.4.0** (see the Unreleased section of `CHANGELOG.md`).
Bańbura, Giannone and Reichlin (2010; ECB WP 966, p. 12, eq. 7) add the
improper prior `|Ψ|^-(n+3)/2` to the dummy observations, which gives
`Ψ | Y ~ iW(Σ̃, T_d + 2 + T - k)`. `minnesota_gibbs` used `T_d + T - k`. The
posterior mean of Σ, `Σ̃ / (ν - n - 1)`, was therefore too large by the factor
`(ν - n + 1)/(ν - n - 1)`: 5.3% on a 3-variable VAR(1) with 39 observations,
1.0% on 3 variables, 4 lags and 200 quarters. Coefficient draws inherit the
scale through `Σ ⊗ (X*'X*)⁻¹`, so their posterior spread was too wide by about
half that percentage. The posterior mean of the coefficients (`A_mean`,
`intercept_mean`), `minnesota_posterior` and the marginal likelihood of
`minnesota_optimal_lambda` are unaffected.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `minnesota_gibbs` `Sigma_draws`, `nu_post`, posterior bands of `A_draws`/`intercept_draws` | always | point estimates from `A_mean`/`intercept_mean` | Re-run; the effect shrinks as `2/(T + T_d - k - n)`. |

### Are you affected?

```python
T, n = Y.shape                              # your data and lag order p
nu_old = (T - p) + (n * p + n + 1) - (n * p + 1)   # T + T_d - k as used by 4.4.0
(nu_old - n + 1) / (nu_old - n - 1) - 1     # relative overstatement of E[Σ | Y]
```

### What to re-run

- Credible bands of impulse responses, forecasts and variance decompositions
  computed from `minnesota_gibbs` draws, above all for short samples or large
  systems.

---

## 2026-10-02 — Spline and Smolyak solvers ignored `gamma` and solved log utility in their Bellman methods, versions 3.3.0 to 4.4.0

**Fixed after 4.4.0** (see the Unreleased section of `CHANGELOG.md`).
Three related errors in the built-in growth model of `puremacro.vfi.splines`
and `puremacro.vfi.smolyak`, checked against the Euler equation of the CRRA
model written out by hand:

1. Both read only `params["sigma"]`. `params={"gamma": 2}` (the spelling of
   `CollocationProblem`, `FEMProblem` and the discrete engines) silently solved
   log utility: Euler residual 0.151 (spline) and 0.0345 (Smolyak, 2 capitals)
   at the returned policy, which reported `converged=True`.
2. The spline Bellman solver (`method="bellman"`) used log utility for every
   curvature: residual 0.151 at `sigma=2`.
3. `solve_smolyak(method="bellman")` iterated on the closed-form policy
   `k'_m = alpha_m beta Y` of log utility with full depreciation and returned
   it with `converged=True` for any `sigma`, `gamma`, `delta` or `return_fn`
   (residual 0.0345 at `sigma=2`). It now raises `NotImplementedError` outside
   that case.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| spline and Smolyak `method="euler"` | `params` has `gamma` and no `sigma` | `sigma` given, or curvature 1 | Re-run on the fixed version, or pass `sigma`. |
| spline `method="bellman"` | `sigma` (or `gamma`) ≠ 1 | log utility | Re-run on the fixed version. |
| `solve_smolyak(method="bellman")` | `sigma`/`gamma` ≠ 1, `delta` ≠ 1 or a `return_fn` | log utility with full depreciation | Use `method="euler"`. |

### Are you affected?

```python
p = problem.params
curv = p.get("sigma", p.get("gamma", 1.0))
("gamma" in p and "sigma" not in p and curv != 1.0) \
    or (problem.method == "bellman" and (curv != 1.0 or p.get("delta", 1.0) != 1.0))   # True → re-run
```

### What to re-run

- Policies, value functions, marginal values and IFT gradients from these
  solvers with non-log utility (and, for Smolyak Bellman, partial
  depreciation).

---

## 2026-10-02 — `vif` on levels data with a constant, versions 2.6.0 to 4.3.0

**Fixed in 4.4.0.** `inference.collinearity.vif` reproduced statsmodels'
`variance_inflation_factor` operation for operation, including its auxiliary
regressions on the raw levels. When the regressors have means that are large
relative to their spread, those regressions cancel catastrophically and the
VIF loses digits, or all of them. With an explicit constant column a VIF is
exactly invariant to shifting any column, so the non-constant columns are now
computed on centered data; the result agrees with exact rational arithmetic
to within `0.33 * eps * max VIF` on test designs with means up to `1e9`.

Measured on 4.3.0 (statsmodels returns the same numbers): three standard-normal
regressors shifted to mean `1e7` gave 1.007122 for an exact 1.007165 (`4e-5`
relative); an interest rate next to a GDP series in yen (`2e15` scale) gave
0.053, which is impossible since every VIF is at least 1 (exact: 1.005); and
shifted random designs were up to 99% off.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `vif` | `exog` has an explicit constant column and regressors whose mean is large relative to their standard deviation (levels data) | regressors are near zero mean, or demeaned before the call; the constant column's own VIF; designs with no constant column | Re-run on 4.4.0. On affected designs the values now differ from statsmodels, which keeps the old error. |

### Are you affected?

```python
X  # your design, with a constant column
abs(X[:, 1:].mean(axis=0) / X[:, 1:].std(axis=0)).max()   # above about 1e5 → re-run
```

The 4.3.0 error grows with that ratio: below `4e-12` relative up to `1e5`,
`3e-8` at `1e6`, `1e-4` at `1e7`, and more when regressors are also nearly
collinear or trending (the GDP example above).

### What to re-run

- **Any collinearity screen on levels data** (GDP, price indices, population)
  computed with `vif` and an intercept. Demeaned or growth-rate data were not
  affected.

---

## 2026-09-30 — Smets-Wouters (2007) markup shocks, bundled data and replication targets, versions 0.92.0 to 4.3.0

**Fixed after 4.3.0** (see the Unreleased section of `CHANGELOG.md`). Found by
the 30 September notebook review, like every entry dated 2026-09-30. Three
separate errors met in the Smets-Wouters (2007) evidence.

1. In the hand-coded model (`dsge.smets_wouters`), the price and wage markup
   processes were `spinf_t = crhopinf*spinf_{t-1} + epinf_{t-1}` and the same
   for `sw`/`ew`: no MA term and a one-quarter delay. SW07 (ECB WP 722,
   pp. 14-15) and the bundled Pfeifer `.mod` have `+ epinf_t - cmap*epinf_{t-1}`.
   `cmap` and `cmaw` therefore never entered the model, `estimate_sw07` returned
   their prior as their posterior, and the `epinf`/`ew` responses were wrong
   even at `cmap = cmaw = 0`. At the `.mod` calibration the inflation response
   to a unit `epinf` shock in quarters 0-2 was 0.932, 1.416, 0.255; it is 1.177,
   0.307, -0.060. The other five shocks were right.
2. The bundled `_sw07_data.csv` measured hours as `log(HOANBS/pop)`, 1/100 of
   SW07's `100*log(NFB average hours x employment / pop)`, and consumption and
   investment as chained PCE and inventories-inclusive investment rather than
   nominal PCE and fixed investment deflated by the GDP deflator (correlation
   with SW's own series: hours 0.950, investment growth 0.624). Every
   likelihood on the bundled data is affected, through the native model and
   through `sw07_pfeifer.mod` alike.
3. The replication targets -1673.72, -1686.09 and -2524.36 (log posterior,
   Laplace and harmonic-mean marginal likelihoods) were puremacro outputs cited
   as SW07 Tables 1 and 2, and the "mode" case compared the best of 200 MCMC
   draws with Table 1a posterior **means**. The cases also read their fixture
   from `tests/fixtures`, so they failed in an installed wheel (since 2.6.0).

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `solve_sw07`, `smets_wouters._shock_irf` and anything built on them | Responses, FEVD or decompositions involving `epinf` or `ew` | The other five shocks (`ea`, `eb`, `eg`, `eqs`, `em`) | Re-run. The responses of all seven shocks now equal the `.mod` solution to 1e-12. |
| `estimate_sw07`, `sw07_observation.make_state_space` | Always: the model and the data both changed | Never | Re-estimate. Do not compare old and new log posteriors or marginal likelihoods. |
| `_sw07_data.csv` with `sw07_pfeifer.mod` (notebooks 41 and 42) | Any estimate, smoother, shock decomposition or marginal likelihood on the bundled data | Your own data | Re-run on the rebuilt file. |
| `dsge_estimation` replication cases (2.6.0 to 4.3.0) | Always | Never | The targets are now labelled puremacro regression values (-822.04, -902.83, -908.03); the mode case compares an optimised mode with the Table 1 Mode column (13 parameters, all within 6.2%). |

### Are you affected?

```python
import pandas as pd
from importlib.resources import files
d = pd.read_csv(files("puremacro.dsge") / "_sw07_data.csv", comment="#")
d["log_hours"].std()   # about 0.046: the 4.3.0 data; about 2.9: the rebuilt data
```

Any SW07 number computed on the 4.3.0 data, and any SW07 response to a markup
shock from the native model, is affected.

### What to re-run

- **Any SW07 estimation, marginal likelihood, smoother or shock decomposition**
  on the bundled data, native or `.mod`.
- **Any `solve_sw07` response, FEVD or decomposition** involving `epinf` or `ew`.
- **Any citation of -1673.72, -1686.09 or -2524.36 as SW07 results.** With the
  fixes, the optimised posterior mode on the bundled data has log posterior
  -822.04 and Laplace log marginal density -902.83. SW07's own -905.8 (Table 2)
  is computed over 1966-2004 with 1956:1-1965:4 as a training sample; since
  4.6.0 `sw07_laplace_mdd` reproduces that computation on the authors' data
  (presample and diffuse initialisation), giving -932.3, and Dynare 8 on the
  public replication files and the authors' own mode gives -923.1 with the
  file's options (puremacro at the same point: -922.4). The printed -905.8 is
  reproduced by neither and remains not a comparable target (see
  `docs/replication.md`).
- **In this repository:** notebooks 41 and 42 (English and Spanish),
  `docs/replication.md` and the JOSS draft cite the affected numbers.

---

## 2026-09-30 — `.mod` model-local variables frozen at the calibration, versions 2.0.0 to 4.3.0

**Fixed after 4.3.0.** A `.mod` model-local variable defined from parameters
only (`#cbeta = 1/(1+constebeta/100);`) was evaluated once, at the file's
calibration, and the number was inlined (from 2.7.0; the regex reader, the only
reader from 2.0.0 to 2.6.0 and a fallback since, stored it as a fake
parameter). Dynare substitutes the expression.
Every computation that re-solves the model at other parameter values therefore
used a hybrid model: parameters that appear directly in the equations moved,
the locals built from them did not.

In `sw07_pfeifer.mod`, `constebeta` enters the model only through locals, so
its likelihood was exactly flat and its posterior equalled its prior; `csigma`,
`ctrend`, `constepinf`, `calfa`, `cg`, `ctou` and `clandaw` entered only in
part. Re-solving at `csigma = 2.5` gave sd(y) = 30.1307 instead of 29.9343 (a
fresh load). On the current bundled data the log-likelihood at the
`estimated_params` initial values is -1766.35, not -1776.93, and `constebeta`
at 0.5 / 1.0 / 1.5 gives -1749.77 / -1794.92 / -1881.05 where all three used to
give -1776.93.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `load_mod(..., params=)`, `osr` and optimal-policy searches, parameter widgets, SMC, analytic gradients and NUTS, `identification()`, `LinearModel.estimate()` / `estimate_dsge` on a `.mod` model | The model has a `#` local whose right-hand side contains only parameters and numbers, and the computation moved one of those parameters | Results at the file's own calibration; locals that involve endogenous variables; models written as Python residual functions; the hand-coded `estimate_sw07` | Re-run with the fixed release. |
| `load_mod(..., params=)` with a `steady_state_model` or `initval` block | The block depends on an overridden parameter | No such dependence | It raised `SteadyStateError`; it now re-evaluates the block at the overrides. |
| Parameter or variable names that are Python keywords (`lambda`, `yield`) | Always | Other names | They raised `SyntaxError` or `ValueError`; they now parse. |

### Are you affected?

Look for a line starting with `#` in the `model;` block whose right-hand side
contains only parameters and numbers. If there is one, and you ran any
computation in the first row at parameter values other than the file's
calibration, you are affected.

### What to re-run

- **Any estimation, identification analysis, `osr` search, gradient or
  comparative-statics result** from such a `.mod` file.
- **In this repository:** notebooks 41 and 42 and curso T02_E
  (`sw07_pfeifer.mod`).

---

## 2026-09-30 — `identification()` evaluated at a mixture of the calibration and the requested point, versions 2.6.0 to 4.3.0

**Fixed after 4.3.0.** `dsge.identification` and `LinearModel.identification`
computed every reported quantity (ranks, `is_identified`, singular values,
condition numbers, null-space combinations, collinearity, Ratto strength) from
a mixture of the calibration and the point the user asked about. Each
structural column moved only its own parameter off the calibration; the shock
and measurement-error columns used the calibrated solution, with names-only
`SE_`/`CORR_` taken at 1 and 0 rather than the declared values; and the
one-sided fallback differenced against the calibrated solution (on an AR(1) it
returned -78,594.7 where the true value is 1.78). `prior_mc` jittered the base
values by 5% instead of drawing from the priors. Separately, `build_dynare`
detects the predetermined variables numerically at the calibration, so a lag
calibrated to 0 was dropped from every re-solve and every derivative with
respect to its coefficient was exactly zero.

Genuinely unidentified parameters could be reported as identified (a `k1*k2`
pair in a `.mod` whose prior means differ from the calibration showed ranks
2/2/2/2 instead of 1/1/1/1), and identified ones as unidentified: SW07's
`crhoms`, `crhopinf`, `crhow`, `cmap` and `cmaw`, calibrated at 0, had
sensitivity exactly 0, with ranks 29/31/29/29 of 36 on the default call instead
of 34/36/34/34.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `identification()` with `params={...}`, `p_dict`, `EstimatedParams` or a names list on a `.mod` with `estimated_params` | The analysed point (values, INITVALs or prior means) differs from the calibration | The point is the calibration and none of the rows below applies | Re-run. |
| The default call on a `.mod` with `estimated_params` | INITVALs or prior means differ from the calibration | They equal it | Re-run. |
| `prior_mc > 0` | Always | Never | Re-run; the identification rates were not prior-based. |
| A parameter on, or within one finite-difference step of, a declared bound | Always | Away from bounds | Re-run. |
| `build(..., linearize="level")` models | A positive steady state, even at the calibration | Log-linearised or zero-steady-state models | Re-run. |
| `SE_<shock>`, `CORR_<s1>_<s2>` or `ME_<obs>` given by name | The declared shock covariance is not the identity, or a measurement error is declared, even at the calibration | Identity covariance, no measurement error | Re-run; names-only `SE_`/`ME_` now use the declared values. |
| `build_dynare` / `load_mod` models | A lag coefficient is 0 at the calibration and an analysed parameter, or the requested point, makes it nonzero | No lag calibrated to 0 | Re-run. |
| `LinearModel.estimate(check_identification=True/"raise")` | Any of the conditions above at the estimation's starting point | As above | The pre-flight could let an unidentified model through or raise spuriously. |

### Are you affected?

Only results computed at the calibration, with no `prior_mc`, away from
bounds, on a model not covered by the last four rows of the table, were
correct. Otherwise re-run with the fixed release. Note that a requested point
with no unique stable solution now raises `ValueError`.

### What to re-run

- **Any identification analysis in the cases above**, including ranks and
  strength rankings reported for Smets-Wouters (2007).
- **In this repository:** notebook 45 calls `identification()` at the
  calibration, where the defect does not bite; its numbers are unchanged.

---

## 2026-09-30 — two-asset HANK sequence-space solver, versions 3.1.0 to 4.3.0

**Fixed after 4.3.0.** Results of the two-asset solver before this fix are not
valid. The household iteration never converged: it settled into a 2-cycle
(sup-norm change 0.39) while `converged` was always True. With the default
calibration β(1+r_a) = 1.0146 > 1, so all illiquid wealth sat at the grid
ceiling a_max = 30. The Fake-News Jacobians had no anticipation terms (their
strict upper triangles were zero) and no ex-ante date shift for r_b, and
wealth leaked where b' > b_max was clipped. The general-equilibrium response to
a monetary shock therefore depended on the horizon: after a 25 bp cut, impact
consumption was -0.0038 at T = 16, +0.0014 at T = 20, -0.0045 at T = 30 and
-0.0027 at T = 40, and the responses did not decay.

With the fix the same calls give +0.0031 at every horizon from 16 to 300 and
the responses decay; the steady state has A = 4.59, B = 0.61 and a
hand-to-mouth share of 0.39, with no mass at a_max. The defaults change (β
0.985 to 0.98, `r_b_ss` 0.01 to 0.005, `r_a_ss` 0.03 to 0.0125, `chi_0` 0.25 to
1.0, `a_max` 30 to 40, `b_max` 15 to 10, new `a_bar = 0.25`),
`steady_state["Y"]` is C + CHI, and the Jacobians are those of the household
sector closed by the income rule z_t N = Y_t - r^b_t B_{t-1} - r^a_t A_{t-1},
under which total wealth A + B is constant in equilibrium.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `solve_two_asset_hank_sequence_space` | Always | Never | Re-run with the fixed release; its calibration differs, so compare shapes, not levels. |
| `solve_hank_bridge` on `hank_two_asset.mod`, `HANKModel` with `assets = 2` | Always | Never | Re-run. |
| One-asset solver (`solve_hank_sequence_space`, `hank_ssj.mod`) | — | Always: its results are bit-identical | Nothing to do. |

### Are you affected?

Any number from the three two-asset surfaces above on 3.1.0-4.3.0. An
illiquid-wealth steady state equal to `a_max`, or a response whose sign
changes with the horizon `T`, is the signature.

### What to re-run

- **Any two-asset HANK response, Jacobian or steady state.**
- **In this repository:** notebook 46 (English and Spanish) printed impact
  consumption -0.0038 while its text says it rises; the call now gives +0.0031.
  The fixed solver is a stylised teaching model checked against brute-force
  Jacobians, budget identities and horizon invariance, not against published
  two-asset results.

---

## 2026-09-30 — Markov-switching VAR estimation, versions 0.92.0 to 4.3.0

**Fixed after 4.3.0.** The EM M-step of `var.regime.ms_var_fit` updated the
shared AR matrix by unweighted OLS, ignoring the regime covariances. When the
regime variances differ, that step does not maximise the EM objective: the
log-likelihood could fall across iterations and the fit stopped below the
maximum-likelihood estimate while reporting `converged=True`. An absolute ridge
of 1e-8 on every regime covariance also made the estimates depend on the units
of the data. The docstring claimed the function reproduces Hamilton (1989); it
estimates a different model, a switching intercept and variance (MSIH) with a
shared AR matrix.

On Hamilton's GNP growth data (K = 2, p = 4) the default call returned loglik
-180.48, not converged; its low-intercept regime had a positive intercept
(0.31), P(stay) = 0.87 and covered 66 of 131 quarters. The corrected fit gives
loglik -179.16, intercepts 1.21 and -0.06, P(stay low) = 0.79 and 35 low-mean
quarters. With one variable in units 1e-4 times smaller, the old fit lost 173
log-likelihood points.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `ms_var_fit` estimates, filtered and smoothed probabilities, and `girf` built on them | Regime variances that differ, or a variable with small absolute variance (for example rates in decimals) | Regime variances nearly equal and variables of unit scale (the bundled business-cycle example moves little: accuracy 98.0% to 97.2%) | Re-estimate; check `loglik_path`, `converged` and `sigma_at_floor`. |
| Claims that `ms_var_fit` reproduces Hamilton (1989) | Always | — | Use statsmodels `MarkovAutoregression(k_regimes=2, order=4, switching_ar=False)` for Hamilton's Table I. |

### Are you affected?

Re-fit with the fixed release and compare `loglik`: a higher value means the
old estimate was not the maximum. On the old release a result with
`converged=False`, or with `n_iter` equal to the cap (100 by default), is a
sign that it stopped early.

### What to re-run

- **Any `ms_var_fit` regime dating, transition matrix or regime-dependent
  response.**
- **In this repository:** notebook 16 and the example `ms_var_business_cycle`.

---

## 2026-09-30 — dynamic-panel two-step Windmeijer standard errors, versions 0.92.0 to 4.3.0

**Fixed after 4.3.0.** `dynpanel.ab_gmm` and `dynpanel.bb_gmm` with the
defaults `two_step=True, windmeijer=True` computed their standard errors with
the derivative of the weight matrix evaluated at the step-2 residuals.
Windmeijer (2005) and Stata use the one-step residuals. Coefficients were not
affected. Separately, `collapse=False`, which is Stata's default layout, raised
`LinAlgError` ("step1 weight Z'HZ: matrix is not positive definite") for any
model with two or more lags of the dependent variable or with lagged exogenous
regressors, because an all-zero instrument column was kept.

On Stata's abdata example ([XT] xtabond Example 4), with the dead column
removed by hand, the old standard errors were 1.4% to 68% too large (`w`:
0.2602 against 0.1546), so inference was too conservative there. On the default
collapsed layout the error is usually below 2% and runs in either direction
(old against corrected: 0.554759 against 0.564362 for `ab_gmm` and 0.084758
against 0.082478 for `bb_gmm` on simulated panels; up to about 4% on abdata). The fixed `ab_gmm`
reproduces every published coefficient and standard error of xtabond Examples
1, 2 and 4.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `ab_gmm`, `bb_gmm` standard errors, z-statistics, p-values, intervals | `two_step=True` and `windmeijer=True` (the defaults) | `two_step=False`, or `windmeijer=False`; coefficient estimates in every case | Re-run and replace the reported standard errors. |
| `diagnostics.windmeijer_correction` called directly | Without the new `residuals_step1=` | With it | Pass the one-step residuals; the old call warns. |
| `collapse=False` | `lag_dep_var >= 2`, lagged exogenous regressors or gaps | Other layouts | It raised; it now prunes the dead columns, as Stata does. |

### Are you affected?

Every two-step fit with the Windmeijer correction, that is every call with the
default settings. Only the standard errors and what is built on them change.

### What to re-run

- **Two-step dynamic-panel fits:** replace their standard errors. To match
  Stata `xtabond`, use `collapse=False` (and `two_step=False, robust=False` for
  its one-step homoskedastic VCE).

---

## 2026-09-30 — Minnesota prior mis-centred for λ₂ ≠ 1, versions 0.92.0 to 4.3.0

**Fixed after 4.3.0.** In `var.bvar._build_minnesota_dummies`, the dummy
observations behind `minnesota_gibbs`, its marginal likelihood and
`minnesota_optimal_lambda` wrote each (lag, variable) cell once per equation,
and the last equation won. With the default λ₂ = 0.5 the own first lag of every
variable except the last one listed was centred on 0.5 instead of 1 (the random
walk), with prior s.d. λ₁λ₂ instead of λ₁; the last variable had λ₂ ignored in
the cross equations; and results depended on the column order. In the
tight-prior limit (λ₁ = 1e-4) `minnesota_gibbs` returned A1 = diag(0.5, 0.5,
1.0) instead of I. The conjugate Normal-inverse-Wishart prior cannot encode
λ₂ ≠ 1 at all: Bańbura, Giannone and Reichlin (2010) impose it "under the
condition that θ = 1". The validation case was pinned at λ₂ = 1, the one value
at which the bug cannot show.

The block is now BGR eq. (5); the NIW paths default to λ₂ = 1, and any other
value warns and uses 1. On a 3-variable VAR(1) with T = 200 and the defaults,
the Gibbs posterior mean of A1[0,0] moves from 0.501 to 0.545.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `minnesota_gibbs`, `minnesota_optimal_lambda`, their marginal likelihood | λ₂ ≠ 1, which includes the old default 0.5 | λ₂ = 1 | Re-run. For a posterior mean with λ₂ < 1, use `minnesota_posterior`; there is no NIW posterior with λ₂ < 1. |
| `minnesota_posterior`, `bvar_sv` | — | Always | Nothing to do. |

### Are you affected?

```python
res = minnesota_gibbs(...)   # your call on 4.3.0 or earlier
res["lambda2"]               # any value other than 1.0 (0.5 by default) → affected
```

### What to re-run

- **Any analysis built on `minnesota_gibbs` or `minnesota_optimal_lambda`.**
- **In this repository:** the examples `bvar_fan_chart` (h = 4 80% band
  [-0.876, +0.692] becomes [-0.947, +0.707]), `posterior_predictive_fan`,
  `mcmc_diagnostics` (chain mean of A1[0,0] 0.51 becomes 0.56),
  `gk_robust_from_gibbs` and `glp_lambda_search` (selected λ₁ 0.2 becomes 0.3).

---

## 2026-09-30 — synthetic DiD intervals, and silent DiD for outcomes in large units, versions 0.92.0 to 4.3.0

**Fixed after 4.3.0.** Two defects in `did.synthetic_did`, and in
`sdid_multi_cohort` through its point estimates.

1. **Under-covering intervals.** `se`, `lo` and `hi` came from a bootstrap that
   resampled the donor units only and held the treated units fixed, with a
   percentile interval, so the treated units' own noise was left out. With one
   treated unit a nominal 90% interval covered the truth 50-67% of the time in
   Monte Carlo, and the reported `se` was about half the true s.d. (notebook-29
   design: mean 0.13-0.14 against 0.23-0.24). On the documentation example with
   8 treated units the `se` was 0.019 against a true s.d. of about 0.057.
2. **Silent collapse to difference-in-differences.** When SLSQP reported
   non-success, the weights fell back to uniform (1/N_co for ω, 1/T_pre for λ)
   without a warning, so τ̂ was the plain DiD estimate. SLSQP's tolerance is
   absolute, so this happened mainly for outcomes in large units (notebook-29
   data times 10 or more, levels in the hundreds). On the paper's California
   Prop 99 data puremacro returned -27.35, the DiD value, instead of the SDID
   -15.6.

The noise level σ̂ was also demeaned period by period rather than as in the
paper's eq. (2.2); point estimates move by about 1e-2 (notebook 29: -4.0940 to
-4.0827).

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `synthetic_did` `se`, `lo`, `hi` | Always | `n_boot=0` (no inference) | Re-run. The default is now the placebo estimator (Algorithm 4) for one or two treated units and the unit bootstrap (Algorithm 2) otherwise; report the placebo `se` for one or two treated units. |
| `synthetic_did` and `sdid_multi_cohort` point estimates | The weight solve failed: uniform weights | The weights are not uniform | Re-run; the fixed solver is scale-free and warns if unconverged. |
| All point estimates | Always, by about 1e-2 through σ̂ | — | Re-run if the second decimal matters. |

### Are you affected?

```python
import numpy as np
res = synthetic_did(...)                          # your call on 4.3.0 or earlier
np.allclose(res.omega.values, 1 / len(res.omega)) # True → the reported "SDID" was DiD
```

Every reported SDID standard error or interval is affected.

### What to re-run

- **Every SDID analysis**, point estimates for outcomes in large units and all
  intervals.
- **In this repository:** notebook 29 (English and Spanish; its stored 90%
  interval excludes the true effect), `docs/did.md` and the example
  `synthetic_did_california_prop99`, which uses simulated data despite its name.

---

## 2026-09-30 — Callaway-Sant'Anna aggregation, versions 0.92.0 to 4.3.0

**Fixed after 4.3.0; the default changes.** `did.callaway_santanna` computed
its group-time effects ATT(g, t) correctly but aggregated them with **equal
cohort weights**, and presented the result as the Callaway-Sant'Anna
estimator. The paper weights every aggregate by cohort size (arXiv:1803.09015v4:
event study eq. 3.4; overall summaries eqs. 3.10-3.12). `att_overall` was the
plain mean of the post-treatment ATT(g, t) cells, which matches none of the
paper's overall parameters. The documentation disclosed the unweighted rule but
attributed it to CS, and the teaching material had the comparison backwards:
puremacro's own `sun_abraham` returned CS eqs. 3.4 and 3.10 exactly.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `att_event_study` | Two or more cohorts identify an event time and they differ in size | All cohorts at that event time have equal size, or equal ATT(g, g+e) | Re-run. Noiseless example: **2.0 against 2.6** at e = 0 with cohorts of 10 and 40 units. |
| `att_overall`, and the legacy dictionary's `overall_att` | Any heterogeneity in ATT(g, t) | Every ATT(g, t) is equal (then all overall summaries coincide) | Now θ^O_sel (eq. 3.11); `aggregation=` selects the others. Notebook 10: 1.1292 to 1.1379; curso T05_C: 7.890 to 6.391 (there `aggregation="simple"` keeps 7.890 and is the cell-average estimand). |

### Are you affected?

With the fixed release, `callaway_santanna(..., aggregation="unweighted")`
reproduces the old numbers exactly. Run it next to the default: where the two
differ, your old result was not the Callaway-Sant'Anna estimand.

### What to re-run

- **Any Callaway-Sant'Anna event study or overall ATT** on a panel whose
  cohorts differ in size and whose effects differ across cohorts or over time.
- **In this repository:** notebooks 10 and 15, curso T05_C and T05_D,
  `docs/did.md`, the example `did_callaway_santanna_demo` (1.0142 to 1.0278)
  and the 2.0 tour `00_whats_new_in_puremacro_2_0` (English and Spanish; overall
  ATT 1.1876 to 1.0904, its event study unchanged because its two cohorts have
  the same size).

---

## 2026-09-30 — Sun-Abraham standard errors, versions 0.92.0 to 4.3.0

**Fixed after 4.3.0.** `did.sun_abraham` aggregated the per-cohort standard
errors as if the cohorts were independent, `sqrt(Σ w_g² se_g²)`. Every cohort's
2×2 comparison subtracts the same control units, so the cohort estimates are
correlated. Since 1.10.0 the bands `lo`/`hi` have been built from this `se`
(before 1.10.0 they had the separate error described in the 2026-09-03 entry),
so they inherited it. The error goes either way: with serially correlated or
trending errors the `se` was 0.74 times the Monte Carlo s.d. in our runs (0.62
in the verification), and 90% bands covered 77%; with iid errors at adjacent
cohorts it was up to 1.40 times too large. Point estimates stand.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| Event-study `se` (0.92.0 to 4.3.0), `lo`/`hi` (1.10.0 to 4.3.0) | Two or more cohorts contribute to an event time | One cohort per event time | Re-run; the `se` is now the bootstrap s.d. of the aggregate over joint unit-level draws (SA 2021, Prop. 6). |
| `att_overall` | No standard error was reported | — | Now `att_overall_se`/`_lo`/`_hi`. |
| `honest_did` fed with a Sun-Abraham event study | As in the first row | As in the first row | Re-run; pass `event_study_vcov` to `honest_did(sigma=)`. |

### Are you affected?

Any event time with more than one contributing cohort. In a unit-trend panel
the e = 0 `se` goes from 0.111 to 0.174 (closed form 0.170).

### What to re-run

- **Any Sun-Abraham event-study `se` or band** at an event time with more than
  one contributing cohort, and any `honest_did` run built on one.
- **In this repository:** notebooks 10 and 15 and curso T05_C.

---

## 2026-09-30 — lag-augmented local projections, versions 0.92.0 to 4.3.0

**Fixed after 4.3.0; the default changes.** `lp.la_lp` (and `lp.la_lp_iv`,
from 4.0.0) added `extra_lags = max(horizons)` lags at every horizon, and gave
controls only `n_lags` lags; the method was credited to Plagborg-Møller and
Wolf (2021), with an unsourced "p_aug = p + h" rule. The lag-augmented local
projection of Montiel Olea and Plagborg-Møller (2021, *Econometrica* 89(4))
adds a single lag. The old default stays asymptotically valid, but its
finite-sample coverage is worse and its bands wider: at T = 228 it covered
0.79-0.84 against 0.84-0.87 for one extra lag (nominal 0.90). The estimate at a
given h also depended on the largest horizon requested, and `horizons=[0]` had
no augmentation.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `la_lp`, `la_lp_iv` point estimates and HC0 bands | `extra_lags` not passed, or any call with `controls=` | An explicit `extra_lags`, no controls | Re-run. Romer-Romer tax LP (T = 228, H = 20): 2-year response -3.15 becomes -2.86, and the peak moves from -3.15 at h = 8 to -2.87 at h = 10. |

### Are you affected?

```python
res = la_lp(df, y, x, horizons=H, n_lags=p)   # your call on 4.3.0 or earlier
res["p_aug"].iloc[0] > p + 1                  # True → the old default augmentation
```

Any call with `controls=` is also affected. To reproduce an old number exactly
with the fixed release, pass `extra_lags=max(horizons), control_lags=n_lags`.

### What to re-run

- **Any `la_lp` / `la_lp_iv` result** made with the default `extra_lags` or
  with `controls`.
- **In this repository:** notebooks 14 and 50, curso T04_A and T05_B,
  `notebooks/course` lesson 06, `docs/lp.md` and the example `la_lp_pmw_demo` (notebook 14's
  specification-curve median goes from -2.30 to -2.64).

---

## 2026-09-30 — fixed-b critical values, versions 0.92.0 to 4.3.0

**Fixed after 4.3.0.** The hard-coded Bartlett fixed-b table behind
`inference.llsw_critical_value` and the `llsw_cv_90` key of
`inference.hac_fixed_b` did not match Kiefer and Vogelsang (2005, Table I). Its
comment cited Lazarus, Lewis, Stock and Watson (2018) Table 2, which holds
bandwidth constants, not critical values, and its two-sided 1% values came from
no published table. From about b = 0.1 the values were too small, and the
shortfall grows with b. The old code interpolated linearly between its wrong
b = 0.1 and b = 0.2 rows, so every b in between is affected: the two-sided 5%
value was 1% below the correct one at b = 0.10, 3% at b = 0.12, 5% at b = 0.15
(2.265 instead of 2.386), 9% at b = 0.2 and 14-20% from b = 0.3 (3.96 instead of
4.771 at b = 1). Tests therefore over-rejected and intervals were too narrow.
The two-sided 1% values were too small at every b from 0.05. Below b = 0.1 at
the 10% level, and up to b = 0.08 at the 5% level, the values were too large,
by up to 8% at b = 0.02, so those tests were conservative (a nominal 10% test
rejected 8.6% of the time at b = 0.05).

| b | Nominal 5% rejected | Nominal 10% rejected | Nominal 1% rejected |
|---|---|---|---|
| 0.05 | 4.5% | 8.6% | 1.1% |
| 0.10 | 5.4% | 9.9% | 1.5% |
| 0.12 | 5.7% | 10.3% | 1.6% |
| 0.15 | 6.2% | 11.0% | 1.8% |
| 0.2 | 7.0% | 12.1% | 2.1% |
| 0.5 | 9.3% | 15.4% | 2.9% |
| 1 | 8.8% | 16.3% | 1.8% |

(null rejection rates of the `hac_fixed_b` t-statistic against the old
`llsw_critical_value(b, alpha)`, iid N(0,1) data, T = 200, averaged over
200,000 draws of the variance estimate. With the corrected values the same
tests reject 4.8-5.0%, 9.5-10.1% and 1.0%.)

Lazarus, Lewis, Stock and Watson recommend the Newey-West bandwidth
S = 1.3√T, that is b = 1.3/√T. That b lies between 0.1 and 0.2 for T between
about 42 and 169.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `llsw_critical_value`, `hac_fixed_b(...)["llsw_cv_90"]` | b above 0.1 at any level (at 5%, from b = 0.1), the shortfall growing from about 1% at b = 0.1 to 9% at b = 0.2 (two-sided 5%) and more above; alpha = 0.01 from b = 0.05; below b = 0.1 at 10% and up to b = 0.08 at 5%, values too large (conservative) | The default b = 0.10 at alpha = 0.10: the old 1.86 compares with 1.861. `hac_fixed_b` now looks its value up at b_eff = (floor(bT) + 1)/T, which gives 1.871 at T = 200 and 1.902 at T = 50; the old 10% test rejected 9.9% and 10.4% of the time there. | Re-run the test or interval. Old and new two-sided values at b = 1: 3.05/3.96/6.29 become 3.764/4.771/7.083 at 10/5/1%. |
| `hac_fixed_b` standard errors, t-statistics, `vcov` | — | Always, apart from a one-lag rounding fix for b·T such as 0.29·100 | Nothing to do. |
| An alpha outside {0.2, 0.1, 0.05, 0.02, 0.01}, or b outside (0, 1] | It was silently snapped or clamped | — | It now raises `ValueError`. |

### Are you affected?

```python
from puremacro.inference import llsw_critical_value
llsw_critical_value(0.15, 0.05)  # 2.265 on 4.3.0 and earlier; 2.386 fixed
llsw_critical_value(1.0, 0.05)   # 3.96 on 4.3.0 and earlier; 4.771 fixed
```

If your b was above 0.1, or your level 1%, re-run. At b = 0.10 and the 5% level
the old test rejected 5.4% of the time at T = 200 and 5.8% at T = 50; decide
whether that margin matters for your result.

### What to re-run

- **Any test or interval built with `llsw_critical_value` or `llsw_cv_90`** at
  b above 0.1, or at alpha = 0.01. This includes the LLSW rule b = 1.3/√T for
  samples of about 42 to 169 observations. No notebook or example in this
  repository uses them.

---

## 2026-09-30 — AKM shift-share standard error, versions 2.4.0 to 4.3.0

**Fixed after 4.3.0.** `se_akm` of `bartik.shift_share_iv`, and therefore the
default `se`, t-statistic, p-value and interval, did not implement Adão,
Kolesár and Morales (2019) whenever the regression had unit-level controls
besides the intercept, or exposure shares that do not sum to one. The code
residualised the raw sector shocks on a share-weighted constant; AKM use X̂, the
regression of the control-partialled instrument on the shares (Remark 5,
eqs. 28 and 39). There was also no way to cluster sectors (eq. 40). The error
goes either way: on the ADH data of the ShiftShareSE vignette the standard
error was 0.1816 instead of 0.2101 (13.5% too small); on a synthetic case with
one regional control it was 0.4117 instead of 0.2120.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `shift_share_iv` AKM standard error and inference | Controls besides the intercept, or shares that do not sum to one | Intercept only and complete shares (for example the `docs/spatial.md` example) | Re-run. Pass `sector_clusters=` when shocks may be correlated within groups of sectors (3-digit SIC clusters reproduce the vignette's 0.2403730). `akm_shocks="residualized"` gives the old number. |

### Are you affected?

```python
import numpy as np
np.allclose(shares.sum(axis=1), 1.0)   # False → affected
# also affected if you passed controls besides the intercept
```

### What to re-run

- **Any AKM standard error, test or interval** from a regression with controls
  or incomplete shares. No notebook in this repository uses them.

---

## 2026-09-30 — MCMC diagnostics: Geweke z-scores and effective sample size, versions 0.92.0 to 4.3.0

**Fixed after 4.3.0.** `mcmc.geweke_z` estimated the long-run variance of each
segment with a Bartlett window of only 5-7 lags on the default segments, far
too short for autocorrelated MCMC output. With positively autocorrelated chains
(random-walk Metropolis and Gibbs, the only kind `estimate_dsge_bayesian`
passes to it) the variance was underestimated, so stationary chains were
declared non-converged far more often than the nominal 5%: with 5,000
stationary AR(1) draws, 9% of the time at φ = 0.5, 29-33% at 0.9, 46-49% at 0.95
and 76% at 0.99, against 6%, 7%, 9% and 19% now (7% at 0.99 with 50,000
draws). These were false alarms, not false passes. With antithetic chains the
old estimator was mildly conservative.

`mcmc.effective_sample_size` summed autocorrelation pairs from lag 1 with no
monotone step. It could never exceed n, and it returned exactly n whenever the
lag-1 plus lag-2 autocorrelation was negative, the typical case for antithetic
chains (true ESS about 3n at φ = -0.5).

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `geweke_z`, `geweke_z_<param>` and `geweke_z_max` of `estimate_dsge_bayesian` (from 2.1.0), `trace_summary` | Persistent chains: the absolute z-score was inflated | Chains with negligible autocorrelation | Re-run the diagnostics. `geweke_z(..., spectrum="bartlett")` reproduces the old numbers. |
| `effective_sample_size`, NUTS bulk and tail ESS | Antithetic chains; noisier for positively autocorrelated ones | — | Re-run; NUTS ESS shifts by a few percent (notebook 43: 87.2 to 90.6). |
| `gelman_rubin` | — | Always: it is the classic statistic, unchanged | Use the new `split=True`, or `trace_summary()["R_hat_split"]`, to detect within-chain drift. |

### Are you affected?

A |z| between 2 and about 5 on a long, well-mixed random-walk Metropolis
chain, reported by 4.3.0 or earlier, was probably a false alarm. On the
synthetic AR(1) Bayesian test model, `geweke_z_max` goes from 1.19 to 0.78.

### What to re-run

- **Convergence diagnostics** of Bayesian DSGE estimation and of any chain
  passed to `geweke_z` or `effective_sample_size`.
- **In this repository:** the example `mcmc_diagnostics` and notebook 43's ESS
  table.

---

## 2026-09-30 — real-time nowcast replays used later vintages, versions 4.1.0 to 4.3.0

**Fixed after 4.3.0.** With `as_of` set before the panel's last vintage,
`realtime_nowcast` computed its news decomposition (`forecast_old`, `revision`,
`news_table`, `revision_table`), `vintage_history` and `news_vs_noise_test`
against vintages published **after** `as_of`: the default baseline was the
panel's second-to-last vintage. When `as_of` was that vintage, the
decomposition was identically zero, and `summary()` could print "Analytical
Identity Error ... (< 1e-10)" while the error was about 1e-2. In an example
with five monthly vintages and `as_of` on the second, the decomposition changes
from forecast_old -0.0828 / revision +0.0270 (error 2.7e-2, baseline the fourth
vintage) to +0.1002 / -0.1560 (error 5.6e-17, baseline the first).

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `news_decomposition`, `vintage_history`, `news_vs_noise_test` | `as_of` earlier than the panel's last vintage | No `as_of` | Re-run the replay. |
| The point nowcast, `forecast_sd`, factors and loadings | — | Always: they were already point-in-time | Nothing to do. |
| Code reading `vintage_history["nowcast"]` | Always | — | Read `vintage_history["last_observed_value"]`: the column always held the last observed data value, never a model nowcast. |

### Are you affected?

Any pseudo-real-time replay that passed `as_of`. Calls without `as_of` change
only in the relabelled `vintage_history` columns.

### What to re-run

- **Any pseudo-real-time news decomposition or news-versus-noise test** made
  with `as_of`. Notebook 59 uses the latest vintage, so its numbers do not
  change; only its printed summary block does.

---

## 2026-09-30 — Gertler-Karadi units (2.3.0 to 4.3.0) and the sign of futures-price surprises (0.92.0 to 4.3.0)

**Fixed after 4.3.0.** Two conventions that the documentation did not state
led to wrong numbers.

1. `GertlerKaradiResult.irf` / `to_frame()` has always held **level**
   deviations x_t - x_ss, because the model is linearised in levels.
   Multiplying them by 100 and calling the result "%" is wrong for every
   variable whose steady state is not 1. At the default solve (OccBin, a -5%
   capital-quality shock), impact values in percent are: capital -5.06 (not
   -28.66; trough -13.49, not -76.38), net worth -49.15 (not -67.88),
   investment -7.10 (not -1.01), output -1.60 (not -1.35), consumption -0.65
   (not -0.35). Q is unaffected because Q_ss = 1.
2. `hfi.gk2015_surprise` returns `post - pre`, scaled, and its docstring
   accepted a futures **price** or an implied rate. With prices (quoted as
   100 - rate) the surprise came out sign-reversed: a 25 bp tightening gave
   -0.5 instead of +0.5. That flips proxy-SVAR responses (`proxy_svar` applies
   no sign normalisation by default) and swaps Jarociński-Karadi monetary and
   information shocks.

`aggregate_to_period` also sums surprises within the calendar month, whereas
Gertler and Karadi (2015, footnote 11) use a period average; results that claim
to replicate their monthly instrument should use the new `method="gk2015"`.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `solve_gertler_karadi` results reported in percent | The level deviations were multiplied by 100 | Level deviations reported as such; Q | Use `to_frame(units="pct")`. |
| `gk2015_surprise` and what is built on it | Futures prices passed as inputs | Implied rates passed as inputs | Re-run with `quote="price"`, or pass implied rates. |

### Are you affected?

```python
res = solve_gertler_karadi()
res.to_frame()["K"].iloc[0]   # -0.2866: a level deviation (K_ss = 5.66), i.e. -5.06%, not -28.66%
```

For the surprise: if your inputs were prices near 95 rather than rates near 5,
your surprises have the wrong sign.

### What to re-run

- **Any Gertler-Karadi response reported in percent**, and **any proxy-SVAR or
  information-shock decomposition** built from price-quoted surprises.
- **In this repository:** curso T02_F (English and Spanish) printed the
  mis-scaled numbers, and the example `hfi_gertler_karadi` used a pure-noise
  instrument quoted as prices.

---

## 2026-09-30 — Blanchard-Quah responses of variables in levels, all versions (documented behaviour, now selectable)

**Changed after 4.3.0; the default is unchanged.** `var.identify.bq_svar`
cumulates the response of every variable, which is right only when every
column of `Y` is differenced, and its result docstring called the output the
"level response of each variable". For the canonical Blanchard-Quah system
(Δlog GNP, unemployment rate in levels), the unemployment response came back as
a running sum that converges to a nonzero constant instead of returning to
zero, and so did its bands; in a known-DGP check it gave +0.687 at h = 40
against a true 0.000. The behaviour was documented, so this is not an error in
the sense of the rules below, but it misled teaching material.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `bq_svar` responses and bands | A column of `Y` in levels | Every column differenced | Pass `cumulate=[indices of the differenced columns]`, for example `cumulate=[0]`; the choice applies to the point estimate and every bootstrap draw. |

### Are you affected?

If any variable in your Blanchard-Quah VAR entered in levels and you read its
`bq_svar` response as a level response, you are affected. Differencing the
response afterwards (`np.diff`) recovers the point estimate but not the bands.

### What to re-run

- **Blanchard-Quah responses of variables in levels**, and their bands.
- **In this repository:** `teaching.bq_canonical.bq_gdp_urate`, curso T02_B
  (which cumulated a second time), `notebooks/course` lessons 02 and 05, the example
  `gali_1999_hours`, and notebook 35 (hours in levels).

---

## 2026-09-30 — continuous Aiyagari: household policy, market clearing and `converged`, versions 3.3.0 to 4.3.0

**Fixed after 4.3.0.** `solve_aiyagari_continuous` stopped the household EGM
silently after 500 iterations, returned `converged=True` unconditionally, and
ignored `solver` and unknown keyword arguments; its wrappers
`AiyagariContinuousModel.solve` and `AiyagariContinuousEquilibrium.solve`
defaulted to a Brent tolerance `xtol=1e-6`, too loose for the documented
clearing tolerance of 1e-4. Where β(1+r*) is close to one, the returned policy
was not the fixed point, and the reported market-clearing error, computed from
that policy, understated the true excess demand: 1.6e-2 true against -2.5e-6
reported in the documented example, 1.67e-2 against 5.95e-4 for
`AiyagariContinuousModel(n_z=3).solve(N_k=150)`, -2.9e-3 against 1.4e-9 at
β = 0.995. The error in r* is small in every case checked, at most 0.0006
percentage points (documented example 0.039421 to 0.039416), with K* moving by
about 4.5e-4 and the Gini by about 5e-4.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `solve_aiyagari_continuous` and the wrappers | n_z = 3 calibrations with ρ_z = 0.9 (the documented example and the transition-document steady state), quarterly β ≥ 0.99, large `a_max`, or the wrappers' old `xtol` | The default calibration (n_z = 5): r within 1e-9; notebook 52 (r to 1e-9, K to 1e-7); notebook 67 (r to 1e-10) | Re-run; check `eq.converged`, `eq.metadata["egm_iterations"]` and `eq.metadata["clearing_ok"]`. |
| `solve_aiyagari_continuous` with `solver="collocation"` or `"fem"` | Always: the result was EGM's | — | It now raises `ValueError`. |
| `solve_continuous_transition` with a permanent shock | Its terminal steady state had the same cap and tolerance | Transitory shocks | Re-run permanent-shock transitions; the terminal steady state now uses the same loop and tolerances. |

### Are you affected?

Re-solve with the fixed release: if `eq.metadata["egm_iterations"]` exceeds
500, the old result was computed with an unconverged household policy.

### What to re-run

- **Steady states computed with n_z = 3, with β ≥ 0.99, or through the wrappers
  with their old default**, and transitions built on them.

---

## 2026-09-30 — `steady()` returned structurally singular steady states silently, versions 2.6.0 to 4.3.0

**Fixed after 4.3.0.** `dsge.steady(solve_algo="block")` is the default path of
`build()`, `build_dynare()` and `load_mod()`. When the static system had no
complete equation-variable matching but a full-system `hybr` solve converged,
it returned that point with no warning. Such a point is one arbitrary member of
a continuum of steady states and depends on the guess: in `{x-1, 2x-2, x+y}`
with `z` in no equation, `z` came back as the guess plus 0.769133 (1.0 gave
1.769133, 7.5 gave 8.269133); a random walk `y' = y + eps` with `c = 2y`
returned `(1.5, 3.0)` from the guess `(4, 3)` and `(0, 0)` from `(1, 0)`.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `steady()`, `build()`, `build_dynare()`, `load_mod()` | Structurally duplicated equations (a subset of equations involving fewer variables than equations), or a variable absent from the steady-state system | Every model with a complete matching: its results are unchanged | These now raise `StructuralSingularityError`; the reported levels were set by the guess. |
| The same, unit-root structure (every over-determined equation reads 0 = 0) | Always | — | Same answer, now with a `StructuralSingularityWarning` saying the level is free. |

Numerically collinear equations whose incidence is complete (`x + y = 2` and
`2x + 2y = 4`) are still not detected: the check is structural.

### Are you affected?

Re-run the steady state with the fixed release. A `StructuralSingularityError`
or `StructuralSingularityWarning` means the old steady state was one arbitrary
point of a continuum. To get the old point, with the warning, pass
`allow_singular=True` to `steady()`, `build()`, `build_dynare()` or `load_mod()`,
or wrap the call in `with allow_structural_singularity():`
(`from puremacro.dsge.steady import allow_structural_singularity`). A built
model keeps the solver's report as `model.steady_state_info`.

### What to re-run

- **Any model that produced a steady state and has either structure above**, and
  everything computed from it.

---

## 2026-09-30 — optimal-policy solvers: silent zero bias, placeholder losses and an inert flag, versions 2.6.0 to 4.3.0

**Fixed after 4.3.0.**

1. **Silent zero bias** (3.1.0 to 4.3.0). `DiscretionaryPolicyResult.stabilization_bias`
   was exactly 0.0, with no warning, whenever the internal `lq_commitment`
   solve raised or the chosen commitment loss was NaN, and `summary()` omitted
   the line. A reported bias of exactly 0.0 may therefore be a failed
   comparison. Values at well-posed calibrations are unchanged
   (1.513166984879534 at notebook 66's baseline).
2. **Meaningless unconditional loss.** For a model whose closed-loop transition
   has a unit root, the commitment `loss` could be a finite number with no
   meaning (4.9e15 for a unit-root cost-push shock). It is now NaN.
3. **`osr` placeholders.** `loss_initial` was set to the `penalty` argument
   (1e6) when the baseline moments failed, with only a UserWarning; `loss_opt`
   was silently the optimiser's last objective (possibly a 1e8 penalty) when
   the model could not be re-solved at the returned coefficients; and
   `variance_reduction_pct` was 0.0 wherever a variance was missing, so
   `summary()` printed "Loss reduction: 0.00%". All three are now NaN with a
   `RuntimeWarning`.
4. **The `timeless` flag did nothing.** `lq_commitment` and `ramsey_model`
   always returned the law of motion shared by the Ramsey plan and the timeless
   rule, responses and `conditional_loss` from the steady state, where the two
   coincide, and `loss` averaged over the stationary distribution. Code that
   passed `timeless=False` expecting another initial condition got the same
   result.
5. **`osr` accuracy.** With SciPy's default Nelder-Mead tolerances, `osr` found
   the optimal-rule allocation only to about 1e-5 when the optimum lay on a
   bound (up to 2.7e-5 at random Clarida-Galí-Gertler calibrations). Losses
   were accurate to 2.5e-10, and coefficients reported to four decimals are
   unaffected.
6. **Citation.** Documentation and docstrings credited the low-β reversal of
   the unconditional ranking to Jensen and McCallum (2002); it is Sauer (2010,
   Prop. 2).

### Are you affected?

```python
res.stabilization_bias == 0.0   # on 4.3.0: possibly a failed comparison, re-run
osr_res.loss_initial == 1e6     # the penalty placeholder, not a loss
```

Any `variance_reduction_pct` of exactly 0.0, and any code that relied on
`timeless=False`, are affected as described.

### What to re-run

- **Stabilization biases of exactly 0.0, `osr` results with placeholder
  losses, and `osr` allocations quoted beyond five significant digits.**
- **In this repository:** notebook 66's numbers are unchanged; only its
  citation for the low-β reversal changes.

---

## 2026-09-30 — the "stochastic steady state" of a pruned solution is the ergodic mean, versions 2.0.0 to 4.3.0

**Clarified after 4.3.0; no number changes.** `stochastic_steady_state()` of
`PrunedDSGESolution` (2.0.0 onwards) and `Order3PrunedSolution` (2.8.0 onwards)
returns E[x] and E[y] of the pruned solution. Its name and docstring ("risk-adjusted
steady state") suggested the zero-shock risky steady state. The ergodic mean is
the precautionary risk term ½ g_σσ σ² plus curvature times dispersion
(½ g_xx vec Ω + ½ g_uu vec σ²Σ_u), a Jensen effect that is present even when
g_σσ = 0. The numbers were always correct as ergodic means.

Reading them as the risky steady state, or as precautionary shifts, is wrong.
In the RBC model of `tests/fixtures/dynare_live/rbc.mod`, capital's ergodic mean
is +0.0785% of its steady state, while the risky steady state and the risk term
are -0.0045% (Dynare's own ghs2 for capital is -9.76e-5). In the 2.0 tour only
1.79e-5 of the 6.77e-4 log-capital shift is the risk term.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `stochastic_steady_state()` | Its output described as a risky steady state or a precautionary effect | Described as the ergodic (unconditional) mean | Use `ergodic_mean()` for the mean, `risky_steady_state()` for the zero-shock fixed point and `risk_decomposition()` to see which term drives the shift. |

### Are you affected?

Only if you interpreted the output as the risky steady state or as a
precautionary effect; `risk_decomposition()` in the fixed release shows how
much of the shift is risk and how much curvature.

### What to re-run

- **Nothing numerically.** Reword any "precautionary" claim based on it.
- **In this repository:** notebook 68 §4, curso T02_B ("Precautionary Premium"
  of capital +0.07711%, whose risky steady state is -0.02807%), the 2.0 tour §7
  and `docs/dsge_build.md` use the label.

---

## 2026-09-30 — trade solver diagnostics and the Hicksian policy closure, versions 4.2.0 and 4.3.0

**Fixed after 4.3.0.** Diagnostics that described a different object from the
one returned, and a closure that the policy searches could not reach.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `trade.solver.solve_cyprus_manifold_step` (alias `solve_cyprus_manifold`, 4.3.0) | `max(abs(dw)) > max_disp`, so the step is clamped: `(residual, converged)` described the unclamped step. Notebook 64 printed 4.4e-16 and `converged=True` for a step whose residual is 3.208, the whole right-hand side. The default `idx_cyp=15` is Colombia, not Cyprus (17) | Unclamped steps, with `idx_cyp` passed explicitly | The fixed release reports the returned step's residual; `return_info=True` gives the unclamped values. Without `eval_cyp_fn` the step is an exact linear solve, uniformly rescaled: a bounded step, not a stabilised solution. |
| `solve_keller_pac` docstrings and `metadata["fold_points"]` (4.3.0) | The docstrings claimed a fold at σ_fold ≈ 0.1238 where Newton diverges; PAC records no fold there and Newton converges in 8 iterations. `fold_points` listed every step with `tau_lambda <= 0`, one entry per remaining step for one factor-price stall | — | Do not cite PAC as traversing a CGE fold. Read the merged events and their `kind`: `factor_price_boundary` is a corner where a factor price goes to zero, not a fold. |
| Hicksian `compute_unilateral_optimal_tariff`, `solve_multilateral_nash_tariffs`, `compute_welfare_payoff_matrix` (4.3.0) | Nonzero baseline foreign balances: only the numeraire closure was reachable, so payoffs depended on which country was listed first (A's EV from a 10% tariff on notebook 63's table: 2.6308% or 2.4952%) | Zero foreign balances | Pass `foreign_saving_units="world_income"` and a baseline solved with it: 2.5404% in either order. |
| Boundary labels (`metadata["boundary"]`, `["best_response_boundaries"]`, 4.3.0) | `tariff_max <= tol`: replies at the ceiling were labelled `"lower"` | Ceilings above `2*tol` | Fixed: the nearer bound is used. |
| `compute_stone_geary_final_demand` (4.2.0 and 4.3.0) | Subsistence spending above the budget: the bundle `c_bar` overspent silently (169% at μ = 0.9 and 5% of benchmark income on a 2x2 table) | Σ_s μ_s θ_s0 below tanh(3)/3 ≈ 0.332 | A `RuntimeWarning` is now emitted; check the budget. |

### Are you affected?

For the Cyprus step: `max(abs(dw)) == max_disp` on a returned step means it was
clamped and its reported residual was not its own. For the Hicksian searches:
re-run with the countries listed in the other order; if a payoff changes, the
closure was order-dependent.

### What to re-run

- **Any clamped Cyprus step reported as converged**, and any fold claim based
  on `fold_points`.
- **Hicksian tariff searches** on tables with nonzero foreign balances.
- **In this repository:** notebooks 63 and 64 (English and Spanish).

---

## 2026-09-23 — third-order theoretical moments, versions 2.8.0 to 4.3.0

**Fixed after 4.3.0** (see the Unreleased section of `CHANGELOG.md`).
`Order3PrunedSolution.theoretical_moments()`, and the `theoretical_moments`
that `stoch_simul(order=3)` returns, reported the **first-order** covariance,
correlations and autocorrelations of a third-order solution. The Skewness and
Kurtosis columns were fixed at 0 and 3 for every variable. The docstring said
the moments matched Dynare's `stoch_simul` at order 3, and
`docs/dsge_higher_order.md` described exact closed-form skewness and kurtosis.
The Mean column was right: it carries the second-order risk correction, which
is the exact mean of the pruned third-order solution under Gaussian shocks.

The error is of order sigma^4 against a variance of order sigma^2, so it is
small when shocks are small and first-order terms dominate. On the RBC model of
`tests/fixtures/dynare_live` the variances of output, consumption and capital
are 0.15%, 0.10% and 0.17% larger than the first-order values. A variable with
no first-order response was reported with zero variance and undefined
autocorrelations. In `correlated_cubic.mod`, `y` has variance 0.788 and lag-1
autocorrelation 0.475; in `quadratic_feedback.mod`, 0.0283 and 0.742.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `theoretical_moments()` of an order-3 solution: covariance, correlation, autocorrelations (2.8.0 to 4.3.0) | Any nonzero second- or third-order coefficient | The Mean column; a solution whose higher-order coefficients are all zero | Re-run with the fixed release, which returns the exact pruned-state-space moments. |
| Its Skewness and Kurtosis columns | Always: they were placeholders | Never | The fixed release reports NaN. Use `ergodic_moments()`, which estimates both from a pruned simulation. |
| `stoch_simul(order=3)` | Its `theoretical_moments` block, as above | Simulated moments (`periods > 0`), impulse responses and decision rules | As above. |

`ergodic_moments()` raised `KeyError: 'Std.Dev'` in the same versions, so it
returned no numbers. Moments computed from `simulate()` are unaffected.

### What to re-run

- **Any variance, standard deviation, correlation or autocorrelation** reported
  from `theoretical_moments()` of a third-order solution, including tables from
  `stoch_simul(order=3)`.
- **Any skewness or kurtosis** taken from those tables: the values were
  placeholders, not estimates.

The fixed release matches Dynare 8's pruned means and variances to machine
precision on six models. It does not match Dynare's order-3 autocorrelations,
which omit a correlation term; a closed form and Dynare's own long simulations
agree with puremacro (see `docs/dsge_higher_order.md` §2.3).

---

## 2026-09-23 — flexible trade equilibrium, versions 4.2.0 and 4.3.0

**Fixed after 4.3.0** (see the Unreleased section of `CHANGELOG.md`).
`solve_flexible_trade_equilibrium` returned the legacy Cobb-Douglas/Leontief
equilibrium, labelled as the flexible result, for every nested-CES,
Stone-Geary (LES), variable-markup and capacity setting. The settings were
stored in `metadata["config"]` but never entered the equilibrium equations:
the default (Newton) dispatch passed them through without reading them, and
only `method="sparse_lu"` or the legacy `"condensed"` path used them, to shape
the Jacobian sparsity pattern. The solve converged, emitted no warning and
reported the requested configuration.

On a 3-country, 3-sector synthetic calibration with a 25% tariff, the v4.2.0
and v4.3.0 releases return a solution vector bit-identical to the default
configuration (maximum absolute difference 0.0, same iterations and residual)
for `rho_va=0.7, sigma_y=0.5`, `subsistence_ratio=0.2`, `variable_markups=True`
and `capacity_margins={(0, 0): 0.3}`. The executed notebook 62 shipped with
those releases reports its homothetic, Stone-Geary and full-flexible (CES,
subsistence and oligopoly markups) solves on the bundled 77x11 table with the
same final residual norm (2.7575e-02); its welfare and markup tables differ only
through post-processing of one and the same equilibrium.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `solve_flexible_trade_equilibrium` (4.2.0, 4.3.0), default `method` | Any non-default technology setting (`rho_va`, `sigma_va`, `sigma_y`, `sigma_inter`, `capacity_margins`), Stone-Geary subsistence (`subsistence_shares`, `subsistence_ratio`, `mu_s`) or `variable_markups=True` | Every flexible setting is at its default: the flexible model then coincides with the legacy model and the result is correct | Re-run with the fixed release and check `result.metadata["flexible_settings_applied"]`. |
| The same function with an explicit `method="quasi_condensed"` (4.2.0, 4.3.0) | The same settings | As above | That path evaluated some technology, capacity and markup settings but ignored Stone-Geary subsistence (`subsistence_ratio=0.2` returned the identical solution); treat its results as affected and re-run with the fixed release. |
| Welfare, EV and markup figures computed from these results | Any affected run | As above | They were derived from the legacy equilibrium; re-run. |

Runs with the default flexible configuration are unaffected, with one
exception: `solve_trade_equilibrium(method="quasi_condensed")` with
`sigma_y > 0` or a flexible `config` now solves the corrected flexible model,
and its results change. The fixed release also changes the default final-demand
sourcing of the flexible solver to fixed coefficients (Armington sourcing
requires an explicit `sigma_trade`).

In the fixed release, models with at most 100 country-sector cells solve the
flexible equations on the quasi-condensed Newton path. Above that size
(including the bundled 77x11 table), or with an explicit `method=` other than
`"quasi_condensed"`, a call with active flexible settings raises `ValueError`
instead of ignoring them. Pass `method="quasi_condensed"` to solve the flexible
equations with a dense Jacobian, or `allow_legacy_fallback=True` to obtain the
legacy equilibrium with a `RuntimeWarning` and
`metadata["flexible_settings_applied"] = False`.

### What to re-run

- **Any counterfactual, welfare or markup number** published from
  `solve_flexible_trade_equilibrium` on 4.2.0 or 4.3.0 with a non-default setting.
- **Notebook 62 as executed in 4.2.0 and 4.3.0**: the `rho_va` sweep, the
  homothetic versus Stone-Geary comparison and the competitive versus
  oligopoly comparison all show the legacy equilibrium.
- **This repository's notebook 62 has been rewritten and re-executed** (English
  and Spanish): it evaluates the flexible cost, demand and pricing blocks at
  given prices and solves one small benchmark with the audited
  consistent-accounting solver; it no longer reports flexible equilibria on the
  bundled table.

---

## 2026-09-23 — collocation curvature ignored or mixed, versions 3.3.0 to 4.3.0

**Fixed after 4.3.0** (see the Unreleased section of `CHANGELOG.md`).
`CollocationProblem(method="bellman")` solved the log-utility growth model
whatever `params["sigma"]` or `params["gamma"]` said. `CollocationProblem` with
`params={"gamma": g}` (no `"sigma"` key) solved the log-utility policy, and
4.3.0 also valued that log policy under CRRA(g) in
`value_coefficients`/`value()`. IFT sensitivities computed on such a problem
(`compute_ift_gradients`, `equilibrium_parameter_jacobian`,
`gmm_objective_and_gradient`) were those of the log-utility problem, with a
zero `gamma` column.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `CollocationProblem(method="bellman")` | `sigma` or `gamma` different from 1 | Log utility (`sigma` or `gamma` equal to 1 or absent) | Re-run the solve with the fixed release. |
| `CollocationProblem` policies and auxiliary values | Curvature spelled `params={"gamma": g}` without a `"sigma"` key, `g != 1` | `method="euler"` with the curvature spelled `"sigma"` | Re-run the solve and the derived values. |
| IFT gradients on those problems | Any affected solve, and the `gamma` column in every case | Log utility with respect to parameters other than `gamma` | Re-run the gradients; the `gamma` column was identically zero. |

In the fixed release one CRRA curvature (`sigma`, else the alias `gamma`,
else 1) enters the Euler residual, the Coleman warm start, the Bellman
objective and the auxiliary value coefficients; log utility is bit-identical
to 4.3.0.

---

## 2026-09-23 — trade inputs read as a different schedule or layout, version 4.3.0

**Fixed after 4.3.0** (see the Unreleased section of `CHANGELOG.md`). Four
trade functions that first shipped in 4.3.0 computed their result for an input
other than the one passed, with no exception and no warning: a continuation
target, a viability schedule, the labels of a FIGARO file's final uses, and
the value-added floor of the MRIO regularization.

| Surface | Affected condition | Unaffected when | Guidance |
|---|---|---|---|
| `solve_keller_pac` (4.3.0) | `tau_target` passed as a NumPy scalar other than `np.float64` (`np.int64`, for example from `np.arange`; `np.float32`; `np.bool_`) or as a 0-d array: only intermediate imports were tariffed, while a Python number also tariffs final-demand imports | `tau_target` is a Python `int`, `float` or `bool`, an `np.float64`, or an array schedule with at least one dimension | Re-run with the fixed release, or pass `float(tau_target)`. |
| `check_hawkins_simon_viability` (4.3.0) | `tau` passed as a Python list (evaluated as the zero-tariff baseline, whatever its values), as a 0-d array (a multiplier of the whole cost matrix), or as an array of shape `(M, M)`, `(M, ns, nc)` or `(ns, nc, ns, nc)` whose diagonal (same country and sector) entries are not 1, which a heuristic could read as rates where `solve_trade_equilibrium` reads multipliers (`np.ones((M, ns, nc)) * 1.10` became a multiplier of 2.10) | `tau` is `None`, a Python or NumPy scalar number, a NumPy `(nc,)` rate vector, or a multiplier array whose diagonal entries are 1 | Re-run the check with the fixed release; a "viable" verdict for such an input may be wrong. |
| `load_figaro` on a harmonized file (4.3.0) | The file lists each country's final uses in an order other than `P3_S14, P3_S15, P3_S13, P51G, P5M`, for example the Eurostat order `P3_S13, P3_S14, P3_S15, P51G, P5M`: final-use columns were labelled by position, so government and household consumption swapped labels | The file follows puremacro's order, or the data are synthetic (`fallback_to_synthetic=True`) | Re-load with the fixed release, which reads final uses by label. |
| `regularize_mrio_table`, and `package_mrio_to_calibration_result` with the loaders that call it (`load_figaro`, `load_exiobase`, `load_wiod`, `load_eora`, `load_oecd_icio_granular`) with `regularize=True` (4.3.0) | A node whose value added is floored has output below 2 in the table's units (2 M USD for the loaders): the floor `max(1e-3 Y, 1)` set its value added to 1, above output when output is below 1, and debited the difference from its taxes less subsidies | Every floored node has output of at least 2, or `regularize=False` | Re-run the calibration with the fixed release, whose floor is `max(1e-3 Y, min(1, 0.5 Y))` and which records every floored node. |

Measured on the 4.3.0 release tag and on the fixed release:

- `solve_keller_pac` on a 2-country, 2-sector test table: `tau_target=np.int64(1)`
  returned `x_sol[:4]` = [0.0965, -0.0050, 0.1579, 0.1302] with
  `converged=True`; `tau_target=1.0` returns [0.2008, 0.0809, 0.3446, 0.3215],
  which the fixed release also returns for `np.int64(1)`.
- `check_hawkins_simon_viability` on the same table: `[0.5, 0.5]` returned
  rho 0.3379685 (the baseline) instead of 0.3762996; `np.array(0.1)` returned
  0.0337969 instead of 0.3456204; `np.ones((M, ns, nc)) * 1.10` returned
  0.7097339, the radius for a multiplier of 2.10, instead of 0.3717654.
- `load_figaro` on a 2-country, 2-sector file in the Eurostat order: the
  final-use shares (`theta`) differed from those of the same data in
  puremacro's order by up to 0.23, and a 10% tariff equilibrium by up to
  9.7e-4 (largest unknown 12.8). The fixed release gives identical
  calibrations for both orders.
- `load_oecd_icio_granular` on the OECD ICIO 2019 file (`2019_SML.csv`):
  4.3.0 raised 41 value-added entries by 9.84 M USD in total and left 10
  nodes with value added above output (taxes less subsidies down to -467% of
  output); the fixed release raises 36 entries by 2.74 M USD, none above
  output.

### What to re-run

- **Continuation paths** from `solve_keller_pac` whose target came from a
  NumPy array or loop variable (for example `for rate in np.arange(1, 4)`).
- **Viability screens** of list schedules, 0-d arrays or multiplier arrays
  with non-unit diagonal entries.
- **Calibrations of harmonized FIGARO files** whose final-use columns were not
  in puremacro's order, including any category-level result.
- **Calibrations of real tables with small nodes** built by the 4.3.0 loaders,
  and every counterfactual solved on them.

---

## 2026-09-22 — provenance of the bundled 77x11 OECD table, all versions

**Not a code defect; a data-provenance finding. Documented and guarded after
4.3.0.**

`puremacro/trade/_datafiles/icio_77c_11s.npz` is a bit-exact copy of the MATLAB
array `data_77c_11s.mat` that the original sectoral-misallocation pipeline built
from a `data_2020_SML.csv` export of the OECD ICIO tables. That export is damaged:
tokens with three or four decimals lost their decimal point and tokens below
0.001 became zero (its MD5 `d1b887aaafa54ab3f28fde78fcd21cdf` is registered as
corrupted by the research workspace that produced the paper's revised results).

Measured against the clean OECD 2020 release (MD5 `d3e0f4979d85d6c0bb7cf4c43e324287`):

| Quantity | Bundled table / corrupted export | Clean OECD 2020 release |
|---|---:|---:|
| World value added, 77x11 aggregation (USD million) | 7.05e11 | 7.97e7 |
| Largest intermediate cell, 77x11 aggregation (USD million) | 5.83e10 | 6.79e6 |
| Intermediate cells of the 77x11 aggregation that are equal | 35.5% | reference |
| Value-added cells of the 77x11 aggregation that are equal | 1 of 847 | reference |
| Positive native 45-sector intermediate cells set to zero in the export (every one below 0.001) | 2.10 million of 7.82 million | none |
| Positive native intermediate cells multiplied by 1e3 or 1e4 in the export | 0.70 million | none |

| Surface | Affected condition | Guidance |
|---|---|---|
| `load_icio_data(source="legacy")`, the bundled 77x11 calibration, notebooks 61-65, `trade_reference_solutions.npz`, `trade_results_workbook.npz` | Any economic magnitude, share or elasticity computed from the bundled table | Treat these as software regression fixtures: the parity suites compare puremacro with MATLAB solutions of the **same** table and remain internally consistent. Do not report their outputs as OECD-based estimates. |
| `load_raw_45sector_icio()` without a `path` (4.3.0 and earlier) | The default search list started with `computation/7_TIO_77c_vf/data_2020_SML.csv`, the corrupted export, and read it without a checksum | The fixed release searches the clean `ICIOextended/2020_SML.csv` and `2019_SML.csv` first, checks the MD5 of the resolved file, refuses the corrupted export with `MRIOIntegrityError` (a `ValueError`) naming the clean-release checksums, and warns on unknown digests. Pass the clean `2019_SML.csv` (MD5 `28cba31491177955445051d459053744`) or `2020_SML.csv` explicitly. |
| Native readers (`load_oecd_icio_granular`, `puremacro.trade.mrio.read_oecd_native`) | None: they read the OECD 2023 regular CSV by labels | Use these for new empirical work; they record the source checksum in metadata. |

The bundled table is retained unchanged so that the historical MATLAB parity
contract stays external. Replacing it with a clean aggregation would silently
change every fixture and is a decision for the author, not for a release. See
[native MRIO tables](trade_mrio.md) for the measurements.

---

## 2026-09-20 — structural models, affected APIs in 4.2.0 and earlier

**Fixed or explicitly restricted in 4.3.0.** The review found incorrect numerical
outputs and success flags in VFI, DSGE/Dynare and trade. Earlier introduction
versions differ by API; the findings were reproduced on the 4.2.0 tree.

| Surface | Affected condition | Rerun guidance |
|---|---|---|
| Third-order DSGE decision rules | Nonlinear stochastic models with timing/Hessian/variance contributions to risk slopes | Regenerate third-order rules and derived simulations, moments or likelihoods. First- and second-order rules do not contain these risk slopes. Five live Dynare models provide bounded order-two/three reference checks. |
| Dynare parity | Order-two comparisons omitted cross/shock tensors or accepted missing/malformed references and moments | Re-run comparisons with the full requested tensors and matching labels. A prior 100% score does not establish complete parity. |
| VFI projections and auxiliary values | Optimizer termination at a nonzero Euler residual; auxiliary value recovery with non-unit productivity or non-log utility | Require the final residual to meet the requested tolerance and regenerate affected values. Convergence at fitting nodes still does not certify off-grid accuracy. |
| Deep Macro gradients | A second network evaluation overwrote the current-state activation cache before the detached-target weight update | Re-run training and validate held-out residuals; a small historical training loss is insufficient. |
| Markov-switching DSGE | Failed coupled solves returned finite impacts, stability or moments | Re-solve and require convergence before interpreting these diagnostics; failed states now report unavailable outputs. |
| Trade flows and fiscal accounts | CES solutions postprocessed with Leontief flows, incomplete fiscal schedules or inconsistent producer/purchaser valuation | Re-run affected counterfactuals. Select `accounting="consistent"` for the derived producer/purchaser model; the legacy default remains for compatibility. |

Tariff policy now selects expenditure-function consumption welfare explicitly
with `metric="hicksian_ev"`. Historical `ev`/`equivalent_variation`/`consumption`
policy aliases remain utility proxies. The new metric uses a fixed comparison
baseline and audited solver recovery; it does not certify a global Nash theorem.
The draft TOT/Alloc/TariffRec and theorem endpoints are unavailable rather than
publishing unsupported conclusions.

See [supported scope and reference evidence](STRUCTURAL_VALIDATION_STATUS.md),
[trade accounting](trade_accounting.md), [consumption welfare](trade_welfare.md)
and [tariff policy](trade_policy.md). Full-size native GE, flexible/GPU welfare
and general higher-order Dynare parity remain outside the validated scope.

---

## 2026-09-16 — three estimators, versions up to and including 4.0.1

**Fixed in 4.0.2.** Found while a course site re-derived puremacro's numbers
independently: an HP filter applied in the time domain, a matrix-form 2SLS, and a
Markov chain solved in 80-digit arithmetic.

| Estimator | What was wrong | Unaffected when | Direction of the error |
|---|---|---|---|
| `theoretical_moments(hp_filter=...)` via `dsge._moments.spectral_moments` (2.9.0–4.0.1) | Weighted the spectral density by the HP cycle filter's transfer function `H(w)` instead of its square | Unfiltered or band-pass theoretical moments, and simulated moments from `stoch_simul(hp_filter=...)`, which filter in the time domain | Filtered variances and autocovariances **overstated**: +22% variance (+10.5% s.d.) for an AR(1) with rho = 0.9 at lambda = 1600; relative volatilities and correlations shifted with them |
| `lp.lp_iv` (0.92.0–4.0.1), `lp.lp_state_dep_iv` (2.0.0–4.0.1), `lp.la_lp_iv` (4.0.0–4.0.1) | The second-stage variance used the residual `y - X_hat beta` (fitted regressor) instead of `y - X beta` | Never for `se`, `lo`, `hi`; point estimates, first-stage F, the Montiel Olea–Pflueger F and the Anderson–Rubin sets were right | **Either way**, by `2 beta cov(u, v) + beta^2 var(v)`: nominal 90% bands covered the truth 100% of the time with `beta = 1`, `cov(u, v) = 0.8`, and 69% with `beta = -1` |
| `vfi.markov_stationary` (0.92.0–4.0.1), `dsge.markov_switching.markov_stationary` (3.1.0–4.0.1) | Took the eigenvector of `P'` for the eigenvalue nearest 1 | States communicate through probabilities well above ~1e-12, as on ordinary Tauchen and Rouwenhorst grids | An arbitrary mixture when several eigenvalues round to 1: **6e-5 off** on a two-state chain with switching probability 1e-13, a **probability of -0.23** on `tauchen(7, 0.995, 0.1, m=5)`; the `dsge` copy fell back to a uniform distribution without a warning |

### What to re-run

- **Any HP-filtered theoretical moment table**, e.g. a Kydland–Prescott
  comparison of a DSGE model with data. Simulated moments were right, so a
  table that mixed the two compared unlike quantities.
- **Any `lp_iv`, `la_lp_iv` or `lp_state_dep_iv` band or t-statistic.** Whether
  it was too wide or too narrow depends on the sign of `beta cov(u, v)`, so it
  cannot be corrected by rescaling; re-run.
- **Stationary distributions of very persistent discretizations** (Tauchen with
  rho at or above about 0.95 and a wide grid) or of Markov-switching models with
  near-absorbing regimes.

---

## 2026-09-03 — twelve more, versions up to and including 1.9.0

**Fixed in 1.10.0.** A follow-up audit asked, of
every estimator fixture in the package, *what condition does this fixture
remove?* — the question all seven defects above turned on. It found twelve more,
each reproduced against a known truth before being touched.

| Estimator | What was wrong | Unaffected when | Direction of the error |
|---|---|---|---|
| `garch.garch11_fit` (< 2.3.1) | L-BFGS-B on the raw series stopped after a handful of iterations on non-unit-scale data (decimal or percent returns) up to 31 log-likelihood units below the maximum, with `converged=True`. | The series has unit variance | Persistence (`beta`) biased, sometimes the starting values returned |
| `inference.weak_iv.cragg_donald_f` (< 2.3.1) | Divided the minimum eigenvalue by the number of endogenous regressors instead of the number of instruments and used a scalar residual variance. | Exactly identified (`l = k = 1`) | Over-identified statistics inflated by a factor `l`: weak instruments read as strong against Stock-Yogo tables |
| `var.identify.proxy` (< 2.3.1) | The wild bootstrap re-signed the residuals but not the instrument, so each draw's impact vector had a random sign. | `n_boot = 0` | 90% bands centred on zero and excluding the point estimate |
| `dsge.build_dynare` / `load_mod` (< 2.3.1) | Leads of state variables were dropped from the Klein system and the control rows of the decision rules were shifted one period; second-order terms inherited both. | Models with no lead on any lagged variable (rare: every RBC Euler equation has one) | Wrong first-order rules, theoretical moments and IRFs; Gertler-Karadi and Smets-Wouters affected |
| `nowcast.nowcast_gdp` (< 2.3.1) | The EM imputation reconstructed missing entries with `sqrt(T) U V'`, dropping the singular values. | No missing entries in the target quarter | Ragged-edge factors shrunk toward zero; nowcast RMSE roughly 2.5x too high |
| `cointegration_modern.dols` | Omitted the contemporaneous `dX_t` term, which is the one that removes the bias. The estimator was OLS with extra regressors. | `dX_t` is uncorrelated with `u_t` — no contemporaneous endogeneity | Bias tracked plain OLS at every sample size: **+0.036 vs OLS's +0.030** at T=100, where correct DOLS is −0.0005 |
| `cointegration_modern.dols` | Kept a row whose missing `dX` lag was **zero-filled**, fabricating a regressor value | `lags = 0` | One extra observation, built from a value that was never measured |
| `cointegration_modern.fm_ols` | Built the Phillips–Hansen correction from `Lambda` alone, dropping the lag-0 `Sigma` block | `Omega` is proportional to `Sigma` — serially uncorrelated innovations | Over-corrected. **RMSE worse than plain OLS** at T=200, 800 and 3200 |
| `var.irf.gfevd` → `connectedness.spillover_index` | Divided by an absolute `1e-12` floor, breaking a scale invariance the estimand has by construction | No residual variance falls near the floor — i.e. the units happen to be large enough | Total connectedness moved **13.43 → 39.48** on one fixed VAR when a single variable was rescaled |
| `dsge.gensys` | `Impact` dropped the `Pi N` term: solved as though expectation errors did not respond to shocks | `Pi N = 0` — no forward-looking behaviour left | On `y_t = a E_t y_{t+1} + eps_t` (truth `y_t = eps_t`) it returned **`[0, -2]`**: the variable did not respond to its own shock |
| `dsge.gensys` | Raised on any model with **no unstable roots** (`Z2` a zero-column slice); uniqueness test was vacuous | The model has at least one unstable root | `x_t = rho x_{t-1} + eps` could not be solved at all |
| `dsge.fertility_adj_costs` `irf`/`fevd` | Advanced the state before applying `F`, but controls read the *lagged* state | Horizon 0 only | Every control IRF **one period early**: reported `y(h)` was the correct `y(h+1)` |
| `did.sun_abraham` | Event-study `lo`/`hi` averaged the per-cohort interval edges instead of using the aggregated `se` | One cohort per event time | Bands **`sqrt(K)` too wide** — measured 1.73 at K=3, 1.37 at K=2 — and inconsistent with the `se` in the same row |
| `inference.weak_iv.kleibergen_paap_f` | HC0 sandwich divided by `n` twice while the bread inverted the raw `Z'Z`, which already carries both factors | Never — every call was affected | Returned **exactly `n^2` times** the correct statistic: 235,470 against a truth of 5.89 at n=200. Stock-Yogo thresholds are near 10, so it reported "overwhelmingly strong" for every dataset and **could never diagnose a weak instrument** |
| `inference.weak_iv.kleibergen_paap_f` | Influence function used `kron(Z_t, V_t)` where the column-major `vec` and the bread both require `kron(V_t, Z_t)` | `k = 1` (one endogenous regressor) — then the two coincide | Wrong robust variance for two or more endogenous regressors |
| `inference.weak_iv.anderson_rubin_band` | Compared an F statistic against `chi2.ppf(ci, df)`, but it is `df*F` that is `chi2(df)` | `df = 1` — the default, and the only value anything exercised | Cutoff `df` times too large. Nominal 90% band had **100.0% coverage** at df=2 and df=4 |
| `lp.panel_lp_dk` (via `_focal_dk_se`) | Bartlett bandwidth derived from the panel row count `N*T`, but the kernel runs on the length-`T` cross-sectional sum | `N = 1` | Bandwidth inflated by `N^(2/9)`: **7 -> 10 -> 13** at fixed T=100 as N went 10 -> 50 -> 200. Adding countries changed the assumed autocorrelation |
| `lp.mean_group_panel_lp` | Did not sort its input, while `lp_hac` builds lags with positional `.shift()` | The caller already passed a sorted frame | On a panel whose true h=1 response is 0.8: **0.765 sorted, 0.055 with the same rows shuffled**. Silent, and attenuated toward zero |
| 25 p-value sites across 16 modules | `1 - cdf` instead of the survival function | The p-value is above ~1e-16 | Returned **exactly 0.0** in the tail. Two-sided normal at \|z\| = 9 is 2.3e-19; chi-square 200 on 5 df is 2.8e-41 |

### What to re-run

- **Any `dols` or `fm_ols` estimate.** Both were systematically biased on the
  data they exist for. `fm_ols` was further from the truth than plain OLS.
- **Any Diebold–Yilmaz connectedness index** computed with the default
  `identification="gfevd"` on data whose residual variances are small in the
  units you used.
- **Any `gensys` IRF.** The impact matrix was wrong for every model with
  forward-looking behaviour, which is every model gensys is for.
- **Any fertility-DSGE IRF or FEVD**, including the figures under
  `docs/research/fertility_bk_diagnosis/`. Shift the control paths back one
  period, or re-run.
- **Any Sun–Abraham event-study band** at an event time with more than one
  contributing cohort. The point estimates and `se` stand; the intervals were
  too wide. **Correction (2026-09-30):** the `se` did not stand; it treated
  cohorts that share control units as independent (see the 2026-09-30 entry
  on Sun-Abraham standard errors). The point estimates do.
- **Any p-value reported as exactly 0.0.** It was never zero.

### A note on the tests

Four separate artifacts asserted the `gensys` defect rather than catching it:
three tests in `tests/test_dsge_gensys_coverage.py`, and the validation case
`dsge.gensys_shock_impact_identity`, whose own citation stated the model
**without** its `Pi eta_t` term — correct algebra from a truncated premise. The
`sun_abraham` band contradicted the standard error printed beside it in the same
row, so that one needed no external reference at all. And
`FertilitySolution`'s docstring described `F` as acting on the current state
while the solver builds it against the lagged one; the IRF loop was written
against the docstring.

---

## 2026-09-02 — seven estimators, versions 0.92.0 through 1.8.0

**Fixed in 1.9.0.** Seven public estimators returned wrong numbers in
every release from 0.92.0 to 1.8.0 inclusive. All seven failures share a
shape: the wrong answer was *internally consistent*, so no invariant the
package checked could detect it, and in every case the test fixture
satisfied the exact condition under which the bug disappears.

### Are you affected?

```python
import puremacro
puremacro.__version__          # < "1.9.0" → read the table
```

If you have **published** a number from any of the seven below on a
version before 1.9.0, re-run it. If you have only run the estimator on a
fixture matching its "unaffected when" column, you are fine.

| Estimator | What was wrong | Unaffected when | Direction of the error |
|---|---|---|---|
| `var.identify.proxy_svar` | Impact vector returned in the `Sigma` metric instead of the `Sigma^-1` one — proportional to `Sigma b_1`, not `b_1`. The identified shock was a mixture of all structural shocks. | `Sigma` is proportional to the identity (i.i.d. residuals) — then the error is exactly zero | Grows with the off-diagonal structure of `Sigma`. 31% on one element of a 3-variable DGP, with the wrong relative sign pattern |
| `var.panel` | Imports the same `_proxy_impact_factory` | as above | as above |
| `inference.swamy_test` | Quadratic form centred on the arithmetic mean instead of the precision-weighted `beta_bar_W` | Every unit is estimated with the **same** precision | Over-rejects slope homogeneity, never under-rejects. Size at a nominal 5%: 0.050 equal, 0.078 at a 2× spread, **0.975** at 0.1 vs 3.0 |
| `garch.dcc_fit` | Raw returns standardised by a volatility fitted on the demeaned ones, so `Qbar` estimated `mu_i · mu_j` | `mean="zero"` — **the default**, and bit-identical to before on already-demeaned input. Only `mean="constant"` is affected | Correlations pulled toward `m²/(m²+1)`, `m = mu/sd`. True 0 reported as **+0.94** at mean 5, sd 1 |
| `state_space.simulation_smoother` | Model intercepts left in the second Durbin–Koopman pass, so `b` was added back a second time | `c` and `d` are both zero — every fixture in the suite, and bit-identical there | Every draw offset by the whole intercept. `d = 5` put the draws exactly −5.0 from the posterior mean, against a Monte Carlo SE of 0.012 |
| `var.wild_bootstrap_var` | Failed draws written into the percentile stack as the point estimate, with no counter and no warning | No draw failed | Bands too **narrow**, monotonically in the failure fraction `f`; zero width once `f ≥ 1−2a`. Worst exactly when `proxy_svar`'s `impact_fn` raises on a weak instrument — the band tightened when it should have widened |
| `var.identify.rigobon_svar` | Bootstrap paired reshuffled residual blocks with calendar-order regime labels, destroying the identification in every draw | Point estimate only; the **band** is what was wrong | Bands about **8× too wide** on a DGP with true variance ratio 3.0 (draws averaged 1.14, never exceeded 1.50 in 500) |
| `var.estimate_var` | Non-finite input accepted; returned all-NaN coefficients without raising | Input is finite | Not a wrong number but a silent one: `Sigma`, residuals, and every IRF, FEVD, historical decomposition and band built on the fit were all-NaN and perfectly well-formed. Now a named `LinAlgError` |

Full derivations, the measured magnitudes, and why each fixture could not
reach its bug are in [`CHANGELOG.md`](https://github.com/jalonso1979/puremacro/blob/main/CHANGELOG.md)
under 1.9.0, "Fixed — affects results published in every release from
0.92.0 to 1.8.0".

### What to re-run

- **Any published proxy-SVAR impulse response** from `proxy_svar` or
  `var.panel`. Both the point estimate and the band change.
- **Any Rigobon band.** The point estimate stands; the band does not.
- **Any `swamy_test` rejection on a panel with unequal per-unit
  precision** — short samples mixed with long, small countries with
  large. This is the normal case, not the exotic one.
- **Any `dcc_fit(mean="constant")` correlation.** The default
  `mean="zero"` path needs nothing.
- **Any `simulation_smoother` draw from a model with a non-zero state
  drift or measurement intercept.**
- **Any `wild_bootstrap_var` band whose run reported bootstrap
  failures** — which, before 1.9.0, it did not report. If the estimator
  was `proxy_svar` with a weak instrument, assume the band was too
  narrow.

### How far the fix travelled, as of 1.9.0

Stated here rather than left for a user to discover:

- **`matlab/` — removed in 4.0.0.** The MATLAB companion toolbox was a
  separate implementation, so a Python fix never reached it. Two estimators
  were ported by hand before removal: `+puremacro/+var/proxy.m` carried the
  identical proxy-SVAR metric error and was corrected, and
  `+puremacro/+var/estimate.m` was made to raise on non-finite input. The
  other five estimators in the table above were **never audited there**.
  Because an unaudited parallel implementation of a corrected estimator is a
  standing hazard, and because the toolbox required proprietary software that
  puremacro otherwise does not, it was deleted in 4.0.0 rather than carried
  forward. **If you ever ran that toolbox, treat its output as unverified and
  re-run it with the Python package. Any proxy-SVAR impulse response it
  produced before 2026-09-02 is wrong.** The code remains in git history at
  tag `v3.4.0` if you need to consult it.
- **This repository's own notebooks have been re-executed.**
  `notebooks/14_tax_multiplier_three_ways`,
  `notebooks/17_identification_spec_curve`, their `_es` twins,
  `notebooks/course/06_lp_narrativa_es` and the `playground/` build all
  display post-fix numbers as of 1.9.0. `notebooks/08_garch_volatility`
  needed no change: it calls `dcc_fit(panel)`, and the default
  `mean="zero"` path is bit-identical.
- **Your notebooks have not.** A committed `.ipynb` stores the outputs
  of the run that made it. Any cell of yours displaying a result from an
  estimator above, executed before 1.9.0, still shows the pre-fix number
  until you re-execute it.

---

## How advisories are decided

An advisory is issued when **all** of the following hold:

1. A released version returned a numerically wrong result from a public
   estimator, or a result whose stated coverage it did not have.
2. The failure was silent — no exception, no warning, no obviously
   malformed output.
3. A user could plausibly have published the number.

A bug that raises, a bug in an unreleased path, and a bug in a private
helper with no public consequence are CHANGELOG entries, not advisories.

The rule this follows is the one in
[`CONTRIBUTING.md`](https://github.com/jalonso1979/puremacro/blob/main/CONTRIBUTING.md):
the package does not substitute a plausible value for a missing one, and
it does not stay quiet about a number it got wrong.
