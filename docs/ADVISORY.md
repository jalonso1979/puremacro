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
  too wide.
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
