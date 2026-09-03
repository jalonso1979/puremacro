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

## 2026-09-03 — eight more, versions up to and including 1.9.0

**Fixed in the next release; not yet published.** A follow-up audit asked, of
every estimator fixture in the package, *what condition does this fixture
remove?* — the question all seven defects above turned on. It found eight more,
each reproduced against a known truth before being touched.

| Estimator | What was wrong | Unaffected when | Direction of the error |
|---|---|---|---|
| `cointegration_modern.dols` | Omitted the contemporaneous `dX_t` term, which is the one that removes the bias. The estimator was OLS with extra regressors. | `dX_t` is uncorrelated with `u_t` — no contemporaneous endogeneity | Bias tracked plain OLS at every sample size: **+0.036 vs OLS's +0.030** at T=100, where correct DOLS is −0.0005 |
| `cointegration_modern.dols` | Kept a row whose missing `dX` lag was **zero-filled**, fabricating a regressor value | `lags = 0` | One extra observation, built from a value that was never measured |
| `cointegration_modern.fm_ols` | Built the Phillips–Hansen correction from `Lambda` alone, dropping the lag-0 `Sigma` block | `Omega` is proportional to `Sigma` — serially uncorrelated innovations | Over-corrected. **RMSE worse than plain OLS** at T=200, 800 and 3200 |
| `var.irf.gfevd` → `connectedness.spillover_index` | Divided by an absolute `1e-12` floor, breaking a scale invariance the estimand has by construction | No residual variance falls near the floor — i.e. the units happen to be large enough | Total connectedness moved **13.43 → 39.48** on one fixed VAR when a single variable was rescaled |
| `dsge.gensys` | `Impact` dropped the `Pi N` term: solved as though expectation errors did not respond to shocks | `Pi N = 0` — no forward-looking behaviour left | On `y_t = a E_t y_{t+1} + eps_t` (truth `y_t = eps_t`) it returned **`[0, -2]`**: the variable did not respond to its own shock |
| `dsge.gensys` | Raised on any model with **no unstable roots** (`Z2` a zero-column slice); uniqueness test was vacuous | The model has at least one unstable root | `x_t = rho x_{t-1} + eps` could not be solved at all |
| `dsge.fertility_adj_costs` `irf`/`fevd` | Advanced the state before applying `F`, but controls read the *lagged* state | Horizon 0 only | Every control IRF **one period early**: reported `y(h)` was the correct `y(h+1)` |
| `did.sun_abraham` | Event-study `lo`/`hi` averaged the per-cohort interval edges instead of using the aggregated `se` | One cohort per event time | Bands **`sqrt(K)` too wide** — measured 1.73 at K=3, 1.37 at K=2 — and inconsistent with the `se` in the same row |
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

- **`matlab/` — partly.** The MATLAB companion toolbox is a separate
  implementation, so a Python fix does not reach it. Two were ported by
  hand: `+puremacro/+var/proxy.m` carried the identical proxy-SVAR
  metric error and is corrected, and `+puremacro/+var/estimate.m` now
  raises on non-finite input. **Any proxy-SVAR impulse response that
  toolbox produced before 2026-09-02 is wrong and should be re-run.**
  The other five estimators in the table above have **not** been audited
  there; where the toolbox implements them, assume the same defects
  until checked. See
  [`matlab/README.md`](https://github.com/jalonso1979/puremacro/blob/main/matlab/README.md).
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
