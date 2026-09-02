# GDP Nowcasting

GDP is quarterly and it is late — four to eight weeks after the quarter ends,
depending on the office. Until it lands, everything known about the quarter in
progress is monthly: industrial production, payrolls, retail sales, surveys,
each on its own release calendar. A nowcast answers "what is this quarter's
growth rate *today*", and the two problems standing in the way are structural
rather than statistical.

**Mixed frequency.** The target is observed four times a year, the predictors
twelve. Any estimator has to commit to what a quarterly value of a monthly
series *is* — an average, a sum, or a latent monthly path constrained to
average out correctly.

**The ragged edge.** On any given day the panel is a rectangle with a torn
bottom. Surveys for month *m* are published days after it closes; industrial
production takes weeks; some series are two months behind. Listwise deletion
throws away precisely the rows the nowcast is about.

`puremacro.nowcast` ships three estimators that make different bargains with
those two problems, plus combination and scoring helpers for judging the
result.

| function | model | you hand it | ragged edge handled by |
|---|---|---|---|
| `nowcast_gdp` | EM-PCA factors + quarterly bridge regression | a monthly frame **and** a quarterly GDP series | iterative PCA imputation |
| `kalman_dfm` | Doz-Giannone-Reichlin (2011) two-step DFM | one frame, one frequency | exact Kalman smoother |
| `mf_var` | Mariano-Murasawa (2003) mixed-frequency VAR | one monthly frame with the quarterly column stamped at quarter ends | exact Kalman smoother |

Nothing on this page touches the network. Every function takes a frame you
already have.

## `nowcast_gdp` — factors plus a bridge

The workhorse, and the one that returns a number you can quote. Four steps,
all inside one call:

1. Standardise the monthly panel, interpolate the holes as a starting value,
   then run PCA and re-impute the missing entries from the rank-*k* fit until
   the factors stop moving (`max_em_iter=50`, `em_tol=1e-4`).
2. Average the monthly factors within each calendar quarter.
3. OLS-regress historical quarterly GDP on those quarterly factor averages —
   the *bridge*.
4. Apply the bridge coefficients to the target quarter's factor average.

```python
from puremacro.nowcast import nowcast_gdp

res = nowcast_gdp(monthly, gdp, n_factors=2)
res.nowcast          # float, in the units of `gdp`
res.target_quarter   # e.g. '2024Q4'
res.model_r2         # bridge R²
print(res.summary())
```

### The input frame

Two objects, two different index contracts, and getting either wrong fails
quietly rather than loudly.

| argument | type | index it needs | missingness |
|---|---|---|---|
| `monthly_data` | `DataFrame` (T_months × N) | `DatetimeIndex`, monthly | NaN anywhere; NaN in the last rows *is the point* |
| `quarterly_gdp` | `Series` | **strings** `'2024Q3'` | none — the historical GDP series must be complete over the quarters it covers |

```python
monthly.index                      # DatetimeIndex, freq 'MS'
gdp.index = gdp.index.to_period("Q").astype(str)   # -> '2015Q1', '2015Q2', …
```

The quarter labels are how the bridge lines the two frequencies up: the
monthly factors are resampled with `resample("QE")` and stamped
`to_period("Q").astype(str)`, then matched against `quarterly_gdp.index` by
label. Three consequences worth knowing before the first call:

- **A `PeriodIndex` on `quarterly_gdp` raises.** The label list is built with
  `.astype(str)` but the row selection is `quarterly_gdp.index.isin(...)`
  against the untouched index, so a `PeriodIndex` matches nothing, the bridge
  design matrix comes out empty, and NumPy raises
  `LinAlgError: Incompatible dimensions`. Cast to `str` yourself.
- **A `DatetimeIndex` on `quarterly_gdp` does *not* raise — it silently falls
  back to positional alignment** of the first `min(len(factors_q), len(gdp))`
  rows, which is what happens whenever fewer than four labels match. When both
  series start in the same quarter the answer happens to be right, to the last
  digit. When the GDP series starts four quarters later than the monthly
  panel, the bridge is fitted on misaligned pairs: on a synthetic
  ten-indicator, 120-month panel where string labels give R² = 0.99 and a
  nowcast of 2.98, the same data with a `DatetimeIndex` gives R² = 0.33 and
  3.36. **Check `res.model_r2`** — a collapse there is the symptom.
- **A non-`DatetimeIndex` `monthly_data` also works, and also can't match.**
  The quarters are then labelled `Q1, Q2, …` positionally — `target_quarter`
  comes back as `'Q40'`, not a date — no label ever matches a GDP quarter, and
  the same positional fallback takes over.

### Where the frame should end

End the monthly frame at the last month with at least one published
observation, and leave that row ragged. Do **not** reindex forward to the end
of the quarter with all-NaN rows.

An all-NaN row has nothing to anchor it, so the EM fit drags it to the panel
mean and its factor comes back at zero. Measured on a one-factor synthetic
panel truncated after the first month of the target quarter, the published
month's factor was 0.161 and the two padded months came back at 3.8e-05 each —
so the quarter's factor average was 0.054, a third of the signal, and the
nowcast was shrunk toward the historical mean rather than sharpened. Over 200
replications, RMSE against the realised GDP value was **1.24 truncated against
1.56 padded**. A fully-NaN last row also empties `news_decomposition`, since
that block is built from the last row alone.

The cost of truncating is real but smaller: the target quarter's factor is
then the mean of however many months are present — one or two, not three —
while the bridge slope was estimated on three-month means.

### What comes back

`NowcastResult` is a frozen dataclass.

| field | type | contents |
|---|---|---|
| `nowcast` | `float` | the point nowcast, **in whatever units `quarterly_gdp` was in** |
| `target_quarter` | `str` | label only — see below |
| `factors` | `DataFrame` (T_months × K) | monthly factors, columns `Factor_1 … Factor_K`, indexed like `monthly_data` |
| `loadings` | `DataFrame` (N × K) | Λ, indexed by series name |
| `news_decomposition` | `DataFrame` | `series`, `actual`, `forecast`, `surprise`, `weight`, `contribution` |
| `model_r2` | `float` | bridge R², clipped into [0, 1] |

Two labels that are not what they appear:

- **`target_quarter` renames the answer, it does not select it.** The nowcast
  is always built from `df_F_q.iloc[-1]` — the last quarter present in the
  resampled factor frame. Passing `target_quarter="1999Q1"` changes the string
  and leaves `nowcast` bit-for-bit unchanged. To nowcast a different quarter,
  truncate `monthly_data`.
- **"annualized" in `summary()` is a string, not a transformation.** No
  annualisation happens anywhere in the module; the nowcast inherits the units
  of `quarterly_gdp`. Hand it annualised rates if you want an annualised
  answer.

`model_r2` is clipped, so a bridge that fits worse than the sample mean
reports `0.0` rather than a negative number. If `quarterly_gdp` has zero
variance the field is set to the literal `0.85`; that value means "no
variance to explain", not a fit.

`p_factor_lags` is accepted and never used — it appears in the signature and
the docstring and nowhere in the body, because this estimator has no factor
transition equation. Use `kalman_dfm` if you want one.

## `kalman_dfm` — two-step DFM with an exact smoother

The Doz-Giannone-Reichlin (2011) estimator: PCA for a starting value, a VAR(p)
on those factors for the transition, OLS loadings for the observation
equation, then one pass of `puremacro.state_space.kalman_smoother` over the
full ragged panel. The smoother is what distinguishes it from static PCA — it
conditions the current month on whatever has actually been published and fills
the rest.

```python
from puremacro.nowcast import kalman_dfm

out = kalman_dfm(monthly, n_factors=2, p=1, standardize=True)
out["factors_df"].tail(3)     # smoothed monthly factors, columns F1, F2
out["X_filled_df"].tail(3)    # the ragged rows completed — this is the nowcast
```

`n_factors` is keyword-only and has no default; `p=1`, `standardize=True` and
`diffuse_scale=1e6` do.

| key | shape | contents |
|---|---|---|
| `factors` | (T, k) | smoothed factor path |
| `loadings` | (n, k) | Λ |
| `A`, `Q` | (k·p, k·p) | companion-form transition and state-shock covariance |
| `H` | (n, n) | diagonal idiosyncratic-noise covariance |
| `X_filled` | (T, n) | observations with model-implied values in place of NaN, back-transformed to levels when `standardize=True` |
| `means`, `stds` | (n,) | the standardisation, for undoing it yourself |
| `factors_df`, `X_filled_df` | DataFrames | added **only** when `X` was a DataFrame |

`kalman_dfm` takes an array or a frame and does not care about the index — it
never resamples. There is no GDP argument and no bridge: it nowcasts the
*columns of the panel you gave it*. To nowcast quarterly GDP with it, put GDP
in the panel yourself, or use `mf_var`, which imposes the aggregation
constraint properly.

**The one trap is complete cases.** Steps 1–3 — PCA, the VAR, Λ and the
idiosyncratic variances — are all estimated on rows with *no* NaN in *any*
column. Only the smoother in step 4 sees the ragged rows. On a panel of 200
months and 8 indicators with a two-month ragged edge, 198 rows are complete
and nothing is lost. Add one series that starts 190 months in and complete
cases fall to **8** — Λ is then estimated from eight observations, with no
warning. The only guard is a hard floor: `complete <= max(n_factors, p) + 2`
raises `ValueError: too few complete-cases rows (…) for n_factors=…, p=…`,
which on a panel where every row has at least one hole reports `(0)`.

So: trim the panel's start to the latest first observation across its columns,
or drop the late-starting series, before calling. The Kalman filter is there
for the bottom edge of the panel, not the left.

## `mf_var` — the aggregation constraint inside the state

Mariano-Murasawa: instead of averaging factors after the fact, posit a latent
*monthly* counterpart of the quarterly variable, put it in a companion VAR
alongside the monthly indicators, and impose

```
y^Q_t = (m*_t + m*_{t-1} + m*_{t-2}) / 3
```

as an observation equation that only binds at quarter-end months. Intra-quarter
months mark the quarterly column NaN, and the smoother interpolates.

```python
from puremacro.nowcast import mf_var

out = mf_var(panel, quarterly_col="gdp", p=3)
out["df_filled"]["gdp_monthly"]   # latent monthly path of the quarterly variable
```

The constraint holds numerically, not approximately: across all 60 quarter
ends of a 180-month synthetic panel the three-month mean of `gdp_monthly`
matched the published value to within 2.1e-06.

| key | shape | contents |
|---|---|---|
| `A`, `Q` | (n·p, n·p) | companion transition, state-shock covariance |
| `Z` | (n, n·p) | observation matrix, carrying the ⅓/⅓/⅓ row |
| `H` | (n, n) | `1e-8 · I` — a numerical ridge, not an estimate |
| `factors_monthly` | (T, n) | smoothed monthly path of every variable |
| `df_monthly` | DataFrame | the same, columns `[monthly cols…, f"{quarterly_col}_monthly"]` |
| `df_filled` | DataFrame | the input frame **plus** one new `f"{quarterly_col}_monthly"` column |

Four things the signature does not tell you:

- **`p` must be ≥ 3, and it raises if it is not.** Three lags of the latent
  state have to live in the companion vector for the aggregation row to reach
  them. `p=2` gives `ValueError: p must be >= 3 so the 3-month aggregation
  lives in the companion state.`
- **`quarter_end_offset` is accepted and has no effect.** The parameter is in
  the signature and the docstring but never read; the aggregation window is
  always backward-looking `(t, t-1, t-2)`. `quarter_end_offset=0` and the
  default `2` return element-for-element identical frames. Stamping quarterly
  values at the *first* month of the quarter and passing `0` therefore applies
  the constraint to the wrong three months — measured on a synthetic panel,
  the fitted backward window closed to 1.7e-06 while the intended forward
  window was off by 1.98. Stamp quarterly observations at quarter-end months.
- **The latent path is a mean, not a sum.** `gdp_monthly` is on the same scale
  as the quarterly variable, not a third of it, because the constraint divides
  by three. If the quarterly column is a quarterly growth rate, `gdp_monthly` is
  *not* a monthly growth rate; it is a monthly series whose three-month mean is
  the quarterly growth rate.
- **The quarterly column is reordered internally.** It need not be last in
  `df`, but `df_monthly` always puts it last and renames it; `df_filled`
  preserves your original column order and appends.

A ragged edge on the *monthly* columns is fine for the smoother, but the two
output frames treat it differently: `df_monthly` reports the smoothed value for
every column, so the holes are filled there, while `df_filled` copies your
input and leaves them NaN.

Initialisation forward-fills the last published quarterly value divided by 3,
then drops every row still carrying a NaN in any column — in practice the
leading months before the first quarterly observation, but a hole in a monthly
column costs that row too. If fewer than `p + 2` months survive, it raises
rather than fitting on nothing.

## Choosing the number of factors

`n_factors` is not estimated by any of the three. `puremacro.factor.bai_ng_ic`
is the standard answer, and it needs a **complete** matrix — NumPy's SVD does
not skip NaN, it raises `LinAlgError: SVD did not converge` — so feed it the
complete-case block.

```python
from puremacro.factor import bai_ng_ic, static_dfm_fit

X = monthly.dropna().to_numpy()

ic = bai_ng_ic(X, kmax=8)
ic["k_hat_IC1"], ic["k_hat_IC2"], ic["k_hat_IC3"]
ic["table"]        # k, V_k, IC1, IC2, IC3 per candidate

fit = static_dfm_fit(X, kmax=8)    # picks k by IC2, then runs PCA
```

Bai-Ng consistency is in *both* n and T, and the small-n failure is not
subtle. On 120-month panels driven by a single true factor, 50 replications
each, `kmax=8`:

| n | what all three ICs return |
|---|---|
| 30 | k̂ = 1 in 50/50 |
| 20 | k̂ = 1 in 47/50 |
| 12 | k̂ = 8 = `kmax` in 40/50 |
| 10, 8 | k̂ = 8 = `kmax` in 50/50 |
| 6 | k̂ = 6 = n in 50/50 |

At k = n the panel is saturated: `V_k` collapses to 5.5e-31, IC1 drops from
−2.96 at k = 5 to −67.85 at k = 6, and the criterion "selects" the rank of the
matrix. **An IC that returns n, or `kmax`, has not selected anything.** Below
roughly 20 series, pick `n_factors` from the variance shares in `pca_factors`
(`variance_share`, `cumulative_share`) and from what the factors look like,
not from the criterion.

## What the news decomposition is, and is not

`res.news_decomposition` is the panel behind "which release moved the
nowcast":

| column | contents |
|---|---|
| `series` | name of the indicator |
| `actual` | its value in the **last row** of `monthly_data`, in the series' own units |
| `forecast` | the factor-implied value for the same month, same units |
| `surprise` | `actual − forecast` |
| `weight` | `Σ_k β_k Λ_ik / (3 σ_i)` — bridge slope times loading, rescaled |
| `contribution` | `weight × surprise`, reported as percentage points |

Only series observed in the last row appear: with three of six indicators
still unpublished, the frame has three rows. If the last row is entirely NaN
it is empty.

Read the table as an ordering of which released series pull the current
quarter up or down. Do not read it as the Bańbura-Modugno news decomposition
the name evokes, for three reasons that are worth stating plainly:

- **Nothing is being decomposed.** A news decomposition splits the *revision*
  between two nowcasts computed on two information sets. `nowcast_gdp` is
  called once, on one data set. There is no earlier nowcast in the call, so
  the contributions do not sum to a revision — there is no revision.
- **`forecast` is not a pre-release forecast.** It is the fitted value at the
  same month, computed from factors that were estimated *with* that
  observation in the panel.
- **`forecast` is mis-scaled.** The EM step reconstructs the panel as
  `F @ Λ.T` with `F = √T · U_k` and `Λ = V_k`, dropping the singular values —
  so each component enters with √T where `S_k` belongs. On a 90-month,
  10-series, one-factor panel the implied fit is exactly `√T / S₁ = 0.343`
  times the correct rank-1 reconstruction, uniformly across every series and
  month; with more than one factor there is no single scalar, because each
  component carries its own `√T / S_k` (0.379 and 0.615 on a two-factor,
  120-month panel). `surprise` therefore carries most of the level the
  reconstruction failed to reproduce, and `weight` is a bridge-and-loading
  product, not a Kalman gain.

Because the bridge regression re-estimates a slope on whatever scale the
factors happen to have, **the nowcast itself is unaffected** by that scaling —
it is invariant to any rescaling of `F`, and rescaling the whole input panel
reproduces the nowcast to six decimals. Only the news block is.

## Combining nowcasts

Five rules, one signature: `(forecasts, realised)` with `forecasts` shaped
(T, M) and `realised` shaped (T,), returning
`{"combined": (T,), "weights": (M,), "method": str}`.

```python
from puremacro.nowcast import (
    equal_weight, inverse_mse, bates_granger, rank_weight, model_confidence_set,
)

inverse_mse(forecasts, realised)["weights"]
model_confidence_set(forecasts, realised, alpha=0.10)   # adds "kept": int
```

| rule | weights | note |
|---|---|---|
| `equal_weight` | 1/M | ignores `realised`; the benchmark that is hard to beat |
| `inverse_mse` | ∝ 1 / MSE_m | |
| `bates_granger` | min-variance, `ridge=1e-10` | unconstrained — weights can go negative |
| `rank_weight` | ∝ 1 / rank(MSE_m) | Stock-Watson-style trimming, insensitive to MSE magnitude |
| `model_confidence_set` | equal over `MSE_m ≤ (1+alpha)·min MSE` | a threshold filter, *not* the Hansen-Lunde-Nason test |

`ridge` and `alpha` are keyword-only. On a 200-period, four-forecaster horse
race with error scales 0.2, 0.5, 0.9 and 2.5, `inverse_mse` returned weights
(0.813, 0.138, 0.042, 0.007) and `bates_granger` returned
(0.842, 0.131, 0.027, **−0.001**). The negative weight is the estimator
working as specified, not a bug — but it is also why the equal-weight
benchmark survives in practice, and `rank_weight` exists as the robust middle.

**All five compute weights in-sample, on the same rows they then combine.**
Comparing their RMSE on those rows ranks the fitting, not the forecasting.
For weights that only ever see past errors, use
`puremacro.forecast.combine_forecasts(forecasts, method=..., errors=...,
rolling=n)`, whose `method` is one of `"equal"`, `"inv_mse"`,
`"bates_granger"`.

## Scoring a predictive distribution

Point-forecast comparison (Diebold-Mariano, Giacomini-White) lives in
`puremacro.forecast.compare`. What is here scores *distributions*.

```python
from puremacro.nowcast import (
    crps_gaussian, crps_ensemble, log_score_gaussian, brier_score, pit_histogram,
)

crps_gaussian(realised, mu, sigma).mean()      # smaller is better
log_score_gaussian(realised, mu, sigma).mean() # larger is better
crps_ensemble(realised, ensemble)              # ensemble is (T, M)
pit_histogram(realised, ensemble, n_bins=10)   # {"pit", "hist", "edges"}
brier_score(realised_binary, predicted_prob)   # e.g. P(negative quarter)
```

The four score functions — everything but `pit_histogram` — return
**per-observation** arrays, not a scalar; take the mean yourself.
`pit_histogram` returns a dict, and a U-shaped histogram means underdispersed
forecasts, an inverted U overdispersed.

### Name collisions with `puremacro.forecast`

`puremacro.nowcast` and `puremacro.forecast` both export
`model_confidence_set`, `crps_gaussian` and `crps_ensemble`, and only
`crps_gaussian` computes the same thing in both.

| name | `puremacro.nowcast` | `puremacro.forecast` |
|---|---|---|
| `model_confidence_set` | MSE threshold filter; positional `(forecasts, realised)` | full Hansen-Lunde-Nason elimination with a stationary bootstrap (`n_boot=1000`); keyword `(forecasts=…, realized=…)`, returns `included`, `excluded`, `pvalues`, `elimination_order` |
| `crps_ensemble` | spread term divided by M² | *fair* estimator, spread term divided by M(M−1) |
| `crps_gaussian` | same closed form | same closed form |

Note the spelling too: the combiners in `nowcast.combine` take `realised`
positionally, the `forecast` MCS takes `realized=` as a keyword.

The `crps_ensemble` gap is the usual finite-ensemble bias: the M² version
overstates CRPS by `E|X − X'| / (2M)`. At M = 500 and σ = 0.5 the two returned
0.117064 and 0.116501 — a gap of 5.63e-04 against a predicted 5.64e-04, or
0.48%. It matters when you compare ensembles of different sizes, and not
otherwise. **When both are in scope, import explicitly from the module you
mean.**

## Backtesting

There is no real-time backtest harness in `puremacro.nowcast` — no rolling
re-estimation loop, no release-calendar replay. Building one has two parts,
and the package supplies both:

- **The vintages.** A backtest that re-runs today's estimator on today's
  revised history is not a backtest. `puremacro.fetch.vintage_panel` returns
  the editions of GDP as they were actually published, so the target you score
  against is the one a forecaster would have been scored against. See
  [Real-time data](real_time_data.md).
- **The tests.** `puremacro.forecast.diebold_mariano` and
  `giacomini_white` for two-model comparisons,
  `puremacro.forecast.model_confidence_set` (the real one) plus
  `losses_from_forecasts` for a field of them.

The loop between the two — truncate the panel at each pseudo-real-time date,
call the estimator, store the nowcast — is yours to write. `nowcast_gdp` runs
in about 3 ms on a 120 × 15 panel, so a few hundred iterations is not the
expensive part.
