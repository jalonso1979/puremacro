> 🇬🇧 English · 🇪🇸 [Español](es/nowcast.md)

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
| `nowcast_gdp` | EM-PCA factors + factor VAR + quarterly bridge regression | a monthly frame **and** a quarterly GDP series | iterative PCA imputation; all-NaN months and the rest of the target quarter by the factor VAR |
| `kalman_dfm` | Doz-Giannone-Reichlin (2011) two-step DFM | one frame, one frequency | exact Kalman smoother |
| `mf_var` | Mariano-Murasawa (2003) mixed-frequency VAR | one monthly frame with the quarterly column stamped once per quarter | exact Kalman smoother |

Nothing on this page touches the network. Every block below runs on the
synthetic panel built here — a one-factor DGP with ten indicators, a ragged
last month and 39 quarters of GDP driven by the three-month factor average:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
T_m, N = 119, 10                                   # 39 full quarters + 2 months of 2024Q4
F = np.zeros(T_m)
for t in range(1, T_m):
    F[t] = 0.85 * F[t - 1] + rng.normal(scale=0.5)
lam = rng.uniform(0.5, 1.5, N)
X = F[:, None] * lam[None, :] + rng.normal(scale=0.3, size=(T_m, N))
monthly = pd.DataFrame(X, index=pd.date_range("2015-01-01", periods=T_m, freq="MS"),
                       columns=[f"ind_{i}" for i in range(N)])
monthly.iloc[-1, [3, 4, 8, 9]] = np.nan            # four series not yet published
gdp = pd.Series(1.0 + 2.0 * F[:117].reshape(39, 3).mean(axis=1) + rng.normal(scale=0.1, size=39),
                index=pd.period_range("2015Q1", periods=39, freq="Q"), name="gdp")
```

## `nowcast_gdp` — factors, a factor VAR, and a bridge

The workhorse, and the one that returns a number you can quote. Five steps,
all inside one call:

1. Standardise the monthly panel, interpolate the holes as a starting value,
   then run PCA and re-impute the missing entries from the rank-*k*
   reconstruction `F @ Λ.T = U_k S_k V_k'` (`F = √T U_k`,
   `Λ = V_k S_k / √T`) until the factors stop moving (`max_em_iter=50`,
   `em_tol=1e-4`). A month with no observation at all carries no information
   and is held at the panel mean during the EM.
2. Fit a VAR(`p_factor_lags`) on the factors (OLS, no intercept). Use it to
   replace the factor of any all-NaN month with the forecast from the
   preceding months, and to forecast the months of the target quarter that lie
   beyond the end of the frame.
3. Average the monthly factors — observed plus forecast — within each
   calendar quarter.
4. OLS-regress historical quarterly GDP on those quarterly factor averages —
   the *bridge* — aligning the two by quarter label.
5. Apply the bridge coefficients to the target quarter's factor average.

```python
from puremacro.nowcast import nowcast_gdp

res = nowcast_gdp(monthly, gdp, n_factors=2)
res.nowcast          # float, in the units of `gdp`
res.target_quarter   # '2024Q4'
res.model_r2         # bridge R²
res.factor_forecast  # the VAR forecast for 2024-12, the month the frame lacks
print(res.summary())
```

### The input frame

Two objects, and the bridge lines them up by **quarter label**.

| argument | type | index it needs | missingness |
|---|---|---|---|
| `monthly_data` | `DataFrame` (T_months × N) | `DatetimeIndex`, monthly | NaN anywhere; NaN in the last rows *is the point* |
| `quarterly_gdp` | `Series` | `PeriodIndex` or `DatetimeIndex` (quarterly), or strings `'2024Q3'` | NaN quarters are dropped from the bridge |

```python
monthly.index                      # DatetimeIndex, freq 'MS'
gdp.index                          # PeriodIndex 'Q-DEC' — a DatetimeIndex or '2015Q1' strings work too
gdp_str = gdp.copy(); gdp_str.index = gdp.index.astype(str)
assert nowcast_gdp(monthly, gdp_str, n_factors=2).nowcast == res.nowcast
```

The monthly factors are grouped into calendar quarters and labelled
`to_period("Q")` style — `'2015Q1'` — and `quarterly_gdp.index` is converted
the same way: a `PeriodIndex` or `DatetimeIndex` through `to_period("Q")`
(quarter-start and quarter-end stamps both map to the same label), anything
else through `str`. Three consequences:

- **`PeriodIndex`, `DatetimeIndex` and string labels give the same answer**,
  to the last digit, and a GDP series that starts four quarters after the
  panel is aligned on the quarters it covers — there is no positional
  fallback any more.
- **Fewer than four matching labels raises**
  `ValueError: nowcast_gdp: fewer than 4 quarter labels of quarterly_gdp match
  the monthly frame (0 matched) …`, with a sample of both label sets in the
  message. A mismatch that used to be silently mis-aligned is now loud.
- **A non-`DatetimeIndex` `monthly_data` is grouped positionally** in blocks
  of three rows labelled `'Q1', 'Q2', …` — so `quarterly_gdp` must then carry
  those labels too (`'Q1'` for the first three rows), and `target_quarter`
  comes back as `'Q40'` rather than a date.

### Where the frame should end

End the monthly frame at the last month with at least one published
observation. Reindexing forward to the end of the quarter with all-NaN rows is
**equivalent**, not harmful: an all-NaN month gets the factor VAR's forecast,
which is exactly what the completion step produces for a month that is absent
from the frame. On the synthetic panel above the two paths agree to 5e-15.

What does cost is information. The target quarter's factor average is always
a three-month average, but the months the frame does not have are forecasts
from an AR process with root 0.85, and forecasts revert. Over 200
replications of the panel above, RMSE against the realised GDP value:

| months of the target quarter observed | RMSE |
|---|---|
| 3 (frame ends on the quarter's last month) | 0.153 |
| 2 | 0.409 |
| 1 | 0.735 |
| 1 + two all-NaN rows appended | 0.735 |
| none (historical mean) | 1.758 |

A fully-NaN last row also empties `news_decomposition`, since that block is
built from the last row alone.

### What comes back

`NowcastResult` is a frozen dataclass.

| field | type | contents |
|---|---|---|
| `nowcast` | `float` | the point nowcast, **in whatever units `quarterly_gdp` was in** |
| `target_quarter` | `str` | label only — see below |
| `factors` | `DataFrame` (T_months × K) | monthly factors, columns `Factor_1 … Factor_K`, indexed like `monthly_data`; all-NaN months carry the VAR forecast |
| `factor_forecast` | `DataFrame` (0–2 × K) | the VAR forecasts for the target-quarter months beyond the frame; empty when the frame ends on a quarter's last month |
| `loadings` | `DataFrame` (N × K) | Λ, indexed by series name, scaled so that `factors @ loadings.T` is the rank-K reconstruction of the standardised panel |
| `bridge_coefficients` | `Series` | `const, Factor_1 … Factor_K`; `nowcast == const + coef @ target-quarter factor average` |
| `factor_var` | `ndarray` (K·p × K) or `None` | stacked VAR coefficients `B`: `F_t = [F_{t-1}, …, F_{t-p}] @ B` |
| `news_decomposition` | `DataFrame` | `series`, `actual`, `forecast`, `surprise`, `weight`, `contribution` |
| `model_r2` | `float` | bridge R², clipped into [0, 1] |

Plus `summary()`, `to_frame()` (the news table), `to_markdown()`,
`to_latex()`, `to_typst()` and `plot()` (factors, with the forecast months
dashed; returns the Figure).

Two labels that are not what they appear:

- **`target_quarter` renames the answer, it does not select it.** The nowcast
  is always built for the last quarter present in the frame. Passing
  `target_quarter="1999Q1"` changes the string and leaves `nowcast` bit-for-bit
  unchanged. To nowcast an earlier quarter, truncate `monthly_data`.
- **No annualisation happens anywhere in the module**; the nowcast inherits
  the units of `quarterly_gdp`, and `summary()` says so. Hand it annualised
  rates if you want an annualised answer.

`model_r2` is clipped, so a bridge that fits worse than the sample mean
reports `0.0` rather than a negative number, and a `quarterly_gdp` with zero
variance also reports `0.0`.

`p_factor_lags` (default 1) is the lag order of the factor VAR of step 2. It
changes the completion of the target quarter and the filling of all-NaN
months, and nothing else; `factor_var` is `None` — with a `RuntimeWarning`,
and forecasts at the factor mean — only when fewer than `K·p + 2` usable
months exist.

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
print(out.summary())
out.to_markdown()             # the loadings table
```

`n_factors` is keyword-only and has no default; `p=1`, `standardize=True` and
`diffuse_scale=1e6` do. The return value is a `KalmanDFMResult` — a `dict`
subclass, so `out["factors"]` and `"factors" in out` work as before — with
`summary()`, `to_frame()` (loadings), `to_markdown()` / `to_latex()` /
`to_typst()` and `plot()` added.

| key | shape | contents |
|---|---|---|
| `factors` | (T, k) | smoothed factor path |
| `loadings` | (n, k) | Λ |
| `A`, `Q` | (k·p, k·p) | companion-form transition and state-shock covariance |
| `H` | (n, n) | diagonal idiosyncratic-noise covariance |
| `X_filled` | (T, n) | observations with model-implied values in place of NaN, back-transformed to levels when `standardize=True` |
| `means`, `stds` | (n,) | the standardisation, for undoing it yourself |
| `loglik` | float | Gaussian log-likelihood of the panel under the fitted model |
| `n_missing` | int | number of NaN entries that were filled |
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

panel = monthly[["ind_0", "ind_1"]].copy()
latent = pd.Series(0.5 + 0.8 * F + rng.normal(scale=0.2, size=T_m), index=monthly.index)
panel["gdp"] = latent.rolling(3).mean().where(panel.index.month % 3 == 0)   # stamped at quarter ends

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
- **`quarter_end_offset` says which month of the quarter carries the stamp**:
  `2` (default) for quarter-end dating, `1` for the middle month, `0` for the
  first month; anything else raises. Internally a value stamped at month `t`
  with offset `o` is moved to `t + (2 − o)`, so the backward-looking
  aggregation row always binds the three months of the quarter the value
  belongs to — the same quarterly values stamped at month 0, 1 or 2 give the
  same monthly path to 1e-6, and on the synthetic panel the constraint closes
  to 1.9e-06 for every offset. When the last stamped value's quarter runs past
  the frame the state is run `2 − o` months beyond it and truncated back, so
  that observation is not dropped. Declare the offset you used: stamping at
  the first month and leaving the default `2` applies the constraint to the
  *previous* quarter's months.
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
| `forecast` | the factor-implied value for the same month, same units: `mean_i + std_i · F_T Λ_i` |
| `surprise` | `actual − forecast` |
| `weight` | `β' (Λ'Λ)⁻¹ Λ_i / (3 σ_i)` — the effect of a unit surprise in series *i* on the nowcast, through the factor projection and the bridge slope |
| `contribution` | `weight × surprise`, in the units of `quarterly_gdp` |

Only series observed in the last row appear: with three of six indicators
still unpublished, the frame has three rows. If the last row is entirely NaN
it is empty (and `to_frame()` returns a one-row placeholder so the table
renderers still work).

`forecast` is on the data scale — `F_T Λ_i` is the row of the rank-K
reconstruction, so on the synthetic panel the six reconstructed values sit
within the idiosyncratic noise of the actuals. `weight` is the exact
sensitivity of the nowcast to that entry under the model: the factor
extraction is `F_t = (Λ'Λ)⁻¹ Λ' x_t`, the quarter average divides by three,
and the bridge slope β maps it into GDP units.

Read the table as an ordering of which released series pull the current
quarter up or down. Do not read it as the Bańbura-Modugno news decomposition
the name evokes, for two reasons that are worth stating plainly:

- **Nothing is being decomposed.** A news decomposition splits the *revision*
  between two nowcasts computed on two information sets. `nowcast_gdp` is
  called once, on one data set. There is no earlier nowcast in the call, so
  the contributions do not sum to a revision — there is no revision.
- **`forecast` is not a pre-release forecast.** It is the fitted value at the
  same month, computed from factors that were estimated *with* that
  observation in the panel. The genuine pre-release expectation would come
  from the factor VAR applied to the previous month.

## Combining nowcasts

Five rules, one signature: `(forecasts, realised)` with `forecasts` shaped
(T, M) and `realised` shaped (T,), returning
`{"combined": (T,), "weights": (M,), "method": str}`.

```python
from puremacro.nowcast import (
    equal_weight, inverse_mse, bates_granger, rank_weight, model_confidence_set,
)

realised = rng.normal(size=200)
forecasts = realised[:, None] + rng.normal(size=(200, 4)) * np.array([0.2, 0.5, 0.9, 2.5])

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
`"bates_granger"`: the weights applied to `forecasts[t]` are built from
`errors[t-n:t]`, the last `n` errors *strictly before* `t` (equal weights
while fewer than two are available), so perturbing `errors[t]` leaves
`weights[t]` untouched and changes `weights[t+1]`.

## Scoring a predictive distribution

Point-forecast comparison (Diebold-Mariano, Giacomini-White) lives in
`puremacro.forecast.compare`. What is here scores *distributions*.

```python
from puremacro.nowcast import (
    crps_gaussian, crps_ensemble, log_score_gaussian, brier_score, pit_histogram,
)

mu, sigma = forecasts[:, 0], np.full(200, 0.5)
ensemble = mu[:, None] + rng.normal(size=(200, 300)) * 0.5
realised_binary = (realised < 0).astype(float)
predicted_prob = np.clip(0.5 - 0.3 * mu, 0.01, 0.99)

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
