# Penalized Forecasting

`puremacro.forecast.forecast_penalized` answers one question: out of a wide
panel of candidate indicators, which few carry *h*-step-ahead information
about the target, and what is the number for `T+h`?

It is the tool for the case where OLS has stopped being an option — 200
monthly indicators against 120 quarters — and for the case where OLS is still
defined but is fitting noise. It returns a sparse coefficient vector and a
single point forecast.

```python
from puremacro.forecast import forecast_penalized

res = forecast_penalized(X_panel, y_target, horizon=1, alpha=1.0, adaptive=True)
res.forecast            # one float: ŷ_{T+h}
res.selected_features   # the names that survived
print(res.summary())
```

Notebook 34 (`notebooks/34_penalized_macro_forecasting.py`) and
`puremacro/examples/penalized_macro_forecasting.py` run the whole thing on a
30-predictor simulated panel. Nothing on this page touches the network.

## The estimator

There is exactly one: an elastic net fitted by pure-NumPy coordinate descent
with soft-thresholding, optionally with adaptive weights. Two arguments select
the corner of that family you get.

| `alpha` | `adaptive` | what you are fitting |
|---|---|---|
| `1.0` | `False` | Lasso |
| `1.0` | `True` | **Adaptive Lasso** (Zou 2006) |
| `0 < α < 1` | `False` | Elastic Net (Zou & Hastie 2005) |
| `0 < α < 1` | `True` | Adaptive Elastic Net |
| `0.0` | either | nominally Ridge — **do not use it**, see below |

The objective, on standardized predictors, is

```
min  (1/2T) Σ_t (y_{t+h} - β₀ - x_t'β)²  +  λ Σ_j w_j [ α|β_j| + ½(1-α)β_j² ]
```

`adaptive=False` sets every `w_j = 1`. `adaptive=True` sets

```
w_j = 1 / (|β̂_ridge,j| + 1e-3),   then rescaled so median(w) = 1
```

Two details of that weight are worth knowing. The `+1e-3` floor caps any
weight at 1000 before rescaling, so a predictor whose first-stage coefficient
is numerically zero is heavily penalized but not hard-excluded — Zou's
`1/|β̂_j|^γ` is `+∞` there. (The coordinate loop *does* zero a predictor
outright at `w_j ≥ 1e8`, but the 1000 cap puts that threshold out of reach
unless the median first-stage coefficient exceeds 1e5.) And the exponent γ is
fixed at 1; there is no `gamma` argument. The first stage is a ridge on the
standardized data with penalty `1e-2 · I`, but `X'X` has diagonal `T`, so with `T > P` that penalty is
a numerical stabiliser, not shrinkage: it agrees with OLS to about 1e-4. Its
job is to keep the first stage solvable when `P > T`, which is the case the
whole function exists for.

## Parameters

Everything after `y_target` is keyword-only.

| parameter | default | what it does |
|---|---|---|
| `X_panel` | — | `(T, P)` DataFrame or ndarray of predictors dated *t* |
| `y_target` | — | `(T,)` Series or ndarray of the target |
| `horizon` | `1` | direct horizon *h*; `y_{t+h}` is regressed on `X_t` |
| `alpha` | `0.5` | elastic-net mixing; `1.0` = Lasso, `0.0` = Ridge |
| `adaptive` | `True` | adaptive weighting of the penalty |
| `n_lambdas` | `40` | points on the λ grid |
| `lambda_min_ratio` | `1e-3` | `λ_min / λ_max` |

The coordinate-descent tolerances (`max_iter=1000`, `tol=1e-6`) are internal to
`_fit_coordinate_descent` and not exposed.

## Direct, never iterated

Multi-step is **direct only**. With `horizon=h`, the function fits

```
y_{t+h} = β₀ + x_t'β
```

on rows `t = 0 … T-h-1`, and evaluates the fitted equation at the last row of
`X_panel` to get `ŷ_{T+h}`. There is no iterated route — no one-step model
chained forward, and no companion form — so a horizon-4 forecast is a
*separately estimated* horizon-4 model, not four applications of a horizon-1
one. That is the right default for a penalized fit (the iterated route needs a
model for every predictor as well as for the target), but it means the
selected predictor set changes with `h` and there is no reason for the
horizon-1 and horizon-4 sets to agree. On the shipped 30-predictor example the
adaptive Lasso selects 4 predictors at `h=1` with in-sample R² 0.963, and a
different 3 at `h=4` with R² 0.096 — the DGP loads on `X_{t-1}` and its
predictors are AR(1) with ρ between 0.3 and 0.8, so nearly all of the one-step
signal has decayed by four steps. That collapse is the estimator working, not
failing.

`horizon=0` is accepted and does something different: it skips the alignment
entirely and returns the **in-sample fitted value at the last observation**,
not a forecast. `horizon` is not validated, and a negative value takes the same
branch.

## How λ is chosen — BIC, and there are no folds

Selection is a grid search minimising the **Bayesian information criterion** on
the estimation sample:

```
BIC(λ) = T_eff · log(MSE(λ))  +  df(λ) · log(T_eff),   df = #{|β_j| > 1e-5} + 1
```

over `n_lambdas` points spaced geometrically from `λ_max` down to
`λ_max · lambda_min_ratio`. The `df` count is taken on the *standardized*
coefficients, while `selected_features` thresholds the un-standardized ones, so
on a panel with wildly different column scales the two counts can differ by one
or two. `λ_max` is the standard closed form — the smallest
penalty that drives every coefficient to zero, `max_j |x_j'(y-ȳ)| / (T α w_j)`
— verified: at `λ_max` exactly 0 coefficients survive and one appears at
`0.999 λ_max`. Ties go to the largest λ, i.e. to the sparser model. Each λ is
fitted from a cold start; there is no warm-starting down the path.

**There is no cross-validation anywhere in this function, or anywhere in
`puremacro.forecast`.** No k-fold, no rolling-origin CV, no blocked or purged
folds, no held-out set. To say the obvious thing explicitly: the question "do
the folds respect time order?" has no answer here because there are no folds.
Nothing is resampled and nothing is shuffled, so no future observation can leak
into a training fold — but the flip side is that λ is chosen to fit the
estimation sample well, not to forecast well, and an information criterion is
a much blunter instrument for that than an out-of-sample loss.

Two consequences to plan around:

- **`optimal_lambda` is not comparable across calls.** The grid is built from
  `α` and the adaptive weights, so λ means a different amount of shrinkage in
  each configuration. On the shipped example, `λ_max` is 1.57 at
  `alpha=1.0, adaptive=False` and 73.55 at `alpha=1.0, adaptive=True`. Compare
  selected sets and losses, not λ.
- **Check the chosen λ is interior.** If it lands on either end of the grid,
  the grid did not bracket the BIC minimum and the answer is an artefact of
  `lambda_min_ratio`:

  ```python
  edges = (res.bic_path.index[0], res.bic_path.index[-1])
  assert res.optimal_lambda not in edges, "widen lambda_min_ratio"
  ```

  It happens for real. At `alpha=0.02, adaptive=True` on the shipped example
  the default grid pins to `λ_min = 3.68` and returns 11 predictors with
  R² 0.956; `lambda_min_ratio=1e-5` moves the optimum into the interior at
  λ\* = 2.29 (13 predictors, R² 0.961), and adding `n_lambdas=60` refines it to
  λ\* = 1.01 (16 predictors, R² 0.966).

## Standardisation, and what a coefficient means

The predictors are standardized before fitting, using the **training rows
only** (`X_mat[:-horizon]`), by that block's mean and its *population* standard
deviation (`np.std`, ddof = 0). A column with zero variance gets its σ forced
to 1, which leaves it demeaned to all zeros — a constant column you leave in
`X_panel` therefore receives a coefficient of exactly `0.0` and never appears
in `selected_features`. Do not add your own intercept column; the intercept is
fitted separately and is never penalized.

The target is **not** standardized. It is only demeaned implicitly, through the
intercept, which is re-estimated as `mean(y - Xβ)` on every sweep.

This matters for reading the output, because the penalty applies to the
standardized coefficients — so it is scale-free, and rescaling a predictor from
percent to basis points does not change which predictors are selected — but the
coefficients that come *back* are un-standardized:

```
β_returned = β̂_std / σ_x        β₀_returned = β̂₀ - Σ_j β_returned,j · μ_x,j
```

So `res.coefficients["unemployment"]` is the effect on `y` of **one unit** of
that predictor in its own units, directly usable in
`res.intercept + X_row @ res.coefficients`, and *not* the standardized effect
you would compare across predictors to rank importance. For that ranking,
multiply back: `res.coefficients * X_panel.iloc[:-horizon].std(ddof=0)`. The
`summary()` table sorts by the un-standardized magnitude, so it ranks by units,
not by importance.

## What comes back

A frozen `PenalizedForecastResult` dataclass:

| field | type | contents |
|---|---|---|
| `forecast` | `float` | `ŷ_{T+h}` — one number, not a path and not an interval |
| `selected_features` | `list[str]` | names whose coefficient exceeds `1e-5` in absolute value, in column order |
| `coefficients` | `pd.Series`, length `P` | **all** candidates, zeros included, original units |
| `intercept` | `float` | un-standardized constant |
| `optimal_lambda` | `float` | the BIC-minimising λ |
| `in_sample_r2` | `float` | R² of the chosen model on the training rows, clipped to `[0, 1]` |
| `bic_path` | `pd.Series`, length `n_lambdas` | index = λ descending from `λ_max`; values = BIC |
| `horizon` | `int` | the `h` you passed, echoed |

Plus `.summary()`, a formatted string. When `X_panel` is an ndarray the feature
names are generated as `X_1 … X_P`.

There is no standard error, no confidence interval and no predictive density —
post-selection inference on a penalized fit is not valid without a debiasing
step this module does not implement, and none is pretended.

## Six things that will bite you

- **`alpha=0.0` is broken.** `λ_max` divides by `max(α·w_j, 1e-4)`, so with
  α = 0 the denominator falls back to the `1e-4` floor and the whole grid is
  10⁴ times too high: on the shipped example it spans `[15.7, 15679]` instead
  of `[0.0016, 1.57]`. The ridge shrinkage denominator in the coordinate step is
  `1 + λ(1-α)w_j`, so even the *least*-penalized point on that grid shrinks
  every coefficient by a factor of about 1/16.7. Worse, `df` is counted as the
  number of non-zero coefficients — correct for the Lasso, wrong for a ridge
  fit, where the effective df is the trace of the hat matrix. The result:
  `alpha=0.0, adaptive=False` minimises BIC at `λ_max` and returns the sample
  mean, in-sample R² **0.0002**, against 0.961 for the Lasso on the same data.
  `alpha=0.0, adaptive=True` pins to the other end and manages 0.880. Use
  `alpha=0.05` for a ridge-like fit — it behaves (4 predictors, R² 0.959) —
  and check the λ is interior.
- **NaNs propagate silently and the R² lies.** There is no `dropna`, no mask
  and no complete-case filter. One NaN anywhere in `X_panel` returns
  `forecast=nan`, `selected_features=[]` and `in_sample_r2=1.0` — the R² is
  clipped into `[0, 1]` after being computed from NaNs, so it comes out at the
  *top* of the range. Drop or interpolate before you call. A NaN confined to
  the **last** row of `X_panel` is the nastier case: training is unaffected, the
  coefficients and R² are fine, and only `forecast` is NaN.
- **Alignment is positional; the index is ignored.** `y_target` is converted
  with `.to_numpy()` and never joined on `X_panel`'s index. Shifting the
  target's DatetimeIndex by five years produces a bit-identical result. Line the
  two up yourself. Mismatched *lengths* do raise, but as an opaque numpy
  `matmul` dimension error rather than a check.
- **`T - horizon` must be at least 10.** Below that it raises
  `ValueError: forecast_penalized: effective training sample too small (9 rows)`.
- **`in_sample_r2` is in-sample and BIC-selected.** With `P > T` it goes to
  1.0 by interpolation — on a `(40, 200)` panel of pure noise the fit selects
  more predictors than it has training rows (40–45 across six seeds, against 39
  rows) and reports R² 1.000. It is a diagnostic of the fit, never evidence
  about the forecast.
- **`alpha` is not validated.** `alpha=2.0` runs and returns a result.

## Turning one number into a track record

`forecast_penalized` produces a single forecast from a single estimation
sample, so evaluating it means refitting at every origin. Refitting is the
point — selecting predictors once on the full sample and then "backtesting"
them is look-ahead bias, and the sparse set genuinely does move. A fit is 40
cold-start coordinate-descent paths; at `T=200, P=40` that is about 45 ms on an
Apple M3 Pro, so an 80-origin loop is a few seconds:

```python
import numpy as np
from puremacro.forecast import forecast_penalized, diebold_mariano

h, start = 1, 120
preds, actuals = [], []
for t in range(start, len(y) - h):
    res = forecast_penalized(X.iloc[: t + 1], y.iloc[: t + 1], horizon=h, alpha=1.0)
    preds.append(res.forecast)
    actuals.append(y.iloc[t + h])

e_pen = np.array(preds) - np.array(actuals)
e_bench = np.array([y.iloc[: t + 1].mean() for t in range(start, len(y) - h)]) - np.array(actuals)

diebold_mariano(e_pen, e_bench, h=h)
# {'stat': ..., 'p_value': ..., 'n_obs': len(preds), 'lag_used': 0}
```

`lag_used` is `max(0, h-1)`, so a one-step comparison uses no Newey-West lags
at all.

Note that `X.iloc[: t + 1]` includes row `t`, which is exactly right: rows
`0 … t-h` train the model and row `t` is the one the forecast is evaluated at.

`puremacro.forecast` carries the rest of the evaluation kit, all of it
independent of how the forecasts were made:

| function | tests |
|---|---|
| `diebold_mariano(e1, e2, h=1, loss="mse", power=2.0, small_sample=True)` | unconditional equal predictive ability; Newey-West at `h-1` lags, Harvey-Leybourne-Newbold small-sample correction on by default |
| `giacomini_white(e1, e2, h=1, conditioning_vars=None, loss="mse", power=2.0)` | *conditional* predictive ability — is the loss differential forecastable? |
| `model_confidence_set(losses, *, alpha=0.10, statistic="tmax", n_boot=1000, bootstrap="stationary", seed=None)` | Hansen-Lunde-Nason (2011) block-bootstrap elimination over M ≥ 2 models; block length defaults to `max(2, round(T**(1/3)))` |
| `losses_from_forecasts(forecasts, realized, *, loss="mse")` | builds the `(T, M)` loss matrix the MCS wants |
| `crps_gaussian`, `crps_ensemble`, `pit`, `pit_uniformity_test`, `berkowitz_test`, `klic_amisano_giacomini`, `combine_forecasts` | density-forecast scoring and combination |

`model_confidence_set` is the one to reach for past two models: a pairwise DM
run over M candidates has no family-wise error control, and the MCS is a
sequential elimination that does. It returns a dict with `included`,
`excluded`, `pvalues`, `elimination_order`, and the settings echoed back. Note
that `puremacro.nowcast.combine.model_confidence_set` is a *different*,
deliberately lightweight function — an MSE threshold used to build combination
weights, with no bootstrap and no size control.
