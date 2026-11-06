> 🇬🇧 English · 🇪🇸 [Español](es/asset_pricing.md)

For portfolio backtesting, allocation, technical/fundamental analysis and
chartist–fundamentalist dynamics, see [investment research](investment_research.md).

# Returns, risk and asset-pricing tests

`puremacro.asset_pricing` measures return risk and tests asset-pricing
models, including models whose factors come from macro data. It covers
returns and risk statistics, value at risk and its backtests, the
time-series (GRS) and Fama-MacBeth factor tests, linear stochastic
discount factor (SDF) GMM, the Hansen-Jagannathan distance and the
Hansen-Singleton consumption CAPM. The moment-based estimators run on the
public GMM engine `puremacro.inference.gmm`. Each estimator is checked
against an independent implementation (statsmodels, linearmodels, closed
forms or a direct minimization) in the test suite. The runtime needs only
NumPy, SciPy and pandas.

All inputs are per-period **decimal** returns. Every estimator refuses NaN
or inf values and names the offending columns; nothing is dropped
silently. A return column that looks quoted in percent (median `|r|` above
0.2, or more than 5% of its values above 1 in absolute value) is refused;
pass `check_scale=False` for genuine decimal returns of that size. Series with a `DatetimeIndex` are paired by date, and
mixed frequencies raise an error.

The examples below run offline on seeded synthetic data.

## Returns and risk

```python
import numpy as np
import pandas as pd
from puremacro import asset_pricing as ap

rng = np.random.default_rng(1)
idx = pd.date_range("2000-01-01", periods=240, freq="MS")
factors = pd.DataFrame(0.005 + 0.04 * rng.standard_normal((240, 2)),
                       index=idx, columns=["mkt", "hml"])
loadings = rng.uniform(0.5, 1.5, (2, 6))
excess = pd.DataFrame(factors.to_numpy() @ loadings + 0.02 * rng.standard_normal((240, 6)),
                      index=idx, columns=[f"p{i}" for i in range(6)])
rf = pd.Series(0.002, index=idx)

prices = 100 * (1 + excess["p0"] + rf).cumprod()
r = ap.to_returns(prices)                       # simple returns, first row dropped
quarterly = ap.compound(excess, to="Q")         # prod(1 + r) - 1 within quarters
print(round(ap.annualize_return(excess["p0"], 12), 4),
      round(ap.annualize_volatility(excess["p0"], 12), 4))

risk = ap.risk_summary(excess, periods_per_year=12)
print(risk.summary())
risk.to_dataframe()[["name", "sharpe", "sharpe_se", "max_drawdown"]]
```

`sharpe_se` is the Lo (2002) delta-method standard error with a Bartlett
long-run covariance (`hac_lags=None` uses `floor(4 (T/100)^(2/9))`;
`hac_lags=0` gives the iid formula). Pass `rf=` to subtract a risk-free
rate before every statistic.

## Value at risk and backtests

```python
loss = excess["p0"]
print(ap.value_at_risk(loss, alpha=0.05),                     # historical
      ap.value_at_risk(loss, alpha=0.05, method="normal"),
      ap.expected_shortfall(loss, alpha=0.05, method="garch"))  # filtered historical simulation

bt = ap.var_backtest(loss, ap.value_at_risk(loss, alpha=0.05), alpha=0.05)
print(bt.summary())
```

VaR and ES are positive losses for the next period. `"garch"` filters the
series with `puremacro.garch.garch11_fit` and scales the standardized
residual quantile by the one-step-ahead volatility (horizon 1 only).
`var_backtest` reports the Kupiec unconditional-coverage test and
Christoffersen's independence and conditional-coverage tests. With no
violations the statistics stay finite (`0 log 0 = 0`). The in-sample VaR
used here only shows the call: a real backtest uses forecasts formed
before each period.

## Factor tests

```python
ts = ap.time_series_test(excess, factors, hac_lags=4)
print(ts.summary())          # alphas, GRS F(N, T-N-K), HAC Wald chi2(N)

fm = ap.fama_macbeth(excess, factors)
print(fm.summary())          # lambdas with Fama-MacBeth, Shanken and Newey-West se
```

`time_series_test` regresses each asset on `[1, f]`. The GRS statistic is
exact `F(N, T-N-K)` only under iid normal errors and with traded
excess-return factors. Pass `traded=False` for macro factors; GRS is then
not reported, but the HAC Wald test still is. `fama_macbeth` accepts either
`factors` (betas from full-sample time-series regressions) or `betas=`.

## Stochastic discount factor and HJ distance

```python
sdf = ap.sdf_gmm(excess, factors)            # m = 1 - b'(f - mu), two-step GMM
print(sdf.summary())

hj = ap.hj_distance(excess, factors, n_sim=20_000, seed=0)
print(hj.summary())
```

`sdf_gmm` estimates `b` (and `mu`, when `demeaned=True`) through
`inference.gmm` and reports the implied premia `lambdas = Sigma_f b`,
pricing errors and the Hansen J test. `hj_distance` uses the second-moment
weighting matrix and the Jagannathan-Wang (1996) weighted chi-square
p-value. It simulates the p-value with the recorded `seed`.

Redundant test assets in the HJ distance and unidentified factor loadings raise
a clear `ValueError`; singular systems do not produce inference. Historical
VaR and ES compound simple returns across overlapping multi-period paths. The
normal method retains its additive iid approximation.

## Consumption CAPM with macro data

```python
T = 5_000
g = pd.Series(np.exp(0.005 + 0.02 * rng.standard_normal(T)))   # gross consumption growth
beta, gamma = 0.98, 4.0
def moment(a):                                                # E[g^a], lognormal growth
    return np.exp(a * 0.005 + 0.5 * a ** 2 * 0.02 ** 2)

gross = pd.DataFrame({"equity": g / (beta * moment(1 - gamma)),   # Lucas-tree claim on C
                      "levered": g ** 3 / (beta * moment(3 - gamma))})  # claim on C^3
rf_gross = pd.Series(np.full(T, 1 / (beta * moment(-gamma))))
cc = ap.ccapm(gross, g, rf=rf_gross, instruments=np.ones((T, 1)))
print(cc.summary())
```

`ccapm` estimates the power-utility Euler equations
`E[(beta g^-gamma R - 1) z_t] = 0`. It needs **gross** returns and growth.
The default instruments are a constant, lagged consumption growth and
lagged returns. With real data, build the inputs from FRED:

```python
# requires: network
g_q = ap.fred_consumption_growth(frequency="Q", start="1959")   # per-capita real ND + services
rf_q = ap.fred_risk_free(frequency="Q", start="1959")           # TB3MS / 100 / 4, quarter average
```

## Data sources

```python
# requires: network
from puremacro.fetch.kenfrench_factors import fetch

ff3 = fetch("ff3")                  # Mkt-RF, SMB, HML, RF in decimals
p25 = fetch("25_size_bm")           # 25 value-weighted Size-B/M portfolios plus RF
sample = slice("1963-07-01", "2013-12-01")
ex25 = p25.loc[sample].drop(columns="RF").sub(p25.loc[sample, "RF"], axis=0)
grs = ap.time_series_test(ex25, ff3.loc[sample, ["Mkt-RF", "SMB", "HML"]], hac_lags=0)
grs.grs_stat     # about 3.57 on the 2026 library; Fama and French (2015) report 3.62
```

- **Ken French data library.** The files state "Copyright Eugene F. Fama
  and Kenneth R. French" and grant no redistribution licence. puremacro
  bundles none of them. `fetch.kenfrench_factors.fetch` downloads them on
  demand into your user HTTP cache (`~/.cache/puremacro/http`, or
  `$PUREMACRO_HTTP_CACHE_DIR`). The source registry marks
  `kenfrench.factors` as not redistributable. The library is revised after
  publication (CRSP corrections, breakpoint updates). Published statistics
  are therefore reproduced only approximately. The network test accepts
  15% and measures 3.567 against the published 3.62.
- **FRED** (no key). `fred_consumption_growth` sums monthly nominal
  nondurables (`PCEND`) and services (`PCES`), each deflated by its own
  chain-type price index (`DNDGRG3M086SBEA`, `DSERRG3M086SBEA`), and divides
  by population (`POPTHM`). It averages within quarters for `"Q"`. Summing
  two chained-dollar series only approximates a chained aggregate. Any
  missing series raises `RuntimeError` rather than giving growth from a
  partial sum. `fred_risk_free` converts `TB3MS` (annual percent, discount
  basis) by simple division, which is an approximation.

## Generic GMM

```python
from puremacro.inference import gmm

zi = rng.standard_normal((2_000, 3))
x = zi @ np.array([0.8, 0.5, 0.3]) + rng.standard_normal(2_000)
y = 1.0 + 2.0 * x + rng.standard_normal(2_000)
Z = np.column_stack([np.ones(2_000), zi])
X = np.column_stack([np.ones(2_000), x])

fit = gmm(lambda th, d: d[2] * (d[0] - d[1] @ th)[:, None], np.zeros(2), (y, X, Z),
          names=("const", "x"))
print(fit.summary())
fit.wald(np.eye(2)[[1]], [2.0])
```

`gmm(moments, theta0, data)` takes per-period moment contributions
`(T, q)`. It supports `weighting="two_step"` (default), `"iterated"`,
`"identity"` or a matrix, an optional analytic `jacobian`, and a Bartlett
long-run covariance. The covariance is the sandwich
`B G'W S W G B / T`, `B = (G'WG)^-1`, which is the efficient form when
`W = S^-1`. Hansen's `J = T gbar'W gbar` is reported for the efficient
weightings only. On linear IV the estimates, standard errors and J match
linearmodels `IV2SLS`/`IVGMM` and `inference.hansen_j` to 1e-8.

## Assumptions and limits

- **GRS** is exact only under iid normal errors and requires traded
  excess-return factors. Otherwise use the HAC Wald test, but note that
  it over-rejects in short samples (about 23% at a nominal 5% with T = 60
  and N = 10): with traded factors and few periods, prefer GRS.
- **HAC bandwidth.** The default for every Bartlett long-run covariance is
  `floor(4 (T/100)^(2/9))`, uncentered. `hac_lags=0` is the iid
  (heteroskedasticity-robust) case.
- **Sharpe annualization** multiplies by `sqrt(periods_per_year)`. It
  ignores Lo's serial-correlation factor eta(q).
- **HJ distance with excess returns.** The SDF mean is not identified, so
  the distance depends on the normalization `m = 1 - b'f` (Kan and
  Robotti 2008).
- **Shanken standard errors** need first-pass residuals. With `betas=`
  given they are `NaN`, and `se` falls back to Fama-MacBeth.
  linearmodels' `LinearFactorModel` reports a GMM robust se. It is
  asymptotically equal to Shanken under iid errors but is a different
  definition: the test measures a gap of about 0.2%.
- **SDF lambdas** treat the factor covariance as known. linearmodels'
  `LinearFactorModelGMM` estimates the beta representation, a different
  estimator, so the two premia agree only approximately.
- **CCAPM gamma is weakly identified** in consumption data (Stock and Wright
  2000). Report the J test and the 95% intervals that `summary()` prints,
  not the point estimate alone.
- **Percent guard.** It is a heuristic applied per return column: median
  `|r| > 0.2` or more than 5% of values with `|r| > 1`. Isolated returns
  above 100% (the 25 Size-B/M file in 1932, single stocks, crypto) pass.
  Volatile series that trip it can be passed with `check_scale=False`.
  Factors are checked for finiteness only, except traded factors in
  `time_series_test(traded=True)`.
- **GMM conventions.** The sandwich covariance has no small-sample
  correction. J is `NaN` for identity or user weighting, and exactly zero
  when the model is just identified.
