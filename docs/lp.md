> 🇬🇧 English · 🇪🇸 [Español](es/lp.md)

# Local Projections (LP)

Local Projections (Jordà 2005) estimate impulse response functions by running a sequence of direct predictive regressions for each horizon $h = 0, 1, \dots, H$:

$$y_{t+h} - y_{t-1} = \alpha_h + \beta_h x_t + \gamma_h' w_t + \sum_{l=1}^p \Gamma_{h,l} w_{t-l} + \varepsilon_{t+h}$$

where $x_t$ is a structural shock or policy variable, $w_t$ contains control variables (which enter contemporaneously and with $p$ lags, together with $p$ lags of $y_t$ and $x_t$), and $\beta_h$ traces out the impulse response function at horizon $h$.

In contrast to Structural VARs, which iterate forward a one-step-ahead linear model, Local Projections:
- **Do not compound specification errors** across horizons.
- **Naturally accommodate non-linearities**, asymmetric regimes, and state-dependent effects.
- **Support flexible inference** via Newey-West HAC or Driscoll-Kraay standard errors without requiring stationary companion matrices.

In `puremacro 2.0`, all local projection estimators return a unified **`LPResult`** object that behaves as a pandas DataFrame while exposing one-call plotting (`.plot()`) and publication-grade export methods (`.to_latex()`, `.to_typst()`, `.to_markdown()`).

The code blocks on this page build on each other: run them in order and the whole page executes verbatim.

---

## Quickstart: Single-Series LP-HAC

To estimate impulse responses with Newey-West heteroskedasticity- and autocorrelation-consistent (HAC) standard errors:

```python
import numpy as np
import pandas as pd
from puremacro.lp import lp_hac

# Synthetic dataset: response of output to a policy shock
rng = np.random.default_rng(42)
T = 200
shock = rng.standard_normal(T)
output = np.cumsum(0.6 * shock + 0.4 * rng.standard_normal(T))
df = pd.DataFrame({"gdp": output, "shock": shock})

# Estimate LP up to horizon 12 with 4 lags of controls and 90% CI
res = lp_hac(df, y="gdp", x="shock", horizon=12, lags=4, ci=0.90)

# 1. Print structured summary (with significance flags)
print(res.summary())

# 2. Inspect point estimates and standard errors
print(res[["h", "beta", "se", "lo", "hi"]].head())

# 3. Plot IRF with confidence bands in 1 line
res.plot(title="Response of GDP to Structural Shock")
```

### The `LPResult` Object

The returned `res` is an `LPResult` (subclassing `pd.DataFrame`). You can treat it as a standard DataFrame or use its specialized econometric attributes:

| Property / Method | Description |
|---|---|
| `res.point` | Point estimates $\hat{\beta}_h$ as a NumPy array of shape `(H+1,)` |
| `res.se` | HAC standard errors $\hat{\text{se}}(\hat{\beta}_h)$ of shape `(H+1,)` (all-NaN for band-only estimators such as `lp_quantile`) |
| `res.ci_lower` | Lower confidence bound $\hat{\beta}_h - z_{\alpha/2} \cdot \hat{\text{se}}$ |
| `res.ci_upper` | Upper confidence bound $\hat{\beta}_h + z_{\alpha/2} \cdot \hat{\text{se}}$ |
| `res.t_stat` | $\hat{\beta}_h / \hat{\text{se}}$ (NaN where no SE is available) |
| `res.horizons` | Array of horizons evaluated `[0, 1, ..., H]` |
| `res.labels` | `[]` for single-coefficient results; `['H', 'L']` / `['pos', 'neg']` for regime or sign results |
| `res.plot()` | Renders an impulse response plot with shaded confidence bands (one line and band per regime for regime results, one line per quantile for `lp_quantile`) |
| `res.summary()` | Formatted tabular summary string with significance flags: `***` p<0.01, `**` p<0.05, `*` p<0.10 (two-sided normal z-test); for band-only estimators a single `*` marks bands that exclude zero |
| `res.to_markdown()` | Renders table formatted for Quarto, GitHub, or Obsidian |
| `res.to_latex()` | Generates a clean `\begin{tabular}` for LaTeX / Overleaf |
| `res.to_typst()` | Generates a `#table(...)` block for modern Typst documents |

For regime or sign results (`lp_state_dep`, `lp_state_dep_iv`, `lp_asymmetric`, `lp_garch_state`) the table carries one coefficient per label (`beta_H`/`beta_L`, `beta_pos`/`beta_neg`) and `res.point`, `res.se`, `res.ci_lower`, `res.ci_upper` and `res.t_stat` return a DataFrame indexed by `h` with one column per label.

---

## Instrumental Variable Local Projections (`lp_iv`)

When the policy intervention $x_t$ is endogenous (for example, policy interest rates reacting contemporaneously to economic shocks), standard OLS local projections suffer from omitted-variable bias. 

`lp_iv` implements Two-Stage Least Squares (2SLS) local projections using an external instrument $z_t$ (such as high-frequency monetary surprises or narrative tax changes):

$$\text{Stage 1: } x_t = \pi_{0,h} + \pi_{1,h} z_t + \text{controls} + v_{t,h}$$
$$\text{Stage 2: } y_{t+h} - y_{t-1} = \alpha_h + \beta_h \hat{x}_t + \text{controls} + \varepsilon_{t+h}$$

The block below first extends the synthetic dataset with the illustrative columns used in the rest of this page (an endogenous policy rate driven by a high-frequency surprise, two controls, government spending driven by a military-news instrument, and an unemployment rate in percent):

```python
from puremacro.lp import lp_iv

# Illustrative columns for the remaining examples (synthetic, seeded above)
df["hf_monetary_surprise"] = rng.standard_normal(T)
df["fedfunds"] = 0.7 * df["hf_monetary_surprise"] + 0.5 * rng.standard_normal(T)
df["inflation"] = rng.standard_normal(T)
df["commodity_prices"] = rng.standard_normal(T)
df["military_news"] = rng.standard_normal(T)
df["gov_spending"] = 0.6 * df["military_news"] + 0.5 * rng.standard_normal(T)
df["unemployment_rate"] = 6.0 + 1.5 * rng.standard_normal(T)   # percent

res_iv = lp_iv(
    df,
    y="gdp",
    x="fedfunds",
    z="hf_monetary_surprise",
    controls=["inflation", "commodity_prices"],
    horizon=16,
    lags=4,
    ci=0.90,
)

# First-stage F statistics across horizons
print("First stage F:", res_iv["first_stage_f"].values)
res_iv.plot(title="LP-IV: Monetary Transmission via External Instrument")
```

---

## State-Dependent Local Projections (`lp_state_dep`)

Macroeconomic transmission often differs across economic regimes (e.g. recessions vs. expansions, or high vs. low debt regimes). `lp_state_dep` partitions responses based on a state variable $s_t$:

$$y_{t+h} - y_{t-1} = \alpha_h + F(s_t) \beta_h^H x_t + (1 - F(s_t)) \beta_h^L x_t + \text{controls} + \varepsilon_{t+h}$$

where $F(s_t) \in [0, 1]$ is the high-regime weight.

### Smooth Logistic vs. Sharp Threshold

- **Sharp Threshold (`transition="threshold"`)**: $F(s_t) = \mathbb{I}\{s_t > c\}$. Observations are discretely partitioned into high and low regimes.
- **Smooth Transition (`transition="logistic"`, default)**: $F(s_t) = \frac{1}{1 + \exp(-\gamma (s_t - c) / \sigma_s)}$, where $\sigma_s$ is the sample standard deviation of the state, so $\gamma > 0$ is the speed of transition in standard deviations of $s_t$ (Auerbach & Gorodnichenko 2012 use $\gamma = 1.5$ on a standardized state).

The cutoff $c$ is the `threshold` argument **on the raw scale of the state variable**: `threshold=6.5` on an unemployment rate in percent means 6.5 %. The default `threshold=None` splits at the sample mean of the state (the standardized-zero convention of Auerbach-Gorodnichenko). A cutoff that leaves every observation in one regime raises a `ValueError` naming the state's range instead of a singular regression.

```python
from puremacro.lp import lp_state_dep

# Smooth transition around 6.5 % unemployment (state_var= is an alias of state=)
res_state = lp_state_dep(
    df,
    y="gdp",
    x="gov_spending",
    state="unemployment_rate",
    transition="logistic",
    gamma=3.0,
    threshold=6.5,
    horizon=12,
    lags=2,
    ci=0.90,
)

print(res_state[["h", "beta_H", "se_H", "beta_L", "se_L"]].head())
print(res_state.summary())              # one row per horizon and regime, with flags
print(res_state.point.head())           # DataFrame with columns H and L
res_state.plot(title="Spending multiplier: high vs low unemployment")
```

---

## State-Dependent LP-IV (`lp_state_dep_iv`)

Following Ramey & Zubairy (2018, *JPE*), evaluating state-dependent government spending multipliers requires instrumenting the interaction terms to identify state-specific causal effects:

$$\text{Endogenous: } [F(s_t) x_t, \; (1 - F(s_t)) x_t]$$
$$\text{Instruments: } [F(s_t) z_t, \; (1 - F(s_t)) z_t]$$

`lp_state_dep_iv` estimates both first-stage regressions, checks regime-specific instrument relevance via first-stage F statistics, and computes second-stage multipliers with HAC inference. `threshold` and `gamma` follow exactly the `lp_state_dep` conventions above:

```python
from puremacro.lp import lp_state_dep_iv

res_rz = lp_state_dep_iv(
    df,
    y="gdp",
    x="gov_spending",
    z="military_news",
    state="unemployment_rate",
    threshold=6.5,          # 6.5 % unemployment, raw scale of the state
    transition="threshold", # Or 'logistic'
    horizon=16,
    lags=4,
    ci=0.90,
)

# Inspect high-slack vs low-slack multipliers and first-stage F stats
print(res_rz[["h", "beta_H", "first_stage_f_H", "beta_L", "first_stage_f_L"]])

# Export the multipliers table directly to LaTeX for your manuscript
latex_table = res_rz.to_latex()
```

---

## Panel Local Projections (`panel_lp` & `panel_lp_dk`)

For cross-country or regional panel datasets, `puremacro` provides two-way fixed-effects panel local projections with cluster-robust (`panel_lp`) or Driscoll-Kraay (1998) standard errors (`panel_lp_dk`, or `panel_lp(..., cov_type="driscoll-kraay")`) robust to cross-sectional spatial correlation and arbitrary serial correlation.

The panel can be a **long-form** frame whose entity and time identifiers are columns (`unit_col=` / `time_col=`), or a frame already indexed by an `(entity, time)` `MultiIndex` whose level names are given by `entity_level=` / `time_level=`:

```python
from puremacro.lp import panel_lp, panel_lp_dk

# Long-form panel: one row per (country, date)
countries = [f"C{i}" for i in range(8)]
dates = list(pd.period_range("2000Q1", periods=60, freq="Q"))
panel_df = pd.DataFrame({
    "country": np.repeat(countries, len(dates)),
    "date": dates * len(countries),
})
panel_df["credit_shock"] = rng.standard_normal(len(panel_df))
panel_df["real_gdp"] = 0.5 * panel_df["credit_shock"] + rng.standard_normal(len(panel_df))

res_panel = panel_lp_dk(
    panel_df,
    y="real_gdp",
    x="credit_shock",
    unit_col="country",
    time_col="date",
    horizon=12,
    lags=2,
)
res_panel.plot(title="Panel LP: Credit Shock Response (Driscoll-Kraay SEs)")

# Same estimates from a frame already indexed by (country, date)
res_indexed = panel_lp_dk(
    panel_df.set_index(["country", "date"]),
    y="real_gdp", x="credit_shock",
    entity_level="country", time_level="date",
    horizon=12, lags=2,
)
assert np.allclose(res_panel.point, res_indexed.point)

# Cluster-robust SEs by entity, or Driscoll-Kraay through cov_type
res_cluster = panel_lp(panel_df, y="real_gdp", x="credit_shock",
                       unit_col="country", time_col="date", horizon=12, lags=2)
res_dk = panel_lp(panel_df, y="real_gdp", x="credit_shock",
                  unit_col="country", time_col="date", horizon=12, lags=2,
                  cov_type="driscoll-kraay")
assert np.allclose(res_dk.se, res_panel.se)
```

### Automatic Row-Order Invariance

In puremacro, all panel LP estimators sort the panel index `(unit, time)` before applying time shifts. Even if your input DataFrame contains unsorted rows, puremacro ensures bit-identical numerical results and guards against positional lag leakage.

---

## Standardized Signatures & Parameter Aliases

Every estimator exported by `puremacro.lp` (including `lp_state_dep` and `lp_did`) accepts the modern keyword-only arguments below in addition to the legacy names:

| Modern Parameter | Accepted Legacy Alias | Default | Description |
|---|---|---|---|
| `lags` | `n_lags` | `2` — except `la_lp` and `smooth_lp` (`4`) and `lp_did` (`0`) | Number of lag controls included in the projection (for `lp_did`: lagged outcome changes $\Delta y_{i,t-k}$ added as controls) |
| `horizon` | `horizons` | `20`, i.e. $h = 0 \dots 20$ — except `lp_iv_lewbel` and `lp_did` (`12`) | Max projection horizon (computes $h = 0 \dots H$) |
| `ci` | `alpha` | `0.90` — except `smooth_lp` (`0.95`) | Confidence interval coverage (e.g. `ci=0.95` $\leftrightarrow$ `alpha=0.05`) |

`ci` and `alpha` must be probabilities in $(0, 1)$: passing a percentage by mistake (`ci=90`) or `alpha=1.5` raises a `ValueError` instead of silently producing NaN bands. `lp_state_dep` additionally accepts `state_var=` as an alias of `state=`.

---

## Comparison: Local Projections vs. SVAR

| Feature | Local Projections (`puremacro.lp`) | Structural VAR (`puremacro.var`) |
|---|---|---|
| **Underlying Philosophy** | Direct multi-step regression | Iterated 1-step linear model |
| **Persistence Assumption** | Robust to non-stationary persistence | Sensitive to unit roots in companion matrix |
| **State Dependence** | Built-in via interaction terms ($F(s_t) x_t$) | Requires non-linear TVP/Threshold VAR |
| **Efficiency at Long Horizons** | Higher variance at large $H$ | More efficient if VAR is correctly specified |
| **Data Requirements** | Long time series needed for high $H$ | Moderate samples suffice |
