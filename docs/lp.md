> 🇬🇧 English · 🇪🇸 [Español](es/lp.md)

# Local Projections (LP)

Local Projections (Jordà 2005) estimate impulse response functions by running a sequence of direct predictive regressions for each horizon $h = 0, 1, \dots, H$:

$$y_{t+h} - y_{t-1} = \alpha_h + \beta_h x_t + \sum_{l=1}^p \Gamma_{h,l} w_{t-l} + \varepsilon_{t+h}$$

where $x_t$ is a structural shock or policy variable, $w_t$ contains control variables (including lags of $y_t$ and $x_t$), and $\beta_h$ traces out the impulse response function at horizon $h$.

In contrast to Structural VARs, which iterate forward a one-step-ahead linear model, Local Projections:
- **Do not compound specification errors** across horizons.
- **Naturally accommodate non-linearities**, asymmetric regimes, and state-dependent effects.
- **Support flexible inference** via Newey-West HAC or Driscoll-Kraay standard errors without requiring stationary companion matrices.

In `puremacro 2.0`, all local projection estimators return a unified **`LPResult`** object that behaves as a pandas DataFrame while exposing one-call plotting (`.plot()`) and publication-grade export methods (`.to_latex()`, `.to_typst()`, `.to_markdown()`).

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

# 1. Print structured summary
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
| `res.se` | HAC standard errors $\hat{\text{se}}(\hat{\beta}_h)$ of shape `(H+1,)` |
| `res.ci_lower` | Lower confidence bound $\hat{\beta}_h - z_{\alpha/2} \cdot \hat{\text{se}}$ |
| `res.ci_upper` | Upper confidence bound $\hat{\beta}_h + z_{\alpha/2} \cdot \hat{\text{se}}$ |
| `res.horizons` | Array of horizons evaluated `[0, 1, ..., H]` |
| `res.plot()` | Renders an impulse response plot with shaded confidence bands |
| `res.summary()` | Formatted tabular summary string with significance flags |
| `res.to_markdown()` | Renders table formatted for Quarto, GitHub, or Obsidian |
| `res.to_latex()` | Generates a clean `\begin{tabular}` for LaTeX / Overleaf |
| `res.to_typst()` | Generates a `#table(...)` block for modern Typst documents |

---

## Instrumental Variable Local Projections (`lp_iv`)

When the policy intervention $x_t$ is endogenous (for example, policy interest rates reacting contemporaneously to economic shocks), standard OLS local projections suffer from omitted-variable bias. 

`lp_iv` implements Two-Stage Least Squares (2SLS) local projections using an external instrument $z_t$ (such as high-frequency monetary surprises or narrative tax changes):

$$\text{Stage 1: } x_t = \pi_{0,h} + \pi_{1,h} z_t + \text{controls} + v_{t,h}$$
$$\text{Stage 2: } y_{t+h} - y_{t-1} = \alpha_h + \beta_h \hat{x}_t + \text{controls} + \varepsilon_{t+h}$$

```python
from puremacro.lp import lp_iv

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

where $F(s_t) \in [0, 1]$ is a transition function.

### Smooth Logistic vs. Sharp Threshold

- **Sharp Threshold (`transition="threshold"`)**: $F(s_t) = \mathbb{I}\{s_t > \bar{s}\}$. Observations are discretely partitioned into high and low regimes.
- **Smooth Transition (`transition="logistic"`)**: $F(s_t) = \frac{1}{1 + \exp(-\gamma (s_t - \bar{s}))}$, where $\gamma > 0$ controls the speed of transition between regimes (Auerbach & Gorodnichenko 2012).

```python
from puremacro.lp import lp_state_dep

# Smooth transition conditional on standardized unemployment rate
res_state = lp_state_dep(
    df,
    y="gdp",
    x="gov_spending",
    state_var="unemployment_rate",
    transition="logistic",
    gamma=3.0,
    threshold=0.0,
    horizon=12,
    lags=2,
)

print(res_state[["h", "beta_high", "se_high", "beta_low", "se_low"]].head())
```

---

## State-Dependent LP-IV (`lp_state_dep_iv`)

Following Ramey & Zubairy (2018, *JPE*), evaluating state-dependent government spending multipliers requires instrumenting the interaction terms to identify state-specific causal effects:

$$\text{Endogenous: } [F(s_t) x_t, \; (1 - F(s_t)) x_t]$$
$$\text{Instruments: } [F(s_t) z_t, \; (1 - F(s_t)) z_t]$$

`lp_state_dep_iv` estimates both first-stage regressions, checks regime-specific instrument relevance via first-stage F statistics, and computes second-stage multipliers with HAC inference:

```python
from puremacro.lp import lp_state_dep_iv

res_rz = lp_state_dep_iv(
    df,
    y="gdp",
    x="gov_spending",
    z="military_news",
    state="unemployment_rate",
    threshold=6.5,          # e.g., 6.5% unemployment threshold
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

For cross-country or regional panel datasets, `puremacro` provides fixed-effects panel local projections with cluster-robust or Driscoll-Kraay (1998) non-parametric standard errors robust to cross-sectional spatial correlation and arbitrary serial correlation:

```python
from puremacro.lp import panel_lp_dk

# Panel dataset with 'country' and 'year_quarter'
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
```

### Automatic Row-Order Invariance

In puremacro, all panel LP estimators sort the panel index `(unit, time)` before applying time shifts. Even if your input DataFrame contains unsorted rows, puremacro ensures bit-identical numerical results and guards against positional lag leakage.

---

## Standardized Signatures & Parameter Aliases

All `puremacro.lp` estimators support standardized modern macro terminology:

| Modern Parameter | Accepted Legacy Alias | Default | Description |
|---|---|---|---|
| `lags` | `n_lags` | `2` | Number of lag controls included in the projection |
| `horizon` | `horizons` | `20` | Max projection horizon (computes $h = 0 \dots H$) |
| `ci` | `alpha` | `0.90` | Confidence interval coverage (e.g. `ci=0.95` $\leftrightarrow$ `alpha=0.05`) |

---

## Comparison: Local Projections vs. SVAR

| Feature | Local Projections (`puremacro.lp`) | Structural VAR (`puremacro.var`) |
|---|---|---|
| **Underlying Philosophy** | Direct multi-step regression | Iterated 1-step linear model |
| **Persistence Assumption** | Robust to non-stationary persistence | Sensitive to unit roots in companion matrix |
| **State Dependence** | Built-in via interaction terms ($F(s_t) x_t$) | Requires non-linear TVP/Threshold VAR |
| **Efficiency at Long Horizons** | Higher variance at large $H$ | More efficient if VAR is correctly specified |
| **Data Requirements** | Long time series needed for high $H$ | Moderate samples suffice |
