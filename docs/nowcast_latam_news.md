> 🇬🇧 English · 🇪🇸 [Español](es/nowcast_latam_news.md)

# Latin America Real-Time Nowcasting & News Decomposition

Macroeconomic monitoring in emerging market economies—particularly across Latin America—presents unique empirical hurdles: data series arrive with staggered publication lags (the *ragged edge*), historical figures undergo frequent statistical revisions, and releases from central banks and national statistical institutes occur at mixed sampling frequencies (monthly industrial production, retail sales, and surveys versus quarterly GDP).

`puremacro.nowcast` provides an institutional-grade, real-time nowcasting and forecast evaluation pipeline designed for central banks, fiscal authorities, and macroeconomic research desks. The framework brings together:

1. **High-Level Real-Time Nowcast Orchestrator (`realtime_nowcast`)**: Unified estimation integrating Latin American central bank and statistical office data connectors (Banxico, INEGI, BCB, BCCh, and ALFRED/St. Louis Fed) across historical data vintages.
2. **Kalman Dynamic Factor Model (`DynamicFactorModel`)**: Expectation-Maximization and exact Kalman smoothing for unbalanced panels with arbitrary missing data patterns (Doz, Giannone, and Reichlin 2011; Bańbura and Modugno 2014).
3. **Analytical News-versus-Noise Decomposition (`banbura_modugno_news`)**: Exact attribution of revisions in the headline nowcast to specific indicator surprises and vintage revisions, mathematically guaranteed to close to numerical precision ($|\Delta \hat{y} - \sum \text{impact}| < 10^{-10}$).
4. **Professional Forecast Evaluation & Fan Charts (`pit_uniformity_test`, `fan_chart`)**: Density calibration via the Berkowitz (2001) Likelihood Ratio test, Kolmogorov-Smirnov test, and central bank quantile fan charts adhering to frozen result dataclasses.

All computations are 100% pure NumPy, SciPy, and pandas, executing client-side in Pyodide or server-side without external compiled binaries or network dependencies.

---

## 1. Theoretical & Mathematical Foundations

### 1.1 Dynamic Factor Model in State Space Form

Let $X_t \in \mathbb{R}^N$ be a standardized monthly panel of $N$ macroeconomic indicators at date $t = 1, \dots, T$. The indicators are assumed to be driven by a low-dimensional vector of $r \ll N$ unobserved common dynamic factors $F_t \in \mathbb{R}^r$ and an idiosyncratic disturbance vector $\xi_t \in \mathbb{R}^N$:

$$X_t = \Lambda F_t + \xi_t, \qquad \xi_t \sim \text{i.i.d.} \, \mathcal{N}(0, R)$$

where $\Lambda \in \mathbb{R}^{N \times r}$ is the factor loadings matrix, and $R = \operatorname{diag}(\sigma_1^2, \dots, \sigma_N^2)$ is diagonal (approximate factor structure). The latent factors follow a VAR($p$) transition process:

$$F_t = A_1 F_{t-1} + A_2 F_{t-2} + \dots + A_p F_{t-p} + u_t, \qquad u_t \sim \text{i.i.d.} \, \mathcal{N}(0, Q)$$

In state space companion form with state vector $\alpha_t = [F_t^\top, F_{t-1}^\top, \dots, F_{t-p+1}^\top]^\top \in \mathbb{R}^{rp}$:

$$\alpha_t = T \alpha_{t-1} + R_{\eta} \eta_t, \qquad \eta_t \sim \mathcal{N}(0, Q)$$

$$X_t = Z_t \alpha_t + \xi_t$$

where $Z_t = W_t \begin{bmatrix} \Lambda & 0_{N \times r(p-1)} \end{bmatrix}$, and $W_t$ is a diagonal selection matrix whose $i$-th diagonal element is $1$ if indicator $i$ is observed at month $t$, and $0$ if missing (accommodating arbitrary ragged edges and publication delays).

### 1.2 Bańbura & Modugno (2014) Analytical News Decomposition

When a forecaster updates a nowcast from data vintage $v-1$ (information set $\Omega_{v-1}$) to updated vintage $v$ ($\Omega_v$), the nowcast of target variable $y_{t^*}$ updates by:

$$\Delta \hat{y}_{t^*|v} = \mathbb{E}[y_{t^*} \mid \Omega_v] - \mathbb{E}[y_{t^*} \mid \Omega_{v-1}]$$

Bańbura and Modugno (2014) prove that because the Kalman filter and smoother represent linear projections in Gaussian state space models, this update can be decomposed analytically into the sum of contributions from newly released data points (**news innovations**) and historical data revisions:

$$\Delta \hat{y}_{t^*|v} = \sum_{j \in \mathcal{I}_{\text{new}}} \omega_j \cdot \underbrace{\left( x_{j, t_j} - \mathbb{E}[x_{j, t_j} \mid \Omega_{v-1}] \right)}_{\text{news innovation } I_{j, v}} + \sum_{k \in \mathcal{I}_{\text{rev}}} \omega_k \cdot \underbrace{\left( x_{k, t_k}^{(v)} - x_{k, t_k}^{(v-1)} \right)}_{\text{data revision } R_{k, v}}$$

$$\Delta \hat{y}_{t^*|v} = \sum_{j \in \mathcal{I}_{\text{new}}} \text{impact}_j + \sum_{k \in \mathcal{I}_{\text{rev}}} \text{impact}_k$$

The weight $\omega_j$ reflects both the statistical correlation between factor loadings and the target variable, and the Kalman gain of the new observation relative to all other contemporaneous indicators. In `puremacro`, this decomposition is exact down to machine precision:

$$|\Delta \hat{y}_{t^*|v} - \text{total\_impact}| < 10^{-10}$$

### 1.3 Mankiw-Shapiro (1986) News vs. Noise Hypothesis

Data revisions $R_{t, v} = y_{t}^{(v)} - y_{t}^{(v-1)}$ are characterized as:
- **News**: Initial releases are optimal forecasts based on available information; revisions are orthogonal to early releases ($\operatorname{Cov}(y_t^{(v-1)}, R_{t, v}) = 0$).
- **Noise**: Initial releases equal true values plus classical measurement error; revisions are correlated with early releases ($\operatorname{Cov}(y_t^{(v)}, R_{t, v}) = 0$).

`realtime_nowcast` computes the Mankiw-Shapiro regression $R_{t, v} = \alpha + \beta y_t^{(v-1)} + \varepsilon_t$ to diagnose whether revision noise dominates publication reporting.

### 1.4 Probability Integral Transform (PIT) & Berkowitz LR Test

For density nowcasts $\hat{f}_{t|t-h}(y_t)$, let $p_t$ denote the empirical PIT values:

$$p_t = \int_{-\infty}^{y_t} \hat{f}_{t|t-h}(u) \, du = \Phi\left( \frac{y_t - \hat{\mu}_t}{\hat{\sigma}_t} \right)$$

If the forecast density is well-calibrated and temporally independent, $p_t \sim \text{i.i.d.} \, \mathcal{U}(0, 1)$. Following Berkowitz (2001), the PIT sequence is transformed via the standard normal quantile function:

$$z_t = \Phi^{-1}(p_t)$$

Under the null hypothesis of perfect calibration and absence of autocorrelation, $z_t \sim \text{i.i.d.} \, \mathcal{N}(0, 1)$. Berkowitz specifies the AR(1) diagnostic model:

$$z_t - \mu = \rho (z_{t-1} - \mu) + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)$$

The null hypothesis $H_0: \mu = 0, \sigma^2 = 1, \rho = 0$ is tested via the Likelihood Ratio statistic:

$$\text{LR} = -2 \left[ \ln L(\mu=0, \sigma^2=1, \rho=0) - \ln L(\hat{\mu}, \hat{\sigma}^2, \hat{\rho}) \right] \sim \chi^2(3)$$

In addition, the two-sided Kolmogorov-Smirnov test evaluates the empirical cumulative distribution function against $\mathcal{U}(0, 1)$.

---

## 2. API Architecture & Result Objects

### Core Functions & Classes

| Object | Type | Description |
|---|---|---|
| `DynamicFactorModel` | Class | High-level DFM estimator with Kalman smoothing, missing-data imputation, and factor extraction. |
| `DynamicFactorModelResult` | Frozen Dataclass | Estimation results (`factors`, `loadings`, `A`, `H`, `Q`, `loglik`, `X_filled`, `.plot()`). |
| `banbura_modugno_news` | Function | Computes Bańbura & Modugno (2014) analytical news decomposition between two data vintages. |
| `NewsDecompositionResult` | Frozen Dataclass | Holds `forecast_old`, `forecast_new`, `revision`, `impact_releases`, `impact_revisions`, and `news_table`. |
| `realtime_nowcast` | Function | Latin American real-time orchestrator connecting country specs (MEX, BRA, CHL, USA) to vintage panels. |
| `RealtimeNowcastResult` | Frozen Dataclass | Holds point nowcast, standard errors, factor trajectories, news decomposition, and fan charts. |
| `pit_uniformity_test` | Function | Computes Berkowitz (2001) LR test, KS test, and PIT histogram. |
| `PITUniformityResult` | Frozen Dataclass | Holds test statistics (`lr_stat`, `lr_pvalue`, `ks_stat`, `ks_pvalue`), `is_uniform`, and diagnostic plots. |
| `fan_chart` | Function | Generates layered central bank quantile forecast ribbons. |
| `FanChartResult` | Frozen Dataclass | Holds quantile series, intervals, and rendering exports (`.plot()`, `.to_markdown()`, `.to_latex()`, `.to_typst()`). |

### Country Specifications (`COUNTRY_SPECS`)

`puremacro.nowcast.realtime_nowcast.COUNTRY_SPECS` configures institutional parameters for major economies:

```python
from puremacro.nowcast.realtime_nowcast import COUNTRY_SPECS

# Mexico: Banco de México & INEGI
COUNTRY_SPECS["MEX"]
# {'name': 'Mexico', 'central_bank': 'Banco de México (Banxico) / INEGI',
#  'default_target': 'gdp', 'palette': 'banxico'}

# Brazil: Banco Central do Brasil
COUNTRY_SPECS["BRA"]
# {'name': 'Brazil', 'central_bank': 'Banco Central do Brasil (BCB)',
#  'default_target': 'gdp', 'palette': 'bcb'}

# Chile: Banco Central de Chile
COUNTRY_SPECS["CHL"]

# United States: Federal Reserve (ALFRED)
COUNTRY_SPECS["USA"]
```

---

## 3. End-to-End Walkthrough

### 3.1 Dynamic Factor Model Estimation with Ragged Edges

```python
import numpy as np
import pandas as pd
from puremacro.nowcast import DynamicFactorModel

# 1. Generate synthetic monthly panel with ragged edge (T=120, N=8)
rng = np.random.default_rng(42)
dates = pd.date_range("2014-01-01", periods=120, freq="MS")
f_latent = np.zeros(120)
for t in range(1, 120):
    f_latent[t] = 0.75 * f_latent[t - 1] + rng.normal(scale=0.5)

loadings = rng.uniform(0.6, 1.4, size=8)
X_raw = f_latent[:, None] @ loadings[None, :] + rng.normal(scale=0.3, size=(120, 8))
df_panel = pd.DataFrame(X_raw, index=dates, columns=[f"ind_{i+1}" for i in range(8)])

# Ragged edge: last month missing for indicators 4 to 8
df_panel.iloc[-1, 3:] = np.nan
# Previous month missing for indicators 7 and 8
df_panel.iloc[-2, 6:] = np.nan

# 2. Fit Dynamic Factor Model
dfm = DynamicFactorModel(n_factors=1, p=1)
dfm.fit(df_panel)
res = dfm.result_

print(f"Log-Likelihood: {res.loglik:.2f}")
print(f"Common factors shape: {res.factors.shape}")
print(f"Imputed panel shape: {res.X_filled.shape}")

# Inspect factor loadings
print(res.loadings)
```

### 3.2 Bańbura & Modugno Analytical News Decomposition

```python
from puremacro.nowcast import banbura_modugno_news

# Create vintage v-1 and updated vintage v
panel_v0 = df_panel.copy()
panel_v1 = df_panel.copy()

# Vintage v publishes new releases for indicators 4 and 5
panel_v1.iloc[-1, 3] = panel_v0.iloc[-2, 3] + 0.45  # new release
panel_v1.iloc[-1, 4] = panel_v0.iloc[-2, 4] - 0.20  # new release
# Vintage v revises indicator 1 at date t-1
panel_v1.iloc[-2, 0] += 0.15                         # historical revision

# Target: indicator 1 at the latest date
news_res = banbura_modugno_news(
    model=dfm,
    old_vintage=panel_v0,
    new_vintage=panel_v1,
    target_series="ind_1",
    target_period=dates[-1],
)

print(news_res.summary())
print(f"Decomposition Error: {news_res.decomposition_error:.2e}")
assert news_res.decomposition_error < 1e-10
```

### 3.3 Institutional Real-Time Nowcasting (`realtime_nowcast`)

```python
from puremacro.fetch.realtime import VintagePanel
from puremacro.nowcast import realtime_nowcast

# Combine historical vintages into a real-time panel (or load via load_realtime_cartridge)
rows = []
for d in dates:
    for col in df_panel.columns:
        val0 = panel_v0.loc[d, col]
        if not np.isnan(val0):
            rows.append({
                "country": "MEX", "variable": col, "date": d,
                "vintage": pd.Timestamp("2023-11-01"), "value": float(val0),
                "provider": "banxico", "series_id": col, "units": "rate",
            })
        val1 = panel_v1.loc[d, col]
        if not np.isnan(val1):
            rows.append({
                "country": "MEX", "variable": col, "date": d,
                "vintage": pd.Timestamp("2023-12-01"), "value": float(val1),
                "provider": "banxico", "series_id": col, "units": "rate",
            })

vp = VintagePanel(pd.DataFrame(rows))

# Run nowcast orchestrator for Mexico using Banxico styling
result = realtime_nowcast(
    country="MEX",
    panel=vp,
    target_variable="ind_1",
    target_period=dates[-1],
    method="dfm",
    n_factors=1,
)

print(f"Country: {result.country}")
print(f"Nowcast: {result.nowcast:.4f} (±{1.96 * result.forecast_sd:.4f})")
print(f"Palette: {result.palette}")

# Print news breakdown table
if result.news_decomposition:
    print(result.news_decomposition.news_table)
```

### 3.4 Forecast Density Evaluation: PIT Uniformity & Berkowitz Test

```python
from puremacro.nowcast import pit_uniformity_test

# Generate realized series and density forecasts
T_eval = 80
y_real = rng.normal(loc=1.0, scale=0.5, size=T_eval)
mu_forecast = np.full(T_eval, 1.0)
sigma_forecast = np.full(T_eval, 0.5)

# Evaluate calibration
pit_res = pit_uniformity_test(
    realised=y_real,
    mu=mu_forecast,
    sigma=sigma_forecast,
    n_bins=10,
)

print(pit_res.summary())
print(f"Berkowitz LR p-value : {pit_res.lr_pvalue:.4f}")
print(f"Kolmogorov-Smirnov p : {pit_res.ks_pvalue:.4f}")
print(f"Calibrated (Uniform) : {pit_res.is_uniform}")
```

### 3.5 Central Bank Quantile Fan Charts

```python
from puremacro.nowcast import fan_chart

history = pd.Series([1.2, 1.5, 1.8, 1.6, 2.0, 2.3], index=["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2"])
forecast_mean = pd.Series([2.2, 2.1, 2.0, 1.9], index=["2025Q3", "2025Q4", "2026Q1", "2026Q2"])
forecast_sd = [0.25, 0.35, 0.45, 0.55]

fc = fan_chart(
    history=history,
    forecast_mean=forecast_mean,
    forecast_sd=forecast_sd,
    levels=(0.50, 0.70, 0.90),
    palette="banxico",
)

# Export tables
print(fc.to_markdown())
# Or render Matplotlib plot
# ax = fc.plot(title="Banxico Headline GDP Growth Projection")
```

---

## 4. Exporting to Publications & Reports

All nowcast result objects include full export pipelines into Markdown, $\LaTeX$ (with `booktabs`), and Typst tables:

```python
# Fan chart quantiles table
md_table = fc.to_markdown()
tex_table = fc.to_latex()
typst_table = fc.to_typst()
```

---

## References

1. Bańbura, M. and Modugno, M. (2014). "Maximum likelihood estimation of factor models on datasets with arbitrary pattern of missing data." *Journal of Applied Econometrics*, 29(1), 133–160.
2. Berkowitz, J. (2001). "Testing density forecasts, with applications to risk management." *Journal of Business & Economic Statistics*, 19(4), 465–474.
3. Doz, C., Giannone, D., and Reichlin, L. (2011). "A two-step estimator for large approximate dynamic factor models based on Kalman filtering." *Journal of Econometrics*, 164(1), 188–205.
4. Mankiw, N. G. and Shapiro, M. D. (1986). "News or noise: An analysis of GNP revisions." *Survey of Current Business*, 66(5), 20–25.
