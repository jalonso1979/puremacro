> 🇬🇧 English · 🇪🇸 [Español](es/bvar_sv.md)

# Bayesian VAR with Stochastic Volatility (BVAR-SV)

`puremacro.var.bvar_sv` implements Bayesian Vector Autoregressions with time-varying residual volatility (BVAR-SV) in pure NumPy/SciPy, following the state-space and MCMC methodologies of **Carriero, Clark, and Marcellino (2016, 2019)**, **Kim, Shephard, and Chib (1998)**, and **Carter and Kohn (1994)**.

Macroeconomic time series are characterized by pronounced regime shifts in volatility—such as the Great Moderation of 1984–2007, the acute shock of the 2007–2008 Global Financial Crisis, and the pandemic disruption of 2020. Standard homoskedastic VAR models distort parameter estimates and produce miscalibrated predictive densities by conflating changes in shock sizes with changes in transmission mechanisms.

BVAR-SV decomposes structural dynamics into constant autoregressive transmission coefficients and **time-varying stochastic log-volatilities**, providing volatility-conditioned impulse responses, posterior-predictive fan charts and honest predictive-density evaluation.

---

## 1. Econometric Framework

### 1.1 The State-Space Representation

Consider an $n$-variable vector autoregression with $p$ lags:

$$y_t = c + \sum_{l=1}^p A_l y_{t-l} + u_t, \quad t = 1, \dots, T$$

The reduced-form residuals $u_t$ follow a time-varying covariance structure:

$$u_t = A^{-1} D_t^{1/2} \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, I_n)$$

where:
1. $A$ is a lower-triangular contemporaneous matrix with unit diagonal capturing contemporaneous structural relations:
   $$A = \begin{pmatrix} 1 & 0 & \dots & 0 \\ a_{2,1} & 1 & \dots & 0 \\ \vdots & \vdots & \ddots & 0 \\ a_{n,1} & a_{n,2} & \dots & 1 \end{pmatrix}$$
2. $D_t = \text{diag}\left( \exp(h_{1,t}), \dots, \exp(h_{n,t}) \right)$ contains time-varying residual variances.
3. The reduced-form residual covariance matrix at date $t$ is:
   $$\Sigma_t = A^{-1} D_t (A^{-1})'$$
4. The unobserved log-volatility states $h_{i,t}$ follow independent stationary $\text{AR}(1)$ processes for each variable $i = 1, \dots, n$:
   $$h_{i,t} - \mu_i = \phi_i (h_{i,t-1} - \mu_i) + \sigma_{h,i} \eta_{i,t}, \quad \eta_{i,t} \sim \mathcal{N}(0, 1), \quad |\phi_i| < 1$$
   with the stationary initial condition $h_{i,1} \sim \mathcal{N}\left(\mu_i, \, \frac{\sigma_{h,i}^2}{1 - \phi_i^2}\right)$.

### 1.2 Priors

- VAR coefficients $\beta = \text{vec}(B')$: independent normal prior. With `minnesota_prior=True` (default) the Minnesota prior of the package (`lambda1`, `lambda2`, `lambda3`; own first lag centred at 1) is used; with `minnesota_prior=False` every lag coefficient gets a diffuse $\mathcal{N}(0, 100^2)$ prior. In both modes the intercepts get $\mathcal{N}(0, \texttt{intercept\_prior\_std}^2)$.
- Contemporaneous coefficients: $a_{i,j} \sim \mathcal{N}(0, 100)$.
- Volatility hyper-parameters (fixed hyper-priors): $\mu_i \sim \mathcal{N}(0, 10)$, $\phi_i \sim \mathcal{N}(0.85, 0.1)$ truncated to $|\phi_i| < 0.999$, $\sigma_{h,i}^2 \sim \text{IG}(2, 0.05)$. The offset in $\log(\nu_{i,t}^2 + c)$ is $c = 10^{-6}$.

---

## 2. The MCMC Gibbs Sampling Algorithm

The model is estimated via a Gibbs sampler across sequential blocks:

1. **Step 1: VAR Coefficients ($\beta$)**
   Conditional on $A$ and the volatility paths $D_{1:T}$, $\beta$ is drawn **jointly** from its exact conditional posterior
   $$\beta \mid A, h, Y \sim \mathcal{N}(\bar{\beta}, \bar{V}_\beta), \qquad \bar{V}_\beta^{-1} = V_0^{-1} + \sum_t \Sigma_t^{-1} \otimes x_t x_t'$$
   using the full $(nk \times nk)$ precision-weighted GLS precision ($k = 1 + np$). This is exact but $O((nk)^3)$ per sweep and is meant for small and medium systems; it is *not* the equation-by-equation triangular algorithm of Carriero, Clark and Marcellino (2019). Draws are accepted only if the companion matrix is stable (posterior truncated to the stationary region). After 50 consecutive rejections the previous $B$ is kept; such sweeps are counted in `n_stuck_iterations` (with `n_unstable_rejections` candidates rejected) and a `UserWarning` is emitted.
2. **Step 2: Contemporaneous Triangular Coefficients ($A$)**
   Each row $i$ of $A$ is sampled equation-by-equation (Cogley and Sargent 2005; Primiceri 2005) via a linear regression of the residual $\hat{u}_{i,t}$ on the preceding residuals $\hat{u}_{1:i-1,t}$, weighted by $\exp(-h_{i,t}/2)$.
3. **Step 3: Kim-Shephard-Chib (1998) 7-Component Mixture Sampler**
   Transforming squared standardized residuals yields a non-Gaussian linear measurement equation:
   $$y_{i,t}^* \equiv \log\left( (A \hat{u}_t)_i^2 + c \right) = h_{i,t} + \log(\varepsilon_{i,t}^2)$$
   where $\log(\varepsilon_{i,t}^2) \sim \log(\chi_1^2)$ is approximated by a 7-component mixture of normal distributions with parameters $(q_k, m_k, v_k^2)$ from Kim, Shephard, and Chib (1998, Table 4). The mixture indicator states $s_{i,t} \in \{1, \dots, 7\}$ are sampled conditionally on $h_{i,t}$.
4. **Step 4: Forward-Filtering Backward-Sampling (FFBS) for Volatility States ($h_{i,t}$)**
   Conditional on the mixture indicators, the state-space is conditionally Gaussian. Carter and Kohn (1994) FFBS draws the full state sequence $h_{i, 1:T}$ jointly in a single backward sweep.
5. **Step 5: Volatility Parameters $(\mu_i, \phi_i, \sigma_{h,i}^2)$**
   $\mu_i$ is drawn from its conjugate normal conditional and $\sigma_{h,i}^2$ from its conjugate inverse-gamma conditional. $\phi_i$ uses the independence Metropolis-Hastings step of Kim, Shephard and Chib (1998, §3.3): the proposal is the Gaussian conditional posterior of the AR(1) regression on $h_{i,2:T}$ and the acceptance probability is the ratio of stationary initial-state densities
   $$\log \frac{p(h_{i,1} \mid \phi')}{p(h_{i,1} \mid \phi)} = \tfrac{1}{2}\log(1-\phi'^2) - \tfrac{1}{2}\log(1-\phi^2) + \tfrac{1}{2}(\phi'^2 - \phi^2)\frac{(h_{i,1}-\mu_i)^2}{\sigma_{h,i}^2}.$$
6. **Step 6: Multi-Chain Gelman-Rubin Convergence Diagnostics ($\hat{R}$)**
   `n_chains` chains are run (sequentially); chains after the first start from jittered initial values (log-volatility level, $\phi_i$, $\sigma_{h,i}^2$). Each chain is split in two halves and the split-chain $\hat{R}$ is computed for every VAR coefficient, every $a_{i,j}$, the mean log-volatility of each variable and each $(\mu_i, \phi_i, \sigma_{h,i})$. Values of $\hat{R} < 1.1$ indicate adequate mixing; $\hat{R}$ is a stochastic diagnostic, so increase `n_draws` / `n_burn` when it fails.

---

## 3. Volatility-Conditioned Impulse Responses

Because the impact matrix $B_t = A^{-1} D_t^{1/2}$ varies across time, structural shock propagation differs depending on the volatility regime.

`BVAR_SVResult.irf(horizon=20, t_idx=..., ci=0.9)` conditions the impulse responses on the volatility state at a specific date $t^*$ (an index into the effective sample, i.e. observation $t^* + p$):
- **High-Volatility Regime (e.g., 2008Q4 or 2020Q2)**: Evaluates impulse responses when financial or macroeconomic uncertainty is elevated.
- **Tranquil Regime (e.g., mid-1990s Great Moderation)**: Evaluates transmission in quiet macroeconomic conditions.

The result is a `BVAR_SV_IRF` array of shape $(H+1, n, n)$ (`[horizon, response, shock]`, Cholesky ordering) holding the posterior median, with `.lower`, `.upper` and the full `.draws` $(D, H+1, n, n)$. `to_frame(target_idx=None, shock_idx=None, names=None)` returns a tidy table; each index acts as an independent filter (pass one, both or neither).

---

## 4. Forecasts, Fan Charts and Predictive Scores

- `forecast(horizon, ci=0.9, seed=None)` simulates one posterior-predictive path per retained draw: the log-volatilities are propagated with the AR(1) law of motion, structural shocks are drawn, and the VAR is iterated forward, so parameter, volatility and shock uncertainty are all integrated out. The returned `BVAR_SVForecast` exposes `paths` $(D, H, n)$, `h_paths`, `median`, `mean`, `lower`, `upper`, `quantile(q)`, a dated `index` (when the data carry a regular `DatetimeIndex`), `to_frame()` and `plot(levels=(0.5, 0.8, 0.95))` fan charts.
- `log_scores` / `predictive_log_score()` is the **in-sample** log pointwise predictive density (lppd): $\sum_t \log \frac{1}{D}\sum_d p(y_t \mid \beta^{(d)}, A^{(d)}, h_t^{(d)})$ evaluated with the *smoothed* volatility draws, which condition on the whole estimation sample. It is a fit measure, not a forecast evaluation.
- `log_score(holdout)` is the **out-of-sample** log predictive score of observations that follow the estimation sample: for every draw the conditional mean uses the realised lags of $y_{T+j}$ and the volatility is projected from the end-of-sample state with $h_{T+j} = \mu + \phi (h_{T+j-1} - \mu) + \sigma_h \eta$. Parameters are not re-estimated and the volatility is not re-filtered on hold-out data. `log_score()` without arguments returns the in-sample lppd.

---

## 5. Basic & Advanced Usage

### Estimating a BVAR-SV Model

```python
import numpy as np
import pandas as pd
from puremacro.var.bvar_sv import bvar_sv

# 1. Synthetic quarterly dataset (Output, Inflation, Rate) with a volatility spike
rng = np.random.default_rng(42)
T = 180
dates = pd.date_range("1975-01-01", periods=T, freq="QE")

data = np.zeros((T, 3))
for t in range(1, T):
    vol_scale = 2.5 if 130 <= t <= 145 else 1.0  # Volatility spike
    data[t] = 0.6 * data[t-1] + vol_scale * 0.5 * rng.standard_normal(3)

df = pd.DataFrame(data, index=dates, columns=["Output", "Inflation", "Rate"])

# Hold out the last 12 quarters for out-of-sample density evaluation
train, test = df.iloc[:168], df.iloc[168:]

# 2. Fit the BVAR-SV with the Gibbs sampler (n_draws is per chain: 2 x 2000 pooled draws)
res_sv = bvar_sv(
    data=train,
    lags=2,
    n_draws=2000,
    n_burn=1000,
    n_chains=2,
    minnesota_prior=True,
    lambda1=0.2,
    lambda2=0.5,
    seed=123,
)

# 3. Inspect convergence diagnostics & posterior summary
print(res_sv.summary())
print(f"Max split-chain R-hat: {res_sv.max_rhat:.3f}")
assert res_sv.max_rhat < 1.1, "Chains must pass Gelman-Rubin convergence (< 1.1)"

# 4. Volatility-conditioned IRFs: crisis (t_idx=135) vs calm period (t_idx=50)
irf_crisis = res_sv.irf(horizon=16, t_idx=135, ci=0.90)
irf_calm = res_sv.irf(horizon=16, t_idx=50, ci=0.90)

print(f"Peak Output response (crisis): {irf_crisis.median[:, 0, 0].max():.3f}")
print(f"Peak Output response (calm)  : {irf_calm.median[:, 0, 0].max():.3f}")
irf_table = irf_crisis.to_frame(shock_idx=0, names=res_sv.names)  # all responses to the Output shock

# 5. Predictive densities: in-sample lppd vs out-of-sample hold-out score
print(f"In-sample lppd (smoothed volatilities): {res_sv.predictive_log_score():.2f}")
print(f"Out-of-sample log score, 12 held-out quarters: {res_sv.log_score(test, seed=0):.2f}")

# 6. Posterior-predictive forecasts with fan charts
fc = res_sv.forecast(horizon=12, ci=0.90, seed=1)
fan_fig = fc.plot(levels=(0.5, 0.8, 0.95))
coverage = np.mean((test.values >= fc.lower) & (test.values <= fc.upper))
print(f"Share of held-out observations inside the 90% predictive band: {coverage:.2f}")

# 7. Volatility paths, conditional SDs and IRF panels; export tables
fig = res_sv.plot(t_idx=135, shock_idx=0, target_idx=0)
latex_table = res_sv.to_latex()
md_table = res_sv.to_markdown()
```

---

## 6. Full API Specification

### `bvar_sv`

```text
bvar_sv(
    data: pd.DataFrame | np.ndarray,
    lags: int = 4,
    n_draws: int = 2000,
    n_burn: int = 1000,
    minnesota_prior: bool = True,
    seed: int | None = None,
    *,
    lambda1: float = 0.2,
    lambda2: float = 0.5,
    lambda3: float = 1.0,
    intercept_prior_std: float = 1e3,
    thin: int = 1,
    n_chains: int = 2,
    p: int | None = None,
) -> BVAR_SVResult
```

#### Parameters:
- `data`: $(T, n)$ dataset of endogenous time series variables; must be finite (NaN / inf raise `ValueError`).
- `lags` / `p`: Autoregressive lag order $p$ (default `4`). The effective sample $T - p$ must exceed the number of regressors per equation $k = 1 + np$.
- `n_draws`: Post-burn-in MCMC draws retained **per chain** (default `2000`, minimum `2`); the result pools `n_chains * n_draws` draws.
- `n_burn`: Initial burn-in iterations discarded per chain (default `1000`).
- `minnesota_prior`: Whether to apply Minnesota prior shrinkage (default `True`); otherwise a diffuse $\mathcal{N}(0, 100^2)$ prior on the lag coefficients.
- `seed`: Reproducibility seed for the MCMC sampler.
- `lambda1`: Overall Minnesota shrinkage hyperparameter (default `0.2`).
- `lambda2`: Cross-variable shrinkage hyperparameter (default `0.5`).
- `lambda3`: Lag-decay exponent (default `1.0`).
- `intercept_prior_std`: Prior standard deviation of the intercepts (default `1e3`), honoured in both prior modes.
- `thin`: Thinning interval (default `1`, minimum `1`): one draw is retained every `thin` sweeps until `n_draws` are collected.
- `n_chains`: Number of MCMC chains run sequentially for the split-chain Gelman-Rubin diagnostics (default `2`, minimum `1`).

A `UserWarning` is emitted when some sweep exhausted the stability retry budget (see Step 1).

---

## 7. Result Interface

The returned `BVAR_SVResult` container provides:

- **Attributes** (`D = n_chains * n_draws` pooled draws):
  - `beta_draws`: Posterior draws of VAR coefficients $(D, 1+np, n)$; `A_draws` $(D, p, n, n)$ and `intercept_draws` $(D, n)$ are derived views.
  - `h_draws`: Posterior draws of latent log-volatility paths $(D, T_{eff}, n)$.
  - `a_draws`: Posterior draws of the contemporaneous impact matrix $A$ $(D, n, n)$.
  - `mu_draws`, `phi_draws`, `sigma_h_draws`: Volatility AR(1) parameter draws $(D, n)$.
  - `r_hat` / `rhat`: Dictionary of split-chain Gelman-Rubin statistics (`beta_max`, `beta_mean`, `a_max`, `h_max`, `h_mean`, `h_<name>_mean`, `mu_<name>`, `phi_<name>`, `sigma_h_<name>`, `max`).
  - `max_rhat`: Maximum $\hat{R}$ observed (`r_hat['max']`).
  - `log_scores`: In-sample log pointwise predictive density per observation $(T_{eff},)$ (smoothed volatilities).
  - `n_draws`, `n_burn`, `n_chains`, `n_total_draws`: Sampler settings and the pooled draw count.
  - `n_unstable_rejections`, `n_stuck_iterations`: Stability-rejection accounting (Step 1).
- **Methods**:
  - `.irf(horizon=20, t_idx=-1, ci=0.9)`: Returns `BVAR_SV_IRF` with `.median`, `.lower`, `.upper`, `.draws` and `.to_frame(target_idx=None, shock_idx=None, names=None)`.
  - `.forecast(horizon=8, ci=0.9, seed=None)`: Returns `BVAR_SVForecast` with posterior-predictive `paths`, bands, `to_frame()` and `.plot()` fan charts.
  - `.log_score(holdout=None, point_by_point=False, seed=None)`: Out-of-sample log predictive score on a hold-out sample; without `holdout`, the in-sample lppd.
  - `.predictive_log_score(point_by_point=False)`: In-sample lppd (total or pointwise).
  - `.gelman_rubin()`: The `r_hat` dictionary.
  - `.summary(ci=0.9)`: Text report with sampler settings, convergence status, the in-sample lppd, stability-rejection counts (if any) and, for every variable, the posterior median, central credible interval and $\hat{R}$ of $\mu_i$, $\phi_i$, $\sigma_{h,i}$.
  - `.plot(t_idx=-1, horizon=20, ci=0.9, shock_idx=0, target_idx=None, ax=None)`: Three panels — log-volatility paths and conditional standard deviations with credible bands for every variable (or only `target_idx`), and the volatility-conditioned IRF of `target_idx` (first variable by default) to `shock_idx`.
  - `.to_frame()`, `.to_markdown()`, `.to_latex()`, `.to_typst()`: Posterior mean / SD table of the volatility parameters per variable with the $\hat{R}$ of its mean log-volatility.

### Limitations

- The VAR coefficients are drawn jointly with an $(nk \times nk)$ Cholesky factorisation per sweep; very large systems (say $nk > 300$) will be slow. The CCM (2019) triangular equation-by-equation sampler is not implemented.
- The hold-out `log_score` projects volatility from the end of the estimation sample and does not re-estimate or re-filter on hold-out data; for a fully recursive evaluation, refit the model at each origin.
- Hyper-priors of the volatility processes are fixed (Section 1.2) and not user-configurable.
