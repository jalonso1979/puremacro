> 🇬🇧 English · 🇪🇸 [Español](es/bvar_sv.md)

# Bayesian VAR with Stochastic Volatility (BVAR-SV)

`puremacro.var.bvar_sv` implements Bayesian Vector Autoregressions with time-varying residual volatility (BVAR-SV), following the state-space and MCMC methodologies of **Carriero, Clark, and Marcellino (2016, 2019)**, **Kim, Shephard, and Chib (1998)**, and **Carter and Kohn (1994)**.

Macroeconomic time series are characterized by pronounced regime shifts in volatility—such as the Great Moderation of 1984–2007, the acute shock of the 2007–2008 Global Financial Crisis, and the pandemic disruption of 2020. Standard homoskedastic VAR models distort parameter estimates and produce miscalibrated predictive densities by conflating changes in shock sizes with changes in transmission mechanisms.

BVAR-SV decomposes structural dynamics into constant autoregressive transmission coefficients and **time-varying stochastic log-volatilities**, providing accurate predictive density distributions and volatility-conditioned impulse response functions.

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
   with unconditional initial distribution $h_{i,0} \sim \mathcal{N}\left(\mu_i, \, \frac{\sigma_{h,i}^2}{1 - \phi_i^2}\right)$.

---

## 2. The MCMC Gibbs Sampling Algorithm

The model is estimated via pure NumPy/SciPy Gibbs sampling across sequential blocks:

1. **Step 1: VAR Lag Coefficients ($\beta$) with Minnesota Shrinkage**  
   Conditional on $A$ and volatility paths $D_{1:T}$, the VAR is transformed via precision-weighted Generalized Least Squares (GLS). Minnesota prior shrinkage shrinks lag coefficients toward random walks or white noise:
   $$\beta \mid A, h, Y \sim \mathcal{N}(\bar{\beta}, \bar{V}_\beta)$$
2. **Step 2: Contemporaneous Triangular Coefficients ($A$)**  
   Using the triangular decomposition of Carriero, Clark, and Marcellino (2016, 2019), each row $i$ of $A$ is sampled equation-by-equation via linear regression of residual $\hat{u}_{i,t}$ on preceding residuals $\hat{u}_{1:i-1,t}$ weighted by $\exp(-h_{i,t})$.
3. **Step 3: Kim-Shephard-Chib (1998) 7-Component Mixture Sampler**  
   Transforming squared standardized residuals yields a non-Gaussian linear measurement equation:
   $$y_{i,t}^* \equiv \log\left( (A \hat{u}_t)_i^2 + c \right) = h_{i,t} + \log(\varepsilon_{i,t}^2)$$
   where $\log(\varepsilon_{i,t}^2) \sim \log(\chi_1^2)$ is approximated by a 7-component mixture of normal distributions with parameters $(q_k, m_k, v_k^2)$ from Kim, Shephard, and Chib (1998, Table 4). The mixture indicator states $s_{i,t} \in \{1, \dots, 7\}$ are sampled conditionally on $h_{i,t}$.
4. **Step 4: Forward-Filtering Backward-Sampling (FFBS) for Volatility States ($h_{i,t}$)**  
   Conditional on the mixture indicators, the state-space is conditionally Gaussian. Carter and Kohn (1994) FFBS draws the full state sequence $h_{i, 1:T}$ jointly in a single backward sweep.
5. **Step 5: Volatility Parameters $(\mu_i, \phi_i, \sigma_{h,i}^2)$**  
   Sampled conjugate Normal-Inverse-Gamma draws conditional on $h_{i, 1:T}$.
6. **Step 6: Multi-Chain Gelman-Rubin Convergence Diagnostics ($\hat{R}$)**  
   Computes split-chain Gelman-Rubin $\hat{R}$ across parameters. Values of $\hat{R} < 1.1$ indicate adequate chain mixing and posterior convergence.

---

## 3. Volatility-Conditioned Impulse Responses

Because the impact matrix $B_t = A^{-1} D_t^{1/2}$ varies across time, structural shock propagation differs dramatically depending on the volatility regime.

`BVAR_SVResult.irf(horizon=20, t_idx=...)` allows researchers to condition impulse responses on specific historical dates $t^*$:
- **High-Volatility Regime (e.g., 2008Q4 or 2020Q2)**: Evaluates impulse responses when financial or macroeconomic uncertainty is elevated.
- **Tranquil Regime (e.g., mid-1990s Great Moderation)**: Evaluates transmission in quiet macroeconomic conditions.

---

## 4. Basic & Advanced Usage

### Estimating a BVAR-SV Model

```python
import numpy as np
import pandas as pd
from puremacro.var.bvar_sv import bvar_sv

# 1. Load or prepare macroeconomic dataset (e.g., GDP Growth, Inflation, Policy Rate)
rng = np.random.default_rng(42)
T = 180
dates = pd.date_range("1975-01-01", periods=T, freq="QE")

# Synthetic data with time-varying volatility
data = np.zeros((T, 3))
for t in range(1, T):
    vol_scale = 2.5 if 130 <= t <= 145 else 1.0  # Volatility spike
    data[t] = 0.6 * data[t-1] + vol_scale * 0.5 * rng.standard_normal(3)

df = pd.DataFrame(data, index=dates, columns=["Output", "Inflation", "Rate"])

# 2. Fit BVAR-SV Model with Gibbs Sampler
res_sv = bvar_sv(
    data=df,
    lags=2,
    n_draws=1500,
    n_burn=500,
    n_chains=2,
    minnesota_prior=True,
    lambda1=0.2,
    lambda2=0.5,
    seed=123,
)

# 3. Inspect Convergence Diagnostics & Posterior Summary
print(res_sv.summary())
print(f"Max Gelman-Rubin R-hat: {res_sv.max_rhat:.3f}")
assert res_sv.max_rhat < 1.1, "Chains must pass Gelman-Rubin convergence (< 1.1)"

# 4. Extract Volatility-Conditioned IRFs
# Compare response during crisis (t_idx=135) vs calm period (t_idx=50)
irf_crisis = res_sv.irf(horizon=16, t_idx=135, ci=0.90)
irf_calm = res_sv.irf(horizon=16, t_idx=50, ci=0.90)

print(f"Peak Output Response (Crisis): {irf_crisis.median[:, 0, 0].max():.3f}")
print(f"Peak Output Response (Calm)  : {irf_calm.median[:, 0, 0].max():.3f}")

# 5. Plot Posterior Volatilities and Conditioned IRFs
fig = res_sv.plot()

# 6. Export Tables
latex_table = res_sv.to_latex()
md_table = res_sv.to_markdown()
```

---

## 5. Full API Specification

### `bvar_sv`

```python
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
- `data`: $(T, n)$ dataset of endogenous time series variables.
- `lags` / `p`: Autoregressive lag order $p$ (default `4`).
- `n_draws`: Post-burn-in MCMC draws retained per chain (default `2000`).
- `n_burn`: Initial burn-in iterations discarded (default `1000`).
- `minnesota_prior`: Whether to apply Minnesota prior shrinkage (default `True`).
- `seed`: Reproducibility seed for MCMC sampler.
- `lambda1`: Overall Minnesota shrinkage hyperparameter (default `0.2`).
- `lambda2`: Cross-variable shrinkage hyperparameter (default `0.5`).
- `lambda3`: Lag-decay exponent (default `1.0`).
- `n_chains`: Number of parallel MCMC chains run for Gelman-Rubin convergence checks (default `2`).
- `thin`: Thinning interval for posterior draws (default `1`).

---

## 6. Result Interface

The returned `BVAR_SVResult` container provides:

- **Attributes**:
  - `beta_draws`: Posterior draws of VAR lag coefficients $(D, 1+np, n)$.
  - `h_draws`: Posterior draws of latent log-volatility paths $(D, T_{eff}, n)$.
  - `a_draws`: Posterior draws of contemporaneous impact matrix $A$ $(D, n, n)$.
  - `phi_draws`: Volatility persistence draws $(D, n)$.
  - `sigma_h_draws`: Volatility of volatility draws $(D, n)$.
  - `rhat`: Array of Gelman-Rubin convergence statistics across parameters.
  - `max_rhat`: Maximum $\hat{R}$ observed.
  - `log_score`: Out-of-sample log predictive score density evaluations.
- **Methods**:
  - `.irf(horizon=20, t_idx=-1, ci=0.9)`: Returns `BVAR_SV_IRF` container with `.median`, `.lower`, and `.upper` arrays.
  - `.plot()`: Matplotlib multi-panel plot illustrating estimated posterior log-volatility trajectories across variables alongside credible intervals.
  - `.summary()`: Comprehensive MCMC diagnostic report showing parameter posterior medians, credible intervals, and $\hat{R}$ values.
  - `.to_markdown()`, `.to_latex()`, `.to_typst()`: Formatted tables for empirical manuscripts.
