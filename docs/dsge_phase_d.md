[Español](es/dsge_phase_d.md) · **English**

# DSGE Frontier: Nonlinear Particle Filtering with Stochastic Volatility & Markov-Switching DSGE

`puremacro` introduces two state-of-the-art computational architectures for frontier nonlinear and regime-shifting macroeconomic analysis:

1. **Pure-Python Vectorized Particle Filtering (Gordon et al. 1993; Pitt & Shephard 1999; Fernández-Villaverde & Rubio-Ramírez 2007)**:
   A high-throughput Sequential Monte Carlo (SMC) engine for exact nonlinear likelihood evaluation on 2nd-order and 3rd-order pruned perturbation models (`PrunedDSGESolution`, `Order3PrunedSolution`). Incorporates autoregressive **Stochastic Volatility (SV)** $\sigma_{j,t} = \bar{\sigma}_j \exp(h_{j,t})$ with state augmentation capturing precautionary saving shifts and skewness, fat-tailed innovation distributions (Student-$t$, Gaussian mixtures), and vectorized O(N) resampling algorithms (systematic, stratified, residual, multinomial) executed with zero Python loops across particles.

2. **Markov-Switching DSGE (MS-DSGE / Regime Switching) (Foerster, Rubio-Ramírez, Waggoner & Zha 2016; Farmer, Waggoner & Zha 2011)**:
   A comprehensive perturbation framework for rational expectations models subject to discrete Markovian parameter regime shifts across monetary and fiscal policy environments. Solves coupled quadratic matrix equations via analytical block Newton-Raphson and damped functional iteration, verifies Mean-Square Stability (MSS) $\rho(M_2) < 1$ and first-moment stability $\rho(M_1) < 1$, computes ergodic stationary distributions and unconditional discrete Lyapunov covariances, and evaluates exact closed-form Generalized Impulse Response Functions (GIRF) $(1_S^\top \otimes I_n) M_1^h z_0$ to machine precision in sub-millisecond execution.

Both modules are strictly compliant with the Pyodide four-package contract (`numpy`, `scipy`, `pandas`, `matplotlib`), requiring zero C-extensions, zero Fortran compilers, and zero external proprietary solvers.

---

## Methodological Comparison

| Dimension | Bootstrap Particle Filter (BPF) | Auxiliary Particle Filter (APF) | Markov-Switching DSGE (MS-DSGE) | Standard Linear Kalman Filter |
|:---|:---|:---|:---|:---|
| **Core Reference** | Gordon et al. (1993); FV-RR (2007) | Pitt & Shephard (1999) | Foerster et al. (2016); FWZ (2011) | Kalman (1960); Hamilton (1994) |
| **Model Class** | 2nd/3rd pruned DSGE + SV | 2nd/3rd pruned DSGE + SV | Discrete Markov regime perturbation | Linear Gaussian DSGE |
| **State Space** | Particles $x_t^i$ + log-vol $h_t^i$ | Particles $x_t^i$ + first-stage indices | Coupled regime rules $(T(s), R(s), c(s))$ | Mean state $\hat{x}_{t\|t}$ + Covariance $P_{t\|t}$ |
| **Likelihood Evaluation** | SMC average $\frac{1}{N} \sum w_t^i$ | Reweighted SMC proposal | Regime-weighted filter / Hamilton | Exact Gaussian prediction error |
| **Computational Core** | Vectorized einsum tensor propagation | Predictive covariance $\Sigma_\mu = Z R Q R^\top Z^\top + H$ | Coupled quadratics (Block Newton-Raphson) | Discrete Riccati update |
| **Primary Economic Insight** | Precautionary saving, skewness, SV | Proposal robustness under outliers | Policy regime shifts, MSS determinacy | Linear Gaussian dynamics |
| **Pyodide Compatible** | Yes (`numpy`, `scipy`) | Yes (`numpy`, `scipy`) | Yes (`numpy`, `scipy`) | Yes (`numpy`, `scipy`) |

---

## 1. Nonlinear Particle Filtering & Stochastic Volatility

### 1.1 State-Space Representation with Pruned Perturbation

Linear perturbation models cannot capture precautionary savings, risk premiums, or asymmetric skewness because certainty equivalence holds at first order. Higher-order perturbations address this limitation, but standard Taylor expansions exhibit artificial explosive paths. To ensure stability, `puremacro` utilizes Kim, Kim, Schaumburg & Sims (2008) **pruned perturbation** solutions.

At 2nd order, the state vector is decomposed into first- and second-order components $x_t = x_t^{\text{I}} + x_t^{\text{II}}$:
$$x_t^{\text{I}} = h_x x_{t-1}^{\text{I}} + h_u \varepsilon_t$$
$$x_t^{\text{II}} = h_x x_{t-1}^{\text{II}} + \frac{1}{2} H_{xx} (x_{t-1}^{\text{I}} \otimes x_{t-1}^{\text{I}}) + H_{xu} (x_{t-1}^{\text{I}} \otimes \varepsilon_t) + \frac{1}{2} H_{uu} (\varepsilon_t \otimes \varepsilon_t) + \frac{1}{2} h_{\sigma\sigma} \sigma^2$$
$$y_t = g_x x_t + \frac{1}{2} G_{xx} (x_t^{\text{I}} \otimes x_t^{\text{I}}) + \frac{1}{2} g_{\sigma\sigma} \sigma^2$$

`puremacro` vectorizes this evaluation across $N$ particles using tensor contractions (`np.einsum("mi,mj->mij", x, x)`), executing 10,000 particles over 50 periods in approximately 50 milliseconds without any Python loops over particles.

### 1.2 Stochastic Volatility Dynamics

Macroeconomic uncertainty varies significantly over time. `puremacro` augments the structural state space with an autoregressive Stochastic Volatility (SV) process for structural shocks $\varepsilon_t$:
$$\varepsilon_{j, t} = \sigma_{j, t} \cdot \zeta_{j, t}, \quad \zeta_{j, t} \sim \mathcal{N}(0, 1)$$
$$\sigma_{j, t} = \bar{\sigma}_j \exp(h_{j, t})$$
$$h_{j, t} = \rho_{h, j} h_{j, t-1} + \sigma_{\eta, j} \eta_{j, t}, \quad \eta_{j, t} \sim \mathcal{N}(0, 1)$$
where $\bar{\sigma}_j$ is the baseline shock scale, $\rho_{h, j} \in [0, 1)$ is the volatility persistence, and $\sigma_{\eta, j} > 0$ is the volatility of volatility.

Each particle maintains both the physical DSGE states $(x_t^{\text{I}}, x_t^{\text{II}})$ and the latent log-volatility vector $h_t$, enabling joint inference on latent macroeconomic states and time-varying economic uncertainty.

### 1.3 Vectorized Bootstrap Particle Filter (BPF)

The Bootstrap Particle Filter propagates an empirical discrete distribution $\{x_t^i, w_t^i\}_{i=1}^N$ approximating the filtering density $p(x_t \mid y_{1:t})$:

1. **Initialization**: Draw $x_0^i \sim p(x_0)$ and $h_0^i \sim \mathcal{N}\left(0, \frac{\sigma_\eta^2}{1 - \rho_h^2}\right)$ with uniform weights $w_0^i = 1/N$.
2. **Propagation**: Sample structural innovations $\zeta_t^i \sim \mathcal{N}(0, I)$ and volatility innovations $\eta_t^i \sim \mathcal{N}(0, I)$. Compute time-varying scales $\sigma_t^i$ and evaluate pruned state transitions:
   $$x_t^i = \mathcal{T}(x_{t-1}^i, \sigma_t^i \zeta_t^i)$$
3. **Measurement Weighting**: Compute importance weights using the observation density:
   $$\tilde{w}_t^i = p(y_t \mid x_t^i) = (2\pi)^{-d_y/2} |H|^{-1/2} \exp\left( -\frac{1}{2} (y_t - g(x_t^i))^\top H^{-1} (y_t - g(x_t^i)) \right)$$
   Normalize weights: $w_t^i = \frac{\tilde{w}_t^i}{\sum_{j=1}^N \tilde{w}_t^j}$.
4. **Log-Likelihood Contribution**:
   $$\ln p(y_t \mid y_{1:t-1}) = \ln \left( \frac{1}{N} \sum_{i=1}^N \tilde{w}_t^i \right)$$
5. **Effective Sample Size (ESS) & Resampling**:
   $$ESS_t = \frac{1}{\sum_{i=1}^N (w_t^i)^2}$$
   When $ESS_t < \tau_{\text{resample}} \cdot N$ (typically $\tau = 0.5$), resample particles using $O(N)$ systematic, stratified, or residual resampling.

### 1.4 Auxiliary Particle Filter (APF)

When measurement error is small or observations fall in the distribution tails, standard BPF proposal densities $p(x_t \mid x_{t-1}^i)$ diverge from the true posterior, causing particle depletion. The Auxiliary Particle Filter (Pitt & Shephard 1999) incorporates the current observation $y_t$ into the first-stage selection weights:
$$\alpha_t^i \propto w_{t-1}^i \cdot p(y_t \mid \mu_t^i)$$
where $\mu_t^i = \mathbb{E}[x_t \mid x_{t-1}^i]$ is the deterministic predictive mean of particle $i$.

`puremacro` enhances standard APF by computing the exact predictive observation covariance:
$$\Sigma_\mu = Z (R Q R^\top) Z^\top + H$$
This guarantees non-singular predictive densities $p(y_t \mid \mu_t^i) = \mathcal{N}(y_t \mid g(\mu_t^i), \Sigma_\mu)$, preventing auxiliary weight collapse and providing stable state tracking.

---

## 2. Markov-Switching DSGE (MS-DSGE / Regime Switching)

### 2.1 Structural Perturbation System

Following Foerster, Rubio-Ramírez, Waggoner & Zha (FRWZ 2016), consider a rational expectations macroeconomic system where structural parameters switch across $S$ discrete regimes $s_t \in \{1, \dots, S\}$:
$$A(s_t) \mathbb{E}_t [y_{t+1}] + B(s_t) y_t + C(s_t) y_{t-1} + K(s_t) + D(s_t) \varepsilon_t = 0$$
where $y_t \in \mathbb{R}^n$, $\varepsilon_t \sim \text{i.i.d.} \mathcal{N}(0, \Sigma_\varepsilon)$, and the regime transitions follow an exogenous ergodic Markov chain:
$$P = [p_{ij}], \quad p_{ij} = \Pr(s_{t+1} = j \mid s_t = i)$$

### 2.2 Minimal State Variable (MSV) Solution

The linear perturbation Minimal State Variable (MSV) equilibrium rule satisfies:
$$y_t = c(s_t) + T(s_t) y_{t-1} + R(s_t) \varepsilon_t$$
Substituting into the structural system and taking expectations conditional on information at date $t$ and regime $s_t = i$:
$$\mathbb{E}_t [y_{t+1}] = \sum_{j=1}^S p_{ij} \left( c_j + T_j y_t \right) = \sum_{j=1}^S p_{ij} c_j + \left( \sum_{j=1}^S p_{ij} T_j \right) \left( c_i + T_i y_{t-1} + R_i \varepsilon_t \right)$$

Matching coefficients across state variables yields the **coupled quadratic matrix equations**:
$$A_i \left( \sum_{j=1}^S p_{ij} T_j \right) T_i + B_i T_i + C_i = 0, \quad \forall i \in \{1, \dots, S\}$$
Given solution matrices $T_1, \dots, T_S$, the shock impact matrices $R_i$ and intercepts $c_i$ are determined uniquely:
$$R_i = - \left[ A_i \left( \sum_{j=1}^S p_{ij} T_j \right) + B_i \right]^{-1} D_i$$
$$\left[ A_i \left( \sum_{j=1}^S p_{ij} T_j \right) + B_i \right] c_i + A_i \sum_{j=1}^S p_{ij} c_j + K_i = 0$$

### 2.3 Solvers: Block Newton-Raphson & Functional Iteration

`puremacro` provides two complementary algorithms to solve the coupled quadratic system:

1. **Analytical Block Newton-Raphson**:
   Vectorizing the system into $F(\mathbf{T}) = 0$ with $\mathbf{T} = [\operatorname{vec}(T_1)^\top, \dots, \operatorname{vec}(T_S)^\top]^\top$. The exact block Jacobian $J \in \mathbb{R}^{Sn^2 \times Sn^2}$ is evaluated analytically:
   $$J_{ii} = I_n \otimes \left( A_i \sum_{j=1}^S p_{ij} T_j + B_i \right) + T_i^\top \otimes (p_{ii} A_i)$$
   $$J_{ij} = T_i^\top \otimes (p_{ij} A_i) \quad (i \neq j)$$
   Newton steps $\mathbf{T}^{(k+1)} = \mathbf{T}^{(k)} - J(\mathbf{T}^{(k)})^{-1} F(\mathbf{T}^{(k)})$ achieve quadratic convergence in 5–10 iterations to machine precision ($10^{-16}$).

2. **Damped Functional Iteration**:
   $$T_i^{(k+1)} = (1 - \alpha) T_i^{(k)} - \alpha \left[ A_i \sum_{j=1}^S p_{ij} T_j^{(k)} + B_i \right]^{-1} C_i$$
   Robust alternative when starting far from the equilibrium basin of attraction.

### 2.4 Stability Diagnostics: Mean and Mean-Square Stability (MSS)

Standard eigenvalues of individual $T_i$ matrices do not determine equilibrium stability because an economy can switch between locally explosive and locally stable regimes. Following Costa, Fragoso & Marques (2005) and Farmer et al. (2011), `puremacro` computes the exact companion stability operators:

1. **First-Moment Stability (Mean Stability)**:
   $$M_1 = (P^\top \otimes I_n) \operatorname{diag}(T_1, \dots, T_S)$$
   The system is first-moment stable if and only if $\rho(M_1) < 1$.

2. **Second-Moment Stability (Mean-Square Stability - MSS)**:
   $$M_2 = (P^\top \otimes I_{n^2}) \operatorname{diag}(T_1 \otimes T_1, \dots, T_S \otimes T_S)$$
   The system is Mean-Square Stable if and only if $\rho(M_2) < 1$. Mean-square stability implies first-moment stability and guarantees finite asymptotic unconditional variances.

### 2.5 Ergodic Moments & Discrete Lyapunov Covariance

Let $\pi_\infty$ be the unique stationary distribution of the Markov chain:
$$\pi_\infty P = \pi_\infty, \quad \sum_{i=1}^S \pi_{\infty, i} = 1$$
The unconditional ergodic mean is:
$$\bar{y} = \sum_{i=1}^S \pi_{\infty, i} c_i$$
The unconditional covariance matrix $\operatorname{Var}(y) = \sum_{i=1}^S \pi_{\infty, i} \Sigma_{y, i}$ satisfies the coupled discrete Lyapunov equation:
$$\Sigma_{y, i} = T_i \left( \sum_{j=1}^S p_{ji} \frac{\pi_{\infty, j}}{\pi_{\infty, i}} \Sigma_{y, j} \right) T_i^\top + R_i \Sigma_\varepsilon R_i^\top$$
solved in vectorized form via $(I_{Sn^2} - M_2)^{-1}$.

### 2.6 Closed-Form Analytical Generalized Impulse Response (GIRF)

Traditional regime-conditional impulse responses assume the regime remains frozen counterfactually forever. In reality, economic agents understand that regimes switch stochastically in the future. The Generalized Impulse Response Function (GIRF; Koop, Pesaran & Potter 1996) integrates over all future regime paths:
$$\operatorname{GIRF}_h(s_0, \varepsilon_0) = \mathbb{E}[y_{t+h} \mid s_t = s_0, \varepsilon_t = \varepsilon_0] - \mathbb{E}[y_{t+h} \mid s_t = s_0]$$

`puremacro` computes the exact analytical GIRF in closed form:
$$\operatorname{GIRF}_h(s_0, \varepsilon_0) = (1_S^\top \otimes I_n) M_1^h z_0$$
where $z_0 = e_{s_0} \otimes (R(s_0) \varepsilon_0) \in \mathbb{R}^{Sn}$, evaluating in $< 1$ millisecond to machine precision without Monte Carlo simulation noise.

---

## 3. Python API & Code Examples

### 3.1 Nonlinear Particle Filtering with Stochastic Volatility

```python
import numpy as np
import pandas as pd
from puremacro.dsge.dynare import load_mod
from puremacro.dsge.particle_filter import particle_filter, StochasticVolatilitySpec

# 1. Compile DSGE model and solve 2nd-order pruned perturbation
mod_code = """
var c k z;
varexo eps;
parameters beta alpha delta rho sigma_pref sigma_eps;
beta = 0.99; alpha = 0.33; delta = 0.025; rho = 0.95; sigma_pref = 1.0; sigma_eps = 0.01;
model;
  exp(-sigma_pref*c) - beta*exp(-sigma_pref*c(+1))*(alpha*exp(z(+1))*exp((alpha-1)*k) + 1 - delta);
  exp(c) + exp(k) - exp(z)*exp(alpha*k(-1)) - (1 - delta)*exp(k(-1));
  z - rho*z(-1) - sigma_eps*eps;
end;
initval; k = 3.8; c = 0.8; z = 0.0; end;
steady;
"""
model = load_mod(mod_code)
solution = model.solve(order=2)

# 2. Prepare observable data
data = pd.DataFrame({"c": [0.80, 0.81, 0.79], "k": [3.31, 3.32, 3.30]})

# 3. Specify Stochastic Volatility: sigma_t = bar{sigma} * exp(h_t)
sv = StochasticVolatilitySpec(rho=0.85, sigma_eta=0.20, base_scale=0.01)

# 4. Run Bootstrap Particle Filter (BPF)
result = particle_filter(
    solution,
    data=data,
    observed_vars=["c", "k"],
    n_particles=10_000,
    method="bootstrap",
    resampling_method="systematic",
    stochastic_volatility=sv,
    seed=42,
)

print(result.summary())
print(f"Log-Likelihood: {result.log_likelihood:.4f}")
print(f"Resampling Frequency: {result.resampling_frequency:.1%}")
```

### 3.2 Markov-Switching DSGE Perturbation Solver

```python
import numpy as np
from puremacro.dsge.markov_switching import solve_ms_dsge

# 1. Structural parameters: 3-equation New Keynesian model
beta, sigma, kappa, rho_i, phi_x = 0.99, 1.0, 0.1, 0.8, 0.1
regime_names = ["Hawkish", "Dovish"]
phi_pi = [1.8, 0.8]  # Active vs Passive Taylor rule response

# Transition probability matrix: 90% hawkish persistence, 80% dovish persistence
P = np.array([
    [0.90, 0.10],
    [0.20, 0.80],
])

# 2. Build regime-specific structural matrices A(s), B(s), C(s), D(s)
A, B, C, D = [], [], [], []
for s in range(2):
    As = np.array([[1.0, 1.0 / sigma, 0.0], [0.0, beta, 0.0], [0.0, 0.0, 0.0]])
    Bs = np.array([[-1.0, 0.0, -1.0 / sigma], [kappa, -1.0, 0.0], [(1 - rho_i) * phi_x, (1 - rho_i) * phi_pi[s], -1.0]])
    Cs = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, rho_i]])
    Ds = np.eye(3)
    A.append(As); B.append(Bs); C.append(Cs); D.append(Ds)

# 3. Solve MS-DSGE via Block Newton-Raphson
ms_res = solve_ms_dsge(
    A, B, C, D, P,
    regime_names=regime_names,
    variable_names=["output_gap", "inflation", "interest_rate"],
    shock_names=["demand", "cost_push", "monetary_policy"],
    method="newton",
)

# 4. Check stability and ergodic moments
print(f"Converged: {ms_res.converged} in {ms_res.iterations} iterations")
print(f"Mean-Square Stable (MSS): {ms_res.mean_square_stable} (rho(M2) = {ms_res.spectral_radius_mss:.4f})")
print("Ergodic Distribution:", ms_res.ergodic_distribution.to_dict())

# 5. Evaluate exact analytical GIRF
girf = ms_res.girf("Hawkish", shock="monetary_policy", horizon=12)
print("Impact GIRF:\n", girf.head(2))
```

---

## 4. Result Object Presentation Contract

In adherence to puremacro's unified presentation architecture, both `ParticleFilterResult` and `MSDSGEResult` implement the standardized six-method presentation contract:

| Method | Return Type | Description |
|:---|:---|:---|
| `.summary()` | `str` or `pd.DataFrame` | Publication-ready formatted statistical summary table |
| `.plot()` | `Figure` or `tuple[Figure, Any]` | Multi-panel publication-grade diagnostic plots |
| `.to_frame()` | `pd.DataFrame` | Structured tabular DataFrame representation |
| `.to_markdown(**kwargs)` | `str` | GitHub-flavored Markdown table |
| `.to_latex(**kwargs)` | `str` | Booktabs publication-quality LaTeX table |
| `.to_typst(**kwargs)` | `str` | Modern Typst scientific document table |

---

## Academic References

- **Costa, O. L., Fragoso, M. D., & Marques, R. P. (2005).** *Discrete-Time Markov Jump Linear Systems*. Springer Science & Business Media.
- **Davig, T., & Leeper, E. M. (2007).** Generalizing the Taylor principle. *American Economic Review*, 97(3), 607–635.
- **Farmer, R. E., Waggoner, D. F., & Zha, T. (2011).** Minimal state variable solutions to Markov-switching rational expectations models. *Journal of Economic Dynamics and Control*, 35(12), 2150–2166.
- **Fernández-Villaverde, J., & Rubio-Ramírez, J. F. (2007).** Estimating macroeconomics models: A likelihood approach. *Review of Economic Studies*, 74(4), 1059–1087.
- **Foerster, A. T., Rubio-Ramírez, J. F., Waggoner, D. F., & Zha, T. (2016).** Perturbation methods for Markov-switching dynamic stochastic general equilibrium models. *Econometrica*, 84(6), 2219–2270.
- **Gordon, N. J., Salmond, D. J., & Smith, A. F. (1993).** Novel approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F (Radar and Signal Processing)*, 140(2), 107–113.
- **Kim, J., Kim, S., Schaumburg, E., & Sims, C. A. (2008).** Calculating and using second-order accurate solutions of discrete time dynamic equilibrium models. *Journal of Economic Dynamics and Control*, 32(11), 3397–3414.
- **Koop, G., Pesaran, M. H., & Potter, S. M. (1996).** Impulse response analysis in nonlinear multivariate models. *Journal of Econometrics*, 74(1), 119–147.
- **Leeper, E. M. (1991).** Equilibria under 'active' and 'passive' monetary and fiscal policies. *Journal of Monetary Economics*, 27(1), 129–147.
- **Pitt, M. K., & Shephard, N. (1999).** Filtering via simulation: Auxiliary particle filters. *Journal of the American Statistical Association*, 94(446), 590–599.
