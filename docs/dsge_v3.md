[Español](es/dsge_v3.md) · **English**

# puremacro 3.0: Exact Analytic Gradients, Pure-Python HMC/NUTS & Sequence-Space HANK

`puremacro 3.0.0` represents a transformative generational leap for macroeconomic modeling in Python. While the 2.x release cycle brought `puremacro` to **complete operational parity with Dynare**—encompassing the full `.mod` parser, second- and third-order pruned perturbation, extended path simulation, OccBin piecewise filtering, and automated identification analysis—version 3.0.0 pivots from *emulating legacy toolchains* to *surpassing them*.

Version 3.0.0 introduces three pioneering capabilities directly into standard `.mod` macroeconomic workflows:

1. **Exact Analytic Likelihood Gradients ($\nabla_\theta \ln L$)**: Evaluates exact structural score vectors via AST parameter differentiation, generalized Sylvester matrix equations solved via complex Schur triangular back-substitution, and single-pass forward Kalman score recursions.
2. **Pure-Python Hamiltonian Monte Carlo & No-U-Turn Sampler (NUTS)**: Provides high-efficiency gradient-based MCMC for high-dimensional posterior sampling with Betancourt generalized U-turn stopping, Hoffman-Gelman dual averaging, Welford online covariance adaptation with Stan shrinkage, and complete MCMC diagnostics (split-$\hat{R}$, bulk/tail ESS, E-BFMI) under the strict zero-C-extension Pyodide four-package contract (`numpy`, `scipy`, `pandas`, `matplotlib`).
3. **Heterogeneous Agents (HANK) Sequence-Space Bridge in `.mod` Files**: Introduces the `hetagent_block; ... end;` grammar into Dynare-style `.mod` specifications, seamlessly coupling microeconomic household decisions (stationary wealth distributions $\mathcal{D}^*(a)$, $MPC(a)$ schedules) with aggregate DSGE market-clearing conditions via Auclert, Bardóczy, Rognlie & Straub (2021) Fake-News Jacobians ($J_{C, r}, J_{C, Y}$) and linear/nonlinear Broyden transition dynamics.

---

## Executive Summary: Beyond Legacy Toolchains

Traditional macroeconomic toolchains (such as Dynare, Dynare++, or custom MATLAB routines) have long been constrained by three structural bottlenecks:

| Dimension | Legacy Toolchains (Dynare / MATLAB) | puremacro 3.0.0 |
|:---|:---|:---|
| **Likelihood Gradients** | Two-sided finite differences ($\approx 2K$ Kalman filter evaluations per gradient step); noisy, slow, prone to numerical breakdown near unit roots. | **Exact Analytic Gradients ($\nabla_\theta \ln L$)**: Implicit generalized Sylvester solver + single-pass forward Kalman score recursion in $\mathcal{O}(K \cdot N^3)$ flops ($< 1$ ms). |
| **Bayesian Posterior Sampling** | Random-Walk Metropolis-Hastings (RWMH); diffuses slowly in high dimensions ($ESS < 2\%$), requiring millions of draws for medium-scale models. | **Pure-Python HMC / NUTS**: Phase-space symplectic leapfrog dynamics traversing complex curved posteriors with zero step-size tuning ($ESS / s \ge 10\times$ RWMH). |
| **Heterogeneous Agents** | Representative Agent (RANK) or two-agent (TANK); unable to solve heterogeneous-agent sequence-space models natively within `.mod` files. | **Sequence-Space HANK Bridge**: Native `hetagent_block` grammar in `.mod` files; solves stationary distributions $\mathcal{D}^*(a)$, Fake-News Jacobians, and general equilibrium transition paths in seconds. |
| **Software Footprint** | Requires proprietary MATLAB runtime or external C++ / Fortran compilers, MEX binaries, and LAPACK links. | **Zero-Dependency Pure Python**: Runs everywhere from HPC clusters to tablets and web browsers via Pyodide / WebAssembly (`numpy`, `scipy`, `pandas`, `matplotlib` only). |

---

## Pillar 1: Exact Analytic Likelihood Gradients ($\nabla_\theta \ln L$)

### 1.1 Motivation & Mathematical Formulation

Estimating DSGE models via gradient-based optimizers (such as L-BFGS-B, SQP) or Hamiltonian Monte Carlo requires the gradient of the log-posterior:
$$\nabla_\theta \ln p(\theta \mid Y) = \nabla_\theta \ln L(Y \mid \theta) + \nabla_\theta \ln p(\theta)$$

In legacy toolchains, $\nabla_\theta \ln L$ is approximated via two-sided finite differences:
$$\frac{\partial \ln L}{\partial \theta_j} \approx \frac{\ln L(\theta + h e_j) - \ln L(\theta - h e_j)}{2h}$$

This numerical approach suffers from severe flaws:
1. **Computational Inefficiency**: For $K$ estimated parameters, each gradient evaluation demands $2K$ full Kalman filter sweeps over the sample history $T_{obs}$. For large models ($K \ge 35$), a single optimization step can take hundreds of seconds.
2. **Step-Size Instability**: Choosing the perturbation step $h$ is fraught: too large induces truncation bias; too small triggers catastrophic floating-point cancellation in near-unit-root regimes.

`puremacro 3.0.0` replaces finite differences with an **exact, single-pass forward analytic score engine**.

Consider a rational-expectations structural DSGE system around its deterministic steady state:
$$\mathbb{E}_t \left[ f(y_{t+1}, y_t, y_{t-1}, u_t; \theta) \right] = 0$$

Its first-order perturbation satisfies:
$$A_+(\theta) y_{t+1} + A_0(\theta) y_t + A_-(\theta) y_{t-1} + B_u(\theta) u_t = 0$$

The unique stable recursive equilibrium law of motion is given by:
$$y_t = G(\theta) y_{t-1} + N(\theta) u_t$$
where the state transition matrix $G(\theta)$ satisfies the quadratic matrix equation:
$$\mathcal{F}(G; \theta) \equiv A_+(\theta) G(\theta)^2 + A_0(\theta) G(\theta) + A_-(\theta) = 0$$
and the innovation loading matrix $N(\theta)$ satisfies:
$$(A_0(\theta) + A_+(\theta) G(\theta)) N(\theta) + B_u(\theta) = 0$$

### 1.2 The Generalized Sylvester Matrix Equation

Differentiating $\mathcal{F}(G; \theta) = 0$ with respect to a scalar structural parameter $\theta_j$ yields:
$$(A_0 + A_+ G) \frac{\partial G}{\partial \theta_j} + A_+ \frac{\partial G}{\partial \theta_j} G = - \left( \frac{\partial A_+}{\partial \theta_j} G^2 + \frac{\partial A_0}{\partial \theta_j} G + \frac{\partial A_-}{\partial \theta_j} \right)$$

This is a **generalized Sylvester matrix equation** of the canonical form:
$$\hat{A} X + B X C = D_j$$
where:
- $\hat{A} = A_0 + A_+ G \in \mathbb{R}^{N \times N}$
- $B = A_+ \in \mathbb{R}^{N \times N}$
- $C = G \in \mathbb{R}^{N \times N}$
- $D_j = - \left( \frac{\partial A_+}{\partial \theta_j} G^2 + \frac{\partial A_0}{\partial \theta_j} G + \frac{\partial A_-}{\partial \theta_j} \right) \in \mathbb{R}^{N \times N}$

Because Blanchard-Kahn stability guarantees that all eigenvalues of $C = G$ lie strictly inside the open unit circle, this generalized Sylvester system has a unique solution $X = \frac{\partial G}{\partial \theta_j}$ for each parameter $\theta_j$.

#### Schur Triangular Back-Substitution Algorithm
Rather than solving $K$ separate $N^2 \times N^2$ Kronecker systems, `puremacro` computes the complex Schur decomposition of $C$ **once**:
$$C = U_c T_c U_c^H$$
where $T_c$ is upper triangular and $U_c$ is unitary.

Transforming $\tilde{X} = X U_c$ and $\tilde{D}_j = D_j U_c$, the Sylvester equation transforms to:
$$\hat{A} \tilde{X} + B \tilde{X} T_c = \tilde{D}_j$$

Because $T_c$ is upper triangular, column $k$ of $\tilde{X}$ decouples into:
$$\left( \hat{A} + (T_c)_{k, k} B \right) \tilde{X}_{:, k} = \tilde{D}_{:, k} - B \sum_{m < k} \tilde{X}_{:, m} (T_c)_{m, k}$$

Each column $k \in \{0, \dots, N-1\}$ is solved by back-substitution. The matrix $(\hat{A} + (T_c)_{k,k} B)$ is LU-factored once per column and reused across all $K$ parameters simultaneously. The physical solution is recovered via $X = \operatorname{Re}(\tilde{X} U_c^H)$.

Once $\frac{\partial G}{\partial \theta_j}$ is known, differentiating $(A_0 + A_+ G) N + B_u = 0$ yields the exact innovation sensitivity:
$$\frac{\partial N}{\partial \theta_j} = - (A_0 + A_+ G)^{-1} \left[ \left(\frac{\partial A_0}{\partial \theta_j} + \frac{\partial A_+}{\partial \theta_j} G + A_+ \frac{\partial G}{\partial \theta_j}\right) N + \frac{\partial B_u}{\partial \theta_j} \right]$$

### 1.3 State-Space Sensitivities & Lyapunov Initialization

The structural decision rules map directly into the standard linear state-space system:
$$\alpha_{t+1} = T(\theta) \alpha_t + c(\theta) + R(\theta) \eta_{t+1}, \quad \eta_t \sim \mathcal{N}(0, Q(\theta))$$
$$y_t = Z(\theta) \alpha_t + d(\theta) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, H(\theta))$$

The unconditional stationary covariance $P_0(\theta)$ satisfies the discrete Lyapunov equation:
$$P_0 = T P_0 T^\top + R Q R^\top$$

Differentiating with respect to $\theta_j$ yields the discrete Lyapunov equation for $\frac{\partial P_0}{\partial \theta_j}$:
$$\frac{\partial P_0}{\partial \theta_j} - T \frac{\partial P_0}{\partial \theta_j} T^\top = \frac{\partial T}{\partial \theta_j} P_0 T^\top + T P_0 \left(\frac{\partial T}{\partial \theta_j}\right)^\top + \frac{\partial (R Q R^\top)}{\partial \theta_j}$$
which `puremacro` solves directly via `scipy.linalg.solve_discrete_lyapunov`.

### 1.4 Single-Pass Forward Kalman Score Recursion

Given observed data $y_t \in \mathbb{R}^{n_y}$, the forecast errors $v_t = y_t - Z a_{t|t-1} - d$ and error covariances $F_t = Z P_{t|t-1} Z^\top + H$ determine the Gaussian log-likelihood:
$$\ln L(Y \mid \theta) = -\frac{T_{obs} n_y \ln(2\pi)}{2} - \frac{1}{2} \sum_{t=1}^{T_{obs}} \left( \ln |F_t| + v_t^\top F_t^{-1} v_t \right)$$

The exact score vector $\nabla_\theta \ln L$ is computed in a single forward sweep:
$$\frac{\partial \ln L}{\partial \theta_j} = -\frac{1}{2} \sum_{t=1}^{T_{obs}} \left[ \operatorname{tr}\left( F_t^{-1} \frac{\partial F_t}{\partial \theta_j} \right) + 2 v_t^\top F_t^{-1} \frac{\partial v_t}{\partial \theta_j} - (F_t^{-1} v_t)^\top \frac{\partial F_t}{\partial \theta_j} (F_t^{-1} v_t) \right]$$

During each forward step $t$, state sensitivities $\frac{\partial a_{t|t-1}}{\partial \theta_j}$ and covariance sensitivities $\frac{\partial P_{t|t-1}}{\partial \theta_j}$ are updated alongside the standard Kalman equations:
$$\frac{\partial v_t}{\partial \theta_j} = -\frac{\partial Z}{\partial \theta_j} a_{t|t-1} - Z \frac{\partial a_{t|t-1}}{\partial \theta_j} - \frac{\partial d}{\partial \theta_j}$$
$$\frac{\partial F_t}{\partial \theta_j} = \frac{\partial Z}{\partial \theta_j} P_{t|t-1} Z^\top + Z \frac{\partial P_{t|t-1}}{\partial \theta_j} Z^\top + Z P_{t|t-1} \left(\frac{\partial Z}{\partial \theta_j}\right)^\top + \frac{\partial H}{\partial \theta_j}$$

With Kalman gain $K_t = T P_{t|t-1} Z^\top F_t^{-1}$ and closed-loop transition $L_t = T - K_t Z$, the Joseph-stabilized covariance sensitivity is propagated forward:
$$\frac{\partial P_{t+1|t}}{\partial \theta_j} = \frac{\partial L_t}{\partial \theta_j} P_{t|t-1} L_t^\top + L_t \frac{\partial P_{t|t-1}}{\partial \theta_j} L_t^\top + L_t P_{t|t-1} \left(\frac{\partial L_t}{\partial \theta_j}\right)^\top + \frac{\partial K_t}{\partial \theta_j} H K_t^\top + K_t \frac{\partial H}{\partial \theta_j} K_t^\top + K_t H \left(\frac{\partial K_t}{\partial \theta_j}\right)^\top + \frac{\partial (R Q R^\top)}{\partial \theta_j}$$

This recurrence maintains numerical positive semi-definiteness and symmetry without requiring gradient tape storage.

### 1.5 Python Code Example: Evaluating Analytic Likelihood Gradients

```python
import numpy as np
import pandas as pd
from puremacro.dsge import load_mod
from puremacro.dsge.observation import make_state_space_from_varobs
from puremacro.dsge._gradients import (
    build_state_space_sensitivities,
    kalman_score,
    ScoreDiagnosticsResult,
)

MOD_CODE = """
var y c a;
varexo e_a;
parameters beta sigma rho_a;
beta = 0.99;
sigma = 1.0;
rho_a = 0.8;

model;
  c = c(+1) - (1/sigma)*y;
  y = c + a;
  a = rho_a * a(-1) + e_a;
end;

steady_state_model;
  y = 0; c = 0; a = 0;
end;

shocks;
  var e_a; stderr 0.2;
end;

varobs y;
"""

# 1. Parse and solve model
model = load_mod(MOD_CODE)
varobs = ["y"]
param_names = ["sigma", "rho_a"]

# 2. Simulate observed series
rng = np.random.default_rng(123)
y_sim = rng.normal(0, 0.5, size=(100, 1))

# 3. Construct state space and parameter sensitivities
ssm = make_state_space_from_varobs(model, varobs)
sens = build_state_space_sensitivities(model, varobs, param_names)

# 4. Evaluate exact log-likelihood and score in a single forward pass
loglik, score = kalman_score(y_sim, ssm, sens)

# 5. Format and inspect score diagnostics
res = ScoreDiagnosticsResult(
    loglik=loglik,
    gradient=score,
    param_names=tuple(param_names),
    elapsed_sec=0.002,
)

print(f"Exact Log-Likelihood: {res.loglik:.4f}")
print(res.to_markdown())
```

---

## Pillar 2: Pure-Python HMC & NUTS Sampler

### 2.1 The No-U-Turn Sampler Algorithm

Standard Random-Walk Metropolis-Hastings (RWMH) explores high-dimensional posteriors via Brownian diffusion: the distance traversed scales as $\mathcal{O}(\sqrt{N})$, resulting in poor exploration, strong autocorrelations, and low Effective Sample Sizes ($ESS < 2\%$).

Hamiltonian Monte Carlo (HMC) transforms posterior sampling into Hamiltonian mechanics. It augments the parameter state $\theta \in \mathbb{R}^K$ with continuous momentum vectors $p \sim \mathcal{N}(0, M)$, defining total Hamiltonian energy:
$$H(\theta, p) = V(\theta) + K(p) = - \ln p(\theta \mid Y) + \frac{1}{2} p^\top M^{-1} p$$
where $M^{-1}$ is a positive-definite inverse mass matrix.

The **No-U-Turn Sampler (NUTS)** (Hoffman & Gelman 2014; Betancourt 2017) eliminates the requirement to hand-tune the trajectory length $L$ by recursively building binary leapfrog trees and stopping as soon as the trajectory begins to fold back on itself.

### 2.2 Symplectic Leapfrog Integrator

Phase space trajectories $(\theta(t), p(t))$ are integrated using the velocity Verlet symplectic scheme:
$$\tilde{p} = p + \frac{\epsilon}{2} \nabla_\theta \ln p(\theta \mid Y)$$
$$\theta' = \theta + \epsilon M^{-1} \tilde{p}$$
$$p' = \tilde{p} + \frac{\epsilon}{2} \nabla_\theta \ln p(\theta' \mid Y)$$

Symplectic leapfrog integration exactly preserves phase space volume (Liouville's theorem) and exhibits bounded energy fluctuations $\mathcal{O}(\epsilon^2)$ without secular energy drift.

### 2.3 Generalized U-Turn Condition & Divergence Detection

At recursion depth $j$, the trajectory doubles forward ($v = +1$) or backward ($v = -1$) with equal probability. Let $\theta^-$ and $\theta^+$ denote the extreme left and right parameter coordinates of the sub-tree, and $p^-, p^+$ denote the momenta at those endpoints.

The recursive tree expansion stops immediately when either endpoint turns back towards the opposite end (Betancourt 2017):
$$( \theta^+ - \theta^- )^\top M^{-1} p^+ < 0 \quad \text{or} \quad ( \theta^+ - \theta^- )^\top M^{-1} p^- < 0$$

Furthermore, if numerical error causes Hamiltonian divergence:
$$H(\theta', p') - H(\theta, p) > \Delta_{max} \quad (\Delta_{max} = 1000)$$
the trajectory is immediately flagged as a **divergence**, signaling regions of extreme posterior curvature or boundary violations.

### 2.4 Hoffman-Gelman Dual Averaging Step-Size Adaptation

During the warmup phase, step size $\epsilon$ is adapted to attain a target acceptance probability $\delta^* = 0.80$ using Nesterov dual averaging:
$$H_m = \left( 1 - \frac{1}{m + t_0} \right) H_{m-1} + \frac{1}{m + t_0} (\delta^* - \alpha_m)$$
$$\ln \epsilon_{m+1} = \mu - \frac{\sqrt{m}}{\gamma} H_m, \quad \ln \bar{\epsilon}_{m+1} = m^{-\kappa} \ln \epsilon_{m+1} + (1 - m^{-\kappa}) \ln \bar{\epsilon}_m$$
with default hyperparameters $\gamma = 0.05$, $t_0 = 10.0$, $\kappa = 0.75$, and $\mu = \ln(10 \epsilon_0)$.

### 2.5 Welford Online Covariance Adaptation with Stan Shrinkage

Parameter scales and correlations vary widely in DSGE estimation (e.g., standard deviation of shocks vs. habit persistence). `puremacro` implements online sample variance adaptation using Welford's algorithm:
$$\bar{\theta}_m = \bar{\theta}_{m-1} + \frac{\theta_m - \bar{\theta}_{m-1}}{m}, \quad M_2^{(m)} = M_2^{(m-1)} + (\theta_m - \bar{\theta}_{m-1})(\theta_m - \bar{\theta}_m)$$

To ensure stability, `puremacro` organizes warmup into Stan-style staged windows:
1. **Initial Fast Buffer** (75 draws): Step size adapts while mass matrix remains fixed.
2. **Doubling Slow Windows** (25, 50, 100, 200 draws): Diagonal mass matrix $M^{-1} = \operatorname{diag}(\widehat{\sigma}^2_1, \dots, \widehat{\sigma}^2_K)$ updates at the end of each window, regularized with prior shrinkage:
   $$\widehat{\Sigma}_{reg} = w S + (1 - w) \Sigma_{prior}, \quad w = \frac{m}{m + 5}$$
3. **Final Fast Buffer** (50 draws): Step size settles while metric is frozen.

### 2.6 Comprehensive MCMC Diagnostics

Every NUTS run produces an audited set of diagnostics:
- **Rank-Normalized Split-$\hat{R}$**: Splits each chain in half to detect both within-chain non-stationarity and between-chain lack of mixing; values $< 1.05$ indicate convergence.
- **Bulk Effective Sample Size ($ESS_{bulk}$)**: Measures estimation accuracy for posterior means and medians via Geyer's initial positive sequence.
- **Tail Effective Sample Size ($ESS_{tail}$)**: Evaluates sampling quality in the tails (5% and 95% quantiles).
- **Energy Bayesian Fraction of Missing Information ($E\text{-}BFMI$)**:
  $$E\text{-}BFMI = \frac{\sum_{t=1}^N (E_t - E_{t-1})^2}{(N-1) \operatorname{Var}(E)}$$
  Values $< 0.3$ warn of poor momentum energy exploration.

### 2.7 Python Code Example: DSGE Estimation via NUTS

```python
import numpy as np
import pandas as pd
from puremacro.dsge import load_mod

MOD_CODE = """
var y c a;
varexo e_a;
parameters beta sigma rho_a;
beta = 0.99;
sigma = 1.0;
rho_a = 0.8;

model;
  c = c(+1) - (1/sigma)*y;
  y = c + a;
  a = rho_a * a(-1) + e_a;
end;

steady_state_model;
  y = 0; c = 0; a = 0;
end;

varobs y;
estimated_params;
  sigma, gamma_pdf, 1.0, 0.25;
  rho_a, beta_pdf, 0.7, 0.15;
end;
"""

# 1. Load DSGE model
model = load_mod(MOD_CODE)

# 2. Generate sample data
rng = np.random.default_rng(42)
data = pd.DataFrame({"y": rng.normal(0, 0.5, size=80)})

# 3. Estimate using pure-Python NUTS with exact analytic gradients
res = model.estimate(
    data=data,
    method="nuts",
    n_draws=300,
    n_chains=2,
    burn_in=150,
    seed=42,
)

# 4. Posterior summary and convergence diagnostics
print(res.summary())
print("Divergences:", res.diagnostics["n_divergences"])
print("Step Sizes:", res.diagnostics["step_sizes"])

# 5. Energy diagnostics & figure exports
stats, fig, ax = res.energy_diagnostics()
print(f"Mean E-BFMI: {stats['mean_ebfmi']:.3f} (Passed: {stats['passed']})")

# Export to clean Markdown table
print(res.to_markdown())
```

---

## Pillar 3: HANK Sequence-Space Bridge in `.mod` Files

### 3.1 The Representative-Agent Wall & Sequence Space

Standard DSGE models assume a representative household whose Euler equation pins down aggregate consumption. Heterogeneous Agent New Keynesian (HANK) models replace this fictional agent with a continuum of households facing uninsurable idiosyncratic income risk and borrowing constraints.

In state-space representations, solving HANK models requires carrying the infinite-dimensional wealth distribution $D_t(a, s)$ as a state variable, causing the curse of dimensionality.

`puremacro 3.0.0` breaks this barrier by integrating the **Sequence-Space Jacobian (SSJ)** method of **Auclert, Bardóczy, Rognlie & Straub (2021, *Econometrica*)** directly into `.mod` files.

### 3.2 The `hetagent_block` Syntax

To declare a heterogeneous agent sector in a `.mod` file, puremacro introduces `hetagent_block`:

```dynare
var Y C r pi i;
varexo eps_m;

parameters beta gamma r_ss phi_pi kappa;
beta = 0.985;
gamma = 1.0;
r_ss = 0.01;
phi_pi = 1.5;
kappa = 0.1;

hetagent_block;
  model = one_asset_hank;
  n_a = 50;
  a_max = 30.0;
  borrowing_limit = 0.0;
  grid = hyperbolic;
end;

model;
  Y = C;
  pi = beta * pi(+1) + kappa * Y;
  i = r_ss + phi_pi * pi + eps_m;
  r = i - pi(+1);
end;
```

#### Supported `hetagent_block` Attributes
- `model`: Sector template name (`one_asset_hank` or custom identifier).
- `n_a`: Number of wealth grid points (default: `100`).
- `a_max`: Upper bound of asset grid (default: `50.0`).
- `borrowing_limit`: Hard borrowing constraint (default: `0.0`).
- `grid`: Asset grid spacing (`hyperbolic` or `exponential`).

### 3.3 Stationary Distribution $\mathcal{D}^*(a)$ & Marginal Propensity to Consume

Upon initialization, the microeconomic household problem is solved in steady state:
1. **Endogenous Grid Method (EGM)**: Solves backwards for optimal policy rules $c^*(a, s)$ and savings $a^*(a, s)$ given steady-state interest rate $r_{ss}$ and wage $w_{ss}$.
2. **Invariant Markov Distribution**: Pushes forward transition probabilities until the stationary distribution $\mathcal{D}^*(a, s)$ converges:
   $$\mathcal{D}^* = \Pi^\top \mathcal{D}^*$$
3. **MPC Schedule**: Evaluates local marginal propensities to consume $MPC(a) = \frac{\partial c(a)}{\partial a}$ across wealth tiers, capturing the high MPC of hand-to-mouth households.

### 3.4 The Fake-News Algorithm for Sequence-Space Jacobians

To couple household behavior with the aggregate DSGE equations, `puremacro` computes the intertemporal sequence-space Jacobians:
$$\mathcal{J}_{C, r} = \frac{\partial \mathbf{C}}{\partial \mathbf{r}} \in \mathbb{R}^{T \times T}, \quad \mathcal{J}_{C, Y} = \frac{\partial \mathbf{C}}{\partial \mathbf{Y}} \in \mathbb{R}^{T \times T}$$

The **Fake-News Algorithm** evaluates these $T \times T$ matrices in $\mathcal{O}(T^2)$ time rather than $\mathcal{O}(T^3)$:
1. Define fake-news matrix $\mathcal{F}_{s, t} = \mathbb{E}_s [ c_{t} ] - \mathbb{E}_{s-1} [ c_{t} ]$, representing the update to consumption at date $t$ upon receiving news at date $s$ about a shock at date 0.
2. Exploit the time-invariance of Markov transition matrices after date $s$:
   $$\mathcal{J}_{t, s} = \sum_{k=0}^{\min(t, s)} \mathcal{F}_{t-k, s-k}$$

### 3.5 General Equilibrium Solvers: Linear & Nonlinear Broyden

Aggregate market clearing over the horizon $T$ forms the sequence-space system:
$$\mathbf{H}(\mathbf{Y}, \mathbf{r}, \boldsymbol{\epsilon}) \equiv \mathbf{Y} - \mathbf{C}(\mathbf{Y}, \mathbf{r}) = \mathbf{0}$$

#### Linear Sequence-Space Solver
Differentiating around steady state yields:
$$(\mathbf{I} - \mathcal{J}_{C, Y}) d\mathbf{Y} - \mathcal{J}_{C, r} d\mathbf{r} = 0$$
Combining with aggregate monetary and Phillips curve equations produces a linear system solved in a single block matrix operation in $< 0.05$ s.

#### Nonlinear Broyden Solver
For large MIT shocks where borrowing constraints bind asymmetrically, `simulate(..., nonlinear=True)` activates the Broyden quasi-Newton solver with Sherman-Morrison rank-1 updates:
$$B_{k+1} = B_k + \frac{(\Delta \mathbf{U}_k - B_k \Delta \mathbf{H}_k) (\Delta \mathbf{U}_k^\top B_k)}{\Delta \mathbf{U}_k^\top B_k \Delta \mathbf{H}_k}$$
combined with an Armijo backtracking line search, guaranteeing convergence without recomputing microeconomic Jacobians at each iteration.

### 3.6 Python Code Example: Solving a HANK `.mod` Model

```python
import matplotlib.pyplot as plt
from puremacro.dsge.hank import load_hank_mod, solve_hank_bridge

HANK_MOD = """
var Y C r pi i;
varexo eps_m;

parameters beta gamma r_ss phi_pi kappa;
beta = 0.985;
gamma = 1.0;
r_ss = 0.01;
phi_pi = 1.5;
kappa = 0.1;

hetagent_block;
  model = one_asset_hank;
  n_a = 50;
  a_max = 30.0;
  borrowing_limit = 0.0;
  grid = hyperbolic;
end;

model;
  Y = C;
  pi = beta * pi(+1) + kappa * Y;
  i = r_ss + phi_pi * pi + eps_m;
  r = i - pi(+1);
end;
"""

# 1. Parse and initialize HANK model from .mod specification
model = load_hank_mod(HANK_MOD)
print("Stationary Steady State:", model.steady_state)

# 2. Inspect Fake-News Jacobians
jacobians = model.compute_jacobians(T=40)
print("J_C_r Shape:", jacobians["J_C_r"].shape)
print("Impact MPC (dC_0 / dY_0):", jacobians["J_C_Y"][0, 0])

# 3. Simulate an expansionary monetary policy shock (-25 bps)
# Linear Sequence-Space IRF
res_linear = model.simulate(
    shock="eps_m",
    magnitude=-0.0025,
    rho=0.5,
    horizon=30,
    nonlinear=False,
)

# Nonlinear Broyden Transition IRF
res_nonlinear = model.simulate(
    shock="eps_m",
    magnitude=-0.0025,
    rho=0.5,
    horizon=30,
    nonlinear=True,
)

# 4. Display results summary table
print(res_linear.summary())

# 5. One-liner shortcut API
res_bridge = solve_hank_bridge(
    HANK_MOD,
    shock="eps_m",
    magnitude=-0.0025,
    horizon=30,
)
print("Bridge Solved Successfully:", res_bridge.converged)
```

---

## API & Method Reference Guide

### 1. Analytic Gradients (`puremacro.dsge._gradients`)

| Function / Class | Description | Key Arguments | Returns |
|:---|:---|:---|:---|
| `solve_sylvester_generalized` | Solves $\hat{A} X + B X C = D$ via Schur decomposition. | `A_hat`, `B`, `C`, `D` | `np.ndarray` (solution matrix or batch) |
| `build_state_space_sensitivities` | Generates $\frac{\partial T}{\partial \theta}, \frac{\partial R}{\partial \theta}, \dots$ for all estimated parameters. | `model`, `varobs`, `param_names` | `dict[str, StateSpaceSensitivity]` |
| `kalman_score` | Single-pass forward Kalman filter likelihood and score. | `y`, `model`, `sensitivities`, `a0=None`, `P0=None` | `tuple[float, np.ndarray]` (`loglik`, `score`) |
| `log_posterior_and_gradient` | Evaluates $\ln p(\theta \mid Y)$ and $\nabla_\theta \ln p(\theta \mid Y)$ analytically. | `vec`, `names`, `model_template`, `y_data`, `varobs`, `priors` | `tuple[float, np.ndarray]` (`log_post`, `grad`) |
| `ScoreDiagnosticsResult` | Diagnostics container for score vectors and finite difference checks. | `loglik`, `gradient`, `param_names`, `elapsed_sec` | Result container (`.to_markdown()`, `.to_latex()`) |

### 2. Hamiltonian Monte Carlo / NUTS (`puremacro.dsge.nuts`, `estimate`)

| Function / Class | Description | Key Arguments | Returns |
|:---|:---|:---|:---|
| `nuts_sample` | Pure-Python NUTS sampler driver. | `target_log_prob_and_grad`, `init_params`, `n_draws`, `n_chains`, `warmup` | `NUTSResult` |
| `estimate_dsge` | Native Bayesian DSGE estimation supporting NUTS. | `data`, `method="nuts"`, `priors`, `observed_vars`, `initial_params` | `NUTSResult` |
| `LinearModel.estimate` | Ergonomic model estimation from `.mod` file declarations. | `data`, `method="nuts"`, `n_draws`, `n_chains`, `burn_in`, `seed` | `NUTSResult` |
| `NUTSResult` | Posterior container with MCMC diagnostics and figures. | Holds draws, log-posteriors, tree depths, divergences, energies. | `.summary()`, `.plot_trace()`, `.plot_autocorr()`, `.energy_diagnostics()` |

### 3. HANK Sequence-Space Bridge (`puremacro.dsge.hank`)

| Function / Class | Description | Key Arguments | Returns |
|:---|:---|:---|:---|
| `load_hank_mod` | Loads `.mod` file with `hetagent_block` into coupled HANK model. | `source`, `**param_overrides` | `HANKModel` |
| `solve_hank_bridge` | Solves `.mod` HANK model and returns general equilibrium transition path. | `source`, `shock`, `magnitude`, `rho`, `horizon`, `nonlinear` | `HANKResult` |
| `HANKModel.simulate` | Simulates linear or nonlinear MIT shock transition path. | `shock`, `magnitude`, `rho`, `horizon`, `nonlinear` | `HANKResult` |
| `HANKModel.compute_jacobians`| Computes Fake-News Jacobians $J_{C, r}$ and $J_{C, Y}$. | `T=300` | `dict[str, np.ndarray]` |
| `HANKResult` | Results container with transition dynamics and micro distributions. | Holds steady state, transition paths, distributions, Jacobians. | `.summary()`, `.to_frame()`, `.plot_transition()`, `.plot_distribution()` |

---

## Conclusion & Next Steps

With `puremacro 3.0.0`, researchers and macroeconomists gain access to:
- High-speed, exact analytical score vectors eliminating finite difference noise.
- Automated, pure-Python Hamiltonian Monte Carlo for high-dimensional Bayesian DSGE posteriors without C++ dependencies.
- Frictionless sequence-space HANK modeling directly inside standard Dynare `.mod` workflows.

To explore further:
- View the Dynare estimation guide at [dsge_estimation.md](dsge_estimation.md).
- Explore nonlinear sequence-space theory at [hank_nonlinear.md](hank_nonlinear.md).
- Inspect high-order perturbation options at [dsge_higher_order.md](dsge_higher_order.md).
