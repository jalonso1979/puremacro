[Español](es/dsge_phase_c.md) · **English**

# DSGE Frontier: Optimal Policy (Discretion vs Commitment), DSGE-VAR Hybrid Modeling, and Anticipated News Shocks

`puremacro` delivers three state-of-the-art computational frontiers for structural macroeconomic analysis:

1. **Optimal Policy Regimes (Discretion vs. Commitment)**: Solves Markov-perfect time-consistent discretionary policy via Dennis (2007) Riccati matrix iteration, alongside timeless-perspective linear-quadratic (LQ) commitment via Lagrange multiplier augmentation and Klein (2000) QZ decomposition. Formalizes the quantification of Kydland-Prescott / Barro-Gordon **inflation bias** and **stabilization bias**.
2. **DSGE-VAR Hybrid Modeling (Del Negro & Schorfheide 2004)**: Bridges structural DSGE microfoundations with flexible vector autoregressions via an analytical Normal-Inverted-Wishart prior centered on theoretical cross-equation moments $\Gamma_k(\theta)$. Provides closed-form evaluation of the log marginal data density $\ln p(Y \mid \lambda, \theta)$, bounded hyperparameter optimization for $\hat{\lambda} \in [\lambda_{\min}, \infty)$, and structural identification via the DSGE rotation matrix $Q^*$.
3. **News & Anticipated Shocks Engine (Beaudry & Portier 2006; Schmitt-Grohé & Uribe 2012)**: Implements companion state-space augmentation for forward-looking announcements $\epsilon_t = \eta_t^0 + \sum_{k=1}^H \eta_{t-k}^k$. Preserves Blanchard-Kahn saddle-path determinacy through nilpotent companion transition operators with zero eigenvalues, computes multi-lead impulse responses, and performs automated variance decompositions.

All algorithms are implemented in **pure Python** under the strict Pyodide four-package contract (`numpy`, `scipy`, `pandas`, `matplotlib`), requiring zero external solvers, zero C-extensions, and zero proprietary licenses.

---

## Methodological Comparison

| Dimension | Optimal Discretion | LQ Commitment | DSGE-VAR($\lambda$) | News Shocks Engine |
|:---|:---|:---|:---|:---|
| **Core Reference** | Dennis (2007); Oudiz & Sachs (1985) | Currie & Levine (1993); Woodford (2003) | Del Negro & Schorfheide (2004) | Beaudry & Portier (2006); Schmitt-Grohé & Uribe (2012) |
| **Equilibrium Concept** | Markov-perfect time-consistent Nash | Timeless perspective subgame-perfect | Bayesian conjugate posterior VAR | Rational expectations forward announcement |
| **State Space** | Physical predetermined states $x_{t-1}$ | Augmented with past shadow prices $\lambda_{t-1}$ | Observables lag companion $X_t$ | Augmented with news pipeline $V_t$ |
| **Computational Core** | Riccati matrix iteration $\|F_{k+1}-F_k\|_\infty < 10^{-9}$ | Augmented generalized Schur (Klein QZ) | Analytical Lyapunov + Inverted-Wishart | Nilpotent shift operator $K_H$ ($\sigma=\{0\}$) |
| **Primary Economic Insight** | Quantifies inflation bias and stabilization bias | Lower bound on quadratic loss under credibility | Quantifies structural misspecification $\hat{\lambda}$ | Distinguishes anticipation from realization |
| **Pyodide Compatible** | Yes (`numpy`, `scipy`) | Yes (`numpy`, `scipy`) | Yes (`numpy`, `scipy`) | Yes (`numpy`, `scipy`) |

---

## 1. Optimal Policy Regimes: Discretion vs Commitment

### 1.1 Mathematical Formulation

Consider a rational expectations structural DSGE model written in first-order linear state-space form:
$$A_0 y_t = A_1 y_{t-1} + A_2 \mathbb{E}_t y_{t+1} + B u_t + C \epsilon_t$$
where $y_t$ is partitioned into $n_x$ predetermined states $x_t$ and $n_z$ non-predetermined forward-looking controls $z_t$, $u_t$ is an $m$-dimensional vector of policy instruments (such as the nominal short-term interest rate $r_t$), and $\epsilon_t \sim \text{i.i.d.} \mathcal{N}(0, \Sigma_\epsilon)$.

The central bank minimizes the expected intertemporal quadratic welfare loss:
$$\mathcal{L}_t = \mathbb{E}_t \sum_{s=0}^\infty \beta^s \left[ \frac{1}{2} y_{t+s}^\top W y_{t+s} + (y_{t+s} - y^*)^\top \Omega (y_{t+s} - y^*) \right]$$
where $W$ and $\Omega$ are positive semi-definite target weighting matrices, $\beta \in (0, 1)$ is the policymaker discount factor, and $y^*$ represents the target distortion (e.g. output target exceeding the flexible-price natural rate).

### 1.2 Discretionary Policy (Dennis 2007)

Under discretion, the policymaker cannot commit to future actions. At each date $t$, the policymaker re-optimizes the instrument $u_t$ taking private sector expectation formation rules as given. The resulting Markov-perfect equilibrium is time-consistent.

The private sector forms forward-looking expectations as a linear function of predetermined states:
$$\mathbb{E}_t z_{t+1} = H x_t$$

The central bank's continuation value function satisfies the Bellman equation:
$$\mathcal{V}(x_{t-1}) = \frac{1}{2} x_{t-1}^\top V x_{t-1} + d$$
where the symmetric positive semi-definite Riccati matrix $V$ satisfies:
$$V = G^\top W_{\text{full}} G + \beta G^\top V G$$
with $G$ being the closed-loop state transition matrix $y_t = G x_{t-1} + N \epsilon_t$.

The policy reaction function converges to:
$$u_t = F x_{t-1}$$
via Dennis (2007) policy iteration:
$$\|F_{k+1} - F_k\|_\infty < 10^{-9}$$

### 1.3 Timeless Commitment

Under commitment, the central bank credibly commits from date $t_0 = -\infty$ to an optimal state-contingent rule. Introducing Lagrange multiplier vectors $\lambda_t$ on the structural forward-looking equations, the first-order necessary conditions yield an augmented system:
$$\begin{bmatrix} y_t \\ \lambda_t \end{bmatrix} = G_{\text{comm}} \begin{bmatrix} y_{t-1} \\ \lambda_{t-1} \end{bmatrix} + N_{\text{comm}} \epsilon_t$$
solved directly via the Klein (2000) generalized Schur (QZ) solver under the timeless perspective ($\lambda_{-1} = 0$).

### 1.4 Welfare Bias Decomposition

`puremacro` quantifies the two fundamental welfare losses identified in monetary economics:

1. **Inflation Bias**: Originating from Kydland & Prescott (1977) and Barro & Gordon (1983), when the central bank targets an output gap $y^* > 0$ above natural potential:
   $$\text{Bias}_{\pi} = \mathbb{E}[\pi^{\text{disc}}] - \mathbb{E}[\pi^{\text{comm}}] = \frac{\kappa \lambda_y}{\lambda_y (1 - \beta) + \kappa^2} y^*$$
2. **Stabilization Bias**: Originating from Clarida, Galí & Gertler (1999) and Woodford (2003), the discretionary policymaker cannot generate credible forward guidance (history dependence) to stabilize current inflation with lower output contraction. The stabilization bias is strictly positive:
   $$\text{Bias}_{\text{stab}} = \mathcal{L}^{\text{disc}} - \mathcal{L}^{\text{comm}} > 0$$

### 1.5 Runnable Python Example

```python
from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.policy import optimal_policy

# 1. Define 3-equation New Keynesian model
mod = """
var y pi r u;
varexo eps_u;
parameters beta sigma kappa phi_pi phi_y rho_u;
beta = 0.99; sigma = 1.0; kappa = 0.5; phi_pi = 1.5; phi_y = 0.5; rho_u = 0.5;
model;
  y = y(+1) - (1/sigma)*(r - pi(+1));
  pi = beta*pi(+1) + kappa*y + u;
  r = phi_pi*pi + phi_y*y;
  u = rho_u*u(-1) + eps_u;
end;
shocks; var eps_u; stderr 1.0; end;
"""
model = build_dynare(mod)

# 2. Solve Discretion vs Commitment
res = optimal_policy(
    model,
    loss={"pi": 1.0, "y": 0.25},
    rule="discretion",
    instruments="r",
    y_star=0.05,
    compare_commitment=True,
)

print(res.summary())
print(f"Inflation Bias     : {res.inflation_bias:.6f}")
print(f"Stabilization Bias : {res.stabilization_bias:.6f}")

# 3. Plot Comparison Impulse Responses
ax = res.plot(compare_commitment=True, periods=16)
ax.figure.savefig("output/dsge_optimal_discretion.png", bbox_inches="tight")
```

---

## 2. DSGE-VAR Hybrid Modeling (Del Negro & Schorfheide 2004)

### 2.1 Motivation & Conceptual Architecture

While structural DSGE models provide microfounded policy counterfactuals, they often exhibit econometric misspecification relative to flexible, unrestricted vector autoregressions (VARs). Del Negro & Schorfheide (2004) design a systematic Bayesian framework where the theoretical cross-equation restrictions of a DSGE model serve as an informative conjugate prior for an unrestricted $\text{VAR}(p)$:
$$Y_t = \Phi_0 + \sum_{l=1}^p \Phi_l Y_{t-l} + u_t, \quad u_t \sim \text{i.i.d.} \mathcal{N}(0, \Sigma)$$

The hyperparameter $\lambda \in [\lambda_{\min}, \infty)$ governs the weight assigned to the DSGE model:
- $\lambda \to \lambda_{\min}$: The hybrid model collapses to the unrestricted sample OLS VAR.
- $\lambda \to \infty$: The hybrid model converges continuously to the pure structural DSGE representation.
- $\hat{\lambda} = \arg\max_\lambda \ln p(Y \mid \lambda, \theta)$: The optimal weight provides a formal test of DSGE misspecification.

### 2.2 Theoretical Autocovariances & Prior Moments

Let the solved DSGE state-space system have state covariance $\Sigma_s$ satisfying the discrete Lyapunov equation:
$$\Sigma_s = G \Sigma_s G^\top + N \Sigma_\epsilon N^\top$$

The theoretical population autocovariances for observables $Y_t$ at lags $l = 0, \dots, p$ are:
$$\Gamma_{YY}^*(\theta) = \mathbb{E}_\theta [Y_t Y_t^\top] = Z \Sigma_s Z^\top$$
$$\Gamma_{YX}^*(\theta) = \mathbb{E}_\theta [Y_t X_t^\top] = \begin{bmatrix} Z G \Sigma_s Z^\top & \cdots & Z G^p \Sigma_s Z^\top \end{bmatrix}$$
$$\Gamma_{XX}^*(\theta) = \mathbb{E}_\theta [X_t X_t^\top]$$

The theoretical DSGE prior mean coefficients $\Phi^*(\theta)$ and innovation covariance $\Sigma^*(\theta)$ are given by:
$$\Phi^*(\theta) = \Gamma_{XX}^{*-1}(\theta) \Gamma_{XY}^*(\theta), \quad \Sigma^*(\theta) = \Gamma_{YY}^*(\theta) - \Gamma_{YX}^*(\theta) \Gamma_{XX}^{*-1}(\theta) \Gamma_{XY}^*(\theta)$$

### 2.3 Conjugate Normal-Inverted-Wishart Prior

The prior density takes the conjugate form:
$$p(\Sigma \mid \lambda, \theta) \sim \mathcal{IW}\left(\lambda T \Sigma^*(\theta), \; \lambda T - k\right)$$
$$p(\Phi \mid \Sigma, \lambda, \theta) \sim \mathcal{N}\left(\Phi^*(\theta), \; \Sigma \otimes (\lambda T \Gamma_{XX}^*(\theta))^{-1}\right)$$
where $k = n \cdot p + 1$ (with intercept) and $T$ is the effective sample size.

**Admissibility Condition**: To ensure the Inverted-Wishart prior is proper and integrable, $\lambda$ must strictly satisfy:
$$\lambda \ge \lambda_{\min} \equiv \frac{k + n}{T}$$

### 2.4 Analytical Log Marginal Data Density

The marginal data density $\ln p(Y \mid \lambda, \theta) = \int p(Y \mid \Phi, \Sigma) p(\Phi, \Sigma \mid \lambda, \theta) d\Phi d\Sigma$ has the exact closed-form expression:
$$\begin{aligned}
\ln p(Y \mid \lambda, \theta) = &-\frac{n T}{2} \ln(2\pi) + \ln \left( \frac{\Gamma_n((1 + \lambda)T - k)}{\Gamma_n(\lambda T - k)} \right) \\
&-\frac{n}{2} \ln \left| \frac{\lambda T \Gamma_{XX}^*(\theta) + X^\top X}{\lambda T \Gamma_{XX}^*(\theta)} \right| \\
&-\frac{(1 + \lambda)T - k}{2} \ln \left| (1 + \lambda)T \tilde{\Sigma}(\lambda) \right| \\
&+\frac{\lambda T - k}{2} \ln \left| \lambda T \Sigma^*(\theta) \right|
\end{aligned}$$
where $\ln \Gamma_n(a) = \frac{n(n-1)}{4} \ln \pi + \sum_{j=1}^n \ln \Gamma(a + \frac{1-j}{2})$ is the log multivariate gamma function.

### 2.5 Structural Identification via Rotation $Q^*$

To compute structural impulse response functions, the reduced-form VAR covariance $\tilde{\Sigma}$ is mapped to structural shocks via:
$$\tilde{A}_0 = \operatorname{chol}(\tilde{\Sigma}) \cdot Q^*$$
where $Q^*$ is an orthonormal matrix ($Q^* Q^{*\top} = I_n$) obtained via QR decomposition aligning the VAR impact matrix with the theoretical DSGE impact matrix $A_0^{\text{dsge}}$:
$$A_0^{\text{dsge}} = \operatorname{chol}(\Sigma^*) \cdot Q_{\text{dsge}} \implies Q^* = Q_{\text{dsge}}$$

### 2.6 Runnable Python Example

```python
from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.dsge_var import estimate_dsge_var

# 1. Compile DSGE model and simulate data
mod = """
var y pi r a u;
varexo eps_a eps_u eps_r;
parameters beta sigma kappa phi_pi phi_y rho_a rho_u;
beta = 0.99; sigma = 1.0; kappa = 0.5; phi_pi = 1.5; phi_y = 0.5;
rho_a = 0.70; rho_u = 0.70;
model;
  y = y(+1) - (1/sigma)*(r - pi(+1)) + (a(+1) - a);
  pi = beta*pi(+1) + kappa*y + u;
  r = phi_pi*pi + phi_y*y + eps_r;
  a = rho_a*a(-1) + eps_a;
  u = rho_u*u(-1) + eps_u;
end;
shocks;
  var eps_a; stderr 0.01;
  var eps_u; stderr 0.01;
  var eps_r; stderr 0.005;
end;
"""
model = build_dynare(mod)
sim_data = model.simulate(300, seed=42)[["y", "pi", "r"]]

# 2. Estimate DSGE-VAR(lambda) optimizing over grid
res = estimate_dsge_var(
    model,
    sim_data,
    p=1,
    lamb="optimal",
    lambda_grid=[0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0],
    identification="dsge",
)

print(res.summary())
print(f"Optimal hat_lambda : {res.hat_lambda:.4f}")

# 3. Forecasts and Structural IRFs
fc = res.forecast(horizon=8, ci=0.90)
fig, ax = res.plot(kind="irf", shock="eps_a", target="y", horizon=16)
fig.savefig("output/dsge_var_irf.png", bbox_inches="tight")
```

---

## 3. News & Anticipated Shocks Engine

### 3.1 Macroeconomic Foundations

Anticipated or "news" shocks represent information that economic agents receive at date $t$ regarding an exogenous innovation that will physically materialize at future date $t+k$ ($k \ge 1$). Standard macroeconomic applications include:
- Multi-year legislative delays in tax policy (Mertens & Ravn 2011).
- Anticipated technological breakthroughs or patent announcements (Beaudry & Portier 2006).
- Forward guidance and anticipated policy path communications (Campbell et al. 2012).

The exogenous shock process $\epsilon_t$ is decomposed into contemporaneous surprises and anticipated announcements:
$$\epsilon_t = \eta_t^0 + \sum_{k=1}^H \eta_{t-k}^k$$
where $\eta_t^0$ is the date-$t$ unexpected surprise shock, and $\eta_t^k$ is the date-$t$ news announcement regarding the shock realizing at date $t+k$.

### 3.2 Companion State-Space Augmentation

`puremacro` introduces an auxiliary news pipeline state vector $V_t = [\nu_{1, t}, \nu_{2, t}, \dots, \nu_{H, t}]^\top \in \mathbb{R}^H$ governed by the linear companion recursion:
$$V_t = K_H V_{t-1} + \eta_t^{\text{news}}$$
where the $H \times H$ companion matrix $K_H$ is strictly upper triangular with ones on the superdiagonal:
$$K_H = \begin{bmatrix}
0 & 1 & 0 & \cdots & 0 \\
0 & 0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \ddots & \vdots \\
0 & 0 & \cdots & 0 & 1 \\
0 & 0 & \cdots & 0 & 0
\end{bmatrix}$$

**Zero-Eigenvalue Invariance**: Because $K_H$ is strictly upper triangular, its characteristic polynomial is:
$$\det(\lambda I_H - K_H) = \lambda^H = 0 \implies \sigma(K_H) = \{0, 0, \dots, 0\}$$
All $H$ augmented eigenvalues are identically zero, lying strictly inside the unit circle ($|\lambda_j| = 0 < 1$). Therefore, news augmentation **strictly preserves Blanchard-Kahn saddle-path determinacy** without altering the model's finite economic eigenvalues.

### 3.3 The Three Cardinal Properties of News Shocks

1. **Zero Revision for Predetermined States ($t < k$)**:
   Physical state variables (such as capital stock $k_t$ or exogenous productivity $a_t$) satisfy:
   $$\Delta x_t = 0 \quad \forall \; 0 \le t < k$$
2. **Immediate Jump of Forward-Looking Controls ($t = 0$)**:
   Forward-looking variables (output gap $y_0$, inflation $\pi_0$, consumption $c_0$) immediately jump at impact:
   $$z_0 - z_{ss} = L_{\text{aug}} \cdot \eta^k \neq 0$$
3. **Exact Physical Realization at Date $t = k$**:
   At horizon $t = k$, the innovation reaches the front of the companion queue ($\nu_{1, k} = \text{size}$), triggering the physical state expansion.

### 3.4 Automated Variance Decomposition

`decompose_news` quantifies the fraction of forecast error variance attributable to surprise vs. news announcements:
$$\text{FEVD}_v(h) = \frac{\sum_{j=0}^h \left( \text{IRF}_{v, \text{surp}}(j) \right)^2}{\text{Total Var}_v(h)} + \sum_{k=1}^H \frac{\sum_{j=0}^h \left( \text{IRF}_{v, \text{news-}k}(j) \right)^2}{\text{Total Var}_v(h)} = 1.0000$$

### 3.5 Dynare `.mod` Anticipated Shocks Syntax

`puremacro` natively parses Dynare-style anticipated shock declarations:
```dynare
shocks;
  var eps_a;
  periods 1:4;
  values 0.01;
end;
```

### 3.6 Runnable Python Example

```python
from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.news import news_irf, decompose_news, plot_news_vs_surprise

mod = """
var y pi r a;
varexo eps_a;
parameters beta sigma kappa phi_pi rho_a;
beta = 0.99; sigma = 1.0; kappa = 0.5; phi_pi = 1.5; rho_a = 0.85;
model;
  y = y(+1) - (1/sigma)*(r - pi(+1)) + (a(+1) - a);
  pi = beta*pi(+1) + kappa*y;
  r = phi_pi*pi;
  a = rho_a*a(-1) + eps_a;
end;
shocks; var eps_a; stderr 0.01; end;
"""
model = build_dynare(mod)

# 1. Compute 4-quarter lead news shock
res_news = news_irf(model, shock="eps_a", lead=4, horizon=20)
print(res_news.summary())

# 2. Decompose variance across surprise and leads 1..8
decomp = decompose_news(model, shock="eps_a", horizon=20, max_lead=8)
print("\nVariance Shares (sum to 1.0):")
print(decomp.variance_shares.round(4))

# 3. Multi-lead trajectory comparison
fig, axes = plot_news_vs_surprise(model, shock="eps_a", leads=[0, 2, 4, 8])
fig.savefig("output/dsge_news_shocks.png", bbox_inches="tight")
```

---

## 4. API Reference Table

### 4.1 Functions & Constructors

| Function / Constructor | Module | Primary Parameters | Return Type | Description |
|:---|:---|:---|:---|:---|
| `optimal_policy(model, loss, rule, ...)` | `puremacro.dsge.policy` | `model`, `loss`, `rule="discretion"\|"commitment"`, `instruments`, `y_star`, `tol=1e-9` | `DiscretionaryPolicyResult` or `PolicyResult` | Top-level dispatcher solving optimal discretion (Dennis 2007) or timeless LQ commitment. |
| `discretionary_policy(model, ...)` | `puremacro.dsge.policy` | `model`, `target_vars`, `weights`, `instruments`, `beta=0.99`, `y_star`, `tol=1e-9` | `DiscretionaryPolicyResult` | Dennis (2007) Markov-perfect policy iteration algorithm. |
| `lq_commitment(model, ...)` | `puremacro.dsge.policy` | `model`, `target_vars`, `weights`, `instruments`, `beta=0.99` | `PolicyResult` | Timeless perspective linear-quadratic optimal commitment policy via Klein QZ. |
| `estimate_dsge_var(model, data, ...)` | `puremacro.dsge.dsge_var` | `model`, `data`, `p=4`, `lamb=None`, `identification="dsge"`, `lambda_grid=None` | `DSGEVARResult` | Del Negro & Schorfheide (2004) DSGE-VAR estimator and hyperparameter optimizer. |
| `news_irf(model, shock, lead, ...)` | `puremacro.dsge.news` | `model`, `shock`, `lead=0`, `horizon=40`, `size=1.0` | `NewsIRFResult` | Companion state-space augmented impulse responses for anticipated news shocks. |
| `decompose_news(model, shock, ...)` | `puremacro.dsge.news` | `model`, `shock=None`, `horizon=40`, `max_lead=8` | `NewsDecompositionResult` | Automated forecast error variance decomposition across surprise and news leads. |
| `plot_news_vs_surprise(model, shock, ...)` | `puremacro.dsge.news` | `model`, `shock`, `leads=(0, 2, 4, 8)`, `variables=None`, `horizon=40` | `tuple[Figure, Any]` | Multi-lead trajectory comparison visualization helper. |

### 4.2 Result Data Structures

| Result Class | Module | Key Attributes | Presentation Methods |
|:---|:---|:---|:---|
| `DiscretionaryPolicyResult` | `puremacro.dsge._results` | `F`, `V`, `inflation_bias`, `stabilization_bias`, `converged`, `iterations`, `diff`, `commitment_result` | `.summary()`, `.plot(compare_commitment=True)`, `.to_frame()`, `.to_markdown()`, `.to_latex()`, `.to_typst()` |
| `DSGEVARResult` | `puremacro.dsge.dsge_var` | `lamb`, `hat_lambda`, `lambda_min`, `log_mdd`, `log_mdd_grid`, `Phi_star`, `Sigma_star`, `Phi_ols`, `Sigma_ols`, `B0` | `.summary()`, `.plot(kind="irf"\|"mdd"\|"forecast")`, `.irf()`, `.forecast()`, `.fevd()`, `.to_frame()`, `.to_markdown()`, `.to_latex()`, `.to_typst()` |
| `NewsIRFResult` | `puremacro.dsge.news` | `irf`, `surprise_irf`, `shock`, `lead`, `horizon`, `size`, `model` | `.summary()`, `.plot(compare_surprise=True)`, `.to_frame()`, `.to_markdown()`, `.to_latex()`, `.to_typst()` |
| `NewsDecompositionResult` | `puremacro.dsge.news` | `variance_shares`, `dynamic_shares`, `shock`, `horizon`, `max_lead`, `model` | `.summary()`, `.plot()`, `.to_frame()`, `.to_markdown()`, `.to_latex()`, `.to_typst()` |

---

## References

- **Barro, R. J., & Gordon, D. B. (1983).** "Rules, discretion and reputation in a model of monetary policy." *Journal of Monetary Economics*, 12(1), 101-121.
- **Beaudry, P., & Portier, F. (2006).** "Stock Prices, News, and Economic Fluctuations." *American Economic Review*, 96(4), 1293-1307.
- **Clarida, R., Galí, J., & Gertler, M. (1999).** "The science of monetary policy: A New Keynesian perspective." *Journal of Economic Literature*, 37(4), 1661-1707.
- **Currie, D., & Levine, P. (1993).** *Rules, Reputation and Macroeconomic Policy Coordination*. Cambridge University Press.
- **Del Negro, M., & Schorfheide, F. (2004).** "Priors from General Equilibrium Models for VARs." *International Economic Review*, 45(2), 643-673.
- **Dennis, R. (2007).** "Optimal Policy in Rational Expectations Models: New Solution Algorithms." *Macroeconomic Dynamics*, 11(1), 31-55.
- **Klein, P. (2000).** "Using the generalized Schur form to solve a multivariate linear rational expectations model." *Journal of Economic Dynamics and Control*, 24(10), 1405-1423.
- **Kydland, F. E., & Prescott, E. C. (1977).** "Rules rather than discretion: The inconsistency of optimal plans." *Journal of Political Economy*, 85(3), 473-491.
- **Oudiz, G., & Sachs, J. (1985).** "International Policy Coordination in Dynamic Macroeconomic Models." In *International Economic Policy Coordination*, Cambridge University Press.
- **Schmitt-Grohé, S., & Uribe, M. (2012).** "What's News in Business Cycles." *Econometrica*, 80(6), 2733-2764.
- **Woodford, M. (2003).** *Interest and Prices: Foundations of a Theory of Monetary Policy*. Princeton University Press.
