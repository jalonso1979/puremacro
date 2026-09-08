# `puremacro.dsge` Tier 2 — Higher Order & Constraints

**Status:** Drafted 2026-09-08. Architectural specification for Tier 2 of the Dynare-parity roadmap (`docs/plans/2026-09-07-puremacro-dsge-dynare-parity-roadmap.md`).  
**Target release:** 2.8.0 (Phases A, B, C, D, E).  
**Baseline:** puremacro 2.7.0 (AST expression DAG, native macro preprocessor, analytical Jacobians and Hessians with CSE, Schur Sylvester solver).  
**Driving lens:** Enable modern macroeconomists to solve, simulate, and estimate models with nonlinearities, risk premia, and structural constraints (such as the Zero Lower Bound and borrowing caps) under the strict Pyodide four-package contract (`numpy`, `scipy`, `pandas`, `matplotlib`).

---

## 1. Motivation & Background

With `puremacro 2.7.0`, the DSGE engine acquired a production-grade front-end: an expression DAG, macro preprocessor, analytical symbolic differentiation with common-subexpression elimination (CSE), and a fast generalized Schur Sylvester solver delivering a 165x speedup on second-order perturbation.

Higher-order perturbation and occasionally binding constraints represent the next frontier of macroeconomic modeling:

1. **Risk Premia & Asset Pricing Require Order 3**:
   - At first order, certainty equivalence holds ($\sigma = 0$).
   - At second order, uncertainty shifts ergodic means ($g_{\sigma\sigma}$), but does not alter policy slopes or induce time-varying risk premia ($g_{x\sigma\sigma} = 0$).
   - At third order, cross-terms between states and shock variances ($g_{x\sigma\sigma}$) appear, generating state-dependent risk premia, stochastic volatility spillovers, and precautionary asset market dynamics.
   - Unpruned higher-order simulations diverge exponentially outside a tight steady-state basin. The pruning algorithm of Andreasen, Fernández-Villaverde, and Rubio-Ramírez (2018) solves this, but requires exact 3rd-order tensor evaluations and recursive state decomposition.

2. **Terminal Transitions & ZLB in Deterministic Models**:
   - `puremacro.dsge.perfect_foresight` currently assumes a single steady-state terminal vector $y_{ss}$. Real-world policy questions (e.g. permanent tax reforms, demographic shifts, sovereign debt resets) transition from an initial state $y_{init}$ to a *different* terminal steady state $y_{end}$ declared via `histval` and `endval`.
   - The Zero Lower Bound (ZLB) in deterministic simulations requires solving a Mixed Complementarity Problem (MCP):
     $$i_t \ge 0 \perp (i_t - i_t^{Taylor}) \ge 0$$
     Dynare solves this via semismooth Newton using the Fischer-Burmeister operator. Stacked sparse Newton must be upgraded to support complementarity conditions directly.

3. **OccBin Multi-Constraint & Estimation**:
   - Guerrieri & Iacoviello (2015) piecewise-linear relaxation is currently limited to two regimes (unconstrained and one constrained).
   - Real macroeconomic crises involve *simultaneous* constraints (e.g., ZLB on the policy rate *and* a collateral constraint on banks, creating $2^2 = 4$ distinct regimes).
   - Furthermore, estimating models with occasionally binding constraints historically required bespoke filters. The **Piecewise Kalman Filter** (Giovannini, Pfeiffer, Ratto 2021) enables likelihood-based estimation of OccBin models directly.

4. **Sequential Monte Carlo (SMC)**:
   - Random Walk Metropolis-Hastings (RWMH) struggles when the posterior is multimodal, irregular, or when parameters are weakly identified.
   - Herbst & Schorfheide (2014, 2015) Sequential Monte Carlo bridges prior to posterior across tempering stages $\phi_n \in [0, 1]$, mutates particles adaptively, and yields an unbiased estimator of the Marginal Data Density (MDD) as a natural product of normalizing constants.

5. **Nonlinear Ramsey Optimal Policy**:
   - Calculating optimal monetary or fiscal policy under commitment requires setting up the planner's Lagrangian and differentiating with respect to all endogenous variables and multipliers. The AST DAG from 2.7.0 provides the foundation to automate this symbolic expansion.

---

## 2. Architecture & Module Map

```
puremacro/dsge/
├── _ast.py                [ext]  Support 3rd-order differentiation, complementarity tags
├── _symbolic.py           [ext]  3rd-order dynamic tensor derivatives with CSE
├── _sylvester.py          [ext]  Schur-Sylvester extension for 3-fold Kronecker products
├── pruning.py             [ext]  3rd-order pruned perturbation state space (Andreasen et al. 2018)
├── perfect_foresight.py   [ext]  histval/endval terminal paths, MCP semismooth Newton
├── occbin.py              [ext]  Multi-constraint OccBin (2^K regimes), Piecewise Kalman Filter
├── smc.py                 [new]  Sequential Monte Carlo (Herbst & Schorfheide) sampler
├── ramsey.py              [new]  Nonlinear Ramsey optimal policy Lagrangian constructor
├── detrending.py          [new]  Balanced Growth Path (BGP) detrending and stationarization
├── _results.py            [ext]  Order3PrunedSolution, SMCResult, MCPResult, RamseyResult
└── dynare.py              [ext]  Grammar parsing for histval, endval, varexo_det, mcp tags, occbin_constraints
```

---

## 3. Detailed Specification by Phase

### Phase A: 3rd-Order Perturbation with Pruning (Andreasen et al. 2018)

#### A1. Symbolic 3rd Derivatives & Tensor Representation
Over the dynamic model $f(y_{t+1}, y_t, y_{t-1}, u_t) = 0$, evaluate the 3rd derivative tensor $\mathcal{T}_f$:
$$\mathcal{T}_{f, i, j, k, l} = \frac{\partial^3 f_i}{\partial z_j \partial z_k \partial z_l}$$
where $z = [y_{t+1}^\top, y_t^\top, y_{t-1}^\top, u_t^\top]^\top$.

- **Symmetry Exploitation**: Permutations of $(j, k, l)$ are identical. Evaluate only non-decreasing indices $j \le k \le l$, reducing evaluations by a factor of 6.
- **Topological CSE**: Extend CSE pass to identify shared subexpressions in 3rd-order Hessian derivatives.
- **Contracted Evaluation**: Compute contractions with first-order decision vectors directly in compiled code to avoid storing dense $N \times (3N + n_e)^3$ arrays.

#### A2. Order-3 Policy Function & Schur Sylvester Recursion
The third-order policy rule takes the form:
$$y_t = y_{ss} + g_x \hat{x}_{t-1} + g_u u_t + \frac{1}{2} g_{xx} (\hat{x}_{t-1} \otimes \hat{x}_{t-1}) + \dots + \frac{1}{6} g_{xxx} (\hat{x}_{t-1} \otimes \hat{x}_{t-1} \otimes \hat{x}_{t-1}) + \frac{1}{2} g_{x\sigma\sigma} \hat{x}_{t-1} \sigma^2 + \dots$$

The state curvature tensor $g_{xxx}$ satisfies the generalized Sylvester equation:
$$(A_0 + A_+ g_x P_s) g_{xxx} + A_+ g_{xxx} (h_x \otimes h_x \otimes h_x) = -K_{xxx}$$
where $K_{xxx}$ collects known contractions of first- and second-order policy matrices with 3rd model derivatives.

- **Schur Decomposition**: Since $h_x = U_h T_h U_h^H$ with upper-triangular $T_h$, the 3-fold Kronecker product:
  $$h_x \otimes h_x \otimes h_x = (U_h \otimes U_h \otimes U_h) (T_h \otimes T_h \otimes T_h) (U_h \otimes U_h \otimes U_h)^H$$
  is upper triangular with diagonal elements $t_{ii} t_{jj} t_{kk}$.
- **Decoupled Column Substitution**: Solve for column $(i, j, k)$ by back-substitution with diagonal block LU factorizations, maintaining $O(n_x^3 \cdot n)$ complexity and $< 5$ MB memory.

#### A3. Third-Order Pruning State Space
The Andreasen, Fernández-Villaverde, and Rubio-Ramírez (2018) state space expands into three components:
$$x_t^{(1)} = G x_{t-1}^{(1)} + N u_t$$
$$x_t^{(2)} = G x_{t-1}^{(2)} + \frac{1}{2} H_{xx} (x_{t-1}^{(1)} \otimes x_{t-1}^{(1)}) + H_{xu} (x_{t-1}^{(1)} \otimes u_t) + \frac{1}{2} H_{uu} (u_t \otimes u_t) + \frac{1}{2} H_{\sigma\sigma} \sigma^2$$
$$x_t^{(3)} = G x_{t-1}^{(3)} + H_{xx} (x_{t-1}^{(1)} \otimes x_{t-1}^{(2)}) + \frac{1}{6} H_{xxx} (x_{t-1}^{(1)} \otimes x_{t-1}^{(1)} \otimes x_{t-1}^{(1)}) + \frac{1}{2} H_{x\sigma\sigma} x_{t-1}^{(1)} \sigma^2 + \dots$$

- Total state deviation: $\hat{x}_t = x_t^{(1)} + x_t^{(2)} + x_t^{(3)}$.
- **Ergodic Stability**: Evaluated strictly on lower-order components; completely suppresses explosive polynomial divergent paths.
- **Analytical Moments**: Compute unconditional 1st, 2nd, 3rd, and 4th moments (mean, variance, skewness, kurtosis) under pruning analytically via Kronecker products of $G$.

---

### Phase B: Perfect Foresight, Boundary Conditions & MCP

#### B1. `histval` and `endval` Transitions
- `histval; ... end;`: Specifies initial conditions $y_0, y_{-1}$ distinct from the steady state.
- `endval; ... end;`: Specifies permanent terminal conditions. If permanent shocks are present, the solver executes a numerical steady-state solve for the terminal steady state $y_{end}$ before setting up the stacked system:
  $$f(y_1, y_0, \dots) = 0, \quad \dots, \quad f(y_{end}, y_T, y_{T-1}) = 0$$
- `varexo_det`: Tracks deterministic exogenous paths that are known to agents in advance across all $t=1, \dots, T$.
- `shocks(surprise)`: Unanticipated shock sequences where agents re-solve the perfect foresight path period-by-period from $t=1$ to $T$.

#### B2. Mixed Complementarity Problem (MCP) via Semismooth Newton
In models with constraints (e.g. ZLB $i_t \ge 0$):
$$f_i(y_{t+1}, y_t, y_{t-1}) \le 0 \perp y_{i, t} \ge \underline{y}_i$$

Reformulate as root-finding using the **Fischer-Burmeister** function:
$$\Phi(a, b) = a + b - \sqrt{a^2 + b^2} = 0 \iff a \ge 0, b \ge 0, ab = 0$$

- **Generalized Jacobian (Clarke Subdifferential)**:
  $$\partial \Phi(a, b) = \left( 1 - \frac{a}{\sqrt{a^2 + b^2}}, 1 - \frac{b}{\sqrt{a^2 + b^2}} \right) \quad \text{for } (a, b) \ne (0, 0)$$
  and $(1 - \xi_1, 1 - \xi_2)$ with $\xi_1^2 + \xi_2^2 \le 1$ at $(0, 0)$.
- **Stacked Semismooth Newton**: Directly augments the block-tridiagonal Jacobian matrix $J_{stacked}$ with the subdifferential weights, solved using `scipy.sparse.linalg.spsolve`.

---

### Phase C: Multi-Constraint OccBin & Piecewise Kalman Filter

#### C1. Multi-Constraint OccBin ($2^K$ Regimes)
Extend Guerrieri & Iacoviello (2015) to $K \ge 2$ occasionally binding constraints:
- Define boolean regime vector $R_t \in \{0, 1\}^K$ for each period $t \in [1, T]$.
- Pre-compute or dynamically solve the first-order Klein matrices $(A_0^r, A_+^r, A_-^r, B_u^r)$ for all active regimes $r \in \{0, \dots, 2^K - 1\}$.
- Backward recursion updates the time-varying state-space matrices:
  $$P_t = - (A_0^{r_t} + A_+^{r_t} P_{t+1})^{-1} A_-^{r_t}$$
  $$D_t = - (A_0^{r_t} + A_+^{r_t} P_{t+1})^{-1} (A_+^{r_t} D_{t+1} + c^{r_t} + B_u^{r_t} u_t)$$
- Iterate over the regime matrix until convergence across all $K$ constraints simultaneously.

#### C2. Piecewise Kalman Filter (Giovannini, Pfeiffer, Ratto 2021)
Enables estimation of models with occasionally binding constraints:
1. At each observation date $t$, given filtered state $x_{t-1|t-1}$, determine the expected future regime sequence over horizon $H$.
2. Compute time-varying transition matrix $T_t(R)$ and constant $C_t(R)$ from the OccBin backward recursion.
3. Compute forecast errors $v_t = y_t^{obs} - Z x_{t|t-1} - d_t$.
4. Update state $x_{t|t}$ and covariance $P_{t|t}$ using standard Kalman gain formulas evaluated on the piecewise matrices.
5. Invert shocks to verify regime consistency; iterate until the expected regime matches the realized bound state.

---

### Phase D: Sequential Monte Carlo (SMC) & Particle Filtering

#### D1. Herbst & Schorfheide (2014, 2015) SMC Algorithm
- **Tempering Schedule**: Define $N_\phi$ stages $\phi_0 = 0 < \phi_1 < \dots < \phi_{N_\phi} = 1$, where $\phi_n = (n / N_\phi)^\lambda$.
- **Particles**: Maintain $N_{part}$ parameter vectors $\{\theta_i\}_{i=1}^{N_{part}}$.
- **Stage Progression**:
  1. **Correction**: Compute incremental weights:
     $$\tilde{w}_{i, n} = \exp\left( (\phi_n - \phi_{n-1}) \ln \mathcal{L}(Y | \theta_{i, n-1}) \right)$$
     Normalize weights $W_{i, n} = \tilde{w}_{i, n} / \sum_{j} \tilde{w}_{j, n}$.
  2. **Effective Sample Size (ESS)**:
     $$ESS_n = \frac{N_{part}}{\sum_{i=1}^{N_{part}} W_{i, n}^2}$$
     If $ESS_n < N_{part} / 2$, resample particles via systematic resampling.
  3. **Mutation**: Mutate each particle via $N_{steps}$ Metropolis-Hastings steps using proposal covariance $\Sigma_n = c_n^2 \sum_{i} W_{i, n} (\theta_i - \bar{\theta}_n)(\theta_i - \bar{\theta}_n)^\top$. Adapt scale $c_n$ to target an acceptance rate of ~25%–30%.
- **Marginal Data Density (MDD)**:
  $$\ln \hat{p}(Y) = \sum_{n=1}^{N_\phi} \ln \left( \sum_{i=1}^{N_{part}} W_{i, n-1} \tilde{w}_{i, n} \right)$$
  Computes MDD standard error via across-run variance or asymptotic variance formulas.

#### D2. Bootstrap Particle Filter on Pruned DSGE Solutions
- For nonlinear order-2 and order-3 pruned state spaces, evaluate likelihoods via the bootstrap particle filter:
  - Propagate particles through the pruned recursion $x_t = g(x_{t-1}, u_t)$.
  - Weight particles by measurement density $p(y_t^{obs} | x_t) \sim \mathcal{N}(H x_t, R)$.
  - Systematic resampling on low ESS.

---

### Phase E: Nonlinear Ramsey Policy & BGP Detrending

#### E1. Nonlinear Ramsey Optimal Policy (`ramsey_model`)
- **Planner Objective**:
  $$\max_{\{y_t, u_t\}_{t=0}^\infty} \mathbb{E}_0 \sum_{t=0}^\infty \beta^t U(y_t, u_t)$$
  subject to structural equilibrium conditions:
  $$f(y_{t+1}, y_t, y_{t-1}, u_t) = 0$$
- **Lagrangian Construction over AST**:
  $$\mathcal{L} = \sum_{t=0}^\infty \beta^t \left[ U(y_t, u_t) + \lambda_t^\top f(y_{t+1}, y_t, y_{t-1}, u_t) \right]$$
- **Symbolic First-Order Conditions**:
  $$\frac{\partial \mathcal{L}}{\partial y_t} = \frac{\partial U}{\partial y_t} + \lambda_t^\top \frac{\partial f_t}{\partial y_t} + \beta^{-1} \lambda_{t-1}^\top \frac{\partial f_{t-1}}{\partial y_{t+1}} + \beta \lambda_{t+1}^\top \frac{\partial f_{t+1}}{\partial y_{t-1}} = 0$$
  $$\frac{\partial \mathcal{L}}{\partial \lambda_t} = f(y_{t+1}, y_t, y_{t-1}, u_t) = 0$$
- **Augmented System**: Automatically generates the combined model DAG of $N$ original equations plus $N$ multiplier FOCs, solved via the first-order Klein solver.

#### E2. Balanced Growth Path (BGP) Detrending
- Parse `trend_var` and `log_trend_var` blocks.
- Associate growth factors with non-stationary variables:
  $$Y_t = y_t \cdot \prod_{s=1}^t \Gamma_s$$
- Automatically perform stationarizing substitutions into the AST DAG before differentiation and steady-state solution.

---

## 4. Presentation Contract & Results Objects

All newly introduced result structures adhere strictly to puremacro's frozen dataclass presentation contract:

| Class | Module | Methods |
|---|---|---|
| `Order3PrunedSolution` | `puremacro.dsge.pruning` | `.summary()`, `.girf()`, `.stoch_simul()`, `.unconditional_moments()`, `.to_markdown()`, `.to_latex()`, `.to_typst()` |
| `MCPResult` | `puremacro.dsge.perfect_foresight` | `.summary()`, `.plot()`, `.binding_periods()`, `.to_markdown()`, `.to_latex()`, `.to_typst()` |
| `PiecewiseKalmanResult` | `puremacro.dsge.occbin` | `.summary()`, `.log_likelihood()`, `.filtered_states()`, `.regime_history()`, `.to_markdown()` |
| `SMCResult` | `puremacro.dsge.smc` | `.summary()`, `.plot_stages()`, `.marginal_likelihood()`, `.posterior_table()`, `.to_markdown()`, `.to_latex()` |
| `RamseyResult` | `puremacro.dsge.ramsey` | `.summary()`, `.planner_foc()`, `.multipliers()`, `.decision_rules()`, `.to_markdown()`, `.to_latex()` |

---

## 5. Acceptance Criteria & Verification Gates

### Phase A Verification (Order 3 Pruning)
- [ ] 3rd-order analytical derivatives match numerical 6th-order central difference within $10^{-9}$ tolerance.
- [ ] Order-3 Schur Sylvester solver computes $g_{xxx}$ on SW07 in $\le 0.8$ seconds with $< 5$ MB memory.
- [ ] Pruned order-3 simulation runs 10,000 periods with 0 explosive trajectories and verified stationary kurtosis.

### Phase B Verification (Perfect Foresight & MCP)
- [ ] Transition between distinct initial and terminal steady states (`histval` $\to$ `endval`) converges to $\le 10^{-10}$ residual norm.
- [ ] Fischer-Burmeister semismooth Newton correctly enforces the ZLB ($i_t \ge 0$) in an NK model with a large deflationary shock, perfectly matching Dynare's deterministic simulation path.

### Phase C Verification (OccBin & Piecewise Kalman)
- [ ] Multi-constraint OccBin correctly tracks 4 regimes in a model with simultaneous ZLB and borrowing constraints.
- [ ] Piecewise Kalman Filter estimates structural parameters on synthetic ZLB data, recovering true parameters within 95% confidence intervals.

### Phase D Verification (SMC & Particle Filter)
- [ ] SMC accurately explores a bimodal mixture posterior where standard RWMH gets trapped in a single mode.
- [ ] SMC marginal likelihood estimate matches analytical benchmark on linear-Gaussian DSGE within $0.10$ log-points.

### Phase E Verification (Nonlinear Ramsey & BGP)
- [ ] `ramsey_model` analytically derives the timeless-perspective Taylor principle and optimal price-stabilization commitment rules.
- [ ] Trended RBC model stationarizes automatically and reproduces stationary moments.

### Release Standards
- [ ] 100% pure Python under the Pyodide four-package contract (`numpy`, `scipy`, `pandas`, `matplotlib` only).
- [ ] Full regression suite passes with 0 failures; `tools/release_check.py` gates clean.
- [ ] Bilingual documentation synced in `docs/` and `docs/es/`.
