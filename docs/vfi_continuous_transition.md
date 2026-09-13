> 🇬🇧 English · 🇪🇸 [Español](es/vfi_continuous_transition.md)

# Continuous Transition Dynamics under MIT Shocks

`puremacro.vfi.continuous_transition` implements non-linear general equilibrium transition dynamics for continuous-state heterogeneous-agent macroeconomic models following unexpected aggregate innovations (**MIT shocks**). The engine couples backward continuous Endogenous Grid Method (EGM) policy recursion with forward non-stochastic density evolution (**Young 2010**), clearing sequence-space goods and asset markets simultaneously via a specialized Broyden Quasi-Newton solver (**Auclert et al. 2021**).

While first-order linearized perturbations provide local approximations near the deterministic steady state, they fail to capture critical non-linearities triggered by large policy interventions—such as sovereign rate spikes, emergency fiscal stimulus, or structural productivity shocks. Under such shocks, asymmetric borrowing constraints bind, precautionary motives intensify, and the cross-sectional wealth distribution $\mu_t(k, z)$ undergoes significant non-linear reallocations. `puremacro` computes the **exact non-linear equilibrium transition path** over arbitrary horizons $T$ with strict mass conservation ($\sum \mu_t = 1.0 \pm 10^{-12}$) at every date.

---

## 1. Theoretical & Algorithmic Framework

### 1.1 The Sequence-Space Transition Problem

Consider an economy initialized at a stationary steady-state distribution $\mu_0(k, z)$ at date $t = 0$. An unexpected, deterministic sequence of aggregate shocks $\mathbf{Z} = (Z_0, Z_1, \dots, Z_{T-1})$ strikes the economy at $t = 0$, where $T$ is chosen sufficiently large that the system converges to a terminal stationary steady state $\mu_T(k, z)$ by period $T$.

General equilibrium is defined as sequences of factor prices $\{r_t, w_t\}_{t=0}^{T-1}$, household policy functions $\{c_t(k, z), a'_t(k, z)\}_{t=0}^{T-1}$, and continuous wealth distributions $\{\mu_t(k, z)\}_{t=0}^{T}$ such that:
1. Households optimize backward taking the sequence of interest rates and wages as given.
2. The cross-sectional distribution evolves forward under time-varying decision rules.
3. Competitive firms maximize profits, and the capital market clears at every date $t \in [0, T-1]$:

$$H_t(\mathbf{r}) \equiv K_t^s(\mathbf{r}) - K_t^d(r_t; Z_t) = 0$$

where aggregate capital supply is the continuous cross-sectional asset integral:

$$K_t^s = \sum_{m=1}^{n_z} \int_{\underline{k}}^{\bar{k}} k \, \mu_t(dk, z_m)$$

### 1.2 Backward Continuous EGM Policy Recursion

Given a candidate sequence of interest rates $\mathbf{r} = (r_0, r_1, \dots, r_{T-1})$ and implied competitive wages $w_t = (1 - \alpha) Z_t (K_t^d / \bar{L})^\alpha$, household consumption and savings policies are solved **backward in time** from $t = T-1$ down to $t = 0$ using the Endogenous Grid Method (EGM) without discretization bias.

At date $t$, given the continuation consumption policy $c_{t+1}(a', z')$, the Euler equation defines expected marginal utility:

$$\mathcal{EMU}_t(a', z_j) = \beta (1 + r_{t+1}) \sum_{m=1}^{n_z} P_z(j, m) \cdot u'\left( c_{t+1}(a', z_m) \right)$$

Inverting the marginal utility function yields endogenous consumption $c_t^{\text{endo}}(a', z_j) = (u')^{-1}(\mathcal{EMU}_t(a', z_j))$. The endogenous pre-decision asset state is obtained analytically from the budget constraint:

$$k_t^{\text{endo}}(a', z_j) = \frac{c_t^{\text{endo}}(a', z_j) + a' - w_t z_j}{1 + r_t}$$

The decision rules $a'_t(k, z)$ and $c_t(k, z)$ are interpolated onto the dense asset grid and histogram grid, enforcing the borrowing constraint $a'_t \ge \underline{k}$ via lower boundary clamping.

### 1.3 Forward Young (2010) Density Propagation

Starting from the predetermined initial distribution $\mu_0 = \mu_{\text{init}}^*$, the cross-sectional distribution is advanced forward period by period for $t = 0, \dots, T-1$ via time-varying Young (2010) lottery operators:

$$\mu_{t+1} = \mathcal{T}_t^* \mu_t = \text{continuous\_push\_distribution}(\mu_t, a'_t, \mathbf{k}, \mathbf{P}_z, \mathbf{z})$$

For each state $(k_i, z_j)$, the continuous choice $a'_t(k_i, z_j)$ is bracketed between adjacent nodes $k_l \le a'_t \le k_{l+1}$ and assigned linear lottery weights:

$$\omega_l = \frac{k_{l+1} - a'_t}{k_{l+1} - k_l}, \quad \omega_{l+1} = 1 - \omega_l$$

This guarantees:
- **Mass conservation**: $\sum_{i, m} \mu_t(k_i, z_m) = 1.0 \pm 10^{-12}$ for all $t \in [0, T]$.
- **Zero numerical diffusion**: Capital supply $K_t^s = \sum_{i, m} k_i \mu_t(k_i, z_m)$ tracks the exact aggregation of individual asset holdings.

### 1.4 Sequence-Space Broyden Market Clearing

The equilibrium interest rate path $\mathbf{r}^* \in \mathbb{R}^T$ solves the stacked non-linear system:

$$\mathbf{H}(\mathbf{r}) = \begin{bmatrix} K_0^s(\mathbf{r}) - K_0^d(r_0; Z_0) \\ \vdots \\ K_{T-1}^s(\mathbf{r}) - K_{T-1}^d(r_{T-1}; Z_{T-1}) \end{bmatrix} = \mathbf{0}$$

To solve $\mathbf{H}(\mathbf{r}) = \mathbf{0}$ efficiently without repeatedly evaluating expensive numerical Jacobians, `puremacro` deploys **Broyden's Quasi-Newton method with Sherman-Morrison rank-1 updates**:

1. **Analytical Initialization**: The initial approximate inverse Jacobian $\mathbf{B}_0 \approx \mathbf{J}^{-1}$ is initialized using the analytical derivative of firm capital demand:
   $$J_{t, t}^{\text{init}} \approx -\frac{\partial K_t^d}{\partial r_t} = \frac{1}{1 - \alpha} \frac{K_t^d}{r_t + \delta} > 0, \quad \mathbf{B}_0 = \text{diag}\left( \frac{1}{J_{t, t}^{\text{init}}} \right)$$
2. **Quasi-Newton Step**: At iteration $k$, the proposed rate update is:
   $$\Delta \mathbf{r}^{(k)} = - \mathbf{B}_k \mathbf{H}(\mathbf{r}^{(k)})$$
3. **Monotone Backtracking Line Search**: A step size $\lambda \in (0, 1]$ is selected to ensure monotone contraction of the sup-norm:
   $$\|\mathbf{H}(\mathbf{r}^{(k)} + \lambda \Delta \mathbf{r}^{(k)})\|_\infty < \|\mathbf{H}(\mathbf{r}^{(k)})\|_\infty$$
4. **Sherman-Morrison Update**: Letting $\Delta \mathbf{H} = \mathbf{H}^{(k+1)} - \mathbf{H}^{(k)}$ and $\Delta \mathbf{r} = \mathbf{r}^{(k+1)} - \mathbf{r}^{(k)}$:
   $$\mathbf{B}_{k+1} = \mathbf{B}_k + \frac{(\Delta \mathbf{r} - \mathbf{B}_k \Delta \mathbf{H}) (\Delta \mathbf{r}^\top \mathbf{B}_k)}{\Delta \mathbf{r}^\top \mathbf{B}_k \Delta \mathbf{H}}$$
5. **Convergence**: The algorithm terminates when $\|\mathbf{H}(\mathbf{r})\|_\infty < \text{tol}$ (typically $10^{-4}$).

---

## 2. Methodological & Solver Options

| Feature | Broyden Quasi-Newton (`solver="broyden"`) | Fixed-Point Shooting (`solver="shooting"`) |
|---|---|---|
| **Convergence Rate** | Superlinear (typically 4–8 iterations) | Linear (20–60 iterations) |
| **Step Mechanism** | Full sequence-space Sherman-Morrison rank-1 update | Damped relaxation: $\mathbf{r}^{(n+1)} = (1 - \omega)\mathbf{r}^{(n)} + \omega \mathbf{r}^{\text{implied}}$ |
| **Line Search** | Monotone backtracking Armijo search | Fixed damping parameter $\omega \in (0, 1]$ |
| **Robustness** | Exceptional for large persistent shocks | Guaranteed contractive for small shocks |
| **Execution Budget** | $< 0.5$ seconds for $T = 150$ | $1.0$–$2.5$ seconds for $T = 150$ |

---

## 3. Canonical Calibration & Shock Specifications

Transition dynamics support three primary classes of aggregate MIT innovations:

1. **Total Factor Productivity Shock (`shock_var="z"`)**:
   $$Z_t = 1.0 + \Delta Z \cdot \rho_Z^t$$
   Transitory ($\rho_Z < 1$) or permanent ($\rho_Z = 1.0$) technological shifts.
2. **Monetary / Interest Rate Wedge Shock (`shock_var="r"`)**:
   $$r_t^{\text{eff}} = r_t + \Delta r \cdot \rho_r^t$$
   Simulates central bank tightening or borrowing spread shocks.
3. **Discount Factor / Patience Shock (`shock_var="beta"`)**:
   $$\beta_t = \beta_{\text{ss}} + \Delta \beta \cdot \rho_\beta^t$$
   Captures flight-to-safety episodes and heightened precautionary thrift.

---

## 4. Runnable Worked Examples

The following script computes the exact non-linear general equilibrium transition path following an unexpected $+5\%$ TFP innovation in an Aiyagari economy:

```python
import numpy as np
from puremacro.vfi import (
    solve_aiyagari_continuous,
    solve_continuous_transition,
    continuous_mit_shock,
    TransitionShock,
)
from puremacro.vfi.aggregate import lorenz_and_gini

# 1. Compute Initial Stationary General Equilibrium
init_ss = solve_aiyagari_continuous(
    beta=0.96,
    gamma=2.0,
    alpha=0.36,
    delta=0.08,
    N_k=100,
    n_z=3,
    max_evals=20,
)

# 2. Simulate Non-Linear MIT Productivity Shock (+5% TFP with persistence 0.75)
T_sim = 40
res_trans = continuous_mit_shock(
    steady_state=init_ss,
    shock_type="tfp",
    shock_size=0.05,
    persistence=0.75,
    horizon=T_sim,
    solver="broyden",
    max_iter=30,
)

assert res_trans.converged
assert len(res_trans.r_path) == T_sim
assert res_trans.mass_conservation_error < 1e-12

# 3. Dynamic Inequality Metrics along Transition Path
k_grid = init_ss.distribution.asset_grid
mu_0 = np.sum(res_trans.distributions[0], axis=1)
mu_T = np.sum(res_trans.distributions[-1], axis=1)

_, _, gini_0 = lorenz_and_gini(k_grid, mu_0)
_, _, gini_T = lorenz_and_gini(k_grid, mu_T)

summary_df = res_trans.summary()
print(f"Transition converged in {res_trans.iterations} iters, max residual: {res_trans.max_residual:.2e}")
print(f"Wealth Gini: t=0 -> {gini_0:.4f}, t={T_sim} -> {gini_T:.4f}")
```

---

## 5. Full API Specification

```text
TransitionShock(
    path: np.ndarray,
    var: str = "z",
)

solve_continuous_transition(
    initial_steady_state: AiyagariContinuousEquilibrium | dict[str, Any],
    terminal_steady_state: AiyagariContinuousEquilibrium | dict[str, Any] | None = None,
    shock_path: np.ndarray | Sequence[float] | None = None,
    shock_var: str = "z",
    horizon: int = 150,
    solver: str = "broyden",
    damping: float = 0.3,
    tol: float = 1e-4,
    max_iter: int = 100,
    backtracking: bool = True,
    backend: str = "numpy",
    r_init_path: np.ndarray | Sequence[float] | None = None,
    **kwargs: Any,
) -> ContinuousTransitionResult

continuous_mit_shock(
    steady_state: AiyagariContinuousEquilibrium | dict[str, Any],
    shock_type: str = "tfp",
    shock_size: float = 0.05,
    persistence: float = 0.8,
    horizon: int = 150,
    solver: str = "broyden",
    **kwargs: Any,
) -> ContinuousTransitionResult
```

#### Parameters:
- `initial_steady_state`: Pre-solved stationary equilibrium container `AiyagariContinuousEquilibrium` or configuration dictionary.
- `terminal_steady_state`: Terminal stationary equilibrium. For transitory shocks, defaults automatically to `initial_steady_state`. For permanent shocks, the terminal steady state is solved at terminal shock values.
- `shock_path`: Exogenous shock trajectory of length $T$.
- `shock_var` / `shock_type`: Target variable: `'tfp'` / `'z'`, `'rate'` / `'r'`, or `'beta'` / `'discount'`.
- `horizon`: Number of simulation periods $T$ (default $150$).
- `solver`: Equilibrium algorithm: `'broyden'` (Quasi-Newton) or `'shooting'` (damped fixed point).
- `damping`: Step damping parameter for shooting or line search fallback.
- `tol`: Convergence tolerance on market clearing $\|K^s - K^d\|_\infty$ (default $10^{-4}$).

---

## 6. Result Interface & Dynamic Inequality Metrics

`ContinuousTransitionResult` encapsulates the complete dynamic transition trajectory and inequality metrics:

### Dataclass Attributes
- `r_path`: Real interest rate trajectory $r_0, \dots, r_{T-1}$.
- `w_path`: Real wage trajectory $w_0, \dots, w_{T-1}$.
- `K_s_path`: Aggregate capital supply sequence $K_t^s = \int k \, d\mu_t$.
- `K_d_path`: Firm capital demand sequence $K_t^d(r_t)$.
- `C_path`: Aggregate consumption sequence $C_t = \int c_t \, d\mu_t$.
- `distributions`: List of length $T+1$ containing joint wealth distributions $\mu_0, \mu_1, \dots, \mu_T$.
- `residuals`: Excess capital demand trajectory $H_t = K_t^s - K_t^d$.
- `max_residual`: Maximum market clearing residual $\|H\|_\infty$.
- `converged`: Boolean convergence flag.
- `mass_conservation_error`: Maximum absolute probability mass deviation across all $T+1$ dates.

### Publication & Presentation Export Methods
- `res.summary() -> pd.DataFrame`: High-level summary of convergence diagnostics, execution time, and initial/terminal capital.
- `res.to_frame() -> pd.DataFrame`: Full sequence-space time-series DataFrame containing $r_t, w_t, K_t^s, K_t^d, C_t$, and $H_t$.
- `res.to_markdown(digits=4) -> str`: Formatted Markdown table.
- `res.to_latex(digits=4) -> str`: Publication-grade LaTeX tabular environment.
- `res.to_typst(digits=4) -> str`: Modern Typst table for technical manuscripts.
- `res.plot(figsize=(14, 8)) -> matplotlib.figure.Figure`: Multi-panel figure displaying interest rate paths, wage dynamics, market clearing, aggregate consumption, and the time-evolution of the continuous wealth distribution $\mu_t(k)$.

---

## References

- Aiyagari, S. R. (1994). "Uninsured Idiosyncratic Risk and Aggregate Saving." *The Quarterly Journal of Economics*, 109(3), 659–684.
- Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models." *Econometrica*, 89(6), 3115–3148.
- Young, E. R. (2010). "Solving the incomplete markets model with aggregate uncertainty using the Krusell-Smith algorithm and a non-stochastic simulations." *Journal of Economic Dynamics and Control*, 34(1), 36–41.
