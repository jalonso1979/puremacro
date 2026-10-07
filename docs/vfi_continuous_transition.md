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
3. **Monotone Backtracking Line Search**: Starting from $\lambda = 1$, the step is halved (at most 12 times) until the Euclidean norm of the residual falls enough:
   $$\|\mathbf{H}(\mathbf{r}^{(k)} + \lambda \Delta \mathbf{r}^{(k)})\|_2 \le (1 - 10^{-4} \lambda)\, \|\mathbf{H}(\mathbf{r}^{(k)})\|_2$$
   If no step passes, $\mathbf{B}_k$ is reset to $\mathbf{B}_0$ and the damped step $\mathbf{r}^{(k)} - \omega\, \mathbf{B}_0 \mathbf{H}(\mathbf{r}^{(k)})$ is taken, with $\omega$ = `damping`. That fallback is the only place where Broyden uses `damping`. With `backtracking=False` every step is the fallback, and the rank-one updates are discarded. The line search never failed in any run of notebook 52: with the default `backtracking=True`, `damping` of 0.2, 0.4 and 0.8 gave bit-identical Broyden paths for the shocks $-8\%$, $-5\%$, $+0.1\%$, $+5\%$ and $+8\%$ at persistence 0.5, 0.8 and 0.92 ($T = 40$). With `backtracking=False` the same $+5\%$ shock took 31, 14 and 17 iterations.
4. **Sherman-Morrison Update**: Letting $\Delta \mathbf{H} = \mathbf{H}^{(k+1)} - \mathbf{H}^{(k)}$ and $\Delta \mathbf{r} = \mathbf{r}^{(k+1)} - \mathbf{r}^{(k)}$:
   $$\mathbf{B}_{k+1} = \mathbf{B}_k + \frac{(\Delta \mathbf{r} - \mathbf{B}_k \Delta \mathbf{H}) (\Delta \mathbf{r}^\top \mathbf{B}_k)}{\Delta \mathbf{r}^\top \mathbf{B}_k \Delta \mathbf{H}}$$
5. **Convergence**: The algorithm terminates when $\|\mathbf{H}(\mathbf{r})\|_\infty < \text{tol}$ (typically $10^{-4}$).

---

## 2. Methodological & Solver Options

| Feature | Broyden Quasi-Newton (`solver="broyden"`) | Fixed-Point Shooting (`solver="shooting"`) |
|---|---|---|
| **Convergence Rate** | Superlinear (typically 4–8 iterations) | Linear (20–60 iterations) |
| **Step Mechanism** | Full sequence-space Sherman-Morrison rank-1 update | Damped relaxation: $\mathbf{r}^{(n+1)} = (1 - \omega)\mathbf{r}^{(n)} + \omega \mathbf{r}^{\text{implied}}$ |
| **Line Search** | Monotone backtracking Armijo search on $\|\mathbf{H}\|_2$; `damping` scales only the fallback step (section 1.4) | Fixed damping parameter $\omega$ = `damping` $\in (0, 1]$ at every step |
| **Robustness** | Exceptional for large persistent shocks | Guaranteed contractive for small shocks |
| **Execution Budget** | $< 0.5$ seconds for $T = 150$ | $1.0$–$2.5$ seconds for $T = 150$ |

---

## 3. Canonical Calibration & Shock Specifications

Transition dynamics support three classes of aggregate MIT innovations. Below, $s_t$ is the entry of `shock_path` at date $t$; `continuous_mit_shock` builds $s_t = \Delta \cdot \rho^t$ from `shock_size` $\Delta$ and `persistence` $\rho$ ($s_t = \Delta$ for every $t$ when $\rho = 1$). For `continuous_mit_shock`, $\Delta$ is always a deviation, never a level: it hands `solve_continuous_transition` the levels $Z_t = 1 + s_t$ or $\beta_t = \beta(1 + s_t)$ whenever the rule below would otherwise read the path as levels. Through 4.3.0 it always passed $s_t$, so `shock_size=0.6` gave $Z_0 = 0.6$, a 40% fall, instead of $1.6$. A TFP shock must be above $-1$.

1. **Total Factor Productivity Shock (`shock_var="z"`)**:
   $$Z_t = 1 + s_t$$
   when every $s_t \le 0.5$; if any entry exceeds $0.5$ the path is read as levels, $Z_t = s_t$. A path shorter than $T$ is held at its last value. Transitory ($\rho < 1$) or permanent ($\rho = 1$) technological shifts; a permanent one solves the terminal steady state at $Z_{T-1}$.
2. **Monetary / Interest Rate Wedge Shock (`shock_var="r"`)**:
   $$r_t^{\text{hh}} = r_t + s_t$$
   Households earn $r_t + s_t$ in the Euler equation and the budget constraint, while firms pay $r_t$. A path shorter than $T$ is padded with zeros. Simulates central bank tightening or borrowing spread shocks. Only transitory wedges are supported (see the terminal-condition limitation below).
3. **Discount Factor / Patience Shock (`shock_var="beta"`)**:
   $$\beta_t = \beta\,(1 + s_t)$$
   multiplicative, with $\beta$ the initial steady state's discount factor, when every $s_t \le 0.5$; if any entry exceeds $0.5$ the path is read as levels, $\beta_t = s_t$. $\beta_t$ discounts date $t+1$ in the date-$t$ Euler equation. A path shorter than $T$ is padded with $\beta$. For example, `shock_size=0.01` with `persistence=0.8` gives $\beta_0 = 1.01\,\beta$ and $\beta_1 = 1.008\,\beta$. A permanent shock solves the terminal steady state at $\beta_{T-1}$. Captures flight-to-safety episodes and heightened precautionary thrift.

### 3.1 Terminal condition and its limitation

The path is solved backward from a terminal steady state at date $T$: the continuation consumption policy $c_T$ and the rate $r_T$.

**`continuous_mit_shock`** chooses it from `persistence` alone:

- `persistence < 1` (transitory): always the initial steady state, and the shock is set to zero from date $T$ on. When the share of the impact still present at $T-1$, $\rho^{T-1}$, exceeds `truncation_tol` (default $10^{-3}$), a `RuntimeWarning` gives the smallest horizon that brings it below the tolerance. `metadata["mit_shock"]` reports the truncation either way (section 6).
- `persistence == 1` (permanent, which must be asked for explicitly): for TFP and discount-factor shocks, a steady state solved internally at $Z = 1 + \Delta$ or $\beta(1 + \Delta)$ with the controls of section 5. For a rate wedge, see below.
- A `terminal_steady_state` passed as a keyword is used as given, for any persistence.

Through 4.3.0 a transitory shock followed the `solve_continuous_transition` rule below. Once $Z_{T-1}$ differed from 1 by more than about $10^{-5}$, the shock was silently solved as a permanent one at $Z_{T-1}$. With $T = 40$ and $\Delta = 0.05$ the model changed at $\rho \approx 0.806$. At $\rho = 0.92$ the economy of section 4 converged to a steady state with 0.19% higher TFP and 0.30% more capital, and its capital path differed from the transitory solution by up to 0.026.

**`solve_continuous_transition`**, called directly with a path, uses:

- `terminal_steady_state`, when you pass one;
- otherwise, when $Z_{T-1}$ differs from 1 or $\beta_{T-1}$ from $\beta$ by more than `numpy.isclose(..., atol=1e-6)` allows (about $10^{-5}$), a steady state solved internally at $Z_{T-1}$ and $\beta_{T-1}$ with the controls of section 5;
- otherwise the initial steady state.

That rule cannot tell a slowly decaying transitory path from a permanent one. For a transitory path that has not died out by $T-1$, pass `terminal_steady_state=` the initial steady state (as `continuous_mit_shock` does), or lengthen the horizon.

**Choosing $T$.** The truncation warning looks only at the exogenous shock, and capital returns more slowly than the shock. In notebook 52 ($\rho = 0.8$, $T = 40$) the shock left at $T-1$ is $1.7 \times 10^{-4}$ of the impact, yet $K_{T-1} - K^*$ is still $0.040$. `metadata["mit_shock"]["capital_gap_last"]` reports that gap; solving again with a longer horizon measures the truncation error.

**A permanent interest-rate wedge is not supported.** With `shock_var="rate"` and a wedge that is still nonzero at $T-1$ (for example `persistence=1`), the terminal condition stays the initial steady state: $c_T$ is the policy at $r^*$ and $r_T = r^*$ carries no wedge, although households face $r + s$ forever. The path is then solved against an inconsistent terminal condition, and `converged=True` only says that the capital market clears at every date on that path. It does not say that the path has reached the new steady state. `solve_continuous_transition` issues no warning; `continuous_mit_shock(..., shock_type="rate", persistence=1.0)` issues a `RuntimeWarning` and sets `metadata["mit_shock"]["truncated"]`. In the section 4 economy ($N_k = 100$, $n_z = 3$), a permanent $+50$ bp wedge reports `converged=True` with $r_{T-1} = 0.0332$ at $T = 40$ and $0.0325$ at $T = 80$, against the terminal $r_T = r^* = 0.0394$: the path is still moving at $T$. `solve_aiyagari_continuous` has no wedge argument, so no terminal steady state with the wedge can be passed either. Use rate wedges that have died out by $T-1$, and check that `r_path[-1]` is back at the terminal rate.

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
    *,
    truncation_tol: float = 1e-3,
    **kwargs: Any,
) -> ContinuousTransitionResult
```

#### Parameters:
- `initial_steady_state`: Pre-solved stationary equilibrium container `AiyagariContinuousEquilibrium`, or a configuration dict that is passed to `solve_aiyagari_continuous` to solve a new steady state. The dict may contain **only** `solve_aiyagari_continuous` keywords (`beta`, `gamma`, `alpha`, `delta`, `rho_z`, `sigma_z`, `n_z`, `a_max`, `N_k`, `n_a`, `P_z`, `z_grid`, ...). Any other key, such as the grids, distribution or prices of a steady state solved elsewhere, raises `TypeError`. To start from a steady state you already have, pass the `AiyagariContinuousEquilibrium` object itself.
- `terminal_steady_state`: Terminal stationary equilibrium (object or dict, as above). When omitted, `solve_continuous_transition` uses the initial steady state for paths that have died out by $T-1$, and solves a steady state internally at $Z_{T-1}$ and $\beta_{T-1}$ otherwise, which includes a transitory path that has not died out yet. `continuous_mit_shock` passes the initial steady state for every transitory shock (section 3.1). A permanent rate wedge is not supported (section 3.1).
- `shock_path`: Exogenous shock trajectory of length $T$, interpreted per `shock_var` as in section 3.
- `shock_var` / `shock_type`: Target variable: `'tfp'` / `'z'` / `'productivity'`, `'rate'` / `'r'` / `'monetary'`, or `'beta'` / `'discount'`.
- `horizon`: Number of simulation periods $T$ (default $150$).
- `solver`: Equilibrium algorithm: `'broyden'` (Quasi-Newton) or `'shooting'` (damped fixed point).
- `damping`: The relaxation weight $\omega$ of `solver='shooting'`, used at every step. With `solver='broyden'` it scales only the fallback step taken when the line search fails (and every step when `backtracking=False`); see section 1.4. With the default `backtracking=True` it had no effect in any run of notebook 52.
- `tol`: Convergence tolerance on market clearing $\|K^s - K^d\|_\infty$ (default $10^{-4}$).

#### `continuous_mit_shock` parameters:
- `steady_state`: The initial steady state (object or configuration dict, as `initial_steady_state` above).
- `shock_type`: `'tfp'`, `'rate'` or `'beta'`, or any alias of `shock_var`.
- `shock_size`: The impact $\Delta = s_0$, always a deviation: $+0.05$ is $+5\%$ TFP, $+0.01$ is $+100$ bp on the rate, and for `'beta'` it gives $\beta_0 = \beta(1 + \Delta)$. A TFP shock must be above $-1$.
- `persistence`: $\rho \in [0, 1]$. $\rho = 1$ is a permanent shock; anything below 1, including $0.999$, is transitory and keeps the initial steady state as the terminal condition (section 3.1).
- `truncation_tol`: The largest share $\rho^{T-1}$ of the impact that a transitory shock may still have at $T-1$ without a `RuntimeWarning` (default $10^{-3}$).
- `**kwargs`: Passed to `solve_continuous_transition` (`damping`, `tol`, `max_iter`, `backtracking`, `backend`, `r_init_path`, `terminal_steady_state` and the keywords below).

#### Structural parameters and `**kwargs`:
- The structural parameters `beta`, `gamma`, `alpha` and `delta` are read from the initial steady state's `metadata["params"]`, which `solve_aiyagari_continuous` records. Passing one of them as a keyword with a different value raises `ValueError`. The keywords are used only when the steady state records none (one built with `continuous_stationary_equilibrium` or by hand); a missing one then falls back to the `solve_aiyagari_continuous` default with a `UserWarning`.
- `egm_tol`, `egm_max_iter`, `xtol`, `tol_ge`, `max_evals`, `dist_options`: controls of the terminal steady state solved internally, with the same meaning as in `solve_aiyagari_continuous`. Unless given, the first five are taken from the initial steady state's metadata, then from that function's defaults ($10^{-8}$, $10\,000$, $10^{-8}$, $10^{-4}$, $100$); `dist_options` is not read from the metadata and defaults to `None`.
- `r_min`, `r_max`: bounds on the trial interest-rate path (defaults $10^{-4}$ and $\max(0.15,\ 1/\beta - 1 + 0.10)$).
- Any other keyword is ignored with a `FutureWarning`; it will raise `TypeError` in a future release.
- `continuous_mit_shock(..., **kwargs)` passes these keywords on to `solve_continuous_transition`.

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
- `metadata`: Solver diagnostics: `params`, `relaxation_converged`, `terminal_steady_state` (the diagnostics of a terminal steady state solved internally, or `None`) and, unless a zero shock returned the steady state at once, `Z_path` and `beta_path`. `continuous_mit_shock` adds `metadata["mit_shock"]`, with `shock_type`, `shock_size`, `persistence`, `permanent`, `terminal_condition` (`'initial_steady_state'`, `'solved_steady_state'` or `'user'`), `shock_at_last_date` ($s_{T-1}$), `remaining_share` ($\rho^{T-1}$), `truncation_tol`, `truncated`, `horizon_needed` (the smallest $T$ with $\rho^{T-1} \le$ `truncation_tol`; `None` for a permanent shock) and `capital_gap_last` ($K_{T-1} - K^*$ for a transitory shock).

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
