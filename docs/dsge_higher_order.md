> 🇬🇧 English · 🇪🇸 [Español](es/dsge_higher_order.md)

# Higher-Order Perturbation, Constraints & Nonlinear DSGE Methods

Puremacro 2.8.0 introduces **Tier 2: Higher Order & Constraints**, expanding the DSGE engine beyond first- and second-order local approximations into global non-linear transitions, state pruning, occasionally binding inequality constraints, sequential Monte Carlo sampling, and Ramsey optimal commitment policy.

Every algorithm in this release is implemented in 100% pure Python under the strict **Pyodide four-package contract** (`numpy`, `scipy`, `pandas`, `matplotlib`), requiring zero C-extensions, zero external parser generators, and zero heavy symbolic algebra dependencies (no SymPy, no JAX).

---

## 1. Overview & Architecture

Linearized DSGE models (e.g. solved via Klein QZ decomposition) provide first-order approximations around a deterministic steady state. While computationally fast, linear solutions:
1. Disregard precautionary behavior and stochastic risk premia ($g_{\sigma\sigma} = 0$).
2. Suffer from certainty equivalence: macroeconomic volatility does not shift policy decision rules or endogenous steady states.
3. Cannot represent asymmetric macroeconomic bounds such as the Zero Lower Bound (ZLB) on nominal interest rates or collateral borrowing constraints.
4. Fail across permanent regime shifts where the economy transitions between fundamentally distinct initial and terminal steady states.

Tier 2 provides a unified suite of methods to overcome these limitations:

| Module | Core Capability | Key Method / Algorithm | Benchmark Reference |
| :--- | :--- | :--- | :--- |
| `puremacro.dsge.pruning` | Order-3 Perturbation & Ergodic Pruning | 3-fold Schur Sylvester solver & 3-state expansion | Andreasen, Fernández-Villaverde & Rubio-Ramírez (2018) |
| `puremacro.dsge.perfect_foresight` | Deterministic Transitions & MCP | Semismooth Newton with Fischer-Burmeister complementarity | Boucekkine (1995); Juillard (1996) |
| `puremacro.dsge.occbin` | Multi-Constraint Piecewise Linear | Dual-regime relaxation & Piecewise Kalman Filter | Guerrieri & Iacoviello (2015); Giovannini et al. (2021) |
| `puremacro.dsge.smc` | Sequential Monte Carlo & Particle Filtering | Adaptive tempering SMC & Nonlinear bootstrap particle filter | Herbst & Schorfheide (2014, 2015) |
| `puremacro.dsge.ramsey` | Ramsey Optimal Policy (linear-quadratic) & BGP | LQ commitment from linearized constraints and the objective Hessian, Klein QZ; symbolic FOCs for display | Dennis (2007); Clarida, Galí & Gertler (1999) |

---

## 2. 3rd-Order Perturbation & Andreasen Pruning

### 2.1 Theoretical Foundations & The Curse of Dimensionality

A dynamic stochastic general equilibrium model is characterized by the system of expectational equations:

$$\mathbb{E}_t \left[ f(y_{t+1}, y_t, y_{t-1}, u_t) \right] = 0$$

where $y_t$ is an $n_y$-dimensional vector of endogenous variables and $u_t \sim \mathcal{N}(0, \Sigma)$ is an $n_u$-dimensional vector of exogenous innovations scaled by perturbation parameter $\sigma$. 

Partitioning $y_t$ into predetermined states $x_t$ and non-predetermined controls $w_t$, the exact policy function takes the form:

$$y_t = g(x_{t-1}, u_t, \sigma)$$

Expanding $g$ up to third order via Taylor perturbation yields:

$$g(x, u, \sigma) \approx g_{ss} + g_x \hat{x} + g_u u + \frac{1}{2} g_{xx} (\hat{x} \otimes \hat{x}) + g_{xu} (\hat{x} \otimes u) + \frac{1}{2} g_{uu} (u \otimes u) + \frac{1}{2} g_{\sigma\sigma} \sigma^2 + \frac{1}{6} g_{xxx} (\hat{x} \otimes \hat{x} \otimes \hat{x}) + \dots + \frac{1}{6} g_{\sigma\sigma\sigma} \sigma^3$$

Evaluating the dynamic third tensor derivative:

$$\mathcal{T}_f = \frac{\partial^3 f}{\partial z_j \, \partial z_k \, \partial z_l}$$

over $z = (y_{t+1}, y_t, y_{t-1}, u_t)$ requires evaluating $O((3n_y + n_u)^3)$ elements. Puremacro's symbolic differentiation engine (`puremacro.dsge._symbolic`) avoids dense cubic tensor allocation by:
- Enforcing exact 6-fold permutation symmetry ($j \le k \le l$).
- Common Subexpression Elimination (CSE) across partial derivatives.
- Direct vectorization into compiled Python callables.

### 2.2 Generalized Schur Sylvester Solver for 3rd-Order Tensors

The third-order state curvature tensor $g_{xxx}$ satisfies the generalized matrix Sylvester equation:

$$\hat{A} g_{xxx} + A_+ g_{xxx} (h_x \otimes h_x \otimes h_x) = - K_{xxx}$$

where $\hat{A} = A_0 + A_+ g_x P_s$ and $h_x = P_s g_x$ is the stable state transition companion matrix. 

Assembling the dense Kronecker matrix $(h_x \otimes h_x \otimes h_x)$ of dimension $n_x^3 \times n_x^3$ would require hundreds of gigabytes of RAM for standard medium-scale DSGE models (such as Smets-Wouters 2007 with $n_x = 14$, where $14^3 = 2,744$, creating an $11 \times 10^7$ element system).

Puremacro implements the decoupled **3-Fold Generalized Schur Sylvester Solver** (`puremacro.dsge._sylvester`):
1. Computes the complex Schur decomposition $h_x = U_h T_h U_h^H$ with upper-triangular factor $T_h$.
2. Transforms $K_{xxx}$ into Schur coordinates: $\tilde{K} = K_{xxx} (U_h \otimes U_h \otimes U_h)$.
3. Solves vector-by-vector via triangular back-substitution with LU caching:

$$\left( \hat{A} + T_h(i, i) T_h(j, j) T_h(k, k) A_+ \right) \tilde{g}_{i,j,k} = - \tilde{K}_{i,j,k} - \text{back-substitution terms}$$

4. Rotates the solution back: $g_{xxx} = \tilde{g}_{xxx} (U_h^H \otimes U_h^H \otimes U_h^H)$.

On the benchmark Smets-Wouters (2007) model, this solver executes in **0.034 seconds** with **< 1 MB memory overhead**, completely bypassing dense tensor assembly.

### 2.3 The Andreasen et al. (2018) Pruning Algorithm

Unpruned polynomial approximations of order $k \ge 2$ generate spurious explosive paths in simulations: large shocks propel states into nonlinear domains where quadratic and cubic terms overpower stable linear feedback roots.

Following **Andreasen, Fernández-Villaverde, and Rubio-Ramírez (2018, *Review of Economic Studies*)**, puremacro decomposes the state deviation $\hat{x}_t = x_t - x_{ss}$ into first-, second-, and third-order components:

$$\hat{x}_t = x_t^{(1)} + x_t^{(2)} + x_t^{(3)}$$

whose recursive state space evolves without higher-order feedback loops:

$$x_t^{(1)} = h_x x_{t-1}^{(1)} + h_u u_t$$

$$x_t^{(2)} = h_x x_{t-1}^{(2)} + \frac{1}{2} h_{xx} (x_{t-1}^{(1)} \otimes x_{t-1}^{(1)}) + h_{xu} (x_{t-1}^{(1)} \otimes u_t) + \frac{1}{2} h_{uu} (u_t \otimes u_t) + \frac{1}{2} h_{\sigma\sigma} \sigma^2$$

$$\begin{aligned}x_t^{(3)} = {} & h_x x_{t-1}^{(3)} + h_{xx} (x_{t-1}^{(1)} \otimes x_{t-1}^{(2)}) + h_{xu} (x_{t-1}^{(2)} \otimes u_t) + \frac{1}{6} h_{xxx} (x_{t-1}^{(1)} \otimes x_{t-1}^{(1)} \otimes x_{t-1}^{(1)}) \\ & + \frac{1}{2} h_{xxu} (x_{t-1}^{(1)} \otimes x_{t-1}^{(1)} \otimes u_t) + \frac{1}{2} h_{xuu} (x_{t-1}^{(1)} \otimes u_t \otimes u_t) + \frac{1}{6} h_{uuu} (u_t \otimes u_t \otimes u_t) \\ & + \frac{1}{2} h_{x\sigma\sigma} x_{t-1}^{(1)} \sigma^2 + \frac{1}{2} h_{u\sigma\sigma} u_t \sigma^2\end{aligned}$$

Control variables $y_t$ are evaluated analogously:

$$\hat{y}_t = y_t^{(1)} + y_t^{(2)} + y_t^{(3)}, \qquad y_t^{(1)} = g_x x_{t-1}^{(1)} + g_u u_t,$$

where $y_t^{(2)}$ and $y_t^{(3)}$ are the right-hand sides of the $x_t^{(2)}$ and $x_t^{(3)}$ recursions with every $h$ replaced by the matching $g$. Both depend on the lagged state components and the current shock.

Two properties follow:
- **Stability**: the pruned system is stable whenever the first-order transition $h_x$ is, so simulations do not explode.
- **Exact moments**: the augmented state $z_t = [x_t^{(1)};\, x_t^{(2)};\, x_t^{(1)} \otimes x_t^{(1)};\, x_t^{(3)};\, x_t^{(1)} \otimes x_t^{(2)};\, x_t^{(1)} \otimes x_t^{(1)} \otimes x_t^{(1)}]$ follows a linear law of motion driven by innovations that are uncorrelated over time. `theoretical_moments()` therefore returns the exact mean, covariance and autocorrelations of every variable under Gaussian shocks. On the five models in `tests/fixtures/dynare_live`, its mean and covariance equal Dynare 8's "theoretical moments based on pruned state space" to machine precision.

  The autocorrelations differ from Dynare's. Dynare's order-3 recursion drops the correlation between the innovation $x_{t-1}^{(1)} \otimes u_t \otimes u_t$ and past shocks. Long simulations, Dynare's own included, agree with puremacro. For `y` in `correlated_cubic.mod` the lag-1 autocorrelation is 0.475, against 0.330 in Dynare's table. The augmented state has $3n + 2n^2 + n^3$ entries for $n$ predetermined states. The default `max_state_dim=3000` admits up to 13 states, and `stoch_simul(order=3)` skips the table with a warning above that.
- **Shape moments**: skewness and kurtosis have no closed form here. `theoretical_moments()` reports them as NaN. `ergodic_moments()` keeps the exact mean and variance and estimates both from a pruned simulation (`periods=100_000` by default). Its Monte Carlo error is large for very persistent variables.
- **Second order**: `PrunedDSGESolution.theoretical_moments(pruning=True)` (and `stoch_simul(order=2, pruning=True)`) gives the exact moments of the pruned second-order solution, which reproduce Dynare 8's `stoch_simul(order=2, pruning)` to 7e-14, autocorrelations included. The default follows Dynare's `stoch_simul(order=2)` without `pruning`: the risk-corrected mean with first-order second moments.

### 2.4 Ergodic Mean, Risky Steady State and the Risk Term

A higher-order solution gives two different answers to "where does the economy sit under risk?", and they need not have the same sign. Both solution classes (`PrunedDSGESolution` at order 2, `Order3PrunedSolution` at order 3) return them as deviations from the deterministic steady state, in the units of the model's variables.

- **Ergodic mean**, `sol.ergodic_mean()`: the unconditional mean of the pruned process, i.e. the long-run average of a simulation. `sol.stochastic_steady_state()` is an alias kept for backward compatibility; despite its name it returns this mean. With $\Omega = h_x \Omega h_x' + h_u \sigma^2 \Sigma_u h_u'$ the first-order state covariance, and $\mathbb{E}[x_{t-1} \otimes u_t] = 0$ in Dynare's timing,

$$\mathbb{E}[\hat{x}] = (I - h_x)^{-1} \Big[ \tfrac{1}{2} h_{\sigma\sigma} \sigma^2 + \tfrac{1}{2} h_{xx} \operatorname{vec}(\Omega) + \tfrac{1}{2} h_{uu} \operatorname{vec}(\sigma^2 \Sigma_u) \Big], \qquad \mathbb{E}[\hat{y}] = g_x \mathbb{E}[\hat{x}] + \tfrac{1}{2} g_{\sigma\sigma} \sigma^2 + \tfrac{1}{2} g_{xx} \operatorname{vec}(\Omega) + \tfrac{1}{2} g_{uu} \operatorname{vec}(\sigma^2 \Sigma_u).$$

  With Gaussian shocks every third-order forcing term has mean zero, so this is also the exact order-3 mean. It equals the `Mean` column of `theoretical_moments()` and Dynare 8's `oo_.mean` with `pruning`.

- **Risky steady state**, `sol.risky_steady_state()`: the point where the economy settles when agents expect shocks but none is realized (Coeurdacier, Rey and Winant 2011). In the pruned system the first-order state stays at zero without shocks, so

$$\hat{x}_{rss} = (I - h_x)^{-1} \tfrac{1}{2} h_{\sigma\sigma} \sigma^2, \qquad \hat{y}_{rss} = g_x \hat{x}_{rss} + \tfrac{1}{2} g_{\sigma\sigma} \sigma^2,$$

  the limit of `sol.simulate(shocks=np.zeros(...), burn=0)`. `risky_steady_state(pruned=False)` solves instead for the fixed point of the unpruned approximated policy by Newton's method; the two differ by terms of order $\sigma^4$.

Only $g_{\sigma\sigma}$ (Dynare's `ghs2`) moves the risky steady state. It is the precautionary effect of anticipated risk. The other two terms of the mean are the curvature of the policy times the realized dispersion of states and shocks, a Jensen effect that is there even when $g_{\sigma\sigma} = 0$. `sol.risk_decomposition()` returns the three terms (`risk`, `state_curvature`, `shock_curvature`) and their sum (`ergodic_mean`).

In the RBC model of `tests/fixtures/dynare_live/rbc.mod` (in levels, shock s.d. 0.01), the terms for capital are, in percent of its deterministic steady state:

| Term | Capital $k$ (% of $k_{ss}$) |
| :--- | ---: |
| Risk term $\tfrac{1}{2} h_{\sigma\sigma}$ (= risky steady state) | −0.0045 |
| State curvature $\tfrac{1}{2} h_{xx} \operatorname{vec}(\Omega)$ | +0.0705 |
| Shock curvature $\tfrac{1}{2} h_{uu} \operatorname{vec}(\Sigma_u)$ | +0.0125 |
| Ergodic mean | +0.0785 |

Dynare's own `ghs2` for $k$ in this model is negative (−9.76e-5 in `tests/fixtures/dynare_live/rbc_order3.npz`), so anticipated risk lowers capital. The positive mean comes from curvature and should not be read as precautionary capital. Two closed forms make the distinction exact, and `tests/test_fix_pruning_risky_steady_state.py` checks both:

- **A claim to $x^2$.** With $x_t = \rho x_{t-1} + e_t$ and $\operatorname{Var}(e) = s^2$, the price $y_t = \beta \mathbb{E}_t y_{t+1} + x_t^2$ is exactly $y = A x^2 + B$ with $A = 1/(1 - \beta\rho^2)$ and $B = \beta A s^2 / (1 - \beta)$. The ergodic mean is $A \operatorname{Var}(x) + B$ and the risky steady state is $B$.
- **Brock and Mirman (1972), in levels.** The policy $k_t = \alpha\beta e^{a_t} k_{t-1}^\alpha$ does not depend on risk, so $g_{\sigma\sigma} = 0$ and the risky steady state is the deterministic one. Still, $\mathbb{E}[k] - k_{ss} = k_{ss} \operatorname{Var}(\log k)/2 > 0$ at second order.

```python
from pathlib import Path
from puremacro.dsge import load_mod

sol3 = load_mod(Path("tests/fixtures/dynare_live/rbc.mod").read_text(), order=3)
dec = sol3.risk_decomposition()                   # deviations from the deterministic steady state
print((100 * dec.div(sol3.steady_state[dec.index], axis=0)).loc[["k", "c", "y"]].round(4))
print(sol3.risky_steady_state()["states"]["k"])   # -0.00129: below k_ss
print(sol3.ergodic_mean()["states"]["k"])         # +0.02224: above k_ss
```

### 2.5 Generalized Impulse Response Functions (GIRF) & API

Because higher-order models violate linear superposition, impulse responses depend on the initial state $x_0$ and the shock sign/magnitude. Puremacro evaluates **Generalized Impulse Responses (GIRF)** (Koop, Pesaran & Potter 1996):

$$\text{GIRF}_t(v, x_0) = \mathbb{E} \left[ y_t \mid u_0 = v, x_0 \right] - \mathbb{E} \left[ y_t \mid u_0 = 0, x_0 \right]$$

```python
import numpy as np
from puremacro.dsge import load_mod

# Load nonlinear DSGE model
mod_text = """
var c k y a;
varexo e_a;
parameters beta alpha delta rho sigma_a;
beta = 0.99; alpha = 0.36; delta = 0.025; rho = 0.95; sigma_a = 0.01;

model;
  1/c = beta * (1/c(+1)) * (alpha * exp(a(+1)) * k^(alpha - 1) + 1 - delta);
  c + k = exp(a) * k(-1)^alpha + (1 - delta) * k(-1);
  y = exp(a) * k(-1)^alpha;
  a = rho * a(-1) + e_a;
end;

initval;
  k = 38.0; c = 2.7; y = 3.6; a = 0;
end;
shocks;
  var e_a; stderr 0.01;
end;
"""

model = load_mod(mod_text)

# Solve order-3 perturbation with Andreasen pruning
sol3 = model.solve(order=3, pruning=True)

# 1. Exact moments of the pruned state space (mean, std. dev., autocorrelations)
moments = sol3.theoretical_moments()
print(moments.moments[["Mean", "Std.Dev."]].round(4))
print("Risk adjustment of mean capital:", moments.moments.loc["k", "Mean"] - sol3.steady_state["k"])

# 2. Skewness and kurtosis have no closed form: estimated from a long pruned simulation
shape = sol3.ergodic_moments(periods=50_000, seed=0)
print(shape[["Skewness", "Kurtosis"]].round(3))

# 3. Generalized Impulse Response Functions (GIRF)
girf = sol3.girf(shock="e_a", size=2.0, horizon=40)
girf.plot(title="Order-3 GIRF to a 2-std Dev Productivity Innovation")

# 4. Pruned stochastic simulation: one finite sample of deviations from the steady state
sim = sol3.simulate(periods=20_000, seed=42)
print("Simulated s.d. of output:", sim["y"].std(), "exact:", moments.moments.loc["y", "Std.Dev."])
```

---

## 3. Deterministic Transitions & Mixed Complementarity Problems (MCP)

### 3.1 Extended Path & Terminal Steady State Shifts

Policy analysis frequently investigates permanent structural shifts, such as:
- A permanent rise in the inflation target $\pi^*$.
- Permanent carbon taxation reforms.
- Permanent productivity level shifts.

Under permanent shocks, the terminal steady state $y_{end}$ differs from the initial steady state $y_{init}$. Puremacro parses `histval; ... end;` and `endval; ... end;` blocks to solve exact non-linear transitions over horizon $T$:

$$f(y_{t+1}, y_t, y_{t-1}, u_t) = 0, \quad t = 1, \dots, T$$

with boundary conditions $y_0 = y_{init}$ and $y_{T+1} = y_{end}$.

### 3.2 Anticipated and Surprise Exogenous Paths

Puremacro differentiates between:
1. **Anticipated deterministic shocks (`varexo_det`)**: Exogenous variables announced at $t=0$ whose future path is known with certainty.
2. **Surprise shocks (`simulate_surprise_shocks`)**: Unanticipated innovations arriving sequentially at date $t$, triggering rolling replanning across the remaining horizon.

### 3.3 Semismooth Newton MCP with Fischer-Burmeister Complementarity

Macroeconomic regimes frequently involve inequality constraints, such as the Zero Lower Bound (ZLB) on nominal interest rates:

$$i_t = \max(0, i_t^*)$$

Expressed as a Mixed Complementarity Problem (MCP):

$$0 \le i_t \perp \lambda_t \ge 0 \quad \text{where } \lambda_t = i_t - i_t^*$$

To avoid combinatorial regime branching, puremacro implements a **Semismooth Newton solver** using the **Fischer-Burmeister complementarity function**:

$$\Phi(a, b) = a + b - \sqrt{a^2 + b^2} = 0 \iff a \ge 0, \, b \ge 0, \, ab = 0$$

Let $\mathbf{F}(\mathbf{Y}) = \mathbf{0}$ denote the stacked equations over horizon $t=1,\dots,T$. For each bounded variable at period $t$, the equality condition is replaced by $\Phi(y_{i,t} - \underline{y}_i, \lambda_{i,t}) = 0$.

The generalized Jacobian $\partial_B \Phi$ is semismooth:

$$\frac{\partial \Phi}{\partial a} = 1 - \frac{a}{\sqrt{a^2 + b^2}}, \quad \frac{\partial \Phi}{\partial b} = 1 - \frac{b}{\sqrt{a^2 + b^2}}$$

Puremacro assembles the stacked block-tridiagonal Jacobian directly in sparse format, solving the entire non-linear trajectory in $O(T)$ operations with superlinear local convergence.

```python
from puremacro.dsge import load_mod, solve_perfect_foresight

MOD_ZLB = """
var y pi i;
varexo eps_r;
parameters beta sigma phi_pi phi_y;
beta = 0.99; sigma = 1.0; phi_pi = 1.5; phi_y = 0.5;

model;
  y = y(+1) - (1/sigma) * (i - pi(+1));
  pi = beta * pi(+1) + 0.1 * y;
  i = max(0, phi_pi * pi + phi_y * y + eps_r);
end;

initval; y = 0; pi = 0; i = 0.02; end;
"""

model = load_mod(MOD_ZLB)

# Solve with Semismooth Newton MCP enforcing the Zero Lower Bound
res_mcp = solve_perfect_foresight(
    model,
    periods=100,
    shocks={"eps_r": [-0.05, -0.03, -0.01]},
    mcp=True,
    mcp_bounds={"i": (0.0, np.inf)},
)

print(f"Converged in {res_mcp.iterations} iterations.")
print("Binding ZLB periods:", res_mcp.binding_periods["i"])
res_mcp.plot(title="NK Deflation Experiment with Active ZLB")
```

---

## 4. Multi-Constraint OccBin & Piecewise Kalman Filter

### 4.1 Multi-Constraint Piecewise Linear Architecture

**OccBin (Guerrieri & Iacoviello 2015)** approximates models with occasionally binding constraints by defining piecewise linear regimes:
- Reference unconstrained regime: $A_0 y_t + A_+ \mathbb{E}_t y_{t+1} + A_- y_{t-1} + C_0 = 0$
- Constrained regimes ($k=1,\dots,K$): $A_0^{(k)} y_t + A_+^{(k)} \mathbb{E}_t y_{t+1} + A_-^{(k)} y_{t-1} + C_0^{(k)} = 0$

Puremacro 2.8.0 extends OccBin to **$K \ge 2$ simultaneous constraints**, supporting up to $2^K$ discrete regimes (such as joint monetary ZLB and collateral borrowing limits).

The algorithm:
1. Initializes a regime sequence guess $(r_1, \dots, r_H)$ across horizon $H$.
2. Performs backward time-varying Riccati recursions:

$$P_t = - (A_0^{(r_t)} + A_+^{(r_t)} P_{t+1})^{-1} A_-^{(r_t)}$$

3. Simulates forward state trajectories and updates shadow multiplier values.
4. Checks convergence of the binding sequence.

#### The `solve_multiconstraint_occbin` API

The multi-constraint solver **predates 3.4.0** — it shipped in 2.8.0 —
and 3.4.0 extends and hardens it rather than introducing it: the regime
splice now carries *every* row in which a constrained model differs from
the reference, non-convergence is reported instead of implied, and the
pairing rules below are enforced with `ValueError`.

```text
solve_multiconstraint_occbin(
    m_unconstrained: LinearModel,
    m_constrained_dict: Mapping[str, LinearModel] | Sequence[LinearModel],
    shock_seq: np.ndarray,                       # (n_shocks,) or (horizon, n_shocks)
    constraints: Mapping[str, OccBinConstraint]
                 | Sequence[OccBinConstraint]
                 | None = None,
    horizon: int = 40,
    max_iter: int = 50,
) -> OccBinResult
```

**One model per constraint, paired one-to-one.** `m_constrained_dict`
holds the model in which *that* constraint, and only that one, binds;
`constraints` must be a mapping with exactly the model keys, a sequence
in the model order, or a single constraint when there is a single model.
Mismatched keys, a wrong-length sequence, or one constraint offered for
several models each raise `ValueError` — the solver never guesses a
pairing. Passing `constraints=None` deduces each constraint from its
model (the pegged variable, $-c/a$ as the threshold, the operator read
off the key or the sign of the threshold).

**The regime sequence is a bitmask.** `result.regimes[t]` is an integer
whose bit $k$ is set when constraint $k$ — the $k$-th entry of
`m_constrained_dict` — binds in period $t$. With two constraints:
`0` neither, `1` the first only, `2` the second only, `3` both. That is
why regime counts are written as mask tests rather than equality tests:

```python
import numpy as np
from puremacro.dsge import build_dynare, OccBinConstraint, solve_multiconstraint_occbin

params = {
    "beta": 0.99, "sigma": 1.0, "kappa": 0.15, "phi_pi": 1.5, "phi_y": 0.25,
    "rho_r": 0.6, "rho_b": 0.5, "rho_g": 0.7, "gamma_y": 0.2, "chi": 0.1,
    "r_ss": 0.015, "b_bar": 0.02,
}
variables = ["y", "pi", "r", "b", "g"]
shocks = ["eps_g", "eps_r", "eps_b"]


def ref_eqs(lead, curr, lag, s, p):          # reference: no constraint binds
    return [
        curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,
        curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r)
                  * (p.phi_pi * curr.pi + p.phi_y * curr.y) + s.eps_r),
        curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + s.eps_b),
        curr.g - p.rho_g * lag.g - s.eps_g,
    ]


def zlb_eqs(lead, curr, lag, s, p):          # only the ZLB binds
    eqs = ref_eqs(lead, curr, lag, s, p)
    eqs[2] = curr.r - (-p.r_ss)
    return eqs


def borr_eqs(lead, curr, lag, s, p):         # only the borrowing cap binds
    eqs = ref_eqs(lead, curr, lag, s, p)
    eqs[3] = curr.b - p.b_bar
    return eqs


ss = {v: 0.0 for v in variables}
kw = dict(variables=variables, shocks=shocks, params=params, steady_state=ss)
m_ref = build_dynare(ref_eqs, **kw)
m_zlb = build_dynare(zlb_eqs, check_steady_state=False, strict=False, **kw)
m_borr = build_dynare(borr_eqs, check_steady_state=False, strict=False, **kw)

c_zlb = OccBinConstraint(variable="r", threshold=-params["r_ss"], operator="<")
c_borr = OccBinConstraint(variable="b", threshold=params["b_bar"], operator=">")

shock_seq = np.zeros((40, 3))
shock_seq[0, 0] = -0.06      # eps_g: demand contraction
shock_seq[0, 2] = 0.05       # eps_b: credit surge

res = solve_multiconstraint_occbin(
    m_unconstrained=m_ref,
    m_constrained_dict={"zlb": m_zlb, "borrowing": m_borr},
    shock_seq=shock_seq,
    constraints={"zlb": c_zlb, "borrowing": c_borr},
    horizon=40,
)

regimes = np.asarray(res.regimes)
print(res.converged, res.iterations)          # True 3
print(regimes[:12])                           # [3 1 1 0 0 0 0 0 0 0 0 0]
print(np.sum((regimes & 1) == 1))             # 3 — ZLB binds, alone or jointly
print(np.sum(regimes == 3))                   # 1 — BOTH bind
print([c for c in res.shadow_path.columns if c.endswith("_shadow")])
#                                             # ['r_shadow', 'b_shadow']
```

`result.constraints` maps each model key to the constraint actually
used, and `result.shadow_path` adds one `<variable>_shadow` column per
constraint (`r_shadow`, `b_shadow` above), which is what the
complementary-slackness check reads.

**Non-convergence is now reported, not implied.** `result.converged` is
`True` only when the regime iteration reached a fixed point, the returned
path respects every constraint where that is meaningful for its style,
and every constraint is slack in the final period so the terminal
condition is verified. Otherwise the function returns `converged=False`
**and emits a single `UserWarning` naming every reason** — `max_iter`
exhausted, a regime cycle damping could not break, a violated bound, a
constraint still binding at $T$. The path that comes back is then the one
solved under `result.regimes`: a diagnostic, not a solution.

Other `ValueError` conditions: a non-positive `horizon` or `max_iter`, no
constrained model at all, a constraint naming an unknown variable or
`relax_variable`, a constrained model identical to the reference, two
constrained models rewriting the same equation row, and an alternative
regime that pegs a variable in a row that does not determine it in the
reference model without a `relax_variable` (the relax test would be
vacuous).

### 4.2 The Piecewise Kalman Filter (Giovannini et al. 2021)

Estimating models with occasionally binding constraints historically required particle filtering or nonlinear likelihood inversions. 

The **Piecewise Kalman Filter (PKF)** of **Giovannini, Pfeiffer, and Ratto (2021, *Journal of Economic Dynamics and Control*)** solves this problem by embedding the piecewise linear state space directly within forward filtering:
1. **Forecast Step**: Given filtered state $\hat{x}_{t-1|t-1}$ and covariance $P_{t-1|t-1}$, solve OccBin backward recursions under zero expected future shocks to obtain time-varying transition matrices $T_t(r)$ and constants $c_t(r)$.
2. **Measurement Update**: Compute forecast errors $v_t = y_t - Z \hat{x}_{t|t-1} - d_t$ and update state and covariance estimates via the standard Kalman gain $K_t$.
3. **Likelihood Accumulation**: Evaluate the log-likelihood Gaussian density piece-by-piece:

$$\ln L(Y \mid \theta) = - \frac{1}{2} \sum_{t=1}^T \left( n_y \ln(2\pi) + \ln |F_t| + v_t^\top F_t^{-1} v_t \right)$$

### 4.3 Bayesian Estimation with PKF

```python
from puremacro.dsge import load_mod, OccBinConstraint

# Define unconstrained and constrained models
m_ref = load_mod("model_unconstrained.mod")
m_zlb = load_mod("model_zlb.mod")

# Define constraint condition: policy rate hits 0 when notional rate <= 0
constraint = OccBinConstraint(
    name="zlb",
    condition="i_notional <= 0",
    constrained_model=m_zlb,
)

# Estimate parameters using the Piecewise Kalman Filter
res_est = m_ref.estimate(
    data=data_df,
    method="piecewise_kalman",
    occbin_regimes=[constraint],
    n_draws=2000,
    burn_in=500,
)

print(res_est.summary())
```

---

## 5. Sequential Monte Carlo (SMC) & Particle Filtering

### 5.1 The Herbst & Schorfheide (2014, 2015) SMC Algorithm

Standard Random Walk Metropolis-Hastings (RWMH) algorithms struggle on complex macroeconomic likelihood surfaces characterized by multiple local modes, ridge collinearities, and flat parameter directions.

Puremacro implements the **Sequential Monte Carlo (SMC)** algorithm of **Herbst & Schorfheide (2014, *Journal of Applied Econometrics*; 2015, *Princeton University Press*)**. SMC bridges the prior $\pi(\theta)$ to the posterior $p(\theta|Y)$ through a sequence of $N$ intermediate tempered distributions:

$$\pi_n(\theta) \propto \pi(\theta) \left[ p(Y \mid \theta) \right]^{\phi_n}, \quad 0 = \phi_0 < \phi_1 < \dots < \phi_N = 1$$

Key features:
1. **Adaptive Tempering**: Instead of a fixed grid, puremacro dynamically chooses $\phi_{n+1}$ by solving:

$$\text{ESS}(\phi_{n+1}) = \alpha^* \cdot N_{\text{particles}}$$

using a 1D bisection solver, ensuring optimal particle diversity across stages.
2. **Systematic Resampling**: Particles with low weights are resampled when effective sample size falls below the threshold $ESS < 0.5 N_{part}$.
3. **Particle Mutation**: Resampled particles mutate through multiple steps of Metropolis-Hastings with empirical covariance matrix $\Sigma_n = \text{Cov}_\pi(\theta)$ and adaptive scaling factor $c_n$.

### 5.2 Exact Marginal Data Density (MDD)

A decisive advantage of SMC over MCMC is the exact computation of the **Marginal Data Density (MDD)** $\ln p(Y)$ as a cumulative product of stage normalizing constants:

$$\ln \hat{p}(Y) = \sum_{n=1}^N \ln \left( \frac{1}{M} \sum_{m=1}^M w_n^{(m)} \right)$$

where $w_n^{(m)} = [p(Y \mid \theta_{n-1}^{(m)})]^{\phi_n - \phi_{n-1}}$ are the incremental importance weights. Puremacro also reports the asymptotic numerical standard error $\text{SE}(\ln \hat{p}(Y))$.

### 5.3 Nonlinear Bootstrap Particle Filter

For nonlinear DSGE models (orders 2 and 3) or non-Gaussian measurement densities, puremacro provides a native pure-NumPy **Bootstrap Particle Filter**:
- Propagates particles through the pruned nonlinear state equations.
- Reweights particles against empirical observation densities.
- Evaluates the exact nonlinear log-likelihood with systematic resampling.

```python
from puremacro.dsge import load_mod
from puremacro.dsge.smc import SMCSampler

model = load_mod("smets_wouters_07.mod")

# Configure SMC Sampler
smc = SMCSampler(
    model,
    data=macro_data,
    varobs=["dy", "dc", "dinve", "dw", "pinf", "robs", "labobs"],
    n_particles=2000,
    n_stages=50,
    adaptive_tempering=True,
    target_ess=0.5,
    seed=123,
)

# Sample posterior
res_smc = smc.sample()

print(f"Log Marginal Data Density (MDD): {res_smc.mdd:.3f} +/- {res_smc.mdd_se:.3f}")
print(res_smc.posterior_summary)

# Visual stage diagnostics
res_smc.plot_stages()
res_smc.plot_posterior()
```

---

## 6. Nonlinear Ramsey Optimal Policy & BGP Detrending

### 6.1 Automated Ramsey Optimal Policy

Under commitment, a benevolent policymaker maximizes intertemporal household welfare subject to the decentralized structural equilibrium conditions of the economy:

$$\max_{\{y_t\}_{t=0}^\infty} \mathbb{E}_0 \sum_{t=0}^\infty \beta^t U(y_t) \quad \text{s.t.} \quad \mathbb{E}_t f(y_{t+1}, y_t, y_{t-1}, u_t) = 0$$

Puremacro's `ramsey_model` (`puremacro.dsge.ramsey`) sets this problem up from an objective string and solves its linear-quadratic approximation:
1. Forms the planner's Lagrangian:

$$\mathcal{L} = \mathbb{E}_0 \sum_{t=0}^\infty \beta^t \left[ U(y_t) + \lambda_t^\top f(y_{t+1}, y_t, y_{t-1}, u_t) \right]$$

2. Derives the First-Order Conditions (FOCs) symbolically over the expression DAG, with respect to every endogenous variable $y_{i,t}$ and Lagrange multiplier $\lambda_{j,t}$. They are **for display only**: `focs` holds them as readable equations in Dynare syntax and `foc_nodes` as expression trees.

$$\frac{\partial U(y_t)}{\partial y_{i,t}} + \left[ \frac{\partial f(y_{t+1}, y_t, y_{t-1}, u_t)}{\partial y_{i,t}} \right]^\top \lambda_t + \beta^{-1} \left[ \frac{\partial f(y_t, y_{t-1}, y_{t-2}, u_{t-1})}{\partial y_{i,t}} \right]^\top \lambda_{t-1} + \beta \mathbb{E}_t \left[ \frac{\partial f(y_{t+2}, y_{t+1}, y_t, u_{t+1})}{\partial y_{i,t}} \right]^\top \lambda_{t+1} = 0$$

3. Assembles the system it actually solves numerically, a linear-quadratic one: the constraints linearized at the steady state (the model's linear matrices, or the `.mod` equations differentiated at the steady state) and the Hessian of the objective at the steady state, with the steady-state multipliers set to zero. The state vector is augmented with the multipliers $\lambda_t$.
4. Solves the saddle-path equilibrium with Klein QZ. The law of motion is the same for the Ramsey plan chosen at $t_0$ and for the timeless-perspective rule (Woodford 2003); impulse responses start from the steady state with $\lambda_{-1} = 0$, where the two coincide (see [DSGE Frontier, §1.3](dsge_phase_c.md)).

The solution is exact when the objective is a quadratic loss centred on the steady state, as in the example below. For a nonlinear welfare objective whose steady-state multipliers are not zero, it is only a linear-quadratic approximation, without the second-order terms of a full Ramsey linearization. On a linear model, `ramsey_model` shares the matrices and the QZ solver with `lq_commitment`.

The model below has two equations for three variables: the interest rate `i` is the instrument and has no rule. `load_mod` needs as many equations as variables, so the `.mod` text goes to `ramsey_model` directly, with the instrument named.

```python
from puremacro.dsge import ramsey_model

# Canonical 3-equation New Keynesian model without a policy rule: the planner's
# constraints are the IS curve and the Phillips curve, and i is the instrument.
NK_MOD = """
var x pi i;
varexo eps_u;
parameters beta sigma kappa;
beta = 0.99; sigma = 1.0; kappa = 0.15;

model;
  x = x(+1) - (1/sigma) * (i - pi(+1));
  pi = beta * pi(+1) + kappa * x + eps_u;
end;
"""

# Optimal commitment policy minimizing the quadratic loss L = pi^2 + 0.1 * x^2
ramsey_res = ramsey_model(
    NK_MOD,
    objective="pi^2 + 0.1 * x^2",
    planner_discount=0.99,
    instruments=["i"],
)

print("Ramsey FOCs (for display):")
for eq in ramsey_res.focs:
    print(" ", eq)

# Responses to a cost-push shock satisfy the commitment target criterion
# pi_t = -(0.1 / kappa) * (x_t - x_{t-1}): pi_0 = 0.6274, x_0 = -0.9411
print(ramsey_res.irf("eps_u", horizon=4).round(4))

# Impulse responses of the endogenous variables and the multipliers
fig = ramsey_res.plot(shock="eps_u", horizon=20)
```

### 6.2 Balanced Growth Path (BGP) Detrending

Empirical macro models often feature non-stationary deterministic trends (e.g. labor-augmenting technological growth $A_t = \gamma^t A_0$ or investment-specific technical progress).

Puremacro provides automated Balanced Growth Path (BGP) stationarization:
- Declare trend variables: `trend_var A;` or `log_trend_var a;`
- Assign deflators to trending variables: `var(deflator=A) Y C I;`
- Puremacro automatically generates the stationary transformed system $y_t^* = Y_t / A_t$ and derives the steady state along the balanced growth path.

---

## 7. Summary Comparison Matrix

| Capability | Module | Input Class | Output Class | Key Analytical Method |
| :--- | :--- | :--- | :--- | :--- |
| **Order-3 Perturbation** | `puremacro.dsge.pruning` | `LinearModel` / `.mod` | `Order3PrunedSolution` | 3-Fold Schur Sylvester & Andreasen Pruning |
| **Deterministic Transitions** | `puremacro.dsge.perfect_foresight` | `LinearModel` / `.mod` | `PerfectForesightResult` | Newton-Raphson sparse block-tridiagonal solve |
| **Complementarity MCP** | `puremacro.dsge.perfect_foresight` | `LinearModel` / `.mod` | `MCPResult` | Semismooth Newton & Fischer-Burmeister |
| **Multi-Constraint OccBin** | `puremacro.dsge.occbin` | Dict of `LinearModel` | `OccBinResult` | Guerrieri-Iacoviello Riccati iterations |
| **Piecewise Kalman Filter** | `puremacro.dsge.estimate` | `LinearModel` + Dict | `PiecewiseKalmanResult` | Giovannini-Pfeiffer-Ratto likelihood |
| **Sequential Monte Carlo** | `puremacro.dsge.smc` | `LinearModel` + Data | `SMCResult` | Herbst-Schorfheide adaptive tempering |
| **Particle Filter** | `puremacro.dsge.smc` | `Order3PrunedSolution` | `tuple[float, np.ndarray]` | Bootstrap sequential importance resampling |
| **Ramsey Optimal Policy** | `puremacro.dsge.ramsey` | `LinearModel` / `.mod` + Obj | `RamseyResult` | LQ commitment (linearized constraints + objective Hessian) + Klein QZ; symbolic FOCs for display |
| **BGP Detrending** | `puremacro.dsge._parser` | `.mod` with `deflator` | `ParsedModelDAG` | Automated algebraic stationarization |

---

## 8. References

- **Andreasen, M. M., Fernández-Villaverde, J., & Rubio-Ramírez, J. F. (2018)**. *The Pruned State-Space System for Non-Linear DSGE Models: Theory and Empirical Applications*. Review of Economic Studies, 85(1), 1–49.
- **Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021)**. *Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models*. Econometrica, 89(6), 2787–2815.
- **Boucekkine, R. (1995)**. *An Alternative Methodology for Solving Nonlinear Forward-Looking Models*. Journal of Economic Dynamics and Control, 19(4), 711–734.
- **Brock, W. A., & Mirman, L. J. (1972)**. *Optimal Economic Growth and Uncertainty: The Discounted Case*. Journal of Economic Theory, 4(3), 479–513.
- **Clarida, R., Galí, J., & Gertler, M. (1999)**. *The Science of Monetary Policy: A New Keynesian Perspective*. Journal of Economic Literature, 37(4), 1661–1707.
- **Coeurdacier, N., Rey, H., & Winant, P. (2011)**. *The Risky Steady State*. American Economic Review, 101(3), 398–401.
- **Dennis, R. (2007)**. *Optimal Policy in Rational Expectations Models: New and Alternative Solutions*. Journal of Economic Dynamics and Control, 31(12), 3959–3983.
- **Giovannini, M., Pfeiffer, P., & Ratto, M. (2021)**. *The Piecewise Kalman Filter for Occasionally Binding Constraints*. Journal of Economic Dynamics and Control, 128, 104128.
- **Guerrieri, L., & Iacoviello, M. (2015)**. *OccBin: A Toolkit for Solving Dynamic Models with Occasionally Binding Constraints Easily*. Journal of Monetary Economics, 75, 38–54.
- **Herbst, E., & Schorfheide, F. (2014)**. *Sequential Monte Carlo Sampling for DSGE Models*. Journal of Applied Econometrics, 29(7), 1073–1098.
- **Herbst, E., & Schorfheide, F. (2015)**. *Bayesian Estimation of DSGE Models*. Princeton University Press.
- **Juillard, M. (1996)**. *Dynare: A Program for the Resolution and Simulation of Dynamic Models with Forward-Looking Variables Through the Use of a Relaxation Algorithm*. CEPREMAP Working Paper 9602.
- **Klein, P. (2000)**. *Using the Generalized Schur Form to Solve a System of Linear Expectational Difference Equations*. Journal of Economic Dynamics and Control, 24(10), 1405–1423.
- **Woodford, M. (2003)**. *Interest and Prices: Foundations of a Theory of Monetary Policy*. Princeton University Press.
