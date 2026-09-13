> 🇬🇧 English · 🇪🇸 [Español](es/dcegm.md)

# Discrete Choice Endogenous Grid Method (DC-EGM) and Non-Convex Dynamic Programming

`puremacro.vfi.dcegm` implements the Discrete Choice Endogenous Grid Method (DC-EGM) developed by **Iskhakov, Jørgensen, Rust, and Schjerning (2017, *Quantitative Economics*)**, extending the continuous Endogenous Grid Method (**Carroll 2006**) to dynamic structural models featuring both **discrete choices** (retirement, labor force participation, mortgage default, durable replacement) and **continuous choices** (consumption, liquid saving, wealth accumulation).

In standard dynamic programming, combining discrete and continuous decisions destroys the concavity of the value function. Discrete choices introduce kinks and non-convex regions in $V(M)$, which cause the continuous Euler equation mapping $a' \mapsto M(a')$ to fold backwards and become multi-valued. Traditional grid search methods require dense lattices and global optimization at every state point. DC-EGM resolves this with machine-level efficiency:

- **Fast EGM Inversion**: Inverts the first-order condition on an exogenous post-decision savings grid $a'$, obtaining candidate consumption $c(a')$ and cash-on-hand $M(a') = a' + c(a')$ without non-linear root finding.
- **Fast Upper Envelope Filtering**: Systematically scans for folds and intersecting branches in $(M, v_d(M))$ space, eliminating sub-optimal branches that satisfy the Euler equation locally but fail to attain global optimality.
- **Extreme Value Taste Shocks**: Incorporates additive Type-I Extreme Value (Gumbel) shocks (**Rust 1987**), providing smooth closed-form logit choice probabilities $P(d \mid M)$ and closed-form inclusive values (Log-Sum-Exp).
- **Exact Envelope Theorem Expectations**: Calculates expected future marginal values $\mathbb{E}[V'(M')]$ directly from choice probabilities without numerical differentiation.

---

## 1. Theoretical & Econometric Framework

### 1.1 Model Formulation & Timing

Consider an agent in period $t$ with cash-on-hand $M_t$. The agent makes a discrete decision $d_t \in \{0, 1, \dots, D-1\}$ (for instance, $d=0$ for working and $d=1$ for retirement) and chooses continuous consumption $c_t \in (0, M_t]$, leaving end-of-period savings $a_{t+1} = M_t - c_t \ge a_{\min}$.

The agent's utility is augmented by additive choice-specific taste shocks $\boldsymbol{\epsilon}_t = (\epsilon_{0, t}, \dots, \epsilon_{D-1, t})$ drawn independently from an Extreme Value Type-I (Gumbel) distribution with scale parameter $\sigma_\epsilon \ge 0$:

$$U(c_t, d_t, \boldsymbol{\epsilon}_t) = u(c_t, d_t) + \epsilon_{d, t}$$

The Bellman equation for the ex-ante value function $V_t(M_t)$ is:

$$V_t(M_t) = \max_{d \in \{0, \dots, D-1\}} \left\{ v_t(M_t, d) + \epsilon_{d, t} \right\}$$

where the choice-specific conditional value function $v_t(M_t, d)$ is:

$$v_t(M_t, d) = \max_{a_{t+1} \ge a_{\min}} \left\{ u(M_t - a_{t+1}, d) + \beta \mathbb{E}\left[ V_{t+1}\left( (1+r)a_{t+1} + y_{t+1}(d_{t+1}, z_{t+1}) \right) \;\middle|\; d_t = d, a_{t+1} \right] \right\}$$

with real interest rate $r$ and gross return $R = 1 + r$.

---

### 1.2 Extreme Value Taste Shocks & Inclusive Value

Integrating analytically over the joint distribution of taste shocks $\boldsymbol{\epsilon}_t$ yields the ex-ante expected value function (**inclusive value** / social surplus):

$$\mathcal{V}_t(M_t) \equiv \mathbb{E}_{\boldsymbol{\epsilon}}\left[ \max_{d \in \{0, \dots, D-1\}} \left\{ v_t(M_t, d) + \epsilon_{d, t} \right\} \right] = \sigma_\epsilon \ln \left( \sum_{d=0}^{D-1} \exp\left(\frac{v_t(M_t, d)}{\sigma_\epsilon}\right) \right) + \gamma_{\text{Euler}} \sigma_\epsilon$$

where $\gamma_{\text{Euler}} \approx 0.5772$ is the Euler-Mascheroni constant.

#### Smooth Choice Probabilities

The conditional choice probability for discrete action $d$ follows the standard multinomial logit formula:

$$P_t(d \mid M_t) = \frac{\exp\left( \frac{v_t(M_t, d)}{\sigma_\epsilon} \right)}{\sum_{d'=0}^{D-1} \exp\left( \frac{v_t(M_t, d')}{\sigma_\epsilon} \right)} = \frac{1}{\sum_{d'=0}^{D-1} \exp\left( \frac{v_t(M_t, d') - v_t(M_t, d)}{\sigma_\epsilon} \right)}$$

In the deterministic limit as $\sigma_\epsilon \to 0$, the logit probability collapses to the indicator function $\mathbf{1}\{d = \arg\max_{d'} v_t(M_t, d')\}$, and the inclusive value converges to $\max_d v_t(M_t, d)$.

---

### 1.3 Expected Marginal Value via the Envelope Theorem

To apply the Endogenous Grid Method, one needs the expected marginal value with respect to savings $a_{t+1}$:

$$\mathfrak{w}_t(a_{t+1}) \equiv \beta R \, \mathbb{E}\left[ \mathcal{V}'_{t+1}(M_{t+1}) \;\middle|\; a_{t+1} \right]$$

Traditional dynamic programming relies on numerical finite differences to approximate $\mathcal{V}'_{t+1}$, which is noisy and computationally expensive. By the **Envelope Theorem** and the chain rule of differentiation on the inclusive value:

$$\mathcal{V}'_{t+1}(M) = \sum_{d=0}^{D-1} \frac{\partial \mathcal{V}_{t+1}}{\partial v_{t+1}(M, d)} \frac{\partial v_{t+1}(M, d)}{\partial M} = \sum_{d=0}^{D-1} P_{t+1}(d \mid M) \, u'\left(c_{t+1}^*(M, d), d\right)$$

This remarkable result provides the exact continuous expected marginal value as a probability-weighted sum of marginal utilities, **with zero numerical differentiation error**:

$$\mathfrak{w}_t(a_{t+1}) = \beta R \sum_{z'} \Pi(z, z') \sum_{d'=0}^{D-1} P_{t+1}\left(d' \;\middle|\; R a_{t+1} + y'(d', z')\right) u'\left(c_{t+1}^*\left(R a_{t+1} + y'(d', z'), d'\right), d'\right)$$

Given an exogenous grid of post-decision savings $a' \in \{a_1', \dots, a_K'\}$, the Euler equation is inverted directly for each discrete choice $d$:

$$c_t(a'; d) = (u')^{-1}\left( \mathfrak{w}_t(a') \right), \qquad M_t(a'; d) = a' + c_t(a'; d)$$

---

### 1.4 The Fast Upper Envelope Algorithm

Because the value function $V_{t+1}$ contains kinks from discrete choices, $v_t(M, d)$ is not necessarily concave in $M$. As a consequence:
1. The mapping $a' \mapsto M(a'; d)$ is not guaranteed to be strictly monotonic: it can fold backwards ($M(a'_{k+1}) < M(a'_k)$).
2. For a given wealth level $M$, there can be multiple values of $a'$ satisfying the Euler equation with equality. Only one of these corresponds to the global maximum; the others are local minima or inferior local maxima.

The **Fast Upper Envelope Algorithm** prunes these sub-optimal branches:

```text
Endogenous Grid with Non-Convex Fold:
  v(M)
   ^               Branch 1 (local max)         Branch 2 (global max)
   |                     /---\                     /----
   |                    /     \                   /
   |                   /       \                 /
   |                  /         \   Crossing    /
   |                 /           \     *       /
   |                /             \---/ \     /
   |               /                     \---/  <-- Pruned Sub-Optimal Branch
   +----------------------------------------------------> M
                                       M*
```

#### Step-by-Step Pruning Procedure

1. **Monotone Segmentation**: Partition the raw sequence of points $(M_k, c_k, v_k)_{k=1}^K$ into continuous monotone segments where $M_k$ is strictly increasing.
2. **Value Evaluation**: For every point on the endogenous grid, compute the choice-specific value:
   $$v_k = u(c_k, d) + \beta \, \mathfrak{W}(a'_k)$$
3. **Intersection Detection**: For any overlapping cash-on-hand intervals $[\underline{M}, \bar{M}]$ where two branches exist, compute the crossing point $M^*$ where $v_1(M^*) = v_2(M^*)$ via linear interpolation.
4. **Dominance Elimination**: Below the intersection $M^*$, discard the lower-valued branch. Above $M^*$, discard the alternative lower-valued branch.
5. **Continuous Interpolation**: Interpolate the resulting monotonically pruned upper envelope onto the target cash-on-hand evaluation grid $M \in \mathbf{M}_{\text{eval}}$.

---

## 2. Model Specifications & Algorithmic Options

`DCEGMProblem` supports both stationary infinite-horizon problems and life-cycle finite-horizon backward induction:

| Feature / Setting | Description | Default |
|---|---|---|
| `horizon` | Finite horizon periods $T$ (or `None` for stationary infinite horizon) | `None` (infinite horizon) |
| `n_choices` | Number of discrete alternatives $D \ge 2$ | `2` (e.g., work vs. retire) |
| `sigma_eps` | Extreme value taste shock scale parameter $\sigma_\epsilon \ge 0$ | `0.20` |
| `r` | Real interest rate (gross return $R = 1 + r$) | `0.04` |
| `beta` | Subjective discount factor $\beta \in (0, 1)$ | `0.96` |
| `income` | Choice-specific labor/pension income vector $y(d)$ or $y(d, z)$ | `[1.0, 0.4]` |
| `discrete_transitions` | Transition feasibility matrix between discrete states | `None` (all choices feasible) |
| `a_grid` | Exogenous savings grid $a' \ge a_{\min}$ | Required 1D array |

---

## 3. Canonical Calibration: The Retirement Choice Problem

The canonical benchmark specification (**Iskhakov et al. 2017**):
- **Choice 0 (Work)**: Earns wage $w = 1.0$, suffers disutility of labor $\psi = 0.5$.
- **Choice 1 (Retire)**: Receives pension $p = 0.4$, zero disutility of labor ($\psi = 0$).
- **Preferences**: CRRA / log utility $u(c, d) = \ln(c) - \psi \cdot \mathbf{1}\{d = 0\}$.
- **Economic Invariants**:
  - Low-wealth households cannot afford to retire ($P(\text{work} \mid M) \to 1$) because pension income is insufficient for consumption smoothing.
  - Wealthy households retire ($P(\text{retire} \mid M) > 0.75$) to enjoy leisure without sacrificing living standards.
  - Consumption exhibits an upward jump or change in slope at the retirement threshold $M^*$.

---

## 4. Runnable Worked Examples

### 4.1 Fast Upper Envelope Filtering on a Folding Grid

This example demonstrates the core `upper_envelope` algorithm on a synthetic endogenous grid with two intersecting branches and a non-monotonic fold:

```python
import numpy as np
from puremacro.vfi import upper_envelope, UpperEnvelopeResult

# Branch 1 (steep value, dominant at high wealth): v1(M) = -2.0 + 1.5 * M
M1 = np.linspace(1.5, 4.0, 30)
c1 = 0.6 * M1
v1 = -2.0 + 1.5 * M1

# Branch 2 (flat value, dominant at low wealth): v2(M) = 0.0 + 0.5 * M
M2 = np.linspace(0.5, 2.5, 30)
c2 = 0.3 * M2
v2 = 0.0 + 0.5 * M2

# Concatenate in reverse to create a non-monotonic endogenous grid fold
M_raw = np.concatenate([M2, M1])
c_raw = np.concatenate([c2, c1])
v_raw = np.concatenate([v2, v1])

# Target exogenous evaluation grid for cash-on-hand M
exog_grid = np.linspace(0.6, 3.5, 40)

# Filter with Upper Envelope
res = upper_envelope(M_raw, c_raw, v_raw, exog_grid)
assert isinstance(res, UpperEnvelopeResult)
c_clean, v_clean = res

# Below intersection (M < 2.0), policy tracks Branch 2 (c = 0.3 * M)
below_kink = exog_grid < 1.9
assert np.allclose(c_clean[below_kink], 0.3 * exog_grid[below_kink], atol=1e-3)

# Above intersection (M > 2.1), policy tracks Branch 1 (c = 0.6 * M)
above_kink = exog_grid > 2.1
assert np.allclose(c_clean[above_kink], 0.6 * exog_grid[above_kink], atol=1e-3)

print("Upper Envelope successfully pruned the sub-optimal branch!")
```

### 4.2 Dynamic Retirement Problem with Extreme Value Shocks

Solving the full infinite-horizon discrete-continuous retirement model:

```python
import numpy as np
from puremacro.vfi import DCEGMProblem, solve_dcegm

# 1. Setup asset grid and model parameters
a_grid = np.linspace(0.0, 12.0, 40)

prob = DCEGMProblem(
    a_grid=a_grid,
    n_choices=2,
    beta=0.96,
    r=0.04,
    sigma_eps=0.25,
    options={"tol": 1e-5, "max_iter": 500},
)

# 2. Solve dynamic programming problem via DC-EGM
sol = solve_dcegm(prob)
assert sol.converged
assert sol.sup_norm < 1e-4

# 3. Inspect economic behavior
# Choice 0 = Work, Choice 1 = Retire
p_work = sol.choice_probabilities[0]
p_retire = sol.choice_probabilities[1]

print(f"Iterations: {sol.n_iter} | Final Sup-Norm Residual: {sol.sup_norm:.2e}")
print(f"P(work | lowest wealth):   {p_work[0]:.3f}")
print(f"P(retire | highest wealth): {p_retire[-1]:.3f}")

# Verification of economic invariants
assert p_work[0] > 0.85, "Poor agents must work"
assert p_retire[-1] > 0.70, "Wealthy agents should retire"
assert np.all(sol.c > 0.0), "Consumption must be strictly positive"
```

### 4.3 Continuous Evaluation and Policy Queries

Querying policy, value, and logit choice probabilities at arbitrary continuous cash-on-hand levels:

```python
# Query at a single continuous cash-on-hand state
m_query = 3.5
c_work = sol.policy(m_query, choice=0)
c_retire = sol.policy(m_query, choice=1)
c_expected = sol.policy(m_query, choice=None)
v_inclusive = sol.value(m_query, choice=None)
p_retire_val = sol.choice_prob(m_query, choice=1)

print(f"At M = {m_query:.2f}:")
print(f"  Consumption if working:  c_0 = {c_work:.3f}")
print(f"  Consumption if retired:  c_1 = {c_retire:.3f}")
print(f"  Expected consumption:    E[c] = {c_expected:.3f}")
print(f"  Probability of retiring: P(d=1) = {p_retire_val:.1%}")

# Vectorized continuous evaluation
m_vec = np.linspace(1.0, 8.0, 10)
p_work_vec = sol.choice_prob(m_vec, choice=0)
assert len(p_work_vec) == 10
assert np.all((p_work_vec >= 0.0) & (p_work_vec <= 1.0))
```

---

## 5. Full API Specification

### `upper_envelope`

```text
upper_envelope(
    m_raw: np.ndarray,
    c_raw: np.ndarray,
    v_raw: np.ndarray,
    m_grid: np.ndarray,
) -> UpperEnvelopeResult
```

#### Parameters:
- `m_raw`: Raw 1D endogenous cash-on-hand coordinates from EGM inversion.
- `c_raw`: Raw candidate consumption values corresponding to `m_raw`.
- `v_raw`: Raw candidate value function values corresponding to `m_raw`.
- `m_grid`: Target strictly increasing 1D evaluation grid.

#### Returns:
- `UpperEnvelopeResult`: 2-tuple `(c_clean, v_clean)` of pruned, monotonic consumption and upper envelope value function. Supports attribute access `.c` and `.v`.

---

### `DCEGMProblem`

```text
DCEGMProblem(
    a_grid: Sequence[float] | np.ndarray,
    m_grid: Sequence[float] | np.ndarray | None = None,
    n_choices: int = 2,
    u_fn: Callable | Sequence[Callable] | None = None,
    u_prime: Callable | Sequence[Callable] | None = None,
    u_prime_inv: Callable | Sequence[Callable] | None = None,
    beta: float = 0.96,
    r: float = 0.04,
    income: Any = None,
    P_z: np.ndarray | None = None,
    z_grid: Sequence[float] | np.ndarray | None = None,
    sigma_eps: float = 0.2,
    a_min: float = 0.0,
    horizon: int | None = None,
    discrete_transitions: np.ndarray | None = None,
    options: dict | None = None,
    metadata: dict | None = None,
)
```

#### Parameters:
- `a_grid`: Strictly increasing exogenous grid of post-decision assets $a' \ge a_{\min}$.
- `m_grid`: Target cash-on-hand evaluation grid $M$. If `None`, automatically generated spanning the feasible wealth domain.
- `n_choices`: Number of discrete choices $D \ge 2$.
- `u_fn`: Utility function $u(c, d)$. Defaults to log utility with labor disutility.
- `u_prime`: Marginal utility function $u'(c, d)$.
- `u_prime_inv`: Inverse marginal utility $(u')^{-1}(w, d)$.
- `beta`: Discount factor $\beta \in (0, 1)$.
- `r`: Real interest rate $r > -1$.
- `income`: Income schedule $y(d)$ or $y(d, z)$ (default `[1.0, 0.4]`).
- `P_z`: Markov transition probability matrix for exogenous income shocks.
- `z_grid`: Discrete support grid for income shocks.
- `sigma_eps`: Gumbel taste shock scale $\sigma_\epsilon \ge 0$.
- `a_min`: Borrowing constraint lower bound on assets.
- `horizon`: Number of backward induction periods $T$ for life-cycle models, or `None` for stationary infinite horizon.
- `discrete_transitions`: Feasibility matrix for transitions between discrete choices.
- `options`: Solver dictionary (`tol`, `max_iter`).

---

### `solve_dcegm`

```text
solve_dcegm(
    problem_or_a_grid: DCEGMProblem | Sequence[float] | np.ndarray,
    m_grid: Sequence[float] | np.ndarray | None = None,
    *args: Any,
    backend: str = "numpy",
    **kwargs: Any,
) -> DCEGMSolution
```

Solves the discrete-continuous dynamic program via DC-EGM and upper envelope filtering.

---

## 6. Result Interface & Manuscript Export

`DCEGMSolution` provides rich inspection, evaluation, and reporting methods:

- **Attributes**:
  - `.c`: ChoiceMapping of pruned consumption policies $c(M, d)$.
  - `.choice_values`: ChoiceMapping of conditional value functions $v_d(M)$.
  - `.choice_probabilities`: ChoiceMapping of logit probabilities $P(d \mid M)$.
  - `.integrated_value`: 1D array of inclusive values $\mathcal{V}(M)$.
  - `.aprime`: ChoiceMapping of savings policies $a'(M, d) = M - c(M, d)$.
  - `.converged`: Boolean convergence flag.
  - `.n_iter`: Total iterations evaluated.
  - `.sup_norm`: Final sup-norm Bellman residual.
- **Continuous Evaluation Methods**:
  - `.policy(m, choice=None)`: Evaluates consumption. Returns choice-specific policy $c(m, d)$ if `choice` is an int, or probability-weighted expected consumption $\mathbb{E}[c(m)]$ if `choice=None`.
  - `.value(m, choice=None)`: Evaluates conditional value $v_d(m)$ or inclusive value $\mathcal{V}(m)$.
  - `.choice_prob(m, choice=None)`: Evaluates logit choice probability $P(d \mid m)$ or all choice probabilities if `choice=None`.
- **Reporting & Manuscript Export**:
  - `.summary()`: Textual diagnostic summary of choices, convergence, and policy statistics.
  - `.plot()`: Spawns a multi-panel Matplotlib figure illustrating policy functions, choice probabilities across wealth, and conditional value functions.
  - `.to_frame()`: Exports policies, values, and probabilities to a `pandas.DataFrame`.
  - `.to_markdown()`, `.to_latex()`, `.to_typst()`: Publication-ready tables formatted for scientific articles.

---

## References

- **Carroll, C. D. (2006)**. The method of endogenous gridpoints for solving dynamic stochastic optimization problems. *Economics Letters*, 91(3), 312–320.
- **Clausen, A., & Strub, C. (2020)**. A continuous method for discrete choice dynamic programming. *Quantitative Economics*, 11(3), 859–894.
- **Iskhakov, F., Jørgensen, T. H., Rust, J., & Schjerning, B. (2017)**. The endogenous grid method for discrete-continuous dynamic choice models with taste shocks. *Quantitative Economics*, 8(2), 317–365.
- **Rust, J. (1987)**. Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher. *Econometrica*, 55(5), 999–1033.
