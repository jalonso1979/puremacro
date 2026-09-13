> 🇬🇧 English · 🇪🇸 [Español](es/vfi_splines_and_sparse_grids.md)

# Shape-Preserving Splines and High-Dimensional Sparse Grids

`puremacro.vfi.splines` and `puremacro.vfi.smolyak` implement frontier continuous state-space dynamic programming methods for localized curvature modeling and multi-dimensional models:

- **Cubic B-Splines** (**De Boor 1978**) provide smooth $C^2$ continuous approximations with flexible knot boundary conditions (clamped, natural, not-a-knot).
- **Schumaker (1983) Shape-Preserving Splines** guarantee strict preservation of monotonicity ($\partial g / \partial k \ge 0$) and concavity ($\partial^2 g / \partial k^2 \le 0$) by inserting adaptive secondary knots, eliminating spurious oscillations and un-economic overshoot near borrowing constraints or sharp curvature changes.
- **Smolyak (1963) Sparse Grid Collocation** (**Krueger & Kubler 2004; Malin, Krueger & Kubler 2011**) solves multi-dimensional continuous state models ($d \in [2, 6]$ continuous states, such as multi-capital, portfolio choice, or open-economy macro models). By combining nested Clenshaw-Curtis extrema nodes and Chebyshev basis polynomials, Smolyak grids scale with complexity $\mathcal{O}(N (\log N)^{d-1})$ rather than the exponential full tensor explosion $\mathcal{O}(N^d)$, reducing grid node counts by up to $100\times$ while preserving high accuracy.

---

## 1. Theoretical & Mathematical Framework

### 1.1 Univariate Cubic B-Splines

B-splines (basis splines) represent continuous functions on an interval $[a, b]$ as linear combinations of localized piecewise polynomial basis functions. Given a non-decreasing knot vector:

$$\tau_0 \le \tau_1 \le \dots \le \tau_{K+3}$$

the $i$-th B-spline basis function $B_{i, p}(x)$ of degree $p$ is defined recursively by the **Cox-De Boor recurrence relation**:

$$B_{i, 0}(x) = \begin{cases} 1 & \text{if } \tau_i \le x < \tau_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

$$B_{i, p}(x) = \frac{x - \tau_i}{\tau_{i+p} - \tau_i} B_{i, p-1}(x) + \frac{\tau_{i+p+1} - x}{\tau_{i+p+1} - \tau_{i+1}} B_{i+1, p-1}(x)$$

For cubic B-splines ($p = 3$):
- Each basis function $B_{i, 3}(x)$ is twice continuously differentiable ($C^2$) across all interior knots.
- It has compact local support spanning only four knot intervals $[\tau_i, \tau_{i+4}]$, ensuring that perturbation of data at one knot does not produce global ripple effects.
- The basis forms a partition of unity: $\sum_{i=0}^{K-1} B_{i, 3}(x) = 1.0$ for all $x \in [a, b]$.

#### Boundary Conditions & Collocation Nodes

`CubicBSplineBasis` (and its alias `SplineBasis`) supports three boundary knot configurations:
1. **Clamped (`bc_type="clamped"`)**: Knots at the endpoints have multiplicity 4 ($\tau_0 = \tau_1 = \tau_2 = \tau_3 = a$ and $\tau_K = \dots = \tau_{K+3} = b$). The first and last basis functions attain exact values $B_0(a) = 1$ and $B_{K-1}(b) = 1$, which is ideal for economic models with fixed boundary states.
2. **Natural (`bc_type="natural"`)**: Imposes zero second derivatives at the boundaries: $S''(a) = S''(b) = 0$.
3. **Not-a-knot (`bc_type="not-a-knot"`)**: Requires continuity of the third derivative $S'''$ across the first and last interior knots.

Collocation equations are enforced at **Greville abscissae** (the local averages of consecutive knots):

$$\xi_i^* = \frac{\tau_{i+1} + \tau_{i+2} + \tau_{i+3}}{3}, \quad i = 0, \dots, K-1$$

---

### 1.2 Schumaker (1983) Shape-Preserving Splines

Standard cubic splines can introduce spurious non-monotonic oscillations (wiggles) when interpolating data with rapid changes in curvature, such as value functions near borrowing limits. Economic decision rules must often obey strict monotonicity ($\partial k' / \partial k \ge 0$) and strict concavity ($\partial^2 V / \partial k^2 \le 0$).

**Schumaker's (1983)** algorithm constructs a $C^1$ continuous quadratic spline $s(x)$ that preserves both monotonicity and concavity.

#### Algorithmic Mechanism

Given grid points $x_1 < x_2 < \dots < x_N$ and values $y_1, y_2, \dots, y_N$:
1. **Slope Estimation**: Compute interval chord slopes $\delta_i = \frac{y_{i+1} - y_i}{x_{i+1} - x_i}$. Assign nodal derivatives $d_i$ that satisfy monotonicity ($d_i \ge 0$ if $\delta_{i-1} \ge 0$ and $\delta_i \ge 0$).
2. **Shape Test**: On each interval $[x_i, x_{i+1}]$, test whether the quadratic interpolant with slopes $d_i, d_{i+1}$ would violate monotonicity or convexity:
   - If $(d_i + d_{i+1})\delta_i \ge 0$ and $|d_i - \delta_i| \le |d_{i+1} - \delta_i|$ (or vice versa), a single quadratic piece cannot preserve shape while matching the slopes.
3. **Adaptive Secondary Knot Insertion**: In intervals requiring shape preservation, Schumaker inserts an interior knot $\xi_i \in (x_i, x_{i+1})$:
   $$\xi_i = x_i + \frac{2(y_{i+1} - y_i) - 2 d_i (x_{i+1} - x_i)}{d_{i+1} - d_i}$$
   and constructs two connected quadratic polynomials on $[x_i, \xi_i]$ and $[\xi_i, x_{i+1}]$.
4. **Guaranteed Properties**:
   - **Monotonicity**: If $y_{i+1} \ge y_i$ for all $i$, then $s'(x) \ge 0$ for all $x \in [x_1, x_N]$.
   - **Concavity**: If chord slopes $\delta_i$ are non-increasing, then $s''(x) \le 0$ for all $x \in [x_1, x_N]$.
   - **No Spurious Oscillations**: The spline never exceeds local extrema: $\min(y_i, y_{i+1}) \le s(x) \le \max(y_i, y_{i+1})$.

---

### 1.3 Smolyak (1963) Sparse Grid Collocation

In multi-dimensional dynamic models with $d$ continuous states (e.g. physical capital $k_t$, human capital $h_t$, foreign assets $b_t$), full tensor product grids explode exponentially: a modest grid of 20 points per dimension requires $20^d$ nodes ($8,000$ for $d=3$, $6.4 \times 10^7$ for $d=6$).

The **Smolyak sparse grid technique** breaks this curse of dimensionality by interpolating on carefully chosen subsets of full tensor product grids.

#### 1. Nested Clenshaw-Curtis Extrema Nodes

In each dimension, 1D interpolation uses nested Clenshaw-Curtis extrema nodes on canonical interval $[-1, 1]$:

$$x_j^i = -\cos\left(\frac{(j - 1)\pi}{m(i) - 1}\right), \quad j = 1, \dots, m(i)$$

with the exponential step size rule:

$$m(1) = 1, \qquad m(i) = 2^{i-1} + 1 \quad \text{for } i \ge 2$$

Because $m(i)$ doubles at each step, nodes are strictly nested: $X^{(1)} \subset X^{(2)} \subset X^{(3)} \subset \dots$.

#### 2. Smolyak Combination Formula

Let $\mathcal{U}^i$ denote the 1D interpolation operator at level $i$. For dimension $d$ and approximation level $\mu \in [1, 5]$, the Smolyak sparse grid operator $\mathcal{A}(d, \mu)$ is defined as:

$$\mathcal{A}(d, \mu) = \sum_{d \le |\mathbf{i}|_1 \le d + \mu} (-1)^{d + \mu - |\mathbf{i}|_1} \binom{d - 1}{d + \mu - |\mathbf{i}|_1} \left( \mathcal{U}^{i_1} \otimes \dots \otimes \mathcal{U}^{i_d} \right)$$

where $\mathbf{i} = (i_1, \dots, i_d)$ is a multi-index vector with $i_k \ge 1$ and $|\mathbf{i}|_1 = \sum_{k=1}^d i_k$.

#### 3. Complexity & Node Count Scaling

Rather than evaluating all $m(\mu + 1)^d$ points of the full tensor product, the Smolyak grid retains only nodes whose multi-index satisfies $|\mathbf{i}|_1 \le d + \mu$:

| Dimension $d$ | Level $\mu$ | Smolyak Nodes $N(d, \mu)$ | Full Tensor Nodes $m(\mu+1)^d$ | Node Reduction Factor |
|---|---|---|---|---|
| **$d = 2$** | $\mu = 2$ | **13** | 25 | $1.9\times$ |
| **$d = 2$** | $\mu = 3$ | **29** | 81 | $2.8\times$ |
| **$d = 3$** | $\mu = 2$ | **25** | 125 | **$5.0\times$** |
| **$d = 3$** | $\mu = 3$ | **69** | 729 | **$10.6\times$** |
| **$d = 4$** | $\mu = 2$ | **41** | 625 | **$15.2\times$** |
| **$d = 5$** | $\mu = 3$ | **241** | 59,049 | **$245\times$** |
| **$d = 6$** | $\mu = 3$ | **389** | 531,441 | **$1,366\times$** |

---

## 2. Methodological & Model Options

| Specification | `SplineCollocationProblem` | `SmolyakProblem` |
|---|---|---|
| **Approximation Basis** | Cubic B-Splines (`"cubic"`) or Schumaker (`"schumaker"`) | Multi-dimensional Chebyshev polynomials on sparse grid |
| **Continuous Dimension** | $d = 1$ (univariate) | $d \in [2, 6]$ (multivariate) |
| **Shape Preservation** | Guaranteed with `spline_type="schumaker"` | Preserves high polynomial accuracy across hypercubes |
| **Knot / Level Tuning** | `n_knots` (default 20), `bc_type="clamped"` | `mu` in $[1, 4]$ (default 2) |
| **Solution Method** | `"euler"` (policy) or `"bellman"` (value) | `"euler"` (policy) or `"bellman"` (value) |
| **Supported Backends** | NumPy, Numba, Apple MLX, CuPy | NumPy, Numba, Apple MLX, CuPy |

---

## 3. Canonical Calibration & Economic Specifications

### 3.1 1D Neoclassical Growth Model (Splines)
- $u(c) = \ln(c)$, $f(k) = k^\alpha$, $\alpha = 0.36$, $\beta = 0.96$, $\delta = 1.0$.
- Steady state: $k_{\text{ss}} = (\alpha \beta)^{\frac{1}{1-\alpha}} \approx 0.1782$.
- Domain: $[0.5 k_{\text{ss}}, 1.5 k_{\text{ss}}]$.

### 3.2 2D Multi-Capital Neoclassical Growth Model (Smolyak)
- Two capital stocks $(k_1, k_2)$ (e.g. equipment vs. structures):
  $$Y = z k_1^{\alpha_1} k_2^{\alpha_2}, \qquad u(c) = \ln(c)$$
  with $\alpha_1 = 0.18, \alpha_2 = 0.18, \beta = 0.96$.
- Analytical policy: $k'_m = \alpha_m \beta Y$ for $m \in \{1, 2\}$.

---

## 4. Runnable Worked Examples

### 4.1 Schumaker Shape-Preserving Spline Verification

This example tests Schumaker's spline on a concave, kinked function $f(x) = \min(2x, 1 + 0.5x)$ where standard unconstrained cubic splines produce overshoot:

```python
import numpy as np
from puremacro.vfi import SchumakerSpline

# Kinked concave function nodes
x_nodes = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
y_nodes = np.minimum(2.0 * x_nodes, 1.0 + 0.5 * x_nodes)

# Fit Schumaker shape-preserving spline
schumaker = SchumakerSpline(x_nodes, y_nodes)

# Dense evaluation grid
x_dense = np.linspace(0.0, 4.0, 200)
y_interp = schumaker(x_dense)
slopes = schumaker.derivative(x_dense)

# Verify strict monotonicity: no negative slopes anywhere
assert np.all(slopes >= -1e-12), "Monotonicity violated!"
print("Schumaker spline verified: slope strictly non-negative across all points.")
```

### 4.2 Continuous Dynamic Programming via Cubic B-Spline Collocation

Solving the continuous Brock-Mirman model using cubic B-splines with clamped boundary knots:

```python
import numpy as np
from puremacro.vfi import (
    CubicBSplineBasis,
    SplineCollocationProblem,
    solve_spline_collocation,
)

alpha = 0.36
beta = 0.96
k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
domain = (0.5 * k_ss, 1.5 * k_ss)

# Setup Spline Collocation Problem with 15 knots
prob_spline = SplineCollocationProblem(
    domain=domain,
    n_knots=15,
    spline_type="cubic",
    bc_type="clamped",
    method="euler",
    return_fn=lambda s, sp, params: np.log(np.maximum(s**alpha - sp, 1e-12)),
    transition_fn=lambda s: s**alpha,
    beta=beta,
    params={"alpha": alpha, "delta": 1.0},
)
sol_spline = solve_spline_collocation(prob_spline)
assert sol_spline.converged

# Evaluate against analytical closed form
eval_k = np.linspace(domain[0], domain[1], 500)
g_true = alpha * beta * (eval_k**alpha)
g_spline = sol_spline.policy(eval_k)

max_spline_err = np.max(np.abs(g_spline - g_true) / g_true)
print(f"Cubic B-Spline Max Policy Relative Error: {max_spline_err:.2e}")
assert max_spline_err < 1e-4
```

### 4.3 High-Dimensional Multi-Capital Model on Smolyak Sparse Grids

Solving a 2D multi-capital model ($k_1, k_2$) on a Smolyak sparse grid of level $\mu = 2$ (only 13 points vs. 25 on full tensor grid):

```python
import numpy as np
from puremacro.vfi import SmolyakGrid, SmolyakProblem, solve_smolyak

alphas = [0.18, 0.18]
beta = 0.96
z = 1.0 / (alphas[0] * beta)
domain_2d = ((0.8, 1.2), (0.8, 1.2))

# 1. Inspect grid reduction ratio
grid = SmolyakGrid(d=2, mu=2, domain=domain_2d)
print(f"2D Smolyak Grid Points (mu=2): {grid.n_nodes} (vs 25 full tensor)")

# 2. Solve 2D Continuous Model
sol_smol = solve_smolyak(
    domain=domain_2d,
    mu=2,
    params={"alphas": alphas, "z": z, "beta": beta},
)
assert sol_smol.converged
assert sol_smol.residual_norm < 1e-4

# 3. Dense out-of-sample evaluation on 500 continuous points
np.random.seed(42)
test_k = np.random.uniform(0.8, 1.2, (500, 2))
kp_approx = sol_smol.policy(test_k)

Y_test = z * (test_k[:, 0] ** alphas[0]) * (test_k[:, 1] ** alphas[1])
kp_true = np.column_stack([alphas[0] * beta * Y_test, alphas[1] * beta * Y_test])

max_smol_err = np.max(np.abs(kp_approx - kp_true) / kp_true)
print(f"Smolyak 2D Max Policy Relative Error: {max_smol_err:.2e}")
assert max_smol_err < 1e-4
```

---

## 5. Full API Specification

### `CubicBSplineBasis` / `SplineBasis`

```text
CubicBSplineBasis(
    knots: Sequence[float] | np.ndarray | None = None,
    domain: tuple[float, float] | None = None,
    n_knots: int = 20,
    degree: int = 3,
    bc_type: str = "clamped",
    knot_spacing: str = "uniform",
)
```

#### Parameters:
- `knots`: Explicit 1D knot vector (if `domain` is None).
- `domain`: Continuous bounds `(k_min, k_max)` for automatic knot generation.
- `n_knots`: Number of breakpoints/knots (default 20).
- `degree`: Spline degree (must be 3 for cubic splines).
- `bc_type`: Boundary condition type: `"clamped"`, `"natural"`, or `"not-a-knot"`.
- `knot_spacing`: Spacing distribution: `"uniform"` or `"geometric"`.

---

### `SchumakerSpline`

```text
SchumakerSpline(
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray | None = None,
    extrapolate: bool = True,
)
```

#### Parameters:
- `x`: Strictly increasing 1D array of state coordinates.
- `y`: 1D array of function values at `x`.
- `s`: Optional pre-specified nodal derivative estimates.
- `extrapolate`: Boolean flag indicating whether to linearly extrapolate beyond `[x[0], x[-1]]`.

---

### `SplineCollocationProblem` & `solve_spline_collocation`

```text
SplineCollocationProblem(
    domain: tuple[float, float] | list,
    n_knots: int = 20,
    spline_type: str = "cubic",
    bc_type: str = "clamped",
    knot_spacing: str = "uniform",
    method: str = "euler",
    euler_residual_fn: Callable | None = None,
    return_fn: Callable | None = None,
    transition_fn: Callable | None = None,
    beta: float = 0.96,
    params: dict = {},
    borrowing_constraint: float | None = None,
    options: dict = {},
)

solve_spline_collocation(
    problem_or_domain: SplineCollocationProblem | tuple[float, float],
    backend: str = "numpy",
    **kwargs: Any,
) -> SplineCollocationSolution
```

---

### `SmolyakGrid`

```text
SmolyakGrid(
    d: int,
    mu: int,
    domain: Sequence[tuple[float, float]] | tuple[tuple[float, float], ...] | None = None,
)
```

#### Parameters:
- `d`: State-space continuous dimension $d \in [2, 6]$.
- `mu`: Smolyak approximation level $\mu \in [1, 5]$ (default 2).
- `domain`: Bounds per dimension `[(a_1, b_1), ..., (a_d, b_d)]`.

---

### `SmolyakProblem` & `solve_smolyak`

```text
SmolyakProblem(
    domain: Sequence[tuple[float, float]] | tuple[tuple[float, float], ...],
    mu: int = 2,
    method: str = "euler",
    return_fn: Callable | None = None,
    transition_fn: Callable | None = None,
    euler_residual_fn: Callable | None = None,
    beta: float = 0.96,
    params: dict = {},
    options: dict = {},
)

solve_smolyak(
    problem: SmolyakProblem | None = None,
    backend: str = "numpy",
    **kwargs: Any,
) -> SmolyakSolution
```

---

## 6. Result Interface & Manuscript Export

Both `SplineCollocationSolution` and `SmolyakSolution` implement the canonical presentation contract:

- **Attributes**:
  - `.c` or `.coefficients`: Fitted basis coefficients vector.
  - `.converged`: Boolean flag certifying solver convergence.
  - `.residual_norm`: Final sup-norm residual $\|\mathcal{R}\|_\infty$.
  - `.n_iter`: Iteration count.
- **Continuous Evaluation Methods**:
  - `.policy(s)`: Continuous decision rule evaluation at arbitrary state points.
  - `.value(s)`: Continuous value function evaluation.
  - `.euler_residual(s)`: Continuous Euler residual evaluation.
- **Reporting & Manuscript Export**:
  - `.summary()`: Textual diagnostic table.
  - `.plot()`: Multi-panel Matplotlib figure.
  - `.to_frame()`: Exports results to `pandas.DataFrame`.
  - `.to_markdown()`, `.to_latex()`, `.to_typst()`: Publication-ready tables.

---

## References

- **De Boor, C. (1978)**. *A Practical Guide to Splines*. Springer-Verlag.
- **Judd, K. L., Maliar, L., Maliar, S., & Valero, R. (2014)**. Smolyak method for solving dynamic economic models: Lagrange interpolation, anisotropic grid and adaptive domain. *Journal of Economic Dynamics and Control*, 44, 92–123.
- **Krueger, D., & Kubler, F. (2004)**. Computing equilibrium in OLG models with stochastic production and incomplete markets. *Journal of Economic Dynamics and Control*, 28(7), 1411–1436.
- **Malin, B. A., Krueger, D., & Kubler, F. (2011)**. Solving the multi-country real business cycle model using a Smolyak-style collocation method. *Journal of Economic Dynamics and Control*, 35(2), 229–239.
- **Schumaker, L. L. (1983)**. On shape preserving quadratic spline interpolation. *SIAM Journal on Numerical Analysis*, 20(4), 854–864.
- **Smolyak, S. A. (1963)**. Quadrature and interpolation formulas for tensor products of certain classes of functions. *Soviet Mathematics Doklady*, 4, 240–243.
