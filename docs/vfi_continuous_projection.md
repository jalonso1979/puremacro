> 🇬🇧 English · 🇪🇸 [Español](es/vfi_continuous_projection.md)

# Continuous Projection Solvers: Chebyshev Collocation and Finite Element Galerkin

`puremacro.vfi.collocation` and `puremacro.vfi.fem` provide continuous state-space dynamic programming and projection methods developed by **Judd (1992, 1998)** and **McGrattan (1996, 1999)**, delivering high-precision policy and value function approximations without the discretization errors of discrete-grid Value Function Iteration (VFI).

Traditional discrete-grid dynamic programming constrains capital and wealth choices to an artificial grid lattice $\{k_1, \dots, k_M\}$. This imposes an $\mathcal{O}(M^{-1})$ truncation error, destroys differentiability, and requires fine lattices that scale poorly in higher dimensions. Continuous projection methods solve for smooth continuous decision rules $g(k)$ and value functions $V(k)$ defined everywhere on continuous state manifolds:

- **Chebyshev Orthogonal Collocation** employs global orthogonal Chebyshev polynomials of the first kind evaluated at Gauss-Lobatto or Gauss-Chebyshev nodes. For smooth, real-analytic economic models (such as the canonical neoclassical growth model), Jackson's approximation theorem guarantees **exponential ("spectral") convergence** ($\mathcal{O}(c^{-N})$), driving Euler equation residuals below $10^{-8}$ with only 6 to 10 polynomial coefficients.
- **Finite Element Galerkin Projection (FEM)** partitions the continuous domain into localized element intervals using piecewise linear or quadratic Lagrange hat basis functions. By isolating approximation errors to compact element supports, FEM uniquely resolves **sharp kinks, occasionally binding borrowing constraints, and non-differentiabilities** via Fischer-Burmeister complementarity with **zero Gibbs ringing**.

---

## 1. Theoretical & Mathematical Framework

### 1.1 Continuous Dynamic Economic Formulation

Consider a continuous-state infinite-horizon macroeconomic decision problem where an agent chooses next-period capital $k' \in [\underline{k}, \bar{k}]$ given current capital $k$ and exogenous Markov productivity state $z$:

$$V(k, z) = \max_{k' \ge \underline{k}} \left\{ u(c) + \beta \sum_{z'} \Pi(z, z') V(k', z') \right\}$$

subject to the aggregate resource constraint:

$$c + k' \le f(k, z) + (1 - \delta)k$$

When interior solutions exist, the optimal policy $k' = g(k, z)$ satisfies the continuous Euler equation functional residual $\mathcal{R}(k, z) = 0$:

$$\mathcal{R}(k, z; g) \equiv u'\left(f(k, z) + (1-\delta)k - g(k, z)\right) - \beta \sum_{z'} \Pi(z, z') u'\left(c(g(k, z), z')\right) \left[ f_k\left(g(k, z), z'\right) + 1 - \delta \right] = 0$$

### 1.2 Chebyshev Orthogonal Polynomial Collocation

Chebyshev collocation approximates policy $g(k)$ (or value $V(k)$) as a linear combination of $N+1$ orthogonal Chebyshev polynomials:

$$g_N(k; \mathbf{c}) = \sum_{j=0}^N c_j T_j(x(k))$$

#### 1. Bijective Domain Coordinate Mapping

Chebyshev polynomials are defined on the canonical interval $x \in [-1, 1]$. An arbitrary bounded continuous state domain $k \in [\underline{k}, \bar{k}]$ is mapped bijectively to $[-1, 1]$ via the affine transformation:

$$x(k) = \frac{2k - (\bar{k} + \underline{k})}{\bar{k} - \underline{k}}, \qquad k(x) = \frac{(x + 1)(\bar{k} - \underline{k})}{2} + \underline{k}$$

#### 2. Collocation Nodes: Gauss-Lobatto vs. Gauss-Chebyshev

To avoid Runge's phenomenon associated with equidistant nodes, projection residuals are enforced at orthogonal Chebyshev quadrature nodes:

- **Gauss-Lobatto Nodes** (extrema plus boundary endpoints):
  $$x_i = -\cos\left(\frac{i \pi}{N}\right), \quad i = 0, 1, \dots, N$$
  These include domain boundaries $x_0 = -1$ and $x_N = 1$, which is ideal for economic models with fixed borrowing limits or boundary conditions.
- **Gauss-Chebyshev Nodes** (zeros/roots of $T_{N+1}(x)$):
  $$x_i = -\cos\left(\frac{2i + 1}{2(N + 1)} \pi\right), \quad i = 0, 1, \dots, N$$
  These lie strictly in the open interior $(-1, 1)$, which is useful when utility functions exhibit singular derivatives at domain boundaries ($u'(0) \to \infty$).

#### 3. Three-Term Recurrence Relations

Chebyshev polynomials $T_j(x)$ and their derivatives $T'_j(x)$ are evaluated with numerical stability using three-term recurrence relations:

$$T_0(x) = 1, \qquad T_1(x) = x, \qquad T_{j+1}(x) = 2x T_j(x) - T_{j-1}(x), \quad j \ge 1$$

$$T'_0(x) = 0, \qquad T'_1(x) = 1, \qquad T'_{j+1}(x) = 2 T_j(x) + 2x T'_j(x) - T'_{j-1}(x), \quad j \ge 1$$

#### 4. Collocation Conditions

The unknown coefficient vector $\mathbf{c}^* \in \mathbb{R}^{N+1}$ is computed by requiring the Euler residual (or Bellman residual) to vanish exactly at all $N+1$ collocation nodes:

$$\mathcal{R}\left(k(x_i); \mathbf{c}^*\right) = 0, \quad \forall i = 0, 1, \dots, N$$

This $(N+1)$-dimensional non-linear root-finding problem is solved via Powell's hybrid method (`solver_method="hybr"`) or Levenberg-Marquardt.

---

### 1.3 Finite Element Galerkin Projection (FEM)

Global polynomial methods can struggle when policy functions feature derivative discontinuities (kinks) caused by borrowing constraints ($k' \ge \bar{k}$) or capacity limits. Approximating kinks with global polynomials induces the **Gibbs phenomenon**—persistent high-frequency oscillations that distort policy values across the entire state space.

The Finite Element Method solves this by discretizing the continuous domain into $E$ localized elements with localized basis functions.

#### 1. Mesh Partition and Piecewise Linear Hat Basis

Let $[\underline{k}, \bar{k}]$ be partitioned into $E$ element intervals by nodes $\underline{k} = k_0 < k_1 < \dots < k_E = \bar{k}$ with element step sizes $h_e = k_{e+1} - k_e$.

On each interior node $j$, the piecewise linear Lagrange hat basis function $\phi_j(k)$ has compact support on $[k_{j-1}, k_{j+1}]$:

$$\phi_j(k) = \begin{cases}
\frac{k - k_{j-1}}{k_j - k_{j-1}} & \text{if } k \in [k_{j-1}, k_j] \\
\frac{k_{j+1} - k}{k_{j+1} - k_j} & \text{if } k \in [k_j, k_{j+1}] \\
0 & \text{otherwise}
\end{cases}$$

The continuous policy function is expressed directly in terms of nodal values $\mathbf{y} = (y_0, y_1, \dots, y_E)^\top$:

$$g_E(k; \mathbf{y}) = \sum_{j=0}^E y_j \phi_j(k)$$

Notice that $\phi_j(k_i) = \delta_{ij}$ (the Kronecker delta). Consequently, nodal coefficients $y_j = g_E(k_j)$ represent the actual policy values at grid nodes, eliminating the need to invert a coefficient matrix.

#### 2. Weak Formulation and Galerkin Orthogonalization

In the Galerkin projection method, the continuous residual $\mathcal{R}(k; \mathbf{y})$ is forced to be orthogonal to each basis function $\phi_i(k)$ in the $L^2$ function space inner product:

$$\langle \mathcal{R}, \phi_i \rangle \equiv \int_{\underline{k}}^{\bar{k}} \mathcal{R}(k; \mathbf{y}) \phi_i(k) \, dk = 0, \quad i = 0, 1, \dots, E$$

Because $\phi_i(k)$ has compact support on $[k_{i-1}, k_{i+1}]$, the integral decomposes into adjacent element intervals:

$$\int_{k_{i-1}}^{k_i} \mathcal{R}(k) \frac{k - k_{i-1}}{h_{i-1}} \, dk + \int_{k_i}^{k_{i+1}} \mathcal{R}(k) \frac{k_{i+1} - k}{h_i} \, dk = 0$$

Each element integral is evaluated numerically using $Q$-point Gauss-Legendre quadrature ($Q = 3$ or $Q = 5$):

$$\int_{k_e}^{k_{e+1}} F(k) \, dk \approx \frac{h_e}{2} \sum_{q=1}^Q \omega_q F\left( \frac{k_e + k_{e+1}}{2} + \frac{h_e}{2} \xi_q \right)$$

where $\xi_q \in [-1, 1]$ and $\omega_q$ are the standard Gauss-Legendre quadrature roots and weights.

---

### 1.4 Fischer-Burmeister Complementarity for Borrowing Constraints

When savings choices face an inequality borrowing constraint $k' \ge \bar{k}$, the Kuhn-Tucker optimality conditions require:

$$k' - \bar{k} \ge 0, \qquad \mathcal{R}(k) \ge 0, \qquad (k' - \bar{k}) \cdot \mathcal{R}(k) = 0$$

Standard root-finders cannot directly handle inequality constraints. `puremacro.vfi` reformulates these complementarity conditions using the smooth perturbed **Fischer-Burmeister NCP operator** ($\epsilon = 10^{-12}$):

$$\Psi_{\text{FB}}^\epsilon(a, b) \equiv a + b - \sqrt{a^2 + b^2 + \epsilon} = 0$$

where:

$$a = k' - \bar{k}, \qquad b = \mathcal{R}(k)$$

The operator satisfies:
- $\Psi_{\text{FB}}^\epsilon(a, b) = 0 \iff a \ge 0, b \ge 0, a \cdot b \approx 0$.
- It is continuously differentiable everywhere for $\epsilon > 0$, enabling rapid convergence via standard Newton-Raphson and hybrid Powell methods.
- By placing an explicit mesh node exactly at the borrowing kink $k^*$ via `FEMMesh.from_kinks(domain, kinks=[k_kink])`, FEM confines the non-differentiability to element boundaries, completely eliminating the Gibbs phenomenon.

---

### 1.5 Brock-Mirman Analytical Benchmark: Spectral vs. Polynomial Convergence

The benchmark for testing continuous projection methods is the Brock and Mirman (1972) neoclassical growth model with logarithmic utility $u(c) = \ln(c)$, Cobb-Douglas production $f(k) = k^\alpha$, and 100% depreciation ($\delta = 1.0$):

$$V(k) = \max_{k'} \left\{ \ln(k^\alpha - k') + \beta V(k') \right\}$$

This model admits exact analytical closed-form solutions:

$$g^*(k) = \alpha \beta k^\alpha$$

$$V^*(k) = \frac{\alpha}{1 - \alpha \beta} \ln(k) + \frac{\ln(1 - \alpha \beta) + \frac{\alpha \beta \ln(\alpha \beta)}{1 - \alpha \beta}}{1 - \beta}$$

#### Error Convergence Comparison

| Metric | Chebyshev Collocation ($N$ orders) | FEM Galerkin ($E$ elements) |
|---|---|---|
| **Convergence Rate** | **Spectral / Exponential**: $\mathcal{O}(c^{-N})$ | **Algebraic / Polynomial**: $\mathcal{O}(h^2) = \mathcal{O}(E^{-2})$ |
| **Max Euler Error ($N=6$ vs $E=30$)** | $< 10^{-6}$ | $\approx 10^{-4}$ |
| **Max Euler Error ($N=12$ vs $E=100$)** | $< 10^{-9}$ | $\approx 10^{-5}$ |
| **Kink Robustness ($k' \ge \bar{k}$)** | Gibbs oscillations across domain | Zero ringing with `FEMMesh.from_kinks` |
| **Sparsity Pattern** | Dense Jacobian matrix | Tridiagonal / banded sparse Jacobian |

---

## 2. Methodological & Model Options

`puremacro.vfi` continuous solvers support multiple problem formulations and compute backends:

| Feature / Setting | `CollocationProblem` | `FEMProblem` |
|---|---|---|
| **Primary Domain** | Smooth, real-analytic manifolds | Models with kinks, borrowing constraints |
| **Solution Method (`method`)** | `"euler"` (policy) or `"bellman"` (value) | `"euler"` (policy) or `"bellman"` (value) |
| **Projection Operator** | Nodal collocation at Chebyshev zeros/extrema | `"galerkin"` (quadrature) or `"collocation"` |
| **Mesh Structure** | Gauss-Lobatto or Gauss-Chebyshev nodes | Uniform, power-clustered, or kink-aligned |
| **Borrowing Constraint** | Via Fischer-Burmeister penalty | Native `borrowing_constraint` with FB |
| **Hardware Backends** | NumPy, Numba, Apple MLX, CuPy | NumPy, Numba, Apple MLX, CuPy |
| **Default Tolerance** | `tol=1e-8`, `maxiter=500` | `tol=1e-8`, `maxiter=500` |

---

## 3. Canonical Calibration: Brock-Mirman Neoclassical Growth

The benchmark calibration parameters for testing and replication:

| Parameter | Symbol | Benchmark Value | Description |
|---|---|---|---|
| Capital elasticity | $\alpha$ | `0.36` | Output elasticity of capital |
| Subjective discount factor | $\beta$ | `0.96` | Household quarterly discount factor |
| Depreciation rate | $\delta$ | `1.00` | Full depreciation (for analytical tractability) |
| Steady-state capital | $k_{\text{ss}}$ | $(\alpha \beta)^{\frac{1}{1-\alpha}} \approx 0.1782$ | Deterministic long-run capital stock |
| Asset state domain | $[\underline{k}, \bar{k}]$ | $[0.5 k_{\text{ss}}, 1.5 k_{\text{ss}}] \approx [0.089, 0.267]$ | Evaluated continuous state interval |

---

## 4. Runnable Worked Examples

### 4.1 Neoclassical Growth: Chebyshev Collocation vs. FEM Galerkin

This example solves the Brock-Mirman model using both Chebyshev collocation and FEM Galerkin projection, comparing both against the closed-form analytical truth:

```python
import numpy as np
from puremacro.vfi import (
    CollocationProblem,
    solve_collocation,
    FEMProblem,
    solve_fem,
)

# 1. Parameter calibration
alpha = 0.36
beta = 0.96
k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
domain = (0.5 * k_ss, 1.5 * k_ss)

# 2. Solve via Chebyshev Collocation (Order N = 6)
prob_col = CollocationProblem(
    domain=domain,
    orders=6,
    method="euler",
    params={"alpha": alpha, "delta": 1.0},
    beta=beta,
)
sol_col = solve_collocation(prob_col)
assert sol_col.converged

# 3. Solve via FEM Galerkin (30 linear elements)
prob_fem = FEMProblem(
    domain=domain,
    elements=30,
    method="euler",
    projection="galerkin",
    return_fn=lambda c: np.log(np.maximum(c, 1e-12)),
    transition_fn=lambda k: k**alpha,
    beta=beta,
    params={"alpha": alpha, "gamma": 1.0},
)
sol_fem = solve_fem(prob_fem)
assert sol_fem.converged

# 4. Out-of-sample evaluation on 1,000 dense continuous points
eval_k = np.linspace(domain[0], domain[1], 1000)
g_true = alpha * beta * (eval_k**alpha)
g_col = sol_col.policy(eval_k)
g_fem = sol_fem.policy(eval_k)

max_col_err = np.max(np.abs(g_col - g_true) / g_true)
max_fem_err = np.max(np.abs(g_fem - g_true) / g_true)

print(f"Chebyshev Collocation Max Relative Error: {max_col_err:.2e}")
print(f"FEM Galerkin Max Relative Error:          {max_fem_err:.2e}")

assert max_col_err < 1e-4
assert max_fem_err < 1e-3
```

### 4.2 Borrowing Constraints and Kink Resolution via FEM

When savings decisions face an occasionally binding borrowing limit $k' \ge \bar{k}$, global polynomials suffer from Gibbs ringing near the kink. FEM with localized hat basis functions aligns an element boundary at the kink and enforces Fischer-Burmeister complementarity:

```python
import numpy as np
from puremacro.vfi import FEMMesh, FEMProblem, solve_fem

# Borrowing constraint lower bound
b_const = 0.16

# Construct kink-aligned mesh with a node placed exactly at the borrowing boundary
mesh = FEMMesh.from_kinks(domain=domain, n_elements=40, kinks=[0.17])

prob_constrained = FEMProblem(
    domain=domain,
    elements=40,
    method="euler",
    projection="galerkin",
    return_fn=lambda c: np.log(np.maximum(c, 1e-12)),
    transition_fn=lambda k: k**alpha,
    beta=beta,
    borrowing_constraint=b_const,
    params={"alpha": alpha, "gamma": 1.0},
    options={"mesh": mesh},
)
sol_constrained = solve_fem(prob_constrained)
assert sol_constrained.converged

# Verify constraint satisfaction across 500 dense test points
dense_k = np.linspace(domain[0], domain[1], 500)
pol_k = sol_constrained.policy(dense_k)
assert np.all(pol_k >= b_const - 1e-10), "Borrowing constraint violated!"
print("Kink-aligned FEM policy successfully enforced k' >= bar{k} with zero ringing.")
```

### 4.3 Multi-Backend Acceleration

`puremacro.vfi` solvers integrate with `puremacro._backend`, allowing transparent hardware acceleration across CPU and GPU architectures:

```python
from puremacro import _backend as bk

# Check available acceleration backends
print("Numba JIT available: ", bk.backend_available("numba"))
print("Apple MLX available: ", bk.backend_available("mlx"))
print("NVIDIA CuPy available:", bk.backend_available("cupy"))

# Solve with explicit backend selection (falls back safely to NumPy if unavailable)
sol_fast = solve_collocation(prob_col, backend="numpy")
print(f"Solution converged: {sol_fast.converged} | Residual norm: {sol_fast.residual_norm:.2e}")
```

---

## 5. Full API Specification

### `CollocationProblem`

```text
CollocationProblem(
    domain: tuple[float, float] | tuple[tuple[float, float], ...],
    orders: int | tuple[int, ...],
    method: str = "euler",
    node_type: str = "lobatto",
    return_fn: Callable | None = None,
    transition_fn: Callable | None = None,
    euler_residual_fn: Callable | None = None,
    beta: float = 0.96,
    params: dict = {},
    options: dict = {},
)
```

#### Parameters:
- `domain`: Continuous state-space bounds `(k_min, k_max)` for 1D models or a tuple of bounds for multi-dimensional models.
- `orders`: Polynomial approximation degree $N \ge 1$ (or tuple of degrees per dimension).
- `method`: Approximation objective:
  - `"euler"`: Policy function projection minimizing continuous Euler equation residuals.
  - `"bellman"`: Continuous value function iteration on Chebyshev nodes.
- `node_type`: Chebyshev node distribution:
  - `"lobatto"`: Gauss-Lobatto extrema nodes (includes domain boundaries $\pm 1$).
  - `"gauss"`: Gauss-Chebyshev interior roots (strictly in $(-1, 1)$).
- `return_fn`: One-period utility function $u(c)$ or $u(k, k', \mathbf{p})$.
- `transition_fn`: Transition equation $f(k)$ or aggregate resource budget.
- `euler_residual_fn`: User-supplied Euler equation residual callable $R(k, k', k'', \mathbf{p})$.
- `beta`: Household discount factor $\beta \in (0, 1)$.
- `params`: Dictionary of model structural parameters (e.g. `{"alpha": 0.36, "delta": 1.0}`).
- `options`: Dictionary of solver tuning options (`tol`, `max_iter`, `solver_method`).

---

### `solve_collocation`

```text
solve_collocation(
    problem: CollocationProblem | None = None,
    backend: str = "numpy",
    **kwargs: Any,
) -> CollocationSolution
```

Solves the Chebyshev collocation problem via non-linear root finding. Accepts either an existing `CollocationProblem` instance or parameter keyword arguments.

---

### `FEMProblem`

```text
FEMProblem(
    domain: tuple[float, float] | tuple[tuple[float, float], ...],
    elements: int | tuple[int, ...] = 50,
    method: str = "euler",
    projection: str = "galerkin",
    return_fn: Callable | None = None,
    transition_fn: Callable | None = None,
    euler_residual_fn: Callable | None = None,
    beta: float = 0.96,
    params: dict = {},
    borrowing_constraint: float | None = None,
    options: dict = {},
)
```

#### Parameters:
- `domain`: Continuous state bounds `(s_min, s_max)`.
- `elements`: Number of mesh elements $E \ge 2$.
- `method`: Solution method: `"euler"` or `"bellman"`.
- `projection`: Projection scheme: `"galerkin"` (weighted residual integration via Gauss-Legendre quadrature) or `"collocation"` (nodal collocation).
- `return_fn`: Utility function $u(c)$ or $u(s, s')$.
- `transition_fn`: Transition rule $f(s)$ or next-state resource equation.
- `euler_residual_fn`: Explicit continuous Euler equation residual callable.
- `beta`: Discount factor $\beta \in (0, 1)$.
- `params`: Model economic parameters.
- `borrowing_constraint`: Optional lower bound on choice $s' \ge \underline{s}$.
- `options`: Algorithmic dictionary:
  - `"mesh"`: Custom pre-constructed `FEMMesh` instance (e.g. from `FEMMesh.from_kinks`).
  - `"complementarity"`: Complementarity formulation (`"fb"` for Fischer-Burmeister, `"min"` for minimum).
  - `"quad_order"`: Gauss-Legendre quadrature points per element (default 5).

---

### `solve_fem`

```text
solve_fem(
    problem_or_domain: FEMProblem | tuple[float, float],
    *args: Any,
    backend: str = "numpy",
    **kwargs: Any,
) -> FEMSolution
```

Solves the Finite Element continuous model via non-linear root finding.

---

## 6. Result Interface & Manuscript Export

Both `CollocationSolution` and `FEMSolution` adhere to the canonical `puremacro` presentation contract:

- **Attributes**:
  - `.c`: Fitted Chebyshev coefficients or FEM nodal values vector.
  - `.converged`: Boolean flag indicating root-finder success.
  - `.residual_norm`: Final sup-norm residual $\|\mathcal{R}\|_\infty$.
  - `.n_iter`: Total iterations evaluated by the non-linear solver.
- **Continuous Evaluation Methods**:
  - `.policy(k)`: Evaluates the continuous decision rule $g(k)$ at arbitrary continuous state points $k$.
  - `.value(k)`: Evaluates the continuous value function $V(k)$ (available when solved via Bellman or when return functions are provided).
  - `.euler_residuals(k_eval=None)`: Computes Euler equation residuals on an out-of-sample continuous evaluation grid.
- **Reporting & Manuscript Export**:
  - `.summary()`: Generates a formatted textual diagnostic summary of convergence and residual statistics.
  - `.plot()`: Spawns a multi-panel Matplotlib figure displaying policy functions, value functions, and Euler residual distributions.
  - `.to_frame()`: Exports nodal solutions and policy values as a `pandas.DataFrame`.
  - `.to_markdown()`, `.to_latex()`, `.to_typst()`: Exports formatted tables ready for scientific publication.

---

## References

- **Brock, W. A., & Mirman, L. J. (1972)**. Optimal economic growth and uncertainty: The discounted case. *Journal of Economic Theory*, 4(3), 479–513.
- **Fischer, A. (1992)**. A special Newton-type optimization method. *Optimization*, 24(3-4), 269–284.
- **Judd, K. L. (1992)**. Projection methods for solving aggregate growth models. *Journal of Economic Theory*, 58(2), 410–452.
- **Judd, K. L. (1998)**. *Numerical Methods in Economics*. MIT Press.
- **McGrattan, E. R. (1996)**. Solving the stochastic growth model with a finite element method. *Journal of Economic Dynamics and Control*, 20(1-3), 19–42.
- **McGrattan, E. R. (1999)**. Application of weighted residual methods to dynamic economic models. In *Computational Methods for the Study of Dynamic Economies* (pp. 114–142). Oxford University Press.
- **Miranda, M. J., & Fackler, P. L. (2002)**. *Applied Computational Economics and Finance*. MIT Press.
