> 🇬🇧 English · 🇪🇸 [Español](es/vfi_continuous_equilibrium.md)

# Continuous Stationary Distributions & General Equilibrium

`puremacro.vfi.continuous_distribution` implements the non-stochastic continuous distribution simulation method developed by **Young (2010, *Journal of Economic Dynamics and Control*)** and the fast two-step operator framework of **Tan (2020, *Computational Economics*)**. It provides machine-precision mass-conserving histogram projections, Compressed Sparse Row (CSR) Markov transition operators, invariant distribution solvers, and general equilibrium market-clearing algorithms for continuous heterogeneous-agent macroeconomic models.

In macroeconomic models with uninsurable idiosyncratic earnings risk (e.g., Aiyagari 1994, Huggett 1993), households optimize over a continuous asset continuum $k \in [\underline{k}, \bar{k}]$. While Monte Carlo simulation of a finite panel of households introduces sampling chatter and slow $O(1/\sqrt{N})$ stochastic error, Young's non-stochastic method projects continuous decision rules onto a fine piecewise-linear density representation. This guarantees exact conservation of probability mass ($\sum \mu^* = 1.0 \pm 10^{-12}$) and local mean asset holdings, enabling rapid general equilibrium price determination via gradient-free root finding.

> **This page is the discrete-time route.** Time is discrete: a policy
> function $g(k, z)$ is solved first, and this module projects it onto a
> fine histogram to obtain the invariant *probability mass* $\mu^*$.
> puremacro also ships a **continuous-time** route —
> `solve_hjb_achdou`, `solve_kfe_achdou` and
> `solve_aiyagari_continuous_hjb` — where an implicit upwind scheme
> solves the Hamilton-Jacobi-Bellman equation and the adjoint
> Kolmogorov Forward Equation $A^\top g = 0$ returns a *density* $g$ in
> one linear solve, with no policy-function iteration and no lottery
> weights. The two are different discretizations of the same economics
> and are not interchangeable: $\mu^*$ here is mass per node and sums to
> 1, whereas $g$ there is a density and integrates to 1 (on a
> non-uniform grid, node mass divided by cell width, normalized with
> trapezoid quadrature weights). See
> [`docs/vfi_hjb_continuous.md`](vfi_hjb_continuous.md).

---

## 1. Theoretical & Algorithmic Framework

### 1.1 The Cross-Sectional Invariant Distribution Problem

Let idiosyncratic labor productivity $z_t \in \mathcal{Z} = \{z_1, \dots, z_{n_z}\}$ evolve according to a discrete Markov chain with transition probability matrix $\mathbf{P}_z$, where $\Pr(z_{t+1} = z_m \mid z_t = z_j) = P_z(j, m)$. Household asset holdings $k_t \in [\underline{k}, \bar{k}]$ obey a continuous optimal policy function $g(k, z): [\underline{k}, \bar{k}] \times \mathcal{Z} \to [\underline{k}, \bar{k}]$ obtained from dynamic programming (such as Chebyshev collocation, FEM Galerkin, or EGM).

The joint cross-sectional distribution of assets and productivity at date $t$ is denoted by the probability density $\mu_t(k, z)$. The distribution evolves according to the adjoint Markov operator $\mathcal{T}^*$:

$$\mu_{t+1}(A, z_m) = \sum_{j=1}^{n_z} P_z(j, m) \int_{k \in g^{-1}(A, z_j)} \mu_t(dk, z_j)$$

for any Borel subset $A \subseteq [\underline{k}, \bar{k}]$. An invariant stationary distribution $\mu^*(k, z)$ is a fixed point of this operator:

$$\mu^* = \mathcal{T}^* \mu^*, \quad \text{with} \quad \sum_{m=1}^{n_z} \int_{\underline{k}}^{\bar{k}} \mu^*(dk, z_m) = 1.0$$

### 1.2 Young (2010) Linear Lottery Mass Projection

To evaluate $\mathcal{T}^*$ numerically without Monte Carlo noise, Young (2010) discretizes the continuous asset state space into a fine histogram grid $\mathbf{k} = \{k_1, k_2, \dots, k_{N_k}\}$ with $k_1 = \underline{k}$ and $k_{N_k} = \bar{k}$, typically spaced with $N_k \in [500, 2000]$ points.

For any continuous savings choice $k' = g(k_i, z_j)$, the policy value generally does not fall precisely on a grid node. The Young method brackets $k'$ between two adjacent nodes $k_l \le k' \le k_{l+1}$ and assigns probability mass via a linear lottery:

$$\omega_l(k') = \frac{k_{l+1} - k'}{k_{l+1} - k_l}, \quad \omega_{l+1}(k') = 1 - \omega_l(k') = \frac{k' - k_l}{k_{l+1} - k_l}$$

Boundary choices $k' < k_1$ or $k' > k_{N_k}$ are clamped to the boundary nodes with weights $1.0$. This lottery formulation preserves two fundamental mathematical invariants:

1. **Strict Probability Mass Conservation**: $\omega_l(k') + \omega_{l+1}(k') = 1.0$ exactly.
2. **First-Moment Preservation**: $\omega_l(k') k_l + \omega_{l+1}(k') k_{l+1} = k'$, ensuring zero numerical bias in aggregate wealth accumulation.

### 1.3 Sparse Markov Transition Operators & Mass Conservation

Combining the asset lottery weights with the exogenous productivity transitions yields a linear operator $\mathbf{T}^*$ advancing the discrete probability vector $\boldsymbol{\mu}_t \in \mathbb{R}^{N_k \cdot n_z}$:

$$\mu_{t+1}(k_l, z_m) = \sum_{j=1}^{n_z} P_z(j, m) \sum_{i=1}^{N_k} \mathbf{1}_{\{g(k_i, z_j) \in [k_l, k_{l+1}]\}} \omega_l(g(k_i, z_j)) \mu_t(k_i, z_j)$$

Because each state $(k_i, z_j)$ distributes mass to at most two asset nodes for each destination productivity state $z_m$, the transition operator is extremely sparse. In `puremacro`, this operator is assembled as a Compressed Sparse Row (CSR) matrix with at most $2 \cdot n_z \cdot N_k$ non-zero entries (sparsity ratio $> 99.5\%$).

The invariant distribution solves the linear eigenvalue system:

$$(\mathbf{T}^* - \mathbf{I}) \boldsymbol{\mu}^* = \mathbf{0}, \quad \text{subject to} \quad \mathbf{1}^\top \boldsymbol{\mu}^* = 1.0$$

`puremacro` solves this system using four interchangeable methods:
- **`sparse_direct`**: Direct SuperLU sparse factorization replacing one redundant equation with the unit-sum constraint.
- **`power`**: The Tan (2020) accelerated two-step forward mass push $\boldsymbol{\mu}^{(n+1)} = \mathbf{T}^* \boldsymbol{\mu}^{(n)}$.
- **`arnoldi`**: Implicitly restarted Arnoldi iteration computing the leading eigenvector corresponding to unit eigenvalue $\lambda_1 = 1$.
- **`auto`**: Direct sparse factorization with automatic fallback to power iteration if matrix condition limits are exceeded.

### 1.4 General Equilibrium Market Clearing in Continuous Space

In the canonical continuous Aiyagari (1994) economy, aggregate effective labor supply is determined by the ergodic mean of the productivity process:

$$\bar{L} = \sum_{m=1}^{n_z} \pi_z(m) z_m$$

Given a candidate real interest rate $r$, competitive firm profit maximization yields aggregate capital demand $K^d(r)$ and equilibrium wages $w(r)$:

$$K^d(r) = \bar{L} \left( \frac{r + \delta}{\alpha A} \right)^{\frac{1}{\alpha - 1}}, \quad w(r) = (1 - \alpha) A \left( \frac{K^d(r)}{\bar{L}} \right)^\alpha$$

Households solve their continuous consumption-savings problem given factor prices $(r, w(r))$ to obtain policy rule $g(k, z; r)$. The invariant distribution $\mu^*(k, z; r)$ implies aggregate capital supply:

$$K^s(r) = \sum_{m=1}^{n_z} \sum_{i=1}^{N_k} k_i \mu^*(k_i, z_m; r)$$

General equilibrium is attained at the market-clearing interest rate $r^*$ satisfying:

$$\Phi(r) \equiv K^s(r) - K^d(r) = 0$$

Because $K^s(r) \to \infty$ as $r \to 1/\beta - 1$ and $K^s(r) \to 0$ as $r \to -\delta$, a unique market-clearing rate $r^* \in (-\delta, 1/\beta - 1)$ exists and is computed using 1D Brent bisection.

---

## 2. Methodological & Model Options

| Feature | `sparse_direct` | `power` (Tan 2020) | `arnoldi` | `auto` |
|---|---|---|---|---|
| **Mathematical Basis** | SuperLU $(\mathbf{I} - \mathbf{T}^*)^{-1}$ | Fixed-point iteration $\boldsymbol{\mu}^{(n+1)} = \mathbf{T}^* \boldsymbol{\mu}^{(n)}$ | Krylov subspace eigensolver | Hybrid sparse direct + power fallback |
| **Iteration Count** | 1 factorization pass | 500–5,000 iterations | 20–80 Krylov vectors | 1 pass (or iterative fallback) |
| **Mass Conservation** | Normalized to $1.0 \pm 10^{-14}$ | Machine precision at every step | Normalized upon convergence | $1.0 \pm 10^{-12}$ guaranteed |
| **Memory Footprint** | $O(N_k \cdot n_z)$ CSR nonzeros | Minimal (2 distribution buffers) | Moderate (Krylov basis storage) | Dynamic allocation |
| **Recommended Scope** | $N_k \le 2000$, standard GE | Ultra-fine grids ($N_k > 5000$) | Non-standard Markov operators | Default general equilibrium solver |

---

## 3. Canonical Calibration & Empirical Specification

The benchmark incomplete-markets economy adopts standard quarterly parameters:

| Parameter | Symbol | Benchmark Value | Economic Rationale |
|---|---|---|---|
| Subjective discount factor | $\beta$ | $0.960$ | Matches annual capital-to-output ratio $\approx 3.0$ |
| Relative risk aversion | $\gamma$ | $2.000$ | Standard empirical intertemporal elasticity of substitution $1/\gamma = 0.5$ |
| Capital output elasticity | $\alpha$ | $0.360$ | National income capital share |
| Capital depreciation rate | $\delta$ | $0.080$ | Annual physical capital depreciation rate $\approx 8\%$ |
| Productivity persistence | $\rho_z$ | $0.900$ | Persistence of idiosyncratic log earnings shocks |
| Productivity shock standard deviation | $\sigma_z$ | $0.200$ | Cross-sectional wage dispersion |
| Productivity grid points | $n_z$ | $5$ | Discretized via Tauchen or Rouwenhorst method |
| Asset histogram grid size | $N_k$ | $1000$ | High-resolution piecewise-linear density representation |
| Borrowing limit | $\underline{k}$ | $0.000$ | Natural or ad-hoc borrowing constraint |

---

## 4. Runnable Worked Examples

The following self-contained script demonstrates:
1. Verifying Young (2010) linear lottery weights and moment conservation.
2. Computing an invariant distribution $\mu^*(k)$ for a continuous policy rule.
3. Solving the full general equilibrium interest rate $r^*$ and capital stock $K^*$ in an Aiyagari economy.

```python
import numpy as np
from puremacro.vfi import (
    continuous_stationary_distribution,
    young_lottery_weights,
    solve_aiyagari_continuous,
)

# 1. Young (2010) Linear Lottery Weights Verification
k_grid = np.linspace(0.0, 10.0, 100)
policy_k = np.clip(0.85 * k_grid + 0.5, 0.0, 10.0)

j_lo, w_lo, w_hi = young_lottery_weights(policy_k, k_grid)
assert np.allclose(w_lo + w_hi, 1.0)
assert np.all(w_lo >= 0.0) and np.all(w_hi >= 0.0)

# 2. Invariant Stationary Distribution (1D AR(1) Asset Policy)
dist_1d = continuous_stationary_distribution(
    policy_k, k_grid, method="sparse_direct"
)
assert dist_1d.converged
assert dist_1d.mass_error <= 1e-12
mean_assets = dist_1d.mean()
assert 3.0 < mean_assets < 3.5

# 3. Continuous General Equilibrium (Aiyagari Economy)
ge_result = solve_aiyagari_continuous(
    beta=0.96,
    gamma=2.0,
    alpha=0.36,
    delta=0.08,
    N_k=150,
    n_z=3,
    max_evals=25,
)
assert ge_result.converged
assert abs(ge_result.capital_market_clearing_error) < 1e-4

summary_df = ge_result.summary()
print(f"Equilibrium r* = {ge_result.r:.4f}, K* = {ge_result.K:.4f}, w* = {ge_result.w:.4f}")
```

---

## 5. Full API Specification

```text
young_lottery_weights(
    kp_eval: np.ndarray,
    k_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]

continuous_push_distribution(
    pdf: np.ndarray,
    policy_fn: Any,
    asset_grid: np.ndarray,
    shock_transition: np.ndarray | None = None,
    shock_grid: np.ndarray | None = None,
) -> np.ndarray

build_continuous_transition_matrix(
    policy_fn: Any,
    asset_grid: np.ndarray,
    shock_transition: np.ndarray | None = None,
    shock_grid: np.ndarray | None = None,
) -> scipy.sparse.csr_matrix

continuous_stationary_distribution(
    policy_fn: Any,
    asset_grid: np.ndarray,
    shock_transition: np.ndarray | None = None,
    shock_grid: np.ndarray | None = None,
    *,
    method: str = "auto",
    tol: float = 1e-12,
    max_iter: int = 50000,
    backend: str = "numpy",
) -> ContinuousStationaryDistribution

solve_aiyagari_continuous(
    beta: float = 0.96,
    gamma: float = 2.0,
    alpha: float = 0.36,
    delta: float = 0.08,
    rho_z: float = 0.90,
    sigma_z: float = 0.20,
    n_z: int = 5,
    a_max: float = 30.0,
    N_k: int = 1000,
    r_bracket: tuple[float, float] | None = None,
    solver: str = "auto",
    backend: str = "numpy",
    xtol: float = 1e-6,
    max_evals: int = 60,
    **kwargs: Any,
) -> AiyagariContinuousEquilibrium
```

#### Parameters:
- `policy_fn`: Continuous policy source. Accepts a 1D/2D NumPy array, a callable `g(k)` or `g(k, z)`, or a converged solution container (`CollocationSolution`, `FEMSolution`, `SplineCollocationSolution`).
- `asset_grid`: 1D strictly increasing histogram asset grid $k_1 < k_2 < \dots < k_{N_k}$ of length $N_k \ge 2$.
- `shock_transition`: Exogenous row-stochastic Markov transition probability matrix $\mathbf{P}_z \in \mathbb{R}^{n_z \times n_z}$.
- `shock_grid`: 1D array of discrete shock values $z_1, \dots, z_{n_z}$.
- `method`: Invariant distribution solver algorithm: `"auto"`, `"sparse_direct"`, `"power"`, or `"arnoldi"`.
- `tol`: Absolute convergence tolerance for invariant distribution solvers (default $10^{-12}$).
- `r_bracket`: Tuple $(r_{\min}, r_{\max})$ bracketing the equilibrium interest rate for Brent's bisection.

---

## 6. Result Interface & Manuscript Export

Both `ContinuousStationaryDistribution` and `AiyagariContinuousEquilibrium` provide complete statistical inspection and manuscript export interfaces:

### `ContinuousStationaryDistribution` Methods

- `dist.mean() -> float`: Mean aggregate asset holdings $\mathbb{E}[k]$.
- `dist.std() -> float`: Standard deviation of asset holdings.
- `dist.percentile(q) -> float`: Value of assets at quantile $q \in [0, 100]$.
- `dist.gini() -> float`: Gini inequality coefficient of wealth.
- `dist.lorenz_curve(n_points=100) -> tuple[np.ndarray, np.ndarray]`: Population and cumulative wealth coordinates $(p, L(p))$ for Lorenz curve analysis.
- `dist.marginal_assets() -> np.ndarray`: Marginal probability density $\mu(k) = \sum_z \mu(k, z)$ over assets.
- `dist.marginal_shocks() -> np.ndarray`: Marginal probability distribution over idiosyncratic shock states.

### Manuscript Export Contract

`AiyagariContinuousEquilibrium` implements the standard `puremacro` publication export methods:
- `res.summary() -> pd.DataFrame`: Overview table of factor prices, aggregate capital stock, effective labor supply, and market clearing residuals.
- `res.to_frame() -> pd.DataFrame`: Alias returning the summary DataFrame.
- `res.to_markdown(digits=4) -> str`: GitHub-flavored Markdown table for documentation and PR reports.
- `res.to_latex(digits=4) -> str`: Publication-grade LaTeX tabular environment.
- `res.to_typst(digits=4) -> str`: Formatted Typst code block for modern scientific reports.
- `res.plot(figsize=(10, 4.5)) -> matplotlib.figure.Figure`: Multi-panel diagnostic figure visualizing the invariant density $\mu^*(k)$, Lorenz curve, and capital market supply-demand balance.

---

## References

- Aiyagari, S. R. (1994). "Uninsured Idiosyncratic Risk and Aggregate Saving." *The Quarterly Journal of Economics*, 109(3), 659–684.
- Huggett, M. (1993). "The risk-free rate in heterogeneous-agent incomplete-insurance economies." *Journal of Economic Dynamics and Control*, 17(5–6), 953–969.
- Tan, C. (2020). "A Fast and Accurate Method for Solving Incomplete Markets Models." *Computational Economics*, 55, 347–362.
- Young, E. R. (2010). "Solving the incomplete markets model with aggregate uncertainty using the Krusell-Smith algorithm and a non-stochastic simulations." *Journal of Economic Dynamics and Control*, 34(1), 36–41.
