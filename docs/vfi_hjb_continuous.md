> 🇬🇧 English · 🇪🇸 [Español](es/vfi_hjb_continuous.md)

# Continuous-Time HJB, Adjoint KFE & Continuous Aiyagari GE

`puremacro.vfi.hjb_achdou` (re-exported from `puremacro.vfi`) implements the continuous-time heterogeneous-agent toolkit of **Achdou, Han, Lasry, Lions & Moll (2022, *Review of Economic Studies*)**: an implicit upwind finite-difference solver for the stationary Hamilton-Jacobi-Bellman (HJB) equation of the income-fluctuation problem (`solve_hjb_achdou`), the adjoint Kolmogorov Forward Equation (KFE) that delivers the stationary wealth distribution without simulation (`solve_kfe_achdou`), and a general equilibrium root-finder for the continuous-time **Aiyagari (1994)** / **Huggett (1993)** economy (`solve_aiyagari_continuous_hjb`). Results come back as the frozen dataclasses `HJBSolution` and `AiyagariContinuousHJBResult`, which keep dictionary-style access for code written against the pre-3.4.0 interface.

In continuous time the household problem is a first-order partial differential equation rather than a Bellman operator over a grid of next-period choices. Discretizing it with an upwind scheme turns each policy-improvement step into one sparse linear solve, and the same infinitesimal generator $\mathbf{A}$ that pushes the value function backward in time pushes the cross-sectional distribution forward. The stationary density is therefore a null vector of $\mathbf{A}^\top$, obtained to machine precision with no Monte Carlo chatter. This page is the continuous-time companion of [`vfi_continuous_equilibrium.md`](vfi_continuous_equilibrium.md), which covers the discrete-time Young (2010) non-stochastic simulation and `solve_aiyagari_continuous`; Section 2.2 spells out how the two relate and where the notebooks fit.

---

## 1. Theoretical & Algorithmic Framework

### 1.1 The Stationary HJB Equation

A household discounts at rate $\rho > 0$, has CRRA utility $u(c) = c^{1-\gamma}/(1-\gamma)$ (with $u(c) = \log c$ when $\gamma = 1$), holds assets $a \in [a_{\min}, a_{\max}]$ earning the interest rate $r$, and receives labor income $w e_j$ where the productivity state $e_j \in \{e_1, \dots, e_{N_e}\}$ follows a continuous-time Markov chain with generator $\boldsymbol{\Lambda} = (\lambda_{jk})$: $\lambda_{jk} \ge 0$ for $k \ne j$ and every row sums to zero. The value function $v_j(a) = v(a, e_j)$ solves

$$\rho\, v_j(a) = \max_{c \ge 0} \Big\{ u(c) + v_j'(a)\, s_j(a) \Big\} + \sum_{k \ne j} \lambda_{jk} \big[ v_k(a) - v_j(a) \big], \qquad s_j(a) = r a + w e_j - c .$$

The first-order condition gives the consumption rule $c_j(a) = \big(v_j'(a)\big)^{-1/\gamma}$ and the savings drift $s_j(a) = r a + w e_j - c_j(a)$. The borrowing limit $a \ge a_{\min}$ enters as a *state-constraint boundary condition* (Achdou et al. 2022): the drift may not point out of the state space at $a_{\min}$, which is equivalent to $v_j'(a_{\min}) \ge u'(r a_{\min} + w e_j)$. The mirror condition $v_j'(a_{\max}) \le u'(r a_{\max} + w e_j)$ (no outward drift at the top) is imposed at the artificial upper boundary $a_{\max}$.

### 1.2 Upwind Finite Differences and the State-Constraint Boundary

On an asset grid $a_1 < a_2 < \dots < a_{N_a}$ with local spacings $\Delta a_i^{+} = a_{i+1} - a_i$ and $\Delta a_i^{-} = a_i - a_{i-1}$, the derivative is approximated one-sidedly in the direction the state is moving ("upwind"):

$$v'_{i,j,F} = \frac{v_{i+1,j} - v_{i,j}}{\Delta a_i^{+}}, \qquad v'_{i,j,B} = \frac{v_{i,j} - v_{i-1,j}}{\Delta a_i^{-}} .$$

Each candidate derivative implies a consumption level $c = (v')^{-1/\gamma}$ and a drift $s_F$ or $s_B$. The scheme uses the forward difference where $s_{F} > 0$, the backward difference where $s_{B} < 0$, and otherwise sets the drift to zero with $c_{i,j} = r a_i + w e_j$ (the household consumes its flow income and stays put). At the two ends of the grid the missing one-sided derivative is replaced by $u'(r a_i + w e_j)$, which makes the corresponding drift exactly zero, so no probability flux ever leaves $[a_{\min}, a_{\max}]$. `solve_hjb_achdou` additionally masks the forward branch at $a_{\max}$ and the backward branch at $a_{\min}$, so the boundary rows can only carry a drift that points inward or vanishes. This is why, in Example 4.1, the low-income household at $a = 0$ has $c = 0.2 = w e_1$ and $s = 0$ exactly.

Two departures from the state-constraint treatment are available:

- **Prescribed boundary derivatives** (`v_prime_boundary=(v'_{\min}, v'_{\max})`): Neumann conditions. The masks are switched off, the given derivative is used on the boundary row, and any outward flux $s \cdot v'_{\text{boundary}}$ is moved to the right-hand side of the linear system as an inhomogeneous term. An entry left as `None` falls back to the state-constraint derivative on that side.
- **Cake-eating benchmark** (`w_rate=0.0`): with no labor income the problem has the closed form $c(a) = \mu a$, $\mu = (\rho - (1-\gamma) r)/\gamma$. The solver then initializes $v$ from that solution, switches the boundary masks off (as in the Neumann case) and uses the analytical marginal utilities $(\mu a_{\min})^{-\gamma}$ and $(\mu a_{\max})^{-\gamma}$ on the boundary rows, which is how `tests/test_hjb_implicit.py` validates the scheme against an exact solution. Because $(\mu a)^{-\gamma}$ is singular at the origin, this mode requires a grid that starts strictly above zero: `w_rate=0.0` with `a_grid[0] <= 0` raises `ValueError` telling you to pass `a_min > 0`.

### 1.3 The Implicit Scheme and the M-Matrix Property

Collecting the upwinded drifts into the sparse infinitesimal generator $\mathbf{A}^n$ (an $N_a N_e \times N_a N_e$ matrix indexed in column-major order, state $(i, j) \mapsto j N_a + i$), each row of the asset block has entries

$$x_{i,j} = \frac{\max(-s_{i,j}, 0)}{\Delta a_i^{-}}, \qquad z_{i,j} = \frac{\max(s_{i,j}, 0)}{\Delta a_i^{+}}, \qquad y_{i,j} = -x_{i,j} - z_{i,j},$$

placed on the sub-diagonal, super-diagonal and diagonal respectively, and the income process contributes $\boldsymbol{\Lambda} \otimes \mathbf{I}_{N_a}$. Off-diagonal entries are non-negative and every row of $\mathbf{A}^n$ sums to zero (the row sums of the generator returned in Example 4.1 are zero to $2.2 \times 10^{-16}$). Rather than stepping the value function explicitly, which is subject to a Courant-Friedrichs-Lewy restriction $\Delta t = O(\Delta a)$, the update is implicit in $v^{n+1}$:

$$\Big[ \Big(\rho + \frac{1}{\Delta}\Big) \mathbf{I} - \mathbf{A}^n \Big] v^{n+1} = u(c^n) + \frac{1}{\Delta} v^n .$$

With prescribed boundary derivatives the outward-flux term of Section 1.2 is added to the right-hand side; under the state constraint it is identically zero. The matrix $\mathbf{B}^n = (\rho + 1/\Delta)\mathbf{I} - \mathbf{A}^n$ has a positive diagonal, non-positive off-diagonals and row sums equal to $\rho + 1/\Delta > 0$, so it is strictly diagonally dominant and hence a non-singular **M-matrix** with $(\mathbf{B}^n)^{-1} \ge 0$ elementwise. A non-negative inverse makes the discrete scheme *monotone*, and the Barles-Souganidis (1991) theorem then guarantees that a monotone, stable and consistent scheme converges to the viscosity solution of the HJB equation as the grid is refined. `solve_hjb_achdou` factorizes $\mathbf{B}^n$ with `scipy.sparse.linalg.spsolve` (CSC storage) on every iteration and stops when $\| v^{n+1} - v^n \|_\infty <$ `tol`.

**The role of $\Delta$.** As $\Delta \to \infty$ the update becomes Howard policy iteration, i.e. Newton's method on the HJB equation, which is why the default `Delta=1e4` converges in a handful of iterations; a small $\Delta$ turns the same system into a damped pseudo-time march that needs many more sweeps. Measured with the defaults (`Na=100`, `tol=1e-8`, `max_iter=100`; Example 4.5):

| `Delta` | Iterations | Converged | $\max \lvert v - v_{\Delta = 10^4} \rvert$ |
|---|---|---|---|
| $10^4$ | 9 | yes | reference |
| $10^2$ | 14 | yes | $9.8 \times 10^{-10}$ |
| $10$ | 50 | yes | $1.8 \times 10^{-8}$ |
| $1$ | 100 | **no** (needs 360 with `max_iter=1000`) | $6.7 \times 10^{-2}$ at the cap |

The scheme is *unconditionally stable* for every $\Delta > 0$; the iteration *count*, not the stability, depends on $\Delta$.

### 1.4 The Adjoint KFE and Mass Normalization

The stationary joint distribution $g(a, e)$ of the controlled process is the null vector of the adjoint generator,

$$\mathbf{A}^\top g = 0, \qquad \sum_{i=1}^{N_a} \sum_{j=1}^{N_e} g_{i,j}\, w_i = 1, \qquad g_{i,j} \ge 0,$$

where the $w_i$ are the cell-width quadrature weights of the asset grid,

$$w_i = \tfrac{1}{2}\big(\Delta a_{i-1} + \Delta a_i\big), \qquad \Delta a_{-1} \equiv \Delta a_0, \quad \Delta a_{N_a} \equiv \Delta a_{N_a - 1},$$

so that $w_i = \Delta a$ at every node of a uniform grid. The generator acts on grid *nodes*, so the null vector of $\mathbf{A}^\top$ is the vector of stationary node masses $\pi_{i,j}$, not a density. `solve_kfe_achdou` therefore takes the generator from the final HJB iteration, replaces the first row of $\mathbf{A}^\top$ with a row of ones and the first right-hand-side entry with $1$ (the standard replacement of one redundant equation by the normalization $\sum \pi = 1$), solves with `spsolve`, clips any negative round-off to zero, rescales $\pi$ to sum to one exactly, and only then converts to a **density** by dividing by the local cell width,

$$g_{i,j} = \pi_{i,j} / w_i .$$

This makes $\sum_{i,j} g_{i,j} w_i = 1$ and $\sum_i g_{i,j} w_i = \pi_j$ (the stationary distribution of the income generator) on any grid, uniform or not, and it makes aggregates of the form $\sum_{i,j} a_i g_{i,j} w_i$ the correct quadrature approximation of $\int a\, g(a, e)\, da$ — the same weights `solve_aiyagari_continuous_hjb` uses for capital supply. On a uniform grid the two views coincide: $w_i = \Delta a$, the probability mass at node $i$ is $g_{i,j} \Delta a$, and $\sum_{i,j} g_{i,j} \Delta a = 1$ (Example 4.2 prints `1.000000000000`).

The reported `mass_residual` is $\lvert \sum_{i,j} g_{i,j} w_i - 1 \rvert$ measured *after* that normalization, so it is of order $10^{-16}$ by construction (`1.11e-16` in Example 4.1); it certifies the normalization, not the accuracy of the linear solve. Before solving, `solve_kfe_achdou` also checks that the generator's transition graph has exactly one closed communicating class, which is the exact condition for the null space to be one-dimensional, and raises `ValueError` otherwise — a reducible income generator such as `A_z=np.zeros((2, 2))` is rejected with a message naming the number of closed classes rather than returning an arbitrary null vector.

### 1.5 General Equilibrium: Root-Finding for $r$ with the Firm First-Order Condition

Firms produce $Y = K^\alpha L^{1-\alpha}$ and rent capital at $r + \delta$, so given a trial interest rate the capital-labor ratio and the wage follow from the firm's first-order conditions:

$$\frac{K^d(r)}{L} = \Big( \frac{\alpha}{r + \delta} \Big)^{\frac{1}{1-\alpha}}, \qquad w(r) = (1-\alpha) \Big( \frac{K^d(r)}{L} \Big)^{\alpha}, \qquad L = \sum_{j} e_j\, \pi_j ,$$

where $\boldsymbol{\pi}$ is the stationary distribution of the income generator ($\boldsymbol{\pi}^\top \boldsymbol{\Lambda} = 0$; with the default symmetric two-state process $\pi = (0.5, 0.5)$ and $L = 0.6$). For each trial $r$, `solve_aiyagari_continuous_hjb` computes $w(r)$, calls `solve_hjb_achdou` (with the KFE), aggregates household wealth $K^s(r) = \sum_{i,j} a_i\, g_{i,j}(r)\, w_i$ with the quadrature weights of Section 1.4 and evaluates the excess supply $\Phi(r) = K^s(r) - K^d(r)$. Brent's method (`scipy.optimize.root_scalar`, `method="brentq"`, at most `max_iter_ge` iterations per pass) locates $r^\ast$ with $\Phi(r^\ast) = 0$ on the bracket $[r_{\min}, r_{\max}]$, whose default upper end is $\rho - 0.002$: with a borrowing constraint the stationary distribution only exists for $r < \rho$, because at $r \ge \rho$ households would accumulate without bound. Before calling Brent the solver checks the sign of $\Phi$ at both ends; if $\Phi(r_{\min}) > 0$ it halves $r_{\min}$ (floor $0.001$, at most five times) and if $\Phi(r_{\max}) < 0$ it moves $r_{\max}$ halfway toward $\rho$ (at most five times). An invalid bracket ($r_{\min} \ge r_{\max}$, $r_{\max} \ge \rho$ or $r_{\min} \le -\delta$) raises `ValueError`. Should no sign change be found, it returns the cached trial rate with the smallest $\lvert \Phi \rvert$. Each HJB solution is cached by $r$, so the run in Example 4.3 needed seven household solves for six Brent iterations.

**`tol_ge` governs the market-clearing residual.** It is not a spectator argument: the $x$-tolerance handed to Brent is derived from `tol_ge` and the secant slope of the bracket, `xtol = min(1e-5, max(0.5 * tol_ge / slope, 1e-12))`, and after the first pass the solver re-brackets from its cached evaluations and re-runs Brent — up to three further passes — until $\lvert K^s(r^\ast) - K^d(r^\ast) \rvert <$ `tol_ge`. The returned `converged` is `True` only when Brent converged **and** that inequality holds, so it means what it says: the capital market clears to `tol_ge`. Tightening the tolerance tightens the residual — with the default calibration `tol_ge=1e-4` stops at $\Phi = -1.76 \times 10^{-6}$ after 6 Brent iterations, while `tol_ge=1e-8` reaches $\Phi = -7.4 \times 10^{-13}$ in 7.

---

## 2. Methodological & Model Options

### 2.1 Options Table

| Feature | Argument(s) | Default | Notes |
|---|---|---|---|
| **Income process** | `e_grid`, `A_z` | `[0.2, 1.0]`, symmetric Poisson switching at intensity $0.1$ | Any $N_e$. `A_z` is validated: shape $(N_e, N_e)$, finite, non-negative off-diagonals, zero row sums (`ValueError` otherwise, with a hint if you passed a transition-probability matrix). With `A_z=None` and $N_e > 2$ the default is $\lambda_{jk} = 0.2/(N_e-1)$ off the diagonal and $-0.2$ on it. |
| **Asset grid** | `Na`, `a_min`, `a_max`, `a_grid` | 100 nodes on $[0, 30]$, uniform | `a_grid` overrides the three scalars and may be non-uniform; the KFE returns a density on either (Section 1.4). Must be strictly increasing with at least two points. |
| **Boundary treatment** | `v_prime_boundary`, `w_rate` | state constraint at both ends | Tuple of derivatives switches to Neumann; `w_rate=0.0` selects the analytical cake-eating boundaries and then requires `a_grid[0] > 0`. |
| **Implicit step** | `Delta` | $10^4$ | Larger is closer to policy iteration (fewer sweeps); see the table in Section 1.3. |
| **Stopping rule** | `max_iter`, `tol` | 100, $10^{-8}$ | Sup-norm on successive value functions; `max_iter` must be an integer $\ge 1$. In 3.3.0 the defaults were 500 and $10^{-7}$. |
| **Distribution** | `compute_kfe` | `True` | `False` leaves `g_dist=None` and `mass_residual=0.0`; `A_generator` is still returned, so `solve_kfe_achdou` can be called later. |
| **GE bracket** | `r_min`, `r_max`, `tol_ge`, `max_iter_ge` | $0.005$, $\rho - 0.002$, $10^{-4}$, 40 | `tol_ge` is the market-clearing tolerance: it sets Brent's $x$-tolerance and `converged` means $\lvert K^s - K^d \rvert <$ `tol_ge` (Section 1.5). |
| **Household solve inside GE** | `max_iter_hjb`, `tol_hjb`, `Delta` | 100, $10^{-8}$, $10^4$ | Forwarded to `solve_hjb_achdou` at every trial rate. |

### 2.2 Relation to the Discrete-Time Pages and to the Notebooks

- [`vfi_continuous_equilibrium.md`](vfi_continuous_equilibrium.md) solves the same Aiyagari economy in **discrete time**: households discount with $\beta = 0.96$, the continuous savings policy is projected on a histogram with Young (2010) linear lotteries, and `solve_aiyagari_continuous` clears the capital market with Brent. That module also ships `ContinuousStationaryDistribution.gini()` and `.lorenz()`. The page you are reading replaces the lottery projection with the adjoint KFE and the Bellman operator with the HJB generator; there is no distribution helper class here, so inequality statistics are computed directly from `g_dist` as in Example 4.2. Calibrations differ ($\rho = 0.05$ versus $\beta = 0.96$, two Poisson income states versus a Tauchen or Rouwenhorst chain), so the equilibrium numbers are not comparable across the two pages.
- `notebooks/22_continuous_time_hjb.py` (and its `_es` twin) introduces the upwind scheme on the income-fluctuation problem. It was written against the 3.3.0 interface: it reads the result with `results["V"]` and passes `max_iter=500, tol=1e-7` explicitly, which are the old defaults. Both still work because `HJBSolution` implements the `Mapping` protocol (Section 6). Its *numbers*, however, are not the 3.3.0 ones: the 3.3.0 explicit solver had no income-switching term, so each productivity state was a separate deterministic-income problem, whereas since 3.4.0 the states are linked by the Poisson generator `A_z` (default symmetric $\lambda = 0.1$). Passing `A_z=np.zeros((Ne, Ne))` together with `compute_kfe=False` reproduces the 3.3.0 economics — the KFE has to be switched off because a generator without switching has no unique stationary distribution (Section 1.4).
- `notebooks/56_implicit_hjb_and_continuous_kfe.py` (English and Spanish) is the showcase for this module: implicit HJB iteration, KFE wealth distribution with Lorenz curve and Gini coefficient, the cake-eating benchmark against the closed form, and the continuous Aiyagari general equilibrium with supply-demand curves. The catalog entry is in [`notebooks.md`](notebooks.md).

---

## 3. Canonical Calibration

The defaults set up a two-state income-fluctuation problem in the spirit of Achdou et al. (2022) with annual-frequency parameters:

| Parameter | Symbol | Default | Argument |
|---|---|---|---|
| Discount rate | $\rho$ | $0.05$ | `rho_val` |
| Relative risk aversion | $\gamma$ | $2.0$ | `gamma_r` |
| Interest rate (partial equilibrium) | $r$ | $0.03$ | `r_rate` |
| Wage (partial equilibrium) | $w$ | $1.0$ | `w_rate` |
| Productivity states | $e$ | $\{0.2, 1.0\}$ | `e_grid` |
| Switching intensities | $\lambda_{12} = \lambda_{21}$ | $0.1$ | `A_z` |
| Asset grid | $[a_{\min}, a_{\max}]$, $N_a$ | $[0, 30]$, 100 uniform nodes | `a_min`, `a_max`, `Na` |
| Implicit step | $\Delta$ | $10^4$ | `Delta` |
| Capital share (GE) | $\alpha$ | $0.33$ | `alpha` |
| Depreciation (GE) | $\delta$ | $0.05$ | `delta` |
| Interest-rate bracket (GE) | $[r_{\min}, r_{\max}]$ | $[0.005, 0.048]$ | `r_min`, `r_max` |

---

## 4. Runnable Worked Examples

All examples are deterministic (there is no random number generation anywhere in the module), need no network access, and finish in well under a tenth of a second of solver time on a laptop; the numbers quoted below are the ones each script printed.

### 4.1 Partial-Equilibrium Solve and `summary()`

```python
import numpy as np
from puremacro.vfi import solve_hjb_achdou

sol = solve_hjb_achdou(r_rate=0.03, w_rate=1.0, rho_val=0.05, gamma_r=2.0, Na=100, a_max=30.0)
print(sol.n_iter, sol.converged, f"{sol.mass_residual:.1e}")
print(sol.V.shape, sol.c_policy.shape, sol.g_dist.shape, sol.A_generator.shape)
print(sol.summary())

# Legacy dict-style access still works (12 keys); the newer fields are attributes.
print(sol["n_iter"] == sol.n_iter, list(sol.keys())[:4], "g_dist" in sol, "g_dist" in sol.keys())
print(f"c(0, e_low) = {sol.c_policy[0, 0]:.4f}   c(0, e_high) = {sol.c_policy[0, 1]:.4f}")
print(f"s(0, e_low) = {sol.s_drift[0, 0]:.4f}   s(0, e_high) = {sol.s_drift[0, 1]:.4f}")
```

The solver converges in 9 iterations to `tol=1e-8` with a mass residual of `1.1e-16`; the arrays are `(100, 2)` and the generator is a `(200, 200)` sparse matrix. `summary()` returns a `DataFrame` indexed by metric: 200 states, elapsed time 0.0144 s (machine-dependent), mean value $-22.000181$, consumption between $0.200000$ and $1.972175$. The last two lines show the state constraint at work: the low-income household at $a = 0$ consumes exactly its labor income $0.2000$ with zero drift, while the high-income household consumes $0.4896$ and saves at rate $0.5104$. `"g_dist" in sol` is `True` but `"g_dist" in sol.keys()` is `False`, because `keys()` lists only the twelve legacy keys while membership and subscripting cover every dataclass field (Section 6).

### 4.2 Lorenz Curve and Gini Coefficient from `g_dist`

```python
import numpy as np
from puremacro.vfi import solve_hjb_achdou

sol = solve_hjb_achdou(r_rate=0.03, w_rate=1.0, rho_val=0.05, gamma_r=2.0, Na=100, a_max=30.0)
a, g = sol.a_grid, sol.g_dist
da = a[1] - a[0]                      # uniform grid: node mass = g * da
mass_a = g.sum(axis=1) * da           # marginal wealth distribution, sums to one
mean_a = float(np.sum(a * mass_a))

# Lorenz curve: cumulative population share vs cumulative wealth share
pop = np.concatenate([[0.0], np.cumsum(mass_a)])
wealth = np.concatenate([[0.0], np.cumsum(a * mass_a) / mean_a])
gini = float(1.0 - np.sum((wealth[1:] + wealth[:-1]) * np.diff(pop)))

top10 = 1.0 - float(np.interp(0.90, pop, wealth))
print(f"mass total = {mass_a.sum():.12f}")
print(f"mean wealth (capital supply) = {mean_a:.4f}")
print(f"share of households at a = 0 : {mass_a[0]:.4f}")
print(f"Gini coefficient of wealth   : {gini:.4f}")
print(f"wealth share of the top 10%  : {top10:.4f}")
print(f"stationary income shares     : {g.sum(axis=0) * da}")
```

The node masses sum to `1.000000000000`. At $r = 0.03$ and $w = 1$ mean wealth (the partial-equilibrium capital supply) is $6.2150$, $3.82\%$ of households sit exactly at the borrowing limit, the wealth Gini coefficient is $0.3611$ and the top decile holds $21.99\%$ of wealth. The income marginal is `[0.5 0.5]`, the stationary distribution of the symmetric two-state chain. The Gini formula is the trapezoid area between the Lorenz curve $(p, L(p))$ and the diagonal, the same expression notebook 56 uses.

### 4.3 General Equilibrium: $r^\ast$, $w^\ast$, $K^\ast$

```python
from puremacro.vfi import solve_aiyagari_continuous_hjb

ge = solve_aiyagari_continuous_hjb(alpha=0.33, delta=0.05, rho_val=0.05, gamma_r=2.0, Na=100, a_max=30.0)
print(f"r* = {ge.r_star:.5f}  w* = {ge.w_star:.4f}  K* = {ge.K_star:.4f}  L* = {ge.L_star:.3f}  Y* = {ge.Y_star:.4f}")
print(f"excess capital = {ge.excess_capital:.2e}  converged = {ge.converged}  Brent iterations = {ge.n_iter_ge}")
print(f"K/Y = {ge.K_star / ge.Y_star:.3f}   household HJB iterations at r*: {ge.hjb_solution.n_iter}")
print(ge.to_markdown())
```

The market clears at $r^\ast = 0.02004$, $w^\ast = 1.4376$, $K^\ast = 6.0656$ with $L^\ast = 0.600$ and $Y^\ast = 1.2874$ (capital-output ratio $4.712$); the excess supply at the root is $-1.76 \times 10^{-6}$, well inside the default `tol_ge=1e-4`, so `converged` is `True` after 6 Brent iterations, and the household problem at $r^\ast$ took 10 HJB iterations. Passing `tol_ge=1e-8` instead drives the same run to $\Phi = -7.4 \times 10^{-13}$ in 7 iterations. `to_markdown()` renders the summary table (the elapsed-time row is machine-dependent):

```text
|                         Metric |     Value |
|--------------------------------|-----------|
| Equilibrium Interest Rate (r*) |  0.020041 |
|          Equilibrium Wage (w*) |  1.437585 |
|         Aggregate Capital (K*) |  6.065605 |
|            Capital Supply (Ks) |  6.065605 |
|            Capital Demand (Kd) |  6.065607 |
|  Capital Market Clearing Error | -1.76e-06 |
|           Aggregate Labor (L*) |  0.600000 |
|          Aggregate Output (Y*) |  1.287389 |
|      GE Root-finding Converged |      True |
|                  GE Iterations |         6 |
|               Elapsed Time (s) |    0.0421 |
```

### 4.4 A Custom Income Process: Passing `e_grid` and `A_z`

```python
import numpy as np
from puremacro.vfi import solve_hjb_achdou

e_grid = np.array([0.5, 1.0, 1.5])
A_z = np.array([[-0.20, 0.20, 0.00],
                [ 0.10,-0.20, 0.10],
                [ 0.00, 0.20,-0.20]])          # generator: off-diagonals >= 0, rows sum to zero
assert np.allclose(A_z.sum(axis=1), 0.0)

sol = solve_hjb_achdou(r_rate=0.03, w_rate=1.0, Na=100, a_max=30.0, e_grid=e_grid, A_z=A_z)
da = sol.a_grid[1] - sol.a_grid[0]
print(sol.n_iter, sol.converged, sol.V.shape)
print("stationary income shares:", np.round(sol.g_dist.sum(axis=0) * da, 4))
print("mean wealth by income state:", np.round((sol.a_grid[:, None] * sol.g_dist * da).sum(axis=0) / (sol.g_dist.sum(axis=0) * da), 3))
```

`A_z` is a generator, not a transition-probability matrix: entry $(j, k)$ is the Poisson intensity of jumping from state $j$ to state $k$, the diagonal is minus the total exit rate, and rows sum to zero. The solver checks this for you — a matrix whose rows sum to one is rejected with `ValueError: A_z rows must sum to zero (continuous-time generator) ... A_z looks like a transition-probability matrix P; the solver expects a continuous-time generator with zero row sums, e.g. (P - I) / dt` — so the `assert` above is belt and braces rather than a necessity. The three-state birth-death chain converges in 8 iterations; the KFE recovers the chain's stationary distribution `[0.25 0.5 0.25]` as the income marginal, and mean wealth rises with productivity, `[1.686 2.837 4.227]`.

### 4.5 Iteration Count as a Function of `Delta`

```python
import numpy as np
from puremacro.vfi import solve_hjb_achdou

base = solve_hjb_achdou(Delta=1e4)
for Delta in (1e4, 1e2, 10.0, 1.0):
    s = solve_hjb_achdou(Delta=Delta)                # max_iter=100, tol=1e-8
    print(f"Delta = {Delta:>8.0f}: n_iter = {s.n_iter:3d}  converged = {s.converged!s:5}  "
          f"max|V - V_ref| = {np.max(np.abs(s.V - base.V)):.1e}")
print("Delta = 1, max_iter = 1000:", solve_hjb_achdou(Delta=1.0, max_iter=1000).n_iter)
```

This is the source of the table in Section 1.3: 9, 14, 50 and 100 iterations for $\Delta = 10^4, 10^2, 10, 1$, with `converged=False` at $\Delta = 1$ under the default `max_iter=100` (the value function is still $6.7 \times 10^{-2}$ away from the reference) and 360 iterations when the cap is raised to 1000. Always check `converged` after lowering `Delta`.

---

## 5. Full API Specification

```text
solve_hjb_achdou(
    r_rate: float = 0.03,
    w_rate: float = 1.0,
    rho_val: float = 0.05,
    gamma_r: float = 2.0,
    Na: int = 100,
    max_iter: int = 100,
    tol: float = 1e-8,
    *,
    a_min: float = 0.0,
    a_max: float = 30.0,
    a_grid: np.ndarray | None = None,
    e_grid: np.ndarray | None = None,
    A_z: np.ndarray | None = None,
    Delta: float = 1e4,
    compute_kfe: bool = True,
    v_prime_boundary: tuple[float | None, float | None] | None = None,
) -> HJBSolution

solve_kfe_achdou(
    A: scipy.sparse.spmatrix,
    a_grid: np.ndarray,
    e_grid: np.ndarray,
    return_residual: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]

solve_aiyagari_continuous_hjb(
    alpha: float = 0.33,
    delta: float = 0.05,
    rho_val: float = 0.05,
    gamma_r: float = 2.0,
    Na: int = 100,
    a_min: float = 0.0,
    a_max: float = 30.0,
    a_grid: np.ndarray | None = None,
    e_grid: np.ndarray | None = None,
    A_z: np.ndarray | None = None,
    r_min: float = 0.005,
    r_max: float | None = None,
    tol_ge: float = 1e-4,
    max_iter_ge: int = 40,
    max_iter_hjb: int = 100,
    tol_hjb: float = 1e-8,
    Delta: float = 1e4,
) -> AiyagariContinuousHJBResult
```

#### `solve_hjb_achdou` parameters

| Parameter | Meaning |
|---|---|
| `r_rate`, `w_rate` | Interest rate on assets and wage per efficiency unit. `w_rate=0.0` switches to the cake-eating boundary treatment (Section 1.2) and then requires `a_grid[0] > 0`, because the closed form $(\mu a)^{-\gamma}$ is singular at $a = 0$; `ValueError` otherwise. |
| `rho_val`, `gamma_r` | Discount rate $\rho$ and CRRA coefficient $\gamma$; $\gamma = 1$ selects log utility. |
| `Na`, `a_min`, `a_max` | Uniform asset grid `np.linspace(a_min, a_max, Na)`; `a_min` is the borrowing limit. |
| `a_grid` | Explicit grid; overrides `Na`, `a_min`, `a_max`. May be non-uniform; the KFE weights each node by its cell width (Section 1.4). Must be 1-D, finite and strictly increasing with at least two points (`ValueError` otherwise). |
| `e_grid` | Productivity states, shape `(Ne,)`; default `[0.2, 1.0]`. |
| `A_z` | Income generator, shape `(Ne, Ne)`, finite, non-negative off-diagonals and zero row sums — all validated, with `ValueError` on any violation. Default: symmetric intensity $0.1$ for two states, $0.2/(N_e - 1)$ off-diagonal with $-0.2$ on the diagonal otherwise, `[[0.]]` for one state. |
| `max_iter`, `tol` | Iteration cap (integer $\ge 1$, `ValueError` otherwise) and sup-norm tolerance on $v^{n+1} - v^n$. `converged` is `False` when the cap binds. |
| `Delta` | Implicit step $\Delta$ (Section 1.3). |
| `compute_kfe` | Solve the adjoint KFE after convergence and fill `g_dist` and `mass_residual`. |
| `v_prime_boundary` | `(v'_min, v'_max)` Neumann derivatives; either entry may be `None`. |

#### `solve_kfe_achdou` parameters

| Parameter | Meaning |
|---|---|
| `A` | Generator from `HJBSolution.A_generator` (or any sparse matrix with the same structure), shape `(Na*Ne, Na*Ne)` in column-major state order. A shape mismatch, or a generator with more than one closed communicating class, raises `ValueError`. |
| `a_grid`, `e_grid` | The grids the generator was built on; only their lengths and the asset spacings are used. `a_grid` must be strictly increasing. |
| `return_residual` | `True` returns `(g, mass_residual)`; `False` returns `g` alone. `g` is a density: mass per unit of $a$, with $\sum_{i,j} g_{i,j} w_i = 1$. |

#### `solve_aiyagari_continuous_hjb` parameters

| Parameter | Meaning |
|---|---|
| `alpha`, `delta` | Cobb-Douglas capital share and depreciation rate. |
| `rho_val`, `gamma_r`, `Na`, `a_min`, `a_max`, `a_grid`, `e_grid`, `A_z`, `Delta` | As for `solve_hjb_achdou`; forwarded to every household solve. |
| `r_min`, `r_max` | Brent bracket; `r_max=None` means $\rho - 0.002$. The bracket is widened automatically as described in Section 1.5. `r_min >= r_max`, `r_max >= rho_val` or `r_min <= -delta` raises `ValueError`. |
| `tol_ge` | Market-clearing tolerance on $\lvert K^s - K^d \rvert$. Sets Brent's $x$-tolerance, drives the re-bracketing passes and decides `converged` (Section 1.5). |
| `max_iter_ge` | Maximum Brent iterations per bracketing pass. |
| `max_iter_hjb`, `tol_hjb` | Passed to `solve_hjb_achdou` as `max_iter` and `tol`. |

---

## 6. Result Interface & Manuscript Export

### `HJBSolution`

A `@dataclass(frozen=True)`: assigning to a field raises `FrozenInstanceError`. Fields:

| Field | Type / shape | Content |
|---|---|---|
| `V` | `(Na, Ne)` | Converged value function $v_j(a_i)$. |
| `c_policy`, `s_drift` | `(Na, Ne)` | Consumption rule and savings drift $s = r a + w e - c$. |
| `a_grid`, `e_grid` | `(Na,)`, `(Ne,)` | Grids actually used. |
| `n_iter`, `elapsed`, `converged` | `int`, `float`, `bool` | Iterations performed, wall-clock seconds, whether `tol` was met. |
| `r_rate`, `w_rate`, `rho_val`, `gamma_r` | `float` | Echo of the calibration. |
| `g_dist` | `(Na, Ne)` or `None` | Stationary **density** from the KFE — mass per unit of $a$, so node mass is $g_{i,j} w_i$ (`None` when `compute_kfe=False`). |
| `A_generator` | `scipy.sparse.csr_matrix` | Generator $\mathbf{A}$ of the last iteration, `(Na*Ne, Na*Ne)`. |
| `mass_residual` | `float` | $\lvert \sum g_{i,j} w_i - 1 \rvert$ after renormalization. |

**Mapping backward compatibility.** `HJBSolution` subclasses `collections.abc.Mapping`. `sol["V"]`, `dict(sol)`, `list(sol)`, `len(sol)` and `sol.get(key, default)` all work. `keys()` (and hence `items()`, `values()`, iteration, `dict(sol)` and `**sol`) returns exactly twelve keys: the seven that the 3.3.0 dictionary carried, `V`, `c_policy`, `s_drift`, `a_grid`, `e_grid`, `n_iter`, `elapsed`, plus the parameter echoes `r_rate`, `w_rate`, `rho_val`, `gamma_r` and the flag `converged`. Subscripting, `in` and `get` are restricted to the **dataclass fields**, i.e. those twelve plus `g_dist`, `A_generator` and `mass_residual`: `"g_dist" in sol` is `True` while `"summary" in sol` is `False` and `sol["summary"]` raises `KeyError`, so no method or private attribute leaks through the mapping interface. `==` compares the fields one by one and is array- and sparse-aware (note that `elapsed` is wall-clock, so two separate solves of the same problem do *not* compare equal); `__hash__` is `None`, so instances are unhashable like a `dict`, and the dataclass is frozen, so attribute assignment raises `FrozenInstanceError` and item assignment raises `TypeError`.

Methods:

- `summary() -> pd.DataFrame`: fourteen rows indexed by `Metric` (grid sizes, iterations, convergence, elapsed time, the four parameters, mean value, minimum and maximum consumption, KFE mass error), as printed in Example 4.1.
- `to_frame() -> pd.DataFrame`: one row per state, with columns `a_idx`, `e_idx`, `asset_a`, `prod_e`, `value_V`, `consumption_c`, `savings_drift_s` and, when the KFE was solved, `density_g` (the density, not the node mass). This is the frame to use for custom plots of the distribution.
- `to_markdown(**kwargs)`, `to_latex(**kwargs)`, `to_typst(**kwargs) -> str`: the summary table rendered through `puremacro.reports.df_to_markdown`, `df_to_latex` and `df_to_typst` (accepting their `index` and `digits` keywords).
- `plot(*, ax=None, figsize=None, show=False)`: three panels, value function, consumption rule and savings drift, one line per income state. Returns the `Figure` when it creates one, or the `Axes` you passed; a sequence of at least three axes fills all panels, a single axis draws only the value function. Headless-safe: nothing is shown unless `show=True`. The stationary distribution is not plotted; build it from `g_dist` or `to_frame()`.

### `AiyagariContinuousHJBResult`

Also a frozen dataclass with the same `Mapping` protocol — here `keys()` lists all thirteen fields, subscripting, `in` and `get` are restricted to those fields, `==` is array-aware field equality and instances are unhashable.

| Field | Content |
|---|---|
| `r_star`, `w_star` | Market-clearing interest rate and wage. |
| `K_star`, `Ks_star`, `Kd_star` | Equilibrium capital ($K^\ast = K^s(r^\ast)$), household supply and firm demand. |
| `L_star`, `Y_star` | Effective labor $\sum_j e_j \pi_j$ and output $Y = (K^d)^\alpha L^{1-\alpha}$. |
| `excess_capital` | $K^s(r^\ast) - K^d(r^\ast)$. |
| `hjb_solution` | The `HJBSolution` at $r^\ast$, $w^\ast$. |
| `g_dist` | Alias of `hjb_solution.g_dist` (a density). |
| `converged`, `n_iter_ge`, `elapsed` | `True` only when Brent converged **and** $\lvert K^s - K^d \rvert <$ `tol_ge`; Brent iterations summed over the bracketing passes; wall-clock seconds for the whole GE solve. |

Methods: `summary()` (eleven rows: prices, capital supply and demand, clearing error, labor, output, convergence, iterations, elapsed time), `to_frame()` (delegates to `hjb_solution.to_frame()`, so it tabulates the household solution at equilibrium), `to_markdown()`, `to_latex()`, `to_typst()` (the summary table, as in Example 4.3) and `plot()` (delegates to `hjb_solution.plot()`, so it shows the three household panels, not the distribution or the capital-market diagram).

---

## 7. Caveats & Limitations

- **`g_dist` is a density, not a vector of node masses.** Multiply by the cell-width weights before summing: node mass is $g_{i,j} w_i$, and $w_i = \Delta a$ only on a uniform grid. The examples on this page do exactly that. This matters most on non-uniform grids — on the quadratic grid `np.linspace(0, 1, 100)**2 * 30` the weighted capital supply is $5.9155$, and reading `g_dist` as a mass vector instead would be wrong by the local cell width at every node.
- **The iteration count depends on `Delta`.** With `Delta=1` the default `max_iter=100` is not enough (Section 4.5). `converged` is the field to check; the solver does not warn.
- **Brent's iteration budget is per pass.** `max_iter_ge` caps each bracketing pass, and `n_iter_ge` reports the sum over passes, so the total number of household solves can exceed `max_iter_ge` when `tol_ge` forces a re-bracket. The HJB solutions are cached by $r$, so a repeated trial rate costs nothing.
- **`mass_residual` is measured after renormalization** and is therefore of order $10^{-16}$ regardless of how accurate the linear solve was; it certifies the normalization, not the KFE residual. Compute `A_generator.T @ (g_dist * w[:, None]).ravel(order="F")` — the generator acts on node masses — if you want the latter.
- **State ordering is column-major.** `A_generator` indexes state $(i, j)$ as $j N_a + i$; reshape vectors with `order="F"`, as `solve_kfe_achdou` does.
- **`keys()` is narrower than `in`.** Membership and subscripting cover all fifteen dataclass fields, but `keys()` (and therefore `dict(sol)`, `**sol` and iteration) lists only the twelve legacy keys, so `dict(sol)` drops `g_dist`, `A_generator` and `mass_residual`. Read those by attribute.
- **Truncation at `a_max`.** The upper boundary is an artificial state constraint. If a visible mass of `g_dist` piles up in the last nodes, raise `a_max` (the default 30 is ample for the default calibration: mean wealth is about 6).
- **Interface and economics changed in 3.4.0.** `solve_hjb_achdou` returned a plain dict with seven keys and used `max_iter=500`, `tol=1e-7` and an explicit scheme; 3.4.0 returns `HJBSolution`, defaults to `max_iter=100`, `tol=1e-8` and the implicit scheme, and makes every keyword after `tol` keyword-only. It also adds the Poisson income-switching term the explicit solver omitted, so value and consumption for identical arguments differ materially from 3.3.0 (Section 2.2). Code that used the dict keeps working; code that relied on the old defaults should pass them explicitly, and code that needs the old *numbers* must pass `A_z=np.zeros((Ne, Ne))` with `compute_kfe=False`.
- **Pure NumPy/SciPy, float64, no accelerators, no network.** The generator is assembled in Python loops over `Na * Ne` states and factorized with SuperLU; a few thousand states solve in milliseconds, but the assembly cost grows linearly with `Na * Ne**2`. The module imports only NumPy, SciPy, pandas and `puremacro.reports` — matplotlib is imported lazily inside `plot`, so `import puremacro.vfi` leaves `matplotlib` out of `sys.modules` — and everything runs offline. That first import still takes seconds (2.4 s in the cold run behind this page), far more than any solve on this page.

---

## References

- Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., & Moll, B. (2022). "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *The Review of Economic Studies*, 89(1), 45–86.
- Aiyagari, S. R. (1994). "Uninsured Idiosyncratic Risk and Aggregate Saving." *The Quarterly Journal of Economics*, 109(3), 659–684.
- Barles, G., & Souganidis, P. E. (1991). "Convergence of approximation schemes for fully nonlinear second order equations." *Asymptotic Analysis*, 4(3), 271–283.
- Huggett, M. (1993). "The risk-free rate in heterogeneous-agent incomplete-insurance economies." *Journal of Economic Dynamics and Control*, 17(5–6), 953–969.
- Young, E. R. (2010). "Solving the incomplete markets model with aggregate uncertainty using the Krusell-Smith algorithm and non-stochastic simulations." *Journal of Economic Dynamics and Control*, 34(1), 36–41.
