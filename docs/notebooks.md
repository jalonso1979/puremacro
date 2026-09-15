> 🇬🇧 English · 🇪🇸 [Español](es/notebooks.md)

# Showcase Notebooks

`puremacro` ships interactive, publication-quality showcase notebooks covering the major heterogeneous-agent macro paradigms, empirical macroeconometrics, Bayesian estimation, specialized climate/historical applications, and frontier applied macroeconomic policy suites.

Every notebook is written in Jupytext percent format (`.py`), executes deterministically with fixed seeds, adheres to the Pyodide 4-package contract (`numpy`, `scipy`, `pandas`, `matplotlib`), and is available with a paired Spanish edition (`_es.py`). Pre-compiled `.ipynb` artifacts with all outputs and figures pre-rendered are generated via `tools/build_notebooks.py`.

---

## The 7-Section Pedagogical Architecture

Following `notebooks/_TEMPLATE.md`, deepened and frontier showcase notebooks adhere to a consistent 7-cell structural flow:

1. **Motivating Question**: 1–2 sentences defining the economic problem and policy trade-off.
2. **The Method in Math**: Governing structural and econometric equations in compact, rigorous LaTeX ($...$ / $$...$$).
3. **Intuition**: An explicit `**Intuition.**` section translating algebraic equations into intuitive economic mechanisms and identification logic.
4. **Worked Code**: Self-contained, pure-NumPy runnable code blocks with explanatory comments on *why* choices are made.
5. **Read the Output**: Dedicated markdown analysis directly interpreting headline numerical outputs, parameter estimates, and generated figures.
6. **Your Turn**: An interactive exploratory exercise with `# ← change this` knobs, runnable defaults with assertions, and graded challenge prompts.
7. **How Comprehensive Is This?**: Contextual cross-references connecting the showcase to related `puremacro` entry points and literature.

## Continuous HJB/KFE, Multi-Constraint OccBin & Latin America Macro Showcases (puremacro 3.4)

Showcases `56` through `58` demonstrate `puremacro`'s frontier continuous-time Hamilton-Jacobi-Bellman (HJB) implicit upwind solvers with adjoint Kolmogorov Forward Equations (KFE), multi-constraint piecewise-linear DSGE perturbation (OccBin $M \ge 2$) coupled with Double / Debiased Machine Learning (DML-PLR), and portable Latin American real-time macroeconomic vintage cartridges (`.pmz`) with Mankiw-Shapiro (1986) news vs. noise econometrics.

### `56_implicit_hjb_and_continuous_kfe`
- **Source**: `notebooks/56_implicit_hjb_and_continuous_kfe.py` (Spanish: `_es.py`, compiled: `.ipynb`)
- **Motivating Economic Question**: How do quantitative macroeconomists solve continuous-time heterogeneous-agent models with uninsurable income risk and borrowing constraints without discrete-time Euler equation inversion errors, and how does the adjoint Kolmogorov Forward Equation (KFE) ensure exact machine-precision mass conservation for stationary wealth distributions?
- **Governing Math & Algorithms**:
  - **Continuous-Time HJB with Implicit Upwind Scheme**: Households solve:
    $$\rho v_j(a) = u(c_j(a)) + s_j(a) v_j'(a) + \sum_{k \ne j} \lambda_{jk} [v_k(a) - v_j(a)]$$
    Subject to $s_j(a) = r a + z_j - c_j(a)$ and $a \ge \underline{a}$. Upwind finite differences evaluate forward ($v_{j, i}'^F$) and backward ($v_{j, i}'^B$) drifts, selecting forward if $s_F > 0$, backward if $s_B < 0$, and zero drift $s_0 = 0$ at stationary turning points.
  - **M-Matrix Structure of Infinitesimal Generator $A$**:
    $$\rho v^{n+1} = u(c^n) + A(v^n) v^{n+1} + \frac{1}{\Delta} (v^{n+1} - v^n)$$
    Matrix $(\rho + 1/\Delta) I - A(v^n)$ is strictly diagonally dominant with negative off-diagonals, guaranteeing existence, uniqueness, and unconditional stability for any $\Delta > 0$.
  - **Adjoint Kolmogorov Forward Equation (KFE)**:
    $$A(v^*)^\top g = 0, \quad \text{s.t.} \quad \sum_{j=1}^J \sum_{i=1}^I g_{j, i} \Delta a_i = 1.0$$
    Replacing one row with normalization ensures machine-precision mass conservation ($|\sum g_i \Delta a_i - 1| \le 10^{-15}$).
  - **Continuous Aiyagari General Equilibrium**:
    $$K^s(r) = \sum_{j, i} a_i g_{j, i} \Delta a_i = K^d(r) = \left( \frac{r + \delta}{\alpha \bar{Z}} \right)^{\frac{1}{\alpha - 1}}$$
- **Economic Intuition**: Discrete-time models require fine time aggregation and suffer from lower-bound binding jitter. Continuous-time formulation replaces discrete choice with a smooth drift function $s(a, z)$. Upwind finite differences mirror the physical direction of asset flows: households accumulating assets look forward ($v_{i+1} - v_i$), while households decumulating assets look backward ($v_i - v_{i-1 interior}$), avoiding non-physical oscillation. Because the transition generator $A$ is an infinitesimal Markov generator, its adjoint $A^\top$ yields the exact stationary wealth density $g(a, z)$ in a single linear solve, preserving mass conservation without Monte Carlo simulation noise.
- **Worked Code**:
  ```text
  from puremacro.vfi import solve_hjb_achdou, solve_aiyagari_continuous_hjb

  # Continuous-time HJB implicit solver
  sol_hjb = solve_hjb_achdou(r=0.03, gamma=2.0, rho=0.05, a_min=0.0, a_max=30.0, n_a=100)

  # Continuous Aiyagari General Equilibrium
  sol_ge = solve_aiyagari_continuous_hjb(gamma=2.0, rho=0.05, alpha=0.33, delta=0.05)
  ```
- **Read the Output & Visualizations**: 4-Panel hero dashboard: (1) Converged value functions $v_j(a)$ exhibiting strict concavity; (2) Optimal consumption policies $c_j(a)$ with liquidity-constrained MPC kinks near $\underline{a}$; (3) Drift trajectories $s_j(a)$ showing asset decumulation for low-income and accumulation for high-income households; (4) Stationary wealth density $g(a, z)$ showing the characteristic precautionary mass spike at the borrowing constraint.
- **Your Turn Interactive Knobs & Asserts**: Knobs `gamma_custom = 1.50`, `a_min_custom = 0.0`, `r_test = 0.025` with assertion gates `sol_custom.converged`, `np.isclose(sol_custom.g.sum() * da, 1.0, atol=1e-10)`, `r_star > 0.0`.
- **Literature & Cross-References**: Achdou, Han, Lasry, Lions & Moll (2022), Aiyagari (1994), Huggett (1993). User guide: [`docs/vfi_continuous_equilibrium.md`](vfi_continuous_equilibrium.md).

### `57_multiconstraint_occbin_and_dml`
- **Source**: `notebooks/57_multiconstraint_occbin_and_dml.py` (Spanish: `_es.py`, compiled: `.ipynb`)
- **Motivating Economic Question**: How do economies respond when multiple occasionally binding constraints (such as the nominal Zero Lower Bound and collateral borrowing caps) bind simultaneously, and how can macroeconomists obtain unbiased causal effect estimates of macro policies using high-dimensional Double / Debiased Machine Learning (DML-PLR)?
- **Governing Math & Algorithms**:
  - **Multi-Constraint OccBin Piecewise-Linear Perturbation**: For $M$ constraints, $2^M$ discrete regimes:
    $$A_{r_t} x_t = B_{r_t} x_{t-1} + C_{r_t} \mathbb{E}_t[x_{t+1}] + D_{r_t} + E_{r_t} \varepsilon_t$$
    Regime dynamic transitions iterate backward from horizon $T$ to determine regime sequence $\{r_1, r_2, \dots, r_T\}$ satisfying complementary slackness conditions for both ZLB ($i_t \ge 0$) and borrowing ($b_t \le \bar{b}$).
  - **Double / Debiased Machine Learning (DML-PLR)**: Partially linear regression model:
    $$Y = D \theta_0 + g_0(X) + U, \quad \mathbb{E}[U | D, X] = 0$$
    $$D = m_0(X) + V, \quad \mathbb{E}[V | X] = 0$$
    Neyman-orthogonal score removes regularization bias of machine learning estimators $\hat{\ell}(X)$ and $\hat{m}(X)$:
    $$\psi(W; \theta, \eta) = (Y - \ell(X)) - \theta (D - m(X))$$
    $K$-fold cross-fitting eliminates overfitting bias, delivering $\sqrt{N}$-consistent and asymptotically normal estimates $\hat{\theta} \sim \mathcal{N}(\theta_0, \sigma^2 / N)$.
- **Economic Intuition**: When a severe contractionary shock hits, households cut borrowing against their credit limit while the central bank lowers interest rates to zero. If both constraints bind concurrently (Regime 3), the economy experiences severe non-linear amplification: monetary policy cannot provide accommodation while credit-constrained households cannot borrow to smooth consumption. In the empirical stage, naive OLS fails catastrophically due to omitted variable bias from 100 high-dimensional macroeconomic controls ($+42.5\%$ bias). DML uses cross-fitted penalized regression to orthogonalize both policy $D$ and outcome $Y$ with respect to confounders $X$, recovering the structural policy parameter with exact statistical coverage.
- **Worked Code**:
  ```text
  from puremacro.dsge import OccBinConstraint, OccBinMultiConstraint, solve_multiconstraint_occbin
  from puremacro.dml import DoubleMLPLR

  # 1. Multi-constraint OccBin solve
  occ_res = solve_multiconstraint_occbin(model, constraints=[zlb_c, borrow_c], shock=shock)

  # 2. DML Partially Linear Regression
  dml_res = DoubleMLPLR(Y, D, X, n_folds=5, estimator="lasso").fit()
  ```
- **Read the Output & Visualizations**: 4-Panel hero dashboard: (1) Discrete regime timeline tracking duration of joint ZLB-borrowing crisis; (2) IRF comparison between linear unconstrained trajectory and multi-constraint piecewise path; (3) Cross-fitting fold residual scatter plots; (4) Sampling distribution density comparing DML estimate against biased naive OLS.
- **Your Turn Interactive Knobs & Asserts**: Knobs `shock_g_custom = -0.05`, `n_folds_custom = 5`, `alpha_custom = 0.05` with assertion gates `res_custom.converged`, `np.abs(dml_custom.theta - theta_true) < 0.20`, `dml_custom.p_value < 0.01`.
- **Literature & Cross-References**: Guerrieri & Iacoviello (2015), Chernozhukov et al. (2018), Belloni, Chernozhukov & Hansen (2014). User guides: [`docs/dsge_higher_order.md`](dsge_higher_order.md), [`docs/forecast.md`](forecast.md).

### `58_latin_america_realtime_macro`
- **Source**: `notebooks/58_latin_america_realtime_macro.py` (Spanish: `_es.py`, compiled: `.ipynb`)
- **Motivating Economic Question**: How do preliminary GDP and macroeconomic data releases by Latin American statistical institutes and central banks evolve through successive historical vintages, and are these revisions driven by rational updates with new information ("news") or errors and noise in initial measurement ("noise")?
- **Governing Math & Algorithms**:
  - **Revision Triangles $(T \times V)$**: Real-time vintage data matrix where rows denote statistical reference quarters $t = 1, \dots, T$ and columns denote vintage publication quarters $v = 1, \dots, V$ ($v \ge t$):
    $$R_{t, k} = y_{t, t+k+1} - y_{t, t+k}, \quad R_{t, \text{final}} = y_{t, \text{final}} - y_{t, \text{first}}$$
  - **Mankiw-Shapiro (1986) News vs. Noise Econometrics**:
    $$\text{Model 1 (News / Rational Forecast)}: \quad y_{t, \text{final}} - y_{t, \text{first}} = \alpha + \beta y_{t, \text{first}} + \varepsilon_t$$
    Under rational statistical forecasting, initial releases use all available information; future revisions are unpredictable from initial data: $H_0: \alpha = 0, \beta = 0$.
    $$\text{Model 2 (Noise / Measurement Error)}: \quad y_{t, \text{first}} = \alpha + \beta y_{t, \text{final}} + u_t$$
    Under classical measurement noise, the preliminary release is a noisy proxy of true GDP: $H_0: \alpha = 0, \beta = 1$.
  - **HAC Newey-West Inference**: Robust covariance estimation accounting for heteroskedasticity and serial correlation in revision residuals.
  - **Cryptographic Offline `.pmz` Cartridges**: Portable, offline-contained `.pmz` vintage archives with SHA-256 integrity verification, schema validation, and instant zero-network unpacking.
- **Economic Intuition**: Preliminary releases published 30–45 days after quarter-end rely on incomplete survey samples and statistical nowcasting models. As hard data (tax filings, balance sheets) arrive, statistical agencies revise historical figures. In emerging Latin American economies (Mexico, Brazil, Chile), understanding whether revisions are news or noise determines whether policymakers should respond immediately to preliminary growth signals or discount them as volatile measurement noise. Empirical results show revisions are news-dominated, confirming Latin American central banks produce rational real-time estimates.
- **Worked Code**:
  ```text
  from puremacro.realtime import (
      build_revision_triangle,
      mankiw_shapiro_test,
      export_realtime_cartridge,
      load_realtime_cartridge,
  )

  # Build revision triangle and run Mankiw-Shapiro test
  tri = build_revision_triangle(vintage_df)
  news_res, noise_res = mankiw_shapiro_test(tri)

  # Package into offline cryptographic cartridge
  export_realtime_cartridge(cartridge_file, panel, metadata={"region": "Latin America"})
  ```
- **Read the Output & Visualizations**: 4-Panel hero dashboard: (1) Triangular heatmap displaying real-time vintage revisions across reference quarters and revision lags; (2) Trajectory spaghetti lines tracing GDP estimates from first release to final benchmark; (3) Histogram of revision sizes with normal fit and skewness diagnostics; (4) Mankiw-Shapiro scatter plot with fitted regression lines and HAC confidence bands.
- **Your Turn Interactive Knobs & Asserts**: Knobs `country_custom = "brazil"`, `hac_lags_custom = 4`, `ci_level_custom = 0.95` with assertion gates `tri_custom.shape[0] > 0`, `news_custom.p_value > 0.05`, `noise_custom.p_value < 0.05`.
- **Literature & Cross-References**: Mankiw & Shapiro (1986), Croushore & Stark (2001), Faust, Rogers & Wright (2005). User guide: [`docs/real_time_data.md`](real_time_data.md).

---

## Continuous Dynamic Programming, Deep Macro & Spatial GE Showcases (puremacro 3.3)

Showcases `51` through `55` demonstrate `puremacro`'s frontier continuous state-space dynamic programming, non-linear wealth distribution transitions, machine-precision adjoint sensitivity gradients, high-dimensional neural network policy solvers, and quantitative spatial trade general equilibrium engines.

### `51_continuous_projection_collocation_and_fem`
- **Source**: `notebooks/51_continuous_projection_collocation_and_fem.py` (Spanish: `_es.py`, compiled: `.ipynb`)
- **Motivating Economic Question**: How do quantitative macroeconomists solve dynamic general equilibrium models without the curse of dimensionality and discretization distortion of grid-based value function iteration, and when should one deploy global orthogonal Chebyshev polynomials versus localized finite element Galerkin projection?
- **Governing Math & Algorithms**:
  - **Chebyshev Orthogonal Collocation**: Coordinates $k \in [a, b]$ map to canonical $x \in [-1, 1]$ via $x = \frac{2k - (a+b)}{b - a}$. Basis polynomials obey the 3-term recurrence $T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)$ with Chebyshev-Gauss-Lobatto extrema nodes $x_i = -\cos(i\pi/N)$. Enforces the continuous Euler equation residual operator:
    $$\mathcal{R}_{\text{Euler}}(\theta; k_i) \equiv 1 - \beta \frac{u'(c(g(k_i), g(g(k_i))))}{u'(c(k_i, g(k_i)))} f'(g(k_i)) = 0$$
  - **Finite Element Galerkin Projection (FEM)**: Partitions $[a, b]$ into $E$ elements with piecewise linear Lagrange hat basis functions $\phi_j(k)$ with compact support $[k_{j-1}, k_{j+1}]$. Residuals are projected against test functions using $Q$-point Gauss-Legendre quadrature:
    $$\int_a^b \mathcal{R}(k) \phi_i(k) \, dk = 0, \quad \forall i = 1, \dots, N_{\text{nodes}}$$
  - **Fischer-Burmeister NCP Operator for Borrowing Constraints**: Smooth perturbed complementarity condition ($\epsilon = 10^{-12}$) resolving $k' \ge \bar{k}$ and $\mathcal{R}(k) \ge 0$:
    $$\Psi_{\text{FB}}^\epsilon(a, b) = a + b - \sqrt{a^2 + b^2 + \epsilon} = 0, \quad a = k' - \bar{k}, \quad b = \mathcal{R}(k)$$
- **Economic Intuition**: Global Chebyshev polynomials achieve exponential spectral convergence ($\mathcal{O}(c^{-N})$) for real-analytic decision rules (e.g. Brock-Mirman growth), but trigger catastrophic Gibbs oscillations and lower-bound violations near sharp corners. FEM abandons global support in favor of localized hat basis functions ($\mathcal{O}(h^2)$ algebraic convergence), completely eliminating Gibbs ringing when an element node is pinned at the borrowing threshold $k^*$.
- **Worked Code**:
  ```text
  import numpy as np
  from puremacro.vfi import CollocationProblem, solve_collocation, FEMProblem, solve_fem

  # Analytical Brock-Mirman (1972) benchmark: u(c)=ln(c), f(k)=k^alpha, delta=1.0
  alpha, beta = 0.36, 0.96
  k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
  domain = (0.5 * k_ss, 1.5 * k_ss)

  def euler_res(k, kp, kpp, s, sp):
      c = k**alpha - kp
      c_next = kp**alpha - kpp
      return 1.0 / c - beta * (1.0 / c_next) * (alpha * kp**(alpha - 1.0))

  prob_col = CollocationProblem(domain=domain, deg=7, beta=beta, euler_residual_fn=euler_res)
  sol_col = solve_collocation(prob_col)

  prob_fem = FEMProblem(
      domain=domain, n_elements=25, deg=1, beta=beta,
      return_fn=lambda k, kp, s: np.log(np.maximum(k**alpha - kp, 1e-8)),
      transition_fn=lambda k, kp, s: kp,
      euler_residual_fn=euler_res,
  )
  sol_fem = solve_fem(prob_fem)
  ```
- **Read the Output & Visualizations**: Four publication figures: (1) Semi-log policy error convergence confirming exponential spectral decay ($L^\infty < 10^{-6}$ at $N=8$) vs algebraic FEM decay; (2) Zoomed kink vicinity demonstrating Chebyshev ringing vs exact FEM boundary compliance; (3) Multi-state stochastic policy curves $g(k, z)$ preserving monotonicity; (4) Multi-backend execution wall-time scaling across NumPy, Numba, and MLX.
- **Your Turn Interactive Knobs & Asserts**: Knobs `order_custom = 8`, `elements_custom = 50`, `borrow_ratio_custom = 0.50` with automated verification assertions `sol_custom_c.converged`, `sol_custom_f.converged`, `err_custom_c < 1e-3`, `viol_custom_f == 0.0`.
- **Literature & Cross-References**: Brock & Mirman (1972), Judd (1992, 1998), McGrattan (1996), Fischer (1992). User guide: [`docs/vfi_continuous_projection.md`](vfi_continuous_projection.md).

### `52_continuous_transition_mit_shocks`
- **Source**: `notebooks/52_continuous_transition_mit_shocks.py` (Spanish: `_es.py`, compiled: `.ipynb`)
- **Motivating Economic Question**: How do continuous household wealth distributions $\mu_t(k, z)$ and general equilibrium factor prices $\{r_t, w_t\}$ transition non-linearly following an unexpected aggregate macroeconomic shock in an incomplete-markets economy, and how does endogenous precautionary savings shape macroeconomic persistence and wealth inequality?
- **Governing Math & Algorithms**:
  - **Backward Continuous EGM**: Inverting marginal utility along continuous assets $a' \ge 0$ given continuation sequence $\{r_t, w_t\}$:
    $$c_t^{\text{endo}}(a', z_i) = \left( \beta (1 + r_{t+1}) \sum_j P_z(z_i, z_j) [c_{t+1}(a', z_j)]^{-\gamma} \right)^{-1/\gamma}$$
    $$a_t^{\text{endo}}(a', z_i) = \frac{c_t^{\text{endo}}(a', z_i) + a' - w_t z_i}{1 + r_t}$$
  - **Forward Time-Dependent Young (2010) Lottery Operator**: Density push-forward preserving mass conservation to machine precision ($\sum \mu_t = 1.0 \pm 10^{-15}$):
    $$\mu_{t+1}(k_m, z_l) = \sum_j P_z(z_j, z_l) \sum_i \mu_t(k_i, z_j) [w_{\text{lo}} \mathbf{1}_{\{m=j_{\text{lo}}\}} + w_{\text{hi}} \mathbf{1}_{\{m=j_{\text{hi}}\}}]$$
  - **Sequence-Space Capital Market Clearing**:
    $$H_t(\mathbf{r}) \equiv \sum_{i, j} k_i \mu_t(k_i, z_j) - L \left( \frac{r_t + \delta}{\alpha Z_t} \right)^{\frac{1}{\alpha - 1}} = 0, \quad \forall t \in [0, T-1]$$
  - **Sequence-Space Broyden Quasi-Newton Solver**: Treats full $T$-period price trajectory $\mathbf{r} \in \mathbb{R}^T$ as a unified vector with Sherman-Morrison rank-1 approximate inverse Jacobian updates.
- **Economic Intuition**: Aggregate capital supply cannot adjust instantaneously at $t=0$ because the wealth distribution $\mu_0(k, z)$ is physically predetermined. To clear factor markets after an unexpected $+5\%$ TFP shock, the real interest rate spikes immediately ($+49.9$ bps). The surge in real wages disproportionately benefits low-wealth, credit-constrained workers, compressing the wealth Gini ($0.5269 \to 0.5212$). As households channel precautionary savings into capital, aggregate capital deepens to peak at $t=7$ ($K^* = 8.792 \to K_{\text{peak}} = 9.039$), demonstrating endogenous propagation far outlasting the exogenous shock ($0.80^7 \approx 0.21$).
- **Worked Code**:
  ```text
  import numpy as np
  from puremacro.vfi import solve_continuous_transition, TransitionShock

  # Unexpected +5% TFP shock with rho=0.80 over T=40 quarters
  shock_path = 0.05 * (0.80 ** np.arange(40))
  shock = TransitionShock(variable="TFP", path=shock_path)
  res = solve_continuous_transition(model, shock=shock, T=40, damping=0.40)
  ```
- **Read the Output & Visualizations**: Hero 4-panel dashboard (Real Rate Path, Capital Supply vs Demand, Wealth Gini Path, Lorenz Curves Dynamics) and a 3D perspective surface $\mu_t(k)$ tracking the dynamic dispersion of precautionary wealth holdings over time.
- **Your Turn Interactive Knobs & Asserts**: Knobs `user_shock_size = 0.05`, `user_persistence = 0.80`, `user_damping = 0.40` with assertions `user_res.converged`, `user_res.max_residual < 1e-4`, `user_res.mass_conservation_error < 1e-12`.
- **Literature & Cross-References**: Aiyagari (1994), Young (2010), Boppart, Krusell & Mitman (2018), Auclert et al. (2021). User guides: [`docs/vfi_continuous_transition.md`](vfi_continuous_transition.md), [`docs/vfi_continuous_equilibrium.md`](vfi_continuous_equilibrium.md).

### `53_exact_analytic_ift_gradients`
- **Source**: `notebooks/53_exact_analytic_ift_gradients.py` (Spanish: `_es.py`, compiled: `.ipynb`)
- **Motivating Economic Question**: How can quantitative macroeconomists perform gradient-based structural estimation (SMM/GMM) of continuous dynamic models without the numerical noise and combinatorial cost of finite-difference approximations, and how does the Implicit Function Theorem deliver machine-precision policy sensitivities in a single linear solve?
- **Governing Math & Algorithms**:
  - **Implicit Function Theorem on Continuous Projections**: Differentiating the residual identity $\mathcal{R}(c^*(\theta); \theta) \equiv \mathbf{0} \in \mathbb{R}^N$ with respect to structural parameters $\theta \in \mathbb{R}^p$:
    $$\nabla_\theta c^*(\theta) = - \left[ \nabla_c \mathcal{R}(c^*; \theta) \right]^{-1} \nabla_\theta \mathcal{R}(c^*; \theta)$$
  - **Continuous Policy & Macroeconomic Aggregate Sensitivities**:
    $$\nabla_\theta g(k; \theta) = \Phi(k) \, \nabla_\theta c^*(\theta), \qquad \frac{d K^*}{d \theta} = \frac{\nabla_\theta g(k^*; \theta)}{1 - g'(k^*; \theta)}$$
  - **Exact Structural GMM Objective & Gradient**:
    $$Q(\theta) = (m(\theta) - \hat{m})^\top W (m(\theta) - \hat{m}), \qquad \nabla_\theta Q(\theta) = 2 \, G(\theta)^\top W (m(\theta) - \hat{m})$$
    where $G(\theta) \equiv \nabla_\theta m(\theta) = \nabla_c m \, \nabla_\theta c^*(\theta)$ is constructed analytically without finite differences.
- **Economic Intuition**: Finite differences suffer from a fundamental step-size dilemma: large steps introduce truncation errors ($\mathcal{O}(h^2)$), while small steps trigger floating-point cancellation and solver stopping chatter, destroying optimization contours. IFT factors the converged residual Jacobian $J_c = \nabla_c \mathcal{R}$ once via LU decomposition; back-substitutions for all $p$ parameters deliver machine-precision sensitivities ($< 10^{-6}$ error) with a **68.8x empirical speedup**, producing smooth parabolic loss contours that enable BFGS to converge quadratically in 17 iterations.
- **Worked Code**:
  ```text
  from puremacro.vfi import (
      CollocationProblem, solve_collocation,
      compute_ift_gradients, policy_parameter_jacobian,
  )

  sol = solve_collocation(problem)
  ift_res = compute_ift_gradients(sol, problem, params=["beta", "sigma", "alpha", "delta"])
  d_policy_fn = policy_parameter_jacobian(sol, problem, params=["beta"])
  ```
- **Read the Output & Visualizations**: 4-Panel hero visualization (Policy Sensitivities across $k$, Wall-Time Benchmark Bar Chart showing 68.8x speedup, 2D Contour Loss Surface $\log_{10} Q(\beta, \sigma)$ with BFGS Trajectory, and Basis Mode Discrepancy Plot).
- **Your Turn Interactive Knobs & Asserts**: Knobs `user_k1 = 0.10`, `user_k2 = 0.28`, `user_init_beta = 0.93`, `user_init_sigma = 1.30` with assertion gates `user_opt.success`, `np.isclose(user_beta_hat, 0.96, atol=1e-4)`, `np.isclose(user_sigma_hat, 1.50, atol=1e-4)`.
- **Literature & Cross-References**: Judd (1998), Su & Judd (2012), Rust (1994). User guide: [`docs/vfi_analytic_gradients.md`](vfi_analytic_gradients.md).

### `54_deep_macro_pinns_high_dim`
- **Source**: `notebooks/54_deep_macro_pinns_high_dim.py` (Spanish: `_es.py`, compiled: `.ipynb`)
- **Motivating Economic Question**: How can quantitative macroeconomists break the exponential curse of dimensionality to solve dynamic general equilibrium models with 10+ continuous state variables in pure Python without specialized deep learning frameworks, and how do physics-informed neural networks enforce exact physical resource feasibility along ergodic lifetime trajectories?
- **Governing Math & Algorithms**:
  - **High-Dimensional Multi-Country Capital Accumulation**: State $\mathbf{s}_t = (k_{1, t}, \dots, k_{N, t}) \in \mathbb{R}_{++}^N$; cash-on-hand $W_i(\mathbf{s}_t) = A_i k_{i, t}^\alpha + (1 - \delta) k_{i, t}$; capital law $k_{i, t+1} = W_i(\mathbf{s}_t) - c_{i, t}$.
  - **Continuous Dimensionless Euler Residual Operator**:
    $$\mathcal{R}_i(\mathbf{s}_t, \mathbf{c}_t, \mathbf{s}_{t+1}, \mathbf{c}_{t+1}) \equiv 1 - \beta \left(\frac{c_{i, t+1}}{c_{i, t}}\right)^{-\gamma} (\alpha A_i k_{i, t+1}^{\alpha-1} + 1 - \delta) = 0, \quad \forall i = 1, \dots, N$$
  - **Neural Policy Parameterization & Physical Viability**: Trainable MLP $\mathbf{z}(\mathbf{s}; \Theta) \in \mathbb{R}^N$ with bounded sigmoid activation:
    $$\phi_i(\mathbf{s}; \Theta) = \epsilon_{\text{bound}} + (1 - 2\epsilon_{\text{bound}}) \sigma(z_i(\ln(\mathbf{s}/\mathbf{s}_{\text{ss}}))), \qquad c_i(\mathbf{s}; \Theta) = \phi_i(\mathbf{s}; \Theta) W_i(\mathbf{s})$$
    where $\epsilon_{\text{bound}} = 10^{-4}$ guarantees $c_i > 0, k'_i > 0, c_i < W_i$ everywhere.
  - **Ergodic Trajectory Sampling (Maliar et al. 2021)**: Simulating equilibrium paths of length $T = 1200$ concentrates computation on the compact stochastic attractor manifold, collapsing $10^{10}$ grid points to $\mathcal{O}(T \cdot D)$. Loss function $\mathcal{L}(\Theta) = \frac{1}{B \cdot N} \sum_{b, i} (\phi_{i, b} - \phi_{i, b}^{\text{target}})^2$ is minimized via Adam with analytical vectorized gradients in pure NumPy.
- **Economic Intuition**: Discretizing a 10-dimensional continuous state vector on a grid of 10 points per dimension requires $10^{10}$ points. Economic systems, however, never explore high-dimensional hypercubes uniformly; diminishing marginal returns and depreciation exert strong mean-reverting gravitational pulls that confine dynamics to a low-dimensional attractor manifold. Parameterizing policy functions with smooth MLPs (SiLU activations) allows the model to train exclusively on visited ergodic trajectories, generalizing globally with zero external deep learning dependencies (PyTorch/TensorFlow).
- **Worked Code**:
  ```text
  from puremacro.vfi import DeepMacroModel, solve_deep_macro

  model = DeepMacroModel(d_in=10, d_out=10, hidden_layers=[64, 64], activation="silu", seed=42)
  res = solve_deep_macro(
      model, euler_loss_fn=euler_loss, state_sampler=sampler,
      n_epochs=120, lr=1e-3, optimizer="adam", batch_size=256,
  )
  ```
- **Read the Output & Visualizations**: Hero 4-panel dashboard: (1) Training loss convergence ($9.58 \times 10^{-10}$); (2) Out-of-sample Euler residual boxplots across all 10 countries ($\text{MSE} = 3.35 \times 10^{-6} < 10^{-3}$); (3) 10-country capital deepening trajectory from severe initial depression; (4) 1D continuous consumption policy slice against cash-on-hand.
- **Your Turn Interactive Knobs & Asserts**: Knobs `hidden_dims_custom = (32, 32)`, `lr_custom = 2e-3`, `n_epochs_custom = 80`, `perturb_custom = 0.60` with assertion gates `sol_custom.converged`, `sol_custom.test_euler_mse < 1e-3`, `sim_custom["physically_viable"]`.
- **Literature & Cross-References**: Maliar, Maliar & Winant (2021), Raissi, Perdikaris & Karniadakis (2019), Judd (1998). User guide: [`docs/deep_macro.md`](deep_macro.md).

### `55_quantitative_spatial_and_trade_ge`
- **Source**: `notebooks/55_quantitative_spatial_and_trade_ge.py` (Spanish: `_es.py`, compiled: `.ipynb`)
- **Motivating Economic Question**: How do international tariff shocks, supply chain disruptions, and regional transport infrastructure investments propagate through domestic and global input-output linkages and spatial labor mobility to reshape trade flows, regional population distributions, and aggregate economic welfare?
- **Governing Math & Algorithms**:
  - **Caliendo-Parro (2015) Exact Hat Algebra**: $N$ countries, $J$ sectors with value-added share $\gamma_n^j$ and intermediate input cost shares $\gamma_n^{j, k}$. Proportional changes $\hat{x} \equiv x'/x$:
    $$\ln \hat{c}_i^j = \gamma_i^j \ln \hat{w}_i + \sum_k \gamma_i^{j, k} \ln \hat{P}_i^k$$
    $$\hat{P}_n^j = \left[ \sum_i \pi_{ni}^j (\hat{c}_i^j \hat{\tau}_{ni}^j \hat{d}_{ni}^j)^{-\theta_j} \right]^{-1/\theta_j}, \qquad \pi_{ni}'^j = \pi_{ni}^j \left(\frac{\hat{c}_i^j \hat{\tau}_{ni}^j \hat{d}_{ni}^j}{\hat{P}_n^j}\right)^{-\theta_j}$$
    Market clearing for factor income $w_n' L_n$ and national welfare change $\hat{W}_n = \hat{I}_n / \prod_j (\hat{P}_n^j)^{\alpha_n^j}$.
  - **Allen-Arkolakis (2014) Quantitative Spatial General Equilibrium**: Continuous geography with $N$ locations, bilateral iceberg costs $\tau_{ij} \ge 1$, Marshallian agglomeration $\alpha > 0$ ($A_i = \bar{A}_i L_i^\alpha$), and amenity congestion $\beta < 0$ ($a_i = \bar{a}_i L_i^\beta$). Equalized real utility $u_i = a_i \frac{w_i}{P_i} = \bar{u}$ yields:
    $$L_i = \bar{L} \frac{(\bar{a}_i w_i / P_i)^{-1/\beta}}{\sum_k (\bar{a}_k w_k / P_k)^{-1/\beta}}, \qquad P_i^{-\theta} = \sum_j \tau_{ji}^{-\theta} \left(\frac{w_j}{\bar{A}_j L_j^\alpha}\right)^{-\theta}$$
    Unique spatial equilibrium guaranteed when $\alpha + \beta \le \frac{\theta}{1+\theta}$ and $\alpha \le 1/\theta$.
- **Economic Intuition**: In international trade, unilateral tariffs provide terms-of-trade improvements (+0.33% US welfare) at the expense of trading partners and third parties (trade diversion). Bilateral trade wars eliminate terms-of-trade advantages, imposing deadweight losses on both nations (-1.11% USA, -0.69% China). Intermediate inputs create supply chain multipliers: tariffs cascade through input-output tables, multiplying welfare losses beyond pure Ricardian models. In geography, transport infrastructure (North-South corridor cost reduction -20%) expands market access, inducing mobile labor to migrate into connected regions (+0.35% population) until agglomeration benefits are checked by rising land/amenity congestion, lifting aggregate welfare (+0.0627%) with exact population conservation.
- **Worked Code**:
  ```text
  from puremacro.trade import CaliendoParroModel
  from puremacro.spatial import AllenArkolakisModel

  # 1. Caliendo-Parro tariff shock simulation
  cp_res = cp_model.solve_counterfactual(tariff_hat=tariff_hat)

  # 2. Allen-Arkolakis transport infrastructure corridor simulation
  aa_res = aa_model.simulate_transport_shock(origin=0, destination=1, cost_reduction=0.20)
  ```
- **Read the Output & Visualizations**: Hero figures: (1) 3-Panel trade plot (Baseline Trade Shares Matrix, Counterfactual Trade War Shares Matrix, Country Welfare Impact Bar Chart); (2) 2-Panel geographic network map with upgraded corridor and regional population/wage change bars; (3) Welfare impact comparison with vs without input-output linkages.
- **Your Turn Interactive Knobs & Asserts**: Knobs `tariff_rate_custom = 0.15`, `cost_reduction_custom = 0.20` with assertions `res_cp_custom.converged`, `res_aa_custom.converged`, `res_cp_custom.market_clearing_residual < 1e-6`, `res_aa_custom.labor_conservation_residual < 1e-12`, `res_aa_custom.welfare_pct > 0.0`.
- **Literature & Cross-References**: Eaton & Kortum (2002), Caliendo & Parro (2015), Allen & Arkolakis (2014), Costinot & Rodríguez-Clare (2014). User guide: [`docs/spatial_and_trade_ge.md`](spatial_and_trade_ge.md).

---

## Applied Macroeconomic Policy Showcases (puremacro 3.2)

Showcases `47` through `50` bridge theoretical macroeconometrics with applied central banking, treasury, and financial market policy practice.

### `47_applied_central_bank_policy_suite`
- **Source**: `notebooks/47_applied_central_bank_policy_suite.py` (Spanish: `_es.py`)
- **Key Capabilities**:
  - Real-time monetary policy stance evaluation against Taylor rules (1993, 1999) and inertia specifications.
  - Counterfactual policy simulation: evaluating macro trajectories under alternative policy paths.
  - Multi-period fan chart projection bands accounting for shock uncertainty and parameter posterior dispersion.
  - Textual sentiment and tone extraction from central bank statements (FOMC, ECB) via dictionary-based sentiment scoring.

### `48_applied_realtime_nowcasting_and_news`
- **Source**: `notebooks/48_applied_realtime_nowcasting_and_news.py` (Spanish: `_es.py`)
- **Key Capabilities**:
  - Dynamic Factor Model (DFM) nowcasting of quarterly GDP following Giannone, Reichlin & Small (2008).
  - Ragged-edge asynchronous data vintage ingestion handling mixed-frequency monthly and quarterly releases.
  - Release-day **news decomposition**: decomposing forecast revisions into surprise news components by data category (labor, output, surveys).
  - Mankiw-Shapiro (1986) news-versus-noise orthogonality tests on real-time data revisions.

### `49_applied_macroprudential_gar_and_stress`
- **Source**: `notebooks/49_applied_macroprudential_gar_and_stress.py` (Spanish: `_es.py`)
- **Key Capabilities**:
  - Growth-at-Risk (GaR) conditional quantile regression following Adrian, Boyarchenko & Giannone (2019).
  - Fitting parametric Azzalini skew-$t$ density distributions over predictive horizons to capture asymmetric downside tail risk.
  - Systemic risk connectedness networks via generalized forecast error variance decomposition (GFEVD) (Diebold & Yílmaz 2014).
  - Macroprudential capital buffer stress scenarios and probability-of-recession fan charts.

### `50_applied_fiscal_multipliers_and_debt_sustainability`
- **Source**: `notebooks/50_applied_fiscal_multipliers_and_debt_sustainability.py` (Spanish: `_es.py`)
- **Key Capabilities**:
  - Tri-method fiscal multiplier estimation on frozen datasets: Blanchard-Perotti SVAR, Romer-Romer narrative Local Projections, and Mertens-Ravn narrative LP-IV with effective first-stage $F$-statistics.
  - Stochastic sovereign Debt Sustainability Analysis (DSA) fan charts modeling joint growth, inflation, and interest-rate $(r - g)$ shocks.
  - Stress testing sovereign debt ratios under physical and transition climate risk scenarios.

---

## Bayesian DSGE & HANK Frontier Showcases (puremacro 3.0 & 3.1)

### `42_dsge_bayesian_estimation_and_diagnostics` (Flagship)
- **Source**: `notebooks/42_dsge_bayesian_estimation_and_diagnostics.py` (Spanish: `_es.py`)
- **Model**: Smets & Wouters (2007, AER) 7-observable, 7-shock US economy.
- **Key Capabilities**: Multi-algorithm mode search (`lbfgs`, `csminwel`, `cmaes`), `mode_check` curvature slices, MCMC sampling with Gelman-Rubin $\hat{R}$ diagnostics, Kalman smoother historical shock decomposition, fan chart dynamic forecasting, and Laplace / Geweke marginal data density comparison.

### `43_dsge_nuts_and_analytic_gradients`
- **Source**: `notebooks/43_dsge_nuts_and_analytic_gradients.py` (Spanish: `_es.py`)
- **Key Capabilities**: Exact analytic Kalman score recursion ($\nabla_\theta \ln L$) via generalized Sylvester solvers, No-U-Turn Sampler (NUTS) with dual averaging step-size adaptation, and energy BFMI diagnostics.

### `44_hank_sequence_space_bridge`
- **Source**: `notebooks/44_hank_sequence_space_bridge.py` (Spanish: `_es.py`)
- **Key Capabilities**: Heterogeneous-Agent Sequence-Space bridge parsed from Dynare `.mod` files (`hetagent_block`), stationary wealth distribution $\mathcal{D}^*(a)$, Fake-News Jacobians ($J_{C,r}, J_{C,Y}$), and nonlinear Broyden MIT transitions.

### `45_dsge_discretion_dsge_var_and_news_shocks`
- **Source**: `notebooks/45_dsge_discretion_dsge_var_and_news_shocks.py` (Spanish: `_es.py`)
- **Key Capabilities**: Optimal discretionary policy vs commitment (inflation/stabilization bias decomposition), Del Negro & Schorfheide (2004) DSGE-VAR($\lambda$) prior optimization, and anticipated news shock companion state-space augmentation.

### `46_dsge_particle_filtering_and_markov_switching`
- **Source**: `notebooks/46_dsge_particle_filtering_and_markov_switching.py` (Spanish: `_es.py`)
- **Key Capabilities**: Differentiable OccBin for NUTS, Foerster et al. (2016) Markov-Switching DSGE with closed-form analytical GIRFs, and vectorized sequential Monte Carlo particle filtering with stochastic volatility.

---

## Complete Showcase Catalog

| Notebook | Topic & Methodology | Spanish Twin |
|---|---|---|
| `00_whats_new_in_puremacro_3_0` | Milestone 3.0: Analytic Kalman gradients, NUTS HMC, and HANK Sequence-Space bridge | `00_whats_new_in_puremacro_3_0_es` |
| `00_whats_new_in_puremacro_2_0` | Milestone 2.0: Unified API (`lags`, `horizon`, `ci`), result objects, and exporters | `00_whats_new_in_puremacro_2_0_es` |
| `01_wealth_inequality` | Aiyagari & Huggett incomplete markets, permanent $\beta$-heterogeneity, Lorenz & Gini | `01_wealth_inequality_es` |
| `02_aggregate_shocks` | Krusell–Smith approximate aggregation, transition dynamics, representative-agent benchmark | `02_aggregate_shocks_es` |
| `03_life_cycle_and_demographics` | Finite-horizon life-cycle consumption-saving, cohort wealth by age, mortality weighting | `03_life_cycle_and_demographics_es` |
| `04_firm_dynamics` | Hopenhayn industry equilibrium with endogenous entry/exit and selection | `04_firm_dynamics_es` |
| `05_portfolios_and_preferences` | Two-asset portfolio choice, Epstein–Zin recursive utility, EGM vs VFI | `05_portfolios_and_preferences_es` |
| `06_svar_identification` | Structural VAR identification: Cholesky recursive ordering vs sign restrictions | `06_svar_identification_es` |
| `07_local_projections` | Jordà Local Projections with Newey-West HAC inference & state-dependent IRFs | `07_local_projections_es` |
| `08_garch_volatility` | Pure-NumPy GARCH(1,1) MLE & Engle Dynamic Conditional Correlation (DCC) | `08_garch_volatility_es` |
| `09_growth_at_risk` | Adrian-Boyarchenko-Giannone Growth-at-Risk via quantile autoregression & skew-$t$ fitting | `09_growth_at_risk_es` |
| `10_staggered_did` | Modern Difference-in-Differences: Callaway-Sant'Anna & Sun-Abraham estimators | `10_staggered_did_es` |
| `11_narrative_uncertainty` | Pure-NumPy Economic Policy Uncertainty (EPU) text index construction | `11_narrative_uncertainty_es` |
| `12_validation_gallery` | Validation scorecard comparing puremacro implementations against independent references | `12_validation_gallery_es` |
| `13_build_your_own_index` | Multi-recipe uncertainty construction: Text $\to$ EPU, Panel $\to$ JLN, Financial $\to$ FCI | `13_build_your_own_index_es` |
| `14_tax_multiplier_three_ways` | US tax multiplier: Blanchard-Perotti SVAR, Romer-Romer narrative LP, Mertens-Ravn LP-IV | `14_tax_multiplier_three_ways_es` |
| `15_lp_did` | Local Projections Difference-in-Differences (LP-DiD) under staggered treatment | `15_lp_did_es` |
| `16_regime_girf` | Generalized Impulse Response Functions (GIRF) for threshold VAR / Markov-switching VAR | `16_regime_girf_es` |
| `17_identification_spec_curve` | Identification specification curves for sensitivity analysis | `17_identification_spec_curve_es` |
| `18_beveridge_curve` | Beveridge curve dynamics, matching efficiency, and vacancy-unemployment shifts | `18_beveridge_curve_es` |
| `19_model_confidence_set` | Hansen-Lunde-Nason Model Confidence Set (MCS) for competitive forecasting | `19_model_confidence_set_es` |
| `20_unit_roots_with_power` | Elliott-Rothenberg-Stock DF-GLS and Ng-Perron high-power unit root tests | `20_unit_roots_with_power_es` |
| `21_dynare_vfi_dsl` | Declarative VFI DSL specification with Howard policy iteration acceleration | `21_dynare_vfi_dsl_es` |
| `22_continuous_time_hjb` | Achdou et al. (2022) Continuous-Time Hamilton-Jacobi-Bellman finite difference scheme | `22_continuous_time_hjb_es` |
| `23_aiyagari_endogenous_labor` | General equilibrium incomplete markets with endogenous labor supply choice | `23_aiyagari_endogenous_labor_es` |
| `24_synthetic_control` | Abadie et al. Synthetic Control Method with in-space and in-time placebo tests | `24_synthetic_control_es` |
| `25_frequency_connectedness` | Baruník & Křehlík frequency-domain variance decomposition spillover networks | `25_frequency_connectedness_es` |
| `26_cycles_and_bandpass` | Baxter-King, Christiano-Fitzgerald, and Beveridge-Nelson cycle filtering | `26_cycles_and_bandpass_es` |
| `27_garch_midas_macro_risk` | Engle-Ghysels-Sohn GARCH-MIDAS mixed-frequency volatility modeling | `27_garch_midas_macro_risk_es` |
| `28_weak_iv_anderson_rubin` | Anderson-Rubin weak-instrument robust confidence sets for LP-IV | `28_weak_iv_anderson_rubin_es` |
| `29_synthetic_did` | Arkhangelsky et al. Synthetic Difference-in-Differences (SDID) | `29_synthetic_did_es` |
| `30_narrative_bursts_and_transcripts` | Kleinberg burst detection on central bank speech transcripts | `30_narrative_bursts_and_transcripts_es` |
| `31_sequence_space_hank` | Auclert et al. (2021) Sequence-Space Jacobian method for HANK models | `31_sequence_space_hank_es` |
| `32_climate_macro_dice` | Nordhaus DICE integrated assessment model & optimal carbon tax policy | `32_climate_macro_dice_es` |
| `33_gdp_nowcasting_news` | Giannone-Reichlin-Small Dynamic Factor Model GDP nowcasting & news decomposition | `33_gdp_nowcasting_news_es` |
| `34_penalized_macro_forecasting` | Elastic Net and Adaptive Lasso for high-dimensional macroeconomic forecasting | `34_penalized_macro_forecasting_es` |
| `35_empirical_benchmark_replications` | Replication scorecard across SVAR, LP, and DiD benchmark literatures | `35_empirical_benchmark_replications_es` |
| `36_climate_sovereign_debt_risk` | Physical and transition climate risk transmission to sovereign debt sustainability | `36_climate_sovereign_debt_risk_es` |
| `37_central_bank_narrative_sentiment` | Central bank communication sentiment scoring and high-frequency monetary LP | `37_central_bank_narrative_sentiment_es` |
| `38_real_time_vintages_and_revisions` | Real-time QNA vintages across 45+ countries & news vs. noise revision tests | `38_real_time_vintages_and_revisions_es` |
| `39_multilingual_narrative_harvesting` | Multi-source narrative harvesting (50+ connectors) and multilingual macro scoring | `39_multilingual_narrative_harvesting_es` |
| `40_quarterly_national_accounts` | Three approaches to GDP accounting: rebasing, identities, and growth contributions | `40_quarterly_national_accounts_es` |
| `41_dynare_frontier_showcase` | Smets-Wouters (2007) native smoother & estimation, OccBin ZLB, Ramsey transitions | `41_dynare_frontier_showcase_es` |
| `42_dsge_bayesian_estimation_and_diagnostics` | Flagship Smets-Wouters Bayesian estimation, mode_check, MCMC, fan charts, MDD | `42_dsge_bayesian_estimation_and_diagnostics_es` |
| `43_dsge_nuts_and_analytic_gradients` | Bayesian DSGE via NUTS with exact analytic Kalman score recursion | `43_dsge_nuts_and_analytic_gradients_es` |
| `44_hank_sequence_space_bridge` | HANK Sequence-Space bridge from `.mod` files, Fake-News Jacobians, MIT transitions | `44_hank_sequence_space_bridge_es` |
| `45_dsge_discretion_dsge_var_and_news_shocks` | Discretionary policy vs commitment, DSGE-VAR prior optimization, news shocks | `45_dsge_discretion_dsge_var_and_news_shocks_es` |
| `46_dsge_particle_filtering_and_markov_switching` | Differentiable OccBin for NUTS, Markov-switching DSGE, particle filtering | `46_dsge_particle_filtering_and_markov_switching_es` |
| `47_applied_central_bank_policy_suite` | Monetary policy stance, counterfactual Taylor rules, fan charts, sentiment | `47_applied_central_bank_policy_suite_es` |
| `48_applied_realtime_nowcasting_and_news` | DFM nowcasting with ragged-edge vintages, release-day news decomposition | `48_applied_realtime_nowcasting_and_news_es` |
| `49_applied_macroprudential_gar_and_stress` | Growth-at-Risk quantile densities (skew-$t$), systemic connectedness networks | `49_applied_macroprudential_gar_and_stress_es` |
| `50_applied_fiscal_multipliers_and_debt_sustainability` | Multi-method fiscal multipliers, stochastic sovereign DSA under climate stress | `50_applied_fiscal_multipliers_and_debt_sustainability_es` |
| `51_continuous_projection_collocation_and_fem` | Continuous state-space DP: Chebyshev orthogonal collocation vs FEM Galerkin with borrowing kinks | `51_continuous_projection_collocation_and_fem_es` |
| `52_continuous_transition_mit_shocks` | Non-linear continuous transition dynamics & MIT shocks via coupled EGM-Young operator | `52_continuous_transition_mit_shocks_es` |
| `53_exact_analytic_ift_gradients` | Exact analytic IFT parameter Jacobians, adjoint sensitivities & structural GMM estimation | `53_exact_analytic_ift_gradients_es` |
| `54_deep_macro_pinns_high_dim` | High-dimensional dynamic models (10+ states) via pure-NumPy Physics-Informed Neural Networks | `54_deep_macro_pinns_high_dim_es` |
| `55_quantitative_spatial_and_trade_ge` | Quantitative spatial & trade general equilibrium: Caliendo-Parro tariffs & Allen-Arkolakis geography | `55_quantitative_spatial_and_trade_ge_es` |
| `56_implicit_hjb_and_continuous_kfe` | Continuous-time HJB implicit upwind solver, adjoint KFE stationary wealth density & Aiyagari GE | `56_implicit_hjb_and_continuous_kfe_es` |
| `57_multiconstraint_occbin_and_dml` | Multi-constraint OccBin ($M \ge 2$: ZLB + credit limits) & Double Machine Learning (DML-PLR) | `57_multiconstraint_occbin_and_dml_es` |
| `58_latin_america_realtime_macro` | Latin American real-time GDP vintages, portable `.pmz` cartridges & Mankiw-Shapiro news vs noise | `58_latin_america_realtime_macro_es` |

---

## Execution & Building Artifacts

Showcase notebooks can be executed and compiled to pre-rendered `.ipynb` artifacts using the built-in CLI:

```bash
# Execute and build all notebooks into .ipynb
python tools/build_notebooks.py

# Build a single notebook
python tools/build_notebooks.py 51_continuous_projection_collocation_and_fem

# Execute without modifying files (fails if any notebook raises an exception)
python tools/build_notebooks.py --check
```
