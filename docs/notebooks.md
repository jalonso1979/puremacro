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

## Trade tutorials (63–65)

Both English and Spanish editions are available as percent-format Python and executed notebooks.

- **`63_trade_wars_and_nash_tariffs`**: Hand-balanced two-country Hicksian tariff game with a fixed baseline, fixed matrix actions, consumption-normalized regret, final best-response checks and audited solver recovery. Boundary solutions and failed deviations are explicit; a Prisoner's Dilemma is tested, not assumed.
- `64_singularities_and_keller_pac`: spectral bounds on the bundled OECD table, a separate scalar toy fold, dense CGE continuation on a small synthetic calibration, and a deliberately planted stiff coordinate. These examples do not establish an economic fold at a particular elasticity.
- **`65_gvc_cascades_and_welfare_decomposition`**: Synthetic-data provenance, fixed-coefficient cost propagation and consistent-accounting Hicksian consumption EV/CV. Endpoint attribution separates purchaser prices, factor income and fiscal transfers. The historical TOT/Alloc/TariffRec decomposition and theorem certification remain unavailable.

See [structural validation status](STRUCTURAL_VALIDATION_STATUS.md) for supported models, failure contracts and evidence. Generated layouts are not empirical replications of OECD, FIGARO or EXIOBASE. The tutorial results are not contemporary tariff-policy forecasts.

## Latin America Nowcasting, Interactive DML & Quantitative Policy Simulators (puremacro 3.5)

Showcases `59` through `61` demonstrate `puremacro`'s frontier Latin American real-time nowcasting with Bańbura-Modugno (2014) news attribution and Berkowitz (2001) density evaluation, high-dimensional causal inference via Interactive Double Machine Learning (IRM & DML-IV) with regularized logistic coordinate descent and Montiel Olea & Pflueger effective $F$ diagnostics, and quantitative macroeconomic policy simulators across 77-country 11-sector OECD ICIO trade general equilibrium and sequence-space HANK monetary transmission.

### `59_latam_realtime_nowcast_and_news`
- **Source**: `notebooks/59_latam_realtime_nowcast_and_news.py` (Spanish: `_es.py`, compiled: `.ipynb`)
- **Motivating Economic Question**: How can central banks and economic research desks across Latin America nowcast quarterly GDP growth in real time from ragged-edge monthly indicator panels, decompose nowcast updates into release surprises and statistical revisions using the exact Bańbura & Modugno (2014) identity, and validate predictive density calibration using Berkowitz (2001) PIT uniformity tests?
- **Governing Math & Algorithms**:
  - **Dynamic Factor Model in State-Space Form**: Standardized indicators $X_t = \Lambda F_t + \xi_t$ driven by latent factors $F_t = \sum_{l=1}^p A_l F_{t-l} + u_t$. Estimated via Doz, Giannone, and Reichlin (2011) two-step principal components and Kalman filter/smoother over unbalanced ragged edges.
  - **Bańbura & Modugno (2014) Exact News Attribution**:
    $$\Delta \hat{y}_{t^*|v} \equiv \hat{y}_{t^*|v} - \hat{y}_{t^*|v-1} = \sum_{j \in \mathcal{I}_{\text{new}}} \omega_j \cdot I_{j, v} + \sum_{k \in \mathcal{I}_{\text{rev}}} \omega_k \cdot R_{k, v}$$
    Weights $\omega$ are determined by Kalman filter gain and state autocovariances, satisfying the exact identity with zero residual ($|\Delta \hat{y}_{t^*|v} - \sum \text{Impact}| < 10^{-10}$).
  - **Predictive Density Evaluation via Berkowitz (2001) LR Test**: Probability Integral Transform $p_t = \Phi((y_t - \mu_t)/\sigma_t)$ transformed to normal errors $z_t = \Phi^{-1}(p_t)$ with autoregression $(z_t - \mu) = \rho (z_{t-1} - \mu) + \varepsilon_t$ tested under $H_0: \mu = 0, \sigma_\varepsilon^2 = 1, \rho = 0$ yielding likelihood ratio statistic $\text{LR} \sim \chi^2(3)$.
  - **Latin America Data Cartridges**: Real-time connectors for Mexico (INEGI, Banxico) and Brazil (BCB) packaged into offline `.pmz` containers with SHA-256 integrity verification.
- **Economic Intuition**: Asynchronous publication schedules produce an unbalanced ragged edge. The Dynamic Factor Model estimates underlying activity despite missing observations. When new figures arrive, Bańbura-Modugno news attribution isolates whether updates reflect genuine economic surprises or retrospective agency revisions. Calibrated fan charts ensure credible density forecasts.
- **Worked Code**:
  ```python
  import numpy as np
  from puremacro.fetch.realtime import VintagePanel, pack_realtime_cartridge, load_realtime_cartridge
  from puremacro.nowcast import (
      DynamicFactorModel, realtime_nowcast, banbura_modugno_news,
      fan_chart, pit_uniformity_test,
  )

  # Dynamic Factor Model nowcasting with ragged edge
  dfm = DynamicFactorModel(n_factors=2, factor_lags=1)
  dfm.fit(X_train)
  nowcast_res = dfm.nowcast(target_series=y_gdp, X_eval=X_ragged)

  # Exact Bańbura-Modugno news decomposition
  news_res = banbura_modugno_news(dfm, y_target=y_gdp, vintage_old=p_old, vintage_new=p_new)
  news_res.total_revision, news_res.decomposition_error  # error < 1e-10

  # Density calibration evaluation
  pit_res = pit_uniformity_test(realizations=y_realized, means=mu_seq, stds=sigma_seq)
  pit_res.lr_stat, pit_res.p_value
  ```
- **Read the Output & Visualizations**: 4-Panel hero dashboard: (1) Real-time GDP nowcast path with fan chart ribbons; (2) Exact Bańbura-Modugno news attribution waterfall; (3) Indicator surprise contributions by sector; (4) Berkowitz PIT empirical CDF with 95% Kolmogorov-Smirnov confidence bands.
- **Your Turn Interactive Knobs & Asserts**: Knobs for factor dimensions, lag orders, and indicator subsets with assertion gates validating convergence and zero decomposition error.
- **Literature & Cross-References**: Bańbura & Modugno (2014), Giannone, Reichlin & Small (2008), Berkowitz (2001). User guides: [`docs/nowcast_latam_news.md`](nowcast_latam_news.md) and [`docs/real_time_latam.md`](real_time_latam.md).

### `60_interactive_dml_irm_and_iv`
- **Source**: `notebooks/60_interactive_dml_irm_and_iv.py` (Spanish: `_es.py`, compiled: `.ipynb`)
- **Motivating Economic Question**: How do economists estimate the causal impact of voluntary savings programs—such as 401(k) pension eligibility and participation—on net wealth accumulation when eligibility and take-up depend on dozens of non-linear socio-demographic confounders, and how can Double Machine Learning recover unbiased Average Treatment Effects (ATE), Treatment on the Treated (ATT), and Instrumental Variable (DML-IV) estimates without regularization bias?
- **Governing Math & Algorithms**:
  - **Interactive Regression Model (IRM)**: Potential outcomes $(Y(1), Y(0)) \perp D \mid X$ with conditional response functions $Y = g_0(D, X) + U$ and propensity score $D = m_0(X) + V$.
  - **Doubly Robust Neyman-Orthogonal Scores for ATE & ATT**:
    $$\psi_{\text{ATE}}(W; \theta, \eta) = g(1, X) - g(0, X) + \frac{D (Y - g(1, X))}{m(X)} - \frac{(1 - D) (Y - g(0, X))}{1 - m(X)} - \theta$$
    $$\psi_{\text{ATT}}(W; \theta, \eta) = \frac{D (Y - g(0, X))}{\mathbb{P}(D = 1)} - \frac{m(X)(1 - D)(Y - g(0, X))}{\mathbb{P}(D = 1)(1 - m(X))} - \theta$$
  - **Pure-NumPy Regularized Logistic Coordinate Descent**: Soft-thresholding updates $\beta_j^{\text{new}} = S(c_j \beta_j - g_j, \lambda)/c_j$ using surrogate curvature upper bound $c_j = \frac{1}{4N} \sum_i X_{ij}^2$, with BIC penalty selection and automatic overlap trimming to $[\varepsilon, 1-\varepsilon]$.
  - **Double ML Instrumental Variables (DML-IV)**: Cross-fitted 2SLS on orthogonal residuals with Montiel Olea & Pflueger (2013) effective $F_{\text{eff}}$ weak-instrument diagnostics.
- **Economic Intuition**: Heterogeneous treatment effects and confounding invalidate constant-effect linear regressions, while naive Lasso introduces severe attenuation bias. DML-IRM pairs regularized non-linear machine learning with Neyman-orthogonal scores and $K$-fold cross-fitting, preserving $\sqrt{N}$-consistent causal inference. When actual program participation is endogenous, DML-IV exploits institutional eligibility as an instrument, verifying identification strength via effective $F$.
- **Worked Code**:
  ```python
  import numpy as np
  from puremacro.causal import (
      DoubleMLIRM, DoubleMLIV, dml_irm, dml_iv,
      LogisticCoordinateDescent,
  )

  # DML-IRM: Average Treatment Effect & Treatment on the Treated
  irm_ate = DoubleMLIRM(ml_g="lasso", ml_m="logistic", n_folds=5, score="ATE", trimming_threshold=0.01)
  res_ate = irm_ate.fit(Y=net_assets, D=e401, X=df_controls)
  res_ate.theta, res_ate.se, res_ate.n_trimmed

  # DML-IV: Endogenous treatment with Montiel Olea-Pflueger F-statistic
  res_iv = dml_iv(Y=net_assets, D=p401, Z=e401, X=df_controls, n_folds=5)
  res_iv.theta, res_iv.first_stage_effective_f, res_iv.weak_instrument
  ```
- **Read the Output & Visualizations**: 4-Panel hero dashboard: (1) Propensity score overlap distributions with trimming thresholds; (2) Cross-estimator causal benchmark (OLS vs Naive Lasso vs DML-ATE vs DML-ATT vs DML-IV); (3) Montiel Olea-Pflueger first-stage orthogonal residual scatter; (4) Coordinate descent regularization tuning path.
- **Your Turn Interactive Knobs & Asserts**: Knobs for trimming threshold $\varepsilon$, fold count $K$, and penalization penalties with assertions validating bounds and confidence intervals.
- **Literature & Cross-References**: Chernozhukov et al. (2018), Belloni, Chernozhukov & Hansen (2014), Montiel Olea & Pflueger (2013). User guide: [`docs/dml_irm_iv.md`](dml_irm_iv.md).

### `61_quantitative_policy_simulators`
- **Source**: `notebooks/61_quantitative_policy_simulators.py` (Spanish: `_es.py`, compiled: `.ipynb`)
- **Motivating Economic Question**: How do bilateral trade disputes and tariff escalations propagate across global input-output linkages to alter terms of trade, sectoral allocation, and real wages in general equilibrium, and how does household wealth and income heterogeneity govern the transmission of monetary policy between hand-to-mouth consumers and unconstrained asset holders?
- **Governing Math & Algorithms**:
  - **Ricardian Trade Policy General Equilibrium (Caliendo & Parro 2015)**: Exact hat algebra on multi-country multi-sector input-output tables:
    $$\hat{\pi}_{ni}^j = \left( \frac{\hat{\kappa}_{ni}^j \hat{c}_i^j}{\hat{P}_n^j} \right)^{-\theta_j}, \quad \hat{c}_i^j = \hat{w}_i^{\gamma_i^j} \prod_{k=1}^J (\hat{P}_i^k)^{\gamma_i^{j, k}}$$
    Solves for equilibrium wage adjustments $\{\hat{w}_i\}$ and goods market clearing, decomposing national welfare changes into terms of trade, input-output efficiency, and tariff revenue.
  - **Sequence-Space Monetary Transmission & KMV Decomposition (Kaplan et al. 2018; Auclert et al. 2021)**: Linearized dynamic consumption response $d\mathbf{C} = \mathbf{J}^{C, r} d\mathbf{r} + \mathbf{J}^{C, Y} d\mathbf{Y}$. In RANK, $\mathbf{J}^{C, Y} = 0$ (100% direct intertemporal substitution). In HANK, hand-to-mouth consumers with steep MPCs drive large indirect general equilibrium labor income multipliers:
    $$\text{Indirect Share} = \frac{(\mathbf{J}^{C, Y} d\mathbf{Y})_0}{dC_0} \times 100\%$$
- **Economic Intuition**: Tariffs generate multi-sector cost cascades through imported intermediates and induce trade diversion toward third countries. Caliendo-Parro exact hat algebra computes these general equilibrium adjustments without estimating structural parameters. In monetary policy, empirical wealth concentration means liquidity-constrained households drive aggregate consumption through indirect income channels rather than direct interest-rate intertemporal smoothing.
- **Worked Code**:
  ```python
  import numpy as np
  from puremacro.trade.data import load_icio_data
  from puremacro.models import (
      TradePolicySimulator, MonetaryTransmissionSimulator,
  )

  # 1. 77-Country 11-Sector OECD ICIO Trade Policy GE Simulation
  trade_sim = TradePolicySimulator(load_icio_data(return_structured=True))
  trade_res = trade_sim.simulate_tariff_shock(
      tariffs={"USA": {"CHN": {"MANU": 0.25}}, "CHN": {"USA": {"MANU": 0.25}}},
  )
  trade_res.welfare_change["USA"], trade_res.welfare_change["MEX"]

  # 2. Sequence-Space HANK vs RANK Monetary Transmission
  mon_sim = MonetaryTransmissionSimulator()
  mon_res = mon_sim.simulate_monetary_shock(r_path=dr_seq)
  mon_res.direct_effect, mon_res.indirect_effect, mon_res.indirect_share
  ```
- **Read the Output & Visualizations**: 4-Panel hero dashboard: (1) General equilibrium welfare impacts across countries; (2) Sectoral bilateral trade diversion and terms-of-trade shifts; (3) HANK vs RANK consumption impulse response trajectories; (4) Kaplan-Moll-Violante direct vs indirect consumption decomposition across wealth deciles.
- **Your Turn Interactive Knobs & Asserts**: Knobs for retaliatory tariff rates, trade elasticities, and hand-to-mouth population shares with assertion gates validating goods market clearing and consumption decomposition.
- **Literature & Cross-References**: Caliendo & Parro (2015), Kaplan, Moll & Violante (2018), Auclert et al. (2021). User guide: [`docs/policy_simulators.md`](policy_simulators.md).

---

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
    Replacing one row with normalization ensures mass conservation to the notebook's gate ($|\sum g_i \Delta a_i - 1| \le 10^{-12}$; the run above reaches $1.1 \times 10^{-16}$). On a non-uniform grid the null vector is node *mass* and `solve_kfe_achdou` divides it by the cell-width quadrature weights, so what comes back is a *density*.
  - **Continuous Aiyagari General Equilibrium**:
    $$K^s(r) = \sum_{j, i} a_i g_{j, i} \Delta a_i = K^d(r) = \left( \frac{r + \delta}{\alpha \bar{Z}} \right)^{\frac{1}{\alpha - 1}}$$
- **Economic Intuition**: Discrete-time models require fine time aggregation and suffer from lower-bound binding jitter. Continuous-time formulation replaces discrete choice with a smooth drift function $s(a, z)$. Upwind finite differences mirror the physical direction of asset flows: households accumulating assets look forward ($v_{i+1} - v_i$), while households decumulating assets look backward ($v_i - v_{i-1}$), avoiding non-physical oscillation. Because the transition generator $A$ is an infinitesimal Markov generator, its adjoint $A^\top$ yields the exact stationary wealth density $g(a, z)$ in a single linear solve, preserving mass conservation without Monte Carlo simulation noise.
- **Worked Code**:
  ```python
  import numpy as np
  from puremacro.vfi import solve_hjb_achdou, solve_aiyagari_continuous_hjb

  # Continuous-time HJB implicit solver (note the parameter names:
  # r_rate / w_rate / rho_val / gamma_r / Na, not r / w / rho / gamma / n_a)
  sol = solve_hjb_achdou(
      r_rate=0.03, w_rate=1.0, rho_val=0.05, gamma_r=2.0,
      Na=100, a_min=0.0, a_max=30.0, tol=1e-8, max_iter=100,
  )
  sol.converged, sol.n_iter          # True, 9
  sol.V.shape, sol.c_policy.shape    # (100, 2), (100, 2)
  sol.mass_residual                  # 1.11e-16  (KFE gate: <= 1e-12)

  da = sol.a_grid[1] - sol.a_grid[0]
  mass = float(np.sum(sol.g_dist * da))   # 1.0 — the density is `g_dist`

  # Continuous Aiyagari general equilibrium: bisection on r until Ks(r) = Kd(r)
  ge = solve_aiyagari_continuous_hjb(
      alpha=0.33, delta=0.05, rho_val=0.05, gamma_r=2.0,
      Na=40, a_max=25.0, tol_ge=1e-4, max_iter_ge=30,
  )
  ge.converged, ge.r_star, ge.K_star      # True, 0.018879, 6.2190
  ```
  Since 3.4.0 `solve_hjb_achdou` adds the two-state Poisson income-switching
  term the 3.3.0 explicit solver omitted (symmetric generator, default
  $\lambda = 0.1$), so the productivity states are no longer independent
  deterministic-income problems; `tol_ge` governs the general-equilibrium
  bisection and `ge.converged` reports $|K^s - K^d| \le$ `tol_ge`.
- **Read the Output & Visualizations**: 4-Panel hero dashboard: (1) Optimal consumption policies $c_j(a)$ by income state, with liquidity-constrained MPC kinks near $\underline{a}$; (2) Savings drift $s(a, z) = r a + w z - c(a, z)$, showing decumulation for low-income and accumulation for high-income households; (3) Stationary wealth distribution from the adjoint KFE, with the characteristic precautionary mass spike at the borrowing constraint; (4) Aiyagari asset-market clearing, $K^s(r)$ against $K^d(r)$ around $r^*$.
- **Your Turn Interactive Knobs & Asserts**: Knobs `rho_custom = 0.05`, `gamma_custom = 2.0`, `Na_custom = 50`, `a_max_custom = 30.0`, re-solved with `solve_hjb_achdou`; assertion gates `sol_custom.converged`, `sol_custom.n_iter <= 25`, `sol_custom.mass_residual <= 1e-12` and `abs(mass_custom - 1.0) <= 1e-12` where `mass_custom = np.sum(sol_custom.g_dist * da_custom)`.
- **Literature & Cross-References**: Achdou, Han, Lasry, Lions & Moll (2022), Aiyagari (1994), Huggett (1993). User guides: [`docs/vfi_hjb_continuous.md`](vfi_hjb_continuous.md) (the continuous-time solver) and [`docs/vfi_continuous_equilibrium.md`](vfi_continuous_equilibrium.md) (the discrete-time histogram route).

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
- **Economic Intuition**: When a severe contractionary shock hits, households cut borrowing against their credit limit while the central bank lowers interest rates to zero. If both constraints bind concurrently (Regime 3), the economy experiences severe non-linear amplification: monetary policy cannot provide accommodation while credit-constrained households cannot borrow to smooth consumption. In the empirical stage the notebook contrasts three estimators on a synthetic panel with $N = 500$, $p = 30$ controls and a true effect $\theta_0 = 1.75$: OLS on all controls ($\hat\theta = 1.686$, 95% CI $[1.570, 1.801]$), a naive Lasso that penalizes the policy variable $D$ alongside $X$ ($\hat\theta = 1.582$ — shrinkage attenuation, and no confidence interval worth quoting), and DML ($\hat\theta = 1.699$, 95% CI $[1.581, 1.817]$). The naive Lasso is the estimator that fails; what DML buys is the right to use a penalized nuisance fit *and* still report a valid interval, via cross-fitting plus the Neyman-orthogonal score. Note the scope of that guarantee: the shipped nuisance learners (`lasso`, `ridge` — there is no elastic net) are **linear in the columns of $X$ you supply**, so DML removes confounding that is linear in those columns. Non-linear confounding needs a basis expansion appended to $X$, or a custom learner passed through `learner=`.
- **Worked Code**:
  ```python
  import numpy as np
  from puremacro.dsge import (
      build_dynare, OccBinConstraint, solve_multiconstraint_occbin, OccBinResult,
  )
  from puremacro.causal import dml_plr, DoubleMLPLR, LassoCoordinateDescent, RidgeGCV

  # 1. Multi-constraint OccBin: one constrained model per constraint, paired by key
  res_occ = solve_multiconstraint_occbin(
      m_unconstrained=m_ref,
      m_constrained_dict={"zlb": m_zlb, "borrowing": m_borr},
      shock_seq=shocks_mat,                      # (horizon, n_shocks)
      constraints={"zlb": c_zlb, "borrowing": c_borr},
      horizon=40,
  )
  regimes = np.asarray(res_occ.regimes)          # bitmask: 1 = ZLB, 2 = cap, 3 = both
  res_occ.converged, regimes[:8]                 # True, [3 1 1 0 0 0 0 0]

  # 2. DML partially linear regression (N = 500, p = 30 controls, theta_0 = 1.75)
  res_dml_lasso = dml_plr(Y_outcome, D_treat, X_mat, n_folds=5,
                          learner="lasso", random_state=42)
  res_dml_ridge = dml_plr(Y_outcome, D_treat, X_mat, n_folds=5,
                          learner="ridge", random_state=42)
  res_dml_lasso.theta, res_dml_lasso.ci_lower, res_dml_lasso.ci_upper
  #                                              # 1.6989, 1.5813, 1.8165
  ```
  `DoubleMLPLR(n_folds=5, learner="lasso").fit(Y, D, X)` is the class form of
  the same estimator — the data go to `.fit()`, not to the constructor, and the
  keyword is `learner=`, not `estimator=`. There is no `OccBinMultiConstraint`
  class and no `puremacro.dml` module; the multi-constraint solver is
  `puremacro.dsge.solve_multiconstraint_occbin` and DML lives in
  `puremacro.causal`.
- **Read the Output & Visualizations**: 4-Panel hero dashboard: (1) Interest-rate trajectory, OccBin against the unconstrained linear simulation; (2) Active structural regime sequence across time (the bitmask); (3) Causal policy estimator comparison with confidence bands — true effect, OLS, naive Lasso, DML-Lasso, DML-Ridge; (4) Neyman-orthogonalized residuals with the fitted policy slope.
- **Your Turn Interactive Knobs & Asserts**: Knobs `shock_g_custom = -0.06`, `shock_b_custom = 0.05`, `b_bar_custom = 0.020`, `n_folds_custom = 5`, `learner_custom = "lasso"` with assertion gates `res_custom_occ.converged` and `res_custom_dml.ci_lower <= theta_true <= res_custom_dml.ci_upper`.
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
  - **Vintages here are snapshot dates.** Banxico, INEGI, BCB and BCCh publish only the edition current at the moment of the request; they overwrite in place. The notebook's panel is therefore synthetic, and against the live connectors a vintage date is the day the series was captured, with revision history accumulating only across repeated captures on different days.
- **Economic Intuition**: Preliminary releases published 30–45 days after quarter-end rely on incomplete survey samples and statistical nowcasting models. As hard data (tax filings, balance sheets) arrive, statistical agencies revise historical figures. In emerging Latin American economies (Mexico, Brazil, Chile), understanding whether revisions are news or noise determines whether policymakers should respond immediately to preliminary growth signals or discount them as volatile measurement noise. Empirical results show revisions are news-dominated, confirming Latin American central banks produce rational real-time estimates.
- **Worked Code**:
  ```python
  from puremacro.fetch.realtime import (
      VintagePanel,
      pack_realtime_cartridge,
      load_realtime_cartridge,
  )

  # `df_raw` is the tidy long frame: country, variable, date, vintage, value,
  # provider, series_id, units
  panel_raw = VintagePanel(df_raw)

  # Package into an offline .pmz cartridge and load it back with SHA-256 checking
  pack_realtime_cartridge(
      panel_raw, cartridge_file,
      source="Banxico, INEGI, BCB, BCCh Regional Real-Time Ecosystem",
      vintage="2026-04-01",
  )
  panel = load_realtime_cartridge(cartridge_file, verify=True)

  # Revision triangles and the Mankiw-Shapiro pair are methods of the panel
  panel.coverage()                            # what came back, per (country, variable)
  panel.as_of("2025-06-01")                   # the information set on that date
  panel.triangle("MEX", "gdp_real")           # reference periods x vintages
  panel.revisions("MEX", "gdp_real")          # preliminary / final / revision
  ms = panel.news_or_noise("MEX", "gdp_real") # MankiwShapiroResult
  ms.verdict, ms.beta_on_preliminary, ms.p_beta_on_final
  panel.news_or_noise_panel()                 # every series, one tidy table
  ```
  There is no `puremacro.realtime` module and no `build_revision_triangle` /
  `mankiw_shapiro_test` / `export_realtime_cartridge` function: the cartridge
  helpers are `pack_realtime_cartridge` / `load_realtime_cartridge` in
  `puremacro.fetch.realtime`, and the revision machinery is on `VintagePanel`
  (with the underlying test in `puremacro.vintages.mankiw_shapiro`).
- **Read the Output & Visualizations**: 4-Panel hero dashboard: (1) Latin American central-bank policy rates; (2) Triangular heatmap of the revision triangle $\mathbf{T}[t, v]$ for Mexican real GDP; (3) Preliminary against final GDP estimates across reference periods; (4) Mankiw-Shapiro scatter of revisions on the preliminary release, annotated with the verdict.
- **Your Turn Interactive Knobs & Asserts**: Knobs `country_custom = "MEX"`, `var_custom = "gdp_real"`, `as_of_custom = "2025-06-01"`, `signif_custom = 0.05` with assertion gates `country_custom in panel.countries`, `var_custom in panel.variables`, `not custom_asof.empty`, `len(custom_rev) > 0` and `hasattr(custom_ms, "verdict")`.
- **Literature & Cross-References**: Mankiw & Shapiro (1986), Croushore & Stark (2001), Faust, Rogers & Wright (2005). User guides: [`docs/real_time_latam.md`](real_time_latam.md) (the four connectors) and [`docs/real_time_data.md`](real_time_data.md).

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
| `59_latam_realtime_nowcast_and_news` | Latin America DFM GDP nowcasting, Bańbura-Modugno news decomposition & Berkowitz density PIT | `59_latam_realtime_nowcast_and_news_es` |
| `60_interactive_dml_irm_and_iv` | Interactive Double ML: IRM (ATE/ATT) with regularized logistic CD, propensity overlap & DML-IV | `60_interactive_dml_irm_and_iv_es` |
| `61_quantitative_policy_simulators` | Quantitative policy simulators: 77-country OECD ICIO trade GE & sequence-space HANK monetary transmission | `61_quantitative_policy_simulators_es` |
| `62_flexible_trade_cge` | Flexible trade GE: Nested CES technology, Stone-Geary LES preferences & Atkeson-Burstein markups | `62_flexible_trade_cge_es` |
| `63_trade_wars_and_nash_tariffs` | Hicksian tariff games: fixed baseline and actions, best-response regret and audited equilibrium recovery | `63_trade_wars_and_nash_tariffs_es` |
| `64_singularities_and_keller_pac` | Singularities, fold bifurcations & Keller's pseudo-arclength continuation (PAC): bordered Jacobians, Cyprus modal clamping & error bounds | `64_singularities_and_keller_pac_es` |
| `65_gvc_cascades_and_welfare_decomposition` | Data provenance, cost propagation and Hicksian EV/CV with price, factor-income and fiscal-transfer attribution | `65_gvc_cascades_and_welfare_decomposition_es` |

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
