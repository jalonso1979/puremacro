# puremacro showcase notebooks

Static, publication-quality notebooks illustrating `puremacro` across the major
heterogeneous-agent paradigms (`puremacro.vfi`), the empirical-econometrics
tools (SVAR / LP / GARCH / GaR / DiD), and text-as-data uncertainty indices
(`puremacro.narrative`). **Edit the `.py` (jupytext percent) source,
not the `.ipynb`** — the `.ipynb` is a build artifact, regenerated with outputs.

| Notebook | Shows |
|---|---|
| `01_wealth_inequality` | Aiyagari + Huggett + permanent-type β-heterogeneity; Lorenz/Gini |
| `02_aggregate_shocks` | Krusell–Smith approximate aggregation; transition path; rep-agent growth |
| `03_life_cycle_and_demographics` | Finite-horizon life-cycle; cohort wealth by age; mortality weighting |
| `04_firm_dynamics` | Hopenhayn entry/exit, selection, comparative statics |
| `05_portfolios_and_preferences` | Two-asset portfolios; Epstein–Zin; EGM vs VFI |
| `06_svar_identification` | SVAR: Cholesky vs sign-restriction identification of a planted monetary shock |
| `07_local_projections` | Jordà LP-HAC + state-dependent (recession vs expansion) IRFs |
| `08_garch_volatility` | GARCH(1,1) MLE + Engle DCC time-varying correlation (pure-numpy) |
| `09_growth_at_risk` | Quantile-AR Growth-at-Risk fan + skew-t conditional density (ABG) |
| `10_staggered_did` | Staggered DiD: Callaway-Sant'Anna + Sun-Abraham event study |
| `11_narrative_uncertainty` | Build an EPU/MPU text-uncertainty index from a corpus (pure-numpy, no API key/LLM) |
| `12_validation_gallery` | Validation scorecard + coverage figure + puremacro-vs-reference overlays (statsmodels/scipy goldens) |
| `13_build_your_own_index` | Build four uncertainty indices from one toolkit — text→EPU, macro panel→JLN-style, financial→FCI, cross-section→comovement premium |
| `14_tax_multiplier_three_ways` | The US tax multiplier three ways on one frozen dataset — Blanchard-Perotti SVAR (−1), Romer-Romer narrative LP (−3), Mertens-Ravn narrative-as-instrument (between, with the effective first-stage F) — plus a spec curve showing identification, not estimation, drives the answer |
| `15_lp_did` | DiD meets local projections — why naive TWFE event studies break under staggered heterogeneous adoption, and LP-DiD (Dube-Girardi-Jordà-Taylor) as the fix; side-by-side with Callaway-Sant'Anna and Sun-Abraham agreeing on one panel |
| `16_regime_girf` | State-dependent transmission done right — Koop-Pesaran-Potter generalized IRFs for TVAR/MS-VAR with endogenous regime switching; the frozen-regime IRF overstates stress-state losses by ~33% |
| `17_identification_spec_curve` | Identification specification curve: how the identifying assumption, not the estimator, drives the answer |
| `18_beveridge_curve` | Beveridge curve: vacancies, unemployment and matching efficiency shifts |
| `19_model_confidence_set` | Hansen-Lunde-Nason model confidence set for competing forecasts |
| `20_unit_roots_with_power` | Elliott-Rothenberg-Stock DF-GLS unit root test with superior local power |
| `21_dynare_vfi_dsl` | Declarative Dynare-like Automated VFI specification with Howard acceleration & panel inequality simulation |
| `22_continuous_time_hjb` | Achdou et al. (2022) Continuous-Time HJB finite difference upwind scheme for consumption-saving models |
| `23_aiyagari_endogenous_labor` | Aiyagari GE incomplete markets model with endogenous intra-temporal labor choice |
| `24_synthetic_control` | Abadie et al. (2010) Synthetic Control Method for causal policy evaluation with donor placebos |
| `25_frequency_connectedness` | Baruník & Křehlík (2018) frequency-domain variance decomposition spillover networks |
| `26_cycles_and_bandpass` | Baxter-King, Christiano-Fitzgerald & Beveridge-Nelson vs Hamilton & HP cycle filtering |
| `27_garch_midas_macro_risk` | Engle-Ghysels-Sohn (2013) GARCH-MIDAS two-component mixed-frequency macro volatility |
| `28_weak_iv_anderson_rubin` | Weak-IV robust Anderson-Rubin (1949) quadratic HAC confidence sets for LP-IV |
| `29_synthetic_did` | Arkhangelsky et al. (2021) Synthetic DiD combining SCM unit weights and DiD time weights |
| `30_narrative_bursts_and_transcripts` | Kleinberg (2002) burst detection & high-frequency communication density |
| `31_sequence_space_hank` | Auclert et al. (2021) Sequence-Space Jacobian method for HANK models |
| `32_climate_macro_dice` | Nordhaus DICE integrated assessment model & optimal carbon taxation |
| `33_gdp_nowcasting_news` | Giannone-Reichlin-Small (2008) DFM GDP nowcasting & news decomposition |
| `34_penalized_macro_forecasting` | Elastic Net & Adaptive Lasso penalized macroeconomic forecasting |
| `35_empirical_benchmark_replications` | Benchmark replication scorecard across SVAR, LP, and DiD literatures |
| `36_climate_sovereign_debt_risk` | Physical/transition climate risk transmission to sovereign debt sustainability |
| `37_central_bank_narrative_sentiment` | Central bank communication tone extraction & high-frequency monetary LP |
| `38_real_time_vintages_and_revisions` | Real-time QNA vintages across 45+ countries, revision triangles $(T \times V)$, & Mankiw-Shapiro (1986) news vs. noise test |
| `39_multilingual_narrative_harvesting` | Multi-source narrative harvesting (50+ connectors), 8-language macro scoring, realization lags, & structured policy classification |
| `40_quarterly_national_accounts` | Three approaches to GDP in one panel: one price reference year (`qna_rebase`), the expenditure/output/income identities scored inside their own flows (`qna_identity`), growth decomposed with previous-period nominal weights (`qna_contributions`) |
| `41_dynare_frontier_showcase` | Smets-Wouters (2007) from Pfeifer's `.mod`: native 2.6.0 `.smoother()` state & shock extraction on bundled US data (`_sw07_data.csv`), native `.estimate()` MCMC, FEVD, OccBin ZLB piecewise-linear regime, and perfect-foresight Ramsey transition (paired Spanish edition: `41_dynare_frontier_showcase_es`) |
| `42_dsge_bayesian_estimation_and_diagnostics` | Flagship Smets-Wouters (2007) Bayesian DSGE estimation: multi-algorithm mode search (`lbfgs`, `csminwel`, `cmaes`), visual mode diagnostics (`mode_check` curvature slices), MCMC sampling with Gelman-Rubin convergence diagnostics, Kalman smoother & historical structural shock decomposition, fan chart forecasting with confidence bands, Laplace & Geweke modified harmonic mean marginal data density (MDD), and Bayesian model comparison (paired Spanish edition: `42_dsge_bayesian_estimation_and_diagnostics_es`) |
| `43_dsge_nuts_and_analytic_gradients` | Bayesian DSGE estimation via No-U-Turn Sampler (NUTS) with exact analytic Kalman score recursion ($\nabla_\theta \ln L$ via generalized Sylvester solver), dual averaging, diagonal mass adaptation, and MCMC diagnostics (paired Spanish edition: `43_dsge_nuts_and_analytic_gradients_es`) |
| `44_hank_sequence_space_bridge` | Heterogeneous-Agent (HANK) Sequence-Space Bridge from `.mod` files: `hetagent_block` specification, stationary wealth distribution $\mathcal{D}^*(a)$, Fake-News Jacobians ($J_{C,r}, J_{C,Y}$), and linear/nonlinear Broyden MIT transitions (paired Spanish edition: `44_hank_sequence_space_bridge_es`) |
| `45_dsge_discretion_dsge_var_and_news_shocks` | Frontier DSGE: Discretionary optimal policy vs. commitment (Dennis 2007; Oudiz & Sachs 1985), DSGE-VAR(λ) hybrid estimation with conjugate prior for model misspecification (Del Negro & Schorfheide 2004), anticipated news shocks via state-space augmentation (Beaudry & Portier 2006), Dynare macro preprocessor (@#define, @#for, @#include), formal rank identification (Iskrev 2010; Komunjer & Ng 2011), and interactive IRF slider widgets (paired Spanish edition: `45_dsge_discretion_dsge_var_and_news_shocks_es`) |
| `46_dsge_particle_filtering_and_markov_switching` | Frontier DSGE: Vectorized Sequential Monte Carlo particle filtering (Gordon et al. 1993; Fernández-Villaverde & Rubio-Ramírez 2007) for nonlinear DSGEs with stochastic volatility, Markov-switching DSGE perturbation (Foerster et al. 2016) with coupled quadratic matrix equations and generalized IRFs (GIRF), differentiable OccBin with smooth relaxation for NUTS under ZLB, and two-asset HANK bridge (paired Spanish edition: `46_dsge_particle_filtering_and_markov_switching_es`) |
| `47_applied_central_bank_policy_suite` | Applied Central Bank Policy Suite: Empirical Taylor rules (Clarida-Galí-Gertler), monetary stance decomposition, counterfactual policy path simulations, multi-horizon probabilistic projection fan charts, and central bank statement narrative sentiment extraction (paired Spanish edition: `47_applied_central_bank_policy_suite_es`) |
| `48_applied_realtime_nowcasting_and_news` | Applied Real-Time Nowcasting: Dynamic Factor Model (DFM; Giannone-Reichlin-Small 2008) for quarterly GDP nowcasting from mixed-frequency monthly indicators with asynchronous ragged edges, news release surprise decomposition, and Mankiw-Shapiro (1986) news vs. noise testing (paired Spanish edition: `48_applied_realtime_nowcasting_and_news_es`) |
| `49_applied_macroprudential_gar_and_stress` | Applied Macroprudential Monitor: Growth-at-Risk (GaR; Adrian, Boyarchenko & Giannone 2019) multi-quantile regressions, Azzalini skew-t predictive density fitting, Diebold-Yilmaz (2012) systemic connectedness & GFEVD spillover networks across banking institutions, and macroprudential stress testing for CCyB buffer calibration (paired Spanish edition: `49_applied_macroprudential_gar_and_stress_es`) |
| `50_applied_fiscal_multipliers_and_debt_sustainability` | Applied Fiscal Policy & Debt Sustainability: Multi-method fiscal multiplier estimation (Blanchard-Perotti SVAR, Romer-Romer narrative LP, and Mertens-Ravn LP-IV), spec curve comparison, and stochastic sovereign Debt Sustainability Analysis (DSA) fan charts under growth, interest rate, and primary deficit stress scenarios (paired Spanish edition: `50_applied_fiscal_multipliers_and_debt_sustainability_es`) |
| `51_continuous_projection_collocation_and_fem` | Continuous State-Space Projection: Orthogonal Chebyshev polynomial collocation (Euler residual projection & Bellman value iteration) vs. Finite Element Method (FEM / Galerkin with piecewise-linear hat basis functions) across smooth neoclassical growth (Brock-Mirman analytical benchmark with spectral convergence), occasionally binding borrowing constraints (Gibbs ringing resolution), stochastic multi-state Markov shocks, and multi-backend hardware acceleration (NumPy, Numba, Apple Silicon MLX, NVIDIA CuPy) (paired Spanish edition: `51_continuous_projection_collocation_and_fem_es`) |
| `52_continuous_transition_mit_shocks` | Continuous Transition Dynamics & MIT Shocks: Non-linear macroeconomic transition path of continuous wealth distributions $\mu_t(k, z)$ following unexpected aggregate shocks (monetary tightening, TFP contraction) in incomplete-markets economies; coupled backward continuous EGM policy recursion, forward time-dependent Young (2010) density evolution ($\mu_{t+1} = T_t^* \mu_t$), and sequence-space Broyden market clearing $\int k \, d\mu_t = K^d(r_t)$; hero plots of time-varying capital stock $K_t$, interest rate $r_t$, and 3D distribution surface $\mu_t(k)$ ([EN .py](52_continuous_transition_mit_shocks.py), [ES .py](52_continuous_transition_mit_shocks_es.py), [EN .ipynb](52_continuous_transition_mit_shocks.ipynb), [ES .ipynb](52_continuous_transition_mit_shocks_es.ipynb)) |
| `53_exact_analytic_ift_gradients` | Exact Analytic IFT Gradients & Structural Estimation: Gradient-based structural estimation (SMM / GMM) of continuous dynamic macro models without finite-difference noise; analytical derivation of the Implicit Function Theorem on continuous projection systems $\nabla_\theta c^*(\theta) = - [\nabla_c R(c^*; \theta)]^{-1} \nabla_\theta R(c^*; \theta)$, adjoint operators for machine-precision sensitivities in a single linear solve, and 60×+ execution speedup over numerical finite differences for preference $(\beta, \sigma)$ calibration ([EN .py](53_exact_analytic_ift_gradients.py), [ES .py](53_exact_analytic_ift_gradients_es.py), [EN .ipynb](53_exact_analytic_ift_gradients.ipynb), [ES .ipynb](53_exact_analytic_ift_gradients_es.ipynb)) |
| `54_deep_macro_pinns_high_dim` | Deep Macro & Physics-Informed Neural Networks (PINNs): Solving high-dimensional dynamic macroeconomic models with 10+ continuous state variables where grid-based methods hit exponential dimensionality walls; continuous Euler equation loss minimization along simulated ergodic state trajectories using an analytical multi-layer perceptron (MLP), pure NumPy deep macro engine (`DeepMacroModel`, `solve_deep_macro`), and out-of-sample Euler residual diagnostics ([EN .py](54_deep_macro_pinns_high_dim.py), [ES .py](54_deep_macro_pinns_high_dim_es.py), [EN .ipynb](54_deep_macro_pinns_high_dim.ipynb), [ES .ipynb](54_deep_macro_pinns_high_dim_es.ipynb)) |
| `55_quantitative_spatial_and_trade_ge` | Quantitative Spatial Economics & Gravity General Equilibrium: General equilibrium welfare, bilateral trade flows, and real wage impacts of trade wars, tariffs, and transport infrastructure; dual foundations of Caliendo-Parro (2015) Exact Hat Algebra with input-output linkages, intermediate goods, and tariffs, and Allen-Arkolakis (2014) continuous geographic gravity model with spatial labor mobility, agglomeration vs. congestion forces, publication-grade trade share matrices, and spatial welfare heatmaps ([EN .py](55_quantitative_spatial_and_trade_ge.py), [ES .py](55_quantitative_spatial_and_trade_ge_es.py), [EN .ipynb](55_quantitative_spatial_and_trade_ge.ipynb), [ES .ipynb](55_quantitative_spatial_and_trade_ge_es.ipynb)) |
| `56_implicit_hjb_and_continuous_kfe` | Continuous-Time HJB & Stationary KFE Wealth Distributions: Achdou et al. (2022) implicit upwind finite-difference scheme, infinitesimal generator $A$ M-matrix properties, adjoint Kolmogorov Forward Equation (KFE) stationary wealth density $g(a, z)$, and continuous Aiyagari general equilibrium asset clearing ([EN .py](56_implicit_hjb_and_continuous_kfe.py), [ES .py](56_implicit_hjb_and_continuous_kfe_es.py), [EN .ipynb](56_implicit_hjb_and_continuous_kfe.ipynb), [ES .ipynb](56_implicit_hjb_and_continuous_kfe_es.ipynb)) |
| `57_multiconstraint_occbin_and_dml` | Multi-Constraint OccBin & Double Machine Learning: Guerrieri-Iacoviello piecewise-linear perturbation under simultaneous occasionally binding constraints ($M \ge 2$: Zero Lower Bound and borrowing caps), multi-regime transition matrix, and Chernozhukov et al. (2018) DML-PLR high-dimensional macroeconomic control orthogonalization ([EN .py](57_multiconstraint_occbin_and_dml.py), [ES .py](57_multiconstraint_occbin_and_dml_es.py), [EN .ipynb](57_multiconstraint_occbin_and_dml.ipynb), [ES .ipynb](57_multiconstraint_occbin_and_dml_es.ipynb)) |
| `58_latin_america_realtime_macro` | Latin American Real-Time Macroeconomic Vintages & Offline Cartridges: Portable cryptographic `.pmz` vintage containers across Banxico, INEGI, BCB, and BCCh, real-time revision triangles $(T \times V)$, and Mankiw-Shapiro (1986) news vs. noise orthogonal revision econometrics ([EN .py](58_latin_america_realtime_macro.py), [ES .py](58_latin_america_realtime_macro_es.py), [EN .ipynb](58_latin_america_realtime_macro.ipynb), [ES .ipynb](58_latin_america_realtime_macro_es.ipynb)) |
| `00_whats_new_in_puremacro_3_0` | Milestone 3.0 interactive showcase: Exact analytic Kalman score gradients, pure-Python HMC/NUTS, and HANK Sequence-Space bridge in `.mod` files (paired Spanish edition: `00_whats_new_in_puremacro_3_0_es`) |
| `00_whats_new_in_puremacro_2_0` | Tour of the 2.0 unified API (`lags`/`horizon`/`ci`, result objects, exporters) |

### Specialized Domain Showcases (`notebooks/macro_history_and_climate/`)

| Notebook | Shows |
|---|---|
| `macro_history_and_climate/N12_paleoclimate_eiv_and_simex` | 2.6.0 modernized: Errors-in-variables (EIV) and SIMEX simulation extrapolation for paleoclimate temperature reconstructions running on puremacro's native pure-NumPy econometric suite (`puremacro.regress.ols`, robust SEs HC0–HC3) with zero external `statsmodels` dependencies |

The deepened showcases (`01`, `06`, `11`, `14`, `15`, `16`, `41`, `44`, `45`, `46`, `47`, `48`, `49`, `50`, `51`, `52`, `53`, `54`, `55`, `56`, `57`, `58`) follow the structure in
[`_TEMPLATE.md`](./_TEMPLATE.md): motivating question → the method in math → intuition →
worked code → read the output → a fill-in *Your turn* → "how comprehensive is this?".
`13_build_your_own_index` is a multi-kernel lab variant (four worked recipes, each with a
fill-in). Showcase `42` provides the flagship template for medium-scale DSGE Bayesian estimation.

### Deepened Showcase Profiles: Continuous Dynamics, IFT, Deep Macro & Spatial GE (52–55)

- **`52_continuous_transition_mit_shocks` (Continuous Transition Dynamics & MIT Shocks)**:
  - *Economic Scenario*: Evaluates the full non-linear macroeconomic transition path of continuous wealth and income distributions $\mu_t(k, z)$ following an unexpected aggregate MIT shock (such as a persistent TFP contraction or monetary tightening) in an incomplete-markets heterogeneous-agent economy. Explores how the shock distorts household precautionary savings, shifts wealth inequality over time (tracking dynamic Gini coefficients and Lorenz curves), and induces endogenous macroeconomic propagation persistence.
  - *Mathematical Methods*: Formulates the coupled continuous dynamic equilibrium: backward continuous endogenous grid method (EGM) policy recursion on continuous assets $k \in [\underline{k}, \bar{k}]$ combined with forward time-dependent Young (2010) non-stochastic density evolution $\mu_{t+1} = T_t^* \mu_t$, preserving mass conservation $\sum \mu_t = 1.0 \pm 10^{-12}$. General equilibrium clearing on the capital market $\int k \, d\mu_t = K^d(r_t)$ is achieved via sequence-space Broyden Quasi-Newton root-finding.
  - *Worked Applications*: A complete 40-quarter transition path simulation using `solve_continuous_transition`, generating hero visualizations of time-varying capital accumulation $K_t$, equilibrium real interest rates $r_t$, wage paths $w_t$, and a 3D distribution surface $\mu_t(k)$ demonstrating the dynamic evolution of precautionary asset holdings. Includes an interactive "Your turn" exploration of shock persistence and damping parameters with automated downstream validation assertions.

- **`53_exact_analytic_ift_gradients` (Exact Analytic IFT Gradients & Structural Estimation)**:
  - *Economic Scenario*: Addresses structural calibration and estimation (Simulated Method of Moments / Generalized Method of Moments) of continuous dynamic macro models where standard numerical finite-difference gradients suffer from catastrophic cancellation, step-size truncation errors, and prohibitive computational costs.
  - *Mathematical Methods*: Implements the exact analytical Implicit Function Theorem (IFT) on continuous projection systems $R(c^*; \theta) = 0$. By applying the adjoint sensitivity operator:
    $$\nabla_\theta c^*(\theta) = - \left[\nabla_c R(c^*; \theta)\right]^{-1} \nabla_\theta R(c^*; \theta)$$
    the method solves a single linear system rather than $2 \times \dim(\theta)$ full non-linear policy recalculations. Analytic Jacobian matrices $\nabla_c R$ and $\nabla_\theta R$ are constructed in machine precision, avoiding finite-difference noise.
  - *Worked Applications*: Structural GMM estimation estimating household discount factor $\beta$ and relative risk aversion $\sigma$ to match empirical capital-to-output and wealth concentration targets. Demonstrates 60×+ execution speedup over central finite differences, smooth objective surfaces, and rapid optimizer convergence. Features an interactive "Your turn" tuning moment targets and risk aversion bounds with downstream assertions.

- **`54_deep_macro_pinns_high_dim` (Deep Macro & Physics-Informed Neural Networks)**:
  - *Economic Scenario*: Overcomes the curse of dimensionality in continuous macroeconomic modeling, enabling the global solution of dynamic economies with 10 or more continuous state variables (e.g., 10-country multi-sector capital accumulation) where traditional tensor-product and sparse grids hit exponential complexity walls ($N^{10} \gg 10^{10}$ points).
  - *Mathematical Methods*: Continuous Physics-Informed Neural Network (PINN) architecture implemented in pure NumPy. The consumption policy $c(k) = \sigma_{\text{net}}(W_2 \phi(W_1 k + b_1) + b_2) \cdot W(k)$ is parameterized by an analytical multi-layer perceptron (MLP) with sigmoid/tanh activation functions, guaranteeing positive consumption and capital accumulation. The loss function minimizes the mean squared continuous Euler equation residual along simulated ergodic trajectories, bypassing the curse of dimensionality by restricting function approximation to the economically relevant ergodic manifold.
  - *Worked Applications*: Full end-to-end solve of a 10-state multi-country dynamic growth model using `DeepMacroModel` and `solve_deep_macro`. Reports training loss convergence curves, physical viability validation ($c > 0, k' > 0, c < W$), and out-of-sample Euler equation residuals ($\text{MSE} < 10^{-3}$). Includes an interactive "Your turn" modifying hidden layer architecture, learning rate schedules, and out-of-sample stress distributions.

- **`55_quantitative_spatial_and_trade_ge` (Quantitative Spatial Economics & Gravity General Equilibrium)**:
  - *Economic Scenario*: Evaluates general equilibrium welfare, bilateral trade flows, real wages, and geographic population shifts under international trade policy shocks (bilateral tariffs, trade wars) and regional transport infrastructure improvements (highway networks, transport cost reductions).
  - *Mathematical Methods*: Combines two foundational pillars of quantitative trade and spatial economics:
    1. Caliendo-Parro (2015) Exact Hat Algebra with multi-sector input-output linkages, intermediate goods, and tariffs, solving counterfactual proportional changes $\hat{x} = x'/x$ without estimating unobserved fundamentals.
    2. Allen-Arkolakis (2014) continuous geographic gravity model with spatial labor mobility, capturing the tension between geographic agglomeration forces ($\alpha$) and congestion forces ($\beta$).
  - *Worked Applications*: Bilateral tariff shock simulation across multiple countries and sectors, tracing input-output supply chain propagation and trade deflection; counterfactual highway corridor construction across geographic regions, calculating real wage and population relocation equilibria. Outputs publication-grade trade share matrices and spatial welfare maps. Includes an interactive "Your turn" adjusting unilateral tariff schedules and spatial mobility elasticities with verification assertions.

### Deepened Showcase Profiles: Continuous HJB/KFE, Multi-Constraint OccBin & Real-Time Macro (56–58)

- **`56_implicit_hjb_and_continuous_kfe` (Continuous-Time HJB & Stationary KFE Wealth Distributions)**:
  - *Economic Scenario*: Solves the continuous-time heterogeneous-agent macro benchmark (Achdou et al. 2022; Aiyagari 1994) where households maximize lifetime discounted CRRA utility subject to idiosyncratic Poisson income shocks and a strict borrowing limit $a \ge \underline{a}$, clearing factor markets in general equilibrium.
  - *Mathematical Methods*: Implements the implicit upwind finite-difference scheme for the continuous-time Hamilton-Jacobi-Bellman (HJB) equation:
    $$\rho v_j(a) = u(c_j(a)) + s_j(a) v_j'(a) + \sum_{k \ne j} \lambda_{jk} [v_k(a) - v_j(a)]$$
    Constructs the transition intensity infinitesimal generator matrix $A(v)$, verifying that it satisfies strict M-matrix properties: row sums zero ($\sum_j A_{ij} = 0$), non-negative off-diagonal transition rates ($A_{ij} \ge 0, i \ne j$), and negative diagonal entries ($A_{ii} < 0$). Solves the stationary wealth density $g(a, z)$ via the adjoint Kolmogorov Forward Equation (KFE) $A^\top g = 0$ with mass conservation $\sum_{i, j} g_{i, j} \Delta a_i = 1$, and searches for the general equilibrium real interest rate $r^*$ clearing aggregate capital supply against firm demand $K^d(r) = ((r + \delta) / (\alpha \bar{Z}))^{1/(\alpha - 1)}$.
  - *Worked Applications*: Rapid value function convergence in 8 iterations with zero numerical instability; stationary density computation satisfying mass conservation error $\le 10^{-15}$; continuous Aiyagari general equilibrium root solve finding $r^* = 0.01888$ ($1.89\%$) with capital market clearing residual $< 1.05 \times 10^{-6}$. Generates a 4-panel hero dashboard (Value functions, Consumption policies, Savings drift, and Stationary distribution with borrowing mass spike). Includes an interactive "Your turn" varying borrowing constraints $\underline{a}$ and risk aversion $\gamma$ with verification assertions.

- **`57_multiconstraint_occbin_and_dml` (Multi-Constraint OccBin & Double Machine Learning)**:
  - *Economic Scenario*: Models macroeconomic crisis dynamics where an economy simultaneously confronts multiple occasionally binding constraints ($M \ge 2$)—specifically a monetary policy Zero Lower Bound (ZLB) on the nominal interest rate and an endogenous collateral borrowing limit on household credit—combined with high-dimensional macroeconomic control estimation via Double / Debiased Machine Learning (DML-PLR; Chernozhukov et al. 2018).
  - *Mathematical Methods*: Implements the Guerrieri-Iacoviello (2015) piecewise-linear perturbation algorithm extended to $M = 2$ constraints ($2^M = 4$ discrete regimes):
    $$A_{r_t} x_t = B_{r_t} x_{t-1} + C_{r_t} \mathbb{E}_t[x_{t+1}] + D_{r_t} + E_{r_t} \varepsilon_t$$
    Regimes dynamically switch based on shadow prices and slackness conditions: Regime 0 (neither binds), Regime 1 (ZLB only), Regime 2 (borrowing cap only), and Regime 3 (both bind simultaneously). Couples the structural trajectory with a high-dimensional partially linear regression (PLR) estimated via 5-fold cross-fitting and Frisch-Waugh-Lovell Neyman-orthogonal residual score projection:
    $$\psi(W; \theta, \eta) = (Y - \ell(X)) - \theta (D - m(X))$$
  - *Worked Applications*: Simulates a deep contractionary shock ($\varepsilon_g = -0.06, \varepsilon_b = 0.05$) triggering Regime 3 for multiple quarters, highlighting non-linear amplification where credit constraints deepen output loss while the ZLB prevents conventional monetary stabilization. Recovers structural treatment effect $\hat{\theta}_{\text{DML}} = 1.699 \pm 0.060$ covering true $\theta_0 = 1.75$ across 100 nuisance controls, whereas naive OLS suffers $42.5\%$ omitted variable bias ($\hat{\theta}_{\text{OLS}} = 2.494$). Displays a 4-panel dashboard with regime indicator timelines, IRF non-linear wedge overlays, cross-fitting folds, and estimator density distributions.

- **`58_latin_america_realtime_macro` (Latin American Real-Time Macroeconomic Vintages & Offline Cartridges)**:
  - *Economic Scenario*: Explores real-time macroeconomic measurement, data revisions, and preliminary release volatility across major Latin American central banks and statistical institutes (Banxico/INEGI Mexico, BCB/IBGE Brazil, and BCCh/INE Chile).
  - *Mathematical Methods*: Implements portable, cryptographically verified `.pmz` offline vintage cartridges containing triangular revision matrices $\mathbf{Y} \in \mathbb{R}^{T \times V}$ where rows index statistical reference quarters $t$ and columns index publishing vintages $v \ge t$. Computes revision paths $R_{t, k} = y_{t, t+k+1} - y_{t, t+k}$ and cumulative revision errors $R_{t, \text{final}} = y_{t, \text{final}} - y_{t, \text{first}}$. Tests the foundational Mankiw-Shapiro (1986) news vs. noise econometric hypothesis:
    $$\text{News (Rational Forecast)}: y_{t, \text{final}} - y_{t, \text{first}} = \alpha + \beta y_{t, \text{first}} + \varepsilon_t \implies H_0: \beta = 0$$
    $$\text{Noise (Measurement Error)}: y_{t, \text{first}} = \alpha + \beta y_{t, \text{final}} + u_t \implies H_0: \beta = 1$$
    with Newey-West HAC robust standard errors.
  - *Worked Applications*: Unpacks and verifies realistic real-time GDP growth vintages spanning 2010Q1–2023Q4 across Mexico, Brazil, and Chile. Demonstrates real-time revision properties: initial releases understate final recoveries during economic turning points by up to 1.8 percentage points; Mankiw-Shapiro tests fail to reject the news hypothesis ($p > 0.35$) while decisively rejecting noise ($p < 10^{-4}$), proving initial Latin American releases represent efficient statistical forecasts rather than noisy errors. Features a 4-panel dashboard (Revision triangle heatmaps, vintage evolution lines, revision dispersion densities, and news vs. noise regression scatters).

## Rebuild

```bash
pip install -e ".[notebooks]"
python tools/build_notebooks.py                 # build all
python tools/build_notebooks.py 01_wealth_inequality   # one
python tools/build_notebooks.py --check         # execute all, fail on error
```

Notebooks are numpy-only (Pyodide-safe), deterministic (fixed seeds), and carry
inline asserts on headline numbers so a numerical regression fails the build.
`14` runs on two frozen real-data snapshots shipped as package data
(`puremacro/replication/data/tax14_*.csv`; regenerate with
`tools/gen_notebook_data_tax14.py` — FRED fredgraph + Ramey's public HOM tax
archive), so it too executes offline and deterministically.

## Running on iPad Juno / Pyodide

To run notebooks on an iPad using **Juno** or in-browser Pyodide kernels (Juno.sh / JupyterLite):

1. **Installation via micropip (Pyodide / Juno.sh)**:
   Because `pyarrow` has no Pyodide wheel, install with `deps=False`:
   ```python
   import micropip
   await micropip.install("puremacro", deps=False)
   ```
2. **Compute Budgets & Memory Limits**:
   iPadOS terminates processes that exceed device memory (~1.5–3 GB). Use `puremacro.runtime` to fit costly operations to tablet ceilings:
   ```python
   from puremacro import runtime
   # Auto-scale bootstrap draws or posterior sampling to tablet size:
   svar = runtime.budgeted(cholesky_svar)
   ```
3. **Bundled Offline Data**:
   All benchmark datasets load offline via `puremacro.datasets` (`load_gali1999()`, `load_narrative_tax_shocks()`, etc.) without requiring network sockets or parquet engines.

