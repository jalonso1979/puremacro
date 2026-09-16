> 🇬🇧 English · 🇪🇸 [Español](es/index.md)

# puremacro

**Production-grade, zero-C-extension Macroeconometric and Heterogeneous-Agent Structural Modeling in Pure Python.**

[![PyPI Version](https://img.shields.io/pypi/v/puremacro.svg)](https://pypi.org/project/puremacro/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![JupyterLite Playground](https://img.shields.io/badge/JupyterLite-Live%20IDE-orange.svg)](https://jalonso1979.github.io/puremacro/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## What is puremacro?

`puremacro` is a unified macroeconomic computing library built entirely in pure Python and NumPy. It eliminates complex Fortran, C++, and MEX toolchains, allowing econometric models to run anywhere: local laptops, high-performance clusters, Google Colab, and **directly inside web browsers via Pyodide / WebAssembly**.

### Key Subsystems

1. **Structural DSGE & Dynare Parity (3.0 – 3.2)**:
   - **Native Dynare `.mod` Parser & `puremacro-dynare` CLI**: Parse and solve standard `.mod` files directly in pure Python.
   - **Exact Analytic Score Recursions**: Symbolic AST differentiation and generalized Sylvester solver evaluating $\nabla_\theta \ln L$ in a single forward pass without finite differences.
   - **Pure-Python HMC & NUTS**: Hamiltonian Monte Carlo with dual-averaging step-size adaptation, diagonal mass adaptation, and rank-normalized split-$\hat{R}$ diagnostics.
   - **Optimal Discretionary Policy vs Commitment**: Dennis (2007) Riccati matrix iterations for Markov-perfect discretionary policy vs timeless-perspective commitment, decomposing inflation and stabilization biases.
   - **DSGE-VAR Hybrid Modeling**: Del Negro & Schorfheide (2004) DSGE-VAR($\lambda$) prior optimization and structural identification.
   - **Anticipated News Shocks**: State-space companion augmentation with nilpotent shift operators preserving Blanchard-Kahn determinacy.
   - **Nonlinear Particle Filtering & MS-DSGE**: Bootstrap and auxiliary particle filters with stochastic volatility, Differentiable OccBin for NUTS, and Foerster et al. (2016) Markov-switching rational expectations solutions with closed-form analytical GIRFs.
   - **Second-Order Pruned Perturbation & OccBin**: Schmitt-Grohé & Uribe (2004) cross-derivatives with Kim et al. (2008) pruning, and Guerrieri & Iacoviello (2015) piecewise-linear Zero Lower Bound (ZLB) regimes.

2. **Heterogeneous-Agent Models (HANK & VFI)**:
   - **Sequence-Space HANK** (Auclert, Bardóczy, Rognlie & Straub 2021, *Econometrica*): General equilibrium incomplete-markets models solved in $\mathcal{O}(T^3)$ sequence space.
   - **Fake News Algorithm**: Fast $\mathcal{O}(T^2)$ computation of sequence-space consumption Jacobians via expectation vectors and cumulation identities.
   - **HANK Sequence-Space Bridge**: Declare `hetagent_block` directly inside standard Dynare `.mod` files.
   - **Targeted Fiscal Transfers**: Dynamic consumption IRFs and cumulative fiscal multipliers across wealth deciles.
   - **Continuous-Time HJB**: Achdou et al. (2022) **implicit** upwind finite-difference M-matrix schemes, the adjoint Kolmogorov forward equation for the stationary density, and continuous-time Aiyagari general equilibrium ([Guide](vfi_hjb_continuous.md)).

3. **Econometric Engines & Local Projections**:
   - **Unified `LPResult`**: Standardized Local Projections (`lp_hac`, `lp_iv`, `lp_state_dep`, `panel_lp`) with Newey-West HAC, fixed-$b$, and Driscoll-Kraay standard errors.
   - **Structural VARs & FAVAR**: Cholesky, Blanchard-Quah, Sign Restrictions, Narrative Sign Restrictions, Proxy/External Instruments, Max-Share/News, and Factor-Augmented VAR.
   - **Modern Difference-in-Differences**: Staggered adoption estimators robust to treatment heterogeneity (Callaway & Sant'Anna, Sun & Abraham, Borusyak-Jaravel-Spiess, Synthetic DiD, Honest DiD).
   - **Spatial Econometrics & Global VAR**: Conley spatial HAC, spatial autoregressive models (`sar`, `sem`, `sdm`), spatial local projections, and the Pesaran Global VAR (`var.gvar`).

4. **Applied Macroeconomic Policy Suites (puremacro 3.2)**:
   - **Central Bank Policy Stance & Fan Charts**: Real-time monetary policy stance evaluation against counterfactual Taylor rules, multi-period fan chart projections, and statement sentiment extraction ([Guide](notebooks.md)).
   - **Real-Time Nowcasting & News Decomposition**: Giannone-Reichlin-Small mixed-frequency dynamic factor models (DFM), handling asynchronous ragged-edge data vintages and release-day news attribution ([Guide](nowcast.md)).
   - **Macroprudential Risk & Financial Stability**: Adrian-Boyarchenko-Giannone Growth-at-Risk (GaR) conditional skew-$t$ densities, and Diebold-Yilmaz systemic connectedness networks ([Guide](notebooks.md)).
   - **Fiscal Policy & Sovereign DSA**: Multi-method fiscal multipliers (SVAR vs LP vs Narrative LP-IV) and stochastic sovereign Debt Sustainability Analysis fan charts under macro-climate stress scenarios.

5. **Global Macro Data Ecosystem & Modular Panels (puremacro 3.2)**:
   - **Emissions & Climate Data**: World Bank WDI and OECD SDMX greenhouse gas accounts across countries and economic sectors ([Guide](data_ecosystem.md)).
   - **Energy Balances & Commodities**: Primary energy consumption, clean energy transition shares, and expanded World Bank Pink Sheet commodity benchmark suites ([Guide](data_ecosystem.md)).
   - **International Financial Stability**: Sovereign yield curves (10Y/2Y), central bank policy rates, BIS credit-to-GDP gaps, and real property prices ([Guide](data_ecosystem.md)).
   - **Modular Panel Builders**: One-call constructors `build_climate_panel` and `build_financial_panel` with automated frequency harmonization (M$\to$Q, A$\to$Q).

6. **Causal ML, Continuous-Time Heterogeneous Agents & Regional Real Time (puremacro 3.4)**:
   - **Double / Debiased Machine Learning**: Chernozhukov et al. (2018) partially linear regression with Neyman-orthogonal Robinson scores, $K$-fold cross-fitting, and pure-NumPy penalized learners — lasso and ridge only, no elastic net ([Guide](causal_dml.md)).
   - **Implicit HJB, Adjoint KFE & Continuous Aiyagari GE**: Sparse M-matrix policy iteration, a stationary density normalized with trapezoid quadrature weights so it integrates to one on non-uniform grids, and a general-equilibrium loop whose `converged` flag means $|K^s - K^d| \le$ `tol_ge` ([Guide](vfi_hjb_continuous.md)).
   - **Multi-Constraint OccBin**: $M \ge 2$ simultaneous occasionally binding constraints, with a regime bitmask (bit $k$ set when constraint $k$ binds) and non-convergence reported as `converged=False` plus a warning naming each reason ([Guide](dsge_build.md)).
   - **Montiel Olea-Pflueger Weak-IV Inference**: Effective $F$-statistic critical values from the closed form $\chi^2_{K,\,1-\alpha}(K/\tau)/K$, and multi-instrument Anderson-Rubin sets inverted exactly rather than on a grid ([Guide](lp.md)).
   - **Latin American Real-Time Connectors**: Banco de México, INEGI, Banco Central do Brasil and Banco Central de Chile. These four sources publish only the current edition, so their vintage date is the local *snapshot* date and revision history accumulates only across repeated captures on different days ([Guide](real_time_latam.md)).
   - **GPU / Apple-Silicon Trade Solvers**: Torch and MLX paths for the Caliendo-Parro equilibrium, imported lazily and matching the NumPy solver's tariff defaults ([Guide](trade_gpu.md)).

7. **Publication-Grade Reporting**:
   - Zero-dependency, camera-ready table export directly to **LaTeX** (`.to_latex()`), **Typst** (`.to_typst()`), and **Markdown** (`.to_markdown()`), complete with standard errors and significance stars ([Guide](reporting.md)).

8. **Running Anywhere**:
   - Full Pyodide/WebAssembly compatibility for tablets and browser notebooks ([Guide](tablet.md)), with automatic Google Colab offloading (`runtime.colab`), chunked long-run execution (`longrun`), and portable `.pmz` data cartridges (`pocket`).

---

## Installation

```bash
pip install puremacro
```

The parquet and Excel readers (the course's ENOE lessons, `build_all`, the shock atlas) need the optional file-format engines:

```bash
pip install "puremacro[io]"
```

Or install with full notebook tools:

```bash
pip install "puremacro[notebooks]"
```
