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

1. **Structural DSGE & Dynare Parity**:
   - **Native Dynare `.mod` Parser & `puremacro-dynare` CLI**: Parse and solve standard `.mod` files directly in pure Python.
   - **Second-Order Pruned Perturbation**: Schmitt-Grohé & Uribe (2004) cross-derivatives ($g_{xu}, g_{uu}$) and risk corrections ($g_{\sigma\sigma}$) with Kim et al. (2008) pruning and full Dynare `oo_.dr` decision rule parity.
   - **OccBin (Occasionally Binding Constraints)**: Guerrieri & Iacoviello (2015) piecewise-linear algorithm for Zero Lower Bound (ZLB) and borrowing limits.
   - **Non-Linear Perfect Foresight**: Boucekkine-Juillard stacked Newton-Raphson relaxation solver for large-scale deterministic transitions.
   - **Bayesian DSGE MCMC Estimation**: Mode-finding via L-BFGS-B / Nelder-Mead, Laplace Hessian covariance, and adaptive Random-Walk Metropolis-Hastings.
   - **Analytical Moments & Shock Decompositions**: Lyapunov theoretical moments, FEVD, and exact Kalman historical shock decompositions.

2. **Heterogeneous-Agent Models (HANK & VFI)**:
   - **Sequence-Space HANK** (Auclert, Bardóczy, Rognlie & Straub 2021, *Econometrica*): General equilibrium incomplete-markets models solved in $\mathcal{O}(T^3)$ sequence space.
   - **Fake News Algorithm**: Fast $\mathcal{O}(T^2)$ computation of sequence-space consumption Jacobians via expectation vectors and cumulation identities.
   - **Targeted Fiscal Transfers**: Dynamic consumption IRFs and cumulative fiscal multipliers across wealth deciles.
   - **DMP Search & Matching** (Mortensen-Pissarides): Labor market transitions, wage rigidity, and Beveridge curve dynamics.

3. **Econometric Engines & Local Projections**:
   - **Unified `LPResult`**: Standardized Local Projections (`lp_hac`, `lp_iv`, `lp_state_dep`, `panel_lp`) with Newey-West HAC and Driscoll-Kraay standard errors.
   - **Structural VARs & FAVAR**: Cholesky, Blanchard-Quah, Sign Restrictions, Proxy/External Instruments, Max-Share/News, and Factor-Augmented VAR.
   - **Modern Difference-in-Differences**: Staggered adoption estimators robust to heterogeneity (Callaway & Sant'Anna, Sun & Abraham, Borusyak-Jaravel-Spiess, Synthetic DiD).

4. **Nowcasting & Machine Learning**:
   - **Mixed-Frequency Dynamic Factor Models (DFM)** (Giannone, Reichlin & Small 2008): Real-time GDP tracking with ragged edges and news decomposition.
   - **Penalized Macro Forecasting**: Coordinate-descent Elastic Net and Adaptive Lasso (Zou 2006) for high-dimensional predictor selection.

5. **Climate Macroeconomics**:
   - **Dynamic Integrated Climate-Economy (DICE)** (Nordhaus 2018): 3-reservoir carbon cycle, warming dynamics, and Social Cost of Carbon (SCC) scenario accounting.

6. **Publication-Grade Reporting**:
   - Zero-dependency, camera-ready table export directly to **LaTeX** (`.to_latex()`), **Typst** (`.to_typst()`), and **Markdown** (`.to_markdown()`), complete with standard errors and significance stars.

7. **Running Anywhere**:
   - Full Pyodide/WebAssembly compatibility for tablets and browser notebooks, with automatic Google Colab offloading (`runtime.colab`), chunked long-run execution (`longrun`), and portable `.pmz` data cartridges (`pocket`).

---

## Installation

```bash
pip install puremacro
```

Or install with full notebook tools:

```bash
pip install "puremacro[notebooks]"
```
