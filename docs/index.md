# puremacro

**Production-grade, zero-C-extension Macroeconometric and Heterogeneous-Agent Structural Modeling in Pure Python.**

[![PyPI Version](https://img.shields.io/pypi/v/puremacro.svg)](https://pypi.org/project/puremacro/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JupyterLite Playground](https://img.shields.io/badge/JupyterLite-Live%20IDE-orange.svg)](https://jalonso1979.github.io/puremacro/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## What is puremacro?

`puremacro` is a unified macroeconomic computing library built entirely in pure Python and NumPy. It eliminates complex Fortran, C++, and MEX toolchains, allowing econometric models to run anywhere: local laptops, high-performance clusters, Google Colab, and **directly inside web browsers via Pyodide / WebAssembly**.

### Key Subsystems

1. **Structural Models**:
   - **Sequence-Space HANK** (Auclert, Bardóczy, Rognlie & Straub 2021, *Econometrica*): General Equilibrium heterogeneous-agent models solved in $O(T^3)$ time via sequence Jacobians.
   - **DMP Search & Matching** (Mortensen-Pissarides): Labor market transitions and Beveridge curve dynamics.
   - **Continuous-Time HJB**: High-order finite difference solvers for wealth distributions.

2. **Time Series & VAR Econometrics**:
   - **Factor-Augmented VAR (FAVAR)** (Bernanke, Boivin & Eliasz 2005): Latent factor extraction and large-cross-section policy impulse responses.
   - **Structural VARs**: Cholesky, Blanchard-Quah, Sign Restrictions, Proxy/External Instruments, and Narrative Signs (Antolín-Díaz & Rubio-Ramírez 2018).
   - **Bayesian VARs**: Minnesota Gibbs sampler with conjugate prior-posterior conjugacy.

3. **Nowcasting & Machine Learning**:
   - **Mixed-Frequency Dynamic Factor Models (DFM)** (Giannone, Reichlin & Small 2008): Real-time GDP tracking with ragged edges and news decomposition.
   - **Penalized Forecasting**: Elastic Net and Adaptive Lasso for sparse macroeconomic predictor selection.

4. **Climate Macroeconomics**:
   - **Dynamic Integrated Climate-Economy (DICE)** (Nordhaus 2018, Golosov et al. 2014): 3-reservoir carbon cycle, surface warming dynamics, and Social Cost of Carbon (SCC) policy analysis.

5. **Narrative Macroeconomics & Natural Language Processing**:
   - Central bank press conference transcript parsing (opening remarks vs. Q&A).
   - Macroeconomic blog RSS aggregation (VoxEU, Apricitas, Marginal Revolution, NBER).
   - Narrative burst anomaly detection and crowd vs. expert sentiment divergence index.

---

## Installation

```bash
pip install puremacro
```

Or install with full notebook tools:

```bash
pip install "puremacro[notebooks]"
```
