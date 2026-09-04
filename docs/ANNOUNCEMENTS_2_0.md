# puremacro 2.0.0 Release Announcements & Community Outreach Kit

This document provides ready-to-publish release announcement templates for **puremacro 2.0.0**, tailored for different academic, social, and developer communities.

---

## 1. EconTwitter / X / Bluesky Thread 🧵

### Tweet 1 (Hook & Value Prop)
> 🚀 Excited to announce **puremacro 2.0.0** — the pure-Python, zero-C-compiler empirical macroeconomics and time-series toolkit!
>
> 📦 `pip install puremacro`
> 🌐 WebAssembly & iPad / Juno ready
> 📚 Docs: https://jalonso1979.github.io/puremacro/
>
> From Local Projections to Frontier Staggered DiD, here is what is new 👇 (1/6)

### Tweet 2 (Frontier Micro/Macro Econometrics)
> 🔬 **Frontier Identification & DiD**:
> • Callaway-Sant'Anna (2021) & Sun-Abraham (2021)
> • Borusyak-Jaravel-Spiess (2022) & Synthetic DiD
> • **Honest DiD Sensitivity Analysis** (Rambachan & Roth 2023): evaluate parallel trends violations with relative magnitude & smoothness bounds + breakdown values $M^*$.
>
> All vectorized in pure NumPy. (2/6)

### Tweet 3 (Time Series & Structural Macro)
> 📈 **Macro Time Series & Structural Identification**:
> • Local Projections with Montiel Olea-Plagborg-Møller (2021) Lag-Augmentation & GARCH-IV
> • State-Dependent & Smooth Transition LPs
> • Factor-Augmented VARs (FAVAR — Bernanke, Boivin & Eliasz 2005)
> • **2nd-Order DSGE Perturbation with Pruning** (Kim, Kim, Schaumburg & Sims 2008): strictly stable, asymmetric GIRFs, and ergodic means! (3/6)

### Tweet 4 (Zero Friction & Tablet/Browser Mobility)
> 📱 **True Academic Mobility**:
> No Fortran, no C++ compilation steps, no BLAS/LAPACK mismatch errors.
>
> Runs 100% inside your browser (Pyodide / JupyterLite) and natively on iPad (Juno). Need to run a 10,000-draw MCMC? One line generates a self-contained Google Colab notebook with cloud auth and embedded data! (4/6)

### Tweet 5 (Publication Pipeline & Rosetta Stone)
> 📑 **From Code to Paper in Seconds**:
> • Instant LaTeX (`\begin{tabular}`) and Typst table export with academic significance stars (`***`, `**`, `*`).
> • Coming from Stata or MATLAB? We included a complete **Rosetta Stone** guide (`reg`, `reghdfe`, `var`, `csdid` $\to$ puremacro). (5/6)

### Tweet 6 (Get Started)
> Try the interactive browser playground today with zero installation:
> 🔗 Playground: https://jalonso1979.github.io/puremacro/playground.html
> 💻 GitHub: https://github.com/jalonso1979/puremacro
>
> Feedback, issues, and contributions are warmly welcome! #EconTwitter #Python #Econometrics #Macroeconomics (6/6)

---

## 2. LinkedIn Post (Academic & Central Bank Economists)

> 📊 **Announcing puremacro 2.0.0: Bringing Modern Empirical Macroeconomics to the Pure-Python Ecosystem**
>
> In macroeconomic research and policy analysis, empirical workflows frequently encounter tooling friction: heavy binary dependencies, platform-specific compilation hurdles, and fragmented codebases split between MATLAB, Stata, and bespoke scripts.
>
> Today, we are releasing **puremacro 2.0.0**, designed from first principles around a zero-C-dependency architecture. If Python and NumPy run on your device, puremacro runs out of the box — across Linux clusters, macOS, Windows, iPad (via Juno), and client-side browsers (via WebAssembly / Pyodide).
>
> 🌟 **Key Highlights in 2.0.0**:
> 1. **Frontier Causal Inference & Difference-in-Differences**:
>    - Full suite of heterogeneity-robust staggered DiD estimators: Callaway & Sant'Anna (2021), Sun & Abraham (2021), Borusyak, Jaravel & Spiess (2022), and Synthetic DiD (Arkhangelsky et al. 2021).
>    - **Honest DiD Sensitivity Analysis** (Rambachan & Roth 2023): evaluate departures from parallel trends using relative magnitude and smoothness restrictions, computing robust confidence sets (Imbens & Manski 2004) and breakdown values $M^*$.
>
> 2. **Structural Macro & Time Series**:
>    - Local Projections (Jordà 2005) with Montiel Olea & Plagborg-Møller (2021) lag-augmentation and robust HAC inference.
>    - State-dependent LPs and Factor-Augmented VARs (FAVAR, Bernanke et al. 2005).
>    - Klein (2000) QZ rational expectations solver and **second-order DSGE perturbation with pruning** (Kim, Kim, Schaumburg & Sims 2008), eliminating explosive quadratic trajectories and delivering asymmetric generalized impulse responses.
>
> 3. **Publication-Ready Reporting**:
>    - Direct export of estimation results to academic LaTeX and Typst tables with conventional significance stars (`* p<0.10`, `** p<0.05`, `*** p<0.01`).
>
> 4. **Cloud Compute Bridge**:
>    - Seamless transition from mobile/tablet sketchpads to Google Colab accelerators with zero file-upload friction via embedded self-contained notebooks.
>
> 🔗 **Interactive Browser Playground**: https://jalonso1979.github.io/puremacro/playground.html
> 📖 **Full Documentation**: https://jalonso1979.github.io/puremacro/
> 📦 **PyPI**: `pip install puremacro`
> 💻 **GitHub**: https://github.com/jalonso1979/puremacro
>
> We look forward to seeing the empirical research and teaching applications enabled by this release.

---

## 3. QuantEcon / PyData Discourse & Forum Post

**Title**: [ANN] puremacro 2.0.0 — Pure-Python, Pyodide-Compatible Empirical Macro & Frontier DiD Toolkit

**Body**:
> Hi everyone,
>
> We are excited to announce the release of **puremacro 2.0.0**!
>
> ### Why puremacro?
> Most macroeconomic toolkits in Python either rely on complex C/Fortran/Cython build matrices (which fail on edge environments, WebAssembly, or tablets) or lack modern post-2020 identification methods.
>
> `puremacro` was built to fill this gap with three core design rules:
> 1. **Zero compiled C extensions**: Only standard library, NumPy, and SciPy.
> 2. **Strict Pyodide compatibility**: The complete estimator suite runs client-side in the browser via WebAssembly without a backend server.
> 3. **Frontier econometric methods**: First-class support for modern macro and panel techniques.
>
> ### What's included in 2.0.0?
> - **Local Projections**: Jordà (2005), Montiel Olea & Plagborg-Møller (2021) lag-augmentation, GARCH-IV shock instruments, smooth transition (STLP).
> - **Vector Autoregressions**: VAR(p), FAVAR (Bernanke, Boivin & Eliasz 2005), historical decomposition, forecast error variance decomposition (FEVD).
> - **Modern Staggered DiD**:
>   - Callaway & Sant'Anna (2021)
>   - Sun & Abraham (2021)
>   - Borusyak, Jaravel & Spiess (2022)
>   - Synthetic Difference-in-Differences (SDID)
>   - **Honest DiD Sensitivity** (Rambachan & Roth 2023) for parallel trends violations
> - **DSGE Modeling**:
>   - Klein (2000) QZ solver with complex-step automatic Jacobians (no hand differentiation!)
>   - Second-order perturbation with pruning (Kim et al. 2008)
> - **Reporting & Workflows**:
>   - Direct export to LaTeX and Typst tables with significance stars (`***`, `**`, `*`).
>   - Stata-to-puremacro Rosetta Stone guide for economists transitioning from `.do` files.
>   - Cloud offload bridge to Google Colab for iPad / Juno users.
>
> ### Try it out
> - Interactive WebAssembly Playground: https://jalonso1979.github.io/puremacro/playground.html
> - Getting Started Guide: https://jalonso1979.github.io/puremacro/getting_started.html
> - GitHub Repository: https://github.com/jalonso1979/puremacro
>
> We would love to hear your feedback, bug reports, and ideas for further extensions!
