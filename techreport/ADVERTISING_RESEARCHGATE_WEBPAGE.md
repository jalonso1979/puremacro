> **Claims checked against the 4.6.0 report, 4 October 2026.** Describe feature availability separately from validation. No universal or bitwise portability claim is supported: the shippable modules are checked to import under Pyodide, and the JupyterLite playground runs the notebooks client-side, and the whole gallery passes in Pyodide 0.28.3 and 314.0.5 under Node.js, but realistic browser and GPU workloads remain unverified. The 114-case gallery mixes internal, analytical and selected external references, with case-specific tolerances; its four trade cases run on the clean OECD 2020 table. See [current structural validation status](../docs/STRUCTURAL_VALIDATION_STATUS.md).

# Promotional and Dissemination Package: puremacro Technical Report
## Tailored Materials for ResearchGate, Personal Webpage, and Academic Networks

This package provides ready-to-use materials to announce and disseminate the comprehensive technical report on `puremacro` (version 4.6, 4 October 2026, 43 pages).

---

## 1. ResearchGate Dissemination Materials

### 1.1 Working Paper / Technical Report Metadata
When uploading the compiled PDF (`puremacro_technical_report.pdf`) to ResearchGate as a **Working Paper** or **Technical Report**:

- **Title**:\
  `puremacro: A Unified, Dependency-Minimal Scientific-Python Engine for Quantitative Macroeconomics and Macroeconometrics`
- **Subtitle**:\
  `Architecture, Algorithmic Foundations, Numerical Validation, and Browser-Native Reproducibility`
- **Authors**:\
  Jorge Alonso Ortiz (Instituto Tecnológico Autónomo de México - ITAM)\
  The report's title page also credits the AI coding agents Claude (Anthropic), Codex (OpenAI) and Antigravity (Google) as coauthors. ResearchGate author fields are for people, so credit them in the description field rather than as author profiles; the authorship statement is on the report's title page and in its "Authorship and contributions" section.
- **Date**:\
  October 2026
- **Publication Type**:\
  Technical Report / Working Paper (v4.6, 43 pages)
- **Research Topics / Disciplines**:\
  Macroeconomics, Econometrics, Computational Economics, Time Series Analysis, International Trade, Economic Policy, Monetary Economics, Quantitative Methods.
- **Skills & Methods**:\
  Python, Structural VAR, Local Projections, Dynamic Stochastic General Equilibrium (DSGE), Bayesian Estimation (NUTS), Heterogeneous-Agent Models (HANK), Sequence-Space Jacobian, Continuous-Time Macroeconomics, Dynamic Programming, Numerical Projections, Synthetic Control, Difference-in-Differences, Minimum-Distance Estimation, Input-Output and Multi-Region Trade Models, Replication, Open Source Science, Pyodide, WebAssembly.

- **Abstract**:\
  Copy and paste into the abstract field (plain-text version of the paper's abstract):
> puremacro is an open-source Python library that brings empirical macroeconometrics and quantitative macroeconomic models into one environment, built around three goals that macroeconomic software has rarely combined: methodological breadth, portability and numerical verifiability. One package covers structural VARs and local projections, volatility models, difference-in-differences and synthetic control, nowcasting and dynamic panels, DSGE perturbation to third order with a native Dynare parser and Bayesian estimation, heterogeneous-agent and sequence-space HANK models, continuous-time and projection methods, and quantitative spatial, input-output and trade general equilibrium. Estimators and solvers return typed result objects with summaries, plots and direct export to LaTeX, Typst and Markdown, so a paper's tables and figures come from the same objects as its estimates. Written entirely in Python on NumPy, SciPy, pandas and Matplotlib, with no compiled extension of its own, it installs with pip on Linux, macOS and Windows and runs client-side in the browser through Pyodide, where its whole validation gallery passes and which hosts a JupyterLite playground of 70 bilingual notebook pairs and a 22-lesson advanced macroeconomics course. Verification is built in: a 114-case validation gallery checks results against closed forms, internal identities and reference outputs frozen offline from statsmodels, arch, linearmodels, SciPy and a MATLAB trade model, and re-runs with one call on the user's machine, without those packages; 15 replication cases and research benchmarks against Dynare, official statistics, published tables and independent software complete the evidence. Versions 4.4 and 4.5 add an empirical-to-structural layer, puremacro.structural, which carries labelled empirical moments and their full covariance into bounded minimum-distance estimation of structural parameters, an original-data Romer–Romer (2010) baseline, an observed-data Smets–Wouters (2007) moment study with finite-sample diagnostics, INEGI ENIGH 2024 distributional tariff incidence, seven input-output trade engines with a dynamic MRIO, and a stacked-time Newton–Krylov solver. For researchers, the result is one reproducible toolchain from data to structural model; for instructors and students, a computational macroeconomics curriculum that opens in a browser tab.

### 1.2 ResearchGate Project Update / Feed Announcement Post
Post this in your ResearchGate feed or update your project log:

> 📢 **New Working Paper & Computational Release: puremacro v4.6**
>
> I am excited to share a 43-page technical report and working paper on **puremacro**, an open-source Python library built around the *Computational Macroeconomics Trilemma*: methodological breadth, portability and numerical verifiability, pursued together in a single package.
>
> 🔹 **The Problem**: Computational macroeconomics has historically been fragmented across MATLAB/Dynare, R, Stata, and specialized C++/Numba libraries, introducing licensing costs, compiler issues, and friction in research and graduate teaching.
>
> 🔹 **The puremacro Approach**:
> - **Core Scientific-Python Stack**: Written entirely in Python on NumPy, SciPy, pandas and Matplotlib, with no compiled extension of its own; installs with `pip` on Linux, macOS and Windows.
> - **Browser Execution**: The same code imports under Pyodide and powers a JupyterLite playground that runs 70 bilingual notebook pairs and a 22-lesson course client-side, with nothing to install.
> - **End-to-End Methodological Coverage**:
>   1. *Macroeconometrics*: SVAR (Cholesky, Blanchard-Quah, sign, zero, narrative, proxy SVAR-IV), local projections (LP-HAC, lag-augmented, panel with Driscoll-Kraay), GARCH and GARCH-MIDAS volatility, HAC, fixed-b and weak-IV inference.
>   2. *Micro-Macro Causal Inference and Nowcasting*: Callaway-Sant'Anna and Sun-Abraham staggered DiD, synthetic control and synthetic DiD, double/debiased machine learning, mixed-frequency dynamic-factor nowcasting, Arellano-Bond dynamic panels and shift-share designs.
>   3. *DSGE Engine*: Native Dynare `.mod` parser, perturbation to third order with Kim et al. pruning, perfect-foresight transitions with a stacked-time Newton–Krylov solver, and Bayesian NUTS estimation with exact analytical Kalman score gradients.
>   4. *Heterogeneous Agents*: Sequence-space Jacobian (HANK) household blocks declared inside `.mod` files, Young (2010) non-stochastic density iteration, and continuous-time upwind HJB-KFE solvers.
>   5. *Continuous Projections*: Chebyshev collocation, Smolyak sparse grids, and finite-element Galerkin with Fischer-Burmeister complementarity.
>   6. *Spatial, Trade and Climate*: Allen-Arkolakis topography, Caliendo-Parro exact hat algebra with flexible CES technology, preferences and markups, seven input-output engines including a perfect-foresight dynamic MRIO, a clean OECD 2020 77x11 table with the legacy MATLAB model as its external reference (4.6), INEGI ENIGH 2024 distributional tariff incidence, and a DICE-2016R simulator.
>   7. *Empirical-to-Structural Bridge*: `puremacro.structural` carries labelled empirical moments and their full covariance into bounded minimum-distance estimation of structural models, with an original-data Romer–Romer (2010) baseline and an observed-data Smets–Wouters (2007) moment study.
> - **Verification Built In**: a 114-case validation gallery (62 internal-consistency, 31 analytical and 21 external-reference cases, with case-specific tolerances) that re-runs with one call on your own machine and passes inside Pyodide; 15 replication cases, including the Smets-Wouters log posterior checked against Dynare 8 at the authors' own mode; and research benchmarks referenced to Dynare, official statistics, published tables and independent software. The test suite runs on nine CI targets (Linux, macOS, Windows; Python 3.11–3.13) and every release must pass it before reaching PyPI.
>
> 📄 Working Paper PDF: Attached on ResearchGate\
> 💻 GitHub: https://github.com/jalonso1979/puremacro\
> 🌐 Interactive Web Platform: https://jalonso1979.github.io/puremacro/\
>
> Feedback and discussions are warmly welcome!

---

## 2. Personal Academic Webpage Announcement

### 2.1 HTML Card (Ready to paste into your personal website / ITAM faculty page)

```html
<!-- puremacro Featured Research Card -->
<div style="border: 1px solid #cbd5e1; border-radius: 10px; padding: 24px; margin: 24px 0; background-color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
  <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
    <div>
      <span style="background-color: #1e3d59; color: #ffffff; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">Working Paper & Software Release</span>
      <h3 style="color: #1e3d59; margin: 12px 0 6px 0; font-size: 22px;">puremacro: Quantitative Macroeconomics on the Core Scientific-Python Stack</h3>
      <p style="color: #475569; font-size: 15px; margin: 0 0 14px 0; font-style: italic;">Architecture, Algorithmic Foundations, Numerical Validation, and Browser-Native Reproducibility</p>
    </div>
  </div>

  <p style="color: #334155; font-size: 14.5px; line-height: 1.6; margin: 0 0 16px 0;">
puremacro brings empirical macroeconometrics and quantitative macroeconomic models into one open-source Python package, written entirely on NumPy, SciPy, pandas and Matplotlib with no compiled extension of its own. It covers structural VARs and local projections, causal inference and nowcasting, DSGE perturbation with a native Dynare parser, heterogeneous-agent and continuous-time models, projection methods, spatial and input-output trade equilibrium, and an empirical-to-structural estimation bridge. Results arrive as typed objects that export to LaTeX, Typst and Markdown; a 114-case validation gallery that also passes inside Pyodide, 15 replication cases and externally referenced research benchmarks ship inside the package.
  </p>

  <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-top: 16px;">
    <a href="techreport/puremacro_technical_report.pdf" target="_blank" style="background-color: #1e3d59; color: #ffffff; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-size: 13.5px; font-weight: 600; display: inline-flex; align-items: center;">
      📄 Download Technical Monograph (PDF, 43 pp.)
    </a>
    <a href="https://github.com/jalonso1979/puremacro" target="_blank" style="background-color: #ffffff; color: #1e3d59; border: 1.5px solid #1e3d59; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-size: 13.5px; font-weight: 600; display: inline-flex; align-items: center;">
      💻 GitHub Repository (v4.6)
    </a>
    <a href="https://jalonso1979.github.io/puremacro/" target="_blank" style="background-color: #17b978; color: #ffffff; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-size: 13.5px; font-weight: 600; display: inline-flex; align-items: center;">
      🌐 Interactive JupyterLite Browser Platform
    </a>
  </div>
</div>
```

### 2.2 Markdown Section (for Jekyll / Hugo / GitHub Pages)

```markdown
### 🚀 puremacro: A Unified Scientific-Python Engine for Macroeconomics

I am pleased to present the comprehensive technical monograph and v4.6 release of **puremacro**:

> **puremacro: A Unified, Dependency-Minimal Scientific-Python Engine for Quantitative Macroeconomics and Macroeconometrics**\
> *Jorge Alonso Ortiz (ITAM, October 2026)*\
> [📄 Download Technical Report (PDF)](techreport/puremacro_technical_report.pdf) · [💻 GitHub](https://github.com/jalonso1979/puremacro) · [🌐 Interactive Browser Platform](https://jalonso1979.github.io/puremacro/) · [📦 PyPI](https://pypi.org/project/puremacro/)

**Key Highlights:**
- **Core Scientific-Python Stack**: A pure-Python package on NumPy + SciPy + pandas + Matplotlib, with no compiled extension of its own; it imports under Pyodide and runs client-side in the JupyterLite playground.
- **Structural Macroeconometrics**: SVAR with Cholesky, Blanchard-Quah, Uhlig sign, Rubio-Ramírez sign/zero, Antolín-Díaz narrative sign, and Stock-Watson / Mertens-Ravn proxy-IV identification; GARCH and GARCH-MIDAS volatility; HAC, fixed-b and weak-IV inference.
- **Local Projections & Causal Inference**: LP-HAC, lag-augmented LP, Driscoll-Kraay panel LP, Callaway-Sant'Anna and Sun-Abraham DiD, synthetic control, synthetic DiD, double/debiased ML, mixed-frequency nowcasting and Arellano-Bond dynamic panels.
- **DSGE & Bayesian NUTS**: Direct parsing of Dynare `.mod` files, perturbation to third order with pruning, stacked-time Newton–Krylov perfect-foresight transitions, and NUTS with exact analytical Kalman score gradients.
- **Heterogeneous Agents & HANK**: Sequence-Space Jacobian (Auclert et al.) household blocks inside `.mod` files, Young (2010) density iteration, and continuous-time HJB-KFE solvers.
- **Continuous Projections**: Chebyshev collocation, Smolyak sparse grids, and finite-element Galerkin with Fischer-Burmeister complementarity.
- **Spatial, Trade & Climate**: Allen-Arkolakis topography, Caliendo-Parro exact hat algebra with flexible CES technology, preferences and markups, seven input-output engines with a perfect-foresight dynamic MRIO, INEGI ENIGH 2024 household tariff incidence, and DICE-2016R simulation.
- **Empirical-to-Structural Bridge**: `puremacro.structural` carries labelled empirical moments with their full covariance into bounded minimum-distance estimation; original-data Romer–Romer (2010) and observed-data Smets–Wouters (2007) studies.
- **Verification Built In**: a 114-case validation gallery (62 internal, 31 analytical and 21 external-reference cases, case-specific tolerances) re-runnable with one call and passing inside Pyodide, 15 replication cases, and research benchmarks against Dynare, official statistics, published tables and independent software; continuous integration on nine targets gates every PyPI release.
```

---

## 3. Social Media & Academic Community Dissemination

### 3.1 Twitter / X Thread (for #EconTwitter)

**Tweet 1 (Hook)**:\
How can we end computational fragmentation in macroeconomics?\
Applied researchers juggle MATLAB/Dynare, R, Stata, & C++/Numba libraries—costing licenses, breaking environments, & eating up class time.\
I'm excited to share a 43-page working paper on **puremacro**: a unified, dependency-minimal Python engine! 🧵👇\
[Link to Paper] #EconTwitter #ComputationalEcon

**Tweet 2 (The Architecture)**:\
The core design invariant: **no package-owned compiled extensions**.\
puremacro's shippable modules import only NumPy, SciPy, pandas and Matplotlib.\
That is what lets the same code import under Pyodide and run client-side in the JupyterLite playground: notebooks on SVARs, DSGE perturbation and HANK models open in a browser tab, nothing to install. 📱💻

**Tweet 3 (Structural Identification Spectrum)**:\
Tired of jumping between MATLAB toolboxes? puremacro unifies the identification spectrum:\
✅ Cholesky recursive & Blanchard-Quah long-run\
✅ Uhlig & Rubio-Ramírez Sign & Zero restrictions\
✅ Antolín-Díaz & Rubio-Ramírez Narrative Sign Restrictions\
✅ Stock-Watson & Mertens-Ravn Proxy SVAR-IV\
✅ Lag-augmented & Driscoll-Kraay Panel Local Projections

**Tweet 4 (DSGE & Bayesian NUTS with Exact Gradients)**:\
A native parser reads Dynare `.mod` syntax directly (no MATLAB required!).\
Solves perturbation to third order with Kim et al. pruning, and perfect-foresight transitions with a stacked-time Newton–Krylov solver.\
Bayesian estimation via the No-U-Turn Sampler (NUTS) uses **exact analytical Kalman score gradients** obtained from generalized Sylvester equations.

**Tweet 5 (Heterogeneous Agents & Sequence Space)**:\
Coupling micro and macro: declare household blocks (`hetagent_block;`) inside your Dynare `.mod` file!\
puremacro uses the Sequence-Space Jacobian fake-news algorithm (Auclert et al. 2021) & continuous-time upwind HJB-KFE solvers (Achdou et al. 2022) to solve HANK and continuous-time models.

**Tweet 6 (Empirical-to-Structural Bridge)**:\
New in 4.4/4.5: `puremacro.structural` carries labelled empirical moments (e.g. local-projection responses with their joint HAC covariance) into bounded minimum-distance estimation of structural models.\
Real-data studies: the original-data Romer–Romer (2010) baseline, observed-data Smets–Wouters (2007) moments with finite-sample diagnostics, and INEGI ENIGH 2024 tariff incidence.

**Tweet 7 (Validation & Pedagogy)**:\
Can you trust the numbers?\
A 114-case validation gallery (62 internal checks, 31 analytical cases, 21 external references, case-specific tolerances) re-runs with one call on your machine and passes inside Pyodide, plus 15 replication cases and research benchmarks against Dynare, official statistics, published tables and independent software.\
Already powering *Macroeconomía Avanzada* at ITAM, with 70 bilingual notebook pairs and 22 course lessons in the browser!

**Tweet 8 (Links)**:\
📄 Working Paper: [Link to ResearchGate / PDF]\
💻 Code (MIT): github.com/jalonso1979/puremacro\
🌐 Interactive Browser Platform: jalonso1979.github.io/puremacro/\
Try it out and let me know what you think! 🚀

---

### 3.2 LinkedIn Scholarly Announcement

> **Announcing puremacro: Overcoming the Computational Macroeconomics Trilemma**
>
> In quantitative macroeconomics and macroeconometrics, software fragmentation has long imposed heavy friction on researchers and instructors. Moving between reduced-form time series, Dynare DSGE perturbation, heterogeneous-agent models, and spatial trade general equilibrium often requires four different languages and proprietary licenses.
>
> To address this, I have authored a 43-page technical monograph detailing **puremacro**, an open-source Python library released on PyPI (version 4.6):
>
> 🔍 **Key Innovations**:
> 1. **Pure Scientific-Python Foundation**: Written entirely in Python on NumPy, SciPy, pandas and Matplotlib (plus requests for data access), with no compiled extension of its own. It installs with pip on Linux, macOS and Windows, and the same code runs client-side in the browser through Pyodide.
> 2. **Complete Econometric Suite**: Recursive, Blanchard-Quah, sign, zero, narrative and proxy SVARs; local projections with HAC, lag-augmented and Driscoll-Kraay panel inference; staggered difference-in-differences, synthetic control and double machine learning; mixed-frequency nowcasting and dynamic panels.
> 3. **Native DSGE & Bayesian NUTS**: Direct parsing of Dynare `.mod` syntax, pruned decision rules to third order, stacked-time Newton–Krylov perfect-foresight transitions, and Bayesian estimation via the No-U-Turn Sampler with exact analytical Kalman score gradients derived from generalized Sylvester equations.
> 4. **Micro-Macro Synthesis**: Sequence-Space Jacobian (SSJ) heterogeneous-agent coupling, continuous-time HJB-KFE solvers, Chebyshev, Smolyak and finite-element Galerkin projections, and quantitative spatial, input-output and trade general equilibrium down to household-level tariff incidence.
> 5. **Empirical-to-Structural Bridge and Built-In Verification**: `puremacro.structural` estimates structural parameters by bounded minimum distance from labelled empirical moments with their full covariance, with original-data Romer–Romer (2010) and observed-data Smets–Wouters (2007) studies. A 114-case validation gallery re-runs on your own machine and passes inside Pyodide, and 15 replication cases and research benchmarks against Dynare, official statistics, published tables and independent software complete the evidence.
>
> The library is the computational backbone of *Macroeconomía Avanzada* at ITAM: students install it with pip or open the 22 course lessons in the browser, with no MATLAB licence and no compiler.
>
> 📖 Read the full technical monograph: [Link to ResearchGate / PDF]\
> 💻 Explore the repository: https://github.com/jalonso1979/puremacro\
> 🌐 Run interactive notebooks in your browser: https://jalonso1979.github.io/puremacro/
>
> I look forward to your thoughts and feedback!
