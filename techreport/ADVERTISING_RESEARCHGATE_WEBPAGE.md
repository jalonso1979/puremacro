> **Draft claims corrected, 19 September 2026.** Describe feature availability separately from validation. No universal or bitwise portability claim is supported. The 107-case gallery mixes internal, analytical and selected external references, with case-specific tolerances and no trade cases. Exact Hicksian EV/theorem certification is unavailable. See [current structural validation status](../docs/STRUCTURAL_VALIDATION_STATUS.md).

# Promotional and Dissemination Package: puremacro Technical Report
## Tailored Materials for ResearchGate, Personal Webpage, and Academic Networks

This package provides ready-to-use materials to announce and disseminate the comprehensive technical report on `puremacro`.

---

## 1. ResearchGate Dissemination Materials

### 1.1 Working Paper / Technical Report Metadata
When uploading the compiled PDF (`puremacro_technical_report.pdf`) to ResearchGate as a **Working Paper** or **Technical Report**:

- **Title**:\
  `puremacro: A Unified, Dependency-Minimal Scientific-Python Engine for Quantitative Macroeconomics and Macroeconometrics`
- **Subtitle**:\
  `Architecture, Algorithmic Foundations, Numerical Validation, and Browser-Native Reproducibility`
- **Authors**:\
  Jorge Alonso Ortiz (Instituto Tecnológico Autónomo de México - ITAM)
- **Date**:\
  September 2026
- **Publication Type**:\
  Technical Report / Working Paper (v4.2)
- **Research Topics / Disciplines**:\
  Macroeconomics, Econometrics, Computational Economics, Time Series Analysis, Economic Policy, Monetary Economics, Quantitative Methods.
- **Skills & Methods**:\
  Python, Structural VAR, Local Projections, Dynamic Stochastic General Equilibrium (DSGE), Heterogeneous-Agent Models (HANK), Sequence-Space Jacobian, Continuous-Time Macroeconomics, Dynamic Programming, Numerical Projections, Synthetic Control, Difference-in-Differences, Open Source Science, Pyodide, WebAssembly.

- **Abstract**:\
  Copy and paste the abstract directly from the paper:
> The 107-case gallery combines 59 internal checks, 29 analytical cases, 13 package references, five SciPy references and one published reference, with case-specific tolerances. It has no trade cases and does not certify the complete library or every runtime.

### 1.2 ResearchGate Project Update / Feed Announcement Post
Post this in your ResearchGate feed or update your project log:

> 📢 **New Working Paper & Computational Release: puremacro v4.2**
>
> I am excited to share a comprehensive 32-page technical report and working paper detailing **puremacro**, an open-source Python library designed to resolve the *Computational Macroeconomics Trilemma*—unifying methodological breadth, portability within supported environments, and numerical verifiability.
>
> 🔹 **The Problem**: Computational macroeconomics has historically been fragmented across MATLAB/Dynare, R, Stata, and specialized C++/Numba libraries, introducing licensing costs, compiler issues, and friction in research and graduate teaching.
>
> 🔹 **The puremacro Solution**:
> - **Zero Compiled Extensions**: Built strictly on pure NumPy, SciPy, pandas, and Matplotlib.
> - **Universal Execution**: Runs natively on workstations and client-side in web browsers via Pyodide/WebAssembly (iPad/JupyterLite).
> - **End-to-End Methodological Coverage**:
>   1. *Macroeconometrics*: SVAR (Cholesky, Blanchard-Quah, Sign, Zero, Narrative, Proxy SVAR-IV) and Local Projections (LP-HAC, Lag-Augmented, Panel with Driscoll-Kraay).
>   2. *DSGE Engine*: Native Dynare `.mod` parser, 2nd-order perturbation with pruning, and Bayesian NUTS estimation via exact analytical Kalman score gradients.
>   3. *Heterogeneous Agents*: Sequence-Space Jacobian (SSJ/HANK) bridge coupled into `.mod` files, continuous-time HJB-KFE solvers, and non-stochastic density transitions.
>   4. *Continuous Projections*: Orthogonal Chebyshev collocation, Smolyak sparse grids, and Finite Elements with Fischer-Burmeister NCP borrowing constraints.
>   5. *Spatial & Trade General Equilibrium*: Allen-Arkolakis topography and Caliendo-Parro exact hat algebra with flexible nested CES and variable markups.
> The 107-case gallery combines 59 internal checks, 29 analytical cases, 13 package references, five SciPy references and one published reference, with case-specific tolerances. It has no trade cases and does not certify the complete library or every runtime.
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
The 107-case gallery combines 59 internal checks, 29 analytical cases, 13 package references, five SciPy references and one published reference, with case-specific tolerances. It has no trade cases and does not certify the complete library or every runtime.
  </p>

  <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-top: 16px;">
    <a href="techreport/puremacro_technical_report.pdf" target="_blank" style="background-color: #1e3d59; color: #ffffff; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-size: 13.5px; font-weight: 600; display: inline-flex; align-items: center;">
      📄 Download Technical Monograph (PDF, 32 pp.)
    </a>
    <a href="https://github.com/jalonso1979/puremacro" target="_blank" style="background-color: #ffffff; color: #1e3d59; border: 1.5px solid #1e3d59; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-size: 13.5px; font-weight: 600; display: inline-flex; align-items: center;">
      💻 GitHub Repository (v4.2)
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

I am pleased to present the comprehensive technical monograph and v4.2 release of **puremacro**:

> **puremacro: A Unified, Dependency-Minimal Scientific-Python Engine for Quantitative Macroeconomics and Macroeconometrics**\
> *Jorge Alonso Ortiz (ITAM, September 2026)*\
> [📄 Download Technical Report (PDF)](techreport/puremacro_technical_report.pdf) · [💻 GitHub](https://github.com/jalonso1979/puremacro) · [🌐 Interactive Browser Platform](https://jalonso1979.github.io/puremacro/) · [📦 PyPI](https://pypi.org/project/puremacro/)

**Key Highlights:**
- **Zero Compiled Binaries**: Runs on pure NumPy + SciPy + pandas + Matplotlib with 100% WebAssembly (Pyodide) browser execution.
- **Structural Macroeconometrics**: SVAR with Cholesky, Blanchard-Quah, Uhlig Sign, Rubio-Ramírez Sign/Zero, Antolín-Díaz Narrative Sign, and Stock-Watson / Mertens-Ravn Proxy-IV.
- **Local Projections & Causal Inference**: LP-HAC, Lag-Augmented LP, Driscoll-Kraay Panel LP, Callaway-Sant'Anna DiD, Synthetic Control, and Synthetic DiD.
- **DSGE & Bayesian NUTS**: Direct parsing of Dynare `.mod` files, 1st/2nd-order perturbation with pruning, and Hamiltonian Monte Carlo with exact analytical Kalman score gradients.
- **Heterogeneous Agents & HANK**: Sequence-Space Jacobian (Auclert et al.) bridge coupled in `.mod` files, continuous-time HJB-KFE solvers, and non-stochastic density iteration.
- **Continuous Projections**: Orthogonal Chebyshev collocation, Smolyak sparse grids ($d \in [2, 6]$), and Finite Element Galerkin with Fischer-Burmeister complementarity.
- **Spatial & Multi-Sector Trade**: Allen-Arkolakis gravity topography and Caliendo-Parro exact hat algebra with flexible nested CES technology.
The 107-case gallery combines 59 internal checks, 29 analytical cases, 13 package references, five SciPy references and one published reference, with case-specific tolerances. It has no trade cases and does not certify the complete library or every runtime.
```

---

## 3. Social Media & Academic Community Dissemination

### 3.1 Twitter / X Thread (for #EconTwitter)

**Tweet 1 (Hook)**:\
How can we end computational fragmentation in macroeconomics?\
Applied researchers juggle MATLAB/Dynare, R, Stata, & C++/Numba libraries—costing licenses, breaking environments, & eating up class time.\
I'm excited to share a 32-page working paper on **puremacro**: a unified, dependency-minimal Python engine! 🧵👇\
[Link to Paper] #EconTwitter #ComputationalEcon

**Tweet 2 (The Architecture)**:\
The core design invariant: **strictly no package-owned compiled extensions**.\
puremacro runs on pure NumPy, SciPy, pandas, and Matplotlib.\
Because of this, the entire library runs in WebAssembly under Pyodide: you can solve second-order DSGEs, HANK models, or narrative SVARs on an iPad with zero install! 📱💻

**Tweet 3 (Structural Identification Spectrum)**:\
Tired of jumping between MATLAB toolboxes? puremacro unifies the identification spectrum:\
✅ Cholesky recursive & Blanchard-Quah long-run\
✅ Uhlig & Rubio-Ramírez Sign & Zero restrictions\
✅ Antolín-Díaz & Rubio-Ramírez Narrative Sign Restrictions\
✅ Stock-Watson & Mertens-Ravn Proxy SVAR-IV\
✅ Lag-augmented & Driscoll-Kraay Panel Local Projections

**Tweet 4 (DSGE & Bayesian NUTS with Exact Gradients)**:\
A native parser reads Dynare `.mod` syntax directly (no MATLAB required!).\
Solves 1st & 2nd order perturbation with Kim et al. pruning.\
Best of all: Bayesian estimation via No-U-Turn Sampler (NUTS) using **exact analytical Kalman score gradients** via matrix Sylvester solvers. Accuracy also depends on structural derivatives and the numerical solution.

**Tweet 5 (Heterogeneous Agents & Sequence Space)**:\
Coupling micro and macro: define household blocks (`hetagent_block;`) inside your Dynare `.mod` file!\
puremacro uses the Sequence-Space Jacobian (Auclert et al. 2021) fake-news algorithm & continuous-time upwind HJB-KFE solvers (Achdou et al. 2022) to solve HANK models in seconds.

**Tweet 6 (Validation & Pedagogy)**:\
Can you trust the numbers?\
The 107-case gallery combines 59 internal checks, 29 analytical cases, 13 package references, five SciPy references and one published reference, with case-specific tolerances. It has no trade cases and does not certify the complete library or every runtime.
Already powering *Macroeconomía Avanzada* at ITAM with 60 bilingual notebooks!

**Tweet 7 (Links)**:\
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
> To solve this, I have authored a comprehensive 32-page technical monograph detailing **puremacro**, an open-source Python library released on PyPI:
>
> 🔍 **Key Innovations**:
> 1. **Pure Scientific-Python Foundation**: Puremacro ships no compiled extension of its own. Its base dependencies include NumPy, SciPy, pandas, Matplotlib and requests; browser execution and backend agreement require separate tests.
> 2. **Complete Econometric Suite**: Recursive, Blanchard-Quah, Sign, Zero, Narrative, and Proxy SVARs, alongside Lag-Augmented, Panel (Driscoll-Kraay), and Staggered Difference-in-Differences.
> 3. **Native DSGE & Bayesian NUTS**: Direct parsing of Dynare `.mod` syntax, pruned 2nd-order decision rules, and Bayesian estimation via the No-U-Turn Sampler with exact analytical Kalman score gradients derived from matrix Sylvester equations.
> 4. **Micro-Macro Synthesis**: Sequence-Space Jacobian (SSJ) heterogeneous-agent coupling, continuous-time HJB-KFE PDE solvers, and high-dimensional Chebyshev, Smolyak sparse grid, and finite element Galerkin projections.
> The 107-case gallery combines 59 internal checks, 29 analytical cases, 13 package references, five SciPy references and one published reference, with case-specific tolerances. It has no trade cases and does not certify the complete library or every runtime.
>
> The library is already in production use in *Macroeconomía Avanzada* at ITAM, eliminating installation issues for students and enabling 100% browser-based computational learning.
>
> 📖 Read the full technical monograph: [Link to ResearchGate / PDF]\
> 💻 Explore the repository: https://github.com/jalonso1979/puremacro\
> 🌐 Run interactive notebooks in your browser: https://jalonso1979.github.io/puremacro/
>
> I look forward to your thoughts and feedback!
