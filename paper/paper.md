---
title: "puremacro: browser-runnable empirical macroeconomics in pure Python"
tags:
  - Python
  - macroeconomics
  - econometrics
  - structural VAR
  - local projections
  - reproducibility
  - Pyodide
authors:
  - name: Jorge Alonso Ortiz
    orcid: 0000-0000-0000-0000  # TODO: fill author ORCID before JOSS submission (see RELEASING.md §6)
    corresponding: true
    affiliation: 1
affiliations:
  - name: Instituto Tecnológico Autónomo de México (ITAM), Mexico City, Mexico
    index: 1
date: 31 May 2026
bibliography: paper.bib
---

# Summary

`puremacro` is a Python library for empirical macroeconomics and quantitative
macro modeling, implemented entirely on the scientific-Python core — NumPy
[@numpy2020], SciPy [@scipy2020], pandas, and Matplotlib — with no compiled or
specialized dependencies at runtime. Because it avoids heavier libraries, it
executes unchanged in the browser through Pyodide [@pyodide], on a tablet, or in
a JupyterLite notebook, as readily as on a workstation. The library spans the
methods an applied macroeconomist reaches for: reduced-form and structural vector
autoregressions [@sims1980; @blanchardquah1989], local projections [@jorda2005],
ARCH/GARCH volatility models [@engle1982; @bollerslev1986],
heteroskedasticity- and autocorrelation-consistent inference [@neweywest1987],
dynamic-panel and weak-instrument estimators, difference-in-differences and
synthetic control, value-function iteration and heterogeneous-agent equilibria
[@aiyagari1994], linear DSGE solution [@smetswouters2007], and text-based
uncertainty indices [@bbd2016].

# Statement of need

Macroeconometrics and computational macro are usually taught and practiced with
tools that demand a substantial local installation — the statsmodels
[@statsmodels2010], `arch`, and `linearmodels` stack — or commercial software,
such as MATLAB for Dynare and most heterogeneous-agent toolkits. That barrier is
invisible to well-resourced users and decisive for everyone else: students on
tablets, instructors in low-bandwidth classrooms, and researchers without budgets
or administrative rights. `puremacro` removes it. Because the library is pure
scientific-Python, a reader can open a browser and reproduce a structural VAR, a
local-projection impulse response, or an Aiyagari equilibrium at zero cost, with
nothing to install.

Breadth alone is not trustworthy, so `puremacro` ships its own evidence. A
built-in **validation gallery** contains 73 cases that continuously check the
library's estimators against independent references — statsmodels [@statsmodels2010],
`arch`, and `linearmodels` where a canonical implementation exists, and
closed-form solutions, published numbers, or internal cross-method identities
elsewhere — across thirteen subsystems (\autoref{fig:scorecard}). A companion
**replication gallery** reproduces published results from heterogeneous-agent and
DSGE models. Both are wired into the CI workflows shipped with the package and
re-run in a single line, `puremacro.validation.scorecard()`, including in the
browser, so a user can verify the library on the same machine that runs it. No existing macro-econometrics
package, to our knowledge, combines this breadth with browser portability and a
continuously verified, self-contained correctness record.

# Features

- **Time series and identification:** VAR, BVAR (Minnesota), VECM, and TVP-VAR;
  structural identification via Cholesky, long-run (Blanchard–Quah), sign and
  sign-zero restrictions, proxy/external instruments, and heteroskedasticity;
  impulse responses, variance decompositions, and bootstrap bands.
- **Single-equation and panel:** local projections (including IV and panel),
  dynamic-panel GMM, HAC and weak-instrument-robust inference,
  difference-in-differences, and synthetic control.
- **Computational macro:** value-function iteration, the endogenous grid method,
  Aiyagari and Huggett equilibria, and linear DSGE solution.
- **Reproducibility and teaching:** an in-browser JupyterLite playground and
  bilingual (English/Spanish) documentation and example notebooks.

![Validation coverage: 73 cases across 13 subsystems, each checked against an independent reference.\label{fig:scorecard}](scorecard.png){ width=70% }

# Acknowledgements

The author thanks the scientific-Python and Pyodide communities, whose work makes
a browser-runnable macroeconometrics library possible.

# References
