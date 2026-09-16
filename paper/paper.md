---
title: "puremacro: macroeconometrics and quantitative macroeconomics on the core scientific-Python stack"
tags:
  - Python
  - macroeconomics
  - econometrics
  - structural VAR
  - local projections
  - heterogeneous agents
  - reproducibility
  - Pyodide
authors:
  - given-names: Jorge
    surname: Alonso Ortiz
    orcid: 0000-0002-5941-9928
    corresponding: true
    affiliation: 1
affiliations:
  - name: Instituto Tecnológico Autónomo de México (ITAM), Mexico City, Mexico
    index: 1
    ror: 029md1766
date: 13 September 2026
bibliography: paper.bib
---

<!-- AUTHOR: the ORCID above is the only "Jorge Alonso Ortiz" (ITAM) record in the
     public ORCID registry (checked 2026-09-13); confirm it is yours. Set `date` to
     the day you submit. -->

<!-- This draft is NOT pinned to an old release: it tracks the current one, 3.4.0.
     Every count in the text (modules, lines, tests, swept library modules,
     validation checks, notebook pairs, commits) was recomputed from the 3.4.0 tree
     on 2026-09-16, and `scorecard.png` was re-verified against `validation.scorecard()`
     on the same tree — 107 checks, 15 subsystems, 19 external / 29 analytical /
     59 internal, all passing, identical to 3.3.0. Re-run the commands in
     RELEASING.md §6.5 before submitting. -->

# Summary

`puremacro` is a Python library for empirical macroeconomics and for solving
quantitative macroeconomic models. It gives researchers, instructors and students
one consistent interface to the field's standard methods, and a way to check the
library's numbers on the machine that produces them.

On the empirical side it covers vector autoregressions (VARs) with recursive,
long-run, sign, sign–zero, narrative and external-instrument identification
[@sims1980; @blanchardquah1989; @uhlig2005; @rubioramirez2010; @antolindiaz2018; @mertensravn2013; @stockwatson2018];
factor-augmented VARs [@bernanke2005]; local projections [@jorda2005]; ARCH, GARCH
and GARCH-MIDAS volatility models [@engle1982; @bollerslev1986; @engleghyselssohn2013];
autocorrelation-robust and weak-instrument-robust inference
[@neweywest1987; @andersonrubin1949; @oleapflueger2013]; dynamic panels
[@arellanobond1991]; difference-in-differences and synthetic control
[@abadie2010; @arkhangelsky2021]; mixed-frequency nowcasting
[@giannone2008; @banburamodugno2014]; and text-based uncertainty indices
[@bbd2016]. On the modelling side it solves dynamic programs by value-function
iteration and endogenous grids [@carroll2006; @iskhakov2017]; heterogeneous-agent
economies, including sequence-space HANK models
[@aiyagari1994; @young2010; @auclert2021]; continuous-time models [@achdou2022];
projection methods on Chebyshev, finite-element and Smolyak grids
[@judd1992; @mcgrattan1996; @smolyak1963; @kruegerkubler2004]; linear
rational-expectations (DSGE) models; quantitative spatial and trade models
[@allenarkolakis2014; @caliendoparro2015]; and climate–economy models
[@nordhaus2018; @golosov2014].

Its estimators and solvers are written against NumPy [@numpy2020], SciPy
[@scipy2020], pandas [@mckinney2010] and Matplotlib [@hunter2007] alone, and the
package contains no compiled code of its own. The same code therefore also runs
in a web browser under Pyodide [@pyodide], which ships those libraries compiled
to WebAssembly.

# Statement of need

An applied macroeconomist's toolkit typically spans several ecosystems:
reduced-form time series and panel econometrics from Python's statsmodels
[@statsmodels2010], arch [@arch] and linearmodels [@linearmodels]; structural
identification from MATLAB toolboxes or R packages such as vars and lpirfs
[@pfaff2008; @adammer2019]; DSGE analysis in Dynare [@dynare2024], which runs
under MATLAB or GNU Octave; and heterogeneous-agent models in Python codes
accelerated with Numba [@auclert2021; @carroll2018hark; @quantecon2024]. Each tool
is good at what it does, but combining them means several installations, often
several languages, sometimes a commercial licence, and conventions that change
from package to package. The cost is highest in teaching, where installation
problems consume class time and students' machines (managed laptops, low-end
hardware, tablets) may not support the full stack.

`puremacro` is written for that audience: instructors, students and applied
researchers in macroeconomics. It puts common estimators and models behind one
set of conventions (shared arguments such as `lags`, `horizon` and `ci`, and
immutable result objects with plotting and table export), keeps the dependencies
of its numerical code to four ubiquitous libraries, and ships the evidence needed
to trust its output: galleries of checks that users can run themselves.

# State of the field

statsmodels, arch and linearmodels are the reference Python implementations of
reduced-form time series, conditional volatility and panel and
instrumental-variable estimation. `puremacro` does not aim to replace them; it
uses them as test oracles (see *Software design*). They do not, however, implement
the identification schemes that define structural macroeconometrics (sign,
sign–zero, narrative and external-instrument restrictions), nor do they solve
heterogeneous-agent or general-equilibrium models. For models, widely used tools
are Dynare, the sequence-space Jacobian toolkit, HARK and QuantEcon.py. In their
domains they are more mature than `puremacro` and, for large heterogeneous-agent
problems, faster: `puremacro` trades speed for portability.

That portability can be stated precisely. As of September 2026 the Pyodide
distribution includes NumPy, SciPy, pandas, Matplotlib and statsmodels, but not
arch, linearmodels or Numba; arch and linearmodels publish no pure-Python wheels
that a browser could install at run time; and HARK and QuantEcon.py declare Numba
as a required dependency, while the sequence-space Jacobian toolkit uses it in its
core modules. In a browser, the volatility, panel-IV and heterogeneous-agent layers
of the usual stack are therefore unavailable, and Dynare needs a different language
runtime altogether.

This also answers the build-versus-contribute question. Individual estimators
could be contributed to existing packages, and some would be welcome there. What
cannot be contributed is the property that makes the collection usable on a
constrained machine: one import surface whose numerical code depends on nothing
beyond four core libraries. That is an invariant of the whole library, enforced by
tests, not a feature that can be added to another project's dependency graph.

# Software design

**An import invariant.** Library modules import NumPy, SciPy, pandas and
Matplotlib (plus requests, in the data layer) at module scope; anything else, from
Parquet support to optional accelerators, is optional and imported only when
available. Two tests enforce the rule against the packages most likely to leak in.
The first imports each of the 547 library modules (examples, teaching helpers,
text-scraping sources and optional Numba kernels are excluded) and fails if
statsmodels, linearmodels, arch or the scraping packages bs4, pdfplumber and pypdf
have entered `sys.modules`. The second repeats the sweep in a subprocess in which
those packages cannot be imported, so a module that would break on a machine
without them fails in continuous integration (CI) on a machine that has them. The
wheel is pure Python, and its base install adds only requests to the four core
libraries; the Parquet and Excel engines are an optional extra.

**Oracles that do not ship.** The packages forbidden at run time are the test
suite's references. Scripts in the repository run statsmodels, arch, linearmodels
and esda once on fixed inputs and store their outputs as package data, so the
validation gallery can compare against them without importing them; an opt-in test
marker recomputes the stored outputs with the installed packages to detect drift.
The reference implementations thus certify the library without becoming its
dependencies, which is what lets breadth and portability coexist.

**Degrading rather than failing.** Where an external tool does better, `puremacro`
uses it if present: seasonal adjustment calls X-13ARIMA-SEATS through statsmodels
when both are installed and otherwise falls back to a native X-11. A `runtime`
module detects the host (CPython or Pyodide), the device class and the four
capabilities that differ away from a workstation (sockets, Parquet, threads and a
writable filesystem), and offers opt-in adaptations such as a browser `fetch`
transport. A few dynamic-programming and projection solvers have optional Numba
kernels and experimental MLX and CuPy paths; NumPy remains the reference path and
the one tested in CI.

**Costs.** Writing everything in vectorised NumPy makes large heterogeneous-agent
problems slower than in JIT-compiled toolkits, and derivatives that other libraries
obtain by automatic differentiation must be coded by hand. The browser is a
best-effort target rather than a supported one: Parquet and Excel files need
engines that not every Pyodide distribution provides. A headless harness in the
repository runs the library in Node.js against a pinned Pyodide, and an opt-in
release gate installs the package there exactly as the playground does and runs a
31-test smoke subset of the suite.

<!-- AUTHOR: re-run `python tools/release_check.py --pyodide` on the submitted
     commit and, if the full validation gallery passes there, say so and name the
     Pyodide version it reported. -->

**Scale.** Version 3.4.0 comprises about 750 modules and 231,000 lines of Python,
exercised by about 14,300 tests that CI runs on Linux, macOS and Windows under
Python 3.11–3.13. Before a release is tagged, a script checks the suite against a
recorded baseline, the import invariant, a snapshot of the public API that must be
regenerated deliberately when the interface changes, that every shipped file still
parses on the oldest supported Python, and that the version string agrees across
the package metadata, changelog and citation file.

## Verification

The validation gallery, `puremacro.validation.scorecard()`, runs 107 checks across
15 subsystems in under a minute, with none of the oracle packages installed; all
pass (\autoref{fig:scorecard}). The checks differ in strength, and each records
its reference. Nineteen compare against an external reference: stored outputs of
statsmodels, arch, linearmodels and esda, SciPy results computed at run time, or a
published table. Agreement with independent implementations is typically at
machine precision (median relative difference of order $10^{-15}$); the exception
is GARCH, whose estimates differ from arch's by up to 0.35% because the two
packages use different optimisers. Twenty-nine checks compare against analytical
results, and 59 test internal consistency, such as agreement between alternative
algorithms and recovery of parameters from simulated data.

A separate replication gallery, `puremacro.replication.scorecard()`, reproduces
published estimates of the return to schooling [@card1995], a textbook logit of
married women's labour-force participation on Mroz's [-@mroz1987] data, and the
output effect of tax changes [@romer2010], as well as two qualitative predictions
of incomplete-markets models [@huggett1993; @aiyagari1994]: precautionary saving
holds the interest rate below the rate of time preference, and the rate falls as
income risk rises. Most model solvers (sequence-space, continuous-time, spatial,
trade and climate) are covered by unit tests but not yet by gallery checks.

![The validation gallery of `puremacro` 3.4.0: 107 checks in 15 subsystems, by kind of reference; all pass. *External reference*: stored outputs of statsmodels, arch, linearmodels or esda, SciPy computed at run time, or a published table. *Analytical result*: a closed form, or an effect planted in simulated data. *Internal consistency*: agreement between alternative algorithms, identities that correct output must satisfy, or recovery of parameters from simulated data. Regenerate with `python paper/make_scorecard_fig.py`.\label{fig:scorecard}](scorecard.png){ width=80% }

# Research impact statement

`puremacro` is young (public since July 2026) and has not yet been used in
published research or by other packages. Its significance is near-term, and rests
on evidence a reviewer can check.

<!-- AUTHOR: add any external use you know of (other courses, theses, working
     papers, downstream code); JOSS weighs realized impact above near-term promise. -->


*Teaching.* The library was written for, and is used in, the author's course
*Macroeconomía Avanzada* at ITAM, whose Fall 2026 edition builds its computational
material on it. The course's 22 Spanish lesson notebooks are in the repository
(`notebooks/course`), and 20 of them call `puremacro` directly.

*Reproducible material.* The repository contains 60 bilingual (English/Spanish)
pairs of worked-example notebooks; all of them, with the lesson notebooks, are also
published as a JupyterLite [@jupyterlite] site that runs them in the browser.

<!-- AUTHOR: the live site's install failure (the 3.2.1 wheel required openpyxl
     while PyPI fallback is disabled) is fixed since 3.3.0. After the next Pages
     deploy, run the first cell of a notebook in a browser to confirm, then delete
     this comment. -->

*Community readiness.* The package is released on PyPI under the MIT licence, with
documentation at <https://jalonso1979.github.io/puremacro/>, contribution
guidelines, and CI on three operating systems.

# AI usage disclosure

Generative AI was used extensively in writing `puremacro`, its documentation and
this paper. Anthropic's Claude models, used through the Claude Code agent (Claude
Opus 5, Claude Fable 5 and Claude Fable 5.1), generated or co-wrote much of the
code, tests, documentation and notebooks: 111 of the 214 commits in the public
repository carry a Claude co-author trailer. Google's Jules coding agent
contributed 20 pull requests (refactoring, performance and test improvements),
each reviewed and merged by the author. Claude also drafted and revised this
paper, including checking its claims against the code and its references against
Crossref.

The author set the scope and made the core design decisions (the import
invariant, validation against stored outputs of reference implementations, and
the shared interface conventions) and reviewed, edited and validated all
AI-assisted output before it was merged. Correctness does not rest on that review
alone: it is checked by the test suite, the validation and replication galleries,
and CI.

<!-- AUTHOR, required before submitting (JOSS treats an incomplete or inaccurate
     disclosure as an ethical breach):
     (1) confirm that the review assertion in the paragraph above is accurate;
     (2) add the AI tools used during private development (2026-04-28 to
         2026-07-20, squashed into the first public commit), and the model and
         version behind Jules;
     (3) refresh the counts: `git rev-list --count HEAD` and
         `git log -i --grep='co-authored-by: claude' --oneline | wc -l`. -->

# Acknowledgements

The author thanks the scientific-Python and Pyodide communities, whose work makes
a browser-capable macroeconomics library possible.

<!-- AUTHOR: JOSS asks for financial support to be acknowledged; name any grant,
     or state that the work received no specific funding. -->

# References
