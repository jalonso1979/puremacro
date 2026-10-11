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

<!-- This draft is NOT pinned to an old release: it tracks the current one, 4.6.0.
     Every count in the text (modules, lines, tests, swept library modules,
     validation checks, notebook pairs, commits) was recomputed from the 4.6.0 tree
     (commit d2634362) on 2026-10-04, and `scorecard.png` was regenerated from
     `validation.scorecard()` on the same tree: 114 checks, 16 subsystems,
     21 external (14 package + 5 SciPy + 2 published) / 31 analytical / 62 internal,
     all passing (4.4.0 had 110: 20 / 31 / 59; 3.4.0 had 107: 19 / 29 / 59). Re-run
     the commands in RELEASING.md §6.5 before submitting. -->

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
immutable result objects with plotting and table export) and ships the evidence
needed to trust its output: galleries of checks that users can run themselves.

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

As of September 2026 the Pyodide distribution includes NumPy, SciPy, pandas,
Matplotlib and statsmodels, but not arch, linearmodels or Numba; arch and linearmodels publish no pure-Python wheels
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
Matplotlib (plus requests, in the data layer) at module scope; anything else is
imported only when available. Two tests enforce the rule against the packages most
likely to leak in.
The first imports each of the 597 library modules (examples, teaching helpers,
text-scraping sources and optional Numba kernels are excluded) and fails if
statsmodels, linearmodels, arch or the scraping packages bs4, pdfplumber and pypdf
have entered `sys.modules`. The second repeats the sweep with those packages
blocked, so a module that needs them fails in continuous integration (CI) even
where they are installed. The
wheel is pure Python, and its base install adds only requests to the four core
libraries.

**Oracles that do not ship.** The packages forbidden at run time are the test
suite's references. Repository scripts run statsmodels, arch, linearmodels and
esda once on fixed inputs and store their outputs as package data, so the
validation gallery can compare against them without importing them; an opt-in test
marker recomputes the stored outputs with the installed packages to detect drift.
The reference implementations thus certify the library without becoming its
dependencies, which is what lets breadth and portability coexist.

**Degrading rather than failing.** Where an external tool does better, `puremacro`
uses it if present: seasonal adjustment calls X-13ARIMA-SEATS through statsmodels
when both are installed and otherwise falls back to a native X-11. A `runtime`
module detects the host (CPython or Pyodide) and the four capabilities that differ
away from a workstation (sockets, Parquet, threads and a
writable filesystem), and offers opt-in adaptations such as a browser `fetch`
transport. Some solvers have optional Numba kernels and experimental MLX and CuPy
paths; NumPy remains the reference path, tested in CI.

**Costs.** Writing everything in vectorised NumPy makes large heterogeneous-agent
problems slower than in JIT-compiled toolkits, and derivatives that other libraries
obtain by automatic differentiation must be coded by hand. The browser is a
best-effort target rather than a supported one: Parquet and Excel files need
engines that not every Pyodide distribution provides. A headless harness runs the
library in Node.js against a pinned Pyodide, and an opt-in release gate installs
the package there as the playground does and runs a 31-test smoke subset. On
3 October 2026 the harness ran the whole validation gallery inside Pyodide 0.28.3
and 314.0.5 (the playground's version): all 114 checks passed on both, with four
margins differing from native arithmetic by at most $5 \times 10^{-4}$.

<!-- AUTHOR: re-run `python tools/pyodide_gallery.py` on the submitted commit and
     update the sentence above if the counts or Pyodide versions changed. -->

**Scale.** Version 4.6.0 comprises about 810 modules and 282,000 lines of Python,
exercised by about 19,100 tests, of which CI runs 18,900 on Linux, macOS and Windows
under Python 3.11–3.13 on every push and the remaining slow, reference and
replication tests weekly and on every release tag, where they gate publication.
Before each release tag, a script checks the suite against a recorded baseline,
the import invariant, a snapshot of the public API, syntax on the oldest supported
Python, and version agreement across the package metadata, changelog and citation
file.

## Verification

The validation gallery, `puremacro.validation.scorecard()`, runs 114 checks across
16 subsystems in under a minute, with none of the oracle packages installed; all
pass (\autoref{fig:scorecard}). The checks differ in strength, and each records
its reference. Twenty-one compare against an external reference: stored outputs of
statsmodels, arch, linearmodels and esda, SciPy results computed at run time, a
published table, or, for the trade model, the solution of the author's legacy
MATLAB implementation. Agreement with independent implementations is typically at
machine precision (median relative difference of order $10^{-15}$); the exception
is GARCH, whose estimates differ from arch's by up to 0.35% because the two
packages use different optimisers. Thirty-one checks compare against analytical
results and 62 test internal consistency.

A separate replication gallery, `puremacro.replication.scorecard()`, reproduces
published estimates of the return to schooling [@card1995], a textbook logit of
married women's labour-force participation on Mroz's [-@mroz1987] data, and the
output effect of tax changes [@romer2010], as well as two qualitative predictions
of incomplete-markets models [@huggett1993; @aiyagari1994]: precautionary saving
holds the interest rate below the rate of time preference, and the rate falls as
income risk rises. Most model solvers (sequence-space, continuous-time, spatial,
and climate) are covered by unit tests but not yet by gallery checks; the trade model gained its first four checks in 4.6.0, on a 77-country, 11-sector OECD table.

![The validation gallery of `puremacro` 4.6.0: 114 checks in 16 subsystems, by kind of reference; all pass. *External reference*: stored outputs of statsmodels, arch, linearmodels or esda, SciPy computed at run time, or a published table. *Analytical result*: a closed form, or an effect planted in simulated data. *Internal consistency*: agreement between alternative algorithms, identities that correct output must satisfy, or recovery of parameters from simulated data. Regenerate with `python paper/make_scorecard_fig.py`.\label{fig:scorecard}](scorecard.png){ width=80% }

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

*Reproducible material.* The repository contains 70 bilingual (English/Spanish)
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
Opus 5, Claude Opus 5.5, Claude Fable 5 and Claude Fable 5.1), generated or
co-wrote much of the code, tests, documentation and notebooks: 205 of the 344
commits in the public repository carry a Claude co-author trailer. Google's Jules coding agent
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
     disclosure as an ethical breach). Only you can confirm items 1-4; do not
     submit with this comment still here.
     (1) TOOLS AND MODELS, PUBLIC PERIOD (2026-07-20 onward). The commit trailers
         on origin/main at d2634362 (4.6.0) name: Claude Opus 5 (105 commits),
         Claude Opus 5.5 (31), Claude Fable 5.1 (29), Claude Opus 5 (1M context)
         (26) and Claude Fable 5 (14); the sentence above now names all four model
         families. Confirm that no other assistant (Copilot, ChatGPT, Cursor,
         Gemini, ...) was used without leaving a trailer.
     (2) TOOLS AND MODELS, PRIVATE PERIOD (2026-04-28 to 2026-07-20, inside the
         private `uncertainty_examples` monorepo and squashed into the first
         public commit, so no trailer records it): list every AI tool and model
         used then, and say roughly how much of the code it produced. The
         sentence above currently covers only the public repository.
     (3) JULES: 20 commits by google-labs-jules[bot], merged through 20 pull
         requests, are on origin/main. Name the model and version behind Jules
         (the commits do not record it), and confirm "each reviewed and merged
         by the author".
     (4) REVIEW ASSERTION: confirm that "reviewed, edited and validated all
         AI-assisted output before it was merged" is literally true, including
         for multi-agent workflow commits; if not, soften it to what is true.
     (5) COUNTS ("205 of the 344 commits"): recomputed on 2026-10-04 at
         d2634362 with RELEASING.md §6.5, i.e.
         `git rev-list --count origin/main` and
         `git log origin/main -i --grep='co-authored-by: claude' --oneline | wc -l`.
         Re-run on the submitted commit; commits after 4.6.0 will move both. -->

# Acknowledgements

The author thanks the scientific-Python and Pyodide communities, whose work makes
a browser-capable macroeconomics library possible.

<!-- AUTHOR: JOSS asks for financial support to be acknowledged; name any grant,
     or state that the work received no specific funding. -->

# References
