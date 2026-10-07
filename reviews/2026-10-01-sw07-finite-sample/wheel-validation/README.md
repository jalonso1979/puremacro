# Finite-sample SW07 wheel and documentation validation

**Passed.** A fresh offline wheel was built from 973 copied files in the final
working tree, then installed under `/tmp/puremacro-sw07-sampling-wheel/installed`.
Every imported `puremacro` module came from that installation; the checkout was
absent from `sys.path`. Network connections, DNS resolution, and statsmodels
imports were blocked, with successful denial controls before package import.

The installed public API produced exact 15-moment expectations and a full
15 × 15 covariance, and reproducible stationary 156 × 7 samples. The public
runner and CLI each executed `published_fixed` with two replications. Both draws
were regular in this environment; the checks allow unresolved draws and verify
their conservative denominator accounting instead of assuming success.

API and CLI draw rows and full arrays matched exactly. All five CSV exports,
the complete NPZ, manifest hashes, four optimizer-start records, and final
checkpoint arrays passed validation. A second CLI invocation reused both saved
draws and reproduced the evidence. No additional Monte Carlo draws were generated
by that resume check. The CLI returned 0, consistent with zero unresolved draws.

The strict MkDocs build was rerun from a fresh temporary documentation copy
after the completed four-scenario simulation results were added to the English
and Spanish finite-sample and empirical-research guides (all four updated pages).
All 973 original wheel source files and all 377 refreshed
documentation and configuration files matched the checkout after validation.
The existing wheel and installed smoke results were preserved; no rebuild or
simulation rerun was necessary. `docs_validation.json` records the refreshed
snapshot and validation time. The standalone
build environment used Python 3.13.12, NumPy 2.4.6, SciPy 1.18.1, and pandas 3.0.6.
Existing setuptools license deprecation notices and a first-run font-cache
directory notice did not affect the build or artifact checks.

This is installation and workflow evidence, **not** a finite-sample scientific
result from two simulations. No package was published.

Key records:

- `summary.json`, `source_match.json`, and `wheel_contents.json`
- `installed_smoke.json`, `cli_status.json`, and `mkdocs-strict.log`
- `installed-api/` and `installed-cli/`: complete two-draw evidence
- `exact_moments.npz` and `stationary_sample.csv`: public-API smoke artifacts
- `offline_controls/`: positive-control records

The wheel SHA256 is
`503fccbb91af039ae9a3379811eadcee180944d25b55e80855113234c0728ef4`.

The driver `validate_wheel.py` records the workflow. Its stages are `build`,
`smoke`, `cli`, `cli_resume`, `docs`, `match`, and `finalize`. The build stage
requires the temporary source directory not to exist, ensuring a fresh copy;
later stages reuse it. All stages use existing offline tools and dependencies.
