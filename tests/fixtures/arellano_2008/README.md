# Arellano (2008): archived code and independent numerical references

This is a numerical validation fixture, not a claim that PureMacro's public
solver reproduces the paper's published moments. See the comparison and
remaining discrepancies in [the study guide](../../../docs/arellano_replication.md).

## Provenance and licenses

Cristina Arellano, “Default Risk and Income Fluctuations in Emerging Economies,”
*American Economic Review* 98(3), 690–712 (2008),
[DOI 10.1257/aer.98.3.690](https://doi.org/10.1257/aer.98.3.690).
The author's [public July 2007 draft](https://mpra.ub.uni-muenchen.de/7867/1/MPRA_paper_7867.pdf)
also contains the Table 4 model targets used here. The draft is not the final
typeset article; we do not bundle the paper.

`original/` contains unmodified files from the author's
[replication archive, version 1](https://doi.org/10.3886/E113248V1), retrieved
2026-10-26 using its public file-preview endpoint. `source_manifest.json` records
SHA-256 hashes. The archive assigns **BSD-3-Clause** to code and **CC-BY 4.0** to
data/text, copyright 2008 American Economic Association. Its full license and
attribution are retained in `original/LICENSE.txt`. These terms apply to the
bundled original files; the new study tools are PureMacro code.

## Original Fortran reference

```sh
python tools/reference_validation/run_arellano_original.py --output /tmp/arellano-fortran
```

Requires gfortran and LAPACK (Accelerate on macOS), only for regeneration.
The frozen `reference.npz` was produced by actually running the author's code;
`reference_manifest.json` records its compiler, commands, source hashes and
convergence. `portability.patch` and `lapack_shim.f90` document these adaptations:
local file paths, cold start because warm-start files are absent, explicit
integer conversion of the zero-debt index, replacement of IMSL matrix inversion
with LAPACK DGESV, and an exported convergence diagnostic. The economic loops
and equations are unchanged. Unsuffixed Fortran literals retain default-real
precision before conversion to real(8); do not add `-fdefault-real-8`.

The independent NumPy implementation matches all 4,200 debt choices and default
decisions, prices exactly, and values within 5.3e-11 on the recorded run. Tests
allow 1e-8 for values/prices and require identical choices. This validates the
**study implementation on the archived grid**, not the public API calibration.

## Exact numerical conventions

- Code calibration: beta .95282, risk aversion 2, gross quarterly risk-free
  return 1.017, reentry probability .282. These are the Fortran literal values,
  not a new fit. The paper rounds beta to .953.
- Income is `10 * exp(YMAT)`. Default consumption is `min(income, 10*.969)`;
  stationary mean income is about 10.0307. The public solver instead normalizes
  stationary mean income and bases the cap on that mean.
- The 21-point archived inputs match Tauchen (1986), rho .945, innovation SD
  .025 and three unconditional standard deviations, to 5e-9. Although the draft
  names Tauchen–Hussey, we use the actual archived inputs as the reference.
- `process.f90` produces 200 assets from about -3.3 to **1.53502537**, with
  zero exactly at index 136 (Python). Midpoint refinements produce 399 and 797
  points with identical endpoints and zero; we do not change the support while
  testing grid resolution. Income refinements keep the log-income endpoints.
- The archived transition rows have mass errors up to 1.05e-8. We preserve the
  matrix in the solver, including tiny negative prices in fully defaulting
  regions. Sampling normalizes rows. No chosen repayment price is nonpositive.
- Archive exclusion value is `W = u_d + beta P[(1-theta)W + theta V(0)]`,
  while default entry has `D = u_d + beta P W`. The canonical timing uses
  `D = W`. Thus the archive guarantees an extra excluded period after entry.
  We test both explicitly and do not silently alter the public Bellman equation.
- CRRA utility is -1/c. PureMacro uses 1-1/c; the additive constant does not
  change choices. The study's public-solver bridge translates units and signs.

## Simulation and moment reference

```sh
python -m tools.reference_validation.validate_arellano_moments --output /tmp/arellano-octave
python -m pytest tests/test_arellano_2008_study.py -q
```

The first command executes the author's MATLAB aggregation block in Octave on
an identical supplied path and windows. It does not compare MATLAB and NumPy
random-number generators. `moments_reference.json` records all 12 moments,
adaptations and errors (maximum 1.1e-14). Ordinary tests require neither compiler
nor Octave and use the frozen references offline.

The main study uses 16 independent seeds (1979–1994), 100,000 quarters per seed,
a 1,000-quarter burn and the first 100 eligible windows of 74 quarters before
default. Each window must contain uninterrupted market access. This follows
the paper's 100-window count; the archived script instead scans candidate
defaults 2:300. It is not a replay of the unpublished original random draws.
Log consumption/output are demeaned; each window's sample SD and correlations
are computed first and then averaged. Debt/output is minus mean next-period
assets divided by mean output, averaged across windows.

Spreads compound quarterly bond returns: `100 * (q**-4 - Rf**4)`. The archive
default-frequency denominator counts repayment periods, so the target-comparison
column is `100 * (1 - (1 - entries/repayment_periods)**4)`. We also report the
hazard using **repayment periods plus entry periods** as exposure. Both numerator
and denominator use the same post-burn sample, unlike the archived script's
minor burn-count inconsistency. Excluded quarters are not new default entries.

Student-t intervals across independent seeded replications quantify Monte
Carlo error in the reported mean. `replication_sd` separately measures variation
across 100-window experiments. These are not parameter confidence intervals or
an acceptance test for matching a single published simulation draw. Fixed
100-window samples mean increasing path length mainly refines the unconditional
default-frequency estimate; more replications/windows are needed to refine
business-cycle moment estimates.

## Public-calibration grid convergence record

`public_grid_convergence.json` is the passing `check_arellano_grids` report for
PureMacro's public calibration, recorded on 2026-10-27. It covers debt grids of
801 to 6,401 nodes and 321 and 641 income states, solved with the certified
monotone search, under the pre-registered precision design: 16 seeds, 500,000
quarters, a 1,000-quarter burn and 1,000 windows. It keeps the protocol,
tolerances, source and array hashes, versions, every seed's moments and both
final comparisons. Tests recompute the comparisons from the per-seed moments
and check the guide's table. It records a numerical property of this
calibration only; it is not a published-moment replication.

## Archived-calibration grid convergence record

`archive_grid_convergence.json` is the passing
`check_arellano_grids --calibration archive` report for the archived Arellano
(2008) calibration with the author's exclusion timing, Fortran default-real
literals and 200-node grid refined by midpoint insertion, recorded on
2026-11-05. It covers 797 to 6,369 debt nodes and 321 and 641 income states,
solved with the certified monotone search, under the pre-registered precision
design (16 seeds, 500,000 quarters, a 1,000-quarter burn and 1,000 windows).
`preceding_standard_design` keeps the 100,000-quarter, 100-window comparisons
whose one unclassifiable bound triggered that design, with every case mean.
Tests recompute the final comparisons and check the guide's table. It records
a numerical property of this calibration only; it is not a published-moment
replication.

## Reproduce the comparison

```sh
MPLBACKEND=Agg python -m tools.reference_validation.run_arellano_study --output /tmp/arellano-study
```

Outputs are JSON (all runs, seeds, diagnostics), CSV and PNG/PDF figures. The
baseline and nested grids use archived timing. A timing-only counterfactual and
a separate current-public-API case expose specification differences. The latter
uses the same study simulator with canonical timing, not the public solver's
short built-in simulation diagnostics. Results are reviewed as discrepancies;
the tool fails on equation/reference failure or insufficient episodes, not
merely because published empirical moments differ.
