# Arellano: original-code replication and grid sensitivity

**The archived solver is reproduced. For PureMacro's public calibration, grid
convergence is established at 6,401 debt nodes and 641 income states under
pre-registered tolerances. The published moments themselves are not
grid-converged: on fine grids the same model produces about half the published
spread volatility.** This benchmark compares the original Arellano code, an
independent implementation, and PureMacro's current public solver. It keeps
their model conventions explicit and makes no new fit to the published targets.

## Run it locally

Use the development checkout with Python 3.11+ and the normal development
dependencies. The full study takes a few minutes on the tested workstation.

```sh
python -m pytest tests/test_arellano_2008_study.py -q
MPLBACKEND=Agg python -m tools.reference_validation.run_arellano_study --output /tmp/arellano-study
```

Open `grid_sensitivity.png` or `grid_sensitivity.pdf`. `moments.csv` contains the
target/result table; `results.json` includes every seed, sampling interval,
episode count, solver diagnostic and code hash. The ordinary run needs neither
network access, gfortran nor Octave; frozen external references are included.

## Independent checks

The author's Fortran code converged in 49 price iterations after a documented
portability patch. The independent study solver reproduces all 4,200 debt choices
and default decisions. Prices match exactly, and the largest value difference is
5.3e-11. Separately, the author's MATLAB moment-aggregation block was executed in
Octave on identical path data and windows: all 12 moments agree within 1.1e-14.
These are checks of code and measurement, not a declaration that the published
quantitative model has been fully replicated.

Sources are the [AER paper](https://doi.org/10.1257/aer.98.3.690), the author's
[July 2007 public draft](https://mpra.ub.uni-muenchen.de/7867/1/MPRA_paper_7867.pdf),
and the [archived replication package](https://doi.org/10.3886/E113248V1).
The draft also contains the Table 4 model targets below. Source files, licenses,
hashes and regeneration details are in the local fixture guide,
`tests/fixtures/arellano_2008/README.md`.

## Published targets and the archived grid

The study uses 16 independent seeds, 100,000 quarters per seed and a 1,000-quarter
burn. Business-cycle moments average 100 windows of 74 quarters immediately
before default, each with uninterrupted market access. The intervals below
measure Monte Carlo uncertainty in the **mean across replications**; they are
not parameter confidence intervals or the dispersion of a single 100-window
experiment. The JSON reports that dispersion separately.

| Moment | Published model | Original 200 × 21 grid | 95% MC interval |
|---|---:|---:|---:|
| Debt / output (%) | 5.95 | 6.283 | [6.083, 6.482] |
| Mean annual spread (%) | 3.58 | 3.405 | [3.362, 3.448] |
| Annual default frequency (%) | 3.00 | 2.822 | [2.756, 2.887] |
| Spread SD (percentage points) | 6.36 | 6.009 | [5.891, 6.126] |
| Trade balance / output SD (percentage points) | 1.50 | 1.485 | [1.463, 1.508] |
| Log consumption SD (%) | 6.38 | 6.391 | [6.323, 6.459] |
| Log output SD (%) | 5.81 | 5.826 | [5.754, 5.898] |
| Correlation: spread, output | -0.29 | -0.251 | [-0.265, -0.236] |
| Correlation: trade balance, output | -0.25 | -0.245 | [-0.252, -0.239] |
| Correlation: consumption, output | 0.97 | 0.971 | [0.970, 0.971] |
| Correlation: trade balance, spread | 0.43 | 0.432 | [0.423, 0.441] |
| Correlation: consumption, spread | -0.36 | -0.331 | [-0.347, -0.316] |

Several gaps persist relative to the estimated simulation mean. The paper's
original random draws are unavailable, and its stated 100-window experiment
differs from the archived MATLAB script, which scans candidate defaults 2:300.
An exact replay of the published table is therefore not claimed.

## The archived calibration is not grid-converged

Debt grids are nested with fixed support and an exact zero. Income grids retain
the same log-income endpoints. Every independently solved case has a Bellman
residual below 4e-12 and a price residual below 1e-8. No simulated repayment
choice hits either debt-grid boundary. Despite those checks, moments move:

| Assets × income states | Debt / output (%) | Mean spread (%) | Annual default (%) | Spread SD (pp) |
|---|---:|---:|---:|---:|
| 200 × 21, original Fortran | 6.283 | 3.405 | 2.822 | 6.009 |
| 399 × 21 | 6.322 | 3.248 | 2.663 | 5.402 |
| 797 × 21 | 6.347 | 3.320 | 2.740 | 5.348 |
| 797 × 41 | 6.010 | 4.043 | 3.558 | 4.442 |
| 797 × 81 | 5.491 | 3.504 | 3.080 | 3.477 |
| 797 × 161 | 5.246 | 3.370 | 2.957 | 3.230 |
| 797 × 321 | 5.144 | 3.321 | 2.942 | 3.098 |

The 161- and 321-state rows come from
`run_arellano_study --income-states 41 81 161 321`, with the same seeds, windows
and archived calibration and timing. Small equation residuals and absence of
boundary mass do not establish economic grid convergence. From 81 to 321 income
states the spread standard deviation still falls, debt/output drifts down and
the trade-balance–spread correlation settles near 0.77 against the published
0.43. The published spread volatility of 6.36 percentage points is therefore
what the archived 200 × 21 grid produces at this calibration; on fine grids the
same model generates about half of it. This is the pattern documented by
Hatchondo, Martinez and Sapriza (2010): the discrete state space technique with
evenly spaced grids needs a large number of points to avoid spurious
interest-rate movements. The public-calibration ladder below shows that debt
nodes matter as much: from 801 to 6,401 nodes spread volatility falls by another
0.44 percentage points. Being closer to one published target on a finer grid
is not evidence that all targets have converged, and no parameter is refitted
here.

Hatchondo, J. C., L. Martinez and H. Sapriza (2010), "Quantitative properties
of sovereign default models: solution methods matter", *Review of Economic
Dynamics* 13(4), 919–933, [doi:10.1016/j.red.2010.03.001](https://doi.org/10.1016/j.red.2010.03.001).

## Specification and measurement differences

The archive sets default output to `min(y, 10*.969)` while mean income is about
10.0307. PureMacro normalizes stationary mean income and applies the cap to that
mean. The archived inputs match a 21-state Tauchen grid to printing precision,
although the draft names Tauchen–Hussey. The archive's actual debt grid also ends
at 1.53502537 rather than its nominal 1.5.

Exclusion timing is material: the archive guarantees an additional excluded
period after entry. With only that timing changed to the canonical Bellman
equation, mean debt/output falls from 6.283% to 5.440% and mean spread rises from
3.405% to 4.184%. The public solver's separate 321 × 21 calibration yields
5.055% debt/output and a 4.329% spread. That case also changes normalization,
grid and literal precision, so it is **not** a timing-only comparison.

The study compounds annual spreads as `q**-4 - Rf**4`, uses within-window sample
standard deviations, and retains raw correlations. The target-comparison
default frequency uses the archived repayment-period denominator; the JSON
also reports the hazard with default-entry periods included in exposure. All
counts use the same post-burn sample. The public solver's short built-in
simulation diagnostics use different measurement conventions and are not used
here. The earlier built-in exclusion loop kept an extra excluded period relative
to its canonical Bellman equation. That bug is now corrected; successful
reentry permits a decision in the same quarter. Both public simulation
interfaces reset debt to the zero node and agree with the same seed. The shared
study simulator continues to follow each case's Bellman timing.

## Check the public calibration's grids

```sh
MPLBACKEND=Agg python -m tools.reference_validation.check_arellano_grids --output /tmp/arellano-grids
```

This uses independently solved public-model equations, checked against
`solve_sovereign_default` on a matched 401 × 9 grid. Calibration, support,
canonical timing and measurement remain fixed. Default runs compare 401/801
asset nodes and 321/641 income states, with 16 paired seeds per case. That
short ladder does not pass; the next section gives the one that does. The public
timing and measurement contract is documented in the
[sovereign guide](sovereign_default.md).

Every percentage moment must change by at most 0.10 percentage points, and
every correlation by at most 0.02. The bound includes the absolute paired mean
change **plus** its 95% Student-t half-width. Both the final debt-grid refinement
and the final income-grid refinement must pass, with at least 797 debt points
and 321 income states on the finest grid. These are numerical tolerances;
published targets do not determine acceptance. A nonzero exit status means
convergence has not been established, distinguishing a large grid shift from
insufficient simulation precision. A passed baseline does not validate new
parameter values, policy effects or the long-term extension.

The report records every seed, source hash, residual, moment and comparison.
Verified solution caches can be reused with `--periods 500000 --windows 1000`
to reduce sampling uncertainty while keeping the seeds fixed. Earlier archived
results above remain historical evidence for a distinct calibration/timing.

## Public calibration: grid convergence established

```sh
MPLBACKEND=Agg python -m tools.reference_validation.check_arellano_grids --output /tmp/arellano-grids \
  --refinements 1 2 3 4 --income-states 321 641
MPLBACKEND=Agg python -m tools.reference_validation.check_arellano_grids --output /tmp/arellano-grids \
  --refinements 1 2 3 4 --income-states 321 641 --periods 500000 --windows 1000
```

The first command solves debt grids of 801, 1,601, 3,201 and 6,401 nodes at 321
and 641 income states with the certified monotone search. The eight solves took
about two hours on the tested workstation, the largest 35 minutes. The second
command reuses the hash-verified solutions and only re-simulates.

Under the standard design, 23 of the 24 final comparisons passed. The
trade-balance–spread correlation moved 0.017 from 3,201 to 6,401 debt nodes,
inside its 0.02 tolerance, but its bound was 0.0204, so sampling precision could
not classify it. The pre-registered rule then applies the same seeds with
500,000 quarters and 1,000 windows per seed. Under that design every comparison
passes:

| Debt nodes × income states | Debt / output (%) | Mean spread (%) | Annual default (%) | Spread SD (pp) | Correlation: trade balance, spread |
|---|---:|---:|---:|---:|---:|
| 801 × 321 | 4.042 | 3.479 | 3.105 | 3.274 | 0.701 |
| 1,601 × 321 | 4.044 | 3.461 | 3.086 | 3.034 | 0.753 |
| 3,201 × 321 | 4.043 | 3.425 | 3.048 | 2.895 | 0.787 |
| 6,401 × 321 | 4.052 | 3.414 | 3.037 | 2.844 | 0.809 |
| 801 × 641 | 3.984 | 3.453 | 3.084 | 3.215 | 0.704 |
| 1,601 × 641 | 3.993 | 3.423 | 3.064 | 2.959 | 0.760 |
| 3,201 × 641 | 4.002 | 3.378 | 3.012 | 2.813 | 0.798 |
| 6,401 × 641 | 4.003 | 3.362 | 2.997 | 2.774 | 0.813 |

In the final debt refinement, 3,201 to 6,401 nodes at 641 states, the largest
bounds are 0.043 pp for spread volatility, against 0.10, and 0.017 for the
trade-balance–spread correlation, against 0.02. In the final income refinement,
321 to 641 states at 6,401 nodes, the largest is 0.076 pp for spread volatility.
Every solve has a Bellman residual below 2e-12 and a price residual below 9e-9.
The exhaustive re-check of the accelerated policies reaches the same Bellman
residual. The frozen record,
`tests/fixtures/arellano_2008/public_grid_convergence.json`, keeps every seed,
moment, hash, residual and comparison. A test recomputes both comparisons from
it and checks this table.

**What the criterion does not bound.** It limits the last change, not the
distance to the limit. Spread volatility and the trade-balance–spread
correlation have moved in one direction at every refinement. On the 641-state
ladder their successive changes shrink by factors of 0.26 to 0.57 and 0.41 to
0.68 per doubling. Geometric extrapolation with those factors places the limit
of spread volatility 0.01 to 0.05 pp below 2.774, inside its tolerance, and the
limit of the correlation 0.01 to 0.03 above 0.813, possibly outside it. Quote
that correlation to two decimals. The extrapolation is a diagnostic, not part
of the registered criterion.

## Solve the converged design with the public solver

```python
from puremacro.sovereign import solve_sovereign_default

converged = solve_sovereign_default(
    beta=0.95282, r_star=0.017, theta=0.282, rho_y=0.945, sigma_y=0.025,
    b_min=-0.15, b_max=0.33, n_b=6401, n_y=641, sigma_eps=0.0,
    y_penalty_type="arellano", enforce_monotonicity=False,
    tol=1e-10, max_iter=10000, policy_search="monotone",
)
```

The dense search would need arrays of debt nodes × debt nodes × income states,
about 210 GB each at this grid. `policy_search="monotone"` searches the
short-term debt choice in O(n_b log n_b) per income state and returns the same
values as the dense search, and the same policies except at exact value ties
between choices with different bond proceeds. After convergence a blocked exhaustive search over
every candidate certifies it; a repayment-value gap above 1e-11 raises an
error. On the tested workstation this solve took 28 minutes, 138 iterations
and 2.4 GB of memory, with a certification gap of zero. Its prices match the
reference solver's 6,401 × 641 solution within 5.4e-9, with identical policies
and default sets; a test repeats the comparison at 401 × 21, and at 1,601 × 21
under `pytest -m slow`.

The moments in the table above come from the study's simulator: 16 paired
seeds, 500,000 quarters and predefault windows. They carry over to this
solution because its arrays match the reference arrays. `converged.simulate()`
returns a raw path with different measurement conventions, so its statistics
are not the same numbers.

### Explicit debt grids

The 6.1 development solver also accepts `debt_grid=coordinates`, a finite,
strictly increasing vector containing an exact zero. It copies and preserves
those coordinates, which override `n_b`, `b_min` and `b_max`; the existing
generated-grid behavior is unchanged. Maturity restructuring retains a supplied
grid. Price tangents on nonuniform grids use spacing-weighted PCHIP slopes.

Concentrating debt nodes where borrowing and bond prices are sensitive can
improve computational efficiency; see [Hatchondo, Martinez and Sapriza (2010)](https://www.imf.org/-/media/websites/imf/imported-full-text-pdf/external/pubs/ft/wp/2010/_wp10100.pdf).
This motivates a separate high-reentry experiment with the original
`[-0.15, 0.33]` support, linear saving nodes and quadratic borrowing nodes.
Its four crossed cases use 6,401/12,801 debt nodes and 641/1,281 income states;
the simulation design and all four 0.10 moment tolerances remain unchanged.
Both final refinements must pass. The original failed uniform-grid evidence
remains in the [sovereign brief](brief_sovereign_risk.md).

```bash
python -m tools.reference_validation.check_sovereign_curved_grids --output grids_curved
```

The command writes its protocol before solving and saves every grid coordinate,
seed, source hash, certification residual and comparison. A nonuniform grid
alone is not evidence of convergence; assess the completed record.

## Next decision

The public calibration now has a fixed numerical design: 6,401 debt nodes, 641
income states and the precision design for simulated moments. The reentry,
income-volatility and persistence experiments can start from that design, but
each scenario requires its own grid check. Keep the archived and public calibrations separate;
the archived calibration's own convergence is not established beyond 797 debt
points. Maturity and restructuring extensions follow their own external
validation. Any recalibration would be a new result, not part of this
validation.
