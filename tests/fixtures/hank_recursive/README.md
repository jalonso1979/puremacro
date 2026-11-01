# Solved stochastic HANK and two-asset evidence

`record.json` is the completed 2026-11-01 validation of the current numerical
sources. Regenerate into a **new** directory with:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
PYTHONPATH=. python tools/reference_validation/validate_hank_recursive.py --output /tmp/hank-recursive-new
```

The record binds exact numerical source bytes, all CSV/NPZ artifacts, seeds,
calibrations, tolerances and library/BLAS identities. The helper verifies
single-thread Apple Accelerate mode when that backend is present. Times are
single observations, not repeated performance estimates. Source-bound tests
recompute Schur invariance, held-out simulations, compression errors and
two-asset fiscal/resource/return accounts from these artifacts.

The snapshot bases use 160 training innovations from seed 1201, standard
deviation 1e-7. Evaluation uses 48 different innovations from seed 9917,
standard deviation 1e-4. These are local first-order equilibrium policies,
with a nonlinear full-distribution rollout audit and nonlinear finite trees
using the solved local stochastic tail. All household marginal controls
remain in the dense Schur system; only the distribution state is reduced.

- One asset: 80 asset points, 3 income states. Full/reduced distribution ranks
  236/16, Schur dimensions 481/261; held-out linear path gap 4.44e-8.
- Two assets: 12 by 12 asset points, 2 income states. Full/reduced distribution
  ranks 285/16, Schur dimensions 867/598; held-out linear path gap 2.90e-10.
- At the fixed 1e-5 audit tolerance, the larger-amplitude one-asset full and
  reduced policies fail (1.65e-5 and 2.74e-5). Both pass at one-quarter shock
  amplitude. Two-asset full/reduced audits pass at the original amplitude
  (8.52e-6 and 9.49e-6), close to that threshold. These outcomes are retained.
- Every corrected finite tree at horizons 4, 8, 16 and 32 clears its equations
  below 1e-8. Dense and Krylov solvers agree within 2.07e-8 (one asset) and
  8.31e-10 (two assets).
- Euler/KKT, upper-grid mass and projection errors remain separate from
  aggregate convergence. The 10/15/20 grid checks recalibrate asset supplies;
  they do not certify convergence for a fixed-supply economy.

`prior_variance_only/` preserves the first completed compression experiment,
including failed audits/determinacy checks, every original artifact, and the
exact source files under `sources/<original path>.txt`. Those hashes refer to
the archived sources, not today's runtime. Truncating distribution variance
alone changed determinacy in several two-asset cases. The current basis first
preserves contemporaneous market observables and inherited illiquid wealth,
then reduces remaining directions. It rejects ranks below the required span.
Do not overwrite the prior record or describe its failed cases as successes.

The final nonlinear audit uses the previously quoted consol price for its
realized return, even when an approximate policy misses an asset market.
Fiscal taxes finance liquid interest and the real consol coupon, excluding
capital gains. This two-asset closure is an original fixed-supply consol
economy, not a productive-capital replication or global stochastic solution.
