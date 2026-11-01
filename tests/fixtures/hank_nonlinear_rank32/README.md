# Bounded rank-32 compression refinement

This separate experiment uses the same 80-by-3 household grid, training seed
1201, 160 training innovations of standard deviation `1e-7`, and held-out seed
9917 with 48 innovations of standard deviation `1e-4`. Only the distribution
snapshot rank increases from 16 to 32. Numerical tolerance stays `1e-8`; horizon
and branching criteria stay `1e-6`. No coefficient is fitted to held-out shocks.

**The experiment finished, but the 48-date nonlinear validation did not.**
`record.json` reports `completed=true`, `converged=false`, 43 accepted dates
(zero-based periods 0–42), and `failure.kind="time_budget_exhausted"` before
period 43. Every stored action passed both refinement criteria and the finite
equation checks before it was transported. There is no failed or still-running
last action.

The 600-second budget was a soft limit checked before each correction. The last
accepted correction took 58.3932 seconds and finished after that limit; total
solve time was 635.3227 seconds and rollout wall time was 636.0066 seconds.
These are observations under verified single-thread Accelerate, with other
verification work sharing the host. They are not hard runtime guarantees.

The accepted prefix has market residual maximum `9.414786110006207e-9`,
Euler/KKT maximum `9.95246505398876e-4`, and terminal distribution-projection L1
maximum `1.5630467921616474e-4`. It still uses horizon 64 on 13 dates. The prefix
does not establish rank 32 as a faster, fully validated replacement for either
of the completed 48-date histories in `../hank_nonlinear_continuation/`.

`record.json`, CSV paths/prices and `nonlinear.npz` are atomically checkpointed
after each accepted date. The NPZ additionally stores asset/income grids and
the income-transition matrix. The offline tests independently reconstruct
lottery transport, household budgets, taxes, returns, asset and goods markets,
and Euler/KKT errors. Source snapshots and artifact hashes bind the experiment.

Reproduce in a new directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/tmp/puremacro-rank32-mpl PYTHONPATH=. \
python tools/reference_validation/validate_hank_nonlinear_rank32.py \
  --output /tmp/puremacro-rank32-new --max-seconds 600
```

The elapsed-time stop point may differ on another host. The criterion values,
saved-state accounting and distinction between completed experiments and
completed histories remain the relevant reproducibility checks.
