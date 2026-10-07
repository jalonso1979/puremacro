# Live Dynare fixtures shipped with the package

`dynare_live/` and `dynare_order3_pruned_moments.json` are byte-identical copies of
`tests/fixtures/dynare_live/` and `tests/fixtures/dynare_order3_pruned_moments.json`
in the repository. They are shipped so that notebook 68
(`notebooks/68_third_order_perturbation_trust.py`) can compare puremacro with
Dynare from an installed wheel and in the Pyodide playground. The notebook reads
them with `importlib.resources` and falls back to the repository copy.

- `dynare_live/`: Dynare 7.0 decision rules and pruned simulations for five models
  at orders 2 and 3, run in MATLAB R2026a on 2026-09-20. `manifest.json` records
  SHA-256 hashes of every model, innovation CSV, raw MATLAB export and normalized
  `.npz`. The innovations are NumPy draws (`default_rng(20260920)`) that Dynare read
  from the CSV files. Regenerate with `tools/reference_validation/export_dynare.m`
  and `validate_dynare.py --freeze` (see `tools/reference_validation/README.md`).
- `dynare_order3_pruned_moments.json`: Dynare 8 (snapshot 8-2026-05-26-1803)
  `stoch_simul(order=3, pruning, ar=5)` theoretical moments for six models and
  4,000,000-period simulations for three, recorded on 2026-09-23. It stores hashes of
  its model inputs only; no script in the repository regenerates it. Its `command`
  fields record the `stoch_simul` lines that were run.

The repository copies are the ones the tools write. After refreshing them, copy
them here again; the two sets must stay byte-identical.
