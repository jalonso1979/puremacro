# MATLAB reference solutions of the legacy trade model on the clean OECD 2020 table

`run_scenarios_clean.m` is the driver that produced `trade_reference_solutions_oecd2020.npz`
(3 October 2026, MATLAB R2026a). It runs the author's legacy MATLAB trade model, whose code
(`calibrar.m`, `ff_equi.m`, `ff_eval.m`) is the external reference of `puremacro.trade`'s
equilibrium solver and is **not** part of this repository, on `data_77c_11s_clean.mat`, a
MATLAB copy of `puremacro/trade/_datafiles/icio_77c_11s_oecd2020.npz`
(`scipy.io.savemat(path, {"data": table})`).

The driver transcribes the scenario loop of the original `Main77c_11s.m` (base and the six
US reciprocal-tariff scenarios t10, t10_25, t10_54, t10_75, t10_125, t10_145, where the
suffix is the tariff on China and Hong Kong) into named cases, seeds t10 from the base
solution and the other scenarios from t10 as the original did, and keeps the original Newton
settings (finite-difference Jacobian with step 1e-2, damping 0.5 on the first three iterations,
L1 residual tolerance 0.0025). The model functions are called unchanged. Each scenario is
saved as `results_77c_11s_<scenario>_clean.mat`; `tools/extract_trade_references.py` copies
the arrays verbatim into the bundled `.npz` and writes `REFERENCE_MANIFEST_OECD2020.json`.

To reproduce: place the three model files and `data_77c_11s_clean.mat` next to the driver and run
`matlab -batch "run('run_scenarios_clean.m')"`. One Newton iteration takes about 45 s on a
12-core laptop (2,001 unknowns, 2,001 model evaluations per Jacobian).

## Tariff scenarios (4 October 2026)

Only the base scenario is bundled. The legacy model's tariff path from the base folds at a US
tariff of about 0.89% on the clean table, so the scenarios have no equilibrium connected to the
base. The cause, the logs of every attempt and the control on the legacy table are in
`reviews/2026-10-04-clean-table-tariff-scenarios/REPORT.md`. The equilibrated Newton is now in
the library as `solve_trade_equilibrium(calib, method="equilibrated_newton")`. The scripts used
are kept here:

* `equilibrated_newton.py <scratch_dir> <rate> <tag>`: the stand-alone prototype of the library
  method (Ruiz row/column equilibration, backtracking) on puremacro's transcription of the
  legacy residual. With `PM_TABLE=legacy` it reproduces MATLAB's bundled t10 reference from a
  cold start to a maximum relative difference of 2.1e-12.
* `equilibrated_lm.py <scratch_dir> <rate> <tag> [warm.mat]`: Levenberg–Marquardt on the
  same equilibrated system, optionally from a stalled iterate.
* `wedge_identity.py [oecd2020|legacy] [rate]`: verifies the per-country budget identity
  `λ_j' F = W1 + W2 + W3 + W4` of the legacy residual and prints how each accounting wedge
  responds along the weakest directions of the equilibrated Jacobian.
* `fold_trace.py [n_steps] [dq]`: follows the clean-table tariff path in Costa Rica's mean log
  price, which passes the fold that natural continuation in the tariff cannot.
* `polish_rate.m`: loads a puremacro root, checks it was computed for the same tariff
  schedule, and re-solves it with the unchanged MATLAB model functions, so that any future
  bundled reference is MATLAB's own solution and remains an external check.

The first two Python scripts write to `<scratch_dir>/matlab/`, which must exist.
