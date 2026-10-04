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
