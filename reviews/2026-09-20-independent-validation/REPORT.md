# Independent Dynare and native OECD validation — 20 September 2026

Both requested follow-ups are implemented and exercised. They found three
issues that ordinary path/accounting tests had missed: asymmetric third-order
mixed derivatives, incorrect OECD final-use placement for the investment
closure, and a mismatch between reported output and the actual transactions
used in calibration. These are corrected. All 197 targeted tests pass.

## Dynare: ten live cases, including risk corrections and simulated moments

Five models ran in **Dynare 7.0 / MATLAB R2026a Update 5
(26.1.0.3346908)**, each at orders two and three. Models cover nonlinear forward
feedback, correlated innovations, nonlinear state transitions, multiple states
and shocks, and RBC dynamics with positive steady states and reordered/static
variables. The source, common innovations and compact external references are
in [the fixture directory](../../tests/fixtures/dynare_live/manifest.json).

The exporter runs Dynare's own perturbation routines and `simult_`, records a
common completion ID and the actual source text, and writes MATLAB exports.
The freezer rejects stale runs, changed source/innovations, wrong orders and
incomplete/nonfinite paths. Fixture creation never calls puremacro to generate
reference values. Rows are unpermuted by `order_var`; columns are aligned by
state/shock labels. The unfolded `gh*` arrays need no factorial adjustment.

All requested tensors are compared, including `ghs2`, `ghxss` and `ghuss`, plus
steady states and innovation covariance. Each model/order uses 2,500 identical
innovations. Path comparison includes every observation; sample means,
covariance and lag-one covariance discard the first 500. Acceptance uses
absolute tolerance 1e-10 and relative tolerance 1e-9.

| Case | Maximum tensor error | Maximum path error | Maximum sample-moment error |
|---|---:|---:|---:|
| correlated_cubic_order2 | 0.000e+00 | 1.388e-17 | 2.429e-17 |
| correlated_cubic_order3 | 8.882e-16 | 1.776e-15 | 2.220e-16 |
| nonlinear_multishock_order2 | 5.551e-17 | 5.551e-17 | 6.939e-18 |
| nonlinear_multishock_order3 | 1.110e-15 | 5.551e-17 | 6.245e-17 |
| nonlinear_state_order2 | 1.388e-17 | 5.551e-17 | 2.776e-17 |
| nonlinear_state_order3 | 1.110e-15 | 5.551e-17 | 7.633e-17 |
| quadratic_feedback_order2 | 4.441e-16 | 2.220e-16 | 1.110e-16 |
| quadratic_feedback_order3 | 4.441e-16 | 3.331e-16 | 2.255e-17 |
| rbc_order2 | 1.377e-14 | 2.096e-13 | 7.860e-14 |
| rbc_order3 | 1.155e-13 | 2.700e-13 | 1.094e-13 |

The RBC `ghxxu` tensor originally differed by **0.0012928883**, despite matching
paths. The code doubled one chain-rule placement instead of adding both state
index permutations. The error was antisymmetric and disappeared when contracted
with x⊗x. The analogous shock-index placements in `ghxuu` needed the same
correction. The implementation now assembles each permutation explicitly,
verified with the nonlinear multiple-shock model as well as RBC.

**Runtime caveat:** all ten exports completed, including the explicit
`DYNARE_EXPORT_COMPLETE` marker, before MATLAB crashed during process shutdown.
The numerical comparisons pass; the MATLAB process did not exit cleanly.
[Processing log](dynare-runtime.log) and [comparison results](dynare-results.json)
are retained. The native crash report contains machine/session details and is
not copied into the repository. This benchmark does not certify arbitrary
models, analytical higher-order moments, callable finite-difference derivatives,
or another runtime. The public parity dashboard still supports orders one/two.

## OECD: actual 2019 member from the provider's 2023 regular archive

Source: [official OECD ICIO dataset page](https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html),
[2016–2020 regular ZIP](https://webfs-sti.oecd.org/files/STI-PIE/ICIO/2023/2016-2020_SML.zip),
and [schema workbook](https://webfs-sti.oecd.org/files/STI-PIE/ICIO/2023/ReadMe_ICIO_small.xlsx).
The complete **2019_SML.csv** contains 3,468 rows × 3,928 columns, representing
77 economies × 45 activities, six final uses per economy, and TLS/VA/OUT margins.
The schema workbook specifies **current million USD**, with basic-price flows
and separate taxes less subsidies. Only the 2019 member was validated.

The ZIP downloaded through Safari after the command-line request encountered
a browser challenge; Safari extracted it automatically. The CSV checksum is:

`985e3141c3da4ad56ccc9dd0721eadf540b363d6929fdbc6d8fbf5417433ac0e`

The [manifest](../../tests/fixtures/trade/oecd_2023/manifest.json) also records the
workbook checksum, source URLs, complete aggregation mappings, raw accounting
measurements and full-calibration adjustments. The original ZIP checksum is
unavailable for this browser-extracted copy; the CSV bytes identify the actual
validated input. The reader also accepts an original ZIP and hashes it when
supplied. Large provider files remain outside the repository.

`load_oecd_icio_granular` now uses exact native labels, accepts row/column
permutations, rejects missing/extra/duplicate/nonfinite data, and checks that
the provider's OUT row agrees with its OUT column. It no longer accepts an
unlabeled positional matrix as this native schema. The reader is explicitly
limited to the **2023 regular** schema; 2025/extended archives are unsupported.

### Findings and adjustments

* The previous six-category calibration assigned the foreign balance to NPISH
  (position 1), while the equilibrium model expects investment there. Native
  final uses now map to **C = HFCE + NPISH + GGFC; I = GFCF + INVNT;
  Cx = DPABR**. Final-demand taxes use the same mapping. Signed inventory
  changes are retained; the source has 453 negative final-demand cells.
* The provider's published transactions and OUT margins are not exactly
  balanced: maximum sales-minus-OUT is **0.4724 million USD**, and maximum
  outlays-minus-OUT is **0.8149 million USD**. These are measured, not declared
  to be roundoff or silently certified as balanced. Regularized calibration
  uses transaction row sales and records the difference from original OUT.
* On the full table, intermediate flows are unchanged. There are 25 inactive
  nodes: household final demand receives 1e-6 million USD per node. Value
  added changes in 41 nodes, totaling **9.835125 million USD**, including the
  inactive-node injections. TLS changes total **−3.444200 million USD**.
  The metadata contains each block's count, maximum absolute change and total.
* Labor/capital allocation of 2/3 and 1/3 and the minimum-VA floors remain
  modeling assumptions. Accounting repair is not a claim about the true
  economic allocation of taxes or factor payments.

The public loader calibrates all **3,465 nodes**. Maximum post-calibration
sales/outlays discrepancy is **1.40e-9 million USD**; budget-share error is
**7.89e-15**; world trade leakage is **−5.82e-10 million USD**. All full-table
accounting checks pass. Productivity is checked by the regularization path.

### Counterfactual checks

A small, conserving aggregation is stored for routine offline regression:
**USA / CHN / REST** and **resources / manufacturing / services**. REST includes
all 75 remaining economies, including the original ROW. Resources cover ISIC
A/B, manufacturing C, and services all remaining activities (including utilities
and construction). Original flows are aggregated before repairs. Every economy,
sector and final-use column is included; no international flows are dropped.

The baseline recovers calibrated output and unit prices/wages/rents. An
independent Leontief inverse reproduces baseline output to **5.59e-9 million
USD**. A 10% tariff on all USA foreign intermediate/final purchases passes:

* Newton and Keller continuation convergence, with price differences below
  **7.28e-13** and output differences below **1.32e-6 million USD**.
* Goods clearing and an independent sum of schedule receipts.
* Zero tariff receipts for the two non-imposing regions.
* Million-to-billion USD scaling: prices differ by at most **2.11e-15**;
  quantities and receipts scale accordingly.

The absolute solver tolerance is **1e-5 million USD (ten dollars)**, scaled to
1e-8 for billion-USD units. A preliminary 1e-8 million-USD tolerance was below
the observed floating-point residual floor; those runs correctly reported
nonconvergence and are not used as successful cases. Final Newton/continuation
residual norms are **1.65e-6 / 3.88e-6 million USD**. The
[full comparison results](oecd-results.json) include all checks and outputs.

The counterfactual uses `tariff_revenue_mode="schedule"`, Leontief inputs and
lump-sum rebates. It retains the model's documented legacy final-composite-price
valuation. This validates the model's numerical/accounting implementation on
these inputs; it does **not** independently establish empirical tariff effects,
Hicksian welfare, full-size native GE convergence, or other providers/editions.

## Reproduction and checks

See [reproduction instructions](../../tools/reference_validation/README.md).
Ordinary CI uses the compact external fixtures and needs no live MATLAB/network.
The native validation command reloads the full provider file and exercises the
public loader. The precise executed commands are in [commands.txt](commands.txt).

[Targeted tests](targeted-tests.log): **197 passed**, with three pre-existing
warnings (two pytest fixture deprecations and one deprecated private import).
The new tests cover external tensor/path/moment references, empirical aggregate
counterfactuals, label permutations, corrupted schemas, ZIP year selection,
negative inventory preservation, final-use mapping and repair disclosures.
The validation matrix, English/Spanish parity documentation, and technical
report source/PDF are updated. Work remains uncommitted.
