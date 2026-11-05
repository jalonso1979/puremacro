# Independent structural references

These tools compare actual external inputs with puremacro. Ordinary tests use
small frozen fixtures and do not require MATLAB or network access.

## One-asset households and deterministic sovereign default

Run the frozen independent checks without additional dependencies:

```sh
python -m pytest tests/test_research_reliability_references.py -q
```

Regenerate and compare references in a separate validation environment:

```sh
python -m pip install 'sequence-jacobian @ git+https://github.com/shade-econ/sequence-jacobian.git@eb4ac4c3157357a4cacbea4b5a7a281f5b48cab9'
python tools/reference_validation/validate_hank_reference.py --output /tmp/hank-reference
python tools/reference_validation/validate_arellano_reference.py --output /tmp/arellano-reference
```

Both commands return nonzero when their comparison tolerances fail. The SSJ
reference checks household policies, distribution, consumption and Jacobians
with matched grids, tax assumptions and timing. The sovereign reference uses
separately implemented Bellman equations, with assets negative for debt and no
production solver helpers. Manifests record scope and hashes. See
`tests/fixtures/research_reliability/README.md` for tolerances and limitations.
Neither comparison claims to reproduce the full moments of a published paper.

## Published one-asset HANK equilibrium

The full endogenous-labor equilibrium now has a separate author-code benchmark:

```sh
python -m pytest tests/test_models/test_one_asset_hank.py -q
python -m tools.reference_validation.check_hank_equilibrium --output /tmp/hank-equilibrium --convergence
```

It checks the published calibration, all equilibrium markets, linear and
nonlinear paths for both signs, and asset/income/bound/horizon refinements.
See [`hank_equilibrium/README.md`](../../tests/fixtures/hank_equilibrium/README.md)
for exact provenance and the retained failed intermediate refinements.

## Arellano original code, published targets and grid convergence

```sh
python -m pytest tests/test_arellano_2008_study.py -q
MPLBACKEND=Agg python -m tools.reference_validation.run_arellano_study --output /tmp/arellano-study
```

This offline study uses an actual original-Fortran export and an Octave check
of the author's MATLAB moment definitions. It tests nested debt grids, income
grids, exclusion timing and simulation uncertainty. Published-moment gaps and
material grid sensitivity remain visible. See
[`tests/fixtures/arellano_2008/README.md`](../../tests/fixtures/arellano_2008/README.md)
for provenance and external-reference regeneration, and the
[comparison guide](../../docs/arellano_replication.md) for results.

For the public solver's own canonical timing and normalized-income calibration:

```sh
MPLBACKEND=Agg python -m tools.reference_validation.check_arellano_grids --output /tmp/arellano-grids
```

This holds calibration/support fixed and compares paired-seed moments across
asset and income refinements. It returns nonzero if either final refinement
exceeds the documented numerical tolerances after accounting for simulation
uncertainty. Hash-verified solution caches permit longer simulations without
resolving the model. Published moments are not convergence targets.

The public cases use `solve(..., policy_search="monotone")`: for u(c) = -1/c the
optimal debt choice is monotone in assets once actions are sorted by cash
proceeds, so a divide-and-conquer search replaces the n_assets^2 scan and
1,601- or 3,201-node grids (`--refinements 2 3`) become affordable. Every
accelerated solve is then re-checked by an independent exhaustive Bellman
evaluation; a disagreement above 1e-11 fails the solve.

The converged public ladder is `--refinements 1 2 3 4 --income-states 321 641`,
then the same command with `--periods 500000 --windows 1000`, which reuses the
hash-verified solutions. The solves take about two hours on the tested
workstation. The passing record is frozen in
`tests/fixtures/arellano_2008/public_grid_convergence.json`; see the
[comparison guide](../../docs/arellano_replication.md) for the result.

## Fiscal sovereign default: grid convergence

```sh
MPLBACKEND=Agg python -m tools.reference_validation.check_fiscal_grids --output /tmp/fiscal-grids
```

The illustrative fiscal calibration (`tau=0.2, psi=1, gamma_c=gamma_g=2`,
public sovereign parameters otherwise; not fitted to any target) is solved
with `solve_fiscal_sovereign_default(..., policy_search="monotone")` at
`n_b` = 1,601, 3,201, 6,401 and `n_y` = 321, 641. There is no independent
fiscal reference: each solve is certified by its own exhaustive search, and
`tau=1` without private consumption is bit-identical to the Arellano-validated
`solve_sovereign_default`. Sixteen paired seeds (1979-1994) of 500,000
periods after a 1,000-period burn-in measure debt/output, mean spread,
annual default frequency, spread volatility, spending share and the
primary-balance/income correlation. Tolerances (0.10 for percent moments,
0.02 for correlations) and the design are written to `grid_protocol.json`
before solving and are not changed afterwards. A failing ladder is frozen as
`convergence_not_established`. The record is
`tests/fixtures/sovereign_fiscal/fiscal_grid_convergence.json`; see the
[fiscal rules guide](../../docs/sovereign_fiscal_rules.md).

## Fiscal rules: registered predictions

```sh
MPLBACKEND=Agg python -m tools.reference_validation.check_fiscal_rule_predictions --output /tmp/rule_predictions_record.json
```

Predictions about the four illustrative fiscal rules (a 0.02 debt ceiling,
the same with an escape clause below `y = 0.92`, a soft ceiling with
`penalty=0.05`, and a 0.005 deficit limit) were committed in
`tests/fixtures/sovereign_fiscal/rule_predictions_protocol.json` before any
rule was solved. The tool reads that protocol, never changes it, solves each
rule on the coarse 401 x 21 grid, simulates 16 paired seeds of 100,000
quarters and writes the outcomes with a verdict per prediction; a failed
prediction stays failed. The frozen record is
`tests/fixtures/sovereign_fiscal/rule_predictions_record.json`; see the
[fiscal rules guide](../../docs/sovereign_fiscal_rules.md). Rules are compared
only within that one coarse grid.

## Dynare 7.0 / MATLAB

Five source models in `tests/fixtures/dynare_live/` cover quadratic forward
feedback, a cubic model with correlated innovations, nonlinear state dynamics,
nonlinear dynamics with multiple correlated shocks, and a positive-steady-state
RBC model with a static output variable. Both orders two and three are run.

From MATLAB, using **absolute paths** appropriate to your installation:

```matlab
addpath('/path/to/puremacro/tools/reference_validation');
export_dynare('/path/to/Dynare/7.0', ...
              '/path/to/puremacro/tests/fixtures/dynare_live', ...
              '/tmp/puremacro-dynare-live');
```

The exporter copies each model into its own working directory, appends
`stoch_simul(order=2/3,pruning,irf=0,nograph,noprint,nomoments,nocorr)`, and
runs Dynare. It saves the actual decision rules, covariance, labels, versions,
innovations and `simult_` paths. Original model files are not modified.

The checked-in innovations use NumPy `default_rng(20260920)`, 2,500 standard
normal draws transformed by each model's Cholesky covariance. MATLAB reads the
same decimal CSV values; it does not draw another random sample. The first
500 observations are omitted only from the moment comparisons.

From the repository root:

```sh
PYTHONPATH=. MPLBACKEND=Agg python tools/reference_validation/validate_dynare.py \
  --freeze /tmp/puremacro-dynare-live --output /tmp/dynare-comparison.json
```

**`--freeze` replaces reference fixtures.** Use it only on exports from an actual
Dynare run. Omit it to verify existing references. The exporter records a common run ID, source text and a completion marker.
The freezer rejects stale exports and sources/innovations that differ from the
executed inputs. The manifest records hashes of model sources, innovation CSVs,
raw MATLAB exports and normalized NPZ files.
Normalization only unpermutes DR rows to declaration order; state and shock
columns are aligned by labels. The `gh*` arrays contain unfolded derivatives,
so no Taylor-factor or folded-column conversion is applied. `ys` is already in
declaration order. Simulations are compared in levels after dropping Dynare's
initial-condition column and adding puremacro's deterministic steady state.

Every requested tensor, including `ghxxu`, `ghxuu`, `ghxss` and `ghuss`, is tested.
The benchmark also compares innovation covariance, every simulated observation,
and sample means, covariance and lag-one covariance. It does **not** compare
analytical higher-order moments, prove approximation accuracy far from steady
state, or extend the public parity dashboard beyond its supported orders.

Runtime caveat for the recorded run: MATLAB R2026a Update 5 executed all ten
cases and saved complete finite exports, then suffered a segmentation fault
during process shutdown on this machine. The benchmark success refers to the
exported numerical comparisons, not a clean MATLAB exit. Regeneration requires
checking that every model/order export exists and its recorded runtime version
is appropriate. The processing log and the caveat are retained in the review.

## OECD 2023 regular ICIO, year 2019

The provider's [2016–2020 regular archive](https://webfs-sti.oecd.org/files/STI-PIE/ICIO/2023/2016-2020_SML.zip)
and [schema workbook](https://webfs-sti.oecd.org/files/STI-PIE/ICIO/2023/ReadMe_ICIO_small.xlsx)
are linked from the [official OECD dataset page](https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html).
`--download` fetches the archive (about 85 MB) from the OECD site once and keeps
it in `$PUREMACRO_OECD_ICIO_DIR` (default `~/.cache/puremacro/oecd_icio`); a copy
already on disk is passed with `--source` instead. Either way the `2019_SML.csv`
member is checked against the registered MD5 before it is read. The manifest
identifies the **CSV member bytes** by SHA256; an original ZIP hash is also
recorded when the reader is given a ZIP. The workbook specifies current million
USD and basic-price flows, with taxes less subsidies supplied separately. No
foreign-exchange conversion is needed.

```sh
PYTHONPATH=. MPLBACKEND=Agg python tools/reference_validation/validate_oecd.py \
  --download --readme /path/to/ReadMe_ICIO_small.xlsx \
  --output /tmp/oecd-comparison.json
```

`--download` and `--source` validate and calibrate **all** 77 economies × 45
activities through `load_oecd_icio_granular`, then refresh the small empirical
fixture. Omit both to rerun the aggregate model checks on the frozen fixture
without the archive. Such a run explicitly reports that it did not reload the
full native archive.

The reader rejects missing, extra, duplicate or nonfinite labels/data and
inconsistent OUT margins. It measures transaction discrepancies before repairs.
Regularization uses actual row sales, records changes to reported output and
TLS, floors inactive output/low value added, and verifies productivity. It does
not present the adjusted table as the untouched provider table. `regularize=False`
does not promise calibration success for an unbalanced release.

Native final uses map to C = HFCE + NPISH + GGFC, I = GFCF + INVNT, Cx = DPABR.
Signed inventory changes are retained. The investment category must be at
index 1 because the model's foreign-balance closure uses that position.
This native adapter returns three model final uses; synthetic teaching data
retain their existing six-category behavior.

The small fixture aggregates **original** flows before regularization into
USA, CHN and REST (all remaining economies, including provider ROW), and
resources (ISIC A/B), manufacturing (C), services (all remaining activities,
including utilities/construction). The manifest lists every mapping. Global
intermediate/final flows, VA, TLS, output and final-demand taxes are conserved.
Floors applied before aggregation would constitute a different calibration.

Checks include baseline recovery, independent Leontief inversion, a 10% import
tariff by the first region, Newton/continuation agreement, goods clearing,
schedule receipts and million-to-billion USD scaling, all under the library's
producer/purchaser accounting. They also check household/government budgets,
fixed foreign balances and income/expenditure GDP. These are numerical and
accounting checks on this model, **not** an independent published estimate of
tariff effects or a validation of Hicksian welfare. Full-size GE and other
providers/editions remain outside this benchmark.

## Independent two-country price and fiscal accounting

```sh
PYTHONPATH=. MPLBACKEND=Agg python tools/reference_validation/validate_trade_accounting.py \
  --output /tmp/two-country-accounting.json
```

The reference derives a scalar relative-price equation directly from a balanced
table with nonzero foreign saving, local taxes and heterogeneous intermediate
and final-use tariffs. It solves household incomes algebraically and clears one
goods market with Brent's method; it does not use puremacro's calibration,
residual evaluator or GE solver inside the oracle. Prices, factor payments,
deliveries, income, duties and rebates are then compared with the full model.
The other goods market is checked separately.

See [the accounting equations](../../docs/trade_accounting.md). This reference
covers the lump-sum, fixed-foreign-saving closure.

## Hicksian consumption welfare

```sh
PYTHONPATH=. MPLBACKEND=Agg python tools/reference_validation/validate_trade_welfare.py \
  --output /tmp/trade-welfare-comparison.json
```

The two-country benchmark separates two consumption baskets from investment.
Its scalar GE reference is followed by a numerical primal expenditure
minimization over origin goods and Leontief quantities at all four combinations
of baseline/counterfactual prices and utility. The three-block Shapley attribution
is independently enumerated over six orderings. The script also checks the
single-basket consumption calculation on the frozen OECD aggregation; it does
not reload the full source archive.

See [the welfare derivation](../../docs/trade_welfare.md).

## Hicksian tariff games and solver recovery

```sh
PYTHONPATH=. MPLBACKEND=Agg python tools/reference_validation/validate_hicksian_policy.py \
  --output /tmp/hicksian-policy-comparison.json
```

The CES extension of the scalar two-country reference independently evaluates
all 1,681 profiles on a 41-by-41 tariff grid. Its baseline expenditure cost is
obtained from primal minimization. The comparison covers unilateral optima,
final Nash deviations, fixed-action payoff matrices and an actual Newton failure
recovered by the hybrid solver. The optima reach the imposed 40% ceiling.
This is a constrained synthetic benchmark, not an empirical policy estimate or
a proof that a grid search finds every possible equilibrium.

See [policy API and failure semantics](../../docs/trade_policy.md).
