# Independent structural references

These tools compare actual external inputs with puremacro. Ordinary tests use
small frozen fixtures and do not require MATLAB or network access.

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

Download the provider's [2016–2020 regular archive](https://webfs-sti.oecd.org/files/STI-PIE/ICIO/2023/2016-2020_SML.zip)
and [schema workbook](https://webfs-sti.oecd.org/files/STI-PIE/ICIO/2023/ReadMe_ICIO_small.xlsx),
linked from the [official OECD dataset page](https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html).
The recorded download required a browser; Safari extracted the ZIP automatically.
The manifest identifies the **CSV member bytes** by SHA256; an original ZIP hash
is also recorded when the reader is given a ZIP. The workbook specifies current
million USD and basic-price flows, with taxes less subsidies supplied separately.
No foreign-exchange conversion is needed.

```sh
PYTHONPATH=. MPLBACKEND=Agg python tools/reference_validation/validate_oecd.py \
  --source /path/to/2019_SML.csv --readme /path/to/ReadMe_ICIO_small.xlsx \
  --output /tmp/oecd-comparison.json
```

`--source` also accepts the original ZIP. It validates and calibrates **all**
77 economies × 45 activities through `load_oecd_icio_granular`, then refreshes
the small empirical fixture. Omit `--source` to rerun the aggregate model checks
without downloading the archive. Such a run explicitly reports that it did not
reload the full native archive.

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

Checks include baseline recovery, independent Leontief inversion, a 10% USA
import tariff, Newton/continuation agreement, goods clearing, schedule receipts
and million-to-billion USD scaling. The compatibility default retains `schedule`
with its documented legacy final-composite-price valuation. Select the corrected
producer/purchaser model explicitly:

```sh
PYTHONPATH=. MPLBACKEND=Agg python tools/reference_validation/validate_oecd.py \
  --accounting consistent --output /tmp/oecd-consistent-comparison.json
```

This also checks household/government budgets, fixed foreign balances and
income/expenditure GDP. These are numerical and
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

See [the accounting equations](../../docs/trade_accounting.md) and the
[recorded results](../../reviews/2026-09-20-trade-accounting/REPORT.md). This
reference covers the lump-sum, fixed-foreign-saving closure. It does not validate
the quarantined Hicksian EV/theorem endpoints or historical policy wrappers.

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

See [the welfare derivation](../../docs/trade_welfare.md) and
[results](../../reviews/2026-09-20-hicksian-welfare/REPORT.md). The former
TOT/Alloc/TariffRec decomposition and theorem endpoints remain unavailable.

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

See [policy API and failure semantics](../../docs/trade_policy.md) and
[recorded results](../../reviews/2026-09-20-hicksian-policy/REPORT.md).
