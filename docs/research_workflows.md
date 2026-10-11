> 🇬🇧 English · 🇪🇸 [Español](es/research_workflows.md)

# Independent evidence, structural estimation and distributional trade

The [cross-country forecasting workflow](cross_country_forecasting.md) adds a
frozen three-country GDP panel, an expanding-window comparison, and portable
data provenance. It is latest-vintage forecast evidence, with no real-time or
causal claim; it is included in puremacro 4.8.0.

These research workflows connect independent validation, structural estimation
and distributional analysis. Each can run offline from an installed package with
the standard numerical dependencies.

The [real-data studies](empirical_research.md) add observed-moment SW07
estimation with explicit conditional-calibration diagnostics and an original-data
Romer–Romer (2010) baseline replication against independent software and published
rounding targets. The synthetic NK example below remains a parameter-recovery
test; the new empirical studies have their own data, assumptions and conclusions.

| Task | Entry point | Guide |
|---|---|---|
| Inspect independent numerical/data evidence | `puremacro.validation.run_research_benchmarks` | [Research benchmarks](research_benchmarks.md) |
| Fit structural models to correlated empirical moments | `puremacro.structural.fit_structural` and `lp_moment_targets` | [Empirical-to-structural bridge](structural_bridge.md) |
| Evaluate household/group incidence | `puremacro.trade.compute_distributional_welfare` | [Distributional trade](trade_distributional.md) |
| Compare SW07 moment corrections and weights | `puremacro.structural.run_sw07_estimator_experiment` | [Paired estimator controls](sw07_estimator_experiment.md) |
| Reconcile tariff prices, factor income and household rebates | `python -m puremacro.examples.distributional_trade_ge` | [Synthetic GE application with ENIGH baskets](distributional_trade_ge.md) |

## 1. Run the independent benchmarks

```python
from puremacro.validation import run_research_benchmarks

report = run_research_benchmarks()
print(report.passed)  # False: the documented RR2010 t-rounding comparison fails.
report.write("research_output/benchmarks")
```

The dossier distinguishes analytical solutions, independent expenditure
optimization, authenticated external Dynare output, official INEGI table
totals and the original Romer–Romer baseline with independent software and
published targets. It records every observed/reference value, source, units,
tolerance, limitation and negative-control result. Seven of eight cases pass;
the original-data RR2010 t statistic misses the paper's rounding interval by
about 0.000003925. That failed comparison is retained, as explained in the
[real-data guide](empirical_research.md). A pass establishes only its stated scope.

## 2. Run a complete empirical-to-structural study

```bash
python -m puremacro.examples.structural_irf_matching --output research_output/structural
```

The application independently simulates a linear three-equation New Keynesian
economy with a persistent policy disturbance and measurement errors. It
estimates output/inflation LP responses, constructs their **full joint HAC
covariance**, and fits risk aversion, Phillips-curve slope and shock persistence
by solving the model's equations through `load_mod`. Model responses and targets
use a 0.01 quarterly-rate innovation (one percentage point).

Horizons 0–4 are fitted; horizons 5–8 are descriptive out-of-fit checks using the
same sample. The application exports the simulated observations, moment vector,
joint covariance, fitted parameters, held-out comparisons, a figure and a
manifest containing the model, data hash and random seed. This is a transparent
synthetic recovery exercise, not a historical monetary-policy replication.

For an empirical study replace the simulator with your transformed data and
identified shocks, or provide your own `MomentTargets`. The same model callback
can wrap a DSGE, VFI or HANK solve. Model equations, moment definitions, shock
units, frequency and cumulative/level conventions must agree. The bridge does
not supply missing identification assumptions or HA equilibrium derivatives.

## 3. Use observed Mexican household expenditure baskets

```python
import pandas as pd
from puremacro.datasets import load_enigh2024_deciles
from puremacro.trade import prepare_household_groups, compute_distributional_welfare

survey = load_enigh2024_deciles()
sectors = list(survey.attrs["categories"])
groups = prepare_household_groups(
    survey[sectors], survey["households"],
    income_exposure=survey[["cash_wages"]],
    monetary_unit="MXN", period="quarter", provenance=survey.attrs,
)
prices = pd.Series(1.0, index=sectors)
prices["food"] = 1.025  # Explicit assumed purchaser-price shock, not an estimate.
incidence = compute_distributional_welfare(
    groups, base_prices=pd.Series(1.0, index=sectors), prices=prices,
    rule="fixed_baskets",
)
print(incidence.groups[["ev_pct", "cv_pct"]])
```

The data are actual INEGI ENIGH 2024 national household-income-decile means,
extracted from [official tables 3.2 and 4.2](https://www.inegi.org.mx/contenidos/programas/enigh/nc/2024/tabulados/enigh2024_ns_basicos_tabulados.xlsx).
Amounts are quarterly MXN **per all households in each decile**; expansion
counts sum to 38,830,230 households. The converter divides source totals by
that denominator, not by the number buying each particular category. The eight
consumption categories exclude outward transfers and nonmonetary consumption.
Cash wages exclude in-kind remuneration. Source/derived checksums, exact rows,
units and transformations travel in `survey.attrs`.

Rebuild the packaged CSV from the reviewed official workbook with
`python tools/build_enigh_distributional_data.py --workbook PATH.xlsx`.
Only this conversion step needs `openpyxl` (`puremacro[io]`). Unknown workbook
digests are refused until the source revision has been reviewed.

Run the complete six-scenario application:

```bash
python -m puremacro.examples.distributional_trade_enigh --output research_output/distributional
```

Its default **illustrative assumptions** are a 10% tariff, 25% food-basket import
exposure and full pass-through, mapped to a 2.5% purchaser-price increase. Import
shares and pass-through are not estimated from ENIGH. Cash wages are unchanged.
An externally supplied pool equal to 0.5% of baseline aggregate consumption is
either not paid, paid equally per household, or paid to the bottom four deciles.
That pool is **not inferred tariff revenue**. Fixed baskets and Cobb–Douglas
give two preference scenarios. Flags expose every policy assumption; use
`--help` for the complete list.

The outputs include group EV/CV, population-weighted monetary totals, separately
defined average percentages, price/income/transfer attribution, a figure and a
manifest. No survey confidence interval, within-decile distribution, industry
concordance, or estimated GE shock is claimed. For a solved trade model, the
distributional API can instead obtain supported purchaser-price composites
from audited GE results, with explicit household income and transfer mappings.

The distinction between consumption patterns and actual import exposure is
substantive: see [Fajgelbaum and Khandelwal](https://www.nber.org/papers/w20331)
and [Borusyak and Jaravel](https://www.nber.org/papers/w28957). These applications
do not claim to reproduce either paper.
