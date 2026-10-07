> 🇬🇧 English · 🇪🇸 [Español](es/trade_household.md)

# Household preferences with exact expenditure-function welfare

`puremacro.trade.household` is a solver-independent household demand layer:
four preference rules over sector composites, a benchmark-normalized
expenditure function, Marshallian and Hicksian demand, analytic Slutsky
matrices and exact money-metric EV/CV. It was ported from the IO research
module `headlinePaper/preferences_2026-09-22/household.py` (2026-09-22) and
plugs into puremacro's trade calibrations and equilibria through two bridges.

```python
import numpy as np
from puremacro.trade.household import calibrate_household, fit_supernumerary_share

x0 = np.array([[25., 10.], [45., 60.], [30., 30.], [0., 0.]])   # benchmark spending, sectors x regions
eta = np.array([[.6, .5], [.9, 1.1], [1.3, 1.2], [np.nan, np.nan]])  # Engel targets (NaN on absent sectors)

prefs = calibrate_household(x0, "stone_geary", expenditure_elasticities=eta,
                            supernumerary_share=[.4, .5],
                            sector_codes=["AGR", "MAN", "SRV", "NONE"], country_codes=["USA", "ROW"])
print(prefs.summary())
print(prefs.calibration.to_dataframe())      # adding-up repair, admissible bounds, subsistence shares

prices = np.array([[1.3, .8], [.9, 1.25], [1.1, 1.05], [2., 3.]])
budget = np.array([105., 110.])
state = prefs.evaluate(prices, budget, validate=True)
welfare = prefs.welfare(prices, budget, validate=True)     # EV/CV against the calibration benchmark
print(welfare.to_dataframe())
S = prefs.slutsky(prices, budget)                           # (4, 4, 2) substitution matrices
```

## What it computes

All arrays have shape `(sector, region)`, except budgets, utility and welfare,
which have shape `(region,)`. A unit of a sector composite is the quantity that
costs one monetary unit at its benchmark purchaser price, so the benchmark
spending matrix `x0` is also the benchmark quantity matrix and benchmark prices
and utility are one. Regional budgets are `m0 = x0.sum(0)`, shares `w = x0 / m0`.
Origin substitution, taxes and the general-equilibrium budget are external.

| rule | demand | expenditure function |
|---|---|---|
| `fixed_baskets` | `q = x0 m / (m0 sum_i w_i p_i)` | `e(p, u) = m0 u sum_i w_i p_i` |
| `cobb_douglas` | `q_i = w_i m / p_i` | `e(p, u) = m0 u prod_i p_i^w_i` |
| `stone_geary` | `q_i = gamma_i + beta_i (m - sum_j p_j gamma_j) / p_i` | `e(p, u) = sum_i p_i gamma_i + S0 P(p) u`, `P(p) = prod_i p_i^beta_i`, `S0 = lambda m0` |
| `ces` | `q_i = w_i m p_i^-sigma P^(sigma-1)` | `e(p, u) = m0 P(p) u`, `P(p) = (sum_i w_i p_i^(1-sigma))^(1/(1-sigma))` |

For Stone-Geary the surplus is `S = m - sum_j p_j gamma_j` and the
benchmark-normalized utility is `u = S / (S0 P(p))`. The full cost-of-living
index at benchmark utility is `e(p, 1) / m0`; `P` alone prices discretionary
spending and is not that index. `ces` with `sigma = 0` reproduces fixed baskets
and with `sigma = 1` Cobb-Douglas (an `expm1`/`log1p` form keeps the index
stable near one). Hicksian demand is `h(p, u) = grad_p e(p, u)`; the analytic
substitution matrices are

```
LES / Cobb-Douglas:  S_ij = S * (beta_i beta_j / (p_i p_j) - delta_ij beta_i / p_i^2)
CES:                 S_ij = sigma * (h_i h_j / m - delta_ij h_i / p_i)
fixed baskets:       S = 0
```

They are symmetric, negative semidefinite and satisfy `S p = 0`.

### Welfare

With 0 the base state and 1 the comparison state,

```
EV = e(p0, u1) - e(p0, u0)
CV = e(p1, u1) - e(p1, u0)
```

Both are positive for welfare gains; `cv` is the amount removable at new prices,
not the compensation required. Against the calibration benchmark this reduces
to `EV = S0 (u1 - 1)` for every rule (including fixed baskets) and
`CV = m1 - e(p1, 1)`; for LES `CV = P(p1) EV`. `HouseholdWelfareResult` reports
`ev`, `cv`, `ev_pct_consumption`, `cv_pct_consumption` (both divided by the base
household budget `e(p0, u0)` so preference systems share one denominator),
`utility`, `base_utility`, `cost_index` and `price_index` per region.
`compute_household_welfare(prefs, p1, m1, base_prices=p0, base_budget=m0)` takes
an arbitrary base state; `prefs.welfare(p, m)` uses the benchmark.

### Calibration from Engel targets

At the benchmark the LES expenditure elasticity is `eta_i = beta_i / w_i`, so
`beta = w eta`. Engel adding-up `sum_i w_i eta_i = 1` is imposed by dividing each
region's targets by their expenditure-weighted mean; the original means, the
maximum adjustment and the repaired targets are recorded in
`prefs.calibration` (a `HouseholdCalibrationResult`; `repair_adding_up=False`
rejects inconsistent targets). Subsistence is `gamma = x0 (1 - lambda eta)` with
`lambda = S0 / m0` the supernumerary share, which must satisfy
`0 < lambda <= 1 / max_i eta_i` in every region for `gamma >= 0`. `lambda = 1`
gives Cobb-Douglas. Unobserved sectors stay exact structural zeros; omitted
targets explicitly mean unit elasticities.

`lambda` is not identified by IO data. `supernumerary_share=None` (the default)
uses the assumed value 0.5 and records
`metadata["supernumerary_share_default_is_assumption"] = True` (a supplied
share records `False`); the calibration audit says so verbatim: "Stone-Geary surplus
shares are supplied assumptions or external estimates, not identified by IO
accounts and Engel targets". `fit_supernumerary_share` projects external
compensated own-price elasticity targets onto the LES benchmark identity
`xi_ii = -lambda eta_i (1 - beta_i)` by regional weighted least squares, clipped
to the admissible interval, and returns a `SupernumeraryFitResult` with the
unconstrained fit, bounds, clipping flags, per-sector residuals and weighted
RMSE. Zero-weight categories have no influence on the fit (their target values
can be placeholders). A region with no weighted price response raises.

```python
own = -prefs.supernumerary_share * prefs.eta * (1 - prefs.beta)   # implied targets
fit = fit_supernumerary_share(x0, eta, own)
print(fit.summary(), fit.to_dataframe())
```

### Domain and complex step

The evaluation kernels retain complex inputs, so `prefs.evaluate(p + 1e-25j * dp, m)`
gives exact directional derivatives, and never clip. Pass `validate=True` (or
call `prefs.validate_domain(p, m)`) at accepted real states: nonpositive
prices, nonfinite inputs and exhausted surplus raise `HouseholdDomainError`,
a `ValueError` subclass. Strictly positive surplus is required for an interior
demand derivative and positive normalized utility.

## Relation to the flexible module

`puremacro.trade.flexible` also carries a Stone-Geary system
(`FlexiblePreferenceConfig.mu_s`). With unit Engel targets and
`lambda = 1 - mu_s` the two give the same demand levels at benchmark income
(the test suite checks `compute_stone_geary_final_demand` against
`prefs.demand` to 1e-12 on the synthetic 2-country, 2-sector calibration). A
per-sector-and-region `mu_s = 1 - lambda*eta` (using the repaired `eta`) does
the same with nonunit targets and a regional `lambda` (also checked to
1e-12); the adding-up repair and the admissibility check
`lambda <= 1/max(eta)` must be performed here first. The flexible
general-equilibrium solver uses a fixed-proportion supernumerary LES that nests
the legacy basket (a uniform `mu_s` is neutral there); the demand-level parity
holds for `compute_stone_geary_final_demand` only.

Only the demand levels coincide. The flexible module scales subsistence with
income (`tanh` smoothing), so its demand has no expenditure function: on the
synthetic calibration its finite-difference Slutsky matrix is asymmetric by
0.06 even at benchmark income, 0.15 at income ratio 0.8 and 0.02 at ratio
1.2, against about 2e-8 for the exact LES here (the test asserts the flexible
asymmetry above 1e-3 and the exact one below 1e-7 at all three ratios), and
its demand levels depart from the exact LES away from benchmark income.
`FlexibleTradeEquilibriumResult.welfare_decomposition` therefore remains a
proxy, as its docstring states; this module gives the exact
expenditure-function alternative.

## Bridges to trade calibrations and equilibria

```python
import numpy as np
from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium, compute_hicksian_welfare
from puremacro.trade.household import (calibrate_household, household_expenditure_from_calibration,
                                       household_prices_from_result, compute_household_welfare_from_results)

# Analytic 2-country x 2-sector table with three final uses (C, I, X); rows: sellers, taxes, labour, capital.
Z = np.array([[10., 15., 5., 5.], [15., 20., 10., 10.], [5., 5., 12., 18.], [10., 10., 18., 22.]])
y = np.array([100., 150., 120., 180.])
va = y - Z.sum(0) - .05 * y
data = np.zeros((7, 10))
data[:4, :4] = Z
data[4, :4], data[5, :4], data[6, :4] = .05 * y, 2 / 3 * va, va / 3
fd = y - Z.sum(1)
data[:2, 4:] = fd[:2, None] * np.array([.50, .25, .05, .10, .08, .02])
data[2:4, 4:] = fd[2:, None] * np.array([.10, .08, .02, .50, .25, .05])
data[4, 4:] = .02 * data[:4, 4:].sum(0)
calib = calibrate_trade_model(data, ns=2, nc=2, nfd=3, country_codes=["HOME", "ROW"], sector_codes=["GOODS", "SERV"])

x0 = household_expenditure_from_calibration(calib, category=0)       # (2, 2) benchmark purchaser spending
rates = np.array([.15, 0.])                                           # HOME taxes all its imports at 15 percent
base = solve_trade_equilibrium(calib, accounting="consistent", tol=1e-10)
cf = solve_trade_equilibrium(calib, tau=rates, tau_fd=rates, accounting="consistent", tol=1e-10)
prices, budget = household_prices_from_result(cf, calib)              # (2, 2) composite prices, (2,) budgets
for rule, kw in (("fixed_baskets", {}), ("cobb_douglas", {}), ("stone_geary", {"supernumerary_share": .6}),
                 ("ces", {"ces_elasticity": .5})):
    prefs = calibrate_household(x0, rule, sector_codes=calib.sector_codes, country_codes=calib.country_codes, **kw)
    welfare = compute_household_welfare_from_results(prefs, calib, cf, base_result=base)
    print(rule, np.round(welfare.ev_pct_consumption, 4))
hicks = compute_hicksian_welfare(calib, cf, base_result=base, target_country="HOME")
print("Leontief basket", round(hicks.ev_pct_consumption, 4))          # equals the fixed_baskets household
```

`household_expenditure_from_calibration(calib, category)` returns
`x0[s, c] = A[s, c] theta[k, c] (l + k + T)[c]`, where `A` is the share of
sector `s` in the category's Leontief origin basket (`sum_i afd[i, s, k, c]`).
Purchaser spending includes the final-use tax, a uniform share of the category
in both accounting modes, so basic-price and purchaser sector shares coincide.
Category 1 (investment) is rejected because its spending subtracts foreign
saving; negative cells raise and are named. On the bundled 77x11 OECD table
category 0 is clean (all 847 sector shares positive) while category 1 carries
negative cells.

`household_prices_from_result(result, calib, category=0, tau_fd=None)` returns
the tariff-inclusive Leontief composite prices relative to the benchmark,
`P[s, c] = sum_i afd[i, s, k, c] p_i tf[i, s, k, c] / sum_i afd[i, s, k, c]`, and
the category's purchaser budget `theta[k] (w l + r k + T)`.

* `accounting="consistent"` results are re-audited exactly as
  `compute_hicksian_welfare` does (recorded tariff schedules, residuals,
  fields) and valued from `_accounting.evaluate`: `tf` is the recorded
  `final_tariff_multipliers`, the budget is the block's `expenditure`. `tau_fd`
  is optional here; when given (the solve's national rates or a full
  multiplier array) it is resolved with the solver's conventions and
  cross-checked against the recorded schedule, and a disagreement raises.
* Legacy results record no tariff schedules, so their residuals cannot be
  re-evaluated: they are accepted only if the recorded residuals meet the
  solve's own tolerance (`metadata["tol"]`, 1e-8 when absent). They are
  valued at the state prices in `x_sol`, from which the legacy evaluator built
  `p_fd` and the final-demand flows; the recorded `p_sol` is the zero-profit
  price and differs from them by up to the solve residual (1.8e-7 on the
  synthetic calibration at the default tolerance 2.5e-3). Pass the `tau_fd`
  used in the solve (`None` means free trade); it is audited against the
  stored `p_fd` (`ppfd_k = sum_i afd_ik p_i tf_ik`) and a mismatch raises.
  That aggregate identifies one foreign multiplier per destination, so
  national rates are verified. A schedule that varies across foreign origins
  or sectors cannot be verified from `p_fd`: on the synthetic calibration a
  different schedule with the same aggregates shifts the importer's EV by 12
  percent for Cobb-Douglas and 22 percent for CES with elasticity 3 (fixed
  baskets are unaffected). Such a schedule therefore raises unless
  `trust_tau_fd_detail=True`, which accepts the detail with only its
  aggregates verified; prefer `accounting="consistent"`, which records the
  schedule. Legacy budgets are available for lump-sum closures only.
* Flexible-model results are not supported by the bridge: compute Armington
  composite prices with `puremacro.trade.flexible` and call
  `compute_household_welfare` directly.

`tol` (default 1e-8) compares recorded fields with their recomputed values
(the consistent audit's field checks; the legacy `p_fd`, `c_fd`, `w_sol`,
`r_sol` and `T_sol`), a supplied `tau_fd` with a recorded consistent
schedule, and the spread of a legacy schedule across foreign origins.
Equilibrium residuals are checked at the solver's recorded tolerance instead,
and domestic final-demand multipliers must equal one to 1e-14. Errors name
the function called and, in `compute_household_welfare_from_results`, the
failing state (`eq_result` or `base_result`); the welfare metadata key
`tariff_schedule_verification` records how each state's schedule was
verified. Preference labels must equal the calibration's `country_codes` and
`sector_codes` unless they are the generated defaults (`C00`, `S00`, ...),
and the welfare result carries the calibration's country labels.

With `rule="fixed_baskets"` and a single consumption category the bridge
reproduces `compute_hicksian_welfare` (Leontief basket) exactly; the test suite
checks EV, CV and percentages to 1e-10 relative for every country on the
synthetic calibration with a 15 percent tariff (categories 0 and 2) and on a
non-square 3-country, 2-sector table, and the legacy path against the
closed-form Leontief identity `EV = m1 ppfd0 / ppfd1 - m0`. Non-homothetic
and CES households give different numbers by construction. Bridged EV/CV
inherit the equilibrium's numeraire and foreign-saving conventions: with
nonzero foreign saving the consistent model's convention depends on the
country order, and so do these numbers, as for `compute_hicksian_welfare`
(on the 3-country table, reversing the order moves the first country's
Stone-Geary EV from 1.26 to 1.19 percent of its budget).

With the bundled OECD table, final-use category 0 is HFCE + NPISH + GGFC, so a
"household" calibrated from it values aggregate consumption including
government final consumption; the result metadata records this. A household-only
split needs a native source with separate final uses (see
[Hicksian consumption welfare](trade_welfare.md)).

## What is and is not validated

Checked in `tests/test_trade_household.py`:

* the twelve behavioral and duality tests of the IO module on its analytic
  4x2 fixture (benchmark reproduction for every rule and CES elasticity in
  {0, 0.4, 1, 2}, budget adding-up, homogeneity, Engel derivatives at 2e-10,
  `lambda = 1` equals Cobb-Douglas, expenditure duality, Shephard's lemma,
  Slutsky symmetry and negative semidefiniteness by central differences at
  7e-8, complex-step derivatives at 2e-9, domain errors, admissibility
  boundary, CES limits, exact recovery of a generating `lambda` to 2e-16,
  binding bounds and zero-weight influence);
* analytic Slutsky matrices against central differences at 5e-8 in every
  region, exact symmetry, `S p = 0` and negative semidefiniteness;
* a primal SLSQP expenditure-minimization oracle for Stone-Geary, Cobb-Douglas
  and CES (0.7, 1.8): minimized expenditure within 1e-6 relative of `e(p, u)`
  and minimizer within 2e-4 of the Hicksian demand;
* general-base welfare: identity gives exactly zero, reversed comparisons
  swap EV and CV, benchmark base reproduces the closed forms; currency scaling
  changes levels and preserves percentages, utility and indices;
* the calibration bridge on the synthetic and bundled 77x11 tables, the
  flexible parity at benchmark income and its divergence away from it, and
  the integrability contrast at income ratios 0.8, 1 and 1.2;
* the consistent and legacy result bridges on the square synthetic table and
  a non-square 3-country, 2-sector table: fixed-basket EV/CV against
  `compute_hicksian_welfare` for every country (categories 0 and 2), an
  independent certificate (sector purchaser spending read from the model's
  final-demand flows equals `P_s q_s` to 1e-12), the legacy Leontief identity
  to 1e-11, legacy solves at the solver's default tolerance, the residual
  gate, rejection of tampered fields and wrong schedules, the explicit trust
  required for sector-specific legacy schedules, label checks and error
  messages;
* in-process parity with the IO implementation on random inputs (calibration
  arrays, evaluation, Hicksian demand, expenditure, welfare, calibration and
  fit audits): the observed maximum difference is 0.0 (bit-identical), skipped
  when the research volume is not mounted.

Not validated here, quoting the IO documentation: the calibration "is a
disclosed transfer and reconciliation, not native microdata estimation"; "The
inherited targets generally cannot be matched exactly by LES, and a calibrated
parameter is not thereby empirically identified for 2019"; LES "substitution
restrictions are therefore substantive: matching income elasticities and
projecting compensated own-price targets does not reproduce a general CDE
substitution matrix"; "These remain static policy comparisons, not lifetime
welfare". No GTAP CDE targets are bundled (the data are licensed), and the IO
general-equilibrium embedding (`PreferenceModel`, `PreferenceInverse`,
tariff continuation, local stability) is not part of this port: use the
bridges with puremacro's own solvers.

Related documentation: [consistent trade accounting](trade_accounting.md),
[Hicksian consumption welfare](trade_welfare.md), [Hicksian tariff policy](trade_policy.md),
[quantitative spatial and trade GE](spatial_and_trade_ge.md).
