# Dynamic Trade Reallocation and Household Incidence

> 🇬🇧 English · 🇪🇸 [Español](es/trade_reallocation_incidence.md)

The module `puremacro.trade.reallocation_incidence` bridges two-country, two-sector perfect-foresight trade transitions ([trade_reallocation.md](trade_reallocation.md)) with benchmark-normalized multi-group household welfare ([trade_household.md](trade_household.md)) and bundled OECD ICIO accounts. Rather than treating each region as a single representative agent, it tracks how sectoral wage shifts, capital returns net of investment outlays, tariff rebates, and consumer price changes distribute gains and losses across household groups at every date $t = 0, \dots, T-1$.

## Mathematical Structure and Accounting Identities

### OECD ICIO Aggregation (`calibrate_reallocation_from_icio`)

Given a multi-region, multi-sector table `(Z, F, VA_L, VA_K, TLS, TFD, output)` and two exhaustive partitions—`country_groups` (defaulting to `USA` vs. `ROW` for OECD ICIO) and `sector_groups` (defaulting to `Tradable` ISIC `A–C` vs. `Services`)—the bridge preserves aggregate value added while mapping empirical expenditure shares into `ReallocationEconomy`:

$$
\theta_{c,s} = \frac{\sum_{i} E_{i,s,c}}{\sum_{s'} \sum_{i} E_{i,s',c}}, \qquad \text{home\_share}_{c,s} = \frac{E_{c,s,c}}{\sum_{i} E_{i,s,c}}, \qquad \alpha^{\text{pool}} = \frac{\sum_{c,s} VA^K_{c,s}}{\sum_{c,s} (VA^L_{c,s} + VA^K_{c,s})}.
$$

Country labor supply is normalized proportionally to regional labor value added ($\sum_c L_c = 2$), while bilateral flow bases (`"final_demand"`, `"consumption"`, or `"total_absorption"`) select whether $E_{i,s,c}$ measures total final demand `C + I + Cx`, consumption `C`, or gross absorption `Z + F`.

### Household-Group Budgets and Welfare (`evaluate_transition_household_incidence`)

At each date $t \in \{0, \dots, T-1\}$ along a converged and audited `ReallocationTransitionResult`, aggregate nominal consumption spending in country $c$ satisfies:

$$
P_{c,t} C_{c,t} = \sum_{s=0}^{1} w_{t,c,s} L_{t,c,s} + \sum_{s=0}^{1} r_{t,c,s} K_{t,c,s} + T^{\text{reb}}_{t,c} - P_{c,t}\bigl(A_{c,t} - C_{c,t}\bigr).
$$

Each household group $g \in \mathcal{G}_c$ receives sector-specific labor weights $(\lambda^L_{g,0}, \lambda^L_{g,1})$, capital weight $\lambda^K_g$, tariff-rebate weight $\lambda^R_g$, and investment/adjustment-outlay share $\lambda^I_g$, each summing to $1.0$ within country $c$:

$$
m_{t,g} = \sum_{s=0}^{1} \lambda^L_{g,s} w_{t,c,s} L_{t,c,s} + \lambda^K_g \sum_{s=0}^{1} r_{t,c,s} K_{t,c,s} + \lambda^R_g T^{\text{reb}}_{t,c} - \lambda^I_g P_{c,t}\bigl(A_{c,t} - C_{c,t}\bigr).
$$

Because every factor-payment and outlay share partitions the country total, group nominal spending sums identically to national consumption ($\sum_{g \in \mathcal{G}_c} m_{t,g} = P_{c,t} C_{c,t}$) and group gross income sums identically to national gross absorption ($P_{c,t} A_{c,t}$) at every date.

Demand preferences are calibrated at initial steady-state benchmark prices $p^{\text{rel}}_{0,c,s} = 1$ via `puremacro.trade.household.calibrate_household` under `"fixed_baskets"`, `"cobb_douglas"`, `"stone_geary"`, or `"ces"`. Along the path, relative sectoral prices $p^{\text{rel}}_{t,c,s} = P_{t,c,s} / P_{0,c,s}$ and budgets $m_{t,g}$ yield:

- **Konus Cost-of-Living Index and Real Income**: $P^{\text{COL}}_{t,g} = e_g(p^{\text{rel}}_t, u_{0,g}) / m_{0,g}$ and $m^{\text{real}}_{t,g} = m_{t,g} / P^{\text{COL}}_{t,g}$.
- **Money-Metric Period Welfare**: Equivalent variation $EV_{t,g} = e_g(1, u_{t,g}) - m_{0,g}$ and compensating variation $CV_{t,g} = m_{t,g} - e_g(p^{\text{rel}}_t, u_{0,g})$.
- **Discounted Path Summaries**: Present-value $PV\_EV_g = \sum_{t=0}^{T-1} \beta^t EV_{t,g} + \frac{\beta^T}{1-\beta} EV^{\text{term}}_g$ and CRRA path consumption-equivalent percentage $\text{CEV}_g$ evaluated on $c^{\text{eq}}_{t,g} = m_{0,g} + EV_{t,g}$.
- **Cross-Group Inequality Gap**: Country-level max-minus-min spread and budget-weighted standard deviation of $EV\%_{t,g}$ along $t = 0, \dots, T-1$.

## Calibration and Incidence Workflow

The example below calibrates `(USA, ROW) x (Tradable, Services)` from the bundled 2019 OECD ICIO table, applies a 20% U.S. tariff on tradable imports, solves the dynamic transition, and evaluates household-group incidence:

```python
from puremacro.trade.reallocation import solve_reallocation_transition
from puremacro.trade.reallocation_incidence import (
    calibrate_reallocation_from_icio,
    evaluate_transition_household_incidence,
    make_sector_household_groups,
    reallocation_tariffs_from_scenarios,
)
from puremacro.trade.scenarios import TariffScenario

cal = calibrate_reallocation_from_icio()
scenario = TariffScenario(
    importer="USA",
    tariffs={("ROW", "Tradable"): 0.20},
    name="us_tradable_20pct",
)
tariffs = reallocation_tariffs_from_scenarios(scenario, cal.economy, horizon=25)
transition = solve_reallocation_transition(cal.economy, tariffs)

groups = make_sector_household_groups(
    cal.economy,
    include_capital_owners=True,
    basket_tilt=0.05,
)
incidence = evaluate_transition_household_incidence(
    transition,
    groups,
    rule="cobb_douglas",
)
print(incidence.summary(lang="en"))
```

## Custom Regions and Heterogeneous Consumer Preferences

Callers can specify arbitrary 2-region x 2-sector ICIO partitions (such as `MEX+USA` vs. `ROW`), inspect disaggregated expenditure-share inflation exposure, and combine different demand rules (`fixed_baskets`, `stone_geary`, `ces`) in one evaluation:

```python
from puremacro.trade.data import load_icio_data
from puremacro.trade.reallocation import solve_reallocation_transition
from puremacro.trade.reallocation_incidence import (
    HouseholdGroupSpec,
    calibrate_reallocation_from_icio,
    evaluate_transition_household_incidence,
    reallocation_tariffs_from_scenarios,
)
from puremacro.trade.scenarios import TariffScenario

icio = load_icio_data(return_structured=True)
mex_usa = ("MEX", "USA")
row = tuple(c for c in icio.country_codes if c not in mex_usa)

cal_na = calibrate_reallocation_from_icio(
    icio,
    country_groups={"MEX+USA": mex_usa, "ROW": row},
    sector_groups={"Goods": ("A", "B", "C"), "Services": tuple(s for s in icio.sector_codes if s not in ("A", "B", "C"))},
    expenditure_basis="consumption",
)
tariffs = reallocation_tariffs_from_scenarios(
    TariffScenario(importer="MEX+USA", tariffs={("ROW", "Goods"): 0.15}),
    cal_na.economy,
    horizon=20,
)
tr = solve_reallocation_transition(cal_na.economy, tariffs)

custom_groups = (
    HouseholdGroupSpec(
        name="NA_Goods_Workers_LowElasticity",
        country="MEX+USA",
        labor_weights=(1.0, 0.0),
        capital_weight=0.0,
        rebate_weight=0.5,
        investment_weight=0.0,
        expenditure_shares=(0.65, 0.35),
        rule="stone_geary",
        supernumerary_share=0.5,
    ),
    HouseholdGroupSpec(
        name="NA_Services_and_Capital",
        country="MEX+USA",
        labor_weights=(0.0, 1.0),
        capital_weight=1.0,
        rebate_weight=0.5,
        investment_weight=1.0,
        expenditure_shares=(0.20, 0.80),
        rule="ces",
        ces_elasticity=2.0,
    ),
    HouseholdGroupSpec(
        name="ROW_Representative",
        country="ROW",
        labor_weights=(1.0, 1.0),
        non_labor_weight=1.0,
        rule="cobb_douglas",
    ),
)
res = evaluate_transition_household_incidence(tr, custom_groups)
print(res.inequality_dataframe().head())
```

## Transparent Calibration Limits and Disclosures

Single-year input-output accounts do not identify intertemporal, substitution, or adjustment parameters. Every call to `calibrate_reallocation_from_icio` records the following modeling boundaries explicitly inside `result.audit.stylized_assumptions` and `result.diagnostic`:

| Parameter or Assumption | Source in ICIO | Identification Status and Disclosure |
|---|---|---|
| `sector_weights` | Observed aggregated demand | Identified from `final_demand`, `consumption`, or `total_absorption` shares. |
| `home_shares` | Observed domestic vs. import flows | Identified under chosen flow basis; high `ROW` home bias (`~0.98`) requires low `sourcing_cost`. |
| `labor_supply` | Aggregated `labor_va` | Relative regional labor income scale normalized to sum to `2.0`. |
| `capital_share` | Bundled `capital_va / va` | Bundled OECD 2023 file uses a convention `2/3` labor, `1/3` capital split; pooled globally to a single scalar. |
| `productivity` | Not separately identified | Single-year nominal USD flows confound prices and physical TFP; normalized to `ones((2, 2))`. |
| `beta`, `delta`, `risk_aversion` | Absent from static IO | Stylized annual macro priors (`0.96`, `0.08`, `2.0`) rather than estimated moments. |
| `trade_elasticity`, adjustment costs | Absent from static IO | Stylized priors (`trade_elasticity=4.0`, `capital_cost=4.0`, `labor_cost=5.0`, `sourcing_cost=0.01`). |
| Intermediate linkages (`Z`) and tariffs | Omitted or merged in `TLS`/`TFD` | `ReallocationEconomy` models value added under financial autarky (balanced trade) and zero baseline bilateral tariffs. |

## Conservation Audits, Diagnostics, and API Reference

Every evaluation enforces fail-closed verification before returning results:

- **Converged & Audited Transitions Only**: Passing an unconverged or corrupted `ReallocationTransitionResult` raises `TransitionError` (a `ValueError` subclass).
- **Exact Factor and Budget Partitioning**: Invalid labor-weight, capital-weight, rebate-weight, or investment-weight country sums (deviating from `1.0`), negative or non-summing expenditure shares, or nonpositive group consumption budgets raise `ValueError` / `TransitionError`.
- **Zero-Shock Identity Verification**: When `tariffs == 0` everywhere, `evaluate_transition_household_incidence` verifies $|EV_{t,g}| < 10^{-12}$, $|CV_{t,g}| < 10^{-12}$, and $|P^{\text{COL}}_{t,g} - 1| < 10^{-12}$ across every group and date.
- **Standardized Diagnostics & Bilingual Exports**: Both `ReallocationCalibrationResult` and `TransitionHouseholdIncidenceResult` attach a `SolverDiagnosticRecord` (`puremacro.solvers.diagnostics`) and expose `.summary(lang="en"|"es")`, `.to_dataframe()`, `.to_markdown()`, `.to_latex()`, `.to_typst()`, and `.to_dict()`.

Related guides: [trade_reallocation.md](trade_reallocation.md) · [trade_household.md](trade_household.md) · [trade_distributional.md](trade_distributional.md) · [trade_stability.md](trade_stability.md).
