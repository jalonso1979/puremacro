> 🇬🇧 English · 🇪🇸 [Español](es/climate.md)

# Climate Macro (DICE)

`puremacro.climate.simulate_dice_model` is a compact forward simulator of
Nordhaus's DICE-2016R model. You supply a carbon-tax path, a climate
sensitivity and a damage coefficient; it returns the joint 2020-2165 trajectory
of output, emissions, the three carbon reservoirs, warming and damages, plus a
**social cost of carbon (SCC)** computed as DICE defines it: the present value
of the consumption lost to one extra ton of CO2 emitted today, obtained by
perturbing the emissions path and discounting with the Ramsey factor.

It is a *scenario* tool, not an optimiser: nothing is maximised, the savings
rate is fixed and the abatement rate follows DICE's first-order condition
against the tax you pass in. Within that scope the blocks are the published
DICE-2016R ones, applied on their published time basis (see "Model blocks").

```python
from puremacro.climate import simulate_dice_model

res = simulate_dice_model()   # defaults: 2020-2165, $40/tCO2 growing 2%/yr, ECS 3.1
print(res.summary())
res.plot()
```

## `simulate_dice_model()` — every argument is keyword-only

| argument | default | meaning |
|---|---|---|
| `n_periods` | `30` | number of steps; with 5-year steps 30 covers 2020-2165 |
| `time_step_years` | `5` | years per step. DICE's constants are per 5-year step; for other values the carbon-cycle and temperature operators are raised to the power `dt/5` (matrix fractional power), so a 10-year run samples the same dynamics |
| `start_year` | `2020` | calendar year of the initial state |
| `carbon_tax_initial` | `40.0` | carbon price in the start year, $/tCO2; `0.0` is the no-policy path |
| `carbon_tax_growth` | `0.02` | annual growth rate of the tax |
| `climate_sensitivity` | `3.1` | equilibrium warming per CO2 doubling, °C (DICE-2016R) |
| `damage_coef` | `0.00236` | `a2` in the quadratic damage function `D(T) = a2 T^2` |
| `discount_rate` | `0.015` | pure rate of time preference `rho` used in the SCC |
| `elasticity_marginal_utility` | `1.45` | `eta` in the Ramsey discount factor used in the SCC |
| `savings_rate` | `0.22` | fixed gross savings rate |
| `scc_horizon_years` | `300` | horizon over which marginal damages are integrated for the SCC |

Inputs are validated: a non-positive `n_periods`, `time_step_years` or
`climate_sensitivity`, or a negative carbon tax, raise `ValueError`.

## What comes back

`DICEResult` is a frozen dataclass:

| field / method | what it is |
|---|---|
| `trajectories` | `DataFrame` indexed by `year` with `output_gross`, `output_net`, `capital`, `consumption`, `population`, `emissions` (GtCO2/yr), `atmospheric_carbon`, `upper_ocean_carbon`, `deep_ocean_carbon` (GtC), `radiative_forcing` (W/m2), `temperature_anomaly`, `ocean_temperature` (°C), `climate_damages` ($T/yr), `damage_fraction`, `abatement_rate` (`mu`), `abatement_fraction`, `carbon_tax` and `social_cost_of_carbon` ($/tCO2) |
| `peak_temperature` | max of `temperature_anomaly` over the horizon |
| `scc_initial` | SCC in the start year |
| `end_century_damages` | `damage_fraction` in 2100 |
| `parameters` | the arguments the run used |
| `summary()`, `to_frame()`, `to_markdown()`, `to_latex()`, `to_typst()`, `plot()` | the standard puremacro presentation interface |

## Running a baseline and a policy scenario

```python
from puremacro.climate import simulate_dice_model

base = simulate_dice_model(
    n_periods=30, time_step_years=5, start_year=2020,
    carbon_tax_initial=40.0, carbon_tax_growth=0.02,
)
policy = simulate_dice_model(
    n_periods=30, time_step_years=5, start_year=2020,
    carbon_tax_initial=80.0, carbon_tax_growth=0.035,
)
no_policy = simulate_dice_model(carbon_tax_initial=0.0, carbon_tax_growth=0.0)

print(base.summary())
warming_saved = (base.trajectories["temperature_anomaly"]
                 - policy.trajectories["temperature_anomaly"])
print(warming_saved.loc[2100])
```

What the runs produce (default calibration, 5-year steps):

| | baseline ($40, +2%/yr) | accelerated ($80, +3.5%/yr) | no policy |
|---|---|---|---|
| abatement rate `mu` in 2020 | 0.194 | 0.300 | 0.000 |
| first year `mu` reaches 1.0 | 2130 | 2070 | never |
| emissions 2020, GtCO2 | 32.4 | 28.5 | 39.6 |
| emissions 2100, GtCO2 | 23.1 | 0.4 | 70.1 |
| `atmospheric_carbon` 2100, GtC | 1325 | 996 | 1679 |
| warming 2100, °C | 3.36 | 2.68 | 3.91 |
| `peak_temperature`, °C | 4.13 | 2.96 | 5.94 |
| `end_century_damages` | 2.66% | 1.70% | 3.62% |
| `scc_initial`, $/tCO2 | 34.7 | 33.5 | 35.2 |

The SCC is almost the same across the three runs because it is a marginal
damage evaluated on each path, not a function of the tax: the no-policy path
is slightly warmer, so its marginal ton does slightly more harm. The
no-policy peak of about 6 °C and the 2100 value of about 3.9 °C are in line
with DICE-2016R's own baseline.

## Model blocks

### Macro

```
L(t)     : logistic path from 7.79 to 11.5 billion
A(t)     : TFP growing at 0.076/5yr, declining 0.5%/yr
Y_gross  = A K^0.30 L^0.70                              $T/yr
K'       = (1 - 0.10)^dt K + dt * savings_rate * Y_net
```

### Emissions and abatement

```
sigma(t) : carbon intensity, GtCO2 per $T, declining at DICE's rate
p_back(t): backstop price, 550 $/tCO2 in 2015 declining 0.5%/yr
tau(t)   = carbon_tax_initial * (1 + carbon_tax_growth)^(t - t0)
mu       = min(1, (tau / p_back)^(1/(theta2 - 1))),   theta2 = 2.6
abatement_fraction = theta1 * mu^theta2,  theta1 = p_back * sigma / (1000 theta2)
E_ind    = sigma * (1 - mu) * Y_gross ;  E_land = 2.6 GtCO2/yr declining 11.5%/5yr
```

### Carbon cycle

Three reservoirs, initialised at `[851, 460, 1740]` GtC. DICE-2016R's transfer
coefficients `b12 = 0.12`, `b23 = 0.007` per 5-year step, with the reverse
flows pinned to the equilibrium stocks (`b21 = b12 * 588/360`,
`b32 = b23 * 360/1720`). The operator is applied so that carbon is
**conserved exactly**: with zero emissions the three stocks always sum to the
same total. Emissions are converted at 3.666 GtCO2 per GtC and injected into
the atmosphere.

### Temperature

Two layers with DICE-2016R's per-step coefficients `c1 = 0.1005`,
`c3 = 0.088`, `c4 = 0.025` (they are *not* multiplied by the step length;
for `time_step_years != 5` the whole 5-year operator is raised to the power
`dt/5`). Forcing is `3.68 log2(M_AT / 588)` plus DICE's exogenous non-CO2
path. The recursion is stable and monotone for any positive climate
sensitivity: at `climate_sensitivity=1.0` warming peaks at 1.45 °C, and at
5.0 at 5.91 °C.

### Damages

`damage_fraction = min(0.90, damage_coef * T_atm^2)` — quadratic only; no
other damage form is provided.

### Social cost of carbon

For each period `t` the emissions path is perturbed by one extra ton of CO2,
the model is re-run, and the consumption losses over the next
`scc_horizon_years` are discounted with `(1 + rho)^-(s-t) (c_t / c_s)^eta`.
The SCC therefore responds to the discount rate and to the climate
parameters:

| `discount_rate` | 0.001 | 0.015 (default) | 0.05 | 0.10 |
|---|---|---|---|---|
| `scc_initial`, $/tCO2 | 151 | 34.7 | 6.1 | 1.8 |

Raising `climate_sensitivity` from 3.1 to 5.0 raises the initial SCC from
$34.7 to $60.3 (+74%) and the peak warming from 4.13 °C to 5.91 °C.

## What is simplified relative to published DICE

- No optimisation: the savings rate is fixed and the abatement rate follows
  the supplied tax; DICE's optimal-policy runs are not reproduced.
- The population, TFP, carbon-intensity, backstop and land-emission paths are
  the closed-form approximations of DICE-2016R's exogenous processes.
- One quadratic damage function; no tipping points, no catastrophic damages.
- The SCC is a marginal-damage calculation along the simulated path with a
  fixed 300-year horizon; DICE reports it from the optimiser's shadow prices.

## Gotchas

- `time_step_years` other than 5 changes the sampling of the same
  continuous-time dynamics, not the dynamics themselves; a 10-year run reports
  a peak of 4.25 °C against the 5-year run's 4.13 °C because it samples the
  path at different dates.
- `carbon_tax_initial=0.0` still prices carbon in the SCC column: the SCC is a
  damage, not the tax.

## The rest of `puremacro.climate`

The subpackage also holds four empirical modules, added in 0.52.0 for
climate-and-fertility work, that have nothing to do with DICE. **They are not
re-exported** — `puremacro.climate.__all__` is `["DICEResult",
"simulate_dice_model"]` only, so `from puremacro.climate import
compute_annual_cdd_hdd` raises `ImportError`. Import from the submodule:

```python
from puremacro.climate.degree_days import compute_monthly_cdd_hdd, compute_annual_cdd_hdd
from puremacro.climate.annual_lp import climate_annual_lp
from puremacro.climate.mediation import climate_mediation_lp
from puremacro.climate.monthly_dl import monthly_dl, make_dl_lags
```

| function | what it does |
|---|---|
| `compute_monthly_cdd_hdd(df, *, temp_col="temp_c", threshold=18.0)` | adds `cdd = max(T-18, 0)` and `hdd = max(18-T, 0)` columns |
| `compute_annual_cdd_hdd(df, *, temp_col, threshold, region_col="region", year_col="year", month_col="month")` | sums those to `annual_cdd` / `annual_hdd` per region-year |
| `climate_annual_lp(panel, *, response, cdd_col="annual_cdd", hdd_col="annual_hdd", horizons=range(0,11), n_lags=2, controls=(), region_col="region", year_col="year", alpha=0.10)` | returns `{"cdd": ..., "hdd": ...}`, each a `panel_lp_dk` frame with `[h, beta, se, t, lo, hi]` |
| `climate_mediation_lp(panel, *, mediator_col, response, ..., n_bins=5)` | baseline vs top-quintile-interacted LP plus `mediation_share_cdd` / `mediation_share_hdd` arrays |
| `monthly_dl(df, *, shock_cols=("cdd","hdd"), response_col="log_births", n_lags=12, ...)` | distributed-lag OLS; HC1 for one region, cluster-by-region in panel mode |

`climate_annual_lp` runs the CDD regression with HDD as a control and vice
versa, so each reported IRF is the *partial* response — heating and cooling
degree days are strongly negatively correlated within a region-year and an
unconditional CDD response is largely the absence of winter.

One dead parameter here too: `climate_mediation_lp(..., top_quintile_only=True)`
is accepted and never read. The top quintile is always the one interacted.
