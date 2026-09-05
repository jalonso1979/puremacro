> 🇬🇧 English · 🇪🇸 [Español](es/climate.md)

# Climate Macro (DICE)

`puremacro.climate` carries a compact Nordhaus DICE simulator whose job is to
price a *scenario*, not to find one. You hand it a carbon-tax path, a climate
sensitivity and a damage coefficient; it returns a 2020-2165 joint trajectory of
output, emissions, atmospheric carbon, warming and damages, so two policies can
be compared on one internally consistent accounting frame without a GAMS
licence, a solver, or a C toolchain.

```python
from puremacro.climate import simulate_dice_model, DICEResult

res = simulate_dice_model()          # defaults: 2020-2165, $40/tCO2 tax, +2%/yr
print(res.summary())
```

Two things to keep in mind before anything else, because the module's own
docstring oversells both:

- **It is a forward simulation, not an optimisation.** Nothing is maximised.
  There is no welfare functional, no Euler equation, no shadow price. The
  savings rate is fixed at 0.22 and the abatement rate is read off a static
  first-order condition against the tax you supplied.
- **The `social_cost_of_carbon` column is not a social cost of carbon.** It is
  the carbon tax you passed in, marked up by twice the current damage share.
  Over the default run that markup ranges from 1.0062x to 1.0138x, so the column
  is your own tax path to within 1.4%. See
  [the SCC section](#the-scc-column-is-the-tax-you-passed-in).

Notebook 32 (`notebooks/32_climate_macro_dice_es.py`) and
`puremacro/examples/climate_dice_simulation.py` run the baseline-vs-policy
comparison end to end; `puremacro/examples/climate_sovereign_debt_risk.py`
feeds three DICE runs into a debt-sustainability block.

## `simulate_dice_model()` — every argument is keyword-only

| parameter | default | what it does |
|---|---|---|
| `n_periods` | `30` | number of steps. With the default step, 30 steps span 2020-2165. |
| `time_step_years` | `5` | years per step. **The carbon cycle does not rescale with it** — see [gotchas](#gotchas-that-will-bite). |
| `start_year` | `2020` | first row of the index; the model's initial state is dated here. |
| `carbon_tax_initial` | `40.0` | carbon price in year 0, $/tCO2. `0.0` gives a genuine no-policy run. |
| `carbon_tax_growth` | `0.02` | **annual** growth of the tax, compounded over calendar years, not steps. |
| `climate_sensitivity` | `3.0` | equilibrium warming per CO2 doubling, °C. Enters only as `lambda = eta / S`. |
| `damage_coef` | `0.00236` | `a` in `D(T) = a*T^2`. |
| `discount_rate` | `0.015` | **has no effect on any output.** The name appears in the signature and the docstring and nowhere in the body. |

`discount_rate` is dead: `simulate_dice_model(discount_rate=0.015)` and
`simulate_dice_model(discount_rate=0.30)` return byte-identical trajectories.
That is a direct consequence of there being no intertemporal problem to
discount — see [what is simplified](#what-is-simplified-relative-to-published-dice).

## What comes back

`DICEResult` is a frozen dataclass with four fields. `trajectories` is indexed
by integer `year`.

| column | unit | timing |
|---|---|---|
| `output_gross` | $T/yr, gross world product before damages | flow over the step |
| `output_net` | $T/yr, after damages **and** abatement spending | flow |
| `capital` | $T | **start**-of-period stock |
| `consumption` | $T/yr, `(1 - 0.22) * output_net` | flow |
| `emissions` | GtCO2/yr, industrial **plus** land use | flow |
| `atmospheric_carbon` | GtC in reservoir 1 | **start**-of-period stock |
| `temperature_anomaly` | °C above pre-industrial | **start**-of-period stock |
| `climate_damages` | $T, `output_gross * damage_fraction` | flow |
| `damage_fraction` | share of gross output | — |
| `abatement_fraction` | share of gross output spent abating | — |
| `social_cost_of_carbon` | $/tCO2 — but read the caveat above | — |

The three stocks are recorded *before* the period's state update, so the
`start_year` row is the initial condition verbatim: `capital` 223.0,
`atmospheric_carbon` 851.0, `temperature_anomaly` 1.15.

| scalar field | meaning |
|---|---|
| `scc_initial` | the `social_cost_of_carbon` value at `start_year` |
| `peak_temperature` | max of `temperature_anomaly` over the whole horizon |
| `end_century_damages` | `damage_fraction` at the index year **nearest 2100** |

`end_century_damages` picks the nearest available year, so a run that ends
before 2100 returns the last year's damages under a field named for 2100:
`simulate_dice_model(n_periods=10)` stops at 2065 and reports 2065's 0.520%
as `end_century_damages`. Check `trajectories.index[-1] >= 2100` before quoting it.

`C + I = output_net` holds exactly, by construction: consumption is `0.78 *
output_net` and investment the remaining `0.22`, with abatement already netted
out of `output_net`. Abatement is a pure resource cost, which is the correct
DICE treatment.

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

print(base.summary())
print(policy.summary())

warming_saved = (base.trajectories["temperature_anomaly"]
                 - policy.trajectories["temperature_anomaly"])
```

What the two runs actually produce:

| | baseline ($40, +2%/yr) | accelerated ($80, +3.5%/yr) | no policy (`carbon_tax_initial=0.0`) |
|---|---|---|---|
| abatement rate `mu` in 2020 | 0.194 | 0.300 | 0.000 |
| first year `mu` hits 1.0 | 2130 | 2070 | never |
| emissions 2020, GtCO2 | 33.3 | 29.3 | 40.7 |
| emissions 2100, GtCO2 | 8.6 | 0.5 | 25.1 |
| `atmospheric_carbon` 2100, GtC | 874.2 | 754.9 | 1003.6 |
| `peak_temperature`, °C | 1.71 | 1.46 | 2.71 |
| `end_century_damages` | 0.628% | 0.264% | 1.071% |
| `scc_initial`, $/tCO2 | 40.25 | 80.50 | 0.00 |

There is no `policy=None` switch. **The no-policy counterfactual is
`carbon_tax_initial=0.0, carbon_tax_growth=0.0`**, which sends `mu` to exactly
zero for the whole run — and also sends the `social_cost_of_carbon` column to
zero, which is the clearest single demonstration that the column is a tax and
not a marginal damage.

## The model blocks

Five blocks, evaluated in this order inside each step. `t` below is elapsed
**calendar years**, `dt` is `time_step_years`.

### Macro

```
L(t) = 11.5 - (11.5 - 7.79) * exp(-0.02*t)          billion people
g_A(t) = 0.015 * exp(-0.005*t)
A(t) = 5.115 * exp(g_A(t) * t)
Y_gross = A * K^0.30 * L^0.70                        $T/yr
K' = (1 - 0.10*dt) * K + 0.22 * Y_net
```

### Emissions and abatement

```
sigma(t) = 0.35 * exp(-0.015*t)                      GtCO2 per $T of output
p_back(t) = 550 * exp(-0.005*t)                      $/tCO2
tau(t) = carbon_tax_initial * (1+carbon_tax_growth)^t
mu = min(1, (tau/p_back)^(1/(theta2-1))),  theta2 = 2.6
abatement_fraction = (p_back * sigma / 1000) * mu^theta2 / theta2
E_ind = sigma * (1-mu) * Y_gross
E_land = 2.6 * exp(-0.02*t)                          GtCO2/yr
```

The `mu` expression is DICE's abatement first-order condition and
`abatement_fraction` is DICE's `theta1 * mu^theta2` with `theta1 =
p_back*sigma/(1000*theta2)`, so this block is faithful. The `/1000` is the unit
conversion that makes `sigma` read as tCO2 per $1000 of output; at 2020 values,
abating every ton at the backstop price would cost `550*0.35/1000 = 19.25%` of
gross output, of which the convex cost function charges `1/theta2` — 7.4% — at
`mu = 1`.

### Carbon cycle

Three reservoirs (atmosphere, upper ocean, deep ocean), initialised at
`[851, 460, 1740]` GtC, stepped by a fixed 3x3 matrix, with emissions converted
at **3.666 GtCO2 per GtC** and added to the atmosphere over the whole step:

```
Phi = [[0.88, 0.12, 0.000],
       [0.05, 0.94, 0.010],
       [0.00, 0.002, 0.998]]
M' = Phi @ M ;  M'[0] += (E_total / 3.666) * dt
```

Read [the carbon-cycle caveat](#the-carbon-cycle-is-not-mass-conserving-as-applied)
before drawing a concentration path from this.

### Temperature

Two layers, forward Euler, with forcing evaluated at the **start**-of-period
atmospheric stock:

```
F = 3.68 * log2(M_AT / 588)                          W/m2
lambda = 3.68 / climate_sensitivity
T_atm'   = T_atm   + 0.1005*dt * (F - lambda*T_atm - 0.088*(T_atm - T_ocean))
T_ocean' = T_ocean + 0.025 * (T_atm - T_ocean)
```

Initial state is `T_atm = 1.15`, `T_ocean = 0.50` °C. This is the whole story
behind the low peak warming: because the carbon cycle holds `M_AT` between
837 and 887 GtC over the whole default run, the equilibrium warming the forcing
implies is `F/lambda` = 1.53-1.78 °C, and the two-layer system simply converges
there.

### Damages

```
damage_fraction = min(0.90, damage_coef * T_atm^2)
Y_net = Y_gross * (1 - damage_fraction - abatement_fraction)
```

Pure quadratic; the module docstring's promise of a "quadratic / exponential"
damage function is not implemented — there is no exponential branch and no
switch to select one. The 0.90 cap is never approached at default parameters
(max `damage_fraction` over the baseline run is 0.69%).

## Where the parameters come from

The hard-coded constants are the **DICE-2016R2** calibration, with DICE's 2015
initial state re-dated to 2020 rather than re-estimated. These are lifted
verbatim:

| constant | value | DICE symbol |
|---|---|---|
| initial capital | 223 $T | `k0` |
| reservoir stocks | 851 / 460 / 1740 GtC | `mat0` / `mu0` / `ml0` |
| pre-industrial carbon | 588 GtC | `mateq` |
| forcing per doubling | 3.68 W/m2 | `fco22x` |
| heat-uptake / transfer / deep-ocean coefficients | 0.1005 / 0.088 / 0.025 | `c1` / `c3` / `c4` |
| capital elasticity | 0.30 | `gama` |
| annual depreciation | 0.10 | `dk` |
| initial TFP | 5.115 | `a0` |
| asymptotic population | 11.5 bn | `popasym` |
| initial carbon intensity | 0.35 | `sigma0` |
| backstop price | 550 $/tCO2 | `pback` |
| abatement cost exponent | 2.6 | `expcost2` |
| damage coefficient | 0.00236 | `a2` (with `a3 = 2`) |
| initial land-use emissions | 2.6 GtCO2/yr | `eland0` |

Four numbers are not DICE's: initial population `7.79` bn and initial warming
`T_atm = 1.15` °C, which are 2020 figures where the rest of the state is dated
2015, and initial ocean temperature `0.50` °C and `climate_sensitivity = 3.0`,
which are round numbers of their own.

`damage_coef = 0.00236` is worth internalising because it sets the scale of
everything downstream: it implies a loss of **2.12% of gross output at 3 °C**.
That is the low end of the published range and the single most contested number
in the DICE literature. The sovereign-debt example varies it from 0.00236 to
0.0035 for exactly this reason.

The module docstring also cites Golosov, Hassler, Krusell & Tsyvinski (2014).
**Nothing from that paper is implemented.** GHKT's contribution is a closed-form
SCC proportional to GDP, derived from log utility, full depreciation and
exponential damages; none of those three assumptions appears anywhere in
`dice.py`, and no closed form is evaluated.

## The SCC column is the tax you passed in

```python
scc_t = tax_t * (1.0 + damage_fraction * 2.0)
```

That is the entire computation. It is not a marginal damage, not a shadow price
on the carbon constraint, and not a discounted integral of future losses — a
real SCC requires perturbing the emissions path, propagating the perturbation
through the carbon cycle and the temperature layers, and discounting the
resulting damage stream, which is why `discount_rate` would matter in a model
that did it.

Three symptoms that make this checkable rather than a matter of opinion:

- `scc_initial` is `carbon_tax_initial` to within 0.7%: `40.0 -> 40.25`,
  `80.0 -> 80.50`, `0.0 -> 0.00`.
- Raising `climate_sensitivity` from 3.0 to 5.0 raises peak warming from 1.71 °C
  to 2.75 °C and moves the SCC path by at most 2.2%, because climate sensitivity
  reaches the column only through `damage_fraction` inside that markup.
- Setting the tax to zero sets the "social cost of carbon" to zero, in a world
  that then warms to 2.71 °C.

Use the column as a record of the carbon price you imposed. If you want a
number to call an SCC, you have to build it yourself from `climate_damages` and
a discount factor of your choosing.

## What is simplified relative to published DICE

The blocks above are recognisable DICE; what surrounds them is not. In rough
order of how much it changes an answer:

**No optimisation, so no optimal path and no shadow prices.** Published DICE
maximises discounted welfare over `mu_t` and `s_t`. Here `s = 0.22` always and
`mu` follows the static FOC against an exogenous tax. Every "optimal" quantity
DICE reports — the optimal carbon price, the optimal control rate, the SCC — is
absent by construction.

**Capital falls 78% in the first thirty years, and it is an artefact.**
Depreciation is applied over the step (`0.10 * dt = 0.50` per 5-year step) while
investment is added as a single year's flow (`0.22 * Y_net`, with no `* dt`).
The two are on different time bases, so the stock collapses from 223 $T in 2020
to 49.4 $T by 2050 and the capital-output ratio falls from 2.05 to 0.42 — against
an empirical world figure around 3. (The step depreciation is also linear where
it should be geometric — `1-(1-0.10)^5 = 0.410`, not `0.500` — but that is much
the smaller of the two errors.) Gross output actually *declines* from 109 to
101.6 $T between 2020 and 2030 before the TFP and population trends overtake the
falling stock. **Read output and consumption levels as an index, never as
dollars.** Emissions, concentrations and temperature are affected through
`Y_gross`, but they are dominated by `sigma` and the carbon cycle.

**The TFP path is a closed form, not DICE's recursion, and it eventually turns
down.** DICE iterates `A_{t+1} = A_t / (1 - g_A(t))` with `g_A` decaying.
Here the *current* growth rate is applied over the *whole* elapsed span,
`A(t) = 5.115 * exp(g_A(t) * t)`, whose exponent `0.015*t*exp(-0.005*t)` peaks at
`t = 200` years. So TFP rises to 15.4 in 2220, then falls back towards 5.115.
Inside the default 145-year span you never see it; run
`simulate_dice_model(n_periods=80)` and gross output peaks in 2230 and declines
for the remaining 185 years of the run.

**Decarbonisation never slows.** `sigma` decays at a flat 1.5% a year forever,
so carbon intensity in 2165 is `exp(-0.015*145)` = 11% of its 2020 value. DICE
decays the *decay rate*, which is the difference between assuming free
technological progress forever and assuming it runs out.

**No exogenous forcing from non-CO2 gases.** DICE adds a `forcoth` term for
other greenhouse gases; there is none here, so the forcing is CO2-only.

**Land-use emissions decay at 2%/yr** rather than DICE's per-period rate, and
asymptote to zero without ever reaching it — the residual land-use flow is why
the accelerated scenario still shows 0.5 GtCO2 in 2100 despite `mu = 1` since
2070.

**Warming under a modest tax is far below published DICE, and it is the carbon
cycle doing it, not the abatement.** A $40 tax growing 2%/yr caps peak warming at
1.71 °C here. The no-policy run (2.71 °C) is the more informative comparison,
because the difference between them is entirely the exogenous tax path driving
`mu` to 1 by 2130.

### The carbon cycle is not mass-conserving as applied

`Phi` is written **row-stochastic** — every row sums to 1.0, and the source
comments read "Atmosphere -> Atmosphere, Upper", i.e. row `i` is where reservoir
`i`'s carbon *goes*. The code applies it as `M' = Phi @ M`, which reads
`Phi[i, j]` as the flow *into* `i` *from* `j` and therefore requires a
column-stochastic matrix. The columns sum to `(0.930, 1.062, 1.008)`.

The consequences are concrete:

- Carbon is not conserved. On the first step the operator destroys **17.1 GtC**
  (3051.0 in, 3033.9 out). The sign flips as the deep ocean fills, so over the
  full default run the net leak is only 12.7 GtC — 0.4% of all carbon in the
  system — but a single step's error is several times the largest five-year
  change in the atmospheric stock the run ever produces (4.8 GtC).
- The exchange rates between atmosphere and upper ocean are swapped. As applied,
  the atmosphere retains 88% of itself *and* draws 12% of the upper ocean, while
  the upper ocean returns only 5% to the atmosphere. `Phi.T @ M` conserves mass
  exactly and gives a very different answer: peak warming 1.40 °C instead of
  1.71 °C, and `atmospheric_carbon` in 2100 of 599.7 GtC instead of 874.2 — that
  is essentially pre-industrial (588 GtC) by the end of the century, which is
  not credible either.

Neither orientation reproduces DICE, because the ocean rows are hand-set rather
than tied to the reservoir sizes. DICE pins the off-diagonals to the
equilibrium stock ratio, `phi_21 = phi_12 * M_AT_eq / M_UP_eq`; the module's own
initial stocks would imply `0.12 * 851/460 = 0.222`, against the `0.05` it uses.
The atmosphere row (0.88 / 0.12) is DICE's; the ocean rows are not.

Treat the concentration and temperature paths as a qualitative ranking of
scenarios, not as a projection. The ranking is robust — more tax, less carbon,
less warming, in the right order and with sensible relative magnitudes. The
levels are not.

## Gotchas that will bite

- **`time_step_years` only half-rescales the model.** `delta_K`, `c1` and the
  emissions injection all multiply by `dt`; `Phi` does not — it is a fixed set
  of 5-year transition probabilities. At `time_step_years=10` the carbon cycle
  runs at half speed relative to everything else, *and* `delta_K` becomes
  `1.0`, so capital fully depreciates every step.
  `simulate_dice_model(n_periods=15, time_step_years=10)` covers 2020-2160 and
  reports a peak of 1.76 °C against the default's 1.71 °C, for reasons that are
  arithmetic rather than physical. Leave it at 5.
- **`summary()` hard-codes "(5-year steps)"** in its horizon line regardless of
  `time_step_years`. A 10-year run prints
  `Horizon : 15 periods (5-year steps)`.
- **`summary()` prints rows only for 2025, 2050, 2075 and 2100**, and only if
  they land on the index. Both `simulate_dice_model(start_year=2021)` and
  `simulate_dice_model(time_step_years=7)` print the `Key Benchmark Years:`
  header followed by nothing.
- `damage_fraction` is capped at 0.90; `abatement_fraction` is not, but it is
  bounded anyway, because `mu` is capped at 1 and so abatement tops out at
  `p_back*sigma/1000/2.6` — 7.4% of gross output at 2020 values. Damages and
  abatement together therefore cannot exceed 0.974 and `output_net` stays
  positive.

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
