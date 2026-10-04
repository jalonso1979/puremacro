# Tariff scenarios of the legacy trade model on the clean OECD 2020 table — 4 October 2026

**Question.** Release 4.6.0 bundled the clean OECD 2020 77x11 table and the
legacy MATLAB model's base solution on it, but none of the six tariff scenarios
of `Main77c_11s.m` (t10 = a 10% US reciprocal tariff on every partner; t10_25 …
t10_145 raise the China/Hong Kong rate). Why, and can a reference be produced?

**Answer.** On the clean table, the legacy model's equilibrium path from the base
**ends in a fold at a US tariff of about 0.89%**. A larger uniform US tariff has
no equilibrium connected to the base, so no root-finder could converge at 1%,
2%, 5% or 10%. Past the fold the path returns to zero tariff at a second
equilibrium, with Costa Rica's prices about 8.5% below base. The cause is the legacy closure, not the solvers or the data. In
that closure every country's foreign balance is an unknown, and each country's
equations satisfy a budget identity, so its foreign-balance equation is implied
by its other equations up to four accounting wedges of the MATLAB code. Each
country's price level relative to the rest of the world is therefore pinned
only by how those wedges respond to it. For Costa Rica on the clean table, two
of them nearly cancel, and the net response goes to zero along the tariff path.
The section "Cause" below gives the identity, the measurements and two causal
checks: correcting the MATLAB operator precedence (one of the wedges) makes all
six scenarios solve in 3–5 iterations; and on the legacy table, where every
scenario solves, Costa Rica's trade-valuation wedge outweighs the other two about
threefold, so nothing cancels.

The search that led here: no root was found by any of seven methods, and the failures share
one signature: at every positive US tariff tried (1%, 2%, 5%, 10%) the iterates
reach a point where the Jacobian of the legacy equation system, after row and
column equilibration, has one singular value of order 1e-7 against a second
smallest of order 1e-4, and the corresponding singular vectors are confined to
Costa Rica (CRI): its transfer `T`, factor prices `r` and `w`, and its sector
prices on the unknown side; its eleven output-market equations on the equation
side. Damped Levenberg–Marquardt converges to a residual minimum with maximum
residual 0.79 (L1 9.4), not to a root. The same equilibrated Newton, started
cold from the base on the **legacy** table, reproduces MATLAB's bundled t10
reference to a maximum relative difference of 2.1e-12 (L1 residual 1.35e-3
against MATLAB's own 1.9e-3), so the method is exact where a root exists.

## What was tried

| Method | Start | Result |
|---|---|---|
| Legacy MATLAB Newton, absolute FD step 1e-2 (`Main77c_11s.m`) | base | L1 residual oscillates between 30 and 2,600 for 176 iterations (`logs/matlab_newton_eps1e-2.log`) |
| Same loop, relative FD step 1e-6 | base | NaN at iteration 49; the other scenarios inherit the NaN seed (`logs/matlab_newton_relative_step.log`) |
| MATLAB `fsolve` trust-region-dogleg | base | stops at its 150-iteration limit, L1 1.3e3 (`logs/matlab_fsolve_t10.log`) |
| puremacro `solve_trade_equilibrium(method="newton")` | base | 50 iterations, max residual 2.6 |
| puremacro Newton, 1% tariff steps warm-started | base | stalls at the first step (`logs/t10_stepwise.log`) |
| Equilibrated Newton with backtracking (`equilibrated_newton.py`) | base | 10%: max residual 8.7 after 27 iterations; 1%: 0.22; 2%: 1.8; 5%: 18.6 (`logs/t10_scaled.log`, `scaled_r0*.log`) |
| Equilibrated Levenberg–Marquardt (`equilibrated_lm.py`) | stalled 10% iterate | `xtol` termination at max residual 0.79, L1 9.4 (`logs/t10_lm.log`) |
| Keller pseudo-arclength continuation (`solve_keller_pac`, legacy accounting) | base | no progress in 40 minutes; stopped |
| **Control: equilibrated Newton on the legacy table** | base | **reaches MATLAB's L1 tolerance in 7 iterations; MATLAB's t10 reference to max rel. diff 2.1e-12** (`logs/scaled_legacy_t10.log`) |

## Diagnostics

* At the base solution on the clean table the raw Jacobian spans seventeen
  orders of magnitude (level equations in USD million next to log equations),
  but after ten Ruiz equilibration sweeps its condition number is 9.4e3 with no
  singular value below 1e-6 of the largest: the base is a regular point and the
  implicit function theorem guarantees nearby solutions for small tariffs.
* On the 1% path the Newton iterates reduce the L1 residual 1.4e4 → 913 → 65 →
  2.97 in three full steps, after which the equilibrated condition number jumps
  from 1e4 to 1e6–1e7 and the line search collapses: the Newton direction is
  dominated by the near-null Costa Rica direction, so even a step scaled by
  3e-8 moves the state by O(0.03) and increases the residual.
* Costa Rica's calibration is unremarkable on both tables (income 6.2e4 USD
  million, foreign saving 2.2% of income, positive sales and value added in
  every sector, consumption shares 0.81/0.19/0.002), and at the stalled
  iterate its income, transfers, prices and net investment demand are all
  interior (income −4.6%, prices −5%, net investment demand 9,987 > 0). No
  clipping or branch of the residual code is active (lump-sum closure, no
  capacity penalties, all outputs positive).
* The five negative final-demand cells of the clean table (CYP, LTU, PHL, UKR
  mining investment; a VNM manufacturing cell) and its fifteen negative
  net-tax cells do not involve Costa Rica.

The near-null vector is close to a uniform rescaling of Costa Rica's nominal
variables (`T` 0.69, `r` 0.46, `w` 0.46, sector prices 0.05–0.16, `XN` 0.05).
An earlier draft of this report expected a fixed foreign-saving term to pin that
direction. There is none: in the legacy closure `XN` is an unknown. The next
section explains what pins it instead.

## Cause: one weakly pinned direction per country

**A budget identity per country.** Write `λ_j` for the weights `p` on country
j's goods-market equations, `y` on its zero-profit equations, `w` and `r` on
its factor-market equations, 1 on its fiscal equation and −1 on its
foreign-balance equation. Then for any state `x`, not only at a root, the legacy
residuals satisfy

    λ_j(x)' F(x) = W1_j + W2_j + W3_j + W4_j

with four accounting wedges:

| Wedge | Definition | Source in the MATLAB code |
|---|---|---|
| W1 | value-added cost minus factor payments | the operator precedence of `ff_equi.m:31`, `w/(1-α)^(1-α)`, makes unit value-added cost homogeneous of degree 1+α in (r, w) |
| W2 | production tax charged in prices minus tax collected, Σ t (pp − 1) y | the fiscal block collects `t*y` rather than `t*p*y` |
| W3 | tariffs paid on imports minus tariff revenue credited | `tauf = 0` in the scenarios credits none (only the US is affected) |
| W4 | legacy trade balance minus the producer-price trade balance | final-demand trade is valued at the importer's composite price `ppfd`, not at the exporter's price |

The identity holds to 3e-8 (values of order 1e5) at the base, at the base state
under t10 tariffs, and at the stalled t10 iterate
(`tools/reference_validation/trade_clean_table/wedge_identity.py`). With
consistent accounting the four wedges vanish, so `λ_j' J = 0` for every country.
Then each country's foreign-balance equation is implied by its other equations,
and its price level relative to the world, together with its foreign balance,
is undetermined. This is the redundancy that `docs/trade_accounting.md` removes
in consistent mode by fixing foreign balances. In the legacy model the wedges
alone break it.

**77 weak directions.** The equilibrated Jacobian at the clean base has exactly
77 singular values between 1e-4 and 1e-2 of the largest, then a gap to 3.4e-2.
Each of the 77 singular vectors is localised on one country (shares 0.3–0.9).
Along each, the wedge responses split into two opposing groups of nearly equal
size: dW1 + dW2 > 0 (as the country's prices rise, value-added cost outgrows
factor income and taxed output value outgrows tax collected) and dW4 < 0 (its
final-demand imports are revalued at its own rising composite price, its
final-demand exports at its partners' unchanged ones). The net response is the
pin. Weakest directions at the clean base:

| Country | singular value | dW1 + dW2 | dW4 | net |
|---|---|---|---|---|
| LTU | 1.06e-4 | 270 | −275 | −4.8 |
| CZE | 1.74e-4 | 818 | −840 | −22 |
| HKG | 2.45e-4 | 715 | −687 | 28 |
| EST | 2.55e-4 | 184 | −176 | 8.2 |
| MEX | 2.64e-4 | 1,771 | −1,681 | 90 |
| CRI | 4.53e-4 | 305 | −286 | 19 |

As a rough guide, dW1 + dW2 is about capital income plus net production taxes,
and dW4 is about final-demand exports plus the domestic-price share of
final-demand imports. The clean-table countries where these two national-account
aggregates are within 10% of each other (AUT, BEL, BGR, CHE, EST, HKG, JOR, LTU,
MEX, POL, THA) are, except BEL (13th), all among the twelve weakest directions.

**Costa Rica's pin goes to zero: a fold.** Natural continuation in the US
tariff (equilibrated Newton, secant predictor) converges at every 0.1-point
step to 0.8%. Meanwhile Costa Rica's singular value falls (2.3e-4 at 0.4%,
1.5e-4 at 0.7%, 1.0e-4 at 0.8%) and its price level falls at an accelerating
rate (−1.5%, −2.7%, −3.3%). Continued past that point in Costa Rica's mean log
price instead of the tariff (`fold_trace.py`), the tariff rises to 0.8928% and
then falls back (0.8652% → 0.8928% → 0.8793% in equal steps), with the smallest
singular value at its minimum (8.5e-6) at the turn. A parabola through the
three points puts the fold at a 0.893% tariff, with Costa Rica's prices 4.6%
below base. Beyond the fold the branch does not turn up again: followed to
q = −0.103, the tariff falls monotonically (0.82%, 0.73%, 0.60%, 0.45%, 0.28%,
0.10%, then −0.07%, −0.23%, −0.36%), with every step converged to a maximum
residual below 1e-5. It crosses zero tariff at q ≈ −0.089. **The legacy model
therefore has a second equilibrium on the clean table at zero tariff**, with
Costa Rica's prices about 8.5% below the calibrated base. The bundled base is
the calibrated one of the two. At the stalled
t10 iterate, Costa Rica's direction has singular value 4e-8, and its wedge
responses are 356 against −356 (net 0.22).

**Why the legacy table solves.** On the legacy table Costa Rica's direction is
pinned firmly (singular value 4.9e-3 at the base and 4.9e-3 at the MATLAB t10
solution): dW1 + dW2 = 8.3e4 against dW4 = −2.5e5, a net response twice the
size of either group's smaller member. The corrupted table gives Costa Rica net
production taxes of −8.3e7 against capital income of 1.8e8, so the two groups
are far apart. On the clean table its net production taxes are 1.1e3 against
capital income of 1.9e4, and the two groups nearly coincide.

**Causal check: remove W1.** With `replicate_matlab_precedence=False` (correct
Cobb–Douglas, W1 ≡ 0; the base equilibrium is unchanged because w = r = 1
there), Costa Rica's pin becomes dW2 + dW4 ≈ −270, with no cancellation. All six
scenarios then converge on the clean table with `method="equilibrated_newton"`:
t10 in 4 iterations, then t10_25, t10_54, t10_75, t10_125 and t10_145 (seeded
from t10, as in `Main77c_11s.m`) in 3, 4, 4, 5 and 5, with maximum residuals of
1e-5 or less. These are solutions of a corrected model, not of the MATLAB code,
so they are not MATLAB references. Consistent accounting, which removes the
redundancy properly, rejects the clean table because of its five negative
final-demand cells.

## Consequence for the bundled references

`trade_reference_solutions_oecd2020.npz` carries the base scenario only, and
`REFERENCE_MANIFEST_OECD2020.json` lists the six tariff scenarios as absent. No
unconverged solution is shipped, and none can be produced for the legacy model
on this table: the scenarios lie beyond the fold. The corrected-precedence
solutions above are not shipped either, because they are not MATLAB references.

The equilibrated Newton is now in the library as
`solve_trade_equilibrium(calib, method="equilibrated_newton", ...)`. From a cold
start it reproduces MATLAB's legacy-table t10 reference to a maximum relative
difference of 2e-10 in 6 iterations (`tests/test_trade_equilibrated_newton.py`,
slow test). When it fails it reports in `metadata` the equilibrated condition
number of the last Jacobian and the country on which the weakest direction
loads (`near_singular_country`, `near_singular_country_weight`). On the clean
t10 problem that country is CRI. `polish_rate.m` turns a puremacro root into a
MATLAB-evaluated one when a root exists.

## Reproduce

```bash
S=$(mktemp -d)/trade; mkdir -p $S/matlab
PYTHONPATH=. python tools/reference_validation/trade_clean_table/equilibrated_newton.py $S 0.10 t10          # clean table
PM_TABLE=legacy PYTHONPATH=. python tools/reference_validation/trade_clean_table/equilibrated_newton.py $S 0.10 t10   # control
PYTHONPATH=. python tools/reference_validation/trade_clean_table/equilibrated_lm.py $S 0.10 t10 $S/matlab/puremacro_t10_oecd2020_scaled.mat

# Cause: budget identity, wedge responses along the weakest directions, fold
PYTHONPATH=. python tools/reference_validation/trade_clean_table/wedge_identity.py oecd2020 0      # clean base
PYTHONPATH=. python tools/reference_validation/trade_clean_table/wedge_identity.py legacy 0.10     # legacy table, t10
PYTHONPATH=. python tools/reference_validation/trade_clean_table/fold_trace.py 16 -0.004           # ~15 min
```

The precedence check is `solve_trade_equilibrium(calib, method="equilibrated_newton",
replicate_matlab_precedence=False, tau=..., tau_fd=..., tauf=0, tauf_fd=0)` with
the scenario tariffs of `wedge_identity.us_reciprocal_tariff`.
