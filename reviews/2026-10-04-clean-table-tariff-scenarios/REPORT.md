# Tariff scenarios of the legacy trade model on the clean OECD 2020 table — 4 October 2026

**Question.** Release 4.6.0 bundled the clean OECD 2020 77x11 table and the
legacy MATLAB model's base solution on it, but none of the six tariff scenarios
of `Main77c_11s.m` (t10 = a 10% US reciprocal tariff on every partner; t10_25 …
t10_145 raise the China/Hong Kong rate). Why, and can a reference be produced?

**Answer.** No root was found by any of seven methods, and the failures share
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
against MATLAB's own 1.9e-3), so the method is exact where a root exists; the
obstruction is specific to the legacy closure on the clean data.

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
variables (`T` 0.69, `r` 0.46, `w` 0.46, sector prices 0.05–0.16, `XN` 0.05),
which the fixed foreign-saving term should pin. Why that pin loses its grip
along the tariff path on the clean table but not on the legacy table is not
resolved here; it is a property of the legacy closure (`accounting="legacy"`,
`tariff_revenue_mode="legacy_national"`, foreign saving fixed in numeraire
units) and would need a model-level analysis.

## Consequence for the bundled references

`trade_reference_solutions_oecd2020.npz` carries the base scenario only, and
`REFERENCE_MANIFEST_OECD2020.json` lists the six tariff scenarios as absent. No
unconverged solution is shipped. The equilibrated Newton of
`tools/reference_validation/trade_clean_table/equilibrated_newton.py` is the
solver to use for any future attempt (it is the one validated against MATLAB to
1e-12 on the legacy table); `polish_rate.m` turns a puremacro root into a
MATLAB-evaluated one when a root exists.

## Reproduce

```bash
S=$(mktemp -d)/trade; mkdir -p $S/matlab
PYTHONPATH=. python tools/reference_validation/trade_clean_table/equilibrated_newton.py $S 0.10 t10          # clean table
PM_TABLE=legacy PYTHONPATH=. python tools/reference_validation/trade_clean_table/equilibrated_newton.py $S 0.10 t10   # control
PYTHONPATH=. python tools/reference_validation/trade_clean_table/equilibrated_lm.py $S 0.10 t10 $S/matlab/puremacro_t10_oecd2020_scaled.mat
```
