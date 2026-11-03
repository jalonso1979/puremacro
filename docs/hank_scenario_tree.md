> 🇬🇧 English · 🇪🇸 [Español](es/hank_scenario_tree.md)

# Finite-tree aggregate-risk HANK

This experimental solver extends the [published one-asset household and
equilibrium equations](hank_equilibrium.md) to a finite tree of monetary
histories. Households choose saving and hours before future shocks are known.
At a node, EGM uses the conditional expectation of successor marginal values.
Each history retains its own conditional household distribution. Idiosyncratic
income transitions are independent of the aggregate branch.

The following small grid demonstrates the API; it does not meet the refinement
protocol reported below.

```python
from puremacro.models import (
    calibrate_one_asset_hank, monetary_scenario_tree, solve_hank_scenario_tree,
)

ss = calibrate_one_asset_hank(n_a=120, n_e=5)
tree = monetary_scenario_tree(horizon=12, branch_date=1,
                             innovations=(-.005, .005), probabilities=(.5, .5))
result = solve_hank_scenario_tree(ss, tree)
print(result.converged, result.diagnostics)
print(result.expected_paths())
```

The probabilities are conditional. The date-one shock is unresolved at date
zero; subsequently it decays at the declared persistence. A mean-zero shock
need not reproduce the no-risk path because households integrate nonlinear
marginal values before choosing. Averaging separately solved announced paths
would impose a different information structure.

## Explicit economic closure

Inflation, wages and output at every node solve asset-market clearing,
labor-market clearing and the conditional expected Phillips equation. The
parent chooses the nominal policy rate; realized real returns depend on child
inflation. Fixed real government debt is financed by taxes equal to realized
net interest times debt. Dividends and taxes remain proportional to skill.

The firm-valuation kernel is **assumed** to be `M_child=1/(1+r_child)`, equivalent
to risk-neutral valuation in nominal units. The stochastic Phillips equation is

$$0=\kappa(w_v-1/\mu)-\log(1+\pi_v)+
\sum_{c\in children(v)}p(c\mid v)\frac{Y_c}{Y_v}
\frac{\log(1+\pi_c)}{1+r_c}.$$

This kernel is not inferred from household marginal utility. On a deterministic
chain the equations reduce to the existing published model. That reduction
does not validate the stochastic firm-valuation assumption empirically.

Leaves use steady-state household marginal values and zero future inflation.
There is no solved stochastic continuation. The API accepts non-recombining,
topologically ordered trees of up to 256 nodes, with one root and a common leaf
date. Different histories cannot silently share a household distribution.

The square equilibrium system uses a Powell hybrid root solver, which updates
its Jacobian between full numerical sweeps. Tests compare it with a separate
least-squares optimization of the same equations. `max_evaluations=None` uses
SciPy's dimension-dependent budget; an explicit integer counts residual calls,
including numerical Jacobian probes. Every returned solution still faces the
same economic and numerical acceptance checks.

## Evidence and limits

Tests compare deterministic paths for both shock signs with the separate
published-model solver; duplicate histories give the same announced path.
Further checks reconstruct node budgets, inherited distributions, nominal-debt
returns, fiscal financing and conditional expected Phillips equations. Changing
branch probabilities changes choices before realization.

```sh
python -m pytest tests/test_models/test_hank_scenario_tree.py -q
python tools/reference_validation/check_hank_scenario_tree.py --output tree-evidence
```

The study saves all node paths, household arrays, source hashes, actual
diagnostics and failed refinement comparisons. Its declared comparison uses
the first six dates of conditional consumption, output and inflation: 1% of
the finer response's peak with a `1e-6` level floor. It compares 12/24-quarter
horizons, 120/240 asset nodes and five/seven income states.
Each grid recalibrates discounting and labor disutility to the same bond and
effective-labor targets; the calibrated coefficients are retained. This is a
comparison of discretized calibrations, not a fixed-coefficient experiment.

The completed record in `tests/fixtures/hank_scenario_tree/` extends the study
to 31 income states, 960 asset nodes, an asset bound of 600 and 96 quarters.
The deterministic oracle gap is `7.10e-10` against a `3e-8` tolerance. The
following final comparisons pass for C, Y and inflation:

| Refinement | Fixed dimensions | Largest absolute change across C/Y/pi |
|---|---|---:|
| Asset bound 300 → 600 | 480 assets, 15 incomes, 48 quarters | 3.76e-7 |
| Income states 21 → 31 | 480 assets, bound 600, 48 quarters | 3.76e-5 |
| Asset nodes 480 → 960 | 31 incomes, bound 600, 48 quarters | 4.67e-7 |
| Horizon 48 → 96 | 480 assets, 31 incomes, bound 600 | 1.16e-7 |

The income comparison reaches 98.3% of its consumption tolerance. Earlier
horizon and income comparisons fail and remain in the record, as do the
narrow-grid failures from policy exits and invalid steady distributions.
`all_refinements_passed` therefore remains false. The accepted comparisons
concern the first six conditional response dates under the stated innovation;
they do not establish infinite-grid accuracy or precision of small uncertainty
corrections. The fixture README includes the complete extension command.

All node paths and expected paths are retained. Large household-array payloads
are omitted from the compact fixture, explicitly listed with their original
hashes, and reproducible using the runners.

`converged` requires numerical market, budget and mass checks and less than
`1e-10` distribution mass choosing outside the asset grid. Euler/KKT errors and
upper-grid mass are additional diagnostics. Small market residuals alone do
not establish horizon or grid accuracy. There is
no independent stochastic author-code reference, general stationary
aggregate-risk equilibrium, infinite-horizon welfare certificate or two-asset
claim. The older reduced aggregate-risk demonstration remains unchanged.
