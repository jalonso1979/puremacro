> 🇬🇧 English · 🇪🇸 [Español](es/trade_adjustment.md)

# Conditional dynamic sourcing

`solve_sourcing_adjustment` chooses origin quantities for one unit of a CES
composite at each date. Delivered prices are exogenous. It minimizes

$$\sum_{t=0}^{T-1}\delta^t\left[p_t'q_t+
\frac{\kappa}{2}\sum_o\frac{(q_{to}-q_{t-1,o})^2}{\alpha_o}\right]$$

subject to CES production of at least one unit:

$$F(q_t)=\left[\sum_o\alpha_o(q_{to}/\alpha_o)^{(\sigma-1)/\sigma}
\right]^{\sigma/(\sigma-1)}\ge1.$$

For `sigma=1`, use the Cobb–Douglas limit. Shares are strictly positive and sum
to one; baseline origin prices are one and initial quantities default to the
shares. CES is concave, its superlevel feasible set is convex, and adjustment
costs are convex. Free disposal is permitted. There is no production requirement
or switching cost after the last date: horizon choice is part of the model.

```python
import pandas as pd
from puremacro.trade import solve_sourcing_adjustment

prices = pd.DataFrame({"domestic": [1., 1., 1., 1.],
                       "imported": [1., 1.3, 1.3, 1.]})
result = solve_sourcing_adjustment(
    prices, pd.Series({"domestic": .7, "imported": .3}),
    elasticity=3., adjustment_cost=2., discount=.99,
)
print(result.accounts)
print(result.diagnostics)
```

Accounts separate procurement, adjustment resources, total cost, output and the
frictionless CES index. Numerical acceptance requires production feasibility,
reconstructed KKT stationarity and complementarity. A failed optimization or
an unresolved artificial quantity floor raises, preserving the failure.

This model supplies a conditional sourcing mechanism. It does not determine
national wages, output, trade balances, worker mobility or tariff recycling.
Passing total unit costs into household prices is an explicit transmission
assumption, as in the [observed household study](household_policy_studies.md).
The existing [dynamic MRIO equilibrium](trade_dynamic.md) has a different
economic closure; the two models are not interchangeable.
