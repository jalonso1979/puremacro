# Adaptation and sovereign risk

[Español](es/sovereign_adaptation.md)

`puremacro.sovereign.adaptation` jointly solves public adaptation investment,
one-period external debt and default on finite debt/stock grids over a finite
horizon. Lenders price future default conditional on both the new debt and
the adaptation stock chosen today. Exact probability propagation separates
new defaults from economies already excluded from borrowing.

This experimental model has an externally specified Markov disaster process,
log household utility, lump-sum taxes and permanent exclusion until the final
date. Households cannot borrow directly. It has no domestic banks, temperature
model, emissions policy, reentry or infinite-horizon terminal approximation.
Its disaster contract forgives part of a one-period payment; it does not defer
that payment or reproduce a long-term disaster clause.

## Use the model

```python
import numpy as np
from puremacro.sovereign import (
    SovereignAdaptationParameters, solve_sovereign_adaptation,
    audit_sovereign_adaptation, audit_adaptation_distribution,
)

result = solve_sovereign_adaptation(
    income=[1.0, 1.0], damage=[0.0, 0.4], disaster=[False, True],
    transition=[[0.9, 0.1], [0.75, 0.25]],
    debt_grid=np.linspace(0.0, 0.4, 21),
    adaptation_grid=np.linspace(0.0, 0.3, 11), horizon=6,
    parameters=SovereignAdaptationParameters(grant_rate=0.5),
)
path = result.distribution([1.0, 0.0])
assert audit_sovereign_adaptation(result)["accepted"]
assert audit_adaptation_distribution(path)["accepted"]
print(path.to_frame()[["investment", "disaster_damage", "new_default_probability"]])
print(path.value)
```

`result.to_frame(date=t)` reports the realized optimal accounts at every
market-access state; `excluded=True` reports excluded states. It includes
taxes, consumption, damage, exclusion losses, grants, debt service, contract
relief, default writeoffs, investment and scrapping. Undefined forced-repayment
utility is missing with `repayment_feasible=False`; selected consumption is
always positive. `pricing_frame(t)` reports price and default probability for
each next-debt/stock choice. Tables are fresh copies and stored arrays own
read-only memory. Markdown, LaTeX and Typst reports are available.

`distribution(initial_probabilities, debt_index=0, adaptation_index=0,
market_access=True)` starts at exact grid nodes and propagates probability
mass without Monte Carlo simulation. An excluded initial state must have zero
debt. `new_default_probability` counts entry events, while
`excluded_probability` includes current defaults and prior exclusion.

## Timing, financing and damage

Dates are (t=0,\ldots,T-1). The current state (z) supplies undamaged income
(y_z>0), disaster flag (h_z), damage fraction \(\theta_z\in[0,1)\), and the
transition row (P_{z,:}). Positive damage requires a disaster flag. Hazard is
exogenous: adaptation reduces severity, not disaster probability.

Installed adaptation (a) protects income before it depreciates:

\[
\ell(a,z)=y_z\theta_z e^{-\eta a},\qquad Y(a,z)=y_z-\ell(a,z).
\]

The government observes inherited face debt (b\geq0), installed stock and
the shock. It then chooses repayment or default. If it repays, the government
chooses next debt (b') and next installed stock (a') from the supplied grids.
Investment and scrapping are

\[
i=\max\{a'-(1-\delta)a,0\},\quad
s=\max\{(1-\delta)a-a',0\},\quad a'=(1-\delta)a+i-s.
\]

New investment protects only from the next date. Free disposal has **no
salvage proceeds**. It allows exact grid transitions without interpolation or
forced upward rounding after depreciation. A coarse stock grid can induce
extra scrapping; this is one reason to refine the stock and debt grids jointly.

A donor reimburses fraction (g\in[0,1]) of investment. The contract promises
payment (m_z b), where (m_z=1-\rho h_z) and \(\rho\in[0,1]\) is disaster
forgiveness. On repayment, the government budget and household consumption are

\[
\tau=m_zb+i-gi-q_t(b',a',z)b',\qquad c=Y(a,z)-\tau.
\]

Default cancels the remaining promised payment, prevents issuance immediately,
and imposes exclusion for all remaining dates. Each excluded period destroys
fraction \(\phi\in[0,1)\) of output after disaster damage. Adaptation remains
available using income and the same donor reimbursement:

\[
c^E=(1-\phi)Y(a,z)-(1-g)i.
\]

Disaster damage, the exclusion loss and contractual forgiveness are separate
entries. Face debt equals paid service plus contractual forgiveness plus the
remaining service written off in default. State by state, resources satisfy

\[
c+i+\text{service}=y_z-\ell-\text{exclusion loss}
  +\text{borrowing proceeds}+\text{grant}.
\]

## Backward solution and lender pricing

The government maximizes expected \(\sum_t\beta^t\log c_t\), with
(0<\beta\leq1). For each current stock/shock, the solver exhausts all next
stock choices in exclusion and all debt/stock pairs on repayment. Risk-neutral
foreign lenders observe both choices. At (t<T-1),

\[
q_t(b',a',z)=\frac{1}{R}\sum_{z'}P_{z,z'}m_{z'}
 [1-d_{t+1}(b',a',z')].
\]

Future default is known from backward induction; no price iteration or price
monotonicity restriction is imposed. The no-default benchmark price is
\(\sum Pm/R\), which can be below (1/R) because of contract forgiveness.
The difference from (1/R) must not all be labelled a default premium.

At (T-1), borrowing and new investment are zero and continuation/salvage
value is exactly zero. Existing stock survives depreciation physically but has
no modeled continuation value. Terminal stock policy indices are `-1` and
terminal price arrays contain zero placeholders. `pricing_frame()` rejects
terminal dates: those placeholders are not market quotes. Terminal default
still compares repayment with exclusion consumption.

Repayment wins default gains no larger than `decision_tolerance` (default
`1e-12`, allowed `[0,1e-8]` in log-utility units). Exact repayment-action ties
prefer lower new debt, then lower next stock; exclusion ties prefer lower next
stock. Infeasible repayment keeps value `-inf` internally, without a consumption
floor. Overflow raises an error. Both grids begin at exact zero and increase
strictly; singleton `[0]` grids impose no borrowing or no adaptation. Transition
rows must sum to one within `1e-12` and are normalized within that roundoff
tolerance. Limits are 30 dates, 500,000 access-state cells, and a conservative
250 million state/action evaluation bound.

## Validation and welfare interpretation

`audit_sovereign_adaptation()` independently recomputes all Bellman maxima,
selected-action values and feasibility, pricing, default decisions and accounts.
`audit_adaptation_distribution()` checks mass flows, timing and equality of
initial Bellman value to expected discounted **log** consumption. It does not
substitute the log of mean consumption. Default tolerance is `1e-9`, with errors
scaled by `1 + abs(reference)`. This checks the specified discrete model, not
continuous-state or infinite-horizon accuracy.

Tests include a separate memoized scalar dynamic program, enumeration of every
shock history in small models, a two-date analytic investment condition, unit
and state-order changes, and no-disaster/no-adaptation/no-borrowing limits.
Terminal default has the reference threshold (m_zb>\phi Y(a,z)), subject to
the utility tie tolerance. With zero exclusion loss, positive unsecured debt
has zero price in this finite model. Full disaster forgiveness eliminates the
contractual payment in that state and is explicitly reflected in price.

For two distributions with identical beta and horizon,
`alternative.consumption_equivalent(reference)` returns
\(\exp[(V_{alt}-V_{ref})/\sum_t\beta^t]-1\). This uniformly scales the reference
consumption stream. It measures domestic household welfare in this model;
donor costs are reported as grants but donor welfare is excluded.

## Reproduce the experiment

```sh
python -m puremacro.examples.sovereign_adaptation --output adaptation-en --lang en
python -m puremacro.examples.sovereign_adaptation --output adaptation-es --lang es
```

The synthetic six-date example compares baseline, a 50% grant, full disaster
forgiveness, no adaptation and no borrowing, on joint debt/stock grids of
21/11, 41/21 and 81/41 points. It retains all grids and refinement criteria.
The pilot's forgiveness case reached the debt ceiling, so two additional
domain checks widen the debt interval while holding grid spacing fixed.
The original no-borrowing investment comparison failed its preset limit;
additional 81/161-node stock checks are reported separately, preserving that
failure. The coarse grant comparison also retains a roundoff-level boundary flag.
Boundary mass, scrapping and failed comparisons remain visible. Full policy/
price/value arrays and probability masses are saved, alongside accounts,
welfare, fingerprints and audits. No empirical calibration is claimed.

[Mallucci (2020)](https://www.federalreserve.gov/econres/ifdp/files/ifdp1291r1.pdf)
motivates the disaster/debt mechanism. The adaptation technology, finite
horizon, permanent exclusion and one-period forgiveness here are distinct
assumptions; this is not a replication of that paper's long-term debt model.
