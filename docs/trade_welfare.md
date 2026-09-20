# Hicksian consumption welfare

Use explicit baseline and counterfactual equilibria from consistent accounting:

```python
from puremacro.trade import solve_trade_equilibrium, compute_hicksian_welfare

base = solve_trade_equilibrium(calibration, accounting="consistent")
counter = solve_trade_equilibrium(
    calibration, tau=import_rates, tau_fd=import_rates,
    accounting="consistent",
)
welfare = compute_hicksian_welfare(
    calibration, counter, base_result=base, target_country="USA",
    consumption_categories=(0,),
)
print(welfare.summary())
print(welfare.to_dataframe())
```

Both states must converge and use the same calibration. The welfare function
re-evaluates equilibrium equations, prices, quantities and fiscal accounts using
the recorded tariff schedules. Legacy states, missing schedules, failed solves,
stale fields and unsupported fiscal closures raise errors. Results created before
tariff schedules were recorded must be solved again.

## What enters utility

The default is final-use category 0, conventionally C. Within each final-use
category the model has fixed origin sourcing coefficients: a Leontief basket.
For multiple selected categories, preferences are Cobb–Douglas with weights
`omega_k = theta_k / sum_selected(theta)`. These weights and sourcing coefficients
stay fixed across the comparison. Zero-weight categories make no contribution.

This is conditional **consumption** welfare. Investment (index 1 in a model with
multiple final uses) is excluded. Its spending subtracts foreign saving and does
not obey the consumption expenditure shares. A single-category model is supported
when foreign saving is zero. A model of lifetime utility, saving or public goods
would need its own preferences and intertemporal closure.

Native OECD's C category is HFCE + NPISH + GGFC. Thus its result values the model's
aggregate consumption basket, including government final consumption; it is not
a household-only welfare estimate. That aggregation assumption is explicit.
The user may select additional non-investment categories with
`consumption_categories=(0, 2)`, declaring that they also belong in utility.

## Derivation and sign

Let c_k denote the composite quantity and P_k its **purchaser** price, including
duties and local final-use taxes. For selected positive weights omega summing to
one, the utility and expenditure functions are

```
U(c) = product(c_k ** omega_k)
e(P, U) = U * product((P_k / omega_k) ** omega_k)
h_k(P, U) = omega_k * e(P, U) / P_k
```

Minimizing purchaser expenditure subject to U(c) >= U gives h_k above. Because
the selected consumption budget is m = kappa * (factor income + transfers), with
kappa the sum of its theta shares, observed demands satisfy
`P_k c_k = omega_k m`. The implementation checks this duality in both states.

With 0 denoting baseline and 1 the counterfactual:

```
EV = e(P0, U1) - e(P0, U0)
CV = e(P1, U1) - e(P1, U0)
```

Both are **positive for welfare gains**. EV is the equivalent transfer to the
selected **consumption budget** at baseline prices; positive CV is the amount
removable from that budget at counterfactual prices while retaining baseline
utility. If an income transfer must continue to follow the model's fixed
consumption share kappa, the required total-income transfer is EV/kappa (or
CV/kappa). The reported fields use consumption-budget units. This is the
expenditure-function convention in
[MIT's consumer-theory notes](https://ocw.mit.edu/courses/14-121-microeconomic-theory-i-fall-2015/1b0f90b4fdada65148edd1e4b16aab18_MIT14_121F15_3S.pdf).

The runtime evaluates utility from quantities, and expenditure from purchaser
prices. It uses log ratios and `expm1` for small changes. The wage-deflated
Laspeyres `cpi`, nominal GDP and tariff receipts alone do not determine EV.
No residual component is used to construct the welfare total.

`ev` and `cv` use the calibration's value units. `ev_pct_consumption` divides EV
by baseline selected consumption expenditure; `ev_pct_gdp` uses baseline GDP.
Neither percentage is silently substituted for the level-valued attribution.
The result does not assume that every calibration is in million USD.

## Exact endpoint attribution

Define A = product((P0_k/P1_k) ** omega_k), factor income F, and the solved total
lump-sum transfer T. Consumption EV also equals
`kappa * [A*(F1+T1) - (F0+T0)]` when the checked demand identities hold.
Symmetrically allocating each price/income interaction gives

```
price_effect           = kappa * (A-1) * (F0+F1+T0+T1) / 2
factor_income_effect   = kappa * (1+A) * (F1-F0) / 2
fiscal_transfer_effect = kappa * (1+A) * (T1-T0) / 2
```

These expressions are the Shapley allocation across the three endpoint blocks:
average each block's marginal contribution over all six possible orders. Their
sum is checked against the **separately calculated expenditure-function EV**.
The fiscal term splits into import-duty and domestic-tax rebate effects by the
same weight. Those are subcomponents, not additional gains. Duties are included
once in purchaser prices and once in the transfer account, as specified by the
equilibrium model. A tax receipt is not itself a social welfare gain.

This attribution is conditional on the fixed model numeraire. Mixed endpoint
blocks need not be equilibria, so it is an accounting attribution, not a causal
GE experiment or a terms-of-trade/allocative-efficiency theorem. Changing nominal
units consistently scales all monetary effects; the percentage EV is unchanged.

## Evidence and boundaries

The independent reference solves a two-country, one-producing-sector economy
with two distinct consumption baskets and a separate investment category. It
then numerically minimizes **origin-level** purchases subject to Leontief and
utility constraints at each price/utility pair. This primal optimization does
not call the runtime expenditure formula. All six Shapley orderings are evaluated
separately. See [reference code](https://github.com/jalonso1979/puremacro/blob/v4.3.0/tools/reference_validation/validate_trade_welfare.py)
and [recorded validation](https://github.com/jalonso1979/puremacro/blob/v4.3.0/reviews/2026-09-20-hicksian-welfare/REPORT.md).

Additional checks cover identity, reversed comparisons, currency scaling,
subsidies, small shocks, stale/failed states and the frozen OECD aggregation.
The new interface supports the [consistent model](trade_accounting.md)'s
Leontief/CES sourcing with lump-sum rebates. Tariff optimizers, Nash searches and payoff matrices now select this calculation
with `metric="hicksian_ev"`; see [policy integration and solver recovery](trade_policy.md).
Historical objective aliases and flexible-model welfare proxies remain unchanged.

`decompose_hicksian_ev_3way` remains unavailable: its `TOT`, `Alloc` and
`TariffRec` fields describe a different, unsupported decomposition. Use
`compute_hicksian_welfare` for the derived EV/CV and attribution above. Theorem
certification also remains unavailable. Existing legacy containers and historical
quarantined tests are retained for compatibility, not counted as validation.
