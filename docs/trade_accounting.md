> 🇬🇧 English · 🇪🇸 [Español](es/trade_accounting.md)

# Producer and purchaser accounting in the trade model

Use the consistent mode for new counterfactuals. This example runs as written
on the hand-balanced two-country table used by the reference check below:

```python
import numpy as np
from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium

# Two countries (A, B), one good, final uses C and I, values in one currency unit.
Z = np.array([[10., 12.], [8., 14.]])                      # intermediate sales
F = np.array([[30., 8., 20., 20.], [15., 15., 50., 18.]])  # final sales to A-C, A-I, B-C, B-I
production_tax = np.array([4., 6.])
final_tax = np.array([2., 1., 3., 2.])
value_added = Z.sum(1) + F.sum(1) - Z.sum(0) - production_tax
data = np.vstack([np.hstack([Z, F]), np.r_[production_tax, final_tax],
                  np.r_[2*value_added/3, np.zeros(4)], np.r_[value_added/3, np.zeros(4)]])
calibration = calibrate_trade_model(data, ns=1, nc=2, nfd=2, country_codes=["A", "B"],
                                    sector_codes=["GOOD"])

# Tariff MULTIPLIERS (1 + rate), indexed (origin node, use, destination country):
# A charges 20% on imports from B, B charges 10% on imports from A.
intermediate_tariff_multipliers = np.ones((2, 1, 2))
intermediate_tariff_multipliers[1, 0, 0] = 1.2
intermediate_tariff_multipliers[0, 0, 1] = 1.1
final_tariff_multipliers = np.ones((2, 2, 2))
final_tariff_multipliers[1, :, 0] = 1.2
final_tariff_multipliers[0, :, 1] = 1.1

result = solve_trade_equilibrium(
    calibration,
    tau=intermediate_tariff_multipliers,
    tau_fd=final_tariff_multipliers,
    accounting="consistent",
    tol=1e-9,  # absolute, in the table's value units
)
assert result.converged
print(result.metadata["account_residuals"])
```

`tol` bounds **absolute** residuals in the calibration's value units. The
attainable floor is roughly `1e-16` times the largest output value: `tol=1e-9`
suits the table above (values in tens), while the frozen OECD 3-region x
3-sector fixture in million USD (outputs up to `6.4e7`) bottoms out near
`5e-8`; use `1e-5` to `1e-7` there, and rescale the tolerance when you change
the currency unit. Consistent accounting needs a calibration whose final-use
coefficients and expenditure shares are nonnegative: the bundled legacy
77-country x 11-sector table (`load_icio_data(source="legacy")`) is rejected with a
`ValueError` (three negative investment cells and one negative investment
share); use native ingestion with `package_mrio_to_calibration_result`, the
frozen OECD fixture, or your own balanced table.

`accounting="legacy"` remains the compatibility default. It reproduces historical
MATLAB conventions, including their price-precedence and final-demand valuation
choices. `tariff_revenue_mode="schedule"` alone does **not** select the new
accounting. Consistent mode selects the complete schedule, homogeneous factor
costs and the explicit closure below. It overrides the legacy precedence and
national-rate switches, and reports the effective conventions in metadata.
Historical tariff-game/policy wrappers retain their existing conventions unless
they explicitly pass the consistent option; this change does not certify their
welfare measures.

## Prices, quantities and fiscal receipts

Let i index an originating country/sector, j a producing destination node, k a
final-use category, and c a destination country. Prices are relative to baseline
producer prices, which equal one. There are no iceberg or distribution margins
in this model; these would require separate resource and valuation equations.

| Item | Definition |
|---|---|
| Producer/border invoice price | p_i |
| Intermediate purchaser price | p_i τ^Z_ij |
| Final composite cost before local tax | Q_kc = Σ_i a^F_ikc p_i τ^F_ikc |
| Final composite purchaser price | P_kc = Q_kc / (1 − δ_kc) |
| Bilateral producer value | p_i × delivered quantity |
| Import duty | (τ − 1) × p_i × delivered quantity |

Tariffs use the exporting good's price for **both** intermediate and final
deliveries. They are paid once by the buyer and rebated once to the importing
country. The local final tax is applied to tariff-inclusive expenditure. Domestic
entries in the import-duty schedule must equal one. Subsidized imports are
allowed through positive multipliers below one.

For a final-use expenditure E_kc, deliveries are

```
C_kc = E_kc / P_kc
F_ikc = a^F_ikc C_kc
local final tax = δ_kc E_kc
```

Thus producer value + import duties + local final tax equals purchaser
expenditure in every category/country. The δ calibration is source final tax
divided by actual source final purchases **including that tax**. Financial
saving is excluded from this denominator. The stored legacy `tax_fd` has a
different denominator; consistent mode recovers the appropriate share from the
calibration table (or reconstructs it from baseline budget coefficients when
the table is absent).

Production combines calibrated input requirements (Leontief or CES sourcing)
with Cobb–Douglas value added. The unit value-added cost is

```
v_j = (r_c / α_j)^α_j (w_c / (1 − α_j))^(1 − α_j) / β_j
(1 − t_j) p_j = v_j + intermediate purchaser cost per unit
production-tax receipt = t_j p_j Y_j
L_j = (1 − α_j) v_j Y_j / w_c
K_j = α_j v_j Y_j / r_c
```

These equations are homogeneous in nominal prices. A charge calibrated as a
share of output **revenue** must collect t_j p_j Y_j, rather than t_j Y_j.
The original raw OECD TLS row aggregates product taxes and subsidies; modeling
its production-column amount as this output-revenue charge is an explicit
calibration assumption. The source does not identify a full bilateral product
or VAT tax schedule. This mode supplies a coherent reduced model, not a claim
that those tax institutions have been independently identified.

These distinctions follow standard price and fiscal accounting principles:
[purchaser versus basic-price valuation in UN national accounts](https://digitallibrary.un.org/record/525790/files/National_accounts_introduction.pdf)
and [production/duty receipts and a numeraire in the GAMS standard CGE example](https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_stdcge.html).
Those sources motivate the conventions; the quantitative reference here is the
separate two-country derivation below, not a replication of that GAMS model.

## Budgets, foreign saving and the numeraire

All domestic taxes and import duties enter government receipts G_c. In the
supported lump-sum closure, the solved transfer T_c equals G_c. Household income
is I_c = w_c L_c + r_c K_c + T_c. Calibrated expenditure shares θ allocate it,
with foreign saving B_c deducted from investment:

```
E_kc = θ_kc I_c − 1[k = investment] B_c
Σ_k E_kc + B_c = I_c
B_c = baseline foreign saving, fixed in numeraire units
Σ_c B_c = 0
```

Investment is category index 1, or index 0 for a single-category model. Native
OECD ingestion first maps the six final uses to C/I/Cx. Final expenditure must
be nonnegative in an accepted solution. The consistent mode rejects any
negative final-use cell. On the native OECD 2019 table, I = GFCF + INVNT has 90
negative cells although every country's I total is positive; the count per
category is recorded in `calib.metadata["negative_final_demand_cells"]`. Such a
table needs aggregation or an explicit inventory model first. The FIGARO,
EXIOBASE, WIOD and Eora loaders condense their final uses to C (final
consumption), I (gross fixed capital formation) and Cx (changes in inventories
and valuables, and EXIOBASE exports); unlike the OECD mapping, their Cx is not
residents' purchases abroad. The mapping is in `metadata["final_use_mapping"]`.

Fixing foreign saving is a substantive closure choice. Allowing all foreign
balances to be endogenous while imposing national budgets leaves independent
cross-country transfer choices unspecified. The coherent mode replaces that
redundancy with baseline foreign balances. The first country/sector's producer
price is fixed at one. Its redundant goods-clearing equation is omitted from
the numerical square system, then **checked independently before convergence
can be reported**. Realized exports minus imports must also recover B_c in every
country, including the last one.

Because B_c is fixed in numeraire units by default
(`foreign_saving_units="numeraire"`), the equilibrium is not homogeneous of
degree zero in nominal prices when foreign saving is nonzero: entering the same
table with a different country listed first changes the real allocation and
Hicksian EV/CV (on the two-country welfare reference table, country A's EV is
-27.55% of baseline consumption with A first and -45.99% with B first). Pass
`foreign_saving_units="world_income"` to hold each country's baseline saving
fixed as a share of world factor income,
`B_c = (B_c^0 / Σ_d (L_d + K_d)) Σ_d (w_d L_d + r_d K_d)`; the system is then
homogeneous of degree zero and the results do not depend on the country order
(the same example gives -36.95% for either order). The default is unchanged.

GDP at factor cost equals factor income. GDP at market prices adds domestic
taxes and duties; it must equal final purchaser expenditure plus net exports
valued at producer/border prices. Household and government budgets and all
physical equilibrium equations are exposed in `metadata["account_residuals"]`.
Solver success alone cannot certify convergence when these physical equations
fail or demand is infeasible.

## Reporting and solver scope

`p_sol` is the actual solved producer price. `p_fd` is Q; `Pfd_final` is P.
`c_fd` is the actual composite quantity after foreign-saving allocation. Bilateral
trade, imports and exports use producer values. Metadata exposes separate
producer and purchaser value matrices, domestic-tax receipts and duties.

`data_model_vf` contains producer transaction values, local tax receipts and
nominal factor payments. `data_tariff_vf` additionally includes import duties:
production columns sum to producer output revenue, and final-use columns sum
to purchaser expenditure. These tables contain nominal values throughout.

The reported `cpi` is a fixed-basket purchaser-price Laspeyres index divided by
the wage relative to the baseline. `metadata["nominal_cpi"]` preserves the
undeflated index. A supplied baseline must also use consistent accounting.
This is an index, not Hicksian welfare or an expenditure-function derivation.

The mode supports NumPy, Leontief/CES intermediate sourcing and lump-sum rebates.
Newton, sparse LU, Krylov, Broyden, SciPy hybrid/LM and full Keller continuation
all solve the same equations. The Newton, sparse-LU, Krylov, Broyden and Keller
solvers stop only when the solved residuals and the independently checked
physical equations (the omitted goods equation and realized foreign balances)
both meet `tol`, so a state the solver accepts is not rejected afterwards at the
same tolerance; their finite-difference Jacobians scale the step of the
value-unit transfer and foreign-balance unknowns with their magnitude, so the
solve does not depend on the currency unit. SciPy hybrid/LM use SciPy's own
termination test and are audited afterwards. A request for
`method="condensed"` uses full Newton and records `effective_method="newton"`; the legacy Schur reduction
embodies different equations. Quasi-condensed, other fiscal recycling schemes,
GPU execution and capacity penalties explicitly raise `NotImplementedError` in
this mode pending their own economic derivations. Legacy mode retains those
existing interfaces without acquiring a new validation claim.

## Independent two-country check

[The reference script](https://github.com/jalonso1979/puremacro/blob/v4.3.0/tools/reference_validation/validate_trade_accounting.py)
uses a hand-balanced transaction table with two countries, one producing sector,
two final uses, nonzero foreign saving, local taxes and heterogeneous tariffs.
It derives the reference directly from the table, without calling puremacro's
calibration, residual or GE routines inside the oracle.

With one sector per country, factor clearing fixes outputs. For each candidate
relative producer price, intermediary quantities are fixed by Leontief inputs.
Let h_kc be the share of final purchaser expenditure returned as local tax plus
import duty. The oracle solves income algebraically:

```
I_c = [p_c Y_c − Σ_i p_i Z_ic − h_investment,c B_c]
      / [1 − Σ_k h_kc θ_kc]
```

It then clears one goods market with a scalar Brent root. The other goods market
is checked separately. This provides independent prices, factor payments,
deliveries and fiscal receipts at baseline and under a nonuniform tariff shock.
The tests also check nominal homogeneity, the Jacobian's full rank after closure,
all budget ledgers, currency scaling and the conserving OECD aggregation.

[Results and reproduction commands](https://github.com/jalonso1979/puremacro/blob/v4.3.0/reviews/2026-09-20-trade-accounting/REPORT.md)
record the achieved errors and remaining limitations. The subsequent
[Hicksian consumption-welfare interface](trade_welfare.md) provides derived EV/CV
and an endpoint attribution. Historical TOT/efficiency decomposition and theorem
certification remain unavailable.
