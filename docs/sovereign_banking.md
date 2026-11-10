# Sovereign risk and domestic banks

[Español](es/sovereign_banking.md)

`puremacro.sovereign.banking` solves a small three-date contract economy.
Government default reduces domestic bank equity before banks supply private
credit. Foreign investors price the sovereign's repayment decisions, and the
government chooses debt on a declared finite grid. All consistent **pure**
price/default equilibria are retained; the caller explicitly selects the
highest or lowest price at each debt level.

This is a separate experimental model. It does not add banking to the recursive
Arellano solver. Domestic bond exposure is mandated, household equity is locked,
and households cannot save or borrow directly. These restrictions are part of
the model, not estimated behavior. Banks can obtain safe foreign funding after
the sovereign decision, subject to a leverage limit. There is no recapitalization,
selective default, bank run, recurring issuance, or mixed-equilibrium solver.

## Solve and inspect

```python
import numpy as np
from puremacro.sovereign import (
    SovereignBankingParameters, solve_sovereign_banking,
    audit_sovereign_banking,
)

result = solve_sovereign_banking(
    income=[0.7, 1.0, 1.3],
    productivity=[0.6, 0.9, 1.2],
    probabilities=[0.25, 0.50, 0.25],
    debt_grid=np.linspace(0.0, 0.6, 51),
    parameters=SovereignBankingParameters(domestic_share=0.4),
    price_selection="highest",
)
assert audit_sovereign_banking(result)["accepted"]
print(result.equilibria_frame().query("selected"))
print(result.to_frame()[["bank_equity", "credit", "consumption1", "consumption2"]])
print(result.debt_frame().query("multiple"))
assert len(result.candidates_frame()) == 51 * 2**3
```

`equilibria_frame()` retains every consistent price, its default probability,
household utility and expected credit. `debt_frame()` records multiplicity,
missing pure equilibria and the selected equilibrium index (`-1` if absent).
`candidates_frame()` retains all rejected masks and their reasons.
`to_frame()` reports realized accounts at the selected debt/price, one row per
joint income/productivity state. Tables are fresh copies; stored arrays are
owned and read-only. Markdown, LaTeX and Typst reports are supported.

`counterfactual_frame()` forces repayment and default **at the same contracted
debt and price**. These are conditional accounting comparisons, not newly priced
equilibria. Infeasible repayment keeps its nonpositive consumption and has
`feasible=False` with missing log utility. Selected allocations always have
positive consumption and finite utility. `sovereign_bank_accounts()` exposes
the same conditional calculation for a user-specified contract.

## Dates, ownership and financing

All stocks and flows use one good. (R>0) is a gross safe return per date;
(0<\beta\leq1) discounts utility. Households own banks and firms. Joint states
((Y,a)) have positive probabilities and are observed at date 1. Productivity
(a) determines date-2 project output and is known before lending.

At date 0, income is (Y_0) and fixed bank equity is (E_0<Y_0). The government
issues face value (B\geq0) at price (q), transfers all proceeds to households,
and requires domestic banks to hold fraction (0\leq s<1) of the debt. Banks
pay (qsB) and save the rest in a safe foreign asset:

\[
A_0=E_0-qsB\geq0,\qquad C_0=Y_0-E_0+qB.
\]

Foreign sovereign lenders buy the remaining fraction. The restriction (s<1)
keeps an external marginal lender to determine the common price. Banks cannot
borrow at date 0 or obtain extra household equity at date 1.

At date 1 the government chooses full default (d=1) or repayment (d=0).
Repayment taxes the household by ((1-d)B). The separate exogenous default
penalty destroys fraction \(\phi\) of current income:

\[
C_1=(1-\phi d)Y-(1-d)B,\qquad
E_1=RA_0+(1-d)sB.
\]

Bank equity is measured **after** settlement. At the same initial contract,
default reduces it by exactly (sB). This loss is not subtracted from income
again: its resource consequences arise through bank lending and future returns.
The exogenous penalty \(\phi dY\) is recorded separately and can be set to zero.

Banks fund competitive projects with production (F(L)=a\log(1+L/k)), where
(k>0) is a credit scale. They can raise foreign funding (D) at return (R),
but loans cannot exceed \(\lambda E_1\), with \(\lambda\geq1\). The credit
market allocation is

\[
L=\min\{\max(a/R-k,0),\lambda E_1\},\quad
D=\max(L-E_1,0),\quad S=\max(E_1-L,0).
\]

The gross loan rate is \(r_L=\max\{R,a/(k+L)\}\). When there is no loan demand,
the quoted rate is (R). When equity is zero and profitable projects exist,
credit is zero and the quoted choke rate is (a/k). Funding (D) is a foreign
bank liability, **not a household deposit or a government guarantee**. Project
returns are known when loans are made and cover this liability. (S) is the
bank's safe asset. The balance sheet is (L+S=E_1+D).

At date 2 banks liquidate. Firm profit is (F-r_LL), bank dividends are
(r_LL+RS-RD), and household terminal income is (Y_2>0):

\[
C_2=Y_2+F(L)+RS-RD.
\]

The sovereign chooses the date-1 decision to maximize
\(\log C_1+\beta\log C_2\). At date 0 it maximizes
\(\log C_0+\beta\mathbb E[\log C_1+\beta\log C_2]\).
These utility values compare this model's specified consumption allocations;
they are not empirical welfare estimates.

## Pricing, multiplicity and numerical scope

For every debt point and each of the (2^n) possible state default masks, set
\(q=\sum_j p_j(1-d_j)/R\). Recompute both sovereign actions at that price and
retain the mask only if decisions agree and (A_0\geq0). Price feeds back into
bank reserves, so more than one consistent price can exist. All-default
equilibria with (q=0) remain visible; debt at zero price raises no resources.
Missing pure equilibria are not replaced by interpolated or mixed solutions.

`price_selection="highest"` or `"lowest"` selects one price at each available
debt point. The government then chooses the best available grid point. This
selection is a maintained assumption, not an economic uniqueness result.
Repayment wins gains from default no larger than `decision_tolerance` (default
`1e-12`, allowed range `[0,1e-8]` in log-utility units). Exact date-0 utility ties
choose the lower debt. Audits retain maximum decision regret and near ties.
Upper-grid-boundary contact is reported; a finite grid is not a continuous
debt optimum or an infinite-horizon equilibrium.

Inputs are finite real vectors. Debt starts at exact zero and increases
strictly. Probabilities must be positive and sum to one within `1e-12`; sums
within that roundoff tolerance are normalized. Enumeration is bounded to 12
states and 200,000 debt/mask pairs. Invalid portfolios are rejected; numerical
overflow raises an error rather than returning clipped outcomes.

## Accounts and reference limits

The independent scalar auditor reconstructs every price mask and state choice,
all allocations, the declared price selection and discrete debt optimality.
It also checks the bank, government and household budgets, credit constraints
and complementarity, plus the three resource identities:

\[
\begin{aligned}
C_0+A_0 &= Y_0+(1-s)qB,\\
C_1+L+S &= Y-\phi dY+RA_0-(1-s)(1-d)B+D,\\
C_2+RD &= Y_2+F(L)+RS.
\end{aligned}
\]

The default audit tolerance is `1e-9`; numeric comparisons divide errors by
`1 + abs(reference)`. This is an equation/finite-grid check, not a bound on
economic approximation error. Independent tests optimize credit surplus
numerically and recover the complete equilibrium set on small fixtures.

At **zero domestic exposure**, bank equity is (RE_0) under either action and
default occurs iff (B>\phi Y), subject to the declared tie tolerance. This
recovers the corresponding three-date baseline, not the recursive Arellano
model. With both lending constraints binding, conditional default cuts credit
by \(\lambda sB\). With both allocations unconstrained, credit can be unchanged.
These separate references prevent confusing sovereign losses with credit losses.

## Reproduce the illustration

```sh
python -m puremacro.examples.sovereign_banking --output banking-en --lang en
python -m puremacro.examples.sovereign_banking --output banking-es --lang es
```

The fixed synthetic three-state example compares exposures (s=0,.2,.4,.6),
both price selections, and 51/101/201-node debt grids on `[0,.6]`. It saves all
candidates, equilibria, accounts, exact arrays, fingerprints, audits and grid
comparisons. A separate fixed-contract stress panel holds `B=.25, q=.8`;
it is explicitly not an equilibrium repricing exercise. Failures of the stated
refinement limits are retained in the output. The state distribution is fixed:
these checks refine the debt grid only.

The balance-sheet mechanism is motivated by [Gennaioli, Martin and Rossi,
Sovereign Default, Domestic Banks, and Financial Institutions](https://crei.cat/wp-content/uploads/users/working-papers/sovereign_default.pdf).
The timing, technology, mandated portfolios and selection rules above define
this implementation; it does not replicate that paper's optimal portfolios.
