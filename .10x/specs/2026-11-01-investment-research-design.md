# Investment research milestones

Approved scope: the user accepted the five-stage proposal and delegated sequencing and implementation on 2026-11-01. Source baseline: main 0b3caf3e. Work is isolated on feat/investment-research.

| Milestone | Deliverable | Acceptance evidence | Status |
|---|---|---|---|
| M1 Historical evaluation | Availability-dated observations; lagged target execution; long-only holdings, cash, costs and drift; reproducible ledger | Hand-calculated accounting, future-data perturbation, invalid/missing input tests | In progress |
| M2 Allocation | Equal weights; Ledoit-Wolf covariance; bounded minimum-variance/mean-variance; turnover penalty; frontier | Closed-form solutions, constrained independent reference, infeasibility and scale tests | Pending |
| M3 Technical research | Momentum, moving averages, breakouts, Wilder RSI, MACD and Bollinger; rolling historical policies; block-bootstrap strategy comparison | Published conventions, scalar references, causality, selection-aware bootstrap controls | Pending |
| M4 Fundamental analysis | As-of filing selection with retained revisions; financial ratios; FCFF/FCFE/dividend valuation; sensitivity scenarios | Filing-lag controls, independent present values, enterprise/equity consistency | Pending |
| M5 Belief dynamics | Brock-Hommes linear forecasting rules, realized-profit selection and memory; seeded deviations and fractions | Independent recurrence, fundamental equilibrium, zero-intensity solution, timing and overflow controls | Pending |
| M6 Integration and delivery | English/Spanish guides and executed notebooks; API snapshot, type/import checks; evidence report and commits | Focused numerical suite, example execution, release-contract gates; broader test outcomes recorded explicitly | Pending |

## Implementation contract

All calculations use the existing NumPy/SciPy/pandas core. New APIs live in puremacro.asset_pricing, retain existing APIs, and are experimental additions to unreleased 6.1.0. Results use the existing report mixin. Prices and returns are supplied by the caller; prices must already reflect the chosen corporate-action convention. Securities are long-only and unlevered in the first backtester. Cash is explicit, trades pay proportional costs, and post-cost target weights obey a self-financing equation. No fabricated fills, delisting returns, missing observations or silent regularization. A missing held return is an error. Universe membership is supplied as dated eligibility, never inferred from future availability. Evaluation does not claim causal identification or investment profitability.

A policy sees only past observations; an externally supplied target matrix is checked for dates/lag but cannot be certified free of user-created look-ahead. Supplied fundamentals contain period end and availability time; revisions never overwrite the historical information set. Network ingestion is a separate adapter concern; the first milestone set works offline on explicit caller inputs.

BH scope: zero outside supply, constant expected dividend and risk-free gross return, common a*sigma^2, affine deviation forecasts, lagged realized-profit fitness and multinomial-logit switching. Explicit shocks are additive deviation-price disturbances. No claim of global stability, empirical calibration, perfect-foresight agents, or full replication of every published bifurcation diagram.

## Deferred extensions

Broker execution/intraday order books, leverage/short borrowing, tax lots, automatic corporate-action or delisting data repair, paid survivor-free market feeds, live SEC normalization across taxonomies, Black-Litterman/risk parity, and empirically estimated belief dynamics remain separately scoped follow-on work. The first release includes the complete usable kernels for each approved milestone, with boundaries documented.
