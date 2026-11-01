# ADR 001: Portable investment research with explicit information timing

Status: Accepted under the user's implementation authorization. Date: 2026-11-01.

Context: existing risk and pricing tests need allocations, signals, valuation and behavioral dynamics. The numerical core must remain portable and deterministic.

Decision: add composable functions under asset_pricing. Returns at t cover the interval ending t. Decisions after close t can affect only returns after the chosen execution lag (default two rows for close-to-close data: execute next close, earn the following interval). Costs are paid at the beginning of the earned interval; target weights are fractions of wealth after costs. Solve q + c*sum(abs(q*w - w_pre)) = 1, where q is post-cost/pre-cost wealth. Cash is the residual. Carry holdings and cash through the period, then derive drifted weights.

Alternatives: external strategy/solver frameworks offer breadth but break the portable dependency contract; target weights multiplied by contemporaneous returns are simpler but permit look-ahead and omit holdings drift; precomputed targets remain supported with explicit provenance limitations.

Consequences: precise timing and accounting, portable kernels and directly testable limits. No order book, borrow financing or automatic data-quality repair. Nonlinear optimizers must report feasibility/optimality diagnostics and reject failure. Belief simulation and empirical backtesting remain separate models.
