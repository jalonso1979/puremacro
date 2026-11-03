"""Conditional dynamic CES sourcing with convex adjustment costs.

A buyer supplies one unit of a CES composite each date at exogenous delivered
origin prices. It chooses input quantities q[t,o], paying their purchase cost
and kappa/2 * sum((q[t]-q[t-1])**2 / alpha). CES output must be at least one;
free disposal is allowed. Baseline prices are one and q[-1]=alpha supplies
one unit. This is a convex, finite-horizon cost problem, not a dynamic GE
model or a labor-reallocation model. There is no cost after the last date.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

__all__ = ["SourcingAdjustmentResult", "solve_sourcing_adjustment"]


@dataclass(frozen=True)
class SourcingAdjustmentResult:
    quantities: pd.DataFrame
    accounts: pd.DataFrame
    diagnostics: dict
    metadata: dict

    def to_frame(self) -> pd.DataFrame:
        return self.accounts.copy(deep=True)


def _production(q, alpha, sigma):
    """Log CES output and its gradient, including the Cobb-Douglas limit."""
    ratio = q / alpha
    if sigma == 1:
        log_output = np.log(ratio) @ alpha
        grad = np.broadcast_to(alpha, q.shape) / q
    else:
        rho = (sigma - 1) / sigma
        log_terms = np.log(alpha)[None, :] + rho * np.log(ratio)
        normalizer = (np.log1p(np.sum(alpha * np.expm1(rho * np.log(ratio)), axis=1))
                      if abs(rho) < 1e-5 else logsumexp(log_terms, axis=1))
        log_output = normalizer / rho
        grad = np.exp(log_terms - normalizer[:, None]) / q
    return log_output, grad


def solve_sourcing_adjustment(
    prices: pd.DataFrame,
    baseline_shares: pd.Series,
    *,
    elasticity: float,
    adjustment_cost: float,
    discount: float = .99,
    initial_quantities: pd.Series | None = None,
    tolerance: float = 1e-6,
    max_iter: int = 2000,
) -> SourcingAdjustmentResult:
    """Minimize discounted procurement and quadratic switching resources.

    Prices are date-by-origin delivered *relative* prices, on a common
    benchmark where each origin price is one. Baseline shares are strictly
    positive and sum to one; remove unsupported zero-share origins explicitly.
    ``elasticity`` is the CES substitution elasticity (>0). Costs are in
    baseline-composite units per date; adjustment resources must be counted
    separately from purchases. Date labels must be unique and increasing.

    Positive quantities plus a concave CES production constraint define a
    convex feasible set. Numerical acceptance additionally requires primal
    feasibility and independently reconstructed KKT stationarity. The finite
    horizon is the entire decision problem, with no assumed terminal tail.
    Consumers may use total unit resource costs as prices only under an
    explicit full-cost-pass-through assumption; no GE closure is supplied.
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a nonempty date-by-origin DataFrame")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("dates must be unique and increasing")
    if not prices.columns.is_unique or any(not isinstance(x, str) or not x.strip() for x in prices.columns):
        raise ValueError("origins require unique nonempty string labels")
    if not isinstance(baseline_shares, pd.Series) or not baseline_shares.index.is_unique or set(baseline_shares.index) != set(prices.columns):
        raise ValueError("baseline_shares must have exactly the origin labels")
    if np.iscomplexobj(prices.to_numpy()) or np.iscomplexobj(baseline_shares.to_numpy()):
        raise ValueError("prices and shares must be real")
    p = prices.to_numpy(dtype=float)
    alpha = baseline_shares.reindex(prices.columns).to_numpy(dtype=float)
    if not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("prices must be finite and positive")
    if not np.isfinite(alpha).all() or np.any(alpha <= 0) or not np.isclose(alpha.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("baseline_shares must be positive and sum to one")
    if not np.isfinite(elasticity) or elasticity <= 0:
        raise ValueError("elasticity must be finite and positive")
    if not np.isfinite(adjustment_cost) or adjustment_cost < 0:
        raise ValueError("adjustment_cost must be finite and nonnegative")
    if not np.isfinite(discount) or not 0 < discount <= 1:
        raise ValueError("discount must lie in (0, 1]")
    if not np.isfinite(tolerance) or tolerance <= 0 or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("tolerance and integer max_iter must be positive")
    initial = alpha.copy()
    if initial_quantities is not None:
        if not isinstance(initial_quantities, pd.Series) or not initial_quantities.index.is_unique or set(initial_quantities.index) != set(prices.columns):
            raise ValueError("initial_quantities must have exactly the origin labels")
        if np.iscomplexobj(initial_quantities.to_numpy()):
            raise ValueError("initial_quantities must be real")
        initial = initial_quantities.reindex(prices.columns).to_numpy(dtype=float)
        if not np.isfinite(initial).all() or np.any(initial <= 0):
            raise ValueError("initial_quantities must be positive and finite")
    T, n = p.shape
    weights = discount ** np.arange(T)
    if weights[-1] <= 1e-12:
        raise ValueError("discounted horizon is numerically unresolved; shorten horizon or change discount")
    if elasticity == 1:
        frictionless_index = np.exp(np.log(p) @ alpha)
    else:
        power = 1 - elasticity
        log_index = (np.log1p(np.sum(alpha * np.expm1(power * np.log(p)), axis=1)) / power
                     if abs(power) < 1e-5 else
                     logsumexp(np.log(alpha)[None, :] + power * np.log(p), axis=1) / power)
        frictionless_index = np.exp(log_index)
    frictionless = alpha * (frictionless_index[:, None] / p) ** elasticity
    kappa = float(adjustment_cost)

    def objective(x):
        q = x.reshape(T, n)
        change = q - np.vstack([initial, q[:-1]])
        costs = np.sum(p * q, axis=1) + kappa / 2 * np.sum(change**2 / alpha, axis=1)
        gradient = weights[:, None] * (p + kappa * change / alpha)
        gradient[:-1] -= weights[1:, None] * kappa * change[1:] / alpha
        return float(weights @ costs), gradient.ravel()

    def constraint(x):
        return _production(x.reshape(T, n), alpha, elasticity)[0]

    def constraint_jac(x):
        derivative = _production(x.reshape(T, n), alpha, elasticity)[1]
        jac = np.zeros((T, T * n))
        for t in range(T):
            jac[t, t*n:(t+1)*n] = derivative[t]
        return jac

    floor = 1e-12 * np.tile(alpha, T)
    if kappa == 0:
        q = frictionless
        success, iterations, message = True, 0, "analytical frictionless CES solution"
    else:
        # The baseline is feasible and avoids a costly discontinuous start.
        start = np.broadcast_to(alpha, (T, n)).copy().ravel()
        result = minimize(objective, start, jac=True, method="SLSQP",
                          bounds=list(zip(floor, np.full(T*n, np.inf))),
                          constraints=[{"type": "ineq", "fun": constraint, "jac": constraint_jac}],
                          options={"ftol": min(tolerance**2, 1e-12), "maxiter": max_iter})
        q = result.x.reshape(T, n)
        success, iterations, message = bool(result.success), int(result.nit), str(result.message)
    log_output, production_grad = _production(q, alpha, elasticity)
    value, gradient_flat = objective(q.ravel())
    gradient = gradient_flat.reshape(T, n) / weights[:, None]
    multipliers = np.maximum(0, np.sum(gradient * production_grad, axis=1) /
                             np.sum(production_grad**2, axis=1))
    stationarity = gradient - multipliers[:, None] * production_grad
    kkt = float(np.max(np.abs(stationarity) / np.maximum(1., np.abs(gradient))))
    feasibility = float(max(0., -np.min(log_output)))
    complementarity = float(np.max(np.abs(multipliers * log_output)))
    near_floor = bool(np.any(q.ravel() <= 10 * floor))
    accepted = success and max(kkt, feasibility, complementarity) <= tolerance and not near_floor
    if not accepted:
        raise RuntimeError(f"sourcing adjustment not accepted: {message}; KKT={kkt:.3g}, "
                           f"feasibility={feasibility:.3g}, complementarity={complementarity:.3g}, floor={near_floor}")
    changes = q - np.vstack([initial, q[:-1]])
    procurement = np.sum(p * q, axis=1)
    adjustment = kappa / 2 * np.sum(changes**2 / alpha, axis=1)
    accounts = pd.DataFrame({"procurement_cost": procurement, "adjustment_resources": adjustment,
                            "total_unit_cost": procurement + adjustment,
                            "frictionless_unit_cost": frictionless_index,
                            "output": np.exp(log_output), "discount_weight": weights}, index=prices.index.copy())
    return SourcingAdjustmentResult(
        pd.DataFrame(q, index=prices.index.copy(), columns=prices.columns.copy()), accounts,
        {"accepted": accepted, "iterations": iterations, "kkt_residual": kkt,
         "production_violation": feasibility, "complementarity_residual": complementarity,
         "discounted_total_cost": value, "tolerance": tolerance},
        {"elasticity": float(elasticity), "adjustment_cost": kappa, "discount": float(discount),
         "baseline_shares": baseline_shares.reindex(prices.columns).to_dict(),
         "initial_quantities": dict(zip(prices.columns, initial.tolist())),
         "prices": prices.to_dict(orient="list"),
         "scope": "conditional fixed-output sourcing; exogenous delivered prices; free disposal; no GE or labor mobility",
         "terminal_condition": "no production requirement or adjustment cost after the last date",
         "units": "benchmark unit-composite expenditure per period"},
    )
