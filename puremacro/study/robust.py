"""Policy comparisons across an explicitly enumerated uncertainty set.

Payoffs must share a welfare definition, population, horizon and numeraire.
States are assumptions, not posterior draws unless the caller establishes
that interpretation. No probabilities are inferred from grid density.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

__all__ = ["RobustPolicyComparison", "compare_policy_payoffs"]


@dataclass(frozen=True)
class RobustPolicyComparison:
    payoffs: pd.DataFrame
    regrets: pd.DataFrame
    summary: pd.DataFrame
    dominance: pd.DataFrame
    metadata: dict

    def to_frame(self) -> pd.DataFrame:
        return self.summary.copy(deep=True)


def compare_policy_payoffs(
    payoffs: pd.DataFrame,
    *,
    objective: str,
    units: str,
    comparability: Mapping[str, str],
    probabilities: pd.Series | None = None,
    maximize: bool = True,
    statuses: pd.DataFrame | None = None,
    tolerance: float = 1e-10,
) -> RobustPolicyComparison:
    """Dominance, worst case and minimax regret on a complete payoff matrix.

    Rows are policies; columns are uncertainty states. All entries must be
    finite, and supplied statuses must all be ``passed``. A missing/failed
    cell prevents ranking the entire matrix, since dropping it could change
    the ex-post best policy and every other policy's regret.

    ``comparability`` must explicitly name population, horizon, numeraire and
    welfare_definition. Matching is the caller's responsibility: descriptive
    metadata is not proof of economic comparability. Expected payoffs appear
    only with nonnegative, exactly labeled probabilities summing to one.
    Dominance means no worse in all states and strictly better in at least
    one, using the absolute ``tolerance`` in the declared payoff units.
    """
    if not isinstance(payoffs, pd.DataFrame) or payoffs.empty:
        raise ValueError("payoffs must be a nonempty policy-by-state DataFrame")
    for axis, labels in (("policies", payoffs.index), ("states", payoffs.columns)):
        if not labels.is_unique or any(not isinstance(x, str) or not x.strip() for x in labels):
            raise ValueError(f"{axis} require unique nonempty string labels")
    for name, value in (("objective", objective), ("units", units)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be explicitly specified")
    required = ("population", "horizon", "numeraire", "welfare_definition")
    if not isinstance(comparability, Mapping) or any(
        not isinstance(comparability.get(k), str) or not comparability[k].strip() for k in required
    ):
        raise ValueError(f"comparability must specify {required}")
    if not isinstance(maximize, (bool, np.bool_)):
        raise ValueError("maximize must be boolean")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    raw = payoffs.to_numpy()
    if np.iscomplexobj(raw):
        raise ValueError("payoffs must be real")
    values = payoffs.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("every policy-state payoff must be finite; failed cells cannot be dropped")
    if statuses is not None:
        if not isinstance(statuses, pd.DataFrame) or not statuses.index.equals(payoffs.index) or not statuses.columns.equals(payoffs.columns):
            raise ValueError("statuses must have the exact payoff label order")
        if not (statuses.to_numpy() == "passed").all():
            raise ValueError("every policy-state status must be passed before ranking")
    oriented = values if maximize else -values
    regret = oriented.max(axis=0)[None, :] - oriented
    if not np.isfinite(regret).all():
        raise ValueError("regret overflowed; rescale payoff units")
    dominates = np.zeros((len(values), len(values)), dtype=bool)
    for i in range(len(values)):
        for j in range(len(values)):
            delta = oriented[i] - oriented[j]
            dominates[i, j] = bool(np.all(delta >= -tolerance) and np.any(delta > tolerance))
    worst = oriented.min(axis=1)
    maxregret = regret.max(axis=1)
    summary = pd.DataFrame({
        "worst_payoff": values.min(axis=1) if maximize else values.max(axis=1),
        "best_payoff": values.max(axis=1) if maximize else values.min(axis=1),
        "maximum_regret": maxregret,
        "undominated": ~dominates.any(axis=0),
        "maximin": worst >= worst.max() - tolerance,
        "minimax_regret": maxregret <= maxregret.min() + tolerance,
    }, index=payoffs.index.copy())
    probability_record = None
    if probabilities is not None:
        if not isinstance(probabilities, pd.Series) or not probabilities.index.is_unique or set(probabilities.index) != set(payoffs.columns):
            raise ValueError("probabilities must have exactly the state labels")
        if np.iscomplexobj(probabilities.to_numpy()):
            raise ValueError("probabilities must be real")
        p = probabilities.reindex(payoffs.columns).to_numpy(dtype=float)
        if not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12):
            raise ValueError("probabilities must be finite, nonnegative and sum to one")
        summary["expected_payoff"] = values @ p
        summary["expected_regret"] = regret @ p
        probability_record = dict(zip(payoffs.columns, p.tolist()))
    return RobustPolicyComparison(
        payoffs.astype(float).copy(deep=True),
        pd.DataFrame(regret, index=payoffs.index.copy(), columns=payoffs.columns.copy()),
        summary, pd.DataFrame(dominates, index=payoffs.index.copy(), columns=payoffs.index.copy()),
        {"objective": objective, "units": units, "maximize": bool(maximize),
         "comparability": dict(comparability), "probabilities": probability_record,
         "tolerance": float(tolerance), "scope": "enumerated states only; no statistical coverage claim",
         "dominance_orientation": "row dominates column"},
    )
