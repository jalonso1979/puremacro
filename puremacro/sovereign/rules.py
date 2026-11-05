"""Fiscal rules for the sovereign fiscal block.

A :class:`FiscalRule` restricts next period's debt ``b'`` in repayment:

- ``debt_ceiling``: ``b' <= debt_ceiling * mean_y``;
- ``deficit_limit``: ``b' - b <= deficit_limit * mean_y`` (new net borrowing);
- ``escape_below``: every component is suspended at incomes
  ``y < escape_below * mean_y``;
- ``penalty``: if set, every component is soft: each violated component costs
  ``penalty`` utils instead of being forbidden (violations add up).

Thresholds are in units of mean output per model period (a quarter). A choice
counts as compliant up to an absolute tolerance of ``1e-12``, so a grid node
at the cap up to floating point is allowed.

Lenders know the rule: it enters the government's problem and therefore the
equilibrium bond-price schedule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np

__all__ = ["FiscalRule"]

_RULE_TOL = 1e-12


@dataclass(frozen=True)
class FiscalRule:
    """A debt ceiling and/or deficit limit, optionally soft and with an escape clause.

    Parameters
    ----------
    debt_ceiling : float, optional
        Cap on next debt, in mean output per quarter.
    escape_below : float, optional
        Income threshold (mean-output units): the rule is suspended when
        ``y < escape_below``.
    penalty : float, optional
        Utility cost per violated component; makes the rule soft. Must be > 0.
    deficit_limit : float, optional
        Cap on new net borrowing ``b' - b``, in mean output per quarter.
        Negative values require debt reduction. Needs
        ``policy_search="exhaustive"`` or ``"dense"``.
    infeasible : {"raise", "default"}, default "raise"
        What to do in states where a hard rule leaves no candidate:
        raise ``ValueError`` before solving, or force default there.
    """

    debt_ceiling: float | None = None
    escape_below: float | None = None
    penalty: float | None = None
    deficit_limit: float | None = None
    infeasible: str = "raise"

    def __post_init__(self) -> None:
        if self.debt_ceiling is None and self.deficit_limit is None:
            raise ValueError("A FiscalRule needs a debt_ceiling or deficit_limit")
        for name in ("debt_ceiling", "escape_below", "penalty", "deficit_limit"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite; got {value}")
        if self.penalty is not None and self.penalty <= 0.0:
            raise ValueError(f"penalty must be > 0; got {self.penalty}")
        if self.infeasible not in ("raise", "default"):
            raise ValueError(f'infeasible must be "raise" or "default"; got {self.infeasible!r}')


class _CompiledRule(NamedTuple):
    """A rule compiled onto a grid.

    ``column_penalty[m, j]`` is the cost of choosing node ``m`` at income ``j``
    (ceiling); ``transition_penalty(first, last, j)`` returns the
    ``(last - first, n_b)`` cost block for debt rows ``first:last`` (deficit
    limit). Hard costs are ``inf``. ``infeasible[i, j]`` marks states where a
    hard rule leaves no candidate.
    """

    column_penalty: np.ndarray | None
    transition_penalty: Callable[[int, int, int], np.ndarray] | None
    infeasible: np.ndarray
    rule: FiscalRule
    b_grid: np.ndarray
    active: np.ndarray
    limit: float | None
    cost: float

    def chosen_penalty(self, m_opt: np.ndarray) -> np.ndarray:
        """Total rule cost of the chosen nodes ``m_opt`` (shape ``(n_b, n_y)``)."""
        total = np.zeros(m_opt.shape)
        if self.column_penalty is not None:
            total = total + np.take_along_axis(self.column_penalty, m_opt, axis=0)
        if self.limit is not None:
            violation = (self.b_grid[m_opt] - self.b_grid[:, None] > self.limit + _RULE_TOL) & self.active[None, :]
            total = total + np.where(violation, self.cost, 0.0)
        return total


def _compile_rule(rule: FiscalRule, b_grid: np.ndarray, y_grid: np.ndarray, mean_y: float) -> _CompiledRule:
    """Compile ``rule`` onto the debt and income grids (thresholds scaled by ``mean_y``)."""
    n_b, n_y = b_grid.shape[0], y_grid.shape[0]
    active = np.ones(n_y, dtype=bool) if rule.escape_below is None else ~(y_grid < rule.escape_below * mean_y)
    soft = rule.penalty is not None
    cost = float("inf") if rule.penalty is None else float(rule.penalty)
    cap = None if rule.debt_ceiling is None else rule.debt_ceiling * mean_y
    limit = None if rule.deficit_limit is None else rule.deficit_limit * mean_y

    column_penalty = None
    if cap is not None:
        violation = (b_grid[:, None] > cap + _RULE_TOL) & active[None, :]
        column_penalty = np.where(violation, cost, 0.0)

    transition_penalty = None
    if limit is not None:
        def transition_penalty(first: int, last: int, j: int) -> np.ndarray:
            if not active[j]:
                return np.zeros((last - first, n_b))
            violation = b_grid[None, :] - b_grid[first:last, None] > limit + _RULE_TOL
            return np.where(violation, cost, 0.0)

    infeasible = np.zeros((n_b, n_y), dtype=bool)
    if not soft:
        bound = np.full(n_b, np.inf)
        if cap is not None:
            bound = np.minimum(bound, cap)
        if limit is not None:
            bound = np.minimum(bound, b_grid + limit)
        infeasible = (b_grid[0] > bound + _RULE_TOL)[:, None] & active[None, :]

    return _CompiledRule(column_penalty=column_penalty, transition_penalty=transition_penalty,
                         infeasible=infeasible, rule=rule, b_grid=b_grid, active=active,
                         limit=limit, cost=cost)


def _infeasible_message(compiled: _CompiledRule, b_grid: np.ndarray, y_grid: np.ndarray) -> str:
    """Count of infeasible states plus up to ten ``(b=..., y=...)`` examples."""
    rows, cols = np.nonzero(compiled.infeasible)
    examples = ", ".join(f"(b={b_grid[i]:.6g}, y={y_grid[j]:.6g})" for i, j in zip(rows[:10], cols[:10]))
    return (f"The fiscal rule leaves no admissible debt choice in {rows.size} states, e.g. {examples}. "
            'Relax the rule, or pass FiscalRule(..., infeasible="default") to force default there.')
