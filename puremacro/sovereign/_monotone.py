"""Monotone policy search for short-term sovereign debt.

The repayment problem is ``max_m u(w_i + x_m) + beta * EV_m`` with wealth
``w_i = y - b_i`` and proceeds ``x_m = q(b_m, y) b_m``. For concave ``u``,
``u(w + x_high) - u(w + x_low)`` decreases in ``w``, so after sorting
candidates by descending proceeds the leftmost optimal position is
nondecreasing in wealth (Topkis). Divide and conquer over rows then needs
O(n_b log n_b) evaluations per income state instead of n_b**2.

Objectives are formed in the same floating-point order as the dense tensor
search in :func:`solve_sovereign_default`, so both return identical values.
Among equal-proceeds ties the lowest debt index wins, as in the dense search;
an exact value tie between candidates with different proceeds resolves to the
higher-proceeds one, which can differ from the dense policy (same value). The accelerated result is
certified separately by :func:`exhaustive_maximize`.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["exhaustive_maximize", "maximize", "min_candidate_consumption"]

Utility = Callable[[np.ndarray], np.ndarray]


def min_candidate_consumption(b_grid: np.ndarray, y_grid: np.ndarray, q: np.ndarray) -> float:
    """Smallest consumption over every (debt, choice, income) candidate."""
    proceeds = q * b_grid[:, None]
    return float(np.min((y_grid - 1.0 * np.max(b_grid)) + proceeds.min(axis=0)))


def maximize(b_grid: np.ndarray, y_grid: np.ndarray, q: np.ndarray, ev: np.ndarray,
             beta: float, u: Utility) -> tuple[np.ndarray, np.ndarray]:
    """Repayment value and debt-index policy with bounded search workspace.

    Income columns are independent conditional on ``ev``. Batching them keeps
    divide-and-conquer candidate arrays small without changing arithmetic,
    sorting, or tie-breaking within any income column.
    """
    n_b, n_y = q.shape
    block_income = 32
    if n_y <= block_income:
        return _maximize_columns(b_grid, y_grid, q, ev, beta, u)
    value = np.empty((n_b, n_y))
    policy = np.empty((n_b, n_y), dtype=int)
    for first in range(0, n_y, block_income):
        stop = min(first + block_income, n_y)
        value[:, first:stop], policy[:, first:stop] = _maximize_columns(
            b_grid, y_grid[first:stop], q[:, first:stop], ev[:, first:stop], beta, u)
    return value, policy


def _maximize_columns(b_grid: np.ndarray, y_grid: np.ndarray, q: np.ndarray,
                      ev: np.ndarray, beta: float, u: Utility) -> tuple[np.ndarray, np.ndarray]:
    """Original column-independent divide-and-conquer search."""
    n_b, n_y = q.shape
    order = np.argsort(-(q * b_grid[:, None]), axis=0, kind="stable")
    sorted_q = np.take_along_axis(q, order, axis=0)
    sorted_b = b_grid[order]
    sorted_ev = np.take_along_axis(ev, order, axis=0)
    value = np.empty((n_b, n_y))
    policy = np.empty((n_b, n_y), dtype=int)
    # Rows are ranked by wealth: rank r is debt index n_b - 1 - r.
    low, high = np.array([0]), np.array([n_b - 1])
    col_low = np.zeros((1, n_y), dtype=int)
    col_high = np.full((1, n_y), n_b - 1, dtype=int)
    while low.size:
        middle = (low + high) // 2
        lengths = (col_high - col_low + 1).ravel()
        starts = np.r_[0, np.cumsum(lengths)[:-1]]
        owner = np.repeat(np.arange(lengths.size), lengths)
        column = np.arange(lengths.sum()) - starts[owner] + col_low.ravel()[owner]
        shock = owner % n_y
        debt = n_b - 1 - middle[owner // n_y]
        consumption = (y_grid[shock] - 1.0 * b_grid[debt]) + sorted_q[column, shock] * sorted_b[column, shock]
        objective = u(consumption) + beta * sorted_ev[column, shock]
        best = np.maximum.reduceat(objective, starts)
        leftmost = np.minimum.reduceat(np.where(objective == best[owner], column, n_b), starts)
        best_col = leftmost.reshape(middle.size, n_y)
        rows = n_b - 1 - middle
        value[rows] = best.reshape(middle.size, n_y)
        policy[rows] = order[best_col, np.arange(n_y)[None, :]]
        poorer, richer = low < middle, middle < high
        low, high = (np.r_[low[poorer], middle[richer] + 1],
                     np.r_[middle[poorer] - 1, high[richer]])
        col_low, col_high = (np.concatenate((col_low[poorer], best_col[richer])),
                             np.concatenate((best_col[poorer], col_high[richer])))
    return value, policy


def exhaustive_maximize(b_grid: np.ndarray, y_grid: np.ndarray, q: np.ndarray, ev: np.ndarray,
                        beta: float, u: Utility, block_rows: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Dense argmax over all candidates, blocked to O(block_rows * n_b) memory."""
    n_b, n_y = q.shape
    value = np.empty((n_b, n_y))
    policy = np.empty((n_b, n_y), dtype=int)
    for j in range(n_y):
        proceeds = q[:, j] * b_grid
        continuation = beta * ev[:, j]
        for first in range(0, n_b, block_rows):
            last = min(first + block_rows, n_b)
            objective = u((y_grid[j] - 1.0 * b_grid[first:last, None]) + proceeds[None, :]) + continuation
            policy[first:last, j] = objective.argmax(axis=1)
            value[first:last, j] = objective.max(axis=1)
    return value, policy
