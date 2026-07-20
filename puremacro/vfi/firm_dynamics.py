"""Firm dynamics with entry and exit (Hopenhayn 1992).

A distinct model class: there is NO endogenous asset -- the state is firm
productivity ``s`` (an AR(1) Markov process). Each period a firm earns
``profit(s)`` then decides, at end of period, whether to continue or exit (exit
value 0); it continues iff the expected continuation ``E[V(s')|s] >= 0``. Entry
is free: a price clears ``E_nu[V(s;p)] = entry_cost`` for an entrant draw ``nu``.
The stationary firm measure is NOT mass-conserving -- entrants flow in and
exiters flow out. numpy-only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class FirmEntryExitEquilibrium:
    """A solved Hopenhayn free-entry equilibrium."""
    price: float
    value: np.ndarray        # V(s) at the equilibrium price
    survive: np.ndarray      # bool (s,): True = continue, False = exit
    entry_value: float       # E_nu[V] at the solved price (== entry_cost)
    residual: float          # E_nu[V] - entry_cost (~ 0)
    n_evals: int


def firm_value_with_exit(profit, P_z, beta, *, tol: float = 1e-12,
                         max_iter: int = 100_000):
    """Value of an incumbent firm with an end-of-period exit option.

    ``V(s) = profit(s) + beta * max(0, E_{s'|s}[V(s')])``, iterated to a fixed
    point. Returns ``(V, survive)`` where ``survive[s] = (E[V'|s] >= 0)`` (the
    firm continues; otherwise it exits for value 0). ``profit`` is (n_s,), ``P_z``
    the (n_s, n_s) row-stochastic productivity transition.
    """
    profit = np.asarray(profit, dtype=float)
    P = np.asarray(P_z, dtype=float)
    n = profit.shape[0]
    if P.shape != (n, n):
        raise ValueError(f"P_z must be ({n},{n}); got {P.shape}")
    beta = float(beta)
    V = profit.copy()
    for _ in range(max_iter):
        EV = P @ V
        V_new = profit + beta * np.maximum(0.0, EV)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    else:
        raise RuntimeError("firm_value_with_exit did not converge")
    survive = (P @ V) >= 0.0
    return V, survive


def firm_stationary_distribution(P_z, survive, entry_dist, *, entry_mass: float = 1.0,
                                 normalize: bool = True):
    """Stationary firm measure under exit + entry.

    Survivors transition; entrants (mass ``entry_mass``) appear per ``entry_dist``
    ``nu``: ``g = g @ S + entry_mass * nu`` with ``S[s,s'] = survive[s]*P[s,s']``.
    Solved as a linear system (``S`` is sub-stochastic, so ``I - S`` is
    invertible). Returns the (n_s,) measure, normalized to sum 1 if ``normalize``.
    """
    P = np.asarray(P_z, dtype=float)
    surv = np.asarray(survive, dtype=float)
    nu = np.asarray(entry_dist, dtype=float)
    n = P.shape[0]
    S = surv[:, None] * P                       # S[s,s'] = survive[s] * P[s,s']
    A = np.eye(n) - S
    g = np.linalg.solve(A.T, float(entry_mass) * nu)   # g @ (I - S) = entry_mass * nu
    g = np.maximum(g, 0.0)
    if normalize:
        total = g.sum()
        if total <= 0.0:
            raise ValueError("degenerate firm distribution (no survivors)")
        g = g / total
    return g


def free_entry_price(profit_at, entry_dist, entry_cost, price_bracket, *, P_z, beta,
                     value_tol: float = 1e-12, xtol: float = 1e-8,
                     max_evals: int = 100):
    """Find the output price that clears free entry: ``E_nu[V(s;p)] = entry_cost``.

    ``profit_at(p) -> (n_s,)`` is the per-state operating profit at price ``p``;
    ``entry_dist`` is the entrant productivity distribution ``nu``. ``E_nu[V]`` is
    increasing in ``p``, so brentq on ``price_bracket=(lo, hi)`` finds the unique
    clearing price. Returns a FirmEntryExitEquilibrium.
    """
    nu = np.asarray(entry_dist, dtype=float)
    P = np.asarray(P_z, dtype=float)
    counter = {"n": 0}

    def _resid(p):
        counter["n"] += 1
        V, _ = firm_value_with_exit(profit_at(p), P, beta, tol=value_tol)
        return float(nu @ V) - float(entry_cost)

    lo, hi = price_bracket
    p_star = brentq(_resid, lo, hi, xtol=xtol, maxiter=max_evals)
    V, survive = firm_value_with_exit(profit_at(p_star), P, beta, tol=value_tol)
    ev = float(nu @ V)
    return FirmEntryExitEquilibrium(price=float(p_star), value=V, survive=survive,
                                    entry_value=ev, residual=ev - float(entry_cost),
                                    n_evals=counter["n"])


__all__ = ["firm_value_with_exit", "firm_stationary_distribution",
           "free_entry_price", "FirmEntryExitEquilibrium"]
