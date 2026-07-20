from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi import markov_stationary, tauchen
from puremacro.vfi.firm_dynamics import (
    FirmEntryExitEquilibrium,
    firm_stationary_distribution,
    firm_value_with_exit,
    free_entry_price,
)


def test_value_with_exit_threshold():
    # profit increasing in s; the survive decision is a productivity threshold
    # (low-s firms exit, high-s survive).
    z_grid, P = tauchen(15, 0.8, 0.3)
    profit = np.exp(z_grid) - 1.2                  # increasing in s, negative at the bottom
    V, survive = firm_value_with_exit(profit, P, beta=0.9)
    assert V.shape == (15,) and survive.shape == (15,)
    assert survive.dtype == bool
    # survive is monotone non-decreasing in s (a single threshold)
    assert np.all(np.diff(survive.astype(int)) >= 0)
    assert survive[-1] and not survive[0]          # top survives, bottom exits


def test_value_with_exit_no_exit_when_all_profitable():
    z_grid, P = tauchen(10, 0.7, 0.2)
    profit = 5.0 + np.exp(z_grid)                   # always strongly positive
    V, survive = firm_value_with_exit(profit, P, beta=0.9)
    assert np.all(survive)                          # nobody exits
    # with no exit, V solves V = profit + beta P V  (geometric)
    np.testing.assert_allclose(V, profit + 0.9 * (P @ V), atol=1e-8)


def test_stationary_distribution_mass_and_selection():
    z_grid, P = tauchen(15, 0.8, 0.3)
    nu = markov_stationary(P)                       # entrant distribution
    profit = np.exp(z_grid) - 1.2
    V, survive = firm_value_with_exit(profit, P, beta=0.9)
    g = firm_stationary_distribution(P, survive, nu)
    np.testing.assert_allclose(g.sum(), 1.0, atol=1e-10)
    assert np.all(g >= -1e-12)
    # selection: incumbents are more productive on average than entrants
    s = np.exp(z_grid)
    assert float(g @ s) > float(nu @ s)


def test_free_entry_clears_and_increasing_in_cost():
    z_grid, P = tauchen(21, 0.9, 0.2)
    nu = markov_stationary(P)
    s = np.exp(z_grid + 1.4)                        # productivity (mean log = 1.4)
    alpha, cf = 2.0 / 3.0, 20.0

    def profit_at(p):
        n_star = (alpha * p * s) ** (1.0 / (1.0 - alpha))
        return p * s * n_star ** alpha - n_star - cf

    eq = free_entry_price(profit_at, nu, entry_cost=40.0, price_bracket=(0.5, 20.0),
                          P_z=P, beta=0.8)
    assert isinstance(eq, FirmEntryExitEquilibrium)
    assert abs(eq.residual) < 1e-6                  # free entry clears: E_nu[V] = c_e
    assert eq.price > 0.0
    assert np.all(np.diff(eq.survive.astype(int)) >= 0)   # exit threshold

    eq_hi = free_entry_price(profit_at, nu, entry_cost=80.0, price_bracket=(0.5, 20.0),
                             P_z=P, beta=0.8)
    assert eq_hi.price > eq.price                   # higher entry cost -> higher price


def test_firm_dynamics_exported():
    from puremacro.vfi import firm_value_with_exit as f1
    from puremacro.vfi import firm_stationary_distribution as f2
    from puremacro.vfi import free_entry_price as f3
    from puremacro.vfi import FirmEntryExitEquilibrium as C

    assert f1 is firm_value_with_exit and f2 is firm_stationary_distribution
    assert f3 is free_entry_price and C is FirmEntryExitEquilibrium
