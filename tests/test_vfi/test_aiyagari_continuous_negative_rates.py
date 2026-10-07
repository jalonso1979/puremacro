"""``solve_aiyagari_continuous`` at negative interest rates.

With persistent, volatile earnings, precautionary saving pushes the equilibrium
return below zero: Aiyagari (1994, Table II) reports -0.3456% for sigma = 0.4,
rho = 0.9, mu = 5. Before the fix the household solver started from the guess
c = r a + w z, negative for large assets when r < 0, so every trial rate below
zero returned NaN, and the default bracket started at r = 1e-3 and could not
contain such an equilibrium.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi import solve_aiyagari_continuous, tauchen

BETA = 0.96


@pytest.fixture(scope="module")
def high_risk_chain():
    """Aiyagari's seven-state Tauchen chain for sigma = 0.4, rho = 0.9 (grid of 3 s.d.)."""
    log_grid, p = tauchen(7, 0.9, 0.4 * np.sqrt(1.0 - 0.9**2), m=3.0)
    return np.exp(log_grid), p


def test_default_bracket_finds_negative_equilibrium(high_risk_chain):
    z, p = high_risk_chain
    eq = solve_aiyagari_continuous(beta=BETA, gamma=5.0, P_z=p, z_grid=z, a_max=200.0, n_a=800)
    assert eq.converged
    assert eq.r < 0.0
    assert abs(eq.capital_market_clearing_error) < 1e-6
    # Notebook 67 solves the same economy with solve_egm on an independent grid and
    # finds -0.0897%, stable to 0.003 pp under refinement.
    assert 100.0 * eq.r == pytest.approx(-0.0897, abs=0.01)


def test_negative_lower_bracket_is_evaluated(high_risk_chain):
    z, p = high_risk_chain
    auto = solve_aiyagari_continuous(beta=BETA, gamma=5.0, P_z=p, z_grid=z, a_max=200.0, n_a=800)
    manual = solve_aiyagari_continuous(beta=BETA, gamma=5.0, P_z=p, z_grid=z, a_max=200.0, n_a=800,
                                       r_bracket=(-0.06, 1.0 / BETA - 1.0 - 2e-4))
    assert manual.r == pytest.approx(auto.r, abs=1e-7)


def test_positive_equilibrium_unaffected_by_negative_lower_bound():
    default = solve_aiyagari_continuous(beta=BETA, n_z=3, a_max=30.0, N_k=400)
    wide = solve_aiyagari_continuous(beta=BETA, n_z=3, a_max=30.0, N_k=400,
                                     r_bracket=(-0.05, 1.0 / BETA - 1.0 - 5e-4))
    assert default.r > 0.0
    assert wide.r == pytest.approx(default.r, abs=1e-7)


def test_lower_bound_must_exceed_minus_delta():
    with pytest.raises(ValueError, match="must exceed -delta"):
        solve_aiyagari_continuous(beta=BETA, delta=0.08, r_bracket=(-0.08, 0.03))


def test_no_sign_change_message_points_to_the_asset_grid():
    # A tiny asset grid caps savings below capital demand at every admissible rate.
    with pytest.raises(ValueError, match="increase a_max"):
        solve_aiyagari_continuous(beta=BETA, n_z=3, a_max=2.0, N_k=200)
