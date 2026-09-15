"""Unit tests for Multi-Constraint OccBin ($M \\ge 2$).

Verifies:
1. Direct export of `solve_multiconstraint_occbin` from `puremacro.dsge`.
2. Polymorphic dispatch from `solve_occbin` with Mapping and Sequence of models/constraints.
3. 2 simultaneous occasionally binding constraints (ZLB + borrowing limit).
4. Terminal slack check: when constraints bind at T, `converged == False` and warning emitted.
5. Successful convergence when terminal period is slack.
"""
from __future__ import annotations

import warnings
import numpy as np
import pytest

from puremacro.dsge import (
    build_dynare,
    LinearModel,
    OccBinConstraint,
    OccBinResult,
    solve_occbin,
    solve_multiconstraint_occbin,
)


@pytest.fixture
def dual_constraint_models():
    """Setup a canonical New Keynesian model with dual constraints:
    - Constraint 1 (ZLB): r_t >= -r_ss (r_t pegged to -r_ss)
    - Constraint 2 (Borrowing cap): b_t <= b_bar (b_t pegged to b_bar)
    """
    params = {
        "beta": 0.99,
        "sigma": 1.0,
        "kappa": 0.15,
        "phi_pi": 1.5,
        "phi_y": 0.25,
        "rho_r": 0.6,
        "rho_b": 0.5,
        "rho_g": 0.7,
        "gamma_y": 0.2,
        "chi": 0.1,
        "r_ss": 0.015,
        "b_bar": 0.02,
    }

    variables = ["y", "pi", "r", "b", "g"]
    shocks = ["eps_g", "eps_r", "eps_b"]

    # Regime 0: unconstrained
    def ref_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_r),
            curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + shocks_v.eps_b),
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    # Regime 1: ZLB
    def zlb_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (-p.r_ss),
            curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + shocks_v.eps_b),
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    # Regime 2: Borrowing cap
    def borr_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_r),
            curr.b - p.b_bar,
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    steady_state = {v: 0.0 for v in variables}
    m_ref = build_dynare(ref_eqs, variables=variables, shocks=shocks, params=params, steady_state=steady_state)
    m_zlb = build_dynare(zlb_eqs, variables=variables, shocks=shocks, params=params, steady_state=steady_state, check_steady_state=False, strict=False)
    m_borr = build_dynare(borr_eqs, variables=variables, shocks=shocks, params=params, steady_state=steady_state, check_steady_state=False, strict=False)

    c_zlb = OccBinConstraint(variable="r", threshold=-params["r_ss"], operator="<")
    c_borr = OccBinConstraint(variable="b", threshold=params["b_bar"], operator=">")

    return m_ref, m_zlb, m_borr, c_zlb, c_borr


def test_public_export_multiconstraint():
    """Verify solve_multiconstraint_occbin is exported from puremacro.dsge."""
    import puremacro.dsge as dsge
    assert hasattr(dsge, "solve_multiconstraint_occbin")
    assert "solve_multiconstraint_occbin" in dsge.__all__
    assert callable(dsge.solve_multiconstraint_occbin)


def test_dual_constraint_solving(dual_constraint_models):
    """Test 2 simultaneous occasionally binding constraints (ZLB + borrowing cap)."""
    m_ref, m_zlb, m_borr, c_zlb, c_borr = dual_constraint_models

    # Severe negative demand shock and positive credit shock
    shocks = np.zeros((40, 3))
    shocks[0, 0] = -0.06  # eps_g: drops y, triggers ZLB
    shocks[0, 2] = 0.05   # eps_b: increases borrowing, triggers borrowing cap

    res = solve_multiconstraint_occbin(
        m_unconstrained=m_ref,
        m_constrained_dict={"zlb": m_zlb, "borrowing": m_borr},
        shock_seq=shocks,
        constraints={"zlb": c_zlb, "borrowing": c_borr},
        horizon=40,
    )

    assert isinstance(res, OccBinResult)
    assert res.converged
    # Verify both constraints bind at least once
    regimes = np.asarray(res.regimes)
    assert np.any((regimes & 1) == 1), "ZLB constraint should bind in some period"
    assert np.any((regimes & 2) == 2), "Borrowing cap constraint should bind in some period"
    # Terminal period must be slack
    assert regimes[-1] == 0


def test_terminal_slack_check(dual_constraint_models):
    """Test that if a constraint binds in the terminal period, converged is False and warning is emitted."""
    m_ref, m_zlb, m_borr, c_zlb, c_borr = dual_constraint_models

    # Large shock with very short horizon (horizon=3) so economy cannot return to steady state
    shocks = np.zeros((3, 3))
    shocks[0, 0] = -0.08

    with pytest.warns(UserWarning, match="constraint still binds at terminal period"):
        res = solve_multiconstraint_occbin(
            m_unconstrained=m_ref,
            m_constrained_dict={"zlb": m_zlb, "borrowing": m_borr},
            shock_seq=shocks,
            constraints={"zlb": c_zlb, "borrowing": c_borr},
            horizon=3,
        )

    assert not res.converged
    assert res.regimes[-1] != 0


def test_polymorphic_dispatch_solve_occbin_mapping(dual_constraint_models):
    """Test polymorphic dispatch from solve_occbin with dict mapping of models and constraints."""
    m_ref, m_zlb, m_borr, c_zlb, c_borr = dual_constraint_models

    shocks = np.zeros((35, 3))
    shocks[0, 0] = -0.05

    # Passing dictionary mapping to solve_occbin
    res = solve_occbin(
        reference_model=m_ref,
        constrained_model={"zlb": m_zlb, "borr": m_borr},
        constraint={"zlb": c_zlb, "borr": c_borr},
        shock_sequence=shocks,
        horizon=35,
    )

    assert isinstance(res, OccBinResult)
    assert res.converged
    assert hasattr(res, "constraints")
    assert "zlb" in res.constraints


def test_polymorphic_dispatch_solve_occbin_sequence(dual_constraint_models):
    """Test polymorphic dispatch from solve_occbin with sequence of models and constraints."""
    m_ref, m_zlb, m_borr, c_zlb, c_borr = dual_constraint_models

    shocks = np.zeros((35, 3))
    shocks[0, 0] = -0.05

    # Passing list to solve_occbin
    res = solve_occbin(
        reference_model=m_ref,
        constrained_model=[m_zlb, m_borr],
        constraint=[c_zlb, c_borr],
        shock_sequence=shocks,
        horizon=35,
    )

    assert isinstance(res, OccBinResult)
    assert res.converged
    assert len(res.regimes) == 35
