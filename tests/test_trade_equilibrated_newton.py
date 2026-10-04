"""Tests for ``solve_trade_equilibrium(method="equilibrated_newton")``.

The equilibrated Newton (Ruiz row/column scaling, relative finite-difference
steps, backtracking on the scaled residual) was validated in
``reviews/2026-10-04-clean-table-tariff-scenarios/`` against MATLAB's t10
reference of the legacy 77x11 table; the full-size parity check is marked slow.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.trade import TradeCalibrationResult, calibrate_trade_model, solve_trade_equilibrium
from puremacro.trade.solver import _equilibrated_newton_solve, _ruiz_equilibrate


@pytest.fixture(scope="module")
def calib2() -> TradeCalibrationResult:
    """Balanced 2-country, 2-sector table (same construction as test_trade_equilibrium)."""
    nc, ns, nfd = 2, 2, 3
    data = np.zeros((ns * nc + 3, ns * nc + nfd * nc))
    data[:4, :4] = np.array([
        [10.0, 15.0, 5.0, 5.0],
        [15.0, 20.0, 10.0, 10.0],
        [5.0, 5.0, 12.0, 18.0],
        [10.0, 10.0, 18.0, 22.0],
    ])
    y = np.array([100.0, 150.0, 120.0, 180.0])
    va = y - data[:4, :4].sum(axis=0)
    taxes = 0.05 * y
    data[4, :4] = taxes
    data[5, :4] = (2.0 / 3.0) * (va - taxes)
    data[6, :4] = (1.0 / 3.0) * (va - taxes)
    fd = y - data[:4, :4].sum(axis=1)
    home = np.array([0.50, 0.25, 0.05]); away = np.array([0.10, 0.08, 0.02])
    for i in range(4):
        data[i, 4:] = fd[i] * (np.r_[home, away] if i < 2 else np.r_[away, home])
    data[4, 4:] = 0.02 * data[:4, 4:].sum(axis=0)
    return calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, validate=True)


def _tariff(calib: TradeCalibrationResult, rate: float) -> tuple[np.ndarray, np.ndarray]:
    nc, ns, nfd = calib.n_countries, calib.n_sectors, calib.n_final_demand
    tau = np.ones((ns * nc, ns, nc)); tau_fd = np.ones((ns * nc, nfd, nc))
    tau[ns:, :, 0] = 1.0 + rate
    tau_fd[ns:, :, 0] = 1.0 + rate
    return tau, tau_fd


# ---------------------------------------------------------------------------
# Ruiz equilibration
# ---------------------------------------------------------------------------
def test_ruiz_equilibration_balances_rows_and_columns_and_reconstructs():
    rng = np.random.default_rng(0)
    J = rng.standard_normal((6, 6)) * np.logspace(-8, 8, 6)[:, None] * np.logspace(4, -4, 6)[None, :]
    A, dr, dc = _ruiz_equilibrate(J, sweeps=20)
    np.testing.assert_allclose(A, dr[:, None] * J * dc[None, :], rtol=1e-12)
    np.testing.assert_allclose(np.abs(A).max(axis=1), 1.0, rtol=1e-3)
    np.testing.assert_allclose(np.abs(A).max(axis=0), 1.0, rtol=1e-3)
    assert np.linalg.cond(A) < 1e-6 * np.linalg.cond(J)


def test_ruiz_equilibration_keeps_unit_scale_for_zero_rows_and_columns():
    J = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 8.0]])
    A, dr, dc = _ruiz_equilibrate(J)
    assert dr[1] == 1.0 and dc[1] == 1.0
    assert np.isfinite(A).all()


# ---------------------------------------------------------------------------
# Generic solver behaviour
# ---------------------------------------------------------------------------
def test_badly_scaled_system_converges():
    """Unknowns and equations spanning sixteen orders of magnitude."""
    s = np.array([1e-8, 1.0, 1e8])
    root = np.array([2.0, -1.0, 3.0]) * s

    def f(x):
        z = x / s
        return s[::-1] * np.array([z[0] ** 2 - 4.0 + 0.1 * (z[1] + 1.0), z[1] + 1.0 + 0.2 * (z[2] - 3.0), z[2] ** 3 - 27.0])

    info: dict = {}
    x, conv, it, max_res, diff, res = _equilibrated_newton_solve(
        f, root * 1.3, tol=1e-9, max_iter=40, fd_scale=np.abs(root), info=info)
    assert conv and info["termination"] == "tolerance"
    np.testing.assert_allclose(x, root, rtol=1e-8)
    assert info["residual_history"][-1] == pytest.approx(max_res)


def test_no_descent_reports_the_weakest_direction():
    """No root, and a fold along x0 = -x1 that scaling cannot remove.

    With u = (x0 + x1)/sqrt(2) and v = (x0 - x1)/sqrt(2), F = (u**2 + 1 + v,
    u**2 + 1 - v) has no zero; the iteration ends near u = 0 where the Jacobian
    is singular along (1, 1)/sqrt(2). (A single vanishing row or column would
    not do: Ruiz scaling rescales it to unit norm.)
    """
    def f(x):
        u = (x[0] + x[1]) / np.sqrt(2.0); v = (x[0] - x[1]) / np.sqrt(2.0)
        return np.array([u ** 2 + 1.0 + v, u ** 2 + 1.0 - v])

    info: dict = {}
    x, conv, *_ = _equilibrated_newton_solve(f, np.array([0.5, 0.2]), tol=1e-10, max_iter=60, info=info)
    assert not conv
    assert info["termination"] in ("no_descent", "max_iter")
    assert info["equilibrated_condition"] > 1e3
    v = info["weakest_direction"]
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert abs(v[0]) == pytest.approx(np.sqrt(0.5), abs=0.02) and v[0] * v[1] > 0


# ---------------------------------------------------------------------------
# Trade model
# ---------------------------------------------------------------------------
def test_baseline_is_returned_without_iterating(calib2):
    res = solve_trade_equilibrium(calib2, method="equilibrated_newton")
    assert res.converged and res.iterations == 0
    assert res.metadata["method"] == "equilibrated_newton"
    assert res.metadata["solver_termination"] == "tolerance"


def test_tariff_equilibrium_matches_hybr(calib2):
    tau, tau_fd = _tariff(calib2, 0.10)
    res = solve_trade_equilibrium(calib2, tau=tau, tau_fd=tau_fd, method="equilibrated_newton", tol=1e-9)
    ref = solve_trade_equilibrium(calib2, tau=tau, tau_fd=tau_fd, method="hybr", tol=1e-9)
    assert res.converged and ref.converged
    assert res.max_residual <= 1e-9
    np.testing.assert_allclose(res.x_sol, ref.x_sol, rtol=1e-7, atol=1e-9)


def test_unconverged_solve_names_the_country_of_the_weakest_direction(calib2):
    tau, tau_fd = _tariff(calib2, 0.10)
    res = solve_trade_equilibrium(calib2, tau=tau, tau_fd=tau_fd, method="equilibrated_newton",
                                  tol=1e-30, max_iter=1)
    md = res.metadata
    assert not res.converged and md["solver_termination"] in ("max_iter", "no_descent")
    assert md["near_singular_country"] in calib2.country_codes
    assert 0.0 < md["near_singular_country_weight"] <= 1.0
    assert md["equilibrated_condition"] >= 1.0
    assert "weakest_direction" not in md            # metadata stays small and serialisable


def test_unknown_method_message_lists_the_new_method(calib2):
    with pytest.raises(ValueError, match="equilibrated_newton"):
        solve_trade_equilibrium(calib2, method="no_such_method")


@pytest.mark.slow
def test_legacy_t10_matches_matlab_reference_from_a_cold_start():
    """Started from the base, the 77x11 legacy t10 equilibrium equals MATLAB's (review 2026-10-04)."""
    from puremacro.trade.data import load_icio_data, load_reference_solution

    ref = load_reference_solution("t10", source="legacy")
    calib = calibrate_trade_model(load_icio_data(source="legacy"), ns=11, nc=77, nfd=3, validate=True)
    base = solve_trade_equilibrium(calib, method="newton", tol=2.5e-3)
    res = solve_trade_equilibrium(calib, tau=ref["tau_a"], tau_fd=ref["taufd_a"], tauf=np.zeros(77),
                                  tauf_fd=np.zeros(77), x0=base.x_sol, method="equilibrated_newton",
                                  tol=2.5e-3, max_iter=20)
    assert res.converged
    x_ref = np.asarray(ref["xx_sol"], dtype=float).ravel()
    rel = np.abs(res.x_sol - x_ref) / np.maximum(np.abs(x_ref), 1.0)
    assert rel.max() < 1e-8
