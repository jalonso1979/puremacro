"""Comprehensive conformance tests for Solver Diagnostic Records (REPORT.md Item 5).

Verifies:
1. Separation of solver convergence, economic accuracy, and support limits.
2. A generic converged=True does not yield an audit PASS if inaccurate or in an unsupported regime.
3. Explicit boundary regimes: explosive VAR (rho >= 1.0), sovereign debt ceiling saturation (P(B'=B_max) > 0.01).
4. Backward compatibility: tuple unpacking, len(), indexing, and legacy attributes remain intact.
5. Integration with puremacro.validation.audit.audit_result across solver families.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.solvers.diagnostics import (
    AccuracyStatus,
    ConvergenceStatus,
    SolverDiagnosticRecord,
    SupportStatus,
    determine_overall_status,
    make_solver_diagnostic_record,
)
from puremacro.trade._results import TradeEquilibriumResult
from puremacro.sovereign._results import SovereignDefaultResult
from puremacro.var._results import VarEstimateResult
from puremacro.models.one_asset_hank import OneAssetHANKResult, OneAssetHANKSteadyState
from puremacro.validation.audit import audit_result


def test_solver_diagnostic_dataclasses_and_hierarchy():
    """Verify SolverDiagnosticRecord hierarchy and builders."""
    # 1. Failed convergence -> FAILED
    c_fail = ConvergenceStatus(converged=False, iterations=100, stopping_reason="max_iters")
    a_pass = AccuracyStatus(is_accurate=True, max_residual=1e-6, residual_norm=1e-6)
    s_pass = SupportStatus(is_supported=True, regime="interior")
    assert determine_overall_status(c_fail, a_pass, s_pass) == "FAILED"

    # 2. Converged but unsupported regime -> UNSUPPORTED_REGIME
    c_pass = ConvergenceStatus(converged=True, iterations=10, stopping_reason="converged")
    s_unsupp = SupportStatus(is_supported=False, regime="boundary_debt_saturation", warnings=("Debt ceiling binds",))
    assert determine_overall_status(c_pass, a_pass, s_unsupp) == "UNSUPPORTED_REGIME"

    # 3. Converged, supported, but inaccurate -> CONVERGED_INACCURATE
    a_inacc = AccuracyStatus(is_accurate=False, max_residual=0.05, residual_norm=0.1, tolerance=0.01)
    assert determine_overall_status(c_pass, a_inacc, s_pass) == "CONVERGED_INACCURATE"

    # 4. Converged, accurate, supported -> PASS
    assert determine_overall_status(c_pass, a_pass, s_pass) == "PASS"

    rec = make_solver_diagnostic_record(
        converged=True,
        iterations=5,
        is_accurate=True,
        max_residual=1e-5,
        is_supported=True,
    )
    assert rec.overall_status == "PASS"
    assert rec.is_pass is True
    d = rec.to_dict()
    assert d["overall_status"] == "PASS"
    assert "status=PASS" in rec.summary()


def test_trade_convergence_vs_accuracy_distinction():
    """Trade equilibrium with loose residual tolerance flags CONVERGED_INACCURATE."""
    nc, ns = 2, 2
    x_sol = np.zeros(2 * ns * nc + 4 * nc - 1)
    p_sol = np.ones((1, ns, nc))
    y_sol = np.ones((1, ns, nc))
    r_sol = np.ones((1, 1, nc))
    w_sol = np.ones((1, 1, nc))
    T_sol = np.zeros((1, 1, nc))
    XN_sol = np.zeros((nc - 1,))

    # Inaccurate trade result: converged is True, but max_residual = 0.05 > accuracy_tolerance 1e-4
    res_inaccurate = TradeEquilibriumResult(
        x_sol=x_sol,
        p_sol=p_sol,
        y_sol=y_sol,
        r_sol=r_sol,
        w_sol=w_sol,
        T_sol=T_sol,
        XN_sol=XN_sol,
        converged=True,
        iterations=8,
        max_residual=0.05,
        diff=0.05,
        metadata={"method": "equilibrated_newton", "tol": 0.1, "accuracy_tolerance": 1e-4},
    )

    diag = res_inaccurate.solver_diagnostics
    assert diag is not None
    assert diag.convergence.converged is True
    assert diag.accuracy.is_accurate is False
    assert diag.overall_status == "CONVERGED_INACCURATE"
    assert diag.is_pass is False
    summary_df = res_inaccurate.summary(detailed=False)
    assert summary_df.loc["Overall Status", "Value"] == "CONVERGED_INACCURATE"

    # Accurate trade result
    res_accurate = TradeEquilibriumResult(
        x_sol=x_sol,
        p_sol=p_sol,
        y_sol=y_sol,
        r_sol=r_sol,
        w_sol=w_sol,
        T_sol=T_sol,
        XN_sol=XN_sol,
        converged=True,
        iterations=8,
        max_residual=1e-6,
        diff=1e-6,
        metadata={"method": "equilibrated_newton", "tol": 1e-5, "accuracy_tolerance": 1e-4},
    )
    assert res_accurate.solver_diagnostics.overall_status == "PASS"
    assert res_accurate.solver_diagnostics.is_pass is True


def test_trade_infeasible_price_support_regime():
    """Trade equilibrium with non-positive price or factor return flags UNSUPPORTED_REGIME."""
    nc, ns = 2, 2
    x_sol = np.zeros(2 * ns * nc + 4 * nc - 1)
    p_bad = np.array([[[-1.0, 1.0], [1.0, 1.0]]])  # Negative price
    y_sol = np.ones((1, ns, nc))
    r_sol = np.ones((1, 1, nc))
    w_sol = np.ones((1, 1, nc))
    T_sol = np.zeros((1, 1, nc))
    XN_sol = np.zeros((nc - 1,))

    res = TradeEquilibriumResult(
        x_sol=x_sol,
        p_sol=p_bad,
        y_sol=y_sol,
        r_sol=r_sol,
        w_sol=w_sol,
        T_sol=T_sol,
        XN_sol=XN_sol,
        converged=True,
        max_residual=1e-6,
    )
    assert res.solver_diagnostics.overall_status == "UNSUPPORTED_REGIME"
    assert res.solver_diagnostics.support.is_supported is False
    assert res.solver_diagnostics.support.regime == "infeasible_prices"


def test_sovereign_boundary_saturation_and_bellman_residual():
    """Sovereign result with debt grid upper bound saturation flags UNSUPPORTED_REGIME."""
    b_grid = np.linspace(0.0, 1.0, 10)
    y_grid = np.linspace(0.8, 1.2, 5)

    # 1. Normal interior sovereign result
    res_normal = SovereignDefaultResult(
        b_grid=b_grid,
        y_grid=y_grid,
        policy_b_prime=np.zeros((10, 5)),  # always chooses zero debt
        converged=True,
        bellman_residual=1e-5,
        iterations=50,
    )
    assert res_normal.solver_diagnostics.overall_status == "PASS"
    assert res_normal.solver_diagnostics.support.is_supported is True
    assert res_normal.summary().loc[res_normal.summary()["metric"] == "overall_status", "value"].iloc[0] == "PASS"

    # 2. Boundary saturated sovereign result (debt choice hits b_max in > 1% of states)
    policy_sat = np.zeros((10, 5))
    policy_sat[5:, :] = 1.0  # 50% of grid chooses b_max = 1.0
    res_sat = SovereignDefaultResult(
        b_grid=b_grid,
        y_grid=y_grid,
        policy_b_prime=policy_sat,
        converged=True,
        bellman_residual=1e-5,
        iterations=50,
    )
    assert res_sat.solver_diagnostics.overall_status == "UNSUPPORTED_REGIME"
    assert res_sat.solver_diagnostics.support.is_supported is False
    assert res_sat.solver_diagnostics.support.regime == "boundary_debt_saturation"

    # 3. Converged but large Bellman residual (> 1e-3)
    res_inacc = SovereignDefaultResult(
        b_grid=b_grid,
        y_grid=y_grid,
        policy_b_prime=np.zeros((10, 5)),
        converged=True,
        bellman_residual=0.05,
        iterations=10,
    )
    assert res_inacc.solver_diagnostics.overall_status == "CONVERGED_INACCURATE"
    assert res_inacc.solver_diagnostics.accuracy.is_accurate is False


def test_var_companion_matrix_spectral_radius_support():
    """VAR with companion matrix spectral radius rho >= 1.0 flags UNSUPPORTED_REGIME."""
    # Stationary VAR(1) with rho = 0.5 < 1.0
    A1_stat = np.array([[0.5, 0.1], [0.0, 0.4]])
    c = np.zeros(2)
    Sigma = np.eye(2)
    resid = np.zeros((100, 2))
    X = np.ones((100, 3))

    res_stat = VarEstimateResult(A_list=[A1_stat], c=c, Sigma=Sigma, resid=resid, X=X)
    assert res_stat.solver_diagnostics.overall_status == "PASS"
    assert res_stat.solver_diagnostics.support.is_supported is True
    assert res_stat.spectral_radius == pytest.approx(0.5, abs=1e-4)
    assert "status            : PASS" in res_stat.summary()

    # Explosive / non-stationary VAR(1) with eigenvalue 1.2 >= 1.0
    A1_exp = np.array([[1.2, 0.0], [0.0, 0.8]])
    res_exp = VarEstimateResult(A_list=[A1_exp], c=c, Sigma=Sigma, resid=resid, X=X)
    assert res_exp.solver_diagnostics.overall_status == "UNSUPPORTED_REGIME"
    assert res_exp.solver_diagnostics.support.is_supported is False
    assert res_exp.solver_diagnostics.support.regime == "non_stationary_explosive"
    assert res_exp.spectral_radius == pytest.approx(1.2, abs=1e-4)
    assert "status            : UNSUPPORTED_REGIME" in res_exp.summary()


def test_var_backward_compatibility():
    """Verify legacy tuple unpacking and indexing contracts for VarEstimateResult."""
    A1 = np.array([[0.5, 0.0], [0.0, 0.5]])
    c = np.zeros(2)
    Sigma = np.eye(2)
    resid = np.ones((10, 2))
    X = np.ones((10, 3))

    res = VarEstimateResult(A_list=[A1], c=c, Sigma=Sigma, resid=resid, X=X)

    # 1. Legacy unpack
    A_list_unpack, c_unpack, Sigma_unpack, resid_unpack, X_unpack = res
    assert np.allclose(A_list_unpack[0], A1)
    assert np.allclose(c_unpack, c)
    assert np.allclose(Sigma_unpack, Sigma)
    assert np.allclose(resid_unpack, resid)
    assert np.allclose(X_unpack, X)

    # 2. Length check
    assert len(res) == 5

    # 3. Indexing check
    assert np.allclose(res[0][0], A1)
    assert np.allclose(res[1], c)
    assert np.allclose(res[2], Sigma)
    assert np.allclose(res[3], resid)
    assert np.allclose(res[4], X)


def test_one_asset_hank_diagnostics_and_euler_kkt():
    """OneAssetHANKResult checks Euler KKT errors and feasibility bounds."""
    # Fake steady state
    ss = OneAssetHANKSteadyState(
        calibration={"beta": 0.98, "r": 0.01, "eis": 0.5},
        asset_grid=np.linspace(0.0, 50.0, 20),
        e_grid=np.array([0.5, 1.5]),
        transition=np.array([[0.9, 0.1], [0.1, 0.9]]),
        policy_a=np.ones((20, 2)),
        policy_c=np.ones((20, 2)),
        policy_n=np.ones((20, 2)),
        marginal_value=np.ones((20, 2)),
        distribution=np.ones((20, 2)) / 40.0,
        aggregates={},
        diagnostics={},
        converged=True,
    )
    paths = pd.DataFrame({"Y": np.zeros(10), "pi": np.zeros(10)})

    # Pass case
    diag_pass = {"market_residual_max_abs": 1e-5, "euler_kkt_max": 0.005, "iterations": 10}
    res_pass = OneAssetHANKResult(paths=paths, steady_state=ss, nonlinear=True, converged=True, diagnostics=diag_pass)
    assert res_pass.solver_diagnostics.overall_status == "PASS"

    # Euler KKT violation > 0.01 -> CONVERGED_INACCURATE
    diag_inacc = {"market_residual_max_abs": 1e-5, "euler_kkt_max": 0.03, "iterations": 10}
    res_inacc = OneAssetHANKResult(paths=paths, steady_state=ss, nonlinear=True, converged=True, diagnostics=diag_inacc)
    assert res_inacc.solver_diagnostics.overall_status == "CONVERGED_INACCURATE"
    assert res_inacc.solver_diagnostics.accuracy.is_accurate is False

    # Negative mass violation -> UNSUPPORTED_REGIME
    diag_unsupp = {"market_residual_max_abs": 1e-5, "euler_kkt_max": 0.005, "minimum_mass": -0.01}
    res_unsupp = OneAssetHANKResult(paths=paths, steady_state=ss, nonlinear=True, converged=True, diagnostics=diag_unsupp)
    assert res_unsupp.solver_diagnostics.overall_status == "UNSUPPORTED_REGIME"


def test_audit_result_integration_across_solvers():
    """Test puremacro.validation.audit.audit_result on all 4 extended solver result types."""
    # 1. TradeEquilibriumResult
    nc, ns = 2, 2
    res_trade = TradeEquilibriumResult(
        x_sol=np.zeros(2 * ns * nc + 4 * nc - 1),
        p_sol=np.ones((1, ns, nc)),
        y_sol=np.ones((1, ns, nc)),
        r_sol=np.ones((1, 1, nc)),
        w_sol=np.ones((1, 1, nc)),
        T_sol=np.zeros((1, 1, nc)),
        XN_sol=np.zeros((nc - 1,)),
        converged=True,
        max_residual=1e-6,
    )
    rep_trade = audit_result(res_trade)
    assert rep_trade.solver_converged is True
    # Verify no category is marked 'unsupported'
    statuses = {r.category: r.status for r in rep_trade.records}
    assert statuses.get("solver") == "passed"
    assert statuses.get("equations") == "passed"
    assert statuses.get("distribution") == "passed"

    # 2. SovereignDefaultResult
    res_sov = SovereignDefaultResult(
        b_grid=np.linspace(0.0, 1.0, 10),
        y_grid=np.linspace(0.8, 1.2, 5),
        policy_b_prime=np.zeros((10, 5)),
        converged=True,
        bellman_residual=1e-5,
    )
    rep_sov = audit_result(res_sov)
    assert rep_sov.solver_converged is True
    statuses_sov = {r.category: r.status for r in rep_sov.records}
    assert statuses_sov.get("solver") == "passed"
    assert statuses_sov.get("equations") == "passed"
    assert statuses_sov.get("distribution") == "passed"

    # 3. VarEstimateResult
    res_var = VarEstimateResult(
        A_list=[np.array([[0.5, 0.0], [0.0, 0.5]])],
        c=np.zeros(2),
        Sigma=np.eye(2),
        resid=np.zeros((10, 2)),
        X=np.ones((10, 3)),
    )
    rep_var = audit_result(res_var)
    assert rep_var.solver_converged is True
    statuses_var = {r.category: r.status for r in rep_var.records}
    assert statuses_var.get("solver") == "passed"
    assert statuses_var.get("equations") == "passed"
    assert statuses_var.get("distribution") == "passed"

    # Explosive VAR should fail distribution/support check in audit report
    res_exp_var = VarEstimateResult(
        A_list=[np.array([[1.5, 0.0], [0.0, 0.5]])],
        c=np.zeros(2),
        Sigma=np.eye(2),
        resid=np.zeros((10, 2)),
        X=np.ones((10, 3)),
    )
    rep_exp = audit_result(res_exp_var)
    exp_checks = {r.check: r.status for r in rep_exp.records}
    assert exp_checks["stationarity_support"] == "failed"
    assert exp_checks["solver_overall_status"] == "failed"
