"""Honest convergence flags and flows of quasi-condensed and routed flexible results.

Oracles: for the default flexible configuration the quasi-condensed model is the
legacy Cobb-Douglas/Leontief model with ``replicate_matlab_precedence=False``
(documented in ``_quasi_condensed_solve``), so the legacy Newton solver in
``puremacro.trade.solver``/``equilibrium`` -- a separate code path -- gives the
reference solution, residual and flows. For the legacy route the reference is
``solve_trade_equilibrium`` called with the same arguments.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.trade import (
    calibrate_trade_model,
    compute_equilibrium_residuals,
    solve_trade_equilibrium,
)
from puremacro.trade.flexible import (
    FlexibleTechnologyConfig,
    FlexibleTradeModelConfig,
    solve_flexible_trade_equilibrium,
)


@pytest.fixture(scope="module")
def calib():
    """2x2 calibration with heterogeneous labour shares (as in notebook 62)."""
    Z = np.array([[10., 15., 5., 5.], [15., 20., 10., 10.], [5., 5., 12., 18.], [10., 10., 18., 22.]])
    y0 = np.array([100., 150., 120., 180.])
    fw = np.array([[.5, .25, .05, .1, .08, .02]] * 2 + [[.1, .08, .02, .5, .25, .05]] * 2)
    F = (y0 - Z.sum(1))[:, None] * fw
    ptax = .05 * y0
    fi = y0 - Z.sum(0) - ptax
    ls = np.array([.7, .4, .6, .3])
    table = np.vstack([
        np.hstack([Z, F]),
        np.r_[ptax, .02 * F.sum(0)],
        np.r_[ls * fi, np.zeros(6)],
        np.r_[(1 - ls) * fi, np.zeros(6)],
    ])
    return calibrate_trade_model(table, ns=2, nc=2, nfd=3, country_codes=["A", "B"], sector_codes=["FOOD", "MANU"])


@pytest.fixture(scope="module")
def tau(calib):
    t = np.ones((calib.n_sectors * calib.n_countries, calib.n_sectors, calib.n_countries))
    t[0, :, 1] = 1.25
    return t


@pytest.fixture(scope="module")
def legacy_oracle(calib, tau):
    """Legacy Newton solve of the model the default quasi-condensed configuration solves."""
    res = solve_trade_equilibrium(calib, tau=tau, tol=1e-10, replicate_matlab_precedence=False)
    assert res.converged and res.max_residual <= 1e-10
    return res


class TestSolverQuasiCondensedFlag:
    """solve_trade_equilibrium(method="quasi_condensed") reports the residual it judged."""

    def test_flag_agrees_with_reported_residual(self, calib, tau, legacy_oracle):
        # 4.4.0 returned converged=True with max_residual 0.0487 (residual_norm 0.155)
        # against tol=1e-10: the legacy residual at the MATLAB pricing convention.
        tol = 1e-10
        res = solve_trade_equilibrium(calib, tau=tau, method="quasi_condensed", tol=tol)
        assert res.converged
        assert res.max_residual <= tol
        assert res.residual_norm == pytest.approx(float(np.sum(np.abs(res.residuals))), rel=1e-12)
        assert np.max(np.abs(res.residuals)) == res.max_residual
        # Independent check: the legacy equations of the solved model vanish at x_sol.
        legacy_res = compute_equilibrium_residuals(
            res.x_sol, calib, tau=tau, replicate_matlab_precedence=False
        )
        assert np.max(np.abs(legacy_res)) <= 1e-8
        assert res.metadata["replicate_matlab_precedence"] is False

    def test_flows_match_the_legacy_solution_of_the_same_model(self, calib, tau, legacy_oracle):
        res = solve_trade_equilibrium(calib, tau=tau, method="quasi_condensed", tol=1e-10)
        np.testing.assert_allclose(res.x_sol, legacy_oracle.x_sol, rtol=0.0, atol=1e-8)
        for name in ("exports", "imports", "cpi", "terms_of_trade", "gdp", "tariffs"):
            np.testing.assert_allclose(
                getattr(res, name), getattr(legacy_oracle, name), rtol=1e-8, atol=1e-8, err_msg=name
            )

    def test_non_convergence_is_reported(self, calib, tau):
        res = solve_trade_equilibrium(calib, tau=tau, method="quasi_condensed", tol=1e-10, max_iter=1)
        assert not res.converged
        assert res.max_residual > 1e-10

    @pytest.mark.parametrize(
        "kw",
        [
            dict(sigma_y=0.5),
            dict(config=FlexibleTradeModelConfig(technology=FlexibleTechnologyConfig(rho_va=0.7))),
        ],
    )
    def test_active_flexible_settings_are_refused(self, calib, tau, kw):
        # The legacy flow fields of TradeEquilibriumResult cannot describe these models.
        with pytest.raises(ValueError, match="solve_flexible_trade_equilibrium"):
            solve_trade_equilibrium(calib, tau=tau, method="quasi_condensed", **kw)

    @pytest.mark.parametrize(
        "kw",
        [
            dict(fiscal_closure="deficit_reduction"),
            dict(sigma=2.0),
            dict(capacity_margins={(0, 0): 0.1}),
        ],
    )
    def test_settings_the_route_ignores_are_refused(self, calib, tau, kw):
        with pytest.raises(ValueError, match="quasi_condensed"):
            solve_trade_equilibrium(calib, tau=tau, method="quasi_condensed", **kw)


class TestRoutedFlexibleResultFlows:
    """exports/imports/cpi/terms_of_trade/gdp describe the solved model or refuse."""

    def test_default_configuration_matches_the_legacy_oracle(self, calib, tau, legacy_oracle):
        # 4.4.0: exports [55.869, 63.122] from the property versus [56.730, 64.264].
        res = solve_flexible_trade_equilibrium(calib, tau=tau, method="quasi_condensed", tol=1e-10)
        assert res.converged
        for name in ("exports", "imports", "cpi", "terms_of_trade", "gdp", "gdp_fc"):
            np.testing.assert_allclose(
                getattr(res, name), getattr(legacy_oracle, name), rtol=1e-8, atol=1e-8, err_msg=name
            )

    @pytest.mark.parametrize("kw", [dict(), dict(replicate_matlab_precedence=False)])
    def test_legacy_route_matches_the_legacy_solver(self, calib, tau, kw):
        # 4.4.0 evaluated the properties without the tariffs: exports of B 68.944 versus 69.191.
        res = solve_flexible_trade_equilibrium(calib, tau=tau, tol=1e-10, **kw)
        assert res.metadata["routing"] == "default"
        ref = solve_trade_equilibrium(calib, tau=tau, tol=1e-10, **kw)
        np.testing.assert_array_equal(res.x_sol, ref.x_sol)
        for name in ("exports", "imports", "cpi", "terms_of_trade", "gdp", "gdp_fc"):
            np.testing.assert_allclose(
                getattr(res, name), getattr(ref, name), rtol=1e-12, atol=1e-12, err_msg=name
            )

    def test_active_configuration_uses_the_solved_flows(self, calib, tau):
        # 4.4.0: exports [58.195, 59.858] from the property versus [62.155, 58.256].
        res = solve_flexible_trade_equilibrium(calib, tau=tau, sigma_trade=5.0, tol=1e-10)
        assert res.metadata["method"] == "quasi_condensed" and res.converged
        bilateral = res.metadata["bilateral_trade"]
        np.testing.assert_allclose(res.exports, bilateral.sum(axis=1), rtol=1e-12)
        np.testing.assert_allclose(res.imports, bilateral.sum(axis=0), rtol=1e-12)
        # GDP is a function of the state alone.
        gdp_fc = res.w_sol.ravel() * calib.l_endow.ravel() + res.r_sol.ravel() * calib.k_endow.ravel()
        np.testing.assert_allclose(res.gdp_fc, gdp_fc, rtol=1e-12)
        np.testing.assert_allclose(res.gdp, gdp_fc + res.T_sol.ravel(), rtol=1e-12)
        with pytest.raises(NotImplementedError, match="cpi"):
            res.cpi
        with pytest.raises(NotImplementedError, match="terms_of_trade"):
            res.terms_of_trade

    def test_welfare_proxy_is_unchanged(self, calib, tau):
        """welfare_decomposition keeps its historical legacy-flow proxy (values from 4.4.0)."""
        base = solve_flexible_trade_equilibrium(calib, sigma_trade=5.0, tol=1e-10)
        arm = solve_flexible_trade_equilibrium(calib, tau=tau, sigma_trade=5.0, tol=1e-10)
        got = arm.welfare_decomposition(base)[["EV", "terms_of_trade", "efficiency"]].to_numpy()
        expected = np.array([
            [-0.05810166006406803, -1.3204686684384077, 1.2623670083743397],
            [0.03146903447353111, 1.3506844622696033, -1.3192154277960721],
        ])
        np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-9)
        leg0 = solve_flexible_trade_equilibrium(calib, tol=1e-10)
        leg1 = solve_flexible_trade_equilibrium(calib, tau=tau, tol=1e-10)
        got = leg1.welfare_decomposition(leg0)[["EV", "terms_of_trade", "efficiency"]].to_numpy()
        expected = np.array([
            [-0.0871163129395569, -3.6374764465263776, 3.5503601335868207],
            [0.17068675445503345, 3.8536665033942965, -3.682979748939263],
        ])
        np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-9)
