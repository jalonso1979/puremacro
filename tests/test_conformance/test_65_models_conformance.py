"""Permanent metamorphic, conservation, and optimization-equivalence conformance suite for 6.5.0 models.

Covers exact mathematical invariants (no mocked core math, no relaxed tolerances):
1. ``puremacro.models.hank_fiscal``: exact finite-horizon Walras identity,
   zero-shock steady-state baseline, government flow/PV budget laws, and
   stationary debt-rule mean reversion across closures and tax types.
2. ``puremacro.lp.cce``: unit-label permutation invariance, outcome affine
   scaling equivariance, unit fixed-shift invariance, and noiseless
   homogeneous-slope equality (Mean Group == Pooled == exact true slope).
3. ``puremacro.workbench``: panel row/unit permutation invariance and joint
   outcome/model unit-scaling equivariance on ``empirical_lp`` and ``irf_scorecard``.
4. ``puremacro.sovereign.smm``: Markov transition row normalization, stationary
   access/exclusion mass conservation, and reproducible seed determinism.
5. Optimization-equivalence regression checks verifying the vectorized/cached
   fast paths in ``puremacro.trade._accounting`` and ``puremacro.lp.cce``
   match unoptimized reference implementations to ``< 1e-12`` and reject
   invalid inputs identically.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from puremacro.lp.cce import (
    _entity_fe_ols,
    _estimate_cce_mean_group_horizon,
    cce_panel_lp,
)
from puremacro.models.hank_fiscal import solve_hank_fiscal
from puremacro.sovereign._fiscal_paths import draw_uniforms, ergodic_moments, simulate_paths
from puremacro.sovereign.smm import (
    _compute_stationary_or_simulated_moments,
    _stationary_access_exclusion_dist,
    solve_fiscal_sovereign_default,
)
from puremacro.trade import calibrate_trade_model
from puremacro.trade._accounting import _parameters, evaluate as evaluate_consistent, postprocess
from puremacro.trade.equilibrium import _get_ces_weights, pack_equilibrium_vector
from puremacro.trade.solver import _resolve_tariffs
from puremacro.workbench import run_empirical_structural_workbench


# ---------------------------------------------------------------------------
# Deterministic synthetic fixtures / generators
# ---------------------------------------------------------------------------


def _make_cce_panel(
    *,
    seed: int = 42,
    n_units: int = 8,
    n_periods: int = 45,
    beta_true: float = 1.4,
    noise_sd: float = 0.25,
    homogeneous_loadings: bool = False,
) -> pd.DataFrame:
    """Build a multi-index factor-augmented panel with deterministic DGP."""
    rng = np.random.default_rng(seed)
    codes = [f"C{i:02d}" for i in range(n_units)]
    dates = pd.date_range("1995-01-01", periods=n_periods, freq="QS")
    idx = pd.MultiIndex.from_product([codes, dates], names=["code", "date"])
    f = rng.normal(size=n_periods)
    blocks: list[np.ndarray] = []
    for i in range(n_units):
        lam = 0.6 if homogeneous_loadings else (0.3 + 0.12 * i)
        x = rng.normal(size=n_periods) + 0.35 * f
        z = rng.normal(size=n_periods) + 0.20 * f
        eps = 0.0 if noise_sd == 0.0 else noise_sd * rng.normal(size=n_periods)
        y = (0.4 * i) + beta_true * x + 0.3 * z + lam * f + eps
        blocks.append(np.column_stack([y, x, z]))
    return pd.DataFrame(np.vstack(blocks), index=idx, columns=["y", "x", "z"])


def _make_trade_calibration(n_countries: int = 3, n_sectors: int = 2):
    """Construct a deterministic balanced consistent-accounting SAM calibration."""
    rng = np.random.default_rng(17)
    nc, ns, nfd = n_countries, n_sectors, 3
    n = nc * ns
    data = np.zeros((n + 3, n + nfd * nc), dtype=float)
    y = np.linspace(180.0, 260.0, n)
    inter = 8.0 + 12.0 * rng.random((n, n))
    # Keep intermediate absorption safely below gross output.
    inter *= 0.35 * y[None, :] / inter.sum(axis=0, keepdims=True)
    data[:n, :n] = inter
    va = y - inter.sum(axis=0)
    taxes = 0.04 * y
    va_fac = va - taxes
    data[n, :n] = taxes
    data[n + 1, :n] = 0.6 * va_fac
    data[n + 2, :n] = 0.4 * va_fac
    fd_rows = y - inter.sum(axis=1)
    for i in range(n):
        c_i = i // ns
        for c_dest in range(nc):
            share = 0.72 if c_dest == c_i else 0.28 / (nc - 1)
            data[i, n + c_dest * nfd : n + (c_dest + 1) * nfd] = (
                fd_rows[i] * share
            ) * np.array([0.60, 0.30, 0.10])
    data[n, n:] = 0.02 * data[:n, n:].sum(axis=0)
    return calibrate_trade_model(
        data,
        ns=ns,
        nc=nc,
        nfd=nfd,
        country_codes=[f"C{c:02d}" for c in range(nc)],
        sector_codes=[f"S{s}" for s in range(ns)],
        validate=True,
    )


def _reference_evaluate_naive(
    x: np.ndarray,
    calib,
    tau: np.ndarray | None,
    tau_fd: np.ndarray | None,
    tauf: np.ndarray | None,
    tauf_fd: np.ndarray | None,
    *,
    sigma: float = 0.0,
    foreign_saving_units: str = "numeraire",
) -> dict[str, np.ndarray]:
    """Unoptimized textbook reference matching original _accounting.evaluate logic."""
    nc, ns, nfd = calib.nc, calib.ns, calib.n_final_demand
    n = nc * ns
    inv = 1 if nfd >= 2 else 0
    if calib.data_calibra is not None:
        d = np.asarray(calib.data_calibra)
        basic0 = d[:n, n:].sum(0).reshape(nc, nfd).T[None]
        tax_amount = d[n, n:].reshape(nc, nfd).T[None]
        spending = basic0 + tax_amount
    else:
        income0 = (calib.l_endow + calib.k_endow).reshape(1, 1, nc)
        calibrated_rate = np.zeros_like(calib.theta) if calib.tax_fd is None else calib.tax_fd
        if calib.T is not None:
            income0 = income0 + np.asarray(calib.T).reshape(1, 1, nc)
        else:
            income0 = (income0 + (calib.tax * calib.ytot).sum(1)[:, None, :]) / (
                1.0 - (calibrated_rate * calib.theta).sum(1)[:, None, :]
            )
        allocated = calib.theta * income0
        tax_amount = calibrated_rate * allocated
        spending = allocated.copy()
        spending[:, inv, :] -= np.asarray(calib.invforT).reshape(1, nc)
        basic0 = spending - tax_amount
    fd_tax = np.divide(tax_amount, spending, out=np.zeros_like(spending), where=spending != 0)
    ta, tf, _, _ = _resolve_tariffs(calib, tau, tau_fd, tauf, tauf_fd)
    for schedule, shape in ((ta, (n, ns, nc)), (tf, (n, nfd, nc))):
        assert schedule.shape == shape and np.isfinite(schedule).all() and np.all(schedule > 0)
        for c in range(nc):
            assert np.allclose(schedule[c * ns : (c + 1) * ns, :, c], 1.0, rtol=0.0, atol=1e-14)
    from puremacro.trade.equilibrium import unpack_equilibrium_vector

    v = unpack_equilibrium_vector(x, ns=ns, nc=nc, nfd=nfd)
    p, y, r, w, T, B = v.p, v.y, v.r, v.w, v.T, v.invforT
    pv = p.ravel(order="F")
    purchaser_inter = pv[:, None, None] * ta
    if sigma == 0:
        inter_cost = np.sum(calib.a * purchaser_inter, axis=0)[None]
        Z = calib.a * y
    else:
        amount, weights = _get_ces_weights(calib)
        inner = np.sum(weights * purchaser_inter ** (1 - sigma), axis=0)
        index = np.where(amount > 0, inner, 1.0) ** (1 / (1 - sigma))
        inter_cost = (amount * index)[None]
        Z = calib.a * y * (index[None] / purchaser_inter) ** sigma
    unit_va = (r / calib.alpha) ** calib.alpha * (w / (1 - calib.alpha)) ** (1 - calib.alpha) / calib.beta
    pp = (unit_va + inter_cost) / (1 - calib.tax)
    labor = (1 - calib.alpha) * unit_va * y / w
    capital = calib.alpha * unit_va * y / r
    Q = np.sum(calib.afd * pv[:, None, None] * tf, axis=0)[None]
    Q = np.where(basic0 > 0, Q, 1.0)
    P = Q / (1 - fd_tax)
    income = w * calib.l_endow.reshape(1, 1, nc) + r * calib.k_endow.reshape(1, 1, nc) + T
    expenditure = calib.theta * income
    expenditure[:, inv, :] -= B
    quantity = expenditure / P
    F = calib.afd * quantity
    final_tax = fd_tax * expenditure
    production_tax = calib.tax * p * y
    duties_i = (ta - 1) * pv[:, None, None] * Z
    duties_f = (tf - 1) * pv[:, None, None] * F
    tariffs = duties_i.sum(axis=(0, 1)) + duties_f.sum(axis=(0, 1))
    government = production_tax.sum(axis=1).ravel() + final_tax.sum(axis=1).ravel() + tariffs
    zm = Z.reshape(n, n, order="F")
    fm = F.reshape(n, nfd * nc, order="F")
    producer_values = np.hstack([zm, fm]) * pv[:, None]
    trade = producer_values[:, :n].reshape(nc, ns, nc, ns).sum(axis=(1, 3))
    trade += producer_values[:, n:].reshape(nc, ns, nc, nfd).sum(axis=(1, 3))
    np.fill_diagonal(trade, 0.0)
    net_exports = trade.sum(1) - trade.sum(0)
    goods = y.ravel(order="F") - zm.sum(1) - fm.sum(1)
    prices = (pp - p).ravel(order="F")
    labor_gap = calib.l_endow.ravel() - labor.sum(1).ravel()
    capital_gap = calib.k_endow.ravel() - capital.sum(1).ravel()
    budget_gap = T.ravel() - government
    if foreign_saving_units == "numeraire":
        balances = v.XN - np.asarray(calib.invforT).ravel()[: nc - 1]
    else:
        base_world = float(np.sum(calib.l_endow) + np.sum(calib.k_endow))
        world = float(np.sum(w.ravel() * calib.l_endow.ravel() + r.ravel() * calib.k_endow.ravel()))
        balances = v.XN - np.asarray(calib.invforT).ravel()[: nc - 1] * (world / base_world)
    solver_goods = goods.copy()
    solver_goods[0] = pv[0] - 1.0
    residuals = np.concatenate([solver_goods, prices, labor_gap, capital_gap, balances, budget_gap])
    physical = np.concatenate([goods, prices, labor_gap, capital_gap, net_exports - B.ravel(), budget_gap])
    return {"residuals": residuals, "physical_residuals": physical, "tariffs": tariffs, "trade": trade}


# ---------------------------------------------------------------------------
# 1. HANK Fiscal Conservation & Zero-Shock Conformance
# ---------------------------------------------------------------------------


class TestHankFiscalConformance:
    """Conservation laws and baseline identities for ``puremacro.models.hank_fiscal``."""

    @pytest.mark.parametrize("closure", ["balanced_budget", "debt_rule"])
    @pytest.mark.parametrize("tax_type", ["proportional", "lump_sum"])
    def test_finite_horizon_walras_and_budget_identities(self, closure: str, tax_type: str) -> None:
        horizon = 300
        shock = 0.01 * (0.7 ** np.arange(horizon))
        res = solve_hank_fiscal(
            shock_seq=shock,
            closure=closure,
            tax_type=tax_type,
            horizon=horizon,
            phi_B=0.15,
            tol=1e-11,
        )
        assert res.converged
        ss = res.steady_state
        b_prev = np.concatenate([[ss["B_ss"]], res.B[:-1]])
        a_prev = np.concatenate([[ss["B_ss"]], res.A[:-1]])

        # Exact finite-horizon Walras identity: (Y_t - C_t - G_t) == (A_t - B_t) - (1 + r_t)(A_{t-1} - B_{t-1})
        lhs_goods_net = res.Y - res.C - res.G
        rhs_net_assets = (res.A - res.B) - (1.0 + res.r) * (a_prev - b_prev)
        assert np.max(np.abs(lhs_goods_net - rhs_net_assets)) < 1e-10
        assert np.max(np.abs(res.goods_market_residual)) < 1e-10
        assert np.max(np.abs(res.asset_market_residual)) < 1e-10

        # Exact flow government budget constraint: B_t = (1 + r_t) B_{t-1} + G_t - T_t
        gov_flow_gap = res.B - ((1.0 + res.r) * b_prev + res.G - res.T)
        assert np.max(np.abs(gov_flow_gap)) < 1e-10
        assert np.max(np.abs(res.gov_budget_residual)) < 1e-10
        assert abs(res.gov_intertemporal_residual) < 1e-10
        assert abs(res.terminal_debt_gap) < 1e-8

        if closure == "balanced_budget":
            assert np.max(np.abs(res.dB)) < 1e-12
        else:
            # Under positive fiscal feedback phi_B, transitory spending shock debt reverts toward 0.
            assert np.max(np.abs(res.dB)) > 1e-4
            assert abs(res.dB[-1]) < 1e-5 * np.max(np.abs(res.dB))

    @pytest.mark.parametrize("closure", ["balanced_budget", "debt_rule"])
    @pytest.mark.parametrize("tax_type", ["proportional", "lump_sum"])
    def test_zero_shock_exact_steady_state_baseline(self, closure: str, tax_type: str) -> None:
        horizon = 60
        zero_shock = np.zeros(horizon, dtype=float)
        res = solve_hank_fiscal(
            shock_seq=zero_shock,
            closure=closure,
            tax_type=tax_type,
            horizon=horizon,
            phi_B=0.10,
            tol=1e-11,
        )
        assert res.converged
        for dev in (res.dY, res.dC, res.dG, res.dT, res.dB):
            assert np.max(np.abs(dev)) < 1e-12
        ss = res.steady_state
        assert np.max(np.abs(res.Y - ss["Y_ss"])) < 1e-12
        assert np.max(np.abs(res.C - ss["C_ss"])) < 1e-12
        assert np.max(np.abs(res.G - ss["G_ss"])) < 1e-12
        assert np.max(np.abs(res.T - ss["T_ss"])) < 1e-12
        assert np.max(np.abs(res.B - ss["B_ss"])) < 1e-12


# ---------------------------------------------------------------------------
# 2. CCE Pooled & Mean Group Metamorphic & Slope Conformance
# ---------------------------------------------------------------------------


class TestCCEMeanGroupConformance:
    """Metamorphic invariances and exact slope recovery for ``puremacro.lp.cce``."""

    @pytest.mark.parametrize("estimator", ["mean_group", "pooled"])
    def test_unit_permutation_and_row_shuffle_invariance(
        self, estimator: Literal["mean_group", "pooled"]
    ) -> None:
        df = _make_cce_panel(seed=21, n_units=8, n_periods=45, noise_sd=0.25)
        horizons = [0, 1, 2, 3, 4]
        res_base = cce_panel_lp(
            df,
            y="y",
            x="x",
            horizons=horizons,
            n_lags=2,
            controls=["z"],
            estimator=estimator,
        )

        # Relabel entities via bijective permutation AND scramble row order.
        orig_codes = df.index.get_level_values("code").unique().tolist()
        perm = np.random.default_rng(99).permutation(len(orig_codes))
        relabel = {orig_codes[i]: f"REL_{perm[i]:02d}" for i in range(len(orig_codes))}
        new_codes = df.index.get_level_values("code").map(relabel)
        new_dates = df.index.get_level_values("date")
        df_perm = df.copy()
        df_perm.index = pd.MultiIndex.from_arrays([new_codes, new_dates], names=["code", "date"])
        df_perm = df_perm.sample(frac=1.0, random_state=123)

        res_perm = cce_panel_lp(
            df_perm,
            y="y",
            x="x",
            horizons=horizons,
            n_lags=2,
            controls=["z"],
            estimator=estimator,
        )

        for col in ("beta", "se", "lo", "hi"):
            assert np.allclose(res_perm[col], res_base[col], atol=1e-12, rtol=1e-12)

    @pytest.mark.parametrize("estimator", ["mean_group", "pooled"])
    def test_outcome_affine_scaling_and_unit_shift_equivariance(
        self, estimator: Literal["mean_group", "pooled"]
    ) -> None:
        df = _make_cce_panel(seed=33, n_units=8, n_periods=45, noise_sd=0.25)
        horizons = [0, 1, 2, 3]
        base = cce_panel_lp(
            df,
            y="y",
            x="x",
            horizons=horizons,
            n_lags=2,
            controls=["z"],
            estimator=estimator,
        )

        scale = 3.7
        df_scaled = df.copy()
        # Apply linear scaling plus distinct unit-specific additive intercepts (must cancel out).
        unit_codes = pd.factorize(df_scaled.index.get_level_values("code"))[0]
        df_scaled["y"] = scale * df_scaled["y"] + 12.5 * (unit_codes + 1)

        scaled = cce_panel_lp(
            df_scaled,
            y="y",
            x="x",
            horizons=horizons,
            n_lags=2,
            controls=["z"],
            estimator=estimator,
        )

        assert np.allclose(scaled["beta"], scale * base["beta"], atol=1e-11, rtol=1e-12)
        assert np.allclose(scaled["se"], scale * base["se"], atol=1e-11, rtol=1e-12)
        t_base = base["beta"] / base["se"]
        t_scaled = scaled["beta"] / scaled["se"]
        assert np.allclose(t_scaled, t_base, atol=1e-12, rtol=1e-12)

    def test_homogeneous_noiseless_slope_equality_mg_and_pooled(self) -> None:
        beta_true = 1.75
        df = _make_cce_panel(
            seed=7,
            n_units=12,
            n_periods=40,
            beta_true=beta_true,
            noise_sd=0.0,
            homogeneous_loadings=True,
        )
        res_mg = cce_panel_lp(
            df,
            y="y",
            x="x",
            horizons=[0],
            n_lags=1,
            controls=["z"],
            estimator="mean_group",
        )
        res_pooled = cce_panel_lp(
            df,
            y="y",
            x="x",
            horizons=[0],
            n_lags=1,
            controls=["z"],
            estimator="pooled",
        )
        assert abs(float(res_mg["beta"].iloc[0]) - beta_true) < 1e-12
        assert abs(float(res_pooled["beta"].iloc[0]) - beta_true) < 1e-12
        assert abs(float(res_mg["beta"].iloc[0]) - float(res_pooled["beta"].iloc[0])) < 1e-12


# ---------------------------------------------------------------------------
# 3. Empirical-Structural Workbench Metamorphic Conformance
# ---------------------------------------------------------------------------


class TestWorkbenchConformance:
    """Permutation and scaling equivariance for ``run_empirical_structural_workbench``."""

    @staticmethod
    def _make_workbench_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(11)
        dates = pd.date_range("1995-01-01", periods=50, freq="QS")
        idx = pd.MultiIndex.from_product(
            [["C0", "C1", "C2", "C3", "C4", "C5"], dates],
            names=["code", "date"],
        )
        n = len(idx)
        x = rng.normal(size=n)
        z = rng.normal(size=n)
        y = 0.8 * x + 0.3 * z + 0.4 * rng.normal(size=n)
        df = pd.DataFrame({"y": y, "x": x, "z": z}, index=idx)
        model_irf = pd.DataFrame({"h": [0, 1, 2, 3, 4], "beta": [0.8, 0.4, 0.2, 0.1, 0.0]})
        return df, model_irf

    def test_panel_row_and_unit_permutation_invariance(self) -> None:
        df, model_irf = self._make_workbench_inputs()
        dos_base = run_empirical_structural_workbench(
            df,
            y="y",
            x="x",
            horizons=[0, 1, 2, 3, 4],
            lags=1,
            controls=["z"],
            strict_preflight=False,
            run_sensitivity=False,
            model_irf=model_irf,
            build_moment_targets=False,
        )

        # Permute unit labels and shuffle row order in a flat DataFrame representation.
        relabel = {"C0": "Z5", "C1": "Z2", "C2": "Z0", "C3": "Z4", "C4": "Z1", "C5": "Z3"}
        flat = df.reset_index()
        flat["code"] = flat["code"].map(relabel)
        flat = flat.sample(frac=1.0, random_state=31).reset_index(drop=True)

        dos_perm = run_empirical_structural_workbench(
            flat,
            y="y",
            x="x",
            unit_col="code",
            time_col="date",
            horizons=[0, 1, 2, 3, 4],
            lags=1,
            controls=["z"],
            strict_preflight=False,
            run_sensitivity=False,
            model_irf=model_irf,
            build_moment_targets=False,
        )

        for col in ("beta", "se", "lo", "hi"):
            assert np.allclose(
                dos_perm.empirical_lp[col], dos_base.empirical_lp[col], atol=1e-12, rtol=1e-12
            )
        assert dos_base.irf_scorecard is not None
        assert dos_perm.irf_scorecard is not None
        assert np.isclose(dos_perm.irf_scorecard.rmse, dos_base.irf_scorecard.rmse, atol=1e-12, rtol=1e-12)
        assert np.isclose(dos_perm.irf_scorecard.mae, dos_base.irf_scorecard.mae, atol=1e-12, rtol=1e-12)
        assert dos_perm.irf_scorecard.sign_concordance == dos_base.irf_scorecard.sign_concordance

    def test_outcome_and_target_unit_scaling_equivariance(self) -> None:
        df, model_irf = self._make_workbench_inputs()
        dos_base = run_empirical_structural_workbench(
            df,
            y="y",
            x="x",
            horizons=[0, 1, 2, 3, 4],
            lags=1,
            controls=["z"],
            strict_preflight=False,
            run_sensitivity=False,
            model_irf=model_irf,
            build_moment_targets=True,
        )

        scale = 4.25
        df_scaled = df.copy()
        df_scaled["y"] = scale * df_scaled["y"]
        irf_scaled = model_irf.copy()
        irf_scaled["beta"] = scale * irf_scaled["beta"]
        dos_scaled = run_empirical_structural_workbench(
            df_scaled,
            y="y",
            x="x",
            horizons=[0, 1, 2, 3, 4],
            lags=1,
            controls=["z"],
            strict_preflight=False,
            run_sensitivity=False,
            model_irf=irf_scaled,
            build_moment_targets=True,
        )

        assert np.allclose(
            dos_scaled.empirical_lp["beta"], scale * dos_base.empirical_lp["beta"], atol=1e-11, rtol=1e-12
        )
        assert np.allclose(
            dos_scaled.empirical_lp["se"], scale * dos_base.empirical_lp["se"], atol=1e-11, rtol=1e-12
        )
        assert dos_base.irf_scorecard is not None
        assert dos_scaled.irf_scorecard is not None
        assert np.isclose(
            dos_scaled.irf_scorecard.rmse, scale * dos_base.irf_scorecard.rmse, atol=1e-11, rtol=1e-12
        )
        assert np.isclose(
            dos_scaled.irf_scorecard.mae, scale * dos_base.irf_scorecard.mae, atol=1e-11, rtol=1e-12
        )
        assert dos_scaled.irf_scorecard.sign_concordance == dos_base.irf_scorecard.sign_concordance
        assert dos_scaled.irf_scorecard.band_coverage == dos_base.irf_scorecard.band_coverage


# ---------------------------------------------------------------------------
# 4. Sovereign SMM Markov & Seed Determinism Conformance
# ---------------------------------------------------------------------------


class TestSovereignSMMConformance:
    """Markov row normalization and RNG seed determinism for ``puremacro.sovereign.smm``."""

    def test_markov_row_normalization_and_stationary_mass_conservation(self) -> None:
        theta = 0.282
        sol = solve_fiscal_sovereign_default(
            tau=0.25,
            psi=0.5,
            theta=theta,
            n_b=51,
            n_y=9,
            max_iter=1500,
            tol=1e-8,
        )
        assert sol.converged
        assert np.all(sol.p_trans >= 0.0)
        assert np.allclose(sol.p_trans.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)

        zero_idx = int(np.argmin(np.abs(sol.b_grid)))
        w_acc, w_exc = _stationary_access_exclusion_dist(
            sol.default_prob, sol.policy_index, sol.p_trans, theta, zero_idx
        )
        assert np.all(w_acc >= 0.0)
        assert np.all(w_exc >= 0.0)
        assert abs(float(w_acc.sum() + w_exc.sum()) - 1.0) < 1e-12

    def test_simulation_seed_determinism_under_identical_budgets(self) -> None:
        sol = solve_fiscal_sovereign_default(
            tau=0.25,
            psi=0.5,
            theta=0.282,
            n_b=51,
            n_y=9,
            max_iter=1500,
            tol=1e-8,
        )
        labels = ("debt_pct", "spread_mean_pct", "annual_default_pct", "spending_pct")
        seeds_a = (101, 202, 303)
        seeds_b = (404, 505, 606)

        m_a1 = _compute_stationary_or_simulated_moments(
            sol, labels, use_simulation=True, seeds=seeds_a, periods=250, burn=50
        )
        m_a2 = _compute_stationary_or_simulated_moments(
            sol, labels, use_simulation=True, seeds=seeds_a, periods=250, burn=50
        )
        m_b = _compute_stationary_or_simulated_moments(
            sol, labels, use_simulation=True, seeds=seeds_b, periods=250, burn=50
        )

        assert np.array_equal(m_a1, m_a2)
        assert np.all(np.isfinite(m_a1))
        assert np.max(np.abs(m_a1 - m_b)) > 1e-6

        # Also verify low-level draw_uniforms -> simulate_paths -> ergodic_moments determinism.
        u1 = draw_uniforms(seeds_a, 200)
        u2 = draw_uniforms(seeds_a, 200)
        assert np.array_equal(u1, u2)
        p1 = simulate_paths(sol, b_index=len(sol.b_grid) // 2, y_index=len(sol.y_grid) // 2, uniforms=u1)
        p2 = simulate_paths(sol, b_index=len(sol.b_grid) // 2, y_index=len(sol.y_grid) // 2, uniforms=u2)
        em1 = ergodic_moments(p1, burn=40)
        em2 = ergodic_moments(p2, burn=40)
        assert em1 == em2


# ---------------------------------------------------------------------------
# 5. Optimization Equivalence Regression Tests
# ---------------------------------------------------------------------------


class TestOptimizedHotspotEquivalence:
    """Exact numerical equivalence (< 1e-12) between optimized fast paths and textbook references."""

    @pytest.mark.parametrize("sigma", [0.0, 1.5])
    @pytest.mark.parametrize("foreign_saving_units", ["numeraire", "world_income"])
    def test_trade_accounting_evaluate_matches_reference_and_cache_safety(
        self, sigma: float, foreign_saving_units: str
    ) -> None:
        calib = _make_trade_calibration(n_countries=3, n_sectors=2)
        nc, ns, nfd = calib.nc, calib.ns, calib.n_final_demand
        n = nc * ns
        x = pack_equilibrium_vector(
            np.full((ns, nc), 1.03),
            calib.ytot[0] * 1.01,
            np.full((1, 1, nc), 0.99),
            np.full((1, 1, nc), 1.02),
            np.full((1, 1, nc), 1.5),
            np.full(nc - 1, 0.2),
        )
        tau = np.full((n, ns, nc), 1.08)
        for c in range(nc):
            tau[c * ns : (c + 1) * ns, :, c] = 1.0
        tfd = np.full((n, nfd, nc), 1.04)
        for c in range(nc):
            tfd[c * ns : (c + 1) * ns, :, c] = 1.0

        # Cold evaluation + warm cached evaluation must agree identically with unoptimized reference.
        out_cold = evaluate_consistent(
            x, calib, tau, tfd, None, None, sigma=sigma, foreign_saving_units=foreign_saving_units
        )
        out_warm = evaluate_consistent(
            x, calib, tau, tfd, None, None, sigma=sigma, foreign_saving_units=foreign_saving_units
        )
        ref = _reference_evaluate_naive(
            x, calib, tau, tfd, None, None, sigma=sigma, foreign_saving_units=foreign_saving_units
        )

        for key in ("residuals", "physical_residuals", "tariffs", "trade"):
            assert np.max(np.abs(out_cold[key] - ref[key])) < 1e-12
            assert np.array_equal(out_cold[key], out_warm[key])

        # Shallow copy or field replacement must never read stale _consistent_params_cache.
        calib_copy = copy.copy(calib)
        assert calib.data_calibra is not None
        d2 = calib.data_calibra.copy()
        d2[n, n:] *= 1.5
        calib_replaced = replace(calib, data_calibra=d2)
        s0, b0, _ = _parameters(calib)
        s_copy, b_copy, _ = _parameters(calib_copy)
        s_rep, b_rep, _ = _parameters(calib_replaced)
        assert np.array_equal(s0, s_copy)
        assert np.array_equal(b0, b_copy)
        assert not np.array_equal(s0, s_rep)
        # Mutable namespace attribute rebinding must invalidate cached parameters cleanly.
        ns_calib = SimpleNamespace(
            nc=nc, ns=ns, n_final_demand=nfd, a=calib.a, afd=calib.afd,
            alpha=calib.alpha, beta=calib.beta, tax=calib.tax, theta=calib.theta,
            ytot=calib.ytot, l_endow=calib.l_endow, k_endow=calib.k_endow,
            invforT=calib.invforT.copy(), data_calibra=None,
            tax_fd=calib.tax_fd, T=calib.T,
        )
        _, b_ns1, _ = _parameters(ns_calib)
        ns_calib.invforT = np.zeros_like(calib.invforT)
        _, b_ns2, _ = _parameters(ns_calib)
        assert not np.array_equal(b_ns1, b_ns2)

        # Mutating returned fd_tax / basic0 arrays must not poison calib cache.
        out_poison = evaluate_consistent(
            x, calib, tau, tfd, None, None, sigma=sigma, foreign_saving_units=foreign_saving_units
        )
        out_poison["fd_tax"][:] = 0.9
        out_poison["basic0"][:] = -999.0
        out_after_poison = evaluate_consistent(
            x, calib, tau, tfd, None, None, sigma=sigma, foreign_saving_units=foreign_saving_units
        )
        assert np.array_equal(out_after_poison["residuals"], out_warm["residuals"])
        assert np.array_equal(out_after_poison["fd_tax"], out_warm["fd_tax"])
        assert np.array_equal(out_after_poison["basic0"], out_warm["basic0"])

        # In-place mutation of calib input arrays must invalidate cache and re-validate.
        calib_inplace = copy.deepcopy(calib)
        assert calib_inplace.data_calibra is not None
        out_pre_edit = evaluate_consistent(
            x, calib_inplace, tau, tfd, None, None, sigma=sigma, foreign_saving_units=foreign_saving_units
        )
        calib_inplace.data_calibra[0, n] += 10.0
        calib_inplace.data_calibra[n, n] += 1.0
        out_post_edit = evaluate_consistent(
            x, calib_inplace, tau, tfd, None, None, sigma=sigma, foreign_saving_units=foreign_saving_units
        )
        assert np.max(np.abs(out_post_edit["basic0"] - out_pre_edit["basic0"])) > 1.0
        assert np.max(np.abs(out_post_edit["residuals"] - out_pre_edit["residuals"])) > 1e-4
        calib_inplace.data_calibra[:] = np.nan
        with pytest.raises(ValueError, match="finite calibration table"):
            evaluate_consistent(
                x, calib_inplace, tau, tfd, None, None, sigma=sigma, foreign_saving_units=foreign_saving_units
            )

        # Invalid off-one domestic tariff still raises ValueError.
        bad_tau = tau.copy()
        bad_tau[0, 0, 0] = 1.01
        with pytest.raises(ValueError, match="domestic transactions"):
            evaluate_consistent(x, calib, bad_tau, tfd, None, None, sigma=sigma)

        pp = postprocess(out_warm, calib)
        assert abs(float(pp["max_residual"]) - float(np.max(np.abs(ref["residuals"])))) < 1e-14

    @pytest.mark.parametrize("use_int_codes", [False, True])
    def test_cce_fast_paths_match_naive_pandas_reference(self, use_int_codes: bool) -> None:
        df = _make_cce_panel(seed=55, n_units=8, n_periods=35, noise_sd=0.25)
        if use_int_codes:
            codes = (pd.factorize(df.index.get_level_values("code"))[0] + 10).astype(int)
            dates = df.index.get_level_values("date")
            df.index = pd.MultiIndex.from_arrays([codes, dates], names=["code", "date"])

        # Scramble rows before testing internal helpers directly as well as end-to-end cce_panel_lp.
        df_sorted = df.sort_index()
        x_cols = ["x", "z"]
        out_fe = _entity_fe_ols(df_sorted.sample(frac=1.0, random_state=7), y_col="y", x_cols=x_cols, entity_level="code")

        # Textbook pandas within-OLS + entity-clustered SE reference
        arr = df_sorted[["y"] + x_cols].to_numpy(dtype=float)
        ent_idx = df_sorted.index.get_level_values("code").to_numpy()
        demeaned = arr - pd.DataFrame(arr).groupby(ent_idx).transform("mean").to_numpy()
        y_d, X_d = demeaned[:, 0], demeaned[:, 1:]
        beta_ref, *_ = np.linalg.lstsq(X_d, y_d, rcond=None)
        u_ref = y_d - X_d @ beta_ref
        score_ref = X_d * u_ref[:, None]
        S_ref = np.zeros((X_d.shape[1], X_d.shape[1]))
        for c in np.unique(ent_idx):
            s_c = score_ref[ent_idx == c].sum(axis=0)
            S_ref += np.outer(s_c, s_c)
        xtx_inv = np.linalg.pinv(X_d.T @ X_d)
        se_ref = np.sqrt(np.maximum(np.diag(xtx_inv @ S_ref @ xtx_inv), 0.0))

        assert np.max(np.abs(out_fe["beta"] - beta_ref)) < 1e-12
        assert np.max(np.abs(out_fe["se"] - se_ref)) < 1e-12

        # Textbook Mean Group reference across out-of-order rows with 1 absent unit warning
        unique_entities = df_sorted.index.get_level_values("code").unique()
        sub_partial = df_sorted[df_sorted.index.get_level_values("code") != unique_entities[-1]].sample(
            frac=1.0, random_state=19
        )
        with pytest.warns(UserWarning, match="excluded entity"):
            row_mg, obs_mg, n_used_mg, excl_mg = _estimate_cce_mean_group_horizon(
                sub=sub_partial,
                col_dy="y",
                x_cols=x_cols,
                unique_entities=unique_entities,
                entity_level="code",
                h=0,
                z_crit=1.6448536269514722,
                min_obs_int=10,
                func_name="cce_panel_lp",
            )
        ref_betas: list[float] = []
        for ent in unique_entities[:-1]:
            g = sub_partial.xs(ent, level="code")
            Z = np.column_stack([np.ones(len(g)), g[x_cols].to_numpy(dtype=float)])
            coef, *_ = np.linalg.lstsq(Z, g["y"].to_numpy(dtype=float), rcond=None)
            ref_betas.append(float(coef[1]))
        assert n_used_mg == len(ref_betas)
        assert obs_mg == len(sub_partial)
        assert unique_entities[-1] in excl_mg
        assert abs(row_mg["beta"] - float(np.mean(ref_betas))) < 1e-12
        assert abs(row_mg["se"] - float(np.sqrt(np.var(ref_betas, ddof=1) / len(ref_betas)))) < 1e-12
