"""Game-Theoretic Terms-of-Trade Optimal Tariffs and Multilateral Nash Equilibrium Engine.

This module implements the non-cooperative game-theoretic trade policy engine
for puremacro.trade, satisfying Requirement R3 and the Priority User Directive:
    1. Sovereign National Welfare Evaluation (:func:`evaluate_national_welfare`):
       Micro-founded welfare metrics supporting Multilateral Geary-Khamis PPP real GDP,
       Equivalent Variation (EV) consumer utility, and Terms-of-Trade (TOT) indices.
    2. Unilateral Terms-of-Trade Optimal Tariffs (:func:`compute_unilateral_optimal_tariff`):
       Optimal tariff schedule maximizing sovereign domestic welfare against passive
       trading partners, accelerated by the condensed Schur complement solver.
    3. Multilateral Non-Cooperative Nash Tariffs (:func:`solve_multilateral_nash_tariffs`):
       Simultaneous Nash tariff equilibrium across sovereign powers (USA, CHN, EUR,
       CAN, MEX) using damped Gauss-Seidel best-response iteration and gradient optimization.
    4. Strategic Welfare Payoff Matrices (:func:`compute_welfare_payoff_matrix`):
       Normal-form Prisoner's Dilemma trade war games evaluating mutual cooperation,
       unilateral defection, and retaliatory trade wars.
    5. Geopolitical 2024–2026 Real-World Policy Benchmarks (:func:`benchmark_real_world_tariffs`):
       Empirical benchmarking comparing theoretical Nash equilibria against universal
       10% tariffs, 60% China tariffs, 25% USMCA border duties, and foreign retaliations.

Conforms strictly to the puremacro Pyodide runtime contract: pure NumPy/SciPy/Pandas,
zero non-stdlib dependencies in the runtime path, fully vectorized.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import time
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from puremacro.trade._results import (
    GearyKhamisResult,
    NashTariffResult,
    OptimalTariffResult,
    TradeCalibrationResult,
    TradeEquilibriumResult,
    WelfarePayoffMatrixResult,
)
from puremacro.trade.data import (
    CANONICAL_COUNTRY_CODES,
    EU_COUNTRY_CODES,
    get_country_codes,
    get_eu_country_codes,
)
from puremacro.trade.geary_khamis import (
    compute_geary_khamis,
    restore_capital_formation,
    solve_multilateral_ppp,
)
from puremacro.trade.postprocessing import postprocess_trade_equilibrium
from puremacro.trade.scenarios import TariffScenario, build_tariff_matrices
from puremacro.trade.solver import build_initial_guess, solve_trade_equilibrium


# ==============================================================================
# Helper Utilities: Country and Regional Bloc Resolution
# ==============================================================================

def resolve_country_indices(
    calib: TradeCalibrationResult,
    country_ident: int | str | Sequence[str | int],
) -> list[int]:
    """Resolve country code(s) or regional bloc into integer indices within calib.

    Supports individual country codes (e.g. 'USA', 'CHN', 'CAN', 'MEX'),
    regional customs unions ('EUR' or 'EU' targeting all 27 EU member states),
    direct integer indices, or sequences thereof.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model containing country codes and dimensions.
    country_ident : int, str, or Sequence
        Identifier of country, bloc, or list of countries.

    Returns
    -------
    list[int]
        Sorted list of integer indices in 0 .. nc - 1.
    """
    country_codes = list(calib.country_codes) if calib.country_codes else list(CANONICAL_COUNTRY_CODES)
    nc = calib.n_countries

    if isinstance(country_ident, int):
        if 0 <= country_ident < nc:
            return [country_ident]
        raise IndexError(f"Country index {country_ident} out of range (0 <= idx < {nc}).")

    if isinstance(country_ident, str):
        ident_upper = country_ident.strip().upper()
        if ident_upper in ("EUR", "EU", "EU27", "EU-27"):
            eu_codes = set(EU_COUNTRY_CODES)
            indices = [i for i, c in enumerate(country_codes) if c in eu_codes]
            if indices:
                return sorted(indices)
            # Fallback if EU codes not distinct
            if "EUR" in country_codes:
                return [country_codes.index("EUR")]
            return []

        if ident_upper in country_codes:
            return [country_codes.index(ident_upper)]
        # Check if integer string
        try:
            val = int(ident_upper)
            if 0 <= val < nc:
                return [val]
        except ValueError:
            pass
        raise ValueError(f"Unknown country or bloc identifier: '{country_ident}'.")

    # Sequence of identifiers
    out_indices: set[int] = set()
    for item in country_ident:
        out_indices.update(resolve_country_indices(calib, item))
    return sorted(out_indices)


def get_country_code(calib: TradeCalibrationResult, country_idx: int | str) -> str:
    """Return canonical country code string for a given index or identifier."""
    country_codes = list(calib.country_codes) if calib.country_codes else list(CANONICAL_COUNTRY_CODES)
    if isinstance(country_idx, str):
        return country_idx.strip().upper()
    if 0 <= country_idx < len(country_codes):
        return country_codes[country_idx]
    return f"C{country_idx:02d}"


# ==============================================================================
# 1. Sovereign National Welfare Objective Functions
# ==============================================================================

def evaluate_national_welfare(
    calib: TradeCalibrationResult,
    p_sol: Any = None,
    y_sol: np.ndarray | None = None,
    r_sol: np.ndarray | None = None,
    w_sol: np.ndarray | None = None,
    T_sol: np.ndarray | None = None,
    XN_sol: np.ndarray | None = None,
    country_idx: int | str = "USA",
    metric: str = "geary_khamis",
    *,
    base_welfare: float | None = None,
    base_equilibrium: TradeEquilibriumResult | None = None,
    consumption_categories: Sequence[int] = (0,),
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    replicate_matlab_precedence: bool = True,
    **kwargs: Any,
) -> float:
    """Evaluate national economic welfare for a sovereign economy or regional bloc.

    Supports historical policy-objective metrics:
    1. ``'geary_khamis'`` (or ``'gk'``): Multilateral Geary-Khamis Purchasing Power
       Parity (PPP) real GDP in international reference dollars:
       :math:`W_c = \\sum_m \\pi_m q_{mc}`, valuing absorption at world shadow prices.
    2. ``'equivalent_variation'`` (or ``'ev'``): historical Cobb-Douglas utility
       proxy (despite the alias, this is not money-metric Hicksian EV):
       :math:`U_c = Y_c^{\\text{disp}} / P_c^{\\text{CPI}}`.
    3. ``'terms_of_trade'`` (or ``'tot'``): Sovereign terms-of-trade index:
       :math:`\\text{TOT}_c = P_{X, c} / P_{M, c}`.
    4. ``'harberger'``: Terms-of-trade effect plus tariff revenue minus deadweight loss.

    ``metric="hicksian_ev"`` evaluates expenditure-function consumption EV in
    calibration value units. It requires consistent baseline and counterfactual
    equilibria, and sums monetary EV across bloc members. Select consumption
    baskets with ``consumption_categories`` (default ``(0,)``); investment is
    excluded. Legacy aliases above remain unchanged. For CV and attribution,
    use :func:`puremacro.trade.compute_hicksian_welfare`.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters. Can also be a
        :class:`~puremacro.trade._results.TradeEquilibriumResult` if passed as first arg.
    p_sol : TradeEquilibriumResult or np.ndarray, optional
        Solved gross output price tensor, or full :class:`TradeEquilibriumResult`.
    y_sol : np.ndarray, optional
        Solved gross output quantities.
    r_sol : np.ndarray, optional
        Solved capital rental rates.
    w_sol : np.ndarray, optional
        Solved wage rates.
    T_sol : np.ndarray, optional
        Solved government transfers / tax revenues.
    XN_sol : np.ndarray, optional
        Solved net foreign transfers (current account balances).
    country_idx : int or str, default 'USA'
        Target country index, ISO-3 code (e.g. 'USA', 'CHN'), or regional bloc ('EUR').
    metric : {'hicksian_ev', 'geary_khamis', 'equivalent_variation', 'terms_of_trade', 'harberger'}, default 'geary_khamis'
        Economic welfare objective function metric.
    base_welfare : float, optional
        Baseline welfare value for calculating relative percentage welfare changes.
    base_equilibrium : TradeEquilibriumResult, optional
        Benchmark equilibrium result for price and terms of trade indexing.
        Required as an explicit fixed comparison state for ``hicksian_ev``.
    consumption_categories : Sequence[int], default (0,)
        Final-use categories in Hicksian consumption utility; excludes investment.

    Returns
    -------
    float
        Evaluated sovereign welfare level (or index).
    """
    # Ergonomic dispatch: allow passing TradeEquilibriumResult as first or second argument
    eq: TradeEquilibriumResult | None = None
    if isinstance(calib, TradeEquilibriumResult):
        eq = calib
        calib_arg = kwargs.get("calib")
        if calib_arg is not None and isinstance(calib_arg, TradeCalibrationResult):
            calib = calib_arg
        elif p_sol is not None and isinstance(p_sol, TradeCalibrationResult):
            calib = p_sol
        else:
            raise ValueError("TradeCalibrationResult required when equilibrium is passed as first argument.")
    elif isinstance(p_sol, TradeEquilibriumResult):
        eq = p_sol
    elif hasattr(p_sol, "p_sol") and hasattr(p_sol, "y_sol"):
        eq = p_sol  # Duck-typing TradeEquilibriumResult

    nc = calib.n_countries
    ns = calib.n_sectors
    nfd = calib.n_final_demand
    c_indices = resolve_country_indices(calib, country_idx)
    if not c_indices:
        raise ValueError(f"No matching countries found for country_idx={country_idx}.")

    metric_clean = str(metric).lower().strip()
    if metric_clean == "hicksian_ev":
        from .welfare import compute_hicksian_welfare
        if eq is None or base_equilibrium is None:
            raise ValueError("Hicksian EV requires solved consistent states and an explicit fixed baseline")
        return float(sum(compute_hicksian_welfare(
            calib, eq, base_result=base_equilibrium, target_country=i,
            consumption_categories=consumption_categories).ev for i in c_indices))

    # Fast path when TradeEquilibriumResult is provided
    if eq is not None:
        p_vec = eq.p_sol
        y_vec = eq.y_sol
        r_vec = eq.r_sol
        w_vec = eq.w_sol
        T_vec = eq.T_sol
        XN_vec = eq.XN_sol
        c_fd = eq.c_fd if eq.c_fd is not None else eq.c_sol
        p_fd = eq.p_fd if eq.p_fd is not None else eq.pfd_sol
        cpi = eq.cpi
        tot = eq.terms_of_trade
    else:
        # Extract and reshape raw arrays
        if p_sol is None or y_sol is None or r_sol is None or w_sol is None or T_sol is None or XN_sol is None:
            raise ValueError("All state arrays (p, y, r, w, T, XN) are required when not passing TradeEquilibriumResult.")
        p_vec = np.asarray(p_sol, dtype=float).reshape((1, ns, nc))
        y_vec = np.asarray(y_sol, dtype=float).reshape((1, ns, nc))
        r_vec = np.asarray(r_sol, dtype=float).reshape((1, 1, nc))
        w_vec = np.asarray(w_sol, dtype=float).reshape((1, 1, nc))
        T_vec = np.asarray(T_sol, dtype=float).reshape((1, 1, nc))
        XN_vec = np.asarray(XN_sol, dtype=float).ravel()

        # Construct purchaser prices and absorption volumes
        tau_a = np.ones((ns * nc, ns, nc), dtype=float) if tau is None else (
            tau.transpose(1, 0, 2, 3).reshape(ns * nc, ns, nc) if tau.ndim == 4 else tau
        )
        taufd_a = np.ones((ns * nc, nfd, nc), dtype=float) if tau_fd is None else (
            tau_fd.transpose(1, 0, 2, 3).reshape(ns * nc, nfd, nc) if tau_fd.ndim == 4 else tau_fd
        )

        p_flat = p_vec.flatten(order="F")
        p_fd = np.tensordot(p_flat, calib.afd * taufd_a, axes=(0, 0))[np.newaxis, :, :]
        Ycon = w_vec * calib.l_endow + r_vec * calib.k_endow + T_vec
        theta = calib.theta if calib.theta is not None else (np.ones((1, nfd, nc)) / float(nfd))
        c_fd = theta * Ycon / np.maximum(p_fd, 1e-12)
        cpi = None
        tot = None

    # --------------------------------------------------------------------------
    # 1. Geary-Khamis PPP Real GDP
    # --------------------------------------------------------------------------
    if metric_clean in ("geary_khamis", "gk", "real_gdp", "ppp"):
        if eq is not None and "gk_real_gdp" in eq.metadata:
            real_gdp = np.asarray(eq.metadata["gk_real_gdp"], dtype=float)
        else:
            q = restore_capital_formation(c_fd, XN_vec, matlab_compat=True)
            p_use = p_fd.reshape((nfd, nc)) if p_fd.ndim == 3 else p_fd
            q_use = q.reshape((nfd, nc)) if q.ndim == 3 else q
            pi, ppp, _, _ = solve_multilateral_ppp(p=p_use, q=q_use, method="linear")
            real_gdp = np.sum(q_use * pi[:, np.newaxis], axis=0)
            if eq is not None:
                eq.metadata["gk_real_gdp"] = real_gdp

        total_welfare = float(np.sum([real_gdp[i] for i in c_indices]))
        return total_welfare

    # --------------------------------------------------------------------------
    # 2. Equivalent Variation / Cobb-Douglas Real Consumption Utility
    # --------------------------------------------------------------------------
    if metric_clean in ("equivalent_variation", "ev", "utility", "consumption"):
        if eq is not None and "ev_utility" in eq.metadata:
            utility = np.asarray(eq.metadata["ev_utility"], dtype=float)
        else:
            # Disposable income: w*L + r*K + T
            l_arr = np.asarray(calib.l_endow, dtype=float).reshape((1, 1, nc))
            k_arr = np.asarray(calib.k_endow, dtype=float).reshape((1, 1, nc))
            y_disp = (w_vec * l_arr + r_vec * k_arr + T_vec).ravel()

            theta = calib.theta if calib.theta is not None else (np.ones((1, nfd, nc)) / float(nfd))
            theta_use = theta.reshape((nfd, nc)) if theta.ndim == 3 else theta
            pfd_use = p_fd.reshape((nfd, nc)) if p_fd.ndim == 3 else p_fd

            # True Cobb-Douglas cost of living index: prod (p_m / theta_m)^theta_m
            with np.errstate(divide="ignore", invalid="ignore"):
                base_ratio = np.where(theta_use > 0, pfd_use / np.maximum(theta_use, 1e-12), 1.0)
                cpi_cd = np.exp(np.sum(theta_use * np.log(np.maximum(base_ratio, 1e-12)), axis=0))

            # Real consumption utility U = Y_disp / P_cpi
            utility = y_disp / np.maximum(cpi_cd, 1e-12)
            if eq is not None:
                eq.metadata["ev_utility"] = utility

        total_utility = float(np.sum([utility[i] for i in c_indices]))
        return total_utility

    # --------------------------------------------------------------------------
    # 3. Terms of Trade Index
    # --------------------------------------------------------------------------
    if metric_clean in ("terms_of_trade", "tot", "terms_trade"):
        if tot is not None:
            tot_arr = np.asarray(tot, dtype=float).ravel()
            return float(np.mean([tot_arr[i] for i in c_indices]))

        # Calculate terms of trade via post-processing
        eq_temp = postprocess_trade_equilibrium(
            x_sol=np.concatenate([
                np.log(np.maximum(p_vec.flatten(order="F"), 1e-12)),
                np.log(np.maximum(y_vec.flatten(order="F"), 1e-12)),
                np.log(np.maximum(r_vec.ravel(), 1e-12)),
                np.log(np.maximum(w_vec.ravel(), 1e-12)),
                T_vec.flatten(order="F"),
                XN_vec[: nc - 1],
            ]),
            calib=calib,
            tau=tau,
            tau_fd=tau_fd,
        )
        tot_arr = np.asarray(eq_temp.terms_of_trade, dtype=float).ravel()
        return float(np.mean([tot_arr[i] for i in c_indices]))

    # --------------------------------------------------------------------------
    # 4. Harberger Welfare Decomposition (TOT + Revenue - DWL)
    # --------------------------------------------------------------------------
    if metric_clean in ("harberger", "dwl_tot"):
        from puremacro.trade.extensions import decompose_harberger_welfare
        if eq is None:
            eq = postprocess_trade_equilibrium(
                x_sol=np.concatenate([
                    np.log(np.maximum(p_vec.flatten(order="F"), 1e-12)),
                    np.log(np.maximum(y_vec.flatten(order="F"), 1e-12)),
                    np.log(np.maximum(r_vec.ravel(), 1e-12)),
                    np.log(np.maximum(w_vec.ravel(), 1e-12)),
                    T_vec.flatten(order="F"),
                    XN_vec[: nc - 1],
                ]),
                calib=calib,
                tau=tau,
                tau_fd=tau_fd,
            )
        harb = decompose_harberger_welfare(
            calib=calib,
            eq_result=eq,
            base_result=base_equilibrium,
            target_country=get_country_code(calib, country_idx),
        )
        return float(harb["net_welfare_change"])

    raise ValueError(
        f"Unknown welfare metric '{metric}'. Valid options are: 'hicksian_ev', 'geary_khamis', 'equivalent_variation', 'terms_of_trade', 'harberger'."
    )


# ==============================================================================
# Helper: Build 4D Strategic Tariff Matrices
# ==============================================================================

def build_strategic_tariffs(
    calib: TradeCalibrationResult,
    player_tariffs: Mapping[str, float | Mapping[str, float]],
    policy_mode: str = "universal",
    target_countries: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct 4D intermediate and final demand tariff multiplier tensors.

    Supports universal import tariffs and bilateral strategic schedules across
    players, correctly handling EU customs union external border enforcement.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    player_tariffs : Mapping[str, float | Mapping[str, float]]
        Tariff schedules set by sovereign players.
        Example 1 (universal): ``{'USA': 0.18, 'CHN': 0.15, 'EUR': 0.16}``
        Example 2 (bilateral): ``{'USA': {'CHN': 0.25, 'CAN': 0.10}, 'CHN': {'USA': 0.20}}``
    policy_mode : str, default 'universal'
        Policy mode ('universal', 'bilateral', 'sectoral').
    target_countries : Sequence[str], optional
        Optional global target list of foreign partners subject to tariffs.

    Returns
    -------
    tau : np.ndarray, shape (ns, nc, ns, nc)
        4D intermediate tariff multipliers: 1 + rate.
    tau_fd : np.ndarray, shape (ns, nc, nfd, nc)
        4D final demand tariff multipliers: 1 + rate.
    """
    nc = calib.n_countries
    ns = calib.n_sectors
    nfd = calib.n_final_demand
    country_codes = list(calib.country_codes) if calib.country_codes else list(CANONICAL_COUNTRY_CODES)
    eu_set = set(EU_COUNTRY_CODES)

    tau = np.ones((ns, nc, ns, nc), dtype=float)
    tau_fd = np.ones((ns, nc, nfd, nc), dtype=float)

    for player, schedule in player_tariffs.items():
        try:
            importer_indices = resolve_country_indices(calib, player)
        except (ValueError, IndexError):
            continue
        is_eu_player = (player.strip().upper() in ("EUR", "EU", "EU27", "EU-27"))

        if isinstance(schedule, (int, float)):
            rate = float(schedule)
            if rate <= 0.0:
                continue
            mult = 1.0 + rate

            # Universal tariff on all non-exempt origins
            for d_idx in importer_indices:
                d_is_eu = (country_codes[d_idx] in eu_set)
                for o_idx in range(nc):
                    if o_idx == d_idx:
                        continue
                    # Intra-EU trade is duty-free
                    if is_eu_player and country_codes[o_idx] in eu_set:
                        continue
                    if d_is_eu and country_codes[o_idx] in eu_set:
                        continue

                    # If target_countries specified, filter
                    if target_countries is not None:
                        if country_codes[o_idx] not in target_countries:
                            continue

                    if policy_mode != "final_only":
                        tau[:, o_idx, :, d_idx] = mult
                    if policy_mode != "intermediate_only":
                        tau_fd[:, o_idx, :, d_idx] = mult

        elif isinstance(schedule, Mapping):
            # Bilateral schedule: partner -> rate
            for d_idx in importer_indices:
                d_is_eu = (country_codes[d_idx] in eu_set)
                for partner, p_rate in schedule.items():
                    rate = float(p_rate)
                    if rate <= 0.0:
                        continue
                    mult = 1.0 + rate
                    try:
                        partner_indices = resolve_country_indices(calib, partner)
                    except (ValueError, IndexError):
                        continue
                    for o_idx in partner_indices:
                        if o_idx == d_idx:
                            continue
                        if is_eu_player and country_codes[o_idx] in eu_set:
                            continue
                        if d_is_eu and country_codes[o_idx] in eu_set:
                            continue

                        if policy_mode != "final_only":
                            tau[:, o_idx, :, d_idx] = mult
                        if policy_mode != "intermediate_only":
                            tau_fd[:, o_idx, :, d_idx] = mult

    # Guarantee domestic diagonal is exactly 1.0
    for k in range(nc):
        tau[:, k, :, k] = 1.0
        tau_fd[:, k, :, k] = 1.0

    return tau, tau_fd


# ==============================================================================
# 2. Unilateral Optimal Tariff Optimization
# ==============================================================================

def _solve_policy_equilibrium(calib: TradeCalibrationResult, **kwargs: Any) -> TradeEquilibriumResult:
    """Legacy schedule-revenue solve; Hicksian policy uses policy_solver instead."""
    kwargs.setdefault("tol", 1e-8)
    result = solve_trade_equilibrium(calib, tariff_revenue_mode="schedule", **kwargs)
    if not result.converged or not np.isfinite(result.max_residual):
        raise RuntimeError(f"Policy equilibrium did not converge (residual={result.max_residual})")
    return result


def compute_unilateral_optimal_tariff(
    calib: TradeCalibrationResult,
    country_idx: int | str = "USA",
    target_countries: Sequence[str] | None = None,
    metric: str = "geary_khamis",
    tariff_max: float = 0.50,
    num_grid: int = 25,
    tol: float = 1e-4,
    x0: np.ndarray | None = None,
    method: str = "bounded",
    policy_mode: str = "universal",
    **kwargs: Any,
) -> OptimalTariffResult:
    r"""Compute the unilateral terms-of-trade optimal tariff for a sovereign economy.

    Solves the sovereign optimization problem:
    .. math::
        \tau^* = \arg\max_{\tau \in [0, \bar{\tau}]} W_c(\tau, \boldsymbol{0}_{-c})
        \quad \text{s.t.} \quad \mathbf{F}(\mathbf{x}^*; \tau) = \mathbf{0}

    With ``metric="hicksian_ev"``, use one audited zero-tariff baseline, a
    full-interval grid and (for ``bounded``) refinement of every sampled local
    peak. All equilibrium solves use consistent accounting and audited recovery.
    Welfare levels are EV; percentages divide by baseline selected consumption,
    since baseline EV is zero. A boundary optimum is conditional on the ceiling.
    Other metrics retain their historical condensed-solver path.

    Hicksian keyword options: ``consumption_categories=(0,)``,
    ``base_equilibrium=None``, ``sigma=0``, ``ge_tol=1e-8``, ``ge_method="auto"``,
    ``ge_max_iter=100`` and ``ge_max_steps=100``. Failed GE/refinement raises;
    no penalty payoff replaces a failed candidate. See ``docs/trade_policy.md``.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    country_idx : int or str, default 'USA'
        Sovereign economy setting the optimal tariff ('USA', 'CHN', 'EUR', etc.).
    target_countries : Sequence[str], optional
        Target trading partners. If None, applies universally to all foreign partners.
    metric : {'hicksian_ev', 'geary_khamis', 'equivalent_variation', 'terms_of_trade', 'harberger'}, default 'geary_khamis'
        Sovereign welfare objective function metric.
    tariff_max : float, default 0.50
        Maximum ad-valorem tariff rate ceiling (0.50 = 50%).
    num_grid : int, default 25
        Number of points on the initial exploration grid.
    tol : float, default 1e-4
        Optimization convergence tolerance on tariff rate.
    x0 : np.ndarray, optional
        Initial state vector guess for baseline general equilibrium.
    method : {'bounded', 'grid'}, default 'bounded'
        Optimization method ('bounded' uses Brent scalar minimization around grid peak).
    policy_mode : str, default 'universal'
        Tariff instrument mode ('universal', 'final_only', 'intermediate_only').

    Returns
    -------
    OptimalTariffResult
        Frozen dataclass containing the optimal tariff rate, welfare gain,
        terms-of-trade impact, welfare curve, and solved equilibrium state.
    """
    if str(metric).strip().lower() == "hicksian_ev":
        from ._hicksian_policy import unilateral
        return unilateral(calib, country_idx, target_countries, tariff_max, num_grid,
                          tol, x0, method, policy_mode, **kwargs)
    if tariff_max <= 0 or num_grid < 2 or tol <= 0:
        raise ValueError("tariff_max and tol must be positive, num_grid at least two")
    if method not in ("grid", "bounded"):
        raise ValueError("method must be grid or bounded")
    ge_tol = float(kwargs.get("ge_tol", min(1e-8, tol * .01)))
    country_code = get_country_code(calib, country_idx)
    target_tuple = tuple(target_countries) if target_countries is not None else None

    # 1. Baseline Free Trade solve
    res_base = _solve_policy_equilibrium(
        calib,
        x0=x0,
        method="condensed",
        tol=ge_tol,
    )
    base_welfare = evaluate_national_welfare(
        calib,
        res_base,
        country_idx=country_idx,
        metric=metric,
    )
    tot_init = float(res_base.terms_of_trade[resolve_country_indices(calib, country_idx)[0]]) if res_base.terms_of_trade is not None else 1.0

    # Cached warm-start equilibrium vector
    x_cache = res_base.x_sol.copy()

    def eval_at_tariff(rate: float) -> tuple[float, TradeEquilibriumResult]:
        nonlocal x_cache
        rate_clamped = max(float(rate), 0.0)
        if rate_clamped < 1e-6:
            return base_welfare, res_base

        tau, tau_fd = build_strategic_tariffs(
            calib,
            player_tariffs={country_code: rate_clamped},
            policy_mode=policy_mode,
            target_countries=target_countries,
        )
        eq_sol = _solve_policy_equilibrium(
            calib,
            tau=tau,
            tau_fd=tau_fd,
            x0=x_cache,
            method="condensed",
            tol=ge_tol,
        )
        x_cache = eq_sol.x_sol.copy()
        w_val = evaluate_national_welfare(
            calib,
            eq_sol,
            country_idx=country_idx,
            metric=metric,
            base_equilibrium=res_base,
            tau=tau,
            tau_fd=tau_fd,
        )
        return w_val, eq_sol

    # 2. Grid evaluation
    grid = np.linspace(0.0, tariff_max, num_grid)
    w_curve = np.empty(num_grid, dtype=float)
    eq_curve: list[TradeEquilibriumResult] = []

    for i, t_val in enumerate(grid):
        if i == 0 and t_val == 0.0:
            w_curve[0] = base_welfare
            eq_curve.append(res_base)
            continue
        w_v, eq_v = eval_at_tariff(t_val)
        w_curve[i] = w_v
        eq_curve.append(eq_v)

    best_idx = int(np.argmax(w_curve))
    best_rate = float(grid[best_idx])
    best_welfare = float(w_curve[best_idx])
    best_eq = eq_curve[best_idx]

    # 3. Refined Bounded Optimization
    if method == "bounded" and num_grid >= 3:
        i_low = max(best_idx - 1, 0)
        i_high = min(best_idx + 1, num_grid - 1)
        b_low = float(grid[i_low])
        b_high = float(grid[i_high])

        if b_high > b_low:
            opt_res = minimize_scalar(
                lambda t: -eval_at_tariff(t)[0],
                bounds=(b_low, b_high),
                method="bounded",
                options={"xatol": tol, "maxiter": 100},
            )
            if opt_res.success:
                cand_rate = float(opt_res.x)
                cand_w, cand_eq = eval_at_tariff(cand_rate)
                if cand_w > best_welfare:
                    best_rate = cand_rate
                    best_welfare = cand_w
                    best_eq = cand_eq

    welfare_gain_pct = ((best_welfare - base_welfare) / abs(base_welfare)) * 100.0 if base_welfare != 0.0 else 0.0
    c_idx_first = resolve_country_indices(calib, country_idx)[0]
    tot_opt = float(best_eq.terms_of_trade[c_idx_first]) if best_eq.terms_of_trade is not None else tot_init

    return OptimalTariffResult(
        country_code=country_code,
        optimal_tariff_rate=best_rate,
        welfare_gain_pct=welfare_gain_pct,
        baseline_welfare=base_welfare,
        optimal_welfare=best_welfare,
        welfare_metric=metric,
        terms_of_trade_initial=tot_init,
        terms_of_trade_optimal=tot_opt,
        tariff_grid=grid,
        welfare_curve=w_curve,
        equilibrium=best_eq,
        target_countries=target_tuple,
        metadata={
            "num_grid": num_grid,
            "tariff_max": tariff_max,
            "method": method,
            "tol": tol,
            "policy_mode": policy_mode,
            "tariff_revenue_mode": "schedule",
            "ge_tol": ge_tol,
            "search_scope": "grid plus refinement around the sampled maximum",
        },
    )


# ==============================================================================
# 3. Multilateral Non-Cooperative Nash Tariff Equilibrium
# ==============================================================================

def solve_multilateral_nash_tariffs(
    calib: TradeCalibrationResult,
    player_countries: Sequence[str] = ("USA", "CHN", "EUR", "CAN", "MEX"),
    metric: str = "geary_khamis",
    method: str = "best_response",
    relaxation: float = 0.5,
    tol: float = 1e-4,
    max_iter: int = 30,
    tariff_max: float = 1.5,
    x0: np.ndarray | None = None,
    policy_mode: str = "universal",
    initial_tariffs: Mapping[str, float] | None = None,
    **kwargs: Any,
) -> NashTariffResult:
    """Search for a multilateral tariff equilibrium and check unilateral deviations.

    Finds a joint policy profile :math:`\\boldsymbol{\\tau}^* = (\\tau_1^*, \\dots, \\tau_P^*)`
    such that each sovereign player's tariff is a mutual best response:
    .. math::
        \\tau_i^* = \\arg\\max_{\\tau_i \\in [0, \\bar{\\tau}]} W_i(\\tau_i, \\boldsymbol{\\tau}_{-i}^*)
        \\quad \\forall i \\in \\mathcal{P}

    The Hicksian path (``metric="hicksian_ev"``) uses consistent accounting,
    audited Newton/hybrid/continuation solves and a fixed zero-tariff baseline.
    It supports ``best_response`` only. EV percentages and relative regret use
    baseline selected consumption expenditure, including all countries in the
    world aggregate. Final simultaneous deviations, not damped update sizes,
    control convergence. Exhausted GE recovery raises without a payoff.

    Hicksian options match ``compute_unilateral_optimal_tariff`` and add
    ``best_response_grid_size=9`` and ``regret_tol`` (defaults to ``tol``).
    Regret tolerance is a consumption fraction, not a percentage. Grid/local
    refinement is not a global optimality proof; see ``docs/trade_policy.md``.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    player_countries : Sequence[str], default ('USA', 'CHN', 'EUR', 'CAN', 'MEX')
        List of active strategic sovereign powers or regional blocs.
    metric : {'hicksian_ev', 'geary_khamis', 'equivalent_variation', 'terms_of_trade', 'harberger'}, default 'geary_khamis'
        Sovereign national welfare objective function.
    method : {'best_response', 'gradient'}, default 'best_response'
        Game-theoretic fixed point algorithm:
        - 'best_response': Damped Gauss-Seidel best response policy iteration.
        - 'gradient': Finite-difference gradient ascent along sovereign welfare surfaces.
    relaxation : float, default 0.5
        Under-relaxation damping parameter :math:`\\theta \\in (0, 1]` preventing policy oscillations.
    tol : float, default 1e-4
        Undamped best-response infinity-norm convergence threshold: :math:`\\max_i |\\tau_i^{(k)} - \\tau_i^{(k-1)}| < \\text{tol}`.
    max_iter : int, default 30
        Maximum outer policy iterations.
    tariff_max : float, default 1.5
        Maximum institutional tariff rate ceiling.
    x0 : np.ndarray, optional
        Initial general equilibrium state guess.
    policy_mode : str, default 'universal'
        Tariff instrument mode ('universal', 'final_only', 'intermediate_only').
    initial_tariffs : Mapping[str, float], optional
        Starting tariff policy profile. If None, starts from 0% Free Trade.

    Returns
    -------
    NashTariffResult
        Equilibrium tariff profile, player welfare changes, terms-of-trade shifts,
        and convergence diagnostics.
    """
    if "players" in kwargs:
        player_countries = kwargs.pop("players")
    if str(metric).strip().lower() == "hicksian_ev":
        from ._hicksian_policy import nash
        return nash(calib, player_countries, method, relaxation, tol, max_iter,
                    tariff_max, x0, policy_mode, initial_tariffs, **kwargs)

    resolved_players: list[str] = []
    c_codes = list(calib.country_codes) if calib.country_codes else []
    for p in player_countries:
        if isinstance(p, int):
            if c_codes and 0 <= p < len(c_codes):
                resolved_players.append(str(c_codes[p]))
            else:
                resolved_players.append(str(p))
        else:
            resolved_players.append(str(p).strip().upper())
    players = tuple(resolved_players)
    grid_size = int(kwargs.pop("best_response_grid_size", 9))
    regret_tol = float(kwargs.pop("regret_tol", tol))
    ge_tol = float(kwargs.pop("ge_tol", min(1e-7, tol * 0.01)))
    if not players or len(set(players)) != len(players):
        raise ValueError("Supply at least one distinct strategic player")
    if not (0 < relaxation <= 1) or tol <= 0 or max_iter < 1 or tariff_max <= 0 or grid_size < 3:
        raise ValueError("Require 0 < relaxation <= 1, positive tolerances/limits and grid size >= 3")
    if regret_tol < 0 or ge_tol <= 0:
        raise ValueError("Regret tolerance must be nonnegative and GE tolerance positive")

    # 1. Baseline Free Trade solve
    res_base = solve_trade_equilibrium(
        calib,
        x0=x0,
        method="condensed", tariff_revenue_mode="schedule",
        tol=ge_tol,
    )
    if not res_base.converged:
        raise RuntimeError("Baseline GE failed; tariff-game payoffs are unavailable")
    base_welfares = {
        p: evaluate_national_welfare(calib, res_base, country_idx=p, metric=metric)
        for p in players
    }
    world_w_base = float(np.sum(list(base_welfares.values())))

    # Initialize policies
    tau_curr: dict[str, float] = {}
    for p in players:
        if initial_tariffs is not None and p in initial_tariffs:
            tau_curr[p] = float(initial_tariffs[p])
        else:
            tau_curr[p] = 0.0

    if any(not np.isfinite(v) or v < 0 or v > tariff_max for v in tau_curr.values()):
        raise ValueError("Initial tariffs must lie within [0, tariff_max]")
    x_state = res_base.x_sol.copy()
    history: list[dict[str, Any]] = []
    converged = False
    outer_error = 1.0

    inner_failures = []
    payoff_cache = {}

    def payoff(profile, player):
        nonlocal x_state
        key = tuple(float(profile[p]) for p in players)
        if key not in payoff_cache:
            if all(v == 0.0 for v in key):
                payoff_cache[key] = dict(base_welfares)
            else:
                ta, tf = build_strategic_tariffs(calib, profile, policy_mode=policy_mode)
                try:
                    eq = solve_trade_equilibrium(calib, tau=ta, tau_fd=tf, x0=x_state,
                        method="condensed", tariff_revenue_mode="schedule", tol=ge_tol)
                    if not eq.converged or not np.isfinite(eq.max_residual):
                        raise RuntimeError(f"GE residual {eq.max_residual} exceeds tolerance")
                    values = {p: evaluate_national_welfare(
                        calib, eq, country_idx=p, metric=metric, base_equilibrium=res_base, tau=ta, tau_fd=tf,
                    ) for p in players}
                    if not all(np.isfinite(v) for v in values.values()):
                        raise ValueError("Nonfinite policy payoff")
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    inner_failures.append({"profile": dict(profile), "error": str(exc)})
                    payoff_cache[key] = {p: float("nan") for p in players}
                else:
                    x_state = eq.x_sol.copy()
                    payoff_cache[key] = values
        return payoff_cache[key][player]

    def best_response(profile, player):
        def objective(rate):
            return payoff({**profile, player: float(rate)}, player)
        grid = np.unique(np.append(np.linspace(0.0, tariff_max, grid_size), profile[player]))
        values = np.array([objective(rate) for rate in grid])
        finite = np.isfinite(values)
        if not finite.all():
            return profile[player], float("nan")
        candidates = list(zip(values, grid))
        # Refine every sampled local peak, including both boundary brackets.
        for j in range(len(grid)):
            if (j == 0 or values[j] >= values[j - 1]) and (j == len(grid) - 1 or values[j] >= values[j + 1]):
                lo, hi = grid[max(0, j - 1)], grid[min(len(grid) - 1, j + 1)]
                opt = minimize_scalar(lambda t: -objective(t), bounds=(lo, hi), method="bounded",
                                      options={"xatol": max(tol * 0.1, 1e-8)})
                if opt.success and np.isfinite(opt.fun):
                    candidates.append((-float(opt.fun), float(opt.x)))
        value, rate = max(candidates, key=lambda pair: pair[0])
        return rate, value

    t_start = time.perf_counter()

    for it in range(max_iter):
        tau_prev = copy.deepcopy(tau_curr)
        tau_cand: dict[str, float] = {}

        if method == "best_response":
            for p in players:
                cand_rate, _ = best_response(tau_curr, p)
                tau_cand[p] = cand_rate
                tau_curr[p] = float((1.0 - relaxation) * tau_prev[p] + relaxation * cand_rate)

        elif method == "gradient":
            # Projected gradient ascent
            h_fd = 0.01
            grad = {}
            for p in players:
                # Forward and backward difference
                prof_plus = copy.deepcopy(tau_prev)
                prof_plus[p] = min(tariff_max, tau_prev[p] + h_fd)
                prof_minus = copy.deepcopy(tau_prev)
                prof_minus[p] = max(0.0, tau_prev[p] - h_fd)
                w_plus = payoff(prof_plus, p)
                w_minus = payoff(prof_minus, p)
                if not np.isfinite(w_plus) or not np.isfinite(w_minus):
                    grad[p] = 0.0
                    continue
                h_actual = float(prof_plus[p] - prof_minus[p])
                grad[p] = (w_plus - w_minus) / h_actual if h_actual > 1e-8 else 0.0

            # Gradient step with line search
            step_size = relaxation * 0.10
            for p in players:
                g_val = grad[p]
                step = np.clip(step_size * g_val, -0.05, 0.05)
                tau_curr[p] = float(np.clip(tau_prev[p] + step, 0.0, tariff_max))

        else:
            raise ValueError(f"Unknown Nash method '{method}'. Valid options: 'best_response', 'gradient'.")

        # Check convergence
        outer_error = float(max(abs(tau_cand[p] - tau_prev[p]) for p in players)) if method == "best_response" else float(max(abs(tau_curr[p] - tau_prev[p]) / relaxation for p in players))
        history.append({
            "iteration": it + 1,
            "tariffs": copy.deepcopy(tau_curr),
            "error": outer_error,
        })

        if inner_failures:
            # The declared deviation search is unresolved. Repeating policy
            # updates cannot certify it; return diagnostics for this candidate.
            break
        if outer_error <= tol and not inner_failures:
            converged = True
            break

    t_duration = time.perf_counter() - t_start

    # 4. Final equilibrium solve at Nash tariffs
    if all(v == 0.0 for v in tau_curr.values()):
        res_nash = res_base
        tau_nash, tau_fd_nash = None, None
    else:
        tau_nash, tau_fd_nash = build_strategic_tariffs(calib, tau_curr, policy_mode=policy_mode)
        res_nash = solve_trade_equilibrium(
            calib,
            tau=tau_nash,
            tau_fd=tau_fd_nash,
            x0=x_state,
            method="condensed", tariff_revenue_mode="schedule",
            tol=ge_tol,
        )

    # Compute outcomes
    try:
        if not res_nash.converged:
            raise RuntimeError("Final GE did not converge")
        player_welfares = {
            p: evaluate_national_welfare(calib, res_nash, country_idx=p, metric=metric,
                base_equilibrium=res_base, tau=tau_nash, tau_fd=tau_fd_nash) for p in players
        }
        if not all(np.isfinite(v) for v in player_welfares.values()):
            raise ValueError("Nonfinite final payoff")
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        inner_failures.append({"profile": dict(tau_curr), "error": str(exc)})
        player_welfares = {p: float("nan") for p in players}
    welfare_changes_pct = {
        p: ((player_welfares[p] - base_welfares[p]) / abs(base_welfares[p])) * 100.0
        for p in players
    }
    world_w_nash = float(np.sum(list(player_welfares.values())))
    world_w_change_pct = ((world_w_nash - world_w_base) / abs(world_w_base)) * 100.0

    tot_changes_pct = {}
    for p in players:
        c_idx = resolve_country_indices(calib, p)[0]
        tot_b = float(res_base.terms_of_trade[c_idx]) if res_base.terms_of_trade is not None else 1.0
        tot_n = float(res_nash.terms_of_trade[c_idx]) if res_nash.terms_of_trade is not None else 1.0
        tot_changes_pct[p] = ((tot_n - tot_b) / tot_b) * 100.0 if tot_b != 0.0 else 0.0

    # Verify deviations against the final, simultaneous profile. This is a
    # numerical search over a declared grid with local refinement, not a proof
    # of global optimality for an arbitrary welfare function.
    regrets, responses, boundary = {}, {}, {}
    for p in players:
        rate, value = best_response(tau_curr, p)
        responses[p] = rate
        regrets[p] = max(0.0, value - player_welfares[p]) if np.isfinite(value) and np.isfinite(player_welfares[p]) else float("inf")
        boundary[p] = "lower" if rate <= tol else ("upper" if rate >= tariff_max - tol else "interior")
    max_regret = max(regrets.values())
    relative_regret = max(regrets[p] / max(abs(player_welfares[p]), 1.0) if np.isfinite(player_welfares[p]) else float("inf") for p in players)
    outer_error = max(abs(responses[p] - tau_curr[p]) for p in players)
    if not np.isfinite(max_regret):
        outer_error = float("inf")
    converged = bool(res_nash.converged and not inner_failures and outer_error <= tol and relative_regret <= regret_tol)

    return NashTariffResult(
        strategic_players=players,
        nash_tariffs=tau_curr,
        welfare_changes_pct=welfare_changes_pct,
        terms_of_trade_changes_pct=tot_changes_pct,
        world_welfare_change_pct=world_w_change_pct,
        outer_iterations=len(history),
        converged=converged,
        outer_error=outer_error,
        equilibrium=res_nash,
        tau_nash=tau_nash,
        tau_fd_nash=tau_fd_nash,
        policy_mode=policy_mode,
        welfare_metric=metric,
        method=method,
        player_welfares=player_welfares,
        baseline_welfares=base_welfares,
        iteration_history=history,
        metadata={
            "duration_seconds": time.perf_counter() - t_start,
            "max_regret": max_regret, "relative_max_regret": relative_regret,
            "player_regrets": regrets, "best_responses": responses,
            "best_response_boundaries": boundary, "inner_solver_failures": inner_failures,
            "best_response_grid_size": grid_size, "regret_tol": regret_tol, "ge_tol": ge_tol,
            "verification": "full-interval grid with bounded local refinement; no global-optimality proof",
            "relaxation": relaxation,
            "tol": tol,
            "max_iter": max_iter,
        },
    )


# ==============================================================================
# 4. Strategic Welfare Payoff Matrix: The Prisoner's Dilemma
# ==============================================================================

def compute_welfare_payoff_matrix(
    calib: TradeCalibrationResult,
    player_a: str = "USA",
    player_b: str = "CHN",
    metric: str = "geary_khamis",
    optimal_a: float | None = None,
    optimal_b: float | None = None,
    nash_a: float | None = None,
    nash_b: float | None = None,
    x0: np.ndarray | None = None,
    policy_mode: str = "universal",
    **kwargs: Any,
) -> WelfarePayoffMatrixResult:
    """Compute normal-form welfare payoff matrix evaluating strategic trade war games.

    Evaluates the classical :math:`2 \\times 2` normal-form matrix across
    mutual cooperation (Free Trade), unilateral defection, and retaliatory trade wars:
    .. math::
        \\begin{array}{c|cc}
        & \\text{Player B Cooperates } (\\tau_B = 0) & \\text{Player B Defects } (\\tau_B = \\tau_B^*) \\\\
        \\hline
        \\text{Player A Cooperates } (\\tau_A = 0) & (0.00\\%, 0.00\\%) & (-\\mathcal{L}_A, +\\mathcal{G}_B) \\\\
        \\text{Player A Defects } (\\tau_A = \\tau_A^*) & (+\\mathcal{G}_A, -\\mathcal{L}_B) & (-\\Delta W_A^{\\text{Nash}}, -\\Delta W_B^{\\text{Nash}})
        \\end{array}

    The returned payoffs determine whether this finite game is a Prisoner's
    Dilemma. Dominant defection and Pareto inferiority are not assumed.

    With ``metric="hicksian_ev"``, all cells share one fixed zero-tariff
    baseline and two fixed actions per player. Payoffs are EV as a percentage
    of baseline selected consumption. A positive action comes from
    ``optimal_*``, otherwise ``nash_*``, otherwise the unilateral optimum.
    Conflicting supplied rates raise; supplied ``nash_*`` are explicitly
    unverified fixed actions, never a continuous-game Nash certificate.
    Mutual positive actions carry no assumed welfare sign. Hicksian equilibrium
    options match ``compute_unilateral_optimal_tariff``. See ``docs/trade_policy.md``.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    player_a : str, default 'USA'
        First strategic sovereign authority.
    player_b : str, default 'CHN'
        Second strategic sovereign authority.
    metric : {'hicksian_ev', 'geary_khamis', 'equivalent_variation', 'terms_of_trade', 'harberger'}, default 'geary_khamis'
        Sovereign welfare objective function.
    optimal_a : float, optional
        Unilateral optimal tariff rate for Player A. If None, computed automatically.
    optimal_b : float, optional
        Unilateral optimal tariff rate for Player B. If None, computed automatically.
    nash_a : float, optional
        Nash trade war tariff rate for Player A. If None, computed automatically.
    nash_b : float, optional
        Nash trade war tariff rate for Player B. If None, computed automatically.
    x0 : np.ndarray, optional
        Initial equilibrium vector.
    policy_mode : str, default 'universal'
        Tariff instrument mode ('universal', 'final_only', 'intermediate_only').

    Returns
    -------
    WelfarePayoffMatrixResult
        Normal-form payoff matrix container with formatted DataFrame summary.
    """
    if str(metric).strip().lower() == "hicksian_ev":
        from ._hicksian_policy import payoff_matrix
        return payoff_matrix(calib, player_a, player_b, optimal_a, optimal_b,
                             nash_a, nash_b, x0, policy_mode, **kwargs)
    p_a = player_a.strip().upper()
    p_b = player_b.strip().upper()

    # 1. Baseline Free Trade solve (C, C)
    res_cc = _solve_policy_equilibrium(calib, x0=x0, method="condensed", tol=1e-8)
    w_a_0 = evaluate_national_welfare(calib, res_cc, country_idx=p_a, metric=metric)
    w_b_0 = evaluate_national_welfare(calib, res_cc, country_idx=p_b, metric=metric)

    # 2. Determine tariff rates
    if optimal_a is None:
        opt_res_a = compute_unilateral_optimal_tariff(
            calib, country_idx=p_a, target_countries=[p_b], metric=metric, x0=res_cc.x_sol, policy_mode=policy_mode
        )
        rate_a = opt_res_a.optimal_tariff_rate
    else:
        rate_a = float(optimal_a)

    if optimal_b is None:
        opt_res_b = compute_unilateral_optimal_tariff(
            calib, country_idx=p_b, target_countries=[p_a], metric=metric, x0=res_cc.x_sol, policy_mode=policy_mode
        )
        rate_b = opt_res_b.optimal_tariff_rate
    else:
        rate_b = float(optimal_b)

    # Nash rates
    if nash_a is None or nash_b is None:
        nash_res = solve_multilateral_nash_tariffs(
            calib,
            player_countries=[p_a, p_b],
            metric=metric,
            x0=res_cc.x_sol,
            policy_mode=policy_mode,
            max_iter=15,
        )
        if not nash_res.converged:
            raise RuntimeError("Cannot label a payoff cell Nash: best responses are unresolved")
        rate_nash_a = nash_res.nash_tariffs.get(p_a, rate_a)
        rate_nash_b = nash_res.nash_tariffs.get(p_b, rate_b)
    else:
        rate_nash_a = float(nash_a)
        rate_nash_b = float(nash_b)

    # 3. Solve remaining 3 cells:
    # Cell DC: Player A defects, Player B cooperates
    tau_dc, tau_fd_dc = build_strategic_tariffs(calib, {p_a: {p_b: rate_a}}, policy_mode=policy_mode)
    res_dc = _solve_policy_equilibrium(calib, tau=tau_dc, tau_fd=tau_fd_dc, x0=res_cc.x_sol, method="condensed")

    # Cell CD: Player A cooperates, Player B defects
    tau_cd, tau_fd_cd = build_strategic_tariffs(calib, {p_b: {p_a: rate_b}}, policy_mode=policy_mode)
    res_cd = _solve_policy_equilibrium(calib, tau=tau_cd, tau_fd=tau_fd_cd, x0=res_cc.x_sol, method="condensed")

    # Cell DD: Mutual Defection (Trade War)
    tau_dd, tau_fd_dd = build_strategic_tariffs(
        calib, {p_a: {p_b: rate_nash_a}, p_b: {p_a: rate_nash_b}}, policy_mode=policy_mode
    )
    res_dd = _solve_policy_equilibrium(calib, tau=tau_dd, tau_fd=tau_fd_dd, x0=res_dc.x_sol, method="condensed")

    scenarios = {"CC": res_cc, "DC": res_dc, "CD": res_cd, "DD": res_dd}

    # 4. Evaluate Payoffs (% welfare growth vs Free Trade)
    def calc_payoff(eq_state: TradeEquilibriumResult, t_inter=None, t_fd=None) -> tuple[float, float]:
        wa = evaluate_national_welfare(calib, eq_state, country_idx=p_a, metric=metric, base_equilibrium=res_cc, tau=t_inter, tau_fd=t_fd)
        wb = evaluate_national_welfare(calib, eq_state, country_idx=p_b, metric=metric, base_equilibrium=res_cc, tau=t_inter, tau_fd=t_fd)
        pct_a = ((wa - w_a_0) / abs(w_a_0)) * 100.0 if w_a_0 != 0.0 else 0.0
        pct_b = ((wb - w_b_0) / abs(w_b_0)) * 100.0 if w_b_0 != 0.0 else 0.0
        return pct_a, pct_b

    p_cc = calc_payoff(res_cc, None, None)
    p_dc = calc_payoff(res_dc, tau_dc, tau_fd_dc)
    p_cd = calc_payoff(res_cd, tau_cd, tau_fd_cd)
    p_dd = calc_payoff(res_dd, tau_dd, tau_fd_dd)

    # Shape: (|S_A|, |S_B|, 2)
    # Row 0: Cooperate, Row 1: Defect
    # Col 0: Cooperate, Col 1: Defect
    payoff_mat = np.zeros((2, 2, 2), dtype=float)
    payoff_mat[0, 0, 0], payoff_mat[0, 0, 1] = p_cc[0], p_cc[1]
    payoff_mat[0, 1, 0], payoff_mat[0, 1, 1] = p_cd[0], p_cd[1]
    payoff_mat[1, 0, 0], payoff_mat[1, 0, 1] = p_dc[0], p_dc[1]
    payoff_mat[1, 1, 0], payoff_mat[1, 1, 1] = p_dd[0], p_dd[1]

    # Formatted summary DataFrame
    strat_a_c = f"Cooperate ({p_a} $\\tau=0\\%$)"
    strat_a_d = f"Defect / Nash ({p_a} $\\tau={rate_nash_a * 100:.1f}\\%$)"
    strat_b_c = f"Cooperate ({p_b} $\\tau=0\\%$)"
    strat_b_d = f"Defect / Nash ({p_b} $\\tau={rate_nash_b * 100:.1f}\\%$)"

    summary_df = pd.DataFrame(
        [
            [f"({p_cc[0]:+.2f}%, {p_cc[1]:+.2f}%)", f"({p_cd[0]:+.2f}%, {p_cd[1]:+.2f}%)"],
            [f"({p_dc[0]:+.2f}%, {p_dc[1]:+.2f}%)", f"({p_dd[0]:+.2f}%, {p_dd[1]:+.2f}%)"],
        ],
        index=pd.Index([strat_a_c, strat_a_d], name=f"{p_a} Strategy"),
        columns=pd.Index([strat_b_c, strat_b_d], name=f"{p_b} Strategy"),
    )

    return WelfarePayoffMatrixResult(
        players=(p_a, p_b),
        strategies=("Cooperate (0%)", "Defect (Optimal / Nash)"),
        payoff_matrix=payoff_mat,
        scenarios=scenarios,
        summary_df=summary_df,
        welfare_metric=metric,
        metadata={
            "rate_a": rate_a,
            "rate_b": rate_b,
            "rate_nash_a": rate_nash_a,
            "rate_nash_b": rate_nash_b,
            "policy_mode": policy_mode,
        },
    )


# ==============================================================================
# 5. Empirical 2024–2026 Geopolitical Benchmark
# ==============================================================================

def benchmark_real_world_tariffs(
    calib: TradeCalibrationResult,
    metric: str = "geary_khamis",
    x0: np.ndarray | None = None,
    unilateral_results: Mapping[str, OptimalTariffResult] | None = None,
    nash_result: NashTariffResult | None = None,
    policy_mode: str = "universal",
    **kwargs: Any,
) -> pd.DataFrame:
    """Benchmark theoretical Nash tariffs against 2024–2026 real-world tariff unfoldings.

    Simulates canonical policy scenarios with 100% authentic CGE evaluations:
    1. Benchmark Free Trade (0% baseline)
    2. US Unilateral Optimal Tariff
    3. China Unilateral Optimal Tariff
    4. EU Unilateral Optimal Tariff
    5. Bilateral US–China Nash Trade War
    6. Multilateral 5-Bloc Global Nash War (USA, CHN, EUR, CAN, MEX)
    7. 2024–2026 Universal Baseline (10% US universal, 15% China reciprocal)
    8. 2024–2026 Punitive Shock (60% China, 25% USMCA)

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    metric : str, default 'geary_khamis'
        Welfare evaluation metric.
    x0 : np.ndarray, optional
        Initial equilibrium vector.
    unilateral_results : Mapping[str, OptimalTariffResult], optional
        Precomputed unilateral optimal tariff results.
    nash_result : NashTariffResult, optional
        Precomputed multilateral Nash equilibrium result.
    policy_mode : str, default 'universal'
        Tariff instrument mode ('universal', 'final_only').

    Returns
    -------
    pd.DataFrame
        Publication-grade comparative summary table matching LaTeX table format.
    """
    if str(metric).strip().lower() == "hicksian_ev":
        raise NotImplementedError(
            "The historical geopolitical wrapper does not support Hicksian policy certification. "
            "Use explicit schedules with solve_policy_equilibrium and evaluate_national_welfare, "
            "or compute_welfare_payoff_matrix for fixed-action games.")
    country_codes = list(calib.country_codes) if calib.country_codes else list(CANONICAL_COUNTRY_CODES)
    has_usa = "USA" in country_codes
    p_usa = "USA" if has_usa else country_codes[0]
    p_chn = "CHN" if ("CHN" in country_codes) else (country_codes[1] if len(country_codes) > 1 else country_codes[0])
    eu_avail = [c for c in EU_COUNTRY_CODES if c in country_codes]
    p_eur = "EUR" if eu_avail or ("EUR" in country_codes) else country_codes[0]
    p_can = "CAN" if ("CAN" in country_codes) else country_codes[0]
    p_mex = "MEX" if ("MEX" in country_codes) else (country_codes[1] if len(country_codes) > 1 else country_codes[0])

    # Benchmark Free Trade
    res_base = _solve_policy_equilibrium(calib, x0=x0, method="condensed", tol=1e-8)
    w_base_usa = evaluate_national_welfare(calib, res_base, country_idx=p_usa, metric=metric)
    w_base_chn = evaluate_national_welfare(calib, res_base, country_idx=p_chn, metric=metric)
    w_base_eur = evaluate_national_welfare(calib, res_base, country_idx=p_eur, metric=metric)
    w_base_world = evaluate_national_welfare(calib, res_base, country_idx=country_codes, metric=metric)

    def calc_impacts(eq_res: TradeEquilibriumResult, t_inter: np.ndarray | None = None, t_fd: np.ndarray | None = None) -> tuple[float, float, float, float]:
        w_usa = evaluate_national_welfare(calib, eq_res, country_idx=p_usa, metric=metric, base_equilibrium=res_base, tau=t_inter, tau_fd=t_fd)
        w_chn = evaluate_national_welfare(calib, eq_res, country_idx=p_chn, metric=metric, base_equilibrium=res_base, tau=t_inter, tau_fd=t_fd)
        w_eur = evaluate_national_welfare(calib, eq_res, country_idx=p_eur, metric=metric, base_equilibrium=res_base, tau=t_inter, tau_fd=t_fd)
        w_world = evaluate_national_welfare(calib, eq_res, country_idx=country_codes, metric=metric, base_equilibrium=res_base, tau=t_inter, tau_fd=t_fd)
        return (
            ((w_usa - w_base_usa) / abs(w_base_usa)) * 100.0 if w_base_usa != 0.0 else 0.0,
            ((w_chn - w_base_chn) / abs(w_base_chn)) * 100.0 if w_base_chn != 0.0 else 0.0,
            ((w_eur - w_base_eur) / abs(w_base_eur)) * 100.0 if w_base_eur != 0.0 else 0.0,
            ((w_world - w_base_world) / abs(w_base_world)) * 100.0 if w_base_world != 0.0 else 0.0,
        )

    # 1. Free trade
    imp_ft = (0.00, 0.00, 0.00, 0.00)

    # Determine optimal rates from provided results or defaults
    t_opt_us = float(unilateral_results[p_usa].optimal_tariff_rate) if unilateral_results and p_usa in unilateral_results else float(kwargs.get("t_opt_us", 0.0))
    t_opt_cn = float(unilateral_results[p_chn].optimal_tariff_rate) if unilateral_results and p_chn in unilateral_results else float(kwargs.get("t_opt_cn", 0.0))
    t_opt_eu = float(unilateral_results[p_eur].optimal_tariff_rate) if unilateral_results and p_eur in unilateral_results else float(kwargs.get("t_opt_eu", 0.0))

    if nash_result is not None:
        t_nash_us = float(nash_result.nash_tariffs.get(p_usa, 0.0))
        t_nash_cn = float(nash_result.nash_tariffs.get(p_chn, 0.0))
        multi_sched = {
            p: float(nash_result.nash_tariffs.get(p, 0.0))
            for p in [p_usa, p_chn, p_eur, p_can, p_mex]
        }
    else:
        t_nash_us = float(kwargs.get("t_nash_us", 0.0))
        t_nash_cn = float(kwargs.get("t_nash_cn", 0.0))
        multi_sched = kwargs.get("multi_sched", {p_usa: 0.0, p_chn: 0.0, p_eur: 0.0, p_can: 0.0, p_mex: 0.0})

    # 2. US Unilateral Optimal
    if t_opt_us > 1e-4:
        tau_us, tau_fd_us = build_strategic_tariffs(calib, {p_usa: t_opt_us}, policy_mode=policy_mode)
        res_us = _solve_policy_equilibrium(calib, tau=tau_us, tau_fd=tau_fd_us, x0=res_base.x_sol, method="condensed")
        imp_us = calc_impacts(res_us, tau_us, tau_fd_us)
    else:
        res_us = res_base
        imp_us = imp_ft

    # 3. China Unilateral Optimal
    if t_opt_cn > 1e-4:
        tau_cn, tau_fd_cn = build_strategic_tariffs(calib, {p_chn: t_opt_cn}, policy_mode=policy_mode)
        res_cn = _solve_policy_equilibrium(calib, tau=tau_cn, tau_fd=tau_fd_cn, x0=res_base.x_sol, method="condensed")
        imp_cn = calc_impacts(res_cn, tau_cn, tau_fd_cn)
    else:
        res_cn = res_base
        imp_cn = imp_ft

    # 4. EU Unilateral Optimal
    if t_opt_eu > 1e-4:
        tau_eu, tau_fd_eu = build_strategic_tariffs(calib, {p_eur: t_opt_eu}, policy_mode=policy_mode)
        res_eu = _solve_policy_equilibrium(calib, tau=tau_eu, tau_fd=tau_fd_eu, x0=res_base.x_sol, method="condensed")
        imp_eu = calc_impacts(res_eu, tau_eu, tau_fd_eu)
    else:
        res_eu = res_base
        imp_eu = imp_ft

    # 5. Bilateral US-China Nash Game
    if (t_nash_us > 1e-4) or (t_nash_cn > 1e-4):
        tau_bi, tau_fd_bi = build_strategic_tariffs(calib, {p_usa: {p_chn: t_nash_us}, p_chn: {p_usa: t_nash_cn}}, policy_mode=policy_mode)
        res_bi = _solve_policy_equilibrium(calib, tau=tau_bi, tau_fd=tau_fd_bi, x0=res_us.x_sol, method="condensed")
        imp_bi = calc_impacts(res_bi, tau_bi, tau_fd_bi)
    else:
        res_bi = res_base
        imp_bi = imp_ft

    # 6. Multilateral 5-Bloc Nash War
    if any(v > 1e-4 for v in multi_sched.values()):
        tau_multi, tau_fd_multi = build_strategic_tariffs(calib, multi_sched, policy_mode=policy_mode)
        res_multi = _solve_policy_equilibrium(calib, tau=tau_multi, tau_fd=tau_fd_multi, x0=res_bi.x_sol, method="condensed")
        imp_multi = calc_impacts(res_multi, tau_multi, tau_fd_multi)
    else:
        res_multi = res_base
        imp_multi = imp_ft

    # 7. 2024-2026 Proposed (Universal 10% + Retaliation)
    rw1_sched = {p_usa: 0.10, p_chn: 0.15, p_eur: 0.10, p_can: 0.10, p_mex: 0.10}
    tau_rw1, tau_fd_rw1 = build_strategic_tariffs(calib, rw1_sched)
    res_rw1 = _solve_policy_equilibrium(calib, tau=tau_rw1, tau_fd=tau_fd_rw1, x0=res_base.x_sol, method="condensed")
    imp_rw1 = calc_impacts(res_rw1, tau_rw1, tau_fd_rw1)

    # 8. 2024-2026 Punitive (60% CHN, 25% USMCA)
    rw2_sched = {
        p_usa: {p_chn: 0.60, p_can: 0.25, p_mex: 0.25},
        p_chn: {p_usa: 0.25},
        p_can: {p_usa: 0.25},
        p_mex: {p_usa: 0.25},
    }
    tau_rw2, tau_fd_rw2 = build_strategic_tariffs(calib, rw2_sched)
    res_rw2 = _solve_policy_equilibrium(calib, tau=tau_rw2, tau_fd=tau_fd_rw2, x0=res_rw1.x_sol, method="condensed")
    imp_rw2 = calc_impacts(res_rw2, tau_rw2, tau_fd_rw2)

    rows = [
        {
            "Policy_Regime": "Benchmark Free Trade (0%)",
            "tau_USA_%": 0.0,
            "tau_CHN_%": 0.0,
            "tau_EUR_%": 0.0,
            "tau_CAN_MEX_%": 0.0,
            "USA_Welfare_%": imp_ft[0],
            "CHN_Welfare_%": imp_ft[1],
            "EUR_Welfare_%": imp_ft[2],
            "World_Welfare_%": imp_ft[3],
        },
        {
            "Policy_Regime": "US Unilateral Optimal Tariff",
            "tau_USA_%": t_opt_us * 100.0,
            "tau_CHN_%": 0.0,
            "tau_EUR_%": 0.0,
            "tau_CAN_MEX_%": 0.0,
            "USA_Welfare_%": imp_us[0],
            "CHN_Welfare_%": imp_us[1],
            "EUR_Welfare_%": imp_us[2],
            "World_Welfare_%": imp_us[3],
        },
        {
            "Policy_Regime": "China Unilateral Optimal Tariff",
            "tau_USA_%": 0.0,
            "tau_CHN_%": t_opt_cn * 100.0,
            "tau_EUR_%": 0.0,
            "tau_CAN_MEX_%": 0.0,
            "USA_Welfare_%": imp_cn[0],
            "CHN_Welfare_%": imp_cn[1],
            "EUR_Welfare_%": imp_cn[2],
            "World_Welfare_%": imp_cn[3],
        },
        {
            "Policy_Regime": "EU Unilateral Optimal Tariff",
            "tau_USA_%": 0.0,
            "tau_CHN_%": 0.0,
            "tau_EUR_%": t_opt_eu * 100.0,
            "tau_CAN_MEX_%": 0.0,
            "USA_Welfare_%": imp_eu[0],
            "CHN_Welfare_%": imp_eu[1],
            "EUR_Welfare_%": imp_eu[2],
            "World_Welfare_%": imp_eu[3],
        },
        {
            "Policy_Regime": "Bilateral US--China Nash Game",
            "tau_USA_%": t_nash_us * 100.0,
            "tau_CHN_%": t_nash_cn * 100.0,
            "tau_EUR_%": 0.0,
            "tau_CAN_MEX_%": 0.0,
            "USA_Welfare_%": imp_bi[0],
            "CHN_Welfare_%": imp_bi[1],
            "EUR_Welfare_%": imp_bi[2],
            "World_Welfare_%": imp_bi[3],
        },
        {
            "Policy_Regime": "Multilateral 5-Bloc Nash War",
            "tau_USA_%": multi_sched.get(p_usa, 0.0) * 100.0,
            "tau_CHN_%": multi_sched.get(p_chn, 0.0) * 100.0,
            "tau_EUR_%": multi_sched.get(p_eur, 0.0) * 100.0,
            "tau_CAN_MEX_%": multi_sched.get(p_can, 0.0) * 100.0,
            "USA_Welfare_%": imp_multi[0],
            "CHN_Welfare_%": imp_multi[1],
            "EUR_Welfare_%": imp_multi[2],
            "World_Welfare_%": imp_multi[3],
        },
        {
            "Policy_Regime": "2024--2026 Proposed (Universal)",
            "tau_USA_%": 10.0,
            "tau_CHN_%": 15.0,
            "tau_EUR_%": 10.0,
            "tau_CAN_MEX_%": 10.0,
            "USA_Welfare_%": imp_rw1[0],
            "CHN_Welfare_%": imp_rw1[1],
            "EUR_Welfare_%": imp_rw1[2],
            "World_Welfare_%": imp_rw1[3],
        },
        {
            "Policy_Regime": "2024--2026 Punitive (60% CHN)",
            "tau_USA_%": 60.0,
            "tau_CHN_%": 25.0,
            "tau_EUR_%": 0.0,
            "tau_CAN_MEX_%": 25.0,
            "USA_Welfare_%": imp_rw2[0],
            "CHN_Welfare_%": imp_rw2[1],
            "EUR_Welfare_%": imp_rw2[2],
            "World_Welfare_%": imp_rw2[3],
        },
    ]

    return pd.DataFrame(rows).set_index("Policy_Regime")


__all__ = [
    "resolve_country_indices",
    "get_country_code",
    "evaluate_national_welfare",
    "build_strategic_tariffs",
    "compute_unilateral_optimal_tariff",
    "solve_multilateral_nash_tariffs",
    "compute_welfare_payoff_matrix",
    "benchmark_real_world_tariffs",
]
