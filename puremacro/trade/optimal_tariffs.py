"""Terms-of-trade optimal tariffs and multilateral Nash tariff games.

This module holds the non-cooperative trade-policy engine of puremacro.trade:

    1. National welfare evaluation (:func:`evaluate_national_welfare`):
       money-metric Hicksian equivalent variation, Geary-Khamis PPP real GDP,
       a Cobb-Douglas real-consumption index and the terms of trade.
    2. Unilateral terms-of-trade optimal tariffs
       (:func:`compute_unilateral_optimal_tariff`): the tariff that maximizes a
       country's (or bloc's) welfare against passive trading partners.
    3. Multilateral non-cooperative Nash tariffs
       (:func:`solve_multilateral_nash_tariffs`): a simultaneous tariff profile
       in which every player's tariff is a best response, found by damped
       Gauss-Seidel best-response iteration (or projected gradient ascent).
    4. Strategic welfare payoff matrices (:func:`compute_welfare_payoff_matrix`):
       2x2 fixed-action payoff matrices (zero tariff or one fixed tariff per
       player) with a Prisoner's Dilemma check. Whether a calibrated game is a
       Prisoner's Dilemma is computed from the payoffs, never assumed; a
       calibration can give mutual tariffs that raise one player's welfare.

Players and countries are named by the calibration's country codes (or
positions); no country is a default. Pure NumPy/SciPy/pandas, so it runs
under Pyodide.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from puremacro.trade._accounting import evaluate as _evaluate_accounting
from puremacro.trade._results import (
    NashTariffResult,
    OptimalTariffResult,
    TradeCalibrationResult,
    TradeEquilibriumResult,
    WelfarePayoffMatrixResult,
)
from puremacro.trade.data import EU_COUNTRY_CODES, synthetic_codes
from puremacro.trade.equilibrium import pack_equilibrium_vector
from puremacro.trade.geary_khamis import restore_capital_formation, solve_multilateral_ppp
from puremacro.trade.postprocessing import postprocess_trade_equilibrium
from puremacro.trade.solver import solve_trade_equilibrium


# ==============================================================================
# Helper Utilities: Country and Regional Bloc Resolution
# ==============================================================================

def _country_codes(calib: TradeCalibrationResult) -> list[str]:
    """The calibration's country codes, or the neutral ``C01, C02, ...`` when it has none."""
    if calib.country_codes:
        return list(calib.country_codes)
    return synthetic_codes(calib.n_countries, calib.n_sectors)[0]


def resolve_country_indices(
    calib: TradeCalibrationResult,
    country_ident: int | str | Sequence[str | int],
) -> list[int]:
    """Resolve country code(s) or regional bloc into integer indices within calib.

    Supports the calibration's country codes (e.g. 'FRA', 'JPN'), the
    customs union 'EUR' or 'EU' (every EU member state in the calibration),
    direct integer indices, or sequences thereof. A calibration without
    country codes is labelled ``C01, C02, ...``.

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
    country_codes = _country_codes(calib)
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


def get_country_code(calib: TradeCalibrationResult, country_idx: int | str | Sequence[str | int]) -> str:
    """Return the country code of a position, or the normalized code of an identifier.

    Raises ``IndexError`` for a position outside the calibration.
    """
    country_codes = _country_codes(calib)
    if isinstance(country_idx, str):
        return country_idx.strip().upper()
    if isinstance(country_idx, (int, np.integer)):
        idx = int(country_idx)
        if 0 <= idx < len(country_codes):
            return country_codes[idx]
        raise IndexError(f"Country index {country_idx} out of range (0 <= idx < {len(country_codes)}).")
    seq = list(country_idx)
    if not seq:
        raise ValueError("country_idx sequence cannot be empty")
    if len(seq) != 1:
        raise ValueError(
            "A strategic player must be one country or a named bloc such as 'EUR'; "
            "a sequence of multiple countries cannot be reduced to one country code."
        )
    return get_country_code(calib, seq[0])


def _reshape_state_tensor(arr: Any, ns: int, nc: int) -> np.ndarray:
    """Reshape a raw state input into ``(1, ns, nc)`` respecting Fortran vector packing."""
    a = np.asarray(arr, dtype=float)
    if a.shape == (1, ns, nc):
        return a
    if a.shape == (ns, nc):
        return a[np.newaxis, :, :]
    if a.ndim == 1 and a.size == ns * nc:
        return a.reshape((1, ns, nc), order="F")
    return a.reshape((1, ns, nc))


def _equilibrium_gk_prices_and_quantities(
    eq_obj: TradeEquilibriumResult,
    nfd: int,
    nc: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract purchaser final-demand prices and restored real national absorption volumes."""
    c_eq = eq_obj.c_fd if eq_obj.c_fd is not None else eq_obj.c_sol
    p_eq = getattr(eq_obj, "Pfd_final", None)
    if p_eq is None:
        p_eq = eq_obj.p_fd if eq_obj.p_fd is not None else eq_obj.pfd_sol
    if c_eq is None or p_eq is None:
        raise ValueError("Equilibrium object is missing final-demand prices or quantities.")
    p_arr = np.asarray(p_eq, dtype=float)
    p_2d = p_arr.reshape((nfd, nc)) if p_arr.ndim == 3 else p_arr
    q_arr = restore_capital_formation(c_eq, eq_obj.XN_sol, p_fd=p_2d)
    q_2d = q_arr.reshape((nfd, nc)) if q_arr.ndim == 3 else q_arr
    return p_2d, q_2d


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
    country_idx: int | str | Sequence[str | int] | None = None,
    metric: str = "geary_khamis",
    *,
    base_welfare: float | None = None,
    base_equilibrium: TradeEquilibriumResult | None = None,
    consumption_categories: Sequence[int] = (0,),
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    **kwargs: Any,
) -> float:
    """Evaluate national economic welfare for a sovereign economy or regional bloc.

    Supports these policy-objective metrics:
    1. ``'geary_khamis'`` (or ``'gk'``): Multilateral Geary-Khamis Purchasing Power
       Parity (PPP) real GDP in international reference dollars:
       :math:`W_c = \\sum_m \\pi_m q_{mc}`, valuing absorption at world shadow prices.
    2. ``'equivalent_variation'`` (or ``'ev'``): a Cobb-Douglas utility
       proxy (despite the alias, this is not money-metric Hicksian EV):
       :math:`U_c = Y_c^{\\text{disp}} / P_c^{\\text{CPI}}`.
    3. ``'terms_of_trade'`` (or ``'tot'``): Sovereign terms-of-trade index:
       :math:`\\text{TOT}_c = P_{X, c} / P_{M, c}`.

    ``metric="hicksian_ev"`` evaluates expenditure-function consumption EV in
    calibration value units. It requires consistent baseline and counterfactual
    equilibria, and sums monetary EV across bloc members. Select consumption
    baskets with ``consumption_categories`` (default ``(0,)``); investment is
    excluded. The aliases above keep their meaning. For CV and attribution,
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
    country_idx : int, str or sequence
        Target country position, country code, regional bloc ('EUR') or a
        sequence of them. Required; there is no default country.
    metric : {'hicksian_ev', 'geary_khamis', 'equivalent_variation', 'terms_of_trade'}, default 'geary_khamis'
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

    if country_idx is None:
        raise TypeError("evaluate_national_welfare() requires country_idx (a country code, position or bloc)")
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
        p_purchaser = getattr(eq, "Pfd_final", None)
        if p_purchaser is None:
            p_purchaser = p_fd
        cpi = eq.cpi
        tot = eq.terms_of_trade
    else:
        # Extract, reshape, and evaluate raw state arrays through consistent accounting
        if p_sol is None or y_sol is None or r_sol is None or w_sol is None or T_sol is None or XN_sol is None:
            raise ValueError("All state arrays (p, y, r, w, T, XN) are required when not passing TradeEquilibriumResult.")
        p_vec = _reshape_state_tensor(p_sol, ns, nc)
        y_vec = _reshape_state_tensor(y_sol, ns, nc)
        r_vec = _reshape_state_tensor(r_sol, 1, nc)
        w_vec = _reshape_state_tensor(w_sol, 1, nc)
        T_vec = _reshape_state_tensor(T_sol, 1, nc)
        XN_vec = np.asarray(XN_sol, dtype=float).ravel()
        if len(XN_vec) not in (nc - 1, nc):
            raise ValueError(f"Expected XN_sol length {nc - 1} or {nc} for {nc} countries, got {len(XN_vec)}.")

        x_raw = pack_equilibrium_vector(
            p_vec, y_vec, r_vec, w_vec, T_vec, XN_vec[: nc - 1], log_transformed=True
        )
        blocks = _evaluate_accounting(
            x_raw,
            calib,
            tau,
            tau_fd,
            kwargs.get("tauf"),
            kwargs.get("tauf_fd"),
            sigma=float(kwargs.get("sigma", 0.0)),
            foreign_saving_units=str(kwargs.get("foreign_saving_units", "numeraire")),
        )
        c_fd = blocks["c"]
        p_fd = blocks["Q"]
        p_purchaser = blocks["P"]
        cpi = None
        tot = None

    assert c_fd is not None and p_fd is not None and p_purchaser is not None

    # --------------------------------------------------------------------------
    # 1. Geary-Khamis PPP Real GDP
    # --------------------------------------------------------------------------
    if metric_clean in ("geary_khamis", "gk", "real_gdp", "ppp"):
        if base_equilibrium is not None:
            if "_gk_pi_linear" in base_equilibrium.metadata:
                pi_base = np.asarray(base_equilibrium.metadata["_gk_pi_linear"], dtype=float)
            else:
                p_base_2d, q_base_2d = _equilibrium_gk_prices_and_quantities(base_equilibrium, nfd, nc)
                pi_base, _, _, _ = solve_multilateral_ppp(p=p_base_2d, q=q_base_2d, method="linear")
                base_equilibrium.metadata["_gk_pi_linear"] = pi_base
            p_cf_2d = p_purchaser.reshape((nfd, nc)) if p_purchaser.ndim == 3 else p_purchaser
            q_cf = restore_capital_formation(c_fd, XN_vec, p_fd=p_cf_2d)
            q_cf_2d = q_cf.reshape((nfd, nc)) if q_cf.ndim == 3 else q_cf
            real_gdp = np.sum(q_cf_2d * pi_base[:, np.newaxis], axis=0)
        elif eq is not None and "gk_real_gdp" in eq.metadata:
            real_gdp = np.asarray(eq.metadata["gk_real_gdp"], dtype=float)
        else:
            p_use = p_purchaser.reshape((nfd, nc)) if p_purchaser.ndim == 3 else p_purchaser
            q = restore_capital_formation(c_fd, XN_vec, p_fd=p_use)
            q_use = q.reshape((nfd, nc)) if q.ndim == 3 else q
            pi, _, _, _ = solve_multilateral_ppp(p=p_use, q=q_use, method="linear")
            real_gdp = np.sum(q_use * pi[:, np.newaxis], axis=0)
            if eq is not None:
                eq.metadata["gk_real_gdp"] = real_gdp
                eq.metadata["_gk_pi_linear"] = pi

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

    raise ValueError(
        f"Unknown welfare metric '{metric}'. Valid options are: 'hicksian_ev', 'geary_khamis', 'equivalent_variation', 'terms_of_trade'."
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
        Example 1 (universal): ``{'FRA': 0.18, 'JPN': 0.15, 'EUR': 0.16}``
        Example 2 (bilateral): ``{'FRA': {'JPN': 0.25, 'BRA': 0.10}, 'JPN': {'FRA': 0.20}}``
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
    country_codes = _country_codes(calib)
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
    """Equilibrium solve of the non-Hicksian metrics; Hicksian policy uses policy_solver instead."""
    kwargs.setdefault("tol", 1e-8)
    result = solve_trade_equilibrium(calib, **kwargs)
    if not result.converged or not np.isfinite(result.max_residual):
        raise RuntimeError(f"Policy equilibrium did not converge (residual={result.max_residual})")
    return result


def compute_unilateral_optimal_tariff(
    calib: TradeCalibrationResult,
    country_idx: int | str | Sequence[str | int],
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
    Other metrics solve each equilibrium with damped Newton.

    Hicksian keyword options: ``consumption_categories=(0,)``,
    ``base_equilibrium=None``, ``sigma=0``, ``ge_tol=1e-8``, ``ge_method="auto"``,
    ``ge_max_iter=100``, ``ge_max_steps=100`` and
    ``foreign_saving_units="numeraire"``. Failed GE/refinement raises;
    no penalty payoff replaces a failed candidate.

    ``foreign_saving_units`` selects the foreign-saving closure of every
    equilibrium in the search (see :func:`solve_trade_equilibrium`). With
    nonzero baseline foreign saving, the default ``"numeraire"`` makes EV
    depend on which country is listed first; ``"world_income"`` gives results
    that do not depend on the country order. A supplied ``base_equilibrium``
    must have been solved with the same closure.

    ``metadata["boundary"]`` is ``"lower"``/``"upper"`` when the optimum lies
    within ``tol`` of 0/``tariff_max`` (the nearer bound when
    ``tariff_max <= 2 * tol``), else ``"interior"``, and
    ``metadata["boundary_bands_overlap"]`` is True when
    ``tariff_max <= 2 * tol``. Both are recorded for every ``metric``.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    country_idx : int, str or sequence
        Country (code or position) or bloc ('EUR') setting the tariff. Required.
    target_countries : Sequence[str], optional
        Target trading partners. If None, applies universally to all foreign partners.
    metric : {'hicksian_ev', 'geary_khamis', 'equivalent_variation', 'terms_of_trade'}, default 'geary_khamis'
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
        method="newton",
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
            method="newton",
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
    from ._hicksian_policy import _boundary_label

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
            "ge_tol": ge_tol,
            "search_scope": "grid plus refinement around the sampled maximum",
            # The same labels as the Hicksian search, which the docstring
            # promises on every metric; this path used to record neither.
            "boundary": _boundary_label(best_rate, tariff_max, tol),
            "boundary_bands_overlap": bool(tariff_max <= 2 * tol),
        },
    )


# ==============================================================================
# 3. Multilateral Non-Cooperative Nash Tariff Equilibrium
# ==============================================================================

def solve_multilateral_nash_tariffs(
    calib: TradeCalibrationResult,
    player_countries: Sequence[str | int] | None = None,
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
    refinement is not a global optimality proof.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    player_countries : Sequence[str | int]
        The strategic players: country codes, positions or blocs ('EUR').
        Required (also accepted as ``players=``); there are no default players.
    metric : {'hicksian_ev', 'geary_khamis', 'equivalent_variation', 'terms_of_trade'}, default 'geary_khamis'
        Sovereign national welfare objective function.
    method : {'best_response', 'gradient'}, default 'best_response'
        Game-theoretic fixed point algorithm:
        - 'best_response': Damped Gauss-Seidel best response policy iteration.
        - 'gradient': Finite-difference gradient ascent along sovereign welfare surfaces.
    relaxation : float, default 0.5
        Under-relaxation damping parameter :math:`\\theta \\in (0, 1]` preventing policy oscillations.
    tol : float, default 1e-4
        Absolute tolerance on tariff rates (1e-4 = 0.01 percentage points).
        ``converged`` requires the final simultaneous best-response gap
        :math:`\\max_i |BR_i(\\boldsymbol{\\tau}_{-i}) - \\tau_i| \\le \\text{tol}`,
        with best responses recomputed at the returned profile (reported as
        ``outer_error``), and relative regret ``<= regret_tol`` (default
        ``tol``). The iteration stops early once a sweep's undamped update gap
        :math:`\\max_i |BR_i - \\tau_i^{(k-1)}|` is ``<= tol`` (on the Hicksian
        path only if the final check then passes); damped step sizes never
        decide convergence. ``tol`` is also the distance within which a best
        response counts as ``"lower"``/``"upper"`` in
        ``metadata["best_response_boundaries"]``; on every path
        ``metadata["boundary_bands_overlap"]`` is True when
        ``tariff_max <= 2 * tol``, so the two bands overlap and the nearer
        bound decides the label.
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
        if player_countries is not None:
            raise TypeError("pass the strategic players once, as player_countries or players")
        player_countries = kwargs.pop("players")
    if player_countries is None:
        raise TypeError("solve_multilateral_nash_tariffs() requires player_countries (country codes or blocs)")
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
        method="newton",
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
                        method="newton", tol=ge_tol)
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
    tau_nash, tau_fd_nash = build_strategic_tariffs(calib, tau_curr, policy_mode=policy_mode)
    if all(v == 0.0 for v in tau_curr.values()):
        res_nash = res_base
    else:
        res_nash = solve_trade_equilibrium(
            calib,
            tau=tau_nash,
            tau_fd=tau_fd_nash,
            x0=x_state,
            method="newton",
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
    from ._hicksian_policy import _boundary_label
    regrets, responses, boundary = {}, {}, {}
    for p in players:
        rate, value = best_response(tau_curr, p)
        responses[p] = rate
        regrets[p] = max(0.0, value - player_welfares[p]) if np.isfinite(value) and np.isfinite(player_welfares[p]) else float("inf")
        boundary[p] = _boundary_label(rate, tariff_max, tol)
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
            "best_response_boundaries": boundary,
            # Same key as the Hicksian searches: True when the lower and upper
            # tolerance bands of the boundary labels overlap.
            "boundary_bands_overlap": bool(tariff_max <= 2 * tol),
            "inner_solver_failures": inner_failures,
            "best_response_grid_size": grid_size, "regret_tol": regret_tol, "ge_tol": ge_tol,
            "verification": "full-interval grid with bounded local refinement; no global-optimality proof",
            "relaxation": relaxation,
            "tol": tol,
            "max_iter": max_iter,
        },
    )


# ==============================================================================
# 4. Strategic Welfare Payoff Matrix (2x2 fixed actions, Prisoner's Dilemma check)
# ==============================================================================

def compute_welfare_payoff_matrix(
    calib: TradeCalibrationResult,
    player_a: str,
    player_b: str,
    metric: str = "geary_khamis",
    optimal_a: float | None = None,
    optimal_b: float | None = None,
    nash_a: float | None = None,
    nash_b: float | None = None,
    x0: np.ndarray | None = None,
    policy_mode: str = "universal",
    **kwargs: Any,
) -> WelfarePayoffMatrixResult:
    """Compute the 2x2 welfare payoff matrix of a two-player tariff game with fixed actions.

    Each player either sets a zero tariff ("cooperate", C) or one positive
    tariff ("defect", D). Cell ``(i, j)`` holds the welfare changes of A and B,
    in percent, relative to mutual zero tariffs:
    .. math::
        \\begin{array}{c|cc}
        & \\tau_B = 0 & \\tau_B = \\tau_B^{D} \\\\
        \\hline
        \\tau_A = 0 & (0, 0) & (W_A(0, \\tau_B^{D}), W_B(0, \\tau_B^{D})) \\\\
        \\tau_A = \\tau_A^{D} & (W_A(\\tau_A^{D}, 0), W_B(\\tau_A^{D}, 0))
            & (W_A(\\tau_A^{DD}, \\tau_B^{DD}), W_B(\\tau_A^{DD}, \\tau_B^{DD}))
        \\end{array}

    No cell has an assumed sign. Whether the game is a Prisoner's Dilemma
    (defection dominant for both, mutual defection worse for both than mutual
    cooperation) is computed from the returned payoffs
    (``WelfarePayoffMatrixResult.is_prisoners_dilemma``); a calibration can
    give mutual tariffs that raise a player's welfare. For the other metrics
    the off-diagonal cells use ``tau^D = optimal_*`` (unilateral optima
    computed when None) and the mutual-tariff cell uses ``tau^DD = nash_*``
    (computed with :func:`solve_multilateral_nash_tariffs`, which must
    converge, when either is None). With ``metric="hicksian_ev"`` each player
    uses one action in every cell, ``tau^DD = tau^D``.

    With ``metric="hicksian_ev"``, all cells share one fixed zero-tariff
    baseline and two fixed actions per player. Payoffs are EV as a percentage
    of baseline selected consumption. A positive action comes from
    ``optimal_*``, otherwise ``nash_*``, otherwise the unilateral optimum.
    Conflicting supplied rates raise; supplied ``nash_*`` are explicitly
    unverified fixed actions, never a continuous-game Nash certificate.
    Mutual positive actions carry no assumed welfare sign. Hicksian equilibrium
    options match ``compute_unilateral_optimal_tariff``.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    player_a : str
        First strategic player (country code or bloc). Required.
    player_b : str
        Second strategic player (country code or bloc). Required.
    metric : {'hicksian_ev', 'geary_khamis', 'equivalent_variation', 'terms_of_trade'}, default 'geary_khamis'
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
    res_cc = _solve_policy_equilibrium(calib, x0=x0, method="newton", tol=1e-8)
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
    res_dc = _solve_policy_equilibrium(calib, tau=tau_dc, tau_fd=tau_fd_dc, x0=res_cc.x_sol, method="newton")

    # Cell CD: Player A cooperates, Player B defects
    tau_cd, tau_fd_cd = build_strategic_tariffs(calib, {p_b: {p_a: rate_b}}, policy_mode=policy_mode)
    res_cd = _solve_policy_equilibrium(calib, tau=tau_cd, tau_fd=tau_fd_cd, x0=res_cc.x_sol, method="newton")

    # Cell DD: Mutual Defection (Trade War)
    tau_dd, tau_fd_dd = build_strategic_tariffs(
        calib, {p_a: {p_b: rate_nash_a}, p_b: {p_a: rate_nash_b}}, policy_mode=policy_mode
    )
    res_dd = _solve_policy_equilibrium(calib, tau=tau_dd, tau_fd=tau_fd_dd, x0=res_dc.x_sol, method="newton")

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


__all__ = [
    "resolve_country_indices",
    "get_country_code",
    "evaluate_national_welfare",
    "build_strategic_tariffs",
    "compute_unilateral_optimal_tariff",
    "solve_multilateral_nash_tariffs",
    "compute_welfare_payoff_matrix",
]
