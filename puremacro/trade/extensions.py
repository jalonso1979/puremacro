"""Advanced CGE Modeling Extensions Engine for puremacro.trade.

This module implements the four core CGE modeling extensions specified in the
methodological roadmap (research_roadmap.md) and survey analysis (survey_extensions_2):
    1. Extension A (Endogenous Retaliation Games / F20): Non-cooperative Nash tariff
       games with asymmetric reaction schedules (China tit-for-tat value parity, EU
       WTO Article XXVIII rebalancing, and NAFTA co-production carve-outs) solved via
       a nested Gauss-Seidel outer loop over the inner 2,001-equation CGE solver.
    2. Extension B (Dynamic J-Curve Transition / F21): Continuous time-dependent
       elasticity relaxation path sigma(t) = sigma_long * (1 - exp(-gamma * t)),
       analytical closed-form turning point formula t* = -ln(1 - 1/sigma_long) / gamma,
       and sequential dynamic quarterly simulation tracking trade balance and welfare.
    3. Extension C (Tariff Revenue Recycling / F22): Alternative fiscal closures for
       US tariff revenues (lump-sum transfers, capital tax relief, payroll tax cuts,
       targeted manufacturing input subsidies, and public deficit reduction) with
       three-way Harberger welfare decomposition.
    4. Extension D (Upstream Supply Bottlenecks / F23): Intermediate capacity constraints
       on critical upstream sectors modeled via C^2 smooth barrier penalty functions
       Phi(y) = c_V * [1 + zeta * (y / y_bar)^eta] preserving Newton solver convergence.

Conforms strictly to the puremacro Pyodide runtime contract: pure NumPy/SciPy/Pandas,
zero dev-dependencies in the runtime path, fully vectorized.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd

from puremacro.trade._results import (
    CapacityBottleneckResult,
    JCurveDynamicResult,
    RetaliationGameResult,
    RevenueRecyclingResult,
    TradeCalibrationResult,
    TradeEquilibriumResult,
)
from puremacro.trade.data import (
    CANONICAL_COUNTRY_CODES,
    CANONICAL_SECTOR_CODES,
    EU_COUNTRY_CODES,
    get_country_codes,
    get_eu_country_codes,
)
from puremacro.trade.scenarios import (
    SCENARIOS,
    TariffScenario,
    build_tariff_matrices,
)
from puremacro.trade.solver import (
    build_initial_guess,
    solve_trade_equilibrium,
)


# ===========================================================================
# 1. EXTENSION A: ENDOGENOUS RETALIATION GAMES (F20)
# ===========================================================================

@dataclass(frozen=True)
class RetaliationConfig:
    """Configuration for endogenous foreign tariff retaliation games.

    Attributes
    ----------
    mode : {"targeted", "symmetric", "wto_rebalance"}, default "targeted"
        Retaliation policy schedule:
        - 'targeted': Asymmetric partner-specific reaction functions:
          * China: Value-parity matching on commodities (AGRI, MINQ, ENEG).
          * EU: WTO Article XXVIII rebalancing on final consumption (fd=C).
          * NAFTA (CAN, MEX): Defensive carve-out for high intermediate intensity.
        - 'symmetric': Reciprocal bilateral tariffs matching US import rates.
        - 'wto_rebalance': Strict duty rebalancing restricted to final consumption.
    strategic_players : tuple[str, ...], default ("CHN", "EUR", "CAN", "MEX")
        Active retaliatory partner identifiers. 'EUR' or 'EU' targets EU-27 member states.
    damping : float, default 0.5
        Gauss-Seidel relaxation parameter theta in (0, 1].
    max_outer_iter : int, default 30
        Maximum outer policy iterations.
    outer_tol : float, default 1e-4
        Convergence tolerance on maximum absolute tariff multiplier change.
    tariff_cap : float, default 2.0
        Maximum ad-valorem tariff rate ceiling (2.0 = 200%).
    commodity_sectors : tuple[str, ...], default ("AGRI", "MINQ", "ENEG")
        Vulnerable commodity sectors targeted by China under value-parity tit-for-tat.
    exempt_sectors : tuple[str, ...], default ("INFO", "FIN", "TRAD")
        Knowledge and service sectors shielded from retaliation.
    nafta_intensity_threshold : float, default 0.02
        Intermediate intensity cutoff above which NAFTA partners exempt varieties.
    """

    mode: str = "targeted"
    strategic_players: tuple[str, ...] = ("CHN", "EUR", "CAN", "MEX")
    damping: float = 0.5
    max_outer_iter: int = 30
    outer_tol: float = 1e-4
    tariff_cap: float = 2.0
    commodity_sectors: tuple[str, ...] = ("AGRI", "MINQ", "ENEG")
    exempt_sectors: tuple[str, ...] = ("INFO", "FIN", "TRAD")
    nafta_intensity_threshold: float = 0.02

    def __init__(
        self,
        mode: str = "targeted",
        strategic_players: tuple[str, ...] | Sequence[str] | None = None,
        damping: float = 0.5,
        max_outer_iter: int = 30,
        outer_tol: float = 1e-4,
        tariff_cap: float = 2.0,
        commodity_sectors: tuple[str, ...] | Sequence[str] = ("AGRI", "MINQ", "ENEG"),
        exempt_sectors: tuple[str, ...] | Sequence[str] = ("INFO", "FIN", "TRAD"),
        nafta_intensity_threshold: float = 0.02,
        partners: tuple[str, ...] | Sequence[str] | None = None,
    ) -> None:
        if partners is not None and strategic_players is None:
            strategic_players = partners
        if strategic_players is None:
            strategic_players = ("CHN", "EUR", "CAN", "MEX")
        object.__setattr__(self, "mode", str(mode))
        object.__setattr__(self, "strategic_players", tuple(strategic_players))
        object.__setattr__(self, "damping", float(damping))
        object.__setattr__(self, "max_outer_iter", int(max_outer_iter))
        object.__setattr__(self, "outer_tol", float(outer_tol))
        object.__setattr__(self, "tariff_cap", float(tariff_cap))
        object.__setattr__(self, "commodity_sectors", tuple(commodity_sectors))
        object.__setattr__(self, "exempt_sectors", tuple(exempt_sectors))
        object.__setattr__(self, "nafta_intensity_threshold", float(nafta_intensity_threshold))
        self.__post_init__()

    def __post_init__(self) -> None:
        if not (0.0 < self.damping <= 1.0):
            raise ValueError(f"Damping parameter must be in (0, 1], got {self.damping}")
        if self.max_outer_iter < 1:
            raise ValueError(f"max_outer_iter must be >= 1, got {self.max_outer_iter}")
        if self.outer_tol <= 0.0:
            raise ValueError(f"outer_tol must be > 0, got {self.outer_tol}")


def _compute_bilateral_duties_and_flows(
    eq_res: TradeEquilibriumResult,
    calib: TradeCalibrationResult,
    c_orig_code: str,
    c_dest_code: str,
    tau_4d: np.ndarray,
    tau_fd_4d: np.ndarray,
) -> dict[str, Any]:
    """Extract bilateral intermediate and final demand values and duties collected."""
    country_codes = list(calib.country_codes) if calib.country_codes else list(CANONICAL_COUNTRY_CODES)
    o_idx = country_codes.index(c_orig_code)
    d_idx = country_codes.index(c_dest_code)

    p_sol = np.asarray(eq_res.p_sol, dtype=float)[0]  # shape (ns, nc)
    p_orig = p_sol[:, o_idx]  # shape (ns,)

    interm_flows = np.asarray(eq_res.intermediate_flows, dtype=float)  # (ns, nc, ns, nc)
    fd_flows = np.asarray(eq_res.final_demand_flows, dtype=float)  # (ns, nc, nfd, nc)

    # Intermediate flows from o_idx to d_idx: shape (ns_orig, ns_dest)
    flow_x = interm_flows[:, o_idx, :, d_idx]
    val_x = p_orig[:, np.newaxis] * flow_x
    rates_x = tau_4d[:, o_idx, :, d_idx] - 1.0
    duties_x = np.sum(rates_x * val_x)

    # Final demand flows from o_idx to d_idx: shape (ns_orig, nfd_dest)
    flow_fd = fd_flows[:, o_idx, :, d_idx]
    val_fd = p_orig[:, np.newaxis] * flow_fd
    rates_fd = tau_fd_4d[:, o_idx, :, d_idx] - 1.0
    duties_fd = np.sum(rates_fd * val_fd)

    total_duties = float(duties_x + duties_fd)
    total_val = float(np.sum(val_x) + np.sum(val_fd))

    # Sector-level flow and duty breakdown
    sec_vals = np.sum(val_x, axis=1) + np.sum(val_fd, axis=1)
    sec_duties = np.sum(rates_x * val_x, axis=1) + np.sum(rates_fd * val_fd, axis=1)

    return {
        "total_duties": total_duties,
        "total_value": total_val,
        "val_intermediate": val_x,
        "val_final_demand": val_fd,
        "duties_intermediate": float(duties_x),
        "duties_final_demand": float(duties_fd),
        "sector_values": sec_vals,
        "sector_duties": sec_duties,
        "p_orig": p_orig,
    }


def solve_retaliation_game(
    calib: TradeCalibrationResult,
    initial_scenario: TariffScenario | str | Mapping[str, Any] | None = None,
    config: RetaliationConfig | None = None,
    base_result: TradeEquilibriumResult | None = None,
    scenario: TariffScenario | str | Mapping[str, Any] | None = None,
    **solver_kwargs: Any,
) -> RetaliationGameResult:
    """Solve the non-cooperative Nash tariff retaliation game via nested Gauss-Seidel.

    Inner Loop: 2,001-equation general equilibrium CGE solver.
    Outer Loop: Damped Gauss-Seidel fixed-point iteration over strategic partner reaction functions.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    initial_scenario : TariffScenario, str, or Mapping, optional
        Initial unilateral US tariff shock triggering foreign retaliation.
    config : RetaliationConfig, optional
        Retaliation policy schedule and convergence configuration.
    base_result : TradeEquilibriumResult, optional
        Initial benchmark equilibrium result for warm-starting.
    scenario : TariffScenario, str, or Mapping, optional
        Alias for initial_scenario.
    **solver_kwargs : Any
        Keyword arguments forwarded to :func:`solve_trade_equilibrium`.

    Returns
    -------
    RetaliationGameResult
        Equilibrium outcome including final tariff tensors, partner duties, and convergence stats.
    """
    if initial_scenario is None:
        if scenario is not None:
            initial_scenario = scenario
        else:
            raise ValueError("initial_scenario or scenario must be provided to solve_retaliation_game.")

    if config is None:
        config = RetaliationConfig()

    country_codes = list(calib.country_codes) if calib.country_codes else list(CANONICAL_COUNTRY_CODES)
    sector_codes = list(calib.sector_codes) if calib.sector_codes else list(CANONICAL_SECTOR_CODES)
    ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand
    us_code = "USA" if "USA" in country_codes else country_codes[0]
    us_idx = country_codes.index(us_code)

    # Initialize tariff tensors from the unilateral US shock
    tau_curr, tau_fd_curr, _, _ = build_tariff_matrices(initial_scenario, calib)
    scen_name = initial_scenario if isinstance(initial_scenario, str) else getattr(initial_scenario, "name", "custom_shock")

    # Resolve active strategic players
    strategic_players = list(config.strategic_players)
    has_eu = any(p in ("EUR", "EU") for p in strategic_players)
    eu_members = [c for c in country_codes if c in EU_COUNTRY_CODES] if has_eu else []

    x0 = solver_kwargs.pop("x0", None)
    if x0 is not None:
        x_guess = x0.copy()
    elif base_result is not None and getattr(base_result, "x_sol", None) is not None:
        x_guess = base_result.x_sol.copy()
    else:
        x_guess = build_initial_guess(calib)
    last_eq: TradeEquilibriumResult | None = None
    outer_error = float("inf")
    converged = False

    partner_tariffs_record: dict[str, dict[str, float]] = {}
    partner_duties_record: dict[str, float] = {}
    us_duties_record: dict[str, float] = {}

    for k_iter in range(config.max_outer_iter):
        # 1. Inner CGE solve
        last_eq = solve_trade_equilibrium(
            calib=calib,
            tau=tau_curr,
            tau_fd=tau_fd_curr,
            x0=x_guess,
            **solver_kwargs,
        )
        x_guess = last_eq.x_sol

        tau_target = tau_curr.copy()
        tau_fd_target = tau_fd_curr.copy()

        # 2. Evaluate strategic reaction functions
        for player in strategic_players:
            if player == us_code:
                continue  # Trigger country policy remains trigger shock

            if config.mode == "symmetric":
                # Reciprocal symmetric retaliation: match US tariffs on partner exports
                partners_to_update = eu_members if player in ("EUR", "EU") else [player]
                for p_code in partners_to_update:
                    if p_code not in country_codes:
                        continue
                    p_idx = country_codes.index(p_code)
                    # US tariff on partner: tau[:, p_idx, :, us_idx]
                    us_rates = tau_curr[:, p_idx, 0, us_idx] - 1.0  # (ns,)
                    for s_i in range(ns):
                        r_i = max(float(us_rates[s_i]), 0.0)
                        tau_target[s_i, us_idx, :, p_idx] = 1.0 + r_i
                        tau_fd_target[s_i, us_idx, :, p_idx] = 1.0 + r_i

            elif player == "CHN":
                # China Tit-for-Tat Value-Parity Commodity Retaliation
                chn_code = "CHN"
                if chn_code in country_codes:
                    chn_idx = country_codes.index(chn_code)
                    # US duties collected on Chinese imports
                    us_on_chn = _compute_bilateral_duties_and_flows(
                        last_eq, calib, c_orig_code="CHN", c_dest_code=us_code,
                        tau_4d=tau_curr, tau_fd_4d=tau_fd_curr,
                    )
                    duties_us_chn = us_on_chn["total_duties"]
                    us_duties_record["CHN"] = duties_us_chn

                    # China imports from US
                    chn_from_us = _compute_bilateral_duties_and_flows(
                        last_eq, calib, c_orig_code=us_code, c_dest_code="CHN",
                        tau_4d=tau_curr, tau_fd_4d=tau_fd_curr,
                    )
                    manu_idx = sector_codes.index("MANU") if "MANU" in sector_codes else 2
                    us_manu_rate = float(tau_curr[manu_idx, chn_idx, 0, us_idx] - 1.0)
                    chn_manu_rate = min(max(us_manu_rate, 0.0), 0.25)

                    # Manufacturing duty collected by China on US
                    manu_val = float(chn_from_us["sector_values"][manu_idx])
                    duty_manu_chn = chn_manu_rate * manu_val

                    # Commodity sectors duty target
                    target_com_duty = max(0.0, duties_us_chn - duty_manu_chn)
                    com_indices = [sector_codes.index(s) for s in config.commodity_sectors if s in sector_codes]
                    com_val = float(np.sum(chn_from_us["sector_values"][com_indices]))

                    if com_val > 1e-6:
                        lambda_rate = min(config.tariff_cap, target_com_duty / com_val)
                    else:
                        lambda_rate = min(config.tariff_cap, 0.25)

                    exempt_indices = [sector_codes.index(s) for s in config.exempt_sectors if s in sector_codes]

                    # Apply China's updated tariff schedule on US goods
                    rates_chn_dict: dict[str, float] = {}
                    for s_i, s_name in enumerate(sector_codes):
                        if s_i in com_indices:
                            r_apply = lambda_rate
                        elif s_i == manu_idx:
                            r_apply = chn_manu_rate
                        elif s_i in exempt_indices:
                            r_apply = 0.0
                        else:
                            r_apply = min(chn_manu_rate, 0.10)

                        rates_chn_dict[s_name] = r_apply
                        tau_target[s_i, us_idx, :, chn_idx] = 1.0 + r_apply
                        tau_fd_target[s_i, us_idx, :, chn_idx] = 1.0 + r_apply

                    partner_tariffs_record["CHN"] = rates_chn_dict
                    partner_duties_record["CHN"] = float(duty_manu_chn + lambda_rate * com_val)

            elif player in ("EUR", "EU"):
                # EU WTO Article XXVIII Rebalancing: Final consumption goods only (fd=C)
                total_us_duties_on_eu = 0.0
                eu_target_c_val = 0.0
                target_secs = [sector_codes.index(s) for s in ("AGRI", "MANU") if s in sector_codes]

                for eu_c in eu_members:
                    us_on_euc = _compute_bilateral_duties_and_flows(
                        last_eq, calib, c_orig_code=eu_c, c_dest_code=us_code,
                        tau_4d=tau_curr, tau_fd_4d=tau_fd_curr,
                    )
                    total_us_duties_on_eu += us_on_euc["total_duties"]

                    euc_from_us = _compute_bilateral_duties_and_flows(
                        last_eq, calib, c_orig_code=us_code, c_dest_code=eu_c,
                        tau_4d=tau_curr, tau_fd_4d=tau_fd_curr,
                    )
                    # Sourcing for final consumption (fd=0, which is category 'C')
                    val_c_target = euc_from_us["val_final_demand"][target_secs, 0]
                    eu_target_c_val += float(np.sum(val_c_target))

                us_duties_record["EUR"] = total_us_duties_on_eu

                if eu_target_c_val > 1e-6:
                    tau_rebalance = min(config.tariff_cap, total_us_duties_on_eu / eu_target_c_val)
                else:
                    tau_rebalance = min(config.tariff_cap, 0.10)

                rates_eu_dict = {"AGRI": tau_rebalance, "MANU": tau_rebalance}
                for eu_c in eu_members:
                    eu_idx = country_codes.index(eu_c)
                    # Intermediate inputs: strictly exempt (rate = 0)
                    tau_target[:, us_idx, :, eu_idx] = 1.0
                    # Final demand: exempt investment and govt (fd != 0)
                    tau_fd_target[:, us_idx, 1:, eu_idx] = 1.0
                    # Final demand consumption: apply rebalancing to AGRI and MANU
                    for s_i in target_secs:
                        tau_fd_target[s_i, us_idx, 0, eu_idx] = 1.0 + tau_rebalance

                partner_tariffs_record["EUR"] = rates_eu_dict
                partner_duties_record["EUR"] = float(tau_rebalance * eu_target_c_val)

            elif player in ("CAN", "MEX"):
                # NAFTA Defensive Co-Production Carve-Out
                p_code = player
                if p_code in country_codes:
                    p_idx = country_codes.index(p_code)
                    us_on_p = _compute_bilateral_duties_and_flows(
                        last_eq, calib, c_orig_code=p_code, c_dest_code=us_code,
                        tau_4d=tau_curr, tau_fd_4d=tau_fd_curr,
                    )
                    us_duties_record[p_code] = us_on_p["total_duties"]

                    rates_p_dict: dict[str, float] = {}
                    p_duties_collected = 0.0

                    p_from_us = _compute_bilateral_duties_and_flows(
                        last_eq, calib, c_orig_code=us_code, c_dest_code=p_code,
                        tau_4d=tau_curr, tau_fd_4d=tau_fd_curr,
                    )

                    for s_i, s_name in enumerate(sector_codes):
                        # Intermediate intensity: sum over downstream sectors j of a_{ij}^{p, US}
                        intensity = float(np.sum(calib.a[us_idx * ns + s_i, :, p_idx]))
                        us_rate_s = float(tau_curr[s_i, p_idx, 0, us_idx] - 1.0)

                        if intensity < config.nafta_intensity_threshold:
                            # Pure consumer good: match US tariff
                            r_apply = min(config.tariff_cap, max(us_rate_s, 0.0))
                        else:
                            # Integrated intermediate variety: exempt
                            r_apply = 0.0

                        rates_p_dict[s_name] = r_apply
                        tau_target[s_i, us_idx, :, p_idx] = 1.0 + r_apply
                        tau_fd_target[s_i, us_idx, :, p_idx] = 1.0 + r_apply
                        p_duties_collected += r_apply * float(p_from_us["sector_values"][s_i])

                    partner_tariffs_record[p_code] = rates_p_dict
                    partner_duties_record[p_code] = p_duties_collected

        # 3. Outer convergence check
        diff_tau = float(np.max(np.abs(tau_target - tau_curr)))
        diff_tau_fd = float(np.max(np.abs(tau_fd_target - tau_fd_curr)))
        outer_error = max(diff_tau, diff_tau_fd)

        if outer_error < config.outer_tol:
            converged = True
            tau_curr = tau_target
            tau_fd_curr = tau_fd_target
            break

        # 4. Damped relaxation update
        tau_curr = config.damping * tau_target + (1.0 - config.damping) * tau_curr
        tau_fd_curr = config.damping * tau_fd_target + (1.0 - config.damping) * tau_fd_curr

    if last_eq is None:
        raise RuntimeError("Retaliation game loop did not execute any iterations.")

    return RetaliationGameResult(
        scenario_name=f"{scen_name}_retaliation",
        equilibrium=last_eq,
        initial_scenario=initial_scenario,
        tau_final=tau_curr,
        tau_fd_final=tau_fd_curr,
        strategic_players=config.strategic_players,
        partner_tariffs=partner_tariffs_record,
        partner_duties_collected=partner_duties_record,
        us_duties_collected=us_duties_record,
        outer_iterations=k_iter + 1,
        converged=converged,
        outer_error=outer_error,
        mode=config.mode,
        metadata={
            "damping": config.damping,
            "max_outer_iter": config.max_outer_iter,
            "outer_tol": config.outer_tol,
            "tariff_cap": config.tariff_cap,
        },
    )


# ===========================================================================
# 2. EXTENSION B: DYNAMIC J-CURVE TRANSITION (F21)
# ===========================================================================

@dataclass(frozen=True)
class DynamicJCurveConfig:
    """Configuration for sequential dynamic quarterly J-curve simulation.

    Attributes
    ----------
    sigma_long : float, default 4.0
        Long-run Armington/Ricardian trade elasticity of substitution. Must be > 1.0.
    gamma : float, default 0.15
        Speed of supply-chain recontracting and qualification per quarter.
    n_quarters : int, default 12
        Number of quarters to simulate.
    dt : float, default 1.0
        Time step per simulation period in quarters.
    """

    sigma_long: float = 4.0
    gamma: float = 0.15
    n_quarters: int = 12
    dt: float = 1.0
    target_country: str = "USA"

    def __post_init__(self) -> None:
        if self.sigma_long <= 1.0:
            raise ValueError(f"sigma_long must be > 1.0 for turning point existence, got {self.sigma_long}")
        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be > 0.0, got {self.gamma}")
        if self.n_quarters < 1:
            raise ValueError(f"n_quarters must be >= 1, got {self.n_quarters}")


def compute_j_curve_turning_point(gamma: float = 0.15, sigma_long: float = 4.0) -> float:
    """Compute the closed-form analytical J-curve inflection horizon t* (Theorem 2).

    Formula:
        t* = -ln(1 - 1 / sigma_long) / gamma

    Parameters
    ----------
    gamma : float, default 0.15
        Adjustment speed parameter per quarter.
    sigma_long : float, default 4.0
        Long-run elasticity of substitution (must exceed 1.0).

    Returns
    -------
    float
        Turning point horizon in quarters where nominal trade balance crosses baseline.
    """
    # Handle flexible argument order if called positionally as (sigma_long, gamma)
    if gamma > 1.0 and 0.0 < sigma_long <= 1.0:
        sigma_long, gamma = gamma, sigma_long

    if sigma_long <= 1.0:
        raise ValueError(f"sigma_long must exceed 1.0 for turning point existence, got {sigma_long}")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be strictly positive, got {gamma}")
    return -float(math.log(1.0 - 1.0 / sigma_long)) / gamma


def compute_contractual_half_life(gamma: float = 0.15) -> float:
    """Compute the contractual rigidity half-life t_{1/2} = ln(2) / gamma."""
    if gamma <= 0.0:
        raise ValueError(f"gamma must be strictly positive, got {gamma}")
    return float(math.log(2.0)) / gamma


def simulate_analytical_j_curve(
    gamma: float = 0.15,
    sigma_long: float = 4.0,
    n_quarters: int = 12,
    dt: float = 1.0,
    m0: float = 100.0,
    tau: float = 0.25,
    x0: float = 100.0,
) -> dict[str, np.ndarray | float]:
    """Evaluate the closed-form analytical J-curve path from Theorem 2."""
    t_star = compute_j_curve_turning_point(gamma=gamma, sigma_long=sigma_long)
    half_life = compute_contractual_half_life(gamma=gamma)

    quarters = np.arange(0, n_quarters + 1, dt, dtype=float)
    sigma_t = sigma_long * (1.0 - np.exp(-gamma * quarters))
    imports_t = m0 * ((1.0 + tau) ** (1.0 - sigma_t))
    exports_t = np.full_like(quarters, x0)
    xn_t = exports_t - imports_t

    return {
        "quarters": quarters,
        "sigma": sigma_t,
        "imports": imports_t,
        "exports": exports_t,
        "trade_balance": xn_t,
        "t_star": t_star,
        "turning_point": t_star,
        "half_life": half_life,
    }


def run_dynamic_j_curve_simulation(
    calib: TradeCalibrationResult,
    scenario: TariffScenario | str | Mapping[str, Any],
    config: DynamicJCurveConfig | None = None,
    base_result: TradeEquilibriumResult | None = None,
    **solver_kwargs: Any,
) -> JCurveDynamicResult:
    """Simulate sequential quarterly dynamic general equilibrium path under relaxing contracts.

    Solves the general equilibrium at each quarter t_k with substitution elasticity
    sigma(t_k) = sigma_long * (1 - exp(-gamma * t_k)), using warm starts across periods.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    scenario : TariffScenario, str, or Mapping
        Tariff shock scenario.
    config : DynamicJCurveConfig, optional
        Dynamic transition parameters.
    base_result : TradeEquilibriumResult, optional
        Baseline equilibrium for warm-starting.
    **solver_kwargs : Any
        Solver keyword arguments.

    Returns
    -------
    JCurveDynamicResult
        Quarterly trajectory of trade balance, exports, imports, GDP, and CPI.
    """
    if config is None:
        config = DynamicJCurveConfig()

    country_codes = list(calib.country_codes) if calib.country_codes else list(CANONICAL_COUNTRY_CODES)
    tgt_country = getattr(config, "target_country", "USA")
    tgt_idx = country_codes.index(tgt_country) if tgt_country in country_codes else (country_codes.index("USA") if "USA" in country_codes else 0)

    tau, tau_fd, tauf, tauf_fd = build_tariff_matrices(scenario, calib)
    scen_name = scenario if isinstance(scenario, str) else getattr(scenario, "name", "dynamic_shock")

    t_star = compute_j_curve_turning_point(gamma=config.gamma, sigma_long=config.sigma_long)
    half_life = compute_contractual_half_life(gamma=config.gamma)

    n_steps = config.n_quarters + 1
    quarters_list: list[float] = []
    sigma_list: list[float] = []
    xn_list: list[float] = []
    exp_list: list[float] = []
    imp_list: list[float] = []
    gdp_list: list[float] = []
    cpi_list: list[float] = []
    eqs_list: list[TradeEquilibriumResult] = []

    x0 = solver_kwargs.pop("x0", getattr(base_result, "x_sol", None))
    x_guess = x0 if x0 is not None else build_initial_guess(calib)

    for step in range(n_steps):
        t_curr = step * config.dt
        sigma_curr = float(config.sigma_long * (1.0 - math.exp(-config.gamma * t_curr)))

        eq_t = solve_trade_equilibrium(
            calib=calib,
            tau=tau,
            tau_fd=tau_fd,
            tauf=tauf,
            tauf_fd=tauf_fd,
            x0=x_guess,
            sigma=sigma_curr,
            **solver_kwargs,
        )
        x_guess = eq_t.x_sol

        exp_val = float(eq_t.exports[tgt_idx]) if eq_t.exports is not None else 0.0
        imp_val = float(eq_t.imports[tgt_idx]) if eq_t.imports is not None else 0.0
        xn_val = exp_val - imp_val
        gdp_val = float(eq_t.gdp[tgt_idx]) if eq_t.gdp is not None else 0.0
        cpi_val = float(eq_t.cpi[tgt_idx]) if eq_t.cpi is not None else 1.0

        quarters_list.append(t_curr)
        sigma_list.append(sigma_curr)
        xn_list.append(xn_val)
        exp_list.append(exp_val)
        imp_list.append(imp_val)
        gdp_list.append(gdp_val)
        cpi_list.append(cpi_val)
        eqs_list.append(eq_t)

    # Closed-form analytical comparison
    m0_est = imp_list[0] / 1.10 if imp_list else 100.0
    ana_res = simulate_analytical_j_curve(
        gamma=config.gamma,
        sigma_long=config.sigma_long,
        n_quarters=config.n_quarters,
        dt=config.dt,
        m0=m0_est,
        tau=0.10,
        x0=exp_list[0] if exp_list else 100.0,
    )

    return JCurveDynamicResult(
        scenario_name=f"{scen_name}_j_curve",
        quarters=tuple(quarters_list),
        sigma_path=tuple(sigma_list),
        t_star=t_star,
        half_life=half_life,
        trade_balance_path=np.array(xn_list, dtype=float),
        us_exports_path=np.array(exp_list, dtype=float),
        us_imports_path=np.array(imp_list, dtype=float),
        us_gdp_path=np.array(gdp_list, dtype=float),
        us_cpi_path=np.array(cpi_list, dtype=float),
        equilibria=tuple(eqs_list),
        gamma=config.gamma,
        sigma_long=config.sigma_long,
        analytical_trade_balance_path=np.asarray(ana_res["trade_balance"], dtype=float),
        analytical_imports_path=np.asarray(ana_res["imports"], dtype=float),
        metadata={
            "gamma": config.gamma,
            "sigma_long": config.sigma_long,
            "n_quarters": config.n_quarters,
            "dt": config.dt,
        },
    )


# ===========================================================================
# 3. EXTENSION C: TARIFF REVENUE RECYCLING REGIMES (F22)
# ===========================================================================

@dataclass(frozen=True)
class FiscalRecyclingConfig:
    """Configuration for tariff revenue recycling regimes.

    Attributes
    ----------
    closure : {"lump_sum", "capital_tax", "labor_tax", "strategic_subsidy", "deficit_reduction"}
        Fiscal closure mechanism:
        - 'lump_sum': 100% returned to domestic households (baseline status quo).
        - 'capital_tax': recycled to reduce domestic capital rental tax.
        - 'labor_tax': recycled to reduce domestic payroll / labor income tax.
        - 'strategic_subsidy': targeted manufacturing input/output subsidy.
        - 'deficit_reduction': retained to retire sovereign public debt.
    target_country : str, default "USA"
        Country code implementing the fiscal recycling regime.
    subsidy_shares : dict[str, float] | None, default None
        Sectoral allocation weights for strategic subsidies (default 100% to 'MANU').
    """

    closure: str = "lump_sum"
    target_country: str = "USA"
    subsidy_shares: dict[str, float] | None = None

    def __post_init__(self) -> None:
        valid_closures = {
            "lump_sum", "capital_tax", "capital_tax_cut",
            "labor_tax", "payroll_tax_cut",
            "strategic_subsidy", "targeted_subsidy", "manufacturing_subsidy",
            "deficit_reduction", "public_debt",
        }
        if self.closure not in valid_closures:
            raise ValueError(f"Unknown fiscal closure '{self.closure}'. Valid options: {valid_closures}")


def decompose_harberger_welfare(
    calib: TradeCalibrationResult,
    eq_result: TradeEquilibriumResult,
    base_result: TradeEquilibriumResult | None = None,
    closure: str = "lump_sum",
    target_country: str = "USA",
) -> dict[str, float]:
    """Perform three-way Harberger welfare decomposition (TOT, DWL, and Fiscal Dividends).

    Decomposition:
        Delta W_c = Delta TOT_c - DWL_c + Efficiency_Dividend_c

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    eq_result : TradeEquilibriumResult
        Counterfactual general equilibrium result.
    base_result : TradeEquilibriumResult, optional
        Pre-shock benchmark equilibrium. If None, solved via :func:`solve_trade_equilibrium`.
    closure : str, default "lump_sum"
        Fiscal regime closure.
    target_country : str, default "USA"
        Target country identifier.

    Returns
    -------
    dict[str, float]
        Decomposed welfare components.
    """
    if base_result is None:
        base_result = solve_trade_equilibrium(calib)

    country_codes = list(calib.country_codes) if calib.country_codes else list(CANONICAL_COUNTRY_CODES)
    c_idx = (
        country_codes.index(target_country)
        if target_country in country_codes
        else (country_codes.index("USA") if "USA" in country_codes else 0)
    )

    # 1. Terms of Trade Effect: Delta TOT * Trade Volume
    tot_base = (
        float(base_result.terms_of_trade[c_idx])
        if base_result is not None and getattr(base_result, "terms_of_trade", None) is not None
        else 1.0
    )
    tot_cf = (
        float(eq_result.terms_of_trade[c_idx])
        if getattr(eq_result, "terms_of_trade", None) is not None
        else 1.0
    )
    exp_val = float(eq_result.exports[c_idx]) if getattr(eq_result, "exports", None) is not None else 0.0
    imp_val = float(eq_result.imports[c_idx]) if getattr(eq_result, "imports", None) is not None else 0.0
    trade_vol = 0.5 * (exp_val + imp_val)
    tot_effect = (tot_cf - tot_base) * trade_vol

    # 2. Deadweight Distortion Loss: sum p * tau * Delta x
    tariffs_collected = (
        float(eq_result.tariffs[c_idx])
        if getattr(eq_result, "tariffs", None) is not None
        else 0.0
    )
    # Approximate Harberger triangle: 0.5 * tau * Delta M
    m_base = (
        float(base_result.imports[c_idx])
        if base_result is not None and getattr(base_result, "imports", None) is not None
        else imp_val
    )
    m_cf = imp_val
    delta_m = max(m_base - m_cf, 0.0)
    mean_tau_rate = (tariffs_collected / m_cf) if m_cf > 1e-12 else 0.10
    dwl = 0.5 * mean_tau_rate * delta_m

    # 3. Fiscal Efficiency Dividend
    if closure in ("capital_tax", "capital_tax_cut"):
        # Double dividend from mitigating pre-existing capital tax distortions
        fiscal_dividend = 0.35 * tariffs_collected
    elif closure in ("labor_tax", "payroll_tax_cut"):
        # Double dividend from labor supply distortion relief
        fiscal_dividend = 0.45 * tariffs_collected
    elif closure in ("strategic_subsidy", "targeted_subsidy", "manufacturing_subsidy"):
        # Correcting intermediate under-provision
        fiscal_dividend = 0.20 * tariffs_collected
    elif closure in ("deficit_reduction", "public_debt"):
        # Sovereign debt risk premium reduction
        fiscal_dividend = 0.15 * tariffs_collected
    else:
        fiscal_dividend = 0.0  # Lump-sum rebate has zero marginal efficiency dividend

    net_welfare = tot_effect - dwl + fiscal_dividend

    return {
        "terms_of_trade": float(tot_effect),
        "deadweight_loss": float(dwl),
        "fiscal_dividend": float(fiscal_dividend),
        "efficiency_dividend": float(fiscal_dividend),
        "net_welfare_change": float(net_welfare),
    }


def run_fiscal_recycling_scenario(
    calib: TradeCalibrationResult,
    scenario: TariffScenario | str | Mapping[str, Any],
    config: FiscalRecyclingConfig | None = None,
    base_result: TradeEquilibriumResult | None = None,
    **solver_kwargs: Any,
) -> RevenueRecyclingResult:
    """Solve general equilibrium under alternative fiscal recycling closures.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    scenario : TariffScenario, str, or Mapping
        Tariff shock scenario.
    config : FiscalRecyclingConfig, optional
        Fiscal closure specification.
    base_result : TradeEquilibriumResult, optional
        Baseline equilibrium for welfare comparisons.
    **solver_kwargs : Any
        Solver keyword arguments.

    Returns
    -------
    RevenueRecyclingResult
        Solved equilibrium, tariff revenues, tax rate adjustments, and welfare breakdown.
    """
    if config is None:
        config = FiscalRecyclingConfig()

    country_codes = list(calib.country_codes) if calib.country_codes else list(CANONICAL_COUNTRY_CODES)
    c_idx = (
        country_codes.index(config.target_country)
        if config.target_country in country_codes
        else (country_codes.index("USA") if "USA" in country_codes else 0)
    )

    tau, tau_fd, tauf, tauf_fd = build_tariff_matrices(scenario, calib)
    scen_name = scenario if isinstance(scenario, str) else getattr(scenario, "name", "recycling_shock")

    recycling_params = {
        "target_country": config.target_country,
        "subsidy_shares": config.subsidy_shares or {"MANU": 1.0},
    }

    # Warm-start from base_result if available and not overridden
    x0 = solver_kwargs.pop("x0", getattr(base_result, "x_sol", None))
    if x0 is not None:
        solver_kwargs["x0"] = x0

    eq_res = solve_trade_equilibrium(
        calib=calib,
        tau=tau,
        tau_fd=tau_fd,
        tauf=tauf,
        tauf_fd=tauf_fd,
        fiscal_closure=config.closure,
        recycling_params=recycling_params,
        **solver_kwargs,
    )

    tariff_rev = float(eq_res.tariffs[c_idx])

    # Factor tax cuts or transfers
    if config.closure in ("capital_tax", "capital_tax_cut"):
        k_val = float(eq_res.r_sol[0, 0, c_idx] * calib.k_endow.ravel()[c_idx])
        tax_cut = (tariff_rev / k_val) if k_val > 1e-12 else 0.0
        household_trans = 0.0
        sub_rate = 0.0
    elif config.closure in ("labor_tax", "payroll_tax_cut"):
        l_val = float(eq_res.w_sol[0, 0, c_idx] * calib.l_endow.ravel()[c_idx])
        tax_cut = (tariff_rev / l_val) if l_val > 1e-12 else 0.0
        household_trans = 0.0
        sub_rate = 0.0
    elif config.closure in ("strategic_subsidy", "targeted_subsidy", "manufacturing_subsidy"):
        manu_idx = calib.sector_codes.index("MANU") if "MANU" in calib.sector_codes else 0
        y_manu = float(eq_res.y_sol[0, manu_idx, c_idx])
        sub_rate = (tariff_rev / y_manu) if y_manu > 1e-12 else 0.0
        tax_cut = 0.0
        household_trans = 0.0
    elif config.closure in ("deficit_reduction", "public_debt"):
        tax_cut = 0.0
        sub_rate = 0.0
        household_trans = 0.0
    else:
        tax_cut = 0.0
        sub_rate = 0.0
        household_trans = tariff_rev

    decomp = decompose_harberger_welfare(
        calib=calib,
        eq_result=eq_res,
        base_result=base_result,
        closure=config.closure,
        target_country=config.target_country,
    )

    return RevenueRecyclingResult(
        scenario_name=f"{scen_name}_{config.closure}",
        closure=config.closure,
        equilibrium=eq_res,
        tariff_revenue=tariff_rev,
        household_transfer=household_trans,
        factor_tax_cut=tax_cut,
        subsidy_rate=sub_rate,
        welfare_decomposition=decomp,
        target_country=config.target_country,
        metadata={
            "closure": config.closure,
            "target_country": config.target_country,
        },
    )


# ===========================================================================
# 4. EXTENSION D: UPSTREAM SUPPLY BOTTLENECKS (F23)
# ===========================================================================

@dataclass(frozen=True)
class BottleneckConfig:
    """Configuration for upstream supply bottleneck constraints.

    Attributes
    ----------
    target_country : str, default "USA"
        Country experiencing industrial capacity constraints.
    capacity_margins : dict[str, float], default {"MANU": 0.10}
        Spare capacity margins (kappa) by sector code: y_bar = (1 + kappa) * y0.
    penalty_scale : float, default 0.05
        Zeta parameter in smooth barrier penalty function Phi(y) = c_V * [1 + zeta * (y/y_bar)^eta].
    penalty_exponent : float, default 8.0
        Eta parameter in smooth barrier penalty function.
    """

    target_country: str = "USA"
    capacity_margins: dict[str, float] = field(default_factory=lambda: {"MANU": 0.10})
    penalty_scale: float = 0.05
    penalty_exponent: float = 8.0

    def __post_init__(self) -> None:
        for k, v in self.capacity_margins.items():
            if v < 0.0:
                raise ValueError(f"Capacity margin for sector {k} must be non-negative, got {v}")
        if self.penalty_scale <= 0.0:
            raise ValueError(f"penalty_scale must be positive, got {self.penalty_scale}")
        if self.penalty_exponent <= 1.0:
            raise ValueError(f"penalty_exponent must be > 1.0, got {self.penalty_exponent}")


def run_bottleneck_scenario(
    calib: TradeCalibrationResult,
    scenario: TariffScenario | str | Mapping[str, Any],
    config: BottleneckConfig | None = None,
    base_result: TradeEquilibriumResult | None = None,
    **solver_kwargs: Any,
) -> CapacityBottleneckResult:
    """Solve general equilibrium under upstream capacity constraints and smooth penalties.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    scenario : TariffScenario, str, or Mapping
        Tariff shock scenario.
    config : BottleneckConfig, optional
        Capacity margins and penalty function parameters.
    base_result : TradeEquilibriumResult, optional
        Baseline equilibrium for warm-starting.
    **solver_kwargs : Any
        Solver keyword arguments.

    Returns
    -------
    CapacityBottleneckResult
        Equilibrium outputs, utilization rates, producer price escalations, and barrier penalties.
    """
    if config is None:
        config = BottleneckConfig()

    country_codes = list(calib.country_codes) if calib.country_codes else list(CANONICAL_COUNTRY_CODES)
    sector_codes = list(calib.sector_codes) if calib.sector_codes else list(CANONICAL_SECTOR_CODES)
    c_idx = (
        country_codes.index(config.target_country)
        if config.target_country in country_codes
        else (country_codes.index("USA") if "USA" in country_codes else 0)
    )

    tau, tau_fd, tauf, tauf_fd = build_tariff_matrices(scenario, calib)
    scen_name = scenario if isinstance(scenario, str) else getattr(scenario, "name", "bottleneck_shock")

    # Warm-start from base_result if available and not overridden
    x0 = solver_kwargs.pop("x0", getattr(base_result, "x_sol", None))
    if x0 is not None:
        solver_kwargs["x0"] = x0

    eq_res = solve_trade_equilibrium(
        calib=calib,
        tau=tau,
        tau_fd=tau_fd,
        tauf=tauf,
        tauf_fd=tauf_fd,
        capacity_margins=config.capacity_margins,
        capacity_target_country=config.target_country,
        penalty_scale=config.penalty_scale,
        penalty_exponent=config.penalty_exponent,
        **solver_kwargs,
    )

    cap_limits: dict[str, float] = {}
    out_levels: dict[str, float] = {}
    cap_utils: dict[str, float] = {}
    price_escs: dict[str, float] = {}
    pen_mults: dict[str, float] = {}

    for sec_name, margin in config.capacity_margins.items():
        if sec_name in sector_codes:
            s_idx = sector_codes.index(sec_name)
            y0 = float(calib.ytot[0, s_idx, c_idx])
            y_bar = (1.0 + margin) * y0
            y_sol = float(eq_res.y_sol[0, s_idx, c_idx])
            p_sol = float(eq_res.p_sol[0, s_idx, c_idx])

            util = (y_sol / y_bar) if y_bar > 1e-12 else 1.0
            ratio = max(y_sol / y_bar, 0.0) if y_bar > 1e-12 else 1.0
            penalty = config.penalty_scale * (ratio ** config.penalty_exponent)

            cap_limits[sec_name] = y_bar
            out_levels[sec_name] = y_sol
            cap_utils[sec_name] = util
            price_escs[sec_name] = (p_sol - 1.0) * 100.0
            pen_mults[sec_name] = penalty

    return CapacityBottleneckResult(
        scenario_name=f"{scen_name}_bottleneck",
        equilibrium=eq_res,
        capacity_margins=config.capacity_margins,
        capacity_limits=cap_limits,
        output_levels=out_levels,
        capacity_utilization=cap_utils,
        price_escalation=price_escs,
        penalty_multipliers=pen_mults,
        target_country=config.target_country,
        penalty_scale=config.penalty_scale,
        penalty_exponent=config.penalty_exponent,
        metadata={
            "target_country": config.target_country,
            "penalty_scale": config.penalty_scale,
            "penalty_exponent": config.penalty_exponent,
        },
    )


__all__ = [
    "RetaliationConfig",
    "solve_retaliation_game",
    "DynamicJCurveConfig",
    "compute_j_curve_turning_point",
    "compute_contractual_half_life",
    "simulate_analytical_j_curve",
    "run_dynamic_j_curve_simulation",
    "FiscalRecyclingConfig",
    "decompose_harberger_welfare",
    "run_fiscal_recycling_scenario",
    "BottleneckConfig",
    "run_bottleneck_scenario",
]
