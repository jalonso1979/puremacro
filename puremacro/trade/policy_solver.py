"""Audited equilibrium solves for consistent-accounting policy experiments."""
from __future__ import annotations

from dataclasses import replace
from numbers import Integral
import warnings
from typing import Any

import numpy as np

from ._accounting import evaluate
from ._results import TradeCalibrationResult, TradeEquilibriumResult
from .solver import (build_initial_guess, solve_trade_equilibrium, _resolve_tariffs,
                     _hawkins_simon_certificate)
from .welfare import _checked_state

__all__ = ["PolicyEquilibriumError", "solve_policy_equilibrium"]


class PolicyEquilibriumError(RuntimeError):
    """All declared solver attempts failed; no policy payoff is available.

    ``attempts`` lists the solver attempts; ``viability`` records the
    Hawkins-Simon pre-screen (``None`` when it did not run). A schedule with a
    certified Hawkins-Simon violation raises this error before any solver runs,
    with an empty ``attempts`` list.
    """

    def __init__(self, message: str, attempts: list[dict[str, Any]],
                 viability: dict[str, Any] | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.viability = viability


_VIABLE_BELOW = 1.0 - 1e-6


def _viability_screen(calib: TradeCalibrationResult, ta: np.ndarray, sigma: float) -> dict[str, Any]:
    """Hawkins-Simon pre-screen of the intermediate schedule.

    With Leontief intermediates (``sigma == 0``) consistent accounting prices
    output as ``p = B_tau' p + v / (1 - t)`` with ``v > 0`` and
    ``B_tau = a * multiplier / (1 - t)``, using the model's exact ``1 - t``
    (no clipping). A positive price system exists if and only if
    rho(B_tau) < 1. The Collatz-Wielandt bounds give one of

    - ``"viable"``: upper bound < 1 - 1e-6;
    - ``"violated"``: lower bound >= 1, a certificate that no positive price
      system exists (the only status that stops the solver chain);
    - ``"near_critical"``: 1 - 1e-6 <= lower bound and upper bound < 1; a
      positive price system exists, with prices of order 1 / (1 - rho) times
      value added;
    - ``"unresolved"``: the bounds straddle 1 - 1e-6 or 1.

    For CES intermediates (``sigma > 0``) the CES unit cost is at most the
    Leontief cost, so a Leontief violation is not a certificate of
    non-existence; the screen is then ``"not_applicable"``.
    """
    if sigma != 0:
        return {"status": "not_applicable",
                "reason": "Hawkins-Simon bounds certify non-existence only for Leontief intermediates (sigma=0)"}
    rho, lower, upper, _ = _hawkins_simon_certificate(calib, ta, exact_tax=True)
    if lower < 1.0 <= upper:
        # The certificate refines only across 1 - 1e-6; refine across 1 as well.
        rho, lower, upper, _ = _hawkins_simon_certificate(calib, ta, max_iter=1000, exact_tax=True)
    if upper < _VIABLE_BELOW:
        status = "viable"
    elif lower >= 1.0:
        status = "violated"
    elif upper < 1.0:
        status = "near_critical"
    else:
        status = "unresolved"
    return {"status": status, "rho": rho, "cw_lower": lower, "cw_upper": upper,
            "threshold": _VIABLE_BELOW, "violation_threshold": 1.0,
            "production_tax": "exact 1 - t"}


def solve_policy_equilibrium(
    calib: TradeCalibrationResult,
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    *,
    x0: np.ndarray | None = None,
    base_result: TradeEquilibriumResult | None = None,
    sigma: float = 0.,
    tol: float = 1e-8,
    max_iter: int = 100,
    max_steps: int = 100,
    method: str = "auto",
    foreign_saving_units: str = "numeraire",
) -> TradeEquilibriumResult:
    """Try Newton, hybrid and full continuation without relaxing model or tolerance.

    ``method="auto"`` uses this sequence; an explicit method selects one attempt.
    Newton may use the supplied warm start. Hybrid resets to the calibrated
    initial state; continuation starts at the calibrated zero-tariff schedule.
    Failed iterates never seed a fallback. Accepted results must pass a fresh
    audit of the requested schedules, all physical equations and returned fields.
    Only consistent accounting, NumPy and lump-sum rebates are supported.

    Metadata records attempts, warnings, seeds, effective method and fallback use.
    ``PolicyEquilibriumError.attempts`` retains diagnostics when all attempts fail.
    Invalid model inputs raise before any fallback search. This is a numerical
    recovery mechanism, not a uniqueness or branch-selection theorem.

    Before any solver runs, the intermediate schedule is screened with the
    Hawkins-Simon Collatz-Wielandt bounds (Leontief intermediates, ``sigma=0``;
    exact production-tax wedge ``1 - t``). A certified violation (lower bound
    >= 1: no positive price system exists) raises
    :class:`PolicyEquilibriumError` immediately, without Newton or hybrid
    attempts. Every other status (``"viable"``, ``"near_critical"`` for bounds
    in ``[1 - 1e-6, 1)``, ``"unresolved"``; ``"not_applicable"`` for
    ``sigma > 0``) lets the chain run. The screen is recorded in
    ``metadata["policy_viability"]`` and ``PolicyEquilibriumError.viability``.

    ``foreign_saving_units`` is passed to :func:`solve_trade_equilibrium`
    (``"numeraire"``, the default, or ``"world_income"``).
    """
    if not isinstance(calib, TradeCalibrationResult):
        raise TypeError("calib must be a TradeCalibrationResult")
    if (not np.isfinite(tol) or tol <= 0 or not np.isfinite(sigma) or sigma < 0
            or not isinstance(max_iter, Integral) or isinstance(max_iter, bool) or max_iter < 1
            or not isinstance(max_steps, Integral) or isinstance(max_steps, bool) or max_steps < 1):
        raise ValueError("Require positive finite tolerance/iteration limits and nonnegative sigma")
    methods = ("newton", "hybr", "keller_pac") if method == "auto" else (method,)
    if any(m not in ("newton", "hybr", "keller_pac") for m in methods):
        raise ValueError("method must be auto, newton, hybr or keller_pac")
    if foreign_saving_units not in ("numeraire", "world_income"):
        raise ValueError("foreign_saving_units must be 'numeraire' or 'world_income'")
    ta, tf, _, _ = _resolve_tariffs(calib, tau, tau_fd)
    initial = build_initial_guess(calib)
    warm = initial if x0 is None else np.asarray(x0, dtype=float).ravel().copy()
    if warm.shape != initial.shape or not np.isfinite(warm).all():
        raise ValueError("Warm start must be a finite equilibrium vector of the expected length")
    # Validate the model and requested tariffs before interpreting solver errors
    # as numerical failures. This also rejects invalid domestic tariff entries.
    evaluate(initial, calib, ta, tf, None, None, sigma=sigma, foreign_saving_units=foreign_saving_units)
    if base_result is not None:
        _checked_state(base_result, calib, 1e-8)
    # Hawkins-Simon pre-screen: a certified violation has no positive price
    # system, so no solver attempt (and no fallback) is launched.
    viability = _viability_screen(calib, ta, sigma)
    if viability["status"] == "violated":
        raise PolicyEquilibriumError(
            "Tariff schedule violates the Hawkins-Simon viability condition: spectral bounds "
            f"[{viability['cw_lower']:.8g}, {viability['cw_upper']:.8g}] certify rho(B_tau) >= 1, "
            "so no positive price system exists; no solver was run", [], viability)
    attempts = []
    for selected in methods:
        seed = warm.copy() if selected == "newton" else initial.copy()
        row: dict[str, Any] = {"method": selected,
                              "seed": "warm" if selected == "newton" and x0 is not None else "calibrated",
                              "requested_tolerance": tol, "accepted": False}
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                result = solve_trade_equilibrium(
                    calib, tau=ta, tau_fd=tf, x0=seed, method=selected,
                    accounting="consistent", sigma=sigma, tol=tol,
                    max_iter=int(max_iter), max_steps=int(max_steps), base_result=base_result,
                    foreign_saving_units=foreign_saving_units,
                )
            row["solver_converged"] = bool(result.converged)
            row["solver_max_residual"] = (float(result.max_residual)
                if np.isfinite(result.max_residual) else None)
            if not result.converged:
                raise ValueError("Solver did not return a converged equilibrium")
            for name, expected in (("intermediate_tariff_multipliers", ta), ("final_tariff_multipliers", tf)):
                actual = result.metadata.get(name)
                if actual is None or np.shape(actual) != expected.shape or not np.array_equal(actual, expected):
                    raise ValueError("Returned tariff schedules differ from the requested policy")
            if result.metadata.get("sigma") != sigma:
                raise ValueError("Returned technology differs from the requested sigma")
            if result.metadata.get("foreign_saving_units", "numeraire") != foreign_saving_units:
                raise ValueError("Returned foreign-saving closure differs from the request")
            blocks, _ = _checked_state(result, calib, 1e-8)
            residual = np.r_[blocks["residuals"], blocks["physical_residuals"]]
            row["audited_max_residual"] = float(np.max(np.abs(residual)))
            if not np.isfinite(residual).all() or row["audited_max_residual"] > tol:
                raise ValueError("Independent physical/accounting audit exceeds the requested tolerance")
            row["accepted"] = True
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
            row["error"] = str(exc)
        row["warnings"] = sorted(set(str(w.message) for w in caught))
        attempts.append(row)
        if row["accepted"]:
            return replace(result, metadata={**result.metadata,
                "policy_solver_attempts": attempts, "policy_solver_method": selected,
                "policy_solver_fallback_used": len(attempts) > 1,
                "policy_solver_requested_tolerance": tol,
                "policy_viability": viability})
    raise PolicyEquilibriumError(
        "All policy equilibrium attempts failed; payoff and deviation coverage are unavailable",
        attempts, viability)
