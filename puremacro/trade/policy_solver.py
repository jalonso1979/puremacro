"""Audited equilibrium solves for consistent-accounting policy experiments."""
from __future__ import annotations

from dataclasses import replace
from numbers import Integral
import warnings
from typing import Any

import numpy as np

from ._accounting import evaluate
from ._results import TradeCalibrationResult, TradeEquilibriumResult
from .solver import build_initial_guess, solve_trade_equilibrium, _resolve_tariffs
from .welfare import _checked_state

__all__ = ["PolicyEquilibriumError", "solve_policy_equilibrium"]


class PolicyEquilibriumError(RuntimeError):
    """All declared solver attempts failed; no policy payoff is available."""

    def __init__(self, message: str, attempts: list[dict[str, Any]]):
        super().__init__(message)
        self.attempts = attempts


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
    ta, tf, _, _ = _resolve_tariffs(calib, tau, tau_fd)
    initial = build_initial_guess(calib)
    warm = initial if x0 is None else np.asarray(x0, dtype=float).ravel().copy()
    if warm.shape != initial.shape or not np.isfinite(warm).all():
        raise ValueError("Warm start must be a finite equilibrium vector of the expected length")
    # Validate the model and requested tariffs before interpreting solver errors
    # as numerical failures. This also rejects invalid domestic tariff entries.
    evaluate(initial, calib, ta, tf, None, None, sigma=sigma)
    if base_result is not None:
        _checked_state(base_result, calib, 1e-8)
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
                "policy_solver_requested_tolerance": tol})
    raise PolicyEquilibriumError(
        "All policy equilibrium attempts failed; payoff and deviation coverage are unavailable", attempts)
