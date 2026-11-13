"""Inspectable audits of solved results; no solves or network work by default."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from ._audit_report import AuditCheck, AuditReport, DerivativeCheck, RefinementCheck, RefinementLadder, ResidualCheck

_CATEGORIES = ("solver", "equations", "distribution", "derivatives", "refinement")
_TOLERANCES = {"equations": 1e-8, "euler": 1e-2, "stationarity": 1e-10,
               "mass": 1e-12, "negative_mass": 1e-14, "boundary_mass": 1e-3,
               "derivatives": 1e-5}


class _Collector:
    def __init__(self, checks, tolerances):
        self.checks = set(checks)
        self.tol = {**_TOLERANCES, **dict(tolerances or {})}
        self.rows: list[AuditCheck] = []
        self.maps: dict[str, np.ndarray] = {}
        self.metadata: dict[str, Any] = {}

    def add(self, category, check, value=None, tolerance=None, units="dimensionless",
            source="not_evaluated", status=None, location=None, **details):
        if status is None:
            status = ("passed" if value is not None and np.isfinite(value)
                      and (bool(value) if tolerance is None else value <= tolerance) else "failed")
        self.rows.append(AuditCheck(category, check, value, tolerance, units, status, source,
                                    location, deepcopy(details)))

    def residual(self, category, check, values, tolerance, *, units="equation units",
                 source="recomputed_equations", weights=None, coordinates=None, **details):
        arr = np.array(values, dtype=float, copy=True)
        arr.setflags(write=False)
        self.maps[check] = arr
        absolute = np.abs(arr)
        worst = int(np.argmax(np.where(np.isfinite(absolute), absolute, np.inf))) if arr.size else 0
        loc = tuple(int(v) for v in np.unravel_index(worst, arr.shape)) if arr.size else None
        maximum = float(np.max(absolute)) if arr.size else float("nan")
        if coordinates is not None and loc is not None:
            details["worst_coordinates"] = {name: float(grid[loc[i]])
                                            for i, (name, grid) in enumerate(coordinates)}
        self.add(category, check, maximum, tolerance, units, source, location=loc, **details)
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            valid = (arr.size > 0 and w.shape == arr.shape and np.all(np.isfinite(w)) and np.all(w >= 0)
                     and abs(float(w.sum()) - 1) <= self.tol["mass"]
                     and np.all(np.isfinite(arr)))
            for q in (50, 95, 99):
                value = None
                if valid:
                    positive = w.ravel() > 0
                    errors, mass = absolute.ravel()[positive], w.ravel()[positive]
                    order = np.argsort(errors, kind="stable")
                    index = min(np.searchsorted(np.cumsum(mass[order]), q / 100 * mass.sum()),
                                len(order) - 1)
                    value = float(errors[order[index]])
                self.add(category, f"{check}_weighted_p{q}", value, tolerance, units, source,
                         status=None if valid else "not_checked",
                         reason="stationary-distribution weighted nearest-rank quantile" if valid
                         else "requires a finite nonnegative unit-mass distribution and residual map")


def _derivative(collector, experiment):
    x = np.array(experiment.point, dtype=float, copy=True)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or not np.isfinite(experiment.step) or experiment.step <= 0:
        raise ValueError("DerivativeCheck requires a finite one-dimensional point and positive step")
    f0 = np.asarray(experiment.function(x.copy()), dtype=float)
    jac = np.asarray(experiment.jacobian, dtype=float)
    if jac.shape != f0.shape + x.shape:
        raise ValueError(f"derivative shape {jac.shape} does not match {f0.shape + x.shape}")
    numeric = np.empty_like(jac)
    for j in range(len(x)):
        h = experiment.step * max(1., abs(x[j]))
        values = []
        for multiplier in (-2., -1., 1., 2.):
            point = x.copy()
            point[j] += multiplier * h
            value = np.asarray(experiment.function(point), dtype=float)
            if value.shape != f0.shape:
                raise ValueError("derivative callback output shape changed")
            values.append(value)
        numeric[..., j] = (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * h)
    errors = (jac - numeric) / np.maximum(1., np.abs(numeric))
    collector.residual("derivatives", experiment.name, errors, experiment.tolerance,
                       units=experiment.units, source=experiment.evidence_source,
                       step=experiment.step, stencil="central five-point", analytic=jac, numeric=numeric)


def _audit_trade_equilibrium(out: _Collector, result: Any, model: Any = None) -> None:
    diag = getattr(result, "solver_diagnostics", None)
    if "solver" in out.checks:
        conv = bool(result.converged)
        status = diag.overall_status if diag else ("PASS" if conv else "FAILED")
        out.add("solver", "solver_converged", conv, source="trade_equilibrium_record")
        out.add("solver", "solver_overall_status", status == "PASS", source="trade_solver_diagnostics",
                status="passed" if status == "PASS" else "failed", overall_status=status)
    if "equations" in out.checks:
        acc = diag.accuracy if diag else None
        max_res = float(acc.max_residual if acc else result.max_residual)
        tol = float(acc.tolerance if acc else 1e-4)
        is_acc = bool(acc.is_accurate if acc else (max_res <= tol))
        out.add("equations", "excess_demand_residual", max_res, tol, units="excess demand",
                source="trade_solver_diagnostics", status="passed" if is_acc else "failed",
                is_accurate=is_acc)
    if "distribution" in out.checks:
        supp = diag.support if diag else None
        is_supp = bool(supp.is_supported if supp else True)
        regime = str(supp.regime if supp else "interior")
        warnings = list(supp.warnings if supp else [])
        out.add("distribution", "price_and_factor_support", is_supp, source="trade_solver_diagnostics",
                status="passed" if is_supp else "failed", regime=regime, warnings=warnings)


def _audit_sovereign_default(out: _Collector, result: Any, model: Any = None) -> None:
    diag = getattr(result, "solver_diagnostics", None)
    if "solver" in out.checks:
        conv = bool(result.converged)
        status = diag.overall_status if diag else ("PASS" if conv else "FAILED")
        out.add("solver", "solver_converged", conv, source="sovereign_solver_record")
        out.add("solver", "solver_overall_status", status == "PASS", source="sovereign_solver_diagnostics",
                status="passed" if status == "PASS" else "failed", overall_status=status)
    if "equations" in out.checks:
        acc = diag.accuracy if diag else None
        b_res = float(result.bellman_residual)
        tol = float(acc.tolerance if acc else 1e-3)
        is_acc = bool(acc.is_accurate if acc else (b_res <= tol))
        out.add("equations", "bellman_residual", b_res, tol, units="value function units",
                source="sovereign_solver_diagnostics", status="passed" if is_acc else "failed",
                is_accurate=is_acc)
    if "distribution" in out.checks:
        supp = diag.support if diag else None
        is_supp = bool(supp.is_supported if supp else True)
        regime = str(supp.regime if supp else "interior")
        p_bmax = float(supp.boundary_metrics.get("p_bmax", 0.0)) if supp else 0.0
        out.add("distribution", "debt_boundary_support", p_bmax, 0.01, units="probability",
                source="sovereign_solver_diagnostics", status="passed" if is_supp else "failed",
                comparison="value <= tolerance", regime=regime)


def _audit_var_estimate(out: _Collector, result: Any, model: Any = None) -> None:
    diag = getattr(result, "solver_diagnostics", None)
    if "solver" in out.checks:
        conv = bool(diag.convergence.converged if diag else True)
        status = diag.overall_status if diag else "PASS"
        out.add("solver", "solver_converged", conv, source="var_estimate_record")
        out.add("solver", "solver_overall_status", status == "PASS", source="var_solver_diagnostics",
                status="passed" if status == "PASS" else "failed", overall_status=status)
    if "equations" in out.checks:
        norm_res = float(diag.accuracy.residual_norm if diag else np.linalg.norm(result.resid))
        out.add("equations", "ols_residual_norm", norm_res, units="residual norm",
                source="var_solver_diagnostics", status="passed" if np.isfinite(norm_res) else "failed")
    if "distribution" in out.checks:
        rho = getattr(result, "spectral_radius", 0.0)
        supp = diag.support if diag else None
        is_supp = bool(supp.is_supported if supp else (rho < 1.0))
        regime = str(supp.regime if supp else ("stationary" if is_supp else "non_stationary_explosive"))
        out.add("distribution", "stationarity_support", rho, 1.0, units="spectral radius",
                source="var_solver_diagnostics", status="passed" if is_supp else "failed",
                comparison="value < tolerance", regime=regime)


def _audit_one_asset_hank(out: _Collector, result: Any, model: Any = None) -> None:
    diag = getattr(result, "solver_diagnostics", None)
    if "solver" in out.checks:
        conv = bool(result.converged)
        status = diag.overall_status if diag else ("PASS" if conv else "FAILED")
        out.add("solver", "solver_converged", conv, source="one_asset_hank_record")
        out.add("solver", "solver_overall_status", status == "PASS", source="one_asset_hank_diagnostics",
                status="passed" if status == "PASS" else "failed", overall_status=status)
    if "equations" in out.checks:
        mkt_res = float(result.diagnostics.get("market_residual_max_abs", 0.0))
        euler_res = float(result.diagnostics.get("euler_kkt_max", 0.0))
        acc = diag.accuracy if diag else None
        is_acc = bool(acc.is_accurate if acc else (euler_res <= 0.01 and mkt_res <= 0.01))
        out.add("equations", "market_clearing_residual", mkt_res, 0.01, units="market residual",
                source="one_asset_hank_diagnostics", status="passed" if mkt_res <= 0.01 else "failed")
        out.add("equations", "euler_kkt_residual", euler_res, 0.01, units="current marginal utility",
                source="one_asset_hank_diagnostics", status="passed" if euler_res <= 0.01 else "failed",
                is_accurate=is_acc)
    if "distribution" in out.checks:
        supp = diag.support if diag else None
        is_supp = bool(supp.is_supported if supp else True)
        regime = str(supp.regime if supp else "interior")
        min_mass = float(result.diagnostics.get("minimum_mass", 0.0))
        mass_err = float(result.diagnostics.get("mass_error", 0.0))
        out.add("distribution", "distribution_support", is_supp, source="one_asset_hank_diagnostics",
                status="passed" if is_supp else "failed", regime=regime,
                minimum_mass=min_mass, mass_error=mass_err)


def _dp_metrics(result):
    from puremacro.dp._results import DPSolution
    if not isinstance(result, DPSolution) or result.distribution is None:
        raise ValueError("default refinement metrics require a DP result with a distribution")
    if getattr(result.raw, "converged", True) is False:
        raise ValueError("refinement solver did not converge")
    dist = np.asarray(result.distribution)
    if not np.all(np.isfinite(dist)) or np.any(dist < 0) or abs(dist.sum() - 1) > 1e-12:
        raise ValueError("refinement result distribution must be finite, nonnegative and unit mass")
    metrics = {"consumption": float(np.sum(dist * result.raw.c))}
    for name in result._c.state_names:
        metrics[f"mean_{name}"] = result.mean(name)
    report = audit_result(result)
    for row in report.records:
        if row.check.startswith("euler_kkt_") or row.check.startswith("mass_at_"):
            if row.value is not None:
                metrics[row.check] = float(row.value)
    return metrics


def audit_result(result: Any, *, model: Any = None,
                 checks: Sequence[str] = ("solver", "equations", "distribution"),
                 tolerances: Mapping[str, float] | None = None,
                 residuals: Sequence[ResidualCheck] = (),
                 derivatives: Sequence[DerivativeCheck] = (),
                 refinements: Sequence[RefinementCheck] = (),
                 refinement_ladders: Sequence[RefinementLadder] = ()) -> AuditReport:
    """Audit supplied household, DSGE, nonlinear path, or dynamic trade results.

    Categories are ``solver``, ``equations``, ``distribution``, ``derivatives``
    and ``refinement``. Unrequested categories remain visible as
    ``not_checked``. Unknown result types get ``unsupported`` records.
    DP carries its model context; a bare Klein solution needs ``model=``.
    Trade certificates are preserved as recorded evidence; with a
    ``DynamicEconomy`` context the steady-state certificate is also rebuilt.

    Default work evaluates equations and pushes a distribution once; it never
    re-solves, differentiates, refines, normalizes a distribution, or accesses
    the network. Expensive callbacks require the corresponding category.
    Inputs are not mutated. User callbacks are responsible for the same
    no-mutation/no-network contract. Failed callbacks produce failed checks.
    Perfect-foresight paths require retained solve-time context or explicit
    ``model=PathAuditContext``. Nonlinear HANK household checks require a solve
    with ``retain_audit_paths=True``; missing paths are never regenerated.
    Editable HANK closures re-evaluate aggregate equations. Their nonlinear
    household consistency compares retained outputs only while household
    input paths are unchanged; it never runs another household solve.
    ``refinement_ladders`` run only with ``refinement`` selected and record
    adjacent-level comparisons, declared budgets and the stopping reason.

    Euler tolerance defaults to 0.01 (normalized marginal-utility units),
    independent of the policy iteration tolerance. This is a screening target,
    not a calibrated guarantee. Set ``tolerances={"euler": ...}`` for an
    application. Upper-bound mass is screened at 0.001, stationarity at 1e-10,
    total mass at 1e-12 and negative mass at 1e-14.
    """
    if isinstance(checks, str) or set(checks) - set(_CATEGORIES):
        raise ValueError(f"checks must be a sequence drawn from {_CATEGORIES}")
    unknown = set(tolerances or {}) - set(_TOLERANCES)
    if unknown or any(not np.isfinite(v) or v < 0 for v in (tolerances or {}).values()):
        raise ValueError(f"tolerances must be nonnegative finite values for {sorted(_TOLERANCES)}")
    # Validate even unrequested experiments so a later category selection
    # cannot accidentally turn an infinite threshold into a passing claim.
    thresholds = [r.tolerance for r in residuals] + [d.tolerance for d in derivatives]
    for threshold in thresholds:
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("experiment tolerance must be finite and nonnegative")
    for refinement in refinements:
        if any(not np.isfinite(v) or v < 0 for v in (refinement.atol, refinement.rtol)):
            raise ValueError("refinement atol and rtol must be finite and nonnegative")
    from ._audit_refinement import audit_ladder, validate_ladders
    validate_ladders(refinement_ladders)
    out = _Collector(checks, tolerances)
    from puremacro.dp._results import DPSolution
    from puremacro.dsge.build import LinearModel
    from puremacro.dsge.klein import KleinSolution
    from puremacro.dsge._results import HANKResult
    from puremacro.dsge.perfect_foresight import PerfectForesightResult
    from puremacro.models.hank_sequence_space import NonlinearHANKResult
    from puremacro.trade.dynamic._results import (
        DynamicSteadyStateResult, DynamicTransitionResult, HorizonComparisonResult,
    )
    from puremacro.trade._results import TradeEquilibriumResult
    from puremacro.sovereign._results import SovereignDefaultResult
    from puremacro.var._results import VarEstimateResult
    from puremacro.models.one_asset_hank import OneAssetHANKResult

    if isinstance(result, DPSolution):
        from ._audit_dp import audit_dp
        audit_dp(out, result, model)
    elif isinstance(result, (LinearModel, KleinSolution)):
        from ._audit_structural import audit_dsge
        audit_dsge(out, result, model)
    elif isinstance(result, PerfectForesightResult):
        from ._audit_paths import audit_perfect_foresight
        audit_perfect_foresight(out, result, model)
    elif isinstance(result, NonlinearHANKResult):
        from ._audit_paths import audit_nonlinear_hank
        audit_nonlinear_hank(out, result, model)
    elif isinstance(result, HANKResult):
        from ._audit_paths import audit_hank_model
        audit_hank_model(out, result, model)
    elif isinstance(result, (DynamicSteadyStateResult, DynamicTransitionResult, HorizonComparisonResult)):
        from ._audit_structural import audit_trade
        audit_trade(out, result, model)
    elif isinstance(result, TradeEquilibriumResult):
        _audit_trade_equilibrium(out, result, model)
    elif isinstance(result, SovereignDefaultResult):
        _audit_sovereign_default(out, result, model)
    elif isinstance(result, VarEstimateResult):
        _audit_var_estimate(out, result, model)
    elif isinstance(result, OneAssetHANKResult):
        _audit_one_asset_hank(out, result, model)
    for experiment in residuals:
        if "equations" in out.checks:
            try:
                out.residual("equations", experiment.name, experiment.evaluate(result, model),
                             experiment.tolerance, units=experiment.units, source=experiment.evidence_source)
            except Exception as exc:
                out.add("equations", experiment.name, status="failed", source=experiment.evidence_source,
                        error=f"{type(exc).__name__}: {exc}")
    if "derivatives" in out.checks:
        for derivative in derivatives:
            try:
                _derivative(out, derivative)
            except Exception as exc:
                out.add("derivatives", derivative.name, status="failed", source=derivative.evidence_source,
                        error=f"{type(exc).__name__}: {exc}")
    if "refinement" in out.checks:
        for ladder in refinement_ladders:
            audit_ladder(out, result, ladder)
        for refinement in refinements:
            try:
                metrics = refinement.metrics or _dp_metrics
                baseline = dict(metrics(result))
                refined = dict(metrics(refinement.solve()))
                if not baseline or baseline.keys() != refined.keys():
                    raise ValueError("refinement metrics must have identical nonempty keys")
                for name in sorted(baseline):
                    before, after = float(baseline[name]), float(refined[name])
                    tolerance = refinement.atol + refinement.rtol * abs(after)
                    default_unit = ("current marginal utility" if name.startswith("euler_kkt_") else
                                    "probability" if name.startswith("mass_at_") else
                                    "goods" if name == "consumption" else
                                    "asset units" if name.startswith("mean_") else "caller metric units")
                    out.add("refinement", f"{refinement.name}:{name}", abs(after - before), tolerance,
                            refinement.units.get(name, default_unit), refinement.evidence_source,
                            baseline=before, refined=after, rtol=refinement.rtol, atol=refinement.atol,
                            relative_change=(abs(after - before) / abs(after) if abs(after) > refinement.atol
                                             else None))
            except Exception as exc:
                out.add("refinement", refinement.name, status="failed", source=refinement.evidence_source,
                        error=f"{type(exc).__name__}: {exc}")
    for category in _CATEGORIES:
        if not any(row.category == category for row in out.rows):
            status = "not_checked" if category not in out.checks or category in ("derivatives", "refinement") else "unsupported"
            out.add(category, category, status=status,
                    reason="not requested or no explicit experiment supplied" if status == "not_checked"
                    else f"no {category} adapter for {type(result).__name__}")
    out.metadata.update({"result_type": f"{type(result).__module__}.{type(result).__name__}",
                         "requested_checks": [c for c in _CATEGORIES if c in out.checks],
                         "tolerances": out.tol,
                         "scope": "numerical evidence for supplied result; no external validation claim"})
    return AuditReport(tuple(out.rows), out.maps, deepcopy(out.metadata))
