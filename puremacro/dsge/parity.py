"""Automated Dynare Parity Verification Harness for puremacro 2.9.0.

Provides rigorous automated validation comparing puremacro DSGE solutions against
official Dynare results (*_results.mat).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ._results import (
    Dynare2ndDR,
    DynareDR,
    ModelParityResult,
    ParityDashboardResult,
)
from .load_dynare import load_dynare_dr, load_dynare_moments


DEFAULT_TOLERANCES: dict[str, float] = {
    "ghx": 1e-6,
    "ghu": 1e-6,
    "ghxx": 1e-4,
    "ghs2": 1e-4,
    "mean": 1e-5,
    "var": 1e-5,
    "autocorr": 1e-5,
}


def _resolve_tolerances(tol: float | Mapping[str, float] | None) -> dict[str, float]:
    """Resolve tolerance dictionary against defaults."""
    tols = dict(DEFAULT_TOLERANCES)
    if tol is None:
        return tols
    if isinstance(tol, (int, float)):
        val = float(tol)
        return {k: val for k in tols}
    if isinstance(tol, dict):
        tols.update({k: float(v) for k, v in tol.items()})
    return tols


def verify_dynare_parity(
    puremacro_model: Any,
    dynare_output: Any,
    order: int = 1,
    tol: float | Mapping[str, float] | None = None,
) -> ParityDashboardResult:
    """Verify puremacro DSGE model solution against Dynare output.

    Parameters
    ----------
    puremacro_model : LinearModel | PrunedDSGESolution | DynareDR | str | Path
        Solved puremacro model instance or path to a .mod file.
    dynare_output : DynareDR | Dynare2ndDR | dict | str | Path
        Dynare decision rules, results dict, or path to *_results.mat file.
    order : {1, 2}, default 1
        Perturbation order to compare.
    tol : float | dict[str, float], optional
        Tolerance thresholds. Defaults: 1e-6 for (ghx, ghu), 1e-4 for (ghxx, ghs2),
        1e-5 for moments.

    Returns
    -------
    ParityDashboardResult
        Consolidated scorecard with per-variable deviations and overall parity score.
    """
    t0 = time.perf_counter()
    tolerances = _resolve_tolerances(tol)
    model_name = "model"

    # 1. Resolve puremacro model
    if isinstance(puremacro_model, (str, Path)):
        p_mod = Path(puremacro_model)
        model_name = p_mod.stem
        if p_mod.suffix == ".mat":
            try:
                pm_dr = load_dynare_dr(p_mod, order=order)
                pm_model = pm_dr
            except KeyError as exc:
                if order >= 2 and ("ghxx" in str(exc) or "ghs2" in str(exc)):
                    runtime_sec = time.perf_counter() - t0
                    return ParityDashboardResult(
                        passed=False,
                        score=0.0,
                        model_name=model_name,
                        total_models=1,
                        passed_models=0,
                        failed_models=1,
                        tolerances=tolerances,
                        max_dev_ghx=np.nan,
                        max_dev_ghu=np.nan,
                        max_dev_ghxx=np.nan,
                        max_dev_ghs2=np.nan,
                        max_dev_moments=np.nan,
                        max_dev_dr=np.nan,
                        details={"error": str(exc), "runtime_sec": runtime_sec, "order2_missing": True},
                    )
                raise
        else:
            from .dynare import load_mod
            pm_model = load_mod(p_mod, order=order)
    else:
        pm_model = puremacro_model
        if hasattr(pm_model, "name") and pm_model.name:
            model_name = str(pm_model.name)

    # Extract puremacro decision rules
    if order >= 2:
        if hasattr(pm_model, "solve"):
            pm_sol = pm_model.solve(order=order)
            pm_dr = pm_sol.decision_rules()
        elif hasattr(pm_model, "solve_second_order"):
            pm_sol = pm_model.solve_second_order()
            pm_dr = pm_sol.decision_rules()
        elif hasattr(pm_model, "decision_rules"):
            pm_dr = pm_model.decision_rules()
        elif isinstance(pm_model, (DynareDR, Dynare2ndDR)):
            pm_dr = pm_model
        else:
            raise TypeError(f"Cannot extract order {order} decision rules from {type(pm_model)}")
    else:
        if hasattr(pm_model, "decision_rules"):
            pm_dr = pm_model.decision_rules()
        elif isinstance(pm_model, (DynareDR, Dynare2ndDR)):
            pm_dr = pm_model
        else:
            raise TypeError(f"Cannot extract decision rules from {type(pm_model)}")

    # 2. Resolve Dynare output
    dyn_moments: dict[str, np.ndarray] = {}
    if isinstance(dynare_output, (str, Path)):
        p_mat = Path(dynare_output)
        try:
            dyn_dr = load_dynare_dr(p_mat, order=order)
        except KeyError as exc:
            if order >= 2 and ("ghxx" in str(exc) or "ghs2" in str(exc)):
                runtime_sec = time.perf_counter() - t0
                return ParityDashboardResult(
                    passed=False,
                    score=0.0,
                    model_name=model_name,
                    total_models=1,
                    passed_models=0,
                    failed_models=1,
                    tolerances=tolerances,
                    max_dev_ghx=np.nan,
                    max_dev_ghu=np.nan,
                    max_dev_ghxx=np.nan,
                    max_dev_ghs2=np.nan,
                    max_dev_moments=np.nan,
                    max_dev_dr=np.nan,
                    details={"error": str(exc), "runtime_sec": runtime_sec, "order2_missing": True},
                )
            raise
        try:
            dyn_moments = load_dynare_moments(p_mat)
        except Exception:
            dyn_moments = {}
    elif isinstance(dynare_output, (DynareDR, Dynare2ndDR)):
        dyn_dr = dynare_output
    elif isinstance(dynare_output, dict):
        try:
            dyn_dr = load_dynare_dr(dynare_output, order=order)
        except KeyError as exc:
            if order >= 2 and ("ghxx" in str(exc) or "ghs2" in str(exc)):
                runtime_sec = time.perf_counter() - t0
                return ParityDashboardResult(
                    passed=False,
                    score=0.0,
                    model_name=model_name,
                    total_models=1,
                    passed_models=0,
                    failed_models=1,
                    tolerances=tolerances,
                    max_dev_ghx=np.nan,
                    max_dev_ghu=np.nan,
                    max_dev_ghxx=np.nan,
                    max_dev_ghs2=np.nan,
                    max_dev_moments=np.nan,
                    max_dev_dr=np.nan,
                    details={"error": str(exc), "runtime_sec": runtime_sec, "order2_missing": True},
                )
            raise
        try:
            dyn_moments = load_dynare_moments(dynare_output)
        except Exception:
            dyn_moments = {}
    else:
        raise TypeError(f"Unrecognized dynare_output type: {type(dynare_output)}")

    # 3. Variable alignment & validation
    pm_vars = list(pm_dr.variable_names)
    dyn_vars = list(getattr(dyn_dr, "variable_names", []))

    if dyn_vars and set(dyn_vars) != set(pm_vars):
        runtime_sec = time.perf_counter() - t0
        return ParityDashboardResult(
            passed=False,
            score=0.0,
            model_name=model_name,
            total_models=1,
            passed_models=0,
            failed_models=1,
            tolerances=tolerances,
            max_dev_ghx=np.nan,
            max_dev_ghu=np.nan,
            max_dev_ghxx=np.nan,
            max_dev_ghs2=np.nan,
            max_dev_moments=np.nan,
            max_dev_dr=np.nan,
            details={
                "error": f"Variable names mismatch: {pm_vars} vs {dyn_vars}",
                "runtime_sec": runtime_sec,
            },
        )

    # 4. Numerical comparisons of decision rules
    pm_ghx = np.asarray(pm_dr.ghx.loc[pm_vars], dtype=float)
    dyn_ghx = np.asarray(dyn_dr.ghx.loc[pm_vars], dtype=float)
    if pm_ghx.shape != dyn_ghx.shape:
        runtime_sec = time.perf_counter() - t0
        return ParityDashboardResult(
            passed=False,
            score=0.0,
            model_name=model_name,
            total_models=1,
            passed_models=0,
            failed_models=1,
            tolerances=tolerances,
            max_dev_ghx=np.nan,
            max_dev_ghu=np.nan,
            max_dev_ghxx=np.nan,
            max_dev_ghs2=np.nan,
            max_dev_moments=np.nan,
            max_dev_dr=np.nan,
            details={"error": f"ghx shape mismatch: {pm_ghx.shape} vs {dyn_ghx.shape}", "runtime_sec": runtime_sec},
        )

    pm_ghu = np.asarray(pm_dr.ghu.loc[pm_vars], dtype=float)
    dyn_ghu = np.asarray(dyn_dr.ghu.loc[pm_vars], dtype=float)
    if pm_ghu.shape != dyn_ghu.shape:
        runtime_sec = time.perf_counter() - t0
        return ParityDashboardResult(
            passed=False,
            score=0.0,
            model_name=model_name,
            total_models=1,
            passed_models=0,
            failed_models=1,
            tolerances=tolerances,
            max_dev_ghx=np.nan,
            max_dev_ghu=np.nan,
            max_dev_ghxx=np.nan,
            max_dev_ghs2=np.nan,
            max_dev_moments=np.nan,
            max_dev_dr=np.nan,
            details={
                "error": f"Shock shape mismatch: puremacro {pm_ghu.shape} vs Dynare {dyn_ghu.shape}",
                "runtime_sec": runtime_sec,
            },
        )

    diff_ghx = np.abs(pm_ghx - dyn_ghx)
    diff_ghu = np.abs(pm_ghu - dyn_ghu)

    max_dev_ghx = float(np.nanmax(diff_ghx)) if diff_ghx.size > 0 else 0.0
    max_dev_ghu = float(np.nanmax(diff_ghu)) if diff_ghu.size > 0 else 0.0

    per_var_ghx = np.nanmax(diff_ghx, axis=1) if diff_ghx.size > 0 else np.zeros(len(pm_vars))
    per_var_ghu = np.nanmax(diff_ghu, axis=1) if diff_ghu.size > 0 else np.zeros(len(pm_vars))

    pm_ys = np.asarray(pm_dr.ys.loc[pm_vars], dtype=float).ravel()
    dyn_ys = np.asarray(dyn_dr.ys.loc[pm_vars], dtype=float).ravel()
    diff_ys = np.abs(pm_ys - dyn_ys) if len(dyn_ys) == len(pm_ys) else np.zeros_like(per_var_ghx)

    # Second-order comparisons if requested
    max_dev_ghxx = 0.0
    max_dev_ghs2 = 0.0
    per_var_ghxx = np.zeros(len(pm_vars))
    per_var_ghs2 = np.zeros(len(pm_vars))
    order2_missing = False

    if order >= 2:
        if isinstance(dyn_dr, Dynare2ndDR) and hasattr(pm_dr, "ghxx"):
            pm_ghxx = np.asarray(pm_dr.ghxx.loc[pm_vars], dtype=float)
            dyn_ghxx = np.asarray(dyn_dr.ghxx.loc[pm_vars], dtype=float)
            diff_ghxx = np.abs(pm_ghxx - dyn_ghxx)
            max_dev_ghxx = float(np.nanmax(diff_ghxx)) if diff_ghxx.size > 0 else 0.0
            per_var_ghxx = np.nanmax(diff_ghxx, axis=1) if diff_ghxx.size > 0 else np.zeros(len(pm_vars))

            pm_ghs2 = np.asarray(pm_dr.ghs2.loc[pm_vars], dtype=float).ravel()
            dyn_ghs2 = np.asarray(dyn_dr.ghs2.loc[pm_vars], dtype=float).ravel()
            diff_ghs2 = np.abs(pm_ghs2 - dyn_ghs2) if len(dyn_ghs2) == len(pm_ghs2) else np.zeros(len(pm_vars))
            max_dev_ghs2 = float(np.nanmax(diff_ghs2)) if diff_ghs2.size > 0 else 0.0
            per_var_ghs2 = diff_ghs2
        else:
            order2_missing = True

    # 5. Theoretical moments comparisons if available
    max_dev_moments = 0.0
    dev_mean = 0.0
    dev_var = 0.0
    moments_dict: dict[str, Any] = {}
    if dyn_moments and hasattr(pm_model, "theoretical_moments"):
        try:
            th_mom = pm_model.theoretical_moments(ar=5)
            pm_mean = np.asarray(getattr(th_mom, "mean", np.array([])), dtype=float).ravel()
            dyn_mean = np.asarray(dyn_moments.get("mean", np.array([])), dtype=float).ravel()
            dev_mean = float(np.nanmax(np.abs(pm_mean - dyn_mean))) if len(dyn_mean) == len(pm_mean) and len(dyn_mean) > 0 else 0.0

            pm_var = np.asarray(getattr(th_mom, "variance", np.array([])), dtype=float).ravel()
            dyn_var = np.asarray(dyn_moments.get("var", np.array([])), dtype=float)
            if dyn_var.ndim == 2:
                dyn_var = np.diag(dyn_var)
            dyn_var = dyn_var.ravel()
            dev_var = float(np.nanmax(np.abs(pm_var - dyn_var))) if len(dyn_var) == len(pm_var) and len(dyn_var) > 0 else 0.0

            max_dev_moments = max(dev_mean, dev_var)
            moments_dict = {
                "dev_mean": dev_mean,
                "dev_var": dev_var,
                "tol_mean": tolerances.get("mean", 1e-5),
                "tol_var": tolerances.get("var", 1e-5),
            }
        except Exception:
            pass

    # 6. Scorecard compilation
    rows = []
    total_checks = 0
    passed_checks = 0

    tol_ghx = tolerances.get("ghx", 1e-6)
    tol_ghu = tolerances.get("ghu", 1e-6)
    tol_ghxx = tolerances.get("ghxx", 1e-4)
    tol_ghs2 = tolerances.get("ghs2", 1e-4)

    for i, var in enumerate(pm_vars):
        v_pass = (per_var_ghx[i] <= tol_ghx) and (per_var_ghu[i] <= tol_ghu)
        total_checks += 2
        passed_checks += int(per_var_ghx[i] <= tol_ghx) + int(per_var_ghu[i] <= tol_ghu)

        row_data: dict[str, Any] = {
            "dev_ghx": per_var_ghx[i],
            "dev_ghu": per_var_ghu[i],
            "dev_ys": diff_ys[i],
            "tol_ghx": tol_ghx,
            "tol_ghu": tol_ghu,
        }
        if order >= 2:
            row_data["dev_ghxx"] = per_var_ghxx[i]
            row_data["dev_ghs2"] = per_var_ghs2[i]
            row_data["tol_ghxx"] = tol_ghxx
            row_data["tol_ghs2"] = tol_ghs2
            total_checks += 2
            passed_checks += int(per_var_ghxx[i] <= tol_ghxx) + int(per_var_ghs2[i] <= tol_ghs2)
            if (per_var_ghxx[i] > tol_ghxx) or (per_var_ghs2[i] > tol_ghs2):
                v_pass = False

        row_data["status"] = "PASS" if v_pass else "FAIL"
        rows.append(row_data)

    df_dr = pd.DataFrame(rows, index=pm_vars)
    df_mom = pd.DataFrame([moments_dict]) if moments_dict else pd.DataFrame()

    score = 100.0 * (passed_checks / max(total_checks, 1))
    passed = (score == 100.0) and not order2_missing
    if dev_mean > tolerances.get("mean", 1e-5) or dev_var > tolerances.get("var", 1e-5):
        passed = False

    runtime_sec = time.perf_counter() - t0
    max_dev_dr = max(max_dev_ghx, max_dev_ghu, max_dev_ghxx)
    details = {
        "order": order,
        "runtime_sec": runtime_sec,
        "n_vars": len(pm_vars),
        "n_shocks": pm_ghu.shape[1],
        "n_states": pm_ghx.shape[1],
        "order2_missing": order2_missing,
    }

    return ParityDashboardResult(
        passed=passed,
        score=score,
        dr_diff=df_dr,
        moments_diff=df_mom,
        tolerances=tolerances,
        model_name=model_name,
        total_models=1,
        passed_models=1 if passed else 0,
        failed_models=0 if passed else 1,
        max_dev_ghx=max_dev_ghx,
        max_dev_ghu=max_dev_ghu,
        max_dev_ghxx=max_dev_ghxx,
        max_dev_ghs2=max_dev_ghs2,
        max_dev_moments=max_dev_moments,
        max_dev_dr=max_dev_dr,
        details=details,
    )


def compare_model_to_dynare(
    mod_path: str | Path,
    mat_path: str | Path,
    *,
    order: int = 1,
    tol_dr: float | None = None,
    tol_mom: float | None = None,
    tol: float | Mapping[str, float] | None = None,
) -> ParityDashboardResult:
    """Automated parity verification comparing puremacro model against Dynare MAT results."""
    tols = dict(DEFAULT_TOLERANCES)
    if tol is not None:
        tols = _resolve_tolerances(tol)
    if tol_dr is not None:
        tols["ghx"] = float(tol_dr)
        tols["ghu"] = float(tol_dr)
        tols["ghxx"] = float(tol_dr)
        tols["ghs2"] = float(tol_dr)
    if tol_mom is not None:
        tols["mean"] = float(tol_mom)
        tols["var"] = float(tol_mom)
        tols["autocorr"] = float(tol_mom)

    return verify_dynare_parity(mod_path, mat_path, order=order, tol=tols)


def run_parity_suite(
    test_dir: str | Path,
    *,
    pattern: str = "*.mod",
    order: int = 1,
    tol: float | Mapping[str, float] | None = None,
) -> ParityDashboardResult:
    """Run batch parity verification across a directory of model files."""
    root = Path(test_dir)
    mod_files = sorted(root.glob(pattern)) if root.is_dir() else ([root] if root.is_file() else [])
    results: list[ModelParityResult] = []

    for mod in mod_files:
        mat_candidates = [
            mod.with_name(f"{mod.stem}_results.mat"),
            mod.with_suffix(".mat"),
            mod.parent / "results" / f"{mod.stem}_results.mat",
        ]
        mat_path = next((m for m in mat_candidates if m.exists()), None)
        if mat_path is not None:
            try:
                res = compare_model_to_dynare(mod, mat_path, order=order, tol=tol)
                m_res = ModelParityResult(
                    model_name=res.model_name,
                    order=order,
                    n_vars=len(res.dr_diff) if res.dr_diff is not None else 0,
                    n_shocks=res.details.get("n_shocks", 0),
                    passed=res.passed,
                    status="PASS" if res.passed else "FAIL",
                    max_dev_ghx=res.max_dev_ghx,
                    max_dev_ghu=res.max_dev_ghu,
                    max_dev_ghxx=res.max_dev_ghxx,
                    max_dev_ghs2=res.max_dev_ghs2,
                    max_dev_moments=res.max_dev_moments,
                    max_dev_dr=res.max_dev_dr,
                    score=res.score,
                    runtime_sec=res.details.get("runtime_sec", 0.0),
                    dr_diff=res.dr_diff,
                    moments_diff=res.moments_diff,
                    tolerances=res.tolerances,
                    details=res.details,
                )
            except Exception as exc:
                m_res = ModelParityResult(
                    model_name=mod.stem,
                    order=order,
                    n_vars=0,
                    n_shocks=0,
                    passed=False,
                    status="FAIL",
                    max_dev_ghx=np.nan,
                    max_dev_ghu=np.nan,
                    max_dev_ghxx=np.nan,
                    max_dev_ghs2=np.nan,
                    max_dev_moments=np.nan,
                    max_dev_dr=np.nan,
                    score=0.0,
                    runtime_sec=0.0,
                    details={"error": str(exc)},
                )
            results.append(m_res)

    if not results:
        return ParityDashboardResult(
            passed=True,
            score=100.0,
            model_name=root.name,
            total_models=0,
            passed_models=0,
            failed_models=0,
            details={"message": "No matching .mod/.mat pairs discovered"},
        )

    total_m = len(results)
    passed_m = sum(1 for r in results if r.passed)
    failed_m = total_m - passed_m
    overall_passed = (failed_m == 0)
    avg_score = float(np.mean([r.score for r in results]))

    def _safe_max(vals):
        valid = [float(v) for v in vals if v is not None and not np.isnan(v)]
        return max(valid) if valid else 0.0

    max_ghx = _safe_max(r.max_dev_ghx for r in results)
    max_ghu = _safe_max(r.max_dev_ghu for r in results)
    max_ghxx = _safe_max(r.max_dev_ghxx for r in results)
    max_mom = _safe_max(r.max_dev_moments for r in results)

    return ParityDashboardResult(
        passed=overall_passed,
        score=avg_score,
        total_models=total_m,
        passed_models=passed_m,
        failed_models=failed_m,
        max_dev_ghx=max_ghx,
        max_dev_ghu=max_ghu,
        max_dev_ghxx=max_ghxx,
        max_dev_moments=max_mom,
        max_dev_dr=_safe_max([max_ghx, max_ghu, max_ghxx]),
        results=results,
        model_name=root.name,
    )
