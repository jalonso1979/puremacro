"""Automated Dynare Parity Verification Harness for puremacro 2.9.0.

Provides rigorous automated validation comparing puremacro DSGE solutions against
an official Dynare ``oo_`` results structure supplied as a Python dict.
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
from .dynare_results import load_dynare_dr, load_dynare_moments, _to_plain_dict

_MAT_INPUT_REMOVED = (
    "puremacro 4.0.0 no longer reads MATLAB .mat files. Pass the Dynare oo_ "
    "results as a Python dict, e.g. "
    "`scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)`, or "
    "build the model with puremacro's own pure-Python .mod parser "
    "(`puremacro.dsge.build_dynare`)."
)


DEFAULT_TOLERANCES: dict[str, float] = {
    "ys": 1e-6,
    "ghxu": 1e-4,
    "ghuu": 1e-4,
    "shock_cov": 1e-10,
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


def _call_theoretical_moments(model: Any, lags: int, pruning: bool = False) -> Any:
    """Call ``model.theoretical_moments`` requesting ``lags`` autocorrelation lags.

    With ``pruning=True`` the exact moments of the pruned solution are
    requested (``PrunedDSGESolution.theoretical_moments(pruning=True)``);
    an object without that option raises ``TypeError``.

    :class:`~puremacro.dsge.build.LinearModel` spells the lag count ``ar=``
    while the pruned higher-order solutions
    (:class:`~puremacro.dsge.pruning.PrunedDSGESolution`) spell it ``lags=``.
    4.3.0 always passed ``ar=``, so every order-2 comparison with supplied
    moments ended ``UNAVAILABLE`` on a swallowed ``TypeError``. Objects whose
    signature accepts ``**kwargs`` keep receiving ``ar=``.
    """
    import inspect

    try:
        params = inspect.signature(model.theoretical_moments).parameters
    except (TypeError, ValueError):  # builtins or callables without a signature
        params = {}
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if pruning:
        if "pruning" not in params:
            raise TypeError(f"{type(model).__name__}.theoretical_moments has no pruned-moment option")
        return model.theoretical_moments(lags=lags, pruning=True)
    if "ar" in params or accepts_any:
        return model.theoretical_moments(ar=lags)
    if "lags" in params:
        return model.theoretical_moments(lags=lags)
    return model.theoretical_moments()


def _resolve_shock_covariance(model: Any) -> np.ndarray | None:
    """Innovation covariance a puremacro object declares, or ``None``.

    * :class:`~puremacro.dsge.build.LinearModel` (it has ``_shock_cov``):
      the covariance declared through the ``shocks`` block of the ``.mod``
      file or ``shock_cov=``, symmetrised as its moment routine
      (``_shock_covariance(None)``) does. A model that declares none resolves
      to ``None``: its moment routines then assume the identity, but that is
      a convention rather than a declaration, so it is not compared with
      ``M_.Sigma_e`` (the order-1 decision rules do not depend on it).
    * Pruned higher-order solutions: the public ``shock_cov`` field, which
      their moments and risk correction use (the declared covariance, or the
      identity a solution built without one fills in at construction).
    * Other objects: a ``shock_cov`` or ``Sigma`` attribute.

    A decision-rule-only object resolves to ``None``. ``None`` makes the
    harness report ``covariance_status == "UNAVAILABLE"``, not counted as a
    check.
    """
    if hasattr(model, "_shock_cov") and not hasattr(model, "shock_cov"):
        if getattr(model, "_shock_cov", None) is None:
            return None
        resolver = getattr(model, "_shock_covariance", None)
        cov = resolver(None) if callable(resolver) else model._shock_cov
    else:
        cov = getattr(model, "shock_cov", None)
        if cov is None:
            cov = getattr(model, "Sigma", None)
    if cov is None:
        return None
    try:
        return np.asarray(cov, dtype=float)
    except (TypeError, ValueError):
        return None


def _zero_variance_rows(th: Any, pm_vars: Sequence[str]) -> np.ndarray | None:
    """Boolean mask of variables whose puremacro theoretical variance is exactly zero."""
    try:
        if hasattr(th, "covariance"):
            var = np.diag(th.covariance.loc[list(pm_vars), list(pm_vars)].to_numpy(dtype=float))
        else:
            var = np.asarray(th.variance, dtype=float).reshape(len(pm_vars))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return ~(var > 0.0)


def verify_dynare_parity(
    puremacro_model: Any,
    dynare_output: Any,
    order: int = 1,
    tol: float | Mapping[str, float] | None = None,
    pruning: bool = False,
) -> ParityDashboardResult:
    """Verify puremacro DSGE model solution against Dynare output.

    Parameters
    ----------
    puremacro_model : LinearModel | PrunedDSGESolution | DynareDR | str | Path
        Solved puremacro model instance or path to a .mod file.
    dynare_output : DynareDR | Dynare2ndDR | dict | str | Path
        Dynare decision rules or an ``oo_`` results dict. Reading MATLAB
        ``.mat`` files was removed in 4.0.0.
    order : {1, 2}, default 1
        Perturbation order to compare.
    tol : float | dict[str, float], optional
        Tolerance thresholds. Defaults: 1e-6 for (ghx, ghu), 1e-4 for (ghxx, ghs2),
        1e-5 for moments.
    pruning : bool, default False
        At order 2, compare supplied moments with the exact moments of the
        pruned solution, the convention of Dynare's
        ``stoch_simul(order=2, pruning)``. Ignored at order 1.

    Returns
    -------
    ParityDashboardResult
        Consolidated scorecard with per-variable deviations and overall parity score.

    Notes
    -----
    * Supplied ``oo_.mean`` / ``oo_.var`` / ``oo_.autocorr`` are compared with
      the theoretical moments of the object solved at the compared order: at
      order 2 that is ``LinearModel.solve(order=2)`` (or the pruned solution
      passed in), so the reference mean is compared with the risk-corrected
      second-order mean and the second moments with the pruned solution's
      first-order second moments. That is the convention of Dynare's
      ``stoch_simul(order=2)`` *without* the ``pruning`` option. With
      ``pruning``, Dynare reports ``oo_.var`` and ``oo_.autocorr`` of the
      pruned second-order system (``disp_th_moments_pruned_state_space``),
      which include O(sigma^4) terms: compare such a reference with
      ``pruning=True``. The lag count follows the supplied autocorrelation
      array.
    * A variable whose puremacro theoretical variance is exactly zero has
      undefined (NaN) autocorrelations, and so does a Dynare reference for
      it. Rows that are non-finite on both sides for such variables are
      treated as matching and listed in ``details["moments_excluded"]``; a
      non-finite value on one side only still makes the moment block
      ``UNAVAILABLE``.
    * ``M_.Sigma_e``, when supplied, is compared with the innovation
      covariance the puremacro object declares: the covariance of a
      ``LinearModel``'s ``shocks`` block / ``shock_cov=``, or ``shock_cov``
      of a pruned solution (always set; the identity when it was built
      without one). The comparison is a counted check with tolerance
      ``tol["shock_cov"]`` (default 1e-10) and is reported in
      ``details["covariance_status"]``. A ``LinearModel`` that declares no
      covariance, or a decision-rule-only object, cannot be compared and
      yields ``covariance_status == "UNAVAILABLE"``, which is not counted as
      a check.
    * An ``oo_`` or ``oo_.dr`` that is present but empty (Dynare initialises
      ``oo_.dr = []`` and only ``stoch_simul``/``check`` populate it; scipy
      loads that as an empty array), or an ``oo_`` loaded as a numpy record
      array (``scipy.io.loadmat`` with its default ``struct_as_record=True``),
      returns status ``UNAVAILABLE`` with a message naming the cause instead
      of raising. Missing or malformed comparisons never pass.
    """
    if order not in (1, 2):
        raise ValueError("Parity supports orders 1 and 2 only")
    t0 = time.perf_counter()
    tolerances = _resolve_tolerances(tol)
    model_name = "model"

    # 1. Resolve puremacro model
    if isinstance(puremacro_model, (str, Path)):
        p_mod = Path(puremacro_model)
        model_name = p_mod.stem
        if p_mod.suffix == ".mat":
            raise TypeError(_MAT_INPUT_REMOVED)
        from .dynare import load_mod
        pm_model = load_mod(p_mod, order=order)
    else:
        pm_model = puremacro_model
        if hasattr(pm_model, "name") and pm_model.name:
            model_name = str(pm_model.name)

    # Extract puremacro decision rules
    pm_sol = None
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
    # Moments and the innovation covariance come from the object solved at the
    # compared order (a first-order model's moments lack the order-2 risk
    # correction that Dynare's oo_.mean carries).
    moment_model = pm_sol if (pm_sol is not None and hasattr(pm_sol, "theoretical_moments")) else pm_model

    def unavailable(message, **details):
        return ParityDashboardResult(
            passed=False, score=0.0, model_name=model_name,
            total_models=1, passed_models=0, failed_models=1,
            tolerances=tolerances, max_dev_ghx=np.nan, max_dev_ghu=np.nan,
            max_dev_ghxx=np.nan, max_dev_ghs2=np.nan,
            max_dev_moments=np.nan, max_dev_dr=np.nan,
            details={"status": "UNAVAILABLE", "error": message,
                     "runtime_sec": time.perf_counter() - t0, **details},
        )

    # Missing tensors are not zero tensors. Incomplete references cannot pass.
    dyn_moments = {}
    raw = None
    if isinstance(dynare_output, (str, Path)):
        raise TypeError(_MAT_INPUT_REMOVED)
    if isinstance(dynare_output, (DynareDR, Dynare2ndDR)):
        dyn_dr = dynare_output
    elif isinstance(dynare_output, dict):
        raw = _to_plain_dict(dynare_output)
        oo = raw.get("oo_")
        if oo is not None and not isinstance(oo, dict):
            if getattr(getattr(oo, "dtype", None), "names", None):
                return unavailable(
                    "oo_ is a numpy record array, not a struct/dict: load Dynare .mat files with "
                    "scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)",
                    order2_missing=order == 2,
                )
            return unavailable(
                f"oo_ is present but empty or not a results structure (got {type(oo).__name__})",
                order2_missing=order == 2,
            )
        oo = oo if isinstance(oo, dict) else {}
        fields = oo.get("dr")
        if fields is not None and not isinstance(fields, dict):
            # Dynare initialises ``oo_.dr = []`` (an empty ndarray once loaded
            # with scipy) and only stoch_simul/check fill it: not a reference.
            return unavailable(
                "oo_.dr is present but empty or not a decision-rule structure "
                f"(got {type(fields).__name__}); the Dynare run never populated it",
                order2_missing=order == 2,
            )
        fields = fields if isinstance(fields, dict) else {}
        required = ["ghx", "ghu"] + (["ghxx", "ghxu", "ghuu", "ghs2"] if order == 2 else [])
        missing = [k for k in required if fields.get(k) is None]
        if fields.get("ys") is None and oo.get("steady_state") is None:
            missing.append("ys")
        if missing:
            return unavailable(f"Missing decision-rule fields: {missing}", order2_missing=order == 2)
        try:
            dyn_dr = load_dynare_dr(raw, order=order)
            parsed = load_dynare_moments(raw)
            # Only actual supplied moments are requested; deterministic steady
            # states are not a substitute for unconditional means at order two.
            dyn_moments = {k: parsed[k] for k in ("mean", "var", "autocorr")
                           if oo.get(k) is not None}
        except (KeyError, TypeError, ValueError) as exc:
            return unavailable(str(exc), order2_missing=order == 2)
    else:
        raise TypeError(f"Unrecognized dynare_output type: {type(dynare_output)}")

    pm_vars = list(pm_dr.variable_names)
    dyn_vars = list(dyn_dr.variable_names)
    for attr in ("variable_names", "state_variables", "shock_names"):
        left, right = list(getattr(pm_dr, attr)), list(getattr(dyn_dr, attr))
        if len(set(left)) != len(left) or len(set(right)) != len(right) or set(left) != set(right):
            return unavailable(f"{attr} mismatch or duplicate labels: {left} vs {right}")
    if not pm_vars:
        return unavailable("No endogenous variables supplied")

    fields = ["ys", "ghx", "ghu"] + (["ghxx", "ghxu", "ghuu", "ghs2"] if order == 2 else [])
    deviations = {}
    n_x, n_u, n_v = len(pm_dr.state_variables), len(pm_dr.shock_names), len(pm_vars)
    sizes = {"ys": None, "ghs2": None, "ghx": n_x, "ghu": n_u,
             "ghxx": n_x**2, "ghxu": n_x * n_u, "ghuu": n_u**2}
    for name in fields:
        try:
            left, right = getattr(pm_dr, name), getattr(dyn_dr, name)
            if not left.index.is_unique or not right.index.is_unique:
                raise ValueError(f"{name}: duplicate row labels")
            if set(left.index) != set(pm_vars) or set(right.index) != set(pm_vars):
                raise ValueError(f"{name}: row labels mismatch")
            if sizes[name] is not None:
                if (not left.columns.is_unique or not right.columns.is_unique
                        or set(left.columns) != set(right.columns)
                        or len(left.columns) != sizes[name]):
                    raise ValueError(f"{name}: column labels or tensor dimensions mismatch")
                a = left.loc[pm_vars].to_numpy(dtype=float)
                b = right.loc[pm_vars, left.columns].to_numpy(dtype=float)
            else:
                a = left.loc[pm_vars].to_numpy(dtype=float).reshape(n_v, 1)
                b = right.loc[pm_vars].to_numpy(dtype=float).reshape(n_v, 1)
            if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                raise ValueError(f"{name}: incompatible shapes or non-finite values")
            deviations[name] = np.max(np.abs(a - b), axis=1, initial=0.0)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            return unavailable(str(exc), order2_missing=order == 2 and not hasattr(dyn_dr, "ghxx"))

    rows = {f"dev_{k}": v for k, v in deviations.items()}
    rows.update({f"tol_{k}": tolerances[k] for k in fields})
    flags = np.column_stack([deviations[k] <= tolerances[k] for k in fields])
    rows["status"] = np.where(np.all(flags, axis=1), "PASS", "FAIL")
    df_dr = pd.DataFrame(rows, index=pm_vars)
    passed_checks, total_checks = int(flags.sum()), flags.size
    moment_rows = []
    unavailable_moments = False
    excluded: dict[str, list[str]] = {}
    if dyn_moments:
        try:
            if not hasattr(moment_model, "theoretical_moments"):
                raise ValueError("Supplied reference moments cannot be compared to a decision-rule-only object")
            lags = int(np.asarray(dyn_moments.get("autocorr", np.empty((0, 5)))).shape[-1])
            th = _call_theoretical_moments(moment_model, lags, pruning=pruning and order == 2)
            zero_var = _zero_variance_rows(th, pm_vars)
            perm = [dyn_vars.index(v) for v in pm_vars]
            for key, expected in dyn_moments.items():
                expected = np.asarray(expected, dtype=float)
                if key == "mean":
                    actual = th.moments.loc[pm_vars, "Mean"].to_numpy() if hasattr(th, "moments") else np.asarray(th.mean).reshape(n_v)
                    expected = expected.reshape(n_v)[perm]
                elif key == "var":
                    actual = th.covariance.loc[pm_vars, pm_vars].to_numpy() if hasattr(th, "covariance") else np.diag(np.asarray(th.variance).reshape(n_v))
                    if expected.size == n_v:
                        expected = expected.reshape(n_v)[perm]
                        actual = np.diag(actual)
                    else:
                        expected = expected.reshape(n_v, n_v)[np.ix_(perm, perm)]
                else:
                    actual = th.autocorr.loc[pm_vars].to_numpy()
                    if expected.ndim == 3:
                        expected = np.stack([np.diag(expected[:, :, j]) for j in range(expected.shape[-1])], axis=1)
                    expected = expected.reshape(n_v, -1)[perm]
                if actual.shape != expected.shape:
                    raise ValueError(f"{key}: moment shape mismatch or non-finite values")
                if key == "autocorr" and zero_var is not None and actual.ndim == 2:
                    # Zero-variance variables have undefined autocorrelations on
                    # both sides (0/0): matching all-NaN rows are not a mismatch.
                    undefined = (zero_var & ~np.isfinite(actual).any(axis=1)
                                 & ~np.isfinite(expected).any(axis=1))
                    if undefined.any():
                        excluded[key] = [v for v, u in zip(pm_vars, undefined) if u]
                        actual, expected = actual[~undefined], expected[~undefined]
                if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual)):
                    raise ValueError(f"{key}: moment shape mismatch or non-finite values")
                dev = float(np.max(np.abs(actual - expected), initial=0.0))
                ok = dev <= tolerances[key]
                moment_rows.append({"moment": key, "deviation": dev, "tolerance": tolerances[key], "status": "PASS" if ok else "FAIL"})
                passed_checks += int(ok)
                total_checks += 1
        except (AttributeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            unavailable_moments = True
            total_checks += 1
            moment_rows.append({"moment": "unavailable", "deviation": np.nan, "status": "UNAVAILABLE", "error": str(exc)})

    # A DR contains covariance-scaled risk terms but no covariance metadata.
    # Compare shock covariance when supplied on both models; otherwise record
    # its unavailability without claiming a moments/innovation-distribution test.
    covariance_status = "UNAVAILABLE"
    M_ = raw.get("M_") if raw is not None else None
    dyn_cov = M_.get("Sigma_e") if isinstance(M_, dict) else None
    # A LinearModel's declaration decides (its order-2 solution fills in the
    # identity when nothing was declared); otherwise the object compared.
    cov_source = pm_model if (hasattr(pm_model, "_shock_cov") and not hasattr(pm_model, "shock_cov")) else moment_model
    pm_cov = _resolve_shock_covariance(cov_source)
    if dyn_cov is not None and pm_cov is not None:
        try:
            a, b = np.asarray(pm_cov, dtype=float), np.asarray(dyn_cov, dtype=float)
            perm = [list(dyn_dr.shock_names).index(v) for v in pm_dr.shock_names]
            b = b.reshape(n_u, n_u)[np.ix_(perm, perm)]
            ok = (a.shape == b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b))
                  and np.max(np.abs(a - b), initial=0.0) <= tolerances["shock_cov"])
        except (ValueError, TypeError):
            ok = False
        passed_checks += int(ok)
        total_checks += 1
        covariance_status = "PASS" if ok else "FAIL"
    passed = passed_checks == total_checks and not unavailable_moments
    maxima = {k: float(np.max(v, initial=0.0)) for k, v in deviations.items()}
    return ParityDashboardResult(
        passed=passed, score=100.0 * passed_checks / total_checks, dr_diff=df_dr,
        moments_diff=pd.DataFrame(moment_rows), tolerances=tolerances, model_name=model_name,
        total_models=1, passed_models=int(passed), failed_models=int(not passed),
        max_dev_ghx=maxima["ghx"], max_dev_ghu=maxima["ghu"],
        max_dev_ghxx=maxima.get("ghxx", 0.0), max_dev_ghs2=maxima.get("ghs2", 0.0),
        max_dev_moments=float("nan") if unavailable_moments else max([r["deviation"] for r in moment_rows] or [0.0]),
        max_dev_dr=max(maxima.values()),
        details={"order": order, "status": "UNAVAILABLE" if unavailable_moments else ("PASS" if passed else "FAIL"),
                 "runtime_sec": time.perf_counter() - t0, "n_vars": n_v, "n_states": n_x, "n_shocks": n_u,
                 "order2_missing": False, "moments_status": "NOT_REQUESTED" if not dyn_moments else ("UNAVAILABLE" if unavailable_moments else "COMPARED"),
                 "covariance_status": covariance_status, "max_deviations": maxima,
                 "moments_excluded": excluded},
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
    dynare_results: Mapping[str, dict] | None = None,
    pattern: str = "*.mod",
    order: int = 1,
    tol: float | Mapping[str, float] | None = None,
) -> ParityDashboardResult:
    """Run batch parity verification across a directory of model files.

    Parameters
    ----------
    test_dir : str or Path
        Directory of ``.mod`` models, or a single model file.
    dynare_results : mapping of str to dict, optional
        Dynare ``oo_`` structures to compare against, keyed by model stem
        (``sw07`` for ``sw07.mod``). Required since 4.0.0: puremacro no longer
        reads MATLAB files, so it cannot discover a ``*_results.mat`` companion
        on disk. Load each reference however you like and pass the mapping.

    Notes
    -----
    A model with no entry in ``dynare_results`` is reported ``UNAVAILABLE`` and
    is **not** counted as passing. If nothing could be compared at all the
    result is ``passed=False``: a suite that checked nothing has not verified
    anything, and saying otherwise would turn a green dashboard into a lie.
    """
    root = Path(test_dir)
    mod_files = sorted(root.glob(pattern)) if root.is_dir() else ([root] if root.is_file() else [])
    references: Mapping[str, dict] = dynare_results or {}
    results: list[ModelParityResult] = []
    unavailable: list[str] = []

    for mod in mod_files:
        reference = references.get(mod.stem)
        if reference is None:
            unavailable.append(mod.stem)
            results.append(ModelParityResult(
                model_name=mod.stem,
                order=order,
                n_vars=0,
                n_shocks=0,
                passed=False,
                status="UNAVAILABLE",
                max_dev_ghx=np.nan,
                max_dev_ghu=np.nan,
                max_dev_ghxx=np.nan,
                max_dev_ghs2=np.nan,
                max_dev_moments=np.nan,
                max_dev_dr=np.nan,
                score=0.0,
                runtime_sec=0.0,
                details={"error": (
                    f"no Dynare reference supplied for {mod.stem!r}; pass one via "
                    "run_parity_suite(..., dynare_results={'" + mod.stem + "': oo_dict})"
                )},
            ))
        else:
            try:
                res = compare_model_to_dynare(mod, reference, order=order, tol=tol)
                m_res = ModelParityResult(
                    model_name=res.model_name,
                    order=order,
                    n_vars=len(res.dr_diff) if res.dr_diff is not None else 0,
                    n_shocks=res.details.get("n_shocks", 0),
                    passed=res.passed,
                    status=res.details.get("status", "PASS" if res.passed else "FAIL"),
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
            passed=False,
            score=0.0,
            model_name=root.name,
            total_models=0,
            passed_models=0,
            failed_models=0,
            details={"message": (
                f"no models matched {pattern!r} under {root}; nothing was verified"
            )},
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
