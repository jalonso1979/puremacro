"""Paired controls for SW07 finite-sample expectations and covariance weights.

The fixed exact covariance is an oracle evaluated at the known generating
parameters. This experiment does not supply composite-null inference or choose
a replacement for the empirical estimator.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
from typing import Callable, Mapping
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2

from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
from .bridge import MomentTargets, fit_structural
from .empirical_sw07 import (ALL_MOMENTS, FIT_MOMENTS, HELD_OUT_MOMENTS,
    PARAMETER_NAMES, PARAMETER_BOUNDS, UNITS, _PUBLISHED_MODE, _labels,
    covariance_moment_targets, load_empirical_sw07_data, sw07_covariance_moments)
from .sw07_sampling import simulate_sw07_sample, sw07_finite_sample_moments
from .sw07_expectations import sw07_finite_sample_expectations
from .sw07_finite_sample import _STARTS, _binomial_interval, _finite_json, _fit_record, _integer

__all__ = ["SW07EstimatorExperiment", "run_sw07_estimator_experiment",
           "compare_sw07_estimator_phases"]

_VARIANTS = ("population_hac", "finite_hac", "population_oracle", "finite_oracle")
_PHASES = ("calibration", "validation")


@dataclass(frozen=True)
class SW07EstimatorExperiment:
    """One independent phase of the four-variant paired experiment."""
    summary: pd.DataFrame
    draws: pd.DataFrame
    observed_fits: pd.DataFrame
    paired_differences: pd.DataFrame
    draw_moments: np.ndarray
    draw_covariances: np.ndarray
    metadata: Mapping


def _fitters(calibration):
    models = {}
    for rule in ("population", "finite"):
        function = sw07_covariance_moments if rule == "population" else sw07_finite_sample_expectations
        def make_model(function, rule):
            @lru_cache(maxsize=8192)
            def model(crr, em):
                params = {**calibration, "crr": crr, "em": em}
                if rule == "finite":
                    return function(params, ALL_MOMENTS, nobs=156, common_max_lag=4).to_numpy()
                return function(params, ALL_MOMENTS).to_numpy()
            return model
        models[rule] = make_model(function, rule)

    def make_fit(model):
        def fit(target):
            selected = target.select(_labels(FIT_MOMENTS))
            held = target.select(_labels(HELD_OUT_MOMENTS))
            candidates, records, errors = [], [], []
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Structural local inference unavailable:")
                for start in _STARTS:
                    try:
                        result = fit_structural(lambda theta: model(*theta)[:9], selected, start,
                            parameter_names=PARAMETER_NAMES, bounds=PARAMETER_BOUNDS,
                            moment_labels=selected.labels, held_out=held,
                            held_out_moments_at=lambda theta: model(*theta)[9:],
                            assume_correct_specification=True, tol=1e-9, max_nfev=500)
                        candidates.append(result)
                        records.append({"start": start, "theta": result.theta.tolist(),
                            "objective": result.objective, "success": result.success,
                            "message": result.message, "evaluations": result.n_evals,
                            "inference_unavailable_reasons": list(result.diagnostics["inference_unavailable_reasons"])})
                    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                        errors.append(f"{type(exc).__name__}: {exc}")
                        records.append({"start": start, "success": False, "exception": errors[-1]})
            if not candidates:
                return {"crr": np.nan, "em": np.nan, "objective": np.nan,
                    "optimizer_success": False, "numerically_regular": False,
                    "boundary": False, "boundary_crr": False, "boundary_em": False,
                    "rank": np.nan, "j_pvalue": np.nan, "unresolved": True,
                    "status": "failed", "start_successes": 0, "starts_agree": False,
                    "start_diagnostics_json": json.dumps(_finite_json(records), allow_nan=False),
                    "error": "All starts failed: " + "; ".join(errors)}
            successful = [r for r in candidates if r.success]
            best = min(successful or candidates, key=lambda r: r.objective)
            agreement = bool(successful and all(abs(r.objective-best.objective)
                <= 1e-5*max(1., best.objective) for r in successful))
            if "nonstationary_solution" in best.diagnostics["inference_unavailable_reasons"]:
                errors.append("Selected optimizer result failed interior first-order stationarity")
            diagnostic = {"start_successes": len(successful), "starts_agree": agreement,
                "unresolved": bool(errors or len(successful) != len(_STARTS) or not agreement),
                "start_records": records, "start_errors": errors}
            return _fit_record(best, diagnostic)
        return fit
    return {name: make_fit(models[name.split("_")[0]]) for name in _VARIANTS}


def _weighted_target(target, covariance, variant):
    if variant.endswith("_hac"):
        return target
    return MomentTargets(target.values, covariance, target.labels, target.units,
        {**target.metadata, "weight_source": "exact covariance at known fixed DGP; oracle control"})


def _rate(events, possible, total, prefix):
    low, _ = _binomial_interval(events, total)
    _, high = _binomial_interval(possible, total)
    return {f"{prefix}_events_lower": events, f"{prefix}_events_upper": possible,
            f"{prefix}_rate_lower": events/total, f"{prefix}_rate_upper": possible/total,
            f"{prefix}_mc95_lower": low, f"{prefix}_mc95_upper": high}


def _summaries(draws, truth):
    rows = []
    for variant in _VARIANTS:
        group = draws.loc[draws.variant == variant]
        usable = ~group.unresolved & np.isfinite(group.objective)
        failures, total = int((~usable).sum()), len(group)
        reject = int((usable & (group.objective > chi2.ppf(.95, 7))).sum())
        errors = group.loc[usable, list(PARAMETER_NAMES)].to_numpy()-truth
        row = {"phase": group.phase.iloc[0], "variant": variant, "replications": total,
               "usable_fits": int(usable.sum()), "unresolved_draws": failures,
               "boundary_fits": int((usable & group.boundary).sum()),
               "regular_fits": int((usable & group.numerically_regular).sum()),
               "criterion_median": group.loc[usable, "objective"].median(),
               "criterion_q95": group.loc[usable, "objective"].quantile(.95),
               **_rate(reject, reject+failures, total, "naive")}
        for j, name in enumerate(PARAMETER_NAMES):
            row[f"bias_{name}"] = errors[:, j].mean() if len(errors) else np.nan
            row[f"rmse_{name}"] = np.sqrt(np.mean(errors[:, j]**2)) if len(errors) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _paired(draws, truth):
    comparisons = (("population_hac", "finite_hac"), ("population_oracle", "finite_oracle"),
                   ("population_hac", "population_oracle"), ("finite_hac", "finite_oracle"))
    rows = []
    for left, right in comparisons:
        a = draws.loc[draws.variant == left].set_index("replication")
        b = draws.loc[draws.variant == right].set_index("replication").reindex(a.index)
        usable = ~a.unresolved & ~b.unresolved & np.isfinite(a.objective) & np.isfinite(b.objective)
        row = {"left": left, "right": right, "difference": "right minus left",
               "paired_usable": int(usable.sum()), "total_pairs": len(a)}
        for j, parameter in enumerate(PARAMETER_NAMES):
            delta = ((b.loc[usable, parameter]-truth[j])**2 - (a.loc[usable, parameter]-truth[j])**2)
            row[f"mean_squared_error_difference_{parameter}"] = delta.mean()
            row[f"paired_mc_se_{parameter}"] = delta.std(ddof=1)/np.sqrt(len(delta)) if len(delta) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _source_hashes():
    package = Path(__file__).resolve().parents[1]
    files = ("structural/sw07_estimator_experiment.py", "structural/sw07_expectations.py",
             "structural/sw07_finite_sample.py", "structural/sw07_sampling.py",
             "structural/empirical_sw07.py", "structural/bridge.py", "_linalg.py",
             "dsge/smets_wouters.py", "dsge/sw07_observation.py", "dsge/klein.py", "dsge/gensys.py")
    return {name: hashlib.sha256((package/name).read_bytes()).hexdigest() for name in files}


def _save_checkpoint(path, design, rows, moments, covariance):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    document = {"design": design, "rows": rows}
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, document=np.asarray(json.dumps(_finite_json(document), allow_nan=False)),
                            moments=moments, covariances=covariance)
    temporary.replace(path)


def _load_checkpoint(path, design):
    with np.load(path, allow_pickle=False) as saved:
        document = json.loads(str(saved["document"]))
        if document["design"] != _finite_json(design):
            raise ValueError("Checkpoint design, sources or environment differ; use resume=False for a fresh run")
        moments, covariance = saved["moments"].copy(), saved["covariances"].copy()
        rows = document["rows"]
    count = len(moments)
    expected = [(i, name) for i in range(count) for name in _VARIANTS]
    if (not 1 <= count <= design["replications"] or moments.shape != (count, 15)
            or covariance.shape != (count, 15, 15)
            or [(row["replication"], row["variant"]) for row in rows] != expected):
        raise ValueError("Malformed paired-experiment checkpoint")
    for row in rows:
        for key in ("objective", "crr", "em", "rank", "j_pvalue"):
            if row.get(key) is None:
                row[key] = np.nan
    return rows, moments, covariance


def run_sw07_estimator_experiment(*, replications: int = 399, phase: str = "calibration",
        seed: int = 20261002, checkpoint_dir: str | Path | None = None,
        resume: bool = True, progress: Callable[[str, int, int], None] | None = None
        ) -> SW07EstimatorExperiment:
    """Run one independent phase, sharing each sample among all four variants.

    Use 399 calibration and 999 validation replications for the recorded study.
    The oracle covariance is evaluated at known generating parameters, fixed
    throughout optimization. No normal parameter inference is supplied. Failed
    fits remain in all rate denominators. Checkpoint resume requires an identical
    design, numerical sources, data and dependency versions.
    """
    replications = _integer(replications, "replications", 2)
    seed = _integer(seed, "seed", 0)
    if phase not in _PHASES:
        raise ValueError("phase must be calibration or validation")
    if not isinstance(resume, (bool, np.bool_)):
        raise ValueError("resume must be boolean")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable")
    calibration = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS, **_PUBLISHED_MODE}
    truth = np.array([calibration[key] for key in PARAMETER_NAMES])
    data, provenance = load_empirical_sw07_data()
    observed = covariance_moment_targets(data, ALL_MOMENTS, units=UNITS, bandwidth=8)
    exact = sw07_finite_sample_moments(calibration, nobs=156, common_max_lag=4)
    fitters = _fitters(calibration)
    observed_rows = []
    for variant in _VARIANTS:
        record = fitters[variant](_weighted_target(observed, exact.covariance, variant))
        observed_rows.append({"variant": variant, **record})
    design = {"schema": 1, "phase": phase, "phase_index": _PHASES.index(phase),
        "replications": replications, "seed": seed, "dgp": calibration,
        "source_hashes": _source_hashes(), "observed_source": provenance["source_csv_normalized_sha256"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "pandas": pd.__version__}}
    path = None if checkpoint_dir is None else Path(checkpoint_dir)/f"{phase}.npz"
    moments = np.full((replications, 15), np.nan)
    covariances = np.full((replications, 15, 15), np.nan)
    rows, completed = [], 0
    if path is not None and resume and path.exists():
        rows, stored_moments, stored_covariance = _load_checkpoint(path, design)
        completed = len(stored_moments)
        moments[:completed], covariances[:completed] = stored_moments, stored_covariance
        if progress is not None:
            progress(phase, completed, replications)
    resumed = completed
    for i in range(completed, replications):
        stream = int(np.random.SeedSequence([seed, _PHASES.index(phase), i]).generate_state(1, dtype=np.uint64)[0])
        sample_error, target, digest = "", None, ""
        try:
            sample = simulate_sw07_sample(calibration, nobs=156, seed=stream)
            digest = hashlib.sha256(sample.to_numpy(dtype="<f8").tobytes()).hexdigest()
            target = covariance_moment_targets(sample, ALL_MOMENTS, units=UNITS, bandwidth=8)
            moments[i], covariances[i] = target.values, target.covariance
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            sample_error = f"{type(exc).__name__}: {exc}"
        for variant in _VARIANTS:
            row = {"phase": phase, "replication": i, "array_row": i, "variant": variant,
                "simulation_seed": str(stream), "sample_sha256": digest,
                "crr": np.nan, "em": np.nan, "objective": np.nan, "optimizer_success": False,
                "numerically_regular": False, "boundary": False, "boundary_crr": False,
                "boundary_em": False, "rank": np.nan, "j_pvalue": np.nan,
                "unresolved": True, "status": "failed", "error": sample_error,
                "start_successes": 0, "starts_agree": False, "start_diagnostics_json": "[]"}
            if target is not None:
                try:
                    row.update(fitters[variant](_weighted_target(target, exact.covariance, variant)))
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        if path is not None and (i == 0 or (i+1) % 10 == 0 or i+1 == replications):
            _save_checkpoint(path, design, rows, moments[:i+1], covariances[:i+1])
        if progress is not None:
            progress(phase, i+1, replications)
    draws = pd.DataFrame(rows)
    metadata = {**design, "source": provenance, "variants": list(_VARIANTS),
        "moment_labels": list(exact.labels), "oracle_covariance": exact.covariance.tolist(),
        "oracle_expectations_at_truth": exact.values.tolist(), "true_parameters": truth.tolist(),
        "nobs": 156, "common_max_lag": 4, "effective_observations": 152, "bandwidth": 8,
        "starts": _STARTS, "bounds": PARAMETER_BOUNDS, "tolerance": 1e-9, "max_nfev": 500,
        "checkpoint_resumed_draws": resumed, "parameter_inference_reported": False,
        "seed_rule": "uint64 first state of SeedSequence([seed, calibration0_or_validation1, replication])",
        "scope": "paired controls at one known fixed Gaussian DGP; oracle weights are not feasible estimated weights",
        "limitations": ["no composite-null or nuisance-calibration uncertainty", "no estimator automatically selected",
                        "no parameter confidence intervals", "revised historical data; possible regime changes",
                        "regular numerical fits do not imply inferential validity"]}
    return SW07EstimatorExperiment(_summaries(draws, truth), draws, pd.DataFrame(observed_rows),
                                   _paired(draws, truth), moments, covariances, metadata)


def _critical_bounds(values, unresolved, alpha=.05):
    values, unresolved = np.asarray(values, dtype=float), np.asarray(unresolved, dtype=bool)
    total = len(values)
    if total == 0 or values.shape != unresolved.shape:
        raise ValueError("critical values require a nonempty matching calibration sample")
    valid = ~unresolved & np.isfinite(values)
    if np.any(values[valid] < 0):
        raise ValueError("minimum-distance criteria must be nonnegative")
    rank = int(np.ceil((1-alpha)*(total+1)))
    if rank > total:
        return np.inf, np.inf, rank
    lo = np.sort(np.where(valid, values, 0.))[rank-1]
    hi = np.sort(np.where(valid, values, np.inf))[rank-1]
    return float(lo), float(hi), rank


def compare_sw07_estimator_phases(calibration: SW07EstimatorExperiment,
                                  validation: SW07EstimatorExperiment) -> pd.DataFrame:
    """Apply predeclared calibration cutoffs to an independent validation phase.

    Missing calibration criteria produce cutoff bounds; missing validation fits
    remain unknown rejections. Monte Carlo intervals condition on the realized
    calibration pool. They are not structural parameter uncertainty intervals.
    """
    if calibration.metadata["phase"] != "calibration" or validation.metadata["phase"] != "validation":
        raise ValueError("comparison requires calibration then validation phases")
    for key in ("dgp", "seed", "source_hashes", "observed_source", "environment", "variants"):
        if calibration.metadata[key] != validation.metadata[key]:
            raise ValueError(f"phase designs differ: {key}")
    if set(calibration.draws.simulation_seed) & set(validation.draws.simulation_seed):
        raise ValueError("calibration and validation simulation streams overlap")
    rows = []
    for variant in _VARIANTS:
        a = calibration.draws.loc[calibration.draws.variant == variant]
        b = validation.draws.loc[validation.draws.variant == variant]
        lo, hi, rank = _critical_bounds(a.objective, a.unresolved)
        usable = ~b.unresolved & np.isfinite(b.objective)
        lower = int((usable & (b.objective > hi)).sum())
        upper = int((usable & (b.objective > lo)).sum() + (~usable).sum())
        row = {"variant": variant, "calibration_draws": len(a), "validation_draws": len(b),
               "calibration_unresolved": int((a.unresolved | ~np.isfinite(a.objective)).sum()),
               "validation_unresolved": int((~usable).sum()),
               "critical_rank": rank, "critical_lower": lo, "critical_upper": hi,
               "critical_lower_unbounded": bool(np.isinf(lo)), "critical_upper_unbounded": bool(np.isinf(hi)),
               **_rate(lower, upper, len(b), "calibrated")}
        rows.append(row)
    return pd.DataFrame(rows)
