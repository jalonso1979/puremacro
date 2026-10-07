"""Conditional finite-sample diagnostics for the observed SW07 moment study.

The experiment repeats its moment, HAC and multistart estimation stages.
Simulated tail fractions describe fixed Gaussian data-generating processes;
they are not composite-null p-values or valid boundary bootstrap inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
from typing import Callable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.stats import beta, chi2

from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
from .bridge import fit_structural
from .empirical_sw07 import (
    ALL_MOMENTS, FIT_MOMENTS, HELD_OUT_MOMENTS, PARAMETER_BOUNDS,
    PARAMETER_NAMES, UNITS, _PUBLISHED_MODE, _labels,
    covariance_moment_targets, load_empirical_sw07_data, sw07_covariance_moments,
)
from .sw07_sampling import simulate_sw07_sample, sw07_finite_sample_moments

__all__ = ["SW07FiniteSampleStudy", "run_sw07_finite_sample"]

_SCENARIOS = ("baseline_fixed", "published_fixed", "baseline_fitted", "published_fitted")
_STARTS = ((.5, .1), (.75, .4), (.95, .7),
           (SW07_POSTERIOR_MODE["crr"], SW07_SHOCK_STDS["em"]))


@dataclass(frozen=True)
class SW07FiniteSampleStudy:
    """Auditable Monte Carlo draws, exact references and conservative summaries."""

    summary: pd.DataFrame
    draws: pd.DataFrame
    moment_audit: pd.DataFrame
    covariance_audit: pd.DataFrame
    observed_fits: pd.DataFrame
    draw_moments: np.ndarray
    draw_covariances: np.ndarray
    metadata: Mapping


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _binomial_interval(events, total):
    """Clopper-Pearson interval for simulation probability, not model parameters."""
    if total == 0:
        return np.nan, np.nan
    lower = 0. if events == 0 else float(beta.ppf(.025, events, total-events+1))
    upper = 1. if events == total else float(beta.ppf(.975, events+1, total-events))
    return lower, upper


def _finite_json(value):
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _checkpoint_design(dgp, *, name, seed, replications, bandwidth, source_hash):
    package = Path(__file__).resolve().parents[1]
    paths = ("structural/sw07_finite_sample.py", "structural/sw07_sampling.py",
             "structural/empirical_sw07.py", "structural/bridge.py", "_linalg.py",
             "dsge/smets_wouters.py", "dsge/sw07_observation.py", "dsge/klein.py", "dsge/gensys.py")
    source = {name: hashlib.sha256((package/name).read_bytes()).hexdigest() for name in paths}
    design = {"schema": 1, "scenario": name, "dgp": dgp, "seed": seed,
              "replications": replications, "bandwidth": bandwidth, "observed_source": source_hash,
              "source_hashes": source, "python": platform.python_version(),
              "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}
    serialized = json.dumps(design, sort_keys=True, allow_nan=False)
    return design, hashlib.sha256(serialized.encode()).hexdigest()


def _save_checkpoint(path, design, fingerprint, rows, moments, covariances, hac, counts):
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"design": design, "fingerprint": fingerprint, "rows": rows,
                "hac_bandwidths": list(hac), "hac_counts": counts}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, document=np.asarray(json.dumps(_finite_json(document), allow_nan=False)),
                            moments=moments, covariances=covariances,
                            hac_sums=np.stack(list(hac.values())))
    temporary.replace(path)


def _load_checkpoint(path, fingerprint, bandwidths, replications, q):
    with np.load(path, allow_pickle=False) as saved:
        document = json.loads(str(saved["document"]))
        if document.get("fingerprint") != fingerprint:
            raise ValueError("Checkpoint settings, source code, data or environment differ; start a fresh run with resume=False")
        rows = document["rows"]
        length = len(rows)
        moments, covariances, hac = (saved[key].copy() for key in ("moments", "covariances", "hac_sums"))
    if (not 1 <= length <= replications or moments.shape != (length, q)
            or covariances.shape != (length, q, q) or hac.shape != (len(bandwidths), q, q)
            or document["hac_bandwidths"] != bandwidths
            or [row["replication"] for row in rows] != list(range(length))
            or not np.isfinite(hac).all()):
        raise ValueError("Malformed Monte Carlo checkpoint")
    counts = {int(key): value for key, value in document["hac_counts"].items()}
    if set(counts) != set(bandwidths) or any(not isinstance(n, int) or not 0 <= n <= length for n in counts.values()):
        raise ValueError("Malformed Monte Carlo checkpoint counts")
    for row in rows:
        for key in ("objective", "crr", "em", "rank", "j_pvalue"):
            if row[key] is None:
                row[key] = np.nan
    return rows, moments, covariances, dict(zip(bandwidths, hac)), counts


def _fitter(calibration):
    # Exact memoization only. No response-surface interpolation or new solver.
    @lru_cache(maxsize=4096)
    def model(crr, em):
        return sw07_covariance_moments({**calibration, "crr": crr, "em": em}, ALL_MOMENTS).to_numpy()

    def fit(targets):
        selected = targets.select(_labels(FIT_MOMENTS))
        held = targets.select(_labels(HELD_OUT_MOMENTS))
        fits, errors, start_records = [], [], []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Structural local inference unavailable:")
            for start in _STARTS:
                try:
                    fits.append(fit_structural(
                        lambda theta: model(*theta)[:9], selected, start,
                        parameter_names=PARAMETER_NAMES, bounds=PARAMETER_BOUNDS,
                        moment_labels=selected.labels, held_out=held,
                        held_out_moments_at=lambda theta: model(*theta)[9:],
                        assume_correct_specification=True, tol=1e-9, max_nfev=500))
                    candidate = fits[-1]
                    start_records.append({"start": start, "theta": candidate.theta.tolist(),
                        "objective": candidate.objective, "success": candidate.success,
                        "message": candidate.message, "evaluations": candidate.n_evals,
                        "inference_unavailable_reasons": list(candidate.diagnostics["inference_unavailable_reasons"])})
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
                    start_records.append({"start": start, "exception": errors[-1], "success": False})
        successful = [candidate for candidate in fits if candidate.success]
        if not fits:
            raise ValueError("All starts failed: " + "; ".join(errors))
        best = min(successful or fits, key=lambda candidate: candidate.objective)
        agreement = bool(successful and all(
            abs(candidate.objective-best.objective) <= 1e-5*max(1., best.objective)
            for candidate in successful))
        if "nonstationary_solution" in best.diagnostics["inference_unavailable_reasons"]:
            errors.append("Selected optimizer result failed the numerical stationarity check")
        unresolved = bool(errors or len(successful) != len(_STARTS) or not agreement)
        return best, {"start_successes": len(successful), "start_failures": len(_STARTS)-len(successful),
                      "starts_agree": agreement, "unresolved": unresolved,
                      "start_objectives": [float(candidate.objective) for candidate in fits],
                      "start_records": start_records,
                      "start_errors": errors}
    return fit


def _fit_record(result, diagnostics):
    regular = bool(result.inference_valid and not diagnostics["unresolved"])
    return {"crr": float(result.theta[0]), "em": float(result.theta[1]),
            "objective": float(result.objective), "optimizer_success": bool(result.success),
            "numerically_regular": regular, "boundary": bool(np.any(result.boundary)),
            "boundary_crr": bool(result.boundary[0]), "boundary_em": bool(result.boundary[1]),
            "rank": result.identification_rank, "j_pvalue": result.j_pvalue if regular else np.nan,
            "start_successes": diagnostics["start_successes"], "starts_agree": diagnostics["starts_agree"],
            "unresolved": diagnostics["unresolved"],
            "start_diagnostics_json": json.dumps(_finite_json(diagnostics["start_records"]), allow_nan=False),
            "status": ("optimization_unresolved" if diagnostics["unresolved"] else
                       "boundary" if np.any(result.boundary) else
                       "regular" if regular else "nonregular"),
            "error": "; ".join(diagnostics["start_errors"])}


def run_sw07_finite_sample(*, replications: int = 399, seed: int = 20261001,
                           bandwidth: int = 8, scenarios: Sequence[str] | None = None,
                           progress: Callable[[str, int, int], None] | None = None,
                           checkpoint_dir: str | Path | None = None,
                           resume: bool = True) -> SW07FiniteSampleStudy:
    """Diagnose finite-sample bias, HAC error and bounded estimation behavior.

    Four default scenarios preserve both original calibrations and add their
    observed crr/em fits as explicitly plug-in diagnostic DGPs. Each draw has
    156 stationary Gaussian quarters and its own 15-moment HAC covariance;
    nine moments fit two parameters through all four original starts.
    Profiles/split-sample reports are not repeated per draw. No parameter
    confidence intervals are produced. Failure counts stay in denominators.

    ``scenarios`` may select any of baseline_fixed, published_fixed,
    baseline_fitted, published_fitted. Seeds are invariant to this selection.
    ``progress(scenario, completed, total)`` is called after every draw.
    Small ``replications`` values are useful for smoke tests only.
    Optional checkpoints are replaced atomically after the first draw, every
    ten draws and scenario completion. Resume requires identical settings,
    numerical source files, data and dependency versions. Checkpoints store
    arrays and strict JSON without executable pickle objects.
    """
    replications = _integer(replications, "replications", 2)
    seed = _integer(seed, "seed", 0)
    bandwidth = _integer(bandwidth, "bandwidth", 0)
    if bandwidth >= 152:
        raise ValueError("bandwidth must be less than the 152 common-window observations")
    selected = _SCENARIOS if scenarios is None else tuple(scenarios)
    if not selected or len(set(selected)) != len(selected) or any(s not in _SCENARIOS for s in selected):
        raise ValueError("scenarios must be a nonempty unique selection of " + ", ".join(_SCENARIOS))
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable")
    if not isinstance(resume, (bool, np.bool_)):
        raise ValueError("resume must be boolean")
    checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
    data, provenance = load_empirical_sw07_data()
    observed = covariance_moment_targets(data, ALL_MOMENTS, units=UNITS, bandwidth=bandwidth)
    base = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    calibrations = {"baseline": base, "published": {**base, **_PUBLISHED_MODE}}
    fitters, observed_results, observed_rows = {}, {}, []
    for name in dict.fromkeys(s.split("_")[0] for s in selected):
        fitters[name] = _fitter(calibrations[name])
        fit, diagnostic = fitters[name](observed)
        if diagnostic["unresolved"]:
            raise ValueError(f"Observed {name} optimization is unresolved; cannot anchor simulation comparison")
        observed_results[name] = fit
        observed_rows.append({"calibration": name, **_fit_record(fit, diagnostic)})
    n_total, q = replications*len(selected), len(ALL_MOMENTS)
    all_moments = np.full((n_total, q), np.nan)
    all_covariance = np.full((n_total, q, q), np.nan)
    draw_rows, summary_rows, moment_rows, covariance_rows = [], [], [], []
    scenario_metadata = {}
    resumed_counts = {}
    audit_bandwidths = sorted({4, 8, 12, bandwidth})

    for scenario_number, name in enumerate(selected):
        calibration_name, kind = name.split("_")
        calibration = calibrations[calibration_name]
        fit_observed = observed_results[calibration_name]
        dgp = dict(calibration)
        if kind == "fitted":
            dgp.update(zip(PARAMETER_NAMES, fit_observed.theta.tolist()))
        truth = np.array([dgp[key] for key in PARAMETER_NAMES])
        exact = sw07_finite_sample_moments(dgp, nobs=156, common_max_lag=4)
        known_mean = sw07_finite_sample_moments(dgp, nobs=156, common_max_lag=4, demean=False)
        population = sw07_covariance_moments(dgp, ALL_MOMENTS)
        scenario_metadata[name] = {"scenario_index": _SCENARIOS.index(name), "dgp": dgp,
            "is_plugin": kind == "fitted", "true_parameters": dict(zip(PARAMETER_NAMES, truth.tolist())),
            "observed_objective": fit_observed.objective, "exact_sampling_metadata": dict(exact.metadata),
            "scope": "fixed Gaussian DGP diagnostic; not composite-null p-value or boundary bootstrap inference"}
        means_hac = {choice: np.zeros((q, q)) for choice in audit_bandwidths}
        counts_hac = dict.fromkeys(audit_bandwidths, 0)
        scenario_draws = []
        offset = scenario_number*replications
        checkpoint = None if checkpoint_dir is None else checkpoint_dir/f"{name}.npz"
        if checkpoint is not None:
            design, fingerprint = _checkpoint_design(dgp, name=name, seed=seed, replications=replications,
                bandwidth=bandwidth, source_hash=provenance["source_csv_normalized_sha256"])
            if resume and checkpoint.exists():
                scenario_draws, saved_moments, saved_covariance, means_hac, counts_hac = _load_checkpoint(
                    checkpoint, fingerprint, audit_bandwidths, replications, q)
                for row in scenario_draws:
                    row["array_row"] = offset+row["replication"]
                length = len(scenario_draws)
                all_moments[offset:offset+length] = saved_moments
                all_covariance[offset:offset+length] = saved_covariance
                draw_rows.extend(scenario_draws)
                if progress is not None:
                    progress(name, length, replications)
        resumed_counts[name] = len(scenario_draws)
        for draw in range(len(scenario_draws), replications):
            seed_components = [seed, _SCENARIOS.index(name), draw]
            stream_seed = int(np.random.SeedSequence(seed_components).generate_state(1, dtype=np.uint64)[0])
            row = {"array_row": offset+draw, "scenario": name, "replication": draw,
                   "simulation_seed": str(stream_seed), "true_crr": truth[0], "true_em": truth[1],
                   "status": "failed", "unresolved": True, "error": "",
                   "objective": np.nan, "crr": np.nan, "em": np.nan,
                   "optimizer_success": False, "numerically_regular": False,
                   "boundary": False, "boundary_crr": False, "boundary_em": False,
                   "rank": np.nan, "j_pvalue": np.nan, "start_successes": 0, "starts_agree": False}
            row["start_diagnostics_json"] = "[]"
            try:
                sample = simulate_sw07_sample(dgp, nobs=156, seed=stream_seed)
                row["sample_sha256"] = hashlib.sha256(sample.to_numpy(dtype="<f8").tobytes()).hexdigest()
                target = covariance_moment_targets(sample, ALL_MOMENTS, units=UNITS, bandwidth=bandwidth)
                all_moments[offset+draw], all_covariance[offset+draw] = target.values, target.covariance
                for choice in audit_bandwidths:
                    audit_target = target if choice == bandwidth else covariance_moment_targets(
                        sample, ALL_MOMENTS, units=UNITS, bandwidth=choice)
                    means_hac[choice] += audit_target.covariance
                    counts_hac[choice] += 1
                fit, diagnostic = fitters[calibration_name](target)
                row.update(_fit_record(fit, diagnostic))
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            scenario_draws.append(row)
            draw_rows.append(row)
            if checkpoint is not None and (draw == 0 or (draw+1) % 10 == 0 or draw+1 == replications):
                _save_checkpoint(checkpoint, design, fingerprint, scenario_draws,
                    all_moments[offset:offset+draw+1], all_covariance[offset:offset+draw+1], means_hac, counts_hac)
            if progress is not None:
                progress(name, draw+1, replications)
        draws = pd.DataFrame(scenario_draws)
        usable = ~draws.unresolved & np.isfinite(draws.objective)
        failures = int((~usable).sum())
        tail = int((usable & (draws.objective >= fit_observed.objective)).sum())
        tail_lower, _ = _binomial_interval(tail, replications)
        _, tail_upper = _binomial_interval(tail+failures, replications)
        regular = usable & draws.numerically_regular
        naive_rejections = int((usable & (draws.objective > chi2.ppf(.95, 7))).sum())
        regular_rejections = int((regular & (draws.j_pvalue < .05)).sum())
        naive_mc_lower, _ = _binomial_interval(naive_rejections, replications)
        _, naive_mc_upper = _binomial_interval(naive_rejections+failures, replications)
        regular_mc_lower, regular_mc_upper = _binomial_interval(regular_rejections, int(regular.sum()))
        parameter_errors = draws.loc[usable, ["crr", "em"]].to_numpy()-truth
        summary = {"scenario": name, "is_plugin": kind == "fitted", "replications": replications,
            "usable_fits": int(usable.sum()), "unresolved_draws": failures,
            "boundary_fits": int((usable & draws.boundary).sum()), "numerically_regular_fits": int(regular.sum()),
            "observed_objective": fit_observed.objective, "tail_exceedances": tail,
            "tail_fraction_lower": tail/replications, "tail_fraction_upper": (tail+failures)/replications,
            "tail_mc_95_lower": tail_lower, "tail_mc_95_upper": tail_upper,
            "naive_chi2_rejections": naive_rejections,
            "naive_chi2_rate_lower": naive_rejections/replications,
            "naive_chi2_rate_upper": (naive_rejections+failures)/replications,
            "naive_chi2_mc_95_lower": naive_mc_lower, "naive_chi2_mc_95_upper": naive_mc_upper,
            "regular_j_rejections": regular_rejections,
            "regular_j_rate_conditional": regular_rejections/int(regular.sum()) if regular.any() else np.nan,
            "regular_j_mc_95_lower_conditional": regular_mc_lower,
            "regular_j_mc_95_upper_conditional": regular_mc_upper,
            "criterion_median": float(draws.loc[usable, "objective"].median()) if usable.any() else np.nan,
            "criterion_q95": float(draws.loc[usable, "objective"].quantile(.95)) if usable.any() else np.nan}
        for j, parameter in enumerate(PARAMETER_NAMES):
            summary[f"true_{parameter}"] = truth[j]
            summary[f"bias_{parameter}_usable"] = float(parameter_errors[:, j].mean()) if usable.any() else np.nan
            summary[f"rmse_{parameter}_usable"] = float(np.sqrt(np.mean(parameter_errors[:, j]**2))) if usable.any() else np.nan
        summary_rows.append(summary)
        raw = all_moments[offset:offset+replications]
        finite = np.isfinite(raw).all(axis=1)
        empirical_cov = np.cov(raw[finite], rowvar=False) if finite.sum() > 1 else np.full((q, q), np.nan)
        empirical_mean = raw[finite].mean(axis=0) if finite.any() else np.full(q, np.nan)
        exact_se = np.sqrt(np.diag(exact.covariance))
        for j, label in enumerate(exact.labels):
            moment_row = {"scenario": name, "label": label, "used_in_fit": j < 9,
                "observed": observed.values[j], "population": population.iloc[j],
                "exact_sample_mean": exact.values[j], "known_mean_expectation": known_mean.values[j],
                "demeaning_bias": exact.values[j]-population.iloc[j],
                "exact_sampling_se": exact_se[j], "observed_minus_exact_in_se": (observed.values[j]-exact.values[j])/exact_se[j],
                "monte_carlo_mean": empirical_mean[j], "mean_mc_se": exact_se[j]/np.sqrt(max(1, finite.sum())),
                "moment_draws": int(finite.sum()), "empirical_variance_ratio": empirical_cov[j, j]/exact.covariance[j, j]}
            for choice in audit_bandwidths:
                moment_row[f"hac{choice}_draws"] = counts_hac[choice]
                moment_row[f"hac{choice}_variance_ratio"] = (means_hac[choice][j, j]/counts_hac[choice]/exact.covariance[j, j]
                                                           if counts_hac[choice] else np.nan)
            moment_rows.append(moment_row)
        matrices = {"empirical_mc": empirical_cov, "exact_known_mean": known_mean.covariance,
                    **{f"hac{choice}": means_hac[choice]/counts_hac[choice] if counts_hac[choice]
                       else np.full((q, q), np.nan) for choice in audit_bandwidths}}
        for estimator, matrix in matrices.items():
            for j, left in enumerate(exact.labels):
                for k, right in enumerate(exact.labels):
                    covariance_rows.append({"scenario": name, "estimator": estimator,
                        "draws": counts_hac[int(estimator[3:])] if estimator.startswith("hac")
                                 else int(finite.sum()) if estimator == "empirical_mc" else 0,
                        "left": left, "right": right, "covariance": matrix[j, k],
                        "exact_demeaned_covariance": exact.covariance[j, k]})

    metadata = {"study": "SW07 finite-sample conditional Gaussian diagnosis", "source": provenance,
        "checkpointing": {"enabled": checkpoint_dir is not None, "resumed_draws": resumed_counts,
                          "interval": "first, every10, final; atomic NPZ/JSON, no pickle"},
        "replications_per_scenario": replications, "master_seed": seed, "scenarios": scenario_metadata,
        "seed_rule": "uint64 first generated state of SeedSequence([master_seed, fixed scenario index, zero-based replication])",
        "all_scenario_indices": dict(zip(_SCENARIOS, range(len(_SCENARIOS)))),
        "observations": 156, "common_max_lag": 4, "effective_observations": 152,
        "moment_labels": list(observed.labels), "fit_moment_count": 9, "reserved_moment_count": 6,
        "bandwidth": bandwidth, "audited_hac_bandwidths": audit_bandwidths,
        "starts": _STARTS, "bounds": PARAMETER_BOUNDS, "optimizer_tolerance": 1e-9, "max_nfev": 500,
        "fitting_stages": "full15 sample covariance/HAC, select9, all4 original bounded minimum-distance starts, local diagnostics,6reserved moments; no per-draw profiles or historical splits",
        "parameter_inference_reported": False,
        "tail_interpretation": "conditional fixed-DGP diagnostic fractions; no composite-null or boundary-bootstrap validity claimed",
        "monte_carlo_uncertainty": "95% Clopper-Pearson binomial intervals; unresolved draws included as both none and all exceedances, never silently discarded",
        "naive_chi2_scope": "deliberately naive chi2(7) rule on usable fits including boundary cases; diagnostic, not valid boundary inference",
        "regular_j_scope": "numerical-regular subset only; selected denominator reported, not general coverage or size proof",
        "parameter_bias_scope": "conditional on usable optimization; failure counts remain explicit",
        "moment_audit_scope": "available empirical moment draws, independently of optimizer success; counts retained",
        "small_replication_warning": replications < 199,
        "limitations": ["Gaussian stationary state-space DGP conditional on fixed calibration", "revised data and possible historical regime instability",
                        "plug-in scenarios do not supply composite-null p-values", "no calibration uncertainty or strong-identification proof",
                        "finite Monte Carlo precision; no normal parameter intervals restored"],
        "references": ["https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp722.pdf", "https://www.nber.org/papers/t0055",
                       "https://jeanmariedufour.research.mcgill.ca/Dufour_2006_JE_MCT.pdf"]}
    return SW07FiniteSampleStudy(pd.DataFrame(summary_rows), pd.DataFrame(draw_rows),
                                pd.DataFrame(moment_rows), pd.DataFrame(covariance_rows),
                                pd.DataFrame(observed_rows), all_moments, all_covariance, metadata)
