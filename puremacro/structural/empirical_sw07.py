"""Observed-data, conditional SW07 covariance matching.

This is a new minimum-distance study on a revised FRED vintage, not a
replication of the SW07 Bayesian estimates. Central second moments fit
policy smoothing and monetary innovation volatility conditional on the
remaining model calibration and its mutually orthogonal latent innovations.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib.resources
import io
from numbers import Real
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize_scalar

from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
from puremacro.dsge.sw07_observation import OBSERVED_VARS, make_state_space
from puremacro.structural.bridge import MomentTargets, StructuralFitResult, fit_structural

SERIES = ("gdp_growth", "infl", "ffr")
UNITS = {"gdp_growth": "100*dlog real per-capita GDP per quarter",
         "infl": "100*dlog GDP deflator per quarter",
         "ffr": "federal funds annual percentage rate / 4"}
FIT_MOMENTS = tuple((a, b, 0) for i, a in enumerate(SERIES) for b in SERIES[i:]) + tuple(
    (a, a, 1) for a in SERIES)
HELD_OUT_MOMENTS = tuple((a, a, h) for h in (2, 4) for a in SERIES)
ALL_MOMENTS = FIT_MOMENTS + HELD_OUT_MOMENTS
PARAMETER_NAMES = ("crr", "em")
PARAMETER_BOUNDS = ((.2, .98), (.01, 1.0))
_SOURCE_NORMALIZED_SHA256 = "8f8e9c123d01a9d9b51ce128fce2fa0cf5d65e1add2bda22522fbdf05717c5e9"
# Published MODE (not posterior mean), ECB WP722 Tables 1a/b. Kept here
# explicitly rather than making an estimator depend on the replication runner.
_PUBLISHED_MODE = {
    "csadjcost": 5.48, "csigma": 1.39, "chabb": .71, "cprobw": .73,
    "csigl": 1.92, "cprobp": .65, "cindw": .59, "cindp": .22,
    "czcap": .54, "cfc": 1.61, "crpi": 2.03, "crr": .81,
    "cry": .08, "crdy": .22, "constepinf": .81, "constebeta": .16,
    "ctrend": .43, "calfa": .19, "crhoa": .95, "crhob": .18,
    "crhog": .97, "crhoqs": .71, "crhoms": .12, "crhopinf": .90,
    "crhow": .97, "cmap": .74, "cmaw": .88, "cgy": .52,
    "ea": .45, "eb": .24, "eg": .52, "eqs": .45, "em": .24,
    "epinf": .14, "ew": .24,
}


def _specifications(moments: Sequence[tuple[str, str, int]]) -> tuple:
    specs = tuple(moments)
    if not specs:
        raise ValueError("at least one covariance moment is required")
    for spec in specs:
        if (not isinstance(spec, (tuple, list)) or len(spec) != 3
                or not all(isinstance(x, str) and x for x in spec[:2])
                or isinstance(spec[2], (bool, np.bool_))
                or not isinstance(spec[2], (int, np.integer)) or spec[2] < 0):
            raise ValueError("moments must contain (left column, right column, nonnegative integer lag)")
    specs = tuple((a, b, int(h)) for a, b, h in specs)
    if len(set(specs)) != len(specs):
        raise ValueError("covariance moment specifications must be unique")
    return specs


def _labels(specs: tuple) -> tuple[str, ...]:
    return tuple(f"cov({a},{b};lag={h})" for a, b, h in specs)


def covariance_moment_targets(
    data: pd.DataFrame, moments: Sequence[tuple[str, str, int]], *,
    units: Mapping[str, str], bandwidth: int = 8,
    metadata: Mapping[str, Any] | None = None,
) -> MomentTargets:
    """Central covariances and full joint Bartlett-HAC estimator covariance.

    A moment (a,b,h) is Cov(a[t], b[t-h]). Every product uses the same
    t=max(lags),...,n-1 window and the series' full-sample estimated means.
    Scores are product minus its sample average. The covariance of their
    average is score' K score / n_effective**2, with Bartlett K and no
    degrees-of-freedom correction or ridge. Estimation of a stationary
    unconditional mean has zero *population* first-order derivative here;
    this is not an exact finite-sample correction. Consistency requires
    stationary weak dependence, finite fourth moments and a suitable HAC
    bandwidth sequence, not merely a positive covariance matrix.
    """
    specs = _specifications(moments)
    if not isinstance(data, pd.DataFrame) or data.columns.has_duplicates:
        raise ValueError("data must be a DataFrame with unique columns")
    columns = tuple(dict.fromkeys(x for spec in specs for x in spec[:2]))
    if any(x not in data for x in columns):
        raise ValueError("data are missing a requested covariance column")
    if any(x not in units or not isinstance(units[x], str) or not units[x] for x in columns):
        raise ValueError("explicit nonempty units are required for every covariance column")
    index = data.index
    if isinstance(index, pd.DatetimeIndex):
        quarters = index.to_period("Q")
    elif isinstance(index, pd.PeriodIndex) and str(index.freqstr).startswith("Q"):
        quarters = index
    else:
        raise ValueError("data require a regular quarterly DatetimeIndex or PeriodIndex")
    if (index.has_duplicates or not index.is_monotonic_increasing
            or quarters.has_duplicates or quarters.hasnans
            or (len(index) > 1 and not np.all(np.diff(quarters.asi8) == 1))):
        raise ValueError("data must contain consecutive unique quarters without gaps")
    if isinstance(index, pd.DatetimeIndex) and len(index) > 2 and pd.infer_freq(index) is None:
        raise ValueError("DatetimeIndex must have regularly spaced quarterly dates")
    raw_values = data.loc[:, list(columns)].to_numpy()
    if np.iscomplexobj(raw_values):
        raise ValueError("covariance data must be real, not complex")
    values = np.asarray(raw_values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("covariance data must be finite; missing rows cannot be silently dropped")
    largest_lag = max(h for _, _, h in specs)
    n = len(data) - largest_lag
    if n < max(20, len(specs) + 2):
        raise ValueError("too few common-window observations for the covariance moments")
    if (isinstance(bandwidth, (bool, np.bool_)) or not isinstance(bandwidth, (int, np.integer))
            or bandwidth < 0 or bandwidth >= n):
        raise ValueError("bandwidth must be an integer between zero and n_effective-1")
    centered = values - values.mean(axis=0)
    positions = {column: j for j, column in enumerate(columns)}
    products = np.column_stack([
        centered[largest_lag:, positions[a]] * centered[largest_lag-h:len(data)-h, positions[b]]
        for a, b, h in specs])
    estimates = products.mean(axis=0)
    scores = products - estimates
    meat = scores.T @ scores
    for lag in range(1, bandwidth + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (bandwidth + 1)) * (cross + cross.T)
    covariance = (meat + meat.T) / (2 * n**2)
    meta = dict(metadata or {})
    meta.update({"estimator": "central covariance on a common time window",
                 "sample_start": str(quarters[0]), "sample_end": str(quarters[-1]),
                 "product_window_start": str(quarters[largest_lag]),
                 "observations": len(data), "effective_observations": n,
                 "frequency": "Q", "bandwidth": int(bandwidth), "kernel": "Bartlett",
                 "sample_means": dict(zip(columns, values.mean(axis=0).tolist())),
                 "mean_treatment": "full-sample estimated means; zero population first-order derivative under stationarity, not exact finite-sample mean adjustment",
                 "covariance_scale": "covariance of empirical moment estimators, not long-run score covariance",
                 "sampling_assumptions": "stationary weak dependence, finite fourth moments and appropriate HAC bandwidth; regime stability is an assumption",
                 "moment_orientation": "Cov(left_t, right_t-lag)",
                 "common_max_lag": largest_lag})
    return MomentTargets(estimates, covariance, _labels(specs),
                         tuple(f"({units[a]}) * ({units[b]})" for a, b, _ in specs), meta)


def sw07_covariance_moments(
    params: Mapping[str, float], moments: Sequence[tuple[str, str, int]] = ALL_MOMENTS,
) -> pd.Series:
    """Exact stationary model covariances from a discrete Lyapunov solve.

    Defaults are the bundled Pfeifer model calibration, not an estimated
    posterior mode despite the historical SW07_POSTERIOR_MODE constant name.
    The Kalman implementation's 1e-8 numerical H ridge is deliberately
    excluded: this study assumes no economic measurement error. Means and
    deterministic observation intercepts do not enter central covariances.
    """
    specs = _specifications(moments)
    calibration = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    if not isinstance(params, Mapping):
        raise ValueError("SW07 parameters must be a mapping of real scalars")
    if set(params) - set(calibration):
        raise ValueError(f"unknown SW07 parameters: {sorted(set(params) - set(calibration))}")
    calibration.update(params)
    if not all(isinstance(v, Real) and not isinstance(v, (bool, np.bool_))
               and np.isfinite(v) for v in calibration.values()):
        raise ValueError("SW07 parameters must be finite real scalars")
    if any(calibration[name] < 0 for name in SW07_SHOCK_STDS):
        raise ValueError("shock standard deviations must be nonnegative")
    if any(a not in OBSERVED_VARS or b not in OBSERVED_VARS for a, b, _ in specs):
        raise ValueError("moment variables must be SW07 observed variable names")
    # make_state_space delegates to the same QZ solution used by the audited
    # SW07 observation implementation. Check BK separately: a stable-looking
    # matrix is insufficient if the rational-expectations equilibrium is not unique.
    from puremacro.dsge.smets_wouters import solve_sw07
    solution = solve_sw07(calibration)
    if solution.eu != (1, 1):
        raise np.linalg.LinAlgError(f"SW07 Blanchard-Kahn condition failed: {solution.eu}")
    model = make_state_space(calibration)
    transition, observation = model.T, model.Z
    radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if not np.isfinite(radius) or radius >= 1 - 1e-10:
        raise np.linalg.LinAlgError(f"SW07 covariance requires stationary transition; spectral radius={radius}")
    innovation = model.R @ model.Q @ model.R.T
    state_covariance = solve_discrete_lyapunov(transition, innovation)
    state_covariance = (state_covariance + state_covariance.T) / 2
    residual = state_covariance - transition @ state_covariance @ transition.T - innovation
    scale = max(1., float(np.linalg.norm(state_covariance, ord="fro")))
    if (not np.all(np.isfinite(state_covariance))
            or np.linalg.norm(residual, ord="fro") > 1e-9 * scale
            or np.min(np.linalg.eigvalsh(state_covariance)) < -1e-9 * scale):
        raise np.linalg.LinAlgError("SW07 Lyapunov covariance failed residual/PSD validation")
    positions = {name: i for i, name in enumerate(OBSERVED_VARS)}
    covariances = {h: observation @ np.linalg.matrix_power(transition, h) @ state_covariance @ observation.T
                   for h in {h for _, _, h in specs}}
    result = pd.Series([covariances[h][positions[a], positions[b]] for a, b, h in specs],
                       index=_labels(specs), dtype=float)
    result.attrs.update({"transition_spectral_radius": radius,
                         "lyapunov_relative_residual": float(np.linalg.norm(residual, ord="fro") / scale),
                         "measurement_error_included": False,
                         "moment_orientation": "Cov(left_t, right_t-lag)"})
    return result


def load_empirical_sw07_data() -> tuple[pd.DataFrame, dict]:
    """Load the fixed bundled revised-FRED vintage and its exact provenance."""
    resource = importlib.resources.files("puremacro.dsge") / "_sw07_data.csv"
    payload = resource.read_bytes()
    normalized_sha256 = hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest()
    if normalized_sha256 != _SOURCE_NORMALIZED_SHA256:
        raise ValueError(
            "bundled SW07 source does not match the authenticated 2026-09-30 snapshot; "
            "rebuild with tools/build_sw07_data.py, review sources/transforms/sample and "
            "update this study's frozen hash and provenance deliberately before using a new vintage")
    text = payload.decode("utf-8")
    data = pd.read_csv(io.StringIO(text), comment="#", index_col="date")
    data.index = pd.PeriodIndex(data.index, freq="Q", name="date")
    expected_index = pd.period_range("1966Q1", "2004Q4", freq="Q", name="date")
    if (tuple(data.columns) != OBSERVED_VARS or not data.index.equals(expected_index)
            or data.shape != (156, 7) or np.iscomplexobj(data.to_numpy())
            or not np.all(np.isfinite(data.to_numpy(dtype=float)))):
        raise ValueError("bundled SW07 snapshot must contain the finite seven-variable 1966Q1-2004Q4 sample")
    fred_series = ("GDPC1", "PCEC", "FPI", "GDPDEF", "PRS85006023", "CE16OV", "COMPNFB", "FEDFUNDS", "CNP16OV")
    provenance = {
        "data_kind": "observed US macroeconomic data", "is_synthetic": False,
        "vintage": "revised FRED snapshot built 2026-09-30; not original SW07 estimation vintage",
        "source_csv_sha256": hashlib.sha256(payload).hexdigest(),
        "source_csv_normalized_sha256": normalized_sha256,
        "source_authentication": "pinned reviewed SHA256 after CRLF normalization, exact seven columns and complete 156-quarter sample",
        "source_csv": "puremacro/dsge/_sw07_data.csv",
        "source_header": [line for line in text.splitlines() if line.startswith("#")],
        "sources": [f"https://fred.stlouisfed.org/series/{name}" for name in fred_series],
        "data_definition_reference": "https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp722.pdf",
        "published_estimation_replication": False,
    }
    return data, provenance


@dataclass(frozen=True)
class SW07MomentStudy:
    """Study results with conservative, explicitly conditional inference gates."""
    fit: StructuralFitResult
    targets: MomentTargets
    multistart: pd.DataFrame
    profiles: pd.DataFrame
    bandwidth_sensitivity: pd.DataFrame
    sample_split: pd.DataFrame
    calibration_sensitivity: pd.DataFrame
    calibration_sensitivity_moments: pd.DataFrame
    metadata: Mapping[str, Any]


def fit_empirical_sw07(*, bandwidth: int = 8, profile_points: int = 11,
                       allow_conditional_inference: bool = False) -> SW07MomentStudy:
    """Run a reproducible two-parameter study on the bundled observed sample.

    All other parameters and independent latent-shock variances are fixed.
    No high-frequency or narrative shock instrument enters this exercise.
    Regular normal inference is withheld by default. Opting in still requires
    interior numerical identification, agreement across starts, nonflat
    descriptive profiles and a nonrejected correct-specification J diagnostic.
    Passing J does not establish correct specification. Profiles and sample
    splits are diagnostics, not weak-identification-robust confidence sets.
    """
    if (isinstance(profile_points, bool) or not isinstance(profile_points, (int, np.integer))
            or profile_points < 5):
        raise ValueError("profile_points must be an integer of at least five")
    if not isinstance(allow_conditional_inference, (bool, np.bool_)):
        raise ValueError("allow_conditional_inference must be boolean")
    data, provenance = load_empirical_sw07_data()
    calibration = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    all_targets = covariance_moment_targets(data, ALL_MOMENTS, units=UNITS,
                                            bandwidth=bandwidth, metadata=provenance)
    fitted = all_targets.select(_labels(FIT_MOMENTS))
    held = all_targets.select(_labels(HELD_OUT_MOMENTS))

    def moments(theta, specs=FIT_MOMENTS, calibration_override=None):
        parameters = dict(calibration if calibration_override is None else calibration_override)
        parameters.update(zip(PARAMETER_NAMES, theta))
        return sw07_covariance_moments(parameters, specs)

    def solve(target, start, include_held=False, calibration_override=None):
        with warnings.catch_warnings():
            # Reasons are retained in the result/tables; the study reports them
            # together instead of emitting the same boundary warning per start.
            warnings.filterwarnings("ignore", message="Structural local inference unavailable:")
            return fit_structural(
                lambda theta: moments(theta, calibration_override=calibration_override),
                target, start, parameter_names=PARAMETER_NAMES,
                bounds=PARAMETER_BOUNDS, moment_labels=target.labels,
                held_out=held if include_held else None,
                held_out_moments_at=(lambda theta: moments(theta, HELD_OUT_MOMENTS, calibration_override)) if include_held else None,
                assume_correct_specification=True, tol=1e-9, max_nfev=500)

    starts = ((.5, .1), (.75, .4), (.95, .7), (calibration["crr"], calibration["em"]))
    fits = [solve(fitted, start, include_held=True) for start in starts]
    successful = [fit for fit in fits if fit.success]
    best = min(successful or fits, key=lambda fit: fit.objective)
    multistart = pd.DataFrame([
        {"start_crr": start[0], "start_em": start[1], "crr": fit.theta[0], "em": fit.theta[1],
         "objective": fit.objective, "optimizer_success": fit.success,
         "numerical_inference_regular": fit.inference_valid,
         "boundary": bool(np.any(fit.boundary)), "message": fit.message}
        for start, fit in zip(starts, fits)])
    whitening = np.linalg.cholesky(fitted.covariance)

    def objective(theta):
        residual = np.linalg.solve(whitening, moments(theta).to_numpy() - fitted.values)
        return float(residual @ residual)

    profiles = []
    flat_parameters = []
    for j, parameter in enumerate(PARAMETER_NAMES):
        other = 1 - j
        grid = np.unique(np.r_[np.linspace(*PARAMETER_BOUNDS[j], profile_points), best.theta[j]])
        rows = []
        for value in grid:
            def at_nuisance(nuisance):
                theta = best.theta.copy()
                theta[j], theta[other] = value, nuisance
                return objective(theta)
            trial = minimize_scalar(at_nuisance, bounds=PARAMETER_BOUNDS[other], method="bounded",
                                    options={"xatol": 1e-7, "maxiter": 200})
            # A bounded scalar routine may not evaluate endpoints; retain them
            # explicitly so a boundary nuisance solution is represented honestly.
            candidates = [(trial.fun, trial.x)] + [(at_nuisance(x), x) for x in PARAMETER_BOUNDS[other]]
            loss, nuisance = min(candidates)
            rows.append({"parameter": parameter, "value": float(value), "nuisance_value": float(nuisance),
                         "objective": float(loss), "optimizer_success": bool(trial.success),
                         "delta_objective": float(loss - best.objective)})
        accepted = [row["value"] for row in rows if row["delta_objective"] <= 1.0]
        if accepted and np.ptp(accepted) >= .8 * np.ptp(PARAMETER_BOUNDS[j]):
            flat_parameters.append(parameter)
        profiles.extend(rows)
    profile_frame = pd.DataFrame(profiles)
    disagreement = bool(any(abs(fit.objective-best.objective) > 1e-5 * max(1., best.objective)
                            for fit in successful))
    profile_better = bool(profile_frame["delta_objective"].min() < -1e-5 * max(1., best.objective))
    rejected = bool(np.isfinite(best.j_pvalue) and best.j_pvalue < .05)
    reasons = list(best.diagnostics["inference_unavailable_reasons"])
    if not allow_conditional_inference:
        reasons.append("correct_specification_inference_not_requested")
    if rejected:
        reasons.append("correct_specification_null_rejected_at_0.05")
    if disagreement or profile_better:
        reasons.append("optimization_disagreement")
    if flat_parameters:
        reasons.append("flat_descriptive_profile")
    profile_failed = not bool(profile_frame["optimizer_success"].all())
    if profile_failed:
        reasons.append("profile_optimization_failed")
    numerical_reasons = list(best.diagnostics["inference_unavailable_reasons"])
    status = ("nonregular_fit" if numerical_reasons else "rejected_model" if rejected
              else "optimization_unresolved" if disagreement or profile_better or profile_failed
              else "flat_profile" if flat_parameters else "conditional_fit_not_rejected")
    diagnostics = dict(best.diagnostics)
    diagnostics.update({"inference_unavailable_reasons": tuple(dict.fromkeys(reasons)),
                        "study_status": status, "reported_inference_is_conditional": True,
                        "study_inference_gate": "J nonrejection does not validate correct-specification inference; explicit opt-in required"})
    if reasons:
        best = replace(best, covariance=np.full((2, 2), np.nan), standard_errors=np.full(2, np.nan),
                       inference_valid=False, diagnostics=diagnostics)
    else:
        best = replace(best, diagnostics=diagnostics)

    sensitivity = []
    for choice in sorted({4, int(bandwidth), 8, 12}):
        target = covariance_moment_targets(data, ALL_MOMENTS, units=UNITS, bandwidth=choice).select(fitted.labels)
        fit = solve(target, best.theta)
        sensitivity.append({"bandwidth": choice, "crr": fit.theta[0], "em": fit.theta[1],
                            "objective": fit.objective, "optimizer_success": fit.success,
                            "boundary": bool(np.any(fit.boundary)), "conditional_null_j_pvalue": fit.j_pvalue})
    splits = []
    for name, sample in (("1966Q1-1979Q2", data.loc[:"1979Q2"]),
                         ("1984Q1-2004Q4", data.loc["1984Q1":])):
        target = covariance_moment_targets(sample, ALL_MOMENTS, units=UNITS, bandwidth=min(bandwidth, 8))
        frame = target.to_frame()
        frame["sample"] = name
        splits.append(frame)
    # Exploratory sensitivity added after observing the baseline boundary fit.
    # The published calibration was estimated on the original version of much
    # of the same historical sample. It is not independent validation evidence.
    published_calibration = {**calibration, **_PUBLISHED_MODE}
    published_fits = [solve(fitted, start, include_held=True,
                           calibration_override=published_calibration) for start in starts]
    published_successful = [fit for fit in published_fits if fit.success]
    published_best = min(published_successful or published_fits, key=lambda fit: fit.objective)
    calibration_rows, calibration_frames = [], []
    for name, fit in (("declared_pfeifer_baseline", best),
                      ("exploratory_published_sw07_mode", published_best)):
        variant_status = ("nonregular_fit" if np.any(fit.boundary) or not np.isfinite(fit.j_pvalue)
                          else "rejected_model" if fit.j_pvalue < .05 else "conditional_fit_not_rejected")
        calibration_rows.append({"calibration": name, "crr": fit.theta[0], "em": fit.theta[1],
                                 "objective": fit.objective, "optimizer_success": fit.success,
                                 "boundary": bool(np.any(fit.boundary)),
                                 "status": variant_status,
                                 "conditional_null_j_pvalue": fit.j_pvalue})
        for reserved in (False, True):
            frame = fit.moment_fit(held_out=reserved)
            frame["calibration"] = name
            frame["residual_over_target_se"] = frame["residual"] / frame["target_se"]
            calibration_frames.append(frame)
    population_diagnostics = dict(moments(best.theta).attrs)
    metadata = {**provenance, "study": "conditional SW07 observed covariance matching",
                "status": status, "estimated_parameter_names": list(PARAMETER_NAMES),
                "parameter_interpretation": {"crr": "Taylor-rule interest-rate smoothing", "em": "monetary innovation standard deviation in quarterly percentage-point model units"},
                "parameter_bounds": dict(zip(PARAMETER_NAMES, PARAMETER_BOUNDS)),
                "fixed_calibration": {key: value for key, value in calibration.items() if key not in PARAMETER_NAMES},
                "calibration_description": "bundled Pfeifer model calibration and shock SDs; historical constant name is not an original SW07 posterior-mode estimate",
                "calibration_reference": "https://github.com/JohannesPfeifer/DSGE_mod/blob/master/Smets_Wouters_2007/Smets_Wouters_2007.mod",
                "observation_equations": {"gdp_growth": "y_t - y_lag_t + ctrend", "infl": "pinf_t + constepinf", "ffr": "r_t + conster"},
                "measurement_error": "zero; excluded the Kalman implementation's 1e-8 numerical H ridge",
                "population_diagnostics": population_diagnostics,
                "population_moments": "Gamma_h=Z T^h P Z'; P=T P T'+R Q R'; exact discrete Lyapunov solution",
                "sampling_covariance": "full joint Bartlett HAC of common-window centered products; no ridge",
                "inference_references": ["https://www.nber.org/papers/t0055",
                    "https://larspeterhansen.org/lph_research/large-sample-properties-of-generalized-method-of-moments-estimators/"],
                "bandwidth": int(bandwidth), "allow_conditional_inference": bool(allow_conditional_inference),
                "correct_specification_null_rejected": rejected,
                "inference_unavailable_reasons": list(dict.fromkeys(reasons)),
                "conditional_j_test_note": "available only for regular interior optimal minimum-distance fits under stationary correct-specification null; nonrejection does not establish specification",
                "profile_diagnostic": "nuisance-reoptimized bounded objective slices; flat if delta<=1 spans 80% of bounds; not confidence sets or a formal weak-ID test",
                "flat_profile_parameters": flat_parameters,
                "held_out_scope": "same-sample unused moments; descriptive only, neither independent validation nor out-of-sample forecasting",
                "identification_scope": "conditional model identification from unconditional covariances; latent monetary innovations are not externally identified empirical shocks",
                "limitations": ["fixed calibration uncertainty omitted", "revised vintage differs from original paper", "stationarity and regime stability assumed", "finite-sample HAC and persistent-series demeaning can be inaccurate", "conditional correct-specification SEs are not misspecification-robust", "no proof of global or strong statistical identification"],
                "sample_split_scope": "descriptive covariance comparison only; not a formal structural-break test"}
    metadata["calibration_sensitivity"] = {
        "design_status": "exploratory sensitivity added after the declared baseline hit its bounds; baseline result remains unchanged",
        "source": "Smets-Wouters (2007), ECB WP722 Tables 1a/b, posterior MODE columns",
        "source_url": "https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp722.pdf",
        "published_mode_values": dict(_PUBLISHED_MODE),
        "additional_fixed_values": {key: value for key, value in calibration.items() if key not in _PUBLISHED_MODE},
        "fixed_calibration": {key: value for key, value in published_calibration.items() if key not in PARAMETER_NAMES},
        "multistart_estimates": [{"theta": fit.theta.tolist(), "objective": fit.objective,
                                  "success": fit.success} for fit in published_fits],
        "population_diagnostics": dict(moments(published_best.theta, calibration_override=published_calibration).attrs),
        "scope": "same moments, data, bounds and optimal weights; reestimate only crr/em; no ordinary SEs reported; no profile-based inference for this exploratory variant",
        "limitations": "published calibration estimated on the original vintage of an overlapping historical sample; not independent validation; rounded published values; fixed calibration uncertainty omitted",
    }
    return SW07MomentStudy(best, all_targets, multistart, profile_frame,
                           pd.DataFrame(sensitivity), pd.concat(splits, ignore_index=True),
                           pd.DataFrame(calibration_rows), pd.concat(calibration_frames, ignore_index=True), metadata)


__all__ = ["SW07MomentStudy", "covariance_moment_targets", "sw07_covariance_moments",
           "load_empirical_sw07_data", "fit_empirical_sw07"]
