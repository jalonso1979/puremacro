"""Exact finite-sample Gaussian covariance moments and stationary SW07 draws.

These are conditional model diagnostics. Gaussian observations do not imply
Gaussian covariance estimators, and these oracles do not supply composite-null
p-values, boundary inference or uncertainty about a fixed calibration.
"""
from __future__ import annotations

from numbers import Real
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov

from puremacro.dsge.smets_wouters import (
    SW07_POSTERIOR_MODE, SW07_SHOCK_STDS, solve_sw07,
)
from puremacro.dsge.sw07_observation import OBSERVED_VARS, make_state_space
from puremacro.structural.bridge import MomentTargets
from puremacro.structural.empirical_sw07 import ALL_MOMENTS, UNITS, _labels, _specifications

_OBSERVATION_UNITS = {
    **UNITS,
    "cons_growth": "100*dlog real per-capita consumption per quarter",
    "inv_growth": "100*dlog real per-capita investment per quarter",
    "wage_growth": "100*dlog real wage per quarter",
    "log_hours": "100*log per-capita hours deviation",
}


def _positive_integer(value, name, minimum=1):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < minimum):
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def _psd_root(matrix: np.ndarray, name: str) -> tuple[np.ndarray, dict]:
    """Factor semidefinite covariance, clipping only eigenvalue roundoff."""
    matrix = np.asarray(matrix)
    if (np.iscomplexobj(matrix) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not np.all(np.isfinite(matrix))):
        raise np.linalg.LinAlgError(f"{name} must be a finite real square covariance")
    matrix = np.asarray(matrix, dtype=float)
    scale = float(np.max(np.abs(matrix)))
    if scale == 0:
        return np.zeros_like(matrix), {
            "negative_eigenvalues_clipped_for_roundoff": 0, "minimum_eigenvalue": 0.,
            "roundoff_tolerance": 0., "factor_relative_residual": 0.}
    scaled = matrix / scale
    tolerance = 64 * np.finfo(float).eps * len(matrix)
    if np.linalg.norm(scaled - scaled.T, ord="fro") > tolerance:
        raise np.linalg.LinAlgError(f"{name} must be symmetric")
    symmetric = .5 * (scaled + scaled.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] < -tolerance:
        raise np.linalg.LinAlgError(f"{name} is not positive semidefinite; no regularization is applied")
    scaled_factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.))
    reconstruction_error = float(np.linalg.norm(scaled_factor @ scaled_factor.T - symmetric, ord="fro"))
    if not np.isfinite(reconstruction_error) or reconstruction_error > tolerance:
        raise np.linalg.LinAlgError(f"{name} square-root reconstruction failed")
    factor = scaled_factor * np.sqrt(scale)
    return factor, {"negative_eigenvalues_clipped_for_roundoff": int(np.sum(eigenvalues < 0)),
                    "minimum_eigenvalue": float(eigenvalues[0] * scale),
                    "roundoff_tolerance": float(tolerance * scale),
                    "factor_relative_residual": reconstruction_error}


def _stationary_system(params: Mapping[str, float]):
    """Return (state-space model, stationary state covariance, diagnostics)."""
    calibration = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    if not isinstance(params, Mapping):
        raise ValueError("SW07 parameters must be a mapping of real scalars")
    if set(params) - set(calibration):
        raise ValueError(f"unknown SW07 parameters: {sorted(set(params) - set(calibration))}")
    calibration.update(params)
    if not all(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
               and np.isfinite(value) for value in calibration.values()):
        raise ValueError("SW07 parameters must be finite real scalars")
    if any(calibration[name] < 0 for name in SW07_SHOCK_STDS):
        raise ValueError("shock standard deviations must be nonnegative")
    solution = solve_sw07(calibration)
    if solution.eu != (1, 1):
        raise np.linalg.LinAlgError(f"SW07 Blanchard-Kahn condition failed: {solution.eu}")
    model = make_state_space(calibration)
    radius = float(np.max(np.abs(np.linalg.eigvals(model.T))))
    if not np.isfinite(radius) or radius >= 1 - 1e-10:
        raise np.linalg.LinAlgError(f"SW07 sampling requires stationarity; spectral radius={radius}")
    if np.any(model.c != 0):
        raise np.linalg.LinAlgError("SW07 sampling requires zero state intercept")
    innovation = model.R @ model.Q @ model.R.T
    covariance = solve_discrete_lyapunov(model.T, innovation)
    covariance = .5 * (covariance + covariance.T)
    scale = max(1., float(np.linalg.norm(covariance, ord="fro")))
    residual = covariance - model.T @ covariance @ model.T.T - innovation
    relative_error = float(np.linalg.norm(residual, ord="fro") / scale)
    if not np.all(np.isfinite(covariance)) or relative_error > 1e-9:
        raise np.linalg.LinAlgError("SW07 stationary Lyapunov covariance failed residual validation")
    _, factor_diagnostics = _psd_root(covariance, "SW07 stationary covariance")
    return model, covariance, {
        "parameters": dict(calibration), "transition_spectral_radius": radius,
        "lyapunov_relative_residual": relative_error,
        "stationary_covariance_factor": factor_diagnostics,
        "measurement_error_included": False,
        "measurement_error_note": "economic H=0; Kalman numerical ridge excluded",
    }


def sw07_finite_sample_moments(
    params: Mapping[str, float], moments: Sequence[tuple[str, str, int]] = ALL_MOMENTS,
    *, nobs: int = 156, common_max_lag: int | None = None, demean: bool = True,
) -> MomentTargets:
    """Exact E and joint Cov of finite-sample covariance estimators.

    A moment (a,b,h) averages (a[t]-mean(a))*(b[t-h]-mean(b)) for
    t=common_max_lag,...,nobs-1. ``demean=True`` uses each series' mean over
    all nobs observations, exactly as ``covariance_moment_targets`` does.
    ``demean=False`` removes the *known population model mean*, a diagnostic
    control; it does not compute raw products including observation intercepts.

    ``common_max_lag=None`` takes the largest requested lag. Pass 4 explicitly
    for a subset of the original nine fit moments to retain the same 152-row
    product window as the full fifteen-moment study.

    The conditional Gaussian fourth-moment identity supplies the complete
    finite-sample estimator covariance. This is neither HAC nor an asymptotic
    approximation. The estimators are quadratic forms and need not be normally
    distributed; exact first two moments alone do not justify a chi-square J
    test, normal parameter intervals or a composite-null test.
    """
    specs = _specifications(moments)
    nobs = _positive_integer(nobs, "nobs", 2)
    if not isinstance(demean, (bool, np.bool_)):
        raise ValueError("demean must be boolean")
    maximum = max(h for _, _, h in specs)
    window_lag = maximum if common_max_lag is None else _positive_integer(common_max_lag, "common_max_lag", 0)
    if window_lag < maximum or window_lag >= nobs:
        raise ValueError("common_max_lag must cover every requested lag and be less than nobs")
    columns = tuple(dict.fromkeys(name for a, b, _ in specs for name in (a, b)))
    if any(name not in OBSERVED_VARS for name in columns):
        raise ValueError("moment variables must be SW07 observed variable names")
    model, state_covariance, diagnostics = _stationary_system(params)
    observation = model.Z[[OBSERVED_VARS.index(name) for name in columns]]
    gamma = np.empty((nobs, len(columns), len(columns)))
    propagated = observation.copy()
    for lag in range(nobs):
        gamma[lag] = propagated @ state_covariance @ observation.T
        propagated = propagated @ model.T
    dates = np.arange(nobs)
    difference = dates[:, None] - dates[None, :]
    distance = np.abs(difference)
    blocks = np.empty((len(columns), len(columns), nobs, nobs))
    for i in range(len(columns)):
        for j in range(len(columns)):
            block = np.where(difference >= 0, gamma[distance, i, j], gamma[distance, j, i])
            if demean:
                block = block - block.mean(axis=0)[None, :] - block.mean(axis=1)[:, None] + block.mean()
            blocks[i, j] = block
    # Tiny asymmetry can enter a near-singular state covariance through the
    # Lyapunov solve; symmetrize its time-series covariance representation only.
    blocks = .5 * (blocks + blocks.transpose(1, 0, 3, 2))
    positions = {name: i for i, name in enumerate(columns)}
    indexed = tuple((positions[a], positions[b], h) for a, b, h in specs)
    t = np.arange(window_lag, nobs)
    effective = len(t)
    expectation = np.array([np.mean(blocks[a, b, t, t-h]) for a, b, h in indexed])
    covariance = np.empty((len(specs), len(specs)))
    for i, (a, b, h) in enumerate(indexed):
        for j in range(i, len(specs)):
            c, d, k = indexed[j]
            # Isserlis: Cov(A B, C D)=Cov(A,C)Cov(B,D)+Cov(A,D)Cov(B,C).
            first = blocks[a, c][np.ix_(t, t)] * blocks[b, d][np.ix_(t-h, t-k)]
            second = blocks[a, d][np.ix_(t, t-k)] * blocks[b, c][np.ix_(t-h, t)]
            covariance[i, j] = covariance[j, i] = float(np.sum(first + second) / effective**2)
    metadata = {
        **diagnostics, "evidence_kind": "exact_conditional_gaussian_sampling_moments",
        "is_observed_data": False, "nobs": nobs, "effective_observations": effective,
        "frequency": "Q", "common_max_lag": window_lag, "demean": bool(demean),
        "mean_treatment": "estimated mean over all nobs observations" if demean else "known model population mean removed; not raw products with intercepts",
        "moment_orientation": "Cov(left_t, right_t-lag)",
        "covariance_scale": "exact covariance of finite-sample moment estimators",
        "covariance_method": "Gaussian Isserlis fourth moments after exact finite-sample centering; not HAC",
        "distribution_warning": "quadratic forms in Gaussian observations are not generally Gaussian; moments alone do not establish finite-sample parameter or J inference",
        "conditioning": "full fixed calibration, stationary Gaussian initial state and independent Gaussian innovations; no calibration uncertainty",
    }
    return MomentTargets(expectation, covariance, _labels(specs),
                         tuple(f"({_OBSERVATION_UNITS[a]}) * ({_OBSERVATION_UNITS[b]})" for a, b, _ in specs),
                         metadata)


def simulate_sw07_sample(params: Mapping[str, float], *, nobs: int = 156, seed: int = 0) -> pd.DataFrame:
    """Draw all seven SW07 observables from their stationary Gaussian law.

    The initial state is drawn from the Lyapunov covariance, independently of
    subsequent Gaussian innovations. No zero-state initialization, approximate
    burn-in or numerical Kalman measurement noise is used. Observation means
    ``model.d`` are included and recorded in ``frame.attrs``. For a known-mean
    control, subtract those means rather than the simulated sample averages.

    A seed creates one reproducible NumPy Generator: first a state-dimension
    standard-normal vector, then a (nobs-1, shock-dimension) innovation array.
    PSD state and innovation covariances use eigensquare-roots with only
    roundoff-negative eigenvalues clipped, never a positive ridge.
    """
    nobs = _positive_integer(nobs, "nobs", 2)
    seed = _positive_integer(seed, "seed", 0)
    model, covariance, diagnostics = _stationary_system(params)
    state_factor, state_factor_diagnostics = _psd_root(covariance, "SW07 stationary covariance")
    shock_factor, shock_factor_diagnostics = _psd_root(model.Q, "SW07 innovation covariance")
    rng = np.random.default_rng(seed)
    state = state_factor @ rng.standard_normal(len(model.T))
    innovations = rng.standard_normal((nobs - 1, model.Q.shape[0]))
    loading = model.R @ shock_factor
    observations = np.empty((nobs, len(OBSERVED_VARS)))
    observations[0] = model.Z @ state + model.d
    for t in range(1, nobs):
        state = model.T @ state + loading @ innovations[t-1]
        observations[t] = model.Z @ state + model.d
    if not np.all(np.isfinite(observations)):
        raise np.linalg.LinAlgError("SW07 Gaussian simulation produced nonfinite observations")
    frame = pd.DataFrame(observations, columns=OBSERVED_VARS,
                         index=pd.period_range("1966Q1", periods=nobs, freq="Q", name="date"))
    frame.attrs.update({**diagnostics, "is_synthetic": True, "seed": seed,
                        "nobs": nobs, "frequency": "Q", "units": dict(_OBSERVATION_UNITS),
                        "initialization": "exact stationary Gaussian initial state from Lyapunov covariance",
                        "observation_intercepts_included": True,
                        "population_means": dict(zip(OBSERVED_VARS, model.d.tolist())),
                        "state_covariance_factor": state_factor_diagnostics,
                        "innovation_covariance_factor": shock_factor_diagnostics,
                        "innovation_distribution": "independent Gaussian",
                        "calendar_note": "synthetic quarterly labels only; not historical observations"})
    return frame


__all__ = ["sw07_finite_sample_moments", "simulate_sw07_sample"]
