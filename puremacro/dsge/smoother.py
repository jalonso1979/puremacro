"""Kalman smoothing and forecasting at calibrated parameters.

This is Dynare's ``calib_smoother`` plus ``forecast``: run the existing
:func:`~puremacro.state_space.kalman_smoother` over the measurement equation
:func:`~puremacro.dsge.observation.make_state_space_from_varobs` builds, and
report the smoothed states, the smoothed structural innovations and the fitted
observables.

The innovations come free. Because the filter state is ``alpha_t = [x_t; u_t]``,
``E[u_t | y_{1:T}]`` is the last ``n_e`` rows of the smoothed state; no
disturbance smoother is needed.

``observation_trends`` is handled by detrending the data before filtering and
adding the trend back to the fitted observables and the forecast, exactly as
Dynare does. ``StateSpaceModel`` is time-invariant by construction, so a
time-varying measurement intercept cannot live inside the filter.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from puremacro.dsge._results import DSGEForecastResult, SmootherResult
from puremacro.dsge.observation import make_state_space_from_varobs
from puremacro.state_space import kalman_filter, kalman_smoother

__all__ = ["smooth_model", "forecast_model"]


def _resolve_varobs(model: Any, varobs: Sequence[str] | None) -> list[str]:
    if varobs is not None:
        return list(varobs)
    declared = getattr(model, "_varobs", None)
    if declared:
        return list(declared)
    raise ValueError(
        "no observables: pass varobs=[...] or load a .mod file that declares "
        "`varobs`. The model carries no varobs declaration."
    )


def _aligned_data(data: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(np.asarray(data))
    missing = [v for v in names if v not in data.columns]
    if missing:
        raise ValueError(
            f"data is missing column(s) {missing} for observable(s) {list(names)}; "
            f"it has {list(data.columns)}"
        )
    return data[list(names)]


def _trend_matrix(
    trends: Mapping[str, float] | None, names: Sequence[str], n_t: int, offset: int = 0
) -> np.ndarray:
    """``(n_t, n_obs)`` of ``slope * t``, ``t`` counted from ``offset``."""
    out = np.zeros((n_t, len(names)))
    if not trends:
        return out
    stray = [k for k in trends if k not in names]
    if stray:
        raise ValueError(
            f"observation_trends names {sorted(stray)}, which are not among the "
            f"observables {list(names)}."
        )
    t = np.arange(offset, offset + n_t, dtype=float)
    for i, v in enumerate(names):
        if v in trends:
            out[:, i] = float(trends[v]) * t
    return out


def _initial_condition(ssm, a0, P0):
    """``(a0, P0)``, defaulting to the model's unconditional distribution.

    The same choice ``estimate_dsge`` makes: a solved linear DSGE is stationary
    by construction, so the Lyapunov solution is its prior, and a diffuse
    ``1e6 * I`` would add an arbitrary scale-dependent constant to the
    log-likelihood.
    """
    if a0 is not None or P0 is not None:
        return a0, P0
    from puremacro.dsge.estimate import _stationary_init

    return _stationary_init(ssm)


def smooth_model(
    model: Any,
    data: pd.DataFrame,
    *,
    varobs: Sequence[str] | None = None,
    shock_cov: np.ndarray | None = None,
    measurement_error: Mapping[str, float] | None = None,
    prefilter: bool = False,
    observation_trends: Mapping[str, float] | None = None,
    ridge: float = 0.0,
    a0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> SmootherResult:
    """Smooth ``data`` through ``model`` at its calibrated parameters."""
    names = _resolve_varobs(model, varobs)
    frame = _aligned_data(data, names)
    ssm = make_state_space_from_varobs(
        model, names, shock_cov=shock_cov, measurement_error=measurement_error,
        prefilter=prefilter, ridge=ridge,
    )

    trend = _trend_matrix(observation_trends, names, len(frame))
    y = frame.to_numpy(dtype=float) - trend

    a_init, P_init = _initial_condition(ssm, a0, P0)
    out = kalman_smoother(y, ssm, a0=a_init, P0=P_init)

    n_s = len(model.states)
    a_sm = out["a_smooth"]
    fitted = (ssm.d + a_sm @ ssm.Z.T) + trend
    idx = frame.index

    return SmootherResult(
        states=pd.DataFrame(a_sm[:, :n_s], columns=list(model.states), index=idx),
        shocks=pd.DataFrame(a_sm[:, n_s:], columns=list(model.shocks), index=idx),
        smoothed_obs=pd.DataFrame(fitted, columns=names, index=idx),
        filtered_states=pd.DataFrame(
            out["a_filt"][:, :n_s], columns=list(model.states), index=idx),
        loglik=float(out["loglik"]),
        varobs=tuple(names),
        _model=model,
        _data=frame,
    )


def forecast_model(
    model: Any,
    horizon: int = 8,
    *,
    data: pd.DataFrame | None = None,
    variables: Sequence[str] | None = None,
    ci: float = 0.90,
    varobs: Sequence[str] | None = None,
    shock_cov: np.ndarray | None = None,
    measurement_error: Mapping[str, float] | None = None,
    prefilter: bool = False,
    observation_trends: Mapping[str, float] | None = None,
    ridge: float = 0.0,
    a0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> DSGEForecastResult:
    """Forecast ``variables`` (default: the observables) ``horizon`` periods.

    With ``data``, the forecast starts from the terminal filtered state and its
    covariance; without it, from the steady state with no state uncertainty.
    """
    from scipy.stats import norm

    if not isinstance(horizon, (int, np.integer)) or int(horizon) < 1:
        raise ValueError(f"horizon must be an integer >= 1, got {horizon!r}")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must lie strictly inside (0, 1), got {ci!r}")
    horizon = int(horizon)

    names = _resolve_varobs(model, varobs)
    ssm = make_state_space_from_varobs(
        model, names, shock_cov=shock_cov, measurement_error=measurement_error,
        prefilter=prefilter, ridge=ridge,
    )
    # Reporting rows: the observables unless the caller widens the request.
    report = list(names) if variables is None else list(variables)
    ssm_out = ssm if report == list(names) else make_state_space_from_varobs(
        model, report, shock_cov=shock_cov, prefilter=prefilter,
    )

    m = ssm.T.shape[0]
    if data is not None:
        frame = _aligned_data(data, names)
        trend = _trend_matrix(observation_trends, names, len(frame))
        y = frame.to_numpy(dtype=float) - trend
        a_init, P_init = _initial_condition(ssm, a0, P0)
        filt = kalman_filter(y, ssm, a0=a_init, P0=P_init)
        a, P = filt["a_filt"][-1].copy(), filt["P_filt"][-1].copy()
        t0 = len(frame)
    else:
        a = np.zeros(m) if a0 is None else np.asarray(a0, dtype=float).copy()
        P = np.zeros((m, m)) if P0 is None else np.asarray(P0, dtype=float).copy()
        t0 = 0

    RQR = ssm.R @ ssm.Q @ ssm.R.T
    z = float(norm.ppf(0.5 * (1.0 + ci)))
    mean = np.zeros((horizon, len(report)))
    sd = np.zeros((horizon, len(report)))
    for h in range(horizon):
        a = ssm.T @ a
        P = ssm.T @ P @ ssm.T.T + RQR
        mean[h] = ssm_out.d + ssm_out.Z @ a
        var = np.diag(ssm_out.Z @ P @ ssm_out.Z.T + ssm_out.H)
        sd[h] = np.sqrt(np.maximum(var, 0.0))

    fwd_trend = _trend_matrix(observation_trends, names, horizon, offset=t0)
    if report == list(names):
        mean = mean + fwd_trend

    idx = pd.RangeIndex(1, horizon + 1, name="horizon")
    return DSGEForecastResult(
        mean=pd.DataFrame(mean, columns=report, index=idx),
        lower=pd.DataFrame(mean - z * sd, columns=report, index=idx),
        upper=pd.DataFrame(mean + z * sd, columns=report, index=idx),
        ci=float(ci),
        horizon=horizon,
    )
