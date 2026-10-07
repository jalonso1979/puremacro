"""Local-projection targets with joint time-series sampling covariance."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from puremacro._linalg import inv_xtx
from puremacro.inference._ols_helpers import ols_hac
from .bridge import MomentTargets


def lp_moment_targets(
    data: pd.DataFrame,
    *,
    responses: Sequence[str],
    shock: str,
    horizons: Sequence[int],
    response_units: Mapping[str, str],
    shock_unit: str,
    frequency: str,
    lags: int = 2,
    controls: Sequence[str] = (),
    bandwidth: int | None = None,
    shock_size: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
) -> MomentTargets:
    """Estimate LP responses and their full cross-response/horizon covariance.

    As in ``lp_hac``, each regression explains ``y[t+h] - y[t-1]`` with
    a constant, ``shock[t]``, lags of that response and the shock, and both
    contemporaneous and lagged controls. Coefficients are rescaled to an
    innovation of ``shock_size`` in the explicitly declared ``shock_unit``.
    This assumes an exogenous shock conditional on those regressors; it is
    not an instrumental-variable estimator or an identification test.

    The coefficient influence contributions are aligned on the original
    time index, zero outside each horizon's estimation window. A single
    Bartlett HAC kernel is applied to their joint matrix, preserving
    covariance across horizons and responses. ``bandwidth`` defaults to
    max(horizons)+1, a transparent heuristic rather than an optimal rule.
    The returned covariance is Cov(the estimators), with no extra n scaling.
    No small-sample correction is applied.

    Rows must represent consecutive, equally spaced observations at
    ``frequency``. For DatetimeIndex/PeriodIndex a regular grid is checked;
    other ordered indices use the caller's explicit frequency declaration.
    Missing/nonfinite data are refused, rather than silently collapsing time.
    All quantities are in supplied units; no name-based percent conversion.
    """
    if not isinstance(data, pd.DataFrame) or data.empty or not data.columns.is_unique:
        raise ValueError("data must be a nonempty DataFrame with unique columns")
    if not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError("data index must be unique and increasing")
    if isinstance(responses, str) or isinstance(controls, str):
        raise ValueError("responses and controls must be sequences of column names")
    responses, controls = tuple(responses), tuple(controls)
    names = responses + (shock,) + controls
    if not responses or any(not isinstance(x, str) or not x.strip() for x in names):
        raise ValueError("responses, shock and controls require nonempty column names")
    if len(set(names)) != len(names):
        raise ValueError("responses, shock and controls must be distinct")
    if not isinstance(response_units, Mapping) or set(response_units) != set(responses):
        raise ValueError("response_units must specify exactly the response columns")
    if any(not isinstance(x, str) or not x.strip() for x in (*response_units.values(), shock_unit, frequency)):
        raise ValueError("response units, shock_unit and frequency must be explicit nonempty strings")
    if isinstance(data.index, pd.PeriodIndex) and len(data) > 1:
        if not data.index.equals(pd.period_range(data.index[0], periods=len(data), freq=frequency)):
            raise ValueError("PeriodIndex must be consecutive at the declared frequency")
    if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
        if not data.index.equals(pd.date_range(data.index[0], periods=len(data), freq=frequency)):
            raise ValueError("DatetimeIndex must be consecutive at the declared frequency")
    if not isinstance(lags, (int, np.integer)) or isinstance(lags, (bool, np.bool_)) or lags < 0:
        raise ValueError("lags must be a nonnegative integer")
    horizons = tuple(horizons)
    if not horizons or any(not isinstance(h, (int, np.integer)) or isinstance(h, (bool, np.bool_)) or h < 0 for h in horizons):
        raise ValueError("horizons must contain nonnegative integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique")
    bw = max(horizons) + 1 if bandwidth is None else bandwidth
    if not isinstance(bw, (int, np.integer)) or isinstance(bw, (bool, np.bool_)) or bw < 0 or bw >= len(data):
        raise ValueError("bandwidth must be a nonnegative integer less than the sample length")
    if not np.isscalar(shock_size) or np.iscomplexobj(shock_size) or not np.isfinite(shock_size) or shock_size == 0:
        raise ValueError("shock_size must be a finite nonzero real scalar")
    missing = set(names) - set(data.columns)
    if missing:
        raise ValueError(f"data lack required columns: {sorted(missing)}")
    if np.iscomplexobj(data.loc[:, list(names)].to_numpy()):
        raise ValueError("LP inputs must be real")
    values = data.loc[:, list(names)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("LP inputs must be finite; do not silently drop or compress missing periods")
    n = len(data)
    scores = np.zeros((n, len(responses) * len(horizons)))
    estimates, labels, units, n_obs = [], [], [], []
    shock_values = data[shock].to_numpy(dtype=float)
    first = max(1, lags)
    for response in responses:
        y = data[response].to_numpy(dtype=float)
        for h in horizons:
            t = np.arange(first, n - h)
            columns = [np.ones(len(t)), shock_values[t]]
            for lag in range(1, lags + 1):
                columns.extend((shock_values[t - lag], y[t - lag]))
                columns.extend(data[c].to_numpy(dtype=float)[t - lag] for c in controls)
            columns.extend(data[c].to_numpy(dtype=float)[t] for c in controls)
            x = np.column_stack(columns)
            if len(t) <= x.shape[1] or bw >= len(t):
                raise ValueError("not enough observations for the requested LP design/horizon/bandwidth")
            fit = ols_hac(y[t + h] - y[t - 1], x, lags=int(bw))
            bread = inv_xtx(x, name="lp_moment_targets")
            j = len(estimates)
            scores[t, j] = (x @ bread[:, 1]) * fit["residuals"] * shock_size
            estimates.append(fit["beta"][1] * shock_size)
            labels.append(f"{response}:{shock}:h={h}")
            units.append(f"{response_units[response]} per {shock_size:g} {shock_unit} innovation")
            n_obs.append(len(t))
    covariance = scores.T @ scores
    for lag in range(1, int(bw) + 1):
        cross = scores[lag:].T @ scores[:-lag]
        covariance += (1 - lag / (bw + 1)) * (cross + cross.T)
    meta = dict(metadata or {})
    meta.update({
        "estimator": "local_projection_joint_hac", "responses": responses, "shock": shock,
        "horizons": horizons, "lags": int(lags), "controls": controls,
        "bandwidth": int(bw), "kernel": "Bartlett", "small_sample_correction": False,
        "frequency": frequency, "shock_unit": shock_unit, "shock_size": float(shock_size),
        "response_units": dict(response_units), "n_observations": n_obs,
        "sample_start": str(data.index[0]), "sample_end": str(data.index[-1]),
        "dependent_variable": "y[t+h] - y[t-1]", "covariance_scale": "empirical_estimator",
        "identification_assumption": "shock exogenous conditional on the declared regressors",
        "scope": "stationary/equally-spaced time-series LP; HAC asymptotic covariance, not weak-IV or uniform-inference correction",
    })
    return MomentTargets(estimates, covariance, labels, units=units, metadata=meta)


__all__ = ["lp_moment_targets"]
