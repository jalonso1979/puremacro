"""Fast exact expectations of finite-sample SW07 covariance estimators.

This module evaluates the same expectations as ``sw07_finite_sample_moments``
without constructing the joint covariance of the moment estimators. It is
intended for repeated evaluations inside a minimum-distance estimator.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from puremacro.dsge.sw07_observation import OBSERVED_VARS
from puremacro.structural.empirical_sw07 import ALL_MOMENTS, _labels, _specifications
from puremacro.structural.sw07_sampling import _positive_integer, _stationary_system


def sw07_finite_sample_expectations(
    params: Mapping[str, float], moments: Sequence[tuple[str, str, int]] = ALL_MOMENTS,
    *, nobs: int = 156, common_max_lag: int | None = None, demean: bool = True,
) -> pd.Series:
    """Exact expectations of common-window sample covariance moments.

    A moment ``(a, b, h)`` averages
    ``(a[t] - mean(a)) * (b[t-h] - mean(b))`` for
    ``t=common_max_lag,...,nobs-1``. Sample means use all ``nobs`` dates.
    Set ``common_max_lag=4`` explicitly when selecting just the nine original
    fit moments to preserve their common window with the six reserved moments.
    ``demean=False`` removes the *known model population means*, not zero;
    deterministic observation intercepts never enter these central moments.

    With ``Gamma[h] = Cov(y[t], y[t-h])``, the row sum of the finite time
    covariance at date ``t`` is ``Gamma[0] + sum(Gamma[1:t+1]) +
    sum(Gamma[1:nobs-t].T)``. Dividing by ``nobs`` gives
    ``Cov(y[t], sample_mean)``. Prefix sums therefore provide exact demeaning
    corrections without constructing quadratic-size time covariance blocks
    or evaluating fourth moments. There is no simulation or interpolation.

    Expectations are conditional on the supplied stationary SW07 calibration.
    Correcting the model moments alone does not establish valid parameter
    confidence intervals or a chi-square specification test.
    """
    specs = _specifications(moments)
    nobs = _positive_integer(nobs, "nobs", 2)
    if not isinstance(demean, (bool, np.bool_)):
        raise ValueError("demean must be boolean")
    maximum = max(h for _, _, h in specs)
    window_lag = (maximum if common_max_lag is None else
                  _positive_integer(common_max_lag, "common_max_lag", 0))
    if window_lag < maximum or window_lag >= nobs:
        raise ValueError("common_max_lag must cover every requested lag and be less than nobs")
    columns = tuple(dict.fromkeys(name for a, b, _ in specs for name in (a, b)))
    if any(name not in OBSERVED_VARS for name in columns):
        raise ValueError("moment variables must be SW07 observed variable names")
    model, state_covariance, diagnostics = _stationary_system(params)
    observation = model.Z[[OBSERVED_VARS.index(name) for name in columns]]
    # Estimated means depend on all dates. With known means, only the
    # requested covariance lags are necessary.
    horizon = nobs if demean else maximum + 1
    gamma = np.empty((horizon, len(columns), len(columns)))
    propagated = observation.copy()
    loading = state_covariance @ observation.T
    for lag in range(horizon):
        gamma[lag] = propagated @ loading
        propagated = propagated @ model.T
    gamma[0] = .5 * (gamma[0] + gamma[0].T)
    positions = {name: i for i, name in enumerate(columns)}
    indexed = tuple((positions[a], positions[b], h) for a, b, h in specs)
    expectation = np.array([gamma[h, a, b] for a, b, h in indexed])
    if demean:
        prefix = np.zeros_like(gamma)
        prefix[1:] = np.cumsum(gamma[1:], axis=0)
        mean_covariance = (gamma[0] + prefix + prefix[::-1].transpose(0, 2, 1)) / nobs
        grand_mean = mean_covariance.mean(axis=0)
        dates = np.arange(window_lag, nobs)
        expectation += np.array([
            -mean_covariance[dates, a, b].mean()
            -mean_covariance[dates - h, b, a].mean() + grand_mean[a, b]
            for a, b, h in indexed
        ])
    result = pd.Series(expectation, index=_labels(specs), dtype=float)
    result.attrs.update({
        **diagnostics, "evidence_kind": "exact_conditional_finite_sample_expectations",
        "nobs": nobs, "effective_observations": nobs - window_lag,
        "common_max_lag": window_lag, "demean": bool(demean),
        "mean_treatment": ("estimated mean over all nobs observations" if demean else
                           "known model population mean removed; not raw products with intercepts"),
        "moment_orientation": "Cov(left_t, right_t-lag)",
        "conditioning": "full fixed stationary model calibration; no calibration uncertainty",
        "inference_warning": "exact expectations alone do not establish parameter or chi-square J inference",
    })
    return result


__all__ = ["sw07_finite_sample_expectations"]
