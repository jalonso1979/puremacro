"""SW07 observation equation: maps model variables to observable series.

Ported from puremacro/dsge/_references/sw07_pfeifer.mod (varobs +
observation equations). Seven observables:

    dy      = y - y(-1) + ctrend           (real per-capita GDP growth)
    dc      = c - c(-1) + ctrend           (real per-capita consumption growth)
    dinve   = inve - inve(-1) + ctrend     (real per-capita investment growth)
    dw      = w - w(-1) + ctrend           (real wage growth)
    labobs  = lab + constelab              (log hours)
    pinfobs = pinf + constepinf            (GDP-deflator inflation)
    robs    = r + conster                  (federal funds rate, quarterly)

State variables are at indices [0:20]; controls at [20:44].
"""
from __future__ import annotations

import numpy as np

from puremacro.dsge.smets_wouters import (
    solve_sw07,
    STATE_NAMES,
    CONTROL_NAMES,
    _compute_derived,
)
from puremacro.state_space import StateSpaceModel


OBSERVED_VARS: tuple[str, ...] = (
    "gdp_growth", "cons_growth", "inv_growth",
    "wage_growth", "log_hours", "infl", "ffr",
)


def _idx(name: str) -> int:
    """Index into the full z_t vector (states first, then controls)."""
    if name in STATE_NAMES:
        return STATE_NAMES.index(name)
    if name in CONTROL_NAMES:
        return len(STATE_NAMES) + CONTROL_NAMES.index(name)
    raise KeyError(f"variable {name!r} not in STATE_NAMES or CONTROL_NAMES")


def make_state_space(params: dict) -> StateSpaceModel:
    """Build the SW07 state-space form for Kalman filtering.

    Parameters
    ----------
    params : dict
        Combined structural + shock-std parameters. Must contain all keys in
        SW07_POSTERIOR_MODE plus the 7 shock-std names
        ('ea', 'eb', 'eg', 'eqs', 'em', 'epinf', 'ew').

    Returns
    -------
    StateSpaceModel
        T (44, 44), Z (7, 44), R (44, 7), Q (7, 7), H (7, 7), d (7,), c (44,)
    """
    sol = solve_sw07(params)
    T_mat = sol.G        # (44, 44) — state transition
    R_mat = sol.Impact   # (44, 7)  — shock loading

    # Observation matrix Z: pick model variables that map to each observable.
    n_state = T_mat.shape[0]
    Z = np.zeros((7, n_state))
    # gdp_growth: y - y_lag
    Z[0, _idx("y")]     =  1.0
    Z[0, _idx("y_lag")] = -1.0
    # cons_growth: c - c_lag
    Z[1, _idx("c")]     =  1.0
    Z[1, _idx("c_lag")] = -1.0
    # inv_growth: inve - inve_lag
    Z[2, _idx("inve")]     =  1.0
    Z[2, _idx("inve_lag")] = -1.0
    # wage_growth: w - w_lag
    Z[3, _idx("w")]     =  1.0
    Z[3, _idx("w_lag")] = -1.0
    # log_hours: lab
    Z[4, _idx("lab")] = 1.0
    # infl: pinf
    Z[5, _idx("pinf")] = 1.0
    # ffr: r
    Z[6, _idx("r")] = 1.0

    # State-shock covariance.
    shock_names = ("ea", "eb", "eg", "eqs", "em", "epinf", "ew")
    sigmas = np.array([params[s] for s in shock_names], dtype=float)
    Q_mat = np.diag(sigmas ** 2)

    # Measurement-error covariance (small ridge for Kalman conditioning).
    H_mat = 1e-8 * np.eye(7)

    # Measurement intercept d.
    ctrend     = params["ctrend"]
    constelab  = params["constelab"]
    constepinf = params["constepinf"]
    derived = _compute_derived(params)
    conster = derived["conster"]

    d_vec = np.array([
        ctrend, ctrend, ctrend, ctrend,
        constelab, constepinf, conster,
    ], dtype=float)

    # State intercept (zeros for SW07 — model is in deviations from steady state).
    c_vec = np.zeros(n_state)

    return StateSpaceModel(
        T=T_mat,
        Z=Z,
        R=R_mat,
        Q=Q_mat,
        H=H_mat,
        c=c_vec,
        d=d_vec,
    )


__all__ = ["OBSERVED_VARS", "make_state_space"]
