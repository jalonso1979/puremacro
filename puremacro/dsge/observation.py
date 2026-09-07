"""Build the Kalman measurement equation from a ``varobs`` declaration.

A solved first-order model reports its variables in the timing

.. math::

    x_{t+1} = A x_t + B u_t, \\qquad v_t = ys + C x_t + D u_t

with ``x_t`` predetermined at the start of ``t``. The observable therefore
loads the **contemporaneous** innovation ``u_t``, while
:class:`~puremacro.state_space.StateSpaceModel` is

.. math::

    \\alpha_t = c + T \\alpha_{t-1} + R \\varepsilon_t, \\qquad
    y_t = d + Z \\alpha_t + \\eta_t

with ``\\eta`` independent of ``\\varepsilon``. There is no exact way to write
``D u_t`` as measurement noise, so the innovation is carried in the state:

.. math::

    \\alpha_t \\equiv \\begin{bmatrix} x_t \\\\ u_t \\end{bmatrix}, \\quad
    T = \\begin{bmatrix} A & B \\\\ 0 & 0 \\end{bmatrix}, \\quad
    R = \\begin{bmatrix} 0 \\\\ I_{n_e} \\end{bmatrix}, \\quad
    \\varepsilon_t = u_t, \\quad Q = \\Sigma_u

which checks: ``T \\alpha_{t-1} + R u_t = [A x_{t-1} + B u_{t-1};\\, 0] +
[0;\\, u_t] = [x_t;\\, u_t]``. The observation rows then read straight off the
decision rules, ``Z = [C_O, D_O]`` and ``d = ys_O``.

Three consequences worth knowing.

* **Growth-rate observables need no special casing.** In a .mod file
  ``dy = y - y(-1) + ctrend;`` declares ``dy`` as an endogenous variable, so
  ``varobs dy`` picks up row ``dy`` of ``(C, D)`` like any other. The index
  arithmetic in ``sw07_observation.py`` exists because that model is
  hand-built, not because the mapping is hard.
* **The smoothed structural shocks are the last ``n_e`` rows of the smoothed
  state.** No separate disturbance smoother is needed.
* **The state block and the innovation block are uncorrelated**, because the
  ``[x, u]`` block of both ``T P T'`` and ``R Q R'`` is zero. So the Lyapunov
  solution gives ``Z P Z' + H`` equal to the model's own analytical covariance
  of the observables.

Not handled here, deliberately: second-order solutions (a linear measurement
equation cannot represent them — refused rather than silently linearised), and
a time-varying measurement intercept, since ``StateSpaceModel`` is
time-invariant by construction. ``observation_trends`` is applied by detrending
the data, exactly as Dynare does.
"""
from __future__ import annotations

import difflib
from typing import Any, Mapping, Sequence

import numpy as np

from puremacro.state_space import StateSpaceModel

__all__ = ["make_state_space_from_varobs"]


def _resolve_observables(model: Any, varobs: Sequence[str]) -> list[str]:
    names = list(varobs)
    if not names:
        raise ValueError(
            "make_state_space_from_varobs: need at least one observable; "
            "the model's `varobs` declaration is empty or was not passed."
        )
    known = list(model.variables)
    unknown = [v for v in names if v not in known]
    if unknown:
        hints = []
        for v in unknown:
            close = difflib.get_close_matches(v, known, n=3, cutoff=0.5)
            hints.append(f"{v!r}" + (f" (did you mean {close}?)" if close else ""))
        raise ValueError(
            "make_state_space_from_varobs: "
            f"{', '.join(hints)} is not a variable of this model. "
            f"Declared variables: {known}"
        )
    dupes = sorted({v for v in names if names.count(v) > 1})
    if dupes:
        raise ValueError(
            f"make_state_space_from_varobs: repeated observable(s) {dupes}; "
            "each series may be observed once."
        )
    return names


def make_state_space_from_varobs(
    model: Any,
    varobs: Sequence[str],
    *,
    shock_cov: np.ndarray | None = None,
    measurement_error: Mapping[str, float] | None = None,
    prefilter: bool = False,
    ridge: float = 0.0,
) -> StateSpaceModel:
    """Map a solved model plus a ``varobs`` list onto a Kalman state space.

    Parameters
    ----------
    model : LinearModel
        A solved **first-order** model. A second-order
        :class:`~puremacro.dsge.PrunedDSGESolution` is refused: it exposes
        ``decision_rules()``, so the companion extractor would silently use its
        first-order block and drop every quadratic term.
    varobs : Sequence[str]
        Observable variables, in the column order of the data.
    shock_cov : ndarray, optional
        Innovation covariance ``Sigma_u``. Defaults to the covariance declared
        with the model (the ``shocks;`` block), correlations included.
    measurement_error : Mapping[str, float], optional
        Measurement-error **standard deviations** by observable name. ``H`` is
        their squares on the diagonal, and **zero** for anything omitted —
        Dynare's default. The ``1e-8`` ridge in ``sw07_observation.py`` is a
        property of that hand-built file, not of this mapping.
    prefilter : bool, default False
        Set ``d = 0`` and expect demeaned data, as Dynare's ``prefilter``.
    ridge : float, default 0.0
        Add ``ridge * I`` to ``H``. Opt-in conditioning, never a default: a
        singular innovation covariance is a specification error worth hearing
        about, and ``estimate_dsge._check_stochastic_singularity`` says so.

    Returns
    -------
    StateSpaceModel
        With ``m = n_states + n_shocks`` and ``r = n_shocks``.

    Notes
    -----
    The measurement intercept follows the model's own units. For a variable
    with ``units[name] == "level"`` the deviation is ``y_t - ss`` and
    ``d = ss``; for ``"log"`` (``build``'s default) it is ``log(y_t) - log(ss)``
    and ``d = log(ss)``, so **the data column must then be in logs**. Models
    from ``build_dynare`` / ``load_mod`` are approximated in levels throughout.
    """
    from puremacro.dsge.decomposition import _extract_companion_matrices
    from puremacro.dsge.pruning import PrunedDSGESolution

    if isinstance(model, PrunedDSGESolution):
        raise TypeError(
            "make_state_space_from_varobs: got a second-order "
            "PrunedDSGESolution. A linear measurement equation cannot "
            "represent it, and reading only its first-order block would "
            "silently discard ghxx, ghxu, ghuu and ghs2. Solve at order=1 for "
            "Kalman filtering; a particle filter for order 2 is a later "
            "release."
        )

    names = _resolve_observables(model, varobs)
    A, B, C, D, ys, variables, shocks, states, sigma_u = _extract_companion_matrices(model)

    n_s, n_e, n_obs = len(states), len(shocks), len(names)
    if n_e == 0:
        raise ValueError(
            "make_state_space_from_varobs: the model declares no shocks, so "
            "the observables are deterministic and there is nothing for the "
            "Kalman filter to infer."
        )

    if shock_cov is not None:
        Q = np.asarray(shock_cov, dtype=float)
        if Q.shape != (n_e, n_e):
            raise ValueError(
                f"make_state_space_from_varobs: shock_cov must be "
                f"({n_e}, {n_e}) for shocks {shocks}; got {Q.shape}"
            )
        Q = 0.5 * (Q + Q.T)
    else:
        Q = np.asarray(sigma_u, dtype=float)

    # alpha_t = [x_t; u_t]
    T = np.zeros((n_s + n_e, n_s + n_e))
    T[:n_s, :n_s] = A
    T[:n_s, n_s:] = B
    R = np.zeros((n_s + n_e, n_e))
    R[n_s:, :] = np.eye(n_e)

    rows = [variables.index(v) for v in names]
    Z = np.hstack([np.asarray(C, dtype=float)[rows],
                   np.asarray(D, dtype=float)[rows]])

    if prefilter:
        d = np.zeros(n_obs)
    else:
        units = getattr(model, "units", {}) or {}
        d = np.empty(n_obs)
        for i, (v, r) in enumerate(zip(names, rows)):
            level = float(ys[r])
            if units.get(v, "level") == "log":
                if level <= 0.0:
                    raise ValueError(
                        f"make_state_space_from_varobs: {v!r} is a log-unit "
                        f"variable with a non-positive steady state ({level!r}); "
                        "its measurement intercept log(ss) is undefined."
                    )
                d[i] = np.log(level)
            else:
                d[i] = level

    me = dict(measurement_error or {})
    stray = [k for k in me if k not in names]
    if stray:
        raise ValueError(
            f"make_state_space_from_varobs: measurement_error names "
            f"{sorted(stray)}, which are not among the observables {names}."
        )
    H = np.diag([float(me.get(v, 0.0)) ** 2 for v in names])
    if ridge:
        H = H + float(ridge) * np.eye(n_obs)

    return StateSpaceModel(
        T=T, Z=Z, Q=Q, H=H, R=R,
        c=np.zeros(n_s + n_e), d=d,
    )
