"""Validation cases for the ``state_space`` subsystem.

Scope: the linear-Gaussian Kalman machinery in :mod:`puremacro.state_space`
(``kalman_filter`` / ``kalman_smoother`` on a :class:`StateSpaceModel`). Two
PACKAGE cross-checks against ``statsmodels.tsa.statespace`` built on *byte-
identical* system matrices and a *known* (finite, non-diffuse) initialisation,
plus four INTERNAL identities (prediction-error decomposition of the
log-likelihood, RTS terminal identity, smoother-variance reduction, and a
simulate-then-recover sanity check).

Pure deps only — the PACKAGE references are the frozen goldens in
``goldens/state_space.json``; live statsmodels is touched solely by the
``reference``-marked drift-guard in
``tests/validation/test_reference_drift_state_space.py``. puremacro estimator
imports live INSIDE the compute callables so importing this module stays
pyodide-light.
"""
from __future__ import annotations

import numpy as np

from ._model import Mechanism, Tol, ValidationCase


# ---------------------------------------------------------------------------
# Seeded demo data (local to this module; NOT added to the shared _fixtures).
# Both DGPs are univariate-observation linear-Gaussian state-space models so
# the puremacro recursions and the statsmodels reference are built from the
# SAME matrices / data, and an exact (analytical) comparison is meaningful.
# ---------------------------------------------------------------------------
def local_level_demo() -> dict:
    """Local level (random walk + measurement noise). Seeded -> identical.

        alpha_{t+1} = alpha_t + eta_t,   eta_t ~ N(0, q)
        y_t         = alpha_t + eps_t,   eps_t ~ N(0, h)
    """
    rng = np.random.default_rng(20260531)
    T_obs = 60
    q, h = 0.5, 1.3
    alpha = np.zeros(T_obs + 1)
    for t in range(T_obs):
        alpha[t + 1] = alpha[t] + np.sqrt(q) * rng.standard_normal()
    y = alpha[:T_obs] + np.sqrt(h) * rng.standard_normal(T_obs)
    return {
        "y": y,
        "alpha": alpha[:T_obs],   # true latent path (recovery check)
        "phi": 1.0,
        "q": q,
        "h": h,
        "a0": np.array([0.0]),
        "P0": np.array([[10.0]]),
    }


def ar1_noise_demo() -> dict:
    """Stationary AR(1) signal observed with measurement noise. Seeded.

        alpha_{t+1} = phi * alpha_t + eta_t,   eta_t ~ N(0, q)
        y_t         = alpha_t + eps_t,         eps_t ~ N(0, h)
    """
    rng = np.random.default_rng(424242)
    T_obs = 80
    phi, q, h = 0.7, 0.4, 0.9
    alpha = np.zeros(T_obs + 1)
    for t in range(T_obs):
        alpha[t + 1] = phi * alpha[t] + np.sqrt(q) * rng.standard_normal()
    y = alpha[:T_obs] + np.sqrt(h) * rng.standard_normal(T_obs)
    return {
        "y": y,
        "alpha": alpha[:T_obs],
        "phi": phi,
        "q": q,
        "h": h,
        "a0": np.array([0.0]),
        "P0": np.array([[5.0]]),
    }


def _build_model(spec: dict):
    """Assemble a univariate-observation StateSpaceModel from a demo spec."""
    from puremacro.state_space import StateSpaceModel

    return StateSpaceModel(
        T=np.array([[spec["phi"]]]),
        Z=np.array([[1.0]]),
        Q=np.array([[spec["q"]]]),
        H=np.array([[spec["h"]]]),
    )


# ---------------------------------------------------------------------------
# Compute callables (puremacro side)
# ---------------------------------------------------------------------------
def _filter_local_level() -> dict:
    from puremacro.state_space import kalman_filter

    spec = local_level_demo()
    out = kalman_filter(spec["y"], _build_model(spec), a0=spec["a0"], P0=spec["P0"])
    return {
        "a_filt": np.asarray(out["a_filt"], dtype=float).ravel(),
        "loglik": float(out["loglik"]),
    }


def _smoother_ar1() -> dict:
    from puremacro.state_space import kalman_smoother

    spec = ar1_noise_demo()
    out = kalman_smoother(spec["y"], _build_model(spec), a0=spec["a0"], P0=spec["P0"])
    return {
        "a_smooth": np.asarray(out["a_smooth"], dtype=float).ravel(),
        "P_smooth": np.asarray(out["P_smooth"], dtype=float).ravel(),
    }


def _loglik_prediction_error_decomp() -> dict:
    """The reported loglik must equal the prediction-error decomposition
    rebuilt independently from the stored innovations ``v_t`` and their
    covariances ``F_t``: ll = sum_t -0.5 (log 2pi + log F_t + v_t^2 / F_t).
    """
    from puremacro.state_space import kalman_filter

    spec = local_level_demo()
    out = kalman_filter(spec["y"], _build_model(spec), a0=spec["a0"], P0=spec["P0"])
    v = np.asarray(out["innov"], dtype=float).ravel()          # (T,) — n_obs = 1
    F = np.asarray(out["F"], dtype=float).reshape(v.shape[0], -1).ravel()
    ll_decomp = float(np.sum(-0.5 * (np.log(2.0 * np.pi) + np.log(F) + v * v / F)))
    return {"loglik": float(out["loglik"]), "loglik_decomp": ll_decomp}


def _rts_terminal_identity() -> dict:
    """At the final period the RTS smoother coincides with the filter:
    a_smooth[-1] == a_filt[-1] and P_smooth[-1] == P_filt[-1]."""
    from puremacro.state_space import kalman_smoother

    spec = ar1_noise_demo()
    out = kalman_smoother(spec["y"], _build_model(spec), a0=spec["a0"], P0=spec["P0"])
    return {
        "a_terminal_gap": float(abs(out["a_smooth"][-1, 0] - out["a_filt"][-1, 0])),
        "P_terminal_gap": float(abs(out["P_smooth"][-1, 0, 0] - out["P_filt"][-1, 0, 0])),
    }


def _smoother_variance_reduction() -> dict:
    """Smoothing uses the whole sample, so the smoothed state variance can
    never exceed the (real-time) filtered variance: P_filt - P_smooth >= 0.
    Returned as the elementwise gap; QUALITATIVE checks it stays >= 0."""
    from puremacro.state_space import kalman_smoother

    spec = ar1_noise_demo()
    out = kalman_smoother(spec["y"], _build_model(spec), a0=spec["a0"], P0=spec["P0"])
    gap = np.asarray(out["P_filt"], dtype=float).ravel() - np.asarray(
        out["P_smooth"], dtype=float
    ).ravel()
    return {"filter_minus_smooth_var": gap}


def _smoother_recovers_latent_state() -> dict:
    """Simulate-then-recover: the smoothed mean must track the true latent
    state better than the raw noisy observation does. Metric = relative RMSE
    improvement 1 - RMSE(a_smooth)/RMSE(y); QUALITATIVE requires a clear
    positive margin."""
    from puremacro.state_space import kalman_smoother

    spec = ar1_noise_demo()
    out = kalman_smoother(spec["y"], _build_model(spec), a0=spec["a0"], P0=spec["P0"])
    truth = spec["alpha"]
    rmse_sm = float(np.sqrt(np.mean((out["a_smooth"].ravel() - truth) ** 2)))
    rmse_y = float(np.sqrt(np.mean((spec["y"] - truth) ** 2)))
    return {"rmse_improvement": 1.0 - rmse_sm / rmse_y}


CASES: list[ValidationCase] = [
    ValidationCase(
        id="filter_local_level_vs_statsmodels",
        subsystem="state_space",
        title="Kalman filtered states and log-likelihood match statsmodels (local level)",
        title_es="Estados filtrados y verosimilitud de Kalman coinciden con statsmodels (nivel local)",
        mechanism=Mechanism.PACKAGE,
        compute=_filter_local_level,
        reference="state_space:filter_local_level_vs_statsmodels",  # frozen golden
        tol=Tol.TIGHT,
        citation=(
            "statsmodels 0.14.6 tsa.statespace.MLEModel.smooth() with "
            "initialization='known' on identical system matrices "
            "(Durbin & Koopman 2012, Time Series Analysis by State Space Methods, ch. 4)."
        ),
        notes="Finite known P0=10*I so puremacro and statsmodels share an exact prior; no diffuse approximation.",
    ),
    ValidationCase(
        id="smoother_ar1_vs_statsmodels",
        subsystem="state_space",
        title="RTS smoothed states and covariances match statsmodels (AR(1)+noise)",
        title_es="Estados y covarianzas suavizados (RTS) coinciden con statsmodels (AR(1)+ruido)",
        mechanism=Mechanism.PACKAGE,
        compute=_smoother_ar1,
        reference="state_space:smoother_ar1_vs_statsmodels",  # frozen golden
        tol=Tol.TIGHT,
        citation=(
            "statsmodels 0.14.6 tsa.statespace.MLEModel.smooth().smoothed_state / "
            "smoothed_state_cov on identical matrices "
            "(Rauch-Tung-Striebel smoother; Durbin & Koopman 2012, §4.4)."
        ),
        notes="Stationary AR(1) signal (phi=0.7) observed with measurement noise; known finite prior P0=5*I.",
    ),
    ValidationCase(
        id="loglik_prediction_error_decomposition",
        subsystem="state_space",
        title="Filter log-likelihood equals its prediction-error decomposition",
        title_es="La verosimilitud del filtro iguala su descomposicion en errores de prediccion",
        mechanism=Mechanism.INTERNAL,
        compute=_loglik_prediction_error_decomp,
        reference=lambda: {"loglik_decomp": _loglik_prediction_error_decomp()["loglik"]},
        tol=Tol.EXACT,
        citation=(
            "Gaussian prediction-error (Schweppe) decomposition of the likelihood: "
            "L = sum_t N(v_t; 0, F_t) (Harvey 1989, Forecasting, Structural Time "
            "Series Models and the Kalman Filter, §3.4; Durbin & Koopman 2012, §7.2)."
        ),
        notes="Reconstructs the loglik from stored innovations v_t and covariances F_t; agreement to machine precision.",
    ),
    ValidationCase(
        id="rts_terminal_matches_filter",
        subsystem="state_space",
        title="RTS smoother coincides with the filter at the terminal period",
        title_es="El suavizador RTS coincide con el filtro en el periodo terminal",
        mechanism=Mechanism.INTERNAL,
        compute=_rts_terminal_identity,
        reference=lambda: {"a_terminal_gap": 0.0, "P_terminal_gap": 0.0},
        tol=Tol.EXACT,
        citation=(
            "Boundary condition of the RTS backward recursion: at t=T the smoothed "
            "and filtered moments are identical (Durbin & Koopman 2012, §4.4.4)."
        ),
    ),
    ValidationCase(
        id="smoother_variance_le_filter",
        subsystem="state_space",
        title="Smoothed state variance never exceeds the filtered variance",
        title_es="La varianza suavizada del estado nunca supera a la varianza filtrada",
        mechanism=Mechanism.INTERNAL,
        compute=_smoother_variance_reduction,
        reference=lambda: {
            "filter_minus_smooth_var": np.zeros(
                np.asarray(ar1_noise_demo()["y"]).shape[0]
            )
        },
        tol=Tol.QUALITATIVE,  # require P_filt - P_smooth >= 0 elementwise
        citation=(
            "Conditioning on the full sample weakly reduces posterior variance: "
            "P_t^smooth <= P_t^filt (Anderson & Moore 1979, Optimal Filtering, ch. 7; "
            "Durbin & Koopman 2012, §4.4)."
        ),
    ),
    ValidationCase(
        id="smoother_recovers_latent_state",
        subsystem="state_space",
        title="Smoother recovers the latent state better than the noisy observation",
        title_es="El suavizador recupera el estado latente mejor que la observacion ruidosa",
        mechanism=Mechanism.INTERNAL,
        compute=_smoother_recovers_latent_state,
        # Conservative lower bound (observed improvement ~0.36 on this seed).
        reference=lambda: {"rmse_improvement": 0.20},
        tol=Tol.QUALITATIVE,  # metric must clear the lower bound
        citation=(
            "Simulate-then-recover validation: the conditional mean E[alpha_t | y_{1:T}] "
            "is the minimum-MSE estimator of the latent state (Durbin & Koopman 2012, "
            "§4.5), so its RMSE against the truth must beat using y as a proxy."
        ),
        notes="DGP-truth comparison on the AR(1)+noise simulation; threshold 0.20 << observed ~0.36.",
    ),
]
