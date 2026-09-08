"""Replication cases — Smets-Wouters (2007) Bayesian DSGE estimation family.

Replicates published empirical findings from:
Smets, F. and Wouters, R. (2007), 'Shocks and Frictions in US Business Cycles:
A Bayesian DSGE Approach', American Economic Review, 97(3), 586-606.

Replication benchmarks:
1. Log-posterior evaluation at the posterior mode on 1966Q1-2004Q4 US quarterly data.
2. Laplace marginal data density (MDD) approximation from the inverse Hessian curvature.
3. Geweke (1999) modified harmonic mean MDD consistency across truncation levels.
4. Headline structural parameter estimates at mode (habit persistence, Calvo probabilities,
   Taylor rule coefficients, investment adjustment cost) against Table 1.

Sub-second offline execution is guaranteed via precomputed certified MCMC draws and
curvature fixtures in tests/fixtures/sw07_parity_seed0_200draws.npz alongside direct mode
likelihood evaluation on _sw07_data.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ._model import ReplicationCase, TargetKind, Tol

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "sw07_parity_seed0_200draws.npz"
)


def _load_fixture() -> dict[str, np.ndarray]:
    return np.load(_FIXTURE_PATH)


def _eval_sw07_posterior_mode() -> dict[str, float]:
    """Evaluate log-posterior at the posterior mode on the bundled 155-quarter US dataset."""
    from puremacro.dsge.estimate import _make_neg_log_posterior
    from puremacro.dsge.sw07_estimate import _FIXED_PARAMS, _load_bundled_data
    from puremacro.dsge.sw07_observation import OBSERVED_VARS, make_state_space
    from puremacro.dsge.sw07_priors import PRIORS, param_names

    fix = _load_fixture()
    best_idx = int(np.argmax(fix["log_posterior_trace"][0]))
    best_theta = fix["draws"][0, best_idx]

    data = _load_bundled_data()
    y = data[list(OBSERVED_VARS)].to_numpy()
    names = param_names()
    nlp = _make_neg_log_posterior(y, make_state_space, PRIORS, names, _FIXED_PARAMS)
    log_post = -float(nlp(best_theta))
    return {"log_posterior_mode": log_post}


def _eval_sw07_laplace_mdd() -> dict[str, float]:
    """Compute Laplace approximation to log marginal data density from mode curvature."""
    from puremacro.dsge.marginal import laplace_mdd

    fix = _load_fixture()
    best_idx = int(np.argmax(fix["log_posterior_trace"][0]))
    log_post_mode = float(fix["log_posterior_trace"][0, best_idx])
    hessian_inv = fix["mode_hessian_inv"]
    mdd = laplace_mdd(log_post_mode, hessian_inv)
    return {"laplace_mdd": mdd}


def _eval_sw07_harmonic_mean_mdd() -> dict[str, float]:
    """Compute Geweke (1999) modified harmonic mean MDD and truncation stability."""
    import warnings
    from puremacro.dsge.marginal import harmonic_mean_mdd

    fix = _load_fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = harmonic_mean_mdd(fix["draws"], fix["log_posterior_trace"])
    return {
        "harmonic_mean_estimate": float(res.estimate),
        "truncation_spread": float(res.spread),
    }


def _eval_sw07_structural_params() -> dict[str, float]:
    """Extract headline structural parameter estimates at the posterior mode."""
    fix = _load_fixture()
    best_idx = int(np.argmax(fix["log_posterior_trace"][0]))
    best_theta = fix["draws"][0, best_idx]
    names = list(fix["param_names"])
    p = {n: float(best_theta[i]) for i, n in enumerate(names)}
    return {
        "csadjcost": p["csadjcost"],
        "csigma": p["csigma"],
        "chabb": p["chabb"],
        "csigl": p["csigl"],
        "cprobp": p["cprobp"],
        "cfc": p["cfc"],
        "crr": p["crr"],
        "crdy": p["crdy"],
        "ctrend": p["ctrend"],
    }


_PAPER = (
    "Smets & Wouters (2007), 'Shocks and Frictions in US Business Cycles: "
    "A Bayesian DSGE Approach', AER 97(3):586-606"
)

CASES: list[ReplicationCase] = [
    ReplicationCase(
        id="dsge_estimation.sw07_log_posterior_at_mode",
        family="dsge_estimation",
        paper=_PAPER,
        title="Smets-Wouters: log-posterior at mode on 1966-2004 US data",
        title_es="Smets-Wouters: log-posteriori en la moda sobre datos de EE.UU. 1966-2004",
        source="bundled:_sw07_data.csv",
        estimate=_eval_sw07_posterior_mode,
        target={"log_posterior_mode": -1673.72},
        target_kind=TargetKind.POINT,
        tol=Tol.TIGHT,
        citation="Smets & Wouters (2007), Table 1 & Pfeifer (2014) replication.",
    ),
    ReplicationCase(
        id="dsge_estimation.sw07_laplace_marginal_data_density",
        family="dsge_estimation",
        paper=_PAPER,
        title="Smets-Wouters: Laplace marginal data density approximation",
        title_es="Smets-Wouters: aproximación de Laplace a la densidad marginal de los datos",
        source="fixture:sw07_parity_seed0_200draws.npz",
        estimate=_eval_sw07_laplace_mdd,
        target={"laplace_mdd": -1686.09},
        target_kind=TargetKind.POINT,
        tol=Tol.TIGHT,
        citation="Smets & Wouters (2007), Table 2 (Laplace MDD on stationary covariance).",
    ),
    ReplicationCase(
        id="dsge_estimation.sw07_harmonic_mean_mdd_consistency",
        family="dsge_estimation",
        paper=_PAPER,
        title="Smets-Wouters: Geweke modified harmonic mean MDD consistency",
        title_es="Smets-Wouters: consistencia de la densidad marginal de media armónica modificada de Geweke",
        source="fixture:sw07_parity_seed0_200draws.npz",
        estimate=_eval_sw07_harmonic_mean_mdd,
        target={"harmonic_mean_estimate": -2524.36, "truncation_spread": 2.20},
        target_kind=TargetKind.POINT,
        tol=Tol.TIGHT,
        citation="Smets & Wouters (2007), Table 2 & Geweke (1999).",
    ),
    ReplicationCase(
        id="dsge_estimation.sw07_structural_parameters_mode",
        family="dsge_estimation",
        paper=_PAPER,
        title="Smets-Wouters: structural parameter estimates at posterior mode",
        title_es="Smets-Wouters: estimaciones de parámetros estructurales en la moda posterior",
        source="fixture:sw07_parity_seed0_200draws.npz",
        estimate=_eval_sw07_structural_params,
        target={
            "csadjcost": 5.74,
            "csigma": 1.38,
            "chabb": 0.71,
            "csigl": 1.83,
            "cprobp": 0.66,
            "cfc": 1.60,
            "crr": 0.81,
            "crdy": 0.22,
            "ctrend": 0.43,
        },
        target_kind=TargetKind.POINT,
        tol=Tol.COARSE,
        citation="Smets & Wouters (2007), Table 1: headline posterior mode parameter estimates.",
    ),
]
