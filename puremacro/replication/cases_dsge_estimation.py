"""Replication cases — Smets-Wouters (2007) Bayesian DSGE estimation family.

Paper: Smets, F. and Wouters, R. (2007), 'Shocks and Frictions in US Business
Cycles: A Bayesian DSGE Approach', American Economic Review, 97(3), 586-606.
Table and page numbers below refer to the ECB Working Paper 722 version
(February 2007, https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp722.pdf).

What is compared with what:

1. ``sw07_structural_parameters_mode`` — PUBLISHED target. The posterior
   mode of puremacro's SW07 model (``puremacro.dsge.smets_wouters``, whose
   impulse responses coincide with the bundled Pfeifer .mod) on the bundled
   1966Q1-2004Q4 dataset, against the *Mode* column of SW07 Table 1a (PDF
   p.35) and Table 1b (PDF p.36). The mode was found by an optimiser
   (``tools/build_sw07_data.py fixture``: L-BFGS-B from two starts, both
   reaching log posterior -822.0446, then Newton polishing to a max
   gradient of 2e-6) and is stored in the package fixture. The bundled data
   follow the SW07 data appendix but are today's FRED vintage, not SW's
   2006 files, so agreement is judged at ``Tol.COARSE``. The 13 checked
   parameters were fixed before the mode was computed: the nine headline
   parameters this case always checked plus the four markup-process
   parameters. Of the other 22 published modes all but price indexation
   ``cindp`` (0.32 vs 0.22) are also within 25%.
2. ``sw07_log_posterior_at_mode``, ``sw07_laplace_marginal_data_density``,
   ``sw07_harmonic_mean_mdd_consistency`` — puremacro REGRESSION values,
   not published numbers. SW07 report no log posterior at the mode, and
   their marginal likelihood (Table 2, PDF p.37: -905.8) is computed with a
   1956:1-1965:4 training sample that puremacro does not implement, so it is
   not a comparable target. These cases pin puremacro's own numbers on the
   bundled data so that any change to the model, data, priors or the
   marginal-likelihood estimators shows up; the log posterior and the
   Laplace approximation are recomputed live at the stored mode. The
   harmonic-mean estimate has NOT converged on the 200 stored draws
   (spread 2.66 > 1 log point); that case pins ``converged = 0`` and lets
   the estimator's warning through rather than presenting the number as a
   converged marginal likelihood.

The fixture ships as package data (``puremacro/replication/data/``), so the
cases also run from an installed wheel. It records the SHA-256 of the CSV it
was built on; a mismatch fails the cases instead of comparing stale numbers.
"""
from __future__ import annotations

import hashlib
from importlib import resources

import numpy as np

from ._model import ReplicationCase, TargetKind, Tol

_FIXTURE_NAME = "sw07_parity_seed0_200draws.npz"

_REGRESSION_NOTE = (
    "puremacro regression value, not published: computed by puremacro on the "
    "bundled FRED rebuild of the SW07 data (tools/build_sw07_data.py). SW07 "
    "report no comparable number; their marginal likelihood (-905.8, ECB WP "
    "722 Table 2) uses a 1956:1-1965:4 training sample that is not implemented."
)

# SW07 posterior MODE column (not the Mean column), ECB WP 722 Table 1a
# (PDF p.35) and Table 1b (PDF p.36), mapped to the .mod parameter names.
# Mean column, for reference: csadjcost 5.74, csigma 1.38, csigl 1.83,
# cprobp 0.66, cfc 1.60, cprobw 0.70, crpi 2.04.
_TABLE1_MODE = {
    "csadjcost": 5.48,   # phi, investment adjustment cost
    "csigma": 1.39,      # sigma_c
    "chabb": 0.71,       # habit (labelled h in Table 1a)
    "cprobw": 0.73,      # xi_w
    "csigl": 1.92,       # sigma_L
    "cprobp": 0.65,      # xi_p
    "cindw": 0.59,       # iota_w
    "cindp": 0.22,       # iota_p
    "czcap": 0.54,       # psi
    "cfc": 1.61,         # Phi
    "crpi": 2.03,        # r_pi
    "crr": 0.81,         # rho
    "cry": 0.08,         # r_y
    "crdy": 0.22,        # r_dy
    "constepinf": 0.81,  # pi-bar
    "constebeta": 0.16,  # 100(beta^-1 - 1)
    "ctrend": 0.43,      # gamma-bar
    "calfa": 0.19,       # alpha
    "crhoa": 0.95, "crhob": 0.18, "crhog": 0.97, "crhoqs": 0.71,
    "crhoms": 0.12, "crhopinf": 0.90, "crhow": 0.97,
    "cmap": 0.74,        # mu_p
    "cmaw": 0.88,        # mu_w
    "cgy": 0.52,         # rho_ga
    # shock standard deviations sigma_a, sigma_b, sigma_g, sigma_I, sigma_r,
    # sigma_p, sigma_w (Table 1b)
    "ea": 0.45, "eb": 0.24, "eg": 0.52, "eqs": 0.45, "em": 0.24,
    "epinf": 0.14, "ew": 0.24,
}

# The parameters the mode case checks: the nine headline structural
# parameters the case has always checked, plus the four markup-process
# parameters whose treatment the 2026-09-30 fix changed.
_MODE_CASE_PARAMS = (
    "csadjcost", "csigma", "chabb", "csigl", "cprobp", "cfc", "crr", "crdy",
    "ctrend",
    "crhopinf", "cmap", "crhow", "cmaw",
)


def _load_fixture() -> dict[str, np.ndarray]:
    """The SW07 posterior fixture from package data (works from a wheel)."""
    res = resources.files("puremacro.replication.data").joinpath(_FIXTURE_NAME)
    with res.open("rb") as fh, np.load(fh) as z:
        fix = {k: z[k] for k in z.files}
    csv = resources.files("puremacro.dsge").joinpath("_sw07_data.csv").read_bytes()
    # Line endings normalised, so a CRLF checkout (git autocrlf) still matches.
    sha = hashlib.sha256(csv.replace(b"\r\n", b"\n")).hexdigest()
    if str(fix["data_sha256"]) != sha:
        raise ValueError(
            "SW07 replication fixture was built on a different _sw07_data.csv "
            f"(fixture {str(fix['data_sha256'])[:12]}, bundled {sha[:12]}); "
            "rebuild it with `python tools/build_sw07_data.py fixture`."
        )
    return fix


def _log_posterior_at(theta: np.ndarray) -> float:
    from puremacro.dsge.estimate import _make_neg_log_posterior
    from puremacro.dsge.sw07_estimate import _FIXED_PARAMS, _load_bundled_data
    from puremacro.dsge.sw07_observation import OBSERVED_VARS, make_state_space
    from puremacro.dsge.sw07_priors import PRIORS, param_names

    y = _load_bundled_data()[list(OBSERVED_VARS)].to_numpy()
    nlp = _make_neg_log_posterior(y, make_state_space, PRIORS, param_names(), _FIXED_PARAMS)
    return -float(nlp(np.asarray(theta, dtype=float)))


def _eval_sw07_posterior_mode() -> dict[str, float]:
    """Log posterior (log likelihood + log prior) at the stored mode, recomputed live."""
    fix = _load_fixture()
    return {"log_posterior_mode": _log_posterior_at(fix["mode_values"])}


def _eval_sw07_laplace_mdd() -> dict[str, float]:
    """Laplace log marginal data density: live log posterior at the mode plus
    the stored inverse Hessian of the negative log posterior there."""
    from puremacro.dsge.marginal import laplace_mdd

    fix = _load_fixture()
    lp_mode = _log_posterior_at(fix["mode_values"])
    return {"laplace_mdd": float(laplace_mdd(lp_mode, fix["mode_hessian_inv"]))}


def _eval_sw07_harmonic_mean_mdd() -> dict[str, float]:
    """Geweke (1999) modified harmonic mean on the stored 200 posterior draws.

    With 200 draws of 36 parameters the estimate moves by 2.66 log points
    across the truncation levels 0.1-0.9, above ``harmonic_mean_mdd``'s
    one-log-point convergence threshold, so the estimator reports
    ``converged=False`` and issues its UserWarning. The warning is left to
    propagate (through 4.3.0 it was suppressed here) and ``converged`` is
    returned as a metric, so the case pins the non-convergence explicitly.
    """
    from puremacro.dsge.marginal import harmonic_mean_mdd

    fix = _load_fixture()
    res = harmonic_mean_mdd(fix["draws"], fix["log_posterior_trace"])
    return {
        "harmonic_mean_estimate": float(res.estimate),
        "truncation_spread": float(res.spread),
        "converged": float(res.converged),
    }


def _eval_sw07_structural_params() -> dict[str, float]:
    """Optimised posterior mode (stored in the fixture) for the checked parameters."""
    fix = _load_fixture()
    names = [str(n) for n in fix["param_names"]]
    mode = {n: float(v) for n, v in zip(names, fix["mode_values"])}
    return {k: mode[k] for k in _MODE_CASE_PARAMS}


def _authors_mode_reference() -> dict:
    """The authors' posterior mode and Dynare 8's log posterior there (package data)."""
    import json

    res = resources.files("puremacro.dsge").joinpath("_references/sw07_usmodel_mode.json")
    return json.loads(res.read_text(encoding="utf-8"))


def _eval_sw07_log_posterior_at_authors_mode() -> dict[str, float]:
    """puremacro's log posterior at the authors' own mode, authors' data, AER .mod options.

    ``first_obs=71`` (1965Q1), ``presample=4`` and ``lik_init=2`` of the
    replication ``.mod``, so the likelihood is that of 1966Q1-2004Q4 with the
    four 1965 quarters initialising the filter from the diffuse start."""
    from puremacro.dsge.sw07_marginal import sw07_log_posterior

    ref = _authors_mode_reference()
    lp = sw07_log_posterior(ref["mode"], dataset="authors", first_obs="1965Q1", last_obs="2004Q4",
                            presample=4, lik_init="diffuse")
    return {"log_posterior_at_authors_mode": float(lp)}


_PAPER = (
    "Smets & Wouters (2007), 'Shocks and Frictions in US Business Cycles: "
    "A Bayesian DSGE Approach', AER 97(3):586-606"
)
_SOURCE = "bundled:_sw07_data.csv + package:replication/data/" + _FIXTURE_NAME

CASES: list[ReplicationCase] = [
    ReplicationCase(
        id="dsge_estimation.sw07_log_posterior_at_authors_mode_vs_dynare",
        family="dsge_estimation",
        paper=_PAPER,
        title="Smets-Wouters: log posterior at the authors' mode on the authors' data equals Dynare 8's",
        title_es="Smets-Wouters: log-posteriori en la moda de los autores sobre sus datos igual a la de Dynare 8",
        source="package:dsge/_references/sw07_usmodel_mode.json + bundled:_sw07_usmodel_data.csv",
        estimate=_eval_sw07_log_posterior_at_authors_mode,
        target={"log_posterior_at_authors_mode": -841.4621},
        target_kind=TargetKind.POINT,
        tol=Tol.TIGHT,
        citation="Dynare 8 (snapshot 8-2026-05-26-1803, MATLAB R2026a, 2026-10-03) on Pfeifer's "
                 "Smets_Wouters_2007.mod with the AER replication data and mode file: 'Initial value "
                 "of the log posterior (or likelihood): -841.4621'; Laplace log data density -923.05.",
        notes="External-software check of the SW07 model, data handling, presample and diffuse "
              "initialisation at one and the same parameter vector: puremacro gives -840.81 (0.65 log "
              "points, 0.08%). With Dynare's own Hessian the Laplace density is -922.40 against "
              "Dynare's -923.05. The paper's Table 2 figure, -905.8, is reproduced by neither: see "
              "docs/replication.md.",
    ),
    ReplicationCase(
        id="dsge_estimation.sw07_log_posterior_at_mode",
        family="dsge_estimation",
        paper=_PAPER,
        title="Smets-Wouters: log posterior at the optimised mode (regression value, not published)",
        title_es="Smets-Wouters: log-posteriori en la moda optimizada (valor de regresión, no publicado)",
        source=_SOURCE,
        estimate=_eval_sw07_posterior_mode,
        target={"log_posterior_mode": -822.04},
        target_kind=TargetKind.POINT,
        tol=Tol.TIGHT,
        citation=_REGRESSION_NOTE,
        notes="Log likelihood (Kalman filter from the unconditional covariance) "
              "plus log prior at the stored mode, recomputed live.",
    ),
    ReplicationCase(
        id="dsge_estimation.sw07_laplace_marginal_data_density",
        family="dsge_estimation",
        paper=_PAPER,
        title="Smets-Wouters: Laplace marginal data density (regression value, not published)",
        title_es="Smets-Wouters: densidad marginal de Laplace (valor de regresión, no publicado)",
        source=_SOURCE,
        estimate=_eval_sw07_laplace_mdd,
        target={"laplace_mdd": -902.83},
        target_kind=TargetKind.POINT,
        tol=Tol.TIGHT,
        citation=_REGRESSION_NOTE,
        notes="log p(Y) ~ log p(Y, th*) + (d/2) log(2 pi) + (1/2) log|Sigma*| "
              "with Sigma* the inverse Hessian of -log posterior at the mode.",
    ),
    ReplicationCase(
        id="dsge_estimation.sw07_harmonic_mean_mdd_consistency",
        family="dsge_estimation",
        paper=_PAPER,
        title="Smets-Wouters: Geweke modified harmonic mean MDD, NOT converged on "
              "200 draws (regression value, not published)",
        title_es="Smets-Wouters: media armónica modificada de Geweke, NO convergida con "
                 "200 extracciones (valor de regresión, no publicado)",
        source=_SOURCE,
        estimate=_eval_sw07_harmonic_mean_mdd,
        target={"harmonic_mean_estimate": -908.03, "truncation_spread": 2.66,
                "converged": 0.0},
        target_kind=TargetKind.POINT,
        tol=Tol.TIGHT,
        citation=_REGRESSION_NOTE,
        notes="Geweke (1999) estimator over truncation levels 0.1..0.9 on the "
              "fixture's 200 thinned random-walk Metropolis draws. It has NOT "
              "converged by puremacro's own rule: the spread across truncations "
              "(2.66 log points) exceeds marginal._SPREAD_TOL = 1.0, so "
              "harmonic_mean_mdd warns (the warning is not suppressed) and the "
              "case pins converged = 0. The -908.03 is the estimator's output on "
              "these draws, not a usable marginal likelihood; use the Laplace "
              "case. Despite the id, the case checks no 'consistency': it pins "
              "the lack of it. More draws would be needed for convergence, which "
              "the offline suite cannot afford; if a rebuilt fixture converges, "
              "the case fails and its target must be revisited.",
    ),
    ReplicationCase(
        id="dsge_estimation.sw07_structural_parameters_mode",
        family="dsge_estimation",
        paper=_PAPER,
        title="Smets-Wouters: posterior mode vs the published Table 1a/1b Mode column",
        title_es="Smets-Wouters: moda posterior frente a la columna Mode publicada de las Tablas 1a/1b",
        source=_SOURCE,
        estimate=_eval_sw07_structural_params,
        target={k: _TABLE1_MODE[k] for k in _MODE_CASE_PARAMS},
        target_kind=TargetKind.POINT,
        tol=Tol.COARSE,
        citation="Smets & Wouters (2007), ECB WP 722, Table 1a (PDF p.35) and "
                 "Table 1b (PDF p.36), posterior Mode column.",
        notes="Optimiser mode of puremacro's SW07 model on the bundled FRED "
              "rebuild of the SW07 dataset (current vintage, not SW's 2006 "
              "files), hence Tol.COARSE.",
    ),
]
