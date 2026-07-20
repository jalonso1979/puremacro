"""Validation cases for the ``forecast`` subsystem.

Probabilistic-forecast scoring (CRPS, PIT calibration) and equal-predictive-
ability testing (Diebold-Mariano). Every reference here is INDEPENDENT of the
puremacro code path under test:

* ``crps_gaussian`` is checked against the closed form re-derived with stdlib
  ``math.erf`` (NOT scipy ``norm``), plus the analytic value at z = 0,
  ``sqrt(2/pi) - 1/sqrt(pi)`` (Gneiting & Raftery 2007). ANALYTICAL / TIGHT.
* ``crps_ensemble`` (the Gneiting-Raftery fair estimator) is checked against the
  Gaussian closed form on a large, well-calibrated Gaussian draw — a cross-method
  agreement between two unrelated algorithms. INTERNAL / NUMERIC.
* ``pit`` + ``pit_uniformity_test`` are checked by the calibration identity:
  the KS test is NOT rejected for a correctly specified Gaussian forecaster and
  IS rejected for an over-confident (too-narrow) one. INTERNAL / EXACT (indicators).
* ``diebold_mariano`` is checked by two construction identities: the loss-
  differential sign is positive when forecast 1 is the noisier of two nested
  forecasts, and the statistic is NaN under a degenerate perfect tie (zero
  long-run variance). INTERNAL / EXACT (indicators).

No reference packages are imported — references are an analytical closed form
(stdlib only) or puremacro cross-method / self-consistency checks. Hence there
is no golden JSON and no drift-guard for this subsystem.
"""
from __future__ import annotations

import math

import numpy as np

from ._model import Mechanism, Tol, ValidationCase


# ---------------------------------------------------------------------------
# Seeded demo data (local to this module; does NOT touch shared _fixtures.py).
# ---------------------------------------------------------------------------
def _crps_points() -> dict:
    """A small, fixed bag of (y, mu, sigma) triples spanning sign(z) and scale."""
    y = np.array([0.3, -1.2, 2.5, 0.0, 1.0, -0.7])
    mu = np.array([0.0, 0.5, 2.0, 0.0, -0.5, -0.7])
    sigma = np.array([1.0, 2.0, 0.7, 1.5, 1.0, 0.4])
    return {"y": y, "mu": mu, "sigma": sigma}


def _ensemble_demo() -> dict:
    """A well-calibrated Gaussian predictive with a large ensemble per period.

    Both ``y`` and the ensemble are drawn from N(mu, sigma); averaging the score
    over many periods drives the Monte-Carlo error of the fair estimator well
    below the NUMERIC tolerance, independent of the particular seed.
    """
    rng = np.random.default_rng(20260531)
    T, M = 400, 2000
    mu = np.full(T, 1.3)
    sigma = np.full(T, 2.0)
    y = mu + sigma * rng.standard_normal(T)
    samples = mu[:, None] + sigma[:, None] * rng.standard_normal((T, M))
    return {"y": y, "mu": mu, "sigma": sigma, "samples": samples}


def _pit_calibration_demo() -> dict:
    """Truth ~ N(0,1); a correct forecaster and an over-confident N(0, 0.5) one."""
    rng = np.random.default_rng(20260601)
    y = rng.standard_normal(600)
    return {"y": y, "sigma_correct": 1.0, "sigma_wrong": 0.5}


def _dm_demo() -> dict:
    """Two nested forecasts where #1 is strictly noisier than #2 (e1 = e2 + extra)."""
    rng = np.random.default_rng(20260602)
    n = 300
    e2 = 0.3 * rng.standard_normal(n)
    e1 = e2 + 1.0 * rng.standard_normal(n)
    return {"e1": e1, "e2": e2}


# ---------------------------------------------------------------------------
# Independent reference: closed-form Gaussian CRPS via stdlib math.erf only.
# ---------------------------------------------------------------------------
def _crps_gaussian_stdlib(y, mu, sigma) -> np.ndarray:
    """CRPS of N(mu, sigma) computed with the standard-normal CDF/PDF built from
    ``math.erf`` — no scipy, so it is independent of puremacro's scipy path."""
    y = np.asarray(y, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    z = (y - mu) / sigma
    Phi = np.array([0.5 * (1.0 + math.erf(zz / math.sqrt(2.0))) for zz in z])
    phi = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


# ---------------------------------------------------------------------------
# compute callables (puremacro imports live INSIDE — module stays pyodide-light)
# ---------------------------------------------------------------------------
def _crps_gaussian_pm() -> dict:
    from puremacro.forecast.density import crps_gaussian

    d = _crps_points()
    return {"crps": np.asarray(crps_gaussian(d["y"], d["mu"], d["sigma"]), dtype=float)}


def _crps_gaussian_at_zero() -> dict:
    from puremacro.forecast.density import crps_gaussian

    # z = 0 (y == mu, sigma == 1): CRPS collapses to the published constant.
    return {"crps0": float(crps_gaussian(np.zeros(1), np.zeros(1), np.ones(1))[0])}


def _crps_ensemble_mean() -> dict:
    from puremacro.forecast.density import crps_ensemble

    d = _ensemble_demo()
    ens = np.asarray(crps_ensemble(d["y"], d["samples"]), dtype=float)
    return {"mean_crps": float(ens.mean())}


def _crps_ensemble_ref() -> dict:
    # Closed-form Gaussian CRPS (stdlib-erf), averaged over the same draws.
    d = _ensemble_demo()
    cf = _crps_gaussian_stdlib(d["y"], d["mu"], d["sigma"])
    return {"mean_crps": float(cf.mean())}


def _pit_calibration() -> dict:
    from scipy.stats import norm

    from puremacro.forecast.density import pit, pit_uniformity_test

    d = _pit_calibration_demo()
    y = d["y"]
    pit_correct = pit(y, lambda yt, t: norm.cdf(yt, 0.0, d["sigma_correct"]))
    pit_wrong = pit(y, lambda yt, t: norm.cdf(yt, 0.0, d["sigma_wrong"]))
    p_correct = pit_uniformity_test(pit_correct)["p_value"]
    p_wrong = pit_uniformity_test(pit_wrong)["p_value"]
    # Indicators: a calibrated forecaster is NOT rejected, an over-confident one IS.
    return {
        "reject_correct_at_5pct": float(p_correct < 0.05),
        "reject_wrong_at_5pct": float(p_wrong < 0.05),
    }


def _dm_sign_and_tie() -> dict:
    from puremacro.forecast.compare import diebold_mariano

    d = _dm_demo()
    r = diebold_mariano(d["e1"], d["e2"], h=1, loss="mse")
    tie = diebold_mariano(d["e1"], d["e1"], h=1, loss="mse")  # perfect tie
    return {
        # e1 is the noisier forecast => E[L1 - L2] > 0 => DM statistic > 0.
        "sign_positive": float(r["stat"] > 0.0),
        # Degenerate tie: loss differential is identically zero => stat is NaN.
        "tie_is_nan": float(math.isnan(tie["stat"])),
    }


CASES: list[ValidationCase] = [
    ValidationCase(
        id="forecast.crps_gaussian_closed_form",
        subsystem="forecast",
        title="Gaussian CRPS matches the closed form (stdlib erf)",
        title_es="El CRPS gaussiano coincide con la forma cerrada (erf de la biblioteca estandar)",
        mechanism=Mechanism.ANALYTICAL,
        compute=_crps_gaussian_pm,
        reference=lambda: {
            "crps": _crps_gaussian_stdlib(
                _crps_points()["y"], _crps_points()["mu"], _crps_points()["sigma"]
            )
        },
        tol=Tol.TIGHT,
        citation=(
            "Gneiting & Raftery (2007, JASA 102:359-378), eq. (5): "
            "CRPS(N(mu,sigma), y) = sigma[z(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi)], z=(y-mu)/sigma."
        ),
    ),
    ValidationCase(
        id="forecast.crps_gaussian_value_at_zero",
        subsystem="forecast",
        title="Gaussian CRPS at z=0 equals sqrt(2/pi) - 1/sqrt(pi)",
        title_es="El CRPS gaussiano en z=0 es igual a sqrt(2/pi) - 1/sqrt(pi)",
        mechanism=Mechanism.ANALYTICAL,
        compute=_crps_gaussian_at_zero,
        reference=lambda: {"crps0": math.sqrt(2.0 / math.pi) - 1.0 / math.sqrt(math.pi)},
        tol=Tol.TIGHT,
        citation=(
            "Gneiting & Raftery (2007, JASA 102:359-378), eq. (5) at y=mu, sigma=1: "
            "CRPS = 2phi(0) - 1/sqrt(pi) = sqrt(2/pi) - 1/sqrt(pi)."
        ),
    ),
    ValidationCase(
        id="forecast.crps_ensemble_vs_gaussian",
        subsystem="forecast",
        title="Fair ensemble CRPS converges to the Gaussian closed form",
        title_es="El CRPS de conjunto (estimador insesgado) converge a la forma cerrada gaussiana",
        mechanism=Mechanism.INTERNAL,
        compute=_crps_ensemble_mean,
        reference=_crps_ensemble_ref,
        tol=Tol.NUMERIC,
        citation=(
            "Gneiting & Raftery (2007, JASA 102:359-378), eq. (27) fair estimator; "
            "the M-member estimate of E|X-y| - 1/2 E|X-X'| is consistent for the "
            "Gaussian closed form when the ensemble is drawn from N(mu,sigma)."
        ),
    ),
    ValidationCase(
        id="forecast.pit_calibration_identity",
        subsystem="forecast",
        title="PIT-KS rejects an over-confident forecaster, not a calibrated one",
        title_es="La prueba KS sobre la PIT rechaza al pronosticador sobreconfiado, no al calibrado",
        mechanism=Mechanism.INTERNAL,
        compute=_pit_calibration,
        reference=lambda: {"reject_correct_at_5pct": 0.0, "reject_wrong_at_5pct": 1.0},
        tol=Tol.EXACT,
        citation=(
            "Diebold, Gunther & Tay (1998, Int. Econ. Rev. 39:863-883): correctly "
            "specified one-step densities have iid Uniform(0,1) PITs; "
            "Kolmogorov-Smirnov uniformity test."
        ),
    ),
    ValidationCase(
        id="forecast.diebold_mariano_sign_and_tie",
        subsystem="forecast",
        title="DM statistic is positive when forecast 1 is noisier; NaN under a tie",
        title_es="El estadistico DM es positivo cuando el pronostico 1 es mas ruidoso; NaN ante un empate",
        mechanism=Mechanism.INTERNAL,
        compute=_dm_sign_and_tie,
        reference=lambda: {"sign_positive": 1.0, "tie_is_nan": 1.0},
        tol=Tol.EXACT,
        citation=(
            "Diebold & Mariano (1995, J. Bus. Econ. Stat. 13:253-263): the test "
            "statistic is d_bar / sqrt(LRV/T) on the loss differential d_t = L(e1_t) - L(e2_t); "
            "sign tracks which forecast has larger expected loss."
        ),
    ),
]
