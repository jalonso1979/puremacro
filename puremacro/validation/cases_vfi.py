"""Validation cases for the ``vfi`` subsystem (discretization + dynamic-programming
solvers).

Every reference here is INDEPENDENT of puremacro:

* ANALYTICAL — AR(1) discretizations (Tauchen 1986, Rouwenhorst 1995) must
  reproduce the closed-form stationary moments of ``x' = rho*x + eps``
  (mean 0, variance ``sigma^2/(1-rho^2)``, lag-1 autocorrelation ``rho``);
  the stochastic-growth value-function solution must reproduce the
  Brock--Mirman (1972) closed-form capital policy under log utility and full
  depreciation.
* SCIPY — the Markov stationary distribution is checked against the
  normalized left eigenvector of the transition matrix computed live by
  ``scipy.linalg.eig`` (a different code base, not puremacro).
* INTERNAL — the Endogenous Grid Method policy must agree with the
  value-function-iteration policy for the *same* income-fluctuation problem
  (two methods, one model).

No reference re-exports puremacro, so none of these checks is circular. All
estimator imports live INSIDE the compute callables to keep module import
pyodide-light (numpy + stdlib only at module top).
"""
from __future__ import annotations

import numpy as np

from ._model import Mechanism, Tol, ValidationCase

# --------------------------------------------------------------------------- #
# Seeded demo data (local to this module — does NOT touch the shared fixtures) #
# --------------------------------------------------------------------------- #

# A single AR(1) calibration reused by the discretization cases, so the
# puremacro chain and its analytical reference share byte-identical parameters.
_AR1 = {"rho": 0.7, "sigma": 0.25}

# Income-fluctuation calibration shared by the EGM-vs-VFI cross-check.
_IFP = {"beta": 0.95, "gamma": 2.0, "r": 0.03, "rho": 0.7, "sigma": 0.20}

# Brock--Mirman (stochastic growth, log utility, full depreciation) calibration.
_BM = {"alpha": 0.30, "beta": 0.95, "rho": 0.6, "sigma": 0.10, "n_z": 5}


def _ar1_uncond_var(rho: float, sigma: float) -> float:
    """Unconditional variance of the AR(1) ``x' = rho*x + eps``, eps~N(0,sigma^2)."""
    return sigma**2 / (1.0 - rho**2)


def _chain_moments(grid: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Exact stationary [mean, variance, lag-1 autocorrelation] of a finite
    Markov chain — computed from the ergodic distribution and the transition
    matrix (no simulation), so the moments are deterministic functions of
    ``(grid, P)``.
    """
    from puremacro.vfi.discretize import markov_stationary

    pi = np.asarray(markov_stationary(P), dtype=float)
    g = np.asarray(grid, dtype=float)
    mean = float(pi @ g)
    dev = g - mean
    var = float(pi @ (dev**2))
    cond = np.asarray(P, dtype=float) @ dev          # E[x' - mean | state i]
    autocov = float(pi @ (dev * cond))               # Cov(x_t, x_{t+1})
    return np.array([mean, var, autocov / var])


# --------------------------------------------------------------------------- #
# Compute callables                                                           #
# --------------------------------------------------------------------------- #


def _rouwenhorst_moments() -> dict:
    from puremacro.vfi.discretize import rouwenhorst

    grid, P = rouwenhorst(7, _AR1["rho"], _AR1["sigma"])
    return {"moments": _chain_moments(grid, P)}


def _tauchen_moments() -> dict:
    from puremacro.vfi.discretize import tauchen

    # Tauchen converges to the AR(1) moments as the grid refines; 31 nodes put
    # the variance/autocorrelation well inside the NUMERIC band.
    grid, P = tauchen(31, _AR1["rho"], _AR1["sigma"])
    return {"moments": _chain_moments(grid, P)}


def _markov_stationary_dist() -> dict:
    from puremacro.vfi.discretize import markov_stationary, tauchen

    _, P = tauchen(6, 0.5, 0.2)
    return {"pi": np.asarray(markov_stationary(P), dtype=float)}


def _markov_stationary_ref() -> dict:
    """Reference: normalized left eigenvector of P for eigenvalue 1, via SciPy.

    ``scipy.linalg.eig`` is an entirely separate code base (LAPACK-backed); the
    left eigenvector of ``P`` is the right eigenvector of ``P.T``.
    """
    from scipy.linalg import eig

    from puremacro.vfi.discretize import tauchen

    _, P = tauchen(6, 0.5, 0.2)
    w, V = eig(np.asarray(P, dtype=float).T)
    idx = int(np.argmin(np.abs(w - 1.0)))            # the eigenvalue-1 mode
    v = np.real(V[:, idx])
    return {"pi": v / v.sum()}                        # normalize to a probability vector


def _brock_mirman_policy() -> dict:
    """VFI capital policy for the stochastic growth model with log utility and
    FULL depreciation (delta=1), evaluated where the closed form lies interior
    to the grid.

    Model: ``c = exp(z) * k^alpha - k'`` (delta=1), ``u(c)=log c``, AR(1) ``z``.
    """
    from puremacro.vfi.discretize import rouwenhorst
    from puremacro.vfi.problem import VFIProblem

    alpha, beta = _BM["alpha"], _BM["beta"]
    z_log, P = rouwenhorst(int(_BM["n_z"]), _BM["rho"], _BM["sigma"])
    z_lev = np.exp(z_log)

    # Deterministic steady state of k' = alpha*beta*z*k^alpha at z=1.
    k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
    k_grid = np.linspace(0.3 * k_ss, 2.5 * k_ss, 1500)

    def return_fn(kp, k, z, xp=np):
        c = xp.exp(z) * k**alpha - kp            # full depreciation: c = y - k'
        u = xp.log(xp.maximum(c, 1e-12))
        return xp.where(c > 0.0, u, -np.inf)

    prob = VFIProblem(
        a_grid=k_grid, z_grid=z_log, P_z=P, return_fn=return_fn, beta=beta,
        options=dict(tol=1e-12, n_howard=60, max_iter=40000),
    )
    sol = prob.solve("numpy")
    k_pol = k_grid[sol.policy_aprime]                       # (n_k, n_z) grid policy

    # Brock--Mirman closed form; compare only where it is interior to the grid
    # (grid-constrained VFI cannot reproduce policy outside the grid range).
    k_closed = alpha * beta * z_lev[None, :] * k_grid[:, None] ** alpha
    interior = (k_closed > k_grid[0] + 1e-12) & (k_closed < k_grid[-1] - 1e-12)
    return {"k_policy": k_pol[interior]}


def _brock_mirman_ref() -> dict:
    """Brock--Mirman closed-form capital policy on the interior of the grid."""
    from puremacro.vfi.discretize import rouwenhorst

    alpha, beta = _BM["alpha"], _BM["beta"]
    z_log, _ = rouwenhorst(int(_BM["n_z"]), _BM["rho"], _BM["sigma"])
    z_lev = np.exp(z_log)
    k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
    k_grid = np.linspace(0.3 * k_ss, 2.5 * k_ss, 1500)
    k_closed = alpha * beta * z_lev[None, :] * k_grid[:, None] ** alpha
    interior = (k_closed > k_grid[0] + 1e-12) & (k_closed < k_grid[-1] - 1e-12)
    return {"k_policy": k_closed[interior]}


def _ifp_problem():
    """Shared income-fluctuation primitives (numpy arrays); deterministic."""
    from puremacro.vfi.discretize import rouwenhorst

    z_grid, P = rouwenhorst(5, _IFP["rho"], _IFP["sigma"])
    income = np.exp(z_grid)                          # y(z) > 0
    a_grid = np.linspace(0.0, 50.0, 2500)            # fine grid → small VFI discretization error
    return a_grid, z_grid, income, P


def _egm_consumption() -> dict:
    from puremacro.vfi.egm import solve_egm

    a_grid, z_grid, income, P = _ifp_problem()
    sol = solve_egm(
        a_grid, z_grid, income, P,
        beta=_IFP["beta"], r=_IFP["r"], gamma=_IFP["gamma"],
    )
    # Interior region: away from the borrowing constraint and the top-grid clamp,
    # where the continuous-EGM and grid-VFI policies are directly comparable.
    mask = (a_grid > 3.0) & (a_grid < 40.0)
    return {"c": np.asarray(sol.c, dtype=float)[mask]}


def _vfi_consumption() -> dict:
    """VFI consumption policy for the SAME income-fluctuation problem EGM solves:
    ``c + a' = (1+r) a + y(z)``, CRRA utility, borrowing constraint ``a' >= a_min``.
    """
    from puremacro.vfi.problem import VFIProblem

    a_grid, z_grid, income, P = _ifp_problem()
    beta, gamma, r = _IFP["beta"], _IFP["gamma"], _IFP["r"]

    def return_fn(ap, a, z, xp=np):
        c = xp.exp(z) + (1.0 + r) * a - ap
        u = (xp.maximum(c, 1e-12) ** (1.0 - gamma) - 1.0) / (1.0 - gamma)
        return xp.where(c > 0.0, u, -np.inf)

    prob = VFIProblem(
        a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=return_fn, beta=beta,
        options=dict(tol=1e-11, n_howard=50, max_iter=30000),
    )
    sol = prob.solve("numpy")
    coh = (1.0 + r) * a_grid[:, None] + income[None, :]      # cash-on-hand
    c_vfi = coh - a_grid[sol.policy_aprime]                   # budget-implied consumption
    mask = (a_grid > 3.0) & (a_grid < 40.0)
    return {"c": c_vfi[mask]}


CASES: list[ValidationCase] = [
    ValidationCase(
        id="vfi.rouwenhorst_ar1_moments",
        subsystem="vfi",
        title="Rouwenhorst chain reproduces AR(1) stationary moments",
        title_es="La cadena de Rouwenhorst reproduce los momentos estacionarios del AR(1)",
        mechanism=Mechanism.ANALYTICAL,
        compute=_rouwenhorst_moments,
        reference=lambda: {
            "moments": np.array(
                [0.0, _ar1_uncond_var(_AR1["rho"], _AR1["sigma"]), _AR1["rho"]]
            )
        },
        tol=Tol.TIGHT,
        citation=(
            "Rouwenhorst (1995), in Cooley (ed.), Frontiers of Business Cycle "
            "Research, ch. 10: the n-state chain matches the AR(1) mean, "
            "variance sigma^2/(1-rho^2) and autocorrelation rho exactly for any n."
        ),
    ),
    ValidationCase(
        id="vfi.tauchen_ar1_moments",
        subsystem="vfi",
        title="Tauchen chain approximates AR(1) stationary moments",
        title_es="La cadena de Tauchen aproxima los momentos estacionarios del AR(1)",
        mechanism=Mechanism.ANALYTICAL,
        compute=_tauchen_moments,
        reference=lambda: {
            "moments": np.array(
                [0.0, _ar1_uncond_var(_AR1["rho"], _AR1["sigma"]), _AR1["rho"]]
            )
        },
        tol=Tol.NUMERIC,
        citation=(
            "Tauchen (1986, Economics Letters 20:177-181): the finite-state "
            "approximation reproduces the AR(1) stationary mean 0, variance "
            "sigma^2/(1-rho^2) and lag-1 autocorrelation rho as the grid refines."
        ),
    ),
    ValidationCase(
        id="vfi.markov_stationary_left_eigvec",
        subsystem="vfi",
        title="Markov stationary distribution equals SciPy left eigenvector",
        title_es="La distribucion estacionaria de Markov coincide con el autovector izquierdo de SciPy",
        mechanism=Mechanism.SCIPY,
        compute=_markov_stationary_dist,
        reference=_markov_stationary_ref,
        tol=Tol.TIGHT,
        citation=(
            "Stationary distribution = normalized left eigenvector of P for "
            "eigenvalue 1 (Perron-Frobenius; Stokey & Lucas 1989, ch. 11). "
            "Reference computed live by scipy.linalg.eig (SciPy 1.x, LAPACK)."
        ),
    ),
    ValidationCase(
        id="vfi.neoclassical_brock_mirman_policy",
        subsystem="vfi",
        title="Stochastic-growth VFI policy matches Brock-Mirman closed form",
        title_es="La politica de VFI del crecimiento estocastico coincide con la forma cerrada de Brock-Mirman",
        mechanism=Mechanism.ANALYTICAL,
        compute=_brock_mirman_policy,
        reference=_brock_mirman_ref,
        tol=Tol.NUMERIC,
        citation=(
            "Brock & Mirman (1972, J. Econ. Theory 4:479-513); Stokey & Lucas "
            "(1989, sec. 2.2): log utility + Cobb-Douglas + full depreciation "
            "give the exact policy k' = alpha*beta*exp(z)*k^alpha."
        ),
    ),
    ValidationCase(
        id="vfi.egm_matches_vfi_policy",
        subsystem="vfi",
        title="EGM consumption policy matches VFI on the same income-fluctuation problem",
        title_es="La politica de consumo del MGE coincide con VFI en el mismo problema de fluctuacion del ingreso",
        mechanism=Mechanism.INTERNAL,
        compute=_egm_consumption,
        reference=_vfi_consumption,
        tol=Tol.NUMERIC,
        citation=(
            "Cross-method identity: the Endogenous Grid Method (Carroll 2006, "
            "Economics Letters 91:312-320) and discrete value-function iteration "
            "solve the same Euler equation, so their policies agree up to the "
            "VFI grid resolution."
        ),
    ),
]
