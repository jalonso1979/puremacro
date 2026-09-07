"""Posterior-mode search — puremacro's ``mode_compute`` menu.

``estimate_dsge`` used to do one bounded L-BFGS-B run, and the 2.5.0 audit
found it returning ``initial_params`` **bit-identically** while reporting
``converged_mle=True``: L-BFGS-B cannot finite-difference across ``+inf``, so
any trial step into the infeasible region stopped it at iteration 1. That was
fixed by giving the optimiser a large finite penalty instead of ``+inf``. What
remains is that one local gradient method from one starting point is a thin
basis for a 36-parameter posterior, and this module supplies the alternatives.

============  ==========================================================
``method``    algorithm
============  ==========================================================
``lbfgs``     bounded L-BFGS-B — today's behaviour, and still the default
              in :func:`~puremacro.dsge.estimate_dsge`
``simplex``   Nelder-Mead with restarts; derivative-free, so it is unmoved
              by the kinks a penalised region introduces
``csminwel``  Sims's quasi-Newton with the perturbed-Hessian retry
``cmaes``     covariance-matrix adaptation evolution strategy; a population
              method, so a posterior with more than one mode does not trap
              it in the nearest one
``none``      skip the search and start from ``x0``
============  ==========================================================

Everything here **minimises**, matching ``scipy.optimize`` and
``estimate_dsge``'s negative log posterior.

On ``csminwel``: this is Sims's algorithm — BFGS on a numerical gradient, with
a bespoke expanding/contracting line search and, the part that distinguishes it
from plain BFGS, a retry from a perturbed inverse Hessian when a step fails to
improve. It is **not** a line-by-line port of ``csminwel.m``; the line-search
constants and the termination test are this implementation's own, and the
number of Hessian resets is reported on the result so a run that leaned on them
can be recognised.

Bounds are honoured natively by ``lbfgs`` and ``simplex``. ``csminwel`` and
``cmaes`` clip candidates into the box, which is the right behaviour for a
prior truncation — the density is ``-inf`` outside it anyway — but means a mode
*on* a bound is reported at the bound rather than diagnosed.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as _scipy_minimize

__all__ = ["find_mode", "csminwel", "cmaes", "mode_check", "METHODS"]

METHODS = ("lbfgs", "simplex", "csminwel", "cmaes", "none")


def _clip(x: np.ndarray, bounds) -> np.ndarray:
    if bounds is None:
        return x
    lo = np.array([-np.inf if b[0] is None else b[0] for b in bounds], dtype=float)
    hi = np.array([np.inf if b[1] is None else b[1] for b in bounds], dtype=float)
    return np.clip(x, lo, hi)


def _numerical_gradient(f: Callable, x: np.ndarray, fx: float, h: float = 1e-5) -> np.ndarray:
    """Central differences, falling back to a one-sided step at a non-finite
    neighbour so a bound or an infeasible region does not poison the gradient."""
    g = np.zeros_like(x)
    for i in range(len(x)):
        step = h * max(abs(x[i]), 1.0)
        xp, xm = x.copy(), x.copy()
        xp[i] += step
        xm[i] -= step
        fp, fm = f(xp), f(xm)
        if np.isfinite(fp) and np.isfinite(fm):
            g[i] = (fp - fm) / (2.0 * step)
        elif np.isfinite(fp):
            g[i] = (fp - fx) / step
        elif np.isfinite(fm):
            g[i] = (fx - fm) / step
        else:
            g[i] = 0.0
    return g


def csminwel(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    bounds: Sequence | None = None,
    max_iter: int = 500,
    tol: float = 1e-8,
    grad_tol: float = 1e-6,
    max_resets: int = 3,
    **_ignored,
) -> OptimizeResult:
    """Sims-style quasi-Newton with the perturbed-Hessian retry.

    See the module docstring for what this is and is not. Returns an
    ``OptimizeResult`` carrying an extra ``n_hessian_resets`` field.
    """
    x = _clip(np.asarray(x0, dtype=float).copy(), bounds)
    n = len(x)
    fx = float(f(x))
    if not np.isfinite(fx):
        return OptimizeResult(x=x, fun=fx, success=False, nit=0, njev=0,
                              n_hessian_resets=0,
                              message="objective is not finite at x0")

    H = np.eye(n)          # inverse-Hessian estimate
    resets = 0
    it = 0
    for it in range(1, max_iter + 1):
        g = _numerical_gradient(f, x, fx)
        gnorm = float(np.max(np.abs(g)))
        if gnorm < grad_tol:
            return OptimizeResult(x=x, fun=fx, success=True, nit=it - 1, njev=it,
                                  n_hessian_resets=resets,
                                  message="gradient below tolerance")

        improved = False
        for attempt in range(max_resets + 1):
            direction = -(H @ g)
            if not np.all(np.isfinite(direction)) or float(direction @ g) >= 0.0:
                direction = -g          # not a descent direction; fall back
            # Expanding / contracting line search.
            best_lam, best_f, best_x = 0.0, fx, x
            lam = 1.0
            for _ in range(40):
                trial = _clip(x + lam * direction, bounds)
                ft = float(f(trial))
                if np.isfinite(ft) and ft < best_f:
                    best_lam, best_f, best_x = lam, ft, trial
                    lam *= 2.0          # keep expanding while it pays
                    continue
                if best_lam > 0.0:
                    break
                lam *= 0.5              # nothing yet: contract
                if lam < 1e-14:
                    break
            if best_lam > 0.0:
                s = best_x - x
                g_new = _numerical_gradient(f, best_x, best_f)
                y = g_new - g
                sy = float(s @ y)
                if sy > 1e-12:          # BFGS update, skipped if not curvature-positive
                    rho = 1.0 / sy
                    I = np.eye(n)
                    H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) \
                        + rho * np.outer(s, s)
                delta, x, fx = fx - best_f, best_x, best_f
                improved = True
                break
            # The step failed. This is where csminwel differs from BFGS:
            # perturb the inverse Hessian and try the same gradient again.
            resets += 1
            H = np.eye(n) * (0.5 ** (attempt + 1))
        if not improved:
            return OptimizeResult(x=x, fun=fx, success=False, nit=it - 1, njev=it,
                                  n_hessian_resets=resets,
                                  message="line search failed after Hessian resets")
        if delta < tol * max(1.0, abs(fx)):
            return OptimizeResult(x=x, fun=fx, success=True, nit=it, njev=it,
                                  n_hessian_resets=resets,
                                  message="function improvement below tolerance")

    return OptimizeResult(x=x, fun=fx, success=False, nit=it, njev=it,
                          n_hessian_resets=resets,
                          message=f"maximum iterations ({max_iter}) reached")


def cmaes(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    sigma0: float | None = None,
    bounds: Sequence | None = None,
    popsize: int | None = None,
    max_iter: int = 2000,
    tol: float = 1e-11,
    seed: int = 0,
    **_ignored,
) -> OptimizeResult:
    """(mu/mu_w, lambda)-CMA-ES (Hansen & Ostermeier), pure numpy.

    A population method: unlike the gradient routes it is not obliged to stop
    at the mode nearest ``x0``. ``sigma0`` is the initial step size and matters
    — it should be roughly the distance over which the objective changes.
    """
    rng = np.random.default_rng(seed)
    xmean = _clip(np.asarray(x0, dtype=float).copy(), bounds)
    n = len(xmean)
    sigma = float(0.3 * max(np.max(np.abs(xmean)), 1.0)) if sigma0 is None else float(sigma0)

    lam = int(4 + math.floor(3 * math.log(n))) if popsize is None else int(popsize)
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / np.sum(weights ** 2)

    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    pc = np.zeros(n)
    ps = np.zeros(n)
    B = np.eye(n)
    D = np.ones(n)
    C = np.eye(n)
    eigeneval = 0
    counteval = 0

    best_x, best_f = xmean.copy(), float(f(xmean))
    it = 0
    for it in range(1, max_iter + 1):
        z = rng.standard_normal((lam, n))
        arx = xmean + sigma * (z @ (B * D).T)
        arx = np.array([_clip(row, bounds) for row in arx])
        arf = np.array([float(f(row)) for row in arx])
        counteval += lam
        arf = np.where(np.isfinite(arf), arf, np.inf)
        order = np.argsort(arf)
        if arf[order[0]] < best_f:
            best_f, best_x = float(arf[order[0]]), arx[order[0]].copy()
        if not np.isfinite(arf[order[:mu]]).all():
            sigma *= 0.5
            continue

        xold = xmean
        xmean = weights @ arx[order[:mu]]
        y = (xmean - xold) / sigma
        Cinv_y = B @ (((B.T @ y) / D))
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * Cinv_y
        hsig = (np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * counteval / lam))
                / chiN) < (1.4 + 2 / (n + 1))
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y

        artmp = (arx[order[:mu]] - xold) / sigma
        C = ((1 - c1 - cmu) * C
             + c1 * (np.outer(pc, pc) + (not hsig) * cc * (2 - cc) * C)
             + cmu * artmp.T @ (weights[:, None] * artmp))
        sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        if counteval - eigeneval > lam / (c1 + cmu) / n / 10:
            eigeneval = counteval
            C = np.triu(C) + np.triu(C, 1).T
            vals, B = np.linalg.eigh(C)
            vals = np.maximum(vals, 1e-20)
            D = np.sqrt(vals)

        spread = float(arf[order[-1]] - arf[order[0]])
        if spread < tol * max(1.0, abs(best_f)) and sigma * float(np.max(D)) < 1e-11:
            return OptimizeResult(x=best_x, fun=best_f, success=True, nit=it,
                                  nfev=counteval, message="converged")

    return OptimizeResult(x=best_x, fun=best_f, success=False, nit=it,
                          nfev=counteval,
                          message=f"maximum iterations ({max_iter}) reached")


def _simplex(f, x0, *, bounds=None, restarts: int = 2, options=None, **_ignored):
    """Nelder-Mead, restarted from its own answer so it can escape a
    prematurely collapsed simplex."""
    x = np.asarray(x0, dtype=float)
    res = None
    for _ in range(max(1, restarts)):
        res = _scipy_minimize(f, x, method="Nelder-Mead", bounds=bounds,
                              options=options)
        if np.allclose(res.x, x, rtol=0, atol=1e-12):
            break
        x = np.asarray(res.x, dtype=float)
    return res


def find_mode(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    method: str = "csminwel",
    bounds: Sequence | None = None,
    options: dict | None = None,
    **kwargs: Any,
) -> OptimizeResult:
    """Minimise ``f`` from ``x0`` by the named method.

    Returns a ``scipy.optimize.OptimizeResult`` with an added ``method`` field.
    ``method="lbfgs"`` forwards to ``scipy.optimize.minimize(...,
    method="L-BFGS-B")`` with the given ``bounds`` and ``options`` and nothing
    else, so that route is bit-identical to calling scipy directly — which is
    what keeps the frozen SW07 parity fixture valid.
    """
    if method not in METHODS:
        raise ValueError(
            f"unknown mode_compute {method!r}; expected one of {list(METHODS)}"
        )
    x0 = np.asarray(x0, dtype=float)

    if method == "none":
        res = OptimizeResult(x=x0.copy(), fun=float(f(x0)), success=True, nit=0,
                             message="mode search skipped (method='none')")
    elif method == "lbfgs":
        res = _scipy_minimize(f, x0, method="L-BFGS-B", bounds=bounds,
                              options=options)
    elif method == "simplex":
        res = _simplex(f, x0, bounds=bounds, options=options, **kwargs)
    elif method == "csminwel":
        res = csminwel(f, x0, bounds=bounds, **kwargs)
    else:
        res = cmaes(f, x0, bounds=bounds, **kwargs)

    res.method = method
    return res


def mode_check(
    f: Callable[[np.ndarray], float],
    mode: np.ndarray,
    names: Sequence[str],
    *,
    n_points: int = 20,
    width: float = 2.0,
    cov: np.ndarray | None = None,
):
    """One-parameter slices of the objective through ``mode``.

    Dynare's ``mode_check`` plots. For each parameter the others are held at
    ``mode`` and the objective is traced over ``mode_i +/- width * sd_i``, with
    ``sd_i`` from ``cov`` when given and ``0.1 * max(|mode_i|, 1)`` otherwise.

    A slice whose minimum is not at the centre is the single most common sign
    that the reported "mode" is not one — :attr:`peaks_at_mode` records that
    per parameter, and :meth:`summary` names the offenders.
    """
    from puremacro.dsge._results import ModeCheckResult

    mode = np.asarray(mode, dtype=float)
    names = list(names)
    if len(names) != len(mode):
        raise ValueError(
            f"mode_check: {len(names)} names for {len(mode)} parameters"
        )
    n_points = int(n_points)
    if n_points < 3:
        raise ValueError(f"mode_check: n_points must be >= 3, got {n_points}")
    if n_points % 2 == 0:
        n_points += 1        # keep the mode itself on the grid

    if cov is not None:
        sd = np.sqrt(np.maximum(np.diag(np.asarray(cov, dtype=float)), 0.0))
    else:
        sd = np.zeros(len(mode))
    sd = np.where(sd > 0.0, sd, 0.1 * np.maximum(np.abs(mode), 1.0))

    import pandas as pd

    f_mode = float(f(mode))
    slices: dict = {}
    peaks: dict = {}
    for i, name in enumerate(names):
        grid = np.linspace(mode[i] - width * sd[i], mode[i] + width * sd[i], n_points)
        grid[n_points // 2] = mode[i]
        vals = np.empty(n_points)
        for j, g in enumerate(grid):
            trial = mode.copy()
            trial[i] = g
            vals[j] = float(f(trial))
        slices[name] = pd.DataFrame({"value": grid, "objective": vals})
        finite = np.isfinite(vals)
        peaks[name] = bool(finite[n_points // 2] and
                           np.nanmin(vals[finite]) >= vals[n_points // 2] - 1e-10)

    return ModeCheckResult(
        slices=slices, peaks_at_mode=peaks, mode=dict(zip(names, mode)),
        objective_at_mode=f_mode,
    )
