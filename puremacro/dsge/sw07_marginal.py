"""Laplace marginal data density of the Smets-Wouters (2007) model, Table 2 style.

Smets and Wouters report a marginal likelihood of -905.8 for the DSGE model
(ECB WP 722, Table 2; AER Table 2), estimated over 1966Q1-2004Q4 "using the
period 1956:1-1965:4 as a training sample (Sims, 2002)" so that it is
comparable with the VAR and BVAR marginal likelihoods in the same table, and
they compute it with the Laplace approximation around the posterior mode
(WP 722, p. 21). In Dynare's terms the DSGE side of that is a ``presample``:
the Kalman filter runs through the training quarters from its initialisation
and the likelihood sums the remaining quarters. :func:`sw07_laplace_mdd`
reproduces that computation on the authors' own series
(:func:`puremacro.dsge.sw07_data.load_sw07_data`), with the sample start,
presample length and filter initialisation as explicit arguments so the
alternatives (no training sample; the 1965 presample of the authors' Dynare
file, ``first_obs=71, presample=4, lik_init=2``) can be computed the same way.

The mode search is L-BFGS-B from the ``.mod`` starting values followed by a
damped Newton polish on central-difference derivatives, the procedure
``tools/build_sw07_data.py`` uses for the replication fixture. New in 4.6.0.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

__all__ = ["SW07MarginalLikelihoodResult", "sw07_laplace_mdd", "sw07_log_posterior"]


@dataclass(frozen=True)
class SW07MarginalLikelihoodResult:
    """Posterior mode and Laplace marginal data density of the SW07 model."""

    mode: dict[str, float]
    log_post_mode: float
    log_likelihood_mode: float
    log_prior_mode: float
    laplace_mdd: float
    mode_hessian_inv: np.ndarray = field(repr=False)
    n_obs_likelihood: int
    n_obs_presample: int
    first_obs: str
    last_obs: str
    dataset: str
    lik_init: str
    diffuse_scale: float
    converged: bool
    n_iterations: int
    message: str
    seconds: float

    def summary(self) -> pd.DataFrame:
        """One row per parameter: mode and Laplace standard error."""
        se = np.sqrt(np.clip(np.diag(self.mode_hessian_inv), 0.0, None))
        return pd.DataFrame({"mode": list(self.mode.values()), "laplace_se": se},
                            index=list(self.mode.keys()))


def _objective(y: np.ndarray, *, presample: int, lik_init: str, diffuse_scale: float):
    from puremacro.dsge.estimate import _OPT_PENALTY, _make_neg_log_posterior
    from puremacro.dsge.sw07_estimate import _FIXED_PARAMS
    from puremacro.dsge.sw07_observation import make_state_space
    from puremacro.dsge.sw07_priors import PRIORS, param_names

    names = param_names()
    kw = dict(presample=presample, lik_init=lik_init, diffuse_scale=diffuse_scale,
              caller="sw07_laplace_mdd")
    nlp = _make_neg_log_posterior(y, make_state_space, PRIORS, names, _FIXED_PARAMS, **kw)
    nlp_opt = _make_neg_log_posterior(y, make_state_space, PRIORS, names, _FIXED_PARAMS,
                                      penalty=_OPT_PENALTY, **kw)
    lb = np.array([PRIORS[n]["lb"] for n in names], dtype=float)
    ub = np.array([PRIORS[n]["ub"] for n in names], dtype=float)
    return names, nlp, nlp_opt, lb, ub


def _central_gradient(f, x, lb, ub, h=1e-5):
    g = np.empty_like(x)
    for i in range(len(x)):
        hi = h * max(1.0, abs(x[i]))
        up, dn = x.copy(), x.copy()
        up[i] = min(x[i] + hi, ub[i])
        dn[i] = max(x[i] - hi, lb[i])
        g[i] = (f(up) - f(dn)) / (up[i] - dn[i])
    return g


def _polish(f, x, fx, lb, ub, *, max_steps: int = 8):
    """Damped Newton steps on central differences until the gain is below 1e-9."""
    from puremacro.numerics import numerical_hessian

    steps = 0
    for _ in range(max_steps):
        g = _central_gradient(f, x, lb, ub)
        H = numerical_hessian(f, x, h=1e-4)
        H = 0.5 * (H + H.T)
        w, V = np.linalg.eigh(H)
        step = -V @ ((V.T @ g) / np.maximum(w, 1e-6 * max(float(w.max()), 1.0)))
        t, improved = 1.0, False
        while t > 1e-4:
            x_new = np.clip(x + t * step, lb + 1e-8, ub - 1e-8)
            f_new = float(f(x_new))
            if f_new < fx:
                improved = True
                break
            t *= 0.5
        if not improved:
            break
        steps += 1
        gain = fx - f_new
        x, fx = x_new, f_new
        if gain < 1e-9:
            break
    return x, fx, steps


def _start_vector(start, names) -> np.ndarray:
    from puremacro.dsge.estimate import _initial_vec_from_dict
    from puremacro.dsge.sw07_priors import PRIORS, SW07_MOD_INITIAL_VALUES

    if isinstance(start, Mapping):
        values = dict(start)
    elif start == "mod":
        values = dict(SW07_MOD_INITIAL_VALUES)
    elif start == "table1":
        from puremacro.dsge.sw07_estimate import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
        values = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    else:
        raise ValueError(f"start must be 'mod', 'table1' or a mapping, got {start!r}")
    return _initial_vec_from_dict(values, PRIORS, caller="sw07_laplace_mdd")


def sw07_log_posterior(
    params: Mapping[str, float],
    *,
    data: pd.DataFrame | None = None,
    dataset: str = "authors",
    first_obs: str = "1956Q1",
    last_obs: str = "2004Q4",
    presample: int = 40,
    lik_init: str = "diffuse",
    diffuse_scale: float = 10.0,
) -> float:
    """Log posterior (log likelihood + log prior) of the SW07 model at ``params``."""
    from puremacro.dsge.sw07_data import load_sw07_data
    from puremacro.dsge.sw07_observation import OBSERVED_VARS

    df = load_sw07_data(dataset, first_obs=first_obs, last_obs=last_obs) if data is None else data
    y = df[list(OBSERVED_VARS)].to_numpy(dtype=float)
    names, nlp, _, _, _ = _objective(y, presample=presample, lik_init=lik_init, diffuse_scale=diffuse_scale)
    vec = np.array([float(params[n]) for n in names])
    return -float(nlp(vec))


def sw07_laplace_mdd(
    data: pd.DataFrame | None = None,
    *,
    dataset: str = "authors",
    first_obs: str = "1956Q1",
    last_obs: str = "2004Q4",
    presample: int = 40,
    lik_init: str = "diffuse",
    diffuse_scale: float = 10.0,
    start="mod",
    maxiter: int = 2000,
    polish: bool = True,
    verbose: bool = False,
) -> SW07MarginalLikelihoodResult:
    """Posterior mode and Laplace log marginal data density of the SW07 model.

    The defaults reproduce the Table 2 computation of Smets and Wouters (2007):
    the authors' data from 1956Q1, the 40 quarters 1956Q1-1965Q4 as a
    presample (training sample) and a diffuse initialisation, so that the
    likelihood is that of 1966Q1-2004Q4 given the training quarters.

    Parameters
    ----------
    data : DataFrame, optional
        Observables with the columns of ``OBSERVED_VARS``; when given,
        ``dataset``, ``first_obs`` and ``last_obs`` are ignored except as labels.
    dataset, first_obs, last_obs
        Which bundled series and which quarters (inclusive); see
        :func:`puremacro.dsge.sw07_data.load_sw07_data`.
    presample : int
        Quarters at the start of the sample that initialise the filter and
        do not enter the likelihood (Dynare ``presample``).
    lik_init : {"diffuse", "stationary"}
        Filter initialisation (Dynare ``lik_init=2`` with ``diffuse_scale``,
        or the model's unconditional distribution).
    start : {"mod", "table1"} or mapping
        Starting values of the mode search.
    maxiter : int
        L-BFGS-B iteration budget before the Newton polish.
    polish : bool
        Run the damped Newton polish after L-BFGS-B.

    Returns
    -------
    SW07MarginalLikelihoodResult
    """
    from puremacro.dsge.marginal import laplace_mdd
    from puremacro.dsge.mode import find_mode
    from puremacro.dsge.priors import log_prior
    from puremacro.dsge.sw07_data import load_sw07_data
    from puremacro.dsge.sw07_estimate import _FIXED_PARAMS
    from puremacro.dsge.sw07_observation import OBSERVED_VARS
    from puremacro.dsge.sw07_priors import PRIORS
    from puremacro.numerics import numerical_hessian

    t0 = time.perf_counter()
    if data is None:
        df = load_sw07_data(dataset, first_obs=first_obs, last_obs=last_obs)
    else:
        df = data
        first_obs, last_obs = str(df.index[0]), str(df.index[-1])
    y = df[list(OBSERVED_VARS)].to_numpy(dtype=float)
    names, nlp, nlp_opt, lb, ub = _objective(y, presample=presample, lik_init=lik_init,
                                             diffuse_scale=diffuse_scale)
    x0 = _start_vector(start, names)
    x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)
    opt = find_mode(nlp_opt, x0, method="lbfgs", bounds=list(zip(lb, ub)),
                    options={"maxiter": int(maxiter), "maxfun": 200_000, "ftol": 1e-12, "gtol": 1e-6})
    x, fx = np.asarray(opt.x, dtype=float), float(opt.fun)
    n_iter = int(getattr(opt, "nit", 0))
    message = str(getattr(opt, "message", ""))
    if verbose:
        print(f"L-BFGS-B: {n_iter} iterations, log posterior {-fx:.6f} ({message})", flush=True)
    if polish:
        x, fx, steps = _polish(nlp_opt, x, fx, lb, ub)
        n_iter += steps
        if verbose:
            print(f"Newton polish: {steps} steps, log posterior {-fx:.9f}", flush=True)
    lp_mode = -float(nlp(x))
    params = dict(zip(names, map(float, x)))
    log_prior_mode = float(log_prior({**params, **_FIXED_PARAMS}, PRIORS))
    H = numerical_hessian(nlp, x, h=1e-4)
    H = 0.5 * (H + H.T)
    eig = np.linalg.eigvalsh(H)
    converged = bool(np.isfinite(lp_mode) and eig.min() > 0)
    if eig.min() <= 0:
        message += f" | Hessian at the mode not positive definite (min eig {eig.min():.3e})"
        inv_H = np.full_like(H, np.nan)
        mdd = float("nan")
    else:
        inv_H = np.linalg.inv(H)
        inv_H = 0.5 * (inv_H + inv_H.T)
        mdd = float(laplace_mdd(lp_mode, inv_H))
    return SW07MarginalLikelihoodResult(
        mode=params, log_post_mode=lp_mode, log_likelihood_mode=lp_mode - log_prior_mode,
        log_prior_mode=log_prior_mode, laplace_mdd=mdd, mode_hessian_inv=inv_H,
        n_obs_likelihood=int(len(y) - presample), n_obs_presample=int(presample),
        first_obs=str(first_obs), last_obs=str(last_obs), dataset=str(dataset) if data is None else "user",
        lik_init=lik_init, diffuse_scale=float(diffuse_scale), converged=converged,
        n_iterations=n_iter, message=message, seconds=time.perf_counter() - t0,
    )
