"""Generic Bayesian DSGE estimator via Random-Walk Metropolis-Hastings.

Model-agnostic. Mode refinement (scipy.optimize.minimize, L-BFGS-B) →
numerical Hessian → proposal cov c²·H⁻¹ (fallback diag(prior_stds²) if
H not PD) → multi-chain RW-MH via puremacro.mcmc.random_walk_metropolis.

This module hosts the helpers that were previously inlined in
``puremacro.dsge.sw07_estimate`` but contained nothing SW07-specific:
``_vec_to_dict``, ``_make_neg_log_posterior``, ``_initial_vec_from_dict``,
``_nearest_pd``, ``_find_finite_start``.
"""
from __future__ import annotations

import warnings
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize as _scipy_minimize

from puremacro.dsge._results import DSGEPosteriorResult
from puremacro.dsge.priors import (
    log_prior, prior_stds, param_bounds, param_names,
)
from puremacro.mcmc import random_walk_metropolis
from puremacro.numerics import numerical_hessian
from puremacro.state_space import StateSpaceModel, kalman_filter


def _vec_to_dict(
    vec: np.ndarray,
    names: Sequence[str],
    fixed_params: dict | None = None,
) -> dict:
    """Combine an estimated-param vector with fixed params into a single dict."""
    if len(vec) != len(names):
        raise ValueError(f"vec length {len(vec)} doesn't match {len(names)} names")
    out = dict(fixed_params or {})
    for nm, val in zip(names, vec):
        out[nm] = float(val)
    return out


def _make_neg_log_posterior(
    y: np.ndarray,
    observation_eq: Callable[[dict], StateSpaceModel],
    priors: dict,
    names: Sequence[str],
    fixed_params: dict | None,
):
    """Closure returning -log_posterior(vec). Returns +inf on numerical failure."""
    def neg_log_post(vec: np.ndarray) -> float:
        try:
            params = _vec_to_dict(vec, names, fixed_params)
        except ValueError:
            return np.inf
        lp = log_prior(params, priors)
        if not np.isfinite(lp):
            return np.inf
        try:
            ssm = observation_eq(params)
            out = kalman_filter(y, ssm)
            ll = out["loglik"]
        except (np.linalg.LinAlgError, ValueError, RuntimeError,
                ZeroDivisionError, FloatingPointError):
            return np.inf
        if not np.isfinite(ll):
            return np.inf
        return -(ll + lp)
    return neg_log_post


def _initial_vec_from_dict(
    initial_params: dict,
    priors: dict,
) -> np.ndarray:
    """Build the initial parameter vector from a dict, snapping out-of-bound
    values to lb + 1e-3 (well inside the support).
    """
    vec = []
    for name, spec in priors.items():
        val = initial_params.get(name)
        if val is None or not (spec["lb"] <= val <= spec["ub"]):
            val = spec["lb"] + 1e-3
        vec.append(val)
    return np.array(vec)


def _nearest_pd(A: np.ndarray) -> np.ndarray:
    """Nearest symmetric positive-definite matrix (Higham 2002, abridged)."""
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = V.T @ np.diag(s) @ V
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    eps = np.finfo(float).eps
    I = np.eye(A.shape[0])
    k = 1
    while True:
        try:
            np.linalg.cholesky(A3)
            return A3
        except np.linalg.LinAlgError:
            mineig = np.min(np.linalg.eigvalsh(A3))
            A3 = A3 + I * (-mineig * k ** 2 + eps)
            k += 1
            if k > 100:
                raise RuntimeError("_nearest_pd did not converge")


def _find_finite_start(
    neg_log_post,
    rng: np.random.Generator,
    priors: dict,
    max_tries: int = 200,
) -> np.ndarray:
    """Random points inside the prior box until neg_log_post is finite."""
    bounds = param_bounds(priors)
    for _ in range(max_tries):
        vec = np.array([rng.uniform(lb, ub) for (lb, ub) in bounds])
        if np.isfinite(neg_log_post(vec)):
            return vec
    # Hard fallback: prior means clipped to interior.
    vec = np.array([
        np.clip(spec["mean"], spec["lb"] + 1e-6, spec["ub"] - 1e-6)
        for spec in priors.values()
    ])
    return vec


def estimate_dsge(
    data: pd.DataFrame,
    *,
    observation_eq: Callable[[dict], StateSpaceModel],
    priors: dict,
    observed_vars: Sequence[str],
    initial_params: dict,
    fixed_params: dict | None = None,
    model_name: str = "unknown",
    n_draws: int = 10_000,
    n_chains: int = 2,
    burn_in: int = 2_000,
    seed: int = 0,
) -> DSGEPosteriorResult:
    """Bayesian DSGE estimation via Random-Walk Metropolis-Hastings."""
    # 1. Validate data.
    missing = set(observed_vars) - set(data.columns)
    if missing:
        raise ValueError(f"data missing columns: {sorted(missing)}")
    if len(data) < 10:
        raise ValueError(f"data has only {len(data)} obs; need >= 10")
    if data[list(observed_vars)].isna().any().any():
        raise ValueError("data contains NaN in observed_vars")
    y = data[list(observed_vars)].to_numpy()

    # 2. Build neg_log_post + initial vec.
    names = param_names(priors)
    fixed = dict(fixed_params or {})
    neg_log_post = _make_neg_log_posterior(
        y, observation_eq, priors, names, fixed,
    )
    init_vec = _initial_vec_from_dict(initial_params, priors)

    # 3. Mode refinement.
    mode_vec = init_vec.copy()
    converged_mle = False
    try:
        opt = _scipy_minimize(
            neg_log_post, init_vec,
            method="L-BFGS-B",
            bounds=param_bounds(priors),
            options={"maxiter": 100, "maxfun": 500 * len(init_vec)},
        )
        if opt.success and np.isfinite(opt.fun):
            mode_vec = np.asarray(opt.x, dtype=float)
            converged_mle = True
        else:
            warnings.warn(
                "estimate_dsge: mode optimisation did not converge; using "
                "the snap-corrected initial_params as the mode.",
                UserWarning,
            )
    except Exception as e:
        warnings.warn(
            f"estimate_dsge: mode optimisation raised {type(e).__name__}; "
            f"using the snap-corrected initial_params as the mode.",
            UserWarning,
        )

    # 4. Hessian-based proposal cov (only when mode converged).
    use_hessian = False
    inv_H: np.ndarray
    if converged_mle:
        H = numerical_hessian(neg_log_post, mode_vec, h=1e-4)
        try:
            H_pd = _nearest_pd(H)
            inv_H = np.linalg.inv(H_pd)
            np.linalg.cholesky(inv_H)
            use_hessian = True
        except (np.linalg.LinAlgError, RuntimeError):
            warnings.warn(
                "estimate_dsge: Hessian non-PD even after _nearest_pd; "
                "falling back to diag(prior_stds**2).",
                UserWarning,
            )
    if not use_hessian:
        stds = np.array([prior_stds(priors)[n] for n in names])
        inv_H = np.diag(stds ** 2)

    # 5. Proposal scaling.
    n_params = len(names)
    c0 = 2.38 / np.sqrt(n_params) if use_hessian else 0.01
    proposal_cov = c0 ** 2 * inv_H

    # 6. Run chains.
    chains_arr = np.empty((n_chains, n_draws, n_params))
    log_post_arr = np.empty((n_chains, n_draws))
    accept_rates = []

    def log_post_fn(vec):
        return -neg_log_post(vec)

    for chain_idx in range(n_chains):
        rng = np.random.default_rng(seed + chain_idx)
        try:
            perturb = rng.multivariate_normal(np.zeros(n_params), 0.0025 * inv_H)
        except np.linalg.LinAlgError:
            perturb = 0.05 * rng.standard_normal(n_params)
        start = mode_vec + perturb
        if not np.isfinite(neg_log_post(start)):
            start = mode_vec.copy()
        if not np.isfinite(neg_log_post(start)):
            start = _find_finite_start(neg_log_post, rng, priors)

        out = random_walk_metropolis(
            log_post_fn, start, proposal_cov, n_draws=n_draws,
            seed=seed + chain_idx, accept_target=0.25, adapt_burnin=burn_in,
        )
        chains_arr[chain_idx] = out["chain"]
        log_post_arr[chain_idx] = out["log_post"]
        accept_rates.append(out["accept_rate"])

        if not (0.10 <= out["accept_rate"] <= 0.50):
            warnings.warn(
                f"estimate_dsge: chain {chain_idx} accept_rate="
                f"{out['accept_rate']:.3f} outside [0.10, 0.50]; "
                f"mixing may be poor.",
                UserWarning,
            )

    mode_dict = _vec_to_dict(mode_vec, names, fixed)

    return DSGEPosteriorResult(
        draws=chains_arr,
        param_names=names,
        log_posterior_trace=log_post_arr,
        accept_rates=tuple(accept_rates),
        mode=mode_dict,
        mode_hessian_inv=inv_H,
        n_burn_in=burn_in,
        data_n_obs=len(data),
        seed=seed,
        model_name=model_name,
    )


__all__ = ["estimate_dsge"]
