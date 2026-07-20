"""Foundational pure-array kernels for the nested heterogeneous-firm DMP model.

Compute kernels are written once against a numpy-like array namespace
``xp`` so the same source runs under NumPy (oracle) and MLX (Apple GPU).
Kernels that use only arithmetic operators take no ``xp`` (operators are
overloaded identically by numpy and mlx); kernels that call array
functions (exp, maximum, reductions) accept ``xp`` (default numpy).

``rouwenhorst`` is numpy-only by design: it is a one-time setup routine
that assembles a transition matrix by in-place slicing, it is not on any
hot path, and it never runs on the GPU.
"""
from __future__ import annotations

import numpy as np


def matching_rate(u, v, *, mu: float, alpha: float):
    """Cobb-Douglas matches m = mu * u^alpha * v^(1-alpha)."""
    return mu * u ** alpha * v ** (1.0 - alpha)


def q_theta(theta, *, mu: float, alpha: float):
    """Job-filling rate q(theta) = mu * theta^(-alpha)."""
    return mu * theta ** (-alpha)


def f_theta(theta, *, mu: float, alpha: float):
    """Job-finding rate f(theta) = mu * theta^(1-alpha) = theta * q(theta)."""
    return mu * theta ** (1.0 - alpha)


def belief_update(
    prior_pi,
    *,
    r_obs,
    sigma,
    r_star: float,
    phi_D: float,
    phi_H: float,
    signal_sd: float,
    xp=np,
):
    """Posterior Pr(Fed = dove) after observing the policy rate.

    Rate signal r = r_star + phi_tau * sigma + N(0, signal_sd^2) for type
    tau in {D, H}. Bayes update of the dove prior; vectorises over array
    inputs. Works under numpy or mlx (uses only xp.exp + arithmetic).
    """
    mu_D = r_star + phi_D * sigma
    mu_H = r_star + phi_H * sigma
    lik_D = xp.exp(-0.5 * ((r_obs - mu_D) / signal_sd) ** 2)
    lik_H = xp.exp(-0.5 * ((r_obs - mu_H) / signal_sd) ** 2)
    num = prior_pi * lik_D
    return num / (num + (1.0 - prior_pi) * lik_H)


def expected_discount(pi, sigma, *, r_star: float, phi_D: float, phi_H: float,
                      phi_sigma: float = 0.0):
    """Expected continuation discount factor under the two-type Fed prior.

    E[beta | pi, sigma] = pi/(1+r_D) + (1-pi)/(1+r_H), with the perceived
    type-rate r_tau = r_star + (phi_tau - phi_sigma)*sigma, the gross-rate
    inverse beta(r)=1/(1+r) averaged over the posterior pi that the Fed is a
    dove. phi_D, phi_H, phi_sigma must be in DECIMAL rate units (the model layer
    converts from the percentage points stored on the parameter object).
    With phi_D<0 (dove cuts) the dove branch rises in sigma and with phi_H>0
    (hawk hikes) the hawk branch falls; the net slope d E[beta]/d sigma is
    positive iff pi > pi* = phi_H/(phi_H - phi_D), the sign-flip threshold.
    phi_sigma>0 (the systematic 'lean against uncertainty' policy lever) cuts
    BOTH type-rates by phi_sigma*sigma, raising E[beta] uniformly while leaving
    the type gap r_H-r_D=(phi_H-phi_D)*sigma -- and hence pi* -- unchanged.
    Vectorises over array inputs (pure arithmetic; numpy or mlx).
    """
    r_D = r_star + (phi_D - phi_sigma) * sigma
    r_H = r_star + (phi_H - phi_sigma) * sigma
    return pi / (1.0 + r_D) + (1.0 - pi) / (1.0 + r_H)


def bellman_iterate(flow, P, beta: float, *, tol: float = 1e-10,
                    max_iter: int = 10_000, xp=np):
    """Value-function iteration for the match value with a separation option.

    Solves V = max(flow + beta * (P @ V), 0), where the outside option
    (separate) is normalised to 0. ``flow`` is the per-state flow payoff
    of continuing; ``P`` is a row-stochastic transition over the
    productivity grid. Assignment-free, so it runs unchanged under numpy
    or mlx. Returns the converged value array.
    """
    V = flow * 0.0
    for _ in range(max_iter):
        cont = flow + beta * (P @ V)
        V_new = xp.maximum(cont, 0.0)
        if float(xp.max(xp.abs(V_new - V))) < tol:
            return V_new
        V = V_new
    raise RuntimeError(
        f"bellman_iterate did not converge in {max_iter} iterations "
        f"(final sup-norm change still exceeded tol={tol})."
    )


def rouwenhorst(n: int, rho: float, sigma_eps: float):
    """Rouwenhorst (1995) discretization of x' = rho*x + eps, eps~N(0,sigma_eps^2).

    Returns ``(grid, P)`` with ``grid`` shape (n,) symmetric about 0 and
    ``P`` the (n,n) row-stochastic transition matrix. numpy-only (in-place
    matrix assembly; one-time setup, not a GPU/hot path).
    """
    if n < 2:
        raise ValueError(f"n must be >= 2; got {n}")
    p = (1.0 + rho) / 2.0
    P = np.array([[p, 1.0 - p], [1.0 - p, p]])
    for k in range(3, n + 1):
        Pk = np.zeros((k, k))
        Pk[:-1, :-1] += p * P
        Pk[:-1, 1:] += (1.0 - p) * P
        Pk[1:, :-1] += (1.0 - p) * P
        Pk[1:, 1:] += p * P
        Pk[1:-1, :] /= 2.0  # interior rows double-counted; renormalise
        P = Pk
    sigma_uncond = sigma_eps / np.sqrt(1.0 - rho ** 2)
    psi = np.sqrt(n - 1) * sigma_uncond
    grid = np.linspace(-psi, psi, n)
    return grid, P


def tauchen_transition(grid, rho: float, sigma_eps: float):
    """AR(1) transition matrix on a GIVEN (fixed) grid (Tauchen 1986).

    P[i,j] = Pr(x' in cell_j | x = grid[i]), x' = rho*grid[i] + eps,
    eps ~ N(0, sigma_eps^2). Cell edges are midpoints between consecutive grid
    points; the first/last cells are half-open. Unlike ``rouwenhorst`` (which
    builds its own grid), this discretizes onto a FIXED grid -- needed when the
    innovation sd varies over a transition while the grid stays put. numpy-only
    (one-time per-period setup, not a GPU/hot path).
    """
    from scipy.stats import norm

    g = np.asarray(grid, dtype=float)
    n = g.size
    edges = 0.5 * (g[:-1] + g[1:])           # n-1 interior cell edges
    P = np.empty((n, n))
    for i in range(n):
        mean = rho * g[i]
        z = (edges - mean) / sigma_eps
        cdf = norm.cdf(z)                     # length n-1
        P[i, 0] = cdf[0]
        P[i, 1:-1] = cdf[1:] - cdf[:-1]
        P[i, -1] = 1.0 - cdf[-1]
    return P


__all__ = [
    "matching_rate", "q_theta", "f_theta",
    "belief_update", "expected_discount", "bellman_iterate", "rouwenhorst",
    "tauchen_transition",
]
