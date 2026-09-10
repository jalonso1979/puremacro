"""Pure-Python Hamiltonian Monte Carlo & No-U-Turn Sampler (NUTS).

Provides high-efficiency gradient-based MCMC sampling for DSGE posterior distributions
and general continuous targets. Adheres strictly to the Pyodide four-package contract:
numpy, scipy, pandas, matplotlib ONLY.

Key Features
------------
1. Symplectic leapfrog integrator with kinetic energy K(p) = 0.5 * p' M^{-1} p.
2. Recursive binary tree expansion with Betancourt (2017) generalized U-turn stopping
   criterion and Hamiltonian divergence detection (Delta_max = 1000).
3. Nesterov / Hoffman-Gelman dual averaging step size adaptation targeting delta* = 0.80.
4. Welford online diagonal mass matrix adaptation M^{-1} ≈ Var(theta) with Stan-style
   staged warmup schedule (fast buffer, doubling slow windows, final fast buffer).
5. Comprehensive MCMC diagnostics: split-R_hat, bulk ESS, tail ESS, E-BFMI, and
   NUTSResult container with visualization methods.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence, Tuple

import numpy as np
import pandas as pd

from puremacro.dsge._results import NUTSResult
from puremacro.mcmc import autocorrelations, effective_sample_size


# ---------------------------------------------------------------------------
# Diagnostics: Split-R_hat, Bulk/Tail ESS, E-BFMI
# ---------------------------------------------------------------------------

def compute_split_rhat(chains: np.ndarray) -> float:
    r"""Compute split Gelman-Rubin \hat{R} for an (n_chains, n_draws) array.

    Splits each chain in half to detect both within-chain non-stationarity
    and between-chain lack of mixing (Gelman et al. BDA3).
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 1:
        chains = chains[None, :]
    n_chains, n_draws = chains.shape
    if n_draws < 4:
        return 1.0

    half = n_draws // 2
    split_chains = np.empty((2 * n_chains, half), dtype=float)
    for c in range(n_chains):
        split_chains[2 * c] = chains[c, :half]
        split_chains[2 * c + 1] = chains[c, half:2 * half]

    M, N = split_chains.shape
    chain_means = split_chains.mean(axis=1)
    grand_mean = float(chain_means.mean())

    B = N * float(np.sum((chain_means - grand_mean) ** 2)) / max(M - 1, 1)
    W = float(np.sum((split_chains - chain_means[:, None]) ** 2)) / (M * max(N - 1, 1))

    if W <= 0.0 or not np.isfinite(W):
        return 1.0

    var_plus = ((N - 1.0) / N) * W + (1.0 / N) * B
    rhat = float(np.sqrt(max(1.0, var_plus / W)))
    return rhat


def compute_bulk_ess(chains: np.ndarray) -> float:
    r"""Compute bulk Effective Sample Size across chains via Geyer's monotone sequence."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 1:
        chains = chains[None, :]
    n_chains = chains.shape[0]
    ess_list = [effective_sample_size(chains[c]) for c in range(n_chains)]
    return float(np.sum(ess_list))


def compute_tail_ess(chains: np.ndarray) -> float:
    r"""Compute tail Effective Sample Size based on 5% and 95% quantile indicators."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 1:
        chains = chains[None, :]
    flat = chains.ravel()
    q05 = float(np.quantile(flat, 0.05))
    q95 = float(np.quantile(flat, 0.95))
    ind05 = (chains <= q05).astype(float)
    ind95 = (chains <= q95).astype(float)
    ess05 = compute_bulk_ess(ind05)
    ess95 = compute_bulk_ess(ind95)
    return float(min(ess05, ess95))


def compute_ebfmi(energy_trace: np.ndarray) -> float:
    r"""Energy Bayesian Fraction of Missing Information (Betancourt 2016).

    E-BFMI = \frac{\sum_{i=1}^{N-1} (E_{i+1} - E_i)^2}{(N - 1) \operatorname{Var}(E)}.
    Values below 0.3 indicate inefficient momentum exploration.
    """
    E = np.asarray(energy_trace, dtype=float).ravel()
    N = len(E)
    if N < 2:
        return 1.0
    numerator = float(np.sum(np.diff(E) ** 2)) / (N - 1.0)
    denominator = float(np.var(E, ddof=1))
    if denominator <= 1e-12 or not np.isfinite(denominator):
        return 1.0
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# Leapfrog Integrator & Stopping Conditions
# ---------------------------------------------------------------------------

def leapfrog(
    theta: np.ndarray,
    p: np.ndarray,
    grad: np.ndarray,
    eps: float,
    log_prob_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    M_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    r"""Single symplectic leapfrog integration step of size eps.

    Parameters
    ----------
    theta : ndarray of shape (K,)
        Position vector.
    p : ndarray of shape (K,)
        Momentum vector.
    grad : ndarray of shape (K,)
        Gradient of target log-posterior at theta.
    eps : float
        Integration step size.
    log_prob_and_grad : Callable
        Evaluator returning (log_prob, grad).
    M_inv : ndarray of shape (K,)
        Diagonal inverse mass matrix elements.

    Returns
    -------
    theta_next, p_next, log_p_next, grad_next
    """
    p_half = p + 0.5 * eps * grad
    theta_next = theta + eps * (M_inv * p_half)
    log_p_next, grad_next = log_prob_and_grad(theta_next)
    p_next = p_half + 0.5 * eps * grad_next
    return theta_next, p_next, log_p_next, grad_next


def _check_u_turn(
    theta_minus: np.ndarray,
    theta_plus: np.ndarray,
    p_minus: np.ndarray,
    p_plus: np.ndarray,
    M_inv: np.ndarray,
) -> bool:
    r"""Check Betancourt generalized U-turn stopping criterion.

    Returns True if trajectory is turning back on itself:
    (theta^+ - theta^-)' M^{-1} p^+ < 0 or (theta^+ - theta^-)' M^{-1} p^- < 0.
    """
    d_theta = theta_plus - theta_minus
    return bool(
        float(np.dot(d_theta, M_inv * p_plus)) < 0.0
        or float(np.dot(d_theta, M_inv * p_minus)) < 0.0
    )


# ---------------------------------------------------------------------------
# Recursive Binary Tree Building (Betancourt 2017 Multinomial NUTS)
# ---------------------------------------------------------------------------

def _build_tree(
    theta: np.ndarray,
    p: np.ndarray,
    grad: np.ndarray,
    log_p: float,
    v: int,
    j: int,
    eps: float,
    H0: float,
    log_prob_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    M_inv: np.ndarray,
    delta_max: float,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, float, np.ndarray, float,
    float, bool, float, int, bool
]:
    r"""Recursively build a binary tree of leapfrog steps in direction v \in {-1, +1}."""
    if j == 0:
        th_new, p_new, lp_new, g_new = leapfrog(
            theta, p, grad, float(v) * eps, log_prob_and_grad, M_inv
        )
        if not math.isfinite(lp_new) or not np.all(np.isfinite(p_new)) or not np.all(np.isfinite(g_new)):
            return (
                th_new, p_new, g_new,
                th_new, p_new, g_new,
                theta, log_p, grad, H0,
                0.0, False, 0.0, 1, True
            )

        K_new = 0.5 * float(np.sum(M_inv * (p_new ** 2)))
        H_new = -lp_new + K_new
        energy_diff = -H_new + H0

        divergent = bool(math.isnan(H_new) or (H_new - H0 > delta_max))
        if divergent:
            return (
                th_new, p_new, g_new,
                th_new, p_new, g_new,
                th_new, lp_new, g_new, H_new,
                0.0, False, 0.0, 1, True
            )

        weight = float(np.exp(min(0.0, energy_diff)))
        alpha = float(min(1.0, np.exp(energy_diff)))
        s = True

        return (
            th_new, p_new, g_new,
            th_new, p_new, g_new,
            th_new, lp_new, g_new, H_new,
            weight, s, alpha, 1, False
        )

    (
        th_m, p_m, g_m,
        th_p, p_p, g_p,
        th_prime, lp_prime, g_prime, H_prime,
        w_prime, s_prime, a_prime, na_prime, div1
    ) = _build_tree(
        theta, p, grad, log_p, v, j - 1, eps, H0,
        log_prob_and_grad, M_inv, delta_max, rng
    )

    if not s_prime:
        return (
            th_m, p_m, g_m,
            th_p, p_p, g_p,
            th_prime, lp_prime, g_prime, H_prime,
            w_prime, False, a_prime, na_prime, div1
        )

    if v == -1:
        (
            th_mm, p_mm, g_mm,
            th_mp, p_mp, g_mp,
            th_2prime, lp_2prime, g_2prime, H_2prime,
            w_2prime, s_2prime, a_2prime, na_2prime, div2
        ) = _build_tree(
            th_m, p_m, g_m, lp_prime, v, j - 1, eps, H0,
            log_prob_and_grad, M_inv, delta_max, rng
        )
        theta_minus, p_minus, grad_minus = th_mm, p_mm, g_mm
        theta_plus, p_plus, grad_plus = th_p, p_p, g_p
    else:
        (
            th_pm, p_pm, g_pm,
            th_pp, p_pp, g_pp,
            th_2prime, lp_2prime, g_2prime, H_2prime,
            w_2prime, s_2prime, a_2prime, na_2prime, div2
        ) = _build_tree(
            th_p, p_p, g_p, lp_prime, v, j - 1, eps, H0,
            log_prob_and_grad, M_inv, delta_max, rng
        )
        theta_minus, p_minus, grad_minus = th_m, p_m, g_m
        theta_plus, p_plus, grad_plus = th_pp, p_pp, g_pp

    w_tot = w_prime + w_2prime
    if w_tot > 0.0 and rng.uniform() < (w_2prime / w_tot):
        cand_th = th_2prime
        cand_lp = lp_2prime
        cand_g = g_2prime
        cand_H = H_2prime
    else:
        cand_th = th_prime
        cand_lp = lp_prime
        cand_g = g_prime
        cand_H = H_prime

    u_turn = _check_u_turn(theta_minus, theta_plus, p_minus, p_plus, M_inv)
    s = bool(s_2prime and (not u_turn))
    alpha_tot = a_prime + a_2prime
    n_alpha_tot = na_prime + na_2prime
    divergent = bool(div1 or div2)

    return (
        theta_minus, p_minus, grad_minus,
        theta_plus, p_plus, grad_plus,
        cand_th, cand_lp, cand_g, cand_H,
        w_tot, s, alpha_tot, n_alpha_tot, divergent
    )


def nuts_step(
    theta: np.ndarray,
    log_p: float,
    grad: np.ndarray,
    eps: float,
    log_prob_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    M_inv: np.ndarray,
    max_tree_depth: int = 10,
    delta_max: float = 1000.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, np.ndarray, float, int, bool, float]:
    r"""Execute a single No-U-Turn Sampler transition."""
    if rng is None:
        rng = np.random.default_rng()

    p0 = rng.standard_normal(len(theta)) / np.sqrt(M_inv)
    K0 = 0.5 * float(np.sum(M_inv * (p0 ** 2)))
    H0 = -log_p + K0

    theta_minus = theta.copy()
    theta_plus = theta.copy()
    p_minus = p0.copy()
    p_plus = p0.copy()
    grad_minus = grad.copy()
    grad_plus = grad.copy()

    theta_curr = theta.copy()
    log_p_curr = log_p
    grad_curr = grad.copy()
    H_curr = H0

    w = 1.0
    s = True
    j = 0
    divergence = False
    alpha_sum = 0.0
    n_alpha_sum = 0

    while s and j < max_tree_depth:
        v = 1 if rng.uniform() < 0.5 else -1
        if v == -1:
            (
                th_m, p_m, g_m,
                _, _, _,
                cand_th, cand_lp, cand_g, cand_H,
                w_cand, s_cand, a_cand, na_cand, div
            ) = _build_tree(
                theta_minus, p_minus, grad_minus, log_p_curr, -1, j, eps, H0,
                log_prob_and_grad, M_inv, delta_max, rng
            )
            theta_minus, p_minus, grad_minus = th_m, p_m, g_m
        else:
            (
                _, _, _,
                th_p, p_p, g_p,
                cand_th, cand_lp, cand_g, cand_H,
                w_cand, s_cand, a_cand, na_cand, div
            ) = _build_tree(
                theta_plus, p_plus, grad_plus, log_p_curr, 1, j, eps, H0,
                log_prob_and_grad, M_inv, delta_max, rng
            )
            theta_plus, p_plus, grad_plus = th_p, p_p, g_p

        if div:
            divergence = True

        if s_cand and w_cand > 0.0:
            if rng.uniform() < (w_cand / (w + w_cand)):
                theta_curr = cand_th
                log_p_curr = cand_lp
                grad_curr = cand_g
                H_curr = cand_H

        w += w_cand
        alpha_sum += a_cand
        n_alpha_sum += na_cand

        u_turn = _check_u_turn(theta_minus, theta_plus, p_minus, p_plus, M_inv)
        s = bool(s_cand and (not u_turn))
        j += 1

    bar_alpha = alpha_sum / max(1, n_alpha_sum)
    return theta_curr, log_p_curr, grad_curr, H_curr, j, divergence, float(min(1.0, bar_alpha))


# ---------------------------------------------------------------------------
# Initial Step Size Heuristic (Hoffman-Gelman 2014 Algorithm 4)
# ---------------------------------------------------------------------------

def find_reasonable_step_size(
    theta: np.ndarray,
    log_p: float,
    grad: np.ndarray,
    log_prob_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    M_inv: np.ndarray,
    rng: np.random.Generator,
) -> float:
    r"""Heuristic for finding a numerically stable initial leapfrog step size eps."""
    eps = 1.0
    p = rng.standard_normal(len(theta)) / np.sqrt(M_inv)
    H0 = -log_p + 0.5 * float(np.sum(M_inv * (p ** 2)))

    th_new, p_new, lp_new, g_new = leapfrog(theta, p, grad, eps, log_prob_and_grad, M_inv)
    if not math.isfinite(lp_new) or not np.all(np.isfinite(p_new)):
        H_new = math.inf
    else:
        H_new = -lp_new + 0.5 * float(np.sum(M_inv * (p_new ** 2)))

    diff = -H_new + H0
    a = 1 if (math.isfinite(diff) and math.exp(min(0.0, diff)) > 0.5) else -1

    count = 0
    while count < 30:
        if a == 1 and not (math.isfinite(diff) and math.exp(min(0.0, diff)) > 0.5):
            break
        if a == -1 and not (not math.isfinite(diff) or math.exp(min(0.0, diff)) <= 0.5):
            break
        eps = eps * (2.0 ** a)
        th_new, p_new, lp_new, g_new = leapfrog(theta, p, grad, eps, log_prob_and_grad, M_inv)
        if not math.isfinite(lp_new) or not np.all(np.isfinite(p_new)):
            H_new = math.inf
        else:
            H_new = -lp_new + 0.5 * float(np.sum(M_inv * (p_new ** 2)))
        diff = -H_new + H0
        count += 1
        if eps < 1e-6 or eps > 1e2:
            break

    return float(np.clip(eps, 1e-5, 10.0))


# ---------------------------------------------------------------------------
# Dual Averaging Step Size Adaptation (Nesterov 2009 / Hoffman-Gelman 2014)
# ---------------------------------------------------------------------------

class DualAveraging:
    r"""Dual averaging step size adaptation targeting acceptance probability delta*."""

    def __init__(
        self,
        eps0: float,
        target_accept: float = 0.80,
        gamma: float = 0.05,
        t0: float = 10.0,
        kappa: float = 0.75,
    ) -> None:
        self.target_accept = target_accept
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa
        self.mu = math.log(max(1e-10, 10.0 * eps0))
        self.log_eps = math.log(max(1e-10, eps0))
        self.log_eps_bar = math.log(max(1e-10, eps0))
        self.H_bar = 0.0
        self.m = 0

    def step(self, alpha: float) -> float:
        r"""Update dual averaging accumulator with acceptance statistic alpha."""
        self.m += 1
        eta = 1.0 / (self.m + self.t0)
        self.H_bar = (1.0 - eta) * self.H_bar + eta * (self.target_accept - alpha)
        self.log_eps = self.mu - (math.sqrt(self.m) / self.gamma) * self.H_bar
        eta_m = self.m ** (-self.kappa)
        self.log_eps_bar = eta_m * self.log_eps + (1.0 - eta_m) * self.log_eps_bar
        return float(np.clip(math.exp(self.log_eps), 1e-6, 10.0))

    def get_step_size(self, finalized: bool = False) -> float:
        r"""Return current step size (smoothed log_eps_bar if finalized else log_eps)."""
        val = self.log_eps_bar if finalized else self.log_eps
        return float(np.clip(math.exp(val), 1e-6, 10.0))

    def reset_for_new_metric(self, current_eps: float) -> None:
        r"""Re-center mu and step size accumulator when mass matrix metric updates."""
        self.mu = math.log(max(1e-10, 10.0 * current_eps))
        self.log_eps = math.log(max(1e-10, current_eps))
        self.log_eps_bar = math.log(max(1e-10, current_eps))
        self.H_bar = 0.0
        self.m = 0


# ---------------------------------------------------------------------------
# Welford Online Covariance Estimator for Mass Matrix Adaptation
# ---------------------------------------------------------------------------

class WelfordVariance:
    r"""Welford's online single-pass sample variance accumulator."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim)
        self.M2 = np.zeros(dim)

    def add_sample(self, x: np.ndarray) -> None:
        r"""Accumulate parameter vector sample."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance(self, regularize_prior_var: np.ndarray | None = None) -> np.ndarray:
        r"""Compute sample variance regularized with prior variance shrinkage."""
        if self.count <= 1:
            if regularize_prior_var is not None:
                return np.asarray(regularize_prior_var, dtype=float)
            return np.ones(self.dim)

        sample_var = self.M2 / max(1, self.count - 1)
        if regularize_prior_var is not None:
            prior_v = np.asarray(regularize_prior_var, dtype=float)
            w = self.count / (self.count + 5.0)
            reg_var = w * sample_var + (1.0 - w) * prior_v
        else:
            w = self.count / (self.count + 5.0)
            reg_var = w * sample_var + (1.0 - w) * 1.0

        return np.clip(reg_var, 1e-6, 1e6)

    def reset(self) -> None:
        r"""Reset accumulator for next adaptation window."""
        self.count = 0
        self.mean = np.zeros(self.dim)
        self.M2 = np.zeros(self.dim)


# ---------------------------------------------------------------------------
# Main NUTS Sampler Driver
# ---------------------------------------------------------------------------

def nuts_sample(
    target_log_prob_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    init_params: np.ndarray,
    n_draws: int = 1000,
    n_chains: int = 2,
    warmup: int = 500,
    target_accept: float = 0.80,
    max_tree_depth: int = 10,
    step_size: float | None = None,
    adapt_step_size: bool = True,
    adapt_mass_matrix: bool = True,
    mass_matrix_diag: np.ndarray | None = None,
    param_names: Sequence[str] | None = None,
    seed: int = 0,
    model_name: str = "nuts_model",
    mode: dict[str, float] | None = None,
    mode_hessian_inv: np.ndarray | None = None,
    data_n_obs: int = 0,
    record_warmup: bool = False,
) -> NUTSResult:
    r"""Sample posterior distribution using Hamiltonian Monte Carlo / No-U-Turn Sampler."""
    init_arr = np.asarray(init_params, dtype=float)
    if init_arr.ndim == 1:
        n_params = len(init_arr)
    else:
        n_params = init_arr.shape[1]

    if param_names is None:
        p_names = tuple(f"param_{i}" for i in range(n_params))
    else:
        p_names = tuple(param_names)

    if mass_matrix_diag is not None:
        base_M_inv = np.asarray(mass_matrix_diag, dtype=float)
    else:
        base_M_inv = np.ones(n_params)

    draws_all = np.empty((n_chains, n_draws, n_params), dtype=float)
    log_post_all = np.empty((n_chains, n_draws), dtype=float)
    tree_depths_all = np.empty((n_chains, n_draws), dtype=int)
    divergences_all = np.empty((n_chains, n_draws), dtype=bool)
    energy_all = np.empty((n_chains, n_draws), dtype=float)
    warmup_draws_all = np.empty((n_chains, warmup, n_params), dtype=float) if record_warmup else None

    accept_rates_list = []
    step_sizes_list = []
    adapted_M_inv_list = []

    if warmup >= 150:
        fast_init_end = 75
        fast_final_start = warmup - 50
    else:
        fast_init_end = warmup
        fast_final_start = warmup

    for c in range(n_chains):
        rng = np.random.default_rng(seed + c)

        if init_arr.ndim == 2:
            theta = init_arr[c].copy()
        else:
            if c == 0:
                theta = init_arr.copy()
            else:
                perturb = rng.normal(0.0, 0.01 * np.sqrt(np.clip(base_M_inv, 1e-4, 1e4)))
                theta = init_arr + perturb

        lp, grad = target_log_prob_and_grad(theta)
        if not math.isfinite(lp):
            theta = init_arr.copy()
            lp, grad = target_log_prob_and_grad(theta)

        M_inv = base_M_inv.copy()

        if step_size is not None:
            eps = float(step_size)
        else:
            eps = find_reasonable_step_size(theta, lp, grad, target_log_prob_and_grad, M_inv, rng)

        da = DualAveraging(eps, target_accept=target_accept)
        welford = WelfordVariance(n_params)

        slow_window_size = 25
        slow_window_end = fast_init_end + slow_window_size

        for m in range(warmup):
            theta, lp, grad, H, depth, div, alpha = nuts_step(
                theta, lp, grad, eps, target_log_prob_and_grad, M_inv,
                max_tree_depth=max_tree_depth, rng=rng
            )

            if record_warmup and warmup_draws_all is not None:
                warmup_draws_all[c, m] = theta

            if adapt_step_size:
                eps = da.step(alpha)

            if adapt_mass_matrix and warmup >= 150:
                if fast_init_end <= m < fast_final_start:
                    welford.add_sample(theta)
                    if m == slow_window_end - 1:
                        M_inv = welford.variance(regularize_prior_var=base_M_inv)
                        da.reset_for_new_metric(eps)
                        welford.reset()
                        slow_window_size *= 2
                        slow_window_end = min(m + 1 + slow_window_size, fast_final_start)
                        if fast_final_start - slow_window_end < 25:
                            slow_window_end = fast_final_start

        if adapt_step_size:
            eps = da.get_step_size(finalized=True)

        step_sizes_list.append(eps)
        adapted_M_inv_list.append(M_inv.copy())

        alphas_retained = np.empty(n_draws, dtype=float)
        for t in range(n_draws):
            theta, lp, grad, H, depth, div, alpha = nuts_step(
                theta, lp, grad, eps, target_log_prob_and_grad, M_inv,
                max_tree_depth=max_tree_depth, rng=rng
            )
            draws_all[c, t] = theta
            log_post_all[c, t] = lp
            tree_depths_all[c, t] = depth
            divergences_all[c, t] = div
            energy_all[c, t] = H
            alphas_retained[t] = alpha

        accept_rates_list.append(float(np.mean(alphas_retained)))

    lp_mode_val = float(mode.get("log_post_mode")) if (mode and "log_post_mode" in mode) else None

    return NUTSResult(
        draws=draws_all,
        param_names=p_names,
        log_posterior_trace=log_post_all,
        accept_rates=tuple(accept_rates_list),
        step_sizes=tuple(step_sizes_list),
        tree_depths=tree_depths_all,
        divergences=divergences_all,
        energy_trace=energy_all,
        mass_matrix_diag=np.array(adapted_M_inv_list),
        mode=mode,
        mode_hessian_inv=mode_hessian_inv,
        n_warmup=warmup,
        data_n_obs=data_n_obs,
        seed=seed,
        model_name=model_name,
        log_post_mode=lp_mode_val,
        warmup_draws=warmup_draws_all,
    )


__all__ = [
    "NUTSResult",
    "leapfrog",
    "nuts_step",
    "nuts_sample",
    "find_reasonable_step_size",
    "DualAveraging",
    "WelfordVariance",
    "compute_split_rhat",
    "compute_bulk_ess",
    "compute_tail_ess",
    "compute_ebfmi",
]
