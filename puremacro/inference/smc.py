r"""Pure-NumPy Vectorized Tempered Sequential Monte Carlo (SMC) Engine.

Following Herbst & Schorfheide (2014, 2016), provides:
1. Adaptive ESS-targeted tempering schedule ($\phi_n \in [0, 1]$).
2. Pure-NumPy vectorized particle evaluation and systematic resampling.
3. Random-Walk (RW-MH), Metropolis-Adjusted Langevin Algorithm (MALA),
   and Hamiltonian Monte Carlo (HMC) leapfrog mutation proposals driven by
   `DualArray`, Kalman, or Whittle score gradients.
4. Exact log marginal data density ($\ln p(Y)$) and numerical standard error
   estimation with importance-sampling diagnostics.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import scipy.stats

from puremacro._linalg import safe_cholesky
from puremacro.core.protocol import DifferentiableModel
from puremacro.dsge.priors import (
    Prior,
    _sample_prior,
    ensure_prior,
    grad_log_prior,
    log_prior,
    param_bounds,
    param_names as get_param_names,
)
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst, result_to_interactive_html

__all__ = [
    "AdaptiveTemperingSchedule",
    "SMCResult",
    "smc_sample",
]


# ===========================================================================
# 1. Adaptive Tempering Schedule
# ===========================================================================

class AdaptiveTemperingSchedule:
    """Adaptive tempering schedule manager for Sequential Monte Carlo.

    Parameters
    ----------
    target_ess_ratio : float, default 0.5
        Target ESS retention ratio from stage to stage (ESS(phi_n) = target_ess_ratio * ESS_{n-1}).
    min_step : float, default 1e-4
        Minimum tempering step Delta phi_n.
    max_stages : int, default 100
        Maximum allowed tempering stages before terminating at phi = 1.0.
    phi_init : float, default 0.0
        Initial tempering parameter phi_0.
    """

    def __init__(
        self,
        target_ess_ratio: float = 0.5,
        min_step: float = 1e-4,
        max_stages: int = 100,
        phi_init: float = 0.0,
    ) -> None:
        if not (0.0 < target_ess_ratio < 1.0):
            raise ValueError(f"target_ess_ratio must be in (0, 1), got {target_ess_ratio}")
        self.target_ess_ratio = float(target_ess_ratio)
        self.min_step = float(min_step)
        self.max_stages = int(max_stages)
        self.phi_curr = float(phi_init)

    def next_phi(
        self,
        phi_curr: float,
        loglik_particles: np.ndarray,
        weights: np.ndarray,
        stage_idx: int,
    ) -> float:
        """Find next phi in (phi_curr, 1.0] targeting ESS decay."""
        if stage_idx >= self.max_stages or phi_curr + self.min_step >= 1.0:
            self.phi_curr = 1.0
            return 1.0

        w_norm = weights / np.sum(weights)
        ess_curr = 1.0 / np.sum(w_norm ** 2)
        n_p = float(len(weights))
        min_floor = self.target_ess_ratio * n_p
        target_ess = max(self.target_ess_ratio * ess_curr, min_floor)

        def _predict_ess(phi: float) -> float:
            d_phi = phi - phi_curr
            v = d_phi * loglik_particles
            v_max = np.max(v)
            w = w_norm * np.exp(v - v_max)
            sw = np.sum(w)
            if sw <= 0.0 or not np.isfinite(sw):
                return 0.0
            p_norm = w / sw
            return 1.0 / np.sum(p_norm ** 2)

        # Check endpoint phi = 1.0 only if predicted ESS remains above target floor
        if _predict_ess(1.0) >= min_floor:
            self.phi_curr = 1.0
            return 1.0

        # Monotonic 1D bisection on [phi_curr + min_step, 1.0]
        # Target with small positive buffer so numerical tolerance never dips below min_floor
        target_solve = min(target_ess + 1e-3 * n_p, 0.99 * n_p)
        low = phi_curr + self.min_step
        high = 1.0

        if _predict_ess(low) <= target_solve:
            self.phi_curr = low
            return low

        for _ in range(40):
            mid = 0.5 * (low + high)
            f_mid = _predict_ess(mid) - target_solve
            if abs(f_mid) < 1e-4 or (high - low) < 1e-6:
                break
            if f_mid > 0:
                low = mid
            else:
                high = mid

        phi_next = 0.5 * (low + high)
        self.phi_curr = phi_next
        return phi_next

    def is_finished(self, phi: float) -> bool:
        """Return True when tempering parameter has reached 1.0."""
        return phi >= 1.0 - 1e-8


# ===========================================================================
# 2. Resampling Helper
# ===========================================================================

def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """O(N) Systematic Resampling."""
    N = len(weights)
    W = weights / np.sum(weights)
    u0 = rng.uniform(0.0, 1.0 / N)
    positions = u0 + np.arange(N) / N
    indexes = np.zeros(N, dtype=int)
    cumulative_sum = np.cumsum(W)
    i, j = 0, 0
    while i < N and j < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    if i < N:
        indexes[i:] = N - 1
    return indexes


# ===========================================================================
# 3. Prior Sampling Helper
# ===========================================================================

def _sample_from_prior(spec: Any, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n independent samples from a supported DSGE prior specification."""
    return _sample_prior(spec, n, rng)


# ===========================================================================
# 4. Result Dataclass
# ===========================================================================

@dataclass(frozen=True)
class SMCResult:
    """Sequential Monte Carlo posterior sampling result.

    Attributes
    ----------
    particles : np.ndarray of shape (N, P)
        Posterior parameter draws.
    weights : np.ndarray of shape (N,)
        Final normalized particle weights.
    stage_tempering : np.ndarray of shape (N_stages + 1,)
        Sequence of tempering stages phi_0, phi_1, ..., phi_{N_stages} = 1.0.
    log_mdd : float
        Estimated log marginal data density ln p(Y).
    mdd_se : float
        Numerical standard error of log marginal data density.
    acceptance_rates : np.ndarray of shape (N_stages,)
        Mutation acceptance rates per stage.
    ess_history : np.ndarray of shape (N_stages + 1,)
        Effective sample size across stages.
    mutation_type : str
        Mutation algorithm ('rw', 'mala', or 'hmc').
    param_names : tuple[str, ...]
        Parameter labels.
    elapsed_seconds : float
        Wall-clock sampling time in seconds.
    metadata : dict[str, Any]
        Additional diagnostics.
    """

    particles: np.ndarray
    weights: np.ndarray
    stage_tempering: np.ndarray
    log_mdd: float
    mdd_se: float
    acceptance_rates: np.ndarray
    ess_history: np.ndarray
    mutation_type: str
    param_names: tuple[str, ...]
    elapsed_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Compute weighted posterior moments table."""
        w = self.weights / np.sum(self.weights)
        P = self.particles.shape[1]
        means = np.sum(self.particles * w[:, None], axis=0)
        variances = np.sum(w[:, None] * (self.particles - means) ** 2, axis=0)
        stds = np.sqrt(np.maximum(1e-16, variances))

        # Weighted percentiles
        p05 = np.zeros(P)
        p50 = np.zeros(P)
        p95 = np.zeros(P)
        for j in range(P):
            order = np.argsort(self.particles[:, j])
            sorted_x = self.particles[order, j]
            sorted_w = w[order]
            cum_w = np.cumsum(sorted_w)
            p05[j] = float(np.interp(0.05, cum_w, sorted_x))
            p50[j] = float(np.interp(0.50, cum_w, sorted_x))
            p95[j] = float(np.interp(0.95, cum_w, sorted_x))

        ess = float(1.0 / np.sum(w ** 2))
        df = pd.DataFrame(
            {
                "Mean": means,
                "Std": stds,
                "Median": p50,
                "P05": p05,
                "P95": p95,
                "ESS": np.full(P, ess),
            },
            index=list(self.param_names),
        )
        return df

    def summary(self) -> str:
        """Produce formatted summary text."""
        lines = [
            f"PureMacro SMC Posterior Estimation Summary",
            f"==========================================",
            f"Mutation Method       : {self.mutation_type.upper()}",
            f"Particles (N)         : {len(self.weights)}",
            f"Stages (N_phi)        : {len(self.stage_tempering) - 1}",
            f"Log Marginal Data Dens: {self.log_mdd:.4f} ± {self.mdd_se:.4f}",
            f"Elapsed Time          : {self.elapsed_seconds:.2f} s",
            f"Mean Acceptance Rate  : {float(np.mean(self.acceptance_rates)):.3f}",
            f"\nPosterior Parameter Estimates:\n",
            self.to_dataframe().to_string(float_format=lambda x: f"{x:.4f}"),
        ]
        return "\n".join(lines)

    def to_markdown(self, **kwargs: Any) -> str:
        """Render posterior table as Markdown."""
        return _df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render posterior table as LaTeX tabular."""
        return _df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render posterior table as Typst table."""
        return _df_to_typst(self.to_dataframe(), **kwargs)

    def to_interactive_html(self, **kwargs: Any) -> str:
        """Render self-contained HTML5 + SVG representation."""
        title = kwargs.pop("title", f"PureMacro SMC Posterior ({self.mutation_type.upper()})")
        return result_to_interactive_html(self.to_dataframe(), title=title, **kwargs)

    def plot_stages(self, **kwargs: Any) -> Any:
        """Plot stage tempering progression and ESS decay."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        stages = np.arange(len(self.stage_tempering))
        ax1.plot(stages, self.stage_tempering, marker="o", color="#1f77b4", lw=2)
        ax1.set_title("Tempering Schedule $\\phi_n$")
        ax1.set_xlabel("Stage $n$")
        ax1.set_ylabel("$\\phi$")
        ax1.grid(True, alpha=0.3)

        ax2.plot(stages, self.ess_history, marker="s", color="#2ca02c", lw=2)
        ax2.set_title("Effective Sample Size (ESS)")
        ax2.set_xlabel("Stage $n$")
        ax2.set_ylabel("ESS")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_posterior(self, **kwargs: Any) -> Any:
        """Plot marginal posterior densities."""
        import matplotlib.pyplot as plt

        P = len(self.param_names)
        cols = min(3, P)
        rows = math.ceil(P / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
        w = self.weights / np.sum(self.weights)

        for j, name in enumerate(self.param_names):
            r, c = divmod(j, cols)
            ax = axes[r, c]
            ax.hist(self.particles[:, j], weights=w, bins=25, density=True, color="#1f77b4", alpha=0.6, edgecolor="black")
            ax.set_title(name)
            ax.grid(True, alpha=0.3)

        # Hide extra subplots
        for j in range(P, rows * cols):
            r, c = divmod(j, cols)
            axes[r, c].set_visible(False)

        fig.tight_layout()
        return fig

    def plot(self, **kwargs: Any) -> Any:
        """Default plot alias."""
        return self.plot_posterior(**kwargs)


# ===========================================================================
# 5. Core SMC Sampler
# ===========================================================================

def smc_sample(
    loglik_and_grad_fn: Union[Callable[[np.ndarray], Tuple[float, np.ndarray]], DifferentiableModel],
    priors: Mapping[str, Any],
    data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    *,
    n_particles: int = 1000,
    mutation: str = "mala",
    schedule: Union[str, AdaptiveTemperingSchedule] = "adaptive",
    target_ess: float = 0.5,
    n_steps: int = 2,
    leapfrog_steps: int = 5,
    step_size_init: float = 0.1,
    seed: Optional[int] = 0,
    batch_size: Optional[int] = None,
    ridge: float = 1e-6,
) -> SMCResult:
    """Execute vectorized tempered Sequential Monte Carlo sampling.

    Parameters
    ----------
    loglik_and_grad_fn : Callable or DifferentiableModel
        Function returning `(loglik, grad)` given parameter vector, or a DifferentiableModel.
    priors : Mapping[str, Any]
        Dictionary of prior distribution specifications.
    data : array-like, optional
        Observed sample dataset forwarded to DifferentiableModel.
    n_particles : int, default 1000
        Number of particles N.
    mutation : {'rw', 'mala', 'hmc'}, default 'mala'
        Mutation operator algorithm.
    schedule : {'adaptive', 'fixed'} or AdaptiveTemperingSchedule, default 'adaptive'
        Tempering schedule strategy.
    target_ess : float, default 0.5
        Resampling trigger ratio (ESS < target_ess * N).
    n_steps : int, default 2
        Number of mutation steps per particle per stage.
    leapfrog_steps : int, default 5
        Number of leapfrog integration steps (for HMC mutation).
    step_size_init : float, default 0.1
        Initial mutation step size / proposal scaling.
    seed : int, optional, default 0
        Random seed for reproducibility.
    batch_size : int, optional
        Batch size for particle evaluation.
    ridge : float, default 1e-6
        Ridge added to particle covariance matrix.

    Returns
    -------
    SMCResult
        Rich dataclass containing posterior particles, weights, log MDD, and diagnostics.
    """
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)

    # 1. Parse priors and parameter dimensions
    parsed_priors = {k: ensure_prior(v) for k, v in priors.items()}
    param_names = get_param_names(parsed_priors)
    P = len(param_names)
    bounds = param_bounds(parsed_priors)
    lb_vec = np.array([b[0] for b in bounds], dtype=float)
    ub_vec = np.array([b[1] for b in bounds], dtype=float)

    # Unified evaluator for loglik and gradient
    is_differentiable_model = isinstance(loglik_and_grad_fn, DifferentiableModel)

    def _eval_loglik_and_grad(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        if np.any(theta < lb_vec) or np.any(theta > ub_vec):
            return -math.inf, np.zeros(P, dtype=float)
        if isinstance(loglik_and_grad_fn, DifferentiableModel):
            ll, g = loglik_and_grad_fn.loglik_and_grad(params=theta, data=data)
        elif callable(loglik_and_grad_fn):
            ll, g = loglik_and_grad_fn(theta)
        else:
            raise TypeError("loglik_and_grad_fn must be a callable or DifferentiableModel")
        return float(ll), np.asarray(g, dtype=float)

    def _eval_prior_and_grad(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        p_dict = {name: float(theta[i]) for i, name in enumerate(param_names)}
        lp = log_prior(p_dict, parsed_priors)
        if not math.isfinite(lp):
            return -math.inf, np.zeros(P, dtype=float)
        g_p = grad_log_prior(theta, parsed_priors, param_names)
        return float(lp), np.asarray(g_p, dtype=float)

    # 2. Draw initial particles from prior (stage 0: phi_0 = 0)
    particles = np.zeros((n_particles, P), dtype=float)
    for j, name in enumerate(param_names):
        particles[:, j] = _sample_from_prior(parsed_priors[name], n_particles, rng)

    # Initial likelihoods and gradients
    logliks = np.zeros(n_particles, dtype=float)
    grads_lik = np.zeros((n_particles, P), dtype=float)
    log_priors = np.zeros(n_particles, dtype=float)
    grads_prior = np.zeros((n_particles, P), dtype=float)

    for i in range(n_particles):
        logliks[i], grads_lik[i] = _eval_loglik_and_grad(particles[i])
        log_priors[i], grads_prior[i] = _eval_prior_and_grad(particles[i])

    weights = np.full(n_particles, 1.0 / n_particles, dtype=float)

    # Tempering schedule initialization
    if isinstance(schedule, AdaptiveTemperingSchedule):
        sched_mgr = schedule
    elif schedule == "fixed":
        fixed_phis = np.linspace(0.0, 1.0, 50)
        sched_mgr = None
    else:
        sched_mgr = AdaptiveTemperingSchedule(target_ess_ratio=target_ess, min_step=1e-4)

    phi_curr = 0.0
    phi_history = [0.0]
    ess_history = [float(n_particles)]
    acc_history = []
    log_c_list: List[float] = []
    w_tilde_var_list: List[float] = []

    step_size = float(step_size_init)
    stage_idx = 0

    mutation_lower = mutation.lower()
    if mutation_lower not in ("rw", "mala", "hmc"):
        raise ValueError(f"Unknown mutation algorithm {mutation!r}; expected 'rw', 'mala', or 'hmc'")

    # 3. Main SMC Tempering Loop
    while phi_curr < 1.0 - 1e-8 and stage_idx < 150:
        stage_idx += 1

        # Determine next phi_n
        if sched_mgr is not None:
            phi_next = sched_mgr.next_phi(phi_curr, logliks, weights, stage_idx)
        else:
            phi_next = float(fixed_phis[stage_idx]) if stage_idx < len(fixed_phis) else 1.0

        d_phi = phi_next - phi_curr

        # --- Correction Step & Incremental Weights ---
        v = d_phi * logliks
        v_max = float(np.max(v))
        w_tilde = np.exp(v - v_max)
        w_inc = weights * w_tilde
        sum_w_inc = float(np.sum(w_inc))

        # Marginal data density increment ln c_n
        log_c_n = v_max + math.log(max(1e-300, sum_w_inc))
        log_c_list.append(log_c_n)

        # Standard error calculation term
        bar_w = sum_w_inc
        w_diff_sq = np.sum(weights * ((w_tilde - bar_w) ** 2)) / (n_particles * max(1e-300, bar_w ** 2))
        w_tilde_var_list.append(float(w_diff_sq))

        # Normalized weights
        weights = w_inc / sum_w_inc
        ess_n = float(1.0 / np.sum(weights ** 2))
        ess_history.append(ess_n)
        phi_history.append(phi_next)

        # --- Selection (Resampling) Step ---
        if ess_n <= target_ess * n_particles + 0.002 * n_particles:
            resample_idx = _systematic_resample(weights, rng)
            particles = particles[resample_idx].copy()
            logliks = logliks[resample_idx].copy()
            grads_lik = grads_lik[resample_idx].copy()
            log_priors = log_priors[resample_idx].copy()
            grads_prior = grads_prior[resample_idx].copy()
            weights = np.full(n_particles, 1.0 / n_particles, dtype=float)

        # Particle empirical covariance for mutation proposals
        theta_bar = np.sum(particles * weights[:, None], axis=0)
        cov_theta = (particles - theta_bar).T @ (weights[:, None] * (particles - theta_bar)) + ridge * np.eye(P)
        L_cov = safe_cholesky(cov_theta, name="SMC particle covariance")

        # --- Mutation Step ---
        n_accepted = 0
        total_proposals = n_particles * n_steps

        for _ in range(n_steps):
            for i in range(n_particles):
                th = particles[i]
                ll_curr = logliks[i]
                lp_curr = log_priors[i]
                pi_curr = phi_next * ll_curr + lp_curr
                g_curr = phi_next * grads_lik[i] + grads_prior[i]

                if mutation_lower == "rw":
                    # Random-Walk Metropolis proposal
                    z = rng.standard_normal(P)
                    th_prop = th + step_size * (L_cov @ z)

                    # Prior bounds check
                    if np.any(th_prop < lb_vec) or np.any(th_prop > ub_vec):
                        continue

                    ll_prop, glik_prop = _eval_loglik_and_grad(th_prop)
                    lp_prop, gprior_prop = _eval_prior_and_grad(th_prop)
                    if not math.isfinite(lp_prop):
                        continue

                    pi_prop = phi_next * ll_prop + lp_prop
                    log_alpha = pi_prop - pi_curr

                    if math.log(rng.uniform(1e-12, 1.0)) < log_alpha:
                        particles[i] = th_prop
                        logliks[i] = ll_prop
                        grads_lik[i] = glik_prop
                        log_priors[i] = lp_prop
                        grads_prior[i] = gprior_prop
                        n_accepted += 1

                elif mutation_lower == "mala":
                    # Metropolis-Adjusted Langevin Algorithm
                    # Drift mu(theta) = theta + 0.5 * eps^2 * Sigma * grad
                    drift_term = 0.5 * (step_size ** 2) * (cov_theta @ g_curr)
                    z = rng.standard_normal(P)
                    th_prop = th + drift_term + step_size * (L_cov @ z)

                    if np.any(th_prop < lb_vec) or np.any(th_prop > ub_vec):
                        continue

                    ll_prop, glik_prop = _eval_loglik_and_grad(th_prop)
                    lp_prop, gprior_prop = _eval_prior_and_grad(th_prop)
                    if not math.isfinite(lp_prop):
                        continue

                    pi_prop = phi_next * ll_prop + lp_prop
                    g_prop = phi_next * glik_prop + gprior_prop
                    drift_rev = 0.5 * (step_size ** 2) * (cov_theta @ g_prop)

                    # Hastings ratio: log q(th | th_prop) - log q(th_prop | th)
                    diff_fwd = z  # L_cov^{-1} (th_prop - mu) / eps = z
                    diff_rev = th - (th_prop + drift_rev)
                    sol_rev = np.linalg.solve(L_cov, diff_rev)
                    quad_rev = np.sum(sol_rev ** 2) / (step_size ** 2)
                    quad_fwd = np.sum(diff_fwd ** 2)

                    log_q_ratio = -0.5 * (quad_rev - quad_fwd)
                    log_alpha = pi_prop - pi_curr + log_q_ratio

                    if math.log(rng.uniform(1e-12, 1.0)) < log_alpha:
                        particles[i] = th_prop
                        logliks[i] = ll_prop
                        grads_lik[i] = glik_prop
                        log_priors[i] = lp_prop
                        grads_prior[i] = gprior_prop
                        n_accepted += 1

                elif mutation_lower == "hmc":
                    # Hamiltonian Monte Carlo leapfrog
                    # Draw momentum p ~ N(0, Sigma^{-1}) => p = L_cov^{-T} z
                    z = rng.standard_normal(P)
                    p_mom = np.linalg.solve(L_cov.T, z)
                    kinetic_init = 0.5 * np.sum(z ** 2)

                    th_traj = th.copy()
                    g_traj = g_curr.copy()
                    p_traj = p_mom.copy()
                    bad_bounds = False

                    # Leapfrog integration
                    for _l in range(leapfrog_steps):
                        p_traj += 0.5 * step_size * g_traj
                        th_traj += step_size * (cov_theta @ p_traj)
                        if np.any(th_traj < lb_vec) or np.any(th_traj > ub_vec):
                            bad_bounds = True
                            break
                        ll_step, glik_step = _eval_loglik_and_grad(th_traj)
                        lp_step, gprior_step = _eval_prior_and_grad(th_traj)
                        g_traj = phi_next * glik_step + gprior_step
                        p_traj += 0.5 * step_size * g_traj

                    if bad_bounds or not math.isfinite(lp_step):
                        continue

                    pi_prop = phi_next * ll_step + lp_step
                    # Kinetic energy of proposal p_traj^T Sigma p_traj
                    L_p = L_cov.T @ p_traj
                    kinetic_prop = 0.5 * np.sum(L_p ** 2)

                    delta_H = -pi_prop + kinetic_prop - (-pi_curr + kinetic_init)
                    log_alpha = -delta_H

                    if math.log(rng.uniform(1e-12, 1.0)) < log_alpha:
                        particles[i] = th_traj
                        logliks[i] = ll_step
                        grads_lik[i] = glik_step
                        log_priors[i] = lp_step
                        grads_prior[i] = gprior_step
                        n_accepted += 1

        stage_acc_rate = n_accepted / max(1, total_proposals)
        acc_history.append(stage_acc_rate)

        # Adaptive step size tuning
        target_acc = 0.574 if mutation_lower == "mala" else (0.65 if mutation_lower == "hmc" else 0.25)
        step_size *= float(0.95 + 0.10 * (1.0 / (1.0 + math.exp(-16.0 * (stage_acc_rate - target_acc)))))
        step_size = max(1e-5, min(2.0, step_size))

        phi_curr = phi_next

    # 4. Final log marginal data density & standard error
    total_log_mdd = float(np.sum(log_c_list))
    mdd_se = float(math.sqrt(max(1e-12, np.sum(w_tilde_var_list))))

    elapsed_time = time.perf_counter() - start_time

    return SMCResult(
        particles=particles,
        weights=weights,
        stage_tempering=np.array(phi_history, dtype=float),
        log_mdd=total_log_mdd,
        mdd_se=mdd_se,
        acceptance_rates=np.array(acc_history, dtype=float),
        ess_history=np.array(ess_history, dtype=float),
        mutation_type=mutation_lower,
        param_names=param_names,
        elapsed_seconds=elapsed_time,
        metadata={
            "n_stages": stage_idx,
            "final_step_size": step_size,
        },
    )
