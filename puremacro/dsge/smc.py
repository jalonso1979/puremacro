"""Sequential Monte Carlo (SMC) Sampler and Particle Filtering.

Implements the Herbst & Schorfheide (2014, 2015) Sequential Monte Carlo sampler
for Bayesian estimation of DSGE models and nonlinear bootstrap particle filtering
for order-2 and order-3 pruned state spaces (Andreasen et al. 2018).

Core Capabilities:
1. Sequential Monte Carlo Sampler (`SMCSampler`, `smc_estimate`):
   - Fixed power tempering schedule: $\\phi_n = (n / N_\\phi)^\\lambda$ (default $\\lambda = 2.1$).
   - Adaptive tempering schedule: bisection root-finding on predicted ESS drop:
     $ESS(\\phi) = \\alpha^* \\cdot ESS_{n-1}$.
   - Exact $O(N)$ Systematic Resampling triggered when $ESS_n < \\tau N_{part}$ (default $\\tau = 0.5$).
   - Particle mutation: adaptive random-walk Metropolis-Hastings with weighted empirical proposal
     covariance $\\Sigma_n = c_n^2 \\sum_i W_{i, n} (\\theta_i - \\bar{\\theta}_n)(\\theta_i - \\bar{\\theta}_n)^\\top$
     and scale adaptation targeting ~25% acceptance rate:
     $c_{n+1} = c_n \\cdot (0.95 + 0.10 \\frac{\\exp(16(\\hat{\\alpha}_n - 0.25))}{1 + \\exp(16(\\hat{\\alpha}_n - 0.25))})$.
   - Exact Marginal Data Density (MDD):
     $\\ln \\hat{p}(Y) = \\sum_{n=1}^{N_\\phi} \\ln \\left( \\sum_{i=1}^{N_{part}} W_{i, n-1} \\tilde{w}_{i, n} \\right)$
     with numerical standard error computation.
2. Bootstrap Particle Filter (`bootstrap_particle_filter`):
   - Likelihood evaluation for nonlinear order-2 and order-3 pruned state spaces.
   - Low-weight rejuvenation and robust log-sum-exp stabilization under extreme outliers.
   - Returns log-likelihood and filtered state trajectories.
3. Presentation Contract (`SMCResult`):
   - Attributes: `particles`, `weights`, `stage_tempering`, `mdd`, `mdd_se`, `acceptance_rates`,
     `ess_history`, `posterior_summary`.
   - Methods: `.summary()`, `.plot_stages()`, `.plot_posterior()`, `.plot()`, `.to_markdown()`,
     `.to_latex()`, `.to_typst()`, `.marginal_likelihood()`, `.posterior_table()`.

Strictly pure Python under the Pyodide 4-package core: numpy, scipy, pandas, matplotlib only.

References
----------
Herbst, E. and Schorfheide, F. (2014). Sequential Monte Carlo Sampling for DSGE Models.
    Journal of Applied Econometrics 29(7), 1073-1098.
Herbst, E. and Schorfheide, F. (2015). Bayesian Estimation of DSGE Models.
    Princeton University Press.
Andreasen, M. M., Fernández-Villaverde, J., and Rubio-Ramírez, J. F. (2018).
    The Pruned State-Space System for Higher-Order DSGE Models: Theory and Econometrics.
    Review of Economic Studies 85(1), 1-49.
Kitagawa, G. (1996). Monte Carlo Filter and Smoother for Non-Gaussian Nonlinear State
    Space Models. Journal of Computational and Graphical Statistics 5(1), 1-25.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize
import scipy.stats

from puremacro.dsge.priors import log_prior, Prior, NormalPrior, BetaPrior, GammaPrior, UniformPrior


# ---------------------------------------------------------------------------
# Systematic Resampling Algorithm in O(N)
# ---------------------------------------------------------------------------

def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Deterministic, low-variance systematic resampling in exact O(N) time.

    Given normalized weights W summing to 1:
    1. Draw single uniform offset u ~ U[0, 1/N).
    2. Form equidistant target points U_i = u + (i - 1)/N for i = 1, ..., N.
    3. Traverse cumulative weights C and targets U in a single linear sweep.

    Parameters
    ----------
    weights : ndarray
        Normalized particle weights summing to 1.0.
    rng : np.random.Generator
        NumPy random number generator.

    Returns
    -------
    indices : ndarray of int
        Resampled particle index array of length N.
    """
    N = len(weights)
    if N <= 0:
        return np.empty(0, dtype=int)
    if N == 1:
        return np.zeros(1, dtype=int)

    w = np.asarray(weights, dtype=float)
    total_w = np.sum(w)
    if total_w <= 0.0 or not np.isfinite(total_w):
        w = np.full(N, 1.0 / N)
    else:
        w = w / total_w

    u = float(rng.uniform(0.0, 1.0 / N))
    targets = u + np.arange(N, dtype=float) / N
    cumsum = np.cumsum(w)
    cumsum[-1] = 1.0  # Guard against floating-point rounding

    indices = np.empty(N, dtype=int)
    k = 0
    for i in range(N):
        while k < N - 1 and targets[i] > cumsum[k]:
            k += 1
        indices[i] = k

    return indices


# ---------------------------------------------------------------------------
# Tempering Schedules
# ---------------------------------------------------------------------------

def fixed_tempering_schedule(n_stages: int, lambda_param: float = 2.1) -> np.ndarray:
    """Compute fixed power tempering schedule phi_n = (n / N_phi)^lambda.

    Parameters
    ----------
    n_stages : int
        Number of intermediate stages N_phi.
    lambda_param : float, default 2.1
        Curvature parameter concentrating steps near phi = 1.

    Returns
    -------
    schedule : ndarray of shape (n_stages + 1,)
        Monotonically increasing array with phi_0 = 0.0 and phi_{N_phi} = 1.0.
    """
    if n_stages < 1:
        raise ValueError(f"n_stages must be >= 1, got {n_stages}")
    if lambda_param <= 0:
        raise ValueError(f"lambda_param must be > 0, got {lambda_param}")
    n_grid = np.arange(n_stages + 1, dtype=float) / float(n_stages)
    phi = n_grid ** float(lambda_param)
    phi[0] = 0.0
    phi[-1] = 1.0
    return phi


def solve_adaptive_phi(
    phi_prev: float,
    log_liks: np.ndarray,
    weights_prev: np.ndarray,
    target_ess_ratio: float = 0.95,
    min_step: float = 1e-4,
) -> float:
    """Solve for next tempering parameter phi_n via 1D bisection on predicted ESS.

    Matches ESS(phi) = target_ess_ratio * ESS_{n-1}.

    Parameters
    ----------
    phi_prev : float
        Current tempering parameter phi_{n-1}.
    log_liks : ndarray
        Particle log-likelihood evaluations ln L_i.
    weights_prev : ndarray
        Current normalized weights W_{i, n-1}.
    target_ess_ratio : float, default 0.95
        Target ESS ratio in (0, 1).
    min_step : float, default 1e-4
        Minimum step size forward.

    Returns
    -------
    phi_next : float
        Selected tempering parameter in (phi_prev, 1.0].
    """
    if phi_prev >= 1.0 - 1e-7:
        return 1.0

    ess_prev = 1.0 / np.sum(weights_prev ** 2)
    target_ess = target_ess_ratio * ess_prev

    def ess_at_phi(phi_val: float) -> float:
        delta = phi_val - phi_prev
        v = delta * log_liks
        v_max = np.max(v)
        if not np.isfinite(v_max):
            return 1.0
        w_tilde = np.exp(v - v_max)
        w_comb = weights_prev * w_tilde
        sum_w = np.sum(w_comb)
        if sum_w <= 0.0 or not np.isfinite(sum_w):
            return 1.0
        W_norm = w_comb / sum_w
        return 1.0 / np.sum(W_norm ** 2)

    # Check terminal endpoint phi = 1.0
    ess_at_1 = ess_at_phi(1.0)
    if ess_at_1 >= target_ess:
        return 1.0

    # Intermediate value root: f(phi_prev) = (1 - target_ess_ratio) * ess_prev > 0,
    # and f(1.0) < 0.
    def obj(p: float) -> float:
        return ess_at_phi(p) - target_ess

    low = min(phi_prev + min_step, 1.0)
    high = 1.0

    # Fast 1D bisection
    for _ in range(50):
        mid = 0.5 * (low + high)
        val = obj(mid)
        if val > 0.0:
            low = mid
        else:
            high = mid
        if (high - low) < 1e-6:
            break

    phi_cand = 0.5 * (low + high)
    return float(np.clip(phi_cand, phi_prev + min_step, 1.0))


# ---------------------------------------------------------------------------
# Prior Sampling and Helper Routines
# ---------------------------------------------------------------------------

def _draw_prior_particles(
    priors: Mapping[str, Any],
    n_particles: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Draw initial particles from declared prior distributions."""
    names = tuple(priors.keys())
    d = len(names)
    particles = np.empty((n_particles, d), dtype=float)

    for j, name in enumerate(names):
        spec = priors[name]
        dist = spec.get("dist") if isinstance(spec, dict) else getattr(spec, "dist", "normal")
        mean = float(spec.get("mean") if isinstance(spec, dict) else getattr(spec, "mean", 0.0))
        std = float(spec.get("std") if isinstance(spec, dict) else getattr(spec, "std", 1.0))
        lb = float(spec.get("lb", -math.inf) if isinstance(spec, dict) else getattr(spec, "lb", -math.inf))
        ub = float(spec.get("ub", math.inf) if isinstance(spec, dict) else getattr(spec, "ub", math.inf))

        if dist == "uniform":
            particles[:, j] = rng.uniform(lb, ub, size=n_particles)
        elif dist == "beta":
            shift = float(spec.get("shift", 0.0) if isinstance(spec, dict) else getattr(spec, "_extra", {}).get("shift", 0.0))
            scale = float(spec.get("scale", 1.0) if isinstance(spec, dict) else getattr(spec, "_extra", {}).get("scale", 1.0))
            m0 = (mean - shift) / scale
            s0 = std / scale
            m0 = np.clip(m0, 1e-5, 1.0 - 1e-5)
            max_s2 = m0 * (1.0 - m0)
            s0 = min(s0, math.sqrt(max_s2) * 0.95)
            s2 = s0 ** 2
            a = m0 * (m0 * (1.0 - m0) / s2 - 1.0)
            b = (1.0 - m0) * (m0 * (1.0 - m0) / s2 - 1.0)
            a, b = max(a, 1e-4), max(b, 1e-4)
            particles[:, j] = shift + scale * rng.beta(a, b, size=n_particles)
        elif dist == "gamma":
            shift = float(spec.get("shift", 0.0) if isinstance(spec, dict) else getattr(spec, "_extra", {}).get("shift", 0.0))
            eff_mean = max(mean - shift, 1e-5)
            k = (eff_mean / std) ** 2
            theta = (std ** 2) / eff_mean
            particles[:, j] = shift + rng.gamma(k, theta, size=n_particles)
        elif dist == "invgamma":
            s_val = float(spec.get("s", 0.1) if isinstance(spec, dict) else getattr(spec, "_extra", {}).get("s", 0.1))
            nu_val = float(spec.get("nu", 2.0) if isinstance(spec, dict) else getattr(spec, "_extra", {}).get("nu", 2.0))
            kind = spec.get("kind", "type1") if isinstance(spec, dict) else getattr(spec, "_extra", {}).get("kind", "type1")
            gamma_draws = rng.gamma(nu_val / 2.0, 2.0 / max(s_val, 1e-8), size=n_particles)
            v = 1.0 / np.maximum(gamma_draws, 1e-12)
            particles[:, j] = np.sqrt(v) if kind == "type1" else v
        else:
            # Normal default with truncation check
            draws = rng.normal(mean, std, size=n_particles)
            if math.isfinite(lb) or math.isfinite(ub):
                mask = (draws < lb) | (draws > ub)
                for _ in range(50):
                    if not np.any(mask):
                        break
                    draws[mask] = rng.normal(mean, std, size=np.sum(mask))
                    mask = (draws < lb) | (draws > ub)
                if np.any(mask):
                    draws[mask] = np.clip(draws[mask], lb if math.isfinite(lb) else mean - 3 * std,
                                          ub if math.isfinite(ub) else mean + 3 * std)
            particles[:, j] = draws

    return particles, names


def _regularize_covariance(cov: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Regularize empirical particle covariance via eigenvalue flooring and adaptive ridge."""
    A = 0.5 * (cov + cov.T)
    d = A.shape[0]
    try:
        w, V = np.linalg.eigh(A)
        w_max = float(np.max(w)) if len(w) > 0 else 1.0
        floor = max(ridge, ridge * w_max, 1e-10)
        w_floored = np.maximum(w, floor)
        out = V @ np.diag(w_floored) @ V.T
        out = 0.5 * (out + out.T)
        out += max(ridge, 1e-9) * np.eye(d)
        return out
    except np.linalg.LinAlgError:
        return np.eye(d) * max(ridge, 1e-4)


# ---------------------------------------------------------------------------
# SMCResult Presentation Contract
# ---------------------------------------------------------------------------

@dataclass
class SMCResult:
    """Sequential Monte Carlo (SMC) estimation result object.

    Attributes
    ----------
    particles : ndarray of shape (n_particles, n_params)
        Final mutated particle matrix approximating the target posterior.
    weights : ndarray of shape (n_particles,)
        Final normalized particle importance weights.
    stage_tempering : ndarray of shape (n_stages,)
        Tempering schedule phi_n from stage 0 to stage N_phi.
    mdd : float
        Exact Marginal Data Density (MDD) ln p(Y) computed from stage normalizing constants.
    mdd_se : float
        Numerical standard error of the MDD estimate.
    acceptance_rates : ndarray of shape (n_stages,)
        Stage-by-stage particle mutation acceptance rates.
    ess_history : ndarray of shape (n_stages,)
        Stage-by-stage Effective Sample Size (ESS) trajectory.
    posterior_summary : pd.DataFrame
        Summary statistics (mean, std, 5%, 50%, 95%) for each parameter.
    param_names : tuple of str
        Names of the estimated parameters.
    stage_diagnostics : pd.DataFrame, optional
        Detailed diagnostics per stage: stage, phi, ess, accept_rate, scale_c, resampled, log_z.
    log_marginal_likelihood : float
        Alias for mdd.
    log_marginal_likelihood_se : float
        Alias for mdd_se.
    priors : dict, optional
        Priors used for estimation.
    data_n_obs : int
        Number of sample observations T.
    seed : int, optional
        Random seed used for reproducibility.
    model_name : str
        Name or descriptor of the estimated model.
    """

    particles: np.ndarray
    weights: np.ndarray
    stage_tempering: np.ndarray
    mdd: float
    mdd_se: float
    acceptance_rates: np.ndarray
    ess_history: np.ndarray
    posterior_summary: pd.DataFrame
    param_names: tuple[str, ...] = ()
    stage_diagnostics: pd.DataFrame | None = None
    log_marginal_likelihood: float = 0.0
    log_marginal_likelihood_se: float = 0.0
    priors: dict | None = None
    data_n_obs: int = 0
    seed: int | None = None
    model_name: str = "unknown"

    def __post_init__(self) -> None:
        if self.log_marginal_likelihood == 0.0 and self.mdd != 0.0:
            self.log_marginal_likelihood = float(self.mdd)
        if self.log_marginal_likelihood_se == 0.0 and self.mdd_se != 0.0:
            self.log_marginal_likelihood_se = float(self.mdd_se)

        if not self.param_names and hasattr(self.posterior_summary, "index"):
            self.param_names = tuple(str(idx) for idx in self.posterior_summary.index)

        if self.stage_diagnostics is None:
            n_stg = len(self.stage_tempering)
            diag_dict = {
                "stage": np.arange(n_stg),
                "phi": np.asarray(self.stage_tempering, dtype=float),
                "ess": np.asarray(self.ess_history[:n_stg], dtype=float)
                if len(self.ess_history) >= n_stg
                else np.pad(self.ess_history, (0, max(0, n_stg - len(self.ess_history))), constant_values=np.nan),
                "accept_rate": np.asarray(self.acceptance_rates[:n_stg], dtype=float)
                if len(self.acceptance_rates) >= n_stg
                else np.pad(self.acceptance_rates, (0, max(0, n_stg - len(self.acceptance_rates))), constant_values=np.nan),
            }
            self.stage_diagnostics = pd.DataFrame(diag_dict)

    def summary(self) -> str:
        """Formatted summary table of SMC posterior estimates and MDD."""
        lines = [
            "=" * 68,
            " Sequential Monte Carlo (SMC) Estimation Results",
            "=" * 68,
            f"Log Marginal Data Density (MDD): {self.mdd:12.4f}",
            f"MDD Numerical Standard Error:    {self.mdd_se:12.4f}",
            f"Number of Stages:               {len(self.stage_tempering):12d}",
            f"Number of Particles:            {len(self.particles):12d}",
            f"Terminal ESS:                   {self.ess_history[-1] if len(self.ess_history) > 0 else np.nan:12.2f}",
            "-" * 68,
            "Posterior Parameter Estimates:",
            self.posterior_summary.to_string(),
            "=" * 68,
        ]
        return "\n".join(lines)

    def to_markdown(self, **kwargs: Any) -> str:
        """Export posterior parameter table and MDD to Markdown format."""
        lines = [
            "### Sequential Monte Carlo (SMC) Estimation Results",
            "",
            f"- **Log Marginal Data Density (MDD)**: `{self.mdd:.4f}` (SE: `{self.mdd_se:.4f}`)",
            f"- **Number of Particles**: `{len(self.particles)}`",
            f"- **Number of Stages**: `{len(self.stage_tempering)}`",
            "",
            self.posterior_summary.to_markdown(),
        ]
        return "\n".join(lines)

    def to_latex(self, **kwargs: Any) -> str:
        """Export posterior parameter table and MDD to LaTeX format."""
        caption = f"SMC Posterior Summary (Log MDD: {self.mdd:.4f} $\\pm$ {self.mdd_se:.4f})"
        return self.posterior_summary.to_latex(caption=caption, **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Export posterior parameter table to Typst document format."""
        cols = len(self.posterior_summary.columns) + 1
        lines = [
            "// Sequential Monte Carlo (SMC) Posterior Summary",
            f"// Log MDD: {self.mdd:.4f} (SE: {self.mdd_se:.4f})",
            f"#table(",
            f"  columns: {cols},",
            "  [Parameter], " + ", ".join(f"[{col}]" for col in self.posterior_summary.columns) + ",",
        ]
        for param, row in self.posterior_summary.iterrows():
            row_items = []
            for val in row:
                if isinstance(val, (int, float, np.floating)):
                    row_items.append(f"[{val:.4f}]")
                else:
                    row_items.append(f"[{val}]")
            lines.append(f"  [{param}], " + ", ".join(row_items) + ",")
        lines.append(")")
        return "\n".join(lines)

    def plot_stages(self, fig: Any = None, axes: Any = None) -> tuple[Any, Any]:
        """Plot stage diagnostics: tempering schedule, ESS, accept rate, and scale."""
        import matplotlib.pyplot as plt

        if axes is None:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

        stages = np.arange(len(self.stage_tempering))

        # 1. Tempering schedule
        ax_flat[0].plot(stages, self.stage_tempering, marker="o", color="#1f77b4", lw=1.8, ms=4)
        ax_flat[0].set_title("Tempering Schedule $\\phi_n$", fontsize=11, fontweight="bold")
        ax_flat[0].set_xlabel("Stage $n$")
        ax_flat[0].set_ylabel("$\\phi$")
        ax_flat[0].grid(True, alpha=0.3)

        # 2. ESS trajectory
        if len(self.ess_history) > 0:
            ess_len = min(len(stages), len(self.ess_history))
            ax_flat[1].plot(stages[:ess_len], self.ess_history[:ess_len], marker="s", color="#2ca02c", lw=1.8, ms=4)
            ax_flat[1].axhline(0.5 * len(self.particles), linestyle="--", color="gray", alpha=0.7, label="Resample Threshold")
            ax_flat[1].legend(loc="lower right")
        ax_flat[1].set_title("Effective Sample Size ($ESS_n$)", fontsize=11, fontweight="bold")
        ax_flat[1].set_xlabel("Stage $n$")
        ax_flat[1].set_ylabel("ESS")
        ax_flat[1].grid(True, alpha=0.3)

        # 3. Acceptance rate
        if len(self.acceptance_rates) > 0:
            ar_len = min(len(stages), len(self.acceptance_rates))
            ax_flat[2].plot(stages[:ar_len], self.acceptance_rates[:ar_len], marker="^", color="#d62728", lw=1.8, ms=4)
            ax_flat[2].axhline(0.25, linestyle="--", color="black", alpha=0.6, label="Target (25%)")
            ax_flat[2].legend(loc="upper right")
        ax_flat[2].set_title("Metropolis Acceptance Rate $\\hat{\\alpha}_n$", fontsize=11, fontweight="bold")
        ax_flat[2].set_xlabel("Stage $n$")
        ax_flat[2].set_ylabel("Acceptance Rate")
        ax_flat[2].grid(True, alpha=0.3)

        # 4. Proposal scale or stage diagnostics
        if self.stage_diagnostics is not None and "scale_c" in self.stage_diagnostics.columns:
            sc = self.stage_diagnostics["scale_c"].to_numpy()
            ax_flat[3].plot(stages[:len(sc)], sc[:len(stages)], marker="d", color="#9467bd", lw=1.8, ms=4)
            ax_flat[3].set_title("Adapted Proposal Scale $c_n$", fontsize=11, fontweight="bold")
            ax_flat[3].set_ylabel("$c_n$")
        else:
            ax_flat[3].plot(stages, self.stage_tempering, alpha=0.0)
            ax_flat[3].set_title("Stage Progress", fontsize=11, fontweight="bold")
        ax_flat[3].set_xlabel("Stage $n$")
        ax_flat[3].grid(True, alpha=0.3)

        if fig is not None:
            fig.tight_layout()
        return fig, axes

    def plot_posterior(self, fig: Any = None, axes: Any = None, bins: int = 25) -> tuple[Any, Any]:
        """Plot posterior parameter marginal distributions."""
        import matplotlib.pyplot as plt

        n_params = self.particles.shape[1]
        if axes is None:
            ncols = min(3, n_params)
            nrows = int(np.ceil(n_params / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows))

        ax_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
        names = self.param_names if len(self.param_names) == n_params else tuple(f"param_{i}" for i in range(n_params))

        for j in range(n_params):
            if j < len(ax_flat):
                ax = ax_flat[j]
                vals = self.particles[:, j]
                ax.hist(
                    vals,
                    weights=self.weights,
                    bins=bins,
                    density=True,
                    alpha=0.65,
                    color="#1f77b4",
                    edgecolor="white",
                )
                ax.set_title(names[j], fontsize=11, fontweight="bold")
                ax.set_xlabel("Value")
                ax.set_ylabel("Posterior Density")
                ax.grid(True, alpha=0.3)

        for k in range(n_params, len(ax_flat)):
            ax_flat[k].axis("off")

        if fig is not None:
            fig.tight_layout()
        return fig, axes

    def plot(self, **kwargs: Any) -> tuple[Any, Any]:
        """Alias for plot_posterior."""
        return self.plot_posterior(**kwargs)

    def marginal_likelihood(self) -> tuple[float, float]:
        """Return tuple of (log_mdd, mdd_standard_error)."""
        return float(self.mdd), float(self.mdd_se)

    def posterior_table(self) -> pd.DataFrame:
        """Return posterior parameter summary DataFrame."""
        return self.posterior_summary


# ---------------------------------------------------------------------------
# SMC Estimation Engine: smc_estimate
# ---------------------------------------------------------------------------

def smc_estimate(
    log_lik: Callable[[dict[str, float] | np.ndarray], float],
    priors: Mapping[str, Any],
    *,
    n_particles: int = 500,
    n_stages: int = 50,
    adaptive_tempering: bool = True,
    target_ess: float = 0.5,
    lambda_tempering: float = 2.1,
    n_steps: int = 1,
    c_init: float = 0.5,
    seed: int | None = None,
    ridge: float = 1e-6,
    model_name: str = "smc_model",
    data_n_obs: int = 0,
) -> SMCResult:
    """Estimate posterior distributions and Marginal Data Density via Sequential Monte Carlo.

    Parameters
    ----------
    log_lik : callable
        Log-likelihood function accepting parameter dict or array.
    priors : dict
        Priors dictionary {param_name: Prior | dict}.
    n_particles : int, default 500
        Number of particles N_part. Must be > 1.
    n_stages : int, default 50
        Maximum number of intermediate tempering stages N_phi.
    adaptive_tempering : bool, default True
        If True, solves for phi_n adaptively targeting ESS decay.
        If False, uses fixed schedule phi_n = (n / N_phi)^lambda.
    target_ess : float, default 0.5
        Resampling trigger ratio: systematic resampling occurs when ESS_n < target_ess * N_part.
    lambda_tempering : float, default 2.1
        Curvature for fixed schedule.
    n_steps : int, default 1
        Number of Metropolis-Hastings mutation steps per particle per stage.
    c_init : float, default 0.5
        Initial proposal covariance scale.
    seed : int, optional
        Random seed.
    ridge : float, default 1e-6
        Ridge floor added to empirical covariance during mutation.
    model_name : str, default "smc_model"
        Identifier for model.
    data_n_obs : int, default 0
        Number of observations in data.

    Returns
    -------
    SMCResult
        Results object containing particles, weights, MDD estimate, and diagnostics.
    """
    if n_particles <= 1:
        raise ValueError(f"n_particles must be > 1, got {n_particles}")
    if n_stages < 1:
        raise ValueError(f"n_stages must be >= 1, got {n_stages}")

    rng = np.random.default_rng(seed)

    # 1. Initialize particle cloud from priors
    particles, param_names = _draw_prior_particles(priors, n_particles, rng)
    n_params = len(param_names)

    # Check if log_lik accepts array or dict
    def eval_ll(theta_vec: np.ndarray) -> float:
        try:
            par_dict = {name: float(theta_vec[j]) for j, name in enumerate(param_names)}
            try:
                res = log_lik(par_dict)
            except (TypeError, ValueError):
                res = log_lik(theta_vec)
            val = float(res)
            return val if np.isfinite(val) else -np.inf
        except Exception:
            return -np.inf

    def eval_lp(theta_vec: np.ndarray) -> float:
        par_dict = {name: float(theta_vec[j]) for j, name in enumerate(param_names)}
        lp = float(log_prior(par_dict, priors))
        return lp if np.isfinite(lp) else -np.inf

    ll_particles = np.empty(n_particles, dtype=float)
    lp_particles = np.empty(n_particles, dtype=float)

    for i in range(n_particles):
        lp_particles[i] = eval_lp(particles[i])
        ll_particles[i] = eval_ll(particles[i]) if np.isfinite(lp_particles[i]) else -np.inf

    # Weights initialization at stage 0 (phi_0 = 0)
    weights = np.full(n_particles, 1.0 / n_particles, dtype=float)

    # Build schedule or prepare adaptive schedule
    if not adaptive_tempering:
        fixed_schedule = fixed_tempering_schedule(n_stages, lambda_param=lambda_tempering)

    phi_curr = 0.0
    c_scale = float(c_init)

    tempering_history = [0.0]
    ess_history = [float(n_particles)]
    accept_rates = []
    log_z_history = []
    scale_history = [c_scale]
    resampled_history = [False]

    # Target ESS decay per stage under adaptive schedule
    adaptive_ratio = max(0.80, min(0.98, (0.5) ** (1.0 / max(n_stages, 4))))

    # 2. Sequential tempering recursion
    for stage_idx in range(1, n_stages + 1):
        if not adaptive_tempering:
            phi_next = float(fixed_schedule[stage_idx])
        else:
            if stage_idx == n_stages:
                phi_next = 1.0
            else:
                phi_next = solve_adaptive_phi(
                    phi_curr,
                    ll_particles,
                    weights,
                    target_ess_ratio=adaptive_ratio,
                    min_step=max(1e-4, (1.0 - phi_curr) / (n_stages - stage_idx + 1)),
                )

        delta_phi = phi_next - phi_curr
        if delta_phi <= 0.0:
            phi_next = min(1.0, phi_curr + 1e-4)
            delta_phi = phi_next - phi_curr

        # a. Correction (Reweighting)
        v = delta_phi * ll_particles
        v_max = float(np.max(v))
        if not np.isfinite(v_max) or v_max < -1e20:
            v_max = 0.0
        w_tilde = np.exp(np.clip(v - v_max, -700.0, 50.0))
        weighted_increments = weights * w_tilde
        sum_incr = float(np.sum(weighted_increments))

        if sum_incr <= 0.0 or not np.isfinite(sum_incr):
            log_z_n = v_max - 50.0
            tilde_W = np.full(n_particles, 1.0 / n_particles)
            v_n = 0.0
        else:
            log_z_n = v_max + float(np.log(sum_incr))
            tilde_W = weighted_increments / sum_incr
            # Variance for numerical standard error
            mean_w = sum_incr
            var_w = float(np.sum(weights * ((w_tilde - mean_w) ** 2)))
            v_n = var_w / (n_particles * (mean_w ** 2 + 1e-30))

        log_z_history.append((log_z_n, v_n))

        # b. Empirical Moments for Proposal Covariance (Herbst & Schorfheide 2014)
        bar_theta = np.sum(tilde_W[:, None] * particles, axis=0)
        diff = particles - bar_theta[None, :]
        hat_Sigma = (diff.T * tilde_W) @ diff

        # Regularize proposal covariance
        hat_Sigma_reg = _regularize_covariance(hat_Sigma, ridge=ridge)
        Sigma_prop = (c_scale ** 2) * hat_Sigma_reg

        try:
            L_prop = scipy.linalg.cholesky(Sigma_prop, lower=True)
        except (scipy.linalg.LinAlgError, np.linalg.LinAlgError):
            L_prop = np.diag(np.sqrt(np.maximum(np.diag(Sigma_prop), 1e-8)))

        # c. Effective Sample Size & Selection (Systematic Resampling)
        ess_n = float(1.0 / np.sum(tilde_W ** 2))
        resample_threshold = target_ess * n_particles

        if ess_n < resample_threshold:
            indices = systematic_resample(tilde_W, rng)
            particles = particles[indices].copy()
            ll_particles = ll_particles[indices].copy()
            lp_particles = lp_particles[indices].copy()
            weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
            resampled_flag = True
        else:
            weights = tilde_W.copy()
            resampled_flag = False

        # d. Particle Mutation (Random-Walk Metropolis-Hastings)
        accept_count = 0
        total_proposals = n_particles * n_steps

        for i in range(n_particles):
            curr_theta = particles[i].copy()
            curr_ll = ll_particles[i]
            curr_lp = lp_particles[i]
            curr_post = phi_next * curr_ll + curr_lp

            for _ in range(n_steps):
                z = rng.normal(0.0, 1.0, size=n_params)
                prop_theta = curr_theta + L_prop @ z
                prop_lp = eval_lp(prop_theta)

                if np.isfinite(prop_lp):
                    prop_ll = eval_ll(prop_theta)
                    if np.isfinite(prop_ll):
                        prop_post = phi_next * prop_ll + prop_lp
                        log_alpha = prop_post - curr_post
                        if math.log(max(rng.uniform(0.0, 1.0), 1e-12)) < log_alpha:
                            curr_theta = prop_theta
                            curr_ll = prop_ll
                            curr_lp = prop_lp
                            curr_post = prop_post
                            accept_count += 1

            particles[i] = curr_theta
            ll_particles[i] = curr_ll
            lp_particles[i] = curr_lp

        hat_alpha = accept_count / float(total_proposals)
        accept_rates.append(hat_alpha)

        # d. Scale adaptation targeting ~25% acceptance
        scale_factor = 0.95 + 0.10 * (math.exp(16.0 * (hat_alpha - 0.25)) / (1.0 + math.exp(16.0 * (hat_alpha - 0.25))))
        c_scale = float(np.clip(c_scale * scale_factor, 1e-4, 10.0))
        scale_history.append(c_scale)

        # e. Record stage state
        phi_curr = phi_next
        tempering_history.append(phi_curr)
        ess_history.append(float(1.0 / np.sum(weights ** 2)))
        resampled_history.append(resampled_flag)

        if phi_curr >= 1.0 - 1e-7:
            break

    # Ensure final phi reached 1.0
    if tempering_history[-1] < 1.0:
        tempering_history[-1] = 1.0

    # 3. Compute Exact Marginal Data Density (MDD) and SE
    mdd_val = float(sum(item[0] for item in log_z_history))
    mdd_var = float(sum(item[1] for item in log_z_history))
    mdd_se_val = float(math.sqrt(max(mdd_var, 0.0)))

    # 4. Compute Weighted Posterior Summary
    w_final = weights / np.sum(weights)
    means = np.sum(particles * w_final[:, None], axis=0)
    stds = np.sqrt(np.maximum(0.0, np.sum(w_final[:, None] * ((particles - means[None, :]) ** 2), axis=0)))

    q05 = np.empty(n_params, dtype=float)
    q50 = np.empty(n_params, dtype=float)
    q95 = np.empty(n_params, dtype=float)

    for j in range(n_params):
        sort_idx = np.argsort(particles[:, j])
        sorted_vals = particles[sort_idx, j]
        sorted_w = w_final[sort_idx]
        cum_w = np.cumsum(sorted_w)
        cum_w[-1] = 1.0
        q05[j] = float(np.interp(0.05, cum_w, sorted_vals))
        q50[j] = float(np.interp(0.50, cum_w, sorted_vals))
        q95[j] = float(np.interp(0.95, cum_w, sorted_vals))

    posterior_df = pd.DataFrame(
        {
            "mean": means,
            "std": stds,
            "5%": q05,
            "50%": q50,
            "95%": q95,
        },
        index=list(param_names),
    )

    stage_diag_df = pd.DataFrame(
        {
            "stage": np.arange(len(tempering_history)),
            "phi": np.array(tempering_history),
            "ess": np.array(ess_history),
            "accept_rate": np.array([0.0] + accept_rates),
            "scale_c": np.array(scale_history),
            "resampled": np.array(resampled_history),
        }
    )

    return SMCResult(
        particles=particles,
        weights=weights,
        stage_tempering=np.array(tempering_history),
        mdd=mdd_val,
        mdd_se=mdd_se_val,
        acceptance_rates=np.array(accept_rates),
        ess_history=np.array(ess_history),
        posterior_summary=posterior_df,
        param_names=param_names,
        stage_diagnostics=stage_diag_df,
        log_marginal_likelihood=mdd_val,
        log_marginal_likelihood_se=mdd_se_val,
        priors=dict(priors),
        data_n_obs=data_n_obs,
        seed=seed,
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# SMCSampler Class Interface
# ---------------------------------------------------------------------------

class SMCSampler:
    """Sequential Monte Carlo Sampler for DSGE models and likelihood targets.

    Parameters
    ----------
    m : LinearModel or Any, optional
        Solved DSGE model (e.g. from build_dynare).
    data : pd.DataFrame or ndarray, optional
        Observed macroeconomic dataset.
    varobs : Sequence of str, optional
        Names of observable variables matching columns in data.
    n_particles : int, default 1000
        Number of particles N_part. Must be > 1.
    n_stages : int, default 100
        Maximum number of intermediate tempering stages N_phi.
    adaptive_tempering : bool, default True
        If True, solves for tempering increments adaptively based on ESS drop.
    target_ess : float, default 0.5
        Resampling trigger fraction of N_part.
    seed : int, optional
        Random seed for reproducibility.
    priors : dict, optional
        Priors dictionary. If None, extracted from model or calibrated defaults.
    log_lik : callable, optional
        Direct likelihood callable if bypassing model Kalman recursion.
    ridge : float, default 1e-4
        Measurement / Kalman ridge regularization.
    measurement_error : Mapping[str, float] or ndarray, optional
        Measurement error standard deviations.
    """

    def __init__(
        self,
        m: Any = None,
        data: pd.DataFrame | np.ndarray | None = None,
        varobs: Sequence[str] | None = None,
        n_particles: int = 1000,
        n_stages: int = 100,
        adaptive_tempering: bool = True,
        target_ess: float = 0.5,
        seed: int | None = None,
        priors: Mapping[str, Any] | None = None,
        log_lik: Callable[..., float] | None = None,
        ridge: float = 1e-4,
        measurement_error: Mapping[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if n_particles <= 1:
            raise ValueError(f"SMCSampler: n_particles must be > 1, got {n_particles}")

        self.m = m
        self.data = data
        self.varobs = list(varobs) if varobs is not None else None
        self.n_particles = int(n_particles)
        self.n_stages = int(n_stages)
        self.adaptive_tempering = bool(adaptive_tempering)
        self.target_ess = float(target_ess)
        self.seed = seed
        self.ridge = float(ridge)
        self.measurement_error = measurement_error
        self.kwargs = kwargs

        # Resolve priors and log_likelihood callable
        self._priors: dict[str, Any]
        self._log_lik: Callable[[dict[str, float] | np.ndarray], float]

        if log_lik is not None and priors is not None:
            self._log_lik = log_lik
            self._priors = dict(priors)
        elif m is not None and data is not None and self.varobs is not None:
            self._setup_dsge_target(priors)
        elif priors is not None and log_lik is not None:
            self._priors = dict(priors)
            self._log_lik = log_lik
        else:
            raise ValueError(
                "SMCSampler: provide either (m, data, varobs) or (log_lik, priors)."
            )

    def _setup_dsge_target(self, custom_priors: Mapping[str, Any] | None) -> None:
        """Configure DSGE Kalman filter likelihood and priors."""
        from puremacro.dsge.dynare import build_dynare
        from puremacro.dsge.observation import make_state_space_from_varobs
        from puremacro.state_space import kalman_filter
        from puremacro.dsge.estimate import _stationary_init

        m = self.m
        data = self.data
        varobs = self.varobs

        if isinstance(data, pd.DataFrame):
            y_arr = data[varobs].to_numpy(dtype=float)
            data_n_obs = len(data)
        else:
            y_arr = np.asarray(data, dtype=float)
            data_n_obs = len(y_arr)

        self.data_n_obs = data_n_obs

        # Resolve priors
        if custom_priors is not None:
            self._priors = dict(custom_priors)
        elif getattr(m, "_estimated_params", None) is not None:
            self._priors = dict(m._estimated_params.priors())
        else:
            # Default sensible priors for monetary policy / structural parameters
            base_params = dict(getattr(m, "_params", {}) or {})
            default_p = {}
            if "phi_pi" in base_params:
                default_p["phi_pi"] = NormalPrior(mean=float(base_params["phi_pi"]), std=0.25, lb=1.0, ub=3.5)
            if "phi_y" in base_params:
                default_p["phi_y"] = NormalPrior(mean=float(base_params["phi_y"]), std=0.15, lb=0.0, ub=1.5)
            if "rho_r" in base_params:
                default_p["rho_r"] = BetaPrior(mean=float(base_params["rho_r"]), std=0.10, lb=0.0, ub=0.99)
            elif "rho" in base_params:
                default_p["rho"] = BetaPrior(mean=float(base_params["rho"]), std=0.10, lb=0.0, ub=0.99)

            if not default_p:
                for k, v in list(base_params.items())[:3]:
                    val = float(v)
                    if 0.0 < val < 1.0:
                        default_p[k] = BetaPrior(mean=val, std=min(val, 1.0 - val) * 0.25, lb=0.0, ub=1.0)
                    else:
                        default_p[k] = NormalPrior(mean=val, std=max(abs(val) * 0.25, 0.2), lb=-math.inf, ub=math.inf)
            self._priors = default_p

        # Pre-resolve structural parameters vs shocks
        base_params = dict(getattr(m, "_params", {}) or {})
        param_names = list(self._priors.keys())

        # Ridge for measurement covariance if shocks < varobs
        n_shocks = len(getattr(m, "shocks", ()))
        n_obs = len(varobs)
        eff_ridge = max(self.ridge, 1e-3 if n_shocks < n_obs else 1e-5)

        def dsge_log_lik(par_dict_or_vec: dict[str, float] | np.ndarray) -> float:
            if isinstance(par_dict_or_vec, dict):
                p_dict = par_dict_or_vec
            else:
                p_dict = {param_names[j]: float(par_dict_or_vec[j]) for j in range(len(param_names))}

            curr_params = dict(base_params)
            curr_params.update(p_dict)

            try:
                if getattr(m, "_dynare_equations", None) is not None:
                    solved_m = build_dynare(
                        m._dynare_equations,
                        variables=m.variables,
                        shocks=m.shocks,
                        params=curr_params,
                        steady_state=m.steady_state,
                        check_steady_state=False,
                        strict=False,
                    )
                else:
                    solved_m = m

                ssm = make_state_space_from_varobs(
                    solved_m,
                    varobs,
                    measurement_error=self.measurement_error,
                    ridge=eff_ridge,
                )
                a0, P0 = _stationary_init(ssm)
                if P0 is None:
                    P0 = 1e6 * np.eye(ssm.m)
                kf = kalman_filter(y_arr, ssm, a0=a0, P0=P0)
                ll = float(kf["loglik"])
                return ll if np.isfinite(ll) else -np.inf
            except Exception:
                return -np.inf

        self._log_lik = dsge_log_lik

    def sample(self) -> SMCResult:
        """Run Sequential Monte Carlo estimation and return SMCResult."""
        return smc_estimate(
            log_lik=self._log_lik,
            priors=self._priors,
            n_particles=self.n_particles,
            n_stages=self.n_stages,
            adaptive_tempering=self.adaptive_tempering,
            target_ess=self.target_ess,
            seed=self.seed,
            ridge=self.ridge,
            data_n_obs=getattr(self, "data_n_obs", 0),
            **self.kwargs,
        )


# ---------------------------------------------------------------------------
# Bootstrap Particle Filter for Nonlinear State Spaces
# ---------------------------------------------------------------------------

def bootstrap_particle_filter(
    pruned_sol: Any,
    data: pd.DataFrame | np.ndarray,
    varobs: Sequence[str],
    n_particles: int = 500,
    seed: int | None = None,
    measurement_error: Mapping[str, float] | np.ndarray | None = None,
    observation_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ridge: float = 1e-4,
) -> tuple[float, np.ndarray]:
    """Evaluate log-likelihood of pruned order-2 or order-3 DSGE solutions via bootstrap particle filtering.

    Supports non-Gaussian observation equations, measurement errors, and state pruning
    (Andreasen et al. 2018), with low-weight rejuvenation under severe outliers.

    Parameters
    ----------
    pruned_sol : PrunedDSGESolution or Order3PrunedSolution
        Solved pruned state-space solution.
    data : pd.DataFrame or ndarray
        Observed time series dataset of shape (T, k).
    varobs : Sequence of str
        Names of the observable series in data.
    n_particles : int, default 500
        Number of state particles M_p.
    seed : int, optional
        Random seed for reproducibility.
    measurement_error : Mapping[str, float] or ndarray, optional
        Measurement error standard deviations.
    observation_fn : callable, optional
        Custom observation function y_t = h(x_t). If None, linear selection.
    ridge : float, default 1e-4
        Small ridge on measurement covariance to prevent singularity.

    Returns
    -------
    log_likelihood : float
        Total log-likelihood scalar ln L(Y | theta).
    filtered_states : ndarray of shape (T, n_vars)
        Filtered mean state trajectories over time.
    """
    if n_particles < 1:
        raise ValueError(f"n_particles must be >= 1, got {n_particles}")

    rng = np.random.default_rng(seed)

    # 1. Parse observed data array
    obs_names = list(varobs)
    k = len(obs_names)

    if isinstance(data, pd.DataFrame):
        y_mat = data[obs_names].to_numpy(dtype=float)
    else:
        y_mat = np.asarray(data, dtype=float)
        if y_mat.shape[1] != k:
            raise ValueError(f"data columns ({y_mat.shape[1]}) must match varobs length ({k})")

    T = len(y_mat)

    # 2. Extract solution matrices and dimensions
    s_names = tuple(getattr(pruned_sol, "state_names", ()))
    c_names = tuple(getattr(pruned_sol, "control_names", ()))
    e_names = tuple(getattr(pruned_sol, "shock_names", ()))
    n_x = len(s_names)
    n_y = len(c_names)
    n_e = len(e_names)
    n_vars = n_x + n_y

    all_vars = tuple(getattr(pruned_sol, "variable_names", None) or (s_names + c_names))

    # Find observable indices in all_vars
    obs_indices = []
    for nm in obs_names:
        if nm in all_vars:
            obs_indices.append(all_vars.index(nm))
        else:
            # Fallback index if names not found
            obs_indices.append(min(len(obs_indices), n_vars - 1))
    obs_indices = np.array(obs_indices, dtype=int)

    # Steady state
    ss = getattr(pruned_sol, "steady_state", None)
    if isinstance(ss, pd.Series):
        ss_vec = ss.to_numpy(dtype=float)
    elif isinstance(ss, (dict, Mapping)):
        ss_vec = np.array([float(ss.get(v, 0.0)) for v in all_vars])
    elif ss is not None:
        ss_vec = np.asarray(ss, dtype=float)
    else:
        ss_vec = np.zeros(n_vars)

    if len(ss_vec) != n_vars:
        ss_vec = np.pad(ss_vec, (0, max(0, n_vars - len(ss_vec))))[:n_vars]

    # Matrices
    G = np.asarray(getattr(pruned_sol, "G", np.zeros((n_x, n_x))), dtype=float)
    N = np.asarray(getattr(pruned_sol, "N", np.zeros((n_x, n_e))), dtype=float)
    F = np.asarray(getattr(pruned_sol, "F", np.zeros((n_y, n_x))), dtype=float)
    L = np.asarray(getattr(pruned_sol, "L", np.zeros((n_y, n_e))), dtype=float)

    H_xx = np.asarray(getattr(pruned_sol, "H_xx", np.zeros((n_x, n_x * n_x))), dtype=float)
    G_xx = np.asarray(getattr(pruned_sol, "G_xx", np.zeros((n_y, n_x * n_x))), dtype=float)

    H_xu = np.asarray(getattr(pruned_sol, "H_xu", np.zeros((n_x, n_x * n_e))), dtype=float)
    G_xu = np.asarray(getattr(pruned_sol, "G_xu", np.zeros((n_y, n_x * n_e))), dtype=float)

    H_uu = np.asarray(getattr(pruned_sol, "H_uu", np.zeros((n_x, n_e * n_e))), dtype=float)
    G_uu = np.asarray(getattr(pruned_sol, "G_uu", np.zeros((n_y, n_e * n_e))), dtype=float)

    H_ss = np.asarray(getattr(pruned_sol, "H_sigmasigma", np.zeros(n_x)), dtype=float).ravel()
    G_ss = np.asarray(getattr(pruned_sol, "G_sigmasigma", np.zeros(n_y)), dtype=float).ravel()

    is_order3 = hasattr(pruned_sol, "H_xxx") and pruned_sol.H_xxx is not None
    if is_order3:
        H_xxx = np.asarray(pruned_sol.H_xxx, dtype=float)
        G_xxx = np.asarray(pruned_sol.G_xxx, dtype=float)
        H_xxu = np.asarray(pruned_sol.H_xxu, dtype=float)
        G_xxu = np.asarray(pruned_sol.G_xxu, dtype=float)
        H_xuu = np.asarray(pruned_sol.H_xuu, dtype=float)
        G_xuu = np.asarray(pruned_sol.G_xuu, dtype=float)
        H_uuu = np.asarray(pruned_sol.H_uuu, dtype=float)
        G_uuu = np.asarray(pruned_sol.G_uuu, dtype=float)
        H_x_ss = np.asarray(pruned_sol.H_x_sigmasigma, dtype=float)
        G_x_ss = np.asarray(pruned_sol.G_x_sigmasigma, dtype=float)
        H_u_ss = np.asarray(pruned_sol.H_u_sigmasigma, dtype=float)
        G_u_ss = np.asarray(pruned_sol.G_u_sigmasigma, dtype=float)

    # Shock covariance
    shock_cov = getattr(pruned_sol, "shock_cov", None)
    if shock_cov is not None:
        Q = np.asarray(shock_cov, dtype=float)
    else:
        Q = np.eye(n_e)
    try:
        L_Q = scipy.linalg.cholesky(Q + 1e-10 * np.eye(n_e), lower=True)
    except Exception:
        L_Q = np.eye(n_e)

    # 3. Measurement error covariance R
    if measurement_error is not None:
        if isinstance(measurement_error, (dict, Mapping)):
            me_diag = [float(measurement_error.get(nm, 0.1)) ** 2 for nm in obs_names]
            R = np.diag(me_diag)
        else:
            R = np.asarray(measurement_error, dtype=float)
            if R.ndim == 1:
                R = np.diag(R ** 2)
    else:
        # Default empirical observation noise
        std_data = np.std(y_mat, axis=0) if T > 1 else np.ones(k)
        std_data = np.where(std_data > 0, std_data * 0.15, 0.1)
        R = np.diag(std_data ** 2)

    R += ridge * np.eye(k)
    log_det_R = float(np.linalg.slogdet(R)[1])
    inv_R = np.linalg.inv(R)

    # 4. Initialize state particles
    x1 = np.zeros((n_particles, n_x), dtype=float)
    x2 = np.zeros((n_particles, n_x), dtype=float)
    x3 = np.zeros((n_particles, n_x), dtype=float)

    weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
    filtered_trajectory = np.empty((T, n_vars), dtype=float)
    total_loglik = 0.0

    const_log_density = -0.5 * (k * math.log(2.0 * math.pi) + log_det_R)

    # 5. Filter recursion over t = 0, ..., T-1
    for t in range(T):
        y_obs = y_mat[t]

        # Draw structural innovations for particles
        z_shocks = rng.normal(0.0, 1.0, size=(n_particles, n_e))
        e_t = z_shocks @ L_Q.T

        # Propagate pruned state components
        # Vectorized 1st order
        x1_next = x1 @ G.T + e_t @ N.T
        y1_next = x1 @ F.T + e_t @ L.T

        # 2nd order outer products
        # kron(x1, x1): (M, n_x*n_x)
        kron_x1_x1 = np.einsum("mi,mj->mij", x1, x1).reshape(n_particles, -1) if n_x > 0 else np.zeros((n_particles, 0))
        kron_x1_e = np.einsum("mi,mj->mij", x1, e_t).reshape(n_particles, -1) if (n_x > 0 and n_e > 0) else np.zeros((n_particles, 0))
        kron_e_e = np.einsum("mi,mj->mij", e_t, e_t).reshape(n_particles, -1) if n_e > 0 else np.zeros((n_particles, 0))

        x2_next = (
            x2 @ G.T
            + 0.5 * (kron_x1_x1 @ H_xx.T)
            + (kron_x1_e @ H_xu.T)
            + 0.5 * (kron_e_e @ H_uu.T)
            + 0.5 * H_ss[None, :]
        )
        y2_next = (
            x2 @ F.T
            + 0.5 * (kron_x1_x1 @ G_xx.T)
            + (kron_x1_e @ G_xu.T)
            + 0.5 * (kron_e_e @ G_uu.T)
            + 0.5 * G_ss[None, :]
        )

        if is_order3:
            kron_x1_x2 = np.einsum("mi,mj->mij", x1, x2).reshape(n_particles, -1) if n_x > 0 else np.zeros((n_particles, 0))
            kron_x2_e = np.einsum("mi,mj->mij", x2, e_t).reshape(n_particles, -1) if (n_x > 0 and n_e > 0) else np.zeros((n_particles, 0))
            kron_x1_3 = np.einsum("mi,mj->mij", kron_x1_x1, x1).reshape(n_particles, -1) if n_x > 0 else np.zeros((n_particles, 0))
            kron_x1_2_e = np.einsum("mi,mj->mij", kron_x1_x1, e_t).reshape(n_particles, -1) if (n_x > 0 and n_e > 0) else np.zeros((n_particles, 0))
            kron_x1_e_2 = np.einsum("mi,mj->mij", x1, kron_e_e).reshape(n_particles, -1) if (n_x > 0 and n_e > 0) else np.zeros((n_particles, 0))
            kron_e_3 = np.einsum("mi,mj->mij", kron_e_e, e_t).reshape(n_particles, -1) if n_e > 0 else np.zeros((n_particles, 0))

            x3_next = (
                x3 @ G.T
                + (kron_x1_x2 @ H_xx.T)
                + (kron_x2_e @ H_xu.T)
                + (1.0 / 6.0) * (kron_x1_3 @ H_xxx.T)
                + 0.5 * (kron_x1_2_e @ H_xxu.T)
                + 0.5 * (kron_x1_e_2 @ H_xuu.T)
                + (1.0 / 6.0) * (kron_e_3 @ H_uuu.T)
                + 0.5 * (x1 @ H_x_ss.T)
                + 0.5 * (e_t @ H_u_ss.T)
            )
            y3_next = (
                x3 @ F.T
                + (kron_x1_x2 @ G_xx.T)
                + (kron_x2_e @ G_xu.T)
                + (1.0 / 6.0) * (kron_x1_3 @ G_xxx.T)
                + 0.5 * (kron_x1_2_e @ G_xxu.T)
                + 0.5 * (kron_x1_e_2 @ G_xuu.T)
                + (1.0 / 6.0) * (kron_e_3 @ G_uuu.T)
                + 0.5 * (x1 @ G_x_ss.T)
                + 0.5 * (e_t @ G_u_ss.T)
            )
        else:
            x3_next = np.zeros_like(x1_next)
            y3_next = np.zeros_like(y1_next)

        # Total model states and controls
        total_states = x1_next + x2_next + x3_next
        total_controls = y1_next + y2_next + y3_next

        # Combined model variable particles of shape (M, n_vars)
        if n_y > 0:
            V_particles = np.hstack([total_states, total_controls]) + ss_vec[None, :]
        else:
            V_particles = total_states + ss_vec[None, :]

        # Predicted observables
        if observation_fn is not None:
            pred_y = np.array([observation_fn(V_particles[m]) for m in range(n_particles)])
        else:
            pred_y = V_particles[:, obs_indices]

        # Residuals and Gaussian observation log density
        res = y_obs[None, :] - pred_y
        quad_form = np.sum((res @ inv_R) * res, axis=1)
        log_w = const_log_density - 0.5 * quad_form

        # Robust log-sum-exp incremental likelihood with low-weight rejuvenation
        log_w_weighted = log_w + np.log(np.maximum(weights, 1e-300))
        max_lw = float(np.max(log_w_weighted))

        if not np.isfinite(max_lw) or max_lw < -600.0:
            # Low-weight rejuvenation under severe outlier observation
            log_lik_incr = max(max_lw, -100.0)
            weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
        else:
            w_exp = np.exp(np.clip(log_w_weighted - max_lw, -700.0, 50.0))
            sum_w_exp = float(np.sum(w_exp))
            if sum_w_exp <= 0.0 or not np.isfinite(sum_w_exp):
                log_lik_incr = max_lw
                weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
            else:
                log_lik_incr = max_lw + math.log(sum_w_exp)
                weights = w_exp / sum_w_exp

        total_loglik += log_lik_incr

        # Filtered state mean at t
        filtered_trajectory[t] = np.sum(weights[:, None] * V_particles, axis=0)

        # Selection: Systematic Resampling on ESS trigger
        ess_t = 1.0 / np.sum(weights ** 2)
        if ess_t < 0.5 * n_particles:
            resample_idx = systematic_resample(weights, rng)
            x1 = x1_next[resample_idx].copy()
            x2 = x2_next[resample_idx].copy()
            x3 = x3_next[resample_idx].copy()
            weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
        else:
            x1 = x1_next.copy()
            x2 = x2_next.copy()
            x3 = x3_next.copy()

    return float(total_loglik), filtered_trajectory


__all__ = [
    "systematic_resample",
    "fixed_tempering_schedule",
    "solve_adaptive_phi",
    "SMCResult",
    "smc_estimate",
    "SMCSampler",
    "bootstrap_particle_filter",
]
