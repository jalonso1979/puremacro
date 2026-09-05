"""General Bayesian DSGE Estimation pipeline for puremacro.

Provides:
- BayesianEstimationResult: frozen dataclass containing estimation mode, SE,
  MCMC chains, posterior summary, and diagnostics.
- estimate_dsge_bayesian: model-agnostic Bayesian estimation driver implementing
  mode finding (L-BFGS-B / Nelder-Mead), Laplace approximation (numerical Hessian),
  and Random-Walk Metropolis-Hastings (RWMH) with adaptive proposal scaling.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from puremacro.dsge.priors import ensure_prior, Prior
from puremacro.mcmc import gelman_rubin, geweke_z


@dataclass(frozen=True)
class BayesianEstimationResult:
    """Consolidated result of general Bayesian DSGE estimation.

    Attributes
    ----------
    mode : np.ndarray, shape (n_params,)
        Posterior mode parameter estimates.
    mode_se : np.ndarray, shape (n_params,)
        Standard errors at the mode from Laplace approximation sqrt(diag(inv(-H))).
    param_names : list[str]
        List of parameter names in estimation order.
    log_posterior_mode : float
        Value of the log-posterior density evaluated at the mode.
    chains : np.ndarray, shape (n_chains, n_draws, n_params)
        Retained post-burn-in MCMC chains.
    acceptance_rate : float
        Mean Metropolis-Hastings acceptance rate across all chains.
    posterior_summary : pd.DataFrame
        Table of posterior summary statistics with columns:
        ['mean', 'std', '16%', '50%', '84%', '5%', '95%'].
    diagnostics : dict[str, float]
        MCMC convergence diagnostics including split-Rhat and Geweke z-scores.
    priors : dict[str, Any] | None, optional
        Dictionary of prior specifications used for estimation.
    """

    mode: np.ndarray
    mode_se: np.ndarray
    param_names: list[str]
    log_posterior_mode: float
    chains: np.ndarray
    acceptance_rate: float
    posterior_summary: pd.DataFrame
    diagnostics: dict[str, float]
    priors: dict[str, Any] | None = None

    def to_frame(self) -> pd.DataFrame:
        """Return posterior summary statistics as a DataFrame."""
        return self.posterior_summary.copy()

    def summary(self) -> pd.DataFrame:
        """Return posterior summary statistics as a DataFrame."""
        return self.posterior_summary.copy()

    def to_markdown(self, **kwargs) -> str:
        """Render posterior summary table as Markdown."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.posterior_summary, **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Render posterior summary table as LaTeX tabular."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.posterior_summary, **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Render posterior summary table as Typst table."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.posterior_summary, **kwargs)

    def plot_posteriors(self, style: str = "publication") -> Figure:
        """Plot marginal posterior distributions for all estimated parameters.

        Parameters
        ----------
        style : str, default 'publication'
            Plot styling theme. 'publication' uses clean academic grayscale
            styling with despined axes and credible intervals.

        Returns
        -------
        matplotlib.figure.Figure
        """
        k = len(self.param_names)
        if k == 1:
            n_cols, n_rows = 1, 1
        elif k <= 4:
            n_cols = 2
            n_rows = (k + 1) // 2
        elif k <= 9:
            n_cols = 3
            n_rows = (k + 2) // 3
        else:
            n_cols = 4
            n_rows = (k + 3) // 4

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.8 * n_cols, 2.8 * n_rows), squeeze=False
        )

        for i, name in enumerate(self.param_names):
            r, c = i // n_cols, i % n_cols
            ax = axes[r, c]
            draws = self.chains[:, :, i].ravel()
            draws = draws[np.isfinite(draws)]

            if style == "publication":
                ax.hist(
                    draws,
                    bins=25,
                    density=True,
                    color="0.85",
                    edgecolor="0.6",
                    linewidth=0.5,
                    alpha=0.6,
                )
                try:
                    kde = stats.gaussian_kde(draws)
                    x_grid = np.linspace(draws.min(), draws.max(), 200)
                    ax.plot(
                        x_grid,
                        kde(x_grid),
                        color="0.1",
                        linewidth=1.6,
                        label="Posterior KDE",
                    )
                except Exception:
                    pass

                mode_val = float(self.mode[i])
                ax.axvline(
                    mode_val,
                    color="0.0",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Mode: {mode_val:.3g}",
                )

                q16, q84 = np.percentile(draws, [16, 84])
                ax.axvspan(
                    q16, q84, color="0.7", alpha=0.25, label="68% CI"
                )

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_title(name, fontsize=11, fontweight="bold")
                ax.set_xlabel("Value", fontsize=9)
                ax.set_ylabel("Density", fontsize=9)
                ax.legend(frameon=False, fontsize=8)
            else:
                ax.hist(draws, bins=25, density=True, alpha=0.7)
                ax.axvline(float(self.mode[i]), color="r", linestyle="--")
                ax.set_title(name)

        # Hide any unused subplots
        for i in range(k, n_rows * n_cols):
            axes[i // n_cols, i % n_cols].set_visible(False)

        fig.tight_layout()
        return fig

    def plot_priors_posteriors(self) -> Figure:
        """Plot prior density vs posterior marginal density for each parameter.

        Returns
        -------
        matplotlib.figure.Figure
        """
        k = len(self.param_names)
        if k == 1:
            n_cols, n_rows = 1, 1
        elif k <= 4:
            n_cols = 2
            n_rows = (k + 1) // 2
        elif k <= 9:
            n_cols = 3
            n_rows = (k + 2) // 3
        else:
            n_cols = 4
            n_rows = (k + 3) // 4

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.8 * n_cols, 2.8 * n_rows), squeeze=False
        )

        for i, name in enumerate(self.param_names):
            r, c = i // n_cols, i % n_cols
            ax = axes[r, c]
            draws = self.chains[:, :, i].ravel()
            draws = draws[np.isfinite(draws)]

            # Determine support range
            d_min, d_max = np.percentile(draws, [0.5, 99.5])
            prior_obj = None
            if self.priors and name in self.priors:
                prior_obj = ensure_prior(self.priors[name])

            if prior_obj is not None:
                p_mean = prior_obj.mean
                p_std = prior_obj.std
                p_lb = prior_obj.lb
                p_ub = prior_obj.ub
                x_lo = max(p_lb, min(d_min, p_mean - 3.2 * p_std))
                x_hi = min(p_ub, max(d_max, p_mean + 3.2 * p_std))
                if not (np.isfinite(x_lo) and np.isfinite(x_hi)) or x_lo >= x_hi:
                    x_lo, x_hi = d_min - 0.1 * abs(d_min), d_max + 0.1 * abs(d_max)
                x_grid = np.linspace(x_lo, x_hi, 300)

                # Prior curve
                try:
                    prior_pdf = prior_obj.pdf(x_grid)
                    ax.plot(
                        x_grid,
                        prior_pdf,
                        color="0.45",
                        linestyle="--",
                        linewidth=1.4,
                        label="Prior",
                    )
                except Exception:
                    pass
            else:
                x_grid = np.linspace(d_min, d_max, 300)

            # Posterior curve
            try:
                kde = stats.gaussian_kde(draws)
                post_pdf = kde(x_grid)
                ax.plot(
                    x_grid,
                    post_pdf,
                    color="0.0",
                    linestyle="-",
                    linewidth=1.7,
                    label="Posterior",
                )
                ax.fill_between(x_grid, 0, post_pdf, color="0.85", alpha=0.45)
            except Exception:
                ax.hist(
                    draws,
                    bins=20,
                    density=True,
                    color="0.8",
                    alpha=0.5,
                    label="Posterior",
                )

            # Mode indicator
            mode_val = float(self.mode[i])
            ax.axvline(
                mode_val,
                color="0.2",
                linestyle=":",
                linewidth=1.2,
                label=f"Mode ({mode_val:.3g})",
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title(name, fontsize=11, fontweight="bold")
            ax.set_xlabel("Value", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.legend(frameon=False, fontsize=8)

        # Hide any unused subplots
        for i in range(k, n_rows * n_cols):
            axes[i // n_cols, i % n_cols].set_visible(False)

        fig.tight_layout()
        return fig


def _compute_numerical_hessian(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    h_scale: float = 1e-4,
) -> np.ndarray:
    """Central-difference numerical Hessian scaled by parameter magnitude."""
    x0 = np.asarray(x0, dtype=float).ravel()
    n = len(x0)
    H = np.zeros((n, n), dtype=float)
    steps = np.zeros(n, dtype=float)
    for i in range(n):
        steps[i] = h_scale * max(abs(x0[i]), 1.0)

    for i in range(n):
        ei = np.zeros(n, dtype=float)
        ei[i] = steps[i]
        for j in range(i, n):
            ej = np.zeros(n, dtype=float)
            ej[j] = steps[j]
            f_pp = f(x0 + ei + ej)
            f_pm = f(x0 + ei - ej)
            f_mp = f(x0 - ei + ej)
            f_mm = f(x0 - ei - ej)
            hij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j])
            H[i, j] = hij
            H[j, i] = hij
    return H


def estimate_dsge_bayesian(
    log_likelihood_fn: Callable[[Any], float],
    priors: dict[str, Any],
    initial_params: np.ndarray | None = None,
    n_draws: int = 1000,
    n_burn: int = 200,
    n_chains: int = 2,
    target_accept: float = 0.28,
    tune_interval: int = 100,
    seed: int = 42,
) -> BayesianEstimationResult:
    """Bayesian estimation of DSGE parameters via Random-Walk Metropolis-Hastings.

    Parameters
    ----------
    log_likelihood_fn : callable
        Log-likelihood function accepting either a 1D parameter array
        or a parameter dict mapping names to floats.
    priors : dict[str, Any]
        Dictionary of parameter priors. Values may be Prior instances
        (e.g., BetaPrior, InvGammaPrior) or specification dicts.
    initial_params : np.ndarray | None, default None
        Initial parameter values for mode optimization. If None, prior means
        are used.
    n_draws : int, default 1000
        Number of retained post-burn-in MCMC draws per chain.
    n_burn : int, default 200
        Number of burn-in draws dropped per chain.
    n_chains : int, default 2
        Number of independent MCMC chains.
    target_accept : float, default 0.28
        Target acceptance rate for adaptive proposal scaling during burn-in.
    tune_interval : int, default 100
        Frequency of proposal scale adjustments during burn-in.
    seed : int, default 42
        Master RNG seed.

    Returns
    -------
    BayesianEstimationResult
    """
    param_names = list(priors.keys())
    d = len(param_names)
    prior_objs = {name: ensure_prior(spec) for name, spec in priors.items()}
    bounds = [(p.lb, p.ub) for p in prior_objs.values()]

    # Initial parameter vector
    if initial_params is not None:
        init_vec = np.asarray(initial_params, dtype=float).ravel().copy()
        if len(init_vec) != d:
            raise ValueError(
                f"initial_params length {len(init_vec)} does not match {d} priors"
            )
    else:
        init_vec = np.zeros(d, dtype=float)
        for i, p in enumerate(prior_objs.values()):
            val = p.mean
            if np.isfinite(p.lb) and val <= p.lb:
                val = p.lb + 1e-3
            if np.isfinite(p.ub) and val >= p.ub:
                val = p.ub - 1e-3
            init_vec[i] = val

    # Prior evaluation
    def log_prior_eval(theta: np.ndarray) -> float:
        total_lp = 0.0
        for i, p in enumerate(prior_objs.values()):
            val = theta[i]
            if not (p.lb <= val <= p.ub):
                return -math.inf
            lp = p.logpdf(val)
            if not np.isfinite(lp):
                return -math.inf
            total_lp += float(lp)
        return total_lp

    # Likelihood evaluation
    def log_likelihood_eval(theta: np.ndarray) -> float:
        for i, p in enumerate(prior_objs.values()):
            if not (p.lb <= theta[i] <= p.ub):
                return -math.inf
        try:
            val = log_likelihood_fn(theta)
            if isinstance(val, (int, float, np.floating)) and np.isfinite(val):
                return float(val)
            return -math.inf
        except (TypeError, KeyError, IndexError):
            try:
                param_dict = {
                    name: float(theta[i]) for i, name in enumerate(param_names)
                }
                val = log_likelihood_fn(param_dict)
                if isinstance(val, (int, float, np.floating)) and np.isfinite(val):
                    return float(val)
                return -math.inf
            except Exception:
                return -math.inf
        except Exception:
            return -math.inf

    # Posterior evaluation
    def log_posterior_eval(theta: np.ndarray) -> float:
        lp = log_prior_eval(theta)
        if not np.isfinite(lp):
            return -math.inf
        ll = log_likelihood_eval(theta)
        if not np.isfinite(ll):
            return -math.inf
        return float(lp + ll)

    def neg_log_posterior(theta: np.ndarray) -> float:
        lp = log_posterior_eval(theta)
        return -lp if np.isfinite(lp) else 1e20

    # Ensure finite start for optimization
    if not np.isfinite(log_posterior_eval(init_vec)):
        rng_init = np.random.default_rng(seed)
        found = False
        for _ in range(300):
            cand = np.array([
                rng_init.uniform(
                    p.lb if np.isfinite(p.lb) else -5.0,
                    p.ub if np.isfinite(p.ub) else 5.0,
                )
                for p in prior_objs.values()
            ])
            if np.isfinite(log_posterior_eval(cand)):
                init_vec = cand
                found = True
                break
        if not found:
            init_vec = np.array([
                np.clip(p.mean, p.lb + 1e-4, p.ub - 1e-4) if (np.isfinite(p.lb) and np.isfinite(p.ub)) else p.mean
                for p in prior_objs.values()
            ])

    # =========================================================================
    # Step 1: Mode Finding
    # =========================================================================
    best_x = init_vec.copy()
    best_fun = neg_log_posterior(init_vec)

    # Try L-BFGS-B
    try:
        opt_lbfgs = minimize(
            neg_log_posterior,
            init_vec,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-8, "gtol": 1e-5},
        )
        if opt_lbfgs.success and np.isfinite(opt_lbfgs.fun) and opt_lbfgs.fun < best_fun:
            best_x = opt_lbfgs.x
            best_fun = opt_lbfgs.fun
    except Exception:
        pass

    # Try Nelder-Mead if L-BFGS-B did not converge or to refine
    try:
        opt_nm = minimize(
            neg_log_posterior,
            best_x,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 1500, "xatol": 1e-5, "fatol": 1e-6},
        )
        if np.isfinite(opt_nm.fun) and opt_nm.fun < best_fun:
            best_x = opt_nm.x
            best_fun = opt_nm.fun
    except Exception:
        pass

    mode = np.asarray(best_x, dtype=float)
    log_posterior_mode = float(-best_fun)

    # =========================================================================
    # Step 2: Laplace Approximation
    # =========================================================================
    H_neg = _compute_numerical_hessian(neg_log_posterior, mode, h_scale=1e-4)
    H_neg_sym = (H_neg + H_neg.T) / 2.0

    try:
        eigvals, eigvecs = np.linalg.eigh(H_neg_sym)
        if np.all(eigvals > 1e-6):
            inv_H = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            sigma_hat = (inv_H + inv_H.T) / 2.0
        else:
            clipped_eigvals = np.maximum(eigvals, 1e-4)
            inv_H = eigvecs @ np.diag(1.0 / clipped_eigvals) @ eigvecs.T
            sigma_hat = (inv_H + inv_H.T) / 2.0
    except (np.linalg.LinAlgError, ValueError):
        prior_vars = np.array([p.std ** 2 for p in prior_objs.values()])
        sigma_hat = np.diag(prior_vars)

    # Ensure strictly positive definite proposal covariance
    sigma_hat = (sigma_hat + sigma_hat.T) / 2.0
    try:
        L_prop = np.linalg.cholesky(sigma_hat)
    except np.linalg.LinAlgError:
        min_eig = np.min(np.linalg.eigvalsh(sigma_hat))
        ridge = max(1e-6, -min_eig + 1e-4)
        sigma_hat = sigma_hat + ridge * np.eye(d)
        L_prop = np.linalg.cholesky(sigma_hat)

    mode_se = np.sqrt(np.maximum(np.diag(sigma_hat), 1e-12))

    # =========================================================================
    # Step 3: Random Walk Metropolis-Hastings (RWMH)
    # =========================================================================
    c_0 = 2.38 / math.sqrt(d)
    chains = np.empty((n_chains, n_draws, d), dtype=float)
    chain_accept_rates = []

    eff_tune_interval = min(tune_interval, max(10, n_burn // 4)) if n_burn > 0 else 1

    for chain_idx in range(n_chains):
        chain_rng = np.random.default_rng(seed + chain_idx * 1000 + 1)

        # Initial point for chain
        if chain_idx == 0:
            curr_theta = mode.copy()
        else:
            perturb = 0.05 * (L_prop @ chain_rng.standard_normal(d))
            curr_theta = mode + perturb
            if not np.isfinite(log_posterior_eval(curr_theta)):
                curr_theta = mode.copy()

        curr_lp = log_posterior_eval(curr_theta)
        c_scale = c_0

        # Burn-in phase with adaptive scale factor
        window_accepts = 0
        window_count = 0
        for _ in range(n_burn):
            z = chain_rng.standard_normal(d)
            prop_theta = curr_theta + c_scale * (L_prop @ z)
            prop_lp = log_posterior_eval(prop_theta)

            if np.isfinite(prop_lp):
                log_alpha = prop_lp - curr_lp
                if np.log(chain_rng.uniform()) < log_alpha:
                    curr_theta = prop_theta
                    curr_lp = prop_lp
                    window_accepts += 1
            window_count += 1

            if window_count >= eff_tune_interval:
                acc_rate = window_accepts / window_count
                c_scale *= math.exp(0.5 * (acc_rate - target_accept))
                c_scale = float(np.clip(c_scale, 1e-3, 50.0))
                window_accepts = 0
                window_count = 0

        # Retained sampling phase (scale factor frozen)
        chain_accepts = 0
        for t in range(n_draws):
            z = chain_rng.standard_normal(d)
            prop_theta = curr_theta + c_scale * (L_prop @ z)
            prop_lp = log_posterior_eval(prop_theta)

            if np.isfinite(prop_lp):
                log_alpha = prop_lp - curr_lp
                if np.log(chain_rng.uniform()) < log_alpha:
                    curr_theta = prop_theta
                    curr_lp = prop_lp
                    chain_accepts += 1

            chains[chain_idx, t, :] = curr_theta

        chain_accept_rates.append(chain_accepts / max(n_draws, 1))

    acceptance_rate = float(np.mean(chain_accept_rates))

    # =========================================================================
    # Step 4: Posterior Summary
    # =========================================================================
    flat_draws = chains.reshape(-1, d)
    summary_df = pd.DataFrame(
        {
            "mean": flat_draws.mean(axis=0),
            "std": flat_draws.std(axis=0, ddof=1) if len(flat_draws) > 1 else np.zeros(d),
            "16%": np.percentile(flat_draws, 16, axis=0),
            "50%": np.percentile(flat_draws, 50, axis=0),
            "84%": np.percentile(flat_draws, 84, axis=0),
            "5%": np.percentile(flat_draws, 5, axis=0),
            "95%": np.percentile(flat_draws, 95, axis=0),
        },
        index=param_names,
    )

    # =========================================================================
    # Step 5: Diagnostics (Gelman-Rubin Split-Rhat and Geweke)
    # =========================================================================
    diagnostics: dict[str, float] = {
        "acceptance_rate": acceptance_rate,
    }

    half = n_draws // 2
    r_hats = []
    gewekes = []

    for i, name in enumerate(param_names):
        # Split-Rhat
        if half >= 4:
            split_chains = np.empty((2 * n_chains, half), dtype=float)
            for c_idx in range(n_chains):
                split_chains[2 * c_idx] = chains[c_idx, :half, i]
                split_chains[2 * c_idx + 1] = chains[c_idx, half : 2 * half, i]
            gr = gelman_rubin(split_chains)
            r_hat_val = float(gr["R_hat"])
        else:
            r_hat_val = 1.0

        # Geweke z-score
        gw_val = float(geweke_z(chains[0, :, i]))

        diagnostics[f"r_hat_{name}"] = r_hat_val
        diagnostics[f"geweke_z_{name}"] = gw_val
        r_hats.append(r_hat_val)
        gewekes.append(gw_val)

    diagnostics["r_hat_max"] = float(np.nanmax(r_hats)) if r_hats else 1.0
    diagnostics["geweke_z_max"] = float(np.nanmax(np.abs(gewekes))) if gewekes else 0.0

    return BayesianEstimationResult(
        mode=mode,
        mode_se=mode_se,
        param_names=param_names,
        log_posterior_mode=log_posterior_mode,
        chains=chains,
        acceptance_rate=acceptance_rate,
        posterior_summary=summary_df,
        diagnostics=diagnostics,
        priors=priors,
    )


__all__ = [
    "BayesianEstimationResult",
    "estimate_dsge_bayesian",
]
