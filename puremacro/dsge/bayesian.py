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
from typing import Any, Callable, Mapping, Sequence
import warnings

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from puremacro.dsge.priors import ensure_prior, Prior, _validate_priors
from puremacro.dsge._results import BayesianIRFResult, PriorPredictiveResult
from puremacro.mcmc import gelman_rubin, geweke_z


# Value neg_log_posterior returns for an infeasible draw. Any Hessian
# stencil point that hits it makes the resulting curvature meaningless.
_NEG_LOG_POST_PENALTY = 1e20

# Exceptions a log-likelihood is *allowed* to raise to mean "this draw is
# infeasible". Anything else is a bug in the caller's function and must
# not be swallowed into a silently degenerate result.
_INFEASIBLE_DRAW_ERRORS = (
    np.linalg.LinAlgError,
    ValueError,
    FloatingPointError,
    ZeroDivisionError,
    OverflowError,
    ArithmeticError,
)

# Consecutive infeasible likelihood evaluations tolerated before giving up.
_MAX_CONSECUTIVE_FAILURES = 500


@dataclass(frozen=True)
class BayesianEstimationResult:
    """Consolidated result of general Bayesian DSGE estimation.

    Attributes
    ----------
    mode : np.ndarray, shape (n_params,)
        Posterior mode parameter estimates.
    mode_se : np.ndarray, shape (n_params,)
        Standard errors at the mode from Laplace approximation
        sqrt(diag(inv(-H))). **NaN** for every parameter when the mode sits
        on a prior bound (or the Hessian stencil otherwise leaves the
        support): the Laplace approximation is undefined there, and a
        finite-looking number would be a measurement of the penalty cliff
        rather than of posterior curvature.
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
    _validate_priors(priors, caller="estimate_dsge_bayesian")
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

    # Likelihood evaluation.
    #
    # The calling convention (array vs dict) is probed ONCE and then bound.
    # The previous code wrapped every call in `except Exception: return
    # -inf`, so a plain bug in the user's log_likelihood_fn -- an
    # AttributeError, a typo in a key -- turned into a chain that never
    # moved and a "result" reporting a degenerate posterior, with no
    # warning. Only the errors in _INFEASIBLE_DRAW_ERRORS are treated as
    # "this draw is infeasible"; everything else propagates.
    call_style: list[str | None] = [None]
    consecutive_failures = [0]

    def _call_log_likelihood(theta: np.ndarray):
        if call_style[0] is None:
            try:
                out = log_likelihood_fn(theta)
                call_style[0] = "array"
                return out
            except (TypeError, KeyError, IndexError) as exc_array:
                try:
                    out = log_likelihood_fn({
                        name: float(theta[i])
                        for i, name in enumerate(param_names)
                    })
                except (TypeError, KeyError, IndexError) as exc_dict:
                    raise TypeError(
                        "estimate_dsge_bayesian: log_likelihood_fn accepts "
                        "neither a 1-D parameter array nor a "
                        "{name: value} dict. Array call raised "
                        f"{type(exc_array).__name__}: {exc_array}; dict call "
                        f"raised {type(exc_dict).__name__}: {exc_dict}."
                    ) from exc_dict
                call_style[0] = "dict"
                return out
        if call_style[0] == "array":
            return log_likelihood_fn(theta)
        return log_likelihood_fn(
            {name: float(theta[i]) for i, name in enumerate(param_names)}
        )

    def log_likelihood_eval(theta: np.ndarray) -> float:
        for i, p in enumerate(prior_objs.values()):
            if not (p.lb <= theta[i] <= p.ub):
                return -math.inf
        try:
            val = _call_log_likelihood(theta)
        except _INFEASIBLE_DRAW_ERRORS as exc:
            consecutive_failures[0] += 1
            if consecutive_failures[0] >= _MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    "estimate_dsge_bayesian: log_likelihood_fn failed on "
                    f"{consecutive_failures[0]} consecutive draws inside the "
                    "prior support; the posterior is not evaluable, so no "
                    "chain would be meaningful. Last failure: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            return -math.inf
        if isinstance(val, (int, float, np.floating)) and np.isfinite(val):
            consecutive_failures[0] = 0
            return float(val)
        consecutive_failures[0] += 1
        if consecutive_failures[0] >= _MAX_CONSECUTIVE_FAILURES:
            raise RuntimeError(
                "estimate_dsge_bayesian: log_likelihood_fn returned a "
                f"non-finite value on {consecutive_failures[0]} consecutive "
                f"draws inside the prior support (last value: {val!r}); the "
                "posterior is not evaluable."
            )
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
        return -lp if np.isfinite(lp) else _NEG_LOG_POST_PENALTY

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
    # A mode ON a prior bound (or anywhere the finite-difference stencil
    # steps into the infeasible region) makes the numerical Hessian a
    # measurement of the 1e20 penalty cliff, not of posterior curvature:
    # the implied variances collapse to ~1e-27, mode_se floors at 1e-6 and
    # -- because the same matrix drives the proposal -- the chain never
    # moves and the reported posterior sd is off by many orders of
    # magnitude. Count the stencil points that hit the penalty and refuse
    # to use such a Hessian for anything.
    penalty_hits = [0]

    def _neg_log_posterior_probed(theta: np.ndarray) -> float:
        val = neg_log_posterior(theta)
        if val >= _NEG_LOG_POST_PENALTY:
            penalty_hits[0] += 1
        return val

    H_neg = _compute_numerical_hessian(
        _neg_log_posterior_probed, mode, h_scale=1e-4
    )
    H_neg_sym = (H_neg + H_neg.T) / 2.0

    prior_vars = np.array([p.std ** 2 for p in prior_objs.values()], dtype=float)
    prior_vars = np.where(np.isfinite(prior_vars) & (prior_vars > 0.0),
                          prior_vars, 1.0)

    at_bound = [
        name for i, (name, p) in enumerate(zip(param_names, prior_objs.values()))
        if min(abs(mode[i] - p.lb), abs(p.ub - mode[i]))
        <= 1e-4 * max(abs(mode[i]), 1.0)
    ]
    hessian_usable = penalty_hits[0] == 0 and np.all(np.isfinite(H_neg_sym))

    if not hessian_usable:
        warnings.warn(
            "estimate_dsge_bayesian: the numerical Hessian at the reported "
            f"mode is not a curvature measurement ({penalty_hits[0]} of the "
            "finite-difference stencil points fall outside the prior "
            "support or gave a non-finite log-posterior"
            + (f"; parameters on a bound: {at_bound}" if at_bound else "")
            + "). The Laplace approximation is undefined there, so mode_se "
            "is returned as NaN and the proposal covariance falls back to "
            "diag(prior_std**2). A mode on a prior bound usually means the "
            "data want a value the prior forbids -- widen the bound rather "
            "than reading these standard errors.",
            UserWarning,
            stacklevel=2,
        )
        sigma_hat = np.diag(prior_vars)
        mode_se = np.full(d, np.nan)
    else:
        try:
            eigvals, eigvecs = np.linalg.eigh(H_neg_sym)
            if np.all(eigvals > 1e-6):
                inv_H = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                sigma_hat = (inv_H + inv_H.T) / 2.0
            else:
                warnings.warn(
                    "estimate_dsge_bayesian: the numerical Hessian at the "
                    f"reported mode has {int(np.sum(eigvals <= 1e-6))} of "
                    f"{d} eigenvalues <= 1e-6 (min {eigvals.min():.3e}); "
                    "that point is not a well-identified local maximum. "
                    "Eigenvalues are floored at 1e-4 to build a usable "
                    "proposal, so mode_se understates the true uncertainty "
                    "in the flat directions.",
                    UserWarning,
                    stacklevel=2,
                )
                clipped_eigvals = np.maximum(eigvals, 1e-4)
                inv_H = eigvecs @ np.diag(1.0 / clipped_eigvals) @ eigvecs.T
                sigma_hat = (inv_H + inv_H.T) / 2.0
        except (np.linalg.LinAlgError, ValueError):
            sigma_hat = np.diag(prior_vars)

        # Ensure strictly positive definite proposal covariance
        sigma_hat = (sigma_hat + sigma_hat.T) / 2.0
        try:
            np.linalg.cholesky(sigma_hat)
        except np.linalg.LinAlgError:
            min_eig = np.min(np.linalg.eigvalsh(sigma_hat))
            ridge = max(1e-6, -min_eig + 1e-4)
            sigma_hat = sigma_hat + ridge * np.eye(d)
        mode_se = np.sqrt(np.maximum(np.diag(sigma_hat), 1e-12))

    sigma_hat = (sigma_hat + sigma_hat.T) / 2.0
    try:
        L_prop = np.linalg.cholesky(sigma_hat)
    except np.linalg.LinAlgError:
        min_eig = np.min(np.linalg.eigvalsh(sigma_hat))
        ridge = max(1e-6, -min_eig + 1e-4)
        sigma_hat = sigma_hat + ridge * np.eye(d)
        L_prop = np.linalg.cholesky(sigma_hat)

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
    if not (0.05 <= acceptance_rate <= 0.60):
        warnings.warn(
            f"estimate_dsge_bayesian: mean acceptance rate "
            f"{acceptance_rate:.3f} is outside [0.05, 0.60]; the chains are "
            "either stuck (too low) or taking steps far smaller than the "
            "posterior scale (too high). The posterior summary below is not "
            "trustworthy.",
            UserWarning,
            stacklevel=2,
        )

    # =========================================================================
    # Step 4: Posterior Summary
    # =========================================================================
    flat_draws = chains.reshape(-1, d)
    draw_sd = flat_draws.std(axis=0, ddof=1) if len(flat_draws) > 1 else np.zeros(d)
    degenerate = [
        name for i, name in enumerate(param_names)
        if draw_sd[i] <= 1e-10 * max(abs(float(mode[i])), 1.0)
    ]
    if degenerate:
        warnings.warn(
            "estimate_dsge_bayesian: the chains did not move for "
            f"{degenerate}; their posterior standard deviation is "
            "numerically zero. This is a failed sampler run, not a sharp "
            "posterior -- do not report these draws as a posterior "
            "distribution.",
            UserWarning,
            stacklevel=2,
        )
    summary_df = pd.DataFrame(
        {
            "mean": flat_draws.mean(axis=0),
            "std": draw_sd,
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


def _resolve_model_with_params(
    model: Any,
    p_dict: Mapping[str, float],
    qz_criterium: float = 1.0 + 1e-8,
) -> Any:
    """Re-solve DSGE model with new parameter values and specified QZ criterium."""
    # 1. Lead-lag Dynare model
    if getattr(model, "_dynare_equations", None) is not None:
        new_params = dict(getattr(model, "_params", {}) or {})
        new_params.update(p_dict)
        from puremacro.dsge.dynare import build_dynare

        ss = getattr(model, "_steady_state_dict", None)
        if ss is None and hasattr(model, "steady_state"):
            ss = model.steady_state.to_dict() if hasattr(model.steady_state, "to_dict") else dict(model.steady_state)

        return build_dynare(
            model._dynare_equations,
            variables=list(model.variables),
            shocks=list(model.shocks),
            params=new_params,
            steady_state=ss,
            check_steady_state=False,
            strict=True,
            qz_criterium=qz_criterium,
        )
    # 2. Parsed Model DAG / AST equations
    elif hasattr(model, "compile_equations") and hasattr(model, "variables"):
        from puremacro.dsge.dynare import build_dynare

        new_params = dict(getattr(model, "parameter_values", {}) or {})
        new_params.update(p_dict)
        eq_fn = model.compile_equations()
        ss = getattr(model, "steady_state", {v: 0.0 for v in model.variables})
        return build_dynare(
            eq_fn,
            variables=list(model.variables),
            shocks=list(model.shocks),
            params=new_params,
            steady_state=ss,
            check_steady_state=False,
            strict=True,
            qz_criterium=qz_criterium,
        )
    # 3. Klein timing build() model
    elif getattr(model, "_equations", None) is not None:
        from puremacro.dsge.build import build

        new_params = dict(getattr(model, "_params", {}) or {})
        new_params.update(p_dict)
        ss = getattr(model, "_steady_state_dict", None)
        if ss is None and hasattr(model, "steady_state"):
            ss = model.steady_state.to_dict() if hasattr(model.steady_state, "to_dict") else dict(model.steady_state)

        return build(
            model._equations,
            variables=list(model.variables),
            states=list(model.states),
            shocks=list(model.shocks),
            params=new_params,
            steady_state=ss,
            strict=True,
            qz_criterium=qz_criterium,
        )
    else:
        raise ValueError(f"Cannot re-solve model of type {type(model).__name__} with updated parameters.")


def _sample_from_prior(spec: Any, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Sample parameter values from a prior specification."""
    if isinstance(spec, Prior):
        spec = spec.to_dict() if hasattr(spec, "to_dict") else {
            "dist": spec.dist,
            "mean": spec.mean,
            "std": spec.std,
            "lb": spec.lb,
            "ub": spec.ub,
            **getattr(spec, "_extra", {}),
        }
    dist = str(spec.get("dist", "normal")).lower()
    mean = float(spec.get("mean", 0.0))
    std = float(spec.get("std", 1.0))
    lb = float(spec.get("lb", -math.inf))
    ub = float(spec.get("ub", math.inf))

    if dist == "uniform":
        low = lb if np.isfinite(lb) else (mean - math.sqrt(3.0) * std)
        high = ub if np.isfinite(ub) else (mean + math.sqrt(3.0) * std)
        return rng.uniform(low, high, size=n_samples)

    elif dist == "normal":
        samples = rng.normal(mean, std, size=n_samples)
        if np.isfinite(lb) or np.isfinite(ub):
            for idx in range(n_samples):
                val = samples[idx]
                attempts = 0
                while (val < lb or val > ub) and attempts < 100:
                    val = rng.normal(mean, std)
                    attempts += 1
                if val < lb or val > ub:
                    val = np.clip(val, lb, ub)
                samples[idx] = val
        return samples

    elif dist == "beta":
        shift = float(spec.get("shift", 0.0))
        scale = float(spec.get("scale", 1.0))
        norm_mean = (mean - shift) / scale
        norm_var = (std / scale) ** 2
        if norm_var >= norm_mean * (1.0 - norm_mean) or norm_mean <= 0.0 or norm_mean >= 1.0:
            return rng.uniform(shift, shift + scale, size=n_samples)
        temp = norm_mean * (1.0 - norm_mean) / norm_var - 1.0
        a = norm_mean * temp
        b = (1.0 - norm_mean) * temp
        raw = rng.beta(a, b, size=n_samples)
        return shift + scale * raw

    elif dist == "gamma":
        shift = float(spec.get("shift", 0.0))
        adj_mean = mean - shift
        if adj_mean <= 0.0 or std <= 0.0:
            return rng.normal(mean, std, size=n_samples)
        k = (adj_mean / std) ** 2
        theta = (std ** 2) / adj_mean
        raw = rng.gamma(shape=k, scale=theta, size=n_samples)
        return shift + raw

    elif dist == "invgamma":
        if "s" in spec and "nu" in spec:
            s_val = float(spec["s"])
            nu_val = float(spec["nu"])
        else:
            nu_val = 4.0
            s_val = mean * math.sqrt(nu_val / 2.0)
        raw_gamma = rng.gamma(shape=nu_val / 2.0, scale=2.0, size=n_samples)
        raw_gamma = np.maximum(raw_gamma, 1e-12)
        raw_x2 = (s_val ** 2) / raw_gamma
        return np.sqrt(np.maximum(raw_x2, 1e-12))

    elif dist == "weibull":
        shape = float(spec.get("shape", 2.0))
        scale = float(spec.get("scale", 1.0))
        shift = float(spec.get("shift", 0.0))
        return shift + rng.weibull(shape, size=n_samples) * scale

    else:
        return rng.normal(mean, std, size=n_samples)


def bayesian_irf(
    model: Any,
    draws: Any = None,
    param_names: Sequence[str] | None = None,
    *,
    priors: Mapping[str, Any] | None = None,
    n_draws: int | None = None,
    seed: int = 0,
    periods: int | None = None,
    shocks: Sequence[str] | str | None = None,
    horizon: int = 40,
    bands: Sequence[float] = (0.68, 0.90, 0.95),
    quantiles: Sequence[float] | None = None,
    shock: str | None = None,
    variables: Sequence[str] | None = None,
    size: float = 1.0,
    burn_in: int = 0,
    qz_criterium: float = 1.0 + 1e-8,
) -> BayesianIRFResult:
    """Compute Bayesian impulse response functions and credible intervals across parameter draws."""
    if periods is not None:
        horizon = int(periods)

    # Swap arguments if order is inverted
    if hasattr(draws, "variables") and not hasattr(model, "variables"):
        model, draws = draws, model
    elif isinstance(model, (np.ndarray, pd.DataFrame)) and not isinstance(draws, (np.ndarray, pd.DataFrame)):
        model, draws = draws, model

    # Resolve parameter draws array and names
    if draws is None and priors is not None:
        num_draws = int(n_draws) if n_draws is not None else 50
        rng_p = np.random.default_rng(seed)
        param_names = list(priors.keys())
        draw_cols = []
        for p_name in param_names:
            p_spec = priors[p_name]
            draw_cols.append(_sample_from_prior(p_spec, num_draws, rng_p))
        draws_arr = np.column_stack(draw_cols)
    elif isinstance(draws, BayesianEstimationResult):
        if param_names is None:
            param_names = list(draws.param_names)
        chains = draws.chains
        if chains.ndim == 3:
            post_burn = chains[:, burn_in:, :]
            draws_arr = post_burn.reshape(-1, chains.shape[-1])
        else:
            draws_arr = np.asarray(chains[burn_in:], dtype=float)
    elif isinstance(draws, pd.DataFrame):
        if param_names is None:
            param_names = list(draws.columns)
        draws_arr = draws[list(param_names)].to_numpy(dtype=float)
    elif isinstance(draws, Mapping):
        if param_names is None:
            param_names = list(draws.keys())
        draws_arr = np.column_stack([draws[k] for k in param_names])
    elif isinstance(draws, np.ndarray):
        if draws.ndim == 3:
            post_burn = draws[:, burn_in:, :]
            draws_arr = post_burn.reshape(-1, draws.shape[-1])
        else:
            draws_arr = np.asarray(draws, dtype=float)
        if param_names is None:
            model_p = getattr(model, "_params", None)
            if model_p:
                param_names = list(model_p.keys())[: draws_arr.shape[1]]
            else:
                param_names = [f"param_{i+1}" for i in range(draws_arr.shape[1])]
    else:
        raise TypeError(f"Unsupported type for draws: {type(draws).__name__}")

    effective_n_draws = len(draws_arr)
    if effective_n_draws == 0:
        raise ValueError("draws is empty")

    p_names = list(param_names)
    model_vars = list(variables) if variables is not None else list(getattr(model, "variables", []))
    model_shocks = list(getattr(model, "shocks", []))

    if shocks is not None and not isinstance(shocks, str):
        target_shocks = [s for s in shocks if s in model_shocks]
    elif shock is not None:
        target_shocks = [shock]
    elif model_shocks:
        target_shocks = [model_shocks[0]]
    else:
        target_shocks = ["e_1"]

    shock_irf_lists: dict[str, list[np.ndarray]] = {s: [] for s in target_shocks}
    n_valid = 0

    for i in range(effective_n_draws):
        p_row = draws_arr[i]
        p_dict = dict(zip(p_names, p_row))
        try:
            new_m = _resolve_model_with_params(model, p_dict, qz_criterium=qz_criterium)
            if not getattr(new_m, "is_determinate", True):
                continue
            for s in target_shocks:
                irf_df = new_m.irf(s, horizon=horizon, size=size)
                v_cols = [v for v in model_vars if v in irf_df.columns]
                shock_irf_lists[s].append(irf_df[v_cols].to_numpy(dtype=float))
            n_valid += 1
        except Exception:
            continue

    if n_valid == 0:
        raise RuntimeError(
            f"All {effective_n_draws} parameter draws were indeterminate or failed to solve."
        )

    determinacy_rate = float(n_valid / effective_n_draws)
    idx = pd.RangeIndex(0, horizon + 1, name="horizon")

    q_set: set[float] = {0.5}
    if quantiles is not None:
        for q in quantiles:
            q_set.add(float(q))
    if bands is not None:
        for b in bands:
            b_val = float(b)
            if b_val > 1.0:
                b_val = b_val / 100.0
            tail = (1.0 - b_val) / 2.0
            q_set.add(round(tail, 4))
            q_set.add(round(1.0 - tail, 4))

    q_list = sorted(q_set)

    all_medians: dict[str, pd.DataFrame] = {}
    all_bands: dict[str, dict[float, tuple[pd.DataFrame, pd.DataFrame]]] = {}
    all_quantiles: dict[str, dict[float, pd.DataFrame]] = {}

    for s in target_shocks:
        stacked = np.array(shock_irf_lists[s])
        s_quantiles: dict[float, pd.DataFrame] = {}
        for q in q_list:
            q_arr = np.quantile(stacked, q, axis=0)
            s_quantiles[q] = pd.DataFrame(q_arr, index=idx, columns=model_vars)

        s_median = s_quantiles[0.5]
        all_medians[s] = s_median
        all_quantiles[s] = s_quantiles

        s_bands: dict[float, tuple[pd.DataFrame, pd.DataFrame]] = {}
        if bands is not None:
            for b in bands:
                b_val = float(b)
                if b_val > 1.0:
                    b_val = b_val / 100.0
                tail = (1.0 - b_val) / 2.0
                low_q = min(s_quantiles.keys(), key=lambda q: abs(q - tail))
                high_q = min(s_quantiles.keys(), key=lambda q: abs(q - (1.0 - tail)))
                s_bands[b_val] = (s_quantiles[low_q], s_quantiles[high_q])
        all_bands[s] = s_bands

    if shock is not None or len(target_shocks) == 1:
        chosen_shock = shock if shock is not None else target_shocks[0]
        final_median = all_medians[chosen_shock]
        final_bands = all_bands[chosen_shock]
        final_quantiles = all_quantiles[chosen_shock]
    else:
        final_median = all_medians
        final_bands = all_bands
        final_quantiles = all_quantiles

    return BayesianIRFResult(
        median=final_median,
        bands=final_bands,
        quantiles=final_quantiles,
        variables=tuple(model_vars),
        shocks=tuple(target_shocks),
        n_draws=effective_n_draws,
        n_valid=n_valid,
        determinacy_rate=determinacy_rate,
        horizon=horizon,
    )


def prior_predictive(
    model: Any,
    priors: Mapping[str, Any] | None = None,
    *,
    n_draws: int = 500,
    draws: np.ndarray | pd.DataFrame | None = None,
    seed: int = 0,
    moments: bool = True,
    irf: int | bool = 40,
    irf_periods: int | None = None,
    qz_criterium: float = 1.0 + 1e-8,
) -> PriorPredictiveResult:
    """Simulate theoretical moments and impulse responses across prior parameter distributions."""
    if irf_periods is not None:
        irf = int(irf_periods)
    rng = np.random.default_rng(seed)

    resolved_priors: dict[str, Any] = {}
    if draws is not None:
        if isinstance(draws, pd.DataFrame):
            param_names = list(draws.columns)
            param_draws = draws.to_numpy(dtype=float)
        else:
            param_draws = np.asarray(draws, dtype=float)
            model_p = getattr(model, "_params", {}) or {}
            param_names = list(model_p.keys())[: param_draws.shape[1]] if model_p else [f"p_{i}" for i in range(param_draws.shape[1])]
        total_draws = len(param_draws)
    else:
        total_draws = int(n_draws)
        if priors is not None:
            resolved_priors = dict(priors)
        elif getattr(model, "_estimated_params", None) is not None:
            resolved_priors = dict(model._estimated_params.get("priors", {}))
        elif hasattr(model, "estimated_params") and model.estimated_params is not None:
            resolved_priors = dict(getattr(model.estimated_params, "priors", {}))
        else:
            base_params = dict(getattr(model, "_params", {}) or getattr(model, "parameter_values", {}) or {})
            for p_name, p_val in base_params.items():
                val = float(p_val)
                if 0.0 < val < 1.0:
                    resolved_priors[p_name] = {
                        "dist": "uniform",
                        "lb": max(1e-4, val * 0.85),
                        "ub": min(0.999, val * 1.15),
                        "mean": val,
                        "std": (val * 0.3) / math.sqrt(12.0),
                    }
                elif val > 0.0:
                    resolved_priors[p_name] = {
                        "dist": "uniform",
                        "lb": max(1e-4, val * 0.85),
                        "ub": val * 1.15,
                        "mean": val,
                        "std": (val * 0.3) / math.sqrt(12.0),
                    }
                else:
                    resolved_priors[p_name] = {
                        "dist": "normal",
                        "mean": val,
                        "std": 0.05,
                    }

        param_names = list(resolved_priors.keys())
        cols = []
        for p_name in param_names:
            p_spec = resolved_priors[p_name]
            cols.append(_sample_from_prior(p_spec, total_draws, rng))
        param_draws = np.column_stack(cols) if cols else np.zeros((total_draws, 0))

    prior_draws_df = pd.DataFrame(param_draws, columns=param_names)

    valid_rows = []
    moment_records: list[dict[str, float]] = []

    for i in range(total_draws):
        row = param_draws[i]
        p_dict = dict(zip(param_names, row))
        try:
            new_m = _resolve_model_with_params(model, p_dict, qz_criterium=qz_criterium)
            if not getattr(new_m, "is_determinate", True):
                continue
            if moments:
                tm = new_m.theoretical_moments()
                tm_df = tm.moments if hasattr(tm, "moments") else tm.to_frame()
                rec = {}
                for v in tm_df.index:
                    for col in tm_df.columns:
                        rec[f"{v}_{col}"] = float(tm_df.loc[v, col])
                moment_records.append(rec)
            valid_rows.append(row)
        except Exception:
            continue

    n_valid = len(valid_rows)
    valid_draws_df = pd.DataFrame(valid_rows, columns=param_names) if valid_rows else pd.DataFrame(columns=param_names)
    determinacy_rate = float(n_valid / total_draws) if total_draws > 0 else 0.0

    if moment_records:
        rec_df = pd.DataFrame(moment_records)
        prior_moments = pd.DataFrame(
            {
                "mean": rec_df.mean(),
                "std": rec_df.std(),
                "p5": rec_df.quantile(0.05),
                "median": rec_df.median(),
                "p95": rec_df.quantile(0.95),
            }
        )
    else:
        prior_moments = pd.DataFrame(columns=["mean", "std", "p5", "median", "p95"])

    return PriorPredictiveResult(
        prior_moments=prior_moments,
        prior_draws=prior_draws_df,
        valid_draws=valid_draws_df,
        param_names=tuple(param_names),
        determinacy_rate=determinacy_rate,
        n_draws=total_draws,
        n_valid=n_valid,
        priors=resolved_priors if draws is None else None,
    )


__all__ = [
    "BayesianEstimationResult",
    "estimate_dsge_bayesian",
    "bayesian_irf",
    "BayesianIRFResult",
    "prior_predictive",
    "PriorPredictiveResult",
]

