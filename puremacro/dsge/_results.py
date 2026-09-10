"""Frozen-dataclass result types for puremacro.dsge."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DSGEPosteriorResult:
    """Result of puremacro.dsge.estimate_dsge (and model-specific wrappers
    like puremacro.dsge.estimate_sw07).

    Attributes
    ----------
    draws : ndarray, shape (n_chains, n_draws, n_params)
        Post-burn-in MCMC draws.
    param_names : tuple of str, length n_params
        Parameter names in column order matching `draws`.
    log_posterior_trace : ndarray, shape (n_chains, n_draws)
        Log-posterior at each retained draw.
    accept_rates : tuple of float, length n_chains
        Per-chain acceptance rate over the retained draws.
    mode : dict[str, float]
        Posterior mode (parameter name → value).
    mode_hessian_inv : ndarray, shape (n_params, n_params)
        Inverse Hessian at the mode when scipy.optimize converges + Hessian is PD;
        otherwise falls back to diag(prior_stds**2). Either way, this is the
        proposal-cov foundation that random_walk_metropolis scales by c0**2.
    n_burn_in : int
        Burn-in iterations dropped (also used as proposal-scale adaptation window).
    data_n_obs : int
        Number of observations in the input dataset.
    seed : int
        Master RNG seed.
    model_name : str, default 'unknown'
        Identifier for the underlying DSGE model. ``estimate_sw07`` sets
        this to ``"SW07"``; ``estimate_dsge`` callers can pass whatever
        string they want. New in 0.53.0.
    """
    draws: np.ndarray
    param_names: Tuple[str, ...]
    log_posterior_trace: np.ndarray
    accept_rates: Tuple[float, ...]
    mode: dict
    mode_hessian_inv: np.ndarray
    n_burn_in: int
    data_n_obs: int
    seed: int
    model_name: str = "unknown"
    #: ``log p(y | th*) + log p(th*)`` at the reported mode. ``None`` on
    #: results produced before 2.6.0, which is why :meth:`log_mdd` says so
    #: rather than guessing. New in 2.6.0.
    log_post_mode: float | None = None

    def log_mdd(self, method: str = "laplace") -> float:
        """Log marginal likelihood ``log p(y)``.

        ``method="laplace"`` needs :attr:`log_post_mode` and a
        positive-definite :attr:`mode_hessian_inv`; ``method="harmonic"`` uses
        Geweke's modified harmonic mean over the draws and warns when its
        estimate is not stable across truncation levels. See
        :mod:`puremacro.dsge.marginal`.
        """
        from .marginal import harmonic_mean_mdd, laplace_mdd

        if method == "laplace":
            if self.log_post_mode is None:
                raise ValueError(
                    "log_mdd(method='laplace') needs log_post_mode, which this "
                    "result does not carry (it predates puremacro 2.6.0). "
                    "Re-run the estimation, or use method='harmonic'."
                )
            return laplace_mdd(self.log_post_mode, self.mode_hessian_inv)
        if method == "harmonic":
            return harmonic_mean_mdd(self.draws, self.log_posterior_trace).estimate
        raise ValueError(
            f"unknown method {method!r}; expected 'laplace' or 'harmonic'"
        )

    def summary(self) -> pd.DataFrame:
        """Per-parameter mean, std, 5%/50%/95% quantiles across all chains."""
        flat = self.draws.reshape(-1, len(self.param_names))
        return pd.DataFrame({
            "mean":  flat.mean(axis=0),
            "std":   flat.std(axis=0),
            "q5":    np.quantile(flat, 0.05, axis=0),
            "q50":   np.quantile(flat, 0.50, axis=0),
            "q95":   np.quantile(flat, 0.95, axis=0),
            "mode":  [self.mode[n] for n in self.param_names],
        }, index=list(self.param_names))

    def to_frame(self) -> pd.DataFrame:
        """Return summary statistics as a DataFrame."""
        return self.summary()

    def to_markdown(self, **kwargs) -> str:
        """Render summary table as Markdown."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Render summary table as LaTeX tabular."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Render summary table as Typst table."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.summary(), **kwargs)


# Backward-compatibility alias for code that imports SW07PosteriorResult.
# Resolves to the same class object; isinstance/pickle/type-hints continue
# to work. Drop in 1.0 if appropriate.
SW07PosteriorResult = DSGEPosteriorResult


@dataclass(frozen=True)
class NUTSResult:
    """Result of No-U-Turn Sampler (NUTS) Hamiltonian Monte Carlo estimation.

    Attributes
    ----------
    draws : ndarray, shape (n_chains, n_draws, n_params)
        Post-warmup MCMC draws.
    param_names : Tuple[str, ...]
        Names of parameters matching column order of draws.
    log_posterior_trace : ndarray, shape (n_chains, n_draws)
        Log-posterior at each retained draw.
    accept_rates : Tuple[float, ...]
        Mean acceptance statistic per chain.
    step_sizes : Tuple[float, ...]
        Adapted step size per chain.
    tree_depths : ndarray, shape (n_chains, n_draws)
        Tree depth reached at each step.
    divergences : ndarray, shape (n_chains, n_draws)
        Boolean divergence indicator mask.
    energy_trace : ndarray, shape (n_chains, n_draws)
        Hamiltonian energy at each retained draw.
    mass_matrix_diag : np.ndarray
        Adapted diagonal inverse mass matrix M^{-1} (or per-chain matrix).
    mode : dict | None
        Posterior mode dict (if mode optimization was performed).
    mode_hessian_inv : np.ndarray | None
        Inverse Hessian at mode (if computed).
    n_warmup : int
        Number of warmup iterations dropped.
    data_n_obs : int
        Number of observations in the data.
    seed : int
        Master random seed.
    model_name : str = "unknown"
        DSGE model name.
    log_post_mode : float | None = None
        Log-posterior at mode.
    warmup_draws : np.ndarray | None = None
        Warmup draws if recorded.
    """
    draws: np.ndarray
    param_names: Tuple[str, ...]
    log_posterior_trace: np.ndarray
    accept_rates: Tuple[float, ...]
    step_sizes: Tuple[float, ...]
    tree_depths: np.ndarray
    divergences: np.ndarray
    energy_trace: np.ndarray
    mass_matrix_diag: np.ndarray
    mode: dict | None = None
    mode_hessian_inv: np.ndarray | None = None
    n_warmup: int = 0
    data_n_obs: int = 0
    seed: int = 0
    model_name: str = "unknown"
    log_post_mode: float | None = None
    warmup_draws: np.ndarray | None = None

    @property
    def n_burn_in(self) -> int:
        """Alias for n_warmup matching DSGEPosteriorResult."""
        return self.n_warmup

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Summary dictionary of NUTS sampling diagnostics."""
        from puremacro.dsge.nuts import compute_ebfmi
        ebfmi_vals = tuple(compute_ebfmi(self.energy_trace[c]) for c in range(len(self.accept_rates)))
        return {
            "n_divergences": int(np.sum(self.divergences)),
            "divergence_rate": float(np.mean(self.divergences)),
            "mean_tree_depth": float(np.mean(self.tree_depths)),
            "max_tree_depth_hit_rate": float(np.mean(self.tree_depths >= 10)),
            "mean_accept_rate": float(np.mean(self.accept_rates)),
            "step_sizes": self.step_sizes,
            "ebfmi": ebfmi_vals,
        }

    def summary(self) -> pd.DataFrame:
        """Per-parameter mean, std, quantiles, split-R_hat, and bulk/tail ESS."""
        from puremacro.dsge.nuts import compute_split_rhat, compute_bulk_ess, compute_tail_ess

        flat = self.draws.reshape(-1, len(self.param_names))
        n_params = len(self.param_names)

        r_hats = [compute_split_rhat(self.draws[:, :, i]) for i in range(n_params)]
        ess_bulks = [compute_bulk_ess(self.draws[:, :, i]) for i in range(n_params)]
        ess_tails = [compute_tail_ess(self.draws[:, :, i]) for i in range(n_params)]

        data: dict[str, Any] = {
            "mean": flat.mean(axis=0),
            "std": flat.std(axis=0),
            "q5": np.quantile(flat, 0.05, axis=0),
            "q50": np.quantile(flat, 0.50, axis=0),
            "q95": np.quantile(flat, 0.95, axis=0),
            "r_hat": r_hats,
            "ess_bulk": ess_bulks,
            "ess_tail": ess_tails,
        }
        if self.mode is not None:
            data["mode"] = [self.mode.get(n, float("nan")) for n in self.param_names]

        return pd.DataFrame(data, index=list(self.param_names))

    def to_frame(self) -> pd.DataFrame:
        """Return summary statistics as a DataFrame."""
        return self.summary()

    def to_markdown(self, **kwargs) -> str:
        """Render summary table as Markdown."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Render summary table as LaTeX tabular."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Render summary table as Typst table."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.summary(), **kwargs)

    def log_mdd(self, method: str = "harmonic") -> float:
        """Compute log marginal data density log p(y)."""
        from .marginal import harmonic_mean_mdd, laplace_mdd
        if method == "laplace":
            if self.log_post_mode is None or self.mode_hessian_inv is None:
                raise ValueError("log_mdd(method='laplace') needs log_post_mode and mode_hessian_inv.")
            return laplace_mdd(self.log_post_mode, self.mode_hessian_inv)
        if method == "harmonic":
            return harmonic_mean_mdd(self.draws, self.log_posterior_trace).estimate
        raise ValueError(f"unknown method {method!r}; expected 'laplace' or 'harmonic'")

    def plot_trace(self, fig: Any = None, axes: Any = None, figsize: tuple[float, float] | None = None) -> tuple[Any, Any]:
        """Plot trace of parameter draws across chains with separate colors."""
        import matplotlib.pyplot as plt

        n_params = len(self.param_names)
        n_chains = self.draws.shape[0]
        n_plots = n_params + 1

        if axes is None:
            ncols = 2
            nrows = int(np.ceil(n_plots / ncols))
            if figsize is None:
                figsize = (10.0, 2.2 * nrows)
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        ax_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        for i, name in enumerate(self.param_names):
            if i < len(ax_flat):
                ax = ax_flat[i]
                for c in range(n_chains):
                    col = colors[c % len(colors)]
                    ax.plot(self.draws[c, :, i], color=col, alpha=0.75, lw=1.0, label=f"Chain {c+1}")
                ax.set_title(name, fontsize=10, fontweight="bold")
                ax.set_xlabel("Iteration")
                ax.grid(True, alpha=0.3)

        if n_params < len(ax_flat):
            ax = ax_flat[n_params]
            for c in range(n_chains):
                col = colors[c % len(colors)]
                ax.plot(self.log_posterior_trace[c], color=col, alpha=0.75, lw=1.0, label=f"Chain {c+1}")
            ax.set_title("Log-Posterior", fontsize=10, fontweight="bold")
            ax.set_xlabel("Iteration")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

        for k in range(n_plots, len(ax_flat)):
            ax_flat[k].axis("off")

        if fig is not None:
            fig.tight_layout()
        return fig, axes

    def plot_posterior(self, fig: Any = None, axes: Any = None, bins: int = 30, figsize: tuple[float, float] | None = None) -> tuple[Any, Any]:
        """Plot marginal posterior distributions for all parameters."""
        import matplotlib.pyplot as plt

        n_params = len(self.param_names)
        if axes is None:
            ncols = min(3, n_params)
            nrows = int(np.ceil(n_params / ncols))
            if figsize is None:
                figsize = (3.5 * ncols, 2.5 * nrows)
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        ax_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
        flat_draws = self.draws.reshape(-1, n_params)

        for i, name in enumerate(self.param_names):
            if i < len(ax_flat):
                ax = ax_flat[i]
                vals = flat_draws[:, i]
                ax.hist(vals, bins=bins, density=True, alpha=0.65, color="#1f77b4", edgecolor="white")
                if self.mode is not None and name in self.mode:
                    ax.axvline(self.mode[name], color="red", linestyle="--", lw=1.5, label="Mode")
                ax.set_title(name, fontsize=10, fontweight="bold")
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")
                ax.grid(True, alpha=0.3)

        for k in range(n_params, len(ax_flat)):
            ax_flat[k].axis("off")

        if fig is not None:
            fig.tight_layout()
        return fig, axes

    def plot_autocorr(self, fig: Any = None, axes: Any = None, max_lag: int = 40, figsize: tuple[float, float] | None = None) -> tuple[Any, Any]:
        """Plot autocorrelation function for all parameters across chains."""
        import matplotlib.pyplot as plt
        from puremacro.mcmc import autocorrelations

        n_params = len(self.param_names)
        n_chains = self.draws.shape[0]

        if axes is None:
            ncols = min(3, n_params)
            nrows = int(np.ceil(n_params / ncols))
            if figsize is None:
                figsize = (3.5 * ncols, 2.5 * nrows)
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        ax_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        for i, name in enumerate(self.param_names):
            if i < len(ax_flat):
                ax = ax_flat[i]
                for c in range(n_chains):
                    col = colors[c % len(colors)]
                    acf = autocorrelations(self.draws[c, :, i], max_lag=max_lag)
                    ax.plot(np.arange(len(acf)), acf, color=col, lw=1.2, alpha=0.8, label=f"Chain {c+1}")
                ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
                ax.set_title(name, fontsize=10, fontweight="bold")
                ax.set_xlabel("Lag")
                ax.set_ylabel("Autocorrelation")
                ax.set_ylim(-0.2, 1.05)
                ax.grid(True, alpha=0.3)

        for k in range(n_params, len(ax_flat)):
            ax_flat[k].axis("off")

        if fig is not None:
            fig.tight_layout()
        return fig, axes

    def energy_diagnostics(self, fig: Any = None, ax: Any = None, bins: int = 30) -> tuple[dict[str, Any], Any, Any]:
        """Betancourt (2016) Energy diagnostic and E-BFMI visualization."""
        import matplotlib.pyplot as plt
        from puremacro.dsge.nuts import compute_ebfmi

        E_flat = self.energy_trace.ravel()
        dE_flat = np.concatenate([np.diff(self.energy_trace[c]) for c in range(self.energy_trace.shape[0])])

        ebfmi_vals = tuple(compute_ebfmi(self.energy_trace[c]) for c in range(self.energy_trace.shape[0]))
        mean_ebfmi = float(np.mean(ebfmi_vals))

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4.5))

        E_std = (E_flat - np.mean(E_flat)) / np.std(E_flat)
        dE_std = (dE_flat - np.mean(dE_flat)) / np.std(dE_flat)

        ax.hist(E_std, bins=bins, density=True, alpha=0.5, color="#1f77b4", edgecolor="white", label="Marginal Energy $\\pi(E)$")
        ax.hist(dE_std, bins=bins, density=True, alpha=0.5, color="#2ca02c", edgecolor="white", label="Energy Transition $\\pi(\\Delta E)$")
        ax.set_title(f"HMC Energy Diagnostics (Mean E-BFMI = {mean_ebfmi:.3f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Standardized Energy")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        if fig is not None:
            fig.tight_layout()

        diag_dict = {
            "ebfmi_per_chain": ebfmi_vals,
            "mean_ebfmi": mean_ebfmi,
            "passed": mean_ebfmi >= 0.3,
        }
        return diag_dict, fig, ax


@dataclass(frozen=True)
class HANKResult:
    """Result of Heterogeneous-Agent New Keynesian (HANK) Sequence-Space solve.

    Attributes
    ----------
    steady_state : dict[str, float]
        Steady-state values of aggregate and microeconomic variables.
    transition_paths : pd.DataFrame
        Time series of variables across the transition horizon (Y, C, r, pi, etc.).
    jacobians : dict[str, np.ndarray]
        Sequence-space Jacobians (e.g. J_C_r, J_C_Y).
    asset_distribution : np.ndarray
        Stationary distribution over asset grid D^*(a).
    asset_grid : np.ndarray
        Asset grid points a.
    mpc_distribution : np.ndarray | None
        Marginal propensity to consume distribution across asset grid.
    shock_name : str
        Name of simulated exogenous shock.
    horizon : int
        Simulation horizon.
    model_name : str
        Model identifier.
    converged : bool
        Whether nonlinear transition / steady state converged.
    """
    steady_state: dict[str, float]
    transition_paths: pd.DataFrame
    jacobians: dict[str, np.ndarray]
    asset_distribution: np.ndarray
    asset_grid: np.ndarray
    mpc_distribution: np.ndarray | None = None
    liquid_asset_grid: np.ndarray | None = None
    joint_distribution: np.ndarray | None = None
    marginal_distribution_b: np.ndarray | None = None
    deposit_distribution: np.ndarray | None = None
    shock_name: str = "eps_m"
    horizon: int = 40
    model_name: str = "hank_sequence_space"
    converged: bool = True

    def summary(self) -> pd.DataFrame:
        """Summary table of peak and on-impact responses across variables."""
        records = []
        for col in self.transition_paths.columns:
            series = self.transition_paths[col].to_numpy()
            impact = float(series[0]) if len(series) > 0 else 0.0
            peak_idx = int(np.argmax(np.abs(series))) if len(series) > 0 else 0
            peak_val = float(series[peak_idx]) if len(series) > 0 else 0.0
            ss_val = float(self.steady_state.get(col, np.nan))
            records.append({
                "variable": col,
                "steady_state": ss_val,
                "impact_response": impact,
                "peak_response": peak_val,
                "peak_period": peak_idx,
            })
        return pd.DataFrame(records).set_index("variable")

    def to_frame(self) -> pd.DataFrame:
        """Return transition paths as DataFrame."""
        return self.transition_paths

    def to_markdown(self, **kwargs) -> str:
        """Render summary table as Markdown."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Render summary table as LaTeX tabular."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Render summary table as Typst table."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.summary(), **kwargs)

    def plot_transition(
        self,
        variables: Sequence[str] | None = None,
        fig: Any = None,
        axes: Any = None,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Any, Any]:
        """Plot transition dynamics (IRFs) across variables."""
        import matplotlib.pyplot as plt

        vars_to_plot = list(variables) if variables is not None else list(self.transition_paths.columns)
        n_vars = len(vars_to_plot)

        if axes is None:
            ncols = min(3, n_vars)
            nrows = int(np.ceil(n_vars / ncols))
            if figsize is None:
                figsize = (3.5 * ncols, 2.5 * nrows)
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        ax_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
        t_grid = np.arange(len(self.transition_paths))

        for i, var in enumerate(vars_to_plot):
            if i < len(ax_flat):
                ax = ax_flat[i]
                ax.plot(t_grid, self.transition_paths[var], color="#1f77b4", lw=2.0)
                ax.axhline(0.0, color="gray", linestyle="--", alpha=0.6)
                ax.set_title(var, fontsize=11, fontweight="bold")
                ax.set_xlabel("Periods")
                ax.grid(True, alpha=0.3)

        for k in range(n_vars, len(ax_flat)):
            ax_flat[k].axis("off")

        if fig is not None:
            fig.tight_layout()
        return fig, axes

    def plot_distribution(
        self,
        fig: Any = None,
        axes: Any = None,
        figsize: tuple[float, float] = (10.0, 4.0),
    ) -> tuple[Any, Any]:
        """Plot stationary asset distribution and MPC distribution (or 2D joint distribution)."""
        import matplotlib.pyplot as plt

        if self.joint_distribution is not None and self.liquid_asset_grid is not None:
            if axes is None:
                fig, axes = plt.subplots(1, 2, figsize=figsize)
            ax_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

            # Left panel: 2D joint distribution contour
            A, B = np.meshgrid(self.asset_grid, self.liquid_asset_grid, indexing="ij")
            cp = ax_flat[0].contourf(A, B, self.joint_distribution, cmap="viridis")
            if fig is not None:
                fig.colorbar(cp, ax=ax_flat[0], fraction=0.046, pad=0.04)
            ax_flat[0].set_title(r"Joint Wealth Distribution $\mathcal{D}^*(a, b)$", fontsize=11, fontweight="bold")
            ax_flat[0].set_xlabel("Illiquid Assets $a$")
            ax_flat[0].set_ylabel("Liquid Assets $b$")
            ax_flat[0].grid(True, alpha=0.3)

            # Right panel: Marginals of illiquid and liquid assets
            ax_flat[1].plot(self.asset_grid, self.asset_distribution, color="#1f77b4", lw=2.0, label=r"Illiquid $a$")
            if self.marginal_distribution_b is not None:
                ax_flat[1].plot(self.liquid_asset_grid, self.marginal_distribution_b, color="#ff7f0e", lw=2.0, label=r"Liquid $b$")
            ax_flat[1].set_title("Marginal Wealth Distributions", fontsize=11, fontweight="bold")
            ax_flat[1].set_xlabel("Assets")
            ax_flat[1].set_ylabel("Density")
            ax_flat[1].legend(loc="upper right", frameon=False)
            ax_flat[1].grid(True, alpha=0.3)

            if fig is not None:
                fig.tight_layout()
            return fig, axes

        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

        ax_flat[0].plot(self.asset_grid, self.asset_distribution, color="#1f77b4", lw=2.0)
        ax_flat[0].set_title("Stationary Wealth Distribution $\\mathcal{D}^*(a)$", fontsize=11, fontweight="bold")
        ax_flat[0].set_xlabel("Assets $a$")
        ax_flat[0].set_ylabel("Density")
        ax_flat[0].grid(True, alpha=0.3)

        if self.mpc_distribution is not None:
            ax_flat[1].plot(self.asset_grid, self.mpc_distribution, color="#d62728", lw=2.0)
            ax_flat[1].set_title("Marginal Propensity to Consume $MPC(a)$", fontsize=11, fontweight="bold")
            ax_flat[1].set_xlabel("Assets $a$")
            ax_flat[1].set_ylabel("MPC")
            ax_flat[1].grid(True, alpha=0.3)
        else:
            cdf = np.cumsum(self.asset_distribution) / np.sum(self.asset_distribution)
            ax_flat[1].plot(self.asset_grid, cdf, color="#2ca02c", lw=2.0)
            ax_flat[1].set_title("Cumulative Wealth Distribution", fontsize=11, fontweight="bold")
            ax_flat[1].set_xlabel("Assets $a$")
            ax_flat[1].set_ylabel("CDF")
            ax_flat[1].grid(True, alpha=0.3)

        if fig is not None:
            fig.tight_layout()
        return fig, axes


@dataclass(frozen=True)
class FertilitySolution:
    """Linear solution of the fertility DSGE around its BGP.

    Attributes
    ----------
    ss : dict[str, float]
        Steady-state values keyed by variable name (matches VAR_NAMES).
    params : dict[str, float]
        All parameters used in the solve (structural + calibration +
        shock-process).
    G : ndarray, shape (n_states, n_states)
        State transition (state at t given state at t-1, no shock).
    N : ndarray, shape (n_states, n_shocks)
        Shock impact on states.
    F : ndarray, shape (n_controls, n_states)
        Control policy on the LAGGED state: y_t = F x_{t-1} + L eps_t, which
        is the partition `solve_fertility` actually builds. This entry used to
        read "control at t given state at t", contradicting the solver, and an
        IRF loop written against that wrong reading put every control one
        period early.
    L : ndarray, shape (n_controls, n_shocks)
        Control response to contemporaneous shock.
    klein_solution : KleinSolution or None
        Raw QZ output for debugging.
    var_names : tuple of str
        All 12 endogenous variable names (states first, then controls).
    shock_names : tuple of str
        Shock names (ea, ep, en).

    Notes
    -----
    The first n_states entries of var_names are the predetermined
    variables (rows of G/N); the remaining are controls (rows of F/L).
    """

    ss: dict
    params: dict
    G: np.ndarray
    N: np.ndarray
    F: np.ndarray
    L: np.ndarray
    klein_solution: object
    var_names: tuple
    shock_names: tuple

    def irf(self, shock, horizon: int = 20) -> pd.DataFrame:
        """Impulse response to a 1-SD shock. See fertility_adj_costs.solve_fertility docstring."""
        from puremacro.dsge.fertility_adj_costs import _compute_irf
        return _compute_irf(self, shock, horizon)

    def fevd(self, horizon: int = 20) -> pd.DataFrame:
        """Forecast-error variance decomposition."""
        from puremacro.dsge.fertility_adj_costs import _compute_fevd
        return _compute_fevd(self, horizon)


@dataclass(frozen=True)
class DynareDR:
    """Decision rule representation matching Dynare's oo_.dr structure.

    First-order approximation around steady state:
        y_t = ys + ghx * (x_{t-1} - xs) + ghu * u_t

    Attributes
    ----------
    ghx : pd.DataFrame
        (n_vars x n_states) matrix of policy derivatives with respect to lagged states.
    ghu : pd.DataFrame
        (n_vars x n_shocks) matrix of policy derivatives with respect to contemporaneous shocks.
    ys : pd.Series
        Steady-state values for all endogenous variables.
    state_variables : tuple[str, ...]
        Names of predetermined state variables.
    variable_names : tuple[str, ...]
        Names of all endogenous variables in model order.
    shock_names : tuple[str, ...]
        Names of structural shocks.
    """

    ghx: pd.DataFrame
    ghu: pd.DataFrame
    ys: pd.Series
    state_variables: tuple[str, ...]
    variable_names: tuple[str, ...]
    shock_names: tuple[str, ...]

    def __getitem__(self, key: str):
        """Allow dict-like access matching Dynare MATLAB struct conventions."""
        if not isinstance(key, str):
            raise KeyError(key)
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"DynareDR has no field {key!r}")

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (hasattr(self, key) or key in getattr(self, "extra", {}))

    def to_frame(self) -> pd.DataFrame:
        """Return transition and policy functions matching Dynare's output layout.

        Rows are [Constant, state(-1)..., shocks...], columns are endogenous variables.
        """
        rows = ["Constant"] + [f"{s}(-1)" for s in self.state_variables] + list(self.shock_names)
        df = pd.DataFrame(index=rows, columns=list(self.variable_names), dtype=float)

        # Constant row
        for v in self.variable_names:
            df.loc["Constant", v] = self.ys.get(v, 0.0)

        # Lagged states rows
        for s in self.state_variables:
            row_lbl = f"{s}(-1)"
            for v in self.variable_names:
                df.loc[row_lbl, v] = self.ghx.loc[v, s]

        # Shocks rows
        for e in self.shock_names:
            for v in self.variable_names:
                df.loc[e, v] = self.ghu.loc[v, e]

        return df

    def summary(self) -> str:
        """Render Dynare-style 'POLICY AND TRANSITION FUNCTIONS' table."""
        df = self.to_frame()
        lines = [
            "POLICY AND TRANSITION FUNCTIONS (Dynare Format)",
            "=" * 72,
            df.round(6).to_string(),
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_markdown(self, **kwargs) -> str:
        """Export decision rules to Markdown table."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export decision rules to LaTeX tabular format."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Export decision rules to Typst table format."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)


@dataclass(frozen=True)
class Dynare2ndDR:
    """Second-order decision rule representation matching Dynare's oo_.dr structure.

    Second-order approximation around steady state:
        y_t = ys + 0.5 * ghs2 * sigma^2 + ghx * (x_{t-1} - xs) + ghu * u_t
              + 0.5 * ghxx * ((x_{t-1} - xs) ⊗ (x_{t-1} - xs))
              + ghxu * ((x_{t-1} - xs) ⊗ u_t)
              + 0.5 * ghuu * (u_t ⊗ u_t)

    Attributes
    ----------
    ghx : pd.DataFrame
        (n_vars x n_states) matrix of first-order state policy derivatives.
    ghu : pd.DataFrame
        (n_vars x n_shocks) matrix of first-order shock policy derivatives.
    ghxx : pd.DataFrame
        (n_vars x n_states^2) matrix of second-order state policy derivatives.
    ghxu : pd.DataFrame
        (n_vars x (n_states * n_shocks)) matrix of cross state-shock derivatives.
    ghuu : pd.DataFrame
        (n_vars x n_shocks^2) matrix of second-order shock derivatives.
    ghs2 : pd.Series
        (n_vars,) vector of volatility / risk correction terms.
    ys : pd.Series
        Steady-state values for all endogenous variables.
    state_variables : tuple[str, ...]
        Names of predetermined state variables.
    variable_names : tuple[str, ...]
        Names of all endogenous variables in model order.
    shock_names : tuple[str, ...]
        Names of structural shocks.
    """

    ghx: pd.DataFrame
    ghu: pd.DataFrame
    ghxx: pd.DataFrame
    ghxu: pd.DataFrame
    ghuu: pd.DataFrame
    ghs2: pd.Series
    ys: pd.Series
    state_variables: tuple[str, ...]
    variable_names: tuple[str, ...]
    shock_names: tuple[str, ...]

    def __getitem__(self, key: str):
        """Allow dict-like access matching Dynare MATLAB struct conventions."""
        if not isinstance(key, str):
            raise KeyError(key)
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Dynare2ndDR has no field {key!r}")

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (hasattr(self, key) or key in getattr(self, "extra", {}))

    def to_frame(self) -> pd.DataFrame:
        """Return transition and policy functions matching Dynare layout."""
        rows = ["Constant", "0.5 * ghs2"]
        rows += [f"{s}(-1)" for s in self.state_variables]
        rows += list(self.shock_names)
        df = pd.DataFrame(index=rows, columns=list(self.variable_names), dtype=float)

        for v in self.variable_names:
            df.loc["Constant", v] = self.ys.get(v, 0.0)
            df.loc["0.5 * ghs2", v] = 0.5 * self.ghs2.get(v, 0.0)
            for s in self.state_variables:
                df.loc[f"{s}(-1)", v] = self.ghx.loc[v, s]
            for e in self.shock_names:
                df.loc[e, v] = self.ghu.loc[v, e]

        return df

    def summary(self) -> str:
        """Render Dynare-style 2nd-order policy functions table."""
        df = self.to_frame()
        lines = [
            "SECOND-ORDER POLICY AND TRANSITION FUNCTIONS (Dynare Format)",
            "=" * 72,
            df.round(6).to_string(),
            "-" * 72,
            f"State cross-terms (ghxx) shape : {self.ghxx.shape}",
            f"State-shock terms (ghxu) shape : {self.ghxu.shape}",
            f"Shock cross-terms (ghuu) shape : {self.ghuu.shape}",
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_markdown(self, **kwargs) -> str:
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)


@dataclass(frozen=True)
class TheoreticalMomentsResult:
    """Analytical theoretical moments matching Dynare's stoch_simul.

    Attributes
    ----------
    moments : pd.DataFrame
        Table of [Mean, Std.Dev., Variance] for each endogenous variable.
    covariance : pd.DataFrame
        Unconditional covariance matrix (n_vars x n_vars).
    correlation : pd.DataFrame
        Unconditional correlation matrix (n_vars x n_vars).
    autocorr : pd.DataFrame
        Theoretical autocorrelation coefficients for lags 1 to n_lags.
    fevd : pd.DataFrame
        Forecast error variance decomposition percentage shares across horizons.

    Notes
    -----
    ``moments`` / ``covariance`` / ``correlation`` / ``autocorr`` are stated in
    the timing the model reports its variables in -- the same timing as
    ``irf()`` and ``simulate()``. ``fevd`` as filled in by
    :meth:`~puremacro.dsge.LinearModel.theoretical_moments` is built from the
    Dynare-timed ``ghx`` / ``ghu`` decision rules, so for a **Klein-timed**
    model (one built with :func:`~puremacro.dsge.build`) the state rows of
    ``fevd`` are dated one period later than the same rows of ``covariance``:
    ``fevd`` reports the state at the *end* of the period, the moments report
    it at the start. Control rows agree in both.

    :func:`~puremacro.dsge.compute_fevd` (also reachable as
    ``LinearModel.fevd_result()``) does not have this offset -- it decomposes
    the variables in the timing they are reported in, so a predetermined state
    correctly has no one-step forecast error. Prefer it when the decomposition
    has to line up with the covariance block above.
    """

    moments: pd.DataFrame
    covariance: pd.DataFrame
    correlation: pd.DataFrame | None
    autocorr: pd.DataFrame
    fevd: pd.DataFrame
    autocorr_matrices: list[pd.DataFrame] = field(default_factory=list)

    def autocorr_matrix(self, lag: int = 1) -> pd.DataFrame:
        """Return the N x N cross-variable autocorrelation matrix at the specified lag."""
        if lag < 1 or lag > len(self.autocorr_matrices):
            raise IndexError(f"Lag {lag} out of range (1..{len(self.autocorr_matrices)})")
        return self.autocorr_matrices[lag - 1]

    @property
    def autocorrelation_matrices(self) -> list[pd.DataFrame]:
        return self.autocorr_matrices

    @property
    def contemporaneous_correlation(self) -> pd.DataFrame | None:
        return self.correlation

    def summary(self) -> str:
        """Render complete Dynare-style theoretical moments report."""
        lines = [
            "THEORETICAL MOMENTS (Dynare stoch_simul)",
            "=" * 72,
            self.moments.round(6).to_string(),
        ]
        if self.correlation is not None and not self.correlation.empty:
            lines += [
                "",
                "MATRIX OF CORRELATIONS",
                "-" * 72,
                self.correlation.round(4).to_string(),
            ]
        lines += [
            "",
            "COEFFICIENTS OF AUTOCORRELATION",
            "-" * 72,
            self.autocorr.round(4).to_string(),
            "",
            "VARIANCE DECOMPOSITION (in percent)",
            "-" * 72,
            self.fevd.round(2).to_string(),
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Return primary theoretical moments table."""
        return self.moments.copy()

    def to_markdown(self, **kwargs) -> str:
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.moments, **kwargs)

    def to_latex(self, **kwargs) -> str:
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.moments, **kwargs)

    def to_typst(self, **kwargs) -> str:
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.moments, **kwargs)


@dataclass(frozen=True)
class StochSimulResult:
    """Consolidated result of Dynare stoch_simul execution.

    Attributes
    ----------
    dr : DynareDR | Dynare2ndDR
        First- or second-order decision rule structure (oo_.dr).
    theoretical_moments : TheoreticalMomentsResult | None
        Analytical unconditional moments, correlations, autocorrelations, and FEVD.
    simulated_moments : pd.DataFrame | None
        Sample moments if simulation with periods > 0 was requested.
    irfs : dict[str, pd.Series]
        Dictionary of impulse response functions keyed by '{var}_{shock}'.
    order : int
        Approximation order (1 or 2).
    variable_names : tuple[str, ...]
        Names of all endogenous variables.
    shock_names : tuple[str, ...]
        Names of structural shocks.
    _sim_corr : pd.DataFrame | None
        Simulated contemporaneous correlation matrix when periods > 0.
    """

    dr: DynareDR | Dynare2ndDR | Any
    theoretical_moments: TheoreticalMomentsResult | None
    simulated_moments: pd.DataFrame | None
    irfs: dict[str, pd.Series]
    order: int
    variable_names: tuple[str, ...]
    shock_names: tuple[str, ...]
    _sim_corr: pd.DataFrame | None = None

    @property
    def contemporaneous_correlation(self) -> pd.DataFrame | None:
        if self.simulated_moments is not None and self._sim_corr is not None:
            return self._sim_corr
        if self.theoretical_moments is not None:
            return self.theoretical_moments.correlation
        return None

    @property
    def autocorr_matrices(self) -> list[pd.DataFrame]:
        if self.theoretical_moments is not None:
            return self.theoretical_moments.autocorr_matrices
        return []

    @property
    def autocorrelation_matrices(self) -> list[pd.DataFrame]:
        return self.autocorr_matrices

    def autocorr_matrix(self, lag: int = 1) -> pd.DataFrame:
        if self.theoretical_moments is not None:
            return self.theoretical_moments.autocorr_matrix(lag)
        raise AttributeError("No theoretical autocorrelation matrices available")

    @property
    def mc_se(self) -> pd.Series | None:
        if self.simulated_moments is not None and "MC Std.Err." in self.simulated_moments.columns:
            return self.simulated_moments["MC Std.Err."]
        return None

    def __getitem__(self, key: str):
        """Allow subscript access matching Dynare struct conventions."""
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.irfs:
            return self.irfs[key]
        raise KeyError(f"StochSimulResult has no attribute or IRF series {key!r}")

    def to_frame(self, shock: str | None = None) -> pd.DataFrame:
        """Return IRFs as a DataFrame.

        If shock is provided, returns (H+1 x n_vars) for that shock.
        Otherwise, returns a wide DataFrame of all '{var}_{shock}' IRF paths.
        """
        if shock is not None:
            cols = {
                v: self.irfs[f"{v}_{shock}"]
                for v in self.variable_names
                if f"{v}_{shock}" in self.irfs
            }
            return pd.DataFrame(cols)
        return pd.DataFrame(self.irfs)

    def summary(self) -> str:
        """Render complete consolidated Dynare stoch_simul report."""
        lines = [
            f"DYNARE STOCH_SIMUL REPORT (Order {self.order})",
            "=" * 72,
            f"Endogenous variables : {len(self.variable_names)}",
            f"Exogenous shocks     : {len(self.shock_names)}",
            "-" * 72,
            self.dr.summary(),
        ]
        if self.theoretical_moments is not None:
            lines += [
                "",
                self.theoretical_moments.summary(),
            ]
        if self.simulated_moments is not None:
            lines += [
                "",
                "MOMENTS OF SIMULATED VARIABLES",
                "-" * 72,
                self.simulated_moments.round(6).to_string(),
                "=" * 72,
            ]
        return "\n".join(lines)

    def to_markdown(self, **kwargs) -> str:
        """Export primary theoretical moments to Markdown."""
        from puremacro.reports import _df_to_markdown

        df = (
            self.theoretical_moments.to_frame()
            if self.theoretical_moments is not None
            else (self.simulated_moments if self.simulated_moments is not None else pd.DataFrame())
        )
        return _df_to_markdown(df, **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export primary theoretical moments to LaTeX."""
        from puremacro.reports import _df_to_latex

        df = (
            self.theoretical_moments.to_frame()
            if self.theoretical_moments is not None
            else (self.simulated_moments if self.simulated_moments is not None else pd.DataFrame())
        )
        return _df_to_latex(df, **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Export primary theoretical moments to Typst."""
        from puremacro.reports import _df_to_typst

        df = (
            self.theoretical_moments.to_frame()
            if self.theoretical_moments is not None
            else (self.simulated_moments if self.simulated_moments is not None else pd.DataFrame())
        )
        return _df_to_typst(df, **kwargs)

    def plot(
        self,
        variables: Sequence[str] | None = None,
        shocks: Sequence[str] | None = None,
        *,
        style: str = "publication",
        figsize: tuple[float, float] | None = None,
    ):
        """Plot impulse response functions (IRFs).

        Parameters
        ----------
        variables : Sequence[str], optional
            Variables to include. Defaults to first 6 variables.
        shocks : Sequence[str], optional
            Structural shocks to plot. Defaults to first shock.
        style : str, default 'publication'
            Plot styling theme ('publication', 'dark', 'grayscale').
        figsize : tuple[float, float], optional
            Figure dimensions (width, height).

        Returns
        -------
        matplotlib.figure.Figure | None
        """
        import matplotlib.pyplot as plt
        from puremacro.plot import _palette

        if not self.irfs:
            return None

        sel_shocks = list(shocks) if shocks is not None else list(self.shock_names[:1])
        if not sel_shocks and self.shock_names:
            sel_shocks = [self.shock_names[0]]

        sel_vars = list(variables) if variables is not None else list(self.variable_names[:6])
        if not sel_vars and self.variable_names:
            sel_vars = list(self.variable_names)

        pairs = [
            (v, s)
            for s in sel_shocks
            for v in sel_vars
            if f"{v}_{s}" in self.irfs
        ]
        if not pairs:
            return None

        n_plots = len(pairs)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        if figsize is None:
            figsize = (3.8 * n_cols, 2.8 * n_rows)

        if style == "grayscale":
            colors = _palette(max(1, len(pairs)))
        else:
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"] * (len(pairs) // 6 + 1)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        ax_flat = axes.flatten()

        for idx, (v, s) in enumerate(pairs):
            ax = ax_flat[idx]
            series = self.irfs[f"{v}_{s}"]
            h = np.arange(len(series))
            col = colors[idx % len(colors)]
            ax.plot(h, series.values, color=col, lw=1.8, label=f"{v} ({s})")
            ax.axhline(0, color="gray", linestyle="--", lw=0.8, alpha=0.7)
            ax.set_title(f"{v} to {s}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Horizon (periods)", fontsize=8)
            ax.set_ylabel("Dev from SS", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.5)

        for idx in range(len(pairs), len(ax_flat)):
            ax_flat[idx].set_visible(False)

        fig.tight_layout()
        return fig


from .perfect_foresight import PerfectForesightResult
from .dsge_var import DSGEVARResult
from .news import NewsIRFResult, NewsDecompositionResult

__all__ = [
    "DSGEPosteriorResult",
    "SW07PosteriorResult",
    "NUTSResult",
    "HANKResult",
    "FertilitySolution",
    "DynareDR",
    "Dynare2ndDR",
    "TheoreticalMomentsResult",
    "StochSimulResult",
    "PerfectForesightResult",
    "ExtendedPathResult",
    "SmootherResult",
    "DSGEForecastResult",
    "DSGEVARResult",
    "NewsIRFResult",
    "NewsDecompositionResult",
    "ModeCheckResult",
    "DiagnosticFinding",
    "EigenvalueTable",
    "ModelDiagnosticsResult",
    "IdentificationResult",
    "OSRResult",
    "PolicyResult",
    "DiscretionaryPolicyResult",
    "ConditionalForecastResult",
    "ShockDecompositionResult",
    "BayesianIRFResult",
    "PriorPredictiveResult",
    "ModelParityResult",
    "ParityDashboardResult",
]


@dataclass(frozen=True)
class SmootherResult:
    """Kalman-smoother output at calibrated parameters (Dynare ``calib_smoother``).

    Attributes
    ----------
    states : pandas.DataFrame
        Smoothed model states, ``(T, n_states)``, in deviation units.
    shocks : pandas.DataFrame
        Smoothed structural innovations ``E[u_t | y_{1:T}]``, ``(T, n_shocks)``.
        These are read straight off the smoothed state: the filter carries
        ``alpha_t = [x_t; u_t]``, so the innovations are its last ``n_e`` rows
        and no separate disturbance smoother is involved.
    smoothed_obs : pandas.DataFrame
        Fitted observables, ``d + Z a_smooth``, with any declared observation
        trend added back. Equal to the data to machine precision when there is
        no measurement error.
    filtered_states : pandas.DataFrame
        One-sided ``E[x_t | y_{1:t}]``, ``(T, n_states)``.
    loglik : float
        Log-likelihood of the data under the calibrated parameters.
    varobs : tuple of str
        Observables, in data-column order.
    """

    states: pd.DataFrame
    shocks: pd.DataFrame
    smoothed_obs: pd.DataFrame
    filtered_states: pd.DataFrame
    loglik: float
    varobs: Tuple[str, ...]
    _model: Any = None
    _data: Any = None

    def shock_decomposition(self, **kwargs):
        """Historical shock decomposition for the same model and data.

        Delegates to :func:`~puremacro.dsge.compute_shock_decomposition`, which
        runs its own smoother. The two routes are pinned against each other in
        ``tests/test_dsge/test_smoother.py`` rather than assumed to agree.
        """
        if self._model is None or self._data is None:
            raise ValueError(
                "SmootherResult.shock_decomposition() needs the model and data "
                "this result was built from; it was constructed without them."
            )
        from .decomposition import compute_shock_decomposition

        return compute_shock_decomposition(self._model, self._data, **kwargs)

    def to_frame(self) -> pd.DataFrame:
        """Per-shock summary of the smoothed innovations."""
        s = self.shocks
        return pd.DataFrame(
            {
                "mean": s.mean(),
                "std": s.std(ddof=0),
                "min": s.min(),
                "max": s.max(),
            },
            index=list(s.columns),
        )

    def summary(self) -> str:
        lines = [
            "KALMAN SMOOTHER (calibrated parameters)",
            "=" * 60,
            f"observables : {', '.join(self.varobs)}",
            f"periods     : {len(self.smoothed_obs)}",
            f"log-likelihood: {self.loglik:.6f}",
            "",
            "SMOOTHED STRUCTURAL SHOCKS",
            "-" * 60,
            self.to_frame().round(6).to_string(),
        ]
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        """Smoothed structural shocks, one panel per shock."""
        import matplotlib.pyplot as plt

        cols = list(self.shocks.columns)
        if ax is None:
            _, ax = plt.subplots(len(cols), 1, sharex=True,
                                 figsize=(8, 2.0 * len(cols)), squeeze=False)
            ax = ax.ravel()
        ax = np.atleast_1d(ax)
        for a, c in zip(ax, cols):
            a.plot(self.shocks.index, self.shocks[c], **kwargs)
            a.axhline(0.0, color="0.6", lw=0.8)
            a.set_ylabel(c)
        ax[-1].set_xlabel("period")
        return ax

    def to_markdown(self, **kwargs) -> str:
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)


@dataclass(frozen=True)
class DSGEForecastResult:
    """Unconditional forecast from a solved model.

    Attributes
    ----------
    mean, lower, upper : pandas.DataFrame
        ``(horizon, n_vars)``, indexed by horizon ``1..horizon``. The band is
        ``mean +/- z_{(1+ci)/2} * sd``, with ``sd`` from the forecast-error
        covariance ``Z P_h Z' + H`` — parameter uncertainty is **not** in it.
    ci : float
        Nominal coverage of the band.
    horizon : int
        Number of periods forecast.
    """

    mean: pd.DataFrame
    lower: pd.DataFrame
    upper: pd.DataFrame
    ci: float
    horizon: int

    def to_frame(self) -> pd.DataFrame:
        out = {}
        for c in self.mean.columns:
            out[c] = self.mean[c]
            out[f"{c}_lower"] = self.lower[c]
            out[f"{c}_upper"] = self.upper[c]
        return pd.DataFrame(out, index=self.mean.index)

    def summary(self) -> str:
        pct = int(round(100 * self.ci))
        lines = [
            f"DSGE FORECAST ({self.horizon} periods, {pct}% band)",
            "=" * 60,
            "The band reflects shock uncertainty only: parameters are held "
            "fixed at the values the model was solved with.",
            "",
            self.to_frame().round(6).to_string(),
        ]
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        cols = list(self.mean.columns)
        if ax is None:
            _, ax = plt.subplots(len(cols), 1, sharex=True,
                                 figsize=(8, 2.4 * len(cols)), squeeze=False)
            ax = ax.ravel()
        ax = np.atleast_1d(ax)
        for a, c in zip(ax, cols):
            a.plot(self.mean.index, self.mean[c], **kwargs)
            a.fill_between(self.mean.index, self.lower[c], self.upper[c], alpha=0.25)
            a.set_ylabel(c)
        ax[-1].set_xlabel("horizon")
        return ax

    def to_markdown(self, **kwargs) -> str:
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)


@dataclass(frozen=True)
class ModeCheckResult:
    """One-parameter slices of an objective through a candidate mode.

    Attributes
    ----------
    slices : dict[str, pandas.DataFrame]
        Per parameter, a frame with ``value`` and ``objective`` columns: the
        objective with every other parameter held at the mode.
    peaks_at_mode : dict[str, bool]
        Whether the slice attains its minimum at the mode. A ``False`` here is
        the most common sign that the reported mode is not one.
    mode : dict[str, float]
        The point the slices pass through.
    objective_at_mode : float
        The objective there. ``find_mode`` minimises, so this is a negative log
        posterior, not a log posterior.
    """

    slices: dict
    peaks_at_mode: dict
    mode: dict
    objective_at_mode: float

    @property
    def failures(self) -> Tuple[str, ...]:
        """Parameters whose slice bottoms out away from the mode."""
        return tuple(n for n, ok in self.peaks_at_mode.items() if not ok)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for name, sl in self.slices.items():
            j = int(sl["objective"].idxmin())
            rows.append({
                "mode": self.mode[name],
                "best_on_slice": float(sl["value"].iloc[j]),
                "objective_gain": float(self.objective_at_mode - sl["objective"].iloc[j]),
                "peaks_at_mode": self.peaks_at_mode[name],
            })
        return pd.DataFrame(rows, index=list(self.slices))

    def summary(self) -> str:
        bad = self.failures
        lines = [
            "MODE CHECK (one-parameter slices)",
            "=" * 60,
            f"objective at the mode: {self.objective_at_mode:.6f}",
            "",
            self.to_frame().round(6).to_string(),
            "",
        ]
        if bad:
            lines.append(
                f"{len(bad)} parameter(s) improve away from the reported mode: "
                f"{', '.join(bad)}. The point supplied is not a mode in those "
                "directions — re-run the search from the better point."
            )
        else:
            lines.append(
                "Every slice bottoms out at the reported mode. That is "
                "necessary, not sufficient: these are one-parameter slices, "
                "so they say nothing about directions in between."
            )
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        names = list(self.slices)
        if ax is None:
            ncol = min(3, len(names))
            nrow = int(np.ceil(len(names) / ncol))
            _, ax = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.6 * nrow),
                                 squeeze=False)
            ax = ax.ravel()
        ax = np.atleast_1d(ax)
        for a, name in zip(ax, names):
            sl = self.slices[name]
            a.plot(sl["value"], sl["objective"], **kwargs)
            a.axvline(self.mode[name], color="0.5", ls="--", lw=0.9)
            a.set_title(name + ("" if self.peaks_at_mode[name] else "  (!)"))
        return ax

    def to_markdown(self, **kwargs) -> str:
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)


@dataclass(frozen=True)
class DiagnosticFinding:
    """A single diagnostic issue or verification confirmation."""

    category: str  # "steady_state", "static_rank", "incidence", "pencil", "unit_root", "stochastic_singularity"
    severity: str  # "error", "warning", "info"
    message: str
    details: Any = None

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] ({self.category}) {self.message}"


@dataclass(frozen=True)
class EigenvalueTable:
    """Frozen dataclass representing the DSGE eigenvalue spectrum and determinacy.

    Attributes
    ----------
    eigenvalues : np.ndarray
        Complex 1D array of generalized eigenvalues, sorted by modulus ascending.
    modulus : np.ndarray
        Float 1D array of eigenvalue moduli |lambda_i|.
    real : np.ndarray
        Float 1D array of real components Re(lambda_i).
    imag : np.ndarray
        Float 1D array of imaginary components Im(lambda_i).
    is_explosive : np.ndarray
        Boolean 1D array indicating whether |lambda_i| >= qz_criterium.
    is_unit_root : np.ndarray
        Boolean 1D array indicating whether ||lambda_i| - 1.0| <= 1e-5.
    n_explosive : int
        Total number of explosive eigenvalues.
    n_forward : int
        Total number of forward-looking (control) variables.
    is_determinate : bool
        True if n_explosive == n_forward and pencil is regular.
    bk_status : str
        Human-readable Blanchard-Kahn verdict message.
    loadings : dict[int, dict[str, float]] | None
        Mapping from offending root index to top variable loadings (|v_ij|),
        populated only when is_determinate is False.
    variables : tuple[str, ...]
        Names of all variables in the system.
    qz_criterium : float, default 1.0 + 1e-6
        Threshold modulus above which an eigenvalue is classified as explosive.
    """

    eigenvalues: np.ndarray
    modulus: np.ndarray
    real: np.ndarray
    imag: np.ndarray
    is_explosive: np.ndarray
    is_unit_root: np.ndarray
    n_explosive: int
    n_forward: int
    is_determinate: bool
    bk_status: str
    loadings: dict[int, dict[str, float]] | None
    variables: tuple[str, ...]
    qz_criterium: float = 1.0 + 1e-6

    @property
    def bk_satisfied(self) -> bool:
        """Alias for is_determinate."""
        return self.is_determinate

    @property
    def offending_loadings(self) -> dict[int, dict[str, float]] | None:
        """Alias for loadings."""
        return self.loadings

    @property
    def summary_message(self) -> str:
        """Alias for bk_status."""
        return self.bk_status

    @property
    def n_stable(self) -> int:
        """Total number of stable eigenvalues (|lambda| < qz_criterium)."""
        return int(np.sum(~self.is_explosive))

    def to_frame(self) -> pd.DataFrame:
        """Return eigenvalue details as a DataFrame."""
        idx = [f"root_{i+1}" for i in range(len(self.eigenvalues))]
        return pd.DataFrame(
            {
                "modulus": self.modulus,
                "real": self.real,
                "imag": self.imag,
                "is_explosive": self.is_explosive,
                "is_unit_root": self.is_unit_root,
            },
            index=idx,
        )

    def summary(self) -> str:
        lines = [
            f"EIGENVALUES & BLANCHARD-KAHN DIAGNOSTICS (qz_criterium={self.qz_criterium:.6f})",
            "=" * 72,
            f"Status       : {self.bk_status}",
            f"Determinacy  : {'UNIQUE STABLE EQUILIBRIUM' if self.is_determinate else 'DETERMINACY FAILED'}",
            f"Forward vars : {self.n_forward}",
            f"Explosive    : {self.n_explosive}",
            f"Stable       : {self.n_stable}",
            f"Unit roots   : {int(np.sum(self.is_unit_root))}",
            "",
            "EIGENVALUE SPECTRUM",
            "-" * 72,
            self.to_frame().round(6).to_string(),
        ]
        if self.loadings:
            lines.extend([
                "",
                "OFFENDING ROOT VARIABLE LOADINGS",
                "-" * 72,
            ])
            for idx, lds in self.loadings.items():
                mod = self.modulus[idx] if idx < len(self.modulus) else float("nan")
                top_items = [f"{v}: {w:.4f}" for v, w in lds.items()]
                lines.append(f"Root #{idx+1} (|lambda| = {mod:.4f}): {', '.join(top_items)}")
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        """Plot eigenvalues on the complex plane against the unit circle."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        if len(self.eigenvalues) == 0:
            ax.text(0.5, 0.5, "No eigenvalues to plot", ha="center", va="center")
            return ax

        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="#4a7bb0", linestyle="--", linewidth=1.2, label="Unit circle (|lambda| = 1)")

        finite_mask = np.isfinite(self.real) & np.isfinite(self.imag)
        re = self.real[finite_mask]
        im = self.imag[finite_mask]
        exp = self.is_explosive[finite_mask]
        unit = self.is_unit_root[finite_mask]
        stable = (~exp) & (~unit)

        if np.any(stable):
            ax.scatter(re[stable], im[stable], color="#2ca02c", marker="o", s=40, label=f"Stable ({np.sum(stable)})", zorder=3)
        if np.any(unit):
            ax.scatter(re[unit], im[unit], color="#ff7f0e", marker="D", s=50, label=f"Unit root ({np.sum(unit)})", zorder=4)
        if np.any(exp):
            ax.scatter(re[exp], im[exp], color="#d62728", marker="s", s=45, label=f"Explosive ({np.sum(exp)})", zorder=3)

        if self.loadings:
            for idx, lds in self.loadings.items():
                if idx < len(self.eigenvalues) and finite_mask[idx]:
                    rx, ix = self.real[idx], self.imag[idx]
                    ax.scatter([rx], [ix], facecolors="none", edgecolors="#17becf", s=130, linewidth=2.0, zorder=5)
                    top_vars = list(lds.keys())[:2]
                    callout = f"{self.modulus[idx]:.2f} ({', '.join(top_vars)})"
                    ax.annotate(callout, (rx, ix), textcoords="offset points", xytext=(5, 5), fontsize=8, color="#17becf", fontweight="bold")

        n_inf = int(np.sum(~finite_mask))
        title = f"Eigenvalue Spectrum (n_fwd={self.n_forward}, n_exp={self.n_explosive})"
        if n_inf > 0:
            title += f" [{n_inf} infinite deflated]"
        ax.set_title(title)
        ax.set_xlabel("Re(lambda)")
        ax.set_ylabel("Im(lambda)")
        ax.axhline(0, color="0.7", linestyle=":", linewidth=0.8)
        ax.axvline(0, color="0.7", linestyle=":", linewidth=0.8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.set_aspect("equal", adjustable="datalim")
        return ax

    def to_markdown(self, **kwargs) -> str:
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)


@dataclass(frozen=True)
class ModelDiagnosticsResult:
    """Frozen dataclass containing the results of DSGE model diagnostics.

    Attributes
    ----------
    passed : bool
        True if no error-severity findings were detected.
    findings : tuple[DiagnosticFinding, ...]
        Tuple of diagnostic findings.
    static_rank : int
        Numerical rank of the static Jacobian.
    static_n_vars : int
        Number of variables evaluated in the static Jacobian.
    collinear_equations : tuple[str, ...]
        Names or combinations of collinear equations identified by SVD.
    collinear_variables : tuple[str, ...]
        Names or combinations of unconstrained variables identified by SVD.
    unused_variables : tuple[str, ...]
        Variables that enter zero equations across all evaluation points.
    unused_equations : tuple[str, ...]
        Equations that depend on zero variables across all evaluation points.
    is_pencil_regular : bool
        True if the dynamic matrix pencil (A, B) is regular.
    stochastic_singularity : bool
        True if n_varobs > n_shocks + n_measurement_errors.
    unit_roots : tuple[int, ...]
        Indices of detected unit roots.
    incidence_matrix : np.ndarray
        Boolean incidence matrix (n_equations, n_variables), union of eval points.
    eval_points : int, default 5
        Number of neighbourhood points evaluated.
    variable_names : tuple[str, ...]
        Names of endogenous variables corresponding to columns of incidence_matrix.
    equation_names : tuple[str, ...]
        Names or labels of equations corresponding to rows of incidence_matrix.
    """

    passed: bool
    findings: tuple[DiagnosticFinding, ...]
    static_rank: int
    static_n_vars: int
    collinear_equations: tuple[str, ...]
    collinear_variables: tuple[str, ...]
    unused_variables: tuple[str, ...] = ()
    unused_equations: tuple[str, ...] = ()
    is_pencil_regular: bool = True
    stochastic_singularity: bool = False
    unit_roots: tuple[int, ...] = ()
    incidence_matrix: np.ndarray | None = None
    eval_points: int = 5
    variable_names: tuple[str, ...] = ()
    equation_names: tuple[str, ...] = ()

    @property
    def rank_deficient(self) -> bool:
        return self.static_rank < self.static_n_vars

    @property
    def n_vars(self) -> int:
        return self.static_n_vars

    @property
    def pencil_regular(self) -> bool:
        return self.is_pencil_regular

    @property
    def redundant_equations(self) -> tuple[str, ...]:
        return self.unused_equations

    @property
    def unit_roots_detected(self) -> int:
        return len(self.unit_roots)

    def to_frame(self) -> pd.DataFrame:
        """Return findings as a DataFrame."""
        if not self.findings:
            return pd.DataFrame([{
                "category": "all",
                "severity": "info",
                "message": "All diagnostic checks passed successfully.",
            }])
        return pd.DataFrame([{
            "category": f.category,
            "severity": f.severity,
            "message": f.message,
        } for f in self.findings])

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        err_count = sum(1 for f in self.findings if f.severity == "error")
        warn_count = sum(1 for f in self.findings if f.severity == "warning")
        lines = [
            "DSGE MODEL DIAGNOSTICS",
            "=" * 72,
            f"Overall status         : {status} ({err_count} errors, {warn_count} warnings)",
            f"Static Jacobian rank   : {self.static_rank} / {self.static_n_vars} "
            f"({'RANK DEFICIENT' if self.rank_deficient else 'FULL RANK'})",
            f"Dynamic pencil regular : {'YES' if self.is_pencil_regular else 'NO (SINGULAR)'}",
            f"Stochastic singularity : {'YES' if self.stochastic_singularity else 'NO'}",
            f"Unit roots detected    : {self.unit_roots_detected}",
            f"Unused variables       : {len(self.unused_variables)}" + (f" ({', '.join(self.unused_variables)})" if self.unused_variables else ""),
            f"Unused equations       : {len(self.unused_equations)}" + (f" ({', '.join(self.unused_equations)})" if self.unused_equations else ""),
        ]
        if self.collinear_equations:
            lines.append("Collinear equations    : " + "; ".join(self.collinear_equations))
        if self.collinear_variables:
            lines.append("Collinear variables    : " + "; ".join(self.collinear_variables))
        lines.extend([
            "",
            "DIAGNOSTIC FINDINGS",
            "-" * 72,
            self.to_frame().to_string(),
        ])
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        """Plot the numeric incidence matrix."""
        import matplotlib.pyplot as plt

        if self.incidence_matrix is None:
            if ax is None:
                fig, ax = plt.subplots(figsize=(5, 3))
            ax.text(0.5, 0.5, "No incidence matrix available", ha="center", va="center")
            return ax

        mat = self.incidence_matrix.astype(float)
        if ax is None:
            n_eq, n_var = mat.shape
            fig, ax = plt.subplots(figsize=(max(5, n_var * 0.4), max(4, n_eq * 0.3)))

        ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"Model Incidence Matrix ({mat.shape[0]} eqs x {mat.shape[1]} vars, {self.eval_points}-pt union)")

        if self.variable_names and len(self.variable_names) == mat.shape[1]:
            ax.set_xticks(range(len(self.variable_names)))
            ax.set_xticklabels(self.variable_names, rotation=90, fontsize=8)
        else:
            ax.set_xlabel("Variables")

        if self.equation_names and len(self.equation_names) == mat.shape[0]:
            ax.set_yticks(range(len(self.equation_names)))
            ax.set_yticklabels(self.equation_names, fontsize=8)
        else:
            ax.set_ylabel("Equations")

        ax.grid(True, which="both", color="0.8", linestyle=":", linewidth=0.5)
        return ax

    def to_markdown(self, **kwargs) -> str:
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)


@dataclass(frozen=True)
class IdentificationResult:
    """Frozen dataclass containing Iskrev (2010) and Komunjer & Ng (2011) parameter identification diagnostics.

    Attributes
    ----------
    is_identified : bool
        True if all four rank criteria (J1, J2, JH, JS) are full rank.
    j1_rank : int
        Numerical rank of state-space Jacobian J1 = d vec(T, R, Q, Z, H) / d theta.
    j1_n_params : int
        Number of parameters tested in J1.
    j1_null_space : np.ndarray
        Orthonormal basis of J1 null space, shape (n_null, n_params).
    j1_null_combinations : tuple[str, ...]
        Human-readable linear parameter combinations spanning J1 null space.
    j1_collinearity : pd.DataFrame
        Pairwise and multi-way collinearity R^2 for J1 columns.
    j2_rank : int
        Numerical rank of theoretical moment Jacobian J2 = d m(theta) / d theta.
    j2_n_params : int
        Number of parameters tested in J2.
    j2_null_space : np.ndarray
        Orthonormal basis of J2 null space, shape (n_null, n_params).
    j2_null_combinations : tuple[str, ...]
        Human-readable linear parameter combinations spanning J2 null space.
    j2_collinearity : pd.DataFrame
        Pairwise and multi-way collinearity R^2 for J2 columns.
    strength : pd.DataFrame
        Ratto (2011) identification strength and sensitivity per parameter.
    param_names : tuple[str, ...]
        Names of evaluated parameters.
    varobs : tuple[str, ...]
        Observables used in identification analysis.
    lags : int, default 1
        Autocovariance lags included in J2 moments.
    prior_mc_results : dict | None, default None
        Optional Monte Carlo identification results over prior distribution.
    is_identified_solution : bool, default True
        True if state-space solution Jacobian J1 is full rank.
    is_identified_moments : bool, default True
        True if theoretical moment Jacobian J2 is full rank.
    is_identified_transfer : bool, default True
        True if frequency-domain transfer function Jacobian JH is full rank.
    is_identified_spectrum : bool, default True
        True if frequency-domain power spectral density Jacobian JS is full rank.
    j1_singular_values : np.ndarray
        Singular values of J1.
    j1_condition_number : float
        Condition number kappa(J1) = s_1 / s_{n_theta}.
    j2_singular_values : np.ndarray
        Singular values of J2.
    j2_condition_number : float
        Condition number kappa(J2) = s_1 / s_{n_theta}.
    jh_rank : int
        Numerical rank of Komunjer & Ng (2011) transfer function Jacobian JH.
    jh_n_params : int
        Number of parameters tested in JH.
    jh_singular_values : np.ndarray
        Singular values of JH.
    jh_condition_number : float
        Condition number kappa(JH) = s_1 / s_{n_theta}.
    jh_null_space : np.ndarray
        Orthonormal basis of JH null space.
    jh_null_combinations : tuple[str, ...]
        Human-readable linear parameter combinations spanning JH null space.
    jh_collinearity : pd.DataFrame
        Pairwise and multi-way collinearity R^2 for JH columns.
    js_rank : int
        Numerical rank of Komunjer & Ng (2011) cross-spectral density Jacobian JS.
    js_n_params : int
        Number of parameters tested in JS.
    js_singular_values : np.ndarray
        Singular values of JS.
    js_condition_number : float
        Condition number kappa(JS) = s_1 / s_{n_theta}.
    js_null_space : np.ndarray
        Orthonormal basis of JS null space.
    js_null_combinations : tuple[str, ...]
        Human-readable linear parameter combinations spanning JS null space.
    js_collinearity : pd.DataFrame
        Pairwise and multi-way collinearity R^2 for JS columns.
    collinear_pairs : tuple[tuple[str, str, float, str], ...]
        Detected parameter pairs with pairwise R^2 > 0.95: (param1, param2, R^2, criterion).
    warnings : tuple[str, ...]
        Structured warnings for rank deficiencies, high condition numbers, and collinearities.
    n_freq : int, default 16
        Number of frequency grid points.
    frequencies : np.ndarray | None, default None
        Grid of frequency evaluation points omega in (0, pi).
    """

    is_identified: bool
    j1_rank: int
    j1_n_params: int
    j1_null_space: np.ndarray
    j1_null_combinations: tuple[str, ...]
    j1_collinearity: pd.DataFrame
    j2_rank: int
    j2_n_params: int
    j2_null_space: np.ndarray
    j2_null_combinations: tuple[str, ...]
    j2_collinearity: pd.DataFrame
    strength: pd.DataFrame
    param_names: tuple[str, ...]
    varobs: tuple[str, ...]
    lags: int = 1
    prior_mc_results: dict | None = None

    # Enhanced criteria fields (with defaults for full backward compatibility)
    is_identified_solution: bool = True
    is_identified_moments: bool = True
    is_identified_transfer: bool = True
    is_identified_spectrum: bool = True

    j1_singular_values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    j1_condition_number: float = 1.0

    j2_singular_values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    j2_condition_number: float = 1.0

    jh_rank: int = 0
    jh_n_params: int = 0
    jh_singular_values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    jh_condition_number: float = 1.0
    jh_null_space: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    jh_null_combinations: tuple[str, ...] = ()
    jh_collinearity: pd.DataFrame = field(default_factory=pd.DataFrame)

    js_rank: int = 0
    js_n_params: int = 0
    js_singular_values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    js_condition_number: float = 1.0
    js_null_space: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    js_null_combinations: tuple[str, ...] = ()
    js_collinearity: pd.DataFrame = field(default_factory=pd.DataFrame)

    collinear_pairs: tuple[tuple[str, str, float, str], ...] = ()
    warnings: tuple[str, ...] = ()
    n_freq: int = 16
    frequencies: np.ndarray | None = None

    def __post_init__(self) -> None:
        n_p = len(self.param_names)
        if self.j1_n_params > 0 and (self.j1_rank < self.j1_n_params):
            object.__setattr__(self, "is_identified_solution", False)
        if self.j2_n_params > 0 and (self.j2_rank < self.j2_n_params):
            object.__setattr__(self, "is_identified_moments", False)
        if self.jh_n_params > 0 and (self.jh_rank < self.jh_n_params):
            object.__setattr__(self, "is_identified_transfer", False)
        if self.js_n_params > 0 and (self.js_rank < self.js_n_params):
            object.__setattr__(self, "is_identified_spectrum", False)

        if self.jh_n_params == 0 and n_p > 0:
            object.__setattr__(self, "jh_n_params", n_p)
            object.__setattr__(self, "jh_rank", self.j1_rank)
            object.__setattr__(self, "is_identified_transfer", bool(self.j1_rank == n_p))
        if self.js_n_params == 0 and n_p > 0:
            object.__setattr__(self, "js_n_params", n_p)
            object.__setattr__(self, "js_rank", self.j2_rank)
            object.__setattr__(self, "is_identified_spectrum", bool(self.j2_rank == n_p))

        if self.jh_collinearity.empty and n_p > 0:
            object.__setattr__(
                self,
                "jh_collinearity",
                self.j1_collinearity.copy() if not self.j1_collinearity.empty else pd.DataFrame(index=list(self.param_names)),
            )
        if self.js_collinearity.empty and n_p > 0:
            object.__setattr__(
                self,
                "js_collinearity",
                self.j2_collinearity.copy() if not self.j2_collinearity.empty else pd.DataFrame(index=list(self.param_names)),
            )

    @property
    def rank_deficient(self) -> bool:
        """True if any active identification criterion is rank deficient."""
        return not self.is_identified

    @property
    def j1_rank_deficient(self) -> bool:
        """True if state-space Jacobian J1 is rank deficient."""
        return self.j1_rank < self.j1_n_params

    @property
    def j2_rank_deficient(self) -> bool:
        """True if moment Jacobian J2 is rank deficient."""
        return self.j2_rank < self.j2_n_params

    @property
    def jh_rank_deficient(self) -> bool:
        """True if Komunjer-Ng transfer Jacobian JH is rank deficient."""
        return self.jh_rank < self.jh_n_params

    @property
    def js_rank_deficient(self) -> bool:
        """True if Komunjer-Ng spectrum Jacobian JS is rank deficient."""
        return self.js_rank < self.js_n_params

    @property
    def n_params(self) -> int:
        """Total number of parameters evaluated."""
        return len(self.param_names)

    def rank_scorecard(self) -> pd.DataFrame:
        """Return publication-ready rank scorecard comparing all 4 identification criteria."""
        records = [
            {
                "criterion": "J1 (Iskrev Solution)",
                "rank": self.j1_rank,
                "total": self.j1_n_params,
                "deficiency": self.j1_n_params - self.j1_rank,
                "condition_number": self.j1_condition_number,
                "status": "FULL RANK" if self.is_identified_solution else f"DEFICIENT by {self.j1_n_params - self.j1_rank}",
            },
            {
                "criterion": f"J2 (Iskrev Moments, p={self.lags})",
                "rank": self.j2_rank,
                "total": self.j2_n_params,
                "deficiency": self.j2_n_params - self.j2_rank,
                "condition_number": self.j2_condition_number,
                "status": "FULL RANK" if self.is_identified_moments else f"DEFICIENT by {self.j2_n_params - self.j2_rank}",
            },
            {
                "criterion": "JH (Komunjer-Ng Transfer)",
                "rank": self.jh_rank,
                "total": self.jh_n_params,
                "deficiency": self.jh_n_params - self.jh_rank,
                "condition_number": self.jh_condition_number,
                "status": "FULL RANK" if self.is_identified_transfer else f"DEFICIENT by {self.jh_n_params - self.jh_rank}",
            },
            {
                "criterion": "JS (Komunjer-Ng Spectrum)",
                "rank": self.js_rank,
                "total": self.js_n_params,
                "deficiency": self.js_n_params - self.js_rank,
                "condition_number": self.js_condition_number,
                "status": "FULL RANK" if self.is_identified_spectrum else f"DEFICIENT by {self.js_n_params - self.js_rank}",
            },
        ]
        return pd.DataFrame(records, index=["J1", "J2", "JH", "JS"])

    def to_frame(self) -> pd.DataFrame:
        """Return parameter identification summary table as a DataFrame."""
        rows = []
        for p in self.param_names:
            j1_r2 = float(self.j1_collinearity.loc[p, "r2"]) if (p in self.j1_collinearity.index and "r2" in self.j1_collinearity.columns) else 0.0
            j2_r2 = float(self.j2_collinearity.loc[p, "r2"]) if (p in self.j2_collinearity.index and "r2" in self.j2_collinearity.columns) else 0.0
            jh_r2 = float(self.jh_collinearity.loc[p, "r2"]) if (p in self.jh_collinearity.index and "r2" in self.jh_collinearity.columns) else 0.0
            js_r2 = float(self.js_collinearity.loc[p, "r2"]) if (p in self.js_collinearity.index and "r2" in self.js_collinearity.columns) else 0.0
            sens = float(self.strength.loc[p, "sensitivity"]) if (p in self.strength.index and "sensitivity" in self.strength.columns) else 0.0
            st = float(self.strength.loc[p, "strength"]) if (p in self.strength.index and "strength" in self.strength.columns) else 0.0
            norm_st = float(self.strength.loc[p, "normalized_strength"]) if (p in self.strength.index and "normalized_strength" in self.strength.columns) else 0.0
            worst_partner = str(self.j2_collinearity.loc[p, "worst_partner"]) if (p in self.j2_collinearity.index and "worst_partner" in self.j2_collinearity.columns) else "None"
            is_ident = bool((j1_r2 < 0.999) and (j2_r2 < 0.999) and (jh_r2 < 0.999) and (js_r2 < 0.999) and (sens > 1e-8))
            rows.append({
                "j1_collinearity": j1_r2,
                "j2_collinearity": j2_r2,
                "jh_collinearity": jh_r2,
                "js_collinearity": js_r2,
                "sensitivity": sens,
                "strength": st,
                "normalized_strength": norm_st,
                "worst_partner": worst_partner,
                "identified": is_ident,
            })
        return pd.DataFrame(rows, index=list(self.param_names))

    def summary(self) -> str:
        """Render publication-ready identification diagnostics summary."""
        status_str = "IDENTIFIED" if self.is_identified else "UNIDENTIFIED (RANK DEFICIENT)"
        lines = [
            "PARAMETER IDENTIFICATION ANALYSIS (Iskrev 2010 / Komunjer-Ng 2011)",
            "=" * 72,
            f"Overall status         : {status_str}",
            f"Parameters evaluated   : {len(self.param_names)}",
            f"Observables (varobs)   : {', '.join(self.varobs)}",
            f"Autocovariance lags    : {self.lags}",
            f"Frequency grid points  : {self.n_freq}",
            f"J1 (Solution) rank     : {self.j1_rank} / {self.j1_n_params} "
            f"({'FULL RANK' if not self.j1_rank_deficient else f'DEFICIENT by {self.j1_n_params - self.j1_rank}'})",
            f"J2 (Moments) rank      : {self.j2_rank} / {self.j2_n_params} "
            f"({'FULL RANK' if not self.j2_rank_deficient else f'DEFICIENT by {self.j2_n_params - self.j2_rank}'})",
            f"JH (Transfer) rank     : {self.jh_rank} / {self.jh_n_params} "
            f"({'FULL RANK' if not self.jh_rank_deficient else f'DEFICIENT by {self.jh_n_params - self.jh_rank}'})",
            f"JS (Spectrum) rank     : {self.js_rank} / {self.js_n_params} "
            f"({'FULL RANK' if not self.js_rank_deficient else f'DEFICIENT by {self.js_n_params - self.js_rank}'})",
            "",
            "RANK CRITERIA SCORECARD",
            "-" * 72,
            self.rank_scorecard().to_string(),
        ]

        has_null = False
        for name, combs in [
            ("J1 NULL SPACE PARAMETER COMBINATIONS", self.j1_null_combinations),
            ("J2 NULL SPACE PARAMETER COMBINATIONS", self.j2_null_combinations),
            ("JH NULL SPACE PARAMETER COMBINATIONS", self.jh_null_combinations),
            ("JS NULL SPACE PARAMETER COMBINATIONS", self.js_null_combinations),
        ]:
            if combs:
                has_null = True
                lines.extend(["", name, "-" * 72])
                for comb in combs:
                    lines.append(f"  {comb}")

        if not has_null:
            lines.extend([
                "",
                "NULL SPACE DIRECTIONS",
                "-" * 72,
                "  None (All parameters locally identified)",
            ])

        if self.warnings:
            lines.extend([
                "",
                "WARNINGS & IDENTIFICATION DIAGNOSTICS",
                "-" * 72,
            ])
            for w in self.warnings:
                lines.append(f"  * {w}")

        lines.extend([
            "",
            "PARAMETER IDENTIFICATION SUMMARY",
            "-" * 72,
            self.to_frame().round(4).to_string(),
            "=" * 72,
        ])
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        """Plot parameter collinearity R^2 and identification strength."""
        import matplotlib.pyplot as plt

        n_p = len(self.param_names)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(4.0, n_p * 0.45)))

        if n_p == 0:
            ax.text(0.5, 0.5, "No parameters evaluated", ha="center", va="center")
            return ax

        y_pos = np.arange(n_p)
        height = 0.25

        r2_j2 = (
            self.j2_collinearity["r2"].to_numpy()
            if "r2" in self.j2_collinearity.columns
            else np.zeros(n_p)
        )
        r2_js = (
            self.js_collinearity["r2"].to_numpy()
            if "r2" in self.js_collinearity.columns
            else np.zeros(n_p)
        )
        st_vals = (
            self.strength["normalized_strength"].to_numpy()
            if "normalized_strength" in self.strength.columns
            else (
                self.strength["strength"].to_numpy()
                if "strength" in self.strength.columns
                else np.zeros(n_p)
            )
        )

        ax.barh(
            y_pos - height,
            r2_j2,
            height=height,
            color="#4a7bb0",
            alpha=0.85,
            label="Collinearity $R^2$ (J2)",
        )
        ax.barh(
            y_pos,
            r2_js,
            height=height,
            color="#9467bd",
            alpha=0.85,
            label="Collinearity $R^2$ (JS Spectrum)",
        )
        ax.barh(
            y_pos + height,
            st_vals,
            height=height,
            color="#2ca02c",
            alpha=0.85,
            label="Norm. Identification Strength",
        )
        ax.axvline(
            1.0,
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label="Collinear (1.0)",
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(self.param_names))
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Metric Value [0, 1]")
        ax.set_title("DSGE Parameter Identification: Collinearity vs Strength")
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8)

        return ax

    def to_markdown(self, table: str = "parameters", **kwargs) -> str:
        """Render summary table as Markdown."""
        from puremacro.reports import _df_to_markdown

        df = self.rank_scorecard() if table == "scorecard" else self.to_frame()
        return _df_to_markdown(df, **kwargs)

    def to_latex(self, table: str = "parameters", **kwargs) -> str:
        """Render summary table as LaTeX tabular."""
        from puremacro.reports import _df_to_latex

        df = self.rank_scorecard() if table == "scorecard" else self.to_frame()
        return _df_to_latex(df, **kwargs)

    def to_typst(self, table: str = "parameters", **kwargs) -> str:
        """Render summary table as Typst table."""
        from puremacro.reports import _df_to_typst

        df = self.rank_scorecard() if table == "scorecard" else self.to_frame()
        return _df_to_typst(df, **kwargs)



@dataclass(frozen=True)
class OSRResult:
    """Frozen dataclass containing Optimal Simple Rule optimization results.

    Attributes
    ----------
    optimal_params : dict[str, float]
        Dictionary of optimal rule parameter values.
    initial_params : dict[str, float]
        Dictionary of baseline/initial rule parameter values.
    loss_opt : float
        Value of quadratic loss at the optimum.
    loss_initial : float
        Value of quadratic loss at initial parameters.
    rule_params : tuple[str, ...]
        Names of optimized rule parameters.
    target_vars : tuple[str, ...]
        Names of target variables in loss function.
    weights : dict[str, float]
        Weights assigned to each target variable.
    variance_table : pd.DataFrame
        DataFrame detailing variances, weights, and weighted losses initially and at optimum.
    converged : bool
        True if the optimizer reported successful convergence.
    message : str
        Termination status message from optimizer.
    n_evaluations : int
        Total function evaluations.
    optimal_model : Any = None
        Model re-solved at optimal parameter values.
    """

    optimal_params: dict[str, float]
    initial_params: dict[str, float]
    loss_opt: float
    loss_initial: float
    rule_params: tuple[str, ...]
    target_vars: tuple[str, ...]
    weights: dict[str, float]
    variance_table: pd.DataFrame
    converged: bool
    message: str
    n_evaluations: int
    optimal_model: Any = None

    @property
    def loss_init(self) -> float:
        """Alias for loss_initial."""
        return self.loss_initial

    @property
    def loss_calib(self) -> float:
        """Alias for loss_initial."""
        return self.loss_initial

    def to_frame(self) -> pd.DataFrame:
        """Return canonical DataFrame representation of the variance comparison table."""
        return self.variance_table.copy()

    def summary(self) -> str:
        """Render human-readable summary of OSR optimization results."""
        improvement_pct = 0.0
        if self.loss_initial > 0:
            improvement_pct = 100.0 * (self.loss_initial - self.loss_opt) / self.loss_initial

        lines = [
            "OPTIMAL SIMPLE RULES (OSR) OPTIMIZATION",
            "=" * 72,
            f"Convergence status      : {'CONVERGED' if self.converged else 'TERMINATED'} ({self.message})",
            f"Function evaluations    : {self.n_evaluations}",
            f"Baseline loss           : {self.loss_initial:.6e}",
            f"Optimal loss            : {self.loss_opt:.6e}",
            f"Loss reduction          : {improvement_pct:.2f}%",
            "",
            "RULE PARAMETERS",
            "-" * 72,
            f"{'Parameter':<20} {'Initial':>15} {'Optimal':>15} {'Change':>15}",
            "-" * 72,
        ]
        for p in self.rule_params:
            init_val = self.initial_params.get(p, np.nan)
            opt_val = self.optimal_params.get(p, np.nan)
            chg = opt_val - init_val if not (np.isnan(init_val) or np.isnan(opt_val)) else np.nan
            lines.append(f"{p:<20} {init_val:>15.6f} {opt_val:>15.6f} {chg:>+15.6f}")

        lines.extend([
            "",
            "TARGET VARIABLE VARIANCES",
            "-" * 72,
            self.variance_table.round(6).to_string(),
            "=" * 72,
        ])
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        """Plot grouped bar chart comparing variances under baseline vs optimal rule."""
        import matplotlib.pyplot as plt

        df = self.variance_table
        vars_list = list(df.index)
        n_vars = len(vars_list)

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(6.0, n_vars * 1.5), 4.5))

        if n_vars == 0:
            ax.text(0.5, 0.5, "No target variables to plot", ha="center", va="center")
            return ax

        x = np.arange(n_vars)
        width = 0.35

        var_init = (
            df["var_initial"].to_numpy()
            if "var_initial" in df.columns
            else (df["var_calib"].to_numpy() if "var_calib" in df.columns else np.zeros(n_vars))
        )
        var_opt = (
            df["var_optimal"].to_numpy()
            if "var_optimal" in df.columns
            else (df["var_opt"].to_numpy() if "var_opt" in df.columns else np.zeros(n_vars))
        )

        ax.bar(x - width / 2, var_init, width, label="Baseline", color="#4a7bb0", alpha=0.85)
        ax.bar(x + width / 2, var_opt, width, label="Optimal Rule", color="#2ca02c", alpha=0.85)

        ax.set_ylabel("Theoretical Variance")
        ax.set_title("Optimal Simple Rules: Variance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(vars_list)
        ax.legend()
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        return ax

    def to_markdown(self, **kwargs) -> str:
        """Render summary variance table as Markdown."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Render summary variance table as LaTeX tabular."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Render summary variance table as Typst table."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)


@dataclass(frozen=True)
class PolicyResult:
    """Frozen dataclass containing optimal policy regime results.

    Attributes
    ----------
    regime : str
        "discretion" or "commitment".
    target_vars : tuple[str, ...]
        Target variable names.
    weights : dict[str, float]
        Weights on target variables.
    instruments : tuple[str, ...]
        Names of policy instruments.
    beta : float
        Policymaker discount factor.
    loss : float
        Expected unconditional quadratic loss.
    policy_rules : pd.DataFrame
        Reaction function coefficients expressing instrument(s) in terms of states.
    transition_matrix : np.ndarray
        Closed-loop state transition matrix G.
    impact_matrix : np.ndarray
        Closed-loop shock loading matrix N.
    multipliers : tuple[str, ...]
        Names of Lagrange multiplier variables (empty in discretion).
    linear_model : Any
        The solved LinearModel under the policy regime, enabling .irf(), .fevd(),
        .simulate(), and .theoretical_moments().
    """

    regime: str
    target_vars: tuple[str, ...]
    weights: dict[str, float]
    instruments: tuple[str, ...]
    beta: float
    loss: float
    policy_rules: pd.DataFrame
    transition_matrix: np.ndarray
    impact_matrix: np.ndarray
    multipliers: tuple[str, ...] = ()
    linear_model: Any = None

    @property
    def augmented_model(self) -> Any:
        """Alias for linear_model."""
        return self.linear_model

    def to_frame(self) -> pd.DataFrame:
        """Return canonical DataFrame representation of the policy rule reaction coefficients."""
        return self.policy_rules.copy()

    def summary(self) -> str:
        """Render human-readable summary of policy regime equilibrium."""
        lines = [
            f"OPTIMAL POLICY REGIME: {self.regime.upper()}",
            "=" * 72,
            f"Policymaker discount (beta): {self.beta:.4f}",
            f"Expected unconditional loss : {self.loss:.6e}",
            f"Policy instruments          : {', '.join(self.instruments)}",
            f"Target variables            : {', '.join(f'{k} (w={v})' for k, v in self.weights.items())}",
        ]
        if self.multipliers:
            lines.append(f"Lagrange multipliers        : {', '.join(self.multipliers)}")
        lines.extend([
            "",
            "POLICY REACTION FUNCTIONS (Coefficients on States)",
            "-" * 72,
            self.policy_rules.round(6).to_string(),
            "=" * 72,
        ])
        return "\n".join(lines)

    def plot(self, ax=None, periods: int = 16, shock: str | None = None, **kwargs):
        """Plot impulse responses under the optimal policy regime."""
        import matplotlib.pyplot as plt

        if self.linear_model is None:
            if ax is None:
                fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No linear model attached", ha="center", va="center")
            return ax

        shock_name = shock if shock is not None else (self.linear_model.shocks[0] if self.linear_model.shocks else None)
        if shock_name is None:
            if ax is None:
                fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No shocks in model", ha="center", va="center")
            return ax
        irf_df = self.linear_model.irf(shock_name, horizon=periods)
        plot_vars = [v for v in self.target_vars if v in irf_df.columns]
        for inst in self.instruments:
            if inst in irf_df.columns and inst not in plot_vars:
                plot_vars.append(inst)
        for mult in self.multipliers:
            if mult in irf_df.columns and mult not in plot_vars:
                plot_vars.append(mult)

        if not plot_vars:
            plot_vars = list(irf_df.columns[:min(4, len(irf_df.columns))])

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.5))

        for v in plot_vars:
            ax.plot(irf_df.index, irf_df[v], label=v, linewidth=1.8)

        ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Deviation")
        ax.set_title(f"Optimal Policy ({self.regime.capitalize()}): Impulse Responses")
        ax.legend(loc="best")
        ax.grid(True, linestyle=":", alpha=0.6)
        return ax

    def to_markdown(self, **kwargs) -> str:
        """Render policy rules table as Markdown."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Render policy rules table as LaTeX tabular."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Render policy rules table as Typst table."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)


@dataclass(frozen=True)
class DiscretionaryPolicyResult(PolicyResult):
    """Frozen dataclass containing optimal discretionary policy regime results (Dennis 2007).

    Attributes
    ----------
    regime : str
        Always "discretion".
    target_vars : tuple[str, ...]
        Target variable names.
    weights : pd.Series
        Quadratic loss weights per target variable.
    instruments : tuple[str, ...]
        Names of policy instruments.
    beta : float
        Policymaker discount factor.
    loss : float
        Expected unconditional quadratic loss.
    policy_rules : pd.DataFrame
        Reaction function coefficients expressing instrument(s) in terms of states.
    F : pd.DataFrame
        Policy feedback matrix F mapping predetermined states to instruments.
    V : np.ndarray
        Converged Riccati continuation value matrix.
    transition_matrix : pd.DataFrame
        Closed-loop state transition matrix G.
    impact_matrix : pd.DataFrame
        Closed-loop shock loading matrix N.
    inflation_bias : float
        Quantified inflation bias E[pi^disc] - E[pi^comm].
    stabilization_bias : float
        Quantified stabilization bias Loss^disc - Loss^comm.
    converged : bool
        Whether Dennis (2007) policy iteration converged within max_iter.
    iterations : int
        Number of policy iterations executed.
    diff : float
        Final sup-norm difference ||F_{k+1} - F_k||_infty.
    linear_model : Any
        The solved LinearModel under discretion.
    commitment_result : Any | None
        Solved PolicyResult under LQ commitment for formal bias comparison.
    """

    F: pd.DataFrame = field(default_factory=pd.DataFrame)
    V: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    inflation_bias: float = 0.0
    stabilization_bias: float = 0.0
    converged: bool = True
    iterations: int = 0
    diff: float = 0.0
    commitment_result: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.weights, pd.Series):
            object.__setattr__(self, "weights", pd.Series(self.weights))
        if not isinstance(self.transition_matrix, pd.DataFrame):
            object.__setattr__(self, "transition_matrix", pd.DataFrame(self.transition_matrix))
        if not isinstance(self.impact_matrix, pd.DataFrame):
            object.__setattr__(self, "impact_matrix", pd.DataFrame(self.impact_matrix))
        if not isinstance(self.F, pd.DataFrame):
            object.__setattr__(self, "F", pd.DataFrame(self.F))
        if not isinstance(self.policy_rules, pd.DataFrame):
            object.__setattr__(self, "policy_rules", pd.DataFrame(self.policy_rules))

    @property
    def transition(self) -> np.ndarray:
        """Closed-loop state transition matrix G as numpy array."""
        if isinstance(self.transition_matrix, pd.DataFrame):
            return self.transition_matrix.values
        return np.asarray(self.transition_matrix)

    @property
    def impact(self) -> np.ndarray:
        """Closed-loop shock loading matrix N as numpy array."""
        if isinstance(self.impact_matrix, pd.DataFrame):
            return self.impact_matrix.values
        return np.asarray(self.impact_matrix)

    def summary(self) -> str:
        """Render human-readable summary of discretionary policy regime equilibrium."""
        lines = [
            f"OPTIMAL POLICY REGIME: {self.regime.upper()} (DENNIS 2007)",
            "=" * 72,
            f"Policymaker discount (beta): {self.beta:.4f}",
            f"Expected unconditional loss : {self.loss:.6e}",
            f"Policy instruments          : {', '.join(self.instruments)}",
            f"Target variables            : {', '.join(f'{k} (w={v})' for k, v in self.weights.items())}",
            f"Convergence status          : {'Converged' if self.converged else 'Did not converge'} in {self.iterations} iterations (diff={self.diff:.2e})",
        ]
        if self.inflation_bias != 0.0:
            lines.append(f"Inflation bias (E[pi^disc] - E[pi^comm]): {self.inflation_bias:.6e}")
        if self.stabilization_bias != 0.0:
            lines.append(f"Stabilization bias (Loss^disc - Loss^comm): {self.stabilization_bias:.6e}")
        if self.V is not None and self.V.size > 0:
            lines.append(f"Riccati value matrix (norm) : {float(np.linalg.norm(self.V)):.6e}")
        lines.extend([
            "",
            "POLICY REACTION FUNCTIONS (Coefficients on States)",
            "-" * 72,
            self.policy_rules.round(6).to_string(),
            "=" * 72,
        ])
        return "\n".join(lines)

    def plot(self, ax=None, periods: int = 16, shock: str | None = None, compare_commitment: bool = False, **kwargs):
        """Plot impulse responses under the optimal discretionary policy regime."""
        import matplotlib.pyplot as plt

        if compare_commitment and self.commitment_result is not None and self.linear_model is not None:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 4.5))
            shock_name = shock if shock is not None else (self.linear_model.shocks[0] if self.linear_model.shocks else None)
            if shock_name is not None:
                irf_disc = self.linear_model.irf(shock_name, horizon=periods)
                irf_comm = self.commitment_result.linear_model.irf(shock_name, horizon=periods)
                plot_vars = [v for v in self.target_vars if v in irf_disc.columns]
                for v in plot_vars:
                    ax.plot(irf_disc.index, irf_disc[v], label=f"{v} (discretion)", linewidth=2.0)
                    if v in irf_comm.columns:
                        ax.plot(irf_comm.index, irf_comm[v], label=f"{v} (commitment)", linestyle="--", linewidth=1.8)
                ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.set_xlabel("Horizon")
                ax.set_ylabel("Deviation")
                ax.set_title(f"Optimal Policy (Discretion vs Commitment): {shock_name}")
                ax.legend(loc="best")
                ax.grid(True, linestyle=":", alpha=0.6)
                return ax
        return super().plot(ax=ax, periods=periods, shock=shock, **kwargs)



@dataclass(frozen=True)
class ExtendedPathResult:
    """Result of Fair & Taylor (1983) non-linear extended path stochastic simulation.

    Extended path simulates non-linear dynamic models without perturbation by
    replacing future mathematical expectations with deterministic forecasts
    under the assumption of zero future innovations (u_{t+s}^e = 0 for s >= 1),
    solving a rolling boundary value problem over horizon T_H at each date t
    via the SuperLU sparse stacked Newton-Raphson engine.

    Attributes
    ----------
    path : pd.DataFrame, shape (periods, n_vars)
        Realized simulation trajectory of endogenous variables from t=1 to t=periods.
    shocks : pd.DataFrame, shape (periods, n_shocks)
        Sequence of structural shock innovations drawn or supplied across periods.
    converged : bool
        Whether the stacked Newton solver converged at all simulation periods.
    iterations : list[int] | int
        Number of Newton iterations per simulation period (or total sum).
    residual_norm : float
        Maximum dynamic equation residual infinity-norm across all periods.
    terminal_error : float
        Maximum boundary error ||y_{t+T_H} - y_ss||_inf across all rolling solves.
    variable_names : tuple[str, ...], default ()
        Names of endogenous variables in declaration or column order.
    shock_names : tuple[str, ...], default ()
        Names of structural shocks in declaration or column order.
    horizon : int, default 100
        Forward anticipation horizon T_H used at each step.
    """

    path: pd.DataFrame
    shocks: pd.DataFrame
    converged: bool
    iterations: list[int] | int
    residual_norm: float
    terminal_error: float
    variable_names: tuple[str, ...] = ()
    shock_names: tuple[str, ...] = ()
    horizon: int = 100

    def __post_init__(self) -> None:
        if not self.variable_names and not self.path.empty:
            object.__setattr__(self, "variable_names", tuple(str(c) for c in self.path.columns))
        if not self.shock_names and not self.shocks.empty:
            object.__setattr__(self, "shock_names", tuple(str(c) for c in self.shocks.columns))

    @property
    def periods(self) -> int:
        """Number of simulation periods."""
        return len(self.path)

    @property
    def n_vars(self) -> int:
        """Number of endogenous variables."""
        return len(self.variable_names)

    @property
    def n_shocks(self) -> int:
        """Number of structural shocks."""
        return len(self.shock_names)

    @property
    def total_iterations(self) -> int:
        """Total Newton-Raphson iterations summed across all periods."""
        if isinstance(self.iterations, (list, tuple, np.ndarray)):
            return int(sum(self.iterations))
        return int(self.iterations)

    @property
    def mean_iterations(self) -> float:
        """Mean Newton-Raphson iterations per period."""
        if isinstance(self.iterations, (list, tuple, np.ndarray)):
            return float(np.mean(self.iterations)) if len(self.iterations) > 0 else 0.0
        return float(self.iterations)

    @property
    def max_iterations(self) -> int:
        """Maximum Newton-Raphson iterations in any single period."""
        if isinstance(self.iterations, (list, tuple, np.ndarray)):
            return int(np.max(self.iterations)) if len(self.iterations) > 0 else 0
        return int(self.iterations)

    def __getitem__(self, key: str) -> pd.Series:
        """Access variable or shock trajectory by column name."""
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.path.columns:
            return self.path[key]
        if key in self.shocks.columns:
            return self.shocks[key]
        raise KeyError(f"Variable or attribute {key!r} not found in ExtendedPathResult.")

    def to_frame(self) -> pd.DataFrame:
        """Return the simulated trajectory of endogenous variables as a DataFrame."""
        return self.path.copy()

    def summary(self, as_dataframe: bool = False) -> str | pd.DataFrame:
        """Render summary report of extended path convergence and trajectory statistics.

        Parameters
        ----------
        as_dataframe : bool, default False
            If True, returns a pandas DataFrame with summary statistics.
            If False, returns a publication-formatted string report.
        """
        stats = self.path.describe().T[["mean", "std", "min", "max"]].copy()
        if not self.path.empty:
            stats["initial"] = self.path.iloc[0].values
            stats["terminal"] = self.path.iloc[-1].values

        if as_dataframe:
            return stats

        status_str = "CONVERGED" if self.converged else "FAILED (solver divergence)"
        lines = [
            "EXTENDED PATH SIMULATION RESULT (Fair-Taylor 1983)",
            "=" * 72,
            f"Convergence status  : {status_str}",
            f"Simulation periods  : {len(self.path)}",
            f"Forward horizon T_H : {self.horizon}",
            f"Total iterations    : {self.total_iterations} (mean: {self.mean_iterations:.2f}, max: {self.max_iterations})",
            f"Residual norm       : {self.residual_norm:.4e}",
            f"Terminal error      : {self.terminal_error:.4e}",
            f"Endogenous vars ({self.n_vars}) : {', '.join(self.variable_names)}",
            f"Structural shocks ({self.n_shocks}) : {', '.join(self.shock_names)}",
            "-" * 72,
            "TRAJECTORY SUMMARY (t=1..T):",
            stats.round(6).to_string(),
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_markdown(self, *, head: int | None = None, index: bool = True, **kwargs) -> str:
        """Render simulated path as a Markdown table."""
        from puremacro.reports import _df_to_markdown

        df = self.path.head(head) if head is not None else self.path
        return _df_to_markdown(df, index=index, **kwargs)

    def to_latex(self, *, head: int | None = None, index: bool = True, **kwargs) -> str:
        """Render simulated path as a LaTeX tabular environment."""
        from puremacro.reports import _df_to_latex

        df = self.path.head(head) if head is not None else self.path
        return _df_to_latex(df, index=index, **kwargs)

    def to_typst(self, *, head: int | None = None, index: bool = True, **kwargs) -> str:
        """Render simulated path as a Typst table."""
        from puremacro.reports import _df_to_typst

        df = self.path.head(head) if head is not None else self.path
        return _df_to_typst(df, index=index, **kwargs)

    def plot(
        self,
        variables: Sequence[str] | None = None,
        style: str = "publication",
        *,
        ax: Any = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        xlabel: str = "Period (t)",
        ylabel: str = "Level",
        subplots: bool = False,
        **kwargs,
    ) -> Any:
        """Plot simulated variable trajectories.

        Parameters
        ----------
        variables : Sequence[str], optional
            Names of variables to plot. Defaults to all variables.
        style : str, default 'publication'
            Theme ('publication', 'grayscale', 'default').
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw on. If None, a new figure is created.
        figsize : tuple[float, float], optional
            Figure dimensions (width, height).
        title : str, optional
            Figure or axes title.
        xlabel : str, default 'Period (t)'
            X-axis label.
        ylabel : str, default 'Level'
            Y-axis label.
        subplots : bool, default False
            If True and ax is None, creates a grid of subplots for each variable.
        **kwargs
            Additional arguments passed to ax.plot.

        Returns
        -------
        matplotlib.figure.Figure | matplotlib.axes.Axes
            Returns Figure when ax is None, or Axes when ax is provided.
        """
        import matplotlib.pyplot as plt
        from puremacro.plot import _new_ax, _palette, _styles

        if variables is not None:
            cols = [v for v in variables if v in self.path.columns]
            if not cols:
                raise ValueError(
                    f"None of requested variables {variables} found in path columns {list(self.path.columns)}"
                )
        else:
            cols = list(self.path.columns)

        if subplots and len(cols) > 1 and ax is None:
            n_plots = len(cols)
            n_cols = min(3, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols
            fig_size = figsize or (3.8 * n_cols, 2.5 * n_rows)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, squeeze=False, sharex=True)
            ax_flat = axes.flatten()

            for idx, col in enumerate(cols):
                a = ax_flat[idx]
                a.plot(self.path.index, self.path[col], linewidth=1.5, **kwargs)
                a.set_title(str(col), fontsize=10, fontweight="bold")
                a.set_ylabel(ylabel, fontsize=8)
                a.grid(True, linestyle=":", alpha=0.5)
                if idx >= (n_rows - 1) * n_cols or idx == n_plots - 1:
                    a.set_xlabel(xlabel, fontsize=8)

            for idx in range(n_plots, len(ax_flat)):
                ax_flat[idx].set_visible(False)

            if title:
                fig.suptitle(title, fontsize=12, fontweight="bold")
            else:
                fig.suptitle("Extended Path Simulation (Fair-Taylor 1983)", fontsize=12, fontweight="bold")
            fig.tight_layout()
            return fig

        fig, target_ax = _new_ax(ax, figsize=figsize or (8.0, 4.5))

        if style in ("publication", "grayscale"):
            try:
                from puremacro.plotting.bw_style import bw_colors, bw_linestyles
                colors = bw_colors(len(cols))
                linestyles = bw_linestyles(len(cols))
            except ImportError:
                colors = _palette(len(cols))
                linestyles = _styles(len(cols))

            for i, col in enumerate(cols):
                target_ax.plot(
                    self.path.index,
                    self.path[col],
                    label=str(col),
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=1.4,
                    **kwargs,
                )
            target_ax.spines["top"].set_visible(False)
            target_ax.spines["right"].set_visible(False)
            target_ax.grid(True, linestyle=":", linewidth=0.5, color="0.7", alpha=0.7)
        else:
            for col in cols:
                target_ax.plot(
                    self.path.index,
                    self.path[col],
                    label=str(col),
                    linewidth=1.4,
                    **kwargs,
                )
            target_ax.grid(True, alpha=0.3)

        target_ax.set_xlabel(xlabel)
        target_ax.set_ylabel(ylabel)
        if title is not None:
            target_ax.set_title(title)
        else:
            target_ax.set_title("Extended Path Simulation (Fair-Taylor 1983)")
        target_ax.legend(loc="best", frameon=False)

        if ax is None:
            return fig
        return target_ax


@dataclass(frozen=True)
class ConditionalForecastResult:
    """Container for DSGE conditional forecast results (Waggoner & Zha 1999).

    Attributes
    ----------
    forecast : pd.DataFrame
        Realized forecast trajectory for all model variables across horizons.
    shocks : pd.DataFrame
        Required structural shock paths across horizons.
    conditions : dict
        Target paths specified for conditioned variables.
    controlled_shocks : tuple[str, ...]
        Names of structural shocks adjusted to satisfy conditions.
    baseline : pd.DataFrame, optional
        Unconditional baseline forecast trajectory with zero future shocks.
    bands : dict[str, pd.DataFrame], optional
        Confidence bands (e.g. "lower", "upper", "lower_68", "upper_68").
    variable_names : tuple[str, ...], optional
        Names of all model variables.
    shock_names : tuple[str, ...], optional
        Names of all structural shocks.
    horizon : int, optional
        Forecast horizon length.
    """

    forecast: pd.DataFrame
    shocks: pd.DataFrame
    conditions: dict = field(default_factory=dict)
    controlled_shocks: tuple[str, ...] = ()
    baseline: pd.DataFrame | None = None
    bands: dict[str, pd.DataFrame] | None = None
    variable_names: tuple[str, ...] = ()
    shock_names: tuple[str, ...] = ()
    horizon: int = 0
    simulations: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.baseline is None:
            object.__setattr__(self, "baseline", self.forecast.copy())
        if not self.variable_names:
            object.__setattr__(self, "variable_names", tuple(self.forecast.columns))
        if not self.shock_names:
            object.__setattr__(self, "shock_names", tuple(self.shocks.columns))
        if self.horizon == 0:
            object.__setattr__(self, "horizon", len(self.forecast))
        if isinstance(self.controlled_shocks, (list, set)):
            object.__setattr__(self, "controlled_shocks", tuple(self.controlled_shocks))

    def summary(self) -> str:
        """Render human-readable summary of conditional forecast."""
        lines = [
            "Conditional Forecast (Waggoner & Zha 1999)",
            "=" * 72,
            f"Forecast Horizon    : {self.horizon} periods",
            f"Controlled Shocks   : {', '.join(self.controlled_shocks) if self.controlled_shocks else 'All declared shocks'}",
            f"Conditioned Vars    : {', '.join(self.conditions.keys()) if self.conditions else 'None (unconditional)'}",
            "-" * 72,
            "TARGET CONDITIONS & REALIZED VALUES:",
        ]
        for var, path in self.conditions.items():
            vals_str = ", ".join(f"h={h+1}: {v:.4f}" for h, v in enumerate(path) if v is not None and not np.isnan(v))
            lines.append(f"  {var:15s}: {vals_str}")
        lines.extend([
            "",
            "REQUIRED STRUCTURAL SHOCKS (Active Horizons):",
            "-" * 72,
        ])
        ctrl_cols = [s for s in self.controlled_shocks if s in self.shocks.columns]
        shock_subset = self.shocks[ctrl_cols] if ctrl_cols else self.shocks
        lines.append(shock_subset.round(6).to_string())
        lines.append("=" * 72)
        return "\n".join(lines)

    def shock_paths(self) -> pd.DataFrame:
        """Return DataFrame of required structural shocks."""
        ctrl_cols = [s for s in self.controlled_shocks if s in self.shocks.columns]
        return self.shocks[ctrl_cols].copy() if ctrl_cols else self.shocks.copy()

    def to_frame(self) -> pd.DataFrame:
        """Return canonical DataFrame representation of the conditional forecast."""
        return self.forecast.copy()

    def plot(
        self,
        variables: Sequence[str] | None = None,
        *,
        ax=None,
        show_baseline: bool = True,
        show_bands: bool = True,
        figsize: tuple[float, float] | None = None,
    ):
        """Plot conditional forecast paths with baseline and target points."""
        import matplotlib.pyplot as plt

        plot_vars = list(variables) if variables is not None else (
            list(self.conditions.keys()) if self.conditions else list(self.forecast.columns[:min(4, len(self.forecast.columns))])
        )

        n_plots = len(plot_vars)
        if ax is None:
            if n_plots == 1:
                fig, ax_arr = plt.subplots(figsize=figsize or (8, 4.5))
                axes = [ax_arr]
            else:
                ncols = 2 if n_plots > 1 else 1
                nrows = (n_plots + ncols - 1) // ncols
                fig, ax_arr = plt.subplots(nrows, ncols, figsize=figsize or (10, 3.5 * nrows), squeeze=False)
                axes = ax_arr.flatten()
        else:
            fig = ax.figure
            axes = [ax] * n_plots

        for i, var in enumerate(plot_vars):
            axi = axes[i]
            if var in self.forecast.columns:
                axi.plot(self.forecast.index, self.forecast[var], label="Conditional", color="#1f77b4", linewidth=2.0)
            if show_baseline and self.baseline is not None and var in self.baseline.columns:
                axi.plot(self.baseline.index, self.baseline[var], label="Baseline", color="#7f7f7f", linestyle="--", linewidth=1.5)
            if var in self.conditions:
                targets = self.conditions[var]
                for h, val in enumerate(targets):
                    if val is not None and not np.isnan(val) and h < len(self.forecast):
                        axi.scatter([self.forecast.index[h]], [val], color="#d62728", marker="x", s=60, zorder=5, label="Target" if h == 0 else None)
            if show_bands and self.bands is not None:
                if "lower" in self.bands and "upper" in self.bands and var in self.bands["lower"].columns:
                    axi.fill_between(self.forecast.index, self.bands["lower"][var], self.bands["upper"][var], color="#1f77b4", alpha=0.2, label="Confidence Band")
            axi.set_title(var, fontweight="bold")
            axi.set_xlabel("Horizon")
            axi.grid(True, linestyle=":", alpha=0.6)
            axi.legend(loc="best", fontsize=9)

        if ax is None and n_plots < len(axes):
            for j in range(n_plots, len(axes)):
                fig.delaxes(axes[j])
        fig.tight_layout()
        return fig

    def to_markdown(self, **kwargs) -> str:
        """Render forecast table as Markdown."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Render forecast table as LaTeX tabular."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Render forecast table as Typst table."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), **kwargs)


@dataclass(frozen=True)
class ShockDecompositionResult:
    """Container for grouped shock decomposition results.

    Attributes
    ----------
    groups : dict[str, tuple[str, ...]]
        Mapping from shock group names to tuples of structural shock names.
    decomposition : dict[str, pd.DataFrame]
        Mapping from component names (group names, 'Others', 'initial', 'residual')
        to DataFrames of shape (T, n_variables).
    components : dict[str, pd.DataFrame]
        Alias for decomposition.
    variables : tuple[str, ...]
        Tuple of endogenous variable names.
    shock_names : tuple[str, ...]
        Tuple of structural shock names in the underlying model.
    actual : pd.DataFrame | None
        Observed or simulated actual data path, if available.
    initial_state : pd.DataFrame | None
        Initial state decay and steady-state contribution path.
    """

    groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    components: dict[str, pd.DataFrame] = field(default_factory=dict)
    decomposition: dict[str, pd.DataFrame] = field(default_factory=dict)
    variables: tuple[str, ...] = ()
    shock_names: tuple[str, ...] = ()
    actual: pd.DataFrame | None = None
    initial_state: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        target_dict = self.components if self.components else self.decomposition
        if not target_dict:
            raise ValueError("Either 'components' or 'decomposition' must be provided.")

        object.__setattr__(self, "components", target_dict)
        object.__setattr__(self, "decomposition", target_dict)

        norm_groups = {k: tuple(v) if not isinstance(v, tuple) else v for k, v in self.groups.items()}
        object.__setattr__(self, "groups", norm_groups)

        if not self.variables and target_dict:
            first_df = next(iter(target_dict.values()))
            object.__setattr__(self, "variables", tuple(first_df.columns))

        if self.initial_state is None:
            init_df = target_dict.get("initial", target_dict.get("initial_state", target_dict.get("initial_condition")))
            object.__setattr__(self, "initial_state", init_df)

        if not self.shock_names:
            all_s = []
            for s_tuple in norm_groups.values():
                all_s.extend(s_tuple)
            object.__setattr__(self, "shock_names", tuple(dict.fromkeys(all_s)))

        if self.actual is not None:
            for var in self.variables:
                if var not in self.actual.columns:
                    continue
                total = np.zeros(len(self.actual))
                for g_df in target_dict.values():
                    if var in g_df.columns:
                        total += g_df[var].to_numpy(dtype=float)
                act = self.actual[var].to_numpy(dtype=float)
                finite = np.isfinite(act)
                if finite.any():
                    err = float(np.max(np.abs(total[finite] - act[finite])))
                    scale = float(np.max(np.abs(act[finite])))
                    tol = max(1e-10 * max(scale, 1.0), 1e-12)
                    if err > tol:
                        raise ValueError(
                            f"Shock decomposition adding-up invariant violated for '{var}': "
                            f"sum(components) differs from actual by max={err:.3e} (tolerance {tol:.3e})."
                        )

    def to_frame(self, variable: str | None = None, *, var: str | None = None) -> pd.DataFrame:
        """Return the decomposition table as a DataFrame."""
        effective_var = var if var is not None else variable
        if effective_var is not None:
            if effective_var not in self.variables:
                found = any(effective_var in comp_df.columns for comp_df in self.components.values())
                if not found:
                    raise KeyError(f"Variable '{effective_var}' not in decomposition variables: {self.variables}")
            data = {}
            for comp_name, comp_df in self.components.items():
                if effective_var in comp_df.columns:
                    data[comp_name] = comp_df[effective_var]
            if self.actual is not None and effective_var in self.actual.columns:
                data["actual"] = self.actual[effective_var]
            first_df = next(iter(self.components.values()))
            return pd.DataFrame(data, index=first_df.index)

        if len(self.variables) == 1:
            return self.to_frame(self.variables[0])

        frames = {v: self.to_frame(v) for v in self.variables}
        return pd.concat(frames, axis=1)

    def summary(self, variable: str | None = None, *, var: str | None = None) -> str:
        """Render a publication-ready text summary of the shock decomposition."""
        effective_var = var if var is not None else variable
        target_var = effective_var if effective_var is not None else (self.variables[0] if self.variables else "Decomposition")
        lines = [
            f"Shock Decomposition: {target_var}",
            "=" * 72,
            "Declared Groups:",
        ]
        for g, shocks in self.groups.items():
            lines.append(f"  - {g}: {', '.join(shocks)}")
        lines.append("-" * 72)

        try:
            df = self.to_frame(target_var)
            lines.append(df.round(6).to_string())
        except Exception:
            for g, df in self.components.items():
                lines.append(f"[{g}]:\n{df.head().to_string()}\n")

        lines.append("=" * 72)
        return "\n".join(lines)

    def to_markdown(self, variable: str | None = None, *, var: str | None = None, **kwargs) -> str:
        """Export decomposition table to Markdown."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(var=var if var is not None else variable), **kwargs)

    def to_latex(self, variable: str | None = None, *, var: str | None = None, **kwargs) -> str:
        """Export decomposition table to LaTeX tabular."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(var=var if var is not None else variable), **kwargs)

    def to_typst(self, variable: str | None = None, *, var: str | None = None, **kwargs) -> str:
        """Export decomposition table to Typst table."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(var=var if var is not None else variable), **kwargs)

    def plot(
        self,
        variable: str | None = None,
        style: str = "publication",
        ax=None,
        *,
        var: str | None = None,
    ):
        """Generate stacked-bar chart of shock group contributions overlaid with actual series."""
        import matplotlib.pyplot as plt
        effective_var = var if var is not None else variable
        target_var = effective_var if effective_var is not None else (self.variables[0] if self.variables else "y")
        df = self.to_frame(target_var)
        T = len(df)
        t = np.arange(T) if isinstance(df.index, pd.RangeIndex) else df.index

        width = 0.8
        if ax is None:
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
        else:
            fig = ax.figure

        comp_cols = [c for c in df.columns if c != "actual"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        pos_bottom = np.zeros(T)
        neg_bottom = np.zeros(T)

        for i, col in enumerate(comp_cols):
            color = colors[i % len(colors)]
            vals = df[col].to_numpy(dtype=float)
            pos_vals = np.where(vals > 0, vals, 0.0)
            neg_vals = np.where(vals < 0, vals, 0.0)

            ax.bar(t, pos_vals, width=width, bottom=pos_bottom, color=color, label=col, alpha=0.85)
            ax.bar(t, neg_vals, width=width, bottom=neg_bottom, color=color, alpha=0.85)

            pos_bottom += pos_vals
            neg_bottom += neg_vals

        if "actual" in df.columns:
            ax.plot(t, df["actual"].to_numpy(dtype=float), color="black", linewidth=1.6, label="Actual")

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(f"Shock Decomposition: {target_var}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Contribution", fontsize=10)
        ax.legend(loc="best", frameon=True, fontsize=8)
        fig.tight_layout()
        return fig


@dataclass(frozen=True)
class BayesianIRFResult:
    """Posterior impulse response functions and credible interval bands.

    Attributes
    ----------
    median : pd.DataFrame | dict[str, pd.DataFrame]
        Posterior median IRF.
    bands : dict[Any, tuple[pd.DataFrame, pd.DataFrame]]
        Mapping from band level (e.g. 0.68, 0.90, 0.95) to (lower_df, upper_df).
    quantiles : dict[float, pd.DataFrame]
        Mapping from quantile level (e.g. 0.05, 0.16, 0.50, 0.84, 0.95) to DataFrame.
    draws : np.ndarray | None
        Raw IRF draws array.
    variables : tuple[str, ...]
        Endogenous variable names.
    shocks : tuple[str, ...]
        Shock names.
    horizon : int
        Impulse response horizon.
    n_draws : int
        Total number of parameter draws evaluated.
    n_valid : int
        Number of determinate parameter draws retained.
    determinacy_rate : float
        Fraction of draws satisfying Blanchard-Kahn conditions (n_valid / n_draws).
    """

    median: pd.DataFrame | dict[str, pd.DataFrame]
    bands: dict[Any, tuple[pd.DataFrame, pd.DataFrame]] = field(default_factory=dict)
    quantiles: dict[float, pd.DataFrame] = field(default_factory=dict)
    draws: np.ndarray | None = None
    variables: tuple[str, ...] = ()
    shocks: tuple[str, ...] = ()
    horizon: int = 0
    n_draws: int = 0
    n_valid: int = 0
    determinacy_rate: float = 1.0

    def __post_init__(self) -> None:
        if not self.variables:
            if isinstance(self.median, pd.DataFrame):
                object.__setattr__(self, "variables", tuple(self.median.columns))
            elif isinstance(self.median, dict) and self.median:
                first_df = next(iter(self.median.values()))
                object.__setattr__(self, "variables", tuple(first_df.columns))
        if self.horizon == 0:
            if isinstance(self.median, pd.DataFrame):
                object.__setattr__(self, "horizon", len(self.median) - 1)
            elif isinstance(self.median, dict) and self.median:
                first_df = next(iter(self.median.values()))
                object.__setattr__(self, "horizon", len(first_df) - 1)

    @property
    def shock_names(self) -> tuple[str, ...]:
        """Alias for shocks."""
        return self.shocks

    @property
    def shock(self) -> str:
        """Alias for the first shock name."""
        return self.shocks[0] if self.shocks else ""

    @property
    def periods(self) -> int:
        """Alias for horizon."""
        return self.horizon

    def to_frame(self, shock: str | None = None) -> pd.DataFrame:
        """Return median IRF DataFrame."""
        if isinstance(self.median, pd.DataFrame):
            return self.median.copy()
        if isinstance(self.median, dict):
            if shock is not None:
                return self.median[shock].copy()
            if len(self.median) == 1:
                return next(iter(self.median.values())).copy()
            return pd.concat(self.median, axis=1)
        raise ValueError("Invalid median structure")

    def summary(self, shock: str | None = None) -> str:
        """Render publication-grade text summary of Bayesian IRF."""
        lines = [
            "BAYESIAN IMPULSE RESPONSE FUNCTIONS",
            "=" * 72,
            f"Horizon             : {self.horizon} periods",
            f"Total Draws         : {self.n_draws}",
            f"Determinate Draws   : {self.n_valid}",
            f"Determinacy Rate    : {self.determinacy_rate * 100:.1f}%",
            f"Credible Bands      : {list(self.bands.keys())}",
            "-" * 72,
            "Posterior Median IRF:",
        ]
        lines.append(self.to_frame(shock).round(6).to_string())
        lines.append("=" * 72)
        return "\n".join(lines)

    def plot(
        self,
        variables: Sequence[str] | None = None,
        shock: str | None = None,
        bands: Sequence[float] = (0.68, 0.90, 0.95),
        *,
        ax=None,
        figsize: tuple[float, float] | None = None,
    ):
        """Plot Bayesian IRF fan chart with median line and shaded credible intervals."""
        import matplotlib.pyplot as plt

        med_df = self.to_frame(shock)
        plot_vars = list(variables) if variables is not None else list(med_df.columns[:min(6, len(med_df.columns))])
        n_plots = len(plot_vars)

        if ax is None:
            if n_plots == 1:
                fig, ax_arr = plt.subplots(figsize=figsize or (8, 4.5))
                axes = [ax_arr]
            else:
                ncols = 2 if n_plots > 1 else 1
                nrows = (n_plots + ncols - 1) // ncols
                fig, ax_arr = plt.subplots(nrows, ncols, figsize=figsize or (10, 3.2 * nrows), squeeze=False)
                axes = ax_arr.flatten()
        else:
            fig = ax.figure
            axes = [ax] * n_plots

        alphas = {0.68: 0.35, 0.90: 0.22, 0.95: 0.12}

        for i, var in enumerate(plot_vars):
            axi = axes[i]
            if var in med_df.columns:
                sorted_bands = sorted(bands, reverse=True)
                for b in sorted_bands:
                    if b in self.bands:
                        low_df, up_df = self.bands[b]
                        if var in low_df.columns and var in up_df.columns:
                            alpha = alphas.get(b, 0.15)
                            axi.fill_between(
                                med_df.index,
                                low_df[var],
                                up_df[var],
                                color="#1f77b4",
                                alpha=alpha,
                                label=f"{int(b*100)}% Band",
                            )
                axi.plot(med_df.index, med_df[var], color="#08519c", linewidth=2.0, label="Median")
                axi.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
                axi.set_title(var, fontweight="bold")
                axi.set_xlabel("Horizon")
                axi.grid(True, linestyle=":", alpha=0.6)
                axi.legend(loc="best", fontsize=8)

        if ax is None and n_plots < len(axes):
            for j in range(n_plots, len(axes)):
                fig.delaxes(axes[j])
        fig.tight_layout()
        return fig

    def to_markdown(self, shock: str | None = None, **kwargs) -> str:
        """Export median IRF table to Markdown."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(shock), **kwargs)

    def to_latex(self, shock: str | None = None, **kwargs) -> str:
        """Export median IRF table to LaTeX tabular."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(shock), **kwargs)

    def to_typst(self, shock: str | None = None, **kwargs) -> str:
        """Export median IRF table to Typst table."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(shock), **kwargs)


@dataclass(frozen=True)
class PriorPredictiveResult:
    """Result of prior predictive simulation.

    Attributes
    ----------
    prior_moments : pd.DataFrame
        Distribution summary (mean, std, 5%, 50%, 95%) of theoretical moments across prior draws.
    prior_draws : pd.DataFrame
        Sampled parameter vectors (n_draws x n_params).
    valid_draws : pd.DataFrame
        Parameter vectors satisfying Blanchard-Kahn determinacy.
    param_names : tuple[str, ...]
        Names of sampled parameters.
    determinacy_rate : float
        Fraction of prior parameter draws that yielded unique stable solutions.
    n_draws : int
        Total draws sampled.
    n_valid : int
        Determinate draws solved.
    irfs : Any | None
        Impulse responses evaluated across prior draws.
    priors : dict[str, Any] | None
        Dictionary of prior specifications used.
    """

    prior_moments: pd.DataFrame
    prior_draws: pd.DataFrame = field(default_factory=pd.DataFrame)
    valid_draws: pd.DataFrame = field(default_factory=pd.DataFrame)
    param_names: tuple[str, ...] = ()
    determinacy_rate: float = 1.0
    n_draws: int = 0
    n_valid: int = 0
    irfs: Any | None = None
    priors: dict[str, Any] | None = None

    @property
    def moments(self) -> pd.DataFrame:
        """Alias for prior_moments."""
        return self.prior_moments

    @property
    def theoretical_moments(self) -> pd.DataFrame:
        """Alias for prior_moments."""
        return self.prior_moments

    @property
    def parameter_draws(self) -> pd.DataFrame:
        """Alias for prior_draws."""
        return self.prior_draws

    def to_frame(self) -> pd.DataFrame:
        """Return canonical DataFrame representation of prior moments."""
        return self.prior_moments.copy()

    def summary(self) -> str:
        """Render publication-grade text summary of prior predictive analysis."""
        lines = [
            "Prior Predictive Analysis",
            "=" * 72,
            f"Total Draws         : {self.n_draws}",
            f"Determinate Draws   : {self.n_valid}",
            f"Determinacy Rate    : {self.determinacy_rate * 100:.1f}%",
            f"Parameters          : {', '.join(self.param_names)}",
            "-" * 72,
            "PRIOR PREDICTIVE MOMENTS DISTRIBUTION:",
            self.prior_moments.round(6).to_string(),
            "=" * 72,
        ]
        return "\n".join(lines)

    def plot(
        self,
        variables: Sequence[str] | None = None,
        *,
        ax=None,
        figsize: tuple[float, float] | None = None,
    ):
        """Plot prior predictive moment distributions."""
        import matplotlib.pyplot as plt

        df = self.to_frame()
        plot_vars = list(variables) if variables is not None else list(df.index[:min(6, len(df.index))])
        n_plots = len(plot_vars)

        if ax is None:
            if n_plots == 1:
                fig, ax_arr = plt.subplots(figsize=figsize or (8, 4.5))
                axes = [ax_arr]
            else:
                ncols = 2 if n_plots > 1 else 1
                nrows = (n_plots + ncols - 1) // ncols
                fig, ax_arr = plt.subplots(nrows, ncols, figsize=figsize or (10, 3.2 * nrows), squeeze=False)
                axes = ax_arr.flatten()
        else:
            fig = ax.figure
            axes = [ax] * n_plots

        for i, var in enumerate(plot_vars):
            axi = axes[i]
            if var in df.index:
                row = df.loc[var]
                cols = [c for c in ["5%", "50%", "95%"] if c in df.columns]
                if not cols:
                    cols = [c for c in df.columns if c in ["mean", "std"]]
                axi.bar(cols, [row[c] for c in cols], color="#1f77b4", alpha=0.7)
                axi.set_title(f"Prior Predictive Moments: {var}", fontweight="bold")
                axi.grid(True, linestyle=":", alpha=0.6)

        if ax is None and n_plots < len(axes):
            for j in range(n_plots, len(axes)):
                fig.delaxes(axes[j])
        fig.tight_layout()
        return fig

    def to_markdown(self, **kwargs) -> str:
        """Export prior moments table to Markdown."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export prior moments table to LaTeX tabular."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Export prior moments table to Typst table."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), **kwargs)


class CallableDataFrame(pd.DataFrame):
    """DataFrame subclass that also allows invocation as a nullary function."""

    @property
    def _constructor(self):
        return CallableDataFrame

    def __call__(self) -> pd.DataFrame:
        return self

    def __deepcopy__(self, memo):
        return CallableDataFrame(copy.deepcopy(pd.DataFrame(self), memo))


@dataclass(frozen=True)
class ModelParityResult:
    """Individual DSGE model Dynare parity verification result."""

    model_name: str
    order: int
    n_vars: int
    n_shocks: int
    passed: bool
    status: str
    max_dev_ghx: float
    max_dev_ghu: float
    max_dev_ghxx: float = 0.0
    max_dev_ghs2: float = 0.0
    max_dev_moments: float = 0.0
    max_dev_irf: float = 0.0
    max_dev_dr: float = 0.0
    score: float = 100.0
    runtime_sec: float = 0.0
    dr_diff: pd.DataFrame = field(default_factory=pd.DataFrame)
    moments_diff: pd.DataFrame = field(default_factory=pd.DataFrame)
    tolerances: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Return model deviation scorecard summary."""
        if self.dr_diff is not None and not self.dr_diff.empty:
            return self.dr_diff.copy()
        return pd.DataFrame([{
            "model": self.model_name,
            "order": self.order,
            "status": self.status,
            "score": f"{self.score:.1f}%",
            "max_dev_ghx": self.max_dev_ghx,
            "max_dev_ghu": self.max_dev_ghu,
            "max_dev_ghxx": self.max_dev_ghxx,
            "max_dev_ghs2": self.max_dev_ghs2,
            "max_dev_moments": self.max_dev_moments,
            "time_s": round(self.runtime_sec, 4),
        }])

    def summary(self) -> str:
        """Render individual model parity summary string."""
        lines = [
            f"Dynare Parity Report: {self.model_name} (Order {self.order})",
            "=" * 72,
            f"Status       : {self.status} ({'PASS' if self.passed else 'FAIL'})",
            f"Parity Score : {self.score:.1f}%",
            f"Variables    : {self.n_vars} endogenous, {self.n_shocks} exogenous",
            f"Max Dev ghx  : {self.max_dev_ghx:.4e}",
            f"Max Dev ghu  : {self.max_dev_ghu:.4e}",
        ]
        if self.order >= 2:
            lines.append(f"Max Dev ghxx : {self.max_dev_ghxx:.4e}")
            lines.append(f"Max Dev ghs2 : {self.max_dev_ghs2:.4e}")
        lines.append(f"Max Dev mom  : {self.max_dev_moments:.4e}")
        lines.append(f"Runtime      : {self.runtime_sec:.4f}s")
        lines.append("=" * 72)
        return "\n".join(lines)

    def to_markdown(self, **kwargs) -> str:
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), **kwargs)

    def plot(
        self,
        variables: Sequence[str] | None = None,
        *,
        figsize: tuple[float, float] | None = None,
        ax: Any | None = None,
        style: str = "default",
    ) -> Any:
        import matplotlib.pyplot as plt
        df = self.dr_diff
        if df is None or df.empty:
            df = self.to_frame()

        if ax is None:
            fig, target_ax = plt.subplots(figsize=figsize or (8.0, 4.5))
        else:
            fig = ax.figure
            target_ax = ax

        if "dev_ghx" in df.columns:
            display_df = df.head(15) if variables is None else df.loc[[v for v in variables if v in df.index]]
            y_pos = np.arange(len(display_df))
            devs = display_df["dev_ghx"].to_numpy(dtype=float)
            tol = self.tolerances.get("ghx", 1e-6)
            log_devs = np.log10(np.maximum(devs, 1e-16))
            log_tol = np.log10(tol)

            colors = ["#2b8cbe" if d <= tol else "#e41a1c" for d in devs]
            if style in ("publication", "grayscale"):
                colors = ["0.3" if d <= tol else "0.7" for d in devs]

            target_ax.barh(y_pos, log_devs, color=colors, height=0.55, align="center")
            target_ax.axvline(log_tol, color="red", linestyle="--", linewidth=1.2, label=f"Tolerance ({tol:.1e})")
            target_ax.set_yticks(y_pos)
            target_ax.set_yticklabels(display_df.index)
            target_ax.set_xlabel(r"$\log_{10}(\text{Absolute Deviation in } ghx)$")
            target_ax.legend(loc="best", frameon=False)
        else:
            target_ax.text(
                0.5, 0.5,
                f"Parity: {self.status} ({self.score:.1f}%)",
                ha="center", va="center", fontsize=14, fontweight="bold",
            )
            target_ax.axis("off")

        target_ax.set_title(f"Parity Deviations: {self.model_name}", fontsize=11, fontweight="bold")
        target_ax.spines["top"].set_visible(False)
        target_ax.spines["right"].set_visible(False)
        target_ax.grid(True, linestyle=":", alpha=0.5)

        if ax is None:
            fig.tight_layout()
            return fig
        return target_ax


@dataclass
class ParityDashboardResult:
    """Dynare parity verification result and publication scorecard."""

    passed: bool = True
    score: float = 100.0
    dr_diff: pd.DataFrame = field(default_factory=pd.DataFrame)
    moments_diff: pd.DataFrame = field(default_factory=pd.DataFrame)
    tolerances: dict[str, float] = field(default_factory=dict)
    model_name: str = "model"
    scorecard: Any = None
    total_models: int = 1
    passed_models: int = 1
    failed_models: int = 0
    max_dev_ghx: float = 0.0
    max_dev_ghu: float = 0.0
    max_dev_ghxx: float = 0.0
    max_dev_ghs2: float = 0.0
    max_dev_moments: float = 0.0
    max_dev_dr: float = 0.0
    results: list[ModelParityResult] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.max_dev_dr == 0.0:
            valid = [float(v) for v in (self.max_dev_ghx, self.max_dev_ghu, self.max_dev_ghxx) if v is not None and not np.isnan(v)]
            self.max_dev_dr = max(valid) if valid else 0.0

        if self.scorecard is not None:
            if not isinstance(self.scorecard, CallableDataFrame):
                self.scorecard = CallableDataFrame(self.scorecard)
        else:
            if self.results:
                rows = []
                for r in self.results:
                    rows.append({
                        "model": getattr(r, "model_name", "model"),
                        "order": getattr(r, "order", 1),
                        "status": getattr(r, "status", ("PASS" if getattr(r, "passed", False) else "FAIL")),
                        "score": f"{getattr(r, 'score', 0.0):.1f}%",
                        "max_dev_dr": getattr(r, "max_dev_dr", 0.0),
                        "max_dev_moments": getattr(r, "max_dev_moments", 0.0),
                        "time_s": round(getattr(r, "runtime_sec", 0.0), 4),
                    })
                self.scorecard = CallableDataFrame(rows)
            else:
                self.scorecard = CallableDataFrame([{
                    "model": self.model_name,
                    "order": self.details.get("order", 1),
                    "status": "PASS" if self.passed else "FAIL",
                    "score": f"{self.score:.1f}%",
                    "max_dev_dr": self.max_dev_dr,
                    "max_dev_moments": self.max_dev_moments,
                    "time_s": round(self.details.get("runtime_sec", 0.0), 4),
                }])

    def to_frame(self) -> pd.DataFrame:
        """Return scorecard table as a DataFrame."""
        return self.scorecard.copy()

    def summary(self) -> str:
        """Render comprehensive Dynare Parity Dashboard summary."""
        lines = [
            "=" * 72,
            "  Dynare Parity Dashboard Scorecard",
            "=" * 72,
            f"Overall Status : {'PASS' if self.passed else 'FAIL'}",
            f"Parity Score   : {self.score:.1f}%",
            f"Models Tested  : {self.total_models} (Passed: {self.passed_models}, Failed: {self.failed_models})",
            "",
            "Tolerances:",
        ]
        if self.tolerances:
            for k, v in sorted(self.tolerances.items()):
                lines.append(f"  {k:<12s}: {v:.1e}")
        else:
            lines.append("  (none specified)")

        lines.extend([
            "",
            "Maximum Absolute Deviations:",
            f"  ghx          : {self.max_dev_ghx:12.4e}  (tol: {self.tolerances.get('ghx', 1e-6):.1e})",
            f"  ghu          : {self.max_dev_ghu:12.4e}  (tol: {self.tolerances.get('ghu', 1e-6):.1e})",
        ])
        if self.max_dev_ghxx > 0.0 or "ghxx" in self.tolerances:
            lines.append(f"  ghxx         : {self.max_dev_ghxx:12.4e}  (tol: {self.tolerances.get('ghxx', 1e-4):.1e})")
        if self.max_dev_ghs2 > 0.0 or "ghs2" in self.tolerances:
            lines.append(f"  ghs2         : {self.max_dev_ghs2:12.4e}  (tol: {self.tolerances.get('ghs2', 1e-4):.1e})")
        if self.max_dev_moments > 0.0 or "mean" in self.tolerances or "var" in self.tolerances:
            lines.append(f"  moments      : {self.max_dev_moments:12.4e}  (tol: {self.tolerances.get('mean', 1e-5):.1e})")

        lines.extend([
            "",
            "Scorecard Table:",
            "-" * 72,
            self.scorecard.to_string(index=False),
            "=" * 72,
        ])
        return "\n".join(lines)

    def to_markdown(self, **kwargs) -> str:
        """Export scorecard table to Markdown format."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export scorecard table to LaTeX tabular format."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Export scorecard table to Typst table format."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), **kwargs)

    def plot(
        self,
        variables: Sequence[str] | None = None,
        *,
        figsize: tuple[float, float] | None = None,
        ax: Any | None = None,
        style: str = "default",
    ) -> Any:
        """Generate publication-grade deviation comparison plot across variables."""
        import matplotlib.pyplot as plt

        df = self.dr_diff
        if df is None or df.empty:
            df = self.to_frame()

        if ax is None:
            fig, target_ax = plt.subplots(figsize=figsize or (8.0, 4.5))
        else:
            fig = ax.figure
            target_ax = ax

        if "dev_ghx" in df.columns:
            display_df = df.head(15) if variables is None else df.loc[[v for v in variables if v in df.index]]
            y_pos = np.arange(len(display_df))
            devs = display_df["dev_ghx"].to_numpy(dtype=float)
            tol = self.tolerances.get("ghx", 1e-6)
            log_devs = np.log10(np.maximum(devs, 1e-16))
            log_tol = np.log10(tol)

            colors = ["#2b8cbe" if d <= tol else "#e41a1c" for d in devs]
            if style in ("publication", "grayscale"):
                colors = ["0.3" if d <= tol else "0.7" for d in devs]

            target_ax.barh(y_pos, log_devs, color=colors, height=0.55, align="center")
            target_ax.axvline(log_tol, color="red", linestyle="--", linewidth=1.2, label=f"Tolerance ({tol:.1e})")
            target_ax.set_yticks(y_pos)
            target_ax.set_yticklabels(display_df.index)
            target_ax.set_xlabel(r"$\log_{10}(\text{Absolute Deviation in } ghx)$")
            target_ax.legend(loc="best", frameon=False)
        else:
            target_ax.text(
                0.5, 0.5,
                f"Dynare Parity Score: {self.score:.1f}%\nStatus: {'PASS' if self.passed else 'FAIL'}",
                ha="center", va="center", fontsize=14, fontweight="bold",
            )
            target_ax.axis("off")

        target_ax.set_title(f"Dynare Parity Dashboard: {self.model_name}", fontsize=11, fontweight="bold")
        target_ax.spines["top"].set_visible(False)
        target_ax.spines["right"].set_visible(False)
        target_ax.grid(True, linestyle=":", alpha=0.5)

        if ax is None:
            fig.tight_layout()
            return fig
        return target_ax






