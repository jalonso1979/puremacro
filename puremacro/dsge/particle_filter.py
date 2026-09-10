"""Pure-Python Vectorized Particle Filtering for Nonlinear & Stochastic Volatility DSGEs.

Provides Sequential Monte Carlo particle filters for DSGE models:
- Bootstrap Particle Filter (BPF) (Gordon et al. 1993; Fernández-Villaverde & Rubio-Ramírez 2007)
- Auxiliary Particle Filter (APF) (Pitt & Shephard 1999) with predictive covariance
  Sigma_mu = Z_D Q Z_D' + H to prevent auxiliary proposal collapse
- Vectorized state propagation over N particles with ZERO Python loops across particles
- Exact nonlinear likelihood evaluation for 2nd and 3rd order pruned perturbation models
  (PrunedDSGESolution, Order3PrunedSolution)
- Stochastic Volatility: sigma_{j, t} = bar{sigma}_j exp(h_{j, t}), h_{j, t} = rho_j h_{j, t-1} + sigma_eta eta_{j, t}
  with state augmentation tracking precautionary saving shifts and skewness
- Fat-tailed innovations (Student-t, Gaussian mixture) and multivariate Student-t observation densities
- Vectorized resampling schemes: systematic, stratified, residual, multinomial in O(N)
- Standardized presentation contract: ParticleFilterResult with .summary(), .plot(),
  .to_markdown(), .to_latex(), .to_typst()

Strictly adheres to the Pyodide four-package contract: numpy, scipy, pandas, matplotlib only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.special
from matplotlib.figure import Figure

from puremacro.plot import _new_ax
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst


# ---------------------------------------------------------------------------
# Specifications & Result Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, init=False)
class StochasticVolatilitySpec:
    """Specification of time-varying shock volatility dynamics.

    Model:
        h_{j, t} = rho_j * h_{j, t-1} + sigma_eta_j * eta_{j, t},   eta_{j, t} ~ N(0, 1)
        sigma_{j, t} = base_scale_j * exp(h_{j, t})

    Parameters
    ----------
    rho : float | np.ndarray | dict[str, float]
        Persistence of log-volatility AR(1) processes (supports alias rho_h).
    sigma_eta : float | np.ndarray | dict[str, float]
        Standard deviation of volatility innovations (supports alias sigma_h).
    base_scale : float | np.ndarray | dict[str, float] | None, optional
        Base standard deviation bar{sigma}_j for each shock (supports alias sigma_bar).
        If None, inferred from shock covariance Q.
    h0 : float | np.ndarray | str | None, optional
        Initial log-volatilities. If "stationary" or None, drawn from stationary
        distribution N(0, sigma_eta^2 / (1 - rho^2)).
    """
    rho: np.ndarray | float | dict[str, float]
    sigma_eta: np.ndarray | float | dict[str, float]
    base_scale: np.ndarray | float | dict[str, float] | None = None
    h0: np.ndarray | float | str | None = None

    def __init__(
        self,
        rho: np.ndarray | float | dict[str, float] | None = None,
        sigma_eta: np.ndarray | float | dict[str, float] | None = None,
        base_scale: np.ndarray | float | dict[str, float] | None = None,
        h0: np.ndarray | float | str | None = None,
        *,
        rho_h: np.ndarray | float | dict[str, float] | None = None,
        sigma_h: np.ndarray | float | dict[str, float] | None = None,
        sigma_bar: np.ndarray | float | dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        actual_rho = rho if rho is not None else (rho_h if rho_h is not None else kwargs.get("rho_h", None))
        actual_sigma_eta = sigma_eta if sigma_eta is not None else (sigma_h if sigma_h is not None else kwargs.get("sigma_h", None))
        actual_base_scale = base_scale if base_scale is not None else (sigma_bar if sigma_bar is not None else kwargs.get("sigma_bar", None))
        actual_h0 = h0 if h0 is not None else kwargs.get("h0", None)

        if actual_rho is None:
            raise TypeError("StochasticVolatilitySpec missing required argument: 'rho' (or 'rho_h')")
        if actual_sigma_eta is None:
            raise TypeError("StochasticVolatilitySpec missing required argument: 'sigma_eta' (or 'sigma_h')")

        object.__setattr__(self, "rho", actual_rho)
        object.__setattr__(self, "sigma_eta", actual_sigma_eta)
        object.__setattr__(self, "base_scale", actual_base_scale)
        object.__setattr__(self, "h0", actual_h0)

    @property
    def rho_h(self) -> np.ndarray | float | dict[str, float]:
        """Alias for rho."""
        return self.rho

    @property
    def sigma_h(self) -> np.ndarray | float | dict[str, float]:
        """Alias for sigma_eta."""
        return self.sigma_eta

    @property
    def sigma_bar(self) -> np.ndarray | float | dict[str, float] | None:
        """Alias for base_scale."""
        return self.base_scale



@dataclass(frozen=True)
class ParticleFilterResult:
    """Standardized presentation result for DSGE particle filtering.

    Attributes
    ----------
    log_likelihood : float
        Total log-likelihood scalar ln L(Y | theta).
    log_likelihood_contributions : pd.Series
        Period-by-period incremental log-likelihood contributions ln p(y_t | y_{1:t-1}).
    filtered_states : pd.DataFrame
        Filtered mean state and control trajectories over time (T, n_vars).
    filtered_states_std : pd.DataFrame
        Filtered state standard deviation trajectories over time (T, n_vars).
    ess : pd.Series
        Effective Sample Size (ESS) trajectory across filtering periods.
    resample_history : pd.Series
        Boolean indicator series flagging dates where resampling was triggered.
    resampling_frequency : float
        Fraction of time steps where resampling occurred.
    n_particles : int
        Number of particles N used in filtering.
    method : str
        Filtering algorithm ("bootstrap" or "auxiliary").
    resampling_method : str
        Resampling algorithm used ("systematic", "stratified", "residual", "multinomial").
    model_name : str
        Model identifier.
    variable_names : tuple[str, ...]
        Names of all model variables reported in filtered states.
    observed_vars : tuple[str, ...]
        Names of observable variables matched to data.
    filtered_volatility : pd.DataFrame | None
        Filtered mean time-varying shock standard deviations (T, n_e), if SV is active.
    filtered_volatility_std : pd.DataFrame | None
        Filtered standard deviation of time-varying shock standard deviations (T, n_e).
    """

    log_likelihood: float
    log_likelihood_contributions: pd.Series
    filtered_states: pd.DataFrame
    filtered_states_std: pd.DataFrame
    ess: pd.Series
    resample_history: pd.Series
    resampling_frequency: float
    n_particles: int
    method: str
    resampling_method: str
    model_name: str
    variable_names: tuple[str, ...]
    observed_vars: tuple[str, ...]
    filtered_volatility: pd.DataFrame | None = None
    filtered_volatility_std: pd.DataFrame | None = None

    def summary(self) -> pd.DataFrame:
        """Return a structured summary table of the particle filter evaluation."""
        T_obs = len(self.log_likelihood_contributions)
        avg_ll = self.log_likelihood / max(1, T_obs)

        metrics = {
            "Log-Likelihood": f"{self.log_likelihood:.4f}",
            "Avg Log-Lik / Obs": f"{avg_ll:.4f}",
            "Observations (T)": f"{T_obs}",
            "Number of Particles (N)": f"{self.n_particles:,}",
            "Filter Method": self.method.capitalize(),
            "Resampling Scheme": self.resampling_method.capitalize(),
            "Resampling Events": f"{int(self.resample_history.sum())}",
            "Resampling Frequency": f"{self.resampling_frequency:.2%}",
            "Mean ESS": f"{self.ess.mean():.1f}",
            "Min ESS": f"{self.ess.min():.1f}",
            "Max ESS": f"{self.ess.max():.1f}",
        }
        if self.filtered_volatility is not None:
            metrics["Stochastic Volatility"] = "Active"

        return pd.DataFrame(list(metrics.values()), index=list(metrics.keys()), columns=["Value"])

    def to_markdown(self, **kwargs) -> str:
        """Export particle filter summary to Markdown format."""
        return _df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export particle filter summary to LaTeX tabular format."""
        return _df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Export particle filter summary to Typst table format."""
        return _df_to_typst(self.summary(), **kwargs)

    def plot(
        self,
        variables: Sequence[str] | str | None = None,
        ax: Any = None,
        figsize: tuple[float, float] | None = None,
        **kwargs,
    ) -> Figure:
        """Plot filtered state trajectories with credible intervals and ESS diagnostics.

        Parameters
        ----------
        variables : Sequence[str] | str | None, optional
            Variables to plot. If None, plots up to 4 variables (prioritizing observed vars).
        ax : matplotlib Axes, optional
            Axes to plot on.
        figsize : tuple[float, float], optional
            Figure size in inches.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if isinstance(variables, str):
            vars_to_plot = [variables]
        elif variables is not None:
            vars_to_plot = [v for v in variables if v in self.filtered_states.columns]
        else:
            # Prioritize observed variables, then other variables
            obs_in_states = [v for v in self.observed_vars if v in self.filtered_states.columns]
            other_vars = [v for v in self.filtered_states.columns if v not in obs_in_states]
            vars_to_plot = (obs_in_states + other_vars)[:4]

        n_vars = len(vars_to_plot)
        has_sv = self.filtered_volatility is not None
        n_plots = n_vars + 1 + (1 if has_sv else 0)

        if figsize is None:
            figsize = (9.5, 2.5 * n_plots)

        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        c_primary = "#1f77b4"
        c_secondary = "#ff7f0e"
        c_accent = "#2ca02c"

        x_vals = self.filtered_states.index

        # 1. State variable trajectories
        for i, var_name in enumerate(vars_to_plot):
            ax_i = axes[i]
            mean_series = self.filtered_states[var_name]
            std_series = self.filtered_states_std[var_name]
            upper = mean_series + 1.96 * std_series
            lower = mean_series - 1.96 * std_series

            ax_i.plot(x_vals, mean_series, color=c_primary, lw=1.6, label=f"Filtered {var_name}")
            ax_i.fill_between(x_vals, lower, upper, color=c_primary, alpha=0.20, label="95% Credible Band")
            ax_i.axhline(0.0, color="gray", linestyle=":", lw=0.8, alpha=0.7)
            ax_i.set_ylabel(var_name, fontweight="bold")
            ax_i.legend(loc="upper right", framealpha=0.8)
            ax_i.grid(True, linestyle="--", alpha=0.4)

        # 2. Stochastic Volatility panel (if present)
        idx_curr = n_vars
        if has_sv:
            ax_sv = axes[idx_curr]
            for col in self.filtered_volatility.columns:
                ax_sv.plot(x_vals, self.filtered_volatility[col], lw=1.4, label=f"Vol {col}")
            ax_sv.set_ylabel(r"$\sigma_t$", fontweight="bold")
            ax_sv.legend(loc="upper right", framealpha=0.8)
            ax_sv.grid(True, linestyle="--", alpha=0.4)
            idx_curr += 1

        # 3. Effective Sample Size (ESS) panel
        ax_ess = axes[idx_curr]
        ax_ess.plot(x_vals, self.ess, color=c_secondary, lw=1.5, label="Effective Sample Size (ESS)")
        thresh_val = 0.5 * self.n_particles
        ax_ess.axhline(thresh_val, color="crimson", linestyle="--", lw=1.2, label=f"Threshold ({thresh_val:.0f})")

        # Mark resampling events
        resample_dates = x_vals[self.resample_history]
        if len(resample_dates) > 0:
            resample_ess = self.ess[self.resample_history]
            ax_ess.scatter(resample_dates, resample_ess, color="crimson", s=20, zorder=5, label="Resampled")

        ax_ess.set_ylabel("ESS", fontweight="bold")
        ax_ess.set_xlabel("Period")
        ax_ess.set_ylim(0, self.n_particles * 1.05)
        ax_ess.legend(loc="lower right", framealpha=0.8)
        ax_ess.grid(True, linestyle="--", alpha=0.4)

        fig.suptitle(f"Particle Filter Diagnostics: {self.model_name} (N={self.n_particles:,})", fontweight="bold", y=0.995)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Vectorized Resampling Algorithms (Pure NumPy, O(N))
# ---------------------------------------------------------------------------

def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized systematic resampling in O(N) using uniform offset and searchsorted.

    Minimizes sampling variance across indices with a single uniform scalar u ~ U[0, 1/N).
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
    cumsum[-1] = 1.0
    return np.clip(np.searchsorted(cumsum, targets), 0, N - 1)


def stratified_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized stratified resampling in O(N) with N independent uniform offsets."""
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

    u = rng.uniform(0.0, 1.0, size=N)
    targets = (np.arange(N, dtype=float) + u) / N
    cumsum = np.cumsum(w)
    cumsum[-1] = 1.0
    return np.clip(np.searchsorted(cumsum, targets), 0, N - 1)


def residual_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized residual resampling in O(N) allocating deterministic integer copies first."""
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

    n_replic = np.floor(N * w).astype(int)
    R = N - int(np.sum(n_replic))
    deterministic_indices = np.repeat(np.arange(N), n_replic)

    if R > 0:
        res_w = (N * w - n_replic) / float(R)
        sum_rw = np.sum(res_w)
        if sum_rw <= 0.0 or not np.isfinite(sum_rw):
            res_w = np.full(N, 1.0 / N)
        else:
            res_w = res_w / sum_rw
        cumsum_r = np.cumsum(res_w)
        cumsum_r[-1] = 1.0
        u = float(rng.uniform(0.0, 1.0 / R))
        targets_r = u + np.arange(R, dtype=float) / R
        residual_indices = np.clip(np.searchsorted(cumsum_r, targets_r), 0, N - 1)
        return np.concatenate([deterministic_indices, residual_indices])
    return deterministic_indices


def multinomial_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized multinomial resampling in O(N) using sorted uniforms and searchsorted."""
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

    u = np.sort(rng.uniform(0.0, 1.0, size=N))
    cumsum = np.cumsum(w)
    cumsum[-1] = 1.0
    return np.clip(np.searchsorted(cumsum, u), 0, N - 1)


# ---------------------------------------------------------------------------
# Model Extraction Helpers
# ---------------------------------------------------------------------------

def _extract_model_tensors(model: Any) -> dict[str, Any]:
    """Extract first, second, and third order perturbation tensors from model."""
    from puremacro.dsge.pruning import Order3PrunedSolution, PrunedDSGESolution

    is_pruned = isinstance(model, (PrunedDSGESolution, Order3PrunedSolution))
    is_order3 = isinstance(model, Order3PrunedSolution) or (
        hasattr(model, "H_xxx") and model.H_xxx is not None
    )

    if is_pruned:
        state_names = tuple(model.state_names)
        control_names = tuple(model.control_names)
        shock_names = tuple(model.shock_names)
        variable_names = tuple(getattr(model, "variable_names", None) or (state_names + control_names))
        order_vars = state_names + control_names
        G = np.asarray(model.G, dtype=float)
        N = np.asarray(model.N, dtype=float)
        F = np.asarray(model.F, dtype=float)
        L = np.asarray(model.L, dtype=float)
        H_xx = np.asarray(model.H_xx, dtype=float)
        G_xx = np.asarray(model.G_xx, dtype=float)
        H_ss = np.asarray(model.H_sigmasigma, dtype=float).ravel()
        G_ss = np.asarray(model.G_sigmasigma, dtype=float).ravel()
        n_x, n_y, n_e = len(state_names), len(control_names), len(shock_names)

        H_xu = np.asarray(getattr(model, "H_xu", None) if getattr(model, "H_xu", None) is not None else np.zeros((n_x, n_x * n_e)), dtype=float)
        G_xu = np.asarray(getattr(model, "G_xu", None) if getattr(model, "G_xu", None) is not None else np.zeros((n_y, n_x * n_e)), dtype=float)
        H_uu = np.asarray(getattr(model, "H_uu", None) if getattr(model, "H_uu", None) is not None else np.zeros((n_x, n_e * n_e)), dtype=float)
        G_uu = np.asarray(getattr(model, "G_uu", None) if getattr(model, "G_uu", None) is not None else np.zeros((n_y, n_e * n_e)), dtype=float)

        steady_state = getattr(model, "steady_state", None)
        shock_cov = getattr(model, "shock_cov", None)
        model_name = getattr(model, "model_name", "pruned_dsge")

        tensors = {
            "G": G, "N": N, "F": F, "L": L,
            "H_xx": H_xx, "G_xx": G_xx, "H_ss": H_ss, "G_ss": G_ss,
            "H_xu": H_xu, "G_xu": G_xu, "H_uu": H_uu, "G_uu": G_uu,
            "state_names": state_names, "control_names": control_names,
            "shock_names": shock_names, "variable_names": variable_names,
            "order_vars": order_vars, "steady_state": steady_state,
            "shock_cov": shock_cov, "model_name": model_name,
            "is_order3": is_order3,
        }

        if is_order3:
            tensors.update({
                "H_xxx": np.asarray(model.H_xxx, dtype=float),
                "G_xxx": np.asarray(model.G_xxx, dtype=float),
                "H_xxu": np.asarray(model.H_xxu, dtype=float),
                "G_xxu": np.asarray(model.G_xxu, dtype=float),
                "H_xuu": np.asarray(model.H_xuu, dtype=float),
                "G_xuu": np.asarray(model.G_xuu, dtype=float),
                "H_uuu": np.asarray(model.H_uuu, dtype=float),
                "G_uuu": np.asarray(model.G_uuu, dtype=float),
                "H_x_ss": np.asarray(model.H_x_sigmasigma, dtype=float),
                "G_x_ss": np.asarray(model.G_x_sigmasigma, dtype=float),
                "H_u_ss": np.asarray(model.H_u_sigmasigma, dtype=float),
                "G_u_ss": np.asarray(model.G_u_sigmasigma, dtype=float),
            })
        return tensors

    # First-order model representations (LinearModel, MSDSGEResult, state-space models)
    if hasattr(model, "solution") and hasattr(model.solution, "G"):
        G = np.atleast_2d(np.asarray(model.solution.G, dtype=float))
        N = np.atleast_2d(np.asarray(model.solution.N, dtype=float))
        F = np.atleast_2d(np.asarray(getattr(model.solution, "F", np.zeros((0, G.shape[0]))), dtype=float))
        L = np.atleast_2d(np.asarray(getattr(model.solution, "L", np.zeros((F.shape[0], N.shape[1]))), dtype=float))
    elif hasattr(model, "solution") and hasattr(model.solution, "T"):
        G = np.atleast_2d(np.asarray(model.solution.T, dtype=float))
        N = np.atleast_2d(np.asarray(getattr(model.solution, "R", getattr(model.solution, "N", None)), dtype=float))
        F = np.atleast_2d(np.asarray(getattr(model.solution, "F", np.zeros((0, G.shape[0]))), dtype=float))
        L = np.atleast_2d(np.asarray(getattr(model.solution, "L", np.zeros((F.shape[0], N.shape[1]))), dtype=float))
    elif hasattr(model, "G") and hasattr(model, "N"):
        G = np.atleast_2d(np.asarray(model.G, dtype=float))
        N = np.atleast_2d(np.asarray(model.N, dtype=float))
        F = np.atleast_2d(np.asarray(getattr(model, "F", np.zeros((0, G.shape[0]))), dtype=float))
        L = np.atleast_2d(np.asarray(getattr(model, "L", np.zeros((F.shape[0], N.shape[1]))), dtype=float))
    elif hasattr(model, "T") and hasattr(model, "R"):
        G = np.atleast_2d(np.asarray(model.T, dtype=float))
        N = np.atleast_2d(np.asarray(model.R, dtype=float))
        F = np.atleast_2d(np.asarray(getattr(model, "F", np.zeros((0, G.shape[0]))), dtype=float))
        L = np.atleast_2d(np.asarray(getattr(model, "L", np.zeros((F.shape[0], N.shape[1]))), dtype=float))
    else:
        raise TypeError(f"Unsupported model type {type(model).__name__} for particle_filter")

    n_x_mat = G.shape[0]
    n_e_mat = N.shape[1] if N.ndim > 1 else 1
    n_y_mat = F.shape[0] if F.ndim > 1 else 0

    raw_states = tuple(getattr(model, "states", ()) or ())
    raw_controls = tuple(getattr(model, "controls", ()) or ())
    raw_shocks = tuple(getattr(model, "shocks", ()) or ())
    raw_variables = tuple(getattr(model, "variables", ()) or ())

    if len(raw_states) == n_x_mat:
        state_names = raw_states
    elif n_y_mat == 0 and len(raw_variables) == n_x_mat:
        state_names = raw_variables
    elif len(raw_variables) == n_x_mat + n_y_mat:
        state_names = raw_variables[:n_x_mat]
    else:
        state_names = tuple(f"x{i}" for i in range(n_x_mat))

    if len(raw_controls) == n_y_mat:
        control_names = raw_controls
    elif n_y_mat > 0 and len(raw_variables) == n_x_mat + n_y_mat:
        control_names = raw_variables[n_x_mat:]
    else:
        control_names = tuple(f"y{i}" for i in range(n_y_mat))

    if len(raw_shocks) == n_e_mat:
        shock_names = raw_shocks
    else:
        shock_names = tuple(f"eps{i}" for i in range(n_e_mat))

    order_vars = state_names + control_names
    if len(raw_variables) == len(order_vars):
        variable_names = raw_variables
    else:
        variable_names = order_vars

    n_x, n_y, n_e = len(state_names), len(control_names), len(shock_names)

    steady_state = getattr(model, "steady_state", None)
    shock_cov = getattr(model, "_shock_cov", None) or getattr(model, "shock_cov", None) or getattr(model, "Q", None)
    model_name = getattr(model, "model_name", getattr(model, "name", "linear_model"))

    return {
        "G": G, "N": N, "F": F, "L": L,
        "H_xx": np.zeros((n_x, n_x * n_x)),
        "G_xx": np.zeros((n_y, n_x * n_x)),
        "H_ss": np.zeros(n_x),
        "G_ss": np.zeros(n_y),
        "H_xu": np.zeros((n_x, n_x * n_e)),
        "G_xu": np.zeros((n_y, n_x * n_e)),
        "H_uu": np.zeros((n_x, n_e * n_e)),
        "G_uu": np.zeros((n_y, n_e * n_e)),
        "state_names": state_names, "control_names": control_names,
        "shock_names": shock_names, "variable_names": variable_names,
        "order_vars": order_vars, "steady_state": steady_state,
        "shock_cov": shock_cov, "model_name": model_name,
        "is_order3": False,
    }


# ---------------------------------------------------------------------------
# Core Vectorized Particle Filtering Engine
# ---------------------------------------------------------------------------

def particle_filter(
    model: Any,
    data: pd.DataFrame | np.ndarray,
    observed_vars: Sequence[str],
    *,
    n_particles: int = 10_000,
    method: str = "bootstrap",
    resampling_method: str = "systematic",
    resampling_threshold: float = 0.5,
    measurement_error: Mapping[str, float] | np.ndarray | None = None,
    measurement_dist: str = "gaussian",
    measurement_df: float = 5.0,
    stochastic_volatility: StochasticVolatilitySpec | dict | None = None,
    innovation_dist: str = "gaussian",
    innovation_df: float = 5.0,
    mixture_params: dict | None = None,
    init_states: str | np.ndarray = "stationary",
    burn_in: int = 50,
    prefilter: bool = False,
    observation_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ridge: float = 1e-6,
    seed: int | None = None,
) -> ParticleFilterResult:
    """Evaluate DSGE log-likelihood via pure-Python vectorized particle filtering.

    Supports Bootstrap Particle Filter (BPF) and Auxiliary Particle Filter (APF)
    over 1st, 2nd, and 3rd order pruned models, models with Stochastic Volatility,
    and fat-tailed / mixture error distributions. State propagation and likelihood
    evaluations are completely vectorized over N particles with zero Python particle loops.

    Parameters
    ----------
    model : LinearModel | PrunedDSGESolution | Order3PrunedSolution
        Solved macroeconomic DSGE model.
    data : pd.DataFrame | np.ndarray
        Observed time series of shape (T, k).
    observed_vars : Sequence[str]
        Names of observable variables corresponding to data columns.
    n_particles : int, default 10_000
        Number of particles N.
    method : {"bootstrap", "auxiliary"}, default "bootstrap"
        Particle filtering algorithm.
    resampling_method : {"systematic", "stratified", "residual", "multinomial"}, default "systematic"
        Resampling algorithm.
    resampling_threshold : float, default 0.5
        Resample when ESS_t < resampling_threshold * N.
    measurement_error : Mapping[str, float] | np.ndarray | None, optional
        Standard deviations of measurement errors by observable name.
    measurement_dist : {"gaussian", "student_t", "mixture"}, default "gaussian"
        Distribution of measurement errors.
    measurement_df : float, default 5.0
        Degrees of freedom for Student-t measurement errors.
    stochastic_volatility : StochasticVolatilitySpec | dict | None, optional
        Time-varying volatility specification sigma_t = bar{sigma} exp(h_t).
    innovation_dist : {"gaussian", "student_t", "mixture"}, default "gaussian"
        Distribution of structural innovations.
    innovation_df : float, default 5.0
        Degrees of freedom for Student-t structural innovations.
    mixture_params : dict, optional
        Parameters for mixture distributions (e.g. {"p_crisis": 0.05, "scale_crisis": 5.0}).
    init_states : {"stationary", "steady_state"} | np.ndarray, default "stationary"
        Initialization distribution for initial particles.
    burn_in : int, default 50
        Burn-in simulation periods for ergodic state initialization under pruning.
    observation_fn : callable, optional
        Custom observation function mapping particles (N, n_vars) to observables (N, k).
    ridge : float, default 1e-6
        Ridge floor added to covariance matrices to ensure strict positive-definiteness.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    ParticleFilterResult
        Frozen dataclass with log-likelihood, filtered states, ESS, and diagnostics.
    """
    if n_particles < 1:
        raise ValueError(f"n_particles must be >= 1, got {n_particles}")

    if not (0.0 <= float(resampling_threshold) <= 1.0):
        raise ValueError(
            f"resampling_threshold must be between 0.0 and 1.0, got {resampling_threshold}"
        )

    method = method.lower()
    if method not in ("bootstrap", "auxiliary"):
        raise ValueError(f"method must be 'bootstrap' or 'auxiliary', got {method!r}")

    resampling_method = resampling_method.lower()
    resample_fn_map = {
        "systematic": systematic_resample,
        "stratified": stratified_resample,
        "residual": residual_resample,
        "multinomial": multinomial_resample,
    }
    if resampling_method not in resample_fn_map:
        raise ValueError(
            f"resampling_method must be one of {list(resample_fn_map.keys())}, got {resampling_method!r}"
        )
    resample_fn = resample_fn_map[resampling_method]

    if innovation_dist == "student_t":
        if float(innovation_df) <= 2.0:
            raise ValueError(
                f"innovation_df must be > 2.0 to guarantee finite variance, got {innovation_df}"
            )
    if measurement_dist == "student_t":
        if float(measurement_df) <= 2.0:
            raise ValueError(
                f"measurement_df must be > 2.0 to guarantee finite variance, got {measurement_df}"
            )

    rng = np.random.default_rng(seed)

    # 1. Parse observed data and index
    obs_names = list(observed_vars)
    k = len(obs_names)
    if k == 0:
        raise ValueError("observed_vars must contain at least one variable name")

    if hasattr(data, "to_frame") and not isinstance(data, pd.DataFrame):
        data = data.to_frame()

    if isinstance(data, pd.DataFrame):
        data_index = data.index
        # Check all observed vars exist
        missing = [v for v in obs_names if v not in data.columns]
        if missing:
            raise ValueError(f"Observed variables {missing} not found in data columns: {list(data.columns)}")
        y_mat = data[obs_names].to_numpy(dtype=float)
    else:
        y_mat = np.asarray(data, dtype=float)
        if y_mat.ndim == 1:
            y_mat = y_mat[:, None]
        if y_mat.shape[1] != k:
            raise ValueError(f"data columns ({y_mat.shape[1]}) must match observed_vars length ({k})")
        data_index = pd.RangeIndex(len(y_mat))

    T = len(y_mat)
    if T == 0:
        raise ValueError("Observed data is empty (T=0)")

    if not np.all(np.isfinite(y_mat)):
        raise ValueError("Observed data contains NaN, Inf, or non-finite values")

    # 2. Extract model matrices and variables
    m_info = _extract_model_tensors(model)
    G = m_info["G"]
    N = m_info["N"]
    F = m_info["F"]
    L = m_info["L"]
    H_xx = m_info["H_xx"]
    G_xx = m_info["G_xx"]
    H_ss = m_info["H_ss"]
    G_ss = m_info["G_ss"]
    H_xu = m_info["H_xu"]
    G_xu = m_info["G_xu"]
    H_uu = m_info["H_uu"]
    G_uu = m_info["G_uu"]

    state_names = m_info["state_names"]
    control_names = m_info["control_names"]
    shock_names = m_info["shock_names"]
    order_vars = m_info["order_vars"]
    variable_names = m_info["variable_names"]
    model_name = m_info["model_name"]
    is_order3 = m_info["is_order3"]

    n_x = len(state_names)
    n_y = len(control_names)
    n_e = len(shock_names)
    n_vars = n_x + n_y

    if is_order3:
        H_xxx = m_info["H_xxx"]
        G_xxx = m_info["G_xxx"]
        H_xxu = m_info["H_xxu"]
        G_xxu = m_info["G_xxu"]
        H_xuu = m_info["H_xuu"]
        G_xuu = m_info["G_xuu"]
        H_uuu = m_info["H_uuu"]
        G_uuu = m_info["G_uuu"]
        H_x_ss = m_info["H_x_ss"]
        G_x_ss = m_info["G_x_ss"]
        H_u_ss = m_info["H_u_ss"]
        G_u_ss = m_info["G_u_ss"]

    # Map observable names to indices in order_vars
    obs_indices = []
    for nm in obs_names:
        if nm in order_vars:
            obs_indices.append(order_vars.index(nm))
        elif nm in variable_names:
            idx = variable_names.index(nm)
            var_target = variable_names[idx]
            if var_target in order_vars:
                obs_indices.append(order_vars.index(var_target))
            else:
                raise ValueError(f"Observable {nm!r} cannot be mapped to model variables: {order_vars}")
        else:
            raise ValueError(f"Observable {nm!r} is not declared in model variables: {order_vars}")
    obs_indices = np.array(obs_indices, dtype=int)

    # Steady state alignment
    ss = m_info["steady_state"]
    ss_vec = np.zeros(n_vars, dtype=float)
    if ss is not None and not prefilter:
        if isinstance(ss, (pd.Series, dict, Mapping)):
            for i, v in enumerate(order_vars):
                ss_vec[i] = float(ss.get(v, 0.0))
        else:
            raw_ss = np.asarray(ss, dtype=float).ravel()
            n_copy = min(len(raw_ss), n_vars)
            ss_vec[:n_copy] = raw_ss[:n_copy]

    # Base shock covariance Q
    raw_Q = m_info["shock_cov"]
    if raw_Q is not None:
        Q = np.asarray(raw_Q, dtype=float)
        if Q.shape != (n_e, n_e):
            Q = np.eye(n_e)
    else:
        Q = np.eye(n_e)

    # Base shock scale
    base_sigmas = np.sqrt(np.maximum(1e-12, np.diag(Q)))
    corr_Q = Q / (base_sigmas[:, None] * base_sigmas[None, :])
    try:
        L_corr = scipy.linalg.cholesky(corr_Q + ridge * np.eye(n_e), lower=True)
    except Exception:
        L_corr = np.eye(n_e)

    # Measurement error covariance R
    if measurement_error is not None:
        if isinstance(measurement_error, (dict, Mapping)):
            me_diag = [float(measurement_error.get(nm, 0.1)) ** 2 for nm in obs_names]
            R = np.diag(me_diag)
        else:
            R_in = np.asarray(measurement_error, dtype=float)
            if R_in.ndim == 1:
                R = np.diag(R_in ** 2)
            else:
                R = R_in
    else:
        std_data = np.std(y_mat, axis=0) if T > 1 else np.ones(k)
        std_data = np.where(std_data > 0, std_data * 0.15, 0.1)
        R = np.diag(std_data ** 2)

    R += ridge * np.eye(k)
    log_det_R = float(np.linalg.slogdet(R)[1])
    inv_R = np.linalg.inv(R)

    # Predictive auxiliary covariance for APF: Sigma_mu = Z_D Q Z_D' + R
    D_all = np.vstack([N, L]) if n_y > 0 else N
    Z_D = D_all[obs_indices, :]
    Sigma_mu = Z_D @ Q @ Z_D.T + R + ridge * np.eye(k)
    log_det_Sigma_mu = float(np.linalg.slogdet(Sigma_mu)[1])
    inv_Sigma_mu = np.linalg.inv(Sigma_mu)

    # Stochastic Volatility setup
    has_sv = stochastic_volatility is not None
    sv_rho = np.zeros(n_e)
    sv_sigma_eta = np.zeros(n_e)
    sv_base_scale = base_sigmas.copy()
    H_logvol = np.zeros((n_particles, n_e), dtype=float)

    if has_sv:
        if isinstance(stochastic_volatility, dict):
            sv_spec = StochasticVolatilitySpec(**stochastic_volatility)
        elif isinstance(stochastic_volatility, StochasticVolatilitySpec):
            sv_spec = stochastic_volatility
        else:
            raise TypeError("stochastic_volatility must be StochasticVolatilitySpec or dict")

        # Parse rho
        if isinstance(sv_spec.rho, (int, float)):
            sv_rho = np.full(n_e, float(sv_spec.rho))
        elif isinstance(sv_spec.rho, Mapping):
            sv_rho = np.array([float(sv_spec.rho.get(s, 0.9)) for s in shock_names])
        else:
            sv_rho = np.asarray(sv_spec.rho, dtype=float)

        # Parse sigma_eta
        if isinstance(sv_spec.sigma_eta, (int, float)):
            sv_sigma_eta = np.full(n_e, float(sv_spec.sigma_eta))
        elif isinstance(sv_spec.sigma_eta, Mapping):
            sv_sigma_eta = np.array([float(sv_spec.sigma_eta.get(s, 0.1)) for s in shock_names])
        else:
            sv_sigma_eta = np.asarray(sv_spec.sigma_eta, dtype=float)

        # Parse base_scale
        if sv_spec.base_scale is not None:
            if isinstance(sv_spec.base_scale, (int, float)):
                sv_base_scale = np.full(n_e, float(sv_spec.base_scale))
            elif isinstance(sv_spec.base_scale, Mapping):
                sv_base_scale = np.array([float(sv_spec.base_scale.get(s, base_sigmas[i])) for i, s in enumerate(shock_names)])
            else:
                sv_base_scale = np.asarray(sv_spec.base_scale, dtype=float)

        # Initialize H_logvol
        stat_var = sv_sigma_eta ** 2 / np.maximum(1e-4, 1.0 - sv_rho ** 2)
        if sv_spec.h0 is None or sv_spec.h0 == "stationary":
            H_logvol = rng.normal(0.0, np.sqrt(stat_var), size=(n_particles, n_e))
        elif isinstance(sv_spec.h0, (int, float)):
            H_logvol = np.full((n_particles, n_e), float(sv_spec.h0))
        else:
            H_logvol = np.tile(np.asarray(sv_spec.h0, dtype=float), (n_particles, 1))

    # 3. Initialize particle state deviations (x1, x2, x3)
    x1 = np.zeros((n_particles, n_x), dtype=float)
    x2 = np.zeros((n_particles, n_x), dtype=float)
    x3 = np.zeros((n_particles, n_x), dtype=float)

    if isinstance(init_states, np.ndarray):
        if init_states.ndim == 1:
            x1 = np.tile(init_states[:n_x], (n_particles, 1))
        else:
            x1 = init_states[:n_particles, :n_x].copy()
    elif init_states == "stationary" and n_x > 0:
        # Solve discrete Lyapunov for first-order stationary state covariance
        NQNT = N @ Q @ N.T
        try:
            P_stat = scipy.linalg.solve_discrete_lyapunov(G, NQNT)
            P_stat = 0.5 * (P_stat + P_stat.T)
            # Clip negative eigenvalues
            eigvals, eigvecs = np.linalg.eigh(P_stat)
            P_stat = eigvecs @ np.diag(np.maximum(1e-12, eigvals)) @ eigvecs.T
            L_P = scipy.linalg.cholesky(P_stat + ridge * np.eye(n_x), lower=True)
            x1 = rng.normal(0.0, 1.0, size=(n_particles, n_x)) @ L_P.T
        except Exception:
            x1 = rng.normal(0.0, 0.1, size=(n_particles, n_x))

        # Ergodic warm-up under 2nd/3rd order pruning
        if (H_xx is not None and np.any(np.abs(H_xx) > 1e-12)) and burn_in > 0:
            for _ in range(burn_in):
                z_b = rng.normal(0.0, 1.0, size=(n_particles, n_e))
                e_b = (z_b * base_sigmas) @ L_corr.T
                x1_n = x1 @ G.T + e_b @ N.T
                kron_xx = np.einsum("mi,mj->mij", x1, x1).reshape(n_particles, -1)
                kron_xe = np.einsum("mi,mj->mij", x1, e_b).reshape(n_particles, -1) if n_e > 0 else np.zeros((n_particles, 0))
                kron_ee = np.einsum("mi,mj->mij", e_b, e_b).reshape(n_particles, -1) if n_e > 0 else np.zeros((n_particles, 0))
                x2_n = (
                    x2 @ G.T
                    + 0.5 * (kron_xx @ H_xx.T)
                    + (kron_xe @ H_xu.T)
                    + 0.5 * (kron_ee @ H_uu.T)
                    + 0.5 * H_ss[None, :]
                )
                x1, x2 = x1_n, x2_n

    weights = np.full(n_particles, 1.0 / n_particles, dtype=float)

    # 4. Storage containers
    log_lik_contributions = np.zeros(T, dtype=float)
    filtered_states_arr = np.zeros((T, n_vars), dtype=float)
    filtered_states_std_arr = np.zeros((T, n_vars), dtype=float)
    ess_history = np.zeros(T, dtype=float)
    resample_history = np.zeros(T, dtype=bool)

    if has_sv:
        filtered_vol_arr = np.zeros((T, n_e), dtype=float)
        filtered_vol_std_arr = np.zeros((T, n_e), dtype=float)

    const_gaussian_obs = -0.5 * (k * math.log(2.0 * math.pi) + log_det_R)
    const_gaussian_aux = -0.5 * (k * math.log(2.0 * math.pi) + log_det_Sigma_mu)

    # 5. Filter recursion over t = 0, ..., T-1
    for t in range(T):
        y_obs = y_mat[t]

        # Draw structural innovations
        if innovation_dist == "student_t":
            nu_e = max(2.01, float(innovation_df))
            z_raw = rng.standard_t(nu_e, size=(n_particles, n_e))
            z_shocks = z_raw * math.sqrt((nu_e - 2.0) / nu_e)
        elif innovation_dist == "mixture":
            p_cris = float(mixture_params.get("p_crisis", 0.05) if mixture_params else 0.05)
            s_cris = float(mixture_params.get("scale_crisis", 5.0) if mixture_params else 5.0)
            sig_calm = 1.0 / math.sqrt(1.0 - p_cris + p_cris * (s_cris ** 2))
            sig_crisis = s_cris * sig_calm
            is_crisis = rng.uniform(0.0, 1.0, size=(n_particles, n_e)) < p_cris
            regime_scales = np.where(is_crisis, sig_crisis, sig_calm)
            z_shocks = rng.normal(0.0, 1.0, size=(n_particles, n_e)) * regime_scales
        else:
            z_shocks = rng.normal(0.0, 1.0, size=(n_particles, n_e))

        if method == "bootstrap":
            # ---------------------------------------------------------------
            # Bootstrap Particle Filter (Gordon et al. 1993)
            # ---------------------------------------------------------------
            # Advance volatility if SV is active
            if has_sv:
                eta = rng.normal(0.0, 1.0, size=(n_particles, n_e))
                H_logvol = H_logvol * sv_rho[None, :] + eta * sv_sigma_eta[None, :]
                sig_t = sv_base_scale[None, :] * np.exp(H_logvol)
                e_t = (z_shocks * sig_t) @ L_corr.T
            else:
                e_t = (z_shocks * base_sigmas[None, :]) @ L_corr.T

            # Vectorized state propagation over N particles
            x1_next = x1 @ G.T + e_t @ N.T
            y1_next = x1 @ F.T + e_t @ L.T

            # 2nd-order outer products
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

            # 3rd-order outer products
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

            tot_states = x1_next + x2_next + x3_next
            tot_controls = y1_next + y2_next + y3_next
            V_particles = (np.hstack([tot_states, tot_controls]) if n_y > 0 else tot_states) + ss_vec[None, :]

            # Predicted observables
            if observation_fn is not None:
                pred_y = observation_fn(V_particles)
            else:
                pred_y = V_particles[:, obs_indices]

            res = y_obs[None, :] - pred_y

            # Observation log density
            if measurement_dist == "student_t":
                nu_m = max(2.01, float(measurement_df))
                quad = np.sum((res @ inv_R) * res, axis=1)
                log_p = (
                    scipy.special.gammaln(0.5 * (nu_m + k))
                    - scipy.special.gammaln(0.5 * nu_m)
                    - 0.5 * k * math.log(math.pi * nu_m)
                    - 0.5 * log_det_R
                    - 0.5 * (nu_m + k) * np.log1p(quad / nu_m)
                )
            else:
                quad = np.sum((res @ inv_R) * res, axis=1)
                log_p = const_gaussian_obs - 0.5 * quad

            # Importance weighting with log-sum-exp
            log_w_weighted = log_p + np.log(np.maximum(weights, 1e-300))
            max_lw = float(np.max(log_w_weighted))

            if not np.isfinite(max_lw) or max_lw < -650.0:
                # Low-weight outlier rejuvenation
                log_lik_incr = max(max_lw, -100.0 * k)
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

            log_lik_contributions[t] = log_lik_incr

            # Filtered states mean & std
            f_mean = np.sum(weights[:, None] * V_particles, axis=0)
            f_var = np.sum(weights[:, None] * (V_particles - f_mean[None, :]) ** 2, axis=0)
            filtered_states_arr[t] = f_mean
            filtered_states_std_arr[t] = np.sqrt(np.maximum(0.0, f_var))

            if has_sv:
                vol_t = sv_base_scale[None, :] * np.exp(H_logvol)
                vol_mean = np.sum(weights[:, None] * vol_t, axis=0)
                vol_var = np.sum(weights[:, None] * (vol_t - vol_mean[None, :]) ** 2, axis=0)
                filtered_vol_arr[t] = vol_mean
                filtered_vol_std_arr[t] = np.sqrt(np.maximum(0.0, vol_var))

            # Effective Sample Size (ESS) & Resampling trigger
            ess_t = 1.0 / np.sum(weights ** 2)
            ess_history[t] = ess_t

            if ess_t < resampling_threshold * n_particles:
                resample_history[t] = True
                resample_idx = resample_fn(weights, rng)
                x1 = x1_next[resample_idx].copy()
                x2 = x2_next[resample_idx].copy()
                x3 = x3_next[resample_idx].copy()
                if has_sv:
                    H_logvol = H_logvol[resample_idx].copy()
                weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
            else:
                resample_history[t] = False
                x1 = x1_next.copy()
                x2 = x2_next.copy()
                x3 = x3_next.copy()

        else:
            # ---------------------------------------------------------------
            # Auxiliary Particle Filter (Pitt & Shephard 1999)
            # ---------------------------------------------------------------
            # 1. Point prediction without shocks (e = 0)
            mu_x1 = x1 @ G.T
            mu_y1 = x1 @ F.T
            kron_x1_x1_pt = np.einsum("mi,mj->mij", x1, x1).reshape(n_particles, -1) if n_x > 0 else np.zeros((n_particles, 0))
            mu_x2 = x2 @ G.T + 0.5 * (kron_x1_x1_pt @ H_xx.T) + 0.5 * H_ss[None, :]
            mu_y2 = x2 @ F.T + 0.5 * (kron_x1_x1_pt @ G_xx.T) + 0.5 * G_ss[None, :]

            if is_order3:
                kron_x1_x2_pt = np.einsum("mi,mj->mij", x1, x2).reshape(n_particles, -1) if n_x > 0 else np.zeros((n_particles, 0))
                kron_x1_3_pt = np.einsum("mi,mj->mij", kron_x1_x1_pt, x1).reshape(n_particles, -1) if n_x > 0 else np.zeros((n_particles, 0))
                mu_x3 = x3 @ G.T + (kron_x1_x2_pt @ H_xx.T) + (1.0 / 6.0) * (kron_x1_3_pt @ H_xxx.T) + 0.5 * (x1 @ H_x_ss.T)
                mu_y3 = x3 @ F.T + (kron_x1_x2_pt @ G_xx.T) + (1.0 / 6.0) * (kron_x1_3_pt @ G_xxx.T) + 0.5 * (x1 @ G_x_ss.T)
            else:
                mu_x3 = np.zeros_like(mu_x1)
                mu_y3 = np.zeros_like(mu_y1)

            mu_tot_s = mu_x1 + mu_x2 + mu_x3
            mu_tot_c = mu_y1 + mu_y2 + mu_y3
            mu_V = (np.hstack([mu_tot_s, mu_tot_c]) if n_y > 0 else mu_tot_s) + ss_vec[None, :]

            if observation_fn is not None:
                hat_y = observation_fn(mu_V)
            else:
                hat_y = mu_V[:, obs_indices]

            # First-stage auxiliary proposal log-density with Sigma_mu
            res_aux = y_obs[None, :] - hat_y
            quad_aux = np.sum((res_aux @ inv_Sigma_mu) * res_aux, axis=1)
            log_lambda_unnorm = const_gaussian_aux - 0.5 * quad_aux + np.log(np.maximum(weights, 1e-300))
            max_ll = float(np.max(log_lambda_unnorm))

            if not np.isfinite(max_ll) or max_ll < -650.0:
                log_sum_lambda = max(max_ll, -100.0 * k)
                lambda_bar = np.full(n_particles, 1.0 / n_particles, dtype=float)
            else:
                lambda_exp = np.exp(np.clip(log_lambda_unnorm - max_ll, -700.0, 50.0))
                sum_lambda = float(np.sum(lambda_exp))
                if sum_lambda <= 0.0 or not np.isfinite(sum_lambda):
                    log_sum_lambda = max_ll
                    lambda_bar = np.full(n_particles, 1.0 / n_particles, dtype=float)
                else:
                    log_sum_lambda = max_ll + math.log(sum_lambda)
                    lambda_bar = lambda_exp / sum_lambda

            # 2. Resample parent indices according to auxiliary weights
            parent_idx = resample_fn(lambda_bar, rng)
            resample_history[t] = True

            x1_parents = x1[parent_idx]
            x2_parents = x2[parent_idx]
            x3_parents = x3[parent_idx]
            hat_y_parents = hat_y[parent_idx]

            # Advance volatility for parents
            if has_sv:
                H_parents = H_logvol[parent_idx]
                eta = rng.normal(0.0, 1.0, size=(n_particles, n_e))
                H_logvol = H_parents * sv_rho[None, :] + eta * sv_sigma_eta[None, :]
                sig_t = sv_base_scale[None, :] * np.exp(H_logvol)
                e_t = (z_shocks * sig_t) @ L_corr.T
            else:
                e_t = (z_shocks * base_sigmas[None, :]) @ L_corr.T

            # 3. Propagate resampled parent particles with drawn innovations
            x1_next = x1_parents @ G.T + e_t @ N.T
            y1_next = x1_parents @ F.T + e_t @ L.T

            kron_x1_x1 = np.einsum("mi,mj->mij", x1_parents, x1_parents).reshape(n_particles, -1) if n_x > 0 else np.zeros((n_particles, 0))
            kron_x1_e = np.einsum("mi,mj->mij", x1_parents, e_t).reshape(n_particles, -1) if (n_x > 0 and n_e > 0) else np.zeros((n_particles, 0))
            kron_e_e = np.einsum("mi,mj->mij", e_t, e_t).reshape(n_particles, -1) if n_e > 0 else np.zeros((n_particles, 0))

            x2_next = (
                x2_parents @ G.T
                + 0.5 * (kron_x1_x1 @ H_xx.T)
                + (kron_x1_e @ H_xu.T)
                + 0.5 * (kron_e_e @ H_uu.T)
                + 0.5 * H_ss[None, :]
            )
            y2_next = (
                x2_parents @ F.T
                + 0.5 * (kron_x1_x1 @ G_xx.T)
                + (kron_x1_e @ G_xu.T)
                + 0.5 * (kron_e_e @ G_uu.T)
                + 0.5 * G_ss[None, :]
            )

            if is_order3:
                kron_x1_x2 = np.einsum("mi,mj->mij", x1_parents, x2_parents).reshape(n_particles, -1) if n_x > 0 else np.zeros((n_particles, 0))
                kron_x2_e = np.einsum("mi,mj->mij", x2_parents, e_t).reshape(n_particles, -1) if (n_x > 0 and n_e > 0) else np.zeros((n_particles, 0))
                kron_x1_3 = np.einsum("mi,mj->mij", kron_x1_x1, x1_parents).reshape(n_particles, -1) if n_x > 0 else np.zeros((n_particles, 0))
                kron_x1_2_e = np.einsum("mi,mj->mij", kron_x1_x1, e_t).reshape(n_particles, -1) if (n_x > 0 and n_e > 0) else np.zeros((n_particles, 0))
                kron_x1_e_2 = np.einsum("mi,mj->mij", x1_parents, kron_e_e).reshape(n_particles, -1) if (n_x > 0 and n_e > 0) else np.zeros((n_particles, 0))
                kron_e_3 = np.einsum("mi,mj->mij", kron_e_e, e_t).reshape(n_particles, -1) if n_e > 0 else np.zeros((n_particles, 0))

                x3_next = (
                    x3_parents @ G.T
                    + (kron_x1_x2 @ H_xx.T)
                    + (kron_x2_e @ H_xu.T)
                    + (1.0 / 6.0) * (kron_x1_3 @ H_xxx.T)
                    + 0.5 * (kron_x1_2_e @ H_xxu.T)
                    + 0.5 * (kron_x1_e_2 @ H_xuu.T)
                    + (1.0 / 6.0) * (kron_e_3 @ H_uuu.T)
                    + 0.5 * (x1_parents @ H_x_ss.T)
                    + 0.5 * (e_t @ H_u_ss.T)
                )
                y3_next = (
                    x3_parents @ F.T
                    + (kron_x1_x2 @ G_xx.T)
                    + (kron_x2_e @ G_xu.T)
                    + (1.0 / 6.0) * (kron_x1_3 @ G_xxx.T)
                    + 0.5 * (kron_x1_2_e @ G_xxu.T)
                    + 0.5 * (kron_x1_e_2 @ G_xuu.T)
                    + (1.0 / 6.0) * (kron_e_3 @ G_uuu.T)
                    + 0.5 * (x1_parents @ G_x_ss.T)
                    + 0.5 * (e_t @ G_u_ss.T)
                )
            else:
                x3_next = np.zeros_like(x1_next)
                y3_next = np.zeros_like(y1_next)

            tot_states = x1_next + x2_next + x3_next
            tot_controls = y1_next + y2_next + y3_next
            V_particles = (np.hstack([tot_states, tot_controls]) if n_y > 0 else tot_states) + ss_vec[None, :]

            if observation_fn is not None:
                tilde_y = observation_fn(V_particles)
            else:
                tilde_y = V_particles[:, obs_indices]

            # 4. Second-stage weights: true observation density / auxiliary proposal density
            res_true = y_obs[None, :] - tilde_y
            quad_true = np.sum((res_true @ inv_R) * res_true, axis=1)

            if measurement_dist == "student_t":
                nu_m = max(2.01, float(measurement_df))
                log_p_true = (
                    scipy.special.gammaln(0.5 * (nu_m + k))
                    - scipy.special.gammaln(0.5 * nu_m)
                    - 0.5 * k * math.log(math.pi * nu_m)
                    - 0.5 * log_det_R
                    - 0.5 * (nu_m + k) * np.log1p(quad_true / nu_m)
                )
            else:
                log_p_true = const_gaussian_obs - 0.5 * quad_true

            res_prop = y_obs[None, :] - hat_y_parents
            quad_prop = np.sum((res_prop @ inv_Sigma_mu) * res_prop, axis=1)
            log_g_prop = const_gaussian_aux - 0.5 * quad_prop

            log_w2 = log_p_true - log_g_prop
            max_lw2 = float(np.max(log_w2))

            if not np.isfinite(max_lw2) or max_lw2 < -650.0:
                log_lik_incr = log_sum_lambda
                weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
            else:
                w2_exp = np.exp(np.clip(log_w2 - max_lw2, -700.0, 50.0))
                sum_w2 = float(np.sum(w2_exp))
                if sum_w2 <= 0.0 or not np.isfinite(sum_w2):
                    log_lik_incr = log_sum_lambda
                    weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
                else:
                    log_lik_incr = log_sum_lambda + max_lw2 + math.log(sum_w2 / n_particles)
                    weights = w2_exp / sum_w2

            log_lik_contributions[t] = log_lik_incr

            # Filtered states mean & std
            f_mean = np.sum(weights[:, None] * V_particles, axis=0)
            f_var = np.sum(weights[:, None] * (V_particles - f_mean[None, :]) ** 2, axis=0)
            filtered_states_arr[t] = f_mean
            filtered_states_std_arr[t] = np.sqrt(np.maximum(0.0, f_var))

            if has_sv:
                vol_t = sv_base_scale[None, :] * np.exp(H_logvol)
                vol_mean = np.sum(weights[:, None] * vol_t, axis=0)
                vol_var = np.sum(weights[:, None] * (vol_t - vol_mean[None, :]) ** 2, axis=0)
                filtered_vol_arr[t] = vol_mean
                filtered_vol_std_arr[t] = np.sqrt(np.maximum(0.0, vol_var))

            ess_t = 1.0 / np.sum(weights ** 2)
            ess_history[t] = ess_t

            x1 = x1_next.copy()
            x2 = x2_next.copy()
            x3 = x3_next.copy()

    total_log_likelihood = float(np.sum(log_lik_contributions))

    # Format result DataFrames
    state_cols = list(order_vars)
    df_filtered_states = pd.DataFrame(filtered_states_arr, index=data_index, columns=state_cols)
    df_filtered_states_std = pd.DataFrame(filtered_states_std_arr, index=data_index, columns=state_cols)

    # Reorder columns to variable_names if distinct
    if set(variable_names) == set(state_cols) and variable_names != tuple(state_cols):
        df_filtered_states = df_filtered_states[list(variable_names)]
        df_filtered_states_std = df_filtered_states_std[list(variable_names)]

    s_ess = pd.Series(ess_history, index=data_index, name="ESS")
    s_resample = pd.Series(resample_history, index=data_index, name="Resampled")
    s_ll_contrib = pd.Series(log_lik_contributions, index=data_index, name="LogLikelihood")
    resampling_freq = float(np.mean(resample_history))

    df_filtered_vol = None
    df_filtered_vol_std = None
    if has_sv:
        vol_cols = [f"sigma_{s}" for s in shock_names]
        df_filtered_vol = pd.DataFrame(filtered_vol_arr, index=data_index, columns=vol_cols)
        df_filtered_vol_std = pd.DataFrame(filtered_vol_std_arr, index=data_index, columns=vol_cols)

    return ParticleFilterResult(
        log_likelihood=total_log_likelihood,
        log_likelihood_contributions=s_ll_contrib,
        filtered_states=df_filtered_states,
        filtered_states_std=df_filtered_states_std,
        ess=s_ess,
        resample_history=s_resample,
        resampling_frequency=resampling_freq,
        n_particles=n_particles,
        method=method,
        resampling_method=resampling_method,
        model_name=model_name,
        variable_names=variable_names,
        observed_vars=tuple(obs_names),
        filtered_volatility=df_filtered_vol,
        filtered_volatility_std=df_filtered_vol_std,
    )


__all__ = [
    "StochasticVolatilitySpec",
    "ParticleFilterResult",
    "particle_filter",
    "systematic_resample",
    "stratified_resample",
    "residual_resample",
    "multinomial_resample",
]
