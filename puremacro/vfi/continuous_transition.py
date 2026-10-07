"""Continuous Transition Dynamics & MIT Shocks in Sequence Space.

Implements non-linear general equilibrium transition dynamics for continuous-state
and continuous-time heterogeneous-agent macro models following unexpected
MIT shocks (permanent or transitory TFP shocks, interest rate jumps, tax/fiscal reforms).

Methodology:
1. Backward EGM Time Iteration:
   Given factor price sequences {r_t, w_t}_{t=0}^{T-1} and terminal stationary policy c_T^*(k, z),
   household consumption c_t(k, z) and savings a'_t(k, z) are solved backward in time
   for t = T-1, ..., 0 via the Endogenous Grid Method (EGM) without discretization bias.
2. Forward Young (2010) Operator Coupling:
   Starting from the initial continuous wealth distribution mu_0(k, z), the distribution is
   advanced forward period by period via the Young (2010) lottery projection:
       mu_{t+1} = T_t^* mu_t = continuous_push_distribution(mu_t, a'_t, K_hist, P_z, z_grid)
   strictly preserving total probability mass (|sum mu_t - 1.0| <= 1e-12) at every date.
3. General Equilibrium Market Clearing Relaxation:
   Solves the stacked non-linear system H_t(r) = K_t^s(r) - K_t^d(r_t; Z_t) = 0 for t = 0, ..., T-1:
   - Damped fixed-point shooting (_solve_transition_shooting):
       r^{(k+1)} = (1 - omega) r^{(k)} + omega r^{implied}(K_t^s)
   - Sequence-space Broyden Quasi-Newton (_solve_transition_broyden):
       Sherman-Morrison rank-1 inverse Jacobian updates with analytical firm-demand
       diagonal initialization and monotone backtracking line search, converging
       to ||K^s - K^d||_inf < 1e-4.

References
----------
- Young, E. R. (2010). "Solving the Incomplete Markets Model with Aggregate Uncertainty
  Using the Krusell-Smith Algorithm and Non-Stochastic Simulations."
  Journal of Economic Dynamics and Control, 34(1), 36-45.
- Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). "Using the Sequence-Space
  Jacobian to Solve and Estimate Heterogeneous-Agent Models."
  Econometrica, 89(6), 3115-3148.
- Aiyagari, S. R. (1994). "Uninsured Idiosyncratic Risk and Aggregate Saving."
  Quarterly Journal of Economics, 109(3), 659-684.
"""
from __future__ import annotations

import inspect
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from puremacro import _backend as _bk
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst
from puremacro.vfi.continuous_distribution import (
    AiyagariContinuousEquilibrium,
    _caller_stacklevel,
    _positive_finite,
    continuous_push_distribution,
    continuous_stationary_distribution,
    solve_aiyagari_continuous,
    young_lottery_weights,
)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TransitionShock:
    """Exogenous aggregate MIT shock path."""

    path: np.ndarray
    var: str = "z"  # "z" (TFP), "r" (rate), "beta" (discount factor)

    def __post_init__(self) -> None:
        self.path = np.asarray(self.path, dtype=np.float64).ravel()
        if not np.all(np.isfinite(self.path)):
            raise ValueError("Shock path values must all be finite numbers")
        self.var = str(self.var).lower().strip()


@dataclass(frozen=True)
class ContinuousTransitionResult:
    """Non-linear general equilibrium continuous transition path under MIT shocks.

    Parameters
    ----------
    r_path : np.ndarray
        Real interest rate path of shape (T,).
    w_path : np.ndarray
        Real wage path of shape (T,).
    K_s_path : np.ndarray
        Aggregate capital supply path int k dmu_t of shape (T,).
    K_d_path : np.ndarray
        Firm aggregate capital demand path of shape (T,).
    C_path : np.ndarray
        Aggregate consumption path int c_t dmu_t of shape (T,).
    distributions : list[np.ndarray]
        Sequence of T+1 continuous wealth distributions mu_0, ..., mu_T,
        each of shape (N_k, n_z) or (N_k,).
    policies_a : list[np.ndarray]
        Sequence of T next-period asset policies a'_t(k, z) on histogram grid K_hist.
    policies_c : list[np.ndarray]
        Sequence of T consumption policies c_t(k, z) on histogram grid K_hist.
    residuals : np.ndarray
        Capital market clearing residuals H_t = K_t^s - K_t^d of shape (T,).
    max_residual : float
        Maximum absolute market clearing error max_t |H_t|.
    iterations : int
        Number of price relaxation iterations performed by the solver.
    converged : bool
        Whether the equilibrium price relaxation solver converged within tolerance.
    elapsed_time : float
        Wall-clock execution time in seconds.
    metadata : dict
        Algorithm diagnostics, shock parameters, and solver configuration.
    """

    r_path: np.ndarray
    w_path: np.ndarray
    K_s_path: np.ndarray
    K_d_path: np.ndarray
    C_path: np.ndarray
    distributions: list[np.ndarray]
    policies_a: list[np.ndarray]
    policies_c: list[np.ndarray]
    residuals: np.ndarray
    max_residual: float
    iterations: int
    converged: bool
    elapsed_time: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        """Length T of the transition horizon."""
        return len(self.r_path)

    @property
    def mass_conservation_error(self) -> float:
        """Maximum deviation of distribution mass from 1.0 across all dates."""
        return max(float(np.abs(np.sum(d) - 1.0)) for d in self.distributions)

    def summary(self) -> pd.DataFrame:
        """Structured summary DataFrame of the transition dynamics."""
        rows = [
            {"Metric": "Horizon (T)", "Value": str(self.horizon)},
            {"Metric": "Solver Method", "Value": str(self.metadata.get("solver", "broyden"))},
            {"Metric": "Converged", "Value": str(self.converged)},
            {"Metric": "Iterations", "Value": str(self.iterations)},
            {"Metric": "Max Residual (||H||_inf)", "Value": f"{self.max_residual:.2e}"},
            {"Metric": "Mean Residual", "Value": f"{float(np.mean(np.abs(self.residuals))):.2e}"},
            {"Metric": "Initial Capital (K_0)", "Value": f"{self.K_s_path[0]:.6f}"},
            {"Metric": "Terminal Capital (K_T)", "Value": f"{self.K_s_path[-1]:.6f}"},
            {"Metric": "Initial Real Rate (r_0)", "Value": f"{self.r_path[0]:.6f}"},
            {"Metric": "Terminal Real Rate (r_T)", "Value": f"{self.r_path[-1]:.6f}"},
            {"Metric": "Initial Wage (w_0)", "Value": f"{self.w_path[0]:.6f}"},
            {"Metric": "Terminal Wage (w_T)", "Value": f"{self.w_path[-1]:.6f}"},
            {"Metric": "Mass Error", "Value": f"{self.mass_conservation_error:.2e}"},
            {"Metric": "Elapsed Time (s)", "Value": f"{self.elapsed_time:.4f}"},
        ]
        return pd.DataFrame(rows).set_index("Metric")

    def to_frame(self) -> pd.DataFrame:
        """Return transition paths as a tidy pandas DataFrame."""
        df = pd.DataFrame(
            {
                "r": self.r_path,
                "w": self.w_path,
                "K_s": self.K_s_path,
                "K_d": self.K_d_path,
                "C": self.C_path,
                "residual": self.residuals,
            }
        )
        df.index.name = "t"
        return df

    def to_markdown(self, **kwargs: Any) -> str:
        """Render summary table as Markdown string."""
        return _df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render summary table as LaTeX tabular string."""
        return _df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render summary table as Typst table string."""
        return _df_to_typst(self.summary(), **kwargs)

    def plot(
        self,
        figsize: Tuple[float, float] = (12, 8),
        show: bool = False,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Plot the equilibrium transition trajectories and wealth distribution dynamics.

        Visualizes:
        1. Real interest rate path r_t with steady-state benchmarks.
        2. Real wage path w_t.
        3. Capital market clearing: Capital supply K_t^s vs firm demand K_t^d.
        4. Aggregate consumption C_t.
        5. Market clearing residual H_t = K_t^s - K_t^d.
        6. Evolution of the marginal continuous asset distribution mu_t(k).
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
        t_grid = np.arange(self.horizon)

        # 1. Real Interest Rate
        ax = axes[0, 0]
        ax.plot(t_grid, self.r_path, color="tab:blue", lw=2, label="$r_t$ (Equilibrium)")
        ax.axhline(self.r_path[-1], color="tab:blue", ls="--", alpha=0.6, label="Terminal SS")
        ax.set_title("Real Interest Rate Path $r_t$")
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Rate")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        # 2. Real Wage
        ax = axes[0, 1]
        ax.plot(t_grid, self.w_path, color="tab:orange", lw=2, label="$w_t$")
        ax.axhline(self.w_path[-1], color="tab:orange", ls="--", alpha=0.6, label="Terminal SS")
        ax.set_title("Real Wage Path $w_t$")
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Wage")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        # 3. Capital Market Clearing (K^s vs K^d)
        ax = axes[0, 2]
        ax.plot(t_grid, self.K_s_path, color="tab:green", lw=2, label="Supply $K^s_t$")
        ax.plot(t_grid, self.K_d_path, color="tab:red", lw=1.5, ls="--", label="Demand $K^d_t$")
        ax.set_title("Capital Market Clearing")
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Capital")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        # 4. Aggregate Consumption
        ax = axes[1, 0]
        ax.plot(t_grid, self.C_path, color="tab:purple", lw=2, label="$C_t$")
        ax.set_title("Aggregate Consumption Path $C_t$")
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Consumption")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        # 5. Market Clearing Residual
        ax = axes[1, 1]
        ax.plot(t_grid, self.residuals, color="tab:brown", lw=1.5, label="$K^s_t - K^d_t$")
        ax.axhline(0.0, color="black", ls=":", alpha=0.5)
        ax.set_title(f"Market Clearing Residual (max={self.max_residual:.2e})")
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Residual")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        # 6. Wealth Distribution Evolution
        ax = axes[1, 2]
        # Extract asset grid from metadata if present
        k_grid = self.metadata.get("asset_grid", None)
        T = self.horizon
        dates_to_plot = [0, T // 4, T // 2, T]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(dates_to_plot)))

        for date, col in zip(dates_to_plot, colors):
            dist = self.distributions[date]
            marginal_k = np.sum(dist, axis=1) if dist.ndim == 2 else dist
            if k_grid is not None and len(k_grid) == len(marginal_k):
                x_vals = k_grid
            else:
                x_vals = np.linspace(0.0, 1.0, len(marginal_k))
            ax.plot(x_vals, marginal_k, color=col, lw=1.8, label=f"$t={date}$")

        ax.set_title("Wealth Distribution Evolution $\\mu_t(k)$")
        ax.set_xlabel("Assets $k$")
        ax.set_ylabel("Probability Mass")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        if show:
            plt.show()
        return fig


# ---------------------------------------------------------------------------
# Core Algorithmic Components
# ---------------------------------------------------------------------------

def _backward_egm_path(
    r_path: np.ndarray,
    w_path: np.ndarray,
    c_term: np.ndarray,
    a_grid_dense: np.ndarray,
    K_hist: np.ndarray,
    P_z: np.ndarray,
    z_grid: np.ndarray,
    beta: float | np.ndarray = 0.96,
    gamma: float = 2.0,
    r_term: float = 0.02,
) -> Tuple[list[np.ndarray], list[np.ndarray]]:
    """Solve household policy functions backward in time via EGM.

    Given factor prices {r_t, w_t}_{t=0}^{T-1} and continuation consumption
    policy c_T(k, z) = c_term, performs backward time iteration:
        EMu_t(a', z_m) = beta_t * (1 + r_{t+1}) * sum_{z'} P_z(z_m, z') * c_{t+1}(a', z')^{-gamma}
        c_endo(a', z_m) = EMu_t^{-1/gamma}
        k_endo(a', z_m) = (c_endo + a' - w_t * z_m) / (1 + r_t)

    Interpolates decision rules onto dense evaluation grid and fine histogram grid K_hist.

    Parameters
    ----------
    r_path : np.ndarray
        Path of real interest rates of length T.
    w_path : np.ndarray
        Path of wages of length T.
    c_term : np.ndarray
        Terminal consumption policy on a_grid_dense of shape (n_a, n_z).
    a_grid_dense : np.ndarray
        1D grid of asset points used for EGM iteration.
    K_hist : np.ndarray
        1D histogram asset grid used for Young (2010) distribution.
    P_z : np.ndarray
        Row-stochastic transition matrix for productivity shocks.
    z_grid : np.ndarray
        Discrete productivity levels of length n_z.
    beta : float or np.ndarray
        Discount factor: a scalar, or a path (beta_0, ..., beta_{T-1}) where
        beta_t discounts date t+1 in the date-t Euler equation.
    gamma : float
        Relative risk aversion coefficient.
    r_term : float
        Terminal interest rate r_T.

    Returns
    -------
    policies_a : list[np.ndarray]
        List of length T of next-period asset policies a'_t on K_hist of shape (N_k, n_z).
    policies_c : list[np.ndarray]
        List of length T of consumption policies c_t on K_hist of shape (N_k, n_z).
    """
    T = len(r_path)
    N_k = len(K_hist)
    n_z = len(z_grid)
    n_a = len(a_grid_dense)
    beta_arr = np.broadcast_to(np.asarray(beta, dtype=np.float64), (T,))

    policies_a = [None] * T
    policies_c = [None] * T

    c_next = np.asarray(c_term, dtype=np.float64).copy()

    for t in range(T - 1, -1, -1):
        r_t = float(r_path[t])
        w_t = float(w_path[t])
        r_tp1 = float(r_path[t + 1]) if t < T - 1 else float(r_term)

        # Expected marginal utility of saving a'
        # EMu has shape (n_a, n_z)
        EMu = float(beta_arr[t]) * (1.0 + r_tp1) * (c_next ** (-gamma) @ P_z.T)
        c_endo = EMu ** (-1.0 / gamma)

        c_dense = np.zeros((n_a, n_z), dtype=np.float64)
        for m in range(n_z):
            zm = z_grid[m]
            # Endogenous beginning-of-period assets
            a_endo = (c_endo[:, m] + a_grid_dense - w_t * zm) / (1.0 + r_t)

            # Interpolate c onto fixed a_grid_dense
            c_interp = np.interp(a_grid_dense, a_endo, c_endo[:, m])

            # Borrowing constraint binds where a < a_endo[0]
            binds = a_grid_dense < a_endo[0]
            c_interp[binds] = (1.0 + r_t) * a_grid_dense[binds] + w_t * zm
            c_dense[:, m] = np.maximum(c_interp, 1e-12)

        # Savings policy a' = (1 + r_t) a + w_t z - c
        ap_dense = np.zeros((n_a, n_z), dtype=np.float64)
        for m in range(n_z):
            zm = z_grid[m]
            ap_dense[:, m] = np.maximum((1.0 + r_t) * a_grid_dense + w_t * zm - c_dense[:, m], 0.0)

        # Interpolate policies onto histogram grid K_hist
        ap_hist = np.zeros((N_k, n_z), dtype=np.float64)
        c_hist = np.zeros((N_k, n_z), dtype=np.float64)
        for m in range(n_z):
            ap_hist[:, m] = np.interp(K_hist, a_grid_dense, ap_dense[:, m])
            c_hist[:, m] = np.interp(K_hist, a_grid_dense, c_dense[:, m])

        policies_a[t] = ap_hist
        policies_c[t] = c_hist
        c_next = c_dense

    return policies_a, policies_c


def _forward_young_path(
    mu_0: np.ndarray,
    policies_a: list[np.ndarray],
    policies_c: list[np.ndarray],
    K_hist: np.ndarray,
    P_z: np.ndarray,
    z_grid: np.ndarray,
) -> Tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Push the continuous wealth distribution forward period by period via Young (2010).

    Computes:
        mu_{t+1} = continuous_push_distribution(mu_t, policies_a[t], K_hist, P_z, z_grid)
    while aggregating:
        K_s[t] = sum_{i, m} K_hist[i] * mu_t[i, m]
        C[t] = sum_{i, m} policies_c[t][i, m] * mu_t[i, m]

    Parameters
    ----------
    mu_0 : np.ndarray
        Initial wealth distribution of shape (N_k, n_z) or (N_k,).
    policies_a : list[np.ndarray]
        List of length T of asset policies.
    policies_c : list[np.ndarray]
        List of length T of consumption policies.
    K_hist : np.ndarray
        Histogram asset grid.
    P_z : np.ndarray
        Markov transition matrix.
    z_grid : np.ndarray
        Productivity grid.

    Returns
    -------
    distributions : list[np.ndarray]
        List of length T+1 of probability distributions mu_0, ..., mu_T.
    K_s_path : np.ndarray
        Aggregate capital supply path of shape (T,).
    C_path : np.ndarray
        Aggregate consumption path of shape (T,).
    """
    T = len(policies_a)
    distributions = [None] * (T + 1)
    K_s_path = np.zeros(T, dtype=np.float64)
    C_path = np.zeros(T, dtype=np.float64)

    curr_pdf = np.asarray(mu_0, dtype=np.float64).copy()
    # Normalize initial mass
    curr_pdf /= np.sum(curr_pdf)
    distributions[0] = curr_pdf

    for t in range(T):
        # Aggregate capital supply and consumption at date t
        if curr_pdf.ndim == 2:
            K_s_path[t] = float(np.sum(K_hist[:, None] * curr_pdf))
            C_path[t] = float(np.sum(policies_c[t] * curr_pdf))
        else:
            K_s_path[t] = float(np.sum(K_hist * curr_pdf))
            C_path[t] = float(np.sum(policies_c[t] * curr_pdf))

        # Forward mass push via Young (2010) operator
        next_pdf = continuous_push_distribution(
            curr_pdf,
            policies_a[t],
            K_hist,
            shock_transition=P_z,
            shock_grid=z_grid,
        )

        # Enforce exact floating point mass conservation (eliminate roundoff drift <= 1e-15)
        mass = float(np.sum(next_pdf))
        if abs(mass - 1.0) > 1e-15:
            next_pdf = next_pdf / mass

        distributions[t + 1] = next_pdf
        curr_pdf = next_pdf

    return distributions, K_s_path, C_path


def _evaluate_transition_system(
    r_path: np.ndarray,
    Z_path: np.ndarray,
    mu_0: np.ndarray,
    c_term: np.ndarray,
    a_grid_dense: np.ndarray,
    K_hist: np.ndarray,
    P_z: np.ndarray,
    z_grid: np.ndarray,
    alpha: float,
    delta: float,
    L_agg: float,
    beta: float | np.ndarray,
    gamma: float,
    r_term: float,
    r_wedge: Optional[np.ndarray] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    """Evaluate market clearing residuals H_t = K_t^s - K_t^d given candidate interest rates."""
    T = len(r_path)

    # 1. Firm factor demand and wages
    # Firm capital demand: Kd = L * ((r + delta)/(alpha * Z)) ** (1/(alpha - 1))
    kl = ((r_path + delta) / (alpha * Z_path)) ** (1.0 / (alpha - 1.0))
    w_path = (1.0 - alpha) * Z_path * (kl ** alpha)
    K_d_path = L_agg * kl

    # Household effective interest rate (allows exogenous rate shock/wedge)
    r_hh_path = r_path + r_wedge if r_wedge is not None else r_path

    # 2. Backward EGM household policy solve
    policies_a, policies_c = _backward_egm_path(
        r_path=r_hh_path,
        w_path=w_path,
        c_term=c_term,
        a_grid_dense=a_grid_dense,
        K_hist=K_hist,
        P_z=P_z,
        z_grid=z_grid,
        beta=beta,
        gamma=gamma,
        r_term=r_term,
    )

    # 3. Forward Young distribution simulation
    distributions, K_s_path, C_path = _forward_young_path(
        mu_0=mu_0,
        policies_a=policies_a,
        policies_c=policies_c,
        K_hist=K_hist,
        P_z=P_z,
        z_grid=z_grid,
    )

    # 4. Market clearing residuals
    residuals = K_s_path - K_d_path

    return (
        residuals,
        K_s_path,
        K_d_path,
        w_path,
        C_path,
        distributions,
        policies_a,
        policies_c,
    )


def _solve_transition_shooting(
    eval_fn: Callable[[np.ndarray], Tuple[Any, ...]],
    r_init: np.ndarray,
    Z_path: np.ndarray,
    L_agg: float,
    alpha: float,
    delta: float,
    tol: float = 1e-4,
    max_iter: int = 100,
    damping: float = 0.3,
    r_min: float = 1e-4,
    r_max: float = 0.041,
) -> Tuple[np.ndarray, Tuple[Any, ...], int, bool]:
    """Solve transition path via damped fixed-point shooting."""
    r = r_init.copy()
    converged = False
    iterations = 0
    last_res: Tuple[Any, ...] = ()

    for it in range(max_iter + 1):
        res = eval_fn(r)
        last_res = res
        residuals = res[0]
        max_err = float(np.max(np.abs(residuals)))

        if max_err < tol:
            converged = True
            iterations = it
            break

        if it == max_iter:
            iterations = it
            break

        Ks = res[1]
        # Implied market clearing interest rate: r_implied = alpha * Z * (Ks / L)^(alpha - 1) - delta
        # (Ks = 0 gives +inf, which the clip below maps to r_max.)
        with np.errstate(divide="ignore"):
            r_implied = alpha * Z_path * ((Ks / L_agg) ** (alpha - 1.0)) - delta
        r_next = (1.0 - damping) * r + damping * r_implied
        r = np.clip(r_next, r_min, r_max)

    return r, last_res, iterations, converged


def _solve_transition_broyden(
    eval_fn: Callable[[np.ndarray], Tuple[Any, ...]],
    r_init: np.ndarray,
    alpha: float,
    delta: float,
    tol: float = 1e-4,
    max_iter: int = 100,
    backtracking: bool = True,
    damping: float = 0.3,
    r_min: float = 1e-4,
    r_max: float = 0.041,
) -> Tuple[np.ndarray, Tuple[Any, ...], int, bool]:
    """Solve transition path via sequence-space Broyden Quasi-Newton."""
    r = r_init.copy()
    res = eval_fn(r)
    residuals = res[0]
    Kd = res[2]

    norm = float(np.max(np.abs(residuals)))
    merit = float(np.linalg.norm(residuals))

    if norm < tol:
        return r, res, 0, True

    # Analytical diagonal Jacobian from firm demand:
    # dKd/dr = 1/(alpha - 1) * Kd / (r + delta) < 0
    # J_{0, tt} = - dKd/dr = 1/(1 - alpha) * Kd / (r + delta) > 0
    # Inverse Jacobian B_0 = diag((1 - alpha) * (r + delta) / Kd)
    B0 = np.diag((1.0 - alpha) * (r + delta) / Kd)
    B = B0.copy()

    converged = False
    iterations = 0
    last_res = res

    for it in range(1, max_iter + 1):
        dr = - B @ residuals
        step = 1.0
        accepted = False

        if backtracking:
            for _ in range(12):
                r_try = np.clip(r + step * dr, r_min, r_max)
                res_try = eval_fn(r_try)
                m_try = float(np.linalg.norm(res_try[0]))
                if m_try <= (1.0 - 1e-4 * step) * merit:
                    accepted = True
                    break
                step *= 0.5

        if not accepted:
            # Fallback: reset inverse Jacobian to B0 and take a damped step
            B = B0.copy()
            dr = - B @ residuals
            r_try = np.clip(r + damping * dr, r_min, r_max)
            res_try = eval_fn(r_try)
            m_try = float(np.linalg.norm(res_try[0]))

        delta_r = r_try - r
        delta_H = res_try[0] - residuals
        u = delta_r - B @ delta_H
        v = delta_r @ B
        denom = float(np.dot(v, delta_H))

        # Sherman-Morrison rank-1 update
        if abs(denom) > 1e-14:
            B = B + np.outer(u, v) / denom

        r = r_try
        last_res = res_try
        residuals = res_try[0]
        merit = m_try
        norm = float(np.max(np.abs(residuals)))
        iterations = it

        if norm < tol:
            converged = True
            break

    return r, last_res, iterations, converged


def _solve_terminal_steady_state(
    init_ss: AiyagariContinuousEquilibrium,
    Z_term: float,
    beta: float,
    gamma: float,
    alpha: float,
    delta: float,
    L_agg: float,
    K_hist: np.ndarray,
    a_grid_dense: np.ndarray,
    P_z: np.ndarray,
    z_grid: np.ndarray,
    r_bracket: Optional[Tuple[float, float]] = None,
    backend: str = "numpy",
    *,
    egm_tol: float = 1e-8,
    egm_max_iter: int = 10_000,
    xtol: float = 1e-8,
    tol_ge: float = 1e-4,
    max_evals: int = 100,
    dist_options: Optional[dict] = None,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve the terminal stationary general equilibrium under TFP ``Z_term``.

    The household problem, stopping rules, bracket and convergence criteria are
    those of ``solve_aiyagari_continuous`` (which has no TFP argument), with
    firm demand K^d = L ((r + delta) / (alpha Z))^(1/(alpha-1)) and wage
    w = (1 - alpha) Z (K/L)^alpha. At each trial rate the Coleman operator is
    iterated until sup|c_n - c_(n-1)| < ``egm_tol`` or ``egm_max_iter``
    applications; Brent's method on r stops at ``xtol``. Through 4.3.0 this
    solver capped the EGM at a hidden 500 iterations and used xtol=1e-6, so at
    Z_term = 1 in the n_z=3 documentation economy it returned r* = 0.03941949
    against the converged initial steady state's 0.03941451, with a consumption
    policy 0.09 away.

    Returns
    -------
    r_star, w_star, K_star, c_star, pdf_star, diagnostics
        ``c_star`` is the consumption policy on ``a_grid_dense`` (n_a, n_z);
        ``diagnostics`` holds ``converged`` (EGM met ``egm_tol`` at r*, the
        stationary distribution converged and |K^s - K^d| < ``tol_ge``) and the
        same counters ``solve_aiyagari_continuous`` records in its metadata.
    """
    n_a = len(a_grid_dense)
    N_k = len(K_hist)
    n_z = len(z_grid)
    dist_opts = dist_options or {}

    r_upper = 1.0 / beta - 1.0
    r_lower = -delta  # firm capital demand is unbounded as r -> -delta
    if r_bracket is None:
        lo, hi = 0.001, r_upper - 0.0005
    else:
        lo, hi = float(r_bracket[0]), float(r_bracket[1])

    egm_stats = {"solves": 0, "cap_hits": 0, "iterations": 0}

    def _eval_excess(r_cand: float) -> Tuple[float, float, float, np.ndarray, Any, bool, int, float]:
        kl = ((r_cand + delta) / (alpha * Z_term)) ** (1.0 / (alpha - 1.0))
        w = (1.0 - alpha) * Z_term * (kl ** alpha)
        Kd = L_agg * kl

        # Same first guess as solve_aiyagari_continuous: consume income, or cash
        # on hand at a negative rate, where income consumption is negative.
        c = np.zeros((n_a, n_z), dtype=np.float64)
        for m in range(n_z):
            if r_cand >= 0.0:
                c[:, m] = r_cand * a_grid_dense + w * z_grid[m]
            else:
                c[:, m] = (1.0 + r_cand) * a_grid_dense + w * z_grid[m]

        egm_ok = False
        step = float("inf")
        n_iter = 0
        for n_iter in range(1, egm_max_iter + 1):
            EMu = beta * (1.0 + r_cand) * (c ** (-gamma) @ P_z.T)
            c_endo = EMu ** (-1.0 / gamma)
            c_new = np.zeros_like(c)
            for m in range(n_z):
                a_endo = (c_endo[:, m] + a_grid_dense - w * z_grid[m]) / (1.0 + r_cand)
                c_interp = np.interp(a_grid_dense, a_endo, c_endo[:, m])
                binds = a_grid_dense < a_endo[0]
                c_interp[binds] = (1.0 + r_cand) * a_grid_dense[binds] + w * z_grid[m]
                c_new[:, m] = c_interp
            step = float(np.max(np.abs(c_new - c)))
            c = c_new
            if not np.isfinite(step):
                break
            if step < egm_tol:
                egm_ok = True
                break

        egm_stats["solves"] += 1
        egm_stats["iterations"] += n_iter
        if not egm_ok:
            egm_stats["cap_hits"] += 1

        ap_dense = np.maximum((1.0 + r_cand) * a_grid_dense[:, None] + w * z_grid[None, :] - c, 0.0)
        ap_hist = np.zeros((N_k, n_z), dtype=np.float64)
        for m in range(n_z):
            ap_hist[:, m] = np.interp(K_hist, a_grid_dense, ap_dense[:, m])

        dist = continuous_stationary_distribution(
            ap_hist, K_hist, shock_transition=P_z, shock_grid=z_grid, backend=backend, **dist_opts
        )
        Ks = float(dist.mean())
        excess = Ks - Kd
        return excess, w, Ks, c, dist, egm_ok, n_iter, step

    f_lo = _eval_excess(lo)[0]
    f_hi = _eval_excess(hi)[0]
    if r_bracket is None and f_lo > 0.0 and f_hi > 0.0:
        # Savings exceed capital demand at r = 1e-3: the terminal rate is lower.
        lo = r_lower + 1e-3
        f_lo = _eval_excess(lo)[0]
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"Terminal steady state at Z={Z_term:.6g}: excess capital supply K^s - K^d does not change "
            f"sign on r in [{lo:.4f}, {hi:.4f}] ({f_lo:+.4f} and {f_hi:+.4f}). Pass a solved "
            "terminal_steady_state instead."
        )

    r_star = float(brentq(lambda r: _eval_excess(r)[0], lo, hi, xtol=xtol, maxiter=max_evals))
    excess_star, w_star, Ks_star, c_star, dist_star, egm_ok, n_iter_star, step_star = _eval_excess(r_star)

    dist_ok = bool(dist_star.converged)
    clearing_ok = bool(np.isfinite(excess_star) and abs(excess_star) < tol_ge)
    reasons: list[str] = []
    if not egm_ok and not np.isfinite(step_star):
        reasons.append(
            f"the household EGM produced a non-finite consumption policy after {n_iter_star} iterations"
        )
    elif not egm_ok:
        reasons.append(
            f"the household EGM stopped after {n_iter_star} iterations (egm_max_iter={egm_max_iter}) "
            f"with sup|c_n - c_(n-1)| = {step_star:.3e}, not below egm_tol={egm_tol:.1e}; raise egm_max_iter"
        )
    if not dist_ok:
        reasons.append(f"the stationary distribution solver did not converge (dist_options={dist_options!r})")
    if not clearing_ok:
        reasons.append(
            f"|K^s - K^d| = {abs(excess_star):.3e} is not below tol_ge={tol_ge:.1e}; tighten xtol (now {xtol:.1e})"
        )

    diagnostics: dict[str, Any] = {
        "converged": egm_ok and dist_ok and clearing_ok,
        "Z": float(Z_term),
        "beta": float(beta),
        "r": r_star,
        "w": float(w_star),
        "K": Ks_star,
        "capital_market_clearing_error": float(excess_star),
        "egm_converged": bool(egm_ok),
        "egm_iterations": int(n_iter_star),
        "egm_residual": float(step_star),
        "egm_tol": float(egm_tol),
        "egm_max_iter": int(egm_max_iter),
        "egm_cap_hits": int(egm_stats["cap_hits"]),
        "household_solves": int(egm_stats["solves"]),
        "egm_iterations_total": int(egm_stats["iterations"]),
        "dist_converged": dist_ok,
        "clearing_ok": clearing_ok,
        "tol_ge": float(tol_ge),
        "xtol": float(xtol),
        "nonconvergence_reasons": reasons,
    }
    return r_star, float(w_star), Ks_star, c_star, dist_star.pdf, diagnostics


# ---------------------------------------------------------------------------
# Steady-state and parameter resolution
# ---------------------------------------------------------------------------

# Keywords a configuration dict may carry: the named parameters of
# solve_aiyagari_continuous (its **kwargs catch-all excluded).
_SS_CONFIG_KEYS = frozenset(
    name
    for name, p in inspect.signature(solve_aiyagari_continuous).parameters.items()
    if p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
)

# Structural parameters of the transition, with the solve_aiyagari_continuous defaults.
_STRUCTURAL_DEFAULTS = (("beta", 0.96), ("gamma", 2.0), ("alpha", 0.36), ("delta", 0.08))

# Controls of the internally solved terminal steady state, with the
# solve_aiyagari_continuous defaults used when the initial one records none.
_TERMINAL_CONTROL_DEFAULTS = (
    ("egm_tol", 1e-8),
    ("egm_max_iter", 10_000),
    ("xtol", 1e-8),
    ("tol_ge", 1e-4),
    ("max_evals", 100),
)

_KNOWN_KWARGS = frozenset(
    {name for name, _ in _STRUCTURAL_DEFAULTS}
    | {name for name, _ in _TERMINAL_CONTROL_DEFAULTS}
    | {"r_min", "r_max", "dist_options"}
)


def _resolve_steady_state(
    steady_state: Any, argname: str, backend: str
) -> AiyagariContinuousEquilibrium:
    """Return ``steady_state`` if solved, or solve the configuration dict it holds.

    A dict is a configuration for ``solve_aiyagari_continuous`` and must hold
    only its keywords. Through 4.3.0 other keys (for example the ``k_grid``,
    ``pdf_ss``, ``policy_a`` or ``r_ss`` of a steady state solved elsewhere)
    were silently dropped, and the transition started from a freshly solved
    economy with the remaining, often default, parameters.
    """
    if isinstance(steady_state, AiyagariContinuousEquilibrium):
        return steady_state
    if isinstance(steady_state, dict):
        unknown = sorted(str(k) for k in steady_state if k not in _SS_CONFIG_KEYS)
        if unknown:
            raise TypeError(
                f"{argname}: a dict is a configuration passed to solve_aiyagari_continuous, which solves "
                f"a new steady state, and it does not accept the key(s) {', '.join(map(repr, unknown))}. "
                "To start from a steady state you have already solved, pass the "
                "AiyagariContinuousEquilibrium that solve_aiyagari_continuous returned. Accepted keys: "
                f"{', '.join(sorted(_SS_CONFIG_KEYS))}."
            )
        config = dict(steady_state)
        config.setdefault("backend", backend)
        return solve_aiyagari_continuous(**config)
    raise TypeError(
        f"{argname} must be an AiyagariContinuousEquilibrium or a configuration dict, "
        f"got {type(steady_state).__name__}"
    )


def _structural_parameters(
    init_ss: AiyagariContinuousEquilibrium, kwargs: dict[str, Any]
) -> dict[str, float]:
    """beta, gamma, alpha and delta of the initial steady state.

    They are read from ``init_ss.metadata["params"]``, which
    ``solve_aiyagari_continuous`` records. A keyword that contradicts the
    recorded value raises ValueError. When the steady state records none (one
    built with ``continuous_stationary_equilibrium`` or by hand), the keywords
    are used, and any that is missing falls back to the
    ``solve_aiyagari_continuous`` default with a UserWarning. Through 4.3.0 the
    keywords or those defaults were always used, so a steady state solved with
    beta=0.99 was paired with beta=0.96 transition dynamics.
    """
    meta = init_ss.metadata if isinstance(init_ss.metadata, dict) else {}
    recorded = meta.get("params") or {}
    out: dict[str, float] = {}
    missing: list[str] = []
    for name, default in _STRUCTURAL_DEFAULTS:
        ss_val = recorded.get(name)
        kw_val = kwargs.pop(name, None)
        if ss_val is not None:
            ss_val = float(ss_val)
            if kw_val is not None and abs(float(kw_val) - ss_val) > 1e-12 * max(1.0, abs(ss_val)):
                raise ValueError(
                    f"{name}={kw_val!r} contradicts the initial steady state, which was solved with "
                    f"{name}={ss_val!r}. The transition uses the steady state's parameters; to study a "
                    f"different {name}, solve the steady state with it."
                )
            out[name] = ss_val
        elif kw_val is not None:
            out[name] = float(kw_val)
        else:
            out[name] = float(default)
            missing.append(name)
    if missing:
        warnings.warn(
            "solve_continuous_transition: the initial steady state does not record "
            + ", ".join(missing)
            + " in metadata['params'] and they were not passed as keywords; using "
            + ", ".join(f"{n}={out[n]}" for n in missing)
            + ". Pass them explicitly if the steady state was solved with other values.",
            UserWarning,
            stacklevel=_caller_stacklevel(),
        )
    return out


def _terminal_controls(
    init_ss: AiyagariContinuousEquilibrium, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Controls for the internally solved terminal steady state.

    Keywords win; otherwise the values recorded in ``init_ss.metadata`` by
    ``solve_aiyagari_continuous``, so both steady states are solved to the same
    accuracy; otherwise that function's defaults.
    """
    meta = init_ss.metadata if isinstance(init_ss.metadata, dict) else {}
    out: dict[str, Any] = {}
    for name, default in _TERMINAL_CONTROL_DEFAULTS:
        out[name] = kwargs.pop(name, meta.get(name, default))
    out["egm_tol"] = _positive_finite("egm_tol", out["egm_tol"])
    out["xtol"] = _positive_finite("xtol", out["xtol"])
    out["tol_ge"] = _positive_finite("tol_ge", out["tol_ge"])
    for name in ("egm_max_iter", "max_evals"):
        value = out[name]
        try:
            ok = int(value) == value and int(value) >= 1
        except (TypeError, ValueError, OverflowError):
            ok = False
        if not ok:
            raise ValueError(f"{name} must be an integer >= 1; got {value!r}")
        out[name] = int(value)
    out["dist_options"] = kwargs.pop("dist_options", None)
    return out


# ---------------------------------------------------------------------------
# Public API Functions
# ---------------------------------------------------------------------------

def solve_continuous_transition(
    initial_steady_state: AiyagariContinuousEquilibrium | dict[str, Any],
    terminal_steady_state: Optional[AiyagariContinuousEquilibrium | dict[str, Any]] = None,
    shock_path: Optional[np.ndarray | Sequence[float]] = None,
    shock_var: str = "z",
    horizon: int = 150,
    solver: str = "broyden",
    damping: float = 0.3,
    tol: float = 1e-4,
    max_iter: int = 100,
    backtracking: bool = True,
    backend: str = "numpy",
    r_init_path: Optional[np.ndarray | Sequence[float]] = None,
    **kwargs: Any,
) -> ContinuousTransitionResult:
    """Solve the non-linear transition path of continuous wealth distributions under MIT shocks.

    Parameters
    ----------
    initial_steady_state : AiyagariContinuousEquilibrium or dict
        Pre-solved initial general equilibrium, or a configuration dict of
        ``solve_aiyagari_continuous`` keywords, which is solved here. A dict with
        any other key (for example the grids, distribution or prices of a steady
        state solved elsewhere) raises TypeError; through 4.3.0 such keys were
        dropped silently and a default economy was solved instead. The
        structural parameters beta, gamma, alpha and delta are read from
        ``metadata["params"]`` of this steady state (see ``**kwargs``).
    terminal_steady_state : AiyagariContinuousEquilibrium or dict, optional
        Pre-solved terminal general equilibrium (a dict as above). If None, it
        is the initial steady state when the shock has died out by date T-1,
        and it is solved here, with the controls listed under ``**kwargs``,
        when Z_{T-1} differs from 1 or beta_{T-1} from beta by more than
        ``numpy.isclose(..., atol=1e-6)`` allows (about 1e-5).
    shock_path : np.ndarray or Sequence[float], optional
        Exogenous shock trajectory over the horizon T. If None, represents a zero shock
        (steady-state invariance check).
    shock_var : {'z', 'tfp', 'r', 'rate', 'beta', 'discount'}, default 'z'
        Variable targeted by the MIT shock:
        - 'z' or 'tfp': Aggregate total factor productivity Z_t.
        - 'r' or 'rate': Exogenous interest rate wedge / monetary shock.
        - 'beta' or 'discount': Discount factor shock beta_t, the factor that
          discounts date t+1 in the date-t Euler equation. Values above 0.5
          are levels of beta_t; smaller values s_t give beta_t = beta (1 + s_t).
          Through 4.3.0 the path was built but never used, so a discount
          factor shock returned the steady state.
    horizon : int, default 150
        Number of transition periods T.
    solver : {'broyden', 'shooting'}, default 'broyden'
        Algorithm for computing the market clearing interest rate path:
        - 'broyden': Sequence-space Quasi-Newton with Sherman-Morrison rank-1 updates.
        - 'shooting': Damped fixed-point iteration.
    damping : float, default 0.3
        Damping parameter for shooting relaxation or Broyden line search fallback.
    tol : float, default 1e-4
        Convergence tolerance on the market clearing residual ||K^s - K^d||_inf.
    max_iter : int, default 100
        Maximum number of price relaxation iterations.
    backtracking : bool, default True
        Whether to use monotone backtracking line search in Broyden's method.
    backend : str, default 'numpy'
        Acceleration backend.
    r_init_path : np.ndarray or Sequence[float], optional
        Optional initial guess for the interest rate trajectory.
    **kwargs : Any
        - ``beta``, ``gamma``, ``alpha``, ``delta``: used only when the initial
          steady state does not record them in ``metadata["params"]`` (one built
          with ``continuous_stationary_equilibrium`` or by hand); a missing one
          then falls back to the ``solve_aiyagari_continuous`` default with a
          UserWarning. A value that contradicts the recorded one raises
          ValueError. Through 4.3.0 these keywords, or the defaults 0.96, 2.0,
          0.36 and 0.08, were always used, whatever the steady state.
        - ``egm_tol``, ``egm_max_iter``, ``xtol``, ``tol_ge``, ``max_evals``,
          ``dist_options``: controls of the terminal steady state solved here,
          as in ``solve_aiyagari_continuous``. Unless given, they are taken from
          the initial steady state's metadata, then from that function's
          defaults (1e-8, 10000, 1e-8, 1e-4, 100). Through 4.3.0 that solve
          stopped the household EGM at a hidden 500 iterations and used
          xtol=1e-6.
        - ``r_min``, ``r_max``: bounds on the trial interest rate path.
        Any other keyword is ignored with a FutureWarning; it will raise
        TypeError in a future release.

    Returns
    -------
    ContinuousTransitionResult
        Frozen dataclass holding equilibrium price paths, capital stocks,
        consumption, full sequence of wealth distributions, and presentation methods.
        ``converged`` is True when the relaxation met ``tol`` and, if a terminal
        steady state was solved here, that solve converged (a RuntimeWarning
        names the failed criteria otherwise). ``metadata`` records ``params``,
        ``relaxation_converged`` and, when solved here,
        ``terminal_steady_state`` with its diagnostics.

    Raises
    ------
    TypeError
        For a steady-state dict with keys ``solve_aiyagari_continuous`` does not
        accept.
    ValueError
        For invalid inputs, or a structural keyword that contradicts the
        initial steady state.
    """
    t_start = time.time()

    # Input validation
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"horizon must be a positive integer >= 1; got {horizon}")
    max_iter = int(max_iter)
    if max_iter < 0:
        raise ValueError(f"max_iter must be non-negative; got {max_iter}")
    tol = float(tol)
    if tol <= 0:
        raise ValueError(f"tol must be strictly positive; got {tol}")

    s_var = str(shock_var).lower().strip()
    valid_shock_vars = ("z", "tfp", "productivity", "r", "rate", "monetary", "beta", "discount")
    if s_var not in valid_shock_vars:
        raise ValueError(f"Invalid shock_var {shock_var!r}. Must be one of {valid_shock_vars}")

    s_solver = str(solver).lower().strip()
    if s_solver not in ("broyden", "shooting"):
        raise ValueError(f"Invalid solver {solver!r}. Must be 'broyden' or 'shooting'")

    kwargs = dict(kwargs)
    unknown_kwargs = sorted(k for k in kwargs if k not in _KNOWN_KWARGS)
    if unknown_kwargs:
        warnings.warn(
            "solve_continuous_transition: unknown keyword argument(s) "
            + ", ".join(repr(k) for k in unknown_kwargs)
            + " ignored. They have never had an effect and will raise TypeError in a future release. "
            f"Known keywords: {', '.join(sorted(_KNOWN_KWARGS))}.",
            FutureWarning,
            stacklevel=_caller_stacklevel(),
        )
        for k in unknown_kwargs:
            kwargs.pop(k)

    # 1. Resolve initial steady state
    init_ss = _resolve_steady_state(initial_steady_state, "initial_steady_state", backend)

    # Extract model parameters from initial steady state
    hh_init = init_ss.household_solution
    P_z = np.asarray(hh_init.P_z, dtype=np.float64)
    z_grid = np.asarray(hh_init.z_grid, dtype=np.float64)
    K_hist = np.asarray(hh_init.a_grid, dtype=np.float64)
    N_k = len(K_hist)
    n_z = len(z_grid)
    L_agg = float(init_ss.L)
    r_ss_init = float(init_ss.r)
    w_ss_init = float(init_ss.w)
    K_ss_init = float(init_ss.K)

    params = _structural_parameters(init_ss, kwargs)
    beta = params["beta"]
    gamma = params["gamma"]
    alpha = params["alpha"]
    delta = params["delta"]
    terminal_controls = _terminal_controls(init_ss, kwargs)

    # Dense asset grid for continuous EGM
    n_a = len(hh_init.policy_c)
    a_max = float(K_hist[-1])
    a_grid_dense = a_max * (np.linspace(0.0, 1.0, n_a) ** 1.5)

    # 2. Parse shock path
    is_tfp_shock = s_var in ("z", "tfp", "productivity")
    is_rate_shock = s_var in ("r", "rate", "monetary")
    is_beta_shock = s_var in ("beta", "discount")

    Z_path = np.ones(horizon, dtype=np.float64)
    r_wedge: Optional[np.ndarray] = None
    beta_path = np.full(horizon, beta, dtype=np.float64)

    is_zero_shock = False
    if shock_path is None:
        is_zero_shock = True
    else:
        s_arr = np.asarray(shock_path, dtype=np.float64).ravel()
        if len(s_arr) > horizon:
            s_arr = s_arr[:horizon]

        if is_tfp_shock:
            # If values passed are around 0 (e.g. +0.05), treat as 1 + shock;
            # if values are around 1 (e.g. 1.05), treat as Z directly
            if np.max(s_arr) > 0.5:
                Z_path[:len(s_arr)] = s_arr
                if len(s_arr) < horizon:
                    Z_path[len(s_arr):] = s_arr[-1]
            else:
                Z_path[:len(s_arr)] = 1.0 + s_arr
                if len(s_arr) < horizon:
                    Z_path[len(s_arr):] = 1.0 + s_arr[-1]
            is_zero_shock = bool(np.allclose(Z_path, 1.0, atol=1e-12))
        elif is_rate_shock:
            r_wedge = np.zeros(horizon, dtype=np.float64)
            r_wedge[:len(s_arr)] = s_arr
            is_zero_shock = bool(np.allclose(r_wedge, 0.0, atol=1e-12))
        elif is_beta_shock:
            if np.max(s_arr) > 0.5:
                beta_path[:len(s_arr)] = s_arr
            else:
                beta_path[:len(s_arr)] = beta * (1.0 + s_arr)
            is_zero_shock = bool(np.allclose(beta_path, beta, atol=1e-12))

    # 3. Resolve terminal steady state
    Z_term = float(Z_path[-1])
    beta_term = float(beta_path[-1])
    is_permanent_tfp = is_tfp_shock and not np.isclose(Z_term, 1.0, atol=1e-6)
    is_permanent_beta = is_beta_shock and not np.isclose(beta_term, beta, atol=1e-6)
    terminal_diag: Optional[dict[str, Any]] = None

    if terminal_steady_state is not None:
        term_ss = _resolve_steady_state(terminal_steady_state, "terminal_steady_state", backend)
        r_term = float(term_ss.r)
        w_term = float(term_ss.w)
        c_term = np.asarray(term_ss.household_solution.policy_c, dtype=np.float64)
    elif is_permanent_tfp or is_permanent_beta:
        r_term, w_term, _, c_term, _, terminal_diag = _solve_terminal_steady_state(
            init_ss=init_ss,
            Z_term=Z_term,
            beta=beta_term,
            gamma=gamma,
            alpha=alpha,
            delta=delta,
            L_agg=L_agg,
            K_hist=K_hist,
            a_grid_dense=a_grid_dense,
            P_z=P_z,
            z_grid=z_grid,
            backend=backend,
            **terminal_controls,
        )
        if not terminal_diag["converged"]:
            warnings.warn(
                f"solve_continuous_transition: the terminal steady state (Z={Z_term:.6g}, "
                f"beta={beta_term:.6g}) did not converge at r*={r_term:.6f}: "
                + "; ".join(terminal_diag["nonconvergence_reasons"])
                + ".",
                RuntimeWarning,
                stacklevel=_caller_stacklevel(),
            )
    else:
        # Transitory shock or zero shock: terminal steady state identical to initial
        r_term = r_ss_init
        w_term = w_ss_init
        c_term = np.asarray(hh_init.policy_c, dtype=np.float64)

    # 4. Construct initial interest rate path guess
    r_upper_bound = 1.0 / beta - 1.0
    r_min = float(kwargs.get("r_min", 1e-4))
    r_max = float(kwargs.get("r_max", max(0.15, r_upper_bound + 0.10)))

    if r_init_path is not None:
        r_init = np.asarray(r_init_path, dtype=np.float64).ravel()
        if len(r_init) < horizon:
            r_full = np.full(horizon, r_term)
            r_full[:len(r_init)] = r_init
            r_init = r_full
        else:
            r_init = r_init[:horizon]
    elif is_zero_shock:
        r_init = np.full(horizon, r_ss_init, dtype=np.float64)
    elif is_permanent_tfp:
        # At t=0, capital stock K0 is predetermined.
        # Implied initial interest rate from firm demand:
        r0_implied = alpha * Z_path[0] * ((K_ss_init / L_agg) ** (alpha - 1.0)) - delta
        r_init = np.linspace(r0_implied, r_term, horizon)
    else:
        # Transitory shock
        if is_rate_shock and r_wedge is not None:
            r_init = np.full(horizon, r_ss_init, dtype=np.float64)
        else:
            r0_implied = alpha * Z_path[0] * ((K_ss_init / L_agg) ** (alpha - 1.0)) - delta
            r_init = r_term + (r0_implied - r_term) * (0.85 ** np.arange(horizon))

    r_init = np.clip(r_init, r_min, r_max)

    # 5. Define evaluation function closure
    def _eval_system(r_arr: np.ndarray) -> Tuple[Any, ...]:
        return _evaluate_transition_system(
            r_path=r_arr,
            Z_path=Z_path,
            mu_0=init_ss.distribution.pdf,
            c_term=c_term,
            a_grid_dense=a_grid_dense,
            K_hist=K_hist,
            P_z=P_z,
            z_grid=z_grid,
            alpha=alpha,
            delta=delta,
            L_agg=L_agg,
            beta=beta_path,
            gamma=gamma,
            r_term=r_term,
            r_wedge=r_wedge,
        )

    # Fast zero-shock steady state check
    if is_zero_shock and np.isclose(r_term, r_ss_init, atol=1e-10):
        initial_eval = _eval_system(r_init)
        resids = initial_eval[0]
        max_res = float(np.max(np.abs(resids)))
        if max_res < tol:
            elapsed = time.time() - t_start
            meta = {
                "solver": s_solver,
                "horizon": horizon,
                "shock_var": s_var,
                "asset_grid": K_hist,
                "mass_error": 0.0,
                "backend": backend,
                "params": dict(params),
                "relaxation_converged": True,
                "terminal_steady_state": terminal_diag,
            }
            return ContinuousTransitionResult(
                r_path=r_init,
                w_path=initial_eval[3],
                K_s_path=initial_eval[1],
                K_d_path=initial_eval[2],
                C_path=initial_eval[4],
                distributions=initial_eval[5],
                policies_a=initial_eval[6],
                policies_c=initial_eval[7],
                residuals=resids,
                max_residual=max_res,
                iterations=0,
                converged=True,
                elapsed_time=elapsed,
                metadata=meta,
            )

    # 6. Execute Relaxation Solver
    if s_solver == "broyden":
        r_sol, last_eval, iters, converged = _solve_transition_broyden(
            eval_fn=_eval_system,
            r_init=r_init,
            alpha=alpha,
            delta=delta,
            tol=tol,
            max_iter=max_iter,
            backtracking=backtracking,
            damping=damping,
            r_min=r_min,
            r_max=r_max,
        )
    else:
        r_sol, last_eval, iters, converged = _solve_transition_shooting(
            eval_fn=_eval_system,
            r_init=r_init,
            Z_path=Z_path,
            L_agg=L_agg,
            alpha=alpha,
            delta=delta,
            tol=tol,
            max_iter=max_iter,
            damping=damping,
            r_min=r_min,
            r_max=r_max,
        )

    elapsed = time.time() - t_start

    residuals = last_eval[0]
    Ks_path = last_eval[1]
    Kd_path = last_eval[2]
    w_path = last_eval[3]
    C_path = last_eval[4]
    distributions = last_eval[5]
    policies_a = last_eval[6]
    policies_c = last_eval[7]
    max_res = float(np.max(np.abs(residuals)))

    if not converged:
        warnings.warn(
            f"solve_continuous_transition did not converge: max residual {max_res:.2e} >= tol {tol:.2e} "
            f"after {iters} iterations. Try increasing horizon, damping, or max_iter.",
            RuntimeWarning,
            stacklevel=_caller_stacklevel(),
        )

    terminal_ok = terminal_diag is None or bool(terminal_diag["converged"])

    meta = {
        "solver": s_solver,
        "horizon": horizon,
        "shock_var": s_var,
        "asset_grid": K_hist,
        "Z_path": Z_path,
        "beta_path": beta_path,
        "backend": backend,
        "params": dict(params),
        "relaxation_converged": bool(converged),
        "terminal_steady_state": terminal_diag,
    }

    return ContinuousTransitionResult(
        r_path=r_sol,
        w_path=w_path,
        K_s_path=Ks_path,
        K_d_path=Kd_path,
        C_path=C_path,
        distributions=distributions,
        policies_a=policies_a,
        policies_c=policies_c,
        residuals=residuals,
        max_residual=max_res,
        iterations=iters,
        converged=bool(converged) and terminal_ok,
        elapsed_time=elapsed,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# continuous_mit_shock helpers
# ---------------------------------------------------------------------------

# shock_type aliases accepted by solve_continuous_transition, by shock class.
_MIT_SHOCK_KINDS = {
    "z": "tfp", "tfp": "tfp", "productivity": "tfp",
    "r": "rate", "rate": "rate", "monetary": "rate",
    "beta": "beta", "discount": "beta",
}


def _mit_shock_kind(shock_type: Any) -> str:
    """'tfp', 'rate' or 'beta' for a ``shock_type`` alias; ValueError otherwise."""
    key = str(shock_type).lower().strip()
    if key not in _MIT_SHOCK_KINDS:
        raise ValueError(f"Invalid shock_type {shock_type!r}. Must be one of {tuple(_MIT_SHOCK_KINDS)}")
    return _MIT_SHOCK_KINDS[key]


def _mit_solver_path(base: float, rel: np.ndarray, name: str) -> np.ndarray:
    """A path that ``solve_continuous_transition`` reads as ``base * (1 + rel)``.

    That function reads a TFP or discount-factor path as levels when any entry
    exceeds 0.5 and as relative deviations otherwise. Through 4.3.0 this
    wrapper always passed the relative path, so ``shock_size=0.6`` gave
    Z_0 = 0.6 (a 40% fall) instead of 1.6, and a discount-factor shock of 0.6
    gave beta_0 = 0.6. The levels are passed whenever they are read as levels,
    which reproduces the old path bit for bit for every shock up to 0.5.
    """
    levels = base * (1.0 + rel)
    if float(np.max(levels)) > 0.5:
        return levels
    if float(np.max(rel)) <= 0.5:
        return rel
    raise ValueError(
        f"continuous_mit_shock: the {name} path {base} * (1 + s_t) cannot be passed unambiguously "
        "to solve_continuous_transition; build the path and call that function directly."
    )


def _mit_transitory_rate_guess(
    init_ss: AiyagariContinuousEquilibrium, Z0: float, alpha: float, delta: float, horizon: int
) -> np.ndarray:
    """The starting rate path ``solve_continuous_transition`` uses for a transitory TFP shock.

    r_t = r* + (r_0 - r*) 0.85^t, where r_0 is the firm's rate at the
    predetermined K* and Z_0 (same formula, same floating-point operations).
    That function switches to a straight line toward the terminal rate once
    Z_(T-1) differs from 1, so passing this guess gives a transitory shock the
    same starting point whatever the persistence.
    """
    r_ss = float(init_ss.r)
    r0 = alpha * Z0 * ((float(init_ss.K) / float(init_ss.L)) ** (alpha - 1.0)) - delta
    return r_ss + (r0 - r_ss) * (0.85 ** np.arange(horizon))


def _mit_horizon_needed(persistence: float, truncation_tol: float) -> int:
    """Smallest horizon T with persistence**(T-1) <= truncation_tol."""
    if truncation_tol >= 1.0:
        return 1
    if persistence <= 0.0:
        return 2
    T = 1 + max(0, int(np.ceil(np.log(truncation_tol) / np.log(persistence))))
    while persistence ** (T - 1) > truncation_tol:
        T += 1
    while T > 1 and persistence ** (T - 2) <= truncation_tol:
        T -= 1
    return T


def continuous_mit_shock(
    steady_state: AiyagariContinuousEquilibrium | dict[str, Any],
    shock_type: str = "tfp",
    shock_size: float = 0.05,
    persistence: float = 0.8,
    horizon: int = 150,
    solver: str = "broyden",
    *,
    truncation_tol: float = 1e-3,
    **kwargs: Any,
) -> ContinuousTransitionResult:
    """Convenience entry point for simulating unexpected aggregate MIT shocks.

    The shock is s_t = shock_size * persistence**t for t = 0, ..., T-1.

    - ``persistence < 1`` is a transitory shock. The terminal condition at date
      T is always the initial steady state, and the shock is set to zero from T
      on. If the shock left at T-1, a share ``persistence**(T-1)`` of the
      impact, exceeds ``truncation_tol``, a RuntimeWarning names the horizon
      that would bring it below the tolerance; the truncation is reported in
      ``metadata["mit_shock"]`` either way. Through 4.3.0 a transitory shock
      whose Z_(T-1) (or beta_(T-1)/beta) differed from 1 by more than about
      1e-5 was silently treated as permanent: a terminal steady state was
      solved at Z_(T-1), so for T=40 and shock_size=0.05 the model changed at
      persistence 0.806, and at persistence 0.92 the economy converged to a
      steady state with 0.19% higher TFP.
    - ``persistence == 1`` is a permanent shock and must be asked for
      explicitly. For 'tfp' and 'beta' a terminal steady state is solved at
      Z = 1 + shock_size or beta (1 + shock_size). A permanent 'rate' wedge is
      not supported: the terminal condition stays the initial steady state,
      and a RuntimeWarning says so.

    Parameters
    ----------
    steady_state : AiyagariContinuousEquilibrium or dict
        Initial general equilibrium state (a dict is a configuration for
        ``solve_aiyagari_continuous``, as in ``solve_continuous_transition``).
    shock_type : {'tfp', 'rate', 'beta'}, default 'tfp'
        Type of aggregate shock (the aliases of ``solve_continuous_transition``'s
        ``shock_var`` are accepted):
        - 'tfp': Total factor productivity, Z_t = 1 + s_t.
        - 'rate': Interest rate wedge earned by households, r_t + s_t.
        - 'beta': Discount factor, beta_t = beta (1 + s_t).
    shock_size : float, default 0.05
        Impact of the shock s_0 (e.g. +0.05 for +5% TFP, or +0.01 for +100 bps
        rate). Always a deviation, never a level: through 4.3.0 a TFP or
        discount-factor shock above 0.5 was read as a level. A TFP shock must
        be above -1, so that Z_0 > 0.
    persistence : float, default 0.8
        Persistence rho in [0.0, 1.0]: 1.0 is a permanent shock, anything
        below is transitory (see above).
    horizon : int, default 150
        Length T of simulation horizon.
    solver : {'broyden', 'shooting'}, default 'broyden'
        Equilibrium relaxation solver.
    truncation_tol : float, default 1e-3
        Largest share of the impact that a transitory shock may still have at
        date T-1 without a truncation warning.
    **kwargs : Any
        Passed to ``solve_continuous_transition``. ``damping`` is the
        relaxation weight of ``solver='shooting'``. With ``solver='broyden'``
        it only scales the fallback step taken when the backtracking line
        search fails to reduce ||H||_2 in 12 halvings (and every step when
        ``backtracking=False``, which also discards the rank-one updates). In
        every run of notebook 52 the line search never failed, and damping
        from 0.2 to 0.8 gave bit-identical Broyden paths. A
        ``terminal_steady_state`` passed here is used as given, for any
        persistence.

    Returns
    -------
    ContinuousTransitionResult
        Solved non-linear transition dynamics. ``metadata["mit_shock"]``
        records ``shock_type``, ``shock_size``, ``persistence``,
        ``permanent``, ``terminal_condition`` ('initial_steady_state',
        'solved_steady_state' or 'user'), ``shock_at_last_date`` (s_(T-1)),
        ``remaining_share`` (persistence**(T-1)), ``truncation_tol``,
        ``truncated``, ``horizon_needed`` (the smallest T with
        persistence**(T-1) <= truncation_tol; None for a permanent shock) and
        ``capital_gap_last`` (K_(T-1) - K* for a transitory shock, a separate
        measure of how far the endogenous response is from over).
    """
    persistence = float(persistence)
    if not (0.0 <= persistence <= 1.0):
        raise ValueError(f"persistence must be in [0.0, 1.0]; got {persistence}")
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"horizon must be a positive integer >= 1; got {horizon}")
    truncation_tol = _positive_finite("truncation_tol", truncation_tol)
    shock_size = float(shock_size)
    if not np.isfinite(shock_size):
        raise ValueError(f"shock_size must be finite; got {shock_size}")
    kind = _mit_shock_kind(shock_type)
    if kind == "tfp" and shock_size <= -1.0:
        raise ValueError(f"a TFP shock must be above -1 so that Z_0 = 1 + shock_size > 0; got {shock_size}")

    kwargs = dict(kwargs)
    init_ss = _resolve_steady_state(steady_state, "steady_state", kwargs.get("backend", "numpy"))
    # The transition reads beta, alpha and delta the same way; any warning or
    # error about them is raised by solve_continuous_transition below.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = _structural_parameters(init_ss, dict(kwargs))

    permanent = persistence == 1.0
    if permanent:
        rel = np.full(horizon, shock_size, dtype=np.float64)
    else:
        rel = shock_size * (persistence ** np.arange(horizon, dtype=np.float64))
    if kind == "tfp":
        shock_path = _mit_solver_path(1.0, rel, "TFP")
    elif kind == "beta":
        shock_path = _mit_solver_path(params["beta"], rel, "discount-factor")
    else:
        shock_path = rel

    user_terminal = kwargs.get("terminal_steady_state") is not None
    if user_terminal:
        terminal_condition = "user"
    elif permanent and kind != "rate":
        terminal_condition = "solved_steady_state"
    else:
        terminal_condition = "initial_steady_state"
        kwargs["terminal_steady_state"] = init_ss
        # (A zero shock keeps the solver's exact steady-state path.)
        if (
            kind == "tfp"
            and not permanent
            and kwargs.get("r_init_path") is None
            and not np.allclose(1.0 + rel, 1.0, atol=1e-12)
        ):
            kwargs["r_init_path"] = _mit_transitory_rate_guess(
                init_ss, 1.0 + float(rel[0]), params["alpha"], params["delta"], horizon
            )

    res = solve_continuous_transition(
        initial_steady_state=init_ss,
        shock_path=shock_path,
        shock_var=shock_type,
        horizon=horizon,
        solver=solver,
        **kwargs,
    )

    remaining_share = float(persistence ** (horizon - 1))
    if permanent:
        truncated = kind == "rate" and shock_size != 0.0
        horizon_needed: Optional[int] = None
    else:
        truncated = shock_size != 0.0 and remaining_share > truncation_tol
        horizon_needed = _mit_horizon_needed(persistence, truncation_tol)
    res.metadata["mit_shock"] = {
        "shock_type": kind,
        "shock_size": shock_size,
        "persistence": persistence,
        "permanent": permanent,
        "terminal_condition": terminal_condition,
        "shock_at_last_date": float(rel[-1]),
        "remaining_share": remaining_share,
        "truncation_tol": truncation_tol,
        "truncated": bool(truncated),
        "horizon_needed": horizon_needed,
        "capital_gap_last": None if permanent else float(res.K_s_path[-1] - float(init_ss.K)),
    }

    if truncated and not user_terminal:
        if permanent:
            message = (
                "continuous_mit_shock: a permanent rate wedge (persistence=1) is not supported. The "
                "terminal condition is the initial steady state, which carries no wedge, so the path is "
                "solved against an inconsistent terminal condition; use a wedge that dies out by T-1."
            )
        else:
            message = (
                f"continuous_mit_shock: the transitory {kind} shock has not died out by the last date: "
                f"s_(T-1) = {rel[-1]:.3g}, {remaining_share:.2%} of the impact (persistence {persistence}, "
                f"horizon {horizon}). The terminal condition is the initial steady state, so the shock is "
                f"truncated to zero from date T = {horizon} on. horizon={horizon_needed} brings it below "
                f"truncation_tol={truncation_tol:g}; persistence=1.0 asks for a permanent shock."
            )
        warnings.warn(message, RuntimeWarning, stacklevel=_caller_stacklevel())

    return res


__all__ = [
    "TransitionShock",
    "ContinuousTransitionResult",
    "solve_continuous_transition",
    "continuous_mit_shock",
]
