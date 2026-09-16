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
    beta: float = 0.96,
    gamma: float = 2.0,
    r_term: float = 0.02,
) -> Tuple[list[np.ndarray], list[np.ndarray]]:
    """Solve household policy functions backward in time via EGM.

    Given factor prices {r_t, w_t}_{t=0}^{T-1} and continuation consumption
    policy c_T(k, z) = c_term, performs backward time iteration:
        EMu_t(a', z_m) = beta * (1 + r_{t+1}) * sum_{z'} P_z(z_m, z') * c_{t+1}(a', z')^{-gamma}
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
    beta : float
        Discount factor.
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

    policies_a = [None] * T
    policies_c = [None] * T

    c_next = np.asarray(c_term, dtype=np.float64).copy()

    for t in range(T - 1, -1, -1):
        r_t = float(r_path[t])
        w_t = float(w_path[t])
        r_tp1 = float(r_path[t + 1]) if t < T - 1 else float(r_term)

        # Expected marginal utility of saving a'
        # EMu has shape (n_a, n_z)
        EMu = beta * (1.0 + r_tp1) * (c_next ** (-gamma) @ P_z.T)
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
    beta: float,
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
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Solve the terminal stationary general equilibrium under TFP Z_term."""
    n_a = len(a_grid_dense)
    N_k = len(K_hist)
    n_z = len(z_grid)

    r_upper = 1.0 / beta - 1.0
    lo = 0.001 if r_bracket is None else r_bracket[0]
    hi = (r_upper - 0.0005) if r_bracket is None else r_bracket[1]

    def _eval_excess(r_cand: float) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        kl = ((r_cand + delta) / (alpha * Z_term)) ** (1.0 / (alpha - 1.0))
        w = (1.0 - alpha) * Z_term * (kl ** alpha)
        Kd = L_agg * kl

        c = np.zeros((n_a, n_z), dtype=np.float64)
        for m in range(n_z):
            c[:, m] = r_cand * a_grid_dense + w * z_grid[m]

        for _ in range(500):
            c_old = c.copy()
            EMu = beta * (1.0 + r_cand) * (c ** (-gamma) @ P_z.T)
            c_endo = EMu ** (-1.0 / gamma)
            c_new = np.zeros_like(c)
            for m in range(n_z):
                a_endo = (c_endo[:, m] + a_grid_dense - w * z_grid[m]) / (1.0 + r_cand)
                c_interp = np.interp(a_grid_dense, a_endo, c_endo[:, m])
                binds = a_grid_dense < a_endo[0]
                c_interp[binds] = (1.0 + r_cand) * a_grid_dense[binds] + w * z_grid[m]
                c_new[:, m] = c_interp
            if np.max(np.abs(c_new - c_old)) < 1e-8:
                break
            c = c_new

        ap_dense = np.maximum((1.0 + r_cand) * a_grid_dense[:, None] + w * z_grid[None, :] - c, 0.0)
        ap_hist = np.zeros((N_k, n_z), dtype=np.float64)
        for m in range(n_z):
            ap_hist[:, m] = np.interp(K_hist, a_grid_dense, ap_dense[:, m])

        dist = continuous_stationary_distribution(
            ap_hist, K_hist, shock_transition=P_z, shock_grid=z_grid, backend=backend
        )
        Ks = float(dist.mean())
        excess = Ks - Kd
        return excess, w, Ks, c, dist.pdf

    r_star = float(brentq(lambda r: _eval_excess(r)[0], lo, hi, xtol=1e-6, maxiter=80))
    _, w_star, Ks_star, c_star, pdf_star = _eval_excess(r_star)
    return r_star, w_star, Ks_star, c_star, pdf_star


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
        Pre-solved initial general equilibrium, or configuration dictionary
        to pass to ``solve_aiyagari_continuous``.
    terminal_steady_state : AiyagariContinuousEquilibrium or dict, optional
        Pre-solved terminal general equilibrium. If None and shock is transitory,
        defaults to ``initial_steady_state``. If None and shock is permanent,
        is automatically solved at the terminal shock value.
    shock_path : np.ndarray or Sequence[float], optional
        Exogenous shock trajectory over the horizon T. If None, represents a zero shock
        (steady-state invariance check).
    shock_var : {'z', 'tfp', 'r', 'rate', 'beta', 'discount'}, default 'z'
        Variable targeted by the MIT shock:
        - 'z' or 'tfp': Aggregate total factor productivity Z_t.
        - 'r' or 'rate': Exogenous interest rate wedge / monetary shock.
        - 'beta' or 'discount': Discount factor shock beta_t.
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
        Additional parameter overrides.

    Returns
    -------
    ContinuousTransitionResult
        Frozen dataclass holding equilibrium price paths, capital stocks,
        consumption, full sequence of wealth distributions, and presentation methods.
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

    # 1. Resolve initial steady state
    if isinstance(initial_steady_state, dict):
        init_ss = solve_aiyagari_continuous(backend=backend, **initial_steady_state)
    elif isinstance(initial_steady_state, AiyagariContinuousEquilibrium):
        init_ss = initial_steady_state
    else:
        raise TypeError(
            f"initial_steady_state must be an AiyagariContinuousEquilibrium or dict, "
            f"got {type(initial_steady_state).__name__}"
        )

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

    beta = float(kwargs.get("beta", 0.96))
    gamma = float(kwargs.get("gamma", 2.0))
    alpha = float(kwargs.get("alpha", 0.36))
    delta = float(kwargs.get("delta", 0.08))

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
    is_permanent_tfp = is_tfp_shock and not np.isclose(Z_term, 1.0, atol=1e-6)

    if terminal_steady_state is not None:
        if isinstance(terminal_steady_state, dict):
            term_ss = solve_aiyagari_continuous(backend=backend, **terminal_steady_state)
        elif isinstance(terminal_steady_state, AiyagariContinuousEquilibrium):
            term_ss = terminal_steady_state
        else:
            raise TypeError(
                f"terminal_steady_state must be AiyagariContinuousEquilibrium or dict, "
                f"got {type(terminal_steady_state).__name__}"
            )
        r_term = float(term_ss.r)
        w_term = float(term_ss.w)
        c_term = np.asarray(term_ss.household_solution.policy_c, dtype=np.float64)
    elif is_permanent_tfp:
        r_term, w_term, _, c_term, _ = _solve_terminal_steady_state(
            init_ss=init_ss,
            Z_term=Z_term,
            beta=beta,
            gamma=gamma,
            alpha=alpha,
            delta=delta,
            L_agg=L_agg,
            K_hist=K_hist,
            a_grid_dense=a_grid_dense,
            P_z=P_z,
            z_grid=z_grid,
            backend=backend,
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
            beta=beta,
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
            stacklevel=2,
        )

    meta = {
        "solver": s_solver,
        "horizon": horizon,
        "shock_var": s_var,
        "asset_grid": K_hist,
        "Z_path": Z_path,
        "backend": backend,
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
        converged=converged,
        elapsed_time=elapsed,
        metadata=meta,
    )


def continuous_mit_shock(
    steady_state: AiyagariContinuousEquilibrium | dict[str, Any],
    shock_type: str = "tfp",
    shock_size: float = 0.05,
    persistence: float = 0.8,
    horizon: int = 150,
    solver: str = "broyden",
    **kwargs: Any,
) -> ContinuousTransitionResult:
    """Convenience entry point for simulating unexpected aggregate MIT shocks.

    Parameters
    ----------
    steady_state : AiyagariContinuousEquilibrium or dict
        Initial general equilibrium state.
    shock_type : {'tfp', 'rate', 'beta'}, default 'tfp'
        Type of aggregate shock:
        - 'tfp': Total factor productivity shock.
        - 'rate': Monetary/interest rate shock.
        - 'beta': Discount factor patience shock.
    shock_size : float, default 0.05
        Initial innovation size (e.g. +0.05 for +5% TFP, or +0.01 for +100 bps rate).
    persistence : float, default 0.8
        Persistence parameter rho in [0.0, 1.0].
        If persistence == 1.0, the shock is permanent.
        If persistence < 1.0, the shock is transitory with decay rho^t.
    horizon : int, default 150
        Length T of simulation horizon.
    solver : {'broyden', 'shooting'}, default 'broyden'
        Equilibrium relaxation solver.
    **kwargs : Any
        Additional options passed to ``solve_continuous_transition``.

    Returns
    -------
    ContinuousTransitionResult
        Solved non-linear transition dynamics.
    """
    persistence = float(persistence)
    if not (0.0 <= persistence <= 1.0):
        raise ValueError(f"persistence must be in [0.0, 1.0]; got {persistence}")

    horizon = int(horizon)
    if persistence == 1.0:
        shock_path = np.full(horizon, float(shock_size), dtype=np.float64)
    else:
        shock_path = float(shock_size) * (persistence ** np.arange(horizon, dtype=np.float64))

    return solve_continuous_transition(
        initial_steady_state=steady_state,
        shock_path=shock_path,
        shock_var=shock_type,
        horizon=horizon,
        solver=solver,
        **kwargs,
    )


__all__ = [
    "TransitionShock",
    "ContinuousTransitionResult",
    "solve_continuous_transition",
    "continuous_mit_shock",
]
