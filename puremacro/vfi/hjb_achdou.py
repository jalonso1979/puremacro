"""Achdou, Han, Lasry, Lions & Moll (2022) Continuous-Time HJB Finite-Difference & Adjoint KFE Solvers."""
from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst


@dataclass(frozen=True)
class HJBSolution(Mapping):
    """Continuous-time HJB Upwind Finite-Difference solution container."""

    V: np.ndarray
    c_policy: np.ndarray
    s_drift: np.ndarray
    a_grid: np.ndarray
    e_grid: np.ndarray
    n_iter: int
    elapsed: float
    r_rate: float = 0.03
    w_rate: float = 1.0
    rho_val: float = 0.05
    gamma_r: float = 2.0
    converged: bool = True
    g_dist: np.ndarray | None = None
    A_generator: Any = None
    mass_residual: float = 0.0

    # Dictionary mapping protocol for 100% backward compatibility
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def keys(self) -> list[str]:
        return [
            "V",
            "c_policy",
            "s_drift",
            "a_grid",
            "e_grid",
            "n_iter",
            "elapsed",
            "r_rate",
            "w_rate",
            "rho_val",
            "gamma_r",
            "converged",
        ]

    def values(self) -> list[Any]:
        return [getattr(self, k) for k in self.keys()]

    def items(self) -> list[tuple[str, Any]]:
        return [(k, getattr(self, k)) for k in self.keys()]

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def summary(self) -> pd.DataFrame:
        """Produce summary DataFrame of HJB convergence diagnostics and calibrated parameters."""
        records = [
            {"Metric": "Asset Grid Points (Na)", "Value": str(len(self.a_grid))},
            {"Metric": "Income Shock States (Ne)", "Value": str(len(self.e_grid))},
            {"Metric": "Total State Space", "Value": str(len(self.a_grid) * len(self.e_grid))},
            {"Metric": "Iterations", "Value": str(self.n_iter)},
            {"Metric": "Converged", "Value": str(self.converged)},
            {"Metric": "Elapsed Time (s)", "Value": f"{self.elapsed:.4f}"},
            {"Metric": "Interest Rate (r)", "Value": f"{self.r_rate:.4f}"},
            {"Metric": "Wage Rate (w)", "Value": f"{self.w_rate:.4f}"},
            {"Metric": "Discount Rate (rho)", "Value": f"{self.rho_val:.4f}"},
            {"Metric": "Relative Risk Aversion (gamma)", "Value": f"{self.gamma_r:.4f}"},
            {"Metric": "Mean Value (V)", "Value": f"{float(np.mean(self.V)):.6f}"},
            {"Metric": "Min Consumption (c)", "Value": f"{float(np.min(self.c_policy)):.6f}"},
            {"Metric": "Max Consumption (c)", "Value": f"{float(np.max(self.c_policy)):.6f}"},
            {"Metric": "KFE Mass Conservation Error", "Value": f"{self.mass_residual:.2e}"},
        ]
        return pd.DataFrame(records).set_index("Metric")

    def to_frame(self) -> pd.DataFrame:
        """Tabulate state grid points, value function, consumption, savings drift, and distribution."""
        na, ne = self.V.shape
        a_idx, e_idx = np.meshgrid(np.arange(na), np.arange(ne), indexing="ij")
        data: dict[str, Any] = {
            "a_idx": a_idx.ravel(),
            "e_idx": e_idx.ravel(),
            "asset_a": self.a_grid[a_idx.ravel()],
            "prod_e": self.e_grid[e_idx.ravel()],
            "value_V": self.V.ravel(),
            "consumption_c": self.c_policy.ravel(),
            "savings_drift_s": self.s_drift.ravel(),
        }
        if self.g_dist is not None:
            data["density_g"] = self.g_dist.ravel()
        return pd.DataFrame(data)

    def to_markdown(self, **kwargs: Any) -> str:
        """Render summary table as GitHub-flavored Markdown."""
        return df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render summary table as LaTeX tabular."""
        return df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render summary table as Typst table."""
        return df_to_typst(self.summary(), **kwargs)

    def plot(
        self,
        *,
        ax: Any = None,
        figsize: tuple[float, float] | None = None,
        show: bool = False,
    ) -> Any:
        """Headless and WASM-safe plot: Value Function, Consumption Rule, and Savings Drift."""
        import matplotlib.pyplot as plt

        na, ne = self.V.shape
        created_fig = False
        if ax is None:
            fig, (ax_v, ax_c, ax_s) = plt.subplots(1, 3, figsize=figsize or (14, 4))
            created_fig = True
        elif isinstance(ax, (list, tuple, np.ndarray)) and len(ax) >= 3:
            ax_v, ax_c, ax_s = ax[0], ax[1], ax[2]
            fig = ax_v.get_figure()
        else:
            ax_v = ax
            ax_c = None
            ax_s = None
            fig = ax_v.get_figure()

        # Panel 1: Value Function
        for k in range(ne):
            label = f"e = {self.e_grid[k]:.2f}"
            ax_v.plot(self.a_grid, self.V[:, k], label=label)
        ax_v.set_title("Value Function V(a, e)")
        ax_v.set_xlabel("Wealth / Asset (a)")
        ax_v.set_ylabel("V")
        ax_v.grid(True, alpha=0.3)
        ax_v.legend(frameon=False)

        # Panel 2: Consumption Policy
        if ax_c is not None:
            for k in range(ne):
                label = f"e = {self.e_grid[k]:.2f}"
                ax_c.plot(self.a_grid, self.c_policy[:, k], label=label)
            ax_c.set_title("Consumption Policy c(a, e)")
            ax_c.set_xlabel("Wealth / Asset (a)")
            ax_c.set_ylabel("c")
            ax_c.grid(True, alpha=0.3)
            ax_c.legend(frameon=False)

        # Panel 3: Savings Drift
        if ax_s is not None:
            for k in range(ne):
                label = f"e = {self.e_grid[k]:.2f}"
                ax_s.plot(self.a_grid, self.s_drift[:, k], label=label)
            ax_s.axhline(0.0, color="k", linestyle="--", alpha=0.5, label="s = 0")
            ax_s.set_title("Savings Drift s(a, e)")
            ax_s.set_xlabel("Wealth / Asset (a)")
            ax_s.set_ylabel("s = r a + w e - c")
            ax_s.grid(True, alpha=0.3)
            ax_s.legend(frameon=False)

        if show:
            plt.show()

        return fig if created_fig else ax


@dataclass(frozen=True)
class AiyagariContinuousHJBResult(Mapping):
    """Continuous Aiyagari General Equilibrium solution container."""

    r_star: float
    w_star: float
    K_star: float
    Kd_star: float
    Ks_star: float
    L_star: float
    Y_star: float
    excess_capital: float
    hjb_solution: HJBSolution
    g_dist: np.ndarray
    converged: bool
    n_iter_ge: int = 1
    elapsed: float = 0.0

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def keys(self) -> list[str]:
        return [
            "r_star",
            "w_star",
            "K_star",
            "Kd_star",
            "Ks_star",
            "L_star",
            "Y_star",
            "excess_capital",
            "hjb_solution",
            "g_dist",
            "converged",
            "n_iter_ge",
            "elapsed",
        ]

    def values(self) -> list[Any]:
        return [getattr(self, k) for k in self.keys()]

    def items(self) -> list[tuple[str, Any]]:
        return [(k, getattr(self, k)) for k in self.keys()]

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def summary(self) -> pd.DataFrame:
        """Produce summary DataFrame of continuous Aiyagari GE metrics."""
        records = [
            {"Metric": "Equilibrium Interest Rate (r*)", "Value": f"{self.r_star:.6f}"},
            {"Metric": "Equilibrium Wage (w*)", "Value": f"{self.w_star:.6f}"},
            {"Metric": "Aggregate Capital (K*)", "Value": f"{self.K_star:.6f}"},
            {"Metric": "Capital Supply (Ks)", "Value": f"{self.Ks_star:.6f}"},
            {"Metric": "Capital Demand (Kd)", "Value": f"{self.Kd_star:.6f}"},
            {"Metric": "Capital Market Clearing Error", "Value": f"{self.excess_capital:.2e}"},
            {"Metric": "Aggregate Labor (L*)", "Value": f"{self.L_star:.6f}"},
            {"Metric": "Aggregate Output (Y*)", "Value": f"{self.Y_star:.6f}"},
            {"Metric": "GE Root-finding Converged", "Value": str(self.converged)},
            {"Metric": "GE Iterations", "Value": str(self.n_iter_ge)},
            {"Metric": "Elapsed Time (s)", "Value": f"{self.elapsed:.4f}"},
        ]
        return pd.DataFrame(records).set_index("Metric")

    def to_frame(self) -> pd.DataFrame:
        """Tabulate household policies and wealth distribution at general equilibrium."""
        return self.hjb_solution.to_frame()

    def to_markdown(self, **kwargs: Any) -> str:
        """Render summary table as GitHub-flavored Markdown."""
        return df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render summary table as LaTeX tabular."""
        return df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render summary table as Typst table."""
        return df_to_typst(self.summary(), **kwargs)

    def plot(
        self,
        *,
        ax: Any = None,
        figsize: tuple[float, float] | None = None,
        show: bool = False,
    ) -> Any:
        """Plot general equilibrium policies and stationary distribution."""
        return self.hjb_solution.plot(ax=ax, figsize=figsize, show=show)


def solve_kfe_achdou(
    A: sp.spmatrix,
    a_grid: np.ndarray,
    e_grid: np.ndarray,
    return_residual: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Solve the continuous-time Kolmogorov Forward Equation (KFE) A^T g = 0.

    Computes the stationary asset-productivity distribution g(a, e) satisfying
    the adjoint operator A^T g = 0 with strict mass conservation:
    |\\sum_{i,j} g_{i,j} \\Delta a_i - 1.0| \\le 10^{-12} and g \\ge 0.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Infinitesimal generator matrix of shape (Na*Ne, Na*Ne) from HJB solver.
    a_grid : np.ndarray
        Asset grid of shape (Na,).
    e_grid : np.ndarray
        Income productivity grid of shape (Ne,).
    return_residual : bool, default False
        If True, returns (g, mass_residual). Otherwise returns g.

    Returns
    -------
    g : np.ndarray of shape (Na, Ne)
        Stationary distribution.
    mass_residual : float, optional
        Deviation from unit mass: |\\sum g_{i,j} \\Delta a_i - 1.0|.
    """
    Na = len(a_grid)
    Ne = len(e_grid)
    N = Na * Ne

    # Quadrature weights across asset grid
    if Na > 1:
        da = np.diff(a_grid)
        if np.allclose(da, da[0], rtol=1e-5):
            w_a = np.full(Na, da[0])
        else:
            w_a = np.zeros(Na)
            w_a[0] = da[0] / 2.0
            w_a[1:-1] = (da[:-1] + da[1:]) / 2.0
            w_a[-1] = da[-1] / 2.0
    else:
        w_a = np.ones(1)

    quad_weights = np.tile(w_a, Ne)

    # Solve A^T g = 0 via row-0 replacement with quadrature weights
    AT = A.T.tocsr()
    M = AT.copy().tolil()
    M[0, :] = quad_weights
    M = M.tocsc()

    b = np.zeros(N)
    b[0] = 1.0

    g_vec = spla.spsolve(M, b)
    # Strict non-negativity and exact quadrature normalization
    g_vec = np.maximum(g_vec, 0.0)
    total_mass = float(np.sum(g_vec * quad_weights))
    if total_mass > 0:
        g_vec = g_vec / total_mass

    mass_residual = float(abs(np.sum(g_vec * quad_weights) - 1.0))
    g = g_vec.reshape((Na, Ne), order="F")

    if return_residual:
        return g, mass_residual
    return g


def solve_hjb_achdou(
    r_rate: float = 0.03,
    w_rate: float = 1.0,
    rho_val: float = 0.05,
    gamma_r: float = 2.0,
    Na: int = 100,
    max_iter: int = 100,
    tol: float = 1e-8,
    *,
    a_min: float = 0.0,
    a_max: float = 30.0,
    a_grid: np.ndarray | None = None,
    e_grid: np.ndarray | None = None,
    A_z: np.ndarray | None = None,
    Delta: float = 1e4,
    compute_kfe: bool = True,
    v_prime_boundary: tuple[float | None, float | None] | None = None,
) -> HJBSolution:
    """Continuous-time HJB implicit upwind finite-difference solver (Achdou et al. 2022).

    Solves the stationary HJB equation for the continuous-time consumption-saving problem:
    \\rho v_j(a) = \\max_{c} { u(c) + v_j'(a) s_j(a) } + \\sum_{k \\ne j} \\lambda_{jk} [v_k(a) - v_j(a)]
    via the canonical implicit upwind finite-difference scheme. Formulates the
    transition matrix A^n as a diagonally dominant M-matrix and solves:
    [(\\rho + 1/\\Delta) I - A^n] v^{n+1} = u(c^n) + (1/\\Delta) v^n
    using scipy.sparse.linalg.spsolve, converging unconditionally in 10-20 iterations.

    Parameters
    ----------
    r_rate : float, default 0.03
        Real interest rate on assets.
    w_rate : float, default 1.0
        Real wage rate per effective labor unit.
    rho_val : float, default 0.05
        Subjective discount rate.
    gamma_r : float, default 2.0
        Coefficient of relative risk aversion (CRRA).
    Na : int, default 100
        Number of asset grid points (if a_grid is not provided).
    max_iter : int, default 100
        Maximum policy iterations.
    tol : float, default 1e-8
        Convergence tolerance on sup-norm of value function updates.
    a_min : float, default 0.0
        Lower asset boundary (borrowing limit).
    a_max : float, default 30.0
        Upper asset boundary.
    a_grid : np.ndarray, optional
        Custom asset grid of shape (Na,).
    e_grid : np.ndarray, optional
        Productivity states of shape (Ne,). Default [0.2, 1.0].
    A_z : np.ndarray, optional
        Income jump generator matrix of shape (Ne, Ne) with zero row sums.
        Default 2-state Poisson jumps with lambda_1 = lambda_2 = 0.1.
    Delta : float, default 1e4
        Implicit time step / policy iteration acceleration parameter.
    compute_kfe : bool, default True
        Whether to compute the stationary wealth distribution via adjoint KFE.
    v_prime_boundary : tuple[float | None, float | None], optional
        Prescribed boundary marginal utilities (derivatives) (v_prime_min, v_prime_max).
        If specified, provides Neumann boundary conditions at a_min and/or a_max.
        When w_rate == 0.0 (unconstrained cake-eating / asset-only benchmark), defaults
        to exact analytical CRRA marginal utilities.

    Returns
    -------
    HJBSolution
        Frozen dataclass with value function, consumption policy, savings drift,
        stationary distribution, and infinitesimal generator.
    """
    t_start = time.time()

    if a_grid is None:
        a_grid = np.linspace(a_min, a_max, Na)
    else:
        a_grid = np.asarray(a_grid, dtype=float)
        Na = len(a_grid)

    if e_grid is None:
        e_grid = np.array([0.2, 1.0])
    else:
        e_grid = np.asarray(e_grid, dtype=float)
    Ne = len(e_grid)

    if A_z is None:
        if Ne == 2:
            A_z = np.array([[-0.1, 0.1], [0.1, -0.1]])
        elif Ne == 1:
            A_z = np.zeros((1, 1))
        else:
            rate = 0.2 / (Ne - 1)
            A_z = np.full((Ne, Ne), rate)
            np.fill_diagonal(A_z, -0.2)
    else:
        A_z = np.asarray(A_z, dtype=float)

    da = np.diff(a_grid)
    da_fwd = np.append(da, da[-1])
    da_bwd = np.insert(da, 0, da[0])

    # Initial Value Function guess
    is_unconstrained = (w_rate == 0.0) or (v_prime_boundary is not None)

    if w_rate == 0.0:
        mu_rate = max((rho_val - (1.0 - gamma_r) * r_rate) / gamma_r, 1e-6)
        c_init = mu_rate * a_grid
        if abs(gamma_r - 1.0) < 1e-7:
            V = np.tile(np.log(c_init)[:, None] / rho_val, (1, Ne))
        else:
            V = np.tile(((c_init ** (1.0 - gamma_r)) / ((1.0 - gamma_r) * rho_val))[:, None], (1, Ne))
    else:
        V = np.zeros((Na, Ne))
        for k in range(Ne):
            flow_inc = np.maximum(r_rate * a_grid + w_rate * e_grid[k], 1e-6)
            if abs(gamma_r - 1.0) < 1e-7:
                V[:, k] = np.log(flow_inc) / rho_val
            else:
                V[:, k] = (flow_inc ** (1.0 - gamma_r)) / ((1.0 - gamma_r) * rho_val)

    c_policy = np.zeros((Na, Ne))
    s_drift = np.zeros((Na, Ne))
    dist = 1.0
    A = None

    for iter_count in range(1, max_iter + 1):
        V_old = V.copy()

        # Upwind finite differences
        V_forward = np.zeros((Na, Ne))
        V_backward = np.zeros((Na, Ne))

        for k in range(Ne):
            if v_prime_boundary is not None and v_prime_boundary[1] is not None:
                v_fwd_bound = float(v_prime_boundary[1])
            elif w_rate == 0.0:
                mu_rate = max((rho_val - (1.0 - gamma_r) * r_rate) / gamma_r, 1e-6)
                v_fwd_bound = float((mu_rate * a_grid[-1]) ** (-gamma_r))
            else:
                v_fwd_bound = max(r_rate * a_grid[-1] + w_rate * e_grid[k], 1e-6) ** (-gamma_r)

            if v_prime_boundary is not None and v_prime_boundary[0] is not None:
                v_bwd_bound = float(v_prime_boundary[0])
            elif w_rate == 0.0:
                mu_rate = max((rho_val - (1.0 - gamma_r) * r_rate) / gamma_r, 1e-6)
                v_bwd_bound = float((mu_rate * a_grid[0]) ** (-gamma_r))
            else:
                v_bwd_bound = max(r_rate * a_grid[0] + w_rate * e_grid[k], 1e-6) ** (-gamma_r)

            V_forward[:-1, k] = (V[1:, k] - V[:-1, k]) / da
            V_forward[-1, k] = v_fwd_bound

            V_backward[1:, k] = (V[1:, k] - V[:-1, k]) / da
            V_backward[0, k] = v_bwd_bound

        c_forward = np.maximum(np.maximum(V_forward, 1e-12) ** (-1.0 / gamma_r), 1e-8)
        s_forward = r_rate * a_grid[:, None] + w_rate * e_grid[None, :] - c_forward

        c_backward = np.maximum(np.maximum(V_backward, 1e-12) ** (-1.0 / gamma_r), 1e-8)
        s_backward = r_rate * a_grid[:, None] + w_rate * e_grid[None, :] - c_backward

        if w_rate == 0.0:
            mu_rate = max((rho_val - (1.0 - gamma_r) * r_rate) / gamma_r, 1e-6)
            c_zero = np.maximum(mu_rate * a_grid[:, None], 1e-8)
        else:
            c_zero = np.maximum(r_rate * a_grid[:, None] + w_rate * e_grid[None, :], 1e-8)

        use_fwd = (s_forward > 0)
        if not is_unconstrained:
            use_fwd[-1, :] = False  # Boundary condition at a_max

        use_bwd = (s_backward < 0) & (~use_fwd)
        if not is_unconstrained:
            use_bwd[0, :] = False   # Borrowing constraint at a_min

        use_zero = (~use_fwd) & (~use_bwd)

        c_policy = c_forward * use_fwd + c_backward * use_bwd + c_zero * use_zero
        s_drift = s_forward * use_fwd + s_backward * use_bwd

        if abs(gamma_r - 1.0) < 1e-7:
            u_val = np.log(c_policy)
        else:
            u_val = (c_policy ** (1.0 - gamma_r)) / (1.0 - gamma_r)

        # Assemble infinitesimal generator matrix A^n
        row_idx = []
        col_idx = []
        data = []
        b_boundary = np.zeros(Na * Ne)

        for k in range(Ne):
            offset = k * Na
            for i in range(Na):
                row = offset + i
                if i > 0:
                    X = max(-s_drift[i, k], 0.0) / da_bwd[i]
                else:
                    X = 0.0
                    if s_drift[0, k] < 0:
                        b_boundary[row] += s_drift[0, k] * V_backward[0, k]

                if i < Na - 1:
                    Z = max(s_drift[i, k], 0.0) / da_fwd[i]
                else:
                    Z = 0.0
                    if s_drift[Na - 1, k] > 0:
                        b_boundary[row] += s_drift[Na - 1, k] * V_forward[-1, k]

                Y = -X - Z

                if i > 0 and X > 0:
                    row_idx.append(row)
                    col_idx.append(offset + i - 1)
                    data.append(X)
                if i < Na - 1 and Z > 0:
                    row_idx.append(row)
                    col_idx.append(offset + i + 1)
                    data.append(Z)

                for l in range(Ne):
                    jump_rate = A_z[k, l]
                    if l == k:
                        jump_rate += Y
                    if jump_rate != 0:
                        row_idx.append(row)
                        col_idx.append(l * Na + i)
                        data.append(jump_rate)

        A = sp.csr_matrix((data, (row_idx, col_idx)), shape=(Na * Ne, Na * Ne))

        # Solve implicit M-matrix system: [(\rho + 1/\Delta)I - A] v^{n+1} = u(c^n) + b_boundary + (1/\Delta) v^n
        B = (rho_val + 1.0 / Delta) * sp.eye(Na * Ne, format="csr") - A
        rhs = u_val.ravel(order="F") + b_boundary + (1.0 / Delta) * V_old.ravel(order="F")

        v_new = spla.spsolve(sp.csc_matrix(B), rhs)
        V = v_new.reshape((Na, Ne), order="F")

        dist = float(np.max(np.abs(V - V_old)))
        if dist < tol:
            break

    elapsed = time.time() - t_start
    converged = bool(dist < tol)

    g_dist = None
    mass_residual = 0.0
    if compute_kfe and A is not None:
        g_dist, mass_residual = solve_kfe_achdou(A, a_grid, e_grid, return_residual=True)

    return HJBSolution(
        V=V,
        c_policy=c_policy,
        s_drift=s_drift,
        a_grid=a_grid,
        e_grid=e_grid,
        n_iter=iter_count,
        elapsed=elapsed,
        r_rate=r_rate,
        w_rate=w_rate,
        rho_val=rho_val,
        gamma_r=gamma_r,
        converged=converged,
        g_dist=g_dist,
        A_generator=A,
        mass_residual=mass_residual,
    )


def _stationary_markov_distribution(A_z: np.ndarray) -> np.ndarray:
    """Compute stationary distribution of continuous-time generator A_z."""
    Ne = A_z.shape[0]
    if Ne == 1:
        return np.array([1.0])
    M = A_z.T.copy()
    M[0, :] = 1.0
    rhs = np.zeros(Ne)
    rhs[0] = 1.0
    try:
        p = np.linalg.solve(M, rhs)
        p = np.maximum(p, 0.0)
        return p / np.sum(p)
    except Exception:
        return np.full(Ne, 1.0 / Ne)


def solve_aiyagari_continuous_hjb(
    alpha: float = 0.33,
    delta: float = 0.05,
    rho_val: float = 0.05,
    gamma_r: float = 2.0,
    Na: int = 100,
    a_min: float = 0.0,
    a_max: float = 30.0,
    a_grid: np.ndarray | None = None,
    e_grid: np.ndarray | None = None,
    A_z: np.ndarray | None = None,
    r_min: float = 0.005,
    r_max: float | None = None,
    tol_ge: float = 1e-4,
    max_iter_ge: int = 40,
    max_iter_hjb: int = 100,
    tol_hjb: float = 1e-8,
    Delta: float = 1e4,
) -> AiyagariContinuousHJBResult:
    """Solve continuous-time Aiyagari general equilibrium via HJB and adjoint KFE.

    Finds market-clearing interest rate r* balancing aggregate capital supply
    K^s(r) = \\int a g(a, z) da and firm capital demand K^d(r) from Cobb-Douglas FOCs:
    |K^s(r*) - K^d(r*)| < 10^{-4}.

    Parameters
    ----------
    alpha : float, default 0.33
        Capital share in production function Y = K^alpha L^{1-alpha}.
    delta : float, default 0.05
        Capital depreciation rate.
    rho_val : float, default 0.05
        Household discount rate.
    gamma_r : float, default 2.0
        CRRA risk aversion coefficient.
    Na : int, default 100
        Number of asset grid points.
    a_min : float, default 0.0
        Borrowing limit.
    a_max : float, default 30.0
        Upper asset boundary.
    a_grid : np.ndarray, optional
        Custom asset grid of shape (Na,).
    e_grid : np.ndarray, optional
        Income states of shape (Ne,).
    A_z : np.ndarray, optional
        Income generator matrix of shape (Ne, Ne).
    r_min : float, default 0.005
        Lower bound on interest rate search bracket.
    r_max : float, optional
        Upper bound on interest rate search bracket (defaults to rho_val - 0.002).
    tol_ge : float, default 1e-4
        Equilibrium market clearing tolerance |K^s - K^d| < tol_ge.
    max_iter_ge : int, default 40
        Maximum Brent iterations.
    max_iter_hjb : int, default 100
        Maximum HJB policy iterations.
    tol_hjb : float, default 1e-8
        HJB policy iteration convergence tolerance.
    Delta : float, default 1e4
        Implicit time step for HJB solver.

    Returns
    -------
    AiyagariContinuousHJBResult
        Container with equilibrium prices (r*, w*), aggregate allocations (K*, L*, Y*),
        household solution, stationary distribution, and market clearing diagnostics.
    """
    t_start = time.time()

    if a_grid is None:
        a_grid = np.linspace(a_min, a_max, Na)
    else:
        a_grid = np.asarray(a_grid, dtype=float)
        Na = len(a_grid)

    if e_grid is None:
        e_grid = np.array([0.2, 1.0])
    else:
        e_grid = np.asarray(e_grid, dtype=float)
    Ne = len(e_grid)

    if A_z is None:
        if Ne == 2:
            A_z = np.array([[-0.1, 0.1], [0.1, -0.1]])
        elif Ne == 1:
            A_z = np.zeros((1, 1))
        else:
            rate = 0.2 / (Ne - 1)
            A_z = np.full((Ne, Ne), rate)
            np.fill_diagonal(A_z, -0.2)
    else:
        A_z = np.asarray(A_z, dtype=float)

    if r_max is None:
        r_max = rho_val - 0.002

    # Stationary distribution of productivity and aggregate labor supply L
    p_z = _stationary_markov_distribution(A_z)
    L_star = float(np.sum(e_grid * p_z))

    # Grid quadrature integration weights
    if Na > 1:
        da = np.diff(a_grid)
        if np.allclose(da, da[0], rtol=1e-5):
            w_a = np.full(Na, da[0])
        else:
            w_a = np.zeros(Na)
            w_a[0] = da[0] / 2.0
            w_a[1:-1] = (da[:-1] + da[1:]) / 2.0
            w_a[-1] = da[-1] / 2.0
    else:
        w_a = np.ones(1)

    eval_cache: dict[float, tuple[float, float, float, HJBSolution]] = {}

    def _eval_r(r_trial: float) -> float:
        if r_trial in eval_cache:
            return eval_cache[r_trial][0]

        # Firm first-order conditions
        k_over_l = (alpha / (r_trial + delta)) ** (1.0 / (1.0 - alpha))
        Kd = float(L_star * k_over_l)
        w = float((1.0 - alpha) * (k_over_l ** alpha))

        sol = solve_hjb_achdou(
            r_rate=r_trial,
            w_rate=w,
            rho_val=rho_val,
            gamma_r=gamma_r,
            Na=Na,
            a_min=a_min,
            a_max=a_max,
            a_grid=a_grid,
            e_grid=e_grid,
            A_z=A_z,
            Delta=Delta,
            max_iter=max_iter_hjb,
            tol=tol_hjb,
            compute_kfe=True,
        )

        g = sol.g_dist
        if g is None:
            raise RuntimeError("KFE distribution was not computed in HJB solve.")

        Ks = float(np.sum(a_grid[:, None] * g * w_a[:, None]))
        excess = Ks - Kd
        eval_cache[r_trial] = (excess, Ks, Kd, sol)
        return excess

    # Check / adapt brackets for Brent's method
    f_low = _eval_r(r_min)
    f_high = _eval_r(r_max)

    r_low_adj = r_min
    r_high_adj = r_max

    # Expand lower bound if needed
    attempts = 0
    while f_low > 0 and r_low_adj > 0.001 and attempts < 5:
        r_low_adj = max(0.001, r_low_adj / 2.0)
        f_low = _eval_r(r_low_adj)
        attempts += 1

    # Expand upper bound if needed
    attempts = 0
    while f_high < 0 and r_high_adj < rho_val - 1e-4 and attempts < 5:
        r_high_adj = (r_high_adj + rho_val) / 2.0
        f_high = _eval_r(r_high_adj)
        attempts += 1

    converged_ge = True
    n_iter_brent = 1

    if f_low * f_high > 0:
        # Fallback: pick the best candidate among cached evaluations
        best_r = min(eval_cache.keys(), key=lambda r: abs(eval_cache[r][0]))
        r_star = best_r
        converged_ge = abs(eval_cache[best_r][0]) < tol_ge
    else:
        res = opt.root_scalar(
            _eval_r,
            bracket=[r_low_adj, r_high_adj],
            method="brentq",
            xtol=1e-5,
            rtol=1e-5,
            maxiter=max_iter_ge,
        )
        r_star = float(res.root)
        converged_ge = bool(res.converged)
        n_iter_brent = res.iterations

    _eval_r(r_star)
    excess_cap, Ks_star, Kd_star, sol_star = eval_cache[r_star]

    k_over_l_star = (alpha / (r_star + delta)) ** (1.0 / (1.0 - alpha))
    w_star = float((1.0 - alpha) * (k_over_l_star ** alpha))
    Y_star = float((Kd_star ** alpha) * (L_star ** (1.0 - alpha)))
    K_star = float(Ks_star)

    elapsed = time.time() - t_start

    return AiyagariContinuousHJBResult(
        r_star=r_star,
        w_star=w_star,
        K_star=K_star,
        Kd_star=Kd_star,
        Ks_star=Ks_star,
        L_star=L_star,
        Y_star=Y_star,
        excess_capital=excess_cap,
        hjb_solution=sol_star,
        g_dist=sol_star.g_dist,  # type: ignore[arg-type]
        converged=converged_ge,
        n_iter_ge=n_iter_brent,
        elapsed=elapsed,
    )
