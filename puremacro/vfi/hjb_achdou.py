"""Achdou, Han, Lasry, Lions & Moll (2022) Continuous-Time HJB Finite-Difference & Adjoint KFE Solvers."""
from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst


def _result_fields_equal(left: Any, right: Any) -> bool:
    """Array-aware field-by-field equality for the frozen result containers."""
    for name in type(left).__dataclass_fields__:
        a = getattr(left, name)
        b = getattr(right, name)
        if a is b:
            continue
        if sp.issparse(a) or sp.issparse(b):
            if not (sp.issparse(a) and sp.issparse(b)) or a.shape != b.shape or (a != b).nnz != 0:
                return False
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not np.array_equal(np.asarray(a), np.asarray(b)):
                return False
        elif not bool(a == b):
            return False
    return True


@dataclass(frozen=True, eq=False)
class HJBSolution(Mapping):
    """Continuous-time HJB Upwind Finite-Difference solution container.

    Mapping protocol
    ----------------
    ``keys()``, iteration, ``len()``, ``dict(sol)`` and ``**sol`` expose the twelve
    legacy keys of the v3.3.0 dict return (``V``, ``c_policy``, ``s_drift``,
    ``a_grid``, ``e_grid``, ``n_iter``, ``elapsed``, ``r_rate``, ``w_rate``,
    ``rho_val``, ``gamma_r``, ``converged``) so legacy consumers keep working.
    Subscripting, ``in`` and ``get`` accept every dataclass field, i.e. the twelve
    legacy keys plus ``g_dist``, ``A_generator`` and ``mass_residual``; methods and
    private attributes are never reachable through the mapping interface.
    ``==`` compares field by field (array-aware), instances are unhashable like a
    dict, and the object is frozen: attribute and item assignment both raise.
    """

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

    # Dictionary mapping protocol for backward compatibility with the v3.3.0 dict
    def __getitem__(self, key: str) -> Any:
        if key not in self:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in type(self).__dataclass_fields__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HJBSolution):
            return NotImplemented
        return _result_fields_equal(self, other)

    __hash__ = None  # type: ignore[assignment]

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
        return getattr(self, key) if key in self else default

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


@dataclass(frozen=True, eq=False)
class AiyagariContinuousHJBResult(Mapping):
    """Continuous Aiyagari General Equilibrium solution container.

    Follows the same mapping protocol as :class:`HJBSolution`: ``keys()`` lists all
    thirteen fields, subscripting/``in``/``get`` are restricted to those fields,
    ``==`` is array-aware field equality and instances are unhashable and frozen.
    ``converged`` is True iff the market-clearing residual at ``r_star`` satisfies
    ``|K^s - K^d| < tol_ge`` (``excess_capital`` is that residual); it does not
    depend on whether Brent's method exhausted ``max_iter_ge``, which ``n_iter_ge``
    reports.
    """

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
        if key not in self:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in type(self).__dataclass_fields__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AiyagariContinuousHJBResult):
            return NotImplemented
        return _result_fields_equal(self, other)

    __hash__ = None  # type: ignore[assignment]

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
        return getattr(self, key) if key in self else default

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


def _quadrature_weights(a_grid: np.ndarray) -> np.ndarray:
    """Cell-width quadrature weights ``w`` on ``a_grid`` (uniform or not).

    ``w_i = (Delta a_{i-1} + Delta a_i) / 2`` with the first and last spacing
    repeated at the two ends, so that ``w == Delta a`` on a uniform grid and
    ``sum_i g_i w_i`` integrates a density ``g`` on any grid. The same weights are
    used by ``solve_kfe_achdou`` (normalisation) and by
    ``solve_aiyagari_continuous_hjb`` (capital supply integral).
    """
    a_grid = np.asarray(a_grid, dtype=float)
    Na = len(a_grid)
    if Na < 2:
        return np.ones(Na)
    da = np.diff(a_grid)
    return 0.5 * (np.insert(da, 0, da[0]) + np.append(da, da[-1]))


def _default_generator(Ne: int) -> np.ndarray:
    """Default Poisson income generator: no jumps (Ne=1), symmetric lambda=0.1 (Ne=2), exit rate 0.2 otherwise."""
    if Ne == 1:
        return np.zeros((1, 1))
    if Ne == 2:
        return np.array([[-0.1, 0.1], [0.1, -0.1]])
    rate = 0.2 / (Ne - 1)
    A_z = np.full((Ne, Ne), rate)
    np.fill_diagonal(A_z, -0.2)
    return A_z


def _validate_generator(A_z: Any, Ne: int, *, tol: float = 1e-10) -> np.ndarray:
    """Coerce and validate a continuous-time income generator of shape (Ne, Ne)."""
    A_z = np.asarray(A_z, dtype=float)
    if A_z.shape != (Ne, Ne):
        raise ValueError(
            f"A_z must have shape ({Ne}, {Ne}) to match e_grid of length {Ne}; got {A_z.shape}"
        )
    if not np.all(np.isfinite(A_z)):
        raise ValueError("A_z must contain only finite entries")
    off_diag = A_z - np.diag(np.diag(A_z))
    if np.any(off_diag < -tol):
        raise ValueError("A_z off-diagonal entries (jump intensities) must be non-negative")
    row_sums = A_z.sum(axis=1)
    if np.max(np.abs(row_sums)) > tol * max(1.0, float(np.max(np.abs(A_z)))):
        hint = ""
        if np.all(A_z >= -tol) and np.allclose(row_sums, 1.0, atol=1e-8):
            hint = (
                " A_z looks like a transition-probability matrix P; the solver expects a "
                "continuous-time generator with zero row sums, e.g. (P - I) / dt."
            )
        raise ValueError(
            f"A_z rows must sum to zero (continuous-time generator); got row sums {row_sums}.{hint}"
        )
    return A_z


def _n_closed_classes(A: sp.spmatrix) -> int:
    """Number of closed communicating classes of the CTMC generator ``A`` (structural).

    A finite chain has a unique stationary distribution iff exactly one class of its
    transition graph is closed (no transition leaves it), so this is the exact test for
    the KFE null space being one-dimensional; a numerically singular solve cannot
    detect a reducible generator whose blocks each carry a valid null vector.
    """
    coo = sp.coo_matrix(A)
    mask = (coo.row != coo.col) & (coo.data != 0.0)
    off = sp.csr_matrix(
        (np.ones(int(mask.sum())), (coo.row[mask], coo.col[mask])), shape=A.shape
    )
    n_comp, labels = connected_components(off, directed=True, connection="strong")
    src, dst = off.nonzero()
    open_classes = np.unique(labels[src[labels[src] != labels[dst]]])
    return int(n_comp - open_classes.size)


def solve_kfe_achdou(
    A: sp.spmatrix,
    a_grid: np.ndarray,
    e_grid: np.ndarray,
    return_residual: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Solve the continuous-time Kolmogorov Forward Equation (KFE) A^T g = 0.

    Computes the stationary asset-productivity density g(a, e). The generator
    ``A`` acts on grid nodes, so the null vector of ``A^T`` is the stationary node
    mass ``pi``; the density is ``g_i = pi_i / w_i`` with the cell-width quadrature
    weights ``w`` of the grid (``w == Delta a`` on a uniform grid,
    ``(Delta a_{i-1} + Delta a_i) / 2`` otherwise). ``g`` is normalised so that
    |\\sum_{i,j} g_{i,j} w_i - 1.0| \\le 10^{-12} and g \\ge 0 on any grid, and the
    weighted marginal \\sum_i g_{i,j} w_i equals the stationary distribution of the
    income generator.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Infinitesimal generator matrix of shape (Na*Ne, Na*Ne) from HJB solver.
    a_grid : np.ndarray
        Asset grid of shape (Na,), strictly increasing, uniform or not.
    e_grid : np.ndarray
        Income productivity grid of shape (Ne,).
    return_residual : bool, default False
        If True, returns (g, mass_residual). Otherwise returns g.

    Returns
    -------
    g : np.ndarray of shape (Na, Ne)
        Stationary density (mass per unit of ``a``).
    mass_residual : float, optional
        Deviation from unit mass: |\\sum g_{i,j} w_i - 1.0|.

    Raises
    ------
    ValueError
        If ``A`` has the wrong shape, ``a_grid`` is not strictly increasing, or ``A``
        admits no unique stationary distribution: its transition graph must have
        exactly one closed communicating class (a reducible income generator with no
        switching between blocks, or a policy with zero drift at isolated grid nodes,
        gives several), and the normalised system must be numerically non-singular.
    """
    a_grid = np.asarray(a_grid, dtype=float)
    Na = len(a_grid)
    Ne = len(e_grid)
    N = Na * Ne
    if tuple(A.shape) != (N, N):
        raise ValueError(f"A must have shape ({N}, {N}) for Na={Na}, Ne={Ne}; got {A.shape}")
    if Na > 1 and not np.all(np.diff(a_grid) > 0):
        raise ValueError("a_grid must be strictly increasing")

    n_closed = _n_closed_classes(A)
    if n_closed != 1:
        raise ValueError(
            f"KFE has no unique stationary distribution: the generator has {n_closed} closed "
            "communicating classes (exactly one is required). Typical causes are a reducible "
            "income generator A_z with no switching between blocks, or a policy with zero "
            "drift at isolated grid nodes. Pass an irreducible A_z or use compute_kfe=False."
        )

    w_a = _quadrature_weights(a_grid)
    quad_weights = np.tile(w_a, Ne)

    # Stationary node mass: A^T pi = 0 with row 0 replaced by the normalisation
    # sum(pi) = 1 (the rows of A^T sum to zero, so dropping one loses nothing).
    M = A.T.tolil()
    M[0, :] = 1.0
    M = M.tocsc()

    b = np.zeros(N)
    b[0] = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", spla.MatrixRankWarning)
        pi = spla.spsolve(M, b)
    if not np.all(np.isfinite(pi)):
        raise ValueError(
            "KFE has no unique stationary distribution: the generator is numerically "
            "singular after normalisation. Check the drift policy (e.g. nodes with "
            "vanishing outflow) or use compute_kfe=False."
        )
    # Strict non-negativity and exact normalisation of the node mass
    pi = np.maximum(pi, 0.0)
    total_mass = float(np.sum(pi))
    if total_mass > 0:
        pi = pi / total_mass

    # Density = mass per unit of a, so that sum(g * w) == 1 on any grid
    g_vec = pi / quad_weights
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

    .. note::
       Since 3.4.0 the income process switches between the states of ``e_grid``
       through the Poisson generator ``A_z`` (default two-state symmetric
       ``lambda = 0.1``). The v3.3.0 explicit solver had no switching term, so its
       productivity states were independent deterministic-income problems; value
       and consumption for identical arguments therefore differ materially from
       3.3.0. Pass ``A_z=np.zeros((Ne, Ne))`` and ``compute_kfe=False`` to reproduce
       the 3.3.0 economics (a generator without switching has no unique stationary
       distribution, so the KFE cannot be computed for it).

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
        Maximum policy iterations (must be >= 1).
    tol : float, default 1e-8
        Convergence tolerance on sup-norm of value function updates.
    a_min : float, default 0.0
        Lower asset boundary (borrowing limit).
    a_max : float, default 30.0
        Upper asset boundary.
    a_grid : np.ndarray, optional
        Custom asset grid of shape (Na,), strictly increasing; may be non-uniform.
    e_grid : np.ndarray, optional
        Productivity states of shape (Ne,). Default [0.2, 1.0].
    A_z : np.ndarray, optional
        Income jump generator matrix of shape (Ne, Ne): non-negative off-diagonal
        intensities and zero row sums. Default 2-state Poisson jumps with
        lambda_1 = lambda_2 = 0.1 (Ne = 1: no jumps; Ne > 2: uniform exit rate 0.2).
    Delta : float, default 1e4
        Implicit time step / policy iteration acceleration parameter.
    compute_kfe : bool, default True
        Whether to compute the stationary wealth distribution via adjoint KFE.
    v_prime_boundary : tuple[float | None, float | None], optional
        Prescribed boundary marginal utilities (derivatives) (v_prime_min, v_prime_max).
        If specified, provides Neumann boundary conditions at a_min and/or a_max.
        When w_rate == 0.0 (unconstrained cake-eating / asset-only benchmark), defaults
        to exact analytical CRRA marginal utilities; that mode requires a_grid[0] > 0
        because the closed form (mu a)^(-gamma) is singular at a = 0. The benchmark
        has no income risk, so its stationary distribution is degenerate: with
        r < rho all mass sits at a_min, with r > rho at a_max, and at r == rho every
        node is absorbing. The KFE is therefore only computable when the discretised
        drift leaves exactly one absorbing node (typically a_min >= 0.5 with r < rho);
        otherwise ``compute_kfe=True`` raises ValueError and ``compute_kfe=False``
        returns the HJB solution alone, which is what the benchmark is for.

    Returns
    -------
    HJBSolution
        Frozen dataclass with value function, consumption policy, savings drift,
        stationary distribution, and infinitesimal generator.

    Raises
    ------
    ValueError
        If ``max_iter < 1``, ``a_grid`` is not strictly increasing with at least two
        points, ``A_z`` is not a valid (Ne, Ne) generator, ``w_rate == 0`` with
        ``a_grid[0] <= 0``, or (with ``compute_kfe=True``) the generator admits no
        unique stationary distribution (a reducible ``A_z``, or the degenerate
        ``w_rate == 0`` benchmark; pass ``compute_kfe=False`` for the HJB alone).
    """
    t_start = time.time()

    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError(f"solve_hjb_achdou: max_iter must be an integer >= 1, got {max_iter}")

    if a_grid is None:
        a_grid = np.linspace(a_min, a_max, Na)
    else:
        a_grid = np.asarray(a_grid, dtype=float)
    if a_grid.ndim != 1 or a_grid.size < 2 or not np.all(np.diff(a_grid) > 0):
        raise ValueError(
            "a_grid must be a finite, strictly increasing 1-D array with at least two points"
        )
    Na = len(a_grid)

    if e_grid is None:
        e_grid = np.array([0.2, 1.0])
    else:
        e_grid = np.asarray(e_grid, dtype=float)
    Ne = len(e_grid)

    A_z = _default_generator(Ne) if A_z is None else _validate_generator(A_z, Ne)

    if w_rate == 0.0 and a_grid[0] <= 0.0:
        raise ValueError(
            "w_rate == 0.0 selects the analytic CRRA cake-eating mode, whose closed-form "
            "guess c(a) = mu * a and marginal utility (mu * a)**(-gamma) are singular at "
            f"a <= 0; got a_grid[0] = {a_grid[0]}. Pass a_min > 0 (e.g. a_min=1.0)."
        )

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
        try:
            g_dist, mass_residual = solve_kfe_achdou(A, a_grid, e_grid, return_residual=True)
        except ValueError as exc:
            if w_rate != 0.0:
                raise
            raise ValueError(
                f"{exc} The w_rate == 0 cake-eating benchmark has no income risk, so its "
                "stationary distribution is degenerate (all mass at a_min for r < rho, at "
                "a_max for r > rho, arbitrary at r == rho) and the discretised drift "
                "vanishes at several nodes; pass compute_kfe=False to obtain the HJB "
                "solution alone."
            ) from exc

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

    Finds the market-clearing interest rate r* in ``[r_min, r_max]`` (``r_max`` must
    lie below the discount rate ``rho_val``, above which households have no
    stationary wealth distribution) balancing aggregate capital supply
    K^s(r) = \\int a g(a, z) da and firm capital demand K^d(r) from Cobb-Douglas FOCs.
    Brent's method runs with an x-tolerance derived from ``tol_ge`` and the
    bracket's secant slope, and the bracket is tightened from the cached
    evaluations (up to three further passes) until |K^s(r*) - K^d(r*)| < tol_ge.
    ``converged`` is True iff that residual criterion holds at the returned
    ``r_star``, whether or not a Brent pass ran out of ``max_iter_ge`` iterations
    (``n_iter_ge`` counts the iterations actually used).

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
        Lower bound on interest rate search bracket (must exceed ``-delta``).
    r_max : float, optional
        Upper bound on interest rate search bracket (defaults to rho_val - 0.002;
        must satisfy ``r_min < r_max < rho_val``).
    tol_ge : float, default 1e-4
        Equilibrium market clearing tolerance |K^s - K^d| < tol_ge. Drives the
        root-finding tolerance and the ``converged`` flag of the result.
    max_iter_ge : int, default 40
        Maximum Brent iterations per bracket pass.
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

    Raises
    ------
    ValueError
        If the bracket is invalid (``r_min >= r_max``, ``r_max >= rho_val`` or
        ``r_min <= -delta``) or ``A_z`` is not a valid (Ne, Ne) generator.
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

    A_z = _default_generator(Ne) if A_z is None else _validate_generator(A_z, Ne)

    if r_max is None:
        r_max = rho_val - 0.002
    if r_max >= rho_val:
        raise ValueError(
            f"r_max ({r_max}) must be below rho_val ({rho_val}): households have no "
            "stationary wealth distribution at r >= rho."
        )
    if r_min >= r_max:
        raise ValueError(f"r_min ({r_min}) must be < r_max ({r_max}).")
    if r_min <= -delta:
        raise ValueError(f"r_min ({r_min}) must exceed -delta ({-delta}) for the firm FOC to be defined.")

    # Stationary distribution of productivity and aggregate labor supply L
    p_z = _stationary_markov_distribution(A_z)
    L_star = float(np.sum(e_grid * p_z))

    # Grid quadrature integration weights (identical to the KFE normalisation weights)
    w_a = _quadrature_weights(a_grid)

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

    n_iter_brent = 1

    if f_low * f_high > 0:
        # Fallback: pick the best candidate among cached evaluations
        r_star = min(eval_cache.keys(), key=lambda r: abs(eval_cache[r][0]))
    else:
        # x-tolerance implied by tol_ge through the bracket's secant slope (f_low < 0 < f_high)
        slope = (f_high - f_low) / (r_high_adj - r_low_adj)
        xtol = min(1e-5, max(0.5 * tol_ge / max(slope, 1e-8), 1e-12))
        rtol = max(4.0 * np.finfo(float).eps, min(1e-5, xtol))
        res = opt.root_scalar(
            _eval_r,
            bracket=[r_low_adj, r_high_adj],
            method="brentq",
            xtol=xtol,
            rtol=rtol,
            maxiter=max_iter_ge,
        )
        r_star = float(res.root)
        n_iter_brent = int(res.iterations)

        # Tighten the bracket from the cached evaluations until |K^s - K^d| < tol_ge.
        # Each pass gets max_iter_ge Brent iterations, whether or not the previous
        # pass shrank its x-interval below xtol: the target is the residual, not xtol.
        for _ in range(3):
            if abs(_eval_r(r_star)) < tol_ge:
                break
            r_lo = max((r for r, v in eval_cache.items() if v[0] < 0.0), default=None)
            r_hi = min((r for r, v in eval_cache.items() if v[0] > 0.0), default=None)
            if r_lo is None or r_hi is None or r_hi <= r_lo:
                break
            xtol = max(xtol * 1e-2, 1e-14)
            rtol = max(4.0 * np.finfo(float).eps, min(rtol, xtol))
            res = opt.root_scalar(
                _eval_r,
                bracket=[r_lo, r_hi],
                method="brentq",
                xtol=xtol,
                rtol=rtol,
                maxiter=max_iter_ge,
            )
            r_star = float(res.root)
            n_iter_brent += int(res.iterations)

    _eval_r(r_star)
    excess_cap, Ks_star, Kd_star, sol_star = eval_cache[r_star]
    # The documented criterion: converged iff the market clears to tol_ge at r_star.
    # Brent's own x-interval flag is deliberately not consulted: an iteration-starved
    # pass whose last iterate already clears the market has found the equilibrium,
    # and a pass that shrank the interval without clearing it has not.
    converged_ge = bool(abs(excess_cap) < tol_ge)

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
