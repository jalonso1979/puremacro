r"""Polynomial Approximation and Collocation Solver for puremacro.vfi.

Provides continuous state-space dynamic programming and projection methods
using orthogonal Chebyshev polynomials of the first kind:
- 3-term recurrence evaluation for basis polynomials and first derivatives.
- Gauss-Chebyshev (roots) and Gauss-Lobatto-Chebyshev (extrema + endpoints) nodes.
- Bijective affine coordinate mapping between [a, b] and [-1, 1].
- Multi-dimensional tensor product polynomial bases via Kronecker products.
- Euler equation residual projection and continuous Bellman value collocation.
- Multi-backend hardware acceleration (NumPy reference, Numba CPU JIT, Apple Silicon MLX GPU, NVIDIA CuPy GPU).
- Full puremacro presentation contract (.summary(), .plot(), .to_frame(), .to_markdown(), .to_latex(), .to_typst()).
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar, root

from puremacro import _backend as _bk
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst


# ---------------------------------------------------------------------------
# Numerical Kernels for Chebyshev Basis & Derivatives
# ---------------------------------------------------------------------------

@_bk.njit_fallback(fastmath=True)
def _chebyshev_basis_1d_kernel(x: np.ndarray, order: int) -> np.ndarray:
    """Evaluate 1D Chebyshev polynomials T_0(x) ... T_order(x) via 3-term recurrence."""
    n = len(x)
    T = np.zeros((n, order + 1), dtype=np.float64)
    for i in range(n):
        T[i, 0] = 1.0
        if order >= 1:
            T[i, 1] = x[i]
        for j in range(1, order):
            T[i, j + 1] = 2.0 * x[i] * T[i, j] - T[i, j - 1]
    return T


@_bk.njit_fallback(fastmath=True)
def _chebyshev_deriv_1d_kernel(x: np.ndarray, order: int) -> np.ndarray:
    """Evaluate first derivatives T'_0(x) ... T'_order(x) via derivative recurrence."""
    n = len(x)
    T = np.zeros((n, order + 1), dtype=np.float64)
    dT = np.zeros((n, order + 1), dtype=np.float64)
    for i in range(n):
        T[i, 0] = 1.0
        dT[i, 0] = 0.0
        if order >= 1:
            T[i, 1] = x[i]
            dT[i, 1] = 1.0
        for j in range(1, order):
            T[i, j + 1] = 2.0 * x[i] * T[i, j] - T[i, j - 1]
            dT[i, j + 1] = 2.0 * T[i, j] + 2.0 * x[i] * dT[i, j] - dT[i, j - 1]
    return dT


def _chebyshev_basis_1d_mlx(x: np.ndarray, order: int) -> np.ndarray:
    """Evaluate 1D Chebyshev polynomials using Apple Silicon MLX GPU."""
    import mlx.core as mx

    x_mx = mx.array(x, dtype=mx.float32)
    cols = [mx.ones_like(x_mx)]
    if order >= 1:
        cols.append(x_mx)
    for j in range(1, order):
        cols.append(2.0 * x_mx * cols[j] - cols[j - 1])
    T_mx = mx.stack(cols, axis=1)
    return np.asarray(T_mx, dtype=np.float64)


def _chebyshev_basis_1d_cupy(x: np.ndarray, order: int) -> np.ndarray:
    """Evaluate 1D Chebyshev polynomials using NVIDIA CuPy GPU."""
    import cupy as cp

    x_cp = cp.asarray(x, dtype=cp.float64)
    cols = [cp.ones_like(x_cp)]
    if order >= 1:
        cols.append(x_cp)
    for j in range(1, order):
        cols.append(2.0 * x_cp * cols[j] - cols[j - 1])
    T_cp = cp.stack(cols, axis=1)
    return cp.asnumpy(T_cp)


def _chebyshev_basis_1d(x: np.ndarray, order: int, backend: str = "numpy") -> np.ndarray:
    """Dispatch 1D Chebyshev polynomial evaluation across backends."""
    b = str(backend).strip().lower()
    x_arr = np.asarray(x, dtype=np.float64)
    # Clip canonical coordinates to [-1, 1] to guard against tiny floating-point overshoot
    x_clipped = np.clip(x_arr, -1.0, 1.0)
    if b == "mlx" and _bk.backend_available("mlx"):
        return _chebyshev_basis_1d_mlx(x_clipped, order)
    if b == "cupy" and _bk.backend_available("cupy"):
        return _chebyshev_basis_1d_cupy(x_clipped, order)
    if b == "numba" and _bk.backend_available("numba"):
        return _chebyshev_basis_1d_kernel(x_clipped, order)
    return _chebyshev_basis_1d_kernel(x_clipped, order)


def _chebyshev_deriv_1d(x: np.ndarray, order: int, backend: str = "numpy") -> np.ndarray:
    """Dispatch 1D Chebyshev derivative evaluation."""
    b = str(backend).strip().lower()
    x_arr = np.asarray(x, dtype=np.float64)
    x_clipped = np.clip(x_arr, -1.0, 1.0)
    return _chebyshev_deriv_1d_kernel(x_clipped, order)


# ---------------------------------------------------------------------------
# CollocationBasis Class
# ---------------------------------------------------------------------------

class CollocationBasis:
    r"""Orthogonal polynomial collocation basis over bounded state spaces.

    Supports Chebyshev polynomials of the first kind on 1D intervals [a, b]
    and multi-dimensional hypercubes \prod_{d=1}^D [a_d, b_d] via tensor products.

    Parameters
    ----------
    domain : tuple[tuple[float, float], ...] | tuple[float, float] | list
        State space bounds. For 1D, (a, b). For multi-D, ((a1, b1), (a2, b2), ...).
    orders : tuple[int, ...] | int
        Polynomial degree order(s). For 1D, integer N >= 1. For multi-D, (N1, N2, ...).
    basis_type : str, default "chebyshev"
        Polynomial basis family ("chebyshev" supported).
    node_type : str, default "lobatto"
        Collocation node distribution ("lobatto" or "gauss").
        - "lobatto": Chebyshev-Gauss-Lobatto extrema nodes, including endpoints -1 and +1.
        - "gauss": Chebyshev-Gauss roots of T_n(x), strictly in interior (-1, 1).
    """

    def __init__(
        self,
        domain: Union[Tuple[Tuple[float, float], ...], Tuple[float, float], list],
        orders: Union[Tuple[int, ...], int],
        basis_type: str = "chebyshev",
        node_type: str = "lobatto",
    ) -> None:
        if basis_type != "chebyshev":
            raise ValueError(f"Unsupported basis_type {basis_type!r}; supported: 'chebyshev'")
        if node_type not in ("lobatto", "gauss"):
            raise ValueError(f"Unsupported node_type {node_type!r}; supported: 'lobatto', 'gauss'")

        # Normalize domain
        if isinstance(domain, (tuple, list)):
            if len(domain) == 2 and not isinstance(domain[0], (tuple, list)):
                norm_domain = ((float(domain[0]), float(domain[1])),)
            else:
                norm_domain = tuple((float(b[0]), float(b[1])) for b in domain)
        else:
            raise ValueError("domain must be a tuple or list of (lower, upper) bounds")

        if len(norm_domain) == 0:
            raise ValueError("domain cannot be empty")

        for d_idx, (a, b) in enumerate(norm_domain):
            if not np.isfinite(a) or not np.isfinite(b):
                raise ValueError(f"domain bounds for dim {d_idx} must be finite numbers; got ({a}, {b})")
            if a >= b:
                raise ValueError(f"domain bounds for dim {d_idx} must satisfy a < b; got ({a}, {b})")

        # Normalize orders
        if isinstance(orders, (int, np.integer)):
            norm_orders = (int(orders),)
        elif isinstance(orders, (tuple, list)):
            norm_orders = tuple(int(o) for o in orders)
        else:
            raise ValueError("orders must be an integer or a tuple of integers")

        if len(norm_orders) != len(norm_domain):
            raise ValueError(
                f"Dimension mismatch: domain has {len(norm_domain)} dimensions, "
                f"but orders has {len(norm_orders)}"
            )

        for o_idx, o in enumerate(norm_orders):
            if o < 1:
                raise ValueError(f"order for dim {o_idx} must be >= 1; got {o}")

        self.domain: Tuple[Tuple[float, float], ...] = norm_domain
        self.orders: Tuple[int, ...] = norm_orders
        self.basis_type: str = basis_type
        self.node_type: str = node_type
        self.n_dims: int = len(self.domain)
        # Total basis functions = product of (order_d + 1)
        self.n_nodes: int = int(np.prod([o + 1 for o in self.orders]))

    def nodes(self, squeeze: bool = False, xp=None) -> np.ndarray:
        """Compute the collocation nodes in physical state space coordinates.

        Parameters
        ----------
        squeeze : bool, default False
            If True and n_dims == 1, returns a 1D array of shape (n_nodes,).
            Otherwise returns shape (n_nodes, n_dims).
        xp : array namespace, optional
            Optional array namespace.

        Returns
        -------
        np.ndarray
            Collocation nodes, shape (n_nodes, n_dims) or (n_nodes,).
        """
        nodes_1d_list = []
        for d in range(self.n_dims):
            a, b = self.domain[d]
            n_d = self.orders[d] + 1
            if self.node_type == "lobatto":
                # Gauss-Lobatto extrema nodes ordered ascending from -1 to 1:
                # x_k = -cos(k * pi / (n_d - 1)) for k = 0, ..., n_d - 1
                if n_d == 1:
                    x_1d = np.array([0.0])
                else:
                    k = np.arange(n_d)
                    x_1d = -np.cos(k * np.pi / (n_d - 1))
            else:
                # Gauss roots of T_{n_d}(x) ordered ascending:
                # x_k = -cos((2k - 1) * pi / (2 * n_d)) for k = 1, ..., n_d
                k = np.arange(1, n_d + 1)
                x_1d = -np.cos((2.0 * k - 1.0) * np.pi / (2.0 * n_d))

            # Affine map from [-1, 1] to [a, b]
            s_1d = 0.5 * (a + b) + 0.5 * (b - a) * x_1d
            nodes_1d_list.append(s_1d)

        if self.n_dims == 1:
            res = nodes_1d_list[0].reshape(-1, 1)
        else:
            grid = np.meshgrid(*nodes_1d_list, indexing="ij")
            res = np.column_stack([g.ravel() for g in grid])

        if squeeze and self.n_dims == 1:
            return res.ravel()
        return res

    def to_canonical(self, s: Union[np.ndarray, float]) -> np.ndarray:
        """Map physical state coordinates in [a, b] to canonical coordinates in [-1, 1]."""
        s_arr = np.asarray(s, dtype=np.float64)
        if s_arr.ndim == 0:
            s_arr = s_arr.reshape(1, 1)
        elif s_arr.ndim == 1:
            if self.n_dims == 1:
                s_arr = s_arr.reshape(-1, 1)
            elif len(s_arr) == self.n_dims:
                s_arr = s_arr.reshape(1, self.n_dims)
            else:
                raise ValueError(f"Expected array with {self.n_dims} state columns; got length {len(s_arr)}")

        x = np.zeros_like(s_arr)
        for d in range(self.n_dims):
            a, b = self.domain[d]
            x[:, d] = (2.0 * s_arr[:, d] - (a + b)) / (b - a)
        return x

    def to_state(self, x: Union[np.ndarray, float]) -> np.ndarray:
        """Map canonical coordinates in [-1, 1] to physical state coordinates in [a, b]."""
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 0:
            x_arr = x_arr.reshape(1, 1)
        elif x_arr.ndim == 1:
            if self.n_dims == 1:
                x_arr = x_arr.reshape(-1, 1)
            elif len(x_arr) == self.n_dims:
                x_arr = x_arr.reshape(1, self.n_dims)
            else:
                raise ValueError(f"Expected array with {self.n_dims} columns; got length {len(x_arr)}")

        s = np.zeros_like(x_arr)
        for d in range(self.n_dims):
            a, b = self.domain[d]
            s[:, d] = 0.5 * (a + b) + 0.5 * (b - a) * x_arr[:, d]
        return s

    def evaluate(self, s: Union[np.ndarray, float], xp=None, backend: str = "numpy") -> np.ndarray:
        """Evaluate the Chebyshev polynomial basis matrix Phi(s).

        Parameters
        ----------
        s : np.ndarray | float
            Evaluation states. Shape (K, D), (K,) if 1D, or scalar float.
        xp : array namespace, optional
            Ignored or passed for namespace compatibility.
        backend : str, default "numpy"
            Backend for evaluation ("numpy", "numba", "mlx", "cupy").

        Returns
        -------
        np.ndarray
            Basis matrix of shape (K, n_nodes).
        """
        x = self.to_canonical(s)
        K = len(x)

        # 1D evaluation for each dimension
        T_dims = []
        for d in range(self.n_dims):
            T_d = _chebyshev_basis_1d(x[:, d], self.orders[d], backend=backend)
            T_dims.append(T_d)

        # Tensor product via Kronecker product of rows
        Phi = T_dims[0]
        for d in range(1, self.n_dims):
            # Outer product of columns per row: shape (K, n_prev, n_curr) -> (K, n_prev * n_curr)
            Phi = (Phi[:, :, None] * T_dims[d][:, None, :]).reshape(K, -1)

        return Phi

    def derivative(
        self, s: Union[np.ndarray, float], dim: int = 0, xp=None, backend: str = "numpy"
    ) -> np.ndarray:
        """Evaluate the partial derivative matrix dPhi/ds_dim(s).

        Parameters
        ----------
        s : np.ndarray | float
            Evaluation states.
        dim : int, default 0
            State dimension along which to differentiate (0 <= dim < n_dims).
        xp : array namespace, optional
        backend : str, default "numpy"

        Returns
        -------
        np.ndarray
            Derivative basis matrix of shape (K, n_nodes).
        """
        if not (0 <= dim < self.n_dims):
            raise ValueError(f"dim must be in [0, {self.n_dims - 1}]; got {dim}")

        x = self.to_canonical(s)
        K = len(x)

        T_dims = []
        for d in range(self.n_dims):
            if d == dim:
                a, b = self.domain[d]
                # Chain rule: dx/ds = 2 / (b - a)
                dT_d = _chebyshev_deriv_1d(x[:, d], self.orders[d], backend=backend) * (2.0 / (b - a))
                T_dims.append(dT_d)
            else:
                T_d = _chebyshev_basis_1d(x[:, d], self.orders[d], backend=backend)
                T_dims.append(T_d)

        Phi_deriv = T_dims[0]
        for d in range(1, self.n_dims):
            Phi_deriv = (Phi_deriv[:, :, None] * T_dims[d][:, None, :]).reshape(K, -1)

        return Phi_deriv

    def fit(self, y: np.ndarray, backend: str = "numpy") -> np.ndarray:
        """Compute basis coefficients theta that interpolate y at collocation nodes.

        Solves Phi_nodes @ theta = y.

        Parameters
        ----------
        y : np.ndarray
            Function values evaluated at the collocation nodes. Shape (n_nodes,) or (n_nodes, P).
        backend : str, default "numpy"

        Returns
        -------
        np.ndarray
            Coefficients array matching the shape of y.
        """
        nodes_arr = self.nodes()
        Phi_nodes = self.evaluate(nodes_arr, backend=backend)
        return np.linalg.solve(Phi_nodes, y)

    def interpolate(
        self, c: np.ndarray, s: Union[np.ndarray, float], backend: str = "numpy"
    ) -> Union[np.ndarray, float]:
        """Evaluate the polynomial approximation at state(s) s.

        Parameters
        ----------
        c : np.ndarray
            Polynomial basis coefficients.
        s : np.ndarray | float
            Evaluation points.
        backend : str, default "numpy"

        Returns
        -------
        np.ndarray | float
            Approximated values. Float if s is scalar; array otherwise.
        """
        Phi_s = self.evaluate(s, backend=backend)
        res = Phi_s @ c
        if np.ndim(s) == 0:
            return float(res[0])
        if self.n_dims == 1 and np.asarray(s).ndim == 1:
            return res.ravel()
        return res


# ---------------------------------------------------------------------------
# CollocationSolution Class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CollocationSolution:
    """Frozen solution object from polynomial collocation.

    Attributes
    ----------
    coefficients : np.ndarray
        Polynomial basis coefficients (policy coefficients for 'euler',
        value coefficients for 'bellman').
    basis : CollocationBasis
        Underlying polynomial basis.
    converged : bool
        Whether the solver reached convergence tolerance.
    n_iter : int
        Number of iterations taken.
    residual_norm : float
        Final residual norm at collocation nodes.
    backend : str
        Compute backend utilized ('numpy', 'numba', 'mlx', 'cupy').
    metadata : dict
        Additional solution metadata, including execution time,
        auxiliary coefficients (policy/value), and node residuals.
    """

    coefficients: np.ndarray
    basis: CollocationBasis
    converged: bool
    n_iter: int
    residual_norm: float
    backend: str
    metadata: dict = field(default_factory=dict)

    def policy(self, s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate the continuous policy function g(s)."""
        method = self.metadata.get("method", "euler")
        if method == "euler":
            return self.basis.interpolate(self.coefficients, s, backend=self.backend)
        # Bellman method stores policy coefficients in metadata
        policy_coefs = self.metadata.get("policy_coefficients")
        if policy_coefs is not None:
            return self.basis.interpolate(policy_coefs, s, backend=self.backend)
        raise ValueError("Policy function not available in this solution.")

    def value(self, s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate the continuous value function V(s)."""
        method = self.metadata.get("method", "euler")
        if method == "bellman":
            return self.basis.interpolate(self.coefficients, s, backend=self.backend)
        # Euler method stores value coefficients in metadata if evaluated
        val_coefs = self.metadata.get("value_coefficients")
        if val_coefs is not None:
            return self.basis.interpolate(val_coefs, s, backend=self.backend)
        raise ValueError("Value function was not solved or not available in this solution.")

    def summary(self) -> pd.DataFrame:
        """Structured summary DataFrame of collocation solver results."""
        domain_str = ", ".join(f"[{a:.4g}, {b:.4g}]" for a, b in self.basis.domain)
        orders_str = ", ".join(str(o) for o in self.basis.orders)
        elapsed = self.metadata.get("elapsed_time", np.nan)

        rows = [
            {"Metric": "Method", "Value": str(self.metadata.get("method", "collocation"))},
            {"Metric": "Basis Family", "Value": str(self.basis.basis_type)},
            {"Metric": "Node Type", "Value": str(self.basis.node_type)},
            {"Metric": "State Dimensions", "Value": str(self.basis.n_dims)},
            {"Metric": "Polynomial Orders", "Value": orders_str},
            {"Metric": "Total Collocation Nodes", "Value": str(self.basis.n_nodes)},
            {"Metric": "State Domain", "Value": domain_str},
            {"Metric": "Converged", "Value": str(self.converged)},
            {"Metric": "Iterations", "Value": str(self.n_iter)},
            {"Metric": "Residual Norm", "Value": f"{self.residual_norm:.4e}"},
            {"Metric": "Backend", "Value": str(self.backend)},
            {"Metric": "Elapsed Time (s)", "Value": f"{elapsed:.4f}" if np.isfinite(elapsed) else "N/A"},
        ]
        return pd.DataFrame(rows).set_index("Metric")

    def to_frame(self) -> pd.DataFrame:
        """Return summary statistics as a DataFrame."""
        return self.summary()

    def to_markdown(self, **kwargs) -> str:
        """Render summary table as Markdown."""
        return _df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Render summary table as LaTeX tabular."""
        return _df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Render summary table as Typst table."""
        return _df_to_typst(self.summary(), **kwargs)

    def plot(
        self, figsize: Tuple[float, float] = (10, 4), show: bool = False
    ) -> plt.Figure:
        """Plot the continuous policy, value function, and node collocation points.

        Returns
        -------
        matplotlib.figure.Figure
            Figure instance containing the diagnostic subplots.
        """
        import matplotlib.pyplot as plt

        if self.basis.n_dims == 1:
            a, b = self.basis.domain[0]
            s_dense = np.linspace(a, b, 200)
            nodes = self.basis.nodes(squeeze=True)

            has_value = True
            try:
                self.value(s_dense[:2])
            except Exception:
                has_value = False

            n_cols = 3 if has_value else 2
            fig, axes = plt.subplots(1, n_cols, figsize=figsize, constrained_layout=True)

            # Panel 1: Policy function g(s)
            ax1 = axes[0]
            pol_dense = self.policy(s_dense)
            pol_nodes = self.policy(nodes)
            ax1.plot(s_dense, pol_dense, color="tab:blue", lw=2, label="$g(s)$")
            ax1.scatter(nodes, pol_nodes, color="tab:red", s=30, zorder=5, label="Nodes")
            ax1.plot(s_dense, s_dense, color="gray", ls="--", alpha=0.7, label="$s'=s$")
            ax1.set_title("Policy Function $s' = g(s)$")
            ax1.set_xlabel("State $s$")
            ax1.set_ylabel("Next State $s'$")
            ax1.legend(loc="best", frameon=True)
            ax1.grid(True, alpha=0.3)

            col_idx = 1
            if has_value:
                # Panel 2: Value function V(s)
                ax2 = axes[1]
                val_dense = self.value(s_dense)
                val_nodes = self.value(nodes)
                ax2.plot(s_dense, val_dense, color="tab:green", lw=2, label="$V(s)$")
                ax2.scatter(nodes, val_nodes, color="tab:red", s=30, zorder=5, label="Nodes")
                ax2.set_title("Value Function $V(s)$")
                ax2.set_xlabel("State $s$")
                ax2.set_ylabel("$V(s)$")
                ax2.grid(True, alpha=0.3)
                col_idx = 2

            # Panel 3: Net change g(s) - s
            ax3 = axes[col_idx]
            net_change = pol_dense - s_dense
            ax3.plot(s_dense, net_change, color="tab:purple", lw=2)
            ax3.axhline(0.0, color="gray", ls="--", alpha=0.7)
            ax3.set_title("Net Policy Change $g(s) - s$")
            ax3.set_xlabel("State $s$")
            ax3.set_ylabel("$\\Delta s$")
            ax3.grid(True, alpha=0.3)

        else:
            # Multi-dimensional: slice through dim 0 holding others at midpoint
            fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
            a, b = self.basis.domain[0]
            s0_dense = np.linspace(a, b, 200)
            pts = np.zeros((200, self.basis.n_dims))
            pts[:, 0] = s0_dense
            for d in range(1, self.basis.n_dims):
                pts[:, d] = 0.5 * (self.basis.domain[d][0] + self.basis.domain[d][1])

            pol_slice = self.policy(pts)
            ax.plot(s0_dense, pol_slice, color="tab:blue", lw=2, label="Policy slice")
            ax.set_title(f"Policy Function (Dim 0 slice)")
            ax.set_xlabel("State $s_0$")
            ax.grid(True, alpha=0.3)

        if show:
            plt.show()
        return fig


# ---------------------------------------------------------------------------
# CollocationProblem Class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CollocationProblem:
    """Continuous dynamic programming problem solved via orthogonal polynomial collocation.

    Parameters
    ----------
    domain : tuple[tuple[float, float], ...] | tuple[float, float] | list
        Bounded state space domain.
    orders : tuple[int, ...] | int
        Polynomial degree for each continuous dimension.
    method : str, default "euler"
        Solution method:
        - "euler": Policy function projection minimizing Euler equation residuals.
        - "bellman": Continuous value function iteration on Chebyshev nodes.
    node_type : str, default "lobatto"
        Chebyshev node distribution ("lobatto" or "gauss").
    return_fn : Callable, optional
        One-period return/utility function u(sp, s, **params).
    transition_fn : Callable, optional
        Transition rule or resource equation.
    euler_residual_fn : Callable, optional
        User Euler equation residual function.
    beta : float, default 0.96
        Subjective discount factor in (0, 1).
    params : dict, optional
        Structural economic parameters (e.g. {'alpha': 0.36, 'delta': 1.0}).
    options : dict, optional
        Algorithmic options: 'tol', 'max_iter', 'solver', 'n_howard', 'verbose'.
    """

    domain: Union[Tuple[Tuple[float, float], ...], Tuple[float, float], list]
    orders: Union[Tuple[int, ...], int]
    method: str = "euler"
    node_type: str = "lobatto"
    return_fn: Callable | None = None
    transition_fn: Callable | None = None
    euler_residual_fn: Callable | None = None
    beta: float = 0.96
    params: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 < self.beta < 1.0):
            raise ValueError(f"beta must be in (0, 1); got {self.beta}")

        if self.method not in ("euler", "bellman"):
            raise ValueError(f"method must be 'euler' or 'bellman'; got {self.method!r}")

        if self.node_type not in ("lobatto", "gauss"):
            raise ValueError(f"node_type must be 'lobatto' or 'gauss'; got {self.node_type!r}")

        # Normalize and validate domain
        if isinstance(self.domain, (tuple, list)):
            if len(self.domain) == 2 and not isinstance(self.domain[0], (tuple, list)):
                norm_domain = ((float(self.domain[0]), float(self.domain[1])),)
            else:
                norm_domain = tuple((float(b[0]), float(b[1])) for b in self.domain)
        else:
            raise ValueError("domain must be a tuple or list of (lower, upper) bounds")

        if len(norm_domain) == 0:
            raise ValueError("domain cannot be empty")

        for d_idx, (a, b) in enumerate(norm_domain):
            if a >= b:
                raise ValueError(f"domain bounds for dim {d_idx} must satisfy a < b; got ({a}, {b})")

        # Normalize and validate orders
        if isinstance(self.orders, (int, np.integer)):
            norm_orders = (int(self.orders),)
        elif isinstance(self.orders, (tuple, list)):
            norm_orders = tuple(int(o) for o in self.orders)
        else:
            raise ValueError("orders must be an integer or a tuple of integers")

        if len(norm_orders) != len(norm_domain):
            raise ValueError(
                f"Dimension mismatch: domain has {len(norm_domain)} dimensions, "
                f"but orders has {len(norm_orders)}"
            )

        for o_idx, o in enumerate(norm_orders):
            if o < 1:
                raise ValueError(f"order for dim {o_idx} must be >= 1; got {o}")

        object.__setattr__(self, "domain", norm_domain)
        object.__setattr__(self, "orders", norm_orders)

        # Validate function requirements
        if self.method == "euler":
            if self.euler_residual_fn is None and self.return_fn is None and "alpha" not in self.params:
                raise ValueError(
                    "method='euler' requires euler_residual_fn, return_fn, or model parameters (e.g. 'alpha')"
                )
        elif self.method == "bellman":
            if self.return_fn is None and "alpha" not in self.params:
                raise ValueError("method='bellman' requires return_fn or model parameters (e.g. 'alpha')")

    def solve(self, backend: str = "numpy") -> CollocationSolution:
        """Solve the dynamic model using orthogonal polynomial collocation.

        Parameters
        ----------
        backend : str, default "numpy"
            Compute backend: "numpy", "numba", "mlx", "cupy".

        Returns
        -------
        CollocationSolution
            Converged polynomial collocation solution.
        """
        backend = str(backend).strip().lower()
        if backend not in _bk.SUPPORTED:
            raise ValueError(f"Unknown backend '{backend}'; supported: {_bk.SUPPORTED}")

        resolved_backend = backend
        if not _bk.backend_available(backend):
            import warnings
            warnings.warn(
                f"Backend '{backend}' requested but not available; falling back to 'numpy'.",
                UserWarning,
                stacklevel=2,
            )
            resolved_backend = "numpy"

        basis = CollocationBasis(
            domain=self.domain,
            orders=self.orders,
            basis_type="chebyshev",
            node_type=self.node_type,
        )

        t0 = time.perf_counter()
        if self.method == "euler":
            sol = _solve_collocation_euler(self, basis, backend=resolved_backend)
        else:
            sol = _solve_collocation_bellman(self, basis, backend=resolved_backend)
        t1 = time.perf_counter()

        sol.metadata["elapsed_time"] = t1 - t0
        sol.metadata["backend"] = resolved_backend
        return sol


# ---------------------------------------------------------------------------
# Solver Implementations: Euler Projection & Bellman Value Collocation
# ---------------------------------------------------------------------------

def _evaluate_euler_residual(
    problem: CollocationProblem,
    basis: CollocationBasis,
    theta: np.ndarray,
    nodes: np.ndarray,
    backend: str,
) -> np.ndarray:
    """Evaluate Euler equation residuals at collocation nodes given candidate theta."""
    # Policy candidate
    def policy_cand(s):
        return basis.interpolate(theta, s, backend=backend)

    # 1D node array
    s = nodes[:, 0] if basis.n_dims == 1 else nodes

    if problem.euler_residual_fn is not None:
        fn = problem.euler_residual_fn
        # Try different user signatures
        try:
            return np.asarray(fn(policy_cand, s, problem.params))
        except TypeError:
            try:
                return np.asarray(fn(policy_cand, s))
            except TypeError:
                pass

        sp = policy_cand(s)
        spp = policy_cand(sp)
        try:
            return np.asarray(fn(s, sp, spp, problem.params))
        except TypeError:
            try:
                return np.asarray(fn(s, sp, spp))
            except TypeError:
                try:
                    return np.asarray(fn(s, sp, problem.params))
                except TypeError:
                    return np.asarray(fn(s, sp))

    # Default: Canonical Neoclassical Growth Model
    alpha = problem.params.get("alpha", 0.36)
    delta = problem.params.get("delta", 1.0)
    sigma = problem.params.get("sigma", 1.0)
    z = float(problem.params.get("z", problem.params.get("A", 1.0)))

    kp = policy_cand(s)
    # Ensure physical viability
    f_k = z * (s**alpha) + (1.0 - delta) * s
    kp_clamped = np.clip(kp, 1e-8, f_k - 1e-8)
    c = np.maximum(f_k - kp_clamped, 1e-12)

    f_kp = z * (kp_clamped**alpha) + (1.0 - delta) * kp_clamped
    kpp = policy_cand(kp_clamped)
    kpp_clamped = np.clip(kpp, 1e-8, f_kp - 1e-8)
    cp = np.maximum(f_kp - kpp_clamped, 1e-12)

    if sigma == 1.0:
        uc = 1.0 / c
        ucp = 1.0 / cp
    else:
        uc = c ** (-sigma)
        ucp = cp ** (-sigma)

    fkp = z * alpha * (kp_clamped ** (alpha - 1.0)) + 1.0 - delta
    euler_res = 1.0 - problem.beta * (ucp / uc) * fkp
    return euler_res


def _solve_collocation_euler(
    problem: CollocationProblem, basis: CollocationBasis, backend: str
) -> CollocationSolution:
    """Solve Euler equation collocation problem."""
    nodes = basis.nodes()
    s = nodes[:, 0] if basis.n_dims == 1 else nodes
    tol = problem.options.get("tol", 1e-8)
    max_iter = problem.options.get("max_iter", 500)
    solver_mode = problem.options.get("solver", "hybrid")

    # Initial guess for policy coefficients: generic, model-agnostic neutral guess
    alpha = problem.params.get("alpha", 0.36)
    delta = problem.params.get("delta", 1.0)
    init_policy = 0.95 * s

    theta_0 = basis.fit(init_policy, backend=backend)

    def residual_obj(th):
        return _evaluate_euler_residual(problem, basis, th, nodes, backend)

    converged = False
    n_iter = 0
    theta_opt = theta_0

    if solver_mode in ("hybrid", "root"):
        if basis.n_dims == 1 and "alpha" in problem.params:
            # Euler time iteration warm-up (Coleman operator) to provide a contractive,
            # physically consistent starting point for the nonlinear root-finder
            z = float(problem.params.get("z", problem.params.get("A", 1.0)))
            theta_curr = theta_0.copy()
            for it in range(20):
                theta_prev = theta_curr.copy()
                kp_next = np.zeros(len(s))
                for i, ki in enumerate(s):
                    f_ki = z * (ki**alpha) + (1.0 - delta) * ki

                    def eq_i(kp):
                        c = max(f_ki - kp, 1e-12)
                        kpp = float(np.clip(basis.interpolate(theta_prev, kp, backend=backend), 1e-6, z * (kp**alpha) - 1e-6))
                        cp = max(z * (kp**alpha) + (1.0 - delta) * kp - kpp, 1e-12)
                        return 1.0 / c - problem.beta * (1.0 / cp) * (z * alpha * (kp ** (alpha - 1.0)) + 1.0 - delta)

                    low = min(1e-5, 0.01 * f_ki)
                    high = max(f_ki - 1e-5, 0.99 * f_ki)
                    if low >= high:
                        low = 0.001 * f_ki
                        high = 0.999 * f_ki
                    try:
                        kp_next[i] = brentq(eq_i, low, high)
                    except Exception:
                        kp_next[i] = 0.5 * f_ki

                theta_curr = basis.fit(kp_next, backend=backend)
                if np.max(np.abs(theta_curr - theta_prev)) < 1e-6:
                    break
            theta_0 = theta_curr

        # Powell's hybrid method
        res = root(residual_obj, theta_0, method="hybr", tol=tol, options={"maxfev": max_iter * 2})
        if res.success:
            theta_opt = res.x
            converged = True
            n_iter = int(res.nfev)
        else:
            # Fallback to LM
            res_lm = root(residual_obj, theta_0, method="lm", tol=tol)
            theta_opt = res_lm.x if res_lm.success else res.x
            converged = bool(res_lm.success or res.success)
            n_iter = int(getattr(res_lm, "nfev", getattr(res, "nfev", 0)))

    final_residuals = residual_obj(theta_opt)
    residual_norm = float(np.max(np.abs(final_residuals)))
    if residual_norm < tol * 10:
        converged = True

    # Policy evaluation for value function coefficients: (I - beta * Phi_next @ Phi_inv) @ V = u
    value_coefficients = None
    try:
        Phi_nodes = basis.evaluate(nodes, backend=backend)
        Phi_inv = np.linalg.inv(Phi_nodes)
        kp_nodes = basis.interpolate(theta_opt, s, backend=backend)
        kp_nodes_clamped = np.clip(kp_nodes, basis.domain[0][0], basis.domain[0][1])
        Phi_next = basis.evaluate(kp_nodes_clamped, backend=backend)

        if problem.return_fn is not None:
            u_nodes = np.asarray(problem.return_fn(kp_nodes_clamped, s, **problem.params))
        elif "alpha" in problem.params:
            c_nodes = np.maximum(s**alpha + (1.0 - delta) * s - kp_nodes_clamped, 1e-12)
            u_nodes = np.log(c_nodes)
        else:
            u_nodes = None

        if u_nodes is not None:
            A_mat = np.eye(len(s)) - problem.beta * (Phi_next @ Phi_inv)
            V_nodes = np.linalg.solve(A_mat, u_nodes)
            value_coefficients = Phi_inv @ V_nodes
    except Exception:
        pass

    metadata = {
        "method": "euler",
        "policy_coefficients": theta_opt,
        "value_coefficients": value_coefficients,
        "residuals": final_residuals,
        "nodes": nodes,
        "backend": backend,
    }

    return CollocationSolution(
        coefficients=theta_opt,
        basis=basis,
        converged=converged,
        n_iter=n_iter,
        residual_norm=residual_norm,
        backend=backend,
        metadata=metadata,
    )


def _solve_collocation_bellman(
    problem: CollocationProblem, basis: CollocationBasis, backend: str
) -> CollocationSolution:
    """Solve continuous Bellman value function collocation problem."""
    nodes = basis.nodes()
    s = nodes[:, 0] if basis.n_dims == 1 else nodes
    tol = problem.options.get("tol", 1e-8)
    max_iter = problem.options.get("max_iter", 500)
    n_howard = problem.options.get("n_howard", 10)

    Phi_nodes = basis.evaluate(nodes, backend=backend)
    Phi_inv = np.linalg.inv(Phi_nodes)

    # Initial guess for value coefficients
    alpha = problem.params.get("alpha", 0.36)
    delta = problem.params.get("delta", 1.0)
    c_v = np.zeros(basis.n_nodes)

    def val_fn(cv, state):
        return basis.interpolate(cv, state, backend=backend)

    converged = False
    kp_opt = np.zeros(len(s))
    V_new = np.zeros(len(s))

    for it in range(1, max_iter + 1):
        c_v_old = c_v.copy()

        for i, ki in enumerate(s):
            a_bound, b_bound = basis.domain[0]
            if "alpha" in problem.params:
                z = float(problem.params.get("z", problem.params.get("A", 1.0)))
                f_ki = z * (ki**alpha) + (1.0 - delta) * ki
                b_bound = min(b_bound, f_ki - 1e-5)
                a_bound = max(a_bound, 1e-5)

            def objective(kp):
                if problem.return_fn is not None:
                    u_val = problem.return_fn(kp, ki, **problem.params)
                else:
                    z = float(problem.params.get("z", problem.params.get("A", 1.0)))
                    c = max(z * (ki**alpha) + (1.0 - delta) * ki - kp, 1e-12)
                    u_val = np.log(c)
                v_cont = val_fn(c_v_old, kp)
                return -(u_val + problem.beta * v_cont)

            res = minimize_scalar(
                objective, bounds=(a_bound, b_bound), method="bounded", options={"xatol": 1e-8}
            )
            kp_opt[i] = res.x
            V_new[i] = -res.fun

        # Howard policy iteration acceleration steps
        if n_howard > 0:
            if problem.return_fn is not None:
                u_howard = np.asarray(problem.return_fn(kp_opt, s, **problem.params))
            else:
                z = float(problem.params.get("z", problem.params.get("A", 1.0)))
                c_howard = np.maximum(z * (s**alpha) + (1.0 - delta) * s - kp_opt, 1e-12)
                u_howard = np.log(c_howard)

            kp_clamped = np.clip(kp_opt, basis.domain[0][0], basis.domain[0][1])
            Phi_kp = basis.evaluate(kp_clamped, backend=backend)
            for _ in range(n_howard):
                V_new = u_howard + problem.beta * (Phi_kp @ (Phi_inv @ V_new))

        c_v = Phi_inv @ V_new
        sup_diff = float(np.max(np.abs(c_v - c_v_old)))
        if sup_diff < tol * (1.0 - problem.beta) / max(problem.beta, 1e-3) or sup_diff < tol:
            converged = True
            break

    residual_norm = sup_diff
    policy_coefficients = basis.fit(kp_opt, backend=backend)

    metadata = {
        "method": "bellman",
        "value_coefficients": c_v,
        "policy_coefficients": policy_coefficients,
        "nodes": nodes,
        "backend": backend,
    }

    return CollocationSolution(
        coefficients=c_v,
        basis=basis,
        converged=converged,
        n_iter=it,
        residual_norm=residual_norm,
        backend=backend,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Functional Entry Point
# ---------------------------------------------------------------------------

def solve_collocation(
    problem: CollocationProblem | None = None,
    backend: str = "numpy",
    **kwargs: Any,
) -> CollocationSolution:
    """Solve a continuous dynamic economic model via polynomial collocation.

    Parameters
    ----------
    problem : CollocationProblem, optional
        Pre-configured problem instance.
    backend : str, default "numpy"
        Compute backend: "numpy", "numba", "mlx", "cupy".
    **kwargs : Any
        Keyword arguments passed to CollocationProblem if ``problem`` is None.

    Returns
    -------
    CollocationSolution
        Converged solution object.
    """
    if problem is None:
        problem = CollocationProblem(**kwargs)
    elif kwargs:
        raise ValueError("Cannot pass both a CollocationProblem instance and keyword arguments.")

    return problem.solve(backend=backend)


__all__ = [
    "CollocationBasis",
    "CollocationProblem",
    "CollocationSolution",
    "solve_collocation",
]
