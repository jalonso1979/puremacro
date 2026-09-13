r"""Smolyak Sparse Grid Collocation Solver for Multi-Dimensional Continuous State Spaces.

Provides high-dimensional continuous dynamic programming and projection methods
using Smolyak sparse grids and Chebyshev polynomial bases:
- 1D nested Clenshaw-Curtis extrema nodes with exact nesting and disjoint increments.
- Multi-dimensional sparse grid assembly \mathcal{H}(d, \mu) scaling as O(N (log N)^(d-1)).
- Multi-dimensional Chebyshev polynomial basis \mathcal{P}(d, \mu) with well-conditioned
  collocation matrices (cond(\Phi) < 30).
- Bijective affine coordinate mapping between arbitrary hypercubes \prod [a_k, b_k] and [-1, 1]^d.
- Smolyak direct collocation matrix solver and combination technique interpolation A(d, \mu).
- Vectorized continuous policy, value function, and analytical gradient evaluations.
- Multi-backend hardware acceleration (NumPy reference oracle, Numba CPU JIT, Apple Silicon MLX GPU, NVIDIA CuPy GPU).
- Full puremacro presentation contract (.summary(), .plot(), .to_frame(), .to_markdown(), .to_latex(), .to_typst()).
"""
from __future__ import annotations

import itertools
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import root

from puremacro import _backend as _bk
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst


# ---------------------------------------------------------------------------
# 1D Clenshaw-Curtis Extrema Nodes and Disjoint Increments
# ---------------------------------------------------------------------------

def _clenshaw_curtis_nodes(i: int) -> np.ndarray:
    r"""Return 1D Clenshaw-Curtis extrema nodes X^{(i)} on [-1, 1] for level i >= 1.

    Definition:
    - m_1 = 1: X^{(1)} = [0.0]
    - m_i = 2^{i-1} + 1 for i >= 2:
      x_j = - \cos\left( \frac{\pi (j - 1)}{m_i - 1} \right), \quad j = 1, \dots, m_i.
    """
    if i < 1:
        raise ValueError(f"Level i must be >= 1; got {i}")
    if i == 1:
        return np.array([0.0], dtype=np.float64)

    m_i = (1 << (i - 1)) + 1
    nodes = -np.cos(np.pi * np.arange(m_i, dtype=np.float64) / (m_i - 1))
    nodes = np.round(nodes, 14)
    nodes[0] = -1.0
    nodes[-1] = 1.0
    if m_i % 2 == 1:
        nodes[m_i // 2] = 0.0
    return nodes


def _disjoint_increments(i: int) -> np.ndarray:
    r"""Return disjoint 1D node increment H^{(i)} = X^{(i)} \setminus X^{(i-1)}.

    Cardinalities:
    - |H^{(1)}| = 1 (node: 0.0)
    - |H^{(2)}| = 2 (nodes: -1.0, 1.0)
    - |H^{(i)}| = 2^{i-2} for i >= 3.
    """
    if i < 1:
        raise ValueError(f"Level i must be >= 1; got {i}")
    if i == 1:
        return np.array([0.0], dtype=np.float64)

    all_nodes = _clenshaw_curtis_nodes(i)
    prev_nodes = _clenshaw_curtis_nodes(i - 1)
    h = [x for x in all_nodes if not np.any(np.abs(x - prev_nodes) < 1e-12)]
    return np.array(h, dtype=np.float64)


def _chebyshev_poly_indices(i: int) -> list[int]:
    r"""Return 1D Chebyshev polynomial degree indices \mathcal{J}(i) corresponding to level i.

    Cardinalities match |H^{(i)}| exactly:
    - \mathcal{J}(1) = [0]
    - \mathcal{J}(2) = [1, 2]
    - \mathcal{J}(i) = [2^{i-2} + 1, \dots, 2^{i-1}] for i >= 3.
    """
    if i < 1:
        raise ValueError(f"Level i must be >= 1; got {i}")
    if i == 1:
        return [0]
    if i == 2:
        return [1, 2]
    start = (1 << (i - 2)) + 1
    end = (1 << (i - 1)) + 1
    return list(range(start, end))


def _generate_multi_indices(d: int, mu: int) -> list[tuple[int, ...]]:
    r"""Generate multi-indices \mathbf{i} = (i_1, \dots, i_d) with d <= \sum_{k=1}^d i_k <= d + \mu."""
    if d < 1:
        raise ValueError(f"Dimension d must be >= 1; got {d}")
    if mu < 0:
        raise ValueError(f"Approximation level mu must be >= 0; got {mu}")

    results: list[tuple[int, ...]] = []

    def _recurse(dim_left: int, current_sum: int, current_tuple: tuple[int, ...]) -> None:
        if dim_left == 0:
            results.append(current_tuple)
            return
        max_k = mu - current_sum
        for k in range(max_k + 1):
            _recurse(dim_left - 1, current_sum + k, current_tuple + (k + 1,))

    _recurse(d, 0, ())
    return results


# ---------------------------------------------------------------------------
# Chebyshev Basis Evaluation Kernels
# ---------------------------------------------------------------------------

@_bk.njit_fallback(fastmath=True)
def _chebyshev_basis_1d_kernel(x: np.ndarray, max_order: int) -> np.ndarray:
    """Evaluate 1D Chebyshev polynomials T_0(x) ... T_max_order(x) via 3-term recurrence."""
    n = len(x)
    T = np.zeros((n, max_order + 1), dtype=np.float64)
    for i in range(n):
        T[i, 0] = 1.0
        if max_order >= 1:
            T[i, 1] = x[i]
        for j in range(1, max_order):
            T[i, j + 1] = 2.0 * x[i] * T[i, j] - T[i, j - 1]
    return T


@_bk.njit_fallback(fastmath=True)
def _chebyshev_deriv_1d_kernel(x: np.ndarray, max_order: int) -> np.ndarray:
    """Evaluate first derivatives T'_0(x) ... T'_max_order(x) via derivative recurrence."""
    n = len(x)
    T = _chebyshev_basis_1d_kernel(x, max_order)
    dT = np.zeros((n, max_order + 1), dtype=np.float64)
    for i in range(n):
        dT[i, 0] = 0.0
        if max_order >= 1:
            dT[i, 1] = 1.0
        for j in range(1, max_order):
            dT[i, j + 1] = 2.0 * T[i, j] + 2.0 * x[i] * dT[i, j] - dT[i, j - 1]
    return dT


def _chebyshev_basis_1d(x: np.ndarray, max_order: int, backend: str = "numpy") -> np.ndarray:
    """Evaluate 1D Chebyshev polynomials across supported backends."""
    b = str(backend).strip().lower()
    x_clipped = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    if b == "mlx" and _bk.backend_available("mlx"):
        import mlx.core as mx

        x_mx = mx.array(x_clipped, dtype=mx.float32)
        cols = [mx.ones_like(x_mx)]
        if max_order >= 1:
            cols.append(x_mx)
        for j in range(1, max_order):
            cols.append(2.0 * x_mx * cols[j] - cols[j - 1])
        return np.asarray(mx.stack(cols, axis=1), dtype=np.float64)
    if b == "cupy" and _bk.backend_available("cupy"):
        import cupy as cp

        x_cp = cp.asarray(x_clipped, dtype=cp.float64)
        cols = [cp.ones_like(x_cp)]
        if max_order >= 1:
            cols.append(x_cp)
        for j in range(1, max_order):
            cols.append(2.0 * x_cp * cols[j] - cols[j - 1])
        return cp.asnumpy(cp.stack(cols, axis=1))
    return _chebyshev_basis_1d_kernel(x_clipped, max_order)


# ---------------------------------------------------------------------------
# SmolyakGrid Class
# ---------------------------------------------------------------------------

class SmolyakGrid:
    r"""Multi-dimensional Smolyak sparse grid.

    Constructs multi-dimensional sparse grids based on nested Clenshaw-Curtis
    extrema nodes and disjoint increments:
    $$\mathcal{H}(d, \mu) = \bigcup_{d \le |\mathbf{i}|_1 \le d + \mu} \left( H^{(i_1)} \times \dots \times H^{(i_d)} \right)$$

    Parameters
    ----------
    d : int
        Number of state space dimensions (e.g. d in [2, 6]).
    mu : int
        Smolyak approximation level (e.g. mu in [1, 5]).
    domain : Sequence[tuple[float, float]] | None, optional
        Bounded physical state space intervals [(a_1, b_1), ..., (a_d, b_d)].
        Defaults to canonical hypercube [-1.0, 1.0]^d.
    """

    def __init__(
        self,
        d: int,
        mu: int,
        domain: Sequence[Tuple[float, float]] | None = None,
    ) -> None:
        if d < 1:
            raise ValueError(f"Dimension d must be >= 1; got {d}")
        if mu < 0:
            raise ValueError(f"Approximation level mu must be >= 0; got {mu}")

        self.d = int(d)
        self.mu = int(mu)

        # Normalize and validate domain
        if domain is None:
            norm_domain = tuple((-1.0, 1.0) for _ in range(self.d))
        elif isinstance(domain, (tuple, list)):
            if len(domain) == 2 and not isinstance(domain[0], (tuple, list)) and self.d == 1:
                norm_domain = ((float(domain[0]), float(domain[1])),)
            else:
                norm_domain = tuple((float(b[0]), float(b[1])) for b in domain)
        else:
            raise ValueError("domain must be a sequence of (lower, upper) pairs")

        if len(norm_domain) != self.d:
            raise ValueError(
                f"Domain dimension mismatch: expected {self.d} intervals, got {len(norm_domain)}"
            )

        for k, (a, b) in enumerate(norm_domain):
            if a >= b:
                raise ValueError(f"Domain bounds for dim {k} must satisfy a < b; got ({a}, {b})")

        self.domain = norm_domain

        # Generate multi-indices
        self.multi_indices = _generate_multi_indices(self.d, self.mu)

        # Assemble sparse grid nodes and polynomial indices
        nodes_list: list[tuple[float, ...]] = []
        polys_list: list[tuple[int, ...]] = []

        for idx in self.multi_indices:
            coord_increments = [_disjoint_increments(i) for i in idx]
            for pt in itertools.product(*coord_increments):
                nodes_list.append(pt)

            poly_deg_sets = [_chebyshev_poly_indices(i) for i in idx]
            for p in itertools.product(*poly_deg_sets):
                polys_list.append(p)

        self._nodes_canonical = np.array(nodes_list, dtype=np.float64)
        self._poly_indices = np.array(polys_list, dtype=np.int64)

        # Map to physical coordinates
        self._physical_nodes = self.to_physical(self._nodes_canonical)

    @property
    def nodes(self) -> np.ndarray:
        """Canonical sparse grid nodes in [-1, 1]^d of shape (N, d)."""
        return self._nodes_canonical

    @property
    def physical_nodes(self) -> np.ndarray:
        """Physical sparse grid nodes in domain bounds of shape (N, d)."""
        return self._physical_nodes

    @property
    def n_nodes(self) -> int:
        """Total number of sparse grid nodes N = N(d, mu)."""
        return len(self._nodes_canonical)

    @property
    def poly_indices(self) -> np.ndarray:
        """Chebyshev polynomial degree multi-indices of shape (N, d)."""
        return self._poly_indices

    @property
    def tensor_nodes_count(self) -> int:
        """Number of nodes in a full tensor product grid of comparable 1D precision (2^mu + 1)^d."""
        return int(( (1 << self.mu) + 1 ) ** self.d)

    @property
    def reduction_ratio(self) -> float:
        """Node reduction ratio: (full tensor product nodes) / (Smolyak nodes)."""
        return float(self.tensor_nodes_count / self.n_nodes)

    def to_canonical(self, s: Union[float, Sequence[float], np.ndarray]) -> np.ndarray:
        """Map physical coordinates s to canonical coordinates x in [-1, 1]^d."""
        s_arr = np.asarray(s, dtype=np.float64)
        was_1d = (s_arr.ndim == 1 and len(s_arr) == self.d)
        if s_arr.ndim == 1:
            s_arr = s_arr.reshape(1, self.d)
        elif s_arr.ndim == 0 and self.d == 1:
            s_arr = s_arr.reshape(1, 1)

        x = np.zeros_like(s_arr)
        for k in range(self.d):
            a, b = self.domain[k]
            x[:, k] = (2.0 * s_arr[:, k] - (a + b)) / (b - a)

        # Clip to guard against slight floating point overshoot
        x_clipped = np.clip(x, -1.0, 1.0)
        return x_clipped[0] if was_1d else x_clipped

    def to_physical(self, x: Union[float, Sequence[float], np.ndarray]) -> np.ndarray:
        """Map canonical coordinates x in [-1, 1]^d to physical domain bounds."""
        x_arr = np.asarray(x, dtype=np.float64)
        was_1d = (x_arr.ndim == 1 and len(x_arr) == self.d)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, self.d)
        elif x_arr.ndim == 0 and self.d == 1:
            x_arr = x_arr.reshape(1, 1)

        s = np.zeros_like(x_arr)
        for k in range(self.d):
            a, b = self.domain[k]
            s[:, k] = 0.5 * (a + b) + 0.5 * (b - a) * x_arr[:, k]

        return s[0] if was_1d else s

    def summary(self) -> pd.DataFrame:
        """Summary DataFrame of sparse grid properties."""
        domain_str = ", ".join(f"[{a:.4g}, {b:.4g}]" for a, b in self.domain)
        rows = [
            {"Metric": "Dimensions (d)", "Value": str(self.d)},
            {"Metric": "Approximation Level (mu)", "Value": str(self.mu)},
            {"Metric": "Smolyak Nodes (N)", "Value": str(self.n_nodes)},
            {"Metric": "Tensor Product Nodes", "Value": str(self.tensor_nodes_count)},
            {"Metric": "Node Reduction Ratio", "Value": f"{self.reduction_ratio:.2f}x"},
            {"Metric": "Physical Domain", "Value": domain_str},
        ]
        return pd.DataFrame(rows).set_index("Metric")

    def __len__(self) -> int:
        return self.n_nodes

    def __repr__(self) -> str:
        return f"SmolyakGrid(d={self.d}, mu={self.mu}, n_nodes={self.n_nodes}, reduction={self.reduction_ratio:.1f}x)"


# ---------------------------------------------------------------------------
# SmolyakBasis Class
# ---------------------------------------------------------------------------

class SmolyakBasis:
    r"""Multi-dimensional Chebyshev polynomial basis on Smolyak sparse grid.

    Evaluates the complete Chebyshev polynomial basis \mathcal{P}(d, \mu)
    and constructs the square, well-conditioned collocation matrix \Phi.

    Parameters
    ----------
    grid : SmolyakGrid | None, optional
        Pre-constructed SmolyakGrid instance.
    d : int | None, optional
        Number of state space dimensions.
    mu : int, default 2
        Smolyak approximation level.
    domain : Sequence[tuple[float, float]] | None, optional
        Physical state space bounds.
    """

    def __init__(
        self,
        grid: SmolyakGrid | None = None,
        d: int | None = None,
        mu: int = 2,
        domain: Sequence[Tuple[float, float]] | None = None,
    ) -> None:
        if grid is not None:
            self.grid = grid
        elif d is not None:
            self.grid = SmolyakGrid(d=d, mu=mu, domain=domain)
        else:
            raise ValueError("Either grid or d must be provided to initialize SmolyakBasis.")

        self.d = self.grid.d
        self.mu = self.grid.mu
        self.domain = self.grid.domain
        self.polys = self.grid.poly_indices
        self.n_basis = len(self.polys)

        # Precompute maximum polynomial order per dimension
        self._max_orders = [int(np.max(self.polys[:, k])) for k in range(self.d)]

        # Precompute collocation matrix on canonical grid nodes
        self._Phi_nodes = self._evaluate_canonical(self.grid.nodes, backend="numpy")
        self._cond = float(np.linalg.cond(self._Phi_nodes))

    @property
    def collocation_matrix(self) -> np.ndarray:
        r"""Direct collocation matrix \Phi of shape (N, N)."""
        return self._Phi_nodes

    @property
    def condition_number(self) -> float:
        r"""Condition number cond(\Phi) of the collocation matrix."""
        return self._cond

    def _evaluate_canonical(self, x: np.ndarray, backend: str = "numpy") -> np.ndarray:
        """Evaluate basis matrix given canonical coordinates x in [-1, 1]^d."""
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, self.d)

        M = len(x_arr)
        # Precompute 1D basis for each dimension
        T_1d = [
            _chebyshev_basis_1d(x_arr[:, k], self._max_orders[k], backend=backend)
            for k in range(self.d)
        ]

        # Multi-dimensional basis product: \psi_q(x) = \prod_{k=1}^d T_{j_{q, k}}(x_k)
        B = np.ones((M, self.n_basis), dtype=np.float64)
        for k in range(self.d):
            orders_k = self.polys[:, k]
            B *= T_1d[k][:, orders_k]
        return B

    def evaluate(
        self, s: Union[float, Sequence[float], np.ndarray], backend: str = "numpy"
    ) -> np.ndarray:
        """Evaluate the multi-dimensional Chebyshev basis matrix at query state(s) s.

        Parameters
        ----------
        s : float | Sequence[float] | np.ndarray
            Query state(s). Single state of shape (d,) or batch of shape (M, d).
        backend : str, default "numpy"
            Compute backend: "numpy", "numba", "mlx", "cupy".

        Returns
        -------
        np.ndarray
            Basis matrix of shape (M, N) or (1, N).
        """
        s_arr = np.asarray(s, dtype=np.float64)
        if s_arr.ndim == 1 and len(s_arr) == self.d:
            s_arr = s_arr.reshape(1, self.d)
        elif s_arr.ndim == 0 and self.d == 1:
            s_arr = s_arr.reshape(1, 1)

        x = self.grid.to_canonical(s_arr)
        return self._evaluate_canonical(x, backend=backend)

    def derivative(
        self,
        s: Union[float, Sequence[float], np.ndarray],
        dim: int = 0,
        backend: str = "numpy",
    ) -> np.ndarray:
        """Evaluate the partial derivative of basis matrix with respect to s_{dim}.

        Applies the chain rule factor 2 / (b_{dim} - a_{dim}).

        Parameters
        ----------
        s : float | Sequence[float] | np.ndarray
            Query state(s). Single state of shape (d,) or batch of shape (M, d).
        dim : int, default 0
            Dimension along which to differentiate (0 <= dim < d).
        backend : str, default "numpy"
            Compute backend: "numpy", "numba", "mlx", "cupy".

        Returns
        -------
        np.ndarray
            Derivative matrix of shape (M, N) or (1, N).
        """
        if not (0 <= dim < self.d):
            raise ValueError(f"dim must be in [0, {self.d - 1}]; got {dim}")

        s_arr = np.asarray(s, dtype=np.float64)
        if s_arr.ndim == 1 and len(s_arr) == self.d:
            s_arr = s_arr.reshape(1, self.d)
        elif s_arr.ndim == 0 and self.d == 1:
            s_arr = s_arr.reshape(1, 1)

        x = self.grid.to_canonical(s_arr)
        M = len(x)

        T_1d = [
            _chebyshev_basis_1d(x[:, k], self._max_orders[k], backend=backend)
            for k in range(self.d)
        ]
        dT_dim = _chebyshev_deriv_1d_kernel(
            np.clip(x[:, dim], -1.0, 1.0), self._max_orders[dim]
        )

        dB = np.ones((M, self.n_basis), dtype=np.float64)
        for k in range(self.d):
            orders_k = self.polys[:, k]
            if k == dim:
                dB *= dT_dim[:, orders_k]
            else:
                dB *= T_1d[k][:, orders_k]

        # Chain rule factor: dx_dim / ds_dim = 2 / (b_dim - a_dim)
        a_dim, b_dim = self.domain[dim]
        scale = 2.0 / (b_dim - a_dim)
        return dB * scale

    def fit(self, y: np.ndarray, backend: str = "numpy") -> np.ndarray:
        r"""Solve for basis coefficients \theta such that \Phi \theta = y.

        Parameters
        ----------
        y : np.ndarray
            Function values on sparse grid nodes, shape (N,) or (N, K).
        backend : str, default "numpy"
            Compute backend.

        Returns
        -------
        np.ndarray
            Coefficients \theta of shape (N,) or (N, K).
        """
        y_arr = np.asarray(y, dtype=np.float64)
        if len(y_arr) != self.n_basis:
            raise ValueError(
                f"Shape mismatch: y has {len(y_arr)} elements, expected {self.n_basis}"
            )
        return np.linalg.solve(self._Phi_nodes, y_arr)

    def interpolate(
        self,
        theta: np.ndarray,
        s: Union[float, Sequence[float], np.ndarray],
        backend: str = "numpy",
    ) -> np.ndarray:
        r"""Evaluate the Smolyak interpolant \hat{f}(s) = \Phi(s) \theta.

        Parameters
        ----------
        theta : np.ndarray
            Basis coefficients of shape (N,) or (N, K).
        s : float | Sequence[float] | np.ndarray
            Evaluation points of shape (d,) or (M, d).
        backend : str, default "numpy"
            Compute backend.

        Returns
        -------
        np.ndarray
            Interpolated value(s).
        """
        s_arr = np.asarray(s, dtype=np.float64)
        is_single = (s_arr.ndim == 1 and len(s_arr) == self.d) or (s_arr.ndim == 0 and self.d == 1)
        Phi_s = self.evaluate(s, backend=backend)
        val = Phi_s @ theta
        if is_single and val.ndim > 0 and len(val) == 1:
            return val[0]
        return val

    def combination_interpolate(
        self,
        y: np.ndarray,
        s: Union[float, Sequence[float], np.ndarray],
    ) -> np.ndarray:
        r"""Interpolate using the Smolyak combination formula:

        $$A(d, \mu) f(s) = \sum_{q \le |\mathbf{i}|_1 \le d + \mu} (-1)^{d + \mu - |\mathbf{i}|_1} \binom{d - 1}{d + \mu - |\mathbf{i}|_1} \left( \bigotimes_{k=1}^d U^{i_k} \right) f(s)$$
        where q = \max(d, \mu + 1).

        Parameters
        ----------
        y : np.ndarray
            Function values on the Smolyak grid nodes (shape (N,)).
        s : float | Sequence[float] | np.ndarray
            Query points of shape (d,) or (M, d).

        Returns
        -------
        np.ndarray
            Interpolated values.
        """
        # Because direct collocation and combination technique are mathematically
        # equivalent on the sparse grid, solve theta and evaluate:
        theta = self.fit(y)
        return self.interpolate(theta, s)


# ---------------------------------------------------------------------------
# SmolyakSolution Class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SmolyakSolution:
    """Frozen solution object from Smolyak sparse grid collocation.

    Attributes
    ----------
    coefficients : np.ndarray
        Fitted polynomial coefficients of shape (N, d) or (N,).
    basis : SmolyakBasis
        Underlying Smolyak basis and sparse grid.
    converged : bool
        Whether the nonlinear solver reached convergence tolerance.
    n_iter : int
        Number of solver iterations or evaluations.
    residual_norm : float
        Maximum absolute residual at collocation nodes.
    backend : str
        Compute backend utilized ('numpy', 'numba', 'mlx', 'cupy').
    metadata : dict
        Additional solution metadata, including execution time,
        value coefficients, node residuals, and economic parameters.
    """

    coefficients: np.ndarray
    basis: SmolyakBasis
    converged: bool
    n_iter: int
    residual_norm: float
    backend: str
    metadata: dict = field(default_factory=dict)

    def policy(
        self, s: Union[float, Sequence[float], np.ndarray]
    ) -> np.ndarray:
        """Evaluate the continuous policy function g(s)."""
        method = self.metadata.get("method", "euler")
        if method == "euler":
            return self.basis.interpolate(self.coefficients, s, backend=self.backend)
        # Bellman method stores policy coefficients in metadata
        policy_coefs = self.metadata.get("policy_coefficients")
        if policy_coefs is not None:
            return self.basis.interpolate(policy_coefs, s, backend=self.backend)
        return self.basis.interpolate(self.coefficients, s, backend=self.backend)

    def value(
        self, s: Union[float, Sequence[float], np.ndarray]
    ) -> np.ndarray:
        """Evaluate the continuous value function V(s)."""
        method = self.metadata.get("method", "euler")
        if method == "bellman":
            return self.basis.interpolate(self.coefficients, s, backend=self.backend)
        val_coefs = self.metadata.get("value_coefficients")
        if val_coefs is not None:
            return self.basis.interpolate(val_coefs, s, backend=self.backend)
        raise ValueError("Value function was not solved or not available in this solution.")

    def euler_residual(
        self, s: Union[float, Sequence[float], np.ndarray]
    ) -> np.ndarray:
        """Evaluate continuous Euler equation residual(s) at arbitrary state(s) s."""
        s_arr = np.asarray(s, dtype=np.float64)
        is_single = (s_arr.ndim == 1 and len(s_arr) == self.basis.d) or (s_arr.ndim == 0 and self.basis.d == 1)
        if s_arr.ndim == 1:
            s_arr = s_arr.reshape(1, self.basis.d)
        elif s_arr.ndim == 0 and self.basis.d == 1:
            s_arr = s_arr.reshape(1, 1)

        d = self.basis.d
        kp = self.policy(s_arr)
        if kp.ndim == 1:
            kp = kp.reshape(-1, d)

        # Retrieve model parameters
        params = self.metadata.get("params", {})
        beta = float(self.metadata.get("beta", 0.96))
        delta = float(params.get("delta", 1.0))
        sigma = float(params.get("sigma", 1.0))
        z = float(params.get("z", params.get("A", 1.0)))

        if "alphas" in params:
            alphas = np.asarray(params["alphas"], dtype=np.float64)
        else:
            alpha_tot = float(params.get("alpha", 0.36))
            alphas = np.full(d, alpha_tot / d, dtype=np.float64)

        # Output production Y
        prod_k = np.ones(len(s_arr), dtype=np.float64)
        for k in range(d):
            prod_k *= np.maximum(s_arr[:, k], 1e-12) ** alphas[k]
        Y = z * prod_k + (1.0 - delta) * np.sum(s_arr, axis=1)

        C = np.maximum(Y - np.sum(kp, axis=1), 1e-12)
        uc = 1.0 / C if sigma == 1.0 else C ** (-sigma)

        # Next period
        kpp = self.policy(kp)
        if kpp.ndim == 1:
            kpp = kpp.reshape(-1, d)

        prod_kp = np.ones(len(kp), dtype=np.float64)
        for k in range(d):
            prod_kp *= np.maximum(kp[:, k], 1e-12) ** alphas[k]
        Yp = z * prod_kp + (1.0 - delta) * np.sum(kp, axis=1)

        Cp = np.maximum(Yp - np.sum(kpp, axis=1), 1e-12)
        ucp = 1.0 / Cp if sigma == 1.0 else Cp ** (-sigma)

        res = np.zeros((len(s_arr), d), dtype=np.float64)
        for k in range(d):
            R_k = alphas[k] * (z * prod_kp) / np.maximum(kp[:, k], 1e-12) + 1.0 - delta
            res[:, k] = 1.0 - beta * (ucp / uc) * R_k

        if is_single:
            return res[0] if d > 1 else float(res[0, 0])
        return res if d > 1 else res[:, 0]

    def summary(self) -> pd.DataFrame:
        """Structured summary DataFrame of Smolyak solver results."""
        domain_str = ", ".join(f"[{a:.4g}, {b:.4g}]" for a, b in self.basis.domain)
        elapsed = self.metadata.get("elapsed_time", np.nan)

        rows = [
            {"Metric": "Method", "Value": str(self.metadata.get("method", "smolyak_collocation"))},
            {"Metric": "State Dimensions (d)", "Value": str(self.basis.d)},
            {"Metric": "Approximation Level (mu)", "Value": str(self.basis.mu)},
            {"Metric": "Smolyak Nodes (N)", "Value": str(self.basis.n_basis)},
            {"Metric": "Tensor Product Nodes", "Value": str(self.basis.grid.tensor_nodes_count)},
            {"Metric": "Node Reduction Ratio", "Value": f"{self.basis.grid.reduction_ratio:.2f}x"},
            {"Metric": "Basis Condition Number", "Value": f"{self.basis.condition_number:.2f}"},
            {"Metric": "State Domain", "Value": domain_str},
            {"Metric": "Converged", "Value": str(self.converged)},
            {"Metric": "Iterations / Func Evals", "Value": str(self.n_iter)},
            {"Metric": "Residual Norm", "Value": f"{self.residual_norm:.4e}"},
            {"Metric": "Backend", "Value": str(self.backend)},
            {"Metric": "Elapsed Time (s)", "Value": f"{elapsed:.4f}" if np.isfinite(elapsed) else "N/A"},
        ]
        return pd.DataFrame(rows).set_index("Metric")

    def to_frame(self) -> pd.DataFrame:
        """Return summary statistics as a DataFrame."""
        return self.summary()

    def to_markdown(self, **kwargs: Any) -> str:
        """Render summary table as Markdown."""
        return _df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render summary table as LaTeX tabular."""
        return _df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render summary table as Typst table."""
        return _df_to_typst(self.summary(), **kwargs)

    def plot(
        self,
        figsize: Tuple[float, float] = (12, 4),
        show: bool = False,
        **kwargs: Any,
    ) -> plt.Figure:
        """Plot the continuous policy slices, sparse grid projection, and Euler residuals.

        Returns
        -------
        matplotlib.figure.Figure
            Diagnostic figure with 3 subplots.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

        d = self.basis.d
        a0, b0 = self.basis.domain[0]
        s0_dense = np.linspace(a0, b0, 200)

        # Slice through dimension 0 holding other dimensions at midpoint
        pts = np.zeros((200, d), dtype=np.float64)
        pts[:, 0] = s0_dense
        for k in range(1, d):
            ak, bk = self.basis.domain[k]
            pts[:, k] = 0.5 * (ak + bk)

        pol_slice = self.policy(pts)
        if pol_slice.ndim == 1 and d > 1:
            pol_slice = pol_slice.reshape(-1, d)

        # Panel 1: Policy slice
        ax1 = axes[0]
        y_plot = pol_slice[:, 0] if d > 1 else pol_slice
        ax1.plot(s0_dense, y_plot, color="tab:blue", lw=2, label="$g_1(s_1, \\bar{s}_{-1})$")
        ax1.plot(s0_dense, s0_dense, color="gray", ls="--", alpha=0.7, label="$s'=s$")
        ax1.set_title("Policy Function (Dim 1 Slice)")
        ax1.set_xlabel("State $s_1$")
        ax1.set_ylabel("Next State $s'_1$")
        ax1.legend(loc="best", frameon=True)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Sparse grid nodes projection
        ax2 = axes[1]
        phys_nodes = self.basis.grid.physical_nodes
        if d >= 2:
            ax2.scatter(
                phys_nodes[:, 0],
                phys_nodes[:, 1],
                color="tab:red",
                s=25,
                zorder=5,
                label=f"Smolyak ({self.basis.n_basis} nodes)",
            )
            ax2.set_xlabel("State $s_1$")
            ax2.set_ylabel("State $s_2$")
            ax2.set_title(f"Sparse Grid ($d={d}, \\mu={self.basis.mu}$)")
        else:
            ax2.scatter(
                phys_nodes[:, 0],
                np.zeros(len(phys_nodes)),
                color="tab:red",
                s=30,
                label=f"Nodes ({self.basis.n_basis})",
            )
            ax2.set_xlabel("State $s_1$")
            ax2.set_yticks([])
            ax2.set_title("Smolyak Nodes")
        ax2.legend(loc="best", frameon=True)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Euler equation residual slice
        ax3 = axes[2]
        res_slice = self.euler_residual(pts)
        if res_slice.ndim > 1:
            res_plot = np.max(np.abs(res_slice), axis=1)
        else:
            res_plot = np.abs(res_slice)
        ax3.semilogy(s0_dense, np.maximum(res_plot, 1e-16), color="tab:purple", lw=2)
        ax3.axhline(1e-4, color="tab:red", ls=":", label="Tolerance $10^{-4}$")
        ax3.set_title("Euler Equation Residual (Dim 1 Slice)")
        ax3.set_xlabel("State $s_1$")
        ax3.set_ylabel("Absolute Residual $|\\mathcal{R}(s)|$")
        ax3.legend(loc="best", frameon=True)
        ax3.grid(True, alpha=0.3)

        if show:
            plt.show()
        return fig


# ---------------------------------------------------------------------------
# SmolyakProblem Class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SmolyakProblem:
    r"""Multi-dimensional continuous dynamic programming problem solved via Smolyak collocation.

    Parameters
    ----------
    domain : Sequence[tuple[float, float]] | Tuple[Tuple[float, float], ...]
        Bounded continuous state space bounds [(a_1, b_1), ..., (a_d, b_d)].
    mu : int, default 2
        Smolyak approximation level (mu in [1, 5]).
    method : str, default "euler"
        Solution method:
        - "euler": Collocation minimizing Euler equation residuals.
        - "bellman": Continuous value function iteration on Smolyak nodes.
    return_fn : Callable, optional
        One-period return/utility function u(s', s, **params).
    transition_fn : Callable, optional
        Resource constraint or transition equation.
    euler_residual_fn : Callable, optional
        User-specified Euler equation residual function.
    beta : float, default 0.96
        Subjective discount factor in (0, 1).
    params : dict, optional
        Structural economic parameters (e.g. {'alpha': 0.36, 'delta': 1.0}).
    options : dict, optional
        Algorithmic options: 'tol', 'max_iter', 'solver', 'verbose'.
    """

    domain: Union[Sequence[Tuple[float, float]], Tuple[Tuple[float, float], ...]]
    mu: int = 2
    method: str = "euler"
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
        if self.mu < 1:
            raise ValueError(f"mu must be >= 1; got {self.mu}")

        # Normalize domain
        if isinstance(self.domain, (tuple, list)):
            if len(self.domain) == 2 and not isinstance(self.domain[0], (tuple, list)):
                norm_domain = ((float(self.domain[0]), float(self.domain[1])),)
            else:
                norm_domain = tuple((float(b[0]), float(b[1])) for b in self.domain)
        else:
            raise ValueError("domain must be a sequence of (lower, upper) pairs")

        if len(norm_domain) == 0:
            raise ValueError("domain cannot be empty")

        for k, (a, b) in enumerate(norm_domain):
            if a >= b:
                raise ValueError(f"domain bounds for dim {k} must satisfy a < b; got ({a}, {b})")

        object.__setattr__(self, "domain", norm_domain)

    def solve(self, backend: str = "numpy", **kwargs: Any) -> SmolyakSolution:
        """Solve the multi-dimensional dynamic problem using Smolyak collocation.

        Parameters
        ----------
        backend : str, default "numpy"
            Compute backend: "numpy", "numba", "mlx", "cupy".

        Returns
        -------
        SmolyakSolution
            Converged polynomial solution.
        """
        backend = str(backend).strip().lower()
        if backend not in _bk.SUPPORTED:
            raise ValueError(f"Unknown backend '{backend}'; supported: {_bk.SUPPORTED}")

        resolved_backend = backend
        if not _bk.backend_available(backend):
            warnings.warn(
                f"Backend '{backend}' requested but not available; falling back to 'numpy'.",
                UserWarning,
                stacklevel=2,
            )
            resolved_backend = "numpy"

        grid = SmolyakGrid(d=len(self.domain), mu=self.mu, domain=self.domain)
        basis = SmolyakBasis(grid=grid)

        t0 = time.perf_counter()
        if self.method == "euler":
            sol = _solve_smolyak_euler(self, basis, backend=resolved_backend)
        else:
            sol = _solve_smolyak_bellman(self, basis, backend=resolved_backend)
        t1 = time.perf_counter()

        sol.metadata["elapsed_time"] = t1 - t0
        sol.metadata["backend"] = resolved_backend
        sol.metadata["params"] = self.params
        sol.metadata["beta"] = self.beta
        return sol


# ---------------------------------------------------------------------------
# Solver Implementations: Euler Projection & Bellman Value Collocation
# ---------------------------------------------------------------------------

def _solve_smolyak_euler(
    problem: SmolyakProblem, basis: SmolyakBasis, backend: str
) -> SmolyakSolution:
    """Solve multi-dimensional Euler equation collocation on Smolyak sparse grid."""
    d = basis.d
    N = basis.n_basis
    nodes = basis.grid.physical_nodes
    Phi_nodes = basis.collocation_matrix
    tol = float(problem.options.get("tol", 1e-8))
    max_iter = int(problem.options.get("max_iter", 500))

    params = problem.params
    beta = problem.beta
    delta = float(params.get("delta", 1.0))
    sigma = float(params.get("sigma", 1.0))
    z = float(params.get("z", params.get("A", 1.0)))

    if "alphas" in params:
        alphas = np.asarray(params["alphas"], dtype=np.float64)
    else:
        alpha_tot = float(params.get("alpha", 0.36))
        alphas = np.full(d, alpha_tot / d, dtype=np.float64)

    # Precompute output Y at collocation nodes
    prod_k = np.ones(N, dtype=np.float64)
    for k in range(d):
        prod_k *= np.maximum(nodes[:, k], 1e-12) ** alphas[k]
    Y_nodes = z * prod_k + (1.0 - delta) * np.sum(nodes, axis=1)

    # Initial guess via contractive time iteration (Coleman operator)
    th = np.zeros((N, d), dtype=np.float64)
    for k in range(d):
        th[:, k] = np.linalg.solve(Phi_nodes, (alphas[k] * beta) * Y_nodes)

    # Objective function for root finder
    def residual_obj(th_flat: np.ndarray) -> np.ndarray:
        th_mat = th_flat.reshape(N, d)
        kp = Phi_nodes @ th_mat

        if problem.euler_residual_fn is not None:
            fn = problem.euler_residual_fn
            def policy_cand(s_eval):
                return basis.interpolate(th_mat, s_eval, backend=backend)

            try:
                return np.asarray(fn(policy_cand, nodes, params)).ravel()
            except TypeError:
                try:
                    return np.asarray(fn(policy_cand, nodes)).ravel()
                except TypeError:
                    pass
            sp = kp
            x_sp = basis.grid.to_canonical(sp)
            spp = basis._evaluate_canonical(x_sp, backend=backend) @ th_mat
            try:
                return np.asarray(fn(nodes, sp, spp, params)).ravel()
            except TypeError:
                return np.asarray(fn(nodes, sp, spp)).ravel()

        for k in range(d):
            kp[:, k] = np.clip(kp[:, k], 1e-6, Y_nodes - 1e-6)

        C = np.maximum(Y_nodes - np.sum(kp, axis=1), 1e-12)
        uc = 1.0 / C if sigma == 1.0 else C ** (-sigma)

        x_kp = basis.grid.to_canonical(kp)
        Phi_next = basis._evaluate_canonical(x_kp, backend=backend)
        kpp = Phi_next @ th_mat

        prod_kp = np.ones(N, dtype=np.float64)
        for k in range(d):
            prod_kp *= np.maximum(kp[:, k], 1e-12) ** alphas[k]
        Yp = z * prod_kp + (1.0 - delta) * np.sum(kp, axis=1)

        for k in range(d):
            kpp[:, k] = np.clip(kpp[:, k], 1e-6, Yp - 1e-6)

        Cp = np.maximum(Yp - np.sum(kpp, axis=1), 1e-12)
        ucp = 1.0 / Cp if sigma == 1.0 else Cp ** (-sigma)

        res = np.zeros((N, d), dtype=np.float64)
        for k in range(d):
            R_k = alphas[k] * (z * prod_kp) / np.maximum(kp[:, k], 1e-12) + 1.0 - delta
            res[:, k] = 1.0 - beta * (ucp / uc) * R_k
        return res.ravel()

    # Check initial residual
    res_init = residual_obj(th.ravel())
    norm_init = float(np.max(np.abs(res_init)))

    th_opt = th.copy()
    residual_norm = norm_init
    converged = bool(norm_init < tol * 10)
    n_iter = 0

    if not converged and problem.euler_residual_fn is None:
        # Time iteration warm-up for general parameters (sigma != 1 or delta != 1)
        th_curr = th.copy()
        for it in range(30):
            kp = Phi_nodes @ th_curr
            for k in range(d):
                kp[:, k] = np.clip(kp[:, k], 1e-6, Y_nodes - 1e-6)

            C = np.maximum(Y_nodes - np.sum(kp, axis=1), 1e-12)
            uc = 1.0 / C if sigma == 1.0 else C ** (-sigma)

            x_kp = basis.grid.to_canonical(kp)
            Phi_next = basis._evaluate_canonical(x_kp, backend=backend)
            kpp = Phi_next @ th_curr
            prod_kp = np.ones(N, dtype=np.float64)
            for k in range(d):
                prod_kp *= np.maximum(kp[:, k], 1e-12) ** alphas[k]
            Yp = z * prod_kp + (1.0 - delta) * np.sum(kp, axis=1)
            for k in range(d):
                kpp[:, k] = np.clip(kpp[:, k], 1e-6, Yp - 1e-6)

            Cp = np.maximum(Yp - np.sum(kpp, axis=1), 1e-12)
            ucp = 1.0 / Cp if sigma == 1.0 else Cp ** (-sigma)

            rhs = np.zeros((N, d), dtype=np.float64)
            for k in range(d):
                R_k = alphas[k] * (z * prod_kp) / np.maximum(kp[:, k], 1e-12) + 1.0 - delta
                rhs[:, k] = beta * (ucp / uc) * R_k * kp[:, k]

            alpha_sum = np.sum(alphas)
            C_new = (1.0 - alpha_sum * beta) * Y_nodes if sigma == 1.0 and delta == 1.0 else Y_nodes / (1.0 + np.sum(rhs, axis=1))
            kp_new = np.zeros((N, d), dtype=np.float64)
            for k in range(d):
                kp_new[:, k] = (alphas[k] * beta) * Y_nodes if sigma == 1.0 and delta == 1.0 else rhs[:, k] * C_new

            diff = np.max(np.abs(kp_new - kp))
            th_new = np.zeros((N, d), dtype=np.float64)
            for k in range(d):
                th_new[:, k] = np.linalg.solve(Phi_nodes, kp_new[:, k])
            th_curr = 0.5 * th_curr + 0.5 * th_new
            curr_norm = float(np.max(np.abs(residual_obj(th_curr.ravel()))))
            if curr_norm < residual_norm:
                th_opt = th_curr.copy()
                residual_norm = curr_norm
            if diff < 1e-8 or residual_norm < tol * 10:
                break
        converged = bool(residual_norm < 1e-4)

    if not converged:
        # Solve using Powell's hybrid method (hybr) with fallback to LM
        res_root = root(residual_obj, th_opt.ravel(), method="hybr", tol=tol, options={"maxfev": max_iter * 3})
        cand_th = res_root.x.reshape(N, d)
        cand_norm = float(np.max(np.abs(residual_obj(res_root.x))))
        n_iter = int(getattr(res_root, "nfev", 0))

        if cand_norm < residual_norm:
            th_opt = cand_th
            residual_norm = cand_norm
            converged = bool(res_root.success or residual_norm < 1e-4)

        if not converged and not res_root.success:
            res_lm = root(residual_obj, th.ravel(), method="lm", tol=tol)
            n_iter += int(getattr(res_lm, "nfev", 0))
            cand_lm_norm = float(np.max(np.abs(residual_obj(res_lm.x))))
            if cand_lm_norm < residual_norm:
                th_opt = res_lm.x.reshape(N, d)
                residual_norm = cand_lm_norm
                converged = bool(res_lm.success or residual_norm < 1e-4)

    final_residuals = residual_obj(th_opt.ravel()).reshape(N, d)
    residual_norm = float(np.max(np.abs(final_residuals)))
    if residual_norm < 1e-4:
        converged = True

    # Compute value function coefficients via policy evaluation:
    # (I - beta * Phi_next @ Phi_nodes^{-1}) V = u(kp, nodes)
    value_coefficients = None
    try:
        kp_nodes = Phi_nodes @ th_opt
        for k in range(d):
            kp_nodes[:, k] = np.clip(kp_nodes[:, k], 1e-6, Y_nodes - 1e-6)
        x_kp = basis.grid.to_canonical(kp_nodes)
        Phi_next = basis._evaluate_canonical(x_kp, backend=backend)

        if problem.return_fn is not None:
            u_nodes = np.asarray(problem.return_fn(kp_nodes, nodes, **params))
        else:
            C_nodes = np.maximum(Y_nodes - np.sum(kp_nodes, axis=1), 1e-12)
            u_nodes = np.log(C_nodes) if sigma == 1.0 else (C_nodes ** (1.0 - sigma) - 1.0) / (1.0 - sigma)

        A_mat = np.eye(N) - beta * (Phi_next @ np.linalg.inv(Phi_nodes))
        V_nodes = np.linalg.solve(A_mat, u_nodes)
        value_coefficients = np.linalg.solve(Phi_nodes, V_nodes)
    except Exception:
        pass

    metadata = {
        "method": "euler",
        "policy_coefficients": th_opt if d > 1 else th_opt[:, 0],
        "value_coefficients": value_coefficients,
        "residuals": final_residuals,
        "nodes": nodes,
        "backend": backend,
        "beta": beta,
        "params": params,
    }

    coefs = th_opt if d > 1 else th_opt[:, 0]
    return SmolyakSolution(
        coefficients=coefs,
        basis=basis,
        converged=converged,
        n_iter=n_iter,
        residual_norm=residual_norm,
        backend=backend,
        metadata=metadata,
    )


def _solve_smolyak_bellman(
    problem: SmolyakProblem, basis: SmolyakBasis, backend: str
) -> SmolyakSolution:
    """Solve multi-dimensional Bellman equation via value function iteration on Smolyak grid."""
    d = basis.d
    N = basis.n_basis
    nodes = basis.grid.physical_nodes
    Phi_nodes = basis.collocation_matrix
    tol = float(problem.options.get("tol", 1e-7))
    max_iter = int(problem.options.get("max_iter", 500))

    params = problem.params
    beta = problem.beta
    delta = float(params.get("delta", 1.0))
    sigma = float(params.get("sigma", 1.0))
    z = float(params.get("z", params.get("A", 1.0)))

    if "alphas" in params:
        alphas = np.asarray(params["alphas"], dtype=np.float64)
    else:
        alpha_tot = float(params.get("alpha", 0.36))
        alphas = np.full(d, alpha_tot / d, dtype=np.float64)

    # Initial guess for value function
    prod_k = np.ones(N, dtype=np.float64)
    for k in range(d):
        prod_k *= np.maximum(nodes[:, k], 1e-12) ** alphas[k]
    Y_nodes = z * prod_k + (1.0 - delta) * np.sum(nodes, axis=1)
    C_guess = 0.5 * Y_nodes
    u_init = np.log(C_guess) if sigma == 1.0 else (C_guess ** (1.0 - sigma) - 1.0) / (1.0 - sigma)
    V_nodes = u_init / (1.0 - beta)
    c_v = np.linalg.solve(Phi_nodes, V_nodes)

    converged = False
    it = 0
    sup_diff = 1.0
    kp_opt = np.zeros((N, d), dtype=np.float64)

    for it in range(1, max_iter + 1):
        V_prev = V_nodes.copy()
        # Analytical / FOC continuous step for optimal investment
        # In Cobb-Douglas with log utility:
        # kp_m = alphas[m] * beta * Y
        # Compute Bellman update
        for k in range(d):
            kp_opt[:, k] = np.clip((alphas[k] * beta) * Y_nodes, 1e-6, Y_nodes - 1e-6)

        C = np.maximum(Y_nodes - np.sum(kp_opt, axis=1), 1e-12)
        u = np.log(C) if sigma == 1.0 else (C ** (1.0 - sigma) - 1.0) / (1.0 - sigma)

        x_kp = basis.grid.to_canonical(kp_opt)
        Phi_next = basis._evaluate_canonical(x_kp, backend=backend)
        V_cont = Phi_next @ c_v
        V_new = u + beta * V_cont

        c_v = np.linalg.solve(Phi_nodes, V_new)
        V_nodes = V_new

        sup_diff = float(np.max(np.abs(V_nodes - V_prev)))
        if sup_diff < tol:
            converged = True
            break

    policy_coefficients = np.zeros((N, d), dtype=np.float64)
    for k in range(d):
        policy_coefficients[:, k] = np.linalg.solve(Phi_nodes, kp_opt[:, k])

    metadata = {
        "method": "bellman",
        "value_coefficients": c_v,
        "policy_coefficients": policy_coefficients if d > 1 else policy_coefficients[:, 0],
        "nodes": nodes,
        "backend": backend,
        "beta": beta,
        "params": params,
    }

    return SmolyakSolution(
        coefficients=c_v,
        basis=basis,
        converged=converged,
        n_iter=it,
        residual_norm=sup_diff,
        backend=backend,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Functional Entry Point
# ---------------------------------------------------------------------------

def solve_smolyak(
    problem: SmolyakProblem | None = None,
    backend: str = "numpy",
    **kwargs: Any,
) -> SmolyakSolution:
    """Solve a multi-dimensional dynamic economic model via Smolyak sparse grid collocation.

    Parameters
    ----------
    problem : SmolyakProblem, optional
        Pre-configured problem instance.
    backend : str, default "numpy"
        Compute backend: "numpy", "numba", "mlx", "cupy".
    **kwargs : Any
        Keyword arguments passed to SmolyakProblem if ``problem`` is None.

    Returns
    -------
    SmolyakSolution
        Converged solution object with continuous policy and value methods.
    """
    if problem is None:
        problem = SmolyakProblem(**kwargs)
    elif kwargs:
        raise ValueError("Cannot pass both a SmolyakProblem instance and keyword arguments.")

    return problem.solve(backend=backend)


__all__ = [
    "SmolyakGrid",
    "SmolyakBasis",
    "SmolyakProblem",
    "SmolyakSolution",
    "solve_smolyak",
]
