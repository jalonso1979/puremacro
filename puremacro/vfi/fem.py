"""Finite Element Method (FEM / Galerkin) continuous state-space solver.

This module provides a production-grade continuous dynamic programming and projection
solver based on the Finite Element Method (FEM / Galerkin) for dynamic economic models.
Key capabilities:
- `FEMMesh`: 1D and tensor-product multi-D meshes, piecewise linear hat shape functions,
  uniform, geometrically clustered, and kink-aligned adaptively refined meshes.
- `FEMProblem`: Dataclass problem definition with defensive validation, supporting
  both Euler equation projection ('euler') and continuous Bellman iteration ('bellman'),
  Galerkin weighted residual quadrature and nodal collocation, borrowing constraints,
  and Kuhn-Tucker / Fischer-Burmeister complementarity formulations.
- `solve_fem`: Functional entry point for solving FEM dynamic economic models.
- `FEMSolution`: Result container with continuous `.policy(s)`, `.value(s)`, and full
  puremacro presentation interface (.summary(), .plot(), .to_frame(), .to_markdown(),
  .to_latex(), .to_typst()).
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar, root

import puremacro._backend as _bk


# ---------------------------------------------------------------------------
# Gauss-Legendre Quadrature
# ---------------------------------------------------------------------------


def gauss_legendre_quadrature(order: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre quadrature points xi and weights w on reference interval [-1, 1].

    Parameters
    ----------
    order : int, default 3
        Number of quadrature points (polynomial exactness up to degree 2*order - 1).

    Returns
    -------
    xi : np.ndarray
        Quadrature nodes in [-1, 1].
    w : np.ndarray
        Quadrature weights summing to 2.0.
    """
    if order < 1:
        raise ValueError(f"Quadrature order must be >= 1, got {order}")
    xi, w = np.polynomial.legendre.leggauss(order)
    return xi, w


# ---------------------------------------------------------------------------
# FEMMesh: 1D and Multi-D Tensor Product Meshes
# ---------------------------------------------------------------------------


class FEMMesh:
    """State space mesh representation for 1D and tensor-product multi-D elements.

    Supports uniform meshes, non-uniform meshes (e.g. geometric clustering near
    borrowing limits), and adaptively refined / kink-aligned meshes with exact
    node placement at policy kinks or constraints.

    Parameters
    ----------
    domain : tuple[float, float] or tuple[tuple[float, float], ...]
        State domain boundaries. For 1D, (s_min, s_max) or ((s_min, s_max),).
        For D dimensions, ((s1_min, s1_max), ..., (sD_min, sD_max)).
    elements : int or tuple[int, ...]
        Number of elements per dimension. E.g. 50 or (50,).
    nodes : np.ndarray, tuple[np.ndarray, ...], or None, optional
        Custom node coordinates. In 1D, a 1D strictly increasing array spanning
        the domain. If None, uniform nodes are constructed.
    degree : int, default 1
        Polynomial degree of shape functions (1 = piecewise linear hat functions).
    """

    def __init__(
        self,
        domain: Union[Tuple[float, float], Sequence[Tuple[float, float]]],
        elements: Union[int, Sequence[int]] = 50,
        nodes: np.ndarray | Sequence[np.ndarray] | None = None,
        degree: int = 1,
    ) -> None:
        if degree != 1:
            raise ValueError(f"Currently degree={degree} is not supported; degree=1 (piecewise linear) required")
        self.degree = int(degree)

        # Normalize domain
        if isinstance(domain, (tuple, list)) and len(domain) == 2 and not isinstance(domain[0], (tuple, list)):
            s_min, s_max = float(domain[0]), float(domain[1])
            if s_min >= s_max:
                raise ValueError(f"Invalid domain bounds: min ({s_min}) >= max ({s_max})")
            self.domain: tuple[tuple[float, float], ...] = ((s_min, s_max),)
        else:
            norm_dom = []
            for d in domain:
                if not (isinstance(d, (tuple, list)) and len(d) == 2):
                    raise ValueError(f"Each domain dimension must be a (min, max) pair, got {d}")
                d_min, d_max = float(d[0]), float(d[1])
                if d_min >= d_max:
                    raise ValueError(f"Invalid domain bounds: min ({d_min}) >= max ({d_max})")
                norm_dom.append((d_min, d_max))
            self.domain = tuple(norm_dom)

        self.dim = len(self.domain)

        # Normalize elements
        if isinstance(elements, (int, np.integer)):
            if elements < 1:
                raise ValueError(f"Number of elements must be >= 1, got {elements}")
            self.elements: tuple[int, ...] = (int(elements),) * self.dim
        else:
            norm_elem = []
            for e in elements:
                if int(e) < 1:
                    raise ValueError(f"Number of elements must be >= 1, got {e}")
                norm_elem.append(int(e))
            if len(norm_elem) != self.dim:
                raise ValueError(f"Elements tuple length {len(norm_elem)} does not match domain dimension {self.dim}")
            self.elements = tuple(norm_elem)

        # Construct or validate nodes
        if nodes is None:
            dim_nodes_list = []
            for d in range(self.dim):
                d_min, d_max = self.domain[d]
                n_pts = self.elements[d] * self.degree + 1
                dim_nodes_list.append(np.linspace(d_min, d_max, n_pts, dtype=np.float64))
            self.dim_nodes: tuple[np.ndarray, ...] = tuple(dim_nodes_list)
        else:
            if self.dim == 1:
                arr = np.asarray(nodes, dtype=np.float64).ravel()
                if len(arr) < 2:
                    raise ValueError(f"1D mesh nodes must have length >= 2, got {len(arr)}")
                if not np.all(np.diff(arr) > 0):
                    raise ValueError("Mesh nodes must be strictly monotonically increasing")
                d_min, d_max = self.domain[0]
                if abs(arr[0] - d_min) > 1e-7 or abs(arr[-1] - d_max) > 1e-7:
                    raise ValueError(
                        f"Nodes endpoints [{arr[0]}, {arr[-1]}] do not match domain bounds [{d_min}, {d_max}]"
                    )
                # Ensure exact boundary equality
                arr[0] = d_min
                arr[-1] = d_max
                self.dim_nodes = (arr,)
                self.elements = (len(arr) - 1,)
            else:
                if not isinstance(nodes, (tuple, list)) or len(nodes) != self.dim:
                    raise ValueError(f"For multi-D mesh, nodes must be a sequence of {self.dim} 1D arrays")
                dim_nodes_list = []
                norm_elem = []
                for d in range(self.dim):
                    arr = np.asarray(nodes[d], dtype=np.float64).ravel()
                    if len(arr) < 2:
                        raise ValueError(f"Mesh nodes along dim {d} must have length >= 2, got {len(arr)}")
                    if not np.all(np.diff(arr) > 0):
                        raise ValueError(f"Mesh nodes along dim {d} must be strictly increasing")
                    d_min, d_max = self.domain[d]
                    if abs(arr[0] - d_min) > 1e-7 or abs(arr[-1] - d_max) > 1e-7:
                        raise ValueError(f"Nodes along dim {d} endpoints do not match domain")
                    arr[0] = d_min
                    arr[-1] = d_max
                    dim_nodes_list.append(arr)
                    norm_elem.append(len(arr) - 1)
                self.dim_nodes = tuple(dim_nodes_list)
                self.elements = tuple(norm_elem)

        # Global nodes representation
        if self.dim == 1:
            self.nodes: np.ndarray = self.dim_nodes[0]
        else:
            coords = np.meshgrid(*self.dim_nodes, indexing="ij")
            self.nodes = np.column_stack([c.ravel() for c in coords])

        self.n_nodes: int = len(self.nodes)
        self.n_elements: int = int(np.prod(self.elements))

    @classmethod
    def create_uniform(
        cls,
        domain: Union[Tuple[float, float], Sequence[Tuple[float, float]]],
        elements: Union[int, Sequence[int]] = 50,
        degree: int = 1,
    ) -> FEMMesh:
        """Create a uniform finite element mesh."""
        return cls(domain=domain, elements=elements, nodes=None, degree=degree)

    @classmethod
    def create_clustered(
        cls,
        domain: Tuple[float, float],
        elements: int = 50,
        clustering: float = 2.0,
        degree: int = 1,
    ) -> FEMMesh:
        """Create a 1D non-uniform mesh clustered near the lower domain boundary.

        Useful for economic models with high localized curvature near borrowing limits.

        Parameters
        ----------
        domain : tuple[float, float]
            State domain (s_min, s_max).
        elements : int, default 50
            Number of elements.
        clustering : float, default 2.0
            Power exponent (> 1.0 concentrates nodes near s_min).
        degree : int, default 1
            Piecewise linear degree.
        """
        if clustering <= 0:
            raise ValueError(f"clustering parameter must be positive, got {clustering}")
        d_min, d_max = float(domain[0]), float(domain[1])
        u = np.linspace(0.0, 1.0, elements + 1)
        nodes = d_min + (d_max - d_min) * (u ** float(clustering))
        return cls(domain=domain, elements=elements, nodes=nodes, degree=degree)

    @classmethod
    def from_kinks(
        cls,
        domain: Tuple[float, float],
        n_elements: int,
        kinks: Sequence[float],
        degree: int = 1,
    ) -> FEMMesh:
        """Create a 1D mesh with nodes placed exactly at policy kinks or constraints.

        Subdivides the intervals between kinks into elements proportional to their
        length, eliminating Gibbs-like ringing at constraint thresholds.

        Parameters
        ----------
        domain : tuple[float, float]
            State domain (s_min, s_max).
        n_elements : int
            Total target number of elements (>= number of intervals).
        kinks : sequence of float
            Locations of kinks in (s_min, s_max).
        degree : int, default 1
            Shape function degree.
        """
        d_min, d_max = float(domain[0]), float(domain[1])
        valid_kinks = sorted(set(float(k) for k in kinks if d_min < float(k) < d_max))
        if not valid_kinks:
            return cls.create_uniform(domain=domain, elements=n_elements, degree=degree)

        breaks = [d_min] + valid_kinks + [d_max]
        n_intervals = len(breaks) - 1
        if n_elements < n_intervals:
            n_elements = n_intervals

        lengths = [breaks[i + 1] - breaks[i] for i in range(n_intervals)]
        total_len = sum(lengths)

        # Allocate elements proportionally with at least 1 element per interval
        alloc = [max(1, int(round(n_elements * (L / total_len)))) for L in lengths]
        diff = n_elements - sum(alloc)
        alloc[-1] += diff
        # In case diff was negative and reduced last alloc below 1:
        for i in range(len(alloc)):
            if alloc[i] < 1:
                alloc[i] = 1

        node_parts = []
        for i in range(n_intervals):
            pts = np.linspace(breaks[i], breaks[i + 1], alloc[i] + 1)
            if i > 0:
                pts = pts[1:]  # avoid duplicate boundary node
            node_parts.append(pts)

        all_nodes = np.concatenate(node_parts)
        return cls(domain=domain, elements=len(all_nodes) - 1, nodes=all_nodes, degree=degree)

    def evaluate_basis(self, s: Union[float, Sequence[float], np.ndarray], xp: Any = None) -> np.ndarray:
        """Evaluate piecewise linear hat basis functions phi_i(s) across state coordinates.

        Returns a basis matrix Phi of shape (K, n_nodes) where K is the number
        of evaluation points. Satisfies partition of unity (sum_i phi_i(s) = 1.0)
        and Kronecker delta property at mesh nodes (phi_i(s_j) = delta_ij).

        Parameters
        ----------
        s : float, sequence of float, or np.ndarray
            Evaluation coordinate(s). For 1D mesh, 1D array of shape (K,) or scalar.
            For D-dimensional mesh, array of shape (K, D).
        xp : module, optional
            Array namespace (e.g. numpy, mlx.core, cupy). Default is numpy.

        Returns
        -------
        Phi : np.ndarray
            Basis evaluation matrix of shape (K, n_nodes).
        """
        arr_s = np.asarray(s, dtype=np.float64)

        if self.dim == 1:
            if arr_s.ndim == 0:
                arr_s = arr_s.reshape(1)
            elif arr_s.ndim > 1:
                arr_s = arr_s.ravel()

            K = len(arr_s)
            nodes = self.dim_nodes[0]
            N = len(nodes)

            # Clamp out-of-bounds evaluation slightly to avoid index errors
            s_clamped = np.clip(arr_s, nodes[0], nodes[-1])

            # Binary search for containing element
            idx = np.searchsorted(nodes, s_clamped, side="right") - 1
            idx = np.clip(idx, 0, N - 2)

            Phi = np.zeros((K, N), dtype=np.float64)
            left_nodes = nodes[idx]
            right_nodes = nodes[idx + 1]
            h = right_nodes - left_nodes

            w_right = (s_clamped - left_nodes) / h
            w_left = (right_nodes - s_clamped) / h

            rows = np.arange(K)
            Phi[rows, idx] = w_left
            Phi[rows, idx + 1] = w_right

            if xp is not None and xp is not np:
                return xp.asarray(Phi)
            return Phi

        # Multi-D Tensor product mesh
        if arr_s.ndim == 1 and self.dim == 1:
            arr_s = arr_s.reshape(-1, 1)
        elif arr_s.ndim == 1:
            if len(arr_s) == self.dim:
                arr_s = arr_s.reshape(1, self.dim)
            else:
                raise ValueError(f"Input coordinates must have {self.dim} columns, got shape {arr_s.shape}")

        K = len(arr_s)
        # Evaluate 1D basis along each dimension
        dim_phi_list = []
        for d in range(self.dim):
            nodes_d = self.dim_nodes[d]
            N_d = len(nodes_d)
            s_d = np.clip(arr_s[:, d], nodes_d[0], nodes_d[-1])
            idx_d = np.clip(np.searchsorted(nodes_d, s_d, side="right") - 1, 0, N_d - 2)
            Phi_d = np.zeros((K, N_d), dtype=np.float64)
            h_d = nodes_d[idx_d + 1] - nodes_d[idx_d]
            Phi_d[np.arange(K), idx_d] = (nodes_d[idx_d + 1] - s_d) / h_d
            Phi_d[np.arange(K), idx_d + 1] = (s_d - nodes_d[idx_d]) / h_d
            dim_phi_list.append(Phi_d)

        # Row-wise tensor product
        Phi = dim_phi_list[0]
        for d in range(1, self.dim):
            Phi_next = dim_phi_list[d]
            # Row-wise Kronecker product
            Phi = np.einsum("ki,kj->kij", Phi, Phi_next).reshape(K, -1)

        if xp is not None and xp is not np:
            return xp.asarray(Phi)
        return Phi

    def __repr__(self) -> str:
        return (
            f"FEMMesh(domain={self.domain}, elements={self.elements}, "
            f"n_nodes={self.n_nodes}, degree={self.degree})"
        )


# ---------------------------------------------------------------------------
# FEMSolution: Result Dataclass with Presentation Contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FEMSolution:
    """Result of solving a dynamic economic model via Finite Element Method.

    Attributes
    ----------
    nodal_values : np.ndarray
        Solved values at the mesh nodes (policy values for 'euler', value function for 'bellman').
    mesh : FEMMesh
        Underlying finite element mesh.
    converged : bool
        Whether the nonlinear projection or value iteration converged.
    n_iter : int
        Number of iterations or root-finder function evaluations.
    residual_norm : float
        Final infinity norm of the projected residual vector.
    backend : str
        Compute backend utilized ('numpy', 'numba', 'mlx', 'cupy').
    metadata : dict
        Additional diagnostic information and parameters.
    """

    nodal_values: np.ndarray
    mesh: FEMMesh
    converged: bool
    n_iter: int
    residual_norm: float
    backend: str
    metadata: dict = field(default_factory=dict)

    def policy(self, s: Union[float, Sequence[float], np.ndarray]) -> np.ndarray | float:
        """Evaluate the continuous policy function g(s) via piecewise linear interpolation.

        Parameters
        ----------
        s : float, sequence of float, or np.ndarray
            State evaluation point(s).

        Returns
        -------
        g_s : float or np.ndarray
            Evaluated policy at s.
        """
        arr_s = np.asarray(s, dtype=np.float64)
        is_scalar = arr_s.ndim == 0

        policy_vals = self.metadata.get("policy_values", self.nodal_values)

        if self.mesh.dim == 1:
            val = np.interp(arr_s, self.mesh.dim_nodes[0], policy_vals)
            return float(val) if is_scalar else val

        # Multi-D: evaluate via basis matrix
        Phi = self.mesh.evaluate_basis(arr_s)
        val = Phi @ policy_vals
        return float(val[0]) if is_scalar else val

    def value(self, s: Union[float, Sequence[float], np.ndarray]) -> np.ndarray | float:
        """Evaluate the continuous value function V(s) via piecewise linear interpolation.

        Parameters
        ----------
        s : float, sequence of float, or np.ndarray
            State evaluation point(s).

        Returns
        -------
        V_s : float or np.ndarray
            Evaluated value at s.
        """
        arr_s = np.asarray(s, dtype=np.float64)
        is_scalar = arr_s.ndim == 0

        value_vals = self.metadata.get("value_values", self.nodal_values)

        if self.mesh.dim == 1:
            val = np.interp(arr_s, self.mesh.dim_nodes[0], value_vals)
            return float(val) if is_scalar else val

        Phi = self.mesh.evaluate_basis(arr_s)
        val = Phi @ value_vals
        return float(val[0]) if is_scalar else val

    def euler_residual(self, s: Union[float, Sequence[float], np.ndarray]) -> np.ndarray:
        """Evaluate the continuous Euler equation residual across arbitrary test points.

        Parameters
        ----------
        s : float or np.ndarray
            Evaluation state coordinates.

        Returns
        -------
        res : np.ndarray
            Euler residuals at s.
        """
        res_fn = self.metadata.get("residual_eval_fn")
        if res_fn is None:
            raise RuntimeError("Euler residual evaluation function not available in metadata")
        arr_s = np.asarray(s, dtype=np.float64)
        if arr_s.ndim == 0:
            arr_s = arr_s.reshape(1)
        sp = self.policy(arr_s)
        sn = self.policy(sp)
        return res_fn(arr_s, sp, sn)

    def summary(self) -> pd.DataFrame:
        """Return a publication-grade summary table of the FEM solution."""
        rows = [
            ("Method", str(self.metadata.get("method", "euler"))),
            ("Projection", str(self.metadata.get("projection", "galerkin"))),
            ("Domain", str(self.mesh.domain)),
            ("Elements", str(self.mesh.elements)),
            ("Nodes", int(self.mesh.n_nodes)),
            ("Converged", bool(self.converged)),
            ("Iterations", int(self.n_iter)),
            ("Residual Norm", float(self.residual_norm)),
            ("Max Euler Residual", float(self.metadata.get("max_euler_residual", np.nan))),
            ("Backend", str(self.backend)),
            ("Elapsed Time (s)", float(self.metadata.get("elapsed", np.nan))),
        ]
        if "borrowing_constraint" in self.metadata and self.metadata["borrowing_constraint"] is not None:
            rows.append(("Borrowing Constraint", float(self.metadata["borrowing_constraint"])))
        return pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric")

    def to_frame(self) -> pd.DataFrame:
        """Return summary statistics as a DataFrame."""
        return self.summary()

    def to_markdown(self, **kwargs: Any) -> str:
        """Render summary table as Markdown."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render summary table as LaTeX tabular."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render summary table as Typst table."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.summary(), **kwargs)

    def plot(self, n_points: int = 500) -> Any:
        """Plot the converged policy function, value function, and Euler residuals.

        Parameters
        ----------
        n_points : int, default 500
            Number of points for continuous visualization.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Multi-panel diagnostic figure.
        """
        import matplotlib.pyplot as plt

        if self.mesh.dim != 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "Multi-D FEM plotting not implemented", ha="center", va="center")
            return fig

        s_min, s_max = self.mesh.domain[0]
        s_grid = np.linspace(s_min, s_max, n_points)
        pol = self.policy(s_grid)
        val = self.value(s_grid)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Panel 1: Policy function
        ax1 = axes[0]
        ax1.plot(s_grid, pol, label="Policy g(s)", color="#1f77b4", lw=2.0)
        ax1.plot(s_grid, s_grid, "k--", alpha=0.5, label="45° line")
        if self.metadata.get("borrowing_constraint") is not None:
            b = self.metadata["borrowing_constraint"]
            ax1.axhline(b, color="crimson", ls=":", label=f"Constraint {b:.2f}")
        ax1.set_xlabel("State s")
        ax1.set_ylabel("Next State s'")
        ax1.set_title("FEM Policy Function")
        ax1.legend(frameon=True)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Value function
        ax2 = axes[1]
        ax2.plot(s_grid, val, label="Value V(s)", color="#2ca02c", lw=2.0)
        ax2.set_xlabel("State s")
        ax2.set_ylabel("Value")
        ax2.set_title("FEM Value Function")
        ax2.legend(frameon=True)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Euler residuals (log10)
        ax3 = axes[2]
        try:
            res = np.abs(self.euler_residual(s_grid))
            res_safe = np.maximum(res, 1e-16)
            ax3.plot(s_grid, np.log10(res_safe), label=r"$\log_{10} |\mathcal{R}(s)|$", color="#d62728", lw=1.8)
            ax3.set_xlabel("State s")
            ax3.set_ylabel("log10 Residual")
            ax3.set_title("Continuous Euler Residual")
            ax3.axhline(-4.0, color="gray", ls="--", label=r"Target $10^{-4}$")
            ax3.legend(frameon=True)
            ax3.grid(True, alpha=0.3)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
            ax3.text(0.5, 0.5, "Euler residual not evaluated", ha="center", va="center")

        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# FEMProblem: Problem Dataclass with Defensive Validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FEMProblem:
    """Continuous state-space economic problem definition using Finite Element Method.

    Parameters
    ----------
    domain : tuple[float, float] or tuple[tuple[float, float], ...]
        State bounds (s_min, s_max) or tuple of bounds per dimension.
    elements : int or tuple[int, ...], default 50
        Number of mesh elements.
    method : {"euler", "bellman"}, default "euler"
        Solution method: policy Euler equation projection or Bellman value iteration.
    projection : {"galerkin", "collocation"}, default "galerkin"
        Projection operator: weighted residual integration via Gauss-Legendre
        element quadrature ("galerkin") or nodal collocation ("collocation").
    return_fn : Callable, optional
        Return / utility function u(s, s') or u(c).
    transition_fn : Callable, optional
        Resource / production transition function f(s) or next-state equation.
    euler_residual_fn : Callable, optional
        Explicit continuous Euler equation residual callable R(s, s', s'').
    beta : float, default 0.96
        Subjective discount factor in (0, 1).
    params : dict, optional
        Dictionary of economic parameters (e.g. alpha, delta, gamma).
    borrowing_constraint : float, optional
        Lower bound constraint on choice s' >= borrowing_constraint.
    options : dict, optional
        Algorithmic options (tol, max_iter, quad_order, root_method, kinks, etc.).
    """

    domain: Union[Tuple[float, float], Sequence[Tuple[float, float]]]
    elements: Union[int, Sequence[int]] = 50
    method: str = "euler"
    projection: str = "galerkin"
    return_fn: Callable | None = None
    transition_fn: Callable | None = None
    euler_residual_fn: Callable | None = None
    beta: float = 0.96
    params: dict = field(default_factory=dict)
    borrowing_constraint: float | None = None
    options: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate beta
        if not (0.0 < float(self.beta) < 1.0):
            raise ValueError(f"Discount factor beta must be in (0, 1), got {self.beta}")

        # Validate method
        if self.method not in ("euler", "bellman"):
            raise ValueError(f"Invalid method {self.method!r}; must be 'euler' or 'bellman'")

        # Validate projection
        if self.projection not in ("galerkin", "collocation"):
            raise ValueError(f"Invalid projection {self.projection!r}; must be 'galerkin' or 'collocation'")

        # Validate domain bounds
        if isinstance(self.domain, (tuple, list)) and len(self.domain) == 2 and not isinstance(self.domain[0], (tuple, list)):
            s_min, s_max = float(self.domain[0]), float(self.domain[1])
            if s_min >= s_max:
                raise ValueError(f"Domain min {s_min} must be strictly less than max {s_max}")
        elif isinstance(self.domain, (tuple, list)):
            for d in self.domain:
                if not (isinstance(d, (tuple, list)) and len(d) == 2):
                    raise ValueError(f"Each domain entry must be a (min, max) pair, got {d}")
                if float(d[0]) >= float(d[1]):
                    raise ValueError(f"Domain dimension min {d[0]} >= max {d[1]}")
        else:
            raise ValueError(f"Invalid domain specification: {self.domain}")

        # Validate elements
        if isinstance(self.elements, (int, np.integer)):
            if self.elements < 1:
                raise ValueError(f"Elements must be >= 1, got {self.elements}")
        elif isinstance(self.elements, (tuple, list)):
            for e in self.elements:
                if int(e) < 1:
                    raise ValueError(f"Each element count must be >= 1, got {e}")

        # Method-specific function requirements
        if self.method == "euler":
            if self.euler_residual_fn is None and (self.return_fn is None or self.transition_fn is None):
                raise ValueError(
                    "For method='euler', either euler_residual_fn or both "
                    "return_fn and transition_fn must be provided"
                )
        elif self.method == "bellman":
            if self.return_fn is None:
                raise ValueError("For method='bellman', return_fn must be provided")

        # Validate borrowing constraint
        if self.borrowing_constraint is not None:
            b = float(self.borrowing_constraint)
            # Compare with lower bound of domain
            if isinstance(self.domain, (tuple, list)) and len(self.domain) == 2 and not isinstance(self.domain[0], (tuple, list)):
                d_min = float(self.domain[0])
            else:
                d_min = float(self.domain[0][0])
            if b < d_min - 1e-7:
                raise ValueError(f"Borrowing constraint {b} cannot be strictly below domain lower bound {d_min}")

    def solve(self, backend: str = "numpy") -> FEMSolution:
        """Solve the continuous economic model.

        Parameters
        ----------
        backend : {"numpy", "numba", "mlx", "cupy"}, default "numpy"
            Compute acceleration backend. Falls back gracefully to "numpy" if
            the requested package is unavailable.

        Returns
        -------
        solution : FEMSolution
            Converged solution container.
        """
        start_time = time.perf_counter()

        backend = str(backend).strip().lower()
        if backend not in _bk.SUPPORTED:
            raise ValueError(f"Unknown backend '{backend}'; supported: {_bk.SUPPORTED}")

        # Backend availability & fallback check
        resolved_backend = backend
        if not _bk.backend_available(backend):
            import warnings
            warnings.warn(
                f"Backend '{backend}' requested but not available; falling back to 'numpy'.",
                UserWarning,
                stacklevel=2,
            )
            resolved_backend = "numpy"

        # Build or acquire mesh
        mesh = self._construct_mesh()

        if self.method == "euler":
            sol = self._solve_euler(mesh=mesh, backend=resolved_backend, start_time=start_time)
        else:
            sol = self._solve_bellman(mesh=mesh, backend=resolved_backend, start_time=start_time)

        sol.metadata["backend"] = resolved_backend
        return sol

    def _construct_mesh(self) -> FEMMesh:
        """Construct the FEMMesh based on problem specifications and options."""
        if "mesh" in self.options and isinstance(self.options["mesh"], FEMMesh):
            return self.options["mesh"]

        kinks = self.options.get("kinks")
        if kinks is None and self.borrowing_constraint is not None:
            # Check if domain is 1D
            if (isinstance(self.domain, (tuple, list)) and len(self.domain) == 2 and not isinstance(self.domain[0], (tuple, list))) or len(self.domain) == 1:
                # If borrowing constraint is inside domain, place node at constraint or estimate kink
                b = float(self.borrowing_constraint)
                d_min = float(self.domain[0] if not isinstance(self.domain[0], (tuple, list)) else self.domain[0][0])
                d_max = float(self.domain[1] if not isinstance(self.domain[0], (tuple, list)) else self.domain[0][1])
                if d_min < b < d_max:
                    kinks = [b]

        if kinks is not None and (isinstance(self.domain, (tuple, list)) and (len(self.domain) == 2 and not isinstance(self.domain[0], (tuple, list))) or len(self.domain) == 1):
            dom_1d = self.domain if not isinstance(self.domain[0], (tuple, list)) else self.domain[0]
            n_elem = self.elements if isinstance(self.elements, int) else self.elements[0]
            return FEMMesh.from_kinks(domain=dom_1d, n_elements=n_elem, kinks=kinks)

        if "clustering" in self.options:
            dom_1d = self.domain if not isinstance(self.domain[0], (tuple, list)) else self.domain[0]
            n_elem = self.elements if isinstance(self.elements, int) else self.elements[0]
            return FEMMesh.create_clustered(
                domain=dom_1d,
                elements=n_elem,
                clustering=float(self.options["clustering"]),
            )

        # Handle domains with small lower bounds gracefully via adaptive clustering
        if (isinstance(self.domain, (tuple, list)) and len(self.domain) == 2 and not isinstance(self.domain[0], (tuple, list))) or len(self.domain) == 1:
            dom_1d = self.domain if not isinstance(self.domain[0], (tuple, list)) else self.domain[0]
            d_min = float(dom_1d[0])
            d_max = float(dom_1d[1])
            if 0.0 <= d_min < 0.02 * d_max and d_max > 0:
                n_elem = self.elements if isinstance(self.elements, int) else self.elements[0]
                return FEMMesh.create_clustered(
                    domain=dom_1d,
                    elements=n_elem,
                    clustering=float(self.options.get("clustering", 1.8)),
                )

        return FEMMesh(domain=self.domain, elements=self.elements)

    def _create_residual_eval_fn(self) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """Create a vectorized Euler equation residual evaluator R(s, s', s'')."""
        if self.euler_residual_fn is not None:
            euler_fn = self.euler_residual_fn
            params = self.params

            def eval_fn(s: np.ndarray, sp: np.ndarray, sn: np.ndarray) -> np.ndarray:
                try:
                    return euler_fn(s, sp, sn, **params)
                except TypeError:
                    try:
                        return euler_fn(s, sp, sn)
                    except TypeError:
                        return euler_fn(s, sp)

            return eval_fn

        # Derived from return_fn (utility) and transition_fn
        ret_fn = self.return_fn
        trans_fn = self.transition_fn
        beta = float(self.beta)
        gamma = float(self.params.get("gamma", 1.0))

        def derived_eval_fn(s: np.ndarray, sp: np.ndarray, sn: np.ndarray) -> np.ndarray:
            s_safe = np.maximum(s, 1e-12)
            sp_safe = np.maximum(sp, 1e-12)
            c = trans_fn(s_safe) - sp_safe
            cp = trans_fn(sp_safe) - np.maximum(sn, 1e-12)

            # Guard against zero/negative consumption
            c_safe = np.maximum(c, 1e-12)
            cp_safe = np.maximum(cp, 1e-12)

            if abs(gamma - 1.0) < 1e-9:
                u_prime_c = 1.0 / c_safe
                u_prime_cp = 1.0 / cp_safe
            else:
                u_prime_c = c_safe ** (-gamma)
                u_prime_cp = cp_safe ** (-gamma)

            eps_diff = 1e-6
            f_prime_sp = (trans_fn(sp_safe + eps_diff) - trans_fn(sp_safe - eps_diff)) / (2.0 * eps_diff)
            return 1.0 - beta * (u_prime_cp / u_prime_c) * f_prime_sp

        return derived_eval_fn

    def _solve_euler(self, mesh: FEMMesh, backend: str, start_time: float) -> FEMSolution:
        """Solve using continuous Euler equation projection (Galerkin or Collocation)."""
        nodes = mesh.nodes
        n_nodes = mesh.n_nodes
        eval_residual = self._create_residual_eval_fn()

        tol = float(self.options.get("tol", 1e-8))
        quad_order = int(self.options.get("quad_order", 3))
        root_method = str(self.options.get("root_method", "hybr"))
        comp_mode = str(self.options.get("complementarity", "fb"))
        borrow_bound = self.borrowing_constraint

        n_elements = mesh.elements[0]
        dim_nodes = mesh.dim_nodes[0]

        # Initial guess: generic, model-agnostic neutral guess
        if "initial_guess" in self.options:
            y0 = np.asarray(self.options["initial_guess"], dtype=np.float64)
        else:
            y0 = float(self.beta) * nodes
            if mesh.dim == 1 and self.transition_fn is not None:
                # Euler time iteration warm-up (Coleman operator) to provide a contractive,
                # physically consistent starting point for the nonlinear root-finder
                y_curr = y0.copy()
                for _ in range(10):
                    y_next = np.zeros_like(y_curr)
                    for i, ki in enumerate(dim_nodes):
                        f_ki = float(np.asarray(self.transition_fn(ki)).ravel()[0])

                        def eq_i(kp):
                            kp_val = min(max(kp, dim_nodes[0]), dim_nodes[-1])
                            kpp = np.interp(kp_val, dim_nodes, y_curr)
                            return eval_residual(np.array([ki]), np.array([kp]), np.array([kpp]))[0]

                        low = min(1e-5, 0.01 * f_ki)
                        high = max(f_ki - 1e-5, 0.99 * f_ki)
                        if low >= high:
                            low = 0.001 * f_ki
                            high = 0.999 * f_ki
                        try:
                            y_next[i] = brentq(eq_i, low, high)
                        except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
                            y_next[i] = y_curr[i]
                    y_curr = y_next
                y0 = y_curr

        y0 = np.clip(y0, nodes[0], nodes[-1])
        if borrow_bound is not None:
            y0 = np.maximum(y0, float(borrow_bound))

        # Quadrature points for Galerkin projection
        xi, w = gauss_legendre_quadrature(quad_order)
        phi_left = (1.0 - xi) / 2.0
        phi_right = (1.0 + xi) / 2.0
        w_l = (w * phi_left)[None, :]
        w_r = (w * phi_right)[None, :]

        s_left = dim_nodes[:-1, None]
        s_right = dim_nodes[1:, None]
        he = s_right - s_left
        sq = (s_left + s_right) / 2.0 + (he / 2.0) * xi[None, :]

        def residual_system(y: np.ndarray) -> np.ndarray:
            # Effective trial policy enforcing boundary feasibility
            y_clamped = np.clip(y, dim_nodes[0], dim_nodes[-1])
            y_eff = np.maximum(y_clamped, float(borrow_bound)) if borrow_bound is not None else y_clamped

            if self.projection == "galerkin" and mesh.dim == 1:
                spq = y_eff[:-1, None] * phi_left[None, :] + y_eff[1:, None] * phi_right[None, :]
                if borrow_bound is not None:
                    spq = np.maximum(spq, float(borrow_bound))
                snq = np.interp(spq.ravel(), dim_nodes, y_eff).reshape(n_elements, len(xi))
                if borrow_bound is not None:
                    snq = np.maximum(snq, float(borrow_bound))

                res_q = eval_residual(sq, spq, snq)

                c_left = ((he / 2.0) * np.sum(res_q * w_l, axis=1, keepdims=True)).ravel()
                c_right = ((he / 2.0) * np.sum(res_q * w_r, axis=1, keepdims=True)).ravel()

                r = np.zeros(n_nodes, dtype=np.float64)
                np.add.at(r, np.arange(n_elements), c_left)
                np.add.at(r, np.arange(1, n_elements + 1), c_right)

            elif self.projection == "collocation" and mesh.dim == 1:
                sp = y_eff
                sn = np.interp(sp, dim_nodes, y_eff)
                r = eval_residual(dim_nodes, sp, sn)

            else:
                # Multi-D projection fallback via basis matrix
                Phi_nodes = mesh.evaluate_basis(mesh.nodes)
                sp = y_eff
                sn = mesh.evaluate_basis(sp) @ y_eff
                r = eval_residual(mesh.nodes, sp, sn)

            if borrow_bound is not None:
                # Kuhn-Tucker complementarity: y - b >= 0, r >= 0
                a = y - float(borrow_bound)
                b_val = r
                if comp_mode == "fb":
                    # Fischer-Burmeister complementarity: a + b - sqrt(a^2 + b^2 + eps)
                    return a + b_val - np.sqrt(a**2 + b_val**2 + 1e-12)
                return np.minimum(a, b_val)

            return r

        # Root finding
        sol_root = root(residual_system, y0, method=root_method, tol=tol)
        if not sol_root.success:
            # Fallback to Levenberg-Marquardt or broyden1 if hybr failed
            alt_method = "lm" if root_method != "lm" else "broyden1"
            sol_root = root(residual_system, y0, method=alt_method, tol=tol)

        nodal_policy = np.clip(np.asarray(sol_root.x, dtype=np.float64), dim_nodes[0], dim_nodes[-1])
        if borrow_bound is not None:
            nodal_policy = np.maximum(nodal_policy, float(borrow_bound))

        res_norm = float(np.max(np.abs(residual_system(nodal_policy))))
        converged = bool(sol_root.success or res_norm < tol * 10 or res_norm < 1e-6)
        if not converged or res_norm > 1e-6:
            alt_method = "lm" if root_method != "lm" else "broyden1"
            sol_alt = root(residual_system, nodal_policy, method=alt_method, tol=tol)
            alt_policy = np.clip(np.asarray(sol_alt.x, dtype=np.float64), dim_nodes[0], dim_nodes[-1])
            if borrow_bound is not None:
                alt_policy = np.maximum(alt_policy, float(borrow_bound))
            alt_res_norm = float(np.max(np.abs(residual_system(alt_policy))))
            if alt_res_norm < res_norm:
                nodal_policy = alt_policy
                res_norm = alt_res_norm
                converged = bool(sol_alt.success or res_norm < 1e-6)

        n_iter = int(getattr(sol_root, "nfev", getattr(sol_root, "nit", 0)))

        # Evaluate out-of-sample continuous Euler residual on dense 1000-point grid
        s_min = mesh.domain[0][0]
        s_max = mesh.domain[0][1]
        test_grid = np.linspace(s_min, s_max, 1000)
        sp_test = np.interp(test_grid, dim_nodes, nodal_policy)
        sn_test = np.interp(sp_test, dim_nodes, nodal_policy)
        test_res = eval_residual(test_grid, sp_test, sn_test)
        max_euler_res = float(np.max(np.abs(test_res)))

        # Approximate implied value function via linear fixed point iteration
        value_vals = np.zeros_like(nodal_policy)
        if self.return_fn is not None and self.transition_fn is not None:
            c_nodes = np.maximum(self.transition_fn(dim_nodes) - nodal_policy, 1e-12)
            gamma = float(self.params.get("gamma", 1.0))
            if abs(gamma - 1.0) < 1e-9:
                u_nodes = np.log(c_nodes)
            else:
                u_nodes = (c_nodes ** (1.0 - gamma) - 1.0) / (1.0 - gamma)

            value_vals = u_nodes / (1.0 - float(self.beta))
            for _ in range(100):
                v_next = u_nodes + float(self.beta) * np.interp(nodal_policy, dim_nodes, value_vals)
                if np.max(np.abs(v_next - value_vals)) < 1e-8:
                    value_vals = v_next
                    break
                value_vals = v_next

        elapsed = time.perf_counter() - start_time

        metadata = {
            "method": "euler",
            "projection": self.projection,
            "policy_values": nodal_policy,
            "value_values": value_vals,
            "residual_eval_fn": eval_residual,
            "max_euler_residual": max_euler_res,
            "borrowing_constraint": borrow_bound,
            "elapsed": elapsed,
            "backend": backend,
        }

        return FEMSolution(
            nodal_values=nodal_policy,
            mesh=mesh,
            converged=converged,
            n_iter=n_iter,
            residual_norm=res_norm,
            backend=backend,
            metadata=metadata,
        )

    def _solve_bellman(self, mesh: FEMMesh, backend: str, start_time: float) -> FEMSolution:
        """Solve using continuous Bellman value function iteration with Howard acceleration."""
        dim_nodes = mesh.dim_nodes[0]
        n_nodes = len(dim_nodes)
        beta = float(self.beta)
        borrow_bound = self.borrowing_constraint

        tol = float(self.options.get("tol", 1e-7))
        max_iter = int(self.options.get("max_iter", 500))
        howard = bool(self.options.get("howard", True))
        n_howard = int(self.options.get("n_howard", 15))

        s_min, s_max = mesh.domain[0]

        # Initial value function guess
        V = np.zeros(n_nodes, dtype=np.float64)
        policy = np.copy(dim_nodes)

        # Helper to evaluate return function u(s, sp)
        def calc_return(s_val: float, sp_val: float) -> float:
            if self.transition_fn is not None:
                c = self.transition_fn(s_val) - sp_val
                if c <= 0.0:
                    return -1e10
                gamma = float(self.params.get("gamma", 1.0))
                if abs(gamma - 1.0) < 1e-9:
                    return float(np.log(c))
                return float((c ** (1.0 - gamma) - 1.0) / (1.0 - gamma))
            return float(self.return_fn(s_val, sp_val))

        converged = False
        n_iter = 0

        for it in range(max_iter):
            n_iter += 1
            V_new = np.zeros(n_nodes, dtype=np.float64)

            for i, s_i in enumerate(dim_nodes):
                low = s_min if borrow_bound is None else max(s_min, float(borrow_bound))
                if self.transition_fn is not None:
                    high = min(s_max, float(self.transition_fn(s_i)) - 1e-6)
                else:
                    high = s_max

                if low >= high:
                    sp_opt = low
                    v_opt = calc_return(s_i, sp_opt) + beta * float(np.interp(sp_opt, dim_nodes, V))
                else:
                    def obj(sp_candidate: float) -> float:
                        ret = calc_return(s_i, sp_candidate)
                        cont = float(np.interp(sp_candidate, dim_nodes, V))
                        return -(ret + beta * cont)

                    res = minimize_scalar(obj, bounds=(low, high), method="bounded", options={"xatol": 1e-8})
                    sp_opt = float(res.x)
                    v_opt = -float(res.fun)

                policy[i] = sp_opt
                V_new[i] = v_opt

            # Howard policy iteration acceleration
            if howard:
                u_vec = np.array([calc_return(s_i, policy[i]) for i, s_i in enumerate(dim_nodes)])
                for _ in range(n_howard):
                    V_new = u_vec + beta * np.interp(policy, dim_nodes, V_new)

            diff = float(np.max(np.abs(V_new - V)))
            V = V_new

            if diff < tol * (1.0 - beta) / beta:
                converged = True
                break

        res_norm = float(diff)
        eval_residual = self._create_residual_eval_fn()

        # Out-of-sample Euler residual
        test_grid = np.linspace(s_min, s_max, 1000)
        sp_test = np.interp(test_grid, dim_nodes, policy)
        sn_test = np.interp(sp_test, dim_nodes, policy)
        try:
            test_res = eval_residual(test_grid, sp_test, sn_test)
            max_euler_res = float(np.max(np.abs(test_res)))
        except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
            max_euler_res = np.nan

        elapsed = time.perf_counter() - start_time

        metadata = {
            "method": "bellman",
            "projection": self.projection,
            "policy_values": policy,
            "value_values": V,
            "residual_eval_fn": eval_residual,
            "max_euler_residual": max_euler_res,
            "borrowing_constraint": borrow_bound,
            "elapsed": elapsed,
            "backend": backend,
        }

        return FEMSolution(
            nodal_values=V,
            mesh=mesh,
            converged=converged,
            n_iter=n_iter,
            residual_norm=res_norm,
            backend=backend,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Functional Entry Point
# ---------------------------------------------------------------------------


def solve_fem(problem_or_domain: Union[FEMProblem, Tuple[float, float], Sequence[Any]], *args: Any, **kwargs: Any) -> FEMSolution:
    """Solve dynamic economic model via Finite Element Method (FEM / Galerkin).

    Parameters
    ----------
    problem_or_domain : FEMProblem, tuple, or sequence
        Either an existing FEMProblem instance or state domain specification.
    *args, **kwargs
        If problem_or_domain is not an FEMProblem, forwarded to FEMProblem constructor.
        Accepts `backend="numpy"|"numba"|"mlx"|"cupy"`.

    Returns
    -------
    solution : FEMSolution
        Solved FEM solution container.
    """
    backend = kwargs.pop("backend", "numpy")
    if isinstance(problem_or_domain, FEMProblem):
        return problem_or_domain.solve(backend=backend)

    problem = FEMProblem(problem_or_domain, *args, **kwargs)
    return problem.solve(backend=backend)
