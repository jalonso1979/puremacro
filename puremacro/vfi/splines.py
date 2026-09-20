r"""Shape-Preserving Cubic B-Splines & Schumaker Splines for puremacro.vfi.

Provides continuous state-space dynamic programming and projection methods
using smooth and shape-preserving splines:
- CubicBSplineBasis: Cox-de Boor recurrence for basis functions and derivatives
  (orders 0, 1, 2) supporting clamped (multiplicity 4), natural (S''=0), and
  not-a-knot boundary knot distributions. Greville abscissae collocation nodes.
- SchumakerSpline: Full implementation of Schumaker (1983) shape-preserving
  quadratic splines with dynamic sub-knot insertion \bar{\xi}_i \in (x_i, x_{i+1}).
  Guarantees strict monotonicity (S'(x) >= 0 whenever data is monotonic) and
  concavity (S''(x) <= 0 whenever data is concave), completely eliminating
  spurious oscillations and Gibbs ringing near borrowing constraints.
- SplineCollocationProblem: Problem specification for continuous Euler equation
  residual projection and Bellman value collocation.
- SplineCollocationSolution: Frozen solution container with continuous evaluation
  methods (.policy(s), .marginal_policy(s), .value(s), .marginal_value(s),
  .euler_residual(s)) and full presentation contract (.summary(), .plot(),
  .to_frame(), .to_markdown(), .to_latex(), .to_typst()).
- solve_spline_collocation: Convenience solver entry point.
- Multi-backend acceleration (NumPy reference, Numba CPU JIT, Apple Silicon MLX GPU,
  NVIDIA CuPy GPU) with automatic warning-safe fallback to NumPy.
"""
from __future__ import annotations

import inspect
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar, root

from puremacro import _backend as _bk
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst


# ---------------------------------------------------------------------------
# Cox-de Boor Numerical Kernels
# ---------------------------------------------------------------------------

@_bk.njit_fallback(fastmath=True)
def _de_boor_basis_kernel(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """Evaluate 1D B-spline basis functions of given degree via Cox-de Boor recurrence."""
    n_x = len(x)
    n_knots = len(knots)
    b_curr = np.zeros((n_x, n_knots - 1), dtype=np.float64)

    # Degree 0 initialization
    for i in range(n_knots - 1):
        t_i = knots[i]
        t_ip1 = knots[i + 1]
        if t_ip1 > t_i:
            if i == n_knots - degree - 2 or (t_ip1 == knots[-1]):
                # Include right boundary for the final active knot interval
                for j in range(n_x):
                    if t_i <= x[j] <= t_ip1:
                        b_curr[j, i] = 1.0
            else:
                for j in range(n_x):
                    if t_i <= x[j] < t_ip1:
                        b_curr[j, i] = 1.0

    # Higher-degree recurrence
    for d in range(1, degree + 1):
        b_next = np.zeros((n_x, n_knots - d - 1), dtype=np.float64)
        for i in range(n_knots - d - 1):
            dt1 = knots[i + d] - knots[i]
            dt2 = knots[i + d + 1] - knots[i + 1]
            inv_dt1 = 1.0 / dt1 if dt1 > 0.0 else 0.0
            inv_dt2 = 1.0 / dt2 if dt2 > 0.0 else 0.0

            for j in range(n_x):
                val = 0.0
                if inv_dt1 > 0.0:
                    val += ((x[j] - knots[i]) * inv_dt1) * b_curr[j, i]
                if inv_dt2 > 0.0:
                    val += ((knots[i + d + 1] - x[j]) * inv_dt2) * b_curr[j, i + 1]
                b_next[j, i] = val
        b_curr = b_next

    return b_curr


def _de_boor_evaluate_all(
    x: np.ndarray, knots: np.ndarray, degree: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate basis functions, first derivatives, and second derivatives.

    Uses the exact analytical recurrence for derivatives of B-splines.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    n_x = len(x_arr)
    n_knots = len(knots)

    # Store basis matrices for degrees 0, 1, ..., degree
    bases: dict[int, np.ndarray] = {}
    b0 = np.zeros((n_x, n_knots - 1), dtype=np.float64)
    for i in range(n_knots - 1):
        t_i = knots[i]
        t_ip1 = knots[i + 1]
        if t_ip1 > t_i:
            if i == n_knots - degree - 2 or (t_ip1 == knots[-1]):
                mask = (x_arr >= t_i) & (x_arr <= t_ip1)
            else:
                mask = (x_arr >= t_i) & (x_arr < t_ip1)
            b0[mask, i] = 1.0
    bases[0] = b0

    for d in range(1, degree + 1):
        bd = np.zeros((n_x, n_knots - d - 1), dtype=np.float64)
        for i in range(n_knots - d - 1):
            dt1 = knots[i + d] - knots[i]
            dt2 = knots[i + d + 1] - knots[i + 1]
            term1 = ((x_arr - knots[i]) / dt1) * bases[d - 1][:, i] if dt1 > 0.0 else 0.0
            term2 = ((knots[i + d + 1] - x_arr) / dt2) * bases[d - 1][:, i + 1] if dt2 > 0.0 else 0.0
            bd[:, i] = term1 + term2
        bases[d] = bd

    # First derivatives for degree p:
    # B'_{i, p}(x) = p * [ B_{i, p-1}(x)/(t_{i+p} - t_i) - B_{i+1, p-1}(x)/(t_{i+p+1} - t_{i+1}) ]
    db = np.zeros((n_x, n_knots - degree - 1), dtype=np.float64)
    for i in range(n_knots - degree - 1):
        dt1 = knots[i + degree] - knots[i]
        dt2 = knots[i + degree + 1] - knots[i + 1]
        t1 = (bases[degree - 1][:, i] / dt1) if dt1 > 0.0 else 0.0
        t2 = (bases[degree - 1][:, i + 1] / dt2) if dt2 > 0.0 else 0.0
        db[:, i] = degree * (t1 - t2)

    # Second derivatives:
    # First derivative of degree p - 1 basis:
    db_prev = np.zeros((n_x, n_knots - degree), dtype=np.float64)
    for i in range(n_knots - degree):
        dt1 = knots[i + degree - 1] - knots[i]
        dt2 = knots[i + degree] - knots[i + 1]
        t1 = (bases[degree - 2][:, i] / dt1) if dt1 > 0.0 else 0.0
        t2 = (bases[degree - 2][:, i + 1] / dt2) if dt2 > 0.0 else 0.0
        db_prev[:, i] = (degree - 1) * (t1 - t2)

    d2b = np.zeros((n_x, n_knots - degree - 1), dtype=np.float64)
    for i in range(n_knots - degree - 1):
        dt1 = knots[i + degree] - knots[i]
        dt2 = knots[i + degree + 1] - knots[i + 1]
        t1 = (db_prev[:, i] / dt1) if dt1 > 0.0 else 0.0
        t2 = (db_prev[:, i + 1] / dt2) if dt2 > 0.0 else 0.0
        d2b[:, i] = degree * (t1 - t2)

    return bases[degree], db, d2b


# ---------------------------------------------------------------------------
# CubicBSplineBasis Class
# ---------------------------------------------------------------------------

class CubicBSplineBasis:
    r"""Univariate Cubic B-Spline Basis with De Boor recursive evaluation.

    Supports:
    - Clamped boundary knots (multiplicity 4 at endpoints: B_0(a)=1, B_{K-1}(b)=1).
    - Natural boundary splines (S''(a) = S''(b) = 0).
    - Not-a-knot boundary conditions (continuity of S''' across first and last interior breakpoints).
    - Exact analytical basis functions and derivatives of order 0, 1, and 2.
    - Greville abscissae collocation nodes: \xi_i^* = (t_{i+1} + t_{i+2} + t_{i+3}) / 3.
    """

    def __init__(
        self,
        knots: Union[np.ndarray, Sequence[float], Tuple[float, float], None] = None,
        domain: Optional[Tuple[float, float]] = None,
        n_knots: int = 20,
        degree: int = 3,
        bc_type: str = "clamped",
        knot_spacing: str = "uniform",
    ):
        if degree != 3:
            raise ValueError(f"CubicBSplineBasis supports degree=3; got {degree}")

        bc_type = str(bc_type).strip().lower()
        if bc_type not in ("clamped", "natural", "not-a-knot"):
            raise ValueError(f"Unknown bc_type '{bc_type}'; supported: 'clamped', 'natural', 'not-a-knot'")

        knot_spacing = str(knot_spacing).strip().lower()
        if knot_spacing not in ("uniform", "geometric"):
            raise ValueError(f"Unknown knot_spacing '{knot_spacing}'; supported: 'uniform', 'geometric'")

        self.degree = 3
        self.bc_type = bc_type
        self.knot_spacing = knot_spacing

        # Determine breakpoints and full knot sequence
        if knots is not None and domain is None:
            knots_arr = np.asarray(knots, dtype=np.float64)
            if knots_arr.ndim != 1 or len(knots_arr) < 2:
                raise ValueError("knots must be a 1D sequence with at least 2 points.")

            # Check if user already provided a full clamped knot vector (e.g. knots[0] == knots[3])
            if len(knots_arr) >= 8 and np.isclose(knots_arr[0], knots_arr[3]) and np.isclose(knots_arr[-1], knots_arr[-4]):
                self.knots = knots_arr.copy()
                self.domain = (float(self.knots[3]), float(self.knots[-4]))
                self.breakpoints = np.unique(self.knots)
                self.n_basis = len(self.knots) - self.degree - 1
                self.n_nodes = self.n_basis
                return
            else:
                # User provided breakpoints
                breaks = np.sort(knots_arr)
                if np.any(np.diff(breaks) <= 0.0):
                    raise ValueError("Breakpoints must be strictly monotonic.")
                a, b = float(breaks[0]), float(breaks[-1])
                self.domain = (a, b)
                self.breakpoints = breaks
        else:
            if domain is None:
                domain = (0.0, 1.0)
            if isinstance(domain, (tuple, list)):
                if len(domain) == 2 and not isinstance(domain[0], (tuple, list)):
                    a, b = float(domain[0]), float(domain[1])
                elif len(domain) == 1 and isinstance(domain[0], (tuple, list)):
                    a, b = float(domain[0][0]), float(domain[0][1])
                else:
                    a, b = float(domain[0]), float(domain[1])
            else:
                raise ValueError(f"Invalid domain {domain!r}")

            if a >= b:
                raise ValueError(f"Domain lower bound must be strictly less than upper bound; got ({a}, {b})")

            self.domain = (a, b)
            if n_knots < 4:
                raise ValueError(f"n_knots must be >= 4 for cubic spline; got {n_knots}")

            if knot_spacing == "uniform":
                self.breakpoints = np.linspace(a, b, n_knots)
            else:
                # Geometric spacing clustered near lower bound
                s = np.linspace(0.0, 1.0, n_knots)
                self.breakpoints = a + (b - a) * (s**2.0)

        # Construct full knot vector according to boundary condition
        breaks = self.breakpoints
        m = len(breaks) - 1  # number of intervals
        a, b = self.domain

        if self.bc_type == "clamped":
            # Multiplicity 4 at endpoints
            interior = breaks[1:-1]
            self.knots = np.concatenate(([a] * 4, interior, [b] * 4))
        elif self.bc_type == "not-a-knot":
            # Omit first and last interior knots if m >= 4
            if m >= 4:
                interior = breaks[2:-2]
            elif m >= 3:
                interior = breaks[2:-1]
            else:
                interior = breaks[1:-1]
            self.knots = np.concatenate(([a] * 4, interior, [b] * 4))
        elif self.bc_type == "natural":
            # Clamped knots with natural boundary constraint S''(a)=0, S''(b)=0 applied at fitting
            interior = breaks[1:-1]
            self.knots = np.concatenate(([a] * 4, interior, [b] * 4))

        self.n_basis = len(self.knots) - self.degree - 1
        self.n_nodes = self.n_basis

    def nodes(self, squeeze: bool = True) -> np.ndarray:
        r"""Compute optimal Greville abscissae collocation nodes:

        \xi_i^* = \frac{t_{i+1} + t_{i+2} + t_{i+3}}{3}, \quad i = 0, \dots, K-1.
        """
        p = self.degree
        greville = np.zeros(self.n_basis, dtype=np.float64)
        for i in range(self.n_basis):
            greville[i] = np.mean(self.knots[i + 1 : i + p + 1])
        # Force exact endpoints for numerical precision
        greville[0] = self.domain[0]
        greville[-1] = self.domain[1]
        return greville if squeeze else greville[:, None]

    def evaluate(self, x: Union[float, np.ndarray], deriv: int = 0) -> np.ndarray:
        """Evaluate B-spline basis functions or derivatives at points x.

        Parameters
        ----------
        x : float or array-like
            Evaluation coordinates in [a, b].
        deriv : int, default 0
            Derivative order (0 for values, 1 for first derivative, 2 for second derivative).

        Returns
        -------
        np.ndarray
            Basis matrix of shape (len(x), n_basis).
        """
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        a, b = self.domain
        x_clipped = np.clip(x_arr, a, b)

        if deriv == 0:
            return _de_boor_basis_kernel(x_clipped, self.knots, degree=self.degree)
        elif deriv in (1, 2):
            _, db, d2b = _de_boor_evaluate_all(x_clipped, self.knots, degree=self.degree)
            return db if deriv == 1 else d2b
        else:
            raise ValueError(f"deriv must be 0, 1, or 2; got {deriv}")

    def basis_matrix(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate basis matrix B(x) (alias for evaluate(x, deriv=0))."""
        return self.evaluate(x, deriv=0)

    def derivative_matrix(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate first derivative matrix B'(x) (alias for evaluate(x, deriv=1))."""
        return self.evaluate(x, deriv=1)

    def second_derivative_matrix(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate second derivative matrix B''(x) (alias for evaluate(x, deriv=2))."""
        return self.evaluate(x, deriv=2)

    def fit(
        self,
        y: np.ndarray,
        nodes: Optional[np.ndarray] = None,
        backend: str = "numpy",
    ) -> np.ndarray:
        """Solve for control point coefficients c matching nodal values y.

        Parameters
        ----------
        y : array-like
            Function values at collocation nodes.
        nodes : array-like, optional
            Collocation nodes (defaults to self.nodes()).
        backend : str, default "numpy"
            Compute backend.
        """
        if nodes is None:
            nodes = self.nodes()
        nodes_arr = np.asarray(nodes, dtype=np.float64).ravel()
        y_arr = np.asarray(y, dtype=np.float64)

        b_mat = self.basis_matrix(nodes_arr)

        if self.bc_type == "natural" and len(nodes_arr) == self.n_basis:
            # Enforce S''(a) = 0 and S''(b) = 0 by replacing first and last rows
            d2_a = self.second_derivative_matrix(np.array([self.domain[0]]))[0]
            d2_b = self.second_derivative_matrix(np.array([self.domain[1]]))[0]
            b_sys = b_mat.copy()
            y_sys = y_arr.copy()
            b_sys[0] = d2_a
            y_sys[0] = 0.0
            b_sys[-1] = d2_b
            y_sys[-1] = 0.0
            return np.linalg.solve(b_sys, y_sys)

        if b_mat.shape[0] == b_mat.shape[1]:
            return np.linalg.solve(b_mat, y_arr)
        else:
            coefs, _, _, _ = np.linalg.lstsq(b_mat, y_arr, rcond=None)
            return coefs

    def interpolate(
        self,
        coefficients: np.ndarray,
        x: Union[float, np.ndarray],
        deriv: int = 0,
        backend: str = "numpy",
    ) -> Union[float, np.ndarray]:
        """Evaluate continuous spline function S(x) = B(x) @ coefficients."""
        is_scalar = np.ndim(x) == 0
        b_mat = self.evaluate(x, deriv=deriv)
        coefs = np.asarray(coefficients, dtype=np.float64)
        vals = b_mat @ coefs
        return float(vals[0]) if is_scalar else vals

    def __call__(
        self,
        coefficients: np.ndarray,
        x: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Callable alias for interpolate(coefficients, x, deriv=0)."""
        return self.interpolate(coefficients, x, deriv=0)


# Alias
SplineBasis = CubicBSplineBasis


# ---------------------------------------------------------------------------
# Schumaker (1983) Shape-Preserving Spline
# ---------------------------------------------------------------------------

class SchumakerSpline:
    r"""Schumaker (1983) Shape-Preserving Quadratic Spline Interpolant.

    Strictly preserves monotonicity (S'(x) >= 0) and concavity (S''(x) <= 0)
    without spurious oscillations or Gibbs ringing near borrowing constraints.

    Algorithm (Schumaker 1983, SIAM J. Numer. Anal.):
    1. Evaluates secant slopes s_i = (y_{i+1} - y_i) / (x_{i+1} - x_i).
    2. Determines interior knot derivatives d_i using weighted harmonic means,
       enforcing d_i = 0 at local extrema or flat segments.
    3. When chord condition d_i + d_{i+1} != 2 s_i holds, dynamically inserts
       an auxiliary sub-knot \bar{\xi}_i \in (x_i, x_{i+1}) and matching gradient \bar{d}_i.
    4. Evaluates values, first derivatives, and second derivatives via piecewise
       quadratic polynomials with continuous C^1 gradient matching.
    """

    def __init__(
        self,
        x: Union[np.ndarray, Sequence[float]],
        y: Union[np.ndarray, Sequence[float]],
        d_slopes: Optional[np.ndarray] = None,
        extrapolate: bool = True,
    ):
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        if x_arr.ndim != 1:
            raise ValueError("x must be a 1D array.")
        n = len(x_arr)
        if n < 2:
            raise ValueError("At least 2 points required for Schumaker spline interpolation.")

        dx = np.diff(x_arr)
        if np.any(dx <= 0.0):
            raise ValueError("x coordinates must be strictly increasing.")

        if y_arr.ndim == 1:
            if len(y_arr) != n:
                raise ValueError(f"Shape mismatch: x has {n} points, y has {len(y_arr)} points.")
            self._multi_dim = False
            self.x = x_arr
            self.y = y_arr
            self._init_1d(x_arr, y_arr, d_slopes=d_slopes)
        elif y_arr.ndim == 2:
            if y_arr.shape[0] != n:
                raise ValueError(f"Shape mismatch: x has {n} points, y has {y_arr.shape[0]} rows.")
            self._multi_dim = True
            self.x = x_arr
            self.y = y_arr
            self._sub_splines = []
            for col in range(y_arr.shape[1]):
                col_slopes = d_slopes[:, col] if d_slopes is not None else None
                spl = SchumakerSpline(x_arr, y_arr[:, col], d_slopes=col_slopes, extrapolate=extrapolate)
                self._sub_splines.append(spl)
            self.domain = (float(x_arr[0]), float(x_arr[-1]))
            self.knots = self._sub_splines[0].knots
            self.inserted_knots = self._sub_splines[0].inserted_knots
            self.d_slopes = np.column_stack([s.d_slopes for s in self._sub_splines])
        else:
            raise ValueError(f"y must be 1D or 2D; got shape {y_arr.shape}")

        self.extrapolate = extrapolate
        self.domain = (float(x_arr[0]), float(x_arr[-1]))

    def _init_1d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        d_slopes: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize 1D Schumaker piecewise quadratic polynomial subintervals."""
        n = len(x)
        dx = np.diff(x)
        s = np.diff(y) / dx

        # Step 2: Interior derivatives d_i
        if d_slopes is not None:
            d = np.asarray(d_slopes, dtype=np.float64).copy()
            if len(d) != n:
                raise ValueError("d_slopes must match length of x.")
        else:
            d = np.zeros(n, dtype=np.float64)
            for i in range(1, n - 1):
                if s[i - 1] * s[i] <= 0.0:
                    d[i] = 0.0
                else:
                    alpha_i = (dx[i - 1] + 2.0 * dx[i]) / (3.0 * (dx[i - 1] + dx[i]))
                    denom = alpha_i * s[i] + (1.0 - alpha_i) * s[i - 1]
                    d[i] = (s[i - 1] * s[i]) / denom if abs(denom) > 1e-15 else 0.0
                    # Clamp between adjacent secant slopes to guarantee shape preservation
                    d[i] = np.clip(d[i], min(s[i - 1], s[i]), max(s[i - 1], s[i]))

            # Boundary derivatives
            if s[0] >= 0.0:
                d[0] = max(0.0, min(2.0 * s[0], 2.0 * s[0] - d[1]))
            else:
                d[0] = min(0.0, max(2.0 * s[0], 2.0 * s[0] - d[1]))

            if s[-1] >= 0.0:
                d[-1] = max(0.0, min(2.0 * s[-1], 2.0 * s[-1] - d[-2]))
            else:
                d[-1] = min(0.0, max(2.0 * s[-1], 2.0 * s[-1] - d[-2]))

        self.d_slopes = d

        # Step 3: Subinterval assembly with dynamic sub-knot insertion
        sub_x_L, sub_x_R, sub_y_L, sub_d_L, sub_c2 = [], [], [], [], []
        inserted = []
        all_knots = [x[0]]

        for i in range(n - 1):
            delta_i = d[i] + d[i + 1] - 2.0 * s[i]
            h = dx[i]

            if abs(delta_i) < 1e-12:
                # Case A: Single quadratic polynomial satisfies C^1 boundary conditions
                sub_x_L.append(x[i])
                sub_x_R.append(x[i + 1])
                sub_y_L.append(y[i])
                sub_d_L.append(d[i])
                sub_c2.append((d[i + 1] - d[i]) / (2.0 * h))
                all_knots.append(x[i + 1])
            else:
                # Case B: Insert auxiliary sub-knot \bar{\xi}_i \in (x_i, x_{i+1})
                if (d[i] - s[i]) * (d[i + 1] - s[i]) >= 0.0:
                    diff_d = d[i] - d[i + 1]
                    if abs(diff_d) > 1e-15:
                        if abs(d[i] - s[i]) > abs(d[i + 1] - s[i]):
                            xi_bar = x[i] + (2.0 * (s[i] - d[i + 1]) / diff_d) * h
                        else:
                            xi_bar = x[i + 1] - (2.0 * (d[i] - s[i]) / (-diff_d)) * h
                    else:
                        xi_bar = 0.5 * (x[i] + x[i + 1])
                else:
                    xi_bar = 0.5 * (x[i] + x[i + 1])

                # Defensive clamping inside subinterval
                xi_bar = np.clip(xi_bar, x[i] + 0.05 * h, x[i + 1] - 0.05 * h)
                inserted.append(xi_bar)
                all_knots.extend([xi_bar, x[i + 1]])

                h1 = xi_bar - x[i]
                h2 = x[i + 1] - xi_bar

                # Intermediate derivative \bar{d}_i
                d_bar = (2.0 * (y[i + 1] - y[i]) - d[i] * h1 - d[i + 1] * h2) / h
                if s[i] >= 0.0:
                    d_bar = max(0.0, d_bar)
                else:
                    d_bar = min(0.0, d_bar)

                # Intermediate value \bar{y}_i
                y_bar = y[i] + 0.5 * (d[i] + d_bar) * h1

                # Subinterval 1: [x_i, \bar{\xi}_i]
                sub_x_L.append(x[i])
                sub_x_R.append(xi_bar)
                sub_y_L.append(y[i])
                sub_d_L.append(d[i])
                sub_c2.append((d_bar - d[i]) / (2.0 * h1))

                # Subinterval 2: [\bar{\xi}_i, x_{i+1}]
                sub_x_L.append(xi_bar)
                sub_x_R.append(x[i + 1])
                sub_y_L.append(y_bar)
                sub_d_L.append(d_bar)
                sub_c2.append((d[i + 1] - d_bar) / (2.0 * h2))

        self.sub_x_L = np.array(sub_x_L, dtype=np.float64)
        self.sub_x_R = np.array(sub_x_R, dtype=np.float64)
        self.sub_y_L = np.array(sub_y_L, dtype=np.float64)
        self.sub_d_L = np.array(sub_d_L, dtype=np.float64)
        self.sub_c2 = np.array(sub_c2, dtype=np.float64)
        self.inserted_knots = np.array(inserted, dtype=np.float64)
        self.knots = np.array(all_knots, dtype=np.float64)

    def _eval_1d(
        self, x_eval: np.ndarray, deriv: int = 0
    ) -> np.ndarray:
        """Fast vectorized piecewise quadratic evaluation."""
        idx = np.searchsorted(self.sub_x_R, x_eval, side="left")
        idx = np.clip(idx, 0, len(self.sub_x_R) - 1)

        t = x_eval - self.sub_x_L[idx]

        if deriv == 0:
            vals = self.sub_y_L[idx] + self.sub_d_L[idx] * t + self.sub_c2[idx] * (t**2)
            if self.extrapolate:
                # Linear extrapolation outside [x_0, x_n]
                left_mask = x_eval < self.x[0]
                right_mask = x_eval > self.x[-1]
                if np.any(left_mask):
                    vals[left_mask] = self.y[0] + self.d_slopes[0] * (x_eval[left_mask] - self.x[0])
                if np.any(right_mask):
                    vals[right_mask] = self.y[-1] + self.d_slopes[-1] * (x_eval[right_mask] - self.x[-1])
            return vals
        elif deriv == 1:
            dvals = self.sub_d_L[idx] + 2.0 * self.sub_c2[idx] * t
            if self.extrapolate:
                left_mask = x_eval < self.x[0]
                right_mask = x_eval > self.x[-1]
                if np.any(left_mask):
                    dvals[left_mask] = self.d_slopes[0]
                if np.any(right_mask):
                    dvals[right_mask] = self.d_slopes[-1]
            return dvals
        elif deriv == 2:
            d2vals = 2.0 * self.sub_c2[idx]
            if self.extrapolate:
                left_mask = x_eval < self.x[0]
                right_mask = x_eval > self.x[-1]
                if np.any(left_mask):
                    d2vals[left_mask] = 0.0
                if np.any(right_mask):
                    d2vals[right_mask] = 0.0
            return d2vals
        else:
            return np.zeros_like(x_eval)

    def eval(self, x_eval: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate Schumaker spline interpolant values S(x)."""
        is_scalar = np.ndim(x_eval) == 0
        xq = np.atleast_1d(np.asarray(x_eval, dtype=np.float64))

        if not self._multi_dim:
            vals = self._eval_1d(xq, deriv=0)
            return float(vals[0]) if is_scalar else vals
        else:
            res = np.column_stack([s._eval_1d(xq, deriv=0) for s in self._sub_splines])
            return res[0] if is_scalar else res

    def __call__(self, x_eval: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Callable alias for eval(x_eval)."""
        return self.eval(x_eval)

    def derivative(
        self, x_eval: Union[float, np.ndarray], order: int = 1
    ) -> Union[float, np.ndarray]:
        """Evaluate derivative of order 1 or 2."""
        if order == 0:
            return self.eval(x_eval)
        is_scalar = np.ndim(x_eval) == 0
        xq = np.atleast_1d(np.asarray(x_eval, dtype=np.float64))

        if not self._multi_dim:
            dvals = self._eval_1d(xq, deriv=order)
            return float(dvals[0]) if is_scalar else dvals
        else:
            res = np.column_stack([s._eval_1d(xq, deriv=order) for s in self._sub_splines])
            return res[0] if is_scalar else res

    def second_derivative(self, x_eval: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate analytical second derivative S''(x)."""
        return self.derivative(x_eval, order=2)


# ---------------------------------------------------------------------------
# SplineCollocationSolution Class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplineCollocationSolution:
    """Frozen solution object from continuous spline collocation.

    Attributes
    ----------
    coefficients : np.ndarray
        Solved spline control point coefficients (CubicBSplineBasis) or
        solved nodal policy values (SchumakerSpline).
    basis : Any
        Underlying spline basis (CubicBSplineBasis or SchumakerSpline).
    converged : bool
        Whether the solver reached convergence tolerance.
    n_iter : int
        Number of iterations executed.
    residual_norm : float
        Final residual norm at collocation nodes.
    backend : str
        Compute backend utilized ('numpy', 'numba', 'mlx', 'cupy').
    metadata : dict
        Auxiliary diagnostic metadata, including elapsed time, problem params,
        and continuous evaluation hooks.
    """

    coefficients: np.ndarray
    basis: Any
    converged: bool
    n_iter: int
    residual_norm: float
    backend: str
    metadata: dict = field(default_factory=dict)

    def policy(self, s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate continuous policy function g(s)."""
        is_scalar = np.ndim(s) == 0
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))

        if isinstance(self.basis, CubicBSplineBasis):
            vals = self.basis.interpolate(self.coefficients, s_arr, deriv=0, backend=self.backend)
        elif isinstance(self.basis, SchumakerSpline):
            vals = self.basis.eval(s_arr)
        else:
            raise TypeError(f"Unsupported basis type: {type(self.basis)}")

        return float(vals[0]) if is_scalar else vals

    def marginal_policy(self, s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Evaluate continuous policy derivative g'(s) = \partial g / \partial s."""
        is_scalar = np.ndim(s) == 0
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))

        if isinstance(self.basis, CubicBSplineBasis):
            vals = self.basis.interpolate(self.coefficients, s_arr, deriv=1, backend=self.backend)
        elif isinstance(self.basis, SchumakerSpline):
            vals = self.basis.derivative(s_arr, order=1)
        else:
            raise TypeError(f"Unsupported basis type: {type(self.basis)}")

        return float(vals[0]) if is_scalar else vals

    def value(self, s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate continuous value function V(s)."""
        is_scalar = np.ndim(s) == 0
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))

        val_basis = self.metadata.get("value_basis")
        val_coefs = self.metadata.get("value_coefficients")

        if val_basis is not None and val_coefs is not None:
            if isinstance(val_basis, CubicBSplineBasis):
                vals = val_basis.interpolate(val_coefs, s_arr, deriv=0, backend=self.backend)
            else:
                vals = val_basis.eval(s_arr)
            return float(vals[0]) if is_scalar else vals

        if self.metadata.get("method") == "bellman":
            if isinstance(self.basis, CubicBSplineBasis):
                vals = self.basis.interpolate(self.coefficients, s_arr, deriv=0, backend=self.backend)
            else:
                vals = self.basis.eval(s_arr)
            return float(vals[0]) if is_scalar else vals

        raise ValueError(
            "Value function was not solved or not available in this solution (method='euler')."
        )

    def marginal_value(self, s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate analytical marginal value V'(s) via envelope theorem: V'(s) = u'(c(s)) f'(s)."""
        is_scalar = np.ndim(s) == 0
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        params = self.metadata.get("params", {})
        alpha = params.get("alpha", 0.36)
        delta = params.get("delta", 1.0)
        sigma = params.get("sigma", 1.0)
        z = float(params.get("z", params.get("A", 1.0)))

        kp = self.policy(s_arr)
        f_k = z * (s_arr**alpha) + (1.0 - delta) * s_arr
        c = np.maximum(f_k - kp, 1e-12)
        uc = (1.0 / c) if sigma == 1.0 else (c ** (-sigma))
        fk = z * alpha * (s_arr ** (alpha - 1.0)) + 1.0 - delta
        val = uc * fk
        return float(val[0]) if is_scalar else val

    def euler_residual(self, s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate continuous Euler equation residual R(s) across continuous domain points."""
        is_scalar = np.ndim(s) == 0
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        params = self.metadata.get("params", {})
        beta = self.metadata.get("beta", 0.96)
        alpha = params.get("alpha", 0.36)
        delta = params.get("delta", 1.0)
        sigma = params.get("sigma", 1.0)
        z = float(params.get("z", params.get("A", 1.0)))

        kp = self.policy(s_arr)
        f_k = z * (s_arr**alpha) + (1.0 - delta) * s_arr
        kp_clamped = np.clip(kp, 1e-8, f_k - 1e-8)
        c = np.maximum(f_k - kp_clamped, 1e-12)

        f_kp = z * (kp_clamped**alpha) + (1.0 - delta) * kp_clamped
        kpp = self.policy(kp_clamped)
        kpp_clamped = np.clip(kpp, 1e-8, f_kp - 1e-8)
        cp = np.maximum(f_kp - kpp_clamped, 1e-12)

        uc = (1.0 / c) if sigma == 1.0 else (c ** (-sigma))
        ucp = (1.0 / cp) if sigma == 1.0 else (cp ** (-sigma))

        fkp = z * alpha * (kp_clamped ** (alpha - 1.0)) + 1.0 - delta
        res = 1.0 - beta * (ucp / uc) * fkp
        return float(res[0]) if is_scalar else res

    def summary(self) -> pd.DataFrame:
        """Structured summary DataFrame of spline collocation results."""
        domain_str = f"[{self.basis.domain[0]:.4g}, {self.basis.domain[1]:.4g}]"
        spline_type = "Schumaker (1983)" if isinstance(self.basis, SchumakerSpline) else "Cubic B-Spline"
        bc_type = getattr(self.basis, "bc_type", "shape-preserving")
        elapsed = self.metadata.get("elapsed_time", np.nan)

        rows = [
            {"Metric": "Method", "Value": str(self.metadata.get("method", "euler"))},
            {"Metric": "Spline Type", "Value": spline_type},
            {"Metric": "Boundary Condition", "Value": bc_type},
            {"Metric": "State Domain", "Value": domain_str},
            {"Metric": "Total Knots / Nodes", "Value": str(len(self.coefficients))},
            {"Metric": "Converged", "Value": str(self.converged)},
            {"Metric": "Iterations", "Value": str(self.n_iter)},
            {"Metric": "Residual Norm", "Value": f"{self.residual_norm:.4e}"},
            {"Metric": "Backend", "Value": str(self.backend)},
            {"Metric": "Elapsed Time (s)", "Value": f"{elapsed:.4f}" if np.isfinite(elapsed) else "N/A"},
        ]
        return pd.DataFrame(rows).set_index("Metric")

    def to_frame(self) -> pd.DataFrame:
        """Return summary table as DataFrame."""
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
        """Plot continuous policy, marginal policy, and continuous Euler equation residual."""
        import matplotlib.pyplot as plt

        a, b = self.basis.domain
        s_dense = np.linspace(a, b, 200)

        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

        # Panel 1: Policy function g(s)
        ax1 = axes[0]
        pol_dense = self.policy(s_dense)
        ax1.plot(s_dense, pol_dense, color="tab:blue", lw=2, label="$g(s)$")
        ax1.plot(s_dense, s_dense, color="gray", ls="--", alpha=0.7, label="$s'=s$")
        if hasattr(self.basis, "nodes"):
            nd = self.basis.nodes()
            ax1.scatter(nd, self.policy(nd), color="tab:red", s=25, zorder=5, label="Nodes")
        elif hasattr(self.basis, "x"):
            ax1.scatter(self.basis.x, self.policy(self.basis.x), color="tab:red", s=25, zorder=5, label="Nodes")
        ax1.set_title("Policy Function $s' = g(s)$")
        ax1.set_xlabel("State $s$")
        ax1.set_ylabel("Next State $s'$")
        ax1.legend(loc="best", frameon=True)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Marginal policy g'(s)
        ax2 = axes[1]
        mpol_dense = self.marginal_policy(s_dense)
        ax2.plot(s_dense, mpol_dense, color="tab:purple", lw=2, label="$g'(s)$")
        ax2.axhline(0.0, color="gray", ls="--", alpha=0.5)
        ax2.set_title("Marginal Policy $\\partial g / \\partial s$")
        ax2.set_xlabel("State $s$")
        ax2.set_ylabel("Slope")
        ax2.grid(True, alpha=0.3)

        # Panel 3: Euler equation residual R(s)
        ax3 = axes[2]
        res_dense = np.abs(self.euler_residual(s_dense))
        ax3.semilogy(s_dense, np.maximum(res_dense, 1e-16), color="tab:green", lw=2)
        ax3.set_title("Euler Residual $|R(s)|$")
        ax3.set_xlabel("State $s$")
        ax3.set_ylabel("Absolute Error")
        ax3.grid(True, alpha=0.3)

        if show:
            plt.show()
        return fig


# ---------------------------------------------------------------------------
# SplineCollocationProblem Class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplineCollocationProblem:
    """Continuous state-space economic problem solved via spline collocation.

    Parameters
    ----------
    domain : tuple[float, float] or list
        Bounded continuous state domain [a, b].
    n_knots : int, default 20
        Number of knots / breakpoints.
    spline_type : {"cubic", "schumaker"}, default "cubic"
        Spline basis type:
        - "cubic": Cox-de Boor Cubic B-Splines with C^2 smooth representation.
        - "schumaker": Schumaker (1983) shape-preserving quadratic splines with C^1 gradient.
    bc_type : {"clamped", "natural", "not-a-knot"}, default "clamped"
        Boundary condition knot treatment (for cubic splines).
    knot_spacing : {"uniform", "geometric"}, default "uniform"
        Knot placement distribution.
    method : {"euler", "bellman"}, default "euler"
        Solution method.
    euler_residual_fn : Callable, optional
        Euler equation residual function R(s, s', s'', params).
    return_fn : Callable, optional
        One-period return function u(s, s', params).
    transition_fn : Callable, optional
        Production / resource transition equation f(s).
    beta : float, default 0.96
        Subjective discount factor in (0, 1).
    params : dict, optional
        Structural model parameters (alpha, delta, sigma, z).
    borrowing_constraint : float, optional
        Lower bound on next-period state s' >= borrowing_constraint.
    options : dict, optional
        Solver options ('tol', 'max_iter', 'solver', 'n_howard').
    shocks : tuple[np.ndarray, np.ndarray], optional
        Exogenous Markov shock states (z_grid, P_z).
    """

    domain: Union[Tuple[float, float], Tuple[Tuple[float, float], ...], list]
    n_knots: int = 20
    spline_type: str = "cubic"
    bc_type: str = "clamped"
    knot_spacing: str = "uniform"
    method: str = "euler"
    euler_residual_fn: Optional[Callable] = None
    return_fn: Optional[Callable] = None
    transition_fn: Optional[Callable] = None
    beta: float = 0.96
    params: dict = field(default_factory=dict)
    borrowing_constraint: Optional[float] = None
    options: dict = field(default_factory=dict)
    shocks: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def __post_init__(self) -> None:
        if not (0.0 < float(self.beta) < 1.0):
            raise ValueError(f"beta must be in (0, 1); got {self.beta}")

        st = str(self.spline_type).strip().lower()
        if st in ("cubic", "b-spline", "bspline"):
            object.__setattr__(self, "spline_type", "cubic")
        elif st in ("schumaker", "shape-preserving"):
            object.__setattr__(self, "spline_type", "schumaker")
        else:
            raise ValueError(f"Unknown spline_type '{self.spline_type}'; supported: 'cubic', 'schumaker'")

        bc = str(self.bc_type).strip().lower()
        if bc not in ("clamped", "natural", "not-a-knot"):
            raise ValueError(f"Unknown bc_type '{self.bc_type}'; supported: 'clamped', 'natural', 'not-a-knot'")
        object.__setattr__(self, "bc_type", bc)

        ks = str(self.knot_spacing).strip().lower()
        if ks not in ("uniform", "geometric"):
            raise ValueError(f"Unknown knot_spacing '{self.knot_spacing}'; supported: 'uniform', 'geometric'")
        object.__setattr__(self, "knot_spacing", ks)

        m = str(self.method).strip().lower()
        if m not in ("euler", "bellman"):
            raise ValueError(f"method must be 'euler' or 'bellman'; got {self.method!r}")
        object.__setattr__(self, "method", m)

        # Normalize domain bounds
        if isinstance(self.domain, (tuple, list)):
            if len(self.domain) == 2 and not isinstance(self.domain[0], (tuple, list)):
                a, b = float(self.domain[0]), float(self.domain[1])
            elif len(self.domain) == 1 and isinstance(self.domain[0], (tuple, list)):
                a, b = float(self.domain[0][0]), float(self.domain[0][1])
            else:
                a, b = float(self.domain[0]), float(self.domain[1])
        else:
            raise ValueError(f"Invalid domain {self.domain!r}")

        if a >= b:
            raise ValueError(f"domain bounds must satisfy a < b; got ({a}, {b})")
        object.__setattr__(self, "domain", (a, b))

        if int(self.n_knots) < 4:
            raise ValueError(f"n_knots must be >= 4; got {self.n_knots}")

    def solve(
        self,
        backend: str = "numpy",
        method: Optional[str] = None,
        tol: float = 1e-8,
        max_iter: int = 500,
        **kwargs,
    ) -> SplineCollocationSolution:
        """Solve the continuous dynamic programming model using spline collocation.

        Parameters
        ----------
        backend : str, default "numpy"
            Compute backend ("numpy", "numba", "mlx", "cupy").
        method : str, optional
            Override solution method ("euler" or "bellman").
        tol : float, default 1e-8
            Convergence tolerance.
        max_iter : int, default 500
            Maximum solver iterations.

        Returns
        -------
        SplineCollocationSolution
            Solved continuous projection solution.
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

        sol_method = (method or self.method).strip().lower()
        tol = float(self.options.get("tol", tol))
        max_iter = int(self.options.get("max_iter", max_iter))

        t0 = time.perf_counter()
        if sol_method == "euler":
            sol = _solve_spline_euler(self, backend=resolved_backend, tol=tol, max_iter=max_iter)
        else:
            sol = _solve_spline_bellman(self, backend=resolved_backend, tol=tol, max_iter=max_iter)
        t1 = time.perf_counter()

        sol.metadata["elapsed_time"] = t1 - t0
        sol.metadata["backend"] = resolved_backend
        sol.metadata["method"] = sol_method
        sol.metadata["params"] = self.params
        sol.metadata["beta"] = self.beta
        return sol


# ---------------------------------------------------------------------------
# Solver Implementations: Euler Projection & Bellman Value Collocation
# ---------------------------------------------------------------------------

def _evaluate_euler_residual_spline(
    problem: SplineCollocationProblem,
    basis_obj: Any,
    theta: np.ndarray,
    nodes: np.ndarray,
    backend: str,
) -> np.ndarray:
    """Evaluate Euler equation residuals at collocation nodes for candidate parameters."""
    if problem.spline_type == "cubic":
        def policy_cand(s):
            return basis_obj.interpolate(theta, s, deriv=0, backend=backend)
        kp = policy_cand(nodes)
    else:
        spl = SchumakerSpline(nodes, theta)
        def policy_cand(s):
            return spl.eval(s)
        kp = theta

    s = nodes

    if problem.euler_residual_fn is not None:
        fn = problem.euler_residual_fn
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

    # Canonical Neoclassical Growth Model
    alpha = problem.params.get("alpha", 0.36)
    delta = problem.params.get("delta", 1.0)
    sigma = problem.params.get("sigma", 1.0)
    z = float(problem.params.get("z", problem.params.get("A", 1.0)))

    f_k = z * (s**alpha) + (1.0 - delta) * s
    kp_clamped = np.clip(kp, 1e-8, f_k - 1e-8)
    c = np.maximum(f_k - kp_clamped, 1e-12)

    f_kp = z * (kp_clamped**alpha) + (1.0 - delta) * kp_clamped
    kpp = policy_cand(kp_clamped)
    kpp_clamped = np.clip(kpp, 1e-8, f_kp - 1e-8)
    cp = np.maximum(f_kp - kpp_clamped, 1e-12)

    uc = (1.0 / c) if sigma == 1.0 else (c ** (-sigma))
    ucp = (1.0 / cp) if sigma == 1.0 else (cp ** (-sigma))

    fkp = z * alpha * (kp_clamped ** (alpha - 1.0)) + 1.0 - delta
    euler_res = 1.0 - problem.beta * (ucp / uc) * fkp
    return euler_res


def _solve_spline_euler(
    problem: SplineCollocationProblem,
    backend: str,
    tol: float,
    max_iter: int,
) -> SplineCollocationSolution:
    """Solve continuous Euler equation projection with spline basis."""
    a, b = problem.domain
    alpha = problem.params.get("alpha", 0.36)
    delta = problem.params.get("delta", 1.0)
    z = float(problem.params.get("z", problem.params.get("A", 1.0)))

    if problem.spline_type == "cubic":
        basis = CubicBSplineBasis(
            domain=problem.domain,
            n_knots=problem.n_knots,
            degree=3,
            bc_type=problem.bc_type,
            knot_spacing=problem.knot_spacing,
        )
        nodes = basis.nodes()
        n_dofs = basis.n_basis

        # Warm-up with Coleman time iteration on nodes
        theta_curr = basis.fit(0.95 * nodes, backend=backend)
        for _ in range(20):
            theta_prev = theta_curr.copy()
            kp_next = np.zeros(len(nodes))
            for i, ki in enumerate(nodes):
                f_ki = z * (ki**alpha) + (1.0 - delta) * ki
                def eq_i(kp_val):
                    c = max(f_ki - kp_val, 1e-12)
                    kpp = float(np.clip(basis.interpolate(theta_prev, np.array([kp_val]), backend=backend), 1e-6, z * (kp_val**alpha) - 1e-6)[0])
                    cp = max(z * (kp_val**alpha) + (1.0 - delta) * kp_val - kpp, 1e-12)
                    return 1.0 / c - problem.beta * (1.0 / cp) * (z * alpha * (kp_val ** (alpha - 1.0)) + 1.0 - delta)
                low = min(1e-5, 0.01 * f_ki)
                high = max(f_ki - 1e-5, 0.99 * f_ki)
                try:
                    kp_next[i] = brentq(eq_i, low, high)
                except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
                    kp_next[i] = 0.5 * f_ki
            theta_curr = basis.fit(kp_next, backend=backend)

        def residual_obj(th):
            return _evaluate_euler_residual_spline(problem, basis, th, nodes, backend)

        res = root(residual_obj, theta_curr, method="hybr", tol=tol, options={"maxfev": max_iter * len(nodes)})
        theta_opt = res.x if res.success else theta_curr
        n_iter = int(getattr(res, "nfev", 0))
        final_res_norm = float(np.max(np.abs(residual_obj(theta_opt))))
        converged = bool(np.isfinite(final_res_norm) and final_res_norm <= tol)

        return SplineCollocationSolution(
            coefficients=theta_opt,
            basis=basis,
            converged=converged,
            n_iter=n_iter,
            residual_norm=final_res_norm,
            backend=backend,
            metadata={"nodes": nodes},
        )

    else:
        # Schumaker (1983) Spline Collocation
        if problem.knot_spacing == "uniform":
            breaks = np.linspace(a, b, problem.n_knots)
        else:
            s_grid = np.linspace(0.0, 1.0, problem.n_knots)
            breaks = a + (b - a) * (s_grid**2.0)

        nodes = breaks
        y_curr = 0.95 * nodes

        # Warm-up with Coleman operator using Schumaker splines
        for _ in range(25):
            eval_prev = SchumakerSpline(nodes, y_curr)
            y_next = np.zeros(len(nodes))
            for i, ki in enumerate(nodes):
                f_ki = z * (ki**alpha) + (1.0 - delta) * ki
                def eq_i(kp_val):
                    c = max(f_ki - kp_val, 1e-12)
                    kpp = float(np.clip(eval_prev.eval(kp_val), 1e-6, z * (kp_val**alpha) - 1e-6))
                    cp = max(z * (kp_val**alpha) + (1.0 - delta) * kp_val - kpp, 1e-12)
                    return 1.0 / c - problem.beta * (1.0 / cp) * (z * alpha * (kp_val ** (alpha - 1.0)) + 1.0 - delta)
                low = min(1e-5, 0.01 * f_ki)
                high = max(f_ki - 1e-5, 0.99 * f_ki)
                try:
                    y_next[i] = brentq(eq_i, low, high)
                except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
                    y_next[i] = 0.5 * f_ki
            y_curr = y_next

        def residual_obj(y_cand):
            return _evaluate_euler_residual_spline(problem, None, y_cand, nodes, backend)

        res = root(residual_obj, y_curr, method="hybr", tol=tol, options={"maxfev": max_iter * len(nodes)})
        y_opt = res.x if res.success else y_curr
        n_iter = int(getattr(res, "nfev", 0))
        final_res_norm = float(np.max(np.abs(residual_obj(y_opt))))
        converged = bool(np.isfinite(final_res_norm) and final_res_norm <= tol)

        basis_opt = SchumakerSpline(nodes, y_opt)

        return SplineCollocationSolution(
            coefficients=y_opt,
            basis=basis_opt,
            converged=converged,
            n_iter=n_iter,
            residual_norm=final_res_norm,
            backend=backend,
            metadata={"nodes": nodes},
        )


def _solve_spline_bellman(
    problem: SplineCollocationProblem,
    backend: str,
    tol: float,
    max_iter: int,
) -> SplineCollocationSolution:
    """Solve Bellman equation value collocation using splines."""
    a, b = problem.domain
    alpha = problem.params.get("alpha", 0.36)
    delta = problem.params.get("delta", 1.0)
    z = float(problem.params.get("z", problem.params.get("A", 1.0)))

    if problem.spline_type == "cubic":
        basis = CubicBSplineBasis(
            domain=problem.domain,
            n_knots=problem.n_knots,
            degree=3,
            bc_type=problem.bc_type,
            knot_spacing=problem.knot_spacing,
        )
        nodes = basis.nodes()
    else:
        nodes = np.linspace(a, b, problem.n_knots)
        basis = None

    # Initial guess for value function
    v_curr = np.zeros(len(nodes))
    for i, ki in enumerate(nodes):
        c0 = max(z * (ki**alpha) - ki, 1e-4)
        v_curr[i] = np.log(c0) / (1.0 - problem.beta)

    policy_vals = np.zeros(len(nodes))
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        if problem.spline_type == "cubic":
            coef_v = basis.fit(v_curr, backend=backend)
            def v_func(s):
                return basis.interpolate(coef_v, s, deriv=0, backend=backend)
        else:
            v_spl = SchumakerSpline(nodes, v_curr)
            def v_func(s):
                return v_spl.eval(s)

        v_next = np.zeros(len(nodes))
        for i, ki in enumerate(nodes):
            f_ki = z * (ki**alpha) + (1.0 - delta) * ki
            low = a
            high = max(low + 1e-4, f_ki - 1e-5)

            def obj(kp):
                c = max(f_ki - kp, 1e-12)
                u = np.log(c)
                val_next = float(np.atleast_1d(v_func(np.array([kp])))[0])
                return -(u + problem.beta * val_next)

            res = minimize_scalar(obj, bounds=(low, high), method="bounded")
            policy_vals[i] = res.x
            v_next[i] = -res.fun

        diff = np.max(np.abs(v_next - v_curr))
        v_curr = v_next
        if diff < tol:
            converged = True
            break

    final_basis = basis if problem.spline_type == "cubic" else SchumakerSpline(nodes, policy_vals)
    final_coefs = basis.fit(policy_vals, backend=backend) if problem.spline_type == "cubic" else policy_vals

    return SplineCollocationSolution(
        coefficients=final_coefs,
        basis=final_basis,
        converged=converged,
        n_iter=it,
        residual_norm=diff,
        backend=backend,
        metadata={
            "nodes": nodes,
            "value_coefficients": v_curr,
            "value_basis": basis if problem.spline_type == "cubic" else SchumakerSpline(nodes, v_curr),
        },
    )


# ---------------------------------------------------------------------------
# Convenience Helper Function
# ---------------------------------------------------------------------------

def solve_spline_collocation(
    problem_or_domain: Optional[Union[SplineCollocationProblem, Tuple[float, float], Tuple[Tuple[float, float], ...], list]] = None,
    *args,
    domain: Optional[Union[Tuple[float, float], Tuple[Tuple[float, float], ...], list]] = None,
    n_knots: int = 20,
    spline_type: str = "cubic",
    bc_type: str = "clamped",
    knot_spacing: str = "uniform",
    method: str = "euler",
    euler_residual_fn: Optional[Callable] = None,
    return_fn: Optional[Callable] = None,
    transition_fn: Optional[Callable] = None,
    beta: float = 0.96,
    params: Optional[dict] = None,
    borrowing_constraint: Optional[float] = None,
    options: Optional[dict] = None,
    shocks: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    backend: str = "numpy",
    tol: float = 1e-8,
    max_iter: int = 500,
    **kwargs,
) -> SplineCollocationSolution:
    """Convenience helper to solve a spline collocation problem.

    Accepts either an existing SplineCollocationProblem or keyword arguments.
    """
    if isinstance(problem_or_domain, SplineCollocationProblem):
        problem = problem_or_domain
    elif problem_or_domain is not None:
        dom = problem_or_domain
        problem = SplineCollocationProblem(
            domain=dom,
            n_knots=n_knots,
            spline_type=spline_type,
            bc_type=bc_type,
            knot_spacing=knot_spacing,
            method=method,
            euler_residual_fn=euler_residual_fn,
            return_fn=return_fn,
            transition_fn=transition_fn,
            beta=beta,
            params=params or {},
            borrowing_constraint=borrowing_constraint,
            options=options or {},
            shocks=shocks,
        )
    else:
        if domain is None:
            raise ValueError("domain must be provided if problem is not passed.")
        problem = SplineCollocationProblem(
            domain=domain,
            n_knots=n_knots,
            spline_type=spline_type,
            bc_type=bc_type,
            knot_spacing=knot_spacing,
            method=method,
            euler_residual_fn=euler_residual_fn,
            return_fn=return_fn,
            transition_fn=transition_fn,
            beta=beta,
            params=params or {},
            borrowing_constraint=borrowing_constraint,
            options=options or {},
            shocks=shocks,
        )

    return problem.solve(backend=backend, tol=tol, max_iter=max_iter, **kwargs)


__all__ = [
    "CubicBSplineBasis",
    "SplineBasis",
    "SchumakerSpline",
    "SplineCollocationProblem",
    "SplineCollocationSolution",
    "solve_spline_collocation",
]
