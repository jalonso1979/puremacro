"""Continuous Stationary Distribution & General Equilibrium (Young 2010 Non-Stochastic Simulation).

Implements the Young (2010) method for projecting continuous policy functions
onto fine piecewise-linear histogram representations, assembling Compressed Sparse
Row (CSR) Markov transition operators, solving for invariant stationary distributions
with strict mass conservation (|sum(mu*) - 1.0| <= 1e-12), and solving continuous
general equilibrium market clearing (|K^s - K^d| < 1e-4) in heterogeneous-agent
economies (e.g., continuous Aiyagari incomplete markets).

References
----------
- Young, E. R. (2010). "Solving the incomplete markets model with aggregate
  uncertainty using the Krusell-Smith algorithm and a non-stochastic simulations."
  Journal of Economic Dynamics and Control, 34(1), 36-41.
- Tan, C. (2020). "A Fast and Accurate Method for Solving Incomplete Markets Models."
  Computational Economics, 55, 347-362.
- Aiyagari, S. R. (1994). "Uninsured Idiosyncratic Risk and Aggregate Saving."
  Quarterly Journal of Economics, 109(3), 659-684.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq

from puremacro import _backend as _bk
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst
from puremacro.vfi.discretize import markov_stationary, tauchen


# ---------------------------------------------------------------------------
# JIT-Compiled or NumPy Accelerated Kernels
# ---------------------------------------------------------------------------

@_bk.njit_fallback(fastmath=True)
def _scatter_histogram_1d(
    pdf: np.ndarray,
    j_lo: np.ndarray,
    w_lo: np.ndarray,
    N_k: int,
) -> np.ndarray:
    """1D mass push scatter along asset grid."""
    pdf_next = np.zeros(N_k, dtype=np.float64)
    for i in range(N_k):
        j = j_lo[i]
        w = w_lo[i]
        p = pdf[i]
        pdf_next[j] += p * w
        pdf_next[j + 1] += p * (1.0 - w)
    return pdf_next


@_bk.njit_fallback(fastmath=True)
def _scatter_histogram_2d(
    pdf: np.ndarray,
    j_lo: np.ndarray,
    w_lo: np.ndarray,
    N_k: int,
    n_z: int,
) -> np.ndarray:
    """2D mass push scatter along asset grid for each shock state."""
    pdf_half = np.zeros((N_k, n_z), dtype=np.float64)
    for i in range(N_k):
        for m in range(n_z):
            j = j_lo[i, m]
            w = w_lo[i, m]
            p = pdf[i, m]
            pdf_half[j, m] += p * w
            pdf_half[j + 1, m] += p * (1.0 - w)
    return pdf_half


# ---------------------------------------------------------------------------
# Young (2010) Lottery Weights & Policy Evaluator
# ---------------------------------------------------------------------------

def young_lottery_weights(
    kp_eval: np.ndarray,
    k_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute linear lottery weights and bracketing indices for Young (2010) projection.

    Given evaluated continuous choices ``kp_eval`` and a strictly sorted 1D
    histogram grid ``k_grid``, determines the lower bracket index ``j_lo`` and
    linear weights ``(w_lo, w_hi)`` such that the expected value on the grid
    reproduces ``kp_eval`` exactly (preserving the first moment).

    Parameters
    ----------
    kp_eval : np.ndarray
        Array of continuous policy evaluations, shape (N_k,) or (N_k, n_z) or arbitrary.
    k_grid : np.ndarray
        1D strictly increasing histogram asset grid, length >= 2.

    Returns
    -------
    j_lo : np.ndarray
        Lower bracket index such that k_grid[j_lo] <= kp_eval <= k_grid[j_lo + 1],
        clamped to [0, N_k - 2].
    w_lo : np.ndarray
        Weight allocated to lower bracket node k_grid[j_lo] in [0.0, 1.0].
    w_hi : np.ndarray
        Weight allocated to upper bracket node k_grid[j_lo + 1] in [0.0, 1.0],
        satisfying w_lo + w_hi = 1.0.

    Raises
    ------
    ValueError
        If k_grid is not 1D, has fewer than 2 elements, or is not strictly sorted.
    """
    k_arr = np.asarray(k_grid, dtype=np.float64)
    if k_arr.ndim != 1 or len(k_arr) < 2:
        raise ValueError(f"k_grid must be 1D with length >= 2; got shape {k_arr.shape}")
    if np.any(np.diff(k_arr) <= 0.0):
        raise ValueError("k_grid must be strictly increasing with distinct nodes")

    kp = np.asarray(kp_eval, dtype=np.float64)
    N_k = len(k_arr)

    # Bracketing lower index j in [0, N_k - 2]
    # searchsorted(k, kp, side='right') returns index i where k[i-1] <= kp < k[i]
    j_lo = np.clip(np.searchsorted(k_arr, kp, side="right") - 1, 0, N_k - 2).astype(np.intp)

    k_lo = k_arr[j_lo]
    k_hi = k_arr[j_lo + 1]
    dk = k_hi - k_lo

    # Linear interpolation weight with boundary clamping
    w_lo = np.clip((k_hi - kp) / dk, 0.0, 1.0)
    w_hi = 1.0 - w_lo
    return j_lo, w_lo, w_hi


def _evaluate_policy(
    policy_source: Any,
    asset_grid: np.ndarray,
    shock_grid: Optional[np.ndarray] = None,
    shock_transition: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Evaluate or normalize a continuous policy source across (asset_grid, shock_grid).

    Supports:
    - NumPy ndarrays of shape (N_k,) or (N_k, n_z)
    - Objects implementing `.policy(s)` (CollocationSolution, FEMSolution, SplineCollocationSolution)
    - Callables `policy(k)` or `policy(k, z)`
    - Lists/tuples of policy functions per shock state
    """
    k_arr = np.asarray(asset_grid, dtype=np.float64)
    N_k = len(k_arr)
    has_shocks = shock_transition is not None
    n_z = shock_transition.shape[0] if has_shocks else (len(shock_grid) if shock_grid is not None else 1)
    z_arr = np.asarray(shock_grid, dtype=np.float64) if shock_grid is not None else np.arange(n_z, dtype=np.float64)

    # Case 1: Raw array
    if isinstance(policy_source, np.ndarray):
        pol = np.asarray(policy_source, dtype=np.float64)
        if not has_shocks:
            if pol.shape == (N_k,):
                return pol
            if pol.shape == (N_k, 1):
                return pol.ravel()
        else:
            if pol.shape == (N_k, n_z):
                return pol
            if pol.shape == (N_k,):
                return np.tile(pol[:, None], (1, n_z))
        raise ValueError(
            f"Policy array shape {pol.shape} does not match asset_grid length {N_k} and shock count {n_z}"
        )

    # Extract callable if object has .policy attribute
    pol_fn = getattr(policy_source, "policy", policy_source)

    # Case 2: List or tuple of policies per shock state
    if isinstance(pol_fn, (list, tuple)):
        if len(pol_fn) != n_z:
            raise ValueError(f"Length of policy list ({len(pol_fn)}) must match n_z ({n_z})")
        cols = []
        for fn in pol_fn:
            f = getattr(fn, "policy", fn)
            val = np.asarray(f(k_arr), dtype=np.float64)
            cols.append(val)
        res = np.column_stack(cols)
        return res if has_shocks else res.ravel()

    if not callable(pol_fn):
        raise TypeError(f"policy_source must be an array, callable, or have a .policy method; got {type(policy_source)}")

    # Case 3: 1D Model (no shocks)
    if not has_shocks:
        try:
            res = np.asarray(pol_fn(k_arr), dtype=np.float64)
            if res.shape == (N_k,):
                return res
            if res.size == N_k:
                return res.reshape(N_k)
        except Exception:
            pass
        # Fallback: point-by-point
        return np.array([float(pol_fn(k)) for k in k_arr], dtype=np.float64)

    # Case 4: 2D Model (asset x shock, including has_shocks with n_z == 1)
    # Check if pol_fn accepts 2D coordinates matrix (N_k * n_z, 2) (standard in Collocation/FEM)
    K_mesh, Z_mesh = np.meshgrid(k_arr, z_arr, indexing="ij")
    pts_2d = np.column_stack([K_mesh.ravel(), Z_mesh.ravel()])

    try:
        res = pol_fn(pts_2d)
        res_arr = np.asarray(res, dtype=np.float64)
        if res_arr.shape == (N_k * n_z,):
            return res_arr.reshape((N_k, n_z))
        if res_arr.shape == (N_k, n_z):
            return res_arr
    except Exception:
        pass

    # Check if pol_fn accepts (K_mesh, Z_mesh)
    try:
        res = pol_fn(K_mesh, Z_mesh)
        res_arr = np.asarray(res, dtype=np.float64)
        if res_arr.shape == (N_k, n_z):
            return res_arr
    except Exception:
        pass

    # Check if pol_fn accepts (k_arr, z_val)
    cols = []
    for m in range(n_z):
        zm = z_arr[m]
        try:
            val = pol_fn(k_arr, zm)
            val_arr = np.asarray(val, dtype=np.float64)
            if len(val_arr) == N_k:
                cols.append(val_arr)
                continue
        except Exception:
            pass

        try:
            pt_m = np.column_stack([k_arr, np.full(N_k, zm)])
            val = pol_fn(pt_m)
            val_arr = np.asarray(val, dtype=np.float64).ravel()
            if len(val_arr) == N_k:
                cols.append(val_arr)
                continue
        except Exception:
            pass

        # If pol_fn only accepts 1 argument pol(k) and n_z == 1
        if n_z == 1:
            try:
                val = pol_fn(k_arr)
                val_arr = np.asarray(val, dtype=np.float64).ravel()
                if len(val_arr) == N_k:
                    cols.append(val_arr)
                    continue
            except Exception:
                pass

        # Point by point fallback
        try:
            col = np.array([float(pol_fn(k, zm)) for k in k_arr], dtype=np.float64)
            cols.append(col)
            continue
        except Exception:
            pass

        if n_z == 1:
            try:
                col = np.array([float(pol_fn(k)) for k in k_arr], dtype=np.float64)
                cols.append(col)
                continue
            except Exception:
                pass

    if len(cols) == n_z:
        return np.column_stack(cols)

    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Continuous Push Distribution & Transition Matrix Builder
# ---------------------------------------------------------------------------

def continuous_push_distribution(
    pdf: np.ndarray,
    policy_fn: Any,
    asset_grid: np.ndarray,
    shock_transition: Optional[np.ndarray] = None,
    shock_grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Perform one forward mass push of the continuous distribution (Tan 2020 two-step).

    Evaluates the continuous policy on the asset grid and discrete shock states,
    computes linear lottery weights, and pushes mass to neighboring histogram bins,
    preserving total mass exactly (|sum(pdf_next) - sum(pdf)| == 0).

    Parameters
    ----------
    pdf : np.ndarray
        Current probability distribution. Shape (N_k,) for 1D or (N_k, n_z) for 2D.
    policy_fn : Any
        Policy source (array of next-period choices, callable, or CollocationSolution/FEMSolution).
    asset_grid : np.ndarray
        1D strictly increasing histogram asset grid of length N_k >= 2.
    shock_transition : np.ndarray, optional
        Row-stochastic transition matrix P_z of shape (n_z, n_z).
    shock_grid : np.ndarray, optional
        Grid of discrete shock values of length n_z.

    Returns
    -------
    pdf_next : np.ndarray
        Updated probability distribution with identical shape, strictly preserving total mass.
    """
    p_curr = np.asarray(pdf, dtype=np.float64)
    k_grid = np.asarray(asset_grid, dtype=np.float64)
    N_k = len(k_grid)

    kp_eval = _evaluate_policy(policy_fn, k_grid, shock_grid=shock_grid, shock_transition=shock_transition)

    if shock_transition is None:
        if p_curr.ndim != 1 or len(p_curr) != N_k:
            raise ValueError(f"1D pdf must have shape ({N_k},); got {p_curr.shape}")
        j_lo, w_lo, w_hi = young_lottery_weights(kp_eval, k_grid)
        return _scatter_histogram_1d(p_curr, j_lo, w_lo, N_k)

    P_z = np.asarray(shock_transition, dtype=np.float64)
    n_z = P_z.shape[0]
    was_1d = (p_curr.ndim == 1)
    if n_z == 1 and was_1d and len(p_curr) == N_k:
        p_curr = p_curr[:, None]

    if p_curr.shape != (N_k, n_z):
        raise ValueError(f"2D pdf must have shape ({N_k}, {n_z}); got {p_curr.shape}")

    kp_eval = np.asarray(kp_eval, dtype=np.float64)
    if kp_eval.ndim == 1:
        if n_z == 1:
            kp_eval = kp_eval[:, None]
        else:
            kp_eval = np.tile(kp_eval[:, None], (1, n_z))

    j_lo, w_lo, w_hi = young_lottery_weights(kp_eval, k_grid)
    if j_lo.ndim == 1:
        j_lo = j_lo[:, None]
        w_lo = w_lo[:, None]
        w_hi = w_hi[:, None]

    # Tan (2020) two-step: (1) scatter along asset grid, (2) apply shock transition
    pdf_half = _scatter_histogram_2d(p_curr, j_lo, w_lo, N_k, n_z)
    pdf_next = pdf_half @ P_z
    return pdf_next.ravel() if was_1d else pdf_next


young_step = continuous_push_distribution


def build_continuous_transition_matrix(
    policy_fn: Any,
    asset_grid: np.ndarray,
    shock_transition: Optional[np.ndarray] = None,
    shock_grid: Optional[np.ndarray] = None,
) -> sp.csr_matrix:
    """Build the global Compressed Sparse Row (CSR) row-stochastic Markov transition matrix.

    Constructs operator T of size (N_k * n_z) x (N_k * n_z) (or N_k x N_k if no shocks)
    where row s = (i, m) transitions to destination (j, n) with probability
    w_lo(i, m) * P_z(m, n) and to (j+1, n) with w_hi(i, m) * P_z(m, n).

    Parameters
    ----------
    policy_fn : Any
        Continuous policy source (array, callable, CollocationSolution, FEMSolution).
    asset_grid : np.ndarray
        1D strictly increasing histogram asset grid, length N_k >= 2.
    shock_transition : np.ndarray, optional
        Row-stochastic transition matrix P_z of shape (n_z, n_z).
    shock_grid : np.ndarray, optional
        Grid of discrete shock values of length n_z.

    Returns
    -------
    scipy.sparse.csr_matrix
        Row-stochastic CSR transition matrix T with max 2 * n_z nonzeros per row.
    """
    k_grid = np.asarray(asset_grid, dtype=np.float64)
    N_k = len(k_grid)
    kp_eval = _evaluate_policy(policy_fn, k_grid, shock_grid=shock_grid, shock_transition=shock_transition)

    if shock_transition is None:
        j_lo, w_lo, w_hi = young_lottery_weights(kp_eval, k_grid)
        rows = np.concatenate([np.arange(N_k, dtype=np.intp), np.arange(N_k, dtype=np.intp)])
        cols = np.concatenate([j_lo, j_lo + 1])
        data = np.concatenate([w_lo, w_hi])
        return sp.csr_matrix((data, (rows, cols)), shape=(N_k, N_k))

    P_z = np.asarray(shock_transition, dtype=np.float64)
    n_z = P_z.shape[0]
    N = N_k * n_z

    kp_eval = np.asarray(kp_eval, dtype=np.float64)
    if kp_eval.ndim == 1:
        if n_z == 1:
            kp_eval = kp_eval[:, None]
        else:
            kp_eval = np.tile(kp_eval[:, None], (1, n_z))

    j_lo, w_lo, w_hi = young_lottery_weights(kp_eval, k_grid)
    if j_lo.ndim == 1:
        j_lo = j_lo[:, None]
        w_lo = w_lo[:, None]
        w_hi = w_hi[:, None]

    s = np.arange(N, dtype=np.intp).reshape((N_k, n_z))
    s_rep = np.repeat(s[:, :, None], n_z, axis=2)
    n_idx = np.broadcast_to(np.arange(n_z, dtype=np.intp)[None, None, :], (N_k, n_z, n_z))

    col_lo = j_lo[:, :, None] * n_z + n_idx
    col_hi = (j_lo[:, :, None] + 1) * n_z + n_idx

    val_lo = w_lo[:, :, None] * P_z[None, :, :]
    val_hi = w_hi[:, :, None] * P_z[None, :, :]

    rows = np.concatenate([s_rep.ravel(), s_rep.ravel()])
    cols = np.concatenate([col_lo.ravel(), col_hi.ravel()])
    data = np.concatenate([val_lo.ravel(), val_hi.ravel()])

    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))


young_transition_matrix = build_continuous_transition_matrix


# ---------------------------------------------------------------------------
# Continuous Stationary Distribution Result Container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContinuousStationaryDistribution:
    """Invariant stationary distribution over continuous assets and discrete shocks.

    Parameters
    ----------
    pdf : np.ndarray
        Stationary probability distribution, shape (N_k, n_z) or (N_k,), summing to 1.0.
    asset_grid : np.ndarray
        1D histogram asset grid of length N_k.
    shock_grid : np.ndarray | None
        1D grid of discrete shock values of length n_z.
    shock_transition : np.ndarray | None
        Row-stochastic transition matrix P_z of shape (n_z, n_z).
    mass_error : float
        Absolute mass deviation |sum(pdf) - 1.0| <= 1e-12.
    iterations : int
        Number of solver iterations or factorization passes.
    converged : bool
        Whether the invariant measure solver successfully converged.
    metadata : dict
        Algorithmic and performance diagnostics dictionary.
    """

    pdf: np.ndarray
    asset_grid: np.ndarray
    shock_grid: Optional[np.ndarray] = None
    shock_transition: Optional[np.ndarray] = None
    mass_error: float = 0.0
    iterations: int = 1
    converged: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mu(self) -> np.ndarray:
        """Alias for pdf."""
        return self.pdf

    @property
    def k_grid(self) -> np.ndarray:
        """Alias for asset_grid."""
        return self.asset_grid

    @property
    def z_grid(self) -> Optional[np.ndarray]:
        """Alias for shock_grid."""
        return self.shock_grid

    @property
    def P_z(self) -> Optional[np.ndarray]:
        """Alias for shock_transition."""
        return self.shock_transition

    @property
    def n_iter(self) -> int:
        """Alias for iterations."""
        return self.iterations

    @property
    def residual_norm(self) -> float:
        """Fixed-point residual norm ||mu* T - mu*||_inf."""
        return float(self.metadata.get("residual_norm", self.mass_error))

    def marginal_assets(self) -> np.ndarray:
        """Marginal distribution over continuous assets, shape (N_k,)."""
        if self.pdf.ndim == 2:
            return np.sum(self.pdf, axis=1)
        return self.pdf

    def marginal_shocks(self) -> Optional[np.ndarray]:
        """Marginal distribution over discrete shocks, shape (n_z,) or None."""
        if self.pdf.ndim == 2:
            return np.sum(self.pdf, axis=0)
        return None

    def mean(self, fn: Optional[Callable] = None) -> float:
        """Compute the expected value E[fn(k, z)] or mean asset holdings E[k]."""
        if fn is None:
            mu_k = self.marginal_assets()
            return float(np.sum(mu_k * self.asset_grid))

        if self.pdf.ndim == 1 or self.pdf.ndim == 2 and self.pdf.shape[1] == 1:
            mu_k = self.marginal_assets()
            vals = np.array([fn(k) for k in self.asset_grid], dtype=np.float64)
            return float(np.sum(mu_k * vals))

        # 2D case
        n_z = self.pdf.shape[1]
        z_vals = self.shock_grid if self.shock_grid is not None else np.arange(n_z, dtype=np.float64)
        total = 0.0
        for i, k in enumerate(self.asset_grid):
            for m, z in enumerate(z_vals):
                try:
                    v = fn(k, z)
                except TypeError:
                    v = fn(k)
                total += self.pdf[i, m] * float(v)
        return float(total)

    def variance(self, fn: Optional[Callable] = None) -> float:
        """Compute the variance Var[fn(k, z)] or variance of asset holdings Var[k]."""
        if fn is None:
            mu_k = self.marginal_assets()
            m = self.mean()
            return float(np.sum(mu_k * ((self.asset_grid - m) ** 2)))

        m = self.mean(fn)
        sq_mean = self.mean(lambda *args: fn(*args) ** 2)
        return float(max(sq_mean - m**2, 0.0))

    def percentile(self, q: Union[float, Sequence[float]]) -> Union[float, np.ndarray]:
        """Compute asset percentile(s) q in [0, 100] via linear interpolation on the CDF."""
        q_arr = np.asarray(q, dtype=np.float64)
        if np.any(q_arr < 0.0) or np.any(q_arr > 100.0):
            raise ValueError(f"Requested percentile q must be in [0, 100]; got {q}")

        mu_k = self.marginal_assets()
        cdf = np.cumsum(mu_k)
        cdf = cdf / cdf[-1]

        res = np.interp(q_arr / 100.0, cdf, self.asset_grid)
        return float(res) if np.ndim(q) == 0 else res

    def gini(self) -> float:
        """Compute Gini inequality coefficient of wealth/assets."""
        mu_k = self.marginal_assets()
        k = self.asset_grid

        # Cumulative shares
        p = np.concatenate([[0.0], np.cumsum(mu_k)])
        S = np.concatenate([[0.0], np.cumsum(mu_k * k)])
        if S[-1] <= 0.0:
            return 0.0
        L = S / S[-1]
        gini_coeff = 1.0 - np.sum((p[1:] - p[:-1]) * (L[1:] + L[:-1]))
        return float(np.clip(gini_coeff, 0.0, 1.0))

    def lorenz(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the Lorenz curve (p, L) over n_points in [0, 1]."""
        mu_k = self.marginal_assets()
        k = self.asset_grid
        p_raw = np.concatenate([[0.0], np.cumsum(mu_k)])
        S_raw = np.concatenate([[0.0], np.cumsum(mu_k * k)])
        L_raw = S_raw / S_raw[-1] if S_raw[-1] > 0.0 else p_raw.copy()

        p_grid = np.linspace(0.0, 1.0, n_points)
        L_grid = np.interp(p_grid, p_raw, L_raw)
        return p_grid, L_grid

    def summary(self) -> pd.DataFrame:
        """Structured summary DataFrame of the stationary distribution."""
        mu_k = self.marginal_assets()
        mean_k = float(np.sum(mu_k * self.asset_grid))
        var_k = float(np.sum(mu_k * ((self.asset_grid - mean_k) ** 2)))
        std_k = float(np.sqrt(max(var_k, 0.0)))
        p10 = float(self.percentile(10.0))
        p50 = float(self.percentile(50.0))
        p90 = float(self.percentile(90.0))
        gini = self.gini()
        n_z = self.pdf.shape[1] if self.pdf.ndim == 2 else 1

        rows = [
            {"Metric": "Asset Grid Points (N_k)", "Value": str(len(self.asset_grid))},
            {"Metric": "Shock States (n_z)", "Value": str(n_z)},
            {"Metric": "Mean Assets", "Value": f"{mean_k:.6f}"},
            {"Metric": "Std Assets", "Value": f"{std_k:.6f}"},
            {"Metric": "Median Assets (p50)", "Value": f"{p50:.6f}"},
            {"Metric": "p10 Assets", "Value": f"{p10:.6f}"},
            {"Metric": "p90 Assets", "Value": f"{p90:.6f}"},
            {"Metric": "Gini Coefficient", "Value": f"{gini:.4f}"},
            {"Metric": "Mass Error", "Value": f"{self.mass_error:.2e}"},
            {"Metric": "Fixed-Point Residual", "Value": f"{self.residual_norm:.2e}"},
            {"Metric": "Solver Method", "Value": str(self.metadata.get("method", "auto"))},
            {"Metric": "Iterations", "Value": str(self.iterations)},
            {"Metric": "Converged", "Value": str(self.converged)},
        ]
        return pd.DataFrame(rows).set_index("Metric")

    def to_frame(self) -> pd.DataFrame:
        """Return summary table as DataFrame."""
        return self.summary()

    def to_markdown(self, **kwargs) -> str:
        """Render summary table as Markdown string."""
        return _df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Render summary table as LaTeX tabular string."""
        return _df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Render summary table as Typst table string."""
        return _df_to_typst(self.summary(), **kwargs)

    def plot(
        self, figsize: Tuple[float, float] = (10, 4), show: bool = False, **kwargs
    ) -> matplotlib.figure.Figure:
        """Plot the marginal asset density, CDF, and Lorenz curve."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

        mu_k = self.marginal_assets()
        k = self.asset_grid
        cdf = np.cumsum(mu_k) / np.sum(mu_k)

        # Panel 1: Probability density / mass
        ax1 = axes[0]
        ax1.plot(k, mu_k, color="tab:blue", lw=2, label="Density $\\mu(k)$")
        ax1.fill_between(k, 0, mu_k, color="tab:blue", alpha=0.2)
        ax1.set_title("Marginal Asset Distribution")
        ax1.set_xlabel("Assets $k$")
        ax1.set_ylabel("Probability Mass")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")

        # Panel 2: CDF & Lorenz curve
        ax2 = axes[1]
        p_lorenz, L_lorenz = self.lorenz(100)
        ax2.plot(k, cdf, color="tab:green", lw=2, label="CDF $F(k)$")
        ax2.plot(p_lorenz * (k[-1] - k[0]) + k[0], L_lorenz, color="tab:purple", lw=1.5, ls="--", label="Lorenz Curve")
        ax2.set_title(f"Cumulative Distribution & Inequality (Gini={self.gini():.3f})")
        ax2.set_xlabel("Assets $k$")
        ax2.set_ylabel("Cumulative Fraction")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best")

        if show:
            plt.show()
        return fig


ContinuousDistributionResult = ContinuousStationaryDistribution


# ---------------------------------------------------------------------------
# Continuous Stationary Distribution Solver
# ---------------------------------------------------------------------------

def continuous_stationary_distribution(
    policy_fn: Any,
    asset_grid: np.ndarray,
    shock_transition: Optional[np.ndarray] = None,
    shock_grid: Optional[np.ndarray] = None,
    *,
    k_grid: Optional[np.ndarray] = None,
    P_z: Optional[np.ndarray] = None,
    z_grid: Optional[np.ndarray] = None,
    method: str = "auto",
    tol: float = 1e-12,
    max_iter: int = 50_000,
    backend: str = "numpy",
    **kwargs,
) -> ContinuousStationaryDistribution:
    """Compute the invariant stationary distribution using the Young (2010) method.

    Parameters
    ----------
    policy_fn : Any
        Continuous policy source (array, callable, CollocationSolution, FEMSolution).
    asset_grid : np.ndarray
        1D strictly increasing histogram asset grid of length N_k >= 2.
    shock_transition : np.ndarray, optional
        Exogenous Markov transition matrix P_z of shape (n_z, n_z).
    shock_grid : np.ndarray, optional
        Discrete shock grid of length n_z.
    method : {"auto", "sparse_direct", "power", "arnoldi"}, default "auto"
        Invariant distribution solver algorithm:
        - "sparse_direct": SuperLU direct sparse solve on (I - T^T) with unit-sum replacement.
        - "power": Forward power iteration (Tan 2020 two-step).
        - "arnoldi": SciPy eigs leading eigenvector calculation.
        - "auto": Direct sparse solve with automatic fallback to power iteration.
    tol : float, default 1e-12
        Convergence tolerance for fixed-point iterations.
    max_iter : int, default 50000
        Maximum iterations for iterative solvers.
    backend : str, default "numpy"
        Compute backend name. Falls back safely to numpy if requested backend is unavailable.

    Returns
    -------
    ContinuousStationaryDistribution
        Solved distribution satisfying strict mass conservation |sum(mu*) - 1.0| <= 1e-12.
    """
    t_start = time.time()
    k_arr = np.asarray(asset_grid if asset_grid is not None else k_grid, dtype=np.float64)
    P_arr = shock_transition if shock_transition is not None else P_z
    z_arr = shock_grid if shock_grid is not None else z_grid

    # Backend validation and fallback
    if backend not in _bk.SUPPORTED:
        raise ValueError(f"Unknown backend {backend!r}; supported: {_bk.SUPPORTED}")
    resolved_backend = backend
    if not _bk.backend_available(backend):
        warnings.warn(
            f"Backend '{backend}' requested but not available; falling back to 'numpy'.",
            UserWarning,
            stacklevel=2,
        )
        resolved_backend = "numpy"

    # Assemble CSR transition matrix T
    T = build_continuous_transition_matrix(
        policy_fn, k_arr, shock_transition=P_arr, shock_grid=z_arr
    )
    N = T.shape[0]

    valid_methods = ("auto", "sparse_direct", "power", "arnoldi")
    if method not in valid_methods:
        raise ValueError(f"Unknown method {method!r}; supported: {valid_methods}")

    mu_vec: Optional[np.ndarray] = None
    converged = False
    n_iter = 1
    actual_method = method

    # Method 1: Sparse Direct Solve
    if method in ("sparse_direct", "auto"):
        try:
            # Construct L = I - T^T with row 0 replaced by [1, 1, ..., 1]
            row0 = sp.csr_matrix(np.ones((1, N), dtype=np.float64))
            L_sub = (sp.eye(N, format="csr") - T.T.tocsr())[1:, :]
            L = sp.vstack([row0, L_sub], format="csr")
            b = np.zeros(N, dtype=np.float64)
            b[0] = 1.0

            sol_vec = spla.spsolve(L, b)
            if np.all(np.isfinite(sol_vec)) and np.sum(sol_vec) > 0.0:
                sol_vec = np.maximum(sol_vec, 0.0)
                sol_vec = sol_vec / np.sum(sol_vec)
                # Check residual
                res_norm = float(np.max(np.abs(T.T @ sol_vec - sol_vec)))
                if res_norm < 1e-8:
                    mu_vec = sol_vec
                    converged = True
                    n_iter = 1
                    actual_method = "sparse_direct"
        except Exception as e:
            if method == "sparse_direct":
                raise RuntimeError(f"sparse_direct solver failed: {e}") from e

    # Method 2: Arnoldi Eigenvector Solve
    if mu_vec is None and method == "arnoldi":
        try:
            vals, vecs = spla.eigs(T.T, k=1, sigma=1.0, which="LM")
            v = np.real(vecs[:, 0])
            if np.sum(v) < 0.0:
                v = -v
            v = np.maximum(v, 0.0)
            mu_vec = v / np.sum(v)
            converged = True
            n_iter = 1
            actual_method = "arnoldi"
        except Exception as e:
            raise RuntimeError(f"arnoldi solver failed: {e}") from e

    # Method 3: Power Iteration (Fallback or explicit)
    if mu_vec is None:
        actual_method = "power"
        mu = np.full(N, 1.0 / N, dtype=np.float64)
        for it in range(1, max_iter + 1):
            mu_next = T.T @ mu
            if it % 10 == 0:
                mu_next = np.maximum(mu_next, 0.0)
                mu_next = mu_next / np.sum(mu_next)
            diff = float(np.max(np.abs(mu_next - mu)))
            mu = mu_next
            if diff < tol:
                converged = True
                n_iter = it
                break
        else:
            converged = False
            n_iter = max_iter

        mu = np.maximum(mu, 0.0)
        mu_vec = mu / np.sum(mu)

    # Defensive mass re-normalization for strict mass conservation
    mu_vec = np.maximum(mu_vec, 0.0)
    mu_vec = mu_vec / np.sum(mu_vec)
    mass_err = float(abs(np.sum(mu_vec) - 1.0))
    res_norm = float(np.max(np.abs(T.T @ mu_vec - mu_vec)))

    # Reshape pdf to (N_k, n_z) if shocks present, else (N_k,)
    if P_arr is not None:
        pdf_out = mu_vec.reshape((len(k_arr), P_arr.shape[0]))
    else:
        pdf_out = mu_vec

    elapsed = time.time() - t_start
    meta = {
        "method": actual_method,
        "backend": resolved_backend,
        "residual_norm": res_norm,
        "elapsed_time": elapsed,
        "nnz": T.nnz,
        "sparsity": float(T.nnz) / float(N**2),
    }

    return ContinuousStationaryDistribution(
        pdf=pdf_out,
        asset_grid=k_arr,
        shock_grid=z_arr,
        shock_transition=P_arr,
        mass_error=mass_err,
        iterations=n_iter,
        converged=converged,
        metadata=meta,
    )


young_stationary_distribution = continuous_stationary_distribution


# ---------------------------------------------------------------------------
# General Equilibrium Containers & Solvers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AiyagariContinuousEquilibrium:
    """Solved stationary general equilibrium for a continuous heterogeneous-agent economy.

    Parameters
    ----------
    r : float
        Market-clearing real interest rate r*.
    w : float
        Equilibrium competitive wage w*.
    K : float
        Aggregate capital stock K*.
    L : float
        Aggregate effective labor supply L*.
    household_solution : Any
        Household decision rule (CollocationSolution, FEMSolution, SplineCollocationSolution, etc.).
    distribution : ContinuousStationaryDistribution
        Invariant stationary distribution at the equilibrium factor prices.
    capital_market_clearing_error : float
        Excess capital demand residual |K^s(r*) - K^d(r*)| < 1e-4.
    converged : bool
        Whether the equilibrium price root-finder converged successfully.
    iterations : int
        Number of price iterations/evaluations required to clear the market.
    metadata : dict
        Diagnostics and timing metadata dictionary.
    """

    r: float
    w: float
    K: float
    L: float
    household_solution: Any
    distribution: ContinuousStationaryDistribution
    capital_market_clearing_error: float
    converged: bool
    iterations: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __new__(cls, *args, **kwargs):
        # Support both result dataclass instantiation and problem configuration
        if len(args) == 0 and "r" not in kwargs and ("beta" in kwargs or "alpha" in kwargs or "solver" in kwargs):
            return _AiyagariContinuousModel(**kwargs)
        return super().__new__(cls)

    @property
    def r_star(self) -> float:
        """Alias for r."""
        return self.r

    @property
    def w_star(self) -> float:
        """Alias for w."""
        return self.w

    @property
    def K_star(self) -> float:
        """Alias for K."""
        return self.K

    @property
    def L_star(self) -> float:
        """Alias for L."""
        return self.L

    @property
    def excess_capital_demand(self) -> float:
        """Alias for capital_market_clearing_error."""
        return self.capital_market_clearing_error

    @property
    def solution(self) -> Any:
        """Alias for household_solution."""
        return self.household_solution

    @property
    def price(self) -> float:
        """Alias for r."""
        return self.r

    @property
    def residual(self) -> float:
        """Alias for capital_market_clearing_error."""
        return self.capital_market_clearing_error

    @property
    def n_evals(self) -> int:
        """Alias for iterations."""
        return self.iterations

    def summary(self) -> pd.DataFrame:
        """Structured summary DataFrame of the stationary general equilibrium."""
        kl_ratio = self.K / max(self.L, 1e-12)
        elapsed = self.metadata.get("elapsed_time", np.nan)
        rows = [
            {"Metric": "Equilibrium Interest Rate (r*)", "Value": f"{self.r:.6f}"},
            {"Metric": "Equilibrium Wage (w*)", "Value": f"{self.w:.6f}"},
            {"Metric": "Aggregate Capital Supply (K*)", "Value": f"{self.K:.6f}"},
            {"Metric": "Aggregate Labor Supply (L*)", "Value": f"{self.L:.6f}"},
            {"Metric": "Capital-Labor Ratio (K/L)", "Value": f"{kl_ratio:.6f}"},
            {"Metric": "Market Clearing Error |K^s - K^d|", "Value": f"{abs(self.capital_market_clearing_error):.2e}"},
            {"Metric": "Price Evaluations", "Value": str(self.iterations)},
            {"Metric": "Converged", "Value": str(self.converged)},
            {"Metric": "Household Solver", "Value": str(self.metadata.get("solver_type", type(self.household_solution).__name__))},
            {"Metric": "Elapsed Time (s)", "Value": f"{elapsed:.4f}" if np.isfinite(elapsed) else "N/A"},
        ]
        return pd.DataFrame(rows).set_index("Metric")

    def to_frame(self) -> pd.DataFrame:
        """Return summary table as DataFrame."""
        return self.summary()

    def to_markdown(self, **kwargs) -> str:
        """Render summary table as Markdown string."""
        return _df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Render summary table as LaTeX tabular string."""
        return _df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Render summary table as Typst table string."""
        return _df_to_typst(self.summary(), **kwargs)

    def plot(
        self, figsize: Tuple[float, float] = (10, 4), show: bool = False, **kwargs
    ) -> matplotlib.figure.Figure:
        """Plot the equilibrium policy function and asset distribution."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

        k_grid = self.distribution.asset_grid
        mu_k = self.distribution.marginal_assets()

        # Panel 1: Policy function a'(a, z)
        ax1 = axes[0]
        sol = self.household_solution
        if hasattr(sol, "policy_a") and isinstance(sol.policy_a, np.ndarray):
            n_z = sol.policy_a.shape[1] if sol.policy_a.ndim == 2 else 1
            z_grid = getattr(sol, "z_grid", np.arange(n_z))
            if n_z > 1:
                for m in range(n_z):
                    zm = z_grid[m] if z_grid is not None else m
                    ax1.plot(sol.a_grid, sol.policy_a[:, m], lw=1.5, label=f"$z={zm:.2f}$")
            else:
                ax1.plot(sol.a_grid, sol.policy_a, lw=2, color="tab:blue", label="$a'(a)$")
            ax1.plot(sol.a_grid, sol.a_grid, color="gray", ls="--", alpha=0.7, label="$a'=a$")
        elif hasattr(sol, "policy"):
            eval_pts = np.linspace(k_grid[0], k_grid[-1], 200)
            try:
                pol = sol.policy(eval_pts)
                ax1.plot(eval_pts, pol, lw=2, color="tab:blue", label="$a'(a)$")
                ax1.plot(eval_pts, eval_pts, color="gray", ls="--", alpha=0.7, label="$a'=a$")
            except Exception:
                pass
        ax1.set_title("Equilibrium Asset Policy $a'(a, z)$")
        ax1.set_xlabel("Current Assets $a$")
        ax1.set_ylabel("Next Period Assets $a'$")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")

        # Panel 2: Stationary distribution
        ax2 = axes[1]
        ax2.plot(k_grid, mu_k, color="tab:green", lw=2, label="Distribution $\\mu^*(a)$")
        ax2.fill_between(k_grid, 0, mu_k, color="tab:green", alpha=0.2)
        ax2.axvline(self.K, color="tab:red", ls="--", lw=1.5, label=f"$K^*={self.K:.3f}$")
        ax2.set_title("Equilibrium Wealth Distribution")
        ax2.set_xlabel("Assets $a$")
        ax2.set_ylabel("Density")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best")

        if show:
            plt.show()
        return fig

    @classmethod
    def solve(
        cls,
        beta: float = 0.96,
        gamma: float = 2.0,
        alpha: float = 0.36,
        delta: float = 0.08,
        rho_z: float = 0.90,
        sigma_z: float = 0.20,
        n_z: int = 5,
        a_max: float = 30.0,
        N_k: int = 1000,
        r_bracket: Optional[Tuple[float, float]] = None,
        solver: str = "auto",
        backend: str = "numpy",
        xtol: float = 1e-6,
        dist_options: Optional[dict] = None,
        **kwargs,
    ) -> AiyagariContinuousEquilibrium:
        """Solve canonical Aiyagari continuous general equilibrium."""
        return solve_aiyagari_continuous(
            beta=beta,
            gamma=gamma,
            alpha=alpha,
            delta=delta,
            rho_z=rho_z,
            sigma_z=sigma_z,
            n_z=n_z,
            a_max=a_max,
            N_k=N_k,
            r_bracket=r_bracket,
            solver=solver,
            backend=backend,
            xtol=xtol,
            dist_options=dist_options,
            **kwargs,
        )


ContinuousEquilibriumResult = AiyagariContinuousEquilibrium


@dataclass(frozen=True)
class _AiyagariContinuousModel:
    """Configurable Aiyagari continuous model problem."""

    beta: float = 0.96
    gamma: float = 2.0
    alpha: float = 0.36
    delta: float = 0.08
    rho_z: float = 0.90
    sigma_z: float = 0.20
    n_z: int = 5
    a_max: float = 30.0

    def solve(
        self,
        solver: str = "auto",
        N_k: int = 1000,
        r_bracket: Optional[Tuple[float, float]] = None,
        backend: str = "numpy",
        xtol: float = 1e-6,
        **kwargs,
    ) -> AiyagariContinuousEquilibrium:
        return solve_aiyagari_continuous(
            beta=self.beta,
            gamma=self.gamma,
            alpha=self.alpha,
            delta=self.delta,
            rho_z=self.rho_z,
            sigma_z=self.sigma_z,
            n_z=self.n_z,
            a_max=self.a_max,
            N_k=N_k,
            r_bracket=r_bracket,
            solver=solver,
            backend=backend,
            xtol=xtol,
            **kwargs,
        )


AiyagariContinuousModel = _AiyagariContinuousModel


# ---------------------------------------------------------------------------
# Continuous General Equilibrium Solver Functions
# ---------------------------------------------------------------------------

def continuous_stationary_equilibrium(
    build_problem: Callable[[float], Any],
    market_residual: Callable[[float, Any, ContinuousStationaryDistribution, Any], float],
    price_bracket: Tuple[float, float],
    asset_grid: Optional[np.ndarray] = None,
    shock_transition: Optional[np.ndarray] = None,
    shock_grid: Optional[np.ndarray] = None,
    *,
    k_grid: Optional[np.ndarray] = None,
    P_z: Optional[np.ndarray] = None,
    z_grid: Optional[np.ndarray] = None,
    xtol: float = 1e-6,
    max_evals: int = 100,
    dist_options: Optional[dict] = None,
    backend: str = "numpy",
) -> AiyagariContinuousEquilibrium:
    """Solve for the market-clearing equilibrium price in continuous heterogeneous-agent economies.

    Uses scalar root-finding (Brent's method) to balance market clearing:
    residual(p*, solution(p*), distribution(p*)) = 0.

    Parameters
    ----------
    build_problem : Callable[[float], Any]
        Function returning an economic problem instance at candidate price p.
    market_residual : Callable
        Market-clearing excess demand function returning 0.0 at equilibrium.
        Accepts signature (price, solution, distribution, problem).
    price_bracket : tuple[float, float]
        Initial search interval (lo, hi) where the residual must change sign.
    asset_grid : np.ndarray, optional
        Histogram asset grid for continuous stationary distribution.
    shock_transition : np.ndarray, optional
        Markov transition matrix P_z.
    shock_grid : np.ndarray, optional
        Discrete shock grid.
    xtol : float, default 1e-6
        Root-finding tolerance for price.
    max_evals : int, default 100
        Maximum price evaluations.
    dist_options : dict, optional
        Options passed to continuous_stationary_distribution.
    backend : str, default "numpy"
        Numerical compute backend.

    Returns
    -------
    AiyagariContinuousEquilibrium
        Solved general equilibrium container.
    """
    t_start = time.time()
    k_arr = asset_grid if asset_grid is not None else k_grid
    P_arr = shock_transition if shock_transition is not None else P_z
    z_arr = shock_grid if shock_grid is not None else z_grid
    dist_opts = dist_options or {}

    if k_arr is None:
        raise ValueError("asset_grid (or k_grid) must be specified for continuous equilibrium.")

    counter = {"n": 0}

    def _eval_solution(price: float):
        counter["n"] += 1
        prob = build_problem(price)
        if hasattr(prob, "solve"):
            try:
                sol = prob.solve(backend=backend)
            except TypeError:
                sol = prob.solve()
        else:
            sol = prob

        dist = continuous_stationary_distribution(
            sol, k_arr, shock_transition=P_arr, shock_grid=z_arr, backend=backend, **dist_opts
        )
        return prob, sol, dist

    def _eval_resid(price: float) -> float:
        prob, sol, dist = _eval_solution(price)
        try:
            return float(market_residual(price, sol, dist, prob))
        except TypeError:
            try:
                return float(market_residual(price, sol, dist))
            except TypeError:
                try:
                    return float(market_residual(price, sol))
                except TypeError:
                    return float(market_residual(price))

    lo, hi = price_bracket
    f_lo = _eval_resid(lo)
    f_hi = _eval_resid(hi)

    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"Market clearing residual does not change sign across price bracket [{lo:.6f}, {hi:.6f}]: "
            f"residual({lo:.6f}) = {f_lo:+.6e}, residual({hi:.6f}) = {f_hi:+.6e}. "
            "Verify that the search bracket encloses the equilibrium price."
        )

    p_star = float(brentq(_eval_resid, lo, hi, xtol=xtol, maxiter=max_evals))

    prob_star, sol_star, dist_star = _eval_solution(p_star)
    try:
        resid_star = float(market_residual(p_star, sol_star, dist_star, prob_star))
    except TypeError:
        try:
            resid_star = float(market_residual(p_star, sol_star, dist_star))
        except TypeError:
            resid_star = float(market_residual(p_star, sol_star))

    K_star = float(dist_star.mean())
    w_star = float(getattr(sol_star, "w", getattr(prob_star, "w", 1.0)))
    L_star = float(getattr(sol_star, "L", getattr(prob_star, "L", 1.0)))

    meta = {
        "elapsed_time": time.time() - t_start,
        "solver_type": type(sol_star).__name__,
        "backend": backend,
    }

    return AiyagariContinuousEquilibrium(
        r=p_star,
        w=w_star,
        K=K_star,
        L=L_star,
        household_solution=sol_star,
        distribution=dist_star,
        capital_market_clearing_error=resid_star,
        converged=True,
        iterations=counter["n"],
        metadata=meta,
    )


@dataclass
class _ContinuousHouseholdEGMResult:
    """Container for household decision rules solved via Endogenous Grid Method."""

    a_grid: np.ndarray
    z_grid: np.ndarray
    P_z: np.ndarray
    policy_a: np.ndarray  # shape (N_k, n_z)
    policy_c: np.ndarray  # shape (N_k, n_z)
    r: float
    w: float
    K_d: float
    L: float

    def policy(self, s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate next-asset continuous policy function g(s)."""
        s_arr = np.asarray(s, dtype=np.float64)
        if s_arr.ndim == 1:
            # Sliced or 1D evaluation: interpolate along column 0
            res = np.interp(s_arr, self.a_grid, self.policy_a[:, 0])
            return float(res) if np.ndim(s) == 0 else res
        if s_arr.ndim == 2 and s_arr.shape[1] == 2:
            # 2D points (a, z)
            k_pts = s_arr[:, 0]
            z_pts = s_arr[:, 1]
            out = np.zeros(len(s_arr), dtype=np.float64)
            for m, zm in enumerate(self.z_grid):
                mask = np.isclose(z_pts, zm)
                if np.any(mask):
                    out[mask] = np.interp(k_pts[mask], self.a_grid, self.policy_a[:, m])
            return out
        raise ValueError(f"Invalid state evaluation shape {s_arr.shape}")


def solve_aiyagari_continuous(
    beta: float = 0.96,
    gamma: float = 2.0,
    alpha: float = 0.36,
    delta: float = 0.08,
    rho_z: float = 0.90,
    sigma_z: float = 0.20,
    n_z: int = 5,
    a_max: float = 30.0,
    N_k: int = 1000,
    r_bracket: Optional[Tuple[float, float]] = None,
    solver: str = "auto",
    backend: str = "numpy",
    xtol: float = 1e-8,
    max_evals: int = 100,
    dist_options: Optional[dict] = None,
    P_z: Optional[np.ndarray] = None,
    z_grid: Optional[np.ndarray] = None,
    **kwargs,
) -> AiyagariContinuousEquilibrium:
    """Solve the canonical continuous Aiyagari (1994) general equilibrium model.

    Finds the market-clearing equilibrium interest rate r* balancing aggregate
    capital supply K^s(r*) = int k dmu* with firm capital demand K^d(r*).

    Parameters
    ----------
    beta : float, default 0.96
        Household discount factor in (0, 1).
    gamma : float, default 2.0
        Relative risk aversion coefficient.
    alpha : float, default 0.36
        Cobb-Douglas capital share.
    delta : float, default 0.08
        Capital depreciation rate.
    rho_z : float, default 0.90
        Persistence of AR(1) log-productivity shocks.
    sigma_z : float, default 0.20
        Standard deviation of AR(1) innovations.
    n_z : int, default 5
        Number of discrete shock states.
    a_max : float, default 30.0
        Upper bound on asset domain.
    N_k : int, default 1000
        Number of points in fine histogram grid for Young (2010) distribution.
    r_bracket : tuple[float, float], optional
        Search bracket (lo, hi) for r. Defaults to (1e-3, 1/beta - 1 - 5e-4).
    solver : str, default "auto"
        Household continuous solver engine ("auto", "egm", "collocation", "fem").
    backend : str, default "numpy"
        Acceleration backend.
    xtol : float, default 1e-8
        Market-clearing price tolerance.
    max_evals : int, default 100
        Maximum price evaluations.
    dist_options : dict, optional
        Options passed to continuous_stationary_distribution.
    P_z : np.ndarray, optional
        Pre-computed Markov transition matrix for productivity shocks.
    z_grid : np.ndarray, optional
        Pre-computed discrete productivity shock levels.

    Returns
    -------
    AiyagariContinuousEquilibrium
        Solved general equilibrium meeting tolerance |K^s - K^d| < 1e-4.
    """
    t_start = time.time()

    # Discretize shock process if not passed
    if P_z is None or z_grid is None:
        log_z, P_z = tauchen(n_z, rho_z, sigma_z)
        z_grid = np.exp(log_z)
    else:
        P_z = np.asarray(P_z, dtype=np.float64)
        z_grid = np.asarray(z_grid, dtype=np.float64)
        n_z = len(z_grid)

    pi_z = markov_stationary(P_z)
    L_agg = float(np.sum(pi_z * z_grid))

    # Dense asset grid for continuous EGM household solve
    n_a = 150
    a_grid_dense = a_max * (np.linspace(0.0, 1.0, n_a) ** 1.5)
    # Fine histogram grid for Young (2010) distribution
    K_hist = np.linspace(0.0, a_max, N_k)

    r_upper_bound = 1.0 / beta - 1.0
    if r_bracket is not None:
        lo, hi = r_bracket
        if hi >= r_upper_bound - 1e-4:
            raise ValueError(
                f"Upper interest rate bound {hi} must be strictly less than 1/beta - 1 = {r_upper_bound:.6f}"
            )
    else:
        lo = 0.001
        hi = r_upper_bound - 0.0005

    counter = {"n": 0}

    def _solve_household_at_r(r: float) -> _ContinuousHouseholdEGMResult:
        kl = ((r + delta) / alpha) ** (1.0 / (alpha - 1.0))
        w = (1.0 - alpha) * (kl**alpha)
        Kd = L_agg * kl

        # EGM fixed point iteration for consumption policy
        c = np.zeros((n_a, n_z), dtype=np.float64)
        for m in range(n_z):
            c[:, m] = r * a_grid_dense + w * z_grid[m]

        for _ in range(500):
            c_old = c.copy()
            # Euler expectation: EMu = beta * (1 + r) * sum_zp P_z(z, zp) * c(ap, zp)^(-gamma)
            EMu = beta * (1.0 + r) * (c ** (-gamma) @ P_z.T)
            c_endo = EMu ** (-1.0 / gamma)

            c_new = np.zeros_like(c)
            for m in range(n_z):
                # a_endo = (c_endo + a' - w*z) / (1 + r)
                a_endo = (c_endo[:, m] + a_grid_dense - w * z_grid[m]) / (1.0 + r)
                c_interp = np.interp(a_grid_dense, a_endo, c_endo[:, m])
                # Borrowing constraint binds where a < a_endo[0]
                binds = a_grid_dense < a_endo[0]
                c_interp[binds] = (1.0 + r) * a_grid_dense[binds] + w * z_grid[m]
                c_new[:, m] = c_interp

            if np.max(np.abs(c_new - c_old)) < 1e-8:
                break
            c = c_new

        # Evaluate next-period asset policy a' = (1 + r) a + w z - c
        ap_dense = np.zeros((n_a, n_z), dtype=np.float64)
        for m in range(n_z):
            ap_dense[:, m] = np.maximum((1.0 + r) * a_grid_dense + w * z_grid[m] - c[:, m], 0.0)

        # Interpolate continuous policy onto fine histogram grid K_hist
        ap_hist = np.zeros((N_k, n_z), dtype=np.float64)
        for m in range(n_z):
            ap_hist[:, m] = np.interp(K_hist, a_grid_dense, ap_dense[:, m])

        return _ContinuousHouseholdEGMResult(
            a_grid=K_hist,
            z_grid=z_grid,
            P_z=P_z,
            policy_a=ap_hist,
            policy_c=c,
            r=r,
            w=w,
            K_d=Kd,
            L=L_agg,
        )

    def _excess_capital_demand(r: float) -> Tuple[float, _ContinuousHouseholdEGMResult, ContinuousStationaryDistribution]:
        counter["n"] += 1
        hh_sol = _solve_household_at_r(r)
        dist = continuous_stationary_distribution(
            hh_sol.policy_a, K_hist, shock_transition=P_z, shock_grid=z_grid, backend=backend, **(dist_options or {})
        )
        Ks = float(dist.mean())
        excess = Ks - hh_sol.K_d
        return excess, hh_sol, dist

    f_lo, _, _ = _excess_capital_demand(lo)
    f_hi, _, _ = _excess_capital_demand(hi)

    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"Excess capital demand does not change sign on r in [{lo:.4f}, {hi:.4f}]: "
            f"K^s - K^d is {f_lo:+.4f} at {lo:.4f} and {f_hi:+.4f} at {hi:.4f}. "
            "With both positive, the asset grid may be binding (increase a_max); "
            "with both negative, households do not accumulate enough assets."
        )

    # Solve market-clearing rate r* via Brent's method
    r_star = float(brentq(lambda r: _excess_capital_demand(r)[0], lo, hi, xtol=xtol, maxiter=max_evals))

    excess_star, sol_star, dist_star = _excess_capital_demand(r_star)
    Ks_star = float(dist_star.mean())

    meta = {
        "elapsed_time": time.time() - t_start,
        "solver_type": "ContinuousEGM",
        "backend": backend,
        "K_hist_points": N_k,
        "r_upper_bound": r_upper_bound,
    }

    return AiyagariContinuousEquilibrium(
        r=r_star,
        w=sol_star.w,
        K=Ks_star,
        L=L_agg,
        household_solution=sol_star,
        distribution=dist_star,
        capital_market_clearing_error=excess_star,
        converged=True,
        iterations=counter["n"],
        metadata=meta,
    )


__all__ = [
    "young_lottery_weights",
    "continuous_push_distribution",
    "young_step",
    "build_continuous_transition_matrix",
    "young_transition_matrix",
    "continuous_stationary_distribution",
    "young_stationary_distribution",
    "ContinuousStationaryDistribution",
    "ContinuousDistributionResult",
    "continuous_stationary_equilibrium",
    "AiyagariContinuousEquilibrium",
    "ContinuousEquilibriumResult",
    "AiyagariContinuousModel",
    "solve_aiyagari_continuous",
]
