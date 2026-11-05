"""Functional / Distributional Vector Autoregression (Chang, Kim & Schorfheide 2024).

Couples aggregate macroeconomic time series with cross-sectional density or
share distributions over ``B`` bins. By projecting the ``B``-bin simplex shares
onto an orthonormal ``(B - 1)``-dimensional zero-sum Helmert contrast basis
:math:`H \\in \\mathbb{R}^{B \\times (B-1)}` satisfying :math:`H' \\mathbf{1}_B = 0`
and :math:`H' H = I_{B-1}`, the joint VAR(:math:`p`) is estimated in an
unconstrained Euclidean subspace while guaranteeing exact cross-sectional
adding-up (:math:`\\sum_{b=1}^B \\Delta \\hat{s}_{t+h, b} = 0`) at every horizon.

From the perturbed distribution paths :math:`s_{t+h} = \\bar{s} + \\Delta \\hat{s}_{t+h}`
evaluated on ``bin_grid``, the estimator computes implied structural impulse
responses for key inequality metrics:
- Gini coefficient (``gini_irf``)
- Top-bin share (``top_share_irf``)
- 90/10 percentile ratio (``p90_p10_irf``)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from puremacro._linalg import safe_cholesky
from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from puremacro.var.estimate import estimate_var
from puremacro.var.irf import irf as compute_irf

__all__ = [
    "FunctionalVARResult",
    "estimate_functional_var",
]


@dataclass(frozen=True)
class FunctionalVARResult:
    """Result container for Functional / Distributional VAR (Presentation Standard 2.0+).

    Attributes
    ----------
    macro_irf : np.ndarray
        Impulse responses of the ``n_m`` aggregate macro variables of shape
        ``(horizon + 1, n_m, n_total)``, where ``n_total = n_m + B - 1``.
    dist_irf : np.ndarray
        Impulse responses of the ``B`` cross-sectional bin shares of shape
        ``(horizon + 1, B, n_total)``, satisfying ``sum_b dist_irf[h, b, j] == 0``.
    gini_irf : np.ndarray
        Implied Gini coefficient impulse responses of shape ``(horizon + 1, n_total)``.
    top_share_irf : np.ndarray
        Implied top-bin share impulse responses of shape ``(horizon + 1, n_total)``.
    p90_p10_irf : np.ndarray
        Implied 90/10 percentile ratio impulse responses of shape ``(horizon + 1, n_total)``.
    adding_up_max_error : float
        Maximum absolute cross-sectional adding-up error
        :math:`\\max_{h, j} |\\sum_{b=1}^B \\text{dist\\_irf}[h, b, j]|`.
    A_list : list of np.ndarray
        Estimated joint VAR coefficient matrices ``[A_1, ..., A_p]``.
    Sigma : np.ndarray
        Joint reduced-form residual covariance matrix of shape ``(n_total, n_total)``.
    bin_grid : np.ndarray
        Grid of ``B`` cross-sectional bin support points.
    macro_names : tuple of str
        Names of the ``n_m`` aggregate macro variables.
    horizon : int
        Maximum impulse response horizon ``H``.
    p : int
        VAR lag order.
    metadata : dict
        Estimation metadata (sample size, baseline inequality metrics, Helmert basis).
    """

    macro_irf: np.ndarray
    dist_irf: np.ndarray
    gini_irf: np.ndarray
    top_share_irf: np.ndarray
    p90_p10_irf: np.ndarray
    adding_up_max_error: float
    A_list: list[np.ndarray]
    Sigma: np.ndarray
    bin_grid: np.ndarray
    macro_names: tuple[str, ...]
    horizon: int
    p: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of macro and inequality responses to each macro shock."""
        n_m = len(self.macro_names)
        rows: list[dict[str, Any]] = []
        for j in range(n_m):
            s_name = f"shock_{self.macro_names[j]}"
            for i in range(n_m):
                m_path = self.macro_irf[:, i, j]
                peak_idx = int(np.argmax(np.abs(m_path)))
                rows.append({
                    "shock": s_name,
                    "target": self.macro_names[i],
                    "category": "macro",
                    "impact": float(m_path[0]),
                    "peak": float(m_path[peak_idx]),
                    "peak_horizon": peak_idx,
                    "terminal": float(m_path[-1]),
                })
            for metric_name, arr in (
                ("gini", self.gini_irf[:, j]),
                ("top_share", self.top_share_irf[:, j]),
                ("p90_p10", self.p90_p10_irf[:, j]),
            ):
                p_idx = int(np.argmax(np.abs(arr)))
                rows.append({
                    "shock": s_name,
                    "target": metric_name,
                    "category": "inequality",
                    "impact": float(arr[0]),
                    "peak": float(arr[p_idx]),
                    "peak_horizon": p_idx,
                    "terminal": float(arr[-1]),
                })
        return pd.DataFrame(rows)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy DataFrame of macro, bin-share, and inequality IRFs for the primary shock."""
        H_plus = self.horizon + 1
        data: dict[str, Any] = {"horizon": np.arange(H_plus, dtype=int)}
        n_m = len(self.macro_names)
        B = len(self.bin_grid)
        for i in range(n_m):
            data[self.macro_names[i]] = self.macro_irf[:, i, 0]
        data["gini_irf"] = self.gini_irf[:, 0]
        data["top_share_irf"] = self.top_share_irf[:, 0]
        data["p90_p10_irf"] = self.p90_p10_irf[:, 0]
        for b in range(B):
            data[f"bin_{b}_irf"] = self.dist_irf[:, b, 0]
        return pd.DataFrame(data)

    def to_frame(self) -> pd.DataFrame:
        """Alias for :meth:`to_dataframe`."""
        return self.to_dataframe()

    def to_markdown(self, **kwargs: Any) -> str:
        """Render the Functional VAR impulse response table as Markdown."""
        return df_to_markdown(self.to_dataframe(), index=False, **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render the Functional VAR impulse response table as LaTeX tabular."""
        return df_to_latex(self.to_dataframe(), index=False, **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render the Functional VAR impulse response table as a Typst ``#table``."""
        return df_to_typst(self.to_dataframe(), index=False, **kwargs)

    def plot(self, ax: Any = None, show: bool = False, **kwargs: Any) -> Any:
        """Plot aggregate macro IRFs, distributional bin-share IRFs, and inequality IRFs."""
        import matplotlib.pyplot as plt

        horizons = np.arange(self.horizon + 1)
        shock_idx = int(kwargs.get("shock_idx", 0))
        figsize = kwargs.get("figsize", (12.0, 3.8))

        if ax is None:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            ax_list = list(axes.ravel())
        else:
            if isinstance(ax, np.ndarray):
                ax_list = list(ax.ravel())
                fig = ax_list[0].figure
            elif isinstance(ax, (list, tuple)):
                ax_list = list(ax)
                fig = ax_list[0].figure
            else:
                ax_list = [ax]
                fig = ax.figure

        # Panel 1: Macro IRFs
        p0 = ax_list[0]
        p0.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
        for i, name in enumerate(self.macro_names):
            p0.plot(horizons, self.macro_irf[:, i, shock_idx], linewidth=1.8, label=name)
        p0.set_title("Aggregate Macro IRFs")
        p0.set_xlabel("Horizon")
        p0.legend(loc="best", frameon=False)
        p0.grid(True, linestyle=":", alpha=0.4)

        # Panel 2: Distributional Bin Share IRFs
        if len(ax_list) >= 2:
            p1 = ax_list[1]
            p1.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
            for b in range(len(self.bin_grid)):
                p1.plot(
                    horizons,
                    self.dist_irf[:, b, shock_idx],
                    linewidth=1.4,
                    label=f"Bin {b + 1}",
                )
            p1.set_title("Cross-Sectional Bin Share IRFs")
            p1.set_xlabel("Horizon")
            p1.grid(True, linestyle=":", alpha=0.4)

        # Panel 3: Inequality Metric IRFs
        if len(ax_list) >= 3:
            p2 = ax_list[2]
            p2.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
            p2.plot(horizons, self.gini_irf[:, shock_idx], linewidth=1.8, label="Gini")
            p2.plot(horizons, self.top_share_irf[:, shock_idx], linewidth=1.8, label="Top Share")
            p2.set_title("Inequality Metric IRFs")
            p2.set_xlabel("Horizon")
            p2.legend(loc="best", frameon=False)
            p2.grid(True, linestyle=":", alpha=0.4)

        if ax is None:
            fig.tight_layout()
        if show:
            plt.show()
        return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _helmert_basis(B: int) -> np.ndarray:
    """Construct orthonormal zero-sum Helmert contrast matrix H of shape (B, B - 1).

    Satisfies ``H.T @ np.ones(B) == 0`` and ``H.T @ H == np.eye(B - 1)``.
    """
    H = np.zeros((B, B - 1), dtype=float)
    for k in range(1, B):
        denom = np.sqrt(float(k * (k + 1)))
        H[:k, k - 1] = 1.0 / denom
        H[k, k - 1] = -float(k) / denom
    return H


def _project_to_simplex(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Ensure a perturbed share vector stays positive and sums to 1."""
    clipped = np.maximum(np.asarray(p, dtype=float), eps)
    return clipped / float(np.sum(clipped))


def _compute_gini(shares: np.ndarray, grid: np.ndarray) -> float:
    """Compute the Gini coefficient for discrete mass ``shares`` on support ``grid``."""
    p = _project_to_simplex(shares)
    mu = float(np.sum(p * grid))
    if mu <= 1e-15:
        return 0.0
    diff_mat = np.abs(grid[:, None] - grid[None, :])
    num = float(p @ diff_mat @ p)
    return num / (2.0 * mu)


def _compute_p90_p10(shares: np.ndarray, grid: np.ndarray) -> float:
    """Compute smooth 90/10 percentile ratio via piecewise-linear CDF inversion on grid."""
    p = _project_to_simplex(shares)
    B = len(grid)
    # Construct strictly positive bin edges surrounding `grid`
    diffs = np.diff(grid)
    step_left = float(diffs[0]) if len(diffs) > 0 else 1.0
    step_right = float(diffs[-1]) if len(diffs) > 0 else 1.0
    edges = np.empty(B + 1, dtype=float)
    edges[0] = max(float(grid[0]) - 0.5 * step_left, 0.1 * float(grid[0]))
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    edges[-1] = float(grid[-1]) + 0.5 * step_right

    cdf = np.concatenate([[0.0], np.cumsum(p)])
    q10 = float(np.interp(0.10, cdf, edges))
    q90 = float(np.interp(0.90, cdf, edges))
    return q90 / max(q10, 1e-12)


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------


def estimate_functional_var(
    macro_data: np.ndarray | pd.DataFrame,
    dist_shares: np.ndarray | pd.DataFrame,
    *,
    bin_grid: Sequence[float] | np.ndarray | None = None,
    p: int = 2,
    horizon: int = 20,
    macro_names: Sequence[str] | None = None,
) -> FunctionalVARResult:
    """Estimate a Chang, Kim & Schorfheide (2024) Functional / Distributional VAR.

    Parameters
    ----------
    macro_data : np.ndarray or pd.DataFrame of shape (T, n_m)
        Aggregate macroeconomic series.
    dist_shares : np.ndarray or pd.DataFrame of shape (T, B)
        Cross-sectional density / share bins at each period ``t`` with ``B >= 3``
        and :math:`\\sum_{b=1}^B s_{t, b} = 1`.
    bin_grid : Sequence[float] or np.ndarray of shape (B,), optional
        Support levels for the ``B`` bins (e.g. income or wealth bin midpoints).
        Defaults to ``np.linspace(1.0, B, B)``.
    p : int, default 2
        Lag order of the joint VAR.
    horizon : int, default 20
        Impulse response horizon ``H``.
    macro_names : Sequence[str], optional
        Names of the aggregate macro variables.

    Returns
    -------
    FunctionalVARResult
    """
    if p < 1:
        raise ValueError(f"Lag order p must be >= 1; got {p}.")
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0; got {horizon}.")

    if isinstance(macro_data, pd.DataFrame):
        inferred_names = tuple(str(c) for c in macro_data.columns)
        Y_macro = macro_data.to_numpy(dtype=float)
    else:
        Y_macro = np.asarray(macro_data, dtype=float)
        if Y_macro.ndim == 1:
            Y_macro = Y_macro[:, None]
        if Y_macro.ndim != 2:
            raise ValueError(f"macro_data must be 1D or 2D; got shape {Y_macro.shape}.")
        inferred_names = tuple(f"macro_{i}" for i in range(Y_macro.shape[1]))

    T, n_m = Y_macro.shape
    if macro_names is not None:
        names = tuple(str(m) for m in macro_names)
        if len(names) != n_m:
            raise ValueError(f"len(macro_names)={len(names)} does not match n_m={n_m}.")
    else:
        names = inferred_names

    if isinstance(dist_shares, pd.DataFrame):
        S_raw = dist_shares.to_numpy(dtype=float)
    else:
        S_raw = np.asarray(dist_shares, dtype=float)
    if S_raw.ndim != 2 or S_raw.shape[0] != T:
        raise ValueError(
            f"dist_shares must have shape (T={T}, B); got {S_raw.shape}."
        )
    B = S_raw.shape[1]
    if B < 3:
        raise ValueError(f"estimate_functional_var requires at least B >= 3 bins; got B={B}.")

    row_sums = np.sum(S_raw, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("dist_shares rows must have positive sum.")
    if np.max(np.abs(row_sums - 1.0)) > 1e-2:
        raise ValueError(
            "dist_shares rows must sum to 1.0 (max deviation "
            f"{float(np.max(np.abs(row_sums - 1.0))):.4g})."
        )
    S = S_raw / row_sums

    if bin_grid is None:
        grid = np.linspace(1.0, float(B), B, dtype=float)
    else:
        grid = np.asarray(bin_grid, dtype=float).ravel()
        if len(grid) != B:
            raise ValueError(f"len(bin_grid)={len(grid)} must match B={B} columns of dist_shares.")
        if np.any(np.diff(grid) <= 0.0):
            grid = np.sort(grid)
        if grid[0] <= 0.0:
            grid = grid - grid[0] + 1.0

    # Orthonormal zero-sum Helmert contrast basis H of shape (B, B - 1)
    H = _helmert_basis(B)
    s_bar = np.mean(S, axis=0)  # (B,)
    C_contrasts = (S - s_bar[None, :]) @ H  # (T, B - 1)

    # Joint VAR(p) on Z_t = [Y_macro_t, C_contrasts_t]
    Z = np.column_stack([Y_macro, C_contrasts])
    n_total = n_m + B - 1
    est = estimate_var(Z, p=p)
    A_list = list(est.A_list)
    Sigma = np.asarray(est.Sigma, dtype=float)
    B0 = safe_cholesky(Sigma, name="estimate_functional_var Sigma")

    irf_Z = compute_irf(A_list, B0, horizon=horizon)  # (horizon + 1, n_total, n_total)
    macro_irf = irf_Z[:, :n_m, :]                     # (horizon + 1, n_m, n_total)
    contrast_irf = irf_Z[:, n_m:, :]                  # (horizon + 1, B - 1, n_total)

    # Map contrast IRFs back to B-bin share space: Delta s_h = H @ Delta c_h
    dist_irf = np.einsum("bk,hkj->hbj", H, contrast_irf)  # (horizon + 1, B, n_total)
    adding_up_max_error = float(np.max(np.abs(np.sum(dist_irf, axis=1))))

    # Compute baseline and perturbed inequality trajectories
    base_gini = _compute_gini(s_bar, grid)
    base_p90_p10 = _compute_p90_p10(s_bar, grid)

    top_share_irf = dist_irf[:, -1, :].copy()  # (horizon + 1, n_total)
    gini_irf = np.zeros((horizon + 1, n_total), dtype=float)
    p90_p10_irf = np.zeros((horizon + 1, n_total), dtype=float)

    for h in range(horizon + 1):
        for j in range(n_total):
            s_pert = s_bar + dist_irf[h, :, j]
            gini_irf[h, j] = _compute_gini(s_pert, grid) - base_gini
            p90_p10_irf[h, j] = _compute_p90_p10(s_pert, grid) - base_p90_p10

    metadata: dict[str, Any] = {
        "T": T,
        "n_m": n_m,
        "B": B,
        "n_total": n_total,
        "mean_shares": s_bar,
        "baseline_gini": base_gini,
        "baseline_top_share": float(s_bar[-1]),
        "baseline_p90_p10": base_p90_p10,
        "helmert_basis": H,
    }
    return FunctionalVARResult(
        macro_irf=macro_irf,
        dist_irf=dist_irf,
        gini_irf=gini_irf,
        top_share_irf=top_share_irf,
        p90_p10_irf=p90_p10_irf,
        adding_up_max_error=adding_up_max_error,
        A_list=A_list,
        Sigma=Sigma,
        bin_grid=grid,
        macro_names=names,
        horizon=horizon,
        p=p,
        metadata=metadata,
    )
