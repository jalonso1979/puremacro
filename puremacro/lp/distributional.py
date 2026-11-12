r"""Distributional and Quantile Local Projections with Exact Adding-Up Constraints.

Implements horizon-by-horizon local projections for cross-sectional quantile
functions, share/Lorenz bins, and continuous density bins (Chang, Gómez-Rodríguez
& Hong 2024), enforcing exact cross-sectional adding-up constraints:

.. math::

    \max_{0 \le h \le H} \left| \sum_{b=1}^B \beta_{b, h} w_b \right| < 10^{-12}

via null-space orthogonal projection:

.. math::

    \boldsymbol{\beta}_h^* = \left( \mathbf{I}_B - \frac{\mathbf{w} \mathbf{w}^\top}{\mathbf{w}^\top \mathbf{w}} \right) \hat{\boldsymbol{\beta}}_h

paired with Newey–West (1987) and Driscoll–Kraay (1998) covariance inference, and
an empirical-to-structural bridge aligning empirical local projection estimates
with Heterogeneous-Agent New Keynesian (HANK) `HetBlock` / `BlockDAG` sequence-space
impulse responses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    import matplotlib.figure

import numpy as np
import pandas as pd
from scipy.stats import norm

from puremacro._linalg import inv_xtx
from puremacro.inference.dk import driscoll_kraay
from puremacro.lp._common import (
    _DefaultFloat,
    _DefaultHorizons,
    _DefaultInt,
    _is_int_like,
    check_regular_index,
    validate_lp_columns_finite,
)
from puremacro.lp._panel_helpers import two_way_fe_within
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst

_DEFAULT_HORIZONS_16: Iterable[int] = _DefaultHorizons(16)
_DEFAULT_N_LAGS_2: int = _DefaultInt(2)
_DEFAULT_CI_90: float = _DefaultFloat(0.90)


def _resolve_distributional_kwargs(
    horizons: Iterable[int],
    n_lags: int,
    ci: float,
    *,
    lags: int | None = None,
    horizon: int | None = None,
    alpha: float | None = None,
    name: str,
) -> tuple[list[int], int, float]:
    """Validate and resolve canonical vs alias options for distributional/quantile LPs."""
    horizons_explicit = type(horizons) is not _DefaultHorizons
    n_lags_explicit = type(n_lags) is not _DefaultInt
    ci_explicit = type(ci) is not _DefaultFloat

    if not _is_int_like(n_lags) or int(n_lags) < 0:
        raise ValueError(f"{name}: lags must be a non-negative integer; got {n_lags!r}")
    if not (0.0 < float(ci) < 1.0):
        raise ValueError(f"{name}: ci must be a coverage probability in (0, 1); got {ci!r}")
    h_list = [int(h) for h in horizons]

    if lags is not None:
        if not _is_int_like(lags) or int(lags) < 0:
            raise ValueError(f"{name}: lags must be a non-negative integer; got {lags!r}")
        if n_lags_explicit and int(n_lags) != int(lags):
            raise ValueError(
                f"{name}: received conflicting lag arguments n_lags={int(n_lags)} "
                f"and lags={int(lags)}; pass only one."
            )
        n_lags = int(lags)

    if horizon is not None:
        if not _is_int_like(horizon) or int(horizon) < 0:
            raise ValueError(
                f"{name}: horizon must be a non-negative integer; got {horizon!r}"
            )
        alias_hs = list(range(0, int(horizon) + 1))
        if horizons_explicit and h_list != alias_hs:
            raise ValueError(
                f"{name}: received conflicting horizon arguments "
                f"horizons={h_list!r} and horizon={int(horizon)}; pass only one."
            )
        h_list = alias_hs

    if alpha is not None:
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(
                f"{name}: alpha must be a two-sided significance level in (0, 1); got {alpha!r}"
            )
        implied_ci = 1.0 - float(alpha)
        if ci_explicit and abs(float(ci) - implied_ci) > 1e-12:
            raise ValueError(
                f"{name}: received conflicting confidence arguments "
                f"ci={float(ci)} and alpha={float(alpha)} "
                f"(implies ci={implied_ci:.6g}); pass only one."
            )
        ci = implied_ci

    return h_list, int(n_lags), float(ci)

__all__ = [
    "DistributionalBridgeResult",
    "DistributionalLPResult",
    "bridge_hank_distributional_irf",
    "estimate_distributional_lp",
    "estimate_quantile_lp",
]


@dataclass(frozen=True)
class DistributionalLPResult:
    """Result container for distributional and quantile local projections.

    Parameters
    ----------
    horizons : np.ndarray
        Array of integer horizons (0, 1, ..., H-1).
    bins : tuple[str, ...]
        Names of cross-sectional deciles, shares, or quantiles.
    irf_matrix : np.ndarray
        Array of shape `(H, B)` with estimated impulse response coefficients.
    se_matrix : np.ndarray
        Array of shape `(H, B)` with standard errors.
    ci_lower : np.ndarray
        Array of shape `(H, B)` with lower confidence bounds.
    ci_upper : np.ndarray
        Array of shape `(H, B)` with upper confidence bounds.
    adding_up_max_error : float
        Maximum absolute deviation from the cross-sectional adding-up constraint.
    cov_type : str
        Covariance estimator used ("hac" | "driscoll_kraay").
    distribution_type : str
        Type of distribution modeled ("shares" | "density" | "quantiles").
    ci : float
        Confidence level (e.g. 0.90 for 90% confidence bands).
    bin_weights : np.ndarray
        Quadrature or metric weights `w_b` (ones for shares and deciles).
    metadata : dict[str, Any]
        Additional estimation metadata.
    """

    horizons: np.ndarray
    bins: tuple[str, ...]
    irf_matrix: np.ndarray
    se_matrix: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    adding_up_max_error: float
    cov_type: str
    distribution_type: str
    ci: float = 0.90
    bin_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bin_weights.size == 0 and len(self.bins) > 0:
            object.__setattr__(self, "bin_weights", np.ones(len(self.bins), dtype=np.float64))

    def check_adding_up(self, tol: float = 1e-12) -> bool:
        """Check whether cross-sectional adding-up holds within tolerance.

        For shares or deciles, sum of beta across bins must be zero.
        For density bins, weighted sum sum(beta * w) must be zero.
        """
        weights = self.bin_weights if self.bin_weights.size == len(self.bins) else np.ones(len(self.bins))
        max_err = float(np.max(np.abs(self.irf_matrix @ weights)))
        return bool(max_err <= tol)

    def to_dataframe(self, horizon: int | None = None) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.

        If `horizon` is specified, returns DataFrame for that horizon indexed by bin.
        If `horizon` is None, returns a long-form DataFrame with all horizons.
        """
        if horizon is not None:
            h_idx = np.where(self.horizons == horizon)[0]
            if len(h_idx) == 0:
                raise ValueError(f"Horizon {horizon} not in estimated horizons {list(self.horizons)}")
            idx = int(h_idx[0])
            return pd.DataFrame(
                {
                    "beta": self.irf_matrix[idx],
                    "se": self.se_matrix[idx],
                    "ci_lower": self.ci_lower[idx],
                    "ci_upper": self.ci_upper[idx],
                },
                index=pd.Index(self.bins, name="bin"),
            )

        rows = []
        for i, h in enumerate(self.horizons):
            for j, b in enumerate(self.bins):
                rows.append(
                    {
                        "horizon": int(h),
                        "bin": str(b),
                        "beta": float(self.irf_matrix[i, j]),
                        "se": float(self.se_matrix[i, j]),
                        "ci_lower": float(self.ci_lower[i, j]),
                        "ci_upper": float(self.ci_upper[i, j]),
                    }
                )
        return pd.DataFrame(rows)

    def to_frame(self, horizon: int | None = None) -> pd.DataFrame:
        """Alias for `to_dataframe()`."""
        return self.to_dataframe(horizon=horizon)

    def summary(self, horizon: int | None = None) -> pd.DataFrame:
        """Generate a formatted summary table of the impulse response coefficients."""
        if horizon is not None:
            return self.to_dataframe(horizon=horizon)
        # Summary across key horizons: impact (0), peak, and last horizon
        key_h = [int(self.horizons[0])]
        if len(self.horizons) > 1:
            mid = int(self.horizons[len(self.horizons) // 2])
            last = int(self.horizons[-1])
            if mid not in key_h:
                key_h.append(mid)
            if last not in key_h:
                key_h.append(last)

        cols: dict[str, list[float]] = {}
        for h in key_h:
            h_idx = int(np.where(self.horizons == h)[0][0])
            cols[f"h={h}_beta"] = [float(x) for x in self.irf_matrix[h_idx]]
            cols[f"h={h}_se"] = [float(x) for x in self.se_matrix[h_idx]]
        df = pd.DataFrame(cols, index=pd.Index(self.bins, name="bin"))
        return df

    def to_markdown(self, **kwargs: Any) -> str:
        """Render markdown representation of the summary table."""
        return _df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render LaTeX representation of the summary table."""
        return _df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render Typst representation of the summary table."""
        return _df_to_typst(self.summary(), **kwargs)

    def plot_deciles(self, figsize: tuple[float, float] = (11.0, 4.5)) -> matplotlib.figure.Figure:
        """Plot impulse responses across deciles/bins over horizons."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

        n_bins = len(self.bins)
        cmap = plt.cm.viridis
        colors = [cmap(i / max(1, n_bins - 1)) for i in range(n_bins)]

        # Panel 1: IRF curves over time for all deciles
        for j, b in enumerate(self.bins):
            ax1.plot(self.horizons, self.irf_matrix[:, j], label=b, color=colors[j], lw=1.8)
        ax1.axhline(0.0, color="0.4", ls=":", lw=0.8)
        ax1.set_title(f"Distributional LP IRF ({self.distribution_type})")
        ax1.set_xlabel("Horizon h")
        ax1.set_ylabel("Response")
        if n_bins <= 10:
            ax1.legend(frameon=False, ncol=2, fontsize="small")

        # Panel 2: Cross-sectional profile at h=0 and terminal h
        h0 = int(self.horizons[0])
        h_end = int(self.horizons[-1])
        x_bars = np.arange(n_bins)
        width = 0.35
        ax2.bar(x_bars - width / 2, self.irf_matrix[0], width, label=f"h={h0} (Impact)", color="#2b5c8f", alpha=0.85)
        ax2.bar(x_bars + width / 2, self.irf_matrix[-1], width, label=f"h={h_end} (Terminal)", color="#c0392b", alpha=0.85)
        ax2.axhline(0.0, color="0.4", ls=":", lw=0.8)
        ax2.set_xticks(x_bars)
        ax2.set_xticklabels(self.bins, rotation=45, ha="right")
        ax2.set_title("Cross-Sectional Incidence Profile")
        ax2.set_ylabel("Response")
        ax2.legend(frameon=False, fontsize="small")

        return fig

    def plot(self, figsize: tuple[float, float] = (11.0, 4.5)) -> matplotlib.figure.Figure:
        """Plot impulse responses across deciles/bins."""
        return self.plot_deciles(figsize=figsize)


@dataclass(frozen=True)
class DistributionalBridgeResult:
    """Result container for empirical LP vs structural HANK distributional comparison.

    Parameters
    ----------
    distance_l2 : float
        Root-mean-squared $L_2$ distance between empirical LP and structural HANK IRFs.
    distance_mahalanobis : float
        Precision-weighted Mahalanobis distance.
    mean_absolute_error : float
        Mean absolute error between empirical and structural IRFs.
    sign_concordance : float
        Fraction of horizon-bin pairs where empirical and structural IRF share the same sign.
    decile_correlations : dict[str, float]
        Trajectory correlation between empirical and structural paths for each bin.
    peak_gaps : dict[str, int]
        Discrepancy in peak horizon (empirical peak horizon - structural peak horizon).
    empirical_irf : np.ndarray
        Empirical LP impulse responses of shape `(H, B)`.
    hank_irf : np.ndarray
        Structural HANK impulse responses of shape `(H, B)`.
    bins : tuple[str, ...]
        Bin names.
    horizons : np.ndarray
        Array of integer horizons.
    metadata : dict[str, Any]
        Additional comparison metadata.
    """

    distance_l2: float
    distance_mahalanobis: float
    mean_absolute_error: float
    sign_concordance: float
    decile_correlations: dict[str, float]
    peak_gaps: dict[str, int]
    empirical_irf: np.ndarray
    hank_irf: np.ndarray
    bins: tuple[str, ...]
    horizons: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Generate summary table of alignment metrics."""
        rows = [
            {"Metric": "L2 Distance (RMSD)", "Value": self.distance_l2},
            {"Metric": "Weighted Mahalanobis Distance", "Value": self.distance_mahalanobis},
            {"Metric": "Mean Absolute Error (MAE)", "Value": self.mean_absolute_error},
            {"Metric": "Directional Sign Concordance", "Value": self.sign_concordance},
            {"Metric": "Average Decile Correlation", "Value": float(np.mean(list(self.decile_correlations.values())))},
            {"Metric": "Max Peak Timing Gap (Periods)", "Value": float(np.max(np.abs(list(self.peak_gaps.values()))))},
        ]
        return pd.DataFrame(rows).set_index("Metric")

    def to_dataframe(self) -> pd.DataFrame:
        """Return decile-by-decile metrics comparison."""
        rows = []
        for b in self.bins:
            rows.append(
                {
                    "bin": b,
                    "correlation": self.decile_correlations.get(b, np.nan),
                    "peak_timing_gap": self.peak_gaps.get(b, 0),
                }
            )
        return pd.DataFrame(rows).set_index("bin")

    def to_frame(self) -> pd.DataFrame:
        """Alias for `to_dataframe()`."""
        return self.to_dataframe()

    def to_markdown(self, **kwargs: Any) -> str:
        """Render markdown representation of the summary table."""
        return _df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render LaTeX representation of the summary table."""
        return _df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render Typst representation of the summary table."""
        return _df_to_typst(self.summary(), **kwargs)

    def plot(self, figsize: tuple[float, float] = (12.0, 5.0)) -> matplotlib.figure.Figure:
        """Plot comparison between empirical LP and structural HANK decile IRFs."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

        n_bins = len(self.bins)
        # Select key bins to display (e.g. first, middle, last)
        key_indices = [0]
        if n_bins > 2:
            key_indices.append(n_bins // 2)
        if n_bins > 1:
            key_indices.append(n_bins - 1)

        palette = ["#2b5c8f", "#27ae60", "#c0392b", "#8e44ad"]
        for idx, k in enumerate(key_indices):
            b_name = self.bins[k]
            c = palette[idx % len(palette)]
            ax1.plot(self.horizons, self.empirical_irf[:, k], label=f"Empirical: {b_name}", color=c, lw=2.0)
            ax1.plot(self.horizons, self.hank_irf[:, k], label=f"HANK: {b_name}", color=c, lw=1.5, ls="--")

        ax1.axhline(0.0, color="0.5", ls=":", lw=0.8)
        ax1.set_title("Empirical LP vs HANK Model IRF")
        ax1.set_xlabel("Horizon h")
        ax1.set_ylabel("Response")
        ax1.legend(frameon=False, fontsize="small")

        # Panel 2: Point-by-point scatter
        ax2.scatter(self.hank_irf.ravel(), self.empirical_irf.ravel(), color="#2b5c8f", alpha=0.6, s=25)
        min_v = min(float(np.min(self.hank_irf)), float(np.min(self.empirical_irf)))
        max_v = max(float(np.max(self.hank_irf)), float(np.max(self.empirical_irf)))
        ax2.plot([min_v, max_v], [min_v, max_v], color="0.4", ls="--", lw=1.0, label="45° Line")
        ax2.set_title(f"Model-Empirical Alignment (L2={self.distance_l2:.3e})")
        ax2.set_xlabel("Structural HANK IRF")
        ax2.set_ylabel("Empirical LP IRF")
        ax2.legend(frameon=False, fontsize="small")

        return fig


def estimate_distributional_lp(
    df: pd.DataFrame,
    shock: str,
    bin_cols: Sequence[str],
    *,
    horizons: Iterable[int] = _DEFAULT_HORIZONS_16,
    n_lags: int = _DEFAULT_N_LAGS_2,
    controls: Sequence[str] = (),
    cov_type: str = "hac",
    time_col: str | None = None,
    unit_col: str | None = None,
    ci: float = _DEFAULT_CI_90,
    distribution_type: str = "shares",
    bin_widths: Sequence[float] | None = None,
    enforce_adding_up: bool = True,
    lags: int | None = None,
    horizon: int | None = None,
    alpha: float | None = None,
) -> DistributionalLPResult:
    r"""Estimate horizon-by-horizon distributional local projections with exact adding-up.

    For each bin $b \in \{1, \dots, B\}$ and horizon $h \in \{0, 1, \dots, H-1\}$:

    .. math::

        \Delta_h s_{b, t} = \alpha_{b, h} + \beta_{b, h} x_t + \sum_{\ell=1}^p \gamma_{b, h, \ell}^\top Z_{t-\ell} + u_{b, t+h}

    When ``enforce_adding_up=True``, estimates are projected onto the orthogonal
    complement of metric weights $\mathbf{w}$:

    .. math::

        \boldsymbol{\beta}_h^* = \left( \mathbf{I} - \frac{\mathbf{w} \mathbf{w}^\top}{\|\mathbf{w}\|_2^2} \right) \hat{\boldsymbol{\beta}}_h

    guaranteeing that $\max_h |\sum_b \beta_{b, h}^* w_b| < 10^{-12}$ to machine precision.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing shock, bin columns, and optional controls.
    shock : str
        Column name of the structural shock or policy innovation $x_t$.
    bin_cols : Sequence[str]
        Column names of the cross-sectional bins, decile shares, or density grid points.
    horizons : Iterable[int], default range(0, 16)
        Impulse response horizons $h = 0, \dots, H-1$.
    n_lags : int, default 2
        Number of autoregressive lags for controls and regressors.
    controls : Sequence[str], default ()
        Additional macroeconomic or control series.
    cov_type : str, default "hac"
        Covariance inference: `"hac"` (Newey-West Bartlett) or `"driscoll_kraay"`.
    time_col : str | None, default None
        Time identifier column (required for `"driscoll_kraay"`).
    unit_col : str | None, default None
        Unit/panel identifier column (for panel data).
    ci : float, default 0.90
        Confidence band coverage level in $(0, 1)$.
    distribution_type : str, default "shares"
        Type of cross-sectional measure: `"shares"`, `"density"`, or `"quantiles"`.
    bin_widths : Sequence[float] | None, default None
        Quadrature or width weights for continuous density integration.
    enforce_adding_up : bool, default True
        Whether to enforce the null-space orthogonal adding-up projection.
    lags : int | None, optional
        Alias for `n_lags`.
    horizon : int | None, optional
        Alias specifying max horizon `range(0, horizon + 1)`.
    alpha : float | None, optional
        Significance level `1 - ci`.

    Returns
    -------
    DistributionalLPResult
        Container with estimated IRF matrix, standard errors, confidence bands,
        and adding-up verification.
    """
    h_list, n_lags, ci = _resolve_distributional_kwargs(
        horizons,
        n_lags,
        ci,
        lags=lags,
        horizon=horizon,
        alpha=alpha,
        name="estimate_distributional_lp",
    )

    bins_tuple = tuple(str(b) for b in bin_cols)
    B = len(bins_tuple)
    if B < 2:
        raise ValueError(f"bin_cols must contain at least 2 bins, got {B}")

    H = len(h_list)
    if H == 0:
        raise ValueError("horizons cannot be empty.")

    ctl_list = [str(c) for c in controls]
    cov_type = str(cov_type).lower().strip().replace("-", "_")
    if cov_type in ("newey_west", "nw"):
        cov_type = "hac"
    if cov_type not in ("hac", "driscoll_kraay"):
        raise ValueError(f"cov_type must be 'hac' or 'driscoll_kraay', got {cov_type!r}")

    if bin_widths is not None:
        w_vec = np.asarray(bin_widths, dtype=np.float64).reshape(-1)
        if w_vec.size != B:
            raise ValueError(f"bin_widths size ({w_vec.size}) must match number of bins ({B})")
    else:
        w_vec = np.ones(B, dtype=np.float64)

    z_crit = float(norm.ppf(1.0 - (1.0 - ci) / 2.0))

    irf_matrix = np.zeros((H, B), dtype=np.float64)
    se_matrix = np.zeros((H, B), dtype=np.float64)

    is_panel = (unit_col is not None and time_col is not None) or cov_type == "driscoll_kraay"

    if is_panel:
        if unit_col is None or time_col is None:
            if isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 2:
                u_name, t_name = df.index.names
            else:
                raise ValueError("Panel LP with cov_type='driscoll_kraay' requires unit_col and time_col.")
        else:
            u_name, t_name = unit_col, time_col

        df_work = df.copy()
        if not (isinstance(df_work.index, pd.MultiIndex) and df_work.index.nlevels == 2):
            df_work = df_work.set_index([u_name, t_name])
        if df_work.index.has_duplicates:
            raise ValueError(
                "estimate_distributional_lp: panel index contains duplicate (unit, time) rows."
            )
        validate_lp_columns_finite(
            df_work, [shock, *bins_tuple, *ctl_list], "estimate_distributional_lp"
        )
        df_work = df_work.sort_index()

        grouped = df_work.groupby(level=0)
        # Create lag columns
        lag_cols: list[str] = []
        for lag in range(1, n_lags + 1):
            col_x_lag = f"__x_L{lag}__"
            df_work[col_x_lag] = grouped[shock].shift(lag)
            lag_cols.append(col_x_lag)
            for c in ctl_list:
                col_c_lag = f"__{c}_L{lag}__"
                df_work[col_c_lag] = grouped[c].shift(lag)
                lag_cols.append(col_c_lag)

        for i_h, h in enumerate(h_list):
            # Fit each bin equation
            beta_raw = np.zeros(B, dtype=np.float64)
            se_raw = np.zeros(B, dtype=np.float64)

            for j_b, b_col in enumerate(bins_tuple):
                dy_col = f"__dy_{b_col}_h{h}__"
                df_work[dy_col] = grouped[b_col].shift(-h) - grouped[b_col].shift(1)
                reg_cols = [shock] + lag_cols + ctl_list
                sub = df_work[[dy_col] + reg_cols].dropna()
                if sub.empty:
                    beta_raw[j_b] = 0.0
                    se_raw[j_b] = 1.0
                    continue

                out = two_way_fe_within(
                    sub,
                    y_col=dy_col,
                    x_cols=reg_cols,
                    entity_level=df_work.index.names[0],
                    time_level=df_work.index.names[1],
                )
                beta_raw[j_b] = float(out["beta"][0])
                if cov_type == "driscoll_kraay":
                    X_w = out["X_within"]
                    u_res = out["residuals"]
                    score = X_w * u_res[:, None]
                    t_keys = out["time_keys"]
                    T_dk = int(len(np.unique(t_keys)))
                    L_bw = max(1, int(np.floor(4 * (T_dk / 100.0) ** (2.0 / 9.0))))
                    S_dk = driscoll_kraay(score, t_keys, lags=L_bw)
                    XtX_inv = out["XtX_inv"]
                    vcov = XtX_inv @ S_dk @ XtX_inv
                    se_raw[j_b] = float(np.sqrt(max(0.0, np.diag(vcov)[0])))
                else:
                    se_raw[j_b] = float(out["se"][0])

            if enforce_adding_up and distribution_type in ("shares", "density"):
                w_norm_sq = float(np.sum(w_vec ** 2))
                discrepancy = float(np.dot(w_vec, beta_raw))
                beta_proj = beta_raw - (discrepancy / w_norm_sq) * w_vec
                # Project standard errors through M = I - w w' / ||w||^2
                M = np.eye(B, dtype=np.float64) - np.outer(w_vec, w_vec) / w_norm_sq
                V_proj = M @ np.diag(se_raw ** 2) @ M.T
                se_proj = np.sqrt(np.maximum(1e-16, np.diag(V_proj)))
                irf_matrix[i_h] = beta_proj
                se_matrix[i_h] = se_proj
            else:
                irf_matrix[i_h] = beta_raw
                se_matrix[i_h] = se_raw

    else:
        # Time-series dimension
        if unit_col is not None or isinstance(df.index, pd.MultiIndex):
            raise ValueError(
                "estimate_distributional_lp: panel input requires both unit_col and time_col "
                "(or a 2-level MultiIndex with cov_type='driscoll_kraay')."
            )
        data = df.copy()
        if time_col is not None:
            if time_col not in data.columns:
                raise KeyError(
                    f"estimate_distributional_lp: time_col={time_col!r} not found in DataFrame."
                )
            data = data.set_index(time_col)
        check_regular_index(
            data.index, "estimate_distributional_lp", cumulative=False, allow_gaps=False
        )
        validate_lp_columns_finite(
            data, [shock, *bins_tuple, *ctl_list], "estimate_distributional_lp"
        )
        for lag in range(1, n_lags + 1):
            data[f"__x_L{lag}__"] = data[shock].shift(lag)
            for c in ctl_list:
                data[f"__{c}_L{lag}__"] = data[c].shift(lag)

        reg_base_cols = [shock] + [f"__x_L{lag}__" for lag in range(1, n_lags + 1)]
        for c in ctl_list:
            reg_base_cols.append(c)
            for lag in range(1, n_lags + 1):
                reg_base_cols.append(f"__{c}_L{lag}__")

        for i_h, h in enumerate(h_list):
            y_dict = {
                f"__dy_{b_col}_h{h}__": data[b_col].shift(-h) - data[b_col].shift(1)
                for b_col in bins_tuple
            }
            y_df_h = pd.DataFrame(y_dict, index=data.index)
            y_cols_h = list(y_dict.keys())

            sub = pd.concat([y_df_h, data[reg_base_cols]], axis=1).dropna()
            if sub.empty:
                continue

            n_obs = len(sub)
            X_mat = np.column_stack([np.ones(n_obs, dtype=np.float64), sub[reg_base_cols].values])
            Y_mat = sub[y_cols_h].values  # shape (n_obs, B)

            XtX_inv = inv_xtx(X_mat, name="estimate_distributional_lp")
            # Joint OLS across all bins: B_hat has shape (k, B)
            B_hat = XtX_inv @ (X_mat.T @ Y_mat)
            beta_raw = B_hat[1, :]  # focal regressor shock is col 1
            U_mat = Y_mat - X_mat @ B_hat  # (n_obs, B)

            # Newey-West HAC covariance for the shock coefficient across all B bins
            # Score vector for regressor 1: g_{t, b} = X_{t, 1} * U_{t, b}
            scores = X_mat[:, 1:2] * U_mat  # (n_obs, B)
            c1_scalar = float(XtX_inv[1, 1])

            L_bw = h + 1
            S_hac = scores.T @ scores
            for ell in range(1, L_bw + 1):
                if ell >= n_obs:
                    break
                w_ell = 1.0 - ell / (L_bw + 1.0)
                Gamma = scores[ell:].T @ scores[:-ell]
                S_hac += w_ell * (Gamma + Gamma.T)

            # Variance of beta_hat across bins
            V_raw = (XtX_inv[1:2, :] @ (X_mat.T @ U_mat))  # structural covariance
            # Using sandwich: V_beta = c1 * S_hac * c1
            # But more accurately, for regressor 1, beta_1 = e_1' (X'X)^-1 X' Y
            # cov(beta_1) = e_1' (X'X)^-1 (sum w_ell X' u_t u_{t-l}' X) (X'X)^-1 e_1
            # Which equals c1_row @ S_all @ c1_col.
            # When simplified across equations:
            e1 = XtX_inv[1, :]  # (k,)
            S_e1 = np.zeros((B, B), dtype=np.float64)
            # Efficiently compute covariance of e1 @ X' u_b
            z_t = X_mat @ e1  # (n_obs,)
            scores_z = z_t[:, None] * U_mat  # (n_obs, B)
            S_e1 = scores_z.T @ scores_z
            for ell in range(1, L_bw + 1):
                if ell >= n_obs:
                    break
                w_ell = 1.0 - ell / (L_bw + 1.0)
                Gamma_z = scores_z[ell:].T @ scores_z[:-ell]
                S_e1 += w_ell * (Gamma_z + Gamma_z.T)

            se_raw = np.sqrt(np.maximum(1e-16, np.diag(S_e1)))

            if enforce_adding_up and distribution_type in ("shares", "density"):
                w_norm_sq = float(np.sum(w_vec ** 2))
                discrepancy = float(np.dot(w_vec, beta_raw))
                beta_proj = beta_raw - (discrepancy / w_norm_sq) * w_vec
                M = np.eye(B, dtype=np.float64) - np.outer(w_vec, w_vec) / w_norm_sq
                V_proj = M @ S_e1 @ M.T
                se_proj = np.sqrt(np.maximum(1e-16, np.diag(V_proj)))
                irf_matrix[i_h] = beta_proj
                se_matrix[i_h] = se_proj
            else:
                irf_matrix[i_h] = beta_raw
                se_matrix[i_h] = se_raw

    ci_lower = irf_matrix - z_crit * se_matrix
    ci_upper = irf_matrix + z_crit * se_matrix

    max_adding_up_err = float(np.max(np.abs(irf_matrix @ w_vec)))

    return DistributionalLPResult(
        horizons=np.asarray(h_list, dtype=np.int64),
        bins=bins_tuple,
        irf_matrix=irf_matrix,
        se_matrix=se_matrix,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        adding_up_max_error=max_adding_up_err,
        cov_type=cov_type,
        distribution_type=distribution_type,
        ci=ci,
        bin_weights=w_vec,
        metadata={"n_lags": n_lags, "controls": ctl_list, "shock": shock},
    )


def estimate_quantile_lp(
    df: pd.DataFrame,
    shock: str | None = None,
    outcome: str | None = None,
    *,
    y: str | None = None,
    x: str | None = None,
    quantiles: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 0.90),
    horizons: Iterable[int] = _DEFAULT_HORIZONS_16,
    n_lags: int = _DEFAULT_N_LAGS_2,
    controls: Sequence[str] = (),
    cov_type: str = "hac",
    time_col: str | None = None,
    unit_col: str | None = None,
    ci: float = _DEFAULT_CI_90,
    lags: int | None = None,
    horizon: int | None = None,
    alpha: float | None = None,
) -> DistributionalLPResult:
    r"""Estimate horizon-by-horizon quantile local projections for a single time series.

    For each quantile $\tau \in (0, 1)$ and horizon $h \in \{0, 1, \dots, H-1\}$:

    .. math::

        Q_\tau(y_{t+h} - y_{t-1} \mid x_t, Z_{t-1}) = \alpha_{\tau, h} + \beta_{\tau, h} x_t + \sum_{\ell=1}^p \gamma_{\tau, h, \ell}^\top Z_{t-\ell}

    Estimated via quantile regression with analytical Newey-West sandwich standard errors
    robust to serial correlation.

    Parameters
    ----------
    df : pd.DataFrame
        Single-series time-series data containing shock, outcome, and controls.
    shock : str | None, optional
        Focal shock column (alias `x`).
    outcome : str | None, optional
        Outcome variable column (alias `y`).
    y : str | None, optional
        Outcome column alias.
    x : str | None, optional
        Shock column alias.
    quantiles : Sequence[float], default (0.10, 0.25, 0.50, 0.75, 0.90)
        Evaluation quantiles.
    horizons : Iterable[int], default range(0, 16)
        Impulse response horizons.
    n_lags : int, default 2
        Lag order for controls and variables.
    controls : Sequence[str], default ()
        Additional macroeconomic controls.
    cov_type : str, default "hac"
        Covariance type ("hac" | "newey_west" | "nw"). Panel covariance types
        such as `"driscoll_kraay"` are not supported and raise `ValueError`.
    time_col : str | None, default None
        Optional single-series time identifier column validated for chronological
        regularity.
    unit_col : str | None, default None
        Must be `None`; panel quantile LP is not implemented and passing
        `unit_col` or a `MultiIndex` raises `ValueError`.
    ci : float, default 0.90
        Confidence coverage level.

    Returns
    -------
    DistributionalLPResult
        Container with quantile IRF estimates, standard errors, and confidence bands.
    """
    from puremacro.lp.quantile import _qreg

    if shock is not None and x is not None and shock != x:
        raise ValueError(
            f"estimate_quantile_lp: received conflicting shock arguments "
            f"shock={shock!r} and x={x!r}; pass only one."
        )
    if outcome is not None and y is not None and outcome != y:
        raise ValueError(
            f"estimate_quantile_lp: received conflicting outcome arguments "
            f"outcome={outcome!r} and y={y!r}; pass only one."
        )
    focal_shock = shock if shock is not None else x
    focal_outcome = outcome if outcome is not None else y
    if focal_shock is None or focal_outcome is None:
        raise ValueError("Must provide both shock (or x) and outcome (or y).")

    if unit_col is not None or isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            "estimate_quantile_lp: panel inputs (unit_col or MultiIndex) are not "
            "supported; estimate_quantile_lp estimates single-series quantile local projections."
        )

    cov_norm = str(cov_type).lower().strip().replace("-", "_")
    if cov_norm in ("hac", "newey_west", "nw"):
        cov_type = "hac"
    else:
        raise ValueError(
            f"estimate_quantile_lp: cov_type={cov_type!r} is not supported; "
            "only time-series Newey-West HAC ('hac' / 'newey_west') is implemented."
        )

    h_list, n_lags, ci = _resolve_distributional_kwargs(
        horizons,
        n_lags,
        ci,
        lags=lags,
        horizon=horizon,
        alpha=alpha,
        name="estimate_quantile_lp",
    )
    if len(h_list) == 0:
        raise ValueError("horizons cannot be empty.")

    tau_list = [float(q) for q in quantiles]
    for q in tau_list:
        if not (0.0 < q < 1.0):
            raise ValueError(f"Quantiles must lie strictly in (0, 1), got {q}")

    H = len(h_list)
    Q = len(tau_list)
    q_names = tuple(f"q{int(round(q * 100)):02d}" for q in tau_list)

    ctl_list = [str(c) for c in controls]
    z_crit = float(norm.ppf(1.0 - (1.0 - ci) / 2.0))

    irf_matrix = np.zeros((H, Q), dtype=np.float64)
    se_matrix = np.zeros((H, Q), dtype=np.float64)

    data = df.copy()
    if time_col is not None:
        if time_col not in data.columns:
            raise KeyError(
                f"estimate_quantile_lp: time_col={time_col!r} not found in DataFrame "
                f"(available: {list(data.columns)!r})."
            )
        data = data.set_index(time_col)
    check_regular_index(
        data.index, "estimate_quantile_lp", cumulative=False, allow_gaps=False
    )
    validate_lp_columns_finite(
        data, [focal_shock, focal_outcome, *ctl_list], "estimate_quantile_lp"
    )
    for lag in range(1, n_lags + 1):
        data[f"__x_L{lag}__"] = data[focal_shock].shift(lag)
        data[f"__y_L{lag}__"] = data[focal_outcome].shift(lag)
        for c in ctl_list:
            data[f"__{c}_L{lag}__"] = data[c].shift(lag)

    reg_cols = [focal_shock]
    for lag in range(1, n_lags + 1):
        reg_cols.append(f"__x_L{lag}__")
        reg_cols.append(f"__y_L{lag}__")
        for c in ctl_list:
            reg_cols.append(f"__{c}_L{lag}__")
    for c in ctl_list:
        reg_cols.append(c)

    for i_h, h in enumerate(h_list):
        dy_col = f"__dy_{focal_outcome}_h{h}__"
        data[dy_col] = data[focal_outcome].shift(-h) - data[focal_outcome].shift(1)
        sub = data[[dy_col] + reg_cols].dropna()
        if sub.empty:
            continue

        n_obs = len(sub)
        X_mat = np.column_stack([np.ones(n_obs, dtype=np.float64), sub[reg_cols].values])
        y_vec = sub[dy_col].values

        for j_q, tau in enumerate(tau_list):
            b_hat = _qreg(y_vec, X_mat, tau)
            beta_q = float(b_hat[1])  # coefficient on shock
            irf_matrix[i_h, j_q] = beta_q

            # Analytical Powell / Koenker-Bassett kernel sandwich standard errors
            res = y_vec - X_mat @ b_hat
            sigma_res = float(np.std(res)) if float(np.std(res)) > 1e-8 else 1.0
            # Silverman's bandwidth rule
            h_bw = max(1e-4, 1.06 * sigma_res * (n_obs ** (-0.2)))
            # Gaussian kernel density estimate at 0
            dens_0 = float(np.mean(norm.pdf(res / h_bw)) / h_bw)
            dens_0 = max(dens_0, 1e-6)

            D1 = dens_0 * (X_mat.T @ X_mat / n_obs)
            D1_inv = np.linalg.pinv(D1)

            # Score: psi_t = tau - I(res_t <= 0)
            psi = tau - (res <= 0.0).astype(np.float64)
            scores = X_mat * psi[:, None]  # (n_obs, k)

            L_bw = h + 1
            S_meat = scores.T @ scores / n_obs
            for ell in range(1, L_bw + 1):
                if ell >= n_obs:
                    break
                w_ell = 1.0 - ell / (L_bw + 1.0)
                Gamma = (scores[ell:].T @ scores[:-ell]) / n_obs
                S_meat += w_ell * (Gamma + Gamma.T)

            vcov = (D1_inv @ S_meat @ D1_inv) / n_obs
            se_val = float(np.sqrt(max(0.0, vcov[1, 1])))
            se_matrix[i_h, j_q] = max(se_val, 1e-8)

    ci_lower = irf_matrix - z_crit * se_matrix
    ci_upper = irf_matrix + z_crit * se_matrix

    return DistributionalLPResult(
        horizons=np.asarray(h_list, dtype=np.int64),
        bins=q_names,
        irf_matrix=irf_matrix,
        se_matrix=se_matrix,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        adding_up_max_error=0.0,
        cov_type=cov_type,
        distribution_type="quantiles",
        ci=ci,
        bin_weights=np.ones(Q, dtype=np.float64),
        metadata={"quantiles": tau_list, "shock": focal_shock, "outcome": focal_outcome},
    )


def bridge_hank_distributional_irf(
    lp_result: DistributionalLPResult,
    hank_irf: Any,
    *,
    weight_matrix: np.ndarray | None = None,
    bin_mapping: Mapping[str, str] | None = None,
) -> DistributionalBridgeResult:
    r"""Bridge empirical distributional LP estimates against structural HANK impulse responses.

    Aligns empirical deciles/bins with structural sequence-space paths generated by
    `HetBlock` / `BlockDAG` or empirical DataFrames, and calculates:

    1. **$L_2$ Root-Mean-Squared Distance**:

       .. math::

           \mathcal{D}_{L_2} = \sqrt{\frac{1}{H \cdot B} \sum_{h=0}^{H-1} \sum_{b=1}^B (\hat{\beta}_{b, h} - \beta_{b, h}^{\text{HANK}})^2}

    2. **Decile-by-Decile Correlation**:

       .. math::

           \rho_b = \operatorname{corr}\left( \hat{\beta}_{b, \cdot}, \beta_{b, \cdot}^{\text{HANK}} \right)

    3. **Mean Absolute Error (MAE)**:

       .. math::

           \text{MAE} = \frac{1}{H \cdot B} \sum_{h=0}^{H-1} \sum_{b=1}^B |\hat{\beta}_{b, h} - \beta_{b, h}^{\text{HANK}}|

    4. **Peak Timing Discrepancy**:

       .. math::

           \Delta h_b^* = \operatorname{argmax}_h |\hat{\beta}_{b, h}| - \operatorname{argmax}_h |\beta_{b, h}^{\text{HANK}}|

    5. **Directional Sign Concordance Rate**:

       .. math::

           \mathcal{S} = \frac{1}{H \cdot B} \sum_{h=0}^{H-1} \sum_{b=1}^B \mathbb{I}\left( \operatorname{sgn}(\hat{\beta}_{b, h}) = \operatorname{sgn}(\beta_{b, h}^{\text{HANK}}) \right)

    Parameters
    ----------
    lp_result : DistributionalLPResult
        Empirical distributional or quantile local projection result.
    hank_irf : Any
        Structural HANK model result (`SSJDAGResult`), `HetBlock`, `pd.DataFrame`,
        or `np.ndarray` of shape `(H, B)`.
    weight_matrix : np.ndarray | None, optional
        Weight matrix for precision-weighted distance. When None, uses diagonal
        inverse empirical variances.
    bin_mapping : Mapping[str, str] | None, optional
        Mapping from empirical bin name to HANK DataFrame column name.

    Returns
    -------
    DistributionalBridgeResult
        Alignment diagnostics and discrepancy metrics.
    """
    H, B = lp_result.irf_matrix.shape
    empirical_mat = lp_result.irf_matrix.copy()

    # Extract HANK matrix of shape (H, B)
    if isinstance(hank_irf, np.ndarray):
        arr = np.asarray(hank_irf, dtype=np.float64)
        if arr.shape == (H, B):
            hank_mat = arr.copy()
        elif arr.shape == (B, H):
            hank_mat = arr.T.copy()
        elif arr.ndim == 2 and arr.shape[1] == B and arr.shape[0] >= H:
            hank_mat = arr[:H, :].copy()
        else:
            raise ValueError(f"hank_irf array has shape {arr.shape}, expected ({H}, {B})")
    elif hasattr(hank_irf, "paths") and isinstance(getattr(hank_irf, "paths"), pd.DataFrame):
        df_paths = hank_irf.paths
        hank_mat = _extract_irf_from_frame(df_paths, lp_result.bins, H, bin_mapping)
    elif isinstance(hank_irf, pd.DataFrame):
        hank_mat = _extract_irf_from_frame(hank_irf, lp_result.bins, H, bin_mapping)
    elif isinstance(hank_irf, Mapping):
        cols = []
        for b in lp_result.bins:
            key = bin_mapping.get(b, b) if bin_mapping else b
            if key not in hank_irf:
                # Try finding matching key
                found = False
                for k in hank_irf:
                    if str(k).lower() == str(key).lower():
                        key = k
                        found = True
                        break
                if not found:
                    raise KeyError(f"Key {key!r} not found in hank_irf mapping.")
            v = np.asarray(hank_irf[key], dtype=np.float64).ravel()
            cols.append(v[:H])
        hank_mat = np.column_stack(cols)
    else:
        raise TypeError(f"Unsupported hank_irf type: {type(hank_irf)}")

    # 1. L2 Distance
    diff = empirical_mat - hank_mat
    distance_l2 = float(np.sqrt(np.mean(diff ** 2)))

    # 2. Mean Absolute Error
    mae = float(np.mean(np.abs(diff)))

    # 3. Sign Concordance
    sgn_emp = np.sign(empirical_mat)
    sgn_hank = np.sign(hank_mat)
    # Count as matching if both zero or exact same sign
    matches = (sgn_emp == sgn_hank)
    sign_concordance = float(np.mean(matches.astype(np.float64)))

    # 4. Decile-by-decile trajectory correlations & peak timing gaps
    corrs: dict[str, float] = {}
    peak_gaps: dict[str, int] = {}
    for j, b in enumerate(lp_result.bins):
        y_emp = empirical_mat[:, j]
        y_hnk = hank_mat[:, j]

        std_emp = float(np.std(y_emp))
        std_hnk = float(np.std(y_hnk))
        if std_emp > 1e-12 and std_hnk > 1e-12:
            r = float(np.corrcoef(y_emp, y_hnk)[0, 1])
            corrs[b] = max(-1.0, min(1.0, r))
        elif np.allclose(y_emp, y_hnk, atol=1e-10):
            corrs[b] = 1.0
        else:
            corrs[b] = 0.0

        p_emp = int(np.argmax(np.abs(y_emp)))
        p_hnk = int(np.argmax(np.abs(y_hnk)))
        peak_gaps[b] = p_emp - p_hnk

    # 5. Weighted Mahalanobis distance
    if weight_matrix is not None:
        W = np.asarray(weight_matrix, dtype=np.float64)
        if W.shape == (B, B):
            # Sum over horizons: mean_h (diff_h' W diff_h)
            quad = float(np.mean([float(diff[h] @ W @ diff[h]) for h in range(H)]))
            dist_mah = float(np.sqrt(max(0.0, quad)))
        elif W.shape == (H * B, H * B):
            vec = diff.ravel()
            dist_mah = float(np.sqrt(max(0.0, float(vec @ W @ vec) / (H * B))))
        else:
            raise ValueError(f"weight_matrix must have shape ({B}, {B}) or ({H*B}, {H*B}), got {W.shape}")
    else:
        # Inverse SE variance weighting
        se_sq = np.maximum(lp_result.se_matrix ** 2, 1e-8)
        dist_mah = float(np.sqrt(np.mean((diff ** 2) / se_sq)))

    return DistributionalBridgeResult(
        distance_l2=distance_l2,
        distance_mahalanobis=dist_mah,
        mean_absolute_error=mae,
        sign_concordance=sign_concordance,
        decile_correlations=corrs,
        peak_gaps=peak_gaps,
        empirical_irf=empirical_mat,
        hank_irf=hank_mat,
        bins=lp_result.bins,
        horizons=lp_result.horizons,
        metadata={"H": H, "B": B},
    )


def _extract_irf_from_frame(
    df: pd.DataFrame,
    bins: Sequence[str],
    H: int,
    bin_mapping: Mapping[str, str] | None,
) -> np.ndarray:
    """Helper to extract (H, B) array from a DataFrame matching bin column names."""
    cols = []
    df_cols = list(df.columns)
    for b in bins:
        target = bin_mapping.get(b, b) if bin_mapping else b
        col_found = None
        if target in df.columns:
            col_found = target
        else:
            # Case-insensitive or fuzzy match (e.g. "D1" matches "d1" or "c_d1" or "C_D1")
            for c in df_cols:
                c_clean = str(c).lower().replace("_", "").replace("-", "")
                t_clean = str(target).lower().replace("_", "").replace("-", "")
                if c_clean == t_clean or c_clean.endswith(t_clean) or t_clean in c_clean:
                    col_found = c
                    break
        if col_found is None:
            raise KeyError(f"Could not find matching column for bin {b!r} (target {target!r}) in DataFrame: {df_cols}")
        v = df[col_found].to_numpy(dtype=np.float64).ravel()
        if len(v) < H:
            # Pad with last value if shorter
            v_pad = np.pad(v, (0, H - len(v)), mode="edge")
            cols.append(v_pad)
        else:
            cols.append(v[:H])
    return np.column_stack(cols)
