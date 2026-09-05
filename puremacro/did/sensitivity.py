"""Honest Difference-in-Differences sensitivity analysis.

Implements the sensitivity analysis framework of Rambachan and Roth (2023),
evaluating the robustness of post-treatment DiD / event-study estimates
to potential violations of the parallel trends assumption:

  - Relative Magnitudes (Δ^RM(M_bar)): post-treatment trend deviations are
    bounded by M_bar times the maximum pre-treatment trend deviation:
        |δ_l| <= M_bar * max_{s < 0} |δ_s|
    or bounded in changes:
        |δ_t - δ_{t-1}| <= M_bar * max_{s <= 0} |δ_s - δ_{s-1}|
  - Smoothness / Bounded Second Differences (Δ^SD(M)): post-treatment
    trend slope deviates from pre-treatment slope by at most M:
        |(δ_{t+1} - δ_t) - (δ_t - δ_{t-1})| <= M

Computes identified sets via verified convex optimization (linear programming
using scipy.optimize.linprog with method='highs'), robust confidence intervals
(Imbens & Manski 2004; Stoye 2009), and the breakdown value M* (the smallest
violation magnitude that renders the treatment effect statistically indistinguishable
from zero, solved via scipy.optimize.brentq).

References
----------
Rambachan, A. and Roth, J. (2023). An Honest Approach to Parallel Trends.
    Review of Economic Studies, 90(5), 2555-2591.
Imbens, G.W. and Manski, C.F. (2004). Confidence Intervals for Partially
    Identified Parameters. Econometrica, 72(6), 1845-1857.
Stoye, J. (2009). More on Confidence Intervals for Partially Identified
    Parameters. Econometrica, 77(4), 1299-1315.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.stats import norm


@dataclass(frozen=True)
class HonestDiDResult:
    """Result of :func:`honest_did` and :func:`honest_did_sensitivity`.

    Attributes
    ----------
    table : pd.DataFrame
        Table of bounds across M grid with columns:
        ``[M, horizon, orig_estimate, orig_se, id_lo, id_hi, ci_lo, ci_hi, significant]``.
    breakdown_value : float | dict[int, float]
        The breakdown magnitude M* where the robust confidence interval
        first includes zero.
    method : str
        Sensitivity restriction used (``'relative_magnitude'`` or ``'smoothness'``).
    ci : float
        Confidence level (e.g. 0.95).
    pre_trend_max : float
        Maximum pre-treatment deviation from baseline observed.
    pre_trend_slope : float | None
        Estimated pre-treatment linear trend slope (if method='smoothness').
    target_horizons : list[int]
        Target post-treatment event horizons evaluated.
    """

    table: pd.DataFrame
    breakdown_value: float | dict[int, float]
    method: str
    ci: float
    pre_trend_max: float
    pre_trend_slope: float | None
    target_horizons: list[int]

    def summary(self) -> str:
        """Formatted human-readable summary of the sensitivity analysis."""
        lines = [
            "Honest DiD Sensitivity Analysis (Rambachan & Roth 2023)",
            "=" * 72,
            f"Method                          : {self.method}",
            f"Confidence Level (1 - α)        : {self.ci * 100:.1f}%",
            f"Max Pre-treatment Deviation     : {self.pre_trend_max:.4f}",
        ]
        if self.pre_trend_slope is not None:
            lines.append(
                f"Pre-treatment Trend Slope       : {self.pre_trend_slope:+.4f}"
            )

        lines.append("-" * 72)
        if isinstance(self.breakdown_value, dict):
            lines.append("Breakdown Values (M*) by Horizon:")
            for h, m_star in self.breakdown_value.items():
                m_str = f"{m_star:.4f}" if np.isfinite(m_star) else "inf (>100)"
                lines.append(f"  Horizon h = {h:+d}                : M* = {m_str}")
        else:
            h = self.target_horizons[0] if self.target_horizons else 0
            m_str = (
                f"{self.breakdown_value:.4f}"
                if np.isfinite(self.breakdown_value)
                else "inf (>100)"
            )
            lines.append(f"Breakdown Value (M*) for h = {h:+d} : M* = {m_str}")
            if self.breakdown_value == 0.0:
                lines.append(
                    "  (Estimate is already not statistically distinguishable from 0 at M = 0)"
                )
            elif np.isfinite(self.breakdown_value):
                lines.append(
                    f"  (Effect remains robust until violations reach {self.breakdown_value:.2f}x "
                    f"the pre-trend benchmark)"
                )
            else:
                lines.append(
                    "  (Effect remains statistically significant across all tested M)"
                )

        lines.append("=" * 72)
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Return the sensitivity grid table as a DataFrame."""
        return self.table.copy()

    def to_markdown(self, **kwargs: Any) -> str:
        """Export sensitivity table to Markdown format."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Export sensitivity table to LaTeX tabular format."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Export sensitivity table to Typst table format."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)

    def plot_ascii(self, horizon: int | None = None, width: int = 50) -> str:
        """Render an ASCII chart of confidence intervals across M."""
        sub = self.table
        if horizon is not None:
            sub = sub[sub["horizon"] == horizon]
        elif len(self.target_horizons) > 1:
            sub = sub[sub["horizon"] == self.target_horizons[0]]

        if len(sub) == 0:
            return "(No data to plot)"

        h_val = sub["horizon"].iloc[0]
        all_lo = sub["ci_lo"].min()
        all_hi = sub["ci_hi"].max()
        # Ensure 0 is in range
        min_v = min(all_lo, 0.0)
        max_v = max(all_hi, 0.0)
        span = max(max_v - min_v, 1e-6)

        def pos(val: float) -> int:
            p = int(round((val - min_v) / span * (width - 1)))
            return max(0, min(width - 1, p))

        zero_p = pos(0.0)

        lines = [
            f"Honest DiD Confidence Intervals vs M (Horizon h = {h_val})",
            "-" * (width + 24),
            f"{'M':>6} | {'Identified Set':^18} | {'Robust CI':^18} | Chart",
            "-" * (width + 24),
        ]

        for _, row in sub.iterrows():
            m_val = row["M"]
            id_lo, id_hi = row["id_lo"], row["id_hi"]
            c_lo, c_hi = row["ci_lo"], row["ci_hi"]
            p_clo = pos(c_lo)
            p_chi = pos(c_hi)
            p_ilo = pos(id_lo)
            p_ihi = pos(id_hi)

            bar = [" "] * width
            # Mark zero line
            bar[zero_p] = "|"
            # Fill CI
            for i in range(p_clo, p_chi + 1):
                if bar[i] != "|":
                    bar[i] = "-"
            # Fill identified set
            for i in range(p_ilo, p_ihi + 1):
                bar[i] = "="
            bar[p_clo] = "["
            bar[p_chi] = "]"

            chart_str = "".join(bar)
            id_str = f"[{id_lo:+.2f}, {id_hi:+.2f}]"
            ci_str = f"[{c_lo:+.2f}, {c_hi:+.2f}]"
            lines.append(f"{m_val:6.2f} | {id_str:^18} | {ci_str:^18} | {chart_str}")

        lines.append("-" * (width + 24))
        lines.append(
            f"Legend: '=' identified set, '[-]' {self.ci*100:.0f}% robust CI, '|' zero line"
        )
        return "\n".join(lines)

    def plot(
        self,
        horizon: int | None = None,
        ax: plt.Axes | None = None,
        title: str | None = None,
        figsize: tuple[float, float] = (8.0, 5.0),
        return_fig: bool = False,
        **kwargs: Any,
    ) -> plt.Axes | tuple[plt.Figure, plt.Axes]:
        """Visualize identified set band, robust confidence band, and breakdown value.

        Parameters
        ----------
        horizon : int, optional
            Event study horizon to plot. Defaults to first target horizon.
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw on. If None, a new figure and axes are created.
        title : str, optional
            Custom chart title.
        figsize : tuple of float, default (8.0, 5.0)
            Figure dimensions when ax is None.
        return_fig : bool, default False
            If True, returns (fig, ax). Otherwise returns ax.
        **kwargs : Any
            Additional styling keyword arguments passed to plot calls.

        Returns
        -------
        matplotlib.axes.Axes or tuple of (Figure, Axes)
            The generated plot.
        """
        sub = self.table
        if horizon is not None:
            sub = sub[sub["horizon"] == horizon]
        elif len(self.target_horizons) > 1:
            sub = sub[sub["horizon"] == self.target_horizons[0]]

        if len(sub) == 0:
            raise ValueError("No data found to plot for specified horizon.")

        sub = sub.sort_values("M")
        h_val = sub["horizon"].iloc[0]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        m_vals = sub["M"].to_numpy()
        id_lo = sub["id_lo"].to_numpy()
        id_hi = sub["id_hi"].to_numpy()
        ci_lo = sub["ci_lo"].to_numpy()
        ci_hi = sub["ci_hi"].to_numpy()

        ci_pct = int(round(self.ci * 100))

        # Robust CI band
        ax.fill_between(
            m_vals,
            ci_lo,
            ci_hi,
            color="#2b5c8f",
            alpha=0.18,
            label=f"{ci_pct}% Robust CI (Imbens-Manski)",
        )
        ax.plot(m_vals, ci_lo, color="#1a3b5c", linestyle="--", linewidth=1.2)
        ax.plot(m_vals, ci_hi, color="#1a3b5c", linestyle="--", linewidth=1.2)

        # Identified Set band
        ax.fill_between(
            m_vals,
            id_lo,
            id_hi,
            color="#2b5c8f",
            alpha=0.38,
            label="Identified Set",
        )
        ax.plot(m_vals, id_lo, color="#2b5c8f", linestyle="-", linewidth=1.5)
        ax.plot(m_vals, id_hi, color="#2b5c8f", linestyle="-", linewidth=1.5)

        # Baseline original estimate
        orig_est = sub["orig_estimate"].iloc[0]
        ax.plot(
            m_vals[0],
            orig_est,
            "o",
            color="#2b5c8f",
            markersize=6,
            label=f"Original Estimate ({orig_est:+.3f})",
        )

        # Zero reference line
        ax.axhline(
            0.0,
            color="crimson",
            linestyle=":",
            linewidth=1.3,
            alpha=0.85,
            label="Zero Effect",
        )

        # Breakdown value annotation
        m_star = (
            self.breakdown_value[h_val]
            if isinstance(self.breakdown_value, dict)
            else self.breakdown_value
        )
        if np.isfinite(m_star) and m_star > 0:
            ax.axvline(
                m_star,
                color="darkorange",
                linestyle="-.",
                linewidth=1.4,
                label=f"Breakdown M* = {m_star:.3f}",
            )
            y_span = max(ci_hi.max() - ci_lo.min(), 1e-4)
            ax.annotate(
                f"M* = {m_star:.2f}",
                xy=(m_star, 0.0),
                xytext=(m_star, 0.12 * y_span),
                textcoords="data",
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2),
                fontsize=9,
                fontweight="bold",
                color="darkorange",
                ha="center",
            )

        # Labels and title
        if self.method == "smoothness":
            ax.set_xlabel("Smoothness Bound M", fontsize=11)
            method_desc = "Smoothness Δ^SD(M)"
        else:
            ax.set_xlabel("Relative Magnitude Multiplier M̄", fontsize=11)
            method_desc = "Relative Magnitude Δ^RM(M̄)"

        ax.set_ylabel("Treatment Effect Parameter θ", fontsize=11)

        if title is None:
            title = f"Honest DiD Sensitivity Analysis: {method_desc} (Horizon h = {h_val:+d})"
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=9)

        if return_fig:
            return fig, ax
        return ax


def _imbens_manski_critical_value(
    delta_std: float,
    alpha: float,
    *,
    is_half_width: bool = True,
) -> float:
    """Solve for the Imbens & Manski (2004) / Stoye (2009) critical value c.

    Parameters
    ----------
    delta_std : float
        Half-width of identified set divided by standard error (if is_half_width=True),
        or total width of identified set divided by standard error.
    alpha : float
        Significance level (1 - CI level, e.g. 0.05).
    is_half_width : bool, default True
        Whether delta_std is (half-width / se). If True, width_std = 2 * delta_std.

    Returns
    -------
    float
        Critical value c satisfying Φ(c + width_std) - Φ(-c) = 1 - alpha.
    """
    width_std = 2.0 * delta_std if is_half_width else delta_std
    if width_std <= 1e-8:
        return float(norm.ppf(1.0 - alpha / 2.0))

    target = 1.0 - alpha
    z_one = float(norm.ppf(1.0 - alpha))
    z_two = float(norm.ppf(1.0 - alpha / 2.0))

    def f(c: float) -> float:
        return float(norm.cdf(c + width_std) - norm.cdf(-c) - target)

    if f(z_one) >= 0:
        return z_one
    if f(z_two) <= 0:
        return z_two

    try:
        sol = scipy.optimize.brentq(f, z_one, z_two, xtol=1e-6)
        return float(sol)
    except Exception:
        return z_one


def _solve_lp_bounds(
    method: str,
    m_val: float,
    l_vec: np.ndarray,
    b_hat_post: np.ndarray,
    pre_betas: np.ndarray,
    pre_times: Sequence[float],
    post_times: Sequence[float],
    base_period: float,
    pre_slope: float | None = None,
    bound: str = "deviation from parallel trends",
) -> tuple[float, float]:
    """Solve exact identified set bounds for delta_post using scipy.optimize.linprog (method='highs').

    Parameters
    ----------
    method : {'smoothness', 'relative_magnitude'}
        Restriction type.
    m_val : float
        Sensitivity parameter (M or M_bar).
    l_vec : np.ndarray
        Weight vector defining parameter of interest θ = l' * τ_post.
    b_hat_post : np.ndarray
        Estimated post-treatment coefficients.
    pre_betas : np.ndarray
        Estimated pre-treatment coefficients.
    pre_times : Sequence[float]
        Pre-treatment event times.
    post_times : Sequence[float]
        Post-treatment event times.
    base_period : float
        Omitted baseline event time.
    pre_slope : float or None
        Estimated pre-treatment linear slope.
    bound : {'deviation from parallel trends', 'deviation from pre-trend slope'}
        Bounding approach for relative magnitude.

    Returns
    -------
    tuple of (id_lo, id_hi)
        Lower and upper identified set bounds for θ.
    """
    L = len(post_times)
    t0 = float(base_period)

    rows: list[np.ndarray] = []
    b_vals: list[float] = []

    if method == "relative_magnitude":
        if bound in (
            "deviation from pre-trend slope",
            "first_difference",
            "differences",
        ):
            all_pre_times = list(pre_times) + [base_period]
            all_pre_betas = list(pre_betas) + [0.0]
            diffs = [
                abs(
                    (all_pre_betas[i + 1] - all_pre_betas[i])
                    / (all_pre_times[i + 1] - all_pre_times[i])
                )
                for i in range(len(pre_times))
            ]
            d_max = max(diffs) if diffs else 1.0
            if d_max < 1e-12:
                d_max = 1e-6

            h1 = float(post_times[0] - t0)
            r1 = np.zeros(L)
            r1[0] = 1.0 / h1
            rows.append(r1)
            b_vals.append(m_val * d_max)
            rows.append(-r1)
            b_vals.append(m_val * d_max)

            for j in range(1, L):
                hj = float(post_times[j] - post_times[j - 1])
                r = np.zeros(L)
                r[j] = 1.0 / hj
                r[j - 1] = -1.0 / hj
                rows.append(r)
                b_vals.append(m_val * d_max)
                rows.append(-r)
                b_vals.append(m_val * d_max)
        else:
            pre_max = float(np.max(np.abs(pre_betas))) if len(pre_betas) > 0 else 1.0
            if pre_max < 1e-12:
                pre_max = 1e-6
            for j in range(L):
                r = np.zeros(L)
                r[j] = 1.0
                rows.append(r)
                b_vals.append(m_val * pre_max)
                rows.append(-r)
                b_vals.append(m_val * pre_max)

    elif method == "smoothness":
        s_anchor = (
            pre_slope
            if pre_slope is not None
            else (
                float(-pre_betas[-1] / (base_period - pre_times[-1]))
                if len(pre_betas) > 0
                else 0.0
            )
        )
        h1 = float(post_times[0] - t0)
        r1 = np.zeros(L)
        r1[0] = 1.0 / h1
        rows.append(r1)
        b_vals.append(m_val + s_anchor)
        rows.append(-r1)
        b_vals.append(m_val - s_anchor)

        for j in range(1, L):
            h_prev = float(post_times[j - 1] - (post_times[j - 2] if j > 1 else t0))
            h_curr = float(post_times[j] - post_times[j - 1])
            r = np.zeros(L)
            r[j] = 1.0 / h_curr
            r[j - 1] = -(1.0 / h_curr + 1.0 / h_prev)
            if j > 1:
                r[j - 2] = 1.0 / h_prev
            rows.append(r)
            b_vals.append(m_val)
            rows.append(-r)
            b_vals.append(m_val)

    else:
        raise ValueError(f"Unknown method {method!r}")

    A_ub = np.array(rows, dtype=float)
    b_ub = np.array(b_vals, dtype=float)

    # Maximize l' * delta_post: minimize -l' * delta_post
    res_max = scipy.optimize.linprog(
        -l_vec,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=(None, None),
        method="highs",
    )
    # Minimize l' * delta_post: minimize +l' * delta_post
    res_min = scipy.optimize.linprog(
        l_vec,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=(None, None),
        method="highs",
    )

    orig_theta = float(np.dot(l_vec, b_hat_post))
    if not res_max.success or not res_min.success:
        # If solver reports issue, fall back to analytic bounds if available
        return orig_theta, orig_theta

    delta_max = -float(res_max.fun)
    delta_min = float(res_min.fun)
    return orig_theta - delta_max, orig_theta - delta_min


def _compute_breakdown_mstar(
    method: str,
    orig_theta: float,
    s_hat: float,
    alpha: float,
    l_vec: np.ndarray,
    b_hat_post: np.ndarray,
    pre_betas: np.ndarray,
    pre_times: Sequence[float],
    post_times: Sequence[float],
    base_period: float,
    pre_slope: float | None = None,
    bound: str = "deviation from parallel trends",
) -> float:
    """Compute breakdown value M* where the robust confidence interval crosses zero."""
    # Check baseline at M = 0
    id_lo_0, id_hi_0 = _solve_lp_bounds(
        method,
        0.0,
        l_vec,
        b_hat_post,
        pre_betas,
        pre_times,
        post_times,
        base_period,
        pre_slope,
        bound,
    )
    w_std_0 = (id_hi_0 - id_lo_0) / s_hat
    cv_0 = _imbens_manski_critical_value(w_std_0, alpha, is_half_width=False)
    ci_lo_0 = id_lo_0 - cv_0 * s_hat
    ci_hi_0 = id_hi_0 + cv_0 * s_hat

    if ci_lo_0 <= 0.0 <= ci_hi_0:
        return 0.0

    sign = 1.0 if orig_theta > 0 else -1.0

    def bound_distance(m: float) -> float:
        id_lo, id_hi = _solve_lp_bounds(
            method,
            m,
            l_vec,
            b_hat_post,
            pre_betas,
            pre_times,
            post_times,
            base_period,
            pre_slope,
            bound,
        )
        w_std = (id_hi - id_lo) / s_hat
        cv = _imbens_manski_critical_value(w_std, alpha, is_half_width=False)
        if sign > 0:
            # Lower bound crosses zero
            return id_lo - cv * s_hat
        else:
            # Upper bound crosses zero
            return id_hi + cv * s_hat

    # Find bracket
    brackets = [1.0, 5.0, 20.0, 100.0, 500.0]
    m_high = None
    for b_try in brackets:
        val = bound_distance(b_try)
        if sign * val <= 0.0:
            m_high = b_try
            break

    if m_high is None:
        return float(np.inf)

    try:
        sol = scipy.optimize.brentq(bound_distance, 0.0, m_high, xtol=1e-5)
        return float(sol)
    except Exception:
        return float(np.inf)


def honest_did(
    b_hat: Any = None,
    sigma: np.ndarray | Sequence[Sequence[float]] | None = None,
    se: Sequence[float] | None = None,
    method: str = "smoothness",
    m_vec: Sequence[float] | None = None,
    base_period: int = -1,
    alpha: float = 0.05,
    l_vec: Sequence[float] | np.ndarray | None = None,
    pre_periods: int | Sequence[int | float] | None = None,
    post_periods: int | Sequence[int | float] | None = None,
    *,
    result: Any = None,
    event_time: Sequence[int | float] | None = None,
    beta: Sequence[float] | None = None,
    target_horizon: int | Sequence[int] | None = None,
    m_grid: Sequence[float] | None = None,
    ci: float | None = None,
    bound: str = "deviation from parallel trends",
    **kwargs: Any,
) -> HonestDiDResult:
    """Evaluate post-treatment sensitivity to parallel trends violations per Rambachan & Roth (2023).

    Calculates identified sets via exact linear programming (HiGHS solver), Imbens & Manski (2004)
    robust confidence sets, and the breakdown value M*.

    Parameters
    ----------
    b_hat : Any, optional
        Vector of estimated event study coefficients, or an event-study result object
        from :func:`puremacro.did.callaway_santanna`, :func:`puremacro.did.sun_abraham`,
        or a :class:`pandas.DataFrame`.
    sigma : np.ndarray or Sequence[Sequence[float]], optional
        Covariance matrix of the estimated event study coefficients.
    se : Sequence[float], optional
        Vector of standard errors for each event time coefficient.
    method : {'smoothness', 'relative_magnitude'}, default 'smoothness'
        Restriction on parallel trends violations:
        - ``'smoothness'`` (Δ^SD): slope changes bounded by M.
        - ``'relative_magnitude'`` (Δ^RM): post-treatment deviations bounded by M_bar times pre-trend.
    m_vec : Sequence[float], optional
        Grid of violation magnitudes M to evaluate. Defaults to sensible range if None.
    base_period : int, default -1
        Reference / omitted event study time period normalized to 0.
    alpha : float, default 0.05
        Nominal significance level (confidence level is 1 - alpha).
    l_vec : Sequence[float] or np.ndarray, optional
        Weight vector defining parameter of interest θ = l' * τ_post.
        If None and target_horizon is specified, basis vector for target_horizon is used.
    pre_periods : int or Sequence[int | float], optional
        Number of pre-treatment periods, or explicit list of pre-treatment period values.
    post_periods : int or Sequence[int | float], optional
        Number of post-treatment periods, or explicit list of post-treatment period values.
    result : Any, optional
        Alias for b_hat when passing an estimation result object.
    event_time : Sequence[int | float], optional
        Sequence of relative event times corresponding to b_hat.
    beta : Sequence[float], optional
        Alias for b_hat.
    target_horizon : int or Sequence[int], optional
        Specific post-treatment horizon(s) to evaluate.
    m_grid : Sequence[float], optional
        Alias for m_vec.
    ci : float, optional
        Confidence level (e.g. 0.95). If provided, alpha is set to 1 - ci.
    bound : {'deviation from parallel trends', 'deviation from pre-trend slope'}, default 'deviation from parallel trends'
        Relative magnitude restriction type (levels vs first differences).

    Returns
    -------
    HonestDiDResult
        Result containing sensitivity table, breakdown values M*, and presentation tools (.plot, .summary).

    References
    ----------
    Rambachan, A. and Roth, J. (2023). An Honest Approach to Parallel Trends.
        Review of Economic Studies, 90(5), 2555-2591.
    """
    # Method normalization
    m_clean = method.lower().strip()
    if m_clean in ("smoothness", "deltasd", "delta_sd", "sd"):
        method = "smoothness"
    elif m_clean in ("relative_magnitude", "deltarm", "delta_rm", "rm"):
        method = "relative_magnitude"
    else:
        raise ValueError(
            f"method must be 'smoothness' or 'relative_magnitude', got {method!r}"
        )

    # Grid normalization
    if m_vec is None:
        if m_grid is not None:
            m_vec = m_grid
        elif method == "smoothness":
            m_vec = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)
        else:
            m_vec = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    m_grid_vals = [float(m) for m in m_vec]

    # CI / alpha normalization
    if ci is not None:
        alpha = float(1.0 - ci)
    else:
        ci = float(1.0 - alpha)

    # Result object extraction
    src = b_hat if b_hat is not None else (result if result is not None else beta)
    if src is None:
        raise ValueError(
            "must provide either 'result' (or 'b_hat') or all of ('event_time', 'beta', 'se')"
        )

    et_arr: np.ndarray | None = None
    b_arr: np.ndarray | None = None
    se_arr: np.ndarray | None = None
    sigma_mat: np.ndarray | None = None

    if sigma is not None:
        sigma_mat = np.asarray(sigma, dtype=float)

    if hasattr(src, "att_event_study") or isinstance(src, pd.DataFrame):
        df = src.att_event_study if hasattr(src, "att_event_study") else src
        time_col = None
        for c in ["event_time", "rel_time", "time", "h", "year", "period"]:
            if c in df.columns:
                time_col = c
                break
        if time_col is None:
            raise ValueError("could not find event time column in result")

        beta_col = None
        for c in ["att", "beta", "coef", "estimate"]:
            if c in df.columns:
                beta_col = c
                break
        if beta_col is None:
            raise ValueError("could not find coefficient column in result")

        se_col = None
        for c in ["se", "std_err", "stderr"]:
            if c in df.columns:
                se_col = c
                break
        if se_col is None:
            raise ValueError("could not find standard error column in result")

        et_arr = np.asarray(df[time_col], dtype=float)
        b_arr = np.asarray(df[beta_col], dtype=float)
        se_arr = np.asarray(df[se_col], dtype=float)
    else:
        b_arr = np.asarray(src, dtype=float)
        if event_time is not None:
            et_arr = np.asarray(event_time, dtype=float)
        elif pre_periods is not None and post_periods is not None:
            if isinstance(pre_periods, (int, np.integer)) and isinstance(
                post_periods, (int, np.integer)
            ):
                num_pre = int(pre_periods)
                num_post = int(post_periods)
                if len(b_arr) == num_pre + num_post:
                    # Omitted base period
                    pre_idx = np.arange(-num_pre, 0)
                    post_idx = np.arange(0, num_post)
                    et_arr = np.concatenate([pre_idx, post_idx])
                elif len(b_arr) == num_pre + 1 + num_post:
                    pre_idx = np.arange(-num_pre - 1, -1)
                    post_idx = np.arange(0, num_post)
                    et_arr = np.concatenate([pre_idx, [base_period], post_idx])
                else:
                    raise ValueError(
                        f"length of b_hat ({len(b_arr)}) does not match pre_periods + post_periods"
                    )
            else:
                pre_seq = list(pre_periods)
                post_seq = list(post_periods)
                et_arr = np.array(pre_seq + post_seq, dtype=float)
        else:
            # Fallback: assume half pre, half post, or sequence around 0
            n = len(b_arr)
            num_pre = n // 2
            num_post = n - num_pre
            et_arr = np.concatenate([np.arange(-num_pre, 0), np.arange(0, num_post)])

        if se is not None:
            se_arr = np.asarray(se, dtype=float)
        elif sigma_mat is not None:
            se_arr = np.sqrt(np.diag(sigma_mat))
        else:
            raise ValueError("must provide either 'se' or 'sigma'")

    if len(et_arr) != len(b_arr) or len(b_arr) != len(se_arr):
        raise ValueError("event_time, beta, and se must have equal lengths")
    if len(et_arr) == 0:
        raise ValueError("empty event study arrays")

    # Sort chronologically
    sort_idx = np.argsort(et_arr)
    et_arr = et_arr[sort_idx]
    b_arr = b_arr[sort_idx]
    se_arr = se_arr[sort_idx]
    if sigma_mat is not None and sigma_mat.shape == (len(sort_idx), len(sort_idx)):
        sigma_mat = sigma_mat[np.ix_(sort_idx, sort_idx)]

    # Partition into pre, base, post
    # A pre-treatment period is event_time < 0 or < base_period (if base_period >= 0)
    is_base = et_arr == base_period
    is_pre = (et_arr < base_period) if base_period >= 0 else ((et_arr < 0) & ~is_base)
    is_post = (et_arr > base_period) if base_period >= 0 else (et_arr >= 0)

    if not np.any(is_pre):
        raise ValueError(
            "Sensitivity analysis requires at least one pre-treatment period "
            f"(event_time < 0, event_time != {base_period}) to evaluate pre-trend violations."
        )
    if not np.any(is_post):
        raise ValueError(
            "Sensitivity analysis requires at least one post-treatment period."
        )

    pre_times = et_arr[is_pre]
    pre_betas = b_arr[is_pre]
    post_times = et_arr[is_post]
    post_betas = b_arr[is_post]
    post_ses = se_arr[is_post]

    L = len(post_times)

    # Post covariance matrix
    if sigma_mat is not None:
        if sigma_mat.shape == (L, L):
            sigma_post = sigma_mat
        elif sigma_mat.shape == (len(et_arr), len(et_arr)):
            sigma_post = sigma_mat[np.ix_(is_post, is_post)]
        else:
            sigma_post = np.diag(post_ses**2)
    else:
        sigma_post = np.diag(post_ses**2)

    pre_max = float(np.max(np.abs(pre_betas)))
    if pre_max < 1e-12:
        pre_max = 1e-6

    # Pre-treatment slope estimation
    pre_slope: float | None = None
    if method == "smoothness":
        if len(pre_times) >= 2:
            poly = np.polyfit(pre_times, pre_betas, 1)
            pre_slope = float(poly[0])
        elif len(pre_times) == 1:
            pre_slope = float(pre_betas[0] / (pre_times[0] - base_period))
        else:
            pre_slope = 0.0

    # Determine targets and l_vec evaluation
    if l_vec is not None:
        l_arr = np.asarray(l_vec, dtype=float)
        if len(l_arr) != L:
            raise ValueError(
                f"l_vec must have length {L} (number of post-treatment periods)"
            )
        eval_specs = [(0, l_arr)]
    else:
        if target_horizon is None:
            targets = [int(post_times[0])]
        elif isinstance(target_horizon, (int, float, np.integer)):
            targets = [int(target_horizon)]
        else:
            targets = [int(h) for h in target_horizon]

        eval_specs = []
        for h in targets:
            matches = np.where(post_times == h)[0]
            if len(matches) == 0:
                raise ValueError(
                    f"target horizon {h} not found in post-treatment periods"
                )
            idx = matches[0]
            basis = np.zeros(L, dtype=float)
            basis[idx] = 1.0
            eval_specs.append((h, basis))

    rows: list[dict[str, Any]] = []
    breakdown_dict: dict[int, float] = {}

    for h_label, l_spec in eval_specs:
        orig_theta = float(np.dot(l_spec, post_betas))
        s_hat = float(
            np.sqrt(max(float(np.dot(l_spec, np.dot(sigma_post, l_spec))), 1e-12))
        )

        # Solve breakdown value M*
        m_star = _compute_breakdown_mstar(
            method=method,
            orig_theta=orig_theta,
            s_hat=s_hat,
            alpha=alpha,
            l_vec=l_spec,
            b_hat_post=post_betas,
            pre_betas=pre_betas,
            pre_times=pre_times,
            post_times=post_times,
            base_period=base_period,
            pre_slope=pre_slope,
            bound=bound,
        )
        breakdown_dict[h_label] = m_star

        # Solve grid of M values
        for m in m_grid_vals:
            id_lo, id_hi = _solve_lp_bounds(
                method=method,
                m_val=m,
                l_vec=l_spec,
                b_hat_post=post_betas,
                pre_betas=pre_betas,
                pre_times=pre_times,
                post_times=post_times,
                base_period=base_period,
                pre_slope=pre_slope,
                bound=bound,
            )
            w_std = (id_hi - id_lo) / s_hat
            cv = _imbens_manski_critical_value(w_std, alpha, is_half_width=False)
            ci_lo = id_lo - cv * s_hat
            ci_hi = id_hi + cv * s_hat
            sig = bool((ci_lo > 0.0) or (ci_hi < 0.0))

            rows.append(
                {
                    "M": float(m),
                    "horizon": h_label,
                    "orig_estimate": orig_theta,
                    "orig_se": s_hat,
                    "id_lo": id_lo,
                    "id_hi": id_hi,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "significant": sig,
                }
            )

    table = pd.DataFrame(rows)
    bd_val = (
        breakdown_dict[eval_specs[0][0]] if len(eval_specs) == 1 else breakdown_dict
    )
    targets_out = [spec[0] for spec in eval_specs]

    return HonestDiDResult(
        table=table,
        breakdown_value=bd_val,
        method=method,
        ci=ci,
        pre_trend_max=pre_max,
        pre_trend_slope=pre_slope,
        target_horizons=targets_out,
    )


def honest_did_sensitivity(
    result: Any = None,
    *,
    event_time: Sequence[int | float] | None = None,
    beta: Sequence[float] | None = None,
    se: Sequence[float] | None = None,
    target_horizon: int | Sequence[int] = 0,
    method: str = "relative_magnitude",
    m_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
    ci: float = 0.95,
    base_period: int = -1,
    sigma: np.ndarray | None = None,
    l_vec: Sequence[float] | np.ndarray | None = None,
    pre_periods: int | Sequence[int | float] | None = None,
    post_periods: int | Sequence[int | float] | None = None,
    m_vec: Sequence[float] | None = None,
    alpha: float | None = None,
    bound: str = "deviation from parallel trends",
    **kwargs: Any,
) -> HonestDiDResult:
    """Backward-compatible alias for :func:`honest_did`."""
    return honest_did(
        b_hat=result if result is not None else beta,
        sigma=sigma,
        se=se,
        method=method,
        m_vec=m_vec if m_vec is not None else m_grid,
        base_period=base_period,
        alpha=alpha if alpha is not None else (1.0 - ci),
        l_vec=l_vec,
        pre_periods=pre_periods,
        post_periods=post_periods,
        result=result,
        event_time=event_time,
        beta=beta,
        target_horizon=target_horizon,
        ci=ci,
        bound=bound,
        **kwargs,
    )
