"""Honest Difference-in-Differences sensitivity analysis.

Implements the sensitivity analysis framework of Rambachan and Roth (2023),
evaluating the robustness of post-treatment DiD / event-study estimates
to potential violations of the parallel trends assumption:

  - Relative Magnitudes (Δ^RM(M)): post-treatment trend deviations are
    bounded by M times the maximum pre-treatment trend deviation:
        |δ_l| <= M * max_{s < 0} |δ_s|
  - Smoothness / Bounded Second Differences (Δ^SD(M)): post-treatment
    trend slope deviates from pre-treatment slope by at most M:
        |(δ_{t+1} - δ_t) - (δ_t - δ_{t-1})| <= M

Computes identified sets, robust confidence intervals (Imbens & Manski 2004;
Stoye 2009), and the breakdown value M* (the smallest violation magnitude
that renders the treatment effect statistically indistinguishable from zero).

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

import numpy as np
import pandas as pd
import scipy.optimize
from scipy.stats import norm


@dataclass(frozen=True)
class HonestDiDResult:
    """Result of :func:`honest_did_sensitivity`.

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
            lines.append(f"Pre-treatment Trend Slope       : {self.pre_trend_slope:+.4f}")

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
                lines.append("  (Estimate is already not statistically distinguishable from 0 at M = 0)")
            elif np.isfinite(self.breakdown_value):
                lines.append(
                    f"  (Effect remains robust until violations reach {self.breakdown_value:.2f}x "
                    f"the pre-trend benchmark)"
                )
            else:
                lines.append("  (Effect remains statistically significant across all tested M)")

        lines.append("=" * 72)
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Return the sensitivity grid table as a DataFrame."""
        return self.table.copy()

    def to_markdown(self, **kwargs) -> str:
        """Export sensitivity table to Markdown format."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export sensitivity table to LaTeX tabular format."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
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
        lines.append(f"Legend: '=' identified set, '[-]' {self.ci*100:.0f}% robust CI, '|' zero line")
        return "\n".join(lines)


def _imbens_manski_critical_value(delta_std: float, alpha: float) -> float:
    """Solve for the Imbens & Manski (2004) critical value c.

    Parameters
    ----------
    delta_std : float
        Half-width of identified set divided by standard error: (Δ / se).
    alpha : float
        Significance level (1 - CI level, e.g. 0.05).

    Returns
    -------
    float
        Critical value c satisfying Φ(c + 2*delta_std) - Φ(-c) = 1 - alpha.
    """
    if delta_std <= 1e-8:
        return float(norm.ppf(1.0 - alpha / 2.0))

    target = 1.0 - alpha
    z_one = float(norm.ppf(1.0 - alpha))
    z_two = float(norm.ppf(1.0 - alpha / 2.0))

    def f(c: float) -> float:
        return float(norm.cdf(c + 2.0 * delta_std) - norm.cdf(-c) - target)

    if f(z_one) >= 0:
        return z_one
    if f(z_two) <= 0:
        return z_two

    try:
        sol = scipy.optimize.brentq(f, z_one, z_two, xtol=1e-6)
        return float(sol)
    except Exception:
        return z_one


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
) -> HonestDiDResult:
    """Evaluate sensitivity of DiD / event-study estimates to parallel trends violations.

    Parameters
    ----------
    result : Any, optional
        An event-study result object from :func:`puremacro.did.callaway_santanna`,
        :func:`puremacro.did.sun_abraham`, :func:`puremacro.did.borusyak_jaravel_spiess`,
        or a :class:`pandas.DataFrame` with event study estimates.
    event_time : Sequence[int | float], optional
        Event time relative to treatment (e.g. ``[-4, -3, -2, -1, 0, 1, 2]``).
    beta : Sequence[float], optional
        Point estimates for each event time.
    se : Sequence[float], optional
        Standard errors for each event time.
    target_horizon : int or Sequence[int], default 0
        Post-treatment event time(s) to evaluate sensitivity for.
    method : {'relative_magnitude', 'smoothness'}, default 'relative_magnitude'
        Restriction on parallel trends violations:
        - ``'relative_magnitude'`` (Δ^RM): violations bounded by M * max pre-trend.
        - ``'smoothness'`` (Δ^SD): slope changes bounded by M.
    m_grid : Sequence[float], default (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
        Grid of violation multipliers M to evaluate.
    ci : float, default 0.95
        Nominal confidence level.
    base_period : int, default -1
        Omitted / reference pre-treatment period (where deviation is 0).

    Returns
    -------
    HonestDiDResult
        Result containing sensitivity table, breakdown values, and reporting tools.

    References
    ----------
    Rambachan, A. and Roth, J. (2023). An Honest Approach to Parallel Trends.
        Review of Economic Studies, 90(5), 2555-2591.
    """
    if method not in ("relative_magnitude", "smoothness"):
        raise ValueError(
            f"method must be 'relative_magnitude' or 'smoothness', got {method!r}"
        )

    # Extract event_time, beta, se from result if passed
    if result is not None:
        if hasattr(result, "att_event_study"):
            df = result.att_event_study
        elif isinstance(result, pd.DataFrame):
            df = result
        else:
            raise TypeError(
                f"unsupported result type: {type(result).__name__}; "
                "expected CallawaySantannaResult, SunAbrahamResult, or DataFrame."
            )

        # Look up column names flexibly
        time_col = None
        for c in ["event_time", "rel_time", "time", "h"]:
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
        if event_time is None or beta is None or se is None:
            raise ValueError(
                "must provide either 'result' or all of ('event_time', 'beta', 'se')"
            )
        et_arr = np.asarray(event_time, dtype=float)
        b_arr = np.asarray(beta, dtype=float)
        se_arr = np.asarray(se, dtype=float)

    if len(et_arr) != len(b_arr) or len(b_arr) != len(se_arr):
        raise ValueError("event_time, beta, and se must have equal lengths")
    if len(et_arr) == 0:
        raise ValueError("empty event study arrays")

    # Find pre-treatment observations (event_time < 0 and != base_period)
    is_pre = (et_arr < 0) & (et_arr != base_period)
    if not np.any(is_pre):
        raise ValueError(
            "Sensitivity analysis requires at least one pre-treatment period "
            f"(event_time < 0, event_time != {base_period}) to evaluate pre-trend violations."
        )

    pre_betas = b_arr[is_pre]
    pre_max = float(np.max(np.abs(pre_betas)))
    if pre_max < 1e-12:
        pre_max = 1e-6  # numeric stability floor

    pre_slope = None
    if method == "smoothness":
        # Estimate pre-treatment linear slope
        pre_times = et_arr[is_pre]
        if len(pre_times) >= 2:
            poly = np.polyfit(pre_times, pre_betas, 1)
            pre_slope = float(poly[0])
        else:
            pre_slope = float(pre_betas[0] / (pre_times[0] - base_period))

    # Determine target horizons
    if isinstance(target_horizon, (int, float)):
        targets = [int(target_horizon)]
    else:
        targets = [int(h) for h in target_horizon]

    alpha = 1.0 - ci
    rows = []
    breakdown_dict: dict[int, float] = {}

    for h in targets:
        # Find row for horizon h
        matches = np.where(et_arr == h)[0]
        if len(matches) == 0:
            raise ValueError(f"target horizon {h} not found in event_time")
        idx = matches[0]
        b_hat = float(b_arr[idx])
        s_hat = float(se_arr[idx])

        # Compute breakdown value M*
        # Baseline at M=0
        z_crit_base = float(norm.ppf(1.0 - alpha / 2.0))
        is_sig_base = abs(b_hat) > z_crit_base * s_hat

        if not is_sig_base:
            m_star = 0.0
        else:
            # Root find for M where CI crosses zero
            sign = 1.0 if b_hat > 0 else -1.0

            def bound_dist(m_val: float) -> float:
                if method == "relative_magnitude":
                    half_w = m_val * pre_max
                else:
                    k = max(h - base_period, 1)
                    c_h = 0.5 * (k) * (k + 1)
                    half_w = c_h * m_val

                c_val = _imbens_manski_critical_value(half_w / s_hat, alpha)
                # Lower bound for positive estimate, upper bound for negative
                if sign > 0:
                    center = b_hat - (
                        (h - base_period) * pre_slope
                        if method == "smoothness" and pre_slope is not None
                        else 0.0
                    )
                    return center - half_w - c_val * s_hat
                else:
                    center = b_hat - (
                        (h - base_period) * pre_slope
                        if method == "smoothness" and pre_slope is not None
                        else 0.0
                    )
                    return center + half_w + c_val * s_hat

            try:
                # Test bracket
                if sign * bound_dist(100.0) >= 0:
                    m_star = np.inf
                else:
                    sol = scipy.optimize.brentq(bound_dist, 0.0, 100.0, xtol=1e-4)
                    m_star = float(sol)
            except Exception:
                m_star = np.inf

        breakdown_dict[h] = m_star

        # Compute grid of M
        for m in m_grid:
            if method == "relative_magnitude":
                half_w = m * pre_max
                center = b_hat
            else:
                k = max(h - base_period, 1)
                c_h = 0.5 * k * (k + 1)
                half_w = c_h * m
                center = b_hat - (k * pre_slope if pre_slope is not None else 0.0)

            id_lo = center - half_w
            id_hi = center + half_w

            c_val = _imbens_manski_critical_value(half_w / s_hat, alpha)
            ci_lo = id_lo - c_val * s_hat
            ci_hi = id_hi + c_val * s_hat

            sig = bool((ci_lo > 0.0) or (ci_hi < 0.0))

            rows.append({
                "M": float(m),
                "horizon": h,
                "orig_estimate": b_hat,
                "orig_se": s_hat,
                "id_lo": id_lo,
                "id_hi": id_hi,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "significant": sig,
            })

    table = pd.DataFrame(rows)
    bd_val = breakdown_dict[targets[0]] if len(targets) == 1 else breakdown_dict

    return HonestDiDResult(
        table=table,
        breakdown_value=bd_val,
        method="relative_magnitude" if method == "relative_magnitude" else "smoothness",
        ci=ci,
        pre_trend_max=pre_max,
        pre_trend_slope=pre_slope,
        target_horizons=targets,
    )
