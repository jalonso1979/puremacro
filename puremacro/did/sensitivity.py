"""Honest Difference-in-Differences sensitivity analysis (Rambachan & Roth 2023).

Sensitivity of post-treatment event-study estimates to violations of parallel
trends, following Rambachan and Roth (2023, *Review of Economic Studies*).
Let ``beta_hat ~ N(tau + delta, Sigma)`` be the event-study coefficients
(``tau_pre = 0``), let ``delta`` be the differential trend, and normalise the
reference period to ``delta_ref = 0``.  Event times are sorted chronologically,
the reference period is inserted if it is not part of the vector, and the
sorted window is treated as a sequence of consecutive periods (RR's ``t``).

Restriction sets (exactly RR's definitions, over the **full** event window):

* ``method='smoothness'`` -- bounded second differences::

      Delta^SD(M) = { delta : |delta_{t+1} - 2 delta_t + delta_{t-1}| <= M
                              for every consecutive triple (t-1, t, t+1) }

  Every triple counts, including the triples inside the pre-period and the
  triple that spans the reference period (which links ``delta_0`` to
  ``beta_{-2}`` through ``delta_{-1} = 0``).

* ``method='relative_magnitude'`` (default ``bound='first_difference'``)::

      Delta^RM(Mbar) = { delta : |delta_{t+1} - delta_t|
                                 <= Mbar * max_{s<0} |delta_{s+1} - delta_s|
                                 for every post-treatment t }

  The pre-treatment benchmark ``max_{s<0}|delta_{s+1} - delta_s|`` includes the
  first difference from the last pre-period to the reference period.
  ``bound='levels'`` is an explicitly named *alternative* (not RR's Delta^RM):
  ``|delta_t| <= Mbar * max_{s<0} |delta_s|`` for every post period ``t``.

Inference (what is implemented -- and what is not):

* ``smoothness``: the fixed-length confidence interval (FLCI) of Armstrong and
  Kolesar, which is RR's recommended procedure for Delta^SD.  An affine
  estimator ``a'beta_hat`` with post weights ``a_post = l`` (so it is unbiased
  for ``theta = l'tau_post`` under exact parallel trends) has worst-case bias
  ``M * ||lambda(a)||_1`` over Delta^SD(M), where ``lambda(a)`` solves
  ``A' lambda = a`` for the second-difference matrix ``A`` (finite only when
  ``a`` annihilates linear trends).  The CI is ``a'beta_hat +/- sigma_a *
  q_{1-alpha}(|N(bias/sigma_a, 1)|)`` with ``sigma_a = sqrt(a'Sigma a)``.  The
  weights are chosen on a precomputed bias/variance frontier (minimum variance
  for each admissible bias bound, solved as small quadratic programmes) to
  minimise the CI length; the frontier depends only on ``Sigma`` and ``l``, so
  the choice does not depend on ``beta_hat`` and coverage is uniform over
  Delta^SD(M).  The full ``Sigma`` (pre- and post-period rows and their
  covariances) enters through ``sigma_a``.

* ``relative_magnitude``: the plug-in identified set (closed form) plus the
  Imbens and Manski (2004) / Stoye (2009) confidence interval
  ``[theta_lo - c*se_lo, theta_hi + c*se_hi]`` where ``se_lo``/``se_hi`` are
  delta-method standard errors of the estimated *endpoints* computed from the
  full ``Sigma`` (the endpoints are functions of ``beta_hat_pre`` through the
  pre-treatment benchmark) and ``c`` solves
  ``Phi(c + (theta_hi - theta_lo)/max(se)) - Phi(-c) = 1 - alpha``.

* Rambachan and Roth's conditional and hybrid (Andrews-Roth-Pakes) confidence
  sets are **not** implemented.

Monte Carlo coverage of the nominal 95% sets is documented in
``tests/test_did_sensitivity.py`` (``test_coverage_monte_carlo_*``).

References
----------
Rambachan, A. and Roth, J. (2023). A More Credible Approach to Parallel
    Trends. Review of Economic Studies, 90(5), 2555-2591.
Armstrong, T.B. and Kolesar, M. (2018). Optimal Inference in a Class of
    Regression Models. Econometrica, 86(2), 655-683.
Imbens, G.W. and Manski, C.F. (2004). Confidence Intervals for Partially
    Identified Parameters. Econometrica, 72(6), 1845-1857.
Stoye, J. (2009). More on Confidence Intervals for Partially Identified
    Parameters. Econometrica, 77(4), 1299-1315.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.stats import norm

__all__ = ["honest_did", "honest_did_sensitivity", "HonestDiDResult"]

_RM_BOUND_ALIASES: dict[str, str] = {
    "first_difference": "first_difference",
    "first_differences": "first_difference",
    "first difference": "first_difference",
    "first differences": "first_difference",
    "differences": "first_difference",
    "deviation from pre-trend slope": "first_difference",
    "levels": "levels",
    "level": "levels",
    "deviation from parallel trends": "levels",
}

_RESTRICTION_TEXT: dict[str, str] = {
    "second_difference": "Δ^SD(M): |δ_{t+1} − 2δ_t + δ_{t−1}| ≤ M for every consecutive triple",
    "first_difference": "Δ^RM(M̄): |δ_{t+1} − δ_t| ≤ M̄ · max_{s<0}|δ_{s+1} − δ_s| for post-treatment t",
    "levels": "level bound (not RR's Δ^RM): |δ_t| ≤ M̄ · max_{s<0}|δ_s| for post-treatment t",
}

_CI_TEXT: dict[str, str] = {
    "flci": "FLCI (Armstrong–Kolesár fixed-length CI, worst-case bias over Δ^SD(M), full Σ)",
    "imbens_manski": "Imbens–Manski / Stoye CI with delta-method endpoint SEs from the full Σ",
}


def _fmt_h(h: Any) -> str:
    """Format a horizon label (int or the string ``'l_vec'``)."""
    if isinstance(h, (int, np.integer)):
        return f"{int(h):+d}"
    return str(h)


@dataclass(frozen=True)
class HonestDiDResult:
    """Result of :func:`honest_did` and :func:`honest_did_sensitivity`.

    Attributes
    ----------
    table : pd.DataFrame
        One row per (M, horizon) with columns
        ``[M, horizon, orig_estimate, orig_se, id_lo, id_hi, ci_lo, ci_hi, significant]``.
        ``id_lo``/``id_hi`` is the plug-in identified set (``delta_pre = beta_hat_pre``);
        under ``'smoothness'`` it is NaN when the pre-period second differences
        exceed ``M`` (the set is empty; the FLCI is still valid).
        ``ci_lo``/``ci_hi`` is the FLCI (``smoothness``) or the Imbens-Manski
        interval (``relative_magnitude``); ``significant`` is ``0 not in CI``.
    breakdown_value : float | dict[int, float]
        Smallest ``M`` (``Mbar``) at which the confidence interval includes zero,
        ``0.0`` when it already does at ``M = 0`` and ``inf`` when no crossing is
        found for ``M`` up to :attr:`m_search_max`.  A dict keyed by horizon when
        several target horizons are evaluated.
    method : str
        ``'smoothness'`` or ``'relative_magnitude'``.
    ci : float
        Confidence level, e.g. ``0.95``.
    pre_trend_max : float
        Pre-treatment benchmark of Delta^RM: ``max_{s<0}|delta_{s+1} - delta_s|``
        over the pre-period first differences (including the difference from
        the last pre-period to the reference period).  With ``bound='levels'``
        it is ``max_{s<0}|delta_s|`` instead.
    pre_trend_slope : float | None
        (``smoothness`` only) first difference of the pre-trend at the reference
        period, ``delta_ref - delta_{ref-1}`` -- the slope that Delta^SD(0)
        continues linearly into the post period.
    target_horizons : list
        Horizon labels evaluated (event times, or ``'l_vec'`` for a custom contrast).
    bound : str
        ``'second_difference'`` (smoothness), ``'first_difference'`` (RR's
        Delta^RM) or ``'levels'``.
    ci_method : str
        ``'flci'`` or ``'imbens_manski'``.
    pre_trend_max_second_diff : float
        ``max |delta_{t+1} - 2 delta_t + delta_{t-1}|`` over triples that lie
        entirely inside the pre-period/reference block (NaN with fewer than two
        pre-periods).  The plug-in Delta^SD(M) set is empty for ``M`` below it.
    m_search_max : float
        Largest ``M`` examined by the breakdown search.
    """

    table: pd.DataFrame
    breakdown_value: float | dict[int, float]
    method: str
    ci: float
    pre_trend_max: float
    pre_trend_slope: float | None
    target_horizons: list[Any]
    bound: str = "second_difference"
    ci_method: str = "flci"
    pre_trend_max_second_diff: float = float("nan")
    m_search_max: float = float("nan")

    def summary(self) -> str:
        """Formatted human-readable summary of the sensitivity analysis."""
        m_label = "M" if self.method == "smoothness" else "M̄"
        lines = [
            "Honest DiD Sensitivity Analysis (Rambachan & Roth 2023)",
            "=" * 72,
            f"Method                          : {self.method}",
            f"Restriction                     : {_RESTRICTION_TEXT.get(self.bound, self.bound)}",
            f"Confidence set                  : {_CI_TEXT.get(self.ci_method, self.ci_method)}",
            f"Confidence Level (1 - α)        : {self.ci * 100:.1f}%",
        ]
        if self.bound == "levels":
            lines.append(f"Max |δ̂_s|, s<0 (RM benchmark)   : {self.pre_trend_max:.4f}")
        else:
            lines.append(f"Max |Δδ̂_s|, s<0 (RM benchmark)  : {self.pre_trend_max:.4f}")
        if self.pre_trend_slope is not None:
            lines.append(
                f"Pre-trend slope at reference    : {self.pre_trend_slope:+.4f}"
            )
        if self.method == "smoothness" and np.isfinite(self.pre_trend_max_second_diff):
            lines.append(
                f"Max |Δ²δ̂| inside pre-period     : {self.pre_trend_max_second_diff:.4f}"
                "  (plug-in identified set empty below this M)"
            )

        lines.append("-" * 72)
        if isinstance(self.breakdown_value, dict):
            lines.append(f"Breakdown Values ({m_label}*) by Horizon:")
            for h, m_star in self.breakdown_value.items():
                m_str = (
                    f"{m_star:.4f}"
                    if np.isfinite(m_star)
                    else f"inf (none found for {m_label} ≤ {self.m_search_max:.4g})"
                )
                lines.append(f"  Horizon h = {_fmt_h(h)}                : {m_label}* = {m_str}")
        else:
            h = self.target_horizons[0] if self.target_horizons else 0
            m_str = (
                f"{self.breakdown_value:.4f}"
                if np.isfinite(self.breakdown_value)
                else f"inf (none found for {m_label} ≤ {self.m_search_max:.4g})"
            )
            lines.append(
                f"Breakdown Value ({m_label}*) for h = {_fmt_h(h)} : {m_label}* = {m_str}"
            )
            if self.breakdown_value == 0.0:
                lines.append(
                    f"  (Estimate is already not statistically distinguishable from 0 at {m_label} = 0)"
                )
            elif np.isfinite(self.breakdown_value):
                if self.method == "smoothness":
                    lines.append(
                        "  (Effect remains significant while second differences of the "
                        f"differential trend are below {self.breakdown_value:.4g})"
                    )
                else:
                    lines.append(
                        f"  (Effect remains significant until post-treatment violations reach "
                        f"{self.breakdown_value:.2f}x the pre-trend benchmark)"
                    )
            else:
                lines.append(
                    f"  (Effect remains statistically significant for all {m_label} searched)"
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

    def _subset(self, horizon: Any) -> pd.DataFrame:
        sub = self.table
        if horizon is not None:
            sub = sub[sub["horizon"] == horizon]
        elif len(self.target_horizons) > 1:
            sub = sub[sub["horizon"] == self.target_horizons[0]]
        return sub

    def plot_ascii(self, horizon: int | None = None, width: int = 50) -> str:
        """Render an ASCII chart of confidence intervals across M."""
        sub = self._subset(horizon)
        if len(sub) == 0:
            return "(No data to plot)"

        h_val = sub["horizon"].iloc[0]
        all_lo = float(sub["ci_lo"].min())
        all_hi = float(sub["ci_hi"].max())
        min_v = min(all_lo, 0.0)
        max_v = max(all_hi, 0.0)
        span = max(max_v - min_v, 1e-6)

        def pos(val: float) -> int:
            p = int(round((val - min_v) / span * (width - 1)))
            return max(0, min(width - 1, p))

        zero_p = pos(0.0)
        ci_label = "FLCI" if self.ci_method == "flci" else "IM CI"

        lines = [
            f"Honest DiD Confidence Intervals vs M (Horizon h = {_fmt_h(h_val)})",
            "-" * (width + 24),
            f"{'M':>6} | {'Identified Set':^18} | {ci_label:^18} | Chart",
            "-" * (width + 24),
        ]

        for _, row in sub.iterrows():
            m_val = float(row["M"])
            id_lo, id_hi = float(row["id_lo"]), float(row["id_hi"])
            c_lo, c_hi = float(row["ci_lo"]), float(row["ci_hi"])
            p_clo, p_chi = pos(c_lo), pos(c_hi)

            bar = [" "] * width
            bar[zero_p] = "|"
            for i in range(p_clo, p_chi + 1):
                if bar[i] != "|":
                    bar[i] = "-"
            if np.isfinite(id_lo) and np.isfinite(id_hi):
                for i in range(pos(id_lo), pos(id_hi) + 1):
                    bar[i] = "="
                id_str = f"[{id_lo:+.2f}, {id_hi:+.2f}]"
            else:
                id_str = "(empty)"
            bar[p_clo] = "["
            bar[p_chi] = "]"

            chart_str = "".join(bar)
            ci_str = f"[{c_lo:+.2f}, {c_hi:+.2f}]"
            lines.append(f"{m_val:6.2f} | {id_str:^18} | {ci_str:^18} | {chart_str}")

        lines.append("-" * (width + 24))
        lines.append(
            f"Legend: '=' plug-in identified set, '[-]' {self.ci*100:.0f}% {ci_label}, '|' zero line"
        )
        return "\n".join(lines)

    def plot(
        self,
        horizon: int | None = None,
        ax: plt.Axes | None = None,
        title: str | None = None,
        figsize: tuple[float, float] = (8.0, 5.0),
        return_fig: bool = False,
    ) -> plt.Axes | tuple[plt.Figure, plt.Axes]:
        """Visualize the plug-in identified set, the confidence band and the breakdown value.

        Parameters
        ----------
        horizon : int, optional
            Event study horizon to plot. Defaults to the first target horizon.
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw on. If None, a new figure and axes are created.
        title : str, optional
            Custom chart title.
        figsize : tuple of float, default (8.0, 5.0)
            Figure dimensions when ax is None.
        return_fig : bool, default False
            If True, returns (fig, ax). Otherwise returns ax.

        Returns
        -------
        matplotlib.axes.Axes or tuple of (Figure, Axes)
            The generated plot.
        """
        sub = self._subset(horizon)
        if len(sub) == 0:
            raise ValueError("No data found to plot for specified horizon.")

        sub = sub.sort_values("M")
        h_val = sub["horizon"].iloc[0]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig_any = ax.get_figure()
            assert isinstance(fig_any, plt.Figure)
            fig = fig_any

        m_vals = sub["M"].to_numpy(dtype=float)
        id_lo = sub["id_lo"].to_numpy(dtype=float)
        id_hi = sub["id_hi"].to_numpy(dtype=float)
        ci_lo = sub["ci_lo"].to_numpy(dtype=float)
        ci_hi = sub["ci_hi"].to_numpy(dtype=float)

        ci_pct = int(round(self.ci * 100))
        ci_label = (
            f"{ci_pct}% FLCI (Armstrong–Kolesár)"
            if self.ci_method == "flci"
            else f"{ci_pct}% Robust CI (Imbens–Manski)"
        )

        ax.fill_between(m_vals, ci_lo, ci_hi, color="#2b5c8f", alpha=0.18, label=ci_label)
        ax.plot(m_vals, ci_lo, color="#1a3b5c", linestyle="--", linewidth=1.2)
        ax.plot(m_vals, ci_hi, color="#1a3b5c", linestyle="--", linewidth=1.2)

        finite = np.isfinite(id_lo) & np.isfinite(id_hi)
        if finite.any():
            ax.fill_between(
                m_vals,
                np.where(finite, id_lo, np.nan),
                np.where(finite, id_hi, np.nan),
                color="#2b5c8f",
                alpha=0.38,
                label="Plug-in identified set",
            )
            ax.plot(m_vals, np.where(finite, id_lo, np.nan), color="#2b5c8f", linestyle="-", linewidth=1.5)
            ax.plot(m_vals, np.where(finite, id_hi, np.nan), color="#2b5c8f", linestyle="-", linewidth=1.5)

        orig_est = float(sub["orig_estimate"].iloc[0])
        ax.plot(
            0.0,
            orig_est,
            "o",
            color="#2b5c8f",
            markersize=6,
            label=f"Original Estimate ({orig_est:+.3f})",
        )

        ax.axhline(0.0, color="crimson", linestyle=":", linewidth=1.3, alpha=0.85, label="Zero Effect")

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
            y_span = max(float(np.nanmax(ci_hi) - np.nanmin(ci_lo)), 1e-4)
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

        if self.method == "smoothness":
            ax.set_xlabel("Smoothness Bound M", fontsize=11)
            method_desc = "Smoothness Δ^SD(M)"
        elif self.bound == "levels":
            ax.set_xlabel("Relative Magnitude Multiplier M̄ (level bound)", fontsize=11)
            method_desc = "Relative magnitude, level bound (M̄)"
        else:
            ax.set_xlabel("Relative Magnitude Multiplier M̄", fontsize=11)
            method_desc = "Relative Magnitude Δ^RM(M̄)"

        ax.set_ylabel("Treatment Effect Parameter θ", fontsize=11)

        if title is None:
            title = f"Honest DiD Sensitivity Analysis: {method_desc} (Horizon h = {_fmt_h(h_val)})"
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=9)

        if return_fig:
            return fig, ax
        return ax


# ---------------------------------------------------------------------------
# Critical values
# ---------------------------------------------------------------------------


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
        sol = scipy.optimize.brentq(f, z_one, z_two, xtol=1e-10)
        return float(sol)
    except Exception:
        return z_one


def _folded_normal_quantile(t: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorised 1-alpha quantile of |N(t, 1)|: q solving Φ(q − t) − Φ(−q − t) = 1 − α."""
    t_arr = np.abs(np.atleast_1d(np.asarray(t, dtype=float)))
    lo = t_arr + float(norm.ppf(1.0 - alpha))
    hi = t_arr + float(norm.ppf(1.0 - alpha / 2.0))
    target = 1.0 - alpha
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        f_mid = norm.cdf(mid - t_arr) - norm.cdf(-mid - t_arr) - target
        lo = np.where(f_mid < 0.0, mid, lo)
        hi = np.where(f_mid < 0.0, hi, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Delta^SD machinery
# ---------------------------------------------------------------------------


def _second_difference_matrix(n: int) -> np.ndarray:
    """(n-2) x n matrix of second differences over consecutive triples."""
    a = np.zeros((max(n - 2, 0), n))
    for i in range(n - 2):
        a[i, i] = 1.0
        a[i, i + 1] = -2.0
        a[i, i + 2] = 1.0
    return a


def _sd_plugin_bounds(
    delta_known: np.ndarray,
    l_vec: np.ndarray,
    m_val: float,
) -> tuple[float, float] | None:
    """Plug-in identified set of ``l'delta_post`` under Delta^SD(M).

    ``delta_known`` holds the pre-period coefficients and the reference period
    (0) in chronological order; the ``L = len(l_vec)`` post periods follow.
    Returns ``(min l'delta_post, max l'delta_post)`` or ``None`` when the set is
    empty (pre-period second differences exceed ``M``).
    """
    n_known = len(delta_known)
    L = len(l_vec)
    T = n_known + L
    A = _second_difference_matrix(T)
    A_known = A[:, :n_known]
    A_post = A[:, n_known:]
    const = A_known @ delta_known
    tol = 1e-9 * (1.0 + abs(m_val))

    pre_only = ~np.any(A_post != 0.0, axis=1)
    if np.any(np.abs(const[pre_only]) > m_val + tol):
        return None

    A_lp = A_post[~pre_only]
    c_lp = const[~pre_only]
    A_ub = np.vstack([A_lp, -A_lp])
    b_ub = np.concatenate([m_val - c_lp, m_val + c_lp])
    bounds = [(None, None)] * L

    res_max = scipy.optimize.linprog(-l_vec, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    res_min = scipy.optimize.linprog(l_vec, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res_max.status == 2 or res_min.status == 2:
        return None
    if not (res_max.success and res_min.success):
        raise RuntimeError(
            "linprog failed while computing the Delta^SD identified set: "
            f"{res_max.message} / {res_min.message}"
        )
    return float(res_min.fun), float(-res_max.fun)


class _SDFrontier:
    """Bias/variance frontier of affine estimators of ``theta = l'tau_post`` under Delta^SD.

    Non-base coefficients are ordered pre (chronological) then post.  ``tvec``
    holds the position of each coefficient relative to the reference period.
    """

    def __init__(
        self,
        sigma: np.ndarray,
        l_vec: np.ndarray,
        pre_idx: np.ndarray,
        post_idx: np.ndarray,
        tvec: np.ndarray,
        n_points: int = 48,
    ) -> None:
        n = sigma.shape[0]
        T = n + 1
        K = len(pre_idx)
        scale = float(np.max(np.diag(sigma)))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        self.sigma = sigma
        self._sig = sigma / scale
        self.m = T - 2

        # Non-base columns of the second-difference matrix on the full window.
        A = _second_difference_matrix(T)
        t_min = float(tvec.min())
        base_pos = int(round(-t_min)) if t_min < 0.0 else 0
        positions = tvec + base_pos
        cols = [int(round(p)) for p in positions]
        A_nb = A[:, cols]
        self._AT = A_nb.T  # n x (T-2), full column rank

        tpre = tvec[pre_idx]
        tpost = tvec[post_idx]
        rhs = -float(l_vec @ tpost)
        v0 = rhs * tpre / float(tpre @ tpre)
        w0 = np.zeros(n)
        w0[post_idx] = l_vec
        w0[pre_idx] = v0
        if K > 1:
            q_full, _ = np.linalg.qr(tpre.reshape(-1, 1), mode="complete")
            null_pre = q_full[:, 1:]
        else:
            null_pre = np.zeros((K, 0))
        N = np.zeros((n, null_pre.shape[1]))
        if null_pre.shape[1] > 0:
            N[pre_idx, :] = null_pre
        self.w0, self.N = w0, N
        self.nfree = N.shape[1]

        self.lam0 = np.linalg.lstsq(self._AT, w0, rcond=None)[0]
        self.Lam = (
            np.linalg.lstsq(self._AT, N, rcond=None)[0]
            if self.nfree > 0
            else np.zeros((self.m, 0))
        )
        if not np.allclose(self._AT @ self.lam0, w0, atol=1e-8):
            raise RuntimeError("internal error: affine weights are not orthogonal to linear trends")

        # Frontier: minimum variance for each admissible worst-case-bias bound.
        vs: list[np.ndarray] = []
        if self.nfree == 0:
            vs.append(np.zeros(0))
        else:
            r_min, v_min = self._min_bias()
            v_max = self._min_var()
            r_max = float(np.abs(self.lam0 + self.Lam @ v_max).sum())
            if r_max - r_min <= 1e-10 * max(1.0, r_max):
                vs.append(v_max)
            else:
                v_prev = v_min
                for r in np.linspace(r_min, r_max, n_points):
                    v_prev = self._qp(float(r), v_prev)
                    vs.append(v_prev)
                vs.append(v_max)
        W = np.array([w0 + (N @ v if self.nfree > 0 else 0.0) for v in vs])
        Lm = np.array([self.lam0 + (self.Lam @ v if self.nfree > 0 else 0.0) for v in vs])
        self.W = W
        self.B = np.abs(Lm).sum(axis=1)  # worst-case bias per unit of M
        var = np.einsum("ki,ij,kj->k", W, sigma, W)
        self.S = np.sqrt(np.maximum(var, 1e-300))

    def _min_bias(self) -> tuple[float, np.ndarray]:
        m, nf = self.m, self.nfree
        c = np.concatenate([np.zeros(nf), np.ones(2 * m)])
        A_eq = np.hstack([self.Lam, -np.eye(m), np.eye(m)])
        bounds = [(None, None)] * nf + [(0.0, None)] * (2 * m)
        res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=-self.lam0, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"linprog failed while computing the minimum-bias weights: {res.message}")
        return float(res.fun), np.asarray(res.x[:nf], dtype=float)

    def _min_var(self) -> np.ndarray:
        nsn = self.N.T @ self._sig @ self.N
        rhs = self.N.T @ self._sig @ self.w0
        return -np.linalg.lstsq(nsn, rhs, rcond=None)[0]

    def _qp(self, r: float, v_start: np.ndarray) -> np.ndarray:
        """Minimum variance subject to ``||lam0 + Lam v||_1 <= r`` (SLSQP on a smooth reformulation)."""
        m, nf = self.m, self.nfree
        nsn = self.N.T @ self._sig @ self.N
        nsw = self.N.T @ self._sig @ self.w0
        wsw = float(self.w0 @ self._sig @ self.w0)
        A_eq = np.hstack([self.Lam, -np.eye(m), np.eye(m)])
        lam0 = self.lam0

        def f(x: np.ndarray) -> float:
            v = x[:nf]
            return float(v @ nsn @ v + 2.0 * nsw @ v + wsw)

        def grad(x: np.ndarray) -> np.ndarray:
            v = x[:nf]
            return np.concatenate([2.0 * nsn @ v + 2.0 * nsw, np.zeros(2 * m)])

        lam_s = lam0 + self.Lam @ v_start
        x0 = np.concatenate([v_start, np.maximum(lam_s, 0.0), np.maximum(-lam_s, 0.0)])
        ineq_jac = np.concatenate([np.zeros(nf), -np.ones(2 * m)])
        cons = [
            {"type": "eq", "fun": lambda x: A_eq @ x + lam0, "jac": lambda x: A_eq},
            {"type": "ineq", "fun": lambda x: r - float(x[nf:].sum()), "jac": lambda x: ineq_jac},
        ]
        bounds = [(None, None)] * nf + [(0.0, None)] * (2 * m)
        res = scipy.optimize.minimize(
            f,
            x0,
            jac=grad,
            constraints=cons,
            bounds=bounds,
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 500},
        )
        v_new = np.asarray(res.x[:nf], dtype=float)
        feasible = float(np.abs(lam0 + self.Lam @ v_new).sum()) <= r * (1.0 + 1e-6) + 1e-9
        if feasible and (res.success or f(res.x) <= f(x0)):
            return v_new
        return np.asarray(v_start, dtype=float)

    def flci(self, beta: np.ndarray, m_val: float, alpha: float) -> tuple[float, float]:
        """Shortest FLCI in the family for Delta^SD(m_val) at level 1 - alpha."""
        half = self.S * _folded_normal_quantile(m_val * self.B / self.S, alpha)
        k = int(np.argmin(half))
        center = float(self.W[k] @ beta)
        return center - float(half[k]), center + float(half[k])


# ---------------------------------------------------------------------------
# Delta^RM machinery (closed form + delta method)
# ---------------------------------------------------------------------------


def _rm_bounds(
    l_vec: np.ndarray,
    beta_nb: np.ndarray,
    sigma_nb: np.ndarray,
    delta_known: np.ndarray,
    known_map: np.ndarray,
    post_idx: np.ndarray,
    m_bar: float,
    bound: str,
) -> tuple[float, float, float, float]:
    """Plug-in identified set and delta-method endpoint SEs under the relative-magnitude bound.

    Returns ``(id_lo, id_hi, se_lo, se_hi)`` for ``theta = l'tau_post``.
    ``known_map[i]`` is the index into ``beta_nb`` of known position ``i`` (-1 for
    the reference period).
    """
    n = len(beta_nb)
    K = len(delta_known) - 1
    theta = float(l_vec @ beta_nb[post_idx])
    g_theta = np.zeros(n)
    g_theta[post_idx] = l_vec

    grad_anchor = np.zeros(n)
    if bound == "first_difference":
        diffs = np.diff(delta_known)
        s = int(np.argmax(np.abs(diffs)))
        bench = float(abs(diffs[s]))
        sgn = float(np.sign(diffs[s]))
        grad_bench = np.zeros(n)
        if known_map[s + 1] >= 0:
            grad_bench[known_map[s + 1]] += sgn
        if known_map[s] >= 0:
            grad_bench[known_map[s]] -= sgn
        tails = np.cumsum(l_vec[::-1])[::-1]
        c_l = float(np.abs(tails).sum())
        anchor = float(l_vec.sum() * delta_known[K])
        if known_map[K] >= 0:
            grad_anchor[known_map[K]] = float(l_vec.sum())
    else:
        levels = np.where(known_map >= 0, np.abs(delta_known), -np.inf)
        s = int(np.argmax(levels))
        bench = float(abs(delta_known[s]))
        sgn = float(np.sign(delta_known[s]))
        grad_bench = np.zeros(n)
        grad_bench[known_map[s]] = sgn
        c_l = float(np.abs(l_vec).sum())
        anchor = 0.0

    half = m_bar * bench * c_l
    grad_half = m_bar * c_l * grad_bench
    center = theta - anchor
    g_lo = g_theta - grad_anchor - grad_half
    g_hi = g_theta - grad_anchor + grad_half
    se_lo = float(np.sqrt(max(float(g_lo @ sigma_nb @ g_lo), 0.0)))
    se_hi = float(np.sqrt(max(float(g_hi @ sigma_nb @ g_hi), 0.0)))
    return center - half, center + half, se_lo, se_hi


def _imbens_manski_interval(
    id_lo: float, id_hi: float, se_lo: float, se_hi: float, alpha: float
) -> tuple[float, float]:
    se_max = max(se_lo, se_hi, 1e-12)
    c = _imbens_manski_critical_value((id_hi - id_lo) / se_max, alpha, is_half_width=False)
    return id_lo - c * se_lo, id_hi + c * se_hi


# ---------------------------------------------------------------------------
# Breakdown value
# ---------------------------------------------------------------------------


def _find_breakdown(
    ci_at: Callable[[float], tuple[float, float]], scale: float
) -> tuple[float, float]:
    """Smallest M at which the CI includes zero; returns ``(m_star, largest M examined)``."""
    lo0, hi0 = ci_at(0.0)
    if lo0 <= 0.0 <= hi0:
        return 0.0, 0.0
    side = 1.0 if lo0 > 0.0 else -1.0

    def g(m: float) -> float:
        lo, hi = ci_at(m)
        return lo if side > 0.0 else -hi

    m_prev = 0.0
    for k in range(-6, 21):
        m_try = scale * 2.0**k
        g_try = g(m_try)
        if g_try <= 0.0:
            if g_try == 0.0:
                return m_try, m_try
            root = scipy.optimize.brentq(g, m_prev, m_try, xtol=1e-10 * scale, rtol=1e-10)
            return float(root), m_try
        m_prev = m_try
    return float(np.inf), m_prev


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _normalise_method(method: str) -> str:
    m_clean = str(method).lower().strip()
    if m_clean in ("smoothness", "deltasd", "delta_sd", "sd"):
        return "smoothness"
    if m_clean in ("relative_magnitude", "deltarm", "delta_rm", "rm"):
        return "relative_magnitude"
    raise ValueError(f"method must be 'smoothness' or 'relative_magnitude', got {method!r}")


def _normalise_bound(bound: str | None, method: str) -> str:
    if method == "smoothness":
        if bound is not None and str(bound).lower().strip() not in ("second_difference", "second differences", "sd"):
            raise ValueError(
                "bound is only meaningful for method='relative_magnitude' "
                "('first_difference' or 'levels'); Delta^SD always bounds second differences"
            )
        return "second_difference"
    if bound is None:
        return "first_difference"
    key = str(bound).lower().strip()
    if key not in _RM_BOUND_ALIASES:
        raise ValueError(
            f"bound must be 'first_difference' (RR's Delta^RM) or 'levels', got {bound!r}"
        )
    return _RM_BOUND_ALIASES[key]


def _extract_frame(src: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = src.att_event_study if hasattr(src, "att_event_study") else src
    time_col = next((c for c in ["event_time", "rel_time", "time", "h", "year", "period"] if c in df.columns), None)
    if time_col is None:
        raise ValueError("could not find event time column in result")
    beta_col = next((c for c in ["att", "beta", "coef", "estimate"] if c in df.columns), None)
    if beta_col is None:
        raise ValueError("could not find coefficient column in result")
    se_col = next((c for c in ["se", "std_err", "stderr"] if c in df.columns), None)
    if se_col is None:
        raise ValueError("could not find standard error column in result")
    return (
        np.asarray(df[time_col], dtype=float),
        np.asarray(df[beta_col], dtype=float),
        np.asarray(df[se_col], dtype=float),
    )


def _event_times_from_counts(
    b_arr: np.ndarray,
    pre_periods: int | Sequence[int | float],
    post_periods: int | Sequence[int | float],
    base_period: float,
) -> np.ndarray:
    if isinstance(pre_periods, (int, np.integer)) and isinstance(post_periods, (int, np.integer)):
        num_pre, num_post = int(pre_periods), int(post_periods)
        if base_period != -1:
            raise ValueError(
                "integer pre_periods/post_periods place the reference period at -1 "
                "(pre = -pre_periods-1..-2, post = 0..post_periods-1); pass event_time "
                "explicitly for another base_period"
            )
        pre_idx = np.arange(-num_pre - 1, -1, dtype=float)
        post_idx = np.arange(0, num_post, dtype=float)
        if len(b_arr) == num_pre + num_post:
            return np.concatenate([pre_idx, post_idx])
        if len(b_arr) == num_pre + 1 + num_post:
            return np.concatenate([pre_idx, [-1.0], post_idx])
        raise ValueError(
            f"length of b_hat ({len(b_arr)}) matches neither pre_periods + post_periods "
            f"({num_pre + num_post}) nor pre_periods + 1 + post_periods ({num_pre + 1 + num_post})"
        )
    if isinstance(pre_periods, (int, np.integer)) or isinstance(post_periods, (int, np.integer)):
        raise ValueError("pre_periods and post_periods must both be integer counts or both be sequences")
    et = np.array(list(pre_periods) + list(post_periods), dtype=float)
    if len(et) != len(b_arr):
        raise ValueError(
            f"pre_periods + post_periods list {len(et)} event times but b_hat has {len(b_arr)} entries"
        )
    return et


@dataclass(frozen=True)
class _EventWindow:
    """Chronologically sorted event window with the reference period normalised to 0."""

    beta_nb: np.ndarray  # K pre coefficients (chronological) followed by L post coefficients
    sigma_nb: np.ndarray  # matching (K+L, K+L) covariance
    tvec: np.ndarray  # position of each non-base coefficient relative to the reference period
    pre_idx: np.ndarray
    post_idx: np.ndarray
    post_times: np.ndarray
    delta_known: np.ndarray  # pre coefficients and the reference period (0), chronological
    known_map: np.ndarray  # index into beta_nb for each known position, -1 for the reference


def _build_window(
    et_arr: np.ndarray,
    b_arr: np.ndarray,
    sigma_full: np.ndarray,
    base_period: float,
) -> _EventWindow:
    if base_period >= 0:
        is_base = et_arr == base_period
        is_pre = et_arr < base_period
        is_post = et_arr > base_period
    else:
        is_base = et_arr == base_period
        is_pre = (et_arr < 0) & ~is_base
        is_post = et_arr >= 0

    if not np.any(is_pre):
        raise ValueError(
            "Sensitivity analysis requires at least one pre-treatment period "
            f"(event_time < 0, event_time != {base_period}) to evaluate pre-trend violations."
        )
    if not np.any(is_post):
        raise ValueError("Sensitivity analysis requires at least one post-treatment period.")
    if int(is_base.sum()) > 1:
        raise ValueError(f"the reference period {base_period} appears more than once in event_time")
    if is_base.any():
        b_base = float(b_arr[is_base][0])
        if abs(b_base) > 1e-12:
            warnings.warn(
                f"coefficient at the reference period {base_period} is {b_base:.4g}; "
                "it is normalised to 0 (the reference period is not part of the analysis)",
                UserWarning,
                stacklevel=3,
            )

    pre_pos = np.where(is_pre)[0]
    post_pos = np.where(is_post)[0]
    pre_pos = pre_pos[np.argsort(et_arr[pre_pos], kind="stable")]
    post_pos = post_pos[np.argsort(et_arr[post_pos], kind="stable")]
    if len(np.unique(et_arr)) != len(et_arr):
        raise ValueError("event_time contains duplicate periods")
    order = np.concatenate([pre_pos, post_pos])
    beta_nb = b_arr[order]
    sigma_nb = sigma_full[np.ix_(order, order)]
    K, L = len(pre_pos), len(post_pos)
    pre_times = et_arr[pre_pos]
    post_times = et_arr[post_pos]

    # Known block: pre periods and the reference period in chronological order.
    known_times = np.concatenate([pre_times, [base_period]])
    known_vals = np.concatenate([b_arr[pre_pos], [0.0]])
    known_idx = np.concatenate([np.arange(K), [-1]])
    k_order = np.argsort(known_times, kind="stable")
    delta_known = known_vals[k_order]
    known_map = known_idx[k_order]
    base_pos = int(np.where(known_map == -1)[0][0])

    positions = np.arange(K + 1 + L, dtype=float)
    tvec = np.empty(K + L)
    for pos_i, idx in enumerate(known_map):
        if idx >= 0:
            tvec[idx] = positions[pos_i] - base_pos
    tvec[K:] = positions[K + 1 :] - base_pos

    full_times = np.concatenate([known_times[k_order], post_times])
    gaps = np.diff(full_times)
    if len(gaps) > 1 and not np.allclose(gaps, gaps[0]):
        warnings.warn(
            "event times are not equally spaced; the restriction sets treat the sorted "
            "event window (including the reference period) as consecutive periods",
            UserWarning,
            stacklevel=3,
        )

    return _EventWindow(
        beta_nb=beta_nb,
        sigma_nb=sigma_nb,
        tvec=tvec,
        pre_idx=np.arange(K),
        post_idx=np.arange(K, K + L),
        post_times=post_times,
        delta_known=delta_known,
        known_map=known_map,
    )


def _horizon_label(h: float) -> Any:
    return int(h) if float(h).is_integer() else float(h)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    bound: str | None = None,
) -> HonestDiDResult:
    """Sensitivity of an event-study estimate to parallel-trends violations (Rambachan & Roth 2023).

    Computes, for every ``M`` in the grid, the plug-in identified set of
    ``theta = l'tau_post`` under the restriction set, a confidence set that
    accounts for sampling uncertainty in **all** event-study coefficients, and
    the breakdown value ``M*``.  See the module docstring for the exact
    restriction sets and inference procedures (FLCI for ``'smoothness'``,
    Imbens-Manski with delta-method endpoint SEs for ``'relative_magnitude'``;
    RR's conditional/hybrid procedures are not implemented).

    Parameters
    ----------
    b_hat : array-like or result object, optional
        Event-study coefficients, or a result object from
        :func:`puremacro.did.callaway_santanna` / :func:`puremacro.did.sun_abraham`
        (anything with an ``att_event_study`` frame) or a :class:`pandas.DataFrame`
        with event-time, coefficient and standard-error columns.
        For an array, the event times must be given through ``event_time`` or
        ``pre_periods``/``post_periods``: the vector is never split by guessing.
    sigma : array-like, optional
        Covariance matrix of the coefficients: shape ``(n, n)`` matching ``b_hat``
        (a row/column for the reference period, if present, is dropped) or
        ``(L, L)`` for the post-treatment block only, in which case ``se`` must
        also be given (pre-period variances are needed).  Any other shape raises.
    se : array-like, optional
        Standard errors (``sigma = diag(se**2)`` when ``sigma`` is not given, i.e.
        the coefficients are treated as uncorrelated).
    method : {'smoothness', 'relative_magnitude'}, default 'smoothness'
        ``'smoothness'`` -> Delta^SD(M); ``'relative_magnitude'`` -> Delta^RM(Mbar).
        Aliases ``'sd'``/``'rm'`` are accepted.
    m_vec : sequence of float, optional
        Non-negative grid of ``M`` (``Mbar``) values.  Defaults to
        ``(0, 0.05, 0.1, 0.15, 0.2, 0.3)`` for smoothness and
        ``(0, 0.25, 0.5, 0.75, 1, 1.5, 2)`` for relative magnitudes.
    base_period : int, default -1
        Reference period (``delta = 0``).  Negative values follow the relative-time
        convention (post = ``event_time >= 0``); non-negative values are calendar
        conventions (post = ``event_time > base_period``).  A coefficient supplied
        at the reference period is normalised to 0 (with a warning if non-zero);
        pre-treatment periods that lie after the reference period (e.g.
        ``base_period=-2`` with ``-1`` present) are handled chronologically.
    alpha : float, default 0.05
        Significance level; the confidence level is ``1 - alpha``.
    l_vec : array-like, optional
        Contrast over the post-treatment coefficients, ``theta = l'tau_post``.
        Cannot be combined with ``target_horizon``; rows are labelled ``'l_vec'``.
    pre_periods, post_periods : int or sequence, optional
        Either integer counts (reference period at -1: pre = ``-pre_periods-1..-2``,
        post = ``0..post_periods-1``; ``b_hat`` may or may not include the
        reference coefficient) or explicit lists of event times.
    result : optional
        Alias for ``b_hat`` when passing an estimation result object.
    event_time : sequence, optional
        Event times matching ``b_hat`` (any order; sorted internally).
    beta : sequence, optional
        Alias for ``b_hat``.
    target_horizon : int or sequence of int, optional
        Post-treatment event time(s) to evaluate.  Defaults to the first post period.
    m_grid : sequence of float, optional
        Alias for ``m_vec`` (``m_vec`` wins if both are given).
    ci : float, optional
        Confidence level; sets ``alpha = 1 - ci``.  Passing a non-default
        ``alpha`` together with an inconsistent ``ci`` raises.
    bound : {'first_difference', 'levels'}, optional
        Relative-magnitude benchmark.  ``'first_difference'`` (default) is RR's
        Delta^RM; ``'levels'`` bounds post-period *levels* by ``Mbar * max|delta_pre|``.
        Only valid with ``method='relative_magnitude'``.

    Returns
    -------
    HonestDiDResult

    Raises
    ------
    ValueError
        For unknown methods/bounds, negative ``M``, inconsistent ``alpha``/``ci``,
        missing event times, mismatched lengths or covariance shapes.
    TypeError
        For unknown keyword arguments.

    Warns
    -----
    UserWarning
        When the plug-in Delta^SD(M) identified set is empty for some ``M`` (the
        pre-period second differences exceed ``M``; ``id_lo``/``id_hi`` are NaN),
        when a non-zero coefficient is supplied at the reference period, or when
        the event times are not equally spaced.
    """
    method = _normalise_method(method)
    bound_norm = _normalise_bound(bound, method)

    # Grid normalisation
    if m_vec is None:
        if m_grid is not None:
            m_vec = m_grid
        elif method == "smoothness":
            m_vec = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)
        else:
            m_vec = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    m_grid_vals = [float(m) for m in m_vec]
    if len(m_grid_vals) == 0:
        raise ValueError("m_vec must contain at least one value")
    if any((not np.isfinite(m)) or m < 0.0 for m in m_grid_vals):
        raise ValueError(f"m_vec must contain finite non-negative values, got {list(m_vec)!r}")

    # CI / alpha normalisation
    if ci is not None:
        if not 0.0 < float(ci) < 1.0:
            raise ValueError(f"ci must lie in (0, 1), got {ci!r}")
        if alpha != 0.05 and abs(float(ci) - (1.0 - float(alpha))) > 1e-9:
            raise ValueError(
                f"alpha={alpha!r} and ci={ci!r} are inconsistent; pass one of them"
            )
        alpha = float(1.0 - float(ci))
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    alpha = float(alpha)
    ci_level = float(1.0 - alpha)

    # Source extraction
    src = b_hat if b_hat is not None else (result if result is not None else beta)
    if src is None:
        raise ValueError(
            "must provide either 'result' (or 'b_hat') or all of ('event_time', 'beta', 'se')"
        )

    sigma_mat: np.ndarray | None = None
    if sigma is not None:
        sigma_mat = np.asarray(sigma, dtype=float)
        if sigma_mat.ndim != 2 or sigma_mat.shape[0] != sigma_mat.shape[1]:
            raise ValueError(f"sigma must be a square matrix, got shape {sigma_mat.shape}")

    se_arr: np.ndarray | None = None
    if hasattr(src, "att_event_study") or isinstance(src, pd.DataFrame):
        et_arr, b_arr, se_arr = _extract_frame(src)
        if event_time is not None and len(event_time) != len(b_arr):
            raise ValueError("event_time, beta, and se must have equal lengths")
    else:
        b_arr = np.asarray(src, dtype=float).ravel()
        if event_time is not None:
            et_arr = np.asarray(event_time, dtype=float).ravel()
        elif pre_periods is not None and post_periods is not None:
            et_arr = _event_times_from_counts(b_arr, pre_periods, post_periods, base_period)
        else:
            raise ValueError(
                "event times are required: pass event_time=[...] (or pre_periods/post_periods) "
                "alongside b_hat; the coefficient vector is never split into pre/post by guessing"
            )
        if se is not None:
            se_arr = np.asarray(se, dtype=float).ravel()
        elif sigma_mat is not None and sigma_mat.shape == (len(b_arr), len(b_arr)):
            se_arr = np.sqrt(np.clip(np.diag(sigma_mat), 0.0, None))
        elif sigma_mat is not None:
            raise ValueError(
                f"sigma has shape {sigma_mat.shape} but b_hat has {len(b_arr)} entries; "
                "pass a full (n, n) covariance, or an (L, L) post-treatment block together with se"
            )
        else:
            raise ValueError("must provide either 'se' or 'sigma'")

    if len(et_arr) != len(b_arr) or len(b_arr) != len(se_arr):
        raise ValueError("event_time, beta, and se must have equal lengths")
    if len(et_arr) == 0:
        raise ValueError("empty event study arrays")
    if not np.all(np.isfinite(b_arr)) or not np.all(np.isfinite(et_arr)):
        raise ValueError("b_hat and event_time must be finite")

    n = len(b_arr)
    if base_period >= 0:
        is_post_raw = et_arr > base_period
    else:
        is_post_raw = et_arr >= 0
    L_raw = int(is_post_raw.sum())

    # Full covariance (n, n) in the caller's order
    sigma_full = np.diag(se_arr**2)
    if sigma_mat is not None:
        if sigma_mat.shape == (n, n):
            sigma_full = sigma_mat.copy()
        elif sigma_mat.shape == (L_raw, L_raw) and L_raw != n:
            if se is None:
                raise ValueError(
                    f"sigma of shape {sigma_mat.shape} covers only the post-treatment coefficients; "
                    "pass se as well so that pre-period variances are known"
                )
            post_pos_raw = np.where(is_post_raw)[0]
            sigma_full[np.ix_(post_pos_raw, post_pos_raw)] = sigma_mat
        else:
            raise ValueError(
                f"sigma has shape {sigma_mat.shape}; expected ({n}, {n}) matching b_hat "
                f"or ({L_raw}, {L_raw}) for the post-treatment block"
            )
        sigma_full = 0.5 * (sigma_full + sigma_full.T)
        if np.all(np.isfinite(sigma_full)):
            eig_min = float(np.linalg.eigvalsh(sigma_full).min())
            if eig_min < -1e-8 * max(float(np.abs(np.diag(sigma_full)).max()), 1e-300):
                warnings.warn(
                    f"sigma is not positive semi-definite (min eigenvalue {eig_min:.3g}); "
                    "standard errors may be unreliable",
                    UserWarning,
                    stacklevel=2,
                )

    win = _build_window(et_arr, b_arr, sigma_full, float(base_period))
    if not np.all(np.isfinite(win.sigma_nb)):
        raise ValueError("se / sigma must be finite for every non-reference period")
    K = len(win.pre_idx)
    L = len(win.post_idx)
    post_betas = win.beta_nb[win.post_idx]
    sigma_post = win.sigma_nb[np.ix_(win.post_idx, win.post_idx)]

    # Pre-trend descriptives
    known_diffs = np.diff(win.delta_known)
    pre_max_first_diff = float(np.max(np.abs(known_diffs)))
    pre_levels = np.abs(win.delta_known[win.known_map >= 0])
    pre_max_level = float(np.max(pre_levels))
    pre_trend_max = pre_max_level if bound_norm == "levels" else pre_max_first_diff
    pre_slope: float | None = None
    if method == "smoothness":
        pre_slope = float(win.delta_known[-1] - win.delta_known[-2])
    if K >= 2:
        pre_second = np.diff(win.delta_known, n=2)
        pre_max_second_diff = float(np.max(np.abs(pre_second)))
    else:
        pre_max_second_diff = float("nan")

    # Targets
    eval_specs: list[tuple[Any, np.ndarray]] = []
    if l_vec is not None:
        if target_horizon is not None:
            raise ValueError("pass either l_vec or target_horizon, not both")
        l_arr = np.asarray(l_vec, dtype=float).ravel()
        if len(l_arr) != L:
            raise ValueError(f"l_vec must have length {L} (number of post-treatment periods)")
        eval_specs.append(("l_vec", l_arr))
    else:
        if target_horizon is None:
            targets = [float(win.post_times[0])]
        elif isinstance(target_horizon, (int, float, np.integer, np.floating)):
            targets = [float(target_horizon)]
        else:
            targets = [float(h) for h in target_horizon]
        for h in targets:
            matches = np.where(win.post_times == h)[0]
            if len(matches) == 0:
                raise ValueError(f"target horizon {_horizon_label(h)} not found in post-treatment periods")
            basis = np.zeros(L, dtype=float)
            basis[matches[0]] = 1.0
            eval_specs.append((_horizon_label(h), basis))

    rows: list[dict[str, Any]] = []
    breakdown_dict: dict[Any, float] = {}
    m_search_max = 0.0
    empty_sets: dict[Any, list[float]] = {}

    for h_label, l_spec in eval_specs:
        orig_theta = float(l_spec @ post_betas)
        s_hat = float(np.sqrt(max(float(l_spec @ sigma_post @ l_spec), 0.0)))

        ci_at: Callable[[float], tuple[float, float]]
        if method == "smoothness":
            frontier = _SDFrontier(win.sigma_nb, l_spec, win.pre_idx, win.post_idx, win.tvec)

            def ci_at_sd(m: float, _fr: _SDFrontier = frontier) -> tuple[float, float]:
                return _fr.flci(win.beta_nb, m, alpha)

            ci_at = ci_at_sd
            scale = max(abs(orig_theta), s_hat, 1e-12)
        else:

            def ci_at_rm(m: float, _l: np.ndarray = l_spec) -> tuple[float, float]:
                id_lo, id_hi, se_lo, se_hi = _rm_bounds(
                    _l, win.beta_nb, win.sigma_nb, win.delta_known, win.known_map,
                    win.post_idx, m, bound_norm,
                )
                return _imbens_manski_interval(id_lo, id_hi, se_lo, se_hi, alpha)

            ci_at = ci_at_rm
            scale = 1.0

        m_star, m_max = _find_breakdown(ci_at, scale)
        breakdown_dict[h_label] = m_star
        m_search_max = max(m_search_max, m_max)

        for m in m_grid_vals:
            if method == "smoothness":
                plug = _sd_plugin_bounds(win.delta_known, l_spec, m)
                if plug is None:
                    id_lo, id_hi = float("nan"), float("nan")
                    empty_sets.setdefault(h_label, []).append(m)
                else:
                    d_min, d_max = plug
                    id_lo, id_hi = orig_theta - d_max, orig_theta - d_min
            else:
                id_lo, id_hi, _, _ = _rm_bounds(
                    l_spec, win.beta_nb, win.sigma_nb, win.delta_known, win.known_map,
                    win.post_idx, m, bound_norm,
                )
            ci_lo, ci_hi = ci_at(m)
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

    if empty_sets:
        detail = "; ".join(
            f"h={_fmt_h(h)}: M in {sorted(set(ms))}" for h, ms in empty_sets.items()
        )
        warnings.warn(
            "plug-in Delta^SD(M) identified set is empty because the pre-period second "
            f"differences (max |Δ²β̂_pre| = {pre_max_second_diff:.4g}) exceed M ({detail}); "
            "id_lo/id_hi are NaN there, the FLCI remains valid",
            UserWarning,
            stacklevel=2,
        )

    table = pd.DataFrame(rows)
    bd_val: float | dict[int, float]
    if len(eval_specs) == 1:
        bd_val = breakdown_dict[eval_specs[0][0]]
    else:
        bd_val = dict(breakdown_dict)
    targets_out = [spec[0] for spec in eval_specs]

    return HonestDiDResult(
        table=table,
        breakdown_value=bd_val,
        method=method,
        ci=ci_level,
        pre_trend_max=pre_trend_max,
        pre_trend_slope=pre_slope,
        target_horizons=targets_out,
        bound=bound_norm,
        ci_method="flci" if method == "smoothness" else "imbens_manski",
        pre_trend_max_second_diff=pre_max_second_diff,
        m_search_max=m_search_max,
    )


def honest_did_sensitivity(
    result: Any = None,
    *,
    event_time: Sequence[int | float] | None = None,
    beta: Sequence[float] | None = None,
    se: Sequence[float] | None = None,
    target_horizon: int | Sequence[int] | None = None,
    method: str = "relative_magnitude",
    m_grid: Sequence[float] | None = None,
    ci: float | None = None,
    base_period: int = -1,
    sigma: np.ndarray | None = None,
    l_vec: Sequence[float] | np.ndarray | None = None,
    pre_periods: int | Sequence[int | float] | None = None,
    post_periods: int | Sequence[int | float] | None = None,
    m_vec: Sequence[float] | None = None,
    alpha: float | None = None,
    bound: str | None = None,
) -> HonestDiDResult:
    """Keyword-only alias for :func:`honest_did` (default ``method='relative_magnitude'``).

    ``alpha`` and ``ci`` are alternative ways of setting the confidence level
    (``ci = 1 - alpha``); both default to a 95% level and passing inconsistent
    values raises.
    """
    if alpha is not None and ci is not None and abs(float(ci) - (1.0 - float(alpha))) > 1e-9:
        raise ValueError(f"alpha={alpha!r} and ci={ci!r} are inconsistent; pass one of them")
    alpha_eff = float(alpha) if alpha is not None else (1.0 - float(ci) if ci is not None else 0.05)
    return honest_did(
        b_hat=result if result is not None else beta,
        sigma=sigma,
        se=se,
        method=method,
        m_vec=m_vec if m_vec is not None else m_grid,
        base_period=base_period,
        alpha=alpha_eff,
        l_vec=l_vec,
        pre_periods=pre_periods,
        post_periods=post_periods,
        event_time=event_time,
        target_horizon=target_horizon,
        bound=bound,
    )
