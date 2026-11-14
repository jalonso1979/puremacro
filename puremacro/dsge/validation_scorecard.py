"""DSGE prior-posterior diagnostics and empirical IRF validation scorecards.

Provides two validation workflows that bridge Bayesian/HANK DSGE output
with convergence diagnostics and empirical moment targets:

* :func:`prior_posterior_comparison` — computes truncation-aware prior moments
  (closed form when truncation omits ``< 1e-6`` probability mass, mode-centered
  1-D quadrature otherwise, and safe ``NaN`` / bounded-quadrature handling for
  raw inverse-gamma ``(s, nu)`` specs), posterior quantiles/mode, Vehtari et al.
  (2021) rank-normalized split-Rhat and bulk/tail ESS, shrinkage ratios, and
  weak-identification / boundary / convergence flags.
* :func:`score_irf_against_targets` — compares :class:`BayesianIRFResult`,
  :class:`HANKResult`, or horizon DataFrames against joint
  :class:`~puremacro.structural.bridge.MomentTargets` (from
  :func:`~puremacro.structural.lp.lp_moment_targets` or
  :meth:`MomentTargets.from_irf`) or marginal :class:`~puremacro.lp.LPResult`
  mappings using rank-aware covariance Wald statistics and sign/band checks.
"""
from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import integrate, stats

from puremacro._mcmc_rank import multichain_ess, rank_normalized_rhat
from puremacro.dsge.priors import (
    Prior,
    _beta_ab,
    _invgamma2_mean_std,
    _invgamma2_s_nu,
    _invgamma_mean_std,
    _invgamma_s_nu,
    _logpdf_beta,
    _logpdf_gamma,
    _logpdf_invgamma2_s_nu,
    _logpdf_invgamma_s_nu,
    _logpdf_normal,
    _weibull_mean_std,
    _weibull_shape_scale,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


_LABEL_COLON_RE = re.compile(r"^(.+?):([^:]+):h=(-?\d+)$")
_LABEL_ARROW_RE = re.compile(r"^(.+?)<-([^\[]+)\[h=(-?\d+)\]$")
_LABEL_VAR_H_RE = re.compile(r"^(.+?):h=(-?\d+)$")
_LABEL_BRACKET_H_RE = re.compile(r"^(.+?)\[h=(-?\d+)\]$")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PriorPosteriorComparisonResult:
    """Result container for :func:`prior_posterior_comparison`.

    Attributes
    ----------
    table : pd.DataFrame
        Per-parameter table of prior moments, posterior summaries, shrinkage
        statistics, rank-normalized MCMC diagnostics, and diagnostic flags.
    ci_level : float
        Central credible interval probability mass.
    n_chains : int
        Number of MCMC chains evaluated.
    n_draws : int
        Number of post-warmup draws per chain.
    model_name : str
        Optional model label carried over from the posterior object.
    warnings : tuple[str, ...]
        Non-fatal diagnostic or missing-metadata notices.
    digits : int
        Default decimal rounding precision for text/table exports.
    """

    table: pd.DataFrame
    ci_level: float = 0.90
    n_chains: int = 0
    n_draws: int = 0
    model_name: str = ""
    warnings: tuple[str, ...] = ()
    digits: int = 3

    def to_frame(self, layout: Literal["full", "compact"] = "full") -> pd.DataFrame:
        """Return prior-posterior comparison table as a pandas DataFrame."""
        if layout == "full":
            return self.table.copy()
        if layout == "compact":
            cols = [
                "param",
                "dist",
                "prior_mean",
                "prior_std",
                "post_mean",
                "post_std",
                "ci_lower",
                "ci_upper",
                "std_ratio",
                "rhat",
                "ess_bulk",
                "flags",
            ]
            keep = [c for c in cols if c in self.table.columns]
            return self.table.loc[:, keep].copy()
        raise ValueError(f"Unknown layout {layout!r}; expected 'full' or 'compact'.")

    def to_dataframe(self, layout: Literal["full", "compact"] = "full") -> pd.DataFrame:
        """Alias for :meth:`to_frame`."""
        return self.to_frame(layout=layout)

    def summary(self) -> str:
        """Return a concise text summary of prior-posterior diagnostics."""
        header = (
            f"Prior-Posterior Comparison "
            f"({len(self.table)} params, {self.n_chains} chains x {self.n_draws} draws, "
            f"CI={int(round(100 * self.ci_level))}%)"
        )
        if self.model_name:
            header = f"{header} [{self.model_name}]"
        lines = [header, self.to_frame("compact").to_string(index=False)]
        if not self.table.empty:
            n_weak = int(self.table["weakly_identified"].sum())
            n_bnd = int(self.table["near_boundary"].sum())
            n_unconv = int(self.table["unconverged_or_divergent"].sum())
            lines.append(
                f"Flags: weakly_identified={n_weak}, near_boundary={n_bnd}, "
                f"unconverged_or_divergent={n_unconv}"
            )
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)

    def to_markdown(
        self,
        layout: Literal["full", "compact"] = "compact",
        digits: int | None = None,
    ) -> str:
        """Render the comparison table as GitHub-Flavored Markdown."""
        from puremacro.reports import _df_to_markdown

        d = self.digits if digits is None else int(digits)
        return _df_to_markdown(self.to_frame(layout=layout), digits=d)

    def to_latex(
        self,
        layout: Literal["full", "compact"] = "compact",
        digits: int | None = None,
    ) -> str:
        """Render the comparison table as a LaTeX tabular environment."""
        from puremacro.reports import _df_to_latex

        d = self.digits if digits is None else int(digits)
        return _df_to_latex(self.to_frame(layout=layout), digits=d)

    def to_typst(
        self,
        layout: Literal["full", "compact"] = "compact",
        digits: int | None = None,
    ) -> str:
        """Render the comparison table as a Typst ``#table(...)`` block."""
        from puremacro.reports import _df_to_typst

        d = self.digits if digits is None else int(digits)
        return _df_to_typst(self.to_frame(layout=layout), digits=d)

    def plot(
        self,
        params: Sequence[str] | None = None,
        ax: Axes | None = None,
        *,
        title: str = "Prior vs Posterior (Standardized z-scale)",
    ) -> Figure | Axes:
        """Plot standardized prior intervals against posterior credible bands.

        Returns the created ``Figure`` when ``ax is None``, or ``ax`` when an
        existing Matplotlib ``Axes`` is provided.
        """
        import matplotlib.pyplot as plt

        df_sub = self.table if params is None else self.table[self.table["param"].isin(list(params))]
        if df_sub.empty:
            raise ValueError("No matching parameters available to plot.")

        if ax is None:
            fig_h = max(3.0, 0.45 * len(df_sub) + 1.4)
            fig, ax_use = plt.subplots(figsize=(7.5, fig_h))
        else:
            ax_use = ax

        z_crit = float(stats.norm.isf(0.5 * (1.0 - self.ci_level)))
        y_pos = np.arange(len(df_sub), dtype=float)[::-1]
        labels = df_sub["param"].astype(str).tolist()

        for y_val, (_, row) in zip(y_pos, df_sub.iterrows()):
            p_mean = float(row["prior_mean"])
            p_std = float(row["prior_std"])
            post_m = float(row["post_mean"])
            ci_lo = float(row["ci_lower"])
            ci_hi = float(row["ci_upper"])
            flagged = bool(row["unconverged_or_divergent"]) or bool(row["near_boundary"])

            if math.isfinite(p_mean) and math.isfinite(p_std) and p_std > 0.0:
                ax_use.hlines(
                    y_val + 0.10,
                    -z_crit,
                    z_crit,
                    color="#7f8c8d",
                    lw=3.0,
                    alpha=0.55,
                    label="Prior CI" if y_val == y_pos[0] else "_nolegend_",
                )
                ax_use.plot(
                    0.0,
                    y_val + 0.10,
                    marker="s",
                    color="#7f8c8d",
                    ms=5,
                    linestyle="None",
                )
                z_lo = (ci_lo - p_mean) / p_std
                z_hi = (ci_hi - p_mean) / p_std
                z_pt = (post_m - p_mean) / p_std
                color = "#c0392b" if flagged else "#1f77b4"
                ax_use.hlines(
                    y_val - 0.10,
                    z_lo,
                    z_hi,
                    color=color,
                    lw=2.5,
                    label="Posterior CI" if y_val == y_pos[0] else "_nolegend_",
                )
                ax_use.plot(
                    z_pt,
                    y_val - 0.10,
                    marker="o",
                    color=color,
                    ms=6,
                    linestyle="None",
                )
            else:
                ax_use.text(
                    0.0,
                    y_val,
                    "undefined prior std",
                    va="center",
                    ha="left",
                    fontsize=8,
                    color="#7f8c8d",
                )

        ax_use.axvline(0.0, color="#95a5a6", ls="--", lw=1.0)
        ax_use.set_yticks(y_pos)
        ax_use.set_yticklabels(labels)
        ax_use.set_xlabel("(value - prior_mean) / prior_std")
        ax_use.set_title(title)
        ax_use.grid(axis="x", alpha=0.25)

        if ax is None:
            fig.tight_layout()
            return fig
        return ax_use


@dataclass(frozen=True)
class IRFScorecardResult:
    """Result container for :func:`score_irf_against_targets`.

    Attributes
    ----------
    table : pd.DataFrame
        Matched horizon-by-horizon comparison table.
    metrics : pd.DataFrame
        Overall and per-variable discrepancy, Wald, sign, and coverage metrics.
    rmse : float
        Root-mean-squared IRF discrepancy across matched moments.
    mae : float
        Mean absolute IRF discrepancy across matched moments.
    wse : float
        Weighted squared error ``d' W d`` using covariance or diagonal weights.
    chi2_stat : float
        Rank-aware Wald chi-squared statistic (``NaN`` when covariance is absent).
    chi2_df : int
        Numerical rank / degrees of freedom used in the Wald test.
    chi2_pvalue : float
        Upper-tail chi-squared p-value computed via ``scipy.stats.chi2.sf``.
    sign_concordance : float
        Share of matched moments with matching sign.
    band_coverage : float
        Share of matched model IRFs falling inside ``[target_lower, target_upper]``.
    determinacy_rate : float
        Posterior determinacy rate carried over from ``BayesianIRFResult`` (or ``NaN``).
    weight_mode_used : str
        Weighting scheme actually applied (``'covariance'``, ``'diagonal'``, or
        ``'unweighted'``).
    covariance_condition_number : float
        Condition number across retained positive covariance eigenvalues.
    warnings : tuple[str, ...]
        Compatibility, unit/sign, conditioning, or dropped-moment notices.
    digits : int
        Default decimal rounding precision for table export methods.
    """

    table: pd.DataFrame
    metrics: pd.DataFrame
    rmse: float
    mae: float
    wse: float
    chi2_stat: float
    chi2_df: int
    chi2_pvalue: float
    sign_concordance: float
    band_coverage: float
    determinacy_rate: float = float("nan")
    weight_mode_used: str = "covariance"
    covariance_condition_number: float = float("nan")
    warnings: tuple[str, ...] = ()
    digits: int = 3

    def to_frame(self, layout: Literal["table", "metrics"] = "table") -> pd.DataFrame:
        """Return matched horizon table or aggregate metrics as a DataFrame."""
        if layout == "table":
            return self.table.copy()
        if layout == "metrics":
            return self.metrics.copy()
        raise ValueError(f"Unknown layout {layout!r}; expected 'table' or 'metrics'.")

    def to_dataframe(self, layout: Literal["table", "metrics"] = "table") -> pd.DataFrame:
        """Alias for :meth:`to_frame`."""
        return self.to_frame(layout=layout)

    def summary(self) -> str:
        """Return a concise text summary of the IRF validation scorecard."""
        chi2_str = (
            f"chi2({self.chi2_df})={self.chi2_stat:.3f}, p={self.chi2_pvalue:.4g}"
            if self.chi2_df > 0 and math.isfinite(self.chi2_stat) and math.isfinite(self.chi2_pvalue)
            else f"wse={self.wse:.3f}" if math.isfinite(self.wse) else "unweighted"
        )
        lines = [
            f"IRF Validation Scorecard ({len(self.table)} moments, weight_mode={self.weight_mode_used}, {chi2_str})",
            f"RMSE={self.rmse:.4f}, MAE={self.mae:.4f}, sign_concordance={self.sign_concordance:.3f}, band_coverage={self.band_coverage:.3f}",
        ]
        if math.isfinite(self.determinacy_rate):
            lines.append(f"Determinacy rate: {self.determinacy_rate:.3f}")
        lines.append(self.metrics.to_string(index=False))
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)

    def to_markdown(
        self,
        layout: Literal["table", "metrics"] = "table",
        digits: int | None = None,
    ) -> str:
        """Render scorecard table or metrics as GitHub-Flavored Markdown."""
        from puremacro.reports import _df_to_markdown

        d = self.digits if digits is None else int(digits)
        return _df_to_markdown(self.to_frame(layout=layout), digits=d)

    def to_latex(
        self,
        layout: Literal["table", "metrics"] = "table",
        digits: int | None = None,
    ) -> str:
        """Render scorecard table or metrics as a LaTeX tabular environment."""
        from puremacro.reports import _df_to_latex

        d = self.digits if digits is None else int(digits)
        return _df_to_latex(self.to_frame(layout=layout), digits=d)

    def to_typst(
        self,
        layout: Literal["table", "metrics"] = "table",
        digits: int | None = None,
    ) -> str:
        """Render scorecard table or metrics as a Typst ``#table(...)`` block."""
        from puremacro.reports import _df_to_typst

        d = self.digits if digits is None else int(digits)
        return _df_to_typst(self.to_frame(layout=layout), digits=d)

    def plot(
        self,
        variable: str | None = None,
        ax: Axes | None = None,
        *,
        title: str = "Model IRF vs Empirical Targets",
    ) -> Figure | Axes:
        """Plot model IRF alongside target estimates and uncertainty bands.

        Returns the created ``Figure`` when ``ax is None``, or ``ax`` when an
        existing Matplotlib ``Axes`` is provided.
        """
        import matplotlib.pyplot as plt

        vars_avail = list(dict.fromkeys(self.table["variable"].astype(str).tolist()))
        if variable is not None:
            if variable not in vars_avail:
                raise ValueError(f"Unknown variable {variable!r}; available {vars_avail}.")
            plot_vars = [variable]
        else:
            plot_vars = vars_avail

        if ax is not None:
            # Draw onto the caller's single Axes; focus on the requested or first variable.
            var_use = plot_vars[0]
            sub = self.table[self.table["variable"] == var_use].sort_values("h")
            _plot_scorecard_panel(ax, sub, var_use, title=title)
            return ax

        n_panels = len(plot_vars)
        fig, axes_obj = plt.subplots(1, n_panels, figsize=(5.8 * n_panels, 4.0), squeeze=False)
        axes_flat = axes_obj.ravel()
        for ax_i, v_name in zip(axes_flat, plot_vars):
            sub = self.table[self.table["variable"] == v_name].sort_values("h")
            _plot_scorecard_panel(ax_i, sub, v_name, title=f"{title}: {v_name}")
        fig.tight_layout()
        return fig


def _plot_scorecard_panel(ax: Any, sub: pd.DataFrame, var_name: str, *, title: str) -> None:
    h_arr = sub["h"].to_numpy(dtype=int)
    t_arr = sub["target"].to_numpy(dtype=float)
    m_arr = sub["model"].to_numpy(dtype=float)
    t_lo = sub["target_lower"].to_numpy(dtype=float)
    t_hi = sub["target_upper"].to_numpy(dtype=float)
    m_lo = sub["model_lower"].to_numpy(dtype=float)
    m_hi = sub["model_upper"].to_numpy(dtype=float)

    if np.any(np.isfinite(t_lo)) and np.any(np.isfinite(t_hi)):
        ax.fill_between(h_arr, t_lo, t_hi, color="#95a5a6", alpha=0.25, label="Target band")
    ax.plot(h_arr, t_arr, marker="o", ms=4, color="#2c3e50", lw=1.6, ls="--", label="Target")

    if np.any(np.isfinite(m_lo)) and np.any(np.isfinite(m_hi)):
        ax.fill_between(h_arr, m_lo, m_hi, color="#1f77b4", alpha=0.18, label="Model band")
    ax.plot(h_arr, m_arr, color="#1f77b4", lw=2.1, label="Model")

    ax.axhline(0.0, color="#7f8c8d", lw=0.9, ls=":")
    ax.set_xlabel("Horizon")
    ax.set_ylabel(var_name)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


# ---------------------------------------------------------------------------
# Truncation-aware prior moment computation
# ---------------------------------------------------------------------------

def _numerical_truncated_moments(
    logpdf_fn: Callable[[float], float],
    lo: float,
    hi: float,
    mode_hint: float,
) -> tuple[float, float, str]:
    """Compute (mean, std, status) on ``(lo, hi)`` via mode-centered 1-D quad."""
    if not (lo < hi):
        return float("nan"), float("nan"), "numerical_failed"

    if math.isfinite(mode_hint) and lo < mode_hint < hi:
        anchor = float(mode_hint)
    elif mode_hint <= lo and math.isfinite(lo):
        step = 1e-9 * max(1.0, abs(lo))
        anchor = float(min(lo + step, 0.5 * (lo + hi)) if math.isfinite(hi) else lo + step)
    elif mode_hint >= hi and math.isfinite(hi):
        step = 1e-9 * max(1.0, abs(hi))
        anchor = float(max(hi - step, 0.5 * (lo + hi)) if math.isfinite(lo) else hi - step)
    elif math.isfinite(lo) and math.isfinite(hi):
        anchor = 0.5 * (lo + hi)
    else:
        anchor = 0.0

    ref_log = float(logpdf_fn(anchor))
    if not math.isfinite(ref_log):
        if math.isfinite(lo) and math.isfinite(hi):
            grid = np.linspace(lo, hi, 33)[1:-1]
            vals = [(float(x), float(logpdf_fn(float(x)))) for x in grid]
            finite_vals = [pair for pair in vals if math.isfinite(pair[1])]
            if not finite_vals:
                return float("nan"), float("nan"), "numerical_failed"
            anchor, ref_log = max(finite_vals, key=lambda pair: pair[1])
        else:
            return float("nan"), float("nan"), "numerical_failed"

    def _weight(x: float) -> float:
        lp = float(logpdf_fn(float(x)))
        if not math.isfinite(lp):
            return 0.0
        diff = lp - ref_log
        if diff < -745.0:
            return 0.0
        return math.exp(min(diff, 700.0))

    pts = [anchor] if (lo < anchor < hi) else None
    shift_c = anchor if math.isfinite(anchor) else 0.0

    z, _ = integrate.quad(_weight, lo, hi, points=pts, epsabs=1e-11, epsrel=1e-9, limit=250)
    if not (math.isfinite(z) and z > 0.0):
        return float("nan"), float("nan"), "numerical_failed"

    m1, _ = integrate.quad(
        lambda x: (x - shift_c) * _weight(x) / z,
        lo,
        hi,
        points=pts,
        epsabs=1e-11,
        epsrel=1e-9,
        limit=250,
    )
    m2, _ = integrate.quad(
        lambda x: ((x - shift_c) ** 2) * _weight(x) / z,
        lo,
        hi,
        points=pts,
        epsabs=1e-11,
        epsrel=1e-9,
        limit=250,
    )
    if not (math.isfinite(m1) and math.isfinite(m2)):
        return float("nan"), float("nan"), "numerical_failed"
    mean_num = shift_c + m1
    var_num = max(0.0, m2 - m1 * m1)
    return float(mean_num), math.sqrt(var_num), "truncated_numerical"


def _require_mean_std(mean_raw: Any, std_raw: Any, name: str, dist: str) -> tuple[float, float]:
    if mean_raw is None or std_raw is None:
        raise ValueError(f"{dist} prior {name!r} requires 'mean' and 'std'.")
    return float(mean_raw), float(std_raw)


def _compute_prior_moments(
    spec: Prior | Mapping[str, Any],
    name: str,
) -> tuple[float, float, float, float, str, str]:
    """Return ``(lb, ub, prior_mean, prior_std, prior_status, dist)`` safely.

    Never instantiates :class:`InvGammaPrior` directly for raw inverse-gamma
    ``(s, nu)`` specs with ``nu <= 2`` (type 1) or ``nu <= 4`` (type 2), so
    bounded raw priors integrate cleanly while unbounded infinite-variance
    priors report ``NaN`` and ``prior_status='undefined_prior_variance'``.
    """
    if not isinstance(spec, (Prior, Mapping)) or "dist" not in spec:
        raise ValueError(f"Prior spec for {name!r} must be a Prior or mapping with 'dist'.")

    dist = str(spec["dist"]).strip().lower()
    lb = float(spec.get("lb", -math.inf))
    ub = float(spec.get("ub", math.inf))
    if not (lb < ub):
        raise ValueError(f"Prior {name!r} has empty support lb={lb!r} >= ub={ub!r}.")

    shift = float(spec.get("shift") or 0.0)
    mean_raw = spec.get("mean")
    std_raw = spec.get("std")
    trunc_tol = 1e-6

    if dist == "uniform":
        if math.isfinite(lb) and math.isfinite(ub):
            return lb, ub, 0.5 * (lb + ub), (ub - lb) / math.sqrt(12.0), "ok", dist
        if mean_raw is not None and std_raw is not None:
            mu = float(mean_raw)
            sd = float(std_raw)
            if math.isfinite(mu) and math.isfinite(sd) and sd > 0.0:
                half = math.sqrt(3.0) * sd
                eff_lo = max(lb, mu - half)
                eff_hi = min(ub, mu + half)
                if math.isfinite(eff_lo) and math.isfinite(eff_hi) and eff_lo < eff_hi:
                    return (
                        eff_lo,
                        eff_hi,
                        0.5 * (eff_lo + eff_hi),
                        (eff_hi - eff_lo) / math.sqrt(12.0),
                        "ok",
                        dist,
                    )
        return lb, ub, float("nan"), float("nan"), "undefined_prior_variance", dist

    if dist == "normal":
        mu, sd = _require_mean_std(mean_raw, std_raw, name, "Normal")
        if not (math.isfinite(mu) and math.isfinite(sd)) or sd <= 0.0:
            return lb, ub, float("nan"), float("nan"), "undefined_prior_variance", dist
        left_om = float(stats.norm.cdf((lb - mu) / sd)) if math.isfinite(lb) else 0.0
        right_om = float(stats.norm.sf((ub - mu) / sd)) if math.isfinite(ub) else 0.0
        if left_om + right_om < trunc_tol:
            return lb, ub, mu, sd, "ok", dist
        p_mean, p_std, status = _numerical_truncated_moments(
            lambda x: _logpdf_normal(x, mu, sd), lb, ub, mu
        )
        return lb, ub, p_mean, p_std, status, dist

    if dist == "beta":
        scale_val = spec.get("scale")
        scale = 1.0 if scale_val is None else float(scale_val)
        if scale <= 0.0:
            raise ValueError(f"Beta prior {name!r}: scale must be > 0, got {scale!r}.")
        mu, sd = _require_mean_std(mean_raw, std_raw, name, "Beta")
        a, b = _beta_ab((mu - shift) / scale, sd / scale)
        eff_lo = max(shift, lb)
        eff_hi = min(shift + scale, ub)
        if not (eff_lo < eff_hi):
            raise ValueError(f"Beta prior {name!r}: empty truncated support [{eff_lo!r}, {eff_hi!r}].")
        u_lo = max(0.0, (eff_lo - shift) / scale)
        u_hi = min(1.0, (eff_hi - shift) / scale)
        left_om = float(stats.beta.cdf(u_lo, a, b)) if u_lo > 0.0 else 0.0
        right_om = float(stats.beta.sf(u_hi, a, b)) if u_hi < 1.0 else 0.0
        if left_om + right_om < trunc_tol:
            return lb, ub, mu, sd, "ok", dist
        if a > 1.0 and b > 1.0:
            u_mode = (a - 1.0) / (a + b - 2.0)
        elif a <= 1.0 and b > 1.0:
            u_mode = 1e-6
        elif b <= 1.0 and a > 1.0:
            u_mode = 1.0 - 1e-6
        else:
            u_mode = 0.5
        mode_hint = shift + scale * u_mode
        p_mean, p_std, status = _numerical_truncated_moments(
            lambda x: _logpdf_beta(x, mu, sd, shift=shift, scale=scale),
            eff_lo,
            eff_hi,
            mode_hint,
        )
        return lb, ub, p_mean, p_std, status, dist

    if dist == "gamma":
        mu, sd = _require_mean_std(mean_raw, std_raw, name, "Gamma")
        adj_mu = mu - shift
        if not (math.isfinite(adj_mu) and math.isfinite(sd)) or adj_mu <= 0.0 or sd <= 0.0:
            raise ValueError(f"Gamma prior {name!r}: require mean > shift and std > 0.")
        k = (adj_mu / sd) ** 2
        theta = (sd ** 2) / adj_mu
        eff_lo = max(shift, lb)
        eff_hi = ub
        if not (eff_lo < eff_hi):
            raise ValueError(f"Gamma prior {name!r}: empty truncated support [{eff_lo!r}, {eff_hi!r}].")
        left_om = float(stats.gamma.cdf(eff_lo - shift, a=k, scale=theta)) if eff_lo > shift else 0.0
        right_om = float(stats.gamma.sf(eff_hi - shift, a=k, scale=theta)) if math.isfinite(eff_hi) else 0.0
        if left_om + right_om < trunc_tol:
            return lb, ub, mu, sd, "ok", dist
        mode_hint = shift + (max(0.0, (k - 1.0) * theta) if k > 1.0 else 1e-6)
        p_mean, p_std, status = _numerical_truncated_moments(
            lambda x: _logpdf_gamma(x, mu, sd, shift=shift), eff_lo, eff_hi, mode_hint
        )
        return lb, ub, p_mean, p_std, status, dist

    if dist == "weibull":
        w_shape = spec.get("shape")
        w_scale = spec.get("scale")
        if w_shape is not None and w_scale is not None:
            shape = float(w_shape)
            scale = float(w_scale)
            base_mu, untrunc_sd = _weibull_mean_std(shape, scale)
            untrunc_mu = base_mu + shift
        else:
            mu, sd = _require_mean_std(mean_raw, std_raw, name, "Weibull")
            shape, scale = _weibull_shape_scale(mu - shift, sd)
            untrunc_mu, untrunc_sd = mu, sd
        eff_lo = max(shift, lb)
        eff_hi = ub
        if not (eff_lo < eff_hi):
            raise ValueError(f"Weibull prior {name!r}: empty truncated support [{eff_lo!r}, {eff_hi!r}].")
        left_om = (
            float(stats.weibull_min.cdf(eff_lo - shift, c=shape, scale=scale))
            if eff_lo > shift
            else 0.0
        )
        right_om = (
            float(stats.weibull_min.sf(eff_hi - shift, c=shape, scale=scale))
            if math.isfinite(eff_hi)
            else 0.0
        )
        if left_om + right_om < trunc_tol:
            return lb, ub, untrunc_mu, untrunc_sd, "ok", dist
        mode_hint = shift + (
            scale * ((shape - 1.0) / shape) ** (1.0 / shape) if shape > 1.0 else 1e-6
        )

        def _lp_weibull(x: float) -> float:
            z = x - shift
            if z <= 0.0:
                return -math.inf
            return float(stats.weibull_min.logpdf(z, c=shape, scale=scale))

        p_mean, p_std, status = _numerical_truncated_moments(_lp_weibull, eff_lo, eff_hi, mode_hint)
        return lb, ub, p_mean, p_std, status, dist

    if dist == "invgamma":
        kind = str(spec.get("kind") or "type1").lower()
        if kind not in ("type1", "type2"):
            raise ValueError(f"InvGamma prior {name!r}: unknown kind {kind!r}.")
        s_val = spec.get("s")
        nu_val = spec.get("nu")
        if s_val is not None and nu_val is not None:
            s, nu = float(s_val), float(nu_val)
        else:
            mu, sd = _require_mean_std(mean_raw, std_raw, name, "InvGamma")
            s, nu = (_invgamma2_s_nu if kind == "type2" else _invgamma_s_nu)(mu, sd)
        if not (s > 0.0 and nu > 0.0 and math.isfinite(s) and math.isfinite(nu)):
            raise ValueError(f"InvGamma prior {name!r}: require finite s > 0, nu > 0.")

        eff_lo = max(0.0, lb)
        eff_hi = ub
        if not (eff_lo < eff_hi):
            raise ValueError(f"InvGamma prior {name!r}: empty support [{eff_lo!r}, {eff_hi!r}].")

        power = 1.0 if kind == "type2" else 2.0
        # Y = X**(-power) ~ Gamma(a=nu/2, scale=2/s); inversion swaps CDF and SF cleanly.
        left_om = (
            float(stats.gamma.sf(eff_lo ** (-power), a=nu / 2.0, scale=2.0 / s))
            if eff_lo > 0.0
            else 0.0
        )
        right_om = (
            float(stats.gamma.cdf(eff_hi ** (-power), a=nu / 2.0, scale=2.0 / s))
            if math.isfinite(eff_hi)
            else 0.0
        )
        var_nu_threshold = 4.0 if kind == "type2" else 2.0
        mean_nu_threshold = 2.0 if kind == "type2" else 1.0

        if nu <= var_nu_threshold and not math.isfinite(eff_hi):
            if nu > mean_nu_threshold and left_om < trunc_tol:
                if kind == "type1":
                    p_mean_val = math.sqrt(s / 2.0) * math.exp(
                        math.lgamma((nu - 1.0) / 2.0) - math.lgamma(nu / 2.0)
                    )
                else:
                    p_mean_val = s / (nu - 2.0)
            else:
                p_mean_val = float("nan")
            return lb, ub, p_mean_val, float("nan"), "undefined_prior_variance", dist

        if nu > var_nu_threshold and left_om + right_om < trunc_tol:
            untrunc_mu, untrunc_sd = (_invgamma2_mean_std if kind == "type2" else _invgamma_mean_std)(s, nu)
            return lb, ub, float(untrunc_mu), float(untrunc_sd), "ok", dist

        mode_hint = (s / (nu + 2.0)) if kind == "type2" else math.sqrt(s / (nu + 1.0))
        logpdf_ig = (
            (lambda x: _logpdf_invgamma2_s_nu(x, s, nu))
            if kind == "type2"
            else (lambda x: _logpdf_invgamma_s_nu(x, s, nu))
        )
        p_mean, p_std, status = _numerical_truncated_moments(logpdf_ig, eff_lo, eff_hi, mode_hint)
        return lb, ub, p_mean, p_std, status, dist

    raise ValueError(f"Unsupported prior distribution {dist!r} for {name!r}.")


# ---------------------------------------------------------------------------
# Prior-Posterior Comparison
# ---------------------------------------------------------------------------

def _resolve_priors_mapping(
    posterior: Any,
    priors: Mapping[str, Any] | Any | None,
) -> dict[str, Any]:
    source = priors if priors is not None else getattr(posterior, "priors", None)
    if source is None:
        raise ValueError(
            "prior_posterior_comparison: no priors supplied and posterior.priors is None. "
            "Pass priors=... (a dict of Prior specs, EstimatedParams, or LinearModel)."
        )
    if isinstance(source, Mapping):
        return dict(source)
    if hasattr(source, "priors") and callable(source.priors):
        return dict(source.priors())
    est_blk = getattr(source, "_estimated_params", None)
    if est_blk is not None and hasattr(est_blk, "priors") and callable(est_blk.priors):
        return dict(est_blk.priors())
    priv_priors = getattr(source, "_priors", None)
    if isinstance(priv_priors, Mapping):
        return dict(priv_priors)
    raise TypeError(
        f"prior_posterior_comparison: cannot extract priors from {type(source).__name__}."
    )


def _extract_chains(posterior: Any) -> tuple[np.ndarray, list[str] | None, dict[str, float], float]:
    raw_draws = getattr(posterior, "draws", None)
    if raw_draws is None:
        raw_draws = getattr(posterior, "chains", None)
    if raw_draws is None:
        raise TypeError(
            f"prior_posterior_comparison: expected posterior with .draws or .chains; "
            f"got {type(posterior).__name__}."
        )
    arr = np.asarray(raw_draws, dtype=float)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D posterior draws array, got shape {arr.shape}.")
    if arr.shape[0] < 1 or arr.shape[1] < 1 or arr.shape[2] < 1:
        raise ValueError(f"Posterior draws array is empty (shape {arr.shape}).")

    raw_names = getattr(posterior, "param_names", None)
    names = [str(n) for n in raw_names] if raw_names is not None else None

    mode_map: dict[str, float] = {}
    raw_mode = getattr(posterior, "mode", None)
    if isinstance(raw_mode, Mapping):
        mode_map = {str(k): float(v) for k, v in raw_mode.items()}
    elif raw_mode is not None and names is not None:
        m_arr = np.asarray(raw_mode, dtype=float).ravel()
        for nm, val in zip(names, m_arr):
            mode_map[nm] = float(val)

    div_arr = getattr(posterior, "divergences", None)
    div_rate = (
        float(np.mean(np.asarray(div_arr, dtype=float)))
        if div_arr is not None and np.asarray(div_arr).size > 0
        else float("nan")
    )
    return arr, names, mode_map, div_rate


def prior_posterior_comparison(
    posterior: Any,
    priors: Mapping[str, Any] | Any | None = None,
    *,
    ci_level: float = 0.90,
    weak_id_threshold: float = 0.80,
    rhat_threshold: float = 1.05,
    ess_threshold: float = 100.0,
    divergence_threshold: float = 0.01,
    boundary_tol: float = 0.02,
    params: Sequence[str] | None = None,
    digits: int = 3,
) -> PriorPosteriorComparisonResult:
    """Compare prior specifications against posterior MCMC draws and diagnostics.

    Parameters
    ----------
    posterior : DSGEPosteriorResult | NUTSResult | BayesianEstimationResult
        Estimation result carrying posterior draws (``.draws`` or ``.chains``)
        and optionally ``.priors``.
    priors : Mapping | EstimatedParams | LinearModel, optional
        Explicit prior mapping or model block; defaults to ``posterior.priors``.
    ci_level : float, default 0.90
        Equal-tailed posterior credible interval coverage in ``(0, 1)``.
    weak_id_threshold : float, default 0.80
        Flag parameter as ``weakly_identified`` when ``post_std / prior_std >= threshold``.
    rhat_threshold : float, default 1.05
        Flag parameter as ``unconverged_or_divergent`` when split-Rhat exceeds this bound.
    ess_threshold : float, default 100.0
        Flag parameter when bulk or tail multichain ESS falls below this bound.
    divergence_threshold : float, default 0.01
        Flag NUTS chains when divergence rate exceeds this share.
    boundary_tol : float, default 0.02
        Relative distance to a finite bound flagging ``near_boundary``.
    params : Sequence[str], optional
        Optional subset/order of parameters to include.
    digits : int, default 3
        Default rounding precision for table export methods.
    """
    if not (0.0 < float(ci_level) < 1.0):
        raise ValueError(f"ci_level must lie strictly in (0, 1), got {ci_level!r}.")

    prior_map = _resolve_priors_mapping(posterior, priors)
    chains, raw_names, mode_map, div_rate = _extract_chains(posterior)
    n_chains, n_draws, n_params = chains.shape
    names = raw_names if raw_names is not None else list(prior_map.keys())[:n_params]
    if len(names) != n_params:
        raise ValueError(
            f"Posterior parameter count ({n_params}) does not match param_names ({len(names)})."
        )

    name_to_idx = {nm: idx for idx, nm in enumerate(names)}
    if params is not None:
        selected = [str(p) for p in params]
        missing_post = [p for p in selected if p not in name_to_idx]
        missing_prior = [p for p in selected if p not in prior_map]
        if missing_post:
            raise ValueError(f"Requested params not in posterior draws: {missing_post}.")
        if missing_prior:
            raise ValueError(f"Requested params missing from priors: {missing_prior}.")
    else:
        selected = [nm for nm in names if nm in prior_map]
        if not selected:
            raise ValueError("No overlap between posterior param_names and prior keys.")

    warn_list: list[str] = []
    if n_chains < 2:
        warn_list.append("Single chain supplied: rank-normalized R-hat requires >= 2 chains (reported as NaN).")

    alpha_half = 0.5 * (1.0 - float(ci_level))
    rows: list[dict[str, Any]] = []
    for nm in selected:
        idx = name_to_idx[nm]
        chain_mat = chains[:, :, idx]
        flat = chain_mat.ravel()
        finite_flat = flat[np.isfinite(flat)]

        lb, ub, p_mean, p_std, p_status, dist = _compute_prior_moments(prior_map[nm], nm)
        if p_status == "undefined_prior_variance":
            warn_list.append(
                f"Prior {nm!r} ({dist}) has undefined variance on [{lb:g}, {ub:g}]; prior_std reported as NaN."
            )
        elif p_status == "numerical_failed":
            warn_list.append(f"Numerical integration failed for truncated prior {nm!r}.")

        if finite_flat.size > 0:
            post_mean = float(np.mean(finite_flat))
            post_median = float(np.median(finite_flat))
            post_std = float(np.std(finite_flat, ddof=1)) if finite_flat.size > 1 else 0.0
            ci_lo = float(np.quantile(finite_flat, alpha_half))
            ci_hi = float(np.quantile(finite_flat, 1.0 - alpha_half))
        else:
            post_mean = post_median = post_std = ci_lo = ci_hi = float("nan")

        post_mode = mode_map.get(nm, float("nan"))
        if math.isfinite(p_std) and p_std > 0.0 and math.isfinite(post_std):
            std_ratio = post_std / p_std
        else:
            std_ratio = float("nan")
        if math.isfinite(p_mean) and math.isfinite(p_std) and p_std > 0.0 and math.isfinite(post_mean):
            prior_shift_z = (post_mean - p_mean) / p_std
        else:
            prior_shift_z = float("nan")

        rhat = float(rank_normalized_rhat(chain_mat))
        ess_bulk = float(multichain_ess(chain_mat, method="bulk"))
        ess_tail = float(multichain_ess(chain_mat, method="tail"))

        weakly_id = bool(
            p_status != "undefined_prior_variance"
            and math.isfinite(std_ratio)
            and std_ratio >= float(weak_id_threshold)
        )

        # Near-boundary check uses full interval width when bounded on both sides,
        # and prior/posterior dispersion scale on half-bounded supports.
        if math.isfinite(lb) and math.isfinite(ub):
            scale_b = max(ub - lb, 1e-12)
        elif math.isfinite(p_std) and p_std > 0.0:
            scale_b = p_std
        else:
            scale_b = max(post_std if math.isfinite(post_std) else 1.0, 1e-12)

        near_lo = bool(math.isfinite(lb) and math.isfinite(post_median) and (post_median - lb) <= float(boundary_tol) * scale_b)
        near_hi = bool(math.isfinite(ub) and math.isfinite(post_median) and (ub - post_median) <= float(boundary_tol) * scale_b)
        near_bnd = near_lo or near_hi

        bad_rhat = (not math.isnan(rhat)) and (rhat > float(rhat_threshold))
        bad_bulk = math.isfinite(ess_bulk) and (ess_bulk < float(ess_threshold))
        bad_tail = math.isfinite(ess_tail) and (ess_tail < float(ess_threshold))
        bad_div = math.isfinite(div_rate) and (div_rate > float(divergence_threshold))
        unconv = bool(bad_rhat or bad_bulk or bad_tail or bad_div)

        active_flags = [
            tag
            for tag, flag in (
                ("weakly_identified", weakly_id),
                ("near_boundary", near_bnd),
                ("unconverged_or_divergent", unconv),
            )
            if flag
        ]
        rows.append(
            {
                "param": nm,
                "dist": dist,
                "lb": lb,
                "ub": ub,
                "prior_mean": p_mean,
                "prior_std": p_std,
                "prior_status": p_status,
                "post_mean": post_mean,
                "post_median": post_median,
                "post_std": post_std,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "post_mode": post_mode,
                "std_ratio": std_ratio,
                "prior_shift_z": prior_shift_z,
                "rhat": rhat,
                "ess_bulk": ess_bulk,
                "ess_tail": ess_tail,
                "divergence_rate": div_rate,
                "weakly_identified": weakly_id,
                "near_boundary": near_bnd,
                "unconverged_or_divergent": unconv,
                "flags": ",".join(active_flags),
            }
        )

    return PriorPosteriorComparisonResult(
        table=pd.DataFrame(rows),
        ci_level=float(ci_level),
        n_chains=n_chains,
        n_draws=n_draws,
        model_name=str(getattr(posterior, "model_name", "") or ""),
        warnings=tuple(dict.fromkeys(warn_list)),
        digits=int(digits),
    )


# ---------------------------------------------------------------------------
# Empirical IRF Validation Scorecard
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _TargetMoment:
    label: str
    variable: str
    shock: str | None
    h: int
    value: float
    se: float
    lower: float
    upper: float


def _parse_target_label(
    label: str,
    default_shock: str | None = None,
    default_var: str | None = None,
) -> tuple[str, str | None, int]:
    text = str(label).strip()
    m1 = _LABEL_COLON_RE.match(text)
    if m1:
        return m1.group(1), m1.group(2), int(m1.group(3))
    m2 = _LABEL_ARROW_RE.match(text)
    if m2:
        return m2.group(1), m2.group(2), int(m2.group(3))
    m3 = _LABEL_VAR_H_RE.match(text)
    if m3:
        return m3.group(1), default_shock, int(m3.group(2))
    m4 = _LABEL_BRACKET_H_RE.match(text)
    if m4:
        return m4.group(1), default_shock, int(m4.group(2))
    if default_var is not None:
        try:
            return default_var, default_shock, int(text)
        except ValueError:
            pass
    raise ValueError(
        f"Cannot parse target moment label {label!r}; expected formats like "
        "'y:eps:h=0' or 'y<-eps[h=0]'."
    )


def _extract_target_moments(
    targets: Any,
    ci_level: float,
) -> tuple[list[_TargetMoment], np.ndarray | None, float | None, list[str]]:
    z_crit = float(stats.norm.isf(0.5 * (1.0 - float(ci_level))))
    warn_list: list[str] = []

    # 1. MomentTargets (from lp_moment_targets or MomentTargets.from_irf)
    if hasattr(targets, "values") and hasattr(targets, "labels") and hasattr(targets, "covariance"):
        vals = np.asarray(targets.values, dtype=float).ravel()
        labels = [str(lbl) for lbl in targets.labels]
        meta = getattr(targets, "metadata", None) or {}
        default_shock = str(meta["shock"]) if isinstance(meta, Mapping) and meta.get("shock") else None
        raw_resp = meta.get("response") if isinstance(meta, Mapping) else None
        default_var = raw_resp if isinstance(raw_resp, str) else None
        shock_sz_raw = meta.get("shock_size") if isinstance(meta, Mapping) else None
        target_shock_sz = float(shock_sz_raw) if shock_sz_raw is not None else None

        cov = (
            np.asarray(targets.covariance, dtype=float)
            if targets.covariance is not None
            else None
        )
        if cov is not None and cov.shape != (len(vals), len(vals)):
            raise ValueError(
                f"MomentTargets covariance shape {cov.shape} does not match {len(vals)} moments."
            )
        diag_se = (
            np.sqrt(np.maximum(np.diag(cov), 0.0))
            if cov is not None
            else np.full(len(vals), np.nan, dtype=float)
        )

        moments: list[_TargetMoment] = []
        for idx, (lbl, val) in enumerate(zip(labels, vals)):
            var, shk, h_val = _parse_target_label(lbl, default_shock=default_shock, default_var=default_var)
            se_i = float(diag_se[idx]) if np.isfinite(diag_se[idx]) and diag_se[idx] > 0.0 else float("nan")
            lo_i = float(val - z_crit * se_i) if math.isfinite(se_i) else float("nan")
            hi_i = float(val + z_crit * se_i) if math.isfinite(se_i) else float("nan")
            moments.append(
                _TargetMoment(
                    label=lbl,
                    variable=var,
                    shock=shk,
                    h=h_val,
                    value=float(val),
                    se=se_i,
                    lower=lo_i,
                    upper=hi_i,
                )
            )
        return moments, cov, target_shock_sz, warn_list

    # 2. Single LPResult/DataFrame with a y_name or single response
    if isinstance(targets, pd.DataFrame) and ("beta" in targets.columns or "point" in targets.columns):
        y_name = str(getattr(targets, "y_name", None) or "y")
        targets = {y_name: targets}

    # 3. Mapping[str, LPResult | DataFrame]
    if isinstance(targets, Mapping):
        moments = []
        for key, df_obj in targets.items():
            if not isinstance(df_obj, pd.DataFrame):
                raise TypeError(
                    f"Target mapping value for {key!r} must be LPResult/DataFrame; got {type(df_obj).__name__}."
                )
            key_str = str(key)
            shk_part: str | None
            if ":" in key_str:
                var_part, shk_part = key_str.split(":", 1)
            else:
                var_part = key_str
                shk_attr = getattr(df_obj, "x_name", None)
                shk_part = str(shk_attr) if shk_attr else None

            h_col = df_obj["h"].to_numpy(dtype=int) if "h" in df_obj.columns else np.asarray(df_obj.index, dtype=int)
            pt_name = "beta" if "beta" in df_obj.columns else ("point" if "point" in df_obj.columns else None)
            if pt_name is None:
                raise ValueError(f"Target DataFrame for {key!r} needs a 'beta' or 'point' column.")
            pts = df_obj[pt_name].to_numpy(dtype=float)
            ses = df_obj["se"].to_numpy(dtype=float) if "se" in df_obj.columns else np.full(len(pts), np.nan)
            lo_col = "lo" if "lo" in df_obj.columns else ("lower" if "lower" in df_obj.columns else None)
            hi_col = "hi" if "hi" in df_obj.columns else ("upper" if "upper" in df_obj.columns else None)
            los = df_obj[lo_col].to_numpy(dtype=float) if lo_col else (pts - z_crit * ses)
            his = df_obj[hi_col].to_numpy(dtype=float) if hi_col else (pts + z_crit * ses)

            for h_v, pt_v, se_v, lo_v, hi_v in zip(h_col, pts, ses, los, his):
                lbl = f"{var_part}:{shk_part}:h={int(h_v)}" if shk_part else f"{var_part}:h={int(h_v)}"
                moments.append(
                    _TargetMoment(
                        label=lbl,
                        variable=var_part,
                        shock=shk_part,
                        h=int(h_v),
                        value=float(pt_v),
                        se=float(se_v),
                        lower=float(lo_v),
                        upper=float(hi_v),
                    )
                )
        return moments, None, None, warn_list

    # 4. Long DataFrame with explicit variable / horizon / target columns
    if isinstance(targets, pd.DataFrame):
        var_col = "variable" if "variable" in targets.columns else ("var" if "var" in targets.columns else None)
        val_col = next((c for c in ("target", "value", "estimate", "point", "beta") if c in targets.columns), None)
        if var_col is None or "h" not in targets.columns or val_col is None:
            raise ValueError(
                "Target DataFrame must either be an LPResult or contain ['variable', 'h', 'target'/'value']."
            )
        moments = []
        for _, row in targets.iterrows():
            var_v = str(row[var_col])
            shk_v = str(row["shock"]) if "shock" in targets.columns and pd.notna(row["shock"]) else None
            h_v = int(row["h"])
            val_v = float(row[val_col])
            se_v = float(row["se"]) if "se" in targets.columns and pd.notna(row["se"]) else float("nan")
            lo_v = (
                float(row["lower"])
                if "lower" in targets.columns and pd.notna(row["lower"])
                else (val_v - z_crit * se_v if math.isfinite(se_v) else float("nan"))
            )
            hi_v = (
                float(row["upper"])
                if "upper" in targets.columns and pd.notna(row["upper"])
                else (val_v + z_crit * se_v if math.isfinite(se_v) else float("nan"))
            )
            lbl = f"{var_v}:{shk_v}:h={h_v}" if shk_v else f"{var_v}:h={h_v}"
            moments.append(_TargetMoment(lbl, var_v, shk_v, h_v, val_v, se_v, lo_v, hi_v))
        return moments, None, None, warn_list

    raise TypeError(f"Unsupported targets container type {type(targets).__name__}.")


def _extract_model_irf_frames(
    model_irf: Any,
    shock: str | None,
    ci_level: float,
    target_shock_size: float | None,
    hank_unit_mode: str,
    warn_list: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, str | None, float]:
    """Return ``(point_df, lower_df, upper_df, resolved_shock, determinacy_rate)``."""
    # 1. BayesianIRFResult
    if hasattr(model_irf, "median") and hasattr(model_irf, "bands") and hasattr(model_irf, "variables"):
        shocks = list(getattr(model_irf, "shocks", ()) or ())
        single_shk = getattr(model_irf, "shock", None)
        avail_shocks = shocks if shocks else ([str(single_shk)] if single_shk else [])
        med_obj = model_irf.median
        det_rate = float(getattr(model_irf, "determinacy_rate", float("nan")))

        if isinstance(med_obj, Mapping):
            avail = list(med_obj.keys())
            if shock is None:
                if len(avail) == 1:
                    shk_name = str(avail[0])
                else:
                    raise ValueError(
                        f"BayesianIRFResult has multiple shocks {avail}; pass shock=..."
                    )
            else:
                shk_name = str(shock)
            if shk_name not in med_obj:
                raise ValueError(f"Shock {shk_name!r} not found in BayesianIRFResult shocks {avail}.")
            pt_df = med_obj[shk_name].copy()
        else:
            pt_df = med_obj.copy()
            if shock is not None and avail_shocks and str(shock) not in avail_shocks:
                raise ValueError(
                    f"Shock {shock!r} not found in BayesianIRFResult shocks {avail_shocks}."
                )
            shk_name = str(shock) if shock is not None else (str(avail_shocks[0]) if avail_shocks else "")

        raw_bands = getattr(model_irf, "bands", None) or {}
        if isinstance(raw_bands, Mapping) and shk_name in raw_bands and isinstance(raw_bands[shk_name], Mapping):
            band_map = raw_bands[shk_name]
        else:
            band_map = raw_bands

        avail_cis: list[tuple[float, Any]] = []
        for k_b in band_map.keys():
            try:
                avail_cis.append((float(k_b), k_b))
            except (TypeError, ValueError):
                continue

        lo_df = hi_df = None
        if avail_cis:
            matched = [pair for pair in avail_cis if abs(pair[0] - float(ci_level)) <= 1e-4]
            chosen_key = matched[0][1] if matched else sorted(avail_cis, key=lambda pair: pair[0])[-1][1]
            lo_raw, hi_raw = band_map[chosen_key]
            lo_df = (lo_raw[shk_name] if isinstance(lo_raw, Mapping) else lo_raw).copy()
            hi_df = (hi_raw[shk_name] if isinstance(hi_raw, Mapping) else hi_raw).copy()

        return pt_df, lo_df, hi_df, (shk_name or None), det_rate

    # 2. HANKResult
    if hasattr(model_irf, "transition_paths") and hasattr(model_irf, "steady_state"):
        df_paths = model_irf.transition_paths.copy()
        hank_shk = str(getattr(model_irf, "shock_name", "") or "")
        shk_out = str(shock) if shock is not None else (hank_shk or None)

        s0: float | None = None
        if hank_shk and hank_shk in df_paths.columns and len(df_paths) > 0:
            val0 = float(df_paths[hank_shk].iloc[0])
            if math.isfinite(val0) and val0 != 0.0:
                s0 = val0
        if s0 is None:
            diag = getattr(model_irf, "diagnostics", None) or {}
            if isinstance(diag, Mapping) and diag.get("shock_magnitude") is not None:
                val_d = float(diag["shock_magnitude"])
                if math.isfinite(val_d) and val_d != 0.0:
                    s0 = val_d

        if s0 is not None:
            if target_shock_size is not None and target_shock_size != 0.0:
                if s0 * target_shock_size < 0.0:
                    warn_list.append(
                        f"HANKResult shock magnitude ({s0:g}) has opposite sign to target shock_size ({target_shock_size:g}); verify expansionary vs contractionary convention."
                    )
                ratio = abs(s0) / abs(target_shock_size)
                if ratio < 0.05 or ratio > 20.0:
                    warn_list.append(
                        f"HANKResult shock magnitude ({s0:g}) differs in scale from target shock_size ({target_shock_size:g})."
                    )
                if hank_unit_mode == "scale_to_target":
                    scale_factor = target_shock_size / s0
                    num_cols = df_paths.select_dtypes(include=[np.number]).columns
                    df_paths.loc[:, num_cols] = df_paths.loc[:, num_cols] * scale_factor
                    warn_list.append(f"Scaled HANKResult transition_paths by {scale_factor:g} to match target shock_size.")
            elif s0 < 0.0 or abs(s0) < 0.05:
                warn_list.append(
                    f"HANKResult shock path starts at {s0:g} (HANKModel.simulate defaults to -0.0025 = -25bps); verify sign and unit scaling against targets."
                )

        return df_paths, None, None, shk_out, float("nan")

    # 3. Mapping[str, DataFrame]
    if isinstance(model_irf, Mapping):
        avail = list(model_irf.keys())
        if shock is None:
            if len(avail) == 1:
                shk_name = str(avail[0])
            else:
                raise ValueError(f"Model IRF mapping has multiple shocks {avail}; pass shock=...")
        else:
            shk_name = str(shock)
        if shk_name not in model_irf:
            raise ValueError(f"Shock {shk_name!r} not in model_irf keys {avail}.")
        return model_irf[shk_name].copy(), None, None, shk_name, float("nan")

    # 4. DataFrame
    if isinstance(model_irf, pd.DataFrame):
        return model_irf.copy(), None, None, shock, float("nan")

    raise TypeError(f"Unsupported model_irf container type {type(model_irf).__name__}.")


def _index_horizon_map(df: pd.DataFrame) -> pd.DataFrame:
    if "h" in df.columns:
        out = df.set_index("h")
    else:
        out = df.copy()
    out.index = out.index.astype(int)
    return out


def score_irf_against_targets(
    model_irf: Any,
    targets: Any,
    *,
    shock: str | None = None,
    variables: Sequence[str] | None = None,
    horizons: Sequence[int] | None = None,
    weight_mode: Literal["auto", "covariance", "diagonal"] = "auto",
    ci_level: float = 0.90,
    cond_warn: float = 1e6,
    hank_unit_mode: Literal["warn", "raw", "scale_to_target"] = "warn",
    digits: int = 3,
) -> IRFScorecardResult:
    """Score a DSGE or HANK impulse response against empirical moment targets.

    Parameters
    ----------
    model_irf : BayesianIRFResult | HANKResult | pd.DataFrame | Mapping[str, pd.DataFrame]
        Simulated or posterior-median model impulse responses.
    targets : MomentTargets | Mapping[str, LPResult] | pd.DataFrame
        Empirical moment targets (e.g. from ``lp_moment_targets`` or
        ``MomentTargets.from_irf``) or marginal LP results by variable.
    shock : str, optional
        Shock name used when ``model_irf`` or ``targets`` carries multiple shocks.
    variables : Sequence[str], optional
        Optional subset of response variables to score.
    horizons : Sequence[int], optional
        Optional subset of horizons to score.
    weight_mode : {'auto', 'covariance', 'diagonal'}, default 'auto'
        Use joint target covariance when finite (rank-aware generalized Wald),
        or diagonal target standard errors.
    ci_level : float, default 0.90
        Target interval coverage used to construct ``target_lower``/``target_upper``
        when only standard errors or covariance are available.
    cond_warn : float, default 1e6
        Warn when retained covariance condition number exceeds this threshold.
    hank_unit_mode : {'warn', 'raw', 'scale_to_target'}, default 'warn'
        Policy when scoring ``HANKResult`` whose default ``-0.0025`` shock may
        differ in sign/scale from unit-shock empirical targets.
    digits : int, default 3
        Default rounding precision for table export methods.
    """
    if weight_mode not in ("auto", "covariance", "diagonal"):
        raise ValueError(f"Unknown weight_mode {weight_mode!r}; expected 'auto', 'covariance', or 'diagonal'.")
    if hank_unit_mode not in ("warn", "raw", "scale_to_target"):
        raise ValueError(f"Unknown hank_unit_mode {hank_unit_mode!r}; expected 'warn', 'raw', or 'scale_to_target'.")

    moments, raw_cov, target_shock_sz, warn_list = _extract_target_moments(targets, ci_level=ci_level)
    shk_set = list(dict.fromkeys(m.shock for m in moments if m.shock is not None))
    effective_shock = shock
    if effective_shock is None and len(shk_set) == 1:
        med_obj = getattr(model_irf, "median", None)
        if isinstance(med_obj, Mapping) and shk_set[0] in med_obj:
            effective_shock = shk_set[0]
        elif isinstance(model_irf, Mapping) and shk_set[0] in model_irf:
            effective_shock = shk_set[0]

    pt_raw, lo_raw, hi_raw, model_shk, det_rate = _extract_model_irf_frames(
        model_irf,
        shock=effective_shock,
        ci_level=ci_level,
        target_shock_size=target_shock_sz,
        hank_unit_mode=hank_unit_mode,
        warn_list=warn_list,
    )
    pt_df = _index_horizon_map(pt_raw)
    lo_df = _index_horizon_map(lo_raw) if lo_raw is not None else None
    hi_df = _index_horizon_map(hi_raw) if hi_raw is not None else None

    var_filter = {str(v) for v in variables} if variables is not None else None
    h_filter = {int(h) for h in horizons} if horizons is not None else None

    # Disambiguate target shock if targets carry multiple shocks.
    target_shock_filter: str | None = shock if (shock is not None and shock in shk_set) else None
    if target_shock_filter is None and len(shk_set) > 1:
        raise ValueError(f"Targets span multiple shocks {shk_set}; pass shock=...")

    kept_idx: list[int] = []
    dropped: list[str] = []
    rows: list[dict[str, Any]] = []

    for idx, m in enumerate(moments):
        if var_filter is not None and m.variable not in var_filter:
            continue
        if h_filter is not None and m.h not in h_filter:
            continue
        if target_shock_filter is not None and m.shock is not None and m.shock != target_shock_filter:
            continue

        if m.variable not in pt_df.columns or m.h not in pt_df.index:
            dropped.append(m.label)
            continue

        m_val = float(pt_df.loc[m.h, m.variable])
        if not (math.isfinite(m_val) and math.isfinite(m.value)):
            dropped.append(m.label)
            continue

        m_lo = (
            float(lo_df.loc[m.h, m.variable])
            if (lo_df is not None and m.variable in lo_df.columns and m.h in lo_df.index)
            else float("nan")
        )
        m_hi = (
            float(hi_df.loc[m.h, m.variable])
            if (hi_df is not None and m.variable in hi_df.columns and m.h in hi_df.index)
            else float("nan")
        )
        diff = m_val - m.value
        z_sc = diff / m.se if (math.isfinite(m.se) and m.se > 0.0) else float("nan")

        if abs(m_val) <= 1e-12 and abs(m.value) <= 1e-12:
            sign_ok = True
        elif abs(m_val) <= 1e-12 or abs(m.value) <= 1e-12:
            sign_ok = False
        else:
            sign_ok = bool((m_val > 0.0) == (m.value > 0.0))

        has_band = math.isfinite(m.lower) and math.isfinite(m.upper)
        in_band = bool(has_band and (m.lower <= m_val <= m.upper))

        kept_idx.append(idx)
        rows.append(
            {
                "label": m.label,
                "variable": m.variable,
                "shock": m.shock or model_shk or "",
                "h": m.h,
                "model": m_val,
                "target": m.value,
                "target_se": m.se,
                "target_lower": m.lower,
                "target_upper": m.upper,
                "model_lower": m_lo,
                "model_upper": m_hi,
                "diff": diff,
                "z_score": z_sc,
                "sign_match": sign_ok,
                "within_target_band": in_band,
            }
        )

    if dropped:
        preview = ", ".join(dropped[:5]) + ("..." if len(dropped) > 5 else "")
        warn_list.append(f"Dropped {len(dropped)} unmatched/non-finite target moments ({preview}).")
    if not rows:
        raise ValueError(
            "score_irf_against_targets: no overlapping finite (variable, horizon) moments between model_irf and targets."
        )

    df_table = pd.DataFrame(rows)
    diffs = df_table["diff"].to_numpy(dtype=float)
    se_arr = df_table["target_se"].to_numpy(dtype=float)
    n_mom = len(df_table)

    rmse = float(np.sqrt(np.mean(diffs ** 2)))
    mae = float(np.mean(np.abs(diffs)))
    sign_concordance = float(df_table["sign_match"].mean())
    has_any_band = df_table["target_lower"].notna() & df_table["target_upper"].notna()
    band_coverage = (
        float(df_table.loc[has_any_band, "within_target_band"].mean())
        if bool(has_any_band.any())
        else float("nan")
    )

    cov_sub = raw_cov[np.ix_(kept_idx, kept_idx)] if raw_cov is not None else None
    cov_finite = cov_sub is not None and bool(np.all(np.isfinite(cov_sub)))

    if weight_mode == "covariance" and not cov_finite:
        raise ValueError(
            "weight_mode='covariance' requires a MomentTargets object with a finite covariance matrix."
        )

    wse = float("nan")
    chi2_stat = float("nan")
    chi2_df = 0
    chi2_pvalue = float("nan")
    cond_num = float("nan")

    if weight_mode in ("auto", "covariance") and cov_finite and cov_sub is not None:
        sym_cov = 0.5 * (cov_sub + cov_sub.T)
        eigvals, eigvecs = np.linalg.eigh(sym_cov)
        max_eig = float(np.max(eigvals)) if eigvals.size else 0.0
        min_eig = float(np.min(eigvals)) if eigvals.size else 0.0
        if max_eig <= 0.0:
            if weight_mode == "covariance":
                raise ValueError("Target covariance matrix has no positive eigenvalues.")
            warn_list.append("Target covariance has no positive eigenvalues; falling back to diagonal weighting.")
            weight_used = "diagonal"
        else:
            if min_eig < -1e-8 * max(max_eig, 1.0):
                warn_list.append(
                    f"Target covariance matrix is indefinite (min eigenvalue {min_eig:.3e}); projecting onto positive subspace."
                )
            tol = max(1e-12, 1e-10 * max_eig)
            keep_eig = eigvals > tol
            chi2_df = int(np.sum(keep_eig))
            pos_eigs = eigvals[keep_eig]
            cond_num = (
                float(pos_eigs.max() / pos_eigs.min())
                if pos_eigs.size >= 2
                else 1.0
            )
            if chi2_df < n_mom:
                warn_list.append(
                    f"Target covariance is rank-deficient (rank {chi2_df} < {n_mom}); Wald chi2 uses df={chi2_df}."
                )
            if cond_num > float(cond_warn):
                warn_list.append(
                    f"Target covariance condition number is {cond_num:.2e} (> {cond_warn:.2e})."
                )
            proj = eigvecs[:, keep_eig].T @ diffs
            chi2_stat = float(np.sum((proj ** 2) / pos_eigs))
            wse = chi2_stat
            # Right-tail chi-squared p-value strictly via .sf (never 1 - .cdf).
            chi2_pvalue = float(stats.chi2.sf(chi2_stat, df=chi2_df))
            weight_used = "covariance"
    else:
        valid_se = np.isfinite(se_arr) & (se_arr > 0.0)
        if np.any(valid_se):
            wse = float(np.sum((diffs[valid_se] / se_arr[valid_se]) ** 2))
            weight_used = "diagonal"
            if cov_sub is None:
                warn_list.append(
                    "Cross-horizon target covariance unavailable; reporting diagonal WSE with chi2_pvalue=NaN."
                )
        else:
            weight_used = "unweighted"
            warn_list.append("No finite target standard errors or covariance supplied; WSE and chi2 are NaN.")

    metric_rows: list[dict[str, Any]] = [
        {
            "scope": "overall",
            "n_moments": n_mom,
            "rmse": rmse,
            "mae": mae,
            "wse": wse,
            "chi2_stat": chi2_stat,
            "chi2_df": chi2_df,
            "chi2_pvalue": chi2_pvalue,
            "sign_concordance": sign_concordance,
            "band_coverage": band_coverage,
            "determinacy_rate": det_rate,
            "weight_mode": weight_used,
        }
    ]
    for var_name in dict.fromkeys(df_table["variable"].tolist()):
        sub_v = df_table[df_table["variable"] == var_name]
        d_v = sub_v["diff"].to_numpy(dtype=float)
        se_v = sub_v["target_se"].to_numpy(dtype=float)
        valid_se_v = np.isfinite(se_v) & (se_v > 0.0)
        has_band_v = sub_v["target_lower"].notna() & sub_v["target_upper"].notna()
        metric_rows.append(
            {
                "scope": str(var_name),
                "n_moments": int(len(sub_v)),
                "rmse": float(np.sqrt(np.mean(d_v ** 2))),
                "mae": float(np.mean(np.abs(d_v))),
                "wse": float(np.sum((d_v[valid_se_v] / se_v[valid_se_v]) ** 2)) if np.any(valid_se_v) else float("nan"),
                "chi2_stat": float("nan"),
                "chi2_df": 0,
                "chi2_pvalue": float("nan"),
                "sign_concordance": float(sub_v["sign_match"].mean()),
                "band_coverage": (
                    float(sub_v.loc[has_band_v, "within_target_band"].mean())
                    if bool(has_band_v.any())
                    else float("nan")
                ),
                "determinacy_rate": det_rate,
                "weight_mode": "diagonal" if np.any(valid_se_v) else "unweighted",
            }
        )

    return IRFScorecardResult(
        table=df_table,
        metrics=pd.DataFrame(metric_rows),
        rmse=rmse,
        mae=mae,
        wse=wse,
        chi2_stat=chi2_stat,
        chi2_df=chi2_df,
        chi2_pvalue=chi2_pvalue,
        sign_concordance=sign_concordance,
        band_coverage=band_coverage,
        determinacy_rate=det_rate,
        weight_mode_used=weight_used,
        covariance_condition_number=cond_num,
        warnings=tuple(dict.fromkeys(warn_list)),
        digits=int(digits),
    )


__all__ = [
    "PriorPosteriorComparisonResult",
    "IRFScorecardResult",
    "prior_posterior_comparison",
    "score_irf_against_targets",
]
