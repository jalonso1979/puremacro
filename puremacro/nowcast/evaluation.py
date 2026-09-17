"""Professional macroeconomic forecast evaluation suite.

Implements:
1. Probability Integral Transform (PIT) uniformity tests:
   - Berkowitz (2001) Likelihood Ratio test for calibration and independence.
   - Kolmogorov-Smirnov test against U(0, 1).
   - PIT histogram and diagnostics returning :class:`PITUniformityResult`.
2. Institutional Central Bank Fan Charts:
   - Shaded quantile ribbons across forecast horizons.
   - Central bank palettes: Banxico, BCB, Bank of England, default.
   - Returns :class:`FanChartResult`.
3. Probabilistic scoring rules (CRPS, log score, Brier score).

References
----------
Berkowitz, J. (2001). "Testing density forecasts, with applications to
    risk management." Journal of Business & Economic Statistics, 19(4), 465-474.
Diebold, F. X., Gunther, T. A. and Tay, A. S. (1998). "Evaluating density
    forecasts with applications to financial risk management." IER 39(4), 863-883.
Gneiting, T. and Raftery, A. E. (2007). "Strictly proper scoring rules,
    prediction, and estimation." JASA 102(477), 359-378.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, kstest, norm

from .scoring import (
    brier_score,
    crps_ensemble,
    crps_gaussian,
    log_score_gaussian,
    pit_histogram,
)


@dataclass(frozen=True)
class PITUniformityResult:
    """Immutable result object from Probability Integral Transform (PIT) evaluation.

    Attributes
    ----------
    pit : np.ndarray
        Empirical PIT values in (0, 1), shape (T,).
    z : np.ndarray
        Standard normal inverse transformed quantiles Φ^-1(pit), shape (T,).
    lr_stat : float
        Berkowitz (2001) Likelihood Ratio test statistic.
    lr_pvalue : float
        P-value of Berkowitz LR test under Chi-square(3).
    df : int
        Degrees of freedom for LR test (default 3: μ=0, σ^2=1, ρ=0).
    mu : float
        Estimated mean of inverse-normal transformed series z_t.
    sigma : float
        Estimated standard deviation of AR(1) innovations.
    rho : float
        Estimated AR(1) persistence coefficient.
    ks_stat : float
        Kolmogorov-Smirnov test statistic against standard uniform U(0, 1).
    ks_pvalue : float
        P-value of Kolmogorov-Smirnov test.
    is_uniform : bool
        True if neither Berkowitz LR test nor KS test reject at nominal level (p > 0.05).
    n_obs : int
        Sample size (number of forecast evaluations T).
    hist_counts : np.ndarray
        Bin counts of the PIT histogram.
    hist_edges : np.ndarray
        Bin edges on [0, 1].
    """

    pit: np.ndarray
    z: np.ndarray
    lr_stat: float
    lr_pvalue: float
    df: int
    mu: float
    sigma: float
    rho: float
    ks_stat: float
    ks_pvalue: float
    is_uniform: bool
    n_obs: int
    hist_counts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hist_edges: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def summary(self) -> str:
        """Text summary of PIT uniformity and calibration tests."""
        verdict = (
            "PASS: Forecast distribution is well-calibrated (cannot reject uniformity)"
            if self.is_uniform
            else "FAIL: Forecast distribution is mis-calibrated (rejects uniformity)"
        )
        lines = [
            "=" * 72,
            "Probability Integral Transform (PIT) Calibration Evaluation",
            "=" * 72,
            f"Observations evaluated (T)     : {self.n_obs}",
            f"Calibration Verdict            : {verdict}",
            "-" * 72,
            "Berkowitz (2001) Likelihood Ratio Test [H0: z ~ i.i.d. N(0, 1)]:",
            f"  LR test statistic            : {self.lr_stat:.4f}",
            f"  Degrees of freedom           : {self.df}",
            f"  p-value                      : {self.lr_pvalue:.4f} "
            + ("(Reject H0 at 5%)" if self.lr_pvalue <= 0.05 else "(Fail to reject H0)"),
            f"  Estimated AR(1) parameters   : μ = {self.mu:+.4f}, σ = {self.sigma:.4f}, ρ = {self.rho:+.4f}",
            "-" * 72,
            "Kolmogorov-Smirnov Goodness-of-Fit Test [H0: PIT ~ U(0, 1)]:",
            f"  KS test statistic            : {self.ks_stat:.4f}",
            f"  p-value                      : {self.ks_pvalue:.4f} "
            + ("(Reject H0 at 5%)" if self.ks_pvalue <= 0.05 else "(Fail to reject H0)"),
            "-" * 72,
            "Diagnostic Interpretation:",
        ]
        if self.mu > 0.15:
            lines.append("  • Positive μ: Forecasts are underpredicting on average (negative bias).")
        elif self.mu < -0.15:
            lines.append("  • Negative μ: Forecasts are overpredicting on average (positive bias).")
        if self.sigma > 1.15:
            lines.append("  • High σ > 1: Forecast uncertainty is underestimated (intervals too narrow).")
        elif self.sigma < 0.85:
            lines.append("  • Low σ < 1: Forecast uncertainty is overestimated (intervals too wide).")
        if abs(self.rho) > 0.2:
            lines.append(f"  • Non-zero ρ ({self.rho:+.2f}): Forecast errors exhibit temporal autocorrelation.")
        if self.is_uniform:
            lines.append("  • Predictive density satisfies both calibration and independence.")
        lines.append("=" * 72)
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Diagnostics summary table."""
        data = [
            {"Test": "Berkowitz LR (3-df)", "Statistic": self.lr_stat, "P-Value": self.lr_pvalue, "Verdict": "Fail to Reject" if self.lr_pvalue > 0.05 else "Reject"},
            {"Test": "Kolmogorov-Smirnov", "Statistic": self.ks_stat, "P-Value": self.ks_pvalue, "Verdict": "Fail to Reject" if self.ks_pvalue > 0.05 else "Reject"},
            {"Test": "AR(1) Mean (μ)", "Statistic": self.mu, "P-Value": float("nan"), "Verdict": f"Null: 0.0"},
            {"Test": "AR(1) Volatility (σ)", "Statistic": self.sigma, "P-Value": float("nan"), "Verdict": f"Null: 1.0"},
            {"Test": "AR(1) Persistence (ρ)", "Statistic": self.rho, "P-Value": float("nan"), "Verdict": f"Null: 0.0"},
        ]
        df = pd.DataFrame(data)
        df[["Statistic", "P-Value"]] = df[["Statistic", "P-Value"]].round(4)
        return df

    def to_markdown(self, **kwargs: Any) -> str:
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), **kwargs)

    def plot(self, *, ax: Any = None, title: str = "PIT Calibration Diagnostics") -> Any:
        """Plot PIT histogram and Normal QQ plot side-by-side."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
        else:
            fig = ax.figure
            ax1, ax2 = ax, None

        # 1. PIT Histogram
        n_bins = len(self.hist_counts) if len(self.hist_counts) else 10
        edges = self.hist_edges if len(self.hist_edges) else np.linspace(0, 1, n_bins + 1)
        ax1.hist(
            self.pit,
            bins=edges,
            density=True,
            color="steelblue",
            alpha=0.7,
            edgecolor="white",
        )
        ax1.axhline(1.0, color="firebrick", lw=1.5, ls="--", label="Uniform Benchmark")
        ax1.set_xlabel("PIT p_t")
        ax1.set_ylabel("Density")
        ax1.set_title("PIT Empirical Distribution")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, ls=":", alpha=0.5)

        # 2. Normal QQ Plot (if second axis available)
        if ax2 is not None:
            sorted_z = np.sort(self.z)
            n = len(sorted_z)
            theoretical_quantiles = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
            ax2.scatter(theoretical_quantiles, sorted_z, color="navy", alpha=0.6, s=16)
            lims = [
                min(theoretical_quantiles.min(), sorted_z.min()) - 0.2,
                max(theoretical_quantiles.max(), sorted_z.max()) + 0.2,
            ]
            ax2.plot(lims, lims, color="firebrick", lw=1.5, ls="--", label="45° Standard Normal")
            ax2.set_xlim(lims)
            ax2.set_ylim(lims)
            ax2.set_xlabel("Theoretical Normal Quantiles")
            ax2.set_ylabel("Transformed Quantiles Φ^-1(p_t)")
            ax2.set_title("Normal QQ Plot (Berkowitz)")
            ax2.legend(loc="upper left", fontsize=8)
            ax2.grid(True, ls=":", alpha=0.5)

        fig.suptitle(title, fontsize=11, fontweight="semibold")
        fig.tight_layout()
        return fig


def pit_uniformity_test(
    realised: np.ndarray | pd.Series | Sequence[float],
    ensemble_or_pit: np.ndarray | pd.Series | None = None,
    mu: np.ndarray | pd.Series | None = None,
    sigma: np.ndarray | pd.Series | None = None,
    *,
    n_bins: int = 10,
    alpha: float = 0.05,
) -> PITUniformityResult:
    """Compute Probability Integral Transform (PIT) uniformity tests.

    Tests whether density forecasts are well-calibrated and temporally independent
    using the Berkowitz (2001) Likelihood Ratio test and the Kolmogorov-Smirnov test.

    Parameters
    ----------
    realised : array-like
        Observed values y_t.
    ensemble_or_pit : array-like, optional
        Either an ensemble of forecast draws of shape (T, M), or precomputed PIT values of shape (T,).
    mu : array-like, optional
        Forecast mean for Gaussian predictive distribution.
    sigma : array-like, optional
        Forecast standard deviation for Gaussian predictive distribution.
    n_bins : int, default 10
        Number of histogram bins on [0, 1].
    alpha : float, default 0.05
        Nominal significance level.

    Returns
    -------
    PITUniformityResult
        Frozen dataclass with test statistics, p-values, parameters, and plots.
    """
    y = np.asarray(realised, dtype=float).ravel()
    T = len(y)
    if T < 4:
        raise ValueError(f"Need at least 4 observations to run Berkowitz test, got {T}")

    # Determine PIT values
    if mu is not None and sigma is not None:
        mu_arr = np.asarray(mu, dtype=float).ravel()
        sig_arr = np.asarray(sigma, dtype=float).ravel()
        sig_safe = np.maximum(sig_arr, 1e-12)
        pit = norm.cdf((y - mu_arr) / sig_safe)
    elif ensemble_or_pit is not None:
        ens_arr = np.asarray(ensemble_or_pit, dtype=float)
        if ens_arr.ndim == 1:
            # Already PIT values
            pit = ens_arr.copy()
        elif ens_arr.ndim == 2:
            M = ens_arr.shape[1]
            # Hazen plotting position: (rank - 0.5) / M to avoid exact 0 or 1
            pit = np.empty(T)
            for t in range(T):
                count_le = np.sum(ens_arr[t] <= y[t])
                pit[t] = (count_le + 0.5) / (M + 1.0)
        else:
            raise ValueError(f"Invalid ensemble dimension: {ens_arr.shape}")
    else:
        raise ValueError("Must provide either (mu, sigma) or ensemble_or_pit.")

    # Clip to avoid infinite z
    eps = 1e-7
    pit_clipped = np.clip(pit, eps, 1.0 - eps)
    z = norm.ppf(pit_clipped)

    # Berkowitz AR(1) Likelihood Ratio test
    # Null model: z_t ~ i.i.d. N(0, 1) -> mu = 0, sigma = 1, rho = 0
    ll_restricted = -0.5 * T * np.log(2.0 * np.pi) - 0.5 * np.sum(z ** 2)

    # Unrestricted model: (z_t - mu) = rho * (z_{t-1} - mu) + eps_t, eps_t ~ N(0, sigma^2)
    def neg_loglik(params):
        mu_cand, log_sig_cand, rho_cand = params
        sig = np.exp(log_sig_cand)
        rho = np.tanh(rho_cand)
        sig2 = sig ** 2

        # Stationary initial state
        ll_0 = (
            -0.5 * np.log(2.0 * np.pi)
            - 0.5 * np.log(sig2 / max(1e-10, 1.0 - rho ** 2))
            - (z[0] - mu_cand) ** 2 * (1.0 - rho ** 2) / (2.0 * sig2)
        )
        resids = (z[1:] - mu_cand) - rho * (z[:-1] - mu_cand)
        ll_trans = (
            -0.5 * (T - 1) * np.log(2.0 * np.pi)
            - 0.5 * (T - 1) * np.log(sig2)
            - np.sum(resids ** 2) / (2.0 * sig2)
        )
        return -(ll_0 + ll_trans)

    # Initial guess
    mu_init = float(np.mean(z))
    sig_init = float(np.std(z, ddof=1)) if np.std(z, ddof=1) > 1e-4 else 1.0
    if len(z) > 1:
        cov01 = np.cov(z[:-1], z[1:])
        rho_init = float(cov01[0, 1] / max(1e-6, cov01[0, 0])) if cov01[0, 0] > 1e-6 else 0.0
        rho_init = np.clip(rho_init, -0.9, 0.9)
    else:
        rho_init = 0.0

    init_params = [mu_init, np.log(max(1e-4, sig_init)), np.arctanh(rho_init)]
    opt = minimize(neg_loglik, init_params, method="L-BFGS-B")

    mu_hat = float(opt.x[0])
    sig_hat = float(np.exp(opt.x[1]))
    rho_hat = float(np.tanh(opt.x[2]))
    ll_unrestricted = -float(opt.fun)

    lr_stat = max(0.0, 2.0 * (ll_unrestricted - ll_restricted))
    lr_pvalue = float(chi2.sf(lr_stat, df=3))

    # Kolmogorov-Smirnov test against U(0, 1)
    ks_res = kstest(pit_clipped, "uniform", args=(0.0, 1.0))
    ks_stat = float(ks_res.statistic)
    ks_pvalue = float(ks_res.pvalue)

    is_uniform = bool(lr_pvalue > alpha and ks_pvalue > alpha)

    hist_counts, hist_edges = np.histogram(pit_clipped, bins=n_bins, range=(0.0, 1.0))

    return PITUniformityResult(
        pit=pit_clipped,
        z=z,
        lr_stat=lr_stat,
        lr_pvalue=lr_pvalue,
        df=3,
        mu=mu_hat,
        sigma=sig_hat,
        rho=rho_hat,
        ks_stat=ks_stat,
        ks_pvalue=ks_pvalue,
        is_uniform=is_uniform,
        n_obs=T,
        hist_counts=hist_counts,
        hist_edges=hist_edges,
    )


@dataclass(frozen=True)
class FanChartResult:
    """Immutable result object from central bank fan chart projection.

    Attributes
    ----------
    history : pd.Series
        Historical observed series leading up to the forecast origin.
    forecast_mean : pd.Series
        Central projection (mean or median) across forecast horizons.
    intervals : dict[float, tuple[pd.Series, pd.Series]]
        Mapping of confidence levels (e.g. 0.3, 0.6, 0.9) to (lower, upper) series.
    quantiles : pd.DataFrame
        Detailed table of quantiles across forecast horizon.
    levels : tuple[float, ...]
        Nominal coverage levels in ascending order.
    palette : str
        Central bank color palette ('banxico', 'bcb', 'bank_of_england', 'default').
    """

    history: pd.Series
    forecast_mean: pd.Series
    intervals: dict[float, tuple[pd.Series, pd.Series]]
    quantiles: pd.DataFrame
    levels: tuple[float, ...]
    palette: str = "default"

    def summary(self) -> str:
        """Text summary of fan chart forecast intervals."""
        H = len(self.forecast_mean)
        lines = [
            "=" * 72,
            f"Central Bank Fan Chart Projection ({self.palette.upper()} Theme)",
            "=" * 72,
            f"Historical periods           : {len(self.history)}",
            f"Forecast horizon (H)         : {H}",
            f"Confidence levels (%)        : {', '.join(f'{int(round(l*100))}%' for l in self.levels)}",
            "-" * 72,
            f"{'Horizon':<12} {'Mean':>10} " + " ".join(f"[{int(round(l*100))}% CI]" for l in self.levels),
            "-" * 72,
        ]
        for idx_h, h_label in enumerate(self.forecast_mean.index):
            m_val = float(self.forecast_mean.iloc[idx_h])
            row_str = f"{str(h_label):<12s} {m_val:>10.4f}"
            for lvl in self.levels:
                lo_s, hi_s = self.intervals[lvl]
                lo = float(lo_s.iloc[idx_h])
                hi = float(hi_s.iloc[idx_h])
                row_str += f"  [{lo:+.2f}, {hi:+.2f}]"
            lines.append(row_str)
        lines.append("=" * 72)
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Quantiles table across forecast horizons."""
        return self.quantiles.copy().round(4)

    def to_markdown(self, **kwargs: Any) -> str:
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), **kwargs)

    def plot(self, *, ax: Any = None, title: str | None = None) -> Any:
        """Plot central bank fan chart with layered shaded ribbons."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
        else:
            fig = ax.figure

        # Palette colors
        palette_map = {
            "banxico": "#006847",       # Mexican green
            "bcb": "#0b3b60",           # Brazilian navy
            "bank_of_england": "#a00000",# Crimson
            "default": "#1f77b4",       # Deep royal blue
        }
        base_color = palette_map.get(self.palette.lower(), palette_map["default"])

        # Plot history
        h_idx = self.history.index
        ax.plot(h_idx, self.history.values, color="black", lw=1.8, label="Historical Data")

        # Connect history last point to forecast origin
        f_idx = self.forecast_mean.index
        bridge_idx = [h_idx[-1]] + list(f_idx)
        bridge_mean = [self.history.iloc[-1]] + list(self.forecast_mean.values)

        # Plot shaded ribbons in descending level order (outer to inner)
        sorted_levels = sorted(self.levels, reverse=True)
        n_ribbons = len(sorted_levels)
        for i, lvl in enumerate(sorted_levels):
            lo_s, hi_s = self.intervals[lvl]
            bridge_lo = [self.history.iloc[-1]] + list(lo_s.values)
            bridge_hi = [self.history.iloc[-1]] + list(hi_s.values)

            # Opacity increases for inner ribbons
            alpha = 0.15 + 0.55 * ((n_ribbons - i) / n_ribbons)
            ax.fill_between(
                bridge_idx,
                bridge_lo,
                bridge_hi,
                color=base_color,
                alpha=alpha,
                edgecolor="none",
                label=f"{int(round(lvl * 100))}% Confidence",
            )

        # Plot central forecast line
        ax.plot(bridge_idx, bridge_mean, color=base_color, lw=2.2, ls="-", label="Central Projection")

        # Vertical line separating history and forecast
        ax.axvline(h_idx[-1], color="grey", lw=1.0, ls=":")

        default_title = f"Central Bank Fan Chart ({self.palette.title()} Theme)"
        ax.set_title(title or default_title, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Period")
        ax.set_ylabel("Value / Rate")
        ax.legend(loc="best", frameon=True, fontsize=8)
        ax.grid(True, ls=":", alpha=0.5)

        fig.tight_layout()
        return fig


def fan_chart(
    history: pd.Series | np.ndarray | Sequence[float],
    forecast_mean: pd.Series | np.ndarray | Sequence[float],
    forecast_sd: pd.Series | np.ndarray | Sequence[float] | float | None = None,
    ensemble: np.ndarray | None = None,
    *,
    levels: Sequence[float] = (0.3, 0.6, 0.9),
    palette: str = "default",
    dates: pd.DatetimeIndex | Sequence[Any] | None = None,
) -> FanChartResult:
    """Generate institutional central bank fan chart projection.

    Parameters
    ----------
    history : array-like or Series
        Historical observed values leading up to the forecast origin.
    forecast_mean : array-like or Series
        Central trajectory (mean or median) across forecast horizons.
    forecast_sd : array-like, float, or None
        Predictive standard deviation for each forecast period.
    ensemble : ndarray of shape (H, M), optional
        Sample draws per horizon if empirical quantiles are used.
    levels : sequence of float, default (0.3, 0.6, 0.9)
        Central confidence intervals to display (e.g. 0.3 for 30%, 0.6 for 60%, 0.9 for 90%).
    palette : str, default 'default'
        Color theme: 'banxico', 'bcb', 'bank_of_england', 'default'.
    dates : sequence, optional
        Time index for the forecast horizon if not inferred from forecast_mean.

    Returns
    -------
    FanChartResult
        Frozen dataclass with intervals, quantiles, and fan chart plotting methods.
    """
    if isinstance(history, pd.Series):
        s_hist = history.copy()
    else:
        h_arr = np.asarray(history, dtype=float).ravel()
        s_hist = pd.Series(h_arr, index=[f"t-{len(h_arr) - 1 - i}" for i in range(len(h_arr))])

    if isinstance(forecast_mean, pd.Series):
        s_mean = forecast_mean.copy()
    else:
        m_arr = np.asarray(forecast_mean, dtype=float).ravel()
        if dates is not None:
            s_mean = pd.Series(m_arr, index=dates)
        else:
            s_mean = pd.Series(m_arr, index=[f"t+{h + 1}" for h in range(len(m_arr))])

    H = len(s_mean)
    sorted_levels = tuple(sorted(levels))
    intervals: dict[float, tuple[pd.Series, pd.Series]] = {}
    quant_dict: dict[str, pd.Series] = {"mean": s_mean}

    if ensemble is not None:
        ens = np.asarray(ensemble, dtype=float)
        if ens.shape[0] != H:
            raise ValueError(f"Ensemble row count {ens.shape[0]} does not match horizon {H}")
        for lvl in sorted_levels:
            alpha = (1.0 - lvl) / 2.0
            q_lo = np.quantile(ens, alpha, axis=1)
            q_hi = np.quantile(ens, 1.0 - alpha, axis=1)
            s_lo = pd.Series(q_lo, index=s_mean.index)
            s_hi = pd.Series(q_hi, index=s_mean.index)
            intervals[lvl] = (s_lo, s_hi)
            pct = int(round(lvl * 100))
            quant_dict[f"q_{int(round(alpha*100))}"] = s_lo
            quant_dict[f"q_{int(round((1-alpha)*100))}"] = s_hi
    elif forecast_sd is not None:
        if isinstance(forecast_sd, (int, float)):
            sd_arr = np.full(H, float(forecast_sd))
        else:
            sd_arr = np.asarray(forecast_sd, dtype=float).ravel()
        if len(sd_arr) != H:
            raise ValueError(f"forecast_sd length {len(sd_arr)} does not match horizon {H}")

        for lvl in sorted_levels:
            alpha = (1.0 - lvl) / 2.0
            z_crit = norm.ppf(1.0 - alpha)
            lo = s_mean.values - z_crit * sd_arr
            hi = s_mean.values + z_crit * sd_arr
            s_lo = pd.Series(lo, index=s_mean.index)
            s_hi = pd.Series(hi, index=s_mean.index)
            intervals[lvl] = (s_lo, s_hi)
            pct = int(round(lvl * 100))
            quant_dict[f"lower_{pct}%"] = s_lo
            quant_dict[f"upper_{pct}%"] = s_hi
    else:
        raise ValueError("Must supply either forecast_sd or ensemble to compute fan chart intervals.")

    df_quantiles = pd.DataFrame(quant_dict, index=s_mean.index)

    return FanChartResult(
        history=s_hist,
        forecast_mean=s_mean,
        intervals=intervals,
        quantiles=df_quantiles,
        levels=sorted_levels,
        palette=palette,
    )


__all__ = [
    "PITUniformityResult",
    "pit_uniformity_test",
    "FanChartResult",
    "fan_chart",
    "crps_gaussian",
    "crps_ensemble",
    "log_score_gaussian",
    "brier_score",
    "pit_histogram",
]
