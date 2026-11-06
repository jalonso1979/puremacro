"""Frozen result containers for ``puremacro.portfolio``.

Implements the Result-Object Presentation Standard 2.0+ across all five
portfolio analysis result classes: ``summary()``, ``to_dataframe()``,
``to_frame()``, ``to_markdown()``, ``to_latex()``, ``to_typst()``, and
``plot(ax=None, show=False, **kwargs)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst

__all__ = [
    "AllocationResult",
    "MacroRegimePortfolioResult",
    "FactorAttributionResult",
    "PortfolioBacktestResult",
    "WealthSimulationResult",
]


def _fmt_val(v: Any, digits: int = 6) -> Any:
    """Format scalar values cleanly for summary DataFrames."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), digits)
    return v


@dataclass(frozen=True)
class AllocationResult:
    """Result container for portfolio optimization and allocation routines.

    Attributes
    ----------
    weights : np.ndarray
        Optimal portfolio weights of shape ``(n_assets,)``, summing to 1.0.
    asset_names : tuple[str, ...]
        Asset identifiers of length ``n_assets``.
    method : str
        Optimization or allocation method name.
    expected_return : float
        Annualized expected portfolio return.
    volatility : float
        Annualized portfolio volatility (standard deviation).
    sharpe_ratio : float
        Annualized Sharpe ratio ``(expected_return - rf) / volatility``.
    cvar_95 : float
        95% Conditional Value-at-Risk (Expected Shortfall, positive loss convention).
    diversification_ratio : float
        Choueifaty-Coignard diversification ratio ``(w' sigma) / sqrt(w' Sigma w)``.
    effective_n_assets : float
        Inverse Herfindahl-Hirschman effective number of constituents ``1 / sum(w_i^2)``.
    risk_contributions : np.ndarray
        Percentage share of portfolio volatility contributed by each asset, summing to 1.0.
    frontier_returns : np.ndarray | None
        Optional efficient frontier target returns of shape ``(n_points,)``.
    frontier_vols : np.ndarray | None
        Optional efficient frontier volatilities of shape ``(n_points,)``.
    frontier_weights : np.ndarray | None
        Optional efficient frontier weights of shape ``(n_points, n_assets)``.
    metadata : dict[str, Any]
        Additional diagnostic metadata (e.g., risk-free rate, shrinkage, BL posteriors).
    """

    weights: np.ndarray
    asset_names: tuple[str, ...]
    method: str
    expected_return: float
    volatility: float
    sharpe_ratio: float
    cvar_95: float
    diversification_ratio: float
    effective_n_assets: float
    risk_contributions: np.ndarray
    frontier_returns: np.ndarray | None = None
    frontier_vols: np.ndarray | None = None
    frontier_weights: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Return a structured summary DataFrame of portfolio metrics and weights."""
        rows: list[tuple[str, Any]] = [
            ("method", self.method),
            ("n_assets", len(self.asset_names)),
            ("expected_return", self.expected_return),
            ("volatility", self.volatility),
            ("sharpe_ratio", self.sharpe_ratio),
            ("cvar_95", self.cvar_95),
            ("diversification_ratio", self.diversification_ratio),
            ("effective_n_assets", self.effective_n_assets),
        ]
        for name, w, rc in zip(self.asset_names, self.weights, self.risk_contributions):
            rows.append((f"weight_{name}", float(w)))
            rows.append((f"risk_contrib_{name}", float(rc)))
        return pd.DataFrame(
            {"metric": [r[0] for r in rows], "value": [_fmt_val(r[1]) for r in rows]}
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return asset-level allocation weights and risk contributions as a DataFrame."""
        return pd.DataFrame(
            {
                "asset": list(self.asset_names),
                "weight": np.asarray(self.weights, dtype=float),
                "risk_contribution": np.asarray(self.risk_contributions, dtype=float),
            }
        )

    def to_frame(self) -> pd.DataFrame:
        """Alias for :meth:`to_dataframe`."""
        return self.to_dataframe()

    def to_markdown(self, digits: int = 4) -> str:
        """Render asset allocation table as GitHub-flavored Markdown."""
        return df_to_markdown(self.to_dataframe(), index=False, digits=digits)

    def to_latex(self, digits: int = 4) -> str:
        """Render asset allocation table as a LaTeX tabular."""
        return df_to_latex(self.to_dataframe(), index=False, digits=digits)

    def to_typst(self, digits: int = 4) -> str:
        """Render asset allocation table as a Typst ``#table``."""
        return df_to_typst(self.to_dataframe(), index=False, digits=digits)

    def plot(
        self,
        ax: Any = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Plot portfolio weights vs. risk contributions (and efficient frontier if present)."""
        import matplotlib.pyplot as plt

        has_frontier = (
            self.frontier_returns is not None and self.frontier_vols is not None
        )
        x = np.arange(len(self.asset_names))
        width = 0.36

        if ax is not None:
            fig = ax.figure
            ax.bar(x - width / 2, self.weights, width=width, label="Weight", color="#1f77b4")
            ax.bar(
                x + width / 2,
                self.risk_contributions,
                width=width,
                label="Risk Contribution",
                color="#ff7f0e",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(self.asset_names)
            ax.set_ylabel("Share")
            ax.set_title(kwargs.get("title", f"Portfolio Allocation ({self.method})"))
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.3)
            if show:
                plt.show()
            return fig

        if has_frontier:
            figsize = kwargs.get("figsize", (10.5, 4.2))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            ax1.plot(
                self.frontier_vols,
                self.frontier_returns,
                color="#1f77b4",
                lw=2.0,
                label="Efficient Frontier",
            )
            ax1.scatter(
                [self.volatility],
                [self.expected_return],
                color="#d62728",
                s=70,
                zorder=5,
                label=f"Optimal ({self.method})",
            )
            ax1.set_xlabel("Annualized Volatility")
            ax1.set_ylabel("Annualized Expected Return")
            ax1.set_title("Mean-Variance Efficient Frontier")
            ax1.legend(frameon=False)
            ax1.grid(True, alpha=0.3)
            bar_ax = ax2
        else:
            figsize = kwargs.get("figsize", (7.5, 4.2))
            fig, bar_ax = plt.subplots(figsize=figsize)

        bar_ax.bar(
            x - width / 2, self.weights, width=width, label="Weight", color="#1f77b4"
        )
        bar_ax.bar(
            x + width / 2,
            self.risk_contributions,
            width=width,
            label="Risk Contribution",
            color="#ff7f0e",
        )
        bar_ax.set_xticks(x)
        bar_ax.set_xticklabels(self.asset_names)
        bar_ax.set_ylabel("Share")
        bar_ax.set_title(kwargs.get("title", f"Weights vs. Risk Contributions ({self.method})"))
        bar_ax.legend(frameon=False)
        bar_ax.grid(True, alpha=0.3)

        fig.tight_layout()
        if show:
            plt.show()
        return fig


@dataclass(frozen=True)
class MacroRegimePortfolioResult:
    """Result container for macro-quadrant All-Weather portfolio construction.

    Attributes
    ----------
    weights : np.ndarray
        Combined All-Weather portfolio weights of shape ``(n_assets,)``.
    asset_names : tuple[str, ...]
        Asset identifiers of length ``n_assets``.
    quadrant_weights : dict[str, np.ndarray]
        Sub-portfolio weights for each of the four macro quadrants.
    quadrant_returns : pd.DataFrame
        Regime-conditional annualized asset returns across quadrants.
    quadrant_vols : pd.DataFrame
        Regime-conditional annualized asset volatilities across quadrants.
    regime_labels : pd.Series | np.ndarray
        Historical macro quadrant classification for each sample period.
    regime_shares : dict[str, float]
        Empirical frequency of each macro quadrant in the sample.
    expected_return : float
        Annualized expected return of the All-Weather portfolio.
    volatility : float
        Annualized volatility of the All-Weather portfolio.
    sharpe_ratio : float
        Annualized Sharpe ratio of the All-Weather portfolio.
    metadata : dict[str, Any]
        Additional diagnostic metadata.
    """

    weights: np.ndarray
    asset_names: tuple[str, ...]
    quadrant_weights: dict[str, np.ndarray]
    quadrant_returns: pd.DataFrame
    quadrant_vols: pd.DataFrame
    regime_labels: pd.Series | np.ndarray
    regime_shares: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Return a structured summary DataFrame of All-Weather metrics and regime shares."""
        rows: list[tuple[str, Any]] = [
            ("expected_return", self.expected_return),
            ("volatility", self.volatility),
            ("sharpe_ratio", self.sharpe_ratio),
        ]
        for q, share in self.regime_shares.items():
            rows.append((f"regime_share_{q}", float(share)))
        for name, w in zip(self.asset_names, self.weights):
            rows.append((f"weight_{name}", float(w)))
        return pd.DataFrame(
            {"metric": [r[0] for r in rows], "value": [_fmt_val(r[1]) for r in rows]}
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return All-Weather and quadrant-specific weights by asset as a DataFrame."""
        data: dict[str, Any] = {
            "asset": list(self.asset_names),
            "all_weather_weight": np.asarray(self.weights, dtype=float),
        }
        for q, qw in self.quadrant_weights.items():
            data[f"{q}_weight"] = np.asarray(qw, dtype=float)
        return pd.DataFrame(data)

    def to_frame(self) -> pd.DataFrame:
        """Alias for :meth:`to_dataframe`."""
        return self.to_dataframe()

    def to_markdown(self, digits: int = 4) -> str:
        """Render All-Weather weight table as GitHub-flavored Markdown."""
        return df_to_markdown(self.to_dataframe(), index=False, digits=digits)

    def to_latex(self, digits: int = 4) -> str:
        """Render All-Weather weight table as a LaTeX tabular."""
        return df_to_latex(self.to_dataframe(), index=False, digits=digits)

    def to_typst(self, digits: int = 4) -> str:
        """Render All-Weather weight table as a Typst ``#table``."""
        return df_to_typst(self.to_dataframe(), index=False, digits=digits)

    def plot(
        self,
        ax: Any = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Plot All-Weather and macro-quadrant portfolio weights."""
        import matplotlib.pyplot as plt

        return_new = ax is None
        if ax is None:
            figsize = kwargs.get("figsize", (8.5, 4.5))
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        x = np.arange(len(self.asset_names))
        quads = list(self.quadrant_weights.keys())
        n_bars = len(quads) + 1
        width = 0.8 / max(n_bars, 1)

        ax.bar(
            x - 0.4 + 0.5 * width,
            self.weights,
            width=width,
            label="All-Weather",
            color="#111111",
        )
        palette = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]
        for i, q in enumerate(quads):
            ax.bar(
                x - 0.4 + (i + 1.5) * width,
                self.quadrant_weights[q],
                width=width,
                label=q.replace("_", " ").title(),
                color=palette[i % len(palette)],
                alpha=0.85,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(self.asset_names)
        ax.set_ylabel("Portfolio Weight")
        ax.set_title(kwargs.get("title", "All-Weather & Macro Quadrant Allocations"))
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

        if return_new:
            fig.tight_layout()
        if show:
            plt.show()
        return fig


@dataclass(frozen=True)
class FactorAttributionResult:
    """Result container for multi-factor return attribution with Newey-West HAC SEs.

    Attributes
    ----------
    alpha : float
        Annualized Jensen's alpha.
    alpha_se : float
        Annualized Newey-West HAC standard error of alpha.
    alpha_tstat : float
        t-statistic for alpha.
    alpha_pvalue : float
        Two-sided p-value for alpha.
    betas : np.ndarray
        Estimated factor loadings of shape ``(n_factors,)``.
    beta_ses : np.ndarray
        Newey-West HAC standard errors of factor loadings.
    beta_tstats : np.ndarray
        t-statistics of factor loadings.
    beta_pvalues : np.ndarray
        Two-sided p-values of factor loadings.
    factor_names : tuple[str, ...]
        Factor names of length ``n_factors``.
    r_squared : float
        Regression R-squared.
    adj_r_squared : float
        Adjusted R-squared.
    tracking_error : float
        Annualized idiosyncratic residual volatility.
    information_ratio : float
        Annualized Information Ratio ``alpha / tracking_error``.
    return_contributions : dict[str, float]
        Annualized return contribution from ``"alpha"`` and each factor.
    rolling_betas : pd.DataFrame | None
        Optional DataFrame of rolling window alpha and factor betas.
    metadata : dict[str, Any]
        Additional diagnostic metadata.
    """

    alpha: float
    alpha_se: float
    alpha_tstat: float
    alpha_pvalue: float
    betas: np.ndarray
    beta_ses: np.ndarray
    beta_tstats: np.ndarray
    beta_pvalues: np.ndarray
    factor_names: tuple[str, ...]
    r_squared: float
    adj_r_squared: float
    tracking_error: float
    information_ratio: float
    return_contributions: dict[str, float]
    rolling_betas: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the factor regression table with estimates, HAC SEs, and attribution."""
        terms = ["alpha", *self.factor_names]
        estimates = [float(self.alpha), *[float(b) for b in self.betas]]
        ses = [float(self.alpha_se), *[float(s) for s in self.beta_ses]]
        tstats = [float(self.alpha_tstat), *[float(t) for t in self.beta_tstats]]
        pvals = [float(self.alpha_pvalue), *[float(p) for p in self.beta_pvalues]]
        contribs = [float(self.return_contributions.get(t, 0.0)) for t in terms]
        return pd.DataFrame(
            {
                "term": terms,
                "estimate": estimates,
                "std_error": ses,
                "t_stat": tstats,
                "p_value": pvals,
                "return_contribution": contribs,
            }
        )

    def summary(self) -> pd.DataFrame:
        """Return a structured summary DataFrame of factor attribution and fit metrics."""
        rows: list[tuple[str, Any]] = [
            ("alpha", self.alpha),
            ("alpha_se", self.alpha_se),
            ("alpha_tstat", self.alpha_tstat),
            ("alpha_pvalue", self.alpha_pvalue),
            ("r_squared", self.r_squared),
            ("adj_r_squared", self.adj_r_squared),
            ("tracking_error", self.tracking_error),
            ("information_ratio", self.information_ratio),
        ]
        for fname, b, se, tval, pval in zip(
            self.factor_names,
            self.betas,
            self.beta_ses,
            self.beta_tstats,
            self.beta_pvalues,
        ):
            rows.append((f"beta_{fname}", float(b)))
            rows.append((f"se_{fname}", float(se)))
            rows.append((f"tstat_{fname}", float(tval)))
            rows.append((f"pvalue_{fname}", float(pval)))
        return pd.DataFrame(
            {"metric": [r[0] for r in rows], "value": [_fmt_val(r[1]) for r in rows]}
        )

    def to_frame(self) -> pd.DataFrame:
        """Alias for :meth:`to_dataframe`."""
        return self.to_dataframe()

    def to_markdown(self, digits: int = 4) -> str:
        """Render factor attribution table as GitHub-flavored Markdown."""
        return df_to_markdown(self.to_dataframe(), index=False, digits=digits)

    def to_latex(self, digits: int = 4) -> str:
        """Render factor attribution table as a LaTeX tabular."""
        return df_to_latex(self.to_dataframe(), index=False, digits=digits)

    def to_typst(self, digits: int = 4) -> str:
        """Render factor attribution table as a Typst ``#table``."""
        return df_to_typst(self.to_dataframe(), index=False, digits=digits)

    def plot(
        self,
        ax: Any = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Plot factor loadings with 95% HAC confidence intervals."""
        import matplotlib.pyplot as plt

        return_new = ax is None
        if ax is None:
            figsize = kwargs.get("figsize", (7.5, 4.2))
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        x = np.arange(len(self.factor_names))
        ci_95 = 1.96 * np.asarray(self.beta_ses, dtype=float)
        ax.bar(
            x,
            self.betas,
            yerr=ci_95,
            capsize=4,
            color="#1f77b4",
            alpha=0.85,
            label="Factor Beta (±1.96 HAC SE)",
        )
        ax.axhline(0.0, color="black", lw=1.0, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(self.factor_names)
        ax.set_ylabel("Factor Loading (Beta)")
        ax.set_title(
            kwargs.get(
                "title",
                f"Factor Attribution (α={self.alpha:.4f}, R²={self.r_squared:.3f})",
            )
        )
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

        if return_new:
            fig.tight_layout()
        if show:
            plt.show()
        return fig


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """Result container for portfolio rebalancing and transaction-cost backtests.

    Attributes
    ----------
    equity_curve : pd.Series | np.ndarray
        Cumulative net portfolio wealth index starting from ``1.0``.
    portfolio_returns : pd.Series | np.ndarray
        Net periodic portfolio returns after transaction costs.
    drawdown_series : pd.Series | np.ndarray
        Underwater percentage drawdown series (``<= 0.0``).
    weight_history : np.ndarray
        Beginning-of-period asset weights of shape ``(T, n_assets)``.
    turnover_series : np.ndarray
        One-way/two-way rebalancing turnover ``sum_i |w_{i,t}^+ - w_{i,t}^-|`` at each period.
    asset_names : tuple[str, ...]
        Asset identifiers of length ``n_assets``.
    cagr : float
        Compound Annual Growth Rate.
    annualized_return : float
        Arithmetic annualized net return.
    annualized_vol : float
        Annualized net return volatility.
    sharpe_ratio : float
        Annualized Sharpe ratio.
    sortino_ratio : float
        Annualized Sortino ratio using downside deviation below ``rf``.
    calmar_ratio : float
        Calmar ratio ``cagr / max_drawdown``.
    omega_ratio : float
        Omega gain-to-loss ratio above periodic risk-free threshold.
    max_drawdown : float
        Maximum peak-to-trough drawdown as a positive fraction in ``[0, 1]``.
    max_drawdown_duration : int
        Maximum consecutive periods spent below a prior high-water mark.
    var_95 : float
        95% periodic Value-at-Risk (positive loss convention).
    cvar_95 : float
        95% periodic Conditional Value-at-Risk (positive loss convention).
    total_tc_paid : float
        Cumulative transaction cost drag paid over the backtest.
    benchmark_equity : pd.Series | np.ndarray | None
        Optional cumulative wealth index for a benchmark series.
    metadata : dict[str, Any]
        Additional diagnostic metadata (e.g., rebalance policy, tc_bps).
    """

    equity_curve: pd.Series | np.ndarray
    portfolio_returns: pd.Series | np.ndarray
    drawdown_series: pd.Series | np.ndarray
    weight_history: np.ndarray
    turnover_series: np.ndarray
    asset_names: tuple[str, ...]
    cagr: float
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    total_tc_paid: float
    benchmark_equity: pd.Series | np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Return a structured summary DataFrame of backtest performance metrics."""
        rows: list[tuple[str, Any]] = [
            ("cagr", self.cagr),
            ("annualized_return", self.annualized_return),
            ("annualized_vol", self.annualized_vol),
            ("sharpe_ratio", self.sharpe_ratio),
            ("sortino_ratio", self.sortino_ratio),
            ("calmar_ratio", self.calmar_ratio),
            ("omega_ratio", self.omega_ratio),
            ("max_drawdown", self.max_drawdown),
            ("max_drawdown_duration", self.max_drawdown_duration),
            ("var_95", self.var_95),
            ("cvar_95", self.cvar_95),
            ("total_tc_paid", self.total_tc_paid),
            ("mean_turnover", float(np.mean(self.turnover_series))),
        ]
        return pd.DataFrame(
            {"metric": [r[0] for r in rows], "value": [_fmt_val(r[1]) for r in rows]}
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return period-by-period equity, return, drawdown, turnover, and weights."""
        data: dict[str, Any] = {
            "equity": np.asarray(self.equity_curve, dtype=float),
            "return": np.asarray(self.portfolio_returns, dtype=float),
            "drawdown": np.asarray(self.drawdown_series, dtype=float),
            "turnover": np.asarray(self.turnover_series, dtype=float),
        }
        if self.benchmark_equity is not None:
            data["benchmark_equity"] = np.asarray(self.benchmark_equity, dtype=float)
        for i, name in enumerate(self.asset_names):
            data[f"weight_{name}"] = np.asarray(self.weight_history[:, i], dtype=float)

        df = pd.DataFrame(data)
        if isinstance(self.equity_curve, pd.Series):
            df.index = self.equity_curve.index
        return df

    def to_frame(self) -> pd.DataFrame:
        """Alias for :meth:`to_dataframe`."""
        return self.to_dataframe()

    def to_markdown(self, digits: int = 4) -> str:
        """Render backtest summary metrics as GitHub-flavored Markdown."""
        return df_to_markdown(self.summary(), index=False, digits=digits)

    def to_latex(self, digits: int = 4) -> str:
        """Render backtest summary metrics as a LaTeX tabular."""
        return df_to_latex(self.summary(), index=False, digits=digits)

    def to_typst(self, digits: int = 4) -> str:
        """Render backtest summary metrics as a Typst ``#table``."""
        return df_to_typst(self.summary(), index=False, digits=digits)

    def plot(
        self,
        ax: Any = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Render a 3-panel backtest tear-sheet (equity, underwater drawdown, weights) or plot onto ``ax``."""
        import matplotlib.pyplot as plt

        t_idx = (
            self.equity_curve.index
            if isinstance(self.equity_curve, pd.Series)
            else np.arange(len(self.equity_curve))
        )
        eq_vals = np.asarray(self.equity_curve, dtype=float)
        dd_vals = np.asarray(self.drawdown_series, dtype=float)

        if ax is not None:
            fig = ax.figure
            ax.plot(t_idx, eq_vals, color="#1f77b4", lw=1.8, label="Portfolio Equity")
            if self.benchmark_equity is not None:
                ax.plot(
                    t_idx,
                    np.asarray(self.benchmark_equity, dtype=float),
                    color="#7f7f7f",
                    lw=1.4,
                    ls="--",
                    label="Benchmark",
                )
            ax.set_ylabel("Cumulative Wealth")
            ax.set_title(kwargs.get("title", "Portfolio Backtest Equity Curve"))
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.3)
            if show:
                plt.show()
            return fig

        figsize = kwargs.get("figsize", (10.0, 8.0))
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)

        ax1.plot(t_idx, eq_vals, color="#1f77b4", lw=1.8, label="Portfolio Equity")
        if self.benchmark_equity is not None:
            ax1.plot(
                t_idx,
                np.asarray(self.benchmark_equity, dtype=float),
                color="#7f7f7f",
                lw=1.4,
                ls="--",
                label="Benchmark",
            )
        ax1.set_ylabel("Wealth Index")
        ax1.set_title(
            kwargs.get(
                "title",
                f"Portfolio Tear-Sheet (CAGR={self.cagr:.2%}, Sharpe={self.sharpe_ratio:.2f}, MaxDD={self.max_drawdown:.2%})",
            )
        )
        ax1.legend(loc="upper left", frameon=False)
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(t_idx, dd_vals, 0.0, color="#d62728", alpha=0.35)
        ax2.plot(t_idx, dd_vals, color="#d62728", lw=1.2, label="Drawdown")
        ax2.set_ylabel("Drawdown")
        ax2.set_title("Underwater Drawdown")
        ax2.grid(True, alpha=0.3)

        ax3.stackplot(
            np.arange(len(eq_vals)) if not isinstance(t_idx, np.ndarray) else t_idx,
            self.weight_history.T,
            labels=list(self.asset_names),
            alpha=0.80,
        )
        ax3.set_ylabel("Weight")
        ax3.set_xlabel("Period")
        ax3.set_title("Asset Allocation Drift & Rebalancing")
        ax3.legend(loc="upper right", frameon=False, ncol=min(len(self.asset_names), 4))
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        if show:
            plt.show()
        return fig


@dataclass(frozen=True)
class WealthSimulationResult:
    """Result container for Monte Carlo accumulation and retirement decumulation glidepaths.

    Attributes
    ----------
    wealth_paths : np.ndarray
        Simulated wealth trajectories of shape ``(n_sims, horizon_years + 1)``.
    quantiles : dict[int, np.ndarray]
        Percentile trajectories for keys ``{5, 10, 25, 50, 75, 90, 95}``.
    initial_wealth : float
        Starting portfolio wealth at year 0.
    horizon_years : int
        Simulation horizon in years.
    annual_contribution : float
        Annual real contribution added at the end of each year.
    annual_withdrawal : float
        Annual real withdrawal subtracted at the end of each year.
    prob_ruin : float
        Probability that portfolio wealth hits zero at any point within ``horizon_years``.
    prob_double : float
        Probability that terminal wealth is at least ``2 * initial_wealth``.
    median_terminal_wealth : float
        Median wealth at ``horizon_years``.
    cvar_5_terminal_wealth : float
        Mean terminal wealth in the worst 5% of scenarios.
    safe_withdrawal_rate_95 : float
        Maximum initial withdrawal rate (as a fraction of ``initial_wealth``) with <= 5% ruin probability.
    metadata : dict[str, Any]
        Additional simulation metadata.
    """

    wealth_paths: np.ndarray
    quantiles: dict[int, np.ndarray]
    initial_wealth: float
    horizon_years: int
    annual_contribution: float
    annual_withdrawal: float
    prob_ruin: float
    prob_double: float
    median_terminal_wealth: float
    cvar_5_terminal_wealth: float
    safe_withdrawal_rate_95: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Return a structured summary DataFrame of Monte Carlo wealth metrics."""
        rows: list[tuple[str, Any]] = [
            ("initial_wealth", self.initial_wealth),
            ("horizon_years", self.horizon_years),
            ("annual_contribution", self.annual_contribution),
            ("annual_withdrawal", self.annual_withdrawal),
            ("prob_ruin", self.prob_ruin),
            ("prob_double", self.prob_double),
            ("median_terminal_wealth", self.median_terminal_wealth),
            ("cvar_5_terminal_wealth", self.cvar_5_terminal_wealth),
            ("safe_withdrawal_rate_95", self.safe_withdrawal_rate_95),
            ("n_sims", int(self.wealth_paths.shape[0])),
        ]
        return pd.DataFrame(
            {"metric": [r[0] for r in rows], "value": [_fmt_val(r[1]) for r in rows]}
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the year-by-year wealth percentile fan chart as a DataFrame."""
        data: dict[str, Any] = {"year": np.arange(self.horizon_years + 1)}
        for q in sorted(self.quantiles.keys()):
            data[f"q{q:02d}"] = np.asarray(self.quantiles[q], dtype=float)
        return pd.DataFrame(data)

    def to_frame(self) -> pd.DataFrame:
        """Alias for :meth:`to_dataframe`."""
        return self.to_dataframe()

    def to_markdown(self, digits: int = 2) -> str:
        """Render wealth simulation summary as GitHub-flavored Markdown."""
        return df_to_markdown(self.summary(), index=False, digits=digits)

    def to_latex(self, digits: int = 2) -> str:
        """Render wealth simulation summary as a LaTeX tabular."""
        return df_to_latex(self.summary(), index=False, digits=digits)

    def to_typst(self, digits: int = 2) -> str:
        """Render wealth simulation summary as a Typst ``#table``."""
        return df_to_typst(self.summary(), index=False, digits=digits)

    def plot(
        self,
        ax: Any = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Plot Monte Carlo wealth fan chart across horizons."""
        import matplotlib.pyplot as plt

        return_new = ax is None
        if ax is None:
            figsize = kwargs.get("figsize", (8.5, 4.5))
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        years = np.arange(self.horizon_years + 1)
        if 5 in self.quantiles and 95 in self.quantiles:
            ax.fill_between(
                years,
                self.quantiles[5],
                self.quantiles[95],
                color="#1f77b4",
                alpha=0.14,
                label="5%–95% Band",
            )
        if 10 in self.quantiles and 90 in self.quantiles:
            ax.fill_between(
                years,
                self.quantiles[10],
                self.quantiles[90],
                color="#1f77b4",
                alpha=0.22,
                label="10%–90% Band",
            )
        if 25 in self.quantiles and 75 in self.quantiles:
            ax.fill_between(
                years,
                self.quantiles[25],
                self.quantiles[75],
                color="#1f77b4",
                alpha=0.35,
                label="25%–75% Band",
            )
        if 50 in self.quantiles:
            ax.plot(
                years,
                self.quantiles[50],
                color="#08306b",
                lw=2.0,
                label="Median Wealth",
            )
        ax.axhline(
            self.initial_wealth,
            color="#333333",
            lw=1.2,
            ls="--",
            label="Initial Wealth",
        )
        ax.set_xlabel("Horizon (Years)")
        ax.set_ylabel("Portfolio Wealth")
        ax.set_title(
            kwargs.get(
                "title",
                f"Wealth Glidepath Fan Chart (Ruin Prob={self.prob_ruin:.1%}, 95% SWR={self.safe_withdrawal_rate_95:.2%})",
            )
        )
        ax.legend(loc="upper left", frameon=False)
        ax.grid(True, alpha=0.3)

        if return_new:
            fig.tight_layout()
        if show:
            plt.show()
        return fig
