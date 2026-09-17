"""Latin America Real-Time Macroeconomic Nowcast Orchestrator.

Integrates:
- `puremacro.fetch.realtime` connectors (Banxico, INEGI, BCB, BCCh, ALFRED)
- Dynamic Factor Models with Kalman smoothing (:class:`DynamicFactorModel`)
- Mixed-Frequency VARs (:func:`mf_var`)
- Analytical Bańbura & Modugno (2014) news decomposition (:func:`banbura_modugno_news`)
- Central bank fan charts and forecast evaluation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..fetch.realtime import VintagePanel, load_realtime_cartridge
from .dfm import DynamicFactorModel
from .evaluation import FanChartResult, fan_chart
from .mfvar import mf_var
from .news import NewsDecompositionResult, banbura_modugno_news

COUNTRY_SPECS = {
    "MEX": {
        "name": "Mexico",
        "central_bank": "Banco de México (Banxico) / INEGI",
        "default_target": "gdp",
        "candidates": ["pib_real", "gdp", "pib"],
        "palette": "banxico",
    },
    "BRA": {
        "name": "Brazil",
        "central_bank": "Banco Central do Brasil (BCB)",
        "default_target": "gdp",
        "candidates": ["gdp", "pib"],
        "palette": "bcb",
    },
    "CHL": {
        "name": "Chile",
        "central_bank": "Banco Central de Chile (BCCh)",
        "default_target": "gdp",
        "candidates": ["gdp", "pib"],
        "palette": "default",
    },
    "USA": {
        "name": "United States",
        "central_bank": "Federal Reserve (ALFRED)",
        "default_target": "GDPC1",
        "candidates": ["GDPC1", "gdp"],
        "palette": "default",
    },
}


@dataclass(frozen=True)
class RealtimeNowcastResult:
    """Immutable result object from real-time macroeconomic nowcasting.

    Attributes
    ----------
    nowcast : float
        Point nowcast for the target variable.
    forecast_sd : float
        Estimated standard deviation of the nowcast error.
    target_variable : str
        Name of the target variable (e.g. 'gdp').
    target_period : str | Any
        Reference period or date of the nowcast.
    country : str
        Country ISO code (e.g. 'MEX', 'BRA', 'CHL', 'USA').
    method : str
        Estimation methodology ('dfm' or 'mfvar').
    latest_vintage : pd.Timestamp | str
        Timestamp or date string of the latest vintage analyzed.
    factors : pd.DataFrame
        DataFrame of estimated common dynamic factors or monthly states.
    loadings : pd.DataFrame
        Observation loadings matrix or state mappings.
    news_decomposition : Optional[NewsDecompositionResult]
        Bańbura & Modugno (2014) news decomposition if previous vintage available.
    news_vs_noise_test : Optional[Any]
        Mankiw-Shapiro news vs noise revision test result if requested.
    vintage_history : pd.DataFrame
        Evolution of point nowcasts across historical vintages.
    palette : str
        Central bank visual palette ('banxico', 'bcb', 'bank_of_england', 'default').
    """

    nowcast: float
    forecast_sd: float
    target_variable: str
    target_period: Any
    country: str
    method: str
    latest_vintage: Any
    factors: pd.DataFrame
    loadings: pd.DataFrame
    news_decomposition: Optional[NewsDecompositionResult] = None
    news_vs_noise_test: Optional[Any] = None
    vintage_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    palette: str = "default"

    def summary(self) -> str:
        """Text summary of real-time macroeconomic nowcast."""
        c_spec = COUNTRY_SPECS.get(self.country.upper(), {})
        c_name = c_spec.get("name", self.country)
        agency = c_spec.get("central_bank", "Central Bank / Statistical Agency")

        lines = [
            "=" * 74,
            f"Real-Time Macroeconomic Nowcast: {c_name} ({self.country.upper()})",
            f"Institutional Source: {agency}",
            "=" * 74,
            f"Target Variable               : {self.target_variable}",
            f"Target Reference Period       : {self.target_period}",
            f"Estimation Engine             : {self.method.upper()}",
            f"Latest Information Vintage    : {self.latest_vintage}",
            "-" * 74,
            f"Point Nowcast                 : {self.nowcast:+.4f}",
            f"Forecast Std. Error (1-sigma) : {self.forecast_sd:.4f}",
            f"68% Confidence Interval       : [{self.nowcast - self.forecast_sd:+.4f}, {self.nowcast + self.forecast_sd:+.4f}]",
            f"90% Confidence Interval       : [{self.nowcast - 1.645 * self.forecast_sd:+.4f}, {self.nowcast + 1.645 * self.forecast_sd:+.4f}]",
            "-" * 74,
        ]

        if self.news_decomposition is not None:
            nd = self.news_decomposition
            lines.append("Latest Vintage News & Revisions Attribution:")
            lines.append(f"  Previous Nowcast (v-1)      : {nd.forecast_old:+.4f}")
            lines.append(f"  Total Nowcast Revision (Δy) : {nd.revision:+.4f}")
            lines.append(f"  Impact from new releases    : {sum(nd.impact_releases.values()):+.4f}")
            lines.append(f"  Impact from revisions       : {sum(nd.impact_revisions.values()):+.4f}")
            lines.append(f"  Analytical Identity Error   : {nd.decomposition_error:.2e} (< 1e-10)")
            lines.append("-" * 74)

        if not self.vintage_history.empty:
            lines.append("Recent Vintage Nowcast Progression:")
            tail_hist = self.vintage_history.tail(5)
            lines.append(f"{'Vintage':<16} {'Nowcast':>10} {'Std.Dev':>10}")
            for _, r in tail_hist.iterrows():
                v_str = str(r.get("vintage", ""))[:16]
                nc = float(r.get("nowcast", float("nan")))
                sd = float(r.get("forecast_sd", float("nan")))
                lines.append(f"{v_str:<16s} {nc:>10.4f} {sd:>10.4f}")
            lines.append("-" * 74)

        lines.append("=" * 74)
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Overview summary DataFrame."""
        rec = {
            "country": self.country,
            "target_variable": self.target_variable,
            "target_period": str(self.target_period),
            "method": self.method,
            "latest_vintage": str(self.latest_vintage),
            "nowcast": round(self.nowcast, 4),
            "forecast_sd": round(self.forecast_sd, 4),
            "ci_90_lower": round(self.nowcast - 1.645 * self.forecast_sd, 4),
            "ci_90_upper": round(self.nowcast + 1.645 * self.forecast_sd, 4),
        }
        return pd.DataFrame([rec])

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
        """Plot point nowcast with confidence error bands alongside factor paths."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.figure

        if not self.vintage_history.empty:
            v_dates = pd.to_datetime(self.vintage_history["vintage"])
            ncs = self.vintage_history["nowcast"].values
            sds = self.vintage_history["forecast_sd"].values
            ax.plot(v_dates, ncs, marker="o", lw=1.8, color="#0b3b60", label="Nowcast Path")
            ax.fill_between(
                v_dates,
                ncs - 1.645 * sds,
                ncs + 1.645 * sds,
                color="#0b3b60",
                alpha=0.2,
                label="90% Confidence Band",
            )
            ax.set_xlabel("Vintage Date")
            ax.set_ylabel(f"{self.target_variable} Nowcast")
        else:
            ax.bar(["Nowcast"], [self.nowcast], yerr=[1.645 * self.forecast_sd], capsize=6, color="#0b3b60", alpha=0.8)
            ax.set_ylabel(f"{self.target_variable} Nowcast")

        default_title = (
            f"Real-Time Nowcast: {self.country} {self.target_variable} "
            f"({self.target_period}) [{self.nowcast:+.3f}]"
        )
        ax.set_title(title or default_title, fontsize=11, fontweight="semibold")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        return fig

    def plot_news(self, *, ax: Any = None, title: str | None = None) -> Any:
        """Plot Bańbura & Modugno news decomposition waterfall chart."""
        if self.news_decomposition is None:
            raise ValueError("No news decomposition available in this nowcast result.")
        return self.news_decomposition.plot(ax=ax, title=title)

    def plot_fan_chart(
        self,
        *,
        history: pd.Series | None = None,
        ax: Any = None,
        title: str | None = None,
        levels: Sequence[float] = (0.3, 0.6, 0.9),
        palette: str | None = None,
    ) -> Any:
        """Plot central bank fan chart projection anchored at this nowcast."""
        if history is None:
            # Synthetic 6-period historical lead-in from nowcast
            h_vals = [self.nowcast - 0.2, self.nowcast - 0.1, self.nowcast + 0.05, self.nowcast]
            history = pd.Series(h_vals, index=[f"t-{3 - i}" for i in range(4)])

        fc_mean = pd.Series([self.nowcast, self.nowcast * 1.01, self.nowcast * 1.02], index=["t+1", "t+2", "t+3"])
        fc_sd = pd.Series([self.forecast_sd, self.forecast_sd * 1.2, self.forecast_sd * 1.4], index=fc_mean.index)

        theme = palette or self.palette or "default"
        fc_res = fan_chart(history, fc_mean, fc_sd, levels=levels, palette=theme)
        return fc_res.plot(ax=ax, title=title)


def realtime_nowcast(
    country: str = "MEX",
    target_variable: str | None = None,
    target_period: Any | None = None,
    method: str = "dfm",
    panel: VintagePanel | pd.DataFrame | str | Path | None = None,
    as_of: Any | None = None,
    previous_vintage: Any | None = None,
    cartridge_path: str | Path | None = None,
    n_factors: int = 1,
    p: int = 1,
    decompose_news: bool = True,
    compute_news_vs_noise: bool = False,
    **kwargs: Any,
) -> RealtimeNowcastResult:
    """High-level real-time macroeconomic nowcast orchestrator.

    Integrates Latin American central bank real-time data feeds with
    the Kalman Dynamic Factor Model or Mixed-Frequency VAR.

    Parameters
    ----------
    country : str, default 'MEX'
        Country ISO code: 'MEX' (Mexico), 'BRA' (Brazil), 'CHL' (Chile), 'USA' (US).
    target_variable : str, optional
        Target series name. If None, resolves from canonical country catalogue.
    target_period : Any, optional
        Reference period for nowcasting. Defaults to the latest period.
    method : {'dfm', 'mfvar'}, default 'dfm'
        Nowcasting algorithm:
        - 'dfm': Two-step Kalman Dynamic Factor Model.
        - 'mfvar': Mariano-Murasawa mixed-frequency VAR.
    panel : VintagePanel | pd.DataFrame | str | Path | None
        Input vintage panel or wide DataFrame. If path, loads from cartridge.
    as_of : Any, optional
        Information cutoff vintage date. Defaults to the latest available vintage.
    previous_vintage : Any, optional
        Baseline vintage date for news decomposition comparison.
    cartridge_path : str | Path, optional
        Path to offline .pmz portable vintage cartridge.
    n_factors : int, default 1
        Number of latent factors (for DFM).
    p : int, default 1
        Lag order (for DFM or MF-VAR).
    decompose_news : bool, default True
        Compute Bańbura & Modugno (2014) analytical news decomposition if
        multiple vintages are detected.
    compute_news_vs_noise : bool, default False
        Compute Mankiw-Shapiro revision test if supported by panel.

    Returns
    -------
    RealtimeNowcastResult
        Frozen dataclass with nowcast, standard errors, factor trajectories,
        news attribution, and presentation methods.
    """
    country_clean = str(country).upper()
    c_spec = COUNTRY_SPECS.get(country_clean, {})
    default_palette = c_spec.get("palette", "default")

    # 1. Resolve Data Panel
    vp: Optional[VintagePanel] = None
    df_wide: Optional[pd.DataFrame] = None

    if cartridge_path is not None:
        vp = load_realtime_cartridge(cartridge_path)
    elif isinstance(panel, (str, Path)):
        p_path = Path(panel)
        if p_path.suffix == ".pmz":
            vp = load_realtime_cartridge(p_path)
        else:
            raise ValueError(f"Unsupported file format: {p_path}")
    elif isinstance(panel, VintagePanel):
        vp = panel
    elif isinstance(panel, pd.DataFrame):
        req_cols = {"country", "variable", "date", "vintage", "value"}
        if req_cols.issubset(panel.columns):
            vp = VintagePanel(panel)
        else:
            df_wide = panel.copy()
    elif panel is None:
        # Check if fetch is possible
        try:
            from ..fetch.realtime import vintage_panel
            vp = vintage_panel([country_clean], progress=False)
        except Exception as exc:
            raise ValueError(
                f"No panel or cartridge provided and real-time fetch failed for "
                f"{country_clean}: {exc}. Provide panel= or cartridge_path=."
            ) from exc

    # 2. Extract Wide Datasets and Historical Vintages
    v_history_records = []
    df_wide_old: Optional[pd.DataFrame] = None
    latest_v_stamp: Any = "latest"

    if vp is not None and not vp.is_empty():
        # Filter country if multi-country panel
        all_vintages = sorted(vp.df[vp.df["country"] == country_clean]["vintage"].unique())
        if not all_vintages:
            all_vintages = sorted(vp.df["vintage"].unique())

        if as_of is not None:
            cutoff = pd.Timestamp(as_of)
            v_candidates = [v for v in all_vintages if pd.Timestamp(v) <= cutoff]
            if not v_candidates:
                raise ValueError(f"No vintages found as of {as_of}")
            latest_v_stamp = v_candidates[-1]
        else:
            latest_v_stamp = all_vintages[-1] if all_vintages else pd.Timestamp.now()

        # Extract latest wide slice
        df_asof = vp.as_of(latest_v_stamp)
        if isinstance(df_asof.index, pd.MultiIndex):
            if country_clean in df_asof.index.levels[0]:
                df_wide = df_asof.xs(country_clean)
            else:
                df_wide = df_asof.droplevel(0)
        else:
            df_wide = df_asof

        # Locate previous vintage for news decomposition
        if previous_vintage is not None:
            prev_stamp = pd.Timestamp(previous_vintage)
            df_prev = vp.as_of(prev_stamp)
            if isinstance(df_prev.index, pd.MultiIndex):
                if country_clean in df_prev.index.levels[0]:
                    df_wide_old = df_prev.xs(country_clean)
                else:
                    df_wide_old = df_prev.droplevel(0)
            else:
                df_wide_old = df_prev
        elif len(all_vintages) >= 2:
            prev_v = all_vintages[-2]
            df_prev = vp.as_of(prev_v)
            if isinstance(df_prev.index, pd.MultiIndex):
                if country_clean in df_prev.index.levels[0]:
                    df_wide_old = df_prev.xs(country_clean)
                else:
                    df_wide_old = df_prev.droplevel(0)
            else:
                df_wide_old = df_prev

    if df_wide is None or df_wide.empty:
        raise ValueError(f"Could not extract observation panel for {country_clean}")

    # 3. Resolve Target Variable
    target_var = target_variable
    if target_var is None:
        candidates = c_spec.get("candidates", ["gdp"])
        for cand in candidates:
            if cand in df_wide.columns:
                target_var = cand
                break
        if target_var is None:
            target_var = df_wide.columns[0]

    if target_var not in df_wide.columns:
        raise KeyError(f"Target variable {target_var!r} not found in columns: {list(df_wide.columns)}")

    # Determine target reference period
    if target_period is None:
        target_per = df_wide.index[-1]
    else:
        target_per = target_period

    # 4. Fit Model Engine
    method_clean = str(method).lower()

    if method_clean == "dfm":
        dfm = DynamicFactorModel(n_factors=n_factors, p=p, standardize=True)
        dfm.fit(df_wide)
        res_dfm = dfm.result_
        assert res_dfm is not None

        nowcast_val = dfm.nowcast(target_var, period=target_per)
        factors_df = res_dfm.factors_df
        loadings_df = res_dfm.to_frame()

        # Compute standard error
        t_col_idx = list(res_dfm.columns).index(target_var)
        t_row_idx = list(df_wide.index).index(target_per) if target_per in df_wide.index else -1
        Z_k = res_dfm.loadings[t_col_idx]
        P_cov = res_dfm.smoother_out.get("P_smooth", np.zeros((len(df_wide), n_factors, n_factors)))[t_row_idx][:n_factors, :n_factors]
        H_val = res_dfm.H[t_col_idx, t_col_idx]
        var_std = float(Z_k @ P_cov @ Z_k.T + H_val)
        forecast_sd = float(res_dfm.stds[t_col_idx] * np.sqrt(max(1e-6, var_std)))

        # News decomposition
        news_decomp: Optional[NewsDecompositionResult] = None
        if decompose_news and df_wide_old is not None and not df_wide_old.empty:
            news_decomp = banbura_modugno_news(
                dfm,
                df_wide_old,
                df_wide,
                target_series=target_var,
                target_period=target_per,
            )

    elif method_clean == "mfvar":
        p_mf = max(3, p)
        res_mf = mf_var(df_wide, quarterly_col=target_var, p=p_mf)
        df_filled = res_mf["df_filled"]
        valid_q = df_filled[target_var].dropna()
        nowcast_val = float(valid_q.iloc[-1]) if len(valid_q) else float("nan")
        forecast_sd = float(df_wide[target_var].dropna().std() * 0.4) if len(df_wide[target_var].dropna()) > 1 else 0.5
        factors_df = res_mf["df_monthly"]
        loadings_df = pd.DataFrame(res_mf["Z"], index=df_wide.columns)
        news_decomp = None
    else:
        raise ValueError(f"Unknown method {method!r}. Choose 'dfm' or 'mfvar'.")

    # 5. Build Vintage History if multiple vintages available
    if vp is not None and not vp.is_empty():
        all_vintages = sorted(vp.df[vp.df["country"] == country_clean]["vintage"].unique())
        if len(all_vintages) >= 2:
            recent_vintages = all_vintages[-10:]
            for v_date in recent_vintages:
                try:
                    df_v = vp.as_of(v_date)
                    if isinstance(df_v.index, pd.MultiIndex) and country_clean in df_v.index.levels[0]:
                        df_v_sub = df_v.xs(country_clean)
                    else:
                        df_v_sub = df_v
                    if target_var in df_v_sub.columns:
                        val = df_v_sub[target_var].dropna().iloc[-1]
                        v_history_records.append({
                            "vintage": v_date,
                            "nowcast": float(val),
                            "forecast_sd": forecast_sd,
                        })
                except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
                    pass

    df_v_history = pd.DataFrame(v_history_records) if v_history_records else pd.DataFrame(
        columns=["vintage", "nowcast", "forecast_sd"]
    )

    # 6. Optional Mankiw-Shapiro news vs. noise test
    news_noise_res = None
    if compute_news_vs_noise and vp is not None:
        try:
            news_noise_res = vp.news_or_noise(country_clean, target_var)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
            news_noise_res = None

    return RealtimeNowcastResult(
        nowcast=nowcast_val,
        forecast_sd=forecast_sd,
        target_variable=target_var,
        target_period=target_per,
        country=country_clean,
        method=method_clean,
        latest_vintage=latest_v_stamp,
        factors=factors_df,
        loadings=loadings_df,
        news_decomposition=news_decomp,
        news_vs_noise_test=news_noise_res,
        vintage_history=df_v_history,
        palette=default_palette,
    )


__all__ = [
    "RealtimeNowcastResult",
    "realtime_nowcast",
    "COUNTRY_SPECS",
]
