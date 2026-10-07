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
from .dfm import DynamicFactorModel, DynamicFactorModelResult
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
        The current information set Ω_v: the last vintage published on or
        before ``as_of`` (the panel's last vintage when ``as_of`` is None).
    factors : pd.DataFrame
        DataFrame of estimated common dynamic factors or monthly states.
    loadings : pd.DataFrame
        Observation loadings matrix or state mappings.
    news_decomposition : Optional[NewsDecompositionResult]
        Bańbura & Modugno news decomposition of the update from the baseline
        information set Ω_{v-1} (``previous_vintage``) to Ω_v
        (``latest_vintage``); None when no earlier vintage exists at the
        information cutoff.
    news_vs_noise_test : Optional[Any]
        Mankiw-Shapiro news-vs-noise revision test, computed only from
        vintages <= ``latest_vintage``; None if not requested or not
        identified on the available editions.
    vintage_history : pd.DataFrame
        One row per vintage <= ``latest_vintage`` (at most the last 10), with
        columns ``vintage``, ``last_observed_period`` and
        ``last_observed_value``: the most recent published value of the
        target series in that vintage and the reference period it refers to.
        These are data, not model nowcasts. Empty when fewer than two
        vintages are available at the information cutoff.
    palette : str
        Central bank visual palette ('banxico', 'bcb', 'bank_of_england', 'default').
    previous_vintage : pd.Timestamp | None
        Baseline date of the news decomposition (Ω_{v-1}); None when no
        decomposition was computed.
    model_result : DynamicFactorModelResult | None
        The fitted DFM (state-space matrices and Kalman smoother output) for
        ``method='dfm'``; None for ``method='mfvar'``. :meth:`fan_chart` needs it.
    observed_panel : pd.DataFrame
        The wide panel of the current vintage Ω_v on which the model was
        fitted: published data only, NaN where a value was not yet published.
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
    previous_vintage: Any = None
    model_result: Optional[DynamicFactorModelResult] = None
    observed_panel: pd.DataFrame = field(default_factory=pd.DataFrame)

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
            err = float(nd.decomposition_error)
            err_note = "(< 1e-10)" if err < 1e-10 else "(IDENTITY FAILS: > 1e-10)"
            lines.append("Latest Vintage News & Revisions Attribution:")
            if self.previous_vintage is not None:
                lines.append(f"  Baseline Vintage (v-1)      : {self.previous_vintage}")
            lines.append(f"  Previous Nowcast (v-1)      : {nd.forecast_old:+.4f}")
            lines.append(f"  Updated Nowcast (v)         : {nd.forecast_new:+.4f}")
            lines.append(f"  Total Nowcast Revision (Δy) : {nd.revision:+.4f}")
            lines.append(f"  Impact from new releases    : {sum(nd.impact_releases.values()):+.4f}")
            lines.append(f"  Impact from revisions       : {sum(nd.impact_revisions.values()):+.4f}")
            lines.append(f"  Analytical Identity Error   : {err:.2e} {err_note}")
            if abs(float(nd.forecast_new) - self.nowcast) > 1e-8 * max(1.0, abs(self.nowcast)):
                lines.append("  Note: the news is decomposed on the model's fitted value; the")
                lines.append("  Point Nowcast is the published figure when the target is observed.")
            lines.append("-" * 74)

        if not self.vintage_history.empty:
            lines.append(f"Last Observed {self.target_variable} by Vintage (data, not nowcasts):")
            tail_hist = self.vintage_history.tail(5)
            lines.append(f"{'Vintage':<16} {'Last period':<16} {'Value':>12}")
            for _, r in tail_hist.iterrows():
                v_str = str(r.get("vintage", ""))[:16]
                p_str = str(r.get("last_observed_period", ""))[:16]
                val = float(r.get("last_observed_value", float("nan")))
                lines.append(f"{v_str:<16s} {p_str:<16s} {val:>12.4f}")
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
        """Plot the point nowcast with its 90% band (±1.645 σ).

        With a vintage history, the last observed target value of each
        vintage (published data, not model nowcasts) is drawn against the
        vintage date, and the current nowcast is marked at ``latest_vintage``.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.figure

        if not self.vintage_history.empty:
            v_dates = pd.to_datetime(self.vintage_history["vintage"])
            vals = self.vintage_history["last_observed_value"].astype(float).values
            ax.plot(v_dates, vals, marker="o", lw=1.8, color="#0b3b60",
                    label=f"Last observed {self.target_variable} (data, by vintage)")
            ax.errorbar(
                [pd.Timestamp(self.latest_vintage)],
                [self.nowcast],
                yerr=[1.645 * self.forecast_sd],
                fmt="D",
                color="#8b1a1a",
                capsize=5,
                label=f"Nowcast for {self.target_period} (90% band)",
            )
            ax.set_xlabel("Vintage Date")
            ax.set_ylabel(f"{self.target_variable}")
        else:
            ax.bar(["Nowcast"], [self.nowcast], yerr=[1.645 * self.forecast_sd], capsize=6, color="#0b3b60", alpha=0.8)
            ax.set_ylabel(f"{self.target_variable} Nowcast")

        default_title = (
            f"Real-Time Nowcast: {self.country} {self.target_variable} "
            f"({self.target_period}) [{self.nowcast:+.3f}]"
        )
        ax.set_title(title or default_title, fontsize=11, fontweight="bold")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        return fig

    def plot_news(self, *, ax: Any = None, title: str | None = None) -> Any:
        """Plot Bańbura & Modugno news decomposition waterfall chart."""
        if self.news_decomposition is None:
            raise ValueError("No news decomposition available in this nowcast result.")
        return self.news_decomposition.plot(ax=ax, title=title)

    def fan_chart(
        self,
        horizon: int = 3,
        *,
        history: pd.Series | None = None,
        n_history: int = 12,
        levels: Sequence[float] = (0.3, 0.6, 0.9),
        palette: str | None = None,
    ) -> FanChartResult:
        """Model-based predictive fan for the target, from the DFM's state space.

        The fan covers every period after the last published value of the
        target: first the unpublished periods still inside the panel (the
        ragged edge, which includes the nowcast), then ``horizon`` periods
        beyond the panel's last date T. With a_{t|T} and P_{t|T} the smoothed
        state mean and variance given the current vintage Ω_v, λ_j the
        target's row of the observation matrix (zero on lagged factors), R_jj
        its idiosyncratic variance, and μ_j, s_j the mean and standard
        deviation used to standardise it:

            mean_t = μ_j + s_j λ_j' a_{t|T},   var_t = s_j² (λ_j' P_{t|T} λ_j + R_jj)

        inside the panel, and beyond it a_{T+h|T} = A a_{T+h-1|T},
        P_{T+h|T} = A P_{T+h-1|T} A' + Q with the same mapping. The bands are
        Gaussian quantiles of these variances. They condition on the
        estimated parameters, so they ignore parameter uncertainty.

        Parameters
        ----------
        horizon : int, default 3
            Periods beyond the last date of the panel (0 draws only the
            ragged edge).
        history : pd.Series, optional
            Observed values drawn before the fan. Defaults to the last
            ``n_history`` published values of the target in the current
            vintage (:attr:`observed_panel`).
        n_history : int, default 12
            Length of the default history.
        levels : sequence of float
            Central coverage levels of the bands.
        palette : str, optional
            Fan colour theme; defaults to the country's palette.

        Returns
        -------
        FanChartResult

        Raises
        ------
        ValueError
            For a result without a fitted DFM (``method='mfvar'``): no
            model-based predictive variances are available, and no fan is
            drawn rather than an invented one.
        """
        fit = self.model_result
        if fit is None:
            raise ValueError(
                f"fan_chart needs the fitted DFM state space, but this result "
                f"(method={self.method!r}) carries none: its predictive "
                "variances are not available, so no fan can be drawn. Use "
                "method='dfm'."
            )
        if int(horizon) < 0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")
        horizon = int(horizon)

        names = [str(c) for c in fit.columns]
        target = str(self.target_variable)
        if target not in names:
            raise KeyError(f"Target {target!r} is not a column of the fitted model: {names}")
        j = names.index(target)

        a_sm = np.asarray(fit.smoother_out["a_smooth"], dtype=float)
        P_sm = np.asarray(fit.smoother_out["P_smooth"], dtype=float)
        idx = pd.Index(fit.index) if fit.index is not None else pd.RangeIndex(len(a_sm))

        if self.observed_panel.empty or target not in self.observed_panel.columns:
            raise ValueError(
                "fan_chart needs the published target values (observed_panel) "
                "to know where the data end; this result has none."
            )
        published = self.observed_panel[target].reindex(idx)
        pub_pos = np.flatnonzero(published.notna().to_numpy())
        first_fan = int(pub_pos[-1]) + 1 if pub_pos.size else 0

        m = fit.A.shape[0]
        lam = np.zeros(m)
        lam[: fit.n_factors] = fit.loadings[j]
        s_j, mu_j, R_jj = float(fit.stds[j]), float(fit.means[j]), float(fit.H[j, j])

        def _moments(a: np.ndarray, P: np.ndarray) -> tuple[float, float]:
            var = float(lam @ P @ lam + R_jj)
            return mu_j + s_j * float(lam @ a), s_j * np.sqrt(max(var, 0.0))

        periods: list[Any] = []
        means: list[float] = []
        sds: list[float] = []
        for t in range(first_fan, len(idx)):
            mu_t, sd_t = _moments(a_sm[t], P_sm[t])
            periods.append(idx[t]); means.append(mu_t); sds.append(sd_t)

        a, P = a_sm[-1].copy(), P_sm[-1].copy()
        for h, when in enumerate(_extend_index(idx, horizon)):
            a = fit.A @ a
            P = fit.A @ P @ fit.A.T + fit.Q
            mu_t, sd_t = _moments(a, P)
            periods.append(when); means.append(mu_t); sds.append(sd_t)

        if not periods:
            raise ValueError(
                "Nothing to fan: the target is published in the last period of "
                "the panel and horizon=0."
            )
        fan_index = pd.Index(periods)
        if history is None:
            history = published.dropna().iloc[-int(n_history):] if pub_pos.size else pd.Series(dtype=float)
            if history.empty:
                raise ValueError(
                    f"The target {target!r} has no published value in the "
                    "current vintage; pass history= to draw the fan."
                )
        history = pd.Series(history).astype(float)

        theme = palette or self.palette or "default"
        return fan_chart(
            history,
            pd.Series(means, index=fan_index),
            pd.Series(sds, index=fan_index),
            levels=levels,
            palette=theme,
        )

    def plot_fan_chart(
        self,
        *,
        horizon: int = 3,
        history: pd.Series | None = None,
        n_history: int = 12,
        ax: Any = None,
        title: str | None = None,
        levels: Sequence[float] = (0.3, 0.6, 0.9),
        palette: str | None = None,
    ) -> Any:
        """Plot the model-based fan of :meth:`fan_chart` after the published data.

        The line before the fan is observed data (the target as published in
        the current vintage, unless ``history`` is given); the fan's centre and
        bands come from the DFM's state space. Raises ValueError for
        ``method='mfvar'`` results, which carry no predictive variances.
        """
        fc = self.fan_chart(
            horizon, history=history, n_history=n_history,
            levels=levels, palette=palette,
        )
        vintage = self.latest_vintage
        if isinstance(vintage, (pd.Timestamp, np.datetime64)):
            vintage = pd.Timestamp(vintage).strftime("%Y-%m-%d")
        default_title = (
            f"{self.country} {self.target_variable}: DFM predictive fan "
            f"(vintage {vintage}; parameter uncertainty ignored)"
        )
        fig = fc.plot(ax=ax, title=title or default_title)
        target_ax = ax if ax is not None else fig.axes[0]
        target_ax.set_ylabel(str(self.target_variable))
        return fig


_VINTAGE_HISTORY_COLUMNS = ("vintage", "last_observed_period", "last_observed_value")


def _extend_index(idx: pd.Index, steps: int) -> pd.Index:
    """The ``steps`` periods after the last element of a regular index.

    A DatetimeIndex is extended at its own (or inferred) frequency and an
    integer index by 1, 2, ...; anything else raises ValueError rather than
    guessing a date.
    """
    if steps <= 0:
        return pd.Index([])
    if isinstance(idx, pd.DatetimeIndex):
        freq = idx.freq or (pd.infer_freq(idx) if len(idx) >= 3 else None)
        if freq is None:
            raise ValueError(
                "Cannot date the forecast periods: the panel's DatetimeIndex "
                "has no regular frequency. Use horizon=0 or a regular index."
            )
        return pd.date_range(idx[-1], periods=steps + 1, freq=freq)[1:]
    if pd.api.types.is_integer_dtype(idx):
        return pd.Index(int(idx[-1]) + np.arange(1, steps + 1))
    raise ValueError(
        f"Cannot extend an index of type {type(idx).__name__} beyond the "
        "panel; use horizon=0."
    )


def _country_slice(vp: VintagePanel, vintage_date: Any, country: str) -> pd.DataFrame:
    """Wide (date x variable) frame of ``country`` as published on ``vintage_date``.

    Uses ``VintagePanel.as_of``, which keeps only rows with vintage <=
    ``vintage_date`` (the data set in hand that day).
    """
    df = vp.as_of(vintage_date)
    if isinstance(df.index, pd.MultiIndex):
        if country in df.index.levels[0]:
            return df.xs(country)
        return df.droplevel(0)
    return df


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
        A wide DataFrame has no vintage dimension, so ``as_of`` and
        ``previous_vintage`` cannot be honoured on it and raise ValueError.
    as_of : Any, optional
        Information cutoff date for a historical (pseudo-real-time) replay.
        Only vintages published on or before ``as_of`` are used anywhere in
        the result: the model fit, the news baseline, ``vintage_history``
        and the news-vs-noise test. The current information set Ω_v is the
        last such vintage. Defaults to the panel's last vintage.
    previous_vintage : Any, optional
        Baseline date Ω_{v-1} for the news decomposition; the baseline data
        set is the panel as published on that date (``VintagePanel.as_of``).
        Must be strictly before the current vintage Ω_v and never after
        ``as_of``; otherwise ValueError. Defaults to the last vintage
        strictly before Ω_v, i.e. two consecutive vintages Ω_{v-1} ⊂ Ω_v
        as in Bańbura & Modugno (2010, ECB WP 1189, sec. 2.3, p. 16). If no
        earlier vintage exists (``as_of`` on the first vintage), no
        decomposition is computed.
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
        Compute Mankiw-Shapiro revision test if supported by panel, on the
        vintages <= the current vintage only.

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
            if as_of is not None or previous_vintage is not None:
                raise ValueError(
                    "as_of / previous_vintage need a vintage dimension, but a "
                    "wide DataFrame has none: the information cutoff cannot "
                    "be enforced. Pass a VintagePanel (or a long DataFrame "
                    "with columns country, variable, date, vintage, value), "
                    "or truncate the wide frame yourself and omit these "
                    "arguments."
                )
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
    #
    # Information set: every output below is built only from the vintages
    # published on or before the current vintage Ω_v (``latest_v_stamp``),
    # which is the last vintage <= as_of. The news baseline Ω_{v-1} is the
    # vintage just before Ω_v, so Ω_{v-1} ⊂ Ω_v as the Bańbura-Modugno
    # decomposition requires.
    v_history_records = []
    df_wide_old: Optional[pd.DataFrame] = None
    latest_v_stamp: Any = "latest"
    prev_v_stamp: Any = None
    eligible_vintages: list = []

    if vp is not None and not vp.is_empty():
        # Filter country if multi-country panel
        all_vintages = sorted(vp.df[vp.df["country"] == country_clean]["vintage"].unique())
        if not all_vintages:
            all_vintages = sorted(vp.df["vintage"].unique())

        if as_of is not None:
            cutoff = pd.Timestamp(as_of)
            eligible_vintages = [v for v in all_vintages if pd.Timestamp(v) <= cutoff]
            if not eligible_vintages:
                raise ValueError(f"No vintages found as of {as_of}")
        else:
            cutoff = None
            eligible_vintages = list(all_vintages)
        latest_v_stamp = eligible_vintages[-1]
        current_ts = pd.Timestamp(latest_v_stamp)
        info_cutoff = cutoff if cutoff is not None else current_ts

        # Extract latest wide slice
        df_wide = _country_slice(vp, latest_v_stamp, country_clean)

        # Locate previous vintage for news decomposition
        if previous_vintage is not None:
            prev_stamp = pd.Timestamp(previous_vintage)
            if prev_stamp > info_cutoff:
                raise ValueError(
                    f"previous_vintage={prev_stamp} is after the information "
                    f"cutoff {info_cutoff} (as_of={as_of!r}); the news baseline "
                    "must be a data set that was available at as_of."
                )
            if prev_stamp >= current_ts:
                raise ValueError(
                    f"previous_vintage={prev_stamp} must be strictly before the "
                    f"current vintage {current_ts}: on or after it the baseline "
                    "is the same information set and the news is identically zero."
                )
            prev_v_stamp = prev_stamp
        elif len(eligible_vintages) >= 2:
            prev_v_stamp = eligible_vintages[-2]
        if prev_v_stamp is not None:
            df_wide_old = _country_slice(vp, prev_v_stamp, country_clean)

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
        fitted_model: Optional[DynamicFactorModelResult] = res_dfm

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
        fitted_model = None
    else:
        raise ValueError(f"Unknown method {method!r}. Choose 'dfm' or 'mfvar'.")

    # 5. Vintage history: the last published value of the target in each of
    # the (at most 10) most recent vintages <= the current vintage. These are
    # observed data, not model nowcasts, and are labelled as such.
    if vp is not None and len(eligible_vintages) >= 2:
        for v_date in eligible_vintages[-10:]:
            df_v_sub = _country_slice(vp, v_date, country_clean)
            if target_var not in df_v_sub.columns:
                continue
            obs = df_v_sub[target_var].dropna()
            if obs.empty:
                continue
            v_history_records.append({
                "vintage": v_date,
                "last_observed_period": obs.index[-1],
                "last_observed_value": float(obs.iloc[-1]),
            })

    df_v_history = pd.DataFrame(v_history_records) if v_history_records else pd.DataFrame(
        columns=list(_VINTAGE_HISTORY_COLUMNS)
    )

    # 6. Optional Mankiw-Shapiro news vs. noise test, on vintages <= Ω_v only
    news_noise_res = None
    if compute_news_vs_noise and vp is not None:
        try:
            vp_info = vp if as_of is None else vp.filter(end_vintage=latest_v_stamp)
            news_noise_res = vp_info.news_or_noise(country_clean, target_var)
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
        previous_vintage=prev_v_stamp if news_decomp is not None else None,
        model_result=fitted_model,
        observed_panel=df_wide.copy(),
    )


__all__ = [
    "RealtimeNowcastResult",
    "realtime_nowcast",
    "COUNTRY_SPECS",
]
