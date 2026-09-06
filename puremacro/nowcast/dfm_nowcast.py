"""Mixed-Frequency Dynamic Factor Model (DFM) GDP Nowcasting.

Implements the nowcasting framework of Giannone, Reichlin & Small (2008,
*JME*) in its EM-PCA + bridge-equation form:

- Static factor model estimated by an EM / iterative-PCA algorithm on a
  standardised monthly panel with missing values and ragged edges
  (Stock & Watson 2002). Missing entries are re-imputed at every iteration
  from the rank-``k`` reconstruction ``U_k S_k V_k'`` (``F = sqrt(T) U_k``,
  ``Lambda = V_k S_k / sqrt(T)``), so the imputation is on the same scale
  as the data.
- A factor VAR(``p_factor_lags``) fitted on the estimated factors. It
  supplies the factor path for months that carry no observation at all
  (all-NaN rows) and forecasts the months of the target quarter that are
  not yet in the frame, so the target quarter's factor average is always a
  three-month average — the quantity the bridge was estimated on.
- Quarterly bridge regression linking three-month factor averages to
  quarterly GDP growth, aligned by quarter label.
- A "news" table: for every series published in the latest month, its
  surprise relative to the factor-implied value and the projection weight
  ``beta' (Lambda'Lambda)^{-1} Lambda_i / (3 sigma_i)`` that maps the
  surprise into the nowcast.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NowcastResult:
    """Results from Mixed-Frequency DFM GDP Nowcast.

    Attributes
    ----------
    nowcast : float
        Point nowcast of the target quarter, in the units of
        ``quarterly_gdp`` (no annualisation is applied).
    target_quarter : str
        Target GDP quarter label (e.g. ``'2024Q1'``). It is the last quarter
        present in the monthly frame; the label can be overridden with
        ``target_quarter=`` but the nowcast is always for that last quarter.
    factors : pd.DataFrame
        Estimated monthly factors ``F_t`` (T_months, K), indexed like
        ``monthly_data``. Rows with no observation at all carry the factor
        VAR forecast.
    loadings : pd.DataFrame
        Factor loadings ``Lambda`` (N_series, K) on the standardised panel,
        scaled so that ``F @ Lambda.T`` is the rank-K reconstruction.
    news_decomposition : pd.DataFrame
        One row per series published in the latest month:
        ``series``, ``actual``, ``forecast`` (factor-implied value, same
        units), ``surprise`` (= actual - forecast), ``weight`` (projection
        weight of the surprise on the nowcast) and ``contribution``
        (= weight * surprise, units of ``quarterly_gdp``).
    model_r2 : float
        In-sample R² of the quarterly bridge regression, clipped to [0, 1];
        ``0.0`` when ``quarterly_gdp`` has no variance.
    factor_forecast : pd.DataFrame
        Factor VAR forecasts for the months of the target quarter that lie
        beyond the end of ``monthly_data`` (0, 1 or 2 rows). Empty when the
        frame ends on a quarter's last month.
    bridge_coefficients : pd.Series
        Bridge OLS coefficients ``[const, Factor_1, …, Factor_K]``; the
        nowcast is ``const + coef @ target_quarter_factor_average``.
    factor_var : np.ndarray or None
        Coefficient matrix ``B`` (K·p, K) of the factor VAR(p) in stacked
        form ``F_t = [F_{t-1}, …, F_{t-p}] @ B`` (no intercept); ``None``
        when there were too few rows to fit it (forecasts then fall back
        to the unconditional mean, zero).
    """
    nowcast: float
    target_quarter: str
    factors: pd.DataFrame
    loadings: pd.DataFrame
    news_decomposition: pd.DataFrame
    model_r2: float
    factor_forecast: pd.DataFrame = field(default_factory=pd.DataFrame)
    bridge_coefficients: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    factor_var: Optional[np.ndarray] = None

    def summary(self) -> str:
        n_fc = len(self.factor_forecast)
        lines = [
            "Dynamic Factor Model GDP Nowcasting (Giannone, Reichlin & Small 2008)",
            "=" * 74,
            f"Target Quarter                  : {self.target_quarter}",
            f"GDP Growth Nowcast              : {self.nowcast:.2f} (units of quarterly_gdp)",
            f"Quarterly Bridge Regression R²  : {self.model_r2:.4f}",
            f"Latent Monthly Factors (K)      : {self.factors.shape[1]}",
            f"Target-quarter months forecast  : {n_fc} (factor VAR)",
            "-" * 74,
            "Latest News & Contribution Decomposition:",
        ]
        if not self.news_decomposition.empty:
            for _, row in self.news_decomposition.iterrows():
                lines.append(
                    f"  {row['series']:<22s} | Surprise: {row['surprise']:+6.2f} | "
                    f"Impact: {row['contribution']:+6.3f}"
                )
        else:
            lines.append("  (no series observed in the latest month)")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """News table (one row per series published in the latest month)."""
        cols = ["series", "actual", "forecast", "surprise", "weight", "contribution"]
        if self.news_decomposition.empty:
            # One placeholder row so the table renderers still emit a header.
            return pd.DataFrame(
                [[np.nan] * (len(cols) - 1)], columns=cols[1:],
                index=pd.Index(["(no series observed in the latest month)"], name="series"),
            )
        return self.news_decomposition[cols].set_index("series").round(4)

    def to_markdown(self, **kwargs: Any) -> str:
        """Export the news table to Markdown."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Export the news table to a LaTeX ``tabular``."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Export the news table to a Typst ``#table``."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), **kwargs)

    def plot(self, *, ax: Any = None, title: str | None = None) -> Any:
        """Plot the monthly factors (solid) and the target-quarter forecast
        months (dashed). Returns the matplotlib Figure."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(7.5, 3.6))
        else:
            fig = ax.figure
        for col in self.factors.columns:
            line, = ax.plot(self.factors.index, self.factors[col], lw=1.6, label=col)
            if not self.factor_forecast.empty:
                x_fc = [self.factors.index[-1], *self.factor_forecast.index]
                y_fc = [self.factors[col].iloc[-1], *self.factor_forecast[col]]
                ax.plot(x_fc, y_fc, lw=1.6, ls="--", color=line.get_color())
        ax.axhline(0.0, color="grey", lw=0.6)
        ax.set_title(title or f"Monthly factors — nowcast {self.target_quarter}: {self.nowcast:.2f}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, ls=":", alpha=0.5)
        return fig


def _quarter_labels(index: pd.Index) -> list[str]:
    """Quarter labels ``'2015Q1'`` for a Period/Datetime index; ``str`` otherwise."""
    if isinstance(index, pd.PeriodIndex):
        return [str(x) for x in index.asfreq("Q")]
    if isinstance(index, pd.DatetimeIndex):
        return [str(x) for x in index.to_period("Q")]
    return [str(x) for x in index]


def _fit_factor_var(F: np.ndarray, p: int, informative: np.ndarray) -> Optional[np.ndarray]:
    """OLS VAR(p) without intercept on the factor rows whose own value and
    all ``p`` lags are informative (carry at least one observation).

    Returns the stacked coefficient matrix ``B`` (K·p, K) such that
    ``F_t = concat(F_{t-1}, …, F_{t-p}) @ B``, or ``None`` when fewer than
    ``K·p + 2`` usable rows exist.
    """
    T, k = F.shape
    rows = [t for t in range(p, T) if informative[t - p:t + 1].all()]
    if len(rows) < k * p + 2:
        return None
    Y = F[rows]
    X = np.column_stack([F[[t - 1 - lag for t in rows]] for lag in range(p)])
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return np.asarray(B, dtype=float)


def _var_step(history: np.ndarray, B: Optional[np.ndarray], p: int) -> np.ndarray:
    """One-step factor forecast from the last ``p`` rows of ``history``."""
    k = history.shape[1]
    if B is None or history.shape[0] < p:
        return np.zeros(k)
    x = np.concatenate([history[-1 - lag] for lag in range(p)])
    return np.asarray(x @ B, dtype=float)


def nowcast_gdp(
    monthly_data: pd.DataFrame,
    quarterly_gdp: pd.Series,
    *,
    target_quarter: str | None = None,
    n_factors: int = 2,
    p_factor_lags: int = 1,
    max_em_iter: int = 50,
    em_tol: float = 1e-4,
) -> NowcastResult:
    """Mixed-frequency GDP nowcast: EM-PCA factors, a factor VAR and a bridge.

    Steps
    -----
    1. Standardise ``monthly_data`` and estimate ``n_factors`` factors by
       EM / iterative PCA: missing entries are re-imputed from the rank-K
       reconstruction ``F @ Lambda.T`` (``F = sqrt(T) U_K``,
       ``Lambda = V_K S_K / sqrt(T)``) until the factors move by less than
       ``em_tol`` (at most ``max_em_iter`` iterations). Months with no
       observation at all carry no information and are held at the
       unconditional mean during the EM.
    2. Fit a VAR(``p_factor_lags``) on the factors. Use it to (a) replace
       the factor of all-NaN months by the forecast from the preceding
       months and (b) forecast the months of the target quarter that lie
       beyond the end of the frame, so the target quarter always has a
       three-month factor average.
    3. Average the factors within each calendar quarter and regress
       ``quarterly_gdp`` on those averages (the bridge), aligning the two by
       quarter label. At least four quarters must match, otherwise a
       ``ValueError`` is raised — there is no positional fallback.
    4. Apply the bridge to the target quarter's factor average.

    Parameters
    ----------
    monthly_data : DataFrame of shape (T_months, N)
        Monthly indicators. NaN anywhere; NaN in the last rows is the
        ragged edge. A ``DatetimeIndex`` (monthly) is grouped into calendar
        quarters; any other index is grouped positionally in blocks of
        three rows labelled ``'Q1', 'Q2', …``. End the frame at the last
        month with at least one observation — appending all-NaN rows is
        equivalent (they are forecast by the factor VAR), not harmful.
    quarterly_gdp : Series
        Historical quarterly GDP growth (any units; the nowcast inherits
        them). Its index is converted to quarter labels: a ``PeriodIndex``
        or ``DatetimeIndex`` becomes ``'2015Q1', …`` via ``to_period("Q")``;
        any other index is used as ``str`` and must match the factor
        labels (``'2015Q1'`` for a datetime-indexed monthly frame, ``'Q1'``
        for a positional one).
    target_quarter : str, optional
        Label to report; defaults to the last quarter of the monthly frame.
        Label only — the nowcast is always for that last quarter.
    n_factors : int, default 2
        Number of factors K (``1 <= K <= N``).
    p_factor_lags : int, default 1
        Lag order of the factor VAR used to fill all-NaN months and to
        complete the target quarter.
    max_em_iter : int, default 50
        Maximum EM iterations.
    em_tol : float, default 1e-4
        Convergence tolerance on ``max|F_new - F_old|``.

    Returns
    -------
    NowcastResult

    Raises
    ------
    ValueError
        If ``n_factors`` or ``p_factor_lags`` are out of range, or if fewer
        than four quarter labels of ``quarterly_gdp`` match the monthly
        frame's quarters.
    """
    if not isinstance(monthly_data, pd.DataFrame):
        raise TypeError("monthly_data must be a pandas DataFrame")
    if not isinstance(quarterly_gdp, pd.Series):
        raise TypeError("quarterly_gdp must be a pandas Series")
    X_raw = monthly_data.copy()
    var_names = [str(c) for c in X_raw.columns]
    T_m, N = X_raw.shape
    if not 1 <= n_factors <= min(N, T_m):
        raise ValueError(f"n_factors must be in [1, min(N, T)] = [1, {min(N, T_m)}]; got {n_factors}")
    if p_factor_lags < 1:
        raise ValueError(f"p_factor_lags must be >= 1; got {p_factor_lags}")
    if T_m < 3:
        raise ValueError(f"monthly_data needs at least 3 rows; got {T_m}")
    p = int(p_factor_lags)
    k = int(n_factors)

    # ---- 1. Standardise and run the EM / iterative PCA -------------------
    means = X_raw.mean(skipna=True)
    stds = X_raw.std(skipna=True).replace(0.0, 1.0).fillna(1.0)
    means = means.fillna(0.0)
    X = (X_raw - means) / stds
    missing_mask = X.isna().to_numpy()
    informative = ~missing_mask.all(axis=1)
    X_obs0 = np.array(X.fillna(0.0).to_numpy(dtype=float), dtype=float, copy=True)
    X_filled = np.array(
        X.interpolate(method="linear", limit_direction="both").fillna(0.0).to_numpy(dtype=float),
        dtype=float, copy=True,
    )
    X_filled[~informative] = 0.0

    F = np.zeros((T_m, k))
    Lambda = np.zeros((N, k))
    sqrt_T = np.sqrt(T_m)
    for _ in range(max_em_iter):
        F_old = F.copy()
        U, S, Vt = np.linalg.svd(X_filled, full_matrices=False)
        F = U[:, :k] * sqrt_T
        Lambda = Vt[:k, :].T * S[:k] / sqrt_T           # (N, K): F @ Lambda.T == U_k S_k V_k'
        fitted = F @ Lambda.T
        X_filled = np.where(missing_mask, fitted, X_obs0)
        X_filled[~informative] = 0.0
        if np.max(np.abs(F - F_old)) < em_tol:
            break

    # ---- 2. Factor VAR(p): all-NaN months and target-quarter completion --
    B = _fit_factor_var(F, p, informative)
    if B is None:
        warnings.warn(
            "nowcast_gdp: too few informative months to fit the factor "
            f"VAR({p}); all-NaN months and target-quarter completion use "
            "the unconditional factor mean (0).",
            RuntimeWarning, stacklevel=2,
        )
    F = F.copy()
    for t in np.where(~informative)[0]:
        F[t] = _var_step(F[:t], B, p)

    idx = monthly_data.index
    factor_cols = [f"Factor_{j + 1}" for j in range(k)]
    if isinstance(idx, pd.DatetimeIndex):
        month_labels = _quarter_labels(idx)
        remaining = (2 - (int(idx[-1].month) - 1) % 3)
        fc_index: pd.Index = pd.DatetimeIndex(
            [idx[-1] + pd.DateOffset(months=h) for h in range(1, remaining + 1)]
        )
        fc_labels = _quarter_labels(fc_index)
    else:
        month_labels = [f"Q{t // 3 + 1}" for t in range(T_m)]
        remaining = (3 - T_m % 3) % 3
        fc_index = pd.Index([T_m + h for h in range(remaining)])
        fc_labels = [f"Q{(T_m + h) // 3 + 1}" for h in range(remaining)]

    hist = F.copy()
    fc_rows = []
    for _ in range(remaining):
        nxt = _var_step(hist, B, p)
        fc_rows.append(nxt)
        hist = np.vstack([hist, nxt[None, :]])
    df_F = pd.DataFrame(F, index=idx, columns=factor_cols)
    df_F_fc = pd.DataFrame(np.array(fc_rows).reshape(remaining, k), index=fc_index, columns=factor_cols)

    # ---- 3. Quarterly averages and the bridge ---------------------------
    df_F_all = pd.concat([df_F, df_F_fc]) if remaining else df_F
    labels_all = np.array(month_labels + list(fc_labels))
    df_F_q = df_F_all.groupby(labels_all, sort=False).mean()

    gdp_labels = _quarter_labels(quarterly_gdp.index)
    pos: dict[str, int] = {}
    for i, lab in enumerate(gdp_labels):
        pos.setdefault(lab, i)
    gdp_vals = quarterly_gdp.to_numpy(dtype=float)
    common_q = [q for q in df_F_q.index if q in pos and np.isfinite(gdp_vals[pos[q]])]
    if len(common_q) < 4:
        raise ValueError(
            "nowcast_gdp: fewer than 4 quarter labels of quarterly_gdp match the "
            f"monthly frame ({len(common_q)} matched). Factor quarters look like "
            f"{list(df_F_q.index[:3])}, quarterly_gdp labels like {gdp_labels[:3]}. "
            "Give quarterly_gdp a PeriodIndex/DatetimeIndex, or string labels of the "
            "same form; positional alignment is not attempted."
        )
    Y_bridge = gdp_vals[[pos[q] for q in common_q]]
    X_bridge = np.column_stack([np.ones(len(common_q)), df_F_q.loc[common_q].to_numpy()])
    beta_bridge = np.linalg.lstsq(X_bridge, Y_bridge, rcond=None)[0]
    y_pred = X_bridge @ beta_bridge
    ss_tot = float(np.sum((Y_bridge - Y_bridge.mean()) ** 2))
    ss_res = float(np.sum((Y_bridge - y_pred) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # ---- 4. Target-quarter nowcast --------------------------------------
    target_label = str(df_F_q.index[-1])
    target_factors = df_F_q.iloc[-1].to_numpy()
    nowcast_val = float(beta_bridge[0] + np.dot(beta_bridge[1:], target_factors))

    # ---- News table for the latest month ---------------------------------
    LtL_inv = np.linalg.pinv(Lambda.T @ Lambda)             # (K, K)
    proj = LtL_inv @ Lambda.T                               # (K, N): F_t = proj @ x_t
    news_records = []
    t_last = T_m - 1
    for i, col in enumerate(var_names):
        actual_val = X_raw.iloc[t_last, i]
        if pd.isna(actual_val):
            continue
        expected_val = float(means.iloc[i] + stds.iloc[i] * np.dot(F[t_last], Lambda[i]))
        surprise = float(actual_val - expected_val)
        w_i = float(beta_bridge[1:] @ proj[:, i] / (3.0 * float(stds.iloc[i])))
        news_records.append({
            "series": col,
            "actual": float(actual_val),
            "forecast": expected_val,
            "surprise": surprise,
            "weight": w_i,
            "contribution": float(w_i * surprise),
        })
    df_news = pd.DataFrame(
        news_records,
        columns=["series", "actual", "forecast", "surprise", "weight", "contribution"],
    )

    return NowcastResult(
        nowcast=nowcast_val,
        target_quarter=str(target_quarter) if target_quarter is not None else target_label,
        factors=df_F,
        loadings=pd.DataFrame(Lambda, index=var_names, columns=factor_cols),
        news_decomposition=df_news,
        model_r2=max(0.0, min(1.0, float(r2))),
        factor_forecast=df_F_fc,
        bridge_coefficients=pd.Series(beta_bridge, index=["const", *factor_cols]),
        factor_var=B,
    )


__all__ = [
    "NowcastResult",
    "nowcast_gdp",
]
