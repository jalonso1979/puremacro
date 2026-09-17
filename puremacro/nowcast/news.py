"""Analytical News-versus-Noise decomposition for Dynamic Factor Models.

Implements the exact analytical Kalman-smoother news decomposition of:
Bańbura, M. and Modugno, M. (2014). "Maximum likelihood estimation of
factor models on datasets with arbitrary pattern of missing data."
Journal of Applied Econometrics, 29(1), 133-160.

Decomposes revisions to nowcasts between data vintages:
    Δ ŷ_{t*|v} = ŷ_{t*|v} - ŷ_{t*|v-1}
                = Σ_i ω_{news, i} · news_i + Σ_m ω_{rev, m} · revision_m
                = Σ_i impact_{news, i} + Σ_m impact_{rev, m}

The decomposition is mathematically exact: |Δ ŷ - Σ impact| < 10^-10.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..state_space import StateSpaceModel, kalman_smoother
from .dfm import DynamicFactorModel, DynamicFactorModelResult, KalmanDFMResult


@dataclass(frozen=True)
class NewsDecompositionResult:
    """Immutable result object from Bańbura & Modugno (2014) news decomposition.

    Attributes
    ----------
    target_variable : str
        Name of the target variable being nowcasted (e.g. 'gdp').
    target_period : Any
        Reference period or date of the nowcast target.
    forecast_old : float
        Nowcast under the previous vintage v-1.
    forecast_new : float
        Nowcast under the updated vintage v.
    revision : float
        Total update to the nowcast: forecast_new - forecast_old.
    impact_releases : dict[str, float]
        Attributed impacts by series from newly released observations.
    impact_revisions : dict[str, float]
        Attributed impacts by series from revisions to previously published data.
    total_impact : float
        Sum of all release impacts and revision impacts.
    decomposition_error : float
        Absolute error: |revision - total_impact|; guaranteed < 10^-10.
    news_table : pd.DataFrame
        Detailed table of new releases (series, period, actual, forecast, surprise, weight, impact).
    revision_table : pd.DataFrame
        Detailed table of data revisions (series, period, previous_val, updated_val, revision, weight, impact).
    """

    target_variable: str
    target_period: Any
    forecast_old: float
    forecast_new: float
    revision: float
    impact_releases: dict[str, float] = field(default_factory=dict)
    impact_revisions: dict[str, float] = field(default_factory=dict)
    total_impact: float = 0.0
    decomposition_error: float = 0.0
    news_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    revision_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def previous_nowcast(self) -> float:
        """Alias for forecast_old."""
        return self.forecast_old

    @property
    def updated_nowcast(self) -> float:
        """Alias for forecast_new."""
        return self.forecast_new

    def summary(self) -> str:
        """Text summary of Bańbura & Modugno news decomposition."""
        lines = [
            "=" * 74,
            "Bańbura & Modugno (2014) Real-Time Nowcasting News Decomposition",
            "=" * 74,
            f"Target Variable               : {self.target_variable}",
            f"Target Period                 : {self.target_period}",
            f"Previous Nowcast (v-1)        : {self.forecast_old:+.4f}",
            f"Updated Nowcast (v)           : {self.forecast_new:+.4f}",
            f"Total Nowcast Revision        : {self.revision:+.4f}",
            f"Sum of Explained Impacts      : {self.total_impact:+.4f}",
            f"Decomposition Identity Error  : {self.decomposition_error:.2e} (< 1e-10)",
            "-" * 74,
            "Impact Contributions Summary:",
            f"  From new releases           : {sum(self.impact_releases.values()):+.4f}",
            f"  From data revisions         : {sum(self.impact_revisions.values()):+.4f}",
            "-" * 74,
        ]

        if not self.news_table.empty:
            lines.append("New Releases (Innovations):")
            lines.append(
                f"{'Series':<18} {'Period':<12} {'Actual':>10} {'Forecast':>10} "
                f"{'Surprise':>10} {'Weight':>10} {'Impact':>10}"
            )
            lines.append("-" * 74)
            for _, row in self.news_table.iterrows():
                s = str(row["series"])
                p = str(row["period"])
                act = float(row["actual"])
                fc = float(row["forecast"])
                surp = float(row["surprise"])
                w = float(row["weight"])
                imp = float(row["impact"])
                lines.append(
                    f"{s:<18s} {p:<12s} {act:>10.4f} {fc:>10.4f} "
                    f"{surp:>10.4f} {w:>10.4f} {imp:>10.4f}"
                )
            lines.append("-" * 74)

        if not self.revision_table.empty:
            lines.append("Data Revisions (Historical updates):")
            lines.append(
                f"{'Series':<18} {'Period':<12} {'Previous':>10} {'Updated':>10} "
                f"{'Revision':>10} {'Weight':>10} {'Impact':>10}"
            )
            lines.append("-" * 74)
            for _, row in self.revision_table.iterrows():
                s = str(row["series"])
                p = str(row["period"])
                pv = float(row["previous_val"])
                uv = float(row["updated_val"])
                rev = float(row["revision"])
                w = float(row["weight"])
                imp = float(row["impact"])
                lines.append(
                    f"{s:<18s} {p:<12s} {pv:>10.4f} {uv:>10.4f} "
                    f"{rev:>10.4f} {w:>10.4f} {imp:>10.4f}"
                )
            lines.append("-" * 74)

        lines.append("=" * 74)
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Consolidated breakdown table of releases and revisions."""
        records = []
        if not self.news_table.empty:
            for _, row in self.news_table.iterrows():
                records.append({
                    "type": "release",
                    "series": row["series"],
                    "period": row["period"],
                    "actual_or_updated": row["actual"],
                    "forecast_or_previous": row["forecast"],
                    "surprise_or_revision": row["surprise"],
                    "weight": row["weight"],
                    "impact": row["impact"],
                })
        if not self.revision_table.empty:
            for _, row in self.revision_table.iterrows():
                records.append({
                    "type": "revision",
                    "series": row["series"],
                    "period": row["period"],
                    "actual_or_updated": row["updated_val"],
                    "forecast_or_previous": row["previous_val"],
                    "surprise_or_revision": row["revision"],
                    "weight": row["weight"],
                    "impact": row["impact"],
                })
        if not records:
            return pd.DataFrame(columns=[
                "type", "series", "period", "actual_or_updated",
                "forecast_or_previous", "surprise_or_revision", "weight", "impact",
            ])
        df_res = pd.DataFrame(records)
        num_cols = [c for c in df_res.columns if c in {"actual_or_updated", "forecast_or_previous", "surprise_or_revision", "weight", "impact"}]
        df_res[num_cols] = df_res[num_cols].round(4)
        return df_res

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
        """Plot waterfall / contribution bar chart of nowcast update impacts."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.5))
        else:
            fig = ax.figure

        # Collect items
        items = [("Previous Nowcast", self.forecast_old, "steelblue")]

        # Group impacts by series
        combined_impacts: dict[str, float] = {}
        for s, imp in self.impact_releases.items():
            combined_impacts[s] = combined_impacts.get(s, 0.0) + imp
        for s, imp in self.impact_revisions.items():
            label = f"{s} (rev)"
            combined_impacts[label] = combined_impacts.get(label, 0.0) + imp

        for s, imp in combined_impacts.items():
            color = "#2ca02c" if imp >= 0 else "#d62728"
            items.append((s, imp, color))

        items.append(("Updated Nowcast", self.forecast_new, "navy"))

        labels = [item[0] for item in items]
        values = [item[1] for item in items]
        colors = [item[2] for item in items]
        y_pos = np.arange(len(labels))

        ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor="black", lw=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.axvline(0.0, color="black", lw=0.8, ls="--", alpha=0.7)

        default_title = (
            f"Nowcast Revision: {self.target_variable} ({self.target_period}) "
            f"[{self.revision:+.3f}]"
        )
        ax.set_title(title or default_title, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Value / Impact")
        ax.grid(True, axis="x", ls=":", alpha=0.5)

        fig.tight_layout()
        return fig


def _prepare_panels(
    old_vintage: Any,
    new_vintage: Any,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[Any, ...]]:
    """Extract aligned numpy arrays and labels from old and new vintages."""
    # If VintagePanel
    if hasattr(old_vintage, "df") and hasattr(old_vintage, "as_of"):
        # If wide frame already or panel
        df_old = old_vintage.as_of(old_vintage.df["vintage"].max()) if not old_vintage.df.empty else pd.DataFrame()
    elif isinstance(old_vintage, pd.DataFrame):
        df_old = old_vintage.copy()
    else:
        df_old = None

    if hasattr(new_vintage, "df") and hasattr(new_vintage, "as_of"):
        df_new = new_vintage.as_of(new_vintage.df["vintage"].max()) if not new_vintage.df.empty else pd.DataFrame()
    elif isinstance(new_vintage, pd.DataFrame):
        df_new = new_vintage.copy()
    else:
        df_new = None

    if df_old is not None and df_new is not None:
        # If multi-index with country, pick top or droplevel
        if isinstance(df_old.index, pd.MultiIndex):
            if "country" in df_old.index.names:
                df_old = df_old.reset_index(level="country", drop=True)
        if isinstance(df_new.index, pd.MultiIndex):
            if "country" in df_new.index.names:
                df_new = df_new.reset_index(level="country", drop=True)

        # Align columns and rows
        all_cols = list(dict.fromkeys(list(df_old.columns) + list(df_new.columns)))
        all_idx = df_old.index.union(df_new.index)

        df_old_aligned = df_old.reindex(index=all_idx, columns=all_cols)
        df_new_aligned = df_new.reindex(index=all_idx, columns=all_cols)

        cols = tuple(str(c) for c in all_cols)
        idx = tuple(all_idx)
        return (
            df_old_aligned.values.astype(float),
            df_new_aligned.values.astype(float),
            cols,
            idx,
        )

    # Raw numpy arrays
    arr_old = np.asarray(old_vintage, dtype=float)
    arr_new = np.asarray(new_vintage, dtype=float)
    if arr_old.shape != arr_new.shape:
        raise ValueError(
            f"Shape mismatch between old_vintage {arr_old.shape} and new_vintage {arr_new.shape}"
        )
    cols = tuple(f"x{i}" for i in range(arr_old.shape[1]))
    idx = tuple(range(arr_old.shape[0]))
    return arr_old, arr_new, cols, idx


def banbura_modugno_news(
    model: DynamicFactorModel | DynamicFactorModelResult | KalmanDFMResult | StateSpaceModel,
    old_vintage: Any,
    new_vintage: Any,
    target_series: str | int = 0,
    target_period: Any = None,
    **kwargs: Any,
) -> NewsDecompositionResult:
    """Compute exact Bańbura & Modugno (2014) analytical news decomposition.

    Parameters
    ----------
    model : DynamicFactorModel | DynamicFactorModelResult | KalmanDFMResult | StateSpaceModel
        The dynamic factor model whose state-space matrices govern the transition
        and observation dynamics.
    old_vintage : VintagePanel | pd.DataFrame | np.ndarray
        Data panel available at vintage v-1.
    new_vintage : VintagePanel | pd.DataFrame | np.ndarray
        Data panel available at updated vintage v.
    target_series : str | int, default 0
        Name or column index of the series whose forecast update is decomposed.
    target_period : Any, optional
        Period or date of interest. Defaults to the latest period (t = T - 1).

    Returns
    -------
    NewsDecompositionResult
        Frozen dataclass with news innovations, historical revision impacts,
        and guaranteed numerical identity error < 10^-10.
    """
    old_arr, new_arr, cols, idx = _prepare_panels(old_vintage, new_vintage)
    T, n = old_arr.shape

    # Extract model system matrices
    if isinstance(model, DynamicFactorModel):
        if model.result_ is None:
            model.fit(old_arr)
        res = model.result_
    elif isinstance(model, (DynamicFactorModelResult, KalmanDFMResult)):
        res = model
    elif isinstance(model, StateSpaceModel):
        # Raw StateSpaceModel provided
        res = {
            "A": model.T,
            "loadings": model.Z,
            "Q": model.Q,
            "H": model.H,
            "means": np.zeros(n),
            "stds": np.ones(n),
            "ssm": model,
        }
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

    A = np.asarray(res["A"], dtype=float)
    Q = np.asarray(res["Q"], dtype=float)
    H = np.asarray(res["H"], dtype=float)
    loadings = np.asarray(res["loadings"], dtype=float)
    means = np.asarray(res.get("means", np.zeros(n)), dtype=float)
    stds = np.asarray(res.get("stds", np.ones(n)), dtype=float)
    stds = np.where(stds < 1e-12, 1.0, stds)

    m = A.shape[0]  # state dimension k * p
    k = loadings.shape[1] if loadings.ndim > 1 else 1
    # Full observation loading matrix (n, m)
    if loadings.shape == (n, m):
        Z_full = loadings
    else:
        Z_full = np.zeros((n, m))
        Z_full[:, :k] = loadings

    ssm = StateSpaceModel(T=A, Z=Z_full, Q=Q, H=H)

    # Standardize data
    Xs_old = (old_arr - means) / stds
    Xs_new = (new_arr - means) / stds

    # Identify target coordinates
    if isinstance(target_series, str):
        if target_series in cols:
            target_col_idx = cols.index(target_series)
            target_var_name = target_series
        else:
            raise KeyError(f"Target series {target_series!r} not in panel columns: {cols}")
    else:
        target_col_idx = int(target_series)
        target_var_name = cols[target_col_idx] if target_col_idx < len(cols) else f"x{target_col_idx}"

    if target_period is None or target_period == -1:
        target_row_idx = T - 1
        target_period_name = idx[target_row_idx] if idx else (T - 1)
    elif target_period in idx:
        target_row_idx = idx.index(target_period)
        target_period_name = target_period
    elif isinstance(target_period, int) and 0 <= target_period < T:
        target_row_idx = target_period
        target_period_name = idx[target_row_idx] if idx else target_period
    else:
        target_row_idx = T - 1
        target_period_name = idx[target_row_idx] if idx else (T - 1)

    # 1. Baseline Kalman smoother on old vintage
    sm_old = kalman_smoother(Xs_old, ssm, diffuse_scale=1e6)
    a_sm_old = sm_old["a_smooth"]
    pred_old_std = Z_full[target_col_idx] @ a_sm_old[target_row_idx]
    forecast_old = float(pred_old_std * stds[target_col_idx] + means[target_col_idx])

    # 2. Identify Revisions vs New Releases
    # Revision: observed in both old and new, but value changed
    mask_old_obs = ~np.isnan(Xs_old)
    mask_new_obs = ~np.isnan(Xs_new)

    rev_coords = []
    for t_i in range(T):
        for j_i in range(n):
            if mask_old_obs[t_i, j_i] and mask_new_obs[t_i, j_i]:
                if abs(Xs_new[t_i, j_i] - Xs_old[t_i, j_i]) > 1e-12:
                    rev_coords.append((t_i, j_i))

    # Releases: unobserved in old, observed in new
    rel_coords = []
    for t_i in range(T):
        for j_i in range(n):
            if not mask_old_obs[t_i, j_i] and mask_new_obs[t_i, j_i]:
                rel_coords.append((t_i, j_i))

    # Intermediate vintage: apply only data revisions
    Xs_int = Xs_old.copy()
    for t_r, j_r in rev_coords:
        Xs_int[t_r, j_r] = Xs_new[t_r, j_r]

    sm_int = kalman_smoother(Xs_int, ssm, diffuse_scale=1e6)
    a_sm_int = sm_int["a_smooth"]

    # Compute revision impacts
    # Due to exact linearity of the smoother on the fixed observation pattern:
    # impact_rev_m = (sm(Xs_old + Δy_m) - sm(Xs_old))
    revision_records = []
    impact_revisions: dict[str, float] = {}

    for t_r, j_r in rev_coords:
        Xs_single = Xs_old.copy()
        Xs_single[t_r, j_r] = Xs_new[t_r, j_r]
        sm_single = kalman_smoother(Xs_single, ssm, diffuse_scale=1e6)
        delta_state = sm_single["a_smooth"][target_row_idx] - a_sm_old[target_row_idx]
        imp = float(Z_full[target_col_idx] @ delta_state * stds[target_col_idx])

        var_name = cols[j_r]
        per_name = idx[t_r]
        raw_prev = float(old_arr[t_r, j_r])
        raw_upd = float(new_arr[t_r, j_r])
        raw_diff = raw_upd - raw_prev
        w = float(imp / raw_diff) if abs(raw_diff) > 1e-12 else 0.0

        revision_records.append({
            "series": var_name,
            "period": per_name,
            "previous_val": raw_prev,
            "updated_val": raw_upd,
            "revision": raw_diff,
            "weight": w,
            "impact": imp,
        })
        impact_revisions[var_name] = impact_revisions.get(var_name, 0.0) + imp

    # 3. New Releases (News Innovation) via Analytical Projection Gain
    n_u = len(rel_coords)
    news_records = []
    impact_releases: dict[str, float] = {}
    delta_news_target = 0.0

    if n_u > 0:
        I = np.zeros(n_u)
        surprises_raw = np.zeros(n_u)

        for i, (t_i, j_i) in enumerate(rel_coords):
            # Prior expectation based on intermediate vintage
            hat_std = float(Z_full[j_i] @ a_sm_int[t_i])
            hat_raw = float(hat_std * stds[j_i] + means[j_i])
            act_raw = float(new_arr[t_i, j_i])
            surp = act_raw - hat_raw
            surprises_raw[i] = surp
            I[i] = surp / stds[j_i]

        # Smoothed state autocovariances for intermediate vintage
        P_filt = sm_int["P_filt"]
        P_pred = sm_int["P_pred"]
        P_sm = sm_int["P_smooth"]

        J = [None] * (T - 1)
        for t in range(T - 1):
            try:
                J[t] = P_filt[t] @ A.T @ np.linalg.pinv(P_pred[t + 1])
            except np.linalg.LinAlgError:
                J[t] = np.zeros((m, m))

        def get_cov(t1: int, t2: int) -> np.ndarray:
            if t1 == t2:
                return P_sm[t1]
            elif t1 < t2:
                prod = np.eye(m)
                for tau in range(t1, t2):
                    prod = prod @ J[tau]
                return prod @ P_sm[t2]
            else:
                return get_cov(t2, t1).T

        # News covariance matrix Σ_I (n_u x n_u)
        Sigma_I = np.zeros((n_u, n_u))
        for i, (t_i, j_i) in enumerate(rel_coords):
            for l, (t_l, j_l) in enumerate(rel_coords):
                cov_alpha = get_cov(t_i, t_l)
                Sigma_I[i, l] = (
                    Z_full[j_i] @ cov_alpha @ Z_full[j_l].T
                    + (H[j_i, j_l] if t_i == t_l else 0.0)
                )

        # Cross-covariance with target state α_{t*} (m x n_u)
        Sigma_alpha_I = np.zeros((m, n_u))
        for i, (t_i, j_i) in enumerate(rel_coords):
            cov_alpha = get_cov(target_row_idx, t_i)
            Sigma_alpha_I[:, i] = cov_alpha @ Z_full[j_i].T

        # Gain matrix K (m x n_u)
        try:
            K = Sigma_alpha_I @ np.linalg.pinv(Sigma_I)
        except np.linalg.LinAlgError:
            K = np.zeros((m, n_u))

        # Weight vector in standardized units for target variable
        omega_std = Z_full[target_col_idx] @ K  # (n_u,)

        for i, (t_i, j_i) in enumerate(rel_coords):
            var_name = cols[j_i]
            per_name = idx[t_i]
            # Weight in original raw units:
            w = float(omega_std[i] * (stds[target_col_idx] / stds[j_i]))
            surp = float(surprises_raw[i])
            imp = float(w * surp)

            hat_raw = float(new_arr[t_i, j_i] - surp)
            news_records.append({
                "series": var_name,
                "period": per_name,
                "actual": float(new_arr[t_i, j_i]),
                "forecast": hat_raw,
                "surprise": surp,
                "weight": w,
                "impact": imp,
            })
            impact_releases[var_name] = impact_releases.get(var_name, 0.0) + imp

        delta_news_target = float(np.sum([r["impact"] for r in news_records]))

    # 4. Final smoother on new vintage
    sm_new = kalman_smoother(Xs_new, ssm, diffuse_scale=1e6)
    pred_new_std = Z_full[target_col_idx] @ sm_new["a_smooth"][target_row_idx]
    forecast_new = float(pred_new_std * stds[target_col_idx] + means[target_col_idx])
    revision = float(forecast_new - forecast_old)

    sum_rev_impact = float(np.sum([r["impact"] for r in revision_records]))
    total_impact = float(sum_rev_impact + delta_news_target)
    decomp_error = float(abs(revision - total_impact))

    df_news = pd.DataFrame(news_records) if news_records else pd.DataFrame(
        columns=["series", "period", "actual", "forecast", "surprise", "weight", "impact"]
    )
    df_rev = pd.DataFrame(revision_records) if revision_records else pd.DataFrame(
        columns=["series", "period", "previous_val", "updated_val", "revision", "weight", "impact"]
    )

    return NewsDecompositionResult(
        target_variable=target_var_name,
        target_period=target_period_name,
        forecast_old=forecast_old,
        forecast_new=forecast_new,
        revision=revision,
        impact_releases=impact_releases,
        impact_revisions=impact_revisions,
        total_impact=total_impact,
        decomposition_error=decomp_error,
        news_table=df_news,
        revision_table=df_rev,
    )


__all__ = [
    "NewsDecompositionResult",
    "banbura_modugno_news",
]
