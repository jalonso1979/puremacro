"""Panel-shaping helpers and trend/cycle filters for puremacro.

No I/O assumed: callers pass in DataFrames they already have. The
``hp_filter``, ``hamilton_filter``, and ``bk_filter`` helpers return
trend/cycle decompositions usable in teaching and applied work.

See also
--------
For data **fetchers** (not transforms), see :mod:`puremacro.fetch`
which exposes ``fetch_fred``, ``fetch_fred_alfred``, and the new
``sdmx_get`` (0.6.0+) for OECD/Eurostat/ECB/IMF SDMX-CSV endpoints.
This module (``puremacro.data``) only contains panel-level
transforms (long↔wide, country slicing, HP / Hamilton / BK filters).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import struct
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import pandas as pd

from ._unit_root_core import adf_test, kpss_test
from .reports import df_to_latex, df_to_markdown, df_to_typst
from .sa.stl import residual_seasonality_F

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_UNIT_COL_CANDIDATES = (
    "unit",
    "code",
    "country",
    "country_code",
    "iso3",
    "id",
    "fips",
    "county_fips",
    "state_fips",
    "state",
    "region",
    "entity",
    "group",
)
_TIME_COL_CANDIDATES = ("date", "time", "period", "quarter", "month", "year", "t")
_SA_MIN_OBS = {"Q": 10, "M": 26}


# ---------------------------------------------------------------------------
# Panel reshaping
# ---------------------------------------------------------------------------
def long_to_wide(panel_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long-form panel to a (code, date) x variable wide frame.

    ``panel_long`` must have columns ``code, date, variable, value``.
    """
    wide = (
        panel_long.pivot_table(
            index=["code", "date"], columns="variable", values="value"
        ).sort_index()
    )
    wide.columns.name = None
    return wide


def slice_country(wide: pd.DataFrame, code: str) -> pd.DataFrame:
    """Return the per-country wide slice with a sorted DatetimeIndex."""
    if code not in wide.index.get_level_values("code"):
        raise KeyError(f"country {code!r} not in panel")
    sub = wide.xs(code, level="code").copy()
    sub.index = pd.DatetimeIndex(sub.index)
    return sub.sort_index()


def align_yx(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Return ``(y, x)`` both non-null, inner-joined in time, within ``[start, end]``."""
    sub = df[[y_col, x_col]].dropna()
    if start is not None:
        sub = sub[sub.index >= pd.Period(start, freq="Q").to_timestamp()]
    if end is not None:
        sub = sub[sub.index <= pd.Period(end, freq="Q").to_timestamp(how="end")]
    return sub[y_col], sub[x_col]


# ---------------------------------------------------------------------------
# Trend / cycle filters
# ---------------------------------------------------------------------------
def hp_filter(y, lamb: float = 1600.0) -> tuple[pd.Series, pd.Series]:
    """Hodrick-Prescott filter via closed-form ``(I + lambda K' K)^{-1} y``.

    Default ``lamb=1600`` is the Ravn-Uhlig-style quarterly value.
    Returns ``(cycle, trend)`` as Series with the same index as ``y``.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    n = y_arr.size
    if n < 4:
        raise ValueError("hp_filter requires at least 4 observations")
    # Second-difference matrix K of shape (n-2, n).
    K = np.zeros((n - 2, n))
    for i in range(n - 2):
        K[i, i] = 1.0
        K[i, i + 1] = -2.0
        K[i, i + 2] = 1.0
    A = np.eye(n) + lamb * (K.T @ K)
    trend_arr = np.linalg.solve(A, y_arr)
    cycle_arr = y_arr - trend_arr
    if isinstance(y, pd.Series):
        idx = y.index
    else:
        idx = pd.RangeIndex(n)
    cycle = pd.Series(cycle_arr, index=idx, name="cycle")
    trend = pd.Series(trend_arr, index=idx, name="trend")
    return cycle, trend


def hamilton_filter(y, *, h: int = 8, p: int = 4) -> tuple[pd.Series, pd.Series]:
    """Hamilton (2018) regression filter.

    Regress ``y_{t+h}`` on a constant and ``y_t, y_{t-1}, ..., y_{t-(p-1)}``;
    the residual is the cycle, the fitted value is the trend. The first
    ``h + p - 1`` entries of the output are NaN by construction.

    Thin wrapper around :func:`puremacro.cycles.hamilton_filter`, which is
    the canonical implementation; this version always returns a pair of
    ``pandas.Series`` named ``"cycle"`` and ``"trend"`` carrying ``y``'s index
    (or a ``RangeIndex`` for array input). It raises ``ValueError`` when
    ``y`` is not finite or ``T < h + 2p + 1``.
    """
    from .cycles import hamilton_filter as _hamilton

    if isinstance(y, pd.Series):
        idx = y.index
        y_arr = y.values.astype(float)
    else:
        y_arr = np.asarray(y, dtype=float).ravel()
        idx = pd.RangeIndex(y_arr.size)
    res = _hamilton(y_arr, h=h, p=p)
    cycle = pd.Series(np.asarray(res.cycle, dtype=float), index=idx, name="cycle")
    trend = pd.Series(np.asarray(res.trend, dtype=float), index=idx, name="trend")
    return cycle, trend


def bk_filter(y, low: int = 6, high: int = 32, K: int = 12) -> pd.Series:
    """Baxter-King symmetric band-pass filter.

    Pass the cyclical band ``[low, high]`` (in periods) using truncation
    ``K`` on each side. Returns a Series with ``K`` NaN at each end.

    Weights:
        omega_l = 2 pi / high
        omega_h = 2 pi / low
        b_0 = (omega_h - omega_l) / pi
        b_h = (sin(h omega_h) - sin(h omega_l)) / (h pi),  h = 1..K
    Weights are normalised by subtracting their mean so they sum to zero.
    """
    if low <= 0 or high <= low:
        raise ValueError("bk_filter requires 0 < low < high")
    if isinstance(y, pd.Series):
        idx = y.index
        y_arr = y.values.astype(float)
    else:
        y_arr = np.asarray(y, dtype=float).ravel()
        idx = pd.RangeIndex(y_arr.size)
    n = y_arr.size
    if n <= 2 * K:
        raise ValueError("bk_filter: series shorter than 2K+1")
    omega_l = 2.0 * np.pi / high
    omega_h = 2.0 * np.pi / low
    b = np.zeros(K + 1)
    b[0] = (omega_h - omega_l) / np.pi
    for hk in range(1, K + 1):
        b[hk] = (np.sin(hk * omega_h) - np.sin(hk * omega_l)) / (hk * np.pi)
    weights = np.concatenate([b[:0:-1], b])  # length 2K+1
    weights = weights - weights.mean()
    out = np.full(n, np.nan)
    for t in range(K, n - K):
        out[t] = float(np.dot(weights, y_arr[t - K: t + K + 1]))
    return pd.Series(out, index=idx, name="cycle")


@dataclass(frozen=True)
class PreflightFinding:
    """One structured diagnostic finding emitted by :func:`preflight_panel`.

    Attributes
    ----------
    check : str
        Category identifier (e.g. ``'duplicates'``, ``'time_gaps'``,
        ``'missingness'``, ``'unit_root'``, ``'seasonality'``).
    severity : {"info", "warning", "error"}
        Severity level; any ``'error'`` finding marks the report as failed.
    variable : str | None
        Affected variable name, or ``None`` for panel-wide checks.
    unit : str | None
        Representative affected unit code, if applicable.
    message : str
        Human-readable summary of the detected issue.
    recommendation : str
        Suggested remediation action prior to estimation.
    metric : float | None
        Key quantitative metric associated with the finding.
    """

    check: str
    severity: Literal["info", "warning", "error"]
    variable: str | None = None
    unit: str | None = None
    message: str = ""
    recommendation: str = ""
    metric: float | None = None


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        fval = float(value)
        return fval if np.isfinite(fval) else None
    if pd.isna(value):
        return None
    return value


@dataclass(frozen=True)
class PreflightReport:
    """Comprehensive panel data readiness, stationarity, and seasonality report.

    Attributes
    ----------
    panel_shape : tuple[int, int]
        Row and column count of the input DataFrame.
    n_units : int
        Number of distinct cross-sectional units (1 for a single time series).
    n_periods : int
        Number of distinct valid time periods observed across the panel.
    unit_col : str | None
        Name of the resolved unit identifier column, if any.
    time_col : str
        Name of the resolved time identifier column.
    inferred_freq : str | None
        Inferred or explicitly supplied frequency code (``'Q'``, ``'M'``, ``'A'``, etc.).
    date_range : tuple[str, str] | None
        Formatted ``(min_time, max_time)`` bounds of the valid time index.
    balanced : bool
        True when duplicate-free, gap-free, and every unit covers the same grid.
    duplicate_count : int
        Number of duplicate ``(unit, time)`` keys detected.
    n_errors : int
        Count of findings with ``severity == 'error'``.
    n_warnings : int
        Count of findings with ``severity == 'warning'``.
    summary_table : pd.DataFrame
        Single-row high-level metadata and health summary DataFrame.
    variable_table : pd.DataFrame
        Per-variable completeness, variance, unit-root, and seasonality summary.
    findings : tuple[PreflightFinding, ...]
        Ordered diagnostic findings.
    recommendations : tuple[str, ...]
        Deduplicated actionable recommendations derived from findings.
    data_sha256 : str
        Deterministic SHA-256 fingerprint over sorted canonical keys and columns.
    digits : int
        Default rounding precision for text/table exports.
    """

    panel_shape: tuple[int, int]
    n_units: int
    n_periods: int
    unit_col: str | None
    time_col: str
    inferred_freq: str | None
    date_range: tuple[str, str] | None
    balanced: bool
    duplicate_count: int
    n_errors: int
    n_warnings: int
    summary_table: pd.DataFrame
    variable_table: pd.DataFrame
    findings: tuple[PreflightFinding, ...]
    recommendations: tuple[str, ...]
    data_sha256: str
    digits: int = 3

    @property
    def passed(self) -> bool:
        """True when zero error-severity findings occurred."""
        return self.n_errors == 0

    @property
    def status(self) -> Literal["passed", "warn", "failed"]:
        """Coarse status label: ``'failed'``, ``'warn'``, or ``'passed'``."""
        if self.n_errors > 0:
            return "failed"
        if self.n_warnings > 0:
            return "warn"
        return "passed"

    def to_frame(
        self,
        layout: Literal["variables", "findings", "summary"] = "variables",
    ) -> pd.DataFrame:
        """Return diagnostic table in ``'variables'``, ``'findings'``, or ``'summary'`` layout."""
        if layout == "variables":
            return self.variable_table.copy()
        if layout == "findings":
            rows = [
                {
                    "check": f.check,
                    "severity": f.severity,
                    "variable": f.variable,
                    "unit": f.unit,
                    "message": f.message,
                    "recommendation": f.recommendation,
                    "metric": f.metric,
                }
                for f in self.findings
            ]
            cols = ["check", "severity", "variable", "unit", "message", "recommendation", "metric"]
            return pd.DataFrame(rows, columns=cols)
        if layout == "summary":
            return self.summary_table.copy()
        raise ValueError(
            f"Unknown layout {layout!r}; expected 'variables', 'findings', or 'summary'"
        )

    def to_dataframe(
        self,
        layout: Literal["variables", "findings", "summary"] = "variables",
    ) -> pd.DataFrame:
        """Alias for :meth:`to_frame`."""
        return self.to_frame(layout=layout)

    def summary(self) -> str:
        """Return a concise text summary of panel readiness diagnostics."""
        dr = f"{self.date_range[0]}..{self.date_range[1]}" if self.date_range else "unknown"
        lines = [
            (
                f"Panel Preflight [{self.status.upper()}] "
                f"shape={self.panel_shape}, units={self.n_units}, periods={self.n_periods}, "
                f"freq={self.inferred_freq}, balanced={self.balanced}, "
                f"duplicates={self.duplicate_count}, errors={self.n_errors}, warnings={self.n_warnings}"
            ),
            f"Range: {dr} | sha256={self.data_sha256[:16]}...",
            self.variable_table.to_string(index=False),
        ]
        if self.findings:
            lines.append("Findings:")
            lines.extend(
                f"  - [{f.severity.upper()}] {f.check}"
                + (f" ({f.variable})" if f.variable else "")
                + f": {f.message}"
                for f in self.findings
            )
        if self.recommendations:
            lines.append("Recommendations:")
            lines.extend(f"  - {r}" for r in self.recommendations)
        return "\n".join(lines)

    def to_markdown(
        self,
        layout: Literal["variables", "findings", "summary"] = "variables",
        digits: int | None = None,
    ) -> str:
        """Render the selected report layout as GitHub-Flavored Markdown."""
        d = self.digits if digits is None else int(digits)
        return df_to_markdown(self.to_frame(layout=layout), index=False, digits=d)

    def to_latex(
        self,
        layout: Literal["variables", "findings", "summary"] = "variables",
        digits: int | None = None,
    ) -> str:
        """Render the selected report layout as LaTeX tabular markup."""
        d = self.digits if digits is None else int(digits)
        return df_to_latex(self.to_frame(layout=layout), index=False, digits=d)

    def to_typst(
        self,
        layout: Literal["variables", "findings", "summary"] = "variables",
        digits: int | None = None,
    ) -> str:
        """Render the selected report layout as a Typst ``#table``."""
        d = self.digits if digits is None else int(digits)
        return df_to_typst(self.to_frame(layout=layout), index=False, digits=d)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report, variable table, findings, and sha256 to JSON."""
        payload = {
            "status": self.status,
            "passed": self.passed,
            "panel_shape": list(self.panel_shape),
            "n_units": self.n_units,
            "n_periods": self.n_periods,
            "unit_col": self.unit_col,
            "time_col": self.time_col,
            "inferred_freq": self.inferred_freq,
            "date_range": list(self.date_range) if self.date_range is not None else None,
            "balanced": self.balanced,
            "duplicate_count": self.duplicate_count,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "data_sha256": self.data_sha256,
            "summary_table": self.summary_table.to_dict(orient="records"),
            "variable_table": self.variable_table.to_dict(orient="records"),
            "findings": [
                {
                    "check": f.check,
                    "severity": f.severity,
                    "variable": f.variable,
                    "unit": f.unit,
                    "message": f.message,
                    "recommendation": f.recommendation,
                    "metric": f.metric,
                }
                for f in self.findings
            ],
            "recommendations": list(self.recommendations),
        }
        return json.dumps(_json_sanitize(payload), indent=indent, allow_nan=False)

    def plot(
        self,
        metric: Literal["missing_pct", "n_valid", "adf_median_p", "kpss_median_p"] = "missing_pct",
        *,
        ax: Axes | None = None,
        title: str | None = None,
    ) -> Figure | Axes:
        """Plot per-variable preflight metric bars across audited columns."""
        import matplotlib.pyplot as plt

        if metric not in self.variable_table.columns:
            raise ValueError(f"Unknown metric {metric!r}; available: {list(self.variable_table.columns)}")
        fig: Figure | None = None
        if ax is None:
            n_vars = max(len(self.variable_table), 1)
            fig, plot_ax = plt.subplots(figsize=(max(5.5, 1.2 * n_vars), 3.8))
        else:
            plot_ax = ax

        labels = [str(v) for v in self.variable_table["variable"]]
        vals = self.variable_table[metric].to_numpy(dtype=float)
        plot_vals = np.where(np.isfinite(vals), vals, 0.0)
        x = np.arange(len(labels), dtype=float)
        plot_ax.bar(x, plot_vals)
        plot_ax.set_xticks(x)
        plot_ax.set_xticklabels(labels)
        plot_ax.set_ylabel(metric)
        plot_ax.set_title(title if title is not None else f"Panel Preflight: {metric}")

        if fig is not None:
            fig.tight_layout()
            return fig
        return plot_ax


def _find_candidate_col(
    columns: Sequence[Any],
    candidates: Sequence[str],
    exclude: set[str],
) -> str | None:
    lower_map = {str(c).lower(): str(c) for c in columns if str(c) not in exclude}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def _resolve_panel_columns(
    df: pd.DataFrame,
    *,
    unit_col: str | None,
    unit: str | None,
    time_col: str | None,
    time: str | None,
    variables: Sequence[str] | None,
) -> tuple[pd.DataFrame, str | None, str, list[str]]:
    if unit_col is not None and unit is not None and unit_col != unit:
        raise ValueError("Pass only one of 'unit_col' or 'unit', or pass identical names")
    if time_col is not None and time is not None and time_col != time:
        raise ValueError("Pass only one of 'time_col' or 'time', or pass identical names")
    req_unit = unit_col if unit_col is not None else unit
    req_time = time_col if time_col is not None else time

    work = df.copy()
    if (
        isinstance(work.index, pd.MultiIndex)
        and req_unit is None
        and req_time is None
        and work.index.nlevels >= 2
    ):
        idx_names = [str(n) if n is not None else f"level_{i}" for i, n in enumerate(work.index.names)]
        work.index = work.index.set_names(idx_names)
        work = work.reset_index()
        req_unit = idx_names[0]
        req_time = idx_names[1]
    elif isinstance(work.index, (pd.DatetimeIndex, pd.PeriodIndex)) and req_time is None:
        t_name = str(work.index.name) if work.index.name is not None else "date"
        if t_name in work.columns:
            req_time = t_name
        else:
            work.index = work.index.rename(t_name)
            work = work.reset_index()
            req_time = t_name

    col_strs = [str(c) for c in work.columns]
    if req_unit is not None:
        if req_unit not in col_strs:
            raise ValueError(f"Unit column {req_unit!r} not found in DataFrame columns")
        resolved_unit: str | None = req_unit
    else:
        resolved_unit = _find_candidate_col(col_strs, _UNIT_COL_CANDIDATES, {req_time} if req_time else set())

    if req_time is not None:
        if req_time not in col_strs:
            raise ValueError(f"Time column {req_time!r} not found in DataFrame columns")
        resolved_time = req_time
    else:
        detected_time = _find_candidate_col(
            col_strs, _TIME_COL_CANDIDATES, {resolved_unit} if resolved_unit else set()
        )
        if detected_time is None:
            raise ValueError("Could not infer time column; pass time_col= or use a DatetimeIndex")
        resolved_time = detected_time

    if variables is not None:
        var_cols = [str(v) for v in variables]
        for v in var_cols:
            if v not in col_strs:
                raise ValueError(f"Variable {v!r} not found in DataFrame columns")
            if not pd.api.types.is_numeric_dtype(work[v]) or pd.api.types.is_bool_dtype(work[v]):
                raise ValueError(f"Variable {v!r} must be numeric for preflight diagnostics")
    else:
        excluded = {resolved_time}
        if resolved_unit is not None:
            excluded.add(resolved_unit)
        var_cols = [
            c
            for c in col_strs
            if c not in excluded
            and pd.api.types.is_numeric_dtype(work[c])
            and not pd.api.types.is_bool_dtype(work[c])
        ]
    if not var_cols:
        raise ValueError("No numeric variables available for panel preflight")
    return work, resolved_unit, resolved_time, var_cols


def _datetime_ns_array(dt_series: pd.Series) -> np.ndarray:
    """Convert a Series of timezone-naive timestamps cleanly to int64 ns across pandas 2/3."""
    dti = pd.DatetimeIndex(dt_series)
    if hasattr(dti, "as_unit"):
        dti = dti.as_unit("ns")
    return np.asarray(dti.asi8, dtype=np.int64)


def _coerce_time_column(series: pd.Series) -> tuple[pd.Series, bool]:
    """Parse ``series`` into normalized timestamps or numeric time coordinates."""
    if isinstance(series.dtype, pd.PeriodDtype):
        dt = series.dt.to_timestamp()
        if dt.dt.tz is not None:
            dt = dt.dt.tz_localize(None)
        return dt, True
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = pd.to_datetime(series, errors="coerce")
        if dt.dt.tz is not None:
            dt = dt.dt.tz_localize(None)
        return dt, True
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        return pd.to_numeric(series, errors="coerce"), False

    n_present = int(series.notna().sum())
    if n_present == 0:
        raise ValueError("Time column contains only missing values")

    num = pd.to_numeric(series, errors="coerce")
    if int(num.notna().sum()) == n_present:
        return num, False

    try:
        parsed = pd.to_datetime(series, format="mixed", errors="coerce")
        if int(parsed.notna().sum()) >= max(1, (n_present + 1) // 2):
            if parsed.dt.tz is not None:
                parsed = parsed.dt.tz_localize(None)
            return parsed, True
    except Exception:
        pass
    try:
        per = pd.PeriodIndex(series.astype(str), freq="Q").to_timestamp()
        return pd.Series(per, index=series.index), True
    except Exception:
        pass
    if num.notna().any():
        return num, False
    raise ValueError(f"Could not parse time column values: sample={series.head(3).tolist()!r}")


def _canonical_freq_label(raw_freq: str | None) -> str | None:
    if raw_freq is None:
        return None
    f = str(raw_freq).upper().strip()
    if f.startswith("Q"):
        return "Q"
    if f.startswith("M"):
        return "M"
    if f.startswith(("A", "Y")):
        return "A"
    if f.startswith("W"):
        return "W"
    if f.startswith("D") or f == "B":
        return "D"
    return str(raw_freq)


def _infer_freq_and_step(
    unique_sorted: np.ndarray,
    *,
    is_datetime: bool,
    requested_freq: str | None,
) -> tuple[str | None, float | None]:
    explicit = _canonical_freq_label(requested_freq)
    if len(unique_sorted) < 2:
        default_steps = {"D": 1.0, "W": 7.0, "M": 30.4, "Q": 91.25, "A": 365.25}
        return explicit, default_steps.get(explicit or "", None)

    if is_datetime:
        ns_arr = np.asarray(unique_sorted, dtype=np.int64)
        diffs_ns = (ns_arr[1:] - ns_arr[:-1]).astype(np.float64)
        pos_days = diffs_ns / (86_400.0 * 1e9)
        pos_days = pos_days[pos_days > 0.0]
        modal_days = (
            float(Counter(np.round(pos_days, 1)).most_common(1)[0][0]) if pos_days.size > 0 else None
        )
        if explicit is not None:
            return explicit, modal_days
        inferred_raw: str | None = None
        if len(unique_sorted) >= 3:
            try:
                inferred_raw = pd.infer_freq(pd.DatetimeIndex(unique_sorted))
            except Exception:
                inferred_raw = None
        if inferred_raw is not None:
            return _canonical_freq_label(inferred_raw), modal_days
        if modal_days is not None:
            if 27.0 <= modal_days <= 32.0:
                return "M", modal_days
            if 88.0 <= modal_days <= 93.0:
                return "Q", modal_days
            if 358.0 <= modal_days <= 367.0:
                return "A", modal_days
            if 6.5 <= modal_days <= 7.5:
                return "W", modal_days
            if 0.9 <= modal_days <= 1.1:
                return "D", modal_days
            return f"step:{modal_days:g}d", modal_days
        return None, None

    pos_steps = np.diff(unique_sorted.astype(float))
    pos_steps = pos_steps[pos_steps > 0.0]
    modal_step = (
        float(Counter(np.round(pos_steps, 6)).most_common(1)[0][0]) if pos_steps.size > 0 else None
    )
    if explicit is not None:
        return explicit, modal_step
    return (f"numeric:{modal_step:g}" if modal_step is not None else None), modal_step


def _count_unit_gaps(
    sorted_vals: np.ndarray,
    *,
    is_datetime: bool,
    freq_label: str | None,
    modal_step: float | None,
) -> int:
    if sorted_vals.size < 2:
        return 0
    if is_datetime and freq_label in ("Q", "M", "A"):
        ords = pd.DatetimeIndex(sorted_vals).to_period(freq_label).asi8
        diffs = np.diff(ords)
        return int(np.sum(np.maximum(diffs - 1, 0)))
    if is_datetime and modal_step is not None and modal_step > 0.0:
        days = sorted_vals.astype(float) / (86_400.0 * 1e9)
        diffs = np.diff(days)
        return int(
            np.sum(
                np.where(diffs > 1.5 * modal_step, np.maximum(np.rint(diffs / modal_step).astype(int) - 1, 0), 0)
            )
        )
    if not is_datetime and modal_step is not None and modal_step > 0.0:
        diffs = np.diff(sorted_vals.astype(float))
        return int(
            np.sum(
                np.where(diffs > 1.5 * modal_step, np.maximum(np.rint(diffs / modal_step).astype(int) - 1, 0), 0)
            )
        )
    return 0


def _compute_panel_sha256(
    df_sorted: pd.DataFrame,
    cols: Sequence[str],
    parsed_time: pd.Series,
    time_col: str,
    is_datetime: bool,
) -> str:
    """Compute a deterministic endian-neutral SHA-256 over canonical rows/columns."""
    h = hashlib.sha256()
    h.update(struct.pack("<II", len(df_sorted), len(cols)))
    for col in cols:
        name_bytes = str(col).encode("utf-8")
        h.update(struct.pack("<I", len(name_bytes)))
        h.update(name_bytes)
        if col == time_col and is_datetime:
            h.update(b"dt64ns")
            arr_ns = _datetime_ns_array(parsed_time.loc[df_sorted.index])
            h.update(np.ascontiguousarray(arr_ns).astype("<i8").tobytes())
        elif (
            (pd.api.types.is_bool_dtype(df_sorted[col]) or pd.api.types.is_integer_dtype(df_sorted[col]))
            and not df_sorted[col].isna().any()
        ):
            h.update(b"i8")
            arr_i = df_sorted[col].to_numpy(dtype=np.int64)
            h.update(np.ascontiguousarray(arr_i).astype("<i8").tobytes())
        elif pd.api.types.is_numeric_dtype(df_sorted[col]):
            h.update(b"f8")
            arr_f = df_sorted[col].to_numpy(dtype=np.float64, na_value=np.nan).copy()
            arr_f = np.where(arr_f == 0.0, 0.0, arr_f)
            arr_f = np.where(np.isnan(arr_f), np.nan, arr_f)
            h.update(np.ascontiguousarray(arr_f).astype("<f8").tobytes())
        else:
            h.update(b"utf8")
            for val in df_sorted[col].tolist():
                if pd.isna(val):
                    h.update(b"\x00" + struct.pack("<I", 0))
                else:
                    vb = str(val).encode("utf-8")
                    h.update(b"\x01" + struct.pack("<I", len(vb)) + vb)
    return h.hexdigest()


def _unit_seasonality_series(
    u_vals: np.ndarray,
    u_times: np.ndarray | None,
    *,
    is_datetime: bool,
    sa_freq: str,
) -> pd.Series:
    if not is_datetime or u_times is None:
        return pd.Series(u_vals)
    dti = pd.DatetimeIndex(u_times).to_period(sa_freq).to_timestamp()
    ser = pd.Series(u_vals, index=dti)
    ser = ser[~ser.index.duplicated(keep="last")].sort_index()
    full_idx = pd.date_range(ser.index.min(), ser.index.max(), freq="QS" if sa_freq == "Q" else "MS")
    return ser.reindex(full_idx)


def preflight_panel(
    df: pd.DataFrame,
    *,
    unit_col: str | None = None,
    unit: str | None = None,
    time_col: str | None = None,
    time: str | None = None,
    variables: Sequence[str] | None = None,
    freq: str | None = None,
    min_obs: int = 12,
    max_missing_share: float = 0.25,
    alpha: float = 0.05,
    run_stationarity: bool = True,
    run_seasonality: bool = True,
    digits: int = 3,
) -> PreflightReport:
    """Audit a panel or time-series DataFrame before estimation.

    Checks schema, duplicate keys, irregular or missing time steps, missingness,
    zero-variance units, unit roots (ADF + KPSS), and quarterly/monthly residual
    seasonality using Pyodide-safe primitives.

    Parameters
    ----------
    df : pd.DataFrame
        Input long/panel DataFrame, MultiIndex ``(unit, date)`` frame, or single
        time-series DataFrame.
    unit_col, unit : str, optional
        Column identifying cross-sectional units (aliases of each other).
    time_col, time : str, optional
        Column identifying time periods (aliases of each other).
    variables : Sequence[str], optional
        Numeric variable columns to audit; defaults to all non-key numeric columns.
    freq : str, optional
        Optional explicit frequency override (e.g. ``'Q'``, ``'M'``, ``'A'``).
    min_obs : int, default 12
        Minimum finite observations per unit before running ADF/KPSS diagnostics.
    max_missing_share : float, default 0.25
        Warning threshold on variable missingness fraction.
    alpha : float, default 0.05
        Significance level used for ADF, KPSS, and residual-seasonality flags.
    run_stationarity : bool, default True
        Whether to evaluate per-unit ADF and KPSS tests.
    run_seasonality : bool, default True
        Whether to test quarterly/monthly variables for residual seasonality.
    digits : int, default 3
        Default decimal precision for table renders.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    if min_obs < 6:
        raise ValueError("min_obs must be at least 6")
    if not 0.0 <= max_missing_share <= 1.0:
        raise ValueError("max_missing_share must lie in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    work, resolved_unit, resolved_time, var_cols = _resolve_panel_columns(
        df, unit_col=unit_col, unit=unit, time_col=time_col, time=time, variables=variables
    )
    parsed_time, is_datetime = _coerce_time_column(work[resolved_time])
    work = work.copy()
    work["_pm_time_key"] = _datetime_ns_array(parsed_time) if is_datetime else parsed_time

    sort_keys = ([resolved_unit] if resolved_unit is not None else []) + ["_pm_time_key"] + var_cols
    work_sorted = work.sort_values(sort_keys, kind="mergesort")
    parsed_sorted = parsed_time.loc[work_sorted.index]

    canon_cols = ([resolved_unit] if resolved_unit is not None else []) + [resolved_time] + var_cols
    data_sha256 = _compute_panel_sha256(
        work_sorted, canon_cols, parsed_sorted, resolved_time, is_datetime
    )

    findings: list[PreflightFinding] = []
    n_missing_time = int(parsed_sorted.isna().sum())
    if n_missing_time > 0:
        findings.append(
            PreflightFinding(
                check="missing_time",
                severity="error",
                message=f"{n_missing_time} row(s) have unparseable or missing time values.",
                recommendation="Drop or repair missing time stamps before estimation.",
                metric=float(n_missing_time),
            )
        )

    valid_mask = parsed_sorted.notna()
    work_valid = work_sorted.loc[valid_mask]
    if work_valid.empty:
        raise ValueError("No valid time values found in panel")

    key_subset = [resolved_unit, "_pm_time_key"] if resolved_unit is not None else ["_pm_time_key"]
    duplicate_count = int(work_valid.duplicated(subset=key_subset).sum())
    if duplicate_count > 0:
        findings.append(
            PreflightFinding(
                check="duplicates",
                severity="error",
                message=f"Found {duplicate_count} duplicate (unit, time) observation(s).",
                recommendation="Deduplicate or aggregate repeated unit-period observations.",
                metric=float(duplicate_count),
            )
        )

    unique_times = np.sort(work_valid["_pm_time_key"].unique())
    n_periods = int(unique_times.size)
    freq_label, modal_step = _infer_freq_and_step(
        unique_times, is_datetime=is_datetime, requested_freq=freq
    )

    if is_datetime:
        dt_min = pd.Timestamp(int(unique_times[0])).strftime("%Y-%m-%d")
        dt_max = pd.Timestamp(int(unique_times[-1])).strftime("%Y-%m-%d")
        date_range: tuple[str, str] | None = (dt_min, dt_max)
    else:
        date_range = (str(unique_times[0]), str(unique_times[-1]))

    unit_groups: list[tuple[str, pd.DataFrame]]
    if resolved_unit is not None:
        unit_groups = [(str(u), g) for u, g in work_valid.groupby(resolved_unit, sort=True)]
    else:
        unit_groups = [("series", work_valid)]
    n_units = len(unit_groups)

    total_gaps = 0
    gap_units: list[str] = []
    unit_lengths: set[int] = set()
    for u_name, u_df in unit_groups:
        u_unique = np.sort(u_df["_pm_time_key"].unique())
        unit_lengths.add(int(u_unique.size))
        ug = _count_unit_gaps(
            u_unique, is_datetime=is_datetime, freq_label=freq_label, modal_step=modal_step
        )
        if ug > 0:
            total_gaps += ug
            gap_units.append(u_name)

    if total_gaps > 0:
        findings.append(
            PreflightFinding(
                check="time_gaps",
                severity="warning",
                unit=gap_units[0] if gap_units else None,
                message=f"Detected {total_gaps} interior time gap(s) across {len(gap_units)} unit(s).",
                recommendation="Check missing calendar periods or align lags on an explicit time grid.",
                metric=float(total_gaps),
            )
        )

    balanced = (
        duplicate_count == 0
        and n_missing_time == 0
        and total_gaps == 0
        and len(unit_lengths) == 1
        and next(iter(unit_lengths)) == n_periods
    )
    if not balanced and total_gaps == 0 and duplicate_count == 0:
        findings.append(
            PreflightFinding(
                check="unbalanced_panel",
                severity="info",
                message="Panel units span different start/end dates or observation counts.",
                recommendation="Verify sample window consistency if estimation assumes a balanced panel.",
                metric=float(n_units),
            )
        )

    sa_freq: Literal["Q", "M"] | None = (
        "Q" if freq_label == "Q" else ("M" if freq_label == "M" else None)
    )
    var_rows: list[dict[str, Any]] = []

    for var in var_cols:
        raw_vals = pd.to_numeric(work_sorted[var], errors="coerce").to_numpy(dtype=float)
        n_obs = int(raw_vals.size)
        n_inf = int(np.sum(np.isinf(raw_vals)))
        n_nan = int(np.sum(np.isnan(raw_vals)))
        finite_vals = raw_vals[np.isfinite(raw_vals)]
        n_valid = int(finite_vals.size)
        missing_pct = 100.0 * (n_nan + n_inf) / max(n_obs, 1)
        std_val = float(np.std(finite_vals)) if n_valid >= 2 else (0.0 if n_valid == 1 else np.nan)
        mean_val = float(np.mean(finite_vals)) if n_valid > 0 else np.nan

        if n_inf > 0:
            findings.append(
                PreflightFinding(
                    check="non_finite",
                    severity="error",
                    variable=var,
                    message=f"Variable {var!r} contains {n_inf} +/-inf value(s).",
                    recommendation=f"Replace infinite values in {var!r} with NaN or clip outliers.",
                    metric=float(n_inf),
                )
            )
        if n_valid == 0 or (np.isfinite(std_val) and std_val <= 1e-12):
            findings.append(
                PreflightFinding(
                    check="zero_variance",
                    severity="error",
                    variable=var,
                    message=f"Variable {var!r} has no finite variation across the panel.",
                    recommendation=f"Drop or replace constant/empty variable {var!r}.",
                    metric=0.0 if not np.isfinite(std_val) else float(std_val),
                )
            )

        missing_share = (n_nan + n_inf) / max(n_obs, 1)
        if missing_share > max_missing_share:
            findings.append(
                PreflightFinding(
                    check="missingness",
                    severity="warning",
                    variable=var,
                    message=f"Variable {var!r} missing {missing_pct:.1f}% (> {100.0 * max_missing_share:.1f}% threshold).",
                    recommendation=f"Inspect sample attrition or impute/drop {var!r}.",
                    metric=float(missing_share),
                )
            )

        const_units: list[str] = []
        adf_ps: list[float] = []
        kpss_ps: list[float] = []
        seasonal_ps: list[float] = []
        joint_ur_units = 0
        seasonal_units = 0

        for u_name, u_df in unit_groups:
            u_raw = pd.to_numeric(u_df[var], errors="coerce").to_numpy(dtype=float)
            fin_mask = np.isfinite(u_raw)
            u_fin = u_raw[fin_mask]
            if u_fin.size == 0:
                continue
            if u_fin.size >= 2 and float(np.std(u_fin)) <= 1e-12:
                const_units.append(u_name)
                continue

            if run_stationarity and u_fin.size >= min_obs:
                adf_p = np.nan
                kpss_p = np.nan
                try:
                    adf_p = float(adf_test(u_fin, regression="c")["p_value"])
                except (ValueError, np.linalg.LinAlgError):
                    adf_p = np.nan
                try:
                    kpss_lags = (
                        int(np.ceil(4 * (u_fin.size / 100.0) ** 0.25))
                        if u_fin.size < 150
                        else None
                    )
                    kpss_p = float(kpss_test(u_fin, regression="c", lags=kpss_lags)["p_value"])
                except (ValueError, np.linalg.LinAlgError):
                    kpss_p = np.nan
                if np.isfinite(adf_p):
                    adf_ps.append(adf_p)
                if np.isfinite(kpss_p):
                    kpss_ps.append(kpss_p)
                if np.isfinite(adf_p) and np.isfinite(kpss_p) and adf_p >= alpha and kpss_p < alpha:
                    joint_ur_units += 1

            if run_seasonality and sa_freq is not None and u_fin.size >= _SA_MIN_OBS[sa_freq]:
                u_times = u_df["_pm_time_key"].to_numpy()[fin_mask] if is_datetime else None
                sa_ser = _unit_seasonality_series(
                    u_fin, u_times, is_datetime=is_datetime, sa_freq=sa_freq
                )
                _, p_seas, _, _ = residual_seasonality_F(sa_ser, freq=sa_freq)
                if np.isfinite(p_seas):
                    seasonal_ps.append(float(p_seas))
                    if p_seas < alpha:
                        seasonal_units += 1

        if const_units and (n_valid > 0 and np.isfinite(std_val) and std_val > 1e-12):
            findings.append(
                PreflightFinding(
                    check="constant_unit",
                    severity="warning",
                    variable=var,
                    unit=const_units[0],
                    message=f"{len(const_units)} unit(s) have zero within-unit variance in {var!r}.",
                    recommendation=f"Drop zero-variance units or avoid within-unit fixed effects for {var!r}.",
                    metric=float(len(const_units)),
                )
            )

        adf_med = float(np.median(adf_ps)) if adf_ps else np.nan
        kpss_med = float(np.median(kpss_ps)) if kpss_ps else np.nan
        seas_min_p = float(np.min(seasonal_ps)) if seasonal_ps else np.nan

        if adf_ps and kpss_ps and (
            joint_ur_units >= max(1, (min(len(adf_ps), len(kpss_ps)) + 1) // 2)
            or (adf_med >= alpha and kpss_med < alpha)
        ):
            findings.append(
                PreflightFinding(
                    check="unit_root",
                    severity="warning",
                    variable=var,
                    message=(
                        f"{var!r} shows non-stationary unit-root evidence "
                        f"(median ADF p={adf_med:.3f}, KPSS p={kpss_med:.3f})."
                    ),
                    recommendation=f"Consider first-differencing, growth rates, or detrending {var!r}.",
                    metric=float(adf_med),
                )
            )

        if seasonal_ps and seasonal_units > 0 and (seasonal_units / len(seasonal_ps) >= 0.25):
            findings.append(
                PreflightFinding(
                    check="seasonality",
                    severity="warning",
                    variable=var,
                    message=(
                        f"{var!r} exhibits residual {sa_freq} seasonality in "
                        f"{seasonal_units}/{len(seasonal_ps)} unit(s) (min p={seas_min_p:.4g})."
                    ),
                    recommendation=f"Seasonally adjust {var!r} via puremacro.sa.deseasonalize(..., freq={sa_freq!r}).",
                    metric=float(seas_min_p),
                )
            )

        var_rows.append({
            "variable": var,
            "n_obs": n_obs,
            "n_valid": n_valid,
            "n_missing": n_nan + n_inf,
            "missing_pct": missing_pct,
            "mean": mean_val,
            "std": std_val,
            "n_constant_units": len(const_units),
            "n_units_tested": len(adf_ps),
            "adf_median_p": adf_med,
            "kpss_median_p": kpss_med,
            "seasonality_min_p": seas_min_p,
            "n_seasonal_units": seasonal_units,
        })

    n_errors = sum(1 for f in findings if f.severity == "error")
    n_warnings = sum(1 for f in findings if f.severity == "warning")
    rec_seen: list[str] = []
    for f in findings:
        if f.recommendation and f.recommendation not in rec_seen:
            rec_seen.append(f.recommendation)

    summary_table = pd.DataFrame([{
        "rows": df.shape[0],
        "cols": df.shape[1],
        "n_units": n_units,
        "n_periods": n_periods,
        "freq": freq_label,
        "balanced": balanced,
        "duplicate_count": duplicate_count,
        "n_errors": n_errors,
        "n_warnings": n_warnings,
        "data_sha256": data_sha256,
    }])

    return PreflightReport(
        panel_shape=(int(df.shape[0]), int(df.shape[1])),
        n_units=n_units,
        n_periods=n_periods,
        unit_col=resolved_unit,
        time_col=resolved_time,
        inferred_freq=freq_label,
        date_range=date_range,
        balanced=balanced,
        duplicate_count=duplicate_count,
        n_errors=n_errors,
        n_warnings=n_warnings,
        summary_table=summary_table,
        variable_table=pd.DataFrame(var_rows),
        findings=tuple(findings),
        recommendations=tuple(rec_seen),
        data_sha256=data_sha256,
        digits=digits,
    )


__all__ = [
    "PreflightFinding",
    "PreflightReport",
    "align_yx",
    "bk_filter",
    "hamilton_filter",
    "hp_filter",
    "long_to_wide",
    "preflight_panel",
    "slice_country",
]
