"""Automated LatAm & macro freshness monitor workflow (Track 4A).

Composes connector freshness auditing (:mod:`puremacro.fetch.freshness`),
catalog provenance (:mod:`puremacro.fetch.registry`), panel preflight
diagnostics (:func:`puremacro.data.preflight_panel`), country indicator &
rolling AR(1) nowcast summaries, and bounded local-projection transmission
sensitivity (:func:`puremacro.lp.sensitivity.lp_sensitivity_grid`) into an
offline-first, fail-closed, bilingual (``en`` / ``es``) monitoring dossier and
UTF-8 artifact bundle.

Pyodide safety: module top-level imports are restricted to the standard library,
``numpy``, and ``pandas``. Internal imports from ``puremacro.fetch.*``,
``puremacro.data``, ``puremacro.lp.*``, ``puremacro.reports``, and
``puremacro.workbench`` are deferred inside functions so importing
:mod:`puremacro.macro_monitor` never loads ``requests``, ``statsmodels``,
``linearmodels``, ``arch``, ``sqlite3``, or ``ssl``.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

__all__ = [
    "MacroMonitorConfig",
    "MacroMonitorReport",
    "run_macro_monitor",
    "main",
]

_VALID_MODES: tuple[str, ...] = ("recorded", "cache", "live")
_VALID_LANGS: tuple[str, ...] = ("en", "es")
_VALID_FORMATS: tuple[str, ...] = ("text", "markdown", "json", "html")

_UNIT_CANDIDATES: tuple[str, ...] = (
    "code",
    "country",
    "iso3",
    "iso",
    "unit",
    "entity",
    "id",
    "region",
    "state",
)
_TIME_CANDIDATES: tuple[str, ...] = (
    "date",
    "time",
    "quarter",
    "period",
    "year",
    "month",
    "obs_date",
)
_QUARTER_RE = re.compile(r"^\s*(\d{4})[-/\s]?[Qq]([1-4])\s*$")
_INT_YEAR_RE = re.compile(r"^\s*(\d{4})\s*$")
_HOST_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"/private/var/[^\s'\",;)\]}]+"),
    re.compile(r"/var/folders/[^\s'\",;)\]}]+"),
    re.compile(r"/Users/[^\s'\",;)\]}]+"),
    re.compile(r"/home/[^\s'\",;)\]}]+"),
    re.compile(r"/tmp/[^\s'\",;)\]}]+"),
    re.compile(r"[A-Za-z]:\\[^\s'\",;)\]}]+"),
)
_HTML_STYLE = """
:root {
  --bg: #f6f8fb;
  --card: #ffffff;
  --border: #d8e0ea;
  --text: #16202a;
  --muted: #526171;
  --accent: #184e77;
  --ok-bg: #e7f6ec;
  --ok-fg: #146c2e;
  --warn-bg: #fff6db;
  --warn-fg: #8a5a00;
  --fail-bg: #fdeaea;
  --fail-fg: #9b1c1c;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 24px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
}
.container { max-width: 1120px; margin: 0 auto; }
.header, .card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 18px 20px;
  margin-bottom: 16px;
}
.header h1 { margin: 0 0 8px 0; font-size: 1.45rem; color: var(--accent); }
.subtitle { margin: 0; color: var(--muted); font-size: 0.94rem; }
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-weight: 600;
  font-size: 0.82rem;
  letter-spacing: 0.02em;
}
.badge-ok { background: var(--ok-bg); color: var(--ok-fg); }
.badge-warn { background: var(--warn-bg); color: var(--warn-fg); }
.badge-fail { background: var(--fail-bg); color: var(--fail-fg); }
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
  gap: 12px;
  margin-top: 14px;
}
.kpi {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  background: #fbfcfe;
}
.kpi-label { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; }
.kpi-value { font-size: 1.12rem; font-weight: 600; margin-top: 2px; }
h2 { margin: 0 0 10px 0; font-size: 1.1rem; color: var(--accent); }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.87rem; }
th, td {
  text-align: left;
  padding: 7px 10px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
th { background: #f1f5f9; font-weight: 600; }
ul.notices { margin: 6px 0 0 18px; padding: 0; }
.notice-warn {
  border-left: 4px solid #d97706;
  background: var(--warn-bg);
  padding: 10px 14px;
  border-radius: 6px;
}
.notice-fail {
  border-left: 4px solid #dc2626;
  background: var(--fail-bg);
  padding: 10px 14px;
  border-radius: 6px;
}
.footer { font-size: 0.82rem; color: var(--muted); text-align: right; }
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.88em;
}
""".strip()


def _normalize_lang(lang: str | None) -> str:
    value = "en" if lang is None else str(lang).strip().lower()
    if value not in _VALID_LANGS:
        raise ValueError(f"Unsupported language {lang!r}; expected one of {_VALID_LANGS}.")
    return value


def _normalize_str_tuple(value: Sequence[str] | str | None, *, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    else:
        items = [str(part).strip() for part in value if str(part).strip()]
    if not items:
        raise ValueError(f"{name} must not be empty when specified.")
    return tuple(items)


def _normalize_int_tuple(value: Sequence[int], *, name: str) -> tuple[int, ...]:
    items = tuple(int(v) for v in value)
    if not items:
        raise ValueError(f"{name} must contain at least one integer.")
    return items


def _sanitize_text(text: Any, *, extra_paths: Sequence[str | Path | None] = ()) -> str:
    out = "" if text is None else str(text)
    for raw in extra_paths:
        if raw is None:
            continue
        p = str(raw).strip()
        if len(p) > 3 and p in out:
            out = out.replace(p, Path(p).name or "<path>")
    for pattern in _HOST_PATH_PATTERNS:
        out = pattern.sub("<local-path>", out)
    return out


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        buf = getattr(sys.stdout, "buffer", None)
        payload = (text + "\n").encode(enc, errors="replace")
        if buf is not None:
            buf.write(payload)
            buf.flush()
        else:
            print(payload.decode(enc, errors="replace"))


def _json_safe(value: Any) -> Any:
    """Convert objects cleanly to RFC 8259 JSON-safe primitives without NaN/Inf."""
    from puremacro.workbench import _json_sanitize

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.strftime("%Y-%m-%d")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    sanitized = _json_sanitize(value)
    if isinstance(sanitized, float) and not math.isfinite(sanitized):
        return None
    return sanitized


def _df_to_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    records: list[dict[str, Any]] = []
    for raw in df.to_dict(orient="records"):
        records.append({str(k): _json_safe(v) for k, v in raw.items()})
    return records


def _parse_time_series(series: pd.Series) -> tuple[pd.Series, int]:
    """Parse date/quarter/year values without mutating row order; count failures."""
    if series.empty:
        return pd.Series([], dtype="datetime64[ns]", index=series.index), 0

    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        return parsed, int(parsed.isna().sum())

    if isinstance(series.dtype, pd.PeriodDtype):
        parsed = series.dt.to_timestamp()
        return parsed, int(parsed.isna().sum())

    if pd.api.types.is_integer_dtype(series):
        vals = series.astype("Int64")
        valid_year = vals.between(1800, 2150, inclusive="both")
        parsed_arr: list[pd.Timestamp] = []
        for idx_val, is_yr in zip(vals.tolist(), valid_year.tolist()):
            if pd.isna(idx_val) or not bool(is_yr):
                parsed_arr.append(pd.NaT)
            else:
                parsed_arr.append(pd.Timestamp(year=int(idx_val), month=1, day=1))
        parsed = pd.Series(parsed_arr, index=series.index, dtype="datetime64[ns]")
        return parsed, int(parsed.isna().sum())

    raw_strs = series.astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    missing_mask = series.isna() | raw_strs.eq("") | raw_strs.str.lower().isin({"nan", "nat", "none", "null"})

    # Fast vectorized ISO/general datetime pass first, then quarterly/integer fallbacks.
    non_missing = ~missing_mask
    if non_missing.any():
        try:
            iso_try = pd.to_datetime(series.loc[non_missing], errors="coerce", format="mixed")
        except TypeError:
            iso_try = pd.to_datetime(series.loc[non_missing], errors="coerce")
        parsed.loc[non_missing] = iso_try

        need_fallback = non_missing & parsed.isna()
        for idx in series.index[need_fallback]:
            token = raw_strs.loc[idx]
            m_q = _QUARTER_RE.match(token)
            if m_q is not None:
                yr, q = int(m_q.group(1)), int(m_q.group(2))
                parsed.loc[idx] = pd.Timestamp(year=yr, month=3 * (q - 1) + 1, day=1)
                continue
            m_y = _INT_YEAR_RE.match(token)
            if m_y is not None:
                yr = int(m_y.group(1))
                if 1800 <= yr <= 2150:
                    parsed.loc[idx] = pd.Timestamp(year=yr, month=1, day=1)

    n_invalid = int((parsed.isna() | missing_mask).sum())
    return parsed, n_invalid


def _resolve_columns(
    df: pd.DataFrame,
    *,
    unit_col: str | None,
    time_col: str | None,
) -> tuple[str | None, str | None, list[str]]:
    errors: list[str] = []
    lower_map = {str(col).lower(): str(col) for col in df.columns}

    resolved_unit: str | None = None
    if unit_col is not None:
        if unit_col in df.columns:
            resolved_unit = unit_col
        elif unit_col.lower() in lower_map:
            resolved_unit = lower_map[unit_col.lower()]
        else:
            errors.append(f"missing_required_column: unit column {unit_col!r} not found.")
    else:
        for cand in _UNIT_CANDIDATES:
            if cand in lower_map:
                resolved_unit = lower_map[cand]
                break
        if resolved_unit is None:
            errors.append(
                f"missing_required_column: no unit/country column found (tried {_UNIT_CANDIDATES})."
            )

    resolved_time: str | None = None
    if time_col is not None:
        if time_col in df.columns:
            resolved_time = time_col
        elif time_col.lower() in lower_map:
            resolved_time = lower_map[time_col.lower()]
        else:
            errors.append(f"missing_required_column: time column {time_col!r} not found.")
    else:
        for cand in _TIME_CANDIDATES:
            if cand in lower_map:
                resolved_time = lower_map[cand]
                break
        if resolved_time is None:
            errors.append(
                f"missing_required_column: no date/time column found (tried {_TIME_CANDIDATES})."
            )

    return resolved_unit, resolved_time, errors


def _build_default_recorded_panel() -> tuple[pd.DataFrame, str, str]:
    """Load bundled QNA panel and derive clean quarterly macro growth series."""
    data_path = Path(__file__).resolve().parent / "replication" / "data" / "qna40_panel.csv"
    raw_bytes = data_path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    raw = pd.read_csv(io.BytesIO(raw_bytes))

    df = raw[["code", "date", "gdp_real", "gdp_defl", "cons_hh_real", "exports_real"]].copy()
    df["code"] = df["code"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)

    level_to_growth = (
        ("gdp_real", "gdp_growth"),
        ("gdp_defl", "inflation_qoq"),
        ("cons_hh_real", "cons_growth"),
        ("exports_real", "export_growth"),
    )
    for src_col, dst_col in level_to_growth:
        vals = pd.to_numeric(df[src_col], errors="coerce")
        log_vals = pd.Series(np.where(vals > 0.0, np.log(vals), np.nan), index=df.index)
        df[dst_col] = 100.0 * log_vals.groupby(df["code"], sort=False).diff()

    # Keep rows with core GDP growth & deflator inflation; drop the leading diff row.
    panel = df[["code", "date", "gdp_growth", "inflation_qoq", "cons_growth", "export_growth"]].copy()
    panel = panel.dropna(subset=["gdp_growth", "inflation_qoq"]).reset_index(drop=True)
    panel["date"] = panel["date"].dt.strftime("%Y-%m-%d")
    return panel, "recorded:qna40_panel", sha256


@dataclass(frozen=True)
class MacroMonitorConfig:
    """Configuration for :func:`run_macro_monitor`."""

    mode: str = "recorded"
    sources: tuple[str, ...] | None = None
    as_of: str | None = None
    db_path: str | None = None
    max_requests: int = 5
    min_interval_seconds: float = 0.5
    timeout_seconds: float = 20.0
    panel_builder: str | None = None
    unit_col: str | None = None
    time_col: str | None = None
    variables: tuple[str, ...] | None = None
    y: str | None = None
    x: str | None = None
    controls: tuple[str, ...] = ()
    horizons: tuple[int, ...] = (0, 1, 2, 3, 4)
    lags: int = 2
    lp_lags: tuple[int, ...] = (1, 2)
    ci: float = 0.90
    vol_window: int = 8
    zscore_threshold: float = 2.0
    short_sample_min_obs: int = 24
    run_lp_sensitivity: bool = True
    freq: str | None = None
    lang: str = "en"
    digits: int = 3
    fail_on_outage: bool = False

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"Unsupported mode {self.mode!r}; expected one of {_VALID_MODES}.")
        norm_lang = _normalize_lang(self.lang)
        object.__setattr__(self, "lang", norm_lang)

        if self.sources is not None:
            object.__setattr__(self, "sources", _normalize_str_tuple(self.sources, name="sources"))
        if self.variables is not None:
            object.__setattr__(
                self, "variables", _normalize_str_tuple(self.variables, name="variables")
            )
        if self.controls:
            object.__setattr__(
                self,
                "controls",
                _normalize_str_tuple(self.controls, name="controls") or (),
            )
        else:
            object.__setattr__(self, "controls", ())

        horizons = _normalize_int_tuple(self.horizons, name="horizons")
        if min(horizons) < 0:
            raise ValueError("horizons must be non-negative.")
        object.__setattr__(self, "horizons", horizons)

        lp_lags = _normalize_int_tuple(self.lp_lags, name="lp_lags")
        if min(lp_lags) < 0:
            raise ValueError("lp_lags must be non-negative.")
        object.__setattr__(self, "lp_lags", lp_lags)

        if int(self.lags) < 0:
            raise ValueError("lags must be >= 0.")
        object.__setattr__(self, "lags", int(self.lags))

        if int(self.max_requests) <= 0:
            raise ValueError("max_requests must be > 0.")
        object.__setattr__(self, "max_requests", int(self.max_requests))

        if float(self.min_interval_seconds) < 0.0:
            raise ValueError("min_interval_seconds must be >= 0.")
        if float(self.timeout_seconds) <= 0.0:
            raise ValueError("timeout_seconds must be > 0.")
        if not (0.0 < float(self.ci) < 1.0):
            raise ValueError("ci must be strictly between 0 and 1.")
        if int(self.vol_window) < 2:
            raise ValueError("vol_window must be >= 2.")
        if float(self.zscore_threshold) <= 0.0:
            raise ValueError("zscore_threshold must be > 0.")
        if int(self.short_sample_min_obs) < 4:
            raise ValueError("short_sample_min_obs must be >= 4.")
        if int(self.digits) < 0 or int(self.digits) > 8:
            raise ValueError("digits must be between 0 and 8.")
        if self.db_path is not None:
            object.__setattr__(self, "db_path", str(self.db_path))


@dataclass(frozen=True)
class MacroMonitorReport:
    """Immutable monitoring dossier returned by :func:`run_macro_monitor`."""

    config: MacroMonitorConfig
    lang: str
    mode: str
    as_of: str
    overall_status: str
    passed: bool
    has_outages: bool
    connectors_ok: int
    connectors_total: int
    outage_connectors: tuple[str, ...]
    stale_connectors: tuple[str, ...]
    preflight_status: str
    preflight_errors: int
    preflight_warnings: int
    input_source: str
    input_sha256: str
    sample_start: str
    sample_end: str
    n_countries: int
    n_rows: int
    countries: tuple[str, ...]
    target_y: str | None
    impulse_x: str | None
    n_lp_specs_ok: int
    n_lp_specs_failed: int
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    _freshness_df: pd.DataFrame = field(repr=False)
    _registry_df: pd.DataFrame = field(repr=False)
    _preflight_variables_df: pd.DataFrame = field(repr=False)
    _latest_indicators_df: pd.DataFrame = field(repr=False)
    _nowcast_df: pd.DataFrame = field(repr=False)
    _lp_metrics_df: pd.DataFrame = field(repr=False)
    _lp_table_df: pd.DataFrame = field(repr=False)
    _status_counts: dict[str, int] = field(repr=False)
    _preflight_summary: dict[str, Any] = field(repr=False)
    _lp_summary: dict[str, Any] = field(repr=False)
    _rate_budget: dict[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lang", _normalize_lang(self.lang))
        object.__setattr__(self, "outage_connectors", tuple(self.outage_connectors))
        object.__setattr__(self, "stale_connectors", tuple(self.stale_connectors))
        object.__setattr__(self, "countries", tuple(self.countries))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))

        for attr in (
            "_freshness_df",
            "_registry_df",
            "_preflight_variables_df",
            "_latest_indicators_df",
            "_nowcast_df",
            "_lp_metrics_df",
            "_lp_table_df",
        ):
            df = getattr(self, attr)
            object.__setattr__(self, attr, df.copy(deep=True).reset_index(drop=True))

        object.__setattr__(self, "_status_counts", dict(self._status_counts))
        object.__setattr__(self, "_preflight_summary", copy.deepcopy(self._preflight_summary))
        object.__setattr__(self, "_lp_summary", copy.deepcopy(self._lp_summary))
        object.__setattr__(self, "_rate_budget", copy.deepcopy(self._rate_budget))

    @property
    def freshness_df(self) -> pd.DataFrame:
        return self._freshness_df.copy(deep=True)

    @property
    def registry_df(self) -> pd.DataFrame:
        return self._registry_df.copy(deep=True)

    @property
    def preflight_variables_df(self) -> pd.DataFrame:
        return self._preflight_variables_df.copy(deep=True)

    @property
    def latest_indicators_df(self) -> pd.DataFrame:
        return self._latest_indicators_df.copy(deep=True)

    @property
    def nowcast_df(self) -> pd.DataFrame:
        return self._nowcast_df.copy(deep=True)

    @property
    def lp_metrics_df(self) -> pd.DataFrame:
        return self._lp_metrics_df.copy(deep=True)

    @property
    def lp_table_df(self) -> pd.DataFrame:
        return self._lp_table_df.copy(deep=True)

    @property
    def status_counts(self) -> dict[str, int]:
        return dict(self._status_counts)

    @property
    def preflight_summary(self) -> dict[str, Any]:
        return copy.deepcopy(self._preflight_summary)

    @property
    def lp_summary(self) -> dict[str, Any]:
        return copy.deepcopy(self._lp_summary)

    @property
    def rate_budget(self) -> dict[str, Any]:
        return copy.deepcopy(self._rate_budget)

    def to_frame(self, table: str = "latest_indicators") -> pd.DataFrame:
        """Return a deep copy of one of the report's structured tables."""
        mapping = {
            "latest_indicators": self._latest_indicators_df,
            "freshness": self._freshness_df,
            "registry": self._registry_df,
            "preflight": self._preflight_variables_df,
            "nowcast": self._nowcast_df,
            "lp_metrics": self._lp_metrics_df,
            "lp_table": self._lp_table_df,
        }
        if table not in mapping:
            raise ValueError(f"Unknown table {table!r}; expected one of {tuple(mapping)}.")
        return mapping[table].copy(deep=True)

    def headline(self, *, lang: str | None = None) -> str:
        """Single-line bilingual headline including outage/preflight/LP status."""
        use_lang = _normalize_lang(self.lang if lang is None else lang)
        flags: list[str] = []
        if self.outage_connectors:
            outage_list = ",".join(self.outage_connectors)
            if use_lang == "es":
                flags.append(f"caídas={outage_list}")
            else:
                flags.append(f"outages={outage_list}")
        if self.stale_connectors:
            stale_list = ",".join(self.stale_connectors)
            if use_lang == "es":
                flags.append(f"desactualizados={stale_list}")
            else:
                flags.append(f"stale={stale_list}")
        if self.errors:
            if use_lang == "es":
                flags.append(f"errores={len(self.errors)}")
            else:
                flags.append(f"errors={len(self.errors)}")

        flag_str = f" | {'; '.join(flags)}" if flags else ""
        if use_lang == "es":
            return (
                f"[{self.overall_status}] Monitor Macro LatAm ({self.mode}, as_of={self.as_of}): "
                f"conectores {self.connectors_ok}/{self.connectors_total}, "
                f"preflight={self.preflight_status}, países={self.n_countries}, "
                f"especificaciones LP={self.n_lp_specs_ok}{flag_str}"
            )
        return (
            f"[{self.overall_status}] LatAm & Macro Monitor ({self.mode}, as_of={self.as_of}): "
            f"connectors {self.connectors_ok}/{self.connectors_total}, "
            f"preflight={self.preflight_status}, countries={self.n_countries}, "
            f"LP specs={self.n_lp_specs_ok}{flag_str}"
        )

    def summary(self, *, lang: str | None = None) -> str:
        """Concise multi-line bilingual executive summary."""
        use_lang = _normalize_lang(self.lang if lang is None else lang)
        lines = [self.headline(lang=use_lang)]
        if use_lang == "es":
            lines.append(
                f"Fuente: {self.input_source} | SHA-256: {self.input_sha256[:16]}... | "
                f"Muestra: {self.sample_start} .. {self.sample_end} ({self.n_rows} filas)"
            )
        else:
            lines.append(
                f"Input: {self.input_source} | SHA-256: {self.input_sha256[:16]}... | "
                f"Sample: {self.sample_start} .. {self.sample_end} ({self.n_rows} rows)"
            )
        if self.errors:
            label = "Errores" if use_lang == "es" else "Errors"
            lines.append(f"{label}: " + " ; ".join(self.errors[:5]))
        if self.warnings:
            label = "Advertencias" if use_lang == "es" else "Warnings"
            lines.append(f"{label}: " + " ; ".join(self.warnings[:5]))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to an RFC 8259 JSON-safe dictionary (zero host paths)."""
        payload: dict[str, Any] = {
            "schema": "puremacro.macro_monitor.report.v1",
            "lang": self.lang,
            "mode": self.mode,
            "as_of": self.as_of,
            "overall_status": self.overall_status,
            "passed": bool(self.passed),
            "has_outages": bool(self.has_outages),
            "headline": self.headline(lang=self.lang),
            "connectors": {
                "ok": int(self.connectors_ok),
                "total": int(self.connectors_total),
                "outage_connectors": list(self.outage_connectors),
                "stale_connectors": list(self.stale_connectors),
                "status_counts": {str(k): int(v) for k, v in self._status_counts.items()},
                "rate_budget": _json_safe(self._rate_budget),
                "records": _df_to_safe_records(self._freshness_df),
            },
            "registry_catalog": {
                "total_sources": int(len(self._registry_df)),
                "records": _df_to_safe_records(self._registry_df),
            },
            "panel": {
                "input_source": self.input_source,
                "input_sha256": self.input_sha256,
                "sample_start": self.sample_start,
                "sample_end": self.sample_end,
                "n_countries": int(self.n_countries),
                "n_rows": int(self.n_rows),
                "countries": list(self.countries),
                "target_y": self.target_y,
                "impulse_x": self.impulse_x,
                "preflight_status": self.preflight_status,
                "preflight_errors": int(self.preflight_errors),
                "preflight_warnings": int(self.preflight_warnings),
                "preflight_summary": _json_safe(self._preflight_summary),
                "preflight_variables": _df_to_safe_records(self._preflight_variables_df),
            },
            "indicators": _df_to_safe_records(self._latest_indicators_df),
            "nowcast": _df_to_safe_records(self._nowcast_df),
            "lp_sensitivity": {
                "n_specs_ok": int(self.n_lp_specs_ok),
                "n_specs_failed": int(self.n_lp_specs_failed),
                "summary": _json_safe(self._lp_summary),
                "metrics": _df_to_safe_records(self._lp_metrics_df),
                "table": _df_to_safe_records(self._lp_table_df),
            },
            "warnings": [str(w) for w in self.warnings],
            "errors": [str(e) for e in self.errors],
        }
        return _json_safe(payload)

    def to_json(self, *, indent: int = 2) -> str:
        """Render the report as strict UTF-8 JSON (NaN/Infinity converted to null)."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False) + "\n"

    def to_markdown(self, *, lang: str | None = None) -> str:
        """Render a bilingual Markdown monitoring dossier."""
        from puremacro.reports import df_to_markdown

        use_lang = _normalize_lang(self.lang if lang is None else lang)
        es = use_lang == "es"
        digits = self.config.digits

        title = (
            "# Monitor Automatizado de Actualidad Macro y LatAm"
            if es
            else "# Automated LatAm & Macro Freshness Monitor"
        )
        lines: list[str] = [title, "", f"**{self.headline(lang=use_lang)}**", ""]

        overview_rows = pd.DataFrame(
            [
                {
                    ("Campo" if es else "Field"): ("Estado global" if es else "Overall status"),
                    ("Valor" if es else "Value"): self.overall_status,
                },
                {
                    ("Campo" if es else "Field"): ("Modo / Fecha de corte" if es else "Mode / As-of"),
                    ("Valor" if es else "Value"): f"{self.mode} / {self.as_of}",
                },
                {
                    ("Campo" if es else "Field"): ("Conectores activos" if es else "Healthy connectors"),
                    ("Valor" if es else "Value"): f"{self.connectors_ok}/{self.connectors_total}",
                },
                {
                    ("Campo" if es else "Field"): ("Fuente del panel" if es else "Panel input"),
                    ("Valor" if es else "Value"): f"{self.input_source} (sha256:{self.input_sha256[:16]})",
                },
                {
                    ("Campo" if es else "Field"): ("Rango muestral" if es else "Sample span"),
                    ("Valor" if es else "Value"): f"{self.sample_start} .. {self.sample_end} ({self.n_countries} / {self.n_rows})",
                },
                {
                    ("Campo" if es else "Field"): ("Preflight estructural" if es else "Structural preflight"),
                    ("Valor" if es else "Value"): f"{self.preflight_status} (errors={self.preflight_errors}, warnings={self.preflight_warnings})",
                },
                {
                    ("Campo" if es else "Field"): ("Sensibilidad LP" if es else "LP sensitivity"),
                    ("Valor" if es else "Value"): (
                        f"y={self.target_y}, x={self.impulse_x} | ok={self.n_lp_specs_ok}, failed={self.n_lp_specs_failed}"
                        if self.target_y and self.impulse_x
                        else f"ok={self.n_lp_specs_ok}, failed={self.n_lp_specs_failed}"
                    ),
                },
            ]
        )
        lines.append("## " + ("Resumen Ejecutivo" if es else "Executive Overview"))
        lines.append(df_to_markdown(overview_rows, index=False, digits=digits))
        lines.append("")

        if self.errors or self.warnings:
            lines.append("## " + ("Alertas, Fallos y Advertencias" if es else "Alerts, Failures & Warnings"))
            for err in self.errors:
                prefix = "ERROR" if not es else "ERROR"
                lines.append(f"- **{prefix}**: {err}")
            for warn in self.warnings:
                prefix = "WARN" if not es else "AVISO"
                lines.append(f"- **{prefix}**: {warn}")
            lines.append("")

        lines.append("## " + ("1. Actualidad y Procedencia de Conectores" if es else "1. Connector Freshness & Provenance"))
        if not self._freshness_df.empty:
            cols = [
                c
                for c in (
                    "source_id",
                    "provider",
                    "status",
                    "mode",
                    "n_obs",
                    "obs_end",
                    "staleness_days",
                    "fallback_used",
                    "error",
                )
                if c in self._freshness_df.columns
            ]
            lines.append(df_to_markdown(self._freshness_df[cols], index=False, digits=digits))
        else:
            lines.append("_" + ("Sin registros de conectores." if es else "No connector records.") + "_")
        lines.append("")

        lines.append("## " + ("2. Diagnóstico Estructural del Panel (Preflight)" if es else "2. Panel Structural Preflight"))
        if not self._preflight_variables_df.empty:
            lines.append(df_to_markdown(self._preflight_variables_df, index=False, digits=digits))
        else:
            lines.append("_" + ("Tabla de variables no disponible por fallo de ingesta." if es else "Variable diagnostics unavailable due to ingestion failure.") + "_")
        lines.append("")

        lines.append("## " + ("3. Indicadores Recientes y Banderas por País" if es else "3. Latest Country Indicators & Anomaly Flags"))
        if not self._latest_indicators_df.empty:
            lines.append(df_to_markdown(self._latest_indicators_df, index=False, digits=digits))
        else:
            lines.append("_" + ("Sin indicadores calculados." if es else "No country indicators computed.") + "_")
        lines.append("")

        lines.append("## " + ("4. Diagnóstico de Nowcast AR(1)" if es else "4. Rolling AR(1) Nowcast Diagnostic"))
        if not self._nowcast_df.empty:
            lines.append(df_to_markdown(self._nowcast_df, index=False, digits=digits))
        else:
            lines.append("_" + ("Sin estimaciones AR(1) disponibles." if es else "No AR(1) nowcasts available.") + "_")
        lines.append("")

        lines.append("## " + ("5. Malla de Sensibilidad de Proyecciones Locales (LP)" if es else "5. Local Projection Sensitivity Grid"))
        if not self._lp_metrics_df.empty:
            lines.append(df_to_markdown(self._lp_metrics_df, index=False, digits=digits))
        else:
            lines.append("_" + ("Malla LP omitida o inestable." if es else "LP sensitivity grid skipped or unstable.") + "_")
        lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def to_html(self, *, lang: str | None = None) -> str:
        """Render a self-contained UTF-8 HTML monitoring dossier (zero external assets)."""
        use_lang = _normalize_lang(self.lang if lang is None else lang)
        es = use_lang == "es"
        digits = self.config.digits

        badge_cls = "badge-ok"
        if self.overall_status == "WARN":
            badge_cls = "badge-warn"
        elif self.overall_status in ("DEGRADED", "FAILED"):
            badge_cls = "badge-fail"

        def _render_table(df: pd.DataFrame, empty_msg: str) -> str:
            if df.empty:
                return f"<p class='subtitle'>{html.escape(empty_msg)}</p>"
            headers = "".join(f"<th>{html.escape(str(col))}</th>" for col in df.columns)
            rows_html: list[str] = []
            for row in df.itertuples(index=False, name=None):
                cells: list[str] = []
                for val in row:
                    if val is None or (isinstance(val, float) and not math.isfinite(val)):
                        txt = ""
                    elif isinstance(val, float):
                        txt = f"{val:.{digits}f}"
                    else:
                        txt = str(val)
                    cells.append(f"<td>{html.escape(txt)}</td>")
                rows_html.append("<tr>" + "".join(cells) + "</tr>")
            body = "".join(rows_html)
            return f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div>"

        title = (
            "Monitor Automatizado de Actualidad Macro y LatAm"
            if es
            else "Automated LatAm & Macro Freshness Monitor"
        )
        notice_block = ""
        if self.errors or self.warnings:
            notice_cls = "notice-fail" if self.errors else "notice-warn"
            notice_title = "Alertas y Diagnósticos" if es else "Alerts & Diagnostics"
            items = "".join(
                f"<li><strong>ERROR:</strong> {html.escape(e)}</li>" for e in self.errors
            ) + "".join(
                f"<li><strong>{'AVISO' if es else 'WARN'}:</strong> {html.escape(w)}</li>"
                for w in self.warnings
            )
            notice_block = (
                f"<section class='card'><div class='{notice_cls}'>"
                f"<h2>{html.escape(notice_title)}</h2>"
                f"<ul class='notices'>{items}</ul></div></section>"
            )

        freshness_cols = [
            c
            for c in (
                "source_id",
                "provider",
                "status",
                "mode",
                "n_obs",
                "obs_end",
                "staleness_days",
                "fallback_used",
                "error",
            )
            if c in self._freshness_df.columns
        ]
        freshness_view = self._freshness_df[freshness_cols] if freshness_cols else self._freshness_df

        kpis = [
            ("Modo" if es else "Mode", self.mode),
            ("Fecha de corte" if es else "As-of", self.as_of),
            ("Conectores" if es else "Connectors", f"{self.connectors_ok}/{self.connectors_total}"),
            ("Preflight", self.preflight_status),
            ("Países / Filas" if es else "Countries / Rows", f"{self.n_countries} / {self.n_rows}"),
            ("Especificaciones LP" if es else "LP Specs", f"{self.n_lp_specs_ok} ok / {self.n_lp_specs_failed} err"),
        ]
        kpi_html = "".join(
            f"<div class='kpi'><div class='kpi-label'>{html.escape(lbl)}</div>"
            f"<div class='kpi-value'>{html.escape(val)}</div></div>"
            for lbl, val in kpis
        )

        return (
            f"<!DOCTYPE html>\n"
            f"<html lang=\"{use_lang}\">\n"
            f"<head>\n"
            f"  <meta charset=\"utf-8\">\n"
            f"  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"  <title>{html.escape(title)}</title>\n"
            f"  <style>{_HTML_STYLE}</style>\n"
            f"</head>\n"
            f"<body>\n"
            f"<div class=\"container\">\n"
            f"  <section class=\"header\">\n"
            f"    <span class=\"badge {badge_cls}\">{html.escape(self.overall_status)}</span>\n"
            f"    <h1>{html.escape(title)}</h1>\n"
            f"    <p class=\"subtitle\">{html.escape(self.headline(lang=use_lang))}</p>\n"
            f"    <p class=\"subtitle\"><code>{html.escape(self.input_source)}</code> · "
            f"SHA-256 <code>{html.escape(self.input_sha256[:16])}</code> · "
            f"{html.escape(self.sample_start)} .. {html.escape(self.sample_end)}</p>\n"
            f"    <div class=\"kpi-grid\">{kpi_html}</div>\n"
            f"  </section>\n"
            f"  {notice_block}\n"
            f"  <section class=\"card\">\n"
            f"    <h2>{html.escape('1. Actualidad y Procedencia de Conectores' if es else '1. Connector Freshness & Provenance')}</h2>\n"
            f"    {_render_table(freshness_view, 'Sin registros de conectores.' if es else 'No connector records.')}\n"
            f"  </section>\n"
            f"  <section class=\"card\">\n"
            f"    <h2>{html.escape('2. Diagnóstico Estructural del Panel' if es else '2. Panel Structural Preflight')}</h2>\n"
            f"    {_render_table(self._preflight_variables_df, 'Sin métricas de variables.' if es else 'No variable diagnostics.')}\n"
            f"  </section>\n"
            f"  <section class=\"card\">\n"
            f"    <h2>{html.escape('3. Indicadores Recientes por País' if es else '3. Latest Country Indicators')}</h2>\n"
            f"    {_render_table(self._latest_indicators_df, 'Sin indicadores calculados.' if es else 'No indicators computed.')}\n"
            f"  </section>\n"
            f"  <section class=\"card\">\n"
            f"    <h2>{html.escape('4. Diagnóstico de Nowcast AR(1)' if es else '4. Rolling AR(1) Nowcast Diagnostic')}</h2>\n"
            f"    {_render_table(self._nowcast_df, 'Sin estimaciones AR(1).' if es else 'No AR(1) nowcasts.')}\n"
            f"  </section>\n"
            f"  <section class=\"card\">\n"
            f"    <h2>{html.escape('5. Sensibilidad de Proyecciones Locales (LP)' if es else '5. Local Projection Sensitivity Grid')}</h2>\n"
            f"    {_render_table(self._lp_metrics_df, 'Malla LP omitida o inestable.' if es else 'LP sensitivity skipped or unstable.')}\n"
            f"  </section>\n"
            f"  <p class=\"footer\">puremacro.macro_monitor · schema v1 · {html.escape(self.as_of)}</p>\n"
            f"</div>\n"
            f"</body>\n"
            f"</html>\n"
        )

    def write_bundle(
        self,
        output_dir: str | Path,
        *,
        lang: str | None = None,
    ) -> dict[str, Path]:
        """Write bilingual dossier files, CSVs, and SHA-256 ``manifest.json`` (UTF-8)."""
        use_lang = _normalize_lang(self.lang if lang is None else lang)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        targets: dict[str, Path] = {
            "monitor.md": out_dir / "monitor.md",
            "monitor.html": out_dir / "monitor.html",
            "monitor.json": out_dir / "monitor.json",
            "freshness.csv": out_dir / "freshness.csv",
            "latest_indicators.csv": out_dir / "latest_indicators.csv",
        }

        targets["monitor.md"].write_text(self.to_markdown(lang=use_lang), encoding="utf-8", newline="\n")
        targets["monitor.html"].write_text(self.to_html(lang=use_lang), encoding="utf-8", newline="\n")
        targets["monitor.json"].write_text(self.to_json(), encoding="utf-8", newline="\n")

        freshness_csv = self._freshness_df.to_csv(index=False, lineterminator="\n")
        indicators_csv = self._latest_indicators_df.to_csv(index=False, lineterminator="\n")
        targets["freshness.csv"].write_text(freshness_csv, encoding="utf-8", newline="")
        targets["latest_indicators.csv"].write_text(indicators_csv, encoding="utf-8", newline="")

        manifest_files: dict[str, dict[str, Any]] = {}
        for rel_name in sorted(targets):
            raw = targets[rel_name].read_bytes()
            manifest_files[rel_name] = {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": int(len(raw)),
            }

        manifest_payload = {
            "schema": "puremacro.macro_monitor.manifest.v1",
            "lang": use_lang,
            "mode": self.mode,
            "as_of": self.as_of,
            "overall_status": self.overall_status,
            "passed": bool(self.passed),
            "has_outages": bool(self.has_outages),
            "input_source": self.input_source,
            "input_sha256": self.input_sha256,
            "files": manifest_files,
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(_json_safe(manifest_payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        targets["manifest.json"] = manifest_path
        return targets


def _audit_connectors(
    cfg: MacroMonitorConfig,
    *,
    as_of: str | pd.Timestamp | None,
    rate_budget: Any | None,
    probe_overrides: Mapping[str, Callable[[], pd.DataFrame | pd.Series]] | None,
) -> tuple[
    str,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, int],
    dict[str, Any],
    bool,
    int,
    int,
    tuple[str, ...],
    tuple[str, ...],
    list[str],
    list[str],
]:
    from puremacro.fetch import registry
    from puremacro.fetch.freshness import (
        OUTAGE_STATUSES,
        RateBudget,
        connector_freshness_report,
    )

    if cfg.panel_builder is not None:
        # Validate requested panel_builder against fetch.registry eagerly.
        registry.info(cfg.panel_builder)

    budget = (
        rate_budget
        if rate_budget is not None
        else RateBudget(
            max_requests=cfg.max_requests,
            min_interval_seconds=cfg.min_interval_seconds,
            timeout_seconds=cfg.timeout_seconds,
        )
    )
    freshness = connector_freshness_report(
        mode=cfg.mode,
        sources=cfg.sources,
        as_of=as_of if as_of is not None else cfg.as_of,
        db_path=cfg.db_path,
        rate_budget=budget,
        probe_overrides=probe_overrides,
    )

    registry_raw = registry.sources()
    reg_cols = [c for c in ("id", "kind", "provider", "freq", "auth", "redistributable") if c in registry_raw.columns]
    registry_df = registry_raw[reg_cols].copy() if reg_cols else pd.DataFrame()

    fresh_df = freshness.to_frame().copy()
    if not fresh_df.empty:
        for text_col in ("error", "note", "identity_detail"):
            if text_col in fresh_df.columns:
                fresh_df[text_col] = fresh_df[text_col].map(
                    lambda x: _sanitize_text(x, extra_paths=(cfg.db_path,)) if x is not None and str(x) != "nan" else x
                )
        if not registry_df.empty and "id" in registry_df.columns and "source_id" in fresh_df.columns:
            kind_map = dict(zip(registry_df["id"].astype(str), registry_df["kind"].astype(str)))
            fresh_df["registry_kind"] = fresh_df["source_id"].astype(str).map(kind_map).fillna("cohort")

    outages: list[str] = []
    stales: list[str] = []
    warnings_list: list[str] = []
    errors_list: list[str] = []
    n_ok = 0

    for rec in freshness.records:
        detail = _sanitize_text(rec.error or rec.note or rec.identity_detail or "", extra_paths=(cfg.db_path,))
        if rec.status in ("fresh", "revised", "recorded", "cached"):
            n_ok += 1
        elif rec.status in OUTAGE_STATUSES:
            outages.append(rec.source_id)
            msg = f"Connector {rec.source_id} status={rec.status}" + (f": {detail}" if detail else "")
            warnings_list.append(msg)
            if rec.status in ("failed", "schema_drift", "empty"):
                errors_list.append(msg)
        elif rec.status == "stale":
            stales.append(rec.source_id)
            warnings_list.append(
                f"Connector {rec.source_id} status=stale (staleness_days={rec.staleness_days}, max={rec.max_staleness_days})"
            )
        else:
            # credential_missing / not_cached
            warnings_list.append(
                f"Connector {rec.source_id} status={rec.status}" + (f": {detail}" if detail else "")
            )

    rate_info = {
        "max_requests": int(budget.max_requests),
        "min_interval_seconds": float(budget.min_interval_seconds),
        "timeout_seconds": float(budget.timeout_seconds),
        "requests_used": int(freshness.requests_used),
    }

    return (
        str(freshness.as_of),
        fresh_df,
        registry_df,
        dict(freshness.status_counts),
        rate_info,
        bool(freshness.has_outages),
        n_ok,
        len(freshness.records),
        tuple(outages),
        tuple(stales),
        warnings_list,
        errors_list,
    )


def _ingest_panel(
    data: pd.DataFrame | str | Path | None,
    cfg: MacroMonitorConfig,
) -> tuple[pd.DataFrame, str, str, list[str], list[str]]:
    """Ingest user DataFrame, CSV/JSON file, registry builder, or recorded QNA panel."""
    ingest_errors: list[str] = []
    ingest_warnings: list[str] = []

    if data is not None:
        if isinstance(data, pd.DataFrame):
            df = data.copy(deep=True)
            canonical_bytes = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
            sha256 = hashlib.sha256(canonical_bytes).hexdigest()
            return df, "dataframe", sha256, ingest_warnings, ingest_errors

        path = Path(data)
        if not path.is_file():
            raise FileNotFoundError(f"Panel input file not found: {path.name}")
        raw_bytes = path.read_bytes()
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        suffix = path.suffix.lower()
        if suffix == ".json":
            df = pd.read_json(io.BytesIO(raw_bytes))
        else:
            df = pd.read_csv(io.BytesIO(raw_bytes))
        return df, f"file:{path.name}", sha256, ingest_warnings, ingest_errors

    if cfg.panel_builder is not None:
        from puremacro.fetch import registry

        info = registry.info(cfg.panel_builder)
        if cfg.mode == "live":
            try:
                loader = registry.load(cfg.panel_builder)
                built = loader()
                if not isinstance(built, pd.DataFrame) or built.empty:
                    ingest_errors.append(
                        f"Panel builder {info.id!r} returned an empty or non-DataFrame payload in live mode."
                    )
                    return pd.DataFrame(), f"registry:{info.id}", hashlib.sha256(b"").hexdigest(), ingest_warnings, ingest_errors
                canonical_bytes = built.to_csv(index=False, lineterminator="\n").encode("utf-8")
                return (
                    built.copy(deep=True),
                    f"registry:{info.id}",
                    hashlib.sha256(canonical_bytes).hexdigest(),
                    ingest_warnings,
                    ingest_errors,
                )
            except Exception as exc:  # fail-closed: zero silent synthetic fallback
                ingest_errors.append(
                    f"Live panel builder {info.id!r} failed: {_sanitize_text(exc)}"
                )
                return pd.DataFrame(), f"registry:{info.id}", hashlib.sha256(b"").hexdigest(), ingest_warnings, ingest_errors

        if cfg.panel_builder == "oecd.qna_panel":
            df, src_label, sha256 = _build_default_recorded_panel()
            return df, f"recorded:{info.id}", sha256, ingest_warnings, ingest_errors

        ingest_errors.append(
            f"Panel builder {info.id!r} has no offline recorded fixture; supply local data or run mode='live'."
        )
        return pd.DataFrame(), f"registry:{info.id}", hashlib.sha256(b"").hexdigest(), ingest_warnings, ingest_errors

    df, src_label, sha256 = _build_default_recorded_panel()
    return df, src_label, sha256, ingest_warnings, ingest_errors


def _run_structural_preflight(
    df: pd.DataFrame,
    cfg: MacroMonitorConfig,
    *,
    input_source: str,
) -> tuple[
    pd.DataFrame | None,
    str | None,
    str | None,
    str | None,
    str | None,
    tuple[str, ...],
    str,
    int,
    int,
    dict[str, Any],
    pd.DataFrame,
    str,
    str,
    list[str],
    list[str],
]:
    """Fail-closed schema, date integrity, chronological order, and preflight_panel checks."""
    from puremacro.data import preflight_panel

    pf_warnings: list[str] = []
    pf_errors: list[str] = []

    if df.empty:
        pf_errors.append(f"Panel input {input_source!r} is empty (0 rows).")
        return (
            None, None, None, None, None, (), "fail", len(pf_errors), 0,
            {"status": "fail", "passed": False, "n_errors": len(pf_errors), "n_warnings": 0},
            pd.DataFrame(), "", "", pf_warnings, pf_errors,
        )

    u_col, t_col, col_errors = _resolve_columns(df, unit_col=cfg.unit_col, time_col=cfg.time_col)
    if cfg.y is not None and cfg.y not in df.columns:
        col_errors.append(f"missing_required_column: target y={cfg.y!r} not in panel columns.")
    if cfg.x is not None and cfg.x not in df.columns:
        col_errors.append(f"missing_required_column: impulse x={cfg.x!r} not in panel columns.")
    for ctrl in cfg.controls:
        if ctrl not in df.columns:
            col_errors.append(f"missing_required_column: control {ctrl!r} not in panel columns.")
    if cfg.variables is not None:
        for var in cfg.variables:
            if var not in df.columns:
                col_errors.append(f"missing_required_column: variable {var!r} not in panel columns.")

    if col_errors or u_col is None or t_col is None:
        pf_errors.extend(col_errors)
        return (
            None, u_col, t_col, cfg.y, cfg.x, tuple(cfg.controls), "fail",
            len(pf_errors), 0,
            {"status": "fail", "passed": False, "n_errors": len(pf_errors), "n_warnings": 0},
            pd.DataFrame(), "", "", pf_warnings, pf_errors,
        )

    work = df.copy(deep=True)
    work[u_col] = work[u_col].astype(str)

    parsed_dates, n_invalid_dates = _parse_time_series(work[t_col])
    if n_invalid_dates > 0:
        pf_errors.append(
            f"corrupted_dates: {n_invalid_dates} row(s) in time column {t_col!r} could not be parsed or are missing."
        )

    # Check chronological order explicitly before preflight_panel sorts internally.
    unsorted_units: list[str] = []
    duplicate_key_count = 0
    work_dates = pd.DataFrame({"_unit": work[u_col], "_dt": parsed_dates})
    for unit_name, grp in work_dates.groupby("_unit", sort=False):
        valid = grp["_dt"].dropna()
        if not valid.empty:
            if bool((valid.diff().dropna() < pd.Timedelta(0)).any()):
                unsorted_units.append(str(unit_name))
            duplicate_key_count += int(valid.duplicated().sum())

    if unsorted_units:
        sample_units = ",".join(unsorted_units[:5])
        pf_errors.append(
            f"unsorted_dates: non-chronological date order detected in {len(unsorted_units)} unit(s) ({sample_units})."
        )
    if duplicate_key_count > 0:
        pf_errors.append(
            f"duplicate_panel_keys: {duplicate_key_count} duplicate ({u_col}, {t_col}) row(s) detected."
        )

    # Determine candidate numeric columns.
    excluded = {u_col, t_col}
    numeric_cols = [
        str(col)
        for col in work.columns
        if str(col) not in excluded and pd.api.types.is_numeric_dtype(work[col])
    ]
    if not numeric_cols:
        pf_errors.append("missing_numeric_variables: panel contains no numeric indicator columns.")
        return (
            None, u_col, t_col, cfg.y, cfg.x, tuple(cfg.controls), "fail",
            len(pf_errors), 0,
            {"status": "fail", "passed": False, "n_errors": len(pf_errors), "n_warnings": 0},
            pd.DataFrame(), "", "", pf_warnings, pf_errors,
        )

    # Resolve target y, impulse x, and controls when omitted.
    if cfg.y is not None:
        target_y: str | None = cfg.y
    elif "gdp_growth" in numeric_cols:
        target_y = "gdp_growth"
    else:
        target_y = numeric_cols[0]

    if cfg.x is not None:
        impulse_x: str | None = cfg.x
    elif "inflation_qoq" in numeric_cols and target_y != "inflation_qoq":
        impulse_x = "inflation_qoq"
    elif "policy_rate" in numeric_cols and target_y != "policy_rate":
        impulse_x = "policy_rate"
    else:
        others = [c for c in numeric_cols if c != target_y]
        impulse_x = others[0] if others else None

    if cfg.controls:
        controls = tuple(cfg.controls)
    elif "cons_growth" in numeric_cols and "cons_growth" not in {target_y, impulse_x}:
        controls = ("cons_growth",)
    else:
        controls = ()

    # Ensure focal y/x/controls are actually numeric.
    for role, col_name in (("y", target_y), ("x", impulse_x), *((f"control", c) for c in controls)):
        if col_name is not None and not pd.api.types.is_numeric_dtype(work[col_name]):
            pf_errors.append(f"non_numeric_column: {role} column {col_name!r} is not numeric.")

    valid_dt = parsed_dates.dropna()
    sample_start = valid_dt.min().strftime("%Y-%m-%d") if not valid_dt.empty else ""
    sample_end = valid_dt.max().strftime("%Y-%m-%d") if not valid_dt.empty else ""

    if n_invalid_dates == len(work) or valid_dt.empty:
        return (
            None, u_col, t_col, target_y, impulse_x, controls, "fail",
            len(pf_errors), 0,
            {"status": "fail", "passed": False, "n_errors": len(pf_errors), "n_warnings": 0},
            pd.DataFrame(), sample_start, sample_end, pf_warnings, pf_errors,
        )

    # Prepare normalized copy (preserving valid rows) for preflight_panel diagnostics.
    pf_df = work.copy(deep=True)
    pf_df[t_col] = parsed_dates
    if n_invalid_dates > 0:
        pf_df = pf_df.loc[parsed_dates.notna()].copy()

    vars_to_check = list(cfg.variables) if cfg.variables is not None else numeric_cols
    try:
        pf = preflight_panel(
            pf_df,
            unit_col=u_col,
            time_col=t_col,
            variables=vars_to_check,
            freq=cfg.freq,
            digits=cfg.digits,
        )
        pf_var_df = pf.to_frame("variables").copy()
        for finding in pf.findings:
            msg = f"preflight[{finding.check}]({finding.variable or 'panel'}): {finding.message}"
            if finding.severity == "error":
                if finding.check not in {"missing_time", "duplicates"} or not pf_errors:
                    pf_errors.append(msg)
            else:
                pf_warnings.append(msg)
        pf_status = "fail" if pf_errors or pf.status == "fail" else pf.status
        n_pf_err = len(pf_errors) if pf_errors else int(pf.n_errors)
        n_pf_warn = len(pf_warnings)
        pf_summary = {
            "status": pf_status,
            "passed": bool(pf_status != "fail"),
            "panel_shape": list(pf.panel_shape),
            "n_units": int(pf.n_units),
            "n_periods": int(pf.n_periods),
            "inferred_freq": pf.inferred_freq,
            "date_range": list(pf.date_range) if pf.date_range is not None else [],
            "balanced": bool(pf.balanced),
            "duplicate_count": int(max(pf.duplicate_count, duplicate_key_count)),
            "n_errors": int(n_pf_err),
            "n_warnings": int(n_pf_warn),
            "recommendations": list(pf.recommendations),
        }
        clean_df = pf_df.sort_values([u_col, t_col], kind="mergesort").reset_index(drop=True)
        if pf_errors:
            # Fail-closed: surface diagnostics while blocking estimation on invalid/corrupted dates.
            clean_panel = None if (n_invalid_dates > 0 or bool(unsorted_units)) else clean_df
        else:
            clean_panel = clean_df
        return (
            clean_panel, u_col, t_col, target_y, impulse_x, controls,
            pf_status, n_pf_err, n_pf_warn, pf_summary, pf_var_df,
            sample_start, sample_end, pf_warnings, pf_errors,
        )
    except Exception as exc:
        pf_errors.append(f"preflight_exception: {_sanitize_text(exc)}")
        return (
            None, u_col, t_col, target_y, impulse_x, controls, "fail",
            len(pf_errors), len(pf_warnings),
            {"status": "fail", "passed": False, "n_errors": len(pf_errors), "n_warnings": len(pf_warnings)},
            pd.DataFrame(), sample_start, sample_end, pf_warnings, pf_errors,
        )


def _infer_periods_per_year(dates: pd.Series, inferred_freq: str | None) -> int:
    freq_txt = (inferred_freq or "").upper()
    if freq_txt.startswith("M"):
        return 12
    if freq_txt.startswith("A") or freq_txt.startswith("Y"):
        return 1
    if freq_txt.startswith("Q"):
        return 4
    diffs = dates.sort_values().drop_duplicates().diff().dropna().dt.days
    if diffs.empty:
        return 4
    med = float(diffs.median())
    if med >= 300:
        return 1
    if 20 <= med <= 45:
        return 12
    return 4


def _is_growth_or_rate_name(col_name: str, values: np.ndarray) -> bool:
    low = col_name.lower()
    if any(tok in low for tok in ("growth", "infl", "rate", "spread", "yoy", "qoq", "mom", "pct", "yield")):
        return True
    if np.any(values <= 0.0):
        return True
    return False


def _compute_country_indicators_and_nowcast(
    clean_df: pd.DataFrame,
    *,
    u_col: str,
    t_col: str,
    variables: Sequence[str],
    target_y: str | None,
    inferred_freq: str | None,
    cfg: MacroMonitorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Compute per-country latest indicator metrics, z-score alerts, and AR(1) nowcasts."""
    ind_rows: list[dict[str, Any]] = []
    nc_rows: list[dict[str, Any]] = []
    diag_warnings: list[str] = []

    periods_per_year = _infer_periods_per_year(clean_df[t_col], inferred_freq)

    for country, grp in clean_df.groupby(u_col, sort=True):
        grp_sorted = grp.sort_values(t_col, kind="mergesort")
        for var in variables:
            sub = grp_sorted[[t_col, var]].dropna()
            if sub.empty:
                ind_rows.append(
                    {
                        "country": str(country),
                        "variable": var,
                        "latest_date": "",
                        "n_obs": 0,
                        "latest_value": np.nan,
                        "qoq_growth": np.nan,
                        "yoy_growth": np.nan,
                        "rolling_mean": np.nan,
                        "rolling_std": np.nan,
                        "z_score": np.nan,
                        "flag": "MISSING",
                    }
                )
                continue

            vals = sub[var].to_numpy(dtype=float)
            dates = sub[t_col]
            n_obs = int(len(vals))
            latest_val = float(vals[-1])
            latest_dt = pd.Timestamp(dates.iloc[-1]).strftime("%Y-%m-%d")

            is_rate = _is_growth_or_rate_name(var, vals)
            if is_rate:
                qoq_val = latest_val
                if n_obs >= periods_per_year:
                    tail_yoy = vals[-periods_per_year:]
                    # Compound periodic percentage changes when plausible (% units); else sum.
                    if np.all(tail_yoy > -90.0) and np.all(tail_yoy < 200.0):
                        yoy_val = float((np.prod(1.0 + tail_yoy / 100.0) - 1.0) * 100.0)
                    else:
                        yoy_val = float(np.sum(tail_yoy))
                else:
                    yoy_val = float("nan")
            else:
                qoq_val = (
                    float(100.0 * (vals[-1] / vals[-2] - 1.0))
                    if n_obs >= 2 and vals[-2] > 0.0
                    else float("nan")
                )
                yoy_val = (
                    float(100.0 * (vals[-1] / vals[-1 - periods_per_year] - 1.0))
                    if n_obs > periods_per_year and vals[-1 - periods_per_year] > 0.0
                    else float("nan")
                )

            tail = vals[-cfg.vol_window :]
            roll_mean = float(np.mean(tail))
            roll_std = float(np.std(tail, ddof=1)) if len(tail) >= 2 else 0.0
            z_score = float((latest_val - roll_mean) / roll_std) if roll_std > 1e-12 else 0.0

            if n_obs < cfg.short_sample_min_obs:
                flag = "SHORT_SAMPLE"
            elif z_score >= cfg.zscore_threshold:
                flag = "Z_ALERT_HIGH"
            elif z_score <= -cfg.zscore_threshold:
                flag = "Z_ALERT_LOW"
            else:
                flag = "OK"

            ind_rows.append(
                {
                    "country": str(country),
                    "variable": var,
                    "latest_date": latest_dt,
                    "n_obs": n_obs,
                    "latest_value": latest_val,
                    "qoq_growth": qoq_val,
                    "yoy_growth": yoy_val,
                    "rolling_mean": roll_mean,
                    "rolling_std": roll_std,
                    "z_score": z_score,
                    "flag": flag,
                }
            )

        if target_y is not None and target_y in grp_sorted.columns:
            sub_y = grp_sorted[[t_col, target_y]].dropna()
            y_vals = sub_y[target_y].to_numpy(dtype=float)
            n_y = int(len(y_vals))
            if n_y >= 6:
                y_lag = y_vals[:-1]
                y_cur = y_vals[1:]
                X = np.column_stack([np.ones_like(y_lag), y_lag])
                coef, _, _, _ = np.linalg.lstsq(X, y_cur, rcond=None)
                intercept = float(coef[0])
                phi = float(coef[1])
                fitted = X @ coef
                resid = y_cur - fitted
                rmse = float(np.sqrt(np.mean(resid ** 2)))
                next_fc = float(intercept + phi * y_vals[-1])
                last_err = float(resid[-1])
                if abs(phi) >= 1.0:
                    nc_flag = "EXPLOSIVE_AR"
                    diag_warnings.append(
                        f"Country {country} AR(1) persistence |phi|={abs(phi):.3f} >= 1.0 on {target_y!r}."
                    )
                elif n_y < cfg.short_sample_min_obs:
                    nc_flag = "SHORT_SAMPLE"
                else:
                    nc_flag = "OK"
                nc_rows.append(
                    {
                        "country": str(country),
                        "variable": target_y,
                        "latest_date": pd.Timestamp(sub_y[t_col].iloc[-1]).strftime("%Y-%m-%d"),
                        "n_obs": n_y,
                        "latest_actual": float(y_vals[-1]),
                        "ar1_intercept": intercept,
                        "ar1_phi": phi,
                        "nowcast_next": next_fc,
                        "last_residual": last_err,
                        "in_sample_rmse": rmse,
                        "flag": nc_flag,
                    }
                )
            else:
                diag_warnings.append(
                    f"Country {country} has insufficient observations (n={n_y}) for AR(1) nowcast on {target_y!r}."
                )
                nc_rows.append(
                    {
                        "country": str(country),
                        "variable": target_y,
                        "latest_date": pd.Timestamp(sub_y[t_col].iloc[-1]).strftime("%Y-%m-%d") if n_y > 0 else "",
                        "n_obs": n_y,
                        "latest_actual": float(y_vals[-1]) if n_y > 0 else np.nan,
                        "ar1_intercept": np.nan,
                        "ar1_phi": np.nan,
                        "nowcast_next": np.nan,
                        "last_residual": np.nan,
                        "in_sample_rmse": np.nan,
                        "flag": "INSUFFICIENT_OBS",
                    }
                )

    ind_df = pd.DataFrame(ind_rows)
    nc_df = pd.DataFrame(nc_rows)
    return ind_df, nc_df, diag_warnings


def _run_lp_sensitivity(
    clean_df: pd.DataFrame,
    *,
    u_col: str,
    t_col: str,
    target_y: str | None,
    impulse_x: str | None,
    controls: tuple[str, ...],
    cfg: MacroMonitorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], int, int, list[str]]:
    """Run bounded LP sensitivity check and surface instability/short-sample warnings."""
    lp_warnings: list[str] = []
    if not cfg.run_lp_sensitivity:
        return pd.DataFrame(), pd.DataFrame(), {"enabled": False}, 0, 0, lp_warnings
    if target_y is None or impulse_x is None or target_y == impulse_x:
        lp_warnings.append("LP sensitivity skipped: distinct target y and impulse x columns are required.")
        return pd.DataFrame(), pd.DataFrame(), {"enabled": True, "skipped": True}, 0, 0, lp_warnings

    req_cols = [target_y, impulse_x, *controls]
    est_df = clean_df[[u_col, t_col, *req_cols]].dropna(subset=[target_y, impulse_x]).copy()
    if est_df.empty:
        lp_warnings.append("LP sensitivity skipped: no complete (y, x) observations after dropping missing values.")
        return pd.DataFrame(), pd.DataFrame(), {"enabled": True, "skipped": True}, 0, 0, lp_warnings

    counts_per_unit = est_df.groupby(u_col, sort=False).size()
    min_unit_obs = int(counts_per_unit.min()) if not counts_per_unit.empty else 0
    if min_unit_obs < cfg.short_sample_min_obs:
        lp_warnings.append(
            f"short_sample: minimum country series length ({min_unit_obs}) is below short_sample_min_obs={cfg.short_sample_min_obs}; LP estimates may be noisy."
        )

    from puremacro.lp.jorda import lp_hac
    from puremacro.lp.la_lp import la_lp
    from puremacro.lp.panel import panel_lp
    from puremacro.lp.panel_dk import panel_lp_dk
    from puremacro.lp.sensitivity import lp_sensitivity_grid

    n_units = int(counts_per_unit.shape[0])
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if n_units >= 2:
                grid: dict[str, Sequence[Any]] = {
                    "lags": list(cfg.lp_lags),
                    "estimator": [panel_lp, panel_lp_dk],
                }
                base_kwargs: dict[str, Any] = {
                    "horizons": list(cfg.horizons),
                    "controls": list(controls),
                    "ci": cfg.ci,
                    "unit_col": u_col,
                    "time_col": t_col,
                }
                cmp = lp_sensitivity_grid(
                    est_df,
                    target_y,
                    impulse_x,
                    grid=grid,
                    estimator=panel_lp,
                    base_kwargs=base_kwargs,
                    on_error="skip",
                    horizon_align="intersection",
                    band_compat="warn",
                )
            else:
                ts_df = est_df.sort_values(t_col, kind="mergesort").set_index(t_col)
                ts_grid: dict[str, Sequence[Any]] = {
                    "lags": list(cfg.lp_lags),
                    "estimator": [lp_hac, la_lp],
                }
                base_kwargs = {
                    "horizons": list(cfg.horizons),
                    "controls": list(controls),
                    "ci": cfg.ci,
                    "allow_index_gaps": True,
                }
                cmp = lp_sensitivity_grid(
                    ts_df,
                    target_y,
                    impulse_x,
                    grid=ts_grid,
                    estimator=lp_hac,
                    base_kwargs=base_kwargs,
                    on_error="skip",
                    horizon_align="intersection",
                    band_compat="warn",
                )

        for cw in caught:
            lp_warnings.append(f"lp_warning: {_sanitize_text(cw.message)}")
        for msg in getattr(cmp, "warnings", ()):
            lp_warnings.append(f"lp_grid: {_sanitize_text(msg)}")

        metrics_df = cmp.metrics.copy() if isinstance(cmp.metrics, pd.DataFrame) else pd.DataFrame()
        table_df = cmp.table.copy() if isinstance(cmp.table, pd.DataFrame) else pd.DataFrame()
        err_dict = getattr(cmp, "errors", {}) or {}
        for spec_name, err_msg in err_dict.items():
            lp_warnings.append(f"lp_spec_failed[{spec_name}]: {_sanitize_text(err_msg)}")

        n_ok = int(len(metrics_df))
        n_fail = int(len(err_dict))
        lp_summary = {
            "enabled": True,
            "n_units": n_units,
            "target_y": target_y,
            "impulse_x": impulse_x,
            "controls": list(controls),
            "horizons": [int(h) for h in getattr(cmp, "horizons", cfg.horizons)],
            "n_specs_ok": n_ok,
            "n_specs_failed": n_fail,
            "baseline": str(getattr(cmp, "baseline", "")),
        }
        return metrics_df, table_df, lp_summary, n_ok, n_fail, lp_warnings
    except Exception as exc:
        lp_warnings.append(f"lp_sensitivity_unstable: {_sanitize_text(exc)}")
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {"enabled": True, "target_y": target_y, "impulse_x": impulse_x, "unstable": True},
            0,
            len(cfg.lp_lags) * 2,
            lp_warnings,
        )


def _derive_overall_status(
    *,
    errors_list: Sequence[str],
    warnings_list: Sequence[str],
    outage_connectors: Sequence[str],
    stale_connectors: Sequence[str],
    preflight_status: str,
    connectors_ok: int,
    connectors_total: int,
) -> tuple[str, bool]:
    if errors_list or preflight_status == "fail":
        return "FAILED", False
    if outage_connectors or stale_connectors or (connectors_total > 0 and connectors_ok == 0):
        return "DEGRADED", False
    if warnings_list or preflight_status == "warn" or connectors_ok < connectors_total:
        return "WARN", True
    return "OK", True


def run_macro_monitor(
    data: pd.DataFrame | str | Path | None = None,
    *,
    config: MacroMonitorConfig | None = None,
    mode: str | None = None,
    sources: Sequence[str] | None = None,
    as_of: str | pd.Timestamp | None = None,
    db_path: str | Path | None = None,
    rate_budget: Any | None = None,
    probe_overrides: Mapping[str, Callable[[], pd.DataFrame | pd.Series]] | None = None,
    panel_builder: str | None = None,
    unit_col: str | None = None,
    time_col: str | None = None,
    variables: Sequence[str] | None = None,
    y: str | None = None,
    x: str | None = None,
    controls: Sequence[str] | None = None,
    horizons: Sequence[int] | None = None,
    lags: int | None = None,
    lp_lags: Sequence[int] | None = None,
    ci: float | None = None,
    vol_window: int | None = None,
    zscore_threshold: float | None = None,
    short_sample_min_obs: int | None = None,
    run_lp_sensitivity: bool | None = None,
    freq: str | None = None,
    lang: str | None = None,
    digits: int | None = None,
    fail_on_outage: bool | None = None,
) -> MacroMonitorReport:
    """Execute the 4-stage macro freshness and LP monitoring workflow."""
    base = config if config is not None else MacroMonitorConfig()
    cfg = MacroMonitorConfig(
        mode=base.mode if mode is None else mode,
        sources=base.sources if sources is None else _normalize_str_tuple(sources, name="sources"),
        as_of=base.as_of if as_of is None else str(as_of),
        db_path=base.db_path if db_path is None else str(db_path),
        max_requests=base.max_requests if rate_budget is None else int(rate_budget.max_requests),
        min_interval_seconds=(
            base.min_interval_seconds
            if rate_budget is None
            else float(rate_budget.min_interval_seconds)
        ),
        timeout_seconds=(
            base.timeout_seconds if rate_budget is None else float(rate_budget.timeout_seconds)
        ),
        panel_builder=base.panel_builder if panel_builder is None else panel_builder,
        unit_col=base.unit_col if unit_col is None else unit_col,
        time_col=base.time_col if time_col is None else time_col,
        variables=(
            base.variables
            if variables is None
            else _normalize_str_tuple(variables, name="variables")
        ),
        y=base.y if y is None else y,
        x=base.x if x is None else x,
        controls=(
            base.controls
            if controls is None
            else (_normalize_str_tuple(controls, name="controls") or ())
        ),
        horizons=base.horizons if horizons is None else tuple(int(h) for h in horizons),
        lags=base.lags if lags is None else int(lags),
        lp_lags=base.lp_lags if lp_lags is None else tuple(int(l) for l in lp_lags),
        ci=base.ci if ci is None else float(ci),
        vol_window=base.vol_window if vol_window is None else int(vol_window),
        zscore_threshold=base.zscore_threshold if zscore_threshold is None else float(zscore_threshold),
        short_sample_min_obs=(
            base.short_sample_min_obs if short_sample_min_obs is None else int(short_sample_min_obs)
        ),
        run_lp_sensitivity=(
            base.run_lp_sensitivity if run_lp_sensitivity is None else bool(run_lp_sensitivity)
        ),
        freq=base.freq if freq is None else freq,
        lang=base.lang if lang is None else lang,
        digits=base.digits if digits is None else int(digits),
        fail_on_outage=base.fail_on_outage if fail_on_outage is None else bool(fail_on_outage),
    )

    all_warnings: list[str] = []
    all_errors: list[str] = []

    # Stage 1: Connector Freshness & Provenance Audit
    (
        resolved_as_of,
        fresh_df,
        registry_df,
        status_counts,
        rate_info,
        has_outages,
        n_ok,
        n_total,
        outage_connectors,
        stale_connectors,
        conn_warnings,
        conn_errors,
    ) = _audit_connectors(
        cfg,
        as_of=as_of,
        rate_budget=rate_budget,
        probe_overrides=probe_overrides,
    )
    all_warnings.extend(conn_warnings)
    all_errors.extend(conn_errors)

    # Stage 2: Panel Ingestion & Structural Preflight
    raw_df, input_source, input_sha256, ing_warnings, ing_errors = _ingest_panel(data, cfg)
    all_warnings.extend(ing_warnings)
    all_errors.extend(ing_errors)

    (
        clean_df,
        u_col,
        t_col,
        target_y,
        impulse_x,
        resolved_controls,
        pf_status,
        n_pf_err,
        n_pf_warn,
        pf_summary,
        pf_var_df,
        sample_start,
        sample_end,
        pf_warnings,
        pf_errors,
    ) = _run_structural_preflight(raw_df, cfg, input_source=input_source)
    all_warnings.extend(pf_warnings)
    all_errors.extend(pf_errors)

    # Stage 3: Country Latest Indicators, Rolling Nowcast & LP Sensitivity
    if clean_df is not None and u_col is not None and t_col is not None:
        excluded = {u_col, t_col}
        if cfg.variables is not None:
            monitored_vars = list(cfg.variables)
        else:
            monitored_vars = [
                str(c)
                for c in clean_df.columns
                if str(c) not in excluded and pd.api.types.is_numeric_dtype(clean_df[c])
            ]
        latest_df, nowcast_df, diag_warnings = _compute_country_indicators_and_nowcast(
            clean_df,
            u_col=u_col,
            t_col=t_col,
            variables=monitored_vars,
            target_y=target_y,
            inferred_freq=pf_summary.get("inferred_freq"),
            cfg=cfg,
        )
        all_warnings.extend(diag_warnings)

        lp_metrics_df, lp_table_df, lp_summary, n_lp_ok, n_lp_fail, lp_warnings = _run_lp_sensitivity(
            clean_df,
            u_col=u_col,
            t_col=t_col,
            target_y=target_y,
            impulse_x=impulse_x,
            controls=resolved_controls,
            cfg=cfg,
        )
        all_warnings.extend(lp_warnings)
        countries = tuple(sorted(clean_df[u_col].astype(str).unique().tolist()))
        n_rows = int(len(clean_df))
    else:
        latest_df = pd.DataFrame()
        nowcast_df = pd.DataFrame()
        lp_metrics_df = pd.DataFrame()
        lp_table_df = pd.DataFrame()
        lp_summary = {"enabled": bool(cfg.run_lp_sensitivity), "skipped": True}
        n_lp_ok = 0
        n_lp_fail = 0
        countries = (
            tuple(sorted(raw_df[u_col].astype(str).unique().tolist()))
            if (u_col is not None and u_col in raw_df.columns)
            else ()
        )
        n_rows = int(len(raw_df))

    # Deduplicate notices while preserving order.
    dedup_warnings = tuple(dict.fromkeys(w for w in all_warnings if w))
    dedup_errors = tuple(dict.fromkeys(e for e in all_errors if e))

    overall_status, passed = _derive_overall_status(
        errors_list=dedup_errors,
        warnings_list=dedup_warnings,
        outage_connectors=outage_connectors,
        stale_connectors=stale_connectors,
        preflight_status=pf_status,
        connectors_ok=n_ok,
        connectors_total=n_total,
    )

    if not resolved_as_of:
        resolved_as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return MacroMonitorReport(
        config=cfg,
        lang=cfg.lang,
        mode=cfg.mode,
        as_of=resolved_as_of,
        overall_status=overall_status,
        passed=passed,
        has_outages=has_outages,
        connectors_ok=n_ok,
        connectors_total=n_total,
        outage_connectors=outage_connectors,
        stale_connectors=stale_connectors,
        preflight_status=pf_status,
        preflight_errors=int(n_pf_err),
        preflight_warnings=int(n_pf_warn),
        input_source=input_source,
        input_sha256=input_sha256,
        sample_start=sample_start,
        sample_end=sample_end,
        n_countries=int(len(countries)),
        n_rows=n_rows,
        countries=countries,
        target_y=target_y,
        impulse_x=impulse_x,
        n_lp_specs_ok=n_lp_ok,
        n_lp_specs_failed=n_lp_fail,
        warnings=dedup_warnings,
        errors=dedup_errors,
        _freshness_df=fresh_df,
        _registry_df=registry_df,
        _preflight_variables_df=pf_var_df,
        _latest_indicators_df=latest_df,
        _nowcast_df=nowcast_df,
        _lp_metrics_df=lp_metrics_df,
        _lp_table_df=lp_table_df,
        _status_counts=status_counts,
        _preflight_summary=pf_summary,
        _lp_summary=lp_summary,
        _rate_budget=rate_info,
    )


def _parse_int_csv(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected non-empty comma-separated integer list.")
    return tuple(int(p) for p in parts)


def _parse_str_list(items: Sequence[str] | None) -> tuple[str, ...] | None:
    if not items:
        return None
    out: list[str] = []
    for item in items:
        for part in str(item).split(","):
            stripped = part.strip()
            if stripped:
                out.append(stripped)
    return tuple(out) if out else None


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint: ``python -m puremacro.macro_monitor``."""
    parser = argparse.ArgumentParser(
        prog="python -m puremacro.macro_monitor",
        description="Automated LatAm & macro freshness and LP monitor.",
    )
    parser.add_argument("--mode", choices=_VALID_MODES, default="recorded")
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--max-requests", type=int, default=5)
    parser.add_argument("--min-interval", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--input", "--data", dest="input_path", default=None)
    parser.add_argument("--panel-builder", default=None)
    parser.add_argument("--unit-col", default=None)
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--variables", nargs="*", default=None)
    parser.add_argument("--y", default=None)
    parser.add_argument("--x", default=None)
    parser.add_argument("--controls", nargs="*", default=None)
    parser.add_argument("--horizons", default=None)
    parser.add_argument("--lp-lags", default=None)
    parser.add_argument("--no-lp", action="store_true")
    parser.add_argument("--freq", default=None)
    parser.add_argument("--lang", choices=_VALID_LANGS, default="en")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--format", choices=_VALID_FORMATS, default="text")
    parser.add_argument("--fail-on-outage", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        cfg = MacroMonitorConfig(
            mode=args.mode,
            sources=_parse_str_list(args.sources),
            as_of=args.as_of,
            db_path=args.db_path,
            max_requests=args.max_requests,
            min_interval_seconds=args.min_interval,
            timeout_seconds=args.timeout,
            panel_builder=args.panel_builder,
            unit_col=args.unit_col,
            time_col=args.time_col,
            variables=_parse_str_list(args.variables),
            y=args.y,
            x=args.x,
            controls=_parse_str_list(args.controls) or (),
            horizons=_parse_int_csv(args.horizons) or (0, 1, 2, 3, 4),
            lp_lags=_parse_int_csv(args.lp_lags) or (1, 2),
            run_lp_sensitivity=not args.no_lp,
            freq=args.freq,
            lang=args.lang,
            fail_on_outage=bool(args.fail_on_outage),
        )
        report = run_macro_monitor(args.input_path, config=cfg)
        if args.output_dir is not None:
            report.write_bundle(args.output_dir, lang=args.lang)

        if args.format == "markdown":
            _safe_print(report.to_markdown(lang=args.lang))
        elif args.format == "json":
            _safe_print(report.to_json())
        elif args.format == "html":
            _safe_print(report.to_html(lang=args.lang))
        else:
            _safe_print(report.summary(lang=args.lang))

        if report.overall_status == "FAILED":
            return 1
        if cfg.fail_on_outage and (report.has_outages or not report.passed):
            return 1
        return 0
    except Exception as exc:
        _safe_print(f"ERROR: {_sanitize_text(exc, extra_paths=(args.input_path, args.db_path))}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
