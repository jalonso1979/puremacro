"""Own-data sovereign default risk, fiscal reaction, and stochastic DSA brief.

Ingests a user-supplied CSV or :class:`pandas.DataFrame` of quarterly or annual
macroeconomic and fiscal series with explicit column, unit, frequency, and
rate-convention mappings; estimates empirical endowment dynamics ``(rho_y, sigma_y)``,
a Bohn fiscal reaction function, and a VAR-based Stochastic Debt Sustainability
Analysis (DSA) fan chart; and runs an Arellano / Chatterjee-Eyigungor sovereign
default sensitivity grid over explicit structural assumptions (patience ``beta``,
risk-free rate ``r_star``, reentry probability ``theta``, and default penalty).

Exports bilingual (``en`` / ``es``) Markdown summaries and tables, CSV tables,
a 2-panel comparison figure, a portable JSON record, and a SHA-256
``manifest.json`` via :mod:`puremacro.examples._brief_record`.

Run via CLI::

    python -m puremacro.examples.sovereign_own_data_brief --data macro.csv --output ./out
    python -m puremacro.examples.sovereign_own_data_brief --demo --output ./out --lang es
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from puremacro.examples import _brief_record
from puremacro.sovereign._results import (
    SovereignDefaultResult,
    StochasticDSAResult,
)
from puremacro.sovereign.arellano import solve_sovereign_default
from puremacro.sovereign.dsa import (
    FiscalReactionEstimateResult,
    estimate_fiscal_reaction_function,
    fit_stochastic_dsa_from_data,
    simulate_stochastic_dsa,
)
from puremacro.sovereign.measurement import ergodic_moments, value_of_market_access

Frequency = Literal["quarterly", "annual"]
RateConvention = Literal["apr", "effective"]
RateUnit = Literal["pct", "fraction"]
DebtUnit = Literal["fraction", "percent"]
SpreadUnit = Literal["pct", "bps"]
DetrendMethod = Literal["hp", "linear", "quad", "none"]
PenaltyType = Literal["arellano", "chatterjee_eyigungor"]
PolicySearch = Literal["monotone", "dense", "grid"]
Language = Literal["en", "es"]

FREQUENCIES: tuple[Frequency, ...] = ("quarterly", "annual")
RATE_CONVENTIONS: tuple[RateConvention, ...] = ("apr", "effective")
RATE_UNITS: tuple[RateUnit, ...] = ("pct", "fraction")
DEBT_UNITS: tuple[DebtUnit, ...] = ("fraction", "percent")
SPREAD_UNITS: tuple[SpreadUnit, ...] = ("pct", "bps")
DETREND_METHODS: tuple[DetrendMethod, ...] = ("hp", "linear", "quad", "none")
PENALTY_TYPES: tuple[PenaltyType, ...] = ("arellano", "chatterjee_eyigungor")
POLICY_SEARCH_MODES: tuple[PolicySearch, ...] = ("monotone", "dense", "grid")

MIN_OBS_REQUIRED: dict[Frequency, int] = {"quarterly": 12, "annual": 8}
SHORT_SAMPLE_OBS: dict[Frequency, int] = {"quarterly": 40, "annual": 12}

_QUARTERLY_RE = re.compile(r"^(\d{4})[-/\s]?[QqTt](\d)$")
_ANNUAL_RE = re.compile(r"^(\d{4})$")

_KNOWN_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "date_col",
        "filter_col",
        "filter_value",
        "frequency",
        "rate_convention",
        "exact_compounding",
        "rate_unit",
        "debt_unit",
        "spread_unit",
        "growth_col",
        "interest_rate_col",
        "primary_balance_col",
        "debt_col",
        "income_col",
        "income_is_log",
        "spread_col",
        "y_gap_col",
        "detrend",
        "hp_lambda",
        "beta",
        "gamma",
        "r_star",
        "theta",
        "default_penalty_type",
        "penalty_d0",
        "penalty_d1",
        "sigma_eps",
        "n_b",
        "n_y",
        "b_min",
        "b_max",
        "tol",
        "max_iter",
        "policy_search",
        "sim_periods",
        "sim_seeds",
        "sim_burn",
        "sensitivity_theta",
        "sensitivity_sigma_scale",
        "dsa_horizon",
        "dsa_n_sims",
        "dsa_seed",
        "dsa_threshold",
        "d0",
        "use_bohn_reaction",
        "cubic_fatigue",
        "pb_max",
        "lang",
        "dataset_label",
        "citation",
        "title_en",
        "title_es",
        "assumptions_en",
        "assumptions_es",
    }
)


@dataclass(frozen=True)
class SovereignBriefConfig:
    """Explicit data mapping, convention, and structural assumptions for sovereign brief."""

    date_col: str = "date"
    filter_col: str | None = None
    filter_value: str | None = None
    frequency: Frequency = "quarterly"
    rate_convention: RateConvention = "apr"
    exact_compounding: bool = False
    rate_unit: RateUnit = "pct"
    debt_unit: DebtUnit = "fraction"
    spread_unit: SpreadUnit = "pct"

    growth_col: str = "g"
    interest_rate_col: str = "r"
    primary_balance_col: str = "pb"
    debt_col: str = "debt"
    income_col: str | None = None
    income_is_log: bool = False
    spread_col: str | None = None
    y_gap_col: str | None = None

    detrend: DetrendMethod = "hp"
    hp_lambda: float | None = None

    # Explicit structural sovereign-default assumptions (not estimated from macro series)
    beta: float = 0.953
    gamma: float = 2.0
    r_star: float = 0.017
    theta: float = 0.282
    default_penalty_type: PenaltyType = "arellano"
    penalty_d0: float = -0.188
    penalty_d1: float = 0.245
    sigma_eps: float | None = None
    n_b: int = 81
    n_y: int = 21
    b_min: float = -0.15
    b_max: float = 0.35
    tol: float = 1e-8
    max_iter: int = 2000
    policy_search: PolicySearch = "monotone"
    sim_periods: int = 20_000
    sim_seeds: tuple[int, ...] = (1979, 1980, 1981, 1982)
    sim_burn: int = 500
    sensitivity_theta: tuple[float, ...] = (0.15, 0.282, 0.45)
    sensitivity_sigma_scale: tuple[float, ...] = (0.8, 1.0, 1.25)

    # Stochastic DSA configuration
    dsa_horizon: int = 20
    dsa_n_sims: int = 500
    dsa_seed: int = 42
    dsa_threshold: float = 0.80
    d0: float | None = None
    use_bohn_reaction: bool = True
    cubic_fatigue: bool = False
    pb_max: float | None = 5.0

    # Bilingual display and provenance metadata
    lang: Language = "en"
    dataset_label: str = "User sovereign macro dataset"
    citation: str = "User-supplied macroeconomic and fiscal time series"
    title_en: str = "Own-Data Sovereign Default Risk & Debt Sustainability Brief"
    title_es: str = "Informe de Riesgo Soberano y Sostenibilidad de Deuda con Datos Propios"
    assumptions_en: str = (
        "Default penalty, discount factor (beta), risk-free rate (r*), and reentry probability "
        "(theta) are imposed structural assumptions evaluated across explicit sensitivity grids; "
        "only the endowment AR(1), Bohn fiscal reaction, and macro VAR are estimated from data."
    )
    assumptions_es: str = (
        "La penalizacion por incumplimiento, el factor de descuento (beta), la tasa libre de "
        "riesgo (r*) y la probabilidad de reingreso (theta) son supuestos estructurales impuestos "
        "y evaluados mediante mallas de sensibilidad; solo el AR(1) de dotacion, la reaccion "
        "fiscal de Bohn y el VAR macroeconomico se estiman a partir de los datos."
    )

    def __post_init__(self) -> None:
        if self.frequency not in FREQUENCIES:
            raise ValueError(f"Unsupported frequency={self.frequency!r}; expected {FREQUENCIES}.")
        if self.rate_convention not in RATE_CONVENTIONS:
            raise ValueError(
                f"Unsupported rate_convention={self.rate_convention!r}; expected {RATE_CONVENTIONS}."
            )
        if self.rate_unit not in RATE_UNITS:
            raise ValueError(f"Unsupported rate_unit={self.rate_unit!r}; expected {RATE_UNITS}.")
        if self.debt_unit not in DEBT_UNITS:
            raise ValueError(f"Unsupported debt_unit={self.debt_unit!r}; expected {DEBT_UNITS}.")
        if self.spread_unit not in SPREAD_UNITS:
            raise ValueError(f"Unsupported spread_unit={self.spread_unit!r}; expected {SPREAD_UNITS}.")
        if self.detrend not in DETREND_METHODS:
            raise ValueError(f"Unsupported detrend={self.detrend!r}; expected {DETREND_METHODS}.")
        if self.default_penalty_type not in PENALTY_TYPES:
            raise ValueError(
                f"Unsupported default_penalty_type={self.default_penalty_type!r}; expected {PENALTY_TYPES}."
            )
        if self.policy_search not in POLICY_SEARCH_MODES:
            raise ValueError(
                f"Unsupported policy_search={self.policy_search!r}; expected {POLICY_SEARCH_MODES}."
            )
        if self.lang not in ("en", "es"):
            raise ValueError(f"Unsupported lang={self.lang!r}; expected 'en' or 'es'.")

        for cname in (
            self.date_col,
            self.growth_col,
            self.interest_rate_col,
            self.primary_balance_col,
            self.debt_col,
        ):
            if not str(cname).strip():
                raise ValueError("Required column names must be non-empty strings.")

        if (self.filter_col is None) != (self.filter_value is None):
            raise ValueError("filter_col and filter_value must either both be set or both be None.")

        lam = (
            float(self.hp_lambda)
            if self.hp_lambda is not None
            else (1600.0 if self.frequency == "quarterly" else 6.25)
        )
        if not math.isfinite(lam) or lam <= 0.0:
            raise ValueError(f"hp_lambda must be finite and > 0; got {self.hp_lambda!r}.")
        object.__setattr__(self, "hp_lambda", lam)

        if not (0.0 < float(self.beta) < 1.0):
            raise ValueError(f"beta must lie strictly in (0, 1); got {self.beta!r}.")
        if not (math.isfinite(float(self.gamma)) and float(self.gamma) > 0.0):
            raise ValueError(f"gamma must be finite and > 0; got {self.gamma!r}.")
        if not (math.isfinite(float(self.r_star)) and float(self.r_star) >= 0.0):
            raise ValueError(f"r_star must be finite and >= 0; got {self.r_star!r}.")
        if not (0.0 < float(self.theta) <= 1.0):
            raise ValueError(f"theta must lie in (0, 1]; got {self.theta!r}.")

        sig_eps = (
            float(self.sigma_eps)
            if self.sigma_eps is not None
            else (0.0 if self.default_penalty_type == "arellano" else 0.01)
        )
        if not math.isfinite(sig_eps) or sig_eps < 0.0:
            raise ValueError(f"sigma_eps must be finite and >= 0; got {self.sigma_eps!r}.")
        object.__setattr__(self, "sigma_eps", sig_eps)

        if int(self.n_b) < 21 or int(self.n_y) < 5:
            raise ValueError(f"Grid sizes too small: n_b={self.n_b} (>=21), n_y={self.n_y} (>=5).")
        if float(self.b_min) >= float(self.b_max):
            raise ValueError(f"b_min ({self.b_min}) must be < b_max ({self.b_max}).")
        if int(self.sim_periods) <= int(self.sim_burn):
            raise ValueError("sim_periods must exceed sim_burn.")
        seeds = tuple(int(s) for s in self.sim_seeds)
        if not seeds:
            raise ValueError("sim_seeds cannot be empty.")
        object.__setattr__(self, "sim_seeds", seeds)

        sens_t = tuple(float(x) for x in self.sensitivity_theta)
        sens_s = tuple(float(x) for x in self.sensitivity_sigma_scale)
        if not sens_t or any(not (0.0 < x <= 1.0) for x in sens_t):
            raise ValueError(f"sensitivity_theta values must be in (0, 1]; got {sens_t!r}.")
        if not sens_s or any(not (math.isfinite(x) and x > 0.0) for x in sens_s):
            raise ValueError(f"sensitivity_sigma_scale values must be positive; got {sens_s!r}.")
        object.__setattr__(self, "sensitivity_theta", sens_t)
        object.__setattr__(self, "sensitivity_sigma_scale", sens_s)

        if int(self.dsa_horizon) < 1 or int(self.dsa_n_sims) < 10:
            raise ValueError("dsa_horizon >= 1 and dsa_n_sims >= 10 are required.")
        if not (math.isfinite(float(self.dsa_threshold)) and float(self.dsa_threshold) > 0.0):
            raise ValueError(f"dsa_threshold must be > 0; got {self.dsa_threshold!r}.")
        if self.d0 is not None and (not math.isfinite(float(self.d0)) or float(self.d0) < 0.0):
            raise ValueError(f"d0 must be finite and >= 0 when supplied; got {self.d0!r}.")


@dataclass
class PreparedSovereignData:
    """Validated, regular-calendar macroeconomic dataset ready for sovereign estimation."""

    df: pd.DataFrame
    period_labels: tuple[str, ...]
    n_obs: int
    start_period: str
    end_period: str
    rho_y: float
    rho_y_raw: float
    sigma_y: float
    rho_clipped: bool
    income_source: str
    empirical_spread_mean: float | None
    empirical_spread_sd: float | None
    warnings_en: list[str]
    warnings_es: list[str]
    input_file: str | None = None
    input_sha256: str | None = None


@dataclass
class SovereignBriefResult:
    """Full structured result of the own-data sovereign risk and DSA brief."""

    config: SovereignBriefConfig
    data: PreparedSovereignData
    dsa: StochasticDSAResult
    fiscal_reaction: FiscalReactionEstimateResult
    baseline_arellano: SovereignDefaultResult
    baseline_moments: pd.DataFrame
    baseline_vma: float
    sensitivity: pd.DataFrame
    var_max_eig: float
    warnings_en: list[str] = field(default_factory=list)
    warnings_es: list[str] = field(default_factory=list)


_TEXT: dict[Language, dict[str, str]] = {
    "en": {
        "exec": "Executive Summary & Explicit Mapping",
        "endowment": "Empirical Endowment Dynamics & Structural Calibration",
        "dsa": "Stochastic Debt Sustainability Analysis (DSA)",
        "bohn": "Estimated Bohn Fiscal Reaction Function",
        "sens": "Sovereign Default Sensitivity Grid (Explicit Structural Assumptions)",
        "assump": "Structural Assumptions vs. Empirical Estimates",
        "warn": "Diagnostic Guards & Short-Sample Warnings",
        "prov": "Data & Code Provenance",
        "fig_left": "Stochastic Debt-to-GDP Fan Chart",
        "fig_right": "Model Spread Sensitivity vs. Reentry (theta) & Vol",
        "fig_xlabel_q": "Horizon (quarters)",
        "fig_xlabel_a": "Horizon (years)",
        "fig_ylabel_d": "Public debt / GDP ratio",
        "fig_ylabel_s": "Mean sovereign spread (%)",
        "threshold": "Prudential threshold",
        "median": "Median path",
        "det": "Deterministic baseline",
        "band90": "5th-95th pct",
        "band50": "25th-75th pct",
    },
    "es": {
        "exec": "Resumen Ejecutivo y Mapeo Explicito",
        "endowment": "Dinamica Empirica de Dotacion y Calibracion Estructural",
        "dsa": "Analisis Estocastico de Sostenibilidad de Deuda (DSA)",
        "bohn": "Funcion de Reaccion Fiscal de Bohn Estimada",
        "sens": "Malla de Sensibilidad de Default Soberano (Supuestos Estructurales Explicitos)",
        "assump": "Supuestos Estructurales vs. Estimaciones Empiricas",
        "warn": "Guardias de Diagnostico y Advertencias de Muestra Corta",
        "prov": "Trazabilidad de Datos y Codigo",
        "fig_left": "Grafico de Abanico de Deuda/PIB Estocastica",
        "fig_right": "Sensibilidad de Spread vs. Reingreso (theta) y Volatilidad",
        "fig_xlabel_q": "Horizonte (trimestres)",
        "fig_xlabel_a": "Horizonte (anos)",
        "fig_ylabel_d": "Deuda publica / PIB",
        "fig_ylabel_s": "Spread soberano medio (%)",
        "threshold": "Umbral prudencial",
        "median": "Mediana",
        "det": "Trayectoria deterministica",
        "band90": "Percentil 5-95",
        "band50": "Percentil 25-75",
    },
}


def load_config(path: str | Path) -> SovereignBriefConfig:
    """Load and validate :class:`SovereignBriefConfig` from a UTF-8 JSON file."""
    cfg_path = Path(path)
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config JSON in {cfg_path.name} must be an object.")
    unknown = set(raw.keys()) - _KNOWN_CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys in {cfg_path.name}: {sorted(unknown)}")

    kwargs: dict[str, Any] = dict(raw)
    for key in ("sim_seeds",):
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = tuple(int(x) for x in kwargs[key])
    for key in ("sensitivity_theta", "sensitivity_sigma_scale"):
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = tuple(float(x) for x in kwargs[key])
    return SovereignBriefConfig(**kwargs)


def parse_period_index(raw_dates: Sequence[Any], frequency: Frequency) -> pd.PeriodIndex:
    """Parse quarterly or annual dates strictly and check unique values."""
    freq_code = "Q" if frequency == "quarterly" else "Y"
    periods: list[pd.Period] = []

    for item in raw_dates:
        if item is None or (isinstance(item, float) and math.isnan(item)):
            raise ValueError("Date column contains missing/NaN values.")
        text = str(item).strip()
        if not text:
            raise ValueError("Date column contains blank entries.")
        if frequency == "quarterly":
            m = _QUARTERLY_RE.match(text)
            if m:
                yr, qtr = int(m.group(1)), int(m.group(2))
                if qtr < 1 or qtr > 4:
                    raise ValueError(f"Invalid quarter in date {text!r}.")
                periods.append(pd.Period(year=yr, quarter=qtr, freq="Q"))
                continue
            try:
                ts = pd.Timestamp(text)
            except Exception as exc:
                raise ValueError(f"Unparseable quarterly date {text!r}.") from exc
            periods.append(ts.to_period("Q"))
        else:
            m_yr = _ANNUAL_RE.match(text)
            if m_yr:
                periods.append(pd.Period(year=int(m_yr.group(1)), freq="Y"))
                continue
            try:
                ts = pd.Timestamp(text)
            except Exception as exc:
                raise ValueError(f"Unparseable annual date {text!r}.") from exc
            periods.append(ts.to_period("Y"))

    pidx = pd.PeriodIndex(periods, freq=freq_code)
    if pidx.has_duplicates:
        dups = pidx[pidx.duplicated()].astype(str).tolist()
        raise ValueError(f"Duplicate {frequency} periods detected: {dups[:5]}.")
    return pidx


def _hp_cycle(series: np.ndarray, lam: float) -> np.ndarray:
    """Two-sided Whittaker-Henderson Hodrick-Prescott cyclical component."""
    n = len(series)
    if n < 4:
        return series - np.mean(series)
    eye = np.eye(n)
    d2 = np.zeros((n - 2, n), dtype=float)
    for i in range(n - 2):
        d2[i, i] = 1.0
        d2[i, i + 1] = -2.0
        d2[i, i + 2] = 1.0
    trend = np.linalg.solve(eye + lam * (d2.T @ d2), series)
    return np.asarray(series - trend, dtype=float)


def _detrend_series(values: np.ndarray, method: DetrendMethod, hp_lambda: float) -> np.ndarray:
    t = np.arange(len(values), dtype=float)
    if method == "hp":
        return _hp_cycle(values, hp_lambda)
    if method == "linear":
        coef = np.polyfit(t, values, 1)
        return np.asarray(values - np.polyval(coef, t), dtype=float)
    if method == "quad":
        coef = np.polyfit(t, values, 2)
        return np.asarray(values - np.polyval(coef, t), dtype=float)
    return np.asarray(values - np.mean(values), dtype=float)


def _estimate_endowment_ar1(cycle: np.ndarray) -> tuple[float, float, float, bool]:
    """Estimate y_t = c + rho * y_{t-1} + eps_t and guard non-stationary domain."""
    if len(cycle) < 6:
        raise ValueError("At least 6 cyclical observations are required to estimate AR(1) endowment.")
    y_lag = cycle[:-1]
    y_cur = cycle[1:]
    x_mat = np.column_stack([np.ones(len(y_lag), dtype=float), y_lag])
    coef, _, _, _ = np.linalg.lstsq(x_mat, y_cur, rcond=None)
    resid = y_cur - x_mat @ coef
    sigma_y = float(np.std(resid, ddof=2))
    if not math.isfinite(sigma_y) or sigma_y < 1e-5:
        raise ValueError(
            f"Endowment residual standard deviation is degenerate ({sigma_y!r}); check income/growth input."
        )
    rho_raw = float(coef[1])
    rho_clipped = False
    rho_use = rho_raw
    if rho_raw >= 0.985 or rho_raw <= 0.10:
        rho_clipped = True
        rho_use = float(min(max(rho_raw, 0.50), 0.985))
    return rho_use, rho_raw, sigma_y, rho_clipped


def prepare_sovereign_data(
    df_or_path: pd.DataFrame | str | Path,
    config: SovereignBriefConfig,
) -> PreparedSovereignData:
    """Validate macro CSV/DataFrame, normalize units cleanly, and estimate endowment AR(1)."""
    input_file: str | None = None
    input_sha: str | None = None
    if isinstance(df_or_path, (str, Path)):
        src_path = Path(df_or_path)
        if not src_path.is_file():
            raise FileNotFoundError(f"Sovereign data CSV not found: {src_path.name}")
        raw_df = pd.read_csv(src_path, encoding="utf-8")
        input_file = src_path.name
        input_sha = _brief_record.sha256(src_path)
    elif isinstance(df_or_path, pd.DataFrame):
        raw_df = df_or_path.copy()
    else:
        raise TypeError(f"Expected DataFrame or path, got {type(df_or_path).__name__}.")

    if config.filter_col is not None:
        if config.filter_col not in raw_df.columns:
            raise ValueError(f"filter_col {config.filter_col!r} not found in columns: {list(raw_df.columns)}.")
        mask = raw_df[config.filter_col].astype(str) == str(config.filter_value)
        raw_df = raw_df.loc[mask].copy()
        if raw_df.empty:
            raise ValueError(
                f"Filter {config.filter_col!r} == {config.filter_value!r} matched zero rows."
            )

    req_cols = [
        config.date_col,
        config.growth_col,
        config.interest_rate_col,
        config.primary_balance_col,
        config.debt_col,
    ]
    if config.income_col is not None:
        req_cols.append(config.income_col)
    if config.spread_col is not None:
        req_cols.append(config.spread_col)
    if config.y_gap_col is not None:
        req_cols.append(config.y_gap_col)

    missing = [c for c in req_cols if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing mapped column(s) in dataset: {missing}.")

    pidx = parse_period_index(raw_df[config.date_col].tolist(), config.frequency)
    raw_df = raw_df.copy()
    raw_df["__period__"] = pidx
    raw_df = raw_df.sort_values("__period__").reset_index(drop=True)
    sorted_pidx = pd.PeriodIndex(raw_df["__period__"], freq="Q" if config.frequency == "quarterly" else "Y")

    # Enforce regular contiguous calendar spacing (zero silent gap interpolation)
    diffs = sorted_pidx[1:].asi8 - sorted_pidx[:-1].asi8
    if np.any(diffs != 1):
        bad_idx = int(np.where(diffs != 1)[0][0])
        raise ValueError(
            f"Non-contiguous {config.frequency} dates between {sorted_pidx[bad_idx]} and {sorted_pidx[bad_idx + 1]}."
        )

    n_obs = len(raw_df)
    min_req = MIN_OBS_REQUIRED[config.frequency]
    if n_obs < min_req:
        raise ValueError(
            f"Insufficient sample length ({n_obs} {config.frequency} observations); at least {min_req} required."
        )

    numeric_cols = [c for c in req_cols if c != config.date_col]
    for col in numeric_cols:
        vals = pd.to_numeric(raw_df[col], errors="raise").to_numpy(dtype=float)
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"Column {col!r} contains NaN or non-finite values.")
        raw_df[col] = vals

    rate_mult = 100.0 if config.rate_unit == "fraction" else 1.0
    debt_div = 100.0 if config.debt_unit == "percent" else 1.0
    spread_div = 100.0 if config.spread_unit == "bps" else 1.0

    norm_df = pd.DataFrame(
        {
            "period": sorted_pidx.astype(str),
            "g": raw_df[config.growth_col].to_numpy(dtype=float) * rate_mult,
            "r": raw_df[config.interest_rate_col].to_numpy(dtype=float) * rate_mult,
            "pb": raw_df[config.primary_balance_col].to_numpy(dtype=float) * rate_mult,
            "debt": raw_df[config.debt_col].to_numpy(dtype=float) / debt_div,
        }
    )
    if np.any(norm_df["debt"].to_numpy(dtype=float) < 0.0):
        raise ValueError("Debt-to-GDP ratio cannot be negative.")
    if config.y_gap_col is not None:
        norm_df["y_gap"] = raw_df[config.y_gap_col].to_numpy(dtype=float) * rate_mult

    warnings_en: list[str] = []
    warnings_es: list[str] = []

    short_thresh = SHORT_SAMPLE_OBS[config.frequency]
    if n_obs < short_thresh:
        warnings_en.append(
            f"Short sample warning: {n_obs} {config.frequency} observations (< {short_thresh}); "
            "VAR(1) dynamics, Bohn fiscal coefficients, and AR(1) persistence carry elevated finite-sample uncertainty."
        )
        warnings_es.append(
            f"Advertencia de muestra corta: {n_obs} observaciones ({config.frequency} < {short_thresh}); "
            "el VAR(1), la reaccion fiscal de Bohn y la persistencia AR(1) presentan mayor incertidumbre de muestra finita."
        )

    if config.frequency == "annual" and (
        math.isclose(config.beta, 0.953, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(config.r_star, 0.017, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(config.theta, 0.282, rel_tol=0.0, abs_tol=1e-6)
    ):
        warnings_en.append(
            "Annual data frequency selected with default quarterly Arellano structural parameters "
            "(beta=0.953, r_star=0.017, theta=0.282); override --beta/--r-star/--theta for annual calibration."
        )
        warnings_es.append(
            "Se selecciono frecuencia anual con parametros estructurales trimestrales por defecto "
            "(beta=0.953, r_star=0.017, theta=0.282); ajuste --beta/--r-star/--theta para calibracion anual."
        )

    assert config.hp_lambda is not None
    if config.income_col is not None:
        inc_vals = raw_df[config.income_col].to_numpy(dtype=float)
        if config.income_is_log:
            log_y = inc_vals
        else:
            if np.any(inc_vals <= 0.0):
                raise ValueError(f"Level income_col {config.income_col!r} must be strictly positive.")
            log_y = np.log(inc_vals)
        income_source = f"{config.income_col} ({config.detrend})"
    else:
        step_scale = 400.0 if config.frequency == "quarterly" else 100.0
        g_step = (norm_df["g"].to_numpy(dtype=float) - float(np.mean(norm_df["g"]))) / step_scale
        log_y = np.cumsum(g_step)
        income_source = f"cumulated demeaned {config.growth_col} proxy ({config.detrend})"
        warnings_en.append(
            "No income_col mapped; endowment AR(1) (rho_y, sigma_y) was reconstructed from demeaned growth rates."
        )
        warnings_es.append(
            "No se mapeo income_col; el AR(1) de dotacion (rho_y, sigma_y) se reconstruyo a partir de las tasas de crecimiento centradas."
        )

    cycle = _detrend_series(log_y, config.detrend, float(config.hp_lambda))
    rho_y, rho_raw, sigma_y, rho_clipped = _estimate_endowment_ar1(cycle)
    if rho_clipped:
        warnings_en.append(
            f"Estimated endowment AR(1) persistence rho_y={rho_raw:.4f} lies outside stationary default bounds "
            f"[0.50, 0.985]; bounded to {rho_y:.4f} for Arellano solver stability."
        )
        warnings_es.append(
            f"La persistencia estimada AR(1) rho_y={rho_raw:.4f} excede el rango estacionario [0.50, 0.985]; "
            f"se acoto explicitamente a {rho_y:.4f} para resolver el equilibrio de Arellano."
        )

    emp_spread_mean: float | None = None
    emp_spread_sd: float | None = None
    if config.spread_col is not None:
        sp = raw_df[config.spread_col].to_numpy(dtype=float) / spread_div
        emp_spread_mean = float(np.mean(sp))
        emp_spread_sd = float(np.std(sp, ddof=1))

    return PreparedSovereignData(
        df=norm_df,
        period_labels=tuple(sorted_pidx.astype(str).tolist()),
        n_obs=n_obs,
        start_period=str(sorted_pidx[0]),
        end_period=str(sorted_pidx[-1]),
        rho_y=rho_y,
        rho_y_raw=rho_raw,
        sigma_y=sigma_y,
        rho_clipped=rho_clipped,
        income_source=income_source,
        empirical_spread_mean=emp_spread_mean,
        empirical_spread_sd=emp_spread_sd,
        warnings_en=warnings_en,
        warnings_es=warnings_es,
        input_file=input_file,
        input_sha256=input_sha,
    )


def _solve_arellano_case(
    config: SovereignBriefConfig,
    *,
    rho_y: float,
    sigma_y: float,
    theta: float,
) -> tuple[SovereignDefaultResult, pd.DataFrame, float]:
    assert config.sigma_eps is not None
    res = solve_sovereign_default(
        beta=config.beta,
        gamma=config.gamma,
        r_star=config.r_star,
        theta=theta,
        rho_y=rho_y,
        sigma_y=sigma_y,
        d0=config.penalty_d0,
        d1=config.penalty_d1,
        sigma_eps=float(config.sigma_eps),
        n_b=config.n_b,
        n_y=config.n_y,
        b_min=config.b_min,
        b_max=config.b_max,
        tol=config.tol,
        max_iter=config.max_iter,
        y_penalty_type=config.default_penalty_type,
        enforce_monotonicity=(config.default_penalty_type != "arellano"),
        policy_search=config.policy_search,
    )
    moms = ergodic_moments(
        res,
        periods=config.sim_periods,
        seeds=config.sim_seeds,
        burn=config.sim_burn,
    )
    vma = float(value_of_market_access(res))
    return res, moms, vma


def compute_brief(
    df_or_path: pd.DataFrame | str | Path,
    config: SovereignBriefConfig | None = None,
) -> SovereignBriefResult:
    """Run full own-data sovereign estimation, DSA simulation, and Arellano sensitivity."""
    cfg = config if config is not None else SovereignBriefConfig()
    data = prepare_sovereign_data(df_or_path, cfg)
    warnings_en = list(data.warnings_en)
    warnings_es = list(data.warnings_es)

    y_gap_name = "y_gap" if cfg.y_gap_col is not None else None
    fr_res = estimate_fiscal_reaction_function(
        data.df,
        pb_col="pb",
        debt_col="debt",
        y_gap_col=y_gap_name,
        cubic_fatigue=cfg.cubic_fatigue,
        pb_max=cfg.pb_max,
    )

    dsa_fit = fit_stochastic_dsa_from_data(
        data.df,
        g_col="g",
        r_col="r",
        pb_col="pb",
        debt_col="debt",
        y_gap_col=y_gap_name,
        d0=cfg.d0,
        horizon=cfg.dsa_horizon,
        n_sims=cfg.dsa_n_sims,
        seed=cfg.dsa_seed,
        threshold=cfg.dsa_threshold,
        use_bohn_reaction=cfg.use_bohn_reaction,
        cubic_fatigue=cfg.cubic_fatigue,
        frequency=cfg.frequency,
    )

    needs_resim = (
        cfg.rate_convention != "apr"
        or cfg.exact_compounding
        or (cfg.use_bohn_reaction and cfg.pb_max != 5.0)
    )
    if needs_resim:
        dsa = simulate_stochastic_dsa(
            d0=float(dsa_fit.d0),
            horizon=cfg.dsa_horizon,
            n_sims=cfg.dsa_n_sims,
            seed=cfg.dsa_seed,
            mu=np.asarray(dsa_fit.metadata["var_mu"], dtype=float),
            A=np.asarray(dsa_fit.metadata["var_A"], dtype=float),
            cov=np.asarray(dsa_fit.metadata["var_Sigma"], dtype=float),
            threshold=cfg.dsa_threshold,
            fiscal_reaction=fr_res.reaction_function if cfg.use_bohn_reaction else None,
            frequency=cfg.frequency,
            exact_compounding=cfg.exact_compounding,
            rate_convention=cfg.rate_convention,
            store_macro_paths=False,
        )
        # Preserve empirical VAR/Bohn metadata alongside updated compounding/convention/pb_max paths
        merged_meta = dict(dsa_fit.metadata)
        merged_meta.update(
            {
                "frequency": cfg.frequency,
                "exact_compounding": cfg.exact_compounding,
                "rate_convention": cfg.rate_convention,
                "fiscal_reaction_estimate": fr_res,
                "pb_max": cfg.pb_max,
            }
        )
        dsa = StochasticDSAResult(
            paths=dsa.paths,
            baseline_path=dsa.baseline_path,
            d0=dsa.d0,
            threshold=dsa.threshold,
            horizon=dsa.horizon,
            frequency=cfg.frequency,
            quantiles=dsa.quantiles,
            prob_breach=dsa.prob_breach,
            prob_terminal_breach=dsa.prob_terminal_breach,
            prob_non_stabilizing=dsa.prob_non_stabilizing,
            dar_95=dsa.dar_95,
            macro_paths=dsa.macro_paths,
            metadata=merged_meta,
        )
    else:
        dsa = dsa_fit

    var_a = np.asarray(dsa.metadata["var_A"], dtype=float)
    var_max_eig = float(np.max(np.abs(np.linalg.eigvals(var_a))))
    if var_max_eig >= 0.98:
        warnings_en.append(
            f"Macro VAR(1) companion matrix spectral radius is near-unit-root ({var_max_eig:.3f})."
        )
        warnings_es.append(
            f"El radio espectral del VAR(1) macroeconomico esta cercano a raiz unitaria ({var_max_eig:.3f})."
        )

    if not fr_res.bohn_solvency_holds:
        warnings_en.append(
            f"Bohn fiscal solvency test is not significant at 5% (t={fr_res.bohn_t_stat:.2f}, p={fr_res.bohn_p_value_one_sided:.3f})."
        )
        warnings_es.append(
            f"La prueba de solvencia fiscal de Bohn no es significativa al 5% (t={fr_res.bohn_t_stat:.2f}, p={fr_res.bohn_p_value_one_sided:.3f})."
        )

    base_res, base_moms, base_vma = _solve_arellano_case(
        cfg, rho_y=data.rho_y, sigma_y=data.sigma_y, theta=cfg.theta
    )
    if not base_res.converged:
        warnings_en.append(
            f"Baseline sovereign model did not reach tolerance {cfg.tol:.1e} in {cfg.max_iter} iterations."
        )
        warnings_es.append(
            f"El modelo soberano base no alcanzo tolerancia {cfg.tol:.1e} en {cfg.max_iter} iteraciones."
        )

    sens_rows: list[dict[str, Any]] = []
    for theta_val in cfg.sensitivity_theta:
        for sig_scale in cfg.sensitivity_sigma_scale:
            sig_use = data.sigma_y * sig_scale
            res_s, moms_s, vma_s = _solve_arellano_case(
                cfg, rho_y=data.rho_y, sigma_y=sig_use, theta=theta_val
            )
            sens_rows.append(
                {
                    "theta": float(theta_val),
                    "sigma_scale": float(sig_scale),
                    "sigma_y": float(sig_use),
                    "rho_y": float(data.rho_y),
                    "converged": bool(res_s.converged),
                    "iterations": int(res_s.iterations),
                    "debt_output_pct": float(moms_s.loc["debt_output_pct", "mean"]),
                    "spread_mean_pct": float(moms_s.loc["spread_mean_pct", "mean"]),
                    "default_annual_pct": float(moms_s.loc["default_annual_pct", "mean"]),
                    "spread_sd_pp": float(moms_s.loc["spread_sd_pp", "mean"]),
                    "value_of_market_access": float(vma_s),
                }
            )
    sens_df = pd.DataFrame(sens_rows)

    return SovereignBriefResult(
        config=cfg,
        data=data,
        dsa=dsa,
        fiscal_reaction=fr_res,
        baseline_arellano=base_res,
        baseline_moments=base_moms,
        baseline_vma=base_vma,
        sensitivity=sens_df,
        var_max_eig=var_max_eig,
        warnings_en=warnings_en,
        warnings_es=warnings_es,
    )


def _df_to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "_(empty)_"
    fmt = df.copy()
    for col in fmt.columns:
        if pd.api.types.is_bool_dtype(fmt[col]) or pd.api.types.is_integer_dtype(fmt[col]):
            continue
        if pd.api.types.is_numeric_dtype(fmt[col]):
            fmt[col] = fmt[col].map(lambda v: f"{float(v):.{digits}f}" if pd.notna(v) else "")
    cols = [str(c) for c in fmt.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(r[c]) for c in fmt.columns) + " |" for _, r in fmt.iterrows()]
    return "\n".join([header, sep, *rows])


def _fiscal_reaction_table(fr: FiscalReactionEstimateResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k, val in fr.params.items():
        rows.append(
            {
                "parameter": k,
                "estimate": float(val),
                "hac_se": float(fr.se.get(k, float("nan"))),
                "t_stat": float(fr.t_stats.get(k, float("nan"))),
                "p_value": float(fr.p_values.get(k, float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def _dsa_paths_frame(dsa: StochasticDSAResult) -> pd.DataFrame:
    cols: dict[str, Any] = {
        "horizon": np.arange(dsa.horizon + 1, dtype=int),
        "baseline_path": np.asarray(dsa.baseline_path, dtype=float),
    }
    for q in sorted(dsa.quantiles.keys()):
        cols[f"p{q}"] = np.asarray(dsa.quantiles[q], dtype=float)
    return pd.DataFrame(cols)


def render_summary(result: SovereignBriefResult, lang: Language = "en") -> str:
    """Render concise bilingual brief summary with explicit assumptions and diagnostics."""
    if lang not in ("en", "es"):
        raise ValueError(f"Unsupported language {lang!r}; expected 'en' or 'es'.")
    cfg = result.config
    data = result.data
    dsa = result.dsa
    fr = result.fiscal_reaction
    t = _TEXT[lang]
    title = cfg.title_en if lang == "en" else cfg.title_es
    assump = cfg.assumptions_en if lang == "en" else cfg.assumptions_es
    warns = result.warnings_en if lang == "en" else result.warnings_es

    lines: list[str] = [f"# {title}", ""]
    lines.append(f"## {t['exec']}")
    lines.append("")
    if lang == "en":
        lines.append(
            f"- **Dataset**: {cfg.dataset_label} | **Span**: `{data.start_period}` to `{data.end_period}` (`n={data.n_obs}`, `{cfg.frequency}`)"
        )
        lines.append(
            f"- **Conventions**: `rate_convention={cfg.rate_convention}`, `exact_compounding={cfg.exact_compounding}`, `rate_unit={cfg.rate_unit}`, `debt_unit={cfg.debt_unit}`"
        )
        lines.append(
            f"- **Column mapping**: date=`{cfg.date_col}`, g=`{cfg.growth_col}`, r=`{cfg.interest_rate_col}`, pb=`{cfg.primary_balance_col}`, debt=`{cfg.debt_col}`"
        )
        lines.append(
            f"- **Endowment AR(1) ({data.income_source})**: `rho_y={data.rho_y:.4f}` (raw `{data.rho_y_raw:.4f}`), `sigma_y={data.sigma_y:.4f}`"
        )
    else:
        lines.append(
            f"- **Datos**: {cfg.dataset_label} | **Periodo**: `{data.start_period}` a `{data.end_period}` (`n={data.n_obs}`, `{cfg.frequency}`)"
        )
        lines.append(
            f"- **Convenciones**: `rate_convention={cfg.rate_convention}`, `exact_compounding={cfg.exact_compounding}`, `rate_unit={cfg.rate_unit}`, `debt_unit={cfg.debt_unit}`"
        )
        lines.append(
            f"- **Mapeo de columnas**: date=`{cfg.date_col}`, g=`{cfg.growth_col}`, r=`{cfg.interest_rate_col}`, pb=`{cfg.primary_balance_col}`, debt=`{cfg.debt_col}`"
        )
        lines.append(
            f"- **AR(1) de dotacion ({data.income_source})**: `rho_y={data.rho_y:.4f}` (bruto `{data.rho_y_raw:.4f}`), `sigma_y={data.sigma_y:.4f}`"
        )
    lines.append("")

    lines.append(f"## {t['dsa']}")
    lines.append("")
    lines.append(_df_to_markdown(dsa.summary(), digits=4))
    lines.append("")

    lines.append(f"## {t['bohn']}")
    lines.append("")
    if lang == "en":
        lines.append(
            f"Bohn debt-response t-stat = `{fr.bohn_t_stat:.3f}` (one-sided p = `{fr.bohn_p_value_one_sided:.4f}`, holds = `{fr.bohn_solvency_holds}`, R^2 = `{fr.r2:.3f}`)."
        )
    else:
        lines.append(
            f"Estadistico t de Bohn = `{fr.bohn_t_stat:.3f}` (p unilateral = `{fr.bohn_p_value_one_sided:.4f}`, solvencia = `{fr.bohn_solvency_holds}`, R^2 = `{fr.r2:.3f}`)."
        )
    lines.append("")
    lines.append(_df_to_markdown(_fiscal_reaction_table(fr), digits=4))
    lines.append("")

    lines.append(f"## {t['endowment']}")
    lines.append("")
    moms_tbl = result.baseline_moments.reset_index()
    lines.append(_df_to_markdown(moms_tbl[["moment", "mean", "se"]], digits=4))
    lines.append("")
    if data.empirical_spread_mean is not None:
        if lang == "en":
            lines.append(
                f"Observed spread benchmark (`{cfg.spread_col}`): mean = `{data.empirical_spread_mean:.3f}%`, sd = `{data.empirical_spread_sd:.3f} pp` vs. baseline model spread `{result.baseline_moments.loc['spread_mean_pct', 'mean']:.3f}%`."
            )
        else:
            lines.append(
                f"Referencia empirica de spread (`{cfg.spread_col}`): media = `{data.empirical_spread_mean:.3f}%`, desv. = `{data.empirical_spread_sd:.3f} pp` vs. modelo base `{result.baseline_moments.loc['spread_mean_pct', 'mean']:.3f}%`."
            )
        lines.append("")

    lines.append(f"## {t['sens']}")
    lines.append("")
    lines.append(_df_to_markdown(result.sensitivity, digits=4))
    lines.append("")

    lines.append(f"## {t['assump']}")
    lines.append("")
    lines.append(assump)
    lines.append(
        f"- `beta={cfg.beta}`, `gamma={cfg.gamma}`, `r_star={cfg.r_star}`, `theta={cfg.theta}`, `penalty={cfg.default_penalty_type}`"
    )
    lines.append("")

    if warns:
        lines.append(f"## {t['warn']}")
        lines.append("")
        for w in warns:
            lines.append(f"- {w}")
        lines.append("")

    lines.append(f"## {t['prov']}")
    lines.append("")
    lines.append(f"- **Citation**: {cfg.citation}")
    lines.append("")
    return "\n".join(lines)


def render_tables_markdown(result: SovereignBriefResult, lang: Language = "en") -> str:
    """Render full structured sovereign DSA, fiscal reaction, and sensitivity tables."""
    if lang not in ("en", "es"):
        raise ValueError(f"Unsupported language {lang!r}; expected 'en' or 'es'.")
    t = _TEXT[lang]
    title = result.config.title_en if lang == "en" else result.config.title_es
    sections = [
        f"# {title} — " + ("Tables" if lang == "en" else "Tablas"),
        "",
        f"## {t['dsa']}",
        "",
        _df_to_markdown(result.dsa.summary(), digits=4),
        "",
        f"## {t['bohn']}",
        "",
        _df_to_markdown(_fiscal_reaction_table(result.fiscal_reaction), digits=4),
        "",
        f"## {t['endowment']}",
        "",
        _df_to_markdown(result.baseline_moments.reset_index(), digits=4),
        "",
        f"## {t['sens']}",
        "",
        _df_to_markdown(result.sensitivity, digits=4),
        "",
    ]
    return "\n".join(sections)


def render_figure(result: SovereignBriefResult, path: Path, *, lang: Language = "en") -> None:
    """Render 2-panel sovereign figure (DSA fan chart + Arellano sensitivity) cleanly via Agg."""
    if lang not in ("en", "es"):
        raise ValueError(f"Unsupported language {lang!r}; expected 'en' or 'es'.")
    t = _TEXT[lang]
    dsa = result.dsa

    fig = Figure(figsize=(12, 4.8), dpi=150)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)

    # Panel 1: Stochastic DSA Fan Chart
    ax0 = axes[0]
    h = np.arange(dsa.horizon + 1, dtype=int)
    if 5 in dsa.quantiles and 95 in dsa.quantiles:
        ax0.fill_between(
            h, dsa.quantiles[5], dsa.quantiles[95], alpha=0.18, color="#2563eb", label=t["band90"]
        )
    if 25 in dsa.quantiles and 75 in dsa.quantiles:
        ax0.fill_between(
            h, dsa.quantiles[25], dsa.quantiles[75], alpha=0.35, color="#2563eb", label=t["band50"]
        )
    if 50 in dsa.quantiles:
        ax0.plot(h, dsa.quantiles[50], color="#1e3a8a", linewidth=2.0, label=t["median"])
    ax0.plot(
        h, dsa.baseline_path, color="#111827", linestyle="--", linewidth=1.6, label=t["det"]
    )
    ax0.axhline(
        dsa.threshold, color="#dc2626", linestyle=":", linewidth=1.5, label=t["threshold"]
    )
    ax0.set_title(t["fig_left"], fontsize=10)
    ax0.set_xlabel(
        t["fig_xlabel_q"] if dsa.frequency == "quarterly" else t["fig_xlabel_a"], fontsize=8.5
    )
    ax0.set_ylabel(t["fig_ylabel_d"], fontsize=8.5)
    ax0.legend(fontsize=7.5, frameon=False)

    # Panel 2: Mean spread across theta by sigma_scale
    ax1 = axes[1]
    sens = result.sensitivity
    for sig_scale, grp in sens.groupby("sigma_scale", sort=True):
        grp_s = grp.sort_values("theta")
        ax1.plot(
            grp_s["theta"].to_numpy(dtype=float),
            grp_s["spread_mean_pct"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.8,
            label=f"sigma x{float(sig_scale):.2f}",
        )
    ax1.set_title(t["fig_right"], fontsize=10)
    ax1.set_xlabel("theta", fontsize=8.5)
    ax1.set_ylabel(t["fig_ylabel_s"], fontsize=8.5)
    ax1.legend(fontsize=7.5, frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=150, format="png")
    fig.clear()


def _json_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.replace({np.nan: None})
    out: list[dict[str, Any]] = []
    for row in clean.to_dict(orient="records"):
        fixed: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, (np.floating, np.integer)):
                fixed[str(k)] = v.item()
            elif isinstance(v, np.bool_):
                fixed[str(k)] = bool(v)
            else:
                fixed[str(k)] = v
        out.append(fixed)
    return out


def build_brief_record(result: SovereignBriefResult) -> dict[str, Any]:
    """Build portable JSON-serializable sovereign record with zero host paths."""
    cfg = result.config
    data = result.data
    fr = result.fiscal_reaction
    dsa = result.dsa
    cfg_dict = asdict(cfg)
    cfg_dict["sim_seeds"] = list(cfg.sim_seeds)
    cfg_dict["sensitivity_theta"] = list(cfg.sensitivity_theta)
    cfg_dict["sensitivity_sigma_scale"] = list(cfg.sensitivity_sigma_scale)

    return {
        "schema": 1,
        "module": "puremacro.examples.sovereign_own_data_brief",
        "config": cfg_dict,
        "data_provenance": {
            "dataset_label": cfg.dataset_label,
            "citation": cfg.citation,
            "input_file": data.input_file,
            "input_sha256": data.input_sha256,
            "frequency": cfg.frequency,
            "rate_convention": cfg.rate_convention,
            "exact_compounding": cfg.exact_compounding,
            "n_obs": data.n_obs,
            "start_period": data.start_period,
            "end_period": data.end_period,
            "income_source": data.income_source,
        },
        "endowment_ar1": {
            "rho_y": data.rho_y,
            "rho_y_raw": data.rho_y_raw,
            "sigma_y": data.sigma_y,
            "rho_clipped": data.rho_clipped,
        },
        "structural_assumptions": {
            "beta": cfg.beta,
            "gamma": cfg.gamma,
            "r_star": cfg.r_star,
            "theta": cfg.theta,
            "default_penalty_type": cfg.default_penalty_type,
            "penalty_d0": cfg.penalty_d0,
            "penalty_d1": cfg.penalty_d1,
            "sigma_eps": cfg.sigma_eps,
            "note_en": cfg.assumptions_en,
            "note_es": cfg.assumptions_es,
        },
        "dsa_summary": _json_safe_records(dsa.summary()),
        "var_max_eig": result.var_max_eig,
        "fiscal_reaction": {
            "bohn_t_stat": float(fr.bohn_t_stat),
            "bohn_p_value_one_sided": float(fr.bohn_p_value_one_sided),
            "bohn_solvency_holds": bool(fr.bohn_solvency_holds),
            "r2": float(fr.r2),
            "n_obs": int(fr.n_obs),
            "params": _json_safe_records(_fiscal_reaction_table(fr)),
        },
        "baseline_sovereign": {
            "converged": bool(result.baseline_arellano.converged),
            "iterations": int(result.baseline_arellano.iterations),
            "value_of_market_access": result.baseline_vma,
            "moments": _json_safe_records(result.baseline_moments.reset_index()),
            "empirical_spread_mean": data.empirical_spread_mean,
            "empirical_spread_sd": data.empirical_spread_sd,
        },
        "sensitivity": _json_safe_records(result.sensitivity),
        "warnings": {"en": list(result.warnings_en), "es": list(result.warnings_es)},
    }


def write_outputs(result: SovereignBriefResult, output_dir: str | Path) -> dict[str, Path]:
    """Export bilingual summaries/tables, CSVs, PNG chart, record.json, and SHA-256 manifest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lang = result.config.lang

    summary_en = render_summary(result, lang="en")
    summary_es = render_summary(result, lang="es")
    summary_primary = summary_en if lang == "en" else summary_es

    written: dict[str, Path] = {}

    def _write_text(rel: str, text: str) -> Path:
        p = out / rel
        p.write_text(text, encoding="utf-8")
        written[rel] = p
        return p

    def _write_csv(rel: str, df: pd.DataFrame) -> Path:
        p = out / rel
        df.to_csv(p, index=False, encoding="utf-8")
        written[rel] = p
        return p

    _write_text("summary.md", summary_primary)
    _write_text("summary_en.md", summary_en)
    _write_text("summary_es.md", summary_es)
    _write_text("tables_en.md", render_tables_markdown(result, lang="en"))
    _write_text("tables_es.md", render_tables_markdown(result, lang="es"))

    _write_csv("dsa_summary.csv", result.dsa.summary())
    _write_csv("dsa_paths.csv", _dsa_paths_frame(result.dsa))
    _write_csv("fiscal_reaction.csv", _fiscal_reaction_table(result.fiscal_reaction))
    _write_csv("sovereign_sensitivity.csv", result.sensitivity)
    _write_csv("baseline_moments.csv", result.baseline_moments.reset_index())

    fig_path = out / "figure.png"
    render_figure(result, fig_path, lang=lang)
    written["figure.png"] = fig_path

    rec_path = out / "record.json"
    _brief_record.write_json_atomic(rec_path, build_brief_record(result))
    written["record.json"] = rec_path

    artifacts = {
        rel: {
            "sha256": _brief_record.sha256(p),
            "bytes": int(p.stat().st_size),
        }
        for rel, p in sorted(written.items())
    }
    manifest = {
        "schema": 1,
        "module": "puremacro.examples.sovereign_own_data_brief",
        "lang": lang,
        "environment": _brief_record.environment(),
        "inputs": {
            "dataset_label": result.config.dataset_label,
            "input_file": result.data.input_file,
            "input_sha256": result.data.input_sha256,
            "frequency": result.config.frequency,
            "rate_convention": result.config.rate_convention,
        },
        "artifacts": artifacts,
    }
    manifest_path = out / "manifest.json"
    _brief_record.write_json_atomic(manifest_path, manifest)
    written["manifest.json"] = manifest_path
    return written


def demo_sovereign_dataframe(*, frequency: Frequency = "quarterly", n_obs: int = 80) -> pd.DataFrame:
    """Build a deterministic synthetic sovereign macro panel for smoke runs and demos."""
    rng = np.random.default_rng(1985)
    if frequency == "quarterly":
        dates = pd.period_range("2005Q1", periods=n_obs, freq="Q").astype(str)
        scale = 400.0
        std_shock = 0.018
    else:
        dates = pd.period_range("1995", periods=min(n_obs, 30), freq="Y").astype(str)
        n_obs = len(dates)
        scale = 100.0
        std_shock = 0.025

    y_cyc = np.zeros(n_obs, dtype=float)
    g = np.zeros(n_obs, dtype=float)
    r = np.zeros(n_obs, dtype=float)
    debt = np.zeros(n_obs, dtype=float)
    pb = np.zeros(n_obs, dtype=float)
    debt[0] = 0.48

    for t in range(n_obs):
        if t > 0:
            y_cyc[t] = 0.90 * y_cyc[t - 1] + float(rng.normal(0.0, std_shock))
        # Mid-sample fiscal stress episode so debt spans ~0.48 to ~0.72 cleanly
        stress = 1.6 if (n_obs // 4 <= t < n_obs // 2) else 0.0
        g[t] = 2.2 + 0.4 * (y_cyc[t] * 100.0) - 0.8 * stress + float(rng.normal(0.0, 0.7))
        r[t] = 2.0 + 3.0 * max(0.0, debt[t] - 0.50) + float(rng.normal(0.0, 0.5))
        pb_target = -0.6 - stress + 9.0 * (debt[t] - 0.55) + 0.25 * (y_cyc[t] * 100.0)
        pb[t] = pb_target if t == 0 else (0.60 * pb[t - 1] + 0.40 * pb_target + float(rng.normal(0.0, 0.3)))
        if t + 1 < n_obs:
            debt[t + 1] = float(
                np.clip(debt[t] * (1.0 + (r[t] - g[t]) / scale) - pb[t] / scale, 0.25, 1.30)
            )
    income = np.exp(4.6 + 0.005 * np.arange(n_obs) + y_cyc)
    spread = np.clip(1.8 + 5.0 * np.maximum(0.0, debt - 0.50) - 12.0 * y_cyc, 0.5, 18.0)
    return pd.DataFrame(
        {
            "date": dates,
            "g": g,
            "r": r,
            "pb": pb,
            "debt": debt,
            "real_gdp": income,
            "spread_pct": spread,
        }
    )


def run_brief(
    df_or_path: pd.DataFrame | str | Path,
    config: SovereignBriefConfig,
    output_dir: str | Path,
) -> tuple[SovereignBriefResult, dict[str, Path]]:
    """Run own-data sovereign brief end-to-end and write portable outputs."""
    res = compute_brief(df_or_path, config)
    paths = write_outputs(res, output_dir)
    return res, paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puremacro.examples.sovereign_own_data_brief",
        description="Own-data sovereign default risk, fiscal reaction, and stochastic DSA brief.",
    )
    parser.add_argument("--config", type=Path, help="JSON config file overriding SovereignBriefConfig.")
    parser.add_argument("--data", type=Path, help="Path to user CSV macro/fiscal dataset.")
    parser.add_argument("--demo", action="store_true", help="Run with reproducible demo sovereign dataset.")
    parser.add_argument("--output", type=Path, help="Output directory for summaries, CSVs, PNG, manifest.")
    parser.add_argument("--lang", choices=("en", "es"), help="Primary output language ('en' or 'es').")
    parser.add_argument("--date-col", type=str, help="Date/period column name.")
    parser.add_argument("--filter-col", type=str, help="Panel entity filter column name.")
    parser.add_argument("--filter-value", type=str, help="Panel entity value to retain.")
    parser.add_argument("--frequency", choices=FREQUENCIES, help="Observation cadence.")
    parser.add_argument("--rate-convention", choices=RATE_CONVENTIONS, help="DSA interest/growth convention.")
    parser.add_argument("--exact-compounding", action="store_true", help="Use exact (1+r)/(1+g) compounding in DSA.")
    parser.add_argument("--rate-unit", choices=RATE_UNITS, help="Unit for g, r, pb columns ('pct' or 'fraction').")
    parser.add_argument("--debt-unit", choices=DEBT_UNITS, help="Unit for debt column ('fraction' or 'percent').")
    parser.add_argument("--spread-unit", choices=SPREAD_UNITS, help="Unit for spread column ('pct' or 'bps').")
    parser.add_argument("--growth-col", type=str, help="Real GDP growth column.")
    parser.add_argument("--interest-rate-col", type=str, help="Real interest rate column.")
    parser.add_argument("--primary-balance-col", type=str, help="Primary fiscal balance column.")
    parser.add_argument("--debt-col", type=str, help="Public debt-to-GDP column.")
    parser.add_argument("--income-col", type=str, help="Optional real GDP/income column for AR(1) endowment.")
    parser.add_argument("--income-is-log", action="store_true", help="Treat income-col as already log output.")
    parser.add_argument("--spread-col", type=str, help="Optional observed sovereign spread column.")
    parser.add_argument("--detrend", choices=DETREND_METHODS, help="Cyclical income detrending method.")
    parser.add_argument("--theta", type=float, help="Explicit default reentry probability per period.")
    parser.add_argument("--beta", type=float, help="Explicit household/sovereign discount factor.")
    parser.add_argument("--r-star", type=float, help="Explicit risk-free rate per period.")
    parser.add_argument("--penalty-type", choices=PENALTY_TYPES, help="Default output penalty specification.")
    parser.add_argument("--dsa-horizon", type=int, help="DSA simulation horizon in periods.")
    parser.add_argument("--dsa-threshold", type=float, help="Prudential debt-to-GDP threshold (fraction).")
    parser.add_argument("--cubic-fatigue", action="store_true", help="Include quadratic/cubic debt fatigue terms in Bohn reaction.")
    return parser


def _build_cli_config(args: argparse.Namespace) -> SovereignBriefConfig:
    base = load_config(args.config) if args.config is not None else SovereignBriefConfig()
    overrides = asdict(base)
    mapping = {
        "lang": args.lang,
        "date_col": args.date_col,
        "filter_col": args.filter_col,
        "filter_value": args.filter_value,
        "frequency": args.frequency,
        "rate_convention": args.rate_convention,
        "rate_unit": args.rate_unit,
        "debt_unit": args.debt_unit,
        "spread_unit": args.spread_unit,
        "growth_col": args.growth_col,
        "interest_rate_col": args.interest_rate_col,
        "primary_balance_col": args.primary_balance_col,
        "debt_col": args.debt_col,
        "income_col": args.income_col,
        "spread_col": args.spread_col,
        "detrend": args.detrend,
        "theta": args.theta,
        "beta": args.beta,
        "r_star": args.r_star,
        "default_penalty_type": args.penalty_type,
        "dsa_horizon": args.dsa_horizon,
        "dsa_threshold": args.dsa_threshold,
    }
    for k, v in mapping.items():
        if v is not None:
            overrides[k] = v
    if args.exact_compounding:
        overrides["exact_compounding"] = True
    if args.income_is_log:
        overrides["income_is_log"] = True
    if args.cubic_fatigue:
        overrides["cubic_fatigue"] = True
    if args.demo and args.income_col is None and base.income_col is None:
        overrides["income_col"] = "real_gdp"
        overrides["spread_col"] = "spread_pct"
        overrides["dataset_label"] = "Bundled reproducible sovereign demo series"
    return SovereignBriefConfig(**overrides)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the own-data sovereign brief."""
    parser = _build_parser()
    arg_list = list(argv) if argv is not None else None
    args = parser.parse_args(arg_list)
    if arg_list is not None and not arg_list:
        parser.print_help()
        return 0
    if args.output is None or (args.data is None and not args.demo):
        parser.print_help()
        return 0
    if args.data is not None and args.demo:
        raise ValueError("Pass either --data CSV or --demo, not both.")

    config = _build_cli_config(args)
    src: pd.DataFrame | Path = (
        demo_sovereign_dataframe(frequency=config.frequency) if args.demo else args.data
    )
    res, paths = run_brief(src, config, args.output)
    print(f"Wrote {len(paths)} artifacts for n={res.data.n_obs} ({res.config.frequency}) observations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
