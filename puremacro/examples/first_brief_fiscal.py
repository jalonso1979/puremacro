"""Fiscal policy brief: the US government spending multiplier at two and four years, with and without slack, on Ramey and Zubairy's historical data.

How large is the US government spending multiplier, and is it larger when
the economy has slack? The brief answers with the one-step local
projections of Ramey and Zubairy (2018) on the bundled snapshot of their
quarterly US data, 1890Q1-2015Q4 (``puremacro/replication/data/rz2018.csv``,
read by :func:`puremacro.examples.ramey_zubairy_2018_multipliers.load_rz_data`).
The multiplier is cumulative: dollars of GDP per dollar of government
purchases, both summed over the quarters t..t+h. The headline horizons are
h = 7 and h = 15, Ramey and Zubairy's 2- and 4-year integrals. The paths
run over h = 0..20, with 90% bands and HAC variances on h + 1 lags.

Run it with::

    python -m puremacro.examples.first_brief_fiscal [--exclude 1941Q3:1945Q4] [--output DIR] [--lang en|es]

The command prints the headline table and its reading in English or
Spanish, and writes into DIR (default ``fiscal_brief``) the tables
``headline.md``, ``headline.csv`` and ``headline.tex``, the paths
``paths.csv``, the figure ``multipliers.png``, the record ``headline.json``
and ``manifest.json`` (data hash, specification, warnings, timings and
environment). ``--exclude`` adds the experiment: the same estimates without
a period, here World War II. :func:`run_brief` does the same from Python.

Each sample gets four estimator calls (:func:`estimate`):

- ``linear``: :func:`puremacro.lp.iv.lp_iv` with ``cumulative=True``. The
  window sum of ``y`` is regressed on the window sum of ``g``, instrumented
  by the military news shock ``newsy``, with four lags of ``y``, ``g`` and
  ``newsy`` as controls (``RZ_CONTROLS``). It reports the HAC effective F
  and the HAC Anderson-Rubin set.
- ``state``: :func:`puremacro.lp.state_dep.lp_state_dep_iv` with
  ``cumulative=True``, on the slack indicator ``slack_l`` =
  1{unemployment_{t-1} >= 6.5}, with every coefficient regime-specific
  (``interact="all"``).
- ``ar_slack`` and ``ar_no_slack``: ``lp_iv`` with ``newsy`` set to NaN
  outside the regime, for each regime's Anderson-Rubin set. Its estimate is
  the regime's, but its HAC sums treat the regime's non-adjacent quarters
  as neighbours.

A cell is weak when its effective F (``mop_f`` in the linear model,
``first_stage_f_H`` / ``first_stage_f_L`` by regime) is below ``lp_iv``'s
``mop_cv_10``, the Montiel Olea-Pflueger cutoff (23.1 for one instrument).
A weak cell is quoted by its weak-instrument-robust Anderson-Rubin set,
never by its point estimate. The test of equal multipliers is reliable only
where neither regime is weak.

An excluded period follows Ramey and Zubairy's rule for World War II
(2018, p. 867), in :func:`exclude_period`. ``y``, ``g`` and ``newsy``
become NaN in the period and every row is kept. The lag controls are
rebuilt from the masked series, and the window sums carry the NaN, so a
quarter drops out when its windows, its shock or one of its lags touch the
period.

The concentration diagnostic (:func:`concentration`) regresses ``newsy`` on
a constant and ``RZ_CONTROLS`` over a regime's estimation rows. It reports
the largest single quarter's share of the squared residuals, and the date of
that quarter. A share of at least ``CONCENTRATION_REPORT_SHARE`` means that one
episode carries much of the variation that identifies the regime's
multiplier.

:func:`compute_brief` returns the record behind every table, JSON-ready,
and the paths over h = 0..20. :func:`page_block` and :func:`summary_text`
word the record in English or Spanish, and :func:`write_outputs` writes the
files.

Español
-------
Informe de política fiscal: el multiplicador del gasto público de EE.UU. a
dos y cuatro años, con y sin holgura, con los datos históricos de Ramey y
Zubairy.

¿Qué tamaño tiene el multiplicador del gasto público de EE.UU. y es mayor
cuando la economía tiene holgura? El informe responde con las proyecciones
locales de un paso de Ramey y Zubairy (2018) sobre el snapshot incluido de
sus datos trimestrales de EE.UU., 1890Q1-2015Q4
(``puremacro/replication/data/rz2018.csv``, leído por
:func:`puremacro.examples.ramey_zubairy_2018_multipliers.load_rz_data`). El
multiplicador es acumulado: dólares de PIB por dólar de compras del
gobierno, ambos sumados en los trimestres t..t+h. Los horizontes
principales son h = 7 y h = 15, las integrales a 2 y 4 años de Ramey y
Zubairy. Las trayectorias cubren h = 0..20, con bandas al 90% y varianzas
HAC con h + 1 rezagos.

Ejecútelo con::

    python -m puremacro.examples.first_brief_fiscal [--exclude 1941Q3:1945Q4] [--output DIR] [--lang en|es]

El comando imprime la tabla principal y su lectura en inglés o en español,
y escribe en DIR (por defecto ``fiscal_brief``) las tablas
``headline.md``, ``headline.csv`` y ``headline.tex``, las trayectorias
``paths.csv``, la figura ``multipliers.png``, el registro ``headline.json``
y ``manifest.json`` (hash de los datos, especificación, advertencias,
tiempos y entorno). ``--exclude`` añade el experimento: las mismas
estimaciones sin un periodo, aquí la Segunda Guerra Mundial.
:func:`run_brief` hace lo mismo desde Python.

Cada muestra recibe cuatro llamadas a los estimadores (:func:`estimate`):

- ``linear``: :func:`puremacro.lp.iv.lp_iv` con ``cumulative=True``. La
  suma de ventana de ``y`` se regresa sobre la suma de ventana de ``g``,
  instrumentada con el choque de noticias militares ``newsy``, con cuatro
  rezagos de ``y``, ``g`` y ``newsy`` como controles (``RZ_CONTROLS``).
  Informa la F efectiva HAC y el conjunto de Anderson-Rubin HAC.
- ``state``: :func:`puremacro.lp.state_dep.lp_state_dep_iv` con
  ``cumulative=True``, sobre el indicador de holgura ``slack_l`` =
  1{desempleo_{t-1} >= 6.5}, con todos los coeficientes propios de cada
  régimen (``interact="all"``).
- ``ar_slack`` y ``ar_no_slack``: ``lp_iv`` con ``newsy`` en NaN fuera del
  régimen, para el conjunto de Anderson-Rubin de cada régimen. Su
  estimación es la del régimen, pero sus sumas HAC tratan como vecinos los
  trimestres no contiguos del régimen.

Una celda es débil cuando su F efectiva (``mop_f`` en el modelo lineal,
``first_stage_f_H`` / ``first_stage_f_L`` por régimen) queda por debajo del
``mop_cv_10`` de ``lp_iv``, el umbral de Montiel Olea-Pflueger (23.1 con un
instrumento). Una celda débil se cita por su conjunto de Anderson-Rubin,
robusto a instrumentos débiles, y nunca por su estimación puntual. La
prueba de multiplicadores iguales solo es fiable donde ningún régimen es
débil.

Un periodo excluido sigue la regla de Ramey y Zubairy para la Segunda
Guerra Mundial (2018, p. 867), en :func:`exclude_period`. ``y``, ``g`` y
``newsy`` pasan a NaN en el periodo y se conservan todas las filas. Los
controles rezagados se reconstruyen a partir de las series enmascaradas, y
las sumas de ventana arrastran el NaN, así que un trimestre sale de la
muestra cuando sus ventanas, su choque o alguno de sus rezagos tocan el
periodo.

El diagnóstico de concentración (:func:`concentration`) regresa ``newsy``
sobre una constante y ``RZ_CONTROLS`` en las filas de estimación de un
régimen. Informa la mayor participación de un solo trimestre en los
residuos al cuadrado y la fecha de ese trimestre. Una participación de al menos
``CONCENTRATION_REPORT_SHARE`` indica que un solo episodio concentra buena
parte de la variación que identifica el multiplicador del régimen.

:func:`compute_brief` devuelve el registro detrás de cada tabla, listo para
JSON, y las trayectorias en h = 0..20. :func:`page_block` y
:func:`summary_text` redactan el registro en inglés o en español, y
:func:`write_outputs` escribe los archivos.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
import re
import sys
import textwrap
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

import puremacro

from ..lp import LPResult
from ..lp._common import check_regular_index, cumulative_windows
from ..lp.iv import lp_iv
from ..lp.state_dep import lp_state_dep_iv
from ..reports import latex_escape
from ._brief_record import environment, sha256, write_json_atomic
from .ramey_zubairy_2018_multipliers import RZ_CONTROLS, load_rz_data

__all__ = [
    "HORIZONS", "HEADLINE_HORIZONS", "ALPHA", "WWII", "DATA_FILE", "CITATION",
    "CONCENTRATION_REPORT_SHARE", "SHORT_SAMPLE_OBS", "SHORT_REGIME_OBS",
    "DEFAULT_OUTPUT", "FiscalBriefConfig", "parse_quarterly_index", "prepare_fiscal_data",
    "parse_exclude", "load_brief_data", "exclude_period", "estimate", "concentration",
    "compute_brief", "page_block", "summary_text", "write_outputs", "run_brief", "main",
]

#: Horizons of the paths: the windows t..t+h for h = 0..20.
HORIZONS: tuple[int, ...] = tuple(range(21))
#: Ramey and Zubairy's 2- and 4-year integrals, the 8- and 16-quarter windows.
HEADLINE_HORIZONS: tuple[int, int] = (7, 15)
#: Two-sided level of every band and Anderson-Rubin set (90%).
ALPHA = 0.10
#: Ramey and Zubairy's World War II exclusion, the brief's experiment.
WWII = "1941Q3:1945Q4"
#: The bundled data snapshot, relative to the directory that holds the package.
DATA_FILE = "puremacro/replication/data/rz2018.csv"
#: The paper behind the data and the specification, in a form that reads the same in EN and ES.
CITATION = (
    "Ramey, V. A. & Zubairy, S. (2018). Government Spending Multipliers in Good Times "
    "and in Bad: Evidence from US Historical Data. Journal of Political Economy, "
    "126(2), 850-901. https://doi.org/10.1086/696277"
)
#: A regime's top quarter is reported when it carries at least this share of
#: the regime's residual instrument variation.
CONCENTRATION_REPORT_SHARE = 0.25
#: Effective linear estimation rows below which the brief emits a short-sample caution.
SHORT_SAMPLE_OBS = 120
#: Effective regime estimation rows below which the brief emits a short-regime caution.
SHORT_REGIME_OBS = 40

_MASKED = ("y", "g", "newsy")       # outcome, policy variable and instrument
_RUNNING_SUMS = ("cumy", "cumg")    # summed before any masking
#: Regime -> (suffix of its ``lp_state_dep_iv`` columns, its value of ``slack_l``).
_REGIMES: dict[str, tuple[str, float]] = {"slack": ("H", 1.0), "no_slack": ("L", 0.0)}
_EXCLUDE = re.compile(r"\s*([0-9]{4})[QT]([1-4])\s*:\s*([0-9]{4})[QT]([1-4])\s*", re.IGNORECASE)
_QUARTER_TOKEN = re.compile(r"^\s*([0-9]{4})\s*[-/\s]?\s*[QT]\s*([1-4])\s*$", re.IGNORECASE)
_SHOCK_MODES = ("external_instrument", "blanchard_perotti")
_SCALING_MODES = ("pre_scaled", "potential_col", "hp_trend", "poly_trend")
_STATE_OPS = (">=", ">", "<=", "<", "==")


@dataclass(frozen=True)
class FiscalBriefConfig:
    """Explicit configuration for running the fiscal multiplier brief on user data.

    Causal identification is never inferred from column names: callers must either supply
    an external shock/instrument column (``shock_mode="external_instrument"``) or opt in
    explicitly to ``shock_mode="blanchard_perotti"`` (contemporaneous fiscal policy
    predetermined within the quarter conditional on lags, with no fiscal foresight).
    """

    outcome: str = "y"
    policy: str = "g"
    instrument: str | None = None
    shock_mode: Literal["external_instrument", "blanchard_perotti"] = "external_instrument"
    date_col: str | None = None
    filter_col: str | None = None
    filter_value: str | None = None
    deflator_col: str | None = None
    potential_col: str | None = None
    scaling: Literal["pre_scaled", "potential_col", "hp_trend", "poly_trend"] = "pre_scaled"
    poly_degree: int = 2
    hp_lambda: float = 1600.0
    state: str | None = None
    state_threshold: float = 0.5
    state_op: Literal[">=", ">", "<=", "<", "=="] = ">="
    state_lag: int = 0
    slack_rule: str | None = None
    n_lags: int = 4
    controls: tuple[str, ...] | None = None
    horizons: tuple[int, ...] = HORIZONS
    headline_horizons: tuple[int, int] = HEADLINE_HORIZONS
    alpha: float = ALPHA
    title_en: str | None = None
    title_es: str | None = None
    units_en: str | None = None
    units_es: str | None = None
    citation: str = "User-supplied fiscal dataset"
    shock_source: str | None = None
    identifying_assumption_en: str | None = None
    identifying_assumption_es: str | None = None
    dataset_label: str | None = None

    def __post_init__(self) -> None:
        if self.shock_mode not in _SHOCK_MODES:
            raise ValueError(
                f"shock_mode must be one of {_SHOCK_MODES}, got {self.shock_mode!r}"
            )
        if self.shock_mode == "external_instrument" and not self.instrument:
            raise ValueError(
                "shock_mode='external_instrument' requires an explicit instrument column "
                "(e.g. instrument='narrative_shock'), or opt in explicitly to "
                "shock_mode='blanchard_perotti'."
            )
        if self.shock_mode == "blanchard_perotti" and self.instrument is not None:
            raise ValueError(
                "shock_mode='blanchard_perotti' cannot be combined with an external instrument "
                f"({self.instrument!r}); set instrument=None or shock_mode='external_instrument'."
            )
        if self.scaling not in _SCALING_MODES:
            raise ValueError(
                f"scaling must be one of {_SCALING_MODES}, got {self.scaling!r}"
            )
        if self.scaling == "potential_col" and not self.potential_col:
            raise ValueError(
                "scaling='potential_col' requires potential_col to name a potential-output column."
            )
        if self.state_op not in _STATE_OPS:
            raise ValueError(f"state_op must be one of {_STATE_OPS}, got {self.state_op!r}")
        if self.state_lag < 0:
            raise ValueError(f"state_lag must be >= 0, got {self.state_lag}")
        if self.n_lags < 0:
            raise ValueError(f"n_lags must be >= 0, got {self.n_lags}")
        if self.poly_degree < 1:
            raise ValueError(f"poly_degree must be >= 1, got {self.poly_degree}")
        if self.hp_lambda <= 0:
            raise ValueError(f"hp_lambda must be > 0, got {self.hp_lambda}")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must lie in (0, 1), got {self.alpha}")
        object.__setattr__(self, "horizons", tuple(int(h) for h in self.horizons))
        object.__setattr__(
            self, "headline_horizons", tuple(int(h) for h in self.headline_horizons)  # type: ignore[arg-type]
        )
        if len(self.headline_horizons) != 2:
            raise ValueError(
                f"headline_horizons must have exactly two horizons, got {self.headline_horizons}"
            )
        if len(self.horizons) == 0 or any(h < 0 for h in self.horizons):
            raise ValueError(f"horizons must be non-empty non-negative integers, got {self.horizons}")
        missing_h = [h for h in self.headline_horizons if h not in self.horizons]
        if missing_h:
            raise ValueError(
                f"headline_horizons {missing_h} must be included in horizons {self.horizons}"
            )
        if self.controls is not None:
            object.__setattr__(self, "controls", tuple(str(c) for c in self.controls))
#: Why an excluded period is rejected, in each language; ``example`` is ``WWII``.
_MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "format": "exclude must be START:END in quarters, for example {example}; got {text!r}",
        "reversed": "exclude starts after it ends ({start} > {end}); write START:END, "
                    "for example {example}",
        "no_overlap": "the period {start}:{end} does not overlap the sample, {first} to {last}",
    },
    "es": {
        "format": "exclude debe ser INICIO:FIN en trimestres, por ejemplo {example}; "
                  "se recibió {text!r}",
        "reversed": "exclude empieza después de terminar ({start} > {end}); escriba "
                    "INICIO:FIN, por ejemplo {example}",
        "no_overlap": "el periodo {start}:{end} no se solapa con la muestra, de {first} a {last}",
    },
}

#: Warnings issued while computing the last brief, as ``"<sample>: <Category>:
#: <message>"``, for the run's manifest. Every :func:`compute_brief` call
#: replaces them.
_LAST_WARNINGS: list[str] = []
#: Wall seconds of the last :func:`compute_brief` call (``compute_seconds``),
#: for the run's manifest; replaced at every call.
_LAST_TIMINGS: dict[str, float] = {}


class _ExcludeError(ValueError):
    """A rejected excluded period: ``str()`` is the English message, :meth:`text` any language."""

    def __init__(self, kind: str, **fields: Any) -> None:
        self.kind = kind
        self.fields = {"example": WWII, **fields}
        super().__init__(self.text("en"))

    def text(self, lang: str) -> str:
        """The message in ``lang`` (``"en"`` or ``"es"``)."""
        return _MESSAGES[lang][self.kind].format(**self.fields)


def _data_path() -> Path:
    """The file behind ``DATA_FILE``, located from the package directory."""
    return Path(__file__).resolve().parents[2] / DATA_FILE


def _quarters(index: pd.Index) -> pd.PeriodIndex:
    """Quarterly periods of a date or period index."""
    if isinstance(index, pd.PeriodIndex):
        return index.asfreq("Q")
    if isinstance(index, pd.DatetimeIndex):
        return index.to_period("Q")
    raise TypeError(f"the brief needs a DatetimeIndex or PeriodIndex, got {type(index).__name__}")


def parse_quarterly_index(
    values: pd.Index | pd.Series | Sequence[Any],
    *,
    name: str = "first_brief_fiscal",
) -> pd.DatetimeIndex:
    """Parse quarterly dates (``YYYYQn``, ``YYYYTn``, ISO dates, or Period/DatetimeIndex).

    Accepts English (``1999Q1``, ``1999-Q1``), Spanish/French/Portuguese *trimestre*
    (``1999T1``, ``1999-T1``), calendar ISO dates (``1999-01-01``), ``PeriodIndex`` (must be
    quarterly), or ``DatetimeIndex``. Enforces chronological order, unique quarters, and no
    missing quarters via :func:`check_regular_index`.

    Returns a regular quarterly-start ``DatetimeIndex``.
    """
    if isinstance(values, pd.PeriodIndex):
        freq_str = str(values.freqstr or "").upper()
        if not freq_str.startswith("Q"):
            raise ValueError(
                f"{name}: expected a quarterly PeriodIndex, got frequency {values.freqstr!r}"
            )
        periods = values.asfreq("Q")
    elif isinstance(values, pd.DatetimeIndex):
        if values.isna().any():
            raise ValueError(f"{name}: date index contains missing (NaT) timestamps")
        periods = values.to_period("Q")
    else:
        seq = list(values)
        if len(seq) == 0:
            raise ValueError(f"{name}: cannot parse an empty date sequence")
        parsed_periods: list[pd.Period] = []
        for raw in seq:
            if pd.isna(raw):
                raise ValueError(f"{name}: date column contains missing values")
            if isinstance(raw, pd.Period):
                if not str(raw.freqstr or "").upper().startswith("Q"):
                    raise ValueError(f"{name}: non-quarterly Period {raw!r}")
                parsed_periods.append(raw.asfreq("Q"))
                continue
            if isinstance(raw, pd.Timestamp):
                parsed_periods.append(raw.to_period("Q"))
                continue
            token = str(raw).strip()
            m = _QUARTER_TOKEN.match(token)
            if m is not None:
                parsed_periods.append(pd.Period(f"{m.group(1)}Q{m.group(2)}", freq="Q"))
            else:
                try:
                    ts = pd.Timestamp(token)
                except Exception as exc:
                    raise ValueError(
                        f"{name}: could not parse quarterly date {raw!r}; expected YYYYQn, "
                        "YYYYTn, or ISO YYYY-MM-DD"
                    ) from exc
                if pd.isna(ts):
                    raise ValueError(f"{name}: invalid timestamp {raw!r}")
                parsed_periods.append(ts.to_period("Q"))
        periods = pd.PeriodIndex(parsed_periods, freq="Q")

    if periods.has_duplicates:
        dups = periods[periods.duplicated()].unique().astype(str).tolist()[:5]
        raise ValueError(
            f"{name}: duplicate quarterly dates {dups}; if using a multi-country panel, "
            "pass filter_col/filter_value (e.g. filter_col='code', filter_value='DEU')."
        )
    dt_index = periods.to_timestamp(how="start")
    check_regular_index(dt_index, name)
    return dt_index


def _hp_trend_positive(series: pd.Series, lam: float) -> pd.Series:
    """Two-sided Hodrick-Prescott trend exp(HP(log(series))) on strictly positive segments."""
    out = pd.Series(np.nan, index=series.index, dtype=float)
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(vals)
    if not valid.any():
        raise ValueError("scaling='hp_trend' requires non-missing outcome observations")
    if np.any(vals[valid] <= 0.0):
        raise ValueError(
            "scaling='hp_trend' requires strictly positive real outcome levels (log trend); "
            "use scaling='pre_scaled' or scaling='potential_col' instead."
        )
    idx = np.flatnonzero(valid)
    if idx.size < 4:
        raise ValueError(
            f"scaling='hp_trend' needs at least 4 valid observations, got {idx.size}"
        )
    # Require contiguous positive span or solve on valid interior
    log_y = np.log(vals[valid])
    n = log_y.size
    d2 = np.zeros((n - 2, n), dtype=float)
    ar = np.arange(n - 2)
    d2[ar, ar] = 1.0
    d2[ar, ar + 1] = -2.0
    d2[ar, ar + 2] = 1.0
    penalty = np.eye(n, dtype=float) + float(lam) * (d2.T @ d2)
    trend = np.linalg.solve(penalty, log_y)
    out.iloc[idx] = np.exp(trend)
    return out


def _poly_trend_positive(series: pd.Series, degree: int) -> pd.Series:
    """Log-polynomial trend exp(P_d(t)) on strictly positive observations."""
    out = pd.Series(np.nan, index=series.index, dtype=float)
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(vals)
    if not valid.any():
        raise ValueError("scaling='poly_trend' requires non-missing outcome observations")
    if np.any(vals[valid] <= 0.0):
        raise ValueError(
            "scaling='poly_trend' requires strictly positive real outcome levels (log trend); "
            "use scaling='pre_scaled' or scaling='potential_col' instead."
        )
    idx = np.flatnonzero(valid)
    if idx.size <= degree + 1:
        raise ValueError(
            f"scaling='poly_trend' with degree={degree} needs more than {degree + 1} valid rows, "
            f"got {idx.size}"
        )
    if degree >= 4 and idx.size < 200:
        warnings.warn(
            f"poly_trend degree={degree} on a {idx.size}-quarter sample risks severe endpoint "
            "overfitting (Ramey & Zubairy's degree-6 trend spans 504 quarters); prefer "
            "scaling='hp_trend' or poly_degree<=2 on postwar/short samples.",
            UserWarning,
            stacklevel=2,
        )
    t_norm = np.linspace(-1.0, 1.0, len(series), dtype=float)
    vander = np.vander(t_norm[valid], N=degree + 1, increasing=True)
    coef = np.linalg.lstsq(vander, np.log(vals[valid]), rcond=None)[0]
    full_vander = np.vander(t_norm[valid], N=degree + 1, increasing=True)
    out.iloc[idx] = np.exp(full_vander @ coef)
    return out


def _coerce_config(
    config: FiscalBriefConfig | Mapping[str, Any] | str | Path | None,
) -> tuple[FiscalBriefConfig, dict[str, Any]]:
    """Normalize ``config`` into ``(FiscalBriefConfig, extra_cli_fields)``.

    A JSON file or mapping may optionally carry ``"data"`` and ``"exclude"`` alongside
    :class:`FiscalBriefConfig` fields so a single JSON file can drive the CLI.
    """
    if config is None:
        return FiscalBriefConfig(instrument="newsy", state="slack_l"), {}
    if isinstance(config, FiscalBriefConfig):
        return config, {}
    raw: dict[str, Any]
    if isinstance(config, (str, Path)):
        cfg_path = Path(config)
        loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"config file {cfg_path.name!r} must contain a JSON object")
        raw = dict(loaded)
    elif isinstance(config, Mapping):
        raw = dict(config)
    else:
        raise TypeError(
            f"config must be FiscalBriefConfig, mapping, or JSON path, got {type(config).__name__}"
        )
    extras: dict[str, Any] = {}
    for extra_key in ("data", "exclude"):
        if extra_key in raw:
            extras[extra_key] = raw.pop(extra_key)
    valid_fields = {f.name for f in fields(FiscalBriefConfig)}
    unknown = sorted(set(raw) - valid_fields)
    if unknown:
        raise ValueError(
            f"unknown FiscalBriefConfig keys {unknown}; valid keys are {sorted(valid_fields)}"
        )
    return FiscalBriefConfig(**raw), extras


def _eval_state_indicator(series: pd.Series, op: str, threshold: float) -> pd.Series:
    """Binary 1.0/0.0 regime indicator preserving NaN where ``series`` is missing."""
    vals = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=vals.index, dtype=float)
    ok = vals.notna()
    v = vals.loc[ok]
    if op == ">=":
        cond = v >= threshold
    elif op == ">":
        cond = v > threshold
    elif op == "<=":
        cond = v <= threshold
    elif op == "<":
        cond = v < threshold
    elif op == "==":
        cond = v == threshold
    else:
        raise ValueError(f"unsupported state_op {op!r}")
    out.loc[ok] = cond.astype(float)
    return out


def prepare_fiscal_data(
    data: pd.DataFrame | str | Path,
    config: FiscalBriefConfig,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Validate and transform user fiscal data into the brief's estimation frame.

    Returns ``(frame, control_cols, data_meta)`` where ``frame`` is indexed by a regular
    quarterly ``DatetimeIndex`` and carries canonical columns ``"y"``, ``"g"``, ``"newsy"``,
    optional ``"slack_l"``, and lag controls ``control_cols``.
    """
    import hashlib

    if isinstance(data, (str, Path)):
        src_path = Path(data)
        raw_bytes = src_path.read_bytes()
        data_sha = hashlib.sha256(raw_bytes).hexdigest()
        file_label = config.dataset_label or src_path.name
        raw_df = pd.read_csv(io.BytesIO(raw_bytes))
    elif isinstance(data, pd.DataFrame):
        raw_df = data.copy()
        csv_bytes = raw_df.to_csv(index=True).encode("utf-8")
        data_sha = hashlib.sha256(csv_bytes).hexdigest()
        file_label = config.dataset_label or "dataframe"
    else:
        raise TypeError(
            f"data must be a pandas DataFrame or CSV file path, got {type(data).__name__}"
        )

    if config.filter_col is not None:
        if config.filter_col not in raw_df.columns:
            raise ValueError(
                f"filter_col {config.filter_col!r} not found in columns {list(raw_df.columns)}"
            )
        if config.filter_value is None:
            raise ValueError("filter_value must be provided when filter_col is set")
        mask = raw_df[config.filter_col].astype(str).str.strip() == str(config.filter_value).strip()
        raw_df = raw_df.loc[mask].copy()
        if raw_df.empty:
            raise ValueError(
                f"no rows matched {config.filter_col} == {config.filter_value!r}"
            )

    if config.date_col is not None:
        if config.date_col not in raw_df.columns:
            raise ValueError(
                f"date_col {config.date_col!r} not found in columns {list(raw_df.columns)}"
            )
        dt_index = parse_quarterly_index(raw_df[config.date_col], name="first_brief_fiscal")
    else:
        if isinstance(raw_df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            dt_index = parse_quarterly_index(raw_df.index, name="first_brief_fiscal")
        elif "date" in raw_df.columns:
            dt_index = parse_quarterly_index(raw_df["date"], name="first_brief_fiscal")
        else:
            raise ValueError(
                "data must have a quarterly DatetimeIndex/PeriodIndex or specify date_col"
            )

    work = raw_df.copy()
    work.index = dt_index

    required_cols = [config.outcome, config.policy]
    if config.shock_mode == "external_instrument":
        assert config.instrument is not None
        required_cols.append(config.instrument)
    if config.deflator_col is not None:
        required_cols.append(config.deflator_col)
    if config.potential_col is not None:
        required_cols.append(config.potential_col)
    if config.state is not None:
        required_cols.append(config.state)
    if config.controls is not None:
        required_cols.extend(config.controls)

    missing_cols = [c for c in required_cols if c not in work.columns]
    if missing_cols:
        raise ValueError(
            f"missing required column(s) {missing_cols}; available columns: {list(work.columns)}"
        )

    y_raw = pd.to_numeric(work[config.outcome], errors="coerce").astype(float)
    g_raw = pd.to_numeric(work[config.policy], errors="coerce").astype(float)

    if config.deflator_col is not None:
        defl = pd.to_numeric(work[config.deflator_col], errors="coerce").astype(float)
        if np.any(defl.dropna() <= 0.0):
            raise ValueError(f"deflator_col {config.deflator_col!r} must be strictly positive")
        y_real = y_raw / defl
        g_real = g_raw / defl
    else:
        y_real = y_raw
        g_real = g_raw

    if config.scaling == "pre_scaled":
        y_scaled = y_real
        g_scaled = g_real
    elif config.scaling == "potential_col":
        assert config.potential_col is not None
        pot = pd.to_numeric(work[config.potential_col], errors="coerce").astype(float)
        if np.any(pot.dropna() <= 0.0):
            raise ValueError(f"potential_col {config.potential_col!r} must be strictly positive")
        y_scaled = y_real / pot
        g_scaled = g_real / pot
    elif config.scaling == "hp_trend":
        pot = _hp_trend_positive(y_real, config.hp_lambda)
        y_scaled = y_real / pot
        g_scaled = g_real / pot
    elif config.scaling == "poly_trend":
        pot = _poly_trend_positive(y_real, config.poly_degree)
        y_scaled = y_real / pot
        g_scaled = g_real / pot
    else:
        raise ValueError(f"unsupported scaling {config.scaling!r}")

    out = pd.DataFrame(index=dt_index)
    out["y"] = y_scaled
    out["g"] = g_scaled
    if config.shock_mode == "external_instrument":
        assert config.instrument is not None
        out["newsy"] = pd.to_numeric(work[config.instrument], errors="coerce").astype(float)
    else:
        # Blanchard-Perotti (2002) in 1-step cumulative LP-IV (Ramey & Zubairy 2018, p. 859):
        # contemporaneous policy g_t instruments cumulative sum_{j=0}^h g_{t+j} conditional on
        # lags of y and g.
        out["newsy"] = g_scaled.copy()

    if config.state is not None:
        s_raw = pd.to_numeric(work[config.state], errors="coerce").astype(float)
        if config.state_lag > 0:
            s_raw = s_raw.shift(config.state_lag)
        out["slack_l"] = _eval_state_indicator(s_raw, config.state_op, config.state_threshold)

    control_cols: list[str] = []
    if config.controls is not None:
        for col in config.controls:
            cname = f"_ctrl_{col}"
            out[cname] = pd.to_numeric(work[col], errors="coerce").astype(float)
            control_cols.append(cname)
    else:
        lag_bases = ["y", "g"]
        if config.shock_mode == "external_instrument":
            lag_bases.append("newsy")
        for k in range(1, config.n_lags + 1):
            for base in lag_bases:
                lname = f"{base}_l{k}"
                out[lname] = out[base].shift(k)
                control_cols.append(lname)

    if config.state is not None:
        valid_base = out[["y", "g", "newsy", *control_cols]].notna().all(axis=1)
        n_hi = int((valid_base & (out["slack_l"] == 1.0)).sum())
        n_lo = int((valid_base & (out["slack_l"] == 0.0)).sum())
        min_needed = len(control_cols) + 3
        if n_hi < min_needed or n_lo < min_needed:
            raise ValueError(
                f"state rule produced too few usable quarters by regime "
                f"(slack={n_hi}, no_slack={n_lo}; need at least {min_needed} in each). "
                "Adjust state_threshold/state_op or use state=None for linear-only estimation."
            )

    meta: dict[str, Any] = {
        "file": file_label,
        "sha256": data_sha,
        "citation": config.citation,
    }
    if config.filter_col is not None:
        meta["filter"] = {"column": config.filter_col, "value": str(config.filter_value)}
    return out, control_cols, meta


def parse_exclude(text: str) -> tuple[pd.Period, pd.Period]:
    """``"1941Q3:1945Q4"`` -> the first and last excluded quarters, both inclusive.

    Raises ``ValueError`` unless ``text`` is ``START:END`` in ``YYYYQn`` (or ``YYYYTn``)
    quarters with START no later than END.
    """
    match = _EXCLUDE.fullmatch(str(text))
    if match is None:
        raise _ExcludeError("format", text=text)
    try:
        start = pd.Period(f"{match[1]}Q{match[2]}", freq="Q")
        end = pd.Period(f"{match[3]}Q{match[4]}", freq="Q")
    except ValueError:                       # a year pandas cannot place, such as 0000
        raise _ExcludeError("format", text=text) from None
    if start > end:
        raise _ExcludeError("reversed", start=start, end=end)
    return start, end


def load_brief_data() -> pd.DataFrame:
    """The bundled Ramey-Zubairy frame from :func:`load_rz_data`, with every row kept.

    Rows are never dropped. The estimators skip the rows with a missing
    state or instrument themselves, and a dropped row would leave a date gap,
    which ``cumulative=True`` rejects; the index is checked here as a guard.
    """
    df = load_rz_data()
    check_regular_index(df.index, "first_brief_fiscal")
    return df


def exclude_period(
    df: pd.DataFrame,
    period: tuple[pd.Period, pd.Period],
    *,
    controls: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Ramey and Zubairy's exclusion of ``period``, both ends inclusive, on a copy of ``df``.

    The copy keeps every row. ``y``, ``g`` and ``newsy`` are NaN in the
    period, and the lag controls ``RZ_CONTROLS`` (or ``controls``) are rebuilt
    from the masked series. The running sums ``cumy`` and ``cumg`` were summed
    before the masking, so they are dropped. ``cumulative=True`` then forms the
    window sums from the masked series: a quarter drops out of horizon h when
    ``y_{t-1}``, any of ``y_t..y_{t+h}`` or ``g_t..g_{t+h}``, ``newsy_t`` or
    a lag control falls in the period.

    Raises ``ValueError`` when no quarter of ``df`` falls in the period.
    """
    start, end = period
    quarters = _quarters(df.index)
    inside = np.asarray((quarters >= start) & (quarters <= end))
    if not inside.any():
        raise _ExcludeError("no_overlap", start=start, end=end, first=quarters[0],
                            last=quarters[-1])
    ctrl_list = list(RZ_CONTROLS) if controls is None else list(controls)
    out = df.copy().drop(columns=list(_RUNNING_SUMS), errors="ignore")
    out.loc[inside, list(_MASKED)] = np.nan
    for name in ctrl_list:                   # "{series}_l{k}" is series.shift(k)
        if "_l" in name and not name.startswith("_ctrl_"):
            series, lag = name.rsplit("_l", 1)
            if series in out.columns and lag.isdigit():
                out[name] = out[series].shift(int(lag))
        elif name in out.columns:
            out.loc[inside, name] = np.nan
    return out


def estimate(
    df: pd.DataFrame,
    *,
    controls: Sequence[str] | None = None,
    horizons: Sequence[int] = HORIZONS,
    alpha: float = ALPHA,
    has_state: bool = True,
) -> dict[str, LPResult]:
    """The brief's estimator calls on one sample, 90% bands by default.

    Returns ``linear`` (:func:`lp_iv` with the HAC effective F and
    Anderson-Rubin set) and, when ``has_state=True``, ``state``
    (:func:`lp_state_dep_iv` by regime of ``slack_l``) plus ``ar_slack`` /
    ``ar_no_slack`` (:func:`lp_iv` with ``newsy`` NaN outside the regime; a
    missing ``slack_l`` counts as outside). All calls are cumulative on the
    full frame.
    """
    ctrl_list = list(RZ_CONTROLS) if controls is None else list(controls)
    hs = tuple(int(h) for h in horizons)
    out: dict[str, LPResult] = {
        "linear": lp_iv(df, y="y", x="g", z="newsy", horizons=hs, n_lags=0,
                        controls=ctrl_list, alpha=alpha, cumulative=True,
                        anderson_rubin=True),
    }
    if has_state:
        out["state"] = lp_state_dep_iv(
            df, y="y", x="g", z="newsy", state="slack_l",
            horizons=hs, n_lags=0, transition="threshold",
            threshold=0.5, controls=ctrl_list, alpha=alpha,
            interact="all", cumulative=True,
        )
        for name, (_, value) in _REGIMES.items():
            masked = df.copy()
            masked.loc[masked["slack_l"] != value, "newsy"] = np.nan
            out[f"ar_{name}"] = lp_iv(
                masked, y="y", x="g", z="newsy", horizons=hs,
                n_lags=0, controls=ctrl_list, alpha=alpha,
                cumulative=True, anderson_rubin=True,
            )
    return out


def concentration(
    df: pd.DataFrame,
    regime: float | None,
    h: int,
    *,
    controls: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The quarter that carries the most of a regime's residual instrument variation at ``h``.

    The rows are horizon ``h``'s estimation rows with ``slack_l == regime`` (or all estimation
    rows when ``regime is None``): both window sums present, and ``newsy`` and every control
    column non-missing. ``newsy`` is regressed on a constant and controls by least squares on
    those rows. Returns ``{"share": max(e^2) / sum(e^2), "date": "YYYYQn"}``, where ``date`` is
    the quarter of the largest squared residual.

    Raises ``ValueError`` when the regime has no more rows than regressors.
    """
    ctrl_list = list(RZ_CONTROLS) if controls is None else list(controls)
    y_w, x_w = cumulative_windows(df["y"], df["g"], h)
    pieces = [y_w, x_w, df["newsy"]]
    if ctrl_list:
        pieces.append(df[ctrl_list])
    present = pd.concat(pieces, axis=1).notna().all(axis=1)
    if regime is None:
        rows = df.loc[present]
        label_desc = "linear"
    else:
        rows = df.loc[present & (df["slack_l"] == regime)]
        label_desc = f"slack_l == {regime:g}"
    cols = [np.ones(len(rows))]
    if ctrl_list:
        cols.append(rows[ctrl_list].to_numpy(dtype=float))
    design = np.column_stack(cols)
    if len(rows) <= design.shape[1]:
        raise ValueError(f"concentration: {label_desc} has {len(rows)} estimation rows "
                         f"at h = {h}, too few for {design.shape[1]} regressors")
    z = rows["newsy"].to_numpy(dtype=float)
    coef = np.linalg.lstsq(design, z, rcond=None)[0]
    e2 = (z - design @ coef) ** 2
    denom = float(e2.sum())
    if denom <= 0.0:
        return {"share": 0.0, "date": str(_quarters(rows.index)[0])}
    top = int(np.argmax(e2))
    return {"share": float(e2[top] / denom), "date": str(_quarters(rows.index)[top])}


def _cell(row: pd.Series, suffix: str, f_column: str, ar: pd.Series,
          cutoff: float) -> dict[str, Any]:
    """Estimate, SE, band and effective F from ``row``; the Anderson-Rubin set from ``ar``."""
    f_eff = float(row[f_column])
    return {
        "multiplier": float(row[f"beta{suffix}"]),
        "se": float(row[f"se{suffix}"]),
        "lo": float(row[f"lo{suffix}"]),
        "hi": float(row[f"hi{suffix}"]),
        "f_eff": f_eff,
        "ar_lo": float(ar["ar_lo"]),
        "ar_hi": float(ar["ar_hi"]),
        "ar_set_type": str(ar["ar_set_type"]),
        "weak": not (f_eff >= cutoff),        # a missing F is weak, never strong
    }


def _cells(fit: dict[str, LPResult], concentrations: dict[str, dict[str, Any]], h: int,
           cutoff: float) -> dict[str, Any]:
    """Every headline cell of one sample at horizon ``h``."""
    linear = fit["linear"].loc[h]
    cells: dict[str, Any] = {"linear": _cell(linear, "", "mop_f", linear, cutoff)}
    if "state" not in fit:
        cells["slack"] = None
        cells["no_slack"] = None
        cells["difference"] = None
        cells["concentration"] = concentrations
        return cells
    state = fit["state"].loc[h]
    for name, (label, _) in _REGIMES.items():
        cells[name] = {**_cell(state, f"_{label}", f"first_stage_f_{label}",
                               fit[f"ar_{name}"].loc[h], cutoff),
                       "n_eff": float(state[f"n_eff_{label}"])}
    cells["difference"] = {
        "estimate": float(state["diff_HL"]),
        "se": float(state["se_diff_HL"]),
        "p_value": float(state["p_diff_HL"]),
        "reliable": not (cells["slack"]["weak"] or cells["no_slack"]["weak"]),
    }
    cells["concentration"] = concentrations
    return cells


def _linear_n_obs_at_horizon(
    df: pd.DataFrame,
    fit: dict[str, LPResult],
    h: int,
    controls: Sequence[str] | None = None,
) -> int:
    """Effective linear estimation quarters at horizon ``h``."""
    spec_attr = fit["linear"].attrs.get("specification")
    if isinstance(spec_attr, dict):
        n_by_h = spec_attr.get("n_obs_by_horizon")
        if isinstance(n_by_h, dict) and h in n_by_h and n_by_h[h] is not None:
            return int(n_by_h[h])
    if "n_obs" in fit["linear"].columns and pd.notna(fit["linear"].loc[h, "n_obs"]):
        return int(fit["linear"].loc[h, "n_obs"])
    y_w, x_w = cumulative_windows(df["y"], df["g"], h)
    ctrl_list = list(RZ_CONTROLS) if controls is None else list(controls)
    pieces = [y_w, x_w, df["newsy"]]
    present_ctrls = [c for c in ctrl_list if c in df.columns]
    if present_ctrls:
        pieces.append(df[present_ctrls])
    return int(pd.concat(pieces, axis=1).notna().all(axis=1).sum())


def _sample_record(
    df: pd.DataFrame,
    fit: dict[str, LPResult],
    concentrations: dict[int, dict[str, dict[str, Any]]],
    cutoff: float,
    *,
    headline_horizons: Sequence[int] = HEADLINE_HORIZONS,
    controls: Sequence[str] | None = None,
    include_guards: bool = False,
) -> dict[str, Any]:
    """One sample's first and last quarters and its cells at each headline horizon."""
    quarters = _quarters(df.index)
    hs = [int(h) for h in headline_horizons]
    rec: dict[str, Any] = {
        "start": str(quarters[0]),
        "end": str(quarters[-1]),
        "horizons": {str(h): _cells(fit, concentrations[h], h, cutoff) for h in hs},
    }
    if include_guards:
        lin_n = [_linear_n_obs_at_horizon(df, fit, h, controls=controls) for h in hs]
        min_lin = min(lin_n) if lin_n else 0
        min_reg: int | None = None
        if "state" in fit:
            reg_ns: list[int] = []
            for h in hs:
                for _, (label, _) in _REGIMES.items():
                    val = fit["state"].loc[h, f"n_eff_{label}"]
                    if pd.notna(val):
                        reg_ns.append(int(val))
            min_reg = min(reg_ns) if reg_ns else 0
        short_sample = min_lin < SHORT_SAMPLE_OBS
        short_regime = min_reg is not None and min_reg < SHORT_REGIME_OBS
        rec["n_obs"] = int(len(df))
        rec["sample_guard"] = {
            "short_sample": short_sample,
            "short_regime": short_regime,
            "min_linear_n_obs": min_lin,
            "min_regime_n_eff": min_reg,
        }
    return rec


def _paths(
    sample: str,
    fit: dict[str, LPResult],
    *,
    horizons: Sequence[int] = HORIZONS,
) -> pd.DataFrame:
    """One row per horizon: the multipliers with their bands and F, and ``p_diff``."""
    hs = [int(h) for h in horizons]
    linear = fit["linear"].loc[hs]
    frame = pd.DataFrame({"sample": sample, "h": hs})
    for column, source in (("linear", "beta"), ("linear_lo", "lo"), ("linear_hi", "hi"),
                           ("linear_f", "mop_f")):
        frame[column] = linear[source].to_numpy(dtype=float)
    if "state" in fit:
        state = fit["state"].loc[hs]
        for name, (label, _) in _REGIMES.items():
            for column, source in ((name, "beta"), (f"{name}_lo", "lo"), (f"{name}_hi", "hi"),
                                   (f"{name}_f", "first_stage_f")):
                frame[column] = state[f"{source}_{label}"].to_numpy(dtype=float)
        frame["p_diff"] = state["p_diff_HL"].to_numpy(dtype=float)
    return frame


def _json_ready(value: Any) -> Any:
    """``value`` with lists for tuples, Python numbers for numpy ones and None for non-finite floats."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"the record has no JSON form for {type(value).__name__}")


def _resolve_custom_spec(
    cfg: FiscalBriefConfig,
    control_cols: Sequence[str],
    cutoff: float,
) -> dict[str, Any]:
    """Build the self-describing ``specification`` dictionary for an own-data run."""
    if cfg.shock_mode == "blanchard_perotti":
        default_source = f"Blanchard-Perotti recursive shock on '{cfg.policy}' conditional on lags"
        default_id_en = (
            "Blanchard-Perotti (2002) recursive timing assumption: fiscal policy does not "
            "respond within the same quarter to contemporaneous non-fiscal output shocks "
            "conditional on lags, and policy changes are unanticipated prior to implementation."
        )
        default_id_es = (
            "Supuesto recursivo de Blanchard y Perotti (2002): la política fiscal no responde "
            "dentro del mismo trimestre a choques contemporáneos no fiscales del producto "
            "condicional en los rezagos, y los cambios fiscales no son anticipados antes de "
            "su ejecución."
        )
        inst_name = cfg.policy
    else:
        assert cfg.instrument is not None
        default_source = f"External instrument column '{cfg.instrument}'"
        default_id_en = (
            f"External-instrument LP-IV assumption: '{cfg.instrument}' is relevant and "
            "contemporaneously/lead-lag exogenous to non-fiscal output innovations conditional "
            "on the lag controls."
        )
        default_id_es = (
            f"Supuesto LP-IV con instrumento externo: '{cfg.instrument}' es relevante y exógeno "
            "frente a innovaciones no fiscales del producto condicional en los controles "
            "rezagados."
        )
        inst_name = cfg.instrument

    if cfg.state is None:
        slack_rule = None
        interact = None
    elif cfg.slack_rule is not None:
        slack_rule = cfg.slack_rule
        interact = "all"
    else:
        lag_tag = f"{{t-{cfg.state_lag}}}" if cfg.state_lag > 0 else "t"
        slack_rule = f"{cfg.state}_{lag_tag} {cfg.state_op} {cfg.state_threshold:g}"
        interact = "all"

    spec: dict[str, Any] = {
        "custom": True,
        "outcome": cfg.outcome,
        "policy": cfg.policy,
        "instrument": inst_name,
        "shock_mode": cfg.shock_mode,
        "shock_source": cfg.shock_source or default_source,
        "identifying_assumption": {
            "en": cfg.identifying_assumption_en or default_id_en,
            "es": cfg.identifying_assumption_es or default_id_es,
        },
        "scaling": cfg.scaling,
        "deflator_col": cfg.deflator_col,
        "potential_col": cfg.potential_col,
        "state": cfg.state,
        "slack_rule": slack_rule,
        "n_lags": cfg.n_lags,
        "controls": list(control_cols),
        "horizons": list(cfg.horizons),
        "headline_horizons": list(cfg.headline_horizons),
        "alpha": cfg.alpha,
        "hac_lags": "h + 1",
        "interact": interact,
        "weak_cutoff": cutoff,
    }
    if cfg.title_en or cfg.title_es:
        spec["title"] = {"en": cfg.title_en, "es": cfg.title_es}
    if cfg.units_en or cfg.units_es:
        spec["units"] = {"en": cfg.units_en, "es": cfg.units_es}
    return spec


def compute_brief(
    *,
    exclude: str | None = None,
    data: pd.DataFrame | str | Path | None = None,
    config: FiscalBriefConfig | Mapping[str, Any] | str | Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Estimate the brief on the full sample and, with ``exclude``, on the experiment.

    Without ``data`` or ``config``, estimates Ramey and Zubairy's (2018) specification on the
    bundled historical US dataset. Passing ``data`` and ``config`` (a :class:`FiscalBriefConfig`,
    mapping, or JSON path) runs the same cumulative 1-step LP-IV diagnostics, weak-instrument
    Anderson-Rubin sets, concentration checks, and short-sample guards on user data.
    """
    _LAST_WARNINGS.clear()
    _LAST_TIMINGS.clear()
    started = time.perf_counter()

    is_custom = data is not None or config is not None
    if not is_custom:
        full = load_brief_data()
        frames = {"full": full}
        period = None
        if exclude is not None:
            period = parse_exclude(exclude)
            frames["experiment"] = exclude_period(full, period)

        fits: dict[str, dict[str, LPResult]] = {}
        concentrations: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
        for sample, frame in frames.items():
            caught: list[warnings.WarningMessage] = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fits[sample] = estimate(frame)
                    concentrations[sample] = {
                        h: {name: concentration(frame, value, h)
                            for name, (_, value) in _REGIMES.items()}
                        for h in HEADLINE_HORIZONS
                    }
            finally:   # also when the sample fails: its warnings may say why
                for w in caught:
                    _LAST_WARNINGS.append(f"{sample}: {w.category.__name__}: {w.message}")
                    warnings.warn_explicit(w.message, w.category, w.filename, w.lineno,
                                           source=w.source)

        cutoff = float(fits["full"]["linear"]["mop_cv_10"].iloc[0])
        record = {
            "puremacro_version": puremacro.__version__,
            "data": {"file": DATA_FILE, "sha256": sha256(_data_path()), "citation": CITATION},
            "specification": {
                "outcome": "y", "policy": "g", "instrument": "newsy", "state": "slack_l",
                "slack_rule": "unemp_{t-1} >= 6.5", "controls": list(RZ_CONTROLS),
                "horizons": list(HORIZONS), "headline_horizons": list(HEADLINE_HORIZONS),
                "alpha": ALPHA, "hac_lags": "h + 1", "interact": "all", "weak_cutoff": cutoff,
            },
            "full": _sample_record(full, fits["full"], concentrations["full"], cutoff),
            "experiment": None if period is None else {
                "exclude": [str(period[0]), str(period[1])],
                **_sample_record(frames["experiment"], fits["experiment"],
                                 concentrations["experiment"], cutoff),
            },
        }
        paths = pd.concat([_paths(sample, fits[sample]) for sample in frames], ignore_index=True)
        ready = _json_ready(record)
        _LAST_TIMINGS["compute_seconds"] = time.perf_counter() - started
        return ready, paths

    # Own-data / custom-configuration execution path
    cfg, extras = _coerce_config(config)
    resolved_data = data if data is not None else extras.get("data")
    if resolved_data is None:
        raise ValueError(
            "custom fiscal brief requires 'data' (DataFrame, CSV path, or 'data' key in config)"
        )
    resolved_exclude = exclude if exclude is not None else extras.get("exclude")

    prep_caught: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as prep_caught:
            warnings.simplefilter("always")
            full, control_cols, data_meta = prepare_fiscal_data(resolved_data, cfg)
    finally:
        for w in prep_caught:
            _LAST_WARNINGS.append(f"prepare: {w.category.__name__}: {w.message}")
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source)

    frames = {"full": full}
    period = None
    if resolved_exclude is not None:
        period = parse_exclude(resolved_exclude)
        frames["experiment"] = exclude_period(full, period, controls=control_cols)

    has_state = cfg.state is not None
    fits = {}
    concentrations = {}
    for sample, frame in frames.items():
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fits[sample] = estimate(
                    frame,
                    controls=control_cols,
                    horizons=cfg.horizons,
                    alpha=cfg.alpha,
                    has_state=has_state,
                )
                conc_sample: dict[int, dict[str, dict[str, Any]]] = {}
                for h in cfg.headline_horizons:
                    if has_state:
                        conc_h: dict[str, dict[str, Any]] = {}
                        for name, (_, value) in _REGIMES.items():
                            try:
                                conc_h[name] = concentration(
                                    frame, value, h, controls=control_cols
                                )
                            except ValueError:
                                conc_h[name] = {"share": None, "date": None}
                        conc_sample[h] = conc_h
                    else:
                        try:
                            conc_sample[h] = {
                                "linear": concentration(frame, None, h, controls=control_cols)
                            }
                        except ValueError:
                            conc_sample[h] = {"linear": {"share": None, "date": None}}
                concentrations[sample] = conc_sample

                # Emit short-sample / short-regime warnings inside the sample capture block
                lin_ns = [
                    _linear_n_obs_at_horizon(frame, fits[sample], h, controls=control_cols)
                    for h in cfg.headline_horizons
                ]
                min_lin = min(lin_ns) if lin_ns else 0
                if min_lin < SHORT_SAMPLE_OBS:
                    warnings.warn(
                        f"short sample at headline horizons (min n_obs={min_lin} < "
                        f"{SHORT_SAMPLE_OBS}); HAC standard errors and Montiel Olea-Pflueger "
                        "effective-F cutoffs rely on large-T asymptotics.",
                        UserWarning,
                        stacklevel=2,
                    )
                if has_state:
                    reg_ns = [
                        int(fits[sample]["state"].loc[h, f"n_eff_{label}"])
                        for h in cfg.headline_horizons
                        for _, (label, _) in _REGIMES.items()
                        if pd.notna(fits[sample]["state"].loc[h, f"n_eff_{label}"])
                    ]
                    min_reg = min(reg_ns) if reg_ns else 0
                    if min_reg < SHORT_REGIME_OBS:
                        warnings.warn(
                            f"short regime sample at headline horizons (min n_eff={min_reg} < "
                            f"{SHORT_REGIME_OBS}); state-dependent bands and regime comparisons "
                            "have low finite-sample reliability.",
                            UserWarning,
                            stacklevel=2,
                        )
        finally:
            for w in caught:
                _LAST_WARNINGS.append(f"{sample}: {w.category.__name__}: {w.message}")
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno,
                                       source=w.source)

    cutoff = float(fits["full"]["linear"]["mop_cv_10"].iloc[0])
    spec_dict = _resolve_custom_spec(cfg, control_cols, cutoff)
    record = {
        "puremacro_version": puremacro.__version__,
        "data": data_meta,
        "specification": spec_dict,
        "full": _sample_record(
            full, fits["full"], concentrations["full"], cutoff,
            headline_horizons=cfg.headline_horizons, controls=control_cols, include_guards=True,
        ),
        "experiment": None if period is None else {
            "exclude": [str(period[0]), str(period[1])],
            **_sample_record(
                frames["experiment"], fits["experiment"], concentrations["experiment"], cutoff,
                headline_horizons=cfg.headline_horizons, controls=control_cols, include_guards=True,
            ),
        },
    }
    paths = pd.concat(
        [_paths(sample, fits[sample], horizons=cfg.horizons) for sample in frames],
        ignore_index=True,
    )
    ready = _json_ready(record)
    _LAST_TIMINGS["compute_seconds"] = time.perf_counter() - started
    return ready, paths


# ---------------------------------------------------------------------------
# Outputs: readings, tables, figure, files and the command line
# ---------------------------------------------------------------------------

#: Where :func:`main` writes when ``--output`` is not given.
DEFAULT_OUTPUT = Path("fiscal_brief")

_ROWS = ("linear", "slack", "no_slack")
_MARK = "\N{DAGGER}"                 # marks a weak cell, shown by its Anderson-Rubin set
#: How a weak cell's Anderson-Rubin set is worded, by ``ar_set_type``: the reading's
#: ``set_<kind>`` sub-template.
_SET_KIND = {"bounded": "bounded", "unbounded_rays": "unbounded", "all_real": "unbounded",
             "empty": "empty"}

#: The reading's sentences, word for word; :func:`_reading` applies the rule that picks them.
#: Placeholders: ``R`` row, ``c1``/``c2`` the 2- and 4-year clauses, ``H`` horizon, ``m``
#: estimate, ``lo``/``hi`` band, ``F`` effective F, ``c`` cutoff, ``a``/``b`` Anderson-Rubin
#: bounds (the excluded period in ``exp_head``), ``d`` difference, ``p`` p-value or percent,
#: ``D`` quarter, ``f`` full-sample estimate, ``S`` a weak cell's set, worded by the filled
#: ``set_<kind>`` sub-template; a trailing 1 or 2 names the horizon. A weak clause is
#: ``weak`` with the set of its type: bounded, unbounded, empty or, when a bound it needs is
#: missing, unavailable.
_READING: dict[str, dict[str, Any]] = {
    "en": {
        "rows": ("Linear", "Slack", "No slack"),
        "horizons": ("2 years", "4 years"),
        "units": ("Multipliers are dollars of GDP per dollar of government purchases, both "
                  "cumulated over the horizon."),
        "row": "{R}: {c1}; {c2}.",
        "strong": "{m} at {H} (90% band {lo} to {hi})",
        "weak": "weak at {H} (effective F {F}, cutoff {c}), {S}",
        "set_bounded": "with a weak-instrument-robust 90% set of {a} to {b}",
        "set_unbounded": ("and its weak-instrument-robust 90% set is unbounded: the data cannot "
                          "bound it"),
        "set_empty": "and its weak-instrument-robust 90% set is empty",
        "set_unavailable": "and its weak-instrument-robust 90% set is not available",
        "diff": "With and without slack, the multipliers differ by {d} at {H} (p = {p}).",
        "diff_no_one": ("The comparison of the multipliers with and without slack is not reliable "
                        "at {H}, because a regime is weak there."),
        "diff_no_both": ("The comparison of the multipliers with and without slack is not "
                         "reliable at either horizon, because at least one regime is weak at "
                         "each."),
        "conc_both_same": ("{R}: one quarter, {D}, carries {p}% of the instrument's residual "
                           "variation at both horizons."),
        "conc_both": ("{R}: one quarter, {D}, carries {p1}% of the instrument's residual "
                      "variation at {H1} and {p2}% at {H2}."),
        "conc_two_dates": ("{R}: {D1} carries {p1}% of the instrument's residual variation at "
                           "{H1}, and {D2} carries {p2}% at {H2}."),
        "conc_one": ("{R}: one quarter, {D}, carries {p}% of the instrument's residual variation "
                     "at {H}."),
        "exp_head": ("Excluding {a}-{b}: every quarter whose window, shock or lags touch the "
                     "period is dropped, as Ramey and Zubairy do for World War II."),
        "cmp": "{R} at {H}: {m} here against {f} in the full sample.",
        "cmp_weak_full": "{R} at {H}: {m} here, while the full-sample estimate is weak, {S}.",
        "cmp_none": ("No headline estimate clears the cutoff in this sample, so none is compared "
                     "with the full sample."),
        "three_step": ("Ramey and Zubairy's Table 3 values without World War II are three-step "
                       "estimates (summed impulse responses); the brief's are one-step, so the "
                       "two need not match."),
    },
    "es": {
        "rows": ("Lineal", "Con holgura", "Sin holgura"),
        "horizons": ("2 años", "4 años"),
        "units": ("Los multiplicadores son dólares de PIB por dólar de compras del gobierno, "
                  "ambos acumulados en el horizonte."),
        "row": "{R}: {c1}; {c2}.",
        "strong": "{m} a {H} (banda al 90%: {lo} a {hi})",
        "weak": "débil a {H} (F efectiva {F}, umbral {c}), {S}",
        "set_bounded": "con un conjunto al 90% robusto a instrumentos débiles de {a} a {b}",
        "set_unbounded": ("y su conjunto al 90% robusto a instrumentos débiles no está acotado: "
                          "los datos no pueden acotarlo"),
        "set_empty": "y su conjunto al 90% robusto a instrumentos débiles está vacío",
        "set_unavailable": "y su conjunto al 90% robusto a instrumentos débiles no está disponible",
        "diff": "Con y sin holgura, los multiplicadores difieren en {d} a {H} (p = {p}).",
        "diff_no_one": ("La comparación de los multiplicadores con y sin holgura no es fiable a "
                        "{H}, porque un régimen es débil ahí."),
        "diff_no_both": ("La comparación de los multiplicadores con y sin holgura no es fiable en "
                         "ninguno de los dos horizontes, porque en cada uno al menos un régimen "
                         "es débil."),
        "conc_both_same": ("{R}: un solo trimestre, {D}, concentra el {p}% de la variación "
                           "residual del instrumento en ambos horizontes."),
        "conc_both": ("{R}: un solo trimestre, {D}, concentra el {p1}% de la variación residual "
                      "del instrumento a {H1} y el {p2}% a {H2}."),
        "conc_two_dates": ("{R}: {D1} concentra el {p1}% de la variación residual del "
                           "instrumento a {H1}, y {D2} el {p2}% a {H2}."),
        "conc_one": ("{R}: un solo trimestre, {D}, concentra el {p}% de la variación residual "
                     "del instrumento a {H}."),
        "exp_head": ("Excluyendo {a}-{b}: se descarta todo trimestre cuya ventana, choque o "
                     "rezagos tocan el periodo, como hacen Ramey y Zubairy con la Segunda Guerra "
                     "Mundial."),
        "cmp": "{R} a {H}: {m} aquí frente a {f} en la muestra completa.",
        "cmp_weak_full": ("{R} a {H}: {m} aquí, mientras que la estimación de la muestra completa "
                          "es débil, {S}."),
        "cmp_none": ("Ninguna estimación principal supera el umbral en esta muestra, así que "
                     "ninguna se compara con la muestra completa."),
        "three_step": ("Los valores sin la Segunda Guerra Mundial de la Tabla 3 de Ramey y "
                       "Zubairy son estimaciones en tres pasos (respuestas al impulso sumadas); "
                       "las del informe son de un paso, así que no tienen por qué coincidir."),
    },
}

#: Every other line, label and message of the outputs, in each language.
_TEXT: dict[str, dict[str, str]] = {
    "en": {
        "title": ("Fiscal policy brief: the US government spending multiplier at two and four "
                  "years, with and without slack"),
        "source": "Data and specification: {citation}",
        "full": "Full sample, {start}-{end}",
        "experiment": "Experiment: excluding {a}-{b}",
        "multiplier": "Multiplier",
        "horizon": "{H} (h = {h})",
        "f_eff": "Effective F",
        "difference": "Slack minus no slack",
        "not_reliable": "not reliable",
        "empty_set": "empty set",
        "table_note": ("Estimate [90% band]. The difference is reported only where neither "
                       "regime is weak."),
        "weak_note": ("effective F below {c}: the Anderson-Rubin set is shown instead of the "
                      "estimate."),
        "no_experiment": ("No experiment was run. To exclude World War II as Ramey and Zubairy "
                          "do, run `python -m puremacro.examples.first_brief_fiscal --exclude "
                          "{period}`."),
        "written": "Outputs written to {path}",
        "fig_title": "Cumulative government spending multipliers by horizon, with 90% bands",
        "fig_x": "Horizon h (quarters)",
        "fig_y": "Cumulative multiplier",
        "fig_strong": "estimate, effective F of at least {c}",
        "fig_weak": "weak: effective F below {c}",
        "fig_band": "90% band (conventional; robust sets in the table for weak cells)",
        "fig_off": "weak estimate beyond the axis",
    },
    "es": {
        "title": ("Informe de política fiscal: el multiplicador del gasto público de EE.UU. a dos "
                  "y cuatro años, con y sin holgura"),
        "source": "Datos y especificación: {citation}",
        "full": "Muestra completa, {start}-{end}",
        "experiment": "Experimento: excluyendo {a}-{b}",
        "multiplier": "Multiplicador",
        "horizon": "{H} (h = {h})",
        "f_eff": "F efectiva",
        "difference": "Con holgura menos sin holgura",
        "not_reliable": "no fiable",
        "empty_set": "conjunto vacío",
        "table_note": ("Estimación [banda al 90%]. La diferencia solo se informa donde ningún "
                       "régimen es débil."),
        "weak_note": ("F efectiva por debajo de {c}: se muestra el conjunto de Anderson-Rubin en "
                      "lugar de la estimación."),
        "no_experiment": ("No se ejecutó ningún experimento. Para excluir la Segunda Guerra "
                          "Mundial como hacen Ramey y Zubairy, ejecute `python -m "
                          "puremacro.examples.first_brief_fiscal --exclude {period}`."),
        "written": "Resultados escritos en {path}",
        "fig_title": ("Multiplicadores acumulados del gasto público por horizonte, con bandas "
                      "al 90%"),
        "fig_x": "Horizonte h (trimestres)",
        "fig_y": "Multiplicador acumulado",
        "fig_strong": "estimación, F efectiva de al menos {c}",
        "fig_weak": "débil: F efectiva por debajo de {c}",
        "fig_band": ("banda al 90% (convencional; conjuntos robustos en la tabla para las celdas "
                     "débiles)"),
        "fig_off": "estimación débil fuera del eje",
    },
}


def _text(lang: str) -> dict[str, str]:
    """The output strings in ``lang``; raises ``ValueError`` unless it is ``"en"`` or ``"es"``."""
    if lang not in _TEXT:
        raise ValueError(f"lang must be 'en' or 'es', got {lang!r}")
    return _TEXT[lang]


def _is_custom(record: dict[str, Any]) -> bool:
    return bool(record.get("specification", {}).get("custom", False))


def _has_state(record: dict[str, Any]) -> bool:
    return record.get("specification", {}).get("state") is not None


def _active_rows(record: dict[str, Any], lang: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    words = _READING[lang]
    if _has_state(record):
        return _ROWS, words["rows"]
    return ("linear",), (words["rows"][0],)


def _horizon_label(h: int, lang: str) -> str:
    quarters = int(h) + 1
    if quarters % 4 == 0:
        years = quarters // 4
        if lang == "en":
            return "1 year" if years == 1 else f"{years} years"
        return "1 año" if years == 1 else f"{years} años"
    if lang == "en":
        return "1 quarter" if quarters == 1 else f"{quarters} quarters"
    return "1 trimestre" if quarters == 1 else f"{quarters} trimestres"


def _title_text(record: dict[str, Any], lang: str) -> str:
    if not _is_custom(record):
        return _TEXT[lang]["title"]
    custom_title = (record["specification"].get("title") or {}).get(lang)
    if custom_title:
        return str(custom_title)
    if lang == "en":
        return "Fiscal policy brief: cumulative fiscal multipliers by horizon"
    return "Informe de política fiscal: multiplicadores fiscales acumulados por horizonte"


def _units_text(record: dict[str, Any], lang: str) -> str:
    if not _is_custom(record):
        return _READING[lang]["units"]
    custom_units = (record["specification"].get("units") or {}).get(lang)
    if custom_units:
        return str(custom_units)
    if lang == "en":
        return (
            "Multipliers are cumulative units of the outcome per cumulative unit of the fiscal "
            "policy variable over the horizon."
        )
    return (
        "Los multiplicadores son unidades acumuladas del producto por unidad acumulada de la "
        "variable de política fiscal en el horizonte."
    )


def _identification_lines(record: dict[str, Any], lang: str) -> list[str]:
    if not _is_custom(record):
        return []
    spec = record["specification"]
    shock_src = spec.get("shock_source") or spec.get("shock_mode")
    assumption = (spec.get("identifying_assumption") or {}).get(lang) or ""
    if lang == "en":
        return [
            f"Shock source: {shock_src}",
            f"Identifying assumption: {assumption}",
        ]
    return [
        f"Fuente del choque: {shock_src}",
        f"Supuesto de identificación: {assumption}",
    ]


def _sample_caution(sample: dict[str, Any], lang: str) -> list[str]:
    guard = sample.get("sample_guard")
    if not isinstance(guard, dict):
        return []
    out: list[str] = []
    if guard.get("short_sample"):
        n_lin = guard.get("min_linear_n_obs")
        if lang == "en":
            out.append(
                f"Short-sample caution: only {n_lin} effective quarters enter at the headline "
                f"horizons (benchmark {SHORT_SAMPLE_OBS}), so HAC bands and Montiel "
                "Olea-Pflueger F cutoffs should be interpreted conservatively."
            )
        else:
            out.append(
                f"Advertencia de muestra corta: solo entran {n_lin} trimestres efectivos en los "
                f"horizontes principales (referencia {SHORT_SAMPLE_OBS}), por lo que las bandas "
                "HAC y los umbrales de F de Montiel Olea-Pflueger deben interpretarse con cautela."
            )
    if guard.get("short_regime"):
        n_reg = guard.get("min_regime_n_eff")
        if lang == "en":
            out.append(
                f"Short-regime caution: at least one regime has only {n_reg} effective quarters "
                f"(benchmark {SHORT_REGIME_OBS}); state-dependent bands have limited finite-sample "
                "precision."
            )
        else:
            out.append(
                f"Advertencia de régimen corto: al menos un régimen cuenta con solo {n_reg} "
                f"trimestres efectivos (referencia {SHORT_REGIME_OBS}); las bandas dependientes "
                "del estado tienen precisión finita limitada."
            )
    return out


def _fmt(value: float | None, digits: int = 2) -> str:
    """``value`` to ``digits`` decimals, with a decimal point in both languages; ``n/a`` if missing."""
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _horizons(record: dict[str, Any], lang: str) -> list[tuple[str, str, int]]:
    """``(record key, label, h)`` of the headline horizons, in order."""
    hs = [int(h) for h in record["specification"]["headline_horizons"]]
    if not _is_custom(record):
        labels = _READING[lang]["horizons"]
        if len(hs) != len(labels):
            raise ValueError(f"the brief words two headline horizons; the record has {hs}")
        return [(str(h), label, h) for h, label in zip(hs, labels)]
    if len(hs) != 2:
        raise ValueError(f"the brief words two headline horizons; the record has {hs}")
    return [(str(h), _horizon_label(h, lang), h) for h in hs]


def _sections(record: dict[str, Any], lang: str) -> list[tuple[str, str]]:
    """``(sample, heading)`` for the full sample and, when it was run, the experiment."""
    t = _TEXT[lang]
    full = record["full"]
    out = [("full", t["full"].format(start=full["start"], end=full["end"]))]
    if record["experiment"] is not None:
        start, end = record["experiment"]["exclude"]
        out.append(("experiment", t["experiment"].format(a=start, b=end)))
    return out


def _no_experiment(lang: str, *, custom: bool = False) -> str:
    if not custom:
        return _TEXT[lang]["no_experiment"].format(period=WWII)
    if lang == "en":
        return "No experiment was run. Pass `--exclude START:END` to re-estimate excluding an episode."
    return (
        "No se ejecutó ningún experimento. Use `--exclude INICIO:FIN` para reestimar excluyendo "
        "un episodio."
    )


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _set_kind(cell: dict[str, Any]) -> str:
    """How a weak cell's Anderson-Rubin set is worded: ``bounded``, ``unbounded`` or ``empty``.

    ``unavailable`` when the set's type needs a bound that is missing, so that no ``n/a``
    bound is ever quoted. Raises ``ValueError`` on an unknown ``ar_set_type``.
    """
    kind = cell["ar_set_type"]
    if kind not in _SET_KIND:
        raise ValueError(f"unknown Anderson-Rubin set type {kind!r}")
    if kind in ("bounded", "unbounded_rays") and not (_finite(cell["ar_lo"])
                                                       and _finite(cell["ar_hi"])):
        return "unavailable"
    return _SET_KIND[kind]


def _set_words(cell: dict[str, Any], words: dict[str, Any]) -> str:
    """A weak cell's Anderson-Rubin set in words, by its kind (:func:`_set_kind`)."""
    return words[f"set_{_set_kind(cell)}"].format(a=_fmt(cell["ar_lo"]), b=_fmt(cell["ar_hi"]))


def _clause(cell: dict[str, Any], horizon: str, words: dict[str, Any], cutoff: str) -> str:
    """A cell at one horizon: its estimate and band when strong, its Anderson-Rubin set when weak."""
    if not cell["weak"]:
        return words["strong"].format(m=_fmt(cell["multiplier"]), H=horizon,
                                      lo=_fmt(cell["lo"]), hi=_fmt(cell["hi"]))
    return words["weak"].format(H=horizon, F=_fmt(cell["f_eff"], 1), c=cutoff,
                                S=_set_words(cell, words))


def _differences(sample: dict[str, Any], horizons: list[tuple[str, str, int]],
                 words: dict[str, Any]) -> list[str]:
    """The test of equal multipliers: quoted where both regimes are strong, else not reliable."""
    tests = [(label, sample["horizons"][key]["difference"]) for key, label, _ in horizons]
    if not any(test["reliable"] for _, test in tests):
        return [words["diff_no_both"]]
    return [words["diff"].format(d=_fmt(test["estimate"]), H=label, p=_fmt(test["p_value"]))
            if test["reliable"] else words["diff_no_one"].format(H=label)
            for label, test in tests]


def _concentration(sample: dict[str, Any], name: str, row: str,
                   horizons: list[tuple[str, str, int]], words: dict[str, Any]) -> list[str]:
    """A regime's top quarter at each horizon where it carries at least the reporting share."""
    found: list[tuple[str, str, int]] = []
    for key, label, _ in horizons:
        top = sample["horizons"][key]["concentration"].get(name)
        if not top:
            continue
        if top["share"] is not None and top["share"] >= CONCENTRATION_REPORT_SHARE:
            found.append((label, top["date"], round(100 * top["share"])))
    if not found:
        return []
    if len(found) == 1:
        label, date, pct = found[0]
        return [words["conc_one"].format(R=row, D=date, p=pct, H=label)]
    (h1, d1, p1), (h2, d2, p2) = found
    if d1 != d2:
        return [words["conc_two_dates"].format(R=row, D1=d1, p1=p1, H1=h1, D2=d2, p2=p2, H2=h2)]
    if p1 == p2:
        return [words["conc_both_same"].format(R=row, D=d1, p=p1)]
    return [words["conc_both"].format(R=row, D=d1, p1=p1, H1=h1, p2=p2, H2=h2)]


def _comparisons(record: dict[str, Any], horizons: list[tuple[str, str, int]],
                 words: dict[str, Any]) -> list[str]:
    """Each strong experiment cell against the same full-sample cell, or why there is none."""
    out: list[str] = []
    row_names, row_labels = _active_rows(record, "en" if words is _READING["en"] else "es")
    for name, row in zip(row_names, row_labels):
        for key, label, _ in horizons:
            cell = record["experiment"]["horizons"][key][name]
            if cell["weak"]:
                continue
            full = record["full"]["horizons"][key][name]
            here = _fmt(cell["multiplier"])
            if not full["weak"]:
                out.append(words["cmp"].format(R=row, H=label, m=here,
                                               f=_fmt(full["multiplier"])))
            else:   # against the full sample's Anderson-Rubin set, never its weak estimate
                out.append(words["cmp_weak_full"].format(R=row, H=label, m=here,
                                                         S=_set_words(full, words)))
    return out or [words["cmp_none"]]


def _reading(record: dict[str, Any], sample_name: str, lang: str) -> str:
    """The reading of one sample, sentence by sentence in the rule's order.

    1. the excluded period (experiment only); 2. the units; 3. one sentence per row,
    linear (and slack / no slack when state-dependent), with a clause per headline horizon;
    4. the test of equal multipliers; 5. each regime's top quarter, when it carries at least
    ``CONCENTRATION_REPORT_SHARE``; 6. the strong experiment cells against the full sample;
    and 7. the three-step note (bundled RZ experiment) or identification/short-sample notes.
    """
    words = _READING[lang]
    sample = record[sample_name]
    horizons = _horizons(record, lang)
    cutoff = _fmt(record["specification"]["weak_cutoff"], 1)
    custom = _is_custom(record)
    has_state = _has_state(record)
    row_names, row_labels = _active_rows(record, lang)
    parts: list[str] = []
    if sample_name == "experiment":
        start, end = sample["exclude"]
        if not custom:
            parts.append(words["exp_head"].format(a=start, b=end))
        elif lang == "en":
            parts.append(
                f"Excluding {start}-{end}: every quarter whose window, shock or lags touch the "
                "period is dropped."
            )
        else:
            parts.append(
                f"Excluyendo {start}-{end}: se descarta todo trimestre cuya ventana, choque o "
                "rezagos tocan el periodo."
            )
    parts.append(_units_text(record, lang))
    for name, row in zip(row_names, row_labels):
        c1, c2 = (_clause(sample["horizons"][key][name], label, words, cutoff)
                  for key, label, _ in horizons)
        parts.append(words["row"].format(R=row, c1=c1, c2=c2))
    if has_state:
        parts += _differences(sample, horizons, words)
        for name, row in zip(row_names[1:], row_labels[1:]):
            parts += _concentration(sample, name, row, horizons, words)
    else:
        parts += _concentration(sample, "linear", row_labels[0], horizons, words)
    if sample_name == "experiment":
        parts += _comparisons(record, horizons, words)
        if not custom:
            parts.append(words["three_step"])
    if custom:
        assumption = (record["specification"].get("identifying_assumption") or {}).get(lang)
        if assumption:
            parts.append(str(assumption))
        parts += _sample_caution(sample, lang)
    return " ".join(parts)


def _entry(cell: dict[str, Any], t: dict[str, str], *, tex: bool = False) -> str:
    """A table entry: estimate [band] when strong; the Anderson-Rubin set and the mark when weak."""
    if not cell["weak"]:
        return f"{_fmt(cell['multiplier'])} [{_fmt(cell['lo'])}, {_fmt(cell['hi'])}]"
    if _set_kind(cell) == "unavailable":                  # also rejects an unknown set type
        return "n/a$^\\dagger$" if tex else f"n/a {_MARK}"
    a, b = _fmt(cell["ar_lo"]), _fmt(cell["ar_hi"])
    if tex:
        return {"bounded": f"[{a}, {b}]$^\\dagger$",
                "unbounded_rays": f"$(-\\infty, {b}] \\cup [{a}, \\infty)^\\dagger$",
                "all_real": "$(-\\infty, \\infty)^\\dagger$",
                "empty": f"{latex_escape(t['empty_set'])}$^\\dagger$"}[cell["ar_set_type"]]
    inf, cup = "\N{INFINITY}", "\N{UNION}"
    return {"bounded": f"[{a}, {b}] {_MARK}",
            "unbounded_rays": f"(-{inf}, {b}] {cup} [{a}, {inf}) {_MARK}",
            "all_real": f"(-{inf}, {inf}) {_MARK}",
            "empty": f"{t['empty_set']} {_MARK}"}[cell["ar_set_type"]]


def _table(record: dict[str, Any], sample: dict[str, Any], lang: str, *,
           tex: bool = False) -> tuple[list[str], list[list[str]], bool]:
    """Header, rows and whether any cell is weak, for one sample's headline table."""
    t = _TEXT[lang]
    row_names, row_labels = _active_rows(record, lang)
    esc = latex_escape if tex else str
    horizons = _horizons(record, lang)
    header = [esc(t["multiplier"])]
    for _, label, h in horizons:
        header += [esc(t["horizon"].format(H=label, h=h)), esc(t["f_eff"])]
    rows: list[list[str]] = []
    weak = False
    for name, row in zip(row_names, row_labels):
        line = [esc(row)]
        for key, _, _ in horizons:
            cell = sample["horizons"][key][name]
            weak = weak or bool(cell["weak"])
            line += [_entry(cell, t, tex=tex), _fmt(cell["f_eff"], 1)]
        rows.append(line)
    if _has_state(record):
        line = [esc(t["difference"])]
        for key, _, _ in horizons:
            test = sample["horizons"][key]["difference"]
            line += [f"{_fmt(test['estimate'])} (p = {_fmt(test['p_value'])})" if test["reliable"]
                     else esc(t["not_reliable"]), ""]
        rows.append(line)
    return header, rows, weak


def _markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join([":---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _table_note_text(record: dict[str, Any], lang: str) -> str:
    if _has_state(record):
        return _TEXT[lang]["table_note"]
    if lang == "en":
        return "Estimate [90% band]."
    return "Estimación [banda al 90%]."


def _table_block(record: dict[str, Any], sample: dict[str, Any], lang: str) -> str:
    """One sample's markdown table and its note; the weak-cell note only when a cell is weak."""
    t = _TEXT[lang]
    header, rows, weak = _table(record, sample, lang)
    note = _table_note_text(record, lang)
    if weak:
        cutoff = _fmt(record["specification"]["weak_cutoff"], 1)
        note += f" {_MARK} " + t["weak_note"].format(c=cutoff)
    return _markdown_table(header, rows) + "\n\n" + note


def page_block(record: dict[str, Any],
               block: Literal["headline", "headline_reading", "experiment", "experiment_reading"],
               lang: str) -> str:
    """One block of the brief's page, as the docs show it, in ``lang`` (``"en"`` or ``"es"``).

    ``headline`` and ``experiment`` are a sample's markdown table and its note;
    ``headline_reading`` and ``experiment_reading`` are its reading. A weak cell, with an
    effective F below the record's ``weak_cutoff``, is shown by its weak-instrument-robust
    Anderson-Rubin set and never by its estimate. Without an experiment in the record, both
    experiment blocks say how to run one.
    """
    _text(lang)
    if block == "headline":
        return _table_block(record, record["full"], lang)
    if block == "headline_reading":
        return _reading(record, "full", lang)
    if block in ("experiment", "experiment_reading"):
        if record["experiment"] is None:
            return _no_experiment(lang, custom=_is_custom(record))
        if block == "experiment":
            return _table_block(record, record["experiment"], lang)
        return _reading(record, "experiment", lang)
    raise ValueError(f"block {block!r} must be 'headline', 'headline_reading', 'experiment' "
                     "or 'experiment_reading'")


def _wrap(paragraph: str) -> str:
    """A paragraph filled to 100 columns for the terminal; tables and `code` stay unbroken."""
    if paragraph.startswith("|"):
        return paragraph
    joined = re.sub(r"`[^`]*`", lambda code: code[0].replace(" ", "\0"), paragraph)
    return textwrap.fill(joined, width=100, break_long_words=False,
                         break_on_hyphens=False).replace("\0", " ")


#: The terminal spelling of the signs of an unbounded set, which a legacy console code page
#: such as cp1252 cannot encode; the written files keep the signs.
_TERMINAL = str.maketrans({"\N{INFINITY}": "inf", "\N{UNION}": "U"})


def summary_text(record: dict[str, Any], lang: str = "en") -> str:
    """The plain-text summary :func:`main` prints: each sample's table and reading.

    An unbounded set's infinity and union signs are spelled ``inf`` and ``U``, as in
    ``(-inf, b] U [a, inf)``, so that any console can print them; the files keep the signs.
    """
    t = _text(lang)
    parts = [_title_text(record, lang), t["source"].format(citation=record["data"]["citation"])]
    parts += _identification_lines(record, lang)
    for name, heading in _sections(record, lang):
        parts += [heading, _table_block(record, record[name], lang), _reading(record, name, lang)]
    if record["experiment"] is None:
        parts.append(_no_experiment(lang, custom=_is_custom(record)))
    text = "\n\n".join(_wrap(p) for p in "\n\n".join(parts).split("\n\n")) + "\n"
    return text.translate(_TERMINAL)


def _markdown_page(record: dict[str, Any], lang: str) -> str:
    """``headline.md``: each sample's table and reading under its heading."""
    t = _TEXT[lang]
    parts = [
        f"# {_title_text(record, lang)}",
        t["source"].format(citation=record["data"]["citation"]),
    ]
    parts += _identification_lines(record, lang)
    for name, heading in _sections(record, lang):
        parts += [f"## {heading}", _table_block(record, record[name], lang),
                  _reading(record, name, lang)]
    if record["experiment"] is None:
        parts.append(_no_experiment(lang, custom=_is_custom(record)))
    return "\n\n".join(parts) + "\n"


_CSV_COLUMNS = ["sample", "h", "row", "label", "estimate", "se", "lo", "hi", "f_eff", "weak",
                "ar_lo", "ar_hi", "ar_set_type", "n_eff", "top_quarter", "top_share", "p_value",
                "reliable"]


def _headline_frame(record: dict[str, Any], lang: str) -> pd.DataFrame:
    """``headline.csv``: the headline table's cells, one row per sample, horizon and estimate.

    The rows ``linear`` (and ``slack`` / ``no_slack`` when state-dependent) hold a cell
    (``estimate`` is its multiplier), and ``difference`` the test of equal multipliers.
    Like the markdown and LaTeX tables, it shows a weak cell by its Anderson-Rubin set and
    effective F only, leaving ``estimate``, ``se``, ``lo`` and ``hi`` empty, and an unreliable
    difference without ``estimate``, ``se`` or ``p_value``. Numbers are at full precision.
    """
    t = _TEXT[lang]
    row_names, row_labels = _active_rows(record, lang)
    has_state = _has_state(record)
    rows: list[dict[str, Any]] = []
    for sample_name, _ in _sections(record, lang):
        for key, _, h in _horizons(record, lang):
            cells = record[sample_name]["horizons"][key]
            for name, label in zip(row_names, row_labels):
                cell = cells[name]
                top = (cells["concentration"] or {}).get(name) or {}
                point = {} if cell["weak"] else {"estimate": cell["multiplier"], "se": cell["se"],
                                                 "lo": cell["lo"], "hi": cell["hi"]}
                rows.append({"sample": sample_name, "h": h, "row": name, "label": label, **point,
                             "f_eff": cell["f_eff"], "weak": cell["weak"],
                             "ar_lo": cell["ar_lo"], "ar_hi": cell["ar_hi"],
                             "ar_set_type": cell["ar_set_type"], "n_eff": cell.get("n_eff"),
                             "top_quarter": top.get("date"), "top_share": top.get("share")})
            if has_state:
                test = cells["difference"]
                quoted = {} if not test["reliable"] else {
                    "estimate": test["estimate"],
                    "se": test["se"],
                    "p_value": test["p_value"],
                }
                rows.append({"sample": sample_name, "h": h, "row": "difference",
                             "label": t["difference"], **quoted, "reliable": test["reliable"]})
    return pd.DataFrame(rows, columns=_CSV_COLUMNS)


def _latex_table(record: dict[str, Any], sample: dict[str, Any], lang: str) -> str:
    """One sample's headline table as a LaTeX ``tabular``, its notes in the last rows."""
    t = _TEXT[lang]
    header, rows, weak = _table(record, sample, lang, tex=True)
    width = len(header)
    lines = ["\\begin{tabular}{l" + "r" * (width - 1) + "}", " & ".join(header) + " \\\\",
             "\\hline"]
    lines += [" & ".join(row) + " \\\\" for row in rows]
    notes = [latex_escape(_table_note_text(record, lang))]
    if weak:
        cutoff = _fmt(record["specification"]["weak_cutoff"], 1)
        notes.append("$^\\dagger$ " + latex_escape(t["weak_note"].format(c=cutoff)))
    lines.append("\\hline")
    lines += [f"\\multicolumn{{{width}}}{{l}}{{\\footnotesize {note}}} \\\\" for note in notes]
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def _latex_page(record: dict[str, Any], lang: str) -> str:
    """``headline.tex``: a ``tabular`` per sample, each under a comment with its heading."""
    t = _TEXT[lang]
    lines = [
        f"% {_title_text(record, lang)}",
        f"% {t['source'].format(citation=record['data']['citation'])}",
    ]
    for extra in _identification_lines(record, lang):
        lines.append(f"% {extra}")
    for name, heading in _sections(record, lang):
        lines += ["", f"% {heading}", _latex_table(record, record[name], lang)]
    return "\n".join(lines) + "\n"


#: Categorical slots 1-3 of a palette checked for colour-vision deficiency, one per path.
_COLORS = {"linear": "#2a78d6", "slack": "#eb6834", "no_slack": "#1baf7a"}
_INK, _INK_2, _AXIS, _SURFACE, _SHADE = "#0b0b0b", "#52514e", "#c3c2b7", "#fcfcfb", "#ebeae4"
_DASHES = (0, (3.5, 2.5))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``(start, stop)`` of each run of True in ``mask``."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _y_range(est: np.ndarray, lo: np.ndarray, hi: np.ndarray,
             strong: np.ndarray) -> tuple[float, float]:
    """A panel's vertical range: 0, 1, every strong band and the weak estimates near them."""
    core = np.concatenate([[0.0, 1.0], lo[strong], hi[strong]])
    core = core[np.isfinite(core)]
    bottom, top = float(core.min()), float(core.max())
    reach = top - bottom
    with np.errstate(invalid="ignore"):
        near = est[~strong & (est >= bottom - reach) & (est <= top + reach)]
    if near.size:
        bottom, top = min(bottom, float(near.min())), max(top, float(near.max()))
    pad = 0.08 * (top - bottom)
    return bottom - pad, top + pad


def _panel(ax: Any, frame: pd.DataFrame, name: str, title: str, cutoff: float,
           ticks: list[int]) -> bool:
    """One path and its band: solid where strong; dashed, hollow and shaded where weak.

    A weak estimate beyond the vertical range is drawn as a triangle on that edge. Returns
    whether there is one.
    """
    h = frame["h"].to_numpy(dtype=float)
    est = frame[name].to_numpy(dtype=float)
    lo = frame[f"{name}_lo"].to_numpy(dtype=float)
    hi = frame[f"{name}_hi"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        strong = frame[f"{name}_f"].to_numpy(dtype=float) >= cutoff     # a missing F is weak
    color = _COLORS[name]
    for start, stop in _runs(~strong):
        ax.axvspan(h[start] - 0.5, h[stop - 1] + 0.5, color=_SHADE, lw=0, zorder=0)
    ax.axhline(0.0, color=_INK_2, lw=0.8, zorder=1)
    ax.axhline(1.0, color=_AXIS, lw=0.8, zorder=1)
    ax.fill_between(h, lo, hi, color=color, alpha=0.2, lw=0, zorder=2)
    ax.plot(h, est, color=color, lw=1.6, ls=_DASHES, zorder=3)
    ax.plot(h, np.where(strong, est, np.nan), color=color, lw=1.6, zorder=4)
    ax.plot(h[strong], est[strong], "o", color=color, ms=4, zorder=5)
    ax.plot(h[~strong], est[~strong], "o", color=color, mfc=_SURFACE, ms=4, zorder=5)
    bottom, top = _y_range(est, lo, hi, strong)
    ax.set_ylim(bottom, top)
    inset = 0.04 * (top - bottom)
    with np.errstate(invalid="ignore"):
        below, above = ~strong & (est < bottom), ~strong & (est > top)
    ax.plot(h[below], np.full(int(below.sum()), bottom + inset), "v", color=color, ms=6, zorder=6)
    ax.plot(h[above], np.full(int(above.sum()), top - inset), "^", color=color, ms=6, zorder=6)
    ax.set_title(title, fontsize=10, color=_INK)
    ax.set_xlim(h.min() - 0.5, h.max() + 0.5)
    ax.set_xticks(ticks)
    ax.tick_params(labelsize=8, colors=_INK_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    return bool(below.any() or above.any())


def _figure_png(record: dict[str, Any], paths: pd.DataFrame, lang: str) -> bytes:
    """``multipliers.png``: linear (and slack / no slack when present), a row per sample."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    t = _TEXT[lang]
    row_names, row_labels = _active_rows(record, lang)
    cutoff = float(record["specification"]["weak_cutoff"])
    sections = _sections(record, lang)
    hs = sorted(int(h) for h in paths["h"].unique())
    ticks = sorted({hs[0], *(h for _, _, h in _horizons(record, lang)), hs[-1]})
    fig_width = 11.0 if len(row_names) > 1 else 5.5
    fig = Figure(figsize=(fig_width, 1.0 + 3.1 * len(sections)), dpi=120, facecolor=_SURFACE,
                 layout="constrained")
    FigureCanvasAgg(fig)
    off_axis = False
    for subfig, (name, heading) in zip(fig.subfigures(len(sections), 1, squeeze=False)[:, 0],
                                       sections):
        subfig.set_facecolor(_SURFACE)
        subfig.suptitle(heading, x=0.01, ha="left", fontsize=10.5, color=_INK)
        frame = paths.loc[paths["sample"] == name].sort_values("h")
        axes = np.atleast_1d(subfig.subplots(1, len(row_names))).ravel()
        for ax, row, title in zip(axes, row_names, row_labels):
            off_axis = _panel(ax, frame, row, title, cutoff, ticks) or off_axis
            ax.set_xlabel(t["fig_x"], fontsize=8.5, color=_INK_2)
        axes[0].set_ylabel(t["fig_y"], fontsize=8.5, color=_INK_2)
    c = _fmt(cutoff, 1)
    handles: list[Any] = [
        Line2D([], [], color=_INK_2, lw=1.6, marker="o", ms=4),
        (Patch(facecolor=_SHADE),
         Line2D([], [], color=_INK_2, lw=1.6, ls=_DASHES, marker="o", ms=4, mfc=_SURFACE)),
        Patch(facecolor=_INK_2, alpha=0.2),
    ]
    labels = [t["fig_strong"].format(c=c), t["fig_weak"].format(c=c), t["fig_band"]]
    if off_axis:
        handles.append(Line2D([], [], color=_INK_2, ls="none", marker="v", ms=6))
        labels.append(t["fig_off"])
    fig.legend(handles, labels, loc="outside lower center", ncols=2 if len(row_names) > 1 else 1,
               frameon=False, fontsize=8.5, labelcolor=_INK_2)
    fig.suptitle(t["fig_title"], fontsize=11.5, color=_INK)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=_SURFACE, metadata={"Software": None})
    return buffer.getvalue()


def write_outputs(record: dict[str, Any], paths: pd.DataFrame, output: str | Path, *,
                  lang: str = "en") -> dict[str, Any]:
    """Write the brief's files into ``output``, in ``lang``, and return the manifest.

    Every file is rendered in memory first, so nothing is written when one cannot be. The
    files are ``headline.md`` (tables and readings), ``headline.csv`` (the headline cells at
    full precision), ``headline.tex``, ``paths.csv`` (h = 0..20), ``multipliers.png``,
    ``headline.json`` (``record``) and ``manifest.json``.

    The three tables, ``headline.md``, ``headline.csv`` and ``headline.tex``, show the same
    cells: a weak cell by its Anderson-Rubin set and effective F, never by its estimate,
    standard error or band, and an unreliable difference without its estimate or p-value.
    ``headline.json`` and ``paths.csv`` keep the complete raw data, for reproducibility and
    the figure: every estimate, with the ``weak`` flags in the record and the effective F
    of every path and horizon in ``paths.csv``.

    The manifest holds ``lang``,
    ``puremacro_version``, ``data`` and ``specification`` from the record; ``experiment``,
    what was run (``{"exclude": [START, END]}``, or None), while its results are in
    ``headline.json``; the ``warnings`` and computing time of the last :func:`compute_brief`
    call, with the outputs' wall seconds (``timings``); ``environment``; and ``artifacts``:
    each other file's SHA-256 and size. It records no absolute path.
    """
    started = time.perf_counter()
    _text(lang)
    json.dumps(record, allow_nan=False)            # JSON-ready, or nothing is written
    texts = {"headline.md": _markdown_page(record, lang),
             "headline.csv": _headline_frame(record, lang).to_csv(index=False),
             "headline.tex": _latex_page(record, lang),
             "paths.csv": paths.to_csv(index=False)}
    content = {name: text.encode("utf-8") for name, text in texts.items()}
    content["multipliers.png"] = _figure_png(record, paths, lang)

    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in content.items():
        (folder / name).write_bytes(data)
    write_json_atomic(folder / "headline.json", record)
    artifacts = {name: {"sha256": sha256(folder / name), "bytes": (folder / name).stat().st_size}
                 for name in (*content, "headline.json")}
    compute = _LAST_TIMINGS.get("compute_seconds")
    experiment = record["experiment"]
    manifest = {
        "lang": lang,
        "puremacro_version": record["puremacro_version"],
        **copy.deepcopy({key: record[key] for key in ("data", "specification")}),
        # what was run, not its results, which are in headline.json
        "experiment": None if experiment is None else {"exclude": list(experiment["exclude"])},
        "warnings": list(_LAST_WARNINGS),
        "timings": {"compute_seconds": None if compute is None else round(compute, 3),
                    "outputs_seconds": round(time.perf_counter() - started, 3)},
        "environment": environment(),
        "artifacts": artifacts,
    }
    write_json_atomic(folder / "manifest.json", manifest)
    return manifest


def run_brief(
    output: str | Path,
    *,
    exclude: str | None = None,
    lang: str = "en",
    data: pd.DataFrame | str | Path | None = None,
    config: FiscalBriefConfig | Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Compute the brief (:func:`compute_brief`), then write its files (:func:`write_outputs`).

    Nothing is created or written unless the computation succeeds. Returns
    ``{"record", "paths", "manifest"}``.
    """
    _text(lang)
    record, paths = compute_brief(exclude=exclude, data=data, config=config)
    manifest = write_outputs(record, paths, output, lang=lang)
    return {"record": record, "paths": paths, "manifest": manifest}


def _echo(text: str) -> None:
    """Print ``text``, replacing any character that stdout's encoding cannot encode.

    The files are written before the summary is printed, so an unencodable character (in
    an output path, say, on a cp1252 console) must not turn a finished run into an error.
    """
    encoding = getattr(sys.stdout, "encoding", None)
    if encoding:
        try:
            text.encode(encoding)
        except UnicodeEncodeError:
            text = text.encode(encoding, errors="replace").decode(encoding)
        except LookupError:      # not a codec Python knows: leave the text to the stream
            pass
    print(text)


def _build_cli_config(args: argparse.Namespace) -> tuple[Path | None, FiscalBriefConfig | None]:
    """Assemble ``(data_path, config)`` from CLI flags, or ``(None, None)`` for the bundled run."""
    custom_flags = (
        args.config, args.data, args.date_col, args.filter_col, args.filter_value,
        args.outcome, args.policy, args.instrument, args.shock_mode, args.deflator,
        args.potential, args.scaling, args.poly_degree, args.hp_lambda, args.state,
        args.state_threshold, args.state_op, args.state_lag, args.slack_rule,
        args.lags, args.headline_horizons, args.max_horizon, args.citation,
        args.shock_source, args.identifying_assumption, args.title, args.units,
    )
    if not any(v is not None for v in custom_flags) and not args.linear_only:
        return None, None

    base_dict: dict[str, Any] = {}
    data_from_cfg: Path | None = None
    if args.config is not None:
        cfg_obj, extras = _coerce_config(args.config)
        base_dict = asdict(cfg_obj)
        if "data" in extras and extras["data"] is not None:
            raw_p = Path(str(extras["data"]))
            data_from_cfg = raw_p if raw_p.is_absolute() else (args.config.parent / raw_p)
        if args.exclude is None and extras.get("exclude") is not None:
            args.exclude = str(extras["exclude"])

    overrides: dict[str, Any] = {}
    if args.date_col is not None:
        overrides["date_col"] = args.date_col
    if args.filter_col is not None:
        overrides["filter_col"] = args.filter_col
    if args.filter_value is not None:
        overrides["filter_value"] = args.filter_value
    if args.outcome is not None:
        overrides["outcome"] = args.outcome
    if args.policy is not None:
        overrides["policy"] = args.policy
    if args.instrument is not None:
        overrides["instrument"] = args.instrument
    if args.shock_mode is not None:
        overrides["shock_mode"] = args.shock_mode
    if args.deflator is not None:
        overrides["deflator_col"] = args.deflator
    if args.potential is not None:
        overrides["potential_col"] = args.potential
    if args.scaling is not None:
        overrides["scaling"] = args.scaling
    if args.poly_degree is not None:
        overrides["poly_degree"] = args.poly_degree
    if args.hp_lambda is not None:
        overrides["hp_lambda"] = args.hp_lambda
    if args.linear_only:
        overrides["state"] = None
    elif args.state is not None:
        overrides["state"] = args.state
    if args.state_threshold is not None:
        overrides["state_threshold"] = args.state_threshold
    if args.state_op is not None:
        overrides["state_op"] = args.state_op
    if args.state_lag is not None:
        overrides["state_lag"] = args.state_lag
    if args.slack_rule is not None:
        overrides["slack_rule"] = args.slack_rule
    if args.lags is not None:
        overrides["n_lags"] = args.lags
    if args.max_horizon is not None:
        if args.max_horizon < 1:
            raise ValueError("--max-horizon must be >= 1")
        overrides["horizons"] = tuple(range(args.max_horizon + 1))
    if args.headline_horizons is not None:
        parts = [p.strip() for p in str(args.headline_horizons).split(",") if p.strip()]
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            raise ValueError("--headline-horizons must be H1,H2 (e.g. 7,15 or 3,7)")
        overrides["headline_horizons"] = (int(parts[0]), int(parts[1]))
    if args.citation is not None:
        overrides["citation"] = args.citation
    if args.shock_source is not None:
        overrides["shock_source"] = args.shock_source
    if args.identifying_assumption is not None:
        overrides["identifying_assumption_en"] = args.identifying_assumption
        overrides["identifying_assumption_es"] = args.identifying_assumption
    if args.title is not None:
        overrides["title_en"] = args.title
        overrides["title_es"] = args.title
    if args.units is not None:
        overrides["units_en"] = args.units
        overrides["units_es"] = args.units

    merged = {**base_dict, **overrides}
    data_path = args.data if args.data is not None else data_from_cfg
    if data_path is None:
        raise ValueError("own-data fiscal brief requires --data CSV_PATH (or 'data' in --config)")
    return data_path, FiscalBriefConfig(**merged)


def main(argv: Sequence[str] | None = None) -> int:
    """The command line: check the arguments, run the brief, print its summary in ``--lang``."""
    parser = argparse.ArgumentParser(
        prog="python -m puremacro.examples.first_brief_fiscal",
        description=("Fiscal policy brief: cumulative government spending multipliers with "
                     "weak-IV Anderson-Rubin sets on bundled RZ2018 or user quarterly data."))
    parser.add_argument("--exclude", metavar="START:END", default=None,
                        help=("also estimate without this period, both quarters included, "
                              f"by Ramey and Zubairy's rule (World War II: {WWII})"))
    parser.add_argument("--output", metavar="DIR", type=Path, default=DEFAULT_OUTPUT,
                        help=f"output directory (default {DEFAULT_OUTPUT})")
    parser.add_argument("--lang", choices=("en", "es"), default="en",
                        help="language of the outputs (default en)")
    parser.add_argument("--config", metavar="JSON", type=Path, default=None,
                        help="JSON configuration file for own-data estimation")
    parser.add_argument("--data", metavar="CSV", type=Path, default=None,
                        help="quarterly CSV dataset for own-data estimation")
    parser.add_argument("--date-col", metavar="COL", default=None,
                        help="date column (YYYYQn, YYYYTn, or ISO YYYY-MM-DD)")
    parser.add_argument("--filter-col", metavar="COL", default=None,
                        help="optional panel filter column (e.g. code)")
    parser.add_argument("--filter-value", metavar="VAL", default=None,
                        help="value to keep in --filter-col (e.g. DEU)")
    parser.add_argument("--outcome", metavar="COL", default=None,
                        help="outcome column name (e.g. gdp_real)")
    parser.add_argument("--policy", metavar="COL", default=None,
                        help="fiscal policy column name (e.g. cons_gov_real)")
    parser.add_argument("--instrument", metavar="COL", default=None,
                        help="external shock/instrument column")
    parser.add_argument("--shock-mode", choices=_SHOCK_MODES, default=None,
                        help="shock identification mode (external_instrument or blanchard_perotti)")
    parser.add_argument("--deflator", metavar="COL", default=None,
                        help="optional price deflator column applied to outcome and policy")
    parser.add_argument("--potential", metavar="COL", default=None,
                        help="optional potential output column for Gordon-Krenn scaling")
    parser.add_argument("--scaling", choices=_SCALING_MODES, default=None,
                        help="scaling transformation (pre_scaled, potential_col, hp_trend, poly_trend)")
    parser.add_argument("--poly-degree", type=int, default=None,
                        help="polynomial degree when --scaling poly_trend (default 2)")
    parser.add_argument("--hp-lambda", type=float, default=None,
                        help="Hodrick-Prescott smoothing when --scaling hp_trend (default 1600)")
    parser.add_argument("--state", metavar="COL", default=None,
                        help="state variable column for slack/no-slack regimes")
    parser.add_argument("--linear-only", action="store_true",
                        help="estimate only the linear cumulative multiplier (no state split)")
    parser.add_argument("--state-threshold", type=float, default=None,
                        help="threshold defining the slack regime (default 0.5)")
    parser.add_argument("--state-op", choices=_STATE_OPS, default=None,
                        help="comparison operator for slack regime (default >=)")
    parser.add_argument("--state-lag", type=int, default=None,
                        help="quarters to lag --state before thresholding (default 0)")
    parser.add_argument("--slack-rule", metavar="TEXT", default=None,
                        help="human-readable slack rule label")
    parser.add_argument("--lags", type=int, default=None,
                        help="lag length for automatic control construction (default 4)")
    parser.add_argument("--headline-horizons", metavar="H1,H2", default=None,
                        help="two comma-separated headline horizons (default 7,15)")
    parser.add_argument("--max-horizon", type=int, default=None,
                        help="maximum horizon H for path estimation 0..H (default 20)")
    parser.add_argument("--citation", metavar="TEXT", default=None,
                        help="dataset citation recorded in outputs and manifest")
    parser.add_argument("--shock-source", metavar="TEXT", default=None,
                        help="explicit shock source description")
    parser.add_argument("--identifying-assumption", metavar="TEXT", default=None,
                        help="explicit causal identifying assumption recorded in the brief")
    parser.add_argument("--title", metavar="TEXT", default=None,
                        help="custom brief title")
    parser.add_argument("--units", metavar="TEXT", default=None,
                        help="custom multiplier units description")

    args = parser.parse_args(argv)
    try:
        data_path, cfg = _build_cli_config(args)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if data_path is None and cfg is None:
        if args.exclude is not None:      # checked before anything is computed or written
            try:
                exclude_period(load_brief_data(), parse_exclude(args.exclude))
            except _ExcludeError as exc:
                parser.error(exc.text(args.lang))
        study = run_brief(args.output, exclude=args.exclude, lang=args.lang)
    else:
        try:
            study = run_brief(
                args.output, exclude=args.exclude, lang=args.lang, data=data_path, config=cfg
            )
        except _ExcludeError as exc:
            parser.error(exc.text(args.lang))
        except (OSError, ValueError, TypeError) as exc:
            parser.error(str(exc))

    _echo(summary_text(study["record"], args.lang))
    _echo(_TEXT[args.lang]["written"].format(path=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
