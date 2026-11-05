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
from collections.abc import Sequence
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
    "CONCENTRATION_REPORT_SHARE", "DEFAULT_OUTPUT", "parse_exclude", "load_brief_data",
    "exclude_period", "estimate", "concentration", "compute_brief", "page_block",
    "summary_text", "write_outputs", "run_brief", "main",
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

_MASKED = ("y", "g", "newsy")       # outcome, policy variable and instrument
_RUNNING_SUMS = ("cumy", "cumg")    # summed before any masking
#: Regime -> (suffix of its ``lp_state_dep_iv`` columns, its value of ``slack_l``).
_REGIMES: dict[str, tuple[str, float]] = {"slack": ("H", 1.0), "no_slack": ("L", 0.0)}
_EXCLUDE = re.compile(r"\s*([0-9]{4})Q([1-4])\s*:\s*([0-9]{4})Q([1-4])\s*", re.IGNORECASE)
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


def parse_exclude(text: str) -> tuple[pd.Period, pd.Period]:
    """``"1941Q3:1945Q4"`` -> the first and last excluded quarters, both inclusive.

    Raises ``ValueError`` unless ``text`` is ``START:END`` in ``YYYYQn``
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


def exclude_period(df: pd.DataFrame, period: tuple[pd.Period, pd.Period]) -> pd.DataFrame:
    """Ramey and Zubairy's exclusion of ``period``, both ends inclusive, on a copy of ``df``.

    The copy keeps every row. ``y``, ``g`` and ``newsy`` are NaN in the
    period, and the lag controls ``RZ_CONTROLS`` are rebuilt from the masked
    series. The running sums ``cumy`` and ``cumg`` were summed before the
    masking, so they are dropped. ``cumulative=True`` then forms the window
    sums from the masked series: a quarter drops out of horizon h when
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
    out = df.copy().drop(columns=list(_RUNNING_SUMS), errors="ignore")
    out.loc[inside, list(_MASKED)] = np.nan
    for name in RZ_CONTROLS:                 # "{series}_l{k}" is series.shift(k)
        series, lag = name.rsplit("_l", 1)
        out[name] = out[series].shift(int(lag))
    return out


def estimate(df: pd.DataFrame) -> dict[str, LPResult]:
    """The brief's four estimator calls on one sample, h = 0..20, 90% bands.

    Returns ``linear`` (:func:`lp_iv` with the HAC effective F and
    Anderson-Rubin set), ``state`` (:func:`lp_state_dep_iv` by regime of
    ``slack_l``), and ``ar_slack`` / ``ar_no_slack`` (:func:`lp_iv` with
    ``newsy`` NaN outside the regime; a missing ``slack_l`` counts as
    outside). All four are cumulative, on ``RZ_CONTROLS`` and the full frame.
    """
    out: dict[str, LPResult] = {
        "linear": lp_iv(df, y="y", x="g", z="newsy", horizons=HORIZONS, n_lags=0,
                        controls=RZ_CONTROLS, alpha=ALPHA, cumulative=True,
                        anderson_rubin=True),
        "state": lp_state_dep_iv(df, y="y", x="g", z="newsy", state="slack_l",
                                 horizons=HORIZONS, n_lags=0, transition="threshold",
                                 threshold=0.5, controls=RZ_CONTROLS, alpha=ALPHA,
                                 interact="all", cumulative=True),
    }
    for name, (_, value) in _REGIMES.items():
        masked = df.copy()
        masked.loc[masked["slack_l"] != value, "newsy"] = np.nan
        out[f"ar_{name}"] = lp_iv(masked, y="y", x="g", z="newsy", horizons=HORIZONS,
                                  n_lags=0, controls=RZ_CONTROLS, alpha=ALPHA,
                                  cumulative=True, anderson_rubin=True)
    return out


def concentration(df: pd.DataFrame, regime: float, h: int) -> dict[str, Any]:
    """The quarter that carries the most of a regime's residual instrument variation at ``h``.

    The rows are horizon ``h``'s estimation rows with ``slack_l == regime``:
    both window sums present, and ``newsy`` and every ``RZ_CONTROLS`` column
    non-missing. ``newsy`` is regressed on a constant and ``RZ_CONTROLS`` by
    least squares on those rows. Returns ``{"share": max(e^2) / sum(e^2),
    "date": "YYYYQn"}``, where ``date`` is the quarter of the largest
    squared residual.

    Raises ``ValueError`` when the regime has no more rows than regressors.
    """
    y_w, x_w = cumulative_windows(df["y"], df["g"], h)
    present = pd.concat([y_w, x_w, df["newsy"], df[RZ_CONTROLS]], axis=1).notna().all(axis=1)
    rows = df.loc[present & (df["slack_l"] == regime)]
    design = np.column_stack([np.ones(len(rows)), rows[RZ_CONTROLS].to_numpy(dtype=float)])
    if len(rows) <= design.shape[1]:
        raise ValueError(f"concentration: slack_l == {regime:g} has {len(rows)} estimation rows "
                         f"at h = {h}, too few for {design.shape[1]} regressors")
    z = rows["newsy"].to_numpy(dtype=float)
    coef = np.linalg.lstsq(design, z, rcond=None)[0]
    e2 = (z - design @ coef) ** 2
    top = int(np.argmax(e2))
    return {"share": float(e2[top] / e2.sum()), "date": str(_quarters(rows.index)[top])}


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
    state = fit["state"].loc[h]
    cells: dict[str, Any] = {"linear": _cell(linear, "", "mop_f", linear, cutoff)}
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


def _sample_record(df: pd.DataFrame, fit: dict[str, LPResult],
                   concentrations: dict[int, dict[str, dict[str, Any]]],
                   cutoff: float) -> dict[str, Any]:
    """One sample's first and last quarters and its cells at each headline horizon."""
    quarters = _quarters(df.index)
    return {"start": str(quarters[0]), "end": str(quarters[-1]),
            "horizons": {str(h): _cells(fit, concentrations[h], h, cutoff)
                         for h in HEADLINE_HORIZONS}}


def _paths(sample: str, fit: dict[str, LPResult]) -> pd.DataFrame:
    """One row per horizon: the three multipliers with their bands and F, and ``p_diff``."""
    hs = list(HORIZONS)
    linear = fit["linear"].loc[hs]
    state = fit["state"].loc[hs]
    frame = pd.DataFrame({"sample": sample, "h": hs})
    for column, source in (("linear", "beta"), ("linear_lo", "lo"), ("linear_hi", "hi"),
                           ("linear_f", "mop_f")):
        frame[column] = linear[source].to_numpy(dtype=float)
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


def compute_brief(*, exclude: str | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    """Estimate the brief on the full sample and, with ``exclude``, on the experiment.

    ``exclude`` is a period ``START:END`` (see :func:`parse_exclude`),
    removed by Ramey and Zubairy's rule (:func:`exclude_period`).

    Returns ``(record, paths)``. ``record`` is JSON-ready: non-finite
    numbers are None, and ``ar_set_type`` says why an Anderson-Rubin bound
    is None. Its keys are ``puremacro_version``, ``data`` (``file``,
    ``sha256``, ``citation``), ``specification``, ``full`` and
    ``experiment`` (None without ``exclude``). Each sample holds ``start``,
    ``end`` and, under ``horizons["7"]`` and ``horizons["15"]``, the cells
    ``linear``, ``slack`` and ``no_slack`` (``multiplier, se, lo, hi, f_eff,
    ar_lo, ar_hi, ar_set_type, weak``, plus ``n_eff`` by regime),
    ``difference`` (``estimate, se, p_value, reliable``) and
    ``concentration`` (``share``, ``date`` by regime). ``paths`` has one row
    per sample and horizon h = 0..20.

    Warnings issued while a sample is computed are kept in
    ``_LAST_WARNINGS`` for the run's manifest, and re-issued once the sample
    is done, or has failed. The call's wall time is kept in ``_LAST_TIMINGS``.
    """
    _LAST_WARNINGS.clear()
    _LAST_TIMINGS.clear()
    started = time.perf_counter()
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
                concentrations[sample] = {h: {name: concentration(frame, value, h)
                                              for name, (_, value) in _REGIMES.items()}
                                          for h in HEADLINE_HORIZONS}
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


def _fmt(value: float | None, digits: int = 2) -> str:
    """``value`` to ``digits`` decimals, with a decimal point in both languages; ``n/a`` if missing."""
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _horizons(record: dict[str, Any], lang: str) -> list[tuple[str, str, int]]:
    """``(record key, label, h)`` of the headline horizons, in order."""
    hs = [int(h) for h in record["specification"]["headline_horizons"]]
    labels = _READING[lang]["horizons"]
    if len(hs) != len(labels):
        raise ValueError(f"the brief words two headline horizons; the record has {hs}")
    return [(str(h), label, h) for h, label in zip(hs, labels)]


def _sections(record: dict[str, Any], lang: str) -> list[tuple[str, str]]:
    """``(sample, heading)`` for the full sample and, when it was run, the experiment."""
    t = _TEXT[lang]
    full = record["full"]
    out = [("full", t["full"].format(start=full["start"], end=full["end"]))]
    if record["experiment"] is not None:
        start, end = record["experiment"]["exclude"]
        out.append(("experiment", t["experiment"].format(a=start, b=end)))
    return out


def _no_experiment(lang: str) -> str:
    return _TEXT[lang]["no_experiment"].format(period=WWII)


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
        top = sample["horizons"][key]["concentration"][name]
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
    for name, row in zip(_ROWS, words["rows"]):
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
    linear, slack and no slack, with a clause per headline horizon; 4. the test of equal
    multipliers; 5. each regime's top quarter, when it carries at least
    ``CONCENTRATION_REPORT_SHARE``; 6. the strong experiment cells against the full sample
    and 7. the three-step note (experiment only).
    """
    words = _READING[lang]
    sample = record[sample_name]
    horizons = _horizons(record, lang)
    cutoff = _fmt(record["specification"]["weak_cutoff"], 1)
    parts: list[str] = []
    if sample_name == "experiment":
        start, end = sample["exclude"]
        parts.append(words["exp_head"].format(a=start, b=end))
    parts.append(words["units"])
    for name, row in zip(_ROWS, words["rows"]):
        c1, c2 = (_clause(sample["horizons"][key][name], label, words, cutoff)
                  for key, label, _ in horizons)
        parts.append(words["row"].format(R=row, c1=c1, c2=c2))
    parts += _differences(sample, horizons, words)
    for name, row in zip(_ROWS[1:], words["rows"][1:]):
        parts += _concentration(sample, name, row, horizons, words)
    if sample_name == "experiment":
        parts += _comparisons(record, horizons, words)
        parts.append(words["three_step"])
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
    t, words = _TEXT[lang], _READING[lang]
    esc = latex_escape if tex else str
    horizons = _horizons(record, lang)
    header = [esc(t["multiplier"])]
    for _, label, h in horizons:
        header += [esc(t["horizon"].format(H=label, h=h)), esc(t["f_eff"])]
    rows: list[list[str]] = []
    weak = False
    for name, row in zip(_ROWS, words["rows"]):
        line = [esc(row)]
        for key, _, _ in horizons:
            cell = sample["horizons"][key][name]
            weak = weak or bool(cell["weak"])
            line += [_entry(cell, t, tex=tex), _fmt(cell["f_eff"], 1)]
        rows.append(line)
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


def _table_block(record: dict[str, Any], sample: dict[str, Any], lang: str) -> str:
    """One sample's markdown table and its note; the weak-cell note only when a cell is weak."""
    t = _TEXT[lang]
    header, rows, weak = _table(record, sample, lang)
    note = t["table_note"]
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
            return _no_experiment(lang)
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
    parts = [t["title"], t["source"].format(citation=record["data"]["citation"])]
    for name, heading in _sections(record, lang):
        parts += [heading, _table_block(record, record[name], lang), _reading(record, name, lang)]
    if record["experiment"] is None:
        parts.append(_no_experiment(lang))
    text = "\n\n".join(_wrap(p) for p in "\n\n".join(parts).split("\n\n")) + "\n"
    return text.translate(_TERMINAL)


def _markdown_page(record: dict[str, Any], lang: str) -> str:
    """``headline.md``: each sample's table and reading under its heading."""
    t = _TEXT[lang]
    parts = [f"# {t['title']}", t["source"].format(citation=record["data"]["citation"])]
    for name, heading in _sections(record, lang):
        parts += [f"## {heading}", _table_block(record, record[name], lang),
                  _reading(record, name, lang)]
    if record["experiment"] is None:
        parts.append(_no_experiment(lang))
    return "\n\n".join(parts) + "\n"


_CSV_COLUMNS = ["sample", "h", "row", "label", "estimate", "se", "lo", "hi", "f_eff", "weak",
                "ar_lo", "ar_hi", "ar_set_type", "n_eff", "top_quarter", "top_share", "p_value",
                "reliable"]


def _headline_frame(record: dict[str, Any], lang: str) -> pd.DataFrame:
    """``headline.csv``: the headline table's cells, one row per sample, horizon and estimate.

    The rows ``linear``, ``slack`` and ``no_slack`` hold a cell (``estimate`` is its
    multiplier), and ``difference`` the test of equal multipliers (slack minus no slack).
    ``label`` is the row's name in ``lang``. Like the markdown and LaTeX tables, it shows a
    weak cell by its Anderson-Rubin set and effective F only, leaving ``estimate``, ``se``,
    ``lo`` and ``hi`` empty, and an unreliable difference without ``estimate``, ``se`` or
    ``p_value``. Numbers are at full precision.
    """
    t, words = _TEXT[lang], _READING[lang]
    rows: list[dict[str, Any]] = []
    for sample_name, _ in _sections(record, lang):
        for key, _, h in _horizons(record, lang):
            cells = record[sample_name]["horizons"][key]
            for name, label in zip(_ROWS, words["rows"]):
                cell = cells[name]
                top = cells["concentration"].get(name, {})
                point = {} if cell["weak"] else {"estimate": cell["multiplier"], "se": cell["se"],
                                                 "lo": cell["lo"], "hi": cell["hi"]}
                rows.append({"sample": sample_name, "h": h, "row": name, "label": label, **point,
                             "f_eff": cell["f_eff"], "weak": cell["weak"],
                             "ar_lo": cell["ar_lo"], "ar_hi": cell["ar_hi"],
                             "ar_set_type": cell["ar_set_type"], "n_eff": cell.get("n_eff"),
                             "top_quarter": top.get("date"), "top_share": top.get("share")})
            test = cells["difference"]
            quoted = {} if not test["reliable"] else {"estimate": test["estimate"],
                                                      "se": test["se"], "p_value": test["p_value"]}
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
    notes = [latex_escape(t["table_note"])]
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
    lines = [f"% {t['title']}", f"% {t['source'].format(citation=record['data']['citation'])}"]
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
    """``multipliers.png``: linear, slack and no slack over h = 0..20, a row per sample."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    t, words = _TEXT[lang], _READING[lang]
    cutoff = float(record["specification"]["weak_cutoff"])
    sections = _sections(record, lang)
    hs = sorted(int(h) for h in paths["h"].unique())
    ticks = sorted({hs[0], *(h for _, _, h in _horizons(record, lang)), hs[-1]})
    fig = Figure(figsize=(11.0, 1.0 + 3.1 * len(sections)), dpi=120, facecolor=_SURFACE,
                 layout="constrained")
    FigureCanvasAgg(fig)
    off_axis = False
    for subfig, (name, heading) in zip(fig.subfigures(len(sections), 1, squeeze=False)[:, 0],
                                       sections):
        subfig.set_facecolor(_SURFACE)
        subfig.suptitle(heading, x=0.01, ha="left", fontsize=10.5, color=_INK)
        frame = paths.loc[paths["sample"] == name].sort_values("h")
        axes = subfig.subplots(1, len(_ROWS))
        for ax, row, title in zip(axes, _ROWS, words["rows"]):
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
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False,
               fontsize=8.5, labelcolor=_INK_2)
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


def run_brief(output: str | Path, *, exclude: str | None = None,
              lang: str = "en") -> dict[str, Any]:
    """Compute the brief (:func:`compute_brief`), then write its files (:func:`write_outputs`).

    Nothing is created or written unless the computation succeeds. Returns
    ``{"record", "paths", "manifest"}``.
    """
    _text(lang)
    record, paths = compute_brief(exclude=exclude)
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


def main(argv: Sequence[str] | None = None) -> int:
    """The command line: check the arguments, run the brief, print its summary in ``--lang``."""
    parser = argparse.ArgumentParser(
        prog="python -m puremacro.examples.first_brief_fiscal",
        description=("Fiscal policy brief: the US government spending multiplier at two and "
                     "four years, with and without slack, on Ramey and Zubairy's data."))
    parser.add_argument("--exclude", metavar="START:END", default=None,
                        help=("also estimate without this period, both quarters included, "
                              f"by Ramey and Zubairy's rule (World War II: {WWII})"))
    parser.add_argument("--output", metavar="DIR", type=Path, default=DEFAULT_OUTPUT,
                        help=f"output directory (default {DEFAULT_OUTPUT})")
    parser.add_argument("--lang", choices=("en", "es"), default="en",
                        help="language of the outputs (default en)")
    args = parser.parse_args(argv)
    if args.exclude is not None:          # checked before anything is computed or written
        try:
            exclude_period(load_brief_data(), parse_exclude(args.exclude))
        except _ExcludeError as exc:
            parser.error(exc.text(args.lang))
    study = run_brief(args.output, exclude=args.exclude, lang=args.lang)
    _echo(summary_text(study["record"], args.lang))
    _echo(_TEXT[args.lang]["written"].format(path=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
