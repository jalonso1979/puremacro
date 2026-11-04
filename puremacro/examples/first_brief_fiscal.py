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
and the paths over h = 0..20.

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
JSON, y las trayectorias en h = 0..20.
"""
from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import puremacro

from ..lp import LPResult
from ..lp._common import check_regular_index, cumulative_windows
from ..lp.iv import lp_iv
from ..lp.state_dep import lp_state_dep_iv
from ._brief_record import environment, sha256, write_json_atomic
from .ramey_zubairy_2018_multipliers import RZ_CONTROLS, load_rz_data

__all__ = [
    "HORIZONS", "HEADLINE_HORIZONS", "ALPHA", "WWII", "DATA_FILE", "CITATION",
    "CONCENTRATION_REPORT_SHARE", "parse_exclude", "load_brief_data", "exclude_period",
    "estimate", "concentration", "compute_brief",
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

#: Warnings issued while computing the last brief, as ``"<sample>: <Category>:
#: <message>"``, for the run's manifest. Every :func:`compute_brief` call
#: replaces them.
_LAST_WARNINGS: list[str] = []


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
        raise ValueError(f"exclude must be START:END in quarters, for example {WWII}; got {text!r}")
    start = pd.Period(f"{match[1]}Q{match[2]}", freq="Q")
    end = pd.Period(f"{match[3]}Q{match[4]}", freq="Q")
    if start > end:
        raise ValueError(f"exclude starts after it ends ({start} > {end}); write START:END, "
                         f"for example {WWII}")
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
        raise ValueError(f"the period {start}:{end} does not overlap the sample, "
                         f"{quarters[0]} to {quarters[-1]}")
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
    is done, or has failed.
    """
    full = load_brief_data()
    frames = {"full": full}
    period = None
    if exclude is not None:
        period = parse_exclude(exclude)
        frames["experiment"] = exclude_period(full, period)

    _LAST_WARNINGS.clear()
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
    return _json_ready(record), paths
