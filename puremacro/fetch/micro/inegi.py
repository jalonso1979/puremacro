"""INEGI ENIGH: Mexico's household income and expenditure survey.

The Encuesta Nacional de Ingresos y Gastos de los Hogares (ENIGH) is
Mexico's reference source for the income distribution, poverty
measurement (CONEVAL) and household consumption. INEGI publishes the
microdata openly, without registration, as zipped CSV files per table and
wave. This module downloads one table through the shared HTTP cache and
returns a :class:`MicroFrame` carrying the survey's sampling design.

WAVES AND TABLES
----------------
The *nueva serie* (``ns``) runs from 2016 every two years. Each wave
publishes several linked tables; the useful ones for most work are

* ``concentradohogar`` — one row per household, INEGI's own summary
  variables (``ing_cor`` current income, ``gasto_mon`` monetary
  spending, ``tot_integ`` household size, ...). The default.
* ``poblacion`` — one row per person (``numren``), demographics,
  schooling, health access.
* ``hogares``, ``viviendas``, ``ingresos``, ``gastoshogar``, ... — the
  detailed tables, joined on ``folioviv`` + ``foliohog`` (+ ``numren``).

Every table carries the design variables used here.

DESIGN
------
ENIGH is a stratified two-stage cluster sample. INEGI publishes no
replicate weights; it publishes the expansion factor ``factor``, the
design stratum ``est_dis`` and the primary sampling unit ``upm``. The
returned frame therefore uses ``method="taylor"``: standard errors by
linearisation with the ultimate-cluster approximation, the method
INEGI's own precision tables use. Estimates for subgroups (``by=``) are
computed as domains over the full design.

MONEY
-----
Income and spending in ``concentradohogar`` are *quarterly* pesos at
prices of August of the survey year (INEGI deflates each month's record
to that reference month). Divide by 3 for monthly figures; compare waves
only after re-deflating.

Identifiers (``folioviv``, ``foliohog``, ``numren``, ``ubica_geo``,
``est_dis``, ``upm``, ``entidad``) are read as strings so that leading
zeros survive; everything else that is fully numeric becomes numeric.

Files (unverified from this module's test environment; checked by the
opt-in live test)::

    https://www.inegi.org.mx/contenidos/programas/enigh/nc/{YEAR}/microdatos/enigh{YEAR}_ns_{TABLE}_csv.zip
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd

from .. import _http
from ._design import MicroFrame, SurveyDesign

FIRST_WAVE = 2016
_ROOT = "https://www.inegi.org.mx/contenidos/programas/enigh/nc/"
TABLES = ("concentradohogar", "hogares", "poblacion", "viviendas",
          "ingresos", "gastoshogar", "gastospersona", "trabajos",
          "agro", "noagro", "erogaciones", "gastotarjetas")
ID_COLUMNS = ("folioviv", "foliohog", "numren", "ubica_geo", "est_dis",
              "upm", "entidad")
_UNIT = {"poblacion": "person", "trabajos": "job", "viviendas": "dwelling"}


def available_years(through: int = 2024) -> list[int]:
    """Waves of the nueva serie (biennial from 2016) up to ``through``."""
    return list(range(FIRST_WAVE, through + 1, 2))


def enigh_url(year: int, table: str = "concentradohogar") -> str:
    """Zip URL of one ENIGH table for one wave."""
    if year < FIRST_WAVE or (year - FIRST_WAVE) % 2:
        raise ValueError(f"no ENIGH nueva serie wave in {year}; waves are "
                         f"{FIRST_WAVE}, {FIRST_WAVE + 2}, ...")
    if table not in TABLES:
        raise ValueError(f"unknown ENIGH table {table!r}; one of {TABLES}")
    return f"{_ROOT}{year}/microdatos/enigh{year}_ns_{table}_csv.zip"


def _pick_csv(names: list[str], table: str) -> str:
    """The data CSV inside an INEGI archive.

    Archives also ship dictionaries and code catalogues as CSV, so prefer
    a file under ``conjunto_de_datos/`` named after the table.
    """
    csvs = [n for n in names if n.lower().endswith(".csv")]
    data = [n for n in csvs
            if "conjunto_de_datos/" in n.lower().replace("\\", "/")
            and table in n.lower().rsplit("/", 1)[-1]]
    if len(data) == 1:
        return data[0]
    named = [n for n in csvs if n.lower().rsplit("/", 1)[-1].startswith(
        ("conjunto_de_datos_" + table, table))]
    if len(named) == 1:
        return named[0]
    if len(csvs) == 1:
        return csvs[0]
    raise ValueError(f"cannot tell which CSV holds the {table!r} data: {csvs}")


def read_table(payload: bytes, table: str = "concentradohogar") -> pd.DataFrame:
    """Parse one ENIGH zip into a DataFrame (lower-case column names)."""
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        raw = zf.read(_pick_csv(zf.namelist(), table))
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False,
                     na_values=["", " "])
    df.columns = [c.strip().lower() for c in df.columns]
    for c in df.columns:
        if c in ID_COLUMNS:
            continue
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().sum() == df[c].notna().sum():
            df[c] = conv
    return df


def fetch_enigh(
    year: int = 2022,
    variables=None,
    *,
    table: str = "concentradohogar",
    refresh: bool = False,
    timeout: int = 300,
) -> MicroFrame:
    """One ENIGH table for one wave, with its stratified cluster design.

    Parameters
    ----------
    year : wave (2016, 2018, ..., 2024).
    variables : columns to keep (case-insensitive); all when ``None``.
        Identifiers and design columns are always kept.
    table : which table; ``"concentradohogar"`` (households, summary
        variables) by default, ``"poblacion"`` for persons.

    Examples
    --------
    >>> enigh = fetch_enigh(2022, ["ing_cor", "tot_integ"])     # doctest: +SKIP
    >>> enigh.mean("ing_cor")                                   # doctest: +SKIP
    """
    url = enigh_url(year, table)
    df = read_table(_http.cached_get(url, refresh=refresh, timeout=timeout), table)
    design = SurveyDesign.enigh()
    if variables is not None:
        wanted = [v.lower() for v in variables]
        absent = [v for v in wanted if v not in df.columns]
        if absent:
            raise KeyError(f"not in ENIGH {year} {table}: {absent}")
        keep = [c for c in ID_COLUMNS if c in df.columns]
        keep += [c for c in design.columns() if c not in keep]
        keep += [c for c in wanted if c not in keep]
        df = df[keep]
    return MicroFrame(data=df.reset_index(drop=True), design=design,
                      unit=_UNIT.get(table, "household"), source="inegi.enigh",
                      vintage=f"{year} {table}", query=url)


__all__ = ["fetch_enigh", "enigh_url", "read_table", "available_years",
           "TABLES", "ID_COLUMNS"]
