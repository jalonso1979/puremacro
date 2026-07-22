"""Eurostat job-vacancy statistics (SDMX-CSV, quarterly).

Dataflow ``jvs_q_nace2``: job vacancy rate (``JVR``, percent), number
of job vacancies (``JOBVAC``, count) and occupied posts (``JOBOCC``)
by NACE Rev. 2 section aggregate, size class and seasonal adjustment.
DSD dimension order (verified live 2026-07-21):
``freq.s_adj.nace_r2.sizeclas.indic_em.geo``.

The European member of the vacancy pair with :mod:`puremacro.fetch.
jolts` — e.g. cross-country Beveridge curves against the LFS urate
panels already in :mod:`puremacro.fetch.labor_eurostat`.

Coverage notes (honest): country participation and NACE/size detail
vary; ``s_adj='SA'`` is not published for every country (rows simply
absent — no silent NSA substitution here; fetch both and compare
coverage if it matters). Geo is normalized ISO-2 → ISO-3 with the same
map as the LFS fetchers; EU/EA aggregates are dropped unless
``include_aggregates=True`` (then kept under their Eurostat codes,
e.g. ``EA20``).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .labor_eurostat import _EUROSTAT_GEO_TO_ISO3
from .sdmx import sdmx_get

_DATAFLOW = "jvs_q_nace2"
_INDICATORS = {"JVR": "jvr", "JOBVAC": "jobvac", "JOBOCC": "jobocc"}

__all__ = ["fetch_eurostat_vacancies"]


def fetch_eurostat_vacancies(
    *,
    indicator: str = "JVR",
    nace: str = "B-S",
    sizeclas: str = "TOTAL",
    s_adj: str = "SA",
    include_aggregates: bool = False,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Quarterly Eurostat vacancy panel, tidy.

    Parameters
    ----------
    indicator : 'JVR' (vacancy rate, %), 'JOBVAC' (vacancies, count)
        or 'JOBOCC' (occupied posts).
    nace : NACE Rev. 2 aggregate (e.g. 'B-S' business economy, 'A-S',
        'C', ...). Passed through to the SDMX key.
    sizeclas : 'TOTAL' or 'GE10' (establishments with >= 10 employees).
    s_adj : 'SA' or 'NSA'. No silent substitution: countries without
        the requested adjustment are simply absent.
    include_aggregates : keep EU/EA aggregate rows (Eurostat codes) in
        addition to countries; default drops them.
    csv_path : optional pre-downloaded SDMX-CSV file (hermetic tests /
        offline use); the SDMX key filters are then applied locally.

    Returns
    -------
    DataFrame ``[code, date, <indicator lowercase>]`` — ``code`` ISO-3
    for countries (Eurostat code for kept aggregates), ``date`` the
    quarter-start Timestamp.
    """
    if indicator not in _INDICATORS:
        raise ValueError(
            f"fetch_eurostat_vacancies: indicator must be one of "
            f"{sorted(_INDICATORS)}, got {indicator!r}")
    key = f"Q.{s_adj}.{nace}.{sizeclas}.{indicator}."
    raw = sdmx_get(provider="eurostat", dataflow=_DATAFLOW, key=key,
                   csv_path=csv_path)
    df = raw.copy()
    # csv_path mode returns the whole file: apply the key filters here.
    for col, want in (("s_adj", s_adj), ("nace_r2", nace),
                      ("sizeclas", sizeclas), ("indic_em", indicator)):
        if col in df.columns:
            df = df[df[col] == want]
    if df.empty:
        raise ValueError(
            f"fetch_eurostat_vacancies: no rows for indicator="
            f"{indicator!r}, nace={nace!r}, sizeclas={sizeclas!r}, "
            f"s_adj={s_adj!r} — combination not published?")

    geo = df["geo"].astype(str)
    code = geo.map(_EUROSTAT_GEO_TO_ISO3)
    if include_aggregates:
        code = code.fillna(geo)
    out = pd.DataFrame({
        "code": code,
        "date": pd.PeriodIndex(df["TIME_PERIOD"].astype(str), freq="Q")
                  .to_timestamp(how="start"),
        _INDICATORS[indicator]: pd.to_numeric(df["OBS_VALUE"],
                                              errors="coerce"),
    })
    out = out.dropna(subset=["code", _INDICATORS[indicator]])
    return out.sort_values(["code", "date"], ignore_index=True)
