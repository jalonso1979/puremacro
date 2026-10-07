"""Eurostat job-vacancy statistics (SDMX-CSV, quarterly).

Dataflow ``jvs_q_r21``: job vacancy rate (``JVR``, percent), number
of job vacancies (``JOBVAC``, count) and occupied posts (``JOBOCC``)
by NACE Rev. 2.1 section aggregate, size class and seasonal adjustment.
DSD dimension order (verified live 2026-10-06):
``freq.nace_r2_1.sizeclas.s_adj.indic_em.geo``.

The NACE Rev. 2 flow ``jvs_q_nace2`` this module read until October 2026
is frozen at 2025Q4. Its business-economy aggregate ``B-S`` becomes
``B-T`` in the Rev. 2.1 flow (over the overlap the two agree to within
0.4 percentage points, 90 percent of quarters identical); ``nace="B-S"``
is accepted as an alias for ``B-T`` with a warning.

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

import warnings

import pandas as pd

from .labor_eurostat import _EUROSTAT_GEO_TO_ISO3
from .sdmx import sdmx_get

_DATAFLOW = "jvs_q_r21"
#: NACE Rev. 2 aggregates of the frozen flow -> their Rev. 2.1 successors.
_NACE_ALIASES = {"B-S": "B-T", "A-S": "A-T"}
_INDICATORS = {"JVR": "jvr", "JOBVAC": "jobvac", "JOBOCC": "jobocc"}

__all__ = ["fetch_eurostat_vacancies"]


def fetch_eurostat_vacancies(
    *,
    indicator: str = "JVR",
    nace: str = "B-T",
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
    nace : NACE Rev. 2.1 aggregate (e.g. 'B-T' business economy, 'A-T',
        'B-O', 'C', ...). Passed through to the SDMX key; the Rev. 2 codes
        'B-S' and 'A-S' are translated to 'B-T' and 'A-T' with a warning.
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
    if nace in _NACE_ALIASES:
        warnings.warn(
            f"fetch_eurostat_vacancies: NACE Rev. 2 code {nace!r} is "
            f"{_NACE_ALIASES[nace]!r} in {_DATAFLOW} (NACE Rev. 2.1)",
            stacklevel=2)
        nace = _NACE_ALIASES[nace]
    key = f"Q.{nace}.{sizeclas}.{s_adj}.{indicator}."
    raw = sdmx_get(provider="eurostat", dataflow=_DATAFLOW, key=key,
                   csv_path=csv_path)
    df = raw.copy()
    # csv_path mode returns the whole file: apply the key filters here. A
    # file saved from the old flow names the activity column nace_r2.
    if "nace_r2" in df.columns and "nace_r2_1" not in df.columns:
        inv = {v: k for k, v in _NACE_ALIASES.items()}
        df = df[df["nace_r2"].isin([nace, inv.get(nace, nace)])]
    for col, want in (("s_adj", s_adj), ("nace_r2_1", nace),
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
    out = out.sort_values(["code", "date"], ignore_index=True)
    # The same attrs keys as the panel builders, so a caller that freezes the
    # frame can record what it is and which release it came from.
    release = None
    if "LAST UPDATE" in df.columns:
        stamps = df["LAST UPDATE"].dropna().astype(str)
        release = max(stamps) if len(stamps) else None
    name = _INDICATORS[indicator]
    units = {"JVR": "percent of occupied plus vacant posts",
             "JOBVAC": "vacant posts (number)", "JOBOCC": "occupied posts (number)"}[indicator]
    out.attrs.update(
        meta=({"variable": name, "units": units, "sa": s_adj, "flow": _DATAFLOW,
               "key": key, "n": int(len(out)), "n_codes": int(out["code"].nunique()),
               "first": str(out["date"].min().date()) if len(out) else None,
               "last": str(out["date"].max().date()) if len(out) else None},),
        source=f"Eurostat {_DATAFLOW} {key}",
        release=release,
        fetched_at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        missing=(),
    )
    return out
