"""OECD LFS sex × age × measure panel fetcher (SDMX-CSV).

The OECD's Labour Force Statistics ``DSD_LFS`` dataflows return
sex × age × measure × country × time observations in SDMX-CSV. This
module builds the dot-key, fetches via :func:`puremacro.fetch.sdmx.sdmx_get`,
and reshapes the response into a long-format panel keyed on
``(code, date, sex, age)`` with measures pivoted to columns and
the three standard rates pre-computed.

Public API
----------
- :func:`fetch_oecd_lfs_panel` — main entry point.

A local CSV/Excel fallback is supported via ``csv_path=`` so the
fetcher works offline against pre-downloaded SDMX dumps or a curated
``LABOR.xlsx`` (with the ``path::sheet`` syntax).

References
----------
SDMX-CSV format: https://sdmx.org/?page_id=4345
OECD SDMX:       https://sdmx.oecd.org/public/rest/
DSD_LFS dataflows: see https://sdmx.oecd.org/public/rest/dataflow/OECD.SDD.TPS/all/latest
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .sdmx import sdmx_get


# Dot-key skeletons per dataflow. Each tuple lists the dimensions in
# the order they appear in the SDMX key, after ``FREQ`` and before the
# (typically empty) trailing slots. The values inserted by the builder
# come from the ``_build_lfs_key`` arguments.
#
# DSD_LFS@DF_IALFS_INDIC:
#   FREQ.REF_AREA.MEASURE.UNIT_MEASURE.TRANSFORMATION.ADJUSTMENT.SEX.AGE.ACTIVITY
_DATAFLOW_KEY_TEMPLATES: dict[str, tuple[str, ...]] = {
    "DSD_LFS@DF_IALFS_INDIC": (
        "freq", "area", "measure", "unit", "transform", "adjust",
        "sex", "age", "activity",
    ),
    "DSD_LFS@DF_IALFS_EMP_Q": (
        "freq", "area", "measure", "unit", "transform", "adjust",
        "sex", "age", "activity",
    ),
    "DSD_LFS@DF_IALFS_UNE_Q": (
        "freq", "area", "measure", "unit", "transform", "adjust",
        "sex", "age", "activity",
    ),
    "DSD_LFS@DF_IALFS_EMP_BY_AR": (
        "freq", "area", "measure", "unit", "transform", "adjust",
        "sex", "age", "activity",
    ),
}


def _build_lfs_key(
    *,
    dataflow: str,
    countries: list[str] | None,
    sexes: tuple[str, ...],
    ages: tuple[str, ...],
    measures: tuple[str, ...],
    activities: tuple[str, ...] | None,
    frequency: str,
    adjustment: str,
) -> str:
    """Assemble the SDMX dot-key for an OECD LFS dataflow.

    Lists collapse to ``+``-joined strings; ``None`` becomes empty (which
    SDMX interprets as "all values for that dimension").
    """
    if dataflow not in _DATAFLOW_KEY_TEMPLATES:
        raise ValueError(
            f"dataflow {dataflow!r} not in known LFS templates: "
            f"{sorted(_DATAFLOW_KEY_TEMPLATES)}"
        )
    template = _DATAFLOW_KEY_TEMPLATES[dataflow]

    def _join(xs):
        if xs is None:
            return ""
        return "+".join(xs)

    slots = {
        "freq":      frequency,
        "area":      _join(countries),
        "measure":   _join(measures),
        "unit":      "",
        "transform": "",
        "adjust":    adjustment,
        "sex":       _join(sexes),
        "age":       _join(ages),
        "activity":  _join(activities),
    }
    return ".".join(slots[name] for name in template)


# Map raw OECD MEASURE codes → lowercased canonical column names.
_MEASURE_RENAME: dict[str, str] = {
    "WAP":  "wa",
    "LF":   "lf",
    "EMP":  "emp",
    "UNE":  "une",
    "IPOP": "ina",
}


def _parse_time_period(s: pd.Series) -> pd.Series:
    """Parse OECD TIME_PERIOD strings ('2020-01', '2020-Q1', '2020') to first-of-period datetimes."""
    s = s.astype(str)
    is_q = s.str.contains("Q")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out.loc[~is_q] = pd.to_datetime(s[~is_q], errors="coerce")
    out.loc[is_q] = pd.PeriodIndex(s[is_q], freq="Q").to_timestamp()
    return out


# Synthesis rules: target_age <- signed sum of source_age levels.
# Order matters: later rules may reference earlier synthesized targets
# (e.g. Y55T64 uses Y25T54).
_AGE_SYNTHESIS_RULES: list[tuple[str, list[tuple[str, str]]]] = [
    ("Y25T54", [("+", "25-34"), ("+", "35-44"), ("+", "45-54")]),
    ("Y15T24", [("+", "15-64"), ("-", "25+"),   ("+", "65+")]),
    # Fallback for países that report only 15+, 25+ (most of Europe in LABOR.xlsx).
    # Y15T24 = Y_GE15 - 25+ is exact since 15+ = (15-24) + 25+.
    # Note: _xlsx_to_sdmx_long maps "15+" → "Y_GE15", so the source is Y_GE15 here.
    ("Y15T24", [("+", "Y_GE15"), ("-", "25+")]),
    ("Y55T64", [("+", "25+"),   ("-", "Y25T54"), ("-", "65+")]),
]

_LEVEL_COLS: tuple[str, ...] = ("wa", "lf", "emp", "une", "ina")


def _synthesize_age_aggregates(wide: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic SDMX age cells derived from EU-style 5-year bands.

    Each rule in :data:`_AGE_SYNTHESIS_RULES` fires per ``(code, date, sex)``
    cell when (a) the target age is absent and (b) every source age is
    present with non-NaN level data. Levels are summed with the signs
    given by the rule; rates are not derived here (that happens later
    in :func:`_reshape_lfs_csv`).

    Rules are applied sequentially, so later rules can reference age
    cells synthesized by earlier rules (e.g. ``Y55T64`` is computed
    from a ``Y25T54`` that may itself have just been synthesized).
    """
    if wide.empty:
        return wide

    out = wide.copy()
    for target_age, parts in _AGE_SYNTHESIS_RULES:
        signs = {age: (1 if sign == "+" else -1) for sign, age in parts}
        source_ages = list(signs)

        # Wide pivot per cell: rows=(code,date,sex), columns=age, values=each level.
        relevant = out[out["age"].isin([target_age, *source_ages])]
        if relevant.empty:
            continue

        # Build a per-(cell, age) lookup keyed for fast access.
        keyed = relevant.set_index(["code", "date", "sex", "age"])
        cells = relevant[["code", "date", "sex"]].drop_duplicates()

        new_rows = []
        for code, date, sex in cells.itertuples(index=False, name=None):
            try:
                ages_here = set(keyed.xs((code, date, sex), level=("code", "date", "sex")).index)
            except KeyError:
                continue
            if target_age in ages_here:
                continue
            if not set(source_ages).issubset(ages_here):
                continue

            row = {"code": code, "date": date, "sex": sex, "age": target_age}
            ok = True
            for col in _LEVEL_COLS:
                total = 0.0
                for age in source_ages:
                    val = keyed.at[(code, date, sex, age), col]
                    if pd.isna(val):
                        ok = False
                        break
                    total += signs[age] * val
                if not ok:
                    break
                row[col] = total
            if ok:
                new_rows.append(row)

        if new_rows:
            out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True)

    return out


def _reshape_lfs_csv(raw: pd.DataFrame, *, frequency: str) -> pd.DataFrame:
    """Reshape an SDMX-CSV LFS response into a canonical long panel.

    Returns columns ``code, date, sex, age, wa, lf, emp, une, ina,
    urate, epop, prate``. Rows where FREQ != ``frequency`` are dropped.
    """
    df = raw.loc[raw["FREQ"] == frequency, [
        "REF_AREA", "TIME_PERIOD", "SEX", "AGE", "MEASURE", "OBS_VALUE",
    ]].copy()
    df = df.rename(columns={
        "REF_AREA":    "code",
        "TIME_PERIOD": "date",
        "SEX":         "sex",
        "AGE":         "age",
    })
    df["date"] = _parse_time_period(df["date"])
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")

    wide = (
        df.pivot_table(
            index=["code", "date", "sex", "age"],
            columns="MEASURE",
            values="OBS_VALUE",
            aggfunc="first",
        )
        .rename(columns=_MEASURE_RENAME)
        .reset_index()
    )
    wide.columns.name = None

    # Ensure all canonical level columns exist (missing → NaN).
    for col in ("wa", "lf", "emp", "une", "ina"):
        if col not in wide.columns:
            wide[col] = np.nan

    # Synthesize SDMX age aggregates from 5-year bands where possible.
    wide = _synthesize_age_aggregates(wide)

    # Synthesize ina = wa - lf where IPOP-derived ina is NaN but wa, lf are present.
    needs_synth = wide["ina"].isna() & wide["wa"].notna() & wide["lf"].notna()
    wide.loc[needs_synth, "ina"] = wide.loc[needs_synth, "wa"] - wide.loc[needs_synth, "lf"]

    # Derive rates with explicit positive-denominator masks (no try/except).
    def _rate(num_col: str, den_col: str) -> pd.Series:
        num = wide[num_col]
        den = wide[den_col]
        mask = den.notna() & (den > 0)
        out = pd.Series(np.nan, index=wide.index, dtype="float64")
        out.loc[mask] = num[mask] / den[mask]
        return out

    wide["urate"] = _rate("une", "lf")
    wide["epop"]  = _rate("emp", "wa")
    wide["prate"] = _rate("lf",  "wa")

    keep = ["code", "date", "sex", "age",
            "wa", "lf", "emp", "une", "ina",
            "urate", "epop", "prate"]
    out = wide[keep].sort_values(["code", "date", "sex", "age"]).reset_index(drop=True)
    return out


# Sex label normalization for the LABOR.xlsx adapter.
_XLSX_SEX_MAP: dict[str, str] = {
    "Total": "_T", "T": "_T", "_T": "_T",
    "Male":  "M",  "M": "M",
    "Female": "F", "F": "F",
}

# LABOR.xlsx-style age labels → OECD LFS codes.
_XLSX_AGE_MAP: dict[str, str] = {
    "15+":   "Y_GE15",
    "15-24": "Y15T24",
    "25-54": "Y25T54",
    "55-64": "Y55T64",
    "55+":   "Y_GE55",
}


def _read_local_input(csv_path: str | Path) -> pd.DataFrame:
    """Read a local SDMX-CSV file or a LABOR.xlsx ``path::sheet`` spec.

    For ``.xlsx`` inputs, normalize column names and sex/age labels to
    the SDMX convention so the downstream :func:`_reshape_lfs_csv`
    sees a uniform schema.
    """
    s = str(csv_path)
    if "::" in s:
        path_part, sheet = s.split("::", 1)
        path = Path(path_part)
        if not path.exists():
            raise FileNotFoundError(f"xlsx not found: {path}")
        try:
            raw = pd.read_excel(path, sheet_name=sheet)
        except ValueError as exc:
            # pandas raises ValueError for missing sheet — re-raise as KeyError
            # so callers can distinguish "wrong file" from "wrong sheet".
            raise KeyError(
                f"sheet {sheet!r} not in {path}; "
                f"available: {pd.ExcelFile(path).sheet_names}"
            ) from exc
        return _xlsx_to_sdmx_long(raw)
    return pd.read_csv(s)


def _xlsx_to_sdmx_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a LABOR.xlsx OECD-AGE-M sheet to SDMX-CSV-shaped long format."""
    df = df.copy()
    rename = {
        "Code":   "REF_AREA",
        "Period": "TIME_PERIOD",
        "Sex":    "SEX",
        "Age":    "AGE",
    }
    df = df.rename(columns=rename)

    # Normalize sex / age labels.
    df["SEX"] = df["SEX"].map(_XLSX_SEX_MAP).fillna(df["SEX"])
    df["AGE"] = df["AGE"].map(_XLSX_AGE_MAP).fillna(df["AGE"])

    # Identify level columns: anything not in the index set.
    index_cols = ["REF_AREA", "TIME_PERIOD", "SEX", "AGE"]
    level_cols = [c for c in df.columns if c not in index_cols]

    long = df.melt(
        id_vars=index_cols,
        value_vars=level_cols,
        var_name="MEASURE",
        value_name="OBS_VALUE",
    )
    # Map LABOR.xlsx lowercase measure names → SDMX uppercase.
    measure_up = {"emp": "EMP", "lf": "LF", "une": "UNE",
                  "wa": "WAP", "ina": "IPOP", "olf": "IPOP"}
    long["MEASURE"] = long["MEASURE"].map(measure_up).fillna(long["MEASURE"].str.upper())
    long["FREQ"] = "M"
    return long


def fetch_oecd_lfs_panel(
    *,
    dataflow: str = "DSD_LFS@DF_IALFS_INDIC",
    countries: list[str] | None = None,
    sexes:     tuple[str, ...] = ("_T", "M", "F"),
    ages:      tuple[str, ...] = ("Y_GE15", "Y15T24", "Y25T54", "Y55T64"),
    measures:  tuple[str, ...] = ("WAP", "LF", "EMP", "UNE", "IPOP"),
    activities: tuple[str, ...] | None = None,
    frequency: str = "M",
    adjustment: str = "Y",
    csv_path:  str | Path | None = None,
    cache_path: str | Path | None = None,
    refresh:   bool = False,
) -> pd.DataFrame:
    """Fetch an OECD LFS sex × age × measure panel as a long DataFrame.

    Parameters
    ----------
    dataflow : str
        OECD SDMX dataflow ID. Default ``"DSD_LFS@DF_IALFS_INDIC"`` returns
        infra-annual labour-stats indicators (WAP, LF, EMP, UNE, IPOP).
        Use ``"DSD_LFS@DF_IALFS_EMP_BY_AR"`` for sex × ISIC sector
        employment.
    countries : list[str] | None
        ISO-3 country codes; ``None`` returns all reporters.
    sexes, ages, measures, activities : tuple[str, ...] | None
        OECD enum codes. ``None`` selects all.
    frequency : str
        ``"M"`` or ``"Q"`` (or ``"A"`` for annual). Mixed-frequency rows
        in the SDMX response are filtered to this value.
    adjustment : str
        ``"Y"`` (calendar + seasonally adjusted, default) or ``"N"`` (NSA).
    csv_path : str | Path | None
        Local SDMX-CSV file or LABOR.xlsx ``"path::sheet"`` spec. When
        set and the file exists, network fetch is skipped.
    cache_path : str | Path | None
        Parquet cache. If exists and ``refresh=False``, returned without
        any reshape work.
    refresh : bool
        Force re-fetch (ignore cache).

    Returns
    -------
    pd.DataFrame
        Long panel with columns ``code, date, sex, age, wa, lf, emp,
        une, ina, urate, epop, prate``. Sorted by
        ``(code, date, sex, age)``.
    """
    # 1. Cache fast path.
    if cache_path is not None and not refresh:
        cp = Path(cache_path)
        if cp.exists():
            return pd.read_parquet(cp)

    # 2. Local file fast path.
    if csv_path is not None:
        raw = _read_local_input(csv_path)
    else:
        # 3. Live SDMX.
        key = _build_lfs_key(
            dataflow=dataflow, countries=countries, sexes=sexes, ages=ages,
            measures=measures, activities=activities,
            frequency=frequency, adjustment=adjustment,
        )
        raw = sdmx_get(provider="oecd", dataflow=dataflow, key=key)

    out = _reshape_lfs_csv(raw, frequency=frequency)

    # 4. Cache write.
    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cp, index=False)

    return out


def fetch_sectoral_panel_union(panels: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine canonical-schema sectoral panels with priority-first column merge.

    For each ``(code, date, sex, eco, column)`` cell, the first listed
    panel's value wins **if non-NaN**; otherwise the next panel fills in.
    Mirrors :func:`fetch_labor_panel_union` for the sectoral 6-col schema:
    ``[code, date, sex, eco, emp, hours_worked]``.

    Parameters
    ----------
    panels : list[pd.DataFrame]
        Frames to combine. Earlier frames win on duplicate keys
        — column-wise, NaN-aware.

    Returns
    -------
    pd.DataFrame
        Combined, sorted on ``(code, date, sex, eco)`` with reset index.
    """
    canonical = ["code", "date", "sex", "eco", "emp", "hours_worked"]
    index_cols = ["code", "date", "sex", "eco"]
    if not panels:
        return pd.DataFrame(columns=canonical)
    result = panels[0][canonical].set_index(index_cols)
    for p in panels[1:]:
        result = result.combine_first(p[canonical].set_index(index_cols))
    return (result.reset_index()[canonical]
            .sort_values(index_cols)
            .reset_index(drop=True))


def fetch_labor_panel_union(panels: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine canonical-schema labor panels with priority-first column merge.

    For each ``(code, date, sex, age, column)`` cell, the first listed
    panel's value wins **if non-NaN**; otherwise the next panel fills in.
    This preserves data from a supplementary source (e.g. Eurostat
    ``urate``) on cells where the higher-priority source (e.g. LABOR.xlsx)
    has a NaN.

    The canonical schema is what :func:`fetch_oecd_lfs_panel` and
    :func:`puremacro.fetch.labor_eurostat.fetch_eurostat_lfs_panel`
    return: ``[code, date, sex, age, wa, lf, emp, une, ina, urate,
    epop, prate]``.

    Parameters
    ----------
    panels : list[pd.DataFrame]
        Frames to combine. Earlier frames win on duplicate keys
        — column-wise, NaN-aware.

    Returns
    -------
    pd.DataFrame
        Combined, sorted on ``(code, date, sex, age)`` with reset index.
    """
    canonical = ["code", "date", "sex", "age",
                 "wa", "lf", "emp", "une", "ina",
                 "urate", "epop", "prate"]
    index_cols = ["code", "date", "sex", "age"]
    if not panels:
        return pd.DataFrame(columns=canonical)
    # Set common (code, date, sex, age) index on each panel, keeping only
    # canonical columns. combine_first does NaN-aware fill: result keeps
    # `self`'s non-NaN values; where `self` is NaN, takes the value from
    # `other`. Iterating over panels[1:] applies them in priority order.
    result = panels[0][canonical].set_index(index_cols)
    for p in panels[1:]:
        result = result.combine_first(p[canonical].set_index(index_cols))
    return (result.reset_index()[canonical]
            .sort_values(index_cols)
            .reset_index(drop=True))
