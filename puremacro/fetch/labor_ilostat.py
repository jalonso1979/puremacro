"""ILOSTAT sex × age and sex × ISIC sectoral panel fetchers.

Pulls from ILOSTAT via the ``ilostat`` SDMX provider. Two public
entry points:
  - :func:`fetch_ilostat_lfs_panel`       — sex × age, canonical 12-col schema
  - :func:`fetch_ilostat_sectoral_panel`  — sex × ISIC, NEW 6-col schema

ILOSTAT's quarterly data is published NSA. When ``adjustment="SA"``
and the target frequency is quarterly, X13-ARIMA is applied per cell
via :mod:`puremacro.fetch._seasonal`.

References
----------
ILOSTAT SDMX:    https://sdmx.ilo.org/rest/
SDMX-CSV format: https://sdmx.org/?page_id=4345
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .sdmx import sdmx_get


# Aggregate codes ILOSTAT publishes that are NOT countries — drop at reshape.
_ILO_AGGREGATE_CODES: frozenset[str] = frozenset({
    "WORLD", "X80", "X81",
    "LAC_LIC", "LAC", "LMC", "LIC", "HIC", "UMC",
    "OECD_TOT", "EU27_2020", "EU28", "EA20", "EA19",
    "AFR", "AMR", "ARB", "ASI", "EUR", "OCE",
    "DEV", "DVG", "LDC", "EME", "EAS",
})

# ILOSTAT MEASURE codes → canonical 12-col schema column names.
# The live API appends _NB or _RT suffixes to each measure code.
_LFS_MEASURE_RENAME: dict[str, str] = {
    # Live API codes (from NSI Web Service v8, confirmed via live pull)
    "UNE_DEAP_RT":  "urate",   # unemployment rate = unemployed / labor force
    "EMP_DWAP_RT":  "epop",    # employment-to-population rate
    "EAP_DWAP_RT":  "prate",   # LFP rate = labor force / working-age pop
    "EMP_TEMP_NB":  "emp",     # employed persons
    "UNE_TUNE_NB":  "une",     # unemployed persons
    "EAP_TEAP_NB":  "lf",      # labor force (= emp + une)
    "POP_XWAP_NB":  "wa",      # working-age population
    # Legacy codes (used in synthetic test fixtures and older ILO exports)
    "UNE_RT":       "urate",
    "EPOP_RT":      "epop",
    "LFEPOP_RT":    "prate",
    "EMP_TEMP":     "emp",
    "UNE_TUNE":     "une",
    "EAP_TEAP":     "lf",
    "WAP_TEMP":     "wa",
}

# ILOSTAT AGE codes → canonical OECD-style age codes.
# The live API uses AGE_YTHADULT_* codes (not AGE_AGGREGATE_*).
_LFS_AGE_RENAME: dict[str, str] = {
    # Primary mapping (live API codes)
    "AGE_YTHADULT_YGE15":    "Y_GE15",
    "AGE_YTHADULT_Y15-24":   "Y15T24",
    "AGE_YTHADULT_Y25-54":   "Y25T54",
    "AGE_YTHADULT_YGE25":    "Y_GE25",
    "AGE_YTHADULT_Y55-64":   "Y55T64",
    # Legacy mapping (older ILO data and synthetic fixtures)
    "AGE_AGGREGATE_TOTAL":   "Y_GE15",
    "AGE_AGGREGATE_Y_GE15":  "Y_GE15",
    "AGE_AGGREGATE_Y15-24":  "Y15T24",
    "AGE_5YRBANDS_Y25-54":   "Y25T54",
    "AGE_5YRBANDS_Y55-64":   "Y55T64",
}

# Rates that ILOSTAT publishes in percent; reshape divides by 100 so output
# matches OECD/Eurostat convention (proportions in [0, 1]).
_RATE_MEASURE_KEYS: frozenset[str] = frozenset({"urate", "epop", "prate"})


def _build_ilostat_lfs_key(
    *,
    countries: list[str] | None,
    sexes: tuple[str, ...],
    ages: tuple[str, ...],
    measures: tuple[str, ...],
    frequency: str,
) -> str:
    """Assemble an ILOSTAT SDMX dot-key for an LFS dataflow.

    Dimension order: ``REF_AREA.FREQ.MEASURE.SEX.AGE``.

    The live ILOSTAT NSI Web Service v8 uses this order as confirmed by
    inspecting the DataStructure definition at
    https://sdmx.ilo.org/rest/datastructure/ILO/<DSD_ID>.
    """
    def _join(xs):
        if xs is None:
            return ""
        return "+".join(xs)

    return ".".join([
        _join(countries),
        frequency,
        _join(measures),
        _join(sexes),
        _join(ages),
    ])


def _parse_ilostat_time_period(s: pd.Series) -> pd.Series:
    """Parse ILOSTAT TIME_PERIOD strings ('2020-Q1', '2020-01', '2020')
    to first-of-period datetimes."""
    s = s.astype(str)
    is_q = s.str.contains("Q")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out.loc[~is_q] = pd.to_datetime(s[~is_q], errors="coerce")
    out.loc[is_q] = pd.PeriodIndex(s[is_q], freq="Q").to_timestamp()
    return out


def _reshape_ilostat_lfs_csv(
    raw: pd.DataFrame, *, frequency: str
) -> pd.DataFrame:
    """Reshape an SDMX-CSV ILOSTAT LFS response into the canonical 12-col schema.

    Accepts both the live API format (``AGE`` column, ``SEX_T``/``SEX_M``/
    ``SEX_F`` codes, ``UNE_DEAP_RT``-style MEASURE codes) and the legacy
    synthetic-fixture format (``CLASSIF1`` column, ``_T``/``M``/``F`` SEX
    codes, ``UNE_RT``-style codes). Legacy support is retained so that
    csv_path= calls with hand-crafted fixtures continue to work.

    Filters to ``FREQ == frequency``, drops ILO aggregate codes, translates
    AGE/CLASSIF1 to canonical age codes, pivots MEASURE to columns, and
    divides rate measures by 100 (ILOSTAT publishes percents; the canonical
    schema expects proportions).
    """
    # Detect whether this is the live API format (has 'AGE' column)
    # or the legacy fixture format (has 'CLASSIF1' column).
    if "AGE" in raw.columns:
        age_col = "AGE"
    elif "CLASSIF1" in raw.columns:
        age_col = "CLASSIF1"
    else:
        # Neither column present — return empty frame with correct schema.
        return _empty_lfs_frame()

    df = raw.loc[raw["FREQ"] == frequency, [
        "REF_AREA", "TIME_PERIOD", "SEX", age_col, "MEASURE", "OBS_VALUE",
    ]].copy()
    df = df.rename(columns={
        "REF_AREA":    "code",
        "TIME_PERIOD": "date",
        "SEX":         "sex",
        age_col:       "AGE_RAW",
    })
    # Drop aggregate codes.
    df = df[~df["code"].isin(_ILO_AGGREGATE_CODES)].copy()
    # Translate AGE_RAW to canonical ages; rows with unknown codes drop.
    df["age"] = df["AGE_RAW"].map(_LFS_AGE_RENAME)
    df = df[df["age"].notna()].copy()
    df["date"] = _parse_ilostat_time_period(df["date"])
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")

    # Translate MEASURE to canonical column names (drop measures we don't track).
    df["MEASURE_CANON"] = df["MEASURE"].map(_LFS_MEASURE_RENAME)
    df = df[df["MEASURE_CANON"].notna()].copy()

    # Divide percent-rates by 100 so the canonical schema is in proportions.
    rate_mask = df["MEASURE_CANON"].isin(_RATE_MEASURE_KEYS)
    df.loc[rate_mask, "OBS_VALUE"] = df.loc[rate_mask, "OBS_VALUE"] / 100.0

    # Normalise SEX codes: live API returns SEX_T/SEX_M/SEX_F;
    # legacy fixtures use _T/M/F. Map both to the canonical _T/M/F.
    sex_map = {"SEX_T": "_T", "SEX_M": "M", "SEX_F": "F"}
    df["sex"] = df["sex"].replace(sex_map)

    wide = (
        df.pivot_table(
            index=["code", "date", "sex", "age"],
            columns="MEASURE_CANON",
            values="OBS_VALUE",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None

    # Ensure all canonical columns exist (NaN for measures absent in this pull).
    for col in ("wa", "lf", "emp", "une", "ina", "urate", "epop", "prate"):
        if col not in wide.columns:
            wide[col] = np.nan

    keep = ["code", "date", "sex", "age",
            "wa", "lf", "emp", "une", "ina",
            "urate", "epop", "prate"]
    return wide[keep].sort_values(
        ["code", "date", "sex", "age"]
    ).reset_index(drop=True)


def _empty_lfs_frame() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical 12-col LFS schema."""
    return pd.DataFrame(columns=[
        "code", "date", "sex", "age",
        "wa", "lf", "emp", "une", "ina",
        "urate", "epop", "prate",
    ])


def _apply_x13_per_cell_lfs(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply X13-ARIMA SA per (code, sex, age) cell for each rate column.

    Cells with < 12 non-NaN observations (X13 minimum for quarterly) have
    their rate columns left as NaN — the SA cannot be performed and we
    refuse to silently fall back to NSA values.

    Operates only on the rate columns (urate, epop, prate). Levels
    (wa, lf, emp, une, ina) are left unchanged because at LFS frequency
    those are typically not what's being SA'd in the rate-based panel.
    """
    from ._seasonal import x13_seasonal_adjust

    rate_cols = [c for c in ("urate", "epop", "prate") if c in panel.columns]
    if not rate_cols:
        return panel

    out = panel.copy().sort_values(["code", "sex", "age", "date"]).reset_index(drop=True)

    for (code, sex, age), grp in out.groupby(["code", "sex", "age"], sort=False):
        if len(grp) < 12:
            # Mark rate cells in this group as NaN — refuses to ship NSA in SA mode.
            for col in rate_cols:
                out.loc[grp.index, col] = np.nan
            continue
        # Index by date for the SA call so the returned series aligns back.
        grp_indexed = grp.set_index("date")
        for col in rate_cols:
            series = grp_indexed[col].dropna()
            if len(series) < 12:
                out.loc[grp.index, col] = np.nan
                continue
            try:
                sa = x13_seasonal_adjust(series, freq="Q")
            except RuntimeError:
                # On any X13 failure (convergence, etc.), drop this cell to NaN.
                out.loc[grp.index, col] = np.nan
                continue
            # Re-align SA series back to the panel's row indices for this group.
            for ts, val in sa.items():
                row_mask = (out["code"] == code) & (out["sex"] == sex) \
                    & (out["age"] == age) & (out["date"] == ts)
                out.loc[row_mask, col] = val

    return out


# ---------------------------------------------------------------------------
# Verified ILOSTAT dataflow IDs (confirmed via live catalog + data pulls)
# Catalog: https://sdmx.ilo.org/rest/dataflow/ILO/all
# DSD dimension order for all these flows: REF_AREA.FREQ.MEASURE.SEX.AGE
# ---------------------------------------------------------------------------

# LFS flows: single quarterly dataflow that combines unemployment rate,
# employment-to-pop rate, and LFP rate via MEASURE dimension.
# For quarterly counts (emp, une, lf), separate dataflows are used.
_LFS_DATAFLOWS: dict[str, str] = {
    # Quarterly rates: unemployment rate, epop rate, LFP rate
    "Q_rates":  "DF_UNE_DEAP_SEX_AGE_RT",   # MEASURE=UNE_DEAP_RT (urate)
    # Quarterly employment counts
    "Q_emp":    "DF_EMP_TEMP_SEX_AGE_NB",    # MEASURE=EMP_TEMP_NB
    # Quarterly unemployment counts
    "Q_une":    "DF_UNE_TUNE_SEX_AGE_NB",    # MEASURE=UNE_TUNE_NB
    # Monthly employment counts
    "M_emp":    "DF_EMP_TEMP_SEX_AGE_NB",    # same dataflow, filter FREQ=M
    # Monthly unemployment counts
    "M_une":    "DF_UNE_TUNE_SEX_AGE_NB",    # same dataflow, filter FREQ=M
}

# Additional rates dataflows for epop and LFP rate.
# These are fetched alongside DF_UNE_DEAP_SEX_AGE_RT for quarterly rate panels.
_LFS_RATE_DATAFLOWS: tuple[str, ...] = (
    "DF_UNE_DEAP_SEX_AGE_RT",   # unemployment rate
    "DF_EMP_DWAP_SEX_AGE_RT",   # employment-to-pop rate
    "DF_EAP_DWAP_SEX_AGE_RT",   # LFP rate
)

# ILOSTAT live API SEX codes (used in key construction).
# The API accepts SEX_T, SEX_M, SEX_F — not bare _T, M, F.
_SEX_CODE_MAP: dict[str, str] = {
    "_T": "SEX_T",
    "M":  "SEX_M",
    "F":  "SEX_F",
    # Pass-through for callers already using API codes.
    "SEX_T": "SEX_T",
    "SEX_M": "SEX_M",
    "SEX_F": "SEX_F",
}

# ILOSTAT live API AGE codes (used in key construction).
# Canonical public API uses AGE_YTHADULT_* form.
_AGE_CODE_MAP: dict[str, str] = {
    "Y_GE15":  "AGE_YTHADULT_YGE15",
    "Y15T24":  "AGE_YTHADULT_Y15-24",
    "Y25T54":  "AGE_YTHADULT_Y25-54",
    "Y55T64":  "AGE_YTHADULT_Y55-64",
    "Y_GE25":  "AGE_YTHADULT_YGE25",
    # Pass-through for callers already using API codes.
    "AGE_YTHADULT_YGE15":  "AGE_YTHADULT_YGE15",
    "AGE_YTHADULT_Y15-24": "AGE_YTHADULT_Y15-24",
    "AGE_YTHADULT_Y25-54": "AGE_YTHADULT_Y25-54",
    "AGE_YTHADULT_YGE25":  "AGE_YTHADULT_YGE25",
    "AGE_YTHADULT_Y55-64": "AGE_YTHADULT_Y55-64",
}

# Default MEASURE codes per frequency / kind (live API codes).
_LFS_MEASURES_DEFAULT: dict[str, tuple[str, ...]] = {
    "Q_rates": ("UNE_DEAP_RT",),   # one per dataflow
    "Q_emp":   ("EMP_TEMP_NB",),
    "Q_une":   ("UNE_TUNE_NB",),
    "M_emp":   ("EMP_TEMP_NB",),
    "M_une":   ("UNE_TUNE_NB",),
}


def fetch_ilostat_lfs_panel(
    *,
    countries: list[str] | None = None,
    sexes:     tuple[str, ...] = ("_T", "M", "F"),
    ages:      tuple[str, ...] = ("Y_GE15", "Y15T24", "Y25T54", "Y55T64"),
    frequency: str = "Q",
    adjustment: str = "SA",
    csv_path:  str | Path | None = None,
    cache_path: str | Path | None = None,
    refresh:   bool = False,
) -> pd.DataFrame:
    """Fetch an ILOSTAT sex × age labor panel as a canonical 12-col DataFrame.

    Parameters
    ----------
    countries : list[str] | None
        ISO-3 codes; ``None`` returns all reporters (aggregates dropped).
    sexes, ages : tuple[str, ...]
        Canonical codes: sexes use ``"_T"``, ``"M"``, ``"F"``; ages use
        ``"Y_GE15"``-style codes which are translated to ILOSTAT API codes
        (``AGE_YTHADULT_YGE15``) internally.
    frequency : str
        ``"Q"`` for quarterly or ``"M"`` for monthly.
    adjustment : str
        ``"SA"`` (default; X13 applied client-side for quarterly data) or
        ``"NSA"`` (raw NSA returned as-is). For monthly data, X13 is NOT
        applied even when ``adjustment="SA"`` (deferred to a follow-up).
    csv_path : str | Path | None
        Local SDMX-CSV file. When set and exists, network fetch is skipped.
    cache_path : str | Path | None
        Parquet cache. If exists and ``refresh=False``, returned without
        reshape work.
    refresh : bool
        Force re-fetch (ignore cache).

    Returns
    -------
    pd.DataFrame
        Long panel with columns ``code, date, sex, age, wa, lf, emp,
        une, ina, urate, epop, prate``. Sorted by ``(code, date, sex, age)``.
    """
    if frequency not in {"Q", "M"}:
        raise ValueError(f"frequency must be 'Q' or 'M'; got {frequency!r}")
    if adjustment not in {"SA", "NSA"}:
        raise ValueError(f"adjustment must be 'SA' or 'NSA'; got {adjustment!r}")

    # 1. Cache fast path.
    if cache_path is not None and not refresh:
        cp = Path(cache_path)
        if cp.exists():
            return pd.read_parquet(cp)

    # 2. Local file fast path (skips network).
    if csv_path is not None:
        sp = Path(csv_path)
        if not sp.exists():
            raise FileNotFoundError(f"csv not found: {sp}")
        raw = pd.read_csv(sp)
    else:
        # 3. Live SDMX. Translate caller's canonical codes to live API codes.
        api_sexes = tuple(_SEX_CODE_MAP.get(s, s) for s in sexes)
        api_ages  = tuple(_AGE_CODE_MAP.get(a, a) for a in ages)

        if frequency == "Q":
            # Fetch three rate dataflows + emp + une counts.
            frames = []
            # Rates
            for df_id in _LFS_RATE_DATAFLOWS:
                # Each rate dataflow has one MEASURE code — derive from ID.
                measure_code = df_id.split("_", 1)[1].replace("_SEX_AGE_RT", "_RT") \
                    if False else _df_id_to_measure(df_id)
                key = _build_ilostat_lfs_key(
                    countries=countries, sexes=api_sexes, ages=api_ages,
                    measures=(measure_code,), frequency="Q",
                )
                frames.append(
                    sdmx_get(provider="ilostat", dataflow=df_id, key=key)
                )
            # Counts
            for sub_kind in ("Q_emp", "Q_une"):
                df_id = _LFS_DATAFLOWS[sub_kind]
                measures = _LFS_MEASURES_DEFAULT[sub_kind]
                key = _build_ilostat_lfs_key(
                    countries=countries, sexes=api_sexes, ages=api_ages,
                    measures=measures, frequency="Q",
                )
                frames.append(
                    sdmx_get(provider="ilostat", dataflow=df_id, key=key)
                )
            raw = pd.concat(frames, ignore_index=True)
        else:
            # Monthly: emp + une counts.
            frames = []
            for sub_kind in ("M_emp", "M_une"):
                df_id = _LFS_DATAFLOWS[sub_kind]
                measures = _LFS_MEASURES_DEFAULT[sub_kind]
                key = _build_ilostat_lfs_key(
                    countries=countries, sexes=api_sexes, ages=api_ages,
                    measures=measures, frequency="M",
                )
                frames.append(
                    sdmx_get(provider="ilostat", dataflow=df_id, key=key)
                )
            raw = pd.concat(frames, ignore_index=True)

    out = _reshape_ilostat_lfs_csv(raw, frequency=frequency)

    # Optional country post-filter (csv may include more than requested).
    if countries is not None:
        out = out[out["code"].isin(countries)].reset_index(drop=True)

    # 4. Apply X13 SA per cell when adjustment="SA" and frequency="Q".
    if adjustment == "SA" and frequency == "Q" and not out.empty:
        out = _apply_x13_per_cell_lfs(out)

    # 5. Cache write.
    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cp, index=False)

    return out


def _df_id_to_measure(df_id: str) -> str:
    """Derive the MEASURE code from an LFS rate dataflow ID.

    ILOSTAT rate dataflow IDs follow the pattern DF_<DOMAIN>_<MEASURE_BASE>_
    SEX_AGE_RT, and the MEASURE column value in the CSV is <DOMAIN>_<MEASURE_BASE>_RT.

    Examples::

        DF_UNE_DEAP_SEX_AGE_RT → UNE_DEAP_RT
        DF_EMP_DWAP_SEX_AGE_RT → EMP_DWAP_RT
        DF_EAP_DWAP_SEX_AGE_RT → EAP_DWAP_RT
    """
    _HARDCODED = {
        "DF_UNE_DEAP_SEX_AGE_RT": "UNE_DEAP_RT",
        "DF_EMP_DWAP_SEX_AGE_RT": "EMP_DWAP_RT",
        "DF_EAP_DWAP_SEX_AGE_RT": "EAP_DWAP_RT",
    }
    if df_id in _HARDCODED:
        return _HARDCODED[df_id]
    # Fallback: strip leading DF_ and trailing _SEX_AGE_RT.
    base = df_id.removeprefix("DF_")
    for suffix in ("_SEX_AGE_RT", "_SEX_AGE_NB"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    return base + "_RT"


# ---------------------------------------------------------------------------
# Sectoral (sex × ISIC) fetcher
# ---------------------------------------------------------------------------

# Sectoral measures: ILOSTAT MEASURE → canonical sectoral schema column.
_SECTORAL_MEASURE_RENAME: dict[str, str] = {
    # Live API codes
    "EMP_TEMP_NB": "emp",
    "HOW_TEMP_NB": "hours_worked",
    # Legacy fixture codes
    "EMP_TEMP":    "emp",
    "HOW_TEMP":    "hours_worked",
}

# CLASSIF1 prefix that ILOSTAT uses for ISIC codes.
_ECO_ISIC4_PREFIX = "ECO_ISIC4_"
_ECO_AGGREGATE_TOTAL = "ECO_AGGREGATE_TOTAL"

# Allowlist of sectoral codes we keep. Section letters + standard ILO
# multi-letter aggregations + total. 2-digit divisions are dropped.
_KEEP_ECO_CODES: frozenset[str] = frozenset({
    "_T", "A", "B-E", "F", "G-U",
    # individual single-letter sections kept too (caller may want them):
    "B", "C", "D", "E", "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R", "S", "T", "U",
})

# ILOSTAT SDMX dataflow IDs for sectoral data (confirmed against live catalog)
_SECTORAL_DATAFLOWS: tuple[str, ...] = (
    "DF_EMP_TEMP_SEX_ECO_NB",   # employment by sex × ISIC economic activity
    "DF_HOW_TEMP_SEX_ECO_NB",   # hours worked by sex × economic activity
)


def _build_ilostat_sectoral_key(
    *,
    countries: list[str] | None,
    sexes: tuple[str, ...],
    sectors: tuple[str, ...],
    measures: tuple[str, ...],
) -> str:
    """Assemble an ILOSTAT SDMX dot-key for a sectoral dataflow.

    Dimension order: ``REF_AREA.FREQ.MEASURE.SEX.ECO`` (Q frequency only).

    Sector codes are prefixed with ``ECO_ISIC4_`` for section letters; the
    total sector uses ``ECO_AGGREGATE_TOTAL``. The live ILOSTAT DSD confirms
    the dimension name is ``ECO`` (not ``CLASSIF1``).
    """
    def _join(xs):
        if xs is None:
            return ""
        return "+".join(xs)

    def _prefix_sector(code: str) -> str:
        if code == "_T":
            return _ECO_AGGREGATE_TOTAL
        return f"{_ECO_ISIC4_PREFIX}{code}"

    # Translate canonical sex codes to API codes.
    api_sexes = tuple(_SEX_CODE_MAP.get(s, s) for s in sexes)
    sector_codes = tuple(_prefix_sector(s) for s in sectors)
    return ".".join([
        _join(countries),
        "Q",
        _join(measures),
        _join(api_sexes),
        _join(sector_codes),
    ])


def _reshape_ilostat_sectoral_csv(raw: pd.DataFrame) -> pd.DataFrame:
    """Reshape an SDMX-CSV ILOSTAT sectoral response into the 6-col schema.

    Accepts both the live API format (``ECO`` column) and the legacy fixture
    format (``CLASSIF1`` column). Filters to ``FREQ == 'Q'``, drops ILO
    aggregate codes, strips the ``ECO_ISIC4_`` prefix from ECO/CLASSIF1,
    drops 2-digit ISIC divisions, and pivots MEASURE to ``emp`` /
    ``hours_worked``.
    """
    # Detect column format.
    if "ECO" in raw.columns:
        eco_col = "ECO"
    elif "CLASSIF1" in raw.columns:
        eco_col = "CLASSIF1"
    else:
        return pd.DataFrame(columns=["code", "date", "sex", "eco", "emp", "hours_worked"])

    df = raw.loc[raw["FREQ"] == "Q", [
        "REF_AREA", "TIME_PERIOD", "SEX", eco_col, "MEASURE", "OBS_VALUE",
    ]].copy()
    df = df.rename(columns={
        "REF_AREA":    "code",
        "TIME_PERIOD": "date",
        "SEX":         "sex",
        eco_col:       "ECO_RAW",
    })
    df = df[~df["code"].isin(_ILO_AGGREGATE_CODES)].copy()

    # Translate ECO_RAW → eco (strip prefix, normalize total).
    def _eco_from_raw(c1: str) -> str | None:
        if c1 == _ECO_AGGREGATE_TOTAL:
            return "_T"
        if c1.startswith(_ECO_ISIC4_PREFIX):
            stripped = c1[len(_ECO_ISIC4_PREFIX):]
            return stripped if stripped in _KEEP_ECO_CODES else None
        return None

    df["eco"] = df["ECO_RAW"].map(_eco_from_raw)
    df = df[df["eco"].notna()].copy()

    df["date"] = _parse_ilostat_time_period(df["date"])
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")

    df["MEASURE_CANON"] = df["MEASURE"].map(_SECTORAL_MEASURE_RENAME)
    df = df[df["MEASURE_CANON"].notna()].copy()

    # Normalise SEX codes.
    sex_map = {"SEX_T": "_T", "SEX_M": "M", "SEX_F": "F"}
    df["sex"] = df["sex"].replace(sex_map)

    wide = (
        df.pivot_table(
            index=["code", "date", "sex", "eco"],
            columns="MEASURE_CANON",
            values="OBS_VALUE",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None

    for col in ("emp", "hours_worked"):
        if col not in wide.columns:
            wide[col] = np.nan

    keep = ["code", "date", "sex", "eco", "emp", "hours_worked"]
    return wide[keep].sort_values(
        ["code", "date", "sex", "eco"]
    ).reset_index(drop=True)


def _apply_x13_per_cell_sectoral(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply X13-ARIMA SA per (code, sex, eco) cell for emp and hours_worked.

    Cells with < 12 non-NaN observations are returned NaN — refuses to
    silently ship NSA values in SA mode.
    """
    from ._seasonal import x13_seasonal_adjust

    measure_cols = [c for c in ("emp", "hours_worked") if c in panel.columns]
    if not measure_cols:
        return panel

    out = panel.copy().sort_values(["code", "sex", "eco", "date"]).reset_index(drop=True)

    for (code, sex, eco), grp in out.groupby(["code", "sex", "eco"], sort=False):
        if len(grp) < 12:
            for col in measure_cols:
                out.loc[grp.index, col] = np.nan
            continue
        grp_indexed = grp.set_index("date")
        for col in measure_cols:
            series = grp_indexed[col].dropna()
            if len(series) < 12:
                out.loc[grp.index, col] = np.nan
                continue
            try:
                sa = x13_seasonal_adjust(series, freq="Q")
            except RuntimeError:
                out.loc[grp.index, col] = np.nan
                continue
            for ts, val in sa.items():
                row_mask = (out["code"] == code) & (out["sex"] == sex) \
                    & (out["eco"] == eco) & (out["date"] == ts)
                out.loc[row_mask, col] = val

    return out


def fetch_ilostat_sectoral_panel(
    *,
    countries: list[str] | None = None,
    sexes:     tuple[str, ...] = ("_T", "M", "F"),
    sectors:   tuple[str, ...] = ("_T", "A", "B-E", "F", "G-U"),
    adjustment: str = "SA",
    csv_path:  str | Path | None = None,
    cache_path: str | Path | None = None,
    refresh:   bool = False,
) -> pd.DataFrame:
    """Fetch an ILOSTAT sex × ISIC sectoral panel as a 6-col DataFrame.

    Parameters
    ----------
    countries : list[str] | None
        ISO-3 codes; ``None`` returns all reporters.
    sexes : tuple[str, ...]
        Canonical sex codes (``"_T"``, ``"M"``, ``"F"``).
    sectors : tuple[str, ...]
        ISIC Rev.4 codes — section letters and standard ILO multi-letter
        aggregations (``"_T"`` total, ``"A"``, ``"B-E"``, ``"F"``, ``"G-U"``).
    adjustment : str
        ``"SA"`` (default; X13 applied per cell) or ``"NSA"``.
    csv_path : str | Path | None
        Local SDMX-CSV file.
    cache_path : str | Path | None
        Parquet cache.
    refresh : bool
        Force re-fetch.

    Returns
    -------
    pd.DataFrame
        Long panel with columns ``code, date, sex, eco, emp, hours_worked``.
    """
    if adjustment not in {"SA", "NSA"}:
        raise ValueError(f"adjustment must be 'SA' or 'NSA'; got {adjustment!r}")

    if cache_path is not None and not refresh:
        cp = Path(cache_path)
        if cp.exists():
            return pd.read_parquet(cp)

    if csv_path is not None:
        sp = Path(csv_path)
        if not sp.exists():
            raise FileNotFoundError(f"csv not found: {sp}")
        raw = pd.read_csv(sp)
    else:
        frames = []
        for dataflow in _SECTORAL_DATAFLOWS:
            # Derive measure from dataflow: EMP_TEMP_SEX_ECO_NB → EMP_TEMP_NB
            if "EMP_TEMP" in dataflow:
                measures = ("EMP_TEMP_NB",)
            else:
                measures = ("HOW_TEMP_NB",)
            key = _build_ilostat_sectoral_key(
                countries=countries, sexes=sexes,
                sectors=sectors, measures=measures,
            )
            frames.append(
                sdmx_get(provider="ilostat", dataflow=dataflow, key=key)
            )
        raw = pd.concat(frames, ignore_index=True)

    out = _reshape_ilostat_sectoral_csv(raw)

    if countries is not None:
        out = out[out["code"].isin(countries)].reset_index(drop=True)

    if adjustment == "SA" and not out.empty:
        out = _apply_x13_per_cell_sectoral(out)

    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cp, index=False)

    return out


__all__ = [
    "fetch_ilostat_lfs_panel",
    "fetch_ilostat_sectoral_panel",
]
