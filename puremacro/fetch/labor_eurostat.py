"""Eurostat LFS fetchers (SDMX-CSV) — monthly urate + quarterly epop / prate.

Two parallel fetchers:

- :func:`fetch_eurostat_lfs_panel` — monthly unemployment-rate panel from
  the ``une_rt_m`` dataflow (sex × age × geo). Only ``urate`` populated.
- :func:`fetch_eurostat_lfs_quarterly_panel` — quarterly employment +
  activity rates from the single ``lfsi_emp_q`` dataflow under two
  ``indic_em`` filter values: ``EMP_LFS`` (→ ``epop``) and ``ACT``
  (→ ``prate``); 4 canonical age bands (``Y15-24``, ``Y25-54``,
  ``Y55-64``, ``Y15-64``); values forward-filled from quarterly to
  monthly so the output joins the monthly union.

Both normalize geo (ISO-2 → ISO-3, aggregates dropped), sex
(``T → _T``), and age. The monthly fetcher uses ``Y_LT25 → Y15T24`` and
``TOTAL → Y_GE15`` only; the quarterly fetcher additionally provides
``Y25-54 → Y25T54`` and ``Y55-64 → Y55T64``.

Output schema matches :func:`puremacro.fetch.labor.fetch_oecd_lfs_panel`:
``[code, date, sex, age, wa, lf, emp, une, ina, urate, epop, prate]``,
with only ``urate`` populated. Caller's ``puremacro.fetch.labor.
fetch_labor_panel_union`` is the recommended way to combine this output
with the OECD/LABOR.xlsx panel.

References
----------
Dataflow ID:  une_rt_m
Endpoint:     https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/
DSD dimensions (verified live): freq.s_adj.age.unit.sex.geo
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .sdmx import sdmx_get


# Eurostat's geo codes (mostly ISO-2) → canonical ISO-3.
# Aggregates (EU27_2020, EA20, EFTA, EU28, EA19, EU27, OECD, …) are
# DELIBERATELY absent: rows whose geo is not in this dict are dropped.
_EUROSTAT_GEO_TO_ISO3: dict[str, str] = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "CH": "CHE", "CY": "CYP",
    "CZ": "CZE", "DE": "DEU", "DK": "DNK", "EE": "EST", "EL": "GRC",
    "ES": "ESP", "FI": "FIN", "FR": "FRA", "HR": "HRV", "HU": "HUN",
    "IE": "IRL", "IS": "ISL", "IT": "ITA", "LI": "LIE", "LT": "LTU",
    "LU": "LUX", "LV": "LVA", "MT": "MLT", "NL": "NLD", "NO": "NOR",
    "PL": "POL", "PT": "PRT", "RO": "ROU", "SE": "SWE", "SI": "SVN",
    "SK": "SVK", "TR": "TUR", "UK": "GBR",
}

_EUROSTAT_SEX_MAP: dict[str, str] = {"T": "_T", "M": "M", "F": "F"}

# Only two age cuts kept; Y25-74 is broader than canonical Y25T54 so it
# is intentionally absent (rows are dropped during reshape).
_EUROSTAT_AGE_MAP: dict[str, str] = {"Y_LT25": "Y15T24", "TOTAL": "Y_GE15"}


# Eurostat quarterly LFS rate indicators inside the single `lfsi_emp_q`
# dataflow ("Employment and activity by sex and age - quarterly data").
# Each is a different `indic_em` filter value:
#   EMP_LFS → employment rate → epop
#   ACT     → activity / LFP rate → prate
# Unit is PC_POP (% of population). Quarterly frequency, ~30 EU/EFTA
# countries.
_LFSI_Q_DATAFLOW: str = "lfsi_emp_q"
_LFSI_Q_INDICATORS: dict[str, str] = {
    "EMP_LFS": "epop",   # employment rate
    "ACT":     "prate",  # activity (LFP) rate
}

# Quarterly age cuts: 4 canonical bands. Eurostat's quarterly LFS rejects
# `Y_GE15` (15+) for the lfsi_emp_q dataflow; the broad age cut is
# `Y15-64` (working-age, 15-64). This is SEMANTICALLY DIFFERENT from the
# Y_GE15 used by OECD/ILOSTAT panels — we surface it as a new canonical
# age band `Y15-64` instead of conflating with Y_GE15.
_EUROSTAT_AGE_MAP_Q: dict[str, str] = {
    "Y15-24": "Y15T24",
    "Y25-54": "Y25T54",
    "Y55-64": "Y55T64",
    "Y15-64": "Y15-64",   # NEW canonical band — NOT a synonym for Y_GE15.
}


def _build_eurostat_une_key(
    *,
    sexes: tuple[str, ...],
    countries: list[str] | None,
) -> str:
    """Assemble the SDMX dot-key for ``une_rt_m``.

    Dimension order (verified from the dataflow's DSD):
    ``freq.s_adj.age.unit.sex.geo``. We hard-code:
      - freq = M
      - s_adj = SA
      - age  = Y_LT25+TOTAL  (the two cuts we keep)
      - unit = PC_ACT        (% of active pop)
    and fill ``sex`` and ``geo`` from the inputs. Empty ``geo`` returns
    all reporters; the downstream filter drops aggregates.
    """
    iso3_to_iso2 = {v: k for k, v in _EUROSTAT_GEO_TO_ISO3.items()}
    if countries is None:
        geo = ""
    else:
        # Will KeyError on unknown ISO-3 (e.g. USA) — surface the bad code.
        geo = "+".join(iso3_to_iso2[c] for c in countries)
    sex_codes = [k for s in sexes for k, v in _EUROSTAT_SEX_MAP.items() if v == s]
    sex_part = "+".join(sex_codes)
    return f"M.SA.Y_LT25+TOTAL.PC_ACT.{sex_part}.{geo}"


def _reshape_eurostat_une_rt_m(raw: pd.DataFrame) -> pd.DataFrame:
    """Reshape an SDMX-CSV ``une_rt_m`` response into the canonical schema.

    Filters to ``s_adj == "SA"`` and ``unit == "PC_ACT"``, drops aggregate
    geo, drops ``Y25-74`` (kept ages: ``Y_LT25, TOTAL``), translates
    geo/sex/age to canonical codes, and divides ``OBS_VALUE`` by 100 to
    get a proportion (0..1) consistent with the OECD-fetcher schema.
    """
    df = raw.loc[
        (raw["s_adj"] == "SA") & (raw["unit"] == "PC_ACT"),
        ["geo", "sex", "age", "TIME_PERIOD", "OBS_VALUE"],
    ].copy()

    # Translate geo first; rows with unknown geo (aggregates) get NaN and dropped.
    df["code"] = df["geo"].map(_EUROSTAT_GEO_TO_ISO3)
    df = df[df["code"].notna()]

    # Translate sex; map preserves all known values, so no drops here.
    df["sex"] = df["sex"].map(_EUROSTAT_SEX_MAP)
    df = df[df["sex"].notna()]

    # Translate age; rows with age not in the map (Y25-74, others) get NaN and dropped.
    df["age"] = df["age"].map(_EUROSTAT_AGE_MAP)
    df = df[df["age"].notna()]

    # Date parse + rate scaling.
    df["date"] = pd.PeriodIndex(df["TIME_PERIOD"].astype(str), freq="M").to_timestamp()
    df["urate"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce") / 100.0

    # Add canonical NaN columns.
    for col in ("wa", "lf", "emp", "une", "ina", "epop", "prate"):
        df[col] = np.nan

    keep = ["code", "date", "sex", "age",
            "wa", "lf", "emp", "une", "ina",
            "urate", "epop", "prate"]
    out = (df[keep]
           .drop_duplicates(subset=["code", "date", "sex", "age"])
           .sort_values(["code", "date", "sex", "age"])
           .reset_index(drop=True))
    return out


def _reshape_eurostat_lfsi_q(raw: pd.DataFrame, *, rate_col: str) -> pd.DataFrame:
    """Reshape an SDMX-CSV ``lfsi_emp_q`` response (one ``indic_em`` value
    per call) into the canonical schema with one rate column populated.

    Parameters
    ----------
    raw : pd.DataFrame
        SDMX-CSV response; columns include ``geo``, ``sex``, ``age``,
        ``TIME_PERIOD``, ``OBS_VALUE``.
    rate_col : str
        Either ``"epop"`` (for indic_em=EMP_LFS) or ``"prate"``
        (for indic_em=ACT).

    Returns
    -------
    pd.DataFrame
        Tidy frame with columns [code, date, sex, age, <rate_col>].
        Aggregate ages and non-EU geos are dropped. Values divided by 100
        (Eurostat rates ship as percent; canonical schema uses fraction).
    """
    if rate_col not in {"epop", "prate"}:
        raise ValueError(f"rate_col must be 'epop' or 'prate', got {rate_col!r}")
    if raw.empty:
        return pd.DataFrame(columns=["code", "date", "sex", "age", rate_col])

    df = raw.copy()
    # Drop aggregate ages first to keep the post-rename frame clean.
    df = df[df["age"].isin(_EUROSTAT_AGE_MAP_Q.keys())]
    df["age"] = df["age"].map(_EUROSTAT_AGE_MAP_Q)
    # Map sex (T → _T, M → M, F → F).
    df = df[df["sex"].isin(_EUROSTAT_SEX_MAP.keys())]
    df["sex"] = df["sex"].map(_EUROSTAT_SEX_MAP)
    # Map geo (drop aggregates: anything not in the ISO-3 map).
    df = df[df["geo"].isin(_EUROSTAT_GEO_TO_ISO3.keys())]
    df["code"] = df["geo"].map(_EUROSTAT_GEO_TO_ISO3)
    # Convert TIME_PERIOD ('YYYY-Q1') → quarter-start Timestamp.
    df["date"] = pd.PeriodIndex(df["TIME_PERIOD"], freq="Q").to_timestamp(how="start")
    # Eurostat rates are percent; canonical schema uses fraction.
    df[rate_col] = pd.to_numeric(df["OBS_VALUE"], errors="coerce") / 100.0

    out = df[["code", "date", "sex", "age", rate_col]].copy()
    out = out.dropna(subset=[rate_col])
    out = out.sort_values(["code", "date", "sex", "age"]).reset_index(drop=True)
    return out


def _reshape_eurostat_une_rt_q(raw: pd.DataFrame) -> pd.DataFrame:
    """Reshape an SDMX-CSV ``une_rt_q`` response into the canonical schema
    with the ``urate`` column populated.

    Mirrors :func:`_reshape_eurostat_lfsi_q` but for ``une_rt_q`` whose
    DSD differs in two ways:
      - 6 dimensions (no `indic_em`): freq.s_adj.age.unit.sex.geo.
      - unit is ``PC_ACT`` (% of active population, not PC_POP).

    Only `Y25-54` rows are expected (the caller restricts the request),
    but for safety we still drop any row whose age is not in our
    canonical map. Geo / sex maps follow the existing Eurostat convention.

    Returns a tidy frame with columns [code, date, sex, age, urate].
    Values divided by 100 (Eurostat rates ship as percent; canonical
    schema uses fraction).
    """
    if raw.empty:
        return pd.DataFrame(columns=["code", "date", "sex", "age", "urate"])

    df = raw.copy()
    # Map age (only Y25-54 expected, but tolerate other accepted codes).
    age_map = {"Y15-24": "Y15T24", "Y25-54": "Y25T54"}
    df = df[df["age"].isin(age_map.keys())]
    df["age"] = df["age"].map(age_map)
    # Map sex.
    df = df[df["sex"].isin(_EUROSTAT_SEX_MAP.keys())]
    df["sex"] = df["sex"].map(_EUROSTAT_SEX_MAP)
    # Map geo (drop aggregates).
    df = df[df["geo"].isin(_EUROSTAT_GEO_TO_ISO3.keys())]
    df["code"] = df["geo"].map(_EUROSTAT_GEO_TO_ISO3)
    # Quarterly TIME_PERIOD → quarter-start Timestamp.
    df["date"] = pd.PeriodIndex(df["TIME_PERIOD"], freq="Q").to_timestamp(how="start")
    # Eurostat rates ship as percent; canonical uses fraction.
    df["urate"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce") / 100.0

    out = df[["code", "date", "sex", "age", "urate"]].dropna(subset=["urate"])
    out = out.sort_values(["code", "date", "sex", "age"]).reset_index(drop=True)
    return out


def _expand_quarterly_to_monthly(q_panel: pd.DataFrame) -> pd.DataFrame:
    """Expand a quarterly panel into monthly rows via forward-fill within quarter.

    Each (code, sex, age, quarter-start-date) row is replicated to 3 rows
    at month-starts (e.g. 2020-Q1 → 2020-01-01, 2020-02-01, 2020-03-01).
    All non-key columns are carried unchanged.

    Parameters
    ----------
    q_panel : pd.DataFrame
        Must contain ``[code, date, sex, age]`` plus any value columns.
        ``date`` must be quarter-start Timestamps.

    Returns
    -------
    pd.DataFrame
        Three rows per input row; sorted by (code, date, sex, age).
    """
    if q_panel.empty:
        return q_panel.copy()

    # Three shifted copies of the frame rather than a dict per output row:
    # `.iterrows()` + `.to_dict()` boxes every cell into a Python object and
    # cost 3.1 s where this costs 0.012 s. Same rows, same values, same dtypes.
    base = q_panel.assign(date=pd.to_datetime(q_panel["date"]))
    out = pd.concat(
        [base.assign(date=base["date"] + pd.DateOffset(months=k)) for k in range(3)],
        ignore_index=True,
    )
    out = out.sort_values(["code", "date", "sex", "age"]).reset_index(drop=True)
    return out


def fetch_eurostat_lfs_quarterly_panel(
    *,
    countries: list[str] | None = None,
    adjustment: str = "SA",
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Pull Eurostat quarterly employment + activity + unemployment rates
    by sex × age.

    Combines three Eurostat dataflows into the canonical 12-col labor panel:
      - lfsi_emp_q with indic_em=EMP_LFS → epop (4 ages: Y15-24, Y25-54,
        Y55-64, Y15-64).
      - lfsi_emp_q with indic_em=ACT     → prate (same ages).
      - une_rt_q                          → urate, prime-age (Y25-54) only.
        Eurostat's une_rt_q rejects Y_GE15/Y15-64/Y55-64; the broader bands
        come from monthly une_rt_m via the parallel fetcher
        :func:`fetch_eurostat_lfs_panel`.

    Quarterly observations are forward-filled to monthly so the output
    joins cleanly with the monthly union in
    :func:`puremacro.fetch.labor.fetch_labor_panel_union`.

    Parameters
    ----------
    countries : list[str] | None
        ISO-3 country filter. Default ``None`` = pass empty geo to SDMX
        and let Eurostat return all reporters; the reshape filters to
        rows whose geo is in :data:`_EUROSTAT_GEO_TO_ISO3` (30 EU/EFTA
        countries). This avoids URL-length errors from passing 29
        explicit country codes. Explicit caller lists are translated
        ISO-3 → Eurostat codes; unknown ISO-3 codes raise KeyError.
        Note: lfsi_emp_q does NOT publish post-Brexit UK data, so GBR
        rows will be empty regardless of how `countries` is set.
    adjustment : str
        Eurostat ``s_adj`` code; default ``"SA"``.
    cache_path : str | Path | None
        Optional parquet cache. If the file exists and is non-empty,
        return its contents without hitting the network.

    Returns
    -------
    pd.DataFrame
        Canonical schema [code, date, sex, age, wa, lf, emp, une, ina,
        urate, epop, prate]; ``epop`` and ``prate`` populated, all
        other rate / stock columns NaN. Sorted by (code, date, sex, age).
    """
    canonical = ["code", "date", "sex", "age",
                 "wa", "lf", "emp", "une", "ina",
                 "urate", "epop", "prate"]

    # Cache hit short-circuit. Fail loud on schema drift so a stale cache
    # from an older version surfaces clearly instead of silently returning
    # the wrong shape.
    if cache_path is not None:
        cp = Path(cache_path)
        if cp.exists() and cp.stat().st_size > 0:
            cached = pd.read_parquet(cp)
            missing = set(canonical) - set(cached.columns)
            if missing:
                raise ValueError(
                    f"cache schema mismatch in {cp}: missing columns {sorted(missing)}. "
                    f"Delete the parquet to force a fresh pull."
                )
            return cached[canonical]

    # Country filter:
    # - countries=None: pass empty geo to SDMX, let Eurostat return all
    #   reporters. The reshape drops any geo not in _EUROSTAT_GEO_TO_ISO3,
    #   so non-EU rows are filtered post-fetch. This avoids URL-length
    #   issues from explicit 29-country lists (Eurostat's SDMX endpoint
    #   rejects very long keys) and matches _build_eurostat_une_key's
    #   pattern.
    # - countries=<list>: translate ISO-3 → Eurostat geo codes, KeyError
    #   on unknown ISO-3 codes (fail-loud per project convention).
    if countries is None:
        geo_codes = []   # empty geo → Eurostat returns all reporters
    else:
        iso3_to_geo = {v: k for k, v in _EUROSTAT_GEO_TO_ISO3.items()}
        # KeyError-on-unknown: surface bad codes rather than silently drop.
        geo_codes = [iso3_to_geo[c] for c in countries]

    # Build SDMX key. Dimension order for lfsi_emp_q (verified live via
    # the dataflow's DSD): freq.indic_em.s_adj.sex.age.unit.geo (7 dims).
    # `indic_em` is iterated below — emp rate vs activity rate live in the
    # SAME dataflow as different indic_em values.
    ages_q_key = "+".join(_EUROSTAT_AGE_MAP_Q.keys())   # "Y15-24+Y25-54+Y55-64+Y15-64"
    sex_key = "T+M+F"
    geo_key = "+".join(geo_codes) if geo_codes else ""

    # Pull both indicators and reshape. Partial failures propagate (e.g.
    # if EMP_LFS succeeds but ACT raises, the whole fetch raises) — the
    # T2b builder wraps this call in try/except per the task plan, so we
    # keep the fetcher itself fail-loud here.
    parts: list[pd.DataFrame] = []
    for indic_em, rate_col in _LFSI_Q_INDICATORS.items():
        key = f"Q.{indic_em}.{adjustment}.{sex_key}.{ages_q_key}.PC_POP.{geo_key}"
        raw = sdmx_get(
            provider="eurostat",
            dataflow=_LFSI_Q_DATAFLOW,
            key=key,
        )
        tidy = _reshape_eurostat_lfsi_q(raw, rate_col=rate_col)
        parts.append(tidy)

    # Third pull: une_rt_q for prime-age urate. The dataflow accepts only
    # Y15-24 and Y25-54 (verified live; Y_GE15 / Y15-64 / Y55-64 / TOTAL all
    # rejected). Y15-24 is already covered by monthly une_rt_m at higher
    # priority, so we only request Y25-54 here. Lifts T2b's urate×Y25T54
    # cohort from ~6 países (OECD-only) to ~24 (Eurostat-Q + OECD).
    # Different DSD from lfsi_emp_q: dimensions are freq.s_adj.age.unit.sex.geo
    # (6 dims), unit is PC_ACT (% of active population, not PC_POP).
    une_q_key = f"Q.{adjustment}.Y25-54.PC_ACT.{sex_key}.{geo_key}"
    try:
        une_q_raw = sdmx_get(
            provider="eurostat",
            dataflow="une_rt_q",
            key=une_q_key,
        )
        une_q_tidy = _reshape_eurostat_une_rt_q(une_q_raw)
        parts.append(une_q_tidy)
    except RuntimeError as exc:
        # une_rt_q failure is non-fatal: epop/prate already provided
        # by the lfsi_emp_q pulls above; just print and continue.
        print(f"  (une_rt_q skipped: {exc})")

    # Merge on (code, date, sex, age) outer-join: a country with epop but
    # no prate (or vice versa) keeps its row with the missing column NaN.
    if not parts:
        merged = pd.DataFrame(columns=["code", "date", "sex", "age", "epop", "prate"])
    else:
        merged = parts[0]
        for p in parts[1:]:
            merged = merged.merge(p, on=["code", "date", "sex", "age"], how="outer")

    # Forward-fill quarterly → monthly.
    merged = _expand_quarterly_to_monthly(merged)

    # Pad to canonical 12-col schema. urate may be present (from une_rt_q
    # pull) or absent (if that pull was skipped); preserve it if present
    # and pad only the strictly-missing stock columns.
    for col in ["wa", "lf", "emp", "une", "ina"]:
        merged[col] = np.nan
    if "urate" not in merged.columns:
        merged["urate"] = np.nan
    out = merged[canonical].sort_values(["code", "date", "sex", "age"]).reset_index(drop=True)

    # Write cache.
    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cp)

    return out


def fetch_eurostat_lfs_panel(
    *,
    countries: list[str] | None = None,        # ISO-3 codes; None = all available
    sexes:     tuple[str, ...] = ("_T", "M", "F"),
    csv_path:  str | Path | None = None,
    cache_path: str | Path | None = None,
    refresh:   bool = False,
) -> pd.DataFrame:
    """Fetch the Eurostat ``une_rt_m`` monthly unemployment-rate panel.

    Parameters
    ----------
    countries : list[str] | None
        ISO-3 codes (e.g. ``["AUT", "DEU", "GRC"]``). ``None`` returns all
        Eurostat reporters that have a known ISO-2 → ISO-3 mapping
        (i.e., excludes aggregates like EU27_2020 which are dropped during
        reshape).
    sexes : tuple[str, ...]
        Canonical sex codes (``"_T", "M", "F"``).
    csv_path : str | Path | None
        Local SDMX-CSV file (e.g. a pre-downloaded one for offline use).
        When set and exists, skips the network.
    cache_path : str | Path | None
        Parquet cache. If exists and ``refresh=False``, returned without
        reshape work.
    refresh : bool
        Force re-fetch (ignore cache).

    Returns
    -------
    pd.DataFrame
        Long panel with the canonical 12-column schema (``code, date,
        sex, age, wa, lf, emp, une, ina, urate, epop, prate``). Only
        ``urate`` is populated; level columns and ``epop, prate`` are
        NaN. Two age cells: ``Y_GE15`` and ``Y15T24``.
    """
    # 1. Cache fast path.
    if cache_path is not None and not refresh:
        cp = Path(cache_path)
        if cp.exists():
            return pd.read_parquet(cp)

    # 2. Local file fast path.
    if csv_path is not None:
        cp = Path(csv_path)
        if not cp.exists():
            raise FileNotFoundError(f"csv not found: {cp}")
        raw = pd.read_csv(cp)
    else:
        # 3. Live SDMX.
        key = _build_eurostat_une_key(sexes=sexes, countries=countries)
        raw = sdmx_get(provider="eurostat", dataflow="une_rt_m", key=key)

    out = _reshape_eurostat_une_rt_m(raw)

    # 4. Optional country post-filter (in case csv had more than requested).
    if countries is not None:
        out = out[out["code"].isin(countries)].reset_index(drop=True)

    # 5. Cache write.
    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cp, index=False)

    return out
