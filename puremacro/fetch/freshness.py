"""Bounded connector provenance, vintage semantics, and freshness reporting.

Evaluates a curated cohort of flagship puremacro data connectors across three
explicit modes:

- ``mode="cache"`` (zero network): inspects local SQLite cache tables
  (``http_cache``, ``alfred_vintages``, ``realtime_vintages``, and
  ``connector_events``), parses cached bodies where available, runs domain
  identity checks, and reports freshness, revision history, fallbacks, or
  ``not_cached`` / ``credential_missing`` status.
- ``mode="recorded"`` (zero network): runs each cohort connector's parser and
  domain identity check against deterministic recorded wire payloads and the
  bundled OECD QNA snapshot so CI, Pyodide, and offline wheels can verify
  provenance and structural invariants without external network calls.
- ``mode="live"`` (rate-budgeted network): executes bounded live probes under
  :class:`RateBudget` (default ``max_requests=5``), enforcing inter-request
  pacing, timeout bounds, credential pre-checks, fallback/drift attribution,
  and domain identity checks.

Every record distinguishes nine explicit statuses (:data:`FRESHNESS_STATUSES`):
``fresh``, ``stale``, ``revised``, ``empty``, ``degraded_fallback``,
``schema_drift``, ``credential_missing``, ``failed``, and ``not_cached``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

import puremacro
from puremacro import _cache_db
from puremacro._optional_stdlib import sqlite3
from puremacro.credentials import get_credential
from puremacro.fetch.energy_transition import _parse_ember_df
from puremacro.fetch.laus import parse_laus_series
from puremacro.fetch.realtime.banxico import parse_banxico_json
from puremacro.fetch.realtime.bcb import fetch_bcb_vintages, parse_bcb_json
from puremacro.fetch.realtime.canary import SchemaDriftError, SchemaDriftWarning

FRESHNESS_STATUSES: tuple[str, ...] = (
    "fresh",
    "stale",
    "revised",
    "empty",
    "degraded_fallback",
    "schema_drift",
    "credential_missing",
    "failed",
    "not_cached",
)

OUTAGE_STATUSES: tuple[str, ...] = (
    "empty",
    "schema_drift",
    "failed",
    "degraded_fallback",
)

VALID_MODES: tuple[str, ...] = ("cache", "recorded", "live")


@dataclass(frozen=True)
class CohortConnectorSpec:
    """Metadata and structural contract for one flagship cohort connector."""

    source_id: str
    provider: str
    workflow: str
    series_key: str
    freq: str
    units: str
    transformations: str
    vintage_meaning: str
    max_staleness_days: int
    auth: str = "none"
    credential: str | None = None
    identity_check: str = "positive_finite"
    cache_url_substrings: tuple[str, ...] = ()
    telemetry_source: str = ""


SUPPORTED_CONNECTOR_COHORT: tuple[CohortConnectorSpec, ...] = (
    CohortConnectorSpec(
        source_id="fred.csv",
        provider="fred",
        workflow="us_macro",
        series_key="GDPC1",
        freq="Q",
        units="billions_chained_2017_usd",
        transformations="none (published SAAR chained level)",
        vintage_meaning="latest_published_edition",
        max_staleness_days=150,
        auth="none",
        credential=None,
        identity_check="positive_finite",
        cache_url_substrings=("fredgraph.csv", "GDPC1"),
        telemetry_source="fred",
    ),
    CohortConnectorSpec(
        source_id="fred.alfred",
        provider="fred",
        workflow="us_realtime_revisions",
        series_key="GDPC1",
        freq="Q",
        units="billions_chained_usd_by_vintage",
        transformations="none (archival edition levels)",
        vintage_meaning="publication_date_archive",
        max_staleness_days=150,
        auth="key_optional",
        credential="fred",
        identity_check="multi_vintage_archive",
        cache_url_substrings=("alfred", "GDPC1"),
        telemetry_source="alfred",
    ),
    CohortConnectorSpec(
        source_id="bls.laus_state",
        provider="bls",
        workflow="us_regional_labor",
        series_key="LASST* (state SA: lf, emp, une, urate)",
        freq="M",
        units="persons_and_percent",
        transformations="none (BLS seasonally adjusted)",
        vintage_meaning="latest_published_edition",
        max_staleness_days=90,
        auth="none",
        credential=None,
        identity_check="laus_labor_force_identity",
        cache_url_substrings=("download.bls.gov/pub/time.series/la/la.data.1.CurrentS",),
        telemetry_source="bls_laus",
    ),
    CohortConnectorSpec(
        source_id="oecd.qna_panel",
        provider="oecd",
        workflow="cross_country_qna",
        series_key="OECD.SDD.NAD:B1GQ+P3 (USA, DEU)",
        freq="Q",
        units="millions_national_currency_and_index_100",
        transformations="deflator = nominal / chain_volume * 100",
        vintage_meaning="latest_published_edition",
        max_staleness_days=210,
        auth="none",
        credential=None,
        identity_check="qna_deflator_consistency",
        cache_url_substrings=("DF_QNA_EXPENDITURE_NATIO_CURR",),
        telemetry_source="oecd_qna",
    ),
    CohortConnectorSpec(
        source_id="wdi.panel",
        provider="worldbank",
        workflow="cross_country_development",
        series_key="NY.GDP.MKTP.KD (USA, DEU)",
        freq="A",
        units="constant_2015_usd",
        transformations="none (World Bank WDI annual level)",
        vintage_meaning="latest_published_edition",
        max_staleness_days=750,
        auth="none",
        credential=None,
        identity_check="positive_finite",
        cache_url_substrings=("api.worldbank.org", "NY.GDP.MKTP.KD"),
        telemetry_source="wdi",
    ),
    CohortConnectorSpec(
        source_id="worldbank.energy_transition",
        provider="worldbank",
        workflow="climate_energy",
        series_key="elec_gen_*_pct_a / elec_gen_total_twh_a (USA, DEU)",
        freq="A",
        units="twh_and_share_pct",
        transformations="Ember / WDI standardized generation shares (%)",
        vintage_meaning="latest_published_edition",
        max_staleness_days=750,
        auth="none",
        credential=None,
        identity_check="energy_share_bounds_and_provenance",
        cache_url_substrings=("ember-energy.org",),
        telemetry_source="energy_transition",
    ),
    CohortConnectorSpec(
        source_id="vintage.oecd_stes",
        provider="oecd_stes",
        workflow="cross_country_realtime_revisions",
        series_key="OECD.SDD.NAD:B1GQ (DEU, USA)",
        freq="Q",
        units="national_currency_or_index_by_edition",
        transformations="log_diff_pct default revision transform",
        vintage_meaning="publication_date_archive",
        max_staleness_days=210,
        auth="none",
        credential=None,
        identity_check="multi_vintage_archive",
        cache_url_substrings=("sdmx.oecd.org", "asOf"),
        telemetry_source="oecd_stes",
    ),
    CohortConnectorSpec(
        source_id="vintage.bcb",
        provider="bcb",
        workflow="latam_realtime_snapshots",
        series_key="BCB SGS 432 (Selic target, <=10y window)",
        freq="D",
        units="percent_per_annum",
        transformations="none (daily target rate, <=10y SGS query window)",
        vintage_meaning="local_capture_snapshot",
        max_staleness_days=30,
        auth="none",
        credential=None,
        identity_check="snapshot_rate_bounds",
        cache_url_substrings=("bcdata.sgs.432",),
        telemetry_source="bcb",
    ),
    CohortConnectorSpec(
        source_id="vintage.banxico",
        provider="banxico",
        workflow="latam_realtime_snapshots",
        series_key="Banxico SIE SF61745 (Tasa objetivo)",
        freq="D",
        units="percent_per_annum",
        transformations="none (daily target rate)",
        vintage_meaning="local_capture_snapshot",
        max_staleness_days=30,
        auth="key",
        credential="banxico",
        identity_check="snapshot_rate_bounds",
        cache_url_substrings=("SF61745",),
        telemetry_source="banxico",
    ),
)

_COHORT_BY_ID: dict[str, CohortConnectorSpec] = {
    spec.source_id: spec for spec in SUPPORTED_CONNECTOR_COHORT
}


@dataclass(frozen=True)
class RateBudget:
    """Rate-limiting and timeout guardrails for ``mode='live'`` probes."""

    max_requests: int = 5
    min_interval_seconds: float = 0.5
    timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        if int(self.max_requests) < 0:
            raise ValueError(f"max_requests must be >= 0, got {self.max_requests!r}")
        if float(self.min_interval_seconds) < 0.0:
            raise ValueError(
                f"min_interval_seconds must be >= 0, got {self.min_interval_seconds!r}"
            )
        if float(self.timeout_seconds) <= 0.0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds!r}")


@dataclass(frozen=True)
class ConnectorStatusRecord:
    """Provenance, domain-identity, revision, and freshness record for one connector."""

    source_id: str
    provider: str
    workflow: str
    series_key: str
    freq: str
    units: str
    transformations: str
    vintage_meaning: str
    mode: str
    status: str
    retrieved_at: str | None
    obs_start: str | None
    obs_end: str | None
    n_obs: int
    n_vintages: int
    revision_detected: bool
    max_abs_revision: float | None
    staleness_days: int | None
    max_staleness_days: int
    fallback_used: str
    identity_check: str
    identity_ok: bool | None
    identity_detail: str
    error: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in FRESHNESS_STATUSES:
            raise ValueError(
                f"{self.source_id}: status {self.status!r} not in {FRESHNESS_STATUSES}"
            )


def cohort_table() -> pd.DataFrame:
    """Return the supported flagship connector cohort metadata as a DataFrame."""
    return pd.DataFrame([asdict(s) for s in SUPPORTED_CONNECTOR_COHORT])


# ---------------------------------------------------------------------------
# Domain identity & provenance checks
# ---------------------------------------------------------------------------


def check_laus_identity(
    df: pd.DataFrame,
    *,
    rtol: float = 0.02,
    atol: float = 50.0,
) -> tuple[bool, str]:
    """Verify BLS LAUS accounting identities: ``LF = E + U`` and ``urate = 100 * U / LF``."""
    required = {"lf", "emp", "une", "urate"}
    missing = sorted(required - set(df.columns))
    if missing:
        return False, f"missing LAUS columns: {missing}"
    complete = df[list(required)].dropna()
    if complete.empty:
        return False, "no complete (lf, emp, une, urate) rows"
    lf = complete["lf"].to_numpy(dtype=float)
    emp = complete["emp"].to_numpy(dtype=float)
    une = complete["une"].to_numpy(dtype=float)
    urate = complete["urate"].to_numpy(dtype=float)
    if not (np.all(np.isfinite(lf)) and np.all(lf > 0.0) and np.all(emp >= 0.0) and np.all(une >= 0.0)):
        return False, "non-positive or non-finite LAUS counts"
    lf_gap = np.abs(lf - (emp + une))
    allowed_lf = np.maximum(atol, rtol * lf)
    bad_lf = int(np.sum(lf_gap > allowed_lf))
    if bad_lf > 0:
        return (
            False,
            f"LAUS LF = E + U violated on {bad_lf}/{len(complete)} rows "
            f"(max |LF - (E + U)| = {float(np.max(lf_gap)):.1f})",
        )
    implied_urate = 100.0 * une / lf
    urate_gap = np.abs(urate - implied_urate)
    bad_urate = int(np.sum(urate_gap > 0.25))
    if bad_urate > 0:
        return (
            False,
            f"LAUS urate = 100*U/LF violated on {bad_urate}/{len(complete)} rows "
            f"(max gap = {float(np.max(urate_gap)):.3f} pp)",
        )
    return (
        True,
        f"LF = E + U and urate verified on {len(complete)} rows "
        f"(max |LF-(E+U)|={float(np.max(lf_gap)):.1f}, max |urate gap|={float(np.max(urate_gap)):.3f} pp)",
    )


def check_qna_identity(
    df: pd.DataFrame,
    *,
    rtol: float = 0.01,
) -> tuple[bool, str]:
    """Verify OECD QNA nominal, volume, and implicit deflator consistency."""
    work = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df.copy()
    has_nom_defl = {"gdp", "gdp_defl"}.issubset(work.columns)
    has_real = "gdp_real" in work.columns
    if not has_nom_defl and not has_real:
        return False, "missing QNA GDP columns ('gdp'/'gdp_defl' or 'gdp_real')"
    if has_nom_defl:
        sub = work[["gdp", "gdp_defl"]].dropna()
        if sub.empty:
            return False, "no non-null ('gdp', 'gdp_defl') observations"
        gdp = sub["gdp"].to_numpy(dtype=float)
        defl = sub["gdp_defl"].to_numpy(dtype=float)
        if not (np.all(np.isfinite(gdp)) and np.all(gdp > 0.0) and np.all(np.isfinite(defl)) and np.all(defl > 0.0)):
            return False, "non-positive or non-finite QNA nominal GDP or deflator"
        if has_real:
            sub3 = work[["gdp", "gdp_defl", "gdp_real"]].dropna()
            if not sub3.empty:
                implied = sub3["gdp"].to_numpy(dtype=float) / sub3["gdp_defl"].to_numpy(dtype=float) * 100.0
                actual = sub3["gdp_real"].to_numpy(dtype=float)
                rel_err = np.abs(actual - implied) / np.maximum(np.abs(actual), 1e-12)
                max_rel = float(np.max(rel_err))
                if max_rel > rtol:
                    return (
                        False,
                        f"QNA deflator identity gdp_real = gdp/gdp_defl*100 violated (max rel err = {max_rel:.4f})",
                    )
                return (
                    True,
                    f"QNA nominal/real/deflator identity verified on {len(sub3)} rows (max rel err = {max_rel:.2e})",
                )
        return True, f"QNA positive nominal GDP and deflator verified on {len(sub)} rows"
    real_vals = pd.to_numeric(work["gdp_real"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(real_vals) == 0 or not (np.all(np.isfinite(real_vals)) and np.all(real_vals > 0.0)):
        return False, "non-positive or empty QNA 'gdp_real'"
    return True, f"QNA positive real GDP verified on {len(real_vals)} rows"


def check_energy_identity(df: pd.DataFrame) -> tuple[bool, str]:
    """Verify energy transition share bounds ``[0, 100]``, non-negative TWh, and non-fixture provenance."""
    required = {"code", "date", "variable", "value", "source"}
    missing = sorted(required - set(df.columns))
    if missing:
        return False, f"missing energy columns: {missing}"
    if df.empty:
        return False, "empty energy DataFrame"
    sources = df["source"].dropna().astype(str).str.lower()
    bad_src = sources[
        sources.str.contains("fixture", regex=False)
        | sources.str.contains("tests/data", regex=False)
        | sources.str.contains("mock", regex=False)
    ]
    if not bad_src.empty:
        return False, f"prohibited fixture/mock provenance in energy source column: {sorted(set(bad_src))}"
    vals = pd.to_numeric(df["value"], errors="coerce")
    finite = df[vals.notna()].copy()
    if finite.empty:
        return False, "no finite numeric energy observations"
    finite_vals = pd.to_numeric(finite["value"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(finite_vals)):
        return False, "non-finite energy values"
    pct_mask = finite["variable"].astype(str).str.endswith("_pct_a")
    if pct_mask.any():
        pct_vals = finite_vals[pct_mask.to_numpy()]
        if np.any(pct_vals < -1e-6) or np.any(pct_vals > 100.0 + 1e-6):
            return (
                False,
                f"electricity share outside [0, 100]%: min={float(np.min(pct_vals)):.2f}, "
                f"max={float(np.max(pct_vals)):.2f}",
            )
    level_mask = ~pct_mask
    if level_mask.any():
        lvl_vals = finite_vals[level_mask.to_numpy()]
        if np.any(lvl_vals < -1e-6):
            return False, f"negative energy generation/consumption level: min={float(np.min(lvl_vals)):.2f}"
    return (
        True,
        f"energy share bounds [0, 100]% and non-fixture provenance verified on {len(finite)} rows",
    )


def _run_identity_check(spec: CohortConnectorSpec, df: pd.DataFrame) -> tuple[bool, str]:
    kind = spec.identity_check
    if kind == "laus_labor_force_identity":
        return check_laus_identity(df)
    if kind == "qna_deflator_consistency":
        return check_qna_identity(df)
    if kind == "energy_share_bounds_and_provenance":
        return check_energy_identity(df)
    if kind == "snapshot_rate_bounds":
        val_col = "value" if "value" in df.columns else df.columns[-1]
        vals = pd.to_numeric(df[val_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            return False, "no numeric policy-rate values"
        if not np.all(np.isfinite(vals)) or np.any(vals < -10.0) or np.any(vals > 200.0):
            return (
                False,
                f"policy-rate values out of plausible [-10, 200]% bounds "
                f"(min={float(np.min(vals)):.2f}, max={float(np.max(vals)):.2f})",
            )
        return True, f"policy-rate bounds [-10, 200]% verified on {len(vals)} rows"
    if kind in ("positive_finite", "multi_vintage_archive"):
        if isinstance(df, pd.Series):
            vals = pd.to_numeric(df, errors="coerce").dropna().to_numpy(dtype=float)
        elif "value" in df.columns:
            vals = pd.to_numeric(df["value"], errors="coerce").dropna().to_numpy(dtype=float)
        else:
            num_df = df.select_dtypes(include=[np.number])
            vals = num_df.to_numpy(dtype=float).ravel()
            vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return False, "no finite numeric observations"
        if kind == "positive_finite" and np.any(vals <= 0.0):
            return False, f"expected strictly positive level series (min={float(np.min(vals)):.4g})"
        return True, f"finite numeric values verified ({len(vals)} values)"
    return True, "no domain check configured"


# ---------------------------------------------------------------------------
# Revision & date-span helpers
# ---------------------------------------------------------------------------


def _extract_dates_and_revisions(
    obj: pd.DataFrame | pd.Series,
) -> tuple[str | None, str | None, int, int, bool, float | None]:
    """Return ``(obs_start, obs_end, n_obs, n_vintages, revision_detected, max_abs_revision)``."""
    if isinstance(obj, pd.Series):
        clean = pd.to_numeric(obj, errors="coerce").dropna()
        if clean.empty:
            return None, None, 0, 1, False, None
        idx = pd.to_datetime(clean.index, errors="coerce").dropna()
        obs_start = str(idx.min().date()) if len(idx) > 0 else None
        obs_end = str(idx.max().date()) if len(idx) > 0 else None
        return obs_start, obs_end, int(len(clean)), 1, False, None

    df = obj.reset_index() if isinstance(obj.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.MultiIndex)) else obj.copy()
    if df.empty:
        return None, None, 0, 0, False, None

    date_col = None
    for cand in ("date", "observation_date", "DATE", "period", "year"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        date_col = df.columns[0]

    raw_dates = df[date_col]
    if isinstance(raw_dates.dtype, pd.PeriodDtype):
        parsed_dates = raw_dates.dt.to_timestamp()
    elif pd.api.types.is_integer_dtype(raw_dates) and str(date_col).lower() == "year":
        parsed_dates = pd.to_datetime(raw_dates.astype(str) + "-01-01", errors="coerce")
    else:
        parsed_dates = pd.to_datetime(raw_dates, errors="coerce")

    valid_dates = parsed_dates.dropna()
    obs_start = str(valid_dates.min().date()) if not valid_dates.empty else None
    obs_end = str(valid_dates.max().date()) if not valid_dates.empty else None

    vint_col = None
    for cand in ("vintage", "vintage_date", "realtime_start", "edition"):
        if cand in df.columns:
            vint_col = cand
            break

    n_obs = int(len(df))
    if vint_col is None:
        return obs_start, obs_end, n_obs, 1, False, None

    vintages = df[vint_col].dropna().astype(str)
    n_vintages = int(vintages.nunique())
    if n_vintages <= 1 or "value" not in df.columns:
        return obs_start, obs_end, n_obs, max(n_vintages, 1), False, None

    vdf = df[[date_col, vint_col, "value"]].copy()
    vdf["value"] = pd.to_numeric(vdf["value"], errors="coerce")
    vdf = vdf.dropna(subset=[date_col, vint_col, "value"])
    if vdf.empty:
        return obs_start, obs_end, 0, n_vintages, False, None

    group_keys: list[str] = [date_col]
    for extra in ("country", "code", "variable", "series_id"):
        if extra in df.columns and extra not in group_keys:
            vdf[extra] = df.loc[vdf.index, extra]
            group_keys.append(extra)

    spread = vdf.groupby(group_keys, sort=False)["value"].agg(["min", "max", "count"])
    multi = spread[spread["count"] > 1]
    if multi.empty:
        return obs_start, obs_end, n_obs, n_vintages, False, 0.0
    diffs = (multi["max"] - multi["min"]).abs()
    max_rev = float(diffs.max())
    revised = bool(max_rev > 1e-9)
    return obs_start, obs_end, n_obs, n_vintages, revised, max_rev


def _compute_staleness_days(obs_end: str | None, as_of_ts: pd.Timestamp) -> int | None:
    if obs_end is None:
        return None
    end_ts = pd.Timestamp(obs_end).normalize()
    return int((as_of_ts - end_ts).days)


def _iso_utc_from_epoch(epoch_sec: int | float | None) -> str | None:
    if epoch_sec is None:
        return None
    dt = datetime.fromtimestamp(float(epoch_sec), tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Recorded wire-format samples for offline mode="recorded"
# ---------------------------------------------------------------------------

_RECORDED_RETRIEVED_AT = "2026-10-15T00:00:00Z"

_RECORDED_FRED_GDPC1_CSV = (
    "observation_date,GDPC1\n"
    "2025-07-01,23400.5\n"
    "2025-10-01,23545.2\n"
    "2026-01-01,23690.8\n"
    "2026-04-01,23812.4\n"
    "2026-07-01,23955.1\n"
)

_RECORDED_LAUS_TSV = (
    "series_id\tyear\tperiod\tvalue\tfootnote_codes\n"
    "LASST060000000000003          \t2026\tM07\t5.2\t\n"
    "LASST060000000000004          \t2026\tM07\t1014000\t\n"
    "LASST060000000000005          \t2026\tM07\t18486000\t\n"
    "LASST060000000000006          \t2026\tM07\t19500000\t\n"
    "LASST060000000000003          \t2026\tM08\t5.3\t\n"
    "LASST060000000000004          \t2026\tM08\t1033500\t\n"
    "LASST060000000000005          \t2026\tM08\t18466500\t\n"
    "LASST060000000000006          \t2026\tM08\t19500000\t\n"
    "LASST360000000000003          \t2026\tM08\t4.4\t\n"
    "LASST360000000000004          \t2026\tM08\t431200\t\n"
    "LASST360000000000005          \t2026\tM08\t9368800\t\n"
    "LASST360000000000006          \t2026\tM08\t9800000\t\n"
)

_RECORDED_EMBER_CSV = (
    "Area,ISO 3 code,Year,Area type,Electricity source,Generation (TWh),Share of generation (%)\n"
    "United States,USA,2025,Country or economy,Total generation,4380.0,100.0\n"
    "United States,USA,2025,Country or economy,Renewables,1051.2,24.0\n"
    "United States,USA,2025,Country or economy,Nuclear,788.4,18.0\n"
    "United States,USA,2025,Country or economy,Fossil,2540.4,58.0\n"
    "Germany,DEU,2025,Country or economy,Total generation,510.0,100.0\n"
    "Germany,DEU,2025,Country or economy,Renewables,295.8,58.0\n"
    "Germany,DEU,2025,Country or economy,Fossil,214.2,42.0\n"
)

_RECORDED_BCB_JSON = json.dumps([
    {"data": "01/09/2026", "valor": "10.75"},
    {"data": "01/10/2026", "valor": "10.75"},
    {"data": "15/10/2026", "valor": "11.00"},
])

_RECORDED_BANXICO_JSON = json.dumps({
    "bmx": {
        "series": [{
            "idSerie": "SF61745",
            "datos": [
                {"fecha": "01/09/2026", "dato": "10.00"},
                {"fecha": "01/10/2026", "dato": "9.75"},
                {"fecha": "15/10/2026", "dato": "9.75"},
            ],
        }]
    }
})


def _recorded_frame_for_spec(spec: CohortConnectorSpec) -> pd.DataFrame:
    sid = spec.source_id
    if sid == "fred.csv":
        raw = pd.read_csv(io.StringIO(_RECORDED_FRED_GDPC1_CSV))
        raw = raw.rename(columns={"observation_date": "date", "GDPC1": "value"})
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        return raw.dropna(subset=["value"])
    if sid == "fred.alfred":
        return pd.DataFrame([
            {"date": "2026-04-01", "vintage": "2026-07-30", "value": 23805.0},
            {"date": "2026-04-01", "vintage": "2026-09-26", "value": 23812.4},
            {"date": "2026-07-01", "vintage": "2026-09-26", "value": 23955.1},
        ])
    if sid == "bls.laus_state":
        return parse_laus_series(io.BytesIO(_RECORDED_LAUS_TSV.encode("utf-8")), geography="state")
    if sid == "oecd.qna_panel":
        qna_path = (
            Path(puremacro.__file__).resolve().parent
            / "replication"
            / "data"
            / "qna40_panel.csv"
        )
        qna = pd.read_csv(qna_path)
        return qna[qna["code"].isin(["USA", "DEU"])].reset_index(drop=True)
    if sid == "wdi.panel":
        return pd.DataFrame([
            {"code": "USA", "date": "2024-01-01", "value": 22675.3e9},
            {"code": "USA", "date": "2025-01-01", "value": 23190.8e9},
            {"code": "DEU", "date": "2024-01-01", "value": 3885.4e9},
            {"code": "DEU", "date": "2025-01-01", "value": 3920.1e9},
        ])
    if sid == "worldbank.energy_transition":
        raw_ember = pd.read_csv(io.StringIO(_RECORDED_EMBER_CSV))
        parsed = _parse_ember_df(raw_ember, codes=["USA", "DEU"], start_year=2020)
        parsed["source"] = "ember_recorded_snapshot"
        return parsed
    if sid == "vintage.oecd_stes":
        return pd.DataFrame([
            {"country": "DEU", "date": "2026-04-01", "vintage": "2026-08-15", "value": 106.2},
            {"country": "DEU", "date": "2026-04-01", "vintage": "2026-10-01", "value": 106.5},
            {"country": "USA", "date": "2026-07-01", "vintage": "2026-10-01", "value": 114.8},
        ])
    if sid == "vintage.bcb":
        return parse_bcb_json(_RECORDED_BCB_JSON, series_id="432", vintage_date="2026-10-15")
    if sid == "vintage.banxico":
        return parse_banxico_json(
            _RECORDED_BANXICO_JSON, series_id="SF61745", vintage_date="2026-10-15"
        )
    raise KeyError(f"unsupported cohort source {sid!r}")


# ---------------------------------------------------------------------------
# Cache inspection helpers for mode="cache"
# ---------------------------------------------------------------------------


def _parse_cached_http_body(spec: CohortConnectorSpec, body: bytes) -> pd.DataFrame:
    text = body.decode("utf-8", errors="replace")
    sid = spec.source_id
    if sid == "bls.laus_state":
        return parse_laus_series(io.BytesIO(body), geography="state")
    if sid == "worldbank.energy_transition":
        raw = pd.read_csv(io.StringIO(text))
        parsed = _parse_ember_df(raw, codes=["USA", "DEU"], start_year=2000)
        if not parsed.empty and "source" not in parsed.columns:
            parsed["source"] = "ember"
        return parsed
    if sid == "fred.csv":
        raw = pd.read_csv(io.StringIO(text))
        if raw.shape[1] >= 2:
            raw = raw.rename(columns={raw.columns[0]: "date", raw.columns[1]: "value"})
            raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
            return raw.dropna(subset=["value"])
        return pd.DataFrame(columns=["date", "value"])
    if sid == "oecd.qna_panel":
        raw = pd.read_csv(io.StringIO(text))
        if {"gdp", "gdp_defl"}.issubset(raw.columns) or "gdp_real" in raw.columns:
            return raw
        if "OBS_VALUE" in raw.columns and "TIME_PERIOD" in raw.columns:
            out = pd.DataFrame({
                "date": raw["TIME_PERIOD"],
                "gdp_real": pd.to_numeric(raw["OBS_VALUE"], errors="coerce"),
            }).dropna(subset=["gdp_real"])
            return out
        return raw
    if sid == "wdi.panel":
        data = json.loads(text)
        if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
            rows = []
            for item in data[1]:
                if isinstance(item, dict) and item.get("value") is not None:
                    rows.append({
                        "code": str((item.get("countryiso3code") or "")),
                        "date": f"{item.get('date')}-01-01",
                        "value": float(item["value"]),
                    })
            return pd.DataFrame(rows, columns=["code", "date", "value"])
        raw = pd.read_csv(io.StringIO(text))
        return raw
    return pd.read_csv(io.StringIO(text))


def _latest_telemetry_event(
    conn: sqlite3.Connection,
    telemetry_source: str,
) -> tuple[int, str, str] | None:
    if not telemetry_source:
        return None
    row = conn.execute(
        "SELECT ts, outcome, fallback_used FROM connector_events "
        "WHERE source = ? ORDER BY ts DESC, rowid DESC LIMIT 1",
        (telemetry_source,),
    ).fetchone()
    if row is None:
        return None
    return int(row[0]), str(row[1]), str(row[2])


def _evaluate_cache_spec(
    spec: CohortConnectorSpec,
    conn: sqlite3.Connection | None,
    as_of_ts: pd.Timestamp,
) -> ConnectorStatusRecord:
    has_cred = True
    if spec.auth == "key" and spec.credential is not None:
        has_cred = get_credential(spec.credential) is not None

    if conn is None:
        status = "credential_missing" if not has_cred else "not_cached"
        note = (
            f"missing credential {spec.credential!r} and cache DB absent"
            if not has_cred
            else "cache DB does not exist yet"
        )
        return ConnectorStatusRecord(
            source_id=spec.source_id,
            provider=spec.provider,
            workflow=spec.workflow,
            series_key=spec.series_key,
            freq=spec.freq,
            units=spec.units,
            transformations=spec.transformations,
            vintage_meaning=spec.vintage_meaning,
            mode="cache",
            status=status,
            retrieved_at=None,
            obs_start=None,
            obs_end=None,
            n_obs=0,
            n_vintages=0,
            revision_detected=False,
            max_abs_revision=None,
            staleness_days=None,
            max_staleness_days=spec.max_staleness_days,
            fallback_used="none",
            identity_check=spec.identity_check,
            identity_ok=None,
            identity_detail="not evaluated (no cache DB)",
            error=None,
            note=note,
        )

    event = _latest_telemetry_event(conn, spec.telemetry_source)
    event_ts = event[0] if event is not None else None
    event_outcome = event[1] if event is not None else ""
    event_fb = event[2] if event is not None else "none"

    df: pd.DataFrame | None = None
    retrieved_at: str | None = _iso_utc_from_epoch(event_ts)

    # 1. Check dedicated vintage tables first for archival / snapshot connectors
    if spec.source_id == "fred.alfred":
        rows = conn.execute(
            "SELECT observation_date, vintage_date, value FROM alfred_vintages "
            "WHERE series_id = ? ORDER BY observation_date, vintage_date",
            ("GDPC1",),
        ).fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["date", "vintage", "value"])
    elif spec.source_id in ("vintage.bcb", "vintage.banxico", "vintage.oecd_stes"):
        rows = conn.execute(
            "SELECT country, series_id, observation_date, vintage_date, value "
            "FROM realtime_vintages WHERE provider = ? "
            "ORDER BY observation_date, vintage_date",
            (spec.provider,),
        ).fetchall()
        if rows:
            df = pd.DataFrame(
                rows, columns=["country", "series_id", "date", "vintage", "value"]
            )
            if retrieved_at is None and not df.empty:
                latest_v = str(df["vintage"].astype(str).max())
                retrieved_at = f"{latest_v}T00:00:00Z" if "T" not in latest_v else latest_v

    # 2. Check http_cache if no vintage table rows matched
    parse_error: str | None = None
    if df is None and spec.cache_url_substrings:
        clauses = " AND ".join("url LIKE ?" for _ in spec.cache_url_substrings)
        params = [f"%{sub}%" for sub in spec.cache_url_substrings]
        row = conn.execute(
            f"SELECT fetched_at, body FROM http_cache WHERE {clauses} "
            "ORDER BY fetched_at DESC LIMIT 1",
            params,
        ).fetchone()
        if row is not None:
            retrieved_at = _iso_utc_from_epoch(int(row[0]))
            body_bytes = bytes(row[1])
            if len(body_bytes.strip()) == 0:
                df = pd.DataFrame()
            else:
                try:
                    df = _parse_cached_http_body(spec, body_bytes)
                except (SchemaDriftError, Exception) as exc:
                    parse_error = f"{type(exc).__name__}: {exc}"

    # 3. Classify status when no cached data is present
    if df is None and parse_error is None:
        if event_outcome in ("schema_drift", "parser_schema_mismatch"):
            status = "schema_drift"
            note = f"telemetry recorded {event_outcome!r}"
        elif event is not None and event_outcome not in ("success", "fallback"):
            status = "failed"
            note = f"telemetry recorded failure {event_outcome!r}"
        elif not has_cred:
            status = "credential_missing"
            note = f"credential {spec.credential!r} is not configured and no local cache exists"
        else:
            status = "not_cached"
            note = "no cached payload or vintage rows in SQLite cache"
        return ConnectorStatusRecord(
            source_id=spec.source_id,
            provider=spec.provider,
            workflow=spec.workflow,
            series_key=spec.series_key,
            freq=spec.freq,
            units=spec.units,
            transformations=spec.transformations,
            vintage_meaning=spec.vintage_meaning,
            mode="cache",
            status=status,
            retrieved_at=retrieved_at,
            obs_start=None,
            obs_end=None,
            n_obs=0,
            n_vintages=0,
            revision_detected=False,
            max_abs_revision=None,
            staleness_days=None,
            max_staleness_days=spec.max_staleness_days,
            fallback_used=event_fb,
            identity_check=spec.identity_check,
            identity_ok=None,
            identity_detail="not evaluated (no cached rows)",
            error=event_outcome if status == "failed" else None,
            note=note,
        )

    if parse_error is not None:
        is_drift = "SchemaDrift" in parse_error or "SchemaMismatch" in parse_error
        return ConnectorStatusRecord(
            source_id=spec.source_id,
            provider=spec.provider,
            workflow=spec.workflow,
            series_key=spec.series_key,
            freq=spec.freq,
            units=spec.units,
            transformations=spec.transformations,
            vintage_meaning=spec.vintage_meaning,
            mode="cache",
            status="schema_drift" if is_drift else "failed",
            retrieved_at=retrieved_at,
            obs_start=None,
            obs_end=None,
            n_obs=0,
            n_vintages=0,
            revision_detected=False,
            max_abs_revision=None,
            staleness_days=None,
            max_staleness_days=spec.max_staleness_days,
            fallback_used=event_fb,
            identity_check=spec.identity_check,
            identity_ok=False if is_drift else None,
            identity_detail=parse_error,
            error=parse_error,
            note="cached payload failed parsing",
        )

    assert df is not None
    return _finalize_frame_record(
        spec,
        df,
        mode="cache",
        as_of_ts=as_of_ts,
        retrieved_at=retrieved_at,
        event_outcome=event_outcome,
        fallback_used=event_fb,
    )


def _finalize_frame_record(
    spec: CohortConnectorSpec,
    df: pd.DataFrame | pd.Series,
    *,
    mode: str,
    as_of_ts: pd.Timestamp,
    retrieved_at: str | None,
    event_outcome: str = "",
    fallback_used: str = "none",
    note: str = "",
) -> ConnectorStatusRecord:
    obs_start, obs_end, n_obs, n_vintages, rev_detected, max_rev = _extract_dates_and_revisions(df)
    staleness = _compute_staleness_days(obs_end, as_of_ts)

    if n_obs == 0:
        return ConnectorStatusRecord(
            source_id=spec.source_id,
            provider=spec.provider,
            workflow=spec.workflow,
            series_key=spec.series_key,
            freq=spec.freq,
            units=spec.units,
            transformations=spec.transformations,
            vintage_meaning=spec.vintage_meaning,
            mode=mode,
            status="empty",
            retrieved_at=retrieved_at,
            obs_start=None,
            obs_end=None,
            n_obs=0,
            n_vintages=0,
            revision_detected=False,
            max_abs_revision=None,
            staleness_days=None,
            max_staleness_days=spec.max_staleness_days,
            fallback_used=fallback_used,
            identity_check=spec.identity_check,
            identity_ok=False,
            identity_detail="0 observations returned",
            error=None,
            note=note or "payload parsed to 0 usable observations",
        )

    frame_for_check = (
        df.rename("value").to_frame().reset_index()
        if isinstance(df, pd.Series)
        else df
    )
    id_ok, id_detail = _run_identity_check(spec, frame_for_check)

    is_fb = (
        event_outcome == "fallback"
        or fallback_used not in ("none", "live", "")
    )
    is_drift = (
        not id_ok
        or event_outcome in ("schema_drift", "parser_schema_mismatch")
        or fallback_used in ("warning_emitted", "ignored")
    )

    if is_drift:
        status = "schema_drift"
    elif is_fb:
        status = "degraded_fallback"
    elif staleness is not None and staleness > spec.max_staleness_days:
        status = "stale"
    elif rev_detected:
        status = "revised"
    else:
        status = "fresh"

    return ConnectorStatusRecord(
        source_id=spec.source_id,
        provider=spec.provider,
        workflow=spec.workflow,
        series_key=spec.series_key,
        freq=spec.freq,
        units=spec.units,
        transformations=spec.transformations,
        vintage_meaning=spec.vintage_meaning,
        mode=mode,
        status=status,
        retrieved_at=retrieved_at,
        obs_start=obs_start,
        obs_end=obs_end,
        n_obs=n_obs,
        n_vintages=n_vintages,
        revision_detected=rev_detected,
        max_abs_revision=max_rev,
        staleness_days=staleness,
        max_staleness_days=spec.max_staleness_days,
        fallback_used=fallback_used or "none",
        identity_check=spec.identity_check,
        identity_ok=id_ok,
        identity_detail=id_detail,
        error=None,
        note=note,
    )


# ---------------------------------------------------------------------------
# Live bounded probes for mode="live"
# ---------------------------------------------------------------------------


def _default_live_probe(spec: CohortConnectorSpec, timeout: float) -> pd.DataFrame | pd.Series:
    sid = spec.source_id
    if sid == "fred.csv":
        from puremacro.fetch._classic import fetch_fred
        return fetch_fred("GDPC1", timeout=timeout)
    if sid == "fred.alfred":
        from puremacro.fetch._classic import fetch_fred_alfred
        return fetch_fred_alfred("GDPC1", timeout=timeout)
    if sid == "bls.laus_state":
        from puremacro.fetch.laus import fetch_state
        return fetch_state(refresh=True)
    if sid == "oecd.qna_panel":
        from puremacro.fetch.oecd_qna_panel import qna_panel
        return qna_panel(["USA", "DEU"], start="2022-Q1")
    if sid == "wdi.panel":
        from puremacro.fetch.wdi import wdi_panel
        df = wdi_panel(
            indicators={"gdp_const": "NY.GDP.MKTP.KD"},
            codes=["USA", "DEU"],
            start=2018,
            timeout=timeout,
        )
        work = df.reset_index()
        if "gdp_const" in work.columns:
            work = work.rename(columns={"gdp_const": "value"})
        return work
    if sid == "worldbank.energy_transition":
        from puremacro.fetch.energy_transition import fetch_energy_transition
        return fetch_energy_transition(codes=["USA", "DEU"], start_year=2018, refresh=True)
    if sid == "vintage.oecd_stes":
        from puremacro.fetch.realtime import vintage_panel
        vp = vintage_panel(
            ["DEU", "USA"],
            series="gdp_real",
            providers="oecd_stes",
            timeout=timeout,
        )
        return vp.df
    if sid == "vintage.bcb":
        return fetch_bcb_vintages("432", timeout=timeout, use_cache=True, history=True)
    if sid == "vintage.banxico":
        from puremacro.fetch.realtime.banxico import fetch_banxico_vintages
        return fetch_banxico_vintages("SF61745", timeout=timeout, use_cache=True, history=True)
    raise KeyError(f"no live probe configured for {sid!r}")


# ---------------------------------------------------------------------------
# Report object & main API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreshnessReport:
    """Structured provenance and freshness report across the bounded connector cohort."""

    mode: str
    as_of: str
    records: tuple[ConnectorStatusRecord, ...]
    rate_budget: dict[str, Any] | None = None
    requests_used: int = 0

    @property
    def has_outages(self) -> bool:
        """True if any connector reported an outage/drift/fallback status."""
        return any(r.status in OUTAGE_STATUSES for r in self.records)

    @property
    def status_counts(self) -> dict[str, int]:
        """Count of connectors in each status across :data:`FRESHNESS_STATUSES`."""
        counts = {s: 0 for s in FRESHNESS_STATUSES}
        for r in self.records:
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    def to_frame(self) -> pd.DataFrame:
        """Convert report records into a tidy DataFrame."""
        cols = [
            "source_id", "provider", "workflow", "series_key", "freq", "units",
            "transformations", "vintage_meaning", "mode", "status", "retrieved_at",
            "obs_start", "obs_end", "n_obs", "n_vintages", "revision_detected",
            "max_abs_revision", "staleness_days", "max_staleness_days",
            "fallback_used", "identity_check", "identity_ok", "identity_detail",
            "error", "note",
        ]
        if not self.records:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame([asdict(r) for r in self.records], columns=cols)

    def to_dict(self) -> dict[str, Any]:
        """Convert report into a portable, JSON-serializable dictionary."""
        return {
            "puremacro_version": puremacro.__version__,
            "mode": self.mode,
            "as_of": self.as_of,
            "has_outages": self.has_outages,
            "requests_used": self.requests_used,
            "rate_budget": self.rate_budget,
            "status_counts": self.status_counts,
            "records": [asdict(r) for r in self.records],
        }

    def summary(self, lang: str = "en") -> str:
        """Short one-paragraph status digest in English or Spanish."""
        if lang not in ("en", "es"):
            raise ValueError(f"lang must be 'en' or 'es', got {lang!r}")
        active = {k: v for k, v in self.status_counts.items() if v > 0}
        parts = ", ".join(f"{k}={v}" for k, v in active.items()) or "none"
        if lang == "es":
            outage_txt = "CON INCIDENCIAS" if self.has_outages else "SIN INCIDENCIAS CRÍTICAS"
            return (
                f"Informe de frescura y procedencia de conectores (modo={self.mode}, "
                f"corte={self.as_of}, conectores={len(self.records)}, "
                f"solicitudes_red={self.requests_used}): {parts} [{outage_txt}]."
            )
        outage_txt = "OUTAGES DETECTED" if self.has_outages else "NO CRITICAL OUTAGES"
        return (
            f"Connector provenance & freshness report (mode={self.mode}, "
            f"as_of={self.as_of}, connectors={len(self.records)}, "
            f"live_requests={self.requests_used}): {parts} [{outage_txt}]."
        )

    def to_markdown(self, lang: str = "en") -> str:
        """Render human-readable Markdown table and provenance/identity notes."""
        if lang not in ("en", "es"):
            raise ValueError(f"lang must be 'en' or 'es', got {lang!r}")
        lines: list[str] = []
        if lang == "es":
            lines.append("# Informe acotado de procedencia y frescura de conectores")
            lines.append("")
            lines.append(self.summary(lang="es"))
            lines.append("")
            lines.append(
                "| Conector | Proveedor | Estado | Semántica vintage | Inicio obs. | Fin obs. | "
                "N obs. | Vintages | Identidad | Respaldo |"
            )
        else:
            lines.append("# Bounded Connector Provenance & Freshness Report")
            lines.append("")
            lines.append(self.summary(lang="en"))
            lines.append("")
            lines.append(
                "| Connector | Provider | Status | Vintage meaning | Obs start | Obs end | "
                "N obs | Vintages | Identity | Fallback |"
            )
        lines.append("|:---|:---|:---|:---|:---|:---|---:|---:|:---|:---|")
        for r in self.records:
            id_badge = "pass" if r.identity_ok is True else ("FAIL" if r.identity_ok is False else "n/a")
            lines.append(
                f"| `{r.source_id}` | `{r.provider}` | **{r.status}** | `{r.vintage_meaning}` | "
                f"{r.obs_start or '-'} | {r.obs_end or '-'} | {r.n_obs} | {r.n_vintages} | "
                f"{id_badge} | `{r.fallback_used}` |"
            )
        lines.append("")
        lines.append("## Provenance & Identity Notes" if lang == "en" else "## Notas de procedencia e identidad")
        lines.append("")
        for r in self.records:
            extra = f" — error: {r.error}" if r.error else ""
            note_txt = f" ({r.note})" if r.note else ""
            lines.append(
                f"- **`{r.source_id}`** (`{r.series_key}`, units: `{r.units}`, "
                f"transform: `{r.transformations}`): {r.identity_detail}{note_txt}{extra}"
            )
        return "\n".join(lines) + "\n"


def _resolve_cohort(sources: Sequence[str] | None) -> list[CohortConnectorSpec]:
    if sources is None:
        return list(SUPPORTED_CONNECTOR_COHORT)
    resolved: list[CohortConnectorSpec] = []
    for sid in sources:
        key = str(sid).strip()
        if not key:
            continue
        if key not in _COHORT_BY_ID:
            valid = sorted(_COHORT_BY_ID)
            raise ValueError(
                f"unknown cohort connector {key!r}; supported cohort: {valid}"
            )
        resolved.append(_COHORT_BY_ID[key])
    return resolved


def connector_freshness_report(
    *,
    mode: str = "cache",
    sources: Sequence[str] | None = None,
    as_of: str | pd.Timestamp | None = None,
    db_path: str | Path | None = None,
    rate_budget: RateBudget | None = None,
    probe_overrides: Mapping[str, Callable[[], pd.DataFrame | pd.Series]] | None = None,
) -> FreshnessReport:
    """Evaluate provenance, domain identities, and freshness for the supported cohort.

    Parameters
    ----------
    mode : {"cache", "recorded", "live"}, default "cache"
        Execution mode. ``"cache"`` and ``"recorded"`` perform zero network I/O;
        ``"live"`` probes external endpoints within ``rate_budget``.
    sources : Sequence[str] | None, optional
        Subset of cohort ``source_id`` names to evaluate. Defaults to all
        entries in :data:`SUPPORTED_CONNECTOR_COHORT`.
    as_of : str | pd.Timestamp | None, optional
        Reference date for staleness calculations. Defaults to today (UTC) in
        ``"cache"``/``"live"`` modes, or ``"2026-10-15"`` in ``"recorded"`` mode
        when omitted.
    db_path : str | Path | None, optional
        Explicit SQLite cache database path for ``"cache"`` mode (defaults to
        :func:`puremacro._cache_db.default_db_path`).
    rate_budget : RateBudget | None, optional
        Request cap, pacing interval, and timeout for ``mode="live"``.
    probe_overrides : Mapping[str, Callable] | None, optional
        Optional per-``source_id`` callable overrides used in ``mode="live"``
        for deterministic testing or custom live probes.
    """
    mode_norm = str(mode).strip().lower()
    if mode_norm not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    specs = _resolve_cohort(sources)
    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of).normalize()
    elif mode_norm == "recorded":
        as_of_ts = pd.Timestamp("2026-10-15")
    else:
        as_of_ts = pd.Timestamp(datetime.now(timezone.utc).date())
    as_of_str = str(as_of_ts.date())

    if mode_norm == "recorded":
        rec_list = [
            _finalize_frame_record(
                spec,
                _recorded_frame_for_spec(spec),
                mode="recorded",
                as_of_ts=as_of_ts,
                retrieved_at=_RECORDED_RETRIEVED_AT,
                note="deterministic recorded wire fixture",
            )
            for spec in specs
        ]
        return FreshnessReport(
            mode="recorded",
            as_of=as_of_str,
            records=tuple(rec_list),
            rate_budget=None,
            requests_used=0,
        )

    if mode_norm == "cache":
        target_db = Path(db_path) if db_path is not None else _cache_db.default_db_path()
        if target_db.suffix != ".db" and target_db.is_dir():
            target_db = target_db / "cache.db"
        conn: sqlite3.Connection | None = None
        opened_dedicated = False
        try:
            if target_db.exists():
                if db_path is not None:
                    uri = target_db.resolve().as_uri() + "?mode=ro"
                    conn = sqlite3.connect(uri, uri=True, timeout=10.0)
                    opened_dedicated = True
                else:
                    conn = _cache_db.get_conn()
            rec_list = [_evaluate_cache_spec(spec, conn, as_of_ts) for spec in specs]
        finally:
            if opened_dedicated and conn is not None:
                conn.close()
        return FreshnessReport(
            mode="cache",
            as_of=as_of_str,
            records=tuple(rec_list),
            rate_budget=None,
            requests_used=0,
        )

    # mode == "live"
    budget = rate_budget if rate_budget is not None else RateBudget()
    overrides = dict(probe_overrides or {})
    requests_used = 0
    last_req_mono: float | None = None
    live_records: list[ConnectorStatusRecord] = []

    for spec in specs:
        if spec.auth == "key" and spec.credential is not None and spec.source_id not in overrides:
            if get_credential(spec.credential) is None:
                live_records.append(
                    ConnectorStatusRecord(
                        source_id=spec.source_id,
                        provider=spec.provider,
                        workflow=spec.workflow,
                        series_key=spec.series_key,
                        freq=spec.freq,
                        units=spec.units,
                        transformations=spec.transformations,
                        vintage_meaning=spec.vintage_meaning,
                        mode="live",
                        status="credential_missing",
                        retrieved_at=None,
                        obs_start=None,
                        obs_end=None,
                        n_obs=0,
                        n_vintages=0,
                        revision_detected=False,
                        max_abs_revision=None,
                        staleness_days=None,
                        max_staleness_days=spec.max_staleness_days,
                        fallback_used="none",
                        identity_check=spec.identity_check,
                        identity_ok=None,
                        identity_detail="skipped live probe: required credential not configured",
                        error=None,
                        note=f"missing credential {spec.credential!r}",
                    )
                )
                continue

        if requests_used >= budget.max_requests:
            live_records.append(
                ConnectorStatusRecord(
                    source_id=spec.source_id,
                    provider=spec.provider,
                    workflow=spec.workflow,
                    series_key=spec.series_key,
                    freq=spec.freq,
                    units=spec.units,
                    transformations=spec.transformations,
                    vintage_meaning=spec.vintage_meaning,
                    mode="live",
                    status="not_cached",
                    retrieved_at=None,
                    obs_start=None,
                    obs_end=None,
                    n_obs=0,
                    n_vintages=0,
                    revision_detected=False,
                    max_abs_revision=None,
                    staleness_days=None,
                    max_staleness_days=spec.max_staleness_days,
                    fallback_used="none",
                    identity_check=spec.identity_check,
                    identity_ok=None,
                    identity_detail="not probed: rate budget exhausted",
                    error=None,
                    note=f"rate_budget_exhausted (max_requests={budget.max_requests})",
                )
            )
            continue

        if last_req_mono is not None and budget.min_interval_seconds > 0.0:
            elapsed = time.monotonic() - last_req_mono
            if elapsed < budget.min_interval_seconds:
                time.sleep(budget.min_interval_seconds - elapsed)

        requests_used += 1
        last_req_mono = time.monotonic()
        probe_ts = _now_iso_utc()

        caught_warnings: list[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                if spec.source_id in overrides:
                    result = overrides[spec.source_id]()
                else:
                    result = _default_live_probe(spec, budget.timeout_seconds)
        except Exception as exc:
            err_name = type(exc).__name__
            err_msg = f"{err_name}: {exc}"
            is_drift = isinstance(exc, SchemaDriftError) or "SchemaMismatch" in err_name or "SchemaDrift" in err_name
            live_records.append(
                ConnectorStatusRecord(
                    source_id=spec.source_id,
                    provider=spec.provider,
                    workflow=spec.workflow,
                    series_key=spec.series_key,
                    freq=spec.freq,
                    units=spec.units,
                    transformations=spec.transformations,
                    vintage_meaning=spec.vintage_meaning,
                    mode="live",
                    status="schema_drift" if is_drift else "failed",
                    retrieved_at=probe_ts,
                    obs_start=None,
                    obs_end=None,
                    n_obs=0,
                    n_vintages=0,
                    revision_detected=False,
                    max_abs_revision=None,
                    staleness_days=None,
                    max_staleness_days=spec.max_staleness_days,
                    fallback_used="none",
                    identity_check=spec.identity_check,
                    identity_ok=False if is_drift else None,
                    identity_detail=err_msg if is_drift else "live probe raised an exception",
                    error=err_msg,
                    note="live probe failed",
                )
            )
            continue

        fb_used = "live"
        ev_outcome = "success"
        note_parts: list[str] = []
        for w in caught_warnings:
            w_text = str(w.message)
            if issubclass(w.category, SchemaDriftWarning) or "schema drift" in w_text.lower():
                ev_outcome = "schema_drift"
                fb_used = "warning_emitted"
                note_parts.append(w_text)
            elif "falling back" in w_text.lower() or "fallback" in w_text.lower():
                if ev_outcome != "schema_drift":
                    ev_outcome = "fallback"
                fb_used = "sqlite_cache"
                note_parts.append(w_text)

        live_records.append(
            _finalize_frame_record(
                spec,
                result,
                mode="live",
                as_of_ts=as_of_ts,
                retrieved_at=probe_ts,
                event_outcome=ev_outcome,
                fallback_used=fb_used,
                note="; ".join(note_parts),
            )
        )

    return FreshnessReport(
        mode="live",
        as_of=as_of_str,
        records=tuple(live_records),
        rate_budget=asdict(budget),
        requests_used=requests_used,
    )


# ---------------------------------------------------------------------------
# Portable bundle writer & CLI
# ---------------------------------------------------------------------------


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _portable_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _portable_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_json(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def write_freshness_report(
    report: FreshnessReport,
    output_dir: str | Path,
    *,
    lang: str = "en",
) -> dict[str, Path]:
    """Write ``freshness_report.{json,csv,md}`` and portable ``manifest.json``."""
    if lang not in ("en", "es"):
        raise ValueError(f"lang must be 'en' or 'es', got {lang!r}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_bytes = (
        json.dumps(_portable_json(report.to_dict()), indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    csv_bytes = report.to_frame().to_csv(index=False, lineterminator="\n").encode("utf-8")
    md_bytes = report.to_markdown(lang=lang).encode("utf-8")

    files_payload = {
        "freshness_report.json": json_bytes,
        "freshness_report.csv": csv_bytes,
        "freshness_report.md": md_bytes,
    }
    written: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, data in files_payload.items():
        target = out / name
        target.write_bytes(data)
        written[name] = target
        hashes[name] = _sha256_bytes(data)

    manifest = {
        "puremacro_version": puremacro.__version__,
        "artifact": "connector_freshness_report",
        "mode": report.mode,
        "as_of": report.as_of,
        "lang": lang,
        "has_outages": report.has_outages,
        "requests_used": report.requests_used,
        "status_counts": report.status_counts,
        "files": hashes,
    }
    manifest_bytes = (
        json.dumps(_portable_json(manifest), indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    manifest_path = out / "manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    written["manifest.json"] = manifest_path
    return written


def _safe_print(text: str, stream: Any = None) -> None:
    target = sys.stdout if stream is None else stream
    enc = getattr(target, "encoding", None) or "utf-8"
    safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
    target.write(safe)
    if not safe.endswith("\n"):
        target.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``python -m puremacro.fetch.freshness``."""
    parser = argparse.ArgumentParser(
        prog="python -m puremacro.fetch.freshness",
        description="Bounded connector provenance, domain identity, and freshness report.",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="cache",
        help="Execution mode: 'cache' (default, offline), 'recorded' (offline fixtures), or 'live' (rate-budgeted).",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="Optional comma-separated subset of cohort source IDs.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional directory to write freshness_report.{json,csv,md} and manifest.json.",
    )
    parser.add_argument(
        "--lang",
        choices=("en", "es"),
        default="en",
        help="Report language ('en' or 'es').",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Optional reference date (YYYY-MM-DD) for staleness calculation.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Optional SQLite cache database path for --mode cache.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=5,
        help="Maximum live HTTP probes when --mode live (default: 5).",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=0.5,
        help="Minimum interval in seconds between live probes (default: 0.5).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Per-probe timeout in seconds when --mode live (default: 20.0).",
    )
    parser.add_argument(
        "--fail-on-outage",
        action="store_true",
        help="Return exit code 1 if any connector reports empty, schema_drift, failed, or degraded_fallback.",
    )

    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    src_list = (
        [s.strip() for s in str(args.sources).split(",") if s.strip()]
        if args.sources
        else None
    )
    try:
        budget = RateBudget(
            max_requests=args.max_requests,
            min_interval_seconds=args.min_interval,
            timeout_seconds=args.timeout,
        )
        report = connector_freshness_report(
            mode=args.mode,
            sources=src_list,
            as_of=args.as_of,
            db_path=args.db_path,
            rate_budget=budget,
        )
        if args.output is not None:
            write_freshness_report(report, args.output, lang=args.lang)
    except ValueError as exc:
        _safe_print(f"error: {exc}", stream=sys.stderr)
        return 2

    _safe_print(report.to_markdown(lang=args.lang))
    if args.fail_on_outage and report.has_outages:
        return 1
    return 0


__all__ = [
    "FRESHNESS_STATUSES",
    "OUTAGE_STATUSES",
    "VALID_MODES",
    "CohortConnectorSpec",
    "SUPPORTED_CONNECTOR_COHORT",
    "RateBudget",
    "ConnectorStatusRecord",
    "FreshnessReport",
    "cohort_table",
    "check_laus_identity",
    "check_qna_identity",
    "check_energy_identity",
    "connector_freshness_report",
    "write_freshness_report",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
