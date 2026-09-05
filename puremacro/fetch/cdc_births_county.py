"""NCHS county-year births fetcher (longest series — public-use NBER mirror).

Why this exists
---------------
NCHS Natality public-use microdata is the longest fully-public source of
US births at the county level. Coverage and column names vary by year:

  1968–1971: 50% sample (recwt=2). NCHS state/county codes (no FIPS).
  1972–1981: full file. Mostly NCHS codes; FIPS county absent or partial.
  1982–2002: full file with `cntyrfip` (residence FIPS, only counties
             with ≥100,000 population are identified; smaller counties
             are masked to 0). This is the workhorse window for county
             panels.
  2003–2004: column rename to `mrcntyfips`. Same suppression rule.
  2005+    : public-use files redact county of residence entirely.
             Data is available only via the NCHS Restricted-Data Center.

So the longest publicly-aggregable county-year series is **1982–2004**
(plus 1972–1981 if you accept NCHS-coded counties).

Two source paths
----------------
1) **Local fixed-width PUB** (preferred when present). The Fertility
   project has cached raw zips at
   `Fertility/data/United States/nvss_natality_raw/Nat[YYYY].zip` and
   their accompanying `natality[YYYY].dct` dictionary files (1982-1993).
   We parse the dct to find `cntyrfip` and `recwt` column positions
   (the schema shifts in 1990) and read fixed-width directly. Disk-
   bound (~1-3 min/year), no network.

2) **NBER CSV streaming** (fallback for years not on disk). We stream
   `https://data.nber.org/natality/{year}/natl{year}.csv` line by line;
   ~1.4 GB/year and bandwidth-bound (NBER is slow, often ~15-20 min/year).

Per-year aggregations are cached as small CSVs under
`data/raw/nchs_county_year/`, so subsequent runs are instant.

Cooperation with the Fertility project
--------------------------------------
Aggregated artifact is written to
`Fertility/data/PROCESSED/nchs_county_year_births.csv` so both projects
read from one canonical file (Fertility currently only has state×month
aggregates from `nber_state_month_births.csv`).

Public API:
    fetch(years=range(1982, 2005), refresh=False) -> pd.DataFrame
        Columns: county_fips, year, births, source
    cache_root() -> Path
"""
from __future__ import annotations

import csv
import io
import logging
import re
import zipfile
from collections import defaultdict
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


_NBER_URL = "https://data.nber.org/natality/{year}/natl{year}.csv"

# Per-year aggregations are cached as small CSVs (a few hundred KB each)
# rather than the source ~1.4 GB microdata.
_LOCAL_CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "nchs_county_year"

#: Root of the separate project's data tree this module reads when it is
#: present. It was an absolute path naming one person's Google Drive account,
#: so on every other machine it silently could not resolve. `PUREMACRO_MAV_ROOT`
#: is the same variable `puremacro/examples/*.py` and `fetch/wui*.py` read.
_FERTILITY_ROOT = Path(os.environ.get(
    "PUREMACRO_FERTILITY_ROOT",
    Path(os.environ.get("PUREMACRO_MAV_ROOT",
                        Path.home() / "Library" / "CloudStorage" / "My Drive"))
    .parent / "Fertility" / "data"))

# Canonical artifact location shared with the Fertility project.
_FERTILITY_DATA = _FERTILITY_ROOT / "PROCESSED"
_FERTILITY_ARTIFACT = _FERTILITY_DATA / "nchs_county_year_births.csv"
_COMBINED_ARTIFACT = _FERTILITY_DATA / "county_year_births.csv"

# Local raw zips + dct dictionary files cached by the Fertility project.
_FERTILITY_NATALITY_RAW = _FERTILITY_ROOT / "United States" / "nvss_natality_raw"

# Residence-county FIPS columns vary by year. 1982-2002 ship a single
# 5-char `cntyrfip` (state+county FIPS). 2003-2004 split it: `mrcntyfips`
# is the 3-char county code only and `mrstate` is the 2-letter abbreviation
# of the state of residence — they must be recombined.
_COUNTY_FIPS_CANDIDATES = ("cntyrfip", "ocntyfips_legacy", "cntocfip")
_COUNTY_NCHS_CANDIDATES = ("cntyres",)
# 2003+: split-FIPS pair. We probe both.
_SPLIT_STATE_ABBR_CANDIDATES = ("mrstate",)
_SPLIT_COUNTY_FIPS_CANDIDATES = ("mrcntyfips",)
_WEIGHT_CANDIDATES = ("recwt",)

# Reverse of the state-abbr → FIPS map embedded in build_subnational_panel.
_ABBR_TO_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12",
    "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
    "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
    "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56",
}

# NCHS↔FIPS crosswalk built once from 1982-1988 NBER PUBs.
_CROSSWALK_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "nchs_to_fips_county.csv"
_NCHS_TO_FIPS: dict[str, str] | None = None


def _load_nchs_to_fips() -> dict[str, str]:
    global _NCHS_TO_FIPS
    if _NCHS_TO_FIPS is None:
        if not _CROSSWALK_PATH.exists():
            logger.warning("NCHS↔FIPS crosswalk not found at %s; pre-1982 years will be skipped", _CROSSWALK_PATH)
            _NCHS_TO_FIPS = {}
            return _NCHS_TO_FIPS
        cw = pd.read_csv(_CROSSWALK_PATH, dtype={"cntyres": str, "cntyrfip": str})
        _NCHS_TO_FIPS = dict(zip(cw["cntyres"], cw["cntyrfip"]))
    return _NCHS_TO_FIPS


def cache_root() -> Path:
    return _LOCAL_CACHE


def _http_error() -> type[BaseException]:
    """``requests.HTTPError`` if requests is installed, else something unraisable.

    An ``except`` clause is evaluated even when nothing raised, so it cannot name
    a module that may be absent.
    """
    try:
        import requests
    except ImportError:
        class _Never(Exception):
            pass
        return _Never
    return requests.HTTPError


def _stream_url(url: str, *, timeout: int = 60):
    """Yield decoded CSV lines from a URL without buffering the whole file."""
    import requests            # deferred; see the module docstring

    resp = requests.get(
        url,
        headers={"User-Agent": "uncertainty_examples/1.0 (research@itam.mx)"},
        timeout=timeout,
        stream=True,
    )
    resp.raise_for_status()
    # iter_lines yields bytes; decode latin-1 (NBER CSVs are mostly ASCII
    # with the occasional non-UTF-8 byte in name-string fields).
    for line in resp.iter_lines(decode_unicode=False, chunk_size=1024 * 1024):
        if line:
            yield line.decode("latin-1", errors="replace")


_DCT_RE = re.compile(
    r"_column\(\s*(\d+)\s*\)\s+\S+\s+(\w+)\s+%(\d+)[fs]\s+\"([^\"]*)\""
)


def _parse_dct(dct_path: Path) -> dict[str, tuple[int, int]]:
    """Parse an NBER .dct dictionary into {variable: (start_col, length)}.

    `_column(N)  type  varname  %Wf|s "label"` — N is 1-based start column,
    W is the field width.
    """
    cols: dict[str, tuple[int, int]] = {}
    for line in dct_path.read_text(encoding="latin-1").splitlines():
        m = _DCT_RE.search(line)
        if m:
            start = int(m.group(1))
            varname = m.group(2)
            width = int(m.group(3))
            # Some dct files declare overlapping aliases (e.g. stoccfip is the
            # first 2 chars of cntocfip). First entry wins.
            cols.setdefault(varname, (start, width))
    return cols


def _aggregate_year_local(year: int) -> Optional[pd.DataFrame]:
    """Parse the locally-cached Nat[YYYY].zip + dct → (county_fips, births).

    Returns None if either the zip or the dct file is missing.
    """
    zip_path = _FERTILITY_NATALITY_RAW / f"Nat{year}.zip"
    dct_path = _FERTILITY_NATALITY_RAW / f"natality{year}.dct"
    if not zip_path.exists() or not dct_path.exists():
        return None

    cols = _parse_dct(dct_path)
    if "cntyrfip" not in cols:
        logger.warning("year %d: dct has no cntyrfip; falling back to network", year)
        return None
    fips_start, fips_len = cols["cntyrfip"]
    if "recwt" in cols:
        wt_start, wt_len = cols["recwt"]
    else:
        wt_start, wt_len = (None, None)

    counts: dict[str, int] = defaultdict(int)
    n_rows = 0
    n_unmasked = 0
    logger.info("year %d: parsing local PUB %s", year, zip_path.name)
    with zipfile.ZipFile(zip_path) as zf:
        member = zf.namelist()[0]
        with zf.open(member, "r") as fh:
            for raw in fh:
                n_rows += 1
                line = raw.decode("latin-1", errors="replace")
                if len(line) < fips_start + fips_len - 1:
                    continue
                fips = line[fips_start - 1 : fips_start - 1 + fips_len].strip()
                if not fips or len(fips) < 4:
                    continue
                if len(fips) == 4:
                    fips = "0" + fips
                if fips in ("00000", "99999") or not fips.isdigit():
                    continue
                # State-level "balance of state" sentinels (XX999) bundle
                # all suppressed <100K-pop counties into one row. They are
                # not real counties — drop them.
                if fips.endswith("999"):
                    continue
                if wt_start is not None and wt_len is not None:
                    w_str = line[wt_start - 1 : wt_start - 1 + wt_len].strip()
                    try:
                        w = int(w_str) if w_str else 1
                    except ValueError:
                        w = 1
                else:
                    w = 1
                counts[fips] += w
                n_unmasked += 1
    if not counts:
        return None
    out = pd.DataFrame(
        {"county_fips": list(counts.keys()),
         "year": year,
         "births": list(counts.values()),
         "source": "nber_pub_local"}
    )
    logger.info(
        "year %d (local): %s rows, %s unmasked, %s counties, %s births",
        year, f"{n_rows:,}", f"{n_unmasked:,}", len(out), f"{out['births'].sum():,}",
    )
    return out


def _aggregate_year(year: int) -> Optional[pd.DataFrame]:
    """Stream NBER's CSV for `year` and aggregate to (county_fips, births)."""
    url = _NBER_URL.format(year=year)
    logger.info("streaming NBER natality %d from %s", year, url)
    try:
        line_iter = _stream_url(url)
        header_line = next(line_iter)
    except StopIteration:
        logger.warning("year %d: empty file", year); return None
    except _http_error() as e:
        resp = getattr(e, "response", None)
        if resp is not None and getattr(resp, "status_code", None) == 404:
            logger.warning("year %d: NBER 404 (file not available)", year)
            return None
        raise

    header = next(csv.reader([header_line]))
    header_lower = [h.strip().strip('"').lower() for h in header]

    fips_col = next((c for c in _COUNTY_FIPS_CANDIDATES if c in header_lower), None)
    nchs_col = next((c for c in _COUNTY_NCHS_CANDIDATES if c in header_lower), None)
    split_state_col = next((c for c in _SPLIT_STATE_ABBR_CANDIDATES if c in header_lower), None)
    split_county_col = next((c for c in _SPLIT_COUNTY_FIPS_CANDIDATES if c in header_lower), None)
    weight_col = next((c for c in _WEIGHT_CANDIDATES if c in header_lower), None)

    crosswalk: dict[str, str] = {}
    fips_idx: int | None = None
    nchs_idx: int | None = None
    split_state_idx: int | None = None
    split_county_idx: int | None = None
    source_kind = ""
    if fips_col is not None:
        fips_idx = header_lower.index(fips_col)
        source_kind = f"nber_natality_{fips_col}"
    elif split_state_col is not None and split_county_col is not None:
        # 2003-2004: state abbr + 3-digit county code
        split_state_idx = header_lower.index(split_state_col)
        split_county_idx = header_lower.index(split_county_col)
        source_kind = f"nber_natality_{split_state_col}+{split_county_col}"
    elif nchs_col is not None:
        nchs_idx = header_lower.index(nchs_col)
        crosswalk = _load_nchs_to_fips()
        if not crosswalk:
            logger.warning(
                "year %d: %s present but no NCHS↔FIPS crosswalk loaded; skipping",
                year, nchs_col,
            )
            return None
        source_kind = f"nber_natality_{nchs_col}_via_crosswalk"
    else:
        logger.warning(
            "year %d: no recognized county column; skipping (header: %s...)",
            year, header_lower[:20],
        )
        return None

    weight_idx = header_lower.index(weight_col) if weight_col else None
    # NBER natality 1968-1971 ships the 50% sample without a recwt column
    # (it's implicit). Each row should count as 2 births to recover the
    # population total. From 1972+ the recwt column is present and is the
    # authoritative weight.
    implicit_weight = 2 if weight_col is None and year in (1968, 1969, 1970, 1971) else 1

    counts: dict[str, int] = defaultdict(int)
    n_rows = 0
    n_unmasked = 0
    n_unmapped = 0
    reader = csv.reader(line_iter)
    for row in reader:
        n_rows += 1
        if fips_idx is not None:
            if len(row) <= fips_idx:
                continue
            fips = row[fips_idx].strip().strip('"')
        elif split_state_idx is not None and split_county_idx is not None:
            if len(row) <= max(split_state_idx, split_county_idx):
                continue
            abbr = row[split_state_idx].strip().strip('"')
            cnty3 = row[split_county_idx].strip().strip('"')
            if not abbr or not cnty3 or not cnty3.isdigit():
                continue
            state_fips = _ABBR_TO_FIPS.get(abbr)
            if state_fips is None:
                n_unmapped += 1
                continue
            cnty3 = cnty3.zfill(3)
            fips = state_fips + cnty3
        else:
            assert nchs_idx is not None
            if len(row) <= nchs_idx:
                continue
            nchs = row[nchs_idx].strip().strip('"')
            if not nchs or not nchs.isdigit():
                continue
            if len(nchs) == 4:
                nchs = "0" + nchs
            fips_maybe: str | None = crosswalk.get(nchs)
            if fips_maybe is None:
                n_unmapped += 1
                continue
            fips = fips_maybe
        if not fips or len(fips) < 4:
            continue
        # 5-digit FIPS; left-pad if leading zero stripped.
        if len(fips) == 4:
            fips = "0" + fips
        # Suppression sentinels:
        # NBER often codes masked counties as "00000" or "99999"; rows where
        # the county is suppressed under the 100K+ rule are bundled into a
        # state-level "balance of state" code "XX999". Drop both.
        if fips in ("00000", "99999") or fips.endswith("999"):
            continue
        try:
            int(fips)
        except ValueError:
            continue
        if weight_idx is not None and weight_idx < len(row):
            w_str = row[weight_idx].strip().strip('"') or "1"
            try:
                w = int(w_str)
            except ValueError:
                w = 1
        else:
            w = implicit_weight
        counts[fips] += w
        n_unmasked += 1

    if not counts:
        logger.warning("year %d: no records aggregated", year)
        return None
    out = pd.DataFrame(
        {"county_fips": list(counts.keys()),
         "year": year,
         "births": list(counts.values()),
         "source": source_kind}
    )
    logger.info(
        "year %d: %s rows, %s unmasked, %s counties, %s births%s",
        year, f"{n_rows:,}", f"{n_unmasked:,}", len(out), f"{out['births'].sum():,}",
        (f", {n_unmapped:,} unmapped NCHS codes" if n_unmapped else ""),
    )
    return out


def _year_cache_path(year: int) -> Path:
    return _LOCAL_CACHE / f"nchs_county_{year}.csv"


def available_local_years() -> list[int]:
    """Years for which we already have data — local PUB OR cached aggregation."""
    years: set[int] = set()
    if _FERTILITY_NATALITY_RAW.exists():
        for zp in _FERTILITY_NATALITY_RAW.glob("Nat*.zip"):
            m = re.search(r"Nat(\d{4})\.zip", zp.name)
            if m:
                year = int(m.group(1))
                if (_FERTILITY_NATALITY_RAW / f"natality{year}.dct").exists():
                    years.add(year)
    if _LOCAL_CACHE.exists():
        for cp in _LOCAL_CACHE.glob("nchs_county_*.csv"):
            m = re.search(r"nchs_county_(\d{4})\.csv", cp.name)
            if m:
                years.add(int(m.group(1)))
    return sorted(years)


def fetch(
    years: Iterable[int] | None = None,
    *,
    refresh: bool = False,
    write_artifact: bool = True,
    local_only: bool = True,
) -> pd.DataFrame:
    """Return a long-form (county_fips, year, births, source) panel.

    Parameters
    ----------
    years : iterable of int or None
        If None (default), uses the years for which local PUB files exist
        in the Fertility project's nvss_natality_raw folder (currently
        1982-1993). Pass an explicit range to extend; years not in the
        local cache will be streamed from NBER (slow, ~20 min/year)
        unless ``local_only=True``.
    refresh : bool
        Re-aggregate every year, ignoring per-year caches.
    write_artifact : bool
        Also write the combined frame to the canonical CSV in the
        Fertility project's PROCESSED folder (when that path exists).
    local_only : bool
        If True (default), skip streaming for years without a local PUB
        file. If False, stream from NBER as a fallback.
    """
    _LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    if years is None:
        years = available_local_years()
        if not years:
            logger.warning(
                "No local NCHS PUB files found at %s — pass years= to enable streaming",
                _FERTILITY_NATALITY_RAW,
            )
            years = []

    frames = []
    for y in years:
        cache_path = _year_cache_path(y)
        if cache_path.exists() and not refresh:
            df = pd.read_csv(cache_path, dtype={"county_fips": str})
        else:
            # Prefer local PUB when present; fall back to streaming NBER CSV.
            df = _aggregate_year_local(y)
            if df is None and not local_only:
                df = _aggregate_year(y)
            elif df is None:
                logger.info("year %d: no local PUB and local_only=True; skipping", y)
            if df is not None:
                df.to_csv(cache_path, index=False)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["county_fips", "year", "births", "source"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["county_fips", "year"]).reset_index(drop=True)

    if write_artifact and _FERTILITY_DATA.exists():
        _FERTILITY_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        all_cached = []
        for csv_path in sorted(_LOCAL_CACHE.glob("nchs_county_*.csv")):
            try:
                all_cached.append(pd.read_csv(csv_path, dtype={"county_fips": str}))
            except Exception as e:
                logger.warning("skip %s while assembling artifact: %s", csv_path, e)
        if all_cached:
            nchs_assembled = pd.concat(all_cached, ignore_index=True)
            nchs_assembled = nchs_assembled.sort_values(
                ["county_fips", "year"]
            ).reset_index(drop=True)
            nchs_assembled.to_csv(_FERTILITY_ARTIFACT, index=False)
            logger.info(
                "wrote NCHS artifact (%d rows, %d years, %d counties) to %s",
                len(nchs_assembled), nchs_assembled["year"].nunique(),
                nchs_assembled["county_fips"].nunique(), _FERTILITY_ARTIFACT,
            )
            _write_combined_artifact(nchs_assembled)
        else:
            combined.to_csv(_FERTILITY_ARTIFACT, index=False)

    return combined


def _write_combined_artifact(nchs: pd.DataFrame) -> None:
    """Combine NCHS + Census PEP into a single county_year_births.csv.

    Source preference per year:
      - Census PEP when the year is in its calendar-year coverage
        (2001-2009, 2011-2019, 2021-2025).
      - NCHS otherwise (1968-2000, 2010 fallback, plus all overlap years
        where Census doesn't cover).

    For the 2003-2004 overlap window (where both have data), Census PEP
    wins because it covers all counties without the 100K+ suppression.
    """
    try:
        from . import census_pep_births
        pep = census_pep_births.fetch_county_year()
    except Exception as e:
        logger.warning(
            "Census PEP fetch failed (%s); combined artifact = NCHS only", e,
        )
        nchs.assign(source=nchs.get("source", "nber_natality")).to_csv(
            _COMBINED_ARTIFACT, index=False,
        )
        return

    pep_years = set(pep["year"].unique().tolist())
    pep["county_fips"] = pep["county_fips"].astype(str).str.zfill(5)
    nchs = nchs.copy()
    nchs["county_fips"] = nchs["county_fips"].astype(str).str.zfill(5)

    # Use NCHS only for years not covered by Census PEP
    nchs_keep = nchs[~nchs["year"].isin(pep_years)].copy()
    combined = pd.concat([pep, nchs_keep], ignore_index=True)
    combined = combined.sort_values(["county_fips", "year"]).reset_index(drop=True)
    combined.to_csv(_COMBINED_ARTIFACT, index=False)
    logger.info(
        "wrote combined artifact (%d rows, years %d-%d, %d counties) to %s",
        len(combined), combined["year"].min(), combined["year"].max(),
        combined["county_fips"].nunique(), _COMBINED_ARTIFACT,
    )
