"""US work-stoppages dataset for β.2 LWUI → strike-count regression.

Originally targeted Cornell ILR's FMCS Federal Mediation strike DB; that
collection (``digitalcommons.ilr.cornell.edu/strikes/``) 404s, and the
BLS Major Work Stoppages monthly listing rejects automated retrieval
(Akamai bot filter on ``bls.gov``). We therefore source from Cornell
ILR's **Labor Action Tracker** (``striketracker.ilr.cornell.edu``),
which exposes its full backing dataset as a single JSON document
(``labor_actions.json``) — the same payload the public dashboard's
front-end consumes via ``fetch()``. Records start 2021-01-01 and cover
every U.S. strike + protest the ILR/U.Illinois LER team has logged.

We filter to ``Action_type == "Strike"`` to match the spec ("strike
database"); ``magnitude`` is ``Approximate_Number_of_Participants``.

The connector NAME stays ``iter_frdb_strikes`` (and ``bank_code="FRDB"``
in metadata) so β-4/β-5 stability-filter wiring is unchanged regardless
of the underlying source switch.

Yields 5-tuples ``(date, text, url, meta, magnitude)``.
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Iterator

import pandas as pd

from ._http import safe_get_bytes


_DATA_URL = "https://striketracker.ilr.cornell.edu/labor_actions.json"
_LANDING = "https://striketracker.ilr.cornell.edu/"

_CACHE = (Path(__file__).resolve().parents[4]
          / "notebooks" / "data_cache" / "frdb_strikes.parquet")


def _normalize_industry(val) -> str:
    """Industry arrives as a list in the Cornell schema; flatten for storage."""
    if isinstance(val, list):
        return ", ".join(str(x) for x in val if x)
    return str(val or "")


def _first_state(locations) -> str:
    if not isinstance(locations, list):
        return ""
    for loc in locations:
        if isinstance(loc, dict) and loc.get("State"):
            return str(loc["State"])
    return ""


def _load(*, refetch: bool) -> pd.DataFrame:
    if _CACHE.exists() and not refetch:
        return pd.read_parquet(_CACHE)
    raw = safe_get_bytes(_DATA_URL)
    if not raw:
        return pd.DataFrame()
    payload = json.loads(raw.decode("utf-8"))
    # Top-level shape is ``dict[str_id, record_dict]``; flatten to list.
    if isinstance(payload, dict):
        records = list(payload.values())
    elif isinstance(payload, list):
        records = payload
    else:
        return pd.DataFrame()
    if not records:
        return pd.DataFrame()
    # Normalise list-typed fields up-front so parquet round-trips cleanly.
    flat = []
    for r in records:
        if not isinstance(r, dict):
            continue
        flat.append({
            "id": r.get("id"),
            "Action_type": r.get("Action_type") or "",
            "Start_date": r.get("Start_date"),
            "End_date": r.get("End_date"),
            "Duration": r.get("Duration"),
            "Employer": r.get("Employer") or "",
            "Labor_Organization": r.get("Labor_Organization") or "",
            "Industry": _normalize_industry(r.get("Industry")),
            "State": _first_state(r.get("locations")),
            "Approximate_Number_of_Participants":
                r.get("Approximate_Number_of_Participants"),
            "Bargaining_Unit_Size": r.get("Bargaining_Unit_Size"),
            "Authorized": r.get("Authorized") or "",
        })
    df = pd.DataFrame(flat)
    try:
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_CACHE)
    except Exception:
        # Cache write is best-effort.
        pass
    return df


def iter_frdb_strikes(
    *,
    since: str | None = "2000-01-01",
    refetch: bool = False,
) -> Iterator[tuple]:
    """Yield 5-tuples for each recorded strike.

    Each yielded record is ``(date, text, url, meta, magnitude)`` where
    ``magnitude`` is the approximate number of workers involved
    (``None`` when the source did not record it).
    """
    df = _load(refetch=refetch)
    if df.empty:
        return

    # Filter to strikes (drop protests + blanks).
    df = df[df["Action_type"].str.lower() == "strike"]
    if df.empty:
        return

    df["Start_date"] = pd.to_datetime(df["Start_date"], errors="coerce")
    df = df.dropna(subset=["Start_date"])
    if since:
        df = df[df["Start_date"] >= pd.Timestamp(since)]

    for _, row in df.iterrows():
        employer = str(row.get("Employer") or "").strip()
        union = str(row.get("Labor_Organization") or "").strip()
        industry = str(row.get("Industry") or "").strip()
        state = str(row.get("State") or "").strip()

        n_workers = None
        v = pd.to_numeric(
            row.get("Approximate_Number_of_Participants"),
            errors="coerce",
        )
        if pd.notna(v) and v >= 0:
            n_workers = float(v)

        duration = None
        dv = pd.to_numeric(row.get("Duration"), errors="coerce")
        if pd.notna(dv) and dv >= 0:
            duration = float(dv)

        if union and industry:
            text = f"{employer} x {union} - {industry} strike"
        elif union:
            text = f"{employer} x {union} strike"
        elif employer:
            text = f"{employer} strike"
        else:
            text = "Labor strike"

        meta = {
            "doctype": "strike",
            "language": "en",
            "bank_code": "FRDB",
            "source": "frdb_strikes",
            "country": "USA",
            "state": state or None,
            "industry": industry,
            "union": union or None,
            "duration_days": duration,
        }
        yield (row["Start_date"], text, _LANDING, meta, n_workers)


__all__ = ["iter_frdb_strikes"]
