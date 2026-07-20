"""Bank Negara Malaysia (BNM) — monetary-policy decisions.

Live: https://www.bnm.gov.my
Decision source: https://api.bnm.gov.my/public/opr/year/{year}

The BNM website (bnm.gov.my) is protected by AWS WAF / CloudFront and
returns an empty 202 challenge page to plain HTTP clients.  However, BNM
publishes a fully-public Open API at api.bnm.gov.my that exposes
historical Overnight Policy Rate (OPR) decisions — one record per MPC
meeting — in a clean JSON format, bypassing the WAF entirely.

The OPR API returns, for each year requested:
  {"data": [{"year": int, "date": "YYYY-MM-DD",
             "change_in_opr": float, "new_opr_level": float}, ...],
   "meta": {"last_updated": "...", "total_result": int}}

The connector fetches the current calendar year by default.  Pass
``years`` to retrieve multiple years.

Speeches archive: BNM does not expose a clean English speeches
archive through the Open API or any unprotected endpoint.  The
website speeches page (bnm.gov.my/speeches) is WAF-gated and
Wayback has no usable snapshots.  ``iter_bnm_speeches`` is therefore
NOT shipped in this connector; see note in __all__.

Verified 2026-05-26 by the F1 Slice A rollout.
FALLBACK_POLICY = ("live",)  — api.bnm.gov.my is a public REST API,
no WAF, no JS rendering.
"""
from __future__ import annotations

import datetime
import json
import warnings
from typing import Iterator

from ._fallback import fetch_with_fallback, FallbackExhaustedError
from ._schema_check import assert_landmarks, ParserSchemaMismatchError
from ._telemetry import log_event


PARSER_SCHEMA_VERSION = 1
FALLBACK_POLICY: tuple[str, ...] = ("live",)

_API_BASE = "https://api.bnm.gov.my/public/opr/year"

# Landmarks verified from live API response captured 2026-05-26.
# The JSON body always contains these substring markers.
_DECISION_LANDMARKS = ['"data"', '"date"', '"new_opr_level"']

_ACCEPT_HEADER_NOTE = "application/vnd.BNM.API.v1+json"


def _decision_url(year: int) -> str:
    return f"{_API_BASE}/{year}"


def _parse_opr_json(
    body: str,
    *,
    year: int,
) -> Iterator[tuple]:
    """Parse one OPR-by-year API response; yield (date, text, url, meta)."""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        warnings.warn(
            f"bnm: JSON decode error for year {year}: {e}",
            UserWarning, stacklevel=3,
        )
        return

    entries = payload.get("data", [])
    if not isinstance(entries, list):
        warnings.warn(
            f"bnm: unexpected 'data' type {type(entries)} for year {year}",
            UserWarning, stacklevel=3,
        )
        return

    # Deduplicate by date (the 2024 API returns some dates twice).
    seen: set[str] = set()
    for entry in entries:
        raw_date = entry.get("date", "")
        if not raw_date or raw_date in seen:
            continue
        seen.add(raw_date)

        try:
            date = datetime.datetime.strptime(raw_date, "%Y-%m-%d")
        except (ValueError, TypeError):
            continue

        change = entry.get("change_in_opr", 0)
        level = entry.get("new_opr_level")

        # Synthesise a brief text description from the structured fields.
        if change > 0:
            direction = f"raised by {change:+.2f}pp"
        elif change < 0:
            direction = f"cut by {change:.2f}pp"
        else:
            direction = "held unchanged"
        text = (
            f"Bank Negara Malaysia MPC: OPR {direction} "
            f"to {level:.2f}% on {raw_date}."
        )

        url = _decision_url(date.year)
        yield (date, text, url, {
            "bank_code": "BNM",
            "country": "MYS",
            "doctype": "decision",
            "language": "en",
            "change_in_opr": change,
            "new_opr_level": level,
        })


def iter_bnm_decision(
    *,
    years: "list[int] | None" = None,
    fetch_body: bool = False,
) -> Iterator[tuple]:
    """Yield (date, text, url, metadata) tuples for BNM MPC decisions.

    Fetches the BNM Open API OPR endpoint for each requested year.

    Parameters
    ----------
    years : list of calendar years to retrieve.  Defaults to the current
        year.  Pass e.g. ``years=[2022, 2023, 2024]`` for multi-year
        retrieval.
    fetch_body : unused (kept for API symmetry with other connectors;
        the API already returns structured data, no body fetch needed).
    """
    if years is None:
        years = [datetime.datetime.now(datetime.timezone.utc).year]

    for year in years:
        url = _decision_url(year)
        try:
            listing = fetch_with_fallback(
                url, policy=FALLBACK_POLICY, source="bnm",
            )
        except FallbackExhaustedError as e:
            warnings.warn(
                f"bnm.iter_bnm_decision: fetch failed for year {year}: {e}",
                UserWarning, stacklevel=2,
            )
            continue

        try:
            assert_landmarks(
                listing, source="bnm",
                expected_version=PARSER_SCHEMA_VERSION,
                landmarks=_DECISION_LANDMARKS,
            )
        except ParserSchemaMismatchError as e:
            log_event(source="bnm", outcome="parser_schema_mismatch",
                      fallback_used="none")
            warnings.warn(
                f"bnm.iter_bnm_decision: schema mismatch for year {year}: {e}",
                UserWarning, stacklevel=2,
            )
            continue

        yield from _parse_opr_json(listing, year=year)


# iter_bnm_speeches is intentionally omitted: BNM does not expose a
# clean English speeches archive through any unprotected endpoint.
# The bnm.gov.my/speeches page requires JS rendering (AWS WAF challenge)
# and Wayback Machine has no usable snapshots.  Ship when BNM adds an
# Open API speeches endpoint or an unprotected RSS feed.

__all__ = ["iter_bnm_decision"]
