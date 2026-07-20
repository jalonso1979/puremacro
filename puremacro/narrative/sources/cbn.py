"""Central Bank of Nigeria (CBN) — monetary-policy decisions + governor speeches.

Live: https://www.cbn.gov.ng
Decision source (JSON REST API):
  https://www.cbn.gov.ng/api/GetAllMpc

The CBN website serves its MPC communique listing via a Kendo Grid widget
that fetches its data from the public REST endpoint ``/api/GetAllMpc``.
The endpoint returns all MPC-related documents (communiques, personal
statements, supplementary annexes) as a flat JSON array, with no
authentication and no WAF challenge.

Decision filtering: the endpoint returns both rate communiques (titles
containing "Communique No.") and supplementary "Personal Statements of
MPC Members" documents.  This connector keeps only the communiques by
filtering to entries whose title matches ``_COMMUNIQUE_TITLE_RX``.

Speech source (JSON REST API):
  https://www.cbn.gov.ng/api/GetAllSpeeches

The speeches endpoint follows the same schema (same field names,
same date format).  ``iter_cbn_speeches`` is shipped because the
endpoint is clean, publicly accessible, and returns 200+ speeches.

Date format: ``"DD/MM/YYYY"`` in the ``documentDate`` field.
Link field: relative path (``/Out/<year>/CCD/<filename>.pdf``).

Verified 2026-05-26 by the F1 Slice A rollout.
FALLBACK_POLICY = ("live",)  — the JSON REST API is public, no WAF,
no JS rendering required.
"""
from __future__ import annotations

import datetime
import json
import re
import warnings
from typing import Iterator

from ._fallback import fetch_with_fallback, FallbackExhaustedError
from ._schema_check import assert_landmarks, ParserSchemaMismatchError
from ._telemetry import log_event


PARSER_SCHEMA_VERSION = 1
FALLBACK_POLICY: tuple[str, ...] = ("live",)

_BASE_URL = "https://www.cbn.gov.ng"
_DECISION_URL = f"{_BASE_URL}/api/GetAllMpc"
_SPEECHES_URL = f"{_BASE_URL}/api/GetAllSpeeches"

# Landmarks verified from live API responses captured 2026-05-26.
# The JSON array body always contains these substring markers.
_DECISION_LANDMARKS = ['"documentDate"', '"refNo"', '"CBN/MPC/COM']
_SPEECHES_LANDMARKS = ['"documentDate"', '"refNo"', '"title"']

# Filter to rate-decision communiques only.
# Excludes "Personal Statements of MPC Members" and other supplementary docs.
_COMMUNIQUE_TITLE_RX = re.compile(
    r"communique\s+no\.?\s*\d+",
    re.IGNORECASE,
)

_DATE_FMT = "%d/%m/%Y"


def _parse_mpc_json(body: str, *, doctype: str) -> Iterator[tuple]:
    """Parse the /api/GetAllMpc or /api/GetAllSpeeches response.

    Yields (date, text, url, metadata) 4-tuples.  When ``doctype`` is
    ``"decision"`` only communique entries (matching
    ``_COMMUNIQUE_TITLE_RX``) are returned; all entries are returned
    for ``doctype="speech"``.
    """
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        warnings.warn(
            f"cbn: JSON decode error in {doctype} response: {e}",
            UserWarning, stacklevel=3,
        )
        return

    if not isinstance(payload, list):
        warnings.warn(
            f"cbn: unexpected top-level type {type(payload)} in "
            f"{doctype} response (expected list)",
            UserWarning, stacklevel=3,
        )
        return

    seen: set[int] = set()
    for entry in payload:
        item_id = entry.get("id")
        if item_id is None or item_id in seen:
            continue
        seen.add(item_id)

        title = (entry.get("title") or "").strip()
        if not title:
            continue

        # For decisions: keep only rate communiques
        if doctype == "decision" and not _COMMUNIQUE_TITLE_RX.search(title):
            continue

        raw_date = (entry.get("documentDate") or "").strip()
        if not raw_date:
            continue

        try:
            date = datetime.datetime.strptime(raw_date, _DATE_FMT)
        except (ValueError, TypeError):
            continue

        link = (entry.get("link") or "").strip()
        url = (f"{_BASE_URL}{link}" if link.startswith("/") else link) or _BASE_URL

        ref_no = (entry.get("refNo") or "").strip()

        yield (date, title, url, {
            "bank_code": "CBN",
            "country": "NGA",
            "doctype": doctype,
            "language": "en",
            "item_id": item_id,
            "ref_no": ref_no,
        })


def iter_cbn_decision(*, fetch_body: bool = False) -> Iterator[tuple]:
    """Yield (date, text, url, metadata) tuples for CBN MPC rate decisions.

    Fetches the CBN public REST API ``/api/GetAllMpc`` and filters to
    entries whose title matches "Communique No. <N>" — the official
    rate-decision communiques published after each MPC meeting.
    "Personal Statements of MPC Members" documents are excluded.

    Parameters
    ----------
    fetch_body : unused (kept for API symmetry; the API already returns
        structured title data as the primary text, and the linked PDFs
        require a PDF reader to extract).
    """
    try:
        listing = fetch_with_fallback(
            _DECISION_URL, policy=FALLBACK_POLICY, source="cbn",
        )
    except FallbackExhaustedError as e:
        warnings.warn(
            f"cbn.iter_cbn_decision: fetch failed: {e}",
            UserWarning, stacklevel=2,
        )
        return

    try:
        assert_landmarks(
            listing, source="cbn", expected_version=PARSER_SCHEMA_VERSION,
            landmarks=_DECISION_LANDMARKS,
        )
    except ParserSchemaMismatchError as e:
        log_event(source="cbn", outcome="parser_schema_mismatch",
                  fallback_used="none")
        warnings.warn(
            f"cbn.iter_cbn_decision: schema mismatch: {e}",
            UserWarning, stacklevel=2,
        )
        return

    yield from _parse_mpc_json(listing, doctype="decision")


def iter_cbn_speeches(*, fetch_body: bool = False) -> Iterator[tuple]:
    """Yield (date, text, url, metadata) tuples for CBN governor speeches.

    Fetches the CBN public REST API ``/api/GetAllSpeeches``, which
    returns all governor and senior-official speeches ordered by date.

    Parameters
    ----------
    fetch_body : unused (kept for API symmetry; the API already returns
        structured title data as the primary text, and the linked PDFs
        require a PDF reader to extract).
    """
    try:
        listing = fetch_with_fallback(
            _SPEECHES_URL, policy=FALLBACK_POLICY, source="cbn",
        )
    except FallbackExhaustedError as e:
        warnings.warn(
            f"cbn.iter_cbn_speeches: fetch failed: {e}",
            UserWarning, stacklevel=2,
        )
        return

    try:
        assert_landmarks(
            listing, source="cbn", expected_version=PARSER_SCHEMA_VERSION,
            landmarks=_SPEECHES_LANDMARKS,
        )
    except ParserSchemaMismatchError as e:
        log_event(source="cbn", outcome="parser_schema_mismatch",
                  fallback_used="none")
        warnings.warn(
            f"cbn.iter_cbn_speeches: schema mismatch: {e}",
            UserWarning, stacklevel=2,
        )
        return

    yield from _parse_mpc_json(listing, doctype="speech")


__all__ = ["iter_cbn_decision", "iter_cbn_speeches"]
