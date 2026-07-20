"""Governed fallback layer for narrative connectors (0.67.0+).

Each participating connector declares a module-level
``FALLBACK_POLICY = ("live", "wayback")`` (or similar tuple of
``SUPPORTED_STAGES``) and calls

    fetch_with_fallback(url, policy=FALLBACK_POLICY, source="<name>")

instead of a hand-rolled ``try/except`` chain. ``fetch_with_fallback``
loops through stages; on success it returns the body; on failure it
logs a telemetry event and moves to the next stage. Raises
``FallbackExhaustedError`` only if every stage fails.

Adding a new stage (e.g. ``tor``, ``paid_proxy``, ``mirrored_s3``) is
a one-line addition to ``SUPPORTED_STAGES`` + a branch in
``_dispatch_stage``.
"""
from __future__ import annotations

import socket
import ssl
import urllib.error

from ._http import safe_get_text
from ..._http import safe_get_text_cached
from ._telemetry import log_event
from ._wayback import wayback_snapshot_url


SUPPORTED_STAGES: frozenset[str] = frozenset({"live", "wayback", "playwright"})


class FallbackExhaustedError(RuntimeError):
    """Raised by fetch_with_fallback when every stage in the policy has
    been tried and none succeeded. Caught by iter_<source> wrappers per
    the yield-don't-raise contract in RETRY_POLICY.md §4.1."""


class FallbackStageUnavailable(RuntimeError):
    """Raised internally by a _stage_* function when its dependency is
    missing (Playwright not installed) or its precondition fails
    (Wayback has no snapshot). The fetch_with_fallback loop classifies
    these via _classify (using the message argument as the outcome key)
    and treats them as a normal stage failure (continue to next stage)."""


def _classify(e: Exception) -> str:
    """Map an exception to one of VALID_OUTCOMES (excluding 'success').

    Order matters — match HTTPError before URLError because HTTPError
    subclasses URLError.
    """
    if isinstance(e, FallbackStageUnavailable):
        return str(e)
    if isinstance(e, urllib.error.HTTPError):
        if e.code == 404:
            return "404"
        if 500 <= e.code < 600:
            return "server_5xx"
        return "other_network_error"
    if isinstance(e, ssl.SSLError):
        return "ssl_fail"
    if isinstance(e, (socket.timeout, TimeoutError)):
        return "timeout"
    if isinstance(e, urllib.error.URLError):
        # Wrapped socket errors. Check the reason if it's a timeout.
        reason = getattr(e, "reason", None)
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return "timeout"
        if isinstance(reason, ssl.SSLError):
            return "ssl_fail"
        return "other_network_error"
    return "other_network_error"


def _stage_live(url: str, *, timeout: float, use_cache: bool) -> str:
    """Live HTTP fetch (via the existing cached helper)."""
    body = safe_get_text_cached(url, timeout=timeout) if use_cache \
        else safe_get_text(url, timeout=timeout)
    if not body or not body.strip():
        raise urllib.error.HTTPError(url, 204, "empty body", None, None)  # type: ignore[arg-type]  # stub omits None for hdrs
    return body


def _stage_wayback(url: str, *, timeout: float, use_cache: bool) -> str:
    """Wayback fetch: CDX lookup + snapshot fetch via cached helper."""
    wb_url = wayback_snapshot_url(url)
    if wb_url is None:
        raise FallbackStageUnavailable("wayback_no_snapshot")
    body = safe_get_text_cached(wb_url, timeout=timeout) if use_cache \
        else safe_get_text(wb_url, timeout=timeout)
    if not body or not body.strip():
        raise urllib.error.HTTPError(wb_url, 204, "empty wayback body", None, None)  # type: ignore[arg-type]  # stub omits None for hdrs
    return body


def _stage_playwright(url: str, *, timeout: float, use_cache: bool = False) -> str:
    """Playwright (stealth-Chromium) fetch. Lazy-imports the helper so
    a pyodide / no-extras install doesn't fail at module load time.
    No cache layer — Playwright is the last resort."""
    try:
        from ._playwright_helper import fetch_with_playwright
    except ImportError:
        raise FallbackStageUnavailable("playwright_unavailable")
    return fetch_with_playwright(url, timeout_ms=int(timeout * 1000))


def _dispatch_stage(stage: str, url: str, *,
                    timeout: float, use_cache: bool) -> str:
    if stage == "live":
        return _stage_live(url, timeout=timeout, use_cache=use_cache)
    if stage == "wayback":
        return _stage_wayback(url, timeout=timeout, use_cache=use_cache)
    if stage == "playwright":
        return _stage_playwright(url, timeout=timeout, use_cache=use_cache)
    raise ValueError(
        f"_fallback: unknown stage {stage!r}. "
        f"Supported: {sorted(SUPPORTED_STAGES)}"
    )


def fetch_with_fallback(
    url: str,
    *,
    policy: tuple[str, ...],
    source: str,
    timeout: float = 30.0,
    use_cache: bool = True,
) -> str:
    """Try each stage in ``policy`` in order; return the body from the
    first one that succeeds.

    Raises ``FallbackExhaustedError`` if every stage fails. Emits one
    telemetry event per stage attempt (outcome='success' for the winner,
    a classified failure outcome for each loss).

    Parameters
    ----------
    url : the URL to fetch.
    policy : tuple of stage names. Each must be in ``SUPPORTED_STAGES``.
        Single-stage policies (e.g. ``("wayback",)``) are valid.
    source : the connector's canonical name. Used as the ``source``
        column in connector_events. Required (no default) so misuse
        like ``fetch_with_fallback(url)`` fails at the call site instead
        of silently mis-attributing telemetry.
    timeout : per-stage timeout in seconds (default 30.0).
    use_cache : if True (default), the live and wayback stages use the
        SQLite HTTP cache; Playwright is always uncached.
    """
    if not policy:
        raise ValueError("fetch_with_fallback: empty policy")
    unknown = set(policy) - SUPPORTED_STAGES
    if unknown:
        raise ValueError(
            f"fetch_with_fallback: unknown stage(s) {sorted(unknown)} "
            f"in policy {policy}. Supported: {sorted(SUPPORTED_STAGES)}"
        )
    last_exc: Exception | None = None
    for stage in policy:
        try:
            body = _dispatch_stage(stage, url, timeout=timeout,
                                    use_cache=use_cache)
            log_event(source=source, outcome="success", fallback_used=stage)
            return body
        except Exception as e:
            last_exc = e
            outcome = _classify(e)
            log_event(source=source, outcome=outcome, fallback_used=stage)
            continue
    raise FallbackExhaustedError(
        f"fetch_with_fallback({source!r}): every stage in {policy} failed; "
        f"last error: {last_exc!r}"
    )


__all__ = [
    "SUPPORTED_STAGES",
    "FallbackExhaustedError",
    "FallbackStageUnavailable",
    "fetch_with_fallback",
]
