# Source-connector retry / timeout / failure policy

This document is the contract every `narrative/sources/*.py` connector — and the canonical-replication loaders in `narrative/replication/*.py` — must satisfy. The contract exists so that a downstream user can call any number of connectors back-to-back without thinking about network failures, SSL handshakes, rate-limit pages, or partial responses.

If you add a new connector or replication loader, **read this file first**. If your change makes anything below stale, update it in the same commit.

## 1. Use the shared helpers, not your own

`puremacro/narrative/sources/_http.py` exposes three helpers covering every fetch mode the package needs:

| Helper | Returns | Use when |
|---|---|---|
| `safe_get_bytes(url, timeout=30)` | `bytes` | Binary payloads (RSS / Atom XML, CSV, PDF). |
| `safe_get_text(url, timeout=30)` | `str` (UTF-8, errors ignored) | HTML scraping. |
| `safe_get_json(url, timeout=30)` | `dict` | JSON APIs. Empty/whitespace bodies → `{}`. |

All three internally route through one `_request` function, so any policy change here propagates everywhere. **Do not write a new `_safe_get`** — import from `_http` instead.

## 2. The transport contract

- **User-Agent**: every request sends `"Mozilla/5.0 (puremacro/narrative)"`. Many ministry / multilateral sites 403 unmarked clients.
- **Default timeout**: 30 s. Override by passing `timeout=` if your source is unusually slow (IMF Article IV PDFs, OECD surveys). Don't hard-code a longer global default — the user will hit it on every other source if so.
- **SSL fallback**: on `URLError` or `SSLError`, one retry with `ssl._create_unverified_context()`. Some public-data sites ship certificates that Python's bundled CA store does not validate; we accept the security trade-off because nothing here authenticates the user.
- **No further retries**. We do **not** retry on HTTP 5xx, on connection resets, or on timeout. See §3.

## 3. Why no exponential backoff

The connectors are called from research notebooks and from tests. Two failure modes drove the policy:

- **Notebook iteration**: a researcher running a 30-source aggregation does not want their kernel to hang for 5 minutes when one ministry site is down. Better: skip that source, keep going.
- **Tests**: smoke tests must terminate. A retry loop with backoff turns a 0.2-second skip into a 30-second hang.

If you genuinely need retries (e.g. a polite poll loop for a rate-limited paid API), build it in the *connector*, not in `_http`. Keep `_http` synchronous and finite.

## 4. The connector's own contract

Every `iter_<source>` connector must satisfy:

1. **Yield, don't raise**. Network errors → empty iterator. Parse errors → skip the offending record, keep yielding the rest.
2. **Yield `(date, text, source_url)` tuples**, where `date` is a non-NaT `pd.Timestamp` and `text` is non-empty. Records that fail this filter must be dropped, not yielded.
3. **No global state**. Each call constructs its own iterator; multiple consumers don't see each other's progress.
4. **No silent caching**. If you cache (e.g. `imf_articleiv.iter_imf_listing`'s on-disk index), expose a `clear_<thing>_cache()` companion so tests and refresh-style scripts can invalidate it.

Connectors that violate (1) — i.e. propagate a network exception — break every multi-source aggregation. Wrap external calls in `try / except Exception` and `return` if you cannot recover.

## 5. The replication-loader contract

`narrative/replication/<dataset>.py` modules use the same `_http` helpers under a slightly stronger contract:

- They must **return a populated `NarrativeInstrument`**, never an empty iterator. If the public mirror is unreachable they fall back to a small built-in synthetic series (DGLP, RR-2017) or raise a clear `RuntimeError` instructing the caller to pass `csv_path=`.
- They **must** also expose a `<dataset>_csv_to_events` helper that takes an already-loaded `pd.DataFrame` and returns a list of `NarrativeEvent`. This is the path used in offline tests and in the homogeneous-panel example.

The split (loader vs. csv-to-events) lets `tests/test_narrative.py` exercise the pure-Python coercion logic without any HTTP dependency.

## 6. What this policy explicitly does not do

- **Authentication**. No connector currently sends API keys. If you add one (NewsAPI paid tier, IMF developer portal), document the env-var name in the connector docstring and `RuntimeError` if it is unset — never silently fail.
- **Rate-limit awareness**. We respect timeouts but do not parse `Retry-After` headers. If a source rate-limits us, the connector returns whatever it has so far. Researcher reruns minutes later when the budget refreshes.
- **Concurrency**. Connectors are synchronous; they do not use `asyncio` or threads. The aggregate cost of fetching 30 sources serially is dominated by the slowest two; parallelism would complicate the failure-isolation contract for ~10× speedup.
- **Caching layer**. Each connector decides whether on-disk caching makes sense. There is no shared cache.

If a future iteration genuinely needs any of these, add it as a separate, opt-in helper module — `_http_cached.py`, `_http_async.py` — and document the deviation here. Do not retrofit `_http.py`.

## §7 User-Agent overrides

Some public endpoints sit behind a WAF that blocks the default
`Mozilla/5.0 (puremacro/narrative)` agent string. Connectors that
hit such endpoints should pass an explicit, realistic browser UA via
the `user_agent=` keyword on `safe_get_bytes` / `safe_get_text` /
`safe_get_json`.

Currently this applies to:
- `us_dod_contracts.iter_dod_contracts` — defense.gov WAF.
