> 🇬🇧 English · 🇪🇸 [Español](es/CONNECTOR_HEALTH.md)

# Connector health

> Available from puremacro **0.67.0** onwards.

`puremacro.narrative.sources._telemetry.connector_health()` aggregates
per-fetch events from the `connector_events` SQLite table (one row per
fetch attempt by a participating connector) and returns a DataFrame
indicating which connectors are healthy, degrading, or down.

## Quickstart

```python
from puremacro.narrative.sources._telemetry import connector_health
import pandas as pd

connector_health(window=pd.Timedelta(days=7))
#         source  n_total  n_success  success_rate  n_fallback  fallback_rate           last_seen
# 0    eu_eurlex      142        119         0.838          23          0.162  2026-05-25 14:22:03
# 1 eu_parliament       89         72         0.809          17          0.191  2026-05-24 09:11:47
# 2       us_cbo       45         45         1.000           0          0.000  2026-05-23 18:05:12
# 3          rba       28         26         0.929           2          0.071  2026-05-26 02:33:01
# 4          bok       31         28         0.903           3          0.097  2026-05-26 03:14:55
```

A `fallback_rate > 0` means the live endpoint failed and a fallback
(Wayback / Playwright) served the request. A `success_rate < 1` means
some requests failed entirely — the `connector_events` rows show
exactly which outcome (timeout, 404, ssl_fail, wayback_no_snapshot,
playwright_unavailable, parser_schema_mismatch, other_network_error).

## Filtering

```python
connector_health(window=pd.Timedelta(days=30))                   # last 30 days
connector_health(sources=["eu_eurlex", "rba"])                   # subset of connectors
connector_health(window=pd.Timedelta(hours=1), sources=["bok"])  # combined
```

## Event schema

```sql
CREATE TABLE connector_events (
    ts             INTEGER NOT NULL,    -- unix epoch seconds
    source         TEXT NOT NULL,       -- 'eu_eurlex', 'rba', 'beige_book', ...
    outcome        TEXT NOT NULL,       -- success / 404 / timeout / ssl_fail /
                                        -- server_5xx / wayback_no_snapshot /
                                        -- playwright_unavailable /
                                        -- parser_schema_mismatch / other_network_error
    fallback_used  TEXT NOT NULL        -- live / wayback / playwright / none
);
CREATE INDEX connector_events_ts_source_idx
    ON connector_events(ts, source);
```

Lives in `~/.cache/puremacro/cache.db` (or `$PUREMACRO_HTTP_CACHE_DIR`).

## What gets logged (Slice B scope)

- **`fetch_with_fallback`** (the 7 fallback-aware connectors): one
  event per stage attempt. Connectors:
  `eu_eurlex`, `eu_parliament`, `us_cbo`, `rba`, `bok`, `riksbank`, `sarb`.
- **`iter_<source>` wrappers for the 8 Slice-A schema-checked
  connectors**: one event per `ParserSchemaMismatchError` catch
  (`outcome="parser_schema_mismatch"`, `fallback_used="none"`).
- **Other connectors**: no events in Slice B. Will not show up in
  `connector_health()` until they opt in (typically by adopting
  `fetch_with_fallback(policy=("live",))` or calling `log_event(...)`
  directly).

## Kill-switch

```bash
export PUREMACRO_NARRATIVE_TELEMETRY=0    # disable all event logging
```

`log_event` becomes a no-op. `connector_health` still reads any rows
inserted before the env var was set. Useful for off-network runs,
strict-reproducibility CI, or users who don't want DB writes from
fetcher code.

## Failure semantics

Telemetry never breaks a fetch. If the DB is locked, unreachable, or
disk-full, `log_event` emits a `UserWarning` and silently returns.
`connector_health` does the same and returns an empty DataFrame with
the expected columns.

## Pyodide

`sqlite3` is Python stdlib — available everywhere puremacro runs.
The event log lives on the Pyodide virtual filesystem; persistence
across page reloads requires IDBFS mount (same caveat as the rest of
the cache DB — see `docs/CACHE_DB.md`).
