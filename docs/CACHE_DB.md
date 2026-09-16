> 🇬🇧 English · 🇪🇸 [Español](es/CACHE_DB.md)

# Cache DB

> Available from puremacro **0.66.0** onwards. Replaces the flat-file
> `~/.cache/puremacro/http/*.bin + *.json` cache from 0.65.0 and earlier.

## Location

Single SQLite file at `~/.cache/puremacro/cache.db` (overridable via
`$PUREMACRO_HTTP_CACHE_DIR`). If the env var ends in `.db`, that path
is used verbatim; otherwise it's treated as a directory and the DB
lives at `<dir>/cache.db`.

## Schema

```sql
CREATE TABLE http_cache (
    key            TEXT PRIMARY KEY,    -- sha256(url) hex
    url            TEXT NOT NULL,
    fetched_at     INTEGER NOT NULL,    -- unix epoch seconds
    content_type   TEXT,
    body           BLOB NOT NULL
);

CREATE TABLE alfred_vintages (
    series_id        TEXT NOT NULL,
    observation_date TEXT NOT NULL,     -- ISO YYYY-MM-DD
    vintage_date     TEXT NOT NULL,     -- ISO YYYY-MM-DD
    value            REAL,
    PRIMARY KEY (series_id, observation_date, vintage_date)
);

CREATE TABLE connector_events (
    ts             INTEGER NOT NULL,    -- unix epoch seconds
    source         TEXT NOT NULL,       -- connector name
    outcome        TEXT NOT NULL,
    fallback_used  TEXT NOT NULL
);

CREATE TABLE realtime_vintages (
    provider         TEXT NOT NULL,     -- 'banxico' / 'inegi' / 'bcb' / 'bcch'
    country          TEXT NOT NULL,     -- ISO-3, upper case
    series_id        TEXT NOT NULL,     -- the provider's own identifier
    observation_date TEXT NOT NULL,     -- ISO YYYY-MM-DD, the reference period
    vintage_date     TEXT NOT NULL,     -- ISO YYYY-MM-DD, the SNAPSHOT date
    value            REAL,
    PRIMARY KEY (provider, country, series_id, observation_date, vintage_date)
);

CREATE TABLE schema_version (
    component TEXT PRIMARY KEY,
    version   INTEGER NOT NULL
);
```

WAL journal mode (`PRAGMA journal_mode=WAL`) is enabled so multiple
notebooks against the same DB do not block each other on writes.

Bootstrap is additive (`CREATE TABLE IF NOT EXISTS` + `INSERT OR
IGNORE`), so a cache file written by an earlier release gains the newer
tables on first open and keeps its rows. `schema_version` is seeded with
one row per table (`http_cache`, `alfred_vintages`, `connector_events`,
`realtime_vintages`, all at version 1).

## Real-time snapshots (`realtime_vintages`)

> Added in puremacro **3.4.0**.

Banxico, INEGI, BCB and BCCh publish only the edition that is current
when you ask — they overwrite in place and carry no vintage field — so
the only revision history available for them is the one this table
accumulates locally. Each fetch is stored stamped with its **snapshot
date**, and a later call returns every stored snapshot as one vintage.
Two captures on the same day are therefore one vintage, not two: the
primary key ends in `vintage_date` and the write is `INSERT OR REPLACE`,
so a same-day re-fetch overwrites rather than duplicates.

```python
import pandas as pd
import puremacro._cache_db as db

snap = pd.DataFrame({
    "provider": ["bcb"] * 3,
    "country": ["BRA"] * 3,
    "series_id": ["432"] * 3,
    "date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
    "vintage": pd.to_datetime(["2026-03-15"] * 3),
    "value": [11.25, 11.25, 11.00],
})

db.store_realtime_vintages(snap)                       # -> 3 rows written
db.query_realtime_vintages("bcb", "BRA", "432")        # every stored snapshot
db.query_realtime_vintages("bcb", "BRA", "432",        # as-of cutoff:
                           vintage_date="2026-03-01")  # keeps vintage_date <= cutoff
db.record_connector_event("bcb", "success", "none")    # fetch telemetry
```

- `store_realtime_vintages(df)` accepts `date` / `vintage` or
  `observation_date` / `vintage_date` column names, upper-cases the
  country, coerces the two dates and the value, skips rows it cannot
  parse, and returns the number of rows written. A missing required
  column raises `ValueError`.
- `query_realtime_vintages(provider, country, series_id, *, vintage_date=None)`
  returns `[date, vintage, value, provider, country, series_id]` ordered
  by `(observation_date, vintage_date)`, or an empty frame with those
  columns. `vintage_date` is an **as-of cutoff**, not an exact match: it
  keeps the snapshots taken on or before that day, which is what
  reconstructs the information set a policymaker had.
- `record_connector_event(source, outcome, fallback_used)` writes one
  row of fetch telemetry; see
  [`docs/CONNECTOR_HEALTH.md`](CONNECTOR_HEALTH.md).

Storage is a convenience, never the point: a snapshot that cannot be
written is still returned to the caller with a `UserWarning`, and a
cache that cannot be read is reported and treated as empty.

## Migration from 0.65.0

The HTTP-cache module runs the migration lazily on first read/write
after upgrade: if `cache_dir/*.bin` files exist and `http_cache` is
empty, the entries are inserted into the DB (without deleting the
originals) and a UserWarning points to the CLI:

```bash
python tools/cache_migrate.py              # dry-run; report count
python tools/cache_migrate.py --apply      # migrate
python tools/cache_migrate.py --apply --rm # migrate + delete originals
```

The migration is idempotent — re-running is a no-op.

## Introspection

```python
import puremacro.cache as C
import pandas as pd

C.http_list_urls()                                 # sorted list of cached URLs
C.http_cache_size_bytes()                          # total body bytes
C.http_cache_clear()                               # clear ALL entries; returns count
C.http_cache_clear(older_than=pd.Timedelta(days=30))  # clear stale only
```

Large deletions (>1000 rows) automatically issue `VACUUM` so the
on-disk file actually shrinks.

## Failure semantics

`cache_read` / `cache_write` must never raise to the caller (this is a
load-bearing contract from 0.65.0). DB failures emit a `UserWarning`
and degrade gracefully — `cache_read` returns `None`, `cache_write`
no-ops. A research notebook running a 30-source aggregation just gets
slower (no cache); it never crashes.

## Pyodide

`sqlite3` is Python stdlib — available on every supported runtime
including Pyodide. The cache file lives on the Pyodide virtual
filesystem; persistence across page reloads requires the user to
mount IDBFS or equivalent.
