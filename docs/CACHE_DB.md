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

CREATE TABLE schema_version (
    component TEXT PRIMARY KEY,
    version   INTEGER NOT NULL
);
```

WAL journal mode (`PRAGMA journal_mode=WAL`) is enabled so multiple
notebooks against the same DB do not block each other on writes.

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
