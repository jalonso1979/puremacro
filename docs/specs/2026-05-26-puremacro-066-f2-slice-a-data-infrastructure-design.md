# puremacro 0.66.0 — F2 Slice A: data-infrastructure foundation (credentials + SQLite cache + ALFRED vintage store + parser schema versioning)

**Status:** Drafted 2026-05-26. Architectural spec for the first slice of the F2 sub-project from the post-Slice-1 sibling roadmap. Implementation in four sub-slices within one 0.66.0 release window.
**Target releases:** 0.66.0 (Slice A — credentials + cache + vintages + schema framework). Slice B (F2.4 governed fallback + F2.5 health telemetry) is queued for 0.67.0.
**Driving lenses:** researcher-facing platform reliability; honest "what does this need from you?" UX for API-keyed fetchers; foundation that F2.4 / F2.5 / F1 / F3 all build on; zero new runtime dependencies (stdlib `sqlite3` + `tomllib`).

## Motivation

After Slice 1 of the signal contract (puremacro 0.65.0), the package's narrative-index surface is well-defined but the data-acquisition layer underneath has four acute reliability gaps:

1. **API-key handling is inconsistent across fetchers.** Every consumer reads `os.environ.get("FRED_API_KEY")` (or `BEA_API_KEY`, `ANTHROPIC_API_KEY`, …) directly with hand-rolled error messages. No discovery beyond env vars; no introspection ("which keys are configured?"); no shared registry of what each service is for or where to get a key. A researcher trying puremacro for the first time hits cryptic `RuntimeError` instead of an actionable next step.
2. **The flat-file HTTP cache is opaque.** `~/.cache/puremacro/http/*.bin + *.json` works but can't be queried, can't be compacted, and fragments the disk. There's no way to ask "what's cached?", "how big is the cache?", or "expire everything older than 30 days." When SQLite is stdlib and would unify all of these, the flat-file cache is technical debt.
3. **ALFRED vintage panels are refetched on every notebook reload.** `vintages.py` has clean `as_of()` slicing but no persistent store. A researcher doing vintage analysis pays the FRED-ALFRED API cost on every kernel restart — slow, and consumes the user's API quota.
4. **Narrative parsers fail silently on upstream layout drift.** When `federalreserve.gov` reshuffles the Beige Book HTML or EUR-Lex bumps a JSON schema, the parser quietly emits broken records. No alarm fires until a researcher notices the index has gone flat. There's no schema-version contract on connectors.

These four gaps are independently painful and tightly coupled through the data layer. F2 Slice A addresses all four with a single coherent SQLite-backed infrastructure, plus a credentials module that every fetcher consults.

## Non-goals

- **No** F2.4 governed-fallback rewrite. The current `live → Wayback` fallbacks in `narrative/sources/_wayback.py` and the per-connector `try/except` blocks keep working as-is. Slice B replaces them with a unified policy.
- **No** F2.5 health telemetry. The `cache_events` / fallback-events tables — required to surface per-connector success/fail/fallback rates over time — land in Slice B (0.67.0).
- **No** rollout of `PARSER_SCHEMA_VERSION` to the remaining ~50 connectors. Slice A ships the framework + 8 high-value adoptions; subsequent slices roll out the rest.
- **No** new external runtime dependencies. `sqlite3` and `tomllib` are stdlib (Python ≥3.12).
- **No** macOS Keychain / system-keyring credential storage. Env vars + TOML config file is the 80% solution; keyring integration is a Slice B+ option behind an optional dep.
- **No** non-FRED ALFRED-style real-time stores. We focus the vintage store on FRED-ALFRED because that's the only consumer wired up; the schema is general enough to extend later but no other source uses it in this slice.

## Architecture

### Module map (deltas only)

```
puremacro/
├── _cache_db.py                 [NEW]      SQLite connection manager.
│                                           Singleton-per-process connection, WAL pragma,
│                                           bootstrap_schema(), migrate_from_flat_files(),
│                                           default_db_path(), close_conn().
├── _http_cache.py               [REWRITTEN] Same public API (cache_read / cache_write /
│                                           cache_key / default_cache_dir) but routes
│                                           through _cache_db. Lazy migration trigger on
│                                           first call after upgrade.
├── _http.py                     [UNCHANGED] Public API and behavior unchanged.
├── cache.py                     [EXTENDED]  New public helpers: list_urls(), size_bytes(),
│                                           clear(older_than=...).
├── credentials.py               [NEW]      ServiceCredentialSpec + SERVICES registry +
│                                           MissingCredentialError + get() / require() /
│                                           status() + default_config_path() + TOML loader.
├── vintages.py                  [EXTENDED]  AlfredVintageStore class +
│                                           as_of_from_store() helper. Existing
│                                           as_of(panel, ...) unchanged.
├── fetch/
│   ├── fred.py                  [EXTENDED]  fetch_fred_alfred() gains `store=` and
│                                           `refresh=` kwargs. Other consumers in this
│                                           file switch to credentials.require("fred").
│   ├── fred_states.py           [UPDATED]   credentials.require("fred")
│   ├── census_bfs.py            [UPDATED]   credentials.require("census")
│   ├── frb_phil_coincident.py   [UPDATED]   credentials.require("fred")
│   ├── bea_cainc.py             [UPDATED]   credentials.require("bea")
│   └── bea_industry_shares.py   [UPDATED]   credentials.require("bea")
├── narrative/
│   ├── scoring/llm.py           [UPDATED]   credentials.require("anthropic"|"openai")
│   ├── indices/_llm_kernel.py   [UPDATED]   credentials.require("anthropic"|"openai")
│   └── sources/
│       ├── _schema_check.py     [NEW]      ParserSchemaMismatchError + assert_landmarks()
│       ├── _fixtures/           [NEW]      Per-connector golden HTML/JSON snapshots
│       │   ├── beige_book_v1.html
│       │   ├── eu_eurlex_v1.html
│       │   ├── eu_parliament_v1.html
│       │   ├── us_cbo_v1.html
│       │   ├── fed_minutes_v1.html
│       │   ├── fed_speeches_v1.html
│       │   ├── bluesky_v1.json
│       │   └── ecb_press_v1.html
│       ├── beige_book.py        [UPDATED]   PARSER_SCHEMA_VERSION = 1 + assert_landmarks
│       ├── eu_eurlex.py         [UPDATED]   same
│       ├── eu_parliament.py     [UPDATED]   same
│       ├── us_cbo.py            [UPDATED]   same
│       ├── fed_minutes.py       [UPDATED]   same
│       ├── fed_speeches.py      [UPDATED]   same
│       ├── bluesky.py           [UPDATED]   same
│       └── ecb_press.py         [UPDATED]   same
└── instruments/_catalog.py      [UPDATED]   credentials.require("fred"|...) where used

tools/
└── cache_migrate.py             [NEW]      One-shot CLI: dry-run / --apply / --apply --rm.
                                            Walks ~/.cache/puremacro/http/*.bin sidecars,
                                            calls _cache_db.migrate_from_flat_files().

docs/
├── CREDENTIALS.md               [NEW]      Resolver priority, env vars, TOML format,
│                                           status() introspection, per-service signup URLs.
└── CACHE_DB.md                  [NEW]      Schema, env vars, introspection helpers,
                                            migration notes.

notebooks/R5_data_infra/
└── R5_01_cache_and_credentials_demo.ipynb  [NEW]   Schema demo (paired builder
                                                    tools/make_notebook_R5_01.py).

pyproject.toml                   [MODIFIED]  requires-python = ">=3.12";
                                            version = "0.66.0".

puremacro/__init__.py            [MODIFIED]  __version__ = "0.66.0".

CHANGELOG.md                     [MODIFIED]  Prepend 0.66.0 section.

ARCHITECTURE.md                  [EXTENDED]  Add "Data infrastructure (0.66.0+)" subsection.
README.md                        [EXTENDED]  Quickstart block: credentials.status() + the
                                            cache introspection helpers.
```

### F2.0 — credentials module (`puremacro/credentials.py`)

```python
@dataclass(frozen=True)
class ServiceCredentialSpec:
    name:        str                       # canonical, lowercased: 'fred', 'bea', 'anthropic'
    env_vars:    tuple[str, ...]           # priority order; first hit wins
    signup_url:  str                       # public docs URL where the user can get a key
    description: str                       # one-line "what is this for"


SERVICES: dict[str, ServiceCredentialSpec] = {
    "fred":      ServiceCredentialSpec(
        name="fred",
        env_vars=("FRED_API_KEY", "PUREMACRO_FRED_API_KEY"),
        signup_url="https://fred.stlouisfed.org/docs/api/api_key.html",
        description="FRED + ALFRED real-time macro data (St. Louis Fed)",
    ),
    "bea":       ServiceCredentialSpec(
        name="bea",
        env_vars=("BEA_API_KEY", "PUREMACRO_BEA_API_KEY"),
        signup_url="https://apps.bea.gov/API/signup/",
        description="BEA NIPA / regional / industry tables",
    ),
    "anthropic": ServiceCredentialSpec(
        name="anthropic",
        env_vars=("ANTHROPIC_API_KEY", "PUREMACRO_ANTHROPIC_API_KEY"),
        signup_url="https://console.anthropic.com/settings/keys",
        description="LLM-scored narrative kernels (narrative.scoring.llm)",
    ),
    "openai":    ServiceCredentialSpec(
        name="openai",
        env_vars=("OPENAI_API_KEY", "PUREMACRO_OPENAI_API_KEY"),
        signup_url="https://platform.openai.com/api-keys",
        description="OpenAI provider for the LLM kernel (alternative to Anthropic)",
    ),
    "census":    ServiceCredentialSpec(
        name="census",
        env_vars=("CENSUS_API_KEY", "PUREMACRO_CENSUS_API_KEY"),
        signup_url="https://api.census.gov/data/key_signup.html",
        description="Census BFS / ACS connectors",
    ),
}


class MissingCredentialError(RuntimeError):
    """Raised by `require()` when a fetcher needs an API key but none is found.

    Message structure (assertable in tests):
      "<service description> needs an API key. Checked env vars (in order):
       <var1>, <var2>. Checked config file: <path> (<found|not found>).
       Get a free key at: <signup_url>"
    """


def default_config_path() -> Path:
    """`$PUREMACRO_CREDENTIALS_FILE` if set; else
       `$XDG_CONFIG_HOME/puremacro/credentials.toml` if XDG_CONFIG_HOME set;
       else `~/.puremacro/credentials.toml`."""


def get(service: str, *, explicit: str | None = None) -> str | None:
    """Resolve an API key for `service`. Priority:
       1. `explicit` kwarg (caller-passed; wins everything)
       2. Env vars in SERVICES[service].env_vars, tried in order
       3. Config file `[service].api_key`
       4. None
       Lookup is side-effect-free; never raises (callers decide whether
       missing == error). Unknown service raises KeyError early."""


def require(service: str, *, explicit: str | None = None) -> str:
    """Like `get(service)` but raises MissingCredentialError with a
       helpful message if no key is found."""


def status() -> pd.DataFrame:
    """One row per service in SERVICES. Columns:
         service, configured (bool), source, description, signup_url
       `source` ∈ {'env:VAR_NAME', 'config_file', 'missing'}.
       Never includes the actual key (regex-asserted in tests)."""
```

**Config file format** (`~/.puremacro/credentials.toml`, optional):

```toml
[fred]
api_key = "abc123..."

[bea]
api_key = "..."

[anthropic]
api_key = "sk-ant-..."
```

**Fetcher integration pattern** — every keyed consumer changes from:
```python
key = os.environ.get("FRED_API_KEY")
if not key:
    raise RuntimeError("FRED_API_KEY must be set in environment")
return Fred(api_key=key)
```
to:
```python
from puremacro import credentials
key = credentials.require("fred", explicit=api_key)
return Fred(api_key=key)
```

Slice A rollout: `fetch/fred.py`, `fetch/fred_states.py`, `fetch/census_bfs.py`, `fetch/frb_phil_coincident.py`, `fetch/bea_cainc.py`, `fetch/bea_industry_shares.py`, `narrative/scoring/llm.py`, `narrative/indices/_llm_kernel.py`, `instruments/_catalog.py`. Lint test `tests/test_credentials/test_no_direct_env_get_in_fetch.py` AST-scans these directories and fails the build if any direct `os.environ.get("*_API_KEY")` remains.

### F2.1 — SQLite cache backend

**Single SQLite file** at `~/.cache/puremacro/cache.db` (or `$PUREMACRO_HTTP_CACHE_DIR`, re-interpreted: if it ends in `.db` use that path verbatim; else treat as a directory and use `<dir>/cache.db`).

**Schema:**

```sql
CREATE TABLE http_cache (
    key            TEXT PRIMARY KEY,    -- sha256(url) hex
    url            TEXT NOT NULL,
    fetched_at     INTEGER NOT NULL,    -- unix epoch seconds
    content_type   TEXT,
    body           BLOB NOT NULL
);
CREATE INDEX http_cache_fetched_at_idx ON http_cache(fetched_at);

CREATE TABLE alfred_vintages (
    series_id        TEXT NOT NULL,
    observation_date TEXT NOT NULL,     -- ISO YYYY-MM-DD
    vintage_date     TEXT NOT NULL,     -- ISO YYYY-MM-DD
    value            REAL,
    PRIMARY KEY (series_id, observation_date, vintage_date)
);
CREATE INDEX alfred_vintages_series_vintage_idx
    ON alfred_vintages(series_id, vintage_date);

CREATE TABLE schema_version (
    component TEXT PRIMARY KEY,         -- 'http_cache' | 'alfred_vintages'
    version   INTEGER NOT NULL
);
-- Bootstrap seed inserted by bootstrap_schema():
--   ('http_cache', 1), ('alfred_vintages', 1)
```

**Connection management** in `_cache_db.py`:
- `PRAGMA journal_mode=WAL` for concurrent reader/writer support.
- `PRAGMA synchronous=NORMAL` (cache durability is best-effort, not banking).
- `PRAGMA foreign_keys=ON` (defensive even though we have no FKs today).
- One `sqlite3.Connection` per process, lazy-opened on first `get_conn()`, kept alive in a module-level singleton.
- `sqlite3.connect(..., timeout=30.0)` to ride out brief contention.
- `close_conn()` available for tests.

**Public API in `_http_cache.py`** — preserved verbatim, only the storage internals change:

```python
def cache_read(cache_dir, url, ttl_seconds=30*24*3600) -> bytes | None: ...
def cache_write(cache_dir, url, body, content_type=None) -> None: ...
def cache_key(url: str) -> str: ...
def default_cache_dir() -> Path: ...    # returns the DB's parent dir for compat
```

**Lazy migration trigger**: on the first `cache_read` or `cache_write` after upgrade, if the legacy `<cache_dir>/*.bin` files exist AND `http_cache` is empty, run `migrate_from_flat_files(remove=False)` automatically with one `warnings.warn()` pointing to the `tools/cache_migrate.py --apply --rm` CLI for users who want to also delete the old files. Idempotent; failures fall through to a fresh empty DB (warn, don't raise).

**Migration CLI** (`tools/cache_migrate.py`):
```
python tools/cache_migrate.py              # dry-run; reports count
python tools/cache_migrate.py --apply      # actually migrate
python tools/cache_migrate.py --apply --rm # migrate + delete flat files
```

**Introspection helpers** in `puremacro/cache.py` (additive, no existing code touched):

```python
def list_urls() -> list[str]: ...                     # sorted
def size_bytes() -> int: ...                          # SUM(LENGTH(body))
def clear(older_than: pd.Timedelta | None = None) -> int: ...
                                                       # rows deleted; VACUUM if >1000
```

### F2.2 — ALFRED vintage store

**`AlfredVintageStore` class** in `puremacro/vintages.py`:

```python
class AlfredVintageStore:
    def __init__(self, db_path: Path | None = None): ...
    def put(self, series_id, observation_date, vintage_date, value) -> None: ...
    def put_many(self, df: pd.DataFrame) -> int: ...
        # df.columns must be ['series_id','observation_date','vintage_date','value']
        # INSERT OR REPLACE — re-inserting the same triple overwrites
    def get(self, series_id, *, vintage_until: str | None = None) -> pd.DataFrame: ...
        # returns ['observation_date','vintage_date','value']; empty on missing series
    def has_series(self, series_id: str) -> bool: ...
    def series_list(self) -> list[str]: ...
    def coverage(self, series_id: str) -> dict | None: ...
        # {'n_rows','first_obs','last_obs','first_vintage','last_vintage','n_vintages'}
        # None if not has_series
```

**Convenience helper** in `vintages.py`:

```python
def as_of_from_store(series_id: str, vintage_date: str,
                     store: AlfredVintageStore) -> pd.Series:
    """Convenience: pull series from store, then apply as_of()."""
```

**Integration with `fetch_fred_alfred`** in `puremacro/fetch/fred.py` — opt-in via new kwargs:

```python
def fetch_fred_alfred(
    series_id: str,
    *,
    api_key: str | None = None,
    vintage_until: str | None = None,
    # ... existing kwargs unchanged ...
    store: "AlfredVintageStore | None" = None,
    refresh: bool = False,
) -> pd.DataFrame: ...
```

Gap-fill semantics with `store=` provided:
1. `credentials.require("fred", explicit=api_key)` resolves the key.
2. If `not refresh` and `store.has_series(series_id)`: read with `store.get(series_id, vintage_until=...)`.
3. If the store's max vintage covers `vintage_until` (or `vintage_until is None`): return store rows; no API call.
4. Otherwise call the API for the gap `(store_max_vintage + 1d) .. vintage_until`.
5. `store.put_many(new_rows)` for everything just fetched.
6. Return the combined DataFrame in the existing format.

`vintages.as_of(df, vintage_date)` continues to work on any in-memory panel — totally unchanged.

### F2.3 — parser schema versioning

**`puremacro/narrative/sources/_schema_check.py`**:

```python
class ParserSchemaMismatchError(RuntimeError):
    """Raised by assert_landmarks() when a parser's expected upstream
    landmarks are missing — i.e., the source's HTML/JSON layout has
    drifted away from what the parser was written against.

    Caught by the iter_<source> wrapper, which yields empty and emits
    a UserWarning naming the connector and the missing landmark
    (per RETRY_POLICY.md §4.1: yield, don't raise)."""


def assert_landmarks(
    text: str,
    *,
    source: str,                        # e.g., 'beige_book'
    expected_version: int,
    landmarks: list[str | tuple[str, str]],
) -> None:
    """Raise ParserSchemaMismatchError if any landmark is missing.

    landmarks: list of either:
      - str: a substring that must appear in `text`
      - (selector, expected_text): a CSS-style hint for HTML callers
        (the selector is informational; the check is `expected_text in text`)
    """
```

**Per-connector pattern** (applied to the 8 named connectors):

```python
# Module-level constant — coverage test enforces this exists.
PARSER_SCHEMA_VERSION = 1


def _parse_beige_book_html(html: str) -> Iterable[tuple]:
    assert_landmarks(
        html,
        source="beige_book",
        expected_version=PARSER_SCHEMA_VERSION,
        landmarks=[
            "Beige Book",
            "Summary of Commentary on Current Economic Conditions",
            ("h1", "Beige Book"),
        ],
    )
    # ... existing parsing ...
```

**Golden fixtures** in `narrative/sources/_fixtures/<source>_v<N>.html|.json` — a saved upstream snapshot per (source, version). A pytest test parses the fixture and asserts the parser produces an expected record count; this is a CI regression guard distinct from the runtime landmark check.

**`iter_<source>` wrapper integration** — each `iter_*` generator's outermost loop already wraps a `try/except`-style "yield, don't raise" pattern from RETRY_POLICY.md §4.1. Slice A extends those wrappers to catch `ParserSchemaMismatchError` explicitly and `warnings.warn` with the connector + missing landmark before yielding empty.

**Slice A rollout**: 8 connectors — `beige_book.py`, `eu_eurlex.py`, `eu_parliament.py`, `us_cbo.py`, `fed_minutes.py`, `fed_speeches.py`, `bluesky.py`, `ecb_press.py`. Coverage test `tests/test_narrative_schema_checks/test_coverage_assertion.py` enforces this list (via AST scan for `PARSER_SCHEMA_VERSION` constant + `assert_landmarks` call). The ~50 remaining connectors get `PARSER_SCHEMA_VERSION` in subsequent slices; coverage assertion intentionally only enforces the 8 in Slice A.

## Data flow

### A. Cached HTTP fetch (most narrative connectors)

```
caller (e.g., iter_beige_book)
  ↓
safe_get_text_cached(url)              ← public API, signature unchanged
  ↓
cache_read(cache_dir, url, ttl)        ← public API, signature unchanged
  ↓
_cache_db.get_conn(db_path)            ← singleton SQLite conn, WAL mode
  ↓
SELECT body, fetched_at FROM http_cache WHERE key = sha256(url)
  ├── hit + fresh → return bytes
  └── miss / stale ↓
       safe_get_text(url)              ← live HTTP per RETRY_POLICY.md
       ↓
       cache_write → INSERT OR REPLACE INTO http_cache
       ↓
       return bytes
  ↓
back in caller:
assert_landmarks(text, source='beige_book',
                 expected_version=1, landmarks=[...])
  ├── all present → continue parsing
  └── any missing → raise ParserSchemaMismatchError
                    caught by iter_beige_book wrapper
                    → warnings.warn + yield empty
```

### B. ALFRED vintage fetch (research notebook)

```
caller
  ↓
fetch_fred_alfred('GDPC1', vintage_until='2020-01-01',
                  store=my_store, refresh=False)
  ↓
credentials.require('fred', explicit=api_key)
  ├── found → continue
  └── missing → raise MissingCredentialError
  ↓
store.has_series('GDPC1')?
  ├── yes ↓
  │   df_existing = store.get('GDPC1', vintage_until='2020-01-01')
  │   gap = '2020-01-01' - store_max_vintage
  │   if gap empty: return df_existing  (no API call)
  └── no → fetch full window from ALFRED
  ↓
  api_rows = ALFRED API call for the gap
  ↓
  store.put_many(api_rows)
  ↓
  return pd.concat([df_existing, api_rows])
```

Subsequent `vintages.as_of(df, vintage_date)` or `vintages.as_of_from_store('GDPC1', '2020-01-01', store)` slices the panel.

## Failure semantics

| Failure mode | Where | Behavior | Why |
|---|---|---|---|
| Missing API key | `credentials.require()` | raise `MissingCredentialError` with service description, env vars checked, config-file path, and signup URL | user fault; needs actionable feedback |
| Unknown service in `credentials.get()` | `credentials.get()` | raise `KeyError` early | programmer error, fail loud |
| Malformed TOML config | `credentials._load_config` | `warnings.warn(UserWarning)` with file path + error; treat as no config | malformed config should not brick all credential resolution |
| Missing TOML config | `credentials._load_config` | silent — return empty dict | config file is optional |
| `sqlite3.OperationalError` on cache | `_http_cache.cache_read/write` | `warnings.warn(UserWarning)`, return None / no-op | preserves existing "cache failures must not break the caller" contract |
| `sqlite3.OperationalError` on vintage store | `AlfredVintageStore.*` | `warnings.warn(UserWarning)`; `get()` returns empty DataFrame; `put_many()` no-op | research path must keep going; user notebook slows but doesn't break |
| Migration failure (disk full, permission) | `_cache_db.migrate_from_flat_files` | `warnings.warn(UserWarning)`, return 0; fall through to fresh DB | migration is opportunistic, not load-bearing |
| WAL lock contention | `_cache_db.get_conn` | SQLite internal busy_timeout (30s); if still contended, surface as cache miss | rare; one notebook waits at most 30s |
| `ParserSchemaMismatchError` | `_schema_check.assert_landmarks` | raise; caught by `iter_<source>` wrapper → `warnings.warn` naming connector + missing landmark, generator yields empty | per `RETRY_POLICY.md` §4.1 "Yield, don't raise"; a 30-source aggregation loses 1, keeps 29 |

**Cross-cutting principle**: the new infrastructure must never make the existing happy path slower or more fragile. SQLite cache hits target <5ms p95; `credentials.require()` adds <1ms; `assert_landmarks` adds <10ms. Stated as design constraints; not enforced by tests, but called out so any obvious slow-path regression is caught in review.

## Pyodide contract

- `puremacro/_cache_db.py`, `puremacro/credentials.py`, `puremacro/narrative/sources/_schema_check.py`: stdlib-only (`sqlite3`, `tomllib`, `pathlib`, `hashlib`, `os`, `time`, `dataclasses`, `re`). All Pyodide-pure under Python ≥3.12.
- `puremacro/vintages.py` extension: pandas-only. Pyodide-pure.
- Cache-DB path resolves to the Pyodide virtual filesystem (`~/.cache/puremacro/cache.db` becomes `/home/pyodide/.cache/puremacro/cache.db`). Persistence across page reloads requires the user to mount IDBFS or equivalent — out of scope for this spec; documented in `docs/CACHE_DB.md` as a known limitation.
- The TOML config file is read from `~/.puremacro/credentials.toml` on the Pyodide virtual FS; same persistence caveat applies.
- `pyproject.toml` `requires-python` bumps from `>=3.10` to `>=3.12`. `tomllib` (3.11+) and the new generic-class syntax (`class Foo[T]:`, 3.12+) become available; no code in Slice A actively uses 3.12-only syntax in shipped runtime, but the package opens that door for future slices.

Extended `tests/test_pyodide_compat.py` asserts none of the above leak forbidden modules at import time.

## Testing

### Headline tests by sub-component

**F2.0 credentials** (`tests/test_credentials/`):
1. Priority order — explicit kwarg > primary env var > secondary env var > config file > `None`. Five scenarios via `monkeypatch.delenv()`.
2. `MissingCredentialError` message — assert message contains service description, every env var checked, config-file path, and signup URL.
3. AST lint (`test_no_direct_env_get_in_fetch.py`) — scans `puremacro/fetch/`, `puremacro/narrative/scoring/`, `puremacro/instruments/`; any direct `os.environ.get(...)` matching `_API_KEY` fails the test with file:line.
4. `status()` DataFrame — every column present; rows == `SERVICES.keys()`; values never include the actual key (regex-asserted).
5. Config-file parsing — valid TOML round-trips; malformed TOML warns + falls through; missing file returns empty dict silently.

**F2.1 cache** (`tests/test_cache_db/`):
6. Schema bootstrap — idempotent; all three tables present; `schema_version` seeded.
7. Roundtrip with TTL — write a key, sleep `ttl + 0.1s`, read → `None`; write again, immediate read → expected bytes.
8. Migration idempotency — fixture flat-file dir → migrate twice; second run inserts 0 rows; with `--rm`, flat files removed after success.
9. Concurrent WAL — two threads, one writes 100 keys / one reads 100 (mixed hit/miss); assert no `OperationalError`.
10. Introspection helpers — `list_urls`, `size_bytes`, `clear`; `clear(older_than=...)` deletes only stale rows; `VACUUM` runs after large deletes.
11. Failure mode — patch `sqlite3.Connection.execute` to raise; cache reads return `None` with warning; writes no-op with warning.

**F2.2 vintage store** (`tests/test_vintages_alfred_store/`):
12. Store roundtrip — `put_many` → `get` → equal frames.
13. Vintage-until filter — `get(vintage_until=)` filters correctly.
14. `has_series`, `series_list`, `coverage` — basic introspection.
15. `fetch_fred_alfred` with store, gap-fill — preload through `2019-12-31`; call with `vintage_until='2020-06-30'` and a mocked ALFRED API patch returning synthetic Jan–Jun rows; assert store's final state == union, exactly 1 API call.
16. `fetch_fred_alfred` with store, no-refresh shortcut — store covers the window; API mock called 0 times.
17. `as_of_from_store` end-to-end — store → `as_of_from_store` → expected `pd.Series`.
18. Store failure mode — DB locked → `get` returns empty + warn; `put_many` → no-op + warn.

**F2.3 schema versioning** (`tests/test_narrative_schema_checks/`):
19. Framework — `assert_landmarks` raises on missing landmark; passes on present; error message names connector + missing landmark.
20. Per-connector parametrized — each of the 8 has `PARSER_SCHEMA_VERSION` constant; importing the module produces no side-effect error.
21. Golden-fixture regression — each connector parses its `_fixtures/<source>_v1.{html,json}` and produces an expected record count.
22. `iter_<source>` swallow-and-yield-empty — patch inner parser to raise `ParserSchemaMismatchError`; assert wrapper yields `()` + emits `UserWarning` with the right text.
23. Coverage assertion — AST-scan the 8 connectors; fail the build if any lacks `PARSER_SCHEMA_VERSION` or doesn't call `assert_landmarks`.

**Cross-cutting:**
24. Pyodide-compat extension — importing `_cache_db`, `credentials`, `_schema_check`, `AlfredVintageStore` does not leak forbidden modules into `sys.modules`. `sqlite3` and `tomllib` are stdlib and allowed.
25. Backwards-compat sweep — fresh-clone simulation: wipe `~/.cache/puremacro/`, run a small fetch loop against EPU + LUI fixtures, assert no behavioral diff vs. 0.65.0 baseline.

## Staging

Four sub-slices inside the 0.66.0 release window, in dependency order. Each is its own series of commits, runnable independently.

**Sub-slice 1 — F2.0 credentials** (~6 commits)
1. `puremacro/credentials.py` + `SERVICES` registry + `MissingCredentialError` + `get` / `require` / `status` + tests.
2. `default_config_path()` + TOML loader + tests for parsing + malformed-TOML behavior.
3. AST lint test (`test_no_direct_env_get_in_fetch.py`) — initially fails.
4. Rollout commit 1: `fetch/fred.py`, `fetch/fred_states.py`, `fetch/census_bfs.py`, `fetch/frb_phil_coincident.py`, `fetch/bea_cainc.py`, `fetch/bea_industry_shares.py` switched.
5. Rollout commit 2: `narrative/scoring/llm.py`, `narrative/indices/_llm_kernel.py`, `instruments/_catalog.py` switched.
6. `docs/CREDENTIALS.md` + README quickstart block showing `credentials.status()`.

Lint test passes only after step 5 → all six commits must land before the 0.66.0 tag.

**Sub-slice 2 — F2.1 cache backend** (~5 commits)
7. `puremacro/_cache_db.py` + tests for schema bootstrap, WAL pragma, concurrent access.
8. Rewrite `puremacro/_http_cache.py` against `_cache_db`; preserve `cache_read` / `cache_write` / `cache_key` / `default_cache_dir` signatures. Tests for roundtrip + TTL + failure modes.
9. Lazy migration trigger in `_http_cache` + tests for idempotency.
10. `puremacro/cache.py` introspection helpers + tests.
11. `tools/cache_migrate.py` CLI + a small smoke test.

**Sub-slice 3 — F2.2 vintage store** (~3 commits)
12. `AlfredVintageStore` class + tests for roundtrip, vintage filter, coverage diagnostic.
13. `as_of_from_store` helper + tests.
14. `fetch_fred_alfred(store=, refresh=)` integration + tests for gap-fill, no-refresh shortcut, store-failure fallthrough.

**Sub-slice 4 — F2.3 schema versioning** (~4 commits)
15. `_schema_check.py` framework + tests for framework + `iter_<source>` swallow-yield-empty.
16. Rollout commit 1: `beige_book.py`, `eu_eurlex.py`, `eu_parliament.py`, `us_cbo.py` + golden fixtures.
17. Rollout commit 2: `fed_minutes.py`, `fed_speeches.py`, `bluesky.py`, `ecb_press.py` + golden fixtures.
18. Coverage assertion test enforcing the 8-connector list.

**Final release commits (~2)**
19. Notebook deliverable: `notebooks/R5_data_infra/R5_01_cache_and_credentials_demo.ipynb` + paired `tools/make_notebook_R5_01.py`. Shows `credentials.status()`, migration on first cache hit, `cache.size_bytes()` before/after, `AlfredVintageStore` populated from mocked-API fixture, `vintages.as_of_from_store`. Executed foreground-side per long-nbconvert rule.
20. Version bump + CHANGELOG: `pyproject.toml` `version = "0.66.0"` and `requires-python = ">=3.12"`; `puremacro/__init__.py` `__version__ = "0.66.0"`; CHANGELOG prepends 0.66.0 entry with F2.0/F2.1/F2.2/F2.3 sections and a roadmap pointer to F2.4/F2.5 (Slice B → 0.67.0).

Total: ~20 commits.

**Critical-path dependency**: F2.0 must land first (the cache module imports `credentials` for FRED-API-key-aware ALFRED store integration). F2.1 must land before F2.2 (the vintage store reuses the DB connection). F2.3 is independent and could parallelize with F2.1+F2.2.

## Done-definition for Slice A (0.66.0)

- All 4 sub-components shipped per their sub-slice.
- Lint test bans direct `os.environ.get("*_API_KEY")` in `puremacro/fetch/`, `puremacro/narrative/scoring/`, `puremacro/instruments/`.
- 8 named connectors all declare `PARSER_SCHEMA_VERSION` and call `assert_landmarks`.
- Cache migration runs idempotently; existing users keep their cached fetches.
- Notebook R5_01 executes cleanly and is committed alongside its builder.
- `docs/CREDENTIALS.md` and `docs/CACHE_DB.md` shipped.
- `requires-python = ">=3.12"` in pyproject.toml.
- Pyodide-compat test passes (sqlite3 + tomllib stdlib, allowed).
- Full narrative test suite shows no NEW regressions vs. the post-Slice-1 baseline.

## Open follow-ups (queued for later slices)

- **Slice B (0.67.0)** — F2.4 governed fallback (unified live → Wayback → Playwright → fail policy) + F2.5 health telemetry (per-connector success/fail/fallback rates over time, surfaced via `cache.alfred_status()` / `cache.connector_health()`).
- Roll out `PARSER_SCHEMA_VERSION` to the remaining ~50 connectors in `narrative/sources/`.
- Optional macOS Keychain / system-keyring credential storage via `keyring` (behind an optional extra).
- IDBFS mount documentation for Pyodide cache persistence (just docs, no code).
- Migration tooling for vintage stores: if FRED-ALFRED ever changes how it returns historical vintages, we need a `vintages_migrate` parallel to `cache_migrate`.
- Generalising the vintage store schema to non-FRED real-time sources (Eurostat, OECD) — currently FRED-shaped only.

---

## Spec self-review (inline)

- **Placeholder scan:** no TBD/TODO. Open-question (3.10 vs. 3.11 vs. 3.12 vs. 3.13) resolved to 3.12 before writing.
- **Internal consistency:** `SERVICES.keys()` listed identically in 5b, in the test plan, and in the rollout staging. The 8 connectors listed identically in the module map, the rollout, and the coverage test. `cache.db` path consistent across `_cache_db`, `_http_cache`, `AlfredVintageStore`, and the CLI.
- **Scope check:** four sub-components is large but each is independently testable and shippable inside the 0.66.0 window. Comparable in size to Slice 1 (which had 2 paired components). Single implementation plan can carry it.
- **Ambiguity check:** `default_cache_dir()` is preserved by name but now returns the DB's parent directory rather than the cache root — flagged explicitly in the F2.1 section. `default_db_path()` is the new canonical path API. Tests pin both.
