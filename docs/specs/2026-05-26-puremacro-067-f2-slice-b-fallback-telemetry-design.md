# puremacro 0.67.0 — F2 Slice B: governed fallback + health telemetry (closes F2)

**Status:** Drafted 2026-05-26. Architectural spec for the second (and final) slice of the F2 sub-project from the post-Slice-1 sibling roadmap. Implementation in three sub-slices within one 0.67.0 release window.
**Target releases:** 0.67.0 (Slice B — fallback + telemetry). Closes F2.
**Driving lenses:** clean up the per-connector fallback duplication that has accumulated across 7 sources; give researchers visibility into per-connector health so they can answer "is EUR-Lex degrading this week?" without a one-off SQL query; zero new runtime dependencies; preserve every existing RETRY_POLICY.md guarantee.

## Motivation

Slice A (0.66.0) shipped the credentials module, the SQLite cache backend, the ALFRED vintage store, and the parser-schema-version framework — the data-infrastructure foundation. Two acute gaps remain in the F2 sub-project:

1. **Fallback logic is duplicated across 7 connectors.** Three connectors (`eu_eurlex`, `eu_parliament`, `us_cbo`) wrap `safe_get_text(url)` calls in hand-rolled `try/except → _wayback.wayback_snapshot_url(url) → safe_get_text(snapshot_url)` blocks. Four others (`rba`, `bok`, `riksbank`, `sarb`) wrap calls in `try/except → _playwright_helper.fetch_with_playwright(url)` blocks. The patterns are similar but not identical, and the per-connector divergence makes it hard to add a new fallback layer (Tor, paid proxy, etc.) without touching seven files.

2. **There is no health telemetry.** A researcher who notices `eu_eurlex` returning empty results today has no way to ask "how often has this been failing over the last month, and which fallback paths are serving it?" The only signal is `UserWarning`s emitted at fetch time, which are lost as soon as the notebook restarts.

F2 Slice B addresses both gaps with one new module each (`_fallback.py`, `_telemetry.py`), a new SQLite table (`connector_events`), and migrations of the 7 fallback connectors. Telemetry is also wired into the 8 Slice-A schema-checked connectors so parser-drift events show up in `connector_health()`.

## Non-goals

- **No** F2.4 rollout to the remaining ~53 narrative connectors. Slice C+ adopts as-needed.
- **No** F2.5 telemetry from connectors that don't use `fetch_with_fallback` or `assert_landmarks`. Those ~45 sources won't show up in `connector_health()` until they opt in. This is a documented scope limit, not a bug.
- **No** new fallback stages beyond `live`/`wayback`/`playwright`. Tor, paid-proxy, mirrored-S3 are all natural future additions; adding a new stage is intentionally a one-line `SUPPORTED_STAGES` change + a `_dispatch_stage` branch.
- **No** OpenTelemetry / Prometheus / external metrics exporters. Telemetry stays in the local SQLite DB for now.
- **No** per-event `url_hash`, `latency_ms`, or request-body size. Slice B keeps the event row minimal (`ts, source, outcome, fallback_used`).
- **No** retention controls (`connector_events_clear`). Added when the table grows large in practice (likely 0.68.0+).
- **No** new runtime dependencies. `sqlite3` is stdlib; pandas already in deps; Playwright stays optional (lazy-imported, same as Slice A).

## Architecture

### Module map (deltas only)

```
puremacro/
├── _cache_db.py                                  [extended] add connector_events DDL +
│                                                            index + ("connector_events", 1)
│                                                            in _SCHEMA_SEED.
├── narrative/sources/
│   ├── _fallback.py                              [NEW]      SUPPORTED_STAGES,
│   │                                                        FallbackExhaustedError,
│   │                                                        FallbackStageUnavailable,
│   │                                                        _classify,
│   │                                                        _dispatch_stage,
│   │                                                        _stage_live / _stage_wayback /
│   │                                                        _stage_playwright,
│   │                                                        fetch_with_fallback.
│   ├── _telemetry.py                             [NEW]      VALID_OUTCOMES,
│   │                                                        VALID_FALLBACK_USED,
│   │                                                        telemetry_enabled,
│   │                                                        log_event,
│   │                                                        connector_health.
│   ├── eu_eurlex.py                              [UPDATED]  FALLBACK_POLICY = ("live", "wayback");
│   │                                                        inline Wayback fallback removed in
│   │                                                        favor of fetch_with_fallback.
│   ├── eu_parliament.py                          [UPDATED]  same.
│   ├── us_cbo.py                                 [UPDATED]  same.
│   ├── rba.py                                    [UPDATED]  FALLBACK_POLICY = ("live", "playwright");
│   │                                                        inline Playwright fallback removed.
│   ├── bok.py                                    [UPDATED]  same.
│   ├── riksbank.py                               [UPDATED]  same.
│   ├── sarb.py                                   [UPDATED]  same.
│   ├── beige_book.py                             [UPDATED]  iter_<source> ParserSchemaMismatchError
│   ├── eu_eurlex.py                              [UPDATED]    handler also calls log_event(outcome=
│   ├── (the 6 other Slice-A connectors)          [UPDATED]    "parser_schema_mismatch", ...).
│   └── (existing _wayback.py + _playwright_helper.py)   unchanged; called by _fallback internals.

docs/
└── CONNECTOR_HEALTH.md                           [NEW]      researcher-facing reference for
                                                             connector_health() + the event schema.

notebooks/R5_data_infra/
└── R5_02_connector_health_demo.ipynb             [NEW]      synthetic + small live-fetch demo of
                                                             connector_health(), with paired
                                                             builder tools/make_notebook_R5_02.py.

ARCHITECTURE.md                                   [UPDATED]  add "F2 closure (0.67.0+)" note
                                                             under the Data infrastructure block.

pyproject.toml + __init__.py + CHANGELOG.md       [UPDATED]  version → 0.67.0.
```

### F2.4 — governed fallback

**`_fallback.py` public surface:**

```python
SUPPORTED_STAGES: frozenset[str] = frozenset({"live", "wayback", "playwright"})


class FallbackExhaustedError(RuntimeError):
    """Raised by fetch_with_fallback when every stage in the policy has been
    tried and none succeeded. Caught by iter_<source> wrappers per the
    yield-don't-raise contract in RETRY_POLICY.md §4.1."""


class FallbackStageUnavailable(RuntimeError):
    """Raised internally by a _stage_* function when its dependency is
    missing (e.g., Playwright not installed). Translated to
    outcome='playwright_unavailable' by fetch_with_fallback's loop and
    treated as a normal stage failure (continue to next stage)."""


def fetch_with_fallback(
    url: str,
    *,
    policy: tuple[str, ...],
    source: str,
    timeout: float = 30.0,
    use_cache: bool = True,
) -> str:
    """Try each stage in `policy` in order; return the body from the first
    one that succeeds. Raises FallbackExhaustedError if all stages fail.
    Emits one log_event per stage attempt (regardless of success/failure)."""
```

**Stage implementations:**

- `_stage_live(url, timeout, use_cache)` — calls `safe_get_text_cached(url, ...)` if `use_cache` else `safe_get_text(url, ...)`. Empty body or HTTP error → raise (caught by main loop).
- `_stage_wayback(url, timeout, use_cache)` — `_wayback.wayback_snapshot_url(url)`; `None` → raise `FallbackStageUnavailable("wayback_no_snapshot")`; otherwise `safe_get_text_cached(snapshot_url, ...)`.
- `_stage_playwright(url, timeout)` — lazy-imports `_playwright_helper.fetch_with_playwright(url)`. `ImportError` → raise `FallbackStageUnavailable("playwright_unavailable")`. No cache (Playwright is last resort).

**Exception classification** (`_classify(e) -> str`): maps common errors into the canonical outcome strings used by telemetry. Table:

| Exception | Outcome |
|---|---|
| `URLError`/`socket.timeout`/`TimeoutError` | `"timeout"` |
| `HTTPError(404)` | `"404"` |
| `HTTPError(500..599)` | `"server_5xx"` |
| `ssl.SSLError` | `"ssl_fail"` |
| `FallbackStageUnavailable("wayback_no_snapshot")` | `"wayback_no_snapshot"` |
| `FallbackStageUnavailable("playwright_unavailable")` | `"playwright_unavailable"` |
| anything else (catch-all) | `"other_network_error"` |

The mapping table is documented in `_fallback.py` and is the single source of truth for `VALID_OUTCOMES`.

**Per-connector wiring** — each of the 7 connectors:

```python
# top of eu_eurlex.py
from ._fallback import fetch_with_fallback, FallbackExhaustedError

FALLBACK_POLICY = ("live", "wayback")

def iter_eurlex(...):
    for celex_id in ...:
        url = _eurlex_url(celex_id)
        try:
            html = fetch_with_fallback(
                url, policy=FALLBACK_POLICY, source="eu_eurlex"
            )
        except FallbackExhaustedError as e:
            warnings.warn(
                f"eu_eurlex: all fallback stages failed for {celex_id}: {e}",
                UserWarning, stacklevel=2,
            )
            continue
        # ... existing parse + yield ...
```

**Per-connector policies:**

| Connector | `FALLBACK_POLICY` |
|---|---|
| `eu_eurlex` | `("live", "wayback")` |
| `eu_parliament` | `("live", "wayback")` |
| `us_cbo` | `("live", "wayback")` |
| `rba` | `("live", "playwright")` |
| `bok` | `("live", "playwright")` |
| `riksbank` | `("live", "playwright")` |
| `sarb` | `("live", "playwright")` |

Coverage assertion (`tests/test_narrative_fallback/test_coverage_assertion.py`) AST-scans each of the 7 named connectors and verifies:
1. `FALLBACK_POLICY` is declared as a tuple,
2. every element is in `SUPPORTED_STAGES`,
3. the module's body contains a call to `fetch_with_fallback(...)`.

### F2.5 — health telemetry

**Schema change** in `_cache_db.py`:

```sql
CREATE TABLE IF NOT EXISTS connector_events (
    ts             INTEGER NOT NULL,
    source         TEXT NOT NULL,
    outcome        TEXT NOT NULL,
    fallback_used  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS connector_events_ts_source_idx
    ON connector_events(ts, source);
```

`_SCHEMA_SEED` gains `("connector_events", 1)`. `bootstrap_schema` creates the table on first connect after upgrade.

**`_telemetry.py` public surface:**

```python
VALID_OUTCOMES = frozenset({
    "success", "404", "timeout", "ssl_fail", "server_5xx",
    "wayback_no_snapshot", "playwright_unavailable",
    "parser_schema_mismatch", "other_network_error",
})
VALID_FALLBACK_USED = frozenset({"live", "wayback", "playwright", "none"})


def telemetry_enabled() -> bool:
    """True unless `PUREMACRO_NARRATIVE_TELEMETRY=0` is set in the environment."""


def log_event(
    *,
    source: str,
    outcome: str,
    fallback_used: str = "none",
) -> None:
    """Insert one row into connector_events. Validates outcome and
    fallback_used against the registries above. Failures emit a
    UserWarning and no-op. Skipped silently if telemetry_enabled() is False."""


def connector_health(
    *,
    window: pd.Timedelta = pd.Timedelta(days=7),
    sources: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate connector_events over the last `window`. Returns one row per
    source with columns:
        source, n_total, n_success, success_rate,
        n_fallback, fallback_rate, last_seen
    Empty DataFrame with the expected columns if no events match."""
```

**Where events get emitted (Slice B scope):**

1. **`fetch_with_fallback`** — one event per stage attempt. `outcome="success"` for the winning stage; `_classify(e)` for each failure. `fallback_used` equals the stage name.
2. **`iter_<source>` wrappers for the 8 Slice-A schema-checked connectors** — when `ParserSchemaMismatchError` is caught, also call `log_event(source=name, outcome="parser_schema_mismatch", fallback_used="none")` before the existing `warnings.warn`.
3. **Other ~45 narrative connectors** — emit no events in Slice B. Documented scope limit. Opt-in path: adopt `fetch_with_fallback(policy=("live",))` even if no real fallback is wanted, OR call `log_event(...)` manually.

**Kill-switch**: `PUREMACRO_NARRATIVE_TELEMETRY=0` disables all event logging. Useful for offline runs, strict-reproducibility CI, or users who don't want any DB writes from fetcher code.

**Failure semantics**: every `log_event` call is wrapped in a `try/except` for `sqlite3.OperationalError`, `sqlite3.DatabaseError`, and `OSError`. On failure: `warnings.warn(UserWarning)` and return silently. Telemetry must NEVER break a fetch.

## Data flow

### A. Cached fetch with fallback (7 connectors)

```
caller (e.g., iter_eu_eurlex)
  ↓
fetch_with_fallback(url, policy=("live", "wayback"), source="eu_eurlex")
  ↓
for stage in policy:
   ├── _dispatch_stage("live", url)    → safe_get_text_cached(url)
   ├── _dispatch_stage("wayback", url) → _wayback.wayback_snapshot_url + cached fetch
   └── _dispatch_stage("playwright", url) → lazy _playwright_helper.fetch_with_playwright
   ↓
   succeeded?
     ├── yes → log_event(source, outcome="success", fallback_used=stage); return body
     └── no  → log_event(source, outcome=_classify(e), fallback_used=stage); continue

every stage failed → raise FallbackExhaustedError(...)
  ↓ caught by iter_eu_eurlex wrapper → warnings.warn + skip record
```

### B. Schema-mismatch event (8 Slice-A connectors)

```
body fetched (cached or live)
  ↓
assert_landmarks(body, source="beige_book", ...)
  ├── present → continue parsing
  └── missing → raise ParserSchemaMismatchError
         ↓ caught by iter_beige_book wrapper
         log_event(source="beige_book", outcome="parser_schema_mismatch", fallback_used="none")
         warnings.warn + yield empty
```

### C. Researcher introspection

```
researcher_notebook
  ↓
connector_health(window=pd.Timedelta(days=7))
  ↓
SELECT source, COUNT(*) AS n_total,
       SUM(outcome='success') AS n_success,
       SUM(fallback_used != 'live') AS n_fallback,
       MAX(ts) AS last_seen
FROM connector_events WHERE ts >= now - window
GROUP BY source
ORDER BY source
  ↓
pd.DataFrame[source, n_total, n_success, success_rate,
             n_fallback, fallback_rate, last_seen]
```

## Failure semantics

| Failure | Where | Behavior | Why |
|---|---|---|---|
| Unknown stage in `policy` | `fetch_with_fallback` validation | raise `ValueError` early | programmer error |
| Empty `policy` | `fetch_with_fallback` validation | raise `ValueError` early | programmer error |
| Single stage fails | `_dispatch_stage` | classify → `log_event`, continue to next stage | the whole point of the layer |
| All stages fail | `fetch_with_fallback` | raise `FallbackExhaustedError` with last-error chain | caught by `iter_<source>` wrapper → warn + skip per RETRY_POLICY.md §4.1 |
| Playwright not installed | `_stage_playwright` | raise `FallbackStageUnavailable`; classified as `"playwright_unavailable"`; continues | Playwright is optional extra |
| Wayback has no snapshot | `_stage_wayback` | `wayback_snapshot_url` returns None → raise; classified as `"wayback_no_snapshot"`; continues | not all URLs are archived |
| `log_event` DB error | `_telemetry.log_event` | `warnings.warn`, return silently | telemetry must never break a fetch |
| Malformed `outcome` / `fallback_used` | `log_event` validation | raise `ValueError` early | programmer error |
| `connector_health` DB error | `_telemetry.connector_health` | `warnings.warn`, return empty DataFrame | researcher introspection must degrade gracefully |
| `PUREMACRO_NARRATIVE_TELEMETRY=0` | `telemetry_enabled()` | `log_event` silently no-ops; `connector_health` reads existing rows | explicit kill-switch |

**Cross-cutting principle** (inherited from Slice A): the new infrastructure must never make the existing happy path slower or more fragile. Telemetry adds <1ms per fetch (single INSERT). Fallback adds zero overhead on the happy path (`"live"` stage just calls `safe_get_text_cached` directly).

## Pyodide contract

- `_fallback.py`: pure stdlib (`urllib.error`, `socket`, `ssl`, `warnings`) + the existing `_wayback`/`_playwright_helper` modules. Playwright stays lazy-imported. Pyodide-pure unless Playwright stage is exercised.
- `_telemetry.py`: pure stdlib (`sqlite3`, `os`, `time`, `warnings`) + pandas. Pyodide-pure.
- `_cache_db.py` extension: pure SQL DDL. Pyodide-pure.
- The kill-switch `PUREMACRO_NARRATIVE_TELEMETRY=0` is respected even in Pyodide where file writes go to the virtual FS.

Extended `tests/test_pyodide_compat.py` asserts none of the above leak forbidden modules at import time.

## Testing

Two new test directories, ~20 new test files.

### F2.4 — fallback (`tests/test_narrative_fallback/`)

1. **Happy path** — mock `_stage_live` to return body; `fetch_with_fallback(policy=("live","wayback"))` returns it; no call to `_stage_wayback`; one `log_event(outcome="success", fallback_used="live")`.
2. **Live fails → Wayback succeeds** — two `log_event` calls (one failure, one success); body returned.
3. **All stages fail** — `FallbackExhaustedError` raised; 2 failure events logged.
4. **`_classify` mapping** — parametrized over `URLError(timeout)` / `HTTPError(404)` / `SSLError` / `HTTPError(500)` / `ImportError` → expected outcome strings.
5. **Unknown stage in `policy`** → `ValueError` at function entry.
6. **Empty `policy`** → `ValueError`.
7. **Per-connector policies** — each of the 7 connectors declares `FALLBACK_POLICY` as a tuple of valid stages.
8. **AST coverage** — each of the 7 connectors imports and calls `fetch_with_fallback`.
9. **Playwright unavailable** — patch `_stage_playwright` to raise `ImportError`; `outcome="playwright_unavailable"`; continues to next stage (or `FallbackExhausted` if last).

### F2.5 — telemetry (`tests/test_narrative_telemetry/`)

10. **Roundtrip** — `log_event(source, outcome, fallback_used)` → SELECT returns the row.
11. **Validation** — `log_event(outcome="not_a_valid_outcome")` → `ValueError`.
12. **DB failure** — patch `_cache_db.get_conn` to return a mock whose `execute` raises → `log_event` warns + returns; no exception escapes.
13. **Kill-switch** — `monkeypatch.setenv("PUREMACRO_NARRATIVE_TELEMETRY", "0")` → `log_event` no-ops, no row inserted.
14. **Aggregation math** — seed events; `connector_health()` row has expected `n_total`, `success_rate`, `fallback_rate`.
15. **Window filter** — events spanning 30 days; `connector_health(window=pd.Timedelta(days=7))` excludes the older 23.
16. **Empty** — `connector_health()` on empty DB returns empty DataFrame with the expected 7 columns.
17. **Sources filter** — `connector_health(sources=["eu_eurlex"])` filters correctly.
18. **End-to-end** — mock `_stage_live` to fail, `_stage_wayback` to succeed; call `fetch_with_fallback(source="eu_eurlex")`; query `connector_events` directly → 2 rows with correct outcomes.

### Cross-cutting

19. **Pyodide compat** — extends `tests/test_pyodide_compat.py`: importing `_fallback`, `_telemetry`, `_cache_db` (with the new table) does not leak forbidden modules. `_playwright_helper` allowed to lazy-import.
20. **Backwards-compat sweep** — Slice 1 signal-contract tests, Slice A cache/credentials/vintages/schema_checks tests, and the narrative suite all stay green.

## Staging

Three sub-slices inside the 0.67.0 release window, in dependency order.

### Sub-slice 1 — F2.5 telemetry foundation (~4 commits)

1. Extend `_cache_db.py`: add `connector_events` DDL + index + `("connector_events", 1)` to `_SCHEMA_SEED`. Test for the new table in `tests/test_cache_db/test_schema_bootstrap.py` extension.
2. New `_telemetry.py`: `VALID_OUTCOMES`, `VALID_FALLBACK_USED`, `telemetry_enabled`, `log_event`. Tests for roundtrip, validation, failure modes, kill-switch.
3. `connector_health(...)` aggregation + tests for shape / math / window / sources / empty.
4. `docs/CONNECTOR_HEALTH.md` reference page.

### Sub-slice 2 — F2.4 fallback layer (~4 commits)

5. New `_fallback.py`: `SUPPORTED_STAGES`, `FallbackExhaustedError`, `FallbackStageUnavailable`, `_classify`, `_dispatch_stage`, `fetch_with_fallback`. Wires `log_event`. Tests for happy path, exhausted, classify, validation, stage dispatch.
6. Rollout commit 1: `eu_eurlex`, `eu_parliament`, `us_cbo` — declare `FALLBACK_POLICY`, swap inline Wayback for `fetch_with_fallback`.
7. Rollout commit 2: `rba`, `bok`, `riksbank`, `sarb` — same for Playwright.
8. AST coverage test: 7 connectors all declare `FALLBACK_POLICY` and call `fetch_with_fallback`.

### Sub-slice 3 — schema-mismatch telemetry + release (~3 commits)

9. Wire `log_event(outcome="parser_schema_mismatch", ...)` into each of the 8 Slice-A `iter_<source>` wrappers (one-line addition).
10. R5_02 notebook + paired builder: `connector_health()` demo on synthetic seeded events + a small live-fetch loop with telemetry on.
11. Version bump to 0.67.0 + CHANGELOG + ARCHITECTURE "F2 closure" subsection. Final sanity sweep.

**Total: ~11 commits**, about half the size of Slice A.

**Critical-path dependency**: F2.5 (Sub-slice 1) lands first so `_fallback.py` (Sub-slice 2) can import `log_event` from a stable module. Sub-slice 3 depends on Slice A's schema-check wrappers being in place (they are — shipped in 0.66.0).

## Done-definition for Slice B (0.67.0)

- `connector_events` table created on bootstrap; `schema_version` row seeded.
- `log_event` + `connector_health` ship with full test coverage; kill-switch works.
- `fetch_with_fallback` ships with 7 supported stages worth of behavior; `FallbackExhaustedError` and `FallbackStageUnavailable` raise correctly.
- 7 fallback connectors migrated to `fetch_with_fallback`; AST coverage assertion enforces both `FALLBACK_POLICY` and the call.
- 8 Slice-A schema-checked connectors emit `parser_schema_mismatch` events.
- R5_02 demo notebook executes cleanly with a sensible `connector_health()` output.
- `docs/CONNECTOR_HEALTH.md` shipped; ARCHITECTURE.md gains "F2 closure (0.67.0+)" subsection.
- `pyproject.toml` at `version = "0.67.0"`; `puremacro/__init__.py` `__version__ = "0.67.0"`; CHANGELOG 0.67.0 entry; final sanity sweep green.
- Pyodide-compat passes; full narrative test suite shows no new regressions vs. the post-Slice-A baseline.

## Open follow-ups (queued for later slices)

- F2.4 rollout to the remaining ~53 connectors (gradual; adopt as new sources need fallback).
- F2.5 telemetry retention (`connector_events_clear(older_than=)`) once the table grows large in practice.
- Per-event `url_hash` + `latency_ms` for deeper diagnostics if the basic schema proves insufficient.
- OpenTelemetry / Prometheus exporter for users running puremacro inside a production data pipeline.
- New fallback stages: `tor`, `paid_proxy`, `mirrored_s3` — each is a one-line `SUPPORTED_STAGES` addition + a `_dispatch_stage` branch.
- F2.5 covering connectors that don't use `fetch_with_fallback` (via a per-iterator `track_fetch` context manager or contextvar-based propagation of `source`).

---

## Spec self-review (inline)

- **Placeholder scan**: no TBD/TODO. Open decisions all resolved before writing (narrow scope, module-constant policy API, `(ts, source, outcome, fallback_used)` schema, single `connector_health` aggregation).
- **Internal consistency**: `SUPPORTED_STAGES = {"live", "wayback", "playwright"}` listed identically in `_fallback.py`, the policy table, the testing section, and the validation test. `VALID_OUTCOMES` (9 entries) listed once and referenced from `_classify` mapping. `VALID_FALLBACK_USED = {"live", "wayback", "playwright", "none"}` consistent across `log_event` validation and the 4-value enum in the schema description.
- **Scope check**: 2 sub-components in 3 sub-slices ≈ 11 commits. Comparable to Slice A's 4 components in 4 sub-slices ≈ 23 commits, but smaller because the foundation is already in place. Single implementation plan can carry it.
- **Ambiguity check**: `fetch_with_fallback` always returns `str` (not `bytes`) — consistent with the existing `safe_get_text`/`safe_get_text_cached` family; the 7 migrated connectors all consume HTML strings. If a future binary use case arrives, a parallel `fetch_with_fallback_bytes` is the natural extension. The `source=` kwarg is required (not optional) so a misuse like `fetch_with_fallback(url)` fails at the call site instead of silently mis-attributing telemetry — flagged explicitly so the implementer doesn't add a default.
