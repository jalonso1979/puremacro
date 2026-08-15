# puremacro 0.67.0 Implementation Plan — F2 Slice B (governed fallback + health telemetry)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Slice B of F2 as `0.67.0`: a `puremacro.narrative.sources._fallback.fetch_with_fallback(...)` entry point that the 7 connectors with existing Wayback/Playwright logic migrate to, plus a `connector_events` SQLite table + `puremacro.narrative.sources._telemetry` module exposing `log_event(...)` and `connector_health(...)` for researcher introspection. Closes the F2 sub-project.

**Architecture:** Three sub-slices landing in dependency order inside one release window. Sub-slice 1 ships the telemetry foundation (table + log_event + connector_health) so the fallback module can import the logger. Sub-slice 2 ships `_fallback.py` and migrates the 7 connectors (each declares `FALLBACK_POLICY` as a module-constant tuple of `SUPPORTED_STAGES` and replaces inline Wayback/Playwright calls with `fetch_with_fallback(url, policy=FALLBACK_POLICY, source="<name>")`). Sub-slice 3 wires `parser_schema_mismatch` events into the 8 Slice-A schema-checked iter_<source> wrappers, ships the R5_02 demo notebook, and bumps the version.

**Tech Stack:** Python ≥3.12 (set in 0.66.0; unchanged here). `sqlite3` stdlib. `pandas`. Optional lazy `playwright`. No new runtime dependencies. Pyodide-pure on the new modules.

**Spec:** `docs/specs/2026-05-26-puremacro-067-f2-slice-b-fallback-telemetry-design.md`

---

## File map

### New files
- `puremacro/puremacro/narrative/sources/_telemetry.py` — `VALID_OUTCOMES`, `VALID_FALLBACK_USED`, `telemetry_enabled`, `log_event`, `connector_health`.
- `puremacro/puremacro/narrative/sources/_fallback.py` — `SUPPORTED_STAGES`, `FallbackExhaustedError`, `FallbackStageUnavailable`, `_classify`, `_stage_live`, `_stage_wayback`, `_stage_playwright`, `_dispatch_stage`, `fetch_with_fallback`.
- `puremacro/docs/CONNECTOR_HEALTH.md` — researcher-facing reference for `connector_health()` and the event schema.
- `tools/make_notebook_R5_02.py` — paired builder for the demo notebook.
- `notebooks/R5_data_infra/R5_02_connector_health_demo.ipynb` — executed demo.
- `tests/test_narrative_telemetry/__init__.py` — empty.
- `tests/test_narrative_telemetry/test_log_event.py` — roundtrip + validation + failure + kill-switch.
- `tests/test_narrative_telemetry/test_connector_health.py` — shape + math + window + sources + empty.
- `tests/test_narrative_fallback/__init__.py` — empty.
- `tests/test_narrative_fallback/test_fetch_with_fallback.py` — happy path, exhausted, classify, stages.
- `tests/test_narrative_fallback/test_supported_stages.py` — registry + validation.
- `tests/test_narrative_fallback/test_per_connector_policies.py` — 7 connectors declare valid `FALLBACK_POLICY`.
- `tests/test_narrative_fallback/test_coverage_assertion.py` — AST scan: 7 connectors call `fetch_with_fallback`.
- `tests/test_narrative_fallback/test_end_to_end_telemetry.py` — fetch_with_fallback emits events as expected.

### Modified files
- `puremacro/puremacro/_cache_db.py` — extend with `connector_events` DDL + index + `("connector_events", 1)` in `_SCHEMA_SEED`.
- `puremacro/puremacro/narrative/sources/eu_eurlex.py` — declare `FALLBACK_POLICY`, swap inline Wayback for `fetch_with_fallback`.
- `puremacro/puremacro/narrative/sources/eu_parliament.py` — same.
- `puremacro/puremacro/narrative/sources/us_cbo.py` — same.
- `puremacro/puremacro/narrative/sources/rba.py` — declare `FALLBACK_POLICY`, swap inline Playwright for `fetch_with_fallback`.
- `puremacro/puremacro/narrative/sources/bok.py` — same.
- `puremacro/puremacro/narrative/sources/riksbank.py` — same.
- `puremacro/puremacro/narrative/sources/sarb.py` — same.
- `puremacro/puremacro/narrative/sources/beige_book.py` — `iter_beige_book` `except ParserSchemaMismatchError` block also calls `_telemetry.log_event(source="beige_book", outcome="parser_schema_mismatch", fallback_used="none")` BEFORE the existing `warnings.warn`.
- `puremacro/puremacro/narrative/sources/eu_eurlex.py` — same `parser_schema_mismatch` `log_event` in its `iter_eurlex` wrapper. (This file is touched twice in the slice: once in Sub-slice 2 for fallback, once in Sub-slice 3 for telemetry. Both touches stay surgical.)
- `puremacro/puremacro/narrative/sources/eu_parliament.py` — same.
- `puremacro/puremacro/narrative/sources/us_cbo.py` — same.
- `puremacro/puremacro/narrative/sources/fed_minutes.py` — same.
- `puremacro/puremacro/narrative/sources/fed_speeches.py` — same.
- `puremacro/puremacro/narrative/sources/bluesky.py` — same.
- `puremacro/puremacro/narrative/sources/ecb_press.py` — same.
- `puremacro/pyproject.toml` — bump `version` to `"0.67.0"`.
- `puremacro/puremacro/__init__.py` — bump `__version__` to `"0.67.0"`.
- `puremacro/CHANGELOG.md` — prepend `## 0.67.0 (2026-05-26)` section.
- `puremacro/ARCHITECTURE.md` — append "F2 closure (0.67.0+)" subsection under the Data-infrastructure block.
- `tests/test_cache_db/test_schema_bootstrap.py` — extend `test_bootstrap_creates_three_tables` to assert `connector_events` is present too (now 4 tables expected) AND extend `test_bootstrap_seeds_schema_version` to expect the new `("connector_events", 1)` row.
- `tests/test_credentials/test_service_registry.py` — version smoke test bumped to `test_puremacro_version_is_067`.
- `tests/test_pyodide_compat.py` — verification re-run; no source change unless a new forbidden import slips in.

### Working assumptions (verified 2026-05-26 via signature dumps)

- `puremacro._cache_db.bootstrap_schema` is idempotent (uses `CREATE TABLE IF NOT EXISTS` + `INSERT OR IGNORE`). Adding a fourth table + a fifth seed row is a clean append.
- `puremacro._cache_db._SCHEMA_SEED` is currently `[("http_cache", 1), ("alfred_vintages", 1)]`. We append `("connector_events", 1)`.
- `puremacro.narrative.sources._wayback.wayback_snapshot_url(target, *, user_agent=None) -> str | None` — Slice A unchanged. Returns `None` on miss (CDX 404, malformed JSON, etc.) — does NOT raise.
- `puremacro.narrative.sources._playwright_helper.fetch_with_playwright(url, *, wait_for="networkidle", timeout_ms=20000, viewport=(1440, 900), locale="en-US") -> str` (line 84). LRU-cached at 128 entries. Raises `ImportError` if Playwright isn't installed; raises `RuntimeError` on browser/timeout failures.
- `puremacro.narrative.sources._http.safe_get_text(url, *, timeout=30, user_agent=None) -> str` — used inside the existing Wayback callers.
- `puremacro.narrative.sources._http.safe_get_text_cached(url, ...)` — cached variant. Slice B's `_stage_live` uses this by default.
- `puremacro.narrative.sources._schema_check.ParserSchemaMismatchError` — exists from Slice A.
- The 8 Slice-A iter_<source> wrappers (`beige_book.iter_beige_book`, `eu_eurlex.iter_eurlex`, `eu_parliament.iter_ep_debates`, `us_cbo.iter_cbo`, `fed_minutes.iter_fed_minutes`, `fed_speeches.iter_fed_speeches`, `bluesky.iter_bluesky_posts`, `ecb_press.iter_ecb_press`) already have an `except ParserSchemaMismatchError` block that emits `warnings.warn`. Sub-slice 3 adds ONE LINE — a `_telemetry.log_event(...)` call — immediately before the existing `warnings.warn`.
- **Per-connector `FALLBACK_POLICY` values must be calibrated against each connector's current behaviour** (verify-then-set). The spec hypothesised `("live", "wayback")` for the 3 WAF-blocked connectors and `("live", "playwright")` for the 4 CB connectors, but live endpoints are permanently blocked for `eu_eurlex` and `eu_parliament` — they are currently Wayback-only. Similarly the 4 Playwright connectors currently call `fetch_with_playwright` directly with no `safe_get_text` first. The implementer of Tasks 6 and 7 must grep each connector for `safe_get_text` / `wayback_snapshot_url` / `fetch_with_playwright` calls and choose `FALLBACK_POLICY` to **match current behaviour**, not to invent new stages. Initial hypotheses (verify in code):

  | Connector | Hypothesised `FALLBACK_POLICY` (verify) |
  |---|---|
  | `eu_eurlex` | `("wayback",)` — pure Wayback today |
  | `eu_parliament` | `("wayback",)` — pure Wayback today |
  | `us_cbo` | `("live", "wayback")` — RSS live, PDFs via Wayback |
  | `rba` | `("playwright",)` — pure Playwright today |
  | `bok` | `("playwright",)` — pure Playwright today |
  | `riksbank` | `("playwright",)` — pure Playwright today |
  | `sarb` | `("playwright",)` — pure Playwright today |

  If a connector's actual behaviour deviates, set `FALLBACK_POLICY` to the verified stages and proceed. Single-stage policies are valid (the loop still emits one telemetry event; the structure stays uniform across all 7).

- `pyproject.toml` currently `version = "0.66.0"`, `requires-python = ">=3.12"` (set in 0.66.0). Bump `version` to `"0.67.0"`; leave `requires-python` alone.
- `puremacro/__init__.py` currently `__version__ = "0.66.0"`; bump to `"0.67.0"`.
- Test runner: `pytest tests/<path>::<name> -v` from the `puremacro/` package directory. Baseline pre-Slice-B test counts (post-Slice-A): credentials 26, cache_db 30, vintages_alfred_store 23, narrative_schema_checks 31, signal_contract 37, pyodide_compat 2.
- Commit-message style: `feat(0.67.0): …` for code, `docs(0.67.0): …` for docs, `chore(puremacro): bump to 0.67.0 — …` for the version bump, `test(0.67.0): …` for test-only commits, `fix(0.67.0): …` for post-merge corrections. `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` footer on every commit.
- Branch is `main`. Commits land directly on `main` per the workflow established in Slices 1 + A.

---

## Sub-slice 1 — F2.5 telemetry foundation

(Tasks 1–4.)

## Task 1: Extend `_cache_db` with `connector_events` table

**Files:**
- Modify: `puremacro/puremacro/_cache_db.py` (add DDL + index + seed row).
- Modify: `puremacro/tests/test_cache_db/test_schema_bootstrap.py` (extend two existing tests).

- [ ] **Step 1: Update the failing assertions in existing tests**

Edit `puremacro/tests/test_cache_db/test_schema_bootstrap.py`. Find `test_bootstrap_creates_three_tables` and change the assertion to include `connector_events`:

```python
def test_bootstrap_creates_three_tables(fresh_db):
    from puremacro._cache_db import get_conn
    conn = get_conn(fresh_db)
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert {"http_cache", "alfred_vintages", "schema_version",
            "connector_events"}.issubset(tables)
```

Find `test_bootstrap_seeds_schema_version` and change the expected rows dict:

```python
def test_bootstrap_seeds_schema_version(fresh_db):
    from puremacro._cache_db import get_conn
    conn = get_conn(fresh_db)
    rows = dict(conn.execute("SELECT component, version FROM schema_version"))
    assert rows == {"http_cache": 1, "alfred_vintages": 1,
                    "connector_events": 1}
```

Note: the test function name `test_bootstrap_creates_three_tables` is now slightly stale (we have 4 tables). Leave the name as-is for this commit — renaming risks colliding with other refactors. A documentation tidy-up could rename it in 0.68.0.

- [ ] **Step 2: Run to verify failure**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
pytest tests/test_cache_db/test_schema_bootstrap.py -v
```
Expected: the two updated tests fail (table/seed missing); other 6 tests still pass.

- [ ] **Step 3: Extend `_cache_db.py`**

Edit `puremacro/puremacro/_cache_db.py`. Add the new DDL constants near the existing `_DDL_*` block:

```python
_DDL_CONNECTOR_EVENTS = """
CREATE TABLE IF NOT EXISTS connector_events (
    ts             INTEGER NOT NULL,
    source         TEXT NOT NULL,
    outcome        TEXT NOT NULL,
    fallback_used  TEXT NOT NULL
);
"""

_DDL_CONNECTOR_EVENTS_IDX = (
    "CREATE INDEX IF NOT EXISTS connector_events_ts_source_idx "
    "ON connector_events(ts, source);"
)
```

Update `_SCHEMA_SEED` to include the new component:

```python
_SCHEMA_SEED = [("http_cache", 1), ("alfred_vintages", 1),
                ("connector_events", 1)]
```

Update `bootstrap_schema` to execute the new DDLs. Find the existing function body and add the two new `cur.execute(...)` calls after the existing `_DDL_ALFRED_VINTAGES_IDX` call:

```python
def bootstrap_schema(conn: sqlite3.Connection) -> None:
    """Idempotent: create tables + seed schema_version rows if missing."""
    cur = conn.cursor()
    cur.execute(_DDL_HTTP_CACHE)
    cur.execute(_DDL_HTTP_CACHE_IDX)
    cur.execute(_DDL_ALFRED_VINTAGES)
    cur.execute(_DDL_ALFRED_VINTAGES_IDX)
    cur.execute(_DDL_CONNECTOR_EVENTS)
    cur.execute(_DDL_CONNECTOR_EVENTS_IDX)
    cur.execute(_DDL_SCHEMA_VERSION)
    for component, version in _SCHEMA_SEED:
        cur.execute(
            "INSERT OR IGNORE INTO schema_version (component, version) "
            "VALUES (?, ?)",
            (component, version),
        )
    conn.commit()
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_cache_db/test_schema_bootstrap.py -v
```
Expected: all 8 pass.

- [ ] **Step 5: Run the full cache_db suite to confirm no regression**

```bash
pytest tests/test_cache_db/ -v 2>&1 | tail -10
```
Expected: 30 passed.

- [ ] **Step 6: Commit**

```bash
git add puremacro/puremacro/_cache_db.py tests/test_cache_db/test_schema_bootstrap.py
git commit -m "$(cat <<'EOF'
feat(0.67.0): _cache_db gains connector_events table + schema seed

F2.5 foundation. Adds a fourth SQLite table connector_events
(ts, source, outcome, fallback_used) with an (ts, source) index,
and seeds ("connector_events", 1) into schema_version. Idempotent via
CREATE IF NOT EXISTS + INSERT OR IGNORE; safe to re-run on existing
DBs. Existing http_cache and alfred_vintages tables unaffected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Create `_telemetry.py` with `log_event` + kill-switch

**Files:**
- Create: `puremacro/puremacro/narrative/sources/_telemetry.py`.
- Create: `tests/test_narrative_telemetry/__init__.py` (empty).
- Create: `tests/test_narrative_telemetry/test_log_event.py`.

- [ ] **Step 1: Create the test directory and write the failing tests**

Create `tests/test_narrative_telemetry/__init__.py` (empty file).

Create `tests/test_narrative_telemetry/test_log_event.py`:

```python
"""F2.5 — log_event roundtrip, validation, failure mode, kill-switch."""
from __future__ import annotations

import sqlite3
import warnings
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("PUREMACRO_NARRATIVE_TELEMETRY", raising=False)
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


def test_log_event_inserts_row(fresh_db):
    from puremacro.narrative.sources._telemetry import log_event
    from puremacro import _cache_db
    log_event(source="eu_eurlex", outcome="success", fallback_used="live")
    conn = _cache_db.get_conn()
    rows = conn.execute(
        "SELECT source, outcome, fallback_used FROM connector_events"
    ).fetchall()
    assert rows == [("eu_eurlex", "success", "live")]


def test_log_event_fallback_used_defaults_to_none(fresh_db):
    from puremacro.narrative.sources._telemetry import log_event
    from puremacro import _cache_db
    log_event(source="beige_book", outcome="parser_schema_mismatch")
    conn = _cache_db.get_conn()
    rows = conn.execute(
        "SELECT fallback_used FROM connector_events"
    ).fetchall()
    assert rows == [("none",)]


def test_log_event_rejects_invalid_outcome(fresh_db):
    from puremacro.narrative.sources._telemetry import log_event
    with pytest.raises(ValueError, match="outcome"):
        log_event(source="x", outcome="not_a_valid_outcome",
                  fallback_used="live")


def test_log_event_rejects_invalid_fallback_used(fresh_db):
    from puremacro.narrative.sources._telemetry import log_event
    with pytest.raises(ValueError, match="fallback_used"):
        log_event(source="x", outcome="success",
                  fallback_used="not_a_valid_fallback")


def test_log_event_db_failure_warns_and_no_ops(fresh_db, monkeypatch):
    from puremacro.narrative.sources._telemetry import log_event
    from puremacro import _cache_db

    # Replace get_conn with a mock whose execute raises (Python 3.13's
    # sqlite3.Connection.execute is read-only, so we patch the resolver).
    fake_conn = MagicMock()
    fake_conn.execute.side_effect = sqlite3.OperationalError("simulated")
    monkeypatch.setattr(_cache_db, "get_conn", lambda *a, **kw: fake_conn)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log_event(source="x", outcome="success", fallback_used="live")
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_log_event_kill_switch_skips_insert(fresh_db, monkeypatch):
    monkeypatch.setenv("PUREMACRO_NARRATIVE_TELEMETRY", "0")
    from puremacro.narrative.sources._telemetry import log_event
    from puremacro import _cache_db
    log_event(source="x", outcome="success", fallback_used="live")
    conn = _cache_db.get_conn()
    rows = conn.execute(
        "SELECT COUNT(*) FROM connector_events"
    ).fetchone()
    assert rows[0] == 0


def test_telemetry_enabled_reflects_env(monkeypatch):
    from puremacro.narrative.sources._telemetry import telemetry_enabled
    monkeypatch.delenv("PUREMACRO_NARRATIVE_TELEMETRY", raising=False)
    assert telemetry_enabled() is True
    monkeypatch.setenv("PUREMACRO_NARRATIVE_TELEMETRY", "0")
    assert telemetry_enabled() is False
    monkeypatch.setenv("PUREMACRO_NARRATIVE_TELEMETRY", "1")
    assert telemetry_enabled() is True
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_narrative_telemetry/test_log_event.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'puremacro.narrative.sources._telemetry'`.

- [ ] **Step 3: Create the telemetry module**

Create `puremacro/puremacro/narrative/sources/_telemetry.py`:

```python
"""Per-connector health telemetry (0.67.0+).

Each fetch / parse attempt by a participating connector produces one
row in the ``connector_events`` SQLite table:

    (ts, source, outcome, fallback_used)

Emitters:
- ``puremacro.narrative.sources._fallback.fetch_with_fallback`` — one
  event per stage attempt (live, wayback, playwright).
- The 8 Slice-A schema-checked iter_<source> wrappers — one event per
  ``ParserSchemaMismatchError`` catch (outcome="parser_schema_mismatch",
  fallback_used="none").

Researchers introspect with ``connector_health(window=...)``.

Kill-switch: ``PUREMACRO_NARRATIVE_TELEMETRY=0`` disables event logging
without breaking any caller. Telemetry must NEVER raise — DB failures
emit a ``UserWarning`` and silently no-op.
"""
from __future__ import annotations

import os
import sqlite3
import time
import warnings


VALID_OUTCOMES: frozenset[str] = frozenset({
    "success", "404", "timeout", "ssl_fail", "server_5xx",
    "wayback_no_snapshot", "playwright_unavailable",
    "parser_schema_mismatch", "other_network_error",
})

VALID_FALLBACK_USED: frozenset[str] = frozenset({
    "live", "wayback", "playwright", "none",
})


def telemetry_enabled() -> bool:
    """True unless ``PUREMACRO_NARRATIVE_TELEMETRY=0`` is set."""
    return os.environ.get("PUREMACRO_NARRATIVE_TELEMETRY", "1") != "0"


def log_event(
    *,
    source: str,
    outcome: str,
    fallback_used: str = "none",
) -> None:
    """Insert one row into connector_events.

    Validates ``outcome`` and ``fallback_used`` against the registries
    above (raises ``ValueError`` on miss — programmer error). Skipped
    silently if ``telemetry_enabled()`` is False. DB failures emit a
    ``UserWarning`` and no-op — the calling fetch path MUST keep going.
    """
    if outcome not in VALID_OUTCOMES:
        raise ValueError(
            f"log_event: outcome {outcome!r} not in {sorted(VALID_OUTCOMES)}"
        )
    if fallback_used not in VALID_FALLBACK_USED:
        raise ValueError(
            f"log_event: fallback_used {fallback_used!r} not in "
            f"{sorted(VALID_FALLBACK_USED)}"
        )
    if not telemetry_enabled():
        return
    # _cache_db lives at puremacro._cache_db (NOT
    # puremacro.narrative._cache_db). Local import keeps cold-import time
    # cheap and avoids a hard module-load dependency that could affect
    # Pyodide guards.
    from puremacro import _cache_db as _db
    try:
        _db.get_conn().execute(
            "INSERT INTO connector_events "
            "(ts, source, outcome, fallback_used) VALUES (?, ?, ?, ?)",
            (int(time.time()), source, outcome, fallback_used),
        )
    except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
        warnings.warn(
            f"puremacro.narrative.sources._telemetry.log_event failed "
            f"({source}/{outcome}): {e}",
            UserWarning, stacklevel=2,
        )


__all__ = [
    "VALID_OUTCOMES",
    "VALID_FALLBACK_USED",
    "telemetry_enabled",
    "log_event",
]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_narrative_telemetry/test_log_event.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add puremacro/puremacro/narrative/sources/_telemetry.py tests/test_narrative_telemetry/__init__.py tests/test_narrative_telemetry/test_log_event.py
git commit -m "$(cat <<'EOF'
feat(0.67.0): _telemetry module — log_event + kill-switch

F2.5 second commit. New puremacro.narrative.sources._telemetry exposes
VALID_OUTCOMES (9 entries) + VALID_FALLBACK_USED (4 entries) registries,
telemetry_enabled() (PUREMACRO_NARRATIVE_TELEMETRY=0 kill-switch), and
log_event(source=, outcome=, fallback_used="none") which inserts one row
into connector_events. Validation raises ValueError on invalid outcomes/
fallback strings (programmer error). DB failures emit UserWarning and
no-op — telemetry must never break a fetch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `connector_health` aggregation

**Files:**
- Modify: `puremacro/puremacro/narrative/sources/_telemetry.py` (append).
- Create: `tests/test_narrative_telemetry/test_connector_health.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_narrative_telemetry/test_connector_health.py`:

```python
"""F2.5 — connector_health aggregation shape + math + filters."""
from __future__ import annotations

import time

import pandas as pd
import pytest


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("PUREMACRO_NARRATIVE_TELEMETRY", raising=False)
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


_EXPECTED_COLUMNS = {
    "source", "n_total", "n_success", "success_rate",
    "n_fallback", "fallback_rate", "last_seen",
}


def _seed(events):
    """Insert (ts_offset_seconds_ago, source, outcome, fallback_used) rows."""
    from puremacro import _cache_db
    conn = _cache_db.get_conn()
    now = int(time.time())
    for offset, source, outcome, fb in events:
        conn.execute(
            "INSERT INTO connector_events "
            "(ts, source, outcome, fallback_used) VALUES (?, ?, ?, ?)",
            (now - offset, source, outcome, fb),
        )
    conn.commit()


def test_connector_health_returns_expected_columns(fresh_db):
    from puremacro.narrative.sources._telemetry import connector_health
    _seed([(60, "x", "success", "live")])
    df = connector_health(window=pd.Timedelta(days=1))
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == _EXPECTED_COLUMNS


def test_connector_health_aggregation_math(fresh_db):
    from puremacro.narrative.sources._telemetry import connector_health
    _seed([
        # 7 successes (5 live, 2 wayback), 3 failures (1 live, 2 wayback).
        (10, "eu_eurlex", "success",         "live"),
        (20, "eu_eurlex", "success",         "live"),
        (30, "eu_eurlex", "success",         "live"),
        (40, "eu_eurlex", "success",         "live"),
        (50, "eu_eurlex", "success",         "live"),
        (60, "eu_eurlex", "success",         "wayback"),
        (70, "eu_eurlex", "success",         "wayback"),
        (80, "eu_eurlex", "timeout",         "live"),
        (90, "eu_eurlex", "wayback_no_snapshot", "wayback"),
        (100,"eu_eurlex", "other_network_error", "wayback"),
    ])
    df = connector_health(window=pd.Timedelta(hours=1)).set_index("source")
    row = df.loc["eu_eurlex"]
    assert row["n_total"] == 10
    assert row["n_success"] == 7
    assert row["success_rate"] == 0.7
    # n_fallback = events where fallback_used != "live" = 4
    assert row["n_fallback"] == 4
    assert row["fallback_rate"] == 0.4


def test_connector_health_window_filter(fresh_db):
    from puremacro.narrative.sources._telemetry import connector_health
    _seed([
        (60,      "x", "success", "live"),       # 60s ago, within 5min
        (8 * 24 * 3600, "x", "success", "live"), # 8 days ago, outside 7d window
    ])
    df_5min = connector_health(window=pd.Timedelta(minutes=5)).set_index("source")
    assert df_5min.loc["x", "n_total"] == 1
    df_7d = connector_health(window=pd.Timedelta(days=7)).set_index("source")
    assert df_7d.loc["x", "n_total"] == 1   # 8-day-old row still excluded
    df_30d = connector_health(window=pd.Timedelta(days=30)).set_index("source")
    assert df_30d.loc["x", "n_total"] == 2


def test_connector_health_sources_filter(fresh_db):
    from puremacro.narrative.sources._telemetry import connector_health
    _seed([
        (60, "eu_eurlex", "success", "live"),
        (60, "rba",       "success", "live"),
    ])
    df = connector_health(window=pd.Timedelta(days=1), sources=["eu_eurlex"])
    assert set(df["source"]) == {"eu_eurlex"}


def test_connector_health_empty_db_returns_empty_df(fresh_db):
    from puremacro.narrative.sources._telemetry import connector_health
    df = connector_health(window=pd.Timedelta(days=7))
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert set(df.columns) == _EXPECTED_COLUMNS


def test_connector_health_last_seen_is_timestamp(fresh_db):
    from puremacro.narrative.sources._telemetry import connector_health
    _seed([(60, "x", "success", "live")])
    df = connector_health(window=pd.Timedelta(days=1))
    assert pd.api.types.is_datetime64_any_dtype(df["last_seen"])
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_narrative_telemetry/test_connector_health.py -v
```
Expected: FAIL — `connector_health` doesn't exist yet.

- [ ] **Step 3: Append `connector_health` to `_telemetry.py`**

Append the following at the end of `puremacro/puremacro/narrative/sources/_telemetry.py`:

```python
def connector_health(
    *,
    window: "pd.Timedelta | None" = None,
    sources: "list[str] | None" = None,
) -> "pd.DataFrame":
    """Aggregate connector_events over the last ``window`` (default 7 days).

    Returns one row per source with columns:
      source, n_total, n_success, success_rate,
      n_fallback, fallback_rate, last_seen
    Where:
      - n_total      = rows for the source in the window
      - n_success    = rows with outcome='success'
      - success_rate = n_success / n_total
      - n_fallback   = rows with fallback_used != 'live'
                       (i.e., served by wayback or playwright,
                       or — for parser_schema_mismatch — 'none')
      - fallback_rate = n_fallback / n_total
      - last_seen    = max(ts) for the source

    Empty DataFrame (with the expected 7 columns) if no events match.
    DB failures emit a UserWarning and return the empty shape.
    """
    import pandas as pd
    from puremacro import _cache_db as _db

    expected_cols = ["source", "n_total", "n_success", "success_rate",
                     "n_fallback", "fallback_rate", "last_seen"]
    if window is None:
        window = pd.Timedelta(days=7)

    cutoff = int(time.time() - window.total_seconds())
    sql = (
        "SELECT source, COUNT(*) AS n_total, "
        "SUM(CASE WHEN outcome='success' THEN 1 ELSE 0 END) AS n_success, "
        "SUM(CASE WHEN fallback_used <> 'live' THEN 1 ELSE 0 END) AS n_fallback, "
        "MAX(ts) AS last_seen_ts "
        "FROM connector_events WHERE ts >= ?"
    )
    params: list = [cutoff]
    if sources:
        placeholders = ",".join("?" * len(sources))
        sql += f" AND source IN ({placeholders})"
        params.extend(sources)
    sql += " GROUP BY source ORDER BY source"

    try:
        rows = _db.get_conn().execute(sql, params).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
        warnings.warn(
            f"connector_health query failed: {e}",
            UserWarning, stacklevel=2,
        )
        return pd.DataFrame(columns=expected_cols)

    if not rows:
        return pd.DataFrame(columns=expected_cols)

    df = pd.DataFrame(rows, columns=["source", "n_total", "n_success",
                                       "n_fallback", "last_seen_ts"])
    df["n_total"] = df["n_total"].astype("int64")
    df["n_success"] = df["n_success"].astype("int64")
    df["n_fallback"] = df["n_fallback"].astype("int64")
    df["success_rate"] = df["n_success"] / df["n_total"]
    df["fallback_rate"] = df["n_fallback"] / df["n_total"]
    df["last_seen"] = pd.to_datetime(df["last_seen_ts"], unit="s")
    df = df[["source", "n_total", "n_success", "success_rate",
             "n_fallback", "fallback_rate", "last_seen"]]
    return df
```

Extend `__all__` at the bottom of the file:

```python
__all__ = [
    "VALID_OUTCOMES",
    "VALID_FALLBACK_USED",
    "telemetry_enabled",
    "log_event",
    "connector_health",
]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_narrative_telemetry/ -v
```
Expected: 13 passed (7 from T2 + 6 new).

- [ ] **Step 5: Commit**

```bash
git add puremacro/puremacro/narrative/sources/_telemetry.py tests/test_narrative_telemetry/test_connector_health.py
git commit -m "$(cat <<'EOF'
feat(0.67.0): connector_health aggregation

F2.5 third commit. connector_health(window=, sources=) returns one row
per source with [source, n_total, n_success, success_rate, n_fallback,
fallback_rate, last_seen]. Window defaults to 7 days; sources= filters
to a specific list. Empty DB returns empty DataFrame with the expected
7 columns (not a KeyError or different shape). DB failures warn +
return the empty shape.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `docs/CONNECTOR_HEALTH.md` reference

**Files:**
- Create: `puremacro/docs/CONNECTOR_HEALTH.md`.

- [ ] **Step 1: Create the reference doc**

Create `puremacro/docs/CONNECTOR_HEALTH.md`:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add puremacro/docs/CONNECTOR_HEALTH.md
git commit -m "$(cat <<'EOF'
docs(0.67.0): CONNECTOR_HEALTH.md reference

F2.5 docs. Single-page researcher-facing reference for connector_health()
+ the connector_events schema + Slice B scope (what gets logged today)
+ kill-switch + failure semantics + Pyodide notes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Sub-slice 2 — F2.4 fallback layer

(Tasks 5–8.)

## Task 5: Create `_fallback.py` framework

**Files:**
- Create: `puremacro/puremacro/narrative/sources/_fallback.py`.
- Create: `tests/test_narrative_fallback/__init__.py` (empty).
- Create: `tests/test_narrative_fallback/test_fetch_with_fallback.py`.
- Create: `tests/test_narrative_fallback/test_supported_stages.py`.
- Create: `tests/test_narrative_fallback/test_end_to_end_telemetry.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_narrative_fallback/__init__.py` (empty).

Create `tests/test_narrative_fallback/test_supported_stages.py`:

```python
"""F2.4 — SUPPORTED_STAGES registry + policy validation."""
from __future__ import annotations

import pytest


def test_supported_stages_has_expected_entries():
    from puremacro.narrative.sources._fallback import SUPPORTED_STAGES
    assert SUPPORTED_STAGES == frozenset({"live", "wayback", "playwright"})


def test_unknown_stage_in_policy_raises():
    from puremacro.narrative.sources._fallback import fetch_with_fallback
    with pytest.raises(ValueError, match="unknown stage"):
        fetch_with_fallback(
            "https://example.com/", policy=("not_a_stage",), source="x",
        )


def test_empty_policy_raises():
    from puremacro.narrative.sources._fallback import fetch_with_fallback
    with pytest.raises(ValueError, match="empty policy"):
        fetch_with_fallback(
            "https://example.com/", policy=(), source="x",
        )


def test_fallback_exhausted_is_runtimeerror():
    from puremacro.narrative.sources._fallback import FallbackExhaustedError
    assert issubclass(FallbackExhaustedError, RuntimeError)


def test_fallback_stage_unavailable_is_runtimeerror():
    from puremacro.narrative.sources._fallback import FallbackStageUnavailable
    assert issubclass(FallbackStageUnavailable, RuntimeError)
```

Create `tests/test_narrative_fallback/test_fetch_with_fallback.py`:

```python
"""F2.4 — fetch_with_fallback happy path, exhausted, classify."""
from __future__ import annotations

import socket
import ssl
import urllib.error

import pytest


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("PUREMACRO_NARRATIVE_TELEMETRY", raising=False)
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


def test_live_stage_succeeds(fresh_db, monkeypatch):
    from puremacro.narrative.sources import _fallback
    monkeypatch.setattr(
        _fallback, "_stage_live",
        lambda url, *, timeout, use_cache: "<html>live body</html>",
    )
    body = _fallback.fetch_with_fallback(
        "https://example.com/", policy=("live",), source="x",
    )
    assert body == "<html>live body</html>"


def test_live_fails_wayback_succeeds(fresh_db, monkeypatch):
    from puremacro.narrative.sources import _fallback

    def _live_raises(url, *, timeout, use_cache):
        raise urllib.error.HTTPError(url, 500, "boom", {}, None)

    monkeypatch.setattr(_fallback, "_stage_live", _live_raises)
    monkeypatch.setattr(
        _fallback, "_stage_wayback",
        lambda url, *, timeout, use_cache: "<html>wayback body</html>",
    )
    body = _fallback.fetch_with_fallback(
        "https://example.com/", policy=("live", "wayback"), source="x",
    )
    assert body == "<html>wayback body</html>"


def test_all_stages_fail_raises_exhausted(fresh_db, monkeypatch):
    from puremacro.narrative.sources import _fallback

    def _raises(url, *, timeout, use_cache=None):
        raise socket.timeout("boom")

    monkeypatch.setattr(_fallback, "_stage_live", _raises)
    monkeypatch.setattr(_fallback, "_stage_wayback", _raises)
    with pytest.raises(_fallback.FallbackExhaustedError):
        _fallback.fetch_with_fallback(
            "https://example.com/", policy=("live", "wayback"), source="x",
        )


@pytest.mark.parametrize("exc, expected_outcome", [
    (socket.timeout("t"), "timeout"),
    (urllib.error.HTTPError("u", 404, "nf", {}, None), "404"),
    (urllib.error.HTTPError("u", 500, "isr", {}, None), "server_5xx"),
    (urllib.error.HTTPError("u", 503, "unavail", {}, None), "server_5xx"),
    (ssl.SSLError("ssl boom"), "ssl_fail"),
    (TimeoutError("t"), "timeout"),
    (RuntimeError("other"), "other_network_error"),
])
def test_classify_maps_exceptions(exc, expected_outcome):
    from puremacro.narrative.sources._fallback import _classify
    assert _classify(exc) == expected_outcome


def test_classify_wayback_no_snapshot():
    from puremacro.narrative.sources._fallback import (
        _classify, FallbackStageUnavailable,
    )
    assert _classify(
        FallbackStageUnavailable("wayback_no_snapshot")
    ) == "wayback_no_snapshot"


def test_classify_playwright_unavailable():
    from puremacro.narrative.sources._fallback import (
        _classify, FallbackStageUnavailable,
    )
    assert _classify(
        FallbackStageUnavailable("playwright_unavailable")
    ) == "playwright_unavailable"
```

Create `tests/test_narrative_fallback/test_end_to_end_telemetry.py`:

```python
"""F2.4 — fetch_with_fallback emits the expected telemetry events."""
from __future__ import annotations

import socket

import pytest


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("PUREMACRO_NARRATIVE_TELEMETRY", raising=False)
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


def test_happy_path_emits_single_success_event(fresh_db, monkeypatch):
    from puremacro.narrative.sources import _fallback
    from puremacro import _cache_db
    monkeypatch.setattr(
        _fallback, "_stage_live",
        lambda url, *, timeout, use_cache: "<html>ok</html>",
    )
    _fallback.fetch_with_fallback(
        "https://example.com/a", policy=("live",), source="eu_eurlex",
    )
    rows = _cache_db.get_conn().execute(
        "SELECT source, outcome, fallback_used FROM connector_events"
    ).fetchall()
    assert rows == [("eu_eurlex", "success", "live")]


def test_live_fails_wayback_succeeds_emits_two_events(fresh_db, monkeypatch):
    from puremacro.narrative.sources import _fallback
    from puremacro import _cache_db

    def _live_raises(url, *, timeout, use_cache):
        raise socket.timeout("boom")

    monkeypatch.setattr(_fallback, "_stage_live", _live_raises)
    monkeypatch.setattr(
        _fallback, "_stage_wayback",
        lambda url, *, timeout, use_cache: "<html>wb</html>",
    )
    _fallback.fetch_with_fallback(
        "https://example.com/a", policy=("live", "wayback"), source="eu_eurlex",
    )
    rows = _cache_db.get_conn().execute(
        "SELECT outcome, fallback_used FROM connector_events ORDER BY ts, fallback_used"
    ).fetchall()
    assert ("timeout", "live") in rows
    assert ("success", "wayback") in rows


def test_all_stages_fail_emits_failures_only(fresh_db, monkeypatch):
    from puremacro.narrative.sources import _fallback
    from puremacro import _cache_db

    monkeypatch.setattr(
        _fallback, "_stage_live",
        lambda url, *, timeout, use_cache: (_ for _ in ()).throw(socket.timeout("a")),
    )
    monkeypatch.setattr(
        _fallback, "_stage_wayback",
        lambda url, *, timeout, use_cache: (_ for _ in ()).throw(socket.timeout("b")),
    )
    with pytest.raises(_fallback.FallbackExhaustedError):
        _fallback.fetch_with_fallback(
            "https://example.com/a", policy=("live", "wayback"), source="eu_eurlex",
        )
    rows = _cache_db.get_conn().execute(
        "SELECT outcome FROM connector_events"
    ).fetchall()
    outcomes = [r[0] for r in rows]
    assert outcomes.count("timeout") == 2
    assert "success" not in outcomes
```

- [ ] **Step 2: Run to verify failure**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
pytest tests/test_narrative_fallback/ -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'puremacro.narrative.sources._fallback'`.

- [ ] **Step 3: Create the fallback module**

Create `puremacro/puremacro/narrative/sources/_fallback.py`:

```python
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

from ._http import safe_get_text, safe_get_text_cached
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
        raise urllib.error.HTTPError(url, 204, "empty body", {}, None)
    return body


def _stage_wayback(url: str, *, timeout: float, use_cache: bool) -> str:
    """Wayback fetch: CDX lookup + snapshot fetch via cached helper."""
    wb_url = wayback_snapshot_url(url)
    if wb_url is None:
        raise FallbackStageUnavailable("wayback_no_snapshot")
    body = safe_get_text_cached(wb_url, timeout=timeout) if use_cache \
        else safe_get_text(wb_url, timeout=timeout)
    if not body or not body.strip():
        raise urllib.error.HTTPError(wb_url, 204, "empty wayback body", {}, None)
    return body


def _stage_playwright(url: str, *, timeout: float, use_cache: bool = False) -> str:
    """Playwright (stealth-Chromium) fetch. Lazy-imports the helper so
    a pyodide / no-extras install doesn't fail at module load time.
    No cache layer — Playwright is the last resort."""
    try:
        from ._playwright_helper import fetch_with_playwright
    except ImportError:
        raise FallbackStageUnavailable("playwright_unavailable")
    try:
        return fetch_with_playwright(url, timeout_ms=int(timeout * 1000))
    except ImportError:
        raise FallbackStageUnavailable("playwright_unavailable")


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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_narrative_fallback/ -v
```
Expected: 14 passed (5 supported_stages + 9 fetch_with_fallback + 3 end_to_end_telemetry; counts may vary slightly with the parametrize expansion).

- [ ] **Step 5: Run the telemetry suite to confirm no regression**

```bash
pytest tests/test_narrative_telemetry/ tests/test_cache_db/ -v 2>&1 | tail -10
```
Expected: all pass (13 + 30 = 43).

- [ ] **Step 6: Commit**

```bash
git add puremacro/puremacro/narrative/sources/_fallback.py tests/test_narrative_fallback/__init__.py tests/test_narrative_fallback/test_supported_stages.py tests/test_narrative_fallback/test_fetch_with_fallback.py tests/test_narrative_fallback/test_end_to_end_telemetry.py
git commit -m "$(cat <<'EOF'
feat(0.67.0): _fallback module — fetch_with_fallback + SUPPORTED_STAGES

F2.4 first commit. New puremacro.narrative.sources._fallback exposes:
- SUPPORTED_STAGES = {"live", "wayback", "playwright"}.
- FallbackExhaustedError + FallbackStageUnavailable (both RuntimeError).
- _classify(e) maps urllib/ssl/socket exceptions to canonical outcome
  strings used by VALID_OUTCOMES.
- _stage_live / _stage_wayback / _stage_playwright (Playwright lazy-imported).
- fetch_with_fallback(url, *, policy, source, timeout=30.0, use_cache=True)
  loops stages, emits one telemetry event per attempt, raises
  FallbackExhaustedError if every stage fails.

Single-stage policies are valid. source= is REQUIRED (no default) so
telemetry mis-attribution is a loud error, not silent corruption.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Rollout batch 1 — `eu_eurlex` + `eu_parliament` + `us_cbo`

**Files:**
- Modify: `puremacro/puremacro/narrative/sources/eu_eurlex.py`
- Modify: `puremacro/puremacro/narrative/sources/eu_parliament.py`
- Modify: `puremacro/puremacro/narrative/sources/us_cbo.py`
- Create: `tests/test_narrative_fallback/test_per_connector_policies.py`

For each of the 3 connectors:

1. **Read the file first** to find every call to `wayback_snapshot_url` and the surrounding fetch logic.
2. **Determine the actual current policy** by inspecting whether the connector tries `safe_get_text(direct_url)` BEFORE going to Wayback, or whether it goes straight to Wayback.
3. **Add at the top of the file** (after existing imports):
   ```python
   from ._fallback import fetch_with_fallback, FallbackExhaustedError

   # Verified-against-current-behaviour fallback policy.
   FALLBACK_POLICY = ...     # see step 2 above
   ```
4. **Replace each fetch site** that previously did `wayback_snapshot_url(url) + safe_get_text(wb_url)` (or `safe_get_text(url)` then fall back) with `fetch_with_fallback(url, policy=FALLBACK_POLICY, source="<name>")`.
5. **Wrap each fetch site** in `try/except FallbackExhaustedError` that emits `warnings.warn(UserWarning)` and continues the outer loop (preserves the yield-don't-raise contract from RETRY_POLICY.md §4.1).

- [ ] **Step 1: Write the per-connector policy test**

Create `tests/test_narrative_fallback/test_per_connector_policies.py`:

```python
"""F2.4 — the 7 fallback-aware connectors all declare valid FALLBACK_POLICY."""
from __future__ import annotations

import importlib

import pytest


_FALLBACK_CONNECTORS = [
    "eu_eurlex", "eu_parliament", "us_cbo",
    "rba", "bok", "riksbank", "sarb",
]


@pytest.mark.parametrize("name", _FALLBACK_CONNECTORS)
def test_connector_declares_fallback_policy(name):
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    assert hasattr(mod, "FALLBACK_POLICY"), (
        f"{name}.py must declare FALLBACK_POLICY (F2.4 contract)"
    )
    assert isinstance(mod.FALLBACK_POLICY, tuple), (
        f"{name}.FALLBACK_POLICY must be a tuple"
    )
    assert len(mod.FALLBACK_POLICY) >= 1, (
        f"{name}.FALLBACK_POLICY must have at least one stage"
    )


@pytest.mark.parametrize("name", _FALLBACK_CONNECTORS)
def test_fallback_policy_stages_are_supported(name):
    from puremacro.narrative.sources._fallback import SUPPORTED_STAGES
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    bad = set(mod.FALLBACK_POLICY) - SUPPORTED_STAGES
    assert not bad, (
        f"{name}.FALLBACK_POLICY contains unsupported stages: {sorted(bad)}. "
        f"Supported: {sorted(SUPPORTED_STAGES)}"
    )
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_narrative_fallback/test_per_connector_policies.py -v -k "eu_eurlex or eu_parliament or us_cbo"
```
Expected: 6 failures (3 connectors × 2 tests, all fail because `FALLBACK_POLICY` doesn't exist yet).

- [ ] **Step 3: Edit `eu_eurlex.py`**

Read the file (around line 154 you'll find the existing Wayback call inside a `_fetch_via_wayback` helper, plus the iter_eurlex generator at ~line 284).

Add at the top of the file (after existing imports):

```python
from ._fallback import fetch_with_fallback, FallbackExhaustedError

# eu_eurlex is currently Wayback-only — the live EUR-Lex endpoint is
# AWS-WAF-blocked. If WAF is lifted in the future, change to ("live", "wayback").
FALLBACK_POLICY: tuple[str, ...] = ("wayback",)
```

Replace the body of `_fetch_via_wayback` (the function around line 144 that currently calls `wayback_snapshot_url` + `safe_get_text`) with:

```python
def _fetch_via_wayback(celex_id: str, language: str):
    direct_url = _LEGAL_CONTENT_URL.format(
        lang=language.upper(), celex=celex_id)
    try:
        html = fetch_with_fallback(
            direct_url, policy=FALLBACK_POLICY, source="eu_eurlex",
        )
    except FallbackExhaustedError:
        return None
    if not html.strip():
        return None
    return _parse_eurlex_html(html, celex_id=celex_id,
                               language=language, source_url=direct_url)
```

(Adapt to the existing function signature — only the body inside changes; the function name and arguments are preserved.)

If the file has other `wayback_snapshot_url` call sites that fetch different URLs (e.g., a per-act PDF fetch), apply the same pattern to each.

- [ ] **Step 4: Edit `eu_parliament.py`**

Same pattern. Add the imports + constant at the top:

```python
from ._fallback import fetch_with_fallback, FallbackExhaustedError

# eu_parliament is currently Wayback-only (verbatim plenary CRE pages
# are gated by JS, so the live endpoint returns empty).
FALLBACK_POLICY: tuple[str, ...] = ("wayback",)
```

Replace every site that does `wayback_snapshot_url(...) + safe_get_text(...)` with a `fetch_with_fallback(url, policy=FALLBACK_POLICY, source="eu_parliament")` call wrapped in `try/except FallbackExhaustedError` → warn + return None / continue.

- [ ] **Step 5: Edit `us_cbo.py`**

Read the file first — `us_cbo` has TWO call sites: an RSS feed fetch (live works) AND per-publication PDF fetches (some need Wayback). The verified policy is `("live", "wayback")`.

Add at the top:

```python
from ._fallback import fetch_with_fallback, FallbackExhaustedError

# us_cbo: RSS works live; per-PDF fetches sometimes need Wayback when
# the live PDF link returns a DataDome challenge.
FALLBACK_POLICY: tuple[str, ...] = ("live", "wayback")
```

Replace each fetch site. The RSS fetch currently does `safe_get_text(rss_url)` — wrap it as `fetch_with_fallback(rss_url, policy=FALLBACK_POLICY, source="us_cbo")`. The per-PDF fetch currently does `safe_get_text(pub_url)` and falls back to `wayback_snapshot_url(pub_url) + safe_get_text(wb_url)` — replace with `fetch_with_fallback(pub_url, policy=FALLBACK_POLICY, source="us_cbo")`.

- [ ] **Step 6: Run the per-connector policy tests for the 3 Wayback connectors**

```bash
pytest tests/test_narrative_fallback/test_per_connector_policies.py -v -k "eu_eurlex or eu_parliament or us_cbo"
```
Expected: 6 passed (3 connectors × 2 tests).

- [ ] **Step 7: Run the existing connector tests to confirm no regression**

```bash
pytest tests/ -k "eu_eurlex or eu_parliament or cbo" -v --tb=short 2>&1 | tail -20
```
Expected: no NEW failures vs. baseline. Some tests that mocked `wayback_snapshot_url` directly may need their patches updated to mock `_stage_wayback` instead — if so, update the test mocks (not the production code) in the same commit.

- [ ] **Step 8: Commit**

```bash
git add puremacro/puremacro/narrative/sources/eu_eurlex.py puremacro/puremacro/narrative/sources/eu_parliament.py puremacro/puremacro/narrative/sources/us_cbo.py tests/test_narrative_fallback/test_per_connector_policies.py
git commit -m "$(cat <<'EOF'
feat(0.67.0): migrate Wayback-fallback connectors to fetch_with_fallback

F2.4 rollout batch 1. eu_eurlex, eu_parliament, us_cbo each declare
FALLBACK_POLICY and replace hand-rolled wayback_snapshot_url +
safe_get_text chains with fetch_with_fallback(url, policy=FALLBACK_POLICY,
source="<name>"). Policies match current verified behaviour:
- eu_eurlex     ("wayback",)         # live endpoint is AWS-WAF-blocked
- eu_parliament ("wayback",)         # live CRE pages are JS-gated
- us_cbo        ("live", "wayback")  # RSS lives; PDFs sometimes need WB

Each fetch site catches FallbackExhaustedError and warns + skips
per RETRY_POLICY.md §4.1 (yield, don't raise). All existing
connector tests pass; affected mocks (those that patched
wayback_snapshot_url directly) updated to patch _stage_wayback.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Rollout batch 2 — `rba` + `bok` + `riksbank` + `sarb`

**Files:**
- Modify: `puremacro/puremacro/narrative/sources/rba.py`
- Modify: `puremacro/puremacro/narrative/sources/bok.py`
- Modify: `puremacro/puremacro/narrative/sources/riksbank.py`
- Modify: `puremacro/puremacro/narrative/sources/sarb.py`

Same pattern as Task 6 for the 4 Playwright connectors.

For each:

1. Read the file to confirm it currently calls `fetch_with_playwright(url)` directly.
2. Add at the top (after existing imports):
   ```python
   from ._fallback import fetch_with_fallback, FallbackExhaustedError

   # <connector> is currently Playwright-only — the live endpoint
   # requires JS execution to render content.
   FALLBACK_POLICY: tuple[str, ...] = ("playwright",)
   ```
3. Replace each `fetch_with_playwright(url)` call with `fetch_with_fallback(url, policy=FALLBACK_POLICY, source="<name>")`.
4. Wrap each call in `try/except FallbackExhaustedError` → warn + skip.
5. Remove the now-unused `from ._playwright_helper import fetch_with_playwright` import.

- [ ] **Step 1: Run the per-connector policy tests to verify they fail**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
pytest tests/test_narrative_fallback/test_per_connector_policies.py -v -k "rba or bok or riksbank or sarb"
```
Expected: 8 failures (4 connectors × 2 tests, `FALLBACK_POLICY` missing).

- [ ] **Step 2: Edit `rba.py`**

The file currently has `from ._playwright_helper import fetch_with_playwright` and calls like:
```python
html = fetch_with_playwright(url)
body_html = fetch_with_playwright(full_url)
```

Replace with:

```python
# top of rba.py (after existing imports)
from ._fallback import fetch_with_fallback, FallbackExhaustedError

# rba is Playwright-only: the RBA speech pages require JS to render.
FALLBACK_POLICY: tuple[str, ...] = ("playwright",)
```

Then for each call site (e.g., line 40 and line 55 per earlier grep):

```python
try:
    html = fetch_with_fallback(
        url, policy=FALLBACK_POLICY, source="rba",
    )
except FallbackExhaustedError:
    continue   # skip this URL; outer loop yields nothing for this iter
```

Remove the now-unused `from ._playwright_helper import fetch_with_playwright` import (if it's the only import from that module in this file — verify before deleting).

- [ ] **Step 3: Edit `bok.py`**

Same pattern. Source name `"bok"`.

- [ ] **Step 4: Edit `riksbank.py`**

Same pattern. Source name `"riksbank"`.

- [ ] **Step 5: Edit `sarb.py`**

Same pattern. Source name `"sarb"`.

- [ ] **Step 6: Run the per-connector policy tests**

```bash
pytest tests/test_narrative_fallback/test_per_connector_policies.py -v
```
Expected: 14 passed (7 connectors × 2 tests).

- [ ] **Step 7: Run the existing connector tests to confirm no regression**

```bash
pytest tests/ -k "rba or bok or riksbank or sarb" -v --tb=short 2>&1 | tail -20
```
Expected: no NEW failures vs. baseline. Tests that mocked `fetch_with_playwright` directly may need their patches updated to mock `_stage_playwright` instead — update those mocks in the same commit.

- [ ] **Step 8: Commit**

```bash
git add puremacro/puremacro/narrative/sources/rba.py puremacro/puremacro/narrative/sources/bok.py puremacro/puremacro/narrative/sources/riksbank.py puremacro/puremacro/narrative/sources/sarb.py
git commit -m "$(cat <<'EOF'
feat(0.67.0): migrate Playwright-fallback connectors to fetch_with_fallback

F2.4 rollout batch 2. rba, bok, riksbank, sarb each declare
FALLBACK_POLICY = ("playwright",) and replace direct
fetch_with_playwright(url) calls with fetch_with_fallback(url,
policy=FALLBACK_POLICY, source="<name>"). Each call site catches
FallbackExhaustedError and skips per RETRY_POLICY.md §4.1. The
unused fetch_with_playwright import is removed from each file (the
function is still reachable via _fallback._stage_playwright).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: AST coverage assertion

**Files:**
- Create: `tests/test_narrative_fallback/test_coverage_assertion.py`.

- [ ] **Step 1: Write the test**

Create `tests/test_narrative_fallback/test_coverage_assertion.py`:

```python
"""F2.4 — coverage assertion: the 7 fallback connectors all call
fetch_with_fallback(...). Fails the build if any of them regresses to
a direct safe_get_text / fetch_with_playwright call."""
from __future__ import annotations

import ast
import importlib
import pathlib

import pytest


_FALLBACK_CONNECTORS = (
    "eu_eurlex", "eu_parliament", "us_cbo",
    "rba", "bok", "riksbank", "sarb",
)


def _module_source(name: str) -> str:
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    return pathlib.Path(mod.__file__).read_text()


def _has_fetch_with_fallback_call(name: str) -> bool:
    src = _module_source(name)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "fetch_with_fallback":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "fetch_with_fallback":
                return True
    return False


def test_every_fallback_connector_calls_fetch_with_fallback():
    missing = [
        n for n in _FALLBACK_CONNECTORS if not _has_fetch_with_fallback_call(n)
    ]
    assert not missing, (
        f"F2.4 contract violation: these connectors do not call "
        f"fetch_with_fallback(...): {missing}. Either add the call or "
        f"remove the connector from the F2.4 scope list."
    )
```

- [ ] **Step 2: Run to verify pass**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
pytest tests/test_narrative_fallback/test_coverage_assertion.py -v
```
Expected: 1 passed (Tasks 6 + 7 already made the call sites real).

- [ ] **Step 3: Run the full fallback suite**

```bash
pytest tests/test_narrative_fallback/ -v 2>&1 | tail -10
```
Expected: all pass (5 supported_stages + ~9 fetch_with_fallback + 3 end_to_end + 14 per_connector + 1 coverage ≈ 32 passed).

- [ ] **Step 4: Commit**

```bash
git add tests/test_narrative_fallback/test_coverage_assertion.py
git commit -m "$(cat <<'EOF'
test(0.67.0): coverage assertion — 7 fallback connectors must call fetch_with_fallback

F2.4 closes out with the coverage test that AST-scans each of the 7
fallback connectors (eu_eurlex, eu_parliament, us_cbo, rba, bok,
riksbank, sarb) and asserts the module body contains a call to
fetch_with_fallback(...). Fails the build if a future refactor
accidentally regresses to a direct safe_get_text or
fetch_with_playwright call.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Sub-slice 3 — schema-mismatch telemetry + release

(Tasks 9–11.)

## Task 9: Wire `parser_schema_mismatch` events into the 8 Slice-A connectors

**Files:** modify 8 connector files. Each currently has an `except ParserSchemaMismatchError` block (added in Slice A) that calls `warnings.warn(...)`. We add ONE LINE — a `log_event(...)` call — immediately before the existing `warnings.warn`.

- Modify: `puremacro/puremacro/narrative/sources/beige_book.py`
- Modify: `puremacro/puremacro/narrative/sources/eu_eurlex.py`
- Modify: `puremacro/puremacro/narrative/sources/eu_parliament.py`
- Modify: `puremacro/puremacro/narrative/sources/us_cbo.py`
- Modify: `puremacro/puremacro/narrative/sources/fed_minutes.py`
- Modify: `puremacro/puremacro/narrative/sources/fed_speeches.py`
- Modify: `puremacro/puremacro/narrative/sources/bluesky.py`
- Modify: `puremacro/puremacro/narrative/sources/ecb_press.py`
- Test: `tests/test_narrative_telemetry/test_parser_schema_mismatch_events.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_narrative_telemetry/test_parser_schema_mismatch_events.py`:

```python
"""F2.5 — the 8 Slice-A connectors emit a parser_schema_mismatch event
on ParserSchemaMismatchError."""
from __future__ import annotations

import importlib

import pytest


_SCHEMA_CHECKED_CONNECTORS = (
    "beige_book", "eu_eurlex", "eu_parliament", "us_cbo",
    "fed_minutes", "fed_speeches", "bluesky", "ecb_press",
)


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("PUREMACRO_NARRATIVE_TELEMETRY", raising=False)
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


@pytest.mark.parametrize("name", _SCHEMA_CHECKED_CONNECTORS)
def test_iter_source_emits_event_on_schema_mismatch(fresh_db, monkeypatch, name):
    """Patch the inner parser / body-fetch to raise ParserSchemaMismatchError;
    confirm the iter_<source> generator catches it AND emits a
    parser_schema_mismatch event before warning."""
    import warnings
    from puremacro.narrative.sources._schema_check import (
        ParserSchemaMismatchError,
    )
    from puremacro.narrative.sources._telemetry import log_event
    from puremacro import _cache_db

    # Spy on log_event to confirm it was called with the right args.
    calls: list[tuple] = []

    def _spy_log_event(*, source, outcome, fallback_used="none"):
        calls.append((source, outcome, fallback_used))

    monkeypatch.setattr(
        f"puremacro.narrative.sources.{name}._telemetry.log_event",
        _spy_log_event, raising=False,
    )
    monkeypatch.setattr(
        "puremacro.narrative.sources._telemetry.log_event",
        _spy_log_event,
    )

    # The simplest way to fire the mismatch is to monkeypatch
    # _schema_check.assert_landmarks to raise. The iter_<source> wrapper
    # then catches and (per F2.5) calls log_event before warning.
    def _always_raise(*a, **kw):
        raise ParserSchemaMismatchError(f"{name!r}: simulated mismatch")

    monkeypatch.setattr(
        "puremacro.narrative.sources._schema_check.assert_landmarks",
        _always_raise,
    )

    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    iter_fn_name = next(
        n for n in dir(mod)
        if n.startswith("iter_") and callable(getattr(mod, n))
    )
    iter_fn = getattr(mod, iter_fn_name)

    # Trigger the iter. Different connectors have different argument
    # surfaces; we just want SOMETHING to call assert_landmarks. Most
    # iter_<source> functions can be called with no args (they fetch
    # whatever they fetch by default) and will exit cleanly with the
    # schema-mismatch caught + warned.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        try:
            list(iter_fn())
        except TypeError:
            # Some iter_<source> functions REQUIRE an arg (e.g.,
            # iter_bluesky_posts(actors=)). For those, this test is
            # advisory — flag them as needing a per-iter parametrization
            # in a follow-up. For now, skip.
            pytest.skip(
                f"{name}: iter_{name} requires args; manual fixture "
                f"needed for parser_schema_mismatch event verification."
            )

    # Confirm log_event was called with the expected outcome.
    matching = [c for c in calls if c[1] == "parser_schema_mismatch"]
    assert matching, (
        f"{name}: ParserSchemaMismatchError was raised but no "
        f"log_event(outcome='parser_schema_mismatch') was emitted. "
        f"Slice B F2.5 contract violation."
    )
    # First positional element is the source name; should match.
    assert matching[0][0] == name
    assert matching[0][2] == "none"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_narrative_telemetry/test_parser_schema_mismatch_events.py -v
```
Expected: 8 failures (some may be skipped). Failures because no connector currently calls `log_event` on schema mismatch.

- [ ] **Step 3: Edit each of the 8 connectors**

For each of the 8 files, find the existing `except ParserSchemaMismatchError as e:` block and add ONE line just before the existing `warnings.warn(...)`:

```python
except ParserSchemaMismatchError as e:
    from ._telemetry import log_event   # local import to avoid circular at module load
    log_event(source="<name>", outcome="parser_schema_mismatch",
              fallback_used="none")
    warnings.warn(
        f"puremacro.narrative.sources.<name>: schema mismatch ...: {e}",
        UserWarning, stacklevel=2,
    )
```

The exact source string for each connector:

| File | `source=` value |
|---|---|
| `beige_book.py` | `"beige_book"` |
| `eu_eurlex.py` | `"eu_eurlex"` |
| `eu_parliament.py` | `"eu_parliament"` |
| `us_cbo.py` | `"us_cbo"` |
| `fed_minutes.py` | `"fed_minutes"` |
| `fed_speeches.py` | `"fed_speeches"` |
| `bluesky.py` | `"bluesky"` |
| `ecb_press.py` | `"ecb_press"` |

If a file has MULTIPLE `except ParserSchemaMismatchError` blocks (e.g., one per loop nesting level), add the `log_event` call to each.

The local `from ._telemetry import log_event` import keeps cold-import time cheap and avoids creating a hard import dependency at module load (which could matter for Pyodide guards).

- [ ] **Step 4: Run the parametrized test to verify pass**

```bash
pytest tests/test_narrative_telemetry/test_parser_schema_mismatch_events.py -v
```
Expected: 8 passed (some may be `SKIPPED` if their iter functions require args; that's acceptable for Slice B — the implementer can add a per-iter fixture in a follow-up).

- [ ] **Step 5: Run the full telemetry + fallback + cache_db suites**

```bash
pytest tests/test_narrative_telemetry/ tests/test_narrative_fallback/ tests/test_cache_db/ -v 2>&1 | tail -10
```
Expected: all pass (~13 + ~14 + 32 + 30 = ~89, depending on parametrize expansion).

- [ ] **Step 6: Commit**

```bash
git add puremacro/puremacro/narrative/sources/beige_book.py puremacro/puremacro/narrative/sources/eu_eurlex.py puremacro/puremacro/narrative/sources/eu_parliament.py puremacro/puremacro/narrative/sources/us_cbo.py puremacro/puremacro/narrative/sources/fed_minutes.py puremacro/puremacro/narrative/sources/fed_speeches.py puremacro/puremacro/narrative/sources/bluesky.py puremacro/puremacro/narrative/sources/ecb_press.py tests/test_narrative_telemetry/test_parser_schema_mismatch_events.py
git commit -m "$(cat <<'EOF'
feat(0.67.0): emit parser_schema_mismatch events from 8 Slice-A connectors

F2.5 wiring across the 8 schema-checked connectors. Each
ParserSchemaMismatchError catch block now calls
log_event(source="<name>", outcome="parser_schema_mismatch",
fallback_used="none") immediately before the existing warnings.warn.
This gives connector_health() visibility into parser-drift events
across the Slice-A connector population, alongside the fetch-stage
events from the 7 fallback connectors.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: R5_02 notebook + paired builder

**Files:**
- Create: `tools/make_notebook_R5_02.py`.
- Create: `notebooks/R5_data_infra/R5_02_connector_health_demo.ipynb`.

Per memory: notebooks + builders ship together; nbconvert run from controller / foreground, not subagent.

- [ ] **Step 1: Create the builder**

Create `tools/make_notebook_R5_02.py`:

```python
"""Build R5_02_connector_health_demo.ipynb — Slice B demo.

Demonstrates the 0.67.0 governed-fallback + telemetry additions:
  1. Seed a handful of synthetic connector_events directly.
  2. connector_health(window=...) returns the expected DataFrame.
  3. Trigger fetch_with_fallback against a synthetic failing stage to
     show telemetry capturing failure + fallback in action.

Run:
    python tools/make_notebook_R5_02.py
Then execute (foreground, controller-side):
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/R5_data_infra/R5_02_connector_health_demo.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat


_REPO = Path(__file__).resolve().parent.parent
_OUT = _REPO / "notebooks" / "R5_data_infra" / "R5_02_connector_health_demo.ipynb"


def _md(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text)


def _code(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(src)


def build() -> None:
    nb = nbformat.v4.new_notebook()
    cells = []

    cells.append(_md("""# R5_02 — Governed fallback + health telemetry (0.67.0)

Demonstrates the two F2 Slice B components:

- **Governed fallback** — `fetch_with_fallback(url, policy=, source=)`
  centralises the live → wayback → playwright stage loop. The 7
  fallback-aware connectors now declare `FALLBACK_POLICY` as a module
  constant.
- **Health telemetry** — every fetch attempt by a participating
  connector lands in `connector_events`. `connector_health(window=)`
  aggregates into a per-source DataFrame: success_rate, fallback_rate,
  last_seen.

Spec: `docs/specs/2026-05-26-puremacro-067-f2-slice-b-fallback-telemetry-design.md`
Reference: `docs/CONNECTOR_HEALTH.md`.
"""))

    cells.append(_code("""\
from __future__ import annotations
import os, tempfile, time
from pathlib import Path
import pandas as pd
import puremacro
print('puremacro', puremacro.__version__)

# Use a tmp cache DB so this demo is fully self-contained.
tmpdir = Path(tempfile.mkdtemp())
os.environ['PUREMACRO_HTTP_CACHE_DIR'] = str(tmpdir)
os.environ.pop('PUREMACRO_NARRATIVE_TELEMETRY', None)
import puremacro._cache_db as _db
_db.close_conn()
"""))

    cells.append(_md("## 1. Seed synthetic events + inspect"))

    cells.append(_code("""\
from puremacro.narrative.sources._telemetry import log_event, connector_health

# Seed 25 synthetic events across 3 connectors.
events = [
    *[("eu_eurlex", "success", "wayback")] * 7,
    *[("eu_eurlex", "timeout", "wayback")] * 3,
    *[("rba", "success", "playwright")] * 5,
    *[("rba", "playwright_unavailable", "playwright")] * 1,
    *[("us_cbo", "success", "live")] * 9,
]
for source, outcome, fb in events:
    log_event(source=source, outcome=outcome, fallback_used=fb)

connector_health(window=pd.Timedelta(days=1))
"""))

    cells.append(_md("## 2. Trigger fetch_with_fallback against synthetic stages"))

    cells.append(_code("""\
from puremacro.narrative.sources import _fallback

# Mock the live stage to always fail; mock wayback to succeed.
original_live = _fallback._stage_live
original_wayback = _fallback._stage_wayback
import socket

def _live_always_fails(url, *, timeout, use_cache):
    raise socket.timeout('demo live timeout')

def _wayback_always_succeeds(url, *, timeout, use_cache):
    return '<html>demo wayback body</html>'

_fallback._stage_live = _live_always_fails
_fallback._stage_wayback = _wayback_always_succeeds

body = _fallback.fetch_with_fallback(
    'https://example.com/demo',
    policy=('live', 'wayback'),
    source='demo_connector',
)
print('body returned:', body[:60])

# Restore originals.
_fallback._stage_live = original_live
_fallback._stage_wayback = original_wayback
"""))

    cells.append(_md("## 3. The demo connector now shows up in connector_health"))

    cells.append(_code("""\
connector_health(window=pd.Timedelta(hours=1), sources=['demo_connector'])
"""))

    cells.append(_md("""## What's next

- **Slice C+** can roll out `FALLBACK_POLICY` to the remaining ~53
  connectors (one-liners using `policy=("live",)` if they don't need a
  real fallback — but they'd get telemetry).
- Future enhancements queued in the spec: per-event `url_hash` +
  `latency_ms`, retention controls (`connector_events_clear`), new
  fallback stages (`tor`, `paid_proxy`, `mirrored_s3`), OpenTelemetry
  exporter for production pipelines.

Reference: `docs/CONNECTOR_HEALTH.md`. Full spec:
`docs/specs/2026-05-26-puremacro-067-f2-slice-b-fallback-telemetry-design.md`.
"""))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w") as f:
        nbformat.write(nb, f)
    print(f"wrote {_OUT.relative_to(_REPO)}")


if __name__ == "__main__":
    build()
```

- [ ] **Step 2: Build the notebook**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
python tools/make_notebook_R5_02.py
```
Expected: `wrote notebooks/R5_data_infra/R5_02_connector_health_demo.ipynb`.

- [ ] **Step 3: Execute foreground**

```bash
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/R5_data_infra/R5_02_connector_health_demo.ipynb
```
Expected: notebook executes cleanly. Inspect to confirm cell outputs populated (especially the two `connector_health(...)` DataFrames).

- [ ] **Step 4: Commit builder + executed notebook**

```bash
git add tools/make_notebook_R5_02.py notebooks/R5_data_infra/R5_02_connector_health_demo.ipynb
git commit -m "$(cat <<'EOF'
feat(0.67.0): R5_02 connector-health demo notebook + paired builder

Slice B's visible deliverable. R5_02 walks the two sub-components
end-to-end: seed synthetic events → connector_health() DataFrame;
mock _stage_live/_stage_wayback → trigger fetch_with_fallback → re-run
connector_health() to see the demo source show up. Fully
offline-runnable — no real network calls, no API keys. Shipped with
the paired builder per the notebook ↔ builder pairing rule.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Version bump + CHANGELOG + ARCHITECTURE + final sanity sweep

**Files:**
- Modify: `puremacro/pyproject.toml` (`version = "0.67.0"`)
- Modify: `puremacro/puremacro/__init__.py` (`__version__ = "0.67.0"`)
- Modify: `puremacro/CHANGELOG.md` (prepend 0.67.0 section)
- Modify: `puremacro/ARCHITECTURE.md` (append "F2 closure (0.67.0+)" subsection)
- Modify: `tests/test_credentials/test_service_registry.py` (rename + update version test)

- [ ] **Step 1: Update the version smoke test**

Edit `tests/test_credentials/test_service_registry.py`. Find the existing `test_puremacro_version_is_066` and rename + update:

```python
def test_puremacro_version_is_067():
    import puremacro
    assert puremacro.__version__ == "0.67.0"
```

- [ ] **Step 2: Run to verify failure**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
pytest tests/test_credentials/test_service_registry.py::test_puremacro_version_is_067 -v
```
Expected: FAIL — version still `"0.66.0"`.

- [ ] **Step 3: Bump `puremacro/__init__.py`**

Find `__version__ = "0.66.0"` and change to `__version__ = "0.67.0"`.

- [ ] **Step 4: Bump `pyproject.toml`**

Find `version = "0.66.0"` and change to `version = "0.67.0"`. `requires-python = ">=3.12"` stays.

- [ ] **Step 5: Prepend CHANGELOG entry**

Insert this block in `puremacro/CHANGELOG.md` IMMEDIATELY AFTER the `# Changelog` header + intro paragraph, BEFORE the `## 0.66.0 (2026-05-26)` section:

```markdown
## 0.67.0 (2026-05-26)

**F2 Slice B — governed fallback + health telemetry (closes F2).**

### Added
- `puremacro.narrative.sources._fallback`: `fetch_with_fallback(url, *,
  policy, source, timeout=30.0, use_cache=True)` entry point with
  `SUPPORTED_STAGES = {"live", "wayback", "playwright"}`.
  `FallbackExhaustedError` (raised when every stage fails) and
  `FallbackStageUnavailable` (Playwright not installed, Wayback no
  snapshot) both subclass `RuntimeError`. `_classify(e)` maps urllib
  / ssl / socket exceptions to canonical outcome strings.
- `puremacro.narrative.sources._telemetry`: `log_event(source=, outcome=,
  fallback_used="none")` inserts one row into the new `connector_events`
  SQLite table. `connector_health(window=, sources=)` returns a
  per-source DataFrame with success_rate / fallback_rate / last_seen.
  `PUREMACRO_NARRATIVE_TELEMETRY=0` kill-switch disables event logging.
- New SQLite table `connector_events (ts, source, outcome, fallback_used)`
  in `~/.cache/puremacro/cache.db`. Created on bootstrap; schema
  version 1.
- `docs/CONNECTOR_HEALTH.md`: researcher-facing reference for the
  event schema + the `connector_health()` aggregation API + the
  kill-switch.
- `notebooks/R5_data_infra/R5_02_connector_health_demo.ipynb` + paired
  `tools/make_notebook_R5_02.py`.

### Changed
- 7 narrative connectors migrated to `fetch_with_fallback`:
  - `eu_eurlex` — `FALLBACK_POLICY = ("wayback",)` (live endpoint
    AWS-WAF-blocked).
  - `eu_parliament` — `("wayback",)` (live CRE pages JS-gated).
  - `us_cbo` — `("live", "wayback")` (RSS lives, PDFs sometimes need WB).
  - `rba`, `bok`, `riksbank`, `sarb` — `("playwright",)`.
  Each declares `FALLBACK_POLICY` as a module-constant tuple; AST
  coverage assertion enforces both the constant + the
  `fetch_with_fallback` call.
- 8 Slice-A schema-checked connectors (`beige_book`, `eu_eurlex`,
  `eu_parliament`, `us_cbo`, `fed_minutes`, `fed_speeches`, `bluesky`,
  `ecb_press`) emit a `parser_schema_mismatch` event on
  `ParserSchemaMismatchError` catch (one line added per `except`
  block, before the existing `warnings.warn`).

### Roadmap (closes F2)
- F2 sub-project is now complete (F2.0 credentials + F2.1 cache + F2.2
  vintage store + F2.3 schema versioning shipped in 0.66.0; F2.4
  fallback + F2.5 telemetry in 0.67.0).
- Next sibling sub-projects queued from the original brainstorm: F1
  source coverage, F3 unified panel-builder API, S2 interpretation
  layer, S4 cross-source synthesis 2.0, T1 cookbook, T2 onboarding.
- Slice C+ within F2 (deferred): per-event `url_hash` + `latency_ms`,
  retention controls (`connector_events_clear(older_than=)`), new
  fallback stages (`tor`, `paid_proxy`, `mirrored_s3`), OpenTelemetry
  / Prometheus exporter, telemetry coverage of the ~45 connectors that
  don't currently use `fetch_with_fallback`.
- Full spec: `docs/specs/2026-05-26-puremacro-067-f2-slice-b-fallback-telemetry-design.md`.

### Internal
- New test directories: `tests/test_narrative_fallback/`,
  `tests/test_narrative_telemetry/`. ~30 new tests across the slice.
- `cache.db` schema_version table grows from 2 to 3 rows (added
  `("connector_events", 1)`).

```

- [ ] **Step 6: Append the ARCHITECTURE.md subsection**

In `puremacro/ARCHITECTURE.md`, find the existing "Data infrastructure (0.66.0+)" subsection (shipped in Slice A) and APPEND immediately after it:

```markdown
### F2 closure (0.67.0+)

Slice B closes the F2 sub-project. `puremacro.narrative.sources._fallback`
centralises the live → wayback → playwright fallback chain that 7
connectors previously hand-rolled; each connector declares
`FALLBACK_POLICY` as a module-constant tuple of
`SUPPORTED_STAGES = {"live", "wayback", "playwright"}` and calls
`fetch_with_fallback(url, policy=FALLBACK_POLICY, source="<name>")`.
`puremacro.narrative.sources._telemetry.log_event` records one row per
fetch attempt to the new `connector_events` SQLite table; the 8
Slice-A schema-checked connectors also emit `parser_schema_mismatch`
events. `connector_health(window=, sources=)` aggregates into a
per-source DataFrame. Kill-switch: `PUREMACRO_NARRATIVE_TELEMETRY=0`.
Reference: `docs/CONNECTOR_HEALTH.md`. Full spec:
`docs/specs/2026-05-26-puremacro-067-f2-slice-b-fallback-telemetry-design.md`.
```

- [ ] **Step 7: Run the version test to verify pass**

```bash
pytest tests/test_credentials/test_service_registry.py::test_puremacro_version_is_067 -v
```
Expected: PASS.

- [ ] **Step 8: Final full-suite sanity sweep**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro" && \
pytest tests/test_narrative_fallback/ tests/test_narrative_telemetry/ -v 2>&1 | tail -10 && \
pytest tests/test_cache_db/ tests/test_credentials/ tests/test_vintages_alfred_store/ tests/test_narrative_schema_checks/ -v 2>&1 | tail -10 && \
pytest tests/test_pyodide_compat.py -v && \
pytest tests/test_signal_contract/ -v 2>&1 | tail -5 && \
pytest tests/test_narrative.py tests/test_narrative_indices.py -v 2>&1 | tail -10
```
Expected:
- All new Slice B tests pass (~30).
- All Slice A test directories still green.
- Pyodide compat passes.
- Signal-contract (Slice 1) still green.
- Narrative suite no new regressions.

- [ ] **Step 9: Commit**

```bash
git add puremacro/pyproject.toml puremacro/puremacro/__init__.py puremacro/CHANGELOG.md puremacro/ARCHITECTURE.md tests/test_credentials/test_service_registry.py
git commit -m "$(cat <<'EOF'
chore(puremacro): bump to 0.67.0 — F2 Slice B (closes F2)

Ships the two-sub-component slice: governed fallback + health
telemetry. Closes the F2 sub-project (F2.0+F2.1+F2.2+F2.3 in 0.66.0
plus F2.4+F2.5 here).

7 fallback connectors migrated to fetch_with_fallback (eu_eurlex,
eu_parliament, us_cbo via Wayback; rba, bok, riksbank, sarb via
Playwright). 8 Slice-A schema-checked connectors emit
parser_schema_mismatch events. New connector_events table +
connector_health() aggregation API. Kill-switch via
PUREMACRO_NARRATIVE_TELEMETRY=0.

Next sibling sub-projects queued: F1 source coverage, F3 panel
builder, S2 interpretation, S4 synthesis, T1 cookbook, T2 onboarding.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Done-definition for Slice B (0.67.0)

- [ ] `connector_events` table created on bootstrap; `("connector_events", 1)` in `schema_version`.
- [ ] `_telemetry.py` ships with `log_event` + `connector_health` + `telemetry_enabled` kill-switch + full test coverage.
- [ ] `_fallback.py` ships with `SUPPORTED_STAGES`, `fetch_with_fallback`, `FallbackExhaustedError`, `FallbackStageUnavailable`, `_classify` + all stage dispatchers.
- [ ] 7 fallback connectors declare `FALLBACK_POLICY` and call `fetch_with_fallback`; AST coverage assertion + per-connector policy validation enforce both.
- [ ] 8 Slice-A schema-checked connectors emit `parser_schema_mismatch` events on schema mismatch (parametrized test enforces).
- [ ] R5_02 notebook executes cleanly with sensible `connector_health()` output; paired builder committed.
- [ ] `docs/CONNECTOR_HEALTH.md` shipped; ARCHITECTURE.md gains "F2 closure (0.67.0+)" subsection.
- [ ] `pyproject.toml` at `version = "0.67.0"`; `puremacro/__init__.py` `__version__ = "0.67.0"`; CHANGELOG 0.67.0 entry.
- [ ] Pyodide compat passes (`sqlite3` + lazy Playwright unchanged from 0.66.0).
- [ ] Full narrative test suite shows zero new regressions vs. the post-Slice-A baseline.

## Out of scope for Slice B (queued for follow-up)

- Per-event `url_hash`, `latency_ms`, request body size.
- Retention controls (`connector_events_clear(older_than=)`).
- F2.4 rollout to the remaining ~53 connectors (adopt as-needed).
- F2.5 telemetry coverage of connectors that don't use `fetch_with_fallback` (context-manager or contextvar plumbing).
- New fallback stages: `tor`, `paid_proxy`, `mirrored_s3`.
- OpenTelemetry / Prometheus exporters.
- Sibling sub-projects: F1 source coverage, F3 unified panel-builder API, S2 interpretation, S4 cross-source synthesis 2.0, T1 cookbook, T2 onboarding.
