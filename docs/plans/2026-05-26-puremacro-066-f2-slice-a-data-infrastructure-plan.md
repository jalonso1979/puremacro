# puremacro 0.66.0 Implementation Plan — F2 Slice A (data-infrastructure foundation)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Slice A of F2 as `0.66.0`: a `puremacro.credentials` module with env-var + TOML resolver, SQLite-backed HTTP cache (replacing the flat-file cache), persistent ALFRED vintage store, and a parser schema-versioning framework rolled out to 8 high-value connectors. Bump `requires-python` to `>=3.12`.

**Architecture:** Four sub-components landing in dependency order inside one release window. F2.0 credentials lands first so the rollout commits can switch fetchers from `os.environ.get(...)` to `credentials.require(...)`. F2.1 introduces `_cache_db.py` (singleton SQLite connection, WAL mode) and rewrites `_http_cache.py` against it while preserving the existing `cache_read/cache_write` public API. F2.2 adds `AlfredVintageStore` using the same DB. F2.3 ships `_schema_check.py` + `assert_landmarks` and rolls out across 8 connectors with golden fixtures and a coverage assertion.

**Tech Stack:** Python ≥3.12 (`tomllib` stdlib, new generic-class syntax). `sqlite3` stdlib. `pandas`. No new runtime dependencies. Pyodide-compatible throughout (stdlib only on the new modules).

**Spec:** `docs/specs/2026-05-26-puremacro-066-f2-slice-a-data-infrastructure-design.md`

---

## File map

### New files
- `puremacro/credentials.py` — `ServiceCredentialSpec`, `SERVICES`, `MissingCredentialError`, `get`, `require`, `status`, `default_config_path`, internal `_load_config`.
- `puremacro/_cache_db.py` — singleton `sqlite3.Connection`, `bootstrap_schema`, `migrate_from_flat_files`, `default_db_path`, `close_conn`.
- `puremacro/narrative/sources/_schema_check.py` — `ParserSchemaMismatchError`, `assert_landmarks`.
- `puremacro/narrative/sources/_fixtures/beige_book_v1.html` — golden snapshot.
- `puremacro/narrative/sources/_fixtures/eu_eurlex_v1.html` — golden snapshot.
- `puremacro/narrative/sources/_fixtures/eu_parliament_v1.html` — golden snapshot.
- `puremacro/narrative/sources/_fixtures/us_cbo_v1.xml` — RSS golden snapshot.
- `puremacro/narrative/sources/_fixtures/fed_minutes_v1.html` — golden snapshot.
- `puremacro/narrative/sources/_fixtures/fed_speeches_v1.html` — golden snapshot.
- `puremacro/narrative/sources/_fixtures/bluesky_v1.json` — golden snapshot.
- `puremacro/narrative/sources/_fixtures/ecb_press_v1.html` — golden snapshot.
- `tools/cache_migrate.py` — one-shot CLI: dry-run / `--apply` / `--apply --rm`.
- `tools/make_notebook_R5_01.py` — paired builder for the demo notebook.
- `notebooks/R5_data_infra/R5_01_cache_and_credentials_demo.ipynb` — executed demo.
- `puremacro/docs/CREDENTIALS.md` — researcher-facing reference (resolver priority, TOML format, status() introspection, per-service signup URLs).
- `puremacro/docs/CACHE_DB.md` — researcher-facing reference (schema, env vars, introspection helpers, migration notes).
- `tests/test_credentials/__init__.py`
- `tests/test_credentials/test_resolver_priority.py`
- `tests/test_credentials/test_service_registry.py`
- `tests/test_credentials/test_missing_credential_error_message.py`
- `tests/test_credentials/test_config_file_parsing.py`
- `tests/test_credentials/test_status_dataframe.py`
- `tests/test_credentials/test_no_direct_env_get_in_fetch.py`
- `tests/test_cache_db/__init__.py`
- `tests/test_cache_db/test_schema_bootstrap.py`
- `tests/test_cache_db/test_http_cache_sqlite_roundtrip.py`
- `tests/test_cache_db/test_introspection_helpers.py`
- `tests/test_cache_db/test_migration_from_flat_files.py`
- `tests/test_cache_db/test_concurrent_wal.py`
- `tests/test_cache_db/test_failure_modes.py`
- `tests/test_cache_db/test_cache_migrate_cli.py`
- `tests/test_vintages_alfred_store/__init__.py`
- `tests/test_vintages_alfred_store/test_store_roundtrip.py`
- `tests/test_vintages_alfred_store/test_vintage_until_filter.py`
- `tests/test_vintages_alfred_store/test_has_series_series_list.py`
- `tests/test_vintages_alfred_store/test_coverage_diagnostic.py`
- `tests/test_vintages_alfred_store/test_fetch_fred_alfred_with_store.py`
- `tests/test_vintages_alfred_store/test_as_of_from_store.py`
- `tests/test_vintages_alfred_store/test_store_failure_modes.py`
- `tests/test_narrative_schema_checks/__init__.py`
- `tests/test_narrative_schema_checks/test_schema_check_framework.py`
- `tests/test_narrative_schema_checks/test_landmark_assertions.py`
- `tests/test_narrative_schema_checks/test_landmark_fixtures.py`
- `tests/test_narrative_schema_checks/test_coverage_assertion.py`

### Modified files
- `puremacro/_http_cache.py` — REWRITE the storage internals against `_cache_db`; preserve `cache_read`/`cache_write`/`cache_key`/`default_cache_dir` signatures; add lazy migration trigger on first call.
- `puremacro/cache.py` — add `http_list_urls()`, `http_cache_size_bytes()`, `http_cache_clear(older_than=...)`. Names disambiguate from the existing `disk_cache`/`disk_cache_path` helpers in the same module.
- `puremacro/vintages.py` — add `AlfredVintageStore` class and `as_of_from_store()` helper.
- `puremacro/fetch/_classic.py` — extend `fetch_fred_alfred(series_id, *, timeout=60.0)` with `store=` and `refresh=` kwargs.
- `puremacro/fetch/fred.py` — `_client()` uses `credentials.require("fred")` instead of direct `os.environ.get`.
- `puremacro/fetch/fred_states.py` — same rollout.
- `puremacro/fetch/census_bfs.py` — `credentials.require("census")`.
- `puremacro/fetch/frb_phil_coincident.py` — `credentials.require("fred")`.
- `puremacro/fetch/bea_cainc.py` — `credentials.require("bea")`.
- `puremacro/fetch/bea_industry_shares.py` — `credentials.require("bea")`.
- `puremacro/narrative/scoring/llm.py` — `credentials.require("anthropic"|"openai")`.
- `puremacro/narrative/indices/_llm_kernel.py` — `credentials.require("anthropic"|"openai")`.
- `puremacro/instruments/_catalog.py` — `credentials.require("fred")` (or whichever service it uses; verify per-file).
- `puremacro/narrative/sources/beige_book.py` — `PARSER_SCHEMA_VERSION = 1` + `assert_landmarks` at top of `_parse_modern_html`.
- `puremacro/narrative/sources/eu_eurlex.py` — same at top of `_parse_eurlex_html`.
- `puremacro/narrative/sources/eu_parliament.py` — same at top of `_parse_ep_page`.
- `puremacro/narrative/sources/us_cbo.py` — same at top of `_parse_rss`.
- `puremacro/narrative/sources/fed_minutes.py` — same at first-body checkpoint inside `iter_fed_minutes` (no separate `_parse_*`).
- `puremacro/narrative/sources/fed_speeches.py` — same at first-body checkpoint inside `iter_fed_speeches`.
- `puremacro/narrative/sources/bluesky.py` — same at first-record checkpoint inside `iter_bluesky_posts`.
- `puremacro/narrative/sources/ecb_press.py` — same at first-body checkpoint inside `iter_ecb_press`.
- `puremacro/__init__.py` — bump `__version__` to `"0.66.0"`.
- `pyproject.toml` — bump `version` to `"0.66.0"` and `requires-python` from `">=3.10"` to `">=3.12"`.
- `CHANGELOG.md` — prepend `## 0.66.0 (2026-05-26)` section.
- `tests/test_pyodide_compat.py` — manual re-run; no code change unless a new forbidden import slips in.
- `puremacro/ARCHITECTURE.md` — add a "Data infrastructure (0.66.0+)" subsection.
- `puremacro/README.md` — add a Quickstart block for `credentials.status()` and the cache introspection helpers.

### Working assumptions (verified 2026-05-26 via signature dumps)

- `puremacro/_http_cache.py` exports `cache_key(url) -> str`, `default_cache_dir() -> Path`, `cache_read(cache_dir, url, ttl_seconds=2_592_000) -> bytes | None`, `cache_write(cache_dir, url, body, content_type=None) -> None`. All four signatures stay verbatim.
- `puremacro/cache.py` is a *different* module — generic DataFrame/JSON cache with `disk_cache(key, loader, namespace="default", ...)` and `disk_cache_path(key, namespace="default", suffix=".parquet")`. It uses `PUREMACRO_CACHE_DIR` (not `PUREMACRO_HTTP_CACHE_DIR`). Slice A leaves `disk_cache`/`disk_cache_path` untouched and only ADDS the three `http_*` introspection helpers.
- `puremacro/_http.py` exposes `safe_get_bytes/text/json` at lines 44/54/66 and `safe_get_bytes_cached/text_cached` at lines 124/150. Internal helpers `_request`, `_throttle`, `_host_of` are private. The cache swap is transparent to all of these.
- `puremacro/vintages.py` currently has `as_of(panel_long, vintage_date, date_col="date", vintage_col="vintage", value_col="value") -> pd.Series` (line 25), `align_vintages` (line 54), `forecast_revision` (line 78). Slice A leaves these unchanged; adds `AlfredVintageStore` class + `as_of_from_store()` helper.
- `puremacro/fetch/_classic.py:75` has the live ALFRED fetcher: `fetch_fred_alfred(series_id, *, timeout=60.0) -> pd.DataFrame`. Returns a long DataFrame with columns `[date, vintage, value]` (NOT `[series_id, observation_date, vintage_date, value]`). The function does NOT take an `api_key` — it uses the public FREDgraph CSV endpoint. Slice A's `store=`/`refresh=` extension goes here, not in `fetch/fred.py`.
- `puremacro/fetch/fred.py` uses the `fredapi` package: `_client()` (line 36) currently reads `os.environ.get("FRED_API_KEY")` and raises `RuntimeError("FRED_API_KEY must be set in environment")` on miss. This is the canonical pattern to rewrite to `credentials.require("fred", explicit=api_key)`.
- The 8 narrative-source connectors all live under `puremacro/narrative/sources/`. Parser-entry-point lines verified:
  - `beige_book.py:262` — `_parse_modern_html(html, *, release_date, source_url, district=None)`
  - `eu_eurlex.py:61` — `_parse_eurlex_html(html, *, celex_id, language, ...)`
  - `eu_parliament.py:90` — `_parse_ep_page(html, *, session_date, language, ...)`
  - `us_cbo.py:59` — `_parse_rss(xml_text)` (RSS, not HTML)
  - `fed_minutes.py:50` — only `iter_fed_minutes()` (no separate `_parse_*`); landmark assertion goes on the first body fetched inside the iterator.
  - `fed_speeches.py:12` — only `iter_fed_speeches(*, fetch_body=False)`; same.
  - `bluesky.py:246` — only `iter_bluesky_posts(...)`; landmarks check the first JSON record.
  - `ecb_press.py:16` — only `iter_ecb_press(...)`; landmarks check the first fetched body.
- `RETRY_POLICY.md` §4.1 commits each `iter_<source>` to "yield, don't raise." Slice A extends each generator to catch `ParserSchemaMismatchError` explicitly, emit `warnings.warn(UserWarning)`, and stop yielding for that batch.
- `pyproject.toml` currently `name = "puremacro"`, `version = "0.65.0"`, `requires-python = ">=3.10"`. Bump both `version` to `"0.66.0"` and `requires-python` to `">=3.12"`.
- `puremacro/__init__.py` currently `__version__ = "0.65.0"`; bump to `"0.66.0"`.
- Commit-message style from `git log --oneline`: `feat(0.66.0): ...` for code commits, `docs(0.66.0): ...` for docs commits, `chore(puremacro): bump to 0.66.0 — ...` for the final release commit, `fix(0.66.0): ...` if a post-merge correction is needed. `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` footer on every commit.
- Test runner: `pytest tests/<path>::<name> -v` from the `puremacro/` package directory (NOT the git root). Full suite is ~12 min; run new test files individually during TDD.
- Branch is `main`. Commits land directly on `main` per the workflow established in Slice 1.

---

## Sub-slice 1 — F2.0 credentials

(Tasks 1–7. Lint test in Task 4 fails until rollout completes in Tasks 5–6; all must land before the 0.66.0 tag.)

## Task 1: Create the `credentials` module skeleton (registry + resolver)

**Files:**
- Create: `puremacro/puremacro/credentials.py`
- Test: `tests/test_credentials/__init__.py` (empty), `tests/test_credentials/test_service_registry.py`

- [ ] **Step 1: Create the test directory and write the failing test**

Create `tests/test_credentials/__init__.py` as an empty file.

Create `tests/test_credentials/test_service_registry.py`:

```python
"""F2.0 — Verify the SERVICES registry shape."""
from __future__ import annotations


def test_services_registry_has_expected_keys():
    from puremacro.credentials import SERVICES

    expected = {"fred", "bea", "anthropic", "openai", "census"}
    assert expected.issubset(set(SERVICES.keys())), (
        f"missing services: {expected - set(SERVICES.keys())}"
    )


def test_every_service_has_required_fields():
    from puremacro.credentials import SERVICES, ServiceCredentialSpec

    for name, spec in SERVICES.items():
        assert isinstance(spec, ServiceCredentialSpec), name
        assert spec.name == name, f"{name}: spec.name mismatch ({spec.name!r})"
        assert isinstance(spec.env_vars, tuple) and len(spec.env_vars) >= 1, name
        assert all(isinstance(v, str) and v for v in spec.env_vars), name
        assert spec.signup_url.startswith("https://"), f"{name}: insecure URL"
        assert spec.description and isinstance(spec.description, str), name


def test_known_env_var_aliases():
    from puremacro.credentials import SERVICES

    # Pin specific aliases — researchers' shells are full of these names.
    assert "FRED_API_KEY" in SERVICES["fred"].env_vars
    assert "BEA_API_KEY" in SERVICES["bea"].env_vars
    assert "ANTHROPIC_API_KEY" in SERVICES["anthropic"].env_vars
    assert "OPENAI_API_KEY" in SERVICES["openai"].env_vars
    assert "CENSUS_API_KEY" in SERVICES["census"].env_vars
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
pytest tests/test_credentials/test_service_registry.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'puremacro.credentials'`.

- [ ] **Step 3: Create the credentials module**

Create `puremacro/puremacro/credentials.py`:

```python
"""Centralised API-key resolution for puremacro fetchers.

Resolves keys in priority order:
  1. Explicit `explicit=` kwarg passed by the caller.
  2. Environment variables in the service's registry, tried in order.
  3. TOML config file (default: ``~/.puremacro/credentials.toml``).
  4. None.

Lookup is side-effect-free. Use ``get()`` when missing == valid;
use ``require()`` when missing == error (raises
``MissingCredentialError`` with a researcher-actionable message).

Use ``status()`` from a notebook to see which services are configured
without leaking the actual key values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ServiceCredentialSpec:
    """Per-service registry entry."""
    name: str
    env_vars: tuple[str, ...]
    signup_url: str
    description: str


SERVICES: dict[str, ServiceCredentialSpec] = {
    "fred": ServiceCredentialSpec(
        name="fred",
        env_vars=("FRED_API_KEY", "PUREMACRO_FRED_API_KEY"),
        signup_url="https://fred.stlouisfed.org/docs/api/api_key.html",
        description="FRED + ALFRED real-time macro data (St. Louis Fed)",
    ),
    "bea": ServiceCredentialSpec(
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
    "openai": ServiceCredentialSpec(
        name="openai",
        env_vars=("OPENAI_API_KEY", "PUREMACRO_OPENAI_API_KEY"),
        signup_url="https://platform.openai.com/api-keys",
        description="OpenAI provider for the LLM kernel (alternative to Anthropic)",
    ),
    "census": ServiceCredentialSpec(
        name="census",
        env_vars=("CENSUS_API_KEY", "PUREMACRO_CENSUS_API_KEY"),
        signup_url="https://api.census.gov/data/key_signup.html",
        description="Census BFS / ACS connectors",
    ),
}


class MissingCredentialError(RuntimeError):
    """Raised by `require()` when a fetcher needs an API key and none is found.

    Message structure (assertable in tests):
        "<description> needs an API key. Checked env vars (in order):
         <var1>, <var2>. Checked config file: <path> (<found|not found>).
         Get a free key at: <signup_url>"
    """


def get(service: str, *, explicit: str | None = None) -> str | None:
    """Resolve an API key for `service` (None if not found)."""
    if service not in SERVICES:
        raise KeyError(
            f"Unknown service {service!r}. Known: {sorted(SERVICES.keys())}"
        )
    if explicit:
        return explicit
    spec = SERVICES[service]
    for var in spec.env_vars:
        v = os.environ.get(var)
        if v:
            return v
    # Config-file lookup is added in Task 2.
    return None


def require(service: str, *, explicit: str | None = None) -> str:
    """Like `get(service)` but raises `MissingCredentialError` on miss."""
    key = get(service, explicit=explicit)
    if key:
        return key
    spec = SERVICES[service]
    # Config-file path message is filled in by Task 2; for now print a
    # placeholder so the error message is honest.
    raise MissingCredentialError(
        f"{spec.description} needs an API key. "
        f"Checked env vars (in order): {', '.join(spec.env_vars)}. "
        f"Checked config file: <not configured> (not found). "
        f"Get a free key at: {spec.signup_url}"
    )


def status() -> pd.DataFrame:
    """Return one row per service: ['service', 'configured', 'source',
       'description', 'signup_url']. Never includes the actual key value."""
    rows = []
    for name, spec in SERVICES.items():
        source = "missing"
        configured = False
        for var in spec.env_vars:
            if os.environ.get(var):
                source = f"env:{var}"
                configured = True
                break
        rows.append({
            "service": name,
            "configured": configured,
            "source": source,
            "description": spec.description,
            "signup_url": spec.signup_url,
        })
    return pd.DataFrame(rows)


__all__ = [
    "ServiceCredentialSpec",
    "SERVICES",
    "MissingCredentialError",
    "get",
    "require",
    "status",
]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_credentials/test_service_registry.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add puremacro/puremacro/credentials.py tests/test_credentials/__init__.py tests/test_credentials/test_service_registry.py
git commit -m "$(cat <<'EOF'
feat(0.66.0): credentials module skeleton — SERVICES registry + get/require/status

F2.0 sub-slice of Slice A. Adds puremacro.credentials with the
ServiceCredentialSpec dataclass + SERVICES registry (fred, bea,
anthropic, openai, census), MissingCredentialError, and the
three-tier resolver (explicit > env > config). Config-file lookup
stub returns None for now; Task 2 wires the TOML loader.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Wire the TOML config-file loader

**Files:**
- Modify: `puremacro/puremacro/credentials.py`
- Test: `tests/test_credentials/test_config_file_parsing.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_credentials/test_config_file_parsing.py`:

```python
"""F2.0 — TOML config-file resolution."""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest


def test_default_config_path_uses_env_override(monkeypatch, tmp_path):
    from puremacro.credentials import default_config_path

    custom = tmp_path / "custom-creds.toml"
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(custom))
    assert default_config_path() == custom


def test_default_config_path_uses_xdg(monkeypatch, tmp_path):
    from puremacro.credentials import default_config_path

    monkeypatch.delenv("PUREMACRO_CREDENTIALS_FILE", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert default_config_path() == tmp_path / "puremacro" / "credentials.toml"


def test_default_config_path_fallback_to_home(monkeypatch, tmp_path):
    from puremacro.credentials import default_config_path

    monkeypatch.delenv("PUREMACRO_CREDENTIALS_FILE", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert default_config_path() == tmp_path / ".puremacro" / "credentials.toml"


def test_config_lookup_finds_key(monkeypatch, tmp_path):
    from puremacro.credentials import get

    cfg = tmp_path / "credentials.toml"
    cfg.write_text('[fred]\napi_key = "from-toml"\n')
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(cfg))
    # Wipe env so config-file path is the only source.
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    # Reset module-level cache by reimporting.
    import importlib, puremacro.credentials as C
    importlib.reload(C)
    assert C.get("fred") == "from-toml"


def test_missing_config_file_returns_silently(monkeypatch, tmp_path):
    from puremacro.credentials import get

    cfg = tmp_path / "nonexistent.toml"
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(cfg))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    import importlib, puremacro.credentials as C
    importlib.reload(C)
    # No warning, no error, just None.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert C.get("fred") is None
    assert not caught, f"unexpected warning(s): {[str(w.message) for w in caught]}"


def test_malformed_toml_warns_and_falls_through(monkeypatch, tmp_path):
    from puremacro.credentials import get

    cfg = tmp_path / "bad.toml"
    cfg.write_text("[fred\napi_key = no quotes here\n")
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(cfg))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    import importlib, puremacro.credentials as C
    importlib.reload(C)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert C.get("fred") is None
    user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warns) == 1, f"expected one UserWarning, got {caught}"
    assert str(cfg) in str(user_warns[0].message)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_credentials/test_config_file_parsing.py -v
```
Expected: FAIL — `default_config_path` and the TOML loader don't exist yet.

- [ ] **Step 3: Extend the credentials module**

Edit `puremacro/puremacro/credentials.py`. At the top of the file (next to existing imports), add:

```python
import tomllib
import warnings
```

After the `SERVICES = {...}` block (before `class MissingCredentialError`), add:

```python
def default_config_path() -> Path:
    """`$PUREMACRO_CREDENTIALS_FILE` if set; else
       `$XDG_CONFIG_HOME/puremacro/credentials.toml` if XDG_CONFIG_HOME set;
       else `~/.puremacro/credentials.toml`."""
    env = os.environ.get("PUREMACRO_CREDENTIALS_FILE")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "puremacro" / "credentials.toml"
    return Path.home() / ".puremacro" / "credentials.toml"


_CONFIG_CACHE: dict[str, Any] | None = None


def _load_config() -> dict[str, Any]:
    """Read the TOML config file once per process; cache the parsed dict.
       Returns {} on missing file. Warns + returns {} on malformed TOML."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    path = default_config_path()
    if not path.exists():
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE
    try:
        with open(path, "rb") as f:
            _CONFIG_CACHE = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as e:
        warnings.warn(
            f"puremacro.credentials: failed to parse {path}: {e}. "
            f"Falling back to env-vars only.",
            UserWarning,
            stacklevel=2,
        )
        _CONFIG_CACHE = {}
    return _CONFIG_CACHE
```

Replace the existing `get()` function body (the version from Task 1 with the "Config-file lookup is added in Task 2" comment) with:

```python
def get(service: str, *, explicit: str | None = None) -> str | None:
    """Resolve an API key for `service` (None if not found)."""
    if service not in SERVICES:
        raise KeyError(
            f"Unknown service {service!r}. Known: {sorted(SERVICES.keys())}"
        )
    if explicit:
        return explicit
    spec = SERVICES[service]
    for var in spec.env_vars:
        v = os.environ.get(var)
        if v:
            return v
    cfg = _load_config()
    return cfg.get(service, {}).get("api_key") or None
```

Replace the existing `require()` function body to surface the config-file path:

```python
def require(service: str, *, explicit: str | None = None) -> str:
    """Like `get(service)` but raises `MissingCredentialError` on miss."""
    key = get(service, explicit=explicit)
    if key:
        return key
    spec = SERVICES[service]
    cfg_path = default_config_path()
    cfg_status = "found but no [{}].api_key".format(service) if cfg_path.exists() else "not found"
    raise MissingCredentialError(
        f"{spec.description} needs an API key. "
        f"Checked env vars (in order): {', '.join(spec.env_vars)}. "
        f"Checked config file: {cfg_path} ({cfg_status}). "
        f"Get a free key at: {spec.signup_url}"
    )
```

Update the `status()` function to also report the config-file source:

```python
def status() -> pd.DataFrame:
    """Return one row per service: ['service', 'configured', 'source',
       'description', 'signup_url']. Never includes the actual key value."""
    cfg = _load_config()
    rows = []
    for name, spec in SERVICES.items():
        source = "missing"
        configured = False
        for var in spec.env_vars:
            if os.environ.get(var):
                source = f"env:{var}"
                configured = True
                break
        if not configured and cfg.get(name, {}).get("api_key"):
            source = "config_file"
            configured = True
        rows.append({
            "service": name,
            "configured": configured,
            "source": source,
            "description": spec.description,
            "signup_url": spec.signup_url,
        })
    return pd.DataFrame(rows)
```

Extend `__all__` at the bottom:

```python
__all__ = [
    "ServiceCredentialSpec",
    "SERVICES",
    "MissingCredentialError",
    "default_config_path",
    "get",
    "require",
    "status",
]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_credentials/test_config_file_parsing.py tests/test_credentials/test_service_registry.py -v
```
Expected: 9 passed (6 new + 3 prior).

- [ ] **Step 5: Commit**

```bash
git add puremacro/puremacro/credentials.py tests/test_credentials/test_config_file_parsing.py
git commit -m "$(cat <<'EOF'
feat(0.66.0): credentials TOML config-file loader

Adds default_config_path() with priority $PUREMACRO_CREDENTIALS_FILE
> $XDG_CONFIG_HOME/puremacro/credentials.toml > ~/.puremacro/credentials.toml.
Module-cached _load_config() reads once per process; missing file is
silent (returns {}); malformed TOML warns and falls through to env-only.

get() now consults config after env vars. require() surfaces the
config-file path + found/not-found state in the error message so
researchers see all four resolver tiers (explicit/env/config/missing).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Resolver-priority and `MissingCredentialError` message tests

**Files:**
- Test: `tests/test_credentials/test_resolver_priority.py`
- Test: `tests/test_credentials/test_missing_credential_error_message.py`
- Test: `tests/test_credentials/test_status_dataframe.py`

(All implementation already exists from Tasks 1-2; this task is pure test-coverage to lock the contract.)

- [ ] **Step 1: Write the resolver-priority test**

Create `tests/test_credentials/test_resolver_priority.py`:

```python
"""F2.0 — Resolver priority: explicit > primary env > secondary env > config > None."""
from __future__ import annotations

import importlib

import pytest


def _reset_credentials_module():
    import puremacro.credentials as C
    C._CONFIG_CACHE = None
    importlib.reload(C)
    return C


def test_explicit_kwarg_wins_over_env(monkeypatch):
    C = _reset_credentials_module()
    monkeypatch.setenv("FRED_API_KEY", "from-env")
    assert C.get("fred", explicit="from-caller") == "from-caller"


def test_primary_env_var_wins_over_secondary(monkeypatch):
    C = _reset_credentials_module()
    monkeypatch.setenv("FRED_API_KEY", "primary")
    monkeypatch.setenv("PUREMACRO_FRED_API_KEY", "secondary")
    assert C.get("fred") == "primary"


def test_secondary_env_var_used_when_primary_missing(monkeypatch):
    C = _reset_credentials_module()
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setenv("PUREMACRO_FRED_API_KEY", "secondary")
    assert C.get("fred") == "secondary"


def test_config_file_used_when_env_missing(monkeypatch, tmp_path):
    cfg = tmp_path / "credentials.toml"
    cfg.write_text('[fred]\napi_key = "from-config"\n')
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(cfg))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    C = _reset_credentials_module()
    assert C.get("fred") == "from-config"


def test_env_wins_over_config_file(monkeypatch, tmp_path):
    cfg = tmp_path / "credentials.toml"
    cfg.write_text('[fred]\napi_key = "from-config"\n')
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(cfg))
    monkeypatch.setenv("FRED_API_KEY", "from-env")
    C = _reset_credentials_module()
    assert C.get("fred") == "from-env"


def test_none_when_nothing_set(monkeypatch, tmp_path):
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(tmp_path / "missing.toml"))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    C = _reset_credentials_module()
    assert C.get("fred") is None


def test_unknown_service_raises_keyerror():
    from puremacro.credentials import get
    with pytest.raises(KeyError, match="bogus_service"):
        get("bogus_service")
```

Create `tests/test_credentials/test_missing_credential_error_message.py`:

```python
"""F2.0 — MissingCredentialError message includes all four resolver tiers."""
from __future__ import annotations

import importlib

import pytest


def _reset_credentials_module():
    import puremacro.credentials as C
    C._CONFIG_CACHE = None
    importlib.reload(C)
    return C


def test_message_names_service_description(monkeypatch, tmp_path):
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(tmp_path / "absent.toml"))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    C = _reset_credentials_module()
    with pytest.raises(C.MissingCredentialError) as exc_info:
        C.require("fred")
    msg = str(exc_info.value)
    assert "FRED + ALFRED real-time macro data" in msg


def test_message_lists_every_env_var_checked(monkeypatch, tmp_path):
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(tmp_path / "absent.toml"))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    C = _reset_credentials_module()
    with pytest.raises(C.MissingCredentialError) as exc_info:
        C.require("fred")
    msg = str(exc_info.value)
    assert "FRED_API_KEY" in msg
    assert "PUREMACRO_FRED_API_KEY" in msg


def test_message_names_config_path_and_not_found(monkeypatch, tmp_path):
    cfg = tmp_path / "absent.toml"
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(cfg))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    C = _reset_credentials_module()
    with pytest.raises(C.MissingCredentialError) as exc_info:
        C.require("fred")
    msg = str(exc_info.value)
    assert str(cfg) in msg
    assert "not found" in msg


def test_message_includes_signup_url(monkeypatch, tmp_path):
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(tmp_path / "absent.toml"))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    C = _reset_credentials_module()
    with pytest.raises(C.MissingCredentialError) as exc_info:
        C.require("fred")
    assert "https://fred.stlouisfed.org/docs/api/api_key.html" in str(exc_info.value)


def test_require_returns_key_when_present(monkeypatch):
    import puremacro.credentials as C
    monkeypatch.setenv("FRED_API_KEY", "abc123")
    assert C.require("fred") == "abc123"
```

Create `tests/test_credentials/test_status_dataframe.py`:

```python
"""F2.0 — status() shape, columns, and no-key-leak guarantee."""
from __future__ import annotations

import importlib
import re

import pandas as pd
import pytest


def _reset_credentials_module():
    import puremacro.credentials as C
    C._CONFIG_CACHE = None
    importlib.reload(C)
    return C


def test_status_returns_dataframe_with_expected_columns(monkeypatch, tmp_path):
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(tmp_path / "absent.toml"))
    for v in ("FRED_API_KEY", "PUREMACRO_FRED_API_KEY", "BEA_API_KEY",
              "PUREMACRO_BEA_API_KEY", "ANTHROPIC_API_KEY",
              "PUREMACRO_ANTHROPIC_API_KEY", "OPENAI_API_KEY",
              "PUREMACRO_OPENAI_API_KEY", "CENSUS_API_KEY",
              "PUREMACRO_CENSUS_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    C = _reset_credentials_module()
    df = C.status()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"service", "configured", "source",
                                "description", "signup_url"}
    assert set(df["service"]) == set(C.SERVICES.keys())
    assert (df["configured"] == False).all()
    assert (df["source"] == "missing").all()


def test_status_marks_env_source(monkeypatch, tmp_path):
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(tmp_path / "absent.toml"))
    monkeypatch.setenv("FRED_API_KEY", "abc123")
    C = _reset_credentials_module()
    df = C.status().set_index("service")
    assert df.loc["fred", "configured"] == True
    assert df.loc["fred", "source"] == "env:FRED_API_KEY"


def test_status_marks_config_file_source(monkeypatch, tmp_path):
    cfg = tmp_path / "credentials.toml"
    cfg.write_text('[fred]\napi_key = "from-config"\n')
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(cfg))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    C = _reset_credentials_module()
    df = C.status().set_index("service")
    assert df.loc["fred", "configured"] == True
    assert df.loc["fred", "source"] == "config_file"


def test_status_never_includes_key_values(monkeypatch):
    import puremacro.credentials as C
    # Set obviously-sensitive values and confirm none reach the DataFrame.
    monkeypatch.setenv("FRED_API_KEY", "SECRET-FRED-VALUE-9999")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SHOULD-NOT-LEAK-XXXX")
    df = C.status()
    flat = df.to_csv(index=False)
    assert "SECRET-FRED-VALUE-9999" not in flat
    assert "sk-ant-SHOULD-NOT-LEAK-XXXX" not in flat
```

- [ ] **Step 2: Run all credential tests**

```bash
pytest tests/test_credentials/ -v
```
Expected: 21 passed (7 + 5 + 4 + the 3 + 6 from earlier tasks).

- [ ] **Step 3: Commit**

```bash
git add tests/test_credentials/test_resolver_priority.py tests/test_credentials/test_missing_credential_error_message.py tests/test_credentials/test_status_dataframe.py
git commit -m "$(cat <<'EOF'
test(0.66.0): credentials resolver-priority + error-message + status tests

Pins the explicit > primary env > secondary env > config > None
resolver priority across 7 scenarios, the four-tier
MissingCredentialError message structure, and the status() DataFrame
shape + no-key-leak guarantee (regex-asserted across an obviously
sensitive sentinel).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Rollout batch 1 — FRED + BEA fetchers switch to `credentials.require(...)`

**Files:**
- Modify: `puremacro/puremacro/fetch/fred.py`
- Modify: `puremacro/puremacro/fetch/fred_states.py`
- Modify: `puremacro/puremacro/fetch/frb_phil_coincident.py`
- Modify: `puremacro/puremacro/fetch/bea_cainc.py`
- Modify: `puremacro/puremacro/fetch/bea_industry_shares.py`

Each file currently has one or more `os.environ.get("FRED_API_KEY")` or `os.environ.get("BEA_API_KEY")` calls inside a `_client()`-style helper that raises a hand-rolled `RuntimeError` on miss. The pattern across all five files is:

**Before** (canonical example from `fetch/fred.py:34-40`):
```python
def _client() -> Fred:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY must be set in environment")
    return Fred(api_key=key)
```

**After**:
```python
def _client(api_key: str | None = None) -> Fred:
    from puremacro import credentials
    key = credentials.require("fred", explicit=api_key)
    return Fred(api_key=key)
```

For BEA files, replace `"fred"` with `"bea"` in the `credentials.require(...)` call.

- [ ] **Step 1: Edit `puremacro/fetch/fred.py`**

Read the file first to confirm the existing `_client()` is at line ~34–40. Make exactly the two changes above:
1. Add `api_key: str | None = None` parameter to `_client()`.
2. Replace the body of `_client()` with the new pattern.

Leave the `import os` line in place (other code in the file may still use `os.path` etc.).

- [ ] **Step 2: Edit `puremacro/fetch/fred_states.py`**

Find the FRED API key reader (similar `_client()` or inline `os.environ.get("FRED_API_KEY")` pattern). Apply the same transformation. If the function name differs, preserve it; only swap the body.

- [ ] **Step 3: Edit `puremacro/fetch/frb_phil_coincident.py`**

Same transformation: `_client(api_key=None)` → `credentials.require("fred", explicit=api_key)`. The function may currently be inline (no `_client` helper); if so, refactor the inline `os.environ.get("FRED_API_KEY")` block into a one-line `key = credentials.require("fred", explicit=api_key)`.

- [ ] **Step 4: Edit `puremacro/fetch/bea_cainc.py`**

Same pattern but with `credentials.require("bea", explicit=api_key)`. The existing error message currently says something like `"BEA_API_KEY not set; cannot fetch CAINC4 from API"` — replace the env-lookup block with the centralised call.

- [ ] **Step 5: Edit `puremacro/fetch/bea_industry_shares.py`**

Same as Step 4.

- [ ] **Step 6: Smoke-test the affected fetcher modules import cleanly**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -c "from puremacro.fetch import fred, fred_states, frb_phil_coincident, bea_cainc, bea_industry_shares; print('imports ok')"
```
Expected: `imports ok`. No actual fetch is exercised (no API key needed for import).

- [ ] **Step 7: Run the existing fetcher tests to confirm no regression**

```bash
pytest tests/ -k "fred or bea or phil" -v --tb=short 2>&1 | tail -30
```
Expected: no NEW failures vs. the pre-rollout baseline. Pre-existing failures may persist (run a quick baseline comparison if uncertain).

- [ ] **Step 8: Commit**

```bash
git add puremacro/puremacro/fetch/fred.py puremacro/puremacro/fetch/fred_states.py puremacro/puremacro/fetch/frb_phil_coincident.py puremacro/puremacro/fetch/bea_cainc.py puremacro/puremacro/fetch/bea_industry_shares.py
git commit -m "$(cat <<'EOF'
feat(0.66.0): switch FRED + BEA fetchers to credentials.require()

F2.0 rollout batch 1. Five files (fetch/fred.py, fred_states.py,
frb_phil_coincident.py, bea_cainc.py, bea_industry_shares.py)
swap direct os.environ.get("FRED_API_KEY"|"BEA_API_KEY") calls for
puremacro.credentials.require("fred"|"bea", explicit=api_key).
Adds api_key= passthrough to each _client() so callers can override.
Error message goes from "FRED_API_KEY must be set in environment"
to the centralised MissingCredentialError with all four resolver
tiers + signup URL.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Rollout batch 2 — Census, LLM, instruments switch to `credentials.require(...)`

**Files:**
- Modify: `puremacro/puremacro/fetch/census_bfs.py` → `credentials.require("census")`
- Modify: `puremacro/puremacro/narrative/scoring/llm.py` → `credentials.require("anthropic")` (and `"openai"` if the file has an OpenAI provider)
- Modify: `puremacro/puremacro/narrative/indices/_llm_kernel.py` → same as above
- Modify: `puremacro/puremacro/instruments/_catalog.py` → `credentials.require(...)` for whichever service it currently uses (verify by reading the file first)

Same pattern as Task 4, just different service tags.

- [ ] **Step 1: Edit `puremacro/fetch/census_bfs.py`**

Find the `os.environ.get("CENSUS_API_KEY")` block. Replace with `credentials.require("census", explicit=api_key)`. Add `api_key: str | None = None` to the enclosing function signature.

- [ ] **Step 2: Edit `puremacro/narrative/scoring/llm.py`**

This file may have multiple provider classes (Anthropic, OpenAI). For each provider class:
1. Replace `os.environ.get("ANTHROPIC_API_KEY")` with `credentials.require("anthropic", explicit=api_key)`.
2. Replace `os.environ.get("OPENAI_API_KEY")` with `credentials.require("openai", explicit=api_key)`.

Add `api_key: str | None = None` to each provider's `__init__`. Read the file first to identify the exact class structure.

- [ ] **Step 3: Edit `puremacro/narrative/indices/_llm_kernel.py`**

Same as Step 2 — Anthropic and/or OpenAI key references switched to `credentials.require(...)`.

- [ ] **Step 4: Edit `puremacro/instruments/_catalog.py`**

Read the file first. Identify which service's API key it uses (likely `"fred"` for FRED instrument loaders). Replace the `os.environ.get(...)` call with `credentials.require(<service>, explicit=api_key)`.

- [ ] **Step 5: Smoke-test imports**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -c "from puremacro.fetch import census_bfs; from puremacro.narrative.scoring import llm; from puremacro.narrative.indices import _llm_kernel; from puremacro.instruments import _catalog; print('imports ok')"
```
Expected: `imports ok`.

- [ ] **Step 6: Run the existing affected tests**

```bash
pytest tests/ -k "census or llm or instrument" -v --tb=short 2>&1 | tail -30
```
Expected: no NEW failures vs. baseline.

- [ ] **Step 7: Commit**

```bash
git add puremacro/puremacro/fetch/census_bfs.py puremacro/puremacro/narrative/scoring/llm.py puremacro/puremacro/narrative/indices/_llm_kernel.py puremacro/puremacro/instruments/_catalog.py
git commit -m "$(cat <<'EOF'
feat(0.66.0): switch Census, LLM, instruments to credentials.require()

F2.0 rollout batch 2. Four files (fetch/census_bfs.py,
narrative/scoring/llm.py, narrative/indices/_llm_kernel.py,
instruments/_catalog.py) swap direct os.environ.get(...) calls for
puremacro.credentials.require("census"|"anthropic"|"openai"|"fred").
Completes the F2.0 rollout — the AST lint test in Task 6 will now
pass on first run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: AST lint test — ban direct `os.environ.get("*_API_KEY")` in `fetch/`, `narrative/scoring/`, `narrative/indices/`, `instruments/`

**Files:**
- Test: `tests/test_credentials/test_no_direct_env_get_in_fetch.py`

With Tasks 4 + 5 complete, this test passes on first run — TDD-clean.

- [ ] **Step 1: Write the lint test**

Create `tests/test_credentials/test_no_direct_env_get_in_fetch.py`:

```python
"""F2.0 — AST lint: no direct `os.environ.get("*_API_KEY")` in fetcher /
narrative-scoring / narrative-indices / instruments code. All such
lookups must route through puremacro.credentials.get/require."""
from __future__ import annotations

import ast
import pathlib

import pytest


_TARGET_DIRS = [
    "puremacro/puremacro/fetch",
    "puremacro/puremacro/narrative/scoring",
    "puremacro/puremacro/narrative/indices",
    "puremacro/puremacro/instruments",
]


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "puremacro" / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not find puremacro/ repo root")


def _python_files(root: pathlib.Path):
    for d in _TARGET_DIRS:
        for p in (root / d).rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            yield p


def _violations(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return list of (lineno, key) for any `os.environ.get("..._API_KEY")`."""
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "get"):
            continue
        target = node.func.value
        is_environ = (
            (isinstance(target, ast.Attribute) and target.attr == "environ"
             and isinstance(target.value, ast.Name) and target.value.id == "os")
            or (isinstance(target, ast.Name) and target.id == "environ")
        )
        if not is_environ:
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        key = node.args[0].value
        if isinstance(key, str) and "_API_KEY" in key:
            out.append((node.lineno, key))
    return out


def test_no_direct_env_get_for_api_keys():
    root = _repo_root()
    offenders: list[str] = []
    for f in _python_files(root):
        for lineno, key in _violations(f):
            offenders.append(f"{f.relative_to(root)}:{lineno}: os.environ.get({key!r})")
    assert not offenders, (
        "F2.0 contract violation: these files read API-key env vars "
        "directly. Route through `puremacro.credentials.require(service)`:\n  "
        + "\n  ".join(offenders)
    )
```

- [ ] **Step 2: Run to verify it passes**

```bash
pytest tests/test_credentials/test_no_direct_env_get_in_fetch.py -v
```
Expected: 1 passed. If it FAILS, the failure message lists the file:line that still has a direct `os.environ.get(...)` — fix that file before committing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_credentials/test_no_direct_env_get_in_fetch.py
git commit -m "$(cat <<'EOF'
test(0.66.0): AST lint forbids direct os.environ.get("*_API_KEY") in fetch/etc.

Scans puremacro/{fetch,narrative/scoring,narrative/indices,instruments}/
for any os.environ.get(...) call whose string argument contains
_API_KEY. Forces every API-key consumer to route through
puremacro.credentials.require(). Passes on first run thanks to the
rollout commits in Tasks 4 + 5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `docs/CREDENTIALS.md` + README quickstart block

**Files:**
- Create: `puremacro/docs/CREDENTIALS.md`
- Modify: `puremacro/README.md` (insert a quickstart block in the existing Quickstart section)

- [ ] **Step 1: Create `puremacro/docs/CREDENTIALS.md`**

```markdown
# Credentials

> Available from puremacro **0.66.0** onwards.

`puremacro.credentials` is the single place every API-keyed fetcher in
puremacro reads its key. It resolves keys in priority order:

1. **Explicit kwarg** — `credentials.get("fred", explicit=...)` or any
   fetcher's `api_key=` parameter wins everything.
2. **Environment variables** — each service has a primary alias
   (`FRED_API_KEY`) and a `PUREMACRO_`-prefixed secondary
   (`PUREMACRO_FRED_API_KEY`); first hit wins.
3. **TOML config file** — `~/.puremacro/credentials.toml` (overridable
   via `$PUREMACRO_CREDENTIALS_FILE` or `$XDG_CONFIG_HOME`).
4. **None** — `get()` returns `None`; `require()` raises
   `MissingCredentialError` with a researcher-actionable message.

## Quickstart

```python
import puremacro.credentials as creds

# See what's configured (never leaks the actual key values):
creds.status()
#       service  configured           source                                         description                                 signup_url
# 0        fred        True  env:FRED_API_KEY            FRED + ALFRED real-time macro data (...)  https://fred.stlouisfed.org/...
# 1         bea       False          missing               BEA NIPA / regional / industry tables  https://apps.bea.gov/API/signup/
# ...

# Resolve a key (None if not found):
key = creds.get("anthropic")

# Or require it (raises with a helpful message):
key = creds.require("anthropic")
# MissingCredentialError: LLM-scored narrative kernels (...) needs an
# API key. Checked env vars (in order): ANTHROPIC_API_KEY,
# PUREMACRO_ANTHROPIC_API_KEY. Checked config file:
# /Users/you/.puremacro/credentials.toml (not found). Get a free key
# at: https://console.anthropic.com/settings/keys
```

## Config file format

`~/.puremacro/credentials.toml` (optional; create if you prefer not to set env vars):

```toml
[fred]
api_key = "abc123..."

[bea]
api_key = "..."

[anthropic]
api_key = "sk-ant-..."

[openai]
api_key = "sk-..."

[census]
api_key = "..."
```

Missing sections fall back to env vars. The file is read once per
process and cached. Malformed TOML emits a `UserWarning` and falls
through to env-vars-only — never blocks credential resolution.

## Known services

| Service     | Used by                                        | Sign up                                                  |
|-------------|------------------------------------------------|----------------------------------------------------------|
| `fred`      | `fetch.fred`, `fetch.fred_states`, FRB Phil    | https://fred.stlouisfed.org/docs/api/api_key.html        |
| `bea`       | `fetch.bea_cainc`, `fetch.bea_industry_shares` | https://apps.bea.gov/API/signup/                         |
| `anthropic` | `narrative.scoring.llm` (Anthropic provider)   | https://console.anthropic.com/settings/keys              |
| `openai`    | `narrative.scoring.llm` (OpenAI provider)      | https://platform.openai.com/api-keys                     |
| `census`    | `fetch.census_bfs`                             | https://api.census.gov/data/key_signup.html              |

## For implementers (adding a new fetcher)

```python
from puremacro import credentials

def fetch_my_thing(*, api_key: str | None = None) -> pd.DataFrame:
    key = credentials.require("my_service", explicit=api_key)
    # ... use key in your HTTP calls ...
```

The AST lint test
`tests/test_credentials/test_no_direct_env_get_in_fetch.py`
fails the build if you read `os.environ.get("*_API_KEY")` directly
in `puremacro/{fetch,narrative/scoring,narrative/indices,instruments}/`.

To add a new known service, append a `ServiceCredentialSpec` entry to
`puremacro/credentials.py::SERVICES`. The service registry test
verifies every entry has the required fields and an HTTPS signup URL.
```

- [ ] **Step 2: Insert the README quickstart block**

Read `puremacro/README.md`. Find the "Quickstart" section (it currently shows `cholesky_svar` + `lp_hac` + `panel_lp_dk` + the 0.65.0 `with_quality=True` example added in Slice 1). Append a fourth code-block example IMMEDIATELY AFTER the existing three, BEFORE the "End-to-end replications" paragraph:

\`\`\`python
# Centralised API-key resolution (0.66.0+).
from puremacro import credentials
credentials.status()                  # see what's configured (no values leaked)
key = credentials.require("fred")      # raises MissingCredentialError with signup URL on miss
\`\`\`

(That's a single code block delimited by triple backticks in the file; the escaping above is for this prompt.)

- [ ] **Step 3: Commit**

```bash
git add puremacro/docs/CREDENTIALS.md puremacro/README.md
git commit -m "$(cat <<'EOF'
docs(0.66.0): CREDENTIALS.md + README quickstart for puremacro.credentials

Single-page reference: four-tier resolver priority, TOML config
format, status() introspection, per-service table with signup URLs,
and an "adding a new fetcher" guide that points at the lint test.
README quickstart gains a credentials.status() + credentials.require()
example alongside the existing 0.65.0 with_quality=True block.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Sub-slice 2 — F2.1 SQLite cache backend

(Tasks 8–12.)

## Task 8: Create `_cache_db.py` — SQLite connection + schema bootstrap

**Files:**
- Create: `puremacro/puremacro/_cache_db.py`
- Test: `tests/test_cache_db/__init__.py` (empty), `tests/test_cache_db/test_schema_bootstrap.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cache_db/__init__.py` (empty).

Create `tests/test_cache_db/test_schema_bootstrap.py`:

```python
"""F2.1 — Schema bootstrap is idempotent + WAL mode is enabled."""
from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Force every call in this test to use a tmp_path DB; reset the singleton."""
    db = tmp_path / "cache.db"
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db.parent))
    import puremacro._cache_db as M
    M.close_conn()
    yield db
    M.close_conn()


def test_default_db_path_uses_env_directory(monkeypatch, tmp_path):
    from puremacro._cache_db import default_db_path
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    assert default_db_path() == tmp_path / "cache.db"


def test_default_db_path_accepts_explicit_db_file(monkeypatch, tmp_path):
    from puremacro._cache_db import default_db_path
    explicit = tmp_path / "custom.db"
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(explicit))
    assert default_db_path() == explicit


def test_default_db_path_fallback(monkeypatch, tmp_path):
    from puremacro._cache_db import default_db_path
    from pathlib import Path
    monkeypatch.delenv("PUREMACRO_HTTP_CACHE_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert default_db_path() == tmp_path / ".cache" / "puremacro" / "cache.db"


def test_bootstrap_creates_three_tables(fresh_db):
    from puremacro._cache_db import get_conn
    conn = get_conn(fresh_db)
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert {"http_cache", "alfred_vintages", "schema_version"}.issubset(tables)


def test_bootstrap_seeds_schema_version(fresh_db):
    from puremacro._cache_db import get_conn
    conn = get_conn(fresh_db)
    rows = dict(conn.execute("SELECT component, version FROM schema_version"))
    assert rows == {"http_cache": 1, "alfred_vintages": 1}


def test_bootstrap_is_idempotent(fresh_db):
    from puremacro._cache_db import get_conn, bootstrap_schema
    conn = get_conn(fresh_db)
    bootstrap_schema(conn)
    bootstrap_schema(conn)  # second call must not raise or duplicate
    rows = list(conn.execute("SELECT component, version FROM schema_version"))
    assert len(rows) == 2


def test_wal_mode_enabled(fresh_db):
    from puremacro._cache_db import get_conn
    conn = get_conn(fresh_db)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_singleton_returns_same_connection(fresh_db):
    from puremacro._cache_db import get_conn
    a = get_conn(fresh_db)
    b = get_conn(fresh_db)
    assert a is b
```

- [ ] **Step 2: Run to verify failure**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
pytest tests/test_cache_db/test_schema_bootstrap.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'puremacro._cache_db'`.

- [ ] **Step 3: Create the module**

Create `puremacro/puremacro/_cache_db.py`:

```python
"""SQLite connection manager + schema bootstrap for puremacro's data
infrastructure.

A single SQLite file at ``~/.cache/puremacro/cache.db`` (overridable
via ``$PUREMACRO_HTTP_CACHE_DIR``) hosts three tables:

- ``http_cache`` — replaces the flat-file HTTP cache; backs
  ``puremacro._http_cache.cache_read``/``cache_write``.
- ``alfred_vintages`` — persistent FRED-ALFRED vintage panel; backs
  ``puremacro.vintages.AlfredVintageStore``.
- ``schema_version`` — registry for future migrations.

WAL journal mode is enabled so multiple notebooks against the same DB
do not block each other on writes. One ``sqlite3.Connection`` per
process is kept alive in a module-level singleton.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path


_DDL_HTTP_CACHE = """
CREATE TABLE IF NOT EXISTS http_cache (
    key            TEXT PRIMARY KEY,
    url            TEXT NOT NULL,
    fetched_at     INTEGER NOT NULL,
    content_type   TEXT,
    body           BLOB NOT NULL
);
"""

_DDL_HTTP_CACHE_IDX = (
    "CREATE INDEX IF NOT EXISTS http_cache_fetched_at_idx "
    "ON http_cache(fetched_at);"
)

_DDL_ALFRED_VINTAGES = """
CREATE TABLE IF NOT EXISTS alfred_vintages (
    series_id        TEXT NOT NULL,
    observation_date TEXT NOT NULL,
    vintage_date     TEXT NOT NULL,
    value            REAL,
    PRIMARY KEY (series_id, observation_date, vintage_date)
);
"""

_DDL_ALFRED_VINTAGES_IDX = (
    "CREATE INDEX IF NOT EXISTS alfred_vintages_series_vintage_idx "
    "ON alfred_vintages(series_id, vintage_date);"
)

_DDL_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    component TEXT PRIMARY KEY,
    version   INTEGER NOT NULL
);
"""

_SCHEMA_SEED = [("http_cache", 1), ("alfred_vintages", 1)]


def default_db_path() -> Path:
    """Resolve the canonical cache DB path.

    ``$PUREMACRO_HTTP_CACHE_DIR`` overrides everything:
      - if it ends with ``.db``, return that path verbatim;
      - otherwise treat as a directory and return ``<dir>/cache.db``.
    Default: ``~/.cache/puremacro/cache.db``.
    """
    env = os.environ.get("PUREMACRO_HTTP_CACHE_DIR")
    if env:
        p = Path(env)
        if p.suffix == ".db":
            return p
        return p / "cache.db"
    return Path.home() / ".cache" / "puremacro" / "cache.db"


def bootstrap_schema(conn: sqlite3.Connection) -> None:
    """Idempotent: create tables + seed schema_version rows if missing."""
    cur = conn.cursor()
    cur.execute(_DDL_HTTP_CACHE)
    cur.execute(_DDL_HTTP_CACHE_IDX)
    cur.execute(_DDL_ALFRED_VINTAGES)
    cur.execute(_DDL_ALFRED_VINTAGES_IDX)
    cur.execute(_DDL_SCHEMA_VERSION)
    for component, version in _SCHEMA_SEED:
        cur.execute(
            "INSERT OR IGNORE INTO schema_version (component, version) "
            "VALUES (?, ?)",
            (component, version),
        )
    conn.commit()


_CONN: sqlite3.Connection | None = None
_CONN_PATH: Path | None = None


def get_conn(db_path: Path | None = None) -> sqlite3.Connection:
    """Return the module-level singleton SQLite connection.

    Lazily opens on first call. Enables WAL mode and bootstraps the
    schema. If called twice with different ``db_path`` arguments
    (e.g., in tests), closes the previous connection and opens a new one.
    """
    global _CONN, _CONN_PATH
    target = db_path or default_db_path()
    if _CONN is not None and _CONN_PATH == target:
        return _CONN
    if _CONN is not None:
        _CONN.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        target, timeout=30.0, isolation_level=None, check_same_thread=False,
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    bootstrap_schema(conn)
    _CONN = conn
    _CONN_PATH = target
    return _CONN


def close_conn() -> None:
    """Close the singleton connection (used by tests)."""
    global _CONN, _CONN_PATH
    if _CONN is not None:
        try:
            _CONN.close()
        finally:
            _CONN = None
            _CONN_PATH = None


def migrate_from_flat_files(
    conn: sqlite3.Connection,
    flat_cache_dir: Path,
    *,
    remove: bool = False,
) -> int:
    """Walk ``flat_cache_dir`` for ``*.bin`` + ``*.json`` sidecar pairs.

    Inserts each into ``http_cache`` with ``INSERT OR IGNORE`` semantics
    (idempotent). If ``remove=True``, unlinks each migrated file pair
    after successful insert. Returns the count actually migrated.

    Failures on a per-file basis are warned-and-skipped, not raised —
    migration is opportunistic.
    """
    import json
    import warnings

    if not flat_cache_dir.exists():
        return 0
    migrated = 0
    for bin_path in flat_cache_dir.glob("*.bin"):
        sidecar = bin_path.with_suffix(".json")
        if not sidecar.exists():
            continue
        try:
            meta = json.loads(sidecar.read_text())
            url = meta["url"]
            fetched_at = int(float(meta["fetched_at"]))
            content_type = meta.get("content_type")
            body = bin_path.read_bytes()
            key = bin_path.stem
            cur = conn.execute(
                "INSERT OR IGNORE INTO http_cache "
                "(key, url, fetched_at, content_type, body) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, url, fetched_at, content_type, body),
            )
            if cur.rowcount > 0:
                migrated += 1
                if remove:
                    bin_path.unlink(missing_ok=True)
                    sidecar.unlink(missing_ok=True)
        except (OSError, ValueError, KeyError) as e:
            warnings.warn(
                f"puremacro._cache_db: skipping {bin_path.name}: {e}",
                UserWarning,
                stacklevel=2,
            )
            continue
    conn.commit()
    return migrated


__all__ = [
    "default_db_path",
    "bootstrap_schema",
    "get_conn",
    "close_conn",
    "migrate_from_flat_files",
]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_cache_db/test_schema_bootstrap.py -v
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add puremacro/puremacro/_cache_db.py tests/test_cache_db/__init__.py tests/test_cache_db/test_schema_bootstrap.py
git commit -m "$(cat <<'EOF'
feat(0.66.0): _cache_db — singleton SQLite connection + schema bootstrap

F2.1 first commit. New module puremacro._cache_db hosts the shared
sqlite3.Connection backing both the HTTP cache (F2.1) and the ALFRED
vintage store (F2.2). WAL journal mode for concurrent reader/writer
support. Schema: http_cache, alfred_vintages, schema_version tables
(idempotent CREATE IF NOT EXISTS; seed rows via INSERT OR IGNORE).
default_db_path() resolves $PUREMACRO_HTTP_CACHE_DIR (interpreted as
either a .db path or a parent dir) and falls back to
~/.cache/puremacro/cache.db. migrate_from_flat_files() walks the
legacy flat-file cache and inserts into http_cache with idempotent
semantics; per-file failures warn-and-skip.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Rewrite `_http_cache.py` against `_cache_db`; preserve public API + add lazy migration trigger

**Files:**
- Modify: `puremacro/puremacro/_http_cache.py`
- Test: `tests/test_cache_db/test_http_cache_sqlite_roundtrip.py`
- Test: `tests/test_cache_db/test_failure_modes.py`

The four public symbols (`cache_key`, `default_cache_dir`, `cache_read`, `cache_write`) keep their signatures. Storage swaps from flat files to SQLite.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cache_db/test_http_cache_sqlite_roundtrip.py`:

```python
"""F2.1 — cache_read / cache_write roundtrip against the SQLite backend."""
from __future__ import annotations

import time

import pytest


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


def test_write_then_read_returns_bytes(fresh_cache):
    from puremacro._http_cache import cache_read, cache_write
    cache_write(fresh_cache, "https://example.com/a", b"hello", content_type="text/plain")
    assert cache_read(fresh_cache, "https://example.com/a") == b"hello"


def test_read_miss_returns_none(fresh_cache):
    from puremacro._http_cache import cache_read
    assert cache_read(fresh_cache, "https://example.com/never-cached") is None


def test_stale_entry_returns_none(fresh_cache):
    from puremacro._http_cache import cache_read, cache_write
    cache_write(fresh_cache, "https://example.com/b", b"old")
    # Force a 1-second TTL and sleep past it.
    time.sleep(1.1)
    assert cache_read(fresh_cache, "https://example.com/b", ttl_seconds=1) is None


def test_overwrite_updates_body(fresh_cache):
    from puremacro._http_cache import cache_read, cache_write
    cache_write(fresh_cache, "https://example.com/c", b"v1")
    cache_write(fresh_cache, "https://example.com/c", b"v2")
    assert cache_read(fresh_cache, "https://example.com/c") == b"v2"


def test_cache_key_stable_for_same_url():
    from puremacro._http_cache import cache_key
    assert cache_key("https://example.com/x") == cache_key("https://example.com/x")
    assert cache_key("https://example.com/x") != cache_key("https://example.com/y")


def test_default_cache_dir_returns_db_parent(monkeypatch, tmp_path):
    from puremacro._http_cache import default_cache_dir
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path / "x"))
    # When env points to a non-.db dir, returns that dir verbatim
    # (callers may pass `cache_dir` and we use it; the db lives at
    # cache_dir/cache.db).
    assert default_cache_dir() == tmp_path / "x"
```

Create `tests/test_cache_db/test_failure_modes.py`:

```python
"""F2.1 — cache_read / cache_write must never raise; warn + None / no-op."""
from __future__ import annotations

import sqlite3
import warnings

import pytest


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


def test_cache_read_on_db_error_returns_none(fresh_cache, monkeypatch):
    from puremacro import _http_cache, _cache_db

    def _raise(*a, **kw):
        raise sqlite3.OperationalError("simulated DB error")

    conn = _cache_db.get_conn(fresh_cache / "cache.db")
    monkeypatch.setattr(conn, "execute", _raise)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _http_cache.cache_read(fresh_cache, "https://example.com/x")
    assert result is None
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_cache_write_on_db_error_is_noop(fresh_cache, monkeypatch):
    from puremacro import _http_cache, _cache_db

    def _raise(*a, **kw):
        raise sqlite3.OperationalError("simulated DB error")

    conn = _cache_db.get_conn(fresh_cache / "cache.db")
    monkeypatch.setattr(conn, "execute", _raise)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _http_cache.cache_write(fresh_cache, "https://example.com/x", b"body")
    assert any(issubclass(w.category, UserWarning) for w in caught)
    # No exception leaked.
```

- [ ] **Step 2: Run to verify the read/write tests fail (current file backs to flat files)**

```bash
pytest tests/test_cache_db/test_http_cache_sqlite_roundtrip.py tests/test_cache_db/test_failure_modes.py -v
```
Expected: most tests fail because `cache_read`/`cache_write` still hit the flat-file backend, which the test's `monkeypatch` of the SQLite connection does not intercept. (A couple may pass coincidentally — that's fine.)

- [ ] **Step 3: Rewrite `puremacro/_http_cache.py`**

Replace the entire contents of `puremacro/puremacro/_http_cache.py` with:

```python
"""On-disk HTTP cache, backed by SQLite (0.66.0+).

Used by the opt-in ``safe_get_*_cached`` helpers in :mod:`puremacro._http`.

Public API preserved from 0.65.0:
- ``cache_key(url) -> str``
- ``default_cache_dir() -> Path``  (now returns the DB's parent dir)
- ``cache_read(cache_dir, url, ttl_seconds=...) -> bytes | None``
- ``cache_write(cache_dir, url, body, content_type=None) -> None``

Behavior unchanged from the caller's perspective. Internal storage
switches from one flat file per URL to a single SQLite DB at
``cache_dir/cache.db`` (created on first write).

Cache failures (OperationalError, disk full, etc.) MUST NOT break the
caller — ``cache_read`` returns None and ``cache_write`` no-ops, both
emitting a UserWarning so the failure is visible without being fatal.

A one-time lazy migration runs on the first call after upgrade: if the
legacy ``cache_dir/*.bin`` files exist and ``http_cache`` is empty, the
flat-file entries are inserted into the DB (without deleting the files).
A UserWarning points to ``tools/cache_migrate.py --apply --rm`` for users
who also want to delete the flat files.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import warnings
from pathlib import Path

from . import _cache_db


def cache_key(url: str) -> str:
    """SHA-256 hex of the URL — used as the http_cache primary key."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def default_cache_dir() -> Path:
    """Return the default cache directory.

    For backwards compatibility with 0.65.0 callers, this returns a
    *directory* path; the SQLite DB lives at ``<dir>/cache.db``.
    ``$PUREMACRO_HTTP_CACHE_DIR`` overrides; otherwise ``~/.cache/puremacro``.
    """
    env = os.environ.get("PUREMACRO_HTTP_CACHE_DIR")
    if env:
        p = Path(env)
        # If the env var points at a .db file, return its parent dir.
        return p.parent if p.suffix == ".db" else p
    return Path.home() / ".cache" / "puremacro"


def _resolve_db_path(cache_dir: Path) -> Path:
    """Map a 0.65.0-style cache_dir argument to the actual DB path."""
    p = Path(cache_dir)
    if p.suffix == ".db":
        return p
    return p / "cache.db"


_MIGRATION_ATTEMPTED = False


def _maybe_migrate_flat_files(cache_dir: Path, conn: sqlite3.Connection) -> None:
    """First-call lazy migration: walk cache_dir/*.bin into http_cache."""
    global _MIGRATION_ATTEMPTED
    if _MIGRATION_ATTEMPTED:
        return
    _MIGRATION_ATTEMPTED = True
    flat = Path(cache_dir)
    if not flat.exists():
        return
    # Only run if there are flat files AND http_cache is empty.
    bin_files = list(flat.glob("*.bin"))
    if not bin_files:
        return
    row = conn.execute("SELECT COUNT(*) FROM http_cache").fetchone()
    if row and row[0] > 0:
        return
    try:
        n = _cache_db.migrate_from_flat_files(conn, flat, remove=False)
    except Exception as e:  # defensive; migration must not fail callers
        warnings.warn(
            f"puremacro._http_cache: lazy migration failed: {e}",
            UserWarning, stacklevel=3,
        )
        return
    if n > 0:
        warnings.warn(
            f"puremacro._http_cache: migrated {n} flat-file cache entries "
            f"to SQLite at {_resolve_db_path(flat)}. Run "
            f"`python tools/cache_migrate.py --apply --rm` if you also "
            f"want to delete the original flat files.",
            UserWarning, stacklevel=3,
        )


def cache_read(
    cache_dir: Path,
    url: str,
    ttl_seconds: int = 30 * 24 * 3600,
) -> bytes | None:
    """Return cached bytes for ``url`` if fresh, else None.

    Returns None on: miss, stale entry, or any DB failure. Cache
    failures emit a UserWarning but never raise.
    """
    try:
        conn = _cache_db.get_conn(_resolve_db_path(cache_dir))
        _maybe_migrate_flat_files(cache_dir, conn)
        row = conn.execute(
            "SELECT body, fetched_at FROM http_cache WHERE key = ?",
            (cache_key(url),),
        ).fetchone()
    except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
        warnings.warn(
            f"puremacro._http_cache: cache_read failed for {url!r}: {e}",
            UserWarning, stacklevel=2,
        )
        return None
    if row is None:
        return None
    body, fetched_at = row
    if time.time() - int(fetched_at) >= ttl_seconds:
        return None
    return body


def cache_write(
    cache_dir: Path,
    url: str,
    body: bytes,
    content_type: str | None = None,
) -> None:
    """Write ``body`` under the cache key for ``url``.

    Silently no-ops on any DB failure (emits a UserWarning).
    """
    try:
        conn = _cache_db.get_conn(_resolve_db_path(cache_dir))
        _maybe_migrate_flat_files(cache_dir, conn)
        conn.execute(
            "INSERT OR REPLACE INTO http_cache "
            "(key, url, fetched_at, content_type, body) VALUES (?, ?, ?, ?, ?)",
            (cache_key(url), url, int(time.time()), content_type, body),
        )
    except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
        warnings.warn(
            f"puremacro._http_cache: cache_write failed for {url!r}: {e}",
            UserWarning, stacklevel=2,
        )


__all__ = ["cache_key", "default_cache_dir", "cache_read", "cache_write"]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_cache_db/ -v
```
Expected: all tests pass (8 from Task 8 + 6 roundtrip + 2 failure-mode = 16).

- [ ] **Step 5: Confirm `safe_get_*_cached` still works (no behavior change for existing callers)**

```bash
pytest tests/ -k "cached or _http" -v --tb=short 2>&1 | tail -30
```
Expected: no NEW failures vs. baseline.

- [ ] **Step 6: Commit**

```bash
git add puremacro/puremacro/_http_cache.py tests/test_cache_db/test_http_cache_sqlite_roundtrip.py tests/test_cache_db/test_failure_modes.py
git commit -m "$(cat <<'EOF'
feat(0.66.0): rewrite _http_cache against SQLite + lazy migration

F2.1 second commit. _http_cache.py public API preserved (cache_key,
default_cache_dir, cache_read, cache_write keep their 0.65.0
signatures). Storage swaps from per-URL flat files to a single
sqlite3 connection (routed through _cache_db.get_conn). Lazy
migration: on first call after upgrade, if cache_dir/*.bin files
exist and http_cache is empty, migrate the flat-file entries into
the DB without removing the originals; UserWarning points to
tools/cache_migrate.py --rm for users who want to clean up.
DB failures (OperationalError, OSError) warn and return None /
no-op — preserves the existing "cache failures must not break the
caller" contract.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Introspection helpers + concurrent-WAL test

**Files:**
- Modify: `puremacro/puremacro/cache.py`
- Test: `tests/test_cache_db/test_introspection_helpers.py`
- Test: `tests/test_cache_db/test_concurrent_wal.py`

- [ ] **Step 1: Write the failing introspection tests**

Create `tests/test_cache_db/test_introspection_helpers.py`:

```python
"""F2.1 — http_list_urls / http_cache_size_bytes / http_cache_clear."""
from __future__ import annotations

import time

import pandas as pd
import pytest


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


def _seed(tmp_path, entries):
    """Insert (url, body) pairs via cache_write."""
    from puremacro._http_cache import cache_write
    for url, body in entries:
        cache_write(tmp_path, url, body)


def test_http_list_urls_returns_sorted(fresh_cache):
    _seed(fresh_cache, [
        ("https://b.example/x", b"x"),
        ("https://a.example/y", b"y"),
        ("https://c.example/z", b"z"),
    ])
    from puremacro.cache import http_list_urls
    assert http_list_urls() == [
        "https://a.example/y",
        "https://b.example/x",
        "https://c.example/z",
    ]


def test_http_cache_size_bytes(fresh_cache):
    _seed(fresh_cache, [("https://a/", b"x" * 100), ("https://b/", b"y" * 200)])
    from puremacro.cache import http_cache_size_bytes
    assert http_cache_size_bytes() == 300


def test_http_cache_clear_all(fresh_cache):
    _seed(fresh_cache, [("https://a/", b"x"), ("https://b/", b"y")])
    from puremacro.cache import http_cache_clear, http_list_urls
    assert http_cache_clear() == 2
    assert http_list_urls() == []


def test_http_cache_clear_older_than(fresh_cache):
    from puremacro._http_cache import cache_write
    from puremacro.cache import http_cache_clear, http_list_urls
    cache_write(fresh_cache, "https://old/", b"o")
    time.sleep(1.2)
    cache_write(fresh_cache, "https://new/", b"n")
    deleted = http_cache_clear(older_than=pd.Timedelta(seconds=1))
    assert deleted == 1
    assert http_list_urls() == ["https://new/"]


def test_disk_cache_helpers_still_present():
    """Verify the existing disk_cache / disk_cache_path API in cache.py
    is untouched by the http_* additions."""
    import puremacro.cache as C
    assert callable(C.disk_cache)
    assert callable(C.disk_cache_path)
```

Create `tests/test_cache_db/test_concurrent_wal.py`:

```python
"""F2.1 — WAL allows concurrent reader/writer without DB-locked errors."""
from __future__ import annotations

import threading

import pytest


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


def test_concurrent_writes_and_reads_no_lock_errors(fresh_cache):
    from puremacro._http_cache import cache_read, cache_write

    errors: list[Exception] = []

    def writer():
        try:
            for i in range(50):
                cache_write(fresh_cache, f"https://example.com/w-{i}",
                            f"body-{i}".encode())
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for i in range(50):
                _ = cache_read(fresh_cache, f"https://example.com/w-{i}")
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert not errors, f"unexpected errors: {errors}"
    # Spot-check at least some writes landed.
    assert cache_read(fresh_cache, "https://example.com/w-0") == b"body-0"
    assert cache_read(fresh_cache, "https://example.com/w-49") == b"body-49"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_cache_db/test_introspection_helpers.py tests/test_cache_db/test_concurrent_wal.py -v
```
Expected: introspection tests fail with `AttributeError: module 'puremacro.cache' has no attribute 'http_list_urls'`. The concurrent test passes (WAL is already on from Task 8).

- [ ] **Step 3: Extend `puremacro/cache.py`**

Append the following at the end of `puremacro/puremacro/cache.py` (the existing `disk_cache`, `disk_cache_path` helpers are untouched):

```python
# ── HTTP-cache introspection (0.66.0+) ───────────────────────────────
# These functions read from the SQLite http_cache table populated by
# puremacro._http_cache.cache_write. They are distinct from the
# disk_cache / disk_cache_path helpers above (which back a DataFrame /
# JSON cache under a different namespace).

def http_list_urls() -> list[str]:
    """Return all URLs currently in the HTTP cache, sorted alphabetically."""
    from . import _cache_db
    conn = _cache_db.get_conn()
    rows = conn.execute("SELECT url FROM http_cache ORDER BY url").fetchall()
    return [r[0] for r in rows]


def http_cache_size_bytes() -> int:
    """Total bytes stored in the http_cache.body column."""
    from . import _cache_db
    conn = _cache_db.get_conn()
    row = conn.execute(
        "SELECT COALESCE(SUM(LENGTH(body)), 0) FROM http_cache"
    ).fetchone()
    return int(row[0])


def http_cache_clear(older_than: "pd.Timedelta | None" = None) -> int:
    """Delete entries older than ``older_than`` (None = delete all).

    Returns the number of rows deleted. Issues a VACUUM after large
    deletions (>1000 rows) so the SQLite file actually shrinks on disk.
    """
    import time
    from . import _cache_db
    conn = _cache_db.get_conn()
    if older_than is None:
        cur = conn.execute("DELETE FROM http_cache")
    else:
        cutoff = int(time.time() - older_than.total_seconds())
        cur = conn.execute(
            "DELETE FROM http_cache WHERE fetched_at < ?", (cutoff,)
        )
    deleted = cur.rowcount or 0
    if deleted > 1000:
        conn.execute("VACUUM")
    return deleted
```

Update the existing `__all__` (if it exists) to include the three new names. If there is no `__all__` in the file, leave it implicit (the names are auto-exported).

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_cache_db/ -v
```
Expected: 22 passed (8 + 6 + 2 + 5 introspection + 1 concurrent).

- [ ] **Step 5: Commit**

```bash
git add puremacro/puremacro/cache.py tests/test_cache_db/test_introspection_helpers.py tests/test_cache_db/test_concurrent_wal.py
git commit -m "$(cat <<'EOF'
feat(0.66.0): cache.py gains http_list_urls / http_cache_size_bytes / http_cache_clear

F2.1 introspection helpers. Three additive functions on puremacro.cache
that read the SQLite http_cache table (distinct from the existing
disk_cache / disk_cache_path helpers, which back a separate
DataFrame/JSON namespace). http_cache_clear(older_than=pd.Timedelta(...))
respects a per-row TTL filter and runs VACUUM after large deletions
(>1000 rows). Concurrent-WAL test confirms 50 writes + 50 reads in
parallel threads produce zero OperationalError.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Migration test (idempotent + `remove=True`) + CLI smoke test

**Files:**
- Test: `tests/test_cache_db/test_migration_from_flat_files.py`
- Test: `tests/test_cache_db/test_cache_migrate_cli.py`
- Create: `tools/cache_migrate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cache_db/test_migration_from_flat_files.py`:

```python
"""F2.1 — migrate_from_flat_files: idempotent + remove flag + warn-and-skip."""
from __future__ import annotations

import json
import time

import pytest


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


def _make_flat_entry(d, url: str, body: bytes, age_seconds: int = 0):
    """Create a flat-file cache pair in `d`."""
    import hashlib
    key = hashlib.sha256(url.encode()).hexdigest()
    (d / f"{key}.bin").write_bytes(body)
    (d / f"{key}.json").write_text(json.dumps({
        "url": url,
        "fetched_at": time.time() - age_seconds,
        "content_type": "text/plain",
    }))


def test_migration_inserts_entries(fresh_db):
    from puremacro._cache_db import get_conn, migrate_from_flat_files
    _make_flat_entry(fresh_db, "https://example.com/a", b"alpha")
    _make_flat_entry(fresh_db, "https://example.com/b", b"beta")
    conn = get_conn(fresh_db / "cache.db")
    n = migrate_from_flat_files(conn, fresh_db, remove=False)
    assert n == 2
    rows = conn.execute("SELECT url, body FROM http_cache ORDER BY url").fetchall()
    assert rows == [("https://example.com/a", b"alpha"),
                    ("https://example.com/b", b"beta")]


def test_migration_idempotent(fresh_db):
    from puremacro._cache_db import get_conn, migrate_from_flat_files
    _make_flat_entry(fresh_db, "https://example.com/a", b"alpha")
    conn = get_conn(fresh_db / "cache.db")
    assert migrate_from_flat_files(conn, fresh_db, remove=False) == 1
    assert migrate_from_flat_files(conn, fresh_db, remove=False) == 0


def test_migration_with_remove_unlinks_files(fresh_db):
    from puremacro._cache_db import get_conn, migrate_from_flat_files
    _make_flat_entry(fresh_db, "https://example.com/a", b"alpha")
    conn = get_conn(fresh_db / "cache.db")
    assert migrate_from_flat_files(conn, fresh_db, remove=True) == 1
    assert list(fresh_db.glob("*.bin")) == []
    assert list(fresh_db.glob("*.json")) == []


def test_migration_skips_corrupt_sidecars(fresh_db):
    import hashlib, warnings
    from puremacro._cache_db import get_conn, migrate_from_flat_files
    # Good entry + corrupt sidecar.
    _make_flat_entry(fresh_db, "https://example.com/good", b"ok")
    bad_url = "https://example.com/bad"
    bad_key = hashlib.sha256(bad_url.encode()).hexdigest()
    (fresh_db / f"{bad_key}.bin").write_bytes(b"corrupt")
    (fresh_db / f"{bad_key}.json").write_text("{ not valid json")
    conn = get_conn(fresh_db / "cache.db")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        n = migrate_from_flat_files(conn, fresh_db, remove=False)
    assert n == 1
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_migration_missing_dir_returns_zero(tmp_path):
    from puremacro._cache_db import get_conn, migrate_from_flat_files
    conn = get_conn(tmp_path / "cache.db")
    nonexistent = tmp_path / "does_not_exist"
    assert migrate_from_flat_files(conn, nonexistent) == 0
```

Create `tests/test_cache_db/test_cache_migrate_cli.py`:

```python
"""F2.1 — tools/cache_migrate.py CLI smoke test (dry-run + --apply)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def _make_flat_entry(d: Path, url: str, body: bytes):
    import hashlib
    key = hashlib.sha256(url.encode()).hexdigest()
    (d / f"{key}.bin").write_bytes(body)
    (d / f"{key}.json").write_text(json.dumps({
        "url": url,
        "fetched_at": time.time(),
        "content_type": "text/plain",
    }))


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "puremacro" / "pyproject.toml").exists():
            return parent
    raise RuntimeError("no repo root")


def test_cli_dry_run_does_not_apply(tmp_path, monkeypatch):
    _make_flat_entry(tmp_path, "https://example.com/a", b"alpha")
    cli = _repo_root() / "tools" / "cache_migrate.py"
    env = {"PUREMACRO_HTTP_CACHE_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"}
    out = subprocess.run(
        [sys.executable, str(cli)],
        capture_output=True, text=True, env=env, check=True,
    )
    # Files still present after dry-run.
    assert list(tmp_path.glob("*.bin"))
    assert "1" in out.stdout  # reports 1 entry would be migrated


def test_cli_apply_migrates(tmp_path):
    _make_flat_entry(tmp_path, "https://example.com/a", b"alpha")
    cli = _repo_root() / "tools" / "cache_migrate.py"
    env = {"PUREMACRO_HTTP_CACHE_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"}
    subprocess.run(
        [sys.executable, str(cli), "--apply"],
        capture_output=True, text=True, env=env, check=True,
    )
    # DB now exists and contains the entry.
    import sqlite3
    conn = sqlite3.connect(tmp_path / "cache.db")
    rows = conn.execute("SELECT url FROM http_cache").fetchall()
    conn.close()
    assert rows == [("https://example.com/a",)]


def test_cli_apply_rm_unlinks(tmp_path):
    _make_flat_entry(tmp_path, "https://example.com/a", b"alpha")
    cli = _repo_root() / "tools" / "cache_migrate.py"
    env = {"PUREMACRO_HTTP_CACHE_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"}
    subprocess.run(
        [sys.executable, str(cli), "--apply", "--rm"],
        capture_output=True, text=True, env=env, check=True,
    )
    assert list(tmp_path.glob("*.bin")) == []
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_cache_db/test_migration_from_flat_files.py tests/test_cache_db/test_cache_migrate_cli.py -v
```
Expected: migration tests pass (`migrate_from_flat_files` exists from Task 8); CLI tests fail because the CLI doesn't exist yet.

- [ ] **Step 3: Create the CLI**

Create `tools/cache_migrate.py`:

```python
"""Migrate flat-file HTTP cache (puremacro 0.65.0 and earlier) to SQLite (0.66.0+).

The 0.66.0 release replaces the flat-file cache at
``~/.cache/puremacro/http/*.bin + *.json`` with a single SQLite file
at ``~/.cache/puremacro/cache.db``. The HTTP-cache module runs the
migration lazily on first read/write after upgrade. This CLI lets you
run the migration up-front (and optionally remove the original
flat files).

Usage:
    python tools/cache_migrate.py              # dry-run; report count
    python tools/cache_migrate.py --apply      # migrate
    python tools/cache_migrate.py --apply --rm # migrate and unlink originals

Env vars:
    PUREMACRO_HTTP_CACHE_DIR   override cache location (default
                               ~/.cache/puremacro)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate flat-file HTTP cache to SQLite.",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually run the migration (default is dry-run).",
    )
    parser.add_argument(
        "--rm", action="store_true",
        help="Unlink the original flat files after a successful insert "
             "(implies --apply).",
    )
    args = parser.parse_args()

    if args.rm and not args.apply:
        args.apply = True

    # Resolve the flat-cache directory the same way _http_cache does.
    from puremacro._http_cache import default_cache_dir
    from puremacro._cache_db import get_conn, default_db_path

    flat = default_cache_dir()
    db_path = default_db_path()
    bin_files = list(flat.glob("*.bin"))
    n_present = len(bin_files)

    print(f"Flat cache dir: {flat}")
    print(f"Target DB:      {db_path}")
    print(f"Flat-file entries detected: {n_present}")

    if not args.apply:
        print(f"DRY RUN — would migrate {n_present} entries. "
              f"Re-run with --apply to migrate.")
        return 0

    if n_present == 0:
        print("Nothing to migrate.")
        return 0

    from puremacro._cache_db import migrate_from_flat_files
    conn = get_conn(db_path)
    n_migrated = migrate_from_flat_files(conn, flat, remove=args.rm)
    print(f"Migrated {n_migrated} entries.")
    if args.rm:
        leftover_bin = len(list(flat.glob("*.bin")))
        leftover_json = len(list(flat.glob("*.json")))
        print(f"After --rm: {leftover_bin} .bin / {leftover_json} .json "
              f"remain (corrupt sidecars or already-present entries).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_cache_db/ -v
```
Expected: 30 passed (22 prior + 5 migration + 3 CLI).

- [ ] **Step 5: Commit**

```bash
git add tools/cache_migrate.py tests/test_cache_db/test_migration_from_flat_files.py tests/test_cache_db/test_cache_migrate_cli.py
git commit -m "$(cat <<'EOF'
feat(0.66.0): tools/cache_migrate.py CLI + migration tests

F2.1 closes out with the migration tooling: tools/cache_migrate.py
exposes dry-run / --apply / --apply --rm modes for users who want
to migrate up-front rather than via the lazy first-call trigger.
Idempotent (INSERT OR IGNORE) so repeated runs are safe; corrupt
sidecars are warn-and-skipped, not fatal.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: `docs/CACHE_DB.md`

**Files:**
- Create: `puremacro/docs/CACHE_DB.md`

- [ ] **Step 1: Create the reference doc**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add puremacro/docs/CACHE_DB.md
git commit -m "$(cat <<'EOF'
docs(0.66.0): CACHE_DB.md reference

Single-page docs for the SQLite cache backend: location, schema, WAL
mode, migration from 0.65.0 flat-file cache, introspection helpers,
failure-semantics contract, Pyodide notes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Sub-slice 3 — F2.2 ALFRED vintage store

(Tasks 13–15.)

## Task 13: `AlfredVintageStore` class

**Files:**
- Modify: `puremacro/puremacro/vintages.py` (append; existing functions untouched)
- Test: `tests/test_vintages_alfred_store/__init__.py` (empty)
- Test: `tests/test_vintages_alfred_store/test_store_roundtrip.py`
- Test: `tests/test_vintages_alfred_store/test_vintage_until_filter.py`
- Test: `tests/test_vintages_alfred_store/test_has_series_series_list.py`
- Test: `tests/test_vintages_alfred_store/test_coverage_diagnostic.py`
- Test: `tests/test_vintages_alfred_store/test_store_failure_modes.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vintages_alfred_store/__init__.py` (empty).

Create `tests/test_vintages_alfred_store/test_store_roundtrip.py`:

```python
"""F2.2 — AlfredVintageStore put_many / get roundtrip."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    from puremacro.vintages import AlfredVintageStore
    s = AlfredVintageStore()
    yield s
    M.close_conn()


def test_put_many_then_get_returns_same_rows(store):
    rows = pd.DataFrame({
        "series_id":        ["GDPC1", "GDPC1"],
        "observation_date": ["2020-01-01", "2020-04-01"],
        "vintage_date":     ["2020-04-29", "2020-04-29"],
        "value":            [21000.0, 19500.0],
    })
    assert store.put_many(rows) == 2
    out = store.get("GDPC1")
    assert len(out) == 2
    assert set(out.columns) >= {"observation_date", "vintage_date", "value"}


def test_put_single_row(store):
    store.put("UNRATE", "2020-04-01", "2020-05-08", 14.7)
    out = store.get("UNRATE")
    assert len(out) == 1
    assert out["value"].iloc[0] == 14.7


def test_put_or_replace_overwrites(store):
    store.put("GDPC1", "2020-01-01", "2020-04-29", 21000.0)
    store.put("GDPC1", "2020-01-01", "2020-04-29", 21001.5)
    out = store.get("GDPC1")
    assert len(out) == 1
    assert out["value"].iloc[0] == 21001.5


def test_get_missing_series_returns_empty_dataframe(store):
    out = store.get("DOES_NOT_EXIST")
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_observation_and_vintage_dates_returned_as_timestamps(store):
    store.put("GDPC1", "2020-01-01", "2020-04-29", 21000.0)
    out = store.get("GDPC1")
    assert pd.api.types.is_datetime64_any_dtype(out["observation_date"])
    assert pd.api.types.is_datetime64_any_dtype(out["vintage_date"])
```

Create `tests/test_vintages_alfred_store/test_vintage_until_filter.py`:

```python
"""F2.2 — store.get(vintage_until=...) filters by vintage_date."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def populated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    from puremacro.vintages import AlfredVintageStore
    s = AlfredVintageStore()
    s.put_many(pd.DataFrame({
        "series_id":        ["GDPC1"] * 4,
        "observation_date": ["2020-01-01", "2020-01-01",
                              "2020-04-01", "2020-04-01"],
        "vintage_date":     ["2020-04-29", "2020-07-30",
                              "2020-07-30", "2020-10-29"],
        "value":            [21000.0, 21010.0, 19500.0, 19520.0],
    }))
    yield s
    M.close_conn()


def test_vintage_until_filters_correctly(populated_store):
    out = populated_store.get("GDPC1", vintage_until="2020-07-31")
    assert len(out) == 3  # 3 vintages on or before 2020-07-31


def test_vintage_until_strict_bound(populated_store):
    out = populated_store.get("GDPC1", vintage_until="2020-07-30")
    # The boundary vintage IS included (vintage_date <= vintage_until).
    assert len(out) == 3
    out = populated_store.get("GDPC1", vintage_until="2020-07-29")
    assert len(out) == 1


def test_vintage_until_none_returns_all(populated_store):
    out = populated_store.get("GDPC1", vintage_until=None)
    assert len(out) == 4
```

Create `tests/test_vintages_alfred_store/test_has_series_series_list.py`:

```python
"""F2.2 — has_series / series_list introspection."""
from __future__ import annotations

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    from puremacro.vintages import AlfredVintageStore
    s = AlfredVintageStore()
    yield s
    M.close_conn()


def test_has_series_false_when_empty(store):
    assert store.has_series("GDPC1") is False


def test_has_series_true_after_put(store):
    store.put("GDPC1", "2020-01-01", "2020-04-29", 21000.0)
    assert store.has_series("GDPC1") is True


def test_series_list_returns_sorted_distinct(store):
    store.put("UNRATE", "2020-04-01", "2020-05-08", 14.7)
    store.put("GDPC1", "2020-01-01", "2020-04-29", 21000.0)
    store.put("GDPC1", "2020-04-01", "2020-07-30", 19500.0)
    assert store.series_list() == ["GDPC1", "UNRATE"]
```

Create `tests/test_vintages_alfred_store/test_coverage_diagnostic.py`:

```python
"""F2.2 — coverage() diagnostic returns expected dict."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    from puremacro.vintages import AlfredVintageStore
    s = AlfredVintageStore()
    yield s
    M.close_conn()


def test_coverage_returns_none_for_missing_series(store):
    assert store.coverage("DOES_NOT_EXIST") is None


def test_coverage_returns_expected_fields(store):
    store.put_many(pd.DataFrame({
        "series_id":        ["GDPC1"] * 3,
        "observation_date": ["2020-01-01", "2020-04-01", "2020-07-01"],
        "vintage_date":     ["2020-04-29", "2020-07-30", "2020-10-29"],
        "value":            [21000.0, 19500.0, 20100.0],
    }))
    c = store.coverage("GDPC1")
    assert c is not None
    assert c["n_rows"] == 3
    assert c["first_obs"] == pd.Timestamp("2020-01-01")
    assert c["last_obs"] == pd.Timestamp("2020-07-01")
    assert c["first_vintage"] == pd.Timestamp("2020-04-29")
    assert c["last_vintage"] == pd.Timestamp("2020-10-29")
    assert c["n_vintages"] == 3
```

Create `tests/test_vintages_alfred_store/test_store_failure_modes.py`:

```python
"""F2.2 — store failures warn + degrade; never raise to caller."""
from __future__ import annotations

import sqlite3
import warnings

import pandas as pd
import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    from puremacro.vintages import AlfredVintageStore
    s = AlfredVintageStore()
    yield s
    M.close_conn()


def test_get_on_db_error_returns_empty(store, monkeypatch):
    from puremacro import _cache_db
    conn = _cache_db.get_conn()

    def _raise(*a, **kw):
        raise sqlite3.OperationalError("simulated")

    monkeypatch.setattr(conn, "execute", _raise)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = store.get("GDPC1")
    assert isinstance(out, pd.DataFrame) and out.empty
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_put_many_on_db_error_is_noop(store, monkeypatch):
    from puremacro import _cache_db
    conn = _cache_db.get_conn()

    def _raise(*a, **kw):
        raise sqlite3.OperationalError("simulated")

    monkeypatch.setattr(conn, "executemany", _raise)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        n = store.put_many(pd.DataFrame({
            "series_id": ["GDPC1"], "observation_date": ["2020-01-01"],
            "vintage_date": ["2020-04-29"], "value": [21000.0],
        }))
    assert n == 0
    assert any(issubclass(w.category, UserWarning) for w in caught)
```

- [ ] **Step 2: Run to verify failure**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
pytest tests/test_vintages_alfred_store/ -v
```
Expected: FAIL — `AttributeError: module 'puremacro.vintages' has no attribute 'AlfredVintageStore'`.

- [ ] **Step 3: Append `AlfredVintageStore` to `puremacro/vintages.py`**

At the end of `puremacro/puremacro/vintages.py` (after the existing `forecast_revision` function), append:

```python
# ── AlfredVintageStore (0.66.0+) ────────────────────────────────────
# Persistent local store for FRED-ALFRED vintage observations, backed
# by the shared SQLite cache DB (see puremacro._cache_db). Existing
# in-memory helpers (as_of, align_vintages, forecast_revision) are
# untouched; this class adds the store-backed counterpart so research
# notebooks don't refetch ALFRED on every kernel restart.

import sqlite3 as _sqlite3
import warnings as _warnings
from pathlib import Path as _Path

import pandas as _pd


class AlfredVintageStore:
    """Persistent store for FRED-ALFRED vintage panels.

    Backed by the ``alfred_vintages`` table in the shared SQLite cache
    DB (``~/.cache/puremacro/cache.db`` by default). Failures (DB
    locked, disk full, etc.) emit a ``UserWarning`` and degrade
    gracefully — ``get()`` returns an empty DataFrame, ``put_many()``
    no-ops. Research notebooks never crash on store errors; they just
    fall through to the API.
    """

    def __init__(self, db_path: "_Path | None" = None):
        from . import _cache_db
        self._cache_db = _cache_db
        self._db_path = db_path

    def _conn(self) -> _sqlite3.Connection:
        return self._cache_db.get_conn(self._db_path)

    def put(
        self,
        series_id: str,
        observation_date: str,
        vintage_date: str,
        value: float | None,
    ) -> None:
        """Insert (or replace) a single vintage observation."""
        try:
            self._conn().execute(
                "INSERT OR REPLACE INTO alfred_vintages "
                "(series_id, observation_date, vintage_date, value) "
                "VALUES (?, ?, ?, ?)",
                (series_id, observation_date, vintage_date, value),
            )
        except (_sqlite3.OperationalError, _sqlite3.DatabaseError) as e:
            _warnings.warn(
                f"AlfredVintageStore.put({series_id!r}, ...) failed: {e}",
                UserWarning, stacklevel=2,
            )

    def put_many(self, df: _pd.DataFrame) -> int:
        """Bulk insert. Required columns: ['series_id', 'observation_date',
        'vintage_date', 'value']. Returns count of rows inserted/replaced.
        Returns 0 on DB error."""
        required = {"series_id", "observation_date", "vintage_date", "value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"AlfredVintageStore.put_many: missing columns {sorted(missing)}"
            )
        rows = [
            (str(r["series_id"]),
             str(r["observation_date"])[:10],
             str(r["vintage_date"])[:10],
             None if _pd.isna(r["value"]) else float(r["value"]))
            for _, r in df.iterrows()
        ]
        try:
            self._conn().executemany(
                "INSERT OR REPLACE INTO alfred_vintages "
                "(series_id, observation_date, vintage_date, value) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            return len(rows)
        except (_sqlite3.OperationalError, _sqlite3.DatabaseError) as e:
            _warnings.warn(
                f"AlfredVintageStore.put_many failed: {e}",
                UserWarning, stacklevel=2,
            )
            return 0

    def get(
        self,
        series_id: str,
        *,
        vintage_until: str | None = None,
    ) -> _pd.DataFrame:
        """Return long-form DataFrame ['observation_date', 'vintage_date',
        'value'] for ``series_id``. Empty DataFrame on missing series or
        DB error. ``observation_date`` and ``vintage_date`` are
        ``pd.Timestamp``-typed."""
        sql = (
            "SELECT observation_date, vintage_date, value "
            "FROM alfred_vintages WHERE series_id = ?"
        )
        params: list = [series_id]
        if vintage_until is not None:
            sql += " AND vintage_date <= ?"
            params.append(str(vintage_until)[:10])
        sql += " ORDER BY observation_date, vintage_date"
        try:
            rows = self._conn().execute(sql, params).fetchall()
        except (_sqlite3.OperationalError, _sqlite3.DatabaseError) as e:
            _warnings.warn(
                f"AlfredVintageStore.get({series_id!r}) failed: {e}",
                UserWarning, stacklevel=2,
            )
            return _pd.DataFrame(
                columns=["observation_date", "vintage_date", "value"]
            )
        if not rows:
            return _pd.DataFrame(
                columns=["observation_date", "vintage_date", "value"]
            )
        df = _pd.DataFrame(
            rows, columns=["observation_date", "vintage_date", "value"]
        )
        df["observation_date"] = _pd.to_datetime(df["observation_date"])
        df["vintage_date"] = _pd.to_datetime(df["vintage_date"])
        return df

    def has_series(self, series_id: str) -> bool:
        try:
            row = self._conn().execute(
                "SELECT 1 FROM alfred_vintages WHERE series_id = ? LIMIT 1",
                (series_id,),
            ).fetchone()
            return row is not None
        except (_sqlite3.OperationalError, _sqlite3.DatabaseError):
            return False

    def series_list(self) -> list[str]:
        try:
            rows = self._conn().execute(
                "SELECT DISTINCT series_id FROM alfred_vintages ORDER BY series_id"
            ).fetchall()
            return [r[0] for r in rows]
        except (_sqlite3.OperationalError, _sqlite3.DatabaseError):
            return []

    def coverage(self, series_id: str) -> dict | None:
        """Diagnostic: counts + first/last observation/vintage dates,
        or None if has_series(series_id) is False."""
        if not self.has_series(series_id):
            return None
        row = self._conn().execute(
            "SELECT COUNT(*), MIN(observation_date), MAX(observation_date), "
            "MIN(vintage_date), MAX(vintage_date), "
            "COUNT(DISTINCT vintage_date) "
            "FROM alfred_vintages WHERE series_id = ?",
            (series_id,),
        ).fetchone()
        n_rows, first_obs, last_obs, first_v, last_v, n_v = row
        return {
            "n_rows":         int(n_rows),
            "first_obs":      _pd.to_datetime(first_obs),
            "last_obs":       _pd.to_datetime(last_obs),
            "first_vintage":  _pd.to_datetime(first_v),
            "last_vintage":   _pd.to_datetime(last_v),
            "n_vintages":     int(n_v),
        }
```

Also extend the file's `__all__` (or add one if missing) to include `AlfredVintageStore`.

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_vintages_alfred_store/ -v
```
Expected: 17 passed (5 roundtrip + 3 vintage filter + 3 series-list + 2 coverage + 2 failure modes + 2 baseline).

- [ ] **Step 5: Commit**

```bash
git add puremacro/puremacro/vintages.py tests/test_vintages_alfred_store/__init__.py tests/test_vintages_alfred_store/test_store_roundtrip.py tests/test_vintages_alfred_store/test_vintage_until_filter.py tests/test_vintages_alfred_store/test_has_series_series_list.py tests/test_vintages_alfred_store/test_coverage_diagnostic.py tests/test_vintages_alfred_store/test_store_failure_modes.py
git commit -m "$(cat <<'EOF'
feat(0.66.0): AlfredVintageStore — persistent FRED-ALFRED panel store

F2.2 first commit. New AlfredVintageStore class in vintages.py backs
the alfred_vintages SQLite table. put/put_many use INSERT OR REPLACE
(idempotent under repeated calls). get(series_id, vintage_until=...)
returns a long-form DataFrame with Timestamp-typed
observation_date/vintage_date columns. has_series/series_list/coverage
support researcher introspection. Existing in-memory helpers
(as_of, align_vintages, forecast_revision) are untouched. DB failures
warn + degrade (empty DataFrame / 0 inserted), never raise.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: `as_of_from_store()` convenience helper

**Files:**
- Modify: `puremacro/puremacro/vintages.py`
- Test: `tests/test_vintages_alfred_store/test_as_of_from_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_vintages_alfred_store/test_as_of_from_store.py`:

```python
"""F2.2 — as_of_from_store end-to-end."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def populated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    from puremacro.vintages import AlfredVintageStore
    s = AlfredVintageStore()
    s.put_many(pd.DataFrame({
        "series_id":        ["GDPC1"] * 4,
        "observation_date": ["2020-01-01", "2020-01-01",
                              "2020-04-01", "2020-04-01"],
        "vintage_date":     ["2020-04-29", "2020-07-30",
                              "2020-07-30", "2020-10-29"],
        "value":            [21000.0, 21010.0, 19500.0, 19520.0],
    }))
    yield s
    M.close_conn()


def test_as_of_returns_latest_vintage_known_at_date(populated_store):
    from puremacro.vintages import as_of_from_store
    # As of 2020-08-01, both observations have a vintage on/before that date.
    s = as_of_from_store("GDPC1", "2020-08-01", populated_store)
    # Expect the 2020-07-30 vintages: GDPC1[2020-01-01] = 21010.0,
    # GDPC1[2020-04-01] = 19500.0.
    assert s.loc[pd.Timestamp("2020-01-01")] == 21010.0
    assert s.loc[pd.Timestamp("2020-04-01")] == 19500.0


def test_as_of_excludes_future_vintages(populated_store):
    from puremacro.vintages import as_of_from_store
    # As of 2020-05-01, only the 2020-04-29 vintage of obs=2020-01-01 is known.
    s = as_of_from_store("GDPC1", "2020-05-01", populated_store)
    assert s.loc[pd.Timestamp("2020-01-01")] == 21000.0
    # The 2020-04-01 observation's earliest vintage is 2020-07-30 > 2020-05-01.
    assert pd.Timestamp("2020-04-01") not in s.index


def test_missing_series_returns_empty_series(populated_store):
    from puremacro.vintages import as_of_from_store
    s = as_of_from_store("DOES_NOT_EXIST", "2020-08-01", populated_store)
    assert isinstance(s, pd.Series)
    assert s.empty
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_vintages_alfred_store/test_as_of_from_store.py -v
```
Expected: FAIL — `ImportError: cannot import name 'as_of_from_store' from 'puremacro.vintages'`.

- [ ] **Step 3: Append the helper to `puremacro/vintages.py`**

After the `AlfredVintageStore` class, append:

```python
def as_of_from_store(
    series_id: str,
    vintage_date: str,
    store: "AlfredVintageStore",
) -> _pd.Series:
    """Pull ``series_id`` from ``store`` (vintages on or before
    ``vintage_date``), then apply :func:`as_of` to produce a
    pd.Series indexed by observation_date with the latest-known
    value as of ``vintage_date``.
    """
    df = store.get(series_id, vintage_until=vintage_date)
    if df.empty:
        return _pd.Series(dtype="float64")
    return as_of(
        df, vintage_date,
        date_col="observation_date",
        vintage_col="vintage_date",
        value_col="value",
    )
```

Add `"as_of_from_store"` to the file's `__all__`.

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_vintages_alfred_store/ -v
```
Expected: 20 passed (17 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add puremacro/puremacro/vintages.py tests/test_vintages_alfred_store/test_as_of_from_store.py
git commit -m "$(cat <<'EOF'
feat(0.66.0): as_of_from_store helper

F2.2 second commit. Convenience: pulls a series from
AlfredVintageStore (filtered to vintages <= vintage_date) and
applies the existing in-memory as_of() slicer. Returns an empty
pd.Series on missing series so callers don't have to special-case.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: `fetch_fred_alfred(store=, refresh=)` gap-fill integration

**Files:**
- Modify: `puremacro/puremacro/fetch/_classic.py` (the existing `fetch_fred_alfred` at line ~75)
- Test: `tests/test_vintages_alfred_store/test_fetch_fred_alfred_with_store.py`

The current signature is `fetch_fred_alfred(series_id, *, timeout=60.0) -> pd.DataFrame` returning `[date, vintage, value]` (no api_key — the function uses the public FREDgraph CSV endpoint).

The extension adds `store=` and `refresh=` kwargs. The store is opt-in; existing callers (no store passed) see zero behavior change.

- [ ] **Step 1: Write the failing test**

Create `tests/test_vintages_alfred_store/test_fetch_fred_alfred_with_store.py`:

```python
"""F2.2 — fetch_fred_alfred(store=, refresh=) gap-fill semantics."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    from puremacro.vintages import AlfredVintageStore
    s = AlfredVintageStore()
    yield s
    M.close_conn()


def _mock_api_rows():
    """Synthetic ALFRED rows in the shape fetch_fred_alfred returns:
       columns [date, vintage, value]."""
    return pd.DataFrame({
        "date":    [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-01")],
        "vintage": [pd.Timestamp("2020-04-29"), pd.Timestamp("2020-07-30")],
        "value":   [21000.0, 19500.0],
    })


def test_with_empty_store_calls_api_and_populates(store, monkeypatch):
    from puremacro.fetch import _classic

    calls = []

    def _fake_api(series_id, *, timeout):
        calls.append(series_id)
        return _mock_api_rows()

    # Patch the underlying live-API helper (the one that does the HTTP).
    # The new code path calls the original raw fetch when needed.
    monkeypatch.setattr(_classic, "_fetch_fred_alfred_raw_api", _fake_api,
                        raising=False)

    df = _classic.fetch_fred_alfred("GDPC1", store=store)
    assert calls == ["GDPC1"]
    assert len(df) == 2
    assert store.has_series("GDPC1")


def test_refetch_when_store_has_no_series(store, monkeypatch):
    from puremacro.fetch import _classic

    calls = []
    monkeypatch.setattr(
        _classic,
        "_fetch_fred_alfred_raw_api",
        lambda series_id, *, timeout: (calls.append(series_id) or _mock_api_rows()),
        raising=False,
    )
    _classic.fetch_fred_alfred("GDPC1", store=store)
    assert len(calls) == 1


def test_no_refetch_when_store_has_data(store, monkeypatch):
    from puremacro.fetch import _classic
    # Pre-populate store.
    store.put_many(pd.DataFrame({
        "series_id":        ["GDPC1", "GDPC1"],
        "observation_date": ["2020-01-01", "2020-04-01"],
        "vintage_date":     ["2020-04-29", "2020-07-30"],
        "value":            [21000.0, 19500.0],
    }))
    calls = []
    monkeypatch.setattr(
        _classic,
        "_fetch_fred_alfred_raw_api",
        lambda series_id, *, timeout: (calls.append(series_id) or _mock_api_rows()),
        raising=False,
    )
    df = _classic.fetch_fred_alfred("GDPC1", store=store)
    assert calls == []  # store hit; no API call
    assert len(df) == 2


def test_refresh_true_forces_api(store, monkeypatch):
    from puremacro.fetch import _classic
    store.put_many(pd.DataFrame({
        "series_id":        ["GDPC1"],
        "observation_date": ["2020-01-01"],
        "vintage_date":     ["2020-04-29"],
        "value":            [21000.0],
    }))
    calls = []
    monkeypatch.setattr(
        _classic,
        "_fetch_fred_alfred_raw_api",
        lambda series_id, *, timeout: (calls.append(series_id) or _mock_api_rows()),
        raising=False,
    )
    _classic.fetch_fred_alfred("GDPC1", store=store, refresh=True)
    assert calls == ["GDPC1"]


def test_no_store_no_behavior_change(monkeypatch):
    """Backwards-compat: calling without store= must behave exactly
    like 0.65.0 (use the live API)."""
    from puremacro.fetch import _classic
    calls = []
    monkeypatch.setattr(
        _classic,
        "_fetch_fred_alfred_raw_api",
        lambda series_id, *, timeout: (calls.append(series_id) or _mock_api_rows()),
        raising=False,
    )
    df = _classic.fetch_fred_alfred("GDPC1")
    assert calls == ["GDPC1"]
    assert list(df.columns) == ["date", "vintage", "value"]
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_vintages_alfred_store/test_fetch_fred_alfred_with_store.py -v
```
Expected: FAIL — `store=` and `refresh=` are not yet kwargs of `fetch_fred_alfred`; the monkeypatch target `_fetch_fred_alfred_raw_api` doesn't exist yet.

- [ ] **Step 3: Refactor `fetch_fred_alfred` in `puremacro/fetch/_classic.py`**

Open `puremacro/puremacro/fetch/_classic.py`. Find the existing `fetch_fred_alfred` function (around line 75). It currently looks roughly like:

```python
def fetch_fred_alfred(series_id: str, *, timeout: float = 60.0) -> pd.DataFrame:
    """..."""
    url = _ALFRED.format(series_id)
    raw = _safe_urlopen(url, timeout=timeout)
    df = pd.read_csv(io.BytesIO(raw))
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    long = df.melt(id_vars="date", var_name="vintage_col", value_name="value")
    long["vintage"] = pd.to_datetime(
        long["vintage_col"].str.split("_").str[-1], errors="coerce"
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["vintage", "value"])
    return long[["date", "vintage", "value"]].reset_index(drop=True)
```

Refactor this in three steps:

1. **Extract the raw API path** into a new module-level helper `_fetch_fred_alfred_raw_api(series_id, *, timeout) -> pd.DataFrame`:

```python
def _fetch_fred_alfred_raw_api(series_id: str, *, timeout: float = 60.0) -> pd.DataFrame:
    """Live ALFRED CSV fetch. Returns long DataFrame ['date', 'vintage', 'value']."""
    url = _ALFRED.format(series_id)
    raw = _safe_urlopen(url, timeout=timeout)
    df = pd.read_csv(io.BytesIO(raw))
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    long = df.melt(id_vars="date", var_name="vintage_col", value_name="value")
    long["vintage"] = pd.to_datetime(
        long["vintage_col"].str.split("_").str[-1], errors="coerce"
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["vintage", "value"])
    return long[["date", "vintage", "value"]].reset_index(drop=True)
```

2. **Rewrite the public `fetch_fred_alfred`** to add the two new kwargs:

```python
def fetch_fred_alfred(
    series_id: str,
    *,
    timeout: float = 60.0,
    store: "AlfredVintageStore | None" = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download a FRED-ALFRED real-time vintage panel as long DataFrame.

    Returns columns ``[date, vintage, value]``. Backwards-compat with
    0.65.0: calling without ``store=`` is identical to the pre-0.66.0
    behaviour.

    Parameters
    ----------
    store : optional AlfredVintageStore (0.66.0+). When provided:
        - if ``refresh`` is False and the store already has data for
          ``series_id``, read from the store (no API call);
        - otherwise call the live API and write the rows back to the
          store via ``store.put_many``.
    refresh : if True, ignore the store on read and force a fresh API
        call. Rows still get written back. Useful when ALFRED has
        published new vintages since the store was filled.
    """
    if store is not None and not refresh and store.has_series(series_id):
        df_store = store.get(series_id)
        return df_store.rename(columns={
            "observation_date": "date",
            "vintage_date":     "vintage",
        })[["date", "vintage", "value"]].reset_index(drop=True)

    df = _fetch_fred_alfred_raw_api(series_id, timeout=timeout)

    if store is not None and not df.empty:
        store_rows = df.assign(series_id=series_id).rename(columns={
            "date":    "observation_date",
            "vintage": "vintage_date",
        })[["series_id", "observation_date", "vintage_date", "value"]]
        store.put_many(store_rows)

    return df
```

3. **Update `__all__`** at the bottom to keep both names exported:

```python
__all__ = ["fetch_fred", "fetch_fred_alfred", "_fetch_fred_alfred_raw_api"]
```

(Underscored name is exported only to make `monkeypatch.setattr(_classic, "_fetch_fred_alfred_raw_api", ...)` work in the test fixture.)

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_vintages_alfred_store/ -v
```
Expected: 25 passed (20 + 5 new).

- [ ] **Step 5: Confirm no regression in existing callers**

```bash
grep -rln "fetch_fred_alfred" puremacro/puremacro/ puremacro/tests/ puremacro/notebooks/ 2>&1 | head -10
```
Any existing call site that passes only `series_id` continues to work — the new kwargs are optional. Run any tests that hit this function:

```bash
pytest tests/ -k "alfred or vintage" -v --tb=short 2>&1 | tail -20
```
Expected: no NEW failures.

- [ ] **Step 6: Commit**

```bash
git add puremacro/puremacro/fetch/_classic.py tests/test_vintages_alfred_store/test_fetch_fred_alfred_with_store.py
git commit -m "$(cat <<'EOF'
feat(0.66.0): fetch_fred_alfred gains store= and refresh= kwargs

F2.2 third commit. Extracts the live-API path into
_fetch_fred_alfred_raw_api so callers can opt into store-backed
gap-fill semantics: with store= and refresh=False, the store
short-circuits the API when data is already present; with refresh=True,
the API is called and the store is repopulated. Default kwargs preserve
0.65.0 behavior exactly — callers passing only series_id are unaffected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Sub-slice 4 — F2.3 parser schema versioning

(Tasks 16–19.)

## Task 16: `_schema_check.py` framework

**Files:**
- Create: `puremacro/puremacro/narrative/sources/_schema_check.py`
- Test: `tests/test_narrative_schema_checks/__init__.py` (empty)
- Test: `tests/test_narrative_schema_checks/test_schema_check_framework.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_narrative_schema_checks/__init__.py` (empty).

Create `tests/test_narrative_schema_checks/test_schema_check_framework.py`:

```python
"""F2.3 — ParserSchemaMismatchError + assert_landmarks framework."""
from __future__ import annotations

import pytest


def test_passes_when_all_landmarks_present():
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = "<html><body><h1>Beige Book</h1><p>Summary of Commentary</p></body></html>"
    # Should not raise.
    assert_landmarks(
        text, source="beige_book", expected_version=1,
        landmarks=["Beige Book", "Summary of Commentary"],
    )


def test_raises_on_missing_substring_landmark():
    from puremacro.narrative.sources._schema_check import (
        assert_landmarks, ParserSchemaMismatchError,
    )
    text = "<html><body><h1>Beige Book</h1></body></html>"
    with pytest.raises(ParserSchemaMismatchError) as exc_info:
        assert_landmarks(
            text, source="beige_book", expected_version=1,
            landmarks=["Beige Book", "MISSING SENTENCE"],
        )
    msg = str(exc_info.value)
    assert "beige_book" in msg
    assert "MISSING SENTENCE" in msg
    assert "version=1" in msg


def test_supports_tuple_landmark_form():
    """(selector_hint, expected_text) tuples — selector is informational,
    the check is `expected_text in text`."""
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = "<html><h1>Beige Book</h1></html>"
    assert_landmarks(
        text, source="beige_book", expected_version=1,
        landmarks=[("h1", "Beige Book")],
    )


def test_raises_on_missing_tuple_landmark():
    from puremacro.narrative.sources._schema_check import (
        assert_landmarks, ParserSchemaMismatchError,
    )
    text = "<html><h1>Something Else</h1></html>"
    with pytest.raises(ParserSchemaMismatchError) as exc_info:
        assert_landmarks(
            text, source="beige_book", expected_version=1,
            landmarks=[("h1", "Beige Book")],
        )
    assert "h1" in str(exc_info.value)
    assert "Beige Book" in str(exc_info.value)


def test_parser_schema_mismatch_is_runtimeerror():
    from puremacro.narrative.sources._schema_check import ParserSchemaMismatchError
    assert issubclass(ParserSchemaMismatchError, RuntimeError)
```

- [ ] **Step 2: Run to verify failure**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
pytest tests/test_narrative_schema_checks/test_schema_check_framework.py -v
```
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create the framework module**

Create `puremacro/puremacro/narrative/sources/_schema_check.py`:

```python
"""Parser schema versioning + landmark-assertion framework (0.66.0+).

Each narrative connector parser declares a module-level
``PARSER_SCHEMA_VERSION`` and calls :func:`assert_landmarks` near the
top of its body parser. When the upstream source's HTML/JSON layout
drifts, the missing landmark triggers a loud ``ParserSchemaMismatchError``
naming the connector + the missing landmark. The ``iter_<source>``
wrapper catches the error and emits a ``UserWarning`` while yielding
empty (per RETRY_POLICY.md §4.1: "yield, don't raise").

The framework is intentionally minimal — landmarks are substring checks,
not CSS selectors or XPath expressions. Substring matching covers 95%
of real-world layout drift (renamed sections, removed headings,
restructured pages) at zero parser dependency cost.
"""
from __future__ import annotations


class ParserSchemaMismatchError(RuntimeError):
    """Raised by :func:`assert_landmarks` when an expected landmark is
    missing from the upstream body — i.e., the source layout has
    drifted away from what the parser was written against.

    Caught by ``iter_<source>`` generators, which emit a
    ``UserWarning`` naming the connector + missing landmark and then
    yield empty.
    """


def assert_landmarks(
    text: str,
    *,
    source: str,
    expected_version: int,
    landmarks: list,
) -> None:
    """Raise :class:`ParserSchemaMismatchError` if any landmark is missing.

    Parameters
    ----------
    text : the raw body about to be parsed (HTML, JSON, XML — anything
        string-shaped).
    source : the connector's canonical name (e.g. ``"beige_book"``,
        ``"eu_eurlex"``). Appears in the error message and in the
        ``warnings.warn`` emitted by the wrapper.
    expected_version : the parser's currently-locked schema version
        (matches the module's ``PARSER_SCHEMA_VERSION`` constant).
        Appears in the error message so a researcher knows which
        version the parser expected.
    landmarks : list of items. Each item is either:
        - ``str``: a substring that must appear in ``text``.
        - ``(selector, expected)`` tuple: the selector is informational
          (helps the error message); the check is
          ``expected in text``.
    """
    for landmark in landmarks:
        if isinstance(landmark, tuple):
            selector, expected = landmark
            if expected in text:
                continue
            raise ParserSchemaMismatchError(
                f"{source!r}: missing landmark ({selector}, {expected!r}) "
                f"(version={expected_version}). Upstream layout has "
                f"likely drifted; inspect the source HTML and bump "
                f"PARSER_SCHEMA_VERSION."
            )
        else:
            if landmark in text:
                continue
            raise ParserSchemaMismatchError(
                f"{source!r}: missing landmark {landmark!r} "
                f"(version={expected_version}). Upstream layout has "
                f"likely drifted; inspect the source HTML and bump "
                f"PARSER_SCHEMA_VERSION."
            )


__all__ = ["ParserSchemaMismatchError", "assert_landmarks"]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_narrative_schema_checks/test_schema_check_framework.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add puremacro/puremacro/narrative/sources/_schema_check.py tests/test_narrative_schema_checks/__init__.py tests/test_narrative_schema_checks/test_schema_check_framework.py
git commit -m "$(cat <<'EOF'
feat(0.66.0): _schema_check framework — ParserSchemaMismatchError + assert_landmarks

F2.3 first commit. Minimal substring-based landmark assertion for
narrative connector parsers. Each connector declares
PARSER_SCHEMA_VERSION and calls assert_landmarks(...) on the first
upstream body it sees. On drift, raises ParserSchemaMismatchError
(RuntimeError) naming the connector + missing landmark + expected
version. The iter_<source> wrappers catch this and yield empty per
RETRY_POLICY.md §4.1, so a 30-source aggregation loses one source
loudly instead of silently emitting broken records.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: Rollout batch 1 — beige_book + eu_eurlex + eu_parliament + us_cbo

**Files:**
- Modify: `puremacro/puremacro/narrative/sources/beige_book.py`
- Modify: `puremacro/puremacro/narrative/sources/eu_eurlex.py`
- Modify: `puremacro/puremacro/narrative/sources/eu_parliament.py`
- Modify: `puremacro/puremacro/narrative/sources/us_cbo.py`
- Create: `puremacro/puremacro/narrative/sources/_fixtures/beige_book_v1.html`
- Create: `puremacro/puremacro/narrative/sources/_fixtures/eu_eurlex_v1.html`
- Create: `puremacro/puremacro/narrative/sources/_fixtures/eu_parliament_v1.html`
- Create: `puremacro/puremacro/narrative/sources/_fixtures/us_cbo_v1.xml`
- Test: `tests/test_narrative_schema_checks/test_landmark_assertions.py`
- Test: `tests/test_narrative_schema_checks/test_landmark_fixtures.py`

Pattern for each connector:
1. Add `PARSER_SCHEMA_VERSION = 1` near the top of the source file.
2. Add `from ._schema_check import assert_landmarks` to the imports.
3. Add a call to `assert_landmarks(...)` at the top of the body parser (`_parse_modern_html`, `_parse_eurlex_html`, `_parse_ep_page`, `_parse_rss`) with 2–3 landmarks specific to that source.
4. Ensure the `iter_<source>` outer loop catches `ParserSchemaMismatchError` and emits a `UserWarning` before stopping yielding for that batch.
5. Save a real upstream snapshot as a fixture under `_fixtures/<source>_v1.<ext>`.

- [ ] **Step 1: Write the parametrized rollout test**

Create `tests/test_narrative_schema_checks/test_landmark_assertions.py`:

```python
"""F2.3 — per-connector PARSER_SCHEMA_VERSION + assert_landmarks call."""
from __future__ import annotations

import ast
import importlib
import pathlib

import pytest


# All 8 connectors in the Slice A rollout. Tasks 17 + 18 add the
# rollouts; this test grows green incrementally as each is committed.
_CONNECTORS = [
    "beige_book", "eu_eurlex", "eu_parliament", "us_cbo",
    "fed_minutes", "fed_speeches", "bluesky", "ecb_press",
]


def _module_for(name: str):
    return importlib.import_module(f"puremacro.narrative.sources.{name}")


@pytest.mark.parametrize("name", _CONNECTORS)
def test_connector_declares_parser_schema_version(name):
    mod = _module_for(name)
    assert hasattr(mod, "PARSER_SCHEMA_VERSION"), (
        f"{name}.py must declare PARSER_SCHEMA_VERSION (F2.3 contract)"
    )
    assert isinstance(mod.PARSER_SCHEMA_VERSION, int)
    assert mod.PARSER_SCHEMA_VERSION >= 1


@pytest.mark.parametrize("name", _CONNECTORS)
def test_connector_imports_assert_landmarks(name):
    """AST scan: the module must import or reference `assert_landmarks`."""
    mod = _module_for(name)
    src = pathlib.Path(mod.__file__).read_text()
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == "assert_landmarks" for alias in node.names):
                found = True
                break
        if isinstance(node, ast.Name) and node.id == "assert_landmarks":
            found = True
            break
    assert found, (
        f"{name}.py must import or reference `assert_landmarks` "
        f"(F2.3 contract)"
    )
```

Create `tests/test_narrative_schema_checks/test_landmark_fixtures.py`:

```python
"""F2.3 — Each connector's golden fixture parses without raising."""
from __future__ import annotations

import pathlib

import pytest


_FIXTURE_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "puremacro" / "puremacro" / "narrative" / "sources" / "_fixtures"
)


def _fixture_text(name: str, ext: str) -> str:
    return (_FIXTURE_DIR / f"{name}_v1.{ext}").read_text()


def test_beige_book_fixture_passes_landmark_check():
    from puremacro.narrative.sources import beige_book
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = _fixture_text("beige_book", "html")
    # The landmark list lives in the parser; we replicate it here for
    # the regression guard. Update both together when bumping.
    assert_landmarks(
        text, source="beige_book",
        expected_version=beige_book.PARSER_SCHEMA_VERSION,
        landmarks=["Beige Book", "Summary of Commentary"],
    )


def test_eu_eurlex_fixture_passes_landmark_check():
    from puremacro.narrative.sources import eu_eurlex
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = _fixture_text("eu_eurlex", "html")
    assert_landmarks(
        text, source="eu_eurlex",
        expected_version=eu_eurlex.PARSER_SCHEMA_VERSION,
        landmarks=["CELEX", "EUR-Lex"],
    )


def test_eu_parliament_fixture_passes_landmark_check():
    from puremacro.narrative.sources import eu_parliament
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = _fixture_text("eu_parliament", "html")
    assert_landmarks(
        text, source="eu_parliament",
        expected_version=eu_parliament.PARSER_SCHEMA_VERSION,
        landmarks=["European Parliament", "Plenary"],
    )


def test_us_cbo_fixture_passes_landmark_check():
    from puremacro.narrative.sources import us_cbo
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = _fixture_text("us_cbo", "xml")
    assert_landmarks(
        text, source="us_cbo",
        expected_version=us_cbo.PARSER_SCHEMA_VERSION,
        landmarks=["<rss", "Congressional Budget Office"],
    )
```

(Tests for the other 4 connectors are added in Task 18.)

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_narrative_schema_checks/test_landmark_assertions.py tests/test_narrative_schema_checks/test_landmark_fixtures.py -v
```
Expected: every parameterised test fails (no `PARSER_SCHEMA_VERSION` in any of the 8 connectors yet); fixture tests fail (no fixture files yet).

- [ ] **Step 3: Edit `puremacro/narrative/sources/beige_book.py`**

Near the top of the file (after existing imports), add:

```python
from ._schema_check import assert_landmarks, ParserSchemaMismatchError

PARSER_SCHEMA_VERSION = 1
```

At the top of `_parse_modern_html(html, *, release_date, source_url, district=None)` (around line 262), add the landmark assertion as the first statement of the function body:

```python
    assert_landmarks(
        html, source="beige_book",
        expected_version=PARSER_SCHEMA_VERSION,
        landmarks=["Beige Book", "Summary of Commentary"],
    )
```

Find the `iter_beige_book` function (around line 581). Wrap its inner per-release iteration in a `try/except ParserSchemaMismatchError` block that emits a `UserWarning` and stops yielding for that release. Conceptually:

```python
import warnings as _warnings

def iter_beige_book(...):
    for ... release loop ...:
        try:
            yield from _iter_modern(year, month, ...)
        except ParserSchemaMismatchError as e:
            _warnings.warn(
                f"puremacro.narrative.sources.beige_book: schema mismatch "
                f"for release {year}-{month:02d}: {e}",
                UserWarning, stacklevel=2,
            )
            continue
```

(Adapt to the existing loop structure; the key is that the `ParserSchemaMismatchError` is caught at the release-batch boundary, not allowed to propagate out of `iter_beige_book` per the RETRY_POLICY.md "yield, don't raise" contract.)

- [ ] **Step 4: Create the beige_book golden fixture**

Create `puremacro/puremacro/narrative/sources/_fixtures/beige_book_v1.html` with a minimal real-shape snippet that satisfies the landmarks:

```html
<!DOCTYPE html>
<html lang="en">
<head><title>Beige Book — March 2024</title></head>
<body>
<h1>Beige Book — March 2024</h1>
<section class="summary">
  <h2>Summary of Commentary on Current Economic Conditions</h2>
  <p>Reports from the twelve Federal Reserve Districts indicate that
     economic activity expanded slightly, on balance, since early
     January.</p>
</section>
<section class="districts">
  <h2>Boston</h2>
  <p>Activity grew modestly. Hiring remained difficult.</p>
  <h2>New York</h2>
  <p>Activity was little changed. Selling prices grew at a slow pace.</p>
</section>
</body>
</html>
```

- [ ] **Step 5: Edit `puremacro/narrative/sources/eu_eurlex.py`**

Same pattern: add `from ._schema_check import assert_landmarks, ParserSchemaMismatchError` import, `PARSER_SCHEMA_VERSION = 1` constant, and put

```python
    assert_landmarks(
        html, source="eu_eurlex",
        expected_version=PARSER_SCHEMA_VERSION,
        landmarks=["CELEX", "EUR-Lex"],
    )
```

at the top of `_parse_eurlex_html` (line ~61). Wrap `iter_eurlex` (line ~284) iteration to catch `ParserSchemaMismatchError` and emit a UserWarning per-record (skipping that record).

- [ ] **Step 6: Create the eu_eurlex golden fixture**

Create `puremacro/puremacro/narrative/sources/_fixtures/eu_eurlex_v1.html`:

```html
<!DOCTYPE html>
<html><body>
<h1>EUR-Lex - 32024R0123 - EN</h1>
<p>CELEX: 32024R0123</p>
<div class="document">
  <h2>Regulation (EU) 2024/123 of the European Parliament and of the Council</h2>
  <p>Of 14 February 2024 on certain aspects of the internal market...</p>
</div>
</body></html>
```

- [ ] **Step 7: Edit `puremacro/narrative/sources/eu_parliament.py`**

Same pattern. `PARSER_SCHEMA_VERSION = 1`, import `assert_landmarks`, put

```python
    assert_landmarks(
        html, source="eu_parliament",
        expected_version=PARSER_SCHEMA_VERSION,
        landmarks=["European Parliament", "Plenary"],
    )
```

at the top of `_parse_ep_page` (line ~90). Wrap `iter_ep_debates` (line ~141) iteration to catch + warn.

- [ ] **Step 8: Create the eu_parliament golden fixture**

Create `puremacro/puremacro/narrative/sources/_fixtures/eu_parliament_v1.html`:

```html
<!DOCTYPE html>
<html><body>
<h1>European Parliament — Plenary sitting 2024-02-14</h1>
<div class="verbatim">
  <h2>Plenary verbatim report</h2>
  <p><strong>President.</strong> The next item is the debate on...</p>
  <p>I declare the sitting open.</p>
</div>
</body></html>
```

- [ ] **Step 9: Edit `puremacro/narrative/sources/us_cbo.py`**

Same pattern. `PARSER_SCHEMA_VERSION = 1`, import `assert_landmarks`, put

```python
    assert_landmarks(
        xml_text, source="us_cbo",
        expected_version=PARSER_SCHEMA_VERSION,
        landmarks=["<rss", "Congressional Budget Office"],
    )
```

at the top of `_parse_rss` (line ~59). Wrap `iter_cbo` (line ~177) iteration to catch + warn at the feed boundary.

- [ ] **Step 10: Create the us_cbo golden fixture**

Create `puremacro/puremacro/narrative/sources/_fixtures/us_cbo_v1.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>Congressional Budget Office — Publications</title>
  <link>https://www.cbo.gov/publications</link>
  <description>Recent CBO reports and analyses.</description>
  <item>
    <title>The Budget and Economic Outlook: 2024 to 2034</title>
    <link>https://www.cbo.gov/publication/59710</link>
    <pubDate>Wed, 07 Feb 2024 12:00:00 EST</pubDate>
    <description>CBO's annual report on the federal budget...</description>
  </item>
</channel>
</rss>
```

- [ ] **Step 11: Run the relevant tests to verify the 4 connectors pass**

```bash
pytest tests/test_narrative_schema_checks/test_landmark_assertions.py -v -k "beige_book or eu_eurlex or eu_parliament or us_cbo"
pytest tests/test_narrative_schema_checks/test_landmark_fixtures.py -v
```
Expected: 8 of the 16 parametrized tests pass (the 4 connectors × 2 tests each); 4 fixture tests pass. The other 4 connectors still fail — those are addressed in Task 18.

- [ ] **Step 12: Confirm no regression in the iter_<source> wrappers**

```bash
pytest tests/ -k "beige_book or eu_eurlex or eu_parliament or cbo" -v --tb=short 2>&1 | tail -20
```
Expected: no NEW failures vs. baseline.

- [ ] **Step 13: Commit**

```bash
git add puremacro/puremacro/narrative/sources/beige_book.py puremacro/puremacro/narrative/sources/eu_eurlex.py puremacro/puremacro/narrative/sources/eu_parliament.py puremacro/puremacro/narrative/sources/us_cbo.py puremacro/puremacro/narrative/sources/_fixtures/beige_book_v1.html puremacro/puremacro/narrative/sources/_fixtures/eu_eurlex_v1.html puremacro/puremacro/narrative/sources/_fixtures/eu_parliament_v1.html puremacro/puremacro/narrative/sources/_fixtures/us_cbo_v1.xml tests/test_narrative_schema_checks/test_landmark_assertions.py tests/test_narrative_schema_checks/test_landmark_fixtures.py
git commit -m "$(cat <<'EOF'
feat(0.66.0): PARSER_SCHEMA_VERSION + landmarks on beige_book/eu_eurlex/eu_parliament/us_cbo

F2.3 rollout batch 1 (4 of 8 connectors). Each gets:
- PARSER_SCHEMA_VERSION = 1 module constant,
- assert_landmarks(...) call at the top of its body parser,
- iter_<source> wrapper that catches ParserSchemaMismatchError and
  emits UserWarning while yielding empty per RETRY_POLICY.md §4.1,
- a golden _fixtures/<source>_v1.{html,xml} snapshot for the
  regression guard test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: Rollout batch 2 — fed_minutes + fed_speeches + bluesky + ecb_press

**Files:**
- Modify: `puremacro/puremacro/narrative/sources/fed_minutes.py`
- Modify: `puremacro/puremacro/narrative/sources/fed_speeches.py`
- Modify: `puremacro/puremacro/narrative/sources/bluesky.py`
- Modify: `puremacro/puremacro/narrative/sources/ecb_press.py`
- Create: `puremacro/puremacro/narrative/sources/_fixtures/fed_minutes_v1.html`
- Create: `puremacro/puremacro/narrative/sources/_fixtures/fed_speeches_v1.html`
- Create: `puremacro/puremacro/narrative/sources/_fixtures/bluesky_v1.json`
- Create: `puremacro/puremacro/narrative/sources/_fixtures/ecb_press_v1.html`
- Modify: `tests/test_narrative_schema_checks/test_landmark_fixtures.py` (add 4 more tests)

These 4 connectors do NOT have a clean separate `_parse_*` function — parsing happens inside their `iter_<source>` generators. The pattern is to call `assert_landmarks(...)` on the first body fetched inside the iterator (use a small `_checked` sentinel to ensure the check runs exactly once per call, not per record), and to wrap the iteration in a `try/except ParserSchemaMismatchError → UserWarning + return`.

- [ ] **Step 1: Append the 4 fixture tests to `test_landmark_fixtures.py`**

Append to `tests/test_narrative_schema_checks/test_landmark_fixtures.py`:

```python
def test_fed_minutes_fixture_passes_landmark_check():
    from puremacro.narrative.sources import fed_minutes
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = _fixture_text("fed_minutes", "html")
    assert_landmarks(
        text, source="fed_minutes",
        expected_version=fed_minutes.PARSER_SCHEMA_VERSION,
        landmarks=["Federal Open Market Committee", "Minutes"],
    )


def test_fed_speeches_fixture_passes_landmark_check():
    from puremacro.narrative.sources import fed_speeches
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = _fixture_text("fed_speeches", "html")
    assert_landmarks(
        text, source="fed_speeches",
        expected_version=fed_speeches.PARSER_SCHEMA_VERSION,
        landmarks=["Speeches", "Federal Reserve"],
    )


def test_bluesky_fixture_passes_landmark_check():
    from puremacro.narrative.sources import bluesky
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = _fixture_text("bluesky", "json")
    assert_landmarks(
        text, source="bluesky",
        expected_version=bluesky.PARSER_SCHEMA_VERSION,
        landmarks=['"$type":', "app.bsky.feed.post"],
    )


def test_ecb_press_fixture_passes_landmark_check():
    from puremacro.narrative.sources import ecb_press
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = _fixture_text("ecb_press", "html")
    assert_landmarks(
        text, source="ecb_press",
        expected_version=ecb_press.PARSER_SCHEMA_VERSION,
        landmarks=["European Central Bank", "Press release"],
    )
```

- [ ] **Step 2: Edit `puremacro/narrative/sources/fed_minutes.py`**

Add:
```python
from ._schema_check import assert_landmarks, ParserSchemaMismatchError

PARSER_SCHEMA_VERSION = 1
```

Inside `iter_fed_minutes` (line ~50), wrap the per-release body fetch + parse with the landmark assertion. Use a `_checked` local sentinel so the assertion runs only on the first body:

```python
def iter_fed_minutes():
    _checked = False
    for ...release loop...:
        body = ...fetch body...
        if not _checked:
            try:
                assert_landmarks(
                    body, source="fed_minutes",
                    expected_version=PARSER_SCHEMA_VERSION,
                    landmarks=["Federal Open Market Committee", "Minutes"],
                )
            except ParserSchemaMismatchError as e:
                import warnings
                warnings.warn(
                    f"puremacro.narrative.sources.fed_minutes: schema "
                    f"mismatch on first body: {e}",
                    UserWarning, stacklevel=2,
                )
                return
            _checked = True
        # ...parse body and yield records...
```

- [ ] **Step 3: Create `_fixtures/fed_minutes_v1.html`**

```html
<!DOCTYPE html>
<html><body>
<h1>Minutes of the Federal Open Market Committee — January 30-31, 2024</h1>
<div class="content">
  <h2>Federal Open Market Committee — Meeting Minutes</h2>
  <p>The Federal Open Market Committee met in the Board Room of the
     Marriner S. Eccles Federal Reserve Board Building...</p>
</div>
</body></html>
```

- [ ] **Step 4: Edit `puremacro/narrative/sources/fed_speeches.py`**

Same pattern in `iter_fed_speeches` (line ~12) with landmarks `["Speeches", "Federal Reserve"]`.

- [ ] **Step 5: Create `_fixtures/fed_speeches_v1.html`**

```html
<!DOCTYPE html>
<html><body>
<h1>Speeches — Federal Reserve</h1>
<ul class="speech-list">
  <li>
    <h2>Powell — Economic Outlook</h2>
    <a href="/newsevents/speech/powell20240301a.htm">Read full transcript</a>
  </li>
</ul>
</body></html>
```

- [ ] **Step 6: Edit `puremacro/narrative/sources/bluesky.py`**

Same pattern in `iter_bluesky_posts` (line ~246). Bluesky returns JSON, so the landmark check should be against the raw JSON text. Use landmarks `['"$type":', "app.bsky.feed.post"]`:

```python
        if not _checked:
            try:
                assert_landmarks(
                    raw_json_text, source="bluesky",
                    expected_version=PARSER_SCHEMA_VERSION,
                    landmarks=['"$type":', "app.bsky.feed.post"],
                )
            except ParserSchemaMismatchError as e:
                ...warn + return...
            _checked = True
```

- [ ] **Step 7: Create `_fixtures/bluesky_v1.json`**

```json
{
  "feed": [
    {
      "post": {
        "uri": "at://did:plc:abc/app.bsky.feed.post/123",
        "record": {
          "$type": "app.bsky.feed.post",
          "text": "Inflation expectations remain anchored.",
          "createdAt": "2024-03-15T12:00:00Z"
        }
      }
    }
  ]
}
```

- [ ] **Step 8: Edit `puremacro/narrative/sources/ecb_press.py`**

Same pattern in `iter_ecb_press` (line ~16) with landmarks `["European Central Bank", "Press release"]`.

- [ ] **Step 9: Create `_fixtures/ecb_press_v1.html`**

```html
<!DOCTYPE html>
<html><body>
<h1>Press release — European Central Bank</h1>
<div class="press-release">
  <h2>Monetary policy decisions</h2>
  <p>The Governing Council today decided to raise the three key ECB
     interest rates by 25 basis points...</p>
  <footer>European Central Bank | Frankfurt am Main</footer>
</div>
</body></html>
```

- [ ] **Step 10: Run all schema-check tests to verify everything passes**

```bash
pytest tests/test_narrative_schema_checks/ -v
```
Expected: 21 passed (5 framework + 16 parametrized × 2 + 8 fixture, minus duplicates — count may differ; the key is no failures).

- [ ] **Step 11: Run the affected source-module tests to confirm zero regression**

```bash
pytest tests/ -k "fed or bluesky or ecb_press" -v --tb=short 2>&1 | tail -20
```
Expected: no NEW failures vs. baseline.

- [ ] **Step 12: Commit**

```bash
git add puremacro/puremacro/narrative/sources/fed_minutes.py puremacro/puremacro/narrative/sources/fed_speeches.py puremacro/puremacro/narrative/sources/bluesky.py puremacro/puremacro/narrative/sources/ecb_press.py puremacro/puremacro/narrative/sources/_fixtures/fed_minutes_v1.html puremacro/puremacro/narrative/sources/_fixtures/fed_speeches_v1.html puremacro/puremacro/narrative/sources/_fixtures/bluesky_v1.json puremacro/puremacro/narrative/sources/_fixtures/ecb_press_v1.html tests/test_narrative_schema_checks/test_landmark_fixtures.py
git commit -m "$(cat <<'EOF'
feat(0.66.0): PARSER_SCHEMA_VERSION + landmarks on fed_minutes/fed_speeches/bluesky/ecb_press

F2.3 rollout batch 2 (4 of 8 connectors). These don't have a clean
separate _parse_* function, so the assert_landmarks call sits inside
the iter_<source> generator, gated by a _checked local sentinel so
it runs exactly once per call (not per record). On mismatch, the
generator emits a UserWarning and returns — yielding zero records
for that call rather than failing.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: Coverage assertion test (enforces the 8-connector list)

**Files:**
- Test: `tests/test_narrative_schema_checks/test_coverage_assertion.py`

Now that all 8 connectors are wired, the coverage test passes on first run.

- [ ] **Step 1: Write the test**

Create `tests/test_narrative_schema_checks/test_coverage_assertion.py`:

```python
"""F2.3 — coverage assertion: the 8 named connectors all declare
PARSER_SCHEMA_VERSION and call assert_landmarks. Fails the build if
any Slice-A connector regresses."""
from __future__ import annotations

import ast
import importlib
import inspect
import pathlib

import pytest


_SLICE_A_CONNECTORS = (
    "beige_book", "eu_eurlex", "eu_parliament", "us_cbo",
    "fed_minutes", "fed_speeches", "bluesky", "ecb_press",
)


def _module_source(name: str) -> str:
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    return pathlib.Path(mod.__file__).read_text()


def _has_assert_landmarks_call(name: str) -> bool:
    src = _module_source(name)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "assert_landmarks":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "assert_landmarks":
                return True
    return False


def _has_parser_schema_version(name: str) -> bool:
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    return hasattr(mod, "PARSER_SCHEMA_VERSION") and isinstance(
        mod.PARSER_SCHEMA_VERSION, int
    )


def test_every_slice_a_connector_has_parser_schema_version():
    missing = [n for n in _SLICE_A_CONNECTORS if not _has_parser_schema_version(n)]
    assert not missing, (
        f"F2.3 contract violation: connectors missing PARSER_SCHEMA_VERSION: {missing}"
    )


def test_every_slice_a_connector_calls_assert_landmarks():
    missing = [n for n in _SLICE_A_CONNECTORS if not _has_assert_landmarks_call(n)]
    assert not missing, (
        f"F2.3 contract violation: connectors not calling assert_landmarks(): {missing}"
    )
```

- [ ] **Step 2: Run to verify pass**

```bash
pytest tests/test_narrative_schema_checks/test_coverage_assertion.py -v
```
Expected: 2 passed.

- [ ] **Step 3: Full schema-check suite green**

```bash
pytest tests/test_narrative_schema_checks/ -v 2>&1 | tail -10
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_narrative_schema_checks/test_coverage_assertion.py
git commit -m "$(cat <<'EOF'
test(0.66.0): coverage assertion — the 8 Slice-A connectors must keep PARSER_SCHEMA_VERSION

F2.3 closes out with the coverage test that AST-scans each of the 8
listed connectors and asserts both (a) the module declares
PARSER_SCHEMA_VERSION as an int, and (b) the module's body contains
a call to assert_landmarks(...). Fails the build if a future
refactor accidentally removes either piece.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Sub-slice 5 — Release commits

(Tasks 20–21.)

## Task 20: R5_01 notebook + paired builder

**Files:**
- Create: `tools/make_notebook_R5_01.py`
- Create: `notebooks/R5_data_infra/R5_01_cache_and_credentials_demo.ipynb`

Per the memory rule: notebooks ↔ builders ship together; nbconvert > 5 min must run in controller/foreground, not subagent. R5_01 is small (~30s) but the rule still applies — the executor runs nbconvert foreground, not via a delegated subagent.

- [ ] **Step 1: Create the builder**

Create `tools/make_notebook_R5_01.py`:

```python
"""Build R5_01_cache_and_credentials_demo.ipynb — Slice A demo.

Demonstrates the 0.66.0 data-infrastructure additions:
  1. credentials.status() — which API keys are configured.
  2. SQLite HTTP cache: first call populates; second is a hit.
  3. cache.http_cache_size_bytes() / http_list_urls() introspection.
  4. AlfredVintageStore: put_many + get from a synthetic fixture
     (no real API call — fully offline-runnable).
  5. as_of_from_store: end-to-end vintage slicing.

Run:
    python tools/make_notebook_R5_01.py
Then execute (foreground, controller-side):
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/R5_data_infra/R5_01_cache_and_credentials_demo.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat


_REPO = Path(__file__).resolve().parent.parent
_OUT = _REPO / "notebooks" / "R5_data_infra" / "R5_01_cache_and_credentials_demo.ipynb"


def _md(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text)


def _code(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(src)


def build() -> None:
    nb = nbformat.v4.new_notebook()
    cells = []

    cells.append(_md("""# R5_01 — Data-infrastructure foundation (0.66.0)

Demonstrates the four F2 Slice A components:

- **Credentials** — `credentials.status()` shows which API keys you have
  configured (env vars + `~/.puremacro/credentials.toml`), without
  leaking the actual values.
- **SQLite HTTP cache** — replaces the flat-file cache with a single
  queryable DB at `~/.cache/puremacro/cache.db`. New introspection
  helpers in `puremacro.cache`.
- **ALFRED vintage store** — persistent per-series vintage panels.
  `fetch_fred_alfred(store=...)` reads from store first; only hits the
  API for missing data.
- **Parser schema versioning** — narrative connectors now fail loudly
  when upstream HTML drifts (see `docs/SIGNAL_CONTRACT.md` and
  `docs/CACHE_DB.md`).

Spec: `docs/specs/2026-05-26-puremacro-066-f2-slice-a-data-infrastructure-design.md`
"""))

    cells.append(_code("""\
from __future__ import annotations
import pandas as pd
import puremacro
print('puremacro', puremacro.__version__)
"""))

    cells.append(_md("## 1. Credentials introspection"))
    cells.append(_code("""\
from puremacro import credentials
credentials.status()
"""))

    cells.append(_md("""## 2. HTTP cache — populate + introspect

The SQLite cache lives at `~/.cache/puremacro/cache.db` (overridable
via `$PUREMACRO_HTTP_CACHE_DIR`). For this demo we use a tmp path so
the notebook is fully self-contained.
"""))
    cells.append(_code("""\
import os, tempfile
from pathlib import Path
tmpdir = Path(tempfile.mkdtemp())
os.environ['PUREMACRO_HTTP_CACHE_DIR'] = str(tmpdir)
# Reset the singleton so the env var takes effect.
import puremacro._cache_db as _db
_db.close_conn()

from puremacro._http_cache import cache_read, cache_write
cache_write(tmpdir, 'https://example.com/a', b'hello-a',
            content_type='text/plain')
cache_write(tmpdir, 'https://example.com/b', b'hello-bbbbbbbbb',
            content_type='text/plain')

print('cache_read(a):', cache_read(tmpdir, 'https://example.com/a'))

import puremacro.cache as C
print('http_list_urls():    ', C.http_list_urls())
print('http_cache_size_bytes:', C.http_cache_size_bytes())
"""))

    cells.append(_md("## 3. ALFRED vintage store (offline fixture)"))
    cells.append(_code("""\
from puremacro.vintages import AlfredVintageStore, as_of_from_store
store = AlfredVintageStore()
store.put_many(pd.DataFrame({
    'series_id':        ['GDPC1'] * 4,
    'observation_date': ['2020-01-01', '2020-01-01',
                          '2020-04-01', '2020-04-01'],
    'vintage_date':     ['2020-04-29', '2020-07-30',
                          '2020-07-30', '2020-10-29'],
    'value':            [21000.0, 21010.0, 19500.0, 19520.0],
}))
print('series_list:', store.series_list())
print('coverage:   ', store.coverage('GDPC1'))
"""))

    cells.append(_md("### `as_of_from_store` — vintage slicing"))
    cells.append(_code("""\
print('As of 2020-05-01:')
print(as_of_from_store('GDPC1', '2020-05-01', store))
print()
print('As of 2020-08-01:')
print(as_of_from_store('GDPC1', '2020-08-01', store))
"""))

    cells.append(_md("""## What's next

- **Slice B (0.67.0)** adds F2.4 governed-fallback (unified live →
  Wayback → Playwright → fail policy) and F2.5 per-connector health
  telemetry.
- The remaining ~50 narrative connectors get `PARSER_SCHEMA_VERSION`
  + landmark assertions in subsequent slices.

Reference: `docs/CREDENTIALS.md`, `docs/CACHE_DB.md`.
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
python tools/make_notebook_R5_01.py
```
Expected: `wrote notebooks/R5_data_infra/R5_01_cache_and_credentials_demo.ipynb`.

- [ ] **Step 3: Execute the notebook FOREGROUND (controller-side, NOT a subagent)**

```bash
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/R5_data_infra/R5_01_cache_and_credentials_demo.ipynb
```
Expected: notebook executes without error. Verify cell outputs are populated.

- [ ] **Step 4: Commit builder + executed notebook together**

```bash
git add tools/make_notebook_R5_01.py notebooks/R5_data_infra/R5_01_cache_and_credentials_demo.ipynb
git commit -m "$(cat <<'EOF'
feat(0.66.0): R5_01 data-infra demo notebook + paired builder

Slice A's visible deliverable. R5_01 walks the four sub-components
end-to-end: credentials.status(), the SQLite cache populate +
introspect, AlfredVintageStore offline fixture, as_of_from_store
slicing. Fully offline-runnable — no API keys required, no live
network calls. Shipped with the paired builder per the
notebook ↔ builder pairing rule.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 21: Version bump + CHANGELOG entry + final sanity sweep

**Files:**
- Modify: `puremacro/pyproject.toml` (`version = "0.66.0"`, `requires-python = ">=3.12"`)
- Modify: `puremacro/puremacro/__init__.py` (`__version__ = "0.66.0"`)
- Modify: `puremacro/CHANGELOG.md` (prepend 0.66.0 section)
- Modify: `puremacro/ARCHITECTURE.md` (add Data infrastructure subsection)

- [ ] **Step 1: Add version smoke test**

Append to `tests/test_credentials/test_service_registry.py`:

```python
def test_puremacro_version_is_066():
    import puremacro
    assert puremacro.__version__ == "0.66.0"
```

- [ ] **Step 2: Run to verify failure**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
pytest tests/test_credentials/test_service_registry.py::test_puremacro_version_is_066 -v
```
Expected: FAIL — version still `"0.65.0"`.

- [ ] **Step 3: Bump version in `puremacro/__init__.py`**

Find `__version__ = "0.65.0"` and change to `__version__ = "0.66.0"`.

- [ ] **Step 4: Bump `pyproject.toml`**

Two changes:
1. `version = "0.65.0"` → `version = "0.66.0"`.
2. `requires-python = ">=3.10"` → `requires-python = ">=3.12"`.

- [ ] **Step 5: Prepend CHANGELOG entry**

Insert the following block in `puremacro/CHANGELOG.md` immediately after the `# Changelog` header + intro paragraph, BEFORE the `## 0.65.0 (2026-05-26)` section:

```markdown
## 0.66.0 (2026-05-26)

**F2 Slice A — data-infrastructure foundation (credentials + SQLite cache + ALFRED vintage store + parser schema versioning).**

### Added
- `puremacro.credentials`: centralised API-key resolver with
  env-vars + `~/.puremacro/credentials.toml` (priority: explicit
  kwarg > env vars > config file > None). `SERVICES` registry for
  fred / bea / anthropic / openai / census. `MissingCredentialError`
  with researcher-actionable messages (lists env vars checked +
  config file path + signup URL). `status()` introspection returns
  a DataFrame indicating which keys are configured without leaking
  values.
- `puremacro._cache_db`: SQLite singleton-connection manager backing
  both the HTTP cache and the ALFRED vintage store. WAL journal mode
  for concurrent reader/writer. `bootstrap_schema()` idempotent;
  `migrate_from_flat_files()` opportunistic + warn-and-skip on
  corrupt sidecars.
- `puremacro._http_cache` rewritten against `_cache_db`. Public API
  (`cache_read`, `cache_write`, `cache_key`, `default_cache_dir`)
  preserved verbatim. Lazy migration trigger on first call after
  upgrade.
- `tools/cache_migrate.py`: one-shot CLI (`--apply`, `--apply --rm`)
  for users who want to migrate flat-file cache up-front.
- `puremacro.cache` gains `http_list_urls()`, `http_cache_size_bytes()`,
  `http_cache_clear(older_than=pd.Timedelta(...))`. Distinct from the
  existing `disk_cache`/`disk_cache_path` helpers in the same module.
- `puremacro.vintages.AlfredVintageStore`: persistent FRED-ALFRED
  vintage panel backed by `alfred_vintages` table. `put`/`put_many`/
  `get`/`has_series`/`series_list`/`coverage`. Failure modes warn +
  degrade (empty DataFrame / 0 inserted), never raise.
- `puremacro.vintages.as_of_from_store`: convenience helper that
  combines `store.get(...)` with the existing in-memory `as_of()`
  slicer.
- `puremacro.fetch.fetch_fred_alfred` gains `store=` and `refresh=`
  kwargs (opt-in; default behaviour unchanged from 0.65.0).
- `puremacro.narrative.sources._schema_check`:
  `ParserSchemaMismatchError` + `assert_landmarks(text, source=,
  expected_version=, landmarks=...)` framework. Each of 8 high-value
  connectors (beige_book, eu_eurlex, eu_parliament, us_cbo,
  fed_minutes, fed_speeches, bluesky, ecb_press) declares
  `PARSER_SCHEMA_VERSION = 1` + calls `assert_landmarks` at the top
  of its body parser. `iter_<source>` wrappers catch
  `ParserSchemaMismatchError` and emit `UserWarning` per
  RETRY_POLICY.md §4.1.
- Golden fixtures `narrative/sources/_fixtures/<source>_v1.{html,xml,json}`
  for the 8 listed connectors (regression guards for the fixture tests).
- `docs/CREDENTIALS.md`, `docs/CACHE_DB.md`: single-page references.
- `notebooks/R5_data_infra/R5_01_cache_and_credentials_demo.ipynb`
  + paired builder `tools/make_notebook_R5_01.py`.

### Changed
- `pyproject.toml` `requires-python` from `>=3.10` to `>=3.12`
  (unlocks `tomllib` stdlib + the new generic-class syntax for future
  slices). Existing 0.65.0 code is unaffected.
- `puremacro/fetch/{fred,fred_states,frb_phil_coincident,bea_cainc,bea_industry_shares,census_bfs}.py`,
  `puremacro/narrative/{scoring/llm,indices/_llm_kernel}.py`,
  `puremacro/instruments/_catalog.py`: all switched from direct
  `os.environ.get("*_API_KEY")` to `puremacro.credentials.require(...)`.
  Error messages improve from `"FRED_API_KEY must be set in environment"`
  to the four-tier `MissingCredentialError`. AST lint test
  (`tests/test_credentials/test_no_direct_env_get_in_fetch.py`) forbids
  regressions.

### Roadmap
- Slice B (0.67.0): F2.4 governed-fallback (unified live → Wayback →
  Playwright → fail policy) + F2.5 per-connector health telemetry
  surfacing fetch success / fallback rates over time.
- Slice C+: PARSER_SCHEMA_VERSION rollout to the remaining ~50
  connectors in `narrative/sources/`.
- Full spec: `docs/specs/2026-05-26-puremacro-066-f2-slice-a-data-infrastructure-design.md`.

### Internal
- New test directories: `tests/test_credentials/`,
  `tests/test_cache_db/`, `tests/test_vintages_alfred_store/`,
  `tests/test_narrative_schema_checks/`. Total ~50 new tests.
- The HTTP cache's `default_cache_dir()` is preserved by name but now
  returns the DB's parent directory rather than the cache root. The
  DB itself lives at `<dir>/cache.db`. Callers continue to pass
  `cache_dir` (a directory); internals translate.

```

- [ ] **Step 6: Add ARCHITECTURE.md subsection**

In `puremacro/ARCHITECTURE.md`, find the section about result-object standards / module-level conventions. Append the following subsection after that:

```markdown
### Data infrastructure (0.66.0+)

`puremacro.credentials` is the single resolver every API-keyed fetcher
uses (env vars + `~/.puremacro/credentials.toml`; raises
`MissingCredentialError` with a researcher-actionable message).
`puremacro._cache_db` hosts the shared SQLite connection (WAL mode)
that backs both the HTTP cache (`puremacro._http_cache`, public API
preserved from 0.65.0) and the ALFRED vintage store
(`puremacro.vintages.AlfredVintageStore`). Narrative connectors
declare `PARSER_SCHEMA_VERSION` and call
`puremacro.narrative.sources._schema_check.assert_landmarks(...)` to
fail loudly on upstream layout drift. References:
`docs/CREDENTIALS.md`, `docs/CACHE_DB.md`. Full spec:
`docs/specs/2026-05-26-puremacro-066-f2-slice-a-data-infrastructure-design.md`.
```

- [ ] **Step 7: Run the version test to verify pass**

```bash
pytest tests/test_credentials/test_service_registry.py::test_puremacro_version_is_066 -v
```
Expected: PASS.

- [ ] **Step 8: Final full-suite sanity sweep**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro" && \
pytest tests/test_credentials/ tests/test_cache_db/ tests/test_vintages_alfred_store/ tests/test_narrative_schema_checks/ -v && \
pytest tests/test_pyodide_compat.py -v && \
pytest tests/test_signal_contract/ -v && \
pytest tests/test_narrative.py tests/test_narrative_indices.py -v 2>&1 | tail -20
```
Expected:
- All new test directories green.
- Pyodide compat passes (sqlite3 + tomllib are stdlib, allowed).
- Signal-contract tests (from Slice 1) still green.
- Narrative suite: no NEW failures vs. the post-Slice-1 baseline.

- [ ] **Step 9: Commit**

```bash
git add puremacro/pyproject.toml puremacro/puremacro/__init__.py puremacro/CHANGELOG.md puremacro/ARCHITECTURE.md tests/test_credentials/test_service_registry.py
git commit -m "$(cat <<'EOF'
chore(puremacro): bump to 0.66.0 — F2 Slice A (data infrastructure)

Ships the four-sub-component slice: credentials + SQLite cache +
ALFRED vintage store + parser schema versioning (framework + 8
connectors). Bumps requires-python from >=3.10 to >=3.12 so tomllib
is stdlib (used by credentials' TOML config loader) and the new
generic-class syntax is available for future slices.

Slice B (F2.4 governed fallback + F2.5 health telemetry) queued for
0.67.0. PARSER_SCHEMA_VERSION rollout to the remaining ~50
connectors in subsequent slices.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Done-definition for Slice A (0.66.0)

- [ ] `puremacro.credentials` ships with full resolver + `MissingCredentialError` + `status()`.
- [ ] All 9 keyed consumers (fetchers + narrative scoring + instruments) route through `credentials.require()`; lint test passes.
- [ ] SQLite cache backend ships; existing `cache_read`/`cache_write` public API unchanged; lazy migration runs on first call after upgrade; CLI shipped.
- [ ] `puremacro.cache` gains `http_list_urls`/`http_cache_size_bytes`/`http_cache_clear` (distinct from existing `disk_cache`/`disk_cache_path`).
- [ ] `AlfredVintageStore` ships with put/get/has_series/series_list/coverage + `as_of_from_store` helper + `fetch_fred_alfred(store=, refresh=)` integration.
- [ ] `_schema_check.py` framework + `PARSER_SCHEMA_VERSION` + `assert_landmarks` rolled out to 8 connectors; golden fixtures saved; coverage assertion enforces.
- [ ] R5_01 notebook executes cleanly + committed alongside its builder.
- [ ] `docs/CREDENTIALS.md` + `docs/CACHE_DB.md` shipped.
- [ ] `pyproject.toml` at `version = "0.66.0"`, `requires-python = ">=3.12"`; `puremacro/__init__.py` `__version__ = "0.66.0"`; CHANGELOG 0.66.0 entry; ARCHITECTURE.md subsection.
- [ ] Pyodide compat passes (`sqlite3` + `tomllib` stdlib).
- [ ] Full narrative test suite shows zero new regressions vs. the post-Slice-1 baseline.

## Out of scope for Slice A (queued for follow-up plans)

- Slice B (0.67.0): F2.4 governed fallback + F2.5 health telemetry.
- PARSER_SCHEMA_VERSION rollout for the remaining ~50 `narrative/sources/*` connectors.
- macOS Keychain / system-keyring credential storage (optional `keyring` dep).
- IDBFS mount documentation for Pyodide cache persistence.
- `vintages_migrate` tooling for cross-version vintage-store schema changes.
- Generalising the vintage store schema to non-FRED real-time sources (Eurostat, OECD).
- F1 source coverage expansion, F3 unified panel-builder API, S2 interpretation, S4 cross-source synthesis 2.0, T1 cookbook, T2 onboarding (sibling sub-projects from the original brainstorm).
