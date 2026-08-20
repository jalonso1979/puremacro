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
    }), encoding="utf-8")


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
    (fresh_db / f"{bad_key}.json").write_text("{ not valid json", encoding="utf-8")
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
