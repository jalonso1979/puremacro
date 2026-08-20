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
    }), encoding="utf-8")


def _repo_root() -> Path:
    # Repo root = first ancestor holding pyproject.toml (the split puremacro
    # repo keeps pyproject.toml and tools/ side by side at its root).
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("no repo root")


def test_cli_dry_run_does_not_apply(tmp_path, monkeypatch):
    _make_flat_entry(tmp_path, "https://example.com/a", b"alpha")
    cli = _repo_root() / "tools" / "cache_migrate.py"
    import os
    env = {**os.environ, "PUREMACRO_HTTP_CACHE_DIR": str(tmp_path)}
    out = subprocess.run(
        [sys.executable, str(cli)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, check=True,
    )
    # Files still present after dry-run.
    assert list(tmp_path.glob("*.bin"))
    assert "1" in out.stdout  # reports 1 entry would be migrated


def test_cli_apply_migrates(tmp_path):
    _make_flat_entry(tmp_path, "https://example.com/a", b"alpha")
    cli = _repo_root() / "tools" / "cache_migrate.py"
    import os
    env = {**os.environ, "PUREMACRO_HTTP_CACHE_DIR": str(tmp_path)}
    subprocess.run(
        [sys.executable, str(cli), "--apply"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, check=True,
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
    import os
    env = {**os.environ, "PUREMACRO_HTTP_CACHE_DIR": str(tmp_path)}
    subprocess.run(
        [sys.executable, str(cli), "--apply", "--rm"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, check=True,
    )
    assert list(tmp_path.glob("*.bin")) == []
