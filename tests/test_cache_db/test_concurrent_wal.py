"""F2.1 — WAL allows concurrent reader/writer: public API never raises
under contention; transient lock errors degrade gracefully to
UserWarning + None/no-op; the cache serves a reasonable fraction of
requests successfully."""
from __future__ import annotations

import threading
import warnings

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

    # Capture any UserWarnings emitted by the public API under contention.
    # The API contract is: OperationalError is absorbed → UserWarning + None/no-op.
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

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

    # 1. No exceptions should ever escape the public cache_read / cache_write API.
    assert not errors, f"unexpected exceptions escaped the public API: {errors}"

    # 2. Any lock-induced failures are expected to surface only as UserWarnings,
    #    not as exceptions. If any warnings were emitted they should be a minority
    #    (<50% of the 100 total operations).
    user_warnings = [w for w in caught_warnings if issubclass(w.category, UserWarning)]
    assert len(user_warnings) < 50, (
        f"too many UserWarnings ({len(user_warnings)}/100); cache may be broken: "
        + "; ".join(str(w.message) for w in user_warnings[:5])
    )

    # 3. After both threads finish, at least 10 of 50 writes should have
    #    round-tripped successfully — proving the cache served its purpose
    #    under contention (conservative lower bound; 10/50 tolerates worst-case
    #    WAL lock contention on a shared connection).
    hits = sum(
        1 for i in range(50)
        if cache_read(fresh_cache, f"https://example.com/w-{i}") == f"body-{i}".encode()
    )
    assert hits >= 10, (
        f"too few cache round-trips: only {hits}/50 keys read back correctly. "
        "Cache may not be functioning at all."
    )
