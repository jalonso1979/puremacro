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
