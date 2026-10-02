"""Persistent SQLite store for harvested narrative documents.

A :class:`CorpusStore` keeps one row per document with its full text, so a
corpus accumulates across harvests instead of living only in the Python session
that scraped it. Most connectors only expose the latest few dozen items of an
RSS feed, so the accumulated store, not any single harvest, is the research
corpus.

The store is its own SQLite file (default ``narrative_corpus.db`` next to the
HTTP cache, see :func:`default_corpus_path`), not a table in the HTTP cache:
clearing the cache must never delete a corpus.

Revisions
---------
A document is keyed by ``doc_id``. When a later harvest returns the same
``doc_id`` with different text, only the latest text is kept; the row is
flagged (``n_revisions``, ``revised_at``) and the hash of the text it replaced
is kept in ``previous_content_hash``, so edited documents can be found with
:meth:`CorpusStore.revised` without storing every version.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import pandas as pd

if TYPE_CHECKING:
    from .harvest import NarrativeCorpus, NarrativeDocument


SCHEMA_VERSION = 1

_DDL = (
    """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id            TEXT PRIMARY KEY,
        source            TEXT NOT NULL,
        country           TEXT NOT NULL,
        date              TEXT NOT NULL,
        url               TEXT NOT NULL,
        title             TEXT,
        text              TEXT NOT NULL,
        language          TEXT NOT NULL,
        policy_domain     TEXT,
        doc_type          TEXT,
        magnitude         REAL,
        metadata_json     TEXT,
        content_hash      TEXT NOT NULL,
        first_fetched_at  INTEGER NOT NULL,
        last_fetched_at   INTEGER NOT NULL,
        n_revisions       INTEGER NOT NULL DEFAULT 0,
        revised_at        INTEGER,
        previous_content_hash TEXT,
        puremacro_version TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS documents_source_date_idx ON documents(source, date);",
    "CREATE INDEX IF NOT EXISTS documents_country_date_idx ON documents(country, date);",
    """
    CREATE TABLE IF NOT EXISTS harvest_runs (
        run_id            INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at        INTEGER NOT NULL,
        finished_at       INTEGER NOT NULL,
        params_json       TEXT,
        report_json       TEXT,
        puremacro_version TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        component TEXT PRIMARY KEY,
        version   INTEGER NOT NULL
    );
    """,
)


def default_corpus_path() -> Path:
    """Default store location: ``narrative_corpus.db`` beside the HTTP cache.

    Follows ``$PUREMACRO_HTTP_CACHE_DIR`` the same way the cache does, so tests
    and sandboxes that redirect the cache redirect the corpus too.
    """
    from .. import _cache_db
    return _cache_db.default_db_path().parent / "narrative_corpus.db"


def _version() -> str:
    from .. import __version__
    return __version__


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return str(obj)


def dumps_metadata(meta: dict[str, Any]) -> str:
    """Serialise a metadata dict; non-JSON values (timestamps, sets) become strings."""
    return json.dumps(meta, default=_json_default, ensure_ascii=False, sort_keys=True)


class CorpusStore:
    """SQLite-backed store of :class:`~puremacro.narrative.harvest.NarrativeDocument`.

    Parameters
    ----------
    path : str or Path, optional
        Database file. Defaults to :func:`default_corpus_path`. ``":memory:"``
        gives a throwaway in-memory store.

    Examples
    --------
    >>> store = CorpusStore("fomc.db")                       # doctest: +SKIP
    >>> harvest_narrative_corpus(sources=["fed_minutes"], store=store)  # doctest: +SKIP
    >>> corpus = store.load(start_date="2015-01-01")         # doctest: +SKIP
    """

    def __init__(self, path: str | Path | None = None):
        if path is None:
            path = default_corpus_path()
        self.path = path if str(path) == ":memory:" else Path(path)
        if isinstance(self.path, Path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=30.0)
        for stmt in _DDL:
            self._conn.execute(stmt)
        self._conn.execute(
            "INSERT OR IGNORE INTO schema_version (component, version) VALUES (?, ?)",
            ("narrative_corpus", SCHEMA_VERSION),
        )
        self._conn.commit()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CorpusStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def __repr__(self) -> str:
        return f"CorpusStore({str(self.path)!r}, n_docs={len(self)})"

    # ── writes ─────────────────────────────────────────────────────────────
    def upsert(
        self,
        docs: Iterable["NarrativeDocument"],
        *,
        fetched_at: int | None = None,
    ) -> dict[str, int]:
        """Insert new documents and update changed ones.

        Returns counts ``{"new": n, "updated": n, "unchanged": n}``. A document
        is *updated* when its ``doc_id`` exists with a different
        ``content_hash``: the new text replaces the old one and the row's
        revision flag is set.
        """
        ts = int(time.time()) if fetched_at is None else int(fetched_at)
        version = _version()
        counts = {"new": 0, "updated": 0, "unchanged": 0}
        cur = self._conn.cursor()
        for d in docs:
            h = d.content_hash
            meta_json = dumps_metadata(d.metadata)
            row = cur.execute(
                "SELECT content_hash FROM documents WHERE doc_id = ?", (d.doc_id,)
            ).fetchone()
            if row is None:
                cur.execute(
                    "INSERT INTO documents (doc_id, source, country, date, url, title, "
                    "text, language, policy_domain, doc_type, magnitude, metadata_json, "
                    "content_hash, first_fetched_at, last_fetched_at, puremacro_version) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (d.doc_id, d.source, d.country, d.date.isoformat(), d.url, d.title,
                     d.text, d.language, d.policy_domain, d.doc_type, d.magnitude,
                     meta_json, h, ts, ts, version),
                )
                counts["new"] += 1
            elif row[0] == h:
                cur.execute(
                    "UPDATE documents SET last_fetched_at = ? WHERE doc_id = ?",
                    (ts, d.doc_id),
                )
                counts["unchanged"] += 1
            else:
                cur.execute(
                    "UPDATE documents SET title = ?, text = ?, magnitude = ?, "
                    "metadata_json = ?, content_hash = ?, last_fetched_at = ?, "
                    "n_revisions = n_revisions + 1, revised_at = ?, "
                    "previous_content_hash = ?, puremacro_version = ? WHERE doc_id = ?",
                    (d.title, d.text, d.magnitude, meta_json, h, ts, ts, row[0],
                     version, d.doc_id),
                )
                counts["updated"] += 1
        self._conn.commit()
        return counts

    def record_run(
        self,
        *,
        started_at: int,
        params: dict[str, Any],
        report: list[dict[str, Any]],
    ) -> int:
        """Log one harvest call; returns its ``run_id``."""
        cur = self._conn.execute(
            "INSERT INTO harvest_runs (started_at, finished_at, params_json, "
            "report_json, puremacro_version) VALUES (?, ?, ?, ?, ?)",
            (int(started_at), int(time.time()),
             json.dumps(params, default=_json_default, sort_keys=True),
             json.dumps(report, default=_json_default), _version()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    # ── reads ──────────────────────────────────────────────────────────────
    def load(
        self,
        *,
        countries: Iterable[str] | None = None,
        sources: Iterable[str] | None = None,
        policy_domains: Iterable[str] | None = None,
        doc_types: Iterable[str] | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> "NarrativeCorpus":
        """Return the stored documents (current versions) as a corpus."""
        from .harvest import NarrativeCorpus, NarrativeDocument

        where: list[str] = []
        args: list[Any] = []
        for col, vals, upper in (
            ("country", countries, True),
            ("source", sources, False),
            ("policy_domain", policy_domains, False),
            ("doc_type", doc_types, False),
        ):
            if vals is not None:
                vals = [vals] if isinstance(vals, str) else list(vals)
                if upper:
                    vals = [v.upper() for v in vals]
                where.append(f"{col} IN ({','.join('?' * len(vals))})")
                args.extend(vals)
        sql = (
            "SELECT doc_id, source, country, date, url, title, text, language, "
            "policy_domain, doc_type, magnitude, metadata_json FROM documents"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        docs = []
        for r in self._conn.execute(sql, args):
            docs.append(NarrativeDocument(
                doc_id=r[0], source=r[1], country=r[2], date=pd.Timestamp(r[3]),
                url=r[4], title=r[5] or "", text=r[6], language=r[7],
                policy_domain=r[8], doc_type=r[9], magnitude=r[10],
                metadata=json.loads(r[11]) if r[11] else {},
            ))
        # Date bounds are compared as Timestamps, not ISO strings, so mixed
        # timezone offsets in stored dates cannot mis-order.
        corpus = NarrativeCorpus(docs=docs, metadata={"store": str(self.path)})
        if start_date is not None or end_date is not None:
            corpus = corpus.filter(start_date=start_date, end_date=end_date)
        return corpus

    def revised(self) -> pd.DataFrame:
        """Documents whose text changed after they were first stored.

        One row per document: ``n_revisions``, ``revised_at`` (unix time of
        the latest change) and ``previous_content_hash`` (hash of the text it
        replaced). Only the latest text itself is stored.
        """
        return pd.read_sql_query(
            "SELECT doc_id, source, country, date, url, title, n_revisions, "
            "first_fetched_at, revised_at, previous_content_hash, content_hash "
            "FROM documents WHERE n_revisions > 0 ORDER BY revised_at",
            self._conn,
        )

    def runs(self) -> pd.DataFrame:
        """Harvest log: one row per :func:`harvest_narrative_corpus` call."""
        return pd.read_sql_query(
            "SELECT run_id, started_at, finished_at, params_json, report_json, "
            "puremacro_version FROM harvest_runs ORDER BY run_id",
            self._conn,
        )


__all__ = ["CorpusStore", "default_corpus_path", "SCHEMA_VERSION"]
