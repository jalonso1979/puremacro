"""Automated multi-source narrative data harvester.

Provides an orchestrator for harvesting, normalizing, and incrementally caching
macro narrative documents across central banks, fiscal authorities, legislative
bodies, and multilateral organizations.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import pandas as pd

from .sources._extractors import extract_body
from .types import NarrativeEvent, RiskIndex


# ─────────────────────────────────────────────────────────────────────────────
# Source Registry & Metadata
# ─────────────────────────────────────────────────────────────────────────────

# Registry of source connectors:
# source_name -> (module_name, function_name, country, policy_domain, doc_type, language)
SOURCE_REGISTRY: dict[str, tuple[str, str, str, str, str, str]] = {
    # US Central Bank & Fiscal
    "fed_decision": ("fed_decision", "iter_fed_decision", "USA", "monetary", "decision", "en"),
    "fed_minutes": ("fed_minutes", "iter_fed_minutes", "USA", "monetary", "minutes", "en"),
    "fed_press_conf": ("fed_press_conf", "iter_fed_press_conf", "USA", "monetary", "press_conf", "en"),
    "fed_speeches": ("fed_speeches", "iter_fed_speeches", "USA", "monetary", "speech", "en"),
    "beige_book": ("beige_book", "iter_beige_book", "USA", "monetary", "report", "en"),
    "us_cbo": ("us_cbo", "iter_cbo", "USA", "fiscal", "report", "en"),
    "us_erp": ("us_erp", "iter_erp", "USA", "fiscal", "report", "en"),
    "us_sotu": ("us_sotu", "iter_sotu", "USA", "fiscal", "speech", "en"),
    "us_federal_register": ("us_federal_register", "iter_federal_register", "USA", "structural", "regulation", "en"),
    "us_dod_contracts": ("us_dod_contracts", "iter_dod_contracts", "USA", "fiscal", "procurement", "en"),
    "us_warn": ("us_warn", "iter_us_warn", "USA", "structural", "layoff_notice", "en"),
    "us_treasury": ("us_treasury", "iter_treasury_press", "USA", "fiscal", "press", "en"),
    # Euro Area / Europe
    "ecb_decision": ("ecb_decision", "iter_ecb_decision", "EA19", "monetary", "decision", "en"),
    "ecb_minutes": ("ecb_minutes", "iter_ecb_minutes", "EA19", "monetary", "minutes", "en"),
    "ecb_press_conf": ("ecb_press_conf", "iter_ecb_press_conf", "EA19", "monetary", "press_conf", "en"),
    "ecb_speeches": ("ecb_speeches", "iter_ecb_speeches", "EA19", "monetary", "speech", "en"),
    "boe_decision": ("boe_decision", "iter_boe_decision", "GBR", "monetary", "decision", "en"),
    "boe_minutes": ("boe_minutes", "iter_boe_minutes", "GBR", "monetary", "minutes", "en"),
    "boe_speeches": ("boe_speeches", "iter_boe_speeches", "GBR", "monetary", "speech", "en"),
    "uk_hmt": ("uk_hmt", "iter_hmt_press", "GBR", "fiscal", "press", "en"),
    "uk_obr": ("uk_obr", "iter_obr_publications", "GBR", "fiscal", "report", "en"),
    "de_bmf": ("de_bmf", "iter_bmf_press", "DEU", "fiscal", "press", "de"),
    "fr_tresor": ("fr_tresor", "iter_tresor_press", "FRA", "fiscal", "report", "fr"),
    "it_mef": ("it_mef", "iter_mef_press", "ITA", "fiscal", "press", "it"),
    "eu_ecfin": ("eu_ecfin", "iter_ecfin_press", "EU27", "fiscal", "report", "en"),
    "eu_eurlex": ("eu_eurlex", "iter_eurlex", "EU27", "structural", "legislation", "en"),
    "eu_parliament": ("eu_parliament", "iter_ep_debates", "EU27", "structural", "debate", "en"),
    "riksbank": ("riksbank", "iter_riksbank_decision", "SWE", "monetary", "press", "en"),
    "norges": ("norges", "iter_norges_decision", "NOR", "monetary", "press", "en"),
    # Asia-Pacific
    "boj_decision": ("boj_decision", "iter_boj_decision", "JPN", "monetary", "decision", "en"),
    "boj_speeches": ("boj_speeches", "iter_boj_speeches", "JPN", "monetary", "speech", "en"),
    "jp_mof": ("jp_mof", "iter_mof_press", "JPN", "fiscal", "press", "en"),
    "pboc": ("pboc", "iter_pboc_decision", "CHN", "monetary", "press", "en"),
    "rba": ("rba", "iter_rba_decision", "AUS", "monetary", "speech", "en"),
    "rbi": ("rbi", "iter_rbi_decision", "IND", "monetary", "press", "en"),
    "rbnz": ("rbnz", "iter_rbnz_decision", "NZL", "monetary", "press", "en"),
    "bok": ("bok", "iter_bok_decision", "KOR", "monetary", "press", "en"),
    "bi": ("bi", "iter_bi_decision", "IDN", "monetary", "press", "en"),
    "bnm": ("bnm", "iter_bnm_decision", "MYS", "monetary", "press", "en"),
    "bot": ("bot", "iter_bot_decision", "THA", "monetary", "press", "en"),
    "bsp": ("bsp", "iter_bsp_decision", "PHL", "monetary", "press", "en"),
    "mas": ("mas", "iter_mas_decision", "SGP", "monetary", "decision", "en"),
    # Latin America
    "banxico": ("banxico", "iter_banxico_decision", "MEX", "monetary", "decision", "es"),
    "bcb": ("bcb", "iter_bcb_decision", "BRA", "monetary", "decision", "pt"),
    "banrep": ("banrep", "iter_banrep_decision", "COL", "monetary", "minutes", "es"),
    "bccl": ("bccl", "iter_bccl_decision", "CHL", "monetary", "decision", "es"),
    "bcra": ("bcra", "iter_bcra_decision", "ARG", "monetary", "press", "es"),
    "ca_dof": ("ca_dof", "iter_dof_press", "CAN", "fiscal", "press", "en"),
    # Africa & Middle East
    "sarb": ("sarb", "iter_sarb_decision", "ZAF", "monetary", "speech", "en"),
    "cbe": ("cbe", "iter_cbe_decision", "EGY", "monetary", "press", "en"),
    "cbk": ("cbk", "iter_cbk_decision", "KEN", "monetary", "press", "en"),
    "cbn": ("cbn", "iter_cbn_decision", "NGA", "monetary", "press", "en"),
    # Multilateral
    "imf_articleiv": ("imf_articleiv", "iter_imf_articleiv", "WLD", "fiscal", "report", "en"),
    "imf_news": ("imf_news", "iter_imf_news", "WLD", "general", "news", "en"),
    "oecd_surveys": ("oecd_surveys", "iter_oecd_surveys", "WLD", "structural", "survey", "en"),
    "bis_speeches": ("bis_speeches", "iter_bis_speeches", "WLD", "monetary", "speech", "en"),
    # Commentary & social (no arguments needed; country comes from record metadata)
    "macro_blogs": ("macro_blogs", "iter_macro_blogs", "WLD", "general", "blog", "en"),
    "bluesky": ("bluesky", "iter_bluesky_posts", "WLD", "general", "social", "en"),
}

# Query-driven connectors: they need arguments (a search query, a subreddit, a
# file path), so a harvest runs them only when ``source_kwargs`` names them.
# Same tuple layout as SOURCE_REGISTRY.
QUERY_SOURCE_REGISTRY: dict[str, tuple[str, str, str, str, str, str]] = {
    "gdelt": ("news_api", "iter_gdelt_v2", "WLD", "general", "news", "en"),
    "google_news": ("google_news", "iter_google_news", "WLD", "general", "news", "en"),
    "reddit": ("reddit", "iter_reddit", "WLD", "general", "social", "en"),
    "hackernews": ("hackernews", "iter_hackernews", "WLD", "general", "social", "en"),
    "local_csv": ("local_csv", "iter_local_csv", "WLD", "general", "document", "en"),
}


# ─────────────────────────────────────────────────────────────────────────────
# NarrativeDocument Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NarrativeDocument:
    """A harvested and normalized narrative policy document."""
    doc_id: str
    source: str
    country: str
    date: pd.Timestamp
    url: str
    title: str
    text: str
    language: str
    policy_domain: str
    doc_type: str
    magnitude: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.date, pd.Timestamp):
            self.date = pd.Timestamp(self.date)
        if not self.doc_id:
            self.doc_id = hashlib.sha256(f"{self.source}:{self.url}:{self.date}".encode()).hexdigest()[:16]

    @property
    def token_count(self) -> int:
        return len(self.text.split())

    @property
    def content_hash(self) -> str:
        """SHA-256 of the document text; identifies a version of the document."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["date"] = self.date.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NarrativeDocument":
        clean = dict(d)
        clean["date"] = pd.Timestamp(clean["date"])
        return cls(**clean)


# ─────────────────────────────────────────────────────────────────────────────
# NarrativeCorpus Container
# ─────────────────────────────────────────────────────────────────────────────

_CORE_COLUMNS = [
    "doc_id", "source", "country", "date", "url", "title",
    "text", "language", "policy_domain", "doc_type", "magnitude",
]

# Prefix of metadata keys promoted to their own columns by
# ``to_frame(metadata="promote")``.
META_PREFIX = "meta_"


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, bool, int, float))


@dataclass
class NarrativeCorpus:
    """A collection of harvested narrative policy documents."""
    docs: list[NarrativeDocument] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.docs)

    def __iter__(self) -> Iterator[NarrativeDocument]:
        return iter(self.docs)

    @property
    def countries(self) -> list[str]:
        return sorted({d.country for d in self.docs})

    @property
    def sources(self) -> list[str]:
        return sorted({d.source for d in self.docs})

    @property
    def policy_domains(self) -> list[str]:
        return sorted({d.policy_domain for d in self.docs})

    @property
    def harvest_report(self) -> pd.DataFrame:
        """Per-source outcome of the harvest that built this corpus.

        Columns: ``source``, ``status`` (``ok`` / ``empty`` / ``error``),
        ``n_docs``, ``n_invalid`` (records rejected by the schema check) and
        ``error``. Empty when the corpus was not built by a harvest.
        """
        rows = self.metadata.get("harvest_report", [])
        return pd.DataFrame(rows, columns=["source", "status", "n_docs", "n_invalid", "error"])

    # ── construction from connector records ───────────────────────────────
    @classmethod
    def from_records(
        cls,
        records: Iterable[Any],
        *,
        source: str,
        country: str = "WLD",
        policy_domain: str = "general",
        doc_type: str = "document",
        language: str = "en",
        clean_html_bodies: bool = True,
    ) -> "NarrativeCorpus":
        """Build a corpus from any connector's output.

        Accepts the ``(date, text, url[, metadata[, magnitude]])`` records every
        ``iter_*`` function yields, so query-driven connectors and your own
        files enter the same corpus::

            from puremacro.narrative.sources import iter_reddit
            c = NarrativeCorpus.from_records(iter_reddit("economics"), source="reddit")

        The keyword arguments are defaults; per-record metadata (``country``,
        ``language``, ``doctype``/``doc_type``, ``policy_domain``, ``title``)
        overrides them.
        """
        defaults = (country, policy_domain, doc_type, language)
        docs, n_invalid = _records_to_docs(
            records, source, defaults, max_docs=None, clean_html_bodies=clean_html_bodies,
        )
        return cls(docs=docs, metadata={"sources": [source], "n_invalid": n_invalid})

    # ── tabular views ─────────────────────────────────────────────────────
    def to_frame(self, metadata: str = "dict") -> pd.DataFrame:
        """Convert corpus to a pandas DataFrame, one row per document.

        Parameters
        ----------
        metadata : {"dict", "json", "promote", "drop"}, default "dict"
            How to represent each document's free-form ``metadata``:

            - ``"dict"``: one ``metadata`` column holding the dicts (the
              historical behaviour).
            - ``"json"``: one ``metadata_json`` string column.
            - ``"promote"``: every key whose values are scalars in all
              documents becomes a ``meta_<key>`` column; the remaining keys go
              to ``metadata_json``. The frame is flat, so it writes to parquet,
              CSV or Stata without loss of the scalar fields.
            - ``"drop"``: no metadata column.

        A ``content_hash`` column (SHA-256 of the text) is always included.
        """
        if metadata not in ("dict", "json", "promote", "drop"):
            raise ValueError(
                f"metadata={metadata!r}; expected 'dict', 'json', 'promote' or 'drop'"
            )
        if not self.docs:
            cols = _CORE_COLUMNS + ["content_hash"]
            if metadata == "dict":
                cols.append("metadata")
            elif metadata in ("json", "promote"):
                cols.append("metadata_json")
            return pd.DataFrame(columns=cols)

        from .store import dumps_metadata

        rows = []
        for d in self.docs:
            row = {c: getattr(d, c) for c in _CORE_COLUMNS}
            row["date"] = d.date.isoformat()
            row["content_hash"] = d.content_hash
            rows.append(row)
        df = pd.DataFrame(rows)

        metas = [d.metadata for d in self.docs]
        if metadata == "dict":
            df["metadata"] = metas
        elif metadata == "json":
            df["metadata_json"] = [dumps_metadata(m) for m in metas]
        elif metadata == "promote":
            keys = sorted({k for m in metas for k in m})
            promoted = [k for k in keys if all(_is_scalar(m.get(k)) for m in metas)]
            for k in promoted:
                df[META_PREFIX + k] = [m.get(k) for m in metas]
            rest = set(keys) - set(promoted)
            df["metadata_json"] = [
                dumps_metadata({k: v for k, v in m.items() if k in rest}) for m in metas
            ]

        df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")
        df["date"] = _naive_if_all_naive(df["date"], self.docs)
        return df.sort_values(["date", "country"], kind="stable").reset_index(drop=True)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "NarrativeCorpus":
        """Inverse of :meth:`to_frame` for any of its ``metadata`` layouts."""
        docs = []
        meta_cols = [c for c in df.columns if c.startswith(META_PREFIX)]
        for rec in df.to_dict("records"):
            meta: dict[str, Any] = {}
            if isinstance(rec.get("metadata"), dict):
                meta.update(rec["metadata"])
            if isinstance(rec.get("metadata_json"), str) and rec["metadata_json"]:
                meta.update(json.loads(rec["metadata_json"]))
            for c in meta_cols:
                v = rec[c]
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    meta[c[len(META_PREFIX):]] = v
            mag = rec.get("magnitude")
            docs.append(NarrativeDocument(
                doc_id=rec["doc_id"], source=rec["source"], country=rec["country"],
                date=pd.Timestamp(rec["date"]), url=rec["url"], title=rec.get("title") or "",
                text=rec["text"], language=rec["language"],
                policy_domain=rec.get("policy_domain"), doc_type=rec.get("doc_type"),
                magnitude=None if mag is None or pd.isna(mag) else float(mag),
                metadata=meta,
            ))
        return cls(docs=docs)

    # ── file export ───────────────────────────────────────────────────────
    def to_jsonl(self, path: str | Path, *, manifest: bool = True) -> Path:
        """Write one JSON object per document (no third-party dependency).

        Metadata stays nested. With ``manifest=True`` a
        ``<stem>.manifest.json`` is written alongside (see :meth:`manifest`).
        """
        from .store import _json_default
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for d in self.docs:
                fh.write(json.dumps(d.to_dict(), default=_json_default, ensure_ascii=False))
                fh.write("\n")
        if manifest:
            self.write_manifest(_manifest_path(path))
        return path

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "NarrativeCorpus":
        """Read a file written by :meth:`to_jsonl`."""
        docs = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    docs.append(NarrativeDocument.from_dict(json.loads(line)))
        return cls(docs=docs)

    def to_parquet(
        self,
        path: str | Path,
        *,
        partition_cols: Sequence[str] | None = None,
        manifest: bool = True,
    ) -> Path:
        """Write the corpus as parquet (needs the ``io`` extra: pyarrow).

        Uses ``to_frame(metadata="promote")``, so scalar metadata fields are
        real columns and the rest is kept in ``metadata_json``. Pass
        ``partition_cols=["source"]`` for a partitioned dataset directory.
        """
        from .._optional import require_engines
        require_engines("parquet", feature="NarrativeCorpus.to_parquet")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_frame(metadata="promote")
        df.to_parquet(path, index=False,
                      partition_cols=list(partition_cols) if partition_cols else None)
        if manifest:
            self.write_manifest(_manifest_path(path))
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> "NarrativeCorpus":
        """Read a file or partitioned directory written by :meth:`to_parquet`."""
        from .._optional import require_engines
        require_engines("parquet", feature="NarrativeCorpus.from_parquet")
        df = pd.read_parquet(path)
        for c in df.columns:  # partition columns come back as categoricals
            if isinstance(df[c].dtype, pd.CategoricalDtype):
                df[c] = df[c].astype(str)
        return cls.from_frame(df)

    # ── reproducibility ───────────────────────────────────────────────────
    def manifest(self) -> dict[str, Any]:
        """Provenance record for this corpus.

        Holds the puremacro version, creation time, per-source counts, the
        harvest report when available, and one ``(doc_id, source, date, url,
        content_hash)`` entry per document. A replication package can ship the
        manifest instead of the (often non-redistributable) text: re-harvest,
        then check the result with :meth:`verify_manifest`.
        """
        from .. import __version__
        dates = [d.date for d in self.docs]
        return {
            "puremacro_version": __version__,
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "n_docs": len(self.docs),
            "date_min": min(dates).isoformat() if dates else None,
            "date_max": max(dates).isoformat() if dates else None,
            "sources": {s: sum(d.source == s for d in self.docs) for s in self.sources},
            "harvest": {k: v for k, v in self.metadata.items()
                        if k in ("harvest_params", "harvest_report")},
            "documents": [
                {"doc_id": d.doc_id, "source": d.source, "date": d.date.isoformat(),
                 "url": d.url, "content_hash": d.content_hash}
                for d in self.docs
            ],
        }

    def write_manifest(self, path: str | Path) -> Path:
        """Write :meth:`manifest` as JSON."""
        from .store import _json_default
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.manifest(), default=_json_default, indent=1),
                        encoding="utf-8")
        return path

    def verify_manifest(self, manifest: dict[str, Any] | str | Path) -> pd.DataFrame:
        """Compare this corpus against a manifest, document by document.

        Returns one row per manifest document with ``status``: ``match`` (same
        text), ``changed`` (same ``doc_id``, different text) or ``missing``.
        """
        if not isinstance(manifest, dict):
            manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        have = {d.doc_id: d.content_hash for d in self.docs}
        rows = []
        for e in manifest.get("documents", []):
            h = have.get(e["doc_id"])
            status = "missing" if h is None else ("match" if h == e["content_hash"] else "changed")
            rows.append({"doc_id": e["doc_id"], "source": e["source"], "date": e["date"],
                         "url": e["url"], "status": status})
        return pd.DataFrame(rows, columns=["doc_id", "source", "date", "url", "status"])

    # ── subsetting & summaries ────────────────────────────────────────────
    def filter(
        self,
        *,
        countries: Iterable[str] | None = None,
        sources: Iterable[str] | None = None,
        policy_domains: Iterable[str] | None = None,
        doc_types: Iterable[str] | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> "NarrativeCorpus":
        """Filter corpus by country, source, domain, or date range."""
        sub = self.docs
        if countries is not None:
            c_set = {c.upper() for c in countries}
            sub = [d for d in sub if d.country in c_set]
        if sources is not None:
            s_set = set(sources)
            sub = [d for d in sub if d.source in s_set]
        if policy_domains is not None:
            dom_set = set(policy_domains)
            sub = [d for d in sub if d.policy_domain in dom_set]
        if doc_types is not None:
            dt_set = set(doc_types)
            sub = [d for d in sub if d.doc_type in dt_set]
        if start_date is not None:
            sd = pd.Timestamp(start_date)
            sub = [d for d in sub if _comparable(d.date, sd) >= sd]
        if end_date is not None:
            ed = pd.Timestamp(end_date)
            sub = [d for d in sub if _comparable(d.date, ed) <= ed]

        return NarrativeCorpus(docs=sub, metadata={**self.metadata, "filtered": True})

    def summary(self) -> pd.DataFrame:
        """Return corpus descriptive statistics grouped by country and domain."""
        if not self.docs:
            return pd.DataFrame()
        df = self.to_frame()
        grp = df.groupby(["country", "policy_domain"])
        out = grp.agg(
            n_docs=("doc_id", "count"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            n_sources=("source", "nunique"),
        ).reset_index()
        return out

    # ── text as data (puremacro.text) ─────────────────────────────────────
    def segment(
        self,
        by: str | Callable[[str], Sequence[str]] = "paragraph",
        *,
        min_chars: int = 1,
    ) -> "NarrativeCorpus":
        """Split every document into paragraphs, sentences or custom units.

        Returns a corpus of segments. Each segment keeps its parent's source,
        country, date and other fields; ``doc_id`` is ``"<parent>:<position>"``
        and ``metadata`` gains ``parent_id``, ``position`` (0-based) and
        ``segment`` (the unit). ``by`` may be any ``callable(text) -> list[str]``.
        """
        from ..text.segment import get_splitter

        split = get_splitter(by)
        unit = by if isinstance(by, str) else getattr(by, "__name__", "custom")
        out: list[NarrativeDocument] = []
        for d in self.docs:
            pos = 0
            for seg in split(d.text):
                seg = seg.strip()
                if len(seg) < min_chars:
                    continue
                out.append(NarrativeDocument(
                    doc_id=f"{d.doc_id}:{pos}", source=d.source, country=d.country,
                    date=d.date, url=d.url, title=d.title, text=seg,
                    language=d.language, policy_domain=d.policy_domain,
                    doc_type=d.doc_type, magnitude=d.magnitude,
                    metadata={**d.metadata, "parent_id": d.doc_id,
                              "position": pos, "segment": unit},
                ))
                pos += 1
        return NarrativeCorpus(docs=out, metadata={**self.metadata, "segmented_by": unit})

    def dtm(
        self,
        *,
        tokenizer: Any = None,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        max_features: int | None = None,
        weighting: str = "count",
        metadata: str = "drop",
    ) -> Any:
        """Sparse document-term matrix of the corpus.

        Returns a :class:`puremacro.text.DocumentTermMatrix` whose ``docs``
        frame is :meth:`to_frame` without the text (``metadata`` is passed to
        it, ``"promote"`` adds the ``meta_*`` columns). Stop words follow each
        document's ``language``. ``weighting="tfidf"`` applies
        :meth:`~puremacro.text.DocumentTermMatrix.tfidf` with its defaults;
        call ``.tfidf(...)`` on a count matrix to choose the options.

        >>> m = corpus.dtm(ngram_range=(1, 2), min_df=5)        # doctest: +SKIP
        >>> panel = m.aggregate("country", freq="MS")           # doctest: +SKIP
        """
        from ..text.dtm import build_dtm

        if weighting not in ("count", "binary", "tfidf"):
            raise ValueError(f"weighting must be 'count', 'binary' or 'tfidf', got {weighting!r}")
        docs = self.to_frame(metadata=metadata).drop(columns=["text"])
        m = build_dtm(
            [d.text for d in self.docs], languages=[d.language for d in self.docs],
            docs=docs, tokenizer=tokenizer, ngram_range=ngram_range,
            min_df=min_df, max_df=max_df, max_features=max_features,
        )
        if weighting == "binary":
            return m.binary()
        if weighting == "tfidf":
            return m.tfidf()
        return m

    def to_events(
        self,
        classifier: Callable[[NarrativeDocument], NarrativeEvent | None] | None = None,
    ) -> list[NarrativeEvent]:
        """Classify corpus documents into discrete NarrativeEvents."""
        if classifier is None:
            from .classifier import PolicyActionClassifier
            classifier = PolicyActionClassifier().classify_document

        events: list[NarrativeEvent] = []
        for d in self.docs:
            try:
                ev = classifier(d)
                if ev is not None:
                    events.append(ev)
            except (ValueError, ArithmeticError, Exception):
                continue
        return events


def _comparable(ts: pd.Timestamp, ref: pd.Timestamp) -> pd.Timestamp:
    """Make ``ts`` comparable with ``ref`` when only one of them carries a timezone.

    RSS connectors return tz-aware dates and scraped pages naive ones; a
    tz-aware date is compared on its UTC wall clock against a naive bound.
    """
    if ts.tzinfo is not None and ref.tzinfo is None:
        return ts.tz_convert("UTC").tz_localize(None)
    if ts.tzinfo is None and ref.tzinfo is not None:
        return ts.tz_localize("UTC")
    return ts


def _naive_if_all_naive(col: pd.Series, docs: Sequence[NarrativeDocument]) -> pd.Series:
    """Keep the historical naive ``date`` dtype unless some document is tz-aware."""
    if all(d.date.tzinfo is None for d in docs):
        return col.dt.tz_localize(None)
    return col


def _manifest_path(path: Path) -> Path:
    return path.with_name(path.stem + ".manifest.json")


# ─────────────────────────────────────────────────────────────────────────────
# Record normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _records_to_docs(
    records: Iterable[Any],
    src_name: str,
    defaults: tuple[str, str, str, str],
    *,
    max_docs: int | None,
    clean_html_bodies: bool,
) -> tuple[list[NarrativeDocument], int]:
    """Validate connector records and turn them into documents.

    Returns ``(docs, n_invalid)``. Legacy 3-tuples ``(date, text, url)``, which
    the RSS/Atom helpers still yield, are accepted with empty metadata; any
    other malformed record is counted in ``n_invalid`` and skipped.
    """
    from .sources._schema import SourceRecordValidationError, _validate_one

    default_c, default_domain, default_dt, default_lang = defaults
    docs: list[NarrativeDocument] = []
    n_invalid = 0
    for i, rec in enumerate(records):
        if isinstance(rec, tuple) and len(rec) == 3:
            rec = (*rec, {})
        try:
            dt, txt, url, meta, mag = _validate_one(rec, ix=i)
        except SourceRecordValidationError:
            n_invalid += 1
            continue
        if not txt or pd.isna(dt):
            continue

        if clean_html_bodies and ("<html" in txt.lower() or "<div" in txt.lower()):
            txt = extract_body(txt, bank_code=src_name.upper())

        explicit_title = meta.get("title") or ""
        key = f"{src_name}:{url}:{dt}:{explicit_title or meta.get('id', '')}"
        docs.append(NarrativeDocument(
            doc_id=hashlib.sha256(key.encode()).hexdigest()[:16],
            source=src_name,
            country=meta.get("country") or default_c,
            date=dt,
            url=url,
            title=explicit_title or (txt[:80].strip() + "..."),
            text=txt,
            language=meta.get("language") or default_lang,
            policy_domain=meta.get("policy_domain") or default_domain,
            # Connectors write "doctype"; "doc_type" is accepted for symmetry.
            doc_type=meta.get("doc_type") or meta.get("doctype") or default_dt,
            magnitude=mag,
            metadata=meta,
        ))
        if max_docs is not None and len(docs) >= max_docs:
            break

    # Two different records with the same (source, url, date, title) key would
    # otherwise collapse into one document and its "revision". Disambiguate them
    # by text, keeping the first id unchanged.
    seen: dict[str, str] = {}
    for d in docs:
        prev = seen.get(d.doc_id)
        if prev is not None and prev != d.content_hash:
            d.doc_id = hashlib.sha256(f"{d.doc_id}:{d.content_hash}".encode()).hexdigest()[:16]
        seen.setdefault(d.doc_id, d.content_hash)
    return docs, n_invalid


def _resolve_target_sources(
    countries: Iterable[str] | str | None,
    sources: Iterable[str] | str | None,
    policy_domains: Iterable[str] | str | None,
) -> list[str]:
    """Filter target sources based on requested criteria."""
    target_sources: list[str] = []
    if sources is not None:
        requested = [sources] if isinstance(sources, str) else list(sources)
        unknown = [s for s in requested if s not in SOURCE_REGISTRY and s not in QUERY_SOURCE_REGISTRY]
        if unknown:
            raise ValueError(
                f"unknown narrative source(s) {unknown}; see SOURCE_REGISTRY "
                f"and QUERY_SOURCE_REGISTRY"
            )
        target_sources = [s for s in requested if s in SOURCE_REGISTRY]
    else:
        target_sources = list(SOURCE_REGISTRY.keys())

    if countries is not None:
        c_set = {countries.upper()} if isinstance(countries, str) else {c.upper() for c in countries}
        target_sources = [s for s in target_sources if SOURCE_REGISTRY[s][2] in c_set]

    if policy_domains is not None:
        p_set = {policy_domains} if isinstance(policy_domains, str) else set(policy_domains)
        target_sources = [s for s in target_sources if SOURCE_REGISTRY[s][3] in p_set]

    return target_sources


def _harvest_source(
    src_name: str,
    src_pkg: Any,
    max_docs_per_source: int | None,
    clean_html_bodies: bool,
    kwargs: dict[str, Any] | None = None,
) -> tuple[list[NarrativeDocument], dict[str, Any]]:
    """Run one connector. Returns its documents and a report row."""
    entry = SOURCE_REGISTRY.get(src_name) or QUERY_SOURCE_REGISTRY[src_name]
    mod_name, fn_name, *defaults = entry
    report = {"source": src_name, "status": "ok", "n_docs": 0, "n_invalid": 0, "error": None}
    try:
        mod = getattr(src_pkg, mod_name, None)
        if mod is None:
            import importlib
            mod = importlib.import_module(f"{src_pkg.__name__}.{mod_name}")
        fn = getattr(mod, fn_name)
        docs, n_invalid = _records_to_docs(
            fn(**(kwargs or {})), src_name, tuple(defaults),
            max_docs=max_docs_per_source, clean_html_bodies=clean_html_bodies,
        )
    except Exception as exc:  # noqa: BLE001 - one broken scraper must not stop the harvest
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
        return [], report
    report.update(n_docs=len(docs), n_invalid=n_invalid)
    if not docs:
        report["status"] = "empty"
    return docs, report


# ─────────────────────────────────────────────────────────────────────────────
# Harvest Orchestration Function
# ─────────────────────────────────────────────────────────────────────────────

def harvest_narrative_corpus(
    countries: Iterable[str] | str | None = None,
    sources: Iterable[str] | str | None = None,
    *,
    policy_domains: Iterable[str] | str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_docs_per_source: int | None = None,
    use_cache: bool = True,
    db_path: Path | None = None,
    store: Any = None,
    source_kwargs: dict[str, dict[str, Any]] | None = None,
    on_error: str = "warn",
    clean_html_bodies: bool = True,
) -> NarrativeCorpus:
    """Harvest narrative policy documents across central banks, treasuries, and agencies.

    Parameters
    ----------
    countries : ISO3 country codes to filter sources (e.g. ``["USA", "GBR", "DEU", "MEX"]``).
    sources : specific source names from :data:`SOURCE_REGISTRY` or
        :data:`QUERY_SOURCE_REGISTRY`.
    policy_domains : policy domain filters (``"fiscal"``, ``"monetary"``, ``"macropru"``, etc.).
    start_date : earliest publication date.
    end_date : latest publication date.
    max_docs_per_source : limit documents fetched per connector.
    use_cache : persist harvested documents in a :class:`~puremacro.narrative.store.CorpusStore`.
        Set ``False`` to keep the harvest in memory only.
    db_path : store file used when ``use_cache`` is true and ``store`` is not
        given. Defaults to :func:`~puremacro.narrative.store.default_corpus_path`.
    store : a :class:`~puremacro.narrative.store.CorpusStore` to write into
        (overrides ``db_path``).
    source_kwargs : per-source keyword arguments, e.g.
        ``{"reddit": {"subreddit": "economics"}, "fed_speeches": {...}}``.
        Query-driven connectors (:data:`QUERY_SOURCE_REGISTRY`) run only when
        named here.
    on_error : ``"warn"`` (default), ``"raise"`` or ``"ignore"`` when a
        connector fails. Failures are always recorded in
        :attr:`NarrativeCorpus.harvest_report`.
    clean_html_bodies : apply bank-specific body text extractors.

    Returns
    -------
    NarrativeCorpus
        The documents fetched by *this* call. The accumulated corpus across
        calls is ``store.load()``.
    """
    import time
    import warnings

    from . import sources as src_pkg

    if on_error not in ("warn", "raise", "ignore"):
        raise ValueError(f"on_error={on_error!r}; expected 'warn', 'raise' or 'ignore'")
    source_kwargs = dict(source_kwargs or {})
    unknown_kw = [s for s in source_kwargs
                  if s not in SOURCE_REGISTRY and s not in QUERY_SOURCE_REGISTRY]
    if unknown_kw:
        raise ValueError(f"source_kwargs names unknown source(s) {unknown_kw}")

    target_sources = _resolve_target_sources(countries, sources, policy_domains)
    requested = None if sources is None else ([sources] if isinstance(sources, str) else list(sources))
    for q in QUERY_SOURCE_REGISTRY:
        if q in source_kwargs and (requested is None or q in requested):
            target_sources.append(q)
        elif requested is not None and q in requested:
            raise ValueError(
                f"source {q!r} is query-driven; pass its arguments via "
                f"source_kwargs={{{q!r}: {{...}}}}"
            )

    started = int(time.time())
    harvested_docs: list[NarrativeDocument] = []
    report: list[dict[str, Any]] = []
    for src_name in target_sources:
        docs, row = _harvest_source(
            src_name, src_pkg, max_docs_per_source, clean_html_bodies,
            source_kwargs.get(src_name),
        )
        harvested_docs.extend(docs)
        report.append(row)

    failed = [r for r in report if r["status"] == "error"]
    if failed and on_error == "raise":
        lines = "; ".join(f"{r['source']}: {r['error']}" for r in failed)
        raise RuntimeError(f"{len(failed)} narrative source(s) failed: {lines}")
    if failed and on_error == "warn":
        warnings.warn(
            f"{len(failed)} of {len(report)} narrative sources failed "
            f"({', '.join(r['source'] for r in failed)}); see corpus.harvest_report",
            UserWarning, stacklevel=2,
        )

    params = {
        "countries": countries if countries is None or isinstance(countries, str) else list(countries),
        "sources": requested,
        "policy_domains": policy_domains if policy_domains is None or isinstance(policy_domains, str) else list(policy_domains),
        "start_date": start_date, "end_date": end_date,
        "max_docs_per_source": max_docs_per_source,
        "source_kwargs": source_kwargs,
    }
    corpus = NarrativeCorpus(docs=harvested_docs, metadata={
        "harvested_sources": target_sources,
        "harvest_params": params,
        "harvest_report": report,
    })
    if start_date or end_date:
        corpus = corpus.filter(start_date=start_date, end_date=end_date)

    if use_cache or store is not None:
        from .store import CorpusStore
        own = store is None
        st = store if store is not None else CorpusStore(db_path)
        try:
            counts = st.upsert(corpus.docs)
            st.record_run(started_at=started, params=params, report=report)
        finally:
            if own:
                st.close()
        corpus.metadata["store"] = str(st.path)
        corpus.metadata["store_counts"] = counts

    return corpus


__all__ = [
    "SOURCE_REGISTRY",
    "QUERY_SOURCE_REGISTRY",
    "NarrativeDocument",
    "NarrativeCorpus",
    "harvest_narrative_corpus",
]
