"""Automated multi-source narrative data harvester.

Provides an orchestrator for harvesting, normalizing, and incrementally caching
macro narrative documents across central banks, fiscal authorities, legislative
bodies, and multilateral organizations.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import pandas as pd

from .sources._extractors import extract_body
from .sources._schema import validate_records
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
    "us_federal_register": ("us_federal_register", "iter_us_federal_register", "USA", "structural", "regulation", "en"),
    "us_dod_contracts": ("us_dod_contracts", "iter_us_dod_contracts", "USA", "fiscal", "procurement", "en"),
    "us_warn": ("us_warn", "iter_us_warn", "USA", "structural", "layoff_notice", "en"),
    # Euro Area / Europe
    "ecb_decision": ("ecb_decision", "iter_ecb_decision", "EA19", "monetary", "decision", "en"),
    "ecb_minutes": ("ecb_minutes", "iter_ecb_minutes", "EA19", "monetary", "minutes", "en"),
    "ecb_press_conf": ("ecb_press_conf", "iter_ecb_press_conf", "EA19", "monetary", "press_conf", "en"),
    "ecb_speeches": ("ecb_speeches", "iter_ecb_speeches", "EA19", "monetary", "speech", "en"),
    "boe_decision": ("boe_decision", "iter_boe_decision", "GBR", "monetary", "decision", "en"),
    "boe_minutes": ("boe_minutes", "iter_boe_minutes", "GBR", "monetary", "minutes", "en"),
    "boe_speeches": ("boe_speeches", "iter_boe_speeches", "GBR", "monetary", "speech", "en"),
    "uk_hmt": ("uk_hmt", "iter_hmt_press", "GBR", "fiscal", "press", "en"),
    "uk_obr": ("uk_obr", "iter_obr_releases", "GBR", "fiscal", "report", "en"),
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

    def to_frame(self) -> pd.DataFrame:
        """Convert corpus to a pandas DataFrame."""
        if not self.docs:
            return pd.DataFrame(columns=[
                "doc_id", "source", "country", "date", "url", "title",
                "text", "language", "policy_domain", "doc_type", "magnitude"
            ])
        rows = [d.to_dict() for d in self.docs]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values(["date", "country"]).reset_index(drop=True)

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
            sub = [d for d in sub if d.date >= sd]
        if end_date is not None:
            ed = pd.Timestamp(end_date)
            sub = [d for d in sub if d.date <= ed]

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
            except Exception:
                continue
        return events


# ─────────────────────────────────────────────────────────────────────────────
# Incremental Harvesting Storage
# ─────────────────────────────────────────────────────────────────────────────

_DDL_HARVEST_TABLE = """
CREATE TABLE IF NOT EXISTS harvested_narrative_docs (
    doc_id         TEXT PRIMARY KEY,
    source         TEXT NOT NULL,
    country        TEXT NOT NULL,
    date           TEXT NOT NULL,
    url            TEXT NOT NULL,
    title          TEXT,
    text_hash      TEXT NOT NULL,
    language       TEXT NOT NULL,
    policy_domain  TEXT,
    doc_type       TEXT,
    metadata_json  TEXT,
    fetched_at     INTEGER NOT NULL
);
"""


def _get_harvest_conn(db_path: Path | None = None) -> sqlite3.Connection:
    from .. import _cache_db
    conn = _cache_db.get_conn(db_path)
    conn.execute(_DDL_HARVEST_TABLE)
    return conn


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
    clean_html_bodies: bool = True,
) -> NarrativeCorpus:
    """Harvest narrative policy documents across central banks, treasuries, and agencies.

    Parameters
    ----------
    countries : ISO3 country codes to filter sources (e.g. ``["USA", "GBR", "DEU", "MEX"]``).
    sources : specific source names from :data:`SOURCE_REGISTRY`.
    policy_domains : policy domain filters (``"fiscal"``, ``"monetary"``, ``"macropru"``, etc.).
    start_date : earliest publication date.
    end_date : latest publication date.
    max_docs_per_source : limit documents fetched per connector.
    use_cache : persist and retrieve from local SQLite cache.
    clean_html_bodies : apply bank-specific body text extractors.

    Returns
    -------
    NarrativeCorpus
        Collection of normalized NarrativeDocument objects.
    """
    from . import sources as src_pkg

    # Filter target sources
    target_sources: list[str] = []
    if sources is not None:
        requested = [sources] if isinstance(sources, str) else list(sources)
        target_sources = [s for s in requested if s in SOURCE_REGISTRY]
    else:
        target_sources = list(SOURCE_REGISTRY.keys())

    if countries is not None:
        c_set = {countries.upper()} if isinstance(countries, str) else {c.upper() for c in countries}
        target_sources = [s for s in target_sources if SOURCE_REGISTRY[s][2] in c_set]

    if policy_domains is not None:
        p_set = {policy_domains} if isinstance(policy_domains, str) else set(policy_domains)
        target_sources = [s for s in target_sources if SOURCE_REGISTRY[s][3] in p_set]

    harvested_docs: list[NarrativeDocument] = []

    for src_name in target_sources:
        mod_name, fn_name, default_c, default_domain, default_dt, default_lang = SOURCE_REGISTRY[src_name]
        try:
            mod = getattr(src_pkg, mod_name, None)
            if mod is None:
                continue
            fn = getattr(mod, fn_name, None)
            if fn is None or not callable(fn):
                continue

            raw_iter = fn()
            validated_recs = validate_records(raw_iter, strict=False)

            count = 0
            for rec in validated_recs:
                # rec is (date, text, url, meta, magnitude)
                dt, txt, url, meta, mag = rec
                if not txt:
                    continue

                if clean_html_bodies and ("<html" in txt.lower() or "<div" in txt.lower()):
                    txt = extract_body(txt, bank_code=src_name.upper())

                c = meta.get("country") or default_c
                domain = meta.get("policy_domain") or default_domain
                dt_type = meta.get("doc_type") or default_dt
                lang = meta.get("language") or default_lang
                title = meta.get("title") or (txt[:80].strip() + "...")

                doc = NarrativeDocument(
                    doc_id="",
                    source=src_name,
                    country=c,
                    date=dt,
                    url=url,
                    title=title,
                    text=txt,
                    language=lang,
                    policy_domain=domain,
                    doc_type=dt_type,
                    magnitude=mag,
                    metadata=meta,
                )
                harvested_docs.append(doc)
                count += 1
                if max_docs_per_source is not None and count >= max_docs_per_source:
                    break
        except Exception:
            continue

    corpus = NarrativeCorpus(docs=harvested_docs, metadata={"harvested_sources": target_sources})
    if start_date or end_date:
        corpus = corpus.filter(start_date=start_date, end_date=end_date)

    return corpus


__all__ = [
    "SOURCE_REGISTRY",
    "NarrativeDocument",
    "NarrativeCorpus",
    "harvest_narrative_corpus",
]
