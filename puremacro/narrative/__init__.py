"""Narrative instruments for fiscal-policy IV identification.

Layered design:
    A) custom corpus → scoring → quarterly IV
    B) replication of canonical narrative datasets (Ramey, Romer-Romer,
       Mertens-Ravn, Cloyne, DGLP)
    C) end-to-end LLM pipeline that ingests text and emits events.

Public API
----------
Core types & aggregation:
    NarrativeEvent, NarrativeInstrument
    events_to_quarterly
    cluster_events, deduplicate, representative
    weak_iv_summary, compare_to, event_density
    HomogeneousFiscalPanel, DGLP_17

Canonical replication loaders + CSV-to-events helpers (re-exported
from :mod:`puremacro.narrative.replication` so callers don't reach into
private modules):
    load_ramey_2011_defense, ramey_csv_to_events
    load_romer_romer_2010, romer_romer_2010_csv_to_events
    load_mertens_ravn_2013, mertens_ravn_csv_to_events
    load_cloyne_2013_uk, cloyne_csv_to_events
    load_romer_romer_2017, romer_romer_2017_csv_to_events
    load_dglp_2011, dglp_csv_to_events
    load_devries_2011, devries_csv_to_events
    load_guajardo_2011, guajardo_csv_to_events
    load_mitchell_historical, mitchell_csv_to_events
    load_adler_2024, adler_csv_to_events
    load_eu_nms_2023, eu_nms_csv_to_events

Submodules ``scoring`` and ``sources`` expose backend-specific entry
points (``score_keyword``, ``score_llm``, ``local_csv``,
``us_federal_register``, …); import them directly when needed.
"""
from .types import NarrativeEvent, NarrativeInstrument, RiskIndex
from .aggregate import events_to_quarterly, index_to_quarterly
from .indices import epu, mpu, gpr, tone, wui, lui, ltui, ltui_up, ltui_down, lwui, lwui_wage, bluesky_ui, consensus_disagreement, CROSS_SOURCE_GROUPS
from .validate import weak_iv_summary, compare_to, event_density
from .dedup import cluster_events, deduplicate, representative
from .panel import HomogeneousFiscalPanel, DGLP_17

# Replication loaders + their CSV → events helpers.
from .replication import (
    load_ramey_2011_defense, ramey_csv_to_events,
    load_romer_romer_2010, romer_romer_2010_csv_to_events,
    load_mertens_ravn_2013, mertens_ravn_csv_to_events,
    load_cloyne_2013_uk, cloyne_csv_to_events,
    load_romer_romer_2017, romer_romer_2017_csv_to_events,
    load_dglp_2011, dglp_csv_to_events,
    load_devries_2011, devries_csv_to_events,
    load_guajardo_2011, guajardo_csv_to_events,
    load_mitchell_historical, mitchell_csv_to_events,
    load_adler_2024, load_eu_nms_2023, load_imf_covid_2022,
)
from .replication.adler_2024_consolidations import adler_csv_to_events
from .replication.eu_nms_2023_consolidations import eu_nms_csv_to_events
from .replication.imf_covid_2022_announcements import imf_covid_2022_csv_to_events

from .validation import gap_filler
from .validation.spec_curve import NarrativeSpecificationCurve
from .validation.report import NarrativeReport
from .validation.stability import NarrativeStabilityReport

from .topics import TfidfVectorizer, NMF, DynamicTopicModel, DynamicTopicModelResult
from .scoring.explain import shapley_keyword_attribution, explain_sentence_contributions
from .scoring.lexicons_es import SPANISH_MACRO_LEXICON, SpanishSentimentResult, score_spanish_macro_sentiment
from .anomalies import NarrativeBurstResult, detect_narrative_bursts
from .crowd_expert import CrowdExpertSpreadResult, compute_crowd_expert_spread

from .scoring.multilingual import (
    MULTILINGUAL_LEXICONS,
    MultilingualScoreResult,
    score_multilingual,
    infer_policy_stance,
)
from .quality.schedule_estimator import estimate_implementation_profile
from .classifier import PolicyActionClassifier
from .harvest import (
    SOURCE_REGISTRY,
    NarrativeDocument,
    NarrativeCorpus,
    harvest_narrative_corpus,
)

# β.1 event-source connectors are re-exported LAZILY (see __getattr__ below):
# each pulls scraper/network deps (bs4/requests) at import, so eager-importing
# them here would break `import puremacro.narrative` under Pyodide.
from .indices import bbui, cboui, ep_ui, erpui, eurlex_ui, sotuui

__all__ = [
    # Core
    "NarrativeEvent", "NarrativeInstrument", "RiskIndex",
    "events_to_quarterly", "index_to_quarterly",
    "weak_iv_summary", "compare_to", "event_density",
    "cluster_events", "deduplicate", "representative",
    "HomogeneousFiscalPanel", "DGLP_17",
    "bluesky_ui", "bbui", "cboui", "ep_ui", "erpui", "eurlex_ui", "sotuui",
    "consensus_disagreement", "CROSS_SOURCE_GROUPS",
    "epu", "mpu", "gpr", "tone", "wui", "lui", "ltui", "ltui_up", "ltui_down", "lwui", "lwui_wage",
    # Harvesting & Corpus
    "SOURCE_REGISTRY", "NarrativeDocument", "NarrativeCorpus", "harvest_narrative_corpus",
    # Multilingual & Classification
    "MULTILINGUAL_LEXICONS", "MultilingualScoreResult", "score_multilingual", "infer_policy_stance",
    "PolicyActionClassifier", "estimate_implementation_profile",
    # Topic modeling, anomalies & forum/discussion analytics
    "TfidfVectorizer", "NMF", "DynamicTopicModel", "DynamicTopicModelResult",
    "shapley_keyword_attribution", "explain_sentence_contributions",
    "SPANISH_MACRO_LEXICON", "SpanishSentimentResult", "score_spanish_macro_sentiment",
    "NarrativeBurstResult", "detect_narrative_bursts",
    "CrowdExpertSpreadResult", "compute_crowd_expert_spread",
    # Replication
    "load_ramey_2011_defense", "ramey_csv_to_events",
    "load_romer_romer_2010", "romer_romer_2010_csv_to_events",
    "load_mertens_ravn_2013", "mertens_ravn_csv_to_events",
    "load_cloyne_2013_uk", "cloyne_csv_to_events",
    "load_romer_romer_2017", "romer_romer_2017_csv_to_events",
    "load_dglp_2011", "dglp_csv_to_events",
    "load_devries_2011", "devries_csv_to_events",
    "load_guajardo_2011", "guajardo_csv_to_events",
    "load_mitchell_historical", "mitchell_csv_to_events",
    "load_adler_2024", "adler_csv_to_events",
    "load_eu_nms_2023", "eu_nms_csv_to_events",
    "load_imf_covid_2022", "imf_covid_2022_csv_to_events",
    # Validation
    "gap_filler",
    "NarrativeSpecificationCurve",
    "NarrativeReport",
    "NarrativeStabilityReport",
    # β.1 event-source connectors
    "BLUESKY_KNOWN_HANDLES", "iter_beige_book", "iter_bluesky_posts", "iter_cbo", "iter_ep_debates", "iter_erp", "iter_eurlex", "iter_frdb_strikes", "iter_sotu", "iter_us_warn",
]

# Lazy re-exports of the source connectors (see the note near the .indices import
# above): importing the narrative package stays browser-clean; the scraper deps
# load only when one of these names is actually accessed.
_LAZY_SOURCE_EXPORTS = frozenset({
    "iter_ep_debates", "iter_eurlex", "iter_us_warn", "iter_frdb_strikes",
    "iter_beige_book", "iter_cbo", "iter_erp", "iter_sotu",
    "iter_bluesky_posts", "BLUESKY_KNOWN_HANDLES",
})


def __getattr__(name):
    if name in _LAZY_SOURCE_EXPORTS:
        from . import sources
        value = getattr(sources, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
