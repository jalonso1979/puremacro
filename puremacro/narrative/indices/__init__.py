"""Text-derived continuous risk indices (EPU / MPU / GPR / tone / WUI / LUI / LTUI)."""
from .bluesky import bluesky_ui
from .epu import epu
from .mpu import mpu
from .gpr import gpr
from .tone import tone
from .wui import wui
from .lui import lui
from .ltui import ltui, ltui_up, ltui_down
from .lwui import lwui, lwui_wage
from .beige_book import bbui
from .us_executive import erpui, sotuui, cboui
from .eu_legislative import eurlex_ui, ep_ui
from ._lexicons import LEXICONS
from ._fed_districts import (
    FedDistrict, FED_DISTRICTS, DISTRICT_NAMES, DISTRICT_NUMBER,
    STATE_TO_DISTRICT, STATE_TO_DISTRICTS_FULL, SPLIT_STATES,
    TERRITORY_TO_DISTRICT, district_states, state_district,
    district_crosswalk,
)
from .cross_source import consensus_disagreement, GROUPS as CROSS_SOURCE_GROUPS

# Cluster B — NLP modernization (kernels available as opt-in primitives).
from ._embedding_kernel import (
    embedding_similarity_kernel,
    build_seed_prototype,
    make_sentence_transformer_embedder,
)
from ._mnl_kernel import mnl_kernel, canonicalize_weights
from ._llm_kernel import (
    llm_prob_kernel, LLMProvider, MockProvider, AnthropicProvider,
    LocalProvider, OllamaProvider, get_default_provider,
)

__all__ = [
    "bluesky_ui",
    "epu", "mpu", "gpr", "tone", "wui", "lui", "ltui", "ltui_up", "ltui_down", "lwui", "lwui_wage", "bbui",
    "cboui", "ep_ui", "erpui", "eurlex_ui", "sotuui",
    "LEXICONS",
    # Fed district crosswalk (BBUI district-level support)
    "FedDistrict", "FED_DISTRICTS", "DISTRICT_NAMES", "DISTRICT_NUMBER",
    "STATE_TO_DISTRICT", "STATE_TO_DISTRICTS_FULL", "SPLIT_STATES",
    "TERRITORY_TO_DISTRICT", "district_states", "state_district",
    "district_crosswalk",
    # Cross-source consensus & disagreement
    "consensus_disagreement", "CROSS_SOURCE_GROUPS",
    # B3 embeddings
    "embedding_similarity_kernel", "build_seed_prototype",
    "make_sentence_transformer_embedder",
    # B1 MNL
    "mnl_kernel", "canonicalize_weights",
    # B4 LLM
    "llm_prob_kernel", "LLMProvider", "MockProvider", "AnthropicProvider",
    "LocalProvider", "OllamaProvider", "get_default_provider",
]
