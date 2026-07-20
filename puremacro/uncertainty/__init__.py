"""T15 long-history uncertainty study primitives.

Cross-country variance decomposition, regime helpers, and per-country
LP wrappers for the GPR/EPU long-history workspace.
"""
from .decomposition import (
    between_within_share,
    rolling_between_within_share,
    correlation_with_clustering,
)
from .regimes import (
    CALENDAR_REGIMES,
    BreakResult,
    add_calendar_regime,
    bai_perron_breaks,
    state_indicator_smooth,
)
from .lp_long import (
    derive_innovation_shock,
    lp_per_country,
    pooled_with_dk_se,
    lp_state_dependent,
)
# Phase 5B absorbs (src/uncertainty/composite.py)
from .composite import (
    DEFAULT_PROXIES,
    build_backbone,
    build_composite,
    build_innovation,
)

__all__ = [
    "between_within_share",
    "rolling_between_within_share",
    "correlation_with_clustering",
    "CALENDAR_REGIMES",
    "BreakResult",
    "add_calendar_regime",
    "bai_perron_breaks",
    "state_indicator_smooth",
    "derive_innovation_shock",
    "lp_per_country",
    "pooled_with_dk_se",
    "lp_state_dependent",
    "DEFAULT_PROXIES",
    "build_backbone",
    "build_composite",
    "build_innovation",
]
