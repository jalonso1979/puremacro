"""Real-time (vintage) macro data across statistical agencies.

A *vintage* is one published edition of a series. Statistical offices
revise, so every reference period has a sequence of editions, and the
difference between the first and the last is the revision that
news-vs-noise tests are about.

Public surface
--------------
- :func:`vintage_panel`      — cross-country panel of editions
- :func:`alfred_vintages`    — every archived edition of one ALFRED series
- :class:`VintagePanel`      — the container, with the revision helpers
- :func:`vintage_catalog`    — which series backs which (country, variable)
- :data:`VINTAGE_SEMANTICS`  — what each provider's vintage date means

Read :data:`VINTAGE_SEMANTICS` before using a vintage date as an event
date: some providers stamp the national release, others the date the
archive ingested it.
"""
from __future__ import annotations

from ._base import (
    VINTAGE_COLUMNS,
    VINTAGE_SEMANTICS,
    VintagePanel,
    available_providers,
    normalize_vintage_frame,
    provider_countries,
    register_provider,
)
from .catalog import (
    CANONICAL_VARIABLES,
    canonical_variable,
    providers_for,
    resolve_series,
    vintage_catalog,
)
from .alfred import alfred_vintages, parse_alfredgraph_csv


def _register_all() -> None:
    """Register every bundled provider.

    Import order matters only in that each module's ``_register`` adds
    its own catalogue entries; providers are otherwise independent.
    """
    from . import alfred as _alfred
    from . import bundesbank as _bundesbank
    from . import banxico as _banxico
    from . import ecb_rtd as _ecb_rtd
    from . import oecd_stes as _oecd_stes
    from . import ons as _ons
    from . import statcan as _statcan
    for mod in (_oecd_stes, _alfred, _bundesbank, _ons, _statcan,
                _ecb_rtd, _banxico):
        try:
            mod._register()
        except Exception as exc:                          # pragma: no cover
            import warnings
            warnings.warn(
                f"puremacro.fetch.realtime: provider {mod.__name__} failed to "
                f"register: {exc}", UserWarning, stacklevel=2,
            )


_register_all()

from .panel import DEFAULT_PROVIDER_ORDER, SUPPORTED_FREQUENCIES, vintage_panel
from .seasonal import (
    SEASONAL_F_MIN,
    SEASONAL_MIN_OBS,
    SEASONAL_RANGE_MIN,
    SEASONAL_WINDOW,
    drop_unadjusted_editions,
    seasonal_signature,
)


__all__ = [
    # container + shape
    "VintagePanel", "VINTAGE_COLUMNS", "VINTAGE_SEMANTICS",
    "normalize_vintage_frame",
    # entry points
    "vintage_panel", "alfred_vintages", "parse_alfredgraph_csv",
    # catalog
    "vintage_catalog", "canonical_variable", "CANONICAL_VARIABLES",
    "resolve_series", "providers_for",
    # registry
    "available_providers", "provider_countries", "register_provider",
    "DEFAULT_PROVIDER_ORDER", "SUPPORTED_FREQUENCIES",
    # seasonality screen
    "seasonal_signature", "drop_unadjusted_editions",
    "SEASONAL_F_MIN", "SEASONAL_RANGE_MIN", "SEASONAL_MIN_OBS", "SEASONAL_WINDOW",
]
