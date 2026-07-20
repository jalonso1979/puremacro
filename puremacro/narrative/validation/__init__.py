"""Validation tools for narrative instruments."""
from . import gap_filler
from .spec_curve import NarrativeSpecificationCurve
from .report import NarrativeReport
from .stability import NarrativeStabilityReport

# The external-benchmark fetchers download reference series over the network
# (``external_benchmarks`` imports ``requests`` at module level), so they are
# exposed LAZILY: importing this package stays Pyodide-clean and ``requests``
# loads only when a fetcher is actually accessed.
_LAZY_EXTERNAL_BENCHMARKS = frozenset({
    "fetch_bbd_ai_epu_monthly",
    "fetch_caldara_iacoviello_gpr_monthly",
    "fetch_ahir_bloom_furceri_wui_panel",
    "fetch_bbd_country_epu_monthly",
})

__all__ = [
    "gap_filler",
    "NarrativeSpecificationCurve",
    "NarrativeReport",
    "NarrativeStabilityReport",
    "fetch_bbd_ai_epu_monthly",
    "fetch_caldara_iacoviello_gpr_monthly",
    "fetch_ahir_bloom_furceri_wui_panel",
    "fetch_bbd_country_epu_monthly",
]


def __getattr__(name):
    if name in _LAZY_EXTERNAL_BENCHMARKS:
        from . import external_benchmarks
        value = getattr(external_benchmarks, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
