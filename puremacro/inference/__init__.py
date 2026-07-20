"""Robust standard errors, bootstraps, weak-IV diagnostics, panel tests."""
from .hac import newey_west_se as newey_west
from .bootstrap import residual_bootstrap as bootstrap_resample
from .block_bootstrap import block_bootstrap
from .moving_block import moving_block_bootstrap
from .wild_bootstrap import wild_bootstrap, wild_bootstrap_var
from .weak_iv import cragg_donald_f, kleibergen_paap_f, olea_pflueger_f, anderson_rubin_test, msw_bands
from .pesaran import pesaran_cd
from .swamy import swamy_test
from .hac_fixed_b import hac_fixed_b, llsw_critical_value
from .over_id import hansen_j, stock_yogo_cv
from ._results import ARTestResult, LewbelIVResult, SupTBandResult
from .lewbel_iv import lewbel_iv
from .spec_curve import enumerate_specs, run_spec_curve, bootstrap_pvalue_median
from .supt import supt_band

# quandt_andrews_supF is loaded lazily because quandt_andrews.py depends on
# puremacro.tests.breaks, which can fail to resolve when the project root is
# on sys.path and creates a namespace-package shadow (e.g. pytest rootdir
# injection).  Use __getattr__ (PEP 562) so the import is deferred until first
# access, at which point sys.path is clean.
def __getattr__(name: str):
    if name == "quandt_andrews_supF":
        from .quandt_andrews import quandt_andrews_supF
        globals()["quandt_andrews_supF"] = quandt_andrews_supF
        return quandt_andrews_supF
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "newey_west",
    "bootstrap_resample",
    "block_bootstrap",
    "moving_block_bootstrap",
    "wild_bootstrap",
    "wild_bootstrap_var",
    "cragg_donald_f",
    "kleibergen_paap_f",
    "olea_pflueger_f",
    "anderson_rubin_test",
    "msw_bands",
    "pesaran_cd",
    "swamy_test",
    "hac_fixed_b",
    "llsw_critical_value",
    "hansen_j",
    "stock_yogo_cv",
    "ARTestResult",
    "LewbelIVResult",
    "lewbel_iv",
    "quandt_andrews_supF",
    "enumerate_specs",
    "run_spec_curve",
    "bootstrap_pvalue_median",
    "supt_band",
    "SupTBandResult",
]
