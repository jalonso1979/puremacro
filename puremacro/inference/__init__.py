"""Robust standard errors, bootstraps, weak-IV diagnostics, panel tests."""
# Load the compatibility submodule first so that a later
# ``import puremacro.inference.newey_west`` finds it in sys.modules and does
# not rebind this package attribute; the public callable then wins the name.
from . import newey_west as _newey_west_module  # noqa: F401
from .hac import newey_west_se as newey_west
from .bootstrap import residual_bootstrap as bootstrap_resample
from .block_bootstrap import block_bootstrap
from .moving_block import moving_block_bootstrap
from .wild_bootstrap import wild_bootstrap, wild_bootstrap_var
from .weak_iv import (
    cragg_donald_f,
    kleibergen_paap_f,
    olea_pflueger_f,
    anderson_rubin_test,
    msw_bands,
    instrument_influence,
)
from .pesaran import pesaran_cd
from .swamy import swamy_test
from .hac_fixed_b import hac_fixed_b, llsw_critical_value
from .over_id import hansen_j, stock_yogo_cv
from ._results import ARTestResult, LewbelIVResult, SupTBandResult
from .lewbel_iv import lewbel_iv
from .spec_curve import enumerate_specs, run_spec_curve, bootstrap_pvalue_median
from .supt import supt_band
from .simex import simex_ols, SIMEXResult
# statsmodels-parity helpers (2.6.0). Module names and exported names are
# distinct here, so no submodule/callable shadowing to work around.
from .multiple import multipletests, fdrcorrection
from .collinearity import vif
from .proportions import proportion_confint
from .diagnostics import durbin_watson
# Proxy-archive inference: effective sample size of smoothed proxy
# products, rotation / feasible-onset references, and the GLS matched filter on
# discrete ice-core samples. Exposed as a submodule; the headline callables
# are also re-exported here.
from typing import TYPE_CHECKING

from . import proxy
from .proxy import effective_sample_size, matched_filter, stacked_matched_filter
# Generic GMM. As with newey_west above, load the submodule first so that a
# later ``import puremacro.inference.gmm`` finds it in sys.modules and does
# not rebind this package attribute: ``puremacro.inference.gmm`` is the
# function.
from . import gmm as _gmm_module  # noqa: F401
from .gmm import gmm
from ._results import GMMFitResult
from .giv import GIVResult, estimate_giv

if TYPE_CHECKING:
    from .quandt_andrews import quandt_andrews_supF as quandt_andrews_supF
    from .smc import (
        AdaptiveTemperingSchedule as AdaptiveTemperingSchedule,
        SMCResult as SMCResult,
        smc_sample as smc_sample,
    )

_LAZY_SMC = ("AdaptiveTemperingSchedule", "SMCResult", "smc_sample")


# quandt_andrews_supF and SMC symbols are loaded lazily via PEP 562:
# - quandt_andrews depends on puremacro.tests.breaks (avoids namespace shadow);
# - smc depends on puremacro.dsge.priors, so deferring it keeps puremacro.lp /
#   puremacro.inference imports decoupled from the 73-module DSGE & trade graphs.
def __getattr__(name: str):
    if name == "quandt_andrews_supF":
        from .quandt_andrews import quandt_andrews_supF
        globals()["quandt_andrews_supF"] = quandt_andrews_supF
        return quandt_andrews_supF
    if name in _LAZY_SMC:
        from . import smc as _smc_mod
        for attr in _LAZY_SMC:
            globals()[attr] = getattr(_smc_mod, attr)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

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
    "instrument_influence",
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
    "simex_ols",
    "SIMEXResult",
    "multipletests",
    "fdrcorrection",
    "vif",
    "proportion_confint",
    "durbin_watson",
    "proxy",
    "effective_sample_size",
    "matched_filter",
    "stacked_matched_filter",
    "AdaptiveTemperingSchedule",
    "SMCResult",
    "smc_sample",
    "gmm",
    "GMMFitResult",
    "GIVResult",
    "estimate_giv",
]
