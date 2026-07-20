"""Regime-switching extensions of the VAR.

- ``ms_var_fit`` : Markov-switching VAR (Hamilton filter + Kim smoother + EM).
- ``tvar_fit``    : self-exciting threshold VAR (Tsay 1998).
- ``tvecm_fit``   : threshold VECM (Hansen-Seo 2002, simplified).
- ``girf``        : Koop-Pesaran-Potter (1996) generalized IRF for all three.
"""
from .girf import RegimeGIRFResult, girf
from .ms_var import ms_var_fit
from .threshold import tvar_fit
from .tvecm import tvecm_fit

__all__ = ["ms_var_fit", "tvar_fit", "tvecm_fit", "girf", "RegimeGIRFResult"]
