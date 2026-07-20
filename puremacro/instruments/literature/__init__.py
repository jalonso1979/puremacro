"""Literature shock instruments — canonical identified-shock series from the
empirical macro literature, wrapped as :class:`puremacro.instruments.Instrument`.

Each loader exposes a top-level ``load(...) -> Instrument`` function. They
are also registered in :mod:`puremacro.instruments._catalog` (alongside
the narrative replications and connectors) so callers can reach them via
:func:`puremacro.instruments.load` by registry key.
"""
from .bloom_2009 import load as load_bloom_2009
from .bbd_epu import load as load_bbd_epu
from .caldara_iacoviello_gpr import load as load_caldara_iacoviello_gpr
from .romer_romer_2004 import load as load_romer_romer_2004

__all__ = [
    "load_bloom_2009",
    "load_bbd_epu",
    "load_caldara_iacoviello_gpr",
    "load_romer_romer_2004",
]
