"""External-CSV instruments — public-data sources outside the
literature/replication ecosystem.

Each provider exposes a generic ``load(*, series_id, ...) -> Instrument``
function. They are also registered in
:mod:`puremacro.instruments._catalog` (alongside narrative replications,
connectors, monetary HFI, and literature shocks) so callers can reach
them via :func:`puremacro.instruments.load` by registry key.
"""
from .fred import load as load_fred
from .bis import load as load_bis
from .imf_weo import load as load_imf_weo

__all__ = ["load_fred", "load_bis", "load_imf_weo"]
