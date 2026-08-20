"""Runtime adaptation: what this machine can do, and how much to ask of it.

The rest of `puremacro` is written against one implicit machine — a
workstation with sockets, pyarrow, threads and RAM. This subpackage is
where the package finds out that it is somewhere else (an iPad running
Juno, a browser kernel, a CI container) and adapts, without any
estimator changing behaviour behind your back.

Four pieces:

:mod:`~puremacro.runtime._capabilities`
    Detect host, device class, and the four capabilities that actually
    break on a tablet: sockets, parquet, threads, writable filesystem.
    Start with ``print(runtime.report())``.

:mod:`~puremacro.runtime.budget`
    Ceilings on bootstrap draws / posterior draws / grid sizes /
    simulation length, sized to the device. Opt-in per call — nothing is
    clamped unless you ask.

:mod:`~puremacro.runtime.transport`
    HTTP for environments with no sockets.
    ``runtime.enable_browser_network()`` routes the whole existing
    ``fetch`` layer through the browser's own networking.

:mod:`~puremacro.runtime.store`
    DataFrame ⇄ npz codec, so data has a portable on-disk form that does
    not need pyarrow. :mod:`puremacro.pocket` builds on it.

Note: ``runtime.capabilities`` is the *function*; the submodule of
that name stays importable as ``puremacro.runtime._capabilities``.
"""
from puremacro.runtime._capabilities import (
    Capabilities,
    capabilities,
    is_pyodide,
    is_tablet,
    refresh,
    report,
)
from puremacro.runtime.budget import (
    Budget,
    BudgetWarning,
    budgeted,
    fit,
    override,
)
from puremacro.runtime.budget import current as current_budget
from puremacro.runtime.transport import (
    TransportError,
    disable_browser_network,
    enable_browser_network,
)
from puremacro.runtime.transport import available as transport_available
from puremacro.runtime import store

__all__ = [
    # capabilities
    "Capabilities", "capabilities", "refresh", "report",
    "is_pyodide", "is_tablet",
    # budget
    "Budget", "BudgetWarning", "current_budget", "fit", "budgeted", "override",
    # transport
    "TransportError", "transport_available",
    "enable_browser_network", "disable_browser_network",
    # storage
    "store",
]
