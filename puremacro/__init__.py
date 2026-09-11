"""puremacro — Pyodide-compatible empirical macro toolbox.

See README.md for the install instructions (a local install is the
supported target) and the module-by-module overview. Public API is
curated in submodules; top-level entry points expose core panel builders
and the package version string.
"""
from .climate_panel import build_climate_panel
from .financial_panel import build_financial_panel

__version__ = "3.2.1"
__all__ = [
    "__version__",
    "build_climate_panel",
    "build_financial_panel",
]
