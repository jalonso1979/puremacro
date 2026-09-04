"""Shim for puremacro.lp._garch_utils (deprecated)."""
from __future__ import annotations

import warnings

warnings.warn(
    "puremacro.lp.garch_utils is private and deprecated, will be removed in 2.0.0; "
    "use puremacro.lp._garch_utils or high-level LP functions instead.",
    FutureWarning,
    stacklevel=2,
)

from ._garch_utils import *  # noqa: F401, F403
