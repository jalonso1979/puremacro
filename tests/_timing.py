"""Wall-clock budgets for timing assertions.

Budgets are written for a developer machine. Shared CI runners are several
times slower and noisier (Windows runners measured 0.56 s against a 0.15 s
budget), so CI sets ``PUREMACRO_TIMING_SCALE`` to widen every budget at once.
A scaled budget still catches the non-terminating loops these tests guard.
"""
from __future__ import annotations

import os


def budget(seconds: float) -> float:
    return seconds * float(os.environ.get("PUREMACRO_TIMING_SCALE", "1"))
