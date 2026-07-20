"""Deterministic, pyodide-safe demo data shared by validation cases AND the
golden generators, so the puremacro computation and its frozen reference are
built from byte-identical inputs.
"""
from __future__ import annotations

import numpy as np


def var_demo_data() -> dict:
    """A stable bivariate VAR(1) sample. Seeded → identical every call."""
    rng = np.random.default_rng(20260531)
    T = 300
    A = np.array([[0.5, 0.1], [0.2, 0.4]])
    e = rng.standard_normal((T, 2)) * 0.5
    Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + e[t]
    return {"Y": Y, "p": 2, "horizon": 12}
