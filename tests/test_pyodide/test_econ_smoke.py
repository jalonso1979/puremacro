"""Pyodide-contract smoke: the econometrics subsystems used by the browser
playground demos import cleanly and run on synthetic data WITHOUT pulling in any
forbidden (non-Pyodide) module. Marked pyodide_smoke so the in-browser Gate 6
runner (tools/pyodide/runner.js) also executes it.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

_FORBIDDEN = ("statsmodels", "linearmodels", "arch", "bs4", "requests", "numba")


@pytest.mark.pyodide_smoke
def test_playground_econ_subsystems_are_pyodide_clean():
    # Measure the DELTA: a forbidden module imported by an unrelated test earlier
    # in the same process must not be misattributed to these subsystems.
    before = set(sys.modules)
    from puremacro.var.identify.cholesky import cholesky_svar
    from puremacro.lp.jorda import lp_hac
    from puremacro.garch import garch11_fit
    from puremacro.gar import qar
    from puremacro.did import callaway_santanna

    rng = np.random.default_rng(0)

    # SVAR on a tiny synthetic VAR
    Y = rng.standard_normal((120, 3)).cumsum(0)
    svar = cholesky_svar(Y, p=2, horizon=8, n_boot=50, ci=0.9, seed=0)
    assert svar.irf_point.shape == (9, 3, 3)

    # LP-HAC
    df = pd.DataFrame({"y": Y[:, 0], "shock": rng.standard_normal(120)})
    lp = lp_hac(df, y="y", x="shock", horizons=range(0, 6), n_lags=2)
    assert {"h", "beta", "lo", "hi"}.issubset(lp.columns)

    # GARCH(1,1)
    g = garch11_fit(pd.Series(rng.standard_normal(300)))
    assert g.persistence < 1.0

    # Quantile-AR
    q = qar(rng.standard_normal(200), quantiles=(0.5,), horizons=(1,), p=1,
            n_boot=20, seed=0)
    assert len(q) >= 1

    # Callaway-Sant'Anna on a tiny staggered panel
    rows = []
    for i in range(40):
        gi = (5, 8, np.nan)[i % 3]
        for t in range(1, 11):
            treated = 1.0 if (not pd.isna(gi)) and t >= gi else 0.0
            rows.append({"unit": i, "time": t,
                         "y": float(i % 5) + 0.1 * t + treated + 0.05 * rng.standard_normal(),
                         "treat_time": gi})
    cs = callaway_santanna(pd.DataFrame(rows), unit="unit", time="time",
                           outcome="y", treat_time="treat_time", n_boot=50, seed=0)
    assert np.isfinite(cs.att_overall)

    bad = [m for m in _FORBIDDEN if m in sys.modules and m not in before]
    assert bad == [], f"forbidden modules imported: {bad}"
