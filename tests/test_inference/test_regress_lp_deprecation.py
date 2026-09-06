"""puremacro.regress.lp deprecation notice (audit garch-vol-inference:
'regress.lp_panel deprecation message').

Before the fix the FutureWarning said the module 'will be removed in 2.0.0'
while the package was at 2.3.0 and the module docstring says retirement is
deferred to a future release.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import puremacro
from puremacro.regress.lp import lp_panel


def _panel(n_units=4, T=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        shock = rng.standard_normal(T)
        y = np.cumsum(0.3 * shock + 0.5 * rng.standard_normal(T))
        rows.append(pd.DataFrame({"unit": i, "date": np.arange(T), "y": y, "shock": shock}))
    return pd.concat(rows, ignore_index=True)


def test_deprecation_message_does_not_name_a_past_release():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lp_panel(_panel(), y="y", shock="shock", horizons=range(0, 3))
    msgs = [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]
    assert msgs, "lp_panel must emit a FutureWarning"
    msg = msgs[0]
    assert "2.0.0" not in msg
    major = int(puremacro.__version__.split(".")[0])
    assert f"removed in {major}." not in msg
    assert "deferred" in msg
    assert "puremacro.lp.panel.panel_lp" in msg
