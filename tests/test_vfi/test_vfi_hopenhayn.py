from __future__ import annotations

import numpy as np

from puremacro.vfi.examples import hopenhayn_equilibrium


def test_hopenhayn_free_entry_and_selection():
    out = hopenhayn_equilibrium(n_z=51)
    assert abs(out["equilibrium"].residual) < 1e-6      # free entry clears
    assert out["price"] > 0.0
    # exit is a productivity threshold; some firms exit, some survive
    surv = out["equilibrium"].survive
    assert np.all(np.diff(surv.astype(int)) >= 0)
    assert surv[-1] and not surv[0]
    # selection: incumbents more productive than entrants on average
    assert out["mean_incumbent_productivity"] > out["mean_entrant_productivity"]
    np.testing.assert_allclose(out["distribution"].sum(), 1.0, atol=1e-10)
    assert 0.0 < out["exit_rate"] < 1.0


def test_hopenhayn_higher_entry_cost_raises_price():
    lo = hopenhayn_equilibrium(n_z=51, ce=40.0)
    hi = hopenhayn_equilibrium(n_z=51, ce=80.0)
    assert hi["price"] > lo["price"]


def test_hopenhayn_exported():
    from puremacro.vfi import hopenhayn_equilibrium as fn

    assert fn is hopenhayn_equilibrium
