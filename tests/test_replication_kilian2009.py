"""Replication tests for ``puremacro.examples.kilian_2009_oil``.

Offline: reads the frozen FRED snapshot
``puremacro/replication/data/kilian2009.csv``. Assertions are qualitative
(Kilian 2009's signature rankings), so they survive a snapshot
regeneration; exact points are documented in the example docstring.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def kilian_result():
    from puremacro.examples.kilian_2009_oil import run_kilian_replication
    # Small bootstrap: only point IRFs / FEVD / HD are asserted.
    return run_kilian_replication(n_boot=20, seed=0)


def test_data_loads_offline():
    from puremacro.examples.kilian_2009_oil import load_kilian_data

    d = load_kilian_data()
    assert {"dprod", "rea", "rpo"} <= set(d.columns)
    assert len(d) > 400  # monthly, 1972-2007
    assert np.isfinite(d.to_numpy()).all()
    # Real price is demeaned by construction.
    assert abs(float(d["rpo"].mean())) < 1e-8


def test_demand_moves_price_more_than_supply(kilian_result):
    """Signature IRF ranking: peak |real-price response| to the aggregate-
    demand shock exceeds the response to the supply shock."""
    pk = kilian_result["price_peaks"]
    assert pk["agg_demand"] > pk["supply"], (
        f"expected aggregate demand to move the real price more than supply, "
        f"got {pk['agg_demand']:.2f} vs {pk['supply']:.2f}"
    )
    # Oil-specific (precautionary) demand moves it most of all on impact.
    assert pk["oil_specific"] > pk["supply"]


def test_fevd_real_price_dominated_by_demand(kilian_result):
    """Kilian Table: supply shocks explain little of the real-price
    forecast-error variance; demand shocks dominate."""
    fe = kilian_result["fevd"]
    supply, agg, spec = fe[12, 2, :]
    assert supply < 0.2
    assert supply < spec
    assert supply < agg + spec
    # Shares are a well-formed decomposition.
    assert np.all(fe >= -1e-12) and np.all(fe <= 1 + 1e-12)
    np.testing.assert_allclose(fe.sum(axis=2), 1.0, atol=1e-8)


def test_historical_decomposition_adds_up(kilian_result):
    """y_t = deterministic_t + sum_j contribution_jt, exactly."""
    assert kilian_result["hd_max_err"] < 1e-8


def test_2003_07_surge_driven_by_aggregate_demand(kilian_result):
    """THE headline: the 2003-07 real-price surge is attributed to global
    aggregate demand, not to supply shocks."""
    ep = kilian_result["episodes"]["2003-07 surge"]
    assert ep["total"] > 40.0  # a big surge, in log points
    assert ep["agg_demand"] > 20.0
    assert ep["agg_demand"] > abs(ep["supply"])
    assert ep["agg_demand"] > ep["oil_specific"]


def test_1990_gulf_war_spike_is_precautionary_demand(kilian_result):
    """Kilian's famous 1990 result: the Gulf-war price spike was mostly
    oil-specific (precautionary) demand, not lost supply."""
    ep = kilian_result["episodes"]["1990 Gulf war"]
    assert ep["oil_specific"] > abs(ep["supply"])
