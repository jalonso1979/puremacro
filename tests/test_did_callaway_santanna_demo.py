"""DGP-recovery test for did_callaway_santanna_demo.

The demo simulates a staggered-treatment panel with a known
constant treatment effect tau = 1.0. The Callaway-Sant'Anna
overall ATT should recover this within wide tolerance at n=50
units × T=12 periods with seed=0.
"""
from __future__ import annotations

import pytest

from puremacro.examples import did_callaway_santanna_demo as demo


def test_demo_runs_without_error(capsys):
    """The demo's main() prints a non-empty report and exits 0."""
    demo.main()
    captured = capsys.readouterr()
    assert "Callaway-Sant'Anna" in captured.out
    assert "ATT (overall)" in captured.out


def test_run_demo_recovers_true_att():
    """run_demo() returns a dict with a CallawaySantannaResult whose
    att_overall is within ±0.5 of the planted tau=1.0."""
    out = demo.run_demo(seed=0)
    assert "cs_result" in out
    assert "sa_result" in out
    cs = out["cs_result"]
    sa = out["sa_result"]
    # Planted treatment effect is +1.0; both estimators should land near it.
    # Wide tolerance (±0.5) reflects the n=50 × T=12 small-sample noise band;
    # the realised CS / SA estimates with seed=0 are within ±0.02 of truth.
    assert abs(cs.att_overall - 1.0) < 0.5, f"CS att={cs.att_overall:.3f}"
    assert abs(sa.att_overall - 1.0) < 0.5, f"SA att={sa.att_overall:.3f}"
