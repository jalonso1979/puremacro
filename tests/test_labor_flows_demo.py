"""Smoke test for labor_flows_demo.

The demo uses the Shimer 2-state fallback path (no CPS-flows cache
needed), constructs a TransitionPanel from a simulated UNRATE series,
and runs supply_demand_decomposition.
"""
from __future__ import annotations

from puremacro.examples import labor_flows_demo as demo


def test_demo_runs_without_error(capsys):
    """main() executes end-to-end on the Shimer fallback path."""
    demo.main()
    captured = capsys.readouterr()
    assert "Labor-force transitions demo" in captured.out
    assert "job-finding" in captured.out.lower()


def test_run_demo_returns_panel_and_decomp():
    out = demo.run_demo(seed=0)
    assert "panel" in out
    assert "decomp" in out
    p = out["panel"]
    # Shimer path: monthly DataFrame should have p_UE and p_EU columns
    assert "p_UE" in p.monthly.columns
    assert "p_EU" in p.monthly.columns
    # Decomp must have demand and supply columns
    decomp = out["decomp"]
    assert any("demand" in c for c in decomp.columns)
