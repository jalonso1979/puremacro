"""Smoke test for shock_atlas_demo.

The demo calls load_all_shocks(root) and prints a registry-coverage
table. Some shocks may not have cached data; the demo must report
those as 'missing' without raising.
"""
from __future__ import annotations

from puremacro.examples import shock_atlas_demo as demo


def test_demo_runs_without_error(capsys):
    demo.main()
    captured = capsys.readouterr()
    assert "shock atlas" in captured.out.lower()
    assert "registry" in captured.out.lower()


def test_run_demo_returns_dict_and_summary():
    out = demo.run_demo()
    assert "loaded" in out
    assert "summary" in out
    summary = out["summary"]
    # summary should be a DataFrame with at least one row per registered shock
    assert len(summary) >= 1
    assert "category" in summary.columns
