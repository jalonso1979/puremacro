"""Smoke test: dsge_fertility_demo.main() runs without exception."""
from __future__ import annotations

import warnings


def test_dsge_fertility_demo_main_runs_without_exception(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from puremacro.examples.dsge_fertility_demo import main
    # Suppress the over-determined-BGP RuntimeWarning from solve_bgp.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
    assert (tmp_path / "dsge_fertility_demo.png").exists()
