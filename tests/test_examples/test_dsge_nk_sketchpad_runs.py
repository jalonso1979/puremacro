"""Smoke test: dsge_nk_sketchpad.main() runs without exception.

The example asserts its own economics (a policy tightening raises the
nominal rate and lowers output and inflation; a sub-unity Taylor
coefficient breaks Blanchard-Kahn), so running it is a real check rather
than an import test.
"""
from __future__ import annotations


def test_dsge_nk_sketchpad_main_runs(capsys):
    from puremacro.examples.dsge_nk_sketchpad import main

    main()
    out = capsys.readouterr().out
    assert "unique stable solution" in out
    assert "BlanchardKahnError" in out
