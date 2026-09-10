"""Smoke test: dsge_optimal_discretion.main() runs without exception and asserts economic outcomes."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from puremacro.dsge._results import DiscretionaryPolicyResult, PolicyResult


def test_dsge_optimal_discretion_main_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    from puremacro.examples.dsge_optimal_discretion import main

    res_disc, res_comm = main()
    captured = capsys.readouterr().out

    # 1. Output string assertions
    assert "PUREMACRO: OPTIMAL MONETARY POLICY" in captured
    assert "OPTIMAL POLICY REGIME: DISCRETION" in captured
    assert "WELFARE BIAS QUANTIFICATION" in captured
    assert "Theoretical Inflation Bias" in captured
    assert "Quantified Inflation Bias" in captured
    assert "Quantified Stabilization Bias" in captured
    assert "Optimal policy analysis completed successfully" in captured

    # 2. Result object and economic metric assertions
    assert isinstance(res_disc, DiscretionaryPolicyResult)
    assert isinstance(res_comm, PolicyResult)
    assert res_disc.converged is True
    assert res_disc.diff < 1e-9
    assert res_disc.inflation_bias > 0.0
    assert res_disc.stabilization_bias > 0.0
    assert np.isclose(res_disc.inflation_bias, 0.024752, atol=1e-4)
    assert res_disc.loss > res_comm.loss

    # 3. Figure creation assertion
    out_png = tmp_path / "output" / "dsge_optimal_discretion.png"
    assert out_png.exists()
    assert out_png.stat().st_size > 1000
