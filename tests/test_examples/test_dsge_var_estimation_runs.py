"""Smoke test: dsge_var_estimation.main() runs without exception and asserts estimation outputs."""
from __future__ import annotations

from pathlib import Path
from puremacro.dsge.dsge_var import DSGEVARResult


def test_dsge_var_estimation_main_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    from puremacro.examples.dsge_var_estimation import main

    res = main()
    captured = capsys.readouterr().out

    # 1. Output string assertions
    assert "PUREMACRO: DSGE-VAR ESTIMATION" in captured
    assert "OPTIMAL PRIOR WEIGHT" in captured
    assert "Log Marginal Data Density" in captured
    assert "Forecast Mean:" in captured
    assert "DSGE-VAR estimation completed successfully" in captured

    # 2. Result object and econometric assertions
    assert isinstance(res, DSGEVARResult)
    assert res.hat_lambda is not None
    assert 0.2 < res.hat_lambda < 5.0
    assert res.log_mdd_grid is not None
    assert len(res.log_mdd_grid) > 5
    assert res.B0.shape == (3, 3)
    assert res.T == 299

    # 3. Figure creation assertion
    out_png = tmp_path / "output" / "dsge_var_estimation.png"
    assert out_png.exists()
    assert out_png.stat().st_size > 1000
