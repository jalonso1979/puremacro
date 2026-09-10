"""Smoke test: dsge_markov_switching.main() runs without exception and asserts MS-DSGE equilibrium & GIRF."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from puremacro.dsge.markov_switching import MSDSGEResult


def test_dsge_markov_switching_main_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    from puremacro.examples.dsge_markov_switching import main

    res, girf = main()
    captured = capsys.readouterr().out

    # 1. Output string assertions
    assert "PUREMACRO: MARKOV-SWITCHING DSGE" in captured
    assert "Specifying Structural System and Policy Regimes" in captured
    assert "Solving Coupled Quadratic Matrix Equations via Block Newton-Raphson" in captured
    assert "Convergence Status : CONVERGED" in captured
    assert "Mean-Square Stable : True" in captured
    assert "Ergodic Distribution & Long-Run Moments" in captured
    assert "Computing Closed-Form Analytical GIRFs" in captured
    assert "Markov-switching DSGE analysis completed successfully" in captured

    # 2. Result object and mathematical properties assertions
    assert isinstance(res, MSDSGEResult)
    assert isinstance(girf, pd.DataFrame)
    assert res.converged
    assert res.mean_square_stable
    assert res.spectral_radius_mss < 1.0
    assert res.spectral_radius_mean < 1.0

    # Ergodic distribution
    np.testing.assert_allclose(res.ergodic_distribution["Hawkish"], 2.0 / 3.0, atol=1e-4)
    np.testing.assert_allclose(res.ergodic_distribution["Dovish"], 1.0 / 3.0, atol=1e-4)

    # GIRF shape (horizon 16 + impact 0 = 17 steps) and economic signs
    assert girf.shape == (17, 3)
    assert girf.loc[0, "interest_rate"] > 0
    assert girf.loc[0, "output_gap"] < 0
    assert girf.loc[0, "inflation"] < 0

    # Reversion toward ergodic mean (0)
    assert abs(girf.loc[16, "interest_rate"]) < abs(girf.loc[0, "interest_rate"])
    assert abs(girf.loc[16, "output_gap"]) < abs(girf.loc[0, "output_gap"])

    # 3. Figure creation assertion
    out_png = tmp_path / "output" / "dsge_markov_switching.png"
    assert out_png.exists()
    assert out_png.stat().st_size > 1000
