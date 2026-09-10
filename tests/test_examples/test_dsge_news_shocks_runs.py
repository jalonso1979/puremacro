"""Smoke test: dsge_news_shocks.main() runs without exception and asserts news shock dynamics."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from puremacro.dsge.news import NewsDecompositionResult, NewsIRFResult


def test_dsge_news_shocks_main_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    from puremacro.examples.dsge_news_shocks import main

    res_news, decomp = main()
    captured = capsys.readouterr().out

    # 1. Output string assertions
    assert "PUREMACRO: DSGE NEWS & ANTICIPATED SHOCKS ENGINE" in captured
    assert "MATHEMATICAL & NUMERICAL VERIFICATION OF NEWS PROPERTIES" in captured
    assert "Predetermined State (a_t) Revision for t < 4" in captured
    assert "Immediate Forward-Looking Controls Jump at Announcement" in captured
    assert "Shock Realization at Date t = 4" in captured
    assert "Confirmed: All variance decomposition rows sum to 1.0000" in captured
    assert "News shocks analysis completed successfully" in captured

    # 2. Result object and macroeconomic property assertions
    assert isinstance(res_news, NewsIRFResult)
    assert isinstance(decomp, NewsDecompositionResult)
    assert res_news.lead == 4
    assert res_news.shock == "eps_a"

    # State predetermined check: a_t == 0 for t in [0, 1, 2, 3]
    np.testing.assert_allclose(res_news.irf.loc[:3, "a"].to_numpy(), 0.0, atol=1e-12)

    # Control jump at t=0
    assert abs(res_news.irf.loc[0, "pi"]) > 0.1

    # Shock realization at t=4
    np.testing.assert_allclose(res_news.irf.loc[4, "a"], 1.0, atol=1e-10)

    # Variance shares sum to 1.0
    np.testing.assert_allclose(decomp.variance_shares.sum(axis=1).to_numpy(), 1.0, atol=1e-10)

    # 3. Figure creation assertion
    out_png = tmp_path / "output" / "dsge_news_shocks.png"
    assert out_png.exists()
    assert out_png.stat().st_size > 1000
