"""Smoke test: dsge_particle_filter_sv.main() runs without exception and asserts SV particle filter dynamics."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from puremacro.dsge.particle_filter import ParticleFilterResult


def test_dsge_particle_filter_sv_main_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    from puremacro.examples.dsge_particle_filter_sv import main

    res_sv, res_const = main()
    captured = capsys.readouterr().out

    # 1. Output string assertions
    assert "PUREMACRO: NONLINEAR DSGE PARTICLE FILTER WITH STOCHASTIC VOLATILITY" in captured
    assert "Compiling DSGE model and solving 2nd-order pruned perturbation" in captured
    assert "Filter with Stochastic Volatility" in captured
    assert "Filter with Constant Volatility Baseline" in captured
    assert "Statistical & Economic Comparison" in captured
    assert "Particle filter simulation completed successfully" in captured

    # 2. Result object & mathematical/numerical assertions
    assert isinstance(res_sv, ParticleFilterResult)
    assert isinstance(res_const, ParticleFilterResult)

    assert np.isfinite(res_sv.log_likelihood)
    assert np.isfinite(res_const.log_likelihood)

    # Filtered state shapes
    assert res_sv.filtered_states.shape == (32, 3)
    assert res_const.filtered_states.shape == (32, 3)

    # Volatility tracking
    assert res_sv.filtered_volatility is not None
    assert res_sv.filtered_volatility.shape == (32, 1)
    assert "sigma_eps" in res_sv.filtered_volatility.columns
    assert np.all(res_sv.filtered_volatility.to_numpy() > 0)
    assert res_const.filtered_volatility is None

    # ESS diagnostics
    assert float(res_sv.ess.mean()) > 10.0
    assert 0.0 <= res_sv.resampling_frequency <= 1.0

    # 3. Figure creation assertion
    out_png = tmp_path / "output" / "dsge_particle_filter_sv.png"
    assert out_png.exists()
    assert out_png.stat().st_size > 1000
