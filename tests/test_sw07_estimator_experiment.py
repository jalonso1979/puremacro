"""Paired SW07 controls preserve estimators, independent phases and failures."""
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
from puremacro.structural import sw07_estimator_experiment as module
from puremacro.structural import sw07_finite_sample as original
from puremacro.structural.empirical_sw07 import (
    ALL_MOMENTS, FIT_MOMENTS, UNITS, _PUBLISHED_MODE, _labels,
    covariance_moment_targets, load_empirical_sw07_data, sw07_covariance_moments,
)
from puremacro.structural.sw07_sampling import simulate_sw07_sample, sw07_finite_sample_moments


CALIBRATION = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS, **_PUBLISHED_MODE}
SMOKE_SEED = 8931


@pytest.fixture(scope="module")
def phases():
    return tuple(module.run_sw07_estimator_experiment(replications=2, phase=phase, seed=SMOKE_SEED)
                 for phase in ("calibration", "validation"))


def _optimizer_result(reasons=()):
    return SimpleNamespace(theta=np.array([.81, .24]), objective=2., success=True,
        message="test optimizer result", n_evals=7,
        diagnostics={"inference_unavailable_reasons": list(reasons)},
        inference_valid=not reasons, boundary=np.zeros(2, dtype=bool),
        identification_rank=2, j_pvalue=.5)


def _record(**updates):
    result = {"crr": .81, "em": .24, "objective": 2., "optimizer_success": True,
        "numerically_regular": True, "boundary": False, "boundary_crr": False,
        "boundary_em": False, "rank": 2, "j_pvalue": .5, "start_successes": 4,
        "starts_agree": True, "unresolved": False, "status": "regular", "error": "",
        "start_diagnostics_json": "[]"}
    result.update(updates)
    return result


def test_population_hac_exactly_preserves_original_estimator(phases):
    data, _ = load_empirical_sw07_data()
    targets = covariance_moment_targets(data, ALL_MOMENTS, units=UNITS, bandwidth=8)
    fitted, diagnostics = original._fitter(CALIBRATION)(targets)
    expected = original._fit_record(fitted, diagnostics)
    actual = phases[0].observed_fits.set_index("variant").loc["population_hac"]
    for key in expected:
        if key == "start_diagnostics_json":
            continue
        if isinstance(expected[key], float):
            assert actual[key] == pytest.approx(expected[key], nan_ok=True)
        else:
            assert actual[key] == expected[key]


def test_finite_fitter_criterion_matches_full_exact_oracle(phases):
    study = phases[0]
    for row in study.draws.loc[study.draws.variant.isin(["finite_hac", "finite_oracle"])].itertuples():
        assert not row.unresolved
        params = {**CALIBRATION, "crr": row.crr, "em": row.em}
        exact = sw07_finite_sample_moments(params, nobs=156, common_max_lag=4)
        residual = exact.values[:9] - study.draw_moments[row.array_row, :9]
        covariance = (study.draw_covariances[row.array_row] if row.variant == "finite_hac" else
                      np.asarray(study.metadata["oracle_covariance"]))
        expected = residual @ np.linalg.solve(covariance[:9, :9], residual)
        assert row.objective == pytest.approx(expected, rel=2e-9, abs=1e-11)


def test_four_variants_share_each_draw_and_hac_and_phases_are_independent(phases):
    for study in phases:
        assert study.draw_moments.shape == (2, 15)
        assert study.draw_covariances.shape == (2, 15, 15)
        for i, group in study.draws.groupby("replication"):
            assert tuple(group.variant) == module._VARIANTS
            assert group.simulation_seed.nunique() == group.sample_sha256.nunique() == 1
            assert group.array_row.eq(i).all()
            sample = simulate_sw07_sample(CALIBRATION, seed=int(group.simulation_seed.iloc[0]))
            assert hashlib.sha256(sample.to_numpy(dtype="<f8").tobytes()).hexdigest() == group.sample_sha256.iloc[0]
            target = covariance_moment_targets(sample, ALL_MOMENTS, units=UNITS, bandwidth=8)
            assert_array_equal(target.values, study.draw_moments[i])
            assert_array_equal(target.covariance, study.draw_covariances[i])
        assert not study.metadata["parameter_inference_reported"]
    assert not set(phases[0].draws.simulation_seed) & set(phases[1].draws.simulation_seed)
    assert not set(phases[0].draws.sample_sha256) & set(phases[1].draws.sample_sha256)


def test_exact_oracle_weight_is_constant_while_model_moments_vary(monkeypatch):
    data, _ = load_empirical_sw07_data()
    target = covariance_moment_targets(data, ALL_MOMENTS, units=UNITS, bandwidth=8)
    exact = sw07_finite_sample_moments(CALIBRATION)
    weighted = module._weighted_target(target, exact.covariance, "finite_oracle")
    original_covariance = weighted.covariance.copy()
    expected = [sw07_finite_sample_moments({**CALIBRATION, "crr": rho, "em": sigma}).values
                for rho, sigma in ((.6, .12), (.9, .6))]
    calls = []
    def checking_optimizer(callback, selected, start, **kwargs):
        assert_array_equal(selected.covariance, original_covariance[:9, :9])
        assert_array_equal(kwargs["held_out"].covariance, original_covariance[9:, 9:])
        predictions = [callback(theta) for theta in ((.6, .12), (.9, .6))]
        held = [kwargs["held_out_moments_at"](theta) for theta in ((.6, .12), (.9, .6))]
        for i in range(2):
            assert_allclose(predictions[i], expected[i][:9], atol=2e-13)
            assert_allclose(held[i], expected[i][9:], atol=2e-13)
        assert np.max(np.abs(predictions[0] - predictions[1])) > .01
        calls.append(start)
        return _optimizer_result()
    monkeypatch.setattr(module, "fit_structural", checking_optimizer)
    fitted = module._fitters(CALIBRATION)["finite_oracle"](weighted)
    assert len(calls) == 4 and not fitted["unresolved"]
    assert_array_equal(weighted.covariance, original_covariance)
    assert module._weighted_target(target, exact.covariance, "population_hac") is target


def test_first_order_stationarity_failure_survives_successful_optimizer_flag(monkeypatch):
    monkeypatch.setattr(module, "fit_structural", lambda *args, **kwargs:
                        _optimizer_result(("nonstationary_solution",)))
    target = sw07_finite_sample_moments(CALIBRATION)
    record = module._fitters(CALIBRATION)["finite_oracle"](target)
    assert record["optimizer_success"]
    assert record["unresolved"] and not record["numerically_regular"]
    assert np.isnan(record["j_pvalue"])
    assert "stationarity" in record["error"]


def test_interrupted_checkpoint_resume_preserves_draws_and_rejects_changed_design(phases, tmp_path, monkeypatch):
    def interrupt(phase, completed, total):
        if completed == 1:
            raise RuntimeError("simulated reset")
    with pytest.raises(RuntimeError, match="simulated reset"):
        module.run_sw07_estimator_experiment(replications=2, seed=SMOKE_SEED,
                                             checkpoint_dir=tmp_path, progress=interrupt)
    progress = []
    resumed = module.run_sw07_estimator_experiment(replications=2, seed=SMOKE_SEED,
        checkpoint_dir=tmp_path, progress=lambda *args: progress.append(args[1]))
    assert progress == [1, 2]
    assert resumed.metadata["checkpoint_resumed_draws"] == 1
    assert_array_equal(resumed.draw_moments, phases[0].draw_moments)
    assert_array_equal(resumed.draw_covariances, phases[0].draw_covariances)
    assert_frame_equal(resumed.draws, phases[0].draws)
    # Completed checkpoints avoid every draw; the observed anchor is still recomputed.
    monkeypatch.setattr(module, "simulate_sw07_sample", lambda *args, **kwargs:
                        pytest.fail("completed checkpoint attempted another simulation"))
    completed = module.run_sw07_estimator_experiment(replications=2, seed=SMOKE_SEED,
                                                    checkpoint_dir=tmp_path)
    assert completed.metadata["checkpoint_resumed_draws"] == 2
    assert_frame_equal(completed.draws, resumed.draws)
    with pytest.raises(ValueError, match="Checkpoint design"):
        module.run_sw07_estimator_experiment(replications=2, seed=SMOKE_SEED+1,
                                             checkpoint_dir=tmp_path)
    monkeypatch.setattr(module, "_source_hashes", lambda: {"changed_numerical_source.py": "123"})
    with pytest.raises(ValueError, match="Checkpoint design"):
        module.run_sw07_estimator_experiment(replications=2, seed=SMOKE_SEED,
                                             checkpoint_dir=tmp_path)


def test_critical_rank_uses_full_calibration_count_and_unknown_criterion_bounds():
    values = np.arange(1., 400.)
    unresolved = np.zeros(399, dtype=bool)
    assert module._critical_bounds(values, unresolved) == (380., 380., 380)
    unresolved[0] = True
    assert module._critical_bounds(values, unresolved) == (380., 381., 380)
    unresolved[:20] = True
    lo, hi, rank = module._critical_bounds(values, unresolved)
    assert lo == 380. and hi == np.inf and rank == 380
    assert module._critical_bounds([1., 2.], [False, False]) == (np.inf, np.inf, 3)
    values[0] = np.nan
    unresolved[:] = False
    assert module._critical_bounds(values, unresolved) == (380., 381., 380)


def _synthetic_phase(study, values, unresolved):
    phase = study.metadata["phase"]
    rows = [{"variant": variant, "phase": phase, "replication": i,
             "simulation_seed": f"{phase}-{i}", "objective": value, "unresolved": bool(bad)}
            for variant in module._VARIANTS for i, (value, bad) in enumerate(zip(values, unresolved))]
    return replace(study, draws=pd.DataFrame(rows))


def test_calibration_and_validation_failures_stay_in_rate_denominators(phases):
    values = np.arange(1., 400.)
    bad = np.zeros(399, dtype=bool)
    # A missing calibration result bounds the critical value by 380 and 381.
    values[0] = np.nan
    calibration = _synthetic_phase(phases[0], values, bad)
    validation = _synthetic_phase(phases[1], [379., 380.5, 382., np.nan], [False]*4)
    compared = module.compare_sw07_estimator_phases(calibration, validation)
    assert compared.calibration_draws.eq(399).all()
    assert compared.calibration_unresolved.eq(1).all()
    assert compared.validation_draws.eq(4).all()
    assert compared.validation_unresolved.eq(1).all()
    assert compared.calibrated_events_lower.eq(1).all()
    assert compared.calibrated_events_upper.eq(3).all()
    assert compared.calibrated_rate_lower.eq(.25).all()
    assert compared.calibrated_rate_upper.eq(.75).all()
    assert (compared.calibrated_mc95_lower < .25).all()
    assert (compared.calibrated_mc95_upper > .75).all()


@pytest.mark.parametrize("key,value", [("seed", 9000), ("source_hashes", {"changed": "abc"}),
    ("environment", {"python": "different"}), ("dgp", {"em": .5})])
def test_phase_comparison_rejects_different_designs(phases, key, value):
    changed = replace(phases[1], metadata={**phases[1].metadata, key: value})
    with pytest.raises(ValueError, match="phase designs differ"):
        module.compare_sw07_estimator_phases(phases[0], changed)


def test_phase_comparison_rejects_reused_simulation_streams(phases):
    draws = phases[1].draws.copy()
    draws["simulation_seed"] = phases[0].draws.simulation_seed
    changed = replace(phases[1], draws=draws)
    with pytest.raises(ValueError, match="streams overlap"):
        module.compare_sw07_estimator_phases(phases[0], changed)


def test_sample_failure_is_retained_for_all_variants_without_artificial_moments(monkeypatch):
    monkeypatch.setattr(module, "_fitters", lambda params: {v: lambda target: _record() for v in module._VARIANTS})
    monkeypatch.setattr(module, "simulate_sw07_sample", lambda *args, **kwargs:
                        (_ for _ in ()).throw(ValueError("injected sample failure")))
    study = module.run_sw07_estimator_experiment(replications=2, seed=SMOKE_SEED+100)
    assert study.draws.unresolved.all() and len(study.draws) == 8
    assert study.draws.error.str.contains("injected sample failure").all()
    assert np.isnan(study.draw_moments).all() and np.isnan(study.draw_covariances).all()
    assert study.summary.replications.eq(2).all() and study.summary.unresolved_draws.eq(2).all()
    assert study.summary.naive_rate_lower.eq(0.).all() and study.summary.naive_rate_upper.eq(1.).all()
    assert study.paired_differences.total_pairs.eq(2).all()
    assert study.paired_differences.paired_usable.eq(0).all()


def test_optimizer_failure_retains_sample_and_all_denominators(monkeypatch):
    def factory(params):
        fits = {}
        for variant in module._VARIANTS:
            def make_fit():
                calls = 0
                def fit(target):
                    nonlocal calls
                    calls += 1
                    if calls > 1:
                        raise ValueError("injected optimizer failure")
                    return _record()
                return fit
            fits[variant] = make_fit()
        return fits
    monkeypatch.setattr(module, "_fitters", factory)
    study = module.run_sw07_estimator_experiment(replications=2, seed=SMOKE_SEED+100)
    assert study.draws.unresolved.all()
    assert study.draws.error.str.contains("injected optimizer failure").all()
    assert np.isfinite(study.draw_moments).all() and np.isfinite(study.draw_covariances).all()
    assert study.summary.naive_events_lower.eq(0).all() and study.summary.naive_events_upper.eq(2).all()
    assert study.summary.naive_mc95_lower.eq(0.).all() and study.summary.naive_mc95_upper.eq(1.).all()


@pytest.mark.parametrize("kwargs", [{"replications": 1}, {"replications": True}, {"seed": -1},
    {"seed": False}, {"phase": "other"}, {"resume": 1}, {"progress": 7}])
def test_invalid_controls_fail_before_fitting(kwargs):
    with pytest.raises(ValueError):
        module.run_sw07_estimator_experiment(**kwargs)
