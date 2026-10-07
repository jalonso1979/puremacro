"""Finite-sample diagnostics retain the original fit and honest MC failures."""
import hashlib

import numpy as np
import pytest

from puremacro.structural import sw07_finite_sample as module
from puremacro.structural.empirical_sw07 import ALL_MOMENTS, UNITS, covariance_moment_targets
from puremacro.structural.sw07_sampling import simulate_sw07_sample


@pytest.fixture(scope="module")
def study():
    return module.run_sw07_finite_sample(replications=3, scenarios=["baseline_fixed", "published_fixed"])


def test_original_observed_fits_and_full_sample_covariance_are_preserved(study):
    observed = study.observed_fits.set_index("calibration")
    np.testing.assert_allclose(observed.loc["baseline", ["crr", "em"]].to_numpy(dtype=float), [.98, .01], atol=1e-7)
    assert observed.loc["baseline", "objective"] == pytest.approx(396736.095789, rel=1e-8)
    assert observed.loc["published", "objective"] == pytest.approx(25.2554104, rel=1e-7)
    assert len(study.draws) == 6
    assert study.draw_moments.shape == (6, 15)
    assert study.draw_covariances.shape == (6, 15, 15)
    assert np.isfinite(study.draw_moments).all()
    assert np.any(np.abs(study.draw_covariances[:, :9, 9:]) > 1e-6)
    assert set(study.draws.start_successes) == {4}
    assert study.metadata["common_max_lag"] == 4
    assert study.metadata["effective_observations"] == 152
    assert not study.metadata["parameter_inference_reported"]
    assert study.metadata["small_replication_warning"]
    assert set(study.covariance_audit.estimator) == {"empirical_mc", "exact_known_mean", "hac4", "hac8", "hac12"}


def test_every_simulated_draw_can_be_replayed_with_its_full_hac(study):
    row = study.draws.iloc[0]
    params = study.metadata["scenarios"][row.scenario]["dgp"]
    sample = simulate_sw07_sample(params, nobs=156, seed=int(row.simulation_seed))
    assert hashlib.sha256(sample.to_numpy(dtype="<f8").tobytes()).hexdigest() == row.sample_sha256
    target = covariance_moment_targets(sample, ALL_MOMENTS, units=UNITS, bandwidth=8)
    np.testing.assert_array_equal(target.values, study.draw_moments[row.array_row])
    np.testing.assert_array_equal(target.covariance, study.draw_covariances[row.array_row])
    one = module.run_sw07_finite_sample(replications=2, scenarios=["published_fixed"])
    published = study.draws.loc[study.draws.scenario == "published_fixed"]
    assert one.draws.simulation_seed.tolist() == published.simulation_seed.iloc[:2].tolist()
    np.testing.assert_array_equal(one.draw_moments, study.draw_moments[3:5])


def test_tail_and_naive_rule_denominators_include_all_draws(study):
    for _, summary in study.summary.iterrows():
        draws = study.draws.loc[study.draws.scenario == summary.scenario]
        usable = ~draws.unresolved & np.isfinite(draws.objective)
        exceed = int((usable & (draws.objective >= summary.observed_objective)).sum())
        assert summary.replications == 3
        assert summary.tail_fraction_lower == exceed / 3
        assert summary.tail_fraction_upper == (exceed + (~usable).sum()) / 3
        assert 0 <= summary.tail_mc_95_lower <= summary.tail_fraction_lower
        assert summary.tail_fraction_upper <= summary.tail_mc_95_upper <= 1
        assert summary.boundary_fits == int((usable & draws.boundary).sum())


def test_failed_fits_are_not_silently_dropped_or_reclassified(monkeypatch):
    original = module._fitter
    def factory(calibration):
        fit = original(calibration)
        calls = 0
        def failing(target):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise ValueError("deliberate per-draw optimization failure")
            return fit(target)
        return failing
    monkeypatch.setattr(module, "_fitter", factory)
    result = module.run_sw07_finite_sample(replications=2, scenarios=["published_fixed"])
    summary = result.summary.iloc[0]
    assert summary.unresolved_draws == 2 and summary.usable_fits == 0
    assert summary.tail_fraction_lower == 0 and summary.tail_fraction_upper == 1
    assert summary.tail_mc_95_lower == 0 and summary.tail_mc_95_upper == 1
    assert result.draws.error.str.contains("deliberate").all()
    # Sample moments survive optimizer failure and contribute to the sampling audit.
    assert np.isfinite(result.draw_moments).all()
    assert result.moment_audit.moment_draws.eq(2).all()


def test_plugin_dgp_is_distinct_and_explicitly_labeled():
    result = module.run_sw07_finite_sample(replications=2, scenarios=["baseline_fitted"])
    scenario = result.metadata["scenarios"]["baseline_fitted"]
    assert scenario["is_plugin"]
    np.testing.assert_allclose([scenario["dgp"]["crr"], scenario["dgp"]["em"]], [.98, .01], atol=1e-7)
    assert "not composite-null" in scenario["scope"]


@pytest.mark.parametrize("kwargs", [{"replications": True}, {"replications": 1}, {"seed": -1},
    {"bandwidth": 152}, {"bandwidth": False}, {"scenarios": []}, {"scenarios": ["missing"]},
    {"scenarios": ["baseline_fixed", "baseline_fixed"]}, {"progress": True}])
def test_bad_controls_fail_before_running(kwargs):
    with pytest.raises(ValueError):
        module.run_sw07_finite_sample(**kwargs)


def test_binomial_simulation_precision_is_nonzero_at_zero_exceedances():
    lo, hi = module._binomial_interval(0, 399)
    assert lo == 0 and .009 < hi < .010
    lo, hi = module._binomial_interval(399, 399)
    assert .990 < lo < .991 and hi == 1


def test_interrupted_run_resumes_without_repeating_or_changing_draws(study, tmp_path):
    def interrupt(scenario, completed, total):
        if completed == 1:
            raise RuntimeError("simulated reset")
    with pytest.raises(RuntimeError, match="simulated reset"):
        module.run_sw07_finite_sample(replications=3, scenarios=["published_fixed"],
                                     checkpoint_dir=tmp_path, progress=interrupt)
    assert (tmp_path / "published_fixed.npz").is_file()
    progress = []
    resumed = module.run_sw07_finite_sample(replications=3, scenarios=["published_fixed"],
        checkpoint_dir=tmp_path, progress=lambda *args: progress.append(args[1]))
    assert progress == [1, 2, 3]
    assert resumed.metadata["checkpointing"]["resumed_draws"] == {"published_fixed": 1}
    np.testing.assert_array_equal(resumed.draw_moments, study.draw_moments[3:])
    np.testing.assert_array_equal(resumed.draw_covariances, study.draw_covariances[3:])
    original = study.draws.loc[study.draws.scenario == "published_fixed"]
    np.testing.assert_allclose(resumed.draws.objective, original.objective, rtol=1e-12)
    assert all(len(__import__("json").loads(value)) == 4 for value in resumed.draws.start_diagnostics_json)
    # A completed checkpoint is also reusable, without any per-draw refitting.
    completed = module.run_sw07_finite_sample(replications=3, scenarios=["published_fixed"], checkpoint_dir=tmp_path)
    assert completed.metadata["checkpointing"]["resumed_draws"]["published_fixed"] == 3
    np.testing.assert_array_equal(completed.draw_moments, resumed.draw_moments)
    with pytest.raises(ValueError, match="Checkpoint settings"):
        module.run_sw07_finite_sample(replications=3, scenarios=["published_fixed"], checkpoint_dir=tmp_path, seed=42)
