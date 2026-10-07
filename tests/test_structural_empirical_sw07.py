"""Real-data SW07 study contracts and economic covariance checks."""
import hashlib
import importlib.resources
import json

import numpy as np
import pandas as pd
import pytest

from puremacro.structural.empirical_sw07 import (
    ALL_MOMENTS, FIT_MOMENTS, HELD_OUT_MOMENTS, UNITS,
    covariance_moment_targets, fit_empirical_sw07,
    load_empirical_sw07_data, sw07_covariance_moments,
)


def test_observed_source_vintage_hash_and_actual_sample():
    data, provenance = load_empirical_sw07_data()
    raw = (importlib.resources.files("puremacro.dsge") / "_sw07_data.csv").read_bytes()
    assert provenance["source_csv_sha256"] == hashlib.sha256(raw).hexdigest()
    assert data.shape == (156, 7)
    assert str(data.index[0]) == "1966Q1"
    assert str(data.index[-1]) == "2004Q4"
    assert not provenance["is_synthetic"]
    assert not provenance["published_estimation_replication"]
    assert "not original SW07" in provenance["vintage"]
    # Quarterly funds rate, not annual percent and not a fraction.
    assert data.iloc[0]["ffr"] == pytest.approx(1.14)
    assert data.iloc[0]["gdp_growth"] == pytest.approx(2.136117715809445)


def test_monetary_innovation_variance_has_quadratic_scale():
    # The conditional equilibrium is independent of innovation variance, so
    # differences of observable covariance contributions must scale as sd².
    zero = sw07_covariance_moments({"em": 0.})
    small = sw07_covariance_moments({"em": .2})
    large = sw07_covariance_moments({"em": .4})
    assert np.linalg.norm(small - zero) > 1e-3
    np.testing.assert_allclose(large - zero, 4 * (small - zero), rtol=1e-8, atol=2e-10)
    assert small.attrs["transition_spectral_radius"] < 1
    assert small.attrs["lyapunov_relative_residual"] < 1e-9
    assert not small.attrs["measurement_error_included"]


@pytest.mark.parametrize("params,match", [
    ({"em": -.1}, "nonnegative"), ({"crr": np.nan}, "finite"),
    ({"not_a_parameter": 1.}, "unknown"),
    ({"em": .2 + 1j}, "real scalars"), ({"em": np.array([.2])}, "real scalars"),
])
def test_invalid_population_parameters(params, match):
    with pytest.raises(ValueError, match=match):
        sw07_covariance_moments(params)


def test_nonstationary_economic_solution_not_silently_repaired():
    with pytest.raises(np.linalg.LinAlgError):
        sw07_covariance_moments({"crhoa": 1.})


def test_changed_observed_snapshot_requires_deliberate_review(tmp_path, monkeypatch):
    import puremacro.structural.empirical_sw07 as module
    raw = (importlib.resources.files("puremacro.dsge") / "_sw07_data.csv").read_bytes()
    (tmp_path / "_sw07_data.csv").write_bytes(raw.replace(b"1.14\n", b"8.14\n", 1))
    assert (tmp_path / "_sw07_data.csv").read_bytes() != raw
    monkeypatch.setattr(module.importlib.resources, "files", lambda package: tmp_path)
    with pytest.raises(ValueError, match="authenticated.*snapshot"):
        module.load_empirical_sw07_data()


def test_source_authentication_accepts_only_line_ending_change(tmp_path, monkeypatch):
    import puremacro.structural.empirical_sw07 as module
    raw = (importlib.resources.files("puremacro.dsge") / "_sw07_data.csv").read_bytes()
    (tmp_path / "_sw07_data.csv").write_bytes(raw.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(module.importlib.resources, "files", lambda package: tmp_path)
    data, metadata = module.load_empirical_sw07_data()
    assert len(data) == 156
    assert metadata["source_csv_normalized_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("problem,match", [
    ("missing", "missing"), ("nonfinite", "finite"), ("gap", "consecutive"),
    ("duplicate", "consecutive"), ("dates", "quarterly"),
    ("units", "units"), ("bandwidth", "bandwidth"),
])
def test_covariance_inputs_fail_closed(problem, match):
    data, _ = load_empirical_sw07_data()
    units, bandwidth = UNITS, 8
    if problem == "missing":
        data = data.drop(columns="ffr")
    elif problem == "nonfinite":
        data.loc[data.index[10], "infl"] = np.inf
    elif problem == "gap":
        data = data.drop(index=data.index[10])
    elif problem == "duplicate":
        data.index = pd.PeriodIndex([data.index[0]] + list(data.index[:-1]), freq="Q")
    elif problem == "dates":
        data.index = pd.RangeIndex(len(data))
    elif problem == "units":
        units = {"ffr": "rate"}
    elif problem == "bandwidth":
        bandwidth = True
    with pytest.raises(ValueError, match=match):
        covariance_moment_targets(data, ALL_MOMENTS, units=units, bandwidth=bandwidth)


@pytest.fixture(scope="module")
def empirical_study():
    return fit_empirical_sw07(profile_points=5)


def test_actual_conditional_fit_preserves_boundary_failure(empirical_study):
    study = empirical_study
    result = study.fit
    assert study.metadata["status"] == "nonregular_fit"
    assert result.success and not result.inference_valid
    np.testing.assert_allclose(result.theta, [.98, .01], atol=2e-5)
    assert result.boundary.all()
    assert np.isnan(result.standard_errors).all()
    assert np.isnan(result.j_pvalue)
    assert result.objective > 1000
    assert "parameter_on_boundary" in study.metadata["inference_unavailable_reasons"]
    assert "correct_specification_inference_not_requested" in study.metadata["inference_unavailable_reasons"]
    assert "crr" not in study.metadata["fixed_calibration"]
    assert "em" not in study.metadata["fixed_calibration"]
    assert study.metadata["fixed_calibration"]["crhoa"] == .9977
    assert len(study.multistart) == 4
    assert study.multistart.optimizer_success.all()
    assert np.ptp(study.multistart.objective) < 1e-4
    assert len(result.targets.values) == len(FIT_MOMENTS) == 9
    assert len(result.held_out_targets.values) == len(HELD_OUT_MOMENTS) == 6
    assert set(result.targets.labels).isdisjoint(result.held_out_targets.labels)
    # The full covariance retains dependence across fitted and unused moments.
    assert np.any(np.abs(study.targets.covariance[:9, 9:]) > 1e-5)
    assert set(study.bandwidth_sensitivity.bandwidth) == {4, 8, 12}
    assert set(study.profiles.parameter) == {"crr", "em"}
    assert study.sample_split["sample"].nunique() == 2
    sensitivity = study.calibration_sensitivity.set_index("calibration")
    alternate = sensitivity.loc["exploratory_published_sw07_mode"]
    assert not alternate.boundary
    assert alternate.crr == pytest.approx(.8383, abs=.002)
    assert alternate.em == pytest.approx(.2541, abs=.002)
    assert alternate.conditional_null_j_pvalue < .01
    assert alternate.objective < sensitivity.loc["declared_pfeifer_baseline", "objective"]
    from puremacro.replication.cases_dsge_estimation import _TABLE1_MODE
    assert study.metadata["calibration_sensitivity"]["published_mode_values"] == _TABLE1_MODE
    assert "added after" in study.metadata["calibration_sensitivity"]["design_status"]


def test_inference_opt_in_cannot_override_boundary():
    study = fit_empirical_sw07(profile_points=5, allow_conditional_inference=True)
    assert not study.fit.inference_valid
    assert np.isnan(study.fit.standard_errors).all()
    assert "parameter_on_boundary" in study.metadata["inference_unavailable_reasons"]
    assert "correct_specification_inference_not_requested" not in study.metadata["inference_unavailable_reasons"]


def test_application_artifacts_are_strict_json_and_replayable(tmp_path):
    from puremacro.examples.empirical_sw07_matching import run_application
    study = run_application(tmp_path, profile_points=5)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"),
                          parse_constant=lambda value: pytest.fail(f"non-JSON number {value}"))
    assert manifest["conditional_null_j_pvalue"] is None
    assert manifest["conditional_null_j_statistic"] is None
    assert manifest["observed_data_artifact_sha256"] == hashlib.sha256(
        (tmp_path / "observed_data.csv").read_bytes()).hexdigest()
    assert manifest["evidence_kind"] == "observed_data_conditional_structural_estimation"
    assert not manifest["reported_inference_available"]
    covariance = pd.read_csv(tmp_path / "joint_covariance.csv", index_col=0)
    np.testing.assert_allclose(covariance, study.targets.covariance)
    assert covariance.index.tolist() == list(study.targets.labels)
    parameters = pd.read_csv(tmp_path / "parameters.csv")
    assert parameters.se.isna().all()
    comparison = pd.read_csv(tmp_path / "fitted_moments.csv")
    np.testing.assert_allclose(comparison.residual_over_target_se,
                               comparison.residual / comparison.target_se)
    assert (tmp_path / "calibration_sensitivity.csv").is_file()
    assert (tmp_path / "profiles.png").stat().st_size > 1000
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "nonregular_fit" in report and "not a reproduction" in report


@pytest.mark.parametrize("kwargs", [{"profile_points": True}, {"profile_points": 4},
                                   {"allow_conditional_inference": "yes"}])
def test_study_controls_validate(kwargs):
    with pytest.raises(ValueError):
        fit_empirical_sw07(**kwargs)
