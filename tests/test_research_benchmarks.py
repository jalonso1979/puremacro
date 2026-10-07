"""Independent numerical references and controls for the evidence mechanism."""
from dataclasses import replace
import json

import numpy as np
import pytest

from puremacro.validation.research import (
    ResearchBenchmark, research_benchmarks, run_research_benchmarks,
)


@pytest.fixture(scope="module")
def report():
    return run_research_benchmarks()


def minimal(**overrides):
    case = ResearchBenchmark(
        "control", "Evidence comparator control", "analytical",
        lambda: {"x": np.array([1., 2.])}, lambda: {"x": np.array([1., 2.])},
        ("Closed-form fixture defined in this test",), {"origin": "synthetic"},
        "No draws", "Identity", {"x": "dimensionless"}, "Comparator only",
    )
    return replace(case, **overrides)


def test_all_research_benchmarks_have_independent_evidence(report):
    assert len(report.results) == 8
    assert not report.passed  # The original-data published t-rounding miss stays visible.
    assert [r.id for r in report.results if not r.passed] == ["rr2010_published_peak"]
    assert {r.evidence_kind for r in report.results} == {
        "analytical", "numerical_oracle", "external_software", "official_data", "published_empirical"}
    for case in report.results:
        assert case.metrics
        assert case.negative_control_rejected
        assert not case.error
        if case.id == "rr2010_published_peak":
            assert case.metrics["response_h10"]["passed"]
            assert case.metrics["trough_horizon"]["passed"]
            assert not case.metrics["t_h10"]["passed"]
            assert 0 < case.metrics["t_h10"]["max_tolerance_excess"] < 4e-6
        else:
            assert all(m["passed"] for m in case.metrics.values())
        assert case.provenance["sources"]
        assert case.provenance["limitations"]


def test_complete_json_and_markdown_dossier(report, tmp_path):
    paths = report.write(tmp_path)
    document = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert document["passed"] is False
    assert set(document["environment"]) >= {"puremacro", "python", "numpy", "scipy", "pandas"}
    assert document["generated_at"].endswith("+00:00")
    markdown = paths["markdown"].read_text(encoding="utf-8")
    for case in document["cases"]:
        assert case["id"] in markdown
        for metric in case["metrics"].values():
            assert np.shape(metric["observed"]) == np.shape(metric["reference"])
            assert metric["unit"]
    json.dumps(report.to_dict(), allow_nan=False)


@pytest.mark.parametrize("evidence", [None, {}, {"y": [1., 2.]}, {"x": []},
                                       {"x": [np.nan, 2.]}, {"x": [np.inf, 2.]},
                                       {"x": [[1., 2.]]}, {"x": [1.+1j, 2.]}])
@pytest.mark.parametrize("side", ["compute", "reference"])
def test_missing_empty_nonfinite_and_shape_mismatch_fail_closed(evidence, side):
    case = minimal(**{side: lambda: evidence})
    result = run_research_benchmarks([case])
    assert not result.passed
    assert result.results[0].error
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("change", [
    {"units": {}}, {"sources": ()}, {"data": {}}, {"sampling": ""},
    {"limitations": ""}, {"evidence_kind": "published_without_evidence"},
    {"rtol": -1.}, {"atol": np.nan}, {"rtol": np.inf}, {"rtol": 1.},
    {"data": {"bad": np.inf}},
])
def test_incomplete_provenance_or_invalid_tolerance_never_passes(change):
    result = run_research_benchmarks([minimal(**change)])
    assert not result.passed
    json.dumps(result.to_dict(), allow_nan=False)


def test_empty_selection_and_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="at least one"):
        run_research_benchmarks([])
    with pytest.raises(ValueError, match="unique"):
        run_research_benchmarks([minimal(), minimal()])


@pytest.mark.parametrize("case", research_benchmarks(), ids=lambda c: c.id)
def test_changed_observed_results_are_rejected(case):
    def wrong():
        result = {k: np.array(v, dtype=float, copy=True) for k, v in case.compute().items()}
        first = next(iter(result))
        result[first].flat[0] += 1000.
        return result

    report = run_research_benchmarks([replace(case, compute=wrong)])
    assert not report.passed
    assert not report.results[0].error
    assert report.results[0].negative_control_rejected
    assert any(not metric["passed"] for metric in report.results[0].metrics.values())


def test_computation_failure_is_recorded_and_later_cases_still_run():
    def broken():
        raise RuntimeError("Numerical solver failed")

    report = run_research_benchmarks([minimal(id="failed", compute=broken), minimal(id="ok")])
    assert not report.passed
    assert "Numerical solver failed" in report.results[0].error
    assert report.results[1].passed


def test_dynare_fixture_checksum_corruption_fails_at_resource_boundary(monkeypatch, tmp_path):
    from importlib import resources
    from puremacro.validation import research

    source = resources.files("puremacro.dsge").joinpath("_references", "dynare_live")
    target = tmp_path / "_references" / "dynare_live"
    target.mkdir(parents=True)
    for name in ("manifest.json", "rbc.mod", "rbc_order2.npz", "rbc_innovations.csv"):
        (target / name).write_bytes(source.joinpath(name).read_bytes())
    content = (target / "rbc_order2.npz").read_bytes()
    (target / "rbc_order2.npz").write_bytes(content[:-1]+bytes([content[-1] ^ 1]))
    monkeypatch.setattr(research.resources, "files", lambda package: tmp_path)
    case = next(c for c in research_benchmarks() if c.id == "dynare_rbc_order2")
    report = run_research_benchmarks([case])
    assert not report.passed
    assert "checksum mismatch" in report.results[0].error


def test_gls_reference_uses_estimator_covariance_without_sample_multiplier(report):
    from puremacro.validation import research
    result = next(r for r in report.results if r.id == "linear_minimum_distance")
    design, values, covariance = research._linear_data()
    # Whitening + QR is a separate verification of the documented normal-equation oracle.
    whitened_design = np.linalg.solve(np.linalg.cholesky(covariance), design)
    whitened_values = np.linalg.solve(np.linalg.cholesky(covariance), values)
    q, r = np.linalg.qr(whitened_design)
    inverse_r = np.linalg.solve(r, np.eye(2))
    np.testing.assert_allclose(result.metrics["parameters"]["reference"], np.linalg.solve(r, q.T@whitened_values))
    np.testing.assert_allclose(result.metrics["parameter_covariance"]["reference"], inverse_r@inverse_r.T)


def test_cli_writes_selected_dossier_and_rejects_unknown_case(tmp_path):
    from tools.run_research_benchmarks import main
    assert main(["--case", "growth_analytical", "--output", str(tmp_path)]) == 0
    assert len(json.loads((tmp_path / "benchmark_report.json").read_text(encoding="utf-8"))["cases"]) == 1
    with pytest.raises(SystemExit) as error:
        main(["--case", "missing"])
    assert error.value.code == 2


def test_rr2010_dossier_hashes_match_authenticated_resources(report):
    from puremacro.replication import load_rr2010_data
    source = load_rr2010_data().attrs
    for case in report.results:
        if case.id.startswith("rr2010_"):
            for name in ("archive_sha256", "csv_sha256", "reference_sha256"):
                assert case.provenance["data"][name] == source[name]
