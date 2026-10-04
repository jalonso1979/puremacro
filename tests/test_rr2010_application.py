"""The published-replication dossier preserves numerical evidence and failures."""
from dataclasses import replace
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from puremacro.examples import romer_romer_2010_replication as application
from puremacro.validation import research_benchmarks, run_research_benchmarks


def test_original_data_application_exports_complete_covariance_and_evidence(tmp_path):
    result, report = application.run_application(tmp_path)
    assert not report.passed and len(report.results) == 2
    cases = {case.id: case for case in report.results}
    assert cases["rr2010_baseline_software"].passed
    published = cases["rr2010_published_peak"]
    assert published.metrics["response_h10"]["passed"]
    assert published.metrics["trough_horizon"]["passed"]
    assert not published.metrics["t_h10"]["passed"]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert not manifest["passed"] and manifest["evidence_kind"] == "published_empirical"
    assert manifest["comparison_status"]["rr2010_baseline_software"]
    assert manifest["nobs"] == 232 and manifest["df_resid"] == 218
    assert manifest["published_targets"]["absolute_rounding_tolerance"] == .005
    for name, digest in manifest["artifact_sha256"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    coefficients = pd.read_csv(tmp_path / "coefficients.csv")
    np.testing.assert_allclose(coefficients.estimate, result.coefficients)
    covariance = pd.read_csv(tmp_path / "coefficient_covariance.csv", index_col=0)
    np.testing.assert_allclose(covariance, result.coefficient_covariance, atol=1e-16)
    response_covariance = pd.read_csv(tmp_path / "response_covariance.csv", index_col=0)
    np.testing.assert_allclose(response_covariance, result.irf_covariance, atol=1e-16)
    responses = pd.read_csv(tmp_path / "responses.csv")
    np.testing.assert_allclose(responses.response, result.irf)
    np.testing.assert_allclose(responses.standard_error, result.standard_errors)
    assert (tmp_path / "tax_response.png").stat().st_size > 1000
    assert "**FAIL**" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "unresolved" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert not json.loads((tmp_path / "benchmark_report.json").read_text(encoding="utf-8"))["passed"]


def test_application_keeps_failed_comparison_visible(tmp_path, monkeypatch):
    case = next(c for c in research_benchmarks() if c.id == "rr2010_published_peak")
    failed = run_research_benchmarks([replace(case, reference=lambda: {
        "response_h10": 3.08, "t_h10": -3.53, "trough_horizon": 10})])
    assert not failed.passed
    monkeypatch.setattr(application, "run_research_benchmarks", lambda cases: failed)
    _, report = application.run_application(tmp_path)
    assert not report.passed
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["passed"] is False
    assert "**FAIL**" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_publication_rounding_is_not_used_for_software_parity():
    cases = {c.id: c for c in research_benchmarks()}
    software = cases["rr2010_baseline_software"]
    published = cases["rr2010_published_peak"]
    assert published.atol == .005 and published.rtol == 0
    values = {name: np.asarray(value).copy() for name, value in software.reference().items()}
    values["irf"][10] += .001
    assert not run_research_benchmarks([replace(software, compute=lambda: values)]).passed
    wrong_horizon = dict(published.reference(), trough_horizon=9)
    assert not run_research_benchmarks([replace(published, compute=lambda: wrong_horizon)]).passed
