"""End-to-end artifact contract for the finite-sample diagnostic application."""
from dataclasses import replace
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from puremacro.examples import sw07_finite_sample as application


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    output = tmp_path_factory.mktemp("sw07_finite_sample_application")
    progress = []
    study = application.run_application(output, replications=2, seed=773, bandwidth=4,
        scenarios=["published_fixed"], progress=lambda *values: progress.append(values))
    return output, study, progress


def test_small_real_run_exports_every_draw_and_joint_covariance(evidence):
    output, study, progress = evidence
    assert progress == [("published_fixed", 1, 2), ("published_fixed", 2, 2)]
    assert study.summary.scenario.tolist() == ["published_fixed"]
    assert study.summary.replications.tolist() == [2]
    for name in application.TABLES:
        stored = pd.read_csv(output / f"{name}.csv")
        original = getattr(study, name)
        assert stored.shape == original.shape
        assert stored.columns.tolist() == original.columns.tolist()
    stored_draws = pd.read_csv(output / "draws.csv")
    np.testing.assert_array_equal(stored_draws.array_row, [0, 1])
    with np.load(output / "draw_evidence.npz", allow_pickle=False) as arrays:
        assert arrays["moments"].shape == (2, 15)
        assert arrays["covariances"].shape == (2, 15, 15)
        np.testing.assert_array_equal(arrays["moments"], study.draw_moments)
        np.testing.assert_array_equal(arrays["covariances"], study.draw_covariances)
        covariance = arrays["covariances"][0]
        assert np.max(np.abs(covariance-np.diag(np.diag(covariance)))) > 1e-5
    assert (output / "criterion_distribution.png").stat().st_size > 1000


def test_manifest_is_strict_json_with_provenance_and_artifact_hashes(evidence):
    output, study, _ = evidence
    def reject_constant(token):
        raise AssertionError(f"Nonstandard JSON constant {token}")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"), parse_constant=reject_constant)
    assert manifest["master_seed"] == 773
    assert manifest["bandwidth"] == 4
    assert manifest["replications_per_scenario"] == 2
    assert manifest["observations"] == 156
    assert manifest["source"]["is_synthetic"] is False
    assert manifest["parameter_inference_reported"] is False
    assert manifest["small_replication_warning"] is True
    assert manifest["summary"][0]["observed_objective"] == study.summary.observed_objective.iloc[0]
    for filename, digest in manifest["artifacts"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == digest
    assert set(manifest["environment"]) == {"python", "numpy", "scipy", "pandas"}


def test_report_preserves_monte_carlo_scope_and_actual_results(evidence):
    output, study, _ = evidence
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Requested simulations per scenario: **2**" in report
    assert "Seed: **773**" in report
    assert "**4**" in report
    assert "Clopper–Pearson" in report
    assert "not validated bootstrap" in report
    assert "not parameter confidence intervals" in report
    assert "conditional on usable optimization" in report
    assert "exact Gaussian finite-sample covariance" in report
    assert f"{study.summary.observed_objective.iloc[0]:.6g}" in report


def test_cli_exit_tracks_computation_instead_of_scientific_rejection(evidence, monkeypatch, capsys):
    _, study, _ = evidence
    # The command boundary receives a genuine study. Alter only its explicit
    # computational outcome to exercise exit reporting without repeating fits.
    assert study.observed_fits.j_pvalue.iloc[0] < .05
    summary = study.summary.copy()
    summary["unresolved_draws"] = 0
    successful = replace(study, summary=summary)
    seen = []
    def completed(output, **options):
        seen.append(options)
        return successful
    monkeypatch.setattr(application, "run_application", completed)
    assert application.main(["--scenario", "published_fixed", "--replications", "2",
                             "--seed", "773", "--bandwidth", "4"]) == 0
    assert seen[0]["scenarios"] == ["published_fixed"]
    assert seen[0]["replications"] == 2 and seen[0]["seed"] == 773
    summary = summary.copy()
    summary["unresolved_draws"] = 1
    monkeypatch.setattr(application, "run_application", lambda *args, **kwargs: replace(study, summary=summary))
    assert application.main(["--scenario", "published_fixed", "--replications", "2"]) == 1
    assert "Evidence:" in capsys.readouterr().out


def test_invalid_simulation_count_does_not_create_a_dossier(tmp_path):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="replications"):
        application.run_application(output, replications=1)
    assert not output.exists()
