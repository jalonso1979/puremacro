"""Authenticated phase exports and independent-reference application checks."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from puremacro.examples.sw07_estimator_experiment import load_experiment, run_application
from puremacro.structural import compare_sw07_estimator_phases


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("sw07-controls")
    a = run_application(root/"calibration", replications=2, seed=8172)
    b = run_application(root/"validation", replications=2, phase="validation", seed=8172,
                        calibration=root/"calibration")
    return root, a, b


def test_full_artifacts_and_phase_reference(evidence):
    root, a, b = evidence
    for name, study in (("calibration", a), ("validation", b)):
        directory = root/name
        manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"),
                              parse_constant=lambda x: pytest.fail(f"invalid JSON number {x}"))
        for file, digest in manifest["artifacts"].items():
            assert hashlib.sha256((directory/file).read_bytes()).hexdigest() == digest
        loaded = load_experiment(directory)
        np.testing.assert_array_equal(loaded.draw_moments, study.draw_moments)
        np.testing.assert_array_equal(loaded.draw_covariances, study.draw_covariances)
        pd.testing.assert_frame_equal(loaded.draws, study.draws, check_dtype=False)
        assert len(loaded.draws) == 8
    actual = pd.read_csv(root/"validation"/"calibrated_validation.csv")
    expected = compare_sw07_estimator_phases(a, b)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    # Two draws cannot estimate the predeclared upper-tail order statistic.
    assert actual.critical_upper_unbounded.all()
    assert (actual.calibrated_events_lower == 0).all()
    manifest = json.loads((root/"validation"/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["calibration_manifest_sha256"] == hashlib.sha256(
        (root/"calibration"/"manifest.json").read_bytes()).hexdigest()


def test_artifact_tampering_refused(evidence, tmp_path):
    import shutil
    root, _, _ = evidence
    copy = tmp_path/"copied"
    shutil.copytree(root/"calibration", copy)
    with (copy/"draws.csv").open("a", encoding="utf-8") as stream:
        stream.write("\ncorrupt\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_experiment(copy)


def test_validation_phase_reference_only(evidence, tmp_path):
    root, _, _ = evidence
    with pytest.raises(ValueError, match="only used"):
        run_application(tmp_path, replications=2, calibration=root/"calibration")
