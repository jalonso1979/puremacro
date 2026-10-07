"""Run from the isolated wheel with network and statsmodels imports denied."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

BASE = Path(os.environ["PUREMACRO_VALIDATION_BASE"])
INSTALLED = BASE / "installed"
ROOT = Path(os.environ["PUREMACRO_REPOSITORY"])
EVIDENCE = BASE / "installed-evidence"
EVIDENCE.mkdir(exist_ok=True)
assert str(ROOT) not in sys.path
assert (BASE / "offline_controls" / f"{os.getpid()}.json").exists()

import numpy as np
import pandas as pd
import scipy
import puremacro
from puremacro.datasets import load_enigh2024_deciles
from puremacro.structural.sw07_expectations import sw07_finite_sample_expectations
from puremacro.structural.sw07_sampling import sw07_finite_sample_moments
from puremacro.structural.empirical_sw07 import _PUBLISHED_MODE
from puremacro.examples.sw07_estimator_experiment import run_application as run_experiment
from puremacro.examples.sw07_estimator_experiment import load_experiment
from puremacro.examples.distributional_trade_ge import run_application as run_trade
from puremacro.validation import run_research_benchmarks

assert Path(puremacro.__file__).is_relative_to(INSTALLED)


def read_json(path):
    def reject(value):
        raise AssertionError(f"Nonstandard JSON number {value}")
    return json.loads(path.read_text(), parse_constant=reject)


def verify_manifest(directory):
    manifest = read_json(directory / "manifest.json")
    assert manifest["artifacts"]
    for name, recorded in manifest["artifacts"].items():
        assert Path(name).name == name
        expected = recorded["sha256"] if isinstance(recorded, dict) else recorded
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
    return manifest


# Check the fast exact-mean path against the separate full-moment covariance path.
means = sw07_finite_sample_expectations(_PUBLISHED_MODE, nobs=156, common_max_lag=4)
exact = sw07_finite_sample_moments(_PUBLISHED_MODE, nobs=156, common_max_lag=4)
np.testing.assert_allclose(means, exact.values, rtol=1e-11, atol=1e-12)

# Two draws exercise pairing, held-out phases, checkpoint replay and export.
# They are installation checks, not scientific Monte Carlo evidence.
cal_path = EVIDENCE / "calibration"
val_path = EVIDENCE / "validation"
cal = run_experiment(cal_path, replications=2, phase="calibration", seed=20261002)
replayed = run_experiment(cal_path, replications=2, phase="calibration", seed=20261002)
assert replayed.metadata["checkpoint_resumed_draws"] == 2
pd.testing.assert_frame_equal(cal.draws, replayed.draws)
np.testing.assert_array_equal(cal.draw_moments, replayed.draw_moments)
np.testing.assert_array_equal(cal.draw_covariances, replayed.draw_covariances)
val = run_experiment(val_path, replications=2, phase="validation", seed=20261002,
                     calibration=cal_path)
assert not set(cal.draws.simulation_seed) & set(val.draws.simulation_seed)
for study in (cal, val):
    assert len(study.draws) == 8 and len(study.summary) == 4
    assert study.draw_moments.shape == (2, 15)
    assert study.draw_covariances.shape == (2, 15, 15)
    assert study.metadata["parameter_inference_reported"] is False
    assert study.draws.groupby("replication").sample_sha256.nunique().eq(1).all()
    assert study.draws.groupby("replication").simulation_seed.nunique().eq(1).all()
for directory in (cal_path, val_path):
    verify_manifest(directory)
    loaded = load_experiment(directory)
    assert len(loaded.draws) == 8
assert read_json(val_path / "manifest.json")["calibration_manifest_sha256"] == hashlib.sha256(
    (cal_path / "manifest.json").read_bytes()).hexdigest()
calibrated = pd.read_csv(val_path / "calibrated_validation.csv")
assert len(calibrated) == 4
assert calibrated.critical_upper_unbounded.all()  # Honest B=2 finite-rank limit.

# The whole six-scenario GE-to-household application must use packaged data.
trade_path = EVIDENCE / "trade_ge"
trade = run_trade(trade_path)
manifest = verify_manifest(trade_path)
observed = load_enigh2024_deciles()
assert len(observed) == 10 and observed.attrs["is_synthetic"] is False
assert observed.households.sum() == 38_830_230
assert manifest["is_synthetic"] is True
assert manifest["observed_data"]["is_synthetic"] is False
assert manifest["model_assumptions"]["empirically_calibrated_trade_economy"] is False
assert manifest["ge_audit"]["both_converged"] is True
assert len(trade["results"]) == 6
pool = manifest["fiscal"]["incremental_tariff_revenue_mxn"]
assert pool > 0
fiscal = trade["fiscal_allocation"]
np.testing.assert_allclose(fiscal.household_rebate_mxn + fiscal.retained_outside_households_mxn,
                           fiscal.incremental_tariff_revenue_mxn, rtol=1e-12, atol=1e-4)
assert fiscal.income_residual_mxn.abs().max() < .1
for key, result in trade["results"].items():
    assert len(result.groups) == 10
    assert np.isfinite(result.groups.ev).all()
    assert np.isclose(observed.households @ result.groups.ev, result.aggregate["total_ev"])
zero = run_trade(EVIDENCE / "trade_zero", tariff_rate=0.)
verify_manifest(EVIDENCE / "trade_zero")
for result in zero["results"].values():
    np.testing.assert_allclose(result.groups.ev, 0, atol=1e-5)

# Exercise the public command entry point separately from Python imports.
command = [sys.executable, "-m", "puremacro.examples.distributional_trade_ge",
           "--output", str(EVIDENCE / "trade_cli")]
process = subprocess.run(command, cwd=BASE, capture_output=True, text=True)
(EVIDENCE / "trade-cli.log").write_text(process.stdout + process.stderr)
assert process.returncode == 0
verify_manifest(EVIDENCE / "trade_cli")
pd.testing.assert_frame_equal(pd.read_csv(trade_path / "all_scenarios.csv"),
                              pd.read_csv(EVIDENCE / "trade_cli/all_scenarios.csv"))

# Preserve the independent benchmark's documented scientific discrepancy.
benchmarks = run_research_benchmarks()
assert len(benchmarks.results) == 8 and not benchmarks.passed
assert [result.id for result in benchmarks.results if not result.passed] == ["rr2010_published_peak"]
assert all(not result.error for result in benchmarks.results)
peak = next(result for result in benchmarks.results if result.id == "rr2010_published_peak")
assert peak.provenance["atol"] == .005 and peak.provenance["rtol"] == 0
assert [name for name, metric in peak.metrics.items() if not metric["passed"]] == ["t_h10"]
benchmarks.write(EVIDENCE / "research_benchmarks")

assert "statsmodels" not in sys.modules
for name, module in tuple(sys.modules.items()):
    if name == "puremacro" or name.startswith("puremacro."):
        path = getattr(module, "__file__", None)
        if path:
            assert Path(path).is_relative_to(INSTALLED), (name, path)

summary = {
    "passed": True, "package_path": puremacro.__file__, "package_version": puremacro.__version__,
    "environment": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "pandas": pd.__version__},
    "network_blocked_with_positive_control": True,
    "statsmodels_blocked_with_positive_control": True,
    "all_loaded_package_modules_from_installed_wheel": True,
    "exact_mean_full_covariance_parity": True,
    "sw07_calibration_samples": 2, "sw07_validation_samples": 2,
    "sw07_variant_fits": 16, "sw07_checkpoint_replay_exact": True,
    "trade_ge_scenarios": 6, "trade_zero_tariff_control": True,
    "trade_api_cli_incidence_identical": True,
    "fiscal_max_residual_mxn": float(fiscal.fiscal_residual_mxn.abs().max()),
    "income_max_residual_mxn": float(fiscal.income_residual_mxn.abs().max()),
    "research_benchmarks_passed": 7, "research_benchmarks_total": 8,
    "documented_rr2010_discrepancy_preserved": True,
    "scope": "Installed workflow and dependency checks; tiny simulation counts are not scientific validation.",
}
(BASE / "installed-smoke.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
