"""Replay selected saved calibration failures; never replace study results."""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import scipy

from puremacro.structural import sw07_estimator_experiment as experiment
from puremacro.structural.bridge import MomentTargets
from puremacro.structural.empirical_sw07 import ALL_MOMENTS, UNITS, covariance_moment_targets
from puremacro.structural.sw07_sampling import simulate_sw07_sample


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "calibration"
PACKAGE = ROOT.parents[1] / "puremacro"


def main():
    manifest_path = ARTIFACTS / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest() == digest, name
    for name, digest in manifest["source_hashes"].items():
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == digest, name
    environment = {"python": platform.python_version(), "numpy": np.__version__,
                   "scipy": scipy.__version__, "pandas": pd.__version__}
    assert environment == manifest["environment"]
    draws = pd.read_csv(ARTIFACTS / "draws.csv", dtype={"simulation_seed": str})
    with np.load(ARTIFACTS / "draw_evidence.npz", allow_pickle=False) as saved:
        moments, covariances = saved["moments"].copy(), saved["covariances"].copy()
    assert moments.shape == (399, 15) and covariances.shape == (399, 15, 15)
    causes = {
        "first_order_stationarity": draws.error.fillna("").str.contains("stationarity"),
        "start_failure": draws.start_successes < len(manifest["starts"]),
        "start_disagreement": ~draws.starts_agree,
    }
    selected, inventory = {}, []
    for variant in manifest["variants"]:
        row = {"variant": variant, "unresolved": int((draws.unresolved & (draws.variant == variant)).sum())}
        for cause, mask in causes.items():
            eligible = draws.loc[draws.unresolved & (draws.variant == variant) & mask].sort_values("replication")
            row[cause] = len(eligible)
            if len(eligible):
                index = int(eligible.index[0])
                selected.setdefault(index, []).append(cause)
        inventory.append(row)
    control = draws.loc[(draws.variant == "finite_oracle") & ~draws.unresolved].sort_values("replication").index[0]
    selected[int(control)] = ["usable_oracle_control"]
    units = tuple(f"({UNITS[a]}) * ({UNITS[b]})" for a, b, _ in ALL_MOMENTS)
    oracle = np.asarray(manifest["oracle_covariance"], dtype=float)
    fitters = experiment._fitters(manifest["dgp"])
    original_fitter = experiment.fit_structural
    captured = []

    def instrumented_fitter(*args, **kwargs):
        result = original_fitter(*args, **kwargs)
        diagnostics = result.diagnostics
        # Independently reconstruct the local first-order projection from a
        # Cholesky factor of the complete fitted-moment covariance. The returned
        # derivative is newly computed in this replay, not stored in the CSV.
        root = np.linalg.cholesky(result.targets.covariance)
        residual = np.linalg.solve(root, result.model_moments-result.targets.values)
        derivative = np.linalg.solve(root, result.jacobian)
        left, _, _ = np.linalg.svd(derivative, full_matrices=False)
        projection = float(np.linalg.norm(left[:, :result.identification_rank].T @ residual)
                           / diagnostics["optimizer_residual_scale"])
        np.testing.assert_allclose(projection, diagnostics["projected_residual_norm"], rtol=1e-7, atol=1e-11)
        captured.append({"start": list(args[2]), "theta": result.theta.tolist(),
            "objective": float(result.objective), "success": bool(result.success),
            "evaluations": int(result.n_evals), "message": result.message,
            "boundary": result.boundary.tolist(), "identification_rank": int(result.identification_rank),
            "inference_unavailable_reasons": list(diagnostics["inference_unavailable_reasons"]),
            "projected_residual_norm": float(diagnostics["projected_residual_norm"]),
            "independently_reconstructed_projection": projection,
            "stationarity_tolerance": float(diagnostics["stationarity_tolerance"]),
            "projection_to_tolerance": float(projection/diagnostics["stationarity_tolerance"]),
            "weighted_first_order_gradient": (derivative.T @ residual).tolist(),
            "optimizer_optimality": float(diagnostics["optimizer_optimality"])})
        return result

    experiment.fit_structural = instrumented_fitter
    records = []
    try:
        for index, selected_causes in selected.items():
            original = draws.loc[index]
            i = int(original.array_row)
            target = MomentTargets(moments[i], covariances[i], tuple(manifest["moment_labels"]), units,
                                   {"common_max_lag": 4, "observations": 156, "effective_observations": 152})
            captured.clear()
            repeated = fitters[original.variant](experiment._weighted_target(target, oracle, original.variant))
            starts = json.loads(original.start_diagnostics_json)
            exact_start_match = True
            for old, new in zip(starts, captured):
                for key in ("start", "theta", "objective", "success", "message", "evaluations", "inference_unavailable_reasons"):
                    exact_start_match &= old[key] == new[key]
            objective_difference = float(repeated["objective"]-original.objective)
            theta_difference = np.array([repeated["crr"]-original.crr, repeated["em"]-original.em])
            np.testing.assert_allclose(objective_difference, 0., atol=1e-9)
            np.testing.assert_allclose(theta_difference, 0., atol=1e-12)
            assert repeated["unresolved"] == original.unresolved
            assert repeated["status"] == original.status
            records.append({"variant": original.variant, "replication": int(original.replication),
                "array_row": i, "selection_causes": selected_causes,
                "original_status": original.status, "replay_status": repeated["status"],
                "original_unresolved": bool(original.unresolved), "replay_unresolved": bool(repeated["unresolved"]),
                "original_objective": float(original.objective), "replay_objective": repeated["objective"],
                "objective_difference": objective_difference, "parameter_difference": theta_difference.tolist(),
                "all_four_original_start_records_exactly_reproduced": bool(exact_start_match),
                "start_replays": list(captured)})
    finally:
        experiment.fit_structural = original_fitter
    # Regenerate one actual unresolved sample from its saved independent stream.
    first = draws.loc[next(index for index in selected if draws.loc[index, "unresolved"])]
    sample = simulate_sw07_sample(manifest["dgp"], nobs=156, seed=int(first.simulation_seed))
    sample_digest = hashlib.sha256(sample.to_numpy(dtype="<f8").tobytes()).hexdigest()
    assert sample_digest == first.sample_sha256
    target = covariance_moment_targets(sample, ALL_MOMENTS, units=UNITS, bandwidth=8)
    np.testing.assert_array_equal(target.values, moments[int(first.array_row)])
    np.testing.assert_array_equal(target.covariance, covariances[int(first.array_row)])
    result = {"status": "passed", "original_outputs_changed": False,
        "selection_rule": "earliest saved unresolved calibration draw per variant and observed cause, plus earliest usable finite_oracle control",
        "calibration_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "environment": environment, "source_hashes_verified": len(manifest["source_hashes"]),
        "artifact_hashes_verified": len(manifest["artifacts"]),
        "failure_inventory": inventory, "replays": records,
        "sample_regeneration": {"replication": int(first.replication), "simulation_seed": first.simulation_seed,
            "sample_sha256": sample_digest, "sample_hash_matched": True,
            "all_15_moments_and_full_hac_matched_exactly": True},
        "limitations": ["Replay uses the same frozen model and optimizer; it is not an independent estimator implementation",
            "First-order derivatives and projected residuals were recomputed, since the original records retain reasons but not those numerical diagnostics",
            "Only observed failure causes can be selected; calibration contained no failed starts or disagreements",
            "Successful replay does not turn unresolved fits into usable fits or prove global optimality",
            "No original failure, Monte Carlo draw, rate denominator or source file was replaced"]}
    (ROOT / "failure_replay.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
