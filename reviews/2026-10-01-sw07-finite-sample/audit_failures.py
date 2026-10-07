"""Summarize saved unresolved draws and replay representative saved targets.

Run from the repository root with .venv/bin/python
reviews/2026-10-01-sw07-finite-sample/audit_failures.py.
Original scenario artifacts and failure decisions are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy

from puremacro.structural.bridge import MomentTargets
from puremacro.structural.empirical_sw07 import ALL_MOMENTS, UNITS, _labels
import puremacro.structural.sw07_finite_sample as module

SCENARIOS = ("baseline_fixed", "published_fixed", "baseline_fitted", "published_fitted")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_ready(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def selected_start(records):
    finite = [record for record in records if np.isfinite(record.get("objective", np.nan))]
    successful = [record for record in finite if record.get("success")]
    return min(successful or finite, key=lambda record: record["objective"]) if finite else None


def classify(row, records):
    reasons = []
    best = selected_start(records)
    if best and "nonstationary_solution" in best.get("inference_unavailable_reasons", []):
        reasons.append("selected_stationarity_failure")
    if not bool(row["starts_agree"]):
        reasons.append("multistart_disagreement")
    if int(row["start_successes"]) < 4:
        reasons.append("unsuccessful_start")
    if not reasons:
        reasons.append("other_unresolved" if bool(row["unresolved"]) else "valid_boundary_control")
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=REVIEW)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    output = args.review / "failure_audit"
    output.mkdir(parents=True, exist_ok=True)
    inputs, source_checks, scenario_rows, failures, starts, replay_results = {}, {}, [], [], [], []
    replay_candidates = []
    for scenario in SCENARIOS:
        folder = args.review / scenario
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        inputs[str(manifest_path)] = sha256(manifest_path)
        for filename in ("draws.csv", "draw_evidence.npz", "summary.csv"):
            path = folder / filename
            actual = sha256(path)
            if actual != manifest["artifacts"][filename]:
                raise ValueError(f"Source artifact checksum mismatch: {path}")
            inputs[str(path)] = actual
        checkpoints = list((folder / "_checkpoints").glob("*.npz"))
        if len(checkpoints) != 1:
            raise ValueError(f"Expected one final checkpoint for {scenario}")
        with np.load(checkpoints[0], allow_pickle=False) as saved:
            document = json.loads(str(saved["document"]))
        design = document["design"]
        fingerprint = hashlib.sha256(json.dumps(design, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if fingerprint != document["fingerprint"]:
            raise ValueError(f"Checkpoint design fingerprint mismatch: {scenario}")
        source_checks[scenario] = {
            name: sha256(ROOT / "puremacro" / name) == digest
            for name, digest in design["source_hashes"].items()}
        if not all(source_checks[scenario].values()):
            raise ValueError(f"Mathematical sources changed since the saved run: {scenario}")
        environment = {"python": platform.python_version(), "numpy": np.__version__,
                       "scipy": scipy.__version__, "pandas": pd.__version__}
        if any(design[key] != value for key, value in environment.items()):
            raise ValueError(f"Numerical software versions changed since the saved run: {scenario}")
        inputs[str(checkpoints[0])] = sha256(checkpoints[0])
        frame = pd.read_csv(folder / "draws.csv", float_precision="round_trip")
        summary = pd.read_csv(folder / "summary.csv", float_precision="round_trip").iloc[0]
        if len(frame) != int(summary.replications) or int(frame.unresolved.sum()) != int(summary.unresolved_draws):
            raise ValueError(f"Saved count mismatch: {scenario}")
        if tuple(manifest["moment_labels"]) != _labels(ALL_MOMENTS):
            raise ValueError("Saved labels do not follow the declared fifteen-moment order")
        categories = Counter()
        selected_categories = set()
        for _, row in frame.loc[frame.unresolved].iterrows():
            records = json.loads(row.start_diagnostics_json)
            if len(records) != 4:
                raise ValueError("Each saved draw must retain all four start records")
            reasons = classify(row, records)
            categories.update(reasons)
            failures.append({"scenario": scenario, "replication": int(row.replication),
                             "array_row": int(row.array_row), "original_objective": row.objective,
                             "original_status": row.status, "original_start_successes": int(row.start_successes),
                             "original_starts_agree": bool(row.starts_agree),
                             "causes": ";".join(reasons)})
            for number, record in enumerate(records):
                starts.append({"scenario": scenario, "replication": int(row.replication),
                               "start_number": number, "start_crr": record["start"][0],
                               "start_em": record["start"][1], "success": record.get("success", False),
                               "objective": record.get("objective"), "evaluations": record.get("evaluations"),
                               "message": record.get("message", record.get("exception", "")),
                               "inference_unavailable_reasons": ";".join(record.get("inference_unavailable_reasons", []))})
            # Select the earliest unresolved draw for each distinct cause within
            # each scenario, before attempting replay; no favorable-result selection.
            if any(cause not in selected_categories for cause in reasons):
                replay_candidates.append((folder, manifest, row.to_dict(), reasons))
                selected_categories.update(reasons)
        scenario_rows.append({"scenario": scenario, "replications": len(frame),
                              "original_unresolved": int(frame.unresolved.sum()),
                              "selected_stationarity_failures": categories["selected_stationarity_failure"],
                              "multistart_disagreements": categories["multistart_disagreement"],
                              "unsuccessful_starts": categories["unsuccessful_start"],
                              "other_unresolved": categories["other_unresolved"]})
        if scenario == "baseline_fixed":
            boundary = frame.loc[~frame.unresolved & frame.boundary]
            if not boundary.empty:
                replay_candidates.append((folder, manifest, boundary.iloc[0].to_dict(), ["valid_boundary_control"]))

    if not args.summary_only:
        original_engine = module.fit_structural
        try:
            for folder, manifest, row, selection_reason in replay_candidates:
                scenario = row["scenario"]
                with np.load(folder / "draw_evidence.npz", allow_pickle=False) as saved:
                    values = saved["moments"][int(row["array_row"])].copy()
                    covariance = saved["covariances"][int(row["array_row"])].copy()
                if not np.isfinite(values).all() or not np.isfinite(covariance).all():
                    replay_results.append({"scenario": scenario, "replication": row["replication"],
                                           "not_replayed": "saved targets unavailable", "original_decision_preserved": True})
                    continue
                targets = MomentTargets(values, covariance, tuple(manifest["moment_labels"]),
                                        tuple(f"({UNITS[a]}) * ({UNITS[b]})" for a, b, _ in ALL_MOMENTS),
                                        {"source": "authenticated saved Monte Carlo draw", "array_row": row["array_row"]})
                calls = []

                def tracking_engine(*engine_args, **engine_kwargs):
                    start = np.asarray(engine_args[2], dtype=float).tolist()
                    try:
                        result = original_engine(*engine_args, **engine_kwargs)
                    except Exception as exc:
                        calls.append({"start": start, "exception": f"{type(exc).__name__}: {exc}"})
                        raise
                    calls.append({"start": start, "success": result.success, "message": result.message,
                                  "evaluations": result.n_evals, "theta": result.theta.tolist(),
                                  "objective": result.objective,
                                  "inference_unavailable_reasons": list(result.diagnostics["inference_unavailable_reasons"]),
                                  "projected_residual_norm": result.diagnostics["projected_residual_norm"],
                                  "stationarity_tolerance": result.diagnostics["stationarity_tolerance"]})
                    return result

                module.fit_structural = tracking_engine
                # The stored DGP dictionary contains the complete fixed calibration.
                # _fitter overrides crr/em at every candidate, so fixed and plug-in
                # scenarios replay the identical original estimation map.
                result, diagnostics = module._fitter(manifest["scenarios"][scenario]["dgp"])(targets)
                record = module._fit_record(result, diagnostics)
                objective_difference = float(result.objective - row["objective"])
                original_calls = json.loads(row["start_diagnostics_json"])
                replay_results.append({"scenario": scenario, "replication": int(row["replication"]),
                    "selection_reasons": selection_reason,
                    "original_status": row["status"], "replay_status": record["status"],
                    "same_unresolved_decision": bool(record["unresolved"] == row["unresolved"]),
                    "same_boundary_decision": bool(record["boundary"] == row["boundary"]),
                    "original_objective": row["objective"], "replay_objective": result.objective,
                    "objective_difference": objective_difference,
                    "objective_matches_at_1e-8_relative": bool(abs(objective_difference) <= 1e-8*max(1., abs(row["objective"]))),
                    "original_start_diagnostics": original_calls, "replayed_start_diagnostics": calls,
                    "original_decision_preserved": True,
                    "note": "Replay is a diagnostic only; neither counts nor original draw records are replaced."})
        finally:
            module.fit_structural = original_engine

    for path, expected in inputs.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Original evidence changed during audit: {path}")
    pd.DataFrame(scenario_rows).to_csv(output / "failure_counts.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "unresolved_draws.csv", index=False)
    pd.DataFrame(starts).to_csv(output / "start_diagnostics.csv", index=False)
    document = {"original_results_unchanged": True, "input_sha256": inputs,
                "mathematical_source_matches": source_checks, "scenario_counts": scenario_rows,
                "unresolved_total": len(failures), "stored_start_records": len(starts),
                "stationarity_failure_definition": "nonstationary_solution is the optimizer's interior first-order check: weighted-residual projection onto the local Jacobian column space. It does not mean nonstationary DSGE dynamics; all generating DGPs passed the dynamic stationarity check.",
                "replay_selection": "earliest unresolved draw per cause per scenario, plus earliest usable baseline boundary draw",
                "replays": replay_results, "summary_only": args.summary_only,
                "scope": "post-run computational audit; no replacement of failures, inferential decisions or Monte Carlo counts"}
    (output / "audit.json").write_text(json.dumps(json_ready(document), indent=2, allow_nan=False)+"\n")
    replayed = [row for row in replay_results if "same_unresolved_decision" in row]
    matches = sum(row["same_unresolved_decision"] for row in replayed)
    readme = (
        "# Post-run unresolved-draw audit\n\n"
        f"All original artifacts and decisions remain unchanged. The four scenarios contain **{len(failures)} unresolved draws** "
        f"with **{len(starts)} saved optimizer-start records**.\n\n"
        "The recorded causes are separated into selected-solution stationarity failures, disagreement among "
        "the four starts, and unsuccessful starts. These are computational diagnostics, not additional model "
        "rejections. In particular, SciPy success alone does not override the weighted-residual stationarity check.\n\n"
        "Here `nonstationary_solution` means failure of the optimizer's interior first-order condition: "
        "the weighted-residual projection onto the local Jacobian column space exceeds its numerical tolerance. "
        "It does **not** mean nonstationary DSGE dynamics; all four generating DGPs passed dynamic stationarity checks.\n\n"
        "failure_counts.csv reports the original counts by scenario; unresolved_draws.csv and start_diagnostics.csv "
        "retain each failure and termination message. Cause flags can overlap in general.\n\n"
        f"Representative replay: **{matches}/{len(replayed)}** saved unresolved/usable decisions reproduced. "
        "The replay uses authenticated NPZ targets and their full covariance, the original four starts, fixed calibration, "
        "bounds and solver. It does not resimulate data or substitute re-estimated targets. audit.json records each "
        "replay, per-start stationarity residual/tolerance, numerical differences, all input hashes and matches "
        "against the checkpoint's nine mathematical source hashes and numerical-library versions.\n\n"
        "No replay result replaces an original failure or changes a Monte Carlo denominator or tail bound. "
        "Tiny optimizer-path differences can arise from numerical state or evaluation history; any replay disagreement "
        "remains visible in audit.json.\n\n"
        "Reproduce from the repository root with `.venv/bin/python reviews/2026-10-01-sw07-finite-sample/audit_failures.py`. "
        "Use `--summary-only` to summarize existing records without optimizer calls.\n"
    )
    (output / "README.md").write_text(readme)
    print(json.dumps({"unresolved": len(failures), "replayed": len(replayed), "same_decisions": matches,
                      "original_results_unchanged": True, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
