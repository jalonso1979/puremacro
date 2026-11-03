"""Measure reduced HANK equations in isolated, sequential worker processes.

Peak RSS covers the whole worker, including imports, calibration and audits.
Matrix bytes are measured separately. Times are observations, never test gates.
Use a new output directory; failed rank/determinacy/audit cases are retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import scipy

from puremacro.models.one_asset_hank import calibrate_one_asset_hank, solve_one_asset_hank
from puremacro.models.hank_two_asset_equilibrium import calibrate_two_asset_equilibrium
from puremacro.models.hank_recursive import solve_hank_stochastic_policy
from puremacro.models.hank_reduction import solve_hank_reduced_policy
from tools.reference_validation.benchmark_hank_equilibrium import blas_threading, linear_algebra_build

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("puremacro/models/hank_reduction.py", "puremacro/models/hank_recursive.py",
    "puremacro/models/_hank_one_asset_recursive.py", "puremacro/models/hank_two_asset_equilibrium.py",
    "puremacro/models/hank_sequence_space.py", "puremacro/models/one_asset_hank.py",
    "puremacro/models/_hank_tangent.py", "puremacro/models/hank_capital.py",
    "tools/reference_validation/benchmark_hank_equilibrium.py",
    "tools/reference_validation/validate_hank_scaling.py")
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
CASES = {
    "one_small_full_controls": dict(model="one", specification=dict(n_a=80, n_e=3, a_max=150.), backend="legacy", distribution_rank=16),
    "one_small_reduced": dict(model="one", specification=dict(n_a=80, n_e=3, a_max=150.), backend="reduced", distribution_rank=16, marginal_rank=64),
    "one_default_low_rank": dict(model="one", specification=dict(n_a=500, n_e=7, a_max=150.), backend="reduced", distribution_rank=24, marginal_rank=128),
    "one_default_coarse": dict(model="one", specification=dict(n_a=500, n_e=7, a_max=150.), backend="reduced", distribution_rank=96, marginal_rank=64),
    "one_default": dict(model="one", specification=dict(n_a=500, n_e=7, a_max=150.), backend="reduced", distribution_rank=96, marginal_rank=128, sequence_reference=True),
    "one_default_finer": dict(model="one", specification=dict(n_a=500, n_e=7, a_max=150.), backend="reduced", distribution_rank=128, marginal_rank=192),
    "two_small_full_controls": dict(model="two", specification=dict(n_a=12, n_b=12, a_max=80., b_max=10.), backend="legacy", distribution_rank=16),
    "two_small_reduced": dict(model="two", specification=dict(n_a=12, n_b=12, a_max=80., b_max=10.), backend="reduced", distribution_rank=16, marginal_rank=64),
    "two_medium": dict(model="two", specification=dict(n_a=30, n_b=30, a_max=80., b_max=10.), backend="reduced", distribution_rank=24, marginal_rank=80),
    "two_fine_coarse": dict(model="two", specification=dict(n_a=40, n_b=40, a_max=80., b_max=10.), backend="reduced", distribution_rank=24, marginal_rank=48),
    "two_fine": dict(model="two", specification=dict(n_a=40, n_b=40, a_max=80., b_max=10.), backend="reduced", distribution_rank=24, marginal_rank=80),
    "two_fine_more_controls": dict(model="two", specification=dict(n_a=40, n_b=40, a_max=80., b_max=10.), backend="reduced", distribution_rank=24, marginal_rank=128),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def peak_resident_bytes():
    try:
        import resource
    except ImportError:
        return None
    units = 1 if sys.platform == "darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*units)


def worker(name, folder):
    spec = CASES[name]
    record = dict(name=name, settings=spec, completed=False, solved=False,
                  artifact_sha256={}, memory_scope="worker lifetime including imports, calibration, solve, audits and any sequence reference")
    start = perf_counter()
    def frame(stem, data):
        path = folder/(name+"_"+stem+".csv")
        data.to_csv(path, float_format="%.17g")
        record["artifact_sha256"][path.name] = sha(path)
    with blas_threading() as control:
        record["blas_control"] = control
        try:
            calibration_start = perf_counter()
            ss = (calibrate_one_asset_hank(**spec["specification"]) if spec["model"] == "one"
                  else calibrate_two_asset_equilibrium(**spec["specification"]))
            record.update(calibration_seconds=perf_counter()-calibration_start,
                calibration=ss.calibration, steady_state=ss.aggregates, steady_diagnostics=ss.diagnostics)
            solve_start = perf_counter()
            if spec["backend"] == "legacy":
                policy = solve_hank_stochastic_policy(ss, distribution_rank=spec["distribution_rank"])
            else:
                policy = solve_hank_reduced_policy(ss, distribution_rank=spec["distribution_rank"],
                    marginal_rank=spec["marginal_rank"], krylov_steps=96)
            record.update(solve_seconds=perf_counter()-solve_start, peak_resident_bytes_at_solve=peak_resident_bytes(),
                          solved=True, policy=policy.diagnostics,
                          state_basis=policy.basis.diagnostics)
            if spec["backend"] == "reduced":
                record["control_basis"] = policy.marginal_basis.diagnostics
            path = folder/(name+"_policy.npz")
            arrays = dict(transition=policy.transition, feedback=policy.feedback, basis=policy.basis.vectors,
                          constraints=policy.basis.constraints, center=policy.basis.center,
                          F0=policy._F0, F1=policy._F1)
            if spec["backend"] == "reduced":
                arrays.update(marginal_basis=policy.marginal_basis.vectors, marginal_center=policy.marginal_basis.center,
                              reduced_feedback=policy.reduced_feedback, full_residual=policy._full_residual)
            np.savez_compressed(path, **arrays)
            record["artifact_sha256"][path.name] = sha(path)
            held_out = np.random.default_rng(9917).normal(0, 1e-4, 48)
            frame("linear", policy.simulate(held_out).paths)
            record["audits"] = []
            for multiplier in (1., .25):
                before = perf_counter()
                audit = policy.audit(multiplier*held_out, tolerance=1e-5)
                record["audits"].append(dict(multiplier=multiplier, converged=audit.converged,
                    seconds=perf_counter()-before, diagnostics=audit.diagnostics))
                frame("audit" if multiplier == 1 else "quarter_audit", audit.paths)
            if spec.get("sequence_reference"):
                # Different solver and a fixed steady terminal boundary at T=200.
                # Compare only the first30dates; remaining horizon error is explicit.
                before = perf_counter()
                reference = solve_one_asset_hank(ss, horizon=200, magnitude=-1e-5, rho=.7, nonlinear=False)
                impulse = policy.simulate(np.r_[-1e-5, np.zeros(199)]).paths
                actual = pd.DataFrame(dict(pi=impulse.pi, w=ss.calibration["w"]*impulse.w, Y=impulse.Y))
                frame("sequence_reference", reference.paths[["pi", "w", "Y"]])
                frame("recursive_impulse", actual)
                gap = float(np.max(abs(actual.iloc[:30].to_numpy()-reference.paths[["pi", "w", "Y"]].iloc[:30].to_numpy())))
                record["sequence_reference"] = dict(seconds=perf_counter()-before, horizon=200,
                    comparison_periods=30, impulse=-1e-5, rho=.7, max_abs_gap=gap,
                    converged=reference.converged, diagnostics=reference.diagnostics,
                    scope="independent sequence-space solver at finite horizon; not an infinite-tail error bound")
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            record["error"] = dict(type=type(exc).__name__, message=str(exc))
        record.update(completed=True, worker_seconds=perf_counter()-start, peak_resident_bytes=peak_resident_bytes())
        save(folder/(name+".json"), record)


def run(output, names):
    if output.exists():
        raise ValueError("use a fresh output directory to preserve earlier evidence")
    output.mkdir(parents=True)
    record = dict(schema_version=1, completed=False, started_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
        threads={key: os.environ.get(key) for key in THREADS},
        numpy_linear_algebra=linear_algebra_build(np), scipy_linear_algebra=linear_algebra_build(scipy),
        source_sha256={name: sha(ROOT/name) for name in SOURCES}, artifact_sha256={}, cases=[], comparisons=[],
        held_out_seed=9917, held_out_std=1e-4, held_out_length=48, audit_tolerance=1e-5,
        timing_context="sequential benchmark workers; regression checks may share this host; no exclusive-machine speed claim",
        scope="same-grid state/control reduction with actual full-grid audits; separate finite-grid sensitivity",
        grid_caveat="two-asset supplies are calibrated at each grid; grid comparisons are sensitivity, not a fixed-supply convergence certificate")
    if any(value != "1" for value in record["threads"].values()):
        raise ValueError("set all four BLAS thread environment variables to1 before benchmarking")
    save(output/"record.json", record)
    for name in names:
        print(name, flush=True)
        with (output/(name+".log")).open("w") as log:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", name,
                                      "--output", str(output.resolve())], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        path = output/(name+".json")
        if process.returncode or not path.exists():
            item = dict(name=name, completed=False, process_exit_code=process.returncode,
                        error="worker did not complete; see retained log")
        else:
            item = json.loads(path.read_text())
            record["artifact_sha256"][path.name] = sha(path)
            record["artifact_sha256"].update(item["artifact_sha256"])
        record["cases"].append(item)
        save(output/"record.json", record)
    indexed = {item["name"]: item for item in record["cases"]}
    for a, b, kind in (("one_small_full_controls", "one_small_reduced", "control_reduction"),
                      ("two_small_full_controls", "two_small_reduced", "control_reduction"),
                      ("one_default_coarse", "one_default", "control_refinement"),
                      ("one_default", "one_default_finer", "joint_rank_refinement"),
                      ("two_fine_coarse", "two_fine", "control_refinement"),
                      ("two_fine", "two_fine_more_controls", "control_refinement"),
                      ("two_medium", "two_fine", "grid_sensitivity")):
        if a in indexed and b in indexed and indexed[a].get("solved") and indexed[b].get("solved"):
            left = pd.read_csv(output/(a+"_linear.csv"), index_col=0, float_precision="round_trip")
            right = pd.read_csv(output/(b+"_linear.csv"), index_col=0, float_precision="round_trip")
            record["comparisons"].append(dict(left=a, right=b, kind=kind,
                held_out_log_path_max_abs_gap=float(np.max(abs(left.to_numpy()-right.to_numpy())))))
    record.update(completed=all(item["completed"] for item in record["cases"]),
                  finished_utc=datetime.now(timezone.utc).isoformat())
    save(output/"record.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=list(CASES))
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
    else:
        run(args.output, args.cases)
