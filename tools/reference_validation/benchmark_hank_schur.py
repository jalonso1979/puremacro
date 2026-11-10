"""Compare real/complex HANK Schur paths in fresh sequential processes.

Run with PYTHONPATH=. and a new --output directory. Timings are observations,
never acceptance gates. First repetitions retain full policies and nonlinear
audits; every repetition records peak RSS at the end of the solve. Historical
failed ranks are attempted once per backend and remain in the output.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter

import numpy as np
import scipy

from puremacro.models import calibrate_one_asset_hank, calibrate_two_asset_equilibrium
from puremacro.models import hank_reduction, _hank_schur
from tools.reference_validation.benchmark_hank_equilibrium import blas_threading, linear_algebra_build
from tools.reference_validation.validate_hank_scaling import peak_resident_bytes

ROOT = Path(__file__).resolve().parents[2]
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
SOURCES = ("puremacro/models/hank_reduction.py", "puremacro/models/_hank_schur.py",
           "puremacro/models/hank_recursive.py", "puremacro/models/_hank_one_asset_recursive.py",
           "puremacro/models/hank_two_asset_equilibrium.py", "puremacro/models/_hank_tangent.py",
           "puremacro/models/one_asset_hank.py", "puremacro/models/hank_sequence_space.py",
           "puremacro/models/hank_capital.py",
           "tools/reference_validation/benchmark_hank_equilibrium.py",
           "tools/reference_validation/validate_hank_scaling.py",
           "tools/reference_validation/benchmark_hank_schur.py")
CASES = {
    "one_default": dict(model="one", grid={}, distribution_rank=96, marginal_rank=128),
    "one_finer": dict(model="one", grid={}, distribution_rank=128, marginal_rank=192),
    "two_fine": dict(model="two", grid=dict(n_a=40, n_b=40, a_max=80., b_max=10.),
                     distribution_rank=24, marginal_rank=48),
    "one_low_rank": dict(model="one", grid={}, distribution_rank=24, marginal_rank=128,
                         historical_failure=True),
    "two_sensitive_rank": dict(model="two", grid=dict(n_a=40, n_b=40, a_max=80., b_max=10.),
                               distribution_rank=24, marginal_rank=80, historical_failure=True),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def worker(folder, name, output, repetition):
    spec = CASES[name]
    stem = f"{name}_{output}_{repetition}"
    record = dict(name=name, schur_output=output, repetition=repetition, settings=spec,
                  completed=False, solved=False, artifact_sha256={}, phases_seconds={})
    phases = record["phases_seconds"]
    originals = []

    def instrument(module, attr, key):
        original = getattr(module, attr)
        originals.append((module, attr, original))

        @wraps(original)
        def measured(*args, **kwargs):
            start = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                phases[key] = phases.get(key, 0.)+perf_counter()-start
        setattr(module, attr, measured)

    for attr, key in (("_make_basis", "distribution_basis"),
                      ("_make_marginal_basis", "marginal_basis"),
                      ("_solve_schur_policy", "schur_policy")):
        instrument(hank_reduction, attr, key)
    instrument(_hank_schur, "ordqz", "qz_decomposition")
    start = perf_counter()
    with blas_threading() as control:
        record["blas_control"] = control
        try:
            before = perf_counter()
            calibrate = calibrate_one_asset_hank if spec["model"] == "one" else calibrate_two_asset_equilibrium
            ss = calibrate(**spec["grid"])
            record.update(calibration_seconds=perf_counter()-before, calibration=ss.calibration,
                          steady_diagnostics=ss.diagnostics, household_shape=list(ss.distribution.shape))
            before = perf_counter()
            try:
                policy = hank_reduction.solve_hank_reduced_policy(
                    ss, distribution_rank=spec["distribution_rank"], marginal_rank=spec["marginal_rank"],
                    krylov_steps=96, schur_output=output)
            finally:
                record["solve_seconds"] = perf_counter()-before
                record["peak_resident_bytes_at_solve"] = peak_resident_bytes()
            phases["assembly_and_other_checks"] = record["solve_seconds"]-sum(
                phases.get(key, 0.) for key in ("distribution_basis", "marginal_basis", "schur_policy"))
            record.update(solved=True, diagnostics=policy.diagnostics,
                          distribution_basis=policy.basis.diagnostics,
                          marginal_basis=policy.marginal_basis.diagnostics)
            if repetition == 0:
                path = folder/(stem+".npz")
                innovations = np.random.default_rng(9917).normal(0., 1e-4, 48)
                linear = policy.simulate(innovations).paths
                arrays = dict(transition=policy.transition, feedback=policy.feedback,
                              reduced_feedback=policy.reduced_feedback, F0=policy._F0, F1=policy._F1,
                              distribution_basis=policy.basis.vectors, marginal_basis=policy.marginal_basis.vectors,
                              full_residual=policy._full_residual, innovations=innovations,
                              linear=linear.to_numpy(), linear_columns=np.asarray(linear.columns))
                record["audits"] = []
                for scale, key in ((1., "full"), (.25, "quarter")):
                    audit = policy.audit(scale*innovations, tolerance=1e-5)
                    arrays[key+"_audit"] = audit.paths.to_numpy()
                    record["audits"].append(dict(scale=scale, converged=audit.converged,
                                                diagnostics=audit.diagnostics))
                np.savez_compressed(path, **arrays)
                record["artifact_sha256"][path.name] = sha(path)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            record["error"] = dict(type=type(exc).__name__, message=str(exc))
        finally:
            record.update(completed=True, worker_seconds=perf_counter()-start,
                          peak_resident_bytes=peak_resident_bytes())
            for module, attr, original in reversed(originals):
                setattr(module, attr, original)
            save(folder/(stem+".json"), record)


def comparisons(folder, cases):
    result = []
    for name in CASES:
        pair = [next(c for c in cases if c["name"] == name and c["schur_output"] == output
                     and c["repetition"] == 0) for output in ("real", "complex")]
        row = dict(name=name, both_solved=all(c["solved"] for c in pair))
        if row["both_solved"]:
            with np.load(folder/f"{name}_real_0.npz", allow_pickle=False) as a, \
                    np.load(folder/f"{name}_complex_0.npz", allow_pickle=False) as b:
                row["same_equations_and_bases"] = all(np.array_equal(a[key], b[key]) for key in
                    ("F0", "F1", "distribution_basis", "marginal_basis", "innovations"))
                row["max_abs_gaps"] = {key: float(np.max(abs(a[key]-b[key]))) for key in
                    ("transition", "feedback", "reduced_feedback", "full_residual", "linear",
                     "full_audit", "quarter_audit")}
                row["policy_close"] = all(np.allclose(a[key], b[key], atol=1e-6, rtol=1e-8)
                    for key in ("transition", "feedback", "reduced_feedback", "full_residual"))
                row["audit_decisions_equal"] = all(a["converged"] == b["converged"]
                    for a, b in zip(pair[0]["audits"], pair[1]["audits"]))
                row["accuracy_passed"] = bool(row["same_equations_and_bases"] and row["policy_close"]
                    and row["audit_decisions_equal"] and row["max_abs_gaps"]["linear"] < 1e-10
                    and max(row["max_abs_gaps"]["full_audit"], row["max_abs_gaps"]["quarter_audit"]) < 1e-8)
        for output in ("real", "complex"):
            matching = [c for c in cases if c["name"] == name and c["schur_output"] == output and c["solved"]]
            if matching:
                row[output] = dict(repetitions=len(matching),
                    median_solve_seconds=median(c["solve_seconds"] for c in matching),
                    median_qz_seconds=median(c["phases_seconds"]["qz_decomposition"] for c in matching),
                    median_peak_resident_bytes_at_solve=median(c["peak_resident_bytes_at_solve"] for c in matching))
        if "real" in row and "complex" in row:
            row["total_solve_speedup"] = row["complex"]["median_solve_seconds"]/row["real"]["median_solve_seconds"]
            row["qz_speedup"] = row["complex"]["median_qz_seconds"]/row["real"]["median_qz_seconds"]
        result.append(row)
    return result


def run(folder, repeats):
    folder.mkdir(parents=True, exist_ok=False)
    record = dict(schema_version=1, completed=False, started_utc=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
                  scipy=scipy.__version__, numpy_build=linear_algebra_build(np), scipy_build=linear_algebra_build(scipy),
                  threads={key: "1" for key in THREADS}, repeats=repeats, cases=[],
                  source_sha256={name: sha(ROOT/name) for name in SOURCES}, artifact_sha256={},
                  timing_scope="fresh sequential workers on a shared host; no wall-clock acceptance thresholds",
                  memory_scope="peak RSS at solve includes imports and calibration; not retained array bytes",
                  phase_scope="qz_decomposition is nested in schur_policy; other reported phases are disjoint")
    env = dict(os.environ, **{key: "1" for key in THREADS}, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT))
    for name, spec in CASES.items():
        count = 1 if spec.get("historical_failure") else repeats
        for repetition in range(count):
            # Alternate order across repetitions to limit systematic order bias.
            order = ("complex", "real") if repetition % 2 == 0 else ("real", "complex")
            for output in order:
                stem = f"{name}_{output}_{repetition}"
                log = folder/(stem+".log")
                with log.open("w") as stream:
                    process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--output", str(folder), "--worker", name, "--schur-output", output,
                        "--repetition", str(repetition)], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                if process.returncode:
                    raise RuntimeError(f"worker {stem} exited {process.returncode}; inspect {log}")
                path = folder/(stem+".json")
                case = json.loads(path.read_text())
                record["cases"].append(case)
                record["artifact_sha256"].update({path.name: sha(path), log.name: sha(log), **case["artifact_sha256"]})
                save(folder/"record.json", record)
                print(f"{stem}: solved={case['solved']}, {case.get('solve_seconds', 0.):.3f} s", flush=True)
    record.update(completed=True, comparisons=comparisons(folder, record["cases"]),
                  sources_unchanged=all(sha(ROOT/name) == expected
                      for name, expected in record["source_sha256"].items()),
                  required_cases_solved=all(c["solved"] for c in record["cases"]
                      if not c["settings"].get("historical_failure")),
                  finished_utc=datetime.now(timezone.utc).isoformat())
    save(folder/"record.json", record)
    if (not record["sources_unchanged"] or not record["required_cases_solved"]
            or any(not row.get("accuracy_passed", False) for row in record["comparisons"]
                   if not CASES[row["name"]].get("historical_failure"))):
        raise RuntimeError("a required backend comparison failed; retained full evidence")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--schur-output", choices=("real", "complex"), default="real")
    parser.add_argument("--repetition", type=int, default=0)
    args = parser.parse_args()
    if args.repeats < 1 or args.repetition < 0:
        parser.error("repeats must be positive and repetition nonnegative")
    if args.worker:
        worker(args.output.resolve(), args.worker, args.schur_output, args.repetition)
    else:
        run(args.output.resolve(), args.repeats)
