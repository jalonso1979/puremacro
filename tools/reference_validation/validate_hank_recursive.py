"""Freeze compression, stochastic-terminal and two-asset equilibrium evidence.

Run with PYTHONPATH=. and OMP/OPENBLAS/MKL/VECLIB thread limits set to 1.
Use a fresh --output directory. Failed audits and insufficient ranks remain
in the record. Timing observations are not a hardware-independent speed gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
import pandas as pd
import scipy

from puremacro.models import (calibrate_one_asset_hank, calibrate_two_asset_equilibrium,
    solve_hank_stochastic_policy, solve_hank_stochastic_tree, solve_hank_equilibrium,
    monetary_scenario_tree)
from tools.reference_validation.benchmark_hank_equilibrium import blas_threading, linear_algebra_build

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ["puremacro/models/hank_recursive.py", "puremacro/models/_hank_one_asset_recursive.py",
    "puremacro/models/hank_two_asset_equilibrium.py", "puremacro/models/_hank_stochastic_tree.py",
    "puremacro/models/hank_equilibrium.py", "puremacro/models/_hank_tangent.py",
    "puremacro/models/hank_scenario_tree.py", "puremacro/models/one_asset_hank.py",
    "puremacro/models/hank_sequence_space.py", "puremacro/dsge/stacked_newton.py",
    "tools/reference_validation/benchmark_hank_equilibrium.py",
    "tools/reference_validation/validate_hank_recursive.py"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(record, out):
    temporary = out/"record.tmp"
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    temporary.replace(out/"record.json")


def summarize(result):
    return dict(converged=result.converged, diagnostics=result.diagnostics)


def chain(horizon, magnitude=-.001):
    return pd.DataFrame(dict(parent=[None]+[f"n{i}" for i in range(horizon-1)],
        conditional_probability=np.ones(horizon), monetary_shock=magnitude*.7**np.arange(horizon)),
        index=[f"n{i}" for i in range(horizon)])


def run(out):
    if out.exists():
        raise ValueError("use a new output directory to preserve earlier evidence")
    out.mkdir(parents=True)
    with blas_threading() as control:
        record = dict(schema_version=1, completed=False, started_utc=datetime.now(timezone.utc).isoformat(),
            scope="local first-order stochastic policies; nonlinear full-distribution audits and finite-tree corrections",
            python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
            scipy=scipy.__version__, pandas=pd.__version__, blas_control=control,
            blas_threads={name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
            numpy_linear_algebra=linear_algebra_build(np), scipy_linear_algebra=linear_algebra_build(scipy),
            source_sha256={name: sha(ROOT/name) for name in SOURCES}, artifact_sha256={},
            training_seed=1201, held_out_seed=9917, training_length=160, held_out_length=48,
            training_std=1e-7, held_out_std=1e-4, rho=.7, shock_std=1e-4,
            audit_tolerance=1e-5, cases=[], grid_sensitivity=[])
        save(record, out)

        def frame(name, value):
            path = out/(name+".csv")
            value.to_csv(path, float_format="%.17g")
            record["artifact_sha256"][path.name] = sha(path)

        def matrices(name, policy):
            path = out/(name+".npz")
            np.savez_compressed(path, transition=policy.transition, feedback=policy.feedback,
                basis=policy.basis.vectors, constraints=policy.basis.constraints,
                center=policy.basis.center, F0=policy._F0, F1=policy._F1)
            record["artifact_sha256"][path.name] = sha(path)

        training = np.random.default_rng(record["training_seed"]).normal(0, record["training_std"], record["training_length"])
        held_out = np.random.default_rng(record["held_out_seed"]).normal(0, record["held_out_std"], record["held_out_length"])
        frame("training_innovations", pd.DataFrame(dict(innovation=training)))
        frame("held_out_innovations", pd.DataFrame(dict(innovation=held_out)))
        specs = dict(one_asset=dict(n_a=80, n_e=3, a_max=150.),
                     two_asset=dict(n_a=12, n_b=12, a_max=80., b_max=10.))
        for name, spec in specs.items():
            print(name, "calibration", flush=True)
            case = dict(name=name, specification=spec, policies=[], terminal=[], completed=False)
            record["cases"].append(case)
            start = perf_counter()
            ss = calibrate_one_asset_hank(**spec) if name == "one_asset" else calibrate_two_asset_equilibrium(**spec)
            case.update(calibration_seconds=perf_counter()-start, calibration=ss.calibration,
                        steady_state=ss.aggregates, steady_diagnostics=ss.diagnostics)
            start = perf_counter()
            full = solve_hank_stochastic_policy(ss, distribution_rank=None)
            case["full_policy_seconds"] = perf_counter()-start
            matrices(name+"_full", full)
            full_linear = full.simulate(held_out)
            training_states = full.simulate(training).distributions
            frame(name+"_full_linear", full_linear.paths)
            print(name, "full-distribution audit", flush=True)
            audited = full.audit(held_out, tolerance=record["audit_tolerance"])
            case["full"] = dict(policy=full.diagnostics, audit=summarize(audited))
            frame(name+"_full_audit", audited.paths)
            smaller = full.audit(.25*held_out, tolerance=record["audit_tolerance"])
            case["full"]["quarter_shock_audit"] = summarize(smaller)
            frame(name+"_full_quarter_shock_audit", smaller.paths)
            save(record, out)
            selected = None
            for method in ("krylov", "snapshots"):
                for rank in (4, 8, 16):
                    print(name, method, rank, flush=True)
                    item = dict(method=method, rank=rank, completed=False)
                    case["policies"].append(item)
                    try:
                        start = perf_counter()
                        policy = solve_hank_stochastic_policy(ss, distribution_rank=rank,
                            snapshots=training_states if method == "snapshots" else None)
                        item["solve_seconds"] = perf_counter()-start
                        linear = policy.simulate(held_out)
                        audit = policy.audit(held_out, tolerance=record["audit_tolerance"])
                        item.update(policy=policy.diagnostics, basis=policy.basis.diagnostics,
                            audit=summarize(audit), linear_max_gap=float(np.max(abs(linear.paths.to_numpy()-full_linear.paths.to_numpy()))),
                            completed=True)
                        prefix = f"{name}_{method}_{rank}"
                        frame(prefix+"_linear", linear.paths)
                        frame(prefix+"_audit", audit.paths)
                        matrices(prefix, policy)
                        if method == "snapshots" and rank == 16:
                            smaller = policy.audit(.25*held_out, tolerance=record["audit_tolerance"])
                            item["quarter_shock_audit"] = summarize(smaller)
                            frame(prefix+"_quarter_shock_audit", smaller.paths)
                            selected = policy
                    except (ValueError, RuntimeError) as exc:
                        item["error"] = str(exc)
                    save(record, out)
            # A failed selected basis is retained, never replaced silently.
            if selected is not None:
                for horizon in (4, 8, 16, 32):
                    print(name, "terminal", horizon, flush=True)
                    start = perf_counter()
                    result = solve_hank_stochastic_tree(selected, chain(horizon))
                    entry = dict(horizon=horizon, seconds=perf_counter()-start, **summarize(result))
                    case["terminal"].append(entry)
                    frame(f"{name}_terminal_{horizon}", result.to_dataframe())
                    save(record, out)
                tree = monetary_scenario_tree(horizon=8, persistence=.7, innovations=(-.001, .001), probabilities=(.3, .7))
                fast, dense = solve_hank_stochastic_tree(selected, tree), solve_hank_stochastic_tree(selected, tree, method="hybr")
                case["dense_comparison"] = dict(krylov=summarize(fast), dense=summarize(dense),
                    path_max_gap=float(np.max(abs(fast.paths.to_numpy()-dense.paths.to_numpy()))))
                frame(name+"_branch_krylov", fast.to_dataframe())
                frame(name+"_branch_dense", dense.to_dataframe())
                if name == "one_asset":
                    old = solve_hank_equilibrium(ss, chain(4))
                    case["steady_terminal_reference"] = summarize(old)
                    frame(name+"_steady_terminal_4", old.to_frame())
            case["completed"] = True
            save(record, out)
        for grid in (10, 15, 20):
            print("two-asset grid sensitivity", grid, flush=True)
            item = dict(n_a=grid, n_b=grid, a_max=80., b_max=10., completed=False)
            record["grid_sensitivity"].append(item)
            try:
                ss = calibrate_two_asset_equilibrium(n_a=grid, n_b=grid)
                policy = solve_hank_stochastic_policy(ss, distribution_rank=16)
                audit = policy.audit(held_out[:12], tolerance=record["audit_tolerance"])
                item.update(steady_state=ss.aggregates, diagnostics=ss.diagnostics,
                            audit=summarize(audit), completed=True)
                frame(f"two_asset_grid_{grid}_audit", audit.paths)
            except (ValueError, RuntimeError) as exc:
                item["error"] = str(exc)
            save(record, out)
        record.update(completed=True, finished_utc=datetime.now(timezone.utc).isoformat())
        save(record, out)
        return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
