"""Compare fixed and nonlinear conditional policies on the retained held-out history.

Use a new output directory. The training seed, held-out seed and the original
1e-5 audit threshold remain unchanged. Failed refinements are saved, not hidden.
Run with the four BLAS thread limits set to one and PYTHONPATH at this checkout.
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
from typing import Any

import numpy as np
import pandas as pd
import scipy

from puremacro.models import calibrate_one_asset_hank, solve_hank_stochastic_policy
from puremacro.models.hank_nonlinear_continuation import simulate_hank_nonlinear, solve_hank_nonlinear_continuation
from tools.reference_validation.benchmark_hank_equilibrium import blas_threading, linear_algebra_build

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("puremacro/models/hank_nonlinear_continuation.py", "puremacro/models/hank_recursive.py",
    "puremacro/models/_hank_stochastic_tree.py", "puremacro/models/_hank_one_asset_recursive.py",
    "puremacro/models/hank_equilibrium.py", "puremacro/models/one_asset_hank.py",
    "puremacro/models/_hank_tangent.py", "puremacro/dsge/stacked_newton.py",
    "tools/reference_validation/benchmark_hank_equilibrium.py",
    "tools/reference_validation/validate_hank_nonlinear_continuation.py")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(output: Path, control):
    if output.exists():
        raise ValueError("use a new output directory; failed evidence is preserved")
    output.mkdir(parents=True)
    record: dict[str, Any] = dict(schema_version=1, completed=False, started_utc=datetime.now(timezone.utc).isoformat(),
        scope="nonlinear receding-horizon equilibrium with local stochastic tails; no global error bound",
        python=platform.python_version(), platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
        pandas=pd.__version__, blas_control=control,
        numpy_linear_algebra=linear_algebra_build(np), scipy_linear_algebra=linear_algebra_build(scipy),
        source_sha256={name: _sha(ROOT/name) for name in SOURCES}, artifact_sha256={},
        source_reference_sha256=_sha(ROOT/"tests/fixtures/hank_recursive/record.json"),
        threads={name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        training_seed=1201, training_std=1e-7, training_periods=160,
        held_out_seed=9917, held_out_std=1e-4, held_out_periods=48, original_audit_tolerance=1e-5,
        numerical_tolerance=1e-8, horizon_tolerance=1e-6, branching_tolerance=1e-6,
        horizons=[4, 8, 16, 32, 64], branch_depths=[1, 2, 3, 4], quadrature_order=2,
        specification=dict(n_a=80, n_e=3, a_max=150.), cases=[])

    def save():
        temporary = output/"record.tmp"
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
        temporary.replace(output/"record.json")

    def frame(name, data):
        path = output/(name+".csv")
        data.to_csv(path, float_format="%.17g")
        record["artifact_sha256"][path.name] = _sha(path)

    for name in SOURCES:
        snapshot = output/"sources"/(name+".txt")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT/name).read_bytes())
    save()
    training = np.random.default_rng(1201).normal(0, 1e-7, 160)
    held_out = np.random.default_rng(9917).normal(0, 1e-4, 48)
    frame("training_innovations", pd.DataFrame(dict(innovation=training)))
    frame("held_out_innovations", pd.DataFrame(dict(innovation=held_out)))
    started = perf_counter()
    ss = calibrate_one_asset_hank(**record["specification"])
    full = solve_hank_stochastic_policy(ss, distribution_rank=None, rho=.7, shock_std=1e-4)
    reduced = solve_hank_stochastic_policy(ss, distribution_rank=16, rho=.7, shock_std=1e-4,
                                          snapshots=full.simulate(training).distributions)
    record.update(setup_seconds=perf_counter()-started, calibration=ss.calibration,
                  steady_state=ss.aggregates, steady_diagnostics=ss.diagnostics)
    for name, policy in (("full", full), ("snapshots16", reduced)):
        print(name, "fixed policy audit", flush=True)
        case: dict[str, Any] = dict(name=name, completed=False, rank=policy.basis.rank)
        record["cases"].append(case)
        start = perf_counter()
        baseline = policy.audit(held_out, tolerance=1e-5)
        case["fixed"] = dict(converged=baseline.converged, seconds=perf_counter()-start,
                             diagnostics=baseline.diagnostics)
        frame(name+"_fixed", baseline.paths)
        save()
        print(name, "nonanticipative nonlinear correction", flush=True)
        start = perf_counter()
        corrected = simulate_hank_nonlinear(policy, held_out, raise_on_failure=False,
            horizons=tuple(record["horizons"]), branch_depths=tuple(record["branch_depths"]),
            horizon_tolerance=record["horizon_tolerance"], branching_tolerance=record["branching_tolerance"],
            tolerance=record["numerical_tolerance"])
        case["nonlinear"] = dict(converged=corrected.converged, seconds=perf_counter()-start,
                                 diagnostics=corrected.diagnostics)
        frame(name+"_nonlinear", corrected.paths)
        frame(name+"_prices", corrected.prices)
        path = output/(name+"_nonlinear.npz")
        np.savez_compressed(path, states=corrected.states, distributions=corrected.distributions,
            expected_marginals=corrected.expected_marginals, a=corrected.policies["a"],
            c=corrected.policies["c"], n=corrected.policies["n"])
        record["artifact_sha256"][path.name] = _sha(path)
        case["completed"] = True
        save()
        print(name, corrected.converged, corrected.diagnostics["completed_periods"],
              corrected.diagnostics["market_residual_max_abs"], case["nonlinear"]["seconds"], flush=True)
    print("matched conditional comparisons", flush=True)
    aggregate = np.array([0., -1e-4])
    comparisons = {}

    def solve(name, policy, **options):
        started = perf_counter()
        result = solve_hank_nonlinear_continuation(policy, aggregate_state=aggregate, **options)
        comparisons[name] = dict(converged=result.converged, seconds=perf_counter()-started,
                                diagnostics=result.diagnostics)
        frame(name, result.tree.to_dataframe())
        return result

    def difference(left, right):
        a, b = left.expected_marginals[0], right.expected_marginals[0]
        return dict(root_log_controls_max_abs=float(np.max(abs(left.aggregate_coordinates[0]-right.aggregate_coordinates[0]))),
            root_expected_marginal_relative_max_abs=float(np.max(abs(a-b)/np.maximum(abs(a), abs(b)))),
            root_next_distribution_l1=float(np.sum(abs(left.post_distributions[0]-right.post_distributions[0]))))

    matched_full = solve("matched_full", full, horizon=8, branch_depth=2)
    matched_reduced = solve("matched_snapshots16", reduced, horizon=8, branch_depth=2)
    comparisons["compression_difference"] = difference(matched_full, matched_reduced)
    coarse_ss = calibrate_one_asset_hank(n_a=40, n_e=2, a_max=150.)
    coarse = solve_hank_stochastic_policy(coarse_ss, distribution_rank=None, rho=.7, shock_std=1e-4)
    coarse_short = solve("coarse_horizon8", coarse, horizon=8, branch_depth=2)
    coarse_long = solve("coarse_horizon32", coarse, horizon=32, branch_depth=2)
    comparisons["horizon_difference"] = difference(coarse_short, coarse_long)
    comparisons["household_grid_sensitivity"] = dict(
        scope="Euler/KKT and recalibrated grid sensitivity; not an exact household-grid error bound",
        coarse_specification=dict(n_a=40, n_e=2, a_max=150.), fine_specification=record["specification"],
        coarse_euler_kkt=coarse_short.diagnostics["euler_kkt_max"],
        fine_euler_kkt=matched_full.diagnostics["euler_kkt_max"],
        root_log_controls_max_abs=float(np.max(abs(coarse_short.aggregate_coordinates[0]-matched_full.aggregate_coordinates[0]))))
    quadrature3 = solve("coarse_quadrature3", coarse, horizon=8, branch_depth=2, quadrature_order=3)
    quadrature5 = solve("coarse_quadrature5", coarse, horizon=8, branch_depth=2, quadrature_order=5)
    comparisons["quadrature_2_to_3"] = difference(coarse_short, quadrature3)
    comparisons["quadrature_3_to_5"] = difference(quadrature3, quadrature5)
    fast = solve("coarse_krylov", coarse, horizon=4, branch_depth=1)
    dense = solve("coarse_dense", coarse, horizon=4, branch_depth=1, method="hybr")
    comparisons["dense_difference"] = difference(fast, dense)
    record["focused_comparisons"] = comparisons
    record.update(completed=True, finished_utc=datetime.now(timezone.utc).isoformat())
    save()
    return record


def run(output: Path):
    with blas_threading() as control:
        return _run(output, control)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
