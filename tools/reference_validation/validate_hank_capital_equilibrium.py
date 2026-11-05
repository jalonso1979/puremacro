"""Freeze productive-capital HANK equations, grid limits, and finite histories.

Use a fresh output directory. Failures are retained, not replaced by another
rank or grid. This is finite-equation and grid-sensitivity evidence, not an
adaptive-tail, global stochastic, or published-model replication certificate.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter
import traceback
from typing import Any

import numpy as np
import pandas as pd
import scipy

from puremacro.models.hank_capital import calibrate_capital_hank, _CapitalKernel
from puremacro.models.hank_reduction import solve_hank_reduced_policy
from puremacro.models.hank_nonlinear_continuation import solve_hank_nonlinear_continuation
from tools.reference_validation.benchmark_hank_equilibrium import blas_threading, linear_algebra_build

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "puremacro/models/hank_capital.py", "puremacro/models/hank_reduction.py",
    "puremacro/models/hank_nonlinear_continuation.py", "puremacro/models/hank_recursive.py",
    "puremacro/models/hank_two_asset_equilibrium.py", "puremacro/models/hank_sequence_space.py",
    "puremacro/models/_hank_stochastic_tree.py", "puremacro/models/hank_equilibrium.py",
    "puremacro/models/_hank_tangent.py", "puremacro/models/_hank_one_asset_recursive.py",
    "puremacro/models/one_asset_hank.py", "puremacro/models/hank_scenario_tree.py",
    "puremacro/dsge/stacked_newton.py",
    "tools/reference_validation/benchmark_hank_equilibrium.py",
    "tools/reference_validation/validate_hank_capital_equilibrium.py",
)
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
FIXED: dict[str, Any] = dict(a_max=200., b_max=30., gamma=2., r_a=.0125, r_b=.0075,
             alpha=.33, delta=.02, epsI=4., mu=1.015, government_share=.2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(out, record):
    temporary = out/"record.tmp"
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    temporary.replace(out/"record.json")


def run(out: Path, host_workload: str) -> None:
    if out.exists():
        raise ValueError("use a new output directory to preserve previous evidence")
    out.mkdir(parents=True)
    begun = perf_counter()
    with blas_threading() as control:
        record: dict[str, Any] = dict(schema_version=1, completed=False,
            started_utc=datetime.now(timezone.utc).isoformat(),
            scope="productive-capital stationary grid ladder, projected local policies, and finite conditional equilibrium histories",
            exclusions=["adaptive horizon acceptance", "adaptive branching acceptance",
                        "quadrature-order acceptance", "global aggregate-risk accuracy",
                        "full published-model replication", "household-grid accuracy certificate"],
            python=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
            numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
            blas_control=control, verified_blas_threading=control["status"] == "verified_single_thread",
            blas_threads={name: os.environ.get(name) for name in THREADS},
            numpy_linear_algebra=linear_algebra_build(np), scipy_linear_algebra=linear_algebra_build(scipy),
            timing_scope="wall times of one sequential benchmark process; no isolated speed claims or timing gates",
            concurrent_host_workload=host_workload,
            source_sha256={name: sha(ROOT/name) for name in SOURCES}, artifact_sha256={},
            fixed_parameters=FIXED, grids=[], rejected_ranks=[], joint_histories=[], medium_policy=None,
            rho=.7, shock_std=1e-4, history_innovations=[-1e-4, 8e-5, -3e-5],
            history_horizon=3, history_branch_depth=1, finite_tolerance=1e-9,
            audit_seed=99, audit_std=1e-5, audit_length=10)
        save(out, record)

        def frame(stem, data):
            path = out/(stem+".csv")
            data.to_csv(path, float_format="%.17g")
            record["artifact_sha256"][path.name] = sha(path)
            return path.name

        def arrays(stem, **values):
            path = out/(stem+".npz")
            np.savez_compressed(path, **values)
            record["artifact_sha256"][path.name] = sha(path)
            return path.name

        def policy_artifact(stem, policy):
            return arrays(stem, transition=policy.transition, feedback=policy.feedback,
                reduced_feedback=policy.reduced_feedback, F0=policy._F0, F1=policy._F1,
                full_residual=policy._full_residual,
                basis=policy.basis.vectors, constraints=policy.basis.constraints,
                center=policy.basis.center, marginal_basis=policy.marginal_basis.vectors,
                marginal_center=policy.marginal_basis.center)

        def corrected(stem, policy, distribution, state, quote, method):
            start = perf_counter()
            result = solve_hank_nonlinear_continuation(policy, distribution=distribution,
                aggregate_state=state, lag_q=quote, horizon=3, branch_depth=1,
                method=method, tolerance=record["finite_tolerance"], max_iterations=200)
            seconds = perf_counter()-start
            policies = result.tree.policies
            node_names = list(result.tree.nodes.index)
            parents = np.array([-1 if pd.isna(v) else node_names.index(v)
                                for v in result.tree.nodes.parent], dtype=int)
            arr = arrays(stem, distribution=result.tree.distributions,
                post_distribution=result.post_distributions,
                aggregate_state=result.aggregate_states,
                next_aggregate_state=result.next_aggregate_state,
                aggregate_coordinates=result.aggregate_coordinates,
                expected_marginals=result.expected_marginals, marginal_values=result.marginal_values,
                terminal_coordinates=result.tree.terminal_states,
                parents=parents, probability=result.tree.nodes.conditional_probability.to_numpy(),
                **policies)
            entry = dict(name=stem, method=method, seconds=seconds, converged=result.converged,
                initial_aggregate_state=np.asarray(state).tolist(), initial_q=float(quote),
                arrays=arr, prices=frame(stem+"_prices", result.prices),
                paths=frame(stem+"_paths", result.tree.paths),
                nodes=frame(stem+"_nodes", result.tree.nodes), diagnostics=result.diagnostics)
            return result, entry

        states = {}
        for n in (10, 15, 20, 30):
            print("capital grid", n, flush=True)
            entry: dict[str, Any] = dict(n_a=n, n_b=n, settings=dict(n_a=n, n_b=n, **FIXED), status="running")
            record["grids"].append(entry)
            save(out, record)
            start = perf_counter()
            try:
                ss = calibrate_capital_hank(n_a=n, n_b=n, **FIXED)
                states[n] = ss
                kernel = _CapitalKernel(ss)
                prices = kernel.prices_from_state(np.zeros(4), np.zeros(3), lag_q=1.)
                h = kernel.backward(kernel.m, prices)
                hh = ss._household
                entry.update(status="solved", calibration=ss.calibration,
                    aggregates=ss.aggregates, diagnostics=ss.diagnostics,
                    household_checks=kernel.household_checks(kernel.d, h, prices, kernel.m),
                    arrays=arrays(f"grid_{n}", distribution=kernel.d, a_grid=hh.a_grid,
                        b_grid=hh.b_grid, productivity=hh.s_grid, income_transition=hh.pi_s,
                        marginal_values=kernel.m, a=hh.a_ss, b=hh.b_ss, c=hh.c_ss, chi=hh.chi_ss))
            except (ValueError, RuntimeError) as exc:
                entry.update(status="failed", exception_type=type(exc).__name__, error=str(exc),
                             traceback=traceback.format_exc())
            entry["seconds"] = perf_counter()-start
            save(out, record)

        for n in (15, 20):
            if n not in states:
                continue
            ss = states[n]
            rejected: dict[str, Any] = dict(grid=n, distribution_rank=16, marginal_rank=64,
                            krylov_steps=96, status="running")
            record["rejected_ranks"].append(rejected)
            start = perf_counter()
            try:
                unexpected = solve_hank_reduced_policy(ss, distribution_rank=16,
                    marginal_rank=64, krylov_steps=96, shock_std=record["shock_std"])
                rejected.update(status="solved", diagnostics=unexpected.diagnostics)
            except (ValueError, RuntimeError) as exc:
                rejected.update(status="failed", exception_type=type(exc).__name__, error=str(exc))
            rejected["seconds"] = perf_counter()-start
            save(out, record)

            print("capital conditional history", n, flush=True)
            entry = dict(grid=n, distribution_rank=32, marginal_rank=128, krylov_steps=96,
                         completed=False, steps=[])
            record["joint_histories"].append(entry)
            start = perf_counter()
            try:
                policy = solve_hank_reduced_policy(ss, distribution_rank=32, marginal_rank=128,
                    krylov_steps=96, rho=record["rho"], shock_std=record["shock_std"])
                entry.update(policy_seconds=perf_counter()-start, policy=policy.diagnostics,
                    policy_arrays=policy_artifact(f"grid_{n}_policy", policy))
                zero, entry["zero_risk_zero_shock"] = corrected(f"grid_{n}_zero",
                    replace(policy, shock_std=0.), policy._kernel.d, np.zeros(3), 1., "hybr")
                distribution, state, quote = policy._kernel.d.copy(), np.zeros(3), 1.
                for date, innovation in enumerate(record["history_innovations"]):
                    state[1] += innovation
                    result, step = corrected(f"grid_{n}_history_{date}", policy,
                        distribution, state, quote, "krylov")
                    step.update(date=date, innovation=innovation)
                    entry["steps"].append(step)
                    if date == 0:
                        dense, comparison = corrected(f"grid_{n}_dense", policy,
                            distribution, state, quote, "hybr")
                        comparison.update(root_coordinate_max_gap=float(np.max(abs(
                            dense.aggregate_coordinates[0]-result.aggregate_coordinates[0]))),
                            next_distribution_l1_gap=float(np.sum(abs(
                                dense.post_distributions[0]-result.post_distributions[0]))))
                        entry["dense_comparison"] = comparison
                    save(out, record)
                    if not result.converged:
                        raise ValueError(f"finite conditional equations failed at history date {date}")
                    distribution = result.post_distributions[0]
                    state = result.next_aggregate_state.copy()
                    quote = float(result.prices.iloc[0]["q"])
                entry.update(completed=True, all_finite_equations_converged=bool(zero.converged
                    and entry["dense_comparison"]["converged"]))
            except (ValueError, RuntimeError) as exc:
                entry.update(error=str(exc), exception_type=type(exc).__name__,
                             traceback=traceback.format_exc())
            entry["total_seconds"] = perf_counter()-start
            save(out, record)

        if 30 in states:
            print("capital 30 by 30 policy and full-distribution audit", flush=True)
            entry = dict(grid=30, distribution_rank=48, marginal_rank=128, krylov_steps=128,
                         completed=False)
            record["medium_policy"] = entry
            start = perf_counter()
            try:
                policy = solve_hank_reduced_policy(states[30], distribution_rank=48,
                    marginal_rank=128, krylov_steps=128, rho=record["rho"],
                    shock_std=record["shock_std"])
                entry.update(policy_seconds=perf_counter()-start, policy=policy.diagnostics,
                    policy_arrays=policy_artifact("grid_30_policy", policy))
                innovations = np.random.default_rng(record["audit_seed"]).normal(
                    0., record["audit_std"], record["audit_length"])
                entry["innovations"] = frame("grid_30_innovations", pd.DataFrame(dict(innovation=innovations)))
                linear = policy.simulate(innovations)
                entry["linear"] = dict(converged=linear.converged, diagnostics=linear.diagnostics,
                    paths=frame("grid_30_linear", linear.paths),
                    arrays=arrays("grid_30_linear", states=linear.states, distributions=linear.distributions))
                before = perf_counter()
                audit = policy.audit(innovations, tolerance=1e-5)
                entry["audit"] = dict(converged=audit.converged, diagnostics=audit.diagnostics,
                    seconds=perf_counter()-before, paths=frame("grid_30_audit", audit.paths),
                    arrays=arrays("grid_30_audit", states=audit.states, distributions=audit.distributions))
                entry["completed"] = True
            except (ValueError, RuntimeError) as exc:
                entry.update(error=str(exc), exception_type=type(exc).__name__,
                             traceback=traceback.format_exc())
            entry["total_seconds"] = perf_counter()-start
            save(out, record)

        current = {name: sha(ROOT/name) for name in SOURCES}
        record["sources_unchanged_during_run"] = current == record["source_sha256"]
        if not record["sources_unchanged_during_run"]:
            record["source_sha256_at_end"] = current
            save(out, record)
            raise RuntimeError("numerical sources changed during the evidence run")
        readme = out/"README.md"
        readme.write_text(
            "# Productive-capital HANK evidence\n\n"
            "Generated by tools/reference_validation/validate_hank_capital_equilibrium.py.\n"
            "The JSON record binds numerical source files and every CSV/NPZ artifact by SHA-256.\n\n"
            "Fixed physical parameters and fixed asset bounds are used on grids 10, 15, 20, and 30.\n"
            "The rejected 10-point household solve and rank-16 determinacy failures are retained.\n"
            "The 15/20 histories solve finite horizon-3, branch-depth-1 conditional equilibria;\n"
            "they do not pass an adaptive horizon, branching, or quadrature acceptance gate.\n"
            "The 30-point audit transports the full distribution under the local policy;\n"
            "its market, full-marginal, compression and household Euler errors remain distinct.\n\n"
            "Wall times were observed with verified single-thread BLAS where the record says so.\n"
            f"Concurrent host workload: {host_workload}.\n"
            "These are not isolated performance measurements or timing requirements.\n\n"
            "The installed-capital production block has a separate official-code comparison.\n"
            "This economy is not a full replication of the published ABRS household/intermediary model.\n")
        record["artifact_sha256"][readme.name] = sha(readme)
        record.update(completed=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                      total_seconds=perf_counter()-begun)
        save(out, record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-workload", default="not reported")
    args = parser.parse_args()
    run(args.output, args.host_workload)
