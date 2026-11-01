"""Checkpointed rank-32 continuation on the unchanged held-out HANK history.

This is a separate compression refinement, not a replacement of rank-16 failed
evidence. The time budget is checked between corrections, not inside a solve.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import scipy

from puremacro.models import calibrate_one_asset_hank, solve_hank_stochastic_policy
from puremacro.models.hank_nonlinear_continuation import correct_hank_stochastic_state
from tools.reference_validation.benchmark_hank_equilibrium import blas_threading, linear_algebra_build
from tools.reference_validation.validate_hank_nonlinear_continuation import SOURCES

ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path, max_seconds: float = 600.):
    if output.exists() or not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("use a new output directory and a positive time budget")
    output.mkdir(parents=True)
    with blas_threading() as control:
        sources = (*SOURCES, "tools/reference_validation/validate_hank_nonlinear_rank32.py")
        record: dict[str, Any] = dict(schema_version=1, completed=False, converged=False,
            started_utc=datetime.now(timezone.utc).isoformat(),
            scope="checkpointed nonanticipative rank-32 nonlinear continuation; local terminal policy",
            python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
            platform=platform.platform(), blas_control=control, numpy_linear_algebra=linear_algebra_build(np),
            scipy_linear_algebra=linear_algebra_build(scipy),
            source_sha256={name: _sha(ROOT/name) for name in sources}, artifact_sha256={},
            training_seed=1201, training_std=1e-7, training_periods=160,
            held_out_seed=9917, held_out_std=1e-4, held_out_periods=48,
            distribution_rank=32, specification=dict(n_a=80, n_e=3, a_max=150.),
            original_audit_tolerance=1e-5, numerical_tolerance=1e-8,
            horizon_tolerance=1e-6, branching_tolerance=1e-6,
            horizons=[4, 8, 16, 32, 64], branch_depths=[1, 2, 3, 4], quadrature_order=2,
            max_seconds=max_seconds, time_budget_scope="soft limit checked before each correction",
            completed_periods=0, period_diagnostics=[], solve_seconds=0., checkpoint_seconds=0.)

        def save():
            temporary = output/"record.tmp"
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
            temporary.replace(output/"record.json")

        def frame(name, value):
            path = output/(name+".csv")
            value.to_csv(path, float_format="%.17g")
            record["artifact_sha256"][path.name] = _sha(path)

        for name in sources:
            snapshot = output/"sources"/(name+".txt")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((ROOT/name).read_bytes())
        save()
        training = np.random.default_rng(1201).normal(0., 1e-7, 160)
        held_out = np.random.default_rng(9917).normal(0., 1e-4, 48)
        frame("training_innovations", pd.DataFrame(dict(innovation=training)))
        frame("held_out_innovations", pd.DataFrame(dict(innovation=held_out)))
        setup = perf_counter()
        ss = calibrate_one_asset_hank(**record["specification"])
        full = solve_hank_stochastic_policy(ss, distribution_rank=None, rho=.7, shock_std=1e-4)
        policy = solve_hank_stochastic_policy(ss, distribution_rank=32,
            snapshots=full.simulate(training).distributions, rho=.7, shock_std=1e-4)
        record.update(setup_seconds=perf_counter()-setup, calibration=ss.calibration,
                      steady_state=ss.aggregates, policy_diagnostics=policy.diagnostics)
        fixed = policy.audit(held_out, tolerance=1e-5)
        record["fixed"] = dict(converged=fixed.converged, diagnostics=fixed.diagnostics)
        frame("fixed", fixed.paths)
        save()
        distribution, aggregate, quote = ss.distribution.copy(), np.zeros(2), 1.
        distributions, states, rows, prices, expected = [distribution.copy()], [], [], [], []
        consumption, saving, hours = [], [], []
        start = perf_counter()
        for period, innovation in enumerate(held_out):
            if perf_counter()-start > max_seconds:
                record["failure"] = dict(kind="time_budget_exhausted", next_period=period)
                break
            print("rank32 period", period, "start", flush=True)
            observed = aggregate.copy()
            observed[1] += innovation
            step = perf_counter()
            try:
                result = correct_hank_stochastic_state(policy, distribution, observed, lag_q=quote,
                    horizons=tuple(record["horizons"]), branch_depths=tuple(record["branch_depths"]),
                    horizon_tolerance=1e-6, branching_tolerance=1e-6, tolerance=1e-8, raise_on_failure=False)
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                record["failure"] = dict(kind="domain_or_solver_error", period=period, message=str(exc))
                break
            elapsed = perf_counter()-step
            record["solve_seconds"] += elapsed
            if not result.converged:
                record["failure"] = dict(kind="refinement_failure", period=period, diagnostics=result.diagnostics)
                break
            states.append(np.r_[policy.basis.project(distribution-policy.basis.center), observed])
            rows.append(result.tree.paths.iloc[0].to_dict())
            prices.append(result.prices.iloc[0].to_dict())
            expected.append(result.expected_marginals[0].copy())
            consumption.append(result.tree.policies["c"][0].copy())
            saving.append(result.tree.policies["a"][0].copy())
            hours.append(result.tree.policies["n"][0].copy())
            distribution, aggregate = result.post_distributions[0].copy(), result.next_aggregate_state.copy()
            quote = float(result.prices.iloc[0]["q"])
            distributions.append(distribution.copy())
            record["completed_periods"] = period+1
            record["period_diagnostics"].append(dict(seconds=elapsed, **result.diagnostics))
            checkpoint = perf_counter()
            frame("nonlinear", pd.DataFrame(rows))
            frame("prices", pd.DataFrame(prices))
            temporary = output/"checkpoint.tmp"
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, distributions=np.asarray(distributions), states=np.asarray(states),
                    expected_marginals=np.asarray(expected), a=np.asarray(saving), c=np.asarray(consumption),
                    n=np.asarray(hours), asset_grid=ss.asset_grid, e_grid=ss.e_grid, transition=ss.transition)
            temporary.replace(output/"nonlinear.npz")
            record["artifact_sha256"]["nonlinear.npz"] = _sha(output/"nonlinear.npz")
            record["checkpoint_seconds"] += perf_counter()-checkpoint
            save()
            print("rank32 period", period, "accepted", "horizon", result.diagnostics["horizon"],
                  "depth", result.diagnostics["branch_depth"], "seconds", elapsed, flush=True)
        record.update(completed=True, converged=record["completed_periods"] == 48,
                      total_rollout_seconds=perf_counter()-start,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        for name in ("market_residual_max_abs", "euler_kkt_max", "terminal_compression_l1"):
            record[name] = max((d[name] for d in record["period_diagnostics"]), default=None)
        save()
        return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-seconds", type=float, default=600.)
    args = parser.parse_args()
    run(args.output, args.max_seconds)
