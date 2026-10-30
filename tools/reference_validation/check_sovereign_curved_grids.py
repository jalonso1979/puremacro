"""Crossed nonuniform-grid study of the high-reentry sovereign brief scenario.

This is a separate follow-up to failed uniform-grid refinements. The original
support, calibration, simulation design and moment tolerances stay fixed. The
grid family and all four cases are written before computation. Every failed
comparison remains in the record; both final debt and income refinements must
pass. It is not an estimate of the distance to an infinitely fine solution.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np

from puremacro.examples.first_brief_sovereign import (
    BASELINE, DESIGN, TOLERANCES, _package_version, _versions, _write_atomic,
    measure_case, paired_change, source_hashes,
)
from puremacro.sovereign import solve_sovereign_default

OVERRIDES = {"theta": .5}
GRIDS = [(6401, 641), (12801, 641), (6401, 1281), (12801, 1281)]
GRID_FAMILY = dict(negative_interval_share=[5, 16], positive_power=2,
                   b_min=BASELINE["b_min"], b_max=BASELINE["b_max"])
INITIAL = Path(__file__).resolve().parents[2] / "tests/fixtures/sovereign_brief/sensitive_grid_initial.json"


def debt_grid(n_b):
    """Nested linear saving nodes and quadratic borrowing nodes, exact zero."""
    if n_b < 17 or (n_b - 1) % 16:
        raise ValueError("curved grid needs n_b = 16*k + 1 for positive integer k")
    negative = (n_b - 1) * 5 // 16
    positive = n_b - 1 - negative
    return np.r_[np.linspace(BASELINE["b_min"], 0., negative + 1),
                 BASELINE["b_max"] * (np.arange(1, positive + 1) / positive) ** 2]


def grid_hash(grid):
    return hashlib.sha256(np.asarray(grid, dtype="<f8").tobytes()).hexdigest()


def study_sources():
    return {**source_hashes(),
            "tools/reference_validation/check_sovereign_curved_grids.py":
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def protocol():
    return dict(schema=1, baseline=BASELINE, overrides=OVERRIDES, design=DESIGN,
                tolerances=TOLERANCES, grid_family=GRID_FAMILY, grids=[list(g) for g in GRIDS],
                source_sha256=study_sources(), versions=_versions(), package_version=_package_version(),
                initial_uniform_record_sha256=hashlib.sha256(INITIAL.read_bytes()).hexdigest(),
                rule="abs(paired mean change) + 95% t half-width <= tolerance")


def validate_case(case, b, y, sources):
    grid = debt_grid(b)
    if ((case["n_b"], case["n_y"]) != (b, y)
            or case["source_sha256"] != sources
            or case["debt_grid_sha256"] != grid_hash(grid)
            or not np.array_equal(np.asarray(case["debt_grid"]), grid)):
        raise ValueError("case grid or numerical provenance differs from its protocol")
    expected = {name: parameter.default
                for name, parameter in inspect.signature(solve_sovereign_default).parameters.items()
                if name != "debt_grid"}
    expected.update(BASELINE, **OVERRIDES, n_b=b, n_y=y)
    if case["parameters"] != expected:
        raise ValueError("case parameters differ from the registered calibration")
    if case["simulation"]["seeds"] != DESIGN["seeds"]:
        raise ValueError("case seeds differ from the registered simulation design")
    for moment in TOLERANCES:
        values = np.asarray(case["simulation"]["per_seed"][moment], dtype=float)
        if values.shape != (len(DESIGN["seeds"]),) or not np.all(np.isfinite(values)):
            raise ValueError("one finite moment is required for every registered seed")
    solver = case["solver"]
    finite = all(np.isfinite(solver[key]) for key in
                 ("residual", "monotone_certification_gap", "exhaustive_bellman_residual"))
    return bool(finite and solver["converged"] and 0. <= solver["residual"] <= BASELINE["tol"]
                and 0. <= solver["monotone_certification_gap"] <= 1e-11
                and 0. <= solver["exhaustive_bellman_residual"] <= 1e-8)


def assess(record):
    """Recompute the prespecified final comparisons from all four cases."""
    expected = protocol()
    # Stored environments can differ from a reader's; the sources and design
    # must agree. combine() additionally requires the workers' environments to match.
    for key in expected.keys() - {"versions", "package_version"}:
        if record.get(key) != expected[key]:
            raise ValueError(f"record differs from the registered {key}")
    cases = record["cases"]
    if set(cases) != {f"{b}x{y}" for b, y in GRIDS}:
        raise ValueError("the complete crossed curved-grid ladder is required")
    numerical = [validate_case(cases[f"{b}x{y}"], b, y, record["source_sha256"]) for b, y in GRIDS]
    comparisons = {}
    for name, coarse, fine in (
        ("assets_at_641_income_states", "6401x641", "12801x641"),
        ("income_at_6401_debt_nodes", "6401x641", "6401x1281"),
        ("final_asset_refinement", "6401x1281", "12801x1281"),
        ("final_income_refinement", "12801x641", "12801x1281"),
    ):
        comparisons[name] = dict(coarse=coarse, fine=fine, **paired_change(cases[coarse], cases[fine]))
    record["comparisons"] = comparisons
    record["passed"] = bool(all(numerical) and comparisons["final_asset_refinement"]["passed"]
                             and comparisons["final_income_refinement"]["passed"])
    record["status"] = "passed" if record["passed"] else "convergence_not_established"
    return record


def run(output, debt_nodes=None):
    output.mkdir(parents=True, exist_ok=True)
    plan = protocol()
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("existing study protocol differs; use a new output directory")
    _write_atomic(path, plan)
    partial = output / "curved_grid_cases.json"
    record = {**copy.deepcopy(plan), "cases": {}}
    if partial.exists():
        prior = json.loads(partial.read_text(encoding="utf-8"))
        if any(prior.get(k) != v for k, v in plan.items()):
            raise ValueError("cached cases differ from the protocol or environment")
        record["cases"] = prior["cases"]
    for b, y in GRIDS:
        if debt_nodes is not None and b != debt_nodes:
            continue
        key = f"{b}x{y}"
        if key in record["cases"]:
            validate_case(record["cases"][key], b, y, plan["source_sha256"])
            print(f"{key}: reused", flush=True)
            continue
        grid = debt_grid(b)
        start = time.perf_counter()
        result = solve_sovereign_default(**{**BASELINE, **OVERRIDES}, debt_grid=grid, n_y=y)
        elapsed = time.perf_counter() - start
        case = measure_case(result, elapsed, **DESIGN)
        case.update(debt_grid=grid.tolist(), debt_grid_sha256=grid_hash(grid),
                    source_sha256=dict(plan["source_sha256"]))
        record["cases"][key] = case
        _write_atomic(partial, record)
        print(json.dumps(dict(grid=key, solver=case["solver"],
                              means={m: v["mean"] for m, v in case["simulation"]["summary"].items()})), flush=True)
        del result
    if debt_nodes is None:
        record = assess(record)
        _write_atomic(output / "curved_grid_convergence.json", record)
    return record


def combine(paths):
    if not paths:
        raise ValueError("completed case records are required")
    records = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    result = {**copy.deepcopy(records[0]), "cases": {}}
    for record in records:
        if any(record.get(k) != result[k] for k in protocol()):
            raise ValueError("cannot combine different protocols or environments")
        for key, case in record["cases"].items():
            if key in result["cases"]:
                raise ValueError(f"duplicate grid case: {key}")
            result["cases"][key] = copy.deepcopy(case)
    result["input_record_sha256"] = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    return assess(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debt-nodes", type=int, choices=[6401, 12801])
    parser.add_argument("--combine", type=Path, nargs="+")
    args = parser.parse_args()
    if args.combine and args.debt_nodes:
        parser.error("--combine and --debt-nodes are mutually exclusive")
    if args.combine:
        record = combine(args.combine)
        args.output.mkdir(parents=True, exist_ok=True)
        _write_atomic(args.output / "curved_grid_convergence.json", record)
    else:
        record = run(args.output, args.debt_nodes)
    if "passed" in record:
        print(json.dumps(dict(status=record["status"], passed=record["passed"])), flush=True)
        raise SystemExit(0 if record["passed"] else 1)
