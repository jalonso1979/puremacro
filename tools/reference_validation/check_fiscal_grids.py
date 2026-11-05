"""Check grid convergence of the illustrative fiscal sovereign-default calibration.

There is no independent fiscal reference: the ladder solves the public
``solve_fiscal_sovereign_default`` with the certified monotone search. Each
solve is checked against a blocked exhaustive search at its own prices, and
``tau=1`` without private consumption reduces bit-for-bit to the
Arellano-validated ``solve_sovereign_default``. Numerical grids change;
calibration, support, timing and measurement stay fixed. Tolerances and the
design are written to a protocol file before any solve and are never loosened
after the fact.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from puremacro.sovereign import FiscalSovereignResult, solve_fiscal_sovereign_default
from puremacro.sovereign import _fiscal_paths as paths_module
from puremacro.sovereign import _monotone, arellano, fiscal
from puremacro.sovereign._fiscal_paths import MOMENTS, draw_uniforms, ergodic_moments, simulate_paths
from tools.reference_validation.check_arellano_grids import paired_comparison

# Illustrative, not fitted to any target (research rule).
ILLUSTRATIVE = dict(tau=.2, psi=1., gamma_c=2., gamma_g=2.)
POLICY_SEARCH = "monotone"
# Percent / percentage-point units; numerical tolerances, not target bands.
TOLERANCES = {name: .02 if name.startswith("corr_") else .10
              for name in MOMENTS if name != "boundary_choice_share_pct"}
SOURCES = (fiscal, arellano, _monotone, paths_module)
GUIDE_COLUMNS = ("debt_pct", "spread_mean_pct", "annual_default_pct", "spread_std_pp", "spending_pct", "corr_pb_y")


def solve_case(n_b: int, n_y: int) -> FiscalSovereignResult:
    """Illustrative calibration, public defaults otherwise, certified monotone search."""
    return solve_fiscal_sovereign_default(**ILLUSTRATIVE, n_b=n_b, n_y=n_y, policy_search=POLICY_SEARCH)


def source_hashes() -> dict[str, str]:
    return {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in SOURCES}


def _calibration() -> dict:
    import inspect

    defaults = {name: p.default for name, p in inspect.signature(solve_fiscal_sovereign_default).parameters.items()
                if p.default is not inspect.Parameter.empty and name not in ("n_b", "n_y")}
    return defaults | ILLUSTRATIVE | dict(policy_search=POLICY_SEARCH)


def _array_fields() -> list[str]:
    # ``rule`` is not an array; the ladder solves the rule-free calibration only.
    return [f.name for f in fields(FiscalSovereignResult)
            if f.name not in ("metadata", "rule") and f.name not in _SCALARS]


_SCALARS = ("spread_mean", "spread_y_corr", "bellman_residual", "converged", "maturity_param_delta",
            "coupon_rate_kappa", "taste_shock_scale", "residual", "iterations", "tau", "psi", "gamma_c",
            "gamma_g", "private_consumption")
_META_ARRAYS = ("y_default_vec", "stationary_dist")


def _diagnostics(result: FiscalSovereignResult, seconds: float) -> dict:
    meta = result.metadata
    return dict(converged=bool(result.converged), iterations=int(result.iterations),
                bellman_residual=float(result.residual), policy_search=meta["policy_search"],
                monotone_certification_gap=meta["monotone_certification_gap"],
                exhaustive_bellman_residual=meta["exhaustive_bellman_residual"],
                solve_seconds=seconds)


def _save(result: FiscalSovereignResult, arrays_path: Path) -> None:
    if result.rule is not None:
        raise ValueError("The fiscal grid cache stores rule-free solutions only")
    arrays = {name: getattr(result, name) for name in _array_fields()}
    arrays |= {"meta_" + key: np.asarray(result.metadata[key]) for key in _META_ARRAYS}
    np.savez_compressed(arrays_path, **arrays)


def _load(arrays_path: Path, manifest: dict) -> FiscalSovereignResult:
    with np.load(arrays_path) as a:
        arrays = {key: a[key] for key in a.files}
    metadata = dict(manifest["metadata"])
    for key in _META_ARRAYS:
        metadata[key] = arrays.pop("meta_" + key)
    return FiscalSovereignResult(**arrays, **manifest["scalars"], metadata=metadata)


def _json_metadata(result: FiscalSovereignResult) -> dict:
    out = {}
    for key, value in result.metadata.items():
        if key in _META_ARRAYS:
            continue
        out[key] = value.item() if isinstance(value, np.generic) else value
    return out


def case(n_b: int, n_y: int, args, source_hash: str) -> dict:
    name = f"fiscal_{n_b}_{n_y}"
    calibration = _calibration()
    identity = dict(schema=1, source_sha256=source_hash, calibration=calibration,
                    n_b=n_b, n_y=n_y)
    solution_identity = hashlib.sha256(json.dumps(
        identity, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    manifest_path = args.output / (name + "_solution.json")
    arrays_path = args.output / (name + ".npz")
    result = diagnostics = None
    if manifest_path.exists() and arrays_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("solution_identity") == solution_identity
                and manifest["arrays_sha256"] == hashlib.sha256(arrays_path.read_bytes()).hexdigest()):
            result, diagnostics = _load(arrays_path, manifest), manifest["diagnostics"]
            print(name, "reused verified solution", flush=True)
    if result is None:
        print("Solving", name, flush=True)
        start = time.perf_counter()
        result = solve_case(n_b, n_y)
        diagnostics = _diagnostics(result, time.perf_counter() - start)
        _save(result, arrays_path)
        scalars = {key: (getattr(result, key).item() if isinstance(getattr(result, key), np.generic)
                         else getattr(result, key)) for key in _SCALARS}
        manifest_path.write_text(json.dumps(dict(
            solution_identity=solution_identity, calibration=calibration,
            source_sha256=source_hash, arrays_sha256=hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
            diagnostics=diagnostics, scalars=scalars, metadata=_json_metadata(result)), indent=2) + "\n", encoding="utf-8")
    print(name, "solved", diagnostics, flush=True)
    design = dict(seeds=list(range(1979, 1979 + args.replications)), periods=args.periods, burn=args.burn,
                  start="zero debt, income node n_y // 2", source_sha256=source_hash,
                  solution_identity=solution_identity,
                  arrays_sha256=hashlib.sha256(arrays_path.read_bytes()).hexdigest())
    record_path = args.output / (name + "_simulation.json")
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["design"] == design:
            print(name, "reused verified simulation", flush=True)
            return record
    start = time.perf_counter()
    paths = simulate_paths(result, int(result.metadata["zero_index"]), n_y // 2,
                           draw_uniforms(design["seeds"], args.periods))
    per_seed = ergodic_moments(paths, args.burn)
    del paths
    runs = [dict(seed=seed, moments=moments) for seed, moments in zip(design["seeds"], per_seed)]
    summary = {key: dict(mean=float(np.mean([r["moments"][key] for r in runs])),
                         sd=float(np.std([r["moments"][key] for r in runs], ddof=1))) for key in MOMENTS}
    diagnostics = dict(diagnostics, simulate_seconds=time.perf_counter() - start)
    record = dict(name=name, n_b=n_b, n_y=n_y, design=design, diagnostics=diagnostics,
                  simulation=dict(runs=runs, moments=summary))
    record_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(name, {key: round(value["mean"], 5) for key, value in summary.items()}, flush=True)
    return record


def write_report(report: dict, output: Path) -> None:
    (output / "grid_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    rows = ["# Illustrative fiscal sovereign default: grid convergence", "",
            "Illustrative calibration (tau = 0.20, psi = 1, gamma_c = gamma_g = 2), not fitted to targets.", "",
            "Pass requires |paired mean change| + its 95% Student-t half-width",
            "to fit every stated tolerance, for the final income and asset refinements.", "",
            "| Grid | " + " | ".join(GUIDE_COLUMNS) + " |", "|---|" + "---:|" * len(GUIDE_COLUMNS)]
    for name, record in report["cases"].items():
        m = record["simulation"]["moments"]
        rows.append("| " + name + " | " + " | ".join(f'{m[k]["mean"]:.4f}' for k in GUIDE_COLUMNS) + " |")
    rows += ["", f'Overall convergence check: **{report["status"]}**.', ""]
    with (output / "grid_changes.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["comparison", "moment", "paired_change", "ci95_low", "ci95_high",
                         "absolute_change_bound", "tolerance", "status"])
        for name, comparison in report["comparisons"].items():
            rows += [f"## {name}", "", "| Moment | Paired change | 95% interval | Bound | Tolerance | Status |",
                     "|---|---:|---|---:|---:|---|"]
            for key, value in comparison["moments"].items():
                writer.writerow([name, key, value["change"], *value["ci95"], value["absolute_change_bound"],
                                 value["tolerance"], value["status"]])
                rows.append(f'| {key} | {value["change"]:.4f} | [{value["ci95"][0]:.4f}, {value["ci95"][1]:.4f}] | '
                            f'{value["absolute_change_bound"]:.4f} | {value["tolerance"]:.2f} | {value["status"]} |')
            rows.append("")
    (output / "grid_report.md").write_text("\n".join(rows) + "\n")


def run(args) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    asset_grids, incomes = sorted(set(args.n_b)), sorted(set(args.n_y))
    if len(asset_grids) < 2 or len(incomes) < 2:
        raise ValueError("At least two asset and two income resolutions are required")
    hashes = source_hashes()
    source_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    report = dict(recorded_at=datetime.now(timezone.utc).isoformat(),
                  repository_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  source_sha256=source_hash, sources=hashes,
                  checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  python=platform.python_version(), numpy=np.__version__,
                  calibration=_calibration(), tolerances=TOLERANCES, cases={}, comparisons={},
                  status="incomplete")
    # Written before solving, so tolerances and design cannot be chosen after seeing results.
    (args.output / "grid_protocol.json").write_text(json.dumps(report | dict(
        n_b=asset_grids, n_y=incomes, replications=args.replications, periods=args.periods,
        burn=args.burn), indent=2) + "\n")
    for n_y in incomes:
        for n_b in asset_grids:
            record = case(n_b, n_y, args, source_hash)
            report["cases"][record["name"]] = record
            (args.output / "grid_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fine_b, coarse_b = asset_grids[-1], asset_grids[-2]
    fine_y, coarse_y = incomes[-1], incomes[-2]
    for name, left, right in (
        ("final_asset_refinement", f"fiscal_{coarse_b}_{fine_y}", f"fiscal_{fine_b}_{fine_y}"),
        ("final_income_refinement", f"fiscal_{fine_b}_{coarse_y}", f"fiscal_{fine_b}_{fine_y}"),
    ):
        report["comparisons"][name] = dict(coarse=left, fine=right, **paired_comparison(
            report["cases"][left]["simulation"], report["cases"][right]["simulation"], tolerances=TOLERANCES))
    report["passed"] = all(c["passed"] for c in report["comparisons"].values())
    report["status"] = "passed" if report["passed"] else "convergence_not_established"
    write_report(report, args.output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-b", type=int, nargs="+", default=[1601, 3201, 6401])
    parser.add_argument("--n-y", type=int, nargs="+", default=[321, 641])
    parser.add_argument("--replications", type=int, default=16)
    parser.add_argument("--periods", type=int, default=500000)
    parser.add_argument("--burn", type=int, default=1000)
    result = run(parser.parse_args())
    print(result["status"], flush=True)
    raise SystemExit(0 if result["passed"] else 1)
