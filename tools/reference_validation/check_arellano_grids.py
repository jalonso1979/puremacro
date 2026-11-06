"""Check economic grid convergence of a deterministic Arellano calibration.

Uses the independent solver checked against the author's Fortran (archive) and
the public API (public). Numerical grids change; calibration, support, timing
and measurement stay fixed. A small Bellman residual is necessary but does not
substitute for stable moments.

``--calibration public`` (default) is PureMacro's public baseline with canonical
timing on 400*2**r+1 uniform debt nodes. ``--calibration archive`` is the
archived Arellano (2008) calibration with the author's exclusion timing on the
author's 200-node grid refined by midpoint insertion (199*2**r+1 nodes), with
income refinements on the archived log-income support.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Mapping

import numpy as np
from scipy.stats import t as student_t

from tools.reference_validation import arellano_2008 as reference

# Percentage quantities are in percent/percentage-point units. These are
# numerical tolerances, not target-fitting tolerances or parameter intervals.
TOLERANCES = {name: .02 if name.startswith("corr_") else .10 for name in reference.MOMENTS}

# The monotone search makes 1,601- and 3,201-node grids affordable; every solve
# is still certified by the solver's independent exhaustive Bellman check.
POLICY_SEARCH = "monotone"

# name: timing, debt-node multiplier (n_b = base*2**refinement+1), theta, gross R
CALIBRATIONS = {
    "public": dict(timing="canonical", base=400, theta=.282, gross_r=1.017),
    "archive": dict(timing="archive", base=199, theta=reference.THETA, gross_r=reference.GROSS_R),
}


def calibration_metadata(calibration: str) -> dict:
    """Recorded description of the fixed calibration, support and timing."""
    spec = CALIBRATIONS[calibration]
    if calibration == "public":
        return dict(beta=.95282, gamma=2., theta=.282, r=.017, rho=.945, sigma=.025,
                    timing=spec["timing"], policy_search=POLICY_SEARCH,
                    mean_income=10., default_cap=9.69,
                    assets_min=-3.3, assets_max=1.5, income_support="three unconditional log standard deviations")
    assets = reference.author_grid()
    log_y, _ = reference.verified_inputs()
    return dict(beta=reference.BETA, gamma=2., theta=reference.THETA, r=reference.GROSS_R-1,
                rho=.945, sigma=.025, timing=spec["timing"], policy_search=POLICY_SEARCH,
                income_scale=10., default_cap=10*reference.CAP,
                assets_min=float(assets[0]), assets_max=float(assets[-1]),
                income_support=[float(log_y[0]), float(log_y[-1])],
                literals="Fortran default-real (float32) beta, theta, R and cap, as archived",
                grid="author 200-node grid refined by midpoint insertion; income refinements "
                     "use Tauchen probabilities on the archived log-income endpoints")


def n_assets(refinement: int, calibration: str = "public") -> int:
    return CALIBRATIONS[calibration]["base"]*2**refinement+1


def solve_case(refinement: int, n_y: int, calibration: str = "public") -> reference.Solution:
    """Certified monotone policy search; timing follows the calibration's Bellman equation."""
    return reference.solve(refinement=refinement, n_y=n_y, timing=CALIBRATIONS[calibration]["timing"],
                           calibration=calibration, policy_search=POLICY_SEARCH)


def paired_comparison(coarse: dict, fine: dict, tolerances: Mapping[str, float] = TOLERANCES) -> dict:
    """Paired-seed change of each moment in ``tolerances`` from ``coarse`` to ``fine``."""
    left = {run["seed"]: run["moments"] for run in coarse["runs"]}
    right = {run["seed"]: run["moments"] for run in fine["runs"]}
    if (len(left) != len(coarse["runs"]) or len(right) != len(fine["runs"])
            or len(left) < 2 or left.keys() != right.keys()):
        raise ValueError("Paired comparisons require the same distinct seeds and at least two replications")
    moments = {}
    for name, tolerance in tolerances.items():
        differences = np.array([right[seed][name]-left[seed][name] for seed in sorted(left)])
        if not np.all(np.isfinite(differences)):
            raise ValueError(f"Nonfinite moment: {name}")
        mean = float(differences.mean())
        half = float(student_t.ppf(.975, len(left)-1)*differences.std(ddof=1)/np.sqrt(len(left)))
        upper = abs(mean)+half
        status = ("passed" if upper <= tolerance else "grid_shift_exceeds_tolerance"
                  if abs(mean) > tolerance else "sampling_precision_insufficient")
        moments[name] = dict(change=mean, ci95=[mean-half, mean+half],
                             absolute_change_bound=upper, tolerance=tolerance, status=status)
    return dict(passed=all(m["status"] == "passed" for m in moments.values()), moments=moments)


def case(refinement: int, n_y: int, args, source_hash: str) -> dict:
    calibration = getattr(args, "calibration", "public")
    spec = CALIBRATIONS[calibration]
    n_b = n_assets(refinement, calibration)
    name = f"{calibration}_{n_b}_{n_y}"
    manifest_path = args.output / (name+"_solution.json")
    arrays_path = args.output / (name+".npz")
    solution = None
    if manifest_path.exists() and arrays_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["source_sha256"] == source_hash
                and manifest["arrays_sha256"] == hashlib.sha256(arrays_path.read_bytes()).hexdigest()):
            with np.load(arrays_path) as a:
                solution = reference.Solution(**{key: a[key] for key in a.files},
                            timing=spec["timing"], theta=spec["theta"], gross_r=spec["gross_r"],
                            diagnostics=manifest["diagnostics"])
    if solution is None:
        print("Solving", name, flush=True)
        solution = solve_case(refinement, n_y, calibration)
        np.savez_compressed(arrays_path, **{key: getattr(solution, key) for key in (
            "assets", "income", "transition", "default_income", "value", "price", "policy", "default")})
        manifest_path.write_text(json.dumps(dict(source_sha256=source_hash,
            arrays_sha256=hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
            diagnostics=solution.diagnostics), indent=2)+"\n")
    print(name, "solved", solution.diagnostics, flush=True)
    design = dict(seeds=list(range(1979, 1979+args.replications)), periods=args.periods,
                  burn=args.burn, windows=args.windows, source_sha256=source_hash,
                  arrays_sha256=hashlib.sha256(arrays_path.read_bytes()).hexdigest())
    record_path = args.output / (name+"_simulation.json")
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["design"] == design:
            print(name, "reused verified simulation", flush=True)
            return record
    simulation = reference.monte_carlo(solution, design["seeds"], args.periods, args.burn, args.windows)
    record = dict(name=name, n_assets=n_b, n_income=n_y, design=design,
                  diagnostics=solution.diagnostics, simulation=simulation)
    record_path.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print(name, {key: round(value["mean"], 5) for key, value in simulation["moments"].items()}, flush=True)
    return record


def write_report(report: dict, output: Path) -> None:
    (output / "grid_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    title = {"public": "Public Arellano baseline", "archive": "Archived Arellano (2008) calibration"}
    rows = [f'# {title[report.get("calibration_name", "public")]}: grid convergence', "",
            "Calibration is fixed; published moments are not acceptance targets.", "",
            "Pass requires |paired mean change| + its 95% Student-t half-width",
            "to fit every stated tolerance, for the final income and asset refinements.", "",
            "| Grid | Debt/output (%) | Mean spread (%) | Annual default (%) | Spread SD (pp) |",
            "|---|---:|---:|---:|---:|"]
    for name, record in report["cases"].items():
        m = record["simulation"]["moments"]
        rows.append("| "+name+" | "+" | ".join(f'{m[k]["mean"]:.4f}' for k in (
            "debt_pct", "spread_mean_pct", "annual_default_pct", "spread_std_pp"))+" |")
    rows += ["", f'Overall convergence check: **{report["status"]}**.', "",
             "Intervals measure numerical/simulation uncertainty at this calibration,",
             "not uncertainty about estimated parameters or policy effects.", ""]
    with (output / "grid_changes.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["comparison", "moment", "paired_change", "ci95_low", "ci95_high", "absolute_change_bound", "tolerance", "status"])
        for name, comparison in report["comparisons"].items():
            rows += [f"## {name}", "", "| Moment | Paired change | 95% interval | Bound | Tolerance | Status |",
                     "|---|---:|---|---:|---:|---|"]
            for key, value in comparison["moments"].items():
                writer.writerow([name, key, value["change"], *value["ci95"], value["absolute_change_bound"], value["tolerance"], value["status"]])
                rows.append(f'| {key} | {value["change"]:.4f} | [{value["ci95"][0]:.4f}, {value["ci95"][1]:.4f}] | '
                            f'{value["absolute_change_bound"]:.4f} | {value["tolerance"]:.2f} | {value["status"]} |')
            rows.append("")
    (output / "grid_report.md").write_text("\n".join(rows)+"\n")


def run(args) -> dict:
    calibration = getattr(args, "calibration", "public")
    if calibration not in CALIBRATIONS:
        raise ValueError("calibration must be public or archive")
    args.output.mkdir(parents=True, exist_ok=True)
    refinements, incomes = sorted(set(args.refinements)), sorted(set(args.income_states))
    if len(refinements) < 2 or len(incomes) < 2:
        raise ValueError("At least two asset and two income resolutions are required")
    source_hash = hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest()
    report = dict(recorded_at=datetime.now(timezone.utc).isoformat(),
                  repository_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  source_sha256=source_hash, checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  python=platform.python_version(), numpy=np.__version__,
                  calibration=calibration_metadata(calibration),
                  tolerances=TOLERANCES, cases={}, comparisons={}, status="incomplete")
    if calibration != "public":
        # Recorded only for non-default calibrations, so public records keep their schema.
        report["calibration_name"] = calibration
    # Written before solving, so the tolerance/design cannot be mistaken for a
    # rule chosen retrospectively to fit the final grid output.
    (args.output / "grid_protocol.json").write_text(json.dumps(report | dict(
        refinements=refinements, income_states=incomes, replications=args.replications,
        periods=args.periods, burn=args.burn, windows=args.windows), indent=2)+"\n")
    for n_y in incomes:
        for refinement in refinements:
            record = case(refinement, n_y, args, source_hash)
            report["cases"][record["name"]] = record
            (args.output / "grid_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    fine_b, coarse_b = (n_assets(r, calibration) for r in refinements[-2:][::-1])
    fine_y, coarse_y = incomes[-1], incomes[-2]
    for name, left, right in (
        ("final_asset_refinement", f"{calibration}_{coarse_b}_{fine_y}", f"{calibration}_{fine_b}_{fine_y}"),
        ("final_income_refinement", f"{calibration}_{fine_b}_{coarse_y}", f"{calibration}_{fine_b}_{fine_y}"),
    ):
        report["comparisons"][name] = dict(coarse=left, fine=right, **paired_comparison(
            report["cases"][left]["simulation"], report["cases"][right]["simulation"]))
    report["passed"] = fine_y >= 321 and fine_b >= 797 and all(c["passed"] for c in report["comparisons"].values())
    report["status"] = "passed" if report["passed"] else "convergence_not_established"
    write_report(report, args.output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", choices=sorted(CALIBRATIONS), default="public")
    parser.add_argument("--income-states", type=int, nargs="+", default=[321, 641])
    parser.add_argument("--refinements", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--replications", type=int, default=16)
    parser.add_argument("--periods", type=int, default=100000)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--windows", type=int, default=100)
    result = run(parser.parse_args())
    print(result["status"], flush=True)
    raise SystemExit(0 if result["passed"] else 1)
