"""Extend an existing tree study without replacing any failed comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from puremacro.models import calibrate_one_asset_hank, monetary_scenario_tree, solve_hank_scenario_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True, help="name:assets:income:horizon[:asset_max]")
    parser.add_argument("--compare", action="append", required=True, help="coarse:fine")
    args = parser.parse_args()
    old_payload = (args.previous / "record.json").read_bytes()
    record = json.loads(old_payload)
    # Verify the recorded numerical implementation before reusing prior data.
    for path in ("puremacro/models/hank_scenario_tree.py", "puremacro/models/one_asset_hank.py"):
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != record["sources"][path]:
            raise ValueError(f"numerical source changed: {path}")
    for name, digest in record["artifacts"].items():
        if hashlib.sha256((args.previous/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"prior artifact changed: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in record["artifacts"]:
        shutil.copyfile(args.previous/name, args.output/name)
    old_hash = hashlib.sha256(old_payload).hexdigest()
    (args.output/f"previous-record-{old_hash[:12]}.json").write_bytes(old_payload)
    record["previous_record_sha256"] = old_hash
    record["extension_environment"] = {"python":platform.python_version(), "numpy":np.__version__, "scipy":scipy.__version__}
    record["sources"]["tools/reference_validation/extend_hank_scenario_tree.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record["extension_complete"] = False

    def checkpoint():
        record["artifacts"] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(args.output.iterdir()) if p.name != "record.json"}
        (args.output/"record.json").write_text(json.dumps(record,indent=2)+"\n")

    checkpoint()
    for spec in args.case:
        parts = spec.split(":")
        name, na, ne, T = parts[:4]
        a_max = float(parts[4]) if len(parts) == 5 else 150.
        if name in record["cases"] or not name.replace("_", "").isalnum():
            raise ValueError("new case names must be unique alphanumeric names")
        na, ne, T = int(na), int(ne), int(T)
        start = time.perf_counter()
        try:
            ss = calibrate_one_asset_hank(n_a=na, n_e=ne, a_max=a_max)
            result = solve_hank_scenario_tree(ss, monetary_scenario_tree(horizon=T, innovations=(-.005,.005)))
        except (ValueError, RuntimeError, ArithmeticError) as error:
            record["cases"][name] = {"asset_nodes":na, "income_states":ne, "horizon":T, "asset_max":a_max,
                "converged":False, "status":"failed", "error":f"{type(error).__name__}: {error}",
                "elapsed_seconds":time.perf_counter()-start}
            checkpoint()
            print(name, json.dumps(record["cases"][name]), flush=True)
            continue
        result.to_frame().to_csv(args.output/f"{name}_nodes.csv")
        result.expected_paths().to_csv(args.output/f"{name}_expectations.csv")
        np.savez_compressed(args.output/f"{name}_households.npz", a=ss.asset_grid, e=ss.e_grid,
                            saving=result.policy_a, consumption=result.policy_c,
                            hours=result.policy_n, distribution=result.distributions)
        record["cases"][name] = {"asset_nodes":na, "income_states":ne, "horizon":T, "asset_max":a_max,
            "elapsed_seconds":time.perf_counter()-start, "converged":result.converged,
            "calibration":ss.calibration, "diagnostics":result.diagnostics, "scope":result.metadata,
            "date0_expected_C":float(result.expected_paths().iloc[0].C)}
        checkpoint()
        print(name, json.dumps({"converged":result.converged,
                               "elapsed":record["cases"][name]["elapsed_seconds"]}), flush=True)
    latest = []
    for comparison in args.compare:
        coarse, fine = comparison.split(":")
        valid = record["cases"][coarse]["converged"] and record["cases"][fine]["converged"]
        if not all((args.output/f"{name}_nodes.csv").exists() for name in (coarse,fine)):
            latest.append({"coarse":coarse,"fine":fine,"passed":False,
                           "reason":"at least one case failed before producing paths"})
            continue
        left = pd.read_csv(args.output/f"{coarse}_nodes.csv", index_col=0)
        right = pd.read_csv(args.output/f"{fine}_nodes.csv", index_col=0)
        labels = left.index[left.period<6]
        for variable in ("C", "Y", "pi"):
            x, y = left.loc[labels,variable].to_numpy(), right.loc[labels,variable].to_numpy()
            tolerance = max(1e-6, .01*float(np.max(np.abs(y))))
            change = float(np.max(np.abs(x-y)))
            latest.append({"coarse":coarse, "fine":fine, "variable":variable, "window":6,
                           "maximum_change":change, "tolerance":tolerance, "solves_converged":bool(valid),
                           "passed":bool(valid and change<=tolerance)})
    record["refinements"].extend(latest)
    record["all_refinements_passed"] = all(r["passed"] for r in record["refinements"])
    record["latest_refinements"] = latest
    record["latest_refinements_passed"] = all(r["passed"] for r in latest)
    record["extension_complete"] = True
    checkpoint()
    print(json.dumps({"latest_refinements_passed":record["latest_refinements_passed"],"comparisons":latest},indent=2))


if __name__ == "__main__":
    main()
