"""Retain both observed-household studies and their actual numerical checks.

Run from the repository with PYTHONPATH=. or from an installed PureMacro wheel.
The package source hashes identify which implementation actually ran.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy

import puremacro
from puremacro.examples.household_trade_policy import run_study as static_study
from puremacro.examples.trade_adjustment_households import run_study as dynamic_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    package = Path(puremacro.__file__).resolve().parent
    sources = [
        "examples/household_trade_policy.py", "examples/trade_adjustment_households.py",
        "trade/compensation.py", "trade/distributional.py", "trade/household.py",
        "trade/adjustment.py", "trade/incidence_transition.py", "study/robust.py",
        "trade/_datafiles/enigh2022_household_policy.npz",
        "trade/_datafiles/enigh2022_household_policy.json",
    ]
    record = {
        "schema": 1,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "pandas": pd.__version__},
        "sources": {f"puremacro/{s}": hashlib.sha256((package/s).read_bytes()).hexdigest() for s in sources},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "passed": False,
        "scope": "observed household inputs; assumed prices/preferences/financing; no causal or national-GE claim",
    }
    try:
        static, dynamic = static_study(), dynamic_study()
        for label, study, tables in (
            ("static", static, ("baseline", "incidence", "aggregate")),
            ("dynamic", dynamic, ("aggregate", "periods", "refinement")),
        ):
            for table in tables:
                study[table].to_csv(args.output/f"{label}_{table}.csv", index=table == "baseline")
            for metric, comparison in study["comparisons"].items():
                for table in ("payoffs", "regrets", "summary", "dominance"):
                    getattr(comparison, table).to_csv(args.output/f"{label}_{metric}_{table}.csv")
            (args.output/f"{label}_manifest.json").write_text(json.dumps(study["metadata"], indent=2)+"\n")
        sourcing = {}
        for i, (state, result) in enumerate(dynamic["sourcing"].items()):
            prefix = f"sourcing_{i}"
            result.accounts.to_csv(args.output/f"{prefix}_accounts.csv")
            result.quantities.to_csv(args.output/f"{prefix}_quantities.csv")
            sourcing[state] = {"file_prefix": prefix, "diagnostics": result.diagnostics, "model": result.metadata}
        record["sourcing"] = sourcing
        checks = {
            "independent_ces_ev": {
                "maximum_scaled_gap": static["metadata"]["maximum_scaled_independent_ev_gap"],
                "tolerance": 1e-10,
            },
            "static_fiscal_balance": {
                "maximum_absolute_residual": float(static["aggregate"].fiscal_residual.abs().max()),
                "tolerance": 1e-3,
            },
            "dynamic_fiscal_balance": {
                "maximum_absolute_residual": float(dynamic["periods"].fiscal_residual.abs().max()),
                "tolerance": 1e-3,
            },
        }
        for check in checks.values():
            value = next(v for k, v in check.items() if k.startswith("maximum_"))
            check["passed"] = value <= check["tolerance"]
        checks["sourcing_horizon"] = {
            "comparisons": dynamic["refinement"].to_dict(orient="records"),
            "passed": bool(dynamic["refinement"].passed.all()),
        }
        record["checks"] = checks
        record["passed"] = all(c["passed"] for c in checks.values())
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["artifacts"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())
        }
        (args.output/"record.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({"passed": record["passed"], "checks": record["checks"]}, indent=2))
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
