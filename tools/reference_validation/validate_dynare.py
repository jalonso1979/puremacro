"""Freeze live Dynare exports or verify puremacro against the frozen references.

Run from the repository root with PYTHONPATH=. See README.md for regeneration.
No reference tensor or simulation is calculated by puremacro during freezing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from puremacro.dsge import load_mod

FIXTURES = Path(__file__).resolve().parents[2] / "tests/fixtures/dynare_live"
AXES = {"ghx": "x", "ghu": "u", "ghxx": "xx", "ghxu": "xu", "ghuu": "uu", "ghs2": "",
        "ghxxx": "xxx", "ghxxu": "xxu", "ghxuu": "xuu", "ghuuu": "uuu", "ghxss": "x", "ghuss": "u"}
ATOL, RTOL, BURN = 1e-10, 1e-9, 500


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(export_dir, fixtures=FIXTURES):
    """Normalize only documented row ordering; unfolded tensors have no scaling."""
    complete = json.loads((Path(export_dir) / "export_complete.json").read_text())
    manifest = {"origin": "Live Dynare execution; not puremacro-generated",
                "run_id": complete["run_id"],
                "burn": BURN, "periods": 2500, "atol": ATOL, "rtol": RTOL, "cases": {}}
    for model in sorted(fixtures.glob("*.mod")):
        for order in (2, 3):
            name = f"{model.stem}_order{order}"
            source = Path(export_dir) / f"{name}.mat"
            ref = loadmat(source, simplify_cells=True)
            if int(ref["order"]) != order:
                raise ValueError(f"Wrong approximation order in {name}")
            if ref["run_id"] != complete["run_id"] or name not in complete["cases"]:
                raise ValueError(f"Stale or incomplete export {name}")
            if str(ref["model_source"]) != model.read_text():
                raise ValueError(f"Model differs from the source Dynare executed: {name}")
            variables = np.atleast_1d(ref["variable_names"]).astype(str)
            shocks = np.atleast_1d(ref["shock_names"]).astype(str)
            states = np.atleast_1d(ref["state_names"]).astype(str)
            n = len(variables)
            order_var = np.asarray(ref["dr"]["order_var"], dtype=int).reshape(-1) - 1
            if sorted(order_var) != list(range(n)):
                raise ValueError("Invalid Dynare order_var")
            rows = np.argsort(order_var)
            arrays = {"variable_names": variables, "state_names": states, "shock_names": shocks,
                      "ys": np.asarray(ref["dr"]["ys"]).reshape(n),
                      "shock_cov": np.asarray(ref["shock_cov"]).reshape(len(shocks), len(shocks)),
                      "path": np.asarray(ref["path"])[:, 1:].T,
                      "innovations": np.asarray(ref["innovations"]).reshape(-1, len(shocks))}
            original_eps = np.loadtxt(fixtures / f"{model.stem}_innovations.csv", delimiter=",").reshape(-1, len(shocks))
            np.testing.assert_allclose(arrays["innovations"], original_eps, rtol=1e-15, atol=0)
            for field in list(AXES)[:6 if order == 2 else 12]:
                arrays[field] = np.asarray(ref["dr"][field]).reshape(n, -1)[rows]
            if not all(np.isfinite(a).all() for a in arrays.values() if a.dtype.kind != "U"):
                raise ValueError(f"Nonfinite export {name}")
            if arrays["path"].shape != (2500, n):
                raise ValueError(f"Incomplete simulation {name}")
            target = fixtures / f"{name}.npz"
            np.savez_compressed(target, **arrays)
            manifest["cases"][name] = {
                "dynare_version": str(ref["dynare_version"]), "matlab_version": str(ref["matlab_version"]),
                "order": order, "model": model.name, "model_sha256": sha256(model),
                "innovations_sha256": sha256(fixtures / f"{model.stem}_innovations.csv"),
                "raw_mat_sha256": sha256(source), "reference_sha256": sha256(target),
            }
    (fixtures / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def sample_moments(path):
    x = path[BURN:]
    return {"sample_mean": x.mean(axis=0), "sample_covariance": np.atleast_2d(np.cov(x, rowvar=False)),
            "sample_lag1_covariance": (x[1:]-x[1:].mean(0)).T @ (x[:-1]-x[:-1].mean(0)) / (len(x)-2)}


def compare_case(name, fixtures=FIXTURES):
    manifest = json.loads((fixtures / "manifest.json").read_text())
    info = manifest["cases"][name]
    model = fixtures / info["model"]
    reference = fixtures / f"{name}.npz"
    innovations = fixtures / f"{model.stem}_innovations.csv"
    if (sha256(model) != info["model_sha256"] or sha256(reference) != info["reference_sha256"]
            or sha256(innovations) != info["innovations_sha256"]):
        raise ValueError(f"Fixture provenance mismatch: {name}")
    sol = load_mod(model.read_text(), order=info["order"])
    dr = sol.decision_rules()
    details = {}

    def check(field, got, expected):
        got, expected = np.asarray(got), np.asarray(expected)
        if got.shape != expected.shape:
            raise ValueError(f"Shape mismatch for {name}/{field}: {got.shape}, {expected.shape}")
        finite = np.isfinite(got).all() and np.isfinite(expected).all()
        details[field] = {"max_abs_error": float(np.max(np.abs(got-expected))),
                          "passed": bool(finite and np.allclose(got, expected, atol=ATOL, rtol=RTOL))}

    with np.load(reference, allow_pickle=False) as ref:
        variables, states, shocks = [list(ref[k]) for k in ("variable_names", "state_names", "shock_names")]
        rows = [list(dr.variable_names).index(v) for v in variables]
        check("ys", np.asarray(dr.ys)[rows], ref["ys"])
        si = [list(dr.shock_names).index(s) for s in shocks]
        check("shock_cov", np.asarray(sol.shock_cov)[np.ix_(si, si)], ref["shock_cov"])
        for field in list(AXES)[:6 if info["order"] == 2 else 12]:
            axes = AXES[field]
            cols = list(product(*(states if a == "x" else shocks for a in axes)))
            owncols = list(product(*(list(dr.state_variables) if a == "x" else list(dr.shock_names) for a in axes)))
            got = np.asarray(getattr(dr, field)).reshape(len(variables), -1)[rows]
            check(field, got[:, [owncols.index(c) for c in cols]], ref[field])
        eps = ref["innovations"][:, [shocks.index(s) for s in sol.shock_names]]
        sim = sol.simulate(periods=len(eps), burn=0, shocks=eps)
        path = pd.concat([sim.states, sim.controls], axis=1)[variables].to_numpy() + np.asarray(dr.ys.loc[variables])
        check("simulation_path", path, ref["path"])
        for field, expected in sample_moments(ref["path"]).items():
            check(field, sample_moments(path)[field], expected)
    return {"passed": all(d["passed"] for d in details.values()), "comparisons": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, metavar="DYNARE_EXPORT_DIR")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    results = {name: compare_case(name) for name in manifest["cases"]}
    report = {"passed": all(r["passed"] for r in results.values()), "cases": results}
    content = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(content)
    print(content)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
