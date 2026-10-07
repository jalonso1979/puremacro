"""tools/extract_trade_references.py -- bundle MATLAB reference solutions of the trade model.

The legacy MATLAB trade model (``calibrar.m``, ``ff_equi.m``, ``ff_eval.m``) is the
external reference for ``puremacro.trade``'s equilibrium solver. Its solutions on the
clean OECD 2020 table (``tools/reference_validation/trade_clean_table/``) are
``results_77c_11s_<scenario>_clean.mat`` files; this tool copies the arrays the parity
suites compare against verbatim (dtype included) into
``puremacro/trade/_datafiles/trade_reference_solutions_oecd2020.npz`` and writes
``REFERENCE_MANIFEST_OECD2020.json``. Nothing is recomputed with puremacro, so the
comparison stays an external check.

Usage (from the repository root)::

    python tools/extract_trade_references.py --mat-dir /path/with/results_77c_11s_*_clean.mat

Scenarios whose ``.mat`` file is absent are left out and listed in the manifest; re-run
the tool when more scenarios finish.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATAFILES = ROOT / "puremacro" / "trade" / "_datafiles"
OUT = DATAFILES / "trade_reference_solutions_oecd2020.npz"
MANIFEST = DATAFILES / "REFERENCE_MANIFEST_OECD2020.json"
TABLE_MANIFEST = DATAFILES / "MANIFEST_OECD2020.json"

SCENARIOS = ("base", "t10", "t10_25", "t10_54", "t10_75", "t10_125", "t10_145")
PER_SCENARIO = ("XN_sol", "c_sol", "pfd_sol", "xx_sol", "w_sol", "ytot_sol", "T_sol", "r_sol",
                "p_sol", "tau_a", "taufd_a", "ff", "ff_sol", "tauf", "tauf_fd", "Reciprocal_Tariff")
BASE_ONLY = ("afd", "alpha", "beta", "KT", "LT", "invforT", "tax", "tax_fd", "theta",
             "ytot", "TT", "TTfd", "dM0", "dMFD", "dX0", "dXFD")
# Left out to keep the wheel small (they are dense and incompressible, 13 MB together):
# the input-output coefficient tensor ``a`` and the calibration-reproduced table
# ``data_calibra``. Both are recomputed from the bundled table by
# ``puremacro.trade.calibrate_trade_model`` and checked there against the data.
OMITTED = ("a", "data_calibra", "data_model_vf", "data_tariff_vf", "Pfd_final", "qxX0_sol", "qxFD0_sol", "qxX0", "qxFD0")


def main(argv=None) -> int:
    import scipy.io as sio  # development tool only; the library never imports scipy.io

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mat-dir", required=True)
    p.add_argument("--suffix", default="_clean")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--manifest", default=str(MANIFEST))
    args = p.parse_args(argv)
    mat_dir = Path(args.mat_dir)
    payload: dict[str, np.ndarray] = {}
    present, absent, runs, unconverged = [], [], {}, {}
    for scen in SCENARIOS:
        f = mat_dir / f"results_77c_11s_{scen}{args.suffix}.mat"
        if not f.is_file():
            absent.append(scen)
            continue
        m = sio.loadmat(str(f))
        if not bool(np.asarray(m["converged"]).ravel()[0]):
            unconverged[scen] = {"source_file": f.name,
                                 "final_residual_l1": float(np.nansum(np.abs(np.asarray(m["ff_sol"])))),
                                 "newton_iterations": int(np.asarray(m["iter"]).ravel()[0])}
            continue
        names = PER_SCENARIO + (BASE_ONLY if scen == "base" else ())
        missing = [n for n in names if n not in m]
        if missing:
            raise SystemExit(f"{f.name} lacks {missing}")
        for n in names:
            payload[f"{scen}__{n}"] = np.asarray(m[n])
        runs[scen] = {
            "converged": bool(np.asarray(m["converged"]).ravel()[0]),
            "newton_iterations": int(np.asarray(m["iter"]).ravel()[0]),
            "final_residual_l1": float(np.abs(np.asarray(m["ff_sol"])).sum()),
            "source_file": f.name,
            "source_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
        }
        present.append(scen)
    if not present:
        raise SystemExit(f"no results_77c_11s_*{args.suffix}.mat in {mat_dir}")
    out = Path(args.out)
    np.savez_compressed(out, **payload)
    table = json.loads(TABLE_MANIFEST.read_text(encoding="utf-8")) if TABLE_MANIFEST.is_file() else {}
    manifest = {
        "file": out.name,
        "built": _dt.date.today().isoformat(),
        "builder": "tools/extract_trade_references.py",
        "scenarios_present": present,
        "scenarios_absent": absent,
        "scenarios_unconverged_not_bundled": unconverged,
        "arrays_per_scenario": list(PER_SCENARIO),
        "arrays_base_only": list(BASE_ONLY),
        "arrays_omitted": list(OMITTED),
        "arrays_omitted_reason": ("dense, incompressible arrays recomputed by calibrate_trade_model from the "
                                  "bundled table (a, data_calibra) or post-processing outputs; left out to keep "
                                  "the wheel small"),
        "naming": "<scenario>__<array>",
        "runs": runs,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "solved_on_table": table.get("file"),
        "table_sha256_array": table.get("sha256_array"),
        "provenance": (
            "Verbatim copies (dtype included) of the MATLAB results_77c_11s_<scenario>_clean.mat "
            "outputs of the author's legacy trade model (calibrar.m, ff_equi.m, ff_eval.m, unchanged) "
            "run in MATLAB R2026a on the clean OECD 2020 table by "
            "tools/reference_validation/trade_clean_table/run_scenarios_clean.m. These remain an "
            "EXTERNAL check on puremacro, not puremacro's own output."
        ),
        "license": "MIT, same as puremacro's own code (author: Jorge Alonso-Ortiz)",
        "see_also": "SOURCES.md in this directory",
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes): scenarios {present}; absent {absent}; unconverged (not bundled) {sorted(unconverged)}")
    for scen, r in runs.items():
        print(f"  {scen}: converged={r['converged']} iterations={r['newton_iterations']} "
              f"final L1 residual={r['final_residual_l1']:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
