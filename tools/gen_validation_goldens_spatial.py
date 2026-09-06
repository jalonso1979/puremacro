"""Regenerate puremacro/validation/goldens/spatial.json (dev-only; uses esda + libpysal).

The two PACKAGE references are esda's Moran and Geary statistics with their
Cliff-Ord normality and randomisation moments (``permutations=0``) on a
row-standardised 6 x 6 rook lattice. The lattice and the three fields are
imported from ``cases_spatial`` so the golden and the puremacro computation see
byte-identical inputs; the fields are re-ordered to libpysal's ``id_order``
before the call because ``W`` may not keep dictionary order.

Run from the package dir:  python tools/gen_validation_goldens_spatial.py
"""
from __future__ import annotations

import json
from pathlib import Path

import esda
import libpysal
import numpy as np

from puremacro.validation.cases_spatial import FIELD_ORDER, lattice_demo_data


def _live_esda_stats() -> dict:
    d = lattice_demo_data()
    w = libpysal.weights.W({u: list(v) for u, v in d["neighbours"].items()})
    w.transform = "r"
    ids = list(d["neighbours"].keys())
    order = [ids.index(i) for i in w.id_order]
    moran = {k: [] for k in ("I", "expected", "variance_norm", "variance_rand", "z_norm", "z_rand")}
    geary = {k: [] for k in ("C", "variance_norm", "variance_rand", "z_norm", "z_rand")}
    for name in FIELD_ORDER:
        x = np.asarray(d["fields"][name], dtype=float)[order]
        m = esda.Moran(x, w, permutations=0)
        g = esda.Geary(x, w, permutations=0)
        moran["I"].append(float(m.I))
        moran["expected"].append(float(m.EI))
        moran["variance_norm"].append(float(m.VI_norm))
        moran["variance_rand"].append(float(m.VI_rand))
        moran["z_norm"].append(float(m.z_norm))
        moran["z_rand"].append(float(m.z_rand))
        geary["C"].append(float(g.C))
        geary["variance_norm"].append(float(g.VC_norm))
        geary["variance_rand"].append(float(g.VC_rand))
        geary["z_norm"].append(float(g.z_norm))
        geary["z_rand"].append(float(g.z_rand))
    return {"morans_i_vs_esda": moran, "gearys_c_vs_esda": geary}


def main() -> None:
    stats = _live_esda_stats()
    out = {
        "_meta": {
            "generated_by": "tools/gen_validation_goldens_spatial.py",
            "reference": (
                f"esda {esda.__version__} / libpysal {libpysal.__version__} "
                "Moran(x, w, permutations=0) and Geary(x, w, permutations=0), w.transform='r'"
            ),
            "dgp": (
                "puremacro.validation.cases_spatial.lattice_demo_data "
                "(6 x 6 rook lattice; checker, gradient and sine-wave fields)"
            ),
            "regenerate": "python tools/gen_validation_goldens_spatial.py",
        },
        **stats,
    }
    path = Path(__file__).resolve().parents[1] / "puremacro" / "validation" / "goldens" / "spatial.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print(f"esda {esda.__version__}")


if __name__ == "__main__":
    main()
