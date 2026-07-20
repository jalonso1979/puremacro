"""Regenerate puremacro/validation/goldens/var.json (dev-only; uses statsmodels).

The reference for the Cholesky-IRF case is statsmodels' orthogonalised IRF,
which is an independent implementation of the same recursive identification.
Run from the package dir:  python tools/gen_validation_goldens_var.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.api import VAR

from puremacro.validation._fixtures import var_demo_data


def main() -> None:
    d = var_demo_data()
    Y, p, H = d["Y"], d["p"], d["horizon"]

    res = VAR(pd.DataFrame(Y)).fit(p)
    orth = np.asarray(res.irf(H).orth_irfs, dtype=float)  # (H+1, k, k)

    out = {
        "_meta": {
            "generated_by": "tools/gen_validation_goldens_var.py",
            "reference": f"statsmodels {statsmodels.__version__} VAR(df).fit(p).irf(H).orth_irfs",
            "dgp": "puremacro.validation._fixtures.var_demo_data (seed=20260531)",
            "regenerate": "python tools/gen_validation_goldens_var.py",
        },
        "cholesky_irf_vs_statsmodels": {"irf": orth.tolist()},
    }
    dest = (
        Path(__file__).resolve().parents[1]
        / "puremacro"
        / "validation"
        / "goldens"
        / "var.json"
    )
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest} (irf shape {orth.shape})")


if __name__ == "__main__":
    main()
