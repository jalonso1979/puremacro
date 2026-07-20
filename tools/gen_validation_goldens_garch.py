"""Regenerate puremacro/validation/goldens/garch.json (dev-only; uses arch).

The references for the two PACKAGE GARCH(1,1) cases are the ``arch`` library's
Gaussian-MLE parameter estimates and its conditional-volatility path — an
independent implementation of the same model fitted to the same simulated
series. Run from the package dir:  python tools/gen_validation_goldens_garch.py
"""
from __future__ import annotations

import json
from pathlib import Path

import arch
import numpy as np
from arch import arch_model

from puremacro.validation.cases_garch import garch_demo_data


def main() -> None:
    d = garch_demo_data()
    y, burn = d["y"], d["burn"]

    res = arch_model(
        y, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False
    ).fit(disp="off")

    params = {
        "omega": float(res.params["omega"]),
        "alpha": float(res.params["alpha[1]"]),
        "beta": float(res.params["beta[1]"]),
    }
    cond_vol_tail = np.asarray(res.conditional_volatility, dtype=float)[burn:]

    out = {
        "_meta": {
            "generated_by": "tools/gen_validation_goldens_garch.py",
            "reference": (
                f"arch {arch.__version__} arch_model(y, mean='Zero', vol='GARCH', "
                "p=1, q=1, dist='normal', rescale=False).fit()"
            ),
            "dgp": (
                "puremacro.validation.cases_garch.garch_demo_data "
                "(seed=20260531, T=5000, true (omega,alpha,beta)=(0.05,0.08,0.90))"
            ),
            "burn": burn,
            "regenerate": "python tools/gen_validation_goldens_garch.py",
        },
        "params_vs_arch": params,
        "cond_vol_vs_arch": {"sigma_tail": cond_vol_tail.tolist()},
    }
    dest = (
        Path(__file__).resolve().parents[1]
        / "puremacro"
        / "validation"
        / "goldens"
        / "garch.json"
    )
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {dest} (params {params}, sigma_tail len {len(cond_vol_tail)})"
    )


if __name__ == "__main__":
    main()
