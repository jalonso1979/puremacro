"""Regenerate puremacro/validation/goldens/inference.json (dev-only; uses statsmodels).

The two PACKAGE references are statsmodels' Newey-West HAC standard errors
(``OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": L}).bse``), an independent
implementation of the same Bartlett-kernel sandwich. The seeded demo design is
imported from ``cases_inference`` so the golden and the puremacro computation
see byte-identical inputs.

Run from the package dir:  python tools/gen_validation_goldens_inference.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import statsmodels
import statsmodels.api as sm

from puremacro.validation.cases_inference import hac_demo_data


def _statsmodels_hac_bse(y: np.ndarray, X: np.ndarray, lags: int) -> np.ndarray:
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)})
    return np.asarray(res.bse, dtype=float)


def main() -> None:
    d = hac_demo_data()
    bse = _statsmodels_hac_bse(d["y"], d["X"], d["lags"])

    out = {
        "_meta": {
            "generated_by": "tools/gen_validation_goldens_inference.py",
            "reference": (
                f"statsmodels {statsmodels.__version__} "
                "OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': L}).bse"
            ),
            "dgp": (
                "puremacro.validation.cases_inference.hac_demo_data "
                "(AR(1) errors, seed=20260531, T=200, L=4)"
            ),
            "regenerate": "python tools/gen_validation_goldens_inference.py",
        },
        # Both PACKAGE cases share the same statsmodels HAC bse: ols_hac and the
        # standalone newey_west alias both reproduce the OLS HAC standard errors
        # on identical data/residuals.
        "ols_hac_vs_statsmodels": {"se": bse.tolist()},
        "newey_west_vs_statsmodels": {"se": bse.tolist()},
    }
    dest = (
        Path(__file__).resolve().parents[1]
        / "puremacro"
        / "validation"
        / "goldens"
        / "inference.json"
    )
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest} (se {bse})")


if __name__ == "__main__":
    main()
