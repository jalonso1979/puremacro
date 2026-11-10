"""Run shared derivatives: python -m puremacro.examples.equilibrium_linearization --output /tmp/equilibrium-derivatives.

The JSON/CSV records describe fixed discretizations, not grid convergence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from puremacro.core import EquilibriumSystem, linearize_equilibrium
from puremacro.vfi import (
    CollocationProblem, linearize_aiyagari, linearize_projection,
    solve_aiyagari_continuous,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("equilibrium-derivatives"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    kss = (.36 * .96) ** (1 / (1 - .36))
    problem = CollocationProblem(
        domain=(.5 * kss, 1.5 * kss), orders=6, method="euler", beta=.96,
        params={"alpha": .36, "delta": 1., "sigma": 1.},
        options={"tol": 1e-12, "max_iter": 500},
    )
    projection = problem.solve()
    growth = linearize_projection(projection, problem, branch_id="growth-benchmark")
    growth_result = growth.sensitivity()
    equilibrium = solve_aiyagari_continuous(
        beta=.95, n_a=30, N_k=60, n_z=3, xtol=1e-11, egm_tol=1e-11, tol_ge=1e-6,
    )
    households = linearize_aiyagari(equilibrium, branch_id="aiyagari-benchmark")
    household_result = households.sensitivity()
    if not growth_result.valid or not household_result.valid:
        raise RuntimeError(f"ineligible example: {growth.reasons}, {households.reasons}")
    nc = equilibrium.household_solution.policy_c.size
    nk, nz = equilibrium.distribution.pdf.shape
    mass_derivatives = household_result.jacobian[nc:-1].reshape(nk, nz, 2)
    capital_derivatives = np.sum(mass_derivatives * equilibrium.distribution.asset_grid[:, None, None], axis=(0, 1))
    table = pd.DataFrame(
        [projection.basis.evaluate(kss)[0] @ growth_result.jacobian,
         capital_derivatives, household_result.jacobian[-1]],
        index=["growth_policy_at_fixed_k", "aiyagari_capital", "aiyagari_interest_rate"],
        columns=growth_result.parameter_names,
    )
    table.to_csv(args.output / "sensitivities.csv", index_label="quantity")
    fold = EquilibriumSystem(lambda x, p: x * x - p, ("x",), ("p",))
    singular = linearize_equilibrium(fold, [0.], [0.], branch_id="fold-example")
    records = {}
    for name, lin in (("growth", growth), ("aiyagari", households)):
        result = lin.sensitivity()
        records[name] = {
            "status": result.status, "states": len(lin.state),
            "parameters": list(result.parameter_names),
            "equilibrium_residual": result.equilibrium_residual,
            "linearized_residual": result.linearized_residual,
            "condition_number": result.condition_number,
            "state_derivative_source": lin.state_derivative_source,
            "parameter_derivative_source": lin.parameter_derivative_source,
            "snapshot_id": result.snapshot_id, "branch_id": result.branch_id,
        }
    report = {
        "scope": "Dense local derivatives of fixed discretized equilibrium equations; no grid-convergence claim.",
        "models": records, "sensitivities": table.to_dict(orient="index"),
        "distribution_derivative_mass": mass_derivatives.sum(axis=(0, 1)).tolist(),
        "singular_example": {"status": singular.status, "reasons": singular.reasons,
                             "valid": singular.sensitivity().valid},
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(table.to_string())
    print(f"Singular example: {singular.status}; ordinary sensitivity withheld.")
    print(f"Records: {args.output}")


if __name__ == "__main__":
    main()
