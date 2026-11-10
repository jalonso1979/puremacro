"""Trace a stylized fixed-saving growth model and independently check its folds.

Run: python -m puremacro.examples.equilibrium_branches --output /tmp/equilibrium-branches
The model is a numerical illustration, not an estimated or optimal-saving economy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from puremacro.core import EquilibriumSystem, continue_equilibrium, linearize_equilibrium


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("equilibrium-branches"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    intercept, depreciation = .05, .1

    def production(k):
        return intercept + k*k/(1+k*k)

    def saving_curve(k):
        return depreciation*k/production(k)

    system = EquilibriumSystem(
        lambda x, p: depreciation*x-p*production(x), ("capital",), ("saving_rate",),
        parameter_scale=[.2], residual_scale=[.1],
        state_jacobian=lambda x, p: np.array([[depreciation-p[0]*2*x[0]/(1+x[0]**2)**2]]),
        parameter_jacobian=lambda x, p: np.array([[-production(x[0])]]),
        label="Stylized fixed-saving growth",
    )
    branch = continue_equilibrium(
        system, [.05], [saving_curve(.05)], step=.03, max_step=.08, max_steps=180,
        parameter_bounds=(.01, .32), branch_id="fixed-saving-growth",
    )
    k, saving = branch.states[:, 0], branch.parameters[:, 0]
    exact_fold_k = np.sqrt(np.sort(np.roots([1+intercept, 2*intercept-1, intercept]).real))
    assert len(branch.folds) == 2 and all(f.status == "simple_fold" for f in branch.folds)
    fold_k = np.array([f.state[0] for f in branch.folds])
    fold_error = float(np.max(np.abs(fold_k-exact_fold_k)))
    assert fold_error < 2e-6
    target = .2
    polynomial_roots = np.sort(np.roots([depreciation, -target*(1+intercept), depreciation, -target*intercept]).real)
    recovered = []
    for i in range(len(k)-1):
        if (saving[i]-target)*(saving[i+1]-target) < 0:
            recovered.append(brentq(lambda capital: depreciation*capital-target*production(capital),
                                    k[i], k[i+1], xtol=1e-13))
    np.testing.assert_allclose(recovered, polynomial_roots, rtol=0, atol=1e-10)
    derivatives = [float(linearize_equilibrium(system, [root], [target]).sensitivity().jacobian[0, 0])
                   for root in recovered]
    curve_error = float(np.max(np.abs(saving-saving_curve(k))))
    assert curve_error < 2e-8
    branch.to_frame().to_csv(args.output/"branch.csv", index=False)
    branch.attempts_frame().to_csv(args.output/"attempts.csv", index=False)
    fold_records = [
        {"capital": float(f.state[0]), "saving_rate": float(f.parameters[0]), "status": f.status,
         "residual": f.equilibrium_residual, "curvature": f.curvature,
         "transversality": f.transversality, "left_index": f.left_index, "right_index": f.right_index}
        for f in branch.folds
    ]
    pd.DataFrame(fold_records).to_csv(args.output/"folds.csv", index=False)
    report = {
        "model": "k_next=(1-0.1)*k+s*(0.05+k^2/(1+k^2))",
        "scope": "One local branch of a stylized fixed-saving model; no automatic branch switching or global discovery.",
        "status": branch.status, "points": len(k),
        "max_scaled_residual": float(np.max(branch.residual_norms)),
        "max_closed_form_curve_error": curve_error, "max_fold_capital_error": fold_error,
        "folds": fold_records, "target_saving_rate": target, "recovered_capital": recovered,
        "polynomial_capital": polynomial_roots.tolist(), "capital_sensitivities": derivatives,
        "rejected_steps": sum(not a.accepted for a in branch.attempts),
        "settings": {"initial_step": .03, "max_step": .08, "max_steps": 180, "fold_tol": 1e-7,
                     "residual_tol": 1e-9, "state_scale": 1., "parameter_scale": .2, "residual_scale": .1},
    }
    (args.output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
    fine = np.linspace(k[0], k[-1], 1000)
    ax.plot(saving_curve(fine), fine, color=".65", linewidth=4, alpha=.55, label="Closed-form curve")
    ax.plot(saving, k, color="#145c80", linewidth=1.5, label="Continued equilibria")
    ax.scatter([f.parameters[0] for f in branch.folds], fold_k, marker="D", color="#ce5b2b",
               s=42, zorder=5, label="Refined folds")
    ax.axvline(target, color=".4", linestyle="--", linewidth=1)
    ax.scatter([target]*len(recovered), recovered, color="#293e33", s=35, zorder=5,
               label="Three roots at s=0.20")
    ax.set(xlabel="Saving rate, s", ylabel="Equilibrium capital, k",
           title="Two folds in a stylized growth equilibrium")
    ax.grid(alpha=.15)
    ax.legend(fontsize=9, loc="upper left")
    fig.savefig(args.output/"branch.png", dpi=170)
    fig.savefig(args.output/"branch.pdf")
    plt.close(fig)
    print(branch.summary().to_string(index=False))
    print(f"Maximum fold-location error: {fold_error:.3e}; roots at s=0.20: {recovered}")
    print(f"Records: {args.output}")


if __name__ == "__main__":
    main()
