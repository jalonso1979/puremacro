"""Empirical-Bayes hyperparameter search for a Minnesota BVAR
(Giannone-Lenza-Primiceri 2015, simplified).

Hand-tuned hyperparameters λ_1, λ_2, λ_3 are a famous BVAR
embarrassment — they're typically chosen by reference to Litterman's
folklore values. GLP 2015 propose treating them as random and
maximising the marginal likelihood of the data. This example runs the
grid version of that procedure on a simulated VAR and shows how the
optimal λ_1 (overall tightness) is data-dependent: bigger sample →
larger λ_1 (looser prior, let the data speak).

Run:
    python -m puremacro.examples.glp_lambda_search
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..var.bvar import minnesota_optimal_lambda, minnesota_gibbs


def _simulate(T: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = np.array([[0.6, 0.10, 0.0],
                   [0.05, 0.5, 0.10],
                   [0.0, 0.05, 0.55]])
    Sigma = np.array([[1.0, 0.3, 0.1],
                       [0.3, 1.0, 0.2],
                       [0.1, 0.2, 1.0]]) * 0.5 ** 2
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, 3))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(3)
    return pd.DataFrame(Y, columns=list("ABC"))


def run_demo() -> dict:
    rows = []
    optimums = {}
    for T in (60, 120, 240, 480):
        Y = _simulate(T=T)
        opt = minnesota_optimal_lambda(Y, p=2)
        optimums[T] = opt
        rows.append({"T": T,
                     "lambda1": opt["lambda1"],
                     "lambda2": opt["lambda2"],
                     "lambda3": opt["lambda3"],
                     "log_ml": opt["log_ml"]})
    return {"summary": pd.DataFrame(rows), "optimums": optimums}


def main() -> None:
    out = run_demo()
    print("GLP-2015 empirical-Bayes Minnesota hyperparameter search")
    print(f"  Sample size grows; expect optimal λ_1 to relax toward larger values.")
    print()
    print(out["summary"].to_string(index=False))
    print()
    print("Inspecting the marginal-likelihood surface at T=240:")
    grid = pd.DataFrame(out["optimums"][240]["grid"])
    grid_pivot = grid.pivot_table(
        index="lambda1", columns="lambda2",
        values="log_ml", aggfunc="max",
    ).round(2)
    print(grid_pivot.to_string())

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
        s = out["summary"]
        ax.semilogx(s["T"], s["lambda1"], marker="o", color="0.0", lw=1.0,
                     label="λ_1*")
        ax.set_xlabel("Sample size T")
        ax.set_ylabel("Optimal λ_1")
        ax.set_title("Empirical-Bayes prior tightness vs sample size")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "glp_lambda_search.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'glp_lambda_search.png'}")


if __name__ == "__main__":
    main()
