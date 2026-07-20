"""Solve the fertility DSGE (Alonso-Ortiz, adjustment-costs variant)
around its calibrated BGP and plot IRFs to the three shocks.

Demonstrates puremacro.dsge.solve_fertility on a model OTHER than SW07,
proving the BGP+Klein machinery is composable with new DSGEs.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from puremacro.dsge import solve_fertility


def main() -> None:
    sol = solve_fertility()
    print("Fertility-adj-costs BGP:")
    for name in ("c", "k", "y", "n", "b", "l_w", "u"):
        print(f"  {name:6s} = {sol.ss[name]:.4f}")
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    horizon = 20
    plot_vars = ["y", "n", "b"]
    for col, shock in enumerate(sol.shock_names):
        irf = sol.irf(shock, horizon=horizon)
        for row, var in enumerate(plot_vars):
            ax = axes[row, col]
            ax.plot(irf.index, irf[var], "k-")
            ax.axhline(0.0, color="0.7", lw=0.5)
            ax.set_title(f"{var} <- shock {shock}", fontsize=9)
            if row == 2:
                ax.set_xlabel("quarter")
    fig.suptitle("Fertility DSGE: IRFs to 1-SD shocks", fontsize=11)
    fig.tight_layout()
    fig.savefig("dsge_fertility_demo.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
