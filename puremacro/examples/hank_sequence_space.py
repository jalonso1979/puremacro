"""Sequence-Space HANK Model Simulation & Multiplier Decomposition.

Demonstrates solving a Heterogeneous-Agent New Keynesian (HANK) model in sequence space
following Auclert, Bardóczy, Rognlie & Straub (2021, *Econometrica*):
- Stationary Aiyagari wealth distribution via Endogenous Grid Method (EGM).
- Sequence-Space consumption Jacobians J_C_r and J_C_Y.
- General Equilibrium transmission of monetary policy shocks.

Outputs:
  puremacro/examples/output/hank_sequence_space.png
"""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from puremacro.models import solve_hank_sequence_space


def main():
    print("--- 1. Solving Sequence-Space HANK Model ---")
    res = solve_hank_sequence_space(
        T=40,
        beta=0.985,
        gamma=1.0,
        r_ss=0.01,
        phi_pi=1.5,
        kappa=0.1,
        shock_magnitude=0.0025, # 25 bps hike
        shock_rho=0.7,
        n_a=60,
    )
    print(res.summary())

    print("\n--- 2. Generating Publication Figures ---")
    out_dir = pathlib.Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hank_sequence_space.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.patch.set_facecolor("#fafbfc")

    # Panel A: Stationary Asset Distribution
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#ffffff")
    ax_a.plot(res.asset_grid, res.steady_state_wealth_dist, color="#1f77b4", lw=2)
    ax_a.fill_between(res.asset_grid, 0, res.steady_state_wealth_dist, color="#1f77b4", alpha=0.25)
    ax_a.set_title("(a) Stationary Wealth Distribution D(a)", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Assets a")
    ax_a.set_ylabel("Probability Density")
    ax_a.grid(True, linestyle=":", alpha=0.6)

    # Panel B: MPC by Wealth Decile
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#ffffff")
    res.mpc_distribution.plot(kind="bar", ax=ax_b, color="#2ca02c", edgecolor="#333", alpha=0.85)
    ax_b.set_title("(b) Marginal Propensity to Consume (MPC) by Decile", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Quarterly MPC")
    ax_b.set_xticklabels(ax_b.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax_b.grid(True, linestyle=":", alpha=0.6)

    # Panel C: GE Output & Inflation Impulse Responses
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#ffffff")
    h = np.arange(len(res.irf_output))
    ax_c.plot(h, res.irf_output * 100, color="#d62728", lw=2, label="Output dY (%)")
    ax_c.plot(h, res.irf_consumption * 100, color="#ff7f0e", lw=2, linestyle="--", label="Consumption dC (%)")
    ax_c.plot(h, res.irf_inflation * 100, color="#1f77b4", lw=2, linestyle=":", label="Inflation dpi (%)")
    ax_c.axhline(0, color="black", lw=0.8, linestyle="--")
    ax_c.set_title("(c) GE Impulse Responses to 25 bps Rate Hike", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Horizon (Quarters)")
    ax_c.set_ylabel("Percentage Deviation")
    ax_c.legend()
    ax_c.grid(True, linestyle=":", alpha=0.6)

    # Panel D: Consumption Jacobian J_C_Y Heatmap
    ax_d = axes[1, 1]
    ax_d.set_facecolor("#ffffff")
    im = ax_d.imshow(res.jacobian_c_y[:15, :15], cmap="Blues", origin="upper")
    ax_d.set_title(r"(d) Sequence-Space Jacobian $\mathcal{J}_{C, Y}$", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Shock Period s")
    ax_d.set_ylabel("Response Period t")
    fig.colorbar(im, ax=ax_d, label="Consumption Response")

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to: {out_file}")


if __name__ == "__main__":
    main()
