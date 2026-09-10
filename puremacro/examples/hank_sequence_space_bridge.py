"""Heterogeneous-Agent New Keynesian (HANK) Sequence-Space Bridge (.mod).

Demonstrates bridging Dynare-style .mod specifications carrying a hetagent_block
with the Sequence-Space Jacobian (SSJ) engine of Auclert, Bardóczy, Rognlie & Straub
(2021, Econometrica):
1. Loading a .mod file with a hetagent_block via `load_hank_mod`.
2. Inspecting the microeconomic stationary wealth distribution D*(a) and MPC(a).
3. Computing intertemporal sequence-space consumption Jacobians J_C_r and J_C_Y
   via the Fake-News Algorithm.
4. Simulating general equilibrium transition dynamics for an expansionary
   monetary policy shock (-25 bps rate cut, rho = 0.5).
5. Printing summary diagnostic tables and saving publication figures.

Outputs:
  puremacro/examples/output/hank_sequence_space_bridge.png

Español
--------
Demuestra la integración de archivos .mod estilo Dynare con bloque de agentes
heterogéneos (hetagent_block) y el motor de Jacobianos en el Espacio de Secuencias
(SSJ) de Auclert et al. (2021):
1. Carga de archivo .mod con hetagent_block mediante `load_hank_mod`.
2. Evaluación de la distribución estacionaria de riqueza D*(a) y MPC(a).
3. Cálculo de Jacobianos intertemporales J_C_r y J_C_Y mediante Fake-News.
4. Simulación de equilibrio general ante un choque monetario expansivo (-25 pb).
5. Impresión de tablas resumen y exportación de gráficos de publicación.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from puremacro.dsge.hank import load_hank_mod


def main():
    print("=" * 75)
    print("puremacro 3.0.0 -- HANK Sequence-Space Bridge Example")
    print("Auclert, Bardóczy, Rognlie & Straub (2021, Econometrica)")
    print("=" * 75)

    # -----------------------------------------------------------------------
    # 1. Load Reference .mod Model
    # -----------------------------------------------------------------------
    mod_path = (
        Path(__file__).resolve().parent.parent
        / "dsge"
        / "_references"
        / "hank_ssj.mod"
    )
    print("\n[Step 1] Loading reference HANK .mod file:")
    print(f"  -> {mod_path}")
    model = load_hank_mod(mod_path)

    print(f"  Declared variables: {model.variables}")
    print(f"  Exogenous shocks:   {model.shocks}")
    print(f"  HetAgent config:    {model.hetagent_config}")
    print(
        f"  Calibrated params:  beta={model.beta}, gamma={model.gamma}, "
        f"r_ss={model.r_ss}, phi_pi={model.phi_pi}, kappa={model.kappa}"
    )

    # -----------------------------------------------------------------------
    # 2. Inspect Stationary Distributions D*(a) and MPC(a)
    # -----------------------------------------------------------------------
    print("\n[Step 2] Microeconomic Stationary Distributions:")
    a_grid = model.asset_grid
    D_ss = model.asset_distribution
    mpc_ss = model.mpc_distribution

    total_mass = float(np.sum(D_ss))
    mean_assets = float(np.sum(a_grid * D_ss))
    mpc_borrowing = float(mpc_ss[0]) if mpc_ss is not None else float("nan")
    mpc_wealthy = float(mpc_ss[-1]) if mpc_ss is not None else float("nan")
    mpc_mean = float(np.sum(mpc_ss * D_ss)) if mpc_ss is not None else float("nan")

    print(f"  Asset grid range:      [{a_grid[0]:.2f}, {a_grid[-1]:.2f}] with {len(a_grid)} points")
    print(f"  Total probability:     {total_mass:.6f}")
    print(f"  Mean steady assets:    {mean_assets:.4f}")
    print(f"  Steady state C:        {model.steady_state['C']:.4f}")
    print(f"  Steady state Y:        {model.steady_state['Y']:.4f}")
    print(f"  MPC at a = 0 (HtM):    {mpc_borrowing:.4f} (constrained hand-to-mouth households)")
    print(f"  MPC at a_max:          {mpc_wealthy:.4f} (wealthy unconstrained households)")
    print(f"  Average aggregate MPC: {mpc_mean:.4f}")

    # -----------------------------------------------------------------------
    # 3. Compute Sequence-Space Jacobians via Fake-News Algorithm
    # -----------------------------------------------------------------------
    T = 40
    print(f"\n[Step 3] Computing Sequence-Space Jacobians (T={T}) via Fake-News Algorithm...")
    jacobians = model.compute_jacobians(T=T)
    J_C_r = jacobians["J_C_r"]
    J_C_Y = jacobians["J_C_Y"]

    dC0_dr0 = float(J_C_r[0, 0])
    dC0_dY0 = float(J_C_Y[0, 0])
    print(f"  Consumption-rate Jacobian shape:   {J_C_r.shape}")
    print(f"  Consumption-income Jacobian shape: {J_C_Y.shape}")
    print(f"  Intertemporal substitution (dC_0 / dr_0): {dC0_dr0:+.4f} (< 0, rate hike depresses consumption)")
    print(f"  Keynesian income multiplier (dC_0 / dY_0):{dC0_dY0:+.4f} (> 0, income stimulates consumption)")

    # -----------------------------------------------------------------------
    # 4. Simulate Expansionary Monetary Policy Shock (-25 bps)
    # -----------------------------------------------------------------------
    magnitude = -0.0025  # -25 basis points (-0.25% quarterly shock)
    rho = 0.5            # AR(1) persistence
    horizon = 40
    print("\n[Step 4] Simulating Expansionary Monetary Policy Shock:")
    print(f"  Shock: eps_m = {magnitude*10000:.1f} bps, rho = {rho}, horizon = {horizon} quarters")

    res = model.simulate(
        shock="eps_m",
        magnitude=magnitude,
        rho=rho,
        horizon=horizon,
        nonlinear=False,
    )

    # -----------------------------------------------------------------------
    # 5. Summary Table
    # -----------------------------------------------------------------------
    print("\n[Step 5] General Equilibrium Transition Dynamics Summary Table:")
    summary_df = res.summary()
    print(summary_df.to_string())

    print("\nMarkdown Representation:")
    print(res.to_markdown())

    # -----------------------------------------------------------------------
    # 6. Save Publication-Ready Figures
    # -----------------------------------------------------------------------
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hank_sequence_space_bridge.png"

    print(f"\n[Step 6] Rendering 4-panel diagnostic figure to:\n  -> {out_file}")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), dpi=150)
    fig.patch.set_facecolor("#fafbfc")

    # Panel A: Stationary Asset Distribution
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#ffffff")
    ax_a.plot(a_grid, D_ss, color="#1f77b4", lw=2.2, label=r"Density $\mathcal{D}^*(a)$")
    ax_a.fill_between(a_grid, 0, D_ss, color="#1f77b4", alpha=0.25)
    ax_a.set_title(r"(a) Stationary Wealth Distribution $\mathcal{D}^*(a)$", fontsize=11, fontweight="bold")
    ax_a.set_xlabel(r"Assets $a$")
    ax_a.set_ylabel("Probability Density")
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(loc="upper right", frameon=True)

    # Panel B: Marginal Propensity to Consume across Wealth Grid
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#ffffff")
    if mpc_ss is not None:
        ax_b.plot(a_grid, mpc_ss, color="#d62728", lw=2.2, label=r"$MPC(a)$")
        ax_b.axhline(mpc_mean, color="#333333", linestyle="--", lw=1.2, label=f"Mean MPC ({mpc_mean:.2f})")
    ax_b.set_title(r"(b) Marginal Propensity to Consume $MPC(a)$", fontsize=11, fontweight="bold")
    ax_b.set_xlabel(r"Assets $a$")
    ax_b.set_ylabel("Quarterly MPC")
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="upper right", frameon=True)

    # Panel C: Sequence-Space Jacobian Heatmap (J_C_Y)
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#ffffff")
    sub_T = 16
    im = ax_c.imshow(J_C_Y[:sub_T, :sub_T], cmap="YlGnBu", origin="upper")
    ax_c.set_title(r"(c) Sequence-Space Jacobian $\mathcal{J}_{C, Y}$ (Fake-News)", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Shock Period $s$")
    ax_c.set_ylabel("Response Period $t$")
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    cbar.set_label("Consumption Response")

    # Panel D: General Equilibrium Transition Paths
    ax_d = axes[1, 1]
    ax_d.set_facecolor("#ffffff")
    t_grid = np.arange(horizon)
    df_paths = res.transition_paths

    ax_d.plot(t_grid, df_paths["Y"] * 10000, color="#1f77b4", lw=2.2, label=r"Output $dY$ (bps)")
    ax_d.plot(t_grid, df_paths["C"] * 10000, color="#2ca02c", lw=2.0, linestyle="--", label=r"Consumption $dC$ (bps)")
    ax_d.plot(t_grid, df_paths["pi"] * 10000, color="#d62728", lw=2.0, linestyle="-.", label=r"Inflation $d\pi$ (bps)")
    ax_d.plot(t_grid, df_paths["r"] * 10000, color="#9467bd", lw=1.8, linestyle=":", label=r"Real Rate $dr$ (bps)")
    ax_d.axhline(0.0, color="#666666", lw=0.8, linestyle="-")
    ax_d.set_title("(d) GE Responses to -25 bps Monetary Policy Shock", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Quarters after Shock")
    ax_d.set_ylabel("Basis Points (bps)")
    ax_d.grid(True, linestyle=":", alpha=0.6)
    ax_d.legend(loc="upper right", fontsize=9, frameon=True)

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figure successfully: {out_file}")
    print("=" * 75)
    print("HANK Sequence-Space Bridge execution completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
