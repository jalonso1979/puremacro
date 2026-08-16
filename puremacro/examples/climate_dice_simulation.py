"""DICE Climate-Macroeconomic Simulation & Policy Analysis.

Demonstrates simulating the Dynamic Integrated Climate-Economy (DICE) model
following Nordhaus (2018, *PNAS*) and Golosov et al. (2014, *Econometrica*):
- 150-year macroeconomic and climate trajectory projections (2020-2170).
- Atmospheric carbon accumulation across 3 oceanic/atmospheric reservoirs.
- Surface warming anomaly and output damage shares.
- Social Cost of Carbon (SCC) trajectory under climate tax policy.

Outputs:
  puremacro/examples/output/climate_dice_simulation.png
"""
from __future__ import annotations

import pathlib
import matplotlib.pyplot as plt

from puremacro.climate import simulate_dice_model


def main():
    print("--- 1. Simulating DICE Model (Baseline vs. Accelerated Carbon Tax) ---")
    res_base = simulate_dice_model(
        n_periods=30,
        time_step_years=5,
        start_year=2020,
        carbon_tax_initial=40.0,
        carbon_tax_growth=0.02,
    )
    print("Baseline Scenario:")
    print(res_base.summary())

    res_policy = simulate_dice_model(
        n_periods=30,
        time_step_years=5,
        start_year=2020,
        carbon_tax_initial=80.0,
        carbon_tax_growth=0.035,
    )
    print("\nAccelerated Carbon Tax Scenario:")
    print(res_policy.summary())

    print("\n--- 2. Generating Publication Figures ---")
    out_dir = pathlib.Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "climate_dice_simulation.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.patch.set_facecolor("#fafbfc")

    # Panel A: Surface Warming Anomaly
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#ffffff")
    ax_a.plot(res_base.trajectories.index, res_base.trajectories["temperature_anomaly"], color="#d62728", lw=2, label="Base Tax ($40/t, +2%/yr)")
    ax_a.plot(res_policy.trajectories.index, res_policy.trajectories["temperature_anomaly"], color="#2ca02c", lw=2, linestyle="--", label="Accelerated Tax ($80/t, +3.5%/yr)")
    ax_a.axhline(1.5, color="gray", linestyle=":", label="1.5°C Paris Target")
    ax_a.axhline(2.0, color="gray", linestyle="-.", label="2.0°C Upper Guardrail")
    ax_a.set_title("(a) Mean Surface Temperature Anomaly (°C above Pre-industrial)", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Year")
    ax_a.set_ylabel("Temperature Anomaly (°C)")
    ax_a.legend(fontsize=8)
    ax_a.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Global CO2 Emissions Trajectory
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#ffffff")
    ax_b.plot(res_base.trajectories.index, res_base.trajectories["emissions"], color="#d62728", lw=2, label="Base Scenario")
    ax_b.plot(res_policy.trajectories.index, res_policy.trajectories["emissions"], color="#2ca02c", lw=2, linestyle="--", label="Accelerated Scenario")
    ax_b.set_title("(b) Annual Global CO2 Emissions (GtCO2/year)", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Year")
    ax_b.set_ylabel("Emissions (GtCO2)")
    ax_b.legend(fontsize=8)
    ax_b.grid(True, linestyle=":", alpha=0.6)

    # Panel C: Climate Damage Fraction of GDP
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#ffffff")
    ax_c.plot(res_base.trajectories.index, res_base.trajectories["damage_fraction"] * 100, color="#d62728", lw=2, label="Base Damages")
    ax_c.plot(res_policy.trajectories.index, res_policy.trajectories["damage_fraction"] * 100, color="#2ca02c", lw=2, linestyle="--", label="Accelerated Damages")
    ax_c.set_title("(c) Macroeconomic Climate Damages (% of Gross World Product)", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Year")
    ax_c.set_ylabel("Damage (% GDP)")
    ax_c.legend(fontsize=8)
    ax_c.grid(True, linestyle=":", alpha=0.6)

    # Panel D: Social Cost of Carbon Trajectory
    ax_d = axes[1, 1]
    ax_d.set_facecolor("#ffffff")
    ax_d.plot(res_base.trajectories.index, res_base.trajectories["social_cost_of_carbon"], color="#1f77b4", lw=2, label="Base SCC")
    ax_d.plot(res_policy.trajectories.index, res_policy.trajectories["social_cost_of_carbon"], color="#ff7f0e", lw=2, linestyle="--", label="Accelerated Carbon Price")
    ax_d.set_title("(d) Social Cost of Carbon & Implied Carbon Tax ($/tCO2)", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Year")
    ax_d.set_ylabel("SCC ($/tCO2)")
    ax_d.legend(fontsize=8)
    ax_d.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to: {out_file}")


if __name__ == "__main__":
    main()
