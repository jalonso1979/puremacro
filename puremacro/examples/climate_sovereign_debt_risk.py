"""Climate Transition Risk and Sovereign Debt Sustainability.

Simulates the macroeconomic feedback between global warming damages, carbon taxation,
sovereign risk premia, and long-term public debt sustainability:
1. Unabated Warming Scenario (No Carbon Policy): Accelerating climate damages erode
   the tax base while climate adaptation costs explode, leading to fiscal insolvency.
2. Disorderly Late Transition: Sudden carbon tax shock in 2040 induces transition
   macro contraction before stabilization.
3. Orderly Green Fiscal Rule: Early carbon tax with full revenue recycling towards
   debt amortisation and green infrastructure, stabilizing debt-to-GDP.

Outputs:
  puremacro/examples/output/climate_sovereign_debt_risk.png
"""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from puremacro.climate import simulate_dice_model


def simulate_sovereign_fiscal_risk(
    *,
    dice_res,
    initial_debt_gdp: float = 0.60,
    base_tax_rate: float = 0.20,
    base_spending_rate: float = 0.19,
    adapt_cost_coef: float = 0.0015,
    spread_debt_coef: float = 0.03,
    spread_climate_coef: float = 0.005,
    r_star: float = 0.02,
) -> pd.DataFrame:
    """Simulate sovereign debt accumulation under climate transition trajectories."""
    df = dice_res.trajectories.copy()
    years = list(df.index)
    dt = 5 # 5-year step

    debt_gdp = []
    sovereign_spreads = []
    primary_balances = []
    adaptation_costs = []

    B_over_Y = initial_debt_gdp

    for yr in years:
        row = df.loc[yr]
        T_clim = row["temperature_anomaly"]
        Y_net = row["output_net"]
        Y_gross = row["output_gross"]
        carbon_tax_rev = (row["social_cost_of_carbon"] * row["emissions"] * 1e-3) / Y_net # as fraction of GDP

        # Adaptation expenditure increases quadratically with warming
        g_adapt = adapt_cost_coef * (T_clim ** 2)
        
        # Primary balance / GDP
        rev = base_tax_rate + carbon_tax_rev
        exp = base_spending_rate + g_adapt
        pb = rev - exp

        # Sovereign interest rate spread
        r_sovereign = r_star + spread_debt_coef * max(0.0, B_over_Y - 0.60) + spread_climate_coef * T_clim
        
        debt_gdp.append(B_over_Y * 100.0)
        sovereign_spreads.append(r_sovereign * 100.0)
        primary_balances.append(pb * 100.0)
        adaptation_costs.append(g_adapt * 100.0)

        # Output growth between steps
        g_growth = 0.015 * dt
        # Debt accumulation equation: B_{t+1}/Y = (1 + r - g) * B_t/Y - PrimaryBalance
        B_over_Y = max(0.0, (1.0 + (r_sovereign - 0.015) * dt) * B_over_Y - pb * dt)

    res_df = pd.DataFrame({
        "year": years,
        "debt_to_gdp": debt_gdp,
        "sovereign_rate": sovereign_spreads,
        "primary_balance": primary_balances,
        "adaptation_cost": adaptation_costs,
        "temperature_anomaly": df["temperature_anomaly"].values,
    }).set_index("year")

    return res_df


def main():
    print("--- 1. Simulating Climate-Fiscal Scenarios ---")
    
    # 1. Unabated Warming (zero carbon tax)
    dice_unabated = simulate_dice_model(
        n_periods=25,
        time_step_years=5,
        carbon_tax_initial=0.0,
        carbon_tax_growth=0.0,
        damage_coef=0.0035,
    )
    fiscal_unabated = simulate_sovereign_fiscal_risk(dice_res=dice_unabated)

    # 2. Disorderly Late Transition (delayed carbon tax)
    dice_late = simulate_dice_model(
        n_periods=25,
        time_step_years=5,
        carbon_tax_initial=10.0,
        carbon_tax_growth=0.01,
        damage_coef=0.0028,
    )
    fiscal_late = simulate_sovereign_fiscal_risk(dice_res=dice_late)

    # 3. Orderly Green Fiscal Transition
    dice_orderly = simulate_dice_model(
        n_periods=25,
        time_step_years=5,
        carbon_tax_initial=60.0,
        carbon_tax_growth=0.03,
        damage_coef=0.00236,
    )
    fiscal_orderly = simulate_sovereign_fiscal_risk(dice_res=dice_orderly)

    print(f"End-Century Debt/GDP (Unabated) : {fiscal_unabated.loc[2100, 'debt_to_gdp']:.1f}%")
    print(f"End-Century Debt/GDP (Late)     : {fiscal_late.loc[2100, 'debt_to_gdp']:.1f}%")
    print(f"End-Century Debt/GDP (Orderly)  : {fiscal_orderly.loc[2100, 'debt_to_gdp']:.1f}%")

    print("\n--- 2. Generating Publication Figures ---")
    out_dir = pathlib.Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "climate_sovereign_debt_risk.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.patch.set_facecolor("#fafbfc")

    # Panel A: Sovereign Debt to GDP
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#ffffff")
    ax_a.plot(fiscal_unabated.index, fiscal_unabated["debt_to_gdp"], color="#d62728", lw=2, label="Unabated Climate Damages")
    ax_a.plot(fiscal_late.index, fiscal_late["debt_to_gdp"], color="#ff7f0e", lw=2, linestyle="--", label="Disorderly Late Transition")
    ax_a.plot(fiscal_orderly.index, fiscal_orderly["debt_to_gdp"], color="#2ca02c", lw=2, label="Orderly Green Fiscal Rule")
    ax_a.axhline(60, color="gray", linestyle=":", label="60% Stability Threshold")
    ax_a.set_title("(a) Sovereign Debt-to-GDP Trajectory (%)", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Year")
    ax_a.set_ylabel("Public Debt (% of GDP)")
    ax_a.legend(fontsize=8)
    ax_a.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Sovereign Borrowing Rate
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#ffffff")
    ax_b.plot(fiscal_unabated.index, fiscal_unabated["sovereign_rate"], color="#d62728", lw=2, label="Unabated Risk")
    ax_b.plot(fiscal_late.index, fiscal_late["sovereign_rate"], color="#ff7f0e", lw=2, linestyle="--", label="Late Transition Risk")
    ax_b.plot(fiscal_orderly.index, fiscal_orderly["sovereign_rate"], color="#2ca02c", lw=2, label="Orderly Transition")
    ax_b.set_title("(b) Sovereign Real Borrowing Rate (r* + Spread)", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Year")
    ax_b.set_ylabel("Real Interest Rate (%)")
    ax_b.legend(fontsize=8)
    ax_b.grid(True, linestyle=":", alpha=0.6)

    # Panel C: Climate Adaptation Fiscal Spending
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#ffffff")
    ax_c.plot(fiscal_unabated.index, fiscal_unabated["adaptation_cost"], color="#d62728", lw=2, label="Unabated Adaptation Need")
    ax_c.plot(fiscal_orderly.index, fiscal_orderly["adaptation_cost"], color="#2ca02c", lw=2, label="Orderly Adaptation Cost")
    ax_c.set_title("(c) Public Climate Adaptation Costs (% of GDP)", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Year")
    ax_c.set_ylabel("Adaptation Spending (% GDP)")
    ax_c.legend(fontsize=8)
    ax_c.grid(True, linestyle=":", alpha=0.6)

    # Panel D: Surface Warming Anomaly
    ax_d = axes[1, 1]
    ax_d.set_facecolor("#ffffff")
    ax_d.plot(fiscal_unabated.index, fiscal_unabated["temperature_anomaly"], color="#d62728", lw=2, label="Unabated Warming")
    ax_d.plot(fiscal_orderly.index, fiscal_orderly["temperature_anomaly"], color="#2ca02c", lw=2, label="Orderly Mitigation")
    ax_d.axhline(1.5, color="gray", linestyle=":", label="1.5°C Paris Target")
    ax_d.axhline(2.0, color="gray", linestyle="-.", label="2.0°C Guardrail")
    ax_d.set_title("(d) Mean Surface Warming Anomaly (°C)", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Year")
    ax_d.set_ylabel("Warming Anomaly (°C)")
    ax_d.legend(fontsize=8)
    ax_d.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to: {out_file}")


if __name__ == "__main__":
    main()
