"""Empirical Benchmark Replications: Galí (1999) & Mertens-Ravn (2013).

Demonstrates exact empirical replication using puremacro's built-in datasets:
1. Galí (1999, *AER*): Long-run Blanchard-Quah (BQ) identification of technology shocks
   on labor productivity and hours worked.
2. Mertens & Ravn (2013, *AER*): External instrument (Proxy SVAR) identification
   of narrative tax multiplier shocks.

Outputs:
  puremacro/examples/output/empirical_benchmark_replications.png
"""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from puremacro.datasets import load_gali1999, load_narrative_tax_shocks, load_macro_quarterly
from puremacro.var import fit_var, irf
from puremacro.var.identify import bq, proxy


def main():
    print("--- 1. Replicating Galí (1999, AER) Long-Run SVAR ---")
    df_gali = load_gali1999()
    # Variables: [dlprod, hours]
    Z_gali = df_gali[["dlprod", "hours"]].to_numpy(dtype=float)
    
    # Estimate VAR(4) with long-run BQ restriction (only technology shock has permanent productivity effect)
    bq_res = bq(Z_gali, p=4, horizon=20)
    print(bq_res.summary())

    # Technology shock is shock 0
    irf_prod_tech = bq_res.irf_point[:, 0, 0]  # Cumulated labor productivity level
    irf_hours_tech = bq_res.irf_point[:, 1, 0]  # Hours worked response

    print("\n--- 2. Replicating Mertens & Ravn (2013, AER) Proxy SVAR ---")
    df_macro_q = load_macro_quarterly()
    df_tax = load_narrative_tax_shocks()

    # Align sample
    common_idx = [idx for idx in df_macro_q.index if idx in df_tax.index]
    sub_macro = df_macro_q.loc[common_idx]
    sub_tax = df_tax.loc[common_idx]

    # Log GDP and Fed Funds
    gdp_log = np.log(sub_macro["real_gdp"].to_numpy(dtype=float)) * 100.0
    ffr = sub_macro["fed_funds"].to_numpy(dtype=float)
    Z_tax = np.column_stack([gdp_log, ffr])

    # Proxy instrument: unanticipated narrative tax shocks
    m_instrument = sub_tax["unanticipated"].to_numpy(dtype=float)

    # Estimate Proxy SVAR identifying the tax shock
    proxy_res = proxy(Z_tax, p=4, horizon=16, instrument_series=m_instrument, shock_target_idx=0)
    print(proxy_res.summary())

    irf_gdp_tax = proxy_res.irf_point[:, 0, 0]
    irf_ffr_tax = proxy_res.irf_point[:, 1, 0]

    print("\n--- 3. Generating Publication Figures ---")
    out_dir = pathlib.Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "empirical_benchmark_replications.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.patch.set_facecolor("#fafbfc")

    h_gali = np.arange(len(irf_hours_tech))

    # Panel A: Galí (1999) - Labor Productivity Level
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#ffffff")
    ax_a.plot(h_gali, irf_prod_tech, color="#1f77b4", lw=2, label="Productivity Level (Cumulative)")
    ax_a.axhline(0, color="black", lw=0.8, linestyle="--")
    ax_a.set_title("(a) Galí (1999): Labor Productivity to Tech Shock", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Horizon (Quarters)")
    ax_a.set_ylabel("Percentage Points")
    ax_a.legend()
    ax_a.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Galí (1999) - Hours Worked (Key Contraction Result)
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#ffffff")
    ax_b.plot(h_gali, irf_hours_tech, color="#d62728", lw=2, label="Hours Worked")
    ax_b.axhline(0, color="black", lw=0.8, linestyle="--")
    ax_b.set_title("(b) Galí (1999): Hours Worked Drop after Positive Tech Shock", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Horizon (Quarters)")
    ax_b.set_ylabel("Percentage Points")
    ax_b.legend()
    ax_b.grid(True, linestyle=":", alpha=0.6)

    h_tax = np.arange(len(irf_gdp_tax))

    # Panel C: Mertens & Ravn (2013) - Real GDP Response to Tax Shock
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#ffffff")
    ax_c.plot(h_tax, irf_gdp_tax, color="#d62728", lw=2, label="Real GDP Response")
    ax_c.axhline(0, color="black", lw=0.8, linestyle="--")
    ax_c.set_title("(c) Mertens & Ravn (2013): Output Contraction to Tax Increase", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Horizon (Quarters)")
    ax_c.set_ylabel("Log GDP (%)")
    ax_c.legend()
    ax_c.grid(True, linestyle=":", alpha=0.6)

    # Panel D: Mertens & Ravn (2013) - Fed Funds Rate Response
    ax_d = axes[1, 1]
    ax_d.set_facecolor("#ffffff")
    ax_d.plot(h_tax, irf_ffr_tax, color="#2ca02c", lw=2, label="Fed Funds Rate")
    ax_d.axhline(0, color="black", lw=0.8, linestyle="--")
    ax_d.set_title("(d) Mertens & Ravn (2013): Monetary Policy Reaction", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Horizon (Quarters)")
    ax_d.set_ylabel("Interest Rate (% pts)")
    ax_d.legend()
    ax_d.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to: {out_file}")


if __name__ == "__main__":
    main()
