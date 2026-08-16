"""Central Bank Narrative Sentiment & Monetary Policy Transmission.

Demonstrates NLP and narrative identification in puremacro:
1. Apel-Blix-Grimaldi (2014) and Picault-Renault (2017) hawkish-dovish tone extraction
   from central bank policy communications.
2. Construction of high-frequency central bank narrative sentiment time series.
3. Jordà (2005) Local Projections estimating the dynamic transmission of narrative
   communication surprises on interbank interest rates, sovereign yields, and inflation.

Outputs:
  puremacro/examples/output/central_bank_narrative_sentiment.png
"""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from puremacro.datasets import load_banxico_stance, load_macro_monthly
from puremacro.lp import lp_hac
from puremacro.narrative.indices import tone


def main():
    print("--- 1. Processing Central Bank Narrative Communications ---")
    # Sample FOMC and Banxico policy statements (opening remarks & minutes)
    corpus = [
        ("2021-06-15", "The Committee decided to maintain the target range for the federal funds rate at 0 to 1/4 percent. Progress on vaccinations has reduced the spread of COVID-19, but inflation has risen, largely reflecting transitory factors.", {}),
        ("2021-11-03", "Inflation is elevated, largely reflecting factors that are expected to be transitory. Supply bottlenecks and price pressures have broadened across sectors.", {}),
        ("2022-03-16", "The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run. In support of these goals, the Committee decided to raise the target range for the federal funds rate.", {}),
        ("2022-06-15", "The Committee is strongly committed to returning inflation to its 2 percent objective. Decided to raise interest rates by 75 basis points to curb persistent inflationary pressures and overheating labor markets.", {}),
        ("2022-09-21", "Recent indicators point to modest growth in spending and production. Price stability is the responsibility of the Federal Reserve and serves as the bedrock of our economy.", {}),
        ("2022-12-14", "The Committee anticipates that ongoing increases in the target range will be appropriate in order to attain a stance of monetary policy that is sufficiently restrictive.", {}),
        ("2023-05-03", "Tighter credit conditions for households and businesses are likely to weigh on economic activity, hiring, and inflation. The extent of these effects remains uncertain.", {}),
        ("2023-12-13", "Inflation has eased over the past year but remains elevated. Economic growth has slowed from its strong pace in the third quarter.", {}),
        ("2024-06-12", "Inflation has eased substantially over the past year, but remains above our 2 percent longer-run goal. Modest further progress toward the Committee's 2 percent inflation objective has occurred.", {}),
    ]

    tone_res = tone(
        corpus,
        country="US",
        language="en",
        method="apel_blix_grimaldi",
        normalize="raw",
    )
    print("Sample Narrative Tone Series:")
    print(tone_res.series)

    print("\n--- 2. Merging Empirical Stance & Monthly Macro Series ---")
    df_banxico = load_banxico_stance()
    
    # Load interbank interest rate and inflation data
    data_dir = pathlib.Path(__file__).parent.parent.parent / "notebooks" / "course" / "data"
    df_rate = pd.read_csv(data_dir / "IR3TIB01MXM156N.csv")
    df_rate["date"] = pd.to_datetime(df_rate.iloc[:, 0])
    df_rate = df_rate.set_index("date")
    df_rate.index = df_rate.index.to_period("M")
    df_rate["rate_3m"] = pd.to_numeric(df_rate.iloc[:, 1], errors="coerce")

    df_cpi = pd.read_csv(data_dir / "CPALTT01MXM659N.csv")
    df_cpi["date"] = pd.to_datetime(df_cpi.iloc[:, 0])
    df_cpi = df_cpi.set_index("date")
    df_cpi.index = df_cpi.index.to_period("M")
    df_cpi["inflation_yoy"] = pd.to_numeric(df_cpi.iloc[:, 1], errors="coerce")

    # Combine into aligned monthly dataframe
    df_lp = pd.concat([df_banxico["banxico_direction"], df_rate["rate_3m"], df_cpi["inflation_yoy"]], axis=1).dropna()
    df_lp["narrative_shock"] = df_lp["banxico_direction"].to_numpy(dtype=float)
    df_lp = df_lp.reset_index(drop=True)

    print("\n--- 3. Estimating Jordà (2005) Local Projections ---")
    # Response of 3-month interest rate to +1 narrative hike shock
    irf_rate = lp_hac(
        df=df_lp,
        y="rate_3m",
        x="narrative_shock",
        horizons=range(0, 19),
        n_lags=2,
        controls=["inflation_yoy"],
        alpha=0.10,
    )

    # Response of CPI inflation to +1 narrative hike shock
    irf_cpi = lp_hac(
        df=df_lp,
        y="inflation_yoy",
        x="narrative_shock",
        horizons=range(0, 19),
        n_lags=2,
        controls=["rate_3m"],
        alpha=0.10,
    )

    print("Interest Rate LP Response (First 6 months):")
    print(irf_rate.head(6))

    print("\n--- 4. Generating Publication Figures ---")
    out_dir = pathlib.Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "central_bank_narrative_sentiment.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.patch.set_facecolor("#fafbfc")

    # Panel A: Tone Series
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#ffffff")
    tone_dates = [str(p) for p in tone_res.series.index]
    ax_a.bar(tone_dates, tone_res.series.values, color="#1f77b4", width=0.4, label="Net Hawkish Tone")
    ax_a.axhline(0, color="black", lw=0.8, linestyle="--")
    ax_a.set_title("(a) FOMC Communication Net Hawkishness Index", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Quarter")
    ax_a.set_ylabel("Net Tone Index (Hawk - Dove)")
    ax_a.legend()
    ax_a.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Narrative Monetary Policy Direction (Banxico)
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#ffffff")
    df_banxico_sub = df_banxico.loc["2015-01":"2023-12"]
    dates_b = [str(p) for p in df_banxico_sub.index]
    step = max(1, len(dates_b) // 10)
    ax_b.plot(range(len(dates_b)), df_banxico_sub["banxico_stance_signed"], color="#ff7f0e", lw=2, label="Cumulative Policy Stance")
    ax_b.set_xticks(range(0, len(dates_b), step))
    ax_b.set_xticklabels([dates_b[i] for i in range(0, len(dates_b), step)], rotation=30)
    ax_b.set_title("(b) Central Bank Cumulative Monetary Stance", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Date")
    ax_b.set_ylabel("Cumulative Stance Level")
    ax_b.legend()
    ax_b.grid(True, linestyle=":", alpha=0.6)

    # Panel C: LP Response of Interbank Rate
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#ffffff")
    ax_c.plot(irf_rate["h"], irf_rate["beta"], color="#1f77b4", lw=2, label="3M Interbank Rate IRF")
    ax_c.fill_between(irf_rate["h"], irf_rate["lo"], irf_rate["hi"], color="#1f77b4", alpha=0.2, label="90% HAC Band")
    ax_c.axhline(0, color="black", lw=0.8, linestyle="--")
    ax_c.set_title("(c) LP: Rate Response to Narrative Monetary Shock", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Horizon (Months)")
    ax_c.set_ylabel("Interest Rate (% pts)")
    ax_c.legend()
    ax_c.grid(True, linestyle=":", alpha=0.6)

    # Panel D: LP Response of Inflation
    ax_d = axes[1, 1]
    ax_d.set_facecolor("#ffffff")
    ax_d.plot(irf_cpi["h"], irf_cpi["beta"], color="#d62728", lw=2, label="Inflation Response IRF")
    ax_d.fill_between(irf_cpi["h"], irf_cpi["lo"], irf_cpi["hi"], color="#d62728", alpha=0.2, label="90% HAC Band")
    ax_d.axhline(0, color="black", lw=0.8, linestyle="--")
    ax_d.set_title("(d) LP: Inflation Cooling after Hawkish Communication", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Horizon (Months)")
    ax_d.set_ylabel("Inflation YoY (% pts)")
    ax_d.legend()
    ax_d.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to: {out_file}")


if __name__ == "__main__":
    main()
