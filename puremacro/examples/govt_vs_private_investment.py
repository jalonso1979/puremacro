"""Government vs private investment shares from the augmented panel.

Uses:
    data_fetch/output/phase_b_govt_gfcf.csv
        — Eurostat sector-account quarterly S.13 (general government)
          gross fixed capital formation.
    data_fetch/output/investment_by_asset_augmented.parquet
        — total GFCF (N11G).

For each country with both, computes the time-varying government share

    g_share_t = GFCF_govt_t / GFCF_total_t

and plots the panel. The standard story: government share collapses
post-2010 in the European austerity bloc (ESP, IRL, PRT, ITA), while
remaining stable in core economies.

Run:
    python -m puremacro.examples.govt_vs_private_investment
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MAV_ROOT = HERE.parents[3]
DATA_GOV = MAV_ROOT / "data_fetch" / "output" / "phase_b_govt_gfcf.csv"
DATA_AUG = MAV_ROOT / "data_fetch" / "output" / "investment_by_asset_augmented.parquet"


def _quarter_to_timestamp(s):
    """Robust YYYY-Qn → quarter-start Timestamp."""
    s = str(s)
    if "-Q" in s and len(s) == 7:
        return pd.Period(s, freq="Q").to_timestamp()
    return pd.Timestamp(s)


def main() -> None:
    if not DATA_GOV.exists() or not DATA_AUG.exists():
        raise SystemExit("Required data files missing. Run "
                          "`python -m data_fetch.run_fill_gaps` first.")
    gov = pd.read_csv(DATA_GOV)
    aug = pd.read_parquet(DATA_AUG)

    # Normalise quarter formats.
    gov["t"] = gov["TIME_PERIOD"].apply(_quarter_to_timestamp)
    aug["t"] = aug["TIME_PERIOD"].astype(str).apply(_quarter_to_timestamp)

    # Total GFCF, current prices preferred.
    total_v = aug[(aug.INSTR_ASSET == "N11G") & (aug.PRICE_BASE == "V")
                   & (aug.FREQ == "Q")]
    if total_v.empty:
        total_v = aug[(aug.INSTR_ASSET == "N11G") & (aug.FREQ == "Q")]

    print("Government investment share, quarterly")
    print(f"  Government series: {len(gov):,} rows, {gov.REF_AREA.nunique()} countries")
    print(f"  Total-GFCF series: {len(total_v):,} rows, {total_v.REF_AREA.nunique()} countries")

    # Merge on (country, t).
    merged = gov.rename(columns={"OBS_VALUE": "g_gfcf"}).merge(
        total_v.rename(columns={"OBS_VALUE": "total_gfcf"})[
            ["REF_AREA", "t", "total_gfcf"]
        ],
        on=["REF_AREA", "t"], how="inner",
    )
    if merged.empty:
        # Currency-mismatch may make the merge return zero overlap when total
        # is in NAC and govt in EUR. Try volume + nominal cross.
        total_l = aug[(aug.INSTR_ASSET == "N11G") & (aug.PRICE_BASE == "L")
                       & (aug.FREQ == "Q")]
        merged = gov.rename(columns={"OBS_VALUE": "g_gfcf"}).merge(
            total_l.rename(columns={"OBS_VALUE": "total_gfcf"})[
                ["REF_AREA", "t", "total_gfcf"]
            ],
            on=["REF_AREA", "t"], how="inner",
        )
    if merged.empty:
        print("No country has both government and total quarterly GFCF in "
              "comparable units; consider deflator-adjustment.")
        return

    merged["g_share"] = merged["g_gfcf"] / merged["total_gfcf"]
    merged = merged[(merged["g_share"] > 0) & (merged["g_share"] < 1)]

    # Headline: per-country pre-2010 vs post-2010 mean shares.
    merged["era"] = np.where(merged["t"] < pd.Timestamp("2010-01-01"),
                                "pre_2010", "post_2010")
    table = (
        merged.groupby(["REF_AREA", "era"])["g_share"]
              .mean().unstack().dropna()
    )
    table["delta"] = table["post_2010"] - table["pre_2010"]
    table = table.sort_values("delta")
    print()
    print("Pre-2010 vs post-2010 mean government share of GFCF:")
    print(table.round(3).to_string())

    out_dir = HERE / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        focus = ["DEU", "FRA", "GBR", "ITA", "ESP", "IRL", "PRT", "NLD"]
        focus = [c for c in focus if c in merged.REF_AREA.unique()]
        fig, axes = plt.subplots(2, 4, figsize=(13, 4.5),
                                  sharex=True, sharey=True)
        axes = np.array(axes).reshape(-1)
        for ax, code in zip(axes, focus):
            sub = merged[merged.REF_AREA == code].sort_values("t")
            sub["g_share_smooth"] = sub["g_share"].rolling(8,
                                                              min_periods=4).mean()
            ax.plot(sub["t"], sub["g_share_smooth"], color="0.0", lw=1.0)
            ax.fill_between(sub["t"], 0, sub["g_share_smooth"],
                              color="0.85")
            ax.axvline(pd.Timestamp("2010-01-01"), color="0.5", lw=0.5, ls="--")
            ax.set_title(code, loc="left", fontsize=10)
            ax.set_ylim(0, 0.40)
        for ax in axes[len(focus):]:
            ax.axis("off")
        for ax in axes[:len(focus)]:
            ax.set_ylabel("G share", fontsize=8)
        fig.suptitle("Government share of GFCF (8-Q rolling, dashed line: 2010)")
        fig.tight_layout()
        fig.savefig(out_dir / "govt_vs_private_investment.png",
                     dpi=120, bbox_inches="tight")
        print()
        print(f"Figure saved: {out_dir / 'govt_vs_private_investment.png'}")


if __name__ == "__main__":
    main()
