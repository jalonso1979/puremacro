"""Global fiscal stance — sum of signed narrative magnitudes across the
homogeneous IV panel, per quarter.

Tracks how synchronised fiscal-policy actions are around the world:
spikes correspond to global tightening / loosening waves
(Maastricht 1992-93, post-2008 stimulus, 2010-12 Eurozone austerity).

Two aggregations:
    raw  = Σ_c sign_c × magnitude_c (% GDP, equal-weighted across countries)
    abs  = Σ_c |sign_c × magnitude_c| (a fiscal-action intensity index,
           agnostic about direction)

Run:
    python -m puremacro.examples.global_fiscal_stance
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
NAR = HERE / "output" / "narrative_iv_panel_quarterly.csv"


def main() -> None:
    if not NAR.exists():
        raise SystemExit(f"Missing {NAR}. Run "
                          "`python -m puremacro.examples.narrative_homogeneous_panel`.")
    df = pd.read_csv(NAR, parse_dates=["date"]).set_index("date")
    df = df.sort_index()

    n_active = df.ne(0).sum(axis=1)
    raw = df.sum(axis=1)
    absv = df.abs().sum(axis=1)

    out = pd.DataFrame({
        "n_active_countries": n_active.astype(int),
        "global_stance":      raw,
        "global_intensity":   absv,
    })

    print("Global fiscal narrative stance")
    print(f"  Quarters with ≥1 country active:  "
          f"{int((n_active > 0).sum())} / {len(out)}")
    print(f"  Peak intensity quarter:           "
          f"{out['global_intensity'].idxmax().date()}  "
          f"(intensity={out['global_intensity'].max():.3f})")
    print(f"  Most contractionary quarter:      "
          f"{out['global_stance'].idxmin().date()}  "
          f"(stance={out['global_stance'].min():.3f})")
    print(f"  Most expansionary quarter:        "
          f"{out['global_stance'].idxmax().date()}  "
          f"(stance={out['global_stance'].max():.3f})")
    print()
    print("Annual aggregate (top 10 highest-intensity years):")
    annual = out["global_intensity"].resample("YE").sum()
    annual.index = annual.index.year
    print(annual.sort_values(ascending=False).head(10).round(3).to_string())

    out_dir = HERE / "output"
    out_dir.mkdir(exist_ok=True)
    out.to_csv(out_dir / "global_fiscal_stance.csv")

    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(9, 4.4), sharex=True)
        # Stance — bars (signed)
        axes[0].bar(
            raw.index, raw.values,
            color=["0.0" if v < 0 else "0.5" for v in raw.values],
            width=70, edgecolor="none",
        )
        axes[0].axhline(0, color="0.6", lw=0.4)
        axes[0].set_ylabel("Σ signed mag (% GDP)")
        axes[0].set_title("Global fiscal narrative stance "
                            "(black = contractionary, gray = expansionary)")
        # Intensity — line
        axes[1].plot(absv.index, absv.values, color="0.0", lw=0.9)
        axes[1].fill_between(absv.index, 0, absv.values,
                               color="0.0", alpha=0.15)
        axes[1].set_ylabel("Σ |signed mag| (% GDP)")
        axes[1].set_xlabel("date")
        axes[1].set_title("Global fiscal-action intensity")
        # Annotate the famous fiscal-policy waves.
        annotations = [
            ("1981-Q3", "Reagan tax cut"),
            ("1992-Q4", "Maastricht consolidation"),
            ("2009-Q1", "post-GFC stimulus"),
            ("2010-Q4", "EU austerity wave"),
            ("2017-Q4", "TCJA"),
        ]
        for q, label in annotations:
            ts = pd.Period(q, freq="Q").to_timestamp()
            for ax in axes:
                ax.axvline(ts, color="0.7", lw=0.5, ls="--")
            axes[0].text(ts, axes[0].get_ylim()[1] * 0.95, label,
                          rotation=90, va="top", ha="right",
                          fontsize=7, color="0.4")
        fig.tight_layout()
        fig.savefig(out_dir / "global_fiscal_stance.png",
                     dpi=120, bbox_inches="tight")
        print()
        print(f"Figure saved: {out_dir / 'global_fiscal_stance.png'}")
        print(f"CSV saved:    {out_dir / 'global_fiscal_stance.csv'}")


if __name__ == "__main__":
    main()
