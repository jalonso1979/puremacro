"""Build the homogeneous cross-country fiscal-narrative-IV panel.

Stacks every canonical narrative-fiscal-shock series in the
replication layer (DGLP 2011, RR 2017, RR 2010, MR 2013, Cloyne 2013,
Ramey 2011), harmonises units / signs / dates, deduplicates across
overlaps (e.g., RR-2010 ⊂ MR-2013), and writes a single
research-grade CSV.

When the public mirrors aren't reachable each loader injects a small
fallback synthetic series — the demo always runs, with a clear
caveat in the printout.

Run:
    python -m puremacro.examples.narrative_homogeneous_panel

Outputs land in ``puremacro/examples/output/`` :
    narrative_iv_panel_long.csv          one row per event
    narrative_iv_panel_quarterly.csv     wide (date × country) signed magnitudes
    narrative_iv_panel_investment.csv    investment-target subset
    narrative_iv_panel_consumption.csv   consumption-target subset
    narrative_iv_panel_dedup_audit.csv   each duplicate joined to its rep
    homogeneous_panel.png                coverage heatmap
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..narrative import HomogeneousFiscalPanel, DGLP_17


# ---------------------------------------------------------------------
# Synthetic fallback CSVs so the demo runs without the public mirrors.
# Real research use should pass ``canonical_csv_paths={...}`` pointing
# at the published replication archives.
# ---------------------------------------------------------------------
_DGLP_FALLBACK_ROWS = [
    # (year, country, tax_pct_gdp, exp_pct_gdp)
    (1981, "USA", 0.5, 0.3), (1982, "USA", 0.4, 0.6),
    (1983, "USA", 0.0, 0.7), (1990, "USA", 0.6, 0.0),
    (1993, "USA", 0.5, 0.4),
    (1980, "GBR", 0.7, 0.3), (1981, "GBR", 0.6, 0.5),
    (1994, "GBR", 0.6, 0.0), (2010, "GBR", 0.4, 1.0),
    (2011, "GBR", 0.3, 0.6),
    (1992, "ITA", 1.4, 0.5), (1993, "ITA", 0.4, 1.6),
    (1996, "ITA", 0.5, 1.0), (1997, "ITA", 0.6, 1.0),
    (2012, "ITA", 0.6, 1.5),
    (1981, "DEU", 0.0, 0.5), (1991, "DEU", 0.5, 0.5),
    (1996, "DEU", 0.0, 0.6), (1997, "DEU", 0.5, 0.4),
    (2003, "DEU", 0.0, 0.6),
    (1979, "FRA", 0.0, 0.4), (1995, "FRA", 0.5, 0.4),
    (1996, "FRA", 0.4, 0.5), (2011, "FRA", 0.5, 0.5),
    (2012, "FRA", 0.6, 0.5),
    (1980, "CAN", 0.0, 0.7), (1995, "CAN", 0.0, 1.5),
    (1996, "CAN", 0.4, 1.4), (1997, "CAN", 0.0, 1.0),
    (1980, "JPN", 0.5, 0.0), (1997, "JPN", 1.0, 0.0),
    (2014, "JPN", 0.7, 0.0),
    (1985, "BEL", 0.0, 0.7), (1993, "BEL", 0.5, 0.6),
    (1981, "DNK", 0.7, 0.5), (1986, "DNK", 0.6, 0.0),
    (1992, "FIN", 0.0, 1.0), (1995, "FIN", 0.5, 0.7),
    (1981, "IRL", 0.6, 0.4), (1987, "IRL", 0.0, 1.4),
    (2009, "IRL", 0.6, 1.5), (2010, "IRL", 0.6, 1.7),
    (1992, "ESP", 0.0, 0.5), (1996, "ESP", 0.0, 0.7),
    (2010, "ESP", 0.4, 1.4), (2011, "ESP", 0.6, 1.4),
    (1993, "SWE", 1.0, 1.0), (1995, "SWE", 0.5, 0.8),
    (1996, "SWE", 0.6, 0.9),
    (1992, "NLD", 0.0, 0.5), (2004, "NLD", 0.0, 0.6),
    (1981, "PRT", 0.5, 0.5), (1983, "PRT", 0.0, 1.0),
    (2011, "PRT", 0.6, 0.7), (2012, "PRT", 0.5, 1.5),
    (1986, "AUT", 0.5, 0.0), (1996, "AUT", 0.5, 0.5),
    (1985, "AUS", 0.0, 0.7), (1986, "AUS", 0.0, 0.5),
    (1987, "AUS", 0.0, 0.5),
]


def _write_fallback_dglp(path: Path) -> None:
    df = pd.DataFrame(_DGLP_FALLBACK_ROWS,
                       columns=["date", "country", "tax_based", "spending_based"])
    df.to_csv(path, index=False)


_RR2017_FALLBACK_ROWS = [
    # tax-cut events (sign = +1 expansionary in the source convention)
    (1962, "USA", -0.012), (1981, "USA", -0.020), (1986, "USA", +0.005),
    (2001, "USA", -0.012), (2003, "USA", -0.005), (2017, "USA", -0.012),
    (1979, "GBR", -0.018), (1988, "GBR", -0.014), (1993, "GBR", +0.010),
    (2010, "GBR", +0.020),
    (1986, "DEU", -0.008), (2001, "DEU", -0.012),
    (2007, "FRA", -0.005), (2013, "FRA", +0.012),
    (1989, "JPN", +0.025), (1997, "JPN", +0.015), (2014, "JPN", +0.012),
    (1991, "CAN", +0.011), (2006, "CAN", -0.010),
    (1987, "AUS", -0.008), (2000, "AUS", -0.012),
    (1986, "DNK", -0.005), (2009, "DNK", -0.008),
    (1991, "IRL", -0.005), (2009, "IRL", +0.020),
    (1992, "ITA", +0.020), (1997, "ITA", +0.015), (2011, "ITA", +0.018),
    (1990, "NLD", -0.005), (1994, "NLD", -0.010),
    (1992, "ESP", +0.010), (2010, "ESP", +0.018),
    (1994, "SWE", -0.005), (2007, "SWE", -0.010),
]


def _write_fallback_rr2017(path: Path) -> None:
    df = pd.DataFrame(_RR2017_FALLBACK_ROWS,
                       columns=["date", "country", "exog_tax"])
    df.to_csv(path, index=False)


def main() -> None:
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    fallback_dir = out_dir / "_fallback_csvs"
    fallback_dir.mkdir(exist_ok=True)

    # Write fallback CSVs and pass them as csv_path overrides so the
    # demo runs without depending on the public mirrors.
    dglp_path = fallback_dir / "dglp_synth.csv"
    rr2017_path = fallback_dir / "rr2017_synth.csv"
    _write_fallback_dglp(dglp_path)
    _write_fallback_rr2017(rr2017_path)

    print("Building homogeneous fiscal narrative-IV panel ...")
    panel = HomogeneousFiscalPanel.from_canonical(
        countries=DGLP_17,
        canonical_csv_paths={
            "dglp_2011": str(dglp_path),
            "rr_2017":   str(rr2017_path),
        },
    )
    print()
    print(f"  Raw events stacked:    {panel.metadata['n_raw_events']}")
    print(f"  After dedup:           {panel.metadata['n_after_dedup']}")
    print(f"  Distinct countries:    {len(panel.countries)} "
          f"({', '.join(panel.countries)})")
    print()
    print(f"  Loader status: ")
    for name, info in panel.metadata["load_summary"].items():
        flag = "OK " if info["ok"] else "FB "
        err = "" if info["ok"] else f"  ({info.get('error', '')[:80]}...)"
        print(f"    [{flag}]  {name}{err}")
    print()

    print("Per-country summary:")
    summ = panel.summary()
    print(summ[["country", "n_events", "first_date", "last_date",
                 "median_magnitude_pct_gdp"]].to_string(index=False))
    print()

    paths = panel.save(out_dir=out_dir)
    print("Saved:")
    for k, v in paths.items():
        print(f"  {k:20s}  {v}")
    if panel.audit is not None and not panel.audit.empty:
        print(f"  dedup_audit (rows): {len(panel.audit)}")

    # Cross-country correlation diagnostic.
    cc = panel.cross_correlations()
    if not cc.empty:
        print()
        print("Cross-country correlations (signed-quarterly panel):")
        print(cc.round(2).to_string())

    # Optional figure: country × year coverage heatmap.
    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return
        long = panel.to_long()
        long["year"] = pd.to_datetime(long["date"]).dt.year
        pivot = (
            long.groupby(["country", "year"]).size().unstack("year",
                                                                 fill_value=0)
            .reindex(panel.countries)
        )
        if pivot.empty:
            return
        fig, ax = plt.subplots(figsize=(10, max(3.5, 0.25 * len(panel.countries))))
        im = ax.imshow(pivot.values, cmap="Greys", aspect="auto",
                        interpolation="none")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_xticks(range(0, pivot.shape[1], max(1, pivot.shape[1] // 12)))
        ax.set_xticklabels(
            [str(pivot.columns[i]) for i in range(0, pivot.shape[1],
                                                       max(1, pivot.shape[1] // 12))],
            rotation=45, ha="right",
        )
        ax.set_title("Homogeneous fiscal narrative-IV panel — "
                       "events per (country, year)")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="events/year")
        fig.tight_layout()
        fig.savefig(out_dir / "homogeneous_panel.png",
                     dpi=120, bbox_inches="tight")
        print()
        print(f"  Heatmap saved: {out_dir / 'homogeneous_panel.png'}")


if __name__ == "__main__":
    main()
