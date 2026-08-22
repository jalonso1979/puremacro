"""Investment composition over time, per country, from the augmented
asset-breakdown panel.

Uses ``data_fetch/output/investment_by_asset_augmented.parquet`` (built
by ``data_fetch.run_fill_gaps`` — ~96k rows, 44 countries, 17 SNA
asset codes). For each selected country, computes the share of total
gross fixed capital formation accruing to:

    structures   = N111G + N112G        (dwellings + other buildings)
    equipment    = N1131G + N1132G + N1139G   (transport + ICT + other)
    IP           = N117G                       (intellectual-property products)

and plots the resulting time-varying composition. The picture is the
standard "rise of intangible capital" story: equipment + IP shares
grow at the expense of structures across all advanced economies.

Run:
    python -m puremacro.examples.asset_composition_dynamics
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MAV_ROOT = HERE.parents[3]
DATA = MAV_ROOT / "data_fetch" / "output" / "investment_by_asset_augmented.parquet"


_STRUCTURES = ["N111G", "N112G"]
_EQUIPMENT  = ["N1131G", "N1132G", "N1139G"]
_IP         = ["N117G"]
_TOTAL      = "N11G"

_FOCUS_COUNTRIES = ["USA", "DEU", "FRA", "GBR", "JPN", "ESP", "ITA", "NLD"]


def _country_shares(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Quarterly shares of structures/equipment/IP in total GFCF (volume).

    Uses ``PRICE_BASE='V'`` (current prices) so the shares are nominal
    expenditure shares — comparable across asset categories with
    different deflators.
    """
    sub = df[(df.REF_AREA == code) & (df.PRICE_BASE == "V")
              & (df.FREQ == "Q")]
    if sub.empty:
        # Fall back to volume terms if nominal isn't available.
        sub = df[(df.REF_AREA == code) & (df.PRICE_BASE == "L")
                  & (df.FREQ == "Q")]
    if sub.empty:
        return pd.DataFrame()
    wide = (
        sub.pivot_table(index="TIME_PERIOD", columns="INSTR_ASSET",
                          values="OBS_VALUE", aggfunc="first")
           .sort_index()
    )
    if _TOTAL not in wide.columns:
        return pd.DataFrame()
    out = pd.DataFrame(index=wide.index)
    out["structures"] = wide[[c for c in _STRUCTURES if c in wide.columns]].sum(axis=1)
    out["equipment"]  = wide[[c for c in _EQUIPMENT  if c in wide.columns]].sum(axis=1)
    out["ip"]         = wide[[c for c in _IP         if c in wide.columns]].sum(axis=1)
    total = wide[_TOTAL].replace(0, np.nan)
    out = out.divide(total, axis=0)
    return out.dropna(how="all")


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(
            f"Augmented panel not found at {DATA}. "
            "Run `python -m data_fetch.run_fill_gaps` first."
        )
    df = pd.read_parquet(DATA)
    print("Investment composition dynamics from "
          f"{DATA.name} ({len(df):,} rows, "
          f"{df.REF_AREA.nunique()} countries)")
    print()

    countries = [c for c in _FOCUS_COUNTRIES
                  if c in df.REF_AREA.unique()]
    print(f"Plotting {len(countries)} countries: {countries}")
    print()

    summary_rows = []
    panels: dict[str, pd.DataFrame] = {}
    for code in countries:
        sh = _country_shares(df, code)
        if sh.empty:
            continue
        panels[code] = sh
        # Decadal averages
        sh_local = sh.copy()
        sh_local["decade"] = pd.to_datetime(sh_local.index, errors="coerce").year // 10 * 10
        decadal = sh_local.groupby("decade")[["structures", "equipment", "ip"]].mean()
        for dec, row in decadal.iterrows():
            summary_rows.append({
                "code": code, "decade": int(dec),
                "structures_share": float(row["structures"]),
                "equipment_share": float(row["equipment"]),
                "ip_share": float(row["ip"]),
            })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        print("Decadal mean shares of GFCF by asset class:")
        pivot = summary.pivot_table(
            index="code", columns="decade",
            values="ip_share",
        ).round(3)
        print()
        print("  IP share by decade:")
        print(pivot.to_string())
        print()
        pivot_struct = summary.pivot_table(
            index="code", columns="decade",
            values="structures_share",
        ).round(3)
        print("  Structures share by decade:")
        print(pivot_struct.to_string())

    out_dir = HERE / "output"
    if out_dir.is_dir() and panels:
        import matplotlib.pyplot as plt
        n = len(panels)
        ncols = 2
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(11, 2.4 * nrows),
                                  sharex=True, sharey=True)
        axes = np.array(axes).reshape(-1)
        for ax, (code, sh) in zip(axes, panels.items()):
            sh_plot = sh.copy()
            try:
                idx = pd.PeriodIndex(sh_plot.index, freq="Q").to_timestamp()
            except Exception:
                idx = pd.to_datetime(sh_plot.index, errors="coerce")
            sh_plot.index = idx
            sh_plot = sh_plot.dropna(how="all").rolling(8, min_periods=4).mean()
            ax.stackplot(
                sh_plot.index,
                sh_plot["structures"].fillna(0),
                sh_plot["equipment"].fillna(0),
                sh_plot["ip"].fillna(0),
                colors=["0.7", "0.4", "0.0"],
                labels=["structures", "equipment", "IP"] if ax is axes[0] else [None]*3,
            )
            ax.set_title(code, loc="left", fontsize=10)
            ax.set_ylim(0, 1)
        for ax in axes[len(panels):]:
            ax.axis("off")
        axes[0].legend(loc="upper right", frameon=False, fontsize=9)
        for ax in axes[:len(panels)]:
            ax.set_ylabel("share of GFCF", fontsize=8)
        fig.suptitle("Investment composition: structures vs equipment vs IP "
                       "(8-Q rolling avg, nominal shares)")
        fig.tight_layout()
        fig.savefig(out_dir / "asset_composition_dynamics.png",
                     dpi=120, bbox_inches="tight")
        print()
        print(f"Figure saved: {out_dir / 'asset_composition_dynamics.png'}")


if __name__ == "__main__":
    main()
