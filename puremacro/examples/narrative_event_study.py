# skip: requires local narrative-IV panel + MAV wedges file (not shipped)
"""Event study around the largest narrative-IV episodes per country.

For each country in the homogeneous narrative-IV panel, identify the
quarters with the largest |signed magnitude| (top-N events). For each
such event, extract the path of real GDP, government consumption, and
gross fixed capital formation around it, and average across events.

The picture: contractionary fiscal events show up as a slowdown in
RGDP and RGCF; expansionary events show the opposite. Event-time
plots make this visible without any IV identification.

Run:
    python -m puremacro.examples.narrative_event_study
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MAV_ROOT = HERE.parents[3]
NAR_LONG = HERE / "output" / "narrative_iv_panel_long.csv"
WEDGES = MAV_ROOT / "wedges_panel_BGP_clean_nu2.csv"


def _wedges(w: pd.DataFrame) -> pd.DataFrame:
    if "qdate" in w.columns:
        w["t"] = pd.to_datetime(w["qdate"], errors="coerce")
    elif "date" in w.columns:
        w["t"] = pd.to_datetime(w["date"])
    w = w.sort_values(["code", "t"]).reset_index(drop=True)
    w["log_rgdp"] = np.log(w["RGDP"])
    w["log_rgcf"] = np.log(w["RGCF"])
    if "RGOV" in w.columns:
        w["log_rgov"] = np.log(w["RGOV"])
    return w


def _normalise_at_event(series: pd.Series, event_idx: int) -> pd.Series:
    """Centre a series so it equals 100 at the event quarter."""
    base = series.iloc[event_idx]
    if pd.isna(base) or base == 0:
        return pd.Series(dtype=float)
    return 100.0 * series / base


def main() -> None:
    if not NAR_LONG.exists() or not WEDGES.exists():
        raise SystemExit("Required input panels are missing.")
    events = pd.read_csv(NAR_LONG, parse_dates=["date"])
    wed = _wedges(pd.read_csv(WEDGES))

    # Pick the 3 largest |signed| events per country; require event date
    # to fall inside the wedges-panel window.
    events["signed"] = events["sign"].astype(int) * events["magnitude"].astype(float)
    events = events[events["date"] >= wed["t"].min()].copy()
    events = events[events["date"] <= wed["t"].max() - pd.Timedelta("365D")]
    events["abs_signed"] = events["signed"].abs()
    top = (
        events.sort_values("abs_signed", ascending=False)
              .groupby("country").head(3)
              .reset_index(drop=True)
    )

    print("Narrative-shock event study")
    print(f"  Top-3 events per country (within wedges-panel window): "
          f"{len(top)} events across {top.country.nunique()} countries")
    print()

    H_pre, H_post = 4, 12
    paths: dict[str, list[dict]] = {var: [] for var in ("log_rgdp", "log_rgcf", "log_rgov")}

    for _, ev in top.iterrows():
        sub = wed[wed.code == ev["country"]].reset_index(drop=True)
        if sub.empty:
            continue
        # Find event quarter index.
        diffs = (sub["t"] - ev["date"]).abs()
        idx = int(diffs.idxmin())
        if diffs.iloc[idx] > pd.Timedelta("90D"):
            continue
        for var in paths:
            if var not in sub.columns:
                continue
            window = sub[var].iloc[max(0, idx - H_pre):idx + H_post + 1]
            if len(window) < (H_pre + H_post + 1):
                continue
            normalised = 100.0 * np.exp(window.values - window.iloc[H_pre]) - 100.0
            paths[var].append({
                "country": ev["country"],
                "sign": int(ev["sign"]),
                "magnitude": float(ev["magnitude"]),
                "h_relative": np.arange(-H_pre, H_post + 1),
                "value": normalised,
            })

    out_dir = HERE / "output"

    print("Mean event-time path (% deviation from event quarter, normalised):")
    print()
    summary_rows = []
    for var, p_list in paths.items():
        if not p_list:
            continue
        # Stack and group by sign.
        stacked = []
        for p in p_list:
            for h, v in zip(p["h_relative"], p["value"]):
                stacked.append({"variable": var,
                                  "sign": p["sign"],
                                  "h": int(h),
                                  "value": float(v)})
        s = pd.DataFrame(stacked)
        for sgn in (+1, -1):
            mean_path = (
                s[s["sign"] == sgn].groupby("h")["value"].mean()
                .reindex(range(-H_pre, H_post + 1))
            )
            count = (
                s[s["sign"] == sgn].groupby("h")["country"].count()
            ) if False else (
                s[s["sign"] == sgn].groupby("h")["value"].count()
                .reindex(range(-H_pre, H_post + 1))
            )
            for h in mean_path.index:
                summary_rows.append({
                    "variable": var, "sign": sgn, "h": h,
                    "mean": float(mean_path.loc[h]) if pd.notna(mean_path.loc[h]) else np.nan,
                    "n": int(count.loc[h]) if pd.notna(count.loc[h]) else 0,
                })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        pivot = summary.pivot_table(
            index="h", columns=["variable", "sign"], values="mean",
        ).round(2)
        print(pivot.to_string())

    if out_dir.is_dir() and not summary.empty:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.0), sharex=True)
        var_titles = {"log_rgdp": "real GDP", "log_rgcf": "real GFCF",
                       "log_rgov": "real govt consumption"}
        for ax, (var, title) in zip(axes, var_titles.items()):
            for sgn, color in [(+1, "0.0"), (-1, "0.5")]:
                sub = summary[(summary["variable"] == var) & (summary["sign"] == sgn)]
                if sub.empty or sub["mean"].isna().all():
                    continue
                ax.plot(sub["h"], sub["mean"],
                         color=color, lw=1.0,
                         label=f"sign = {sgn:+d}")
            ax.axhline(0.0, color="0.6", lw=0.4)
            ax.axvline(0.0, color="0.6", lw=0.4, ls="--")
            ax.set_title(title)
            ax.set_xlabel("event-time horizon (Q)")
            if ax is axes[0]:
                ax.legend(frameon=False)
                ax.set_ylabel("% deviation from event Q")
        fig.suptitle("Event-time response of real activity to top-3 |narrative| shocks per country")
        fig.tight_layout()
        fig.savefig(out_dir / "narrative_event_study.png",
                     dpi=120, bbox_inches="tight")
        print()
        print(f"Figure saved: {out_dir / 'narrative_event_study.png'}")


if __name__ == "__main__":
    main()
