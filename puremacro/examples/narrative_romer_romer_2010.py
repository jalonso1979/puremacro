"""Replication-layer demo (workflow B): Romer-Romer (2010) US tax
narrative shocks → comparison against the Ramey defense-news series.

Mostly a structural-comparison demo: how do tax shocks (consumption-
relevant via disposable income) line up with defense news (mostly
government purchases)? Expected: low contemporaneous correlation,
since they identify different shocks.

Run:
    python -m puremacro.examples.narrative_romer_romer_2010
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..narrative import NarrativeInstrument, compare_to
from ..narrative.replication import (
    load_ramey_2011_defense, load_romer_romer_2010,
)
from ..narrative.types import NarrativeEvent


def _fallback_rr() -> NarrativeInstrument:
    rows = [
        ("1948-Q2", -0.005, "Postwar tax cuts"),
        ("1962-Q4", -0.012, "Kennedy tax cuts"),
        ("1981-Q3", -0.020, "Reagan ERTA"),
        ("1986-Q4", +0.005, "Tax Reform Act"),
        ("1993-Q3", +0.008, "OBRA 1993 tax increase"),
        ("2001-Q2", -0.012, "EGTRRA"),
        ("2003-Q2", -0.005, "JGTRRA"),
    ]
    events = [
        NarrativeEvent(
            date=pd.Period(q, freq="Q").to_timestamp(),
            country="USA",
            magnitude=abs(v),
            magnitude_unit="pct_gdp",
            target="consumption",
            subtarget="tax",
            sign=int(1 if v > 0 else -1),
            confidence=0.7,
            source_text=note,
            source_url="(synthetic fallback)",
            scoring_method="manual",
            metadata={"replication": "rr_2010_synthetic_fallback"},
        )
        for q, v, note in rows
    ]
    return NarrativeInstrument.from_events(
        events, target="consumption", aggregation="sum",
    )


def _fallback_ramey() -> NarrativeInstrument:
    rows = [
        ("1950-Q3", +0.020, "Korean War"),
        ("1965-Q1", +0.012, "Vietnam"),
        ("1981-Q1", +0.010, "Reagan defense"),
        ("2001-Q4", +0.007, "Post-9/11"),
        ("2013-Q1", -0.005, "Sequester"),
    ]
    events = [
        NarrativeEvent(
            date=pd.Period(q, freq="Q").to_timestamp(),
            country="USA",
            magnitude=abs(v),
            magnitude_unit="pct_gdp",
            target="both",
            subtarget="defense",
            sign=int(1 if v > 0 else -1),
            confidence=0.75,
            source_text=note,
            source_url="(synthetic fallback)",
            scoring_method="manual",
        )
        for q, v, note in rows
    ]
    return NarrativeInstrument.from_events(events, target="both")


def run_demo() -> dict:
    try:
        rr = load_romer_romer_2010()
        used_rr_mirror = True
    except Exception:
        rr = _fallback_rr()
        used_rr_mirror = False
    try:
        ram = load_ramey_2011_defense()
        used_ram_mirror = True
    except Exception:
        ram = _fallback_ramey()
        used_ram_mirror = False
    cmp = compare_to(rr.quarterly, ram.quarterly, max_lag=8)
    return {"rr": rr, "ram": ram, "cmp": cmp,
            "used_mirrors": (used_rr_mirror, used_ram_mirror)}


def main() -> None:
    out = run_demo()
    print("Romer-Romer 2010 vs Ramey 2011 narrative-instrument comparison")
    print(f"  RR: {len(out['rr'].events)} events; "
          f"mirror={'yes' if out['used_mirrors'][0] else 'fallback'}")
    print(f"  Ramey: {len(out['ram'].events)} events; "
          f"mirror={'yes' if out['used_mirrors'][1] else 'fallback'}")
    print()
    cmp = out["cmp"]
    print(f"  Quarterly overlap: {cmp['n_obs']} quarters")
    print(f"  Contemporaneous correlation: {cmp['correlation']:+.3f}  "
          f"(expected near 0 — different identifying variations)")
    print(f"  Non-zero overlap quarters: {cmp['nonzero_overlap_quarters']}")
    if "ccf" in cmp:
        print()
        print("  Lead-lag cross-correlation (RR leads Ramey at -k):")
        for k in (-4, -2, -1, 0, +1, +2, +4):
            if k in cmp["ccf"]:
                print(f"    lag {k:+d}q : {cmp['ccf'][k]:+.3f}")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        rr_q = out["rr"].quarterly
        ram_q = out["ram"].quarterly
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.bar(rr_q.index - pd.Timedelta(days=20), rr_q.values, width=25,
               color="0.0", label="Romer-Romer (2010) tax")
        ax.bar(ram_q.index + pd.Timedelta(days=20), ram_q.values, width=25,
               color="0.5", label="Ramey (2011) defense news")
        ax.axhline(0, color="0.5", lw=0.4)
        ax.set_title("Two narrative instruments — tax (RR) vs spending (Ramey)")
        ax.set_ylabel("Signed magnitude (% of GDP)")
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "narrative_rr_vs_ramey.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'narrative_rr_vs_ramey.png'}")


if __name__ == "__main__":
    main()
