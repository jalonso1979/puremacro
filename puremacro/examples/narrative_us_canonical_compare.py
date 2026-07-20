"""Link a user's US narrative-instrument series to the canonical
published instruments: Ramey (2011) defense news, Romer-Romer (2010)
exogenous tax, Mertens-Ravn (2013) unanticipated tax.

Workflow:
  1. Build a custom US series from a small fiscal-policy corpus
     (the same toy corpus used in ``narrative_custom_corpus``).
  2. Load each canonical instrument via the replication layer (with
     synthetic fallbacks when the public mirror is unreachable, so the
     demo always runs end-to-end).
  3. Pairwise correlate, lead-lag CCF, count event overlap.
  4. Plot a side-by-side calendar timeline.

The point is not to claim the toy corpus reproduces these published
series — it doesn't, and the diagnostic shows that explicitly. The
point is the *plumbing*: any custom narrative instrument the user
builds can now be benchmarked against the published ones with a few
lines of code.

Run:
    python -m puremacro.examples.narrative_us_canonical_compare
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..narrative import NarrativeInstrument, compare_to
from ..narrative.replication import (
    load_ramey_2011_defense, load_romer_romer_2010, load_mertens_ravn_2013,
)
from ..narrative.scoring import score_keyword
from ..narrative.types import NarrativeEvent


_CORPUS = [
    (pd.Timestamp("2009-02-17"),
     "President Obama signed the American Recovery and Reinvestment "
     "Act, an 787 billion stimulus package including infrastructure "
     "investment.", "http://example.com/arra"),
    (pd.Timestamp("2013-03-01"),
     "Sequester takes effect: 85 billion spending cut across federal "
     "agencies and defense.", "http://example.com/sequester"),
    (pd.Timestamp("2017-12-22"),
     "Tax Cuts and Jobs Act signed: 1.5 trillion of legislated tax "
     "cuts over 10 years.", "http://example.com/tcja"),
    (pd.Timestamp("2020-03-27"),
     "CARES Act: 2.2 trillion stimulus package for transfer payments "
     "and unemployment benefits.", "http://example.com/cares"),
    (pd.Timestamp("2021-11-15"),
     "Bipartisan infrastructure bill: 550 billion infrastructure "
     "investment over 5 years.", "http://example.com/iija"),
]


def _fallback_ramey() -> NarrativeInstrument:
    rows = [("1950-Q3", +0.020), ("1965-Q1", +0.012), ("1981-Q1", +0.010),
            ("2001-Q4", +0.007), ("2013-Q1", -0.005)]
    events = [NarrativeEvent(
        date=pd.Period(q, freq="Q").to_timestamp(),
        country="USA", magnitude=abs(v), magnitude_unit="pct_gdp",
        target="both", subtarget="defense",
        sign=int(1 if v > 0 else -1), confidence=0.75,
        source_text="ramey fallback", source_url="(synth)",
        scoring_method="manual",
    ) for q, v in rows]
    return NarrativeInstrument.from_events(events, target="both")


def _fallback_rr() -> NarrativeInstrument:
    rows = [("1962-Q4", -0.012), ("1981-Q3", -0.020), ("1986-Q4", +0.005),
            ("1993-Q3", +0.008), ("2001-Q2", -0.012), ("2003-Q2", -0.005),
            ("2017-Q4", -0.012)]
    events = [NarrativeEvent(
        date=pd.Period(q, freq="Q").to_timestamp(),
        country="USA", magnitude=abs(v), magnitude_unit="pct_gdp",
        target="consumption", subtarget="tax",
        sign=int(1 if v > 0 else -1), confidence=0.7,
        source_text="rr fallback", source_url="(synth)",
        scoring_method="manual",
    ) for q, v in rows]
    return NarrativeInstrument.from_events(events, target="consumption")


def _fallback_mr() -> NarrativeInstrument:
    rows = [("1962-Q4", -0.010), ("1981-Q3", -0.015), ("1986-Q4", +0.004),
            ("1993-Q3", +0.005), ("2001-Q2", -0.008), ("2003-Q2", -0.003),
            ("2017-Q4", -0.009)]
    events = [NarrativeEvent(
        date=pd.Period(q, freq="Q").to_timestamp(),
        country="USA", magnitude=abs(v), magnitude_unit="pct_gdp",
        target="consumption", subtarget="tax",
        sign=int(1 if v > 0 else -1), confidence=0.85,
        source_text="mr fallback", source_url="(synth)",
        scoring_method="manual",
    ) for q, v in rows]
    return NarrativeInstrument.from_events(events, target="consumption")


def _try_load(loader, fallback, **kwargs):
    try:
        return loader(**kwargs), True
    except Exception:
        return fallback(), False


def run_demo() -> dict:
    custom_events = score_keyword(_CORPUS, country="USA")
    custom = NarrativeInstrument.from_events(custom_events, target=None)
    ramey, ramey_live = _try_load(load_ramey_2011_defense, _fallback_ramey)
    rr, rr_live = _try_load(load_romer_romer_2010, _fallback_rr)
    mr, mr_live = _try_load(load_mertens_ravn_2013, _fallback_mr,
                              kind="unanticipated")

    series = {
        "Custom (toy corpus)":     custom.quarterly,
        "Ramey 2011 (defense)":    ramey.quarterly,
        "Romer-Romer 2010 (tax)":  rr.quarterly,
        "Mertens-Ravn 2013 (unant. tax)": mr.quarterly,
    }
    return {
        "custom": custom, "ramey": ramey, "rr": rr, "mr": mr,
        "series": series,
        "live_flags": (ramey_live, rr_live, mr_live),
    }


def main() -> None:
    out = run_demo()
    series = out["series"]
    rl, rrl, mrl = out["live_flags"]
    print("US narrative-instrument link demo")
    print(f"  Sources: Ramey={'live' if rl else 'fallback'}, "
          f"RR={'live' if rrl else 'fallback'}, "
          f"MR={'live' if mrl else 'fallback'}.")
    print()
    print("Per-instrument summary:")
    print(f"  {'name':40s}  n_events  n_q  date_min  date_max")
    for name, q in series.items():
        nz = q[q != 0]
        if len(nz):
            print(f"  {name:40s}  {len(nz):>8d}  {len(q):>3d}  "
                  f"{nz.index.min().date()}  {nz.index.max().date()}")
        else:
            print(f"  {name:40s}  {len(nz):>8d}    -  -         -")
    print()

    # Pairwise correlations on the common quarterly grid.
    df = pd.concat(series, axis=1).fillna(0.0)
    corr = df.corr()
    print("Pairwise contemporaneous correlation (zeros included):")
    print(corr.round(3).to_string())
    print()

    # Lead-lag CCF: custom vs each canonical
    print("Lead-lag cross-correlation: custom IV vs canonical (lag in Q)")
    print(f"  {'lag':>4}", end="")
    for name in list(series.keys())[1:]:
        print(f"  {name[:18]:>18s}", end="")
    print()
    for lag in (-4, -2, -1, 0, +1, +2, +4):
        print(f"  {lag:+4d}", end="")
        for name in list(series.keys())[1:]:
            cmp = compare_to(series["Custom (toy corpus)"], series[name],
                              max_lag=4)
            v = cmp.get("ccf", {}).get(lag, np.nan)
            print(f"  {v:>18.3f}", end="")
        print()

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(series), 1,
                                  figsize=(10, 1.5 * len(series)),
                                  sharex=True)
        for ax, (name, q) in zip(axes, series.items()):
            ax.bar(q.index, q.values, width=70, color="0.0")
            ax.axhline(0, color="0.5", lw=0.4)
            ax.set_title(name, loc="left")
            ax.set_ylabel(q.name or "value")
        axes[-1].set_xlabel("date")
        fig.suptitle(
            "US narrative instruments — custom IV vs canonical published series"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "narrative_us_canonical_compare.png",
                    dpi=120, bbox_inches="tight")
        print(f"\n  Figure saved: {out_dir / 'narrative_us_canonical_compare.png'}")


if __name__ == "__main__":
    main()
