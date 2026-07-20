"""Custom-corpus pipeline (workflow A): user-supplied texts → keyword
scoring → quarterly IV → diagnostics. No network, no LLM.

This is the most basic narrative-IV pipeline. We hand-write a tiny
corpus of US fiscal-policy announcements (ARRA 2009, sequester 2013,
TCJA 2017, CARES 2020, IIJA 2021, IRA 2022) and let the keyword
scorer turn it into two parallel quarterly series — government
investment and government consumption — ready to feed into
``proxy_svar`` or ``lp_iv``.

Run:
    python -m puremacro.examples.narrative_custom_corpus
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..narrative import NarrativeInstrument, event_density
from ..narrative.scoring import score_keyword


_CORPUS = [
    (pd.Timestamp("2009-02-17"),
     "Today the President signed the American Recovery and Reinvestment "
     "Act, a 787 billion stimulus package including infrastructure "
     "investment in highways, broadband, and clean energy.",
     "http://example.com/arra"),
    (pd.Timestamp("2010-08-10"),
     "Congress passed a 26 billion package for state aid and government "
     "wages.",
     "http://example.com/2010-aid"),
    (pd.Timestamp("2013-03-01"),
     "The sequester takes effect: 85 billion spending cut across "
     "federal agencies and defense.",
     "http://example.com/sequester"),
    (pd.Timestamp("2020-03-27"),
     "CARES Act signed: 2.2 trillion stimulus package emphasising "
     "transfer payments, unemployment benefits, and household relief.",
     "http://example.com/cares"),
    (pd.Timestamp("2020-12-27"),
     "Consolidated Appropriations Act 2021: 900 billion stimulus "
     "package extending unemployment benefits.",
     "http://example.com/caa"),
    (pd.Timestamp("2021-03-11"),
     "American Rescue Plan: 1.9 trillion stimulus package for transfer "
     "payments, including direct stimulus checks.",
     "http://example.com/arp"),
    (pd.Timestamp("2021-11-15"),
     "Bipartisan infrastructure bill signed: 550 billion infrastructure "
     "investment over 5 years for highways and broadband.",
     "http://example.com/iija"),
    (pd.Timestamp("2022-08-16"),
     "Inflation Reduction Act: 369 billion clean energy capital "
     "expenditure programme.",
     "http://example.com/ira"),
]


def run_demo() -> dict:
    events = score_keyword(_CORPUS, country="USA")
    inv = NarrativeInstrument.from_events(events, target="investment")
    con = NarrativeInstrument.from_events(events, target="consumption")
    return {"events": events, "investment": inv, "consumption": con}


def main() -> None:
    out = run_demo()
    print(f"Custom-corpus demo: {len(out['events'])} events scored from "
          f"{len(_CORPUS)} announcements")
    print()
    print("Events (date / target / subtarget / sign / magnitude USD bn):")
    for e in out["events"]:
        print(f"  {e.date.date()}  {e.target:>11s}  "
              f"{(e.subtarget or '-'):>10s}  sign={e.sign:+d}  "
              f"mag={e.magnitude:7.1f}")
    print()
    print("Investment instrument (non-zero quarters):")
    inv_q = out["investment"].quarterly
    print(inv_q[inv_q != 0].to_string())
    print()
    print("Consumption instrument (non-zero quarters):")
    con_q = out["consumption"].quarterly
    print(con_q[con_q != 0].to_string())
    print()
    inv_d = out["investment"].diagnostics()
    con_d = out["consumption"].diagnostics()
    print(f"Investment: n_events={inv_d['n_events']}, "
          f"n_quarters={inv_d['n_quarters']}, "
          f"max_zero_run={inv_d['density']['max_gap_q']}q")
    print(f"Consumption: n_events={con_d['n_events']}, "
          f"n_quarters={con_d['n_quarters']}, "
          f"max_zero_run={con_d['density']['max_gap_q']}q")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.bar(inv_q.index - pd.Timedelta(days=15), inv_q.values, width=20,
               color="0.0", label="Government investment IV")
        ax.bar(con_q.index + pd.Timedelta(days=15), con_q.values, width=20,
               color="0.5", label="Government consumption IV")
        ax.axhline(0, color="0.5", lw=0.4)
        ax.set_title("Custom narrative instruments (US, 2009-2022)")
        ax.set_ylabel("Signed magnitude (USD bn)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "narrative_custom_corpus.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'narrative_custom_corpus.png'}")


if __name__ == "__main__":
    main()
