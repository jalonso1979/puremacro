"""G7 narrative-instrument panel demo.

Builds a per-country panel from
    - Romer-Romer 2017 cross-country tax shocks (or fallback);
    - Cloyne 2013 UK narrative tax shocks (or fallback);
    - Smoke-tests the live-feed connectors for HMT, OBR, BMF, DG-ECFIN,
      DOF, and prints how many press items each returned.

This is the c-endgame proof: the data model is country-agnostic; the
country-scope expansion is purely additive (new ``sources/*.py`` and
new ``replication/*.py``), and the rest of the package consumes them
without changes.

Run:
    python -m puremacro.examples.narrative_g7_panel
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..narrative import NarrativeInstrument
from ..narrative.replication import (
    load_romer_romer_2017, load_cloyne_2013_uk,
)
from ..narrative.types import NarrativeEvent


_FALLBACK_RR2017_EVENTS = [
    # (country, date, value pct_gdp, note)
    ("USA", "1981-Q3", -0.020, "Reagan ERTA"),
    ("USA", "2001-Q2", -0.012, "EGTRRA"),
    ("USA", "2017-Q4", -0.012, "TCJA"),
    ("GBR", "1979-Q2", -0.018, "Thatcher VAT shift"),
    ("GBR", "1988-Q1", -0.014, "Lawson income-tax cut"),
    ("GBR", "2010-Q3", +0.020, "VAT rise austerity"),
    ("DEU", "1986-Q1", -0.008, "income-tax reform"),
    ("DEU", "2001-Q1", -0.012, "Schroeder tax reform"),
    ("FRA", "2007-Q3", -0.005, "TEPA loi"),
    ("FRA", "2013-Q1", +0.012, "deficit-reduction VAT"),
    ("ITA", "1992-Q4", +0.020, "Amato budget"),
    ("ITA", "2011-Q4", +0.018, "Monti austerity"),
    ("JPN", "1989-Q2", +0.025, "Consumption tax intro"),
    ("JPN", "1997-Q2", +0.015, "Consumption tax 5%"),
    ("CAN", "1991-Q1", +0.011, "GST introduction"),
    ("CAN", "2006-Q3", -0.010, "GST cut to 6%"),
]


def _fallback_rr2017() -> dict[str, NarrativeInstrument]:
    by_country: dict[str, list[NarrativeEvent]] = {}
    for code, q, v, note in _FALLBACK_RR2017_EVENTS:
        ev = NarrativeEvent(
            date=pd.Period(q, freq="Q").to_timestamp(),
            country=code, magnitude=abs(v), magnitude_unit="pct_gdp",
            target="consumption", subtarget="tax",
            sign=int(1 if v > 0 else -1), confidence=0.7,
            source_text=note, source_url="(synth fallback)",
            scoring_method="manual",
        )
        by_country.setdefault(code, []).append(ev)
    return {
        c: NarrativeInstrument.from_events(evs, target="consumption")
        for c, evs in by_country.items()
    }


def _fallback_cloyne() -> NarrativeInstrument:
    rows = [("1979-Q2", -0.018), ("1988-Q1", -0.014), ("1993-Q1", +0.010),
            ("2010-Q3", +0.020), ("2012-Q2", +0.005)]
    events = [NarrativeEvent(
        date=pd.Period(q, freq="Q").to_timestamp(),
        country="GBR", magnitude=abs(v), magnitude_unit="pct_gdp",
        target="consumption", subtarget="tax",
        sign=int(1 if v > 0 else -1), confidence=0.85,
        source_text="cloyne fallback", source_url="(synth)",
        scoring_method="manual",
    ) for q, v in rows]
    return NarrativeInstrument.from_events(events, target="consumption")


def _smoke_country_feeds() -> dict[str, int]:
    """Try the live country-source connectors and return item counts."""
    from ..narrative.sources import (
        iter_hmt_press, iter_obr_publications, iter_bmf_press,
        iter_tresor_press, iter_mef_press, iter_dof_press, iter_ecfin_press,
    )
    feeds = {
        "HMT (UK Treasury Atom)":     iter_hmt_press,
        "OBR (UK fiscal RSS)":        iter_obr_publications,
        "BMF (DE Min. Finance RSS)":  iter_bmf_press,
        "DG Trésor (FR RSS)":         iter_tresor_press,
        "MEF (IT RSS)":               iter_mef_press,
        "DOF (CA Atom)":              iter_dof_press,
        "DG-ECFIN (EU Atom)":         iter_ecfin_press,
    }
    counts = {}
    for name, fn in feeds.items():
        try:
            counts[name] = sum(1 for _ in fn())
        except Exception:
            counts[name] = 0
    return counts


def run_demo() -> dict:
    try:
        rr = load_romer_romer_2017()
        rr_live = True
    except Exception:
        rr = _fallback_rr2017()
        rr_live = False
    try:
        cloyne = load_cloyne_2013_uk()
        cloyne_live = True
    except Exception:
        cloyne = _fallback_cloyne()
        cloyne_live = False
    counts = _smoke_country_feeds()
    return {"rr2017": rr, "cloyne": cloyne, "counts": counts,
            "live_flags": (rr_live, cloyne_live)}


def main() -> None:
    out = run_demo()
    rr_live, cl_live = out["live_flags"]
    print("G7 narrative-instrument panel demo")
    print(f"  RR2017 panel: {'live' if rr_live else 'fallback'}, "
          f"{len(out['rr2017'])} countries.")
    print(f"  Cloyne UK:    {'live' if cl_live else 'fallback'}, "
          f"{len(out['cloyne'].events)} events.")
    print()
    print("RR-2017-style panel — per-country event count and span:")
    print(f"  {'country':>7s}  {'events':>7s}  {'first':>10s}  {'last':>10s}  "
          f"{'sum':>8s}")
    for code in sorted(out["rr2017"]):
        inst = out["rr2017"][code]
        nz = inst.quarterly[inst.quarterly != 0]
        if not len(nz):
            continue
        print(f"  {code:>7s}  {len(nz):>7d}  "
              f"{str(nz.index.min().date()):>10s}  "
              f"{str(nz.index.max().date()):>10s}  "
              f"{float(nz.sum()):>+8.3f}")
    print()

    print("Cloyne UK 2013 vs RR2017 GBR — quarterly correlation:")
    if "GBR" in out["rr2017"]:
        s_cl = out["cloyne"].quarterly
        s_rr = out["rr2017"]["GBR"].quarterly
        common = pd.concat([s_cl, s_rr], axis=1).dropna()
        if len(common) > 5:
            rho = float(common.corr().iloc[0, 1])
            print(f"  N common quarters: {len(common)}, ρ = {rho:+.3f}")
    print()

    print("Live country-source-connector smoke test (item counts):")
    for name, n in out["counts"].items():
        flag = "OK " if n > 0 else "no "
        print(f"  [{flag}] {name:35s}  {n:>4d} items")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        codes = sorted(out["rr2017"])
        ymin = 0.5
        for i, code in enumerate(codes):
            inst = out["rr2017"][code]
            q = inst.quarterly
            nz = q[q != 0]
            if not len(nz):
                continue
            ax.scatter(nz.index, [i] * len(nz),
                       s=30 + 600 * np.abs(nz.values),
                       c=["0.0" if v > 0 else "0.5" for v in nz.values],
                       edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(codes)))
        ax.set_yticklabels(codes)
        ax.set_xlabel("date")
        ax.set_title("RR-2017-style cross-country tax-shock dot plot "
                       "(filled=increase / gray=cut, size ∝ |%GDP|)")
        fig.tight_layout()
        fig.savefig(out_dir / "narrative_g7_panel.png",
                    dpi=120, bbox_inches="tight")
        print(f"\n  Figure saved: {out_dir / 'narrative_g7_panel.png'}")


if __name__ == "__main__":
    main()
