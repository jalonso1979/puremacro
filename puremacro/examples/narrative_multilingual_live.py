"""Live multilingual fetch + IMF Article-IV + Google News across the
G7 countries plus the IMF and ECB.

Demonstrates the full c-endgame surface:
  - Verified live RSS/Atom for HMT, OBR, BMF (DE), Trésor (FR),
    MEF (IT), Department of Finance Canada, EU press corner, ECB.
  - Multilingual Google News search per country (en, de, fr, it, ja, …).
  - IMF Article IV consultation reports for an arbitrary list of
    country names — the multilateral source that covers every country
    in the wedges panel.
  - Optional LLM scoring (Anthropic / OpenAI via env vars) on a
    handful of recent items per country.

Run:
    python -m puremacro.examples.narrative_multilingual_live
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from ..narrative.sources import (
    iter_hmt_press, iter_obr_publications,
    iter_bmf_press, iter_tresor_press, iter_mef_press,
    iter_dof_press, iter_ecfin_press, iter_ecb_press,
    iter_google_news, iter_imf_articleiv,
)


# Country-by-source matrix.
_PROBES = [
    ("USA", "US Federal Register feed proxy via Google News",
     lambda: list(iter_google_news("infrastructure bill OR stimulus package",
                                     country="USA"))[:10]),
    ("GBR", "HMT (UK Treasury) Atom",
     lambda: list(iter_hmt_press())[:10]),
    ("GBR", "OBR fiscal RSS",
     lambda: list(iter_obr_publications())[:10]),
    ("DEU", "BMF (DE) press releases (German)",
     lambda: list(iter_bmf_press(language="de"))[:10]),
    ("DEU", "Google News DE — Bundeshaushalt",
     lambda: list(iter_google_news("Bundeshaushalt", country="DEU"))[:10]),
    ("FRA", "DG Trésor Atom (French)",
     lambda: list(iter_tresor_press())[:10]),
    ("FRA", "Google News FR — plan de relance",
     lambda: list(iter_google_news("plan de relance", country="FRA"))[:10]),
    ("ITA", "MEF press releases (Italian)",
     lambda: list(iter_mef_press(category="press"))[:10]),
    ("ITA", "Google News IT — manovra di bilancio",
     lambda: list(iter_google_news("manovra di bilancio", country="ITA"))[:10]),
    ("JPN", "Google News JA — 補正予算",
     lambda: list(iter_google_news("補正予算", country="JPN"))[:10]),
    ("CAN", "Department of Finance Canada Atom",
     lambda: list(iter_dof_press())[:10]),
    ("EU",  "European Commission press corner (all DGs)",
     lambda: list(iter_ecfin_press(filter_ecfin=False))[:10]),
    ("EUR", "ECB press releases (English)",
     lambda: list(iter_ecb_press(language="en"))[:10]),
]


_IMF_COUNTRIES = ["United States", "Mexico", "Japan", "Germany", "France",
                   "Italy", "United Kingdom", "Brazil", "India", "Canada"]


def main() -> None:
    print("=== Multilingual live fetch — G7 + EU + IMF + ECB ===\n")
    print(f"  {'country':>5s}  {'feed':<55s}  {'items':>5s}  example title")
    print(f"  {'-'*5}  {'-'*55}  {'-'*5}  {'-'*60}")
    rows: list[dict[str, Any]] = []
    for country, label, fn in _PROBES:
        try:
            items = fn()
        except Exception:
            items = []
        n = len(items)
        sample = ""
        if n > 0:
            sample = items[0][1][:60].replace("\n", " ")
        rows.append({"country": country, "feed": label, "n": n,
                       "sample": sample})
        flag = "OK" if n else "--"
        print(f"  [{flag}]    {country:>5s}  {label[:55]:<55s}  {n:>5d}  {sample!r}")

    print()
    print("=== IMF Article IV: latest report per country ===\n")
    print(f"  {'country':<22s}  {'date':>10s}  title")
    for cname in _IMF_COUNTRIES:
        items = list(iter_imf_articleiv(cname, fetch_abstracts=False,
                                          max_items=2))
        if not items:
            print(f"  {cname:<22s}  {'-':>10s}  (no recent report in series listing)")
            continue
        for d, t, _ in items[:1]:
            print(f"  {cname:<22s}  {str(d.date()):>10s}  {t[:60]!r}")

    # If an LLM key is set, demonstrate scoring on a handful of recent
    # German + French headlines (where the keyword lexicon would fail).
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    print()
    if not (has_anthropic or has_openai):
        print("(no LLM key set; skipping multilingual LLM scoring demo)")
    else:
        from ..narrative.scoring import score_llm, AnthropicBackend, OpenAIBackend
        backend = AnthropicBackend() if has_anthropic else OpenAIBackend()
        items_de = list(iter_google_news("Bundeshaushalt OR Steuerreform",
                                            country="DEU"))[:3]
        if items_de:
            print(f"\n=== LLM scoring (German, {type(backend).__name__}) ===")
            events = score_llm(items_de, backend=backend, country="DEU")
            for ev in events:
                print(f"  {ev.date.date()}  {ev.target:>11s}  sign={ev.sign:+d}  "
                      f"mag={ev.magnitude:7.2f}  conf={ev.confidence:.2f}")

    print()
    print(f"=== {sum(r['n'] for r in rows):,} total items pulled across "
          f"{len([r for r in rows if r['n']>0])}/{len(rows)} live feeds. ===")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(8, 4.0))
        labels = [f"{r['country']:<3s} | {r['feed'][:38]}" for r in rows]
        colors = ["0.0" if r["n"] > 0 else "0.7" for r in rows]
        ax.barh(range(len(rows)), [r["n"] for r in rows], color=colors)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Items returned (capped at 10 for display)")
        ax.set_title(
            "Live multilingual feed coverage — G7 + EU + ECB + IMF + Google News"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "narrative_multilingual_live.png",
                    dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'narrative_multilingual_live.png'}")


if __name__ == "__main__":
    main()
