"""LWUI-wage demo: labor x uncertainty x wage triple co-occurrence.

Companion to ``narrative_indices_demo`` but specifically showcasing the
new ``lwui_wage`` index (added 2026-05-17). On a synthetic 8-document
corpus we contrast two flavours of labour-uncertainty:

  lwui      = labor_domain x uncertainty_tone x war_domain   (geopolitical risk)
  lwui_wage = labor_domain x uncertainty_tone x wage_domain  (pay / bargaining)

Same labour and uncertainty terms in both indices; the third domain
is what discriminates "war-flavoured labour uncertainty" from
"wage-flavoured labour uncertainty". Each index should score the
corresponding corpus higher.

Run:
    python -m puremacro.examples.lwui_wage_demo
"""
from __future__ import annotations

import pandas as pd

from ..narrative import lwui, lwui_wage


# Wage-heavy quarterly corpus: layoffs framed as pay / cost-of-living risk
_WAGE_CORPUS = [
    ("2020-01-15", "Workers and employers face mounting uncertainty about "
                   "wage growth and the next round of collective bargaining."),
    ("2020-04-15", "Wage settlements remain uncertain; labour unions warn of "
                   "pay cuts and a wage freeze for hourly employees."),
    ("2020-07-15", "Real wages stagnated and unit labor cost growth is "
                   "unclear for workers entering wage negotiations."),
    ("2020-10-15", "Minimum-wage increases are debated while pay raises and "
                   "cost-of-living adjustments are deferred amid uncertainty."),
]

# War-heavy quarterly corpus: labour stress framed as geopolitical risk
_WAR_CORPUS = [
    ("2020-01-15", "Geopolitical war risk weighs on the labour market as "
                   "uncertainty about sanctions and military mobilisation "
                   "drives layoffs in defence-exposed sectors."),
    ("2020-04-15", "Armed conflict and trade embargoes raise uncertainty "
                   "about workers in occupied territories."),
    ("2020-07-15", "Wartime uncertainty over invasion and missile attacks "
                   "displaces labour markets across the region."),
    ("2020-10-15", "Sanctions and military deployment heighten labour-market "
                   "uncertainty; supply-chain disruption complicates hiring."),
]


def _records(corpus):
    for date, text in corpus:
        yield (pd.Timestamp(date), text, "https://test/" + date,
               {"language": "en", "doctype": "press"})


def run_demo() -> dict:
    """Return both indices on both corpora — 4 series total."""
    wage_rec = list(_records(_WAGE_CORPUS))
    war_rec = list(_records(_WAR_CORPUS))
    return {
        "wage/lwui_wage": lwui_wage(wage_rec, country="USA", normalize="raw"),
        "wage/lwui":      lwui(wage_rec,      country="USA", normalize="raw"),
        "war/lwui_wage":  lwui_wage(war_rec,  country="USA", normalize="raw"),
        "war/lwui":       lwui(war_rec,       country="USA", normalize="raw"),
    }


def main() -> None:
    out = run_demo()
    print("LWUI variants — directional separation demo")
    print(f"  Wage corpus: {len(_WAGE_CORPUS)} docs; "
          f"War corpus: {len(_WAR_CORPUS)} docs\n")
    print(f"  {'series':<22s}  {'mean':>10s}  {'std':>10s}  {'max':>10s}")
    for name, ri in out.items():
        s = ri.series.dropna()
        if s.empty:
            print(f"  {name:<22s}   (empty)")
            continue
        print(f"  {name:<22s}  {s.mean():>+10.4f}  {s.std():>10.4f}  "
              f"{s.max():>+10.4f}")

    print("\nDirectional check (mean lwui_wage on wage > mean lwui on wage,")
    print("and mean lwui on war > mean lwui_wage on war):")
    m_ww = out["wage/lwui_wage"].series.dropna().mean()
    m_lw = out["wage/lwui"].series.dropna().mean()
    m_ww_r = out["war/lwui_wage"].series.dropna().mean()
    m_lw_r = out["war/lwui"].series.dropna().mean()
    print(f"  wage corpus: lwui_wage={m_ww:.4f}  vs  lwui (war)={m_lw:.4f}"
          f"  -> {'PASS' if m_ww > m_lw else 'FAIL'}")
    print(f"  war  corpus: lwui_wage={m_ww_r:.4f}  vs  lwui (war)={m_lw_r:.4f}"
          f"  -> {'PASS' if m_lw_r > m_ww_r else 'FAIL'}")


if __name__ == "__main__":
    main()
