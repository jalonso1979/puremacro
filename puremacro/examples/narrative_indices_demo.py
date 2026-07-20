"""Narrative indices demo — assembles all six indices from a synthetic
in-process corpus and prints summary statistics.

Run:
    python -m puremacro.examples.narrative_indices_demo

Español
-------
Demo de índices narrativos — construye los seis índices a partir de un
corpus sintético en memoria e imprime estadísticas de resumen.

Ejecución:
    python -m puremacro.examples.narrative_indices_demo
"""
from __future__ import annotations

import pandas as pd

from ..narrative import epu, mpu, gpr, tone, wui, lui


_SYNTHETIC_CORPUS = [
    # Q1 2020: turbulent
    ("2020-01-15", "Economic policy uncertainty rose sharply as the federal "
                   "reserve weighed an emergency rate cut amid pandemic risk."),
    ("2020-02-15", "Layoffs and hiring freezes spread across sectors as "
                   "geopolitical tensions and war risk unsettled markets."),
    ("2020-03-15", "The Fed cut rates 50 basis points; markets called the move "
                   "dovish and signalled further accommodation."),
    # Q2 2020: stabilising
    ("2020-04-15", "Conditions began to stabilise; uncertainty receded though "
                   "labor shortages persisted in some sectors."),
    ("2020-05-15", "Unemployment remained elevated; central banks signalled "
                   "lower for longer interest-rate guidance."),
    ("2020-06-15", "Tone became less dovish; some FOMC members hinted at "
                   "tightening conditions if recovery accelerated."),
    # Q3 2020: hawkish drift
    ("2020-07-15", "FOMC raised the target range and signalled tightening; "
                   "uncertainty about geopolitical sanctions remained."),
    ("2020-08-15", "Hawkish tone dominated; participants raised the policy "
                   "rate amid persistent inflationary pressure."),
]


def _records_4tuple(corpus):
    for date, text in corpus:
        yield (pd.Timestamp(date), text, "https://test/" + date,
               {"language": "en", "doctype": "press"})


def run_demo() -> dict:
    rec = list(_records_4tuple(_SYNTHETIC_CORPUS))
    return {
        "epu":  epu(rec, country="USA", normalize="raw"),
        "mpu":  mpu(rec, country="USA", normalize="raw"),
        "gpr":  gpr(rec, country="USA", normalize="raw"),
        "tone": tone(rec, country="USA", normalize="raw"),
        "wui":  wui(rec, country="USA", normalize="raw"),
        "lui":  lui(rec, country="USA", normalize="raw"),
    }


def main() -> None:
    out = run_demo()
    print("Narrative indices — synthetic-corpus demo")
    print(f"  Corpus size: {len(_SYNTHETIC_CORPUS)} documents over 8 months\n")
    for name, ri in out.items():
        d = ri.diagnostics()
        print(f"  {name:5s}  n_q={d['n_quarters']:>2d}  "
              f"mean={d['mean']:>+8.3f}  std={d['std']:>+7.3f}  "
              f"first={d['first_date']}  last={d['last_date']}")


if __name__ == "__main__":
    main()
