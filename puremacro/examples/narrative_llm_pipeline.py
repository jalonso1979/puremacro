"""End-to-end LLM pipeline (workflow C): a small corpus → LLM scoring →
``NarrativeInstrument``.

If ``ANTHROPIC_API_KEY`` (or ``OPENAI_API_KEY``) is set, the demo
sends a few short fiscal-policy paragraphs to the LLM, parses the
returned JSON event list, and shows the resulting quarterly IV. Useful
for confirming the prompt + JSON-schema validation work end-to-end.

If no key is set, the demo falls back to keyword scoring on the same
texts so it always produces output, just less precise.

Run:
    python -m puremacro.examples.narrative_llm_pipeline
"""
from __future__ import annotations

import os

import pandas as pd

from ..narrative import NarrativeInstrument
from ..narrative.scoring import score_keyword, score_llm, AnthropicBackend, OpenAIBackend


_CORPUS = [
    (pd.Timestamp("2009-02-17"),
     "President Obama signed the American Recovery and Reinvestment Act "
     "today, providing $787 billion in fiscal stimulus including $105 "
     "billion for transportation infrastructure investment, $54 billion "
     "for clean-energy capital expenditure, and roughly $260 billion in "
     "tax relief and direct transfer payments to households.",
     "https://example.com/arra"),
    (pd.Timestamp("2013-03-01"),
     "Across-the-board federal spending cuts of approximately $85 "
     "billion took effect today as the budget sequester began. Defense "
     "outlays will be reduced by roughly $42.7 billion, and non-defense "
     "discretionary outlays by a similar amount.",
     "https://example.com/sequester"),
    (pd.Timestamp("2021-11-15"),
     "President Biden signed the Bipartisan Infrastructure Law, "
     "authorising $550 billion of new federal investment in roads, "
     "bridges, public transit, broadband, and the electric grid over "
     "the next five years.",
     "https://example.com/iija"),
]


def _backend_or_none():
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicBackend()
        except Exception:
            return None
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIBackend()
        except Exception:
            return None
    return None


def main() -> None:
    backend = _backend_or_none()
    if backend is not None:
        print(f"LLM scoring backend: {type(backend).__name__} ({backend.model})")
        events = score_llm(_CORPUS, backend=backend, country="USA")
    else:
        print("No LLM API key set; falling back to keyword scoring.")
        events = score_keyword(_CORPUS, country="USA")

    print(f"Scored {len(events)} events from {len(_CORPUS)} announcements.")
    for e in events:
        print(f"  {e.date.date()}  {e.target:>11s}  "
              f"{(e.subtarget or '-'):>10s}  sign={e.sign:+d}  "
              f"mag={e.magnitude:7.1f} {e.magnitude_unit}  "
              f"({e.scoring_method}, conf={e.confidence:.2f})")
    if not events:
        return

    inst_inv = NarrativeInstrument.from_events(events, target="investment")
    inst_con = NarrativeInstrument.from_events(events, target="consumption")
    print()
    print("Investment IV (non-zero quarters):")
    nz = inst_inv.quarterly[inst_inv.quarterly != 0]
    print(nz.to_string() if len(nz) else "  (none)")
    print()
    print("Consumption IV (non-zero quarters):")
    nz = inst_con.quarterly[inst_con.quarterly != 0]
    print(nz.to_string() if len(nz) else "  (none)")


if __name__ == "__main__":
    main()
