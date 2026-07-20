"""Run puremacro's narrative LLM features for FREE on a local model.

No API key, no paid API. Install a local engine once:

    pip install "puremacro[local-llm]"     # MLX (Apple Silicon) + llama.cpp
    # OR install Ollama (https://ollama.com) and: ollama pull qwen2.5:3b

Then:  python -m puremacro.examples.narrative_local_llm

With no engine installed it falls back to the offline Mock (so this script
always runs); install an engine to see real local inference.

Español
-------
Ejecuta las funciones narrativas LLM de puremacro de forma GRATUITA con un
modelo local.

Sin clave de API ni API de pago. Instala un motor local una sola vez:

    pip install "puremacro[local-llm]"     # MLX (Apple Silicon) + llama.cpp
    # O instala Ollama (https://ollama.com) y: ollama pull qwen2.5:3b

Luego:  python -m puremacro.examples.narrative_local_llm

Si no hay ningún motor instalado, el script recurre al Mock sin conexión
(por lo que siempre se puede ejecutar); instala un motor para ver
inferencia local real.
"""
from __future__ import annotations

from puremacro.narrative.indices import get_default_provider, llm_prob_kernel
from puremacro.narrative.scoring import get_default_backend, score_llm

_CORPUS = [
    ("2020-03-15",
     "The government announced a $500 billion infrastructure investment package "
     "to be implemented next quarter.",
     "http://example.test/a"),
    ("2020-04-01",
     "Officials warned the outlook is highly uncertain and policy could shift "
     "abruptly amid unpredictable risks.",
     "http://example.test/b"),
]


def main() -> None:
    print("=== puremacro local LLM demo (free, $0) ===")

    backend = get_default_backend("qwen2.5-3b-instruct")
    events = score_llm(_CORPUS, backend=backend, kind="fiscal")
    print(f"[events] extracted {len(events)} fiscal event(s)")
    for ev in events:
        print(f"  - {ev.date.date()} sign={ev.sign} mag={ev.magnitude} "
              f"{ev.magnitude_unit}: {ev.source_text[:80]}")

    provider = get_default_provider("qwen2.5-3b-instruct")
    series = list(llm_prob_kernel(_CORPUS, provider=provider,
                                  category="economic uncertainty"))
    print("[index] P(uncertainty) per document:")
    for date, p in series:
        print(f"  - {date.date()}: {p:.3f}")


if __name__ == "__main__":
    main()
