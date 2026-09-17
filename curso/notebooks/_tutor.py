"""Offline AI tutor for the MAV course companion (notebook-side helper).

Wraps puremacro's local-LLM engines so a learner can ask questions or check
answers fully offline ($0, no API key). The course runs on a local install with
the network closed by default; when no local engine is installed on the machine
this degrades gracefully, pointing to the lesson's AI-exploration prompts.
Never raises — a tutor failure must not break a lesson notebook.
Provides bilingual support (English and Spanish) based on caller notebook or prompt language.
"""
from __future__ import annotations

import inspect
import re

_SYSTEM_EN = (
    "You are a concise teaching assistant for a graduate macroeconomics course "
    "that uses the puremacro Python library. Explain clearly, show the key step, "
    "and keep answers short. When asked to check an answer, say whether it is "
    "correct and briefly why."
)

_SYSTEM_ES = (
    "Eres un asistente docente conciso para un curso de posgrado en macroeconomía "
    "que utiliza la librería puremacro. Explica con claridad, muestra el paso clave "
    "y mantén las respuestas breves. Cuando te pidan revisar una respuesta, indica "
    "si es correcta y brevemente por qué."
)

_FALLBACK_ES = (
    "[tutor sin conexión] No hay ningún motor de LLM local instalado en esta "
    "máquina. Instálalo con `pip install \"puremacro[local-llm]\"` junto con un "
    "modelo pequeño (por ejemplo vía Ollama o MLX), o usa las preguntas de "
    "exploración con IA de más abajo con el asistente que prefieras. El resto "
    "del cuaderno funciona igual sin el tutor."
)

_FALLBACK_EN = (
    "[offline tutor] No local LLM engine is installed on this machine. "
    "Install one with `pip install \"puremacro[local-llm]\"` along with a "
    "small model (for example via Ollama or MLX), or use the AI-exploration "
    "questions below with your preferred assistant. The rest of the notebook "
    "works without the tutor."
)

_FALLBACK = _FALLBACK_ES


def _detect_lang(question: str = "", context: str = "", lang: str = "auto") -> str:
    """Detect whether Spanish or English should be used for prompt and fallback."""
    if lang in ("en", "es"):
        return lang

    # Inspect call stack for caller notebook filename
    try:
        frame = inspect.currentframe()
        depth = 0
        while frame and depth < 6:
            fname = frame.f_code.co_filename.lower()
            if "_es." in fname or fname.endswith("_es") or "/colab_es/" in fname:
                return "es"
            if any(k in fname for k in ("syllabus.", "business_cycle_facts.", "cross_country_facts.", "high_frequency.")):
                return "en"
            frame = frame.f_back
            depth += 1
    except Exception:
        pass

    # Linguistic heuristic on question and context text
    text = f"{question} {context}".lower()
    es_special = set("¿¡áéíóúüñ")
    if any(ch in es_special for ch in text):
        return "es"

    tokens = re.findall(r"\b\w+\b", text)
    if not tokens:
        return "en"

    es_words = {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "para",
        "por", "con", "sin", "sobre", "que", "qué", "cómo", "cuál", "cuáles", "donde",
        "dónde", "por qué", "porque", "explica", "cuaderno", "desempleo", "inflación",
        "producción", "trabajo", "horas", "tasa", "tasas", "país", "países", "ciclo",
        "modelo", "modelos", "consumo", "inversión", "duradero", "duraderos", "salarios",
        "renta", "brecha", "estados", "unidos", "españa", "méxico", "bienes", "medida",
        "subió", "cayó", "durante", "mientras", "frases", "comportan"
    }
    en_words = {
        "the", "a", "an", "of", "in", "for", "to", "with", "without", "on", "that", "which",
        "what", "how", "why", "explain", "briefly", "notebook", "unemployment", "inflation",
        "output", "labor", "labour", "hours", "rate", "rates", "country", "countries", "cycle",
        "model", "models", "consumption", "investment", "durable", "durables", "wages",
        "income", "gap", "united", "states", "spain", "mexico", "goods", "measure", "measured",
        "rose", "fell", "during", "while", "sentences", "behave", "is", "are", "was", "were"
    }

    es_count = sum(1 for tok in tokens if tok in es_words)
    en_count = sum(1 for tok in tokens if tok in en_words)

    if es_count > en_count:
        return "es"
    if en_count > es_count:
        return "en"
    return "en"


def tutor(question: str, *, context: str = "", model: str = "qwen2.5:3b-instruct",
          engine: str = "auto", lang: str = "auto") -> str:
    """Ask the offline tutor. Returns the answer, or a friendly fallback string
    if no local engine is available. Never raises.
    
    Supports bilingual English/Spanish fallback and system prompts.
    """
    resolved_lang = _detect_lang(question, context, lang)
    fallback_text = _FALLBACK_ES if resolved_lang == "es" else _FALLBACK_EN
    engine_detail = "detalle del motor" if resolved_lang == "es" else "engine detail"

    try:
        from puremacro.narrative._local_engines import chat
    except Exception:
        return fallback_text

    system_prompt = _SYSTEM_ES if resolved_lang == "es" else _SYSTEM_EN
    prompt = system_prompt + "\n\n"
    if context:
        ctx_label = "Contexto de la lección" if resolved_lang == "es" else "Lesson context"
        prompt += f"{ctx_label}:\n{context}\n\n"
    q_label = "Pregunta" if resolved_lang == "es" else "Question"
    prompt += f"{q_label}: {question}"

    try:
        return chat(model, prompt, engine=engine, max_tokens=512, temperature=0.2)
    except Exception as exc:  # graceful: any backend/availability error -> fallback
        return f"{fallback_text}\n({engine_detail}: {type(exc).__name__})"
