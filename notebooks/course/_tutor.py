"""Offline AI tutor for the puremacro course companion (notebook-side helper).

Wraps puremacro's local-LLM engines so a learner can ask questions or check
answers fully offline ($0, no API key). Degrades gracefully when no engine is
available (or in the browser), pointing to the lesson's AI-exploration prompts.
Never raises — a tutor failure must not break a lesson notebook.
"""
from __future__ import annotations

_SYSTEM = (
    "You are a concise teaching assistant for a graduate macroeconomics course "
    "that uses the puremacro Python library. Explain clearly, show the key step, "
    "and keep answers short. When asked to check an answer, say whether it is "
    "correct and briefly why."
)

_FALLBACK = (
    "[tutor offline] No local LLM engine is available here (expected in the "
    "browser). Install one with `pip install puremacro[local-llm]` and a small "
    "model (e.g. via Ollama or MLX), or use the AI-exploration prompts below with "
    "any AI assistant."
)


def tutor(question: str, *, context: str = "", model: str = "qwen2.5:3b-instruct",
          engine: str = "auto") -> str:
    """Ask the offline tutor. Returns the answer, or a friendly fallback string
    if no local engine is available. Never raises."""
    try:
        from puremacro.narrative._local_engines import chat
    except Exception:
        return _FALLBACK
    prompt = _SYSTEM + "\n\n"
    if context:
        prompt += f"Lesson context:\n{context}\n\n"
    prompt += f"Question: {question}"
    try:
        return chat(model, prompt, engine=engine, max_tokens=512, temperature=0.2)
    except Exception as exc:  # graceful: any backend/availability error -> fallback
        return f"{_FALLBACK}\n(engine note: {type(exc).__name__})"
