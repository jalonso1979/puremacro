"""Offline AI tutor for the puremacro course companion (notebook-side helper).

Wraps puremacro's local-LLM engines so a learner can ask questions or check
answers fully offline ($0, no API key). Degrades gracefully when no engine is
installed, pointing to the lesson's AI-exploration prompts.
Never raises — a tutor failure must not break a lesson notebook.
"""
from __future__ import annotations

_SYSTEM = (
    "Eres un asistente de enseñanza conciso para un curso de posgrado de "
    "macroeconomía que usa la biblioteca de Python puremacro. RESPONDE SIEMPRE "
    "EN ESPAÑOL, aunque la pregunta venga en otro idioma. Explica con claridad, "
    "muestra el paso clave y sé breve. Cuando te pidan revisar una respuesta, di "
    "si es correcta y por qué, en pocas palabras."
)

_FALLBACK = (
    "[tutor sin conexión] No hay ningún motor de LLM local disponible en esta "
    "instalación (el tutor es opcional). Puedes instalar uno con "
    "`pip install puremacro[local-llm]` más un modelo pequeño (por ejemplo vía "
    "Ollama o MLX), o bien usar las indicaciones de la sección «Explora con IA» "
    "de esta lección con cualquier asistente de IA."
)

# Motivos frecuentes por los que el motor no arranca, en español y sin filtrar el
# nombre crudo de la excepción de Python al alumno.
_MOTIVOS = {
    "ImportError": "faltan las dependencias del motor local",
    "ModuleNotFoundError": "faltan las dependencias del motor local",
    "FileNotFoundError": "no se encontró el modelo en disco",
    "OSError": "no se pudo leer el modelo en disco",
    "ConnectionError": "no se pudo contactar al servidor local (¿Ollama apagado?)",
    "TimeoutError": "el motor local tardó demasiado en responder",
}


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
        prompt += f"Contexto de la lección:\n{context}\n\n"
    prompt += f"Pregunta: {question}"
    try:
        return chat(model, prompt, engine=engine, max_tokens=512, temperature=0.2)
    except Exception as exc:  # graceful: any backend/availability error -> fallback
        motivo = _MOTIVOS.get(type(exc).__name__, "el motor local no está disponible")
        return f"{_FALLBACK}\n(motivo: {motivo})"
