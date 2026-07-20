"""Scoring backends for narrative-event extraction."""
from .keyword import score_keyword, DEFAULT_FISCAL_LEXICON, regex_billions
from .llm import (
    score_llm, AnthropicBackend, OpenAIBackend,
    LocalBackend, OllamaBackend, MockBackend, get_default_backend,
)
from .manual import load_manual_csv, validate_events

__all__ = [
    "score_keyword", "DEFAULT_FISCAL_LEXICON", "regex_billions",
    "score_llm", "AnthropicBackend", "OpenAIBackend",
    "LocalBackend", "OllamaBackend", "MockBackend", "get_default_backend",
    "load_manual_csv", "validate_events",
]
