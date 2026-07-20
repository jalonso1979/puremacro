"""LLM-prob scoring kernel (Cluster B, item B4).

Sends each text window (sentence or paragraph) to an LLM with a prompt
asking "what is the probability this window expresses <category>?" and
returns the average probability per document, in [0, 1].

This is the highest-quality kernel in the package: it handles
qualifiers, irony, discourse structure, paraphrase — anything a
lexicon or MNL misses. But it's also the most expensive: each window
is an API call, and a single corpus pass can cost $5–30.

Design priorities, in order:
  1. **Opt-in**: the entire module imports lazily and only triggers
     when ``llm_prob_kernel(...)`` is actually called.
  2. **Cached**: scores are keyed on
     ``(provider, model, prompt_hash, text_hash)`` in a SQLite store
     under ``~/.cache/puremacro/llm_scores.sqlite``. Repeat runs are
     instant and free.
  3. **Cost-capped**: a per-call budget (max API calls) is enforced
     unless ``PUREMACRO_LLM_BUDGET=spend`` is exported.
  4. **Provider-agnostic**: Anthropic Claude / OpenAI GPT / local
     Ollama all expose a ``score_paragraph(text, category) -> float``
     method via the ``LLMProvider`` ABC.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from puremacro import credentials
from ._kernels import _split_paragraphs, _split_sentences


# ---------------------------------------------------------------------------
# Cache layer (SQLite)
# ---------------------------------------------------------------------------
def _cache_path() -> Path:
    return Path(os.environ.get(
        "PUREMACRO_LLM_CACHE",
        Path.home() / ".cache" / "puremacro" / "llm_scores.sqlite",
    ))


def _ensure_cache_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS llm_scores ("
        " key TEXT PRIMARY KEY,"
        " provider TEXT,"
        " model TEXT,"
        " prompt_hash TEXT,"
        " text_hash TEXT,"
        " score REAL,"
        " created_at TEXT DEFAULT CURRENT_TIMESTAMP"
        ")"
    )


def _cache_key(provider: str, model: str, prompt: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(provider.encode())
    h.update(b"|")
    h.update(model.encode())
    h.update(b"|")
    h.update(hashlib.sha256(prompt.encode()).hexdigest().encode())
    h.update(b"|")
    h.update(hashlib.sha256(text.encode()).hexdigest().encode())
    return h.hexdigest()


def _cache_get(conn: sqlite3.Connection, key: str) -> float | None:
    row = conn.execute(
        "SELECT score FROM llm_scores WHERE key = ?", (key,),
    ).fetchone()
    return float(row[0]) if row else None


def _cache_set(
    conn: sqlite3.Connection, key: str,
    provider: str, model: str, prompt: str, text: str, score: float,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO llm_scores "
        " (key, provider, model, prompt_hash, text_hash, score)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (
            key, provider, model,
            hashlib.sha256(prompt.encode()).hexdigest(),
            hashlib.sha256(text.encode()).hexdigest(),
            float(score),
        ),
    )


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------
class LLMProvider(ABC):
    """Minimal interface every provider must implement."""

    name: str = "abstract"
    model: str = "abstract"

    @abstractmethod
    def score_paragraph(self, text: str, category: str) -> float:
        """Return P(category | text) ∈ [0, 1]."""


class MockProvider(LLMProvider):
    """Deterministic provider for tests and offline development.

    Returns 1.0 if any keyword in ``hot_keywords`` appears in the text,
    else 0.0. Useful for testing the cache + aggregation logic without
    making real API calls.
    """
    name = "mock"
    model = "mock-v0"

    def __init__(self, hot_keywords: list[str] | None = None):
        self.hot_keywords = [k.lower() for k in (hot_keywords or ["uncertain"])]

    def score_paragraph(self, text: str, category: str) -> float:
        return 1.0 if any(k in text.lower() for k in self.hot_keywords) else 0.0


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider. Lazy-imports ``anthropic`` at construction."""
    name = "anthropic"

    def __init__(self, *, model: str = "claude-haiku-4-5-20251001",
                 api_key: str | None = None):
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise ImportError(
                "anthropic SDK not installed. Run `pip install puremacro[llm]` "
                "to enable Anthropic-backed scoring."
            ) from e
        self.model = model
        self._client = anthropic.Anthropic(
            api_key=credentials.require("anthropic", explicit=api_key),
        )

    def score_paragraph(self, text: str, category: str) -> float:
        import anthropic as _anthropic  # already imported at __init__; safe to re-import
        prompt = (
            f"Score how strongly this paragraph expresses '{category}' "
            f"on a scale from 0.0 (not at all) to 1.0 (strongly). "
            f"Reply with ONLY a single decimal number.\n\n"
            f"Paragraph: {text}"
        )
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b for b in resp.content if isinstance(b, _anthropic.types.TextBlock)]
        content = text_blocks[0].text.strip() if text_blocks else "0"
        try:
            return max(0.0, min(1.0, float(content)))
        except ValueError:
            return 0.0


class LocalProvider(LLMProvider):
    """Free local-LLM provider for ``llm_prob_kernel`` (MLX / llama.cpp / HTTP).

    No API key. ``name`` is set to the resolved engine so the SQLite score
    cache partitions per engine. Robustly extracts the first float in the reply.
    """

    def __init__(self, model: str = "qwen2.5-3b-instruct", *,
                 engine: str = "auto", base_url: str = "http://localhost:11434",
                 timeout: float = 120.0):
        from .._local_engines import resolve_engine
        self._engine = resolve_engine(engine, base_url=base_url, timeout=timeout)
        self.name = self._engine.name
        self.model = model

    def score_paragraph(self, text: str, category: str) -> float:
        import re

        from .._local_engines import resolve_model_id
        prompt = (
            f"Score how strongly this paragraph expresses '{category}' "
            f"on a scale from 0.0 (not at all) to 1.0 (strongly). "
            f"Reply with ONLY a single decimal number.\n\nParagraph: {text}"
        )
        model_id = resolve_model_id(self.model, self._engine.name)
        out = self._engine.complete(model_id, prompt, max_tokens=8,
                                    temperature=0.0, json_mode=False)
        m = re.search(r"\d*\.?\d+", out or "")
        if not m:
            return 0.0
        try:
            return max(0.0, min(1.0, float(m.group())))
        except ValueError:
            return 0.0


class OllamaProvider(LocalProvider):
    """Preset: ``LocalProvider(engine="ollama")``."""

    def __init__(self, model: str = "qwen2.5:3b", *,
                 base_url: str = "http://localhost:11434", **kw):
        super().__init__(model=model, engine="ollama", base_url=base_url, **kw)


def get_default_provider(model: str = "qwen2.5-3b-instruct", **kw) -> LLMProvider:
    """Best available local provider, or ``MockProvider`` if no engine present."""
    from .._local_engines import LocalLLMUnavailable
    try:
        p = LocalProvider(model=model, **kw)
        print(f"[puremacro] local LLM provider: engine={p.name}")
        return p
    except LocalLLMUnavailable as e:
        print(f"[puremacro] {e}\n[puremacro] using MockProvider.")
        return MockProvider()


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
def _prompt_for(category: str) -> str:
    return (
        f"Score the strength with which this paragraph expresses "
        f"'{category}' on a 0.0–1.0 scale. Reply with ONLY the decimal."
    )


def llm_prob_kernel(
    records: Iterable[tuple],
    *,
    provider: LLMProvider,
    category: str,
    window: str = "paragraph",
    language: str = "en",
    max_calls: int | None = 5000,
) -> Iterator[tuple]:
    """Yield ``(date, mean_P_category)`` per document.

    Score per window is provided by ``provider.score_paragraph(text, category)``;
    results are cached on disk by (provider, model, prompt_hash, text_hash).
    The mean over all windows in a doc is yielded.

    Parameters
    ----------
    max_calls : safety cap on API calls per kernel invocation. Set to
        None to disable. Default 5000 (≈ $5–30 at Haiku pricing).
        ``PUREMACRO_LLM_BUDGET=spend`` removes the cap entirely.
    """
    if os.environ.get("PUREMACRO_LLM_BUDGET") == "spend":
        max_calls = None
    if window not in ("paragraph", "sentence"):
        raise ValueError(f"window must be 'sentence' or 'paragraph'; got {window!r}")

    splitter = _split_paragraphs if window == "paragraph" else _split_sentences
    prompt = _prompt_for(category)

    cache_path = _cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cache_path))
    _ensure_cache_table(conn)

    n_calls = 0
    try:
        for record in records:
            if len(record) == 4:
                date, text, _, meta = record
                lang = (meta or {}).get("language", language)
            else:
                date, text, _ = record
                lang = language
            windows = splitter(text or "", lang)
            if not windows:
                yield (pd.Timestamp(date), 0.0)
                continue
            scores = []
            for w in windows:
                key = _cache_key(provider.name, provider.model, prompt, w)
                cached = _cache_get(conn, key)
                if cached is not None:
                    scores.append(cached)
                    continue
                if max_calls is not None and n_calls >= max_calls:
                    scores.append(0.0)  # over budget — bail
                    continue
                s = provider.score_paragraph(w, category)
                n_calls += 1
                _cache_set(conn, key, provider.name, provider.model,
                           prompt, w, s)
                scores.append(s)
            conn.commit()
            yield (pd.Timestamp(date), float(sum(scores) / len(scores)))
    finally:
        conn.close()


__all__ = [
    "LLMProvider",
    "MockProvider",
    "AnthropicProvider",
    "LocalProvider",
    "OllamaProvider",
    "get_default_provider",
    "llm_prob_kernel",
]
