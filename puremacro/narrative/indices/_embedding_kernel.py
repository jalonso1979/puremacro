"""Embeddings-based scoring kernels (Cluster B, item B3).

A document's score is the cosine similarity between its embedding and a
"seed prototype" — the average embedding of a small set of representative
sentences for the index. Higher cosine = more semantically similar to
the prototype.

Key advantages over the lexicon-based kernels:
  - Paraphrase-aware: "the labor market is weakening" matches even if
    none of the LUI lexicon terms appear.
  - Multilingual native (with a multilingual model): the same prototype
    works across en/es/pt/de/fr/it/ja/zh.
  - Continuous score in [-1, +1] (cosine), not a 0/1 hit count.

Design:
  - Pure numpy at the core (cosine + aggregation).
  - The ``embed_fn`` is plug-in: caller passes any callable mapping
    ``str -> np.ndarray`` (or ``list[str] -> np.ndarray`` for batch).
  - The convenience factory ``make_sentence_transformer_embedder``
    lazy-imports ``sentence_transformers`` — keep the indices layer
    pyodide-clean by default; the heavy ML deps land only when the
    user actually opts into embeddings via ``pip install puremacro[embeddings]``.

Default model recommendation: ``paraphrase-multilingual-MiniLM-L12-v2``
(~120MB, CPU-friendly, 50 languages).
"""
from __future__ import annotations

from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


EmbedFn = Callable[[Sequence[str]], np.ndarray]
"""Signature: takes a list of strings, returns an (N, D) float matrix."""


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. NaN-safe (zero vectors
    return 0.0)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_seed_prototype(
    seeds: Sequence[str],
    *,
    embed_fn: EmbedFn,
) -> np.ndarray:
    """Embed each seed sentence, return the average vector.

    Empty / whitespace-only seeds are stripped. If all seeds are empty,
    raises ``ValueError``.
    """
    cleaned = [s for s in seeds if s and s.strip()]
    if not cleaned:
        raise ValueError("seed_prototype is empty after stripping whitespace")
    matrix = np.asarray(embed_fn(cleaned), dtype=float)
    if matrix.ndim != 2:
        raise ValueError(
            f"embed_fn must return a 2-D matrix (N, D); got shape {matrix.shape}"
        )
    return matrix.mean(axis=0)


def embedding_similarity_kernel(
    records: Iterable[tuple],
    *,
    seed_prototype: np.ndarray | Sequence[str],
    embed_fn: EmbedFn,
    language: str = "en",
    batch_size: int = 16,
) -> Iterator[tuple]:
    """Yield ``(date, cosine_score)`` per document.

    If ``seed_prototype`` is a sequence of strings, it's embedded once
    upfront and averaged. If it's already a 1-D numpy array, it's used
    directly.

    Docs are batched to ``batch_size`` to amortise embedding overhead.
    Empty docs yield 0.0.
    """
    if not isinstance(seed_prototype, np.ndarray):
        proto = build_seed_prototype(list(seed_prototype), embed_fn=embed_fn)
    else:
        proto = np.asarray(seed_prototype, dtype=float)
        if proto.ndim != 1:
            raise ValueError(
                f"numpy seed_prototype must be 1-D; got shape {proto.shape}"
            )

    buffer: list[tuple] = []
    for rec in records:
        if len(rec) == 4:
            date, text, _, _meta = rec
        else:
            date, text, _ = rec
        buffer.append((pd.Timestamp(date), text or ""))
        if len(buffer) >= batch_size:
            yield from _flush(buffer, proto, embed_fn)
            buffer = []
    if buffer:
        yield from _flush(buffer, proto, embed_fn)


def _flush(
    buffer: list[tuple],
    proto: np.ndarray,
    embed_fn: EmbedFn,
) -> Iterator[tuple]:
    """Embed a batch of docs, yield (date, cosine) per doc."""
    dates = [d for d, _ in buffer]
    texts = [t for _, t in buffer]
    # Replace empty docs with a placeholder so embed_fn doesn't crash;
    # we'll override their score to 0.0 below.
    safe_texts = [t if t.strip() else " " for t in texts]
    matrix = np.asarray(embed_fn(safe_texts), dtype=float)
    for i, (date, text) in enumerate(zip(dates, texts)):
        if not text.strip():
            yield (date, 0.0)
        else:
            yield (date, _cosine(matrix[i], proto))


# ---------------------------------------------------------------------------
# Sentence-transformers convenience factory (opt-in, lazy-imported)
# ---------------------------------------------------------------------------
def make_sentence_transformer_embedder(
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    *,
    device: str | None = None,
) -> EmbedFn:
    """Return an ``EmbedFn`` backed by sentence-transformers.

    The sentence-transformers / torch stack is heavy (~1GB on disk) and
    is NOT a default puremacro dependency. Install via::

        pip install puremacro[embeddings]
        # or, equivalently:
        pip install sentence-transformers

    Raises ``ImportError`` if the optional stack isn't installed.

    Default model: ``paraphrase-multilingual-MiniLM-L12-v2`` (50
    languages, ~120MB download on first call).
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is not installed. Install with "
            "`pip install puremacro[embeddings]` to enable the default "
            "embedding-similarity kernel."
        ) from e

    model = SentenceTransformer(model_name, device=device)

    def _embed(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False),
            dtype=float,
        )

    return _embed


__all__ = [
    "EmbedFn",
    "embedding_similarity_kernel",
    "build_seed_prototype",
    "make_sentence_transformer_embedder",
]
