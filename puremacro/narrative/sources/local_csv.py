"""Universal escape-hatch source: read a CSV with at minimum
``date``, ``text``, ``source_url`` columns and yield records."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd


def iter_local_csv(
    path: str | Path,
    *,
    date_col: str = "date",
    text_col: str = "text",
    url_col: str = "source_url",
    encoding: str = "utf-8",
) -> Iterator[tuple]:
    """Yield ``(pd.Timestamp, str, str)`` records."""
    df = pd.read_csv(path, encoding=encoding)
    for col in (date_col, text_col):
        if col not in df.columns:
            raise ValueError(f"CSV missing required column {col!r}")
    if url_col not in df.columns:
        df[url_col] = ""
    for _, row in df.iterrows():
        yield pd.Timestamp(row[date_col]), str(row[text_col]), str(row[url_col])


__all__ = ["iter_local_csv"]
