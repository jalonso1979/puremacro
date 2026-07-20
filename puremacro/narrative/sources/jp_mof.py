"""Japanese Ministry of Finance (MOF) news feed.

MOF maintains an English news index at https://www.mof.go.jp/english/.
There is no public RSS feed, so this connector reads a precompiled
local CSV (point ``iter_mof_press`` at one) or returns nothing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .local_csv import iter_local_csv


def iter_mof_press(*, csv_path: str | Path | None = None) -> Iterator[tuple]:
    if csv_path is not None:
        yield from iter_local_csv(csv_path)


__all__ = ["iter_mof_press"]
