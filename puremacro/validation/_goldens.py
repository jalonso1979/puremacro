"""Loader for frozen reference numbers (PACKAGE / PUBLISHED cases).

Goldens are JSON files under ``goldens/<subsystem>.json``, shipped as package
data. A golden key is ``"<subsystem>:<case_id>"``. The top-level ``_meta`` key
in each file is provenance, not a case.
"""
from __future__ import annotations

import functools
import json
from importlib import resources
from typing import Mapping


@functools.lru_cache(maxsize=None)
def _load_file(subsystem: str) -> dict:
    res = resources.files("puremacro.validation.goldens").joinpath(f"{subsystem}.json")
    with res.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_golden(key: str) -> Mapping[str, object]:
    subsystem, _, case = key.partition(":")
    if not case:
        raise ValueError(f"golden key must be 'subsystem:case_id', got {key!r}")
    data = _load_file(subsystem)
    if case not in data:
        raise KeyError(f"golden {key!r} not found in {subsystem}.json")
    return data[case]
