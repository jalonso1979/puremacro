"""Loader for vendored replication data snapshots (public source data, frozen).

Snapshots live under ``data/<name>.csv``, shipped as package data. Pyodide-safe
(pandas only); cases read the snapshot offline — the FRED API key is used only by
the dev-only generator in ``tools/``, never at run time.
"""
from __future__ import annotations

from importlib import resources

import pandas as pd


def load_csv(name: str) -> pd.DataFrame:
    res = resources.files("puremacro.replication.data").joinpath(f"{name}.csv")
    with res.open("r", encoding="utf-8") as fh:
        return pd.read_csv(fh)
