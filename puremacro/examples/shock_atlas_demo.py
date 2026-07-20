"""Shock-atlas registry demo.

Loads every shock the ``puremacro.shock_atlas`` registry knows about,
against the current project root, and prints a summary table: name,
category, n_obs, coverage range, and whether the loader returned data.

Run:
    python -m puremacro.examples.shock_atlas_demo
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import shock_atlas


def run_demo(root: Path | None = None) -> dict:
    """Walk the shock_atlas registry and summarise loader coverage.

    Returns a dict with keys ``loaded`` (dict[str, pd.Series]) and
    ``summary`` (pd.DataFrame with columns name/category/n_obs/start/end/ok).
    """
    if root is None:
        root = Path.cwd()
    try:
        loaded = shock_atlas.load_all_shocks(root)
    except Exception:
        loaded = {}
    rows = []
    for name, series in loaded.items():
        if series is None or len(series) == 0:
            rows.append({"name": name, "category": shock_atlas.categorise(name),
                         "n_obs": 0, "start": None, "end": None, "ok": False})
        else:
            rows.append({
                "name": name,
                "category": shock_atlas.categorise(name),
                "n_obs": len(series),
                "start": str(series.index.min())[:10],
                "end":   str(series.index.max())[:10],
                "ok": True,
            })
    if not rows:
        rows.append({"name": "(none loaded)", "category": "n/a",
                     "n_obs": 0, "start": None, "end": None, "ok": False})
    summary = pd.DataFrame(rows).sort_values(["category", "name"]).reset_index(drop=True)
    return {"loaded": loaded, "summary": summary}


def main() -> None:
    out = run_demo()
    summary = out["summary"]
    print("puremacro shock atlas — registry summary")
    print(f"  Total registered: {len(summary)}  ·  loaded: {summary['ok'].sum()}")
    print()
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                            "display.width", 100):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
