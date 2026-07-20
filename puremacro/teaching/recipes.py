"""Pedagogical wrappers around sparse-panel artifacts.

``load_recipe`` reads ``data/processed/sample_recipes.csv``; the other
two helpers operate directly on the long-form panel DataFrame (no other
files are read). Together they give notebooks a one-line answer to
"which countries qualify for this combination of variables?".
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
_DEFAULT_RECIPES = _DATA_DIR / "sample_recipes.csv"


# Inline recipes — registered here so a panel rebuild is not required to
# pick them up. Each entry is keyed by ``recipe`` name and mirrors the
# columns of ``sample_recipes.csv`` (variables as a comma-separated string).
# These are merged into the CSV-backed recipes by ``load_recipe``; if the
# same name appears in both, the inline definition wins.
_INLINE_RECIPES: dict[str, dict] = {
    "t1_quarterly_core_wide": {
        "recipe": "t1_quarterly_core_wide",
        "variables": (
            "unc_innov_q,log_gdp_real,log_gfcf_real,log_oil_brent_real_local_q"
        ),
        # 31 countries. Energy control (`log_oil_brent_real_local_q`) now uses
        # the implicit GDP deflator instead of log_cpi — see
        # `tools/inject_fx_deflator_oil.py`. log_gdp_real / log_gfcf_real come
        # from `tools/unify_qna_sources.py`. MEX falls just below min_obs=60
        # on unc_innov_q (59 obs); included in WUI universe but not core_wide.
        "n_countries": 31,
        "description": (
            "Core uncertainty roster with energy control (GDP-deflator-based). "
            "31 economies post data unification."
        ),
        "min_obs": 60,
    },
    "t1_quarterly_wui_universe": {
        "recipe": "t1_quarterly_wui_universe",
        "variables": "unc_pc_q,log_gdp_real",
        # 45 countries (was 12 before unification). Post-`unify_qna_sources.py`:
        # log_gdp_real covers ~60 economies; intersection with unc_pc_q (75
        # countries) delivers 45 — matches the post-OECD-QNA-SDMX design goal.
        "n_countries": 45,
        "description": (
            "All countries with quarterly composite uncertainty + real GDP. "
            "45 economies (matches design goal after QNA unification)."
        ),
        "min_obs": 40,
    },
}


def load_recipe(name: str, *, recipes_path: str | Path | None = None) -> dict:
    """Look up *name* in ``sample_recipes.csv`` (or the inline registry).

    Returned keys: ``recipe`` (str), ``variables`` (list[str]),
    ``n_countries`` (int), plus pass-through ``first_obs`` / ``last_obs``
    (or ``description``/``min_obs`` for inline recipes).
    """
    if name in _INLINE_RECIPES:
        row = dict(_INLINE_RECIPES[name])
        row["variables"] = [v.strip() for v in row["variables"].split(",")]
        row["n_countries"] = int(row["n_countries"])
        return row

    p = Path(recipes_path) if recipes_path is not None else _DEFAULT_RECIPES
    if not p.exists():
        raise FileNotFoundError(
            f"sample_recipes.csv not found at {p}. "
            "Re-run build_all (e.g. via 00_data_spine.ipynb or "
            "`python -c 'from puremacro.build_panel import build_all; build_all()'`) "
            "to generate it."
        )
    df = pd.read_csv(p)
    hit = df[df["recipe"] == name]
    if hit.empty:
        raise KeyError(f"recipe {name!r} not found in {p}")
    row = hit.iloc[0].to_dict()
    row["variables"] = [v.strip() for v in row["variables"].split(",")]
    row["n_countries"] = int(row["n_countries"])
    return row


def available_countries(
    variables: list[str],
    *,
    panel: pd.DataFrame | None = None,
    panel_path: str | Path | None = None,
    min_obs: int = 24,
) -> list[str]:
    """Return the sorted list of countries with >= ``min_obs`` rows on every
    variable in ``variables``.

    Either ``panel`` (long-form DataFrame) or ``panel_path`` (parquet path)
    must be supplied. The function intersects the per-variable country
    sets after applying the ``min_obs`` filter to each.
    """
    if panel is None:
        if panel_path is None:
            raise ValueError("supply either panel or panel_path")
        panel = pd.read_parquet(panel_path)
    sets: list[set[str]] = []
    for v in variables:
        sub = panel[panel["variable"] == v]
        counts = sub.groupby("code").size()
        keep = set(counts[counts >= min_obs].index)
        sets.append(keep)
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


def coverage_heatmap(
    variables: list[str],
    *,
    panel: pd.DataFrame | None = None,
    panel_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return a (countries x variables) matrix of obs counts.

    Rows are sorted by total observations descending so the densest
    countries appear at the top of the heatmap.
    """
    if panel is None:
        if panel_path is None:
            raise ValueError("supply either panel or panel_path")
        panel = pd.read_parquet(panel_path)
    sub = panel[panel["variable"].isin(variables)]
    h = (
        sub.groupby(["code", "variable"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=variables, fill_value=0)
    )
    h = h.loc[h.sum(axis=1).sort_values(ascending=False).index]
    return h


__all__ = ["load_recipe", "available_countries", "coverage_heatmap"]
