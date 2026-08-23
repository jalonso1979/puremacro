"""One producer per variable, at one scale.

`merge_frames` de-duplicates on ``(code, date, variable)`` keeping the first
frame that supplied a key. That is only safe if every producer of a given
variable measures it the same way. It did not hold: `oecd.fetch_qna_expenditure`
emitted ``log_gdp_real`` from ``PRICE_BASE="LR"`` — a chain-linked volume
*index* — while `oecd_qna_expenditure` and the local workbook emit the same name
in XDC millions. Measured on Germany the two sit about 9 log points apart, and
which one a country received depended on append order and on which fetch
happened to return rows that day.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from puremacro.build_panel import merge_frames

_BUILD_PANEL = Path(__file__).resolve().parents[1] / "puremacro" / "build_panel.py"

#: ``module.function`` pairs that must not be producers in ``build_all``,
#: with the reason, so a future reader knows what they would be reintroducing.
_NOT_PRODUCERS = {
    ("oecd", "fetch_qna_expenditure"):
        "emits log_gdp_real/log_gfcf_real as a volume INDEX (PRICE_BASE=LR); "
        "oecd_qna_expenditure and the local workbook emit those names in XDC "
        "millions, ~9 log points away",
}


def _called_attributes(path: Path) -> set[tuple[str, str]]:
    """Every ``mod.func(...)`` call site in a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)):
            out.add((node.func.value.id, node.func.attr))
    return out


def test_no_variable_has_two_producers_on_different_scales():
    called = _called_attributes(_BUILD_PANEL)
    clashes = {f"{m}.{f}": why for (m, f), why in _NOT_PRODUCERS.items()
               if (m, f) in called}
    assert not clashes, (
        "build_panel calls a producer that collides with another on scale: "
        f"{clashes}. merge_frames keeps first, so this silently decides a "
        "series' units by append order.")


def test_the_call_scan_sees_the_real_producers():
    """Guards the guard: if the AST scan stopped finding call sites, the check
    above would pass while inspecting nothing."""
    called = _called_attributes(_BUILD_PANEL)
    assert ("oecd", "fetch_labor_monthly") in called, "the scan found no known producer"
    assert ("oecd_qna_expenditure", "fetch_qna_expenditure") in called


def test_mixing_scales_under_one_name_is_what_the_rule_prevents():
    """Demonstrates the hazard the rule exists for: merge_frames cannot detect
    it, so nothing downstream would flag a 9-log-point step inside one series."""
    def frame(dates, value, source):
        return pd.DataFrame({"code": "DEU", "date": dates,
                             "variable": "log_gdp_real", "value": value,
                             "sa_source": "oecd", "source": source})

    d = pd.period_range("2000Q1", periods=8, freq="Q").to_timestamp()
    index_like = frame(d[:4], np.full(4, 4.6), "LR-index")
    level_like = frame(d[4:], np.full(4, 13.6), "XDC-millions")
    merged = merge_frames([index_like, level_like])

    s = merged.sort_values("date")["value"].to_numpy()
    assert np.abs(np.diff(s)).max() > 5.0, (
        "the fixture should contain a seam; if it does not, this test is vacuous")
    assert len(merged) == 8          # merge_frames is happy to keep both
