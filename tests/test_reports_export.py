"""Unit tests for publication reporting pipeline (.to_markdown, .to_latex, .to_typst)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.lp import lp_hac
from puremacro.reports import coef_table
from puremacro.var.identify._results import ProxySVARResult


def test_lp_result_table_exports():
    rng = np.random.default_rng(42)
    y = rng.standard_normal(100)
    x = rng.standard_normal(100)

    res = lp_hac(y, x, horizons=range(3), lags=1)

    md = res.to_markdown()
    assert "| h |" in md or "| beta |" in md or "| point |" in md
    assert "\n" in md

    ltx = res.to_latex()
    assert "\\begin{tabular}" in ltx
    assert "\\end{tabular}" in ltx

    typ = res.to_typst()
    assert "#table(" in typ
    assert "columns:" in typ


def test_irf_result_table_exports():
    H = 4
    n = 2
    point = np.zeros((H + 1, n, n))
    point[:, 0, 0] = np.linspace(1.0, 0.2, H + 1)
    lower = point - 0.1
    upper = point + 0.1

    res = ProxySVARResult(
        irf_point=point,
        irf_lower=lower,
        irf_upper=upper,
        B=np.eye(n),
        first_stage_F=25.0,
        n_boot=100,
        ci=0.90,
    )

    df_tidy = res.to_frame()
    assert "h" in df_tidy.columns
    assert "response" in df_tidy.columns
    assert "shock" in df_tidy.columns
    assert "point" in df_tidy.columns
    assert len(df_tidy) == (H + 1) * n * n

    df_pair = res.to_frame(target_idx=0, shock_idx=0)
    assert len(df_pair) == H + 1
    assert "point" in df_pair.columns

    md = res.to_markdown(target_idx=0, shock_idx=0)
    assert "|" in md

    ltx = res.to_latex(target_idx=0, shock_idx=0)
    assert "\\begin{tabular}" in ltx

    typ = res.to_typst(target_idx=0, shock_idx=0)
    assert "#table(" in typ


def test_coef_table_academic_stars():
    beta = np.array([1.5, 0.05, 0.001])
    se = np.array([0.2, 0.02, 0.1])  # z ~ 7.5 (***), z ~ 2.5 (**), z ~ 0.01 (no stars)

    out = coef_table(beta, se, stars=True, fmt="markdown")
    assert "***" in out
    assert "**" in out

    out_typ = coef_table(beta, se, stars=True, fmt="typst")
    assert "#table(" in out_typ
    assert "***" in out_typ
