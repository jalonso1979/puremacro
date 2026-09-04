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


def test_did_result_table_exports():
    from puremacro.did._results import CallawaySantannaResult, SyntheticDiDResult

    df_gt = pd.DataFrame([{"g": 2010, "t": 2010, "event_time": 0, "att": 1.2, "se": 0.3, "lo": 0.6, "hi": 1.8}])
    df_es = pd.DataFrame([{"event_time": 0, "att": 1.2, "se": 0.3, "lo": 0.6, "hi": 1.8, "n_cohorts": 1}])
    cs = CallawaySantannaResult(att_gt=df_gt, att_event_study=df_es, att_overall=1.2)

    assert "|" in cs.to_markdown()
    assert "\\begin{tabular}" in cs.to_latex()
    assert "#table(" in cs.to_typst()

    sdid = SyntheticDiDResult(tau=2.5, omega=pd.Series([1.0]), lambda_w=pd.Series([1.0]), se=0.5, lo=1.5, hi=3.5, treatment_time=2015.0)
    assert "|" in sdid.to_markdown()
    assert "\\begin{tabular}" in sdid.to_latex()
    assert "#table(" in sdid.to_typst()

