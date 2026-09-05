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


def test_var_estimate_result_exports():
    from puremacro.var import fit_var

    rng = np.random.default_rng(42)
    Y = rng.standard_normal((100, 2))
    res = fit_var(Y, p=2)

    md = res.to_markdown(horizon=5)
    assert "|" in md
    assert "response" in md

    ltx = res.to_latex(horizon=5)
    assert "\\begin{tabular}" in ltx

    typ = res.to_typst(horizon=5)
    assert "#table(" in typ


def test_reports_helper_exports():
    from puremacro.reports import (
        irf_to_dataframe,
        irf_to_latex,
        irf_to_markdown,
        irf_to_typst,
        summary_to_latex,
        summary_to_markdown,
        summary_to_typst,
        df_to_latex,
        df_to_markdown,
        df_to_typst,
        IRFResult,
    )

    pt = np.array([[1.0, 0.5], [0.8, 0.3]])
    lo = pt - 0.2
    hi = pt + 0.2

    df = irf_to_dataframe(pt, lo, hi)
    assert "y0" in df.columns
    assert "y1" in df.columns

    md = irf_to_markdown(pt, lo, hi)
    assert "|" in md

    ltx = irf_to_latex(pt, lo, hi)
    assert "\\begin{tabular}" in ltx

    typ = irf_to_typst(pt, lo, hi)
    assert "#table(" in typ

    d = {"n_obs": 100, "r2": 0.85, "name": "test"}
    assert "|" in summary_to_markdown(d)
    assert "\\begin{tabular}" in summary_to_latex(d)
    assert "#table(" in summary_to_typst(d)

    # IRFResult
    irf = IRFResult(point=pt, lower=lo, upper=hi, var_names=("gdp", "cpi"))
    assert "gdp" in irf.to_frame().columns
    assert "\\begin{tabular}" in irf.to_latex()
    assert "#table(" in irf.to_typst()


def test_dsge_and_dynpanel_exports():
    from puremacro.dsge._results import DSGEPosteriorResult
    from puremacro.dynpanel._results import GMMResult

    dsge = DSGEPosteriorResult(
        draws=np.ones((2, 10, 2)),
        param_names=("alpha", "beta"),
        log_posterior_trace=np.zeros((2, 10)),
        accept_rates=(0.3, 0.3),
        mode={"alpha": 0.3, "beta": 0.99},
        mode_hessian_inv=np.eye(2),
        n_burn_in=2,
        data_n_obs=100,
        seed=42,
    )
    assert "|" in dsge.to_markdown()
    assert "\\begin{tabular}" in dsge.to_latex()
    assert "#table(" in dsge.to_typst()

    gmm = GMMResult(
        coefs=np.array([0.5]),
        se=np.array([0.1]),
        cov=np.array([[0.01]]),
        names=("lag_y",),
        hansen_j=1.2,
        hansen_j_p=0.27,
        hansen_j_df=1,
        ar1_p=0.01,
        ar2_p=0.45,
        n_instruments=2,
        n_obs=80,
        n_panels=20,
        step=2,
        windmeijer=True,
        estimator="ab",
        converged=True,
    )
    assert "|" in gmm.to_markdown()
    assert "\\begin{tabular}" in gmm.to_latex()
    assert "#table(" in gmm.to_typst()

