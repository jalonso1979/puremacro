"""Presentation contract on the puremacro.inference result objects
(audit M65): ``ARTestResult`` and ``SupTBandResult`` used to expose only
``summary()``; they now implement ``to_frame`` / ``to_markdown`` /
``to_latex`` / ``to_typst`` / ``plot`` through the puremacro.reports helpers
(``LewbelIVResult`` is covered in test_lewbel_iv.py).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from puremacro.inference import anderson_rubin_test, supt_band
from puremacro.inference._results import ARTestResult, SupTBandResult


def _ar_result():
    rng = np.random.default_rng(3)
    T = 300
    z = rng.standard_normal(T)
    v = rng.standard_normal(T)
    x = 0.6 * z + v
    y = 1.5 * x + 0.5 * v + rng.standard_normal(T)
    return anderson_rubin_test(1.5, y, x, z)


def test_ar_test_result_tables_and_plot():
    res = _ar_result()
    assert isinstance(res, ARTestResult)
    frame = res.to_frame()
    assert list(frame.columns) == ["statistic", "value"]
    assert list(frame["statistic"]) == ["F", "p_value", "df_num", "df_den", "residual_ss"]
    md = res.to_markdown()
    assert md.startswith("|") and "p_value" in md
    tex = res.to_latex()
    assert tex.startswith("\\begin{tabular}") and "p\\_value" in tex
    typ = res.to_typst()
    assert typ.startswith("#table(") and "p\\_value" in typ  # underscore is Typst-escaped
    fig = res.plot()
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_ar_test_result_plot_handles_degenerate_result():
    res = ARTestResult(stat=float("nan"), p_value=float("nan"), df_num=1,
                       df_den=0, residual_ss=0.0)
    fig = res.plot()
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_supt_band_result_tables_and_plot():
    H = 6
    res = supt_band(np.linspace(1, 0, H), np.eye(H) * 0.04, alpha=0.10,
                    n_mc=2000, rng=1)
    assert isinstance(res, SupTBandResult)
    frame = res.to_frame()
    assert list(frame.columns) == ["h", "center", "scale", "lower", "upper"]
    assert len(frame) == H
    assert (frame["lower"] <= frame["center"]).all()
    assert (frame["center"] <= frame["upper"]).all()
    assert res.to_markdown().startswith("|")
    assert res.to_latex().startswith("\\begin{tabular}")
    assert res.to_typst().startswith("#table(")
    fig = res.plot(title="band")
    assert isinstance(fig, Figure)
    plt.close(fig)
