"""Regression tests for the 3.4.0 ``BVAR_SVForecast`` table exporters.

``to_markdown`` / ``to_latex`` / ``to_typst`` advertised ``**kwargs`` but
forwarded ``index=False`` themselves, so ``digits`` was the only keyword
that worked and ``index=`` failed with a confusing "multiple values"
``TypeError``. The signature is now explicit: a keyword-only ``digits``.
"""
from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import pytest

from puremacro.var.bvar_sv import BVAR_SVForecast

_METHODS = ["to_markdown", "to_latex", "to_typst"]
_COLUMNS = ["horizon", "period", "variable", "median", "mean", "lower", "upper"]


@pytest.fixture
def forecast() -> BVAR_SVForecast:
    rng = np.random.default_rng(0)
    n_draws, horizon, n = 200, 3, 2
    paths = rng.standard_normal((n_draws, horizon, n))
    index = pd.period_range("2025Q1", periods=horizon, freq="Q")
    history = pd.DataFrame(
        rng.standard_normal((10, n)), columns=["gdp", "infl"],
        index=pd.period_range("2022Q3", periods=10, freq="Q"),
    )
    return BVAR_SVForecast(paths=paths, h_paths=paths.copy(), index=index,
                           names=["gdp", "infl"], history=history, ci=0.9)


@pytest.mark.parametrize("method", _METHODS)
def test_exporters_take_only_a_keyword_only_digits(method):
    params = inspect.signature(getattr(BVAR_SVForecast, method)).parameters
    assert list(params) == ["self", "digits"]
    assert params["digits"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["digits"].default is None


@pytest.mark.parametrize("method", _METHODS)
def test_index_keyword_is_rejected_with_a_clear_error(forecast, method):
    for value in (True, False):
        with pytest.raises(TypeError, match="unexpected keyword argument 'index'"):
            getattr(forecast, method)(index=value)


def test_markdown_has_no_index_column_and_honours_digits(forecast):
    out = forecast.to_markdown(digits=2)
    lines = out.splitlines()
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    assert header == _COLUMNS
    n_rows = forecast.horizon * len(forecast.names)
    assert len(lines) == 2 + n_rows
    cells = [c.strip() for line in lines[2:] for c in line.strip("|").split("|")]
    floats = [c for c in cells if re.fullmatch(r"-?\d+\.\d+", c)]
    assert len(floats) == 4 * n_rows
    assert all(len(c.split(".")[1]) == 2 for c in floats)
    assert forecast.to_markdown() != out  # default keeps up to six decimals


def test_latex_and_typst_have_seven_columns_and_no_index(forecast):
    tex = forecast.to_latex(digits=3)
    assert tex.startswith("\\begin{tabular}{lrrrrrr}")
    assert " & ".join(_COLUMNS) in tex
    typ = forecast.to_typst(digits=3)
    assert typ.startswith("#table(\n  columns: 7,")
    assert all(f"[* {c} *]" in typ for c in _COLUMNS)
    assert "[* index *]" not in typ
