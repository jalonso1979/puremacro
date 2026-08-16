"""Tests for built-in macroeconomic dataset loaders."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from puremacro.datasets import (
    load_gali1999,
    load_narrative_tax_shocks,
    load_macro_quarterly,
    load_macro_monthly,
    load_dice_parameters,
    list_datasets,
)


def test_load_gali1999():
    df = load_gali1999()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "dlprod" in df.columns or len(df.columns) >= 2
    assert len(df) > 100


def test_load_narrative_tax_shocks():
    df = load_narrative_tax_shocks()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) > 100


def test_load_macro_quarterly():
    df = load_macro_quarterly()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "real_gdp" in df.columns
    assert "fed_funds" in df.columns
    assert len(df) > 100


def test_load_macro_monthly():
    df = load_macro_monthly()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "cpi" in df.columns
    assert "unemployment_rate" in df.columns
    assert "fed_funds" in df.columns
    assert len(df) > 100


def test_load_banxico_stance():
    from puremacro.datasets import load_banxico_stance
    df = load_banxico_stance()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "banxico_direction" in df.columns
    assert len(df) > 100


def test_load_dice_parameters():
    params = load_dice_parameters()
    assert isinstance(params, dict)
    assert "capital_elasticity_alpha" in params
    assert params["capital_elasticity_alpha"] == 0.30
    assert "carbon_transition_matrix" in params
    assert params["carbon_transition_matrix"].shape == (3, 3)


def test_list_datasets():
    catalog = list_datasets()
    assert isinstance(catalog, pd.DataFrame)
    assert len(catalog) >= 5
    assert "Dataset Name" in catalog.columns
    assert "Citation" in catalog.columns
