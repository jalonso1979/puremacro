"""Tests for the registry primitives. Catalog discipline tests live in
test_catalog.py."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import (
    Instrument, InstrumentSpec,
    list_available, load, describe, register,
)


def _toy_loader() -> Instrument:
    idx = pd.date_range("2000-01-01", periods=10, freq="QS")
    return Instrument(
        series=pd.Series(np.zeros(10), index=idx),
        name="toy", source="toy synthetic",
        category="literature", frequency="Q",
    )


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot the registry before each test, restore after.

    Lets tests register synthetic specs without leaking into other tests.
    """
    from puremacro.instruments import _registry
    # Shallow copy is safe: InstrumentSpec is frozen, so specs cannot
    # be mutated in-place after registration.
    saved = dict(_registry._REGISTRY)
    yield
    _registry._REGISTRY.clear()
    _registry._REGISTRY.update(saved)


def test_instrument_spec_is_frozen():
    spec = InstrumentSpec(
        key="toy", name="Toy", category="literature",
        description="syn", reference="N/A",
        loader=_toy_loader, country=None, frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.key = "other"


def test_register_and_load_round_trip():
    spec = InstrumentSpec(
        key="toy", name="Toy", category="literature",
        description="syn", reference="N/A",
        loader=_toy_loader, country=None, frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    register(spec)
    inst = load("toy")
    assert isinstance(inst, Instrument)
    assert inst.name == "toy"


def test_load_missing_key_raises_keyerror_with_help():
    with pytest.raises(KeyError, match="not found"):
        load("definitely_not_in_registry_xyz")


def test_list_available_returns_dataframe_with_documented_columns():
    spec = InstrumentSpec(
        key="toy", name="Toy", category="literature",
        description="syn", reference="N/A",
        loader=_toy_loader, country="USA", frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    register(spec)
    df = list_available()
    assert isinstance(df, pd.DataFrame)
    expected_cols = {
        "key", "name", "category", "country", "frequency",
        "reference", "available", "requires_network", "requires_fixture",
    }
    assert expected_cols <= set(df.columns)
    assert "toy" in df["key"].values


def test_list_available_filters_by_category():
    spec1 = InstrumentSpec(
        key="lit1", name="Lit1", category="literature",
        description="d", reference="r", loader=_toy_loader,
        country=None, frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    spec2 = InstrumentSpec(
        key="narr1", name="Narr1", category="narrative_replication",
        description="d", reference="r", loader=_toy_loader,
        country=None, frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    register(spec1); register(spec2)
    df = list_available(category="literature")
    assert "lit1" in df["key"].values
    assert "narr1" not in df["key"].values


def test_list_available_filters_by_country():
    spec_usa = InstrumentSpec(
        key="usa1", name="USA1", category="literature",
        description="d", reference="r", loader=_toy_loader,
        country="USA", frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    spec_gbr = InstrumentSpec(
        key="gbr1", name="GBR1", category="literature",
        description="d", reference="r", loader=_toy_loader,
        country="GBR", frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    register(spec_usa); register(spec_gbr)
    df = list_available(country="USA")
    assert "usa1" in df["key"].values
    assert "gbr1" not in df["key"].values


def test_list_available_excludes_unavailable_by_default():
    spec_avail = InstrumentSpec(
        key="avail", name="Avail", category="literature",
        description="d", reference="r", loader=_toy_loader,
        country=None, frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    spec_net = InstrumentSpec(
        key="needs_net", name="N", category="literature",
        description="d", reference="r", loader=_toy_loader,
        country=None, frequency="Q",
        requires_network=True, requires_fixture=False,
    )
    register(spec_avail); register(spec_net)
    default = list_available()
    assert "avail" in default["key"].values
    assert "needs_net" not in default["key"].values
    full = list_available(include_unavailable=True)
    assert "needs_net" in full["key"].values


def test_describe_returns_multiline_with_reference():
    spec = InstrumentSpec(
        key="toy", name="Toy", category="literature",
        description="syn description here",
        reference="Author (2020). Journal X 1(1).",
        loader=_toy_loader, country=None, frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    register(spec)
    s = describe("toy")
    assert "syn description here" in s
    assert "Author (2020)" in s
    assert "\n" in s


def test_list_available_with_no_matches_returns_empty_dataframe_with_columns():
    """Filter that matches nothing must still return a DataFrame with the
    full column set (callers may iterate columns even on empty results)."""
    spec = InstrumentSpec(
        key="lit1", name="Lit1", category="literature",
        description="d", reference="r", loader=_toy_loader,
        country=None, frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    register(spec)
    df = list_available(category="bogus_nonexistent")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    expected_cols = {
        "key", "name", "category", "country", "frequency",
        "reference", "available", "requires_network", "requires_fixture",
    }
    assert expected_cols == set(df.columns)


def test_describe_missing_key_raises_keyerror():
    with pytest.raises(KeyError):
        describe("missing_key_xyz")


def test_register_overwrite_warns():
    """register() must warn when overwriting an existing key."""
    spec1 = InstrumentSpec(
        key="dup", name="First", category="literature",
        description="first", reference="r", loader=_toy_loader,
        country=None, frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    spec2 = InstrumentSpec(
        key="dup", name="Second", category="literature",
        description="second", reference="r", loader=_toy_loader,
        country=None, frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    register(spec1)
    with pytest.warns(UserWarning, match="already exists"):
        register(spec2)
    # Second wins.
    df = list_available()
    assert "dup" in df["key"].values
