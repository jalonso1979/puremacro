"""The source registry: shape, loaders, and guards."""
from __future__ import annotations

import importlib.util

import pytest

from puremacro import credentials
from puremacro.fetch import registry


def test_every_non_text_loader_resolves():
    df = registry.sources()
    for sid in df.loc[df["kind"] != "text", "id"]:
        assert callable(registry.load(sid)), sid


def test_text_loaders_point_at_existing_modules():
    df = registry.sources("text")
    assert len(df) >= 50
    for loader in df["loader"]:
        module = loader.split(":")[0]
        assert importlib.util.find_spec(module) is not None, loader


def test_credentials_named_by_entries_exist():
    df = registry.sources()
    named = set(df["credential"].dropna())
    assert named <= set(credentials.SERVICES), named - set(credentials.SERVICES)
    keyed = df[df["auth"].isin(["key", "key_optional"])]
    assert keyed["credential"].notna().all()


def test_micro_sources_are_listed_with_terms():
    micro = registry.sources("micro").set_index("id")
    assert {"census.acs_pums", "census.cps_basic", "fed.scf"} <= set(micro.index)
    assert micro.loc["census.acs_pums", "credential"] == "census"
    assert micro["terms_url"].str.startswith("https://").all()


def test_filters_and_validation():
    assert set(registry.sources(provider="census")["provider"]) == {"census"}
    with pytest.raises(ValueError, match="kind must be"):
        registry.sources("nonsense")
    with pytest.raises(KeyError, match="unknown source"):
        registry.info("no.such.source")
    with pytest.raises(ValueError, match="already registered"):
        registry.register(registry.info("fred.csv"))
    with pytest.raises(ValueError, match="auth"):
        registry.SourceInfo("x", "series", "p", "m:f", auth="maybe")
    with pytest.raises(ValueError, match="module:attribute"):
        registry.SourceInfo("x", "series", "p", "m.f")


def test_register_replace_round_trip():
    orig = registry.info("fred.csv")
    new = registry.SourceInfo(**{**orig.__dict__, "note": "patched"})
    registry.register(new, replace=True)
    try:
        assert registry.info("fred.csv").note == "patched"
    finally:
        registry.register(orig, replace=True)
