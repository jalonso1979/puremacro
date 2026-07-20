"""Catalog discipline: every Phase-1 entry has the expected shape."""
from __future__ import annotations

import pytest

from puremacro.instruments import list_available, _registry


_EXPECTED_REPLICATION_KEYS = {
    "ramey_2011_defense",
    "romer_romer_2010_fiscal",
    "mertens_ravn_2013_tax",
    "cloyne_2013_uk_tax",
    "romer_romer_2017_fiscal",
    "dglp_2011_consolidations",
}


def test_all_six_replication_entries_registered():
    keys = set(_registry._REGISTRY.keys())
    missing = _EXPECTED_REPLICATION_KEYS - keys
    assert not missing, f"replication entries missing: {missing}"


def test_every_replication_entry_has_non_empty_reference():
    for key in _EXPECTED_REPLICATION_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.reference and len(spec.reference) > 10, (
            f"{key} has empty/short reference"
        )


def test_every_replication_entry_loader_is_callable():
    for key in _EXPECTED_REPLICATION_KEYS:
        spec = _registry._REGISTRY[key]
        assert callable(spec.loader), f"{key} loader is not callable"


def test_every_replication_entry_country_is_iso3_or_none():
    for key in _EXPECTED_REPLICATION_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.country is None or (
            isinstance(spec.country, str) and len(spec.country) == 3
            and spec.country.isupper()
        ), f"{key} country={spec.country!r} not ISO3 or None"


def test_replication_entries_appear_in_list_available_with_include_flag():
    df = list_available(include_unavailable=True, category="narrative_replication")
    for key in _EXPECTED_REPLICATION_KEYS:
        assert key in df["key"].values


# ---------------------------------------------------------------------------
# Connector entries
# ---------------------------------------------------------------------------
_EXPECTED_CONNECTOR_KEYS = {
    "us_treasury_press",
    "us_federal_register",
    "us_dod_contracts",
    "oecd_surveys",
    "imf_articleiv",
    "gdelt_v2_news",
}


def test_all_six_connector_entries_registered():
    keys = set(_registry._REGISTRY.keys())
    missing = _EXPECTED_CONNECTOR_KEYS - keys
    assert not missing, f"connector entries missing: {missing}"


def test_every_connector_entry_has_non_empty_reference():
    for key in _EXPECTED_CONNECTOR_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.reference and len(spec.reference) > 5


def test_every_connector_entry_requires_network_or_fixture():
    for key in _EXPECTED_CONNECTOR_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.requires_network or spec.requires_fixture, (
            f"{key} should require network OR fixture"
        )


# ---------------------------------------------------------------------------
# Monetary HFI + total size
# ---------------------------------------------------------------------------
def test_monetary_hfi_entry_registered():
    assert "gk2015_ffr_surprise" in _registry._REGISTRY
    spec = _registry._REGISTRY["gk2015_ffr_surprise"]
    assert spec.category == "monetary_hfi"


def test_total_catalog_size_at_least_40():
    """6 replications + 6 connectors + 1 monetary HFI + 12 stubs + 4 literature + 7 external + 4 fetch = 40."""
    assert len(_registry._REGISTRY) >= 40


def test_stub_entries_appear_in_include_unavailable_listing():
    df = list_available(include_unavailable=True, category="narrative_connector")
    assert len(df) >= 18  # 6 active + 12 stubs


def test_no_two_entries_share_a_key():
    keys = list(_registry._REGISTRY.keys())
    assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# Fix 1 + Fix 2 + Fix 5 coverage (added in 0.5.0 Cluster 4 code review)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["uk_obr", "de_bmf", "local_csv"])
def test_stub_loaders_raise_not_implemented(key):
    """Stub loaders must raise NotImplementedError when called."""
    with pytest.raises(NotImplementedError, match="discoverability only"):
        _registry._REGISTRY[key].loader()


def test_stub_loader_echoes_kwargs_in_error_message():
    """When called with kwargs, the error must list them so the user
    knows their args were received."""
    with pytest.raises(NotImplementedError, match="my_arg"):
        _registry._REGISTRY["uk_obr"].loader(my_arg="value")


def test_dict_returning_replication_requires_select_country():
    """RR2017 and DGLP loaders return per-country dicts; the wrapper
    must require select_country= at load time and raise a helpful
    error otherwise."""
    from puremacro.instruments._catalog import _wrap_narrative
    # Synthetic dict-returning loader for unit-test isolation (no network).
    from puremacro.narrative import NarrativeEvent, NarrativeInstrument
    import pandas as pd
    def _fake_dict_loader(**_kwargs):
        ev = NarrativeEvent(
            date=pd.Timestamp("2001-01-01"),
            country="USA", magnitude=1.0, magnitude_unit="USD_bn",
            target="both", subtarget=None, sign=1, confidence=1.0,
            source_text="t", source_url="u", scoring_method="manual",
            metadata={},
        )
        return {"USA": NarrativeInstrument.from_events([ev])}
    wrapped = _wrap_narrative(_fake_dict_loader, "test_key", "test source")
    with pytest.raises(ValueError, match="select_country"):
        wrapped()
    # Picks the right country when select_country is passed.
    inst = wrapped(select_country="USA")
    assert inst.metadata.get("registry_key") == "test_key"
    # Errors on missing country.
    with pytest.raises(KeyError, match="GBR"):
        wrapped(select_country="GBR")


def test_cross_country_connector_requires_country_at_load_time():
    """Connector entries registered with country=None require country=
    kwarg at load time; otherwise raise ValueError pointing at user."""
    from puremacro.instruments._catalog import _wrap_connector
    # Synthetic iter_fn for isolation.
    def _fake_iter_fn(**_kwargs):
        return iter([])
    wrapped = _wrap_connector(_fake_iter_fn, registry_key="test_xx",
                              source="test", country=None)
    with pytest.raises(ValueError, match="cross-country source"):
        wrapped()


def test_total_catalog_size_is_exactly_40():
    """6 replications + 6 connectors + 1 monetary HFI + 12 stubs + 4 literature + 7 external + 4 fetch = 40."""
    assert len(_registry._REGISTRY) == 40


_EXPECTED_LITERATURE_KEYS = {
    "bloom_2009_uncertainty",
    "bbd_epu_us",
    "caldara_iacoviello_gpr",
    "rr_2004_monetary",
}


def test_all_four_literature_entries_registered():
    keys = set(_registry._REGISTRY.keys())
    missing = _EXPECTED_LITERATURE_KEYS - keys
    assert not missing, f"literature entries missing: {missing}"


def test_bloom_2009_is_available_by_default():
    """Bloom 2009 events are baked-in (no network, no fixture). It must
    appear in the default list_available() output without flags."""
    df = list_available()
    assert "bloom_2009_uncertainty" in df["key"].values
    spec = _registry._REGISTRY["bloom_2009_uncertainty"]
    assert spec.requires_network is False
    assert spec.requires_fixture is False


def test_three_literature_entries_require_network():
    """BBD EPU, GPR, RR2004 fetch CSVs from canonical URLs."""
    for key in ("bbd_epu_us", "caldara_iacoviello_gpr", "rr_2004_monetary"):
        spec = _registry._REGISTRY[key]
        assert spec.requires_network is True, f"{key} should require network"


def test_every_literature_entry_has_non_empty_reference():
    for key in _EXPECTED_LITERATURE_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.reference and len(spec.reference) > 10


_EXPECTED_EXTERNAL_KEYS = {
    "fred_nfci",
    "fred_vixcls",
    "fred_fedfunds",
    "fred_stlfsi4",
    "bis_credit_to_gdp_gap_us",
    "imf_weo_debt_gdp_usa",
    "imf_weo_primary_balance_gdp_usa",
}


def test_all_seven_external_entries_registered():
    keys = set(_registry._REGISTRY.keys())
    missing = _EXPECTED_EXTERNAL_KEYS - keys
    assert not missing, f"external entries missing: {missing}"


def test_every_external_entry_is_external_csv_category():
    for key in _EXPECTED_EXTERNAL_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.category == "external_csv", (
            f"{key} category={spec.category!r}, expected 'external_csv'"
        )


def test_every_external_entry_requires_network_or_fixture():
    for key in _EXPECTED_EXTERNAL_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.requires_network or spec.requires_fixture


def test_every_external_entry_has_non_empty_reference():
    for key in _EXPECTED_EXTERNAL_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.reference and len(spec.reference) > 10


_EXPECTED_FETCH_KEYS = {
    "fetch_fred_csv",
    "fetch_bis_neer_us",
    "oecd_sdmx_stan_usa_valadd",
    "oecd_sdmx_stan_usa_empn",
}


def test_all_four_fetch_entries_registered():
    keys = set(_registry._REGISTRY.keys())
    missing = _EXPECTED_FETCH_KEYS - keys
    assert not missing, f"fetch entries missing: {missing}"


def test_every_fetch_entry_is_external_csv_category():
    for key in _EXPECTED_FETCH_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.category == "external_csv"


def test_every_fetch_entry_requires_network():
    for key in _EXPECTED_FETCH_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.requires_network is True


def test_every_fetch_entry_has_non_empty_reference():
    for key in _EXPECTED_FETCH_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.reference and len(spec.reference) > 10
