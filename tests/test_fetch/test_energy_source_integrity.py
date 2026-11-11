"""Data availability must not silently change a real fetch into synthetic data."""
import json
import warnings

import pytest

from puremacro.fetch import energy_transition as energy


@pytest.fixture
def synthetic_csv(tmp_path, monkeypatch):
    path = tmp_path / "ember_generation_mock.csv"
    path.write_text(
        "ISO 3 code,Year,Electricity source,Generation (TWh),Share of generation (%)\n"
        "USA,2024,Total generation,999999,100\n"
    )
    # Works against the old implementation and guards against its reintroduction.
    monkeypatch.setattr(energy, "_FIXTURE_PATH", path, raising=False)
    return path


def test_failed_download_never_substitutes_checkout_fixture(monkeypatch, synthetic_csv):
    def offline(url, **kwargs):
        raise OSError("test transport unavailable")

    monkeypatch.setattr(energy, "cached_get", offline)
    with pytest.warns(RuntimeWarning, match="Ember"):
        result = energy.fetch_energy_transition(codes=["USA"], refresh=True)
        assert result.empty
    assert list(result.columns) == ["code", "date", "variable", "value", "sa_source", "source"]


@pytest.mark.parametrize("payload", [b"", b"not a CSV", b"country,year,value\nUS,2024,1\n"])
def test_unusable_payload_never_substitutes_fixture(monkeypatch, synthetic_csv, payload):
    def download(url, **kwargs):
        return payload if url == energy._EMBER_URL else b"[{}, []]"

    monkeypatch.setattr(energy, "cached_get", download)
    with pytest.warns(RuntimeWarning, match="Ember"):
        result = energy.fetch_energy_transition(codes=["USA"])
        assert result.empty


def test_partial_country_coverage_preserves_observed_values(monkeypatch, synthetic_csv):
    payload = (
        "ISO 3 code,Year,Electricity source,Generation (TWh),Share of generation (%)\n"
        "FRA,2024,Total generation,543.21,100\n"
    ).encode()

    def download(url, **kwargs):
        return payload if url == energy._EMBER_URL else b"[{}, []]"

    monkeypatch.setattr(energy, "cached_get", download)
    with pytest.warns(RuntimeWarning, match="USA"):
        result = energy.fetch_energy_transition(codes=["USA", "FRA"])
        assert result["code"].tolist() == ["FRA"]
    assert result["value"].tolist() == [543.21]
    assert result["source"].tolist() == ["Ember:ElectricityReview"]


def test_real_wdi_fallback_keeps_its_source_after_ember_failure(monkeypatch, synthetic_csv):
    def download(url, **kwargs):
        if url == energy._EMBER_URL:
            raise OSError("test transport unavailable")
        value = 2000 if "EG.USE.PCAP" in url else 100
        return json.dumps([{"pages": 1, "total": 1}, [{
            "countryiso3code": "USA", "date": "2024", "value": value,
        }]]).encode()

    monkeypatch.setattr(energy, "cached_get", download)
    with pytest.warns(RuntimeWarning, match="Ember"):
        result = energy.fetch_energy_transition(codes=["USA"])
        assert result["value"].tolist() == [200.0]  # kg/capita * people / kg per tonne
    assert result["variable"].tolist() == ["primary_energy_cons_toe_a"]
    assert result["source"].tolist() == ["WorldBank:WDI:EG.USE.PCAP.KG.OE"]


def test_offline_fixture_requires_explicit_selection(monkeypatch, synthetic_csv):
    def no_network(url, **kwargs):
        pytest.fail("explicit CSV must not download data")

    monkeypatch.setattr(energy, "cached_get", no_network)
    result = energy.fetch_energy_transition(codes=["USA"], csv_path=synthetic_csv)
    assert result["value"].tolist() == [999999.0]
    assert result["code"].tolist() == ["USA"]


def test_complete_download_has_no_availability_warning(monkeypatch, synthetic_csv):
    payload = synthetic_csv.read_bytes()

    def download(url, **kwargs):
        return payload if url == energy._EMBER_URL else b"[{}, []]"

    monkeypatch.setattr(energy, "cached_get", download)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = energy.fetch_energy_transition(codes=["USA"])
    assert result["value"].tolist() == [999999.0]
    assert not caught
