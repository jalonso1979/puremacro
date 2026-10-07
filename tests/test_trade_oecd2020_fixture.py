"""The clean OECD 2020 77x11 table bundled in 4.6.0 and the ``source`` switch of ``load_icio_data``.

The legacy fixture ``icio_77c_11s.npz`` is a bit-exact copy of a MATLAB table built
from a corrupted OECD export; ``icio_77c_11s_oecd2020.npz`` is built by
``tools/build_icio_77c_11s.py`` from the clean OECD ICIO 2023-edition 2020 file with
the legacy aggregation's two indexing defects corrected. These tests pin the new
file's digest, its conservation of the native OECD totals, its economic sanity, the
absence of the legacy defect, and the loader's behaviour for both sources.
"""
from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np
import pytest

from puremacro.trade import data as trade_data
from puremacro.trade.data import (
    BUNDLED_ICIO_PROVENANCE,
    CANONICAL_COUNTRY_CODES,
    OECD2020_ICIO_PROVENANCE,
    bundled_icio_path,
    load_icio_data,
)

NC, NS, NFD = 77, 11, 3
N_IND = NC * NS


def _manifest() -> dict:
    return json.loads((bundled_icio_path("oecd2020").parent / "MANIFEST_OECD2020.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def clean() -> np.ndarray:
    return load_icio_data(source="oecd2020")


@pytest.fixture(scope="module")
def legacy() -> np.ndarray:
    return load_icio_data(source="legacy")


def test_clean_fixture_matches_its_manifest(clean):
    m = _manifest()
    assert clean.shape == tuple(m["shape"]) == (850, 1078)
    assert clean.dtype == np.float64
    digest = hashlib.sha256(np.ascontiguousarray(clean).tobytes()).hexdigest()
    assert digest == m["sha256_array"], "icio_77c_11s_oecd2020.npz differs from MANIFEST_OECD2020.json"
    assert m["source_md5"] == OECD2020_ICIO_PROVENANCE["source_export_md5"] == "d3e0f4979d85d6c0bb7cf4c43e324287"
    assert m["source_status"] == "clean" and m["source_year"] == 2020
    assert m["countries_in_file"] == list(CANONICAL_COUNTRY_CODES)


def test_clean_fixture_conserves_the_native_oecd_totals(clean):
    """Aggregation moves cells; it must not create or destroy value added or output."""
    m = _manifest()["source_totals"]
    va = clean[N_IND + 1, :N_IND] + clean[N_IND + 2, :N_IND]
    np.testing.assert_allclose(va.sum(), m["world_value_added"], rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(clean[:N_IND].sum(), m["world_gross_output"], rtol=1e-12, atol=1e-6)
    # The clean world value added is of the order of 8e7 USD million; the legacy table's was 7e11.
    assert 7.5e7 < va.sum() < 8.5e7


def test_clean_fixture_is_economically_well_formed(clean):
    va = clean[N_IND + 1, :N_IND] + clean[N_IND + 2, :N_IND]
    sales = clean[:N_IND].sum(axis=1)
    purchases = clean[:N_IND, :N_IND].sum(axis=0)
    assert (va > 0).all(), "a country-sector with non-positive value added"
    assert (sales > 0).all(), "a country-sector with zero sales"
    assert (clean[:N_IND, :N_IND] >= 0).all(), "a negative intermediate flow"
    # Column balance: sales = purchases + net taxes + labour + capital (net taxes are the residual).
    balance = purchases + clean[N_IND, :N_IND] + va
    np.testing.assert_allclose(balance, sales, rtol=1e-10, atol=1e-6)
    # Value added split 2/3 labour, 1/3 capital.
    np.testing.assert_allclose(clean[N_IND + 2, :N_IND], 0.5 * clean[N_IND + 1, :N_IND], rtol=1e-12)


def test_legacy_row_final_demand_defect_is_absent_in_the_clean_table(clean, legacy):
    """Agregar_11s.m left the rest-of-world final-demand block empty; the clean build fills it."""
    row_rows = slice(76 * NS, 77 * NS)                 # ROW's eleven sectors
    others_fd = slice(N_IND, N_IND + 76 * NFD)          # final-demand columns of the other 76 countries
    assert np.all(legacy[row_rows, others_fd] == 0.0), "the legacy table now has ROW final-demand sales"
    assert clean[row_rows, others_fd].sum() > 0.0
    assert (clean[row_rows, others_fd] >= 0.0).sum() > 0.9 * clean[row_rows, others_fd].size
    # And the TLS/VA cells of ROW's own final-demand columns are populated.
    row_fd = slice(N_IND + 76 * NFD, N_IND + 77 * NFD)
    assert np.all(legacy[N_IND:N_IND + 2, row_fd] == 0.0)
    assert np.any(clean[N_IND, row_fd] != 0.0)


def test_source_switch_and_provenance(legacy):
    assert bundled_icio_path("legacy").name == "icio_77c_11s.npz"
    assert bundled_icio_path("oecd2020").name == "icio_77c_11s_oecd2020.npz"
    with np.load(bundled_icio_path("legacy")) as z:
        assert np.array_equal(legacy, z["data"])
    s_leg = load_icio_data(source="legacy", return_structured=True)
    s_new = load_icio_data(source="oecd2020", return_structured=True)
    assert s_leg.metadata["is_regression_fixture"] is True
    assert s_leg.metadata == BUNDLED_ICIO_PROVENANCE
    assert s_new.metadata["is_regression_fixture"] is False
    assert s_new.metadata == OECD2020_ICIO_PROVENANCE
    assert s_new.country_codes == CANONICAL_COUNTRY_CODES and len(s_new.sector_codes) == 11
    with pytest.raises(ValueError, match="unknown bundled ICIO source"):
        load_icio_data(source="oecd2019")
    with pytest.raises(ValueError, match="not both"):
        load_icio_data(path=bundled_icio_path("oecd2020"), source="oecd2020")
    # An explicit path to the clean file is read as an 11-sector table despite the "2020" in its name.
    assert load_icio_data(path=bundled_icio_path("oecd2020")).shape == (850, 1078)


def test_implicit_default_is_legacy_with_a_single_future_warning(monkeypatch):
    monkeypatch.setattr(trade_data, "_DEFAULT_SOURCE_WARNED", False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        a = load_icio_data()
        b = load_icio_data()
    fw = [w for w in rec if issubclass(w.category, FutureWarning)]
    assert len(fw) == 1 and "source='oecd2020'" in str(fw[0].message)
    assert np.array_equal(a, b) and np.array_equal(a, load_icio_data(source="legacy"))
