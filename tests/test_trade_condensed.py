"""Tests of puremacro.trade.condensed: the condensed one-factor Leontief tariff model.

Fixtures: a seeded synthetic 3-country x 4-sector table (random flows with the
column residual equal to the drawn TLS), the bundled 3-region x 3-sector OECD
2019 fixture (tests/fixtures/trade/oecd_2023) and the bundled 77x11 table
bridged through ``BalancedIOTable.from_trade_calibration``. Tolerances are
stated in every assertion. Parity tests import the vendored IO engine from
/Volumes/BIGDATA/Research/IO in a subprocess and skip when the volume is
absent.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from puremacro.trade import calibrate_trade_model, load_icio_data
from puremacro.trade.condensed import (
    BalancedIOTable,
    CalibrationError,
    CertificationFailure,
    CondensedCalibration,
    CondensedEquilibriumResult,
    CondensedLeontiefModel,
    CondensedMeasuresResult,
    CondensedSolveError,
    ContinuationFailure,
    DataIntegrityError,
    EquilibriumNotFound,
    ExistenceViolation,
    FoldDetected,
    MultipleEquilibria,
    ProductivityUncertified,
    RawFlowCertificate,
    TableBuildReport,
    TariffWedges,
    UnsupportedExtension,
    aggregation_gaps,
    arclength,
    build_tariff_wedges,
    calibrate_condensed,
    certify_productivity,
    certify_raw_flows,
    collatz_wielandt_bound,
    compute_measures,
    constant_price_rgdp,
    consumer_price_relative_to_wages,
    continuation,
    ev_split,
    existence_gate,
    multistart_near,
    nested_fallback,
    newton,
    pe_first_order_decomposition,
    pe_unit_cost_bound,
    real_national_income,
    scale_wedges,
    sector_incidence,
    separate_baseline_tariffs,
    solve_condensed,
    table_from_arrays,
    tariff_revenue_and_effective_rate,
    tot_fisher,
)
from puremacro.trade.condensed import solve as solve_mod
from puremacro.trade.condensed.table import infer_merchandise_mask
from puremacro.trade.data import RawIOData, generate_synthetic_mrio
from puremacro.trade.scenarios import TariffScenario

# The IO research workspace; parity tests skip when it is absent. Override with PUREMACRO_IO_ROOT.
IO_ROOT = Path(os.environ.get("PUREMACRO_IO_ROOT", "/Volumes/BIGDATA/Research/IO"))
VENDOR = IO_ROOT / "headlinePaper" / "rebuild" / "vendor"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "trade" / "oecd_2023" / "2019_3regions_3sectors.npz"
OECD6 = ("HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR")
CODES3 = ("USA", "CHN", "ROW")
SECTORS4 = ("A01_02", "C24", "G", "K")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def synthetic_arrays(seed: int = 7, scale: float = 1.0) -> dict:
    """Seeded 3x4 economy: random nonnegative flows, VA chosen so the column residual equals the drawn TLS."""
    rng = np.random.default_rng(seed)
    n, s = 3, 4
    m = n * s
    Z = rng.uniform(0.0, 1.0, size=(m, m)) * 4.0
    for k in range(n):
        Z[k * s:(k + 1) * s, k * s:(k + 1) * s] *= 4.0
    F6 = rng.uniform(0.0, 1.0, size=(m, n, 6)) * 3.0
    F6[:, :, 4] *= 0.1
    F6[:, :, 5] *= 0.1
    rng.uniform(5.0, 10.0, size=m)  # the IO assessment fixture drew (and discarded) a VA vector here
    TLS = rng.uniform(-0.2, 0.4, size=m)
    TFD6 = rng.uniform(0.0, 0.3, size=(n, 6))
    out = Z.sum(axis=1) + F6.sum(axis=(1, 2))
    VA = out - Z.sum(axis=0) - TLS
    Z, F6, VA, TLS, TFD6, out = [x * scale for x in (Z, F6, VA, TLS, TFD6, out)]
    return {"Z": Z, "F": F6, "VA": VA, "TLS": TLS, "TFD": TFD6, "output": out}


def synthetic_table(seed: int = 7, scale: float = 1.0) -> BalancedIOTable:
    arr = synthetic_arrays(seed, scale)
    table, _ = table_from_arrays(arr["Z"], arr["F"], arr["VA"], arr["TLS"], arr["TFD"], country_codes=CODES3,
                                 sector_codes=SECTORS4, fd_codes=OECD6, output=arr["output"], source="synthetic")
    return table


def oecd3_arrays() -> dict:
    with np.load(FIXTURE, allow_pickle=False) as r:
        return {"Z": r["Z"], "F": r["F"].reshape(9, 3, 6), "VA": r["VA"], "TLS": r["TLS"],
                "TFD": r["TLS_fd"].reshape(3, 6), "output": r["Y"],
                "countries": tuple(str(c) for c in r["countries"]), "sectors": tuple(str(x) for x in r["sectors"])}


@pytest.fixture(scope="module")
def synth_table() -> BalancedIOTable:
    return synthetic_table()


@pytest.fixture(scope="module")
def synth_calib(synth_table) -> CondensedCalibration:
    return calibrate_condensed(synth_table)


@pytest.fixture(scope="module")
def oecd3_table() -> BalancedIOTable:
    a = oecd3_arrays()
    table, _ = table_from_arrays(a["Z"], a["F"], a["VA"], a["TLS"], a["TFD"], country_codes=a["countries"],
                                 sector_codes=a["sectors"], fd_codes=OECD6, output=a["output"],
                                 reference_country="REST", source="oecd_2023 fixture", year=2019)
    return table


@pytest.fixture(scope="module")
def oecd3_calib(oecd3_table) -> CondensedCalibration:
    return calibrate_condensed(oecd3_table)


@pytest.fixture(scope="module")
def legacy_calib():
    return calibrate_trade_model(load_icio_data())


@pytest.fixture(scope="module")
def bundled_calib(legacy_calib) -> CondensedCalibration:
    table = BalancedIOTable.from_trade_calibration(legacy_calib, negative_investment="to_inventory")
    return calibrate_condensed(table, allow_empty_purchases_abroad=True)


def us10(calib: CondensedCalibration) -> TariffWedges:
    return build_tariff_wedges(calib, 0.10, importer="USA")


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------

def test_six_and_four_category_layouts_give_the_same_table():
    a = synthetic_arrays()
    t6, r6 = table_from_arrays(a["Z"], a["F"], a["VA"], a["TLS"], a["TFD"], country_codes=CODES3,
                               sector_codes=SECTORS4, fd_codes=OECD6, output=a["output"])
    F4 = np.stack([a["F"][:, :, :3].sum(2), a["F"][:, :, 3], a["F"][:, :, 5], a["F"][:, :, 4]], axis=2)
    T4 = np.stack([a["TFD"][:, :3].sum(1), a["TFD"][:, 3], a["TFD"][:, 5], a["TFD"][:, 4]], axis=1)
    t4, _ = table_from_arrays(a["Z"], F4, a["VA"], a["TLS"], T4, country_codes=CODES3, sector_codes=SECTORS4,
                              fd_codes=("C", "G", "X", "V"), output=a["output"])
    # Permuted four-category order maps by name.
    tp, _ = table_from_arrays(a["Z"], F4[:, :, [3, 0, 2, 1]], a["VA"], a["TLS"], T4[:, [3, 0, 2, 1]],
                              country_codes=CODES3, sector_codes=SECTORS4, fd_codes=("V", "C", "X", "G"))
    for other in (t4, tp):
        for name in ("Z", "F", "VA", "TLS", "TFD"):
            np.testing.assert_allclose(getattr(other, name), getattr(t6, name), rtol=0, atol=1e-13)
    assert t6.fd_codes == ("C", "G", "X", "V")
    assert r6.fd_codes_in == OECD6 and r6.n_zero_output == 0 and not r6.valuables_merged
    assert t6.reference_country == "ROW" and t6.reference == 2
    assert t6.merchandise_mask.tolist() == [True, True, False, False]
    assert abs(t6.output - a["output"]).max() < 1e-12
    assert abs(t6.net_exports().sum()) < 1e-12
    for arr in (t6.Z, t6.F, t6.VA, t6.TLS, t6.TFD, t6.active, t6.merchandise_mask):
        assert not arr.flags.writeable


def test_valuables_are_merged_into_inventories():
    a = synthetic_arrays()
    F4 = np.stack([a["F"][:, :, :3].sum(2), a["F"][:, :, 3], a["F"][:, :, 5], a["F"][:, :, 4]], axis=2)
    T4 = np.stack([a["TFD"][:, :3].sum(1), a["TFD"][:, 3], a["TFD"][:, 5], a["TFD"][:, 4]], axis=1)
    val = 0.25 * F4[:, :, 3]
    F5 = np.concatenate([F4[:, :, :3], F4[:, :, 3:4] - val[:, :, None], val[:, :, None]], axis=2)
    T5 = np.concatenate([T4[:, :3], 0.5 * T4[:, 3:4], 0.5 * T4[:, 3:4]], axis=1)
    t5, r5 = table_from_arrays(a["Z"], F5, a["VA"], a["TLS"], T5, country_codes=CODES3, sector_codes=SECTORS4,
                               fd_codes=("C", "G", "X", "V", "VAL"))
    np.testing.assert_allclose(t5.F[:, :, 3], F4[:, :, 3], atol=1e-13)
    np.testing.assert_allclose(t5.TFD, T4, atol=1e-13)
    assert r5.valuables_merged is True and t5.report.valuables_merged is True


def test_three_category_layout_needs_a_negative_investment_policy():
    a = synthetic_arrays()
    F3 = np.stack([a["F"][:, :, :3].sum(2), a["F"][:, :, 3] + a["F"][:, :, 4], a["F"][:, :, 5]], axis=2)
    T3 = np.stack([a["TFD"][:, :3].sum(1), a["TFD"][:, 3] + a["TFD"][:, 4], a["TFD"][:, 5]], axis=1)
    F3[2, 1, 1] = -0.7  # a negative merged investment+inventory cell (CHN buys USA_C24)
    VA = a["Z"].sum(1) + F3.sum((1, 2)) - a["Z"].sum(0) - a["TLS"]  # keep the accounts balanced
    with pytest.raises(DataIntegrityError, match="negative investment"):
        table_from_arrays(a["Z"], F3, VA, a["TLS"], T3, country_codes=CODES3, sector_codes=SECTORS4,
                          fd_codes=("C", "G", "X"))
    with pytest.raises(UnsupportedExtension):
        table_from_arrays(a["Z"], F3, VA, a["TLS"], T3, country_codes=CODES3, sector_codes=SECTORS4,
                          fd_codes=("C", "G", "X"), negative_investment="clip")
    t3, r3 = table_from_arrays(a["Z"], F3, VA, a["TLS"], T3, country_codes=CODES3, sector_codes=SECTORS4,
                               fd_codes=("C", "I", "Cx"), negative_investment="to_inventory")
    assert r3.negative_investment == "to_inventory"
    assert r3.negative_investment_cells == ("USA_G->CHN",)
    assert t3.F[2, 1, 1] == 0.0 and t3.F[2, 1, 3] == -0.7
    assert np.all(t3.F[:, :, 1] >= 0)
    assert r3.negative_fd_cells == {"C": 0, "G": 0, "X": 0, "V": 1}
    calib = calibrate_condensed(t3)
    assert calib.qV[2, 1] == -0.7


def test_table_rejects_inconsistent_input():
    a = synthetic_arrays()
    kw = dict(country_codes=CODES3, sector_codes=SECTORS4, fd_codes=OECD6)
    with pytest.raises(DataIntegrityError, match="shapes"):
        table_from_arrays(a["Z"][:, :5], a["F"], a["VA"], a["TLS"], a["TFD"], **kw)
    bad = a["Z"].copy()
    bad[0, 0] = np.nan
    with pytest.raises(DataIntegrityError, match="NaN"):
        table_from_arrays(bad, a["F"], a["VA"], a["TLS"], a["TFD"], **kw)
    bad = a["Z"].copy()
    bad[0, 1] = -1.0
    with pytest.raises(DataIntegrityError, match="negative intermediate"):
        table_from_arrays(bad, a["F"], a["VA"], a["TLS"], a["TFD"], **kw)
    with pytest.raises(DataIntegrityError, match="unknown final-demand code"):
        table_from_arrays(a["Z"], a["F"], a["VA"], a["TLS"], a["TFD"], country_codes=CODES3, sector_codes=SECTORS4,
                          fd_codes=("HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "OTHER"))
    with pytest.raises(UnsupportedExtension, match="unknown sector codes"):
        table_from_arrays(a["Z"], a["F"], a["VA"], a["TLS"], a["TFD"], country_codes=CODES3,
                          sector_codes=("X1", "X2", "X3", "X4"), fd_codes=OECD6)
    ok, _ = table_from_arrays(a["Z"], a["F"], a["VA"], a["TLS"], a["TFD"], country_codes=CODES3,
                              sector_codes=("X1", "X2", "X3", "X4"), fd_codes=OECD6,
                              merchandise_mask=np.array([True, False, False, True]))
    assert ok.merchandise_mask.tolist() == [True, False, False, True]
    with pytest.raises(DataIntegrityError, match="reference country"):
        table_from_arrays(a["Z"], a["F"], a["VA"], a["TLS"], a["TFD"], reference_country="XXX", **kw)
    # A zero-output cell with entries is junk; a genuinely empty one gets a phantom.
    Z = a["Z"].copy()
    F = a["F"].copy()
    VA = a["VA"].copy()
    TLS = a["TLS"].copy()
    Z[5, :] = 0.0
    F[5] = 0.0
    with pytest.raises(DataIntegrityError, match="zero-output cell"):
        table_from_arrays(Z, F, VA, TLS, a["TFD"], **kw)
    Z[:, 5] = 0.0
    VA[5] = 0.0
    TLS[5] = 0.0
    with pytest.raises(DataIntegrityError, match="median relative row identity"):
        table_from_arrays(Z, F, VA, TLS, a["TFD"], output=np.full(12, 100.0), **kw)


def test_phantoms_floors_and_residual_tls_are_recorded():
    a = synthetic_arrays()
    Z, F, TLS = a["Z"].copy(), a["F"].copy(), a["TLS"].copy()
    Z[5, :] = 0.0  # CHN_C24 is empty
    Z[:, 5] = 0.0
    F[5] = 0.0
    TLS[5] = 0.0
    out = Z.sum(1) + F.sum((1, 2))
    VA = out - Z.sum(0) - TLS  # balanced columns with the drawn TLS
    VA[5] = 0.0
    VA[2] = -0.5  # active cell with non-positive value added: floored, offset in TLS
    TLS[2] = out[2] - Z.sum(0)[2] + 0.5
    table, report = table_from_arrays(Z, F, VA, TLS, a["TFD"], country_codes=CODES3, sector_codes=SECTORS4,
                                      fd_codes=OECD6)
    assert report.phantoms == ("CHN_C24",) and report.n_zero_output == 1
    assert table.active.tolist() == [i != 5 for i in range(12)]
    assert table.F[5, 1, 0] == 1e-6 and table.VA[5] == 1e-6
    assert len(report.floored) == 1 and report.floored[0][0] == "USA_G"
    assert table.VA[2] == pytest.approx(1e-3 * table.output[2])
    assert report.dtls_floor_abs > 0.5 and report.dtls_accounting_abs <= 1e-12
    # Column identity holds exactly after the residual closure.
    np.testing.assert_allclose(table.Z.sum(0) + table.TLS + table.VA, table.Z.sum(1) + table.F.sum((1, 2)),
                               rtol=1e-13)
    calib = calibrate_condensed(table)
    assert not calib.active[5] and calib.gates["b_min_active"] > 0
    frame = report.to_dataframe()
    assert "phantom cells" in frame.index
    assert "phantom" in report.to_markdown() and "\\begin{tabular}" in report.to_latex() and "#table" in report.to_typst()
    assert report.to_dict()["phantoms"] == ["CHN_C24"]


def test_from_io_table_accepts_the_shared_contract(synth_table):
    obj = SimpleNamespace(Z=csr_matrix(synth_table.Z), F=synth_table.F, fd_codes=("C", "G", "X", "V"),
                          VA=synth_table.VA, TLS=synth_table.TLS, TFD=synth_table.TFD, output=synth_table.output,
                          country_codes=CODES3, sector_codes=SECTORS4, merchandise_mask=synth_table.merchandise_mask,
                          metadata={"source": "duck"})
    table = BalancedIOTable.from_io_table(obj)
    for name in ("Z", "F", "VA", "TLS", "TFD"):
        np.testing.assert_allclose(getattr(table, name), getattr(synth_table, name), rtol=0, atol=1e-12)
    assert table.report.source == "duck" and table.metadata["source"] == "duck"


def test_from_raw_matches_the_direct_fixture_build(oecd3_table):
    a = oecd3_arrays()
    raw = RawIOData(countries=list(a["countries"]), sectors=list(a["sectors"]), intermediate_matrix=a["Z"],
                    final_demand_matrix=a["F"].reshape(9, 18), value_added=a["VA"], taxes_less_subsidies=a["TLS"],
                    gross_output=a["output"], fd_categories=list(OECD6), taxes_less_subsidies_fd=a["TFD"].ravel(),
                    year=2019, source="OECD_ICIO_2023_regular_aggregated", unit="M_USD")
    table = BalancedIOTable.from_raw(raw, reference_country="REST")
    for name in ("Z", "F", "VA", "TLS", "TFD"):
        np.testing.assert_allclose(getattr(table, name), getattr(oecd3_table, name), rtol=0, atol=1e-9)
    assert table.report.year == 2019 and table.report.unit == "M_USD"
    assert table.merchandise_mask.tolist() == [True, True, False]
    assert oecd3_table.report.negative_fd_cells == {"C": 0, "G": 0, "X": 0, "V": 0}
    assert oecd3_table.report.dtls_accounting_abs == pytest.approx(8.362, abs=0.01)


_LOADER_FD_MAP = {"P3_S14": "C", "P3_S15": "C", "P3_S13": "C", "P51G": "G", "P5M": "V",
                  "HFCE": "C", "NPISH": "C", "GGFC": "C", "GFCF": "G", "INVNT": "V", "VALUABLES": "V",
                  "ACQ_VAL": "V", "DPABR": "X", "CONS_h": "C", "CONS_np": "C", "CONS_g": "C", "INVT": "V"}


@pytest.mark.parametrize("dataset", ["figaro", "exiobase", "wiod", "eora", "oecd"])
def test_from_raw_accepts_the_loader_vocabularies(dataset):
    """RawIOData in every puremacro.trade.data loader layout condenses to (C, G, X, V); layouts without a
    purchases-abroad column calibrate only with allow_empty_purchases_abroad=True."""
    raw = generate_synthetic_mrio(dataset, custom_c=3, custom_s=3)
    M, C, K = raw.M, raw.C, raw.K_F
    codes = list(raw.fd_categories)
    F = np.asarray(raw.final_demand_matrix, dtype=float).reshape(M, C, K)
    mask = None if dataset == "oecd" else np.array([True, True, True])
    if dataset == "exiobase":
        # A closed table has no sales outside the modelled world: the synthetic EXPORT column is refused.
        with pytest.raises(DataIntegrityError, match="identically zero"):
            BalancedIOTable.from_raw(raw, merchandise_mask=mask)
        F[:, :, codes.index("EXPORT")] = 0.0
        Z = np.asarray(raw.intermediate_matrix, dtype=float)
        out = Z.sum(1) + F.sum((1, 2))
        va = out - Z.sum(0) - np.asarray(raw.taxes_less_subsidies, dtype=float)
        raw = dataclasses.replace(raw, final_demand_matrix=F.reshape(M, C * K), gross_output=out, value_added=va)
    table = BalancedIOTable.from_raw(raw, merchandise_mask=mask)
    assert table.fd_codes == ("C", "G", "X", "V") and table.report.fd_codes_in == tuple(codes)
    for k, cat in enumerate(("C", "G", "X", "V")):
        idx = [j for j, code in enumerate(codes) if _LOADER_FD_MAP.get(code) == cat]
        expected = F[:, :, idx].sum(2) if idx else np.zeros((M, C))
        np.testing.assert_allclose(table.F[:, :, k], expected, rtol=0, atol=1e-12)
    assert table.report.valuables_merged == (dataset in ("exiobase", "eora"))
    if dataset == "oecd":
        assert table.merchandise_mask.tolist() == [True, True, True]  # A01_02, A03, B05_06 are inferred
        calib = calibrate_condensed(table)
    else:
        assert np.all(table.F[:, :, 2] == 0.0)
        with pytest.raises(CalibrationError, match="allow_empty_purchases_abroad"):
            calibrate_condensed(table)
        calib = calibrate_condensed(table, allow_empty_purchases_abroad=True)
    w = build_tariff_wedges(calib, 0.10, importer=str(raw.countries[0]))
    res = solve_condensed(calib, w)
    assert res.passed and res.certificate.max_block <= 1e-10
    with pytest.raises(DataIntegrityError, match="must contain consumption"):
        table_from_arrays(table.Z, table.F[:, :, [2, 3]], table.VA, table.TLS, table.TFD[:, [2, 3]],
                          country_codes=table.country_codes, sector_codes=table.sector_codes, fd_codes=("X", "V"))


def test_from_trade_calibration_bridges_the_bundled_table(legacy_calib):
    with pytest.raises(DataIntegrityError, match="3 negative investment"):
        BalancedIOTable.from_trade_calibration(legacy_calib)
    table = BalancedIOTable.from_trade_calibration(legacy_calib, negative_investment="to_inventory")
    assert table.n_countries == 77 and table.n_sectors == 11 and table.reference_country == "ROW"
    assert table.report.negative_investment_cells == ("LTU_MINQ->LTU", "UKR_MINQ->UKR", "VNM_MANU->VNM")
    assert table.report.negative_fd_cells["V"] == 3 and table.report.n_zero_output == 0
    assert table.metadata["investment_includes_inventories"] is True
    assert table.merchandise_mask.tolist() == [c in ("AGRI", "MINQ", "MANU") for c in table.sector_codes]
    # Value added is labour plus capital; production taxes are the TLS row.
    M = 847
    np.testing.assert_allclose(table.VA, legacy_calib.data_calibra[M + 1, :M] + legacy_calib.data_calibra[M + 2, :M])
    assert table.to_dataframe().shape == (77, 5)


@pytest.mark.parametrize("loader", ["load_figaro", "load_wiod", "load_eora", "load_exiobase"])
def test_from_trade_calibration_never_labels_provider_inventories_as_purchases_abroad(loader):
    import puremacro.trade.data as trade_data

    calib = getattr(trade_data, loader)(custom_c=3, custom_s=2, seed=1, fallback_to_synthetic=True)
    mask = np.array([True, False])
    M, n = calib.n_countries * calib.n_sectors, calib.n_countries
    cx = np.asarray(calib.data_calibra)[:M, M:].reshape(M, n, 3)[:, :, 2]
    if loader == "load_exiobase":
        with pytest.raises(ValueError, match=r"EXPORT.*cx_category"):
            BalancedIOTable.from_trade_calibration(calib, negative_investment="to_inventory", merchandise_mask=mask)
        table = BalancedIOTable.from_trade_calibration(calib, negative_investment="to_inventory",
                                                       merchandise_mask=mask, cx_category="V")
    else:
        table = BalancedIOTable.from_trade_calibration(calib, negative_investment="to_inventory", merchandise_mask=mask)
    assert not table.F[:, :, 2].any()          # X: residents' purchases abroad
    np.testing.assert_array_equal(table.F[:, :, 3], cx)
    assert table.metadata["final_use_bridge"]["cx_category"] == "V"
    assert table.metadata["investment_includes_inventories"] is False
    with pytest.raises(CalibrationError, match="allow_empty_purchases_abroad"):
        calibrate_condensed(table)
    assert calibrate_condensed(table, allow_empty_purchases_abroad=True).table is table


def test_from_trade_calibration_keeps_signed_provider_inventories_in_v():
    from puremacro.trade.data import load_figaro

    calib = load_figaro(custom_c=3, custom_s=2, seed=1, fallback_to_synthetic=True)
    M = calib.n_countries * calib.n_sectors
    D = np.array(calib.data_calibra, dtype=float, copy=True)
    cx_col, c_col = M + 3 * 1 + 2, M + 3 * 1
    D[0, c_col] += D[0, cx_col] + 1.0          # keep the row total
    D[0, cx_col] = -1.0                        # one inventory drawdown
    table = BalancedIOTable.from_trade_calibration(dataclasses.replace(calib, data_calibra=D),
                                                   negative_investment="to_inventory",
                                                   merchandise_mask=np.array([True, False]))
    assert table.F[0, 1, 3] == -1.0            # V keeps the signed value
    assert not table.F[:, :, 2].any()          # X stays empty
    assert table.report.negative_investment_cells == ()


def test_from_trade_calibration_oecd_mapping_bridges_cx_as_purchases_abroad():
    from puremacro.trade.data import load_oecd_icio_granular

    calib = load_oecd_icio_granular(custom_c=3, custom_s=2, seed=1, fallback_to_synthetic=True)
    mask = np.array([True, False])
    table = BalancedIOTable.from_trade_calibration(calib, negative_investment="to_inventory", merchandise_mask=mask)
    unmapped = dataclasses.replace(calib, metadata={k: v for k, v in calib.metadata.items()
                                                    if k != "final_use_mapping"})
    legacy = BalancedIOTable.from_trade_calibration(unmapped, negative_investment="to_inventory", merchandise_mask=mask)
    for name in ("Z", "F", "VA", "TLS", "TFD"):
        np.testing.assert_array_equal(getattr(table, name), getattr(legacy, name))
    assert table.F[:, :, 2].sum() > 0
    odd = dataclasses.replace(calib, metadata={**calib.metadata,
                                               "final_use_mapping": {"C": ["C"], "I": ["I"], "Cx": ["FOO"]}})
    with pytest.raises(ValueError, match=r"BalancedIOTable.from_trade_calibration.*\['FOO'\]"):
        BalancedIOTable.from_trade_calibration(odd, negative_investment="to_inventory", merchandise_mask=mask)


def test_infer_merchandise_mask_registries():
    assert infer_merchandise_mask(("AGRI", "MINQ", "MANU", "ENEG")).tolist() == [True, True, True, False]
    assert infer_merchandise_mask(("AGRI_MIN_FOOD", "MANUF_EXFOOD", "UTILITIES")).tolist() == [True, True, False]
    assert infer_merchandise_mask(("A", "B", "C", "DE", "REST")).tolist() == [True, True, True, False, False]
    with pytest.raises(UnsupportedExtension):
        infer_merchandise_mask(("AGRI", "ZZZ"))


# ---------------------------------------------------------------------------
# Calibration and productivity gates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("closure", ["factor", "gdp", "own"])
@pytest.mark.parametrize("numeraire", ["factor", "USA", "ROW", "w_USA"])
def test_benchmark_reproduction_synthetic(synth_calib, closure, numeraire):
    model = CondensedLeontiefModel(synth_calib, TariffWedges.free_trade(synth_calib), closure=closure,
                                   numeraire=numeraire)
    st = model.state(model.z0)
    assert st.max_scaled_residual <= 1e-14 and abs(st.walras_residual) <= 1e-14
    np.testing.assert_allclose(st.p, 1.0, atol=1e-14)
    np.testing.assert_allclose(st.P, 1.0, atol=1e-14)
    np.testing.assert_allclose(st.y, synth_calib.y0, rtol=1e-13)
    np.testing.assert_allclose(st.XN, synth_calib.XN0, atol=1e-12)
    np.testing.assert_allclose(st.TR, 0.0, atol=1e-14)
    assert model.n_vars == 6


@pytest.mark.parametrize("closure", ["factor", "gdp", "own"])
def test_benchmark_reproduction_fixtures(oecd3_calib, bundled_calib, closure):
    for calib in (oecd3_calib, bundled_calib):
        model = CondensedLeontiefModel(calib, TariffWedges.free_trade(calib), closure=closure)
        st = model.state(model.z0)
        assert st.max_scaled_residual <= 1e-14 and abs(st.walras_residual) <= 1e-14
        np.testing.assert_allclose(st.y, calib.y0, rtol=1e-12)
        cert = certify_raw_flows(calib, model.wedges, st, closure=closure)
        assert cert.passed and cert.max_block <= 1e-12


def test_calibration_gates_and_errors(synth_table, synth_calib, oecd3_table):
    g = synth_calib.gates
    assert g["cell_balance_rel"] <= 1e-10 and g["national_accounts_rel"] <= 1e-12 and g["theta_sum_abs"] <= 1e-12
    assert g["rho_a_lower"] <= g["rho_a_estimate"] <= g["rho_a_upper"] < 1 - 1e-3
    assert g["rho_a_method"] == "collatz_wielandt" and g["deficit_share_sum"] == 0.0
    np.testing.assert_allclose(synth_calib.theta.sum(1), 1.0, atol=1e-13)
    np.testing.assert_allclose(synth_calib.a.sum(0) + synth_calib.b + synth_calib.t, 1.0, atol=1e-13)
    np.testing.assert_allclose(synth_calib.Y0 - synth_calib.A0.sum(1) - synth_calib.EV0 - synth_calib.XN0, 0.0,
                               atol=1e-12)
    assert synth_calib.s_factor.sum() == 0.0 and abs(synth_calib.s_gdp.sum()) < 1e-16
    assert "3 economies" in synth_calib.summary() and synth_calib.to_dataframe().shape == (3, 7)
    assert not synth_calib.a.flags.writeable
    assert not synth_calib.active.flags.writeable and not synth_calib.goods.flags.writeable
    assert synth_calib.active.dtype == bool and synth_calib.goods.dtype == bool
    # A country without purchases abroad needs an explicit opt-in; its X basket copies C.
    a = synthetic_arrays()
    F = a["F"].copy()
    F[:, 1, 5] = 0.0
    TFD = a["TFD"].copy()
    TFD[1, 5] = 0.0
    VA = a["Z"].sum(1) + F.sum((1, 2)) - a["Z"].sum(0) - a["TLS"]
    empty_x, _ = table_from_arrays(a["Z"], F, VA, a["TLS"], TFD, country_codes=CODES3, sector_codes=SECTORS4,
                                   fd_codes=OECD6)
    with pytest.raises(CalibrationError, match="zero basic value"):
        calibrate_condensed(empty_x)
    cx = calibrate_condensed(empty_x, allow_empty_purchases_abroad=True)
    assert cx.theta[1, 2] == 0.0 and cx.B0[1, 2] == 0.0
    np.testing.assert_array_equal(cx.omega[:, 1, 2], cx.omega[:, 1, 0])
    res = solve_condensed(cx, us10(cx))
    assert res.passed and res.state.q[2, 1] == 0.0
    # A negative basket coefficient in an endogenous category is refused.
    F = synth_table.F.copy()
    F[0, 1, 0] = -0.1
    VA = synth_table.Z.sum(1) + F.sum((1, 2)) - synth_table.Z.sum(0) - synth_table.TLS
    bad, _ = table_from_arrays(synth_table.Z, F, VA, synth_table.TLS, synth_table.TFD,
                               country_codes=CODES3, sector_codes=SECTORS4, fd_codes=("C", "G", "X", "V"))
    with pytest.raises(CalibrationError, match="negative basket coefficient"):
        calibrate_condensed(bad)
    with pytest.raises(TypeError):
        calibrate_condensed(synth_table.Z)


@pytest.mark.parametrize("B", [np.array([[.99, 1.], [0., .99]]), np.array([[0., 1.2], [.3, 0.]])])
def test_productivity_bounds_are_not_rejections(B):
    """Ported from the IO test_repaired_math: a periodic or triangular matrix with rho < 1 is accepted."""
    cert = certify_productivity(B)
    assert cert.passed and cert.upper < 1 and cert.lower <= cert.rho <= cert.upper + 1e-12
    assert max(abs(np.linalg.eigvals(B))) < 1
    assert cert.to_dataframe().shape == (6, 1) and "passed" in cert.summary() and "upper" in cert.to_markdown()


def test_productivity_reducible_nonproductive_and_inconclusive():
    B = np.zeros((201, 201))
    B[0, 1] = 1.0201
    B[1, 0] = 1.0
    B[2:, 2:] = np.eye(199) * .1
    with pytest.raises(ExistenceViolation):
        certify_productivity(B)
    # Bounds that certify rho < 1 but not rho < 1 - margin are reported as a margin violation (fail closed).
    with pytest.raises(ProductivityUncertified, match="margin not met") as exc:
        certify_productivity(np.array([[0.9995, 0.0], [0.0, 0.9995]]), margin=1e-3)
    assert exc.value.lower == pytest.approx(0.9995, abs=1e-9) and exc.value.upper == pytest.approx(0.9995, abs=1e-9)
    assert exc.value.upper < 1.0 and exc.value.margin == 1e-3
    assert certify_productivity(np.array([[0.9995, 0.0], [0.0, 0.9995]]), margin=1e-4).passed
    with pytest.raises(ValueError):
        certify_productivity(np.array([[0.1, -0.1], [0.0, 0.1]]))
    up, lo = collatz_wielandt_bound(lambda x: B[2:, 2:] @ x, 199)
    assert lo <= 0.1 + 1e-12 and up >= 0.1 - 1e-12


def test_existence_gate_extreme_tariffs(oecd3_calib, synth_calib):
    for r in (1.0, 2.0, 5.0, 9.9):
        b = existence_gate(oecd3_calib, build_tariff_wedges(oecd3_calib, r, importer="USA"))
        assert b.lower <= b.upper < 1 - 1e-3 and b.passed
    with pytest.raises(ExistenceViolation):
        existence_gate(synth_calib, build_tariff_wedges(synth_calib, 30.0, importer="USA"))


def test_existence_gate_rejects_wedges_of_another_table(oecd3_calib, synth_calib):
    """Wedges built for a 12-cell table are refused by name before any arithmetic on a 9-cell one."""
    with pytest.raises(CalibrationError, match="another table"):
        existence_gate(oecd3_calib, us10(synth_calib))
    with pytest.raises(CalibrationError, match="wedges.tau has shape"):
        solve_condensed(oecd3_calib, us10(synth_calib))


# ---------------------------------------------------------------------------
# Tariff wedges
# ---------------------------------------------------------------------------

def test_build_tariff_wedges_semantics(synth_calib):
    calib = synth_calib
    m, n, s = calib.n_cells, calib.n_countries, calib.n_sectors
    w = us10(calib)
    usa = calib.index("USA")
    cols = slice(usa * s, (usa + 1) * s)
    goods = calib.goods
    for o in range(n):
        rows = slice(o * s, (o + 1) * s)
        expected = np.where(goods, 1.10, 1.0) if o != usa else np.ones(s)
        np.testing.assert_allclose(w.tau[rows, cols], np.broadcast_to(expected[:, None], (s, s)), atol=1e-15)
        np.testing.assert_allclose(w.tau_fd[rows, usa, 0], expected)
        np.testing.assert_allclose(w.tau_fd[rows, usa, 1], expected)
        np.testing.assert_allclose(w.tau_V[rows, usa], expected)
    assert np.all(w.tau_fd[:, :, 2] == 1.0)
    other = [k for k in range(n) if k != usa]
    assert np.all(w.tau[:, [j for k in other for j in range(k * s, (k + 1) * s)]] == 1.0)
    assert w.rate_table(usa).shape == (n, s) and w.rate_table(usa)[1, 0] == 0.10 and w.rate_table(usa)[1, 2] == 0.0
    assert len(w.sha256) == 64 and w.sha256 == us10(calib).sha256 and not w.is_free_trade
    assert w.metadata["importers"] == ("USA",) and w.metadata["fd_tariffed"] == ("C", "G", "V")
    all_cov = build_tariff_wedges(calib, 0.10, importer="USA", coverage="all")
    assert np.all(all_cov.tau[3 * s:, cols] == 1.10) or np.all(all_cov.tau[s:2 * s, cols] == 1.10)
    only_c = build_tariff_wedges(calib, 0.10, importer="USA", fd_tariffed=("C",))
    assert np.all(only_c.tau_fd[:, :, 1] == 1.0) and np.all(only_c.tau_V == 1.0)
    untar = build_tariff_wedges(calib, 0.10, importer="USA", inventories="untariffed")
    assert np.all(untar.tau_V == 1.0)
    full = np.zeros((n, n, s))
    full[1, usa, :] = 0.25
    with pytest.warns(RuntimeWarning, match="non-merchandise"):
        f = build_tariff_wedges(calib, full)
    np.testing.assert_allclose(f.tau[s:2 * s, cols], np.broadcast_to(np.where(goods, 1.25, 1.0)[:, None], (s, s)))
    with pytest.warns(RuntimeWarning, match="non-merchandise"):
        arr = build_tariff_wedges(calib, np.full((n, s), 0.25), importer="USA")
    assert np.all(arr.tau[cols, cols] == 1.0)
    half = scale_wedges(w, 0.5)
    np.testing.assert_allclose(half.tau - 1.0, 0.5 * (w.tau - 1.0))
    np.testing.assert_allclose(half.rates, 0.5 * w.rates)
    assert TariffWedges.free_trade(calib).is_free_trade


def test_callable_rates_match_mapping_rates(synth_calib):
    scen = TariffScenario(name="t10_25", default_us_tariff=0.10, us_import_tariffs={"CHN": 0.25})
    # The callable names every sector, so trimming its service rates to merchandise is announced.
    with pytest.warns(RuntimeWarning, match="non-merchandise"):
        a = build_tariff_wedges(synth_calib, scen.get_rate)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # scalar per-origin rates mean "every covered sector": silent
        b = build_tariff_wedges(synth_calib, {"USA": {"*": 0.10, "CHN": 0.25}})
    assert a.sha256 == b.sha256
    np.testing.assert_array_equal(a.tau, b.tau)
    np.testing.assert_array_equal(a.rates, b.rates)
    assert a.metadata["dropped_service_rates"] == ("G", "K") and b.metadata["dropped_service_rates"] == ()


def test_named_origins_override_the_wildcard_in_any_order(synth_calib):
    usa, chn, row = (synth_calib.index(c) for c in CODES3)
    goods = np.asarray(synth_calib.goods)
    first = build_tariff_wedges(synth_calib, {"USA": {"CHN": 0.25, "*": 0.10}})
    last = build_tariff_wedges(synth_calib, {"USA": {"*": 0.10, "CHN": 0.25}})
    np.testing.assert_array_equal(first.rates, last.rates)
    np.testing.assert_array_equal(first.tau, last.tau)
    assert first.sha256 == last.sha256
    assert np.all(first.rates[chn, usa, goods] == 0.25) and np.all(first.rates[row, usa, goods] == 0.10)
    assert np.all(first.rates[:, usa, ~goods] == 0.0) and np.all(first.rates[usa, usa] == 0.0)


def test_goods_coverage_warns_on_explicit_service_rates(synth_calib):
    n, s = synth_calib.n_countries, synth_calib.n_sectors
    tab = np.zeros((n, s))
    tab[1, :] = 0.25
    with pytest.warns(RuntimeWarning, match="non-merchandise sectors \\['G', 'K'\\]"):
        w = build_tariff_wedges(synth_calib, tab, importer="USA")
    assert w.rate_table(0)[1].tolist() == [0.25, 0.25, 0.0, 0.0]
    assert w.metadata["dropped_service_rates"] == ("G", "K") and w.metadata["coverage"] == "goods"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scalar = build_tariff_wedges(synth_calib, 0.25, importer="USA")
        kept = build_tariff_wedges(synth_calib, tab, importer="USA", coverage="all")
        goods_only = build_tariff_wedges(synth_calib, tab * np.array([1, 1, 0, 0]), importer="USA")
    assert scalar.metadata["dropped_service_rates"] == () and goods_only.metadata["dropped_service_rates"] == ()
    assert kept.rate_table(0)[1].tolist() == [0.25] * 4 and kept.metadata["dropped_service_rates"] == ()
    np.testing.assert_array_equal(goods_only.tau, w.tau)


def test_wedge_validation(synth_calib):
    calib = synth_calib
    m, n = calib.n_cells, calib.n_countries
    with pytest.raises(ValueError, match="nonnegative"):
        build_tariff_wedges(calib, -0.1, importer="USA")
    with pytest.raises(ValueError, match="importer"):
        build_tariff_wedges(calib, 0.1)
    with pytest.raises(UnsupportedExtension):
        build_tariff_wedges(calib, 0.1, importer="USA", fd_tariffed=("C", "X"))
    with pytest.raises(UnsupportedExtension):
        build_tariff_wedges(calib, 0.1, importer="USA", coverage="services")
    with pytest.raises(DataIntegrityError):
        build_tariff_wedges(calib, {"USA": np.ones(5)})
    tau = np.ones((m, m))
    tau[0, 1] = 1.2  # domestic USA transaction
    with pytest.raises(DataIntegrityError, match="domestic"):
        TariffWedges.from_arrays(tau, np.ones((m, n, 3)), np.ones((m, n)))
    tfd = np.ones((m, n, 3))
    tfd[5, 0, 2] = 1.1
    with pytest.raises(UnsupportedExtension):
        TariffWedges.from_arrays(np.ones((m, m)), tfd, np.ones((m, n)))
    with pytest.raises(DataIntegrityError, match="positive"):
        TariffWedges.from_arrays(np.zeros((m, m)), np.ones((m, n, 3)), np.ones((m, n)))
    w = us10(calib)
    assert not w.tau.flags.writeable
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.sha256 = "x"


# ---------------------------------------------------------------------------
# Solve, certificate and its sensitivity
# ---------------------------------------------------------------------------

def test_solve_certificate_arclength_multistart_synthetic(synth_calib):
    w = us10(synth_calib)
    res = solve_condensed(synth_calib, w, run_arclength=True, run_multistart=True)
    assert isinstance(res, CondensedEquilibriumResult) and res.converged and res.passed
    assert res.max_scaled_residual <= 1e-12 and abs(res.walras_residual) <= 1e-10
    assert isinstance(res.certificate, RawFlowCertificate) and res.certificate.max_block <= 1e-10
    assert len(res.certificate.blocks) == 10 and res.certificate.passed
    assert res.path[-1]["lam"] == 1.0 and res.path[-1]["ok"]
    assert len(res.arclength_path) >= 1 and all(h["dlam_ds"] > 0 for h in res.arclength_path)
    assert len(res.multistart) == 8 and all(r["converged"] and r["dist"] <= 1e-9 for r in res.multistart)
    assert res.existence["passed"] and res.existence["upper"] < 1 - 1e-3
    assert res.options["closure"] == "factor" and res.metadata["n_unknowns"] == 6
    again = certify_raw_flows(synth_calib, w, res.state)
    assert again.blocks == res.certificate.blocks
    # Numbers observed with the IO engine on this fixture (fixture_small.log).
    m = compute_measures(synth_calib, w, res.state, country="USA", partners=("CHN",))
    assert m.pc_over_w_pct == pytest.approx(7.0518197, abs=1e-6)
    assert m.ev_pct_base_gdp == pytest.approx(0.34507718, abs=1e-7)
    assert m.tot_fisher_pct == pytest.approx(0.31147314, abs=1e-7)
    assert m.partner_ev_pct["CHN"] == pytest.approx(-0.15352148, abs=1e-7)
    assert m.pc_pe_pct == pytest.approx(4.6591515, abs=1e-6)


def test_solve_oecd_fixture_headline(oecd3_calib):
    w = us10(oecd3_calib)
    res = solve_condensed(oecd3_calib, w, run_arclength=True, run_multistart=True)
    assert res.passed and res.certificate.max_block <= 1e-10
    m = compute_measures(oecd3_calib, w, res.state, country="USA")
    # Values recorded by the assessment with the IO engine on this fixture (fixture_oecd3x3.log).
    assert m.pc_over_w_pct == pytest.approx(0.749739, abs=2e-6)
    assert m.ev_pct_base_gdp == pytest.approx(-0.002658, abs=2e-6)
    assert m.effective_rate_dutiable_pct == pytest.approx(10.0, abs=1e-10)
    recip = build_tariff_wedges(oecd3_calib, {"USA": {"*": 0.10}, "CHN": {"USA": 0.10}, "REST": {"USA": 0.10}})
    r2 = solve_condensed(oecd3_calib, recip)
    m2 = compute_measures(oecd3_calib, recip, r2.state, country="USA")
    assert m2.ev_pct_base_gdp == pytest.approx(-0.001959, abs=2e-6)
    assert r2.state.TR[1] > 0 and r2.state.TR[2] > 0


def test_solve_bundled_77x11(bundled_calib):
    w = us10(bundled_calib)
    res = solve_condensed(bundled_calib, w)
    assert res.passed and res.certificate.max_block <= 1e-10 and res.iterations <= 6
    m = compute_measures(bundled_calib, w, res.state, country="USA", partners=("CHN", "MEX", "CAN"),
                         blocs={"EU27": ("DEU", "FRA", "ITA", "ESP")})
    # Values recorded by the assessment with the IO engine on this bridge (bridge_bundled_77x11_split.log).
    assert m.pc_over_w_pct == pytest.approx(0.723399, abs=2e-6)
    assert m.ev_pct_base_gdp == pytest.approx(-0.029063, abs=2e-6)
    assert m.tot_fisher_pct == pytest.approx(-0.105502, abs=2e-6)
    assert m.tariff_revenue_share_pct == pytest.approx(0.800143, abs=2e-6)
    assert m.effective_rate_dutiable_pct == pytest.approx(10.0, abs=1e-10)
    assert set(m.partner_ev_pct) == {"CHN", "MEX", "CAN"} and "EU27" in m.bloc_ev_pct
    frame = res.to_dataframe()
    assert frame.shape == (77, 9) and frame.loc["USA", "pc_over_w_pct"] == pytest.approx(m.pc_over_w_pct)


@pytest.mark.slow
def test_bundled_77x11_arclength_and_multistart(bundled_calib):
    res = solve_condensed(bundled_calib, us10(bundled_calib), run_arclength=True, run_multistart=True)
    assert res.passed and all(h["dlam_ds"] > 0 and h["sign_det"] != 0 for h in res.arclength_path)
    assert all(r["converged"] and r["dist"] <= 1e-9 for r in res.multistart)


def test_certificate_fault_injection_names_the_block(synth_calib):
    w = us10(synth_calib)
    res = solve_condensed(synth_calib, w)
    st = res.state

    def triggers(field, fn, block):
        bad = dataclasses.replace(st, **{field: fn(getattr(st, field).copy())})
        with pytest.raises(CertificationFailure) as info:
            certify_raw_flows(synth_calib, w, bad)
        failed = [b for b, v in info.value.blocks.items() if v > 1e-10]
        assert block in failed, (field, failed)
        lenient = certify_raw_flows(synth_calib, w, bad, strict=False)
        assert not lenient.passed and block in lenient.failed_blocks

    triggers("w", lambda a: a * (1 + 1e-5), "numeraire_identity")
    triggers("p", lambda a: a * (1 + 1e-5), "zero_profit_active")
    triggers("y", lambda a: a * (1 + 1e-5), "goods_clearing")
    triggers("Y", lambda a: a * (1 + 1e-5), "household_budget")
    # The documented sensitivity: a relative 1e-5 perturbation of any of the six state vectors.
    triggers("XN", lambda a: a * (1 + 1e-5), "current_account_identity")
    triggers("TR", lambda a: a * (1 + 1e-5), "tariff_revenue_identity")
    # Sensitivity threshold: a 1e-9 wage perturbation is detected in the income block.
    w_bad = st.w.copy()
    w_bad[0] += 1e-9
    with pytest.raises(CertificationFailure) as info:
        certify_raw_flows(synth_calib, w, dataclasses.replace(st, w=w_bad))
    assert info.value.blocks["national_income"] > 1e-10
    frame = res.certificate.to_dataframe()
    assert frame.shape == (10, 2) and bool(frame["Passed"].all())
    assert "zero_profit_active" in res.certificate.to_markdown()


def test_certificate_cell_floor_governs_per_cell_sensitivity():
    """The per-cell blocks are scaled by max(value, cell_floor): in a table whose cells are below one unit
    the default floor of one judges them at absolute 1e-10, a smaller floor restores relative sensitivity."""
    calib = calibrate_condensed(synthetic_table(scale=1e-6))
    w = us10(calib)
    res = solve_condensed(calib, w)
    assert res.passed and res.certificate.cell_floor == 1.0 and res.certificate.to_dict()["cell_floor"] == 1.0
    bad = dataclasses.replace(res.state, p=res.state.p * (1 + 1e-6))
    default = certify_raw_flows(calib, w, bad, strict=False)
    small = certify_raw_flows(calib, w, bad, strict=False, cell_floor=1e-9)
    assert default.blocks["zero_profit_active"] < 1e-10 < small.blocks["zero_profit_active"]
    assert small.blocks["zero_profit_active"] > 1e3 * default.blocks["zero_profit_active"]
    assert not default.passed and "national_income" in default.failed_blocks  # the whole certificate still rejects
    assert small.cell_floor == 1e-9 and "zero_profit_active" in small.failed_blocks
    with pytest.raises(ValueError, match="cell_floor"):
        certify_raw_flows(calib, w, res.state, cell_floor=0.0)


def _table_with_empty_cells(cells, scale: float = 1.0) -> BalancedIOTable:
    """The synthetic 3x4 table with the listed cells emptied (they become 1e-6 phantoms)."""
    a = synthetic_arrays(scale=scale)
    Z, F, TLS = a["Z"].copy(), a["F"].copy(), a["TLS"].copy()
    for j in cells:
        Z[j, :] = 0.0
        Z[:, j] = 0.0
        F[j] = 0.0
        TLS[j] = 0.0
    out = Z.sum(1) + F.sum((1, 2))
    VA = out - Z.sum(0) - TLS
    VA[list(cells)] = 0.0
    table, report = table_from_arrays(Z, F, VA, TLS, a["TFD"], country_codes=CODES3, sector_codes=SECTORS4,
                                      fd_codes=OECD6, output=out)
    assert report.n_zero_output == len(cells)
    return table


def test_phantom_cells_enter_the_certificate_with_their_true_output():
    """Regression: the model lets phantom output move (factor demand b_j y_j) and phantom value added is part
    of Fbar, so the certificate must use yhat_j = y_j / y0_j on phantoms too. The IO rule yhat_j = 1 on
    inactive cells rejected these valid equilibria of a natural-unit table (Fbar ~ 50, phantoms 1e-6) with
    factor_markets violations of 1.3e-10 to 2.1e-9."""
    old_rule_violation = 0.0
    for cells, importer, rate in (([5], "ROW", 2.0), ([5, 9], "USA", 0.25), ([0, 1, 4, 5, 9, 10], "USA", 0.25)):
        calib = calibrate_condensed(_table_with_empty_cells(cells))
        assert not calib.active[cells].any()
        w = build_tariff_wedges(calib, rate, importer=importer)
        res = solve_condensed(calib, w)
        assert res.passed and res.certificate.factor_markets <= 1e-12, (cells, res.certificate.blocks)
        yhat = res.state.y / calib.y0
        # What the IO rule dropped: sum over phantoms of VA_j (yhat_j - 1), relative to Fbar_k.
        dropped = np.bincount(calib.country_of_cell[cells], weights=calib.table.VA[cells] * (yhat[cells] - 1.0),
                              minlength=calib.n_countries) / calib.Fbar
        old_rule_violation = max(old_rule_violation, float(np.max(np.abs(dropped))))
        # Real GDP at base prices uses the same yhat on every cell as the certificate.
        k = calib.index(importer)
        rows = slice(k * 4, (k + 1) * 4)
        expected = (np.sum((calib.table.VA + calib.table.TLS)[rows] * yhat[rows])
                    + np.sum(calib.table.TFD[k, :3] * res.state.q[:, k] / calib.B0[k, :])
                    + calib.table.TFD[k, 3]) / calib.Y0[k] - 1.0
        assert constant_price_rgdp(calib, res.state, country=importer) == pytest.approx(expected, abs=1e-14)
    assert old_rule_violation > 1e-10  # the old rule would have failed at least one of these cases


def test_uncertified_results_are_never_passed(synth_calib):
    w = us10(synth_calib)
    res = solve_condensed(synth_calib, w, certify=False)
    assert res.converged and res.certificate is None and not res.passed and not res.certified
    assert "no certificate" in res.summary()
    full = solve_condensed(synth_calib, w)
    assert full.passed and full.certified and np.array_equal(full.state.z, res.state.z)


def test_certificate_rejects_wrong_options(synth_calib):
    w = us10(synth_calib)
    res = solve_condensed(synth_calib, w, closure="gdp", numeraire="USA")
    assert res.certificate.metadata["closure"] == "gdp"
    with pytest.raises(CertificationFailure):
        certify_raw_flows(synth_calib, w, res.state, closure="factor", numeraire="USA")
    with pytest.raises(CertificationFailure):
        certify_raw_flows(synth_calib, w, res.state, closure="gdp", numeraire="factor")
    with pytest.raises(ValueError):
        certify_raw_flows(synth_calib, w, res.state, closure="other")


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

def test_ev_split_additivity_every_country(synth_calib, oecd3_calib):
    for calib in (synth_calib, oecd3_calib):
        w = us10(calib)
        res = solve_condensed(calib, w)
        for code in calib.country_codes:
            sp = ev_split(calib, res.state, country=code)
            assert abs(sp.residual_pp) <= 1e-12
            assert sp.revenue_pp == pytest.approx(sp.tariff_revenue_pp + sp.other_taxes_pp, abs=1e-12)
            assert sp.ev_pct_base_gdp == pytest.approx(res.ev_pct_gdp[code], abs=1e-12)
            assert sp.country == code
        frame = sp.to_dataframe()
        assert frame.shape == (7, 1) and "Equivalent variation" in sp.to_markdown()


def _all_measures(calib, w, state):
    m = compute_measures(calib, w, state, country="USA", partners=("CHN",))
    return {**m.headline(), **{"split_" + k: v for k, v in m.ev_split.to_dict().items()},
            "pc_pe": m.pc_pe_pct, "ge_minus_pe": m.ge_minus_pe_pct, "world": m.world_ev_pct,
            "chn": m.partner_ev_pct["CHN"]}


def test_numeraire_invariance(synth_calib):
    w = us10(synth_calib)
    ref = None
    for numeraire in ("factor", "USA", "ROW", "w_USA"):
        res = solve_condensed(synth_calib, w, method="newton", numeraire=numeraire)
        vals = _all_measures(synth_calib, w, res.state)
        if ref is None:
            ref = vals
            continue
        assert len(vals) == 18
        assert max(abs(vals[k] - ref[k]) for k in vals) <= 1e-9
    with pytest.raises(UnsupportedExtension):
        solve_condensed(synth_calib, w, numeraire="XXX")


def test_inventory_duty_neutrality_under_factor_closure(synth_calib):
    calib = synth_calib
    m, n = calib.n_cells, calib.n_countries
    tau_V = np.ones((m, n))
    foreign = calib.country_of_cell != calib.index("USA")
    tau_V[foreign, calib.index("USA")] = 1.10
    wv = TariffWedges.from_arrays(np.ones((m, m)), np.ones((m, n, 3)), tau_V)
    res = solve_condensed(calib, wv, closure="factor")
    # Exactly neutral up to the 1e-12 Newton stopping rule: the factor closure does not depend on
    # inventory prices, so a duty on fixed inventory quantities only moves revenue.
    assert np.max(np.abs(res.state.p - 1.0)) <= 1e-12 and np.max(np.abs(res.state.w - 1.0)) <= 1e-12
    assert max(abs(v) for v in res.ev_pct_gdp.values()) <= 1e-10
    assert res.state.TR[calib.index("USA")] > 0
    res_gdp = solve_condensed(calib, wv, closure="gdp")
    assert np.max(np.abs(res_gdp.state.p - 1.0)) > 1e-6


def test_homogeneity_in_units(synth_calib):
    w = us10(synth_calib)
    base = _all_measures(synth_calib, w, solve_condensed(synth_calib, w).state)
    calib_k = calibrate_condensed(synthetic_table(scale=1000.0))
    w_k = us10(calib_k)
    scaled = _all_measures(calib_k, w_k, solve_condensed(calib_k, w_k).state)
    assert max(abs(base[k] - scaled[k]) for k in base) <= 1e-10


def test_effective_rate_is_exact_for_a_uniform_duty(synth_calib, oecd3_calib):
    for calib in (synth_calib, oecd3_calib):
        for rate in (0.10, 0.37):
            w = build_tariff_wedges(calib, rate, importer="USA")
            res = solve_condensed(calib, w)
            share, eff = tariff_revenue_and_effective_rate(calib, w, res.state, country="USA")
            assert eff == pytest.approx(100 * rate, abs=1e-10) and share > 0
            share_c, eff_c = tariff_revenue_and_effective_rate(calib, w, res.state, country="CHN")
            assert share_c == 0.0 and eff_c == 0.0


def test_pe_bound_and_first_order_decomposition(synth_calib):
    w = us10(synth_calib)
    res = solve_condensed(synth_calib, w)
    p_pe, pi_pe = pe_unit_cost_bound(synth_calib, w, country="USA")
    pi_ge = consumer_price_relative_to_wages(synth_calib, res.state, country="USA")
    assert p_pe.shape == (12,) and pi_pe <= pi_ge
    m = compute_measures(synth_calib, w, res.state, country="USA")
    assert m.ge_minus_pe_pct >= 0.0 and m.pc_pe_pct == pytest.approx(100 * pi_pe)
    total, direct, cascade = pe_first_order_decomposition(synth_calib, w, country="USA")
    assert total == pytest.approx(direct + cascade, abs=1e-15) and direct > 0 and cascade > 0
    assert abs(total - pi_pe) < 0.1 * abs(pi_pe) + 1e-3
    # Zero wedges give zero PE effects and a benchmark real national income of zero.
    z0 = TariffWedges.free_trade(synth_calib)
    assert pe_unit_cost_bound(synth_calib, z0, country="USA")[1] == pytest.approx(0.0, abs=1e-14)
    st0 = CondensedLeontiefModel(synth_calib, z0).state(CondensedLeontiefModel(synth_calib, z0).z0)
    assert real_national_income(synth_calib, st0, country="USA") == pytest.approx(0.0, abs=1e-14)
    assert tot_fisher(synth_calib, st0, country="USA") == pytest.approx(0.0, abs=1e-14)


def test_sector_incidence_and_aggregation_gaps(synth_calib):
    w = us10(synth_calib)
    res = solve_condensed(synth_calib, w)
    dlp, dly, disp = sector_incidence(synth_calib, res.state, country="USA", groups=[1, 1, 2, 2])
    assert dlp.shape == (4,) and dly.shape == (4,) and set(disp) == {1, 2} and all(v >= 0 for v in disp.values())
    y0 = synth_calib.y0[:4]
    wts = y0[:2] / y0[:2].sum()
    mean = float(np.sum(wts * np.log(res.state.p[:2])))
    assert disp[1] == pytest.approx(float(np.sum(wts * (np.log(res.state.p[:2]) - mean) ** 2)))
    assert sector_incidence(synth_calib, res.state, country=0)[2] == {}
    with pytest.raises(ValueError):
        sector_incidence(synth_calib, res.state, country="USA", groups=[1, 2])
    gaps = aggregation_gaps({"a": 1.5, "b": 2.0, "c": 3.0}, {"a": 1.0, "b": 0.0})
    assert gaps["a"] == {"fine": 1.5, "coarse": 1.0, "absolute_gap": 0.5, "relative_gap": 0.5}
    assert np.isnan(gaps["b"]["relative_gap"]) and "c" not in gaps


def test_measures_result_exports(synth_calib):
    w = us10(synth_calib)
    res = solve_condensed(synth_calib, w)
    m = compute_measures(synth_calib, w, res.state, country=0, partners=("CHN",), blocs={"PAIR": ("CHN", "ROW", "ZZZ")})
    assert isinstance(m, CondensedMeasuresResult) and m.country == "USA"
    frame = m.to_dataframe()
    assert frame.shape[1] == 1 and "Consumer prices relative to wages (%)" in frame.index
    assert "USA" in m.to_markdown() and "\\begin{tabular}" in m.to_latex() and "#table" in m.to_typst()
    assert "numeraire-free" in m.summary() and m.metadata["wedges_sha256"] == w.sha256
    d = m.to_dict()
    assert d["bloc_ev_pct"]["PAIR"] == pytest.approx(100 * (
        res.ev_pct_gdp["CHN"] / 100 * synth_calib.Y0[1] + res.ev_pct_gdp["ROW"] / 100 * synth_calib.Y0[2]
    ) / (synth_calib.Y0[1] + synth_calib.Y0[2]), abs=1e-12)
    world = sum(res.ev_pct_gdp[c] / 100 * synth_calib.Y0[k] for k, c in enumerate(synth_calib.country_codes))
    assert m.world_ev_pct == pytest.approx(100 * world / synth_calib.Y0.sum(), abs=1e-12)
    with pytest.raises(ValueError):
        compute_measures(synth_calib, w, res.state, country=7)


# ---------------------------------------------------------------------------
# Jacobian, Newton, continuation, arclength, multistart, fallback
# ---------------------------------------------------------------------------

def test_forward_and_central_jacobians_agree(synth_calib):
    w = us10(synth_calib)
    res = solve_condensed(synth_calib, w)
    model = CondensedLeontiefModel(synth_calib, w)
    Jf = model.jacobian(res.state.z)
    Jc = model.jacobian(res.state.z, scheme="central", h=1e-6)
    assert np.max(np.abs(Jf - Jc)) / np.max(np.abs(Jc)) <= 1e-6
    assert np.linalg.matrix_rank(Jf) == 6
    with pytest.raises(ValueError):
        model.jacobian(res.state.z, scheme="complex")


def test_newton_stopping_rule_and_failure_contract(synth_calib):
    w = build_tariff_wedges(synth_calib, 1.0, importer="USA")
    model = CondensedLeontiefModel(synth_calib, w)
    z, info = newton(model)
    assert info["max_scaled_residual"] <= 1e-12 and info["walras_residual"] <= 1e-10 and info["jacobians"] >= 1
    z_again, info0 = newton(model, z0=z)
    assert info0["iterations"] == 0 and np.array_equal(z_again, z)
    with pytest.raises(EquilibriumNotFound) as exc:
        newton(model, max_iter=1)
    assert exc.value.z_last is not None and exc.value.z_last.shape == (6,) and exc.value.residual_last > 1e-12
    assert isinstance(exc.value, CondensedSolveError)
    with pytest.raises(ValueError):
        newton(model, z0=np.zeros(5))
    # A failed solve never returns converged=True.
    with pytest.raises(EquilibriumNotFound):
        solve_condensed(synth_calib, w, method="newton", max_iter=1)
    with pytest.raises(ValueError):
        solve_condensed(synth_calib, w, method="broyden")


def test_continuation_stages_and_stall(synth_calib):
    w = build_tariff_wedges(synth_calib, 1.0, importer="USA")
    z_direct, path_direct, _ = continuation(synth_calib, w)
    assert len(path_direct) == 1 and path_direct[0]["lam"] == 1.0
    # Five Newton iterations per stage are not enough for lambda = 1 directly: bisection and the
    # secant predictor still reach lambda = 1 with every accepted stage at 1e-12.
    z, path, model = continuation(synth_calib, w, max_iter=5)
    assert len(path) > 1 and path[-1]["lam"] == 1.0 and path[-1]["ok"]
    assert all(p["max_scaled_residual"] <= 1e-12 for p in path if p["ok"])
    assert any(not p["ok"] for p in path)
    assert np.max(np.abs(z - z_direct)) <= 1e-9
    with pytest.raises(ContinuationFailure) as exc:
        continuation(synth_calib, w, max_iter=1, min_step=0.1)
    assert exc.value.lam_reached < 1.0 and exc.value.z_last is not None


def test_nested_fallback_after_continuation_failure(synth_calib, monkeypatch):
    w = us10(synth_calib)
    direct = solve_condensed(synth_calib, w)

    def fail(*args, **kwargs):
        raise ContinuationFailure("simulated", lam_reached=0.3)

    monkeypatch.setattr(solve_mod, "continuation", fail)
    res = solve_condensed(synth_calib, w)
    assert res.converged and res.passed and res.path[-1].get("fallback") == "nested"
    assert res.path[-1]["continuation_lam_reached"] == 0.3
    assert np.max(np.abs(res.state.z - direct.state.z)) <= 1e-9


def test_arclength_monitors_landing_and_fold_detection(synth_calib, monkeypatch):
    w = build_tariff_wedges(synth_calib, 1.0, importer="USA")
    model = CondensedLeontiefModel(synth_calib, w)
    z_direct, _ = newton(model)
    history = arclength(model, z_direct, landing_tol=1e-10)
    # The last predictor aims at lambda = 1; the corrector moves it by second order only.
    assert len(history) >= 2 and abs(history[-1]["lam"] - 1.0) < 0.05
    assert all(history[i]["lam"] < history[i + 1]["lam"] for i in range(len(history) - 1))
    for step in history:
        assert step["dlam_ds"] > 0 and step["sign_det"] != 0 and step["sigma_min_rel"] > 1e-4
    with pytest.raises(ContinuationFailure):
        arclength(model, z_direct, max_steps=1)
    with pytest.raises(CondensedSolveError):
        arclength(model, z_direct + 1e-3, landing_tol=1e-12)
    # The determinant-sign monitor: det J_z is evaluated once at lambda = 0 and once per accepted
    # step; a sign change reported at the second accepted step must raise FoldDetected there.
    real_det = solve_mod.la.det
    calls = {"n": 0}

    def flipped_det(J, *args, **kwargs):
        calls["n"] += 1
        value = real_det(J, *args, **kwargs)
        return -value if calls["n"] == 3 else value

    monkeypatch.setattr(solve_mod.la, "det", flipped_det)
    with pytest.raises(FoldDetected) as exc:
        arclength(model, z_direct)
    assert exc.value.lam == pytest.approx(history[1]["lam"], rel=1e-6) and exc.value.sigma_min > 0
    assert exc.value.z_last is not None and calls["n"] == 3


def test_existence_boundary_fails_closed_on_the_synthetic_economy(synth_calib):
    """Fail-closed regression at an existence boundary, not a turning point. A 500% USA goods duty passes
    the existence gate, but the equilibrium with a positive U.S. wage ceases to exist near a 259% duty
    (lambda* = 0.51773 of 500%): along the branch lambda rises monotonically while the U.S. wage tends to
    zero. Natural-parameter continuation stalls at lambda = 0.515625, pseudo-arclength raises FoldDetected
    where the Jacobian is numerically singular (the determinant-sign monitor flips), and the driver ends in
    EquilibriumNotFound naming the stall (never converged=True)."""
    w5 = build_tariff_wedges(synth_calib, 5.0, importer="USA")
    assert existence_gate(synth_calib, w5).passed
    with pytest.raises(ContinuationFailure) as stall:
        continuation(synth_calib, w5)
    assert stall.value.lam_reached == pytest.approx(0.515625, abs=1e-12) and stall.value.z_last is not None
    model = CondensedLeontiefModel(synth_calib, w5)
    with pytest.raises(FoldDetected) as fold:
        arclength(model, model.z0, max_steps=400)
    assert fold.value.lam == pytest.approx(0.5177, rel=1e-3) and fold.value.lam > stall.value.lam_reached
    assert fold.value.sigma_min < 1e-8 and fold.value.z_last is not None and fold.value.z_last.shape == (6,)
    assert np.exp(fold.value.z_last[synth_calib.index("USA")]) < 1e-4  # the U.S. wage is collapsing
    with pytest.raises(EquilibriumNotFound, match="continuation stalled at lambda=0.515625"):
        solve_condensed(synth_calib, w5)


def test_multistart_near_and_multiple_equilibria(synth_calib, monkeypatch):
    w = us10(synth_calib)
    model = CondensedLeontiefModel(synth_calib, w)
    z_star, _ = newton(model)
    out = multistart_near(model, z_star, n_starts=5, seed=1)
    assert len(out) == 5 and all(r["converged"] and r["dist"] <= 1e-9 for r in out)
    assert multistart_near(model, z_star, n_starts=0) == []

    def other_root(model_, z0=None, **kwargs):
        return z_star + 1e-3, {"iterations": 1}

    monkeypatch.setattr(solve_mod, "newton", other_root)
    with pytest.raises(MultipleEquilibria) as exc:
        multistart_near(model, z_star, n_starts=1)
    assert len(exc.value.roots) == 2


def test_far_start_newton_fails_and_nested_fallback_recovers(bundled_calib):
    """Ported from the IO adversarial suite on the 77x11 table: sigma_w = 0.6, sigma_Y = 0.35 far starts."""
    model = CondensedLeontiefModel(bundled_calib, us10(bundled_calib))
    z_star, _ = newton(model)
    rng = np.random.default_rng(42)
    n = 77
    z0_near = np.concatenate([rng.normal(0, 0.10, n), np.maximum(0.5, 1.0 + rng.normal(0, 0.05, n))])
    z_near, _ = newton(model, z0=z0_near, max_iter=30)
    assert np.max(np.abs(z_near - z_star)) <= 1e-9
    z0_far = np.concatenate([rng.normal(0, 0.60, n), np.maximum(0.25, 1.0 + rng.normal(0, 0.35, n))])
    with pytest.raises(EquilibriumNotFound):
        newton(model, z0=z0_far, max_iter=30)
    z_nested, _ = nested_fallback(model, z0=z0_far, max_iter=30)
    assert np.max(np.abs(z_nested - z_star)) <= 1e-9


# ---------------------------------------------------------------------------
# Baseline duties, final-demand rules, inventory treatment
# ---------------------------------------------------------------------------

def test_separate_baseline_tariffs(synth_calib):
    w = us10(synth_calib)
    cb = separate_baseline_tariffs(synth_calib, w)
    assert cb.gates["explicit_tariff_cost_error"] <= 1e-11 and cb.gates["explicit_tariff_quantity_error"] <= 1e-11
    assert cb.baseline_tariffs is not None and "explicit baseline duties" in cb.summary()
    frame = cb.baseline_tariffs.to_dataframe()
    assert frame.shape == (3, 8) and frame.index.tolist() == list(CODES3) and "TR0" in cb.baseline_tariffs.to_markdown()
    assert "benchmark duty" in cb.baseline_tariffs.summary() and cb.baseline_tariffs.metadata["tol"] == 1e-11
    table = synth_calib.table
    duty = ((w.tau - 1) * table.Z).sum() + ((w.tau_fd - 1) * table.F[:, :, :3]).sum() + ((w.tau_V - 1) * table.F[:, :, 3]).sum()
    assert cb.TR0[0] == pytest.approx(duty, rel=1e-12) and cb.TR0[1] == 0.0 and cb.TR0[2] == 0.0
    np.testing.assert_allclose(cb.T0 + cb.TR0, synth_calib.T0, rtol=1e-12)
    np.testing.assert_allclose(cb.P0[:, 1:], 1.0, atol=1e-15)
    assert np.all(cb.P0[[0, 1], 0] > 1.0) and cb.P0[2, 0] == pytest.approx(1.0, abs=1e-15)
    # Levels equal to the baseline levels reproduce the benchmark and certify.
    model = CondensedLeontiefModel(cb, w)
    st = model.state(model.z0)
    assert st.max_scaled_residual <= 1e-14 and abs(st.walras_residual) <= 1e-14
    assert certify_raw_flows(cb, w, st).max_block <= 1e-12
    res = solve_condensed(cb, w)
    assert res.iterations == 0 and res.certificate.passed
    assert compute_measures(cb, w, res.state, country="USA").ev_pct_base_gdp == pytest.approx(0.0, abs=1e-12)
    # Raising the duty to 20% from a 10% baseline is a smaller shock than 20% from zero.
    w20 = build_tariff_wedges(synth_calib, 0.20, importer="USA")
    from_base = compute_measures(cb, w20, solve_condensed(cb, w20).state, country="USA")
    from_zero = compute_measures(synth_calib, w20, solve_condensed(synth_calib, w20).state, country="USA")
    assert 0 < from_base.pc_over_w_pct < from_zero.pc_over_w_pct
    total, direct, cascade = pe_first_order_decomposition(cb, w20, country="USA")
    assert total == pytest.approx(direct + cascade) and 0 < total < pe_first_order_decomposition(synth_calib, w20, country="USA")[0]
    with pytest.raises(CalibrationError, match="already been separated"):
        separate_baseline_tariffs(cb, w)
    bad = TariffWedges.from_arrays(w.tau, w.tau_fd, w.tau_V)
    tau_low = w.tau.copy()
    tau_low[0, 4] = 0.9
    # TariffWedges refuses multipliers below one; the calibration gate also refuses any duck-typed levels.
    with pytest.raises(DataIntegrityError, match=r"tau\[0, 4\] = 0.9 is below one"):
        TariffWedges.from_arrays(tau_low, w.tau_fd, w.tau_V)
    with pytest.raises(CalibrationError, match="gross benchmark"):
        separate_baseline_tariffs(synth_calib, SimpleNamespace(tau=tau_low, tau_fd=w.tau_fd, tau_V=w.tau_V))
    assert separate_baseline_tariffs(synth_calib, bad).TR0[0] == pytest.approx(cb.TR0[0])


def test_fixed_investment_rule_and_inventory_treatments(synth_calib):
    calib = synth_calib
    free = TariffWedges.free_trade(calib)
    mf = CondensedLeontiefModel(calib, free, fd_rule="fixed_investment")
    assert mf.state(mf.z0).max_scaled_residual <= 1e-14
    z = mf.z0.copy()
    z[:3] = np.linspace(-.03, .04, 3)
    z[3:] *= 1.1
    st = mf.state(z)
    np.testing.assert_allclose(st.q[1], calib.B0[:, 1], rtol=1e-14)
    np.testing.assert_allclose(st.E.sum(0), st.A_tilde, rtol=1e-14)
    w = us10(calib)
    rf = solve_condensed(calib, w, fd_rule="fixed_investment")
    assert rf.passed and rf.certificate.max_block <= 1e-10
    ru = solve_condensed(calib, w, inventories="untariffed")
    rt = solve_condensed(calib, w)
    assert ru.passed and ru.state.TR[0] < rt.state.TR[0]
    # The effective rate follows the solve's inventory treatment: untariffed inventories are not dutiable.
    _, eff_u = tariff_revenue_and_effective_rate(calib, w, ru.state, country="USA")
    assert eff_u == pytest.approx(10.0, abs=1e-10)
    mu = compute_measures(calib, w, ru.state, country="USA")
    assert mu.effective_rate_dutiable_pct == pytest.approx(10.0, abs=1e-10)
    assert mu.metadata["effective_rate_inventories"] == "untariffed"
    w_u = build_tariff_wedges(calib, 0.10, importer="USA", inventories="untariffed")
    ru2 = solve_condensed(calib, w_u)
    assert np.max(np.abs(ru2.state.z - ru.state.z)) <= 1e-12
    assert tariff_revenue_and_effective_rate(calib, w_u, ru2.state, country="USA")[1] == pytest.approx(eff_u, abs=1e-10)
    # An explicit treatment overrides the state's: counting inventories that bore no duty dilutes the rate.
    _, eff_forced = tariff_revenue_and_effective_rate(calib, w, ru.state, country="USA", inventories="tariffed")
    assert eff_forced < 10.0 - 1e-3
    assert tariff_revenue_and_effective_rate(calib, w, rt.state, country="USA")[1] == pytest.approx(10.0, abs=1e-10)
    with pytest.raises(ValueError, match="inventories"):
        tariff_revenue_and_effective_rate(calib, w, ru.state, country="USA", inventories="merged")
    with pytest.raises(CertificationFailure):
        certify_raw_flows(calib, w, ru.state, inventories="tariffed")
    with pytest.raises(UnsupportedExtension, match="table-level"):
        CondensedLeontiefModel(calib, w, inventories="merged")
    for kw in ({"closure": "trade"}, {"fd_rule": "invest"}, {"inventories": "none"}):
        with pytest.raises(UnsupportedExtension):
            CondensedLeontiefModel(calib, w, **kw)
    with pytest.raises(TypeError):
        CondensedLeontiefModel(calib, w.tau)


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

def test_result_objects_are_frozen_and_exportable(synth_calib):
    w = us10(synth_calib)
    res = solve_condensed(synth_calib, w)
    assert res.z is res.state.z and not res.state.z.flags.writeable and not res.state.p.flags.writeable
    for obj in (res, res.state, res.certificate, synth_calib.table.report, synth_calib.table):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, "metadata", {})
        assert isinstance(obj.to_markdown(), str) and "\\begin{tabular}" in obj.to_latex() and "#table" in obj.to_typst()
    frame = res.to_dataframe()
    assert list(frame.columns) == ["w", "Y_hat", "XN", "T", "TR", "PT", "pc_over_w_pct", "ev_pct_gdp", "tot_pct"]
    assert frame.index.tolist() == list(CODES3)
    assert set(res.pc_over_w_pct) == set(CODES3) and res.pc_over_w_pct["USA"] == pytest.approx(7.0518197, abs=1e-6)
    assert "certificate max block" in res.summary() and res.state.to_dataframe(CODES3).shape == (3, 12)
    assert isinstance(synth_calib.table.report, TableBuildReport)
    assert res.state.metadata["closure"] == "factor"
    labelled = res.state.to_markdown(CODES3)
    assert "USA" in labelled and "ROW" in res.state.to_latex(country_codes=CODES3) and "CHN" in res.state.to_typst(CODES3)
    # TariffWedges: the nonzero rates, labelled from the registries recorded at build time.
    rates = w.to_dataframe()
    assert list(rates.columns) == ["origin", "importer", "sector", "rate"] and len(rates) == 4
    assert set(rates["importer"]) == {"USA"} and set(rates["origin"]) == {"CHN", "ROW"}
    assert set(rates["sector"]) == {"A01_02", "C24"} and np.allclose(rates["rate"], 0.10)
    assert "4 nonzero" in w.summary() and "A01_02" in w.to_markdown()
    assert "\\begin{tabular}" in w.to_latex() and "#table" in w.to_typst()
    assert TariffWedges.free_trade(synth_calib).to_dataframe().empty
    assert w.to_dataframe(country_codes=("a", "b", "c"), sector_codes=("s", "t", "u", "v"))["origin"].tolist()[0] in ("b", "c")
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(w, "metadata", {})


# ---------------------------------------------------------------------------
# Parity with the vendored IO engine (subprocess with the vendor path first)
# ---------------------------------------------------------------------------

_IO_REFERENCE = """
import json, os, sys
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2'); os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import numpy as np
import puremacro
assert 'vendor' in puremacro.__file__, puremacro.__file__
from puremacro.trade.corrected import (table_from_arrays, calibrate_corrected, CorrectedTariffScenario,
                                       build_corrected_tensors, solve_corrected, compute_corrected_measures,
                                       ev_split, pe_first_order_decomposition)
from puremacro.trade.corrected.calibration import separate_baseline_tariffs
spec = json.load(open(sys.argv[2]))
arr = {k: np.array(spec[k]) for k in ('Z', 'F', 'VA', 'TLS', 'TFD')}
table, _ = table_from_arrays(arr['Z'], arr['F'], arr['VA'], arr['TLS'], arr['TFD'],
                             country_codes=spec['countries'], sector_codes=spec['sectors'],
                             output=np.array(spec['output']), fd_layout=spec['fd_layout'],
                             reference_country=spec['reference'])
calib = calibrate_corrected(table, merchandise_mask=np.array(spec['goods'], dtype=bool),
                            allow_empty_purchases_abroad=spec['allow_empty_x'])
out = {}
for name, sc in spec['scenarios'].items():
    opts = spec.get('options', {}).get(name, {})
    c = calib
    if opts.get('baseline'):
        base = CorrectedTariffScenario(id='base', label='base', episode='parity', **opts['baseline'])
        c = separate_baseline_tariffs(calib, build_corrected_tensors(calib, base))
    scen = CorrectedTariffScenario(id=name, label=name, episode='parity', **sc)
    res = solve_corrected(c, scen, method='newton', closure=opts.get('closure', 'factor'),
                          numeraire=opts.get('numeraire', 'factor'), fd_rule=opts.get('fd_rule', 'absorption'),
                          invnt=opts.get('invnt', 'tariffed'))
    meas = compute_corrected_measures(c, res.tensors, res.state, country='USA')
    split = ev_split(c, res.tensors, res.state, country='USA')
    out[name] = {'z': res.state.z.tolist(), 'w': res.state.w.tolist(), 'Y': res.state.Y.tolist(),
                 'p': res.state.p.tolist(), 'y': res.state.y.tolist(), 'T': res.state.T.tolist(),
                 'TR': res.state.TR.tolist(), 'PT': res.state.PT.tolist(), 'XN': res.state.XN.tolist(),
                 'headline': meas['headline'], 'ev_split': split.to_dict(), 'pe': meas['pe_objects'],
                 'world_ev': meas['aggregates']['world_EV_pct_world_income'],
                 'first_order': list(pe_first_order_decomposition(c, res.tensors)),
                 'certificate': res.certificate, 'iterations': res.iterations,
                 'tau_sum': float(res.tensors.tau.sum()), 'tau_fd_sum': float(res.tensors.tau_fd.sum()),
                 'tau_V_sum': float(res.tensors.tau_V.sum()),
                 'ev_all': {c: float(v) for c, v in res.ev.items()},
                 'tot_all': {c: float(v) for c, v in res.tot.items()},
                 'pi_c_all': {c: float(v) for c, v in res.pi_c.items()}}
json.dump(out, open(sys.argv[3], 'w'))
print('REFERENCE_OK')
"""


def _io_reference(tmp_path: Path, spec: dict) -> dict:
    """Run the vendored IO engine in a subprocess (never in-process: it is another ``puremacro``)."""
    if not (VENDOR / "puremacro" / "trade" / "corrected" / "solve.py").exists():
        pytest.skip("IO research volume not mounted")
    spec_path = tmp_path / "spec.json"
    ref_path = tmp_path / "ref.json"
    spec_path.write_text(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in spec.items()}), encoding="utf-8")
    proc = subprocess.run([sys.executable, "-c", _IO_REFERENCE, str(VENDOR), str(spec_path), str(ref_path)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0 and "REFERENCE_OK" in proc.stdout, proc.stderr[-3000:]
    return json.loads(ref_path.read_text(encoding="utf-8"))


_HEADLINE_KEYS = {"pc_over_w_pct": "PC_over_w_pct", "ev_pct_base_gdp": "EV_pct_base_gdp",
                  "tot_fisher_pct": "ToT_fisher_pct", "rgdp_base_prices_pct": "RGDP_base_prices_pct",
                  "tariff_revenue_share_pct": "tariff_revenue_share_Y_pct",
                  "effective_rate_dutiable_pct": "effective_rate_on_dutiable_imports_pct",
                  "real_national_income_pct": "real_national_income_pct"}
_SPLIT_KEYS = {"factor_income_pp": "split_factor_income_pp", "revenue_pp": "split_revenue_pp",
               "tariff_revenue_pp": "split_tariff_revenue_pp", "other_taxes_pp": "split_other_taxes_pp",
               "deficit_inventory_pp": "split_deficit_inventory_pp"}


def _assert_parity(calib: CondensedCalibration, wedges: dict, ref: dict, tol: float = 1e-12,
                   options: dict | None = None) -> dict:
    """Compare the ported solver with the IO reference; returns the observed maximum differences.

    ``options[name]`` may hold ``calib`` (a baseline-separated calibration) and ``solve_condensed``
    keyword arguments (closure, numeraire, fd_rule, inventories) matching the IO options of that case.
    """
    observed = {}
    for name, w in wedges.items():
        r = ref[name]
        opts = dict((options or {}).get(name, {}))
        c = opts.pop("calib", calib)
        assert abs(w.tau.sum() - r["tau_sum"]) < 1e-9 and abs(w.tau_fd.sum() - r["tau_fd_sum"]) < 1e-9
        assert abs(w.tau_V.sum() - r["tau_V_sum"]) < 1e-9
        res = solve_condensed(c, w, method="newton", **opts)
        assert res.iterations == r["iterations"]
        d = {}
        for key in ("z", "w", "p"):
            d[key] = float(np.max(np.abs(getattr(res.state, key) - np.array(r[key]))))
        for key in ("Y", "y", "T", "TR", "PT", "XN"):
            refv = np.array(r[key])
            d[key] = float(np.max(np.abs(getattr(res.state, key) - refv) / np.maximum(1.0, np.abs(refv))))
        m = compute_measures(c, w, res.state, country="USA")
        hl = m.headline()
        d["headline"] = max(abs(hl[k] - r["headline"][v]) for k, v in _HEADLINE_KEYS.items())
        sp = m.ev_split.to_dict()
        d["ev_split"] = max(abs(sp[k] - r["ev_split"][v]) for k, v in _SPLIT_KEYS.items())
        d["pe"] = max(abs(m.pc_pe_pct - r["pe"]["PC_PE_pct"]),
                      abs(m.ge_minus_pe_pct - r["pe"]["GE_minus_PE_relative_wage_term_pct"]))
        d["world_ev"] = abs(m.world_ev_pct - r["world_ev"])
        fo = pe_first_order_decomposition(c, w, country="USA")
        d["first_order"] = max(abs(a - b) for a, b in zip(fo, r["first_order"]))
        d["ev_all"] = max(abs(res.ev_pct_gdp[c] / 100 - v) for c, v in r["ev_all"].items())
        d["tot_all"] = max(abs(res.tot_pct[c] / 100 - v) for c, v in r["tot_all"].items())
        d["pi_c_all"] = max(abs(res.pc_over_w_pct[c] / 100 - v) for c, v in r["pi_c_all"].items())
        d["certificate"] = max(abs(res.certificate.blocks[k] - v) for k, v in r["certificate"].items())
        assert max(d.values()) <= tol, (name, d)
        observed[name] = d
    return observed


def test_parity_with_io_engine_synthetic(tmp_path, synth_calib):
    a = synthetic_arrays()
    spec = {**{k: a[k] for k in ("Z", "F", "VA", "TLS", "TFD", "output")}, "countries": list(CODES3),
            "sectors": list(SECTORS4), "fd_layout": "icio6", "reference": "ROW",
            "goods": [True, True, False, False], "allow_empty_x": False,
            "scenarios": {"us10": {"default_rate": 0.10}, "big": {"default_rate": 1.0},
                          "recip": {"rates": {"USA": {"CHN": 0.10}, "CHN": {"USA": 0.10}}},
                          "base15": {"default_rate": 0.15}, "gdp_wrow": {"default_rate": 0.25},
                          "fixinv": {"default_rate": 0.25}, "untar": {"default_rate": 0.25, "invnt": "untariffed"}},
            # Options the IO runner passes to solve_corrected (and the baseline it separates first).
            "options": {"base15": {"baseline": {"default_rate": 0.05}},
                        "gdp_wrow": {"closure": "gdp", "numeraire": "w_ROW"},
                        "fixinv": {"fd_rule": "fixed_investment"}, "untar": {"invnt": "untariffed"}}}
    ref = _io_reference(tmp_path, spec)
    based = separate_baseline_tariffs(synth_calib, build_tariff_wedges(synth_calib, 0.05, importer="USA"))
    wedges = {"us10": us10(synth_calib), "big": build_tariff_wedges(synth_calib, 1.0, importer="USA"),
              "recip": build_tariff_wedges(synth_calib, {"USA": {"CHN": 0.10}, "CHN": {"USA": 0.10}}),
              "base15": build_tariff_wedges(synth_calib, 0.15, importer="USA"),
              "gdp_wrow": build_tariff_wedges(synth_calib, 0.25, importer="USA"),
              "fixinv": build_tariff_wedges(synth_calib, 0.25, importer="USA"),
              "untar": build_tariff_wedges(synth_calib, 0.25, importer="USA", inventories="untariffed")}
    options = {"base15": {"calib": based}, "gdp_wrow": {"closure": "gdp", "numeraire": "w_ROW"},
               "fixinv": {"fd_rule": "fixed_investment"}, "untar": {"inventories": "untariffed"}}
    observed = _assert_parity(synth_calib, wedges, ref, tol=1e-12, options=options)
    assert set(observed) == set(wedges)
    assert all(v <= 1e-12 for d in observed.values() for v in d.values())


def test_parity_with_io_engine_oecd_fixture(tmp_path, oecd3_calib):
    a = oecd3_arrays()
    spec = {**{k: a[k] for k in ("Z", "F", "VA", "TLS", "TFD", "output")}, "countries": list(a["countries"]),
            "sectors": list(a["sectors"]), "fd_layout": "icio6", "reference": "REST",
            "goods": [True, True, False], "allow_empty_x": False,
            "scenarios": {"us10": {"default_rate": 0.10},
                          "recip": {"rates": {"USA": {"*": 0.10}, "CHN": {"USA": 0.10}, "REST": {"USA": 0.10}}}}}
    ref = _io_reference(tmp_path, spec)
    wedges = {"us10": us10(oecd3_calib),
              "recip": build_tariff_wedges(oecd3_calib, {"USA": {"*": 0.10}, "CHN": {"USA": 0.10}, "REST": {"USA": 0.10}})}
    _assert_parity(oecd3_calib, wedges, ref, tol=1e-12)


def test_parity_with_io_engine_bundled_77x11(tmp_path, legacy_calib, bundled_calib):
    data = legacy_calib.data_calibra
    M, nc = 847, 77
    F3 = data[:M, M:].reshape(M, nc, 3)
    F4 = np.zeros((M, nc, 4))
    F4[:, :, 0] = F3[:, :, 0]
    F4[:, :, 1] = np.maximum(F3[:, :, 1], 0.0)
    F4[:, :, 3] = np.minimum(F3[:, :, 1], 0.0)
    F4[:, :, 2] = F3[:, :, 2]
    T4 = np.zeros((nc, 4))
    T4[:, :3] = data[M, M:].reshape(nc, 3)
    spec = {"Z": data[:M, :M], "F": F4, "VA": data[M + 1, :M] + data[M + 2, :M], "TLS": data[M, :M], "TFD": T4,
            "output": data[:M, :M].sum(1) + F3.sum((1, 2)), "countries": list(legacy_calib.country_codes),
            "sectors": list(legacy_calib.sector_codes), "fd_layout": "cgxv", "reference": "ROW",
            "goods": [c in ("AGRI", "MINQ", "MANU") for c in legacy_calib.sector_codes], "allow_empty_x": True,
            "scenarios": {"us10": {"default_rate": 0.10}}}
    ref = _io_reference(tmp_path, spec)
    _assert_parity(bundled_calib, {"us10": us10(bundled_calib)}, ref, tol=1e-12)
