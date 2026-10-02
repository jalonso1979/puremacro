"""Regression tests for the 2026-09-22 audit of MRIO ingestion and regularization.

Each test pins one confirmed finding of the ``ingestion-regularize`` and
``solvers-recovery`` audit dimensions (cluster DATA of the post-4.3.0 fix round):

1. ``compute_spectral_radius`` shifted its power iteration by the maximum row
   sum, so the Collatz-Wielandt bounds stayed unresolved on real tables.
2. The absolute 1 M USD value-added floor lifted value added above gross output
   on tiny nodes and the adjustments carried no node labels.
3. FIGARO/EXIOBASE/WIOD/Eora loaders and the OECD synthetic branches put the
   foreign-balance closure on final-use index 1 = NPISH instead of investment.
4. FIGARO final-use columns were read by position.
5. ``load_raw_45sector_icio`` floored value added at 5% of sales while its
   docstring said 1e-4, without any record.
6. The corrupted ``data_2020_SML.csv`` export was the first default candidate
   of ``load_raw_45sector_icio`` and was read without a checksum.
7. ``validate_accounting_identities`` did not perform its documented VA check,
   ``regularize_mrio_table`` accepted a ``Y`` inconsistent with row sales
   silently, and RAS/GRAS gave no convergence signal.
8. ``load_exiobase`` failed on native archives with pandas errors.

Review round 2 adds: large periodic blocks (second power pass with a wide
shift), RAS/GRAS convergence at the floating-point floor of large margins,
compressed harmonized files, the ``final_use_semantics`` record, the
closure-category warning of ``package_mrio_to_calibration_result`` and
positive file-branch tests for EXIOBASE, WIOD and Eora.

Tests needing ``/Volumes/BIGDATA/Research/IO`` skip when it is absent.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.trade import _accounting
from puremacro.trade import data as trade_data
from puremacro.trade._oecd_icio import condense_final_demand, read_native
from puremacro.trade.calibration import calibrate_trade_model
from puremacro.trade.data import (
    FIGARO_FD_CATEGORIES,
    ICIOData,
    generate_synthetic_mrio,
    load_icio_data,
    load_raw_45sector_icio,
    package_mrio_to_calibration_result,
)
from puremacro.trade.regularize import (
    balance_gras,
    balance_ras,
    compute_spectral_radius,
    regularize_mrio_table,
    validate_accounting_identities,
)

# The IO research workspace; parity tests skip when it is absent. Override with PUREMACRO_IO_ROOT.
IO_ROOT = Path(os.environ.get("PUREMACRO_IO_ROOT", "/Volumes/BIGDATA/Research/IO"))
OECD_2019 = IO_ROOT / "ICIOextended" / "2019_SML.csv"
CORRUPTED_2020 = IO_ROOT / "computation" / "7_TIO_77c_vf" / "data_2020_SML.csv"
needs_oecd_2019 = pytest.mark.skipif(not OECD_2019.is_file(), reason="OECD 2019_SML.csv not available")


def _eig_radius(B: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(B))))


def _assert_brackets(bounds, truth, rel=1e-12):
    rho, lower, upper = bounds
    slack = rel * max(1.0, abs(truth))
    assert lower - slack <= truth <= upper + slack
    assert lower - slack <= rho <= upper + slack


# ---------------------------------------------------------------------------
# 1. Collatz-Wielandt shift
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bundled_calibration():
    return calibrate_trade_model(load_icio_data(), ns=11, nc=77, nfd=3)


def _bundled_cost_matrix(calib, rate: float) -> np.ndarray:
    """B_tau = a * (1 + rate on cross-border blocks) / (1 - t), as in check_hawkins_simon_viability."""
    nc, ns = calib.n_countries, calib.n_sectors
    M = nc * ns
    a_2d = np.asarray(calib.a).reshape((M, M), order="F")
    tax = np.clip(np.asarray(calib.tax, dtype=float).flatten(order="F"), -0.9, 0.999)
    country = np.repeat(np.arange(nc), ns)
    mult = np.where(country[:, None] != country[None, :], 1.0 + rate, 1.0)
    return np.maximum(a_2d * mult / np.maximum(1.0 - tax, 1e-12)[None, :], 0.0)


@pytest.mark.parametrize("rate", [3.8, 5.0, 9.0, 10.0])
def test_bundled_prohibitive_tariffs_are_certified_non_viable_fast(bundled_calibration, rate):
    """4.3.0: bounds [0.004, 1.698] at rate 9 after 1000 iterations -> 'unresolved'."""
    B = _bundled_cost_matrix(bundled_calibration, rate)
    elapsed = []
    for _ in range(3):
        t0 = time.perf_counter()
        bounds = compute_spectral_radius(B, max_iter=50, tol=1e-12)
        elapsed.append(time.perf_counter() - t0)
    assert bounds[1] >= 1.0, f"lower bound {bounds[1]} does not certify the violation"
    assert min(elapsed) < 1.0
    if rate in (3.8, 9.0):
        truth = _eig_radius(B)
        _assert_brackets(bounds, truth, rel=1e-9)
        assert bounds[2] - bounds[1] < 1e-6 * truth


def test_bundled_subcritical_rate_is_certified_viable(bundled_calibration):
    B = _bundled_cost_matrix(bundled_calibration, 3.65)
    rho, lower, upper = compute_spectral_radius(B, max_iter=50, tol=1e-12)
    truth = _eig_radius(B)
    assert upper < 1.0 - 1e-6
    _assert_brackets((rho, lower, upper), truth, rel=1e-9)
    assert abs(rho - truth) < 1e-6


@pytest.mark.parametrize("B", [
    np.array([[0.0, 2.0], [0.2, 0.0]]),                                  # period 2
    np.roll(np.eye(3), 1, axis=1) * 0.8,                                  # period 3
    np.array([[0.5, 1.0, 0.0], [0.0, 0.2, 0.3], [0.0, 0.1, 0.8]]),       # reducible
    np.array([[0.0, 0.0, 0.0], [0.0, 0.4, 0.2], [0.0, 0.3, 0.1]]),       # isolated zero node
])
def test_synthetic_small_matrices_match_eigvals(B):
    bounds = compute_spectral_radius(B, max_iter=60, tol=1e-10)
    truth = _eig_radius(B)
    assert abs(bounds[0] - truth) < 1e-8
    _assert_brackets(bounds, truth, rel=1e-10)
    assert bounds[2] - bounds[1] < 1e-8


def test_periodic_block_above_eigvals_cutoff_is_resolved():
    rng = np.random.default_rng(1)
    B = np.zeros((300, 300))
    B[:150, 150:] = rng.uniform(0, 1, (150, 150)) * (rng.uniform(0, 1, (150, 150)) < 0.1)
    B[150:, :150] = rng.uniform(0, 1, (150, 150)) * (rng.uniform(0, 1, (150, 150)) < 0.1)
    B *= 0.9 / _eig_radius(B)
    bounds = compute_spectral_radius(B, max_iter=300, tol=1e-10)
    _assert_brackets(bounds, 0.9, rel=1e-10)
    assert abs(bounds[0] - 0.9) < 1e-8
    assert bounds[2] - bounds[1] < 1e-8


def _bipartite(n: int, rho_target: float, *, density: float = 1.0, spread: bool = False, seed: int = 0):
    """Period-2 block B = [[0, P], [Q, 0]] scaled to spectral radius ``rho_target``.

    rho(B)^2 = rho(P Q), so the truth needs only an (n/2) x (n/2) eigensolve.
    """
    rng = np.random.default_rng(seed)
    h = n // 2
    P = rng.uniform(0, 1, (h, n - h)) * (rng.uniform(0, 1, (h, n - h)) < density)
    Q = rng.uniform(0, 1, (n - h, h)) * (rng.uniform(0, 1, (n - h, h)) < density)
    if spread:
        P *= rng.uniform(0.2, 5.0, (h, 1))
        Q *= rng.uniform(0.2, 5.0, (n - h, 1))
    rho = np.sqrt(_eig_radius(P @ Q))
    B = np.zeros((n, n))
    B[:h, h:] = P
    B[h:, :h] = Q
    return B * (rho_target / rho)


@pytest.mark.parametrize("n, density, spread, target", [
    (1100, 1.0, True, 0.9),       # round 1: [0.89785, 0.90215], 4.3.0 exact
    (1600, 0.05, False, 0.9985),  # round 1: [0.99776628, 0.99923426] -> 'unresolved'
])
def test_periodic_block_above_resolvent_cutoff_is_resolved(n, density, spread, target):
    """Blocks above 1024 rows get no inverse iteration; a second, wide-shift pass resolves them."""
    B = _bipartite(n, target, density=density, spread=spread)
    bounds = compute_spectral_radius(B, max_iter=300, tol=1e-10)
    _assert_brackets(bounds, target, rel=1e-10)
    assert bounds[2] - bounds[1] < 1e-8


def test_large_periodic_productive_table_is_certified():
    """Round 1 raised 'Spectral viability unresolved' on this productive table (rho 0.9985)."""
    A = _bipartite(1600, 0.9985, density=0.05)
    n = A.shape[0]
    Y = np.linalg.solve(np.eye(n) - A, np.full(n, 10.0))
    Z = A * Y[None, :]
    F = (Y - Z.sum(axis=1)).reshape(n, 1)
    assert np.all(F > 0)
    VA = 0.5 * (Y - Z.sum(axis=0))
    TLS = Y - Z.sum(axis=0) - VA
    *_, report = regularize_mrio_table(Z, F, VA, TLS, None, n_countries=1, n_sectors=n, return_report=True)
    spectral = report["spectral"]
    assert spectral["collatz_wielandt_upper"] < 0.999
    assert spectral["collatz_wielandt_lower"] <= 0.9985 <= spectral["collatz_wielandt_upper"]


def _weakly_coupled_near_critical(rho_target: float, seed: int = 0) -> np.ndarray:
    """Four dense blocks with weak coupling (lambda_2 ~ 0.87 rho) and one row summing to ~15."""
    rng = np.random.default_rng(seed)
    size, n_blocks = 100, 4
    n = size * n_blocks
    B = np.zeros((n, n))
    for b in range(n_blocks):
        blk = rng.uniform(0, 1, (size, size))
        B[b * size:(b + 1) * size, b * size:(b + 1) * size] = blk * (0.95 - 0.08 * b) / _eig_radius(blk)
    B += 1e-4 * rng.uniform(0, 1, (n, n))
    B[5, :] += 15.0 / n
    return B * (rho_target / _eig_radius(B))


def test_near_critical_productive_table_is_not_unresolved():
    """4.3.0: bounds [0.9198, 1.0033] -> 'Spectral viability unresolved' for rho = 0.998."""
    A = _weakly_coupled_near_critical(0.998)
    assert A.sum(axis=1).max() > 10.0
    bounds = compute_spectral_radius(A, max_iter=300, tol=1e-10)
    _assert_brackets(bounds, 0.998, rel=1e-10)
    assert bounds[2] - bounds[1] < 1e-8
    M = A.shape[0]
    Y = np.full(M, 100.0)
    Z = A * Y[None, :]
    F = np.maximum(Y - Z.sum(axis=1), 1.0).reshape(M, 1)
    Y = Z.sum(axis=1) + F.ravel()
    Z = A * Y[None, :]
    F = (Y - Z.sum(axis=1)).reshape(M, 1)
    VA = 0.3 * (Y - Z.sum(axis=0))
    TLS = Y - Z.sum(axis=0) - VA
    *_, report = regularize_mrio_table(Z, F, VA, TLS, None, n_countries=1, n_sectors=M, return_report=True)
    assert report["spectral"]["collatz_wielandt_upper"] < 0.999


def test_shift_keyword_is_validated():
    with pytest.raises(ValueError, match="shift"):
        compute_spectral_radius(np.eye(2) * 0.5, shift=0.0)
    with pytest.raises(ValueError, match="shift"):
        compute_spectral_radius(np.eye(2) * 0.5, shift=float("nan"))


@needs_oecd_2019
def test_real_2019_table_scaled_productive_is_certified():
    """4.3.0: regularize_mrio_table(tau=1.55) raised 'unresolved [0.0805, 0.99958]', true rho 0.983001."""
    raw = condense_final_demand(read_native(OECD_2019, 2019))
    *_, report = regularize_mrio_table(raw.Z, raw.F, raw.VA, raw.TLS, None, tau=np.full(raw.Z.shape, 1.55),
                                       n_countries=raw.C, n_sectors=raw.S, return_report=True)
    spectral = report["spectral"]
    assert spectral["collatz_wielandt_upper"] - spectral["collatz_wielandt_lower"] < 1e-8
    assert abs(spectral["rho"] - 0.983001) < 1e-6
    base = validate_accounting_identities(*regularize_mrio_table(raw.Z, raw.F, raw.VA, raw.TLS, None,
                                                                 n_countries=raw.C, n_sectors=raw.S))
    assert base["spectral_certified"]
    assert abs(base["spectral_radius"] - 0.634194) < 1e-5   # 4.3.0 reported 0.6039


# ---------------------------------------------------------------------------
# 2. Value-added floor never exceeds output; adjustments carry node labels
# ---------------------------------------------------------------------------


def _tiny_node_table():
    Z = np.array([[20.0, 5.0, 0.01], [5.0, 30.0, 0.02], [0.01, 0.02, 0.01]])
    F = np.array([[60.0, 10.0], [50.0, 15.0], [0.1, 0.06]])
    Y = Z.sum(axis=1) + F.sum(axis=1)                  # node 2 has Y = 0.2 M USD
    VA = np.array([40.0, 35.0, 0.05])
    TLS = Y - Z.sum(axis=0) - VA
    return Z, F, VA, TLS, Y


def test_absolute_va_floor_is_capped_at_half_of_output():
    """4.3.0 lifted VA to 1.0 on a node with Y = 0.2 (VA > Y) and recorded only counts."""
    Z, F, VA, TLS, Y = _tiny_node_table()
    Zc, Fc, VAc, TLSc, Yc, report = regularize_mrio_table(Z, F, VA, TLS, Y, n_countries=1, n_sectors=3,
                                                         return_report=True)
    assert VAc[2] == pytest.approx(0.5 * Y[2])
    assert np.all(VAc <= Yc)
    np.testing.assert_allclose(Zc.sum(axis=0) + VAc + TLSc, Yc, atol=1e-12)
    floor = report["va_floor"]
    assert floor["nodes"] == [2]
    assert floor["value_added_before"] == [pytest.approx(0.05)]
    assert floor["value_added_after"] == [pytest.approx(0.5 * Y[2])]
    assert floor["binding"] == ["absolute_capped"]
    # The ratio floor and the full absolute floor on larger nodes are unchanged.
    VA_neg = VA.copy()
    VA_neg[0] = -10.0
    VAc2 = regularize_mrio_table(Z, F, VA_neg, Y - Z.sum(axis=0) - VA_neg, Y, n_countries=1, n_sectors=3)[2]
    assert VAc2[0] == pytest.approx(max(1e-3 * Y[0], 1.0))


def test_package_records_floored_node_labels():
    raw = generate_synthetic_mrio("oecd", custom_c=2, custom_s=2, seed=3)
    VA = raw.VA.copy()
    VA[1] = -5.0
    raw.value_added[:] = VA
    raw.taxes_less_subsidies[:] = raw.Y - raw.Z.sum(axis=0) - VA
    calib = package_mrio_to_calibration_result(condense_final_demand(raw))
    nodes = calib.metadata["regularization"]["va_floored_nodes"]
    assert [n["node"] for n in nodes] == [f"{raw.countries[0]}_{raw.sectors[1]}"]
    assert nodes[0]["value_added_before"] == pytest.approx(-5.0)
    assert nodes[0]["value_added_after"] <= nodes[0]["gross_output"]
    assert all(set(v) == {"changed_entries", "max_abs_change", "net_change"}
               for v in calib.metadata["adjustments"].values())


@needs_oecd_2019
def test_real_2019_floor_never_lifts_value_added_above_output():
    """4.3.0: 16 floored nodes, 10 with VA > Y (e.g. ISL_B09 0.0464 -> 1.0 on Y = 0.2016)."""
    calib = package_mrio_to_calibration_result(condense_final_demand(read_native(OECD_2019, 2019)))
    nodes = calib.metadata["regularization"]["va_floored_nodes"]
    assert len(nodes) == 11
    assert all(n["value_added_after"] <= 0.5 * n["gross_output"] + 1e-12 or n["binding"] == "absolute"
               for n in nodes)
    assert not any(n["value_added_after"] > n["gross_output"] for n in nodes)
    isl = next(n for n in nodes if n["node"] == "ISL_B09")
    assert isl["value_added_after"] == pytest.approx(0.5 * isl["gross_output"])
    assert float(np.min(calib.tax)) > -1.0          # 4.3.0: -4.67


# ---------------------------------------------------------------------------
# 3. Every loader applies the foreign-balance closure to investment
# ---------------------------------------------------------------------------

EXPECTED_MAPPINGS = {
    "load_figaro": {"C": ["P3_S14", "P3_S15", "P3_S13"], "I": ["P51G"], "Cx": ["P5M"]},
    "load_exiobase": {"C": ["HFCE", "NPISH", "GGFC"], "I": ["GFCF"], "Cx": ["INVNT", "VALUABLES", "EXPORT"]},
    "load_wiod": {"C": ["CONS_h", "CONS_np", "CONS_g"], "I": ["GFCF"], "Cx": ["INVT"]},
    "load_eora": {"C": ["HFCE", "NPISH", "GGFC"], "I": ["GFCF"], "Cx": ["INVNT", "ACQ_VAL"]},
    "load_oecd_icio_granular": {"C": ["HFCE", "NPISH", "GGFC"], "I": ["GFCF", "INVNT"], "Cx": ["DPABR"]},
}


def _closure_gaps(calib) -> np.ndarray:
    """theta * Income minus purchaser spending per (country, category)."""
    nc, ns, nfd = calib.nc, calib.ns, calib.n_final_demand
    n = nc * ns
    d = calib.data_calibra
    purchases = d[:n, n:].reshape(nc, ns, nc, nfd).sum(axis=(0, 1)) + d[n, n:].reshape(nc, nfd)
    income = (calib.TT + calib.TTfd + calib.l_endow + calib.k_endow).ravel()
    return calib.theta[0].T * income[:, None] - purchases


@pytest.mark.parametrize("name", sorted(EXPECTED_MAPPINGS))
def test_loader_closure_lands_on_investment(name):
    """4.3.0: index 1 was P3_S15 / NPISH / CONS_np / NPISH / NPISH and consistent accounting raised."""
    calib = getattr(trade_data, name)(custom_c=3, custom_s=2, seed=1, fallback_to_synthetic=True)
    assert calib.metadata["fd_categories"] == ("C", "I", "Cx")
    assert calib.metadata["final_use_mapping"] == EXPECTED_MAPPINGS[name]
    assert calib.theta.shape == (1, 3, 3)
    gaps = _closure_gaps(calib)
    invfor = np.asarray(calib.invforT).ravel()
    scale = max(1.0, float(np.max(np.abs(invfor))))
    np.testing.assert_allclose(gaps[:, 1], invfor, atol=1e-10 * scale)
    np.testing.assert_allclose(np.delete(gaps, 1, axis=1), 0.0, atol=1e-10 * scale)
    assert np.all(calib.theta >= 0)
    _accounting._parameters(calib)   # accounting="consistent" accepts the calibration


def test_oecd_full_synthetic_fallback_uses_three_categories():
    calib = trade_data.load_oecd_icio_granular(year=2019, fallback_to_synthetic=True)
    assert calib.n_final_demand == 3
    assert calib.metadata["source_fd_categories"] == list(trade_data.RAW_6_FINAL_DEMAND_CODES)


def test_condensation_records_negative_cells_per_category():
    raw = generate_synthetic_mrio("oecd", custom_c=2, custom_s=2, seed=4)
    F = raw.F.reshape(raw.M, raw.C, raw.K_F).copy()
    F[0, 1, 4] = -(F[0, 1, 3] + 5.0)                   # INVNT larger than GFCF in one cell
    raw.final_demand_matrix[:] = F.reshape(raw.M, -1)
    condensed = condense_final_demand(raw)
    assert condensed.taxes_less_subsidies_fd is None
    assert condensed.metadata["negative_final_demand_cells"] == {"C": 0, "I": 1, "Cx": 0}


@pytest.mark.parametrize("name", sorted(EXPECTED_MAPPINGS))
def test_loader_records_what_cx_means(name):
    """Only the OECD Cx is residents' purchases abroad (DPABR); the (C, I, Cx) -> (C, G, X) bridges must check."""
    calib = getattr(trade_data, name)(custom_c=2, custom_s=2, seed=1, fallback_to_synthetic=True)
    semantics = calib.metadata["final_use_semantics"]
    assert set(semantics) == {"C", "I", "Cx"}
    assert "closure" in semantics["I"]
    if name == "load_oecd_icio_granular":
        assert "DPABR" in semantics["Cx"] and "NOT" not in semantics["Cx"]
    else:
        assert "inventories" in semantics["Cx"] and "NOT residents' direct purchases abroad" in semantics["Cx"]
    assert calib.metadata["closure_category"] == "I"


def test_package_warns_when_the_closure_would_land_on_a_non_investment_category():
    """A native six-code OECD roster has NPISH at index 1; 4.3.0 put the foreign balance there silently."""
    raw = generate_synthetic_mrio("oecd", custom_c=2, custom_s=2, seed=2)
    with pytest.warns(RuntimeWarning, match="index 1, which is 'NPISH'"):
        calib = package_mrio_to_calibration_result(raw)
    assert calib.metadata["closure_category"] == "NPISH"
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        condensed = package_mrio_to_calibration_result(condense_final_demand(raw))
    assert condensed.metadata["closure_category"] == "I"


# ---------------------------------------------------------------------------
# 4. FIGARO final uses are read by label
# ---------------------------------------------------------------------------

FIGARO_VALUES = {"P3_S13": 1.0, "P3_S14": 2.0, "P3_S15": 3.0, "P51G": 4.0, "P5M": 0.5}


def _write_figaro(path: Path, fd_order) -> Path:
    countries, sectors = ["AAA", "BBB"], ["S1", "S2"]
    nodes = [f"{c}_{s}" for c in countries for s in sectors]
    fd_cols = [f"{c}_{code}" for c in countries for code in fd_order]
    frame = pd.DataFrame(0.0, index=nodes + ["TLS", "VA"], columns=nodes + fd_cols)
    frame.loc[nodes, nodes] = 1.0
    for col in fd_cols:
        frame.loc[nodes, col] = FIGARO_VALUES[col.split("_", 1)[1]]
    sales = frame.loc[nodes].sum(axis=1)
    frame.loc["VA", nodes] = sales.values - frame.loc[nodes, nodes].sum(axis=0).values
    frame.to_csv(path)
    return path


def test_figaro_native_and_permuted_orders_give_identical_calibrations(tmp_path):
    """4.3.0 labelled columns by position: the native order swapped government and households."""
    native = _write_figaro(tmp_path / "native.csv", ("P3_S13", "P3_S14", "P3_S15", "P51G", "P5M"))
    permuted = _write_figaro(tmp_path / "permuted.csv", ("P5M", "P51G", "P3_S15", "P3_S13", "P3_S14"))
    a = trade_data.load_figaro(file_path=native, eur_to_usd=1.0, regularize=False)
    b = trade_data.load_figaro(file_path=permuted, eur_to_usd=1.0, regularize=False)
    np.testing.assert_allclose(a.theta, b.theta, rtol=0, atol=1e-14)
    np.testing.assert_allclose(a.afd, b.afd, rtol=0, atol=1e-14)
    assert a.metadata["source_fd_column_order"] == ["P3_S13", "P3_S14", "P3_S15", "P51G", "P5M"]
    # C = 1 + 2 + 3, I = 4, Cx = 0.5 per origin node and destination country.
    n = a.nc * a.ns
    per_category = a.data_calibra[:n, n:].reshape(n, a.nc, 3)
    np.testing.assert_allclose(per_category[:, :, 0], 6.0)
    np.testing.assert_allclose(per_category[:, :, 1], 4.0)
    np.testing.assert_allclose(per_category[:, :, 2], 0.5)


def test_figaro_unknown_or_missing_codes_raise(tmp_path):
    path = _write_figaro(tmp_path / "t.csv", FIGARO_FD_CATEGORIES)
    frame = pd.read_csv(path, index_col=0)
    frame.rename(columns={"AAA_P5M": "AAA_P5X"}).to_csv(tmp_path / "unknown.csv")
    with pytest.raises(ValueError, match="Unknown FIGARO final-use code"):
        trade_data.load_figaro(file_path=tmp_path / "unknown.csv")
    frame.rename(columns={"AAA_P5M": "AAA_P51G"}).to_csv(tmp_path / "duplicate.csv")
    with pytest.raises(ValueError, match="duplicate column labels"):
        trade_data.load_figaro(file_path=tmp_path / "duplicate.csv")


def test_figaro_compressed_file_reads_like_the_plain_file(tmp_path):
    """Round 1 sniffed the header as plain text: a .csv.gz raised UnicodeDecodeError (4.3.0 read it)."""
    plain = _write_figaro(tmp_path / "figaro_2019.csv", ("P3_S13", "P3_S14", "P3_S15", "P51G", "P5M"))
    packed = tmp_path / "figaro_2019.csv.gz"
    packed.write_bytes(gzip.compress(plain.read_bytes()))
    tsv = tmp_path / "figaro_2019.tsv.gz"
    tsv.write_bytes(gzip.compress(pd.read_csv(plain, index_col=0).to_csv(sep="\t").encode()))
    a = trade_data.load_figaro(file_path=plain, eur_to_usd=1.0, regularize=False)
    for other in (packed, tsv):
        b = trade_data.load_figaro(file_path=other, eur_to_usd=1.0, regularize=False)
        np.testing.assert_array_equal(a.data_calibra, b.data_calibra)
        np.testing.assert_array_equal(a.theta, b.theta)


# ---------------------------------------------------------------------------
# 5 and 6. Legacy 45-sector loader: floor keyword, record and MD5 integrity
# ---------------------------------------------------------------------------


def _write_raw45(path: Path, va0: float = 2.0) -> Path:
    """nc = 2, ns = 2, six final uses: every node sells 100, buys 40."""
    n_ind, nc = 4, 2
    table = np.zeros((n_ind + 2, n_ind + 6 * nc))
    table[:n_ind, :n_ind] = 10.0
    table[:n_ind, n_ind:] = 5.0
    va = np.array([va0, 40.0, 40.0, 40.0])
    table[n_ind + 1, :n_ind] = va
    table[n_ind, :n_ind] = 100.0 - 40.0 - va
    np.savetxt(path, table, delimiter=",")
    return path


def _register(monkeypatch, *, corrupted=None, clean=None):
    import puremacro.trade.mrio as mrio
    if corrupted is not None:
        monkeypatch.setitem(mrio.OECD_KNOWN_CORRUPTED_MD5, corrupted, "test export with lost decimal points")
    if clean is not None:
        monkeypatch.setitem(mrio.OECD_ICIO_MD5, 1999, clean)


def test_raw45_floor_keyword_default_and_record(tmp_path, monkeypatch):
    path = _write_raw45(tmp_path / "2020_SML.csv")
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    _register(monkeypatch, clean=md5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        default = load_raw_45sector_icio(path=path, nc=2, ns=2, return_structured=True)
    assert isinstance(default, ICIOData)
    # Default 0.05 (unchanged behaviour): VA 2 < 0.05 * 100 -> 5; TLS absorbs.
    assert default.labor_va[0] + default.capital_va[0] == pytest.approx(5.0)
    reg = default.metadata["regularization"]
    assert reg["floor_va_ratio"] == 0.05 and reg["va_floored_nodes"] == 1 and reg["zero_sales_nodes"] == 0
    assert reg["va_net_change"] == pytest.approx(3.0)
    assert default.metadata["md5"] == md5 and default.metadata["md5_status"] == "whitelisted"
    relaxed = load_raw_45sector_icio(path=path, nc=2, ns=2, return_structured=True, floor_va_ratio=1e-3)
    assert relaxed.labor_va[0] + relaxed.capital_va[0] == pytest.approx(2.0)
    assert relaxed.metadata["regularization"]["va_floored_nodes"] == 0
    assert "floor_va_ratio" in (load_raw_45sector_icio.__doc__ or "")
    assert "0.05" in load_raw_45sector_icio.__doc__ and "1e-4*y" not in load_raw_45sector_icio.__doc__


def test_raw45_refuses_registered_corrupted_checksum(tmp_path, monkeypatch):
    from puremacro.trade.mrio import MRIOIntegrityError
    path = _write_raw45(tmp_path / "data_2020_SML.csv")
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    _register(monkeypatch, corrupted=md5)
    with pytest.raises(MRIOIntegrityError) as info:
        load_raw_45sector_icio(path=path, nc=2, ns=2)
    assert isinstance(info.value, ValueError)
    message = str(info.value)
    assert str(path) in message and md5 in message
    assert "lost their decimal point" in message and "below 0.001 became 0" in message
    assert "28cba31491177955445051d459053744" in message and "d3e0f4979d85d6c0bb7cf4c43e324287" in message


def test_raw45_warns_on_unknown_checksum(tmp_path):
    path = _write_raw45(tmp_path / "2020_SML.csv", va0=3.0)
    with pytest.warns(RuntimeWarning, match="not a whitelisted OECD ICIO"):
        result = load_raw_45sector_icio(path=path, nc=2, ns=2, return_structured=True)
    assert result.metadata["md5_status"] == "unknown"


def test_raw45_default_search_prefers_clean_releases(tmp_path, monkeypatch):
    here = Path(trade_data.__file__).resolve()
    for root in (here.parents[3] / "IO", here.parents[2] / "IO"):
        if (root / "ICIOextended").exists() or (root / "computation").exists():
            pytest.skip("a real IO tree next to the checkout would shadow the temporary one")
    for var in ("IO_RAW45_PATH", "IO_DATA_PATH", "IO_COMPUTATION_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "computation" / "7_TIO_77c_vf" / "data_2020_SML.csv"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("0\n")
    assert trade_data._resolve_raw_45_path() == Path.cwd() / "computation" / "7_TIO_77c_vf" / "data_2020_SML.csv"
    clean_2019 = tmp_path / "ICIOextended" / "2019_SML.csv"
    clean_2019.parent.mkdir()
    clean_2019.write_text("0\n")
    assert trade_data._resolve_raw_45_path().name == "2019_SML.csv"
    (tmp_path / "ICIOextended" / "2020_SML.csv").write_text("0\n")
    assert trade_data._resolve_raw_45_path().name == "2020_SML.csv"


@pytest.mark.skipif(not CORRUPTED_2020.is_file(), reason="legacy corrupted export not available")
def test_real_corrupted_export_is_refused():
    with pytest.raises(ValueError, match="d1b887aaafa54ab3f28fde78fcd21cdf"):
        load_raw_45sector_icio(path=CORRUPTED_2020)


@pytest.mark.slow
@needs_oecd_2019
def test_real_2019_raw45_records_checksum_and_floor_counts():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        icio = load_raw_45sector_icio(path=OECD_2019, return_structured=True)
    assert icio.metadata["md5"] == "28cba31491177955445051d459053744"
    assert icio.metadata["whitelisted_year"] == 2019
    reg = icio.metadata["regularization"]
    assert (reg["va_changed_nodes"], reg["va_floored_nodes"], reg["zero_sales_nodes"]) == (63, 38, 25)
    assert reg["va_net_change"] == pytest.approx(3816.98, abs=0.01)


# ---------------------------------------------------------------------------
# 7. Validation, supplied output and balancing signals
# ---------------------------------------------------------------------------


def _regularized_wiod():
    raw = generate_synthetic_mrio("wiod", custom_c=3, custom_s=3, seed=3)
    return raw, regularize_mrio_table(raw.Z, raw.F, raw.VA, raw.TLS, raw.Y, n_countries=3, n_sectors=3)


def test_validate_enforces_documented_value_added_floor():
    """4.3.0 returned valid=True with VA = -5 on an active node."""
    _, (Z, F, VA, TLS, Y) = _regularized_wiod()
    assert validate_accounting_identities(Z, F, VA, TLS, Y)["valid"]
    for bad in (-5.0, 0.5):
        VA_bad, TLS_bad = VA.copy(), TLS.copy()
        TLS_bad[0] += VA[0] - bad
        VA_bad[0] = bad
        report = validate_accounting_identities(Z, F, VA_bad, TLS_bad, Y)
        assert report["max_outlays_error"] < 1e-9 * Y.max()
        assert report["valid"] is False
        assert report["value_added_floor_violations"] == 1


def test_supplied_output_inconsistent_with_row_sales_is_reported():
    """4.3.0 returned silently with the sales identity broken (max_sales_error 566)."""
    raw = generate_synthetic_mrio("wiod", custom_c=3, custom_s=3, seed=3)
    Y_wrong = raw.Y * 1.05
    with pytest.warns(RuntimeWarning, match="differs from row sales"):
        out = regularize_mrio_table(raw.Z, raw.F, raw.VA, raw.TLS, Y_wrong, n_countries=3, n_sectors=3)
    assert validate_accounting_identities(*out)["valid"] is False
    with pytest.raises(ValueError, match="differs from row sales"):
        regularize_mrio_table(raw.Z, raw.F, raw.VA, raw.TLS, Y_wrong, n_countries=3, n_sectors=3,
                              output_mismatch="raise")
    *fixed, report = regularize_mrio_table(raw.Z, raw.F, raw.VA, raw.TLS, Y_wrong, n_countries=3, n_sectors=3,
                                           output_mismatch="row_sales", return_report=True)
    assert report["output_mismatch"]["replaced_by_row_sales"] and len(report["output_mismatch"]["nodes"]) == 9
    assert validate_accounting_identities(*fixed)["valid"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        regularize_mrio_table(raw.Z, raw.F, raw.VA, raw.TLS, Y_wrong, n_countries=3, n_sectors=3,
                              output_mismatch="ignore")
        regularize_mrio_table(raw.Z, raw.F, raw.VA, raw.TLS, raw.Y, n_countries=3, n_sectors=3)


def test_ras_reports_convergence_and_infeasibility():
    """4.3.0 returned [[2, 0], [0, 1]] for u = (1, 1), v = (2, 0) with no signal."""
    Z0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    Z, info = balance_ras(Z0, [4.0, 6.0], [5.0, 5.0], return_info=True)
    assert info["converged"] and info["max_deviation"] <= 1e-10 and info["iterations"] >= 1
    with pytest.warns(RuntimeWarning, match="stopped after"):
        _, bad = balance_ras(np.eye(2), [1.0, 1.0], [2.0, 0.0], return_info=True)
    assert bad["converged"] is False and bad["max_deviation"] >= 0.5
    # A zero row target is met by zeroing the row (4.3.0 left row sum 0.0597).
    Zz, zero = balance_ras(Z0, [3.0, 0.0], [1.5, 1.5], return_info=True)
    assert zero["converged"] and np.allclose(Zz[1], 0.0)
    with pytest.warns(RuntimeWarning, match="rescaled"):
        _, scaled = balance_ras(Z0, [3.0, 4.0], [5.0, 5.0], return_info=True)
    assert scaled["targets_rescaled"] and scaled["column_target_scale"] == pytest.approx(0.7)
    with pytest.raises(ValueError, match="non-negative target"):
        balance_ras(Z0, [-1.0, 11.0], [5.0, 5.0])


def test_gras_reports_infeasible_sign_pattern():
    """4.3.0 returned margins [-2.27, 9.27] for targets [1, 6] without a signal."""
    Z0 = np.array([[-1.0, -2.0], [3.0, 4.0]])
    with pytest.warns(RuntimeWarning, match="stopped after"):
        _, info = balance_gras(Z0, [1.0, 6.0], [2.0, 5.0], max_iter=200, return_info=True)
    assert info["converged"] is False
    Zg, ok = balance_gras(np.array([[3.0, -1.0], [-2.0, 4.0]]), [2.0, 2.0], [1.0, 3.0], return_info=True)
    assert ok["converged"]
    np.testing.assert_allclose(Zg.sum(axis=1), [2.0, 2.0], atol=1e-10)


@pytest.mark.parametrize("n, scale", [(50, 1e4), (200, 1e5)])
def test_feasible_large_scale_balancing_is_converged_without_warning(n, scale):
    """Round 1 warned 'infeasible' at 7.5e-09 (6e-16 relative) on feasible 200 x 200 margins of 1.2e7."""
    rng = np.random.default_rng(0)
    Z0 = rng.uniform(0.1, 1, (n, n)) * scale
    target = rng.uniform(0.1, 1, (n, n)) * scale
    u, v = target.sum(axis=1), target.sum(axis=0)
    for balance in (balance_ras, balance_gras):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            Z, info = balance(Z0, u, v, return_info=True)
        assert info["converged"] is True
        if n == 200:
            assert info["max_deviation"] > info["tol"]        # the absolute 1e-10 is below resolution
        assert info["relative_deviation"] <= 1e-12
        assert info["effective_tol"] == pytest.approx(1e-12 * info["margin_scale"])
        np.testing.assert_allclose(Z.sum(axis=1), u, rtol=1e-12)
    # A genuinely infeasible target still warns at any scale.
    with pytest.warns(RuntimeWarning, match="stopped after"):
        _, bad = balance_ras(np.eye(2) * scale, [scale, scale], [2 * scale, 0.0], return_info=True)
    assert bad["converged"] is False


# ---------------------------------------------------------------------------
# 8. Native archives are refused with a pointer to puremacro.trade.mrio
# ---------------------------------------------------------------------------


def test_exiobase_native_zip_and_labelled_tables_are_refused(tmp_path):
    archive = tmp_path / "IOT_2019_ixi.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("IOT_2019_ixi/Z.txt", "region\tAT\n")
        z.writestr("IOT_2019_ixi/Y.txt", "region\tAT\n")
    with pytest.raises(ValueError, match="read_exiobase_native"):
        trade_data.load_exiobase(year=2019, file_path=archive)
    labelled = tmp_path / "Z.txt"
    labelled.write_text("region\tAT\tAT\nsector\tA\tB\nAT\t1\t2\n")
    with pytest.raises(ValueError, match="read_exiobase_native"):
        trade_data.load_exiobase(year=2019, file_path=labelled)


def test_ambiguous_data_directory_is_refused(tmp_path):
    (tmp_path / "exio_ixi_2019_a.txt").write_text("1\n")
    (tmp_path / "exio_ixi_2019_b.txt").write_text("1\n")
    with pytest.raises(ValueError, match="Ambiguous"):
        trade_data.load_exiobase(year=2019, data_dir=tmp_path)


def test_wiod_wrong_layout_names_expected_columns(tmp_path):
    path = tmp_path / "wiot_2014.csv"
    pd.DataFrame({"Year": [2014], "Value": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="WIOD 2016 wide-table columns"):
        trade_data.load_wiod(year=2014, file_path=path)


def test_exiobase_rejects_multi_member_zip_but_reads_a_single_member_one(tmp_path, monkeypatch):
    """Round 1 refused every ZIP; 4.3.0 read a single-member harmonized ZIP through pandas."""
    path = _write_exiobase(tmp_path / "exio_ixi_2019.txt", monkeypatch)
    single = tmp_path / "exio_ixi_2019_single.zip"
    with zipfile.ZipFile(single, "w") as z:
        z.write(path, arcname="exio_ixi_2019.txt")
    a = trade_data.load_exiobase(year=2019, file_path=path, eur_to_usd=1.0, regularize=False)
    b = trade_data.load_exiobase(year=2019, file_path=single, eur_to_usd=1.0, regularize=False)
    np.testing.assert_array_equal(a.data_calibra, b.data_calibra)
    multi = tmp_path / "exio_ixi_2019_multi.zip"
    with zipfile.ZipFile(multi, "w") as z:
        z.write(path, arcname="a.txt")
        z.write(path, arcname="b.txt")
    with pytest.raises(ValueError, match="read_exiobase_native"):
        trade_data.load_exiobase(year=2019, file_path=multi)


# ---------------------------------------------------------------------------
# Harmonized file branches of EXIOBASE, WIOD and Eora (positive paths)
# ---------------------------------------------------------------------------

EXIO_FD_VALUES = {"HFCE": 2.0, "NPISH": 0.5, "GGFC": 1.0, "GFCF": 1.5, "INVNT": 0.25, "VALUABLES": 0.05,
                  "EXPORT": 0.1}
EORA_FD_VALUES = {"HFCE": 2.0, "NPISH": 0.5, "GGFC": 1.0, "GFCF": 1.5, "INVNT": 0.25, "ACQ_VAL": 0.05}
WIOD_FD_VALUES = {"CONS_h": 2.0, "CONS_np": 0.5, "CONS_g": 1.0, "GFCF": 1.5, "INVT": 0.3}


def _positional_table(n_countries: int, n_sectors: int, fd_values: dict, n_va_rows: int) -> np.ndarray:
    """Headerless harmonized table: M node rows then value-added rows; M node columns then final uses."""
    m = n_countries * n_sectors
    k = len(fd_values)
    table = np.zeros((m + n_va_rows, m + n_countries * k))
    table[:m, :m] = 1.0
    table[:m, m:] = np.tile(list(fd_values.values()), n_countries)
    sales = table[:m].sum(axis=1)
    table[m, :m] = sales - table[:m, :m].sum(axis=0)      # all value added in the first row
    return table


def _write_exiobase(path: Path, monkeypatch, compress: bool = False) -> Path:
    monkeypatch.setattr(trade_data, "EXIOBASE_COUNTRIES", ("AAA", "BBB"))
    monkeypatch.setattr(trade_data, "EXIOBASE_163_SECTORS", ("i001", "i002"))
    text = "\n".join("\t".join(f"{x:g}" for x in row) for row in _positional_table(2, 2, EXIO_FD_VALUES, 7))
    if compress:
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text + "\n")
    return path


def _assert_condensed(calib, values: dict, mapping: dict, scale: float = 1.0):
    n = calib.nc * calib.ns
    per_category = calib.data_calibra[:n, n:].reshape(n, calib.nc, 3)
    for j, slot in enumerate(("C", "I", "Cx")):
        np.testing.assert_allclose(per_category[:, :, j], scale * sum(values[c] for c in mapping[slot]))
    assert calib.theta.shape == (1, 3, calib.nc)
    gaps = _closure_gaps(calib)
    invfor = np.asarray(calib.invforT).ravel()
    np.testing.assert_allclose(gaps[:, 1], invfor, atol=1e-10)
    np.testing.assert_allclose(np.delete(gaps, 1, axis=1), 0.0, atol=1e-10)
    _accounting._parameters(calib)


@pytest.mark.parametrize("compress", [False, True])
def test_exiobase_harmonized_file_is_condensed(tmp_path, monkeypatch, compress):
    """Round 1 sniffed the first line as plain text: a .txt.gz was refused as a 'native archive'."""
    path = _write_exiobase(tmp_path / ("exio_ixi_2019.txt" + (".gz" if compress else "")), monkeypatch, compress)
    calib = trade_data.load_exiobase(year=2019, file_path=path, eur_to_usd=1.0, regularize=False)
    mapping = EXPECTED_MAPPINGS["load_exiobase"]
    assert calib.metadata["final_use_mapping"] == mapping
    assert calib.metadata["source_fd_categories"] == list(trade_data.EXIOBASE_FD_CATEGORIES)
    _assert_condensed(calib, EXIO_FD_VALUES, mapping)


def test_eora_harmonized_file_is_condensed(tmp_path, monkeypatch):
    monkeypatch.setattr(trade_data, "EORA_189_COUNTRIES", ("AAA", "BBB"))
    monkeypatch.setattr(trade_data, "EORA_26_SECTORS", ("SEC01", "SEC02"))
    path = tmp_path / "Eora26_2015_bp.txt"
    np.savetxt(path, _positional_table(2, 2, EORA_FD_VALUES, 6) * 1000.0, delimiter="\t")
    calib = trade_data.load_eora(year=2015, file_path=path, regularize=False)
    mapping = EXPECTED_MAPPINGS["load_eora"]
    assert calib.metadata["final_use_mapping"] == mapping
    _assert_condensed(calib, EORA_FD_VALUES, mapping)       # thousand USD -> million USD
    with pytest.raises(ValueError, match="harmonized Eora26 layout needs at least"):
        np.savetxt(tmp_path / "short.txt", np.ones((3, 3)), delimiter="\t")
        trade_data.load_eora(year=2015, file_path=tmp_path / "short.txt")


def test_wiod_wide_file_is_condensed(tmp_path):
    countries, n_sectors = ["AAA", "BBB"], 56
    nodes = [(c, f"S{r:02d}") for c in countries for r in range(1, n_sectors + 1)]
    m = len(nodes)
    columns = {"Country": [c for c, _ in nodes] + ["TOT"] * 3,
               "IndustryCode": [code for _, code in nodes] + ["VA", "TXSP", "GO"]}
    Z = np.full((m, m), 0.01)
    fd = np.tile(list(WIOD_FD_VALUES.values()), (m, 1))
    Y = Z.sum(axis=1) + fd.sum(axis=1) * len(countries)
    VA = Y - Z.sum(axis=0)
    for ci, c in enumerate(countries):
        for r in range(1, n_sectors + 1):
            j = ci * n_sectors + r - 1
            columns[f"v{c}{r}"] = list(Z[:, j]) + [VA[j], 0.0, Y[j]]
        for r, value in zip(range(57, 62), WIOD_FD_VALUES.values()):
            columns[f"v{c}{r}"] = [value] * m + [0.0, 0.0, 0.0]
    path = tmp_path / "wiot_2014.csv"
    pd.DataFrame(columns).to_csv(path, index=False)
    calib = trade_data.load_wiod(year=2014, file_path=path, regularize=False)
    mapping = EXPECTED_MAPPINGS["load_wiod"]
    assert calib.metadata["final_use_mapping"] == mapping
    assert calib.nc == 2 and calib.ns == n_sectors
    _assert_condensed(calib, WIOD_FD_VALUES, mapping)
